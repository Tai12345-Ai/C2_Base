from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin
import torch

from src.data.captures import CaptureDataset, IdentityBatchSampler
from src.data.manifest import (
    DatasetAuditError, SplitPlan, _validate_records, audit_digiface, eval_records,
    load_manifest, train_records,
)


FIXTURE_PLAN = SplitPlan(train=4, dev=4, cal=2, test=3, captures=3,
                         pilot_train=3, pilot_captures=2, threshold_dev=2)


def make_crops(root: Path, plan: SplitPlan = FIXTURE_PLAN):
    rng = np.random.default_rng(234)
    n = plan.train + plan.dev + plan.cal + plan.test
    for identity in range(n):
        parent = root / f"p{identity // 5}" / f"id{identity:03d}"
        parent.mkdir(parents=True)
        center = rng.integers(20, 235, size=(32, 32, 3))
        for capture in range(plan.captures):
            pixels = (center + rng.integers(-20, 21, size=center.shape)).astype(np.uint8)
            Image.fromarray(pixels).save(parent / f"{capture}.png")


def test_manifest_counts_identity_disjointness_and_hash_reproducibility(tmp_path):
    root = tmp_path / "crops"
    make_crops(root)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    archives = {"p0": {"archive": "part1.zip", "sha256": "b" * 64}}
    r = audit_digiface(root, first, plan=FIXTURE_PLAN, archives=archives)
    audit_digiface(root, second, plan=FIXTURE_PLAN, archives=archives)
    assert first.read_bytes() == second.read_bytes()
    assert r["split_ids"] == {"train": 4, "dev": 4, "cal": 2, "test": 3}
    assert r["pilot_ids"] == 3 and r["pilot_images"] == 6
    records = load_manifest(first)
    ids_by_split = [{x.identity_id for x in records if x.split == split}
                    for split in ("train", "dev", "cal", "test")]
    assert len(set.union(*ids_by_split)) == sum(map(len, ids_by_split))
    assert len(train_records(records, "pilot")) == 6
    assert len(train_records(records, "full")) == 12
    threshold_ids, chosen = eval_records(records, "dev", 4, "threshold")
    selection_ids, _ = eval_records(records, "dev", 4, "selection")
    assert len(chosen) == 4 and not set(threshold_ids) & set(selection_ids)
    assert all(x.archive_sha256 == "b" * 64 for x in records if x.relative_path.startswith("p0/"))

    dataset = CaptureDataset(root, chosen)
    x, label = dataset[0]
    flipped, same_label = dataset[0, True]
    assert x.shape == (3, 112, 112) and x.dtype == torch.float32
    assert x.min() >= -1 and x.max() <= 1
    assert label == same_label and torch.equal(flipped, x.flip(-1))
    (root / chosen[0].relative_path).write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="bytes changed"):
        dataset[0]
    with pytest.raises(ValueError, match="leakage"):
        leaked = replace(records[0], split="test", relative_path="other/other.png", sha256="2" * 64,
                         capture_id="other", pilot=0)
        _validate_records([*records, leaked])


def test_audit_reports_pixel_duplicates_across_identity_and_writes_no_manifest(tmp_path):
    root = tmp_path / "crops"
    make_crops(root)
    source = root / "p0/id000/0.png"
    duplicate = root / "p0/id001/1.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("example", "Different encoded bytes")
    with Image.open(source) as im:
        im.save(duplicate, pnginfo=metadata)
    assert source.read_bytes() != duplicate.read_bytes()
    output = tmp_path / "bad.csv"
    with pytest.raises(DatasetAuditError, match="duplicate") as fail:
        audit_digiface(root, output, plan=FIXTURE_PLAN)
    assert fail.value.report["duplicate_counts"] == {"cross_identity": 1}
    assert not output.exists()


def test_sampler_preserves_two_distinct_captures_and_mid_epoch_cursor():
    labels = np.repeat(np.arange(6), 3).tolist()
    full = list(IdentityBatchSampler(labels, 4, 19, 0, 8))
    suffix = list(IdentityBatchSampler(labels, 4, 19, 3, 8))
    assert full[3:] == suffix
    assert full != list(IdentityBatchSampler(labels, 4, 20, 0, 8))
    for batch in full:
        assert len(set(index for index, _ in batch)) == 8
        sampled = [labels[index] for index, _ in batch]
        assert sampled[0::2] == sampled[1::2] and len(set(sampled)) == 4
