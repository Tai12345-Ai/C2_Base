"""Exercise the normal CLI paths on temporary synthetic crops with the real IR-18."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

from scripts import evaluate, train
from scripts.report_runs import profile, summarize_seeds
from src.data.manifest import audit_digiface
from src.training.checkpoint import read_checkpoint
from tests.test_data import FIXTURE_PLAN, make_crops


def test_full_ir18_cpu_warmup_matched_training_and_locked_evaluation(tmp_path, monkeypatch):
    with TemporaryDirectory(dir=tmp_path) as folder:
        run_root = Path(folder)
        crops = run_root / "crops"
        make_crops(crops)
        manifest = run_root / "captures.csv"
        audit_digiface(crops, manifest, plan=FIXTURE_PLAN)
        config = run_root / "config.yaml"
        config.write_text(
            "identities_per_batch: 3\nwarmup_epochs: 1\npilot_epochs: 1\n"
            "workers: 0\neval_batch_size: 4\ncheckpoint_every: 1\n",
            encoding="utf-8",
        )
        common = ["--manifest", str(manifest), "--data-root", str(crops), "--device", "cpu"]
        for arm in ("warmup", "nominal", "row", "c2"):
            options = ["--config", str(config), "--out", str(run_root / arm)]
            if arm != "warmup":
                options.extend(["--warmup-dir", str(run_root / "warmup")])
            monkeypatch.setattr(sys, "argv", ["train", arm, *common, *options])
            train.main()
            path = run_root / arm / "last.pt"
            saved = read_checkpoint(path)
            assert saved["cursor"]["updates"] == saved["cursor"]["attempts"] == 1
            stem = saved["model"]["encoder.stem.0.weight"].clone()
            del saved
            if arm == "warmup":
                warmup_stem = stem
                assert (run_root / arm / "tau_train.json").exists()
            else:
                assert not torch.equal(stem, warmup_stem)

        steps = [json.loads(line) for line in (run_root / "c2/training.jsonl").read_text().splitlines()]
        step = next(event for event in steps if event["event"] == "step")
        assert step["oracle"]["branch"] in ("secular", "hard")
        assert step["oracle"]["nonnegative_witness"] and step["oracle"]["kkt_residual"] < 1e-9
        output = run_root / "evaluation"
        for action in ("lock", "cal", "test"):
            options = ["--run-root", str(run_root), "--output", str(output)]
            if action == "lock":
                options += ["--bootstrap-reps", "30"]
            monkeypatch.setattr(sys, "argv", ["evaluate", action, *common, *options])
            evaluate.main()
        report = json.loads((output / "test.json").read_text(encoding="utf-8"))
        assert len(report["identities"]) == FIXTURE_PLAN.test and report["identical_sample_schedule"]
        op = report["hard"]["0.01"]
        assert op["frontier"]["0.1"]["brackets"]["c2"]["status"] == "PASS"
        assert set(op["ordinary"]["nominal"]["identity_bootstrap"]["paired_differences"]) == {
            "row", "c2", "c2_minus_row",
        }
        cal = json.loads((output / "cal.json").read_text(encoding="utf-8"))
        assert cal["ids"] == FIXTURE_PLAN.cal and cal["family_size"] == 30
        profiling = profile(run_root, limit=500)
        assert profiling["c2"]["profiled_updates"] == 1
        assert profiling["c2"]["oracle"]["nonnegative_witness_fraction"] == 1.0
        # Exercise seed aggregation with a duplicated synthetic fixture and a different seed key.
        stats = summarize_seeds([report, report | {"seed": 37}])
        assert stats["hard"]["0.01"]["ordinary"]["nominal"]["fmr"]["c2_minus_row"]["std"] == 0
