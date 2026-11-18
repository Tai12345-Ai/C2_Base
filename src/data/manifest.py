"""Audit already acquired crops and construct a hash-locked identity protocol."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from io import BytesIO
from pathlib import Path, PurePosixPath
import csv
import hashlib
import re

from PIL import Image, ImageOps

from src.utils.io import atomic_output

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SPLITS = ("train", "dev", "cal", "test")


@dataclass(frozen=True)
class SplitPlan:
    train: int = 6000
    dev: int = 1000
    cal: int = 1000
    test: int = 2000
    captures: int = 72
    pilot_train: int = 2000
    pilot_captures: int = 20
    threshold_dev: int = 200


@dataclass(frozen=True)
class Capture:
    relative_path: str
    identity_id: str
    capture_id: str
    sha256: str
    source_archive: str
    archive_sha256: str
    preprocessing: str
    split: str
    dev_role: str
    pilot: int


class DatasetAuditError(ValueError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = dict(report, status="FAIL", error=message)


def hash_key(seed: int, namespace: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{key}".encode("utf-8")).hexdigest()


def _validate_plan(plan: SplitPlan):
    if (
        min(getattr(plan, split) for split in SPLITS) < 2
        or not 2 <= plan.pilot_train <= plan.train
        or not 2 <= plan.pilot_captures <= plan.captures
        or not 2 <= plan.threshold_dev <= plan.dev - 2
    ):
        raise ValueError("Invalid split or capture counts")


def _inventory(root: Path, plan: SplitPlan, report: dict) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    owner_dirs: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        if path.parent == root or not path.resolve().is_relative_to(root):
            raise DatasetAuditError(f"Expected a safe identity directory for {path.name}", report)
        identity = path.parent.name
        if owner_dirs.setdefault(identity, path.parent) != path.parent:
            raise DatasetAuditError(f"Identity directory appears more than once: {identity}", report)
        groups[identity].append(path)
    report["found_ids"] = len(groups)
    report["found_images"] = sum(map(len, groups.values()))
    report["captures_per_id_histogram"] = dict(sorted(Counter(map(len, groups.values())).items()))
    if len(groups) != sum(getattr(plan, split) for split in SPLITS):
        raise DatasetAuditError("Unexpected identity count. Supply the 72-capture partition.", report)
    if any(len(paths) != plan.captures for paths in groups.values()):
        raise DatasetAuditError("Unexpected capture count. Check the extracted parts.", report)
    if any(len({p.stem for p in paths}) != len(paths) for paths in groups.values()):
        raise DatasetAuditError("Repeated capture ID within an identity", report)
    return groups


def audit_digiface(
    data_root: Path, output: Path, seed: int = 20261111, plan: SplitPlan = SplitPlan(),
    archives: dict | None = None,
) -> dict:
    """Audit bytes and capture IDs before deriving split assignments."""
    _validate_plan(plan)
    root = data_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("The dataset root must be a directory")
    report = {
        "status": "PASS", "split_seed": seed, "plan": asdict(plan),
        "source": "https://github.com/microsoft/DigiFace1M",
        "license": "https://github.com/microsoft/DigiFace1M/blob/main/LICENSE",
    }
    groups = _inventory(root, plan, report)
    seen_pixels: dict[str, str] = {}
    seen_bytes: dict[str, str] = {}
    duplicate_examples = []
    duplicates = Counter()
    dimensions = Counter()
    hashes = {}
    for identity, paths in groups.items():
        for path in paths:
            relative = path.relative_to(root).as_posix()
            try:
                raw = path.read_bytes()
                with Image.open(BytesIO(raw)) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    dimensions[f"{rgb.width}x{rgb.height}"] += 1
                    pixel_hash = hashlib.sha256(
                        f"{rgb.width}x{rgb.height}\0".encode("ascii") + rgb.tobytes()
                    ).hexdigest()
            except (OSError, ValueError) as exc:
                raise DatasetAuditError(f"Capture decode failed: {relative}", report) from exc
            content_hash = hashlib.sha256(raw).hexdigest()
            hashes[path] = content_hash
            byte_owner = seen_bytes.setdefault(content_hash, relative)
            pixel_owner = seen_pixels.setdefault(pixel_hash, relative)
            previous = byte_owner if byte_owner != relative else pixel_owner
            if previous != relative:
                duplicates["cross_identity" if PurePosixPath(previous).parent.name != identity
                           else "within_identity"] += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append([previous, relative])

    report.update(dimensions=dict(dimensions), duplicate_counts=dict(duplicates),
                  duplicate_examples=duplicate_examples)
    if duplicates:
        raise DatasetAuditError("Exact duplicate captures found. Resolve provenance before splitting.", report)

    ordered = sorted(groups, key=lambda identity: hash_key(seed, "outer-split", identity))
    split_ids = {}
    start = 0
    for split in SPLITS:
        end = start + getattr(plan, split)
        split_ids[split] = ordered[start:end]
        start = end
    pilot_ids = set(sorted(split_ids["train"], key=lambda i: hash_key(seed, "pilot-ids", i))
                    [:plan.pilot_train])
    threshold_ids = set(sorted(split_ids["dev"], key=lambda i: hash_key(seed, "threshold-dev", i))
                        [:plan.threshold_dev])
    records = []
    archive_sources = dict(archives or {})
    for split, identities in split_ids.items():
        for identity in identities:
            paths = groups[identity]
            pilot_paths = set()
            if identity in pilot_ids:
                pilot_paths = set(sorted(paths, key=lambda p: hash_key(
                    seed, "pilot-captures", f"{identity}/{p.stem}"
                ))[:plan.pilot_captures])
            role = ""
            if split == "dev":
                role = "threshold" if identity in threshold_ids else "selection"
            for path in paths:
                relative = path.relative_to(root)
                part = relative.parts[0] if len(relative.parts) > 2 else "."
                archive = archive_sources.get(part, {})
                archive_hash = archive.get("sha256", "")
                if archive_hash and not re.fullmatch(r"[a-fA-F0-9]{64}", archive_hash):
                    raise ValueError(f"Invalid source archive SHA-256 for {part}")
                records.append(Capture(
                    relative.as_posix(), identity, path.stem, hashes[path],
                    archive.get("archive", ""), archive_hash.lower(),
                    "released_crop_validated", split, role, int(path in pilot_paths),
                ))

    _validate_records(records)
    with atomic_output(output) as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Capture)])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    report.update(
        split_ids={split: len(ids) for split, ids in split_ids.items()},
        split_images=dict(Counter(record.split for record in records)),
        pilot_ids=len(pilot_ids), pilot_images=sum(record.pilot for record in records),
        threshold_dev_ids=len(threshold_ids),
        archive_provenance_supplied=sum(bool(record.archive_sha256) for record in records),
        manifest=output.name,
    )
    return report


def _validate_records(records: list[Capture]) -> None:
    assignments = {}
    captures = set()
    paths = set()
    digests = set()
    if not records:
        raise ValueError("Manifest is empty")
    for record in records:
        path = PurePosixPath(record.relative_path)
        if (
            path.is_absolute() or ".." in path.parts or len(path.parts) < 2
            or "\\" in record.relative_path or ":" in record.relative_path
            or not record.identity_id or not record.capture_id
            or record.split not in SPLITS or not re.fullmatch("[a-f0-9]{64}", record.sha256)
            or record.pilot not in (0, 1) or (record.pilot and record.split != "train")
            or (record.split == "dev" and record.dev_role not in ("threshold", "selection"))
            or (record.split != "dev" and record.dev_role)
        ):
            raise ValueError(f"Invalid manifest row: {record.relative_path}")
        assignment = (record.split, record.dev_role)
        if assignments.setdefault(record.identity_id, assignment) != assignment:
            raise ValueError(f"Identity leakage: {record.identity_id}")
        key = (record.identity_id, record.capture_id)
        if key in captures or path in paths or record.sha256 in digests:
            raise ValueError(f"Duplicate capture, path or content: {record.relative_path}")
        captures.add(key)
        paths.add(path)
        digests.add(record.sha256)


def load_manifest(path: Path) -> list[Capture]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [f.name for f in fields(Capture)]:
            raise ValueError("Unsupported manifest columns")
        records = [Capture(**dict(row, pilot=int(row["pilot"]))) for row in reader]
    _validate_records(records)
    return records


def train_records(records: list[Capture], scope: str) -> list[Capture]:
    if scope not in ("pilot", "full"):
        raise ValueError("Expected pilot or full training scope")
    chosen = [row for row in records if row.split == "train" and (scope == "full" or row.pilot)]
    if not chosen:
        raise ValueError(f"No training captures in scope {scope}")
    return sorted(chosen, key=lambda row: (row.identity_id, row.capture_id))


def eval_records(
    records: list[Capture], split: str, seed: int, dev_role: str = ""
) -> tuple[list[str], list[Capture]]:
    groups = defaultdict(list)
    for row in records:
        if row.split == split and (not dev_role or row.dev_role == dev_role):
            groups[row.identity_id].append(row)
    identities = sorted(groups)
    if len(identities) < 2 or min(map(len, groups.values())) < 2:
        raise ValueError("Evaluation requires at least two IDs and two captures per ID")
    selected = [
        row for identity in identities
        for row in sorted(groups[identity], key=lambda r: hash_key(
            seed, "eval-captures", f"{identity}/{r.capture_id}"
        ))[:2]
    ]
    return identities, selected
