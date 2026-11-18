"""Fit operating thresholds on dev, then evaluate the locked pilot protocol."""

import argparse
import json
import math
from pathlib import Path

from src.data.manifest import load_manifest
from src.evaluation.data import score_split
from src.evaluation.stress import evaluate_calibration, evaluate_stress, fit_operating_points
from src.models.face import IR18
from src.optimizers.composition import validate_radius
from src.training.checkpoint import read_checkpoint
from src.training.config import TrainConfig
from src.training.engine import configure_runtime
from src.utils.io import sha256_file, write_json
from scripts.train import ARMS, device_from_arg


def _signature(run_root: Path, manifest_hash: str) -> tuple[dict, dict, dict]:
    hashes = {}
    skipped = {}
    common = None
    for arm in ARMS:
        path = run_root / arm / "last.pt"
        state = read_checkpoint(path)
        signature = state["signature"]
        if (
            signature["arm"] != arm or signature["manifest_sha256"] != manifest_hash
            or state["cursor"]["updates"] != signature["target_updates"]
            or not signature["initialization_sha256"]
        ):
            raise ValueError(f"Run {arm} is incomplete or belongs to a different experiment")
        shared = {key: value for key, value in signature.items() if key != "arm"}
        if common is not None and shared != common:
            raise ValueError("Arms differ in initialization, identity map, thresholds or matched training config")
        common = shared
        hashes[arm] = sha256_file(path)
        skipped[arm] = state["cursor"]["attempts"] - state["cursor"]["updates"]
    return common, hashes, skipped


def _score_arms(run_root, records, data_root, plan, device, split, batch_size, workers):
    ids = None
    scores_by_arm = {}
    fallbacks = {}
    for arm in ARMS:
        path = run_root / arm / "last.pt"
        if sha256_file(path) != plan["checkpoint_sha256"][arm]:
            raise ValueError(f"Checkpoint changed after lock: {arm}")
        state = read_checkpoint(path)
        encoder = IR18().to(device)
        prefix = "encoder."
        encoder.load_state_dict({name.removeprefix(prefix): tensor
                                 for name, tensor in state["model"].items() if name.startswith(prefix)})
        del state
        got_ids, scores, count = score_split(
            encoder, records, data_root, split, plan["protocol_seed"], device, batch_size, workers,
            dev_role="selection" if split == "dev" else "",
        )
        if ids is not None and got_ids != ids:
            raise RuntimeError("Identity order changed between arms")
        ids = got_ids
        scores_by_arm[arm], fallbacks[arm] = scores, count
    return ids, scores_by_arm, fallbacks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("lock", "test", "cal"))
    parser.add_argument("--run-root", type=Path, required=True, help="Parent of nominal/, row/ and c2/")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for locked plan and reports")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--radii", type=float, nargs="+")
    parser.add_argument("--levels", type=float, nargs="+")
    parser.add_argument("--bootstrap-reps", type=int)
    args = parser.parse_args()
    lock = args.output / "locked_protocol.json"
    if args.action != "lock" and any(value is not None for value in (args.radii, args.levels, args.bootstrap_reps)):
        parser.error("Test and cal use the thresholds and options from the locked protocol")

    manifest_hash = sha256_file(args.manifest)
    records = load_manifest(args.manifest)
    device = device_from_arg(args.device)
    if args.action == "lock":
        if lock.exists():
            raise FileExistsError(f"Protocol already locked at {lock}. Use that lock or a new output directory.")
        signature, hashes, skipped = _signature(args.run_root, manifest_hash)
        config = TrainConfig(**signature["config"])
        configure_runtime(config)
        levels = args.levels if args.levels is not None else [0.01, 0.001]
        radii = args.radii if args.radii is not None else [0.10, 0.25, 0.50]
        reps = args.bootstrap_reps if args.bootstrap_reps is not None else 500
        if (
            reps < 2 or len(set(radii)) != len(radii) or len(set(levels)) != len(levels)
            or any(not math.isfinite(t) or not 0 < t < 1 for t in levels)
        ):
            parser.error("Use unique valid levels/radii and >= 2 bootstrap replicates")
        dev_size = len({row.identity_id for row in records if row.dev_role == "selection"})
        for radius in radii:
            validate_radius(dev_size, radius)
        plan = {
            "format": 1, "manifest_sha256": manifest_hash, "checkpoint_sha256": hashes,
            "initialization_sha256": signature["initialization_sha256"], "seed": config.seed,
            "scope": signature["scope"], "protocol_seed": config.protocol_seed,
            "tau_train": signature["tau_train"], "temperature": config.temperature,
            "levels": levels, "radii": radii, "bootstrap_reps": reps,
            "bootstrap_seed": config.protocol_seed + 1, "audit_delta": 0.05,
            "device_type": device.type,
            "workers": config.workers, "eval_batch_size": config.eval_batch_size,
            "skipped_updates": skipped, "identical_sample_schedule": not any(skipped.values()),
            "captures_per_id": 2, "threshold_split": "dev/selection",
        }
        ids, scores, fallbacks = _score_arms(
            args.run_root, records, args.data_root, plan, device, "dev", config.eval_batch_size, config.workers,
        )
        plan["operating_points"] = fit_operating_points(scores, levels)
        dev = evaluate_stress(scores, ids, plan, test=False)
        write_json(args.output / "dev.json", dev | {"norm_fallbacks": fallbacks})
        write_json(lock, plan)
        print(json.dumps({"locked": str(lock), "dev_ids": len(ids), "skipped_updates": skipped}))
        return

    plan = json.loads(lock.read_text(encoding="utf-8"))
    if (plan.get("format") != 1 or plan["manifest_sha256"] != manifest_hash
        or plan["device_type"] != device.type):
        raise ValueError("Locked evaluation requires the original manifest and eval device type")
    ids, scores, fallbacks = _score_arms(
        args.run_root, records, args.data_root, plan, device, args.action,
        plan["eval_batch_size"], plan["workers"],
    )
    if args.action == "cal":
        report = evaluate_calibration(scores, ids, plan)
    else:
        report = evaluate_stress(scores, ids, plan, test=True)
    output = args.output / f"{args.action}.json"
    write_json(output, report | {"plan_sha256": sha256_file(lock),
                                "manifest_sha256": manifest_hash, "seed": plan["seed"],
                                "scope": plan["scope"], "norm_fallbacks": fallbacks,
                                "identical_sample_schedule": plan["identical_sample_schedule"]})
    print(json.dumps({"report": str(output), "split": args.action, "ids": len(ids)}))


if __name__ == "__main__":
    main()
