"""Summarize measured training profiles or matched held-out results across seeds."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from src.utils.io import write_json

ARMS = ("nominal", "row", "c2")


def profile(run_root: Path, limit: int) -> dict:
    if limit < 1:
        raise ValueError("Profile needs at least one update")
    output = {}
    for arm in ARMS:
        steps = {}
        epochs = {}
        with (run_root / arm / "training.jsonl").open(encoding="utf-8") as log:
            for line in log:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # A process interrupted while writing its final line may leave a partial line.
                    continue
                if event["event"] == "start":
                    steps = {i: row for i, row in steps.items() if i <= event["from_attempts"]}
                    epochs = {i: row for i, row in epochs.items() if row["updates"] <= event["from_updates"]}
                elif event["event"] == "step":
                    steps[event["attempt"]] = event
                elif event["event"] == "epoch":
                    epochs[event["epoch"]] = event
        sampled = [row for row in steps.values() if row["status"] == "updated"][:limit]
        if not sampled:
            raise ValueError(f"No completed training steps for {arm}")
        measured = [row for row in steps.values() if row["attempt"] <= sampled[-1]["attempt"]]
        elapsed = sum(row["step_seconds"] for row in measured)
        images = sum(row["images_per_second"] * row["step_seconds"] for row in sampled)
        vram = [row["peak_vram_bytes"] for row in measured if row["peak_vram_bytes"] is not None]
        oracle = [row["oracle"] for row in sampled if row["oracle"] is not None]
        output[arm] = {
            "requested_updates": limit, "profiled_updates": len(sampled),
            "profiled_skips": len(measured) - len(sampled),
            "total_skips": sum(row["status"] == "skipped" for row in steps.values()),
            "images_per_second_excluding_loader": images / elapsed,
            "peak_vram_allocated_bytes": max(vram) if vram else None,
            "pair_path_fraction_including_oracle": sum(
                row["pair_seconds_including_oracle"] for row in sampled
            ) / elapsed,
            "epochs": list(epochs.values()),
            "oracle": None if not oracle else {
                "nonnegative_witness_fraction": mean(row["nonnegative_witness"] for row in oracle),
                "minimum_q": min(row["min_q"] for row in oracle),
                "maximum_kkt_residual": max(row["kkt_residual"] for row in oracle),
                "maximum_relaxation_gap": max(row["relaxation_gap"] for row in oracle),
                "maximum_numerical_gap": max(row["numerical_gap"] for row in oracle),
                "min_denominator": min(row["denominator_min"] for row in oracle),
            },
        }
    return output


def summarize_seeds(reports: list[dict]) -> dict:
    seeds = [report["seed"] for report in reports]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Summarize requires at least two distinct seeds")
    reference = reports[0]
    scope_keys = ("scope", "identities", "manifest_sha256")
    if any(any(report[k] != reference[k] for k in scope_keys) for report in reports):
        raise ValueError("Reports cover different manifest, training scope or test identities")
    expected = {level: set(row["frontier"]) for level, row in reference["hard"].items()}
    if any({level: set(row["frontier"]) for level, row in report["hard"].items()} != expected
           for report in reports):
        raise ValueError("Reports use different operating levels or stress radii")

    def summary(values):
        return {"mean": mean(values), "std": stdev(values), "per_seed": dict(zip(seeds, values, strict=True))}

    hard = {}
    for level, details in reference["hard"].items():
        level_per_seed = [report["hard"][level] for report in reports]
        ordinary = {}
        for shift in details["ordinary"]:
            ordinary[shift] = {}
            for metric in ("fmr", "fnmr", "tar"):
                across_arms = {
                    arm: [row["ordinary"][shift]["arms"][arm][metric] for row in level_per_seed]
                    for arm in ARMS
                }
                ordinary[shift][metric] = {
                    "arms": {arm: summary(values) for arm, values in across_arms.items()},
                    "paired_seed_differences_vs_nominal": {
                        arm: summary([a - b for a, b in zip(across_arms[arm], across_arms["nominal"], strict=True)])
                        for arm in ("row", "c2")
                    },
                    "c2_minus_row": summary([
                        a - b for a, b in zip(across_arms["c2"], across_arms["row"], strict=True)
                    ]),
                }
        brackets = {}
        for radius in details["frontier"]:
            brackets[radius] = {}
            for arm in ARMS:
                rows = [row["frontier"][radius]["brackets"][arm] for row in level_per_seed]
                failures = [seed for seed, row in zip(seeds, rows, strict=True) if row["status"] != "PASS"]
                brackets[radius][arm] = {"failed_seeds": failures}
                if not failures:
                    brackets[radius][arm] |= {
                        key: summary([row[key] for row in rows])
                        for key in ("feasible_lower", "relaxed_upper", "relaxation_gap")
                    }
        hard[level] = {"ordinary": ordinary, "brackets": brackets}
    return {
        "seeds": seeds, "scope": reference["scope"], "test_ids": len(reference["identities"]),
        "manifest_sha256": reference["manifest_sha256"], "hard": hard,
        "seeds_with_mismatched_sampling": [
            row["seed"] for row in reports if not row["identical_sample_schedule"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    profile_args = subparsers.add_parser("profile")
    profile_args.add_argument("--run-root", required=True, type=Path)
    profile_args.add_argument("--limit", default=500, type=int)
    profile_args.add_argument("--output", required=True, type=Path)
    seed_args = subparsers.add_parser("seeds")
    seed_args.add_argument("--reports", required=True, nargs="+", type=Path)
    seed_args.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "profile":
        result = profile(args.run_root, args.limit)
    else:
        result = summarize_seeds([json.loads(path.read_text(encoding="utf-8")) for path in args.reports])
    write_json(args.output, result)
    print(json.dumps({"report": str(args.output)}))


if __name__ == "__main__":
    main()
