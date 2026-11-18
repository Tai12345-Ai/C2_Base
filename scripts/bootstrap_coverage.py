"""Simulate coverage under rare-hub and degenerate identity kernels."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.statistics import paired_identity_bootstrap
from src.utils.io import write_json


def coverage(ids: int, trials: int, reps: int, seed: int, hub_rate: float) -> dict:
    if ids < 3 or trials < 1 or reps < 2 or not 0 < hub_rate < 1:
        raise ValueError("Check identity count, trial/replicate counts and rare-hub mass")
    rng = np.random.default_rng(seed)
    results = {}
    for scenario, probability in (("rare_hub", hub_rate), ("equal_rows", 0.5)):
        truth = 2 * probability - probability**2 if scenario == "rare_hub" else 0.5
        covered = empty = missed_hubs = 0
        for _ in range(trials):
            types = rng.random(ids) < probability
            if scenario == "rare_hub":
                h = np.logical_or.outer(types, types)
                missed_hubs += int(not types.any())
            else:
                h = np.equal.outer(types, types)
            h = h.astype(float)
            np.fill_diagonal(h, 0)
            r = paired_identity_bootstrap(
                {"nominal": (h, np.zeros(ids))}, np.ones(ids), reps,
                seed=int(rng.integers(0, 2**32)),
            )
            if r["valid_replicates"] == 0:
                empty += 1
                continue
            lower, upper = r["intervals"]["nominal"]["fmr"]
            covered += int(lower - 1e-15 <= truth <= upper + 1e-15)
        results[scenario] = {
            "true_population_risk": truth, "covered": covered,
            "proportion_covered": covered / (trials - empty) if trials > empty else None,
            "trials_without_valid_replicates": empty, "no_hub_observed": missed_hubs,
        }
    return {"ids": ids, "trials": trials, "replicates": reps, "seed": seed,
            "nominal_confidence": 0.95, "hub_rate": hub_rate, "scenarios": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=int, default=64)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--hub-rate", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = coverage(args.ids, args.trials, args.reps, args.seed, args.hub_rate)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
