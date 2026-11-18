"""Frozen ordinary shifts, finite-graph stress and paired identity summaries."""

import numpy as np
from scipy.special import expit

from src.optimizers.composition import solve_composition
from src.optimizers.trust_region import OracleNumericalError
from .statistics import finite_family_audit, paired_identity_bootstrap
from .verification import composition_metrics, ordinary_profiles


def fit_operating_points(scores_by_arm: dict, levels: list[float]) -> dict:
    return {arm: scores.fit_thresholds(levels) for arm, scores in scores_by_arm.items()}


def _hard_stress(errors, profiles, identities, radii, reps, seed, *, use_bootstrap):
    n = len(identities)
    uniform = np.full(n, 1 / n)
    ordinary = {}
    for name, weights in profiles.items():
        q = weights / weights.sum()
        ordinary[name] = {
            "radius": float(np.sqrt(n * np.sum((q - uniform)**2))),
            "arms": {arm: composition_metrics(h, g, q) for arm, (h, g) in errors.items()},
        }
        if use_bootstrap:
            ordinary[name]["identity_bootstrap"] = paired_identity_bootstrap(errors, weights, reps, seed)

    controls = {}
    for group in range(4):
        probe = profiles[f"group_{group}_up"]
        reference = profiles[f"group_{(group + 1) % 4}_up"]
        probe = probe / probe.sum()
        reference = reference / reference.sum()
        controls[f"group_{group}_up"] = {
            "fixed_gallery": {arm: composition_metrics(h, g, probe, uniform)
                              for arm, (h, g) in errors.items()},
            "independent_roles": {arm: composition_metrics(h, g, probe, reference)
                                  for arm, (h, g) in errors.items()},
        }

    per_identity = {
        arm: {"fmr": (h.sum(axis=1) / (n - 1)).tolist(), "fnmr": g.tolist(),
              "fmr_quantiles": np.quantile(h.sum(axis=1) / (n - 1), [0, .25, .5, .75, 1]).tolist()}
        for arm, (h, g) in errors.items()
    }
    frontier = {}
    for radius in radii:
        brackets = {}
        transfer = {}
        for arm, (h, _) in errors.items():
            try:
                witness = solve_composition(h, radius)
            except OracleNumericalError as exc:
                brackets[arm] = {"status": "FAIL", "reason": str(exc)}
                continue
            brackets[arm] = {"status": "PASS", **witness.diagnostics()}
            transfer[arm] = {
                other: composition_metrics(other_h, other_g, witness.q_feasible)
                for other, (other_h, other_g) in errors.items()
            }
        frontier[str(radius)] = {"brackets": brackets, "cross_adversary_transfer": transfer}
    return {"ordinary": ordinary, "role_shift_controls": controls, "frontier": frontier,
            "per_identity": per_identity}


def evaluate_stress(scores_by_arm: dict, identities: list[str], plan: dict, *, test: bool) -> dict:
    profiles = ordinary_profiles(identities, plan["protocol_seed"])
    metrics = {}
    for level in plan["levels"]:
        key = str(level)
        errors = {
            arm: scores.errors_at(plan["operating_points"][arm][key]["threshold"])
            for arm, scores in scores_by_arm.items()
        }
        metrics[key] = _hard_stress(
            errors, profiles, identities, plan["radii"], plan["bootstrap_reps"],
            plan["bootstrap_seed"], use_bootstrap=test,
        )
        metrics[key]["operating_points"] = {arm: plan["operating_points"][arm][key] for arm in errors}

    surrogate = {}
    tau = plan["tau_train"]
    for arm, scores in scores_by_arm.items():
        temperature = plan["temperature"]
        negative = expit((scores.impostor - tau) / temperature).mean(axis=(2, 3), dtype=np.float64)
        positive = expit((tau - scores.genuine) / temperature).astype(np.float64)
        h = 0.5 * negative + 0.25 * (positive[:, None] + positive[None, :])
        h = (h + h.T) / 2.0
        np.fill_diagonal(h, 0)
        surrogate[arm] = {}
        for radius in plan["radii"]:
            try:
                result = solve_composition(h, radius)
            except OracleNumericalError as exc:
                surrogate[arm][str(radius)] = {"status": "FAIL", "reason": str(exc)}
            else:
                surrogate[arm][str(radius)] = {"status": "PASS", **result.diagnostics()}
    return {
        "identities": identities, "hard": metrics, "surrogate_at_training_threshold": surrogate,
        "bootstrap_scope": "paired identity weights, conditional on locked thresholds and observed captures",
        "bootstrap_method": "exploratory percentile intervals with no rare-event coverage guarantee",
    }


def evaluate_calibration(scores_by_arm: dict, identities: list[str], plan: dict) -> dict:
    profiles = ordinary_profiles(identities, plan["protocol_seed"])
    family_size = len(profiles) * len(scores_by_arm) * len(plan["levels"])
    results = {}
    for level in plan["levels"]:
        key = str(level)
        for arm, scores in scores_by_arm.items():
            threshold = plan["operating_points"][arm][key]["threshold"]
            hard, _ = scores.errors_at(threshold)
            results[f"{arm}@{key}"] = {
                name: finite_family_audit(hard, weights, 2.0, family_size, plan["audit_delta"])
                for name, weights in profiles.items()
            }
    return {"ids": len(identities), "family_size": family_size, "delta": plan["audit_delta"],
            "frozen_family": results}
