"""Identity-level empirical intervals and the blueprint's finite-family audit."""

from math import log, sqrt

import numpy as np


def paired_identity_bootstrap(
    errors: dict[str, tuple[np.ndarray, np.ndarray]], raw_weights: np.ndarray,
    reps: int, seed: int, *, baseline: str = "nominal", confidence: float = 0.95,
) -> dict:
    """Percentile intervals conditional on locked thresholds and observed captures."""
    raw_weights = np.asarray(raw_weights, dtype=np.float64)
    if raw_weights.ndim != 1:
        raise ValueError("Expected one raw weight per identity")
    n = len(raw_weights)
    if n < 2 or reps < 1 or not 0 < confidence < 1 or baseline not in errors:
        raise ValueError("Invalid bootstrap design")
    if not np.isfinite(raw_weights).all() or np.min(raw_weights) < 0 or np.count_nonzero(raw_weights) < 2:
        raise ValueError("Bootstrap requires nonnegative weights with support on at least two identities")
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(n, np.full(n, 1 / n), size=reps)
    q = multiplicity * raw_weights[None, :]
    mass = q.sum(axis=1, keepdims=True)
    q = np.divide(q, mass, out=np.zeros_like(q), where=mass > 0)
    # The squared mass of each original ID removes all copy-copy collisions.
    denominator = 1 - np.sum(q**2, axis=1)
    valid = (denominator > 0) & (mass[:, 0] > 0)
    q = q[valid]
    tail = (1 - confidence) / 2
    if not len(q):
        return {"valid_replicates": 0, "replicates": reps, "intervals": {}, "paired_differences": {}}

    draws = {}
    for arm, (impostor, genuine) in errors.items():
        numerator = np.einsum("ri,ri->r", q @ impostor, q)
        fnmr = q @ genuine
        draws[arm] = {"fmr": numerator / denominator[valid], "fnmr": fnmr, "tar": 1 - fnmr}

    def interval(values):
        return np.quantile(values, [tail, 1 - tail]).tolist()

    contrasts = {
        arm: {metric: interval(values - draws[baseline][metric]) for metric, values in metrics.items()}
        for arm, metrics in draws.items() if arm != baseline
    }
    if "row" in draws and "c2" in draws:
        contrasts["c2_minus_row"] = {
            metric: interval(values - draws["row"][metric]) for metric, values in draws["c2"].items()
        }
    return {
        "valid_replicates": len(q), "replicates": reps,
        "confidence": confidence,
        "intervals": {arm: {metric: interval(values) for metric, values in metrics.items()}
                      for arm, metrics in draws.items()},
        "paired_differences": contrasts,
        "baseline": baseline,
    }


def finite_family_audit(
    h: np.ndarray, raw_weights: np.ndarray, bound: float, family_size: int, delta: float
) -> dict:
    """Equations (29)–(31) for a frozen kernel and bounded measurable weights."""
    n = len(raw_weights)
    if (n < 2 or h.shape != (n, n) or family_size < 1 or not 0 < delta < 1
        or not np.isfinite([bound, delta]).all() or bound <= 0
        or not np.isfinite(raw_weights).all() or np.any(raw_weights < 0)
        or np.any(raw_weights > bound) or not np.isfinite(h).all()
        or np.any(h < 0) or np.any(h > 1) or not np.array_equal(h, h.T) or np.any(h.diagonal())):
        raise ValueError("Invalid finite-family audit input")
    ordered_pairs = n * (n - 1)
    numerator = float(raw_weights @ h @ raw_weights) / ordered_pairs
    denominator = float(raw_weights.sum()**2 - raw_weights @ raw_weights) / ordered_pairs
    epsilon = bound**2 * sqrt(log(4 * family_size / delta) / (2 * (n // 2)))
    upper = min(1.0, (numerator + epsilon) / (denominator - epsilon)) if denominator > epsilon else None
    return {"numerator": numerator, "denominator": denominator, "epsilon": epsilon, "upper": upper}
