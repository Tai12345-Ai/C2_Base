import numpy as np
import pytest

from src.evaluation.statistics import finite_family_audit, paired_identity_bootstrap
from src.evaluation.verification import VerificationScores, composition_metrics, ordinary_profiles


def test_threshold_ties_and_actual_impostor_control():
    scores = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]], [[1, 0], [-1, 0]]], np.float32)
    verified = VerificationScores.from_embeddings(scores)
    fit = verified.fit_thresholds([0.00 + 1/12, 0.4])
    q = np.ones(3) / 3
    for target, details in fit.items():
        impostor, genuine = verified.errors_at(details["threshold"])
        assert details["impostor_capture_comparisons"] == 12
        assert np.diag(impostor).sum() == 0
        actual = composition_metrics(impostor, genuine, q)
        assert actual["fmr"] <= float(target)
        assert actual["fmr"] == pytest.approx(details["achieved_fmr"])
    h, g = verified.errors_at(0.0)
    assert g[1] == 0  # decision accepts at equality
    assert h[0, 1] == 1


def test_paired_bootstrap_removes_all_same_identity_copies():
    n = 4
    constant_h = np.ones((n, n)) - np.eye(n)
    genuine = np.zeros(n)
    r = paired_identity_bootstrap(
        {"nominal": (constant_h, genuine), "c2": (constant_h * 0.75, genuine)},
        np.array([1, 2, 1, 2]), 1000, 21,
    )
    assert 0 < r["valid_replicates"] < r["replicates"]
    assert r["intervals"]["nominal"]["fmr"] == pytest.approx([1, 1])
    assert r["paired_differences"]["c2"]["fmr"] == pytest.approx([-0.25, -0.25])


def test_one_sided_and_common_laws_match_explicit_distinct_trial_enumeration():
    h = np.array([[0., 0.3, 0.7], [0.3, 0, 1], [0.7, 1, 0]])
    genuine = np.array([0.0, 0.5, 1.0])
    q, p = np.array([0.2, 0.4, 0.4]), np.array([0.7, 0.2, 0.1])
    for gallery in (p, q):
        result = composition_metrics(h, genuine, q, gallery)
        pairs = [(q[i] * gallery[j], h[i, j], 1 - genuine[i])
                 for i in range(len(q)) for j in range(len(q)) if i != j]
        denom = sum(row[0] for row in pairs)
        assert result["fmr"] == pytest.approx(sum(w * error for w, error, _ in pairs) / denom)
        assert result["endpoint_tar"] == pytest.approx(sum(w * tar for w, _, tar in pairs) / denom)


def test_finite_family_bound_and_predefined_stress_weights():
    n = 64
    constant = 0.15 * (np.ones((n, n)) - np.eye(n))
    profiles = ordinary_profiles([str(x) for x in range(n)], 5)
    for weights in profiles.values():
        r = finite_family_audit(constant, weights, 2.0, len(profiles), 0.05)
        assert r["numerator"] / r["denominator"] == pytest.approx(0.15)
        assert r["upper"] is None or r["upper"] >= 0.15
