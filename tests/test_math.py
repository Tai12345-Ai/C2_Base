import math

import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.optimize import brentq, minimize
import torch

from src.objectives.losses import composition_loss, nominal_pair_risk, row_pair_risk
from src.objectives.pairwise import pairwise_kernel
from src.optimizers.composition import solve_composition
from src.optimizers.trust_region import maximize_quadratic


def symmetric(rng, size):
    h = rng.uniform(0, 1, size=(size, size))
    h = (h + h.T) * 0.5
    np.fill_diagonal(h, 0)
    return h


def risk(h, q):
    mask = ~np.eye(len(q), dtype=bool)
    products = np.outer(q, q)[mask]
    return products @ h[mask] / products.sum()


def fractional_reference(h, radius):
    """Theorem 6 root via a SciPy secular solve on an independent null basis."""
    size = len(h)
    q0 = np.full(size, 1 / size)
    basis = null_space(np.ones((1, size))) / math.sqrt(size)
    eigenvalues, rotation = np.linalg.eigh(basis.T @ h @ basis)
    d = rotation.T @ basis.T @ h @ q0
    c = q0 @ h @ q0
    d0 = 1 - 1 / size

    def residual(t):
        shifted = eigenvalues + t / size
        floor = max(0.0, shifted[-1])
        gaps = floor - shifted
        if floor == 0 and min(gaps) > 0 and np.linalg.norm(d / gaps) <= radius:
            w = d / gaps
        else:
            # Random instances have nonzero d in the top eigenspace.
            def norm_error(offset):
                return np.linalg.norm(d / (gaps + offset)) - radius

            upper = 2 * np.linalg.norm(d) / radius
            offset = brentq(norm_error, 1e-16, upper, xtol=1e-15)
            w = d / (gaps + offset)
        return c - t * d0 + 2 * d @ w + shifted @ (w**2)

    bound = (1 + radius**2) / (1 - (1 + radius**2) / size)
    return brentq(residual, -bound, bound, xtol=1e-13)


def test_pairwise_matches_equations_and_different_identity_conditioning():
    z = torch.tensor([[[1.0, 0], [0, 1]], [[1, 1], [-1, 0]], [[0, -1], [-1, -1]]],
                     dtype=torch.float64, requires_grad=True)
    h = pairwise_kernel(z, 0.23, 0.05)
    unit = torch.nn.functional.normalize(z, dim=-1)
    sigmoid = lambda t: torch.sigmoid(t / 0.05)
    for i in range(len(z)):
        for j in range(len(z)):
            if i == j:
                assert h[i, j] == 0
                continue
            negative = sum(sigmoid(unit[i, a] @ unit[j, b] - 0.23)
                           for a in range(2) for b in range(2)) / 4
            genuine = sum(sigmoid(0.23 - unit[k, 0] @ unit[k, 1]) for k in [i, j])
            torch.testing.assert_close(h[i, j], 0.5 * negative + 0.25 * genuine)
    assert torch.equal(h.T, h)
    assert h.dtype == torch.float64
    torch.testing.assert_close(torch.autograd.grad(h.diag().sum(), z)[0], torch.zeros_like(z))


@pytest.mark.parametrize("size,radius", [(2, 0.8), (5, 0.0), (64, 0.1), (5, 1.75)])
def test_constant_kernel(size, radius):
    h = 0.37 * (np.ones((size, size)) - np.eye(size))
    r = solve_composition(h, radius)
    assert r.branch == "constant"
    assert r.relaxed_value == r.feasible_lower == r.relaxed_upper == 0.37
    q = np.full(size, 1 / size)
    shift = radius / math.sqrt(2 * size)
    q[:2] += [shift, -shift]
    assert risk(h, q) == pytest.approx(0.37)
    assert q @ h @ q == pytest.approx(0.37 * (1 - q @ q))
    np.testing.assert_allclose(r.gradient_h.sum(), 1.0)
    assert np.all(np.diag(r.gradient_h) == 0)


def test_nominal_first_order_and_trace_correction():
    h = symmetric(np.random.default_rng(23), 7)
    q0 = np.full(7, 1 / 7)
    basis = null_space(np.ones((1, 7)))
    a = basis.T @ h @ basis / 7
    b = basis.T @ h @ q0 / math.sqrt(7)
    r0 = risk(h, q0)
    c = a + r0 * np.eye(6) / 7
    assert np.trace(c) == pytest.approx(0.0, abs=1e-15)
    assert np.linalg.eigvalsh(c)[0] < 0 < np.linalg.eigvalsh(c)[-1]
    for multiplier in [-1, 1]:
        v = multiplier * np.arange(6) / 25
        q = q0 + basis @ v / math.sqrt(7)
        assert risk(h, q) - r0 == pytest.approx((2 * b @ v + v @ c @ v) / (1 - q @ q))
    r = solve_composition(h, 0.0)
    assert r.branch == "nominal"
    assert r.relaxed_value == pytest.approx(r0)
    tensor = torch.from_numpy(h).requires_grad_()
    assert nominal_pair_risk(tensor).item() == pytest.approx(r0)
    assert row_pair_risk(tensor, 0.1).item() == pytest.approx(r0 + 0.2 * np.linalg.norm(b) / (6/7))


def test_population_expansion_and_one_sided_control():
    rng = np.random.default_rng(20261111)
    for _ in range(40):
        h = symmetric(rng, 9) + np.diag(rng.random(9))
        p = rng.dirichlet(np.ones(9))
        q = rng.dirichlet(np.ones(9))
        u = q / p - 1
        mu = p @ h @ p
        g = h @ p - mu
        centered = h - mu - g[:, None] - g[None, :]
        shift = p * u
        assert q @ h @ q == pytest.approx(mu + 2*g @ shift + shift @ centered @ shift)
        assert q @ h @ p == pytest.approx(mu + g @ shift)


def test_hard_cosine_graphs_and_signed_curvature():
    clique = np.ones((4, 4)) - np.eye(4)
    zero = np.zeros_like(clique)
    a1 = np.block([[clique, zero], [zero, clique]])
    a2 = np.block([[zero, clique], [clique, zero]])
    balanced = [np.array([1, 1, 1, 1, -1, -1, -1, -1]),
                np.array([1, 1, -1, -1, -1, -1, 1, 1])]
    for graph, sign, weight in zip([a1, a2], balanced, [3, 1], strict=True):
        gram = np.eye(8) + graph / 4
        cholesky = np.linalg.cholesky(gram)
        np.testing.assert_array_equal(cholesky @ cholesky.T >= 1 / 8, graph + np.eye(8))
        for radius in [0.1, 0.8, 1.0]:
            r = solve_composition(graph, radius)
            expected = (3 + weight * radius**2) / (7 - radius**2)
            assert r.branch == "hard"
            assert r.relaxed_value == pytest.approx(expected, abs=1e-13)
            assert r.kkt_residual < 1e-12
            q = (1 + radius * sign) / 8
            assert min(q) >= 0
            assert risk(graph, q) == pytest.approx(expected)
            assert r.relaxed_upper + 1e-13 >= expected >= r.feasible_lower - 1e-13
    assert maximize_quadratic(np.diag([2.0, -1.0, -1.0]), np.zeros(3), 1).value == 2
    assert maximize_quadratic(np.diag([-2.0, 1.0, 1.0]), np.zeros(3), 1).value == 1


@pytest.mark.parametrize("pole_component, expected_branch", [(0.0, "hard"), (1e-11, "secular")])
def test_hard_and_near_hard_trs(pole_component, expected_branch):
    matrix = np.diag([2.0, 2.0, -1.0])
    b = np.array([pole_component, 0.0, 0.3])
    r = maximize_quadratic(matrix, b, 0.5)
    assert r.branch == expected_branch
    assert np.linalg.norm(r.point) == pytest.approx(0.5)
    assert r.point[-1] == pytest.approx(0.1, abs=1e-10)
    assert r.value == pytest.approx(0.53, abs=2e-11)
    assert r.dual_upper >= r.value
    assert r.kkt_residual < 1e-12


def test_negative_definite_trs_interior_optimum():
    a = -np.diag([2.0, 3.0])
    r = maximize_quadratic(a, np.array([0.3, -0.2]), 0.7)
    assert r.branch == "interior"
    np.testing.assert_allclose(r.point, [0.15, -0.2/3])
    assert r.dual_upper == pytest.approx(r.value, abs=1e-12)


def test_single_eigendecomposition(monkeypatch):
    h = symmetric(np.random.default_rng(32), 8)
    original = np.linalg.eigh
    count = 0

    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    solve_composition(h, 0)
    assert count == 0
    solve_composition(h, 0.15)
    assert count == 1


def test_fractional_reference_and_feasible_bracket():
    rng = np.random.default_rng(20261111)
    worst_reference_error = worst_kkt = worst_positivity_gap = 0.0
    for index in range(200):
        size = 3 + index % 22
        h = symmetric(rng, size)
        exact_radius = 1 / math.sqrt(size - 1)
        radius = [0.1, exact_radius, 0.25, 1.0][index % 4]
        r = solve_composition(h, radius)
        reference = fractional_reference(h, radius)
        worst_reference_error = max(worst_reference_error, abs(reference - r.relaxed_value))
        worst_kkt = max(worst_kkt, r.kkt_residual)
        worst_positivity_gap = max(worst_positivity_gap, r.diagnostics()["relaxation_gap"])
        assert r.relaxed_value == pytest.approx(reference, abs=5e-12)
        assert r.relaxed_upper + 1e-13 >= reference
        assert r.denominator == pytest.approx(r.denominator_min, abs=1e-13)
        assert np.sum(r.q_signed) == pytest.approx(1.0, abs=1e-14)
        q0 = np.full(size, 1 / size)
        assert min(r.q_feasible) >= 0
        assert r.q_feasible.sum() == pytest.approx(1.0, abs=1e-14)
        assert size * np.sum((r.q_feasible - q0)**2) <= radius**2 + 1e-13
        assert r.feasible_lower == pytest.approx(risk(h, r.q_feasible))
        assert r.relaxed_upper - r.relaxed_value < 5e-12
        if radius <= exact_radius:
            assert min(r.q_signed) >= -1e-14
            assert r.feasible_lower == pytest.approx(r.relaxed_value, abs=1e-13)
        for _ in range(16):
            random_q = rng.dirichlet(np.ones(size))
            alpha = min(1.0, radius / (math.sqrt(size) * np.linalg.norm(random_q - q0)))
            candidate = q0 + alpha * (random_q - q0)
            assert risk(h, candidate) <= r.relaxed_upper + 1e-13
    print(f"fractional cross-check n=200, max_abs={worst_reference_error:.3e}, "
          f"max_kkt={worst_kkt:.3e}, max_positivity_gap={worst_positivity_gap:.3e}")


def test_negative_weights_are_retained_and_slsqp_stays_in_bracket():
    size, radius = 7, 1.3
    h = np.ones((size, size)) - np.eye(size)
    h[0, :] = h[:, 0] = 0
    r = solve_composition(h, radius)
    assert r.q_signed.min() < 0
    assert r.gradient_h.min() < 0
    assert r.relaxed_upper > r.feasible_lower
    q0 = np.full(size, 1 / size)
    constraints = [
        {"type": "eq", "fun": lambda q: q.sum() - 1},
        {"type": "ineq", "fun": lambda q: radius**2 - size * np.sum((q - q0)**2)},
    ]
    for initial in (q0, r.q_feasible):
        independent = minimize(lambda q: -risk(h, q), initial, method="SLSQP",
                               bounds=[(0.0, 0.85)] * size, constraints=constraints,
                               options={"ftol": 1e-12})
        assert independent.success, independent.message
        assert r.feasible_lower - 1e-10 <= -independent.fun <= r.relaxed_upper + 1e-10


def test_envelope_gradient_and_signed_h():
    rng = np.random.default_rng(421)
    h = symmetric(rng, 6)
    tensor = torch.tensor(h, requires_grad=True)
    loss, r = composition_loss(tensor, 0.6)
    assert loss.item() == pytest.approx(r.relaxed_value)
    loss.backward()
    np.testing.assert_allclose(tensor.grad.numpy(), r.gradient_h, atol=1e-15)
    direction = symmetric(rng, 6) - h
    step = 1e-6
    fd = (solve_composition(h + step * direction, 0.6).relaxed_value
          - solve_composition(h - step * direction, 0.6).relaxed_value) / (2 * step)
    assert fd == pytest.approx(float(np.sum(r.gradient_h * direction)), abs=2e-9)


def test_active_subgradient_at_tie():
    h = np.ones((5, 5)) - np.eye(5)
    r = solve_composition(h, 0.4)
    direction = symmetric(np.random.default_rng(3), 5)
    h_new = h + 0.001 * direction
    assert solve_composition(h_new, 0.4).relaxed_value >= (
        r.relaxed_value + np.sum(r.gradient_h * (h_new - h)) - 1e-13
    )


@pytest.mark.parametrize("rho", [1e-12, 0.1, 0.9999 * math.sqrt(6)])
def test_near_constant_and_ill_conditioned_but_resolvable(rho):
    h = -0.35 * (np.ones((7, 7)) - np.eye(7))
    h += 1e-13 * symmetric(np.random.default_rng(7), 7)
    r = solve_composition(h, rho)
    assert np.isfinite(r.relaxed_value)
    assert r.relaxed_upper >= r.feasible_lower


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_one_ulp_offdiagonal_change_keeps_the_nonconstant_trs_on_boundary(direction):
    size = 64
    base = 0.4341445796581056
    h = np.full((size, size), base)
    np.fill_diagonal(h, 0.0)
    h[0, 1] = h[1, 0] = np.nextafter(base, direction)
    result = solve_composition(h, 0.1)
    assert result.branch in ("hard", "secular")
    assert result.nonnegative_witness
    assert result.denominator == pytest.approx(result.denominator_min, abs=1e-14)


@pytest.mark.parametrize("rho", [-0.1, np.inf, np.nan, 2.0])
def test_invalid_radius_fails_closed(rho):
    with pytest.raises(ValueError, match="rho"):
        solve_composition(np.ones((5, 5)) - np.eye(5), rho)


@pytest.mark.parametrize("h", [
    [[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, np.nan], [np.nan, 0]], [[0, np.inf], [np.inf, 0]],
    [[0, 1j], [1j, 0]],
])
def test_invalid_h_fails_closed(h):
    with pytest.raises(ValueError, match="H"):
        solve_composition(np.asarray(h), 0.1)
