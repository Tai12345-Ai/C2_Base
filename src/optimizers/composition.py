"""Blueprint (11)–(23) and Algorithm 2 for uniform actual identities."""

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt

import numpy as np
from scipy.linalg import helmert

from .trust_region import EPS, TOL, FloatArray, OracleNumericalError, maximize_quadratic


@dataclass(frozen=True)
class CompositionResult:
    q_signed: FloatArray
    q_feasible: FloatArray
    nominal_risk: float
    relaxed_value: float
    relaxed_upper: float
    feasible_lower: float
    denominator: float
    denominator_min: float
    kkt_residual: float
    branch: str
    secular_iterations: int

    @property
    def gradient_h(self) -> FloatArray:
        weights = np.outer(self.q_signed, self.q_signed) / self.denominator
        np.fill_diagonal(weights, 0.0)
        return weights

    @property
    def nonnegative_witness(self) -> bool:
        return bool(np.min(self.q_signed) >= 0)

    def diagnostics(self) -> dict[str, float | bool | str]:
        return {
            "nominal_risk": self.nominal_risk,
            "relaxed_value": self.relaxed_value,
            "relaxed_upper": self.relaxed_upper,
            "feasible_lower": self.feasible_lower,
            "relaxation_gap": max(0.0, self.relaxed_value - self.feasible_lower),
            "numerical_gap": self.relaxed_upper - self.relaxed_value,
            "bracket_width": self.relaxed_upper - self.feasible_lower,
            "denominator": self.denominator,
            "denominator_min": self.denominator_min,
            "min_q": float(np.min(self.q_signed)),
            "nonnegative_witness": self.nonnegative_witness,
            "nonnegative_coordinate_fraction": float(np.mean(self.q_signed >= 0)),
            "kkt_residual": self.kkt_residual,
            "branch": self.branch,
            "secular_iterations": self.secular_iterations,
        }


@lru_cache(maxsize=3)
def _centered_basis(size: int) -> FloatArray:
    return np.ascontiguousarray(helmert(size, full=False).T)


def validate_radius(size: int, radius: float) -> float:
    if size < 2 or not np.isfinite(radius) or not 0 <= radius < sqrt(size):
        raise ValueError("Expected B >= 2 and 0 <= rho^2 < B - 1")
    numerator = size - 1 - radius * radius
    if not numerator > 0:
        raise ValueError("Expected B >= 2 and 0 <= rho^2 < B - 1")
    return numerator / size


def _ratio(h: FloatArray, q: FloatArray) -> tuple[float, float]:
    denominator = 1.0 - float(q @ q)
    if not denominator > 0:
        raise OracleNumericalError("Nonpositive direct different-identity denominator")
    return float(q @ (h @ q)) / denominator, denominator


def solve_composition(h: FloatArray, radius: float) -> CompositionResult:
    """Return the signed optimizer and a feasible bracket for the simplex problem."""
    if np.iscomplexobj(h):
        raise ValueError("H must be real")
    h = np.asarray(h, dtype=np.float64)
    if (
        h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2
        or not np.isfinite(h).all()
        or not np.array_equal(h, h.T)
        or np.any(np.diag(h))
    ):
        raise ValueError("H must be finite, square (B >= 2), symmetric and zero-diagonal")

    size = len(h)
    d_min = validate_radius(size, radius)
    q0 = np.full(size, 1 / size)
    d0 = (size - 1) / size
    offdiag = h[~np.eye(size, dtype=bool)]
    reference = float(offdiag[0])
    constant = bool(np.all(offdiag == reference))
    mean_deviation = 0.0 if constant else float((offdiag - reference).mean())
    nominal = reference + mean_deviation
    if constant or radius == 0:
        return CompositionResult(
            q0, q0.copy(), nominal, nominal, nominal, nominal, d0, d_min, 0.0,
            "constant" if constant else "nominal", 0,
        )

    # Keep the base level and mean deviation separate so rounding r0 cannot
    # inject spurious curvature when the off-diagonals differ by only one ULP.
    e = (h - reference) - mean_deviation
    np.fill_diagonal(e, 0.0)
    basis = _centered_basis(size)
    curvature = basis.T @ (e @ basis) / size
    curvature = (curvature + curvature.T) * 0.5
    linear = basis.T @ (e @ q0) / sqrt(size)
    trs = maximize_quadratic(curvature, linear, radius)
    u = sqrt(size) * (basis @ trs.point)
    u -= u.mean()

    # Keep the direct witness inside the ball in case the basis transform rounds out.
    u_norm = float(np.linalg.norm(u))
    u_max = radius * sqrt(size)
    if u_norm > u_max:
        u *= np.nextafter(u_max / u_norm, 0.0)
    q = (1.0 + u) / size
    direct, denominator = _ratio(h, q)
    reduced = nominal + trs.value / d_min
    scale = max(float(np.max(np.abs(offdiag))), abs(reduced), abs(direct))
    tolerance = TOL * scale
    if (
        not np.isfinite([direct, reduced]).all()
        or abs(denominator - d_min) > TOL * d_min
        or abs(direct - reduced) > tolerance
    ):
        raise OracleNumericalError(
            f"Single-TRS ratio mismatch (direct={direct:.10g}, reduced={reduced:.10g}, "
            f"D={denominator:.3g}, D_min={d_min:.3g})"
        )

    shrink = min(1.0, 1.0 / max(1.0, -float(np.min(u))))
    if shrink < 1.0:
        shrink = np.nextafter(shrink, 0.0)
    feasible = (1.0 + shrink * u) / size
    lower, _ = _ratio(h, feasible)
    if lower < nominal:
        feasible, lower = q0.copy(), nominal

    dual_upper = nominal + trs.dual_upper / d_min
    if (
        not np.isfinite([lower, dual_upper]).all()
        or np.min(feasible) < 0
        or abs(float(feasible.sum()) - 1.0) > TOL
        or np.linalg.norm(sqrt(size) * (feasible - q0)) > radius + TOL
        or dual_upper < max(direct, lower) - tolerance
        or dual_upper - direct > tolerance
    ):
        raise OracleNumericalError(
            f"Composition bracket validation failed (KKT={trs.kkt_residual:.3g}, "
            f"lower={lower:.10g}, direct={direct:.10g}, dual={dual_upper:.10g})"
        )

    upper = max(dual_upper, direct, lower) + 32 * EPS * scale
    return CompositionResult(
        q, feasible, nominal, direct, upper, lower, denominator, d_min,
        trs.kkt_residual, trs.branch, trs.iterations,
    )
