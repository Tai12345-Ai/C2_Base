"""Global dense maximization of a quadratic over a Euclidean ball."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
EPS = np.finfo(np.float64).eps
TOL = 2e-10


class OracleNumericalError(ArithmeticError):
    """The computed witness did not meet the numerical contract."""


@dataclass(frozen=True)
class TrustRegionResult:
    point: FloatArray
    value: float
    dual_upper: float
    kkt_residual: float
    branch: str
    iterations: int


def _secular(d: FloatArray, gaps: FloatArray) -> tuple[FloatArray, float, int]:
    """Solve ||d / (gaps + delta)|| = 1 on the positive side of the pole."""
    low = 0.0
    high = float(np.linalg.norm(d))
    if not high > 0:
        raise OracleNumericalError("Degenerate secular bracket")
    for iteration in range(1, 257):
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            norm = np.linalg.norm(d / (gaps + middle))
        if norm > 1.0:
            low = middle
        else:
            high = middle
        if high - low <= 8 * EPS * high:
            break
    return d / (gaps + high), high, iteration


def maximize_quadratic(
    matrix: FloatArray, linear: FloatArray, radius: float
) -> TrustRegionResult:
    """Maximize v.T @ matrix @ v + 2 * linear.T @ v with ||v|| <= radius."""
    if np.iscomplexobj(matrix) or np.iscomplexobj(linear):
        raise ValueError("TRS coefficients must be real")
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    if (
        linear.ndim != 1
        or not linear.size
        or matrix.shape != (linear.size, linear.size)
        or not np.array_equal(matrix, matrix.T)
        or not np.isfinite(matrix).all()
        or not np.isfinite(linear).all()
        or not np.isfinite(radius)
        or radius < 0
    ):
        raise ValueError("TRS requires finite symmetric coefficients and radius >= 0")

    # v = radius * x. Scale by radius * scale for a well-scaled unit-ball solve.
    scale = max(float(np.max(np.abs(matrix))) * radius, float(np.max(np.abs(linear))))
    if radius == 0 or scale == 0:
        return TrustRegionResult(np.zeros_like(linear), 0.0, 0.0, 0.0, "trivial", 0)
    if not np.isfinite(scale):
        raise OracleNumericalError("TRS coefficient scale overflow")

    m = (matrix * radius) / scale
    g = linear / scale
    try:
        eigenvalues, basis = np.linalg.eigh(m)
    except np.linalg.LinAlgError as exc:
        raise OracleNumericalError("TRS eigendecomposition failed") from exc
    d = basis.T @ g
    eta0 = max(0.0, float(eigenvalues[-1]))
    gaps = eta0 - eigenvalues
    tolerance = 64 * EPS * max(1.0, float(np.max(np.abs(eigenvalues))))
    singular = gaps <= tolerance

    w = np.divide(d, gaps, out=np.zeros_like(d), where=~singular)
    norm = float(np.linalg.norm(w))
    delta = 0.0
    iterations = 0
    if np.linalg.norm(d[singular]) <= tolerance and norm <= 1.0:
        branch = "interior"
        if eta0 > 0.0:
            # Pick a direction from the top eigenspace via its projector.
            top = basis[:, singular]
            coordinate = np.argmax(np.sum(top * top, axis=1))
            direction = top[coordinate, :].copy()
            direction /= np.linalg.norm(direction)
            if direction @ d[singular] < 0:
                direction *= -1
            w[singular] = sqrt((1 - norm) * (1 + norm)) * direction
            branch = "hard"
    else:
        w, delta, iterations = _secular(d, gaps)
        branch = "secular"
    x = basis @ w
    x_norm = float(np.linalg.norm(x))
    eta = eta0 + delta
    kkt = max(
        float(np.linalg.norm(((eta0 * x - m @ x) + delta * x) - g)),
        max(0.0, x_norm - 1.0),
        max(0.0, float(eigenvalues[-1]) - eta),
        abs(eta * (x_norm**2 - 1.0)),
    )
    value = float(x @ (m @ x + 2 * g))

    # Evaluate (34) above the spectral endpoint, including in the hard case.
    # The spectral offset is kept separate to avoid cancellation at the pole.
    padding = max(tolerance, 16 * EPS * eta)
    dual = float(eta + padding + np.sum(d * (d / (gaps + delta + padding))))
    dual += 32 * EPS * max(1.0, abs(dual), abs(value))
    if (
        not np.isfinite([value, dual, kkt]).all()
        or kkt > TOL
        or dual < value
        or dual - value > TOL * max(1.0, abs(value))
    ):
        raise OracleNumericalError(
            f"TRS validation failed (branch={branch}, KKT={kkt:.3g}, dual gap={dual - value:.3g})"
        )

    factor = scale * radius
    if not np.isfinite([value * factor, dual * factor]).all():
        raise OracleNumericalError("TRS objective overflow")
    return TrustRegionResult(x * radius, value * factor, dual * factor, kkt, branch, iterations)
