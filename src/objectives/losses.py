"""Matched nominal, row-only and C2 pair objectives."""

from math import sqrt

import torch
from torch import Tensor

from src.optimizers.composition import CompositionResult, solve_composition, validate_radius
from .pairwise import pairwise_kernel


def nominal_pair_risk(h: Tensor) -> Tensor:
    size = h.shape[0]
    return h.sum() / (size * (size - 1))


def row_pair_risk(h: Tensor, radius: float) -> Tensor:
    """Uniform risk plus the support function of its first-order row response."""
    size = h.shape[0]
    validate_radius(size, radius)
    row_mean = h.mean(dim=1)
    b_norm = torch.linalg.vector_norm(row_mean - row_mean.mean()) / sqrt(size)
    return nominal_pair_risk(h) + 2 * radius * b_norm / ((size - 1) / size)


def composition_loss(h: Tensor, radius: float) -> tuple[Tensor, CompositionResult]:
    """Evaluate (23) on live H, using a detached CPU float64 witness."""
    result = solve_composition(h.detach().to(device="cpu", dtype=torch.float64).numpy(), radius)
    weights = torch.as_tensor(result.gradient_h, device=h.device, dtype=h.dtype)
    return (weights * h).sum(), result


def pair_objective(
    features: Tensor, arm: str, tau: float, radius: float, temperature: float
) -> tuple[Tensor, CompositionResult | None]:
    """Pair term of (27) or its matched nominal/row-only control."""
    h = pairwise_kernel(features, tau, temperature)
    if arm == "c2":
        return composition_loss(h, radius)
    if arm == "nominal":
        return nominal_pair_risk(h), None
    if arm == "row":
        return row_pair_risk(h, radius), None
    raise ValueError(f"Unknown pair objective: {arm}")
