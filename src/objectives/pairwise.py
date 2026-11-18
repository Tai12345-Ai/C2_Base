"""Differentiable balanced pair loss, Blueprint (1), (24)–(26)."""

from math import isfinite

import torch
from torch import Tensor


def normalize_embeddings(features: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize the last dimension, with a fixed unit fallback below the norm floor."""
    if not features.is_floating_point() or features.ndim < 1 or features.shape[-1] == 0:
        raise ValueError("Expected floating embeddings with a nonempty feature dimension")
    x = features if features.dtype == torch.float64 else features.float()
    # Scale before computing the norm so even finite, very large features stay finite.
    scale = x.abs().amax(dim=-1, keepdim=True)
    scaled = x / scale.clamp_min(eps)
    scaled_norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    unit = scaled / scaled_norm.clamp_min(eps)
    fallback = torch.zeros_like(unit)
    fallback[..., 0] = 1
    degenerate = (scale <= eps) & (scaled_norm <= 1.0)
    return torch.where(degenerate, fallback, unit)


def pairwise_kernel(features: Tensor, tau: float, temperature: float = 0.05) -> Tensor:
    """Build H from [B, 2, D] with two distinct captures per identity."""
    if features.ndim != 3 or features.shape[0] < 2 or features.shape[1] != 2:
        raise ValueError("Expected [B, 2, D] for B >= 2 and two distinct captures per ID")
    if not isfinite(temperature) or temperature <= 0 or not isfinite(tau):
        raise ValueError("Expected finite tau and positive finite temperature")

    with torch.autocast(device_type=features.device.type, enabled=False):
        z = normalize_embeddings(features)
        size, captures, dim = z.shape
        flat = z.reshape(size * captures, dim)
        scores = (flat @ flat.T).reshape(size, captures, size, captures)
        negative = torch.sigmoid((scores - tau) / temperature).mean(dim=(1, 3))
        positive_cosine = (z[:, 0] * z[:, 1]).sum(dim=-1)
        positive = torch.sigmoid((tau - positive_cosine) / temperature)
        h = 0.5 * negative + 0.25 * (positive[:, None] + positive[None, :])
        offdiag = ~torch.eye(size, dtype=torch.bool, device=z.device)
        return ((h + h.T) * 0.5).masked_fill(~offdiag, 0.0)
