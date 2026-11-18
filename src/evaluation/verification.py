"""Hard cosine verification and distinct-identity composition metrics."""

from dataclasses import dataclass
import math

import numpy as np
import torch

from src.data.manifest import hash_key
from src.objectives.pairwise import normalize_embeddings


@dataclass(frozen=True)
class VerificationScores:
    genuine: np.ndarray
    impostor: np.ndarray  # [identity, identity, capture, capture]

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray):
        z = np.asarray(embeddings, dtype=np.float32)
        if z.ndim != 3 or z.shape[0] < 2 or z.shape[1] != 2 or not np.isfinite(z).all():
            raise ValueError("Expected finite [B, 2, D] embeddings with B >= 2")
        n, _, d = z.shape
        if d == 0 or not np.allclose(np.linalg.norm(z, axis=-1), 1, atol=3e-6, rtol=0):
            raise ValueError("Verification expects L2-normalized embeddings")
        flat = z.reshape(n * 2, d)
        all_scores = np.clip(flat @ flat.T, -1, 1)
        all_scores = (all_scores + all_scores.T) / 2
        genuine = all_scores[2 * np.arange(n), 2 * np.arange(n) + 1]
        return cls(genuine, all_scores.reshape(n, 2, n, 2).transpose(0, 2, 1, 3))

    def fit_thresholds(self, levels: list[float]) -> dict:
        """Lock the smallest representable threshold obeying the FMR budget at ties."""
        n = len(self.genuine)
        rows, cols = np.triu_indices(n, k=1)
        pair_scores = self.impostor[rows, cols].reshape(-1)
        m = len(pair_scores)
        if not levels or any(not math.isfinite(level) or not 0 < level < 1 for level in levels):
            raise ValueError("Target FMR levels must be in (0, 1)")
        indexes = [m - math.floor(level * m) - 1 for level in levels]
        partitioned = np.partition(pair_scores, indexes)
        thresholds = {}
        for level, index in zip(levels, indexes, strict=True):
            boundary = partitioned[index]
            threshold = float(np.nextafter(boundary, np.float32(np.inf)))
            thresholds[str(level)] = {
                "threshold": threshold,
                "achieved_fmr": float(np.mean(pair_scores >= threshold)),
                "impostor_capture_comparisons": m,
            }
        return thresholds

    def errors_at(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        if not math.isfinite(threshold):
            raise ValueError("Verification threshold must be finite")
        impostor_error = np.mean(self.impostor >= threshold, axis=(2, 3))
        impostor_error = (impostor_error + impostor_error.T) / 2
        np.fill_diagonal(impostor_error, 0.0)
        return impostor_error, (self.genuine < threshold).astype(float)


def composition_metrics(
    impostor: np.ndarray, genuine_error: np.ndarray, probe: np.ndarray,
    gallery: np.ndarray | None = None,
) -> dict:
    """Condition on different original identities. Report genuine by probe and pair endpoint."""
    gallery = probe if gallery is None else gallery
    denominator = 1.0 - float(probe @ gallery)
    if denominator <= 0:
        raise ValueError("Different-identity trials require positive mass")
    fmr = float(probe @ impostor @ gallery) / denominator
    fnmr = float(probe @ genuine_error)
    endpoint = probe * (1 - gallery) / denominator
    return {"fmr": fmr, "fnmr": fnmr, "tar": 1 - fnmr,
            "endpoint_tar": float(endpoint @ (1 - genuine_error))}


def ordinary_profiles(identities: list[str], seed: int) -> dict[str, np.ndarray]:
    """Predetermined score-blind hash groups, with bounded raw weights 1 or 2."""
    bins = np.array([int(hash_key(seed, "ordinary-shift", key), 16) % 4 for key in identities])
    return {"nominal": np.ones(len(identities))} | {
        f"group_{group}_up": 1.0 + (bins == group) for group in range(4)
    }


def infer_embeddings(encoder, loader, device: torch.device) -> tuple[np.ndarray, int]:
    encoder.eval()
    chunks = []
    fallback_count = 0
    with torch.inference_mode():
        for images, _ in loader:
            raw = encoder(images.to(device, non_blocking=True)).float()
            fallback_count += int((torch.linalg.vector_norm(raw, dim=1) <= 1e-12).sum().item())
            features = normalize_embeddings(raw)
            if not torch.isfinite(features).all():
                raise ArithmeticError("Encoder produced non-finite verification embeddings")
            chunks.append(features.cpu().numpy())
    return np.concatenate(chunks).reshape(-1, 2, encoder.embedding_dim), fallback_count
