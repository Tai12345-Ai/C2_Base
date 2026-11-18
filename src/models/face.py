"""IR-18 encoder and ArcFace classification head for the closed training split."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.objectives.pairwise import normalize_embeddings


class IRBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.PReLU(channels_out),
            nn.Conv2d(channels_out, channels_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
        )
        if channels_in == channels_out:
            self.shortcut = nn.Identity() if stride == 1 else nn.MaxPool2d(1, stride=stride)
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class IR18(nn.Module):
    embedding_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        stages = []
        channels = 64
        for width in (64, 128, 256, 512):
            stages.extend([IRBlock(channels, width, 2), IRBlock(width, width, 1)])
            channels = width
        self.body = nn.Sequential(*stages)
        self.projection = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim, affine=False),
        )
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, images: Tensor) -> Tensor:
        """Return features before L2 normalization."""
        if images.ndim != 4 or images.shape[1:] != (3, 112, 112):
            raise ValueError("IR-18 expects [N, 3, 112, 112] images")
        return self.projection(self.body(self.stem(images)))

    def embed(self, images: Tensor) -> Tensor:
        """Verification entry point (1), used in eval mode."""
        return normalize_embeddings(self(images))


class ArcFace(nn.Module):
    def __init__(
        self, num_identities: int, embedding_dim: int = 512, scale: float = 64, margin: float = 0.5
    ) -> None:
        super().__init__()
        if num_identities < 2 or embedding_dim < 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("ArcFace needs >= 2 classes, positive dimension and finite scale > 0")
        if not 0 <= margin < math.pi / 2:
            raise ValueError("ArcFace margin must be in [0, pi/2)")
        self.weight = nn.Parameter(torch.empty(num_identities, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.boundary = math.cos(math.pi - margin)
        self.fallback_offset = math.sin(math.pi - margin) * margin

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.weight.shape[1]:
            raise ValueError("ArcFace expects [N, embedding_dim] features")
        if labels.shape != (len(features),) or labels.dtype != torch.long:
            raise ValueError("ArcFace expects a long class index per image")
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = normalize_embeddings(features)
            w = normalize_embeddings(self.weight.to(dtype=z.dtype))
            cosine = F.linear(z, w).clamp(-1, 1)
            if self.margin == 0:
                return self.scale * cosine
            indices = labels[:, None]
            target = cosine.gather(1, indices)
            eps = 4 * torch.finfo(cosine.dtype).eps
            safe_target = target.clamp(-1 + eps, 1 - eps)
            sine = torch.sqrt((1 - safe_target) * (1 + safe_target))
            angular = safe_target * self.cos_margin - sine * self.sin_margin
            target_logit = torch.where(
                target > self.boundary, angular, target - self.fallback_offset
            )
            return self.scale * cosine.scatter(1, indices, target_logit)


class FaceModel(nn.Module):
    def __init__(self, num_identities: int) -> None:
        super().__init__()
        self.encoder = IR18()
        self.head = ArcFace(num_identities, self.encoder.embedding_dim)
