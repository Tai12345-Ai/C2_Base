"""Validated configuration shared across the matched training arms."""

from dataclasses import dataclass, replace
import math
from pathlib import Path

import yaml

from src.optimizers.composition import validate_radius


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    protocol_seed: int = 20261111
    identities_per_batch: int = 64
    captures_per_identity: int = 2
    temperature: float = 0.05
    beta: float = 1.0
    radius: float = 0.1
    warmup_epochs: int = 20
    pilot_epochs: int = 10
    lr: float = 0.01
    min_lr: float = 0.00001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    amp: str = "off"
    deterministic: bool = True
    workers: int = 2
    eval_batch_size: int = 128
    checkpoint_every: int = 250
    max_skipped: int = 8

    def __post_init__(self):
        integer_fields = (
            self.seed, self.protocol_seed, self.identities_per_batch, self.captures_per_identity,
            self.warmup_epochs, self.pilot_epochs, self.workers, self.eval_batch_size,
            self.checkpoint_every, self.max_skipped,
        )
        if (
            any(type(value) is not int for value in integer_fields)
            or min(self.seed, self.protocol_seed, self.workers, self.max_skipped) < 0
            or min(self.warmup_epochs, self.pilot_epochs, self.eval_batch_size, self.checkpoint_every) < 1
            or type(self.deterministic) is not bool
            or self.captures_per_identity != 2
            or self.amp not in ("off", "fp16", "bf16")
        ):
            raise ValueError("Invalid training counts or AMP mode (off, fp16, bf16)")
        values = (self.lr, self.min_lr, self.temperature, self.momentum, self.beta, self.weight_decay)
        if (
            not all(map(math.isfinite, values))
            or not 0 < self.min_lr <= self.lr
            or self.temperature <= 0 or not 0 <= self.momentum < 1
            or min(self.beta, self.weight_decay) < 0
        ):
            raise ValueError("Invalid optimizer or pair-loss hyperparameters")
        validate_radius(self.identities_per_batch, self.radius)


def load_config(path: Path, seed: int | None = None, amp: str | None = None) -> TrainConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Training config must be a YAML mapping")
    overrides = {k: v for k, v in {"seed": seed, "amp": amp}.items() if v is not None}
    return TrainConfig(**(data | overrides))
