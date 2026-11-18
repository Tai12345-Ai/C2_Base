"""Atomic checkpoint with all state needed to resume a single-device run."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch

from src.utils.io import atomic_output


@dataclass
class Cursor:
    updates: int = 0
    attempts: int = 0
    best_dev_tar: float | None = None
    best_epoch: int | None = None


def capture_rng(device: torch.device) -> dict:
    algorithm, state, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (algorithm, state.tolist(), position, has_gauss, cached_gaussian),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    torch.random.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device=device)


def save_checkpoint(path, model, optimizer, scheduler, scaler, cursor, signature, device) -> None:
    state = {
        "format": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "cursor": asdict(cursor),
        "signature": signature,
        "rng": capture_rng(device),
    }
    with atomic_output(path, binary=True) as handle:
        torch.save(state, handle)


def read_checkpoint(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != 1:
        raise ValueError("Unsupported checkpoint format")
    return state


def resume_checkpoint(path, model, optimizer, scheduler, scaler, signature, device) -> Cursor:
    state = read_checkpoint(path)
    if state["signature"] != signature:
        raise ValueError("Resume requires the same arm, config, data hash, device and initialization")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"], device)
    return Cursor(**state["cursor"])
