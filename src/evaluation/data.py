"""Load a fixed, score-blind set of captures and return verification scores."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.captures import CaptureDataset
from src.data.manifest import Capture, eval_records
from .verification import VerificationScores, infer_embeddings


def score_split(
    encoder, records: list[Capture], data_root: Path, split: str, protocol_seed: int,
    device: torch.device, batch_size: int, num_workers: int, *, dev_role: str = "",
):
    identities, selected = eval_records(records, split, protocol_seed, dev_role)
    loader = DataLoader(
        CaptureDataset(data_root, selected), batch_size=batch_size, num_workers=num_workers,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(protocol_seed),
    )
    embeddings, fallback_count = infer_embeddings(encoder, loader, device)
    return identities, VerificationScores.from_embeddings(embeddings), fallback_count
