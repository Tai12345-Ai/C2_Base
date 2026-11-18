"""RGB crops and deterministic identity-balanced batches."""

from collections import defaultdict
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler

from .manifest import Capture


class CaptureDataset(Dataset):
    def __init__(self, root: Path, records: list[Capture]):
        self.root = root.resolve()
        self.records = records
        self.identities = sorted({row.identity_id for row in records})
        self.labels = {identity: i for i, identity in enumerate(self.identities)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, request: int | tuple[int, bool]):
        index, flip = request if isinstance(request, tuple) else (request, False)
        row = self.records[index]
        path = (self.root / row.relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Capture escapes the dataset root: {row.relative_path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.sha256:
            raise ValueError(f"Capture bytes changed after manifest audit: {row.relative_path}")
        with Image.open(BytesIO(raw)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb = rgb.resize((112, 112), Image.Resampling.BILINEAR)
            if flip:
                rgb = ImageOps.mirror(rgb)
            pixels = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()
        return pixels.div_(127.5).sub_(1.0), self.labels[row.identity_id]


class IdentityBatchSampler(Sampler[list[tuple[int, bool]]]):
    """Index plan is a pure function of seed and attempt, including flip decisions."""

    def __init__(
        self, labels: list[int], identities_per_batch: int, seed: int,
        start: int, stop: int, *, flip: bool = True,
    ):
        groups = defaultdict(list)
        for index, label in enumerate(labels):
            groups[label].append(index)
        if not 2 <= identities_per_batch <= len(groups):
            raise ValueError("Batch must contain between 2 and the number of eligible IDs")
        if min(map(len, groups.values())) < 2:
            raise ValueError("Two distinct capture indices per training ID are required")
        if not 0 <= start <= stop or seed < 0:
            raise ValueError("Invalid sampler cursor or seed")
        self.groups = [groups[label] for label in sorted(groups)]
        self.size = identities_per_batch
        self.seed = seed
        self.start = start
        self.stop = stop
        self.flip = flip

    def __iter__(self):
        for attempt in range(self.start, self.stop):
            generator = np.random.default_rng(np.random.SeedSequence([self.seed, attempt]))
            choices = generator.choice(len(self.groups), self.size, replace=False)
            batch = []
            for label in choices:
                captures = generator.choice(self.groups[label], 2, replace=False)
                flips = generator.integers(0, 2, size=2) if self.flip else (0, 0)
                batch.extend((int(index), bool(flip)) for index, flip in zip(captures, flips, strict=True))
            yield batch

    def __len__(self) -> int:
        return self.stop - self.start
