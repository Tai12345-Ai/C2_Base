from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from src.models.face import ArcFace
from src.optimizers.trust_region import OracleNumericalError
from src.training.checkpoint import read_checkpoint
from src.training.config import TrainConfig
from src.training.engine import configure_runtime, run_training, train_update


class SyntheticCaptures(Dataset):
    def __init__(self):
        rng = torch.Generator().manual_seed(214)
        centers = torch.randn(5, 1, 7, generator=rng)
        self.samples = (centers + 0.18 * torch.randn(5, 3, 7, generator=rng)).reshape(15, 7)
        self.labels = list(np.repeat(np.arange(5), 3))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index, flip = index
        features = self.samples[index]
        return features.flip(0) if flip else features, int(self.labels[index])


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(7, 12), nn.BatchNorm1d(12), nn.Dropout(0.25))
        self.head = ArcFace(5, 12)


def same_state(x, y):
    if isinstance(x, torch.Tensor):
        assert torch.equal(x, y)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for key in x:
            same_state(x[key], y[key])
    elif isinstance(x, (tuple, list)):
        assert len(x) == len(y)
        for a, b in zip(x, y, strict=True):
            same_state(a, b)
    else:
        assert x == y


def test_checkpoint_resume_optimizer_scheduler_scaler_rng_and_seed_reproducibility(tmp_path):
    config = TrainConfig(identities_per_batch=3, warmup_epochs=1, pilot_epochs=2,
                         workers=0, checkpoint_every=2, max_skipped=0)
    data = SyntheticCaptures()
    device = torch.device("cpu")
    signature = {"dataset": "synthetic"}
    checkpoint_states = []
    for folder in ("uninterrupted", "resumed", "repeat"):
        configure_runtime(config)
        model = SmallModel()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        out = tmp_path / folder
        if folder == "resumed":
            partial = run_training(model, data, data.labels, config, "c2", 0.1,
                                   out, signature, device, stop_after=4)
            assert partial.updates == 4 and partial.attempts == 4
            # Force fresh model and random states before loading a mid-epoch checkpoint.
            random.seed(991)
            np.random.seed(991)
            torch.manual_seed(991)
            model = SmallModel()
        cursor = run_training(model, data, data.labels, config, "c2", 0.1,
                              out, signature, device, resume=(folder == "resumed"))
        assert cursor.updates == cursor.attempts == 6
        state = read_checkpoint(out / "last.pt")
        assert not torch.equal(before["encoder.0.weight"], state["model"]["encoder.0.weight"])
        assert not torch.equal(before["head.weight"], state["model"]["head.weight"])
        assert state["optimizer"]["state"] and state["scheduler"]["last_epoch"] == 6
        assert state["scaler"] == {} and state["rng"]["torch_cuda"] is None
        checkpoint_states.append(state)
        events = [json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()]
        steps = [event for event in events if event["event"] == "step"]
        assert len(steps) == 6
        assert all(step["oracle"]["nonnegative_witness"] for step in steps)
    for state in checkpoint_states[1:]:
        same_state(checkpoint_states[0], state)

    with pytest.raises(ValueError, match="Resume requires"):
        run_training(SmallModel(), data, data.labels, replace(config, radius=0.2), "c2",
                     0.1, tmp_path / "resumed", signature, device, resume=True)


@pytest.mark.parametrize("failure", ["oracle", "nan"])
def test_skip_restores_bn_and_gradients_and_does_not_step(failure, monkeypatch):
    config = TrainConfig(identities_per_batch=3, workers=0)
    configure_runtime(config)
    model = SmallModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    images = torch.randn(6, 7)
    if failure == "nan":
        images[0, 0] = float("nan")
    else:
        def fail(*args, **kwargs):
            raise OracleNumericalError("forced residual failure")

        monkeypatch.setattr("src.training.engine.pair_objective", fail)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    stats = train_update(model, (images, labels), optimizer, scaler, config,
                         "c2", 0.1, torch.device("cpu"))
    assert stats["status"] == "skipped"
    same_state(before, model.state_dict())
    assert not optimizer.state and all(p.grad is None for p in model.parameters())
