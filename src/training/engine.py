"""Train an ArcFace or matched pair-loss arm with reproducible step-based scheduling."""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.captures import IdentityBatchSampler
from src.objectives.losses import pair_objective
from src.optimizers.trust_region import OracleNumericalError
from src.training.checkpoint import Cursor, resume_checkpoint, save_checkpoint
from src.training.config import TrainConfig
from src.utils.io import write_json


def configure_runtime(config: TrainConfig) -> None:
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = not config.deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def train_update(model, batch, optimizer, scaler, config, arm, tau, device) -> dict:
    images, labels = batch
    identities = config.identities_per_batch
    paired_labels = labels.reshape(identities, 2)
    if not torch.equal(paired_labels[:, 0], paired_labels[:, 1]) or paired_labels[:, 0].unique().numel() != identities:
        raise ValueError("Training batch must contain B distinct identities with two grouped captures")
    optimizer.zero_grad(set_to_none=True)
    saved_buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    def skip(reason: str) -> dict:
        with torch.no_grad():
            for buffer, original in saved_buffers:
                buffer.copy_(original)
        optimizer.zero_grad(set_to_none=True)
        return {"status": "skipped", "reason": reason}

    amp_enabled = config.amp != "off"
    dtype = torch.float16 if config.amp == "fp16" else torch.bfloat16
    with torch.autocast(device.type, enabled=amp_enabled, dtype=dtype):
        features = model.encoder(images)
    if not torch.isfinite(features).all().item():
        return skip("nonfinite_encoder")
    norms = torch.linalg.vector_norm(features.float(), dim=-1)
    logits = model.head(features, labels)
    arcface = F.cross_entropy(logits, labels)
    pair = None
    oracle = None
    compute_pair_seconds = 0.0
    if arm != "warmup":
        start = perf_counter()
        try:
            pair, oracle = pair_objective(
                features.reshape(identities, 2, -1), arm, tau, config.radius, config.temperature
            )
        except OracleNumericalError as exc:
            return skip(f"oracle: {exc}")
        compute_pair_seconds = perf_counter() - start
    loss = arcface if pair is None else arcface + config.beta * pair
    if not torch.isfinite(loss).item():
        return skip("nonfinite_loss")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    local_norms = torch.stack([param.grad.detach().float().norm() for param in model.parameters()
                               if param.grad is not None])
    grad_norm = float(local_norms.norm().item())
    if not math.isfinite(grad_norm):
        if scaler.is_enabled():
            scaler.update(new_scale=scaler.get_scale() * 0.5)
        return skip("nonfinite_gradient")
    scaler.step(optimizer)
    scaler.update()
    return {
        "status": "updated",
        "loss": float(loss.item()), "arcface": float(arcface.item()),
        "pair": None if pair is None else float(pair.item()),
        "grad_norm": grad_norm, "norm_fallbacks": int((norms <= 1e-12).sum().item()),
        "pair_seconds_including_oracle": compute_pair_seconds,
        "oracle": None if oracle is None else oracle.diagnostics(),
    }


def run_training(
    model, dataset, labels: list[int], config: TrainConfig, arm: str, tau: float | None,
    out: Path, signature: dict, device: torch.device, *, resume: bool = False,
    after_epoch=None, stop_after: int | None = None,
) -> Cursor:
    """Run to the matched update target. stop_after permits a controlled pause for resume."""
    if arm not in ("warmup", "nominal", "row", "c2") or (arm != "warmup" and tau is None):
        raise ValueError("Fine-tuning needs a valid arm and the locked training threshold")
    if config.amp == "fp16" and device.type != "cuda":
        raise ValueError("FP16 mode requires CUDA. CPU AMP uses bf16.")
    if config.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected device does not support CUDA bfloat16")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if len(dataset) != len(labels):
        raise ValueError("Each training capture needs a sampler identity")

    size = config.identities_per_batch * 2
    per_epoch = math.ceil(len(dataset) / size)
    epochs = config.warmup_epochs if arm == "warmup" else config.pilot_epochs
    total = per_epoch * epochs
    if stop_after is not None and not 1 <= stop_after <= total:
        raise ValueError("--stop-after must be inside the iteration budget")
    end = total if stop_after is None else stop_after
    signature = signature | {"arm": arm, "config": asdict(config), "tau_train": tau,
                            "device_type": device.type, "target_updates": total}
    last = out / "last.pt"
    out.mkdir(parents=True, exist_ok=True)
    if not resume and (last.exists() or (out / "run.json").exists()):
        raise FileExistsError(f"{out} has an existing run. Use --resume or choose a new run directory.")

    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total, eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp == "fp16", init_scale=1024)
    cursor = resume_checkpoint(last, model, optimizer, scheduler, scaler, signature, device) if resume else Cursor()
    if not resume:
        save_checkpoint(last, model, optimizer, scheduler, scaler, cursor, signature, device)
    if not (out / "run.json").exists():
        write_json(out / "run.json", signature | {
            "torch": str(torch.__version__),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "steps_per_epoch": per_epoch,
        })
    if cursor.updates >= end:
        return cursor

    sampler = IdentityBatchSampler(
        labels, config.identities_per_batch, config.seed,
        cursor.attempts, total + config.max_skipped,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config.workers,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(config.seed),
    )
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch_start = perf_counter()
    segment_start_updates = cursor.updates
    log_path = out / "training.jsonl"
    incomplete_line = False
    if log_path.exists() and log_path.stat().st_size:
        with log_path.open("rb") as previous_log:
            previous_log.seek(-1, os.SEEK_END)
            incomplete_line = previous_log.read() != b"\n"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        if incomplete_line:
            log.write("\n")

        def publish(event: dict):
            log.write(json.dumps(event, allow_nan=False) + "\n")

        publish({"event": "start", "resume": resume, "from_updates": cursor.updates,
                 "from_attempts": cursor.attempts})
        for batch in loader:
            begin = perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            details = train_update(model, batch, optimizer, scaler, config, arm, tau, device)
            update = details["status"] == "updated"
            cursor.attempts += 1
            if update:
                cursor.updates += 1
                scheduler.step()
            seconds = perf_counter() - begin
            publish({"event": "step", "attempt": cursor.attempts, "updates": cursor.updates,
                     "lr": lr, "step_seconds": seconds, "images_per_second": size / seconds,
                     "peak_vram_bytes": _peak_vram(device), **details})

            epoch_end = update and cursor.updates % per_epoch == 0
            if epoch_end:
                epoch = cursor.updates // per_epoch
                epoch_seconds = perf_counter() - epoch_start
                completed_images = (cursor.updates - segment_start_updates) * size
                summary = {"event": "epoch", "epoch": epoch, "updates": cursor.updates,
                           "train_segment_seconds": epoch_seconds,
                           "epoch_is_partial_after_resume": segment_start_updates % per_epoch != 0,
                           "successful_images_per_second": completed_images / epoch_seconds,
                           "peak_vram_bytes": _peak_vram(device)}
                metrics = after_epoch(model, epoch) if after_epoch else {}
                if arm == "warmup" and "dev_tar" in metrics:
                    value = metrics["dev_tar"]
                    if cursor.best_dev_tar is None or value > cursor.best_dev_tar:
                        cursor.best_dev_tar, cursor.best_epoch = value, epoch
                        save_checkpoint(out / "best.pt", model, optimizer, scheduler, scaler, cursor, signature, device)
                publish(summary | metrics)
                print(json.dumps(summary | metrics, allow_nan=False), flush=True)
                model.train()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                segment_start_updates = cursor.updates
                epoch_start = perf_counter()

            too_many_skips = cursor.attempts - cursor.updates > config.max_skipped
            if epoch_end or (update and cursor.updates % config.checkpoint_every == 0) or cursor.updates == end or too_many_skips:
                save_checkpoint(last, model, optimizer, scheduler, scaler, cursor, signature, device)
            if too_many_skips:
                raise ArithmeticError("Skipped-update budget exhausted. Check training.jsonl.")
            if cursor.updates == end:
                break

    if cursor.updates != end:
        raise ArithmeticError("Successful update target was not reached. Check training.jsonl.")
    return cursor


def _peak_vram(device: torch.device) -> int | None:
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
