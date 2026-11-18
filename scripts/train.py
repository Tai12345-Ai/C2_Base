"""Common ArcFace warmup and matched nominal, row-risk and C2 training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.captures import CaptureDataset
from src.data.manifest import load_manifest, train_records
from src.evaluation.data import score_split
from src.evaluation.verification import composition_metrics
from src.models.face import FaceModel
from src.training.checkpoint import read_checkpoint
from src.training.config import load_config
from src.training.engine import configure_runtime, run_training
from src.utils.io import sha256_file, write_json

ARMS = ("nominal", "row", "c2")


def device_from_arg(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda", 0 if device.index is None else device.index)
    elif device.type != "cpu":
        raise ValueError("Use cpu, cuda or auto")
    return device


def warmup_threshold(shared: Path, signature: dict, protocol_seed: int, model: FaceModel) -> tuple[float, str]:
    lock = json.loads((shared / "tau_train.json").read_text(encoding="utf-8"))
    best = shared / "best.pt"
    checkpoint_hash = sha256_file(best)
    if (
        lock["checkpoint_sha256"] != checkpoint_hash
        or lock["manifest_sha256"] != signature["manifest_sha256"]
        or lock["scope"] != signature["scope"]
        or lock["protocol_seed"] != protocol_seed
    ):
        raise ValueError("The training threshold lock must match the common checkpoint, split and protocol")
    state = read_checkpoint(best)
    training_signature = state["signature"]
    if (
        training_signature["arm"] != "warmup"
        or training_signature["manifest_sha256"] != signature["manifest_sha256"]
        or training_signature["scope"] != signature["scope"]
        or training_signature["config"]["seed"] != signature["seed"]
    ):
        raise ValueError("Expected a common ArcFace checkpoint for this scope and seed")
    model.load_state_dict(state["model"])
    return lock["threshold"], checkpoint_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("warmup", *ARMS))
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup-dir", type=Path)
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Pause after this many successful updates")
    args = parser.parse_args()
    if args.arm != "warmup" and args.warmup_dir is None:
        parser.error("Matched arms require --warmup-dir")
    if args.warmup_dir and args.out.resolve() == args.warmup_dir.resolve():
        parser.error("Use a separate output directory for each arm")

    config = load_config(args.config, args.seed, args.amp)
    configure_runtime(config)
    device = device_from_arg(args.device)
    records = load_manifest(args.manifest)
    manifest_hash = sha256_file(args.manifest)
    train = train_records(records, args.scope)
    dataset = CaptureDataset(args.data_root, train)
    labels = [dataset.labels[row.identity_id] for row in train]
    model = FaceModel(len(dataset.identities))
    signature = {
        "manifest_sha256": manifest_hash, "scope": args.scope, "seed": config.seed,
        "identities": dataset.identities, "initialization_sha256": None,
    }
    tau = None
    if args.arm != "warmup":
        tau, signature["initialization_sha256"] = warmup_threshold(
            args.warmup_dir, signature, config.protocol_seed, model
        )
        # Start all three arms from the same stochastic stream after loading weights.
        configure_runtime(config)

    def score_dev(encoder, role):
        return score_split(
            encoder, records, args.data_root, "dev", config.protocol_seed, device,
            config.eval_batch_size, config.workers, dev_role=role,
        )

    def select_common_checkpoint(trained, _epoch):
        identities, scores, fallbacks = score_dev(trained.encoder, "selection")
        fit = scores.fit_thresholds([0.01])["0.01"]
        impostor, genuine = scores.errors_at(fit["threshold"])
        uniform = np.full(len(identities), 1 / len(identities))
        metrics = composition_metrics(impostor, genuine, uniform)
        return {"dev_tar": metrics["tar"], "dev_fmr": metrics["fmr"],
                "dev_target_fmr": 0.01, "dev_norm_fallbacks": fallbacks}

    cursor = run_training(
        model, dataset, labels, config, args.arm, tau, args.out, signature, device,
        resume=args.resume, stop_after=args.stop_after,
        after_epoch=select_common_checkpoint if args.arm == "warmup" else None,
    )
    state_config = json.loads((args.out / "run.json").read_text(encoding="utf-8"))
    complete = cursor.updates == state_config["target_updates"]
    print(json.dumps({"arm": args.arm, "updates": cursor.updates, "attempts": cursor.attempts,
                      "complete": complete, "best_epoch": cursor.best_epoch}))
    if complete and args.arm == "warmup":
        best = args.out / "best.pt"
        model.load_state_dict(read_checkpoint(best)["model"])
        ids, scores, fallbacks = score_dev(model.encoder, "threshold")
        lock = scores.fit_thresholds([0.01])["0.01"] | {
            "checkpoint_sha256": sha256_file(best),
            "manifest_sha256": manifest_hash, "scope": args.scope,
            "protocol_seed": config.protocol_seed, "split": "dev", "dev_role": "threshold",
            "target_fmr": 0.01, "ids": len(ids), "norm_fallbacks": fallbacks,
        }
        lock_path = args.out / "tau_train.json"
        if lock_path.exists():
            old_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            expected = {k: lock[k] for k in ("checkpoint_sha256", "manifest_sha256", "scope", "protocol_seed")}
            if any(old_lock.get(k) != value for k, value in expected.items()):
                raise ValueError("Existing training threshold lock refers to a different initialization")
            lock = old_lock
        else:
            write_json(lock_path, lock)
        print(json.dumps({"tau_train_locked": lock}, allow_nan=False))


if __name__ == "__main__":
    main()
