from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from v2x_edge.data import (
    DetectionManifestDataset,
    build_detection_transform,
    detection_collate_fn,
    split_dataset,
)
from v2x_edge.models import build_detection_model, checkpoint_meta
from v2x_edge.utils import resolve_device, set_seed

from .engine import (
    AverageMeter,
    CheckpointManager,
    EarlyStopping,
    RunDirectory,
    Stopwatch,
    build_amp,
    build_optimizer,
    build_scheduler,
    clip_gradients,
    format_duration,
)
from .metrics import MeanAveragePrecision


def build_detection_loaders(cfg: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader | None, int]:
    data = cfg["data"]
    num_classes = int(data["num_classes"])
    transform = build_detection_transform(data.get("augment"))
    train_set = DetectionManifestDataset(data["train_manifest"], data.get("root"), transform=transform)
    train_set.validate_label_range(num_classes)

    val_set = None
    if data.get("val_manifest"):
        val_set = DetectionManifestDataset(data["val_manifest"], data.get("root"))
        val_set.validate_label_range(num_classes)
    elif data.get("val_fraction"):
        # Same seed and length give both calls the identical partition, so the
        # validation half comes from the un-augmented copy of the dataset.
        fraction = float(data["val_fraction"])
        plain = DetectionManifestDataset(data["train_manifest"], data.get("root"))
        train_set, _ = split_dataset(train_set, fraction, seed)
        _, val_set = split_dataset(plain, fraction, seed)

    workers = int(data.get("num_workers", 0))
    common = {
        "num_workers": workers,
        "collate_fn": detection_collate_fn,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=int(data.get("batch_size", 2)), shuffle=True, drop_last=False, **common
    )
    val_loader = (
        DataLoader(val_set, batch_size=int(data.get("batch_size", 2)), shuffle=False, **common)
        if val_set is not None
        else None
    )
    return train_loader, val_loader, num_classes


@torch.inference_mode()
def evaluate_detector(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    score_threshold: float = 0.05,
) -> dict[str, Any]:
    model.eval()
    metric = MeanAveragePrecision(num_classes)
    for images, targets in loader:
        outputs = model([image.to(device, non_blocking=True) for image in images])
        kept = []
        for output in outputs:
            keep = output["scores"] >= score_threshold
            kept.append({key: value[keep] for key, value in output.items()})
        metric.update(kept, targets)
    return metric.compute()


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    clip_norm: float | None,
) -> float:
    model.train()
    meter = AverageMeter()
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ]
        with torch.amp.autocast("cuda", enabled=use_amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {float(loss.detach().cpu())}")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip_norm:
            scaler.unscale_(optimizer)
            clip_gradients(model, clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        meter.update(float(loss.detach().cpu()), len(images))
    return meter.mean


def train_detector(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg, train_cfg = cfg["run"], cfg["train"]
    seed = int(run_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(run_cfg.get("device", "auto")))
    run = RunDirectory(run_cfg.get("output_dir", "outputs"), run_cfg["name"])
    run.save_config(cfg)

    train_loader, val_loader, num_classes = build_detection_loaders(cfg, seed)
    class_names = cfg["data"].get("class_names")
    if class_names is not None and len(class_names) != num_classes - 1:
        raise ValueError("'class_names' must contain exactly num_classes - 1 entries")

    model_cfg = cfg["model"]
    model = build_detection_model(model_cfg, num_classes).to(device)
    architecture = checkpoint_meta(model)

    epochs = int(cfg["schedule"].get("epochs", 10))
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["schedule"], max(1, len(train_loader)))
    use_amp, scaler = build_amp(device, bool(train_cfg.get("amp", True)))
    checkpoints = CheckpointManager(run.checkpoints, mode="max")
    stopper = EarlyStopping(int(train_cfg.get("early_stopping_patience", 0)), mode="max")
    val_interval = max(1, int(train_cfg.get("val_interval", 1)))
    clip_norm = cfg["optim"].get("clip_grad_norm")

    watch = Stopwatch()
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, use_amp, clip_norm
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(loss, 6),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "elapsed_s": round(watch.elapsed, 1),
        }
        monitored = None
        if val_loader is not None and (epoch % val_interval == 0 or epoch == epochs):
            metrics = evaluate_detector(model, val_loader, device, num_classes)
            monitored = float(metrics["map"])
            record.update(
                {
                    "val_map": round(monitored, 6),
                    "val_map_50": round(float(metrics["map_50"]), 6),
                    "val_map_75": round(float(metrics["map_75"]), 6),
                }
            )

        payload = {
            "format_version": 3,
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "class_names": list(class_names) if class_names else None,
            "epoch": epoch,
            "metrics": {key: value for key, value in record.items() if isinstance(value, (int, float))},
            **architecture,
        }
        improved = checkpoints.save(payload, monitored, epoch)
        record["best"] = improved
        run.log(record)
        history.append(record)
        print(
            f"epoch={epoch}/{epochs} loss={loss:.4f}"
            + (f" val_map={monitored:.4f}" if monitored is not None else "")
            + (" *" if improved else "")
            + f" [{format_duration(watch.elapsed)}]"
        )
        if monitored is not None and stopper.step(monitored):
            print(f"early stopping after {epoch} epochs")
            break

    summary = {
        "run": str(run.path),
        "epochs_completed": len(history),
        "best_epoch": checkpoints.best_epoch,
        "best_val_map": None if math.isinf(checkpoints.best_value) else checkpoints.best_value,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "checkpoints": str(run.checkpoints),
    }
    run.save_summary(summary)
    return summary
