from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from v2x_edge.data import SegmentationFolderDataset, build_segmentation_transform
from v2x_edge.models import build_segmentation_model, checkpoint_meta
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
from .metrics import SegmentationMetrics


def build_segmentation_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader | None, int, int]:
    data = cfg["data"]
    num_classes = int(data["num_classes"])
    ignore_index = int(data.get("ignore_index", 255))
    size = (int(data.get("width", 640)), int(data.get("height", 384)))

    train_set = SegmentationFolderDataset(
        data["train_images"],
        data["train_masks"],
        transform=build_segmentation_transform(data.get("augment"), size, ignore_index, training=True),
    )
    val_set = None
    if data.get("val_images") and data.get("val_masks"):
        val_set = SegmentationFolderDataset(
            data["val_images"],
            data["val_masks"],
            transform=build_segmentation_transform(None, size, ignore_index, training=False),
        )

    workers = int(data.get("num_workers", 0))
    batch_size = int(data.get("batch_size", 4))
    common = {"num_workers": workers, "persistent_workers": workers > 0}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common)
    val_loader = (
        DataLoader(val_set, batch_size=batch_size, shuffle=False, **common) if val_set is not None else None
    )
    return train_loader, val_loader, num_classes, ignore_index


def validate_mask_range(masks: torch.Tensor, num_classes: int, ignore_index: int) -> None:
    valid = masks != ignore_index
    if not torch.any(valid):
        return
    values = masks[valid]
    if int(values.min()) < 0 or int(values.max()) >= num_classes:
        raise ValueError(
            f"Mask class IDs must be in [0, {num_classes - 1}] or equal ignore_index={ignore_index}"
        )


@torch.inference_mode()
def evaluate_segmenter(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, Any]:
    model.eval()
    metric = SegmentationMetrics(num_classes, ignore_index)
    for images, masks in loader:
        logits = model(images.to(device, non_blocking=True))
        metric.update(logits.argmax(dim=1), masks)
    return metric.compute()


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    clip_norm: float | None,
    num_classes: int,
    ignore_index: int,
) -> float:
    model.train()
    meter = AverageMeter()
    for images, masks in loader:
        validate_mask_range(masks, num_classes, ignore_index)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(images), masks)
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
        meter.update(float(loss.detach().cpu()), images.shape[0])
    return meter.mean


def train_segmenter(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg, train_cfg = cfg["run"], cfg["train"]
    set_seed(int(run_cfg.get("seed", 42)))
    device = resolve_device(str(run_cfg.get("device", "auto")))
    run = RunDirectory(run_cfg.get("output_dir", "outputs"), run_cfg["name"])
    run.save_config(cfg)

    train_loader, val_loader, num_classes, ignore_index = build_segmentation_loaders(cfg)
    model = build_segmentation_model(cfg["model"], num_classes).to(device)
    architecture = checkpoint_meta(model)

    weights = cfg["data"].get("class_weights")
    criterion = nn.CrossEntropyLoss(
        ignore_index=ignore_index,
        weight=torch.tensor(weights, dtype=torch.float32, device=device) if weights else None,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )

    epochs = int(cfg["schedule"].get("epochs", 20))
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
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, use_amp, clip_norm, num_classes, ignore_index,
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(loss, 6),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "elapsed_s": round(watch.elapsed, 1),
        }
        monitored = None
        if val_loader is not None and (epoch % val_interval == 0 or epoch == epochs):
            metrics = evaluate_segmenter(model, val_loader, device, num_classes, ignore_index)
            monitored = float(metrics["miou"])
            record.update(
                {
                    "val_miou": round(monitored, 6),
                    "val_pixel_accuracy": round(float(metrics["pixel_accuracy"]), 6),
                    "val_mean_accuracy": round(float(metrics["mean_accuracy"]), 6),
                }
            )

        payload = {
            "format_version": 3,
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "ignore_index": ignore_index,
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
            + (f" val_miou={monitored:.4f}" if monitored is not None else "")
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
        "best_val_miou": None if math.isinf(checkpoints.best_value) else checkpoints.best_value,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "checkpoints": str(run.checkpoints),
    }
    run.save_summary(summary)
    return summary
