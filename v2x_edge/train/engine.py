"""Shared training infrastructure: run directories, meters, optimizers, schedules, AMP."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * weight
        self.count += weight

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class RunDirectory:
    """outputs/<name>/ holding the resolved config, metrics.jsonl and checkpoints."""

    def __init__(self, root: str | Path, name: str) -> None:
        if not name:
            raise ValueError("run name cannot be empty")
        self.path = Path(root) / name
        self.checkpoints = self.path / "checkpoints"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.path / "metrics.jsonl"

    def save_config(self, config: dict[str, Any]) -> None:
        with (self.path / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

    def log(self, record: dict[str, Any]) -> None:
        with self._metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    def save_summary(self, summary: dict[str, Any]) -> None:
        with (self.path / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, allow_nan=False)


class CheckpointManager:
    """Writes last.pt every epoch and best.pt when the monitored metric improves."""

    def __init__(self, directory: str | Path, mode: str = "max") -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_value = -math.inf if mode == "max" else math.inf
        self.best_epoch = 0

    def is_improvement(self, value: float) -> bool:
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def save(self, payload: dict[str, Any], metric: float | None, epoch: int) -> bool:
        torch.save(payload, self.directory / "last.pt")
        if metric is None or not math.isfinite(metric) or not self.is_improvement(metric):
            return False
        self.best_value = float(metric)
        self.best_epoch = int(epoch)
        torch.save(payload, self.directory / "best.pt")
        return True


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 5e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    if lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("lr must be positive and weight_decay non-negative")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Per-iteration schedule with optional linear warmup."""
    name = str(cfg.get("name", "cosine")).lower()
    if name == "none":
        return None
    epochs = int(cfg.get("epochs", 10))
    warmup_steps = max(0, int(float(cfg.get("warmup_epochs", 0.0)) * steps_per_epoch))
    total_steps = max(1, epochs * steps_per_epoch)
    min_factor = float(cfg.get("min_lr_factor", 0.01))
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError("'min_lr_factor' must be in [0, 1]")

    if name == "cosine":

        def factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, progress)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    elif name == "step":
        gamma = float(cfg.get("gamma", 0.1))
        step_epochs = int(cfg.get("step_epochs", max(1, epochs // 3)))

        def factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            return gamma ** ((step // steps_per_epoch) // max(1, step_epochs))

    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_amp(device: torch.device, enabled: bool) -> tuple[bool, torch.amp.GradScaler]:
    """Mixed precision is only enabled on CUDA; elsewhere the scaler is a no-op."""
    use_amp = bool(enabled) and device.type == "cuda"
    return use_amp, torch.amp.GradScaler("cuda", enabled=use_amp)


def clip_gradients(model: nn.Module, max_norm: float | None) -> float:
    if not max_norm:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm)))


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = int(patience)
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Returns True when training should stop."""
        if self.patience <= 0:
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class Stopwatch:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start
