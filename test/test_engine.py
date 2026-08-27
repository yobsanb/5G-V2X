import pytest
import torch
from torch import nn

from v2x_edge.train.engine import (
    AverageMeter,
    CheckpointManager,
    EarlyStopping,
    RunDirectory,
    build_optimizer,
    build_scheduler,
)


def test_average_meter_weights_by_batch_size():
    meter = AverageMeter()
    meter.update(1.0, 2)
    meter.update(4.0, 1)
    assert abs(meter.mean - 2.0) < 1e-9


def test_checkpoint_manager_only_overwrites_best_on_improvement(tmp_path):
    manager = CheckpointManager(tmp_path, mode="max")
    assert manager.save({"model_state": {}}, 0.4, epoch=1)
    assert not manager.save({"model_state": {}}, 0.3, epoch=2)
    assert manager.save({"model_state": {}}, 0.5, epoch=3)
    assert manager.best_epoch == 3
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()


def test_checkpoint_manager_ignores_missing_metric(tmp_path):
    manager = CheckpointManager(tmp_path, mode="max")
    assert not manager.save({"model_state": {}}, None, epoch=1)
    assert (tmp_path / "last.pt").is_file()
    assert not (tmp_path / "best.pt").exists()


def test_cosine_schedule_warms_up_then_decays():
    model = nn.Linear(2, 2)
    optimizer = build_optimizer(model, {"name": "adamw", "lr": 1.0})
    scheduler = build_scheduler(
        optimizer, {"name": "cosine", "epochs": 2, "warmup_epochs": 1, "min_lr_factor": 0.0}, 10
    )
    start = optimizer.param_groups[0]["lr"]
    peak = 0.0
    for _ in range(20):
        peak = max(peak, optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert start < peak
    assert optimizer.param_groups[0]["lr"] < peak


def test_unknown_optimizer_and_scheduler_rejected():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError):
        build_optimizer(model, {"name": "rmsprop"})
    with pytest.raises(ValueError):
        build_scheduler(build_optimizer(model, {}), {"name": "triangular"}, 10)


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="max")
    assert not stopper.step(0.5)
    assert not stopper.step(0.4)
    assert stopper.step(0.3)


def test_early_stopping_disabled_when_patience_is_zero():
    stopper = EarlyStopping(patience=0, mode="max")
    assert not any(stopper.step(0.1) for _ in range(10))


def test_run_directory_writes_config_and_metrics(tmp_path):
    run = RunDirectory(tmp_path, "demo")
    run.save_config({"run": {"name": "demo"}})
    run.log({"epoch": 1, "train_loss": 0.5})
    run.save_summary({"epochs_completed": 1})
    assert (run.path / "config.yaml").is_file()
    assert (run.path / "summary.json").is_file()
    assert (run.path / "metrics.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert run.checkpoints.is_dir()


def test_model_without_trainable_parameters_rejected():
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError):
        build_optimizer(model, {})


def test_amp_scaler_is_disabled_on_cpu():
    from v2x_edge.train.engine import build_amp

    use_amp, scaler = build_amp(torch.device("cpu"), enabled=True)
    assert use_amp is False
    assert scaler.is_enabled() is False
