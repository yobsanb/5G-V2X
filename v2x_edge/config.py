"""YAML loading and eager validation for the edge and training schemas."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from v2x_edge.registry import (
    DETECTION_BACKENDS,
    DETECTION_MODELS,
    OPTIMIZERS,
    SCHEDULERS,
    SEGMENTATION_BACKENDS,
    SEGMENTATION_MODELS,
)


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping")
    return value


def _finite_number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"'{name}' must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"'{name}' must be >= {minimum}")
    return value


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"'{name}' must be an integer >= {minimum}")
    return value


def _string_list(value: Any, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"'{name}' must be a list of non-empty strings")
    return list(value)


def _scale_pair(value: Any, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"'{name}' must be [min, max]")
    low = _finite_number(value[0], f"{name}[0]", 0.0)
    high = _finite_number(value[1], f"{name}[1]", 0.0)
    if not 0.0 < low <= high:
        raise ValueError(f"'{name}' must satisfy 0 < min <= max")


def _probability(value: Any, name: str) -> None:
    if value is None:
        return
    if not 0.0 <= _finite_number(value, name) <= 1.0:
        raise ValueError(f"'{name}' must be in [0, 1]")


def _reject_unknown(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = section.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown '{name}' configuration keys: {sorted(unknown)}")


# --------------------------------------------------------------------------- #
# Runtime edge system
# --------------------------------------------------------------------------- #

# Union of keys valid for any backend. Per-backend requirements are checked below;
# this catches typos, which would otherwise be accepted and silently do nothing.
DETECTOR_KEYS = {
    "backend",
    "confidence_threshold",
    "allowed_labels",
    "pretrained",
    "pretrained_model",
    "image_size",
    "checkpoint",
    "class_names",
}
SEGMENTATION_KEYS = {
    "enabled",
    "backend",
    "pretrained",
    "pretrained_model",
    "inference_size",
    "checkpoint",
}

RISK_KEYS = {
    "horizon_seconds",
    "collision_distance_m",
    "minimum_relative_speed_mps",
    "max_object_age_seconds",
    "max_pair_time_skew_seconds",
    "critical_ttc_seconds",
    "warning_ttc_seconds",
}


def validate_edge_config(data: dict[str, Any]) -> None:
    system = _mapping(data, "system")
    camera = _mapping(data, "camera")
    perception = _mapping(data, "perception")
    detector = _mapping(perception, "detector")
    tracking = _mapping(data, "tracking")
    localization = _mapping(data, "localization")
    world = _mapping(data, "world")
    risk = _mapping(data, "risk")
    v2x = _mapping(data, "v2x")

    if not isinstance(system.get("device", "auto"), str):
        raise ValueError("'system.device' must be a string")
    seed = system.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("'system.seed' must be an integer")
    if not isinstance(camera.get("source"), (str, int)) or isinstance(camera.get("source"), bool):
        raise ValueError("'camera.source' must be a camera index or path")
    for key in ("width", "height", "fps"):
        _finite_number(camera.get(key), f"camera.{key}", 1.0)

    backend = detector.get("backend")
    if not isinstance(backend, str) or backend not in DETECTION_BACKENDS:
        raise ValueError(f"Unsupported detector backend: {backend}")
    _reject_unknown(detector, DETECTOR_KEYS, "perception.detector")
    if "image_size" in detector:
        _positive_int(detector["image_size"], "perception.detector.image_size", 32)
    confidence = _finite_number(
        detector.get("confidence_threshold", 0.45), "perception.detector.confidence_threshold", 0.0
    )
    if confidence > 1.0:
        raise ValueError("'perception.detector.confidence_threshold' must be <= 1")
    if backend == "checkpoint":
        if not isinstance(detector.get("checkpoint"), str) or not detector["checkpoint"]:
            raise ValueError("checkpoint detector requires 'perception.detector.checkpoint'")
        _string_list(detector.get("class_names"), "perception.detector.class_names")
    _string_list(detector.get("allowed_labels"), "perception.detector.allowed_labels")

    segmentation = perception.get("segmentation") or {}
    if not isinstance(segmentation, dict):
        raise ValueError("'perception.segmentation' must be a mapping")
    _reject_unknown(segmentation, SEGMENTATION_KEYS, "perception.segmentation")
    if segmentation.get("enabled"):
        segmentation_backend = str(segmentation.get("backend", "segformer"))
        if segmentation_backend not in SEGMENTATION_BACKENDS:
            raise ValueError(f"Unsupported segmentation backend: {segmentation_backend}")
        if segmentation_backend == "checkpoint" and not segmentation.get("checkpoint"):
            raise ValueError("checkpoint segmenter requires 'perception.segmentation.checkpoint'")
    size = segmentation.get("inference_size")
    if size is not None:
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError("'perception.segmentation.inference_size' must be [width, height]")
        for index, value in enumerate(size):
            _positive_int(value, f"perception.segmentation.inference_size[{index}]")

    iou = _finite_number(tracking.get("iou_threshold"), "tracking.iou_threshold", 0.0)
    if iou > 1.0:
        raise ValueError("'tracking.iou_threshold' must be <= 1")
    _positive_int(tracking.get("max_age"), "tracking.max_age", 0)
    _positive_int(tracking.get("min_hits"), "tracking.min_hits", 1)

    if not isinstance(localization.get("homography_path"), str) or not localization["homography_path"]:
        raise ValueError("'localization.homography_path' must be a path string")
    _finite_number(world.get("stale_after_seconds"), "world.stale_after_seconds", 0.0)

    horizon = _finite_number(risk.get("horizon_seconds"), "risk.horizon_seconds", 0.0)
    collision_distance = _finite_number(risk.get("collision_distance_m"), "risk.collision_distance_m", 0.0)
    if horizon <= 0.0 or collision_distance <= 0.0:
        raise ValueError("risk horizon and collision distance must be positive")
    _finite_number(risk.get("minimum_relative_speed_mps", 0.0), "risk.minimum_relative_speed_mps", 0.0)
    for key in ("max_object_age_seconds", "max_pair_time_skew_seconds"):
        if key in risk:
            _finite_number(risk[key], f"risk.{key}", 0.0)
    critical_ttc = _finite_number(risk.get("critical_ttc_seconds", 1.5), "risk.critical_ttc_seconds", 0.0)
    warning_ttc = _finite_number(risk.get("warning_ttc_seconds", 3.0), "risk.warning_ttc_seconds", 0.0)
    if not 0.0 < critical_ttc <= warning_ttc <= horizon:
        raise ValueError("risk TTC thresholds must satisfy 0 < critical <= warning <= horizon")
    # The whole section is splatted into RiskEngine, so unknown keys must fail here.
    _reject_unknown(risk, RISK_KEYS, "risk")

    if not isinstance(v2x.get("rsu_id"), str) or not v2x["rsu_id"].strip():
        raise ValueError("'v2x.rsu_id' must be a non-empty string")
    if v2x.get("transport") not in {"udp"}:
        raise ValueError("Only 'udp' transport is available from YAML configuration")
    if not isinstance(v2x.get("host"), str) or not v2x["host"]:
        raise ValueError("'v2x.host' must be a non-empty string")
    port = v2x.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("'v2x.port' must be in [1, 65535]")
    if _finite_number(v2x.get("max_message_age_seconds"), "v2x.max_message_age_seconds", 0.0) <= 0.0:
        raise ValueError("'v2x.max_message_age_seconds' must be positive")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

_COMMON_MODEL_KEYS = {"name", "pretrained", "config_overrides"}
DETECTION_MODEL_KEYS = {
    "dfine": _COMMON_MODEL_KEYS | {"pretrained_model", "image_size", "max_detections", "freeze_backbone"},
    "fasterrcnn": _COMMON_MODEL_KEYS | {"trainable_backbone_layers", "pretrained_backbone"},
}
SEGMENTATION_MODEL_KEYS = {
    "segformer": _COMMON_MODEL_KEYS | {"pretrained_model", "freeze_encoder"},
    "lraspp": _COMMON_MODEL_KEYS | {"pretrained_backbone"},
}


def _validate_run(data: dict[str, Any]) -> None:
    run = _mapping(data, "run")
    if not isinstance(run.get("name"), str) or not run["name"].strip():
        raise ValueError("'run.name' must be a non-empty string")
    if "/" in run["name"] or "\\" in run["name"]:
        raise ValueError("'run.name' must be a single directory name")
    if not isinstance(run.get("output_dir", "outputs"), str):
        raise ValueError("'run.output_dir' must be a string")
    if not isinstance(run.get("device", "auto"), str):
        raise ValueError("'run.device' must be a string")
    if isinstance(run.get("seed", 42), bool) or not isinstance(run.get("seed", 42), int):
        raise ValueError("'run.seed' must be an integer")


def _validate_optim_and_schedule(data: dict[str, Any]) -> None:
    optim = _mapping(data, "optim")
    if str(optim.get("name", "adamw")).lower() not in OPTIMIZERS:
        raise ValueError(f"'optim.name' must be one of {sorted(OPTIMIZERS)}")
    if _finite_number(optim.get("lr", 5e-4), "optim.lr", 0.0) <= 0.0:
        raise ValueError("'optim.lr' must be positive")
    _finite_number(optim.get("weight_decay", 1e-4), "optim.weight_decay", 0.0)
    if optim.get("clip_grad_norm") is not None:
        _finite_number(optim["clip_grad_norm"], "optim.clip_grad_norm", 0.0)

    schedule = _mapping(data, "schedule")
    if str(schedule.get("name", "cosine")).lower() not in SCHEDULERS:
        raise ValueError(f"'schedule.name' must be one of {sorted(SCHEDULERS)}")
    _positive_int(schedule.get("epochs", 10), "schedule.epochs")
    _finite_number(schedule.get("warmup_epochs", 0.0), "schedule.warmup_epochs", 0.0)
    if not 0.0 <= _finite_number(schedule.get("min_lr_factor", 0.01), "schedule.min_lr_factor") <= 1.0:
        raise ValueError("'schedule.min_lr_factor' must be in [0, 1]")

    train = _mapping(data, "train")
    if not isinstance(train.get("amp", True), bool):
        raise ValueError("'train.amp' must be a boolean")
    _positive_int(train.get("val_interval", 1), "train.val_interval")
    _positive_int(train.get("early_stopping_patience", 0), "train.early_stopping_patience", 0)


def _validate_loader(data: dict[str, Any]) -> None:
    _positive_int(data.get("batch_size", 2), "data.batch_size")
    _positive_int(data.get("num_workers", 0), "data.num_workers", 0)


def validate_detection_train_config(data: dict[str, Any]) -> None:
    _validate_run(data)
    _validate_optim_and_schedule(data)
    dataset = _mapping(data, "data")
    _validate_loader(dataset)

    if not isinstance(dataset.get("train_manifest"), str) or not dataset["train_manifest"]:
        raise ValueError("'data.train_manifest' is required")
    if dataset.get("root") is not None and not isinstance(dataset["root"], str):
        raise ValueError("'data.root' must be a string")
    num_classes = _positive_int(dataset.get("num_classes"), "data.num_classes", 2)
    names = _string_list(dataset.get("class_names"), "data.class_names")
    if names is not None and len(names) != num_classes - 1:
        raise ValueError("'data.class_names' must contain exactly num_classes - 1 entries")
    if dataset.get("val_manifest") and dataset.get("val_fraction"):
        raise ValueError("Set either 'data.val_manifest' or 'data.val_fraction', not both")
    if dataset.get("val_fraction") is not None:
        fraction = _finite_number(dataset["val_fraction"], "data.val_fraction", 0.0)
        if not 0.0 < fraction < 1.0:
            raise ValueError("'data.val_fraction' must be in (0, 1)")

    augment = dataset.get("augment") or {}
    if not isinstance(augment, dict):
        raise ValueError("'data.augment' must be a mapping")
    _reject_unknown(augment, {"horizontal_flip", "photometric", "scale_jitter"}, "data.augment")
    _probability(augment.get("horizontal_flip"), "data.augment.horizontal_flip")
    _probability(augment.get("photometric"), "data.augment.photometric")
    _scale_pair(augment.get("scale_jitter"), "data.augment.scale_jitter")

    model = _mapping(data, "model")
    name = str(model.get("name", "dfine")).lower()
    if name not in DETECTION_MODELS:
        raise ValueError(f"'model.name' must be one of {sorted(DETECTION_MODELS)}")
    if not isinstance(model.get("pretrained", True), bool):
        raise ValueError("'model.pretrained' must be a boolean")
    # Reject keys belonging to the other architecture: silently ignoring them is how a
    # training run ends up not doing what the config appears to say.
    _reject_unknown(model, DETECTION_MODEL_KEYS[name], f"model ({name})")
    if name == "dfine":
        _positive_int(model.get("image_size", 640), "model.image_size", 32)
        _positive_int(model.get("max_detections", 300), "model.max_detections")
        if not isinstance(model.get("pretrained_model", ""), str):
            raise ValueError("'model.pretrained_model' must be a string")
        if not isinstance(model.get("freeze_backbone", False), bool):
            raise ValueError("'model.freeze_backbone' must be a boolean")
    else:
        layers = model.get("trainable_backbone_layers", 6)
        if isinstance(layers, bool) or not isinstance(layers, int) or not 0 <= layers <= 6:
            raise ValueError("'model.trainable_backbone_layers' must be an integer in [0, 6]")


def validate_segmentation_train_config(data: dict[str, Any]) -> None:
    _validate_run(data)
    _validate_optim_and_schedule(data)
    dataset = _mapping(data, "data")
    _validate_loader(dataset)

    for key in ("train_images", "train_masks"):
        if not isinstance(dataset.get(key), str) or not dataset[key]:
            raise ValueError(f"'data.{key}' is required")
    if bool(dataset.get("val_images")) != bool(dataset.get("val_masks")):
        raise ValueError("'data.val_images' and 'data.val_masks' must be set together")
    num_classes = _positive_int(dataset.get("num_classes"), "data.num_classes", 2)
    _positive_int(dataset.get("width", 640), "data.width")
    _positive_int(dataset.get("height", 384), "data.height")
    ignore_index = dataset.get("ignore_index", 255)
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise ValueError("'data.ignore_index' must be an integer")
    weights = dataset.get("class_weights")
    if weights is not None:
        if not isinstance(weights, list) or len(weights) != num_classes:
            raise ValueError("'data.class_weights' must contain one weight per class")
        for index, weight in enumerate(weights):
            _finite_number(weight, f"data.class_weights[{index}]", 0.0)

    augment = dataset.get("augment") or {}
    if not isinstance(augment, dict):
        raise ValueError("'data.augment' must be a mapping")
    _reject_unknown(augment, {"horizontal_flip", "photometric", "scale_crop"}, "data.augment")
    _probability(augment.get("horizontal_flip"), "data.augment.horizontal_flip")
    _probability(augment.get("photometric"), "data.augment.photometric")
    _scale_pair(augment.get("scale_crop"), "data.augment.scale_crop")

    model = _mapping(data, "model")
    name = str(model.get("name", "segformer")).lower()
    if name not in SEGMENTATION_MODELS:
        raise ValueError(f"'model.name' must be one of {sorted(SEGMENTATION_MODELS)}")
    if not isinstance(model.get("pretrained", True), bool):
        raise ValueError("'model.pretrained' must be a boolean")
    _reject_unknown(model, SEGMENTATION_MODEL_KEYS[name], f"model ({name})")
    if name == "segformer":
        if not isinstance(model.get("pretrained_model", ""), str):
            raise ValueError("'model.pretrained_model' must be a string")
        if not isinstance(model.get("freeze_encoder", False), bool):
            raise ValueError("'model.freeze_encoder' must be a boolean")


_VALIDATORS = {
    "edge": validate_edge_config,
    "detection_train": validate_detection_train_config,
    "segmentation_train": validate_segmentation_train_config,
}


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must contain a YAML mapping")
    return data


def load_config(path: str | Path, schema: str = "edge") -> dict[str, Any]:
    if schema not in _VALIDATORS:
        raise ValueError(f"Unknown config schema '{schema}'; expected one of {sorted(_VALIDATORS)}")
    data = read_yaml(path)
    _VALIDATORS[schema](data)
    return data


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Applies `section.key=value` CLI overrides, parsing values as YAML scalars."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must use section.key=value")
        dotted, raw = override.split("=", 1)
        keys = [key for key in dotted.split(".") if key]
        if not keys:
            raise ValueError(f"Override '{override}' has an empty key path")
        target = data
        for key in keys[:-1]:
            nested = target.get(key)
            if not isinstance(nested, dict):
                nested = {}
                target[key] = nested
            target = nested
        target[keys[-1]] = yaml.safe_load(raw)
    return data


# Backwards-compatible alias for the original runtime-only validator name.
validate_config = validate_edge_config
