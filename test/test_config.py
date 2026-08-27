import copy
from pathlib import Path

import pytest

from v2x_edge.config import apply_overrides, load_config, validate_config

ROOT = Path(__file__).parents[1] / "config"


def test_default_edge_config_loads():
    config = load_config(ROOT / "edge.yaml")
    assert config["v2x"]["port"] == 5005


def test_invalid_config_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("system: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_risk_key_rejected():
    # The risk section is splatted into RiskEngine, so typos must fail at load time.
    config = load_config(ROOT / "edge.yaml")
    config["risk"]["bogus_key"] = 1.0
    with pytest.raises(ValueError):
        validate_config(config)


def test_shipped_training_configs_are_valid():
    assert load_config(ROOT / "detector.yaml", schema="detection_train")["data"]["num_classes"] == 7
    assert load_config(ROOT / "segmenter.yaml", schema="segmentation_train")["data"]["num_classes"] == 8


def test_unknown_schema_rejected():
    with pytest.raises(ValueError):
        load_config(ROOT / "edge.yaml", schema="nonsense")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["data"].update(class_names=["only_one"]),
        lambda c: c["data"].update(val_manifest="a.jsonl", val_fraction=0.2),
        lambda c: c["data"].update(val_fraction=1.5),
        lambda c: c["data"]["augment"].update(scale_jitter=[1.5, 0.5]),
        lambda c: c["data"]["augment"].update(unknown_knob=1),
        lambda c: c["model"].update(name="yolo"),
        lambda c: c["model"].update(image_size=0),
        lambda c: c["model"].update(max_detections=0),
        # a Faster R-CNN key under name: dfine must fail rather than be ignored
        lambda c: c["model"].update(trainable_backbone_layers=4),
        lambda c: c["optim"].update(name="rmsprop"),
        lambda c: c["optim"].update(lr=0),
        lambda c: c["schedule"].update(name="triangular"),
        lambda c: c["schedule"].update(min_lr_factor=2.0),
        lambda c: c["run"].update(name="nested/path"),
    ],
)
def test_detection_training_config_rejects_bad_values(mutate):
    from v2x_edge.config import validate_detection_train_config

    config = copy.deepcopy(load_config(ROOT / "detector.yaml", schema="detection_train"))
    mutate(config)
    with pytest.raises(ValueError):
        validate_detection_train_config(config)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["data"].pop("val_masks"),
        lambda c: c["data"].update(class_weights=[1.0]),
        lambda c: c["data"].update(num_classes=1),
        lambda c: c["data"]["augment"].update(horizontal_flip=1.5),
        lambda c: c["data"].update(width=0),
        lambda c: c["model"].update(name="unet"),
        lambda c: c["model"].update(freeze_encoder="yes"),
        # an LR-ASPP key under name: segformer must fail rather than be ignored
        lambda c: c["model"].update(pretrained_backbone=True),
    ],
)
def test_segmentation_training_config_rejects_bad_values(mutate):
    from v2x_edge.config import validate_segmentation_train_config

    config = copy.deepcopy(load_config(ROOT / "segmenter.yaml", schema="segmentation_train"))
    mutate(config)
    with pytest.raises(ValueError):
        validate_segmentation_train_config(config)


def test_overrides_parse_yaml_scalars_and_create_paths():
    config = apply_overrides({}, ["schedule.epochs=5", "train.amp=false", "run.name=demo"])
    assert config["schedule"]["epochs"] == 5
    assert config["train"]["amp"] is False
    assert config["run"]["name"] == "demo"


def test_malformed_override_rejected():
    with pytest.raises(ValueError):
        apply_overrides({}, ["schedule.epochs"])


def test_fasterrcnn_model_section_still_validates():
    from v2x_edge.config import validate_detection_train_config

    config = copy.deepcopy(load_config(ROOT / "detector.yaml", schema="detection_train"))
    config["model"] = {"name": "fasterrcnn", "pretrained": False, "trainable_backbone_layers": 3}
    validate_detection_train_config(config)
    config["model"]["trainable_backbone_layers"] = 9
    with pytest.raises(ValueError):
        validate_detection_train_config(config)


def test_lraspp_model_section_still_validates():
    from v2x_edge.config import validate_segmentation_train_config

    config = copy.deepcopy(load_config(ROOT / "segmenter.yaml", schema="segmentation_train"))
    config["model"] = {"name": "lraspp", "pretrained": False}
    validate_segmentation_train_config(config)


@pytest.mark.parametrize(
    "mutate",
    [
        # A typo in a perception key must fail rather than be silently ignored.
        lambda c: c["perception"]["detector"].update(pretrained_modell="x"),
        lambda c: c["perception"]["detector"].update(image_size=0),
        lambda c: c["perception"]["detector"].update(backend="yolo"),
        lambda c: c["perception"]["segmentation"].update(inference_sizes=[1, 2]),
        lambda c: c["perception"]["segmentation"].update(inference_size=[1024]),
        lambda c: c["perception"]["segmentation"].update(inference_size=[1024, 0]),
        lambda c: c["perception"]["segmentation"].update(enabled=True, backend="checkpoint"),
    ],
)
def test_edge_perception_config_rejects_bad_values(mutate):
    config = copy.deepcopy(load_config(ROOT / "edge.yaml"))
    mutate(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_edge_config_accepts_a_checkpoint_detector():
    config = copy.deepcopy(load_config(ROOT / "edge.yaml"))
    config["perception"]["detector"] = {
        "backend": "checkpoint",
        "checkpoint": "outputs/detector/checkpoints/best.pt",
        "confidence_threshold": 0.4,
    }
    validate_config(config)
