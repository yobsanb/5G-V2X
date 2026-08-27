"""D-FINE adapter: conventions, decoding, and the torchvision-compatible API."""

import numpy as np
import pytest
import torch

from v2x_edge.models import DFineDetector, TrainableDFine, build_dfine_from_checkpoint
from v2x_edge.models.dfine import (
    coco_label_names,
    normalized_cxcywh_to_xyxy,
    xyxy_to_normalized_cxcywh,
)

# decoder_layers must stay >= 2: D-FINE ties bbox_embed.0 into the later layers.
TINY = {"encoder_layers": 1, "decoder_layers": 2, "num_queries": 12}


def _model(num_classes=3, image_size=128, max_detections=12):
    return TrainableDFine(
        num_classes=num_classes,
        pretrained=False,
        image_size=image_size,
        max_detections=max_detections,
        config_overrides=TINY,
    )


def _target(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_box_conversion_round_trips():
    boxes = torch.tensor([[10.0, 20.0, 110.0, 140.0], [0.0, 0.0, 50.0, 50.0]])
    back = normalized_cxcywh_to_xyxy(xyxy_to_normalized_cxcywh(boxes, 240, 320), 240, 320)
    assert torch.allclose(back, boxes, atol=1e-4)


def test_box_conversion_handles_empty():
    empty = torch.zeros(0, 4)
    assert xyxy_to_normalized_cxcywh(empty, 10, 10).shape == (0, 4)
    assert normalized_cxcywh_to_xyxy(empty, 10, 10).shape == (0, 4)


def test_repo_labels_shift_to_zero_indexed_for_dfine():
    model = _model()
    labels = model._hf_labels([_target([[10, 20, 110, 140]], [1])], [(240, 320)], torch.device("cpu"))
    # Repo reserves 0 for background; D-FINE has no background class.
    assert labels[0]["class_labels"].tolist() == [0]


def test_decode_matches_reference_post_processor():
    # Guards the hand-written focal-loss decode against the transformers implementation.
    from transformers import RTDetrImageProcessor

    torch.manual_seed(0)
    model = _model(max_detections=300).eval()
    images = [torch.rand(3, 240, 320), torch.rand(3, 200, 400)]
    with torch.inference_mode():
        pixel_values, sizes = model._batch(images)
        outputs = model.model(pixel_values=pixel_values)
        mine = model.decode(outputs.logits, outputs.pred_boxes, sizes)
    reference = RTDetrImageProcessor().post_process_object_detection(
        outputs, threshold=0.0, target_sizes=list(sizes), use_focal_loss=True
    )
    for got, expected in zip(mine, reference, strict=True):
        count = min(len(got["scores"]), len(expected["scores"]))
        assert torch.allclose(got["boxes"][:count], expected["boxes"][:count], atol=1e-4)
        assert torch.allclose(got["scores"][:count], expected["scores"][:count], atol=1e-6)
        # Reference labels are 0-indexed; ours are shifted into repo convention.
        assert torch.equal(got["labels"][:count] - 1, expected["labels"][:count])


def test_training_returns_single_summed_loss():
    model = _model().train()
    losses = model(
        [torch.rand(3, 128, 128), torch.rand(3, 96, 160)],
        [_target([[10, 10, 60, 60]], [1]), _target([], [])],
    )
    # D-FINE reports 60+ auxiliary terms; outputs.loss already sums them, so the
    # adapter must expose exactly one entry or the trainer would double count.
    assert list(losses) == ["loss"]
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_training_requires_targets():
    with pytest.raises(ValueError):
        _model().train()([torch.rand(3, 128, 128)])


def test_eval_output_matches_torchvision_detection_api():
    model = _model().eval()
    with torch.inference_mode():
        outputs = model([torch.rand(3, 240, 320)])
    assert isinstance(outputs, list) and len(outputs) == 1
    assert {"boxes", "scores", "labels"} == set(outputs[0])
    assert outputs[0]["boxes"].shape[1] == 4
    assert int(outputs[0]["labels"].min()) >= 1


def test_variable_input_sizes_are_resized_to_a_square():
    model = _model(image_size=96)
    pixel_values, sizes = model._batch([torch.rand(3, 240, 320), torch.rand(3, 100, 100)])
    assert pixel_values.shape == (2, 3, 96, 96)
    assert sizes == [(240, 320), (100, 100)]


def test_num_classes_must_include_background():
    with pytest.raises(ValueError):
        TrainableDFine(num_classes=1, pretrained=False, config_overrides=TINY)


def test_checkpoint_round_trip_rebuilds_architecture(tmp_path):
    model = _model(num_classes=3, image_size=96, max_detections=7)
    payload = {
        "model_state": model.state_dict(),
        "num_classes": 3,
        "class_names": ["car", "person"],
        **model.checkpoint_meta(),
    }
    path = tmp_path / "dfine.pt"
    torch.save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["model_type"] == "dfine"
    rebuilt, names = build_dfine_from_checkpoint(loaded)
    assert names == ["car", "person"]
    assert rebuilt.image_size == 96 and rebuilt.max_detections == 7

    image = torch.rand(3, 120, 160)
    model.eval()
    with torch.inference_mode():
        before = model([image])[0]["boxes"]
        after = rebuilt([image])[0]["boxes"]
    assert torch.allclose(before, after, atol=1e-5)


def test_checkpoint_rejects_class_name_mismatch():
    model = _model(num_classes=3)
    payload = {
        "model_state": model.state_dict(),
        "num_classes": 3,
        "class_names": ["only_one"],
        **model.checkpoint_meta(),
    }
    with pytest.raises(ValueError):
        build_dfine_from_checkpoint(payload)


def test_runtime_detector_emits_valid_detections():
    model = _model(num_classes=3, image_size=96)
    detector = DFineDetector(
        confidence_threshold=0.0, device="cpu", class_names=["car", "person"], model=model
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = detector.predict(frame)
    assert detections, "threshold 0 should keep every query"
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        assert 0.0 <= x1 < x2 <= 160.0
        assert 0.0 <= y1 < y2 <= 120.0
        assert detection.label in {"car", "person"}


def test_runtime_detector_applies_label_filter():
    detector = DFineDetector(
        confidence_threshold=0.0,
        allowed_labels=["car"],
        device="cpu",
        class_names=["car", "person"],
        model=_model(num_classes=3, image_size=96),
    )
    labels = {d.label for d in detector.predict(np.zeros((120, 160, 3), dtype=np.uint8))}
    assert labels <= {"car"}


def test_runtime_detector_rejects_bad_frames():
    detector = DFineDetector(
        device="cpu", class_names=["car", "person"], model=_model(num_classes=3, image_size=96)
    )
    with pytest.raises(ValueError):
        detector.predict(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        detector.predict(np.zeros((10, 10, 3), dtype=np.float32))


def test_class_names_must_match_model_output():
    with pytest.raises(ValueError):
        DFineDetector(device="cpu", class_names=["car"], model=_model(num_classes=3, image_size=96))


def test_coco_names_are_normalised_to_torchvision_spelling():
    class _Config:
        num_labels = 4
        id2label = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike"}

    # 'motorbike' is COCO-80 spelling; the edge config filters on 'motorcycle'.
    assert coco_label_names(_Config()) == ["person", "bicycle", "car", "motorcycle"]


def test_preprocessing_matches_the_shipped_processor():
    """Our resize/scale path must agree with RTDetrImageProcessor and stay in [0, 1]."""
    from transformers import RTDetrImageProcessor

    size = 128
    rng = np.random.default_rng(0)
    frame_bgr = rng.integers(0, 255, (97, 151, 3), dtype=np.uint8)
    rgb = np.ascontiguousarray(frame_bgr[..., ::-1])

    model = _model(image_size=size)
    ours, sizes = model._batch(
        [torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0]
    )
    theirs = RTDetrImageProcessor(size={"height": size, "width": size})(
        images=rgb, return_tensors="pt"
    )["pixel_values"]

    assert ours.shape == theirs.shape == (1, 3, size, size)
    assert sizes == [(97, 151)]
    # Both must stay in [0, 1]: the processor sets do_normalize=False.
    for tensor in (ours, theirs):
        assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
    # Resize kernels differ (torch bilinear vs the processor's), so compare loosely
    # on the statistics rather than pixel for pixel.
    assert abs(float(ours.mean()) - float(theirs.mean())) < 0.02
    assert abs(float(ours.std()) - float(theirs.std())) < 0.05


def test_imagenet_normalization_would_be_detected():
    """Negative control for the test above: normalized input leaves [0, 1]."""
    rng = np.random.default_rng(0)
    tensor = torch.from_numpy(
        np.ascontiguousarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    ).permute(2, 0, 1).to(torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    assert float(((tensor - mean) / std).min()) < 0.0
