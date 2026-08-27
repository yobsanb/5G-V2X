"""SegFormer adapter: resolution handling, internal normalization, checkpoints."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from v2x_edge.models import SegFormerSegmenter, TrainableSegFormer, build_segformer_from_checkpoint

TINY = {"depths": [1, 1, 1, 1], "hidden_sizes": [16, 32, 64, 128], "decoder_hidden_size": 128}


def _model(num_classes=4):
    return TrainableSegFormer(num_classes=num_classes, pretrained=False, config_overrides=TINY)


def test_logits_come_back_at_input_resolution():
    # SegFormer emits H/4 x W/4; the adapter must upsample so callers see full res.
    model = _model().eval()
    with torch.inference_mode():
        logits = model(torch.rand(2, 3, 192, 320))
    assert logits.shape == (2, 4, 192, 320)


def test_non_multiple_of_four_resolution_still_matches():
    model = _model().eval()
    with torch.inference_mode():
        assert model(torch.rand(1, 3, 150, 230)).shape == (1, 4, 150, 230)


def test_normalization_happens_inside_forward():
    model = _model().eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        through_adapter = model(x)
        normalized = model.model(pixel_values=(x - model._mean) / model._std).logits
        raw = model.model(pixel_values=x).logits
    upsampled = F.interpolate(normalized, size=(128, 128), mode="bilinear", align_corners=False)
    assert torch.allclose(through_adapter, upsampled, atol=1e-5)
    # If normalization were skipped the two would coincide; they must not.
    assert (normalized - raw).abs().max() > 1e-4


def test_cross_entropy_with_ignore_index_lines_up():
    model = _model().train()
    masks = torch.randint(0, 4, (2, 96, 128))
    masks[0, :5, :5] = 255
    loss = nn.CrossEntropyLoss(ignore_index=255)(model(torch.rand(2, 3, 96, 128)), masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_rejects_malformed_input():
    model = _model()
    with pytest.raises(ValueError):
        model(torch.rand(3, 64, 64))
    with pytest.raises(ValueError):
        model(torch.rand(1, 1, 64, 64))


def test_num_classes_lower_bound():
    with pytest.raises(ValueError):
        TrainableSegFormer(num_classes=1, pretrained=False, config_overrides=TINY)


def test_freeze_encoder_leaves_decoder_trainable():
    model = TrainableSegFormer(
        num_classes=3, pretrained=False, freeze_encoder=True, config_overrides=TINY
    )
    assert not any(p.requires_grad for p in model.model.segformer.parameters())
    assert any(p.requires_grad for p in model.model.decode_head.parameters())


def test_checkpoint_round_trip_rebuilds_architecture(tmp_path):
    model = _model(num_classes=5)
    payload = {"model_state": model.state_dict(), "num_classes": 5, **model.checkpoint_meta()}
    path = tmp_path / "segformer.pt"
    torch.save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["model_type"] == "segformer"
    rebuilt = build_segformer_from_checkpoint(loaded)

    x = torch.rand(1, 3, 96, 128)
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(model(x), rebuilt(x), atol=1e-5)


def test_checkpoint_rejects_missing_fields():
    with pytest.raises(ValueError):
        build_segformer_from_checkpoint({"model_state": {}, "num_classes": 3})


def test_runtime_segmenter_returns_input_sized_map():
    segmenter = SegFormerSegmenter(device="cpu", inference_size=(128, 96), model=_model())
    mask = segmenter.predict(np.zeros((240, 400, 3), dtype=np.uint8))
    assert mask.shape == (240, 400)
    assert mask.dtype == np.int32
    assert mask.min() >= 0 and mask.max() < 4


def test_runtime_segmenter_without_resize():
    segmenter = SegFormerSegmenter(device="cpu", inference_size=None, model=_model())
    assert segmenter.predict(np.zeros((64, 96, 3), dtype=np.uint8)).shape == (64, 96)


def test_runtime_segmenter_rejects_bad_frames():
    segmenter = SegFormerSegmenter(device="cpu", inference_size=None, model=_model())
    with pytest.raises(ValueError):
        segmenter.predict(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        segmenter.predict(np.zeros((10, 10, 3), dtype=np.float32))
