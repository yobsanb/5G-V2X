import numpy as np
import torch

from v2x_edge.models import TorchvisionCocoDetector, TrainableFasterRCNN


def test_offline_coco_detector_smoke():
    detector = TorchvisionCocoDetector(pretrained=False, confidence_threshold=0.99, device="cpu")
    output = detector.predict(np.zeros((64, 96, 3), dtype=np.uint8))
    assert isinstance(output, list)


def test_trainable_detector_forward():
    model = TrainableFasterRCNN(num_classes=3, pretrained_backbone=False).eval()
    with torch.inference_mode():
        output = model([torch.rand(3, 64, 96)])
    assert isinstance(output, list)
    assert {"boxes", "labels", "scores"}.issubset(output[0].keys())


def test_trainable_detector_loss_forward():
    model = TrainableFasterRCNN(num_classes=2, pretrained_backbone=False).train()
    image = torch.rand(3, 64, 96)
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 40.0, 50.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    losses = model([image], [target])
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
