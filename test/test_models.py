import torch
from torch import nn

from v2x_edge.models import TrainableLRASPP


def test_segmentation_model_forward():
    model = TrainableLRASPP(num_classes=4, pretrained_backbone=False).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))
    assert output.shape == (1, 4, 64, 64)


def test_segmentation_model_backward():
    model = TrainableLRASPP(num_classes=3, pretrained_backbone=False).train()
    image = torch.rand(1, 3, 64, 64)
    mask = torch.zeros((1, 64, 64), dtype=torch.int64)
    loss = nn.CrossEntropyLoss()(model(image), mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
