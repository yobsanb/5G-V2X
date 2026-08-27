#!/usr/bin/env python3
"""Build and run every model architecture offline, without downloading weights."""

from __future__ import annotations

import numpy as np
import torch

from v2x_edge.models import (
    TorchvisionCocoDetector,
    TrainableDFine,
    TrainableFasterRCNN,
    TrainableLRASPP,
    TrainableSegFormer,
)

# decoder_layers must stay >= 2: D-FINE ties bbox_embed.0 into the later layers.
DFINE_TINY = {"encoder_layers": 1, "decoder_layers": 2, "num_queries": 12}
SEGFORMER_TINY = {"depths": [1, 1, 1, 1], "hidden_sizes": [16, 32, 64, 128], "decoder_hidden_size": 128}


def main() -> None:
    image = np.zeros((96, 128, 3), dtype=np.uint8)

    dfine = TrainableDFine(
        num_classes=3, pretrained=False, image_size=128, config_overrides=DFINE_TINY
    ).eval()
    with torch.inference_mode():
        prediction = dfine([torch.rand(3, 96, 128)])[0]
    print(f"D-FINE forward OK; keys={sorted(prediction.keys())} boxes={tuple(prediction['boxes'].shape)}")

    dfine.train()
    losses = dfine(
        [torch.rand(3, 96, 128)],
        [{"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])}],
    )
    print(f"D-FINE loss OK; loss={float(sum(losses.values()).detach()):.2f}")

    segformer = TrainableSegFormer(
        num_classes=4, pretrained=False, config_overrides=SEGFORMER_TINY
    ).eval()
    with torch.inference_mode():
        logits = segformer(torch.rand(1, 3, 96, 128))
    print(f"SegFormer forward OK; shape={tuple(logits.shape)} (input resolution)")

    coco = TorchvisionCocoDetector(pretrained=False, confidence_threshold=0.99, device="cpu")
    print(f"SSDLite baseline OK; detections={len(coco.predict(image))}")

    detector = TrainableFasterRCNN(num_classes=4, pretrained_backbone=False).eval()
    with torch.inference_mode():
        output = detector([torch.rand(3, 96, 128)])[0]
    print(f"Faster R-CNN baseline OK; keys={sorted(output.keys())}")

    segmenter = TrainableLRASPP(num_classes=5, pretrained_backbone=False).eval()
    with torch.inference_mode():
        shape = tuple(segmenter(torch.rand(1, 3, 96, 128)).shape)
    print(f"LR-ASPP baseline OK; shape={shape}")


if __name__ == "__main__":
    main()
