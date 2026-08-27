#!/usr/bin/env python3
"""Export a trained segmentation checkpoint to ONNX (detection export is not offered)."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

from v2x_edge.models import TrainableLRASPP, build_segformer_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/segmenter.onnx")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    if args.height < 1 or args.width < 1:
        raise ValueError("height and width must be positive")

    try:
        __import__("onnx")
    except ImportError as exc:
        raise RuntimeError("ONNX export requires the optional 'export' dependencies") from exc

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "num_classes" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Invalid segmenter checkpoint format")
    model_type = str(checkpoint.get("model_type", ""))
    if model_type == "segformer":
        model = build_segformer_from_checkpoint(checkpoint)
    elif model_type in {"", "lraspp_mobilenet_v3_large"}:
        model = TrainableLRASPP(int(checkpoint["num_classes"]), pretrained_backbone=False)
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise ValueError(f"Cannot export model_type '{model_type}' to ONNX")
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": ["image"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        "opset_version": args.opset,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, torch.rand(1, 3, args.height, args.width), str(output), **export_kwargs)
    print(f"saved {output} ({model_type or 'lraspp_mobilenet_v3_large'})")


if __name__ == "__main__":
    main()
