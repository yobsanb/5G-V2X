#!/usr/bin/env python3
"""Train the roadside object detector from a YAML config."""

from __future__ import annotations

import argparse
import json

from v2x_edge.config import apply_overrides, load_config, validate_detection_train_config
from v2x_edge.train import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/detector.yaml")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config entries, e.g. --set schedule.epochs=5 optim.lr=1e-4",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, schema="detection_train")
    if args.set:
        validate_detection_train_config(apply_overrides(cfg, args.set))
    print(json.dumps(train_detector(cfg), indent=2))


if __name__ == "__main__":
    main()
