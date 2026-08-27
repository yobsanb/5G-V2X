#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v2x_edge.localization import HomographyProjector


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate an image-to-road homography")
    parser.add_argument("--points", required=True, help="JSON with image_points and world_points")
    parser.add_argument("--output", default="calibration/homography.npy")
    parser.add_argument("--ransac-threshold", type=float, default=0.5, help="RANSAC error threshold in road units")
    args = parser.parse_args()

    data = json.loads(Path(args.points).read_text(encoding="utf-8"))
    image_points = np.asarray(data["image_points"], dtype=np.float64)
    world_points = np.asarray(data["world_points"], dtype=np.float64)
    projector, mask = HomographyProjector.estimate(
        image_points,
        world_points,
        ransac_reproj_threshold=args.ransac_threshold,
    )
    projector.save(args.output)
    inliers = int(mask.sum()) if mask is not None else len(image_points)
    print(f"saved {args.output}; inliers={inliers}/{len(image_points)}")
    print(projector.H)


if __name__ == "__main__":
    main()
