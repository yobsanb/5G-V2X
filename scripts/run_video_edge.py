#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from v2x_edge.config import load_config
from v2x_edge.edge import EdgePipeline, OpenCVVideoSource
from v2x_edge.localization import HomographyProjector, TrackToWorldProjector
from v2x_edge.models import build_detector_from_config, build_segmenter_from_config
from v2x_edge.safety import RiskEngine
from v2x_edge.tracking import MultiObjectTracker
from v2x_edge.utils import set_seed
from v2x_edge.v2x import UdpSender
from v2x_edge.visualization import draw_tracks
from v2x_edge.world import WorldModel


def parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run roadside video perception and V2X transmission")
    parser.add_argument("--config", default="config/edge.yaml")
    parser.add_argument("--source", default=None, help="Override camera index or video path")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-send", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Pace video files using their media timestamps")
    args = parser.parse_args()

    config = load_config(args.config)
    homography_path = Path(config["localization"]["homography_path"])
    if not homography_path.exists():
        raise FileNotFoundError(
            f"Missing {homography_path}. Run scripts/calibrate_homography.py for the fixed camera first."
        )

    set_seed(int(config["system"].get("seed", 42)))
    device = config["system"].get("device", "auto")
    detector = build_detector_from_config(config["perception"]["detector"], device=device)
    segmenter = build_segmenter_from_config(config["perception"].get("segmentation", {}), device=device)
    tracking = config["tracking"]
    tracker = MultiObjectTracker(tracking["iou_threshold"], tracking["max_age"], tracking["min_hits"])
    projector = TrackToWorldProjector(HomographyProjector.from_file(homography_path))
    world_model = WorldModel(config["world"]["stale_after_seconds"])
    risk_engine = RiskEngine(**config["risk"])
    transport = None if args.no_send else UdpSender(config["v2x"]["host"], config["v2x"]["port"])
    pipeline = EdgePipeline(
        rsu_id=config["v2x"]["rsu_id"],
        detector=detector,
        tracker=tracker,
        projector=projector,
        world_model=world_model,
        risk_engine=risk_engine,
        segmenter=segmenter,
        transport=transport,
    )

    source_value = args.source if args.source is not None else config["camera"]["source"]
    if isinstance(source_value, str):
        source_value = parse_source(source_value)
    source = OpenCVVideoSource(
        source_value,
        config["camera"]["width"],
        config["camera"]["height"],
        config["camera"]["fps"],
        realtime=args.realtime or not args.no_send,
    )

    try:
        while True:
            ok, frame, timestamp = source.read()
            if not ok or timestamp is None:
                break
            result = pipeline.process_frame(frame, timestamp=timestamp)
            print(
                f"det={len(result.detections)} tracks={len(result.tracks)} "
                f"objects={len(result.objects)} risks={len(result.risks)}"
            )
            if not args.no_display:
                view = draw_tracks(frame, result.tracks)
                cv2.imshow("V2X Edge", view)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
    finally:
        source.close()
        pipeline.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
