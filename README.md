# 5G-V2X Edge System

Python implementation of a roadside V2X perception system that detects and tracks road users, projects them into road coordinates, estimates collision risk, and sends scene information to nearby vehicles.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

## Features

- Roadside object detection with D-FINE or Faster R-CNN MobileNetV3
- Multi-object tracking with Kalman filtering and Hungarian matching
- Camera-to-road localization using homography
- Optional camera-radar late fusion
- Collision and vulnerable road user risk estimation
- Semantic segmentation with SegFormer or LR-ASPP MobileNetV3
- Versioned V2X JSON messages over UDP
- Virtual OBU receiver for local testing
- Training, evaluation, checkpointing, and ONNX export utilities

Physical PC5/C-V2X transmission is vendor-specific. `VendorRSUTransport` provides the integration point for a supplier SDK.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Requirements: Python 3.10+ and PyTorch 2.4+. Training requires a supported GPU for practical performance; the basic examples and tests can run on CPU.

## Usage

Run the synthetic example:

```bash
python scripts/run_synthetic_demo.py
```

Calibrate a fixed roadside camera:

```bash
python scripts/calibrate_homography.py --points examples/homography_points.json
```

Start a virtual OBU:

```bash
python scripts/run_virtual_obu.py --port 5005
```

Run roadside perception from a video or camera:

```bash
python scripts/run_video_edge.py --source clip.mp4
python scripts/run_video_edge.py --source 0
```

Use at least four non-collinear calibration points distributed across the visible road area. A planar homography assumes an approximately flat road surface.

## Training

Detector example:

```bash
python tools/inspect_dataset.py detector --config config/detector.yaml
python tools/train_detector.py --config config/detector.yaml --set schedule.epochs=40
python tools/evaluate.py detector --config config/detector.yaml \
    --checkpoint outputs/detector/checkpoints/best.pt
```

Segmenter training uses the corresponding `segmenter` configuration and tools.

Training outputs are stored under `outputs/<name>/`, including the resolved configuration, metrics, and `last.pt` and `best.pt` checkpoints.

### Detection data

JSONL with one sample per line. Bounding boxes use `xyxy` pixel coordinates and label `0` is reserved for background.

```json
{"image":"images/000001.jpg","boxes":[[110,140,280,310]],"labels":[1]}
```

### Segmentation data

Place images and masks in `images/` and `masks/` with matching filename stems. Mask pixels contain class IDs; `ignore_index` defaults to `255`.

For video datasets, use recording-level train and validation splits to avoid leakage between adjacent frames.

## Models

| Task | Default | Alternative |
| --- | --- | --- |
| Detection | D-FINE | Faster R-CNN MobileNetV3 |
| Segmentation | SegFormer | LR-ASPP MobileNetV3 |

Pretrained weights are downloaded when requested and are not included in this repository. Check the license of each pretrained model before deployment. D-FINE weights are Apache-2.0; NVIDIA SegFormer checkpoints may have non-commercial restrictions. Set `pretrained: false` when appropriate.

## Configuration

- `config/edge.yaml` controls runtime perception, tracking, localization, risk estimation, and V2X communication.
- `config/detector.yaml` controls detector training.
- `config/segmenter.yaml` controls segmenter training.

Configuration files are validated at startup and reject unsupported keys or invalid values.

## Project structure

```text
config/          runtime and training configuration
scripts/         calibration, demos, video execution, virtual OBU
tools/           training, evaluation, dataset inspection, ONNX export
test/            automated tests
v2x_edge/
  data/          datasets and transforms
  models/        detection and segmentation models
  tracking/      multi-object tracking
  localization/  homography and coordinate transforms
  fusion/        camera-radar fusion
  world/         scene state
  safety/        collision and VRU risk
  v2x/           messages, codec, and transport
  obu/           virtual OBU
  edge/          runtime processing
```

## V2X messages

The system uses a versioned JSON message containing RSU information, tracked objects, and detected risks. The codec validates schema versions, numeric values, and field ranges. The virtual OBU rejects stale, duplicate, and out-of-order messages.

UDP messages must fit within a single datagram. The JSON format is an application-level representation, not a PC5 wire format.

## License

MIT. See [LICENSE](LICENSE).
