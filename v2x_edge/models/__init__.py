"""Model definitions and the name -> builder registry used by configs.

Defaults are D-FINE and SegFormer; the torchvision CNNs remain as baselines.
"""

from .detection import CheckpointDetector, TorchvisionCocoDetector, TrainableFasterRCNN
from .dfine import DEFAULT_DFINE_MODEL, DFineDetector, TrainableDFine, build_dfine_from_checkpoint
from .factory import (
    build_detection_model,
    build_detector_from_config,
    build_segmentation_model,
    build_segmenter_from_config,
    checkpoint_meta,
)
from .segformer import (
    DEFAULT_SEGFORMER_MODEL,
    SegFormerSegmenter,
    TrainableSegFormer,
    build_segformer_from_checkpoint,
)
from .segmentation import CheckpointSegmenter, LRASPPSegmenter, TrainableLRASPP

__all__ = [
    # detection
    "TrainableDFine",
    "DFineDetector",
    "DEFAULT_DFINE_MODEL",
    "build_dfine_from_checkpoint",
    "TrainableFasterRCNN",
    "TorchvisionCocoDetector",
    "CheckpointDetector",
    # segmentation
    "TrainableSegFormer",
    "SegFormerSegmenter",
    "DEFAULT_SEGFORMER_MODEL",
    "build_segformer_from_checkpoint",
    "TrainableLRASPP",
    "LRASPPSegmenter",
    "CheckpointSegmenter",
    # builders
    "build_detection_model",
    "build_segmentation_model",
    "build_detector_from_config",
    "build_segmenter_from_config",
    "checkpoint_meta",
]
