"""Model names and pretrained-weight licences.

Dependency-free so config validation can import it without pulling in torch.
"""

from __future__ import annotations

import warnings

# Training architectures. The first entry of each pair is the default.
DETECTION_MODELS = frozenset({"dfine", "fasterrcnn"})
SEGMENTATION_MODELS = frozenset({"segformer", "lraspp"})

# Runtime perception backends selectable from config/edge.yaml.
DETECTION_BACKENDS = frozenset(
    {
        "checkpoint",
        "dfine",
        "ssdlite",
        "torchvision_ssdlite",
        "ssdlite320_mobilenet_v3_large",
        "fasterrcnn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
    }
)

SEGMENTATION_BACKENDS = frozenset({"checkpoint", "segformer", "lraspp", "lraspp_mobilenet_v3_large"})

OPTIMIZERS = frozenset({"adamw", "sgd"})

SCHEDULERS = frozenset({"cosine", "step", "none"})


# Licences declared on each model card. No weights ship with this repository, so its
# MIT licence is unaffected; this governs what a deployment may do with them.
PERMISSIVE_LICENCES = frozenset({"apache-2.0", "mit", "bsd-3-clause"})

# nvidia/* SegFormer cards declare `license: other` pointing at NVIDIA Source Code
# License-NC: https://github.com/NVlabs/SegFormer/blob/master/LICENSE
_NVIDIA_NC = "nvidia-source-code-license-nc"

PRETRAINED_WEIGHT_LICENCES: dict[str, str] = {
    **{
        f"ustc-community/dfine-{size}-coco": "apache-2.0"
        for size in ("nano", "small", "medium", "large", "xlarge")
    },
    **{
        f"nvidia/segformer-b{index}-finetuned-cityscapes-1024-1024": _NVIDIA_NC
        for index in range(6)
    },
    **{f"nvidia/mit-b{index}": _NVIDIA_NC for index in range(6)},
}


def weight_licence(model_id: str) -> str | None:
    """Declared licence for a known pretrained checkpoint, or None if unlisted."""
    return PRETRAINED_WEIGHT_LICENCES.get(model_id)


def warn_if_restricted_weights(model_id: str) -> str | None:
    """Warn for a checkpoint whose declared licence is known to be non-permissive."""
    licence = PRETRAINED_WEIGHT_LICENCES.get(model_id)
    if licence is None or licence in PERMISSIVE_LICENCES:
        return licence
    warnings.warn(
        f"Pretrained weights '{model_id}' are licensed '{licence}', which is not a "
        "permissive licence and restricts commercial use. The code in this repository "
        "is MIT and ships no weights; this constraint applies to your deployment. Use "
        "pretrained: false, or point pretrained_model at a checkpoint you are licensed "
        "to use.",
        UserWarning,
        stacklevel=3,
    )
    return licence
