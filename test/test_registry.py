"""Pretrained-weight licence gate: restricted weights must never load silently."""

import warnings

import pytest

from v2x_edge.models import DEFAULT_DFINE_MODEL, DEFAULT_SEGFORMER_MODEL
from v2x_edge.registry import (
    PERMISSIVE_LICENCES,
    PRETRAINED_WEIGHT_LICENCES,
    warn_if_restricted_weights,
    weight_licence,
)


def test_default_detection_weights_are_permissive():
    assert weight_licence(DEFAULT_DFINE_MODEL) in PERMISSIVE_LICENCES


def test_default_segmentation_weights_are_known_and_restricted():
    # Deliberate: the Cityscapes init is worth it, but must never be silent.
    licence = weight_licence(DEFAULT_SEGFORMER_MODEL)
    assert licence is not None
    assert licence not in PERMISSIVE_LICENCES


def test_restricted_weights_warn():
    with pytest.warns(UserWarning, match="not a permissive licence"):
        warn_if_restricted_weights(DEFAULT_SEGFORMER_MODEL)


def test_permissive_weights_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_if_restricted_weights(DEFAULT_DFINE_MODEL) in PERMISSIVE_LICENCES


def test_unlisted_weights_are_silent_and_report_unknown():
    # We cannot vouch for a checkpoint we have not catalogued; say so rather than guess.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_if_restricted_weights("some-org/private-finetune") is None
    assert weight_licence("some-org/private-finetune") is None


@pytest.mark.parametrize("model_id", sorted(PRETRAINED_WEIGHT_LICENCES))
def test_every_catalogued_licence_is_a_nonempty_string(model_id):
    licence = PRETRAINED_WEIGHT_LICENCES[model_id]
    assert isinstance(licence, str) and licence.strip()
