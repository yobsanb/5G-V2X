"""Roadside V2X cooperative perception, training, and safety analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth is pyproject.toml; nothing here to drift out of sync.
    __version__ = version("v2x-edge-system")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+source"

__all__ = ["__version__"]
