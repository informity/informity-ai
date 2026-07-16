"""Module for utils torch utils."""

from __future__ import annotations

_MPS_DETECTION_EXCEPTIONS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    TypeError,
    ValueError,
)


def is_mps_available() -> bool:
    """Is mps available."""
    try:
        import torch

        return torch.backends.mps.is_available()
    except _MPS_DETECTION_EXCEPTIONS:
        return False
