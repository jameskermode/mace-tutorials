"""Small runtime helpers shared by the HPC-oriented tutorials."""

from __future__ import annotations


def choose_device(requested: str = "auto") -> str:
    """Return a MACE-compatible device without assuming a GPU is available."""
    if requested != "auto":
        return requested

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
