"""Deterministic checkpoint save and restore utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all supported RNGs for reproducible training and sampling."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch, and CUDA RNG states."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by `capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_all"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a checkpoint atomically where possible."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(output)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a DATALUS training checkpoint."""

    return torch.load(path, map_location=map_location, weights_only=False)
