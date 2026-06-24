"""Deterministic checkpoint save and restore utilities."""

from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


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


def prune_checkpoints(
    checkpoint_dir: Path | str,
    keep_last: int,
    preserve_files: list[str] | None = None,
) -> None:
    """Delete oldest checkpoints, keeping only the most recent `keep_last` ones.

    Preserves special files like 'checkpoint_latest.pt' and 'checkpoint_best.pt'
    outside the rotation count.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of regular checkpoints to retain
        preserve_files: Files to exclude from pruning (default: latest, best)
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    if preserve_files is None:
        preserve_files = ["checkpoint_latest.pt", "checkpoint_best.pt"]

    # Find all step-based checkpoint files, excluding preserved files
    checkpoint_files = sorted(
        [
            f
            for f in checkpoint_dir.glob("checkpoint_step_*.pt")
            if f.name not in preserve_files
        ],
        key=lambda f: f.stat().st_mtime,  # Sort by modification time (oldest first)
    )

    # Delete oldest files if we exceed keep_last
    if len(checkpoint_files) > keep_last:
        for old_checkpoint in checkpoint_files[:-keep_last]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Deleted old checkpoint: {old_checkpoint.name}")
            except OSError as e:
                logger.warning(f"Failed to delete {old_checkpoint.name}: {e}")


def update_best_checkpoint(
    checkpoint_dir: Path | str,
    current_loss: float,
    current_checkpoint_path: Path | str,
    best_checkpoint_name: str = "checkpoint_best.pt",
) -> None:
    """Update the 'best' checkpoint if current loss is lower than previous best.

    Args:
        checkpoint_dir: Directory containing checkpoints
        current_loss: Loss value of current checkpoint
        current_checkpoint_path: Path to the current checkpoint file
        best_checkpoint_name: Name of best checkpoint file to maintain
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / best_checkpoint_name

    # Load best checkpoint if it exists, extract its loss
    if best_path.exists():
        try:
            best_checkpoint = load_checkpoint(best_path, map_location="cpu")
            best_loss = best_checkpoint.get("loss", float("inf"))
        except Exception as e:
            logger.warning(f"Failed to load best checkpoint: {e}. Resetting.")
            best_loss = float("inf")
    else:
        best_loss = float("inf")

    # Update best if current is better
    if current_loss < best_loss:
        try:
            shutil.copy2(current_checkpoint_path, best_path)
            logger.info(f"New best checkpoint: loss={current_loss:.6f}")
        except OSError as e:
            logger.warning(f"Failed to update best checkpoint: {e}")
