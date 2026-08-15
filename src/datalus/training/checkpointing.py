"""Deterministic checkpoint save and restore utilities.

Writes atomically and loads with torch's secure ``weights_only`` unpickler;
portable RNG state keeps the allowlist empty.
"""

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


def _portable_rng_value(value: Any) -> Any:
    """Store numpy arrays as tagged plain lists so ``weights_only`` needs no allowlist."""

    if isinstance(value, np.ndarray):
        return {
            "__datalus_ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.ravel().astype(value.dtype).tolist(),
        }
    if isinstance(value, tuple):
        return tuple(_portable_rng_value(item) for item in value)
    if isinstance(value, list):
        return [_portable_rng_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _portable_rng_value(item) for key, item in value.items()}
    return value


def _restore_rng_value(value: Any) -> Any:
    """Reverse ``_portable_rng_value``, rebuilding numpy arrays."""

    if isinstance(value, dict) and value.get("__datalus_ndarray__"):
        array = np.array(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(value["shape"])
    if isinstance(value, tuple):
        return tuple(_restore_rng_value(item) for item in value)
    if isinstance(value, list):
        return [_restore_rng_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_rng_value(item) for key, item in value.items()}
    return value


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch, and CUDA RNG states portably."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return _portable_rng_value(state)


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by `capture_rng_state`."""

    state = _restore_rng_value(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_all"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a checkpoint atomically and clean up stale temporary files."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_tmp_files(output.parent)
    temp = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(payload, temp)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def _cleanup_stale_tmp_files(directory: Path) -> None:
    """Remove leftover temporary files from interrupted atomic saves."""

    for stale in directory.glob("*.pt.tmp"):
        try:
            stale.unlink()
            logger.debug("Removed stale checkpoint temp file: %s", stale.name)
        except OSError:
            pass


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint with the secure weights-only unpickler.

    Loudly falls back to the legacy unpickler for pre-portable checkpoints.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as secure_exc:  # noqa: BLE001 - legacy-format detection
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as legacy_exc:
            raise RuntimeError(
                f"Checkpoint {path} could not be loaded: it is corrupt or not a "
                f"DATALUS checkpoint (secure load: {secure_exc}; legacy load: {legacy_exc})."
            ) from legacy_exc
        logger.warning(
            "Checkpoint %s is not in the secure weights-only format (%s). "
            "Re-saving the checkpoint will upgrade it; loading legacy files "
            "permits arbitrary code execution and should be limited to trusted "
            "sources.",
            path,
            secure_exc,
        )
        return payload


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

    # Sort by the step number in the filename so retention survives mtime ties.
    checkpoint_files = [
        f for f in checkpoint_dir.glob("checkpoint_step_*.pt") if f.name not in preserve_files
    ]
    checkpoint_files.sort(key=_checkpoint_step)

    if len(checkpoint_files) > keep_last:
        for old_checkpoint in checkpoint_files[:-keep_last]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Deleted old checkpoint: {old_checkpoint.name}")
            except OSError as e:
                logger.warning(f"Failed to delete {old_checkpoint.name}: {e}")


def _atomic_copy_checkpoint(source: str | Path, destination: Path) -> None:
    """Copy a checkpoint atomically via a temporary file and rename."""

    temp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        shutil.copy2(source, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


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

    if best_path.exists():
        try:
            best_checkpoint = load_checkpoint(best_path, map_location="cpu")
            best_loss = best_checkpoint.get("loss", float("inf"))
        except Exception as e:  # noqa: BLE001 - any load failure resets best loss
            logger.warning(f"Failed to load best checkpoint: {e}. Resetting.")
            best_loss = float("inf")
    else:
        best_loss = float("inf")

    if current_loss < best_loss:
        try:
            _atomic_copy_checkpoint(current_checkpoint_path, best_path)
            logger.info(f"New best checkpoint: loss={current_loss:.6f}")
        except OSError as e:
            logger.warning(f"Failed to update best checkpoint: {e}")


_ARCHITECTURE_KEYS = (
    "denoiser_type",
    "hidden_dims",
    "transformer_d_model",
    "transformer_blocks",
    "transformer_heads",
    "transformer_ff_factor",
    "transformer_num_inds",
    "transformer_num_cls",
    "transformer_rope_base",
    "transformer_ffn_chunk_size",
    "transformer_row_chunk_size",
    "num_timesteps",
    "lambda_num",
    "lambda_cat",
    "target_column",
    "feature_augmentation",
    "quantile_noise",
    "rtdl_quantile_dynamic",
    "outlier_threshold",
    "numeric_standardize",
    "cat_encoder_mode",
    "min_cat_frequency",
)

_REQUIRED_CHECKPOINT_KEYS = {
    "diffusion_state",
    "projector_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "ema_state",
    "rng_state",
    "epoch",
    "batch_index",
    "global_step",
    "config",
    "config_hash",
}


def validate_checkpoint_for_resume(checkpoint: dict[str, Any], current_config: dict[str, Any]) -> None:
    """Validate a checkpoint for resume, raising on missing keys or architecture changes.

    Legacy checkpoints missing new keys stay resumable; a config-hash
    difference is only a warning since run hyperparameters may change.
    """

    missing = _REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing)}. It may be corrupt or from an incompatible version."
        )
    saved_config = checkpoint.get("config", {})
    for key in _ARCHITECTURE_KEYS:
        if key not in saved_config:
            continue
        if saved_config.get(key) != current_config.get(key):
            raise ValueError(
                f"Checkpoint architecture setting '{key}' differs "
                f"({saved_config.get(key)!r} vs {current_config.get(key)!r}). "
                "Resume with the exact TrainingConfig that produced the checkpoint."
            )
    if checkpoint.get("config_hash") != current_config.get("config_hash"):
        logger.warning(
            "Checkpoint config hash differs from the current config. Model "
            "identity was verified, but run hyperparameters may have changed."
        )


def _checkpoint_step(path: Path) -> tuple[int, Path]:
    """Return a sort key from the step number in a checkpoint filename.

    Unrecognized names sort first by name so unrelated files still order
    deterministically.
    """

    step = 0
    name = path.name
    if name.startswith("checkpoint_step_") and name.endswith(".pt"):
        try:
            step = int(name[len("checkpoint_step_") : -len(".pt")])
        except ValueError:
            step = 0
    if step == 0:
        return step, Path(name)
    return step, Path(f"{step:08d}")
