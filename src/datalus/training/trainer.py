"""Training orchestration for DATALUS.

The trainer coordinates the training loop but delegates framework-specific work
to the feature modules: Polars data loading, PyTorch networks, the diffusion
engine, and checkpoint persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from rich.progress import Progress
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from datalus._console import console as _shared_console
from datalus.config import TrainingConfig
from datalus.data.augmentation import FeatureAugmentation
from datalus.data.encoding import TabularEncoder
from datalus.data.loader import ChunkedParquetBatches
from datalus.models.diffusion import TabularDiffusion
from datalus.models.nn import EMA, FeatureProjector, build_denoiser
from datalus.training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
    update_best_checkpoint,
    validate_checkpoint_for_resume,
)

logger = logging.getLogger(__name__)


class DatalusTrainer:
    """Training orchestrator with deterministic checkpointing and AMP."""

    def __init__(self, config: TrainingConfig) -> None:
        """Build the trainer and prepare models, optimizer, and scheduler."""

        if config.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(config.seed)
        self.schema_metadata = json.loads(Path(config.schema_path).read_text(encoding="utf-8"))
        if config.target_column and config.target_column in self.schema_metadata:
            self.schema_metadata[config.target_column]["is_target"] = True
        self._augmentation = FeatureAugmentation(
            mode=config.feature_augmentation,
            random_state=config.seed,
        )
        self._base_encoder: TabularEncoder | None = None
        self.encoder = self._fit_encoder()
        self.encoder.save(self.output_dir / "encoder_config.json")
        data_path = self._materialize_augmented()
        self.batches = ChunkedParquetBatches(data_path, config.batch_size, config.seed)
        self.context_dim, self.target_is_categorical = self._resolve_context_dim()
        self.projector = FeatureProjector(
            self.encoder.schema_metadata,
            self.encoder.numerical_columns,
            self.encoder.categorical_columns,
        ).to(self.device)
        self.denoiser = build_denoiser(
            self.projector.total_latent_dim,
            self.projector.num_dim,
            self.projector.cat_dims,
            config.model_dump(mode="json"),
            context_dim=self.context_dim,
        ).to(self.device)
        self.diffusion = TabularDiffusion(
            self.denoiser,
            num_timesteps=config.num_timesteps,
            condition_dropout=config.condition_dropout,
            num_dim=self.projector.num_dim,
            cat_dims=self.projector.cat_dims,
            lambda_num=config.lambda_num,
            lambda_cat=config.lambda_cat,
        ).to(self.device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"Activating DataParallel on {torch.cuda.device_count()} GPUs.")
            self.projector = torch.nn.DataParallel(self.projector)
            self.diffusion = torch.nn.DataParallel(self.diffusion)

        self.optimizer = AdamW(
            list(self.diffusion.parameters()) + list(self.projector.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = max(1, config.epochs * max(1, len(self.batches.offsets_for_epoch(0, False))))
        warmup = min(config.warmup_steps, max(1, total_steps - 1))
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup),
                CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup), eta_min=1e-6),
            ],
            milestones=[warmup],
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.ema = EMA(getattr(self.diffusion, "module", self.diffusion), decay=config.ema_decay)
        self.start_epoch = 0
        self.start_batch_index = 0
        self.global_step = 0
        self.loss_history: list[float] = []
        self._last_checkpoint_bytes = 0

    def _encoder_policy(self) -> dict[str, Any]:
        """Return the TabFM-style encoding policy derived from the config."""

        return {
            "quantile_noise": self.config.quantile_noise,
            "rtdl_quantile_dynamic": self.config.rtdl_quantile_dynamic,
            "outlier_threshold": self.config.outlier_threshold,
            "numeric_standardize": self.config.numeric_standardize,
            "cat_encoder_mode": self.config.cat_encoder_mode,
            "min_cat_frequency": self.config.min_cat_frequency,
            "random_state": self.config.seed,
        }

    def _fit_encoder(self) -> TabularEncoder:
        """Fit a reversible encoder on a bounded sample of the training data."""

        sample = pl.scan_parquet(self.config.data_path).head(self.config.max_encoder_fit_rows).collect()
        policy = self._encoder_policy()
        if self.config.feature_augmentation == "none":
            self._base_encoder = None
            return TabularEncoder(self.schema_metadata, **policy).fit(sample)
        base_encoder = TabularEncoder(dict(self.schema_metadata), **policy).fit(sample)
        self._base_encoder = base_encoder
        self._augmentation.fit(base_encoder, sample)
        self.schema_metadata.update(
            FeatureAugmentation.schema_metadata_for(
                self.config.feature_augmentation,
                self._augmentation.column_names,
            )
        )
        augmented_sample = self._augmentation.transform_frame(sample, base_encoder)
        return TabularEncoder(self.schema_metadata, **policy).fit(augmented_sample)

    def _materialize_augmented(self) -> Path:
        """Write the training-time augmented Parquet dataset when enabled."""

        if self.config.feature_augmentation == "none" or self._base_encoder is None:
            return Path(self.config.data_path)
        output = self.output_dir / "augmented.parquet"
        lazy = pl.scan_parquet(self.config.data_path)
        num_rows = int(lazy.select(pl.len()).collect().item())
        chunk_size = self.config.batch_size
        chunks: list[pl.LazyFrame] = []
        for offset in range(0, num_rows, chunk_size):
            frame = lazy.slice(offset, chunk_size).collect()
            chunks.append(self._augmentation.transform_frame(frame, self._base_encoder).lazy())
        pl.concat(chunks).sink_parquet(output, compression="snappy")
        logger.info(
            f"Materialized augmented training data with {len(self._augmentation.column_names)} "
            f"columns to {output}."
        )
        return output

    def _resolve_context_dim(self) -> tuple[int | None, bool]:
        """Resolve the CFG context dimension from the optional target column."""

        target = self.config.target_column
        if target is None:
            return None, False
        if target in self.encoder.categorical_columns:
            return self.encoder.categorical_vocabs[target].size, True
        if target in self.encoder.numerical_columns:
            return 1, False
        logger.warning(
            f"Target column '{target}' is not retained by the encoder; conditioning disabled."
        )
        return None, False

    def _build_context(self, encoded: Any) -> torch.Tensor | None:
        """Build the per-batch CFG context vector from the target column."""

        if self.context_dim is None:
            return None
        n_rows = len(encoded.x_cat) if encoded.x_cat is not None else len(encoded.x_num)
        if self.target_is_categorical:
            column_idx = self.encoder.categorical_columns.index(self.config.target_column)
            one_hot = np.zeros((n_rows, self.context_dim), dtype=np.float32)
            targets = encoded.x_cat[:, column_idx].astype(np.int64)
            one_hot[np.arange(n_rows), targets] = 1.0
            return torch.from_numpy(one_hot).to(self.device)
        column_idx = self.encoder.numerical_columns.index(self.config.target_column)
        return torch.from_numpy(encoded.x_num[:, column_idx]).to(self.device).unsqueeze(-1)

    def resume(self, checkpoint_path: str | Path) -> None:
        """Restore model, optimizer, scheduler, and RNG state from a checkpoint."""

        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
        serialized = self.config.model_dump(mode="json")
        serialized["config_hash"] = _config_hash(serialized)
        validate_checkpoint_for_resume(checkpoint, serialized)
        if int(checkpoint["global_step"]) < self.global_step:
            raise ValueError(
                f"Checkpoint global_step ({checkpoint['global_step']}) is behind "
                f"the current step ({self.global_step})."
            )
        getattr(self.diffusion, "module", self.diffusion).load_state_dict(checkpoint["diffusion_state"])
        getattr(self.projector, "module", self.projector).load_state_dict(checkpoint["projector_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.ema.load_state_dict(checkpoint["ema_state"])
        self.start_epoch = int(checkpoint["epoch"])
        self.start_batch_index = int(checkpoint["batch_index"])
        self.global_step = int(checkpoint["global_step"])
        self.loss_history = list(checkpoint.get("loss_history", []))
        restore_rng_state(checkpoint["rng_state"])

    def train(self, max_steps: int | None = None) -> Path:
        """Run the full training loop and return the final checkpoint path."""

        self.diffusion.train()
        self.projector.train()

        # Detect an interactive terminal for live progress rendering.
        show_progress = sys.stdout.isatty()

        with Progress(console=_shared_console, disable=not show_progress) as progress:
            epoch_task = progress.add_task(
                f"[bold cyan]Training[/bold cyan] {self.config.epochs} epochs",
                total=self.config.epochs,
            )

            for epoch in range(self.start_epoch, self.config.epochs):
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs} (global_step={self.global_step})")
                offsets = self.batches.offsets_for_epoch(epoch)
                batch_start = self.start_batch_index if epoch == self.start_epoch else 0

                batch_task = progress.add_task(
                    "  [cyan]Batches[/cyan]",
                    total=len(offsets),
                    visible=show_progress,
                )

                epoch_ran = False
                for batch_index, offset in enumerate(offsets[batch_start:], start=batch_start):
                    epoch_ran = True
                    loss_value = self._train_batch(self.batches.read_offset(offset))
                    self.loss_history.append(loss_value)
                    self.global_step += 1

                    logger.debug(
                        f"Step {self.global_step}: loss={loss_value:.6f}, "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

                    # Step-cadence saves run inside the loop; the epoch
                    # cadence fires once at each epoch boundary below.
                    step_cadence = (
                        self.config.checkpoint_every_steps > 0
                        and self.global_step % self.config.checkpoint_every_steps == 0
                    )
                    if step_cadence:
                        self.save_checkpoint(epoch, batch_index + 1, loss_value)
                        logger.info(f"Checkpoint saved at step {self.global_step}, loss={loss_value:.6f}")

                    if max_steps is not None and self.global_step >= max_steps:
                        logger.info(f"Reached max_steps={max_steps}. Stopping training.")
                        return self.save_checkpoint(epoch, batch_index + 1, loss_value)

                    if show_progress:
                        progress.update(batch_task, advance=1)

                if epoch_ran and (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch, batch_index + 1, loss_value)
                    logger.info(f"Epoch checkpoint saved at step {self.global_step}, loss={loss_value:.6f}")

                if show_progress:
                    progress.update(batch_task, visible=False)
                    progress.update(epoch_task, advance=1)

            self.start_batch_index = 0

        if not self.loss_history:
            raise RuntimeError(
                "No training batches were processed. Check that the dataset is "
                "not empty and that the batch size does not exceed the row count."
            )

        logger.info(
            f"Training complete. Final loss: {self.loss_history[-1]:.6f}. Total steps: {self.global_step}"
        )
        return self.save_checkpoint(
            self.config.epochs,
            0,
            self.loss_history[-1],
        )

    def _train_batch(self, frame: pl.DataFrame) -> float:
        """Train on one batch and return the detached scalar loss."""

        encoded = self.encoder.transform(frame)
        x_num = (
            torch.from_numpy(encoded.x_num).to(self.device, non_blocking=True)
            if encoded.x_num is not None
            else None
        )
        x_cat = (
            torch.from_numpy(encoded.x_cat).to(self.device, non_blocking=True)
            if encoded.x_cat is not None
            else None
        )
        context = self._build_context(encoded)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.device.type, enabled=self.config.amp and self.device.type == "cuda"):
            latent = self.projector(x_num, x_cat)
            loss = self.diffusion(latent, x_cat, context)["loss"].mean()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.diffusion.parameters()) + list(self.projector.parameters()),
            self.config.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update(getattr(self.diffusion, "module", self.diffusion))
        return float(loss.detach().cpu().item())

    def save_checkpoint(
        self,
        epoch: int,
        batch_index: int,
        loss: float,
        name: str | None = None,
    ) -> Path:
        """Write a checkpoint and maintain latest, best, and retention files."""

        self._assert_free_space()
        checkpoint_name = name or f"checkpoint_step_{self.global_step:08d}.pt"
        path = self.checkpoint_dir / checkpoint_name
        payload = {
            "epoch": epoch,
            "batch_index": batch_index,
            "global_step": self.global_step,
            "loss": loss,
            "loss_history": self.loss_history,
            "config": self.config.model_dump(mode="json"),
            "config_hash": _config_hash(self.config.model_dump(mode="json")),
            "diffusion_state": getattr(self.diffusion, "module", self.diffusion).state_dict(),
            "projector_state": getattr(self.projector, "module", self.projector).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "ema_state": self.ema.state_dict(),
            "rng_state": capture_rng_state(),
        }
        save_checkpoint(path, payload)
        self._last_checkpoint_bytes = path.stat().st_size

        latest = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest != path and not self._publish_latest(path, latest):
            save_checkpoint(latest, payload)

        if self.config.save_strategy == "best":
            update_best_checkpoint(self.checkpoint_dir, loss, path)

        if self.config.keep_last > 0:
            prune_checkpoints(self.checkpoint_dir, self.config.keep_last)

        return path

    @staticmethod
    def _publish_latest(step_path: Path, latest: Path) -> bool:
        """Atomically symlink checkpoint_latest.pt to the newest step file."""

        temp = latest.with_name(latest.name + ".tmp")
        try:
            temp.unlink(missing_ok=True)
            os.symlink(step_path.name, temp)
            os.replace(temp, latest)
            return True
        except OSError:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _assert_free_space(self) -> None:
        """Warn or refuse checkpoint writes when the disk is low on space."""

        threshold = self.config.min_free_space_gb
        if threshold <= 0:
            return
        free_bytes = shutil.disk_usage(self.checkpoint_dir).free
        min_free_bytes = threshold * 1024**3
        critical_bytes = max(min_free_bytes * 0.25, self._last_checkpoint_bytes * 1.5)
        free_gib = free_bytes / 1024**3
        if free_bytes < critical_bytes:
            raise RuntimeError(
                f"Insufficient free disk space ({free_gib:.2f} GiB) to safely "
                "write a checkpoint. Free space or use --keep-last N to rotate "
                "checkpoints."
            )
        if free_bytes < min_free_bytes:
            logger.warning(
                f"Free disk space ({free_gib:.2f} GiB) is below "
                f"--min-free-space ({threshold:.1f} GiB). Use --keep-last N to "
                "rotate checkpoints and avoid running out of space."
            )


def _config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash of the serialized training config."""

    serialized = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
