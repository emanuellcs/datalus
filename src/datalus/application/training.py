"""Training use case for DATALUS.

The application layer coordinates the training workflow but delegates all
framework-specific work to infrastructure adapters: Polars loaders, PyTorch
modules, and checkpoint persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from datalus.domain.schemas import TrainingConfig
from datalus.infrastructure.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
)
from datalus.infrastructure.encoding import TabularEncoder
from datalus.infrastructure.polars_loader import ChunkedParquetBatches
from datalus.infrastructure.torch_diffusion import TabularDiffusion
from datalus.infrastructure.torch_nn import EMA, FeatureProjector, TabularDenoiserMLP

logger = logging.getLogger(__name__)


class DatalusTrainer:
    """Training orchestrator with deterministic checkpointing and AMP."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(config.seed)
        self.schema_metadata = json.loads(
            Path(config.schema_path).read_text(encoding="utf-8")
        )
        self.batches = ChunkedParquetBatches(
            config.data_path, config.batch_size, config.seed
        )
        self.encoder = self._fit_encoder()
        self.encoder.save(self.output_dir / "encoder_config.json")
        self.projector = FeatureProjector(
            self.encoder.schema_metadata,
            self.encoder.numerical_columns,
            self.encoder.categorical_columns,
        ).to(self.device)
        self.denoiser = TabularDenoiserMLP(
            d_in=self.projector.total_latent_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.diffusion = TabularDiffusion(
            self.denoiser,
            num_timesteps=config.num_timesteps,
            condition_dropout=config.condition_dropout,
        ).to(self.device)
        self.optimizer = AdamW(
            list(self.diffusion.parameters()) + list(self.projector.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = max(
            1, config.epochs * max(1, len(self.batches.offsets_for_epoch(0, False)))
        )
        warmup = min(config.warmup_steps, max(1, total_steps - 1))
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup),
                CosineAnnealingLR(
                    self.optimizer, T_max=max(1, total_steps - warmup), eta_min=1e-6
                ),
            ],
            milestones=[warmup],
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.amp and self.device.type == "cuda"
        )
        self.ema = EMA(self.diffusion, decay=config.ema_decay)
        self.start_epoch = 0
        self.start_batch_index = 0
        self.global_step = 0
        self.loss_history: list[float] = []

    def _fit_encoder(self) -> TabularEncoder:
        sample = (
            pl.scan_parquet(self.config.data_path)
            .head(self.config.max_encoder_fit_rows)
            .collect()
        )
        encoder = TabularEncoder(self.schema_metadata)
        return encoder.fit(sample)

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
        self.diffusion.load_state_dict(checkpoint["diffusion_state"])
        self.projector.load_state_dict(checkpoint["projector_state"])
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
        self.diffusion.train()
        self.projector.train()
        for epoch in range(self.start_epoch, self.config.epochs):
            offsets = self.batches.offsets_for_epoch(epoch)
            batch_start = self.start_batch_index if epoch == self.start_epoch else 0
            for batch_index, offset in enumerate(
                offsets[batch_start:], start=batch_start
            ):
                loss_value = self._train_batch(self.batches.read_offset(offset))
                self.loss_history.append(loss_value)
                self.global_step += 1
                if self.global_step % self.config.checkpoint_every_steps == 0:
                    self.save_checkpoint(epoch, batch_index + 1, loss_value)
                if max_steps is not None and self.global_step >= max_steps:
                    return self.save_checkpoint(
                        epoch, batch_index + 1, loss_value, name="checkpoint_latest.pt"
                    )
            self.start_batch_index = 0
        return self.save_checkpoint(
            self.config.epochs, 0, self.loss_history[-1], name="checkpoint_latest.pt"
        )

    def _train_batch(self, frame: pl.DataFrame) -> float:
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
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            self.device.type, enabled=self.config.amp and self.device.type == "cuda"
        ):
            latent = self.projector(x_num, x_cat)
            loss = self.diffusion(latent)["loss"]
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.diffusion.parameters()) + list(self.projector.parameters()),
            self.config.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update(self.diffusion)
        return float(loss.detach().cpu().item())

    def save_checkpoint(
        self,
        epoch: int,
        batch_index: int,
        loss: float,
        name: str | None = None,
    ) -> Path:
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
            "diffusion_state": self.diffusion.state_dict(),
            "projector_state": self.projector.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "ema_state": self.ema.state_dict(),
            "rng_state": capture_rng_state(),
        }
        save_checkpoint(path, payload)
        latest = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest != path:
            save_checkpoint(latest, payload)
        return path


def _config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
