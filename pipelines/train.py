"""
DATALUS MLOps Pipeline
Layer: Training Orchestrator & Deterministic Checkpointing
"""

import os
import time
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import polars as pl

# Importação dos módulos do núcleo matemático (Camada 2 e 3)
# Assumindo execução a partir da raiz do monorepo
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from nn.embeddings import FeatureProjector
from nn.mlp import TabularDenoiserMLP
from diffusion.tab_ddpm import TabularDiffusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Trainer")


class DatalusTrainer:
    """
    Robust training orchestrator with Automatic Mixed Precision (AMP) 
    and deterministic checkpointing for ephemeral cloud environments.
    """
    def __init__(
        self,
        schema_path: str,
        data_path: str,
        output_dir: str,
        batch_size: int = 4096,
        epochs: int = 1000,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        checkpoint_freq: int = 50,
        resume_from: Optional[str] = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DATALUS Trainer on device: {self.device}")
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.checkpoint_freq = checkpoint_freq
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # 1. Load Schema and Data
        self._load_infrastructure(schema_path, data_path)
        
        # 2. Build Neural Architecture
        self._build_models()
        
        # 3. Setup MLOps (Optimizer, Scheduler, AMP Scaler)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        
        self.start_epoch = 0
        self.loss_history = []
        
        # 4. Resume State (If running on Spot Instances/Colab)
        if resume_from:
            self._load_checkpoint(resume_from)

    def _seed_everything(self, seed: int = 42) -> None:
        """Ensures reproducibility across interrupted sessions."""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_infrastructure(self, schema_path: str, data_path: str) -> None:
        """Loads metadata schema and initializes lazy data loaders."""
        logger.info("Loading schema and preparing memory-efficient tensors...")
        
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema_metadata = json.load(f)
            
        # Filter schema to remove dropped columns
        self.active_schema = {
            col: meta for col, meta in self.schema_metadata.items() 
            if meta["encoding_strategy"] != "DROP" and not meta["is_target"]
        }
        
        # Read Parquet using Polars (eager evaluation here as we need tensors in RAM for training)
        # For massive datasets (>RAM), this should be replaced with a streaming Dataset class.
        df = pl.read_parquet(data_path)
        
        num_cols = [col for col, meta in self.active_schema.items() if "NUMERICAL" in meta["inferred_topology"]]
        cat_cols = [col for col, meta in self.active_schema.items() if "CATEGORICAL" in meta["inferred_topology"] or meta["inferred_topology"] == "BOOLEAN"]
        
        x_num = torch.tensor(df.select(num_cols).to_numpy(), dtype=torch.float32) if num_cols else None
        x_cat = torch.tensor(df.select(cat_cols).to_numpy(), dtype=torch.long) if cat_cols else None
        
        # If dataset is purely categorical or purely numerical, handle None
        tensors = []
        if x_num is not None: tensors.append(x_num)
        if x_cat is not None: tensors.append(x_cat)
            
        dataset = TensorDataset(*tensors)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        logger.info(f"DataLoader ready. Batches per epoch: {len(self.dataloader)}")

    def _build_models(self) -> None:
        """Instantiates the Latent Projector, the MLP, and the Diffusion Engine."""
        logger.info("Building Model Architecture...")
        
        # Layer 2: Heterogeneous Embeddings
        self.projector = FeatureProjector(self.active_schema).to(self.device)
        latent_dim = self.projector.total_latent_dim
        logger.info(f"Total Continuous Latent Dimension: {latent_dim}")
        
        # Layer 3: Denoiser and Markov Chain
        self.denoiser = TabularDenoiserMLP(d_in=latent_dim, hidden_dims=(512, 1024, 1024, 512)).to(self.device)
        self.model = TabularDiffusion(denoiser=self.denoiser, num_timesteps=1000).to(self.device)

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Saves a fully deterministic snapshot to disk."""
        checkpoint_path = self.checkpoints_dir / f"datalus_ckpt_epoch_{epoch:04d}.pt"
        
        state = {
            "epoch": epoch,
            "loss": loss,
            "loss_history": self.loss_history,
            "model_state": self.model.state_dict(),
            "projector_state": self.projector.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_python": random.getstate()
        }
        
        if self.device.type == "cuda":
            state["rng_state_cuda"] = torch.cuda.get_rng_state()
            
        torch.save(state, checkpoint_path)
        logger.info(f"Deterministic checkpoint saved: {checkpoint_path.name}")
        
        # Keep only the last 3 checkpoints to save disk space
        checkpoints = sorted(self.checkpoints_dir.glob("*.pt"))
        for ckpt in checkpoints[:-3]:
            ckpt.unlink()

    def _load_checkpoint(self, path: str) -> None:
        """Restores training exactly from where it was interrupted."""
        logger.info(f"Resuming training from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.projector.load_state_dict(checkpoint["projector_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.loss_history = checkpoint.get("loss_history", [])
        
        # Restore deterministic RNG states
        torch.set_rng_state(checkpoint["rng_state_torch"])
        np.random.set_state(checkpoint["rng_state_numpy"])
        random.setstate(checkpoint["rng_state_python"])
        if "rng_state_cuda" in checkpoint and self.device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["rng_state_cuda"])

    def train(self) -> None:
        """The main training loop with Automatic Mixed Precision."""
        logger.info(f"Starting training from epoch {self.start_epoch} to {self.epochs}")
        self.model.train()
        self.projector.train()
        
        # Pre-allocate variables to avoid tracking None across iterations
        has_num = any("NUMERICAL" in meta["inferred_topology"] for meta in self.active_schema.values())
        has_cat = any("CATEGORICAL" in meta["inferred_topology"] or meta["inferred_topology"] == "BOOLEAN" for meta in self.active_schema.values())
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss = 0.0
            
            for batch in self.dataloader:
                x_num = batch[0].to(self.device) if has_num else None
                x_cat = batch[1].to(self.device) if has_cat and len(batch) > 1 else (batch[0].to(self.device) if has_cat else None)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with AMP for 2x speedup and 0.5x VRAM
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    # 1. Map to continuous latent space
                    x_latent = self.projector(x_num, x_cat)
                    
                    # 2. Compute Diffusion MSE Loss
                    loss_dict = self.model(x_latent)
                    loss = loss_dict["loss"]
                
                # Backward pass with GradScaler to prevent underflow
                self.scaler.scale(loss).backward()
                
                # Gradient Clipping to prevent exploding gradients in MLPs
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                
            self.scheduler.step()
            avg_loss = epoch_loss / len(self.dataloader)
            self.loss_history.append(avg_loss)
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                elapsed = time.time() - start_time
                logger.info(f"Epoch [{epoch:04d}/{self.epochs:04d}] | Loss: {avg_loss:.6f} | LR: {self.scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")
                
            if epoch > 0 and epoch % self.checkpoint_freq == 0:
                self._save_checkpoint(epoch, avg_loss)
                
        # Final save (Master Weights)
        logger.info("Training complete. Saving Master Weights.")
        torch.save(self.model.state_dict(), self.output_dir / "datalus_master_fp32.pt")
        torch.save(self.projector.state_dict(), self.output_dir / "datalus_projector_fp32.pt")


if __name__ == "__main__":
    # Example execution configuration
    # Change resume_from to a path (e.g., "outputs/checkpoints/datalus_ckpt_epoch_0050.pt") to test recovery
    trainer = DatalusTrainer(
        schema_path="configs/schema_config.json",
        data_path="data/processed/datasus_sih_clean.parquet",
        output_dir="outputs/model_registry",
        batch_size=4096,  # 4096 is optimal for TabDDPM on 16GB VRAM
        epochs=1000,
        checkpoint_freq=50,
        resume_from=None 
    )
    
    # trainer.train()
    pass