"""Model-bundle reconstruction and latent decoding helpers.

Reconstructs a trained diffusion model, projector, and encoder from DATALUS
artifacts and converts latent tensors back into Polars DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch

from datalus.data.encoding import TabularEncoder
from datalus.models.diffusion import TabularDiffusion
from datalus.models.nn import EMA, FeatureProjector, TabularDenoiserMLP


def load_model_bundle(
    checkpoint_path: str | Path,
    encoder_path: str | Path,
    use_ema: bool = False,
) -> tuple[TabularDiffusion, FeatureProjector, TabularEncoder, torch.device]:
    """Reconstruct model, projector, and encoder from DATALUS artifacts."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = TabularEncoder.load(encoder_path)
    projector = FeatureProjector(
        encoder.schema_metadata,
        encoder.numerical_columns,
        encoder.categorical_columns,
    )
    projector.load_state_dict(checkpoint["projector_state"])
    hidden_dims = tuple(checkpoint.get("config", {}).get("hidden_dims", (512, 1024, 1024, 512)))
    num_timesteps = int(checkpoint.get("config", {}).get("num_timesteps", 1000))
    denoiser = TabularDenoiserMLP(
        d_in=projector.total_latent_dim,
        hidden_dims=hidden_dims,
    )
    diffusion = TabularDiffusion(denoiser, num_timesteps=num_timesteps)
    diffusion.load_state_dict(checkpoint["diffusion_state"])
    if use_ema and "ema_state" in checkpoint:
        ema = EMA(diffusion)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(diffusion)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return diffusion.to(device).eval(), projector.to(device).eval(), encoder, device


@torch.no_grad()
def decode_latent(
    latent: torch.Tensor,
    projector: FeatureProjector,
    encoder: TabularEncoder,
) -> pl.DataFrame:
    """Decode latent tensors into a Polars DataFrame using fitted artifacts."""

    x_num = projector.split_numerical(latent)
    x_cat = projector.nearest_category_indices(latent)
    return encoder.inverse_transform(
        x_num.detach().cpu().numpy() if x_num is not None else None,
        x_cat.detach().cpu().numpy() if x_cat is not None else None,
    )


def latent_known_mask(
    frame: pl.DataFrame,
    projector: FeatureProjector,
    encoder: TabularEncoder,
) -> torch.Tensor:
    """Build a latent-space mask where one means observed and zero means missing."""

    parts: list[torch.Tensor] = []
    for column in encoder.numerical_columns:
        known = (~frame.get_column(column).is_null()).cast(pl.Float32).to_numpy()
        parts.append(torch.from_numpy(known[:, None]))
    for column, (_, emb_dim) in zip(encoder.categorical_columns, projector.cat_dims, strict=False):
        known = (~frame.get_column(column).is_null()).cast(pl.Float32).to_numpy()
        parts.append(torch.from_numpy(known[:, None]).repeat(1, emb_dim))
    return torch.cat(parts, dim=1).float()


def intervention_latent_mask(
    projector: FeatureProjector,
    encoder: TabularEncoder,
    intervention_columns: list[str],
) -> torch.Tensor:
    """Build a latent mask for coordinates fixed by do-style interventions."""

    active = set(intervention_columns)
    parts: list[torch.Tensor] = []
    for column in encoder.numerical_columns:
        parts.append(torch.tensor([[1.0 if column in active else 0.0]]))
    for column, (_, emb_dim) in zip(encoder.categorical_columns, projector.cat_dims, strict=False):
        value = 1.0 if column in active else 0.0
        parts.append(torch.full((1, emb_dim), value))
    return torch.cat(parts, dim=1).float()
