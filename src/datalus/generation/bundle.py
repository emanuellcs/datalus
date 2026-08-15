"""Model-bundle reconstruction and latent decoding helpers.

Reconstructs a trained diffusion model, projector, and encoder from DATALUS
artifacts and converts latent tensors back into Polars DataFrames. The bundle
reads the serialized model configuration from the checkpoint so the denoiser
architecture (MLP or transformer, with optional categorical heads) is rebuilt
identically to training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from datalus.data.encoding import TabularEncoder
from datalus.models.diffusion import TabularDiffusion
from datalus.models.nn import EMA, CategoricalHeadedDenoiser, FeatureProjector, build_denoiser
from datalus.training.checkpointing import load_checkpoint


def _context_dim_from_config(model_config: dict, encoder: TabularEncoder) -> tuple[int | None, bool]:
    """Resolve the CFG context dimension and target topology from a checkpoint."""

    target = model_config.get("target_column")
    if target is None:
        return None, False
    if target in encoder.categorical_columns:
        return encoder.categorical_vocabs[target].size, True
    if target in encoder.numerical_columns:
        return 1, False
    return None, False


def load_model_bundle(
    checkpoint_path: str | Path,
    encoder_path: str | Path,
    use_ema: bool = False,
) -> tuple[TabularDiffusion, FeatureProjector, TabularEncoder, torch.device, dict]:
    """Reconstruct model, projector, and encoder from DATALUS artifacts.

    Returns the serialized model config so callers can rebuild CFG contexts.
    """

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    encoder = TabularEncoder.load(encoder_path)
    projector = FeatureProjector(
        encoder.schema_metadata,
        encoder.numerical_columns,
        encoder.categorical_columns,
    )
    projector.load_state_dict(checkpoint["projector_state"])
    model_config = checkpoint.get("config", {})
    num_timesteps = int(model_config.get("num_timesteps", 1000))
    context_dim, _ = _context_dim_from_config(model_config, encoder)
    denoiser = build_denoiser(
        projector.total_latent_dim,
        projector.num_dim,
        projector.cat_dims,
        model_config,
        context_dim=context_dim,
    )
    diffusion = TabularDiffusion(
        denoiser,
        num_timesteps=num_timesteps,
        num_dim=projector.num_dim,
        cat_dims=projector.cat_dims,
        lambda_num=float(model_config.get("lambda_num", 1.0)),
        lambda_cat=float(model_config.get("lambda_cat", 0.0)),
    )
    diffusion.load_state_dict(checkpoint["diffusion_state"])
    if use_ema and "ema_state" in checkpoint:
        ema = EMA(diffusion)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(diffusion)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return diffusion.to(device).eval(), projector.to(device).eval(), encoder, device, model_config


def base_denoiser_for_export(diffusion: TabularDiffusion) -> torch.nn.Module:
    """Return the noise-only denoiser for ONNX export, unwrapping logit heads."""

    denoiser = diffusion.denoiser
    if isinstance(denoiser, CategoricalHeadedDenoiser):
        return denoiser.base
    return denoiser


def build_context_vector(
    encoder: TabularEncoder,
    model_config: dict,
    conditions: dict | None,
    n_records: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the CFG context tensor for the checkpoint's target column."""

    if not conditions:
        return None
    target, target_is_categorical = _context_dim_from_config(model_config, encoder)
    if target is None:
        return None
    if target not in conditions:
        return None
    vector = np.zeros((n_records, target), dtype=np.float32)
    if target_is_categorical:
        vocab = encoder.categorical_vocabs[target]
        category_idx = vocab.transform(np.array([conditions[target]]))[0]
        vector[:, int(category_idx)] = 1.0
    else:
        transform = encoder.numeric_transforms[target]
        value = transform.transform(np.array([float(conditions[target])]))[0]
        vector[:, 0] = value
    return torch.from_numpy(vector).to(device)


def drop_augmented_columns(frame: pl.DataFrame, encoder: TabularEncoder) -> pl.DataFrame:
    """Drop training-time augmented columns from decoded output frames."""

    augmented = [column for column in encoder.augmented_columns if column in frame.columns]
    if not augmented:
        return frame
    return frame.drop(augmented)


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
