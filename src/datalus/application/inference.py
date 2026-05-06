"""Inference and artifact-export use cases.

Interfaces call these functions to execute business workflows. The functions
depend on infrastructure adapters for PyTorch tensors, ONNX export, and Polars
I/O, but the orchestration policy stays in the application layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import torch

from datalus.domain.schemas import RePaintConfig
from datalus.infrastructure.encoding import TabularEncoder
from datalus.infrastructure.onnx_export import (
    export_denoiser_onnx,
    quantize_int8,
    validate_int8_cfg_parity,
    validate_onnx_parity,
    write_manifest,
)
from datalus.infrastructure.torch_diffusion import TabularDiffusion
from datalus.infrastructure.torch_nn import EMA, FeatureProjector, TabularDenoiserMLP


def sample_records(
    checkpoint_path: Path,
    encoder_path: Path,
    n_records: int,
    ddim_steps: int,
    seed: int,
    cfg_scale: float = 1.0,
) -> pl.DataFrame:
    """Generate ab-initio synthetic records from a trained checkpoint."""

    diffusion, projector, encoder, device = load_model_bundle(
        checkpoint_path, encoder_path
    )
    latent = diffusion.sample_ddim(
        (n_records, projector.total_latent_dim),
        device=device,
        ddim_steps=ddim_steps,
        seed=seed,
        cfg_scale=cfg_scale,
    )
    return decode_latent(latent, projector, encoder)


def augment_records(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    n_records: int,
    ddim_steps: int,
    seed: int,
    cfg_scale: float = 1.0,
) -> pl.DataFrame:
    """Append ab-initio synthetic rows to an existing tabular dataset."""

    original = pl.read_parquet(input_path)
    synthetic = sample_records(
        checkpoint_path, encoder_path, n_records, ddim_steps, seed, cfg_scale
    )
    return pl.concat(
        [original, synthetic.select(original.columns)], how="vertical_relaxed"
    )


def balance_records(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    target_column: str,
    target_distribution: dict[str, int],
    ddim_steps: int,
    seed: int,
    cfg_scale: float = 1.0,
    max_attempts: int = 10,
    strict: bool = False,
) -> pl.DataFrame:
    """Generate rows until requested target-class counts are reached."""

    original = pl.read_parquet(input_path)
    if target_column not in original.columns:
        raise ValueError(f"Target column '{target_column}' is not present.")
    current_counts = {
        str(row[target_column]): int(row["len"])
        for row in original.group_by(target_column).len().iter_rows(named=True)
    }
    needed = {
        str(label): max(0, int(target) - current_counts.get(str(label), 0))
        for label, target in target_distribution.items()
    }
    remaining = sum(needed.values())
    generated_parts: list[pl.DataFrame] = []
    attempt = 0
    while remaining > 0 and attempt < max_attempts:
        attempt += 1
        candidate = sample_records(
            checkpoint_path,
            encoder_path,
            max(remaining * 2, 1),
            ddim_steps,
            seed + attempt,
            cfg_scale,
        )
        for label, count in list(needed.items()):
            if count <= 0 or target_column not in candidate.columns:
                continue
            matched = candidate.filter(pl.col(target_column).cast(pl.String) == label)
            take = matched.head(count)
            if len(take):
                generated_parts.append(take)
                needed[label] -= len(take)
        remaining = sum(needed.values())
    if strict and remaining > 0:
        raise RuntimeError(
            "Unable to satisfy the requested target distribution within max_attempts."
        )
    if not generated_parts:
        return original
    generated = pl.concat(generated_parts, how="vertical_relaxed")
    return pl.concat(
        [original, generated.select(original.columns)], how="vertical_relaxed"
    )


def inpaint_records(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    ddim_steps: int,
    jump_length: int,
    jump_n_sample: int,
    seed: int,
) -> pl.DataFrame:
    """Fill null values in tabular records with RePaint-style mask harmonization."""

    diffusion, projector, encoder, device = load_model_bundle(
        checkpoint_path, encoder_path
    )
    frame = pl.read_parquet(input_path)
    encoded = encoder.transform(frame)
    x_num = (
        torch.from_numpy(encoded.x_num).to(device)
        if encoded.x_num is not None
        else None
    )
    x_cat = (
        torch.from_numpy(encoded.x_cat).to(device)
        if encoded.x_cat is not None
        else None
    )
    original_latent = projector(x_num, x_cat)
    mask = latent_known_mask(frame, projector, encoder).to(device)
    latent = diffusion.inpaint_repaint(
        original_latent,
        mask,
        RePaintConfig(
            num_inference_steps=ddim_steps,
            jump_length=jump_length,
            jump_n_sample=jump_n_sample,
        ),
        seed=seed,
    )
    return decode_latent(latent, projector, encoder)


def counterfactual_records(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    intervention_json: str,
    ddim_steps: int,
    seed: int,
) -> pl.DataFrame:
    """Generate records under explicit do-style column interventions."""

    diffusion, projector, encoder, device = load_model_bundle(
        checkpoint_path, encoder_path
    )
    frame = pl.read_parquet(input_path)
    interventions: dict[str, Any] = json.loads(intervention_json)
    intervened = frame.with_columns(
        [pl.lit(value).alias(column) for column, value in interventions.items()]
    )
    encoded = encoder.transform(intervened)
    x_num = (
        torch.from_numpy(encoded.x_num).to(device)
        if encoded.x_num is not None
        else None
    )
    x_cat = (
        torch.from_numpy(encoded.x_cat).to(device)
        if encoded.x_cat is not None
        else None
    )
    intervention_latent = projector(x_num, x_cat)
    mask = intervention_latent_mask(projector, encoder, list(interventions)).to(device)
    latent = diffusion.inpaint_repaint(
        intervention_latent,
        mask.repeat(len(frame), 1),
        RePaintConfig(num_inference_steps=ddim_steps, jump_length=10, jump_n_sample=5),
        seed=seed,
    )
    return decode_latent(latent, projector, encoder)


def export_onnx_artifacts(
    checkpoint_path: Path,
    encoder_path: Path,
    output_dir: Path,
    quantize: bool,
) -> None:
    """Export EMA denoiser weights and write an artifact manifest."""

    diffusion, projector, encoder, _ = load_model_bundle(
        checkpoint_path,
        encoder_path,
        use_ema=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32 = export_denoiser_onnx(
        diffusion.denoiser,
        output_dir / "model_fp32.onnx",
        projector.total_latent_dim,
    )
    parity = validate_onnx_parity(diffusion.denoiser, fp32, projector.total_latent_dim)
    artifacts = {"model_fp32": fp32.name}
    int8_parity = None
    if quantize:
        int8 = quantize_int8(fp32, output_dir / "model_int8.onnx")
        artifacts["model_int8"] = int8.name
        int8_parity = validate_int8_cfg_parity(
            fp32,
            int8,
            projector.total_latent_dim,
            cfg_scale=3.0,
        )
    encoder.save(output_dir / "encoder_config.json")
    write_manifest(output_dir / "projector_config.json", projector.to_browser_config())
    write_manifest(
        output_dir / "manifest.json",
        {
            "latent_dim": projector.total_latent_dim,
            "artifacts": artifacts,
            "onnx_parity": parity,
            "int8_cfg_parity": int8_parity,
        },
    )


def load_model_bundle(
    checkpoint_path: Path,
    encoder_path: Path,
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
    hidden_dims = tuple(
        checkpoint.get("config", {}).get("hidden_dims", (512, 1024, 1024, 512))
    )
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
    for column, (_, emb_dim) in zip(encoder.categorical_columns, projector.cat_dims):
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
    for column, (_, emb_dim) in zip(encoder.categorical_columns, projector.cat_dims):
        value = 1.0 if column in active else 0.0
        parts.append(torch.full((1, emb_dim), value))
    return torch.cat(parts, dim=1).float()
