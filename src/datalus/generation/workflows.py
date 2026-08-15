"""High-level synthetic data generation workflows.

These functions orchestrate the model bundle, encoder, and diffusion engine to
produce synthetic records in every supported mode (ab-initio, augmentation,
balancing, inpainting, counterfactual) and to export ONNX artifacts. The heavy
lifting happens in ``datalus.generation.bundle``, ``datalus.models``, and
``datalus.data``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import torch

from datalus.config import RePaintConfig
from datalus.export import (
    export_denoiser_onnx,
    quantize_int8,
    validate_int8_cfg_parity,
    validate_onnx_parity,
    write_manifest,
)
from datalus.generation.bundle import (
    base_denoiser_for_export,
    build_context_vector,
    decode_latent,
    drop_augmented_columns,
    intervention_latent_mask,
    latent_known_mask,
    load_model_bundle,
)


def sample_records(
    checkpoint_path: Path,
    encoder_path: Path,
    n_records: int,
    ddim_steps: int,
    seed: int,
    cfg_scale: float = 1.0,
    conditions: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Generate ab-initio synthetic records, optionally CFG-conditioned."""

    diffusion, projector, encoder, device, model_config = load_model_bundle(
        checkpoint_path,
        encoder_path,
    )
    context = build_context_vector(
        encoder,
        model_config,
        conditions,
        n_records,
        device,
    )
    latent = diffusion.sample_ddim(
        (n_records, projector.total_latent_dim),
        device=device,
        ddim_steps=ddim_steps,
        context=context,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    return drop_augmented_columns(decode_latent(latent, projector, encoder), encoder)


def augment_records(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    n_records: int,
    ddim_steps: int,
    seed: int,
    cfg_scale: float = 1.0,
    conditions: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Append ab-initio synthetic rows to an existing tabular dataset."""

    original = pl.read_parquet(input_path)
    synthetic = sample_records(
        checkpoint_path,
        encoder_path,
        n_records,
        ddim_steps,
        seed,
        cfg_scale,
        conditions,
    )
    return pl.concat([original, synthetic.select(original.columns)], how="vertical_relaxed")


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
    conditions: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Generate rows until requested class counts are reached, using per-class CFG contexts."""

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
        for label, count in list(needed.items()):
            if count <= 0:
                continue
            label_conditions = dict(conditions or {})
            label_conditions[target_column] = label
            candidate = sample_records(
                checkpoint_path,
                encoder_path,
                max(count * 2, 1),
                ddim_steps,
                seed + attempt,
                cfg_scale,
                label_conditions,
            )
            matched = candidate.filter(pl.col(target_column).cast(pl.String) == label)
            take = matched.head(count)
            if len(take):
                generated_parts.append(take)
                needed[label] -= len(take)
        remaining = sum(needed.values())
    if strict and remaining > 0:
        raise RuntimeError("Unable to satisfy the requested target distribution within max_attempts.")
    if not generated_parts:
        return original
    generated = pl.concat(generated_parts, how="vertical_relaxed")
    return pl.concat([original, generated.select(original.columns)], how="vertical_relaxed")


def inpaint_records(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    ddim_steps: int,
    jump_length: int,
    jump_n_sample: int,
    seed: int,
) -> pl.DataFrame:
    """Fill null values in tabular records with RePaint-style masks."""

    diffusion, projector, encoder, device, _ = load_model_bundle(checkpoint_path, encoder_path)
    frame = pl.read_parquet(input_path)
    encoded = encoder.transform(frame)
    x_num = torch.from_numpy(encoded.x_num).to(device) if encoded.x_num is not None else None
    x_cat = torch.from_numpy(encoded.x_cat).to(device) if encoded.x_cat is not None else None
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
    return drop_augmented_columns(decode_latent(latent, projector, encoder), encoder)


def counterfactual_records(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    intervention_json: str,
    ddim_steps: int,
    seed: int,
) -> pl.DataFrame:
    """Generate records under explicit do-style column interventions."""

    diffusion, projector, encoder, device, _ = load_model_bundle(checkpoint_path, encoder_path)
    frame = pl.read_parquet(input_path)
    interventions: dict[str, Any] = json.loads(intervention_json)
    intervened = frame.with_columns([pl.lit(value).alias(column) for column, value in interventions.items()])
    encoded = encoder.transform(intervened)
    x_num = torch.from_numpy(encoded.x_num).to(device) if encoded.x_num is not None else None
    x_cat = torch.from_numpy(encoded.x_cat).to(device) if encoded.x_cat is not None else None
    intervention_latent = projector(x_num, x_cat)
    mask = intervention_latent_mask(projector, encoder, list(interventions)).to(device)
    latent = diffusion.inpaint_repaint(
        intervention_latent,
        mask.repeat(len(frame), 1),
        RePaintConfig(num_inference_steps=ddim_steps, jump_length=10, jump_n_sample=5),
        seed=seed,
    )
    return drop_augmented_columns(decode_latent(latent, projector, encoder), encoder)


def export_onnx_artifacts(
    checkpoint_path: Path,
    encoder_path: Path,
    output_dir: Path,
    quantize: bool,
) -> None:
    """Export EMA denoiser weights and write an artifact manifest."""

    diffusion, projector, encoder, _, _ = load_model_bundle(
        checkpoint_path,
        encoder_path,
        use_ema=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    denoiser = base_denoiser_for_export(diffusion)
    fp32 = export_denoiser_onnx(
        denoiser,
        output_dir / "model_fp32.onnx",
        projector.total_latent_dim,
    )
    parity = validate_onnx_parity(denoiser, fp32, projector.total_latent_dim)
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
