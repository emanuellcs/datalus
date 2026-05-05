"""Command line interface for DATALUS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import polars as pl
import torch
import typer

from datalus.audit import PrivacyEvaluator, UtilityEvaluator, write_audit_report
from datalus.diffusion import RePaintConfig, TabularDiffusion
from datalus.encoding import TabularEncoder
from datalus.export import (
    export_denoiser_onnx,
    quantize_int8,
    validate_onnx_parity,
    write_manifest,
)
from datalus.nn import EMA, FeatureProjector, TabularDenoiserMLP
from datalus.preprocessing import ZeroShotPreprocessor
from datalus.training import DatalusTrainer, TrainingConfig

app = typer.Typer(help="DATALUS synthetic tabular data framework.")


@app.command()
def ingest(
    input_path: Path,
    output_path: Path,
    schema_path: Path = Path("artifacts/schema_config.json"),
    target_column: Optional[str] = None,
) -> None:
    """Infer schema and stream retained data to Parquet."""

    prep = ZeroShotPreprocessor(target_column=target_column)
    prep.fit_transform_to_parquet(input_path, output_path, schema_path)
    typer.echo(f"Schema written to {schema_path}")
    typer.echo(f"Processed Parquet written to {output_path}")


@app.command()
def train(
    schema_path: Path,
    data_path: Path,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 2048,
    max_steps: Optional[int] = None,
    resume_from: Optional[Path] = None,
) -> None:
    """Train DATALUS with deterministic checkpointing."""

    trainer = DatalusTrainer(
        TrainingConfig(
            schema_path=str(schema_path),
            data_path=str(data_path),
            output_dir=str(output_dir),
            epochs=epochs,
            batch_size=batch_size,
        )
    )
    if resume_from is not None:
        trainer.resume(resume_from)
    checkpoint = trainer.train(max_steps=max_steps)
    typer.echo(f"Checkpoint written to {checkpoint}")


@app.command()
def sample(
    checkpoint_path: Path,
    encoder_path: Path,
    output_path: Path,
    n_records: int = 100,
    ddim_steps: int = 50,
    seed: int = 42,
) -> None:
    """Generate synthetic records from a trained checkpoint."""

    diffusion, projector, encoder, device = _load_model_bundle(
        checkpoint_path, encoder_path
    )
    latent = diffusion.sample_ddim(
        (n_records, projector.total_latent_dim),
        device=device,
        ddim_steps=ddim_steps,
        seed=seed,
    )
    frame = _decode_latent(latent, projector, encoder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="snappy")
    typer.echo(f"Synthetic records written to {output_path}")


@app.command("inpaint")
def inpaint(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    output_path: Path,
    ddim_steps: int = 50,
    jump_length: int = 10,
    jump_n_sample: int = 10,
    seed: int = 42,
) -> None:
    """Fill null values in tabular records using RePaint-style masks."""

    diffusion, projector, encoder, device = _load_model_bundle(
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
    mask = _latent_known_mask(frame, projector, encoder).to(device)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _decode_latent(latent, projector, encoder).write_parquet(
        output_path, compression="snappy"
    )
    typer.echo(f"Inpainted records written to {output_path}")


@app.command("counterfactual")
def counterfactual(
    checkpoint_path: Path,
    encoder_path: Path,
    input_path: Path,
    output_path: Path,
    intervention_json: str,
    ddim_steps: int = 50,
    seed: int = 42,
) -> None:
    """Generate records under explicit do-style column interventions."""

    diffusion, projector, encoder, device = _load_model_bundle(
        checkpoint_path, encoder_path
    )
    frame = pl.read_parquet(input_path)
    interventions = json.loads(intervention_json)
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
    mask = _intervention_latent_mask(projector, encoder, list(interventions)).to(device)
    mask = mask.repeat(len(frame), 1)
    latent = diffusion.inpaint_repaint(
        intervention_latent,
        mask,
        RePaintConfig(num_inference_steps=ddim_steps, jump_length=10, jump_n_sample=5),
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _decode_latent(latent, projector, encoder).write_parquet(
        output_path, compression="snappy"
    )
    typer.echo(f"Counterfactual records written to {output_path}")


@app.command("export-onnx")
def export_onnx(
    checkpoint_path: Path,
    encoder_path: Path,
    output_dir: Path,
    quantize: bool = True,
) -> None:
    """Export EMA denoiser weights to ONNX and optional INT8."""

    diffusion, projector, encoder, _ = _load_model_bundle(
        checkpoint_path, encoder_path, use_ema=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32 = export_denoiser_onnx(
        diffusion.denoiser, output_dir / "model_fp32.onnx", projector.total_latent_dim
    )
    parity = validate_onnx_parity(diffusion.denoiser, fp32, projector.total_latent_dim)
    artifacts = {"model_fp32": fp32.name}
    if quantize:
        int8 = quantize_int8(fp32, output_dir / "model_int8.onnx")
        artifacts["model_int8"] = int8.name
    encoder.save(output_dir / "encoder_config.json")
    write_manifest(
        output_dir / "manifest.json",
        {
            "latent_dim": projector.total_latent_dim,
            "artifacts": artifacts,
            "onnx_parity": parity,
        },
    )
    typer.echo(f"ONNX artifacts written to {output_dir}")


@app.command()
def audit(
    real_train_path: Path,
    synthetic_path: Path,
    schema_path: Path,
    report_path: Path,
    target_column: Optional[str] = None,
    real_holdout_path: Optional[Path] = None,
) -> None:
    """Run privacy and optional utility audits."""

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    real_train = pl.read_parquet(real_train_path)
    synthetic = pl.read_parquet(synthetic_path)
    real_holdout = pl.read_parquet(real_holdout_path) if real_holdout_path else None
    report = PrivacyEvaluator(real_train, synthetic, schema, real_holdout).run_audit()
    if (
        target_column is not None
        and target_column in real_train.columns
        and target_column in synthetic.columns
    ):
        report.update(
            UtilityEvaluator(real_train, synthetic, schema, target_column).run_audit()
        )
    write_audit_report(report_path, report)
    typer.echo(f"Audit report written to {report_path}")


@app.command()
def serve(
    registry_path: Path = Path("artifacts"), host: str = "0.0.0.0", port: int = 8000
) -> None:
    """Serve DATALUS artifacts for browser-local inference."""

    import os
    import uvicorn

    os.environ["DATALUS_REGISTRY_PATH"] = str(registry_path)
    uvicorn.run(
        "datalus.api:create_app", host=host, port=port, factory=True, reload=False
    )


@app.command("streamlit")
def streamlit_app() -> None:
    """Launch the Brazilian Portuguese Streamlit interface."""

    import subprocess

    subprocess.run(
        ["streamlit", "run", "frontend/streamlit/app.py"],
        check=True,
    )


def _load_model_bundle(
    checkpoint_path: Path,
    encoder_path: Path,
    use_ema: bool = False,
):
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
        d_in=projector.total_latent_dim, hidden_dims=hidden_dims
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
def _decode_latent(
    latent: torch.Tensor, projector: FeatureProjector, encoder: TabularEncoder
) -> pl.DataFrame:
    x_num = projector.split_numerical(latent)
    x_cat = projector.nearest_category_indices(latent)
    return encoder.inverse_transform(
        x_num.detach().cpu().numpy() if x_num is not None else None,
        x_cat.detach().cpu().numpy() if x_cat is not None else None,
    )


def _latent_known_mask(
    frame: pl.DataFrame, projector: FeatureProjector, encoder: TabularEncoder
) -> torch.Tensor:
    parts = []
    for column in encoder.numerical_columns:
        known = (~frame.get_column(column).is_null()).cast(pl.Float32).to_numpy()
        parts.append(torch.from_numpy(known[:, None]))
    for column, (_, emb_dim) in zip(encoder.categorical_columns, projector.cat_dims):
        known = (~frame.get_column(column).is_null()).cast(pl.Float32).to_numpy()
        parts.append(torch.from_numpy(known[:, None]).repeat(1, emb_dim))
    return torch.cat(parts, dim=1).float()


def _intervention_latent_mask(
    projector: FeatureProjector,
    encoder: TabularEncoder,
    intervention_columns: list[str],
) -> torch.Tensor:
    active = set(intervention_columns)
    parts = []
    for column in encoder.numerical_columns:
        parts.append(torch.tensor([[1.0 if column in active else 0.0]]))
    for column, (_, emb_dim) in zip(encoder.categorical_columns, projector.cat_dims):
        value = 1.0 if column in active else 0.0
        parts.append(torch.full((1, emb_dim), value))
    return torch.cat(parts, dim=1).float()
