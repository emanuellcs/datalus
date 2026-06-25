"""CLI delivery adapter with enterprise-grade help system and verbosity control.

The command line interface is deliberately thin. It translates user input into
application use-case calls, then reports file locations and audit outcomes. The
CLI does not own PyTorch, ONNX, or Polars workflow logic.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer

from datalus._console import setup_logging

from datalus.application.audit import (
    PrivacyEvaluator,
    UtilityEvaluator,
    write_audit_report,
)
from datalus.application.inference import (
    augment_records,
    balance_records,
    counterfactual_records,
    export_onnx_artifacts,
    inpaint_records,
    sample_records,
)
from datalus.application.training import DatalusTrainer
from datalus.domain.schemas import ShadowMIAConfig, TrainingConfig
from datalus.infrastructure.polars_preprocessing import ZeroShotPreprocessor

# ============================================================================
# Logger
# ============================================================================

_logger = logging.getLogger("datalus")

# ============================================================================
# Shared --verbose option default for subcommands.
#
# Every subcommand accepts ``--verbose`` / ``-v`` with a default of ``None``.
# When ``None`` the callback's global setting is left untouched.  When the
# user explicitly passes ``--verbose`` on the subcommand we re-configure
# logging so the later flag wins.
# ============================================================================

_VERBOSE_HELP = (
    "Override logging level for this command. "
    "Choices: WARNING (errors only), INFO (progress), DEBUG (detailed). "
    "If omitted, inherits the global --verbose setting."
)


def _apply_verbose(verbose: str | None) -> None:
    """Re-configure logging only when the subcommand explicitly sets --verbose."""
    if verbose is not None:
        setup_logging(verbose)


# ============================================================================
# Root application with global options
# ============================================================================

app = typer.Typer(
    help="DATALUS \u2014 Synthetic tabular data generation via diffusion models.",
    no_args_is_help=True,
    rich_help_panel="DATALUS",
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)


@app.callback()
def global_config(
    verbose: str = typer.Option(
        "WARNING",
        "--verbose",
        "-v",
        help="Global logging level: WARNING (errors only), INFO (progress), or DEBUG (detailed). "
        "Choices: WARNING, INFO, DEBUG.",
        rich_help_panel="Global Options",
    ),
) -> None:
    """Configure global logging verbosity before running commands.

    The --verbose flag can be placed before or after a subcommand:

      datalus --verbose INFO train ...
      datalus train --verbose INFO ...
    """
    setup_logging(verbose)


# ============================================================================
# Data Engineering & Preparation
# ============================================================================


@app.command(
    rich_help_panel="Data Engineering & Preparation",
    short_help="Infer schema and preprocess data to Parquet",
)
def ingest(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    input_path: Path = typer.Argument(..., help="Input data file or directory path"),
    output_path: Path = typer.Argument(..., help="Output Parquet file path"),
    schema_path: Path = typer.Option(
        Path("artifacts/schema_config.json"),
        "--schema-path",
        help="Path to save inferred schema JSON (default: artifacts/schema_config.json)",
    ),
    target_column: Optional[str] = typer.Option(
        None,
        "--target-column",
        help="Optional target column name for supervised learning context",
    ),
) -> None:
    """Infer schema from raw tabular data and write preprocessed output to Parquet.

    The ingest command analyzes your input data, infers a zero-shot schema,
    and writes standardized Parquet output suitable for training. The schema
    is saved as JSON for later use in train, sample, and audit commands.

    Example:
        datalus ingest raw_data.csv data.parquet
        datalus ingest --verbose INFO raw_data.csv data.parquet
    """
    _apply_verbose(verbose)
    _logger.info(f"Ingesting data from {input_path}...")
    prep = ZeroShotPreprocessor(target_column=target_column)
    prep.fit_transform_to_parquet(input_path, output_path, schema_path)
    _logger.info(f"Schema written to {schema_path}")
    typer.echo(f"Processed Parquet written to {output_path}")


# ============================================================================
# Core Modeling & Training
# ============================================================================


@app.command(
    rich_help_panel="Core Modeling & Training",
    short_help="Train diffusion model with checkpointing",
)
def train(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    schema_path: Path = typer.Argument(..., help="Path to schema JSON file"),
    data_path: Path = typer.Argument(..., help="Path to Parquet training data"),
    output_dir: Path = typer.Argument(
        ..., help="Directory for checkpoints and outputs"
    ),
    epochs: int = typer.Option(
        1, "--epochs", "-e", help="Number of training epochs (default: 1)"
    ),
    batch_size: int = typer.Option(
        2048, "--batch-size", "-b", help="Batch size for training (default: 2048)"
    ),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps", help="Stop training after N global steps (optional)"
    ),
    resume_from: Optional[Path] = typer.Option(
        None, "--resume-from", "-r", help="Resume training from checkpoint (optional)"
    ),
    gpu: Optional[str] = typer.Option(
        None, "--gpu", help="CUDA device indices, e.g., '0' or '0,1' (optional)"
    ),
    keep_last: Optional[int] = typer.Option(
        None, "--keep-last", help="Keep only N most recent checkpoints (optional)"
    ),
    save_every: int = typer.Option(
        1, "--save-every", help="Save checkpoint every N epochs (default: 1)"
    ),
    save_strategy: str = typer.Option(
        "latest",
        "--save-strategy",
        help="Strategy: 'all' (keep all), 'latest' (rolling), or 'best' (track lowest loss)",
    ),
) -> None:
    """Train a TabDDPM diffusion model on tabular data with deterministic checkpointing.

    The train command initializes a diffusion model, trains it on your data,
    and saves periodic checkpoints. Supports resuming from interruptions with
    full RNG state restoration. Use --verbose INFO to see epoch summaries and
    throughput metrics.

    Checkpoint Management:
      - Use --keep-last N to automatically delete old checkpoints (keeps N most recent)
      - Use --save-strategy best to maintain a checkpoint_best.pt file tracking lowest loss
      - Use --save-strategy all to keep every checkpoint (default: latest - rolling)

    Example:
        datalus train schema.json data.parquet ./checkpoints --epochs 10
        datalus train --verbose INFO schema.json data.parquet ./checkpoints
        datalus train --verbose DEBUG schema.json data.parquet ./checkpoints --keep-last 3
    """
    _apply_verbose(verbose)
    _logger.info(
        f"Training on {data_path} with schema {schema_path}. "
        f"Saving checkpoints to {output_dir}"
    )

    trainer = DatalusTrainer(
        TrainingConfig(
            schema_path=str(schema_path),
            data_path=str(data_path),
            output_dir=str(output_dir),
            epochs=epochs,
            batch_size=batch_size,
            gpu=gpu,
            keep_last=keep_last,
            save_every=save_every,
            save_strategy=save_strategy,
        )
    )
    if resume_from is not None:
        _logger.info(f"Resuming training from {resume_from}")
        trainer.resume(resume_from)
    checkpoint = trainer.train(max_steps=max_steps)
    typer.echo(f"Checkpoint written to {checkpoint}")


# ============================================================================
# Inference & Generation
# ============================================================================


@app.command(
    rich_help_panel="Inference & Generation",
    short_help="Generate synthetic dataset ab initio",
)
def sample(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    checkpoint_path: Path = typer.Argument(..., help="Path to training checkpoint"),
    encoder_path: Path = typer.Argument(..., help="Path to encoder JSON config"),
    output_path: Path = typer.Argument(..., help="Output Parquet file path"),
    n_records: int = typer.Option(
        100, "--n-records", "-n", help="Number of records to generate (default: 100)"
    ),
    ddim_steps: int = typer.Option(
        50, "--ddim-steps", help="DDIM reverse diffusion steps (default: 50)"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed (default: 42)"),
    cfg_scale: float = typer.Option(
        1.0,
        "--cfg-scale",
        help="Classifier-free guidance scale (default: 1.0 = disabled)",
    ),
    checkpoint_source: str = typer.Option(
        "latest",
        "--checkpoint-source",
        help="Use 'latest' or 'best' checkpoint (default: latest)",
    ),
) -> None:
    """Generate a new synthetic dataset from learned distributions.

    The sample command loads a trained checkpoint and generates records
    ab initio by reverse-diffusion sampling. Useful for synthetic data
    augmentation or privacy-preserving data sharing.

    Example:
        datalus sample checkpoint.pt encoder.json output.parquet --n-records 5000
        datalus sample --verbose INFO checkpoint.pt encoder.json output.parquet
    """
    _apply_verbose(verbose)
    _logger.info(f"Sampling {n_records} records from checkpoint {checkpoint_path}...")
    frame = sample_records(
        checkpoint_path, encoder_path, n_records, ddim_steps, seed, cfg_scale
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="snappy")
    typer.echo(f"Synthetic records written to {output_path}")


@app.command(
    rich_help_panel="Inference & Generation",
    short_help="Append synthetic records to existing dataset",
)
def augment(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    checkpoint_path: Path = typer.Argument(..., help="Path to training checkpoint"),
    encoder_path: Path = typer.Argument(..., help="Path to encoder JSON config"),
    input_path: Path = typer.Argument(..., help="Input Parquet file to augment"),
    output_path: Path = typer.Argument(..., help="Output Parquet file path"),
    n_records: int = typer.Option(
        100, "--n-records", "-n", help="Number of records to append (default: 100)"
    ),
    ddim_steps: int = typer.Option(
        50, "--ddim-steps", help="DDIM reverse diffusion steps (default: 50)"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed (default: 42)"),
    cfg_scale: float = typer.Option(
        1.0,
        "--cfg-scale",
        help="Classifier-free guidance scale (default: 1.0 = disabled)",
    ),
    checkpoint_source: str = typer.Option(
        "latest",
        "--checkpoint-source",
        help="Use 'latest' or 'best' checkpoint (default: latest)",
    ),
) -> None:
    """Append synthetic records to scale up a small dataset.

    The augment command loads your real data and generates synthetic
    records matching the learned distribution, then appends them to the
    original. Useful for addressing data scarcity or class imbalance.

    Example:
        datalus augment checkpoint.pt encoder.json input.parquet output.parquet
        datalus augment --verbose DEBUG checkpoint.pt encoder.json input.parquet output.parquet
    """
    _apply_verbose(verbose)
    _logger.info(f"Augmenting {input_path} with {n_records} synthetic records...")
    frame = augment_records(
        checkpoint_path,
        encoder_path,
        input_path,
        n_records,
        ddim_steps,
        seed,
        cfg_scale,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="snappy")
    typer.echo(f"Augmented dataset written to {output_path}")


@app.command(
    rich_help_panel="Inference & Generation",
    short_help="Generate records to achieve target class distribution",
)
def balance(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    checkpoint_path: Path = typer.Argument(..., help="Path to training checkpoint"),
    encoder_path: Path = typer.Argument(..., help="Path to encoder JSON config"),
    input_path: Path = typer.Argument(
        ..., help="Input Parquet with class labels to balance"
    ),
    output_path: Path = typer.Argument(..., help="Output Parquet file path"),
    target_column: str = typer.Argument(
        ..., help="Target column name for class labels"
    ),
    target_distribution_json: str = typer.Argument(
        ..., help='Target distribution as JSON, e.g., \'{"A": 0.5, "B": 0.5}\''
    ),
    ddim_steps: int = typer.Option(
        50, "--ddim-steps", help="DDIM reverse diffusion steps (default: 50)"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed (default: 42)"),
    cfg_scale: float = typer.Option(
        1.0,
        "--cfg-scale",
        help="Classifier-free guidance scale (default: 1.0 = disabled)",
    ),
    max_attempts: int = typer.Option(
        10, "--max-attempts", help="Maximum sampling attempts (default: 10)"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Fail if target distribution not achieved"
    ),
    checkpoint_source: str = typer.Option(
        "latest",
        "--checkpoint-source",
        help="Use 'latest' or 'best' checkpoint (default: latest)",
    ),
) -> None:
    """Generate records to approach a requested class distribution.

    The balance command generates synthetic records to rebalance your dataset
    toward a target class distribution. Useful for addressing imbalanced
    classification datasets or enforcing fairness constraints.

    Example:
        datalus balance checkpoint.pt encoder.json data.parquet label \\
            '{"positive": 0.5, "negative": 0.5}' output.parquet
    """
    _apply_verbose(verbose)
    _logger.info(
        f"Balancing {input_path} toward target distribution for {target_column}..."
    )
    frame = balance_records(
        checkpoint_path,
        encoder_path,
        input_path,
        target_column,
        json.loads(target_distribution_json),
        ddim_steps,
        seed,
        cfg_scale,
        max_attempts,
        strict,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="snappy")
    typer.echo(f"Balanced dataset written to {output_path}")


@app.command(
    rich_help_panel="Inference & Generation",
    short_help="Fill null values using diffusion masks",
)
def inpaint(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    checkpoint_path: Path = typer.Argument(..., help="Path to training checkpoint"),
    encoder_path: Path = typer.Argument(..., help="Path to encoder JSON config"),
    input_path: Path = typer.Argument(..., help="Input Parquet with null values"),
    output_path: Path = typer.Argument(..., help="Output Parquet file path"),
    ddim_steps: int = typer.Option(
        50, "--ddim-steps", help="DDIM reverse diffusion steps (default: 50)"
    ),
    jump_length: int = typer.Option(
        10, "--jump-length", help="RePaint jump length (default: 10)"
    ),
    jump_n_sample: int = typer.Option(
        10, "--jump-n-sample", help="RePaint jump N sample (default: 10)"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed (default: 42)"),
    checkpoint_source: str = typer.Option(
        "latest",
        "--checkpoint-source",
        help="Use 'latest' or 'best' checkpoint (default: latest)",
    ),
) -> None:
    """Fill null values in tabular records using RePaint-style masks.

    The inpaint command uses diffusion-based inpainting to fill missing
    values while preserving the observed values and learned data distribution.
    Based on RePaint masking strategy for controlled generation.

    Example:
        datalus inpaint checkpoint.pt encoder.json data.parquet output.parquet
        datalus inpaint --verbose INFO checkpoint.pt encoder.json data.parquet output.parquet
    """
    _apply_verbose(verbose)
    _logger.info(f"Inpainting null values in {input_path}...")
    frame = inpaint_records(
        checkpoint_path,
        encoder_path,
        input_path,
        ddim_steps,
        jump_length,
        jump_n_sample,
        seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="snappy")
    typer.echo(f"Inpainted records written to {output_path}")


@app.command(
    rich_help_panel="Inference & Generation",
    short_help="Generate records under do-style interventions",
)
def counterfactual(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    checkpoint_path: Path = typer.Argument(..., help="Path to training checkpoint"),
    encoder_path: Path = typer.Argument(..., help="Path to encoder JSON config"),
    input_path: Path = typer.Argument(..., help="Input Parquet for context"),
    output_path: Path = typer.Argument(..., help="Output Parquet file path"),
    intervention_json: str = typer.Argument(
        ...,
        help='Column interventions as JSON, e.g., \'{"age": 50, "income": 100000}\'',
    ),
    ddim_steps: int = typer.Option(
        50, "--ddim-steps", help="DDIM reverse diffusion steps (default: 50)"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed (default: 42)"),
    checkpoint_source: str = typer.Option(
        "latest",
        "--checkpoint-source",
        help="Use 'latest' or 'best' checkpoint (default: latest)",
    ),
) -> None:
    """Generate records under explicit do-style column interventions.

    The counterfactual command generates synthetic records that respect
    explicitly set column values (interventions) while allowing other
    features to vary according to the learned distribution. Useful for
    causal analysis, fairness auditing, and what-if scenarios.

    Example:
        datalus counterfactual checkpoint.pt encoder.json input.parquet \\
            '{"sensitive_attr": "protected_group"}' output.parquet
    """
    _apply_verbose(verbose)
    _logger.info("Generating counterfactuals with interventions...")
    frame = counterfactual_records(
        checkpoint_path,
        encoder_path,
        input_path,
        intervention_json,
        ddim_steps,
        seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression="snappy")
    typer.echo(f"Counterfactual records written to {output_path}")


# ============================================================================
# Auditing & Compliance
# ============================================================================


@app.command(
    rich_help_panel="Auditing & Compliance",
    short_help="Run privacy and utility audits",
)
def audit(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    real_train_path: Path = typer.Argument(
        ..., help="Path to real training Parquet data"
    ),
    synthetic_path: Path = typer.Argument(..., help="Path to synthetic Parquet data"),
    schema_path: Path = typer.Argument(..., help="Path to schema JSON file"),
    report_path: Path = typer.Argument(..., help="Output audit report path"),
    target_column: Optional[str] = typer.Option(
        None,
        "--target-column",
        help="Optional target column for utility metrics (classification)",
    ),
    real_holdout_path: Optional[Path] = typer.Option(
        None,
        "--real-holdout-path",
        help="Optional holdout dataset for privacy metrics (optional)",
    ),
    mia_mode: str = typer.Option(
        "release",
        "--mia-mode",
        help="Privacy audit mode: 'release' (less strict) or 'strict'",
    ),
    max_audit_rows: Optional[int] = typer.Option(
        None,
        "--max-audit-rows",
        help="Limit audit to N rows for speed (optional)",
    ),
) -> None:
    """Run privacy (Shadow MIA) and optional utility audits.

    The audit command compares real and synthetic datasets using
    Shadow Membership Inference Attacks (MIA) to quantify privacy risk,
    and optionally runs utility metrics if a target column is provided.

    Example:
        datalus audit real.parquet synthetic.parquet schema.json report.json
        datalus audit --verbose DEBUG real.parquet synthetic.parquet schema.json report.json
    """
    _apply_verbose(verbose)
    _logger.info("Running privacy and utility audits...")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    real_train = pl.read_parquet(real_train_path)
    synthetic = pl.read_parquet(synthetic_path)
    real_holdout = pl.read_parquet(real_holdout_path) if real_holdout_path else None
    report = PrivacyEvaluator(real_train, synthetic, schema, real_holdout).run_audit(
        config=ShadowMIAConfig(mode=mia_mode, max_rows=max_audit_rows)
    )
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


@app.command(
    rich_help_panel="Auditing & Compliance",
    short_help="Export EMA denoiser to ONNX format",
)
def export_onnx(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    checkpoint_path: Path = typer.Argument(..., help="Path to training checkpoint"),
    encoder_path: Path = typer.Argument(..., help="Path to encoder JSON config"),
    output_dir: Path = typer.Argument(..., help="Output directory for ONNX files"),
    quantize: bool = typer.Option(
        True,
        "--quantize/--no-quantize",
        help="Export INT8 quantized model (default: yes)",
    ),
) -> None:
    """Export EMA denoiser weights to ONNX and optional INT8 quantization.

    The export-onnx command saves the EMA denoiser model in ONNX format
    for cross-platform inference. Optionally quantizes to INT8 for reduced
    model size and faster CPU inference.

    Example:
        datalus export-onnx checkpoint.pt encoder.json ./onnx_model
        datalus export-onnx --verbose INFO checkpoint.pt encoder.json ./onnx_model
    """
    _apply_verbose(verbose)
    _logger.info(f"Exporting checkpoint to ONNX (quantize={quantize})...")
    export_onnx_artifacts(checkpoint_path, encoder_path, output_dir, quantize)
    typer.echo(f"ONNX artifacts written to {output_dir}")


# ============================================================================
# Interfaces & Services
# ============================================================================


@app.command(
    rich_help_panel="Interfaces & Services",
    short_help="Serve FastAPI interface for inference",
)
def serve(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
    registry_path: Path = typer.Option(
        Path("artifacts"),
        "--registry-path",
        help="Artifact registry directory (default: artifacts)",
    ),
    host: str = typer.Option(
        "0.0.0.0", "--host", help="Server bind address (default: 0.0.0.0)"
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="Server listen port (default: 8000)"
    ),
) -> None:
    """Serve DATALUS artifacts via FastAPI for browser-local inference.

    The serve command starts a FastAPI server for running inference on
    your trained checkpoint. Access the interactive API docs at
    http://localhost:<port>/docs

    Example:
        datalus serve --registry-path ./checkpoints --port 8000
    """
    _apply_verbose(verbose)
    _logger.info(f"Starting FastAPI server on {host}:{port}...")

    import os

    import uvicorn

    os.environ["DATALUS_REGISTRY_PATH"] = str(registry_path)
    uvicorn.run(
        "datalus.interfaces.api:create_app",
        host=host,
        port=port,
        factory=True,
        reload=False,
    )


@app.command(
    rich_help_panel="Interfaces & Services",
    short_help="Launch interactive Streamlit interface",
)
def streamlit_app(
    verbose: Optional[str] = typer.Option(None, "--verbose", "-v", help=_VERBOSE_HELP),
) -> None:
    """Launch the interactive Brazilian Portuguese Streamlit interface.

    The streamlit command opens a web-based UI for exploring DATALUS
    artifacts, running inference, and visualizing synthetic data.

    Example:
        datalus streamlit
    """
    _apply_verbose(verbose)
    _logger.info("Launching Streamlit interface...")

    import subprocess

    subprocess.run(
        ["streamlit", "run", "frontend/streamlit/app.py"],
        check=True,
    )


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    app()
