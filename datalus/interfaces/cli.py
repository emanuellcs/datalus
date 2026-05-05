"""CLI delivery adapter.

The command line interface is deliberately thin. It translates user input into
application use-case calls, then reports file locations and audit outcomes. The
CLI does not own PyTorch, ONNX, or Polars workflow logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import polars as pl
import typer

from datalus.application.audit import (
    PrivacyEvaluator,
    UtilityEvaluator,
    write_audit_report,
)
from datalus.application.inference import (
    counterfactual_records,
    export_onnx_artifacts,
    inpaint_records,
    sample_records,
)
from datalus.application.training import DatalusTrainer
from datalus.domain.schemas import TrainingConfig
from datalus.infrastructure.polars_preprocessing import ZeroShotPreprocessor

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

    frame = sample_records(checkpoint_path, encoder_path, n_records, ddim_steps, seed)
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


@app.command("export-onnx")
def export_onnx(
    checkpoint_path: Path,
    encoder_path: Path,
    output_dir: Path,
    quantize: bool = True,
) -> None:
    """Export EMA denoiser weights to ONNX and optional INT8."""

    export_onnx_artifacts(checkpoint_path, encoder_path, output_dir, quantize)
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
    registry_path: Path = Path("artifacts"),
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Serve DATALUS artifacts for browser-local inference."""

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


@app.command("streamlit")
def streamlit_app() -> None:
    """Launch the Brazilian Portuguese Streamlit interface."""

    import subprocess

    subprocess.run(
        ["streamlit", "run", "frontend/streamlit/app.py"],
        check=True,
    )
