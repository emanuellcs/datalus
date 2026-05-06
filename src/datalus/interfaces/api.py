"""FastAPI artifact service for DATALUS deployments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

START_TIME = time.time()


class GenerationRequest(BaseModel):
    """HTTP contract for ab-initio synthetic tabular generation."""

    domain: str
    n_records: int = Field(default=100, ge=1)
    ddim_steps: int = Field(default=50, ge=1)
    seed: int = 42
    cfg_scale: float = Field(default=1.0, ge=0.0)


class AugmentationRequest(GenerationRequest):
    """HTTP contract for scaling an existing Parquet dataset."""

    input_path: str


class BalancingRequest(AugmentationRequest):
    """HTTP contract for long-tail and minority-class balancing."""

    target_column: str
    target_distribution: dict[str, int] = Field(default_factory=dict)
    max_attempts: int = Field(default=10, ge=1)
    strict: bool = False


class InpaintingRequest(GenerationRequest):
    """HTTP contract for RePaint-style missing-value completion."""

    input_path: str
    jump_length: int = Field(default=10, ge=1)
    jump_n_sample: int = Field(default=10, ge=1)


class CounterfactualRequest(GenerationRequest):
    """HTTP contract for latent-space interventions on existing records."""

    input_path: str
    interventions: dict[str, Any] = Field(default_factory=dict)


def create_app(
    registry_path: str | Path | None = None, enable_server_generation: bool = False
) -> FastAPI:
    """Create an API that serves artifacts for browser-local inference."""

    registry = Path(
        registry_path or os.getenv("DATALUS_REGISTRY_PATH", "artifacts")
    ).resolve()
    app = FastAPI(
        title="DATALUS Artifact Service",
        description="Serves ONNX artifacts and schema metadata for local browser inference.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "operational",
            "uptime_seconds": round(time.time() - START_TIME, 3),
            "registry_path": str(registry),
        }

    @app.get("/artifacts")
    async def artifacts() -> dict[str, Any]:
        if not registry.exists():
            return {"artifacts": []}
        return {
            "artifacts": sorted(
                path.name for path in registry.iterdir() if path.is_dir()
            )
        }

    @app.get("/artifacts/{domain}/manifest")
    async def artifact_manifest(domain: str):
        return _json_file(_domain_root(registry, domain) / "manifest.json")

    @app.get("/artifacts/{domain}/schema")
    async def artifact_schema(domain: str):
        return _json_file(_domain_root(registry, domain) / "schema_config.json")

    @app.get("/artifacts/{domain}/{file_name}")
    async def artifact_file(domain: str, file_name: str):
        allowed = {
            "model_fp32.onnx",
            "model_fp16.onnx",
            "model_int8.onnx",
            "schema_config.json",
            "encoder_config.json",
            "projector_config.json",
            "model_config.json",
            "audit_report.json",
            "manifest.json",
        }
        if file_name not in allowed:
            raise HTTPException(status_code=403, detail="Artifact file is not public.")
        path = _domain_root(registry, domain) / file_name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(path)

    @app.get("/audit/latest")
    async def latest_audit():
        candidates = sorted(
            registry.glob("*/audit_report.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise HTTPException(status_code=404, detail="No audit report found.")
        return _json_file(candidates[0])

    @app.post("/generate")
    async def generate(request: GenerationRequest):
        if not enable_server_generation:
            raise HTTPException(
                status_code=403,
                detail="Server-side generation is disabled. Use the browser ONNX Runtime Web component.",
            )
        from datalus.application.inference import sample_records

        checkpoint, encoder = _model_paths(registry, request.domain)
        frame = sample_records(
            checkpoint,
            encoder,
            request.n_records,
            request.ddim_steps,
            request.seed,
            request.cfg_scale,
        )
        return {"records": frame.to_dicts()}

    @app.post("/augment")
    async def augment(request: AugmentationRequest):
        if not enable_server_generation:
            raise HTTPException(
                status_code=403, detail="Server-side generation is disabled."
            )
        from datalus.application.inference import augment_records

        checkpoint, encoder = _model_paths(registry, request.domain)
        frame = augment_records(
            checkpoint,
            encoder,
            Path(request.input_path),
            request.n_records,
            request.ddim_steps,
            request.seed,
            request.cfg_scale,
        )
        return {"records": frame.to_dicts()}

    @app.post("/balance")
    async def balance(request: BalancingRequest):
        if not enable_server_generation:
            raise HTTPException(
                status_code=403, detail="Server-side generation is disabled."
            )
        from datalus.application.inference import balance_records

        checkpoint, encoder = _model_paths(registry, request.domain)
        frame = balance_records(
            checkpoint,
            encoder,
            Path(request.input_path),
            request.target_column,
            request.target_distribution,
            request.ddim_steps,
            request.seed,
            request.cfg_scale,
            request.max_attempts,
            request.strict,
        )
        return {"records": frame.to_dicts()}

    @app.post("/inpaint")
    async def inpaint(request: InpaintingRequest):
        if not enable_server_generation:
            raise HTTPException(
                status_code=403, detail="Server-side generation is disabled."
            )
        from datalus.application.inference import inpaint_records

        checkpoint, encoder = _model_paths(registry, request.domain)
        frame = inpaint_records(
            checkpoint,
            encoder,
            Path(request.input_path),
            request.ddim_steps,
            request.jump_length,
            request.jump_n_sample,
            request.seed,
        )
        return {"records": frame.to_dicts()}

    @app.post("/counterfactual")
    async def counterfactual(request: CounterfactualRequest):
        if not enable_server_generation:
            raise HTTPException(
                status_code=403, detail="Server-side generation is disabled."
            )
        from datalus.application.inference import counterfactual_records

        checkpoint, encoder = _model_paths(registry, request.domain)
        frame = counterfactual_records(
            checkpoint,
            encoder,
            Path(request.input_path),
            json.dumps(request.interventions),
            request.ddim_steps,
            request.seed,
        )
        return {"records": frame.to_dicts()}

    return app


def _json_file(path: Path) -> JSONResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="JSON artifact not found.")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


def _model_paths(registry: Path, domain: str) -> tuple[Path, Path]:
    """Resolve the training checkpoint and encoder artifacts for a domain."""

    root = _domain_root(registry, domain)
    checkpoint_candidates = [
        root / "checkpoint_latest.pt",
        root / "checkpoints" / "checkpoint_latest.pt",
    ]
    checkpoint = next((path for path in checkpoint_candidates if path.exists()), None)
    encoder = root / "encoder_config.json"
    if checkpoint is None or not encoder.exists():
        raise HTTPException(
            status_code=404, detail="Model checkpoint or encoder not found."
        )
    return checkpoint, encoder


def _domain_root(registry: Path, domain: str) -> Path:
    """Resolve an artifact domain while preventing registry path traversal."""

    root = (registry / domain).resolve()
    try:
        root.relative_to(registry)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid artifact domain.") from exc
    return root


app = create_app()
