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

START_TIME = time.time()


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
        return _json_file(registry / domain / "manifest.json")

    @app.get("/artifacts/{domain}/schema")
    async def artifact_schema(domain: str):
        return _json_file(registry / domain / "schema_config.json")

    @app.get("/artifacts/{domain}/{file_name}")
    async def artifact_file(domain: str, file_name: str):
        allowed = {
            "model_fp32.onnx",
            "model_fp16.onnx",
            "model_int8.onnx",
            "schema_config.json",
            "encoder_config.json",
            "model_config.json",
            "audit_report.json",
            "manifest.json",
        }
        if file_name not in allowed:
            raise HTTPException(status_code=403, detail="Artifact file is not public.")
        path = registry / domain / file_name
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
    async def generate_disabled():
        if not enable_server_generation:
            raise HTTPException(
                status_code=403,
                detail="Server-side generation is disabled. Use the browser ONNX Runtime Web component.",
            )
        raise HTTPException(
            status_code=501, detail="Server-side generation is not configured."
        )

    return app


def _json_file(path: Path) -> JSONResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="JSON artifact not found.")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


app = create_app()
