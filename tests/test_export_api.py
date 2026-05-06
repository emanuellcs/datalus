import json

import pytest
import torch
from fastapi.testclient import TestClient

from datalus.infrastructure.onnx_export import (
    export_denoiser_onnx,
    quantize_int8,
    validate_int8_cfg_parity,
)
from datalus.infrastructure.torch_nn import TabularDenoiserMLP
from datalus.interfaces.api import create_app


def test_artifact_api_serves_manifest(tmp_path):
    domain = tmp_path / "demo"
    domain.mkdir()
    (domain / "manifest.json").write_text(
        json.dumps({"name": "demo"}), encoding="utf-8"
    )
    app = create_app(tmp_path)
    client = TestClient(app)
    response = client.get("/artifacts/demo/manifest")
    assert response.status_code == 200
    assert response.json()["name"] == "demo"


def test_artifact_api_rejects_domain_path_traversal(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    response = client.get("/artifacts/../manifest")
    assert response.status_code in {403, 404}


def test_int8_cfg_parity_guard_runs_on_small_onnx_export(tmp_path):
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    denoiser = TabularDenoiserMLP(d_in=3, hidden_dims=(8, 8), dim_t=8).eval()
    fp32 = export_denoiser_onnx(denoiser, tmp_path / "model_fp32.onnx", latent_dim=3)
    int8 = quantize_int8(fp32, tmp_path / "model_int8.onnx")
    parity = validate_int8_cfg_parity(fp32, int8, latent_dim=3, cfg_scale=3.0)
    assert parity["cfg_scale"] == 3.0
    assert "amplified_max_abs_diff" in parity
