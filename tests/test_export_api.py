"""Tests for the artifact API and ONNX export parity guards."""

import json

import pytest
import torch
from fastapi.testclient import TestClient

from datalus.api import create_app
from datalus.export import (
    export_denoiser_onnx,
    quantize_int8,
    validate_int8_cfg_parity,
)
from datalus.generation.bundle import base_denoiser_for_export
from datalus.models.diffusion import TabularDiffusion
from datalus.models.nn import CategoricalHeadedDenoiser, TabularDenoiserMLP
from datalus.models.transformer import TabularTransformerDenoiser


def test_artifact_api_serves_manifest(tmp_path):
    """The API serves a domain manifest as JSON."""

    domain = tmp_path / "demo"
    domain.mkdir()
    (domain / "manifest.json").write_text(json.dumps({"name": "demo"}), encoding="utf-8")
    app = create_app(tmp_path)
    client = TestClient(app)
    response = client.get("/artifacts/demo/manifest")
    assert response.status_code == 200
    assert response.json()["name"] == "demo"


def test_artifact_api_rejects_domain_path_traversal(tmp_path):
    """The API rejects registry paths that escape the domain root."""

    app = create_app(tmp_path)
    client = TestClient(app)
    response = client.get("/artifacts/../manifest")
    assert response.status_code in {403, 404}


def test_int8_cfg_parity_guard_runs_on_small_onnx_export(tmp_path):
    """The INT8 parity guard reports CFG-amplified drift on a small export."""

    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    denoiser = TabularDenoiserMLP(d_in=3, hidden_dims=(8, 8), dim_t=8).eval()
    fp32 = export_denoiser_onnx(denoiser, tmp_path / "model_fp32.onnx", latent_dim=3)
    int8 = quantize_int8(fp32, tmp_path / "model_int8.onnx")
    parity = validate_int8_cfg_parity(fp32, int8, latent_dim=3, cfg_scale=3.0)
    assert parity["cfg_scale"] == 3.0
    assert "amplified_max_abs_diff" in parity


def test_transformer_denoiser_exports_to_onnx_with_parity(tmp_path):
    """The attention denoiser exports with the same (x_t, timestep) contract."""

    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    denoiser = TabularTransformerDenoiser(
        d_in=4,
        num_dim=2,
        cat_dims=[(5, 2)],
        dim_t=8,
        d_model=8,
        num_blocks=1,
        nhead=4,
        num_inds=4,
    ).eval()
    fp32 = export_denoiser_onnx(denoiser, tmp_path / "model_transformer.onnx", latent_dim=4)
    from datalus.export import validate_onnx_parity

    parity = validate_onnx_parity(denoiser, fp32, latent_dim=4)
    assert parity["passed"]


def test_onnx_export_unwraps_categorical_heads():
    """Export unwraps CategoricalHeadedDenoiser to the noise-only base."""

    torch.manual_seed(0)
    base = TabularDenoiserMLP(d_in=3, hidden_dims=(8, 8), dim_t=8)
    headed = CategoricalHeadedDenoiser(base, [5], 8)
    diffusion = TabularDiffusion(headed, num_timesteps=20)
    unwrapped = base_denoiser_for_export(diffusion)
    assert unwrapped is base
    x = torch.randn(2, 3)
    t = torch.tensor([1, 2])
    assert unwrapped(x, t).shape == (2, 3)
