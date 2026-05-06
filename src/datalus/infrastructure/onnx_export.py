"""ONNX export and quantization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ONNXDenoiserWrapper(nn.Module):
    """Export-friendly wrapper around the denoiser."""

    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x_t, timestep, None)


def export_denoiser_onnx(
    denoiser: nn.Module,
    output_path: str | Path,
    latent_dim: int,
    opset_version: int = 17,
) -> Path:
    """Export the PyTorch denoiser with dynamic batch axes."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ONNXDenoiserWrapper(denoiser).eval()
    device = next(wrapper.parameters()).device
    dummy_x = torch.randn(2, latent_dim, device=device, dtype=torch.float32)
    dummy_t = torch.zeros(2, device=device, dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_t),
        str(output),
        input_names=["x_t", "timestep"],
        output_names=["predicted_noise"],
        dynamic_axes={
            "x_t": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "predicted_noise": {0: "batch_size"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    return output


def quantize_int8(fp32_path: str | Path, output_path: str | Path) -> Path:
    """Apply ONNX Runtime dynamic INT8 quantization."""

    from onnxruntime.quantization import QuantType, quantize_dynamic

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(fp32_path), str(output), weight_type=QuantType.QInt8)
    return output


def validate_onnx_parity(
    denoiser: nn.Module,
    onnx_path: str | Path,
    latent_dim: int,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare ONNX Runtime output against PyTorch for a fixed input."""

    import onnxruntime as ort

    denoiser.eval()
    device = next(denoiser.parameters()).device
    x = torch.randn(4, latent_dim, device=device)
    t = torch.randint(0, 1_000, (4,), device=device, dtype=torch.long)
    with torch.no_grad():
        torch_out = denoiser(x, t, None).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"x_t": x.cpu().numpy(), "timestep": t.cpu().numpy()})[
        0
    ]
    max_abs = float(np.max(np.abs(torch_out - ort_out)))
    return {
        "max_abs_diff": max_abs,
        "passed": bool(np.allclose(torch_out, ort_out, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def validate_int8_cfg_parity(
    fp32_path: str | Path,
    int8_path: str | Path,
    latent_dim: int,
    cfg_scale: float = 3.0,
    atol: float = 2e-1,
    categorical_slices: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Bound quantization drift after high-scale CFG noise amplification."""

    import onnxruntime as ort

    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, latent_dim)).astype(np.float32)
    t = rng.integers(0, 1_000, size=(8,), dtype=np.int64)
    fp32_session = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    int8_session = ort.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"]
    )
    feeds = {
        fp32_session.get_inputs()[0].name: x,
        fp32_session.get_inputs()[1].name: t,
    }
    fp32 = fp32_session.run(None, feeds)[0]
    feeds_int8 = {
        int8_session.get_inputs()[0].name: x,
        int8_session.get_inputs()[1].name: t,
    }
    int8 = int8_session.run(None, feeds_int8)[0]

    # CFG computes eps_uncond + w * (eps_cond - eps_uncond). Export artifacts may
    # not include a context input, so the parity guard measures the worst-case
    # amplification of raw epsilon drift under the same guidance scale.
    amplified_abs = float(np.max(np.abs((fp32 - int8) * cfg_scale)))
    categorical_agreement: float | None = None
    if categorical_slices:
        agreements: list[float] = []
        for start, stop in categorical_slices:
            fp32_idx = np.argmax(fp32[:, start:stop], axis=1)
            int8_idx = np.argmax(int8[:, start:stop], axis=1)
            agreements.append(float(np.mean(fp32_idx == int8_idx)))
        categorical_agreement = float(np.mean(agreements)) if agreements else None
    return {
        "cfg_scale": cfg_scale,
        "amplified_max_abs_diff": amplified_abs,
        "categorical_agreement": categorical_agreement,
        "passed": bool(amplified_abs <= atol),
        "atol": atol,
    }


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
