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


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
