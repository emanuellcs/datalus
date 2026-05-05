"""Python wrapper for the DATALUS browser inference Streamlit component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parents[1] / "component" / "dist"

datalus_browser_inference = components.declare_component(
    "datalus_browser_inference",
    path=str(_COMPONENT_DIR) if _COMPONENT_DIR.exists() else None,
    url=None if _COMPONENT_DIR.exists() else "http://localhost:5173",
)


def run_browser_inference(
    artifact_base_url: str,
    schema: dict[str, Any],
    n_records: int,
    ddim_steps: int,
    seed: int,
    guidance_scale: float,
    conditions: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Invoke the React component and return generated records from the browser."""

    return datalus_browser_inference(
        artifactBaseUrl=artifact_base_url,
        schema=schema,
        nRecords=n_records,
        ddimSteps=ddim_steps,
        seed=seed,
        guidanceScale=guidance_scale,
        conditions=conditions or {},
        default=None,
    )
