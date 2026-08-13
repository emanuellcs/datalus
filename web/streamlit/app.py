"""Streamlit interface for DATALUS."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from web.streamlit.component import run_browser_inference

st.set_page_config(page_title="DATALUS", layout="wide")

ARTIFACT_BASE_URL = os.getenv("DATALUS_ARTIFACT_BASE_URL", "http://localhost:8000/artifacts")
LOCAL_REGISTRY = Path(os.getenv("DATALUS_REGISTRY_PATH", "artifacts"))

st.title("DATALUS")
st.caption("Tabular synthetic generation with local browser execution")

domains = (
    sorted(path.name for path in LOCAL_REGISTRY.iterdir() if path.is_dir()) if LOCAL_REGISTRY.exists() else []
)
domain = st.sidebar.selectbox("Model", domains or ["datasus_sih"])
precision = st.sidebar.selectbox("Precision", ["model_int8.onnx", "model_fp32.onnx", "model_fp16.onnx"])
n_records = st.sidebar.number_input("Records", min_value=1, max_value=10_000, value=100, step=10)
ddim_steps = st.sidebar.slider("DDIM steps", min_value=10, max_value=100, value=50, step=5)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
guidance_scale = st.sidebar.slider("Guidance scale", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

schema_path = LOCAL_REGISTRY / domain / "schema_config.json"
encoder_path = LOCAL_REGISTRY / domain / "encoder_config.json"
projector_path = LOCAL_REGISTRY / domain / "projector_config.json"
manifest_path = LOCAL_REGISTRY / domain / "manifest.json"
schema = {}
encoder = {}
projector = {}
manifest = {}
if schema_path.exists():
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
if encoder_path.exists():
    encoder = json.loads(encoder_path.read_text(encoding="utf-8"))
if projector_path.exists():
    projector = json.loads(projector_path.read_text(encoding="utf-8"))
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

st.subheader("Browser inference")
st.write(
    "The ONNX model is downloaded to the browser cache and DDIM sampling runs locally on the user's device."
)

artifact_base = f"{ARTIFACT_BASE_URL.rstrip('/')}/{domain}"
if st.button("Generate synthetic data", type="primary"):
    result = run_browser_inference(
        artifact_base_url=artifact_base,
        schema=schema,
        encoder=encoder,
        projector=projector,
        manifest=manifest,
        n_records=int(n_records),
        ddim_steps=int(ddim_steps),
        seed=int(seed),
        guidance_scale=float(guidance_scale),
        conditions={"precision": precision},
    )
    if result is None:
        st.info("Waiting for the browser component response.")
    elif result.get("error"):
        st.error(result["error"])
    else:
        st.success("Generation completed in the browser.")
        st.dataframe(result.get("records", []), use_container_width=True)

with st.expander("Artifact schema"):
    st.json(schema if schema else {"message": "No local schema_config.json was found."})
