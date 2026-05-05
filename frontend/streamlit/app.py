"""Streamlit interface for DATALUS with pt-BR visible text."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from frontend.streamlit.component import run_browser_inference

st.set_page_config(page_title="DATALUS", layout="wide")

ARTIFACT_BASE_URL = os.getenv(
    "DATALUS_ARTIFACT_BASE_URL", "http://localhost:8000/artifacts"
)
LOCAL_REGISTRY = Path(os.getenv("DATALUS_REGISTRY_PATH", "artifacts"))

st.title("DATALUS")
st.caption("Geração sintética tabular com execução local no navegador")

domains = (
    sorted(path.name for path in LOCAL_REGISTRY.iterdir() if path.is_dir())
    if LOCAL_REGISTRY.exists()
    else []
)
domain = st.sidebar.selectbox("Modelo", domains or ["datasus_sih"])
precision = st.sidebar.selectbox(
    "Precisão", ["model_int8.onnx", "model_fp32.onnx", "model_fp16.onnx"]
)
n_records = st.sidebar.number_input(
    "Registros", min_value=1, max_value=10_000, value=100, step=10
)
ddim_steps = st.sidebar.slider(
    "Passos DDIM", min_value=10, max_value=100, value=50, step=5
)
seed = st.sidebar.number_input(
    "Semente", min_value=0, max_value=2_147_483_647, value=42, step=1
)
guidance_scale = st.sidebar.slider(
    "Escala de orientação", min_value=1.0, max_value=5.0, value=2.0, step=0.1
)

schema_path = LOCAL_REGISTRY / domain / "schema_config.json"
schema = {}
if schema_path.exists():
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

st.subheader("Inferência no navegador")
st.write(
    "O modelo ONNX é baixado para o cache do navegador e a amostragem DDIM roda localmente no dispositivo do usuário."
)

artifact_base = f"{ARTIFACT_BASE_URL.rstrip('/')}/{domain}"
if st.button("Gerar dados sintéticos", type="primary"):
    result = run_browser_inference(
        artifact_base_url=artifact_base,
        schema=schema,
        n_records=int(n_records),
        ddim_steps=int(ddim_steps),
        seed=int(seed),
        guidance_scale=float(guidance_scale),
        conditions={"precision": precision},
    )
    if result is None:
        st.info("Aguardando retorno do componente do navegador.")
    elif result.get("error"):
        st.error(result["error"])
    else:
        st.success("Geração concluída no navegador.")
        st.dataframe(result.get("records", []), use_container_width=True)

with st.expander("Esquema do artefato"):
    st.json(
        schema
        if schema
        else {"mensagem": "Nenhum schema_config.json local foi encontrado."}
    )
