FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY datalus ./datalus
COPY frontend ./frontend
RUN pip install ".[frontend]"
RUN cd frontend/component && (npm ci || npm install) && npm run build

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
