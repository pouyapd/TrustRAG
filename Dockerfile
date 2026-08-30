FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for sentence-transformers / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU-only PyTorch build BEFORE the rest of the requirements.
# sentence-transformers depends on torch, and the default PyPI wheel bundles
# the entire CUDA runtime (cuBLAS, cuDNN, Triton and friends), which added
# roughly 8 GB to a CPU-only image. Installing the CPU wheel first satisfies
# the dependency, so the later resolve leaves it alone. Embeddings are
# identical: this image never had a GPU to use.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/documents/ ./data/documents/

# Non-root user
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/chroma /app/reports \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
