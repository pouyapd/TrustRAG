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

# Non-root user, created before the model download so that it owns the model cache.
RUN useradd --create-home --uid 1000 appuser

# Without an OpenAI key the service embeds with the local sentence-transformers model
# (LocalEmbeddings in src/rag/providers.py), and it loads that model before it answers
# /health. Fetching it at container start took minutes on a first start, and every new
# container fetched it again, so it is downloaded once here, into the image. The service
# then runs offline by default, so startup does not depend on the network; set
# HF_HUB_OFFLINE=0 to let it fetch other models. This layer comes before the source copy,
# so a code change does not download the model again.
ENV HF_HOME=/home/appuser/.cache/huggingface
USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
USER root

COPY src/ ./src/
COPY data/documents/ ./data/documents/

RUN mkdir -p /app/data/chroma /app/reports \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Startup imports torch and loads the model from the image. With networking disabled it was
# healthy after 25-55 s on an idle laptop CPU and after 122 s while another container kept the
# CPU busy; the start period leaves room for that. Docker still reports the container healthy
# as soon as /health answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
