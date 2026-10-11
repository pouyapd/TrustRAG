"""The container setup: what the image must contain for the service to start offline.

The service loads its fallback embedding model before it answers /health. The Dockerfile
downloads that model at build time so that a new container neither waits for nor depends
on the network; these tests keep the two in step.
"""
import inspect
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = (REPO / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = (REPO / "docker-compose.yml").read_text(encoding="utf-8")


def _seconds(value: str) -> int:
    number, unit = re.fullmatch(r"(\d+)([sm])", value).groups()
    return int(number) * (60 if unit == "m" else 1)


def test_the_image_contains_the_services_fallback_embedding_model():
    from src.rag.providers import LocalEmbeddings

    default = inspect.signature(LocalEmbeddings.__init__).parameters["model_name"].default
    assert f"SentenceTransformer('{default}')" in DOCKERFILE


def test_the_model_is_fetched_before_the_source_is_copied_and_then_used_offline():
    download = DOCKERFILE.index("SentenceTransformer(")
    assert download < DOCKERFILE.index("COPY src/")        # a code change keeps the model layer
    assert DOCKERFILE.index("useradd") < download          # the non-root user owns the cache
    assert "HF_HUB_OFFLINE=1" in DOCKERFILE[download:]


def test_dockerfile_and_compose_allow_the_same_startup_time():
    dockerfile = _seconds(re.search(r"--start-period=(\w+)", DOCKERFILE).group(1))
    compose = _seconds(re.search(r"start_period:\s*(\w+)", COMPOSE).group(1))
    assert dockerfile == compose >= 120   # a busy first start took 122 s
