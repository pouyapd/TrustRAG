"""Pluggable LLM and embedding providers."""
from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.config import settings
from src.logging_setup import get_logger

log = get_logger(__name__)


class LLMProvider(ABC):
    """Abstract LLM interface."""

    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Generate a completion from system + user prompts."""
        ...


class EmbeddingProvider(ABC):
    """Abstract embedding interface.

    `embed` encodes passages (the things stored in the index). `embed_query`
    encodes a search query and defaults to the same thing, which is correct for
    symmetric models. Asymmetric retrieval models — E5 and BGE both — are
    trained with distinct query and passage roles and lose measurable retrieval
    quality if a query is encoded as a passage, so they override it. Keeping the
    default delegation means every existing provider stays valid unchanged.
    """

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self.embed([text])[0]


# ---------- OpenAI ----------

class OpenAILLM(LLMProvider):
    """OpenAI chat completion provider."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return [d.embedding for d in resp.data]


# ---------- Anthropic ----------

class AnthropicLLM(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return resp.content[0].text if resp.content else ""


# ---------- Local (sentence-transformers fallback) ----------

class LocalEmbeddings(EmbeddingProvider):
    """Local sentence-transformers provider.

    Used both as the offline / no-API-key fallback for the service and as the
    experiment embedder, where the model is named explicitly rather than
    discovered through the fallback chain.

    `query_prefix` and `passage_prefix` exist because the instruction-tuned
    retrieval models are not symmetric. E5 requires literal `query:` and
    `passage:` prefixes; BGE prefixes only the query. Omitting them does not
    error — it silently costs retrieval quality, which in a robustness sweep
    would look like a property of the model rather than of how it was called.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        emb = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return emb.tolist()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


# ---------- Deterministic hash-based embedder ----------

class HashEmbeddings(EmbeddingProvider):
    """Deterministic hash-based bag-of-words embedder.

    Used in CI and offline environments where neither network access nor
    LLM API keys are available. Quality is lower than real embeddings, but
    semantics are preserved enough for retrieval to work on small corpora,
    and behaviour is fully deterministic — making evaluation regression
    reproducible across machines.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._token_re = __import__("re").compile(r"\w+")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import hashlib
        import math
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = self._token_re.findall(text.lower())
            for tok in tokens:
                # Stable per-token hash -> two indices and a sign
                h = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                idx2 = int.from_bytes(h[4:8], "big") % self.dim
                sign = 1.0 if h[8] & 1 else -1.0
                vec[idx] += sign
                vec[idx2] += sign * 0.5
            # L2 normalize so cosine similarity is well-defined
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors


# ---------- Factory ----------

def get_llm() -> LLMProvider:
    """Return the configured LLM provider."""
    provider = settings.llm_provider
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAILLM(settings.openai_api_key, settings.llm_model)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return AnthropicLLM(settings.anthropic_api_key, settings.llm_model)
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_embeddings() -> EmbeddingProvider:
    """Return the configured embedding provider.

    Resolution order:
    1. OpenAI (if provider=openai and key present)
    2. Local sentence-transformers (good quality, ~80MB download on first use)
    3. Hash-based (no network, deterministic, lower quality — used in CI)
    """
    if settings.llm_provider == "openai" and settings.openai_api_key and \
            not settings.openai_api_key.startswith("test-"):
        return OpenAIEmbeddings(settings.openai_api_key, settings.embedding_model)

    try:
        log.info("trying_local_sentence_transformer")
        return LocalEmbeddings()
    except Exception as e:
        log.info("local_embeddings_unavailable_using_hash", reason=str(e)[:80])
        return HashEmbeddings()
