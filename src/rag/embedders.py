"""The embedding models the study sweeps over, named and described in one place.

Why a registry rather than an `if` chain in the experiment script: a robustness
claim is only as good as the record of what was actually compared. Every entry
here carries the exact HuggingFace repository id, the embedding dimension, the
licence, and — crucially — the query and passage prefixes the model was trained
with. A sweep that silently dropped E5's `query:` prefix would report a weaker
model rather than a different one, and the conclusion would be about our
plumbing instead of about embedding choice.

Selection criteria, applied before any result was seen:

- **CPU-feasible.** Everything here runs on CPU in minutes, not hours.
- **Genuinely different, not sibling checkpoints.** MiniLM and MPNet are both
  Sentence-Transformers models but differ in backbone, dimension and capacity;
  BGE and E5 come from independent groups (BAAI and Microsoft) with different
  training data and objectives. Three training lineages, four configurations.
- **Openly licensed and pinned.** Apache-2.0 or MIT, downloadable without
  credentials, and pinned by revision so a rerun fetches the same weights.
- **Retrieval models.** All four are trained for semantic search rather than
  being generic language-model encoders.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.rag.providers import EmbeddingProvider, HashEmbeddings, LocalEmbeddings


@dataclass(frozen=True)
class EmbedderSpec:
    """Everything a run needs to record about the embedder it used."""

    key: str
    repo_id: str
    dimension: int
    license_spdx: str
    #: Prepended to a search query before encoding. Empty for symmetric models.
    query_prefix: str = ""
    #: Prepended to a passage before encoding.
    passage_prefix: str = ""
    normalize: bool = False
    #: Pinned commit on the HuggingFace hub. `None` means "whatever main is",
    #: which is recorded honestly rather than implied to be pinned.
    revision: str | None = None
    family: str = ""
    note: str = ""

    @property
    def identity(self) -> str:
        """The string written into every run's provenance block."""
        return f"{self.repo_id}@{self.revision}" if self.revision else self.repo_id


#: The sweep. `minilm` is the baseline every previously reported number used;
#: its spec must not change, or the frozen results stop being reproducible.
EMBEDDERS: dict[str, EmbedderSpec] = {
    "minilm": EmbedderSpec(
        key="minilm",
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        license_spdx="Apache-2.0",
        family="sentence-transformers",
        note="Baseline for every previously reported figure. 6 layers, 22M parameters.",
    ),
    "mpnet": EmbedderSpec(
        key="mpnet",
        repo_id="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        license_spdx="Apache-2.0",
        family="sentence-transformers",
        note="Same family, larger backbone and double the dimension. Tests capacity.",
    ),
    "bge": EmbedderSpec(
        key="bge",
        repo_id="BAAI/bge-small-en-v1.5",
        dimension=384,
        license_spdx="MIT",
        query_prefix="Represent this sentence for searching relevant passages: ",
        normalize=True,
        family="BAAI BGE",
        note="Independent lineage. Asymmetric: the query is prefixed, the passage is not.",
    ),
    "e5": EmbedderSpec(
        key="e5",
        repo_id="intfloat/e5-small-v2",
        dimension=384,
        license_spdx="MIT",
        query_prefix="query: ",
        passage_prefix="passage: ",
        normalize=True,
        family="Microsoft E5",
        note="Independent lineage. Both sides prefixed; omitting them degrades it badly.",
    ),
    #: Not a real embedding model — a deterministic bag-of-words hash used by CI
    #: and offline smoke tests. Kept in the registry so experiments can name it
    #: the same way, never presented as a robustness data point.
    "hash": EmbedderSpec(
        key="hash",
        repo_id="HashEmbeddings(deterministic-bag-of-words)",
        dimension=384,
        license_spdx="MIT",
        family="none",
        note="Deterministic control for CI. Not a semantic model; excluded from sweeps.",
    ),
}

#: The configurations that constitute the embedder robustness experiment.
#: `hash` is deliberately absent: it is a control, not a competing model.
SWEEP_KEYS = ("minilm", "mpnet", "bge", "e5")


def get_embedder_spec(key: str) -> EmbedderSpec:
    if key not in EMBEDDERS:
        raise ValueError(f"unknown embedder {key!r}; available: {sorted(EMBEDDERS)}")
    return EMBEDDERS[key]


def build_embedder(key: str) -> tuple[EmbeddingProvider, EmbedderSpec]:
    """Construct an embedder by explicit name, with its spec for provenance.

    Named explicitly rather than resolved through `get_embeddings()`'s fallback
    chain: a service should degrade gracefully when a model is unavailable, but
    an experiment must fail loudly instead of quietly measuring a different
    model than the one it reports.
    """
    spec = get_embedder_spec(key)
    if key == "hash":
        return HashEmbeddings(dim=spec.dimension), spec
    provider = LocalEmbeddings(
        spec.repo_id,
        query_prefix=spec.query_prefix,
        passage_prefix=spec.passage_prefix,
        normalize=spec.normalize,
    )
    return provider, spec
