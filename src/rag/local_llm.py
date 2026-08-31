"""A real language model that runs locally on CPU, for the generation study.

Why this exists. Every retrieval and evidence result in this repository is
produced with a deterministic extractive control, which is the right baseline:
it is reproducible, needs no credentials, and cannot flatter the system. But it
cannot hallucinate, cannot refuse, and cannot be inconsistent, so it says
nothing about how a *language model* behaves when the evidence it needs is
missing from its context. That question needs a model that can actually fail in
those ways.

No API key was available in the environment this was built in, so rather than
leave the generation experiment unrun, it uses small open-weight
instruction-tuned models through `transformers` on CPU. The trade is explicit
and must be carried into every claim: these are 0.1-0.5B-parameter models, far
weaker than a frontier system. Findings here are about *whether evidence status
predicts generation failure at all*, not about how any particular deployed
model behaves. A hosted model can be swapped in through the same interface —
`scripts/run_llm_experiment.py --generator openai:gpt-4o-mini` — and the
experiment is otherwise unchanged.

Determinism: generation is greedy (`do_sample=False`), so a rerun on the same
machine reproduces the same text. Across machines, floating-point differences in
the backend can still change a token; the run records the model revision and
torch version so a divergence is diagnosable rather than mysterious.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.logging_setup import get_logger
from src.rag.providers import LLMProvider

log = get_logger(__name__)


@dataclass(frozen=True)
class LocalLLMSpec:
    """A generator configuration, recorded verbatim in the run's provenance."""

    key: str
    repo_id: str
    license_spdx: str
    params: str
    note: str = ""
    max_new_tokens: int = 96


#: Two independently trained instruction-tuned families, both openly licensed
#: and both small enough to run on CPU. They are not siblings: Qwen and SmolLM
#: come from different organisations with different data and training recipes,
#: which is what makes running both worth anything.
LOCAL_GENERATORS: dict[str, LocalLLMSpec] = {
    "qwen0.5b": LocalLLMSpec(
        key="qwen0.5b",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        license_spdx="Apache-2.0",
        params="0.5B",
        note="Alibaba Qwen2.5 instruction-tuned.",
    ),
    "smollm360m": LocalLLMSpec(
        key="smollm360m",
        repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        license_spdx="Apache-2.0",
        params="0.36B",
        note="HuggingFace SmolLM2 instruction-tuned; different lineage from Qwen.",
    ),
}


@dataclass
class LocalCausalLM(LLMProvider):
    """An instruction-tuned causal LM run locally through `transformers`.

    Loaded lazily so that importing this module — which the test suite does —
    never triggers a model download.
    """

    spec: LocalLLMSpec
    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading_local_llm", repo_id=self.spec.repo_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.repo_id, dtype=torch.float32
        )
        self._model.eval()

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Greedy generation from a chat-formatted prompt.

        `temperature` is accepted for interface compatibility and deliberately
        ignored: the experiment is run greedily so that a rerun reproduces, and
        silently sampling instead would make the results unreproducible while
        appearing to honour the argument.
        """
        import torch

        self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (ValueError, AttributeError):
            # A model without a chat template still has to produce something.
            text = f"{system}\n\n{user}\n\nANSWER:"

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=3072)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.spec.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_generator(name: str) -> tuple[LLMProvider, dict]:
    """Resolve a generator name to a provider plus the identity to record.

    Accepted forms:
      `mock`                 the deterministic extractive control (no download)
      `qwen0.5b`             a local open-weight model, no credentials needed
      `openai:MODEL`         hosted, requires OPENAI_API_KEY
      `anthropic:MODEL`      hosted, requires ANTHROPIC_API_KEY

    Hosted providers fail loudly when the key is absent rather than silently
    falling back to a different model, because a generation experiment that
    quietly changed generator would be worse than one that did not run.
    """
    if name == "mock":
        from src.rag.mock_llm import MockExtractiveLLM

        return MockExtractiveLLM(), {
            "name": "MockExtractiveLLM",
            "kind": "extractive control condition",
            "is_language_model": False,
        }

    if name in LOCAL_GENERATORS:
        spec = LOCAL_GENERATORS[name]
        return LocalCausalLM(spec), {
            "name": spec.repo_id,
            "kind": "local open-weight instruction-tuned LM",
            "is_language_model": True,
            "params": spec.params,
            "license": spec.license_spdx,
            "decoding": "greedy (do_sample=False)",
            "max_new_tokens": spec.max_new_tokens,
        }

    if ":" in name:
        provider, model = name.split(":", 1)
        if provider == "openai":
            from src.config import settings
            from src.rag.providers import OpenAILLM

            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; refusing to substitute another generator"
                )
            return OpenAILLM(settings.openai_api_key, model), {
                "name": model, "kind": "hosted API", "is_language_model": True,
                "provider": "openai", "decoding": "temperature=0",
            }
        if provider == "anthropic":
            from src.config import settings
            from src.rag.providers import AnthropicLLM

            if not settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set; refusing to substitute another generator"
                )
            return AnthropicLLM(settings.anthropic_api_key, model), {
                "name": model, "kind": "hosted API", "is_language_model": True,
                "provider": "anthropic", "decoding": "temperature=0",
            }

    raise ValueError(
        f"unknown generator {name!r}; expected 'mock', one of {sorted(LOCAL_GENERATORS)}, "
        "or 'openai:MODEL' / 'anthropic:MODEL'"
    )
