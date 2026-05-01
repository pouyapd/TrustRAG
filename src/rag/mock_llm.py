"""Mock LLM for testing and offline demos.

This is a deterministic, dependency-free LLM stand-in. It produces answers
extracted directly from the retrieved context, which makes it useful as both
a test fixture and a baseline 'no-LLM' RAG mode (extractive QA).
"""
from __future__ import annotations

import re

from src.rag.providers import LLMProvider


class MockExtractiveLLM(LLMProvider):
    """Deterministic extractive LLM substitute.

    Strategy:
    - For 'generate' prompts containing CONTEXT and QUESTION, return the
      sentence in the context with the highest token overlap with the question.
    - For faithfulness scoring prompts, return a confidence based on the
      ratio of answer tokens that also appear in context.

    Useful for: unit tests, CI evaluation regression, offline demos.
    Not a substitute for a real LLM in production.
    """

    _SENT_RE = re.compile(r"(?<=[.!?])\s+")
    _TOKEN_RE = re.compile(r"\w+")

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        # Faithfulness scoring path: system prompt asks for a number
        if "Output only a number" in system or "single decimal number" in user:
            return self._score(user)
        return self._extract_answer(user)

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return set(cls._TOKEN_RE.findall(text.lower()))

    @classmethod
    def _extract_answer(cls, user_prompt: str) -> str:
        """Pull CONTEXT and QUESTION out of the prompt, return best sentence."""
        ctx_match = re.search(r"CONTEXT:\s*(.*?)\n\s*QUESTION:", user_prompt, re.DOTALL)
        q_match = re.search(r"QUESTION:\s*(.*?)\n", user_prompt, re.DOTALL)
        if not ctx_match or not q_match:
            return "I cannot answer this from the provided context."

        context = ctx_match.group(1).strip()
        question = q_match.group(1).strip()

        if not context or context == "(no context available)":
            return "I cannot answer this from the provided context."

        q_tokens = cls._tokens(question)
        if not q_tokens:
            return "I cannot answer this from the provided context."

        # Score every sentence by token overlap with the question
        sentences = [s.strip() for s in cls._SENT_RE.split(context) if s.strip()]
        if not sentences:
            return "I cannot answer this from the provided context."

        best = max(sentences, key=lambda s: len(cls._tokens(s) & q_tokens))
        if not (cls._tokens(best) & q_tokens):
            return "I cannot answer this from the provided context."
        return best

    @classmethod
    def _score(cls, user_prompt: str) -> str:
        """Estimate faithfulness as fraction of answer tokens present in context."""
        ans_match = re.search(r"ANSWER:\s*(.*?)\nCONTEXT:", user_prompt, re.DOTALL)
        ctx_match = re.search(r"CONTEXT:\s*(.*?)\n\s*SCORE:", user_prompt, re.DOTALL)
        if not ans_match or not ctx_match:
            return "0.5"

        ans_tokens = cls._tokens(ans_match.group(1))
        ctx_tokens = cls._tokens(ctx_match.group(1))
        if not ans_tokens:
            return "0.0"
        score = len(ans_tokens & ctx_tokens) / len(ans_tokens)
        return f"{score:.2f}"
