"""RAG pipeline: retrieve -> generate -> attach faithfulness signal."""
import time
from dataclasses import dataclass

from src.logging_setup import get_logger
from src.rag.providers import LLMProvider, get_embeddings, get_llm
from src.rag.vector_store import RetrievalResult, VectorStore

log = get_logger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using the provided context.

Rules:
- If the context does not contain the answer, say "I cannot answer this from the provided context."
- Do not invent facts. Cite sources by their [source: NAME] tag when relevant.
- Be concise and direct.
"""


FAITHFULNESS_PROMPT = """Given a QUESTION, an ANSWER, and the CONTEXT used to produce it, score from 0.0 to 1.0 how faithful the answer is to the context.

- 1.0 = every claim in the answer is directly supported by the context.
- 0.5 = some claims supported, others not.
- 0.0 = answer contradicts or is unrelated to the context.

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation.

QUESTION: {question}
ANSWER: {answer}
CONTEXT:
{context}

SCORE:"""


@dataclass
class RAGResponse:
    """Structured RAG output."""
    answer: str
    sources: list[RetrievalResult]
    faithfulness_score: float | None
    latency_ms: float


class RAGPipeline:
    """End-to-end RAG: retrieve -> prompt -> generate -> score."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.vector_store = vector_store or VectorStore(get_embeddings())

    @staticmethod
    def _format_context(results: list[RetrievalResult]) -> str:
        """Format retrieved chunks for the prompt."""
        if not results:
            return "(no context available)"
        return "\n\n".join(
            f"[source: {r.source}]\n{r.text}" for r in results
        )

    def query(
        self,
        question: str,
        top_k: int = 4,
        score_faithfulness: bool = True,
    ) -> RAGResponse:
        """Run a full RAG query."""
        t0 = time.perf_counter()

        retrieved = self.vector_store.search(question, top_k=top_k)
        context = self._format_context(retrieved)

        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        faithfulness: float | None = None
        if score_faithfulness and retrieved:
            faithfulness = self._score_faithfulness(question, answer, context)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        log.info(
            "rag_query_complete",
            question_len=len(question),
            num_retrieved=len(retrieved),
            faithfulness=faithfulness,
            latency_ms=round(latency_ms, 1),
        )

        return RAGResponse(
            answer=answer,
            sources=retrieved,
            faithfulness_score=faithfulness,
            latency_ms=latency_ms,
        )

    def _score_faithfulness(self, question: str, answer: str, context: str) -> float:
        """LLM-as-judge faithfulness score in [0, 1]."""
        try:
            prompt = FAITHFULNESS_PROMPT.format(
                question=question, answer=answer, context=context[:4000]
            )
            raw = self.llm.generate(
                "You are a strict evaluator. Output only a number.", prompt
            )
            score = float(raw.strip().split()[0])
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError) as e:
            log.warning("faithfulness_parse_failed", error=str(e), raw=raw[:50])
            return 0.0
