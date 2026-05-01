"""Integration tests for the full RAG pipeline using mock LLM."""


def test_pipeline_retrieves_and_answers(populated_pipeline):
    response = populated_pipeline.query("What is the capital of France?", top_k=2)
    assert response.answer
    assert len(response.sources) > 0
    # The Paris doc should be in the top results
    doc_ids = [s.doc_id for s in response.sources]
    assert "doc_a" in doc_ids


def test_pipeline_handles_unknown_question(populated_pipeline):
    """Pipeline should still respond, possibly with a refusal, on out-of-scope queries."""
    response = populated_pipeline.query("What is the chemical symbol for gold?", top_k=2)
    assert response.answer
    assert len(response.sources) > 0  # retrieval still happens
    # Faithfulness will typically be low for off-topic answers
    assert response.faithfulness_score is not None


def test_pipeline_returns_latency(populated_pipeline):
    response = populated_pipeline.query("Python language?", top_k=2)
    assert response.latency_ms > 0


def test_evaluation_runner_inline(populated_pipeline):
    from src.evaluation.runner import run_evaluation_inline

    dataset = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris.",
            "relevant_doc_ids": ["doc_a"],
        },
        {
            "question": "Who created Python?",
            "answer": "Guido van Rossum.",
            "relevant_doc_ids": ["doc_b"],
        },
    ]
    summary, rows = run_evaluation_inline(dataset, pipeline=populated_pipeline, top_k=2)
    assert summary["total"] == 2
    assert len(rows) == 2
    assert "failure_modes" in summary
    assert "precision_at_k_mean" in summary
