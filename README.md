# TrustRAG — Production RAG with Systematic Evaluation

[![CI](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/pouyapd/TrustRAG/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-77%25-green.svg)](tests/)

A production-ready Retrieval-Augmented Generation (RAG) system with a built-in **evaluation and failure analysis framework**. Beyond standard RAG, TrustRAG measures *when and why* the system fails — retrieval quality, hallucination rate, answer faithfulness — and exposes it through an observable, reproducible pipeline.

> **Why this exists:** Most RAG systems in production have no idea when they're wrong. TrustRAG treats evaluation as a first-class citizen, not an afterthought. See [SAMPLE_EVALUATION.md](docs/SAMPLE_EVALUATION.md) for a real evaluation run on the bundled dataset.

---

## 🎯 Key Features

- **Modular RAG pipeline** — pluggable LLMs, embedders, and vector stores (OpenAI / Anthropic / local Ollama)
- **Vector retrieval** with ChromaDB and configurable chunking strategies
- **Systematic evaluation framework** measuring:
  - Retrieval quality (Precision@k, Recall@k, MRR)
  - Answer faithfulness (grounding in retrieved context)
  - Hallucination detection
  - Answer relevance
- **Failure analysis layer** — interpretable categorization of failure modes (inspired by my prior work on neural model failure analysis)
- **Production API** — FastAPI with async, structured logging, request tracing
- **Containerized** — Docker + docker-compose for one-command deploy
- **CI/CD** — GitHub Actions pipeline with automated tests and evaluation regression checks
- **Observability** — Prometheus metrics + structured JSON logs

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │───▶│   FastAPI    │───▶│  RAG Core   │
└─────────────┘    │   Gateway    │    │  Pipeline   │
                   └──────┬───────┘    └──────┬──────┘
                          │                   │
                          ▼                   ▼
                   ┌──────────────┐    ┌─────────────┐
                   │  Prometheus  │    │  ChromaDB   │
                   │   Metrics    │    │  (vectors)  │
                   └──────────────┘    └─────────────┘
                          │                   │
                          ▼                   ▼
                   ┌─────────────────────────────────┐
                   │    Evaluation & Failure Layer   │
                   │  (faithfulness, hallucination,  │
                   │   retrieval quality, F-modes)   │
                   └─────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (recommended)

```bash
git clone https://github.com/pouyapd/TrustRAG.git
cd TrustRAG
cp .env.example .env   # add your OPENAI_API_KEY (or use Ollama for local)
docker-compose up --build
```

API will be available at `http://localhost:8000/docs`.

### Option 2: Local Python

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingest --path data/documents
uvicorn src.api.main:app --reload
```

---

## 📡 API Endpoints

| Method | Endpoint              | Description                              |
|--------|-----------------------|------------------------------------------|
| POST   | `/ingest`             | Ingest documents into the vector store   |
| POST   | `/query`              | Ask a question, get an answer + sources  |
| POST   | `/evaluate`           | Run evaluation suite on a Q/A dataset    |
| GET    | `/metrics`            | Prometheus metrics                       |
| GET    | `/health`             | Health check                             |

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

```json
{
  "answer": "The capital of France is Paris.",
  "sources": [{"doc_id": "geo_001", "score": 0.91}],
  "faithfulness_score": 0.97,
  "latency_ms": 412
}
```

---

## 📊 Evaluation Framework

TrustRAG ships with an evaluation suite that runs against a labeled Q/A dataset and produces:

- **Per-query metrics** — faithfulness, relevance, retrieval quality
- **Aggregate report** — mean / median / failure rate
- **Failure case catalog** — every failure tagged with its category (no_retrieval, wrong_retrieval, hallucination, partial_answer)
- **Markdown report** suitable for stakeholders

Run it:

```bash
python -m src.evaluation.runner --dataset data/eval/qa_test.jsonl --out reports/
```

Sample output:

```
=== TrustRAG Evaluation Report ===
Total queries:           50
Faithfulness (mean):     0.87
Answer relevance (mean): 0.91
Retrieval MRR:           0.78
Failure rate:            12% (6/50)

Failure mode breakdown:
  - hallucination:    3
  - wrong_retrieval:  2
  - no_retrieval:     1
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

CI runs unit tests + a small evaluation regression on every PR.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11, FastAPI, Pydantic v2, async/await
- **LLM/Embeddings:** OpenAI, Anthropic, Ollama (pluggable)
- **Vector store:** ChromaDB
- **Evaluation:** custom framework + RAGAS-style metrics
- **Infra:** Docker, docker-compose, GitHub Actions
- **Observability:** Prometheus, structlog

---

## 🧭 Why "TrustRAG"?

My background is in **safety-critical AI evaluation** — analyzing when and why neural models fail in autonomous robotics ([SafeTraj](https://github.com/pouyapd/SafeTraj-Prototype)). This project applies the same evaluation-first mindset to LLM systems: don't just build it — measure it, characterize its failure modes, and make them visible.

---

## 📄 License

MIT

---

## 👤 Author

**Pouya Bathaei Pourmand** — ML Engineer · Safe AI & Evaluation
[GitHub](https://github.com/pouyapd) · [LinkedIn](https://www.linkedin.com/in/pouya-pourmand-021654325)
