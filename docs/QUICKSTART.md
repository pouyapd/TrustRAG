# TrustRAG — Quickstart

Three ways to run TrustRAG, in increasing order of fidelity.

## 1. Offline mode (no API key, ~2 min)

This runs the full pipeline using a mock extractive LLM and local sentence-transformer embeddings. Useful for: smoke-testing your install, running the eval regression, demos.

```bash
git clone https://github.com/pouyapd/TrustRAG.git
cd TrustRAG
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the full evaluation pipeline offline
python scripts/run_offline_eval.py
```

Expected output: a `reports/` directory with `summary.json`, `rows.jsonl`, and `report.md`.

## 2. Local Python with a real LLM

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
pip install -r requirements.txt

# Ingest the sample corpus
python -m src.ingest --path data/documents --reset

# Start the API
uvicorn src.api.main:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for interactive API docs.

## 3. Docker (production-like)

```bash
cp .env.example .env  # set OPENAI_API_KEY
docker-compose up --build
```

This starts:
- TrustRAG API on `http://localhost:8000`
- Prometheus on `http://localhost:9090`

## Try it

### Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "doc_id": "test_001",
      "source": "test.md",
      "text": "Our refund window is 30 days for annual plans and 14 days for monthly plans."
    }],
    "reset": true
  }'
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How long is the annual refund window?"}'
```

### Run an evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": [
      {
        "question": "How long is the annual refund window?",
        "answer": "30 days",
        "relevant_doc_ids": ["test_001"]
      }
    ]
  }'
```

## Tests

```bash
pytest tests/ -v --cov=src
```
