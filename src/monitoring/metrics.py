"""Prometheus metrics for observability."""
from prometheus_client import Counter, Gauge, Histogram

# Request counters
QUERIES_TOTAL = Counter(
    "trustrag_queries_total",
    "Total number of /query requests",
    labelnames=("status",),
)

INGEST_TOTAL = Counter(
    "trustrag_ingest_total",
    "Total chunks ingested",
)

# Latency
QUERY_LATENCY = Histogram(
    "trustrag_query_latency_seconds",
    "End-to-end query latency",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# Quality
FAITHFULNESS_SCORE = Histogram(
    "trustrag_faithfulness_score",
    "Faithfulness score distribution",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)

FAILURE_MODES = Counter(
    "trustrag_failure_modes_total",
    "Failure mode counts",
    labelnames=("mode",),
)

# Store size
VECTORS_IN_STORE = Gauge(
    "trustrag_vectors_in_store",
    "Number of vectors currently in the store",
)
