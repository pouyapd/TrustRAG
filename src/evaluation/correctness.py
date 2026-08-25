"""Normalized answer-correctness measures.

The legacy `token_overlap` in `metrics.py` is a *set*-based unigram F1 over raw
text. It is retained unchanged for reproducibility, but it is not a defensible
correctness measure: it ignores token multiplicity, punctuation, articles, and
it gives no way to distinguish an answer that is *incomplete* from one that is
*wrong*.

This module adds:

- SQuAD-style answer normalization (lowercase, strip punctuation and articles).
- Exact match and multiset token F1 over normalized text.
- "Key fact" recall — the fraction of the reference answer's *salient* tokens
  (numbers, identifiers, content words) that appear in the prediction.

Key-fact recall is what lets the v2 taxonomy separate `partial_answer` from
`incorrect_answer` with an interpretable rule: an answer that reproduces some
of the reference's facts is incomplete, one that reproduces none of them while
still asserting something is wrong. See `docs/TAXONOMY.md`.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"\w+")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")

# Punctuation maps to a *space*, not to nothing. The usual SQuAD normalizer
# deletes it, which silently merges hyphenated compounds: "30-day" becomes the
# single token "30day", which then matches neither "30" nor "day" in the
# reference. That produced a false "incomplete answer" on a verbatim-correct
# response during development, so the split is deliberate.
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

# Refusal markers. This list mirrors the private tuple in `failure_modes.py`
# on purpose: the v1 classifier is frozen for reproducibility, so the v2 stack
# carries its own public copy. `tests/test_taxonomy.py` asserts the two stay
# identical, so they cannot drift silently.
REFUSAL_MARKERS: tuple[str, ...] = (
    "cannot answer",
    "i don't know",
    "i do not know",
    "no information",
    "not contain",
    "not provided",
    "unable to answer",
)

# Function words that carry no factual content. Deliberately short: this is a
# stop list for *fact extraction*, not for retrieval, so over-pruning would
# discard real answer content.
_STOPWORDS: frozenset[str] = frozenset({
    # articles, copulas, auxiliaries
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "done", "have", "has", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # prepositions and conjunctions
    "of", "to", "in", "on", "at", "by", "for", "from", "with", "without", "into",
    "and", "or", "but", "if", "then", "than", "per", "via", "as",
    # determiners and pronouns
    "that", "this", "these", "those", "it", "its",
    "you", "your", "i", "we", "our", "they", "them", "their", "he", "she", "his", "her",
    # interrogatives
    "there", "here", "what", "which", "who", "whom", "when", "where", "why", "how",
    # polarity and quantifiers
    "not", "no", "yes", "only", "also", "more", "most", "some", "any", "all",
    "both", "each",
})


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation/articles, collapse space."""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    no_articles = _ARTICLES_RE.sub(" ", no_punct)
    return " ".join(no_articles.split())


def normalized_tokens(text: str) -> list[str]:
    """Tokens of the normalized answer, preserving multiplicity."""
    return _TOKEN_RE.findall(normalize_answer(text))


def exact_match(predicted: str, reference: str) -> float:
    """1.0 if the normalized strings are identical, else 0.0."""
    if not reference.strip():
        return 0.0
    return float(normalize_answer(predicted) == normalize_answer(reference))


def answer_precision_recall_f1(predicted: str, reference: str) -> tuple[float, float, float]:
    """Multiset token precision/recall/F1 over normalized answers.

    Unlike the legacy set-based overlap, repeated tokens count once per
    occurrence, which is the standard SQuAD definition.
    """
    pred = Counter(normalized_tokens(predicted))
    ref = Counter(normalized_tokens(reference))
    if not pred or not ref:
        return 0.0, 0.0, 0.0

    common = sum((pred & ref).values())
    if common == 0:
        return 0.0, 0.0, 0.0

    precision = common / sum(pred.values())
    recall = common / sum(ref.values())
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def normalized_answer_f1(predicted: str, reference: str) -> float:
    """Convenience wrapper returning only the F1 component."""
    return answer_precision_recall_f1(predicted, reference)[2]


def s_stem(token: str) -> str:
    """Strip a plural 's' so "days" and "day" are the same fact.

    This is the classic Harman S-stemmer, the most conservative stemmer there
    is: it only removes a trailing 's' from tokens longer than three characters
    that do not end in 'ss'. Nothing more aggressive is used, because the whole
    point of the taxonomy is that a human can predict what it will do.

    Applied to key-fact matching only. The F1 measures stay on the standard
    unstemmed SQuAD definition so they remain comparable with published work.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def key_facts(reference: str) -> set[str]:
    """Salient tokens of a reference answer.

    A token is salient when it is normalized, not a function word, and either
    contains a digit or is at least three characters long. Numbers survive
    regardless of length because quantities ("14", "29") are usually the whole
    answer in a factoid QA setting. Tokens are S-stemmed so that a singular /
    plural difference is not counted as a missing fact.
    """
    facts: set[str] = set()
    for tok in normalized_tokens(reference):
        if tok in _STOPWORDS:
            continue
        if any(ch.isdigit() for ch in tok) or len(tok) >= 3:
            facts.add(s_stem(tok))
    return facts


def key_fact_recall(predicted: str, reference: str) -> float | None:
    """Fraction of the reference's key facts that appear in the prediction.

    Returns None when the reference contains no key facts at all, so callers
    can fall back to F1 instead of dividing by zero.
    """
    facts = key_facts(reference)
    if not facts:
        return None
    pred_tokens = {s_stem(t) for t in normalized_tokens(predicted)}
    return len(facts & pred_tokens) / len(facts)


def is_refusal(answer: str) -> bool:
    """True when the answer contains an explicit refusal / abstention marker."""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def abstention_rates(
    *,
    answerable: Sequence[bool],
    abstained: Sequence[bool],
) -> dict[str, float | int | None]:
    """Abstention behaviour over a set of evaluated rows.

    - false_answer_rate: answered although the corpus cannot support an answer
      (failure to abstain). Denominator is the unanswerable questions.
    - false_refusal_rate: refused although the question is answerable.
      Denominator is the answerable questions.
    - abstention_accuracy: fraction of all questions where the decision to
      answer or abstain was correct.

    Rates whose denominator is empty are reported as None rather than 0.0, so
    an absent condition is never mistaken for a perfect score.
    """
    if len(answerable) != len(abstained):
        raise ValueError("answerable and abstained must be the same length")

    n_unanswerable = sum(1 for a in answerable if not a)
    n_answerable = sum(1 for a in answerable if a)

    false_answers = sum(1 for a, r in zip(answerable, abstained, strict=True) if not a and not r)
    false_refusals = sum(1 for a, r in zip(answerable, abstained, strict=True) if a and r)
    correct = sum(
        1
        for a, r in zip(answerable, abstained, strict=True)
        if (a and not r) or (not a and r)
    )
    total = len(answerable)

    return {
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "false_answer_rate": (false_answers / n_unanswerable) if n_unanswerable else None,
        "false_refusal_rate": (false_refusals / n_answerable) if n_answerable else None,
        "abstention_accuracy": (correct / total) if total else None,
    }
