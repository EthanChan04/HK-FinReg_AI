"""RAG-specific quality evaluation metrics."""

from __future__ import annotations

import re

P0_SOURCES = {"HKMA", "SFC", "HK_IA", "HKMA_BNM"}


def _get_attr(chunk, key):
    """Get attribute from chunk, supporting both objects and dicts."""
    val = getattr(chunk, key, None)
    if val is not None:
        return val
    if isinstance(chunk, dict):
        return chunk.get(key)
    return None


def evaluate_rag_quality(
    question: str,
    evidence_chunks: list,
    expected_topics: list[str] | None = None,
) -> dict:
    """Compute metrics that measure RAG retrieval quality.

    Metrics returned
    -----------------
    *evidence_count*
        Number of evidence chunks retrieved.
    *topic_coverage*
        Fraction of *expected_topics* that appear in chunk metadata.
    *regulator_diversity*
        Number of distinct regulators found across chunks.
    *avg_score*
        Average retrieval score across chunks (``None`` if no scores).
    *has_p0_source*
        Whether any chunk originates from a P0 priority regulator.
    """
    expected_topics = expected_topics or []

    evidence_count = len(evidence_chunks)
    regulators: set[str] = set()
    scores: list[float] = []
    has_p0 = False
    topics_found: set[str] = set()

    for chunk in evidence_chunks:
        regulator = _get_attr(chunk, "regulator")
        if regulator:
            regulators.add(regulator)
            if regulator in P0_SOURCES:
                has_p0 = True

        score = _get_attr(chunk, "score")
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass

        meta = _get_attr(chunk, "metadata") or {}
        topics_raw = meta.get("topics", "")
        if isinstance(topics_raw, str):
            topics_list = [t.strip() for t in topics_raw.split(",") if t.strip()]
        else:
            topics_list = list(topics_raw)
        for topic in topics_list:
            topics_found.add(str(topic).lower())

    topic_coverage = 0.0
    if expected_topics:
        expected_lower = {t.lower() for t in expected_topics}
        matched = sum(1 for t in expected_lower if t in topics_found)
        topic_coverage = round(matched / len(expected_topics), 3)

    avg_score = round(sum(scores) / len(scores), 4) if scores else None

    return {
        "evidence_count": evidence_count,
        "topic_coverage": topic_coverage,
        "regulator_diversity": len(regulators),
        "avg_score": avg_score,
        "has_p0_source": has_p0,
    }


def compute_retrieval_precision(
    evidence_chunks: list,
    expected_doc_ids: list[str],
) -> float:
    """Compute precision of retrieved documents against ground-truth doc IDs.

    Precision = |{retrieved doc IDs} intersect {expected doc IDs}|
               / |{retrieved doc IDs}|  (0 if nothing was retrieved).
    """
    if not evidence_chunks:
        return 0.0

    expected_set = {d.lower() for d in expected_doc_ids}
    retrieved_ids: set[str] = set()

    for chunk in evidence_chunks:
        doc_id = _get_attr(chunk, "doc_id")
        if doc_id:
            retrieved_ids.add(doc_id.lower())

    if not retrieved_ids:
        return 0.0

    hits = retrieved_ids & expected_set
    return round(len(hits) / len(retrieved_ids), 3)


def _claim_tokens(claim: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", claim.casefold())
    }


def _chunk_text(chunk) -> str:
    value = _get_attr(chunk, "page_content")
    if value is not None:
        return str(value)
    value = _get_attr(chunk, "text")
    return str(value or "")


_SENTENCE_BOUNDARIES = re.compile(r"(?<=[.!?。！？])\s*")
_CJK_CHARS = re.compile(r"[\u4e00-\u9fff]")
_LATIN_WORDS = re.compile(r"[a-zA-Z]+")


def _content_units(sentence: str) -> int:
    """Count semantic content units: CJK characters + Latin words.

    A length filter based on raw characters is wrong for CJK text
    (no spaces), so faithfulness claim splitting filters on semantic
    units instead.
    """
    return len(_CJK_CHARS.findall(sentence)) + len(_LATIN_WORDS.findall(sentence))


def split_response_claims(response: str, min_content_units: int = 4) -> list[str]:
    """Split a generator response into claim-sized sentences.

    Supports English and Traditional Chinese sentence boundaries.
    Short fragments that carry no verifiable content (e.g. "Yes.", "OK.")
    are skipped.
    """
    if not response or not response.strip():
        return []
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARIES.split(response.strip())
        if sentence.strip()
    ]
    return [
        sentence
        for sentence in sentences
        if _content_units(sentence) >= min_content_units
    ]


def evaluate_generation_faithfulness(
    response: str,
    evidence_chunks: list,
) -> dict:
    """Independently measure faithfulness of a GENERATOR's actual response.

    Unlike claim_recall (which measures whether benchmark expected claims
    are retrieved), this metric extracts claims from the model's own
    response and verifies each against the retrieved context.

    Returns ``faithfulness=None`` when the response contains no verifiable
    claims (empty / noise-only response), so callers never mistake a
    missing measurement for a passing score.
    """
    claims = split_response_claims(response)
    if not claims:
        return {
            "faithfulness": None,
            "hallucination_rate": None,
            "per_claim": [],
        }

    claim_tokens = [_claim_tokens(claim) for claim in claims]
    evidence_tokens = [_claim_tokens(_chunk_text(chunk)) for chunk in evidence_chunks]

    def supported(tokens: set[str]) -> bool:
        return bool(tokens) and any(
            len(tokens & chunk) / len(tokens) >= 0.5 for chunk in evidence_tokens
        )

    per_claim = [
        {
            "claim": claim,
            "supported": supported(tokens),
            "evidence_indices": [
                index
                for index, chunk in enumerate(evidence_tokens)
                if bool(tokens) and len(tokens & chunk) / len(tokens) >= 0.5
            ],
            "reason": (
                "supported by retrieved context"
                if supported(tokens)
                else "no retrieved evidence chunk met the support threshold"
            ),
        }
        for claim, tokens in zip(claims, claim_tokens)
    ]
    supported_count = sum(item["supported"] for item in per_claim)
    faithfulness = supported_count / len(per_claim)
    return {
        "faithfulness": round(faithfulness, 3),
        "hallucination_rate": round(1.0 - faithfulness, 3),
        "per_claim": per_claim,
    }


def evaluate_claim_level_metrics(
    claims: list[str],
    evidence_chunks: list,
    noisy_evidence_chunks: list | None = None,
    generated_response: str | None = None,
) -> dict:
    """Compute deterministic claim-level retrieval and grounding diagnostics.

    ``generated_response`` (optional): the generator's actual answer text.
    When provided, ``faithfulness`` is measured independently from the
    response's own claims. When omitted, ``faithfulness`` is ``None`` --
    it is never silently reported as ``claim_recall``.
    """

    claim_tokens = [_claim_tokens(claim) for claim in claims if claim.strip()]
    evidence_tokens = [_claim_tokens(_chunk_text(chunk)) for chunk in evidence_chunks]

    def supported(tokens: set[str], chunks: list[set[str]]) -> bool:
        return bool(tokens) and any(len(tokens & chunk) / len(tokens) >= 0.5 for chunk in chunks)

    diagnostics = []
    for claim, tokens in zip((claim for claim in claims if claim.strip()), claim_tokens):
        supporting_indices = [
            index
            for index, chunk in enumerate(evidence_tokens)
            if supported(tokens, [chunk])
        ]
        diagnostics.append(
            {
                "claim": claim,
                "supported": bool(supporting_indices),
                "evidence_indices": supporting_indices,
                "reason": (
                    "token overlap met deterministic support threshold"
                    if supporting_indices
                    else "no evidence chunk met deterministic support threshold"
                ),
            }
        )

    supported_claims = sum(item["supported"] for item in diagnostics)
    claim_recall = supported_claims / len(claim_tokens) if claim_tokens else 0.0
    relevant_chunks = sum(
        any(supported(tokens, [chunk]) for tokens in claim_tokens)
        for chunk in evidence_tokens
    )
    context_precision = relevant_chunks / len(evidence_tokens) if evidence_tokens else 0.0

    # Independent generation faithfulness: measured from the generator's
    # actual response when available; otherwise None (never claim_recall).
    if generated_response is not None:
        generation_metrics = evaluate_generation_faithfulness(
            generated_response, evidence_chunks
        )
        faithfulness = generation_metrics["faithfulness"]
        hallucination_rate = generation_metrics["hallucination_rate"]
    else:
        faithfulness = None
        hallucination_rate = None

    noise_sensitivity = 0.0
    if noisy_evidence_chunks is not None:
        noisy_tokens = [_claim_tokens(_chunk_text(chunk)) for chunk in noisy_evidence_chunks]
        noisy_relevant_chunks = sum(
            any(supported(tokens, [chunk]) for tokens in claim_tokens)
            for chunk in noisy_tokens
        )
        noisy_precision = (
            noisy_relevant_chunks / len(noisy_tokens) if noisy_tokens else 0.0
        )
        noise_sensitivity = max(0.0, round(context_precision - noisy_precision, 3))

    return {
        "claim_recall": round(claim_recall, 3),
        "context_precision": round(context_precision, 3),
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
        "faithfulness_measured": generated_response is not None,
        "noise_sensitivity": noise_sensitivity,
        "context_utilization": round(claim_recall, 3),
        "claim_diagnostics": diagnostics,
    }
