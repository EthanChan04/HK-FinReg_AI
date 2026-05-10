"""RAG-specific quality evaluation metrics."""

from __future__ import annotations

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
