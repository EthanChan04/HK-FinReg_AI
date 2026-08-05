"""T3-02: CDD conflict diagnostics (NR-02, arXiv 2605.14473).

Context-Driven Decomposition adapted as a diagnostic tool: separate the
"context answer" (grounded only in retrieved evidence) from the "prior
answer" (the model's internal knowledge) to detect when RAG knowledge
conflicts with retrieved context.

Research prototype only -- NOT a release gate. The 15% accuracy figure in
the paper comes from a worst-case misconception-injection test and must not
be generalized to all RAG scenarios.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.evaluation.rag_eval import _claim_tokens, _chunk_text


class ConflictDiagnosis(BaseModel):
    claim: str
    context_supported: bool
    prior_supported: bool
    conflict: bool
    reason: str
    evidence_indices: list[int] = Field(default_factory=list)


class ConflictReport(BaseModel):
    diagnoses: list[ConflictDiagnosis] = Field(default_factory=list)
    conflict_detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    total_claims: int = 0
    summary: str = ""


# Generic tokens that carry no regulatory signal; they inflate overlap
# between context and prior knowledge and would mask real conflicts
# (e.g. "licence from the" matching both an HKMA and an SFC requirement).
_STOPWORDS = {
    "the", "and", "or", "of", "a", "an", "is", "are", "was", "were",
    "must", "should", "shall", "may", "for", "with", "in", "on", "at",
    "to", "from", "by", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "as", "than", "not", "no", "but", "if",
}


def _distinctive_tokens(claim_tokens: set[str]) -> set[str]:
    return {token for token in claim_tokens if token not in _STOPWORDS}


def _chunk_text(chunk) -> str:
    """Extract text from a chunk: str, dict, or object with text attrs."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return str(chunk.get("page_content") or chunk.get("text") or "")
    value = getattr(chunk, "page_content", None)
    if value is not None:
        return str(value)
    return str(getattr(chunk, "text", "") or "")


def _support_status(claim_tokens: set[str], chunk_token_lists: list[set[str]]) -> bool:
    """Deterministic support check on distinctive (non-stopword) tokens.

    Stopword-free overlap is stricter than the plain token-overlap used by
    claim_recall: CDD conflict detection must distinguish between evidence
    that merely shares boilerplate and evidence that actually addresses
    the claim's substantive terms (e.g. SFC vs HKMA).
    """
    distinctive = _distinctive_tokens(claim_tokens)
    if not distinctive:
        return bool(claim_tokens) and any(
            len(claim_tokens & chunk) / len(claim_tokens) >= 0.5
            for chunk in chunk_token_lists
        )
    return any(
        len(distinctive & chunk) / len(distinctive) >= 0.5
        for chunk in chunk_token_lists
    )


def diagnose_conflicts(
    claims: list[str],
    context_chunks: list,
    prior_chunks: list,
    *,
    conflicting_claims: list[str] | None = None,
) -> ConflictReport:
    """Diagnose knowledge conflicts between context and prior evidence.

    Args:
        claims: claims extracted from the generator's response.
        context_chunks: retrieved evidence (what the system actually saw).
        prior_chunks: prior-knowledge documents (e.g. stale regulation,
            misleading summary) used to simulate the model's internal answer.
        conflicting_claims: ground truth of which claims SHOULD conflict
            (for computing detection/false-positive rates). When omitted,
            the report only describes observed conflicts.

    A conflict is detected when a claim is supported by prior knowledge but
    NOT by the retrieved context (the model may be answering from memory
    instead of the evidence).
    """
    context_tokens = [_claim_tokens(_chunk_text(c)) for c in context_chunks]
    prior_tokens = [_claim_tokens(_chunk_text(c)) for c in prior_chunks]
    ground_truth = set(conflicting_claims or [])

    diagnoses = []
    for claim in claims:
        if not claim.strip():
            continue
        tokens = _claim_tokens(claim)
        ctx_supported = _support_status(tokens, context_tokens)
        prior_supported = _support_status(tokens, prior_tokens)
        conflict = prior_supported and not ctx_supported
        indices = [
            i
            for i, chunk in enumerate(context_tokens)
            if bool(tokens) and len(tokens & chunk) / len(tokens) >= 0.5
        ]
        diagnoses.append(
            ConflictDiagnosis(
                claim=claim,
                context_supported=ctx_supported,
                prior_supported=prior_supported,
                conflict=conflict,
                reason=(
                    "conflict: prior knowledge supports claim but context does not"
                    if conflict
                    else (
                        "context answer and prior answer agree"
                        if ctx_supported and prior_supported
                        else "no conflict detected"
                    )
                ),
                evidence_indices=indices,
            )
        )

    detected = [d for d in diagnoses if d.conflict]
    detection_rate = 0.0
    false_positive_rate = 0.0
    if ground_truth:
        true_conflicts = sum(1 for d in diagnoses if d.claim in ground_truth)
        detected_true = sum(1 for d in detected if d.claim in ground_truth)
        detection_rate = detected_true / true_conflicts if true_conflicts else 0.0
        false_positives = sum(1 for d in detected if d.claim not in ground_truth)
        false_positive_rate = false_positives / len(detected) if detected else 0.0

    return ConflictReport(
        diagnoses=diagnoses,
        conflict_detection_rate=round(detection_rate, 3),
        false_positive_rate=round(false_positive_rate, 3),
        total_claims=len(diagnoses),
        summary=(
            f"{len(detected)}/{len(diagnoses)} claims conflict between prior "
            f"knowledge and retrieved context"
        ),
    )
