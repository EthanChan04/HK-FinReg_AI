"""Render structured evidence chunks into prompt context."""

from app.schemas.evidence import EvidenceChunk


def render_evidence_context(evidence_chunks: list[EvidenceChunk]) -> str:
    """Render EvidenceChunk objects in the legacy Source N prompt format."""

    parts: list[str] = []
    for index, evidence in enumerate(evidence_chunks, start=1):
        page = evidence.page if evidence.page is not None else "?"
        score = f"{evidence.score:.3f}" if isinstance(evidence.score, float) else "-"
        parts.append(
            f"[Source {index} | doc_id={evidence.doc_id} | regulator={evidence.regulator} | "
            f"p.{page} | score={score}]\n"
            f"Title: {evidence.title or '-'}\n"
            f"Section: {evidence.section_title or '-'}\n"
            f"Path: {evidence.hierarchy_path or '-'}\n\n"
            f"{evidence.text}"
        )
    return "\n\n---\n\n".join(parts)
