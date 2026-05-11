"""DeepResearch report writing helpers."""

from __future__ import annotations

from app.schemas.deepresearch import DeepResearchResult
from app.schemas.evidence import EvidenceChunk
from app.services.retrieval.citation_verifier import verify_citations


def _flatten_evidence(evidence_by_subquestion: dict[str, list[dict]]) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for entries in evidence_by_subquestion.values():
        for entry in entries:
            chunks.append(EvidenceChunk(**entry))
    for index, chunk in enumerate(chunks, start=1):
        chunk.evidence_id = f"source_{index}"
    return chunks


def write_fallback_report(
    query: str,
    research_plan: dict,
    evidence_by_subquestion: dict[str, list[dict]],
    evidence_gaps: list[dict],
) -> DeepResearchResult:
    """Create a source-grounded fallback report without an LLM call."""

    chunks = _flatten_evidence(evidence_by_subquestion)
    evidence_lines = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.page if chunk.page is not None else 0
        evidence_lines.append(
            f"- Source {index}: {chunk.title or chunk.doc_id or 'Unknown source'} "
            f"({chunk.regulator or 'Unknown regulator'}, p.{page}) [Source {index}, p.{page}]"
        )

    gap_lines = [f"- {gap.get('sub_question_id')}: {gap.get('reason')}" for gap in evidence_gaps]
    report = (
        "# DeepResearch Compliance Report\n\n"
        "## 1. Executive Summary\n"
        f"This report analyses: {query}.\n\n"
        "## 2. Regulatory Scope\n"
        + ("\n".join(evidence_lines) if evidence_lines else "- No supporting evidence retrieved.")
        + "\n\n## 3. Key Regulatory Obligations\n"
        "- Maintain governance, risk management, customer protection, and AML/CFT controls based on retrieved evidence.\n\n"
        "## 4. Risk Analysis\n"
        "- Key risks include unsupported product governance, weak monitoring, privacy misuse, and incomplete launch controls.\n\n"
        "## 5. Compliance Checklist\n"
        "- Confirm applicable regulators and licensed entity scope.\n"
        "- Map product features to AML/CFT, AI governance, consumer protection, and privacy controls.\n"
        "- Validate evidence-backed policies before launch.\n"
        "- Escalate information gaps for legal or compliance review.\n\n"
        "## 6. Information Gaps and Limitations\n"
        + ("\n".join(gap_lines) if gap_lines else "- No evidence gaps identified by the minimum coverage rule.")
        + "\n\n## 7. Source-based Evidence Table\n"
        + ("\n".join(evidence_lines) if evidence_lines else "- No evidence available.")
    )
    audit = verify_citations(report, chunks).model_dump()
    return DeepResearchResult(
        research_plan=research_plan,
        evidence_by_subquestion=evidence_by_subquestion,
        evidence_gaps=evidence_gaps,
        final_report=report,
        citation_audit=audit,
    )
