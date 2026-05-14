"""Deterministic source citation verification."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.schemas.evidence import EvidenceChunk


CITATION_PATTERN = re.compile(
    r"\[(?:Source:\s*)?(?:Source\s*)?(\d+),\s*p\.?\s*(\d+)\]",
    re.IGNORECASE,
)


class CitationCheck(BaseModel):
    source_number: int
    page: int
    evidence_id: str | None = None
    doc_id: str | None = None
    reason: str | None = None
    explanation: str | None = None


class CitationAudit(BaseModel):
    supported_citations: list[CitationCheck] = Field(default_factory=list)
    unsupported_citations: list[CitationCheck] = Field(default_factory=list)
    unsupported_claim_rate: float = 0.0
    summary: str = ""


def verify_citations(report_text: str, evidence_chunks: list[EvidenceChunk]) -> CitationAudit:
    """Verify source-number and page references against evidence chunks."""

    supported: list[CitationCheck] = []
    unsupported: list[CitationCheck] = []

    for source_number_raw, page_raw in CITATION_PATTERN.findall(report_text or ""):
        source_number = int(source_number_raw)
        page = int(page_raw)
        index = source_number - 1
        if index < 0 or index >= len(evidence_chunks):
            unsupported.append(
                CitationCheck(
                    source_number=source_number,
                    page=page,
                    reason="source_not_found",
                    explanation=f"Source {source_number} is not present in retrieved evidence.",
                )
            )
            continue

        evidence = evidence_chunks[index]
        if evidence.page is not None and evidence.page != page:
            unsupported.append(
                CitationCheck(
                    source_number=source_number,
                    page=page,
                    evidence_id=evidence.evidence_id,
                    doc_id=evidence.doc_id,
                    reason="page_mismatch",
                    explanation=(
                        f"Source {source_number} points to page {page}, "
                        f"but retrieved evidence is page {evidence.page}."
                    ),
                )
            )
            continue

        supported.append(
            CitationCheck(
                source_number=source_number,
                page=page,
                evidence_id=evidence.evidence_id,
                doc_id=evidence.doc_id,
                explanation=f"Source {source_number} page {page} matches retrieved evidence.",
            )
        )

    total = len(supported) + len(unsupported)
    unsupported_rate = round(len(unsupported) / total, 3) if total else 0.0
    summary = (
        f"Checked {total} citations: {len(supported)} supported, "
        f"{len(unsupported)} unsupported."
    )
    return CitationAudit(
        supported_citations=supported,
        unsupported_citations=unsupported,
        unsupported_claim_rate=unsupported_rate,
        summary=summary,
    )
