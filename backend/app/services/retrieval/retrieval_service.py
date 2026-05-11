"""Metadata-aware retrieval service wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.schemas.evidence import EvidenceChunk


def document_to_evidence(
    doc: Document,
    index: int,
    retrieval_method: str = "hybrid",
) -> EvidenceChunk:
    """Convert a LangChain Document into a structured EvidenceChunk."""

    metadata = doc.metadata or {}
    score = metadata.get("rerank_score", metadata.get("score"))
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None

    return EvidenceChunk(
        evidence_id=f"source_{index}",
        chunk_id=metadata.get("chunk_id"),
        doc_id=metadata.get("doc_id") or metadata.get("source_document"),
        title=metadata.get("title") or metadata.get("source_document"),
        regulator=metadata.get("regulator"),
        doc_type=metadata.get("doc_type") or metadata.get("document_type"),
        issue_date=metadata.get("issue_date"),
        effective_date=metadata.get("effective_date"),
        page=metadata.get("page"),
        section_title=metadata.get("section_title"),
        hierarchy_path=metadata.get("hierarchy_path"),
        source_url=metadata.get("source_url"),
        text=doc.page_content,
        retrieval_method=retrieval_method,  # type: ignore[arg-type]
        score=score,
        metadata=metadata,
    )


def metadata_matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Return whether document metadata matches simple list/scalar filters."""

    if not filters:
        return True
    for key, expected in filters.items():
        if expected in (None, [], ""):
            continue
        actual = metadata.get(key)
        expected_values = expected if isinstance(expected, list) else [expected]
        if isinstance(actual, list):
            actual_values = actual
        elif isinstance(actual, str) and "," in actual:
            actual_values = [value.strip() for value in actual.split(",")]
        else:
            actual_values = [actual]
        normalized_actual = {str(value).lower() for value in actual_values if value is not None}
        normalized_expected = {str(value).lower() for value in expected_values if value is not None}
        if not normalized_actual.intersection(normalized_expected):
            return False
    return True


def priority_boost(metadata: dict[str, Any]) -> float:
    """Small deterministic boost for high-priority regulatory sources."""

    priority = str(metadata.get("priority", "P1")).upper()
    return {"P0": 0.12, "P1": 0.06, "P2": 0.02, "P3": 0.0}.get(priority, 0.0)


@dataclass
class RetrievalService:
    """Thin adapter around an existing LangChain retriever."""

    retriever: Any | None = None

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        retrieval_mode: str = "rag",
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        """Retrieve and normalize evidence chunks."""

        if self.retriever is None:
            return []

        docs = self.retriever.invoke(query)
        filtered_docs = [
            doc for doc in docs if metadata_matches_filters(getattr(doc, "metadata", {}) or {}, filters)
        ]
        if not filtered_docs:
            filtered_docs = docs

        def score_for(doc: Document) -> float:
            metadata = doc.metadata or {}
            base_score = metadata.get("rerank_score", metadata.get("score", 0.0))
            try:
                base = float(base_score)
            except (TypeError, ValueError):
                base = 0.0
            return base + priority_boost(metadata)

        ranked = sorted(filtered_docs, key=score_for, reverse=True)[:top_k]
        method = "graph" if retrieval_mode == "kag" else "hybrid"
        return [document_to_evidence(doc, index + 1, method) for index, doc in enumerate(ranked)]
