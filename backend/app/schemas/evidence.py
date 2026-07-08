"""Structured evidence schemas shared by RAG, KAG, and DeepResearch."""

from typing import Any, Literal

from pydantic import BaseModel, Field


RetrievalMethod = Literal[
    "bm25",
    "dense",
    "hybrid",
    "rerank",
    "graph",
    "deep_research",
    "cache",
]


class EvidenceChunk(BaseModel):
    """A source-grounded chunk returned by any retrieval path."""

    evidence_id: str
    chunk_id: str | None = None
    doc_id: str | None = None
    title: str | None = None
    regulator: str | None = None
    jurisdiction: str = "Hong Kong"
    doc_type: str | None = None
    issue_date: str | None = None
    effective_date: str | None = None
    page: int | None = None
    section_title: str | None = None
    hierarchy_path: str | None = None
    source_url: str | None = None
    text: str
    retrieval_method: RetrievalMethod = "hybrid"
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    """A normalized retrieval result passed into analyzers and evaluators."""

    query: str
    retrieval_mode: Literal["rag", "kag", "deep_research"]
    evidence_chunks: list[EvidenceChunk]
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    query_plan: dict[str, Any] | None = None
    retrieval_strategy: dict[str, Any] | None = None
