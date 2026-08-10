"""Schemas for the curated Hong Kong financial regulatory corpus."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """A manifest entry describing one regulatory source document."""

    doc_id: str = Field(..., description="Stable unique document id")
    title: str
    regulator: str
    jurisdiction: str = "Hong Kong"
    doc_type: str
    issue_date: str | None = None
    effective_date: str | None = None
    status: Literal["active", "superseded", "archived", "unknown"] = "active"
    sector: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    risk_tags: list[str] = Field(default_factory=list)
    module_tags: list[str] = Field(default_factory=list)
    regulatory_functions: list[str] = Field(default_factory=list)
    file_path: str
    source_url: str | None = None
    required_for_demo: bool = False
    priority: Literal["P0", "P1", "P2", "P3"] = "P1"
    language: str = "en"
    notes: str | None = None
    metadata_note: str | None = None
    supersedes: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    resolved_path: Path | None = Field(default=None, exclude=True)


class SourceManifest(BaseModel):
    """Container model for manifest validation when needed."""

    documents: list[SourceDocument]


@dataclass(frozen=True)
class CorpusIngestionFailure:
    """One source that could not produce usable corpus chunks."""

    doc_id: str
    path: str
    required: bool
    error_type: str
    message: str


@dataclass
class CorpusIngestionResult:
    """Structured outcome for a complete manifest ingestion attempt."""

    documents: list[Document]
    loaded_source_ids: list[str]
    failures: list[CorpusIngestionFailure]

    @property
    def required_failures(self) -> list[CorpusIngestionFailure]:
        return [failure for failure in self.failures if failure.required]
