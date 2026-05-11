"""Schemas for the curated Hong Kong financial regulatory corpus."""

from pathlib import Path
from typing import Literal

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
    module_tags: list[str] = Field(default_factory=list)
    file_path: str
    source_url: str | None = None
    priority: Literal["P0", "P1", "P2", "P3"] = "P1"
    language: str = "en"
    notes: str | None = None
    resolved_path: Path | None = Field(default=None, exclude=True)


class SourceManifest(BaseModel):
    """Container model for manifest validation when needed."""

    documents: list[SourceDocument]
