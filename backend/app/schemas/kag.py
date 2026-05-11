"""Schemas for the lightweight regulatory knowledge graph."""

from typing import Literal

from pydantic import BaseModel, Field


GraphNodeType = Literal[
    "Regulator",
    "Document",
    "Clause",
    "Topic",
    "Obligation",
    "Risk",
    "Product",
    "InstitutionType",
    "Chunk",
]


class GraphNode(BaseModel):
    node_id: str
    node_type: GraphNodeType
    title: str
    metadata: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    evidence_chunk_id: str | None = None
    metadata: dict = Field(default_factory=dict)
