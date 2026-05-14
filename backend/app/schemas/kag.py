"""Schemas for the lightweight regulatory knowledge graph."""

from typing import Literal

from pydantic import BaseModel, Field


GraphNodeType = Literal[
    "Regulator",
    "RegulatoryDocument",
    "Clause",
    "Topic",
    "Obligation",
    "Risk",
    "Product",
    "Activity",
    "Control",
    "UseCase",
    "EvidenceChunk",
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


class ProductProfile(BaseModel):
    product_type: str | None = None
    business_activity: list[str] = Field(default_factory=list)
    target_customers: list[str] = Field(default_factory=list)
    data_used: list[str] = Field(default_factory=list)
    ai_used: bool = False
    cross_border: bool = False
    regulated_entities: list[str] = Field(default_factory=list)


class ObligationMapRequest(BaseModel):
    query: str
    product_profile: ProductProfile | None = None


class ObligationItem(BaseModel):
    obligation: str
    regulator: str
    risk: str
    controls: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ObligationMapResponse(BaseModel):
    applicable_regulators: list[str] = Field(default_factory=list)
    applicable_products: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    obligations: list[ObligationItem] = Field(default_factory=list)
    graph_paths: list[dict] = Field(default_factory=list)
