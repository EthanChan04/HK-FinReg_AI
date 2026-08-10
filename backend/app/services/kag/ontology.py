"""Ontology types for the regulatory knowledge graph."""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    REGULATOR = "Regulator"
    DOCUMENT = "RegulatoryDocument"
    CLAUSE = "Clause"
    PRODUCT = "Product"
    ACTIVITY = "Activity"
    RISK = "Risk"
    OBLIGATION = "Obligation"
    CONTROL = "Control"
    USE_CASE = "UseCase"
    TOPIC = "Topic"
    EVIDENCE_CHUNK = "EvidenceChunk"
    CHAPTER = "Chapter"
    SECTION = "Section"
    DEFINITION = "Definition"
    EXCEPTION = "Exception"
    ANNEX = "Annex"
    REGULATORY_TRIPLE = "RegulatoryTriple"


class RelationType(str, Enum):
    ISSUED_BY = "ISSUED_BY"
    CONTAINS = "CONTAINS"
    APPLIES_TO = "APPLIES_TO"
    GOVERNS = "GOVERNS"
    IMPOSES = "IMPOSES"
    MITIGATES = "MITIGATES"
    REQUIRES = "REQUIRES"
    TRIGGERS = "TRIGGERS"
    SUPPORTED_BY = "SUPPORTED_BY"
    REFERENCES = "REFERENCES"
    SUPERSEDES = "SUPERSEDES"
    RELATED_TO = "RELATED_TO"
    HAS_JURISDICTION = "HAS_JURISDICTION"
    ASSERTS = "ASSERTS"


NODE_TYPES = {item.value for item in NodeType}
RELATIONS = {item.value for item in RelationType}
