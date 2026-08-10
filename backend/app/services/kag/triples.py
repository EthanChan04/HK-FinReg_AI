"""Deterministic, provenance-first regulatory SPO extraction for the graph pilot."""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel, Field

from app.schemas.corpus import SourceDocument
from app.schemas.evidence import EvidenceChunk


class TripleSource(BaseModel):
    doc_id: str
    clause_id: str | None = None
    page: int | None = None
    official_url: str | None = None
    hierarchy_path: str | None = None


class TripleExtraction(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    extractor_version: str
    review_status: str = "pending"


class RegulatoryTriple(BaseModel):
    triple_id: str
    subject: str
    predicate: str
    object: str
    qualifiers: dict[str, str] = Field(default_factory=dict)
    source: TripleSource
    extraction: TripleExtraction


PREDICATE_BY_VERB = {
    "perform": "must_perform",
    "maintain": "must_maintain",
    "establish": "must_establish",
    "implement": "must_implement",
    "conduct": "must_conduct",
    "retain": "must_retain",
    "monitor": "must_monitor",
    "report": "must_report",
    "notify": "must_notify",
    "assess": "must_assess",
    "review": "must_review",
    "document": "must_document",
}

_SPO_PATTERN = re.compile(
    r"(?P<subject>[A-Za-z][A-Za-z0-9 /()&-]{2,}?)\s+"
    r"(?:shall|must|required to)\s+"
    r"(?P<verb>perform|maintain|establish|implement|conduct|retain|monitor|"
    r"report|notify|assess|review|document)\s+"
    r"(?P<object>[^.;]+?)(?=\s+(?:before|when|if|where|unless)\b|[.;]|$)",
    re.IGNORECASE,
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,:;")


def _stable_id(subject: str, predicate: str, object_: str, source: TripleSource) -> str:
    payload = {
        "subject": subject.casefold(),
        "predicate": predicate,
        "object": object_.casefold(),
        "doc_id": source.doc_id,
        "clause_id": source.clause_id,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"triple:{digest[:20]}"


def extract_regulatory_triples(
    evidence_chunks: list[EvidenceChunk],
    documents: list[SourceDocument],
    *,
    extractor_version: str = "spo-rule-v1",
) -> list[RegulatoryTriple]:
    """Extract small, reviewable SPO facts while retaining clause provenance."""

    docs_by_id = {doc.doc_id: doc for doc in documents}
    triples: dict[str, RegulatoryTriple] = {}
    for evidence in evidence_chunks:
        if not evidence.doc_id or evidence.doc_id not in docs_by_id:
            continue
        doc = docs_by_id[evidence.doc_id]
        source = TripleSource(
            doc_id=doc.doc_id,
            clause_id=evidence.chunk_id,
            page=evidence.page,
            official_url=doc.source_url or evidence.source_url,
            hierarchy_path=evidence.hierarchy_path,
        )
        for match in _SPO_PATTERN.finditer(evidence.text):
            subject = _clean(match.group("subject"))
            verb = match.group("verb").lower()
            object_ = _clean(match.group("object"))
            if not subject or not object_:
                continue
            predicate = PREDICATE_BY_VERB[verb]
            qualifiers = {}
            if evidence.hierarchy_path:
                qualifiers["hierarchy_path"] = evidence.hierarchy_path
            triple = RegulatoryTriple(
                triple_id=_stable_id(subject, predicate, object_, source),
                subject=subject,
                predicate=predicate,
                object=object_,
                qualifiers=qualifiers,
                source=source,
                extraction=TripleExtraction(
                    confidence=0.9,
                    extractor_version=extractor_version,
                ),
            )
            triples[triple.triple_id] = triple
    return list(triples.values())
