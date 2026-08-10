"""Build regulatory graphs from manifest metadata and extracted obligations."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.schemas.corpus import SourceDocument
from app.schemas.evidence import EvidenceChunk
from app.services.kag.graph_store import NetworkXGraphStore
from app.services.kag.obligation_extractor import extract_obligations_from_chunks
from app.services.kag.ontology import NodeType, RelationType
from app.services.kag.triples import extract_regulatory_triples


def _structure_type(title: str) -> str:
    lowered = title.casefold()
    if lowered.startswith(("chapter ", "part ", "schedule ")):
        return NodeType.CHAPTER.value
    if lowered.startswith("section "):
        return NodeType.SECTION.value
    if lowered.startswith(("clause ", "paragraph ")):
        return NodeType.CLAUSE.value
    if lowered.startswith("definition"):
        return NodeType.DEFINITION.value
    if lowered.startswith("exception"):
        return NodeType.EXCEPTION.value
    if lowered.startswith(("annex ", "appendix ")):
        return NodeType.ANNEX.value
    return NodeType.SECTION.value


def _add_dual_graph_evidence(store: NetworkXGraphStore, documents, evidence_chunks) -> None:
    """Add document structure and provenance-rich SPO nodes to the existing graph."""

    document_by_id = {doc.doc_id: doc for doc in documents}
    structure_nodes: dict[tuple[str, str], str] = {}
    for evidence in evidence_chunks:
        if not evidence.doc_id or evidence.doc_id not in document_by_id:
            continue
        hierarchy = [part.strip() for part in (evidence.hierarchy_path or "").split(">") if part.strip()]
        parent_id = evidence.doc_id
        for index, title in enumerate(hierarchy):
            key = (evidence.doc_id, " > ".join(hierarchy[: index + 1]))
            stable_suffix = hashlib.sha256(key[1].encode("utf-8")).hexdigest()[:12]
            node_id = structure_nodes.setdefault(key, f"structure:{evidence.doc_id}:{index}:{stable_suffix}")
            if node_id not in store.graph:
                store.add_node(
                    node_id,
                    node_type=_structure_type(title),
                    title=title,
                    doc_id=evidence.doc_id,
                    hierarchy_path=key[1],
                    page=evidence.page,
                    source_url=document_by_id[evidence.doc_id].source_url,
                )
            if not store.graph.has_edge(parent_id, node_id):
                store.add_edge(parent_id, node_id, relation=RelationType.CONTAINS.value)
            parent_id = node_id
        if hierarchy:
            store.add_edge(
                parent_id,
                f"chunk:{evidence.chunk_id or evidence.evidence_id}",
                relation=RelationType.SUPPORTED_BY.value,
                page=evidence.page,
            )

    for triple in extract_regulatory_triples(evidence_chunks, documents):
        triple_node = triple.triple_id
        source = triple.source.model_dump()
        extraction = triple.extraction.model_dump()
        store.add_node(
            triple_node,
            node_type=NodeType.REGULATORY_TRIPLE.value,
            title=f"{triple.subject} {triple.predicate} {triple.object}",
            subject=triple.subject,
            predicate=triple.predicate,
            object=triple.object,
            qualifiers=triple.qualifiers,
            source=source,
            extraction=extraction,
        )
        source_node = triple.source.doc_id
        clause_key = (triple.source.doc_id, triple.source.hierarchy_path or "")
        if clause_key in structure_nodes:
            source_node = structure_nodes[clause_key]
        store.add_edge(source_node, triple_node, relation=RelationType.ASSERTS.value)
        chunk_node = f"chunk:{triple.source.clause_id}"
        if chunk_node in store.graph:
            store.add_edge(
                triple_node,
                chunk_node,
                relation=RelationType.SUPPORTED_BY.value,
                page=triple.source.page,
            )


def build_graph_from_sources(
    documents: list[SourceDocument],
    evidence_chunks: list[EvidenceChunk],
    graph_path: str | Path,
) -> NetworkXGraphStore:
    """Build a metadata-first graph, preserving backward compatibility."""

    store = NetworkXGraphStore(graph_path)
    document_ids = {doc.doc_id for doc in documents}

    for doc in documents:
        doc_node = doc.doc_id
        regulator_node = f"regulator:{doc.regulator}"
        metadata = doc.model_dump(exclude={"resolved_path", "title"})

        store.add_node(
            doc_node,
            node_type=NodeType.DOCUMENT.value,
            title=doc.title,
            **metadata,
        )
        store.add_node(
            regulator_node,
            node_type=NodeType.REGULATOR.value,
            title=doc.regulator,
        )
        store.add_edge(doc_node, regulator_node, relation=RelationType.ISSUED_BY.value)

        for referenced_doc_id in doc.references:
            if referenced_doc_id in document_ids:
                store.add_edge(
                    doc_node,
                    referenced_doc_id,
                    relation=RelationType.REFERENCES.value,
                )
        for superseded_doc_id in doc.supersedes:
            if superseded_doc_id in document_ids:
                store.add_edge(
                    doc_node,
                    superseded_doc_id,
                    relation=RelationType.SUPERSEDES.value,
                )

        for topic in doc.topics:
            topic_node = f"topic:{topic}"
            store.add_node(topic_node, node_type=NodeType.TOPIC.value, title=topic)
            store.add_edge(doc_node, topic_node, relation=RelationType.RELATED_TO.value)

        for tag in doc.module_tags:
            product_node = f"product:{tag}"
            store.add_node(product_node, node_type=NodeType.PRODUCT.value, title=tag)
            store.add_edge(doc_node, product_node, relation=RelationType.APPLIES_TO.value)

        for risk in doc.risk_tags:
            risk_node = f"risk:{risk}"
            store.add_node(risk_node, node_type=NodeType.RISK.value, title=risk)
            store.add_edge(doc_node, risk_node, relation=RelationType.RELATED_TO.value)

    for evidence in evidence_chunks:
        if not evidence.chunk_id and not evidence.evidence_id:
            continue
        if not evidence.doc_id:
            continue
        chunk_ref = evidence.chunk_id or evidence.evidence_id
        chunk_node = f"chunk:{chunk_ref}"
        store.add_node(
            chunk_node,
            node_type=NodeType.EVIDENCE_CHUNK.value,
            title=evidence.section_title or chunk_ref,
            page=evidence.page,
            text_preview=evidence.text[:300],
        )
        store.add_edge(evidence.doc_id, chunk_node, relation=RelationType.CONTAINS.value)
        store.add_edge(
            evidence.doc_id,
            chunk_node,
            relation=RelationType.SUPPORTED_BY.value,
            evidence_chunk_id=evidence.evidence_id,
        )

    _add_dual_graph_evidence(store, documents, evidence_chunks)

    store.save()
    return store


def build_obligation_graph_from_evidence(
    documents: list[SourceDocument],
    evidence_chunks: list[EvidenceChunk],
    graph_path: str | Path,
) -> NetworkXGraphStore:
    """Build an enhanced graph with obligation/risk/control nodes."""

    store = build_graph_from_sources(documents, evidence_chunks, graph_path)
    obligations = extract_obligations_from_chunks(evidence_chunks)
    doc_by_id = {doc.doc_id: doc for doc in documents}

    for item in obligations:
        obligation_node = f"obligation:{item.name}"
        risk_node = f"risk:{item.risk_type}"
        control_node = f"control:{item.control_hint}"
        chunk_node = f"chunk:{item.source_chunk_id}"

        store.add_node(obligation_node, node_type=NodeType.OBLIGATION.value, title=item.name)
        store.add_node(risk_node, node_type=NodeType.RISK.value, title=item.risk_type)
        store.add_node(control_node, node_type=NodeType.CONTROL.value, title=item.control_hint)

        store.add_edge(obligation_node, control_node, relation=RelationType.REQUIRES.value)
        store.add_edge(obligation_node, risk_node, relation=RelationType.MITIGATES.value)
        if chunk_node in store.graph:
            store.add_edge(obligation_node, chunk_node, relation=RelationType.SUPPORTED_BY.value)

        # Link back to the source document when possible.
        for evidence in evidence_chunks:
            if evidence.evidence_id != item.source_chunk_id or not evidence.doc_id:
                continue
            if evidence.doc_id in doc_by_id:
                store.add_edge(evidence.doc_id, obligation_node, relation=RelationType.IMPOSES.value)

    store.save()
    return store
