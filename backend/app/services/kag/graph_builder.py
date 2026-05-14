"""Build regulatory graphs from manifest metadata and extracted obligations."""

from __future__ import annotations

from pathlib import Path

from app.schemas.corpus import SourceDocument
from app.schemas.evidence import EvidenceChunk
from app.services.kag.graph_store import NetworkXGraphStore
from app.services.kag.obligation_extractor import extract_obligations_from_chunks
from app.services.kag.ontology import NodeType, RelationType


def build_graph_from_sources(
    documents: list[SourceDocument],
    evidence_chunks: list[EvidenceChunk],
    graph_path: str | Path,
) -> NetworkXGraphStore:
    """Build a metadata-first graph, preserving backward compatibility."""

    store = NetworkXGraphStore(graph_path)

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

