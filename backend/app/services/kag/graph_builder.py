"""Build the lightweight regulatory graph from manifest and evidence metadata."""

from __future__ import annotations

from pathlib import Path

from app.schemas.corpus import SourceDocument
from app.schemas.evidence import EvidenceChunk
from app.services.kag.graph_store import NetworkXGraphStore


def build_graph_from_sources(
    documents: list[SourceDocument],
    evidence_chunks: list[EvidenceChunk],
    graph_path: str | Path,
) -> NetworkXGraphStore:
    """Create a graph using deterministic metadata-derived relationships."""

    store = NetworkXGraphStore(graph_path)

    for doc in documents:
        doc_node = doc.doc_id
        regulator_node = f"regulator:{doc.regulator}"
        metadata = doc.model_dump(exclude={"resolved_path", "title"})
        store.add_node(doc_node, node_type="Document", title=doc.title, **metadata)
        store.add_node(regulator_node, node_type="Regulator", title=doc.regulator)
        store.add_edge(doc_node, regulator_node, relation="issued_by")

        for topic in doc.topics:
            topic_node = f"topic:{topic}"
            store.add_node(topic_node, node_type="Topic", title=topic)
            store.add_edge(doc_node, topic_node, relation="related_to")

        for tag in doc.module_tags:
            product_node = f"product:{tag}"
            store.add_node(product_node, node_type="Product", title=tag)
            store.add_edge(doc_node, product_node, relation="applies_to")

    for evidence in evidence_chunks:
        if not evidence.chunk_id or not evidence.doc_id:
            continue
        chunk_node = f"chunk:{evidence.chunk_id}"
        store.add_node(
            chunk_node,
            node_type="Chunk",
            title=evidence.section_title or evidence.chunk_id,
            page=evidence.page,
            text_preview=evidence.text[:300],
        )
        store.add_edge(evidence.doc_id, chunk_node, relation="contains")
        store.add_edge(evidence.doc_id, chunk_node, relation="supported_by", evidence_chunk_id=evidence.chunk_id)

    store.save()
    return store
