"""Build the persisted regulatory graph from the validated JSON corpus cache."""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.schemas.evidence import EvidenceChunk
from app.services.corpus.cache import manifest_digest, read_corpus_cache
from app.services.corpus.manifest_loader import load_source_manifest
from app.services.kag.graph_builder import build_graph_from_sources


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_graph_cache():
    """Materialize the metadata and evidence graph from the safe corpus cache."""

    settings = get_settings()
    backend_root = _backend_root()
    manifest_path = backend_root / "data" / "source_manifest.json"
    cache_path = Path(settings.CORPUS_INDEX_DIR)
    if not cache_path.is_absolute():
        cache_path = backend_root / cache_path
    cache_path = cache_path / "corpus_documents.json"
    documents = load_source_manifest()
    cached = read_corpus_cache(
        cache_path,
        manifest_digest=manifest_digest(manifest_path),
        parser_version="hierarchy-v1",
    )
    if not documents:
        raise RuntimeError("source manifest has no locally available documents")
    if not cached:
        raise RuntimeError("validated corpus cache is missing or stale")

    evidence_chunks = []
    for index, document in enumerate(cached):
        metadata = document.metadata or {}
        doc_id = metadata.get("source_document") or metadata.get("doc_id")
        if not doc_id:
            continue
        evidence_chunks.append(
            EvidenceChunk(
                evidence_id=str(metadata.get("chunk_id") or f"cache-{index}"),
                chunk_id=str(metadata.get("chunk_id") or f"cache-{index}"),
                doc_id=str(doc_id),
                title=metadata.get("title"),
                regulator=metadata.get("regulator"),
                page=metadata.get("page"),
                section_title=metadata.get("section_title"),
                hierarchy_path=metadata.get("hierarchy_path"),
                source_url=metadata.get("source_url"),
                text=document.page_content,
                metadata=metadata,
            )
        )

    graph_path = Path(settings.GRAPH_STORE_PATH)
    if not graph_path.is_absolute():
        graph_path = backend_root / graph_path
    return build_graph_from_sources(documents, evidence_chunks, graph_path)



def main() -> None:
    store = build_graph_cache()
    print(
        f"Built regulatory graph: {store.graph.number_of_nodes()} nodes, "
        f"{store.graph.number_of_edges()} edges"
    )


if __name__ == "__main__":
    main()
