"""Ingest manifest-backed regulatory PDFs into LangChain Documents."""

from __future__ import annotations

from pathlib import Path
import logging

from langchain_core.documents import Document

from app.schemas.corpus import SourceDocument
from app.services.corpus.manifest_loader import load_source_manifest

logger = logging.getLogger(__name__)


def _metadata_value(value):
    """Keep Chroma-compatible scalar metadata while preserving searchability."""

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value


def _document_metadata(source: SourceDocument) -> dict:
    raw = source.model_dump(exclude={"resolved_path"})
    return {
        key: _metadata_value(value)
        for key, value in raw.items()
        if value is not None and value != "" and value != []
    }


def enrich_document_metadata(doc: Document, source: SourceDocument) -> Document:
    """Attach manifest metadata to one parsed chunk."""

    metadata = {**(doc.metadata or {}), **_document_metadata(source)}
    metadata.setdefault("source_document", source.doc_id)
    metadata.setdefault("title", source.title)
    if "chunk_id" not in metadata:
        page = metadata.get("page", "unknown")
        metadata["chunk_id"] = f"{source.doc_id}_p{page}_{abs(hash(doc.page_content)) % 1000000}"
    return Document(page_content=doc.page_content, metadata=metadata)


def load_corpus_documents(
    manifest_path: str | Path | None = None,
    reg_doc_dir: str | Path | None = None,
) -> list[Document]:
    """Load all manifest PDFs and return metadata-enriched chunks."""

    sources = load_source_manifest(manifest_path=manifest_path, reg_doc_dir=reg_doc_dir)
    if not sources:
        return []

    from langchain_community.document_loaders import PyPDFLoader
    from app.services.agents.document_parser import (
        parse_pdf_with_hierarchy,
        regulation_chunks_to_documents,
    )

    all_docs: list[Document] = []
    for source in sources:
        if source.resolved_path is None:
            continue
        try:
            loader = PyPDFLoader(str(source.resolved_path))
            pages = loader.load()
            chunks = parse_pdf_with_hierarchy(pages, source_name=source.doc_id)
            parsed_docs = regulation_chunks_to_documents(chunks)
            all_docs.extend(enrich_document_metadata(doc, source) for doc in parsed_docs)
        except Exception as exc:
            logger.warning("Failed to ingest source document %s: %s", source.doc_id, exc)
            continue

    return all_docs
