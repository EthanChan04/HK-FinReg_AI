"""Ingest manifest-backed regulatory PDFs into LangChain Documents."""

from __future__ import annotations

from pathlib import Path
import logging

from langchain_core.documents import Document

from app.schemas.corpus import (
    CorpusIngestionFailure,
    CorpusIngestionResult,
    SourceDocument,
)
from app.services.corpus.manifest_loader import load_source_manifest

logger = logging.getLogger(__name__)


class EmptyDocument(RuntimeError):
    """Raised when a source yields no pages or no usable chunks."""


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


def _load_source_pages(source: SourceDocument) -> list[Document]:
    if source.resolved_path is None:
        raise FileNotFoundError(f"No resolved path for {source.doc_id}")
    if not source.resolved_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source.resolved_path}")

    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(source.resolved_path)).load()


def _parse_source_pages(pages: list[Document], source: SourceDocument) -> list[Document]:
    from app.services.agents.document_parser import (
        parse_pdf_with_hierarchy,
        regulation_chunks_to_documents,
    )

    chunks = parse_pdf_with_hierarchy(pages, source_name=source.doc_id)
    return regulation_chunks_to_documents(chunks)


def ingest_corpus_documents(
    manifest_path: str | Path | None = None,
    reg_doc_dir: str | Path | None = None,
) -> CorpusIngestionResult:
    """Attempt every manifest source and report all successes and failures."""

    sources = load_source_manifest(
        manifest_path=manifest_path,
        reg_doc_dir=reg_doc_dir,
        include_missing=True,
    )
    if not sources:
        return CorpusIngestionResult(documents=[], loaded_source_ids=[], failures=[])

    all_docs: list[Document] = []
    loaded_source_ids: list[str] = []
    failures: list[CorpusIngestionFailure] = []
    for source in sources:
        try:
            pages = _load_source_pages(source)
            if not pages:
                raise EmptyDocument("PDF loader returned zero pages")
            parsed_docs = _parse_source_pages(pages, source)
            if not parsed_docs:
                raise EmptyDocument("PDF parser returned zero chunks")
            all_docs.extend(enrich_document_metadata(doc, source) for doc in parsed_docs)
            loaded_source_ids.append(source.doc_id)
        except Exception as exc:
            logger.warning("Failed to ingest source document %s: %s", source.doc_id, exc)
            failures.append(
                CorpusIngestionFailure(
                    doc_id=source.doc_id,
                    path=str(source.resolved_path or source.file_path),
                    required=source.required_for_demo,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )

    return CorpusIngestionResult(
        documents=all_docs,
        loaded_source_ids=loaded_source_ids,
        failures=failures,
    )


def load_corpus_documents(
    manifest_path: str | Path | None = None,
    reg_doc_dir: str | Path | None = None,
) -> list[Document]:
    """Compatibility wrapper returning only successfully ingested chunks."""

    return ingest_corpus_documents(
        manifest_path=manifest_path,
        reg_doc_dir=reg_doc_dir,
    ).documents
