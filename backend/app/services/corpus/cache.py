"""Safe, versioned JSON cache for parsed corpus documents."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1


def manifest_digest(path: str | Path) -> str:
    """Return the SHA-256 digest used to bind a cache to its manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        return "missing"
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def write_corpus_cache(
    path: str | Path,
    documents: Iterable[Document],
    *,
    manifest_digest: str,
    parser_version: str,
) -> None:
    """Write only JSON-compatible document data with an explicit schema."""

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "manifest_digest": manifest_digest,
        "parser_version": parser_version,
        "documents": [
            {
                "page_content": str(document.page_content),
                "metadata": dict(document.metadata or {}),
            }
            for document in documents
        ],
    }
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(cache_path)


def read_corpus_cache(
    path: str | Path,
    *,
    manifest_digest: str,
    parser_version: str,
) -> list[Document]:
    """Read a validated cache; return an empty list for any unsafe/stale input."""

    cache_path = Path(path)
    if not cache_path.exists():
        return []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("cache root must be an object")
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported cache schema version")
        if payload.get("manifest_digest") != manifest_digest:
            raise ValueError("cache manifest digest mismatch")
        if payload.get("parser_version") != parser_version:
            raise ValueError("cache parser version mismatch")
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            raise ValueError("cache documents must be a list")

        documents: list[Document] = []
        for item in raw_documents:
            if not isinstance(item, dict):
                raise ValueError("cache document must be an object")
            page_content = item.get("page_content")
            metadata = item.get("metadata")
            if not isinstance(page_content, str) or not isinstance(metadata, dict):
                raise ValueError("cache document has invalid fields")
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Corpus cache rejected and will be rebuilt: %s", exc)
        return []
