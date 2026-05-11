"""Load and validate the regulatory source manifest."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.schemas.corpus import SourceDocument

logger = logging.getLogger(__name__)


def _resolve_backend_relative(path_value: str | Path) -> Path:
    """Resolve relative config paths from the backend working directory."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / path


def load_source_manifest(
    manifest_path: str | Path | None = None,
    reg_doc_dir: str | Path | None = None,
) -> list[SourceDocument]:
    """Load valid manifest documents whose local files exist.

    Missing manifest files and missing source PDFs are tolerated so the legacy
    single-PDF path can remain a fallback during rollout.
    """

    settings = get_settings()
    manifest = _resolve_backend_relative(manifest_path or settings.SOURCE_MANIFEST_PATH)
    doc_dir = _resolve_backend_relative(reg_doc_dir or settings.REG_DOC_DIR)

    if not manifest.exists():
        logger.warning("Source manifest not found: %s", manifest)
        return []

    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw_docs = raw.get("documents", [])
    else:
        raw_docs = raw

    docs: list[SourceDocument] = []
    for item in raw_docs:
        doc = SourceDocument(**item)
        resolved = doc_dir / doc.file_path
        if not resolved.exists():
            logger.warning("Source file not found: %s", resolved)
            continue
        doc.resolved_path = resolved
        docs.append(doc)

    logger.info("Loaded %s source manifest documents", len(docs))
    return docs
