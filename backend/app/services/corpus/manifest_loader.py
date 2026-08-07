"""Load and validate the regulatory source manifest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import get_settings
from app.schemas.corpus import SourceDocument

logger = logging.getLogger(__name__)

OFFICIAL_REGULATOR_DOMAINS = {
    "HKMA": {"hkma.gov.hk"},
    "SFC": {"sfc.hk", "apps.sfc.hk"},
    "PCPD": {"pcpd.org.hk"},
}


def validate_source_metadata(item: dict) -> None:
    """Validate release-critical provenance fields for one manifest entry."""

    url = str(item.get("source_url") or "")
    if not url.startswith("https://"):
        raise ValueError(f"{item.get('doc_id', '<unknown>')} requires an https source_url")
    hostname = (urlparse(url).hostname or "").lower()
    allowed = OFFICIAL_REGULATOR_DOMAINS.get(str(item.get("regulator", "")).upper(), set())
    if hostname not in allowed and not any(hostname.endswith("." + domain) for domain in allowed):
        raise ValueError(
            f"{item.get('doc_id', '<unknown>')} source_url must use an official regulator domain"
        )
    if item.get("status") not in {"active", "superseded", "archived"}:
        raise ValueError(f"{item.get('doc_id', '<unknown>')} has no valid status")
    if not item.get("effective_date") and not item.get("metadata_note"):
        raise ValueError(
            f"{item.get('doc_id', '<unknown>')} requires effective_date or metadata_note"
        )


def validate_manifest_release_gate(manifest_path: str | Path | None = None) -> list[SourceDocument]:
    """Load every manifest entry and fail closed on missing release metadata."""

    settings = get_settings()
    manifest = _resolve_backend_relative(manifest_path or settings.SOURCE_MANIFEST_PATH)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw_docs = raw.get("documents", []) if isinstance(raw, dict) else raw
    if not isinstance(raw_docs, list) or not raw_docs:
        raise ValueError("source manifest must contain at least one document")
    for item in raw_docs:
        validate_source_metadata(item)
        SourceDocument(**item)
    return [SourceDocument(**item) for item in raw_docs]


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
    enforce_metadata = manifest_path is None
    for item in raw_docs:
        if enforce_metadata:
            try:
                validate_source_metadata(item)
            except ValueError as exc:
                logger.error("Invalid source metadata: %s", exc)
                continue
        doc = SourceDocument(**item)
        resolved = doc_dir / doc.file_path
        if not resolved.exists():
            logger.warning("Source file not found: %s", resolved)
            continue
        doc.resolved_path = resolved
        docs.append(doc)

    logger.info("Loaded %s source manifest documents", len(docs))
    return docs
