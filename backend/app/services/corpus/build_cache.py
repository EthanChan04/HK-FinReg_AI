"""Build the safe JSON corpus cache from the checked-in source manifest."""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.services.corpus.cache import manifest_digest, write_corpus_cache
from app.services.corpus.corpus_ingestor import ingest_corpus_documents


def main() -> None:
    settings = get_settings()
    backend_root = Path(__file__).resolve().parents[3]
    manifest_path = backend_root / "data" / "source_manifest.json"
    cache_path = Path(settings.CORPUS_INDEX_DIR)
    if not cache_path.is_absolute():
        cache_path = backend_root / cache_path
    cache_path = cache_path / "corpus_documents.json"
    result = ingest_corpus_documents()
    if result.required_failures:
        failed_ids = ", ".join(failure.doc_id for failure in result.required_failures)
        raise RuntimeError(f"required demo sources failed ingestion: {failed_ids}")
    if not result.documents:
        raise RuntimeError("corpus ingestion produced no documents")
    write_corpus_cache(
        cache_path,
        result.documents,
        manifest_digest=manifest_digest(manifest_path),
        parser_version="hierarchy-v1",
    )
    print(
        "Built safe corpus cache: "
        f"sources_loaded={len(result.loaded_source_ids)}, "
        f"sources_failed={len(result.failures)}, "
        f"chunks={len(result.documents)}"
    )


if __name__ == "__main__":
    main()
