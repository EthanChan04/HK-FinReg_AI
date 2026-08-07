"""Build the safe JSON corpus cache from the checked-in source manifest."""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.services.corpus.cache import manifest_digest, write_corpus_cache
from app.services.corpus.corpus_ingestor import load_corpus_documents


def main() -> None:
    settings = get_settings()
    backend_root = Path(__file__).resolve().parents[3]
    manifest_path = backend_root / "data" / "source_manifest.json"
    cache_path = Path(settings.CORPUS_INDEX_DIR)
    if not cache_path.is_absolute():
        cache_path = backend_root / cache_path
    cache_path = cache_path / "corpus_documents.json"
    documents = load_corpus_documents()
    if not documents:
        raise RuntimeError("corpus ingestion produced no documents")
    write_corpus_cache(
        cache_path,
        documents,
        manifest_digest=manifest_digest(manifest_path),
        parser_version="hierarchy-v1",
    )
    print(f"Built safe corpus cache: {len(documents)} chunks")


if __name__ == "__main__":
    main()
