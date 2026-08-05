"""Safety tests for the versioned JSON corpus cache (T2-04, risk R-06).

Every unsafe or stale cache input must be rejected with an empty list so the
caller rebuilds the corpus instead of trusting corrupted data.
"""

import json

from langchain_core.documents import Document

from app.services.corpus.cache import read_corpus_cache, write_corpus_cache

MANIFEST_DIGEST = "abc123"
PARSER_VERSION = "hierarchy-v1"


def _write_valid_cache(path):
    write_corpus_cache(
        path,
        [Document(page_content="CDD obligation", metadata={"regulator": "HKMA"})],
        manifest_digest=MANIFEST_DIGEST,
        parser_version=PARSER_VERSION,
    )


def test_tampered_schema_version_returns_empty_list(tmp_path):
    cache_path = tmp_path / "corpus_documents.json"
    _write_valid_cache(cache_path)

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        read_corpus_cache(
            cache_path,
            manifest_digest=MANIFEST_DIGEST,
            parser_version=PARSER_VERSION,
        )
        == []
    )


def test_manifest_digest_mismatch_returns_empty_list(tmp_path):
    cache_path = tmp_path / "corpus_documents.json"
    _write_valid_cache(cache_path)

    # Cache was written for a different source manifest -> must be rejected.
    assert (
        read_corpus_cache(
            cache_path,
            manifest_digest="different-digest",
            parser_version=PARSER_VERSION,
        )
        == []
    )


def test_corrupted_json_returns_empty_list(tmp_path):
    cache_path = tmp_path / "corpus_documents.json"
    cache_path.write_text("{not valid json!!", encoding="utf-8")

    assert (
        read_corpus_cache(
            cache_path,
            manifest_digest=MANIFEST_DIGEST,
            parser_version=PARSER_VERSION,
        )
        == []
    )
