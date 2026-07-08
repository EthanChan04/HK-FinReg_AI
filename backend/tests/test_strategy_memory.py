from pathlib import Path


def test_strategy_memory_records_scrubbed_fingerprint_without_raw_pii(tmp_path: Path):
    from app.services.retrieval.strategy_memory import StrategyExperienceStore, fingerprint_query

    path = tmp_path / "retrieval_experiences.jsonl"
    store = StrategyExperienceStore(path)

    record = store.record(
        query="CDD review for test@example.com and 51234567",
        query_traits=["cdd", "kyc"],
        strategy_id="aml_kyc_balanced_rerank",
        retrieval_mode="rag",
        bm25_weight=0.45,
        dense_weight=0.55,
        top_k=6,
        evidence_count=4,
        citation_supported_rate=0.8,
        unsupported_claim_rate=0.2,
        source_precision=0.75,
    )

    raw = path.read_text(encoding="utf-8")
    assert "test@example.com" not in raw
    assert "51234567" not in raw
    assert record.query_fingerprint == fingerprint_query("CDD review for [EMAIL REDACTED] and [PHONE REDACTED]")


def test_strategy_memory_ignores_corrupt_lines_and_finds_matching_traits(tmp_path: Path):
    from app.services.retrieval.strategy_memory import StrategyExperienceStore

    path = tmp_path / "retrieval_experiences.jsonl"
    path.write_text(
        '{"query_fingerprint":"one","query_traits":["aml","cdd"],"strategy_id":"aml_kyc_balanced_rerank",'
        '"retrieval_mode":"rag","bm25_weight":0.45,"dense_weight":0.55,"top_k":6,"evidence_count":3,'
        '"citation_supported_rate":0.9,"unsupported_claim_rate":0.1,"source_precision":0.8,'
        '"human_review_outcome":null,"created_at":"2026-07-08T00:00:00Z"}\n'
        "not-json\n",
        encoding="utf-8",
    )
    store = StrategyExperienceStore(path)

    matches = store.find_similar(["cdd", "aml"], limit=3)

    assert len(matches) == 1
    assert matches[0].strategy_id == "aml_kyc_balanced_rerank"
