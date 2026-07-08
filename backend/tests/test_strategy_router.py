from app.services.retrieval.query_classifier import classify_query


def _plan_for(query: str):
    from app.services.retrieval.query_planner import build_query_plan
    from app.services.retrieval.term_statistics import TermStatistics

    return build_query_plan(
        query,
        profile=classify_query(query),
        term_statistics=TermStatistics(document_count=1, document_frequency={}),
    )


def test_strategy_router_selects_aml_kyc_balanced_rerank():
    from app.services.retrieval.strategy_router import select_retrieval_strategy

    query = "What are SVF CDD and KYC requirements?"
    strategy = select_retrieval_strategy(classify_query(query), _plan_for(query))

    assert strategy.strategy_id == "aml_kyc_balanced_rerank"
    assert strategy.retrieval_mode == "rag"
    assert strategy.bm25_weight == 0.45
    assert strategy.dense_weight == 0.55
    assert strategy.use_rerank is True


def test_strategy_router_selects_ai_governance_kag():
    from app.services.retrieval.strategy_router import select_retrieval_strategy

    query = "Which regulators apply to an AI wealth advisory product?"
    strategy = select_retrieval_strategy(classify_query(query), _plan_for(query))

    assert strategy.strategy_id == "ai_governance_kag"
    assert strategy.retrieval_mode == "kag"
    assert strategy.use_graph is True
    assert "ai_governance" in strategy.reason_codes


def test_strategy_router_prefers_high_quality_memory_experience():
    from app.services.retrieval.strategy_memory import StrategyExperience
    from app.services.retrieval.strategy_router import select_retrieval_strategy

    query = "What are AML CDD requirements?"
    experience = StrategyExperience(
        query_fingerprint="abc",
        query_traits=["aml", "cdd"],
        strategy_id="clause_lookup_sparse_heavy",
        retrieval_mode="rag",
        bm25_weight=0.75,
        dense_weight=0.25,
        top_k=8,
        evidence_count=5,
        citation_supported_rate=0.95,
        unsupported_claim_rate=0.05,
        source_precision=0.9,
        created_at="2026-07-08T00:00:00Z",
    )

    strategy = select_retrieval_strategy(classify_query(query), _plan_for(query), experiences=[experience])

    assert strategy.strategy_id == "clause_lookup_sparse_heavy"
    assert strategy.bm25_weight == 0.75
    assert "memory_preferred" in strategy.reason_codes
