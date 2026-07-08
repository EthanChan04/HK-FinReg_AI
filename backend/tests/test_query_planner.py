from app.services.retrieval.query_classifier import classify_query


def test_query_planner_expands_svf_cdd_terms_with_audit_reasons():
    from app.services.retrieval.query_planner import build_query_plan
    from app.services.retrieval.term_statistics import TermStatistics

    stats = TermStatistics(
        document_count=100,
        document_frequency={
            "stored value facility": 12,
            "customer due diligence": 9,
            "hkma": 20,
            "required": 80,
        },
    )
    profile = classify_query("What are SVF CDD requirements?")

    plan = build_query_plan("What are SVF CDD requirements?", profile=profile, term_statistics=stats)

    assert "stored value facility" in plan.expansion_terms
    assert "customer due diligence" in plan.expansion_terms
    assert "HKMA" in plan.expansion_terms
    assert "required" in plan.rejected_terms
    assert "stored value facility" in plan.bm25_query
    assert plan.dense_query == "What are SVF CDD requirements?"
    assert any("alias" in reason for reason in plan.reasons)


def test_query_planner_expands_ai_product_launch_terms():
    from app.services.retrieval.query_planner import build_query_plan
    from app.services.retrieval.term_statistics import TermStatistics

    stats = TermStatistics(
        document_count=50,
        document_frequency={
            "genai": 5,
            "product launch": 7,
            "governance": 6,
        },
    )
    profile = classify_query("AI wealth advisory product launch")

    plan = build_query_plan("AI wealth advisory product launch", profile=profile, term_statistics=stats)

    assert "GenAI" in plan.expansion_terms
    assert "product launch" in plan.expansion_terms
    assert "governance" in plan.expansion_terms
    assert {"AI", "GenAI", "ai_governance"}.issubset(set(plan.filters.get("topics", [])))
    assert {"wealth_management", "consumer_protection", "suitability", "personal_data"}.issubset(
        set(plan.filters.get("topics", []))
    )


def test_query_planner_scrubs_pii_before_audit_metadata():
    from app.services.retrieval.query_planner import build_query_plan
    from app.services.retrieval.term_statistics import TermStatistics

    plan = build_query_plan(
        "Please review CDD for test@example.com and phone 51234567",
        profile=classify_query("CDD"),
        term_statistics=TermStatistics(document_count=1, document_frequency={}),
    )

    assert "test@example.com" not in plan.scrubbed_query
    assert "51234567" not in plan.scrubbed_query
    assert "[EMAIL REDACTED]" in plan.scrubbed_query
    assert "[PHONE REDACTED]" in plan.scrubbed_query


def test_query_planner_expands_ai_wealth_regulatory_terms_from_filters():
    from app.services.retrieval.query_planner import build_query_plan
    from app.services.retrieval.term_statistics import TermStatistics

    profile = classify_query("AI wealth advisory product launch")
    stats = TermStatistics(
        document_count=100,
        document_frequency={
            "hkma": 20,
            "sfc": 18,
            "pcpd": 12,
            "consumer protection": 9,
            "suitability": 8,
            "personal data": 10,
            "wealth management": 7,
        },
    )

    plan = build_query_plan(
        "AI wealth advisory product launch",
        profile=profile,
        term_statistics=stats,
    )

    assert "HKMA" in plan.expansion_terms
    assert "SFC" in plan.expansion_terms
    assert "PCPD" in plan.expansion_terms
    assert "consumer protection" in plan.expansion_terms
    assert "suitability" in plan.expansion_terms
    assert "personal data" in plan.expansion_terms
    assert "wealth management" in plan.expansion_terms
