def test_query_classifier_routes_basic_svf_aml_to_rag():
    from app.services.retrieval.query_classifier import classify_query

    profile = classify_query("What are the CDD requirements for an SVF licensee?")

    assert profile.retrieval_mode == "rag"
    assert profile.filters["module_tags"] == ["svf"]
    assert "CDD" in profile.filters["topics"]


def test_query_classifier_routes_multi_regulator_ai_to_kag():
    from app.services.retrieval.query_classifier import classify_query

    profile = classify_query(
        "Which regulators and obligations are relevant when a virtual bank launches an AI wealth advisory product?"
    )

    assert profile.retrieval_mode == "kag"
    assert "AI" in profile.filters["topics"]


def test_query_classifier_routes_research_report_to_deep_research():
    from app.services.retrieval.query_classifier import classify_query

    profile = classify_query(
        "請分析香港虛擬銀行推出 AI 投資顧問的合規風險，並生成上線前 checklist。"
    )

    assert profile.retrieval_mode == "deep_research"
    assert profile.confidence >= 0.7


def test_query_classifier_identifies_privacy_regulator():
    from app.services.retrieval.query_classifier import classify_query

    profile = classify_query("What privacy obligations apply when using personal data in GenAI?")

    assert profile.filters["regulator"] == ["PCPD"]
    assert "privacy" in [topic.lower() for topic in profile.filters["topics"]]


def test_query_classifier_expands_ai_wealth_advisory_regulators_and_topics():
    from app.services.retrieval.query_classifier import classify_query

    profile = classify_query(
        "Which regulators and obligations are relevant when a Hong Kong virtual bank launches an AI wealth advisory product?"
    )

    assert profile.retrieval_mode == "kag"
    assert profile.filters["regulator"] == ["HKMA", "SFC", "PCPD"]
    assert "wealth_management" in profile.filters["topics"]
    assert "consumer_protection" in profile.filters["topics"]
    assert "suitability" in profile.filters["topics"]
    assert "personal_data" in profile.filters["topics"]
    assert "ai_wealth_product_launch" in profile.reasons


def test_query_classifier_keeps_deepresearch_mode_with_ai_launch_regulatory_expansion():
    from app.services.retrieval.query_classifier import classify_query

    profile = classify_query(
        "Analyze compliance risks for launching an AI investment advisor and generate a pre-launch checklist."
    )

    assert profile.retrieval_mode == "deep_research"
    assert profile.filters["regulator"] == ["HKMA", "SFC", "PCPD"]
    assert "consumer_protection" in profile.filters["topics"]
    assert "suitability" in profile.filters["topics"]
    assert "personal_data" in profile.filters["topics"]
