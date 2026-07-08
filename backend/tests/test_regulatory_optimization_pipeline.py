from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document


AI_WEALTH_QUERY = (
    "Which regulators and obligations are relevant when a Hong Kong virtual bank "
    "launches an AI wealth advisory product?"
)


def _term_stats():
    from app.services.retrieval.term_statistics import TermStatistics

    return TermStatistics(
        document_count=100,
        document_frequency={
            "ai": 30,
            "genai": 12,
            "governance": 18,
            "product launch": 9,
            "hkma": 20,
            "sfc": 18,
            "pcpd": 12,
            "consumer protection": 9,
            "suitability": 8,
            "personal data": 10,
            "wealth management": 7,
        },
    )


def _regulator_evidence_rows():
    return [
        {"evidence_id": "p1", "regulator": "PCPD", "text": "PCPD AI personal data guidance"},
        {"evidence_id": "p2", "regulator": "PCPD", "text": "PCPD privacy governance"},
        {"evidence_id": "h1", "regulator": "HKMA", "text": "HKMA AI governance expectations"},
        {"evidence_id": "s1", "regulator": "SFC", "text": "SFC suitability obligations"},
    ]


def test_ai_wealth_regulatory_pipeline_is_deterministic_under_concurrency():
    from app.services.deepresearch.workflow import _apply_regulator_diversity_gate
    from app.services.retrieval.query_classifier import classify_query
    from app.services.retrieval.query_planner import build_query_plan

    def classify_plan_and_gate() -> tuple:
        profile = classify_query(AI_WEALTH_QUERY)
        plan = build_query_plan(AI_WEALTH_QUERY, profile=profile, term_statistics=_term_stats())
        selected = _apply_regulator_diversity_gate(
            _regulator_evidence_rows(),
            required_regulators=profile.filters["regulator"],
            top_k=3,
        )
        return (
            profile.retrieval_mode,
            tuple(profile.filters["regulator"]),
            tuple(profile.filters["topics"]),
            tuple(plan.expansion_terms),
            tuple(item["regulator"] for item in selected),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: classify_plan_and_gate(), range(40)))

    assert len(set(results)) == 1
    mode, regulators, topics, expansions, selected_regulators = results[0]
    assert mode == "kag"
    assert regulators == ("HKMA", "SFC", "PCPD")
    assert {"consumer_protection", "suitability", "personal_data"}.issubset(topics)
    assert {"HKMA", "SFC", "PCPD", "consumer protection", "suitability", "personal data"}.issubset(
        expansions
    )
    assert selected_regulators == ("HKMA", "SFC", "PCPD")


def test_ai_wealth_retrieval_recalls_diverse_regulator_evidence_before_generation():
    from app.services.deepresearch.workflow import _apply_regulator_diversity_gate
    from app.services.retrieval.query_classifier import classify_query
    from app.services.retrieval.query_planner import build_query_plan
    from app.services.retrieval.retrieval_service import RetrievalService
    from app.services.retrieval.strategy_router import select_retrieval_strategy

    class StaticRetriever:
        def invoke(self, query: str):
            return [
                Document(
                    page_content="PCPD personal data and AI governance guidance.",
                    metadata={"regulator": "PCPD", "topics": ["AI"], "score": 0.99, "page": 1, "title": "PCPD AI"},
                ),
                Document(
                    page_content="PCPD privacy risk management for AI systems.",
                    metadata={"regulator": "PCPD", "topics": ["AI"], "score": 0.98, "page": 2, "title": "PCPD Privacy"},
                ),
                Document(
                    page_content="PCPD data protection expectations for AI advisory onboarding.",
                    metadata={"regulator": "PCPD", "topics": ["AI"], "score": 0.97, "page": 3, "title": "PCPD Data"},
                ),
                Document(
                    page_content="HKMA governance and risk management expectations for AI product launch.",
                    metadata={"regulator": "HKMA", "topics": ["AI"], "score": 0.5, "page": 4, "title": "HKMA AI"},
                ),
                Document(
                    page_content="SFC suitability obligations for wealth advisory and investment recommendations.",
                    metadata={"regulator": "SFC", "topics": ["AI"], "score": 0.4, "page": 5, "title": "SFC Suitability"},
                ),
            ]

    profile = classify_query(AI_WEALTH_QUERY)
    plan = build_query_plan(AI_WEALTH_QUERY, profile=profile, term_statistics=_term_stats())
    strategy = select_retrieval_strategy(profile, plan)
    evidence = RetrievalService(StaticRetriever()).retrieve(
        AI_WEALTH_QUERY,
        filters={"regulator": profile.filters["regulator"]},
        retrieval_mode=strategy.retrieval_mode,
        top_k=5,
        query_plan=plan,
        strategy=strategy,
    )

    raw_regulators = [chunk.regulator for chunk in evidence]
    selected = _apply_regulator_diversity_gate(
        [chunk.model_dump() for chunk in evidence],
        required_regulators=profile.filters["regulator"],
        top_k=3,
    )

    assert raw_regulators[:3] == ["PCPD", "PCPD", "PCPD"]
    assert {chunk.regulator for chunk in evidence} == {"HKMA", "SFC", "PCPD"}
    assert [item["regulator"] for item in selected] == ["HKMA", "SFC", "PCPD"]
    assert evidence[0].metadata["query_plan"]["expansion_terms"]
    assert evidence[0].metadata["retrieval_strategy"]["strategy_id"] == "ai_governance_kag"


def test_generated_deepresearch_report_preserves_regulator_diversity_and_citations():
    from app.services.deepresearch.report_writer import write_fallback_report

    evidence_by_subquestion = {
        "SQ1": [
            {
                "evidence_id": "h1",
                "doc_id": "hkma_ai",
                "title": "HKMA AI Governance",
                "regulator": "HKMA",
                "page": 4,
                "text": "HKMA governance and risk management expectations for AI product launch.",
            },
            {
                "evidence_id": "s1",
                "doc_id": "sfc_suitability",
                "title": "SFC Suitability",
                "regulator": "SFC",
                "page": 5,
                "text": "SFC suitability obligations for wealth advisory and investment recommendations.",
            },
            {
                "evidence_id": "p1",
                "doc_id": "pcpd_data",
                "title": "PCPD Personal Data",
                "regulator": "PCPD",
                "page": 1,
                "text": "PCPD personal data and AI governance guidance.",
            },
        ]
    }

    result = write_fallback_report(
        query="AI wealth advisory launch",
        research_plan={"research_goal": "AI wealth advisory launch"},
        evidence_by_subquestion=evidence_by_subquestion,
        evidence_gaps=[],
    )

    assert "Compliance Checklist" in result.final_report
    assert "HKMA AI Governance (HKMA, p.4)" in result.final_report
    assert "SFC Suitability (SFC, p.5)" in result.final_report
    assert "PCPD Personal Data (PCPD, p.1)" in result.final_report
    assert result.citation_audit["unsupported_claim_rate"] == 0.0
    assert len(result.citation_audit["supported_citations"]) == 6
