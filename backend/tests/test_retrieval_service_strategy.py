from langchain_core.documents import Document


class RecordingRetriever:
    def __init__(self):
        self.last_query = None

    def invoke(self, query):
        self.last_query = query
        return [
            Document(
                page_content="CDD evidence",
                metadata={"doc_id": "doc_1", "regulator": "HKMA", "score": 0.7},
            )
        ]


def test_retrieval_service_uses_query_plan_bm25_query_and_attaches_audit_metadata():
    from app.services.retrieval.query_classifier import classify_query
    from app.services.retrieval.query_planner import build_query_plan
    from app.services.retrieval.retrieval_service import RetrievalService
    from app.services.retrieval.strategy_router import select_retrieval_strategy

    query = "What are SVF CDD requirements?"
    profile = classify_query(query)
    plan = build_query_plan(query, profile=profile)
    strategy = select_retrieval_strategy(profile, plan)
    retriever = RecordingRetriever()

    evidence = RetrievalService(retriever=retriever).retrieve(query, query_plan=plan, strategy=strategy)

    assert retriever.last_query == plan.bm25_query
    assert "stored value facility" in retriever.last_query
    assert evidence[0].metadata["query_plan"]["query_plan_id"] == plan.query_plan_id
    assert evidence[0].metadata["retrieval_strategy"]["strategy_id"] == strategy.strategy_id
