"""LangGraph DeepResearch workflow with real retrieval, gap loop, and citation verification."""

from __future__ import annotations

import logging
from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.schemas.deepresearch import ResearchPlan, ResearchRequest
from app.schemas.evidence import EvidenceChunk
from app.services.agents.builder import build_reranked_retriever
from app.services.deepresearch.evidence_evaluator import evaluate_evidence_coverage
from app.services.deepresearch.planner import build_research_plan
from app.services.deepresearch.report_writer import write_fallback_report
from app.services.retrieval.citation_verifier import verify_citations
from app.services.retrieval.query_classifier import classify_query
from app.services.retrieval.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class DeepResearchState(TypedDict, total=False):
    original_query: str
    request: dict
    research_plan: dict
    evidence_by_subquestion: Dict[str, List[dict]]
    evidence_gaps: List[dict]
    iteration: int
    final_report: str
    citation_audit: dict


def _synthetic_evidence(question: str, sub_question_id: str) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=f"source_{sub_question_id}",
        chunk_id=f"chunk_{sub_question_id}",
        doc_id="deepresearch_fallback",
        title="Fallback DeepResearch Evidence",
        regulator="HKMA",
        page=0,
        section_title=sub_question_id,
        text=f"Fallback evidence collected for: {question}",
        retrieval_method="deep_research",
    )


def _build_retrieval_service() -> RetrievalService | None:
    """Build the RetrievalService, returning None if no retriever is available."""
    try:
        base_retriever = build_reranked_retriever()
        if base_retriever is None:
            logger.warning("DeepResearch: no retriever available, falling back to synthetic evidence")
            return None
        return RetrievalService(retriever=base_retriever)
    except Exception as exc:
        logger.warning("DeepResearch: failed to build retriever (%s), falling back to synthetic evidence", exc)
        return None


def _retrieve_for_sub_question(
    retrieval_service: RetrievalService | None,
    question: str,
    sub_question_id: str,
    retrieval_mode: str,
    evidence_min_count: int,
) -> list[dict]:
    """Retrieve evidence for a single sub-question, falling back to synthetic on failure."""
    if retrieval_service is None:
        return [_synthetic_evidence(question, sub_question_id).model_dump()]

    try:
        profile = classify_query(question)
        top_k = evidence_min_count + 3
        evidence = retrieval_service.retrieve(
            query=question,
            filters=profile.filters,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
        )
        if evidence:
            return [chunk.model_dump() for chunk in evidence]
        logger.warning("DeepResearch: empty evidence for sub-question %s, using synthetic", sub_question_id)
        return [_synthetic_evidence(question, sub_question_id).model_dump()]
    except Exception as exc:
        logger.warning("DeepResearch: retrieval failed for sub-question %s (%s), using synthetic", sub_question_id, exc)
        return [_synthetic_evidence(question, sub_question_id).model_dump()]


def build_deepresearch_graph():
    """Build a DeepResearch graph with real retrieval, gap-retrieval loop, and citation verification.

    Graph structure:
        planner -> retrieval -> evidence_evaluator
            -> [gap_retriever <-> evidence_evaluator] (loop, up to max iterations)
            -> report_writer -> citation_verifier -> END
    """
    settings = get_settings()
    max_iterations = settings.DEEP_RESEARCH_MAX_ITERATIONS

    def planner_node(state: DeepResearchState):
        request_payload = state.get("request") or {"query": state["original_query"]}
        request = ResearchRequest(**request_payload)
        plan = build_research_plan(state["original_query"], request=request)
        return {"research_plan": plan.model_dump(), "iteration": 0}

    def retrieval_node(state: DeepResearchState):
        plan = ResearchPlan(**state["research_plan"])
        retrieval_service = _build_retrieval_service()

        evidence: Dict[str, List[dict]] = {}
        for sq in plan.sub_questions:
            evidence[sq.id] = _retrieve_for_sub_question(
                retrieval_service=retrieval_service,
                question=sq.question,
                sub_question_id=sq.id,
                retrieval_mode=sq.retrieval_mode,
                evidence_min_count=sq.evidence_min_count,
            )
        return {"evidence_by_subquestion": evidence}

    def evidence_evaluator_node(state: DeepResearchState):
        plan = ResearchPlan(**state["research_plan"])
        gaps = evaluate_evidence_coverage(plan, state.get("evidence_by_subquestion", {}))
        return {"evidence_gaps": [gap.model_dump() for gap in gaps]}

    def gap_retriever_node(state: DeepResearchState):
        """Re-retrieve evidence for sub-questions that still have gaps."""
        gaps = state.get("evidence_gaps", [])
        if not gaps:
            return {"iteration": state.get("iteration", 0) + 1}

        retrieval_service = _build_retrieval_service()
        evidence = dict(state.get("evidence_by_subquestion", {}))

        for gap in gaps:
            sub_question_id = gap.get("sub_question_id", "")
            followup_query = gap.get("suggested_followup_query", "")
            if not followup_query:
                continue

            # Retrieve more evidence using the suggested follow-up query
            new_evidence = _retrieve_for_sub_question(
                retrieval_service=retrieval_service,
                question=followup_query,
                sub_question_id=f"{sub_question_id}_gap_{state.get('iteration', 0)}",
                retrieval_mode="rag",  # broader retrieval mode for gap-filling
                evidence_min_count=1,
            )
            existing = evidence.get(sub_question_id, [])
            evidence[sub_question_id] = existing + new_evidence

        return {
            "evidence_by_subquestion": evidence,
            "iteration": state.get("iteration", 0) + 1,
        }

    def report_writer_node(state: DeepResearchState):
        result = write_fallback_report(
            query=state["original_query"],
            research_plan=state["research_plan"],
            evidence_by_subquestion=state.get("evidence_by_subquestion", {}),
            evidence_gaps=state.get("evidence_gaps", []),
        )
        return {
            "final_report": result.final_report,
            "citation_audit": result.citation_audit,
        }

    def citation_verifier_node(state: DeepResearchState):
        """Re-verify citations on the final report for an independent audit."""
        from app.services.deepresearch.report_writer import _flatten_evidence

        report_text = state.get("final_report", "")
        evidence_chunks = _flatten_evidence(state.get("evidence_by_subquestion", {}))
        audit = verify_citations(report_text, evidence_chunks)
        return {"citation_audit": audit.model_dump()}

    def _route_after_evaluation(state: DeepResearchState) -> str:
        """Route to gap_retriever if gaps exist and iteration < max, else to report_writer."""
        gaps = state.get("evidence_gaps", [])
        iteration = state.get("iteration", 0)
        if gaps and iteration < max_iterations:
            return "gap_retriever"
        return "report_writer"

    graph = StateGraph(DeepResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("evidence_evaluator", evidence_evaluator_node)
    graph.add_node("gap_retriever", gap_retriever_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("citation_verifier", citation_verifier_node)

    graph.set_entry_point("planner")

    # Linear chain: planner -> retrieval -> evidence_evaluator
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "evidence_evaluator")

    # Conditional: evidence_evaluator -> gap_retriever (if gaps remain) | report_writer (if done)
    graph.add_conditional_edges(
        "evidence_evaluator",
        _route_after_evaluation,
        {
            "gap_retriever": "gap_retriever",
            "report_writer": "report_writer",
        },
    )

    # Gap loop: gap_retriever -> evidence_evaluator (re-evaluate after filling gaps)
    graph.add_edge("gap_retriever", "evidence_evaluator")

    # Final chain: report_writer -> citation_verifier -> END
    graph.add_edge("report_writer", "citation_verifier")
    graph.add_edge("citation_verifier", END)

    return graph.compile()
