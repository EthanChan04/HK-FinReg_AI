"""Tool routing for Compliance Copilot intents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.schemas.copilot import CopilotChatRequest, CopilotRuntimePayload
from app.schemas.deepresearch import ProductProfile, ResearchRequest
from app.services.agents.builder import build_reranked_retriever
from app.services.copilot.intent_classifier import IntentDecision
from app.services.deepresearch.workflow import build_deepresearch_graph
from app.services.kag.graph_retriever import GraphRetriever
from app.services.kag.graph_store import NetworkXGraphStore
from app.services.kag.obligation_mapper import ObligationMapper
from app.services.retrieval.retrieval_service import RetrievalService


@dataclass
class ToolRouteResult:
    payload: CopilotRuntimePayload = field(default_factory=CopilotRuntimePayload)
    tool_name: str = "mimo"


def _build_retrieval_service() -> RetrievalService:
    return RetrievalService(retriever=build_reranked_retriever())


def _map_workflow_recommendation(case_context: dict[str, Any], message: str) -> dict[str, Any]:
    workflow_name = str(case_context.get("workflow_name") or "")
    normalized = message.lower()

    if any(word in normalized for word in ("launch", "product", "ai governance", "產品", "产品", "上線", "上线")):
        return {
            "workflow_id": "product-launch-review",
            "workflow_name": "Product & Business Launch Review",
            "reason": "Message indicates product launch or AI governance.",
        }

    if any(word in normalized for word in ("memo", "compare", "policy", "impact", "備忘錄", "备忘录", "政策")):
        return {
            "workflow_id": "regulatory-memo",
            "workflow_name": "Regulatory Research & Policy Change",
            "reason": "Message indicates cross-regulator memo or policy impact analysis.",
        }

    if workflow_name:
        return {
            "workflow_id": case_context.get("workflow_id"),
            "workflow_name": workflow_name,
            "reason": "Use current active workflow as default recommendation.",
        }

    return {
        "workflow_id": "account-kyc-review",
        "workflow_name": "Customer & Account Compliance",
        "reason": "Fallback to KYC workflow.",
    }


def _as_product_profile(case_context: dict[str, Any]) -> ProductProfile:
    workflow_name = str(case_context.get("workflow_name") or "").lower()
    ai_used = "ai" in workflow_name or "genai" in workflow_name
    cross_border = "cross-border" in workflow_name or "跨境" in workflow_name
    return ProductProfile(
        product_type=case_context.get("workflow_name") or "Compliance Workflow",
        ai_used=ai_used,
        cross_border=cross_border,
    )


def route_tools(intent: IntentDecision, request: CopilotChatRequest, compact_context: dict[str, Any]) -> ToolRouteResult:
    message = request.message
    case_context = compact_context.get("case_context", {})
    payload = CopilotRuntimePayload()

    if intent.intent == "regulatory_qa":
        try:
            retrieval = _build_retrieval_service()
            payload.evidence_chunks = retrieval.retrieve(message, retrieval_mode="rag", top_k=6)
            return ToolRouteResult(payload=payload, tool_name="rag")
        except Exception as exc:
            payload.notes.append(f"RAG retrieval unavailable, fallback to MiMo direct answer: {type(exc).__name__}")
            return ToolRouteResult(payload=payload, tool_name="mimo")

    if intent.intent == "obligation_mapping":
        try:
            settings = get_settings()
            store = NetworkXGraphStore(settings.GRAPH_STORE_PATH)
            store.load()
            graph_retriever = GraphRetriever(store)

            retrieval = _build_retrieval_service()
            mapper = ObligationMapper()
            mapping = mapper.map_obligations(
                query=message,
                product_profile=_as_product_profile(case_context),
                graph_retriever=graph_retriever,
                retrieval_service=retrieval,
            )
            payload.graph_paths = mapping.graph_paths
            payload.evidence_chunks = retrieval.retrieve(message, retrieval_mode="kag", top_k=6)
            payload.notes.append(
                f"Mapped regulators: {', '.join(mapping.applicable_regulators) if mapping.applicable_regulators else 'N/A'}"
            )
            return ToolRouteResult(payload=payload, tool_name="kag")
        except Exception as exc:
            payload.notes.append(f"KAG routing unavailable, fallback to MiMo direct answer: {type(exc).__name__}")
            return ToolRouteResult(payload=payload, tool_name="mimo")

    if intent.intent == "deep_research":
        try:
            graph = build_deepresearch_graph()
            result = graph.invoke(
                {
                    "original_query": message,
                    "request": ResearchRequest(
                        query=message,
                        task_type="cross_regulator_analysis",
                        output_format="memo",
                    ).model_dump(),
                    "iteration": 0,
                }
            )
            payload.research_plan = result.get("research_plan")
            payload.notes.append("DeepResearch workflow executed for complex query.")
            ev_map = result.get("evidence_by_subquestion", {})
            for evidence_list in ev_map.values():
                for item in evidence_list[:2]:
                    if isinstance(item, dict):
                        payload.evidence_chunks.append(item)
            return ToolRouteResult(payload=payload, tool_name="deepresearch")
        except Exception as exc:
            payload.notes.append(
                f"DeepResearch unavailable for this request, fallback to MiMo direct answer: {type(exc).__name__}"
            )
            return ToolRouteResult(payload=payload, tool_name="mimo")

    if intent.intent == "case_explanation":
        raw_evidence = case_context.get("evidence_chunks") or []
        for item in raw_evidence[:6]:
            payload.evidence_chunks.append(item)
        payload.graph_paths = list(case_context.get("graph_paths") or [])[:6]
        if not payload.evidence_chunks:
            try:
                retrieval = _build_retrieval_service()
                payload.evidence_chunks = retrieval.retrieve(message, retrieval_mode="rag", top_k=4)
                payload.notes.append("Case context had no evidence; augmented with RAG retrieval.")
                return ToolRouteResult(payload=payload, tool_name="rag")
            except Exception as exc:
                payload.notes.append(
                    f"Case-context RAG augmentation unavailable, fallback to MiMo direct answer: {type(exc).__name__}"
                )
                return ToolRouteResult(payload=payload, tool_name="mimo")
        return ToolRouteResult(payload=payload, tool_name="rag")

    if intent.intent == "workflow_recommendation":
        payload.workflow_recommendation = _map_workflow_recommendation(case_context, message)
        return ToolRouteResult(payload=payload, tool_name="workflow_router")

    if intent.intent == "human_review_help":
        confidence = case_context.get("confidence_data") or {}
        payload.review_guidance = {
            "current_gate": case_context.get("current_gate"),
            "gate_message": case_context.get("gate_message"),
            "confidence_data": confidence,
            "recommended_actions": [
                "Check weak evidence links and missing citations.",
                "Request additional supporting documents from business owner.",
                "Escalate to second-line compliance reviewer if confidence stays low.",
            ],
        }
        return ToolRouteResult(payload=payload, tool_name="human_review")

    payload.notes.append("Direct MiMo response without additional tools.")
    return ToolRouteResult(payload=payload, tool_name="mimo")
