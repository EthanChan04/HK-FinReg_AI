"""
SVF 合规审查路由 (Agentic RAG)
迁移自 core_logic.py 中的 generate_risk_report 多智能体
"""
import time
import json
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.requests import ComplianceRequest, ComplianceResponse, ComplianceMetrics
from app.services.utils import pii_scrubber, format_output, get_current_timestamp
from app.services.agents.builder import build_zhipu_llm, build_reranked_retriever
from app.core.monitoring import get_tracker

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/svf", tags=["SVF Compliance"])


# ---- LangGraph State ----
class SVFState(TypedDict, total=False):
    original_input: str
    extracted_entities: str
    retrieved_docs: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str


def _run_svf_graph(safe_input: str) -> str:
    """执行 SVF 多智能体图"""
    llm = build_zhipu_llm()
    retriever = build_reranked_retriever()

    def extractor_node(state: SVFState):
        prompt = f"Extract key compliance details (entity types, license info, transaction patterns) from this query into JSON:\\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"extracted_entities": resp.content}

    def retriever_node(state: SVFState):
        """
        Hybrid + Cohere Reranker 检索节点。
        ContextualCompressionRetriever.invoke() 内部会自动执行：
          1. HybridRetriever (BM25+Dense) → ~20 docs
          2. CohereRerankerCompressor → Top-5 docs
        LangSmith Trace 中会统一展示压缩前/后的文档数量变化。
        """
        if retriever is None:
            return {"retrieved_docs": "RAG engine not available."}

        query = state.get("extracted_entities", state["original_input"])

        # .invoke() 返回经过 Rerank 精排后的 Top-K Document 列表
        top_docs = retriever.invoke(query)

        # 拼接为上下文文本，附带来源页码和 Rerank 分数
        context_parts = []
        for i, doc in enumerate(top_docs):
            page = doc.metadata.get('page', '?')
            score = doc.metadata.get('rerank_score', '-')
            context_parts.append(
                f"[Source {i+1}, p.{page}, relevance={score}]\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        print(f"📚 Reranked Retriever: {len(top_docs)} top docs selected for Analyzer")
        return {"retrieved_docs": context}

    def analyzer_node(state: SVFState):
        from app.services.agents.prompts import ANALYZER_SYSTEM_PROMPT
        feedback = state.get("reviewer_feedback", "None")
        prompt = ANALYZER_SYSTEM_PROMPT.format(
            query=state['original_input'],
            extracted_entities=state.get('extracted_entities', ''),
            retrieved_docs=state.get('retrieved_docs', ''),
            reviewer_feedback=feedback,
            timestamp=get_current_timestamp()
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_report": resp.content}

    def reviewer_node(state: SVFState):
        from app.services.agents.prompts import REVIEWER_SYSTEM_PROMPT
        draft = state.get("draft_report", "")
        prompt = REVIEWER_SYSTEM_PROMPT.format(draft_report=draft)
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        rev_count = state.get("revision_count", 0)
        if content.startswith("APPROVED") or rev_count >= 1:
            return {"final_report": draft, "revision_count": rev_count}
        else:
            return {"reviewer_feedback": content, "revision_count": rev_count + 1}

    def should_continue(state: SVFState):
        if state.get("final_report"):
            return "end"
        return "revise"

    workflow = StateGraph(SVFState)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "reviewer")
    workflow.add_conditional_edges("reviewer", should_continue, {"end": END, "revise": "analyzer"})
    app_graph = workflow.compile()

    final_state = app_graph.invoke({
        "original_input": safe_input,
        "revision_count": 0,
        "final_report": "",
        "reviewer_feedback": ""
    })
    return final_state.get("final_report", "❌ Report generation failed.")


# ---- SSE Streaming Generator ----
async def _stream_svf(safe_input: str) -> AsyncGenerator[str, None]:
    agents = [
        ("Extractor Agent", "正在从自然语言中提取合规审查关键实体..."),
        ("Retriever Agent", "正在调用 ChromaDB 向量库检索 HKMA 法规条款..."),
        ("Analyzer Agent", "正在基于检索结果起草合规风险报告初稿..."),
        ("Reviewer Agent", "正在执行红蓝对抗审查，验证法规引用与逻辑自洽性...")
    ]
    for agent_name, msg in agents:
        yield f"event: agent_state\ndata: {json.dumps({'agent': agent_name, 'status': 'running', 'message': msg}, ensure_ascii=False)}\n\n"

    report = _run_svf_graph(safe_input)
    formatted = format_output(report)

    # 逐段发送 token
    for line in formatted.split("\n"):
        yield f"event: token\ndata: {json.dumps({'text': line + chr(10)}, ensure_ascii=False)}\n\n"

    yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"


# ---- Endpoints ----
@router.post("/analyze", response_model=ComplianceResponse)
async def svf_analyze(req: ComplianceRequest):
    """SVF 合规审查 — 阻塞式"""
    tracker = get_tracker()
    start = time.time()
    safe_input = pii_scrubber(req.application_data)
    report = _run_svf_graph(safe_input)
    formatted = format_output(report)
    elapsed = time.time() - start
    tracker.log_query("SVF Multi-Agent (RAG)", elapsed, len(req.application_data), "success")
    return ComplianceResponse(
        scrubbed_input=safe_input,
        final_report=formatted,
        metrics=ComplianceMetrics(processing_time=round(elapsed, 2))
    )


@router.post("/analyze/stream")
async def svf_analyze_stream(req: ComplianceRequest):
    """SVF 合规审查 — SSE 流式"""
    safe_input = pii_scrubber(req.application_data)
    return StreamingResponse(
        _stream_svf(safe_input),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
