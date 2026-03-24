"""
SME 企业贷款信用评估路由 (Agentic Thinking Model)
迁移自 core_logic.py 中的 assess_sme_credit 多智能体
"""
import time
import json
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.requests import ComplianceRequest, ComplianceResponse, ComplianceMetrics
from app.services.utils import pii_scrubber, format_output, get_current_timestamp
from app.services.agents.builder import build_thinking_llm
from app.core.monitoring import get_tracker

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/sme", tags=["SME Lending"])


class SMEState(TypedDict, total=False):
    original_input: str
    parsed_financials: str
    risk_analysis: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str


def _run_sme_graph(safe_input: str) -> str:
    llm = build_thinking_llm()

    def data_node(state: SMEState):
        prompt = f"Extract Company Name, Business Type, Operating Years, Revenue, Margins, and requested Amount into a structured format:\\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"parsed_financials": resp.content}

    def analyst_node(state: SMEState):
        prompt = f"Analyze this financial data:\\n{state.get('parsed_financials', '')}\\nDetermine the Business Viability (Stable/Volatile) and highlight any severe Industry or Currency Risks. Provide a rigorous quantitative logic chain."
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"risk_analysis": resp.content}

    def officer_node(state: SMEState):
        prompt = f'''You are a Senior Credit Risk Analyst at a Hong Kong Virtual Bank specializing in SME Lending.
Draft the SME Credit Assessment Report.

APPLICATION DATA: {state['original_input']}
FINANCIAL ANALYSIS: {state.get('risk_analysis', '')}
Committee Feedback: {state.get('reviewer_feedback', 'None')}

Generate a structured SME Credit Assessment Report.
Report Date: {get_current_timestamp()}
Answer in English. Use standard Markdown.'''
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_report": resp.content}

    def committee_node(state: SMEState):
        draft = state.get("draft_report", "")
        prompt = f"""You are the Credit Committee Chair. Review this Draft Credit Report:
{draft}
If the 'Final Credit Rating' logically contradicts significant risks in the analysis, it is a red flag.
Reply exactly with 'APPROVED' if the Rating accurately reflects the stated risks.
Reply exactly with 'REJECTED: [detailed reason]' if the rating is too lenient."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        rev_count = state.get("revision_count", 0)
        if content.startswith("APPROVED") or rev_count >= 1:
            return {"final_report": draft, "revision_count": rev_count}
        else:
            return {"reviewer_feedback": content, "revision_count": rev_count + 1}

    def should_continue(state: SMEState):
        if state.get("final_report"):
            return "end"
        return "revise"

    workflow = StateGraph(SMEState)
    workflow.add_node("data_node", data_node)
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("officer_node", officer_node)
    workflow.add_node("committee_node", committee_node)
    workflow.set_entry_point("data_node")
    workflow.add_edge("data_node", "analyst_node")
    workflow.add_edge("analyst_node", "officer_node")
    workflow.add_edge("officer_node", "committee_node")
    workflow.add_conditional_edges("committee_node", should_continue, {"end": END, "revise": "officer_node"})
    app_graph = workflow.compile()

    final_state = app_graph.invoke({
        "original_input": safe_input,
        "revision_count": 0,
        "final_report": "",
        "reviewer_feedback": ""
    })
    return final_state.get("final_report", "❌ Report generation failed.")


async def _stream_sme(safe_input: str) -> AsyncGenerator[str, None]:
    agents = [
        ("Data Processor", "正在从杂乱财报中提取营收、工龄、及利润率等核心指标..."),
        ("Financial Analyst", "正在解构企业现金流底稿，并量化市场与行业风险敞口..."),
        ("Credit Officer", "正在起草信贷评级长篇报告，推演初版评级 (A-E)..."),
        ("Credit Committee", "正在严格比对信贷评级与底层财务风险标的，执行逻辑阻断校验...")
    ]
    for agent_name, msg in agents:
        yield f"event: agent_state\ndata: {json.dumps({'agent': agent_name, 'status': 'running', 'message': msg}, ensure_ascii=False)}\n\n"

    report = _run_sme_graph(safe_input)
    formatted = format_output(report)
    for line in formatted.split("\n"):
        yield f"event: token\ndata: {json.dumps({'text': line + chr(10)}, ensure_ascii=False)}\n\n"
    yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"


@router.post("/credit-rating", response_model=ComplianceResponse)
async def sme_credit_rating(req: ComplianceRequest):
    tracker = get_tracker()
    start = time.time()
    safe_input = pii_scrubber(req.application_data)
    report = _run_sme_graph(safe_input)
    formatted = format_output(report)
    elapsed = time.time() - start
    tracker.log_query("SME Credit Multi-Agent", elapsed, len(req.application_data), "success")
    return ComplianceResponse(
        scrubbed_input=safe_input, final_report=formatted,
        metrics=ComplianceMetrics(processing_time=round(elapsed, 2))
    )


@router.post("/credit-rating/stream")
async def sme_credit_rating_stream(req: ComplianceRequest):
    safe_input = pii_scrubber(req.application_data)
    return StreamingResponse(
        _stream_sme(safe_input),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
