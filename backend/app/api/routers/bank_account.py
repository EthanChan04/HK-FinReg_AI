"""
银行开户审查路由 (Agentic LLM-only)
迁移自 core_logic.py 中的 check_virtual_bank_eligibility 多智能体
"""
import time
import json
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.requests import ComplianceRequest, ComplianceResponse, ComplianceMetrics
from app.services.utils import pii_scrubber, format_output, get_current_timestamp
from app.services.agents.builder import build_zhipu_llm
from app.core.monitoring import get_tracker

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/bank-account", tags=["Bank Account"])


class VBState(TypedDict, total=False):
    original_input: str
    extracted_kyc_data: str
    cdd_assessment: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str


def _run_vb_graph(safe_input: str) -> str:
    llm = build_zhipu_llm()

    def kyc_node(state: VBState):
        prompt = f"Extract critical KYC entities (Name, Identity Type, Occupation, Income Source) from this application text into a concise framework:\\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"extracted_kyc_data": resp.content}

    def cdd_node(state: VBState):
        prompt = f"Review this KYC data:\\n{state.get('extracted_kyc_data', '')}\\nDetermine the ML/TF Risk Level (Low/Medium/High) and specify the appropriate CDD Level (Simplified/Standard/Enhanced CDD). Briefly justify."
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"cdd_assessment": resp.content}

    def approval_node(state: VBState):
        prompt = f'''You are a Senior Compliance Officer at a Hong Kong Bank.
Assess the account opening application and generate a professional eligibility report.

APPLICATION DATA:
{state['original_input']}

KYC STRUCTURAL DATA:
{state.get('extracted_kyc_data', '')}

CDD PROFILE:
{state.get('cdd_assessment', '')}

Reviewer Feedback to Address (if any):
{state.get('reviewer_feedback', 'None')}

Generate a structured Account Opening Eligibility Report.
Report Date: {get_current_timestamp()}
Answer in English. Use only standard Markdown formatting.'''
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_report": resp.content}

    def cro_node(state: VBState):
        draft = state.get("draft_report", "")
        prompt = f"""You are the Chief Risk Officer (CRO). Review this draft account opening report:
{draft}
If the 'Required CDD Level' correctly matches the 'Risk Level' and the 'Decision' is logically sound, reply exactly with 'APPROVED'.
If the logic is contradictory, reply exactly with 'REJECTED: [detailed reason]'."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        rev_count = state.get("revision_count", 0)
        if content.startswith("APPROVED") or rev_count >= 1:
            return {"final_report": draft, "revision_count": rev_count}
        else:
            return {"reviewer_feedback": content, "revision_count": rev_count + 1}

    def should_continue(state: VBState):
        if state.get("final_report"):
            return "end"
        return "revise"

    workflow = StateGraph(VBState)
    workflow.add_node("kyc_node", kyc_node)
    workflow.add_node("cdd_node", cdd_node)
    workflow.add_node("approval_node", approval_node)
    workflow.add_node("cro_node", cro_node)
    workflow.set_entry_point("kyc_node")
    workflow.add_edge("kyc_node", "cdd_node")
    workflow.add_edge("cdd_node", "approval_node")
    workflow.add_edge("approval_node", "cro_node")
    workflow.add_conditional_edges("cro_node", should_continue, {"end": END, "revise": "approval_node"})
    app_graph = workflow.compile()

    final_state = app_graph.invoke({
        "original_input": safe_input,
        "revision_count": 0,
        "final_report": "",
        "reviewer_feedback": ""
    })
    return final_state.get("final_report", "❌ Report generation failed.")


async def _stream_vb(safe_input: str) -> AsyncGenerator[str, None]:
    agents = [
        ("KYC Analyst", "正在从自然语言抽取申请人的身份与财务背景要素..."),
        ("CDD Specialist", "正在根据身份数据评估洗钱风险等级并制定尽职调查策略..."),
        ("Approval Officer", "正在起草开户资格审查报告初稿..."),
        ("Chief Risk Officer", "正在执行强对抗逻辑复核，验证 CDD 等级与开户结果是否逻辑自洽...")
    ]
    for agent_name, msg in agents:
        yield f"event: agent_state\ndata: {json.dumps({'agent': agent_name, 'status': 'running', 'message': msg}, ensure_ascii=False)}\n\n"

    report = _run_vb_graph(safe_input)
    formatted = format_output(report)
    for line in formatted.split("\n"):
        yield f"event: token\ndata: {json.dumps({'text': line + chr(10)}, ensure_ascii=False)}\n\n"
    yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"


@router.post("/verify", response_model=ComplianceResponse)
async def bank_account_verify(req: ComplianceRequest):
    tracker = get_tracker()
    start = time.time()
    safe_input = pii_scrubber(req.application_data)
    report = _run_vb_graph(safe_input)
    formatted = format_output(report)
    elapsed = time.time() - start
    tracker.log_query("Bank Onboarding Multi-Agent", elapsed, len(req.application_data), "success")
    return ComplianceResponse(
        scrubbed_input=safe_input, final_report=formatted,
        metrics=ComplianceMetrics(processing_time=round(elapsed, 2))
    )


@router.post("/verify/stream")
async def bank_account_verify_stream(req: ComplianceRequest):
    safe_input = pii_scrubber(req.application_data)
    return StreamingResponse(
        _stream_vb(safe_input),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
