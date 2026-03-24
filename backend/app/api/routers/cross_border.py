"""
跨境汇款风险评估路由 (Agentic Thinking Model)
迁移自 core_logic.py 中的 assess_cross_border_transaction 多智能体
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

router = APIRouter(prefix="/cross-border", tags=["Cross-Border Remittance"])


class CBState(TypedDict, total=False):
    original_input: str
    parsed_funds: str
    sanctions_screening: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str


def _run_cb_graph(safe_input: str) -> str:
    llm = build_thinking_llm()

    def extractor_node(state: CBState):
        prompt = f"Analyze this remittance log and extract the exact Sender, Beneficiary, Amount, Currency, Destination Country, and Purpose into a clean summary:\\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"parsed_funds": resp.content}

    def sanctions_node(state: CBState):
        prompt = f"Screen these entities and destination against global sanction frameworks (UN, OFAC SDN, EU, HK). Determine strict Confirm/Clear status:\\n{state.get('parsed_funds', '')}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"sanctions_screening": resp.content}

    def investigator_node(state: CBState):
        prompt = f'''You are a Senior Compliance Officer specializing in Cross-Border Remittance.
Assess the transaction logic based on the extracted data and sanction results.

ORIGINAL LOG: {state['original_input']}
PARSED DATA: {state.get('parsed_funds', '')}
SANCTIONS RESULT: {state.get('sanctions_screening', '')}
Reviewer Feedback: {state.get('reviewer_feedback', 'None')}

Generate a structured Cross-Border Transaction Risk Assessment Report.
Report Date: {get_current_timestamp()}
Answer in English. Use standard Markdown.'''
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_report": resp.content}

    def qa_node(state: CBState):
        draft = state.get("draft_report", "")
        prompt = f"""You are the Compliance Director. Review this Draft Remittance Risk Report:
{draft}
If the Sanctions Screening indicates a 'Match' or 'Potential Match', but the Final Decision is 'Approve', it is a FATAL logic error.
Reply exactly with 'APPROVED' if the decision is logically safe.
Reply exactly with 'REJECTED: [detailed reason]' if contradictory."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        rev_count = state.get("revision_count", 0)
        if content.startswith("APPROVED") or rev_count >= 1:
            return {"final_report": draft, "revision_count": rev_count}
        else:
            return {"reviewer_feedback": content, "revision_count": rev_count + 1}

    def should_continue(state: CBState):
        if state.get("final_report"):
            return "end"
        return "revise"

    workflow = StateGraph(CBState)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("sanctions", sanctions_node)
    workflow.add_node("investigator", investigator_node)
    workflow.add_node("qa", qa_node)
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "sanctions")
    workflow.add_edge("sanctions", "investigator")
    workflow.add_edge("investigator", "qa")
    workflow.add_conditional_edges("qa", should_continue, {"end": END, "revise": "investigator"})
    app_graph = workflow.compile()

    final_state = app_graph.invoke({
        "original_input": safe_input,
        "revision_count": 0,
        "final_report": "",
        "reviewer_feedback": ""
    })
    return final_state.get("final_report", "❌ Report generation failed.")


async def _stream_cb(safe_input: str) -> AsyncGenerator[str, None]:
    agents = [
        ("Extraction Specialist", "正在切分资金链路，抽离发送方与收款方金融特征..."),
        ("Sanctions Screener", "正在调用全球制裁名单库 (OFAC/UN/EU) 执行刚性碰撞匹配..."),
        ("AML Investigator", "正在唤醒 LongCat 深度拆解跨境汇款经济合理性..."),
        ("Compliance Director", "正在进行最终的逻辑自洽测试与否决拦截校验...")
    ]
    for agent_name, msg in agents:
        yield f"event: agent_state\ndata: {json.dumps({'agent': agent_name, 'status': 'running', 'message': msg}, ensure_ascii=False)}\n\n"

    report = _run_cb_graph(safe_input)
    formatted = format_output(report)
    for line in formatted.split("\n"):
        yield f"event: token\ndata: {json.dumps({'text': line + chr(10)}, ensure_ascii=False)}\n\n"
    yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"


@router.post("/assess", response_model=ComplianceResponse)
async def cross_border_assess(req: ComplianceRequest):
    tracker = get_tracker()
    start = time.time()
    safe_input = pii_scrubber(req.application_data)
    report = _run_cb_graph(safe_input)
    formatted = format_output(report)
    elapsed = time.time() - start
    tracker.log_query("Cross-Border Multi-Agent", elapsed, len(req.application_data), "success")
    return ComplianceResponse(
        scrubbed_input=safe_input, final_report=formatted,
        metrics=ComplianceMetrics(processing_time=round(elapsed, 2))
    )


@router.post("/assess/stream")
async def cross_border_assess_stream(req: ComplianceRequest):
    safe_input = pii_scrubber(req.application_data)
    return StreamingResponse(
        _stream_cb(safe_input),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
