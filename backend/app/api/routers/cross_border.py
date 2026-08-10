"""
跨境汇款风险评估路由 (Agentic Thinking Model + 反思循环)
迁移自 core_logic.py 中的 assess_cross_border_transaction 多智能体

P4.2 升级：
  - 结构化 REJECTED 输出（含 Rejection Type）
  - 反思循环条件边（build_review_edges）
  - 统一 SSE 流式响应（create_streaming_response）
  - asyncio.to_thread 修复同步阻塞
"""
import asyncio
import re
import time
from typing import AsyncGenerator, TypedDict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.requests import ComplianceRequest, ComplianceResponse, ComplianceMetrics
from app.services.utils import pii_scrubber, format_output, get_current_timestamp
from app.services.agents.builder import build_thinking_llm
from app.core.monitoring import get_tracker
from app.api.routers.workflow_utils import (
    format_sse_event,
    create_streaming_response,
    build_review_edges,
    create_initial_state,
)

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/cross-border", tags=["Cross-Border Remittance"])

# 反思循环常量
MAX_REVISIONS = 2


class CBState(TypedDict, total=False):
    original_input: str
    parsed_funds: str
    sanctions_screening: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str
    # P4.2 新增：结构化拒绝类型
    rejection_type: str  # "" | "insufficient_info" | "quality_issue"


def _parse_qa_output(content: str) -> dict:
    """解析 Compliance Director 输出，提取结构化拒绝类型"""
    stripped = content.strip()
    if stripped.startswith("APPROVED"):
        return {"decision": "APPROVED"}

    rejection_type_match = re.search(
        r"Rejection Type:\s*(insufficient_info|quality_issue)\s*$",
        stripped, re.MULTILINE | re.IGNORECASE,
    )
    rejection_type = rejection_type_match.group(1).lower() if rejection_type_match else None

    # Fallback: 关键词推断
    if not rejection_type:
        text = stripped.lower()
        insufficient_keywords = [
            "missing", "insufficient", "lack", "not found",
            "缺少", "不足", "未找到",
        ]
        rejection_type = "insufficient_info" if any(kw in text for kw in insufficient_keywords) else "quality_issue"

    return {
        "decision": "REJECTED",
        "rejection_type": rejection_type,
        "feedback": stripped,
    }


def _build_cb_graph():
    """构建 CB 多智能体图（含反思循环），返回编译后的 CompiledGraph"""
    llm = build_thinking_llm()

    def extractor_node(state: CBState):
        prompt = (
            "Analyze this remittance log and extract the exact Sender, Beneficiary, "
            "Amount, Currency, Destination Country, and Purpose into a clean summary:\n"
            f"{state['original_input']}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"parsed_funds": resp.content}

    def sanctions_node(state: CBState):
        prompt = (
            "Screen these entities and destination against global sanction frameworks "
            f"(UN, OFAC SDN, EU, HK). Determine strict Confirm/Clear status:\n{state.get('parsed_funds', '')}"
        )
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
        """Compliance Director 审查节点 — 结构化 REJECTED 输出"""
        draft = state.get("draft_report", "")
        rev_count = state.get("revision_count", 0)
        prompt = f"""You are the Compliance Director. Review this Draft Remittance Risk Report:
{draft}
If the Sanctions Screening indicates a 'Match' or 'Potential Match', but the Final Decision is 'Approve', it is a FATAL logic error.

If the decision is logically safe, reply exactly with:
APPROVED

If there are issues, reply in this exact format:
REJECTED: [specific issue number(s)]
Reason: [Detailed explanation of what went wrong]
Required Fix: [Exact instruction for the Investigator to correct the issue]
Rejection Type: [One of: insufficient_info | quality_issue]

Rejection Type Rules:
- Use "insufficient_info" if the report lacks critical transaction data or screening details.
- Use "quality_issue" if the report has logical errors, contradictions, or formatting problems."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        parsed = _parse_qa_output(content)

        if parsed["decision"] == "APPROVED" or rev_count >= MAX_REVISIONS:
            return {"final_report": draft, "revision_count": rev_count, "rejection_type": ""}
        return {
            "reviewer_feedback": (
                f"REJECTED\n"
                f"Reason: {parsed.get('feedback', content)}\n"
                f"Rejection Type: {parsed['rejection_type']}"
            ),
            "revision_count": rev_count + 1,
            "rejection_type": parsed["rejection_type"],
        }

    # ---- 构建图 ----
    workflow = StateGraph(CBState)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("sanctions", sanctions_node)
    workflow.add_node("investigator", investigator_node)
    workflow.add_node("qa", qa_node)

    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "sanctions")
    workflow.add_edge("sanctions", "investigator")
    workflow.add_edge("investigator", "qa")

    # P4.2: 使用统一反思循环条件边
    build_review_edges(
        workflow,
        reviewer_node="qa",
        analyzer_node="investigator",
        max_revisions=MAX_REVISIONS,
    )

    return workflow.compile()


def _run_cb_graph(safe_input: str) -> str:
    """同步执行 CB 多智能体图"""
    app_graph = _build_cb_graph()
    initial = create_initial_state(safe_input, rejection_type="")
    final_state = app_graph.invoke(initial)
    return final_state.get("final_report", "❌ Report generation failed.")


async def _stream_cb(safe_input: str) -> AsyncGenerator[str, None]:
    """SSE 流式输出 — 使用统一工具函数"""
    agent_steps = [
        ("Extraction Specialist", "正在切分资金链路，抽离发送方与收款方金融特征..."),
        ("Sanctions Screener", "正在调用全球制裁名单库 (OFAC/UN/EU) 执行刚性碰撞匹配..."),
        ("AML Investigator", "正在唤醒 DeepSeek V4 Flash 深度拆解跨境汇款经济合理性..."),
        ("Compliance Director", "正在进行最终的逻辑自洽测试与否决拦截校验..."),
    ]
    async for chunk in create_streaming_response(_run_cb_graph, safe_input, agent_steps):
        yield chunk


@router.post("/assess", response_model=ComplianceResponse)
async def cross_border_assess(req: ComplianceRequest):
    tracker = get_tracker()
    start = time.time()
    safe_input = pii_scrubber(req.application_data)
    report = await asyncio.to_thread(_run_cb_graph, safe_input)
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
