"""
银行开户审查路由 (Agentic LLM-only + 反思循环)
迁移自 core_logic.py 中的 check_virtual_bank_eligibility 多智能体

P4.2 升级：
  - 结构化 REJECTED 输出（含 Rejection Type）
  - 反思循环条件边（build_review_edges）
  - 统一 SSE 流式响应（create_streaming_response）
  - asyncio.to_thread 修复同步阻塞
"""
import asyncio
import time
import re
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

router = APIRouter(prefix="/bank-account", tags=["Bank Account"])

# 反思循环常量
MAX_REVISIONS = 2


class VBState(TypedDict, total=False):
    original_input: str
    extracted_kyc_data: str
    cdd_assessment: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str
    # P4.2 新增：结构化拒绝类型
    rejection_type: str  # "" | "insufficient_info" | "quality_issue"


def _parse_cro_output(content: str) -> dict:
    """解析 CRO 输出，提取结构化拒绝类型"""
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


def _build_vb_graph():
    """构建 VB 多智能体图（含反思循环），返回编译后的 CompiledGraph"""
    llm = build_thinking_llm()

    def kyc_node(state: VBState):
        prompt = (
            f"Extract critical KYC entities (Name, Identity Type, Occupation, Income Source) "
            f"from this application text into a concise framework:\n{state['original_input']}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"extracted_kyc_data": resp.content}

    def cdd_node(state: VBState):
        prompt = (
            f"Review this KYC data:\n{state.get('extracted_kyc_data', '')}\n"
            "Determine the ML/TF Risk Level (Low/Medium/High) and specify the appropriate "
            "CDD Level (Simplified/Standard/Enhanced CDD). Briefly justify."
        )
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
        """CRO 审查节点 — 结构化 REJECTED 输出"""
        draft = state.get("draft_report", "")
        rev_count = state.get("revision_count", 0)
        prompt = f"""You are the Chief Risk Officer (CRO). Review this draft account opening report:
{draft}

If the 'Required CDD Level' correctly matches the 'Risk Level' and the 'Decision' is logically sound, reply exactly with:
APPROVED

If there are issues, reply in this exact format:
REJECTED: [specific issue number(s)]
Reason: [Detailed explanation of what went wrong]
Required Fix: [Exact instruction for the Approval Officer to correct the issue]
Rejection Type: [One of: insufficient_info | quality_issue]

Rejection Type Rules:
- Use "insufficient_info" if the report is missing critical data or analysis that should have been provided.
- Use "quality_issue" if the report has logical errors, contradictions, or formatting problems."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        parsed = _parse_cro_output(content)

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
    workflow = StateGraph(VBState)
    workflow.add_node("kyc_node", kyc_node)
    workflow.add_node("cdd_node", cdd_node)
    workflow.add_node("approval_node", approval_node)
    workflow.add_node("cro_node", cro_node)

    workflow.set_entry_point("kyc_node")
    workflow.add_edge("kyc_node", "cdd_node")
    workflow.add_edge("cdd_node", "approval_node")
    workflow.add_edge("approval_node", "cro_node")

    # P4.2: 使用统一反思循环条件边
    build_review_edges(
        workflow,
        reviewer_node="cro_node",
        analyzer_node="approval_node",
        max_revisions=MAX_REVISIONS,
    )

    return workflow.compile()


def _run_vb_graph(safe_input: str) -> str:
    """同步执行 VB 多智能体图"""
    app_graph = _build_vb_graph()
    initial = create_initial_state(safe_input, rejection_type="")
    final_state = app_graph.invoke(initial)
    return final_state.get("final_report", "❌ Report generation failed.")


async def _stream_vb(safe_input: str) -> AsyncGenerator[str, None]:
    """SSE 流式输出 — 使用统一工具函数"""
    agent_steps = [
        ("KYC Analyst", "正在从自然语言抽取申请人的身份与财务背景要素..."),
        ("CDD Specialist", "正在根据身份数据评估洗钱风险等级并制定尽职调查策略..."),
        ("Approval Officer", "正在起草开户资格审查报告初稿..."),
        ("Chief Risk Officer", "正在执行强对抗逻辑复核，验证 CDD 等级与开户结果是否逻辑自洽..."),
    ]
    async for chunk in create_streaming_response(_run_vb_graph, safe_input, agent_steps):
        yield chunk


@router.post("/verify", response_model=ComplianceResponse)
async def bank_account_verify(req: ComplianceRequest):
    tracker = get_tracker()
    start = time.time()
    safe_input = pii_scrubber(req.application_data)
    report = await asyncio.to_thread(_run_vb_graph, safe_input)
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
