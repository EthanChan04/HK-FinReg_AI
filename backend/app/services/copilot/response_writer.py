"""Bilingual response writer for Compliance Copilot."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.copilot import CopilotRuntimePayload
from app.services.copilot.guardrails import EN_HEADER, ZH_HEADER, apply_guardrails
from app.services.copilot.intent_classifier import IntentDecision
from app.services.copilot.model import build_copilot_llm

_SYSTEM_PROMPT = f"""
You are Compliance Copilot for a Hong Kong bank internal regulatory intelligence platform.
Always answer in two sections:

{ZH_HEADER}
Use professional Traditional Chinese suitable for Hong Kong banking compliance teams.

{EN_HEADER}
Provide the equivalent professional English answer.

Do not provide final compliance approval. Distinguish regulatory facts, analysis, evidence gaps,
and recommended next steps. If evidence is insufficient, say so explicitly. Prefer concise,
audit-friendly answers with source references when evidence is available.
""".strip()


def _compact_runtime_payload(payload: CopilotRuntimePayload | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, CopilotRuntimePayload):
        data = payload.model_dump()
    else:
        data = dict(payload)

    evidence = []
    for item in data.get("evidence_chunks", [])[:6]:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if isinstance(item, dict):
            evidence.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "regulator": item.get("regulator"),
                    "title": item.get("title"),
                    "page": item.get("page"),
                    "section_title": item.get("section_title"),
                    "text": str(item.get("text", ""))[:220],
                }
            )
    data["evidence_chunks"] = evidence
    data["graph_paths"] = data.get("graph_paths", [])[:5]
    return data


def _fallback_response(intent: IntentDecision, compact_context: dict[str, Any], runtime_payload: dict[str, Any]) -> str:
    workflow_name = compact_context.get("case_context", {}).get("workflow_name") or "N/A"
    evidence_count = len(runtime_payload.get("evidence_chunks", []))
    graph_count = len(runtime_payload.get("graph_paths", []))

    return (
        f"{ZH_HEADER}\n"
        "1. 摘要\n"
        f"- 問題已分類為 `{intent.intent}`，目前工作流：{workflow_name}。\n"
        "2. 監管依據\n"
        f"- 已檢索證據數量：{evidence_count}。\n"
        "3. 風險 / 義務 / 控制分析\n"
        f"- 知識圖譜路徑數量：{graph_count}。\n"
        "4. 證據缺口\n"
        "- 如證據不足，建議補充來源文件與情境資料。\n"
        "5. 建議下一步\n"
        "- 需要時可升級到 Deep Research 或 Human Review。\n"
        "6. 置信度 / 限制\n"
        "- 本答案基於當前上下文與內部語料，不含外部網絡資料。\n\n"
        f"{EN_HEADER}\n"
        "1. Summary\n"
        f"- The request is classified as `{intent.intent}` under workflow `{workflow_name}`.\n"
        "2. Regulatory Basis\n"
        f"- Retrieved evidence count: {evidence_count}.\n"
        "3. Risk / Obligation / Control Analysis\n"
        f"- Graph path count: {graph_count}.\n"
        "4. Evidence Gaps\n"
        "- If evidence is insufficient, provide additional source material and case context.\n"
        "5. Recommended Next Steps\n"
        "- Escalate to Deep Research or Human Review when required.\n"
        "6. Confidence / Limitations\n"
        "- This response relies on current internal context and corpus only."
    )


def write_bilingual_response(
    message: str,
    intent: IntentDecision,
    compact_context: dict[str, Any],
    runtime_payload: CopilotRuntimePayload | dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Generate bilingual copilot response and apply safety guardrails."""

    runtime_data = _compact_runtime_payload(runtime_payload)
    context_text = json.dumps(compact_context, ensure_ascii=False)
    runtime_text = json.dumps(runtime_data, ensure_ascii=False)

    user_prompt = (
        "Please answer the user request in the required bilingual format.\n"
        f"Intent: {intent.intent} / Engine: {intent.engine}\n"
        f"Intent reason: {intent.reason}\n"
        f"User message: {message}\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        "Tool output:\n"
        f"{runtime_text}\n\n"
        "Required structure in both language sections:\n"
        "1. Summary\n"
        "2. Regulatory Basis\n"
        "3. Risk / Obligation / Control Analysis\n"
        "4. Evidence Gaps\n"
        "5. Recommended Next Steps\n"
        "6. Confidence / Limitations"
    )

    raw_text = ""
    try:
        llm = build_copilot_llm()
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        raw_text = str(response.content or "").strip()
    except Exception:
        raw_text = _fallback_response(intent, compact_context, runtime_data)

    return apply_guardrails(
        raw_text,
        evidence_chunks=runtime_data.get("evidence_chunks", []),
        confidence_data=compact_context.get("case_context", {}).get("confidence_data", {}),
    )
