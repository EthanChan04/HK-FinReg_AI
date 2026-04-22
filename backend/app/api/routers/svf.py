"""
SVF 合规审查路由 (Agentic RAG + 反思循环)
迁移自 core_logic.py 中的 generate_risk_report 多智能体

P1 升级：
  - 规则化 SubQueryPlanner（零 LLM 调用）
  - 跨轮文档累积去重（content_hash 替代字符串拼接）
  - 三路条件边（APPROVED / revise / re_retrieve）
  - asyncio.to_thread 修复同步阻塞
"""
import asyncio
import hashlib
import json
import re
import time
from typing import AsyncGenerator, Dict, List, TypedDict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.schemas.requests import (
    AnalyzerOutput,
    ComplianceMetrics,
    ComplianceRequest,
    ComplianceResponse,
    RegulatoryFact,
    ReviewerVerdict,
    SourceCitation,
)
from app.services.utils import pii_scrubber, format_output, get_current_timestamp
from app.services.agents.builder import (
    build_profiled_retriever,
    build_reranked_retriever,
    build_zhipu_llm,
    classify_query_type,
    get_structured_reviewer_llm,
)
from app.core.monitoring import get_tracker

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/svf", tags=["SVF Compliance"])


# ---- LangGraph State ----
class SVFState(TypedDict, total=False):
    original_input: str
    extracted_entities: str
    query_type: str
    retrieved_docs: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str
    format_retry_count: int
    validation_errors: str
    # P1 新增：反思循环
    retrieval_round: int              # 当前检索轮次（0=首次，1=反思二次检索）
    sub_queries: List[str]            # 规则化子查询列表
    accumulated_docs: List[Dict]      # 跨轮累积文档（结构化，替代字符串拼接）
    # P2 预留：置信度
    confidence_score: float           # 检索置信度（Rerank Top-1 分数）
    confidence_warning: str           # 置信度警告标记
    # M2: Analyzer 推理置信度
    reasoning_confidence: float       # Analyzer 自评推理置信度
    low_confidence_areas: str         # 低置信度领域 JSON
    high_confidence_areas: str        # 高置信度领域 JSON
    # M3: Reviewer 交叉验证
    reviewer_confidence: float        # Reviewer 独立置信度打分
    cross_validation_passed: bool     # 偏差检测是否通过
    # 结构化拒绝类型
    rejection_type: str               # "insufficient_info" | "quality_issue" | ""


# ---- 反思循环常量 ----
MAX_REVISIONS = 2       # 最大修订次数
MAX_RETRIEVAL = 1       # 最大二次检索次数

SOURCE_CITATION_PATTERN = re.compile(
    r"\[(?:Source:\s*)?(?:Source\s*)?(\d+),\s*p\.?\s*(\d+)\]",
    re.IGNORECASE,
)
REVIEWER_CHECK_PATTERN = re.compile(r"\d+")
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$")


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _parse_markdown_sections(markdown_text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current_heading = ""

    for line in markdown_text.splitlines():
        stripped = line.strip()
        heading_match = HEADING_PATTERN.match(stripped)
        if heading_match:
            current_heading = heading_match.group(1).strip("* ").strip()
            sections[current_heading] = []
            continue
        if current_heading:
            sections[current_heading].append(line)

    return {
        heading: "\n".join(lines).strip()
        for heading, lines in sections.items()
    }


def _find_section_content(sections: Dict[str, str], *keywords: str) -> str:
    lowered_keywords = [re.sub(r"[\s*`_]+", "", keyword.lower()) for keyword in keywords]
    for heading, content in sections.items():
        lowered_heading = re.sub(r"[\s*`_]+", "", heading.lower())
        if any(keyword in lowered_heading for keyword in lowered_keywords):
            return content.strip()
    return ""


def _extract_title(markdown_text: str) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _extract_report_date(markdown_text: str) -> str:
    match = re.search(r"\*\*[^*]*(?:報告日期|报告日期)[^*]*\*\*\s*:\s*(.+)", markdown_text)
    if match:
        return match.group(1).strip()
    match = re.search(r"(?:報告日期|报告日期)\s*:\s*(.+)", markdown_text)
    return match.group(1).strip() if match else ""


def _remove_source_citations(text: str) -> str:
    return _normalize_whitespace(SOURCE_CITATION_PATTERN.sub("", text))


def _extract_regulatory_facts(section_text: str) -> List[RegulatoryFact]:
    blocks: List[str] = []
    current_block: List[str] = []

    for raw_line in section_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if current_block:
                blocks.append(" ".join(current_block).strip())
                current_block = []
            continue
        if stripped.startswith("### "):
            if current_block:
                blocks.append(" ".join(current_block).strip())
                current_block = []
            continue
        if re.match(r"^[-*]\s+", stripped) and current_block:
            blocks.append(" ".join(current_block).strip())
            current_block = []
        current_block.append(stripped)

    if current_block:
        blocks.append(" ".join(current_block).strip())

    facts: List[RegulatoryFact] = []
    for block in blocks:
        citations: List[SourceCitation] = []
        for source_number, page in SOURCE_CITATION_PATTERN.findall(block):
            try:
                citations.append(
                    SourceCitation(
                        source_number=int(source_number),
                        page=int(page),
                        content=_remove_source_citations(block),
                    )
                )
            except (TypeError, ValueError):
                continue
        if not citations:
            continue
        statement = _remove_source_citations(re.sub(r"^[-*\d.\s]+", "", block))
        facts.append(RegulatoryFact(statement=statement, citations=citations))

    return facts


def _extract_risk_rating(section_text: str) -> str:
    match = re.search(r"(低風險|中風險|高風險)", section_text)
    return match.group(1) if match else ""


def _parse_analyzer_markdown(markdown_text: str) -> dict:
    sections = _parse_markdown_sections(markdown_text)

    applicant_overview = _find_section_content(sections, "申請人概覽", "申请人概览")
    facts_section = _find_section_content(sections, "法規事實摘要", "法规事实摘要")
    gap_analysis = _find_section_content(sections, "合規差距分析", "合规差距分析")
    recommendations = _find_section_content(sections, "合規建議", "合规建议")
    risk_section = _find_section_content(sections, "風險評級", "风险评级")
    insufficiency = _find_section_content(sections, "資訊不足聲明", "资料不足声明", "信息不足声明")

    return {
        "report_title": _extract_title(markdown_text),
        "report_date": _extract_report_date(markdown_text),
        "applicant_overview": _normalize_whitespace(applicant_overview),
        "regulatory_facts": _extract_regulatory_facts(facts_section),
        "gap_analysis": _normalize_whitespace(gap_analysis),
        "recommendations": _normalize_whitespace(recommendations),
        "risk_rating": _extract_risk_rating(risk_section),
        "insufficiency_disclaimer": _normalize_whitespace(insufficiency),
    }


def _parse_reviewer_output(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("APPROVED"):
        return {"decision": "APPROVED", "failed_checks": []}

    first_line = stripped.splitlines()[0] if stripped else ""
    failed_checks = [int(value) for value in REVIEWER_CHECK_PATTERN.findall(first_line)]

    reason_match = re.search(r"^Reason:\s*(.+)$", stripped, re.MULTILINE)
    required_fix_match = re.search(r"^Required Fix:\s*(.+)$", stripped, re.MULTILINE)
    rejection_type_match = re.search(r"^Rejection Type:\s*(insufficient_info|quality_issue)\s*$", stripped, re.MULTILINE)

    rejection_type = rejection_type_match.group(1).strip() if rejection_type_match else None

    return {
        "decision": "REJECTED",
        "failed_checks": failed_checks,
        "reason": reason_match.group(1).strip() if reason_match else None,
        "required_fix": required_fix_match.group(1).strip() if required_fix_match else None,
        "rejection_type": rejection_type,
    }


def _infer_rejection_type(content: str) -> str:
    """当 ReviewerVerdict 验证失败时，从原始文本推断 rejection_type 作为 fallback"""
    text = content.lower()
    insufficient_keywords = [
        "missing", "insufficient", "lack", "not found", "not available",
        "inadequate", "缺少", "不足", "未找到", "需要更多信息",
    ]
    if any(kw in text for kw in insufficient_keywords):
        return "insufficient_info"
    return "quality_issue"


def _flatten_json_payload(payload) -> List[str]:
    lines: List[str] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                nested = _flatten_json_payload(value)
                for entry in nested:
                    lines.append(f"{key}.{entry}")
            else:
                lines.append(f"{key}: {value}")
        return lines

    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            if isinstance(value, (dict, list)):
                nested = _flatten_json_payload(value)
                for entry in nested:
                    lines.append(f"[{idx}].{entry}")
            else:
                lines.append(f"[{idx}]: {value}")
        return lines

    lines.append(str(payload))
    return lines


def _normalize_extracted_entities(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    # 處理 markdown code block: 去掉 ```json 和 ```
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```json"):
                text = "\n".join(lines[1:-1]).strip() if lines[-1].startswith("```") else "\n".join(lines[1:]).strip()
            elif lines[0].startswith("```"):
                text = "\n".join(lines[1:-1]).strip() if lines[-1].startswith("```") else "\n".join(lines[1:]).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return raw_text

    flattened = _flatten_json_payload(parsed)
    if not flattened:
        return raw_text
    return "\n".join(flattened)


def _compute_content_hash(content: str) -> str:
    """计算文档内容哈希用于跨轮去重"""
    normalized = re.sub(r'\s+', ' ', content.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def _compute_retrieval_confidence(docs_or_accumulated: list) -> tuple:
    """计算检索置信度（三维指标：Rerank Top-1 + Top-5 gap + 警告）

    支持 Document 列表和 accumulated_docs 字典列表两种输入。

    Returns:
        (top1_score, top5_gap, warning): 检索置信度分数、Top-5 gap、警告信息
    """
    from app.core.config import get_settings
    settings = get_settings()

    if not docs_or_accumulated:
        return 0.0, 0.0, "⚠️ 低置信度警告：未检索到任何相关文档，报告结论可能缺乏法规依据"

    # 兼容 Document 和 dict 两种类型
    first = docs_or_accumulated[0]
    if isinstance(first, dict):
        metadata = first.get("metadata", {})
    elif hasattr(first, 'metadata'):
        metadata = first.metadata
    else:
        metadata = {}

    top1_score = float(metadata.get("rerank_score", 0.0))

    # 计算 Top-5 gap（Top-1 与 Top-5 均值的差距）
    top5_scores = []
    for i, item in enumerate(docs_or_accumulated[:5]):
        if isinstance(item, dict):
            s = float(item.get("metadata", {}).get("rerank_score", 0.0))
        elif hasattr(item, 'metadata'):
            s = float(item.metadata.get("rerank_score", 0.0))
        else:
            s = 0.0
        top5_scores.append(s)
    top5_mean = sum(top5_scores) / len(top5_scores) if top5_scores else 0.0
    top5_gap = max(0.0, top1_score - top5_mean)

    warning = None
    if top1_score < settings.CONFIDENCE_LOW_THRESHOLD:
        warning = (
            f"⚠️ 低置信度警告：最相关文档的 Rerank 分数仅为 {top1_score:.2f}，"
            "报告结论可能缺乏充分的法规依据。建议关注引用来源的完整性。"
        )
    elif top1_score < settings.CONFIDENCE_MED_THRESHOLD:
        warning = (
            f"📋 置信度提示：Rerank Top-1 分数 {top1_score:.2f}，"
            "建议关注引用来源的完整性。"
        )

    return top1_score, top5_gap, warning


# ---- 节点名称 → 前端展示映射 ----
NODE_DISPLAY_MAP = {
    "extractor":        ("Extractor Agent",         "正在从自然语言中提取合规审查关键实体..."),
    "retriever":        ("Retriever Agent",         "正在调用 ChromaDB 向量库检索 HKMA 法规条款..."),
    "analyzer":         ("Analyzer Agent",          "正在基于检索结果起草合规风险报告初稿..."),
    "format_validator": ("Format Validator Agent",  "正在验证报告结构与引用格式..."),
    "reviewer":         ("Reviewer Agent",          "正在执行红蓝对抗审查，验证法规引用与逻辑自洽性..."),
    "sub_query_planner":("Sub-Query Planner",       "反思循环：正在规划二次检索策略..."),
}


def _build_svf_graph():
    """构建 SVF 多智能体图（含反思循环），返回编译后的 CompiledGraph"""
    llm = build_zhipu_llm()
    base_retriever = build_reranked_retriever()

    def extractor_node(state: SVFState):
        prompt = f"Extract key compliance details (entity types, license info, transaction patterns) from this query into JSON:\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        extracted = _normalize_extracted_entities(resp.content)
        if extracted:
            print("[SVF][EXTRACTOR] Parsed JSON output into normalized text payload.")
        return {
            "extracted_entities": extracted or state["original_input"],
            "retrieval_round": state.get("retrieval_round", 0),
        }

    def retriever_node(state: SVFState):
        """支持多轮检索的 Retriever：首次检索或基于子查询的二次检索"""
        from app.services.semantic_cache import get_semantic_cache

        if base_retriever is None:
            return {"retrieved_docs": "RAG engine not available."}

        cache = get_semantic_cache()
        is_re_retrieve = state.get("retrieval_round", 0) > 0

        if is_re_retrieve:
            # ---- 反思二次检索：对每个子查询分别检索 ----
            sub_queries = state.get("sub_queries", [])
            new_docs_raw: List = []
            existing_hashes = {
                d.get("content_hash")
                for d in state.get("accumulated_docs", [])
            }

            for sq in sub_queries:
                query_type = classify_query_type(sq)
                active_retriever = build_profiled_retriever(base_retriever, query_type)

                cached_docs = None
                scrubbed_query = sq
                query_vector: List[float] = []
                if cache:
                    cached_docs, scrubbed_query, query_vector = cache.get(sq)

                if cached_docs is not None:
                    docs = cached_docs
                else:
                    docs = active_retriever.invoke(sq) if active_retriever else []
                    if cache:
                        cache.put(scrubbed_query, query_vector, docs)

                for doc in docs:
                    chash = _compute_content_hash(doc.page_content)
                    if chash not in existing_hashes:
                        new_docs_raw.append(doc)
                        existing_hashes.add(chash)

            # 合并到累积文档
            accumulated = list(state.get("accumulated_docs", []))
            for doc in new_docs_raw:
                chash = _compute_content_hash(doc.page_content)
                accumulated.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "content_hash": chash,
                })

            # 重新格式化上下文
            context_parts = []
            for i, d in enumerate(accumulated):
                page = d["metadata"].get("page", "?")
                score = d["metadata"].get("rerank_score", "-")
                context_parts.append(
                    f"[Source {i+1}, p.{page}, relevance={score}]\n{d['page_content']}"
                )
            context = "\n\n---\n\n".join(context_parts)

            print(f"[SVF][RETRIEVER] Re-retrieval: {len(new_docs_raw)} new docs, {len(accumulated)} total accumulated.")

            # P2: 重新计算置信度（基于全量累积文档）
            score, top5_gap, warning = _compute_retrieval_confidence(accumulated)
            return {
                "retrieved_docs": context,
                "accumulated_docs": accumulated,
                "confidence_score": score,
                "confidence_warning": warning or "",
            }

        else:
            # ---- 首次检索 ----
            query = state.get("extracted_entities", state["original_input"])
            query_type = classify_query_type(query)
            active_retriever = build_profiled_retriever(base_retriever, query_type)

            cached_docs = None
            scrubbed_query = query
            query_vector: List[float] = []
            if cache:
                cached_docs, scrubbed_query, query_vector = cache.get(query)

            if cached_docs is not None:
                top_docs = cached_docs
            else:
                top_docs = active_retriever.invoke(query) if active_retriever else []
                if cache:
                    cache.put(scrubbed_query, query_vector, top_docs)

            # 结构化存储
            accumulated = []
            context_parts = []
            for i, doc in enumerate(top_docs):
                chash = _compute_content_hash(doc.page_content)
                accumulated.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "content_hash": chash,
                })
                page = doc.metadata.get("page", "?")
                score = doc.metadata.get("rerank_score", "-")
                context_parts.append(
                    f"[Source {i+1}, p.{page}, relevance={score}]\n{doc.page_content}"
                )
            context = "\n\n---\n\n".join(context_parts)

            print(f"[SVF][RETRIEVER] Initial: selected {len(top_docs)} top docs for Analyzer.")

            # P2: 计算置信度
            score, top5_gap, warning = _compute_retrieval_confidence(top_docs)
            print(f"[SVF][RETRIEVER] Confidence: score={score:.2f}, top5_gap={top5_gap:.2f}, warning={'Yes' if warning else 'None'}")
            return {
                "retrieved_docs": context,
                "query_type": query_type,
                "accumulated_docs": accumulated,
                "confidence_score": score,
                "confidence_warning": warning or "",
            }

    def analyzer_node(state: SVFState):
        from app.services.agents.prompts import ANALYZER_SYSTEM_PROMPT
        feedback_parts = []
        reviewer_feedback = state.get("reviewer_feedback", "").strip()
        validation_errors = state.get("validation_errors", "").strip()
        if reviewer_feedback:
            feedback_parts.append(reviewer_feedback)
        if validation_errors:
            feedback_parts.append(
                "FORMAT VALIDATION FEEDBACK:\n"
                f"{validation_errors}\n"
                "Revise the report so it matches the required markdown structure and citation rules."
            )
        feedback = "\n\n".join(feedback_parts) if feedback_parts else "None"
        prompt = ANALYZER_SYSTEM_PROMPT.format(
            query=state['original_input'],
            extracted_entities=state.get('extracted_entities', ''),
            retrieved_docs=state.get('retrieved_docs', ''),
            reviewer_feedback=feedback,
            timestamp=get_current_timestamp()
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content

        # M2: 解析 Analyzer 置信度自评 JSON 块
        reasoning_confidence = 0.5
        low_confidence_areas = "[]"
        high_confidence_areas = "[]"
        confidence_match = re.search(
            r'```json\s*\n?\s*(\{[^`]*?"overall_confidence"\s*:[^}]*\})\s*\n?\s*```',
            content, re.DOTALL,
        )
        if not confidence_match:
            # Fallback: 尝试匹配裸 JSON
            confidence_match = re.search(
                r'(\{[^{}]*"overall_confidence"\s*:\s*[\d.]+[^{}]*\})',
                content,
            )
        if confidence_match:
            try:
                import json as _json
                confidence_data = _json.loads(confidence_match.group(1))
                reasoning_confidence = float(confidence_data.get("overall_confidence", 0.5))
                reasoning_confidence = max(0.0, min(1.0, reasoning_confidence))
                low_confidence_areas = _json.dumps(
                    confidence_data.get("low_confidence_areas", []), ensure_ascii=False
                )
                high_confidence_areas = _json.dumps(
                    confidence_data.get("high_confidence_areas", []), ensure_ascii=False
                )
                print(f"[SVF][ANALYZER] Self-assessed confidence: {reasoning_confidence:.2f}")
            except (ValueError, TypeError) as e:
                print(f"[SVF][ANALYZER] Failed to parse confidence JSON: {e}")

        # 从报告中移除置信度 JSON 块（避免出现在最终报告中）
        content = re.sub(
            r'```json\s*\n?\s*\{[^`]*?"overall_confidence"[^}]*\}\s*\n?\s*```',
            '', content, flags=re.DOTALL,
        ).rstrip()
        # 也移除裸 JSON 块
        content = re.sub(
            r'\n\s*\{[^{}]*"overall_confidence"\s*:\s*[\d.]+[^{}]*\}\s*$',
            '', content,
        ).rstrip()

        return {
            "draft_report": content,
            "reasoning_confidence": reasoning_confidence,
            "low_confidence_areas": low_confidence_areas,
            "high_confidence_areas": high_confidence_areas,
        }

    def format_validator_node(state: SVFState):
        draft = state.get("draft_report", "")
        retry_count = state.get("format_retry_count", 0)
        max_format_retries = 2

        try:
            parsed = _parse_analyzer_markdown(draft)
            AnalyzerOutput(**parsed)
            return {"format_retry_count": retry_count, "validation_errors": ""}
        except ValidationError as exc:
            if retry_count >= max_format_retries:
                return {
                    "format_retry_count": retry_count,
                    "validation_errors": (
                        "FORMAT_VALIDATION_FAILED: report structure is invalid after max retries. "
                        "Please regenerate with strict section headings and source citations."
                    ),
                }
            return {
                "format_retry_count": retry_count + 1,
                "validation_errors": f"FORMAT_VIOLATION ({len(exc.errors())} errors)\n{exc}",
            }

    def reviewer_node(state: SVFState):
        from app.services.agents.prompts import REVIEWER_SYSTEM_PROMPT
        draft = state.get("draft_report", "")
        prompt = REVIEWER_SYSTEM_PROMPT.format(draft_report=draft)
        rev_count = state.get("revision_count", 0)
        verdict = None
        raw_content = None  # 保存 LLM 原始输出，避免重复调用

        # P2.5: 尝试使用 with_structured_output（若 LLM 支持 function calling）
        # 缓存机制：首次调用后缓存结果，后续不再重复尝试
        structured_llm = get_structured_reviewer_llm()
        if structured_llm is not None:
            try:
                result = structured_llm.invoke([HumanMessage(content=prompt)])
                if isinstance(result, ReviewerVerdict):
                    verdict = result
                else:
                    # structured output 返回了原始内容而非 ReviewerVerdict，
                    # 保存以供 fallback 正则解析复用，避免再调一次 LLM
                    raw_content = str(result) if result else None
            except Exception as e:
                print(f"[SVF][REVIEWER] structured output failed, falling back: {e}")

        # Fallback: 正则解析（复用 structured_llm 的原始输出，避免二次 LLM 调用）
        reviewer_confidence = 0.5
        if verdict is None:
            if raw_content is None:
                # 只在没有 structured_llm 或 structured_llm 抛异常时才调 LLM
                resp = llm.invoke([HumanMessage(content=prompt)])
                raw_content = resp.content.strip()

            content = raw_content.strip()
            # 去除 markdown 代码块包裹（LLM 有时返回 ```...\n```）
            if content.startswith("```"):
                lines = content.splitlines()
                inner = [l for l in lines[1:] if not l.strip().startswith("```")]
                content = "\n".join(inner).strip()
            # M3: 提取 Reviewer Confidence 打分
            rc_match = re.search(
                r'Reviewer Confidence:\s*([\d.]+)', content, re.IGNORECASE
            )
            if rc_match:
                try:
                    reviewer_confidence = max(0.0, min(1.0, float(rc_match.group(1))))
                except ValueError:
                    pass
            try:
                parsed = _parse_reviewer_output(content)
                verdict = ReviewerVerdict(**parsed)
            except ValidationError as exc:
                rejection_type = _infer_rejection_type(content)
                verdict = ReviewerVerdict(
                    decision="REJECTED",
                    failed_checks=[],
                    reason=f"Reviewer output format invalid: {exc.errors()[0]['msg']}",
                    required_fix="Return the exact REJECTED format with Reason/Required Fix/Rejection Type.",
                    rejection_type=rejection_type,
                )
        else:
            # M3: 从结构化输出中获取 reviewer_confidence
            reviewer_confidence = verdict.reviewer_confidence or 0.5

        # M3: 交叉验证 — 检测 retrieval 与 reasoning 置信度偏差
        from app.core.config import get_settings
        settings = get_settings()
        retrieval_conf = state.get("confidence_score", 0.0)
        reasoning_conf = state.get("reasoning_confidence", 0.5)
        deviation = abs(retrieval_conf - reasoning_conf)
        cross_validation_passed = deviation <= settings.CONFIDENCE_CROSS_VALIDATION_THRESHOLD

        if not cross_validation_passed:
            print(
                f"[SVF][REVIEWER] Cross-validation WARNING: "
                f"retrieval={retrieval_conf:.2f} vs reasoning={reasoning_conf:.2f} "
                f"(deviation={deviation:.2f} > {settings.CONFIDENCE_CROSS_VALIDATION_THRESHOLD})"
            )

        # 构建 ConfidenceScore 三维模型
        from app.schemas.requests import ConfidenceScore
        top5_gap = 0.0  # top5_gap 存在于 retriever 但不在 state 中，用 0 占位
        confidence_model = ConfidenceScore(
            retrieval_confidence=retrieval_conf,
            top5_gap=top5_gap,
            reasoning_confidence=reasoning_conf,
            low_confidence_areas=json.loads(state.get("low_confidence_areas", "[]")),
            high_confidence_areas=json.loads(state.get("high_confidence_areas", "[]")),
            cross_validation_passed=cross_validation_passed,
            cross_validation_deviation=deviation,
        )
        print(
            f"[SVF][REVIEWER] ConfidenceScore: "
            f"overall={confidence_model.overall_confidence:.3f} "
            f"warning_level={confidence_model.warning_level} "
            f"(retrieval={retrieval_conf:.2f}, reasoning={reasoning_conf:.2f}, "
            f"reviewer={reviewer_confidence:.2f}, deviation={deviation:.2f})"
        )

        # 修复：当即将达到修订上限时也输出 final_report
        # 原逻辑 rev_count >= MAX_REVISIONS 永远不会为真，
        # 因为 route_after_review 会在 rev_count 达到上限时拦截路由到 "end"。
        # 改为 rev_count >= MAX_REVISIONS - 1，确保最后一轮 Reviewer 输出 final_report。
        if verdict.decision == "APPROVED" or rev_count >= MAX_REVISIONS - 1:
            # P2: 置信度静态标记（只读，不参与控制流）
            confidence_warning = state.get("confidence_warning", "")
            final = draft
            warning_parts = []
            if confidence_warning:
                warning_parts.append(confidence_warning)
            if not cross_validation_passed:
                warning_parts.append(
                    f"⚠️ 交叉验证警告：检索置信度 ({retrieval_conf:.2f}) 与推理置信度 ({reasoning_conf:.2f}) "
                    f"偏差过大 ({deviation:.2f})，可能存在过度自信或信息不足。"
                )
            if warning_parts:
                final = f"{draft}\n\n---\n" + "\n\n".join(warning_parts)

            return {
                "final_report": final,
                "revision_count": rev_count,
                "rejection_type": "",
                "reviewer_confidence": reviewer_confidence,
                "cross_validation_passed": cross_validation_passed,
            }
        return {
            "reviewer_feedback": (
                f"REJECTED: {verdict.failed_checks}\n"
                f"Reason: {verdict.reason}\n"
                f"Required Fix: {verdict.required_fix}"
            ),
            "revision_count": rev_count + 1,
            "rejection_type": verdict.rejection_type or "",
            "reviewer_confidence": reviewer_confidence,
            "cross_validation_passed": cross_validation_passed,
        }

    def sub_query_planner_node(state: SVFState):
        """规则化子查询生成（零 LLM 调用）

        从 Reviewer 的 required_fix 反馈中提取：
        1. 法条引用（Chapter/Section/Paragraph + 编号）
        2. 关键术语（引号内的高价值名词）
        3. Fallback: 截取反馈前 100 字符
        """
        feedback = state.get("reviewer_feedback", "")
        sub_queries = []

        # 提取法条引用
        clause_refs = re.findall(
            r'(Chapter|Section|Paragraph|Clause)\s+[\d.]+',
            feedback, re.IGNORECASE
        )
        for ref in clause_refs[:2]:
            sub_queries.append(f"{ref} requirements and compliance obligations")

        # 提取关键术语（双引号内的内容）
        key_terms = re.findall(r'"([^"]{3,40})"', feedback)
        for term in key_terms[:2]:
            sub_queries.append(term)

        # Fallback
        if not sub_queries:
            sub_queries.append(feedback[:100].strip())

        # 限制最多 3 个子查询
        sub_queries = sub_queries[:3]

        new_round = state.get("retrieval_round", 0) + 1
        print(f"[SVF][SUB_QUERY_PLANNER] Round {new_round}: generated {len(sub_queries)} sub-queries: {sub_queries}")
        return {
            "sub_queries": sub_queries,
            "retrieval_round": new_round,
        }

    # ---- 条件边 ----

    def should_retry_format(state: SVFState):
        validation_errors = state.get("validation_errors", "").strip()
        retry_count = state.get("format_retry_count", 0)
        max_format_retries = 2
        if (
            validation_errors
            and validation_errors.startswith("FORMAT_VIOLATION")
            and retry_count < max_format_retries
        ):
            return "retry_analyzer"
        return "proceed_to_reviewer"

    def route_after_review(state: SVFState) -> str:
        """三路条件边：基于结构化 rejection_type 路由"""
        rev_count = state.get("revision_count", 0)
        retrieval_round = state.get("retrieval_round", 0)
        feedback = state.get("reviewer_feedback", "")
        rejection_type = state.get("rejection_type", "")

        # 已通过或超过修订上限
        if not feedback or rev_count >= MAX_REVISIONS:
            return "end"

        # ---- 主路径：基于结构化 rejection_type ----
        if rejection_type == "insufficient_info" and retrieval_round < MAX_RETRIEVAL:
            print(f"[SVF][ROUTE] rejection_type=insufficient_info → re_retrieve (round {retrieval_round + 1})")
            return "re_retrieve"

        if rejection_type == "quality_issue":
            print(f"[SVF][ROUTE] rejection_type=quality_issue → revise (round {rev_count})")
            return "revise"

        # ---- Fallback: rejection_type 为空时退化关键词匹配 ----
        if rejection_type == "" and retrieval_round < MAX_RETRIEVAL:
            needs_retrieval_keywords = [
                "missing", "insufficient", "lack", "not found", "not available",
                "inadequate", "缺少", "不足", "未找到", "需要更多信息",
            ]
            if any(kw in feedback.lower() for kw in needs_retrieval_keywords):
                print(f"[SVF][ROUTE] Fallback keyword match → re_retrieve (round {retrieval_round + 1})")
                return "re_retrieve"

        print(f"[SVF][ROUTE] Default → revise (round {rev_count})")
        return "revise"

    # ---- 构建图 ----

    workflow = StateGraph(SVFState)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("format_validator", format_validator_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("sub_query_planner", sub_query_planner_node)

    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "format_validator")

    workflow.add_conditional_edges(
        "format_validator",
        should_retry_format,
        {"retry_analyzer": "analyzer", "proceed_to_reviewer": "reviewer"},
    )

    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "end": END,
            "revise": "analyzer",
            "re_retrieve": "sub_query_planner",
        },
    )

    workflow.add_edge("sub_query_planner", "retriever")

    return workflow.compile()


# 初始状态模板
_INITIAL_STATE = {
    "original_input": "",
    "revision_count": 0,
    "retrieval_round": 0,
    "format_retry_count": 0,
    "final_report": "",
    "reviewer_feedback": "",
    "validation_errors": "",
    "accumulated_docs": [],
    "sub_queries": [],
    "rejection_type": "",
    "reasoning_confidence": 0.5,
    "low_confidence_areas": "[]",
    "high_confidence_areas": "[]",
    "reviewer_confidence": 0.5,
    "cross_validation_passed": True,
}


def _run_svf_graph(safe_input: str) -> str:
    """同步执行 SVF 多智能体图（阻塞式端点使用）"""
    app_graph = _build_svf_graph()
    initial = {**_INITIAL_STATE, "original_input": safe_input}
    final_state = app_graph.invoke(initial)
    return final_state.get("final_report", "❌ Report generation failed.")


# ---- SSE Streaming Generator (astream_events 真异步) ----
async def _stream_svf(safe_input: str) -> AsyncGenerator[str, None]:
    """基于 astream_events 的真·实时 SSE 流

    利用 LangGraph 的 astream_events 捕获每个节点的 on_chain_start/on_chain_end，
    实时推送 agent_state 事件，而非预播报模式。
    """
    app_graph = _build_svf_graph()
    initial = {**_INITIAL_STATE, "original_input": safe_input}

    # 已发送 done 的节点集合（避免重复发送）
    completed_nodes: set = set()
    report_text = ""
    retrieval_confidence = None
    confidence_warning = None
    reasoning_confidence = None
    reviewer_confidence = None
    cross_validation_passed = None

    async for event in app_graph.astream_events(initial, version="v2"):
        kind = event.get("event", "")

        # ---- 节点开始 ----
        if kind == "on_chain_start":
            tags = event.get("tags", [])
            node_name = event.get("name", "")
            if node_name in NODE_DISPLAY_MAP and node_name not in completed_nodes:
                agent_name, msg = NODE_DISPLAY_MAP[node_name]
                yield (
                    f"event: agent_state\n"
                    f"data: {json.dumps({'agent': agent_name, 'status': 'running', 'message': msg}, ensure_ascii=False)}\n\n"
                )

        # ---- 节点完成 ----
        elif kind == "on_chain_end":
            node_name = event.get("name", "")
            if node_name in NODE_DISPLAY_MAP and node_name not in completed_nodes:
                completed_nodes.add(node_name)
                agent_name, _ = NODE_DISPLAY_MAP[node_name]

                # 检查是否有 final_report（终态）
                output_data = event.get("data", {}).get("output", {})
                if isinstance(output_data, dict):
                    # P2: 从 retriever 节点捕获检索置信度
                    if node_name == "retriever":
                        cs = output_data.get("confidence_score")
                        cw = output_data.get("confidence_warning")
                        if cs is not None:
                            retrieval_confidence = cs
                            confidence_warning = cw
                            # 实时推送检索置信度事件
                            yield (
                                f"event: confidence\n"
                                f"data: {json.dumps({'score': cs, 'warning': cw or None, 'dimension': 'retrieval'}, ensure_ascii=False)}\n\n"
                            )

                    # M2: 从 analyzer 节点捕获推理置信度
                    if node_name == "analyzer":
                        rc = output_data.get("reasoning_confidence")
                        if rc is not None:
                            reasoning_confidence = rc
                            yield (
                                f"event: confidence\n"
                                f"data: {json.dumps({'score': rc, 'dimension': 'reasoning'}, ensure_ascii=False)}\n\n"
                            )

                    # M3: 从 reviewer 节点捕获交叉验证结果
                    if node_name == "reviewer":
                        rvc = output_data.get("reviewer_confidence")
                        cvp = output_data.get("cross_validation_passed")
                        if rvc is not None:
                            reviewer_confidence = rvc
                        if cvp is not None:
                            cross_validation_passed = cvp

                        # 推送完整的三维置信度事件
                        confidence_payload = {
                            "dimension": "full",
                            "retrieval": retrieval_confidence,
                            "reasoning": reasoning_confidence,
                            "reviewer": reviewer_confidence,
                            "cross_validation_passed": cross_validation_passed,
                        }
                        yield (
                            f"event: confidence\n"
                            f"data: {json.dumps(confidence_payload, ensure_ascii=False)}\n\n"
                        )

                    if output_data.get("final_report"):
                        report_text = output_data["final_report"]

                yield (
                    f"event: agent_state\n"
                    f"data: {json.dumps({'agent': agent_name, 'status': 'done', 'message': ''}, ensure_ascii=False)}\n\n"
                )

    # 如果没有通过 on_chain_end 拿到 final_report，回退同步执行
    if not report_text:
        # 兜底：同步执行获取报告（极端情况）
        report_text = await asyncio.to_thread(_run_svf_graph, safe_input)

    # 逐行发送报告 token
    formatted = format_output(report_text)
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
    report = await asyncio.to_thread(_run_svf_graph, safe_input)
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
