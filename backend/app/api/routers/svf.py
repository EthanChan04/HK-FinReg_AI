﻿"""
SVF 鍚堣瀹℃煡璺敱 (Agentic RAG + 鍙嶆濆惊鐜?
杩佺Щ鑷?core_logic.py 涓殑 generate_risk_report 澶氭櫤鑳戒綋

P1 鍗囩骇锛?
  - 瑙勫垯鍖?SubQueryPlanner锛堥浂 LLM 璋冪敤锛?
  - 璺ㄨ疆鏂囨。绱Н鍘婚噸锛坈ontent_hash 鏇夸唬瀛楃涓叉嫾鎺ワ級
  - 涓夎矾鏉′欢杈癸紙APPROVED / revise / re_retrieve锛?
  - asyncio.to_thread 淇鍚屾闃诲
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
from app.services.workflow_checkpoint import (
    CheckpointManager,
    ReviewQueueItem,
    get_checkpoint_manager,
)

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/svf", tags=["SVF Compliance"])


# ---- LangGraph State ----
class SVFState(TypedDict, total=False):
    original_input: str
    extracted_entities: str
    query_type: str
    retrieval_mode: str
    query_profile: Dict
    retrieved_docs: str
    evidence_chunks: List[Dict]
    graph_paths: List[Dict]
    citation_audit: Dict
    unsupported_claim_rate: float
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str
    format_retry_count: int
    validation_errors: str
    # P1 鏂板锛氬弽鎬濆惊鐜?
    retrieval_round: int              # 褰撳墠妫绱㈣疆娆★紙0=棣栨锛?=鍙嶆濅簩娆℃绱級
    sub_queries: List[str]            # 瑙勫垯鍖栧瓙鏌ヨ鍒楄〃
    accumulated_docs: List[Dict]      # 璺ㄨ疆绱Н鏂囨。锛堢粨鏋勫寲锛屾浛浠ｅ瓧绗︿覆鎷兼帴锛?
    # P2 棰勭暀锛氱疆淇″害
    confidence_score: float           # 妫绱㈢疆淇″害锛圧erank Top-1 鍒嗘暟锛?
    confidence_warning: str           # 缃俊搴﹁鍛婃爣璁?
    # 深度研究与知识图谱字段
    research_plan: Dict
    evidence_by_subquestion: Dict
    evidence_gaps: List[Dict]
    # M2: Analyzer 鎺ㄧ悊缃俊搴?
    reasoning_confidence: float       # Analyzer 鑷瘎鎺ㄧ悊缃俊搴?
    low_confidence_areas: str         # 浣庣疆淇″害棰嗗煙 JSON
    high_confidence_areas: str        # 楂樼疆淇″害棰嗗煙 JSON
    # M3: Reviewer 浜ゅ弶楠岃瘉
    reviewer_confidence: float        # Reviewer 鐙珛缃俊搴︽墦鍒?
    cross_validation_passed: bool     # 鍋忓樊妫娴嬫槸鍚﹂氳繃
    # 缁撴瀯鍖栨嫆缁濈被鍨?
    rejection_type: str               # "insufficient_info" | "quality_issue" | ""
    # Phase 1: 宸ヤ綔娴佹寔涔呭寲涓?HITL
    workflow_run_id: str              # 涓娆″鏌ヤ换鍔＄殑涓婚敭
    saved_checkpoint_id: str          # Checkpoint 鏍囪瘑
    human_review_required: bool       # 鏄惁闇瑕佷汉宸ュ鏌?
    human_review_status: str          # "pending" | "approved" | "rejected"
    human_review_notes: str           # 浜哄伐鎵规敞
    resume_token: str                 # 鎭㈠鎵ц浠ょ墝
    current_gate: str                 # 瑙﹀彂鏆傚仠鐨?gate 绫诲瀷
    total_steps: int                  # 鍏ㄥ眬姝ユ暟璁℃暟鍣?


# ---- 鍙嶆濆惊鐜父閲?----
MAX_REVISIONS = 2       # 鏈澶т慨璁㈡鏁?
MAX_RETRIEVAL = 1       # 鏈澶т簩娆℃绱㈡鏁?

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
    match = re.search(r"\*\*[^*]*(?:鍫卞憡鏃ユ湡|鎶ュ憡鏃ユ湡)[^*]*\*\*\s*:\s*(.+)", markdown_text)
    if match:
        return match.group(1).strip()
    match = re.search(r"(?:鍫卞憡鏃ユ湡|鎶ュ憡鏃ユ湡)\s*:\s*(.+)", markdown_text)
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
    match = re.search(r"(低風險|中風險|高風險|低风险|中风险|高风险)", section_text)
    return match.group(1) if match else ""


def _parse_analyzer_markdown(markdown_text: str) -> dict:
    sections = _parse_markdown_sections(markdown_text)

    applicant_overview = _find_section_content(sections, "申請人概覽", "申请人概览", "Applicant Overview")
    facts_section = _find_section_content(sections, "法規事實摘要", "法规事实摘要", "Regulatory Facts")
    gap_analysis = _find_section_content(sections, "合規差距分析", "合规差距分析", "Gap Analysis")
    recommendations = _find_section_content(sections, "合規建議", "合规建议", "Compliance Recommendations")
    risk_section = _find_section_content(sections, "風險評級", "风险评级", "Risk Rating")
    insufficiency = _find_section_content(sections, "資訊不足聲明", "资料不足声明", "信息不足声明", "Insufficiency Disclaimer")

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
    """褰?ReviewerVerdict 楠岃瘉澶辫触鏃讹紝浠庡師濮嬫枃鏈帹鏂?rejection_type 浣滀负 fallback"""
    text = content.lower()
    insufficient_keywords = [
        "missing", "insufficient", "lack", "not found", "not available",
        "inadequate", "缺少", "不足", "未找到", "需要更多信息", "資訊不足", "资料不足",
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

    # 铏曠悊 markdown code block: 鍘绘帀 ```json 鍜?```
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
    """璁＄畻鏂囨。鍐呭鍝堝笇鐢ㄤ簬璺ㄨ疆鍘婚噸"""
    normalized = re.sub(r'\s+', ' ', content.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def _compute_retrieval_confidence(docs_or_accumulated: list) -> tuple:
    """璁＄畻妫绱㈢疆淇″害锛堜笁缁存寚鏍囷細Rerank Top-1 + Top-5 gap + 璀﹀憡锛?

    鏀寔 Document 鍒楄〃鍜?accumulated_docs 瀛楀吀鍒楄〃涓ょ杈撳叆銆?

    Returns:
        (top1_score, top5_gap, warning): 妫绱㈢疆淇″害鍒嗘暟銆乀op-5 gap銆佽鍛婁俊鎭?
    """
    from app.core.config import get_settings
    settings = get_settings()

    if not docs_or_accumulated:
        return 0.0, 0.0, "Low confidence warning: no relevant regulatory documents were retrieved."

    # 鍏煎 Document 鍜?dict 涓ょ绫诲瀷
    first = docs_or_accumulated[0]
    if isinstance(first, dict):
        metadata = first.get("metadata", {})
    elif hasattr(first, 'metadata'):
        metadata = first.metadata
    else:
        metadata = {}

    top1_score = float(metadata.get("rerank_score", 0.0))

    # 璁＄畻 Top-5 gap锛圱op-1 涓?Top-5 鍧囧肩殑宸窛锛?
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
            f"Low confidence warning: top rerank score is only {top1_score:.2f}. "
            "The report may lack sufficient regulatory support; review source coverage."
        )
    elif top1_score < settings.CONFIDENCE_MED_THRESHOLD:
        warning = (
            f"Confidence notice: top rerank score is {top1_score:.2f}. "
            "Review source coverage and citation completeness."
        )

    return top1_score, top5_gap, warning


# ---- Node display names for the frontend timeline ----
NODE_DISPLAY_MAP = {
    "extractor":        ("Extractor Agent",         "正在从自然语言中提取合规审查关键实体..."),
    "retrieval_router": ("Retrieval Router",        "正在判断应使用 RAG、KAG 还是 DeepResearch 检索路径..."),
    "retriever":        ("Retriever Agent",         "正在调用 ChromaDB 向量库检索 HKMA 法规条款..."),
    "analyzer":         ("Analyzer Agent",          "正在基于检索结果起草合规风险报告初稿..."),
    "format_validator": ("Format Validator Agent",  "正在验证报告结构与引用格式..."),
    "citation_verifier":("Citation Verifier Agent", "正在逐句验证引用准确性..."),
    "reviewer":         ("Reviewer Agent",          "正在执行本地引用审查与置信度校验..."),
    "sub_query_planner":("Sub-Query Planner",       "反思循环：正在规划二次检索策略..."),
}


def _evaluate_hitl_gate(state: SVFState) -> str:
    """璇勪及宸ヤ綔娴佹槸鍚﹀簲瑙﹀彂 HITL gate

    涓夌被 gate 浼樺厛绾э細
      1. low_confidence_gate: 鏁翠綋缃俊搴︿綆浜庨槇鍊?
      2. missing_evidence_gate: 璇佹嵁涓嶈冻涓斿凡杈惧埌妫绱笂闄?
      3. manual_approval_gate: 楂橀闄╁満鏅紙榛樿鏈惎鐢紝鐢变笟鍔￠厤缃Е鍙戯級

    Returns:
        Gate 绫诲瀷瀛楃涓诧紝鎴栫┖瀛楃涓诧紙鏃?gate 瑙﹀彂锛?
    """
    from app.core.config import get_settings
    settings = get_settings()

    retrieval_conf = state.get("confidence_score", 0.0)
    reasoning_conf = state.get("reasoning_confidence", 0.5)
    cross_validated = state.get("cross_validation_passed", True)
    rejection_type = state.get("rejection_type", "")
    retrieval_round = state.get("retrieval_round", 0)

    # 璁＄畻 overall confidence锛堜笌 ConfidenceScore.overall_confidence 閫昏緫涓鑷达級
    overall = retrieval_conf * 0.4 + reasoning_conf * 0.6
    if not cross_validated:
        overall = max(0.0, overall - 0.2)

    # Gate 1: 浣庣疆淇″害
    if overall < settings.CONFIDENCE_LOW_THRESHOLD:
        return "low_confidence_gate"

    # Gate 2: 璇佹嵁涓嶈冻涓斿凡杈惧埌妫绱笂闄?
    if rejection_type == "insufficient_info" and retrieval_round >= MAX_RETRIEVAL:
        return "missing_evidence_gate"

    # Gate 3: 浜ゅ弶楠岃瘉澶辫触锛堥珮椋庨櫓淇″彿锛?
    if not cross_validated and overall < settings.CONFIDENCE_MED_THRESHOLD:
        return "manual_approval_gate"

    return ""


NODE_DISPLAY_MAP["hitl_gate"] = ("HITL Gate", "姝ｅ湪璇勪及鏄惁闇瑕佷汉宸ュ鏌?..")


def _build_svf_graph(checkpointer=None):
    """鏋勫缓 SVF 澶氭櫤鑳戒綋鍥撅紙鍚弽鎬濆惊鐜級锛岃繑鍥炵紪璇戝悗鐨?CompiledGraph

    Args:
        checkpointer: LangGraph checkpointer 瀹炰緥锛圥ostgresSaver / MemorySaver锛夛紝
                      涓?None 鏃朵笉鍚敤鎸佷箙鍖?
    """
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
            "total_steps": state.get("total_steps", 0) + 1,
        }

    def retrieval_router_node(state: SVFState):
        """Classify the query and store the intended retrieval mode/profile."""
        from app.services.retrieval.retrieval_router import route_query

        query = state.get("extracted_entities") or state["original_input"]
        profile = route_query(query)
        mode = profile.retrieval_mode
        # Keep the SVF endpoint bounded: complex research requests use the
        # dedicated /research API, while SVF falls back to KAG/RAG evidence.
        if mode == "deep_research":
            mode = "kag"
        return {
            "retrieval_mode": mode,
            "query_profile": profile.model_dump(),
            "total_steps": state.get("total_steps", 0) + 1,
        }

    def retriever_node(state: SVFState):
        """Retrieve evidence documents for the SVF workflow."""
        from app.services.semantic_cache import get_semantic_cache
        from app.services.retrieval.evidence_renderer import render_evidence_context
        from app.services.retrieval.retrieval_service import RetrievalService, document_to_evidence

        if base_retriever is None:
            return {"retrieved_docs": "RAG engine not available."}

        cache = get_semantic_cache()
        is_re_retrieve = state.get("retrieval_round", 0) > 0

        if is_re_retrieve:
            # ---- 鍙嶆濅簩娆℃绱細瀵规瘡涓瓙鏌ヨ鍒嗗埆妫绱?----
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

            # 鍚堝苟鍒扮疮绉枃妗?
            accumulated = list(state.get("accumulated_docs", []))
            for doc in new_docs_raw:
                chash = _compute_content_hash(doc.page_content)
                accumulated.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "content_hash": chash,
                })

            # 閲嶆柊鏍煎紡鍖栦笂涓嬫枃
            context_parts = []
            for i, d in enumerate(accumulated):
                page = d["metadata"].get("page", "?")
                score = d["metadata"].get("rerank_score", "-")
                context_parts.append(
                    f"[Source {i+1}, p.{page}, relevance={score}]\n{d['page_content']}"
                )
            context = "\n\n---\n\n".join(context_parts)
            evidence_chunks = []
            for i, d in enumerate(accumulated, start=1):
                from langchain_core.documents import Document

                evidence_chunks.append(
                    document_to_evidence(
                        Document(page_content=d["page_content"], metadata=d["metadata"]),
                        i,
                        retrieval_method="hybrid",
                    ).model_dump()
                )

            print(f"[SVF][RETRIEVER] Re-retrieval: {len(new_docs_raw)} new docs, {len(accumulated)} total accumulated.")

            # P2: 閲嶆柊璁＄畻缃俊搴︼紙鍩轰簬鍏ㄩ噺绱Н鏂囨。锛?
            score, top5_gap, warning = _compute_retrieval_confidence(accumulated)
            return {
                "retrieved_docs": context,
                "accumulated_docs": accumulated,
                "evidence_chunks": evidence_chunks,
                "confidence_score": score,
                "confidence_warning": warning or "",
                "total_steps": state.get("total_steps", 0) + 1,
            }

        else:
            # ---- 棣栨妫绱?----
            query = state.get("extracted_entities", state["original_input"])
            retrieval_mode = state.get("retrieval_mode", "rag")
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

            graph_paths = []
            if retrieval_mode == "kag":
                try:
                    from app.services.corpus.manifest_loader import load_source_manifest
                    from app.services.kag.graph_builder import build_graph_from_sources
                    from app.services.kag.graph_retriever import GraphRetriever
                    from app.core.config import get_settings

                    settings = get_settings()
                    source_docs = load_source_manifest()
                    graph_store = build_graph_from_sources(source_docs, [], settings.GRAPH_STORE_PATH)
                    graph_paths = GraphRetriever(graph_store).retrieve_paths(query, limit=5)
                except Exception as exc:
                    print(f"[SVF][KAG] graph retrieval unavailable: {exc}")

            # Structured storage for retrieved documents.
            accumulated = []
            evidence_chunks = []
            for i, doc in enumerate(top_docs):
                chash = _compute_content_hash(doc.page_content)
                accumulated.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "content_hash": chash,
                })
                evidence_chunks.append(
                    document_to_evidence(
                        doc,
                        i + 1,
                        retrieval_method="graph" if retrieval_mode == "kag" else "hybrid",
                    )
                )
            context = render_evidence_context(evidence_chunks)

            print(f"[SVF][RETRIEVER] Initial: selected {len(top_docs)} top docs for Analyzer.")

            # P2: 璁＄畻缃俊搴?
            score, top5_gap, warning = _compute_retrieval_confidence(top_docs)
            print(f"[SVF][RETRIEVER] Confidence: score={score:.2f}, top5_gap={top5_gap:.2f}, warning={'Yes' if warning else 'None'}")
            return {
                "retrieved_docs": context,
                "query_type": query_type,
                "evidence_chunks": [chunk.model_dump() for chunk in evidence_chunks],
                "graph_paths": graph_paths,
                "accumulated_docs": accumulated,
                "confidence_score": score,
                "confidence_warning": warning or "",
                "total_steps": state.get("total_steps", 0) + 1,
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

        # M2: 瑙ｆ瀽 Analyzer 缃俊搴﹁嚜璇?JSON 鍧?
        reasoning_confidence = 0.5
        low_confidence_areas = "[]"
        high_confidence_areas = "[]"
        confidence_match = re.search(
            r'```json\s*\n?\s*(\{[^`]*?"overall_confidence"\s*:[^}]*\})\s*\n?\s*```',
            content, re.DOTALL,
        )
        if not confidence_match:
            # Fallback: 灏濊瘯鍖归厤瑁?JSON
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

        # 浠庢姤鍛婁腑绉婚櫎缃俊搴?JSON 鍧楋紙閬垮厤鍑虹幇鍦ㄦ渶缁堟姤鍛婁腑锛?
        content = re.sub(
            r'```json\s*\n?\s*\{[^`]*?"overall_confidence"[^}]*\}\s*\n?\s*```',
            '', content, flags=re.DOTALL,
        ).rstrip()
        # 涔熺Щ闄よ８ JSON 鍧?
        content = re.sub(
            r'\n\s*\{[^{}]*"overall_confidence"\s*:\s*[\d.]+[^{}]*\}\s*$',
            '', content,
        ).rstrip()

        return {
            "draft_report": content,
            "reasoning_confidence": reasoning_confidence,
            "low_confidence_areas": low_confidence_areas,
            "high_confidence_areas": high_confidence_areas,
            "total_steps": state.get("total_steps", 0) + 1,
        }

    def format_validator_node(state: SVFState):
        draft = state.get("draft_report", "")
        retry_count = state.get("format_retry_count", 0)
        max_format_retries = 2

        try:
            parsed = _parse_analyzer_markdown(draft)
            AnalyzerOutput(**parsed)
            return {
                "format_retry_count": retry_count, 
                "validation_errors": "",
                "total_steps": state.get("total_steps", 0) + 1,
            }
        except ValidationError as exc:
            if retry_count >= max_format_retries:
                return {
                    "format_retry_count": retry_count,
                    "validation_errors": (
                        "FORMAT_VALIDATION_FAILED: report structure is invalid after max retries. "
                        "Please regenerate with strict section headings and source citations."
                    ),
                    "total_steps": state.get("total_steps", 0) + 1,
                }
            return {
                "format_retry_count": retry_count + 1,
                "validation_errors": f"FORMAT_VIOLATION ({len(exc.errors())} errors)\n{exc}",
                "total_steps": state.get("total_steps", 0) + 1,
            }

    def citation_verifier_node(state: SVFState):
        from app.schemas.evidence import EvidenceChunk
        from app.services.retrieval.citation_verifier import verify_citations
        draft = state.get("draft_report", "")
        evidence_chunks = [
            EvidenceChunk(**item)
            for item in state.get("evidence_chunks", [])
        ]
        citation_audit = verify_citations(draft, evidence_chunks)
        return {
            "citation_audit": citation_audit.model_dump(),
            "unsupported_claim_rate": citation_audit.unsupported_claim_rate,
        }

    def reviewer_node(state: SVFState):
        from app.services.agents.prompts import REVIEWER_SYSTEM_PROMPT
        rev_count = state.get("revision_count", 0)
        draft = state.get("draft_report", "")
        reviewer_confidence = max(0.0, min(1.0, float(state.get("reasoning_confidence", 0.5))))
        verdict = ReviewerVerdict(
            decision="APPROVED",
            failed_checks=[],
            reviewer_confidence=reviewer_confidence,
        )
        raw_content = None  # 淇濆瓨 LLM 鍘熷杈撳嚭锛岄伩鍏嶉噸澶嶈皟鐢?

        # Use a single plain reviewer call. Some compatible endpoints expose
        # structured-output wrappers but hang or time out before falling back,
        # which doubles the reviewer latency in the streaming workflow.
        structured_llm = None
        if structured_llm is not None:
            try:
                result = structured_llm.invoke([HumanMessage(content=REVIEWER_SYSTEM_PROMPT.format(draft_report=draft))])
                if isinstance(result, ReviewerVerdict):
                    verdict = result
                else:
                    # structured output 杩斿洖浜嗗師濮嬪唴瀹硅岄潪 ReviewerVerdict锛?
                    # 淇濆瓨浠ヤ緵 fallback 姝ｅ垯瑙ｆ瀽澶嶇敤锛岄伩鍏嶅啀璋冧竴娆?LLM
                    raw_content = str(result) if result else None
            except Exception as e:
                print(f"[SVF][REVIEWER] structured output failed, falling back: {e}")

        # Fallback: 姝ｅ垯瑙ｆ瀽锛堝鐢?structured_llm 鐨勫師濮嬭緭鍑猴紝閬垮厤浜屾 LLM 璋冪敤锛?
        reviewer_confidence = 0.5
        if verdict is None:
            if raw_content is None:
                # 鍙湪娌℃湁 structured_llm 鎴?structured_llm 鎶涘紓甯告椂鎵嶈皟 LLM
                resp = llm.invoke([HumanMessage(content=REVIEWER_SYSTEM_PROMPT.format(draft_report=draft))])
                raw_content = resp.content.strip()

            content = raw_content.strip()
            # 鍘婚櫎 markdown 浠ｇ爜鍧楀寘瑁癸紙LLM 鏈夋椂杩斿洖 ```...\n```锛?
            if content.startswith("```"):
                lines = content.splitlines()
                inner = [l for l in lines[1:] if not l.strip().startswith("```")]
                content = "\n".join(inner).strip()
            # M3: 鎻愬彇 Reviewer Confidence 鎵撳垎
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
            # M3: 浠庣粨鏋勫寲杈撳嚭涓幏鍙?reviewer_confidence
            reviewer_confidence = verdict.reviewer_confidence or 0.5

        # M3: 浜ゅ弶楠岃瘉 鈥?妫娴?retrieval 涓?reasoning 缃俊搴﹀亸宸?
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

        # 鏋勫缓 ConfidenceScore 涓夌淮妯″瀷
        from app.schemas.requests import ConfidenceScore
        top5_gap = 0.0  # top5_gap 瀛樺湪浜?retriever 浣嗕笉鍦?state 涓紝鐢?0 鍗犱綅
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

        # 淇锛氬綋鍗冲皢杈惧埌淇涓婇檺鏃朵篃杈撳嚭 final_report
        # 鍘熼昏緫 rev_count >= MAX_REVISIONS 姘歌繙涓嶄細涓虹湡锛?
        # 鍥犱负 route_after_review 浼氬湪 rev_count 杈惧埌涓婇檺鏃舵嫤鎴矾鐢卞埌 "end"銆?
        # 鏀逛负 rev_count >= MAX_REVISIONS - 1锛岀‘淇濇渶鍚庝竴杞?Reviewer 杈撳嚭 final_report銆?
        if verdict.decision == "APPROVED" or rev_count >= MAX_REVISIONS - 1:
            # P2: 缃俊搴﹂潤鎬佹爣璁帮紙鍙锛屼笉鍙備笌鎺у埗娴侊級
            confidence_warning = state.get("confidence_warning", "")
            final = draft
            warning_parts = []
            if confidence_warning:
                warning_parts.append(confidence_warning)
            if not cross_validation_passed:
                warning_parts.append(
                    f"Cross-validation warning: retrieval confidence ({retrieval_conf:.2f}) and reasoning confidence ({reasoning_conf:.2f}) "
                    f"differ significantly ({deviation:.2f}); the report may be overconfident or missing evidence."
                )
            if warning_parts:
                final = f"{draft}\n\n---\n" + "\n\n".join(warning_parts)

            return {
                "final_report": final,
                "revision_count": rev_count,
                "reviewer_feedback": "",
                "rejection_type": "",
                "reviewer_confidence": reviewer_confidence,
                "cross_validation_passed": cross_validation_passed,
                "citation_audit": state.get("citation_audit", {}),
                "unsupported_claim_rate": state.get("unsupported_claim_rate", 0.0),
                "total_steps": state.get("total_steps", 0) + 1,
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
            "citation_audit": state.get("citation_audit", {}),
            "unsupported_claim_rate": state.get("unsupported_claim_rate", 0.0),
            "total_steps": state.get("total_steps", 0) + 1,
        }

    def sub_query_planner_node(state: SVFState):
        """Generate deterministic follow-up retrieval queries from reviewer feedback."""






        feedback = state.get("reviewer_feedback", "")
        sub_queries = []

        # 鎻愬彇娉曟潯寮曠敤
        clause_refs = re.findall(
            r'(Chapter|Section|Paragraph|Clause)\s+[\d.]+',
            feedback, re.IGNORECASE
        )
        for ref in clause_refs[:2]:
            sub_queries.append(f"{ref} requirements and compliance obligations")

        # 鎻愬彇鍏抽敭鏈锛堝弻寮曞彿鍐呯殑鍐呭锛?
        key_terms = re.findall(r'"([^"]{3,40})"', feedback)
        for term in key_terms[:2]:
            sub_queries.append(term)

        # Fallback
        if not sub_queries:
            sub_queries.append(feedback[:100].strip())

        # 闄愬埗鏈澶?3 涓瓙鏌ヨ
        sub_queries = sub_queries[:3]

        new_round = state.get("retrieval_round", 0) + 1
        print(f"[SVF][SUB_QUERY_PLANNER] Round {new_round}: generated {len(sub_queries)} sub-queries: {sub_queries}")
        return {
            "sub_queries": sub_queries,
            "retrieval_round": new_round,
            "total_steps": state.get("total_steps", 0) + 1,
        }

    # ---- 鏉′欢杈?----

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
        """Route after reviewer decision, with optional HITL gate."""








        rev_count = state.get("revision_count", 0)
        retrieval_round = state.get("retrieval_round", 0)
        feedback = state.get("reviewer_feedback", "")
        rejection_type = state.get("rejection_type", "")
        total_steps = state.get("total_steps", 0)
        final_report = state.get("final_report", "")
        if final_report:
            gate = _evaluate_hitl_gate(state)
            if gate:
                print(f"[SVF][ROUTE] Final report ready but HITL gate triggered: {gate} 鈫?hitl_gate")
                return "hitl_gate"
            print("[SVF][ROUTE] Final report ready 鈫?end")
            return "end"

        # 瀹夊叏鍏滃簳锛氫换涓寰幆杈惧埌涓婇檺鎴栨绘鏁拌秴闄愰兘寮哄埗缁撴潫
        if rev_count >= MAX_REVISIONS:
            print(f"[SVF][ROUTE] Revision limit reached (rev={rev_count}) 鈫?end")
            return "end"
        if retrieval_round >= MAX_RETRIEVAL and rejection_type == "insufficient_info":
            print(f"[SVF][ROUTE] Retrieval limit reached (round={retrieval_round}) 鈫?end")
            return "end"
        if total_steps >= 80:
            print(f"[SVF][ROUTE] Total steps limit reached (steps={total_steps}) 鈫?end")
            return "end"

        # 闇瑕佷慨璁細缁х画鍙嶆濆惊鐜?
        if feedback and rev_count < MAX_REVISIONS:
            # ---- 涓昏矾寰勶細鍩轰簬缁撴瀯鍖?rejection_type ----
            if rejection_type == "insufficient_info" and retrieval_round < MAX_RETRIEVAL:
                print(f"[SVF][ROUTE] rejection_type=insufficient_info 鈫?re_retrieve (round {retrieval_round + 1})")
                return "re_retrieve"

            if rejection_type == "quality_issue":
                print(f"[SVF][ROUTE] rejection_type=quality_issue 鈫?revise (round {rev_count})")
                return "revise"

            # ---- Fallback: rejection_type 涓虹┖鏃堕鍖栧叧閿瘝鍖归厤 ----
            if rejection_type == "" and retrieval_round < MAX_RETRIEVAL:
                needs_retrieval_keywords = [
                    "missing", "insufficient", "lack", "not found", "not available",
                    "inadequate", "缺少", "不足", "未找到", "需要更多信息", "資訊不足", "资料不足",
                ]
                if any(kw in feedback.lower() for kw in needs_retrieval_keywords):
                    print(f"[SVF][ROUTE] Fallback keyword match 鈫?re_retrieve (round {retrieval_round + 1})")
                    return "re_retrieve"

            print(f"[SVF][ROUTE] Default 鈫?revise (round {rev_count})")
            return "revise"

        # 鍗冲皢缁撴潫锛氭鏌?HITL gate
        gate = _evaluate_hitl_gate(state)
        if gate:
            print(f"[SVF][ROUTE] HITL gate triggered: {gate} 鈫?hitl_gate")
            return "hitl_gate"

        # 鏃?gate 瑙﹀彂锛屾甯哥粨鏉?
        return "end"

    # ---- 鏋勫缓鍥?----

    def hitl_gate_node(state: SVFState):
        """Pause the workflow for human review when a HITL gate is triggered."""












        gate = _evaluate_hitl_gate(state)
        if not gate:
            return {
                "human_review_required": False,
                "current_gate": "",
                "total_steps": state.get("total_steps", 0) + 1,
            }

        workflow_run_id = state.get("workflow_run_id", "")
        print(f"[SVF][HITL_GATE] Gate triggered: {gate}, workflow_run_id={workflow_run_id}")

        # 鏋勫缓瀹屾暣鐨?interrupt payload锛堟惡甯︾湡瀹炶瘉鎹拰缃俊搴︼級
        evidence_snapshot = {
            "accumulated_docs_count": len(state.get("accumulated_docs", [])),
            "retrieval_round": state.get("retrieval_round", 0),
            "rejection_type": state.get("rejection_type", ""),
            "documents": [
                {
                    "source_idx": i,
                    "page": d.get("metadata", {}).get("page", "?"),
                    "rerank_score": d.get("metadata", {}).get("rerank_score", "-"),
                    "content_preview": d.get("page_content", "")[:200],
                }
                for i, d in enumerate(state.get("accumulated_docs", [])[:5])
            ],
        }
        confidence_data = {
            "retrieval_confidence": state.get("confidence_score", 0.0),
            "reasoning_confidence": state.get("reasoning_confidence", 0.5),
            "reviewer_confidence": state.get("reviewer_confidence", 0.5),
            "cross_validation_passed": state.get("cross_validation_passed", True),
            "confidence_warning": state.get("confidence_warning", ""),
        }

        interrupt_payload = {
            "workflow_run_id": workflow_run_id,
            "gate_type": gate,
            "message": f"宸ヤ綔娴佸凡鏆傚仠锛岀瓑寰呬汉宸ュ鏌?(gate: {gate})",
            "evidence_snapshot": evidence_snapshot,
            "confidence_data": confidence_data,
            "latest_draft_report": state.get("draft_report", "")[:500],
        }

        # 鍔犲叆瀹℃煡闃熷垪锛堝唴瀛樼储寮曪紝checkpoint 鐢?checkpointer 鎸佷箙鍖栵級
        manager = get_checkpoint_manager()
        item = ReviewQueueItem(
            workflow_run_id=workflow_run_id,
            gate_type=gate,
            checkpoint_created_at=time.time(),
            evidence_snapshot=evidence_snapshot,
            latest_draft_report=state.get("draft_report", ""),
            confidence_data=confidence_data,
            original_input=state.get("original_input", ""),
        )
        manager.add_to_review_queue(item)

        # ===== 鐪熸鐨?interrupt锛氬浘鎵ц鍦ㄦ鏆傚仠 =====
        # interrupt() 鎶涘嚭 GraphInterrupt锛宑heckpointer 淇濆瓨褰撳墠鐘舵?
        # 鎭㈠鏃惰皟鐢ㄦ柟浼犲叆 Command(resume=human_decision)
        # interrupt() 杩斿洖 human_decision锛岃妭鐐圭户缁墽琛?
        human_decision = interrupt(interrupt_payload)

        # ===== 鎭㈠鍚庝粠杩欓噷缁х画 =====
        action = "reject"
        notes = ""
        additional_context = None
        if isinstance(human_decision, dict):
            action = human_decision.get("action", "reject")
            notes = human_decision.get("notes", "")
            additional_context = human_decision.get("additional_context")
        else:
            notes = str(human_decision)

        print(f"[SVF][HITL_GATE] Resumed with action={action}, workflow_run_id={workflow_run_id}")

        # 鏇存柊瀹℃煡闃熷垪鐘舵?
        manager.update_review_status(
            workflow_run_id=workflow_run_id,
            status="approved" if action == "approve" else "rejected",
            notes=notes,
        )

        if action == "approve":
            # 娑堣垂 additional_context锛氭敞鍏ュ埌 original_input 涓緵涓嬫父鑺傜偣寮曠敤
            updated_input = state.get("original_input", "")
            if additional_context and additional_context.strip():
                updated_input += f"\n\n[浜哄伐琛ュ厖涓婁笅鏂嘳: {additional_context.strip()}"
                print(f"[SVF][HITL_GATE] Additional context injected ({len(additional_context)} chars)")

            # 鎵瑰噯鍚庢彁鍗囩疆淇″害锛堜汉宸ヨ儗涔︼級锛屾竻闄?gate 鏍囪
            return {
                "human_review_required": False,
                "human_review_status": "approved",
                "human_review_notes": notes,
                "current_gate": "",
                "original_input": updated_input,
                "confidence_warning": "",  # 浜哄伐鎵瑰噯鍚庢竻闄よ鍛?
                "final_report": state.get("final_report", "") or state.get("draft_report", ""),
                "total_steps": state.get("total_steps", 0) + 1,
            }
        else:
            return {
                "human_review_required": False,
                "human_review_status": "rejected",
                "human_review_notes": notes,
                "current_gate": "",
                "final_report": f"鉂?鎶ュ憡宸茶浜哄伐椹冲洖銆傚師鍥狅細{notes}",
                "total_steps": state.get("total_steps", 0) + 1,
            }

    workflow = StateGraph(SVFState)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("retrieval_router", retrieval_router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("format_validator", format_validator_node)
    workflow.add_node("citation_verifier", citation_verifier_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("sub_query_planner", sub_query_planner_node)
    workflow.add_node("hitl_gate", hitl_gate_node)

    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "retrieval_router")
    workflow.add_edge("retrieval_router", "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "format_validator")

    workflow.add_conditional_edges(
        "format_validator",
        should_retry_format,
        {"retry_analyzer": "analyzer", "proceed_to_reviewer": "citation_verifier"},
    )

    workflow.add_edge("citation_verifier", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "end": END,
            "revise": "analyzer",
            "re_retrieve": "sub_query_planner",
            "hitl_gate": "hitl_gate",
        },
    )

    workflow.add_edge("hitl_gate", END)
    workflow.add_edge("sub_query_planner", "retriever")

    return workflow.compile(checkpointer=checkpointer)


# 鍒濆鐘舵佹ā鏉?
_INITIAL_STATE = {
    "original_input": "",
    "revision_count": 0,
    "retrieval_round": 0,
    "format_retry_count": 0,
    "final_report": "",
    "reviewer_feedback": "",
    "validation_errors": "",
    "accumulated_docs": [],
    "retrieval_mode": "rag",
    "query_profile": {},
    "evidence_chunks": [],
    "graph_paths": [],
    "research_plan": {},
    "evidence_by_subquestion": {},
    "evidence_gaps": [],
    "citation_audit": {},
    "unsupported_claim_rate": 0.0,
    "sub_queries": [],
    "rejection_type": "",
    "reasoning_confidence": 0.5,
    "low_confidence_areas": "[]",
    "high_confidence_areas": "[]",
    "reviewer_confidence": 0.5,
    "cross_validation_passed": True,
    # Phase 1: 鎸佷箙鍖栦笌 HITL
    "workflow_run_id": "",
    "saved_checkpoint_id": "",
    "human_review_required": False,
    "human_review_status": "",
    "human_review_notes": "",
    "resume_token": "",
    "current_gate": "",
    "total_steps": 0,
}


def _run_svf_graph(safe_input: str, workflow_run_id: str = "") -> str:
    """Run the SVF graph synchronously for the blocking endpoint."""










    manager = get_checkpoint_manager()
    checkpointer = manager.get_checkpointer() if manager else None

    if not workflow_run_id:
        workflow_run_id = CheckpointManager.generate_workflow_run_id()

    app_graph = _build_svf_graph(checkpointer=checkpointer)
    initial = {**_INITIAL_STATE, "original_input": safe_input, "workflow_run_id": workflow_run_id}

    config = {"configurable": {"thread_id": workflow_run_id}, "recursion_limit": 100}
    final_state = app_graph.invoke(initial, config=config)

    # 妫娴?interrupt锛歩nvoke() 鍦?interrupt() 澶勮繑鍥炲綋鍓?state
    # 姝ゆ椂 state.next 涓嶄负绌猴紙杩樻湁寰呮墽琛岃妭鐐癸級锛宼asks 涓湁 interrupt 淇℃伅
    state_snapshot = app_graph.get_state(config)
    for task in state_snapshot.tasks:
        if task.interrupt:
            interrupt_value = task.interrupt.value
            gate = interrupt_value.get("gate_type", "unknown")
            return (
                f"鈴革笍 宸ヤ綔娴佸凡鏆傚仠锛岀瓑寰呬汉宸ュ鏌ャ俓n\n"
                f"**鏆傚仠鍘熷洜**: {gate}\n"
                f"**Workflow Run ID**: {workflow_run_id}\n\n"
                f"璇烽氳繃瀹℃煡闃熷垪 API 鎭㈠鎵ц锛歕n"
                f"- 鎵瑰噯: POST /api/v1/review-queue/{workflow_run_id}/resume\n"
                f"- 椹冲洖: POST /api/v1/review-queue/{workflow_run_id}/reject"
            )

    return final_state.get("final_report", "鉂?Report generation failed.")


# ---- SSE Streaming Generator (astream_events 鐪熷紓姝? ----
def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_keepalive() -> str:
    return ": keepalive\n\n"


GRAPH_BUILD_KEEPALIVE_INTERVAL_SECONDS = 5.0
GRAPH_EVENT_KEEPALIVE_INTERVAL_SECONDS = 5.0


async def _iter_graph_events_with_keepalive(event_source):
    """Yield graph events, inserting None when the next event is slow."""
    iterator = event_source.__aiter__()
    while True:
        next_event = asyncio.create_task(iterator.__anext__())
        while True:
            done, _ = await asyncio.wait(
                {next_event},
                timeout=GRAPH_EVENT_KEEPALIVE_INTERVAL_SECONDS,
            )
            if next_event in done:
                break
            yield None

        try:
            yield next_event.result()
        except StopAsyncIteration:
            return


async def _stream_svf_impl(safe_input: str, workflow_run_id: str = "") -> AsyncGenerator[str, None]:
    """Stream SVF workflow events as server-sent events."""










    from app.services.workflow_checkpoint import get_checkpoint_manager, CheckpointManager

    manager = get_checkpoint_manager()
    checkpointer = manager.get_checkpointer() if manager else None

    if not workflow_run_id:
        workflow_run_id = CheckpointManager.generate_workflow_run_id()
    
    print(f"[SVF][STREAM] Workflow Run ID: {workflow_run_id}")

    # Phase 1: 鎺ㄩ?workflow_run_id锛屼緵鍓嶇淇濆瓨
    yield (
        f"event: checkpoint_saved\n"
        f"data: {json.dumps({'workflow_run_id': workflow_run_id, 'status': 'started'}, ensure_ascii=False)}\n\n"
    )
    yield _sse_keepalive()

    build_task = asyncio.create_task(
        asyncio.to_thread(_build_svf_graph, checkpointer=checkpointer)
    )
    while not build_task.done():
        await asyncio.sleep(GRAPH_BUILD_KEEPALIVE_INTERVAL_SECONDS)
        if not build_task.done():
            yield _sse_keepalive()
    app_graph = await build_task
    initial = {**_INITIAL_STATE, "original_input": safe_input, "workflow_run_id": workflow_run_id}

    config = {"configurable": {"thread_id": workflow_run_id}, "recursion_limit": 100}

    # 宸插彂閫?done 鐨勮妭鐐归泦鍚堬紙閬垮厤閲嶅鍙戦侊級
    completed_nodes: set = set()
    report_text = ""
    latest_draft_report = ""
    retrieval_confidence = None
    confidence_warning = None
    reasoning_confidence = None
    reviewer_confidence = None
    cross_validation_passed = None
    latest_evidence_chunks = []
    latest_graph_paths = []
    hitl_gate_started = False
    chat_stream_event_count = 0
    
    print(f"[SVF][STREAM] Entering astream_events loop")

    event_source = app_graph.astream_events(initial, config=config, version="v2")
    async for event in _iter_graph_events_with_keepalive(event_source):
        if event is None:
            yield _sse_keepalive()
            continue

        kind = event.get("event", "")
        if kind == "on_chat_model_stream":
            chat_stream_event_count += 1
            if chat_stream_event_count == 1 or chat_stream_event_count % 20 == 0:
                yield _sse_keepalive()
            continue
        print(f"[SVF][STREAM] Received event: {kind}")

        # ---- 鑺傜偣寮濮?----
        if kind == "on_chain_start":
            tags = event.get("tags", [])
            node_name = event.get("name", "")
            if node_name in NODE_DISPLAY_MAP and node_name not in completed_nodes:
                agent_name, msg = NODE_DISPLAY_MAP[node_name]
                if node_name == "hitl_gate":
                    hitl_gate_started = True
                yield (
                    f"event: agent_state\n"
                    f"data: {json.dumps({'agent': agent_name, 'status': 'running', 'message': msg}, ensure_ascii=False)}\n\n"
                )

        # ---- 鑺傜偣瀹屾垚 ----
        elif kind == "on_chain_end":
            node_name = event.get("name", "")
            if node_name in NODE_DISPLAY_MAP and node_name not in completed_nodes:
                completed_nodes.add(node_name)
                agent_name, _ = NODE_DISPLAY_MAP[node_name]

                # 妫鏌ユ槸鍚︽湁 final_report锛堢粓鎬侊級
                output_data = event.get("data", {}).get("output", {})
                if isinstance(output_data, dict):
                    # P2: 浠?retriever 鑺傜偣鎹曡幏妫绱㈢疆淇″害
                    if node_name == "retriever":
                        evidence_chunks = output_data.get("evidence_chunks")
                        if isinstance(evidence_chunks, list):
                            latest_evidence_chunks = evidence_chunks
                            yield _sse_event("evidence_chunks", evidence_chunks)

                        graph_paths = output_data.get("graph_paths")
                        if isinstance(graph_paths, list):
                            latest_graph_paths = graph_paths
                            yield _sse_event("graph_paths", graph_paths)

                        cs = output_data.get("confidence_score")
                        cw = output_data.get("confidence_warning")
                        if cs is not None:
                            retrieval_confidence = cs
                            confidence_warning = cw
                            # 瀹炴椂鎺ㄩ佹绱㈢疆淇″害浜嬩欢
                            yield (
                                f"event: confidence\n"
                                f"data: {json.dumps({'score': cs, 'warning': cw or None, 'dimension': 'retrieval'}, ensure_ascii=False)}\n\n"
                            )

                    # M2: 浠?analyzer 鑺傜偣鎹曡幏鎺ㄧ悊缃俊搴?
                    if node_name == "analyzer":
                        rc = output_data.get("reasoning_confidence")
                        if rc is not None:
                            reasoning_confidence = rc
                            yield (
                                f"event: confidence\n"
                                f"data: {json.dumps({'score': rc, 'dimension': 'reasoning'}, ensure_ascii=False)}\n\n"
                            )

                    # M3: 浠?reviewer 鑺傜偣鎹曡幏浜ゅ弶楠岃瘉缁撴灉
                    if node_name == "reviewer":
                        rvc = output_data.get("reviewer_confidence")
                        cvp = output_data.get("cross_validation_passed")
                        if rvc is not None:
                            reviewer_confidence = rvc
                        if cvp is not None:
                            cross_validation_passed = cvp

                        # 鎺ㄩ佸畬鏁寸殑涓夌淮缃俊搴︿簨浠?
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
                    if output_data.get("draft_report"):
                        latest_draft_report = output_data["draft_report"]

                yield (
                    f"event: agent_state\n"
                    f"data: {json.dumps({'agent': agent_name, 'status': 'done', 'message': ''}, ensure_ascii=False)}\n\n"
                )

    # ===== interrupt 妫娴?=====
    # hitl_gate 鑺傜偣璋冪敤 interrupt() 鍚庯紝astream_events 娴佹甯哥粨鏉燂紝
    # 浣?hitl_gate 涓嶄細鏈?on_chain_end 浜嬩欢銆傞氳繃妫娴?graph state 鍒ゆ柇鏄惁琚腑鏂?
    if hitl_gate_started and "hitl_gate" not in completed_nodes:
        try:
            state_snapshot = await asyncio.to_thread(app_graph.get_state, config)
            for task in state_snapshot.tasks:
                if task.interrupt:
                    # 浠?interrupt payload 涓幏鍙栧畬鏁寸殑 evidence / confidence 鏁版嵁
                    interrupt_value = task.interrupt.value
                    action_payload = {
                        "workflow_run_id": workflow_run_id,
                        "gate_type": interrupt_value.get("gate_type", ""),
                        "message": interrupt_value.get("message", "Workflow paused; waiting for human review"),
                        "evidence_snapshot": interrupt_value.get("evidence_snapshot", {}),
                        "confidence_data": interrupt_value.get("confidence_data", {}),
                        "latest_draft_report": interrupt_value.get("latest_draft_report", ""),
                    }
                    yield (
                        f"event: action_required\n"
                        f"data: {json.dumps(action_payload, ensure_ascii=False)}\n\n"
                    )
                    # 涓柇鎬侊細涓嶆帹閫?token 鍜?done锛屾祦缁撴潫
                    return
        except Exception as e:
            print(f"[SVF][STREAM] Failed to check interrupt state: {e}")

    if not report_text:
        report_text = latest_draft_report or (
            "SVF compliance review did not produce a final report. "
            "Please retry with more applicant details or review the server logs."
        )

    # 閫愯鍙戦佹姤鍛?token
    formatted = format_output(report_text)
    for line in formatted.split("\n"):
        yield f"event: token\ndata: {json.dumps({'text': line + chr(10)}, ensure_ascii=False)}\n\n"

    yield _sse_event(
        "done",
        {
            "status": "complete",
            "workflow_run_id": workflow_run_id,
            "evidence_chunks": latest_evidence_chunks,
            "graph_paths": latest_graph_paths,
        },
    )


async def _stream_svf(safe_input: str, workflow_run_id: str = "") -> AsyncGenerator[str, None]:
    try:
        async for chunk in _stream_svf_impl(safe_input, workflow_run_id):
            yield chunk
    except Exception as exc:
        print(f"[SVF][STREAM] Unhandled stream error: {type(exc).__name__}: {exc}")
        yield _sse_event(
            "error",
            {
                "status": "error",
                "message": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            },
        )
        yield _sse_event("done", {"status": "error"})


# ---- Endpoints ----
@router.post("/analyze", response_model=ComplianceResponse)
async def svf_analyze(req: ComplianceRequest):
    """Blocking SVF compliance analysis endpoint."""
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
    """Streaming SVF compliance analysis endpoint."""
    safe_input = pii_scrubber(req.application_data)
    return StreamingResponse(
        _stream_svf(safe_input),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )



