"""
LLM 工厂 & 检索引擎模块 (Builder)

职责：
  1. 构建 LLM 实例 (Zhipu GLM / LongCat Thinking)
  2. 构建 Hybrid Retriever (BM25 + ChromaDB Dense, 自定义 RRF 融合)

所有构建函数均使用 @lru_cache 做单例缓存。
"""
import os
import re
from functools import lru_cache
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.core.config import get_settings


# ==========================================
# Dynamic query profile (Phase 2)
# ==========================================

DYNAMIC_WEIGHT_PROFILES = {
    "specific_clause": {"bm25": 0.7, "dense": 0.3},
    "risk_assessment": {"bm25": 0.3, "dense": 0.7},
    "entity_lookup": {"bm25": 0.8, "dense": 0.2},
    "default": {"bm25": 0.4, "dense": 0.6},
}


def classify_query_type(extracted_entities: str) -> str:
    """Classify query intent for dynamic BM25/Dense weighting."""
    text = (extracted_entities or "").lower()

    if re.search(r"\b(chapter|paragraph|section|clause)\s*[:\s]\s*\d+(?:\.\d+)*\b", text):
        return "specific_clause"
    if re.search(r"\b(license\s*(?:no|number)\.?\s*[:#-]?\s*\w+|svf-?\d+|registration\s*(?:no|number)?)\b", text):
        return "entity_lookup"
    if re.search(r"\b(risk|assessment|evaluation|exposure)\b", text):
        return "risk_assessment"
    return "default"


# ==========================================
# LLM 实例构建
# ==========================================

@lru_cache()
def build_zhipu_llm() -> ChatOpenAI:
    """构建 Zhipu GLM-4.7-Flash LLM 实例"""
    settings = get_settings()
    return ChatOpenAI(
        model_name=settings.ZHIPU_MODEL,
        temperature=0,
        openai_api_key=settings.ZHIPU_API_KEY,
        openai_api_base=settings.ZHIPU_BASE_URL,
        timeout=settings.LLM_TIMEOUT_SECONDS
    )


@lru_cache()
def build_thinking_llm() -> ChatOpenAI:
    """构建 LongCat-Flash-Thinking 深度推理实例"""
    settings = get_settings()
    return ChatOpenAI(
        model_name=settings.LONGCAT_MODEL,
        temperature=0,
        openai_api_key=settings.LONGCAT_API_KEY,
        openai_api_base=settings.LONGCAT_BASE_URL,
        timeout=settings.LLM_TIMEOUT_SECONDS
    )


# 缓存 structured LLM 尝试结果（避免每次 reviewer_node 调用都重复尝试）
_structured_llm_cache: dict = {"result": None, "checked": False}


def get_structured_reviewer_llm():
    """P2.5: 尝试构建支持 with_structured_output 的 Reviewer LLM

    验证 GLM-4-Flash 是否支持 function calling / tool_choice。
    如果支持 → 返回结构化 LLM（直接输出 ReviewerVerdict）
    如果不支持 → 返回 None（调用方 fallback 到正则解析）

    结果会被缓存，避免每次调用都重复尝试。

    Returns:
        结构化 LLM 实例 或 None
    """
    if _structured_llm_cache["checked"]:
        return _structured_llm_cache["result"]

    from app.schemas.requests import ReviewerVerdict
    llm = build_zhipu_llm()
    try:
        structured = llm.with_structured_output(ReviewerVerdict)
        # 简单验证：检查返回对象是否有 invoke 方法
        if hasattr(structured, 'invoke'):
            _structured_llm_cache["checked"] = True
            _structured_llm_cache["result"] = structured
            return structured
    except (NotImplementedError, TypeError, Exception) as e:
        print(f"[SVF][BUILDER] with_structured_output not supported: {type(e).__name__}: {e}")

    _structured_llm_cache["checked"] = True
    _structured_llm_cache["result"] = None
    return None


# ==========================================
# PDF 加载 & 切片 (共享)
# ==========================================

# ==========================================
# P3.1: 法规感知正则切分 (V1)
# ==========================================

# 法规文档常见章节标题模式
REG_SECTION_PATTERNS = [
    r'^#{1,3}\s+(Chapter|Section|Paragraph|Part|Schedule)\s+[\d.]+',
    r'^#{1,3}\s+\d+\.\d*(?:\s+[A-Z])?',
    r'^#{1,3}\s+Appendix\s+[A-Z]',
]


def reg_aware_split(text: str, metadata: dict) -> list:
    """V1: 基于正则的法规感知切分

    尝试按法规章节标题切分；如果标题太少（< 3 个），
    fallback 到 CharacterTextSplitter。

    Args:
        text: 单页 PDF 的文本内容
        metadata: 该页的元数据（page number 等）

    Returns:
        切分后的 Document 列表
    """
    lines = text.split('\n')
    sections = []
    current_lines = []
    current_title = "Preamble"

    for line in lines:
        is_section_header = any(
            re.match(pat, line.strip()) for pat in REG_SECTION_PATTERNS
        )

        if is_section_header and current_lines:
            sections.append((current_title, '\n'.join(current_lines)))
            current_title = line.strip().lstrip('#').strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, '\n'.join(current_lines)))

    # Fallback: 如果切出的段太少，说明不是结构化法规文档
    if len(sections) < 3:
        from langchain_text_splitters import CharacterTextSplitter
        settings = get_settings()
        splitter = CharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(text)
        return [Document(page_content=c, metadata=metadata) for c in chunks]

    # 正常切分
    docs = []
    for title, content in sections:
        if len(content.strip()) < 50:
            continue
        doc_meta = {**metadata, "section_title": title}
        docs.append(Document(page_content=content, metadata=doc_meta))

    return docs


@lru_cache()
def _load_and_split_pdf() -> tuple:
    """加载 PDF 并切片，返回 Document 元组 (lru_cache 需要 hashable)

    根据 PARSER_MODE 配置选择解析模式：
      - "hierarchy": 使用 document_parser 的层级感知解析器（M4 完整版）
      - "reg_aware": 使用 reg_aware_split() V1 版本
      - "flat": 使用纯 CharacterTextSplitter
    """
    settings = get_settings()
    pdf_path = settings.PDF_PATH
    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF not found: {pdf_path}")
        return ()

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"📄 PDF loaded: {len(documents)} pages")

    parser_mode = getattr(settings, 'PARSER_MODE', 'reg_aware')

    if parser_mode == "hierarchy":
        # M4: 完整层级解析器
        from app.services.agents.document_parser import (
            parse_pdf_with_hierarchy,
            regulation_chunks_to_documents,
        )
        source_name = os.path.basename(pdf_path)
        chunks = parse_pdf_with_hierarchy(documents, source_name=source_name)
        all_docs = regulation_chunks_to_documents(chunks)
        # 统计层级信息
        level_counts = {}
        for chunk in chunks:
            level_counts[chunk.hierarchy_level] = level_counts.get(chunk.hierarchy_level, 0) + 1
        print(
            f"✂️ [hierarchy] Chunked into {len(all_docs)} segments "
            f"(levels: {dict(sorted(level_counts.items()))})"
        )
        return tuple(all_docs)

    elif parser_mode == "reg_aware":
        # V1: 正则感知切分
        all_docs = []
        reg_split_count = 0
        fallback_count = 0
        for doc in documents:
            split_docs = reg_aware_split(doc.page_content, doc.metadata)
            if split_docs and hasattr(split_docs[0].metadata, 'get') and split_docs[0].metadata.get('section_title'):
                reg_split_count += 1
            else:
                fallback_count += 1
            all_docs.extend(split_docs)
        print(f"✂️ [reg_aware] Chunked into {len(all_docs)} segments (reg-aware: {reg_split_count} pages, fallback: {fallback_count} pages)")
        return tuple(all_docs)

    else:
        # "flat": 纯 CharacterTextSplitter
        text_splitter = CharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        splits = text_splitter.split_documents(documents)
        print(f"✂️ [flat] Chunked into {len(splits)} segments")
        return tuple(splits)


def _maybe_build_graph() -> None:
    """Build and persist the regulatory graph if it doesn't already exist.

    Uses the source manifest documents and an empty evidence list to create
    a deterministic metadata-derived graph (regulators, topics, products).
    Failures are non-fatal and logged as warnings.
    """
    import os

    from app.core.config import get_settings

    settings = get_settings()
    graph_path = settings.GRAPH_STORE_PATH
    if os.path.exists(graph_path):
        return

    try:
        from app.services.kag.graph_builder import build_graph_from_sources
        from app.services.corpus.manifest_loader import load_source_manifest

        source_docs = load_source_manifest()
        if not source_docs:
            print("Regulatory graph: no source documents, skipping")
            return

        store = build_graph_from_sources(
            documents=source_docs,
            evidence_chunks=[],
            graph_path=graph_path,
        )
        n_nodes = store.graph.number_of_nodes()
        n_edges = store.graph.number_of_edges()
        print(f"Regulatory graph built: {n_nodes} nodes, {n_edges} edges")
    except Exception as exc:
        print(f"Regulatory graph build skipped (non-fatal): {exc}")


@lru_cache()
def _load_and_split_corpus() -> tuple:
    """Load manifest-backed corpus chunks, falling back to the legacy PDF path."""
    import pickle

    settings = get_settings()
    os.makedirs(settings.CORPUS_INDEX_DIR, exist_ok=True)
    cache_path = os.path.join(settings.CORPUS_INDEX_DIR, "corpus_documents.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as cache_file:
                cached_docs = pickle.load(cache_file)
            if cached_docs:
                print(f"Regulatory corpus loaded from cache: {len(cached_docs)} chunks")
                _maybe_build_graph()
                return tuple(cached_docs)
        except Exception as exc:
            print(f"Regulatory corpus cache unreadable, rebuilding: {exc}")

    try:
        from app.services.corpus.corpus_ingestor import load_corpus_documents

        corpus_docs = load_corpus_documents()
        if corpus_docs:
            print(f"Regulatory corpus loaded: {len(corpus_docs)} chunks")
            try:
                with open(cache_path, "wb") as cache_file:
                    pickle.dump(corpus_docs, cache_file)
                print(f"Regulatory corpus cached to {cache_path}")
            except Exception as exc:
                print(f"Regulatory corpus cache write failed: {exc}")
            _maybe_build_graph()
            return tuple(corpus_docs)
        print("Regulatory corpus empty, falling back to legacy PDF_PATH")
    except Exception as exc:
        print(f"Regulatory corpus ingestion failed, falling back to legacy PDF_PATH: {exc}")

    return _load_and_split_pdf()


@lru_cache()
def _build_chroma_db():
    """Build or load the persisted ChromaDB vector store."""
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.ZHIPU_EMBEDDING_MODEL,
        openai_api_key=settings.ZHIPU_API_KEY,
        openai_api_base=settings.ZHIPU_BASE_URL,
        chunk_size=64,
    )

    persist_directory = os.path.join(
        settings.CORPUS_INDEX_DIR,
        f"chroma_{settings.CHROMA_COLLECTION}",
    )
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        db = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        print(f"ChromaDB vector store loaded from {persist_directory}")
        return db

    splits = list(_load_and_split_corpus())
    if not splits:
        return None

    os.makedirs(persist_directory, exist_ok=True)
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=settings.CHROMA_COLLECTION,
        persist_directory=persist_directory,
    )
    print(f"ChromaDB vector store initialized and persisted to {persist_directory}")
    return db


# ==========================================
# Reciprocal Rank Fusion (RRF) 算法
# ==========================================

def _compute_content_hash(content: str) -> str:
    """计算文档内容哈希用于去重（P3.4 RRF 修复）"""
    import hashlib
    normalized = re.sub(r'\s+', ' ', content.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    weights: List[float],
    k: int = 60
) -> List[Document]:
    """
    RRF 分数 = Σ weight_i / (k + rank_i)
    其中 rank_i 是该 Document 在第 i 路检索结果中的排名（从 1 开始）。

    P3.4 修复: 去重键用 content_hash 替代 page_content[:200]，
    避免不同文档前缀相同时被错误合并。
    """
    score_map: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results, start=1):
            doc_key = _compute_content_hash(doc.page_content)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
                score_map[doc_key] = 0.0
            score_map[doc_key] += weight / (k + rank)

    sorted_keys = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


# ==========================================
# Hybrid Retriever (BM25 + Dense, RRF 融合)
# ==========================================

class HybridRetriever(BaseRetriever):
    """
    自定义 Hybrid Retriever，继承 LangChain BaseRetriever，
    使得 LangSmith 能自动追踪 .invoke() 调用。
    """
    bm25_retriever: BM25Retriever
    dense_retriever: BaseRetriever
    bm25_weight: float = 0.4
    dense_weight: float = 0.6

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        bm25_results = self.bm25_retriever.invoke(query)
        dense_results = self.dense_retriever.invoke(query)

        print(f"  🔤 BM25 returned {len(bm25_results)} docs")
        print(f"  🧠 Dense returned {len(dense_results)} docs")

        fused = reciprocal_rank_fusion(
            result_lists=[bm25_results, dense_results],
            weights=[self.bm25_weight, self.dense_weight]
        )

        print(f"  🔀 RRF fused: {len(fused)} unique docs")
        return fused


@lru_cache()
def build_hybrid_retriever() -> HybridRetriever | None:
    """
    构建混合检索器：
      - Sparse: BM25 (关键词精确匹配)
      - Dense:  ChromaDB (语义向量匹配)
      - Fusion: RRF，权重 BM25=0.4 / Dense=0.6
    """
    splits = list(_load_and_split_corpus())
    if not splits:
        print("⚠️ No documents to build retriever")
        return None

    chroma_db = _build_chroma_db()
    if chroma_db is None:
        return None

    dense_retriever = chroma_db.as_retriever(search_kwargs={"k": 15})
    bm25_retriever = BM25Retriever.from_documents(splits, k=15)

    hybrid = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        bm25_weight=0.4,
        dense_weight=0.6
    )

    print("✅ Hybrid Retriever (BM25 + Dense) initialized — RRF weights [0.4, 0.6]")
    return hybrid


@lru_cache()
def build_reranked_retriever():
    """
    构建 Hybrid + Cohere Reranker 全链路检索器。

    架构：
      RerankedRetriever._get_relevant_documents()
           │
           ├── HybridRetriever.invoke()  → ~20 docs
           │
           └── rerank_documents()        → Top-5 docs
           │
      最终上下文 → Analyzer Agent

    RerankedRetriever 继承 BaseRetriever，
    LangSmith 会自动生成 Trace 节点。

    如果 COHERE_API_KEY 未配置，则 fallback 到纯 HybridRetriever。
    """
    settings = get_settings()

    hybrid = build_hybrid_retriever()
    if hybrid is None:
        return None

    # 如果没有 Cohere Key，回退到纯 Hybrid
    if not settings.COHERE_API_KEY:
        print("⚠️ COHERE_API_KEY not set, using Hybrid Retriever without reranking")
        return hybrid

    reranked = RerankedRetriever(
        hybrid_retriever=hybrid,
        rerank_model=settings.COHERE_RERANK_MODEL,
        top_k=settings.RERANK_TOP_K
    )

    print(f"✅ Reranked Retriever initialized (Hybrid → Cohere {settings.COHERE_RERANK_MODEL} → Top-{settings.RERANK_TOP_K})")
    return reranked


class RerankedRetriever(BaseRetriever):
    """
    自定义 Reranked Retriever，继承 BaseRetriever。
    内部串联 HybridRetriever + Cohere Reranker。
    LangSmith 会自动追踪 .invoke() 调用。
    """
    hybrid_retriever: HybridRetriever
    rerank_model: str = "rerank-v3.5"
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        from app.services.agents.reranker import rerank_documents

        # Step 1: Hybrid 粗捞 (~20 docs)
        candidates = self.hybrid_retriever.invoke(query)
        print(f"  📥 Hybrid returned {len(candidates)} candidates for reranking")

        # Step 2: Cohere 精排 → Top-K
        top_docs = rerank_documents(
            query=query,
            documents=candidates,
            top_k=self.top_k,
            model=self.rerank_model
        )

        return top_docs


def build_profiled_retriever(base_retriever: BaseRetriever, query_type: str) -> BaseRetriever:
    """
    Build a profiled retriever for one request without mutating cached singleton retrievers.
    """
    profile = DYNAMIC_WEIGHT_PROFILES.get(query_type, DYNAMIC_WEIGHT_PROFILES["default"])

    if isinstance(base_retriever, RerankedRetriever):
        base_hybrid = base_retriever.hybrid_retriever
        profiled_hybrid = HybridRetriever(
            bm25_retriever=base_hybrid.bm25_retriever,
            dense_retriever=base_hybrid.dense_retriever,
            bm25_weight=profile["bm25"],
            dense_weight=profile["dense"]
        )
        print(f"[SVF][RRF] {query_type} -> BM25={profile['bm25']}, Dense={profile['dense']}")
        return RerankedRetriever(
            hybrid_retriever=profiled_hybrid,
            rerank_model=base_retriever.rerank_model,
            top_k=base_retriever.top_k
        )

    if isinstance(base_retriever, HybridRetriever):
        print(f"[SVF][RRF] {query_type} -> BM25={profile['bm25']}, Dense={profile['dense']}")
        return HybridRetriever(
            bm25_retriever=base_retriever.bm25_retriever,
            dense_retriever=base_retriever.dense_retriever,
            bm25_weight=profile["bm25"],
            dense_weight=profile["dense"]
        )

    return base_retriever
