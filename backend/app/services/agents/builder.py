"""
LLM 工厂 & 检索引擎模块 (Builder)

职责：
  1. 构建显式 DeepSeek V4 Flash LLM 实例
  2. 构建 Hybrid Retriever (BM25 + ChromaDB Dense, 自定义 RRF 融合)

DeepSeek 客户端由中央工厂按运行配置缓存。
"""
import os
import re
import math
import hashlib
import shutil
import builtins
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

from app.core.config import get_settings
from app.services.llm.deepseek import build_deepseek_llm


def _safe_print(*args, **kwargs) -> None:
    """Print without crashing on non-UTF-8 Windows consoles."""

    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        sanitized = [
            str(arg).encode("ascii", errors="replace").decode("ascii")
            for arg in args
        ]
        builtins.print(*sanitized, **kwargs)


print = _safe_print


# ==========================================
# Embedding client builder
# ==========================================

class LocalHashEmbeddings:
    """Deterministic local embedding fallback with no external dependency."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = max(32, int(dimensions))

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = -1.0 if digest[4] % 2 else 1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _embedding_runtime_config():
    settings = get_settings()
    provider = (getattr(settings, "EMBEDDING_PROVIDER", "") or "local_hash").lower()
    model = getattr(settings, "EMBEDDING_MODEL", "") or "local-hash"
    base_url = getattr(settings, "EMBEDDING_BASE_URL", "")
    api_key = getattr(settings, "EMBEDDING_API_KEY", "")
    dimensions = int(getattr(settings, "EMBEDDING_DIMENSIONS", 256) or 256)
    return provider, model, base_url, api_key, dimensions


def get_embedding_signature() -> str:
    provider, model, base_url, _, dimensions = _embedding_runtime_config()
    return f"{provider}|{model}|{base_url}|{dimensions}"


def build_embeddings_client():
    provider, model, base_url, api_key, dimensions = _embedding_runtime_config()

    if provider == "local_hash":
        print(f"✅ Embeddings initialized with local provider (model=local-hash-{dimensions})")
        return LocalHashEmbeddings(dimensions=dimensions)

    emb = OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        chunk_size=64,
    )
    try:
        emb.embed_query("embedding health check")
        print(f"✅ Embeddings initialized (provider=openai_compatible, model={model})")
        return emb
    except Exception as exc:
        print(
            "⚠️ OpenAI-compatible embeddings probe failed; falling back to local hash embeddings: "
            f"{type(exc).__name__}: {exc}"
        )
        return LocalHashEmbeddings(dimensions=dimensions)


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

def build_zhipu_llm() -> ChatOpenAI:
    """Deprecated compatibility wrapper for the interactive DeepSeek profile."""
    warnings.warn(
        "build_zhipu_llm is deprecated; use build_deepseek_llm",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_deepseek_llm("interactive")


def build_thinking_llm() -> ChatOpenAI:
    """Build the DeepSeek reasoning profile."""
    return build_deepseek_llm("reasoning")


# 缓存 structured LLM 尝试结果（避免每次 reviewer_node 调用都重复尝试）
_structured_llm_cache: dict = {"result": None, "checked": False}


def get_structured_reviewer_llm():
    """P2.5: 尝试构建支持 with_structured_output 的 Reviewer LLM

    验证当前 DeepSeek 模型是否支持 function calling / tool_choice。
    如果支持 → 返回结构化 LLM（直接输出 ReviewerVerdict）
    如果不支持 → 返回 None（调用方 fallback 到正则解析）

    结果会被缓存，避免每次调用都重复尝试。

    Returns:
        结构化 LLM 实例 或 None
    """
    if _structured_llm_cache["checked"]:
        return _structured_llm_cache["result"]

    from app.schemas.requests import ReviewerVerdict
    llm = build_deepseek_llm("interactive")
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
    backend_root = Path(__file__).resolve().parents[3]
    graph_path = Path(settings.GRAPH_STORE_PATH)
    if not graph_path.is_absolute():
        graph_path = backend_root / graph_path
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
    from app.services.corpus.cache import manifest_digest, read_corpus_cache, write_corpus_cache

    settings = get_settings()
    backend_root = Path(__file__).resolve().parents[3]
    cache_dir = Path(settings.CORPUS_INDEX_DIR)
    if not cache_dir.is_absolute():
        cache_dir = backend_root / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "corpus_documents.json"
    manifest_path = backend_root / "data" / "source_manifest.json"
    cached_docs = read_corpus_cache(
        cache_path,
        manifest_digest=manifest_digest(manifest_path),
        parser_version="hierarchy-v1",
    )
    if cached_docs:
        print(f"Regulatory corpus loaded from cache: {len(cached_docs)} chunks")
        _maybe_build_graph()
        return tuple(cached_docs)

    try:
        from app.services.corpus.corpus_ingestor import load_corpus_documents

        corpus_docs = load_corpus_documents()
        if corpus_docs:
            print(f"Regulatory corpus loaded: {len(corpus_docs)} chunks")
            try:
                write_corpus_cache(
                    cache_path,
                    corpus_docs,
                    manifest_digest=manifest_digest(manifest_path),
                    parser_version="hierarchy-v1",
                )
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
    embeddings = build_embeddings_client()

    persist_directory = os.path.join(
        settings.CORPUS_INDEX_DIR,
        f"chroma_{settings.CHROMA_COLLECTION}",
    )
    signature_path = os.path.join(persist_directory, "embedding_signature.txt")
    current_signature = get_embedding_signature()
    existing_signature = None
    if os.path.exists(signature_path):
        try:
            with open(signature_path, "r", encoding="utf-8") as sig_file:
                existing_signature = sig_file.read().strip()
        except Exception:
            existing_signature = None

    sqlite_path = os.path.join(persist_directory, "chroma.sqlite3")
    needs_rebuild = False
    if os.path.exists(sqlite_path):
        if existing_signature is None:
            needs_rebuild = True
            print("♻️ Legacy Chroma index without embedding signature detected. Rebuilding index.")
        elif existing_signature != current_signature:
            needs_rebuild = True
            print("♻️ Embedding config changed. Rebuilding Chroma vector index from scratch.")

    if needs_rebuild:
        abs_index_dir = os.path.abspath(settings.CORPUS_INDEX_DIR)
        abs_persist_dir = os.path.abspath(persist_directory)
        if abs_persist_dir.startswith(abs_index_dir):
            shutil.rmtree(persist_directory, ignore_errors=True)

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
    try:
        with open(signature_path, "w", encoding="utf-8") as sig_file:
            sig_file.write(current_signature)
    except Exception as exc:
        print(f"⚠️ Unable to persist embedding signature: {exc}")
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
    fused_docs: List[Document] = []
    for key in sorted_keys:
        doc = doc_map[key]
        if doc.metadata is None:
            doc.metadata = {}
        # P0: expose deterministic RRF score so downstream UI can show it.
        doc.metadata["rrf_score"] = round(score_map[key], 6)
        fused_docs.append(doc)
    return fused_docs


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        bm25_results: List[Document] = []
        dense_results: List[Document] = []

        try:
            bm25_results = self.bm25_retriever.invoke(query)
        except Exception as exc:
            print(f"  ⚠️ BM25 retrieval failed: {type(exc).__name__}: {exc}")

        try:
            dense_results = self.dense_retriever.invoke(query)
        except Exception as exc:
            print(f"  ⚠️ Dense retrieval failed, fallback to available retriever: {type(exc).__name__}: {exc}")

        print(f"  🔤 BM25 returned {len(bm25_results)} docs")
        print(f"  🧠 Dense returned {len(dense_results)} docs")

        if bm25_results and dense_results:
            fused = reciprocal_rank_fusion(
                result_lists=[bm25_results, dense_results],
                weights=[self.bm25_weight, self.dense_weight]
            )
            print(f"  🔀 RRF fused: {len(fused)} unique docs")
            return fused

        if bm25_results:
            fallback = reciprocal_rank_fusion([bm25_results], [1.0])
            print(f"  🔀 Fallback fused from BM25: {len(fallback)} docs")
            return fallback

        if dense_results:
            fallback = reciprocal_rank_fusion([dense_results], [1.0])
            print(f"  🔀 Fallback fused from Dense: {len(fallback)} docs")
            return fallback

        return []


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
