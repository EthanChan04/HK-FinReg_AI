"""
LLM 工厂 & 检索引擎模块 (Builder)

职责：
  1. 构建 LLM 实例 (Zhipu GLM / LongCat Thinking)
  2. 构建 Hybrid Retriever (BM25 + ChromaDB Dense, 自定义 RRF 融合)

所有构建函数均使用 @lru_cache 做单例缓存。
"""
import os
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
        openai_api_base=settings.ZHIPU_BASE_URL
    )


@lru_cache()
def build_thinking_llm() -> ChatOpenAI:
    """构建 LongCat-Flash-Thinking 深度推理实例"""
    settings = get_settings()
    return ChatOpenAI(
        model_name=settings.LONGCAT_MODEL,
        temperature=0,
        openai_api_key=settings.LONGCAT_API_KEY,
        openai_api_base=settings.LONGCAT_BASE_URL
    )


# ==========================================
# PDF 加载 & 切片 (共享)
# ==========================================

@lru_cache()
def _load_and_split_pdf() -> tuple:
    """加载 PDF 并切片，返回 Document 元组 (lru_cache 需要 hashable)"""
    settings = get_settings()
    pdf_path = settings.PDF_PATH
    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF not found: {pdf_path}")
        return ()

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"📄 PDF loaded: {len(documents)} pages")

    text_splitter = CharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    print(f"✂️ Chunked into {len(splits)} segments")
    return tuple(splits)


@lru_cache()
def _build_chroma_db():
    """构建 ChromaDB 向量库"""
    settings = get_settings()
    splits = list(_load_and_split_pdf())
    if not splits:
        return None

    embeddings = OpenAIEmbeddings(
        model=settings.ZHIPU_EMBEDDING_MODEL,
        openai_api_key=settings.ZHIPU_API_KEY,
        openai_api_base=settings.ZHIPU_BASE_URL,
        chunk_size=64
    )

    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=settings.CHROMA_COLLECTION
    )
    print("✅ ChromaDB vector store initialized")
    return db


# ==========================================
# Reciprocal Rank Fusion (RRF) 算法
# ==========================================

def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    weights: List[float],
    k: int = 60
) -> List[Document]:
    """
    RRF 分数 = Σ weight_i / (k + rank_i)
    其中 rank_i 是该 Document 在第 i 路检索结果中的排名（从 1 开始）。
    """
    score_map: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results, start=1):
            doc_key = doc.page_content[:200]
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
    splits = list(_load_and_split_pdf())
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
