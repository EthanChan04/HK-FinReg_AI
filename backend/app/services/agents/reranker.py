"""
Cohere Reranker 模块

使用 Cohere Rerank API 对 Hybrid Retriever 返回的候选文档进行二次语义精排。

无需依赖 LangChain 的 ContextualCompressionRetriever（该类在 langchain 1.2.x 中
不可用），而是直接作为独立函数被 builder.py 中的 RerankedRetriever 调用。

架构：
  HybridRetriever (BM25+Dense)
        │ ~20 docs
        ▼
  RerankedRetriever._get_relevant_documents()
        │ 调用 rerank_documents()
        ▼
  Cohere Rerank API
        │ top-5 docs
        ▼
  最终上下文 → Analyzer Agent
"""
import cohere
from functools import lru_cache
from typing import List
from langchain_core.documents import Document

from app.core.config import get_settings


@lru_cache()
def _get_cohere_client() -> cohere.ClientV2:
    """获取 Cohere 客户端单例"""
    settings = get_settings()
    if not settings.COHERE_API_KEY:
        raise ValueError(
            "COHERE_API_KEY not found in .env! "
            "Register at https://dashboard.cohere.com/api-keys"
        )
    return cohere.ClientV2(api_key=settings.COHERE_API_KEY)


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
    model: str = "rerank-v3.5"
) -> List[Document]:
    """
    调用 Cohere Rerank API 对文档做语义精排。

    Args:
        query: 检索查询文本
        documents: 候选 Document 列表 (~20 篇)
        top_k: 保留的精排文档数
        model: Cohere Rerank 模型名

    Returns:
        按 Cohere 相关性分数降序排列的 Top-K Document 列表。
        如果请求失败，fallback 到未排序的前 top_k 篇。
    """
    if not documents:
        return []

    doc_texts = [doc.page_content for doc in documents]

    try:
        client = _get_cohere_client()

        response = client.rerank(
            model=model,
            query=query,
            documents=doc_texts,
            top_n=top_k
        )

        reranked = []
        for result in response.results:
            idx = result.index
            doc = documents[idx]
            doc.metadata["rerank_score"] = round(result.relevance_score, 4)
            reranked.append(doc)

        scores_preview = [round(r.relevance_score, 3) for r in response.results[:3]]
        print(
            f"  🎯 Cohere Reranker: {len(documents)} → {len(reranked)} docs "
            f"(top scores: {scores_preview}...)"
        )
        return reranked

    except Exception as e:
        print(f"  ⚠️ Cohere Rerank failed: {e}, falling back to unranked top-{top_k}")
        return list(documents[:top_k])
