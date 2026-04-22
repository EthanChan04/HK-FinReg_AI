"""
Semantic cache for retriever results.
"""
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings
from app.services.utils import pii_scrubber


@dataclass
class _CacheEntry:
    vector: List[float]
    scrubbed_query: str
    docs: List[Document]
    created_at: float


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        similarity_threshold: float = 0.80,
        max_entries: int = 200,
        ttl_seconds: int = 3600,
    ):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._lock = Lock()

    def _evict_expired(self) -> None:
        now = time.time()
        expired_keys = [
            key
            for key, item in self._entries.items()
            if now - item.created_at >= self.ttl_seconds
        ]
        for key in expired_keys:
            self._entries.pop(key, None)

    def _prepare_query(self, query: str) -> Tuple[str, List[float]]:
        scrubbed_query = pii_scrubber((query or "").strip())
        query_vector = self.embeddings.embed_query(scrubbed_query)
        return scrubbed_query, query_vector

    def get(self, query: str) -> Tuple[Optional[List[Document]], str, List[float]]:
        scrubbed_query, query_vector = self._prepare_query(query)
        with self._lock:
            self._evict_expired()
            for key, entry in list(self._entries.items()):
                similarity = _cosine_similarity(query_vector, entry.vector)
                if similarity >= self.threshold:
                    self._entries.move_to_end(key, last=True)
                    print(f"[SVF][CACHE] HIT (similarity={similarity:.4f}, cache_size={len(self._entries)})")
                    return entry.docs, scrubbed_query, query_vector
        return None, scrubbed_query, query_vector

    def put(self, scrubbed_query: str, query_vector: List[float], docs: List[Document]) -> None:
        if not scrubbed_query:
            return
        with self._lock:
            self._evict_expired()
            self._entries[scrubbed_query] = _CacheEntry(
                vector=query_vector,
                scrubbed_query=scrubbed_query,
                docs=docs,
                created_at=time.time(),
            )
            self._entries.move_to_end(scrubbed_query, last=True)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


@lru_cache()
def get_semantic_cache() -> Optional[SemanticCache]:
    settings = get_settings()
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None
    if not settings.ZHIPU_API_KEY:
        print("[SVF][CACHE] Disabled: missing embedding API key.")
        return None

    embeddings = OpenAIEmbeddings(
        model=settings.ZHIPU_EMBEDDING_MODEL,
        openai_api_key=settings.ZHIPU_API_KEY,
        openai_api_base=settings.ZHIPU_BASE_URL,
        chunk_size=64,
    )
    return SemanticCache(
        embeddings=embeddings,
        similarity_threshold=settings.SEMANTIC_CACHE_THRESHOLD,
        max_entries=settings.SEMANTIC_CACHE_MAX_ENTRIES,
        ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
    )
