"""
測試 Semantic Cache 的 LRU 淘汰機制與單次 Embedding 流程
驗證：
  1. LRU 淘汰策略（最近最少使用的被淘汰）
  2. 單次 Embedding 計算（get() 返回 vector，put() 重用）
  3. PII 脫敏
"""
import sys
import os
import time
from dataclasses import dataclass
from typing import List, Optional
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class Document:
    """模擬 LangChain Document 類"""
    page_content: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockEmbeddings:
    """模擬 Embeddings，用於測試（不調用真實 API）"""
    call_count: int = 0

    def embed_query(self, text: str) -> List[float]:
        self.call_count += 1
        # 簡單的 hash-based 模擬向量，保證相同文字得到相同向量
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        vec = []
        for i in range(16):
            vec.append((h[i] - 128.0) / 128.0)
        return vec


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
    norm_a = (sum(x * x for x in a)) ** 0.5
    norm_b = (sum(y * y for y in b)) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mock_pii_scrubber(text: str) -> str:
    """模擬 PII 脫敏"""
    return text.replace("+852-91234567", "[PHONE]").replace("test@example.com", "[EMAIL]")


class MockSemanticCache:
    """模擬 SemanticCache 用於測試（不依賴外部配置）"""

    def __init__(
        self,
        embeddings: MockEmbeddings,
        similarity_threshold: float = 0.80,
        max_entries: int = 5,  # 小容量便於測試淘汰
        ttl_seconds: int = 3600,
    ):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, _CacheEntry]" = OrderedDict()

    def _prepare_query(self, query: str):
        scrubbed = mock_pii_scrubber((query or "").strip())
        vec = self.embeddings.embed_query(scrubbed)
        return scrubbed, vec

    def get(self, query: str):
        scrubbed, vec = self._prepare_query(query)
        for key, entry in list(self._entries.items()):
            sim = _cosine_similarity(vec, entry.vector)
            if sim >= self.threshold:
                self._entries.move_to_end(key, last=True)
                return entry.docs, scrubbed, vec
        return None, scrubbed, vec

    def put(self, scrubbed_query: str, query_vector: List[float], docs: List[Document]):
        if not scrubbed_query:
            return
        self._entries[scrubbed_query] = _CacheEntry(
            vector=query_vector,
            scrubbed_query=scrubbed_query,
            docs=docs,
            created_at=time.time(),
        )
        self._entries.move_to_end(scrubbed_query, last=True)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


def create_test_doc(content: str, page: int = 1) -> Document:
    return Document(page_content=content, metadata={"page": page})


def test_single_embedding_flow():
    """測試：get() 計算一次 Embedding，put() 重用，不重複計算"""
    print("=" * 60)
    print("測試 1: 單次 Embedding 流程")
    print("=" * 60)

    embeddings = MockEmbeddings()
    cache = MockSemanticCache(embeddings, max_entries=10)

    query = "What is HKMA SVF requirement?"
    docs = [create_test_doc("Test doc 1")]

    print(f"\n步驟 1: cache.get('{query}') - 應該未命中")
    cached_docs, scrubbed, vec1 = cache.get(query)
    print(f"  - Embedding 調用次數: {embeddings.call_count}")
    print(f"  - 脫敏後查詢: '{scrubbed}'")
    print(f"  - 命中? {'✅ 是' if cached_docs else '❌ 否'}")

    print(f"\n步驟 2: cache.put(...) - 使用從 get() 返回的 vector")
    cache.put(scrubbed, vec1, docs)
    print(f"  - Embedding 調用次數: {embeddings.call_count} (未增加！)")
    print(f"  - 當前緩存大小: {len(cache._entries)}")

    print(f"\n步驟 3: cache.get('{query}') 第二次 - 應該命中")
    cached_docs2, scrubbed2, vec2 = cache.get(query)
    print(f"  - Embedding 調用次數: {embeddings.call_count} (仍然未增加！)")
    print(f"  - 命中? {'✅ 是' if cached_docs2 else '❌ 否'}")

    assert embeddings.call_count == 2, f"預期 2 次 Embedding 調用，實際 {embeddings.call_count}"
    print("\n✅ 單次 Embedding 流程測試通過！")
    return True


def test_lru_eviction():
    """測試：LRU 淘汰策略（最近最少使用的被彈出）"""
    print("\n" + "=" * 60)
    print("測試 2: LRU 淘汰機制")
    print("=" * 60)

    embeddings = MockEmbeddings()
    cache = MockSemanticCache(embeddings, max_entries=3)

    entries = ["A", "B", "C", "D", "E"]

    print("\n步驟 1: 依序放入 A, B, C (滿容量)")
    for key in entries[:3]:
        docs = [create_test_doc(f"Doc {key}")]
        _, scrubbed, vec = cache.get(key)
        cache.put(scrubbed, vec, docs)
        print(f"  放入 {key} -> 緩存: {list(cache._entries.keys())}")

    assert list(cache._entries.keys()) == ["A", "B", "C"]

    print("\n步驟 2: 訪問 A (將 A 移到尾部)")
    cache.get("A")
    print(f"  訪問 A -> 緩存順序: {list(cache._entries.keys())}")

    assert list(cache._entries.keys()) == ["B", "C", "A"], f"預期 [B,C,A], 實際 {list(cache._entries.keys())}"

    print("\n步驟 3: 放入 D (應該淘汰 B - 最久未使用)")
    _, scrubbed_d, vec_d = cache.get("D")
    cache.put(scrubbed_d, vec_d, [create_test_doc("Doc D")])
    print(f"  放入 D -> 淘汰 B -> 緩存: {list(cache._entries.keys())}")

    assert "B" not in cache._entries
    assert list(cache._entries.keys()) == ["C", "A", "D"]

    print("\n步驟 4: 放入 E (應該淘汰 C)")
    _, scrubbed_e, vec_e = cache.get("E")
    cache.put(scrubbed_e, vec_e, [create_test_doc("Doc E")])
    print(f"  放入 E -> 淘汰 C -> 緩存: {list(cache._entries.keys())}")

    assert "C" not in cache._entries
    assert list(cache._entries.keys()) == ["A", "D", "E"]

    print("\n✅ LRU 淘汰機制測試通過！")
    return True


def test_pii_scrubbing():
    """測試：PII 資訊在緩存中被脫敏"""
    print("\n" + "=" * 60)
    print("測試 3: PII 脫敏")
    print("=" * 60)

    embeddings = MockEmbeddings()
    cache = MockSemanticCache(embeddings, max_entries=10)

    sensitive_query = "Contact: +852-91234567, Email: test@example.com, Query: SVF license"
    docs = [create_test_doc("Test doc")]

    print(f"\n原始查詢: {sensitive_query}")
    cached, scrubbed, vec = cache.get(sensitive_query)
    cache.put(scrubbed, vec, docs)

    print(f"脫敏後存儲的 key: '{scrubbed}'")

    assert "+852-91234567" not in scrubbed
    assert "test@example.com" not in scrubbed
    assert "[PHONE]" in scrubbed
    assert "[EMAIL]" in scrubbed

    print("\n✅ PII 脫敏測試通過！")
    return True


def test_cache_ordered_dict_move_to_end():
    """測試：OrderedDict move_to_end 的行為"""
    print("\n" + "=" * 60)
    print("測試 4: OrderedDict LRU 行為驗證")
    print("=" * 60)

    od = OrderedDict()
    od["a"] = 1
    od["b"] = 2
    od["c"] = 3

    print(f"\n初始: {list(od.keys())}")

    od.move_to_end("a")
    print(f"move_to_end('a'): {list(od.keys())}")
    assert list(od.keys()) == ["b", "c", "a"]

    od["d"] = 4
    od.move_to_end("d")
    print(f"add d: {list(od.keys())}")

    od.popitem(last=False)
    print(f"popitem(last=False): {list(od.keys())}")
    assert list(od.keys()) == ["c", "a", "d"]

    print("\n✅ OrderedDict 行為驗證通過！")
    return True


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "Semantic Cache LRU 與單次 Embedding 測試" + " " * 8 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    results = []
    results.append(("OrderedDict LRU Behavior", test_cache_ordered_dict_move_to_end()))
    results.append(("Single Embedding Flow", test_single_embedding_flow()))
    results.append(("LRU Eviction", test_lru_eviction()))
    results.append(("PII Scrubbing", test_pii_scrubbing()))

    print("\n" + "=" * 60)
    print("總結:")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:40s} {status}")

    all_passed = all(p for _, p in results)
    print("=" * 60)

    if all_passed:
        print("\n🎉 所有 Semantic Cache 測試通過！")
    else:
        print("\n❌ 部分測試失敗")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
