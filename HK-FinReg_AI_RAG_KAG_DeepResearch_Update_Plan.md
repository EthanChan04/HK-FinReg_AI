# HK-FinReg_AI 项目更新计划书：RAG + KAG + DeepResearch 检索升级

> 建议保存为：`docs/HK-FinReg_AI_RAG_KAG_DeepResearch_Update_Plan.md`  
> 目标：让 Agent IDE 按阶段稳定执行代码更新，将当前项目从“单 PDF + SVF RAG”升级为“多监管文档语料库 + Advanced RAG + KAG 图谱推理 + DeepResearch 多步研究工作流”。

---

## 0. 项目更新总目标

当前项目 `HK-FinReg_AI` 已经具备以下基础能力：

- FastAPI 后端；
- Next.js 前端；
- LangGraph 多智能体工作流；
- SVF 模块接入 Hybrid RAG；
- BM25 + ChromaDB Dense Retrieval；
- RRF 融合；
- Cohere Reranker；
- Reviewer 反思循环；
- 三维置信度；
- 层级法规文档解析器；
- LangSmith 可观测性。

本次升级不应推倒重写，而应在现有基础上完成以下目标：

```text
从：
单一 PDF + SVF 专用 RAG

升级为：
多文档监管语料库 + Metadata-aware RAG + KAG Regulatory Graph + DeepResearch Workflow
```

最终系统应支持三类查询模式：

| 查询类型 | 示例 | 检索路径 |
|---|---|---|
| 普通法规问答 | “SVF licensee 的 CDD 要求是什么？” | Advanced RAG |
| 关系型法规推理 | “虚拟银行推出 AI 投顾涉及哪些监管机构和义务？” | KAG + RAG |
| 多步骤研究报告 | “分析香港虚拟银行推出 AI 投资顾问的合规风险，并生成上线前 checklist。” | DeepResearch + KAG + RAG |

---

## 1. 技术依据与设计原则

### 1.1 LangGraph 工作流设计依据

本项目后端已经采用 LangGraph。LangGraph 的核心建模方式是将 agent workflow 表示为由 `State`、`Nodes` 和 `Edges` 组成的图结构：`State` 表示共享状态，`Nodes` 执行业务逻辑，`Edges` 决定下一步流向。该模型非常适合本项目的 Retriever、Analyzer、Reviewer、Citation Verifier、DeepResearch Planner 等多节点工作流。

LangGraph 支持 conditional edges，因此可以继续沿用当前 Reviewer 之后的条件路由逻辑，并扩展为：

```text
Reviewer
├── approved → END
├── insufficient_info → DeepResearch Gap Retrieval
├── citation_failed → Citation Repair
└── quality_issue → Analyzer Revision
```

### 1.2 Chroma metadata filter 设计依据

本次升级需要让 RAG 根据 `regulator`、`module_tags`、`topics`、`doc_type`、`issue_date` 等 metadata 过滤监管文档。Chroma 支持通过 `where` 做 metadata filter，也支持通过 `where_document` 做文档内容过滤，因此可以在现有 Chroma 向量库基础上实现 metadata-aware retrieval。

### 1.3 FastAPI SSE 保留原则

当前项目已有 SSE 流式输出。FastAPI 支持使用 `text/event-stream` 格式向前端推送 Server-Sent Events，适合 AI chat streaming、日志和可观测性场景。升级过程中应继续保留 SSE，而不是改成普通阻塞式接口。

### 1.4 KAG 图谱存储选择

第一阶段建议使用 NetworkX，而不是直接引入 Neo4j。NetworkX 支持将图以 node-link 格式序列化为 JSON，便于本地开发、GitHub 展示和前端可视化；后续如需生产化，再迁移到 Neo4j。

---

## 2. 本次升级范围

### 2.1 必须完成

本计划要求 Agent IDE 完成以下内容：

1. 建立多文档监管语料库结构；
2. 新增 `source_manifest.json`；
3. 改造配置项，从单 `PDF_PATH` 升级为多文档目录；
4. 新增 `EvidenceChunk` 统一证据模型；
5. 改造 Retriever，使其返回结构化 evidence；
6. 实现 metadata-aware retrieval；
7. 新增 citation semantic verifier；
8. 新增 KAG 最小可用图谱；
9. 新增 retrieval router；
10. 新增 DeepResearch 多步研究 workflow；
11. 更新 SVF 模块，使其可以调用新检索管线；
12. 新增评估 benchmark；
13. 更新 README / docs；
14. 保证现有 SVF 接口不破坏。

### 2.2 暂不完成

以下内容不在本次第一阶段升级范围内：

1. 不接入 Neo4j；
2. 不接入 OpenSPG 官方 KAG 框架；
3. 不重写整个前端；
4. 不移除现有 LangGraph SVF workflow；
5. 不删除原有 RAG 逻辑；
6. 不引入复杂微服务架构；
7. 不做生产级权限系统；
8. 不实现真实银行内部数据接入；
9. 不把所有业务模块一次性全部 RAG 化；
10. 不追求 fully automated legal advice。

---

## 3. Agent IDE 执行总原则

### 3.1 每个阶段必须遵守

Agent IDE 在执行每个阶段时必须遵守：

```text
1. 先阅读相关文件，再修改代码。
2. 每个 Phase 单独提交一次 commit。
3. 不要在同一轮中同时重构后端、前端和文档。
4. 每次修改后运行最小可用测试。
5. 保留现有接口兼容性。
6. 不要删除旧逻辑，除非新逻辑已经完全替代且测试通过。
7. 所有新增模块必须有清晰 docstring。
8. 所有新增配置必须有默认值，避免 .env 缺失导致启动失败。
9. 所有 LLM 调用必须可 fallback。
10. 所有 DeepResearch 循环必须有最大轮次限制，防止死循环。
```

### 3.2 推荐分支

```bash
git checkout main
git pull
git checkout -b feature/rag-kag-deepresearch-upgrade
```

### 3.3 推荐提交节奏

```text
commit 1: add corpus manifest and config
commit 2: add evidence schema and metadata-aware ingestion
commit 3: refactor retrieval pipeline
commit 4: add citation verifier
commit 5: add KAG graph prototype
commit 6: add retrieval router
commit 7: add DeepResearch workflow
commit 8: integrate SVF with new retrieval pipeline
commit 9: add evaluation benchmark
commit 10: update docs and README
```

---

## 4. 目标目录结构

升级后推荐目录如下：

```text
HK-FinReg_AI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routers/
│   │   │       ├── svf.py
│   │   │       └── research.py                  # 新增：DeepResearch API，可选
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── monitoring.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── requests.py
│   │   │   ├── evidence.py                      # 新增
│   │   │   ├── corpus.py                        # 新增
│   │   │   ├── kag.py                           # 新增
│   │   │   └── deepresearch.py                  # 新增
│   │   │
│   │   ├── services/
│   │   │   ├── agents/
│   │   │   │   ├── builder.py
│   │   │   │   ├── document_parser.py
│   │   │   │   └── prompts.py
│   │   │   │
│   │   │   ├── corpus/                          # 新增
│   │   │   │   ├── manifest_loader.py
│   │   │   │   ├── corpus_ingestor.py
│   │   │   │   ├── metadata_normalizer.py
│   │   │   │   └── document_registry.py
│   │   │   │
│   │   │   ├── retrieval/                       # 新增
│   │   │   │   ├── query_classifier.py
│   │   │   │   ├── hybrid_retriever.py
│   │   │   │   ├── rerank_retriever.py
│   │   │   │   ├── evidence_renderer.py
│   │   │   │   ├── citation_verifier.py
│   │   │   │   ├── retrieval_router.py
│   │   │   │   └── retrieval_service.py
│   │   │   │
│   │   │   ├── kag/                             # 新增
│   │   │   │   ├── ontology.py
│   │   │   │   ├── entity_extractor.py
│   │   │   │   ├── relation_extractor.py
│   │   │   │   ├── graph_store.py
│   │   │   │   ├── graph_builder.py
│   │   │   │   ├── graph_retriever.py
│   │   │   │   └── graph_reasoner.py
│   │   │   │
│   │   │   ├── deepresearch/                    # 新增
│   │   │   │   ├── planner.py
│   │   │   │   ├── query_decomposer.py
│   │   │   │   ├── evidence_evaluator.py
│   │   │   │   ├── gap_detector.py
│   │   │   │   ├── report_writer.py
│   │   │   │   ├── workflow.py
│   │   │   │   └── prompts.py
│   │   │   │
│   │   │   └── evaluation/                      # 新增
│   │   │       ├── benchmark_loader.py
│   │   │       ├── rag_eval.py
│   │   │       ├── citation_eval.py
│   │   │       ├── graph_eval.py
│   │   │       └── run_eval.py
│   │   │
│   │   └── tests/
│   │       ├── test_manifest_loader.py
│   │       ├── test_evidence_schema.py
│   │       ├── test_retrieval_router.py
│   │       ├── test_kag_graph_store.py
│   │       └── test_citation_verifier.py
│   │
│   ├── data/
│   │   ├── regulations/
│   │   │   ├── hkma_svf/
│   │   │   ├── hkma_aml_ai/
│   │   │   ├── sfc_aml_vasp/
│   │   │   ├── ai_governance_privacy/
│   │   │   └── stablecoin_optional/
│   │   │
│   │   ├── source_manifest.json                # 新增/已下载后补全
│   │   ├── indexes/
│   │   ├── graph/
│   │   │   └── regulatory_graph.json
│   │   └── evaluation/
│   │       └── benchmark_questions.json
│   │
│   └── requirements.txt
│
├── frontend/
│   └── ...
│
└── docs/
    ├── regulatory_corpus_plan.md
    ├── rag_kag_deepresearch_update_plan.md
    └── evaluation_protocol.md
```

---

## 5. Phase 1：多文档监管语料库与 Manifest

### 5.1 目标

将项目从单 `PDF_PATH` 升级为多文档监管语料库。

当前问题：

```text
backend/app/core/config.py 中只有 PDF_PATH。
这会导致系统只能索引一份 PDF，无法支持多文档、多模块、多监管来源检索。
```

目标：

```text
支持从 backend/data/source_manifest.json 加载多个监管文档。
每份文档必须带 metadata。
每个 chunk 继承文档 metadata。
```

### 5.2 修改 `backend/app/core/config.py`

新增配置项：

```python
# --- Corpus Config ---
REG_DOC_DIR: str = "data/regulations"
SOURCE_MANIFEST_PATH: str = "data/source_manifest.json"
CORPUS_INDEX_DIR: str = "data/indexes"

# --- Graph / KAG Config ---
GRAPH_STORE_BACKEND: str = "networkx"
GRAPH_STORE_PATH: str = "data/graph/regulatory_graph.json"

# --- Retrieval Router Config ---
RETRIEVAL_ROUTER_ENABLED: bool = True
DEFAULT_RETRIEVAL_MODE: str = "rag"  # rag | kag | deep_research

# --- DeepResearch Config ---
DEEP_RESEARCH_ENABLED: bool = True
DEEP_RESEARCH_MAX_ITERATIONS: int = 3
DEEP_RESEARCH_MAX_SUBQUESTIONS: int = 8
DEEP_RESEARCH_MIN_EVIDENCE_PER_SUBQUESTION: int = 2
```

保留原有：

```python
PDF_PATH: str = "../Fintech/AML Guideline for LCs_Eng_30 Sep 2021.pdf"
```

但标记为 legacy fallback：

```python
# Legacy fallback. Used only when SOURCE_MANIFEST_PATH is missing or empty.
PDF_PATH: str = "../Fintech/AML Guideline for LCs_Eng_30 Sep 2021.pdf"
```

### 5.3 新增 `backend/app/schemas/corpus.py`

```python
from pydantic import BaseModel, Field
from typing import Literal


class SourceDocument(BaseModel):
    doc_id: str = Field(..., description="Stable unique document id")
    title: str
    regulator: str
    jurisdiction: str = "Hong Kong"
    doc_type: str
    issue_date: str | None = None
    effective_date: str | None = None
    status: Literal["active", "superseded", "archived", "unknown"] = "active"
    sector: list[str] = []
    topics: list[str] = []
    module_tags: list[str] = []
    file_path: str
    source_url: str | None = None
    priority: Literal["P0", "P1", "P2", "P3"] = "P1"
    language: str = "en"
    notes: str | None = None
```

### 5.4 新增 `backend/app/services/corpus/manifest_loader.py`

功能要求：

```text
1. 读取 source_manifest.json。
2. 校验每个文档符合 SourceDocument schema。
3. 将相对路径解析为绝对路径。
4. 检查文件是否存在。
5. 对缺失文件给 warning，但不中断系统启动。
6. 如果 manifest 不存在，fallback 到 legacy PDF_PATH。
```

伪代码：

```python
import json
from pathlib import Path
from app.core.config import get_settings
from app.schemas.corpus import SourceDocument


def load_source_manifest() -> list[SourceDocument]:
    settings = get_settings()
    manifest_path = Path(settings.SOURCE_MANIFEST_PATH)

    if not manifest_path.exists():
        return []

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = [SourceDocument(**item) for item in raw]

    existing_docs = []
    for doc in docs:
        path = Path(settings.REG_DOC_DIR) / doc.file_path
        if path.exists():
            existing_docs.append(doc)
        else:
            print(f"⚠️ Source file not found: {path}")

    return existing_docs
```

### 5.5 创建 `backend/data/source_manifest.json`

Agent IDE 需要根据用户已下载文档所在目录创建初始 manifest。

示例结构：

```json
[
  {
    "doc_id": "hkma_svf_supervision_guideline_2016",
    "title": "Guideline on Supervision of Stored Value Facility Licensees",
    "regulator": "HKMA",
    "jurisdiction": "Hong Kong",
    "doc_type": "Guideline",
    "issue_date": "2016-09",
    "effective_date": null,
    "status": "active",
    "sector": ["SVF", "Payment"],
    "topics": ["supervision", "governance", "risk_management", "licensing"],
    "module_tags": ["svf", "licensing", "supervision"],
    "file_path": "hkma_svf/guideline_supervision_svf_2016.pdf",
    "source_url": "",
    "priority": "P0",
    "language": "en"
  },
  {
    "doc_id": "hkma_svf_practice_note_2025",
    "title": "Practice Note on Supervision of Stored Value Facility Licensees",
    "regulator": "HKMA",
    "jurisdiction": "Hong Kong",
    "doc_type": "Practice Note",
    "issue_date": "2025-10",
    "effective_date": null,
    "status": "active",
    "sector": ["SVF", "Payment"],
    "topics": ["supervision", "governance", "operational_risk", "risk_management"],
    "module_tags": ["svf", "supervision"],
    "file_path": "hkma_svf/practice_note_supervision_svf_2025.pdf",
    "source_url": "",
    "priority": "P0",
    "language": "en"
  },
  {
    "doc_id": "hkma_svf_amlcft_guideline_2023",
    "title": "Guideline on Anti-Money Laundering and Counter-Financing of Terrorism for Stored Value Facility Licensees",
    "regulator": "HKMA",
    "jurisdiction": "Hong Kong",
    "doc_type": "Guideline",
    "issue_date": "2023-05",
    "effective_date": "2023-06-01",
    "status": "active",
    "sector": ["SVF", "Payment"],
    "topics": ["AML", "CFT", "CDD", "KYC", "transaction_monitoring", "suspicious_activity"],
    "module_tags": ["svf", "aml", "kyc"],
    "file_path": "hkma_svf/amlcft_guideline_svf_2023.pdf",
    "source_url": "",
    "priority": "P0",
    "language": "en"
  }
]
```

Agent IDE 需要根据实际文件名调整 `file_path`，确保路径存在。

### 5.6 Phase 1 验收标准

必须满足：

```bash
cd backend
python -m pytest tests/test_manifest_loader.py -v
```

新增测试应覆盖：

1. manifest 文件存在时能正确加载；
2. manifest 文件不存在时不崩溃；
3. 文件缺失时 warning 而非 raise；
4. `SourceDocument` schema 校验有效；
5. `module_tags`、`topics`、`priority` 字段可读取。

---

## 6. Phase 2：EvidenceChunk 结构化证据模型

### 6.1 目标

当前 Retriever 将文档拼接为字符串：

```text
[Source 1, p.X, relevance=...]
...
```

这对 prompt 有用，但不利于：

- citation verification；
- DeepResearch evidence coverage；
- 前端证据展示；
- KAG 回溯；
- evaluation。

因此需要新增统一证据模型。

### 6.2 新增 `backend/app/schemas/evidence.py`

```python
from pydantic import BaseModel, Field
from typing import Any, Literal


RetrievalMethod = Literal[
    "bm25",
    "dense",
    "hybrid",
    "rerank",
    "graph",
    "deep_research",
    "cache"
]


class EvidenceChunk(BaseModel):
    evidence_id: str
    chunk_id: str | None = None
    doc_id: str | None = None
    title: str | None = None
    regulator: str | None = None
    jurisdiction: str = "Hong Kong"
    doc_type: str | None = None
    issue_date: str | None = None
    effective_date: str | None = None
    page: int | None = None
    section_title: str | None = None
    hierarchy_path: str | None = None
    source_url: str | None = None
    text: str
    retrieval_method: RetrievalMethod = "hybrid"
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    query: str
    retrieval_mode: Literal["rag", "kag", "deep_research"]
    evidence_chunks: list[EvidenceChunk]
    graph_paths: list[dict[str, Any]] = []
    warnings: list[str] = []
```

### 6.3 新增 `backend/app/services/retrieval/evidence_renderer.py`

功能：

```text
将 EvidenceChunk[] 渲染为 Analyzer prompt 可读的 context。
```

输出格式：

```text
[Source 1 | doc_id=hkma_svf_amlcft_guideline_2023 | regulator=HKMA | p.12 | score=0.87]
Title: Guideline on AML/CFT for SVF Licensees
Section: 4. Customer Due Diligence
Path: Chapter 4 > Section 4.1

<chunk text>
```

实现：

```python
def render_evidence_context(evidence_chunks: list[EvidenceChunk]) -> str:
    parts = []
    for i, ev in enumerate(evidence_chunks, start=1):
        page = ev.page if ev.page is not None else "?"
        score = f"{ev.score:.3f}" if isinstance(ev.score, float) else "-"
        parts.append(
            f"[Source {i} | doc_id={ev.doc_id} | regulator={ev.regulator} | "
            f"p.{page} | score={score}]\n"
            f"Title: {ev.title or '-'}\n"
            f"Section: {ev.section_title or '-'}\n"
            f"Path: {ev.hierarchy_path or '-'}\n\n"
            f"{ev.text}"
        )
    return "\n\n---\n\n".join(parts)
```

### 6.4 新增转换工具

在 `backend/app/services/retrieval/retrieval_service.py` 中新增：

```python
from langchain_core.documents import Document
from app.schemas.evidence import EvidenceChunk


def document_to_evidence(
    doc: Document,
    index: int,
    retrieval_method: str = "hybrid",
) -> EvidenceChunk:
    meta = doc.metadata or {}
    return EvidenceChunk(
        evidence_id=f"source_{index}",
        chunk_id=meta.get("chunk_id"),
        doc_id=meta.get("doc_id") or meta.get("source_document"),
        title=meta.get("title") or meta.get("source_document"),
        regulator=meta.get("regulator"),
        doc_type=meta.get("doc_type"),
        issue_date=meta.get("issue_date"),
        effective_date=meta.get("effective_date"),
        page=meta.get("page"),
        section_title=meta.get("section_title"),
        hierarchy_path=meta.get("hierarchy_path"),
        source_url=meta.get("source_url"),
        text=doc.page_content,
        retrieval_method=retrieval_method,
        score=meta.get("rerank_score"),
        metadata=meta,
    )
```

### 6.5 Phase 2 验收标准

新增测试：

```bash
python -m pytest tests/test_evidence_schema.py -v
```

测试内容：

1. `EvidenceChunk` 可正常实例化；
2. `Document` 可转换为 `EvidenceChunk`；
3. `render_evidence_context()` 输出包含 `Source N`、`doc_id`、`page`、`section_title`；
4. Analyzer 仍能接收 `retrieved_docs` 字符串，不破坏旧流程。

---

## 7. Phase 3：Corpus Ingestion 与 Metadata-aware Chunking

### 7.1 目标

让 `_load_and_split_pdf()` 支持多文档，且每个 chunk 自动继承 manifest metadata。

当前 `builder.py` 中 `_load_and_split_pdf()` 只读取一个 `PDF_PATH`。本阶段应新增 corpus ingestion，不直接删除旧函数。

### 7.2 新增 `backend/app/services/corpus/corpus_ingestor.py`

功能：

```text
1. 加载 manifest 中的所有 PDF。
2. 对每个 PDF 调用现有 hierarchy parser。
3. 将 SourceDocument metadata 注入每个 LangChain Document。
4. 返回 tuple[Document, ...]，供 lru_cache 使用。
```

伪代码：

```python
from pathlib import Path
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.corpus.manifest_loader import load_source_manifest
from app.services.agents.document_parser import (
    parse_pdf_with_hierarchy,
    regulation_chunks_to_documents,
)


def _inject_source_metadata(doc: Document, source) -> Document:
    metadata = dict(doc.metadata or {})
    metadata.update({
        "doc_id": source.doc_id,
        "title": source.title,
        "regulator": source.regulator,
        "jurisdiction": source.jurisdiction,
        "doc_type": source.doc_type,
        "issue_date": source.issue_date,
        "effective_date": source.effective_date,
        "status": source.status,
        "sector": ",".join(source.sector),
        "topics": ",".join(source.topics),
        "module_tags": ",".join(source.module_tags),
        "source_url": source.source_url or "",
        "priority": source.priority,
        "language": source.language,
    })
    return Document(page_content=doc.page_content, metadata=metadata)


@lru_cache()
def load_and_split_corpus() -> tuple[Document, ...]:
    settings = get_settings()
    sources = load_source_manifest()

    if not sources:
        return tuple()

    all_docs: list[Document] = []

    for source in sources:
        pdf_path = Path(settings.REG_DOC_DIR) / source.file_path
        if not pdf_path.exists():
            print(f"⚠️ Skipping missing file: {pdf_path}")
            continue

        pages = PyPDFLoader(str(pdf_path)).load()
        chunks = parse_pdf_with_hierarchy(pages, source_name=source.doc_id)
        docs = regulation_chunks_to_documents(chunks)

        docs = [_inject_source_metadata(doc, source) for doc in docs]
        all_docs.extend(docs)

        print(f"✅ Ingested {source.doc_id}: {len(docs)} chunks")

    return tuple(all_docs)
```

注意：Chroma metadata 不适合直接存 list，因此 `sector`、`topics`、`module_tags` 初期可用逗号拼接字符串。后续 filter 时用 `$contains` 或在 query router 侧做候选筛选。

### 7.3 修改 `backend/app/services/agents/builder.py`

新增：

```python
def _load_and_split_documents() -> tuple:
    from app.services.corpus.corpus_ingestor import load_and_split_corpus

    corpus_docs = load_and_split_corpus()
    if corpus_docs:
        print(f"✅ Loaded corpus documents: {len(corpus_docs)} chunks")
        return corpus_docs

    print("⚠️ Corpus manifest empty, falling back to legacy PDF_PATH")
    return _load_and_split_pdf()
```

然后将：

```python
splits = list(_load_and_split_pdf())
```

替换为：

```python
splits = list(_load_and_split_documents())
```

影响位置：

```text
_build_chroma_db()
build_hybrid_retriever()
```

不要删除 `_load_and_split_pdf()`，它作为 fallback。

### 7.4 Phase 3 验收标准

运行：

```bash
cd backend
python -m pytest tests/test_manifest_loader.py tests/test_evidence_schema.py -v
python -c "from app.services.corpus.corpus_ingestor import load_and_split_corpus; print(len(load_and_split_corpus()))"
```

必须满足：

1. 能读取多份 PDF；
2. 每个 chunk metadata 包含 `doc_id`、`regulator`、`module_tags`、`topics`；
3. 如果 manifest 为空，仍能 fallback 到旧 PDF；
4. 原有 SVF 接口不报错。

---

## 8. Phase 4：Metadata-aware Retrieval

### 8.1 目标

让系统根据问题自动选择检索范围。例如：

| 问题 | 应优先检索 |
|---|---|
| SVF licensee AML/CFT | `module_tags` 包含 `svf`, `aml` |
| AI suspicious activity monitoring | `topics` 包含 `AI`, `transaction_monitoring`, `suspicious_activity` |
| PEP controls | `topics` 包含 `PEP`, `CDD`, `EDD` |
| SFC VASP AML | `regulator=SFC`, `topics=VASP`, `AML` |
| GenAI consumer protection | `topics=GenAI`, `consumer_protection` |

### 8.2 新增 `backend/app/services/retrieval/query_classifier.py`

```python
from pydantic import BaseModel


class QueryProfile(BaseModel):
    retrieval_mode: str = "rag"
    module_tags: list[str] = []
    regulators: list[str] = []
    topics: list[str] = []
    query_type: str = "default"
    bm25_weight: float = 0.4
    dense_weight: float = 0.6


def classify_query_profile(query: str) -> QueryProfile:
    text = query.lower()

    profile = QueryProfile()

    if any(k in text for k in ["svf", "stored value", "stored value facility", "儲值支付", "储值支付"]):
        profile.module_tags.append("svf")
        profile.regulators.append("HKMA")

    if any(k in text for k in ["aml", "cft", "money laundering", "洗钱", "反洗錢"]):
        profile.module_tags.append("aml")
        profile.topics.extend(["AML", "CFT"])

    if any(k in text for k in ["kyc", "cdd", "customer due diligence", "客户尽职", "客戶盡職"]):
        profile.topics.extend(["KYC", "CDD"])

    if any(k in text for k in ["pep", "politically exposed"]):
        profile.topics.append("PEP")

    if any(k in text for k in ["ai", "genai", "artificial intelligence", "生成式", "人工智能"]):
        profile.topics.extend(["AI", "GenAI"])

    if any(k in text for k in ["vasp", "virtual asset", "虚拟资产", "虛擬資產"]):
        profile.regulators.append("SFC")
        profile.topics.append("VASP")

    if any(k in text for k in ["stablecoin", "稳定币", "穩定幣"]):
        profile.topics.append("stablecoin")
        profile.regulators.append("HKMA")

    return profile
```

### 8.3 修改 Retriever 以支持 metadata filter

初期不要强依赖 Chroma `where` filter，因为现有 LangChain retriever 抽象可能不方便传动态 where。建议第一阶段采用两层过滤：

```text
1. 检索前：根据 query profile 调整 query。
2. 检索后：对返回 docs 根据 metadata 加权重排。
```

新增：

```python
def metadata_boost(doc, profile: QueryProfile) -> float:
    meta = doc.metadata or {}
    boost = 0.0

    module_tags = str(meta.get("module_tags", "")).lower()
    topics = str(meta.get("topics", "")).lower()
    regulator = str(meta.get("regulator", "")).lower()

    for tag in profile.module_tags:
        if tag.lower() in module_tags:
            boost += 0.08

    for topic in profile.topics:
        if topic.lower() in topics:
            boost += 0.05

    for reg in profile.regulators:
        if reg.lower() == regulator:
            boost += 0.05

    if meta.get("priority") == "P0":
        boost += 0.03

    return boost
```

在 rerank 后，将最终 score 调整为：

```python
final_score = rerank_score + metadata_boost
```

并写入 metadata：

```python
doc.metadata["metadata_boost"] = boost
doc.metadata["final_score"] = final_score
```

### 8.4 Phase 4 验收标准

新增测试：

```bash
python -m pytest tests/test_query_classifier.py -v
```

测试内容：

1. “SVF AML CDD” 能识别 `svf`、`aml`、`CDD`；
2. “AI suspicious activity monitoring” 能识别 `AI`、`suspicious_activity`；
3. “VASP AML” 能识别 `SFC`、`VASP`；
4. metadata boost 不报错；
5. 原有 retriever 在无 metadata 时仍可工作。

---

## 9. Phase 5：Citation Semantic Verifier

### 9.1 目标

当前 prompt 要求每条 claim 以 `[Source: N, p.X]` 结尾，但这只能检查格式，不能证明该 source 真的支持 claim。

本阶段新增语义级引用核验：

```text
claim + cited evidence → supported / partial / unsupported
```

### 9.2 新增 `backend/app/schemas/evidence.py` 中的 CitationAudit

追加：

```python
class CitationAuditItem(BaseModel):
    claim: str
    source_number: int | None = None
    cited_evidence_id: str | None = None
    support_status: str  # supported | partial | unsupported | missing_source
    explanation: str = ""
    confidence: float = 0.0


class CitationAuditReport(BaseModel):
    total_claims: int
    supported_claims: int
    partial_claims: int
    unsupported_claims: int
    missing_source_claims: int
    unsupported_claim_rate: float
    items: list[CitationAuditItem]
```

### 9.3 新增 `backend/app/services/retrieval/citation_verifier.py`

功能：

1. 从报告中抽取带 `[Source: N, p.X]` 的 claim；
2. 根据 source number 找到对应 evidence；
3. 调用 LLM 判断 claim 是否被 evidence 支持；
4. 输出 `CitationAuditReport`。

伪代码：

```python
import re
import json
from app.schemas.evidence import CitationAuditItem, CitationAuditReport

SOURCE_PATTERN = re.compile(r"\[Source:\s*(\d+),\s*p\.?\s*(\d+|\?)\]", re.IGNORECASE)


def extract_cited_claims(report: str) -> list[tuple[str, int]]:
    claims = []
    for line in report.splitlines():
        match = SOURCE_PATTERN.search(line)
        if match:
            source_no = int(match.group(1))
            clean_claim = SOURCE_PATTERN.sub("", line).strip(" -•")
            if clean_claim:
                claims.append((clean_claim, source_no))
    return claims


def verify_claim_with_llm(claim: str, evidence_text: str, llm) -> dict:
    prompt = f"""
You are a strict financial regulatory citation auditor.

Claim:
{claim}

Evidence:
{evidence_text}

Determine whether the evidence supports the claim.

Return JSON only:
{{
  "support_status": "supported | partial | unsupported",
  "explanation": "...",
  "confidence": 0.0
}}
"""
    resp = llm.invoke(prompt)
    try:
        return json.loads(resp.content)
    except Exception:
        return {
            "support_status": "partial",
            "explanation": "Failed to parse verifier output.",
            "confidence": 0.3,
        }
```

必须提供 fallback：

```python
def verify_citations(report, evidence_chunks, llm=None):
    if llm is None:
        # fallback: only format/source existence check
        ...
```

### 9.4 集成到 `svf.py`

在 `reviewer_node` 前或后新增节点：

```text
analyzer
→ format_validator
→ citation_verifier
→ reviewer
```

新增 state 字段：

```python
citation_audit: Dict
unsupported_claim_rate: float
```

如果 `unsupported_claim_rate > 0.15`，Reviewer 应更倾向于 `quality_issue`。

### 9.5 Phase 5 验收标准

新增测试：

```bash
python -m pytest tests/test_citation_verifier.py -v
```

测试内容：

1. 能抽取 `[Source: 1, p.3]`；
2. source number 不存在时标记 `missing_source`；
3. LLM 不可用时 fallback 不报错；
4. audit report 能计算 unsupported rate；
5. SVF workflow 中 citation audit 字段存在。

---

## 10. Phase 6：KAG Regulatory Graph 最小可用版本

### 10.1 目标

基于已有 `RegulationChunk` metadata 和 manifest metadata 构建轻量监管知识图谱。

第一阶段 KAG 不追求自动精准抽取所有法规关系，而是先做半规则化图谱：

```text
Document → contains → Clause/Chunk
Document → issued_by → Regulator
Clause/Chunk → related_to → Topic
Clause/Chunk → applies_to → Sector/Module
Clause/Chunk → supported_by → EvidenceChunk
```

### 10.2 更新 `requirements.txt`

新增：

```txt
networkx>=3.2.0
```

### 10.3 新增 `backend/app/schemas/kag.py`

```python
from pydantic import BaseModel, Field
from typing import Any


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    evidence_chunk_id: str | None = None
    confidence: float = 0.8
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphPath(BaseModel):
    path_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    evidence_chunk_ids: list[str] = []
    score: float = 0.0
```

### 10.4 新增 `backend/app/services/kag/ontology.py`

```python
ENTITY_TYPES = [
    "Regulator",
    "Document",
    "Clause",
    "Institution",
    "LicenseType",
    "Product",
    "Obligation",
    "Risk",
    "Process",
    "DataCategory",
    "Topic",
    "Sector",
]

RELATION_TYPES = [
    "issued_by",
    "contains",
    "applies_to",
    "requires",
    "prohibits",
    "references",
    "supersedes",
    "effective_from",
    "related_to",
    "supported_by",
]
```

### 10.5 新增 `backend/app/services/kag/graph_store.py`

使用 NetworkX：

```python
import json
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path
from app.core.config import get_settings


class RegulatoryGraphStore:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: str, node_type: str, name: str, **properties):
        self.graph.add_node(
            node_id,
            node_type=node_type,
            name=name,
            **properties,
        )

    def add_edge(self, source: str, target: str, relation: str, **properties):
        self.graph.add_edge(
            source,
            target,
            relation=relation,
            **properties,
        )

    def save(self, path: str | None = None):
        settings = get_settings()
        target = Path(path or settings.GRAPH_STORE_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.graph)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | None = None):
        settings = get_settings()
        target = Path(path or settings.GRAPH_STORE_PATH)
        if not target.exists():
            return
        data = json.loads(target.read_text(encoding="utf-8"))
        self.graph = json_graph.node_link_graph(data, directed=True, multigraph=True)
```

### 10.6 新增 `backend/app/services/kag/graph_builder.py`

从 evidence chunks 构建图：

```python
def build_graph_from_documents(documents) -> RegulatoryGraphStore:
    store = RegulatoryGraphStore()

    for doc in documents:
        meta = doc.metadata or {}

        doc_id = meta.get("doc_id") or meta.get("source_document")
        regulator = meta.get("regulator", "Unknown")
        chunk_id = meta.get("chunk_id")
        title = meta.get("title") or doc_id

        if not doc_id or not chunk_id:
            continue

        store.add_node(regulator, "Regulator", regulator)
        store.add_node(doc_id, "Document", title, **meta)
        store.add_node(chunk_id, "Clause", meta.get("section_title", chunk_id), **meta)

        store.add_edge(doc_id, regulator, "issued_by")
        store.add_edge(doc_id, chunk_id, "contains")
        store.add_edge(chunk_id, doc_id, "supported_by", evidence_chunk_id=chunk_id)

        for topic in str(meta.get("topics", "")).split(","):
            topic = topic.strip()
            if topic:
                topic_id = f"topic:{topic.lower()}"
                store.add_node(topic_id, "Topic", topic)
                store.add_edge(chunk_id, topic_id, "related_to", evidence_chunk_id=chunk_id)

        for tag in str(meta.get("module_tags", "")).split(","):
            tag = tag.strip()
            if tag:
                tag_id = f"module:{tag.lower()}"
                store.add_node(tag_id, "Sector", tag)
                store.add_edge(chunk_id, tag_id, "applies_to", evidence_chunk_id=chunk_id)

    return store
```

### 10.7 新增 `backend/app/services/kag/graph_retriever.py`

最小功能：

```text
输入 query profile；
根据 topics/module_tags/regulators 找相关节点；
取一跳/两跳邻居；
返回 graph paths；
回溯 chunk_id；
再从 evidence index 中取原文。
```

伪代码：

```python
def retrieve_graph_paths(query: str, profile: QueryProfile) -> list[GraphPath]:
    store = RegulatoryGraphStore()
    store.load()

    seed_node_ids = []

    for topic in profile.topics:
        seed_node_ids.append(f"topic:{topic.lower()}")

    for tag in profile.module_tags:
        seed_node_ids.append(f"module:{tag.lower()}")

    for reg in profile.regulators:
        seed_node_ids.append(reg)

    paths = []
    for seed in seed_node_ids:
        if seed not in store.graph:
            continue

        for predecessor in store.graph.predecessors(seed):
            # predecessor 通常是 chunk_id
            # 再找 Document / Regulator
            paths.append(...)
    return paths
```

### 10.8 Phase 6 验收标准

新增测试：

```bash
python -m pytest tests/test_kag_graph_store.py -v
```

测试内容：

1. 可创建 graph；
2. 可添加 node/edge；
3. 可保存为 JSON；
4. 可从 JSON 读取；
5. 图谱中至少包含 Regulator、Document、Clause、Topic；
6. 输入 `SVF AML` 能召回相关 graph path；
7. graph path 能回溯到 evidence chunk。

---

## 11. Phase 7：Retrieval Router

### 11.1 目标

根据问题自动选择：

```text
rag
kag
deep_research
```

### 11.2 新增 `backend/app/services/retrieval/retrieval_router.py`

```python
from typing import Literal
from app.services.retrieval.query_classifier import classify_query_profile


RetrievalMode = Literal["rag", "kag", "deep_research"]


def classify_retrieval_mode(query: str) -> RetrievalMode:
    text = query.lower()

    deep_keywords = [
        "分析", "比較", "比较", "生成报告", "研究", "上线前", "檢查清單", "检查清单",
        "compare", "analyze", "analysis", "report", "framework", "roadmap", "checklist"
    ]

    graph_keywords = [
        "适用于", "適用於", "涉及哪些", "监管机构", "監管機構",
        "义务", "義務", "关系", "關係", "取代", "引用",
        "supersede", "applies to", "obligation", "effective date", "relationship"
    ]

    if any(k in text for k in deep_keywords):
        return "deep_research"

    if any(k in text for k in graph_keywords):
        return "kag"

    return "rag"


def route_query(query: str):
    profile = classify_query_profile(query)
    mode = classify_retrieval_mode(query)
    profile.retrieval_mode = mode
    return profile
```

### 11.3 Phase 7 验收标准

新增测试：

```bash
python -m pytest tests/test_retrieval_router.py -v
```

测试样例：

| 输入 | 预期 |
|---|---|
| “SVF 的 CDD 要求是什么？” | `rag` |
| “AI 投顾涉及哪些监管机构和义务？” | `kag` |
| “请分析虚拟银行推出 AI 投顾的合规风险并生成 checklist” | `deep_research` |
| “Compare HKMA and SFC requirements for AI monitoring” | `deep_research` |

---

## 12. Phase 8：DeepResearch Workflow

### 12.1 目标

将现有 Reviewer 触发的二次检索升级为完整 DeepResearch 工作流。

DeepResearch 不是调用某个外部产品，而是在本项目内部实现以下能力：

```text
复杂问题
→ 研究计划
→ 子问题拆解
→ 每个子问题路由到 RAG/KAG
→ 证据覆盖评估
→ 缺口补充检索
→ 结构化报告生成
→ 引用核验
```

### 12.2 新增 `backend/app/schemas/deepresearch.py`

```python
from pydantic import BaseModel, Field
from typing import Any


class ResearchSubQuestion(BaseModel):
    id: str
    question: str
    retrieval_mode: str = "rag"
    required_topics: list[str] = []
    evidence_min_count: int = 2


class ResearchPlan(BaseModel):
    research_goal: str
    sub_questions: list[ResearchSubQuestion]
    expected_output_sections: list[str] = []


class EvidenceGap(BaseModel):
    sub_question_id: str
    reason: str
    suggested_followup_query: str


class DeepResearchResult(BaseModel):
    original_query: str
    research_plan: ResearchPlan
    evidence_by_subquestion: dict[str, list[dict]]
    evidence_gaps: list[EvidenceGap] = []
    final_report: str
    citation_audit: dict[str, Any] = {}
```

### 12.3 新增 `backend/app/services/deepresearch/planner.py`

```python
def build_research_plan(query: str, llm) -> ResearchPlan:
    prompt = f"""
You are a financial regulatory research planner for Hong Kong financial compliance.

Given the user query, decompose it into 4-8 focused sub-questions.
Each sub-question should be answerable by RAG or KAG retrieval.

User Query:
{query}

Return JSON matching this schema:
{{
  "research_goal": "...",
  "sub_questions": [
    {{
      "id": "SQ1",
      "question": "...",
      "retrieval_mode": "rag | kag",
      "required_topics": [],
      "evidence_min_count": 2
    }}
  ],
  "expected_output_sections": []
}}
"""
    ...
```

Fallback：如果 LLM 输出解析失败，使用规则拆解：

```python
def fallback_research_plan(query: str) -> ResearchPlan:
    return ResearchPlan(
        research_goal=query,
        sub_questions=[
            ResearchSubQuestion(id="SQ1", question=f"What are the relevant HKMA requirements for: {query}", retrieval_mode="rag"),
            ResearchSubQuestion(id="SQ2", question=f"What are the relevant SFC requirements for: {query}", retrieval_mode="rag"),
            ResearchSubQuestion(id="SQ3", question=f"What AML/CFT obligations are relevant to: {query}", retrieval_mode="kag"),
            ResearchSubQuestion(id="SQ4", question=f"What AI/data/privacy risks are relevant to: {query}", retrieval_mode="kag"),
        ],
        expected_output_sections=[
            "Executive Summary",
            "Regulatory Scope",
            "Key Obligations",
            "Risk Analysis",
            "Compliance Checklist",
            "Information Gaps",
        ],
    )
```

### 12.4 新增 `backend/app/services/deepresearch/evidence_evaluator.py`

功能：

```text
判断每个 sub-question 是否有足够证据。
```

规则：

```python
def evaluate_evidence_coverage(plan, evidence_by_subquestion):
    gaps = []
    for sq in plan.sub_questions:
        evidence = evidence_by_subquestion.get(sq.id, [])
        if len(evidence) < sq.evidence_min_count:
            gaps.append(
                EvidenceGap(
                    sub_question_id=sq.id,
                    reason=f"Only {len(evidence)} evidence chunks found; minimum {sq.evidence_min_count} required.",
                    suggested_followup_query=sq.question,
                )
            )
    return gaps
```

### 12.5 新增 `backend/app/services/deepresearch/workflow.py`

使用 LangGraph：

```python
from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END


class DeepResearchState(TypedDict, total=False):
    original_query: str
    research_plan: dict
    evidence_by_subquestion: Dict[str, List[dict]]
    evidence_gaps: List[dict]
    iteration: int
    draft_report: str
    citation_audit: dict
    final_report: str


def build_deepresearch_graph():
    graph = StateGraph(DeepResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("evidence_evaluator", evidence_evaluator_node)
    graph.add_node("gap_retriever", gap_retriever_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("citation_verifier", citation_verifier_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "evidence_evaluator")

    graph.add_conditional_edges(
        "evidence_evaluator",
        route_after_evidence_eval,
        {
            "retrieve_more": "gap_retriever",
            "write_report": "report_writer",
        }
    )

    graph.add_edge("gap_retriever", "evidence_evaluator")
    graph.add_edge("report_writer", "citation_verifier")
    graph.add_edge("citation_verifier", END)

    return graph.compile()
```

路由函数：

```python
def route_after_evidence_eval(state: DeepResearchState) -> str:
    max_iter = 3
    if state.get("evidence_gaps") and state.get("iteration", 0) < max_iter:
        return "retrieve_more"
    return "write_report"
```

### 12.6 DeepResearch Report Writer Prompt

新增 `backend/app/services/deepresearch/prompts.py`：

```python
DEEP_RESEARCH_REPORT_PROMPT = """
You are a senior Hong Kong financial regulatory compliance analyst.

Your task is to generate a structured research report based strictly on the provided evidence.

Rules:
1. Every regulatory claim must cite a Source ID.
2. Do not use unstated regulatory knowledge.
3. Separate facts from recommendations.
4. Clearly flag information gaps.
5. Provide a practical compliance checklist.

Original Query:
{query}

Research Plan:
{research_plan}

Evidence:
{evidence_context}

Required Report Structure:
# DeepResearch Compliance Report

## 1. Executive Summary
## 2. Regulatory Scope
## 3. Key Regulatory Obligations
## 4. Risk Analysis
## 5. Compliance Checklist
## 6. Information Gaps and Limitations
## 7. Source-based Evidence Table

Write in Traditional Chinese suitable for Hong Kong financial compliance context.
"""
```

### 12.7 Phase 8 验收标准

必须能运行：

```python
from app.services.deepresearch.workflow import build_deepresearch_graph

graph = build_deepresearch_graph()
result = graph.invoke({
    "original_query": "請分析香港虛擬銀行推出 AI 投資顧問的合規風險，並生成上線前檢查清單。"
})
print(result["final_report"])
```

必须满足：

1. 生成 research plan；
2. 至少 4 个 sub-questions；
3. 每个 sub-question 有 evidence；
4. 不超过最大迭代次数；
5. final report 包含 checklist；
6. citation audit 可执行；
7. LLM 解析失败时 fallback 不崩溃。

---

## 13. Phase 9：SVF 模块集成新检索系统

### 13.1 目标

当前 `svf.py` 中的 `retriever_node` 直接调用 `base_retriever`。本阶段要将其升级为：

```text
Extractor
→ Retrieval Router
→ RAG / KAG / DeepResearch
→ Analyzer
→ Format Validator
→ Citation Verifier
→ Reviewer
```

### 13.2 扩展 `SVFState`

新增字段：

```python
retrieval_mode: str
query_profile: Dict
evidence_chunks: List[Dict]
graph_paths: List[Dict]
research_plan: Dict
evidence_by_subquestion: Dict
evidence_gaps: List[Dict]
citation_audit: Dict
unsupported_claim_rate: float
```

### 13.3 新增 `retrieval_router_node`

```python
def retrieval_router_node(state: SVFState):
    from app.services.retrieval.retrieval_router import route_query

    query = state.get("extracted_entities") or state["original_input"]
    profile = route_query(query)

    return {
        "retrieval_mode": profile.retrieval_mode,
        "query_profile": profile.model_dump(),
    }
```

### 13.4 改造 `retriever_node`

逻辑：

```python
def retriever_node(state: SVFState):
    mode = state.get("retrieval_mode", "rag")

    if mode == "deep_research":
        return run_deepresearch_for_state(state)

    if mode == "kag":
        return run_kag_retrieval_for_state(state)

    return run_rag_retrieval_for_state(state)
```

注意：

```text
第一阶段为了稳定，可以让 DeepResearch 只在新的 /research endpoint 启用。
SVF 原 endpoint 默认仍使用 RAG/KAG。
等测试稳定后，再允许 SVF endpoint 自动进入 deep_research。
```

### 13.5 Graph 路由修改

原始大概是：

```text
extractor → retriever → analyzer → format_validator → reviewer
```

升级为：

```text
extractor
→ retrieval_router
→ retriever
→ analyzer
→ format_validator
→ citation_verifier
→ reviewer
```

Reviewer 后继续保留原条件边：

```text
approved → END
insufficient_info → sub_query_planner / retriever
quality_issue → analyzer
```

### 13.6 Phase 9 验收标准

运行原有 API：

```bash
uvicorn app.main:app --reload --port 8000
```

测试：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/svf/analyze \
  -H "Content-Type: application/json" \
  -d '{"application_data": "An SVF licensee wants to strengthen CDD and suspicious transaction monitoring for high-risk customers. What are the key AML/CFT obligations?"}'
```

必须满足：

1. 原 SVF API 可用；
2. 响应仍包含最终报告；
3. 检索来源不再只来自单 PDF；
4. evidence chunks 至少包含 `doc_id`；
5. citation audit 被执行；
6. Reviewer 反思循环不被破坏。

---

## 14. Phase 10：新增 DeepResearch API

### 14.1 目标

新增独立研究接口，避免 SVF 原接口过于复杂。

路径：

```text
backend/app/api/routers/research.py
```

接口：

```text
POST /api/v1/research/analyze
POST /api/v1/research/analyze/stream
```

### 14.2 Request Schema

可复用 `ComplianceRequest`，或新增：

```python
class ResearchRequest(BaseModel):
    query: str
    module: str | None = None
    max_iterations: int = 3
    language: str = "zh-HK"
```

### 14.3 Router 示例

```python
from fastapi import APIRouter
from app.schemas.deepresearch import DeepResearchResult
from app.services.deepresearch.workflow import build_deepresearch_graph

router = APIRouter(prefix="/research", tags=["DeepResearch"])


@router.post("/analyze")
def analyze_research(request: ResearchRequest):
    graph = build_deepresearch_graph()
    result = graph.invoke({
        "original_query": request.query,
        "iteration": 0,
    })
    return result
```

SSE 版本可第二阶段再做，第一阶段先完成阻塞式接口。

### 14.4 Phase 10 验收标准

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "請分析香港虛擬銀行推出 AI 投資顧問的合規風險，並生成上線前檢查清單。"}'
```

必须返回：

```json
{
  "research_plan": "...",
  "evidence_by_subquestion": "...",
  "final_report": "...",
  "citation_audit": "..."
}
```

---

## 15. Phase 11：Evaluation Benchmark

### 15.1 目标

让项目从 demo 变成可评估系统。

新增：

```text
backend/data/evaluation/benchmark_questions.json
```

示例：

```json
[
  {
    "id": "RAG_SVF_AML_001",
    "question": "What are the key CDD obligations for an SVF licensee?",
    "expected_retrieval_mode": "rag",
    "expected_topics": ["CDD", "AML", "SVF"],
    "expected_regulators": ["HKMA"]
  },
  {
    "id": "KAG_AI_ADVISOR_001",
    "question": "Which regulators and obligations are relevant when a Hong Kong virtual bank launches an AI wealth advisory product?",
    "expected_retrieval_mode": "kag",
    "expected_topics": ["AI", "wealth_management", "consumer_protection"],
    "expected_regulators": ["HKMA", "SFC", "PCPD"]
  },
  {
    "id": "DR_AI_ADVISOR_001",
    "question": "Analyze the compliance risks for a Hong Kong virtual bank launching an AI investment advisor and generate a pre-launch checklist.",
    "expected_retrieval_mode": "deep_research",
    "expected_topics": ["AI", "AML", "CDD", "data_privacy", "consumer_protection"],
    "expected_regulators": ["HKMA", "SFC", "PCPD"]
  }
]
```

### 15.2 新增 `backend/app/services/evaluation/run_eval.py`

输出指标：

| 指标 | 含义 |
|---|---|
| retrieval_mode_accuracy | Router 是否选对模式 |
| evidence_count | 每个问题召回 evidence 数 |
| regulator_coverage | 是否覆盖预期监管机构 |
| topic_coverage | 是否覆盖预期 topic |
| citation_supported_rate | 引用支持率 |
| unsupported_claim_rate | 未支持结论比例 |
| graph_path_count | KAG 路径数量 |
| deepresearch_gap_count | DeepResearch 证据缺口数量 |

### 15.3 Phase 11 验收标准

运行：

```bash
python -m app.services.evaluation.run_eval
```

输出类似：

```text
Evaluation Summary
- total_questions: 10
- retrieval_mode_accuracy: 0.90
- avg_evidence_count: 5.4
- avg_citation_supported_rate: 0.82
- avg_unsupported_claim_rate: 0.08
```

---

## 16. Phase 12：前端展示升级

### 16.1 目标

前端不需要大改，但建议新增三个展示区域：

```text
1. Evidence Panel
2. Knowledge Graph Panel
3. DeepResearch Plan Panel
```

### 16.2 Evidence Panel

展示字段：

```text
Source ID
Document Title
Regulator
Page
Section
Retrieval Method
Score
Citation Support Status
```

### 16.3 Knowledge Graph Panel

展示：

```text
Regulator → Document → Clause → Topic / Obligation / Risk
```

第一阶段可用文本树，不必立即做图可视化。

示例：

```text
HKMA
└── Guideline on AML/CFT for SVF Licensees
    └── Section 4: Customer Due Diligence
        ├── related_to: CDD
        ├── applies_to: SVF
        └── supported_by: Source 3
```

### 16.4 DeepResearch Plan Panel

展示：

```text
Research Goal
Sub-questions
Evidence coverage per sub-question
Evidence gaps
Final report
```

### 16.5 Phase 12 验收标准

1. 原有报告展示不破坏；
2. 如果后端返回 `evidence_chunks`，前端能展示；
3. 如果后端返回 `graph_paths`，前端能展示文本路径；
4. 如果后端返回 `research_plan`，前端能展示子问题；
5. SSE 流式输出继续可用。

---

## 17. 最终验收测试清单

### 17.1 后端启动

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

启动时应看到：

```text
✅ Loaded source manifest
✅ Ingested corpus documents
✅ ChromaDB vector store initialized
✅ Hybrid Retriever initialized
✅ Reranked Retriever initialized
✅ Regulatory graph loaded or built
```

### 17.2 SVF RAG 测试

输入：

```text
An SVF licensee wants to improve CDD and suspicious transaction monitoring for high-risk customers. What are the key AML/CFT obligations?
```

期望：

```text
retrieval_mode = rag
evidence 包含 HKMA SVF AML/CFT 文档
报告包含 Source citation
citation audit 执行
```

### 17.3 KAG 测试

输入：

```text
虛擬銀行推出 AI 財富管理顧問時，涉及哪些監管機構、合規義務和風險類型？
```

期望：

```text
retrieval_mode = kag
graph_paths 非空
evidence 包含 HKMA / SFC / PCPD 相关来源
报告中解释监管关系
```

### 17.4 DeepResearch 测试

输入：

```text
請分析香港虛擬銀行推出 AI 投資顧問的合規風險，並生成產品上線前的合規檢查清單。
```

期望：

```text
retrieval_mode = deep_research
research_plan 存在
sub_questions >= 4
evidence_by_subquestion 非空
final_report 包含 checklist
citation_audit 存在
```

---

## 18. 风险控制与回滚策略

### 18.1 风险：多文档索引导致检索噪声增加

解决：

```text
1. 加 metadata boost。
2. 加 module_tags filter。
3. 对 P0 文档加 priority boost。
4. benchmark 中观察 topic coverage。
```

### 18.2 风险：DeepResearch 成本和延迟太高

解决：

```text
1. 默认 SVF 接口不自动进入 DeepResearch。
2. 新增独立 /research/analyze。
3. 设置 DEEP_RESEARCH_MAX_ITERATIONS=3。
4. 设置 DEEP_RESEARCH_MAX_SUBQUESTIONS=8。
```

### 18.3 风险：LLM JSON 输出不稳定

解决：

```text
1. 所有 JSON parser 必须 try/except。
2. 每个 LLM planner 必须有 fallback。
3. 不允许因为 planner 失败导致主流程崩溃。
```

### 18.4 风险：Citation verifier 误判

解决：

```text
1. verifier 输出只作为 confidence signal。
2. 不直接删除报告内容。
3. unsupported_claim_rate 高时交给 Reviewer 判断。
```

### 18.5 回滚方式

保留：

```python
PDF_PATH
_load_and_split_pdf()
build_reranked_retriever()
原 SVF analyzer / reviewer prompt
```

如果 corpus ingestion 出错：

```text
自动 fallback 到 legacy PDF_PATH。
```

---

## 19. Agent IDE 最终执行指令

Agent IDE 应按以下顺序执行：

```text
Step 1:
读取 backend/app/core/config.py、backend/app/services/agents/builder.py、
backend/app/services/agents/document_parser.py、backend/app/api/routers/svf.py。

Step 2:
新增 corpus schema、manifest_loader、corpus_ingestor。
不要删除旧 PDF_PATH 逻辑。

Step 3:
新增 evidence schema 和 evidence_renderer。
修改 retriever 输出，使其同时保留 retrieved_docs 字符串和 evidence_chunks 结构化数据。

Step 4:
新增 query_classifier 和 metadata-aware retrieval boost。
不要破坏现有 BM25 + Chroma + Cohere Rerank。

Step 5:
新增 citation_verifier。
先做格式与 source existence fallback，再接 LLM semantic verification。

Step 6:
新增 KAG ontology、graph_store、graph_builder、graph_retriever。
第一版使用 NetworkX，不使用 Neo4j。

Step 7:
新增 retrieval_router。
将 query 路由为 rag / kag / deep_research。

Step 8:
新增 DeepResearch schema、planner、evidence_evaluator、workflow、report_writer。
所有循环必须有最大轮次。

Step 9:
将 SVF workflow 接入 retrieval_router、evidence_chunks、citation_verifier。
保持原 endpoint 可用。

Step 10:
新增 /research/analyze 独立接口。
先实现阻塞式，SSE 后续再做。

Step 11:
新增 benchmark_questions.json 和 run_eval.py。

Step 12:
更新 README 和 docs。
说明如何添加新监管文档、如何更新 manifest、如何重建索引、如何运行评估。
```

---

## 20. 完成后的项目定位

完成本计划后，项目可以从：

```text
HK-FinReg AI：香港 SVF 合规 RAG 审查平台
```

升级为：

```text
HK-FinReg Research Agent：
面向香港金融监管场景的可信 RAG + KAG + DeepResearch 合规研究系统
```

技术亮点应表述为：

```text
1. 多来源香港金融监管文档语料库；
2. Metadata-aware Hybrid RAG；
3. BM25 + Dense Retrieval + RRF + Cohere Reranker；
4. EvidenceChunk 结构化证据追踪；
5. Citation semantic verification；
6. Regulatory Knowledge Graph；
7. KAG-style graph path retrieval；
8. DeepResearch-style query decomposition and evidence gap detection；
9. LangGraph multi-agent reflection workflow；
10. Retrieval / reasoning / citation 多维可信度评估。
```

最终简历 bullet 可写为：

```text
Developed HK-FinReg Research Agent, a trustworthy financial compliance research system integrating metadata-aware Hybrid RAG, regulatory knowledge graph retrieval, and DeepResearch-style multi-step evidence synthesis for Hong Kong financial regulation analysis.

Built a structured regulatory corpus with source manifest, document-level metadata, EvidenceChunk tracking, citation verification, and retrieval evaluation across HKMA, SFC, and PCPD guidance documents.

Implemented a LangGraph-based research workflow for query decomposition, RAG/KAG routing, evidence gap detection, iterative retrieval, and source-grounded compliance report generation.
```

---

## 21. 最小成功版本定义

如果时间有限，Agent IDE 至少必须完成以下最小版本：

```text
MVP 必须包含：
1. source_manifest.json
2. 多 PDF ingestion
3. EvidenceChunk
4. metadata-aware RAG
5. citation verifier fallback
6. NetworkX regulatory graph
7. retrieval_router
8. /research/analyze 基础 DeepResearch
9. benchmark_questions.json
10. README 更新
```

MVP 不要求：

```text
1. Neo4j
2. 前端复杂图谱可视化
3. 完整 OpenSPG KAG
4. 所有业务模块 RAG 化
5. 全自动法律判断
```

---

## 22. 最终交付物清单

Agent IDE 完成后，应交付：

```text
代码文件：
- backend/app/schemas/corpus.py
- backend/app/schemas/evidence.py
- backend/app/schemas/kag.py
- backend/app/schemas/deepresearch.py
- backend/app/services/corpus/*
- backend/app/services/retrieval/*
- backend/app/services/kag/*
- backend/app/services/deepresearch/*
- backend/app/services/evaluation/*
- backend/app/api/routers/research.py

数据文件：
- backend/data/source_manifest.json
- backend/data/graph/regulatory_graph.json
- backend/data/evaluation/benchmark_questions.json

测试文件：
- backend/tests/test_manifest_loader.py
- backend/tests/test_evidence_schema.py
- backend/tests/test_query_classifier.py
- backend/tests/test_retrieval_router.py
- backend/tests/test_kag_graph_store.py
- backend/tests/test_citation_verifier.py

文档：
- docs/rag_kag_deepresearch_update_plan.md
- docs/evaluation_protocol.md
- README.md 更新
```

---

## 23. 一句话总结

本次更新的核心不是“再加一些 PDF”，而是要把这些新下载的监管文档转化为一个**可路由、可追踪、可推理、可核验、可评估**的香港金融监管知识系统：

```text
Regulatory Corpus
→ Metadata-aware Advanced RAG
→ EvidenceChunk Tracking
→ Citation Verification
→ Regulatory Knowledge Graph
→ KAG Retrieval
→ DeepResearch Workflow
→ Compliance Research Report
```
