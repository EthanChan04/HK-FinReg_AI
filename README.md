<div align="center">

# 🏛️ HK-FinReg AI

### 香港金融科技合規 AI 平台

**多智能體 (Multi-Agent) · 混合檢索 (Hybrid RAG) · 三維置信度 · 反思循環 · 深度防幻覺 · 全鏈路可觀測**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js_14-Frontend-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B35?logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-1C3C3C?logo=langchain&logoColor=white)](https://smith.langchain.com/)
[![Cohere](https://img.shields.io/badge/Cohere-Reranker-39594D?logo=cohere&logoColor=white)](https://cohere.com/)

</div>

---

## 📋 目錄

- [項目概覽](#項目概覽)
- [系統架構](#系統架構)
- [RAG 引擎亮點](#rag-引擎亮點)
- [反思循環與三維置信度](#反思循環與三維置信度)
- [深度可觀測性](#深度可觀測性)
- [業務模組](#業務模組)
- [快速啟動](#快速啟動)
- [項目結構](#項目結構)
- [技術棧](#技術棧)
- [授權條款](#授權條款)

---

## 項目概覽

**HK-FinReg AI** 是一個面向香港金融監管場景的 AI 合規審查平台，採用**前後端分離架構**，結合大語言模型（LLM）、多智能體協同（Multi-Agent）與進階 RAG 技術，為金融機構提供自動化的合規風險評估。

### 核心設計理念

| 設計原則 | 實現方式 |
|---------|---------|
| **嚴謹性** | 強制溯源機制 — 報告中每條法規引用必須標注 `[Source: Source N, p.X]`，精確到頁碼；Pydantic 結構化輸出驗證 |
| **抗幻覺** | 四重防線 — ① Hybrid RAG 精準檢索 ② Cohere Reranker 精排 ③ FormatValidator 格式驗證 ④ Reviewer Agent 紅藍對抗審計 |
| **自糾正** | 反思循環 — 三路條件邊（通過/修訂/二次檢索），報告質量不足時自動觸發修訂或補充檢索 |
| **可量化** | 三維置信度 — 檢索置信度 × 推理置信度 × 交叉驗證，為報告結論提供質量信號 |
| **可觀測** | LangSmith 全鏈路追蹤 + SSE 逐節點實時推送 — 每個 Agent 節點、Token 消耗、檢索質量一覽無遺 |
| **抗焦慮** | SSE 流式推送 + Agent 思考鏈路動態可視化 + 三維置信度實時展示 — 審查過程全程可感知 |

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 14)                        │
│  App Router · Tailwind CSS · react-markdown · Dark Mode             │
│                                                                     │
│  ┌──────────┐  ┌─────────────────┐  ┌───────────────────────┐      │
│  │ 模組選擇  │  │ Agent Pipeline  │  │  Markdown Report      │      │
│  │ 4-Tab Nav │  │ 思考鏈路指示器   │  │  流式富文本 + 置信度   │      │
│  └──────────┘  └─────────────────┘  └───────────────────────┘      │
│                         ▲ SSE (text/event-stream)                   │
│                           agent_state / token / confidence / done   │
└─────────────────────────┼───────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────────┐
│                  Backend (FastAPI)                                   │
│                         │                                           │
│  ┌──────────────────────▼──────────────────────────────────┐        │
│  │           LangGraph Multi-Agent Workflow                 │        │
│  │                                                          │        │
│  │  ┌───────────┐    ┌────────────┐    ┌──────────────┐    │        │
│  │  │ Extractor  │───▶│  Retriever │───▶│   Analyzer   │    │        │
│  │  │   Agent    │    │   Agent    │    │    Agent     │    │        │
│  │  └───────────┘    └────────────┘    └──────┬───────┘    │        │
│  │                          │                  │            │        │
│  │                     ┌────▼────┐    ┌────────▼────────┐  │        │
│  │                     │ Hybrid  │    │ Format Validator │  │        │
│  │                     │  RAG    │    │  (Pydantic 校驗) │  │        │
│  │                     │ Engine  │    └────────┬────────┘  │        │
│  │                     └─────────┘             │            │        │
│  │                                    ┌────────▼────────┐  │        │
│  │                                    │    Reviewer     │  │        │
│  │                                    │     Agent       │  │        │
│  │                                    │  (紅藍對抗+     │  │        │
│  │                                    │   交叉驗證)     │  │        │
│  │                                    └───┬────────┬────┘  │        │
│  │                     ┌─────────────┐   │        │       │        │
│  │                     │SubQueryPlan │◀──┘        │       │        │
│  │                     │ (規則化·零LLM)│           │       │        │
│  │                     └──────┬──────┘           │       │        │
│  │                            │                  │       │        │
│  │                     ┌──────▼──────┐           │       │        │
│  │                     │  Retriever  │           │       │        │
│  │                     │ (二次檢索)  │──────────▶│       │        │
│  │                     └─────────────┘    (反思循環)      │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐        │
│  │ ChromaDB     │  │ BM25 Index   │  │ Cohere Reranker   │        │
│  │ (Dense)      │  │ (Sparse)     │  │ (Cloud API)       │        │
│  └──────────────┘  └──────────────┘  └───────────────────┘        │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐        │
│  │Semantic Cache│  │ Doc Parser   │  │ workflow_utils    │        │
│  │ (Embedding)  │  │ (層級感知)   │  │ (共享工具函數)    │        │
│  └──────────────┘  └──────────────┘  └───────────────────┘        │
│                                                                    │
│              LangSmith ← 全鏈路 Trace 追蹤                         │
└────────────────────────────────────────────────────────────────────┘
```

### 前端 — Next.js 14

- **App Router** + React Server Components
- **Tailwind CSS** 深色模式設計，體現金融科技專業感
- **SSE 流式解析** — `useAgentStream` Hook 實時解析 `agent_state` / `token` / `confidence` / `done` 四類事件
- **Agent Pipeline 可視化** — 橫向管道進度指示器，16 種 Agent 圖標映射，審查過程全程動態感知
- **三維置信度展示** — 檢索/推理/交叉驗證三維指標實時渲染，帶警告級別標記
- **react-markdown + remark-gfm** — 流式 Markdown 富文本渲染（表格、標題、引用、代碼）

### 後端 — FastAPI

- **高性能異步 API** — 基於 Starlette ASGI，天然支持長連接流式推送
- **Server-Sent Events (SSE)** — 基於 `astream_events` 的逐節點實時推送（真異步）
- **asyncio.to_thread** — 同步 LangGraph 圖在線程池中執行，避免阻塞 ASGI 事件循環
- **Pydantic V2** — 嚴格的請求/響應資料校驗，8 個核心模型（含三維置信度）
- **CORS + API Key 認證** — 可選 Bearer Token 保護，開箱即用的前後端跨域支持

### AI 編排 — LangGraph Multi-Agent

六個專業 Agent 組成的協同工作流（以 SVF 路由為例）：

| Agent | 職責 | 耗時佔比 |
|-------|------|---------|
| **Extractor Agent** 🔍 | 從自然語言中提取合規審查關鍵實體（公司名稱、牌照類型、交易模式） | ~5% |
| **Retriever Agent** 📚 | 混合檢索 HKMA 法規條款 + Cohere 重排序精排 + 語義緩存 + 跨輪文檔累積 | ~10% |
| **Analyzer Agent** 🧠 | 基於檢索結果撰寫合規風險評估報告（強制溯源、事實/建議隔離、推理置信度自評） | ~55% |
| **Format Validator** ✅ | Pydantic 結構化驗證 — 檢查報告是否符合 `AnalyzerOutput` 模型，不合規則回退重試 | ~3% |
| **Reviewer Agent** ⚖️ | 紅藍對抗審計 — 五項審查清單 + 結構化裁決（`ReviewerVerdict`）+ 交叉驗證置信度偏差 | ~22% |
| **SubQuery Planner** 🔄 | 規則化子查詢生成（零 LLM 調用）— 從 Reviewer 反饋中提取法條引用和關鍵術語，驅動二次檢索 | ~5% |

---

## RAG 引擎亮點

本項目的 RAG 管線遠超常見的「Embedding → 向量搜索 → 喂給 LLM」基礎方案，實現了四層精進：

### 第一層：混合檢索 (Hybrid Search)

```
用戶查詢 → classify_query_type() → 動態權重調整
    │
    ├─── ChromaDB Dense Retrieval ──→ Top-15 (語義模糊匹配)
    │
    ├─── BM25 Sparse Retrieval    ──→ Top-15 (關鍵詞精確命中)
    │
    └─── Reciprocal Rank Fusion (RRF) ◄── 双路融合 (動態權重)
                    │
              去重後 ~20 篇候選文檔
```

- **Dense (ChromaDB)** — 處理自然語言查詢，如「如何進行客戶盡職審查？」
- **Sparse (BM25)** — 精確命中法規條款編號，如「Section 4.2.1」、「AML」
- **動態查詢畫像** — 4 種查詢類型自動調整 BM25/Dense 權重：

| 查詢類型 | 觸發條件 | BM25 權重 | Dense 權重 |
|---------|---------|----------|-----------|
| `specific_clause` | 含 Chapter/Section 等法條引用 | 0.7 | 0.3 |
| `entity_lookup` | 含牌照號、SVF 編號等實體 | 0.8 | 0.2 |
| `risk_assessment` | 含 risk/assessment 等關鍵詞 | 0.3 | 0.7 |
| `default` | 其他一般查詢 | 0.4 | 0.6 |

- **RRF 去重修復** — 使用 `content_hash`（MD5）替代 `page_content[:200]`，避免不同文檔前綴相同導致丟失

### 第二層：雲端重排序 (Cohere Reranker)

```
~20 篇候選文檔  ──→  Cohere rerank-v3.5  ──→  Top-5 精排文檔
```

- 使用 Cohere 的 Cross-Encoder 模型逐一對 `(查詢, 文檔)` 配對打分
- 有效過濾語義漂移（相似但法律主體不同的條款）
- 每篇文檔附帶 `rerank_score`，在 LangSmith 中可追蹤精排質量
- **零本地依賴** — 純 REST API 調用，不引入 PyTorch
- **Fallback** — Cohere API 不可用時回退到純 Hybrid 結果

### 第三層：法規文檔層級感知解析

替代扁平 `CharacterTextSplitter` 的結構化解析器，支持三種模式：

| 模式 | 配置值 | 說明 |
|------|--------|------|
| **層級解析** | `hierarchy` | 完整的五層結構化切分（Document → Chapter → Section → Paragraph → Chunk），含父子關係和交叉引用預留 |
| **正則感知** | `reg_aware` | 基於法規標題正則切分，切出段太少時 fallback 到 CharacterTextSplitter |
| **扁平切分** | `flat` | 傳統 CharacterTextSplitter，作為兜底方案 |

每個 `RegulationChunk` 包含：`hierarchy_path`、`parent_id`、`children_ids`、`cross_references`（預留）、`section_title`。

### 第四層：語義緩存 (Semantic Cache)

- **Embedding + Cosine Similarity** 做查詢匹配（閾值 0.80）
- **LRU 淘汰策略**（OrderedDict, max_entries=200）
- **TTL 過期**（默認 3600 秒）
- **線程安全**（Lock）
- **PII 脫敏**後再緩存，避免敏感信息洩露
- **單次 Embedding 流程** — `get()` 返回 query_vector，`put()` 重用，避免重複計算
- 默認關閉（`SEMANTIC_CACHE_ENABLED=False`），通過環境變量啟用

### 第五層：深度防幻覺 (Anti-Hallucination Prompt Chain)

| 機制 | 規則 |
|------|------|
| **強制溯源** | 每條法規引用必須以 `[Source: Source N, p.X]` 結尾 |
| **能力邊界聲明** | 上下文不足時，必須輸出「根據所提供的文件，暫無足夠資訊以驗證此項合規要求」 |
| **事實/建議隔離** | 報告嚴格區分「法規事實摘要」與「合規建議」兩大章節 |
| **Pydantic 格式驗證** | FormatValidator 節點用 `AnalyzerOutput` 模型強制驗證報告結構，不合規則回退重試（最多 2 次） |
| **Reviewer 五項審計** | 引用完整性、幻覺檢測、事實/建議混淆、知識盲區披露、邏輯一致性 |
| **結構化裁決** | Reviewer 輸出 `ReviewerVerdict` 模型，含 `decision`/`rejection_type`/`reviewer_confidence` |

---

## 反思循環與三維置信度

### 反思循環 (Reflection Loop)

本項目的核心創新之一：當 Reviewer 發現報告質量不足時，不是直接輸出，而是觸發自糾正機制：

```
Reviewer 裁決
    │
    ├── APPROVED / 超過修訂上限 ──→ END（輸出最終報告）
    │
    ├── rejection_type = "insufficient_info"
    │   └──→ SubQuery Planner（規則化，零 LLM）
    │       └──→ Retriever（二次檢索，跨輪文檔累積去重）
    │           └──→ Analyzer（修訂報告）
    │
    └── rejection_type = "quality_issue"
        └──→ Analyzer（直接修訂，不再檢索）
```

**關鍵設計決策：**
- **SubQuery Planner 為規則化節點**（不調 LLM）— 用正則從 Reviewer 反饋提取法條引用和關鍵術語，零 Token 消耗、零延遲、零故障點
- **跨輪文檔累積去重** — 使用 `content_hash`（MD5 前 12 位）作為去重鍵，二次檢索結果與首次結果合併，避免重複
- **修訂上限保護** — `MAX_REVISIONS=2`，`MAX_RETRIEVAL=1`，防止死循環
- **格式驗證循環** — FormatValidator 檢測到結構不合規 → 回退 Analyzer 重試（最多 2 次）

### 三維置信度模型 (3D Confidence)

為報告結論提供多維度質量信號，以**只讀標記**方式呈現，不參與控制流：

| 維度 | 指標 | 來源 |
|------|------|------|
| **檢索置信度** | `retrieval_confidence`（Rerank Top-1 分數）+ `top5_gap`（Top-1 與 Top-5 均值差距） | Retriever 節點 |
| **推理置信度** | `reasoning_confidence`（Analyzer 自評，0.0-1.0） | Analyzer 輸出中的 JSON 塊 |
| **交叉驗證** | `cross_validation_passed`（retrieval vs reasoning 偏差 ≤ 閾值）+ `reviewer_confidence` | Reviewer 節點 |

**綜合置信度計算：**
```
overall = retrieval × 0.4 + reasoning × 0.6
if cross_validation_failed:
    overall -= 0.2  (懲罰)
```

**警告級別：**
- `overall < 0.4` → 🔴 **high**（報告結論可能缺乏充分法規依據）
- `overall < 0.6` → 🟡 **medium**（建議關注引用來源完整性）
- `otherwise` → 🟢 **low**（置信度正常）

---

## 深度可觀測性

全面接入 **LangSmith**，實現生產級別的 AI 系統可觀測性：

```
LangSmith Trace Tree（每次請求自動生成）

RerankedRetriever
├── HybridRetriever
│   ├── BM25Retriever      → 15 docs, 0.3s
│   └── ChromaDB Retriever → 15 docs, 1.2s
│   └── RRF Fusion         → 20 unique docs
├── Cohere Rerank          → 5 docs, 1.8s
│
Extractor Agent (LLM)      → input: 230 tokens, output: 85 tokens
Analyzer Agent (LLM)       → input: 3200 tokens, output: 2800 tokens
Format Validator            → Pydantic validation: PASS/FAIL
Reviewer Agent (LLM)       → input: 3100 tokens, output: 120 tokens
SubQuery Planner (rules)   → 2 sub-queries generated (zero LLM)
Retriever (2nd round)      → 8 new docs merged → 28 total
Analyzer Agent (revision)  → input: 4500 tokens, output: 3100 tokens
Reviewer Agent (final)     → APPROVED
```

**可追蹤指標**：Agent 節點延遲、Input/Output Token 消耗、檢索 Chunk 內容與相關性分數、Conditional Edge 決策路徑、三維置信度數值、反思循環輪次。

---

## 業務模組

| 模組 | 端點 | LLM | 反思循環 | RAG | 共享工具 |
|------|------|-----|---------|-----|---------|
| 📋 SVF 合規審查 | `/api/v1/svf/analyze/stream` | glm-4.5-air | ✅ 三路條件邊 + 二次檢索 | ✅ Hybrid RAG | ✅ |
| 🏦 銀行開戶審查 | `/api/v1/bank-account/verify/stream` | LongCat-Flash-Chat | ✅ build_review_edges | ❌ | ✅ |
| 💱 跨境匯款評估 | `/api/v1/cross-border/assess/stream` | LongCat-Flash-Chat | ✅ build_review_edges | ❌ | ✅ |
| 📈 SME 信貸評估 | `/api/v1/sme/credit-rating/stream` | LongCat-Flash-Chat | ✅ build_review_edges | ❌ | ✅ |

> 每個路由均提供阻塞式 `/analyze` 和流式 `/analyze/stream` 雙端點模式。

---

## 快速啟動

### 前置要求

- Python 3.10+
- Node.js 18+
- npm

### 1. 環境變量配置

在 `backend/` 目錄下創建 `.env` 文件：

```env
# ===== LLM API Keys =====
ZHIPU_API_KEY=your_zhipu_api_key          # 智譜 AI (glm-4.5-air + Embedding)
LONGCAT_API_KEY=your_longcat_api_key      # LongCat (LongCat-Flash-Chat)

# ===== Cohere Reranker =====
COHERE_API_KEY=your_cohere_api_key        # https://dashboard.cohere.com/api-keys

# ===== LangSmith Observability =====
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key  # https://smith.langchain.com
LANGCHAIN_PROJECT="Fintech-PoC-Backend"

# ===== Semantic Cache (可選) =====
SEMANTIC_CACHE_ENABLED=false              # 默認關閉，設為 true 啟用
SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.80  # 語義匹配閾值
SEMANTIC_CACHE_MAX_ENTRIES=200            # LRU 最大條目數
SEMANTIC_CACHE_TTL_SECONDS=3600           # 緩存過期時間（秒）

# ===== Confidence Thresholds =====
CONFIDENCE_LOW_THRESHOLD=0.5              # 檢索置信度低閾值
CONFIDENCE_MED_THRESHOLD=0.7              # 檢索置信度中閾值
CONFIDENCE_CROSS_VALIDATION_THRESHOLD=0.3 # 交叉驗證偏差閾值

# ===== Document Parser =====
PARSER_MODE=hierarchy                     # hierarchy / reg_aware / flat

# ===== API Key Auth (可選) =====
API_KEY_ENABLED=true                      # 安全默認：啟用後端 Bearer 鑑權
API_KEY=your_secret_api_key               # 必填，前端也需配置對應 token

# ===== CORS =====
CORS_ORIGINS=http://localhost:3000        # 前端地址
```

前端 `frontend/.env.local` 也應同步配置：

```env
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
NEXT_PUBLIC_API_KEY=your_secret_api_key
```

若只在本機做無鑑權原型驗證，可明確將 `API_KEY_ENABLED=false`；否則保持預設值。

### 2. 啟動後端

```bash
cd backend

# 建立虛擬環境（推薦）
python -m venv ../HKFinReg
../HKFinReg/Scripts/activate        # Windows
# source ../HKFinReg/bin/activate   # macOS/Linux

# 安裝依賴
pip install -r requirements.txt

# 啟動 FastAPI
uvicorn app.main:app --reload --port 8000
```

啟動成功後，終端應顯示：
```
✅ LangSmith tracing ENABLED for project: [Fintech-PoC-Backend]
✅ Hybrid Retriever (BM25 + Dense) initialized — Dynamic weights by query type
✅ Reranked Retriever initialized (Hybrid → Cohere rerank-v3.5 → Top-5)
✅ Semantic Cache: DISABLED (set SEMANTIC_CACHE_ENABLED=true to enable)
✅ Document Parser: hierarchy mode
```

### 3. 啟動前端

```bash
cd frontend

# 安裝依賴
npm install

# 啟動開發服務器
npm run dev
```

訪問 **http://localhost:3000** — 深色模式 Dashboard 即刻呈現。

### 4. 端到端測試（可選）

```bash
cd backend
python auto_test.py
```

此腳本會自動向 SSE 端點發送測試請求，在終端以打字機效果渲染流式輸出，並提示您前往 LangSmith 檢查 Trace 鏈路。

### 5. 單元測試（可選）

```bash
cd backend

# 語義緩存測試
python -m pytest tests/test_semantic_cache.py -v

# Extractor 分類測試
python -m pytest tests/test_extractor_classification.py -v

# 集成測試（反思循環驗證）
python -m pytest tests/test_integration.py -v
```

---

## 項目結構

```
MyFintech/
│
├── backend/                          # FastAPI 後端
│   ├── app/
│   │   ├── api/routers/              # 4 大業務路由 (SSE streaming)
│   │   │   ├── svf.py                # SVF 合規審查（6節點 LangGraph + 反思循環 + RAG）
│   │   │   ├── bank_account.py       # 銀行開戶（4節點 + 反思循環）
│   │   │   ├── cross_border.py       # 跨境匯款（4節點 + 反思循環）
│   │   │   ├── sme_lending.py        # SME 信貸（4節點 + 反思循環）
│   │   │   └── workflow_utils.py     # 共享工作流工具函數集（5個函數）
│   │   ├── core/
│   │   │   ├── config.py             # Pydantic Settings（環境變量 + 置信度閾值 + 緩存配置）
│   │   │   ├── monitoring.py         # LangSmith 追蹤初始化 + PerformanceTracker
│   │   │   └── security.py           # 可選 Bearer API Key 認證
│   │   ├── schemas/
│   │   │   └── requests.py           # Pydantic 模型（8個核心模型，含三維置信度）
│   │   ├── services/
│   │   │   ├── agents/
│   │   │   │   ├── builder.py        # LLM 工廠 + HybridRetriever + RRF融合 + 動態查詢畫像
│   │   │   │   ├── document_parser.py # 法規文檔層級感知解析器（3種模式）
│   │   │   │   ├── reranker.py       # Cohere Rerank API 封裝（含 Fallback）
│   │   │   │   └── prompts.py        # 防幻覺 Prompt 模組 + 結構化裁決格式
│   │   │   ├── semantic_cache.py     # 語義緩存（Embedding + Cosine + LRU + TTL）
│   │   │   └── utils.py              # PII 清洗、格式化工具、時間戳
│   │   └── main.py                   # FastAPI 入口（4路由掛載 + 健康檢查）
│   ├── tests/                        # 測試套件
│   │   ├── test_integration.py       # 端到端集成測試（反思循環驗證）
│   │   ├── test_semantic_cache.py    # 語義緩存測試（LRU/Embedding/PII）
│   │   └── test_extractor_classification.py  # Extractor 分類功能測試
│   ├── auto_test.py                  # 自動化 SSE 測試腳本
│   ├── requirements.txt
│   └── .env                          # 環境變量（不提交至 Git）
│
├── frontend/                         # Next.js 14 前端
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css           # 深色主題 + Markdown 排版
│   │   │   ├── layout.tsx            # 全局 Layout
│   │   │   └── page.tsx              # 主頁 Dashboard（4模組 Tab）
│   │   ├── components/
│   │   │   ├── AgentTimeline.tsx      # Agent 思考鏈路管道指示器（16種圖標）
│   │   │   └── ReportPanel.tsx        # 流式 Markdown 渲染面板 + 三維置信度展示
│   │   ├── hooks/
│   │   │   └── useAgentStream.ts      # SSE 流式解析 Hook（4類事件 + 三維置信度）
│   │   ├── lib/
│   │   │   └── modules.ts            # 業務模組配置（4大業務線）
│   │   └── types/
│   │       └── index.ts              # TypeScript 類型定義（含 ConfidenceEvent）
│   ├── .env.local                    # 前端環境變量
│   └── package.json
│
└── Fintech/                          # 原始 Streamlit 版本（歸檔）
    ├── app.py
    ├── core_logic.py
    ├── performance_monitor.py
    └── data/                         # 法規 PDF 文件
```

---

## 技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| **前端框架** | Next.js 14 (App Router) | React Server Components, 路由, SSR |
| **前端樣式** | Tailwind CSS | 深色模式 UI, 響應式設計 |
| **前端渲染** | react-markdown + remark-gfm | 流式 Markdown 報告渲染 |
| **後端框架** | FastAPI | 異步 API, SSE 推流 |
| **AI 編排** | LangGraph | 多智能體狀態機工作流 + 條件邊 |
| **LLM** | 智譜 glm-4.5-air / LongCat-Flash-Chat | 報告生成, 實體提取, 深度推理 |
| **Embedding** | 智譜 embedding-3 | 文檔向量化 + 語義緩存匹配 |
| **向量庫** | ChromaDB | Dense Retrieval |
| **關鍵詞檢索** | rank_bm25 | Sparse Retrieval |
| **重排序** | Cohere rerank-v3.5 | Cross-Encoder 精排 |
| **可觀測性** | LangSmith | 全鏈路 Trace 追蹤 |
| **配置管理** | Pydantic BaseSettings | 環境變量校驗, .env 載入 |
| **結構化輸出** | Pydantic V2 Models | AnalyzerOutput / ReviewerVerdict / ConfidenceScore |
| **語義緩存** | 自研 SemanticCache | Embedding 相似度匹配 + LRU + TTL |

---

## 授權條款

本項目僅供學術研究與技術展示用途。金融合規審查結論不構成任何法律意見。

---

<div align="center">

**Built with ❤️ for Hong Kong FinTech Compliance**

</div>
