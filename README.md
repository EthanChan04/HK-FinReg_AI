<div align="center">

# 🏛️ HK-FinReg AI

### 香港金融科技合規 AI 平台

**多智能體 (Multi-Agent) · 混合檢索 (Hybrid RAG) · 深度防幻覺 · 全鏈路可觀測**

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
| **嚴謹性** | 強制溯源機制 — 報告中每條法規引用必須標注 `[Source: Source N, p.X]`，精確到頁碼 |
| **抗幻覺** | 三重防線 — ① Hybrid RAG 精準檢索 ② Cohere Reranker 精排 ③ Reviewer Agent 紅藍對抗審計 |
| **可觀測** | LangSmith 全鏈路追蹤 — 每個 Agent 節點、Token 消耗、檢索質量一覽無遺 |
| **抗焦慮** | SSE 流式推送 + Agent 思考鏈路動態可視化 — 200 秒的審查過程全程可感知 |

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 14)                    │
│  App Router · Tailwind CSS · react-markdown · Dark Mode         │
│                                                                 │
│  ┌──────────┐  ┌─────────────────┐  ┌───────────────────┐      │
│  │ 模組選擇  │  │ Agent Pipeline  │  │  Markdown Report  │      │
│  │ 4-Tab Nav │  │ 思考鏈路指示器   │  │  流式富文本渲染    │      │
│  └──────────┘  └─────────────────┘  └───────────────────┘      │
│                         ▲ SSE (text/event-stream)               │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│                  Backend (FastAPI)                               │
│                         │                                       │
│  ┌──────────────────────▼──────────────────────────┐            │
│  │           LangGraph Multi-Agent Workflow          │            │
│  │                                                   │            │
│  │  ┌───────────┐    ┌────────────┐    ┌──────────┐ │            │
│  │  │ Extractor  │───▶│  Retriever │───▶│ Analyzer │ │            │
│  │  │   Agent    │    │   Agent    │    │  Agent   │ │            │
│  │  └───────────┘    └────────────┘    └─────┬────┘ │            │
│  │                         │                 │      │            │
│  │                    ┌────▼────┐       ┌────▼────┐ │            │
│  │                    │ Hybrid  │       │Reviewer │ │            │
│  │                    │  RAG    │       │ Agent   │ │            │
│  │                    │ Engine  │       │(紅藍對抗)│ │            │
│  │                    └─────────┘       └─────────┘ │            │
│  └──────────────────────────────────────────────────┘            │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐      │
│  │ ChromaDB     │  │ BM25 Index   │  │ Cohere Reranker   │      │
│  │ (Dense)      │  │ (Sparse)     │  │ (Cloud API)       │      │
│  └──────────────┘  └──────────────┘  └───────────────────┘      │
│                                                                  │
│              LangSmith ← 全鏈路 Trace 追蹤                       │
└──────────────────────────────────────────────────────────────────┘
```

### 前端 — Next.js 14

- **App Router** + React Server Components
- **Tailwind CSS** 深色模式設計，體現金融科技專業感
- **SSE 流式解析** — `useAgentStream` Hook 實時解析 `agent_state` / `token` / `done` 三類事件
- **Agent Pipeline 可視化** — 橫向管道進度指示器，200 秒審查過程全程動態感知
- **react-markdown + remark-gfm** — 流式 Markdown 富文本渲染（表格、標題、引用、代碼）

### 後端 — FastAPI

- **高性能異步 API** — 基於 Starlette ASGI，天然支持長連接流式推送
- **Server-Sent Events (SSE)** — 即時推送 Agent 執行狀態與報告 Token
- **Pydantic V2** — 嚴格的請求/響應資料校驗
- **CORS 預配置** — 開箱即用的前後端跨域支持

### AI 編排 — LangGraph Multi-Agent

四個專業 Agent 組成的協同工作流：

| Agent | 職責 | 耗時佔比 |
|-------|------|---------|
| **Extractor Agent** 🔍 | 從自然語言中提取合規審查關鍵實體（公司名稱、牌照類型、交易模式） | ~5% |
| **Retriever Agent** 📚 | 混合檢索 HKMA 法規條款 + Cohere 重排序精排 | ~10% |
| **Analyzer Agent** 🧠 | 基於檢索結果撰寫合規風險評估報告（強制溯源、事實/建議隔離） | ~60% |
| **Reviewer Agent** ⚖️ | 紅藍對抗審計 — 五項審查清單驗證引用完整性、幻覺檢測、邏輯一致性 | ~25% |

---

## RAG 引擎亮點

本項目的 RAG 管線遠超常見的「Embedding → 向量搜索 → 喂給 LLM」基礎方案，實現了三層精進：

### 第一層：混合檢索 (Hybrid Search)

```
用戶查詢
    │
    ├─── ChromaDB Dense Retrieval ──→ Top-15 (語義模糊匹配)
    │
    ├─── BM25 Sparse Retrieval    ──→ Top-15 (關鍵詞精確命中)
    │
    └─── Reciprocal Rank Fusion (RRF) ◄── 双路融合
                    │
              去重後 ~20 篇候選文檔
```

- **Dense (ChromaDB)** — 處理自然語言查詢，如「如何進行客戶盡職審查？」
- **Sparse (BM25)** — 精確命中法規條款編號，如「Section 4.2.1」、「AML」
- **RRF 權重** — BM25 = 0.4 / Dense = 0.6（金融合規場景實測最佳配比）

### 第二層：雲端重排序 (Cohere Reranker)

```
~20 篇候選文檔  ──→  Cohere rerank-v3.5  ──→  Top-5 精排文檔
```

- 使用 Cohere 的 Cross-Encoder 模型逐一對 `(查詢, 文檔)` 配對打分
- 有效過濾語義漂移（相似但法律主體不同的條款）
- 每篇文檔附帶 `relevance_score`，在 LangSmith 中可追蹤精排質量
- **零本地依賴** — 純 REST API 調用，不引入 PyTorch

### 第三層：深度防幻覺 (Anti-Hallucination Prompt Chain)

| 機制 | 規則 |
|------|------|
| **強制溯源** | 每條法規引用必須以 `[Source: Source N, p.X]` 結尾 |
| **能力邊界聲明** | 上下文不足時，必須輸出「根據所提供的文件，暫無足夠資訊以驗證此項合規要求」 |
| **事實/建議隔離** | 報告嚴格區分「法規事實摘要」與「合規建議」兩大章節 |
| **Reviewer 五項審計** | 引用完整性、幻覺檢測、事實/建議混淆、知識盲區披露、邏輯一致性 |

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
Reviewer Agent (LLM)       → input: 3100 tokens, output: 120 tokens
```

**可追蹤指標**：Agent 節點延遲、Input/Output Token 消耗、檢索 Chunk 內容與相關性分數、Conditional Edge 決策路徑。

---

## 業務模組

| 模組 | 端點 | 適用場景 |
|------|------|---------|
| 📋 SVF 合規審查 | `/api/v1/svf/analyze/stream` | 儲值支付工具牌照合規風險評估 |
| 🏦 銀行開戶審查 | `/api/v1/bank-account/verify/stream` | KYC/CDD 合規性驗證 |
| 💱 跨境匯款評估 | `/api/v1/cross-border/assess/stream` | AML/CTF 跨境交易風險篩查 |
| 📈 SME 信貸評估 | `/api/v1/sme/credit-rating/stream` | 中小企業貸款信用風險量化 |

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
ZHIPU_API_KEY=your_zhipu_api_key          # 智譜 AI (GLM-4-Flash + Embedding)
LONGCAT_API_KEY=your_longcat_api_key      # LongCat (深度推理模型)

# ===== Cohere Reranker =====
COHERE_API_KEY=your_cohere_api_key        # https://dashboard.cohere.com/api-keys

# ===== LangSmith Observability =====
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key  # https://smith.langchain.com
LANGCHAIN_PROJECT="Fintech-PoC-Backend"
```

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
✅ Hybrid Retriever (BM25 + Dense) initialized — RRF weights [0.4, 0.6]
✅ Reranked Retriever initialized (Hybrid → Cohere rerank-v3.5 → Top-5)
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

---

## 項目結構

```
MyFintech/
│
├── backend/                          # FastAPI 後端
│   ├── app/
│   │   ├── api/routers/              # 4 大業務路由 (SSE streaming)
│   │   │   ├── svf.py                # SVF 合規審查
│   │   │   ├── bank_account.py       # 銀行開戶
│   │   │   ├── cross_border.py       # 跨境匯款
│   │   │   └── sme_lending.py        # SME 信貸
│   │   ├── core/
│   │   │   ├── config.py             # Pydantic Settings (環境變量)
│   │   │   └── monitoring.py         # LangSmith 追蹤初始化
│   │   ├── schemas/
│   │   │   └── requests.py           # Pydantic 請求/響應模型
│   │   ├── services/
│   │   │   ├── agents/
│   │   │   │   ├── builder.py        # LLM 工廠 + HybridRetriever + RerankedRetriever
│   │   │   │   ├── reranker.py       # Cohere Rerank API 封裝
│   │   │   │   └── prompts.py        # 防幻覺 Prompt 模組
│   │   │   └── utils.py              # PII 清洗、格式化工具
│   │   └── main.py                   # FastAPI 入口
│   ├── auto_test.py                  # 自動化 SSE 測試腳本
│   ├── requirements.txt
│   └── .env                          # 環境變量（不提交至 Git）
│
├── frontend/                         # Next.js 14 前端
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css           # 深色主題 + Markdown 排版
│   │   │   ├── layout.tsx            # 全局 Layout
│   │   │   └── page.tsx              # 主頁 Dashboard
│   │   ├── components/
│   │   │   ├── AgentTimeline.tsx      # Agent 思考鏈路管道指示器
│   │   │   └── ReportPanel.tsx        # 流式 Markdown 渲染面板
│   │   ├── hooks/
│   │   │   └── useAgentStream.ts      # SSE 流式解析 Hook
│   │   ├── lib/
│   │   │   └── modules.ts            # 業務模組配置
│   │   └── types/
│   │       └── index.ts              # TypeScript 類型定義
│   ├── .env.local                    # 前端環境變量
│   └── package.json
│
└── Fintech/                          # 原始 Streamlit 版本（歸檔）
    ├── app.py
    ├── core_logic.py
    └── performance_monitor.py
```

---

## 技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| **前端框架** | Next.js 14 (App Router) | React Server Components, 路由, SSR |
| **前端樣式** | Tailwind CSS | 深色模式 UI, 響應式設計 |
| **前端渲染** | react-markdown + remark-gfm | 流式 Markdown 報告渲染 |
| **後端框架** | FastAPI | 異步 API, SSE 推流 |
| **AI 編排** | LangGraph | 多智能體狀態機工作流 |
| **LLM** | 智譜 GLM-4-Flash / LongCat-Thinking | 報告生成, 實體提取 |
| **Embedding** | 智譜 embedding-3 | 文檔向量化 |
| **向量庫** | ChromaDB | Dense Retrieval |
| **關鍵詞檢索** | rank_bm25 | Sparse Retrieval |
| **重排序** | Cohere rerank-v3.5 | Cross-Encoder 精排 |
| **可觀測性** | LangSmith | 全鏈路 Trace 追蹤 |
| **配置管理** | Pydantic BaseSettings | 環境變量校驗, .env 載入 |

---

## 授權條款

本項目僅供學術研究與技術展示用途。金融合規審查結論不構成任何法律意見。

---

<div align="center">

**Built with ❤️ for Hong Kong FinTech Compliance**

</div>
