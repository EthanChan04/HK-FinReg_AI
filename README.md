# HK-FinReg AI: Trustworthy RAG + KAG + DeepResearch Compliance System

A full-stack Hong Kong fintech regulatory analysis platform with **FastAPI + Next.js**, supporting **streaming multi-agent workflows**, **evidence-grounded retrieval**, and **human review queue** operations.

## Highlights

- Trustworthy compliance analysis with evidence references and confidence signals.
- Hybrid retrieval pipeline: BM25 + dense vector search + RRF + optional reranker.
- KAG-style knowledge graph support for regulatory reasoning paths.
- DeepResearch workflow for multi-step analysis planning, gap detection, and report drafting.
- Real-time SSE streaming from backend to frontend for agent progress and outputs.
- Human-in-the-loop review queue for escalations and continuation.

## Business Modules

| Module | Endpoint | Description |
| --- | --- | --- |
| SVF Compliance | `POST /api/v1/svf/analyze/stream` | Stored Value Facility compliance analysis |
| Bank Account | `POST /api/v1/bank-account/verify/stream` | Account opening / verification compliance checks |
| Cross-Border | `POST /api/v1/cross-border/assess/stream` | Cross-border payment risk and compliance review |
| SME Lending | `POST /api/v1/sme/credit-rating/stream` | SME credit/risk compliance workflow |
| DeepResearch | `POST /api/v1/research/analyze` | Multi-step regulatory research |
| Review Queue | `/api/v1/review-queue/` | Human review continuation workflow |

## Architecture

```text
frontend (Next.js 16 + React 19)
  |- AgentTimeline / ReportPanel / EvidencePanel / KnowledgeGraphPanel
  |- SSE + REST
  v
backend (FastAPI)
  |- api/routers/: module APIs + streaming routes
  |- services/agents/: model builder, parser, prompts, reranker
  |- services/retrieval/: routing, evidence rendering, citation verification
  |- services/kag/: graph store, graph builder, graph retriever
  |- services/deepresearch/: planner, gap detector, report writer
  |- services/corpus/: manifest loading and corpus ingestion
  |- services/evaluation/: benchmark evaluation
```

## Tech Stack

| Layer | Stack |
| --- | --- |
| Frontend | Next.js `16.2.6`, React `19.2.4`, TypeScript |
| Backend | FastAPI, Pydantic Settings, SSE |
| Workflow | LangGraph |
| Retrieval | ChromaDB, BM25, RRF |
| Graph | NetworkX |
| Rerank | Cohere (`rerank-v3.5`, optional) |
| LLM/Embedding | Zhipu GLM, LongCat, Zhipu Embedding |
| Observability | LangSmith |

## Quick Start

### 1) Backend env (`backend/.env`)

```env
ZHIPU_API_KEY=your_zhipu_api_key
LONGCAT_API_KEY=your_longcat_api_key
COHERE_API_KEY=your_cohere_api_key

ZHIPU_MODEL=glm-4.5-air
ZHIPU_EMBEDDING_MODEL=embedding-3
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LONGCAT_MODEL=LongCat-Flash-Chat
LONGCAT_BASE_URL=https://api.longcat.chat/openai/v1

API_KEY_ENABLED=true
API_KEY=your_local_api_key

REG_DOC_DIR=data/regulations
SOURCE_MANIFEST_PATH=data/source_manifest.json
CORPUS_INDEX_DIR=data/indexes
CHROMA_COLLECTION=hk_finreg_corpus

GRAPH_STORE_BACKEND=networkx
GRAPH_STORE_PATH=data/graph/regulatory_graph.json

RETRIEVAL_ROUTER_ENABLED=true
DEFAULT_RETRIEVAL_MODE=rag
DEEP_RESEARCH_ENABLED=true
```

### 2) Frontend env (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
NEXT_PUBLIC_API_KEY=your_local_api_key
```

### 3) Run services

Backend:

```powershell
cd F:\MyFintech\backend
F:\MyFintech\HKFinReg\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd F:\MyFintech\frontend
npm install
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open: `http://127.0.0.1:3000`

## Security Notes

- Real secrets (`.env`, `.env.local`, keys) are ignored by Git and should stay local.
- API key auth can be enabled with `API_KEY_ENABLED=true`.
- Citation verification and evidence checks are built into the backend retrieval workflow.
- For dependency auditing, use:

```powershell
python -m pip_audit -r backend/requirements.txt
cd frontend; npm audit --omit=dev
```

## Evaluation & Tests

Run retrieval/deepresearch benchmark:

```powershell
cd F:\MyFintech\backend
python -m app.services.evaluation.run_eval
```

Run backend tests:

```powershell
cd F:\MyFintech
python -m pytest backend\tests -q
```

Run frontend checks:

```powershell
cd F:\MyFintech\frontend
npm run lint
npm run build
```

## Repository Layout

```text
F:\MyFintech
|- backend/
|  |- app/
|  |- data/
|  |- tests/
|- frontend/
|  |- src/
|- docs/
|- SECURITY.md
|- README.md
```

## Disclaimer

This project is for research/prototyping. Outputs are not legal advice and must be reviewed by qualified compliance/legal professionals before production use.
