# HK-FinReg Research Agent: Trustworthy RAG + KAG + DeepResearch Compliance System

Hong Kong financial regulatory compliance review system with a FastAPI backend, a Next.js frontend, and multi-agent compliance workflows for SVF, bank account, cross-border, SME lending, DeepResearch, and human review queue scenarios.

## Current Status

The project currently runs as a local full-stack application:

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`
- Health check: `GET /api/v1/health`
- API authentication: Bearer token when `API_KEY_ENABLED=true`
- Main UI: four business modules with streaming agent progress and markdown report output

The SVF compliance workflow has been optimized to avoid long silent waits:

- Chroma vector index is persisted under `backend/data/indexes/chroma_hk_finreg_corpus`.
- Parsed corpus chunks are cached in `backend/data/indexes/corpus_documents.pkl`.
- The streaming API sends SSE keepalive events during graph build and long model calls.
- SVF reviewer validation uses local citation/confidence checks to avoid an extra blocking LLM call.

## Features

- Multi-agent SVF compliance review with LangGraph.
- Hybrid retrieval using BM25, Chroma dense retrieval, RRF fusion, and optional Cohere reranking.
- Metadata-backed regulatory corpus from `backend/data/source_manifest.json`.
- KAG-style regulatory graph support through NetworkX.
- DeepResearch endpoint for broader multi-step regulatory analysis.
- Citation verification and confidence scoring.
- LangSmith tracing support.
- SSE streaming from backend to frontend for agent states, confidence events, tokens, errors, and completion.

## Business Modules

| Module | Streaming Endpoint | Purpose |
| --- | --- | --- |
| SVF Compliance | `POST /api/v1/svf/analyze/stream` | Stored value facility compliance review with RAG/KAG support |
| Bank Account | `POST /api/v1/bank-account/verify/stream` | Bank account opening / verification workflow |
| Cross-Border | `POST /api/v1/cross-border/assess/stream` | Cross-border payment compliance assessment |
| SME Lending | `POST /api/v1/sme/credit-rating/stream` | SME lending and credit risk workflow |
| DeepResearch | `POST /api/v1/research/analyze` | Multi-step regulatory research |
| Review Queue | `/api/v1/review-queue/` | Human-in-the-loop workflow continuation |

## Architecture

```text
frontend/ Next.js 16 + React 19
  | EvidencePanel | KnowledgeGraphPanel | DeepResearchPlanPanel | ReportPanel
  | SSE / REST
  v
backend/ FastAPI
  |
  +-- api/routers/
  |   +-- svf.py
  |   +-- bank_account.py
  |   +-- cross_border.py
  |   +-- sme_lending.py
  |   +-- research.py
  |   +-- review_queue.py
  |
  +-- services/
      +-- agents/        LLM builders, prompts, document parser, reranker
      +-- corpus/        source manifest loading and PDF ingestion
      +-- retrieval/     query routing, evidence rendering, citation audit
      +-- kag/           graph store, graph builder, graph retriever
      +-- deepresearch/  planner, gap detector, report writer, workflow
      +-- evaluation/    benchmark runner
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js `16.2.1`, React `19.2.4`, TypeScript, Tailwind CSS |
| Backend | FastAPI, Pydantic settings, SSE streaming |
| Workflow | LangGraph |
| LLM | Zhipu GLM, LongCat |
| Embeddings | Zhipu `embedding-3` |
| Retrieval | ChromaDB, BM25, reciprocal rank fusion |
| Reranking | Cohere `rerank-v3.5` |
| Graph | NetworkX |
| Observability | LangSmith |

## Environment

Create `backend/.env`:

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

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=Fintech-PoC-Backend
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
NEXT_PUBLIC_API_KEY=your_local_api_key
```

## Run Locally

### Backend

PowerShell:

```powershell
cd F:\MyFintech\backend
F:\MyFintech\HKFinReg\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/v1/health -UseBasicParsing
```

### Frontend

PowerShell:

```powershell
cd F:\MyFintech\frontend
npm install
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000
```

## Useful Logs

When launched from Codex, local logs are written here:

```text
backend/logs/backend_8000_stdout.log
backend/logs/backend_8000_stderr.log
frontend/logs/frontend_3000_stdout.log
frontend/logs/frontend_3000_stderr.log
```

## SVF Workflow Notes

The SVF route is the most retrieval-heavy module. On a clean machine, the first run may build local caches:

```text
backend/data/indexes/chroma_hk_finreg_corpus/
backend/data/indexes/corpus_documents.pkl
```

After these files exist, subsequent starts should reach the first SVF agent quickly. If the UI appears to wait during long model calls, the backend should still emit SSE keepalive comments so the browser connection remains open.

## Running Evaluation

Run the deterministic retrieval benchmark:

```bash
cd backend
python -m app.services.evaluation.run_eval
```

This evaluates query routing accuracy, topic/regulator coverage, evidence counts, graph path counts, citation support rates, and DeepResearch gap detection against the questions in `data/evaluation/benchmark_questions.json`. See `docs/evaluation_protocol.md` for a full metric reference and how to add new benchmark questions.

## Tests

Run the targeted SVF streaming tests:

```powershell
cd F:\MyFintech
python -m pytest backend\tests\test_svf_stream_errors.py -q
```

Run backend compile checks for the recently touched modules:

```powershell
python -m py_compile backend\app\api\routers\svf.py backend\app\services\agents\builder.py
```

Run frontend checks:

```powershell
cd F:\MyFintech\frontend
npm run lint
npm run build
```

## Project Layout

```text
F:\MyFintech
|-- backend/
|   |-- app/
|   |   |-- api/routers/
|   |   |-- core/
|   |   |-- schemas/
|   |   |-- services/
|   |-- data/
|   |   |-- regulations/
|   |   |-- source_manifest.json
|   |   |-- indexes/
|   |-- tests/
|   |-- logs/
|
|-- frontend/
|   |-- src/app/
|   |-- src/components/
|   |-- src/hooks/
|   |-- src/types/
|   |-- logs/
|
|-- HKFinReg/
|-- Fintech/
```

## Adding New Regulatory Documents

1. Place the PDF in `backend/data/regulations/<category>/` (e.g., `hkma_svf/`).
2. Add a corresponding entry to `backend/data/source_manifest.json` with fields: `doc_id`, `title`, `regulator`, `doc_type`, `issue_date`, `sector`, `topics`, `module_tags`, `file_path`, `priority`, `language`.
3. Restart the backend -- the corpus is rebuilt automatically on startup.
4. The KAG regulatory graph is auto-built from the corpus on startup as well.

## Troubleshooting

### Frontend says `Failed to fetch`

Check:

- Backend is running on `127.0.0.1:8000`.
- `frontend/.env.local` points to `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000`.
- `NEXT_PUBLIC_API_KEY` matches `backend/.env` `API_KEY`.
- Browser request is not blocked by CORS.

### SVF runs for a long time

Check whether the cache files exist:

```powershell
Get-ChildItem F:\MyFintech\backend\data\indexes
```

If `corpus_documents.pkl` or `chroma_hk_finreg_corpus` is missing, the next SVF run may rebuild them and take longer.

### Embedding errors

If logs show `429`, `insufficient balance`, or provider quota errors, verify the Zhipu `embedding-3` account balance and `ZHIPU_API_KEY`.

### Duplicate backend processes

List and stop uvicorn processes:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*uvicorn*' } |
  Select-Object ProcessId, CommandLine
```

## Disclaimer

This project is for research, prototyping, and technical demonstration. Generated compliance analysis is not legal advice and should be reviewed by qualified professionals before use.
