# HK-FinReg RAG/KAG/DeepResearch Upgrade

This upgrade keeps the existing SVF LangGraph workflow intact while adding:

- manifest-backed multi-document regulatory corpus
- structured `EvidenceChunk` tracing
- metadata-aware retrieval routing
- deterministic citation verification
- NetworkX-based KAG graph retrieval
- bounded DeepResearch workflow and `/api/v1/research/analyze`
- lightweight evaluation runner

## Add A Regulatory Document

1. Place the PDF under `backend/data/regulations/<category>/`.
2. Add one entry to `backend/data/source_manifest.json`.
3. Include `doc_id`, `title`, `regulator`, `doc_type`, `topics`, `module_tags`, `file_path`, and `priority`.
4. Restart the backend so cached retrievers rebuild from the manifest.

## Run Backend Tests

```bash
cd backend
python -m pytest tests/test_manifest_loader.py tests/test_evidence_schema.py tests/test_query_classifier.py tests/test_retrieval_router.py tests/test_citation_verifier.py tests/test_kag_graph_store.py tests/test_deepresearch.py -v
```

## Run DeepResearch

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_KEY>" \
  -d "{\"query\":\"請分析香港虛擬銀行推出 AI 投資顧問的合規風險，並生成上線前檢查清單。\"}"
```

## Run Evaluation

```bash
cd backend
python -m app.services.evaluation.run_eval
```
