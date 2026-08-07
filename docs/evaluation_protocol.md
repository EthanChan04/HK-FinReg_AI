# HK-FinReg Evaluation Protocol

## Running the Benchmark

```bash
cd backend
python -m app.services.evaluation.run_eval
python -m app.services.evaluation.run_eval --captured-responses path/to/actual-responses.json
python -m app.services.corpus.build_cache
python -m app.services.kag.build_graph_cache
python -m app.services.evaluation.gold_packages
```

The default run is a deterministic, real-retrieval benchmark against the questions in `data/evaluation/benchmark_questions.json`. Generation faithfulness is measured only when `--captured-responses` supplies a JSON object mapping benchmark case IDs to actual generator response text. Missing responses remain explicitly unmeasured.

## Metrics

| Metric | What it measures |
|---|---|
| `retrieval_mode_accuracy` | Fraction of questions where the query router selected the expected mode (rag / kag / deep_research). |
| `topic_coverage` | Fraction of `expected_topics` that appear in the classifier's filter output. |
| `classifier_regulator_coverage` | Fraction of `expected_regulators` that appear in the classifier's filter output. |
| `evidence_regulator_coverage` | Fraction of `expected_regulators` represented by retrieved evidence metadata. |
| `regulator_coverage` | Backward-compatible alias for `classifier_regulator_coverage` during the metric transition. |
| `strategy_accuracy` | Fraction of questions where the Experience-RAG strategy router selected `expected_strategy_id`. |
| `expansion_term_coverage` | Fraction of `expected_expansion_terms` produced by the SIRA-style query planner. |
| `query_plan_drift_rate` | Fraction of questions where the dense query differs from the scrubbed original query; should stay low to avoid semantic drift. |
| `evidence_count` | Average number of evidence chunks retrieved per question (real Chroma/BM25 retrieval). |
| `graph_path_count` | Average number of KAG graph paths returned per question (real NetworkX traversal). |
| `citation_supported_rate` | Fraction of citations that match the retrieved evidence (via `citation_verifier`). |
| `unsupported_claim_rate` | Fraction of citations flagged as unsupported by the citation audit. |
| `deepresearch_gap_count` | Average number of evidence gaps found by DeepResearch's `evidence_evaluator` (only for `deep_research` mode questions). |
| `claim_recall` | Fraction of expected claims supported by at least one retrieved evidence chunk. **Retrieval-quality metric.** |
| `context_precision` | Fraction of retrieved chunks that support at least one expected claim. **Retrieval-quality metric.** |
| `faithfulness` | **Generation-faithfulness metric (measured independently since 2026-08-05).** Fraction of claims in the *generator's actual response* supported by the retrieved context. `None` when no generator response was evaluated — never silently equal to `claim_recall`. |
| `hallucination_rate` | One minus generation faithfulness. `None` when faithfulness was not measured. |
| `noise_sensitivity` | Loss in claim support when unrelated evidence is introduced. **Retrieval-quality metric.** |
| `context_utilization` | Alias for deterministic claim support used to report generator use of retrieved context. |
| `claim_diagnostics` | Per-claim support result, evidence indexes, and deterministic evaluation reason. |

Metrics are grouped in three tiers (see report section 3.2): **retrieval quality** (`claim_recall`, `context_precision`, `noise_sensitivity`), **generation faithfulness** (`faithfulness`, `hallucination_rate` — only meaningful once a generator response is evaluated), and **citation correctness** (`citation_supported_rate`, `unsupported_claim_rate`). The release gate only blocks on retrieval-quality floors until generation faithfulness is independently measured on a real generator response.

## Adding New Benchmark Questions

Edit `data/evaluation/benchmark_questions.json` and add an object:

```json
{
  "id": "RAG_SVF_AML_002",
  "question": "What are the transaction monitoring requirements?",
  "expected_retrieval_mode": "rag",
  "expected_strategy_id": "aml_kyc_balanced_rerank",
  "expected_expansion_terms": ["AML", "CDD", "HKMA"],
  "expected_topics": ["AML", "transaction_monitoring", "SVF"],
  "expected_regulators": ["HKMA"],
  "expected_claims": ["An SVF licensee must perform customer due diligence."],
  "language": "en",
  "task_type": "obligation_extraction",
  "noise_documents": []
}
```

Accepted `expected_retrieval_mode` values: `rag`, `kag`, `deep_research`.

Re-run `python -m app.services.evaluation.run_eval` -- the new question is picked up automatically.

## Interpreting Results

- **retrieval_mode_accuracy < 1.0**: the query classifier is routing to the wrong mode. Check classifier rules.
- **topic_coverage < 1.0**: expected topic tags are missing from the classifier or the source documents.
- **classifier_regulator_coverage < 1.0**: expected regulators are missing from classifier filters; check query classification and planning rules.
- **evidence_regulator_coverage < classifier_regulator_coverage**: retrieved evidence lacks the regulator diversity implied by the query; inspect corpus metadata and DeepResearch diversity gates.
- **strategy_accuracy < 0.8**: strategy routing is not selecting the expected retrieval recipe for benchmark cases.
- **expansion_term_coverage < 0.7**: SIRA-style query planning is missing expected regulatory aliases or discriminative terms.
- **query_plan_drift_rate > 0.1**: dense queries are being rewritten too aggressively; keep dense retrieval close to the user's original wording.
- **citation_supported_rate < 0.5**: the retriever is returning chunks that don't match the claims the report makes.
- **unsupported_claim_rate > 0.3**: the system is generating claims not grounded in retrieved evidence.
- **deepresearch_gap_count > 2**: sub-questions are not producing enough evidence -- consider broadening retrieval or adding sources.

The checked-in benchmark contains 108 auditable cases across HKMA, SFC, PCPD, cross-regulatory, English/Traditional Chinese, validity-conflict and refusal scenarios. The matching decision packages live in `data/evaluation/gold_packages/benchmark-gold-packages.json`; their current human-review status is tracked in `docs/eval-baselines/gold-review-2026-08-07.md`. The release gate requires complete provenance metadata, 100% evidence regulator coverage for the AI-advisor anchor cases, and the calibrated baseline floors (`claim_recall >= 0.45`, `context_precision >= 0.15`, measured `faithfulness >= 0.45`, `unsupported_claim_rate <= 0.1`). The recommended tightening targets are `0.90`, `0.75`, `0.95`, and `0.05` respectively only after all gold packages are human-approved.
