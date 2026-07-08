# SIRA-Style Retrieval and Experience-RAG Strategy Memory

## Purpose

HK-FinReg AI now has an optional retrieval planning layer for compliance-grade RAG. It improves recall and auditability by separating three decisions:

- What regulatory terms should be searched.
- Which retrieval recipe should be used.
- What evidence quality was observed for that recipe.

The feature is disabled by default and can be enabled gradually.

## SIRA-Style Query Planning

The query planner builds a `QueryPlan` before retrieval. It keeps the dense query close to the user's scrubbed wording and expands the BM25 query with regulatory aliases such as:

- `SVF` -> `stored value facility`, `HKMA`
- `CDD` -> `customer due diligence`
- `AI` -> `GenAI`, `governance`
- product launch wording -> `product launch`

Expansion terms are filtered with corpus document-frequency rules unless they are protected regulator, product, or compliance terms.

## Experience-RAG Strategy Memory

The strategy router chooses a retrieval recipe from query traits and optional prior experience. Initial deterministic strategies are:

- `aml_kyc_balanced_rerank`
- `ai_governance_kag`
- `cross_regulator_deepresearch`
- `clause_lookup_sparse_heavy`
- `default_hybrid`

Experience memory stores only scrubbed query fingerprints and traits. It does not store raw user queries or raw PII.

## Audit Metadata

Evidence metadata can include:

- `query_plan.query_plan_id`
- `query_plan.expansion_terms`
- `query_plan.rejected_terms`
- `retrieval_strategy.strategy_id`
- `retrieval_strategy.reason_codes`
- `retrieval_strategy.memory_hit`

## Feature Flags

```env
SIRA_QUERY_PLANNER_ENABLED=False
SIRA_TERM_STATS_PATH=data/indexes/term_statistics.json
EXPERIENCE_RAG_ENABLED=False
EXPERIENCE_RAG_MEMORY_PATH=data/strategy_memory/retrieval_experiences.jsonl
EXPERIENCE_RAG_RECORDING_ENABLED=False
EXPERIENCE_RAG_MAX_RECORDS=1000
```

## Release Gates

- `strategy_accuracy >= 0.80`
- `avg_expansion_term_coverage >= 0.70`
- `query_plan_drift_rate <= 0.10`
- `avg_unsupported_claim_rate` must not increase from baseline.
- `avg_citation_supported_rate` must not decrease from baseline.

## Rollback

Disable:

```env
SIRA_QUERY_PLANNER_ENABLED=False
EXPERIENCE_RAG_ENABLED=False
```

Preserve the strategy memory file for audit unless a redaction issue is found. If redaction fails, archive and rotate the file according to the security policy.
