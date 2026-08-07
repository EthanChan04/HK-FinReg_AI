# Golden-set human review ledger — 2026-08-07

## Status

- Benchmark cases: **108**
- Structurally valid gold packages: **108**
- Human-approved packages: **0**
- Review state: **pending compliance-domain review**

Codex generated the decision-package structure from the checked-in benchmark
but did not mark any answer as human-approved. Approval requires a real reviewer
name, review date, evidence identifiers, clause identifiers and revision notes.

## Review batches

| Batch | Cases | Status | Reviewer | Reviewed at |
|---|---:|---|---|---|
| SFC | 32 | Pending | — | — |
| HKMA | 28 | Pending | — | — |
| HKMA + SFC + PCPD | 22 | Pending | — | — |
| PCPD | 18 | Pending | — | — |
| HKMA + SFC | 5 | Pending | — | — |
| HKMA + PCPD | 3 | Pending | — | — |

Language distribution is en 90 / zh-Hant 18. Retrieval-mode distribution is
rag 55 / kag 39 / deep_research 14.

## Required review procedure

For each package in
`backend/data/evaluation/gold_packages/benchmark-gold-packages.json`:

1. Verify every `decision` claim against the cited official document.
2. Populate `evidence_ids`, `clause_ids` and the package-level `clause_set`.
3. Record corrections in `revision_history`.
4. Set `review.status` to `approved` or `rejected` and record the real reviewer
   identity and ISO-8601 review date.
5. Re-run `python -m app.services.evaluation.gold_packages` only to rebuild
   pending packages; do not overwrite reviewed packages.

The roadmap thresholds must not be enabled until all 108 packages are approved
and the captured-response faithfulness run has been reviewed.
