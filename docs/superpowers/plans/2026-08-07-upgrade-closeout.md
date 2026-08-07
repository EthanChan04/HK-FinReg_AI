# HK-FinReg AI Upgrade Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a self-contained, auditable Phase 0–3 upgrade closeout while keeping Phase 4 frozen.

**Architecture:** First separate the approved closeout changes from unrelated and Phase 4 work. Then close the evaluation, benchmark-review and frontend-quality gaps with behaviour-level tests. Finish by verifying both the dirty development worktree and a clean `HEAD` archive.

**Tech Stack:** Python 3.11, FastAPI, pytest, Next.js 16, TypeScript, Vitest, Playwright, GitHub Actions.

## Global Constraints

- Preserve all unrelated user changes.
- Never mark gold answers human-reviewed without a real reviewer.
- Keep roadmap release thresholds disabled.
- Do not commit SPO, dual-graph or GraphRAG architecture work.
- Use TDD for every new production behaviour.

---

### Task 1: Make Phase 0–2 controls self-contained

**Files:**
- Create: `backend/app/core/health.py`
- Create: `backend/app/services/corpus/cache.py`
- Create: `backend/app/services/corpus/build_cache.py`
- Create: `backend/app/services/kag/build_graph_cache.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/services/agents/builder.py`
- Modify: `backend/data/source_manifest.json`
- Modify: `requirements.txt`
- Test: `backend/tests/test_risk_controls.py`
- Test: `backend/tests/test_corpus_cache_safety.py`

- [ ] Verify the clean-HEAD missing-module failure.
- [ ] Separate Phase 4 graph changes from required cache/build behaviour.
- [ ] Run focused risk-control and cache tests.
- [ ] Run backend tests and release gate.
- [ ] Commit only approved closeout files.

### Task 2: Wire actual-response faithfulness

**Files:**
- Modify: `backend/app/services/evaluation/run_eval.py`
- Modify: `backend/app/services/evaluation/release_gate.py`
- Test: `backend/tests/test_generation_faithfulness.py`
- Test: `backend/tests/test_eval_versioning.py`

**Interface:** `run_eval(response_provider: Callable[[dict, list[Document]], str] | None = None) -> dict`.

- [ ] Add a failing test proving a supplied generator response reaches faithfulness evaluation.
- [ ] Implement the minimal response-provider path.
- [ ] Add measured-row counts to provenance/gate output.
- [ ] Run focused and full backend tests.

### Task 3: Add auditable gold packages without fabricating review

**Files:**
- Create: `backend/app/services/evaluation/gold_packages.py`
- Create: `backend/data/evaluation/gold_packages/benchmark-gold-packages.json`
- Create: `docs/eval-baselines/gold-review-2026-08-07.md`
- Modify: `docs/evaluation_protocol.md`
- Test: `backend/tests/test_gold_packages.py`

- [ ] Add failing structural-validation tests.
- [ ] Implement deterministic package generation and validation.
- [ ] Generate packages for all 108 cases with `review_status=pending`.
- [ ] Document the human-review workflow and unresolved metadata.

### Task 4: Enforce meaningful frontend quality gates

**Files:**
- Modify: `frontend/e2e/smoke.spec.ts`
- Modify: `frontend/vitest.config.ts`
- Modify: `.github/workflows/release-gates.yml`
- Test: relevant frontend component/hook tests.

- [ ] Identify a deterministic core journey that needs no external credentials.
- [ ] Add failing behaviour tests for currently uncovered critical branches.
- [ ] Raise coverage to a truthful passing baseline.
- [ ] Add coverage and Playwright commands to CI.
- [ ] Run lint, typecheck, unit, coverage, E2E and build.

### Task 5: Reconcile experiment evidence and documentation

**Files:**
- Modify: `docs/experiments/2026-08-05-pea-cae-ab.md`
- Modify: `docs/experiments/2026-08-05-cdd-conflict-diagnosis.md`
- Modify: `docs/experiments/2026-08-05-ctrag-chunking-ab.md`
- Modify: `docs/upgrade-summary-2026-08-06.md`

- [ ] Re-run deterministic experiments from the finalized corpus.
- [ ] Record limitations and remove unsupported readiness claims.
- [ ] Update progress and remaining human gates.

### Task 6: Verify clean deliverability

- [ ] Run all current-worktree verification commands.
- [ ] Create a temporary archive from `HEAD` and run the same checks.
- [ ] Confirm `git status` retains only unrelated user changes.
- [ ] Commit the closeout and report exact evidence and remaining gates.
