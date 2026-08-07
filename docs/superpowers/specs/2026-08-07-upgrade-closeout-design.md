# HK-FinReg AI Upgrade Closeout Design

## Goal

Turn the 2026-08-05 to 2026-08-06 research upgrade into a self-contained,
auditable branch that satisfies the accepted Phase 0–3 controls without
starting Phase 4 architecture work.

## Scope

The closeout includes five boundaries:

1. A clean checkout must contain every module referenced by tests and CI.
2. Generation faithfulness must be evaluated from an actual generated
   response when one is supplied; unmeasured rows remain explicitly `None`.
3. The 108-case benchmark must have a machine-checkable review package and a
   transparent human-review status. Codex may prepare and validate the package,
   but must not impersonate a named compliance reviewer.
4. Frontend CI must exercise meaningful user behaviour and enforce a coverage
   baseline that the checked-in suite actually passes.
5. Experiment reports must distinguish deterministic prototype evidence from
   production evidence and must not overstate CDD or CTRAG readiness.

## Architecture and Data Flow

- Corpus ingestion writes a versioned JSON cache bound to the manifest digest
  and parser version. Runtime code never deserializes the legacy Pickle cache.
- Liveness is process-only. Readiness reports each configured local dependency
  separately and returns a degraded status when any required check fails.
- `run_eval` accepts an optional response provider. When present, each generated
  response is passed to claim-level evaluation; when absent, faithfulness is
  unmeasured rather than silently passed.
- Gold review packages are generated deterministically from benchmark data and
  store decision, witness trace, clause set, review status, reviewer and review
  date. Release automation validates structure while human approval remains an
  explicit external gate.
- Browser E2E covers the repository's real interactive flow that can run
  deterministically without external LLM credentials. CI runs unit tests,
  coverage and Playwright.

## Safety and Non-goals

- Do not commit or activate current experimental SPO/dual-graph files.
- Do not mark human review complete without a real reviewer identity and date.
- Do not tighten roadmap thresholds to 0.90/0.75/0.95/0.05 in this closeout.
- Preserve unrelated frontend styling, local logs and debugging scripts.
- Do not claim production readiness; this remains a research/prototype branch.

## Verification

Completion requires both the current worktree and a fresh archive of `HEAD` to
pass backend tests, release gates, frontend lint/typecheck/unit/coverage/E2E,
build, dependency checks and high-severity security audit. The final summary
must report any remaining human or infrastructure-dependent work.
