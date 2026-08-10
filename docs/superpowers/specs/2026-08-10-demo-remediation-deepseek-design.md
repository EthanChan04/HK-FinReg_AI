# HK-FinReg AI Demo Remediation with Explicit DeepSeek Runtime

**Date:** 2026-08-10

**Status:** Approved design direction; implementation plan pending user review.

## 1. Goal

Deliver a stable, reproducible demo release that calls a real DeepSeek V4 Flash model, fails visibly when required regulatory sources are unusable, and can be accepted from a clean Git checkout.

This is a controlled demo, not a production compliance decision system. Human approval of all 108 gold cases is not a release blocker.

## 2. Approved Scope

The remediation includes:

- an explicit DeepSeek runtime for every chat/generation path;
- repair and fail-closed validation of required demo corpus documents;
- a deterministic 108-case retrieval gate plus a 12-case live-LLM gate;
- missing demo-critical error and cancellation tests;
- clean separation of Phase 4 experiments from the demo release;
- corrected, versioned acceptance documentation.

The remediation does not include:

- human review of all gold answers;
- production Redis or PostgreSQL deployment;
- production SLA, high availability, or unattended compliance decisions;
- SPO, dual-graph, multi-hop, GraphRAG, or other Phase 4 architecture;
- removal of the two known moderate frontend advisories when doing so requires an unplanned Next.js upgrade.

## 3. Runtime Architecture

### 3.1 Explicit DeepSeek configuration

DeepSeek is the only generation provider for the demo. Configuration uses provider-specific names:

```text
DEEPSEEK_API_KEY=<secret>
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT_SECONDS=60
DEEPSEEK_INTERACTIVE_THINKING=false
DEEPSEEK_REASONING_THINKING=true
```

The API key is loaded from the environment and must never be written to logs, captured-response artifacts, test fixtures, or Git.

The official model identifier is `deepseek-v4-flash`. The runtime uses DeepSeek's OpenAI Chat Completions compatibility through the existing `langchain-openai` transport, but no public configuration or error message refers to the provider as Zhipu, LongCat, MiMo, or a generic OpenAI-compatible service.

### 3.2 Central factory

A single module owns model construction:

```python
build_deepseek_llm(profile: Literal["interactive", "reasoning", "evaluation"]) -> ChatOpenAI
```

Profiles have explicit behavior:

- `interactive`: thinking disabled for Copilot and ordinary streamed demo requests;
- `reasoning`: thinking enabled for DeepResearch and reviewer paths;
- `evaluation`: thinking disabled and temperature zero for repeatable live benchmark responses.

Existing functions such as `build_zhipu_llm`, `build_thinking_llm`, and the separate Copilot builder are replaced or retained only as temporary deprecated wrappers during migration. All active callers must resolve through the central factory before acceptance.

### 3.3 Embeddings are separate

`deepseek-v4-flash` is a chat model and must not be sent to an embeddings endpoint. The controlled demo uses:

```text
EMBEDDING_PROVIDER=local_hash
```

This keeps corpus construction reproducible and leaves DeepSeek responsible only for real generation. A future production-quality embedding provider is outside this remediation.

## 4. Corpus Integrity

The two currently unreadable HKMA files are replaced from their official manifest URLs:

- `hkma_amlcft_surveillance_capability_digitalisation_2024.pdf`
- `hkma_svf_amlcft_guideline_2023.pdf`

Corpus ingestion returns a structured result containing successfully parsed sources, skipped optional sources, and failed required sources. The manifest gains an explicit `required_for_demo` boolean for the sources needed by the demo cases.

`python -m app.services.corpus.build_cache` must exit non-zero when any required source:

- is missing;
- is not a valid PDF;
- contains no catalog/pages;
- produces zero text chunks;
- fails checksum or manifest validation.

Optional-source failures remain warnings and appear in the build summary. A successful build reports source counts and chunk counts, not only a total chunk count.

The 17 missing `effective_date` values remain a documented demo limitation. They do not block the demo, but the UI/report must not describe their effective status as human-verified.

## 5. Live LLM Evaluation

### 5.1 Two complementary gates

The release pipeline has two evaluation layers:

1. The existing 108-case deterministic retrieval/citation gate runs without network dependence.
2. A 12-case live gate retrieves evidence, sends a grounded prompt to DeepSeek V4 Flash, captures the actual answer, and feeds that answer into the existing independent generation-faithfulness path.

The 12 cases are selected by checked-in case IDs and cover:

- HKMA, SFC, PCPD, and cross-regulator questions;
- English and Traditional Chinese;
- RAG, KAG, and DeepResearch modes.

### 5.2 Acceptance rules

The live gate passes only when:

- the configured model returned by the API is `deepseek-v4-flash`;
- all 12 selected cases received non-empty responses;
- faithfulness measurement coverage is `12/12`;
- no response contains an API, timeout, parsing, or empty-output error;
- average faithfulness meets the current baseline of `0.45`;
- the selected cases' existing citation unsupported-claim rate is no greater than `0.10`.

The same LLM is not used to judge its own correctness. DeepSeek generates the answers; the checked-in deterministic claim/evidence metric performs the release calculation. This keeps the demo gate repeatable while satisfying the requirement to call a real LLM.

The response-derived hallucination rate is reported but is not a separate blocking threshold for this unreviewed demo set; it is mathematically complementary to the response faithfulness score. The `0.10` unsupported-claim threshold remains the existing citation/retrieval diagnostic, not a second name for response hallucination.

### 5.3 Artifacts and cost control

Each live run records:

- case ID;
- model ID and thinking profile;
- response text;
- evidence IDs;
- latency;
- token usage when supplied by the API;
- benchmark, corpus, and prompt fingerprints;
- per-claim faithfulness diagnostics.

Secrets and authorization headers are excluded. The latest accepted summary is checked into `docs/eval-baselines/`; raw run artifacts live in a Git-ignored artifacts directory. Retries are limited to two attempts for 429 and transient 5xx responses, using bounded exponential backoff.

## 6. Demo Error Handling

Startup readiness reports `degraded` when the DeepSeek key is missing and includes a non-secret reason. Live calls distinguish:

- 401/403: configuration/authentication failure, no retry;
- 429: bounded retry, then a user-visible capacity error;
- timeout/5xx: bounded retry, then a user-visible service error;
- malformed or empty output: fail the request and live gate;
- user cancellation: abort the upstream request and restore an actionable UI state.

No fallback model is allowed. Silently switching providers would invalidate the explicit-runtime requirement.

## 7. Test and CI Design

Backend tests cover:

- DeepSeek configuration and profile-specific request parameters;
- missing key, 401, 429, timeout, malformed output, and empty output;
- secret redaction;
- required corpus failure and optional corpus warning behavior;
- deterministic 12-case selection;
- live artifact schema and gate calculations.

Coverage enforcement is scoped to the new/changed DeepSeek, corpus-integrity, and live-evaluation modules, with a minimum line coverage of 70%. The existing full pytest suite remains required.

Frontend E2E adds:

- cancellation of an in-flight agent request;
- visible recovery from a DeepSeek/API stream error.

Human-review E2E is not required for this demo scope. Existing report/evidence and Copilot E2E remain required.

CI separates deterministic and credentialed work:

- pull requests run dependency checks, backend tests, cache/graph builds, deterministic gates, frontend coverage, E2E, and build;
- the live DeepSeek gate runs only when `DEEPSEEK_API_KEY` is available, and is mandatory for the tagged demo-acceptance workflow;
- absence of the key in ordinary forked pull requests is reported as skipped, never as passed.

## 8. Repository and Release Boundary

Implementation starts in an isolated worktree created from the accepted closeout commit. The current dirty workspace and its Phase 4 files are preserved unchanged.

The demo release includes the target evaluation report, risk assessment, implementation plan, remediation design/plan, and corrected acceptance report in version control. Debug scripts, root-level experimental package files, logs, triples, and dual-graph changes are excluded unless separately reviewed.

Final acceptance is performed from a new clean checkout, not the development worktree.

## 9. Demo Acceptance Checklist

The controlled demo is accepted when all of the following are true:

1. A clean checkout installs from lockfiles without broken requirements.
2. Every `required_for_demo` corpus source parses and produces chunks.
3. Backend full tests and the 70% targeted coverage gate pass.
4. The 108-case deterministic release gate passes.
5. The 12-case real DeepSeek gate passes with `12/12` measured responses.
6. Frontend lint, typecheck, unit coverage, six E2E cases, and production build pass.
7. `npm audit --audit-level=high` reports zero critical/high vulnerabilities; moderate findings are documented.
8. The Git worktree used for the demo tag is clean and contains no Phase 4 implementation.
9. The acceptance report states that gold packages remain pending and that this is a controlled demo.

## 10. References

- DeepSeek model list: https://api-docs.deepseek.com/api/list-models/
- DeepSeek V4 API change log: https://api-docs.deepseek.com/updates/
- DeepSeek models and pricing: https://api-docs.deepseek.com/quick_start/pricing/
