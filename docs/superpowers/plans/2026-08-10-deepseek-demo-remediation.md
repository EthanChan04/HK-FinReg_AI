# DeepSeek Demo Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a clean, stable demo release that uses real `deepseek-v4-flash` generation, fails closed on required corpus damage, and passes deterministic plus 12-case live acceptance gates.

**Architecture:** Centralize all chat-model construction in one explicit DeepSeek factory while keeping local deterministic embeddings separate. Make corpus ingestion return an auditable result and block cache builds on required-source failures. Add a credentialed live-evaluation layer above the existing deterministic 108-case gate, then close demo-critical frontend and CI gaps.

**Tech Stack:** Python 3.11, FastAPI, LangChain `ChatOpenAI` transport, DeepSeek Chat Completions, Pydantic, pytest/pytest-cov, Next.js 16, Vitest, Playwright, GitHub Actions.

## Global Constraints

- Implement in an isolated Git worktree created from the approved design commit; preserve the dirty `F:\MyFintech` workspace unchanged.
- Use only `https://api.deepseek.com` and model ID `deepseek-v4-flash` for chat/generation.
- Never log, serialize, commit, or echo `DEEPSEEK_API_KEY` or authorization headers.
- Use `EMBEDDING_PROVIDER=local_hash`; never call an embeddings endpoint with `deepseek-v4-flash`.
- Gold packages may remain `pending`; human review is not a demo blocker.
- Real Redis/PostgreSQL, production SLA, and Phase 4 graph work are out of scope.
- Keep 0 critical/high npm vulnerabilities; document moderate findings without forcing an unplanned Next.js upgrade.
- Use TDD for each production behavior and commit after every independently testable task.

---

### Task 1: Add the explicit DeepSeek runtime and migrate callers

**Files:**
- Create: `backend/app/services/llm/__init__.py`
- Create: `backend/app/services/llm/deepseek.py`
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/services/agents/builder.py`
- Modify: `backend/app/services/copilot/model.py`
- Modify: `backend/app/main.py`
- Modify: `backend/.env.example`
- Test: `backend/tests/test_deepseek_runtime.py`
- Test: `backend/tests/test_embedding_fallback.py`

**Interfaces:**
- Produces: `build_deepseek_llm(profile: Literal["interactive", "reasoning", "evaluation"]) -> ChatOpenAI`.
- Produces: `deepseek_runtime_status() -> dict[str, str | bool]` with `configured`, `provider`, `model`, and non-secret `reason`.
- Consumes: `Settings.DEEPSEEK_*` fields and the existing LangChain `ChatOpenAI` dependency.

- [ ] **Step 1: Write failing runtime tests**

```python
def test_interactive_profile_uses_v4_flash_without_thinking(monkeypatch):
    settings = SimpleNamespace(
        DEEPSEEK_API_KEY="secret",
        DEEPSEEK_BASE_URL="https://api.deepseek.com",
        DEEPSEEK_MODEL="deepseek-v4-flash",
        DEEPSEEK_TIMEOUT_SECONDS=60,
        DEEPSEEK_INTERACTIVE_THINKING=False,
        DEEPSEEK_REASONING_THINKING=True,
    )
    monkeypatch.setattr(deepseek, "get_settings", lambda: settings)
    monkeypatch.setattr(deepseek, "ChatOpenAI", CapturingChatOpenAI)
    deepseek.build_deepseek_llm.cache_clear()

    client = deepseek.build_deepseek_llm("interactive")

    assert client.kwargs["model"] == "deepseek-v4-flash"
    assert client.kwargs["base_url"] == "https://api.deepseek.com"
    assert client.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_missing_key_fails_without_exposing_secret(monkeypatch):
    monkeypatch.setattr(deepseek, "get_settings", lambda: settings_with_key(""))
    deepseek.build_deepseek_llm.cache_clear()
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is required"):
        deepseek.build_deepseek_llm("interactive")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd backend && ..\.venv\Scripts\python.exe -m pytest tests/test_deepseek_runtime.py tests/test_embedding_fallback.py -v`

Expected: FAIL because `app.services.llm.deepseek` and `DEEPSEEK_*` settings do not exist.

- [ ] **Step 3: Implement settings and the central factory**

```python
DeepSeekProfile = Literal["interactive", "reasoning", "evaluation"]


@lru_cache(maxsize=3)
def build_deepseek_llm(profile: DeepSeekProfile) -> ChatOpenAI:
    settings = get_settings()
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required for real LLM requests")
    thinking = (
        settings.DEEPSEEK_REASONING_THINKING
        if profile == "reasoning"
        else settings.DEEPSEEK_INTERACTIVE_THINKING
    )
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        temperature=0,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.DEEPSEEK_TIMEOUT_SECONDS,
        max_retries=0,
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )


def deepseek_runtime_status() -> dict[str, str | bool]:
    settings = get_settings()
    configured = bool(settings.DEEPSEEK_API_KEY)
    return {
        "configured": configured,
        "provider": "deepseek",
        "model": settings.DEEPSEEK_MODEL,
        "reason": "configured" if configured else "DEEPSEEK_API_KEY is missing",
    }
```

Add exact defaults to `Settings`: base URL `https://api.deepseek.com`, model `deepseek-v4-flash`, timeout `60`, interactive thinking `False`, reasoning thinking `True`. Change the embedding default to `local_hash`.

- [ ] **Step 4: Migrate active callers**

```python
def build_zhipu_llm() -> ChatOpenAI:
    warnings.warn("build_zhipu_llm is deprecated; use build_deepseek_llm", DeprecationWarning)
    return build_deepseek_llm("interactive")


def build_thinking_llm() -> ChatOpenAI:
    return build_deepseek_llm("reasoning")


def build_copilot_llm() -> ChatOpenAI:
    return build_deepseek_llm("interactive")
```

Update startup/readiness configuration checks to use only `DEEPSEEK_API_KEY`, and replace the MiMo/Zhipu/LongCat blocks in `.env.example` with the approved DeepSeek variables.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `cd backend && ..\.venv\Scripts\python.exe -m pytest tests/test_deepseek_runtime.py tests/test_embedding_fallback.py tests/test_copilot_response_writer.py tests/test_svf_stream_errors.py -v`

Expected: all selected tests PASS; no test output contains the sample API key.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/llm backend/app/core/config.py backend/app/services/agents/builder.py backend/app/services/copilot/model.py backend/app/main.py backend/.env.example backend/tests/test_deepseek_runtime.py backend/tests/test_embedding_fallback.py
git commit -m "feat(llm): add explicit DeepSeek V4 Flash runtime"
```

---

### Task 2: Make required corpus ingestion fail closed

**Files:**
- Modify: `backend/app/schemas/corpus.py`
- Modify: `backend/app/services/corpus/corpus_ingestor.py`
- Modify: `backend/app/services/corpus/build_cache.py`
- Modify: `backend/data/source_manifest.json`
- Create: `backend/tests/test_corpus_ingestion_report.py`
- Modify: `backend/tests/test_corpus_cache_safety.py`

**Interfaces:**
- Produces: `CorpusIngestionFailure(doc_id: str, path: str, required: bool, error_type: str, message: str)`.
- Produces: `CorpusIngestionResult(documents, loaded_source_ids, failures)` with `required_failures` property.
- Produces: `ingest_corpus_documents(...) -> CorpusIngestionResult`.
- Preserves: `load_corpus_documents(...) -> list[Document]` as a compatibility wrapper returning `.documents`.

- [ ] **Step 1: Write failing ingestion-result tests**

```python
def test_required_pdf_failure_is_reported(monkeypatch, tmp_path):
    source = source_document(tmp_path / "broken.pdf", required_for_demo=True)
    monkeypatch.setattr(corpus_ingestor, "load_source_manifest", lambda **_: [source])
    monkeypatch.setattr(corpus_ingestor, "_load_source_pages", broken_pdf)

    result = corpus_ingestor.ingest_corpus_documents()

    assert result.documents == []
    assert [failure.doc_id for failure in result.required_failures] == [source.doc_id]


def test_zero_chunk_required_pdf_is_a_failure(monkeypatch, tmp_path):
    source = source_document(tmp_path / "empty.pdf", required_for_demo=True)
    monkeypatch.setattr(corpus_ingestor, "load_source_manifest", lambda **_: [source])
    monkeypatch.setattr(corpus_ingestor, "_load_source_pages", lambda _: [])
    assert corpus_ingestor.ingest_corpus_documents().required_failures[0].error_type == "EmptyDocument"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd backend && ..\.venv\Scripts\python.exe -m pytest tests/test_corpus_ingestion_report.py -v`

Expected: FAIL because structured ingestion results do not exist.

- [ ] **Step 3: Implement result types and fail-closed build behavior**

```python
@dataclass(frozen=True)
class CorpusIngestionFailure:
    doc_id: str
    path: str
    required: bool
    error_type: str
    message: str


@dataclass
class CorpusIngestionResult:
    documents: list[Document]
    loaded_source_ids: list[str]
    failures: list[CorpusIngestionFailure]

    @property
    def required_failures(self) -> list[CorpusIngestionFailure]:
        return [failure for failure in self.failures if failure.required]
```

Add `required_for_demo: bool = False` to `SourceDocument`. Mark every checked-in demo source explicitly in `source_manifest.json`; required sources must be the full 20-document checked-in demo corpus.

In `build_cache.main()`, call `ingest_corpus_documents()`, raise `RuntimeError` listing required doc IDs before writing a cache when `required_failures` is non-empty, and print `sources_loaded`, `sources_failed`, and `chunks` on success.

- [ ] **Step 4: Verify RED-to-GREEN behavior at the command boundary**

Add a test that patches `ingest_corpus_documents` to return one required failure and asserts `build_cache.main()` raises before `write_corpus_cache` is called.

Run: `cd backend && ..\.venv\Scripts\python.exe -m pytest tests/test_corpus_ingestion_report.py tests/test_corpus_cache_safety.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas/corpus.py backend/app/services/corpus/corpus_ingestor.py backend/app/services/corpus/build_cache.py backend/data/source_manifest.json backend/tests/test_corpus_ingestion_report.py backend/tests/test_corpus_cache_safety.py
git commit -m "fix(corpus): fail closed on required demo sources"
```

---

### Task 3: Safely refresh and validate the two damaged HKMA PDFs

**Files:**
- Create: `backend/scripts/refresh_required_sources.py`
- Create: `backend/tests/test_refresh_required_sources.py`
- Replace: `backend/data/regulations/hkma_aml_ai/hkma_amlcft_surveillance_capability_digitalisation_2024.pdf`
- Replace: `backend/data/regulations/hkma_svf/hkma_svf_amlcft_guideline_2023.pdf`

**Interfaces:**
- Produces: `refresh_source(source: SourceDocument, destination_root: Path, opener=urlopen) -> Path`.
- Validates: HTTPS official-domain URL, `%PDF-` header, readable catalog, at least one page, and non-empty extracted text before atomic replacement.

- [ ] **Step 1: Write failing safe-refresh tests**

```python
def test_invalid_download_never_replaces_existing_file(tmp_path):
    destination = tmp_path / "source.pdf"
    destination.write_bytes(b"original")
    with pytest.raises(ValueError, match="valid PDF"):
        refresh_source(source_for(destination), tmp_path, opener=lambda _: BytesIO(b"html"))
    assert destination.read_bytes() == b"original"


def test_valid_download_is_atomically_installed(tmp_path, valid_pdf_bytes):
    destination = refresh_source(
        source_for(tmp_path / "source.pdf"),
        tmp_path,
        opener=lambda _: BytesIO(valid_pdf_bytes),
    )
    assert PdfReader(str(destination)).pages
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd backend && ..\.venv\Scripts\python.exe -m pytest tests/test_refresh_required_sources.py -v`

Expected: FAIL because the refresh script does not exist.

- [ ] **Step 3: Implement download-to-temp and atomic replace**

```python
with NamedTemporaryFile(dir=destination.parent, suffix=".pdf", delete=False) as handle:
    handle.write(downloaded_bytes)
    temporary = Path(handle.name)
try:
    reader = PdfReader(str(temporary))
    if not reader.pages or not any((page.extract_text() or "").strip() for page in reader.pages):
        raise ValueError(f"{source.doc_id} did not contain readable PDF text")
    temporary.replace(destination)
finally:
    temporary.unlink(missing_ok=True)
```

Use `validate_source_metadata` before network access and never print response headers containing credentials.

- [ ] **Step 4: Run unit tests and refresh exactly the two failed sources**

Run:

```powershell
cd backend
..\.venv\Scripts\python.exe scripts/refresh_required_sources.py --doc-id hkma_amlcft_surveillance_capability_digitalisation_2024 --doc-id hkma_svf_amlcft_guideline_2023
..\.venv\Scripts\python.exe -m app.services.corpus.build_cache
```

Expected: both files validate, cache output reports 20 loaded sources, 0 required failures, and no `Cannot find Root object` warning.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/refresh_required_sources.py backend/tests/test_refresh_required_sources.py backend/data/regulations/hkma_aml_ai/hkma_amlcft_surveillance_capability_digitalisation_2024.pdf backend/data/regulations/hkma_svf/hkma_svf_amlcft_guideline_2023.pdf
git commit -m "fix(corpus): replace damaged HKMA demo sources"
```

---

### Task 4: Capture 12 real DeepSeek responses with auditable artifacts

**Files:**
- Create: `backend/app/services/evaluation/live_demo_eval.py`
- Modify: `backend/app/services/evaluation/run_eval.py`
- Create: `backend/tests/test_live_demo_eval.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `LIVE_DEMO_CASE_IDS: tuple[str, ...]` with exactly 12 checked-in IDs.
- Produces: `build_grounded_prompt(item: dict, evidence: list[Document]) -> list[BaseMessage]`.
- Produces: `capture_live_responses(output_dir: Path, llm=None) -> dict`.
- Produces: `load_live_response_provider(document: dict) -> ResponseProvider`.
- Consumes: `build_deepseek_llm("evaluation")` and public `retrieve_eval_documents(question, top_k=10)`.

- [ ] **Step 1: Write failing selection and prompt tests**

```python
EXPECTED_IDS = (
    "RAG_SVF_AML_001", "KAG_AI_ADVISOR_001", "DR_AI_ADVISOR_001",
    "EXP_051", "EXP_062", "EXP_071", "EXP_077", "EXP_082",
    "EXP_088", "EXP_089", "EXP_090", "EXP_095",
)


def test_live_selection_is_fixed_and_stratified():
    assert LIVE_DEMO_CASE_IDS == EXPECTED_IDS
    selected = select_live_cases(load_benchmark_questions())
    assert {item["language"] for item in selected} == {"en", "zh-Hant"}
    assert {item["expected_retrieval_mode"] for item in selected} == {"rag", "kag", "deep_research"}


def test_prompt_contains_only_numbered_evidence():
    messages = build_grounded_prompt(case(), [Document(page_content="Rule", metadata={"doc_id": "D1"})])
    assert "[E1]" in messages[-1].content
    assert "If evidence is insufficient" in messages[0].content
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd backend && ..\.venv\Scripts\python.exe -m pytest tests/test_live_demo_eval.py -v`

Expected: FAIL because `live_demo_eval.py` does not exist.

- [ ] **Step 3: Expose retrieval and implement capture**

Rename `_retrieve_eval_documents` to `retrieve_eval_documents` in `run_eval.py` and retain a compatibility alias for existing experiments.

Capture each case with `time.perf_counter()`, extract `AIMessage.content` and `response_metadata`, and write this schema:

```python
{
    "schema_version": 1,
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "prompt_version": "demo-grounded-v1",
    "cases": [{
        "case_id": case_id,
        "response": response_text,
        "evidence_ids": evidence_ids,
        "latency_ms": latency_ms,
        "usage": safe_usage,
        "error": None,
    }],
}
```

Retry only 429 and transient 5xx errors twice with 1-second then 2-second waits. Do not retry 401/403, empty responses, or malformed responses. Write raw artifacts under `artifacts/evaluation/live/` and add that directory to `.gitignore`.

- [ ] **Step 4: Test retries, redaction, and empty output**

```python
def test_auth_failure_is_not_retried_and_secret_is_absent(fake_llm): ...
def test_rate_limit_retries_twice(fake_llm, fake_sleep): ...
def test_empty_response_is_recorded_as_error(fake_llm): ...
def test_artifact_contains_no_api_key(monkeypatch, tmp_path): ...
```

Run: `cd backend && ..\.venv\Scripts\python.exe -m pytest tests/test_live_demo_eval.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/evaluation/live_demo_eval.py backend/app/services/evaluation/run_eval.py backend/tests/test_live_demo_eval.py .gitignore
git commit -m "feat(eval): capture live DeepSeek demo responses"
```

---

### Task 5: Enforce the credentialed live demo gate

**Files:**
- Create: `backend/app/services/evaluation/live_demo_gate.py`
- Create: `backend/tests/test_live_demo_gate.py`
- Create: `.github/workflows/demo-acceptance.yml`
- Modify: `docs/evaluation_protocol.md`

**Interfaces:**
- Produces: `evaluate_live_demo_gate(live_document: dict, summary: dict) -> dict`.
- Produces CLI: `python -m app.services.evaluation.live_demo_gate --output-dir artifacts/evaluation/live`.
- Requires: model `deepseek-v4-flash`, 12 non-empty responses, 12 measured faithfulness rows, no live errors, `avg_faithfulness >= 0.45`, selected-case `avg_unsupported_claim_rate <= 0.10`.

- [ ] **Step 1: Write failing gate tests**

```python
def test_gate_rejects_wrong_model():
    result = evaluate_live_demo_gate(live_document(model="deepseek-chat"), measured_summary())
    assert result["passed"] is False
    assert result["failures"][0]["metric"] == "model"


def test_gate_requires_all_twelve_measured_rows():
    summary = measured_summary(measured=11)
    assert evaluate_live_demo_gate(live_document(), summary)["passed"] is False


def test_gate_passes_approved_demo_baseline():
    result = evaluate_live_demo_gate(live_document(), measured_summary(faithfulness=.70, unsupported=.08))
    assert result["passed"] is True
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd backend && ..\.venv\Scripts\python.exe -m pytest tests/test_live_demo_gate.py -v`

Expected: FAIL because the gate module does not exist.

- [ ] **Step 3: Implement the gate and CLI**

The CLI must:

1. fail immediately when `DEEPSEEK_API_KEY` is empty;
2. call `capture_live_responses`;
3. call `run_eval(response_provider=load_live_response_provider(document))`;
4. evaluate only the selected 12 rows for live coverage and selected unsupported rate;
5. write a secret-free JSON result and exit non-zero on failure.

- [ ] **Step 4: Add the protected acceptance workflow**

```yaml
name: Demo Acceptance
on:
  workflow_dispatch:
  push:
    tags: ["demo-*"]
jobs:
  live-deepseek-gate:
    runs-on: ubuntu-latest
    env:
      DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
      EMBEDDING_PROVIDER: local_hash
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: test -n "$DEEPSEEK_API_KEY"
      - run: python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
      - working-directory: backend
        run: ../.venv/bin/python -m app.services.corpus.build_cache
      - working-directory: backend
        run: ../.venv/bin/python -m app.services.kag.build_graph_cache
      - working-directory: backend
        run: ../.venv/bin/python -m app.services.evaluation.live_demo_gate --output-dir artifacts/evaluation/live
```

Upload the secret-free gate JSON as an artifact even when the gate fails.

- [ ] **Step 5: Run tests and deterministic gate**

Run:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest tests/test_live_demo_gate.py tests/test_generation_faithfulness.py -v
..\.venv\Scripts\python.exe -m app.services.evaluation.release_gate
```

Expected: tests PASS; deterministic gate reports 108 cases and does not claim live coverage.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/evaluation/live_demo_gate.py backend/tests/test_live_demo_gate.py .github/workflows/demo-acceptance.yml docs/evaluation_protocol.md
git commit -m "feat(eval): enforce live DeepSeek demo gate"
```

---

### Task 6: Close demo-critical error, cancellation, and coverage gaps

**Files:**
- Modify: `frontend/e2e/critical-workflows.spec.ts`
- Modify: `.github/workflows/release-gates.yml`
- Modify: `backend/pyproject.toml`
- Modify: `backend/requirements.lock`
- Modify: `requirements.txt` only if its lockfile pointer changes
- Test: `backend/tests/test_deepseek_runtime.py`
- Test: `backend/tests/test_corpus_ingestion_report.py`
- Test: `backend/tests/test_live_demo_eval.py`
- Test: `backend/tests/test_live_demo_gate.py`

**Interfaces:**
- Adds two E2E scenarios, bringing the checked-in total from four to six.
- Adds a 70% targeted line-coverage command for the changed DeepSeek, corpus-integrity, and live-evaluation modules.

- [ ] **Step 1: Add failing Playwright cases**

```typescript
test("cancels an in-flight analysis and restores the submit action", async ({ page }) => {
  await page.route("**/api/backend/api/v1/bank-account/verify/stream", () => new Promise(() => {}));
  await page.goto("/");
  await page.getByRole("button", { name: "Submit Analysis" }).click();
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByRole("button", { name: "Submit Analysis" })).toBeEnabled();
});


test("shows a recoverable message after an LLM stream error", async ({ page }) => {
  await page.route("**/api/backend/api/v1/bank-account/verify/stream", route => route.fulfill({
    status: 503,
    contentType: "application/json",
    body: JSON.stringify({ detail: "DeepSeek is temporarily unavailable" }),
  }));
  await page.goto("/");
  await page.getByRole("button", { name: "Submit Analysis" }).click();
  await expect(page.getByText(/DeepSeek is temporarily unavailable|HTTP 503/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Submit Analysis" })).toBeEnabled();
});
```

- [ ] **Step 2: Run E2E and verify RED**

Run: `cd frontend && npm run test:e2e -- --workers=1`

Expected: at least the cancellation case FAILS until cancellation state handling is corrected; if existing behavior already passes, keep the characterization test and verify the error case independently.

- [ ] **Step 3: Apply the explicit request-outcome state correction**

Track the request outcome outside React's batched updater so `finally` never reads a stale `prev.error` value:

```typescript
let requestOutcome: "success" | "error" | "aborted" = "success";
try {
  // existing fetch and SSE loop
} catch (err: unknown) {
  if (err instanceof Error && err.name === "AbortError") {
    requestOutcome = "aborted";
  } else {
    requestOutcome = "error";
    setState((prev) => ({
      ...prev,
      error: err instanceof Error ? err.message : "Unknown error",
    }));
  }
} finally {
  if (timerRef.current) clearInterval(timerRef.current);
  setState((prev) => ({
    ...prev,
    isStreaming: false,
    phase: prev.phase === "action_required"
      ? "action_required"
      : requestOutcome === "success" ? "done" : "idle",
    error: requestOutcome === "aborted" ? null : prev.error,
    elapsedTime: Math.round((performance.now() - startTimeRef.current) / 1000),
  }));
}
```

- [ ] **Step 4: Add pytest-cov and the targeted coverage gate**

Add `pytest-cov>=7.0.0` to the test dependency group and regenerated lockfile. Add this CI command after the full backend suite:

```bash
.venv/bin/python -m pytest \
  tests/test_deepseek_runtime.py \
  tests/test_corpus_ingestion_report.py \
  tests/test_live_demo_eval.py \
  tests/test_live_demo_gate.py \
  --cov=app.services.llm.deepseek \
  --cov=app.services.corpus.corpus_ingestor \
  --cov=app.services.evaluation.live_demo_eval \
  --cov=app.services.evaluation.live_demo_gate \
  --cov-report=term-missing \
  --cov-fail-under=70
```

- [ ] **Step 5: Run the complete frontend and targeted backend chains**

Run:

```powershell
cd frontend
npm run lint
npm run typecheck
npm run test:config
npm run test:coverage
npm run test:e2e -- --workers=1
npm run build
```

Then run the targeted pytest-cov command locally with `..\.venv\Scripts\python.exe`.

Expected: six Playwright tests PASS; targeted line coverage is at least 70%; frontend build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/e2e/critical-workflows.spec.ts frontend/src/hooks/useAgentStream.ts .github/workflows/release-gates.yml backend/pyproject.toml backend/requirements.lock requirements.txt backend/tests
git commit -m "test(demo): enforce recovery and targeted coverage gates"
```

---

### Task 7: Produce a self-contained demo release and acceptance report

**Files:**
- Add from the original workspace: `docs/system-evaluation-report-2026-08-04.md`
- Add from the original workspace: `docs/risk-assessment-2026-08-04.md`
- Add from the original workspace: `docs/superpowers/plans/2026-08-05-system-upgrade-optimization.md`
- Modify: `docs/upgrade-summary-2026-08-06.md`
- Create: `docs/demo-acceptance-2026-08-10.md`
- Create after a successful live run: `docs/eval-baselines/deepseek-demo-live-2026-08-10.json`

**Interfaces:**
- Produces a clean, versioned evidence chain from original target through remediation and final demo acceptance.
- Does not add root debug packages, log files, Phase 4 modules, or API secrets.

- [ ] **Step 1: Add the three already-authored target documents without altering their content**

Copy the exact files from `F:\MyFintech\docs\...` into the isolated worktree, verify their SHA-256 hashes match the source files, then stage only those three paths.

- [ ] **Step 2: Run the real 12-case DeepSeek gate**

Run:

```powershell
if (-not $env:DEEPSEEK_API_KEY) { throw 'Set DEEPSEEK_API_KEY in the secure process environment first' }
cd backend
..\.venv\Scripts\python.exe -m app.services.evaluation.live_demo_gate --output-dir artifacts/evaluation/live
```

Expected: model `deepseek-v4-flash`, 12/12 responses, 12/12 measured faithfulness, no API errors, and gate PASS. Do not paste or store the API key in a script or document.

- [ ] **Step 3: Write the sanitized live summary**

The checked-in JSON contains aggregate metrics, case IDs, fingerprints, model ID, latency/token totals, and the raw-artifact SHA-256. It excludes response bodies if they contain user-entered data and always excludes secrets.

- [ ] **Step 4: Correct the upgrade summary and write final demo acceptance**

Record fresh evidence:

- backend pass/skip/warning counts;
- 20/20 required sources parsed and total chunks;
- 108-case deterministic metrics;
- 12-case DeepSeek metrics;
- frontend unit coverage and six E2E results;
- two moderate npm findings as accepted demo risk;
- gold review status `108 pending` as non-blocking;
- explicit statement that Phase 4 is excluded.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/system-evaluation-report-2026-08-04.md docs/risk-assessment-2026-08-04.md docs/superpowers/plans/2026-08-05-system-upgrade-optimization.md docs/upgrade-summary-2026-08-06.md docs/demo-acceptance-2026-08-10.md docs/eval-baselines/deepseek-demo-live-2026-08-10.json
git commit -m "docs: record DeepSeek demo acceptance"
```

- [ ] **Step 6: Verify a new clean checkout**

From a newly exported `HEAD` or detached clean worktree, run:

```powershell
F:\MyFintech\.venv\Scripts\python.exe -m pip check
F:\MyFintech\.venv\Scripts\python.exe -m pytest tests -q
F:\MyFintech\.venv\Scripts\python.exe -m app.services.corpus.build_cache
F:\MyFintech\.venv\Scripts\python.exe -m app.services.kag.build_graph_cache
F:\MyFintech\.venv\Scripts\python.exe -m app.services.evaluation.release_gate
```

In the clean frontend, run `npm ci`, lint, typecheck, config tests, coverage, six E2E tests, `npm audit --audit-level=high`, and build. Then run the credentialed live gate once from the release candidate commit.

Expected: all non-live commands exit 0; live gate exits 0; `git status --short` is empty; no Phase 4 files appear in `git diff HEAD^..HEAD` or the release tree.

---

## Execution Order and Checkpoints

1. Tasks 1–3 establish provider and corpus correctness; checkpoint with full backend tests and a clean cache build.
2. Tasks 4–5 establish real-LLM evidence and its protected gate; checkpoint with mocked live tests before spending API credits.
3. Task 6 closes automated demo regressions; checkpoint with the complete local quality chain.
4. Task 7 is the only step allowed to claim demo acceptance, after a real credentialed DeepSeek run and clean-checkout verification.
