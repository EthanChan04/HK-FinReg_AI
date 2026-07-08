# Regulatory Coverage and DeepResearch Diversity Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve AI wealth advisory and product launch retrieval so classifier/planner metadata covers HKMA, SFC, PCPD, consumer protection, suitability, and personal data, while evaluation separates classifier regulator coverage from evidence regulator coverage and DeepResearch avoids PCPD-only evidence saturation.

**Architecture:** Keep the current deterministic `classify_query(...)`, `build_query_plan(...)`, `RetrievalService.retrieve(...)`, and DeepResearch graph as the integration points. Add narrowly scoped regulatory expansion rules, metric helpers for classifier-vs-evidence coverage, and a post-retrieval regulator diversity gate used by DeepResearch sub-question retrieval and deterministic eval.

**Tech Stack:** Python, FastAPI backend services, Pydantic schemas, LangChain `Document`/evidence conversion, pytest, existing deterministic evaluation runner.

## Global Constraints

- Preserve current public interfaces unless a task explicitly adds an optional keyword argument.
- Do not add external LLM or network dependencies.
- Make all new routing, metric, and diversity behavior deterministic and unit-testable.
- Do not reduce existing citation audit, retrieval mode, strategy, or query planner metrics.
- For AI/wealth advisory/product launch cases, target regulators are exactly `HKMA`, `SFC`, and `PCPD`.
- For AI/wealth advisory/product launch cases, target topic expansions include `consumer_protection`, `suitability`, and `personal_data`.
- Evaluation summary must expose both `avg_classifier_regulator_coverage` and `avg_evidence_regulator_coverage`.
- Existing `avg_regulator_coverage` may remain temporarily as an alias, but it must not be the only regulator coverage metric in summary output.

---

## Current Context

Relevant existing behavior:

- `backend/app/services/retrieval/query_classifier.py` recognizes AI and privacy, but AI wealth/product launch cases currently only guarantee AI topics. Privacy terms add `PCPD`, but HKMA/SFC/PCPD coverage is not guaranteed for wealth advisory launch wording.
- `backend/app/services/retrieval/query_planner.py` expands `ai`, `genai`, and product launch terms, but does not yet protect or emit `consumer_protection`, `suitability`, or `personal_data`.
- `backend/app/services/evaluation/run_eval.py` computes one `regulator_coverage` from classifier filters or SVF fallback. This is classifier coverage, not evidence coverage.
- `backend/app/services/deepresearch/workflow.py` retrieves top evidence per sub-question directly from `RetrievalService.retrieve(...)`, so a strong PCPD/AI match can fill all top evidence slots.
- `backend/data/evaluation/benchmark_questions.json` already contains `KAG_AI_ADVISOR_001` and `DR_AI_ADVISOR_001` with expected regulators `HKMA`, `SFC`, `PCPD`.

## File Structure

### Modify

- `backend/app/services/retrieval/query_classifier.py`
  - Add deterministic cross-regulator expansion for AI wealth advisory and AI product launch intent.

- `backend/app/services/retrieval/query_planner.py`
  - Add aliases/protected terms for wealth management, consumer protection, suitability, personal data, HKMA, SFC, and PCPD.

- `backend/app/services/evaluation/run_eval.py`
  - Split classifier regulator coverage and evidence regulator coverage.
  - Reuse evidence regulator extraction for normal eval and DeepResearch eval.

- `backend/app/services/deepresearch/workflow.py`
  - Add regulator diversity gate after retrieval for sub-questions and gap retrieval.

- `backend/tests/test_query_classifier.py`
  - Add AI wealth/product launch regulator and topic coverage tests.

- `backend/tests/test_query_planner.py`
  - Add expansion tests for consumer protection, suitability, and personal data.

- `backend/tests/test_evaluation_error_reporting.py`
  - Add metric summary regression for separate regulator coverage fields.

- `backend/tests/test_deepresearch.py`
  - Add diversity gate tests that prove PCPD evidence cannot occupy every selected slot when HKMA/SFC evidence exists.

### Optional Later Modify

- `docs/evaluation_protocol.md`
  - Document the two regulator coverage metrics and release thresholds after implementation stabilizes.

---

## Task 1: Classifier Regulatory Expansion

**Files:**
- Modify: `backend/app/services/retrieval/query_classifier.py`
- Modify: `backend/tests/test_query_classifier.py`

**Interfaces:**
- Consumes: `classify_query(query: str) -> QueryProfile`
- Produces: for AI wealth advisory/product launch queries, `QueryProfile.filters["regulator"] == ["HKMA", "SFC", "PCPD"]` and `topics` includes `AI`, `GenAI`, `ai_governance`, `wealth_management`, `consumer_protection`, `suitability`, `personal_data`.

- [ ] **Step 1: Write failing classifier tests**

Add these tests to `backend/tests/test_query_classifier.py`:

```python
def test_query_classifier_expands_ai_wealth_advisory_regulators_and_topics():
    from app.services.retrieval.query_classifier import classify_query

    profile = classify_query(
        "Which regulators and obligations are relevant when a Hong Kong virtual bank launches an AI wealth advisory product?"
    )

    assert profile.retrieval_mode == "kag"
    assert profile.filters["regulator"] == ["HKMA", "SFC", "PCPD"]
    assert "wealth_management" in profile.filters["topics"]
    assert "consumer_protection" in profile.filters["topics"]
    assert "suitability" in profile.filters["topics"]
    assert "personal_data" in profile.filters["topics"]
    assert "ai_wealth_product_launch" in profile.reasons


def test_query_classifier_keeps_deepresearch_mode_with_ai_launch_regulatory_expansion():
    from app.services.retrieval.query_classifier import classify_query

    profile = classify_query(
        "Analyze compliance risks for launching an AI investment advisor and generate a pre-launch checklist."
    )

    assert profile.retrieval_mode == "deep_research"
    assert profile.filters["regulator"] == ["HKMA", "SFC", "PCPD"]
    assert "consumer_protection" in profile.filters["topics"]
    assert "suitability" in profile.filters["topics"]
    assert "personal_data" in profile.filters["topics"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_query_classifier.py -q
```

Expected: the two new tests fail because `regulator`, `wealth_management`, `consumer_protection`, `suitability`, and `personal_data` are missing.

- [ ] **Step 3: Implement minimal classifier helper**

In `backend/app/services/retrieval/query_classifier.py`, add helpers near `_add_filter(...)`:

```python
def _has_any(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def _is_ai_wealth_product_launch(text: str, reasons: list[str]) -> bool:
    has_ai = "ai" in reasons or _has_any(text, r"\b(ai|artificial intelligence|genai)\b")
    has_wealth = _has_any(text, r"\b(wealth|investment|investor|advisory|advisor|suitability|portfolio)\b")
    has_launch_or_product = _has_any(text, r"\b(product|launch|launches|launching|pre[-\s]?launch|onboarding)\b")
    return has_ai and has_wealth and has_launch_or_product
```

Then, after the existing AI/privacy blocks and before the final research block, add:

```python
    if _is_ai_wealth_product_launch(text, reasons):
        _add_filter(filters, "regulator", ["HKMA", "SFC", "PCPD"])
        _add_filter(
            filters,
            "topics",
            [
                "wealth_management",
                "consumer_protection",
                "suitability",
                "personal_data",
            ],
        )
        mode = "kag"
        confidence = max(confidence, 0.79)
        reasons.append("ai_wealth_product_launch")
```

Keep the existing research block after this code so checklist/report/analyze wording can still upgrade `mode` to `deep_research`.

- [ ] **Step 4: Run classifier tests to verify pass**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_query_classifier.py -q
```

Expected: all classifier tests pass.

- [ ] **Step 5: Commit**

```powershell
cd F:\MyFintech
git add backend/app/services/retrieval/query_classifier.py backend/tests/test_query_classifier.py
git commit -m "feat: expand ai wealth advisory regulator classification"
```

---

## Task 2: Query Planner Regulatory Expansion Terms

**Files:**
- Modify: `backend/app/services/retrieval/query_planner.py`
- Modify: `backend/tests/test_query_planner.py`

**Interfaces:**
- Consumes: `build_query_plan(query: str, *, profile: QueryProfile, term_statistics: TermStatistics | None = None) -> QueryPlan`
- Produces: expansion terms containing `HKMA`, `SFC`, `PCPD`, `consumer protection`, `suitability`, and `personal data` when classifier filters contain the corresponding regulators/topics.

- [ ] **Step 1: Write failing query planner test**

Add this test to `backend/tests/test_query_planner.py`:

```python
def test_query_planner_expands_ai_wealth_regulatory_terms_from_filters():
    from app.services.retrieval.query_planner import build_query_plan
    from app.services.retrieval.term_statistics import TermStatistics

    profile = classify_query("AI wealth advisory product launch")
    stats = TermStatistics(
        document_count=100,
        document_frequency={
            "hkma": 20,
            "sfc": 18,
            "pcpd": 12,
            "consumer protection": 9,
            "suitability": 8,
            "personal data": 10,
            "wealth management": 7,
        },
    )

    plan = build_query_plan(
        "AI wealth advisory product launch",
        profile=profile,
        term_statistics=stats,
    )

    assert "HKMA" in plan.expansion_terms
    assert "SFC" in plan.expansion_terms
    assert "PCPD" in plan.expansion_terms
    assert "consumer protection" in plan.expansion_terms
    assert "suitability" in plan.expansion_terms
    assert "personal data" in plan.expansion_terms
    assert "wealth management" in plan.expansion_terms
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_query_planner.py -q
```

Expected: the new test fails because aliases/protected terms are missing.

- [ ] **Step 3: Extend aliases and protected terms**

In `backend/app/services/retrieval/query_planner.py`, extend `_ALIASES`:

```python
    "wealth": ["wealth management", "suitability"],
    "advisory": ["wealth management", "suitability"],
    "advisor": ["wealth management", "suitability"],
    "investment": ["wealth management", "suitability"],
    "wealth_management": ["wealth management"],
    "consumer_protection": ["consumer protection"],
    "suitability": ["suitability"],
    "personal_data": ["personal data", "PCPD"],
    "privacy": ["personal data", "PCPD"],
    "hkma": ["HKMA"],
    "sfc": ["SFC"],
    "pcpd": ["PCPD"],
```

Extend `_PROTECTED_TERMS`:

```python
    "consumer protection",
    "personal data",
    "suitability",
    "wealth management",
```

- [ ] **Step 4: Add regulator filter candidates**

In `_candidate_terms(...)`, after the module tag loop, add:

```python
    for regulator in profile.filters.get("regulator", []):
        regulator_l = regulator.lower()
        if regulator_l in _ALIASES:
            candidates.extend((term, f"filter_regulator:{regulator}") for term in _ALIASES[regulator_l])
```

- [ ] **Step 5: Run planner tests to verify pass**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_query_planner.py -q
```

Expected: all query planner tests pass.

- [ ] **Step 6: Commit**

```powershell
cd F:\MyFintech
git add backend/app/services/retrieval/query_planner.py backend/tests/test_query_planner.py
git commit -m "feat: expand ai wealth query planning terms"
```

---

## Task 3: Separate Classifier and Evidence Regulator Coverage

**Files:**
- Modify: `backend/app/services/evaluation/run_eval.py`
- Modify: `backend/tests/test_evaluation_error_reporting.py`

**Interfaces:**
- Produces: row fields `classifier_regulator_coverage` and `evidence_regulator_coverage`.
- Produces: summary fields `avg_classifier_regulator_coverage` and `avg_evidence_regulator_coverage`.
- Keeps: `avg_regulator_coverage` as a backward-compatible alias for classifier coverage during transition.

- [ ] **Step 1: Write failing evaluation summary test**

Add this test to `backend/tests/test_evaluation_error_reporting.py`:

```python
def test_evaluation_splits_classifier_and_evidence_regulator_coverage(monkeypatch):
    from langchain_core.documents import Document

    from app.services.evaluation import run_eval

    monkeypatch.setattr(
        run_eval,
        "load_benchmark_questions",
        lambda: [
            {
                "id": "KAG_AI_ADVISOR_001",
                "question": "AI wealth advisory product launch",
                "expected_retrieval_mode": "kag",
                "expected_strategy_id": "ai_governance_kag",
                "expected_topics": [],
                "expected_regulators": ["HKMA", "SFC", "PCPD"],
                "expected_expansion_terms": [],
            }
        ],
    )
    monkeypatch.setattr(
        run_eval,
        "_retrieve_eval_documents",
        lambda question, top_k=10: [
            Document(page_content="HKMA governance", metadata={"regulator": "HKMA"}),
            Document(page_content="SFC suitability", metadata={"regulator": "SFC"}),
            Document(page_content="PCPD personal data", metadata={"regulator": "PCPD"}),
        ],
    )
    monkeypatch.setattr(run_eval, "_compute_graph_path_count", lambda item: 0)
    monkeypatch.setattr(run_eval, "_compute_citation_audit", lambda item: (1.0, 0.0))
    monkeypatch.setattr(run_eval, "_compute_deepresearch_gap_count", lambda item: 0)

    summary = run_eval.run_eval()
    row = summary["rows"][0]

    assert row["classifier_regulator_coverage"] == 1.0
    assert row["evidence_regulator_coverage"] == 1.0
    assert summary["avg_classifier_regulator_coverage"] == 1.0
    assert summary["avg_evidence_regulator_coverage"] == 1.0
    assert summary["avg_regulator_coverage"] == summary["avg_classifier_regulator_coverage"]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_evaluation_error_reporting.py -q
```

Expected: failure because the new row and summary keys do not exist.

- [ ] **Step 3: Add evidence regulator extraction helpers**

In `backend/app/services/evaluation/run_eval.py`, add:

```python
def _document_regulators(docs: list) -> list[str]:
    regulators: list[str] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        value = metadata.get("regulator")
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item and str(item) not in regulators:
                regulators.append(str(item))
    return regulators


def _compute_evidence_regulator_coverage(item: dict) -> float:
    docs = _retrieve_eval_documents(item["question"], top_k=10)
    return _coverage(item.get("expected_regulators", []), _document_regulators(docs))
```

- [ ] **Step 4: Update row generation**

In `run_eval()`, rename the existing classifier coverage calculation:

```python
        classifier_regulator_coverage = _coverage(
            item.get("expected_regulators", []), actual_regulators
        )
```

Add metric execution:

```python
        evidence_regulator_coverage = _run_metric(
            question_id,
            "evidence_regulator_coverage",
            metric_errors,
            lambda: _compute_evidence_regulator_coverage(item),
            0.0,
        )
```

Replace the row key:

```python
                "classifier_regulator_coverage": classifier_regulator_coverage,
                "regulator_coverage": classifier_regulator_coverage,
                "evidence_regulator_coverage": evidence_regulator_coverage,
```

- [ ] **Step 5: Update summary and CLI printing**

In `summary`, replace the single regulator coverage block with:

```python
        "avg_classifier_regulator_coverage": round(
            sum(row["classifier_regulator_coverage"] for row in rows) / total, 3
        ),
        "avg_regulator_coverage": round(
            sum(row["classifier_regulator_coverage"] for row in rows) / total, 3
        ),
        "avg_evidence_regulator_coverage": round(
            sum(row["evidence_regulator_coverage"] for row in rows) / total, 3
        ),
```

In `main()`, print these row values:

```python
                print(f"    classifier_regulator_coverage: {row['classifier_regulator_coverage']}")
                print(f"    evidence_regulator_coverage: {row['evidence_regulator_coverage']}")
```

- [ ] **Step 6: Run evaluation tests**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_evaluation_error_reporting.py -q
```

Expected: all evaluation error reporting tests pass.

- [ ] **Step 7: Commit**

```powershell
cd F:\MyFintech
git add backend/app/services/evaluation/run_eval.py backend/tests/test_evaluation_error_reporting.py
git commit -m "feat: split classifier and evidence regulator coverage"
```

---

## Task 4: DeepResearch Regulator Diversity Gate

**Files:**
- Modify: `backend/app/services/deepresearch/workflow.py`
- Modify: `backend/tests/test_deepresearch.py`

**Interfaces:**
- Produces: `_apply_regulator_diversity_gate(evidence: list[dict], required_regulators: list[str], top_k: int) -> list[dict]`
- Consumes: `ResearchSubQuestion.required_topics` and query classifier `profile.filters["regulator"]`
- Behavior: selected evidence should include available required regulators first, then fill remaining slots by original rank.

- [ ] **Step 1: Write failing diversity gate test**

Add this test to `backend/tests/test_deepresearch.py`:

```python
def test_deepresearch_regulator_diversity_gate_prioritizes_available_regulators():
    from app.services.deepresearch.workflow import _apply_regulator_diversity_gate

    evidence = [
        {"evidence_id": "p1", "regulator": "PCPD", "text": "PCPD AI guidance"},
        {"evidence_id": "p2", "regulator": "PCPD", "text": "PCPD personal data"},
        {"evidence_id": "p3", "regulator": "PCPD", "text": "PCPD privacy"},
        {"evidence_id": "h1", "regulator": "HKMA", "text": "HKMA governance"},
        {"evidence_id": "s1", "regulator": "SFC", "text": "SFC suitability"},
    ]

    selected = _apply_regulator_diversity_gate(
        evidence,
        required_regulators=["HKMA", "SFC", "PCPD"],
        top_k=3,
    )

    assert [item["regulator"] for item in selected] == ["HKMA", "SFC", "PCPD"]
```

Add a missing-regulator fallback test:

```python
def test_deepresearch_regulator_diversity_gate_preserves_rank_when_no_alternative_exists():
    from app.services.deepresearch.workflow import _apply_regulator_diversity_gate

    evidence = [
        {"evidence_id": "p1", "regulator": "PCPD", "text": "PCPD AI guidance"},
        {"evidence_id": "p2", "regulator": "PCPD", "text": "PCPD personal data"},
    ]

    selected = _apply_regulator_diversity_gate(
        evidence,
        required_regulators=["HKMA", "SFC", "PCPD"],
        top_k=3,
    )

    assert [item["evidence_id"] for item in selected] == ["p1", "p2"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_deepresearch.py -q
```

Expected: import failure for `_apply_regulator_diversity_gate`.

- [ ] **Step 3: Implement diversity helper**

In `backend/app/services/deepresearch/workflow.py`, add:

```python
def _evidence_regulator(item: dict) -> str | None:
    value = item.get("regulator") or (item.get("metadata") or {}).get("regulator")
    return str(value).upper() if value else None


def _apply_regulator_diversity_gate(
    evidence: list[dict],
    required_regulators: list[str],
    top_k: int,
) -> list[dict]:
    """Select ranked evidence with one available item per required regulator first."""
    if top_k <= 0 or not evidence:
        return []

    required = [reg.upper() for reg in required_regulators if reg]
    selected: list[dict] = []
    selected_ids: set[int] = set()

    for regulator in required:
        for index, item in enumerate(evidence):
            if index in selected_ids:
                continue
            if _evidence_regulator(item) == regulator:
                selected.append(item)
                selected_ids.add(index)
                break
        if len(selected) >= top_k:
            return selected[:top_k]

    for index, item in enumerate(evidence):
        if index in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break

    return selected
```

- [ ] **Step 4: Use gate in sub-question retrieval**

In `_retrieve_for_sub_question(...)`, add optional parameter:

```python
    required_regulators: list[str] | None = None,
```

After `evidence = retrieval_service.retrieve(...)`, replace the return with:

```python
        if evidence:
            dumped = [chunk.model_dump() for chunk in evidence]
            return _apply_regulator_diversity_gate(
                dumped,
                required_regulators or [],
                top_k=top_k,
            )
```

In `retrieval_node(...)`, compute and pass required regulators:

```python
            profile = classify_query(sq.question)
            required_regulators = [
                topic for topic in sq.required_topics if topic in {"HKMA", "SFC", "PCPD"}
            ] or profile.filters.get("regulator", [])
```

Then pass:

```python
                required_regulators=required_regulators,
```

In `gap_retriever_node(...)`, call `_retrieve_for_sub_question(...)` with:

```python
                required_regulators=classify_query(followup_query).filters.get("regulator", []),
```

- [ ] **Step 5: Run DeepResearch tests**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_deepresearch.py -q
```

Expected: all DeepResearch tests pass.

- [ ] **Step 6: Commit**

```powershell
cd F:\MyFintech
git add backend/app/services/deepresearch/workflow.py backend/tests/test_deepresearch.py
git commit -m "feat: add deepresearch regulator diversity gate"
```

---

## Task 5: Apply Diversity Gate in Deterministic DeepResearch Eval

**Files:**
- Modify: `backend/app/services/evaluation/run_eval.py`
- Modify: `backend/tests/test_evaluation_error_reporting.py`

**Interfaces:**
- Consumes: `_apply_regulator_diversity_gate(...)` from DeepResearch workflow.
- Produces: DeepResearch eval gap count that uses the same diversity behavior as runtime retrieval.

- [ ] **Step 1: Add eval regression test for gate reuse**

Add this test to `backend/tests/test_evaluation_error_reporting.py`:

```python
def test_deepresearch_gap_eval_uses_regulator_diversity_gate(monkeypatch):
    from langchain_core.documents import Document

    from app.schemas.deepresearch import ResearchPlan, ResearchSubQuestion
    from app.services.evaluation import run_eval

    plan = ResearchPlan(
        research_goal="AI advisor launch",
        sub_questions=[
            ResearchSubQuestion(
                id="SQ1",
                question="AI advisor launch regulators",
                retrieval_mode="kag",
                required_topics=["HKMA", "SFC", "PCPD"],
                evidence_min_count=3,
            )
        ],
    )

    monkeypatch.setattr(run_eval, "build_research_plan", lambda question: plan, raising=False)
    monkeypatch.setattr(
        run_eval,
        "_retrieve_eval_documents",
        lambda question, top_k=10: [
            Document(page_content="PCPD AI", metadata={"regulator": "PCPD"}),
            Document(page_content="PCPD privacy", metadata={"regulator": "PCPD"}),
            Document(page_content="HKMA governance", metadata={"regulator": "HKMA"}),
            Document(page_content="SFC suitability", metadata={"regulator": "SFC"}),
        ],
    )

    assert run_eval._compute_deepresearch_gap_count(
        {
            "id": "DR_AI_ADVISOR_001",
            "question": "AI advisor launch",
            "expected_retrieval_mode": "deep_research",
        }
    ) == 0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_evaluation_error_reporting.py::test_deepresearch_gap_eval_uses_regulator_diversity_gate -q
```

Expected: failure until `_compute_deepresearch_gap_count(...)` imports and applies the gate.

- [ ] **Step 3: Update `_compute_deepresearch_gap_count(...)` imports**

Inside `_compute_deepresearch_gap_count(...)`, add:

```python
        from app.services.deepresearch.workflow import _apply_regulator_diversity_gate
```

- [ ] **Step 4: Apply gate before evaluator receives evidence**

Replace the evidence list assignment with:

```python
            dumped_evidence = [chunk.model_dump() for chunk in evidence]
            required_regulators = [
                topic for topic in sq.required_topics if topic in {"HKMA", "SFC", "PCPD"}
            ]
            evidence_by_subquestion[sq.id] = _apply_regulator_diversity_gate(
                dumped_evidence,
                required_regulators,
                top_k=sq.evidence_min_count + 3,
            )
```

- [ ] **Step 5: Run evaluation tests**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_evaluation_error_reporting.py tests/test_deepresearch.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
cd F:\MyFintech
git add backend/app/services/evaluation/run_eval.py backend/tests/test_evaluation_error_reporting.py
git commit -m "test: align deepresearch evaluation with regulator diversity gate"
```

---

## Task 6: Benchmark and Release Gate Verification

**Files:**
- Modify if needed: `backend/data/evaluation/benchmark_questions.json`
- Optional modify: `docs/evaluation_protocol.md`

**Interfaces:**
- Consumes: existing benchmark cases `KAG_AI_ADVISOR_001` and `DR_AI_ADVISOR_001`.
- Produces: evaluation summary with separate classifier/evidence regulator coverage.

- [ ] **Step 1: Confirm benchmark expected fields**

Inspect `backend/data/evaluation/benchmark_questions.json` and confirm these records exist:

```json
{
  "id": "KAG_AI_ADVISOR_001",
  "expected_regulators": ["HKMA", "SFC", "PCPD"],
  "expected_topics": ["AI", "wealth_management", "consumer_protection"]
}
```

```json
{
  "id": "DR_AI_ADVISOR_001",
  "expected_regulators": ["HKMA", "SFC", "PCPD"],
  "expected_topics": ["AI", "AML", "CDD", "data_privacy", "consumer_protection"]
}
```

If `KAG_AI_ADVISOR_001.expected_topics` does not include `suitability` and `personal_data`, update it to:

```json
"expected_topics": ["AI", "wealth_management", "consumer_protection", "suitability", "personal_data"]
```

- [ ] **Step 2: Run targeted tests**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_query_classifier.py tests/test_query_planner.py tests/test_deepresearch.py tests/test_evaluation_error_reporting.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run full backend tests**

Run:

```powershell
cd F:\MyFintech\backend
python -m pytest -q
```

Expected: all backend tests pass.

- [ ] **Step 4: Run deterministic evaluation**

Run:

```powershell
cd F:\MyFintech\backend
python -m app.services.evaluation.run_eval
```

Expected output contains:

```text
- avg_classifier_regulator_coverage:
- avg_evidence_regulator_coverage:
- avg_regulator_coverage:
```

Expected interpretation:

- `avg_classifier_regulator_coverage` measures classifier/query profile coverage.
- `avg_evidence_regulator_coverage` measures actual retrieved evidence source coverage.
- If classifier coverage is lower than evidence coverage, summary should no longer imply evidence retrieval failed.

- [ ] **Step 5: Commit docs or benchmark updates**

```powershell
cd F:\MyFintech
git add backend/data/evaluation/benchmark_questions.json docs/evaluation_protocol.md
git commit -m "docs: document regulator coverage evaluation split"
```

Skip this commit if neither file changed.

---

## Acceptance Criteria

- AI wealth advisory/product launch classifier output includes regulators `HKMA`, `SFC`, `PCPD`.
- AI wealth advisory/product launch classifier output includes topics `consumer_protection`, `suitability`, `personal_data`, and `wealth_management`.
- Query planner expands the same intent into protected retrieval terms for `HKMA`, `SFC`, `PCPD`, `consumer protection`, `suitability`, `personal data`, and `wealth management`.
- Evaluation rows expose both `classifier_regulator_coverage` and `evidence_regulator_coverage`.
- Evaluation summary exposes both `avg_classifier_regulator_coverage` and `avg_evidence_regulator_coverage`.
- DeepResearch sub-question retrieval applies a regulator diversity gate after ranked retrieval and before evidence evaluation.
- DeepResearch diversity gate preserves original rank when no alternative regulator evidence exists.
- Targeted tests pass:

```powershell
cd F:\MyFintech\backend
python -m pytest tests/test_query_classifier.py tests/test_query_planner.py tests/test_deepresearch.py tests/test_evaluation_error_reporting.py -q
```

- Full backend tests pass:

```powershell
cd F:\MyFintech\backend
python -m pytest -q
```

- Deterministic eval prints both classifier and evidence regulator coverage metrics:

```powershell
cd F:\MyFintech\backend
python -m app.services.evaluation.run_eval
```

## Rollback Plan

- Revert classifier/planner expansion commits if routing or recall worsens.
- Keep evaluation metric split if possible; it is diagnostic and should remain useful even if expansions roll back.
- Disable DeepResearch gate by reverting only the workflow/eval gate commits; retrieval will return to rank-only selection.
- If a partial rollback is needed, prefer this order:
  1. Roll back DeepResearch diversity gate.
  2. Roll back planner expansions.
  3. Roll back classifier expansions.
  4. Keep or separately evaluate the metric split.

## Self-Review Notes

- Spec coverage: the plan covers all requested items: classifier/planner regulatory expansion, split evaluation metrics, and DeepResearch regulator diversity.
- Completeness scan: every task contains concrete files, commands, and expected outcomes.
- Type consistency: new helper signatures are defined before tasks consume them.
- Scope control: no frontend, database, model provider, or external search changes are included.
