# Compliance Copilot Chat Bot Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a context-aware Chat Bot, named **Compliance Copilot**, to HK-FinReg AI so bank users can ask regulatory questions, explain current cases, choose workflows, and escalate complex questions to KAG or Deep Research with bilingual Traditional Chinese and English output.

**Architecture:** Add a new backend `copilot` service and streaming API that uses `MiMo-v2.5` as the chat model, classifies the user's intent, builds page/workflow context, routes to RAG, KAG, Deep Research, or Human Review tools, and returns grounded SSE responses. Add a frontend chat panel that is available across all bank workspaces and can include the current workflow, report, evidence chunks, graph paths, confidence data, and review state as context.

**Tech Stack:** FastAPI, Pydantic, LangChain `ChatOpenAI` compatible client, `MiMo-v2.5`, Hybrid RAG, KAG, DeepResearch, SSE streaming, Next.js, TypeScript, existing `bankWorkspaces`, existing `useAgentStream`, pytest, frontend lint/build.

---

## 1. Product Positioning

The Chat Bot should not be a generic FAQ bot. It should be a **bank compliance workflow copilot**.

Recommended user-facing name:

```text
Compliance Copilot
```

Primary role:

```text
Natural-language operating layer for HK-FinReg AI.
```

It should help users:

- Find the right workspace and workflow.
- Ask evidence-backed regulatory questions.
- Explain current compliance reports.
- Explain why a case is low confidence or missing evidence.
- Map scenarios to regulators, obligations, risks, and controls.
- Generate working drafts such as action lists, reviewer notes, management summaries, and memo outlines.
- Escalate complex product launch, AI governance, policy impact, and multi-party investigation questions to Deep Research.

The Chat Bot should appear as a cross-workspace assistant, not as a seventh top-level board.

---

## 2. Output Language Requirement

Every final Chat Bot answer must be bilingual:

```text
## 繁體中文
[Traditional Chinese answer]

## English
[English answer]
```

Rules:

- Traditional Chinese must appear first.
- English must appear second.
- Regulatory terms may keep common English abbreviations, such as AML, CDD, EDD, KYC, HKMA, SFC, PCPD, SVF, STR, SAR.
- If evidence is insufficient, both language sections must say so clearly.
- If citations are used, both language sections should refer to the same source identifiers.
- The model must not silently switch to Simplified Chinese.

Recommended system instruction:

```text
You are Compliance Copilot for a Hong Kong bank internal regulatory intelligence platform.
Always answer in two sections:

## 繁體中文
Use professional Traditional Chinese suitable for Hong Kong banking compliance teams.

## English
Provide the equivalent professional English answer.

Do not provide final compliance approval. Distinguish regulatory facts, analysis, evidence gaps, and recommended next steps. If evidence is insufficient, say so explicitly. Prefer concise, audit-friendly answers with source references when evidence is available.
```

---

## 3. Model Requirement

The Chat Bot must use `MiMo-v2.5`.

The current project already has compatible model configuration in `backend/app/core/config.py`:

```python
ZHIPU_MODEL: str = "MiMo-v2.5"
ZHIPU_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
LONGCAT_MODEL: str = "MiMo-v2.5"
LONGCAT_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
```

Implementation should add explicit Copilot settings rather than overloading unrelated model names:

```python
COPILOT_MODEL: str = "MiMo-v2.5"
COPILOT_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
COPILOT_API_KEY: str = ""
COPILOT_TIMEOUT_SECONDS: int = 60
COPILOT_MAX_CONTEXT_CHARS: int = 16000
COPILOT_MAX_HISTORY_MESSAGES: int = 8
```

Fallback rule:

- Prefer `COPILOT_API_KEY`.
- If `COPILOT_API_KEY` is empty, fallback to `ZHIPU_API_KEY`.
- If both are empty, return a safe backend error explaining that Copilot model credentials are not configured.

Do not use external web browsing for the first implementation. The Chat Bot should rely on the internal regulatory corpus, KAG graph, current case context, and Deep Research workflow.

---

## 4. User Scenarios

### 4.1 Workflow Recommendation

Example:

```text
我們準備推出 AI 信貸評分工具，會使用銀行流水及外部資料，應該做哪種審查？
```

Expected behavior:

- Classify as product launch / AI governance.
- Recommend `Product & Business Launch Review`.
- Explain that Deep Research is appropriate.
- Offer to start the `product-launch-review` or `ai-governance-review` workflow.

Engine:

```text
Intent classification + workflow routing
```

### 4.2 Regulatory Q&A

Example:

```text
HKMA 對非面對面開戶和 eKYC 有甚麼主要要求？
```

Expected behavior:

- Use RAG.
- Return source-grounded bilingual answer.
- Include evidence cards and citation identifiers.
- Avoid broad legal conclusions if evidence is weak.

Engine:

```text
Hybrid RAG + citation verification
```

### 4.3 Obligation and Risk Path Explanation

Example:

```text
為甚麼 AI onboarding 會同時涉及 HKMA 和 PCPD？
```

Expected behavior:

- Use RAG + KAG.
- Explain regulator, obligation, risk, and control relationship.
- Return graph paths if available.

Engine:

```text
RAG + KAG obligation mapper / graph retriever
```

### 4.4 Current Report Follow-up

Example:

```text
這份報告哪些地方證據不足？可以幫我生成 RM 補件清單嗎？
```

Expected behavior:

- Use current `reportText`, `evidenceChunks`, confidence data, and gate status from frontend context.
- Identify weak or missing evidence.
- Generate bilingual action list.

Engine:

```text
Current case context + RAG evidence grounding
```

### 4.5 Deep Research Escalation

Example:

```text
比較 HKMA、SFC 和 PCPD 對生成式 AI 在金融機構使用上的要求，並生成管理層 memo。
```

Expected behavior:

- Classify as regulatory memo / cross-regulator analysis.
- Ask for or automatically trigger Deep Research depending on UX decision.
- Return Deep Research plan, evidence gaps, final bilingual summary, and citation audit.

Engine:

```text
Deep Research
```

### 4.6 Human Review Assistance

Example:

```text
這個 case 為甚麼進入人工覆核？reviewer 應該看甚麼？
```

Expected behavior:

- Use current gate and review queue context.
- Explain low confidence, missing evidence, or manual approval trigger.
- Draft reviewer notes without approving the case.

Engine:

```text
Human Review context + RAG/KAG explanation
```

---

## 5. Engine Routing Policy

Implement routing as explicit backend logic, not hidden inside prompts.

```text
regulatory_qa
  -> RAG

case_explanation
  -> current case context + RAG

obligation_mapping
  -> RAG + KAG

workflow_recommendation
  -> intent classifier + bank workspace metadata

deep_research
  -> Deep Research

human_review_help
  -> review context + optional RAG/KAG

smalltalk_or_help
  -> MiMo-v2.5 direct answer with product guidance
```

Routing should return one of:

```python
CopilotIntent = Literal[
    "regulatory_qa",
    "case_explanation",
    "obligation_mapping",
    "workflow_recommendation",
    "deep_research",
    "human_review_help",
    "smalltalk_or_help",
]
```

The first implementation can use deterministic rules before adding LLM classification:

- Mentions of "memo", "compare", "impact", "launch", "AI governance", "policy change" -> `deep_research`.
- Mentions of "why", "obligation", "risk path", "control", "regulator applies" -> `obligation_mapping`.
- Mentions of "current report", "this case", "evidence insufficient", "low confidence" -> `case_explanation`.
- Mentions of "which module", "which workflow", "where should I go" -> `workflow_recommendation`.
- Mentions of "review", "approve", "reject", "pending" -> `human_review_help`.
- Otherwise -> `regulatory_qa`.

---

## 6. Backend Design

### 6.1 New Files

Create:

```text
backend/app/api/routers/copilot.py
backend/app/schemas/copilot.py
backend/app/services/copilot/__init__.py
backend/app/services/copilot/model.py
backend/app/services/copilot/intent_classifier.py
backend/app/services/copilot/context_builder.py
backend/app/services/copilot/tool_router.py
backend/app/services/copilot/response_writer.py
backend/app/services/copilot/guardrails.py
```

### 6.2 Modify Existing Files

Modify:

```text
backend/app/core/config.py
backend/app/main.py
backend/tests/
.env.example
README.md
```

### 6.3 API Endpoint

Add:

```text
POST /api/v1/copilot/chat/stream
```

Request schema:

```python
class CopilotMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class CopilotCaseContext(BaseModel):
    workspace_id: str | None = None
    workflow_id: str | None = None
    workflow_name: str | None = None
    input_text: str | None = None
    report_text: str | None = None
    evidence_chunks: list[dict] = Field(default_factory=list)
    graph_paths: list[dict] = Field(default_factory=list)
    research_plan: dict | None = None
    confidence_data: dict = Field(default_factory=dict)
    workflow_run_id: str | None = None
    current_gate: str | None = None
    gate_message: str | None = None

class CopilotChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    conversation_id: str | None = None
    history: list[CopilotMessage] = Field(default_factory=list)
    case_context: CopilotCaseContext = Field(default_factory=CopilotCaseContext)
    preferred_language: Literal["zh-HK+en"] = "zh-HK+en"
```

Response is SSE:

```text
event: intent
data: {"intent":"regulatory_qa","engine":"rag"}

event: tool_call
data: {"tool":"retrieval","status":"running"}

event: evidence
data: {"evidence_chunks":[...]}

event: graph
data: {"graph_paths":[...]}

event: token
data: {"text":"..."}

event: citation_audit
data: {"unsupported_claim_rate":0.0}

event: done
data: {"conversation_id":"...","intent":"...","engine":"..."}
```

### 6.4 Model Builder

`backend/app/services/copilot/model.py` should create a dedicated Copilot LLM:

```python
from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.core.config import get_settings

@lru_cache()
def build_copilot_llm() -> ChatOpenAI:
    settings = get_settings()
    api_key = settings.COPILOT_API_KEY or settings.ZHIPU_API_KEY
    if not api_key:
        raise RuntimeError("COPILOT_API_KEY or ZHIPU_API_KEY must be configured for Compliance Copilot.")
    return ChatOpenAI(
        model_name=settings.COPILOT_MODEL,
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=settings.COPILOT_BASE_URL,
        timeout=settings.COPILOT_TIMEOUT_SECONDS,
    )
```

### 6.5 Context Builder

`context_builder.py` should:

- Limit history to `COPILOT_MAX_HISTORY_MESSAGES`.
- Trim current report and input to fit `COPILOT_MAX_CONTEXT_CHARS`.
- Include only compact evidence metadata and short snippets.
- Include graph paths as short relationship paths.
- Include confidence and human review gate fields.
- Never include secrets, API keys, or raw environment variables.

### 6.6 Tool Router

`tool_router.py` should call existing services:

- RAG:
  - `build_reranked_retriever()`
  - `RetrievalService`
- KAG:
  - `NetworkXGraphStore`
  - `GraphRetriever`
  - `ObligationMapper`
- Deep Research:
  - `build_deepresearch_graph()`
- Human Review:
  - Use current context first.
  - Later can integrate `/review-queue` data if needed.

### 6.7 Response Writer

`response_writer.py` should enforce the bilingual answer contract:

```text
## 繁體中文
1. 摘要
2. 監管依據
3. 風險 / 義務 / 控制分析
4. 證據缺口
5. 建議下一步
6. 置信度 / 限制

## English
1. Summary
2. Regulatory Basis
3. Risk / Obligation / Control Analysis
4. Evidence Gaps
5. Recommended Next Steps
6. Confidence / Limitations
```

For simple questions, shorter answers are acceptable, but the two sections are still mandatory.

### 6.8 Guardrails

`guardrails.py` should enforce:

- No final approval or rejection of customers, transactions, or products.
- No legal advice claim.
- Evidence insufficiency must be explicit.
- Cite retrieved sources when regulatory facts are stated.
- If confidence is low, recommend human review.
- The answer must include `## 繁體中文` and `## English`.

---

## 7. Frontend Design

### 7.1 New Files

Create:

```text
frontend/src/components/chat/ComplianceCopilot.tsx
frontend/src/components/chat/ChatMessageList.tsx
frontend/src/components/chat/ChatInput.tsx
frontend/src/components/chat/SuggestedPrompts.tsx
frontend/src/components/chat/ToolCallTimeline.tsx
frontend/src/components/chat/CitationCards.tsx
frontend/src/hooks/useCopilotChat.ts
frontend/src/lib/copilotPrompts.ts
```

### 7.2 Modify Existing Files

Modify:

```text
frontend/src/app/page.tsx
frontend/src/types/index.ts
```

### 7.3 UI Placement

Add a right-side collapsible panel:

```text
Main Workspace
  left: workflow selector + input
  center: report/evidence/graph/research plan
  right: Compliance Copilot
```

If screen width is small, Copilot should become a floating drawer.

### 7.4 Context Passed From Page

`page.tsx` should pass:

```ts
{
  workspace_id: activeBoardId,
  workflow_id: currentModule.id,
  workflow_name: currentModule.name,
  input_text: inputText,
  report_text: stream.reportText,
  evidence_chunks: stream.evidenceChunks,
  graph_paths: stream.graphPaths,
  research_plan: stream.researchPlan,
  confidence_data: {
    retrieval: stream.confidenceScore,
    reasoning: stream.reasoningConfidence,
    reviewer: stream.reviewerConfidence,
    cross_validation_passed: stream.crossValidationPassed,
  },
  workflow_run_id: stream.workflowRunId,
  current_gate: stream.currentGate,
  gate_message: stream.gateMessage,
}
```

### 7.5 Suggested Prompts

Show context-aware prompt chips:

Customer & Account Compliance:

```text
解釋此客戶的主要 KYC 風險 / Explain this customer's key KYC risks
哪些資料不足？ / What information is missing?
生成 RM 補件清單 / Draft an RM follow-up list
```

Transaction & Payment Compliance:

```text
這筆交易為何可疑？ / Why is this transaction suspicious?
是否需要升級 EDD？ / Should this be escalated to EDD?
生成 STR/SAR 草稿重點 / Draft key STR/SAR points
```

Product & Business Launch Review:

```text
是否需要 Deep Research？ / Should this use Deep Research?
涉及哪些監管機構？ / Which regulators are involved?
生成上線前 checklist / Generate a pre-launch checklist
```

Regulatory Research & Policy Change:

```text
生成管理層摘要 / Generate a management summary
比較 HKMA、SFC、PCPD 要求 / Compare HKMA, SFC, and PCPD expectations
指出證據缺口 / Identify evidence gaps
```

Human Review & Audit:

```text
為何進入人工覆核？ / Why was this sent to human review?
生成 reviewer notes / Draft reviewer notes
哪些結論置信度較低？ / Which conclusions have lower confidence?
```

Knowledge Base:

```text
解釋義務映射 / Explain the obligation mapping
哪些監管文件支持此結論？ / Which regulatory documents support this?
顯示風險與控制路徑 / Show risk-control paths
```

---

## 8. Frontend Types

Add to `frontend/src/types/index.ts`:

```ts
export interface CopilotMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
}

export type CopilotIntent =
  | "regulatory_qa"
  | "case_explanation"
  | "obligation_mapping"
  | "workflow_recommendation"
  | "deep_research"
  | "human_review_help"
  | "smalltalk_or_help";

export interface CopilotCaseContext {
  workspace_id?: string | null;
  workflow_id?: string | null;
  workflow_name?: string | null;
  input_text?: string | null;
  report_text?: string | null;
  evidence_chunks?: EvidenceChunk[];
  graph_paths?: Array<{ path: string[]; matched_node?: string; matched_topics?: string[] }>;
  research_plan?: ResearchPlan | null;
  confidence_data?: Record<string, unknown>;
  workflow_run_id?: string | null;
  current_gate?: string | null;
  gate_message?: string | null;
}

export interface CopilotToolEvent {
  tool: "rag" | "kag" | "deepresearch" | "human_review" | "workflow_router" | "mimo";
  status: "running" | "done" | "error";
  message?: string;
}
```

---

## 9. Implementation Tasks

### Task 1: Add Copilot Config

Files:

- Modify: `backend/app/core/config.py`
- Modify: `.env.example`

Steps:

- [ ] Add `COPILOT_MODEL`, `COPILOT_BASE_URL`, `COPILOT_API_KEY`, `COPILOT_TIMEOUT_SECONDS`, `COPILOT_MAX_CONTEXT_CHARS`, and `COPILOT_MAX_HISTORY_MESSAGES`.
- [ ] Set default model to `MiMo-v2.5`.
- [ ] Set default base URL to the existing MIMO-compatible base URL.
- [ ] Document the variables in `.env.example`.
- [ ] Run `python -m pytest backend/tests/test_* -q` or at least import config with `python -c "from app.core.config import get_settings; print(get_settings().COPILOT_MODEL)"` from `backend`.

### Task 2: Add Copilot Schemas

Files:

- Create: `backend/app/schemas/copilot.py`

Steps:

- [ ] Add `CopilotMessage`.
- [ ] Add `CopilotCaseContext`.
- [ ] Add `CopilotChatRequest`.
- [ ] Add `CopilotIntentEvent`, `CopilotToolEvent`, and simple response helper models if useful.
- [ ] Add tests for max message length and default `preferred_language`.

### Task 3: Add Intent Classifier

Files:

- Create: `backend/app/services/copilot/intent_classifier.py`
- Test: `backend/tests/test_copilot_intent_classifier.py`

Steps:

- [ ] Implement deterministic classifier first.
- [ ] Cover regulatory Q&A, case explanation, obligation mapping, workflow recommendation, deep research, human review, and smalltalk/help.
- [ ] Keep the classifier explainable by returning `{intent, engine, reason}`.

### Task 4: Add Context Builder

Files:

- Create: `backend/app/services/copilot/context_builder.py`
- Test: `backend/tests/test_copilot_context_builder.py`

Steps:

- [ ] Trim history to `COPILOT_MAX_HISTORY_MESSAGES`.
- [ ] Trim report/input/evidence to `COPILOT_MAX_CONTEXT_CHARS`.
- [ ] Preserve evidence IDs, regulator, title, page, section title, and short text snippets.
- [ ] Exclude secrets and raw environment values.

### Task 5: Add Tool Router

Files:

- Create: `backend/app/services/copilot/tool_router.py`
- Test: `backend/tests/test_copilot_tool_router.py`

Steps:

- [ ] For `regulatory_qa`, call existing retrieval service.
- [ ] For `obligation_mapping`, call existing KAG obligation mapper or graph retriever.
- [ ] For `deep_research`, call `build_deepresearch_graph()`.
- [ ] For `case_explanation`, use provided context first and augment with RAG only if required.
- [ ] For `workflow_recommendation`, return bank workspace recommendation metadata without LLM dependency.
- [ ] For `human_review_help`, use gate and confidence context to draft review guidance.

### Task 6: Add Bilingual Response Writer

Files:

- Create: `backend/app/services/copilot/response_writer.py`
- Create: `backend/app/services/copilot/guardrails.py`
- Test: `backend/tests/test_copilot_response_writer.py`

Steps:

- [ ] Build MiMo-v2.5 prompt with mandatory Traditional Chinese + English output.
- [ ] Include retrieved evidence and KAG paths as grounded context.
- [ ] Ensure final output contains `## 繁體中文` and `## English`.
- [ ] Add fallback repair if one section is missing.
- [ ] Add guardrail checks for approval/legal-advice language.

### Task 7: Add Streaming API Router

Files:

- Create: `backend/app/api/routers/copilot.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_copilot_api.py`

Steps:

- [ ] Add `POST /api/v1/copilot/chat/stream`.
- [ ] Stream `intent`, `tool_call`, `evidence`, `graph`, `token`, `citation_audit`, and `done` events.
- [ ] Include API key dependency like other protected routers.
- [ ] Mount router in `main.py`.
- [ ] Test that missing model credentials return safe error.

### Task 8: Add Frontend Copilot Hook

Files:

- Create: `frontend/src/hooks/useCopilotChat.ts`
- Modify: `frontend/src/types/index.ts`

Steps:

- [ ] Add Copilot types.
- [ ] Implement SSE streaming parser for `/api/v1/copilot/chat/stream`.
- [ ] Store messages, tool events, evidence cards, intent, error, and loading state.
- [ ] Support cancel/reset.

### Task 9: Add Frontend Chat Components

Files:

- Create: `frontend/src/components/chat/ComplianceCopilot.tsx`
- Create: `frontend/src/components/chat/ChatMessageList.tsx`
- Create: `frontend/src/components/chat/ChatInput.tsx`
- Create: `frontend/src/components/chat/SuggestedPrompts.tsx`
- Create: `frontend/src/components/chat/ToolCallTimeline.tsx`
- Create: `frontend/src/components/chat/CitationCards.tsx`
- Create: `frontend/src/lib/copilotPrompts.ts`

Steps:

- [ ] Render assistant panel with message list.
- [ ] Add suggested prompts based on active board/workflow.
- [ ] Show engine/tool activity.
- [ ] Show citations/evidence returned by backend.
- [ ] Keep styling consistent with current dark professional bank UI.

### Task 10: Integrate Copilot Into Main Page

Files:

- Modify: `frontend/src/app/page.tsx`

Steps:

- [ ] Import `ComplianceCopilot`.
- [ ] Build `CopilotCaseContext` from current workflow and stream state.
- [ ] Add right-side collapsible panel on desktop.
- [ ] Add drawer/floating entry for smaller screens if feasible.
- [ ] Ensure the existing workflow submit path remains unchanged.

### Task 11: Documentation

Files:

- Modify: `README.md`
- Create: `docs/product/compliance-copilot.md`

Steps:

- [ ] Document Compliance Copilot positioning.
- [ ] Document bilingual output rule.
- [ ] Document MiMo-v2.5 model requirement.
- [ ] Document RAG/KAG/Deep Research routing.
- [ ] Document safety limits: no final approval, no legal advice, cite evidence.

### Task 12: Verification

Commands:

```powershell
python -m pytest backend/tests/test_copilot_intent_classifier.py backend/tests/test_copilot_context_builder.py backend/tests/test_copilot_response_writer.py backend/tests/test_copilot_api.py -q
cd frontend
npm run lint
npm run build
```

Expected:

- Backend Copilot tests pass.
- Frontend lint passes.
- Frontend build passes.
- Manual test: ask one regulatory Q&A question and verify output has `## 繁體中文` and `## English`.
- Manual test: ask one product launch question and verify Deep Research routing is selected.
- Manual test: ask about current report evidence gaps and verify case context is used.

---

## 10. Acceptance Criteria

- Compliance Copilot is visible in the app without replacing existing workflows.
- Copilot uses `MiMo-v2.5` through the configured OpenAI-compatible base URL.
- Every final answer is bilingual: Traditional Chinese first, English second.
- Regulatory answers are grounded in RAG evidence when available.
- Obligation/risk/control questions can use KAG.
- Complex product launch, AI governance, regulatory memo, and policy impact questions can route to Deep Research.
- Current report follow-up questions can use active workflow context.
- Human review questions explain the gate and recommend reviewer actions without approving or rejecting the case.
- The backend streams structured events for intent, tools, evidence, graph, tokens, citation audit, and completion.
- The system refuses to provide unsupported final legal/compliance approval.

---

## 11. Non-Goals for First Version

- Do not add internet search.
- Do not add long-term memory.
- Do not add multi-user collaboration.
- Do not allow Copilot to approve or reject cases.
- Do not let Copilot write directly to the regulatory knowledge base.
- Do not replace the existing workspace workflow UI.
- Do not introduce a large new frontend state library.

---

## 12. Recommended MVP Scope

Build the MVP in this order:

1. Backend Copilot schemas, config, intent classifier, and model builder.
2. RAG-backed regulatory Q&A with bilingual output.
3. Current report follow-up with context injection.
4. KAG obligation/risk/control explanation.
5. Deep Research escalation for product launch and regulatory memo.
6. Frontend right-side Copilot panel with suggested prompts and streaming output.

This keeps the first release useful without turning the Chat Bot into a sprawling autonomous agent too early.

---

## 13. Self-Review

- The plan is aligned with the current bank workspace architecture.
- The plan reuses existing RAG, KAG, DeepResearch, SSE, and MiMo-v2.5-compatible configuration.
- The plan makes bilingual Traditional Chinese + English output mandatory.
- The plan avoids duplicating existing workflow modules.
- The plan includes backend, frontend, documentation, tests, and verification.
- The plan has no placeholder implementation sections.
