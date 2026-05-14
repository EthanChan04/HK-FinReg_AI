# Compliance Copilot

## Product Positioning

Compliance Copilot is the natural-language operating layer for HK-FinReg AI.
It is a cross-workspace assistant for bank compliance workflows, not a standalone board.

It helps teams:
- Recommend the right workflow and workspace.
- Answer evidence-backed regulatory questions.
- Explain current case evidence gaps and low-confidence outputs.
- Map regulator-obligation-risk-control paths.
- Escalate complex cross-regulator analysis to Deep Research.
- Support Human Review with reviewer-oriented drafting.

## Bilingual Output Contract

Every final answer must contain these two sections:

```text
## 绻侀珨涓枃
...

## English
...
```

Rules:
- Traditional Chinese first, English second.
- Both sections must reflect the same evidence and citation IDs.
- If evidence is insufficient, explicitly state that in both sections.

## Model Requirement

Compliance Copilot uses `MiMo-v2.5` via OpenAI-compatible endpoint settings:
- `COPILOT_MODEL`
- `COPILOT_BASE_URL`
- `COPILOT_API_KEY` (fallback to `ZHIPU_API_KEY`)

## Routing Policy

Intent routes are explicit backend logic:
- `regulatory_qa` -> RAG
- `case_explanation` -> case context + optional RAG
- `obligation_mapping` -> RAG + KAG
- `workflow_recommendation` -> workflow router
- `deep_research` -> Deep Research workflow
- `human_review_help` -> human review context
- `smalltalk_or_help` -> MiMo direct guidance

## API

Streaming endpoint:
- `POST /api/v1/copilot/chat/stream`

SSE events:
- `intent`
- `tool_call`
- `evidence`
- `graph`
- `token`
- `citation_audit`
- `done`

## Safety Limits

Compliance Copilot must:
- Not provide final customer/product/transaction approval or rejection.
- Not claim to provide legal advice.
- Surface evidence insufficiency clearly.
- Recommend human review when confidence is low.
- Prefer source-grounded statements with citation verification.
