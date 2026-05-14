# Hong Kong Bank ToB Module Rearchitecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize HK-FinReg AI from a demo-style list of 10 overlapping modules into a Hong Kong bank internal ToB compliance platform with fewer task-oriented boards and explicit RAG, KAG, and Deep Research routing.

**Architecture:** Keep the existing FastAPI + Next.js foundation. Replace the flat frontend module list with banking workspaces, add a lightweight frontend routing model that chooses the correct existing backend endpoint, and introduce backend-facing documentation/tests for when to use RAG-only, RAG+KAG, or RAG+KAG+Deep Research. Do not delete existing backend routers in the first pass; wrap and relabel them so the platform stays functional while the product information architecture improves.

**Tech Stack:** Next.js, TypeScript, FastAPI, Pydantic, Hybrid RAG, KAG, DeepResearch, SSE streaming, pytest, Next.js lint/build.

---

## Product Decision

The target product is a bank-internal ToB system for Hong Kong banks such as ZA Bank, Bank of China Hong Kong, HSBC, Hang Seng, Standard Chartered, and virtual banks. It should be positioned as:

```text
Regulatory Intelligence & Compliance Operations Platform
```

The platform should help bank teams complete:

- Customer and account compliance review.
- Transaction and payment compliance review.
- Product and business launch compliance assessment.
- Regulatory research and policy change impact analysis.
- Human review, audit trail, and model output governance.
- Regulatory knowledge base management.

The platform should not expose RAG, KAG, and Deep Research as separate user-facing products. These are reasoning engines selected behind each workflow.

## Target Information Architecture

Replace the current flat navigation with these six first-level boards:

```text
Dashboard
Customer & Account Compliance
Transaction & Payment Compliance
Product & Business Launch Review
Regulatory Research & Policy Change
Human Review & Audit
Regulatory Knowledge Base
```

### Board 1: Customer & Account Compliance

Primary users:

- KYC operations.
- AML compliance analysts.
- Relationship managers.
- Account opening reviewers.

Submodules:

- Account Opening / KYC Review.
- Customer Risk Rating.
- PEP / Sanctions / Adverse Media Risk Summary.
- Customer Information Gap Check.
- Enhanced Due Diligence Recommendation.

Engine policy:

- Default: RAG.
- Upgrade to KAG when customer attributes must be mapped to obligations, risk factors, and controls.
- Upgrade to Deep Research only for complex customers, multi-layer ownership, trust structures, PEP association, cross-border wealth source, or high-risk institutional accounts.

Existing module mapping:

- `bank` remains the primary endpoint for account review.
- Relevant customer due diligence parts of `svf` should be folded into this board as a scenario type, not kept as a separate first-level module.

### Board 2: Transaction & Payment Compliance

Primary users:

- Transaction monitoring team.
- Payment operations.
- AML investigation team.
- Cross-border payment team.

Submodules:

- Cross-Border Payment Compliance Assessment.
- Unusual Transaction Explanation.
- STR / SAR Drafting Assistant.
- Sanctions and High-Risk Jurisdiction Check.
- Payment Scenario AML Review.

Engine policy:

- Default: RAG for standard regulatory evidence retrieval.
- Upgrade to KAG when a transaction pattern must be mapped to risk typologies, obligations, evidence requirements, and control actions.
- Upgrade to Deep Research for complex investigations involving multiple parties, multiple accounts, multiple jurisdictions, layered flows, or repeated transaction patterns.

Existing module mapping:

- `crossborder` remains the primary endpoint.
- SVF payment scenarios should become a transaction scenario type, not a separate top-level module.

### Board 3: Product & Business Launch Review

Primary users:

- Product managers.
- Compliance advisory.
- Legal.
- Model risk management.
- Data protection officers.
- AI governance teams.

Submodules:

- New Product Launch Compliance Review.
- AI / GenAI Financial Use Case Review.
- Data and Privacy Impact Review.
- Third-Party / Outsourcing Compliance Check.
- Launch Checklist Generation.

Engine policy:

- Default: RAG + KAG.
- Use Deep Research for most complete product launch assessments because launch review is multi-dimensional: product type, customer segment, data use, AI use, cross-border flow, outsourcing, consumer protection, AML, privacy, and licensing.

Existing module mapping:

- Merge `ai-governance` into this board as a scenario.
- Merge `product-launch` into this board as the default workflow.
- Use `/api/v1/research/analyze` with `task_type: "product_launch_review"` or `task_type: "ai_governance_review"`.

### Board 4: Regulatory Research & Policy Change

Primary users:

- Compliance policy team.
- Legal team.
- Regulatory affairs.
- Senior management.

Submodules:

- Regulatory Memo Generation.
- Policy Change Impact Analysis.
- Cross-Regulator Comparison.
- Existing Process Impact Assessment.
- Management Brief / Board Paper Assistant.

Engine policy:

- Default: RAG + KAG.
- Use Deep Research for memo generation, cross-regulator comparison, and regulatory change impact.

Existing module mapping:

- Merge `reg-memo` and `change-impact`.
- Use `/api/v1/research/analyze` with `task_type: "regulatory_memo"`, `task_type: "cross_regulator_analysis"`, or `task_type: "regulatory_change_impact"`.

### Board 5: Human Review & Audit

Primary users:

- Compliance managers.
- Second-line reviewers.
- Internal audit.
- Model governance reviewers.

Submodules:

- Pending Human Review Queue.
- Low Confidence Cases.
- Missing Evidence Cases.
- Report Approval / Rejection.
- Audit Trail.
- Model Output Quality Monitoring.

Engine policy:

- Default: no new LLM call required.
- Use RAG to display supporting evidence.
- Use KAG to explain obligation/risk/control paths when a reviewer asks why a case was flagged.
- Use Deep Research only when a reviewer explicitly requests further investigation.

Existing module mapping:

- Keep `/api/v1/review-queue/*`.
- Surface it as a first-class board instead of hiding it inside stream status.

### Board 6: Regulatory Knowledge Base

Primary users:

- Compliance knowledge managers.
- RegTech administrators.
- Model governance.
- Engineering support.

Submodules:

- Regulatory Document Library.
- Obligation Map.
- Knowledge Graph Explorer.
- Regulator / Document / Obligation / Risk / Control Relationships.
- Citation and Version Management.
- Golden Cases and Regression Gates.

Engine policy:

- RAG manages document retrieval and source evidence.
- KAG is the core engine for obligations, risks, controls, products, and regulator relationships.
- Deep Research may suggest knowledge updates, but production knowledge updates require human approval.

Existing module mapping:

- Move `obligation-map` and `graph-explorer` into this board.
- Keep `/api/v1/kag/obligation-map` and `/api/v1/kag/graph/search`.

---

## Engine Routing Rules

Codex should implement these rules as explicit configuration, not scattered `if` statements.

```text
Routine scenario review
  -> RAG

Scenario review with obligation/risk/control mapping
  -> RAG + KAG

Product launch, AI governance, policy impact, regulatory memo
  -> RAG + KAG + Deep Research

Low confidence, missing evidence, manual approval
  -> Human Review
```

Required routing metadata:

```ts
export type EngineMode = "rag" | "rag_kag" | "deepresearch" | "human_review";

export type BankBoardId =
  | "dashboard"
  | "customer-account"
  | "transaction-payment"
  | "product-launch"
  | "regulatory-research"
  | "human-review"
  | "knowledge-base";
```

---

## File Structure

### Create

- `frontend/src/lib/bankWorkspaces.ts`
  - Owns the new bank ToB board and workflow configuration.
- `frontend/src/lib/engineRouting.ts`
  - Owns engine mode types and payload construction rules.
- `frontend/src/components/WorkspaceNav.tsx`
  - Renders first-level bank boards.
- `frontend/src/components/WorkflowSelector.tsx`
  - Renders workflows inside the selected board.
- `docs/product/bank-tob-information-architecture.md`
  - Product-facing explanation of boards, users, module mapping, and engine routing.

### Modify

- `frontend/src/types/index.ts`
  - Add `EngineMode`, `BankBoardId`, `BankWorkflowConfig`, and update module typing.
- `frontend/src/lib/modules.ts`
  - Stop being the product source of truth. Either re-export flattened workflows from `bankWorkspaces.ts` for backward compatibility or delete only after all imports move.
- `frontend/src/hooks/useAgentStream.ts`
  - Move payload construction to `buildWorkflowPayload()` from `engineRouting.ts`.
- `frontend/src/app/page.tsx`
  - Replace flat module navigation with board navigation plus workflow selector.
- `README.md`
  - Update positioning from “10 modules” to bank ToB boards.

### Test

- `frontend/src/lib/engineRouting.test.ts`
  - Unit tests for payload routing.
- `frontend/src/lib/bankWorkspaces.test.ts`
  - Unit tests that every workflow has board id, endpoint, engine mode, Chinese label, English label, and default input.

If the project has no frontend test runner installed, add a minimal TypeScript validation script instead of introducing a large test framework:

- `frontend/scripts/validate-workspaces.mjs`
  - Imports compiled workspace configuration or reads the source file and validates required fields.
- `frontend/package.json`
  - Add `"validate:workspaces": "node scripts/validate-workspaces.mjs"`.

---

## Task 1: Add Product Architecture Document

**Files:**

- Create: `docs/product/bank-tob-information-architecture.md`

- [ ] **Step 1: Create product docs directory**

Run:

```powershell
New-Item -ItemType Directory -Force -Path docs/product
```

Expected: `docs/product` exists.

- [ ] **Step 2: Create the product architecture document**

Create `docs/product/bank-tob-information-architecture.md` with this content:

```markdown
# HK-FinReg AI Bank ToB Information Architecture

## Positioning

HK-FinReg AI is a Hong Kong bank internal Regulatory Intelligence & Compliance Operations Platform. It supports compliance analysts, KYC teams, AML investigators, product teams, legal teams, regulatory affairs, and model governance teams.

## Core Principle

The product is organized by bank work tasks, not by AI technology. RAG, KAG, and Deep Research are selected behind each workflow.

## Boards

1. Customer & Account Compliance
2. Transaction & Payment Compliance
3. Product & Business Launch Review
4. Regulatory Research & Policy Change
5. Human Review & Audit
6. Regulatory Knowledge Base

## Engine Policy

| Work Type | Engine |
| --- | --- |
| Routine compliance review | RAG |
| Obligation/risk/control mapping | RAG + KAG |
| Product launch, AI governance, policy impact, regulatory memo | RAG + KAG + Deep Research |
| Low confidence or missing evidence | Human Review |

## Legacy Module Mapping

| Legacy Module | New Board |
| --- | --- |
| SVF Compliance | Customer & Account Compliance / Transaction & Payment Compliance |
| Bank Account / KYC Review | Customer & Account Compliance |
| Cross-Border Payment Review | Transaction & Payment Compliance |
| SME Lending Review | Customer & Account Compliance or Product & Business Launch Review, depending on use case |
| AI Governance | Product & Business Launch Review |
| Product Launch | Product & Business Launch Review |
| Regulatory Memo | Regulatory Research & Policy Change |
| Change Impact | Regulatory Research & Policy Change |
| Obligation Map | Regulatory Knowledge Base |
| Graph Explorer | Regulatory Knowledge Base |
```

- [ ] **Step 3: Verify the document exists**

Run:

```powershell
Test-Path docs/product/bank-tob-information-architecture.md
```

Expected: `True`.

- [ ] **Step 4: Commit**

```powershell
git add docs/product/bank-tob-information-architecture.md
git commit -m "docs: define bank tob information architecture"
```

Expected: commit succeeds.

---

## Task 2: Add Strong Types for Bank Boards and Engine Modes

**Files:**

- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Add new type definitions**

Append or insert near the existing `ModuleConfig` definition:

```ts
export type EngineMode = "rag" | "rag_kag" | "deepresearch" | "human_review";

export type BankBoardId =
  | "dashboard"
  | "customer-account"
  | "transaction-payment"
  | "product-launch"
  | "regulatory-research"
  | "human-review"
  | "knowledge-base";

export interface BankBoardConfig {
  id: BankBoardId;
  name: string;
  nameZh: string;
  description: string;
  primaryUsers: string[];
}

export interface BankWorkflowConfig extends ModuleConfig {
  boardId: BankBoardId;
  engineMode: EngineMode;
  description: string;
  primaryUsers: string[];
  scenarioType:
    | "customer_review"
    | "transaction_review"
    | "product_launch"
    | "regulatory_research"
    | "human_review"
    | "knowledge_management";
}
```

- [ ] **Step 2: Run TypeScript check**

Run:

```powershell
cd frontend
npm run lint
```

Expected: existing lint either passes or only reports unrelated pre-existing issues. If TypeScript reports duplicate exported names, move the new types above `ModuleConfig` and ensure names appear once.

- [ ] **Step 3: Commit**

```powershell
git add frontend/src/types/index.ts
git commit -m "feat: add bank workspace and engine mode types"
```

Expected: commit succeeds.

---

## Task 3: Create Bank Workspace Configuration

**Files:**

- Create: `frontend/src/lib/bankWorkspaces.ts`
- Modify: `frontend/src/lib/modules.ts`

- [ ] **Step 1: Create `bankWorkspaces.ts`**

Create `frontend/src/lib/bankWorkspaces.ts`:

```ts
import type { BankBoardConfig, BankWorkflowConfig } from "@/types";

export const bankBoards: BankBoardConfig[] = [
  {
    id: "customer-account",
    name: "Customer & Account Compliance",
    nameZh: "客户与账户合规",
    description: "KYC, CDD, customer risk rating, account opening, and enhanced due diligence workflows.",
    primaryUsers: ["KYC Operations", "AML Compliance", "Relationship Managers", "Account Reviewers"],
  },
  {
    id: "transaction-payment",
    name: "Transaction & Payment Compliance",
    nameZh: "交易与支付合规",
    description: "Cross-border payment, suspicious transaction, sanctions, and payment AML workflows.",
    primaryUsers: ["Transaction Monitoring", "Payment Operations", "AML Investigators"],
  },
  {
    id: "product-launch",
    name: "Product & Business Launch Review",
    nameZh: "产品与业务上线审查",
    description: "New product, AI governance, data privacy, outsourcing, and launch checklist workflows.",
    primaryUsers: ["Product Teams", "Compliance Advisory", "Legal", "Model Risk", "Data Protection"],
  },
  {
    id: "regulatory-research",
    name: "Regulatory Research & Policy Change",
    nameZh: "监管研究与政策变化",
    description: "Regulatory memos, policy impact, cross-regulator comparison, and management brief workflows.",
    primaryUsers: ["Compliance Policy", "Legal", "Regulatory Affairs", "Senior Management"],
  },
  {
    id: "human-review",
    name: "Human Review & Audit",
    nameZh: "人工复核与审计",
    description: "Low-confidence cases, missing evidence cases, approval queue, and audit trail workflows.",
    primaryUsers: ["Compliance Managers", "Second-Line Reviewers", "Internal Audit", "Model Governance"],
  },
  {
    id: "knowledge-base",
    name: "Regulatory Knowledge Base",
    nameZh: "监管知识库",
    description: "Regulatory documents, obligations, knowledge graph, citations, versions, and regression gates.",
    primaryUsers: ["Knowledge Managers", "RegTech Administrators", "Model Governance", "Engineering Support"],
  },
];

export const bankWorkflows: BankWorkflowConfig[] = [
  {
    id: "account-kyc-review",
    boardId: "customer-account",
    name: "Account Opening / KYC Review",
    nameZh: "开户 / KYC 审查",
    description: "Review account opening data against Hong Kong KYC, CDD, and AML expectations.",
    endpoint: "/api/v1/bank-account/verify/stream",
    icon: "KYC",
    status: "production",
    requestKind: "compliance",
    engineMode: "rag",
    scenarioType: "customer_review",
    primaryUsers: ["KYC Operations", "AML Compliance"],
    defaultInput: `Account Opening Application:
Name: Chan Tai Man
ID Type: HKID
Occupation: Restaurant Owner
Monthly Income: HKD 35,000
Source of Wealth: Business Income
Purpose of Account: Business Operations
PEP Status: No
Country of Tax Residence: Hong Kong`,
  },
  {
    id: "customer-risk-rating",
    boardId: "customer-account",
    name: "Customer Risk Rating",
    nameZh: "客户风险评级",
    description: "Assess customer risk factors and map them to due diligence obligations.",
    endpoint: "/api/v1/kag/obligation-map",
    icon: "RISK",
    status: "experimental",
    requestKind: "kag",
    engineMode: "rag_kag",
    scenarioType: "customer_review",
    primaryUsers: ["AML Compliance", "Account Reviewers"],
    defaultInput: "Assess the customer risk factors for a non-face-to-face onboarding case involving eKYC, cross-border source of funds, and a politically exposed close associate.",
  },
  {
    id: "cross-border-payment",
    boardId: "transaction-payment",
    name: "Cross-Border Payment Assessment",
    nameZh: "跨境支付合规评估",
    description: "Assess cross-border payments against AML, sanctions, and payment compliance expectations.",
    endpoint: "/api/v1/cross-border/assess/stream",
    icon: "PAY",
    status: "production",
    requestKind: "compliance",
    engineMode: "rag",
    scenarioType: "transaction_review",
    primaryUsers: ["Transaction Monitoring", "Payment Operations"],
    defaultInput: `Transaction Log:
Sender: Li Wei, PRC Passport E12345678
Beneficiary: Li Jun (brother), HKID A987654(3)
Amount: USD 48,000
Destination: Hong Kong
Origin: Shenzhen, China
Purpose: Family Support
Frequency: Monthly
Bank: China Construction Bank -> HSBC HK`,
  },
  {
    id: "complex-transaction-investigation",
    boardId: "transaction-payment",
    name: "Complex Transaction Investigation",
    nameZh: "复杂交易调查",
    description: "Investigate multi-party or multi-jurisdiction transaction patterns using research workflow.",
    endpoint: "/api/v1/research/analyze",
    icon: "INV",
    status: "experimental",
    requestKind: "research",
    taskType: "routine_review",
    outputFormat: "report",
    engineMode: "deepresearch",
    scenarioType: "transaction_review",
    primaryUsers: ["AML Investigators", "Compliance Managers"],
    defaultInput: "Investigate a pattern of repeated cross-border transfers involving multiple senders, related beneficiaries, high-risk jurisdictions, and unclear source of funds.",
  },
  {
    id: "product-launch-review",
    boardId: "product-launch",
    name: "New Product Launch Review",
    nameZh: "新产品上线合规评估",
    description: "Review new banking or fintech products before launch across licensing, conduct, AML, privacy, and control obligations.",
    endpoint: "/api/v1/research/analyze",
    icon: "LAUNCH",
    status: "experimental",
    requestKind: "research",
    taskType: "product_launch_review",
    outputFormat: "report",
    engineMode: "deepresearch",
    scenarioType: "product_launch",
    primaryUsers: ["Product Teams", "Compliance Advisory", "Legal"],
    defaultInput: "We plan to launch an AI-powered SME credit scoring platform in Hong Kong using bank statements and external data sources. Assess compliance risks.",
  },
  {
    id: "ai-governance-review",
    boardId: "product-launch",
    name: "AI / GenAI Governance Review",
    nameZh: "AI / GenAI 治理审查",
    description: "Assess AI governance, human oversight, privacy, model risk, and customer protection expectations.",
    endpoint: "/api/v1/research/analyze",
    icon: "AI",
    status: "experimental",
    requestKind: "research",
    taskType: "ai_governance_review",
    outputFormat: "report",
    engineMode: "deepresearch",
    scenarioType: "product_launch",
    primaryUsers: ["Model Risk", "Data Protection", "Compliance Advisory"],
    defaultInput: "Assess the compliance obligations for a generative-AI customer service chatbot used in Hong Kong retail banking.",
  },
  {
    id: "regulatory-memo",
    boardId: "regulatory-research",
    name: "Regulatory Memo",
    nameZh: "监管备忘录",
    description: "Generate evidence-backed regulatory memos for internal policy and management use.",
    endpoint: "/api/v1/research/analyze",
    icon: "MEMO",
    status: "experimental",
    requestKind: "research",
    taskType: "regulatory_memo",
    outputFormat: "memo",
    engineMode: "deepresearch",
    scenarioType: "regulatory_research",
    primaryUsers: ["Compliance Policy", "Legal", "Regulatory Affairs"],
    defaultInput: "Compare HKMA, SFC and PCPD expectations on the use of Generative AI in financial institutions.",
  },
  {
    id: "policy-change-impact",
    boardId: "regulatory-research",
    name: "Policy Change Impact",
    nameZh: "监管变化影响分析",
    description: "Analyze how new regulatory guidance affects existing products, controls, and obligations.",
    endpoint: "/api/v1/research/analyze",
    icon: "IMPACT",
    status: "experimental",
    requestKind: "research",
    taskType: "regulatory_change_impact",
    outputFormat: "report",
    engineMode: "deepresearch",
    scenarioType: "regulatory_research",
    primaryUsers: ["Compliance Policy", "Senior Management"],
    defaultInput: "New Source: PCPD AI Model Personal Data Protection Framework. Which existing products, controls, and obligations are affected?",
  },
  {
    id: "review-queue",
    boardId: "human-review",
    name: "Pending Human Review",
    nameZh: "待人工复核",
    description: "Review low-confidence, missing-evidence, and manual-approval compliance cases.",
    endpoint: "/api/v1/review-queue",
    icon: "REVIEW",
    status: "production",
    requestKind: "compliance",
    engineMode: "human_review",
    scenarioType: "human_review",
    primaryUsers: ["Compliance Managers", "Internal Audit"],
    defaultInput: "Open the review queue to inspect pending low-confidence or missing-evidence cases.",
  },
  {
    id: "obligation-map",
    boardId: "knowledge-base",
    name: "Obligation Map",
    nameZh: "监管义务映射",
    description: "Map scenarios to applicable regulators, obligations, risks, controls, and evidence.",
    endpoint: "/api/v1/kag/obligation-map",
    icon: "KAG",
    status: "experimental",
    requestKind: "kag",
    engineMode: "rag_kag",
    scenarioType: "knowledge_management",
    primaryUsers: ["Knowledge Managers", "RegTech Administrators"],
    defaultInput: "A Hong Kong SVF operator plans to use facial recognition eKYC and AI-based transaction monitoring for cross-border payment services.",
  },
  {
    id: "graph-explorer",
    boardId: "knowledge-base",
    name: "Knowledge Graph Explorer",
    nameZh: "监管知识图谱查询",
    description: "Explore regulator, document, obligation, risk, control, product, and customer-segment relationships.",
    endpoint: "/api/v1/kag/graph/search",
    icon: "GRAPH",
    status: "experimental",
    requestKind: "kag",
    engineMode: "rag_kag",
    scenarioType: "knowledge_management",
    primaryUsers: ["RegTech Administrators", "Engineering Support"],
    defaultInput: "Which regulators and obligations apply to AI onboarding in retail banking?",
  },
];

export const defaultWorkflow = bankWorkflows[0];
```

- [ ] **Step 2: Re-export flattened workflows from `modules.ts`**

Replace the contents of `frontend/src/lib/modules.ts` with:

```ts
import { bankWorkflows } from "@/lib/bankWorkspaces";

export const modules = bankWorkflows;
```

- [ ] **Step 3: Run lint**

```powershell
cd frontend
npm run lint
```

Expected: no new type errors from the workspace config.

- [ ] **Step 4: Commit**

```powershell
git add frontend/src/lib/bankWorkspaces.ts frontend/src/lib/modules.ts
git commit -m "feat: add bank workspace configuration"
```

Expected: commit succeeds.

---

## Task 4: Centralize Engine Payload Routing

**Files:**

- Create: `frontend/src/lib/engineRouting.ts`
- Modify: `frontend/src/hooks/useAgentStream.ts`

- [ ] **Step 1: Create `engineRouting.ts`**

Create `frontend/src/lib/engineRouting.ts`:

```ts
import type { BankWorkflowConfig } from "@/types";

export type WorkflowPayload =
  | {
      application_data: string;
      business_context?: string;
      stream_agents_state: boolean;
    }
  | {
      query: string;
      task_type?: BankWorkflowConfig["taskType"];
      output_format?: BankWorkflowConfig["outputFormat"];
      max_iterations?: number;
    };

export function buildWorkflowPayload(
  workflow: BankWorkflowConfig,
  inputText: string
): WorkflowPayload {
  if (workflow.engineMode === "human_review") {
    return {
      application_data: inputText,
      business_context: "human_review_queue",
      stream_agents_state: false,
    };
  }

  if (workflow.requestKind === "research" || workflow.engineMode === "deepresearch") {
    return {
      query: inputText,
      task_type: workflow.taskType ?? "routine_review",
      output_format: workflow.outputFormat ?? "report",
      max_iterations: 3,
    };
  }

  if (workflow.requestKind === "kag" || workflow.engineMode === "rag_kag") {
    return {
      query: inputText,
    };
  }

  return {
    application_data: inputText,
    stream_agents_state: true,
  };
}
```

- [ ] **Step 2: Update `useAgentStream.ts` imports**

Add:

```ts
import { buildWorkflowPayload } from "@/lib/engineRouting";
```

Change the `ModuleConfig` import to `BankWorkflowConfig`:

```ts
import type {
  AgentStateEvent,
  ActionRequiredEvent,
  CheckpointSavedEvent,
  EvidenceChunk,
  BankWorkflowConfig,
  ResearchPlan,
} from "@/types";
```

- [ ] **Step 3: Update `startStream` signature**

Change:

```ts
async (module: ModuleConfig, applicationData: string) => {
```

to:

```ts
async (module: BankWorkflowConfig, applicationData: string) => {
```

- [ ] **Step 4: Replace inline payload construction**

Replace the current `const payload = ...` block in `useAgentStream.ts` with:

```ts
const payload = buildWorkflowPayload(module, applicationData);
```

- [ ] **Step 5: Run lint**

```powershell
cd frontend
npm run lint
```

Expected: no `ModuleConfig` import remains in `useAgentStream.ts`; no type error for `buildWorkflowPayload()`.

- [ ] **Step 6: Commit**

```powershell
git add frontend/src/lib/engineRouting.ts frontend/src/hooks/useAgentStream.ts
git commit -m "feat: centralize workflow engine routing"
```

Expected: commit succeeds.

---

## Task 5: Replace Flat Navigation With Board + Workflow Selection

**Files:**

- Create: `frontend/src/components/WorkspaceNav.tsx`
- Create: `frontend/src/components/WorkflowSelector.tsx`
- Modify: `frontend/src/app/page.tsx`

- [ ] **Step 1: Create `WorkspaceNav.tsx`**

Create `frontend/src/components/WorkspaceNav.tsx`:

```tsx
"use client";

import type { BankBoardConfig, BankBoardId } from "@/types";

interface WorkspaceNavProps {
  boards: BankBoardConfig[];
  activeBoardId: BankBoardId;
  disabled?: boolean;
  onChange: (boardId: BankBoardId) => void;
}

export default function WorkspaceNav({
  boards,
  activeBoardId,
  disabled = false,
  onChange,
}: WorkspaceNavProps) {
  return (
    <nav className="border-b border-white/[0.06] px-6">
      <div className="max-w-[1400px] mx-auto flex gap-2 py-3 overflow-x-auto">
        {boards.map((board) => (
          <button
            key={board.id}
            onClick={() => onChange(board.id)}
            disabled={disabled}
            className={`px-4 py-3 rounded-xl text-left min-w-[210px] transition-all border ${
              board.id === activeBoardId
                ? "bg-white/[0.08] border-white/[0.14] text-white"
                : "bg-white/[0.02] border-white/[0.06] text-gray-400 hover:text-gray-200 hover:bg-white/[0.04]"
            } disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            <div className="text-sm font-semibold">{board.nameZh}</div>
            <div className="text-[11px] text-gray-500 mt-1 line-clamp-2">{board.name}</div>
          </button>
        ))}
      </div>
    </nav>
  );
}
```

- [ ] **Step 2: Create `WorkflowSelector.tsx`**

Create `frontend/src/components/WorkflowSelector.tsx`:

```tsx
"use client";

import type { BankWorkflowConfig } from "@/types";

interface WorkflowSelectorProps {
  workflows: BankWorkflowConfig[];
  activeWorkflowId: string;
  disabled?: boolean;
  onChange: (workflow: BankWorkflowConfig) => void;
}

function engineLabel(engineMode: BankWorkflowConfig["engineMode"]) {
  if (engineMode === "rag") return "RAG";
  if (engineMode === "rag_kag") return "RAG + KAG";
  if (engineMode === "deepresearch") return "Deep Research";
  return "Human Review";
}

export default function WorkflowSelector({
  workflows,
  activeWorkflowId,
  disabled = false,
  onChange,
}: WorkflowSelectorProps) {
  return (
    <div className="space-y-2">
      {workflows.map((workflow) => (
        <button
          key={workflow.id}
          onClick={() => onChange(workflow)}
          disabled={disabled}
          className={`w-full text-left rounded-xl border px-4 py-3 transition-all ${
            workflow.id === activeWorkflowId
              ? "bg-blue-500/10 border-blue-400/30 text-white"
              : "bg-white/[0.03] border-white/[0.07] text-gray-400 hover:text-gray-200"
          } disabled:opacity-40 disabled:cursor-not-allowed`}
        >
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-semibold">{workflow.nameZh}</span>
            <span className="text-[10px] rounded-full border border-white/[0.12] px-2 py-1 text-gray-500">
              {engineLabel(workflow.engineMode)}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-1">{workflow.description}</p>
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Update `page.tsx` imports**

Replace:

```ts
import { modules } from "@/lib/modules";
```

with:

```ts
import WorkspaceNav from "@/components/WorkspaceNav";
import WorkflowSelector from "@/components/WorkflowSelector";
import { bankBoards, bankWorkflows, defaultWorkflow } from "@/lib/bankWorkspaces";
import type { BankBoardId, BankWorkflowConfig } from "@/types";
```

- [ ] **Step 4: Replace active module state**

Replace:

```ts
const [activeModule, setActiveModule] = useState(0);
const [inputText, setInputText] = useState(modules[0].defaultInput);
const stream = useAgentStream();
const currentModule = modules[activeModule];
```

with:

```ts
const [activeBoardId, setActiveBoardId] = useState<BankBoardId>(defaultWorkflow.boardId);
const [currentModule, setCurrentModule] = useState<BankWorkflowConfig>(defaultWorkflow);
const [inputText, setInputText] = useState(defaultWorkflow.defaultInput);
const stream = useAgentStream();
const workflowsForBoard = bankWorkflows.filter((workflow) => workflow.boardId === activeBoardId);
```

- [ ] **Step 5: Replace module switch handler**

Replace `handleModuleSwitch` with:

```ts
const handleBoardSwitch = (boardId: BankBoardId) => {
  if (stream.isStreaming) return;
  const firstWorkflow = bankWorkflows.find((workflow) => workflow.boardId === boardId);
  if (!firstWorkflow) return;
  setActiveBoardId(boardId);
  setCurrentModule(firstWorkflow);
  setInputText(firstWorkflow.defaultInput);
  stream.reset();
};

const handleWorkflowSwitch = (workflow: BankWorkflowConfig) => {
  if (stream.isStreaming) return;
  setCurrentModule(workflow);
  setInputText(workflow.defaultInput);
  stream.reset();
};
```

- [ ] **Step 6: Replace top navigation JSX**

Replace the existing `<nav>` that maps over `modules` with:

```tsx
<WorkspaceNav
  boards={bankBoards}
  activeBoardId={activeBoardId}
  disabled={stream.isStreaming}
  onChange={handleBoardSwitch}
/>
```

- [ ] **Step 7: Add workflow selector in the left panel**

Inside the left panel, above the input heading, add:

```tsx
<div>
  <h2 className="text-sm font-semibold text-gray-300 mb-2">Workspace Workflows</h2>
  <WorkflowSelector
    workflows={workflowsForBoard}
    activeWorkflowId={currentModule.id}
    disabled={stream.isStreaming}
    onChange={handleWorkflowSwitch}
  />
</div>
```

- [ ] **Step 8: Run lint and build**

```powershell
cd frontend
npm run lint
npm run build
```

Expected: both commands pass. If build fails because of existing unrelated `.next` or dependency state, capture the exact error before editing unrelated files.

- [ ] **Step 9: Commit**

```powershell
git add frontend/src/components/WorkspaceNav.tsx frontend/src/components/WorkflowSelector.tsx frontend/src/app/page.tsx
git commit -m "feat: replace flat modules with bank workspaces"
```

Expected: commit succeeds.

---

## Task 6: Add Workspace Validation Script

**Files:**

- Create: `frontend/scripts/validate-workspaces.mjs`
- Modify: `frontend/package.json`

- [ ] **Step 1: Create scripts directory**

```powershell
New-Item -ItemType Directory -Force -Path frontend/scripts
```

Expected: `frontend/scripts` exists.

- [ ] **Step 2: Create validation script**

Create `frontend/scripts/validate-workspaces.mjs`:

```js
import fs from "node:fs";
import path from "node:path";

const sourcePath = path.join(process.cwd(), "src", "lib", "bankWorkspaces.ts");
const source = fs.readFileSync(sourcePath, "utf8");

const requiredBoardIds = [
  "customer-account",
  "transaction-payment",
  "product-launch",
  "regulatory-research",
  "human-review",
  "knowledge-base",
];

const requiredEngineModes = ["rag", "rag_kag", "deepresearch", "human_review"];

const errors = [];

for (const boardId of requiredBoardIds) {
  if (!source.includes(`id: "${boardId}"`)) {
    errors.push(`Missing board id: ${boardId}`);
  }
}

for (const engineMode of requiredEngineModes) {
  if (!source.includes(`engineMode: "${engineMode}"`)) {
    errors.push(`Missing engine mode: ${engineMode}`);
  }
}

for (const field of ["nameZh", "description", "primaryUsers", "defaultInput", "endpoint"]) {
  if (!source.includes(field)) {
    errors.push(`Missing required workflow field in source: ${field}`);
  }
}

if (errors.length > 0) {
  console.error(errors.join("\n"));
  process.exit(1);
}

console.log("Workspace configuration validation passed.");
```

- [ ] **Step 3: Add package script**

In `frontend/package.json`, add this script under `scripts`:

```json
"validate:workspaces": "node scripts/validate-workspaces.mjs"
```

- [ ] **Step 4: Run validation**

```powershell
cd frontend
npm run validate:workspaces
```

Expected:

```text
Workspace configuration validation passed.
```

- [ ] **Step 5: Commit**

```powershell
git add frontend/scripts/validate-workspaces.mjs frontend/package.json
git commit -m "test: validate bank workspace configuration"
```

Expected: commit succeeds.

---

## Task 7: Update README Positioning

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Replace top product description**

Replace:

```markdown
HK-FinReg AI is an extensible Hong Kong financial regulatory intelligence platform.
```

with:

```markdown
HK-FinReg AI is a Hong Kong bank internal Regulatory Intelligence & Compliance Operations Platform.
It helps bank compliance, AML, KYC, product, legal, and regulatory affairs teams perform evidence-backed compliance review, regulatory research, policy impact analysis, and human review workflows.
```

- [ ] **Step 2: Replace "Platform Modules (10)" section**

Replace the current module list with:

```markdown
### Bank ToB Workspaces

1. Customer & Account Compliance
2. Transaction & Payment Compliance
3. Product & Business Launch Review
4. Regulatory Research & Policy Change
5. Human Review & Audit
6. Regulatory Knowledge Base

### Engine Routing

- Routine compliance review uses Hybrid RAG.
- Obligation, risk, and control mapping uses RAG + KAG.
- Product launch, AI governance, regulatory memo, and policy impact workflows use RAG + KAG + DeepResearch.
- Low-confidence and missing-evidence cases are routed to human review.
```

- [ ] **Step 3: Verify README still renders**

Run:

```powershell
Get-Content README.md -TotalCount 80
```

Expected: the top section describes the bank ToB workspaces and no longer claims 10 top-level modules.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: reposition platform for bank tob workflows"
```

Expected: commit succeeds.

---

## Task 8: Verification Gate

**Files:**

- No new files.

- [ ] **Step 1: Run workspace validation**

```powershell
cd frontend
npm run validate:workspaces
```

Expected: validation passes.

- [ ] **Step 2: Run frontend lint**

```powershell
cd frontend
npm run lint
```

Expected: lint passes.

- [ ] **Step 3: Run frontend build**

```powershell
cd frontend
npm run build
```

Expected: build passes.

- [ ] **Step 4: Run backend regression tests relevant to routing and KAG/DeepResearch**

```powershell
cd ..
python -m pytest backend/tests/test_deepresearch.py backend/tests/test_obligation_mapper.py backend/tests/test_kag_graph_store.py backend/tests/test_retrieval_router.py -q
```

Expected: all selected backend tests pass.

- [ ] **Step 5: Inspect git diff**

```powershell
git status --short
git diff -- README.md docs/product/bank-tob-information-architecture.md frontend/src/types/index.ts frontend/src/lib/bankWorkspaces.ts frontend/src/lib/modules.ts frontend/src/lib/engineRouting.ts frontend/src/hooks/useAgentStream.ts frontend/src/components/WorkspaceNav.tsx frontend/src/components/WorkflowSelector.tsx frontend/src/app/page.tsx frontend/scripts/validate-workspaces.mjs frontend/package.json
```

Expected: only intended files are changed.

- [ ] **Step 6: Final commit if verification caused extra changes**

If package lock or generated files changed due to script setup, inspect first. If the changes are expected, commit them:

```powershell
git add README.md docs/product/bank-tob-information-architecture.md frontend/src/types/index.ts frontend/src/lib/bankWorkspaces.ts frontend/src/lib/modules.ts frontend/src/lib/engineRouting.ts frontend/src/hooks/useAgentStream.ts frontend/src/components/WorkspaceNav.tsx frontend/src/components/WorkflowSelector.tsx frontend/src/app/page.tsx frontend/scripts/validate-workspaces.mjs frontend/package.json
git commit -m "chore: verify bank workspace rearchitecture"
```

Expected: commit succeeds only if there are remaining intended changes.

---

## Non-Goals for This Plan

Do not do these in the first execution pass:

- Do not delete existing backend routers.
- Do not rewrite the retrieval stack.
- Do not replace the existing SSE streaming protocol.
- Do not remove KAG or DeepResearch experimental endpoints.
- Do not introduce a large state management library.
- Do not create separate pages for every board unless the single-page workspace becomes unmanageable after this refactor.

---

## Acceptance Criteria

- The frontend no longer presents 10 flat modules as first-level product navigation.
- The user sees bank-oriented boards and then chooses a workflow inside a board.
- Each workflow declares its engine mode: `rag`, `rag_kag`, `deepresearch`, or `human_review`.
- RAG is used for routine compliance review.
- KAG is used for obligation/risk/control mapping and knowledge-base workflows.
- Deep Research is used for product launch, AI governance, regulatory memo, policy change, and complex investigation workflows.
- Existing backend endpoints remain compatible.
- README and product docs describe the platform as a Hong Kong bank internal ToB program.
- Frontend lint and build pass.
- Relevant backend tests for DeepResearch, KAG, and retrieval routing pass.

---

## Self-Review

- Spec coverage: The plan covers product repositioning, board/module consolidation, frontend configuration, routing, documentation, and verification.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: `EngineMode`, `BankBoardId`, `BankBoardConfig`, and `BankWorkflowConfig` are introduced before use. `bankWorkflows` re-exports through `modules` for compatibility. `buildWorkflowPayload()` accepts `BankWorkflowConfig`, matching the updated hook.
