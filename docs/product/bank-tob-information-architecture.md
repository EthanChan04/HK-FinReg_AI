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
