import type { BankBoardConfig, BankWorkflowConfig } from "@/types";

export const bankBoards: BankBoardConfig[] = [
  {
    id: "customer-account",
    name: "Customer & Account Compliance",
    nameZh: "客戶及賬戶合規",
    description: "KYC, CDD, customer risk rating, account opening, and enhanced due diligence workflows.",
    primaryUsers: ["KYC Operations", "AML Compliance", "Relationship Managers", "Account Reviewers"],
  },
  {
    id: "transaction-payment",
    name: "Transaction & Payment Compliance",
    nameZh: "交易及支付合規",
    description: "Cross-border payment, suspicious transaction, sanctions, and payment AML workflows.",
    primaryUsers: ["Transaction Monitoring", "Payment Operations", "AML Investigators"],
  },
  {
    id: "product-launch",
    name: "Product & Business Launch Review",
    nameZh: "產品及業務上線審查",
    description: "New product, AI governance, data privacy, outsourcing, and launch checklist workflows.",
    primaryUsers: ["Product Teams", "Compliance Advisory", "Legal", "Model Risk", "Data Protection"],
  },
  {
    id: "regulatory-research",
    name: "Regulatory Research & Policy Change",
    nameZh: "監管研究及政策變更",
    description: "Regulatory memos, policy impact, cross-regulator comparison, and management brief workflows.",
    primaryUsers: ["Compliance Policy", "Legal", "Regulatory Affairs", "Senior Management"],
  },
  {
    id: "human-review",
    name: "Human Review & Audit",
    nameZh: "人工覆核及審計",
    description: "Low-confidence cases, missing evidence cases, approval queue, and audit trail workflows.",
    primaryUsers: ["Compliance Managers", "Second-Line Reviewers", "Internal Audit", "Model Governance"],
  },
  {
    id: "knowledge-base",
    name: "Regulatory Knowledge Base",
    nameZh: "監管知識庫",
    description: "Regulatory documents, obligations, knowledge graph, citations, versions, and regression gates.",
    primaryUsers: ["Knowledge Managers", "RegTech Administrators", "Model Governance", "Engineering Support"],
  },
];

export const bankWorkflows: BankWorkflowConfig[] = [
  {
    id: "account-kyc-review",
    boardId: "customer-account",
    name: "Account Opening / KYC Review",
    nameZh: "開戶 / KYC 審查",
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
    nameZh: "客戶風險評級",
    description: "Assess customer risk factors and map them to due diligence obligations.",
    endpoint: "/api/v1/kag/obligation-map",
    icon: "RISK",
    status: "experimental",
    requestKind: "kag",
    engineMode: "rag_kag",
    scenarioType: "customer_review",
    primaryUsers: ["AML Compliance", "Account Reviewers"],
    defaultInput:
      "Assess the customer risk factors for a non-face-to-face onboarding case involving eKYC, cross-border source of funds, and a politically exposed close associate.",
  },
  {
    id: "cross-border-payment",
    boardId: "transaction-payment",
    name: "Cross-Border Payment Assessment",
    nameZh: "跨境支付合規評估",
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
    nameZh: "複雜交易調查",
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
    defaultInput:
      "Investigate a pattern of repeated cross-border transfers involving multiple senders, related beneficiaries, high-risk jurisdictions, and unclear source of funds.",
  },
  {
    id: "product-launch-review",
    boardId: "product-launch",
    name: "New Product Launch Review",
    nameZh: "新產品上線合規評估",
    description:
      "Review new banking or fintech products before launch across licensing, conduct, AML, privacy, and control obligations.",
    endpoint: "/api/v1/research/analyze",
    icon: "LAUNCH",
    status: "experimental",
    requestKind: "research",
    taskType: "product_launch_review",
    outputFormat: "report",
    engineMode: "deepresearch",
    scenarioType: "product_launch",
    primaryUsers: ["Product Teams", "Compliance Advisory", "Legal"],
    defaultInput:
      "We plan to launch an AI-powered SME credit scoring platform in Hong Kong using bank statements and external data sources. Assess compliance risks.",
  },
  {
    id: "ai-governance-review",
    boardId: "product-launch",
    name: "AI / GenAI Governance Review",
    nameZh: "AI / GenAI 治理評估",
    description:
      "Assess governance, model risk, consumer protection, and data obligations for AI-enabled financial use cases.",
    endpoint: "/api/v1/research/analyze",
    icon: "AIGC",
    status: "experimental",
    requestKind: "research",
    taskType: "ai_governance_review",
    outputFormat: "report",
    engineMode: "deepresearch",
    scenarioType: "product_launch",
    primaryUsers: ["Model Risk", "Compliance Advisory", "Legal"],
    defaultInput:
      "Assess the compliance obligations for a generative-AI customer service chatbot used in Hong Kong retail banking.",
  },
  {
    id: "regulatory-memo",
    boardId: "regulatory-research",
    name: "Regulatory Memo",
    nameZh: "監管備忘錄",
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
    defaultInput:
      "Compare HKMA, SFC and PCPD expectations on the use of Generative AI in financial institutions.",
  },
  {
    id: "policy-change-impact",
    boardId: "regulatory-research",
    name: "Policy Change Impact",
    nameZh: "政策變更影響分析",
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
    defaultInput:
      "New Source: PCPD AI Model Personal Data Protection Framework. Which existing products, controls, and obligations are affected?",
  },
  {
    id: "review-queue",
    boardId: "human-review",
    name: "Pending Human Review",
    nameZh: "待人工覆核隊列",
    description: "Review low-confidence, missing-evidence, and manual-approval compliance cases.",
    endpoint: "/api/v1/review-queue/pending",
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
    nameZh: "監管義務映射",
    description: "Map scenarios to applicable regulators, obligations, risks, controls, and evidence.",
    endpoint: "/api/v1/kag/obligation-map",
    icon: "KAG",
    status: "experimental",
    requestKind: "kag",
    engineMode: "rag_kag",
    scenarioType: "knowledge_management",
    primaryUsers: ["Knowledge Managers", "RegTech Administrators"],
    defaultInput:
      "A Hong Kong SVF operator plans to use facial recognition eKYC and AI-based transaction monitoring for cross-border payment services.",
  },
  {
    id: "graph-explorer",
    boardId: "knowledge-base",
    name: "Knowledge Graph Explorer",
    nameZh: "監管知識圖譜查詢",
    description:
      "Explore regulator, document, obligation, risk, control, product, and customer-segment relationships.",
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

export const defaultWorkflow: BankWorkflowConfig = bankWorkflows[0]!;
