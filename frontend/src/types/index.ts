// TypeScript 类型定义：后端 SSE 事件结构
export interface AgentStateEvent {
  agent: string;
  status: "running" | "done";
  message: string;
}

export interface TokenEvent {
  text: string;
}

export interface DoneEvent {
  status: string;
  processing_time?: number;
  workflow_run_id?: string;
}

// P2 置信度事件（三维）
export interface ConfidenceEvent {
  score?: number;
  warning?: string | null;
  dimension?: "retrieval" | "reasoning" | "full";
  // 三维置信度（dimension=full 时存在）
  retrieval?: number | null;
  reasoning?: number | null;
  reviewer?: number | null;
  cross_validation_passed?: boolean | null;
}

// Phase 1: HITL 事件
export interface ActionRequiredEvent {
  workflow_run_id: string;
  gate_type: "low_confidence_gate" | "missing_evidence_gate" | "manual_approval_gate";
  message: string;
  evidence_snapshot?: Record<string, unknown>;
  confidence_data?: {
    retrieval?: number | null;
    reasoning?: number | null;
    reviewer?: number | null;
    cross_validation_passed?: boolean | null;
  };
}

export interface CheckpointSavedEvent {
  workflow_run_id: string;
  status: string;
}

export interface ResumeReadyEvent {
  workflow_run_id: string;
  message: string;
}

export interface ComplianceRequest {
  application_data: string;
  business_context?: string;
  stream_agents_state?: boolean;
}

export interface EvidenceChunk {
  evidence_id: string;
  chunk_id?: string | null;
  doc_id?: string | null;
  title?: string | null;
  regulator?: string | null;
  jurisdiction?: string;
  doc_type?: string | null;
  page?: number | null;
  section_title?: string | null;
  hierarchy_path?: string | null;
  source_url?: string | null;
  text: string;
  retrieval_method?: string;
  score?: number | null;
  metadata?: Record<string, unknown>;
}

export interface CitationAudit {
  supported_citations: Array<Record<string, unknown>>;
  unsupported_citations: Array<Record<string, unknown>>;
  unsupported_claim_rate: number;
}

export interface ResearchPlan {
  research_goal?: string;
  sub_questions?: Array<Record<string, unknown>>;
  expected_output_sections?: string[];
}

export interface DeepResearchResponse {
  research_plan: ResearchPlan | Record<string, unknown>;
  evidence_by_subquestion: Record<string, EvidenceChunk[]>;
  evidence_gaps: Array<Record<string, unknown>>;
  final_report: string;
  citation_audit: CitationAudit | Record<string, unknown>;
}

export interface ModuleConfig {
  id: string;
  name: string;
  nameZh: string;
  endpoint: string;
  icon: string;
  defaultInput: string;
}

// Phase 1: 审查队列条目
export interface ReviewQueueItem {
  workflow_run_id: string;
  gate_type: string;
  checkpoint_created_at: number;
  evidence_snapshot: Record<string, unknown>;
  latest_draft_report: string;
  confidence_data: Record<string, unknown>;
  original_input: string;
  human_review_status: "pending" | "approved" | "rejected";
  human_review_notes: string;
  reviewed_at: number | null;
  reviewed_by: string | null;
}
