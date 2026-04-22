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

export interface ComplianceRequest {
  application_data: string;
  business_context?: string;
  stream_agents_state?: boolean;
}

export interface ModuleConfig {
  id: string;
  name: string;
  nameZh: string;
  endpoint: string;
  icon: string;
  defaultInput: string;
}
