// 核心 SSE 流式解析 Hook — 针对 200s 长链路优化
// 解析 agent_state / token / done / action_required / checkpoint_saved 事件
"use client";

import { useState, useCallback, useRef } from "react";
import type {
  AgentStateEvent,
  ActionRequiredEvent,
  CheckpointSavedEvent,
  EvidenceChunk,
  ResearchPlan,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

function buildHeaders(): HeadersInit {
  const headers: HeadersInit = { "Content-Type": "application/json" };
  if (API_KEY) {
    headers.Authorization = `Bearer ${API_KEY}`;
  }
  return headers;
}

interface StreamState {
  isStreaming: boolean;
  agentStates: AgentStateEvent[];
  currentAgent: string | null;
  reportText: string;
  error: string | null;
  elapsedTime: number;
  phase: "idle" | "agents" | "streaming" | "done" | "action_required";
  // P2: 置信度（三维）
  confidenceScore: number | null;
  confidenceWarning: string | null;
  reasoningConfidence: number | null;
  reviewerConfidence: number | null;
  crossValidationPassed: boolean | null;
  // Phase 1: HITL 状态
  workflowRunId: string | null;
  humanReviewRequired: boolean;
  currentGate: string | null;
  gateMessage: string | null;
  // RAG/KAG/DeepResearch 升级：新数据字段
  evidenceChunks: EvidenceChunk[];
  graphPaths: Array<{ path: string[]; matched_node: string; matched_topics: string[] }>;
  researchPlan: ResearchPlan | null;
  evidenceBySubquestion: Record<string, EvidenceChunk[]>;
  evidenceGaps: Array<{ sub_question_id: string; reason: string }>;
}

const INITIAL_STATE: StreamState = {
  isStreaming: false,
  agentStates: [],
  currentAgent: null,
  reportText: "",
  error: null,
  elapsedTime: 0,
  phase: "idle",
  confidenceScore: null,
  confidenceWarning: null,
  reasoningConfidence: null,
  reviewerConfidence: null,
  crossValidationPassed: null,
  // Phase 1
  workflowRunId: null,
  humanReviewRequired: false,
  currentGate: null,
  gateMessage: null,
  // RAG/KAG/DeepResearch
  evidenceChunks: [],
  graphPaths: [],
  researchPlan: null,
  evidenceBySubquestion: {},
  evidenceGaps: [],
};

export function useAgentStream() {
  const [state, setState] = useState<StreamState>(INITIAL_STATE);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const startTimeRef = useRef<number>(0);

  const startStream = useCallback(
    async (endpoint: string, applicationData: string) => {
      // Reset
      setState({ ...INITIAL_STATE, isStreaming: true, phase: "agents" });
      startTimeRef.current = performance.now();

      // Elapsed timer — 每秒更新
      timerRef.current = setInterval(() => {
        setState((prev) => ({
          ...prev,
          elapsedTime: Math.round(
            (performance.now() - startTimeRef.current) / 1000
          ),
        }));
      }, 1000);

      abortRef.current = new AbortController();

      try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
          method: "POST",
          headers: buildHeaders(),
          body: JSON.stringify({
            application_data: applicationData,
            stream_agents_state: true,
          }),
          signal: abortRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let eventType = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));

                if (eventType === "agent_state") {
                  const agentEvent = data as AgentStateEvent;
                  // 反思循环：Sub-Query Planner 标记
                  const message = agentEvent.agent === "Sub-Query Planner"
                    ? "反思循环：正在规划二次检索策略..."
                    : agentEvent.message;
                  setState((prev) => ({
                    ...prev,
                    phase: "agents",
                    currentAgent: agentEvent.agent,
                    agentStates: [...prev.agentStates, { ...agentEvent, message }],
                  }));
                } else if (eventType === "token") {
                  setState((prev) => ({
                    ...prev,
                    phase: "streaming",
                    currentAgent: null,
                    reportText: prev.reportText + (data.text || ""),
                  }));
                } else if (eventType === "done") {
                  setState((prev) => ({
                    ...prev,
                    phase: "done",
                    workflowRunId: data.workflow_run_id || prev.workflowRunId,
                    // 如果 done 事件内联包含 RAG/KAG/DeepResearch 数据，一并提取
                    evidenceChunks: data.evidence_chunks
                      ? (Array.isArray(data.evidence_chunks) ? data.evidence_chunks : [])
                      : prev.evidenceChunks,
                    graphPaths: data.graph_paths
                      ? (Array.isArray(data.graph_paths) ? data.graph_paths : [])
                      : prev.graphPaths,
                    researchPlan: data.research_plan
                      ? (data.research_plan || null)
                      : prev.researchPlan,
                    evidenceBySubquestion: data.evidence_by_subquestion
                      ? (data.evidence_by_subquestion || {})
                      : prev.evidenceBySubquestion,
                    evidenceGaps: data.evidence_gaps
                      ? (Array.isArray(data.evidence_gaps) ? data.evidence_gaps : [])
                      : prev.evidenceGaps,
                  }));
                } else if (eventType === "error") {
                  setState((prev) => ({
                    ...prev,
                    error: data.message || data.detail || "Stream failed",
                    phase: "idle",
                    isStreaming: false,
                  }));
                } else if (eventType === "confidence") {
                  // P2: 三维置信度事件
                  const dim = data.dimension;
                  if (dim === "full") {
                    setState((prev) => ({
                      ...prev,
                      confidenceScore: data.retrieval ?? prev.confidenceScore,
                      reasoningConfidence: data.reasoning ?? null,
                      reviewerConfidence: data.reviewer ?? null,
                      crossValidationPassed: data.cross_validation_passed ?? null,
                    }));
                  } else if (dim === "reasoning") {
                    setState((prev) => ({
                      ...prev,
                      reasoningConfidence: data.score ?? null,
                    }));
                  } else {
                    // retrieval (默认)
                    setState((prev) => ({
                      ...prev,
                      confidenceScore: data.score ?? null,
                      confidenceWarning: data.warning ?? null,
                    }));
                  }
                } else if (eventType === "action_required") {
                  // Phase 1: HITL 人工审查事件
                  const actionEvent = data as ActionRequiredEvent;
                  setState((prev) => ({
                    ...prev,
                    phase: "action_required",
                    isStreaming: false,
                    humanReviewRequired: true,
                    workflowRunId: actionEvent.workflow_run_id || prev.workflowRunId,
                    currentGate: actionEvent.gate_type || null,
                    gateMessage: actionEvent.message || null,
                  }));
                } else if (eventType === "checkpoint_saved") {
                  // Phase 1: Checkpoint 保存事件
                  const cpEvent = data as CheckpointSavedEvent;
                  setState((prev) => ({
                    ...prev,
                    workflowRunId: cpEvent.workflow_run_id || prev.workflowRunId,
                  }));
                } else if (eventType === "resume_ready") {
                  // Phase 1: 工作流已恢复
                  setState((prev) => ({
                    ...prev,
                    phase: "agents",
                    humanReviewRequired: false,
                    currentGate: null,
                    gateMessage: null,
                  }));
                } else if (eventType === "evidence_chunks") {
                  // RAG/KAG: 证据块事件
                  const chunks = data as EvidenceChunk[];
                  setState((prev) => ({
                    ...prev,
                    evidenceChunks: Array.isArray(chunks) ? chunks : [],
                  }));
                } else if (eventType === "graph_paths") {
                  // KAG: 知识图谱路径事件
                  const paths = data as Array<{ path: string[]; matched_node: string; matched_topics: string[] }>;
                  setState((prev) => ({
                    ...prev,
                    graphPaths: Array.isArray(paths) ? paths : [],
                  }));
                } else if (eventType === "research_plan") {
                  // DeepResearch: 研究计划事件
                  const plan = data as ResearchPlan;
                  setState((prev) => ({
                    ...prev,
                    researchPlan: plan || null,
                  }));
                } else if (eventType === "evidence_by_subquestion") {
                  // DeepResearch: 子问题证据映射
                  const evMap = data as Record<string, EvidenceChunk[]>;
                  setState((prev) => ({
                    ...prev,
                    evidenceBySubquestion: evMap || {},
                  }));
                } else if (eventType === "evidence_gaps") {
                  // DeepResearch: 证据差距事件
                  const gaps = data as Array<{ sub_question_id: string; reason: string }>;
                  setState((prev) => ({
                    ...prev,
                    evidenceGaps: Array.isArray(gaps) ? gaps : [],
                  }));
                }
              } catch {
                // skip non-JSON lines
              }
              eventType = "";
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== "AbortError") {
          setState((prev) => ({
            ...prev,
            error: err instanceof Error ? err.message : "Unknown error",
          }));
        }
      } finally {
        if (timerRef.current) clearInterval(timerRef.current);
        setState((prev) => {
          // action_required 状态不改变 phase
          if (prev.phase === "action_required") {
            return { ...prev, isStreaming: false };
          }
          return {
            ...prev,
            isStreaming: false,
            phase: prev.error ? "idle" : "done",
            elapsedTime: Math.round(
              (performance.now() - startTimeRef.current) / 1000
            ),
          };
        });
      }
    },
    []
  );

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  const setResumedResult = useCallback((finalReport: string, approved: boolean) => {
    void approved;
    setState((prev) => ({
      ...prev,
      phase: "done",
      isStreaming: false,
      reportText: finalReport,
      humanReviewRequired: false,
      currentGate: null,
      gateMessage: null,
    }));
  }, []);

  return { ...state, startStream, cancelStream, reset, setResumedResult };
}
