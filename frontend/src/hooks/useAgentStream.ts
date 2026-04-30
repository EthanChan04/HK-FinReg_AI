// 核心 SSE 流式解析 Hook — 针对 200s 长链路优化
// 解析 agent_state / token / done / action_required / checkpoint_saved 事件
"use client";

import { useState, useCallback, useRef } from "react";
import type { AgentStateEvent, ActionRequiredEvent, CheckpointSavedEvent } from "@/types";

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
