"use client";

import { useCallback, useRef, useState } from "react";

import { buildWorkflowPayload } from "@/lib/engineRouting";
import {
  formatHumanReviewQueueReport,
  formatJsonReport,
  formatTextReport,
} from "@/lib/reportFormatting";
import type {
  ActionRequiredEvent,
  AgentStateEvent,
  BankWorkflowConfig,
  CheckpointSavedEvent,
  EvidenceChunk,
  ResearchPlan,
} from "@/types";

const API_PROXY_BASE = "/api/backend";

function buildHeaders(): HeadersInit {
  return { "Content-Type": "application/json" };
}

function buildProxyUrl(endpoint: string): string {
  return `${API_PROXY_BASE}${endpoint}`;
}

interface StreamState {
  isStreaming: boolean;
  agentStates: AgentStateEvent[];
  currentAgent: string | null;
  reportText: string;
  error: string | null;
  elapsedTime: number;
  phase: "idle" | "agents" | "streaming" | "done" | "action_required";
  confidenceScore: number | null;
  confidenceWarning: string | null;
  reasoningConfidence: number | null;
  reviewerConfidence: number | null;
  crossValidationPassed: boolean | null;
  workflowRunId: string | null;
  humanReviewRequired: boolean;
  currentGate: string | null;
  gateMessage: string | null;
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
  workflowRunId: null,
  humanReviewRequired: false,
  currentGate: null,
  gateMessage: null,
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
  const lastModuleRef = useRef<BankWorkflowConfig | null>(null);

  const startStream = useCallback(async (module: BankWorkflowConfig, applicationData: string) => {
    setState({ ...INITIAL_STATE, isStreaming: true, phase: "agents" });
    startTimeRef.current = performance.now();
    lastModuleRef.current = module;

    timerRef.current = setInterval(() => {
      setState((prev) => ({
        ...prev,
        elapsedTime: Math.round((performance.now() - startTimeRef.current) / 1000),
      }));
    }, 1000);

    abortRef.current = new AbortController();

    try {
      if (module.engineMode === "human_review") {
        const response = await fetch(buildProxyUrl(module.endpoint), {
          method: "GET",
          headers: buildHeaders(),
          signal: abortRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const queueItems = (await response.json()) as Array<{
          workflow_run_id?: string;
          gate_type?: string;
          human_review_status?: string;
        }>;

        setState((prev) => ({
          ...prev,
          isStreaming: false,
          phase: "done",
          reportText: formatHumanReviewQueueReport(module, queueItems),
        }));
        return;
      }

      const payload = buildWorkflowPayload(module, applicationData);
      const response = await fetch(buildProxyUrl(module.endpoint), {
        method: "POST",
        headers: buildHeaders(),
        body: JSON.stringify(payload),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get("content-type") || "";
      if (!contentType.includes("text/event-stream")) {
        const data = await response.json();
        const reportText =
          typeof data?.final_report === "string"
            ? formatTextReport(module, data.final_report)
            : formatJsonReport(module, data);

        setState((prev) => ({
          ...prev,
          isStreaming: false,
          phase: "done",
          reportText,
          researchPlan: data?.research_plan ?? prev.researchPlan,
          evidenceBySubquestion: data?.evidence_by_subquestion ?? prev.evidenceBySubquestion,
          evidenceGaps: Array.isArray(data?.evidence_gaps) ? data.evidence_gaps : prev.evidenceGaps,
          graphPaths: Array.isArray(data?.paths) ? data.paths : prev.graphPaths,
        }));
        return;
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
                setState((prev) => ({
                  ...prev,
                  phase: "agents",
                  currentAgent: agentEvent.agent,
                  agentStates: [...prev.agentStates, agentEvent],
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
                  reportText: formatTextReport(module, prev.reportText),
                  workflowRunId: data.workflow_run_id || prev.workflowRunId,
                  evidenceChunks: data.evidence_chunks
                    ? Array.isArray(data.evidence_chunks)
                      ? data.evidence_chunks
                      : []
                    : prev.evidenceChunks,
                  graphPaths: data.graph_paths
                    ? Array.isArray(data.graph_paths)
                      ? data.graph_paths
                      : []
                    : prev.graphPaths,
                  researchPlan: data.research_plan ? data.research_plan || null : prev.researchPlan,
                  evidenceBySubquestion: data.evidence_by_subquestion
                    ? data.evidence_by_subquestion || {}
                    : prev.evidenceBySubquestion,
                  evidenceGaps: data.evidence_gaps
                    ? Array.isArray(data.evidence_gaps)
                      ? data.evidence_gaps
                      : []
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
                  setState((prev) => ({
                    ...prev,
                    confidenceScore: data.score ?? null,
                    confidenceWarning: data.warning ?? null,
                  }));
                }
              } else if (eventType === "action_required") {
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
                const cpEvent = data as CheckpointSavedEvent;
                setState((prev) => ({
                  ...prev,
                  workflowRunId: cpEvent.workflow_run_id || prev.workflowRunId,
                }));
              } else if (eventType === "resume_ready") {
                setState((prev) => ({
                  ...prev,
                  phase: "agents",
                  humanReviewRequired: false,
                  currentGate: null,
                  gateMessage: null,
                }));
              } else if (eventType === "evidence_chunks") {
                const chunks = data as EvidenceChunk[];
                setState((prev) => ({
                  ...prev,
                  evidenceChunks: Array.isArray(chunks) ? chunks : [],
                }));
              } else if (eventType === "graph_paths") {
                const paths = data as Array<{
                  path: string[];
                  matched_node: string;
                  matched_topics: string[];
                }>;
                setState((prev) => ({
                  ...prev,
                  graphPaths: Array.isArray(paths) ? paths : [],
                }));
              } else if (eventType === "research_plan") {
                const plan = data as ResearchPlan;
                setState((prev) => ({
                  ...prev,
                  researchPlan: plan || null,
                }));
              } else if (eventType === "evidence_by_subquestion") {
                const evMap = data as Record<string, EvidenceChunk[]>;
                setState((prev) => ({
                  ...prev,
                  evidenceBySubquestion: evMap || {},
                }));
              } else if (eventType === "evidence_gaps") {
                const gaps = data as Array<{ sub_question_id: string; reason: string }>;
                setState((prev) => ({
                  ...prev,
                  evidenceGaps: Array.isArray(gaps) ? gaps : [],
                }));
              }
            } catch {
              // ignore malformed JSON chunks
            }
            eventType = "";
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        setState((prev) => ({
          ...prev,
          error: err.message || "Unknown error",
        }));
      }
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setState((prev) => {
        if (prev.phase === "action_required") {
          return { ...prev, isStreaming: false };
        }
        return {
          ...prev,
          isStreaming: false,
          phase: prev.error ? "idle" : "done",
          elapsedTime: Math.round((performance.now() - startTimeRef.current) / 1000),
        };
      });
    }
  }, []);

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  const setResumedResult = useCallback((finalReport: string, approved: boolean) => {
    void approved;
    const currentModule = lastModuleRef.current;
    setState((prev) => ({
      ...prev,
      phase: "done",
      isStreaming: false,
      reportText: currentModule ? formatTextReport(currentModule, finalReport) : finalReport,
      humanReviewRequired: false,
      currentGate: null,
      gateMessage: null,
    }));
  }, []);

  return { ...state, startStream, cancelStream, reset, setResumedResult };
}
