// 核心 SSE 流式解析 Hook — 针对 200s 长链路优化
// 解析 agent_state / token / done 三类 SSE 事件
"use client";

import { useState, useCallback, useRef } from "react";
import type { AgentStateEvent } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

interface StreamState {
  isStreaming: boolean;
  agentStates: AgentStateEvent[];
  currentAgent: string | null;
  reportText: string;
  error: string | null;
  elapsedTime: number;
  phase: "idle" | "agents" | "streaming" | "done";
}

const INITIAL_STATE: StreamState = {
  isStreaming: false,
  agentStates: [],
  currentAgent: null,
  reportText: "",
  error: null,
  elapsedTime: 0,
  phase: "idle",
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
          headers: { "Content-Type": "application/json" },
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
        setState((prev) => ({
          ...prev,
          isStreaming: false,
          phase: prev.error ? "idle" : "done",
          elapsedTime: Math.round(
            (performance.now() - startTimeRef.current) / 1000
          ),
        }));
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

  return { ...state, startStream, cancelStream, reset };
}
