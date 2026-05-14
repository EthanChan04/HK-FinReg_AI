"use client";

import { useCallback, useRef, useState } from "react";

import type {
  CopilotCaseContext,
  CopilotIntent,
  CopilotMessage,
  CopilotToolEvent,
  EvidenceChunk,
} from "@/types";

const API_PROXY_BASE = "/api/backend";

function buildHeaders(): HeadersInit {
  return { "Content-Type": "application/json" };
}

interface CopilotState {
  messages: CopilotMessage[];
  toolEvents: CopilotToolEvent[];
  evidenceChunks: EvidenceChunk[];
  graphPaths: Array<{ path: string[]; matched_node?: string; matched_topics?: string[] }>;
  intent: CopilotIntent | null;
  conversationId: string | null;
  unsupportedClaimRate: number | null;
  isLoading: boolean;
  error: string | null;
}

const INITIAL_STATE: CopilotState = {
  messages: [],
  toolEvents: [],
  evidenceChunks: [],
  graphPaths: [],
  intent: null,
  conversationId: null,
  unsupportedClaimRate: null,
  isLoading: false,
  error: null,
};

function newMessageId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function useCopilotChat() {
  const [state, setState] = useState<CopilotState>(INITIAL_STATE);
  const abortRef = useRef<AbortController | null>(null);
  const activeAssistantMessageId = useRef<string | null>(null);

  const appendAssistantToken = useCallback((token: string) => {
    setState((prev) => {
      let nextMessages = [...prev.messages];
      let messageId = activeAssistantMessageId.current;

      if (!messageId) {
        messageId = newMessageId("assistant");
        activeAssistantMessageId.current = messageId;
        nextMessages.push({
          id: messageId,
          role: "assistant",
          content: token,
          createdAt: Date.now(),
        });
      } else {
        nextMessages = nextMessages.map((message) =>
          message.id === messageId ? { ...message, content: message.content + token } : message
        );
      }

      return { ...prev, messages: nextMessages };
    });
  }, []);

  const sendMessage = useCallback(
    async (content: string, caseContext: CopilotCaseContext) => {
      const trimmed = content.trim();
      if (!trimmed) return;

      const userMessage: CopilotMessage = {
        id: newMessageId("user"),
        role: "user",
        content: trimmed,
        createdAt: Date.now(),
      };

      setState((prev) => ({
        ...prev,
        isLoading: true,
        error: null,
        toolEvents: [],
        intent: null,
        unsupportedClaimRate: null,
        messages: [...prev.messages, userMessage],
      }));
      activeAssistantMessageId.current = null;

      abortRef.current = new AbortController();

      try {
        const response = await fetch(`${API_PROXY_BASE}/api/v1/copilot/chat/stream`, {
          method: "POST",
          headers: buildHeaders(),
          signal: abortRef.current.signal,
          body: JSON.stringify({
            message: trimmed,
            conversation_id: state.conversationId,
            history: [...state.messages, userMessage].map((item) => ({
              role: item.role,
              content: item.content,
            })),
            case_context: caseContext,
            preferred_language: "zh-HK+en",
          }),
        });

        if (!response.ok) {
          const maybeError = await response.text();
          throw new Error(`HTTP ${response.status}: ${maybeError || response.statusText}`);
        }

        if (!response.body) {
          throw new Error("No response stream available from Copilot API");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = "";
        let currentEvent = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line) continue;
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
              continue;
            }
            if (!line.startsWith("data: ")) continue;

            try {
              const payload = JSON.parse(line.slice(6));

              if (currentEvent === "intent") {
                setState((prev) => ({ ...prev, intent: payload.intent || null }));
              } else if (currentEvent === "tool_call") {
                setState((prev) => ({
                  ...prev,
                  toolEvents: [
                    ...prev.toolEvents,
                    {
                      tool: payload.tool,
                      status: payload.status,
                      message: payload.message,
                    },
                  ],
                }));
              } else if (currentEvent === "evidence") {
                setState((prev) => ({
                  ...prev,
                  evidenceChunks: Array.isArray(payload.evidence_chunks) ? payload.evidence_chunks : prev.evidenceChunks,
                }));
              } else if (currentEvent === "graph") {
                setState((prev) => ({
                  ...prev,
                  graphPaths: Array.isArray(payload.graph_paths) ? payload.graph_paths : prev.graphPaths,
                }));
              } else if (currentEvent === "token") {
                appendAssistantToken(payload.text || "");
              } else if (currentEvent === "citation_audit") {
                setState((prev) => ({
                  ...prev,
                  unsupportedClaimRate:
                    typeof payload.unsupported_claim_rate === "number"
                      ? payload.unsupported_claim_rate
                      : prev.unsupportedClaimRate,
                }));
              } else if (currentEvent === "done") {
                setState((prev) => ({
                  ...prev,
                  conversationId: payload.conversation_id || prev.conversationId,
                }));
                activeAssistantMessageId.current = null;
              }
            } catch {
              // Ignore malformed SSE chunks.
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") {
          setState((prev) => ({ ...prev, error: "Copilot request cancelled." }));
        } else if (err instanceof Error) {
          setState((prev) => ({ ...prev, error: err.message }));
        } else {
          setState((prev) => ({ ...prev, error: "Unknown Copilot error" }));
        }
      } finally {
        setState((prev) => ({ ...prev, isLoading: false }));
      }
    },
    [appendAssistantToken, state.conversationId, state.messages]
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    activeAssistantMessageId.current = null;
    setState(INITIAL_STATE);
  }, []);

  return {
    ...state,
    sendMessage,
    cancel,
    reset,
  };
}
