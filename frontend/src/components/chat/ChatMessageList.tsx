"use client";

import type { CopilotMessage } from "@/types";

interface ChatMessageListProps {
  messages: CopilotMessage[];
  isLoading: boolean;
}

export default function ChatMessageList({ messages, isLoading }: ChatMessageListProps) {
  return (
    <div className="flex-1 space-y-3 overflow-y-auto px-3 py-3">
      {messages.length === 0 && (
        <div className="rounded-xl border border-slate-300/20 bg-slate-900/55 p-3 text-xs leading-6 text-slate-400">
          Ask Compliance Copilot about obligations, risk paths, current report gaps, or workflow recommendations.
        </div>
      )}

      {messages.map((message) => {
        const isUser = message.role === "user";
        return (
          <div key={message.id} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[88%] rounded-2xl px-3 py-2 text-xs leading-6 ${
                isUser
                  ? "border border-cyan-300/35 bg-gradient-to-r from-cyan-500/20 to-blue-500/15 text-cyan-50"
                  : "border border-slate-300/20 bg-slate-900/60 text-slate-100"
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
            </div>
          </div>
        );
      })}

      {isLoading && (
        <div className="text-[11px] text-slate-400">Copilot is analyzing context and preparing response...</div>
      )}
    </div>
  );
}
