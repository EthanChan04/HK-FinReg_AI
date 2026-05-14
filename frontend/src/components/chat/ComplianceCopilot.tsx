"use client";

import { useMemo, useState } from "react";

import ChatInput from "@/components/chat/ChatInput";
import ChatMessageList from "@/components/chat/ChatMessageList";
import CitationCards from "@/components/chat/CitationCards";
import SuggestedPrompts from "@/components/chat/SuggestedPrompts";
import ToolCallTimeline from "@/components/chat/ToolCallTimeline";
import { useCopilotChat } from "@/hooks/useCopilotChat";
import { getCopilotPrompts } from "@/lib/copilotPrompts";
import type { BankBoardId, CopilotCaseContext } from "@/types";

interface ComplianceCopilotProps {
  activeBoardId: BankBoardId;
  caseContext: CopilotCaseContext;
}

export default function ComplianceCopilot({ activeBoardId, caseContext }: ComplianceCopilotProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const copilot = useCopilotChat();
  const prompts = useMemo(() => getCopilotPrompts(activeBoardId), [activeBoardId]);

  const handlePrompt = (prompt: string) => {
    void copilot.sendMessage(prompt, caseContext);
  };

  const handleSubmit = (value: string) => {
    void copilot.sendMessage(value, caseContext);
  };

  const panelBody = (
    <>
      <div className="px-3 pt-2 text-[11px] text-slate-400">
        {copilot.intent ? `Routing Intent: ${copilot.intent}` : "Routing Intent: pending"}
        {typeof copilot.unsupportedClaimRate === "number" && (
          <span className="ml-2 rounded-full border border-amber-300/30 bg-amber-500/10 px-2 py-0.5 text-[10px] text-amber-200">
            Citation risk: {Math.round(copilot.unsupportedClaimRate * 100)}%
          </span>
        )}
      </div>

      {copilot.error && (
        <div className="mx-3 mt-2 rounded-lg border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-200">
          {copilot.error}
        </div>
      )}

      <SuggestedPrompts prompts={prompts} disabled={copilot.isLoading} onSelect={handlePrompt} />
      <ChatMessageList messages={copilot.messages} isLoading={copilot.isLoading} />
      <ToolCallTimeline events={copilot.toolEvents} />
      <CitationCards evidence={copilot.evidenceChunks} />
      <ChatInput isLoading={copilot.isLoading} onSubmit={handleSubmit} onCancel={copilot.cancel} />
    </>
  );

  return (
    <>
      <aside
        className={`hidden h-full shrink-0 border-l border-slate-300/10 bg-slate-950/40 backdrop-blur-sm transition-all duration-200 lg:flex lg:flex-col ${
          collapsed ? "w-14" : "w-[380px]"
        }`}
      >
        <div className="flex items-center justify-between border-b border-slate-300/15 px-3 py-3">
          {!collapsed && (
            <div>
              <h3 className="text-sm font-semibold tracking-tight text-slate-100">Compliance Copilot</h3>
              <p className="text-[10px] text-slate-400">Bilingual workflow-aware assistant</p>
            </div>
          )}
          <button
            aria-label={collapsed ? "Expand Copilot panel" : "Collapse Copilot panel"}
            onClick={() => setCollapsed((prev) => !prev)}
            className="rounded-md border border-slate-300/20 bg-slate-900/60 px-2 py-1 text-[11px] font-medium text-slate-300 hover:border-cyan-300/40 hover:text-cyan-100"
          >
            {collapsed ? "Open" : "Hide"}
          </button>
        </div>

        {!collapsed && panelBody}
      </aside>

      <button
        aria-label="Open Compliance Copilot"
        onClick={() => setMobileOpen(true)}
        className="fixed bottom-4 right-4 z-40 rounded-full border border-cyan-300/40 bg-gradient-to-r from-cyan-500/25 to-blue-500/20 px-4 py-2 text-xs font-semibold text-cyan-100 shadow-[0_10px_24px_rgba(31,140,179,0.35)] lg:hidden"
      >
        Open Copilot
      </button>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-[2px] lg:hidden">
          <div className="absolute right-0 top-0 h-full w-[92%] max-w-[420px] border-l border-slate-300/15 bg-[#0b162a]">
            <div className="flex items-center justify-between border-b border-slate-300/15 px-3 py-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">Compliance Copilot</h3>
                <p className="text-[10px] text-slate-400">Mobile Assistant Panel</p>
              </div>
              <button
                aria-label="Close Compliance Copilot"
                onClick={() => setMobileOpen(false)}
                className="rounded-md border border-slate-300/20 bg-slate-900/60 px-2 py-1 text-[11px] font-medium text-slate-300"
              >
                Close
              </button>
            </div>
            <div className="flex h-[calc(100%-57px)] flex-col">{panelBody}</div>
          </div>
        </div>
      )}
    </>
  );
}
