"use client";

import type { BankWorkflowConfig } from "@/types";

interface WorkflowSelectorProps {
  workflows: BankWorkflowConfig[];
  activeWorkflowId: string;
  disabled?: boolean;
  onChange: (workflow: BankWorkflowConfig) => void;
}

function engineLabel(engineMode: BankWorkflowConfig["engineMode"]) {
  if (engineMode === "rag") return "RAG";
  if (engineMode === "rag_kag") return "RAG + KAG";
  if (engineMode === "deepresearch") return "Deep Research";
  return "Human Review";
}

export default function WorkflowSelector({
  workflows,
  activeWorkflowId,
  disabled = false,
  onChange,
}: WorkflowSelectorProps) {
  return (
    <div className="space-y-2.5">
      {workflows.map((workflow) => (
        <button
          key={workflow.id}
          onClick={() => onChange(workflow)}
          disabled={disabled}
          className={`w-full rounded-xl border px-4 py-3 text-left transition-all duration-200 ${
            workflow.id === activeWorkflowId
              ? "border-cyan-300/40 bg-gradient-to-r from-cyan-400/14 to-emerald-300/10 text-slate-100 shadow-[0_10px_22px_rgba(25,129,173,0.16)]"
              : "border-slate-300/12 bg-slate-950/30 text-slate-400 hover:border-slate-300/25 hover:bg-slate-900/70 hover:text-slate-200"
          } disabled:cursor-not-allowed disabled:opacity-40`}
        >
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-semibold tracking-tight">{workflow.nameZh}</span>
            <span className="shrink-0 rounded-md border border-slate-300/20 bg-slate-950/55 px-2.5 py-1 text-[10px] font-medium text-slate-400">
              {engineLabel(workflow.engineMode)}
            </span>
          </div>
          <p className="mt-1 text-xs leading-5 text-slate-500">{workflow.description}</p>
        </button>
      ))}
    </div>
  );
}
