"use client";

import { useState } from "react";

interface ChatInputProps {
  isLoading: boolean;
  onSubmit: (value: string) => void;
  onCancel: () => void;
}

export default function ChatInput({ isLoading, onSubmit, onCancel }: ChatInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;
    onSubmit(trimmed);
    setValue("");
  };

  return (
    <div className="border-t border-slate-300/15 bg-slate-950/45 p-3">
      <textarea
        aria-label="Ask Compliance Copilot"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            event.preventDefault();
            handleSubmit();
          }
        }}
        disabled={isLoading}
        className="h-24 w-full resize-none rounded-xl border border-slate-300/20 bg-slate-900/65 p-3 text-[13px] text-slate-100 outline-none transition-colors focus:border-cyan-300/40 focus:ring-2 focus:ring-cyan-400/10 disabled:opacity-60"
        placeholder="Ask Compliance Copilot... (Ctrl/Cmd + Enter to send)"
      />

      <div className="mt-2 flex items-center justify-end gap-2">
        <button
          onClick={onCancel}
          disabled={!isLoading}
          className="rounded-lg border border-rose-400/40 bg-rose-500/10 px-3 py-1.5 text-[11px] font-medium text-rose-200 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={isLoading || !value.trim()}
          className="rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 px-3 py-1.5 text-[11px] font-semibold text-white shadow-[0_8px_16px_rgba(22,133,173,0.28)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}
