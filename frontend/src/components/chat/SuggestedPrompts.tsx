"use client";

interface SuggestedPromptsProps {
  prompts: string[];
  disabled?: boolean;
  onSelect: (prompt: string) => void;
}

export default function SuggestedPrompts({ prompts, disabled = false, onSelect }: SuggestedPromptsProps) {
  return (
    <div className="flex flex-wrap gap-2 px-3 pb-2">
      {prompts.map((prompt) => (
        <button
          key={prompt}
          onClick={() => onSelect(prompt)}
          disabled={disabled}
          className="rounded-full border border-slate-300/20 bg-slate-900/60 px-2.5 py-1 text-[10px] text-slate-300 transition-colors hover:border-cyan-300/40 hover:bg-cyan-500/10 hover:text-cyan-100 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {prompt}
        </button>
      ))}
    </div>
  );
}
