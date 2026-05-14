"use client";

import type { BankBoardConfig, BankBoardId } from "@/types";

interface WorkspaceNavProps {
  boards: BankBoardConfig[];
  activeBoardId: BankBoardId;
  disabled?: boolean;
  onChange: (boardId: BankBoardId) => void;
}

export default function WorkspaceNav({
  boards,
  activeBoardId,
  disabled = false,
  onChange,
}: WorkspaceNavProps) {
  return (
    <nav className="border-b border-slate-300/10 bg-slate-900/30 px-4 backdrop-blur-sm md:px-6">
      <div className="mx-auto flex max-w-[1500px] gap-2.5 overflow-x-auto py-3">
        {boards.map((board) => (
          <button
            key={board.id}
            onClick={() => onChange(board.id)}
            disabled={disabled}
            className={`min-w-[220px] rounded-2xl border px-4 py-3 text-left transition-all duration-200 ${
              board.id === activeBoardId
                ? "border-cyan-300/35 bg-cyan-500/12 text-slate-100 shadow-[0_12px_24px_rgba(22,140,171,0.18)]"
                : "border-slate-300/15 bg-slate-900/40 text-slate-400 hover:border-slate-300/25 hover:bg-slate-800/60 hover:text-slate-200"
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            <div className="text-sm font-semibold tracking-tight">{board.nameZh}</div>
            <div className="mt-1 line-clamp-2 text-[11px] text-slate-500">{board.name}</div>
          </button>
        ))}
      </div>
    </nav>
  );
}
