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
    <nav className="border-b border-slate-300/10 bg-slate-950/50 px-4 backdrop-blur-xl md:px-6">
      <div className="mx-auto flex max-w-[1500px] gap-2 overflow-x-auto py-3">
        {boards.map((board) => (
          <button
            key={board.id}
            onClick={() => onChange(board.id)}
            disabled={disabled}
            className={`group min-w-[220px] rounded-xl border px-4 py-3 text-left transition-all duration-200 ${
              board.id === activeBoardId
                ? "border-cyan-300/45 bg-cyan-300/10 text-slate-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_14px_28px_rgba(22,140,171,0.2)]"
                : "border-slate-300/12 bg-slate-950/30 text-slate-400 hover:border-emerald-200/25 hover:bg-slate-900/70 hover:text-slate-200"
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            <div className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${
                  board.id === activeBoardId ? "bg-cyan-200 shadow-[0_0_14px_rgba(103,232,249,0.75)]" : "bg-slate-600"
                }`}
              />
              <span className="text-sm font-semibold tracking-tight">{board.nameZh}</span>
            </div>
            <div className="mt-1.5 line-clamp-2 text-[11px] leading-4 text-slate-500 transition-colors group-hover:text-slate-400">
              {board.name}
            </div>
          </button>
        ))}
      </div>
    </nav>
  );
}
