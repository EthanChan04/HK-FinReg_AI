"use client";

import type { EvidenceChunk } from "@/types";

interface CitationCardsProps {
  evidence: EvidenceChunk[];
}

export default function CitationCards({ evidence }: CitationCardsProps) {
  if (!evidence.length) return null;

  return (
    <div className="space-y-2 border-t border-slate-300/15 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">Evidence Cards</p>
      {evidence.slice(0, 4).map((item) => (
        <div key={item.evidence_id} className="rounded-lg border border-slate-300/20 bg-slate-900/60 p-2">
          <p className="text-[11px] font-semibold text-slate-100">
            {item.evidence_id} {item.title ? `· ${item.title}` : ""}
          </p>
          <p className="mt-0.5 text-[10px] text-slate-500">
            {item.regulator || "Unknown regulator"}
            {typeof item.page === "number" ? ` · p.${item.page}` : ""}
          </p>
          <p className="mt-1 line-clamp-3 text-[10px] text-slate-400">{item.text}</p>
        </div>
      ))}
    </div>
  );
}
