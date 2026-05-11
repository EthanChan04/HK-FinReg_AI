// Evidence Chunks Panel — 可折叠的证据卡片列表
// 显示检索到的证据片段，含来源、监管机构、分数等元信息
"use client";

import { useState } from "react";
import type { EvidenceChunk } from "@/types";

interface Props {
  evidence: EvidenceChunk[];
  isLoading: boolean;
}

function ScoreBar({ score }: { score: number | null | undefined }) {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  const color =
    pct >= 80
      ? "bg-emerald-500"
      : pct >= 60
        ? "bg-blue-500"
        : pct >= 40
          ? "bg-amber-500"
          : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] font-mono tabular-nums text-gray-400 w-8 text-right">
        {pct}%
      </span>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4 space-y-3 animate-pulse"
        >
          <div className="h-3 bg-white/[0.06] rounded w-1/3" />
          <div className="h-4 bg-white/[0.06] rounded w-2/3" />
          <div className="flex gap-2">
            <div className="h-4 bg-white/[0.06] rounded w-16" />
            <div className="h-4 bg-white/[0.06] rounded w-20" />
          </div>
          <div className="h-12 bg-white/[0.06] rounded" />
          <div className="h-2 bg-white/[0.06] rounded w-full" />
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center py-12">
      <div className="text-center space-y-2">
        <span className="text-3xl block opacity-30">🔍</span>
        <p className="text-sm text-gray-500">No evidence retrieved yet</p>
        <p className="text-[11px] text-gray-700">
          Evidence chunks will appear here after retrieval
        </p>
      </div>
    </div>
  );
}

export default function EvidencePanel({ evidence, isLoading }: Props) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  if (isLoading) {
    return (
      <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-px w-6 bg-gradient-to-r from-blue-500/60 to-transparent" />
          <span className="text-xs font-medium text-blue-400 tracking-widest uppercase">
            Evidence
          </span>
          <div className="h-px w-6 bg-gradient-to-l from-blue-500/60 to-transparent" />
        </div>
        <Skeleton />
      </div>
    );
  }

  if (evidence.length === 0) {
    return (
      <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-px w-6 bg-gradient-to-r from-blue-500/60 to-transparent" />
          <span className="text-xs font-medium text-blue-400 tracking-widest uppercase">
            Evidence
          </span>
          <div className="h-px w-6 bg-gradient-to-l from-blue-500/60 to-transparent" />
        </div>
        <EmptyState />
      </div>
    );
  }

  return (
    <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
      <div className="flex items-center gap-2 mb-4 shrink-0">
        <div className="h-px w-6 bg-gradient-to-r from-blue-500/60 to-transparent" />
        <span className="text-xs font-medium text-blue-400 tracking-widest uppercase">
          Evidence
        </span>
        <div className="h-px w-6 bg-gradient-to-l from-blue-500/60 to-transparent" />
        <span className="text-[10px] text-gray-500 font-mono ml-auto">
          {evidence.length} chunk{evidence.length !== 1 ? "s" : ""}
        </span>
      </div>

      <div className="space-y-2 overflow-y-auto max-h-[500px] pr-1">
        {evidence.map((chunk) => {
          const isExpanded = expandedIds.has(chunk.evidence_id);
          return (
            <div
              key={chunk.evidence_id}
              className="bg-white/[0.02] border border-white/[0.06] rounded-xl transition-all duration-200 hover:border-white/[0.1]"
            >
              {/* Collapsible header */}
              <button
                onClick={() => toggleExpand(chunk.evidence_id)}
                className="w-full flex items-start gap-3 px-4 py-3 text-left"
              >
                <span
                  className={`mt-0.5 text-[10px] text-gray-600 transition-transform duration-200 ${
                    isExpanded ? "rotate-90" : ""
                  }`}
                >
                  ▶
                </span>
                <div className="flex-1 min-w-0 space-y-1.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    {chunk.title && (
                      <span className="text-xs font-medium text-gray-300 truncate max-w-[200px]">
                        {chunk.title}
                      </span>
                    )}
                    {chunk.regulator && (
                      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 whitespace-nowrap">
                        {chunk.regulator}
                      </span>
                    )}
                    {chunk.retrieval_method && (
                      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 whitespace-nowrap">
                        {chunk.retrieval_method}
                      </span>
                    )}
                  </div>

                  {/* Score bar — always visible */}
                  <ScoreBar score={chunk.score} />

                  {/* Metadata row */}
                  <div className="flex items-center gap-3 text-[10px] text-gray-600">
                    {chunk.page != null && <span>p.{chunk.page}</span>}
                    {chunk.section_title && (
                      <span className="truncate">{chunk.section_title}</span>
                    )}
                    {chunk.jurisdiction && (
                      <span className="ml-auto">{chunk.jurisdiction}</span>
                    )}
                  </div>
                </div>
              </button>

              {/* Expanded content */}
              <div
                className={`overflow-hidden transition-all duration-300 ease-in-out ${
                  isExpanded ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"
                }`}
              >
                <div className="px-4 pb-4 pt-0 border-t border-white/[0.04] mt-2">
                  <p className="text-[11px] text-gray-400 leading-relaxed whitespace-pre-wrap">
                    {chunk.text}
                  </p>
                  <div className="flex flex-wrap gap-2 mt-2 text-[10px] text-gray-600">
                    {chunk.evidence_id && (
                      <span className="font-mono">ID: {chunk.evidence_id}</span>
                    )}
                    {chunk.doc_id && <span className="font-mono">Doc: {chunk.doc_id}</span>}
                    {chunk.doc_type && <span>Type: {chunk.doc_type}</span>}
                    {chunk.hierarchy_path && <span>Path: {chunk.hierarchy_path}</span>}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
