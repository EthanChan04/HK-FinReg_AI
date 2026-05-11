// Deep Research Plan Panel — 深度研究计划面板
// 展示研究目标、子问题列表、证据差距和预期输出章节
"use client";

import type { ResearchPlan, EvidenceChunk } from "@/types";

interface EvidenceGap {
  sub_question_id: string;
  reason: string;
}

interface Props {
  researchPlan: ResearchPlan | null;
  evidenceBySubquestion: Record<string, EvidenceChunk[]>;
  evidenceGaps: EvidenceGap[];
  isLoading: boolean;
}

function Skeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="space-y-2">
        <div className="h-3 bg-white/[0.06] rounded w-1/4" />
        <div className="h-4 bg-white/[0.06] rounded w-3/4" />
        <div className="h-3 bg-white/[0.06] rounded w-1/2" />
      </div>
      <div className="space-y-2">
        <div className="h-3 bg-white/[0.06] rounded w-1/5" />
        {[1, 2, 3].map((i) => (
          <div key={i} className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-3 space-y-2">
            <div className="h-3 bg-white/[0.06] rounded w-1/2" />
            <div className="h-3 bg-white/[0.06] rounded w-2/3" />
          </div>
        ))}
      </div>
      <div className="space-y-2">
        <div className="h-3 bg-white/[0.06] rounded w-1/6" />
        {[1, 2].map((i) => (
          <div key={i} className="h-12 bg-white/[0.06] rounded-xl" />
        ))}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center py-12">
      <div className="text-center space-y-2">
        <span className="text-3xl block opacity-30">📋</span>
        <p className="text-sm text-gray-500">No research plan available</p>
        <p className="text-[11px] text-gray-700">
          Deep research planning will appear here when activated
        </p>
      </div>
    </div>
  );
}

export default function DeepResearchPlanPanel({
  researchPlan,
  evidenceBySubquestion,
  evidenceGaps,
  isLoading,
}: Props) {
  if (isLoading) {
    return (
      <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-px w-6 bg-gradient-to-r from-emerald-500/60 to-transparent" />
          <span className="text-xs font-medium text-emerald-400 tracking-widest uppercase">
            Research Plan
          </span>
          <div className="h-px w-6 bg-gradient-to-l from-emerald-500/60 to-transparent" />
        </div>
        <Skeleton />
      </div>
    );
  }

  if (!researchPlan) {
    return (
      <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-px w-6 bg-gradient-to-r from-emerald-500/60 to-transparent" />
          <span className="text-xs font-medium text-emerald-400 tracking-widest uppercase">
            Research Plan
          </span>
          <div className="h-px w-6 bg-gradient-to-l from-emerald-500/60 to-transparent" />
        </div>
        <EmptyState />
      </div>
    );
  }

  const subquestions = researchPlan.sub_questions ?? [];
  const outputSections = researchPlan.expected_output_sections ?? [];

  return (
    <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
      <div className="flex items-center gap-2 mb-4 shrink-0">
        <div className="h-px w-6 bg-gradient-to-r from-emerald-500/60 to-transparent" />
        <span className="text-xs font-medium text-emerald-400 tracking-widest uppercase">
          Research Plan
        </span>
        <div className="h-px w-6 bg-gradient-to-l from-emerald-500/60 to-transparent" />
      </div>

      {/* Research Goal */}
      {researchPlan.research_goal && (
        <div className="mb-4">
          <span className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">
            Research Goal
          </span>
          <p className="mt-1 text-sm text-gray-300 leading-relaxed">
            {researchPlan.research_goal}
          </p>
        </div>
      )}

      {/* Sub-questions */}
      {subquestions.length > 0 && (
        <div className="mb-4">
          <span className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">
            Sub-questions ({subquestions.length})
          </span>
          <div className="mt-2 space-y-1.5">
            {subquestions.map((sq: Record<string, unknown>, idx: number) => {
              const sqId = (sq.id as string) || String(idx);
              const sqText = (sq.question as string) || (sq.text as string) || "";
              const hasEvidence =
                evidenceBySubquestion[sqId] && evidenceBySubquestion[sqId].length > 0;
              return (
                <div
                  key={sqId}
                  className={`flex items-start gap-2.5 px-3 py-2 rounded-lg border text-xs transition-colors ${
                    hasEvidence
                      ? "bg-emerald-500/5 border-emerald-500/15"
                      : "bg-white/[0.02] border-white/[0.06]"
                  }`}
                >
                  <span className="mt-0.5 shrink-0 text-[10px] font-mono text-gray-600 w-5">
                    {idx + 1}.
                  </span>
                  <span
                    className={`mt-0.5 shrink-0 ${
                      hasEvidence ? "text-emerald-400" : "text-gray-500"
                    }`}
                  >
                    {hasEvidence ? "✓" : "○"}
                  </span>
                  <span className="flex-1 text-gray-300">{sqText}</span>
                  {hasEvidence && (
                    <span className="shrink-0 text-[10px] text-emerald-500/60 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                      {evidenceBySubquestion[sqId].length} evidence
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Evidence Gaps */}
      {evidenceGaps.length > 0 && (
        <div className="mb-4">
          <span className="text-[10px] font-medium text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
            <span>⚠</span>
            Evidence Gaps ({evidenceGaps.length})
          </span>
          <div className="mt-2 space-y-1.5">
            {evidenceGaps.map((gap, idx) => (
              <div
                key={idx}
                className="bg-amber-500/5 border border-amber-500/15 rounded-xl px-3 py-2.5"
              >
                <div className="flex items-start gap-2">
                  <span className="text-amber-500/60 text-[10px] mt-0.5 shrink-0">⚠</span>
                  <div>
                    <span className="text-[11px] text-amber-300/80 font-mono">
                      {gap.sub_question_id}
                    </span>
                    <p className="text-[11px] text-gray-400 mt-0.5">{gap.reason}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expected Output Sections */}
      {outputSections.length > 0 && (
        <div>
          <span className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">
            Report Sections ({outputSections.length})
          </span>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {outputSections.map((section, idx) => (
              <span
                key={idx}
                className="text-[10px] px-2 py-1 rounded-lg bg-white/[0.04] border border-white/[0.06] text-gray-400"
              >
                {section}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
