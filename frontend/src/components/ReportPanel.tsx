// 流式 Markdown 报告渲染面板 — 增强版
// react-markdown 富文本渲染 + 平滑自动滚动 + 进度指示 + HITL 人工接管
"use client";

import { useRef, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_PROXY_BASE = "/api/backend";

function buildApiHeaders(): HeadersInit {
  return { "Content-Type": "application/json" };
}

interface Props {
  text: string;
  isStreaming: boolean;
  phase: "idle" | "agents" | "streaming" | "done" | "action_required";
  elapsedTime: number;
  confidenceScore?: number | null;
  confidenceWarning?: string | null;
  reasoningConfidence?: number | null;
  reviewerConfidence?: number | null;
  crossValidationPassed?: boolean | null;
  // Phase 1: HITL
  workflowRunId?: string | null;
  humanReviewRequired?: boolean;
  currentGate?: string | null;
  gateMessage?: string | null;
  // Phase 1: 恢复后回调
  onResumeResult?: (finalReport: string, approved: boolean) => void;
}

export default function ReportPanel({
  text,
  isStreaming,
  phase,
  elapsedTime,
  confidenceScore,
  confidenceWarning,
  reasoningConfidence,
  reviewerConfidence,
  crossValidationPassed,
  workflowRunId,
  humanReviewRequired,
  currentGate,
  gateMessage,
  onResumeResult,
}: Props) {
  void isStreaming;
  const containerRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const [reviewNotes, setReviewNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (phase === "streaming") {
      endRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [text, phase]);

  // Phase: idle — 空白占位
  if (phase === "idle") {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-3">
          <span className="text-5xl block opacity-30">📋</span>
          <p className="text-sm text-gray-600">合規審查報告將在此處以流式方式渲染</p>
          <p className="text-[11px] text-gray-700">選擇業務模組 → 提交請求 → 等待 Agent 分析</p>
        </div>
      </div>
    );
  }

  // Phase: agents — agents 正在执行，报告尚未开始
  if (phase === "agents" && !text) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="flex justify-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: "0ms" }} />
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: "150ms" }} />
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: "300ms" }} />
          </div>
          <p className="text-sm text-blue-300/70">多智能體正在協同工作中...</p>
          <p className="text-[11px] text-gray-600 font-mono">{elapsedTime}s elapsed</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto px-6 py-5">
      <article className="prose-report">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        {(phase === "streaming" || (phase === "agents" && text)) && (
          <span className="inline-block w-2 h-5 bg-blue-400 animate-pulse ml-0.5 align-middle rounded-sm" />
        )}
      </article>
      {/* P2: 三维置信度徽章 */}
      {(confidenceScore != null && phase === "done") && (
        <div className={`mt-4 px-4 py-3 rounded-lg border ${
          confidenceScore < 0.5
            ? "bg-red-900/20 border-red-500/40"
            : confidenceScore < 0.7
            ? "bg-yellow-900/20 border-yellow-500/40"
            : "bg-green-900/20 border-green-500/40"
        }`}>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">
                {confidenceScore < 0.5 ? "⚠️" : confidenceScore < 0.7 ? "📋" : "✅"}
              </span>
              <span className="text-sm">
                檢索置信度：<strong>{(confidenceScore * 100).toFixed(0)}%</strong>
              </span>
              {confidenceWarning && (
                <span className="text-xs opacity-80 ml-2">{confidenceWarning}</span>
              )}
            </div>
            {/* 三维置信度详情 */}
            {(reasoningConfidence != null || reviewerConfidence != null) && (
              <div className="flex gap-4 text-xs text-gray-400 mt-1 pt-2 border-t border-white/[0.06]">
                {reasoningConfidence != null && (
                  <span>
                    🧠 推理：<strong className={reasoningConfidence < 0.5 ? "text-red-400" : reasoningConfidence < 0.7 ? "text-yellow-400" : "text-green-400"}>
                      {(reasoningConfidence * 100).toFixed(0)}%
                    </strong>
                  </span>
                )}
                {reviewerConfidence != null && (
                  <span>
                    ⚖️ 審查：<strong className={reviewerConfidence < 0.5 ? "text-red-400" : reviewerConfidence < 0.7 ? "text-yellow-400" : "text-green-400"}>
                      {(reviewerConfidence * 100).toFixed(0)}%
                    </strong>
                  </span>
                )}
                {crossValidationPassed != null && (
                  <span>
                    {crossValidationPassed ? "✅" : "⚠️"} 交叉驗證{crossValidationPassed ? "通過" : "未通過"}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Phase 1: HITL 人工接管区 */}
      {phase === "action_required" && humanReviewRequired && (
        <div className="mt-4 px-4 py-4 rounded-lg border border-amber-500/40 bg-amber-900/15">
          <div className="flex flex-col gap-3">
            {/* 暂停原因 */}
            <div className="flex items-center gap-2">
              <span className="text-lg">🔍</span>
              <span className="text-sm font-semibold text-amber-300">工作流已暫停 — 等待人工審查</span>
            </div>

            {/* Gate 信息 */}
            {currentGate && (
              <div className="text-xs text-gray-400 bg-black/20 rounded px-3 py-2 space-y-1">
                <p>
                  <span className="text-gray-500">暫停原因：</span>
                  <span className="text-amber-300 font-medium">
                    {currentGate === "low_confidence_gate" && "低置信度警告"}
                    {currentGate === "missing_evidence_gate" && "證據不足且已達檢索上限"}
                    {currentGate === "manual_approval_gate" && "高風險場景需人工批准"}
                    {!["low_confidence_gate", "missing_evidence_gate", "manual_approval_gate"].includes(currentGate) && currentGate}
                  </span>
                </p>
                {gateMessage && <p className="text-gray-500">{gateMessage}</p>}
              </div>
            )}

            {/* 关键指标 */}
            <div className="flex gap-3 text-xs">
              {confidenceScore != null && (
                <span className={`px-2 py-1 rounded ${
                  confidenceScore < 0.5 ? "bg-red-900/30 text-red-400" : "bg-yellow-900/30 text-yellow-400"
                }`}>
                  檢索 {(confidenceScore * 100).toFixed(0)}%
                </span>
              )}
              {reasoningConfidence != null && (
                <span className={`px-2 py-1 rounded ${
                  reasoningConfidence < 0.5 ? "bg-red-900/30 text-red-400" : "bg-yellow-900/30 text-yellow-400"
                }`}>
                  推理 {(reasoningConfidence * 100).toFixed(0)}%
                </span>
              )}
              {crossValidationPassed != null && (
                <span className={`px-2 py-1 rounded ${
                  crossValidationPassed ? "bg-green-900/30 text-green-400" : "bg-red-900/30 text-red-400"
                }`}>
                  交叉驗證{crossValidationPassed ? "通過" : "未通過"}
                </span>
              )}
            </div>

            {/* 审查批注输入 */}
            <div className="space-y-2">
              <textarea
                value={reviewNotes}
                onChange={(e) => setReviewNotes(e.target.value)}
                placeholder="輸入審查批註（可選）..."
                className="w-full bg-black/20 border border-white/[0.08] rounded-lg px-3 py-2 text-xs text-gray-300 font-mono resize-none outline-none focus:border-amber-500/40 transition-colors"
                rows={3}
              />
              <div className="flex gap-2">
                <button
                  onClick={async () => {
                    if (!workflowRunId) return;
                    setSubmitting(true);
                    try {
                      const resp = await fetch(`${API_PROXY_BASE}/api/v1/review-queue/${workflowRunId}/resume`, {
                        method: "POST",
                        headers: buildApiHeaders(),
                        body: JSON.stringify({ notes: reviewNotes, reviewed_by: "web_user" }),
                      });
                      if (resp.ok) {
                        const data = await resp.json();
                        const finalReport = data.final_report || "";
                        if (onResumeResult) {
                          onResumeResult(finalReport, true);
                        }
                      }
                    } catch (err) {
                      console.error("Resume failed:", err);
                    } finally {
                      setSubmitting(false);
                    }
                  }}
                  disabled={submitting || !workflowRunId}
                  className="flex-1 py-2 rounded-lg text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {submitting ? "提交中..." : "✅ 批准並恢復執行"}
                </button>
                <button
                  onClick={async () => {
                    if (!workflowRunId) return;
                    setSubmitting(true);
                    try {
                      const resp = await fetch(`${API_PROXY_BASE}/api/v1/review-queue/${workflowRunId}/reject`, {
                        method: "POST",
                        headers: buildApiHeaders(),
                        body: JSON.stringify({ notes: reviewNotes || "駁回", reviewed_by: "web_user" }),
                      });
                      if (resp.ok) {
                        const data = await resp.json();
                        const finalReport = data.final_report || "❌ 報告已被人工駁回";
                        if (onResumeResult) {
                          onResumeResult(finalReport, false);
                        }
                      }
                    } catch (err) {
                      console.error("Reject failed:", err);
                    } finally {
                      setSubmitting(false);
                    }
                  }}
                  disabled={submitting || !workflowRunId}
                  className="flex-1 py-2 rounded-lg text-xs font-medium text-white bg-red-600 hover:bg-red-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {submitting ? "提交中..." : "❌ 駁回"}
                </button>
              </div>
            </div>

            {/* workflow_run_id 显示 */}
            {workflowRunId && (
              <p className="text-[10px] text-gray-600 font-mono">ID: {workflowRunId}</p>
            )}
          </div>
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}
