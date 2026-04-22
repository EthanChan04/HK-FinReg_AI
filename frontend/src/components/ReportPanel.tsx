// 流式 Markdown 报告渲染面板 — 增强版
// react-markdown 富文本渲染 + 平滑自动滚动 + 进度指示
"use client";

import { useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  text: string;
  isStreaming: boolean;
  phase: "idle" | "agents" | "streaming" | "done";
  elapsedTime: number;
  confidenceScore?: number | null;
  confidenceWarning?: string | null;
  reasoningConfidence?: number | null;
  reviewerConfidence?: number | null;
  crossValidationPassed?: boolean | null;
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
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

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
      {confidenceScore !== null && confidenceScore !== undefined && phase === "done" && (
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
            {(reasoningConfidence !== null || reviewerConfidence !== null) && (
              <div className="flex gap-4 text-xs text-gray-400 mt-1 pt-2 border-t border-white/[0.06]">
                {reasoningConfidence !== null && (
                  <span>
                    🧠 推理：<strong className={reasoningConfidence < 0.5 ? "text-red-400" : reasoningConfidence < 0.7 ? "text-yellow-400" : "text-green-400"}>
                      {(reasoningConfidence * 100).toFixed(0)}%
                    </strong>
                  </span>
                )}
                {reviewerConfidence !== null && (
                  <span>
                    ⚖️ 審查：<strong className={reviewerConfidence < 0.5 ? "text-red-400" : reviewerConfidence < 0.7 ? "text-yellow-400" : "text-green-400"}>
                      {(reviewerConfidence * 100).toFixed(0)}%
                    </strong>
                  </span>
                )}
                {crossValidationPassed !== null && (
                  <span>
                    {crossValidationPassed ? "✅" : "⚠️"} 交叉驗證{crossValidationPassed ? "通過" : "未通過"}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}
