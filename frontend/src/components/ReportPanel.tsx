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
}

export default function ReportPanel({ text, isStreaming, phase, elapsedTime }: Props) {
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
      <div ref={endRef} />
    </div>
  );
}
