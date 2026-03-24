// 主页面：4 模块 Dashboard + SSE 流式交互 — 200s UX 优化版
"use client";

import { useState } from "react";
import { useAgentStream } from "@/hooks/useAgentStream";
import AgentTimeline from "@/components/AgentTimeline";
import ReportPanel from "@/components/ReportPanel";
import { modules } from "@/lib/modules";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default function Home() {
  const [activeModule, setActiveModule] = useState(0);
  const [inputText, setInputText] = useState(modules[0].defaultInput);
  const stream = useAgentStream();

  const currentModule = modules[activeModule];

  const handleModuleSwitch = (idx: number) => {
    if (stream.isStreaming) return;
    setActiveModule(idx);
    setInputText(modules[idx].defaultInput);
    stream.reset();
  };

  const handleSubmit = () => {
    if (!inputText.trim() || stream.isStreaming) return;
    stream.startStream(currentModule.endpoint, inputText);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* ===== Header ===== */}
      <header className="border-b border-white/[0.06] px-6 py-5">
        <div className="max-w-[1400px] mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-gradient">
              HK-FinReg AI
            </h1>
            <p className="text-sm text-gray-400 mt-1 tracking-wide">
              Multi-Agent Compliance Engine&ensp;·&ensp;Hybrid RAG + Cohere Reranker
            </p>
          </div>

          {/* Streaming indicator / timer */}
          {stream.isStreaming && (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2.5 text-xs bg-blue-500/10 border border-blue-500/20 px-4 py-2 rounded-full">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
                </span>
                <span className="text-blue-300 font-mono tabular-nums">
                  {formatTime(stream.elapsedTime)}
                </span>
                <span className="text-blue-400/60">
                  {stream.phase === "agents" ? "Agent Processing" : "Streaming Report"}
                </span>
              </div>
              <button
                onClick={stream.cancelStream}
                className="text-xs text-red-400 hover:text-red-300 border border-red-500/30 hover:border-red-500/50 px-3 py-2 rounded-full transition-all"
              >
                ✕ Cancel
              </button>
            </div>
          )}

          {/* Completed indicator */}
          {stream.phase === "done" && !stream.isStreaming && (
            <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-4 py-2 rounded-full">
              <span>✅</span>
              <span>Completed in {formatTime(stream.elapsedTime)}</span>
            </div>
          )}
        </div>
      </header>

      {/* ===== Module Tabs ===== */}
      <nav className="border-b border-white/[0.06] px-6">
        <div className="max-w-[1400px] mx-auto flex gap-1 py-2 overflow-x-auto">
          {modules.map((mod, idx) => (
            <button
              key={mod.id}
              onClick={() => handleModuleSwitch(idx)}
              disabled={stream.isStreaming}
              className={`
                flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap
                ${
                  idx === activeModule
                    ? "bg-white/[0.08] text-white border border-white/[0.1]"
                    : "text-gray-500 hover:text-gray-300 hover:bg-white/[0.03]"
                }
                disabled:opacity-40 disabled:cursor-not-allowed
              `}
            >
              <span>{mod.icon}</span>
              <span className="hidden sm:inline">{mod.nameZh}</span>
              <span className="sm:hidden">{mod.name}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* ===== Main: 2-Column ===== */}
      <main className="flex-1 flex min-h-0">
        {/* ── Left: Input ── */}
        <div className="w-[400px] border-r border-white/[0.06] flex flex-col p-5 gap-4 shrink-0">
          <div>
            <h2 className="text-sm font-semibold text-gray-300 mb-1">
              📥 輸入: {currentModule.nameZh}
            </h2>
            <p className="text-xs text-gray-600">
              輸入業務申請數據，點擊提交開始多智能體審查
            </p>
          </div>

          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={stream.isStreaming}
            className="flex-1 bg-white/[0.03] border border-white/[0.08] rounded-xl p-4 text-sm text-gray-300 font-mono resize-none outline-none focus:border-blue-500/40 transition-colors disabled:opacity-50"
            placeholder="輸入業務數據..."
          />

          <button
            onClick={handleSubmit}
            disabled={stream.isStreaming || !inputText.trim()}
            className="w-full py-3 rounded-xl font-semibold text-sm text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: stream.isStreaming
                ? "#333"
                : "linear-gradient(135deg, #667eea, #764ba2)",
            }}
          >
            {stream.isStreaming
              ? `⏳ 智能體運行中... ${formatTime(stream.elapsedTime)}`
              : "🚀 提交審查請求"}
          </button>

          {/* RAG Pipeline Info */}
          <div className="text-[10px] text-gray-700 space-y-0.5 border-t border-white/[0.04] pt-3">
            <p>🔍 BM25 + Dense Hybrid Retrieval (RRF)</p>
            <p>🎯 Cohere Reranker v3.5 → Top-5</p>
            <p>🛡️ Anti-Hallucination Prompt Chain</p>
          </div>
        </div>

        {/* ── Right: Output ── */}
        <div className="flex-1 flex flex-col min-h-0 p-5 gap-2">
          <h2 className="text-sm font-semibold text-gray-300 shrink-0">
            🛡️ 輸出: 合規審查報告
          </h2>

          {/* Error */}
          {stream.error && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-xs px-4 py-2.5 rounded-lg shrink-0">
              ❌ {stream.error}
            </div>
          )}

          {/* Agent Pipeline Timeline */}
          <div className="shrink-0 max-h-[320px] overflow-y-auto">
            <AgentTimeline
              agents={stream.agentStates}
              currentAgent={stream.currentAgent}
              isStreaming={stream.isStreaming}
              elapsedTime={stream.elapsedTime}
              phase={stream.phase}
            />
          </div>

          {/* Report Output */}
          <div className="flex-1 min-h-0 bg-white/[0.02] border border-white/[0.06] rounded-xl flex flex-col overflow-hidden">
            <ReportPanel
              text={stream.reportText}
              isStreaming={stream.isStreaming}
              phase={stream.phase}
              elapsedTime={stream.elapsedTime}
            />

            {/* Footer metrics */}
            {stream.phase === "done" && stream.reportText && (
              <div className="border-t border-white/[0.06] px-4 py-2.5 text-[11px] text-gray-500 flex gap-4 shrink-0">
                <span>⏱️ {formatTime(stream.elapsedTime)}</span>
                <span>📝 {stream.reportText.length.toLocaleString()} chars</span>
                <span>🤖 {stream.agentStates.length} agent steps</span>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
