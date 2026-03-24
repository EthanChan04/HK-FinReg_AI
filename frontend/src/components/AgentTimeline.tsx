// Agent 思考链路指示器
// 横向管道式进度条 + 展开详情，解决 200s 等待焦虑
"use client";

import type { AgentStateEvent } from "@/types";

const agentIcons: Record<string, string> = {
  "Extractor Agent": "🔍",
  "Retriever Agent": "📚",
  "Analyzer Agent": "🧠",
  "Reviewer Agent": "⚖️",
  "KYC Analyst": "🪪",
  "CDD Specialist": "🔬",
  "Approval Officer": "✍️",
  "Chief Risk Officer": "👨‍⚖️",
  "Extraction Specialist": "✂️",
  "Sanctions Screener": "🛡️",
  "AML Investigator": "🕵️",
  "Compliance Director": "📋",
  "Data Processor": "🧮",
  "Financial Analyst": "📈",
  "Credit Officer": "✍️",
  "Credit Committee": "👨‍⚖️",
};

interface Props {
  agents: AgentStateEvent[];
  currentAgent: string | null;
  isStreaming: boolean;
  elapsedTime: number;
  phase: "idle" | "agents" | "streaming" | "done";
}

export default function AgentTimeline({
  agents,
  currentAgent,
  isStreaming,
  elapsedTime,
  phase,
}: Props) {
  if (agents.length === 0 && phase === "idle") return null;

  // 从 agents 列表中提取唯一的 agent 名称（按出现序）
  const uniqueAgents: { name: string; message: string }[] = [];
  const seen = new Set<string>();
  for (const a of agents) {
    if (!seen.has(a.agent)) {
      seen.add(a.agent);
      uniqueAgents.push({ name: a.agent, message: a.message });
    }
  }

  return (
    <div className="mb-4">
      {/* 标题 */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="h-px w-8 bg-gradient-to-r from-blue-500/60 to-transparent" />
          <span className="text-xs font-medium text-blue-400 tracking-widest uppercase">
            Agent Pipeline
          </span>
          <div className="h-px w-8 bg-gradient-to-l from-blue-500/60 to-transparent" />
        </div>
        {isStreaming && (
          <span className="text-[11px] text-gray-500 font-mono tabular-nums">
            {elapsedTime}s
          </span>
        )}
      </div>

      {/* 横向管道进度指示器 */}
      <div className="flex items-center gap-1 mb-4 overflow-x-auto pb-1">
        {uniqueAgents.map((agent, idx) => {
          const isActive = agent.name === currentAgent && isStreaming;
          const isDone = !isActive && (phase === "streaming" || phase === "done" || idx < uniqueAgents.length - 1);
          const icon = agentIcons[agent.name] || "🤖";

          return (
            <div key={agent.name} className="flex items-center shrink-0">
              {/* Node */}
              <div
                className={`
                  flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-300
                  ${isActive
                    ? "bg-blue-500/20 text-blue-300 border border-blue-500/40 shadow-[0_0_12px_rgba(59,130,246,0.15)]"
                    : isDone
                      ? "bg-emerald-500/10 text-emerald-400/80 border border-emerald-500/20"
                      : "bg-white/[0.03] text-gray-600 border border-white/[0.06]"
                  }
                `}
              >
                <span>{icon}</span>
                <span className="hidden sm:inline whitespace-nowrap">{agent.name.replace(" Agent", "")}</span>
                {isActive && (
                  <span className="relative flex h-1.5 w-1.5 ml-0.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-blue-400" />
                  </span>
                )}
                {isDone && <span className="text-[9px]">✓</span>}
              </div>

              {/* Connector arrow */}
              {idx < uniqueAgents.length - 1 && (
                <div className={`mx-1 text-[10px] ${isDone ? "text-emerald-500/40" : "text-gray-700"}`}>
                  →
                </div>
              )}
            </div>
          );
        })}

        {/* "正在执行" 的脉冲点（当还在 agents 阶段且管道尾部） */}
        {phase === "agents" && isStreaming && (
          <div className="flex items-center gap-1 ml-1 text-[10px] text-blue-400/60">
            <span>→</span>
            <span className="animate-pulse">···</span>
          </div>
        )}
      </div>

      {/* 展开的详情列表 */}
      <div className="space-y-1.5">
        {uniqueAgents.map((agent, idx) => {
          const isActive = agent.name === currentAgent && isStreaming;
          const icon = agentIcons[agent.name] || "🤖";

          return (
            <div
              key={`detail-${idx}`}
              className={`
                flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-300
                ${isActive
                  ? "bg-blue-950/40 border border-blue-500/30"
                  : "bg-transparent border border-transparent"
                }
              `}
            >
              <span className="text-base shrink-0">{icon}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-semibold ${isActive ? "text-blue-300" : "text-gray-500"}`}>
                    {agent.name}
                  </span>
                </div>
                <p className={`text-[11px] mt-0.5 truncate ${isActive ? "text-blue-200/60" : "text-gray-600"}`}>
                  {agent.message}
                </p>
              </div>
              {isActive ? (
                <div className="shrink-0 flex gap-0.5">
                  <span className="w-1 h-1 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-1 h-1 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-1 h-1 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              ) : (
                <span className="text-[9px] text-emerald-500/80 font-medium bg-emerald-500/10 px-1.5 py-0.5 rounded shrink-0">
                  DONE
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
