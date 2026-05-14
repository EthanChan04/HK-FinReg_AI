"use client";

import { useMemo, useState } from "react";

import AgentTimeline from "@/components/AgentTimeline";
import ComplianceCopilot from "@/components/chat/ComplianceCopilot";
import DeepResearchPlanPanel from "@/components/DeepResearchPlanPanel";
import EvidencePanel from "@/components/EvidencePanel";
import KnowledgeGraphPanel from "@/components/KnowledgeGraphPanel";
import ReportPanel from "@/components/ReportPanel";
import WorkflowSelector from "@/components/WorkflowSelector";
import WorkspaceNav from "@/components/WorkspaceNav";
import { useAgentStream } from "@/hooks/useAgentStream";
import { bankBoards, bankWorkflows, defaultWorkflow } from "@/lib/bankWorkspaces";
import type { BankBoardId, BankWorkflowConfig, CopilotCaseContext } from "@/types";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default function Home() {
  const [activeBoardId, setActiveBoardId] = useState<BankBoardId>(defaultWorkflow.boardId);
  const [currentModule, setCurrentModule] = useState<BankWorkflowConfig>(defaultWorkflow);
  const [inputText, setInputText] = useState(defaultWorkflow.defaultInput);
  const stream = useAgentStream();

  const workflowsForBoard = useMemo(
    () => bankWorkflows.filter((workflow) => workflow.boardId === activeBoardId),
    [activeBoardId]
  );
  const copilotCaseContext = useMemo<CopilotCaseContext>(
    () => ({
      workspace_id: activeBoardId,
      workflow_id: currentModule.id,
      workflow_name: currentModule.name,
      input_text: inputText,
      report_text: stream.reportText,
      evidence_chunks: stream.evidenceChunks,
      graph_paths: stream.graphPaths,
      research_plan: stream.researchPlan,
      confidence_data: {
        retrieval: stream.confidenceScore,
        reasoning: stream.reasoningConfidence,
        reviewer: stream.reviewerConfidence,
        cross_validation_passed: stream.crossValidationPassed,
      },
      workflow_run_id: stream.workflowRunId,
      current_gate: stream.currentGate,
      gate_message: stream.gateMessage,
    }),
    [activeBoardId, currentModule, inputText, stream]
  );

  const handleBoardSwitch = (boardId: BankBoardId) => {
    if (stream.isStreaming) return;
    const firstWorkflow = bankWorkflows.find((workflow) => workflow.boardId === boardId);
    if (!firstWorkflow) return;
    setActiveBoardId(boardId);
    setCurrentModule(firstWorkflow);
    setInputText(firstWorkflow.defaultInput);
    stream.reset();
  };

  const handleWorkflowSwitch = (workflow: BankWorkflowConfig) => {
    if (stream.isStreaming) return;
    setCurrentModule(workflow);
    setInputText(workflow.defaultInput);
    stream.reset();
  };

  const handleSubmit = () => {
    if (!inputText.trim() || stream.isStreaming) return;
    stream.startStream(currentModule, inputText);
  };

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-slate-300/10 bg-slate-950/40 px-4 py-5 backdrop-blur-sm md:px-6">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between">
          <div>
            <h1 className="text-gradient text-3xl font-extrabold tracking-tight md:text-4xl">HK-FinReg AI</h1>
            <p className="mt-1 text-sm tracking-wide text-slate-400">
              Regulatory Intelligence & Compliance Operations Platform
            </p>
          </div>

          {stream.isStreaming && (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2.5 rounded-full border border-cyan-300/30 bg-cyan-500/12 px-4 py-2 text-xs">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-300" />
                </span>
                <span className="font-mono tabular-nums text-cyan-100">{formatTime(stream.elapsedTime)}</span>
                <span className="text-cyan-100/70">
                  {stream.phase === "agents" ? "Agent Processing" : "Streaming Report"}
                </span>
              </div>
              <button
                onClick={stream.cancelStream}
                className="rounded-full border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200 transition-all hover:border-rose-300/70"
              >
                Cancel
              </button>
            </div>
          )}

          {stream.phase === "action_required" && !stream.isStreaming && (
            <div className="flex items-center gap-2 rounded-full border border-amber-300/30 bg-amber-500/10 px-4 py-2 text-xs text-amber-200">
              <span>!</span>
              <span>
                Waiting for Human Review /{" "}
                {stream.currentGate === "low_confidence_gate"
                  ? "Low Confidence"
                  : stream.currentGate === "missing_evidence_gate"
                    ? "Missing Evidence"
                    : "Manual Approval"}
              </span>
            </div>
          )}

          {stream.phase === "done" && !stream.isStreaming && (
            <div className="flex items-center gap-2 rounded-full border border-emerald-300/30 bg-emerald-500/10 px-4 py-2 text-xs text-emerald-200">
              <span>Done</span>
              <span>Completed in {formatTime(stream.elapsedTime)}</span>
            </div>
          )}
        </div>
      </header>

      <WorkspaceNav
        boards={bankBoards}
        activeBoardId={activeBoardId}
        disabled={stream.isStreaming}
        onChange={handleBoardSwitch}
      />

      <main className="flex min-h-0 flex-1 overflow-hidden">
        <div className="flex w-[430px] shrink-0 flex-col gap-4 border-r border-slate-300/10 bg-slate-950/35 p-5">
          <div className="rounded-2xl border border-slate-300/15 bg-slate-900/45 p-3">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">Workspace Workflows</h2>
            <WorkflowSelector
              workflows={workflowsForBoard}
              activeWorkflowId={currentModule.id}
              disabled={stream.isStreaming}
              onChange={handleWorkflowSwitch}
            />
          </div>

          <div className="rounded-2xl border border-slate-300/15 bg-slate-900/45 p-4">
            <h2 className="mb-1 text-sm font-semibold text-slate-100">Input: {currentModule.nameZh}</h2>
            <p className="text-xs text-slate-400">
              Provide scenario details, then submit to start compliance analysis.
            </p>

            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={stream.isStreaming}
              className="mt-3 h-[230px] w-full resize-none rounded-xl border border-slate-300/20 bg-slate-900/70 p-4 font-mono text-sm text-slate-200 outline-none transition-colors focus:border-cyan-300/40 focus:ring-2 focus:ring-cyan-300/10 disabled:opacity-50"
              placeholder="Enter compliance scenario..."
            />

            <button
              onClick={handleSubmit}
              disabled={stream.isStreaming || !inputText.trim()}
              className="mt-3 w-full rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 py-3 text-sm font-semibold text-white shadow-[0_14px_28px_rgba(25,131,171,0.32)] transition-all disabled:cursor-not-allowed disabled:opacity-40"
            >
              {stream.isStreaming ? `Running... ${formatTime(stream.elapsedTime)}` : "Submit Analysis"}
            </button>
          </div>

          <div className="space-y-1 rounded-2xl border border-slate-300/10 bg-slate-900/35 p-3 text-[10px] text-slate-500">
            <p>Dynamic routing: RAG / KAG / Deep Research</p>
            <p>Hybrid retrieval with citation-ready evidence</p>
            <p>Human review queue for low-confidence cases</p>
          </div>
        </div>

        <div className="flex min-w-0 flex-1 overflow-hidden">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2 p-5">
            <h2 className="shrink-0 text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">
              Output: Compliance Analysis Report
            </h2>

            {stream.error && (
              <div className="shrink-0 rounded-lg border border-rose-400/30 bg-rose-500/10 px-4 py-2.5 text-xs text-rose-200">
                {stream.error}
              </div>
            )}

            <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
              <div className="max-h-[320px] shrink-0 overflow-y-auto">
                <AgentTimeline
                  agents={stream.agentStates}
                  currentAgent={stream.currentAgent}
                  isStreaming={stream.isStreaming}
                  elapsedTime={stream.elapsedTime}
                  phase={stream.phase}
                />
              </div>

              <div className="flex min-h-[250px] flex-col overflow-hidden rounded-2xl border border-slate-300/12 bg-slate-950/35">
                <ReportPanel
                  text={stream.reportText}
                  isStreaming={stream.isStreaming}
                  phase={stream.phase}
                  elapsedTime={stream.elapsedTime}
                  confidenceScore={stream.confidenceScore}
                  confidenceWarning={stream.confidenceWarning}
                  reasoningConfidence={stream.reasoningConfidence}
                  reviewerConfidence={stream.reviewerConfidence}
                  crossValidationPassed={stream.crossValidationPassed}
                  workflowRunId={stream.workflowRunId}
                  humanReviewRequired={stream.humanReviewRequired}
                  currentGate={stream.currentGate}
                  gateMessage={stream.gateMessage}
                  onResumeResult={stream.setResumedResult}
                />

                {(stream.phase === "done" || stream.phase === "action_required") && stream.reportText && (
                  <div className="flex shrink-0 gap-4 border-t border-slate-300/12 px-4 py-2.5 text-[11px] text-slate-400">
                    <span>Time: {formatTime(stream.elapsedTime)}</span>
                    <span>Length: {stream.reportText.length.toLocaleString()} chars</span>
                    <span>Steps: {stream.agentStates.length}</span>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                <EvidencePanel
                  evidence={stream.evidenceChunks}
                  isLoading={stream.isStreaming && stream.phase === "agents"}
                />
                <KnowledgeGraphPanel
                  paths={stream.graphPaths}
                  isLoading={stream.isStreaming && stream.phase === "agents"}
                />
              </div>

              {stream.researchPlan && (
                <DeepResearchPlanPanel
                  researchPlan={stream.researchPlan}
                  evidenceBySubquestion={stream.evidenceBySubquestion}
                  evidenceGaps={stream.evidenceGaps}
                  isLoading={stream.isStreaming && stream.phase === "agents"}
                />
              )}
            </div>
          </div>
          <ComplianceCopilot activeBoardId={activeBoardId} caseContext={copilotCaseContext} />
        </div>
      </main>
    </div>
  );
}
