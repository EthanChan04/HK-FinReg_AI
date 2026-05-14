import type { BankWorkflowConfig } from "@/types";

export type WorkflowPayload =
  | {
      application_data: string;
      business_context?: string;
      stream_agents_state: boolean;
    }
  | {
      query: string;
      task_type?: BankWorkflowConfig["taskType"];
      output_format?: BankWorkflowConfig["outputFormat"];
      max_iterations?: number;
    };

export function buildWorkflowPayload(
  workflow: BankWorkflowConfig,
  inputText: string
): WorkflowPayload {
  if (workflow.engineMode === "human_review") {
    return {
      application_data: inputText,
      business_context: "human_review_queue",
      stream_agents_state: false,
    };
  }

  if (workflow.requestKind === "research" || workflow.engineMode === "deepresearch") {
    return {
      query: inputText,
      task_type: workflow.taskType ?? "routine_review",
      output_format: workflow.outputFormat ?? "report",
      max_iterations: 3,
    };
  }

  if (workflow.requestKind === "kag" || workflow.engineMode === "rag_kag") {
    return {
      query: inputText,
    };
  }

  return {
    application_data: inputText,
    stream_agents_state: true,
  };
}
