import type { BankWorkflowConfig } from "@/types";

type Primitive = string | number | boolean | null | undefined;

function boardLabel(boardId: BankWorkflowConfig["boardId"]): string {
  switch (boardId) {
    case "customer-account":
      return "客戶與賬戶合規 / Customer & Account Compliance";
    case "transaction-payment":
      return "交易與支付合規 / Transaction & Payment Compliance";
    case "product-launch":
      return "產品與業務上線審查 / Product & Business Launch Review";
    case "regulatory-research":
      return "監管研究與政策變化 / Regulatory Research & Policy Change";
    case "human-review":
      return "人工覆核與審計 / Human Review & Audit";
    case "knowledge-base":
      return "監管知識庫 / Regulatory Knowledge Base";
    default:
      return "總覽 / Dashboard";
  }
}

function baseHeader(module: BankWorkflowConfig): string {
  return `# ${module.nameZh} / ${module.name}

## 模組資訊 / Module Context
- 看板 / Workspace: ${boardLabel(module.boardId)}
- 工作流 / Workflow ID: ${module.id}
- 推理模式 / Engine Mode: ${module.engineMode}

## 輸出結果 / Output`;
}

function toDisplayValue(value: Primitive): string {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "boolean") return value ? "Yes / 是" : "No / 否";
  return String(value);
}

function isObligationPayload(data: unknown): data is {
  applicable_regulators?: string[];
  risks?: string[];
  obligations?: Array<{
    obligation?: string;
    regulator?: string;
    risk?: string;
    controls?: string[];
    evidence_ids?: string[];
  }>;
} {
  if (!data || typeof data !== "object") return false;
  const candidate = data as Record<string, unknown>;
  return Array.isArray(candidate.applicable_regulators) && Array.isArray(candidate.obligations);
}

function isGraphPayload(data: unknown): data is {
  paths?: Array<{ path?: string[]; matched_node?: string; matched_topics?: string[] }>;
  graph_paths?: Array<{ path?: string[]; matched_node?: string; matched_topics?: string[] }>;
} {
  if (!data || typeof data !== "object") return false;
  const candidate = data as Record<string, unknown>;
  return Array.isArray(candidate.paths) || Array.isArray(candidate.graph_paths);
}

export function formatTextReport(module: BankWorkflowConfig, text: string): string {
  const cleanText = (text || "").trim();
  const body = cleanText.length > 0 ? cleanText : "No output generated. / 暫無輸出。";
  return `${baseHeader(module)}\n\n${body}`;
}

export function formatHumanReviewQueueReport(
  module: BankWorkflowConfig,
  queueItems: Array<{
    workflow_run_id?: string;
    gate_type?: string;
    human_review_status?: string;
  }>
): string {
  if (!queueItems.length) {
    return `${baseHeader(module)}

目前沒有待人工覆核個案。  
There are no pending human review cases.`;
  }

  const rows = queueItems
    .map((item, idx) => {
      const id = item.workflow_run_id ?? "unknown";
      const gate = item.gate_type ?? "unknown";
      const status = item.human_review_status ?? "pending";
      return `### ${idx + 1}. 個案 / Case ${idx + 1}
- 工作流編號 / Workflow Run ID: ${id}
- 觸發原因 / Gate Type: ${gate}
- 覆核狀態 / Review Status: ${status}`;
    })
    .join("\n\n");

  return `${baseHeader(module)}\n\n${rows}`;
}

function formatObligationMapReport(
  module: BankWorkflowConfig,
  data: {
    applicable_regulators?: string[];
    risks?: string[];
    obligations?: Array<{
      obligation?: string;
      regulator?: string;
      risk?: string;
      controls?: string[];
      evidence_ids?: string[];
    }>;
  }
): string {
  const regulators = (data.applicable_regulators || []).join(", ") || "N/A";
  const risks = (data.risks || []).join(", ") || "N/A";
  const obligations = (data.obligations || [])
    .map((item, idx) => {
      const controls = Array.isArray(item.controls) && item.controls.length ? item.controls.join(", ") : "N/A";
      const evidenceIds =
        Array.isArray(item.evidence_ids) && item.evidence_ids.length ? item.evidence_ids.join(", ") : "N/A";
      return `### ${idx + 1}. ${item.obligation || "Unknown Obligation"}
- 監管機構 / Regulator: ${item.regulator || "N/A"}
- 風險類型 / Risk Type: ${item.risk || "N/A"}
- 控制措施 / Controls: ${controls}
- 證據編號 / Evidence IDs: ${evidenceIds}`;
    })
    .join("\n\n");

  return `${baseHeader(module)}

## 適用監管機構 / Applicable Regulators
${regulators}

## 主要風險 / Key Risks
${risks}

## 監管義務映射 / Obligation Mapping
${obligations || "N/A"}`;
}

function formatGraphReport(
  module: BankWorkflowConfig,
  data: {
    paths?: Array<{ path?: string[]; matched_node?: string; matched_topics?: string[] }>;
    graph_paths?: Array<{ path?: string[]; matched_node?: string; matched_topics?: string[] }>;
  }
): string {
  const paths = (data.paths || data.graph_paths || []).map((item, idx) => {
    const route = Array.isArray(item.path) ? item.path.join(" -> ") : "N/A";
    const node = item.matched_node || "N/A";
    const topics = Array.isArray(item.matched_topics) ? item.matched_topics.join(", ") : "N/A";
    return `### ${idx + 1}. 路徑 / Path ${idx + 1}
- 圖譜路徑 / Graph Path: ${route}
- 命中節點 / Matched Node: ${node}
- 命中主題 / Matched Topics: ${topics}`;
  });

  return `${baseHeader(module)}

## 圖譜查詢結果 / Graph Query Result
${paths.length ? paths.join("\n\n") : "N/A"}`;
}

function formatGenericObject(module: BankWorkflowConfig, data: Record<string, unknown>): string {
  const sections = Object.entries(data).map(([key, value]) => {
    if (Array.isArray(value)) {
      if (value.every((v) => typeof v !== "object" || v === null)) {
        return `### ${key}\n- 值 / Value: ${value.map((v) => toDisplayValue(v as Primitive)).join(", ") || "N/A"}`;
      }
      const items = value
        .map((item, idx) => {
          if (!item || typeof item !== "object") {
            return `${idx + 1}. ${toDisplayValue(item as Primitive)}`;
          }
          const fields = Object.entries(item as Record<string, unknown>)
            .map(([k, v]) => `- ${k}: ${Array.isArray(v) ? v.join(", ") : toDisplayValue(v as Primitive)}`)
            .join("\n");
          return `#### ${idx + 1}\n${fields}`;
        })
        .join("\n\n");
      return `### ${key}\n${items || "N/A"}`;
    }

    if (value && typeof value === "object") {
      const fields = Object.entries(value as Record<string, unknown>)
        .map(([k, v]) => `- ${k}: ${Array.isArray(v) ? v.join(", ") : toDisplayValue(v as Primitive)}`)
        .join("\n");
      return `### ${key}\n${fields || "N/A"}`;
    }

    return `### ${key}\n- 值 / Value: ${toDisplayValue(value as Primitive)}`;
  });

  return `${baseHeader(module)}

## 結構化輸出 / Structured Output
${sections.join("\n\n")}`;
}

export function formatJsonReport(module: BankWorkflowConfig, data: unknown): string {
  if (isObligationPayload(data)) {
    return formatObligationMapReport(module, data);
  }

  if (isGraphPayload(data)) {
    return formatGraphReport(module, data);
  }

  if (Array.isArray(data)) {
    const list = data
      .map((item, idx) => `${idx + 1}. ${typeof item === "object" ? JSON.stringify(item) : String(item)}`)
      .join("\n");
    return `${baseHeader(module)}

## 清單輸出 / List Output
${list || "N/A"}`;
  }

  if (data && typeof data === "object") {
    return formatGenericObject(module, data as Record<string, unknown>);
  }

  return `${baseHeader(module)}

## 基礎輸出 / Basic Output
${toDisplayValue(data as Primitive)}`;
}
