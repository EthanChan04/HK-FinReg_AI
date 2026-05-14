import type { BankBoardId } from "@/types";

const promptMap: Record<BankBoardId, string[]> = {
  "customer-account": [
    "解釋此客戶的主要 KYC 風險 / Explain this customer's key KYC risks",
    "哪些資料不足？/ What information is missing?",
    "生成 RM 跟進清單 / Draft an RM follow-up list",
  ],
  "transaction-payment": [
    "這筆交易為何可疑？/ Why is this transaction suspicious?",
    "是否需要升級到 EDD？/ Should this be escalated to EDD?",
    "生成 STR/SAR 重點草稿 / Draft key STR/SAR points",
  ],
  "product-launch": [
    "是否需要 Deep Research？/ Should this use Deep Research?",
    "涉及哪些監管機構？/ Which regulators are involved?",
    "生成上線前 checklist / Generate a pre-launch checklist",
  ],
  "regulatory-research": [
    "生成管理層摘要 / Generate a management summary",
    "比較 HKMA、SFC、PCPD 要求 / Compare HKMA, SFC, and PCPD expectations",
    "指出證據缺口 / Identify evidence gaps",
  ],
  "human-review": [
    "為何進入人工覆核？/ Why was this sent to human review?",
    "生成 reviewer notes / Draft reviewer notes",
    "哪些結論置信度較低？/ Which conclusions have lower confidence?",
  ],
  "knowledge-base": [
    "解釋義務映射 / Explain the obligation mapping",
    "哪些監管文件支持此結論？/ Which regulatory documents support this?",
    "顯示風險控制路徑 / Show risk-control paths",
  ],
  dashboard: [
    "What should I review first today?",
    "Show recent high-risk compliance signals.",
    "Summarize pending human review cases.",
  ],
};

export function getCopilotPrompts(boardId: BankBoardId): string[] {
  return promptMap[boardId] || promptMap["customer-account"];
}
