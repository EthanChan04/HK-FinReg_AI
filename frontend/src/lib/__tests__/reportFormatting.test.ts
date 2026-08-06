import { describe, expect, it } from "vitest";

import {
  formatHumanReviewQueueReport,
  formatJsonReport,
  formatTextReport,
} from "@/lib/reportFormatting";
import type { BankWorkflowConfig } from "@/types";

const workflowModule: BankWorkflowConfig = {
  id: "onboarding-review",
  name: "Onboarding Review",
  nameZh: "開戶審查",
  boardId: "customer-account",
  engineMode: "rag",
  endpoint: "/api/compliance",
  icon: "🏦",
  defaultInput: "",
  status: "production",
  description: "",
  primaryUsers: [],
  scenarioType: "customer_review",
};

describe("reportFormatting", () => {
  it("formatTextReport：生成含模块头部与正文的报告", () => {
    const report = formatTextReport(workflowModule, "  AML check passed.  ");
    expect(report).toContain("# 開戶審查 / Onboarding Review");
    expect(report).toContain("AML check passed.");
    expect(report).not.toContain("No output generated");
  });

  it("formatTextReport：空文本时回退为 'No output generated'", () => {
    const report = formatTextReport(workflowModule, "   ");
    expect(report).toContain("No output generated. / 暫無輸出。");
  });

  it("formatJsonReport：义务映射载荷渲染监管机构与义务条目", () => {
    const report = formatJsonReport(workflowModule, {
      applicable_regulators: ["HKMA"],
      risks: ["AML"],
      obligations: [
        { obligation: "Conduct CDD", regulator: "HKMA", controls: ["ID check"] },
      ],
    });
    expect(report).toContain("## 適用監管機構 / Applicable Regulators");
    expect(report).toContain("HKMA");
    expect(report).toContain("Conduct CDD");
    expect(report).toContain("ID check");
  });

  it("formatHumanReviewQueueReport：空队列给出无待审提示", () => {
    const report = formatHumanReviewQueueReport(workflowModule, []);
    expect(report).toContain("There are no pending human review cases.");
  });
});
