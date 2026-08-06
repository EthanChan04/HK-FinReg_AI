import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import WorkflowSelector from "@/components/WorkflowSelector";
import type { BankWorkflowConfig } from "@/types";

const workflows: BankWorkflowConfig[] = [
  {
    id: "customer-onboarding",
    name: "Customer Onboarding Review",
    nameZh: "客戶開戶合規審查",
    boardId: "customer-account",
    engineMode: "rag",
    endpoint: "/api/compliance",
    icon: "🏦",
    defaultInput: "",
    status: "production",
    description: "開戶資料與 AML 審查",
    primaryUsers: ["compliance"],
    scenarioType: "customer_review",
  },
  {
    id: "deep-research",
    name: "Regulatory Deep Research",
    nameZh: "監管深度研究",
    boardId: "regulatory-research",
    engineMode: "deepresearch",
    endpoint: "/api/research",
    icon: "🔬",
    defaultInput: "",
    status: "production",
    description: "多智能體深度研究",
    primaryUsers: ["compliance"],
    scenarioType: "regulatory_research",
  },
];

describe("WorkflowSelector", () => {
  it("渲染全部工作流：中文名 + 引擎模式标签", () => {
    render(
      <WorkflowSelector
        workflows={workflows}
        activeWorkflowId="customer-onboarding"
        onChange={() => {}}
      />
    );
    expect(screen.getByText("客戶開戶合規審查")).toBeInTheDocument();
    expect(screen.getByText("監管深度研究")).toBeInTheDocument();
    expect(screen.getByText("RAG")).toBeInTheDocument();
    expect(screen.getByText("Deep Research")).toBeInTheDocument();
  });

  it("点击工作流触发 onChange 并传入对应配置", () => {
    const onChange = vi.fn();
    render(
      <WorkflowSelector
        workflows={workflows}
        activeWorkflowId="customer-onboarding"
        onChange={onChange}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /監管深度研究/ }));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith(workflows[1]);
  });

  it("disabled 状态下按钮不可点击，onChange 不被触发", () => {
    const onChange = vi.fn();
    render(
      <WorkflowSelector
        workflows={workflows}
        activeWorkflowId="customer-onboarding"
        disabled
        onChange={onChange}
      />
    );
    const buttons = screen.getAllByRole("button");
    expect(buttons[0]).toBeDisabled();
    fireEvent.click(buttons[0]);
    expect(onChange).not.toHaveBeenCalled();
  });
});
