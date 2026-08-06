import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import EvidencePanel from "@/components/EvidencePanel";
import type { EvidenceChunk } from "@/types";

const chunks: EvidenceChunk[] = [
  {
    evidence_id: "ev-001",
    title: "HKMA SPM Module 1",
    regulator: "HKMA",
    retrieval_method: "hybrid",
    score: 0.85,
    page: 12,
    section_title: "Capital Adequacy",
    jurisdiction: "HK",
    text: "Banks must maintain minimum capital ratios at all times.",
    metadata: { rerank_score: 0.9 },
  },
  {
    evidence_id: "ev-002",
    title: "SFC Code of Conduct",
    regulator: "SFC",
    score: 0.42,
    text: "Intermediaries must act with due skill, care and diligence.",
  },
];

describe("EvidencePanel", () => {
  it("加载状态：显示骨架屏而不是空状态或证据列表", () => {
    render(<EvidencePanel evidence={[]} isLoading />);
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.queryByText("No evidence retrieved yet")).not.toBeInTheDocument();
    // 骨架屏有 animate-pulse 占位块
    const skeleton = document.querySelector(".animate-pulse");
    expect(skeleton).not.toBeNull();
  });

  it("空状态：无证据时提示 'No evidence retrieved yet'", () => {
    render(<EvidencePanel evidence={[]} isLoading={false} />);
    expect(screen.getByText("No evidence retrieved yet")).toBeInTheDocument();
    expect(screen.getByText(/Evidence chunks will appear here after retrieval/)).toBeInTheDocument();
  });

  it("渲染证据卡片：标题、监管机构、分数与数量统计", () => {
    render(<EvidencePanel evidence={chunks} isLoading={false} />);
    expect(screen.getByText("HKMA SPM Module 1")).toBeInTheDocument();
    expect(screen.getByText("SFC Code of Conduct")).toBeInTheDocument();
    expect(screen.getByText("HKMA")).toBeInTheDocument();
    expect(screen.getByText("85%")).toBeInTheDocument();
    expect(screen.getByText("2 chunks")).toBeInTheDocument();
  });

  it("交互：点击卡片头部展开证据正文，再次点击收起", () => {
    render(<EvidencePanel evidence={chunks} isLoading={false} />);
    const bodyText = screen.getByText(/Banks must maintain minimum capital ratios/);
    const expandable = bodyText.closest(".overflow-hidden");
    expect(expandable).toHaveClass("max-h-0");

    fireEvent.click(screen.getByRole("button", { name: /HKMA SPM Module 1/ }));
    expect(expandable).toHaveClass("max-h-[500px]");

    fireEvent.click(screen.getByRole("button", { name: /HKMA SPM Module 1/ }));
    expect(expandable).toHaveClass("max-h-0");
  });
});
