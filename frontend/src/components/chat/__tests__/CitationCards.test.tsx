import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import CitationCards from "@/components/chat/CitationCards";
import type { EvidenceChunk } from "@/types";

function makeChunk(id: string, title?: string): EvidenceChunk {
  return {
    evidence_id: id,
    title,
    regulator: "HKMA",
    page: 3,
    text: `Body of ${id}`,
  };
}

describe("CitationCards", () => {
  it("无证据时渲染 null（不产生任何内容）", () => {
    const { container } = render(<CitationCards evidence={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("渲染证据卡片：标题、监管机构、页码与正文", () => {
    render(<CitationCards evidence={[makeChunk("c-1", "Guideline 2.1")]} />);
    expect(screen.getByText("Evidence Cards")).toBeInTheDocument();
    expect(screen.getByText(/c-1 · Guideline 2\.1/)).toBeInTheDocument();
    expect(screen.getByText(/HKMA/)).toBeInTheDocument();
    expect(screen.getByText(/p\.3/)).toBeInTheDocument();
    expect(screen.getByText(/Body of c-1/)).toBeInTheDocument();
  });

  it("超过 4 条证据时最多只渲染前 4 条", () => {
    const evidence = Array.from({ length: 6 }, (_, i) => makeChunk(`c-${i}`));
    render(<CitationCards evidence={evidence} />);
    expect(screen.getAllByText(/Body of c-/)).toHaveLength(4);
    expect(screen.queryByText(/Body of c-4/)).not.toBeInTheDocument();
  });
});
