import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useAgentStream } from "@/hooks/useAgentStream";
import { bankWorkflows } from "@/lib/bankWorkspaces";


describe("useAgentStream cancellation", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("marks an aborted request as cancelled and returns to an actionable state", async () => {
    const fetchMock = vi.fn((_url: string, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("The operation was aborted", "AbortError"));
        });
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const workflowConfig = bankWorkflows.find(
      (workflow) => workflow.id === "account-kyc-review",
    )!;
    const { result } = renderHook(() => useAgentStream());

    let request: Promise<void>;
    act(() => {
      request = result.current.startStream(workflowConfig, "cancel test");
    });
    await waitFor(() => expect(result.current.isStreaming).toBe(true));

    act(() => result.current.cancelStream());
    await act(async () => request!);

    expect(result.current.isStreaming).toBe(false);
    expect(result.current.phase).toBe("idle");
    expect(result.current.error).toBe("Analysis cancelled.");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
