import { expect, test } from "@playwright/test";

function sse(events: Array<[string, unknown]>): string {
  return events
    .map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
    .join("");
}

test("streams a compliance report and exposes its supporting evidence", async ({ page }) => {
  await page.route("**/api/backend/api/v1/bank-account/verify/stream", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sse([
        ["agent_state", { agent: "retriever", status: "running", message: "Searching SFC sources" }],
        ["evidence_chunks", [{
          evidence_id: "SFC-CDD-001",
          doc_id: "sfc-cdd-guideline",
          title: "SFC Customer Due Diligence Guideline",
          regulator: "SFC",
          jurisdiction: "Hong Kong",
          page: 12,
          score: 0.93,
          text: "Verify the customer's identity using reliable and independent source documents.",
        }]],
        ["token", { text: "## Assessment\nEnhanced customer due diligence is required.\n" }],
        ["confidence", { dimension: "full", retrieval: 0.91, reasoning: 0.86, reviewer: 0.84, cross_validation_passed: true }],
        ["done", { workflow_run_id: "workflow-e2e-001" }],
      ]),
    });
  });

  await page.goto("/");
  await page.getByPlaceholder("Enter compliance scenario...").fill("Assess this high-risk onboarding case.");
  await page.getByRole("button", { name: "Submit Analysis" }).click();

  await expect(page.getByText("Completed in")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Assessment" })).toBeVisible();
  await expect(page.getByText("Enhanced customer due diligence is required.")).toBeVisible();
  await expect(page.getByText("SFC Customer Due Diligence Guideline")).toBeVisible();
  await expect(page.getByText("1 chunk")).toBeVisible();

  await page.getByRole("button", { name: /SFC Customer Due Diligence Guideline/ }).click();
  await expect(page.getByText("Verify the customer's identity using reliable and independent source documents.")).toBeVisible();
});

test("sends report context to Copilot and renders its streamed answer", async ({ page }) => {
  let requestBody: Record<string, unknown> | null = null;

  await page.route("**/api/backend/api/v1/copilot/chat/stream", async (route) => {
    requestBody = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sse([
        ["intent", { intent: "obligation_mapping" }],
        ["tool_call", { tool: "kag", status: "done", message: "Mapped obligations" }],
        ["token", { text: "Apply enhanced due diligence and retain the verification record." }],
        ["citation_audit", { unsupported_claim_rate: 0 }],
        ["done", { conversation_id: "conversation-e2e-001" }],
      ]),
    });
  });

  await page.goto("/");
  await page.getByLabel("Ask Compliance Copilot").fill("What obligation applies?");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("What obligation applies?")).toBeVisible();
  await expect(page.getByText("Apply enhanced due diligence and retain the verification record.")).toBeVisible();
  await expect(page.getByText("Routing Intent: obligation_mapping")).toBeVisible();
  await expect(page.getByText("Citation risk: 0%")).toBeVisible();

  expect(requestBody).toMatchObject({
    message: "What obligation applies?",
    preferred_language: "zh-HK+en",
    case_context: {
      workspace_id: "customer-account",
      workflow_id: "account-kyc-review",
    },
  });
});

test("reports a stream HTTP error and allows a clean retry", async ({ page }) => {
  let attempts = 0;
  await page.route("**/api/backend/api/v1/bank-account/verify/stream", async (route) => {
    attempts += 1;
    if (attempts === 1) {
      await route.fulfill({ status: 503, contentType: "application/json", body: "{}" });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sse([
        ["token", { text: "Recovered grounded report." }],
        ["done", { workflow_run_id: "workflow-retry-001" }],
      ]),
    });
  });

  await page.goto("/");
  const input = page.getByPlaceholder("Enter compliance scenario...");
  await input.fill("Test transient failure recovery.");
  await page.getByRole("button", { name: "Submit Analysis" }).click();

  await expect(page.getByText(/HTTP 503/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Submit Analysis" })).toBeEnabled();

  await page.getByRole("button", { name: "Submit Analysis" }).click();
  await expect(page.getByText("Recovered grounded report.")).toBeVisible();
  await expect(page.getByText("Completed in")).toBeVisible();
});

test("cancels an in-flight analysis without reporting completion", async ({ page }) => {
  await page.route("**/api/backend/api/v1/bank-account/verify/stream", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 2_000));
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sse([["done", { workflow_run_id: "too-late" }]]),
    }).catch(() => undefined);
  });

  await page.goto("/");
  await page.getByPlaceholder("Enter compliance scenario...").fill("Cancel this analysis.");
  await page.getByRole("button", { name: "Submit Analysis" }).click();
  const cancel = page.getByRole("banner").getByRole("button", { name: "Cancel" });
  await expect(cancel).toBeVisible();
  await cancel.click();

  await expect(page.getByText("Analysis cancelled.")).toBeVisible();
  await expect(page.getByText("Completed in")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Submit Analysis" })).toBeEnabled();
});
