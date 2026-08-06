import { expect, test } from "@playwright/test";

// 冒烟测试：首页加载、标题与核心元素存在
test("首页加载：标题与主标题正确", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("HK-FinReg AI — Multi-Agent Compliance Engine");
  await expect(
    page.getByRole("heading", { level: 1, name: "HK-FinReg AI" })
  ).toBeVisible();
});

test("首页加载：核心功能徽章可见", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Evidence-first review")).toBeVisible();
  await expect(page.getByText("RAG + KAG + DeepResearch")).toBeVisible();
  await expect(page.getByText("Human review gates")).toBeVisible();
});
