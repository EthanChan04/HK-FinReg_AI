import { defineConfig } from "@playwright/test";

// Playwright E2E 配置（T2-01 前端测试基建）
// webServer 自动启动 Next dev server（本地已运行则复用）
// Windows 本地使用系统 Edge（channel: msedge）—— 规避 ms-playwright 下载目录
// 被安全软件拦截执行的问题；Linux CI 使用 Playwright 自带 chromium。
const isWindows = process.platform === "win32";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3000",
    channel: isWindows ? "msedge" : undefined,
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run dev -- --webpack",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
