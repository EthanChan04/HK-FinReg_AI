import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Vitest 单元测试配置（T2-01 前端测试基建）
// - jsdom 环境 + jest-dom matchers（见 vitest.setup.ts）
// - 覆盖率：istanbul provider，起步阈值 lines/statements/functions/branches >= 50
//   （起步阈值，随测试覆盖提升后可逐步上调）
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    coverage: {
      provider: "istanbul",
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/**/*.{test,spec}.{ts,tsx}",
        "src/**/__tests__/**",
        "src/types/**",
        "src/app/**",
      ],
      reporter: ["text", "json-summary", "html"],
      reportsDirectory: "coverage",
      thresholds: {
        // 起步阈值 —— 仅为防止覆盖率倒退，非最终目标
        lines: 50,
        statements: 50,
        functions: 50,
        branches: 50,
      },
    },
  },
});
