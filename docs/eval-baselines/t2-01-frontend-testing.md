# T2-01 前端测试基建 — 基线报告（frontend）

- 日期：2026-08-06
- 范围：`F:\MyFintech\frontend`（Next.js 16.2.11 / React 19.2.4 / TS 5）
- 风险对应：R-03（无前端组件测试、E2E 与覆盖率阈值）
- 分支：codex/bank-tob-module-rearchitecture（未提交改动均为既有工作，本次仅改文件、未 git add/commit/push）

## 1. 安装的依赖（devDependencies，npm install -D）

| 包 | 版本 |
|---|---|
| vitest | ^4.1.10 |
| @vitejs/plugin-react | ^6.0.5 |
| @testing-library/react | ^16.3.2 |
| @testing-library/jest-dom | ^7.0.0 |
| jsdom | ^29.1.1 |
| @vitest/coverage-istanbul | ^4.1.10 |
| @playwright/test | ^1.62.1 |

Playwright 浏览器：`npx playwright install chromium` 成功下载（chromium-1234 / headless shell）。package-lock.json 已同步更新。

## 2. 测试文件（4 个文件，14 个用例）

| 文件 | 用例数 | 覆盖要点 |
|---|---|---|
| `src/components/__tests__/EvidencePanel.test.tsx` | 4 | 加载骨架屏、空状态提示、证据卡片渲染（标题/监管机构/分数/数量）、点击展开/收起正文 |
| `src/components/__tests__/WorkflowSelector.test.tsx` | 3 | 工作流与引擎标签渲染、点击触发 onChange 且传参正确、disabled 时不可点击 |
| `src/components/chat/__tests__/CitationCards.test.tsx` | 3 | 空证据渲染 null、卡片渲染（ID/标题/监管机构/页码/正文）、超过 4 条只渲染前 4 条 |
| `src/lib/__tests__/reportFormatting.test.ts` | 4 | formatTextReport 头部+正文、空文本回退、义务映射 JSON 渲染、空审查队列提示（纯函数，无 DOM） |

选择依据：EvidencePanel（证据展开是核心交互）、WorkflowSelector（合规分析入口）、CitationCards（引用展示）、reportFormatting（无副作用的纯函数，最适合起步）。App Router 页面（page.tsx）是巨型客户端组件、依赖 SSE hooks，暂不纳入组件测试。

## 3. 配置

- `vitest.config.ts`：@vitejs/plugin-react + `@/` → `src/` 别名 + jsdom 环境 + setup 文件 + `include: src/**/*.{test,spec}.{ts,tsx}`；coverage 用 istanbul provider，`include: src/**/*.{ts,tsx}`，排除测试文件/`src/types`/`src/app`（路由层），reporter 含 text/json-summary/html。
- `vitest.setup.ts`：注册 jest-dom matchers + 显式 `afterEach(cleanup)`（未启用 vitest globals）。
- `package.json` scripts 新增：`test`（vitest run）、`test:watch`（vitest）、`test:coverage`（vitest run --coverage）、`test:e2e`（playwright test）。
- `eslint.config.mjs`：globalIgnores 增加 `coverage/**`、`e2e/**`（避免 lint 扫产物目录）。
- `.gitignore`：增加 `/test-results/`、`/playwright-report/`（/coverage 原有）。

## 4. 覆盖率首次报告（2026-08-06）

`npm run test:coverage` 可正常产出报告。全量（含未测组件/hooks）：

| 指标 | 全量 |
|---|---|
| Statements | 13.87% |
| Branches | 13.85% |
| Functions | 18.38% |
| Lines | 14.58% |

已测文件（起点样本）：EvidencePanel 94.44% lines / 90% stmts、CitationCards 100% lines、WorkflowSelector 85.71% lines。

阈值说明：按任务要求设 `lines/statements/functions/branches >= 50` 作为**起步阈值**（防倒退）。当前全量覆盖率未达 50%，`test:coverage` 会以非零退出码结束并打印阈值 ERROR——这是**预期行为**（基线阶段），CI 目前只跑 `npm run test`（不含 coverage），不受影响。后续随测试补充覆盖 >50% 后自然转绿；建议下一轮优先补 AgentTimeline、SuggestedPrompts、ReportPanel 的 idle/action_required 分支。

## 5. CI 改动（.github/workflows/release-gates.yml）

frontend-build job 中新增一步（位于 "Frontend config tests" 之后、"Audit production dependencies" 之前），其余步骤未动：

```yaml
      - name: Run frontend unit tests
        run: npm run test
```

CI 用 `npm ci`，package-lock.json 已含新 devDeps，可直接安装。

## 6. Playwright E2E — 完成

- `playwright.config.ts`：testDir `./e2e`，webServer 自动 `npm run dev`（`reuseExistingServer: !CI`，超时 120s），baseURL localhost:3000。
- `e2e/smoke.spec.ts`：2 个冒烟用例（首页加载后标题正确 + h1 "HK-FinReg AI" 可见；核心徽章 "Evidence-first review" / "RAG + KAG + DeepResearch" / "Human review gates" 可见）。
- **Windows 本机注意**：ms-playwright 下载的 chromium 可执行文件被系统安全软件拦截（`spawn UNKNOWN` / Permission denied），故配置为 `channel: isWindows ? "msedge" : undefined` —— Windows 本地用系统 Edge 跑（已验证 2/2 通过），Linux CI 仍用 Playwright 自带 chromium。若安全软件放行 ms-playwright 目录，可去掉 channel 回退。
- 首次实跑结果：`2 passed (5.8s)`。

## 7. 验证命令输出摘要

| 命令 | 结果 |
|---|---|
| `npm run test` | 4 files passed / 14 tests passed（~2.5s） |
| `npm run test:coverage` | 报告正常产出；阈值 50% 未达（预期，见第 4 节） |
| `npm run typecheck` | tsc --noEmit 通过（含测试文件） |
| `npm run lint` | 0 errors（coverage/e2e 已忽略） |
| `npm run test:config` | 通过（既有 node --test 配置测试不受影响） |
| `npm run build` | Next.js 16.2.11 编译成功，静态页 4/4，TypeScript 通过 |
| `npx playwright test` | 2 passed (5.8s)（系统 Edge channel） |
| 后端 | 未改动任何 backend 文件（git status 中 backend 的 M 均为既有改动），未运行后端测试 |

## 8. 后续待办（建议）

1. 补 SuggestedPrompts、AgentTimeline、ReportPanel（idle/action_required/置信度徽章分支）、KnowledgeGraphPanel 等组件测试，把全量覆盖率推到 >50%。
2. 达到阈值后在 CI frontend job 增加 `npm run test:coverage` 步骤（当前刻意不加，避免红门）。
3. 如需在 Linux CI 跑 E2E，需在 workflow 中加 `npx playwright install --with-deps chromium`（本报告未加入，任务范围外）。
