# HK-FinReg AI 项目升级收尾报告

**原执行周期：** 2026-08-05～2026-08-06

**收尾核验：** 2026-08-07

**目标依据：** `docs/system-evaluation-report-2026-08-04.md`

**定位：** Phase 0～3 的工程与研究原型收尾；不构成生产级合规能力声明，Phase 4 架构升级保持冻结。

## 一、最终结论

此次升级的代码收尾已经完成，关键控制可以从干净提交独立构建与验证；但若按目标文档逐条严格验收，整体约 **83%**，不能宣称 100% 完成。

未完成部分不是继续堆叠代码可以替代的：108 题黄金集尚无人工审核，17 份语料有效日期仍待领域核实，真实生成回答尚未批量接入 faithfulness，生产 Redis 与真实摄取链路仍缺环境验证。因此 Phase 4 不得启动，长期门禁阈值也未启用。

| 阶段 | 状态 | 判断 |
|---|---:|---|
| Phase 0：阻断风险 | 约 75% | 高危依赖、CI、安全缓存与健康检查已处理；语料日期及 SUPERSEDES 数据待人工补齐 |
| Phase 1：评测与黄金集 | 约 80% | 108 题、独立 faithfulness 通路、审计包已完成；人工审核和真实回答测量未完成 |
| Phase 2：工程质量 | 100% | 后端、前端、覆盖率防回退、E2E、构建和 CI 门禁均已落地 |
| Phase 3：研究原型 | 100%（研究范围） | 三项确定性实验复跑完成；结论均限制在原型证据范围内 |
| Phase 4：架构升级 | 0%（按计划冻结） | 前置条件未满足；SPO、双图、GraphRAG 不属于本次发布 |

## 二、严格对照目标

### 已严格完成

- faithfulness 已与 claim_recall 解耦，并可通过 `--captured-responses` 或 `response_provider` 接收真实生成回答。
- release gate 明确报告实际 faithfulness 测量覆盖数；无回答时显示 **0/108**，不伪装为已测量。
- 黄金集扩展为 108 题，并生成 108 个可审计 decision package；全部保持 `pending`，未伪造人工审核。
- 发布代码已包含健康检查、Redis 降级、版本化 JSON 语料缓存、图缓存构建器、依赖锁定与 CI 门禁。
- 前端新增两条关键路径 E2E，总计 **4 passed**：报告流式生成与证据展开、Copilot 上下文与流式回答。
- 前端覆盖率门禁按实测基线启用：statements 13%、branches 13%、functions 18%、lines 14%；这是防回退基线，不是质量终点。
- 三项 A/B/诊断实验在 2026-08-07 复跑，输出与报告一致。

### 部分完成

- 黄金集规模达标，但 **0/108 人工审核通过**；审计包见 `backend/data/evaluation/gold_packages/benchmark-gold-packages.json`。
- faithfulness 计算通路已完成，但当前发布门禁没有真实生成回答输入，因此 **0/108 实测**。
- SFC 语料缺口已补，但 17 份语料的 `effective_date` 尚待官网/领域人员核实。
- REFERENCES 已产生实际边；SUPERSEDES 代码路径存在，但 manifest 尚无可验证的替代关系数据。
- `npm audit --audit-level=high` 门禁可通过；仍有 moderate 风险需要独立依赖升级决策。

### 未执行且不应在本轮执行

- 未启用长期阈值 0.90/0.75/0.95/0.05。
- 未把 CDD、PEA-CAE 或 CTRAG 原型直接并入发布门禁或生产检索。
- 未提交/启用 SPO、双图、多跳或 GraphRAG Phase 4 代码。

## 三、可复现证据

| 验证项 | 收尾结果 |
|---|---|
| 后端全量测试 | **211 passed, 4 skipped**；Phase 4 实验测试默认跳过 |
| 发布门禁 | **108 benchmark cases passed**；generation faithfulness measured **0/108** |
| 黄金集审计包 | **108 valid / 108 pending / 0 human-approved** |
| 前端单元测试 | **14 passed**（4 files） |
| 前端覆盖率 | statements 13.79%、branches 13.81%、functions 18.24%、lines 14.51% |
| 前端 E2E | **4 passed** |
| 前端质量链 | lint、typecheck、config tests、production build 全通过 |
| PEA-CAE | 48 场景；0 escalate；recall 0.979→0.979 |
| CDD | 3 场景/5 条合成声明；冲突场景检测率 1.0、误报率 0.0；仅原型信号 |
| CTRAG | 60 场景；recall 0.417→0.500；6 胜/1 负/53 平；尚未分层达标 |

## 四、收尾新增提交

```text
e6d06ae docs: define upgrade closeout design and plan
10608b9 fix: make phase 0-2 controls self-contained
4be9c29 feat(eval): wire captured generator responses into faithfulness
dcd3278 feat(eval): add auditable gold decision packages
dcbb8f6 test(frontend): enforce critical workflow gates
```

## 五、仍需人工完成的验收清单

1. 合规人员按 decision package 审核 108 题，记录 reviewer、时间、依据与决定；不得批量自动标记通过。
2. 领域人员核实 17 份语料的生效日期、状态、来源 URL，并补充真实 SUPERSEDES 关系。
3. 采集实际生成回答，执行 `python -m app.services.evaluation.run_eval --captured-responses <file>`，达到有代表性的 faithfulness 覆盖率后再讨论收紧门禁。
4. 在真实 Redis、生产式 PDF 摄取与检索后端上执行集成验证。
5. 仅当以上项目完成并经业务批准后，重新评估 Phase 4；PEA-CAE 当前结果明确不支持以“稳定改善”为理由启动架构升级。

## 六、最终判定

- **升级是否完善：** 工程收尾完善，业务/合规验收尚未完善。
- **是否严格按目标执行：** 实施顺序与边界已严格纠正；目标中依赖人工审核和真实环境的条目尚未完成，因此不能称为完全达标。
- **完成进度：** Phase 0～3 严格口径约 **83%**；代码可交付项已完成，Phase 4 按目标要求保持冻结。

---

## 七、2026-08-10 Demo 修复实施补充

本轮按“Demo 可暂不人工审核，但必须调用真实 LLM，并显式使用 DeepSeek V4 Flash”的范围实施。历史报告中的 `108 pending / 0 human-approved` 对本 Demo 不再是阻断项；Phase 4 仍明确排除。

### 已完成

- 聊天运行时统一为显式 `deepseek` / `deepseek-v4-flash`，固定官方 API 基址；无密钥、错误模型或错误 profile 均失败关闭，不允许备用模型。
- 已执行一次真实模型握手：模型返回精确 `runtime-ok`，且响应包含用量元数据；该结果只证明真实运行时可用，不代替 12 条正式质量验收。
- 20 份 Demo 语料全部标为必需；任一必需源缺失、损坏、空页或零分块时，缓存构建在写盘前失败。
- 增加官方域名校验、50 MiB 上限、PDF 结构/文本验证及原子替换的安全刷新工具。
- PCPD 6 页扫描 PDF 已通过本地 RapidOCR 摄取，得到 10,947 个字符和 6 个分块。
- 增加固定 12 条分层用例的真实 DeepSeek 抓取器、脱敏工件、有限重试和自动质量门禁。
- 增加独立的手动触发 `DeepSeek Demo Acceptance` CI：先刷新损坏的 HKMA 文件，再构建完整语料，最后执行真实 12 条门禁；不配置人工审批环境。
- 前端已覆盖 HTTP 503 恢复和用户取消流程；Playwright 共 6 条通过，取消不会再误报完成。
- 新增模块定向后端覆盖率门禁为 70%；本地实测 28 项通过、综合覆盖率 84.13%。

### 当前阻断与严格判定

最新全量摄取实测为 **18/20 源成功、2 源失败、1,094 分块**。剩余两份 HKMA 本地 PDF 均为 1,048,576 字节的截断文件，解析错误为 `Cannot find Root object in pdf`：

1. `hkma_amlcft_surveillance_capability_digitalisation_2024`
2. `hkma_svf_amlcft_guideline_2023`

安全刷新已在本机重复尝试，但 HKMA 域名的 TLS 握手在 60 秒后超时；原子替换保证旧文件没有被半成品覆盖。因此完整缓存未生成，12 条真实 DeepSeek 质量门禁没有执行，也没有生成可签收的 `deepseek-demo-live-2026-08-10.json`。

**结论：修复代码已完成，真实 DeepSeek 运行时已验证，但本次 Demo 尚未达到最终验收通过条件。** 当前不能把最小真实握手、旧缓存或 18/20 语料冒充为 12/12 正式验收。官方文件成功刷新并跑通自动工作流后，方可把状态改为“通过”。
