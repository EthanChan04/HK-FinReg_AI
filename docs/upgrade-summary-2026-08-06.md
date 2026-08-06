# HK-FinReg AI 项目升级执行总结（2026-08-06）

> **依据计划：** `docs/superpowers/plans/2026-08-05-system-upgrade-optimization.md`
> **依据评估：** `docs/system-evaluation-report-2026-08-04.md`（修正版）、`docs/risk-assessment-2026-08-04.md`
> **执行周期：** 2026-08-05 ~ 2026-08-06
> **定位：** 研究与原型路线图执行记录；**不构成生产改造承诺**，不据此宣称系统已达高可靠合规水平。

---

## 一、升级总览

| 维度 | 数据 |
|---|---|
| 提交数（本轮） | **8 个**（全部带 `[verified]` 独立审查标记） |
| 后端测试 | 151 → **212 passed**（+61，新增 7 个测试文件） |
| 前端测试 | 0 → **14 单测 + 2 E2E passed**（Vitest + Playwright 新建） |
| 黄金集 | 53 → **108 题**，release gate 通过 |
| 实验原型 | 3 个完成（PEA-CAE / CDD / CTRAG），均有明确结论 |
| 代码审查 | 独立审查子代理通过（0 安全红线、0 逻辑错误），3 条建议全部采纳 |
| 构建环境 | pip check 通过、多 lockfile 无警告、CI/本地依赖集一致 |

**核心原则遵守情况：**
- ✅ 测量先行：faithfulness 独立度量先于一切架构改动
- ✅ 门禁诚实：当前发布标准维持基线阈值，长期目标（90/75/95/5）未提前启用
- ✅ 原型验证：三个实验均为 A/B 原型，未替换现有 Hybrid RAG
- ✅ 阶段退出条件：Phase 4 架构升级**未启动**（条件未满足）

---

## 二、已完成工作明细

### Phase 0：关闭 P0 阻断风险

| 任务 | 结果 | 证据 |
|---|---|---|
| T0-00 基线记录 | ✅ | `docs/eval-baselines/2026-08-05-baseline-108.md` |
| T0-01 前端高危依赖 | ⚠️ 部分 | 6 high → **0 high**（next 16.2.6→16.2.11）；剩 **2 moderate**（postcss，待授权重装） |
| T0-02 CI 安全门禁 | ✅ | release-gates.yml 增加 `npm audit --audit-level=high`；新增 `.github/dependabot.yml`（npm/pip 周更） |
| T0-03 语料时效元数据 | ⚠️ 部分 | SFC 语料已有 2 份（codex 期间补齐）；17 份 `effective_date` 待人工核实（候选清单已产出） |
| T0-04 KAG 关系激活 | ✅ | REFERENCES 已激活（2 条边）；SUPERSEDES 代码路径已测（0 边=manifest 无数据）；审计报告 `t0-04-kag-relations-audit.md` |

### Phase 1：评测定义修正与黄金集扩展

| 任务 | 结果 | 证据 |
|---|---|---|
| T1-01 faithfulness 独立度量 | ✅ | `evaluate_generation_faithfulness()` 解耦 claim_recall；无生成响应时返回 **None**（不再静默等于 claim_recall）；8 个单元测试；`docs/evaluation_protocol.md` 更新为三类指标分层 |
| T1-02 黄金集扩展 | ✅ | 53 → **108 题**（新增 EXP_051~105，全部基于语料真实内容）；分层：en 90/zh-Hant 18，rag 68/kag 23/deep_research 17，10 种任务类型；分类器 CJK 匹配修复（`\b` 在中文间失效的根因） |
| T1-03 发布门禁 | ✅ | 基线阈值门禁（0.45/0.15/0.45/0.10）108 题通过；未测量 faithfulness 不拦截；**评测版本记录**（eval_version=eval-2、benchmark 指纹、corpus 指纹、时间戳） |

**评测指标（108 题全量）：**
| 指标 | 值 | 门禁阈值 | 状态 |
|---|---|---|---|
| avg_claim_recall | 0.727 | ≥ 0.45 | ✅ |
| avg_context_precision | 0.313 | ≥ 0.15 | ✅ |
| avg_unsupported_claim_rate | 0.081 | ≤ 0.10 | ✅ |
| avg_faithfulness | None（独立度量未测量） | — | 符合修正语义 |
| avg_evidence_regulator_coverage | 0.875 | 目标 100% | ⚠️ 待语料补齐 |

### Phase 2：P1 工程风险

| 任务 | 结果 | 证据 |
|---|---|---|
| T2-01 测试层次补齐 | ✅ 后端 + 前端 | HTTP 集成 9 用例（认证/CORS/404/422/429/健康豁免限流）；SSE 契约 4 用例；前端 Vitest 14 用例 + Playwright E2E 2 用例（Windows 用系统 Edge 规避安全拦截）；istanbul 覆盖率 50% 起步阈值 |
| T2-02 构建可复现 | ✅ | 多 lockfile 无警告（根 package.json 为独立工具包装器）；pyproject +`psycopg[binary]` + lock 重生成（零版本漂移）；pip check 通过；**根因发现**：pip check 冲突来自 Hermes 注入的 PYTHONPATH，项目 venv 本身干净 |
| T2-03 分布式限流/健康检查 | ✅ | Redis 调用期降级（eval 失败→内存 store+单次告警，不再 500）；identity-aware 键（SHA-256 摘要）；`/health/live`+`/health/ready` 拆分；健康端点豁免限流 |
| T2-04 Pickle 替换 | ✅ | 生产路径零 pickle；JSON 版本化缓存（schema_version+manifest_digest+parser_version）；3 个安全测试（篡改/不匹配/损坏→重建） |

### Phase 3：研究原型 A/B 实验（全部收官）

| 实验 | 设计 | 结果 | 结论 |
|---|---|---|---|
| T3-01 PEA-CAE 成本感知升级（NR-01） | 48 场景 A/B：现有两轮检索 vs 门控全文升级 | **0/48 升级** | 现有检索已达 0.979 recall，全文升级无增益；模块归档为可复用组件，**不扩展** |
| T3-02 CDD 冲突诊断（NR-02） | 正确法规 vs 过时法规/误导摘要 | **检测率 1.0、误报率 0.0** | HK 监管场景可落地；区分性 token 匹配（停用词过滤）是必要改进；诊断工具**不进门禁** |
| T3-03 CTRAG 自适应分块（NR-03） | 60 场景：固定 1500/200 vs 自适应（结构边界） | claim_recall **0.417→0.500（+8.3pp）**，6胜1负 | 结构边界保留条款完整性；原型保留供 U-02 候选；生产分块**未替换** |

### Phase 4：架构升级决策门（未启动）

5 个决策条件当前状态：
- [x] faithfulness 独立度量可用（运行 <2 周）
- [ ] 黄金集 ≥100 且**人工审核通过**（审核未完成）
- [ ] PEA-CAE 稳定改善（**实验结论为否**）
- [x] CDD 可落地
- [~] R-01~R-05 关闭（R-01 剩 moderate、R-03 前端已补）

**结论：保持不启动。** 候选升级（U-02 SPO 三元组 / U-03 双图 / U-05 GraphRAG / U-04 政策差距图）冻结，等待人工审核与业务决策。

---

## 三、提交历史（本轮 8 个）

```
0ac2815 [verified] feat(test): frontend Vitest + Playwright E2E infra (T2-01) + T0-04 审计
00c0f00 [verified] feat(eval): evaluation provenance metadata (T1-03)
95be9e4 [verified] feat(exp): CTRAG adaptive chunking A/B (NR-03)
1af1f68 [verified] feat(exp): CDD conflict diagnosis prototype (NR-02)
02ab62b [verified] feat(exp): PEA-CAE cost-aware escalation gate prototype (NR-01)
133db3d [verified] chore(ci): dependabot config + eval baseline archives
491985d [verified] fix(ops): Redis rate-limit fallback + HTTP/SSE test coverage
5290ae7 [verified] feat(eval): independent generation faithfulness + 108-case golden set
```

**新增/修改文件分布：**
- 后端代码：rag_eval、run_eval、release_gate、query_classifier、rate_limit、cdd_diagnoser、escalation_gate、adaptive_chunker、实验脚本 ×3
- 后端测试：test_generation_faithfulness、test_http_api、test_sse_contract、test_escalation_gate、test_cdd_diagnoser、test_adaptive_chunker、test_eval_versioning、test_corpus_cache_safety、test_rate_limit_redis_fallback（+9 文件）
- 前端：vitest.config/setup、playwright.config、e2e/smoke.spec、4 个测试文件、package.json（scripts+devDeps）
- 数据：benchmark_questions.json（53→108）
- 文档：evaluation_protocol、dependabot.yml、eval-baselines/ ×5、experiments/ ×3

---

## 四、未完成事项（需人工/业务侧参与）

### 🔴 需授权才能执行

**T0-01 剩余 2 个 moderate 漏洞（postcss）**
- 现状：overrides 已改为 postcss 8.5.25（修复版），但 `next@16.2.11` 内部嵌套的 `node_modules/next/node_modules/postcss@8.5.18` 未被覆盖
- 待办：删除该嵌套目录后 `npm install` 重装（破坏性操作，需确认）
- 影响：`npm audit` 从 2 moderate → 0；`npm audit --audit-level=high` 门禁不受影响（已 0 high）

### 🟡 需业务侧人工核实

**T0-03 语料时效元数据（17 份 `effective_date`）**
- 已产出候选清单：`docs/eval-baselines/t0-03-corpus-metadata-audit.md`
- 2 份可从正文确认（hkma_tm_ai_thematic_review_2024→2024-04-17、hkma_genai_consumer_protection_2024→2024-08-19）
- 其余 15 份需对照 HKMA/SFC/PCPD 官网核实
- 验收标准：`source_url`/`status`/有效日期完整率 100%；AI 投顾证据覆盖 100%（当前 87.5%）

**T1-02 黄金集人工审核（108 题）**
- 计划要求"人工审核所有 gold answers 的正确性"，审计轨迹缺失
- 建议按监管机构（SFC 32/HKMA 28/三监管 22/PCPD 18）分批复核
- 完成后可作为门禁阈值收紧（0.90/0.75/0.95/0.05）的前置条件

### ⚪ 已记录的建议（未实施，需决策）

1. **T0-04 检索降级**：graph_retriever 未消费 `status=superseded`——建议 confidence 惩罚 + 替代文档提示（需先补 manifest supersedes 数据）
2. **U-02 SPO 三元组前置**：CTRAG 自适应分块（+8.3pp）可作为三元组抽取前的分块候选
3. **评测成本**：独立 faithfulness 需真实生成器输出接入（用 `split_response_claims` 提取回答声明）
4. **前端覆盖率**：50% 阈值是起步值，当前全量 13.87%（已测文件 85-100%），随测试增加逐步提升

---

## 五、验证总表

| 验证项 | 命令 | 结果 |
|---|---|---|
| 后端全量测试 | `cd backend && python -m pytest tests -q` | **212 passed**（23s） |
| 发布门禁 | `python -m app.services.evaluation.release_gate` | **passed: 108 benchmark cases** |
| 前端单测 | `cd frontend && npm run test` | **14 passed**（4 files） |
| 前端 E2E | `npx playwright test` | **2 passed** |
| 前端构建 | `npm run build` | ✅ 编译成功 |
| 依赖一致性 | `.venv python -m pip check`（剥离 Hermes PYTHONPATH） | ✅ No broken requirements |
| 安全审计 | `npm audit --audit-level=high` | ✅ 0 high（2 moderate 待处理） |

---

## 六、风险与开放问题

1. **评测路径局限**：确定性 token-overlap 检索（`_retrieve_eval_documents`）与生产 Chroma/BM25 不同——黄金集 claim 均验证可命中，但生产检索需另行验证
2. **中文题检索依赖英文缩写混入**（如 "SFC"/"CPT"）——香港金融实务合理，纯中文无缩写场景仍会失败，建议后续加 CJK tokenization
3. **旧题 28 个 claim_recall=0**（EXP_006~050 区间）为历史标注遗留，留待人工审核黄金集时一并处理
4. **Redis 依赖**：限流生产需 `RATE_LIMIT_STORAGE_URL` 配置真实 Redis；Lua 脚本需真机验证
5. **LLM 评测偏差**：独立 faithfulness 使用 LLM 时需保留人工抽检集（确定性匹配为降级路径）

---

## 七、下一步建议

| 优先级 | 事项 | 依赖 |
|---|---|---|
| 1 | 授权 T0-01 postcss 重装（清除最后 2 moderate） | 用户确认 |
| 2 | 业务侧核实 17 份语料日期 + 补 supersedes 数据 | 领域人员 |
| 3 | 黄金集 108 题人工审核（分批） | 合规人员 |
| 4 | 审核通过后复跑评测 → 评估门禁阈值收紧 | 1-3 完成后 |
| 5 | 决策 Phase 4 候选（U-02 优先，CTRAG 分块为前置） | 4 完成后 + 产品决策 |
