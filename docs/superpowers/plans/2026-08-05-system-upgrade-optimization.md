# HK-FinReg AI 项目升级与优化计划

> **依据文档：**
> - `docs/system-evaluation-report-2026-08-04.md`（修正版，本计划主依据）
> - `docs/risk-assessment-2026-08-04.md`（风险 R-01 ~ R-06）
> - `docs/arxiv-github-upgrade-options-2026-08-04.md`（升级方案 U-01 ~ U-05）
> - `docs/evaluation_protocol.md`（评测协议与门禁）
>
> **制定日期：** 2026-08-05
> **定位：** 研究与原型路线图的执行计划。**不作为生产改造承诺**，不据此宣称系统已达高可靠合规水平。

**Goal：** 按「先测量 → 再实验 → 后架构」的顺序，关闭 P0/P1 风险、建立可信评测体系、完成 3 个研究原型 A/B 实验，为是否进入架构升级提供可验证的数据决策。

**核心原则：**
1. **测量先行**——任何架构改动之前，先修正 `faithfulness` 定义，建立独立生成忠实度度量。
2. **门禁诚实**——当前发布标准维持基线阈值（0.45 / 0.15 / 0.45 / 0.10）；90%/75%/95%/5% 仅是长期目标，不提前启用。
3. **原型验证**——所有新方案（PEA-CAE、CDD、CTRAG、双图、GraphRAG）以单场景 A/B 实验验证，**不直接替换**现有 Hybrid RAG。
4. **每阶段有退出条件**——不满足条件不进入下一阶段。

---

## 执行顺序总览

| Phase | 内容 | 关键交付 | 退出条件 |
|---|---|---|---|
| Phase 0 | 关闭 P0 阻断风险（R-01 前端漏洞、R-02 语料缺口） | `npm audit` 归零；SFC 语料补齐 | R-01、R-02 验收标准达成 |
| Phase 1 | 评测定义修正 + 黄金集扩展 + 门禁更新 | 独立 faithfulness 度量；100+ 人工审核黄金集 | 度量可信、基准分层、门禁生效 |
| Phase 2 | 关闭 P1 工程风险（R-03 ~ R-05）+ P2（R-06） | 分层测试、可复现构建、分布式限流、就绪检查 | R-03~R-06 关闭或有可验证控制 |
| Phase 3 | 研究原型 A/B 实验（NR-01 / NR-02 / NR-03） | 三个实验各一份 A/B 数据报告 | 每实验有明确结论 |
| Phase 4 | 架构升级决策门（U-02 / U-03 / U-05 等） | 决策评审 | 见 Phase 4 决策条件 |

**建议并行：** Phase 0 与 Phase 1 的 T1-01（faithfulness 解耦）互不依赖，可并行；Phase 2 与 Phase 3 可在 Phase 1 门禁生效后并行开展。

---

## Phase 0：关闭 P0 阻断风险（目标：本周）

> 报告结论：**关闭 R-01 和 R-02 远比架构升级紧迫**。本阶段不动检索架构。

### T0-00 记录当前基线（先做，10 分钟）

**目的：** 为后续所有改动建立可对比的基线快照。

**步骤：**
1. `cd backend && python -m pytest -q` → 记录结果（当前 151 passed, 21.75s）
2. `python -m app.services.evaluation.run_eval` → 保存输出到 `docs/eval-baselines/2026-08-05-baseline.txt`
3. `cd frontend && npm run lint && npm run typecheck && npm run build` → 记录结果

**验证：** 三份基线记录齐全，可随时 diff 对比。

### T0-01 升级前端高危依赖（R-01）

**文件：**
- `frontend/package.json`
- `frontend/package-lock.json`

**步骤：**
1. `cd frontend && npm audit` 确认当前漏洞清单（6 high、1 low）。
2. 将 `next` 与 `eslint-config-next` 升级到同一 minor 内已修复版本（如 16.2.x 最新 patch）；同时刷新 `postcss`、`sharp`、`js-yaml`、`brace-expansion`、`picomatch` 的实际解析版本（`npm ls postcss sharp js-yaml brace-expansion picomatch` 核对）。
3. `npm audit fix` 处理可自动修复项；剩余手工升级。
4. 重新执行 `npm run lint && npm run typecheck && npm run build`。
5. 浏览器冒烟回归：代理白名单、认证转发、SSE 流式、主要工作区流程（Copilot / DeepResearch / 证据面板）。

**验收：**
- `npm audit` 输出 0 critical、0 high。
- lint / typecheck / build 全部通过；冒烟回归通过。

### T0-02 CI 安全审计门禁（R-01 固化）

**文件：**
- `.github/workflows/*.yml`（前端 CI 工作流）

**步骤：**
1. CI 增加 `npm audit --audit-level=high` 步骤，检出 high/critical 即失败。
2. 配置 Dependabot（或等价依赖更新机制），每周自动提 PR。

**验收：** 推送一个含已知 high 漏洞的测试依赖时 CI 红灯；Dependabot 配置生效。

### T0-03 补齐 SFC 语料与时效元数据（R-02）

**文件：**
- `backend/data/source_manifest.json`
- `backend/data/regulations/`（语料文件）
- 修改：`backend/app/services/corpus/`（如需支持新元数据字段）

**步骤：**
1. 优先采集与 **AI 投顾、投资建议、适当性、产品说明、客户保障** 相关的 SFC 官方材料（当前 SFC 语料为 0，AI 投顾锚点场景必须覆盖 HKMA/SFC/PCPD 三家）。
2. 为每份文档补齐 `source_url`、`effective_date`、`status`、`supersedes`、`references`；入库阶段验证官方域名、文件哈希、发布日期，并保留采集时间。
3. 重新采集被完整性校验安全跳过的 2 份 HKMA PDF。
4. `python -m app.services.corpus.build_cache` 重建缓存，`python -m app.services.evaluation.run_eval` 观察 `evidence_regulator_coverage`。

**验收：**
- 每个目标业务场景具备其预期监管机构的有效官方证据；AI 投顾基准的 `evidence_regulator_coverage` 达到 100%。
- 生产语料 `source_url` / `status` / 有效日期完整率 100%（或对不适用的字段记录明确原因）。
- 已失效/被替代文档可自动降级或排除，并展示替代关系。

### T0-04 激活 KAG 的 REFERENCES / SUPERSEDES 关系（R-02 辅助）

**文件：**
- `backend/app/services/kag/graph_builder.py`
- `backend/app/services/kag/ontology.py`

**步骤：**
1. 将 manifest 中已有的 `supersedes` / `references` 字段接入图构建，形成 `Document → SUPERSEDES → Document`、`Clause → REFERENCES → Clause/Document` 边。
2. `python -m app.services.kag.build_graph_cache` 重建图缓存。
3. 为已失效文档的检索结果增加降级/排除逻辑（查询时过滤 `status != active` 的文档，或标记展示替代关系）。

**验收：** 图谱面板可展示替代关系；检索不再返回已失效法规而不加说明。

---

## Phase 1：评测定义修正与黄金集扩展（目标：2 周内）

> 报告结论：**引入任何新架构之前，先建立准确的测量体系**。当前 `faithfulness = claim_recall`（`rag_eval.py:164,170`），不是独立测量。

### T1-01 实现独立的生成忠实度度量（最高优先）

**文件：**
- 修改：`backend/app/services/evaluation/rag_eval.py`（解耦第 164、170 行）
- 新建：`backend/app/services/evaluation/generation_faithfulness.py`
- 修改：`backend/app/services/evaluation/run_eval.py`（输出新指标列）
- 测试：`backend/tests/test_generation_faithfulness.py`

**设计（核心接口）：**

```python
# generation_faithfulness.py —— 独立于 claim_recall 的生成忠实度评估
def evaluate_generation_faithfulness(
    query: str,
    response: str,            # 生成器实际回答（真实输出，而非基准中的 expected_claims）
    retrieved_context: list[dict],  # 检索上下文 chunks
    extractor,                # claim 提取（LLM，带降级 fallback）
    verifier,                 # claim 支持验证（LLM + 确定性证据匹配双重路径）
) -> dict:
    """返回:
    {
      "faithfulness": float,      # 回答中声明被上下文支持的比例（独立测量）
      "hallucination_rate": float,
      "per_claim": [              # 每条声明可追溯
        {"claim": str, "supported": bool, "evidence_indexes": [...],
         "reason": str}
      ]
    }
    """
```

**步骤（TDD）：**
1. 写失败测试：构造「回答含 3 条声明、上下文仅支持 2 条」的样例，断言 `faithfulness == 2/3` 且不等于 `claim_recall`。
2. 运行 `pytest tests/test_generation_faithfulness.py -v` → 预期 FAIL（模块不存在）。
3. 实现 `generation_faithfulness.py`：LLM 提取回答中的声明 → 逐条验证是否被检索上下文支持；LLM 路径失败时降级到确定性关键词/证据匹配，保证评测可复现。
4. 修改 `rag_eval.py`：删除 `faithfulness = claim_recall` 直接赋值，改为调用独立度量；`claim_recall` 保持原语义（检索质量指标）。
5. 运行测试 → PASS；再跑全量 `python -m pytest -q`（151+ 用例全绿，含 9/9 arXiv 回归）。
6. 更新 `docs/evaluation_protocol.md` 指标表：按三类分层描述。

**指标三分层（写入协议文档）：**
- **检索质量**：`claim_recall`、`context_precision`、`noise_sensitivity`
- **生成忠实度**：`faithfulness`（独立测量）、`hallucination_rate`
- **引用正确性**：`citation_supported_rate`、`unsupported_claim_rate`

**验收：** 评测报告中 `faithfulness` 不再等于 `claim_recall`；每条回答声明可追溯到证据片段或明确的「无证据」理由。

### T1-02 黄金集扩展 53 → 100+（配合 T0-03 语料）

**文件：**
- `backend/data/evaluation/benchmark_questions.json`
- 新建：`backend/data/evaluation/gold_packages/`（ScenarioBench 风格 gold 决策包）

**步骤：**
1. 按矩阵分层补齐：**监管机构**（HKMA / SFC / PCPD / 跨监管）× **语言**（en / zh-Hant）× **任务类型**（`routine_review` / `product_launch` / `regulatory_memo` / `obligation_extraction` / 有效性冲突 / 拒答）。
2. 参考 ScenarioBench（NR-05）的 YAML schema 与 `witness_trace` 概念，为每条新增题目建立 gold 决策包：`decision`（标准答案）+ `witness_trace`（最小见证轨迹：证据→条款→决策推理）+ `clause_set`（涉及条款集合）。
3. 新增题目优先覆盖 SFC 场景（配合 T0-03），确保 `expected_regulators` 与证据侧一致。
4. **人工审核全部 gold answers**：审核记录存 `docs/eval-baselines/gold-review-2026-08-*.md`，包含审核人、日期、修订历史。

**验收：** 基准 ≥100 题；按监管机构/语言/任务类型分层统计可输出；gold answers 全部有人工审核记录；`run_eval` 自动拾取新题。

### T1-03 发布门禁更新与评测版本记录

**文件：**
- `backend/app/services/evaluation/release_gate.py`

**步骤：**
1. 基线阈值**保持不变**（`claim_recall >= 0.45`、`context_precision >= 0.15`、`faithfulness >= 0.45`、`unsupported_claim_rate <= 0.10`），但门禁输出按 T1-01 的三类指标分组展示。
2. 将长期目标（0.90 / 0.75 / 0.95 / 0.05）标记为 roadmap 指标，仅在黄金集扩展 + 人工审核完成后由协议文档正式收紧，**不提前启用**。
3. 每次评测记录：评测模型、prompt 版本、语料清单哈希、索引/图缓存版本、日期 → 存 `docs/eval-baselines/`。

**验收：** 门禁脚本输出三类指标分组；每次运行可复现评测版本信息；CI 能在指标显著退化时红灯。

---

## Phase 2：P1 工程风险与可复现构建（目标：与 Phase 1 并行，3 周内）

### T2-01 测试层次补齐（R-03）

**文件：**
- 新建：`backend/tests/test_http_api.py`（FastAPI HTTP 集成测试）
- 新建：`backend/tests/test_sse_contract.py`（SSE 事件契约 + 断线恢复）
- 新建：`frontend/src/**/*.test.tsx`（Vitest + Testing Library 组件测试）
- 新建：`frontend/e2e/`（Playwright 关键旅程 E2E）
- 修改：`.github/workflows/*.yml`（覆盖率阈值步骤）

**步骤：**
1. 后端集成测试：认证、CORS、限流响应（429）、错误响应、SSE 事件顺序（`meta`/`delta`/`done` 契约）；增加 `@pytest.mark.integration` 标记区分快慢测试。
2. 真实模型/SSE 断线测试：连接中断后重连、取消、超时恢复（以集成标记运行，不进默认快测）。
3. 前端组件测试：提交分析、证据展开、Copilot 对话、人工复核面板、取消与异常恢复。
4. Playwright E2E：核心用户旅程「提交分析 → 证据展开 → Copilot → 人工复核 → 取消/异常恢复」。
5. 配置覆盖率：后端 `pytest --cov` 阈值（核心模块如 retrieval/evaluation 不低于 70% 起步），前端 `vitest --coverage` 阈值。

**验收：** 关键 API 与前端主流程均有自动化测试；CI 同时阻断检索退化、引用不支持率上升与前端 E2E 回归；覆盖率报告可见且阈值生效。

### T2-02 构建环境可复现（R-04）

**文件：**
- `backend/requirements.txt`、`backend/requirements.lock`、`backend/pyproject.toml`
- 根目录 `package.json` / `package-lock.json`
- `frontend/next.config.ts`

**步骤：**
1. 确认根目录 `package.json` 归属：若为项目工具依赖则正式配置 workspace；否则移出项目工作区。**目标：消除「仓库根 + frontend/ 双 lockfile」构建警告。**
2. 后端：核对 `requirements.txt` 与 `requirements.lock`、`pyproject.toml` 的一致性；CI 用锁文件安装（`pip install -r requirements.lock` 或等价的锁定安装），并运行 `pip check`。
3. 项目专用 `.venv`（仓库根已存在，确认 CI/本地均使用它，不复用其他工具的 Python 环境）。
4. `frontend/next.config.ts` 显式设置 Turbopack / 输出追踪根目录（`outputFileTracingRoot`），消除推断警告。

**验收：** 全新环境单条文档化命令可完成安装、测试、构建；`pip check` 无冲突；Next.js 构建无多 lockfile 警告；本地与 CI 解析的核心依赖版本一致。

### T2-03 分布式限流与健康检查语义修正（R-05）

**文件：**
- `backend/app/core/rate_limit.py`（限流）
- `backend/app/main.py`（健康接口）
- `frontend/src/app/api/backend/[...path]/route.ts`（代理，如需调整转发头）

**步骤：**
1. 限流改用 Redis（项目已有 Redis 依赖）：`INCR` + `EXPIRE` 滑动窗口；键优先用租户/用户/API 凭证，仅当可信代理链配置完成后才用原始客户端 IP。
2. `main.py` 拆分为 `/health/live`（进程存活，快速返回）与 `/health/ready`（验证模型服务、数据库/向量库、语料索引是否就绪；外部探测设短超时 + 缓存 + `degraded` 状态，避免放大故障）。
3. 测试：多用户不共享限额、代理后按身份限流、多副本共享计数、依赖故障时就绪检查返回非 2xx。

**验收：** 多副本限流计数一致；模型/数据库/索引不可用时 `/health/ready` 非 2xx 或明确 degraded；监控可区分限流、上游故障、索引未就绪与应用异常。

### T2-04 Pickle 缓存替换（R-06，P2 低成本先行）

**文件：**
- `backend/app/services/agents/builder.py`（缓存读写）
- `backend/app/services/evaluation/run_eval.py`（缓存读取）

**步骤：**
1. `corpus_documents.pkl` 改为 **JSONL**（或 Parquet）安全格式，带显式 schema 版本。
2. 缓存键包含：语料清单哈希 + 解析器版本 + 数据结构版本；缓存损坏/版本不匹配时安全失败并重建，记录结构化告警。
3. 测试：篡改、版本不匹配、缓存损坏三类用例均安全失败并重建。

**验收：** 生产运行路径不再对可被非可信主体修改的文件执行 `pickle.load()`；缓存具备 schema/version 检测能力。

---

## Phase 3：研究原型 A/B 实验（目标：Phase 1 门禁生效后，4 周内）

> 定位：**研究原型**，全部以单场景 A/B 进行，**不替换现有流程、不进发布门禁**。每实验输出一份数据报告，作为 Phase 4 决策输入。

### T3-01 PEA-CAE 成本感知升级原型（NR-01，高优先）

**场景选择：** AI governance review（已有充足语料，配合 T0-03 后证据更全）。

**文件：**
- 修改：`backend/app/services/deepresearch/workflow.py`（增加实验分支，默认关闭）
- 新建：`backend/app/services/deepresearch/escalation_gate.py`（两阶段：低成本检索 → 预期证据增益不足时才升级全文读取）

**步骤：**
1. 实现最小化 escalation gate：第一阶段高精度检索（现有 hybrid 检索），仅当证据增益预期超过成本阈值时升级到全文读取（第二阶段）。
2. 在扩展后的黄金集上 A/B：**对照组**=现有 DeepResearch 流程；**实验组**=带 escalation gate。
3. 对比指标：质量侧（`claim_recall`、独立 `faithfulness`、`unsupported_claim_rate`）+ 成本侧（token 数、延迟、检索调用次数）。
4. 输出报告 `docs/experiments/2026-XX-pea-cae-ab.md`。

**验收：** 报告包含质量 vs 成本完整对比；**仅当**实验显示稳定改善才讨论扩展到更多场景，否则原型归档。

### T3-02 CDD 冲突诊断实验（NR-02）

**文件：**
- 新建：`backend/data/evaluation/conflict_cases.json`（知识冲突用例集）
- 修改：`backend/app/services/copilot/` 下 `citation_verifier`（增加 CDD 风格诊断分支）

**步骤：**
1. 构造两类冲突用例：**正确法规 vs 过时法规**、**正确法规 vs 误导性摘要**（各 5~10 条）。
2. 在 citation_verifier 中实现 CDD 风格诊断：推理时分离「上下文答案」（仅依据检索上下文）与「先验答案」（模型内部知识），比较两者以识别知识冲突。
3. 记录**冲突检测率**与**误报率**；作为诊断工具输出，**不直接作为质量门禁**。
4. 输出报告 `docs/experiments/2026-XX-cdd-conflict-diagnosis.md`。注意：论文 15% 准确率为 TruthfulQA 误导注入最坏情况结果，**不得概括为所有 RAG 场景的预期**。

**验收：** 冲突用例集可复现运行；报告含检测率/误报率与典型失败案例分析。

### T3-03 CTRAG 自适应分块 A/B（NR-03）

**文件：**
- 新建：`backend/app/services/retrieval/adaptive_chunker.py`（实验分支）
- 修改：`backend/app/services/corpus/`（摄取时可选分块策略）

**步骤：**
1. 选取高价值监管文档（如 HKMA AML Guideline）实现自适应分块（按章节/条款边界 + 语义完整性动态确定块大小）。
2. 与现有固定分块方案在黄金集上 A/B：对比 `claim_recall`、`context_precision`、`noise_sensitivity`。
3. **不替换**现有 RAG 分块；结果仅作为 U-02（SPO 三元组）前置决策输入。
4. 输出报告 `docs/experiments/2026-XX-ctrag-chunking-ab.md`。注意：CTRAG 论文验证成熟度为「一家四大 POC」，报告中如实标注参考价值等级。

**验收：** A/B 数据完整；若自适应分块在至少 2 个任务类型上稳定优于固定分块，才进入三元组阶段的候选方案。

---

## Phase 4：架构升级决策门（远期，仅满足全部条件后启动）

**决策条件（全部满足才讨论架构升级）：**
- [ ] T1-01 独立 faithfulness 度量可用，且作为门禁运行 ≥ 2 周
- [ ] 黄金集 ≥ 100 题且人工审核通过（T1-02）
- [ ] PEA-CAE A/B 显示超越基线的稳定改善（T3-01）
- [ ] CDD 实验证明冲突诊断在 HK 监管场景可落地（T3-02）
- [ ] R-01 ~ R-05 已关闭或形成可验证控制措施（Phase 0 + Phase 2）

**候选升级（按依赖顺序，均为原型先行）：**

| 候选 | 依据 | 前置 | 触发方式 |
|---|---|---|---|
| U-02 监管 SPO 三元组统一索引 | RAGulating Compliance | T3-03 分块实验结论 | 小规模三元组黄金集（先 3~5 份高价值文档） |
| U-03 双图检索（结构图 + 语义图） | BifrostRAG；参考 NR-07 MEGRAG 三粒度（三元组/句子/段落）融合 | U-02 三元组可用 | 多跳黄金集 A/B |
| U-05 GraphRAG 全局问题试点 | Microsoft GraphRAG；参考 NR-06 RAPTOR 轻量替代 | 语料稳定 + 成本预算 | 仅限全局/主题问题集，与 DeepResearch 对照 |
| U-04 政策—法规差距图 | PrivComp-KG | 产品决策 | 单一业务场景受控试点（如 CDD 或 AI 产品上线政策） |

**明确不做：**
- ❌ 不因论文指标直接替换现有 Hybrid RAG。
- ❌ 不在评测修正与黄金集扩展完成前用目标阈值当发布标准。
- ❌ 不由 LLM 单独决定合规结论或完成最终审批。
- ❌ 不用 GraphRAG 社区摘要回答精确条款/页码核验问题。

---

## 文件变更总览

| 文件 | 变更 | 对应任务 |
|---|---|---|
| `frontend/package.json` / `package-lock.json` | 升级 next 及传递依赖 | T0-01 |
| `.github/workflows/*.yml` | npm audit 门禁、覆盖率、pip check、锁文件校验 | T0-02 / T2-01 / T2-02 |
| `backend/data/source_manifest.json` | SFC 语料 + 元数据补齐 | T0-03 |
| `backend/data/regulations/` | SFC 官方材料、重采 2 份 HKMA PDF | T0-03 |
| `backend/app/services/kag/graph_builder.py` | REFERENCES/SUPERSEDES 关系接入 | T0-04 |
| `backend/app/services/evaluation/rag_eval.py` | faithfulness 解耦 | T1-01 |
| `backend/app/services/evaluation/generation_faithfulness.py` | **新建**：独立生成忠实度度量 | T1-01 |
| `backend/app/services/evaluation/run_eval.py` | 新指标输出、缓存格式 | T1-01 / T2-04 |
| `backend/app/services/evaluation/release_gate.py` | 三类指标分组门禁 + 版本记录 | T1-03 |
| `backend/data/evaluation/benchmark_questions.json` | 黄金集 53 → 100+ | T1-02 |
| `backend/data/evaluation/gold_packages/` | **新建**：ScenarioBench 风格 gold 决策包 | T1-02 |
| `backend/tests/test_http_api.py` 等 | **新建**：集成/SSE/覆盖率测试 | T2-01 |
| `frontend/src/**/*.test.tsx`、`frontend/e2e/` | **新建**：组件测试 + Playwright E2E | T2-01 |
| `backend/requirements*.txt/.lock`、`backend/pyproject.toml` | 锁定依赖、一致性校验 | T2-02 |
| 根目录 `package.json` / `frontend/next.config.ts` | 工作区归属、显式追踪根目录 | T2-02 |
| `backend/app/core/rate_limit.py` | Redis 分布式限流 | T2-03 |
| `backend/app/main.py` | `/health/live` + `/health/ready` | T2-03 |
| `backend/app/services/agents/builder.py` | Pickle → JSONL 缓存 | T2-04 |
| `backend/app/services/deepresearch/workflow.py` + `escalation_gate.py` | **新建/实验分支**：PEA-CAE | T3-01 |
| `backend/data/evaluation/conflict_cases.json` | **新建**：CDD 冲突用例 | T3-02 |
| `backend/app/services/retrieval/adaptive_chunker.py` | **新建/实验分支**：CTRAG 分块 | T3-03 |
| `docs/evaluation_protocol.md` | 指标三分层、黄金集扩展说明 | T1-01 / T1-02 |
| `docs/experiments/*.md`、`docs/eval-baselines/*` | **新建**：实验报告与基线快照 | 各阶段 |

## 验证总表

| 阶段 | 验证命令 | 预期 |
|---|---|---|
| 基线 | `cd backend && python -m pytest -q` | 151+ passed（新增用例后只增不减） |
| 评测 | `python -m app.services.evaluation.run_eval` | 新指标列；faithfulness ≠ claim_recall |
| 语料 | `python -m app.services.corpus.build_cache` | 无跳过文档；SFC 证据覆盖率达标 |
| 图谱 | `python -m app.services.kag.build_graph_cache` | 替代关系边可见 |
| 前端 | `cd frontend && npm run lint && npm run typecheck && npm run build` | 零错误；无多 lockfile 警告 |
| 安全 | `cd frontend && npm audit --audit-level=high` | 0 high |
| 环境 | `pip check`（项目 .venv 内） | 无冲突 |
| 健康 | `curl /health/live`、`curl /health/ready` | 存活/就绪语义正确，故障时可区分 |

## 风险与开放问题

1. **评测成本上升**：独立 faithfulness 与 claim 验证增加 LLM 调用 → 设缓存、限制完整评测集频率（小黄金集每次发布跑，完整集按计划任务跑）。
2. **LLM-as-judge 偏差**：同一模型生成与评测可能互相放大偏差 → 保留人工抽检集（T1-02 审核记录复用），确定性证据匹配作为降级路径。
3. **SFC 语料采集**：需确认官方来源可获取性与版权/使用条款（T0-03 前置确认）。
4. **实验结论不确定性**：PEA-CAE / CDD / CTRAG 可能无稳定增益 → 原型可归档，不强行上线（Phase 4 决策条件已覆盖）。
5. **Redis 依赖**：T2-03 引入 Redis 作为限流基础设施 → 确认现有 Redis 实例可用性与高可用配置。
6. **需要业务侧输入**：SFC 语料优先级排序、黄金集人工审核人力、U-04 政策差距图是否立项（开放问题，待产品决策）。

---

## 下一步（立即执行）

1. T0-00 记录基线（10 分钟）。
2. T0-01 前端依赖升级 + T1-01 faithfulness 解耦（可并行，最高优先）。
3. T0-03 SFC 语料采集与业务侧确认优先级。
4. 每完成一个任务：运行对应验证命令 + 提交（`git commit` 粒度 = 单个任务）。
