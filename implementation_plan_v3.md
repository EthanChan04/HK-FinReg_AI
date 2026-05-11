# HK-FinReg AI 实施计划 v3

> 适用代码基线：`F:\MyFintech` 当前仓库状态（2026-04-29）
>
> 对应架构文档：[future_architecture_analysis.md](/F:/MyFintech/future_architecture_analysis.md)
>
> 核心原则：先把系统做成**可恢复、可审计、可评估、可治理**，再继续扩展 `MCP`、`GraphRAG` 和跨系统 Agent 协作。

---

## 一、当前仓库落点

当前仓库已经具备以下底座：

- 后端主工作流位于 [svf.py](/F:/MyFintech/backend/app/api/routers/svf.py)，已包含：
  - LangGraph 多节点工作流
  - 反思循环
  - 二次检索
  - 结构化输出校验
  - 三维置信度字段
- 共享工作流能力位于 [workflow_utils.py](/F:/MyFintech/backend/app/api/routers/workflow_utils.py)
- 检索与文档解析位于 [builder.py](/F:/MyFintech/backend/app/services/agents/builder.py) 和 [document_parser.py](/F:/MyFintech/backend/app/services/agents/document_parser.py)
- 配置位于 [config.py](/F:/MyFintech/backend/app/core/config.py)
- Schema 位于 [requests.py](/F:/MyFintech/backend/app/schemas/requests.py)
- 前端流式事件消费位于 [useAgentStream.ts](/F:/MyFintech/frontend/src/hooks/useAgentStream.ts) 和 [ReportPanel.tsx](/F:/MyFintech/frontend/src/components/ReportPanel.tsx)

这意味着 v3 计划不从“新建一套系统”开始，而是在现有 `SVF + shared workflow + streaming UI` 基础上做生产级增强。

---

## 二、v3 总体目标

### Phase 1

把 `SVF` 工作流升级为可持久化、可中断、可恢复、可人工审批的生产级主链路。

### Phase 2

把“证据血缘、法规版本、审计日志”做成一等公民，而不是只停留在最终报告引用格式。

### Phase 3

建立可信评估门禁，避免后续调模型、调检索、接工具时质量漂移不可见。

### Phase 4

引入受治理的 `MCP` 只读工具层，扩展外部事实能力，但不破坏现有控制边界。

### Phase 5

在 `SVF / AML / CDD` 子域试点 `GraphRAG`，只对高复杂度查询启用，不全量替换现有检索栈。

### Phase 6

将 Phase 1-5 的规范下沉到其他业务路由，形成统一的 workflow policy layer。

---

## 三、Phase 1：工作流持久化与 HITL

目标：让 `svf.py` 从“能跑完一条链”升级到“可暂停、可恢复、可接管、可追责”。

### 3.1 后端状态持久化

改动文件：

- [svf.py](/F:/MyFintech/backend/app/api/routers/svf.py)
- [config.py](/F:/MyFintech/backend/app/core/config.py)
- 新建 `backend/app/services/workflow_checkpoint.py`

工程任务：

1. 抽离 `_build_svf_graph()` 和 `_run_svf_graph()`，确保图构建与执行分离。
2. 为 LangGraph 编译时注入 checkpointer，优先使用 `PostgresSaver`。
3. 在配置层新增：
   - `WORKFLOW_DB_URL`
   - `WORKFLOW_CHECKPOINT_ENABLED`
   - `WORKFLOW_THREAD_PREFIX`
4. 在 SVF 请求入口生成稳定 `workflow_run_id`，作为一次审查任务的主键。
5. State 扩展以下字段：
   - `workflow_run_id`
   - `checkpoint_id`
   - `human_review_required`
   - `human_review_status`
   - `human_review_notes`
   - `resume_token`

### 3.2 标准化中断点

改动文件：

- [svf.py](/F:/MyFintech/backend/app/api/routers/svf.py)
- [requests.py](/F:/MyFintech/backend/app/schemas/requests.py)

工程任务：

1. 统一定义三类中断 gate：
   - `low_confidence_gate`
   - `missing_evidence_gate`
   - `manual_approval_gate`
2. 在 `reviewer_node` 后新增 gate 判定逻辑：
   - `overall_confidence < threshold`
   - `cross_validation_passed == False`
   - `rejection_type == insufficient_info` 且达到检索上限
3. 被 gate 拦截时不直接结束，转为：
   - 持久化状态
   - 输出 `action_required` 事件
   - 等待人工恢复

### 3.3 前端人工接管面板

改动文件：

- [useAgentStream.ts](/F:/MyFintech/frontend/src/hooks/useAgentStream.ts)
- [ReportPanel.tsx](/F:/MyFintech/frontend/src/components/ReportPanel.tsx)
- [page.tsx](/F:/MyFintech/frontend/src/app/page.tsx)
- [types/index.ts](/F:/MyFintech/frontend/src/types/index.ts)

工程任务：

1. 新增 SSE 事件类型：
   - `action_required`
   - `checkpoint_saved`
   - `resume_ready`
2. 在前端保存 `workflowRunId` 和 `humanReviewStatus`。
3. 在 `ReportPanel` 或主页面侧栏增加人工接管区：
   - 显示暂停原因
   - 显示关键信心指标
   - 显示待确认法规证据
   - 允许填写 reviewer notes
4. 预留“恢复执行”按钮的前端状态流转。

### 3.4 API 补齐

改动文件：

- [svf.py](/F:/MyFintech/backend/app/api/routers/svf.py)
- 新建 `backend/app/api/routers/review_queue.py`
- [main.py](/F:/MyFintech/backend/app/main.py)

工程任务：

1. 增加人工审查相关接口：
   - `GET /api/v1/review-queue`
   - `GET /api/v1/review-queue/{workflow_run_id}`
   - `POST /api/v1/review-queue/{workflow_run_id}/resume`
   - `POST /api/v1/review-queue/{workflow_run_id}/reject`
2. 统一返回结构，包含：
   - `workflow_run_id`
   - `current_gate`
   - `checkpoint_created_at`
   - `evidence_snapshot`
   - `latest_draft_report`

### 3.5 验收标准

1. 后端重启后，可按 `workflow_run_id` 恢复指定工作流。
2. `SVF` 在低置信度或证据不足场景下，不再直接输出最终报告，而是进入可恢复暂停态。
3. 前端能展示待人工处理状态，并区分“运行中”“暂停中”“已恢复”“已驳回”。

---

## 四、Phase 2：证据血缘、法规版本化与审计日志

目标：把“报告有引用”升级为“整条工作流的证据和法规版本可重建”。

### 4.1 法规治理元数据扩展

改动文件：

- [document_parser.py](/F:/MyFintech/backend/app/services/agents/document_parser.py)
- [builder.py](/F:/MyFintech/backend/app/services/agents/builder.py)
- 新建 `backend/app/services/regulatory_metadata.py`

工程任务：

1. 为 `RegulationChunk` 增加元数据字段：
   - `jurisdiction`
   - `effective_date`
   - `last_updated_date`
   - `document_version`
   - `regulatory_topic`
   - `supersedes`
   - `superseded_by`
2. 在 `hierarchy` 与 `reg_aware` 两种解析路径里统一填充 metadata。
3. 为 PDF / 法规文档建立单独的 source registry，避免不同版本重复混入同一 collection。

### 4.2 引用从文本级升级到工作流级

改动文件：

- [svf.py](/F:/MyFintech/backend/app/api/routers/svf.py)
- [requests.py](/F:/MyFintech/backend/app/schemas/requests.py)
- 新建 `backend/app/services/provenance.py`

工程任务：

1. 为每条引用增加 provenance 结构：
   - `source_number`
   - `chunk_id`
   - `page`
   - `retrieval_round`
   - `retriever_path`
   - `introduced_by_node`
2. 将 `AnalyzerOutput` 中的 citation 解析逻辑升级为保留 provenance。
3. 在 `reviewer_node` 中校验：
   - 引用 chunk 是否存在
   - 引用法规版本是否有效
   - 同一报告内 jurisdiction 是否冲突

### 4.3 审计包导出

改动文件：

- 新建 `backend/app/services/audit_bundle.py`
- [svf.py](/F:/MyFintech/backend/app/api/routers/svf.py)

工程任务：

1. 每份最终报告生成 audit bundle：
   - 输入摘要
   - chunk 证据快照
   - 模型版本
   - prompt 版本
   - reviewer verdict
   - human review notes
   - confidence 历史
2. 先支持 JSON 导出，后续再考虑 PDF / archive 包装。

### 4.4 验收标准

1. 任意最终报告都能反查到对应 chunk、检索轮次和引入节点。
2. 同一法规文档新旧版本能被区分，不再混在一套 metadata 里。
3. 系统可导出单次审查的完整审计包。

---

## 五、Phase 3：评估门禁与回归集

目标：让“调参数、换模型、接工具”都能被质量门禁拦住。

### 5.1 评估目录与数据集结构

新建目录：

- `backend/evals/`
- `backend/evals/datasets/`
- `backend/evals/cases/`
- `backend/evals/reports/`

工程任务：

1. 定义基础数据结构：
   - `query`
   - `expected_regulations`
   - `forbidden_failures`
   - `business_module`
   - `jurisdiction`
2. 先建立 20-50 条 `SVF / AML / CDD` 高价值 gold cases。
3. 从现有集成测试中抽出可复用样例，转成评估集。

### 5.2 自动评估脚本

新建文件：

- `backend/evals/run_ragas_eval.py`
- `backend/evals/run_regtech_rules_eval.py`
- `backend/evals/run_full_eval.py`

工程任务：

1. 集成 `RAGAS` 指标：
   - `context_precision`
   - `faithfulness`
   - `answer_relevance`
2. 追加规则评估：
   - `citation_validity`
   - `version_correctness`
   - `jurisdiction_consistency`
   - `no_unsupported_claims`
3. 输出统一报告：
   - markdown summary
   - json raw result

### 5.3 CI 入口

改动文件：

- 新建 `backend/tests/test_eval_smoke.py`
- 如仓库已有 CI 配置则接入；若没有，先在文档中定义本地执行命令

工程任务：

1. 约定三个执行层级：
   - `smoke`
   - `regression`
   - `release-gate`
2. 在 `README` 后续更新中加入评估命令。
3. 先将 `release-gate` 作为人工执行步骤，再逐步自动化。

### 5.4 验收标准

1. 修改 `retriever`、`prompt` 或模型名后，可以跑出可比较的评估报告。
2. 至少有一套规则可以发现：
   - 无效引用
   - 失效法规
   - 跨 jurisdiction 混用
3. 评估结果可以被纳入发版前检查。

---

## 六、Phase 4：MCP 工具接入层

目标：扩展事实源，但保证工具调用不会绕开治理链。

### 6.1 工具层抽象

新建文件：

- `backend/app/services/tools/mcp_registry.py`
- `backend/app/services/tools/tool_policies.py`
- `backend/app/services/tools/tool_audit.py`

工程任务：

1. 定义工具 capability schema：
   - `regulatory_lookup`
   - `watchlist_check`
   - `kyc_lookup`
   - `document_version_lookup`
2. 每个工具配置：
   - `read_only`
   - `timeout_seconds`
   - `allowed_params`
   - `audit_enabled`

### 6.2 在 SVF 中引入只读工具调用

改动文件：

- [svf.py](/F:/MyFintech/backend/app/api/routers/svf.py)
- [workflow_utils.py](/F:/MyFintech/backend/app/api/routers/workflow_utils.py)

工程任务：

1. 只在 `Extractor` 或 `SubQueryPlanner` 后增加受控工具调用点。
2. 工具返回值不得直接写入最终报告，必须先进入：
   - normalized evidence
   - validator
   - reviewer
3. 工具调用结果写入审计日志与 provenance。

### 6.3 前端事件扩展

改动文件：

- [useAgentStream.ts](/F:/MyFintech/frontend/src/hooks/useAgentStream.ts)
- [types/index.ts](/F:/MyFintech/frontend/src/types/index.ts)

工程任务：

1. 新增工具事件类型：
   - `tool_call_started`
   - `tool_call_finished`
   - `tool_call_failed`
2. 在时间线中展示“外部事实核验”步骤。

### 6.4 验收标准

1. 至少一类只读工具可被 `SVF` 工作流调用并记录审计信息。
2. 工具调用失败时系统能回退，不会直接导致整条链不可用。
3. 工具输出与法规引用在报告中能被区分。

---

## 七、Phase 5：GraphRAG 子域试点

目标：只在高复杂度推理问题上引入图检索增强，不影响普通查询路径稳定性。

### 7.1 图谱抽取离线流程

新建文件：

- `backend/app/services/graphrag/extract_graph.py`
- `backend/app/services/graphrag/graph_schema.py`
- `backend/app/services/graphrag/graph_store.py`

工程任务：

1. 从 `document_parser.py` 输出中抽取初始关系：
   - `references`
   - `amends`
   - `supersedes`
   - `applies_to`
   - `requires`
2. 第一版优先规则抽取，LLM 抽取作为补强步骤。
3. 建立离线构图命令，而不是在在线请求时建图。

### 7.2 查询路由

改动文件：

- [builder.py](/F:/MyFintech/backend/app/services/agents/builder.py)
- [svf.py](/F:/MyFintech/backend/app/api/routers/svf.py)

工程任务：

1. 在 `classify_query_type()` 基础上新增：
   - `impact_analysis`
   - `cross_reference_reasoning`
2. 针对这两类查询启用 graph retrieval 分支。
3. 维持原 Hybrid RAG 为默认路径。

### 7.3 输出整合

工程任务：

1. graph retrieval 输出必须转换为可引用证据块。
2. provenance 中增加 `retriever_path = graph`。
3. Reviewer 新增对图谱证据的格式与来源校验。

### 7.4 验收标准

1. 复杂影响分析查询的召回和解释能力优于纯 Hybrid RAG。
2. 普通条款检索路径性能不受明显影响。
3. 图谱证据仍然能进入统一的引用、审计和评估体系。

---

## 八、Phase 6：规范下沉到多业务模块

目标：将 `SVF` 路线沉淀为可复用的 workflow policy，而不是永远停留在单一路由特化。

### 8.1 共享策略层

改动文件：

- [workflow_utils.py](/F:/MyFintech/backend/app/api/routers/workflow_utils.py)
- 新建 `backend/app/services/workflow_policy.py`

工程任务：

1. 下沉共享策略：
   - review edge policy
   - confidence gate policy
   - checkpoint policy
   - audit logging policy
2. 各路由只声明：
   - 节点
   - 风险阈值
   - 是否需要检索
   - 是否需要人工审批

### 8.2 扩展到其他路由

改动文件：

- [bank_account.py](/F:/MyFintech/backend/app/api/routers/bank_account.py)
- [cross_border.py](/F:/MyFintech/backend/app/api/routers/cross_border.py)
- [sme_lending.py](/F:/MyFintech/backend/app/api/routers/sme_lending.py)

工程任务：

1. 接入统一 checkpoint / review queue / audit bundle。
2. 对非 RAG 路由也接入统一的人工审批与审计逻辑。
3. 前端模块页根据路由能力展示：
   - 是否支持人工接管
   - 是否支持证据追踪
   - 是否支持外部工具核验

### 8.3 验收标准

1. 四条业务路由都能共享相同的运行状态语义和审计语义。
2. 不再需要在各路由中重复维护相似的暂停、恢复、审计逻辑。

---

## 九、测试与验证矩阵

### 单元测试

建议新增：

- `backend/tests/test_checkpointing.py`
- `backend/tests/test_review_queue.py`
- `backend/tests/test_provenance.py`
- `backend/tests/test_regulatory_metadata.py`
- `backend/tests/test_mcp_policies.py`
- `backend/tests/test_graphrag_routing.py`

### 集成测试

重点覆盖：

1. `SVF` 低置信度触发暂停
2. 人工恢复后继续执行
3. 审计包导出完整
4. MCP 工具失败回退
5. GraphRAG 仅在指定查询类型触发

### 回归测试

以 `backend/evals/` 中的 gold set 为准，覆盖：

- AML / CDD
- SVF 牌照义务
- 跨 jurisdiction 风险
- 法规更新后重跑场景

---

## 十、推荐排期

### Sprint 1

- Phase 1.1 - 1.4
- 先打通 `SVF` checkpoint + action_required + review queue

### Sprint 2

- Phase 2 全部
- Phase 3.1 - 3.2

### Sprint 3

- Phase 3.3
- Phase 4 全部

### Sprint 4

- Phase 5 试点
- Phase 6 共享策略层下沉

---

## 十一、v3 的执行顺序

本计划明确采用以下顺序，不建议跳步：

`SVF 持久化/HITL`
→ `证据血缘/法规版本/审计`
→ `评估门禁`
→ `MCP 只读工具`
→ `GraphRAG 子域试点`
→ `多路由统一治理`

原因很简单：如果没有前 3 步，后面的外部工具、图检索和跨模块扩展都会扩大风险面，而不是提高系统可信度。
