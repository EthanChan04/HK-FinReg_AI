# HK-FinReg AI 系统评估与升级方案报告（修正版）

**日期：** 2026-08-04  
**评估范围：** 代码质量、测试覆盖、架构审查、arXiv 最新研究对标  
**定位：** 研究与原型路线图；**不作为生产改造计划，不据此宣称系统已达高可靠合规水平。**

---

## 一、代码库概况

| 维度 | 数据 |
|---|---|
| 后端 Python 代码量 | ~10,981 行（83 个 .py 文件）|
| 测试文件数 | 35 个测试文件 |
| 测试用例数 | 151 个（全部通过）|
| 前端框架 | Next.js 16.2.11 + React 19.2.4 + TypeScript 5 |
| 技术栈 | FastAPI + LangGraph + ChromaDB + NetworkX + Cohere + Redis |
| Python 版本 | 3.11.15（要求 >=3.11, <3.13）|
| 评测案例数 | 53 个（覆盖 HKMA/SFC/PCPD/跨监管/中英文/冲突/拒答场景）|

### 模块结构

```
backend/app/
├── api/routers/        # 9 个路由模块 (SVF/Copilot/KAG/Research/BankAccount 等)
├── core/               # 配置、安全、监控、限流、健康检查
├── schemas/            # 请求/响应模型 (Copilot/DeepResearch/KAG/Evidence)
├── services/
│   ├── agents/         # Agent 构建器、文档解析、Reranker
│   ├── copilot/        # 意图分类、上下文构建、Guardrails、工具路由、响应写入
│   ├── corpus/         # 语料缓存、清单加载、语料摄取
│   ├── deepresearch/   # 规划器、证据评估、缺口检测、报告生成
│   ├── evaluation/     # 基准评测、RAG 评估、义务映射回归、发布门禁
│   ├── kag/            # 图构建/检索/推理、义务抽取、本体、三元组、A/B 评估
│   └── retrieval/      # 查询分类/规划、检索路由、引用验证、策略记忆
└── main.py
```

---

## 二、测试结果总览

### 后端测试 — ✅ 全部通过

```
151 passed, 1 warning in 21.75s
```

慢速测试 Top 5 均为 SVF 流式传输 keepalive 测试（各约 5 秒），属于超时等待正常设计。

### 前端检查 — ✅ 全部通过

| 检查项 | 结果 |
|---|---|
| `npm run lint` (ESLint) | ✅ 零错误 |
| `npm run typecheck` (tsc) | ✅ 零类型错误 |
| `npm run build` (Next.js) | ✅ 构建成功，4 个静态页面 |

### arXiv 升级方案测试 — ✅ 9/9 通过

项目已包含针对 arXiv 论文方案的预置测试（SPO 三元组规范化与溯源、双图构建验证、双图 A/B 比较、多跳候选扩展边界、质量门禁回归检测），全部通过。

---

## 三、现有质量门禁指标（需修正理解）

### 3.1 当前实际发布门禁

根据 `docs/evaluation_protocol.md:76`，**当前发布门禁使用的是基线阈值，不是目标阈值：**

| 指标 | 当前基线阈值 | 实际评测值 | 说明 |
|---|---|---|---|
| Claim Recall | ≥ 0.45 | **0.472** | 正确声明所需证据被召回的比例 |
| Context Precision | ≥ 0.15 | **0.187** | 检索上下文中有效证据占比 |
| Faithfulness | ≥ 0.45 | **0.472** | ⚠️ **当前实现 `= claim_recall`，并非基于真实生成回答评估** |
| Unsupported Claim Rate | ≤ 0.10 | **0.100** | 缺少证据支持的声明比例 |

### 3.2 ⚠️ faithfulness 指标的关键限制

**代码证据** (`rag_eval.py:164,170`)：

```python
claim_recall = supported_claims / len(claim_tokens) if claim_tokens else 0.0
# ...
faithfulness = claim_recall          # ← 直接赋值，非独立测量
hallucination_rate = 1.0 - faithfulness
```

当前 `faithfulness` **完全等同于 `claim_recall`**——它测量的是"基准中的预期声明是否被检索到的证据 chunk 所支持"，而不是"模型生成的实际回答中，每个声明是否被检索上下文支持"。这意味着：

- 当前系统**没有**对生成器输出的忠实度进行独立评估。
- "Faithfulness 45% → 95%"不能作为严格结论——需要先实现真正的生成忠实度测量，再讨论提升目标。
- 在实现独立生成忠实度度量之前，应仅将 `claim_recall` 和 `context_precision` 作为可靠的检索质量指标。

### 3.3 长期目标（非当前标准）

`evaluation_protocol.md:76` 明确说明，90%/75%/95%/5% 是 **"在人工审核黄金集扩展之后的推荐收紧目标"**，不是当前发布标准：

| 指标 | 当前基线 | 长期目标 | 前置条件 |
|---|---|---|---|
| Claim Recall | 0.45 | 0.90 | 黄金集扩展 + 检索架构改进 |
| Context Precision | 0.15 | 0.75 | 检索去噪 + 语料元数据富化 |
| Faithfulness | 0.45 | 0.95 | **先实现独立的生成忠实度度量** |
| Unsupported Claim Rate | 0.10 | 0.05 | citation_verifier 增强 |

---

## 四、生产前风险（来自现有风险评估文档）

根据 `docs/risk-assessment-2026-08-04.md`，以下风险必须在讨论大规模架构升级之前处理：

| 编号 | 优先级 | 风险 | 当前证据 |
|---|---|---|---|
| R-01 | **P0** | 前端依赖存在高危漏洞 | `npm audit`：6 high、1 low；涉及 Next.js 权限绕过、SSRF 等 |
| R-02 | **P0** | 监管语料与时效元数据不足 | SFC 语料为 0；证据监管机构覆盖率 78%；两份 HKMA PDF 待重采 |
| R-03 | P1 | 评测集与测试层次不足 | 无前端组件测试、E2E、覆盖率阈值；真实模型/SSE 断线测试覆盖有限 |
| R-04 | P1 | 构建环境不可完全复现 | Python 依赖使用宽松下限；共享环境存在版本冲突；多 lockfile 警告 |
| R-05 | P1 | 生产限流与健康检查语义不可靠 | 进程内 IP 限流；`/health` 固定返回 `available` 而不验证依赖 |
| R-06 | P2 | Pickle 缓存不安全反序列化 | `pickle.load()` 直接加载本地 `.pkl` |

> **结论：** 关闭 R-01（前端安全漏洞）和 R-02（SFC 语料缺口）是远比架构升级更紧迫的事项。

---

## 五、arXiv 最新论文发现（2025-2026）

### 🔴 高优先级——可做研究原型

#### NR-01: PEA-CAE — 递进式证据获取与成本感知升级

- **论文：** [From Naive RAG to Deep Agentic Retrieval](https://arxiv.org/abs/2607.24791)（2026-06）
- **来源：** Ontario Power Generation (OPG) 生产环境
- **核心思路：** PEA-CAE (Progressive Evidence Acquisition with Cost-Aware Escalation)——从低成本高精度检索开始，仅在预期证据增益值得时才升级到全文读取。经历了 naive RAG → hybrid RAG → agentic retrieval → deep multi-agent 四阶段演进。
- **当前项目缺口：** DeepResearch 已有问题分解和证据收集，但缺少"成本感知升级"决策机制。
- **建议：** 在 DeepResearch workflow 中以**单场景 A/B 原型**方式加入 escalation gate，不直接替换现有流程。

#### NR-02: CDD — 上下文驱动分解诊断 RAG 知识冲突

- **论文：** [Does RAG Know When Retrieval Is Wrong?](https://arxiv.org/abs/2605.14473)（2026-05）
- **核心思路：** Context-Driven Decomposition (CDD) 在推理时分离"上下文答案"和"先验答案"，诊断 RAG 在知识冲突下的行为。
- **指标说明：** ⚠️ 论文报告的 15% 准确率来自特定 TruthfulQA 误导注入（misconception-injection）最坏情况测试，**不应概括为所有 RAG 场景的准确率**。
- **当前项目缺口：** 项目有 citation_verifier 和 unsupported_claim_rate，但缺少系统化的知识冲突诊断。
- **建议：** 在 citation_verifier 中做 CDD 风格的**冲突诊断实验**，作为诊断工具而非发布标准。

#### NR-03: CTRAG — 自适应分块合规检查

- **论文：** [CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking](https://arxiv.org/abs/2608.02472)（2026-08-03）
- **核心思路：** 自适应分块 + 动态检索配置 + in-context learning，从法规文本中提取控制问题并与公司文档交叉引用。F1=78%，recall=85%。
- **部署范围：** ⚠️ 论文称在**一家四大专业服务机构完成概念验证（POC）**，并非"在四大会计师事务所全面部署"。措辞应准确反映其验证成熟度。
- **当前项目缺口：** 项目更偏重监管 QA 和报告生成，缺少系统化的"控制问题提取→文档交叉引用"闭环。
- **建议：** 参考其**自适应分块策略做 A/B 实验**，不直接替换现有 RAG 分块方案。

### 🟡 中优先级——需适配

#### NR-04: Citation-Enforced RAG for Tax Compliance

- **论文：** [Citation-Enforced RAG for Fiscal Document Intelligence](https://arxiv.org/abs/2603.14170)（2026-03）
- **核心思路：** source-first 摄取策略、页面级溯源、生成时引用强制执行、证据不足时拒答。
- **建议：** 参考其拒答机制增强 Copilot guardrails（在实现独立生成忠实度度量之后）。

#### NR-05: ScenarioBench — 合规评测基准

- **论文：** [ScenarioBench: Trace-Grounded Compliance Evaluation](https://arxiv.org/abs/2509.24212)（2025-09）
- **核心思路：** YAML 场景定义、黄金标准决策包（决策+最小见证轨迹+规范条款集）、条款级证据评估。
- **建议：** 参考其 YAML schema 和 witness_trace 概念增强 benchmark_questions.json。

#### NR-06: GridCodex — 多阶段查询精炼 + RAPTOR

- **论文：** [GridCodex: A RAG-Driven AI Framework for Power Grid Code Reasoning](https://arxiv.org/abs/2508.12682)（2025-08）
- **核心思路：** 多阶段查询精炼 + RAPTOR（递归抽象式摘要）增强检索。报告答案质量提升 26.4%，召回率提升 10 倍以上（⚠️ 来自不同领域数据集，不宜直接作为金融监管场景的预期效果）。
- **建议：** 作为 RAPTOR 原型实验参考，非优先项。

### 🟢 参考价值

#### NR-07: MEGRAG — 多粒度证据图
- **论文：** [MEGRAG: Multi-Granular Evidence Graphs for Answer-Aware Multi-Hop RAG](https://arxiv.org/abs/2608.02195)（2026-08-03）
- 路径结构化的多粒度证据图（三元组→句子→段落），与现有 BifrostRAG 方向互补。

#### NR-08: LLM-Guided Planning for Nuclear Regulatory Documents
- **论文：** [arxiv.org/abs/2606.29399](https://arxiv.org/abs/2606.29399)（2026-06）
- 跨模态核监管文档的多跳推理规划。

#### NR-09: KG for Medical Device Regulatory Compliance
- **论文：** [arxiv.org/abs/2606.28364](https://arxiv.org/abs/2606.28364)（2026-06）
- LLM 驱动的医疗器械合规知识图谱。

#### NR-10: RAG + Cognitive Computing for Regulatory KM
- **论文：** [arxiv.org/abs/2607.24352](https://arxiv.org/abs/2607.24352)（2026-07）
- 将 RAG 作为认知计算架构组件用于监管知识管理。

---

## 六、与现有升级方案的对照

项目已有 `docs/arxiv-github-upgrade-options-2026-08-04.md` 中规划了 5 个升级方向。新发现论文的增量价值（均定位为研究原型级别）：

| 现有方案 U-ID | 新增补充 | 关系 | 实验阶段 |
|---|---|---|---|
| U-01 RAGChecker | **NR-02 CDD** 知识冲突诊断 | 互补——在门禁中增加冲突诊断维度 | 需先修正 faithfulness 定义 |
| U-01 RAGChecker | **NR-05 ScenarioBench** trace-ground 评测 | 互补——增强基准格式 | 黄金集扩展后 |
| U-02 SPO 三元组 | **NR-03 CTRAG** 自适应分块 | 前置增强——三元组抽取前优化分块 | A/B 实验 |
| U-03 双图检索 | **NR-07 MEGRAG** 多粒度证据图 | 互补——三粒度（三元组/句子/段落）融合 | 远期 |
| U-05 GraphRAG | **NR-06 GridCodex** RAPTOR | 替代参考——轻量递归摘要 | 远期 |
| — | **NR-01 PEA-CAE** 成本感知升级 | **新方案**：DeepResearch 升级决策原型 | 单场景 A/B |
| — | **NR-04 Citation-Enforced** 拒答机制 | **新方案**：Copilot 安全增强 | faithfulness 度量独立后 |

---

## 七、建议实施顺序（修正版）

### 第 1 步：修正评测定义（近期，优先）

**目标：** 在引入任何新架构之前，先建立准确的测量体系。

1. 将 `faithfulness` 从 `claim_recall` 中解耦，实现真正的生成忠实度评估（基于模型生成的实际回答，而非基准中的预期声明）。
2. 明确区分三类指标：
   - **检索质量**：claim_recall、context_precision、noise_sensitivity
   - **生成忠实度**：faithfulness（独立测量）、hallucination_rate
   - **引用正确性**：citation_supported_rate、unsupported_claim_rate
3. 关闭 R-01（前端安全漏洞：`npm audit` 0 high），为后续实验提供安全基线。

### 第 2 步：扩充并人工审核黄金集（近期）

**目标：** 从 53 个案例扩展到 100+，按监管机构、语言、任务类型分层。

1. 补齐 R-02 中的 SFC 语料缺口和证据监管机构覆盖率。
2. 新增案例按 HKMA/SFC/PCPD/跨监管分别统计。
3. 按中英文、routine_review/product_launch/regulatory_memo 等任务类型分层。
4. 参考 ScenarioBench 的 YAML schema 和 witness_trace 概念，增加 gold-standard 决策包。
5. 人工审核所有 gold answers 的正确性。

### 第 3 步：PEA-CAE 单场景 A/B 原型（中期实验）

**目标：** 验证成本感知升级在 HK 监管场景中的增量价值。

1. 选择一个已有充足语料的任务类型（如 AI governance review）。
2. 实现最小化 PEA-CAE escalation gate（检索→全文读取两阶段即可）。
3. 在扩展后的黄金集上做 A/B 比较。
4. 仅当实验结果表明稳定改善时才考虑扩展到更多场景。

### 第 4 步：CDD 冲突诊断实验（中期实验）

**目标：** 诊断系统在知识冲突场景下的行为。

1. 构造少量已知的知识冲突测试用例（正确法规 vs 过时法规 / 正确法规 vs 误导性摘要）。
2. 用 CDD 方法分离上下文答案和先验答案。
3. 记录冲突检测率和误报率。
4. 作为诊断工具使用，不直接作为质量门禁。

### 第 5 步：CTRAG 自适应分块实验（中期实验）

**目标：** 评估自适应分块对监管文档检索质量的影响。

1. 对高价值监管文档（如 HKMA AML Guideline）进行自适应分块。
2. 与现有固定分块方案做检索召回率和精度 A/B 比较。
3. 不直接替换现有 RAG 分块，仅作为实验对照。
4. 注意论文中 CTRAG 的验证成熟度（一家四大 POC，非生产部署）。

### 第 6 步：评估实验结果，决定是否进入架构升级（远期）

**仅在所有以下条件满足后才考虑：**
- 评测定义修正完成，faithfulness 独立测量可用
- 黄金集规模显著扩大且人工审核通过
- PEA-CAE A/B 实验显示出超越基线的稳定改善
- CDD 实验证明冲突诊断在 HK 监管场景中可落地
- R-01 至 R-05 生产前风险已关闭或形成可验证控制措施

满足条件后，再讨论双图、SPO、GraphRAG 等较大架构升级。

---

## 八、系统健康检查总结

| 检查项 | 状态 | 详情 |
|---|---|---|
| 后端测试 | ✅ 151/151 通过 | 21.75s |
| 前端 Lint | ✅ 0 错误 | ESLint 9 |
| 前端 TypeScript | ✅ 0 类型错误 | tsc --noEmit |
| 前端构建 | ✅ 4/4 页面 | Next.js 16 |
| arXiv 升级方案测试 | ✅ 9/9 通过 | SPO/双图/质量门禁 |
| `faithfulness` 定义 | ⚠️ 需修正 | 当前 `= claim_recall`，非独立生成忠实度 |
| 前端安全 | 🔴 6 high 漏洞 | R-01 阻断风险 |
| SFC 语料 | 🔴 缺失 | R-02 阻断风险 |
| 弃用/警告 | ⚠️ 1 个 | langchain-community 已弃用 |

---

## 九、总结

**采纳结论：** 本报告定位为**研究与原型路线图**，方向正确但实施顺序需调整，部分研究结论需降级表述。

**最值得采纳的部分：**

1. 先完善评测与质量门禁，再改检索架构。
2. PEA-CAE 作为 DeepResearch 的单场景成本感知升级原型。
3. CDD 作为知识冲突和误导性检索的诊断实验。
4. CTRAG 自适应分块作为 A/B 实验，不直接替换现有 RAG。
5. 扩充人工审核的黄金集，按监管机构、语言、场景分层统计。

**需要修正的表述（本次修订已完成）：**

- ✅ `faithfulness` 当前等于 `claim_recall`，未独立测量生成忠实度——已修正为 "需要先实现独立度量"
- ✅ 90%/75%/95%/5% 是长期目标，不是当前发布标准——已明确区分基线阈值与收紧目标
- ✅ CTRAG 是"在一家四大完成 POC"，不是"已部署验证"——已修正措辞
- ✅ CDD 的 15% 是最坏情况对抗测试结果——已加注 "不应概括为所有 RAG 场景准确率"
- ✅ 增加了生产前风险（R-01 到 R-06）的明确引用
- ✅ 重新排列实施顺序为：评测修正→黄金集扩展→A/B 原型实验→架构升级

**不采纳的声明：**

- ❌ 不将本报告作为生产改造计划。
- ❌ 不据此宣称系统已经达到高可靠合规水平。
- ❌ 不在评测定义修正和黄金集扩展完成之前，用目标阈值作为发布标准。
