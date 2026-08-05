# T3-02 CDD 冲突诊断实验报告（NR-02）

**日期：** 2026-08-05
**论文依据：** [Does RAG Know When Retrieval Is Wrong?](https://arxiv.org/abs/2605.14473)（Context-Driven Decomposition）
**定位：** 诊断工具研究原型，**不进入发布门禁**。论文 15% 准确率为 TruthfulQA 误导注入最坏情况结果，不概括为所有 RAG 场景预期。

## 实现

**新增模块：** `backend/app/services/retrieval/cdd_diagnoser.py`
- `diagnose_conflicts(claims, context_chunks, prior_chunks, conflicting_claims=None)`：分离"上下文答案"（仅依据检索证据）与"先验答案"（模型内部知识，用 prior 文档模拟），检测冲突
- **冲突定义：** claim 被先验知识支持但**不被**检索上下文支持（模型可能凭记忆作答而非依据证据）
- **区分性 token 匹配：** 排除 35 个通用停用词（the/from/must/should 等）后计算重叠，避免 "licence from the" 之类的样板文本掩盖 SFC vs HKMA 的真实冲突
- 输入兼容 str / dict / Document 三种 chunk 形态
- 输出 ConflictReport：逐 claim 诊断 + 冲突检测率 + 误报率 + 摘要

**单元测试：** `tests/test_cdd_diagnoser.py` 7 用例（冲突检测、一致判定、双不支持、检测率/误报率、空输入）全通过

## 实验设计

三个场景（claims 用语料原文措辞模拟忠实生成器）：

| 场景 | 上下文（正确法规） | 先验（冲突源） | 预期 |
|---|---|---|---|
| 1. 正确法规 vs 过时法规 | SFC 适当性 FAQ：须评估客户风险态度 | 过时规则：可仅依赖客户投资目标 | claim[1] 冲突 |
| 2. 正确法规 vs 误导性摘要 | PCPD AI 框架：C-level 高管指定 | 误导摘要：经理可自证 AI 治理 | claim[1] 冲突 |
| 3. 无冲突基线 | HKMA GenAI 透明度 | 一致先验 | 无冲突 |

## 实验结果

| 场景 | 检测率 | 误报率 | 冲突判定 |
|---|---|---|---|
| 正确法规 vs 过时法规 | **1.0** | **0.0** | claim[1] 冲突 ✓ |
| 正确法规 vs 误导性摘要 | **1.0** | **0.0** | claim[1] 冲突 ✓ |
| 无冲突基线 | 0.0（无真值） | 0.0 | 全一致 ✓ |

## 结论

1. **CDD 诊断在 HK 监管场景可落地**：正确法规 vs 过时法规/误导摘要的两类典型冲突（监管规则更替、摘要失真）均被稳定检出，无误报。
2. **区分性 token 匹配是必要的**：实验中发现 plain token-overlap 会把 "licence from the" 样板文本误判为支持（0.6 重叠），排除停用词后 SFC/HKMA 差异正确区分。
3. **用途（诊断工具，非门禁）**：
   - 审计生成回答是否"凭记忆作答"而非依据检索证据
   - 作为 citation_verifier 的补充维度（报告引用正确性 ≠ 知识一致性）
   - 检测语料中的过时/冲突文档（prior 侧命中率高但 context 缺失 → 提示语料缺口）

## 建议

- **作为诊断工具保留**（不进 release_gate），供人工抽检与语料审计使用。
- 未来增强：接入真实生成器输出（用 T1-01 的 `split_response_claims` 提取回答声明），并在 golden 集扩展时加入更多冲突用例。
- 已知局限：确定性 token 匹配无法处理同义改写（"require" vs "mandate"），生产级诊断可加 LLM 验证路径（确定性优先、LLM 兜底）。

## 复现命令

```bash
cd backend
python scripts/run_cdd_experiment.py
python -m pytest tests/test_cdd_diagnoser.py -v   # 7 passed
```
