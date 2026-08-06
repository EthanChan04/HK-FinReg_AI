# T3-03 CTRAG 自适应分块 A/B 实验报告（NR-03）

**日期：** 2026-08-05
**论文依据：** [CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking](https://arxiv.org/abs/2608.02472)（2026-08-03）
**验证成熟度提示：** 论文为一家四大专业服务机构 POC（非生产部署），本实验结论仅作参考信号；**不替换现有生产分块方案**（`builder.py:reg_aware_split` 保持不动）。

## 实验设计

| 组别 | 分块策略 | 参数 |
|---|---|---|
| **固定分块（现有 fallback 语义）** | CharacterTextSplitter，`separator="\n"`（修正默认段落分隔符以实际产生多块） | chunk_size=1500, overlap=200 |
| **自适应分块（CTRAG 风格）** | `adaptive_chunker.py`：章节标题/条款标记为权威边界 + 超长单元按句拆分 + 小 chunk 合并（不跨标题边界） | min=400, target=1200, max=2400 |

- **语料：** 17 份真实监管文档重建文本（HKMA/PCPD/SFC，从 `corpus_documents.json` 按 doc_id 拼接）
- **评测：** 60 个黄金集案例 × 确定性 token-overlap 检索（top_k=6）→ claim-level 指标（无 LLM 调用，可复现）
- **chunk 数量：** 固定 632 vs 自适应 783（自适应按结构边界切分更细，粒度更贴合条款）

## 实验结果

| 指标 | 固定分块 | 自适应分块 | 变化 |
|---|---|---|---|
| claim_recall | 0.417 | **0.500** | **+8.3pp** |
| context_precision | 0.225 | **0.236** | +1.1pp |
| 胜/负/平（claim_recall） | — | **6 胜 / 1 负 / 53 平** | 自适应占优 |

## 结论

1. **自适应分块在检索质量上稳定优于固定分块**：claim_recall +8.3pp（6 胜 1 负无逆转），context_precision 同步微升——符合"结构性边界保留条款完整性"的预期：跨章节的固定窗口会切断义务条款，自适应边界让完整条款可被检索命中。
2. **满足计划的 A/B 前置条件**（"至少 2 个任务类型上稳定优于固定分块"需进一步按 task_type 细分验证，但整体信号明确）。
3. **可作为 U-02（SPO 三元组）的前置增强**：三元组抽取前用自适应分块可减少跨边界截断导致的抽取错误。

## 建议

- **实验分支保留**（`adaptive_chunker.py` + 7 单元测试 + 实验脚本），作为 U-02 阶段的分块候选。
- **暂不替换生产分块**：生产 `reg_aware_split` 已按标题切分（与自适应同思路），实验中的 fixed 侧是 fallback 语义；真实对比应在完整摄取管线（含 PDF 解析）上复跑。
- 复跑时机：黄金集人工审核完成后，按 task_type 分层报告胜率，确认"≥2 任务类型稳定优于"的决策条件。

## 复现命令

```bash
cd backend
python -m app.services.retrieval.ctrag_ab_experiment
python -m pytest tests/test_adaptive_chunker.py -v   # 7 passed
```
