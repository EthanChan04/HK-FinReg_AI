# DeepSeek V4 Flash Demo 验收报告

日期：2026-08-10  
范围：受控 Demo；不要求人工审核黄金集；Phase 4 不在本次范围内  
最终状态：**通过（受控 Demo）**

## 1. 验收结论

实现层面的修复、20/20 必需语料缓存和固定 12 条真实 `deepseek-v4-flash` 质量门禁均已通过。黄金集人工审核依本轮范围保持非阻断；本结论仅适用于受控 Demo。

## 2. 可复现证据

| 验证项 | 结果 | 判定 |
|---|---:|---|
| 显式运行时 | `deepseek` / `deepseek-v4-flash` / `https://api.deepseek.com` | 通过 |
| 最小真实 LLM 调用 | 返回 `runtime-ok`；存在 usage 元数据 | 通过，但不替代质量门禁 |
| 后端全量测试（候选分支） | 248 passed, 4 skipped；3 个既有 warning | 通过 |
| 新增模块定向测试/覆盖率 | 28 passed；84.13% | 通过（阈值 70%） |
| PCPD 扫描件 OCR | 6 页；10,947 字符；6 分块 | 通过 |
| 全量语料摄取 | 20/20 成功；0 失败；1,339 分块 | 通过 |
| 前端单元测试 | 15 passed；statements 21.88%，lines 23.28% | 通过 |
| 前端 E2E | 6 passed | 通过 |
| 前端 lint / typecheck / config / production build | 全部通过 | 通过 |
| `npm audit --audit-level=high` | 0 high/critical；2 moderate | Demo 接受风险 |
| 固定 12 条真实 DeepSeek 门禁 | 12/12 回答；12/12 faithfulness；0 API/评测错误 | 通过 |
| 平均 faithfulness | 0.866（阈值 ≥ 0.45） | 通过 |
| 平均 unsupported-claim rate | 0.092（阈值 ≤ 0.10） | 通过 |
| 真实调用统计 | 46,493 tokens；平均 4,370.8 ms | 已记录 |
| 人工审核黄金集 | 108 pending | Demo 范围内非阻断 |

脱敏、无回答正文的运行摘要见 `docs/eval-baselines/deepseek-demo-live-2026-08-10.json`；原始回答工件保留在 Git 忽略目录。

## 3. 已落地控制

- DeepSeek 工厂集中管理 interactive、reasoning、evaluation 三种 profile；缺少密钥立即失败。
- 抓取工件固定记录 provider、model、prompt version、case ID、证据 ID、延迟和 token usage，不记录密钥。
- 仅 HTTP 429 和临时 5xx 最多重试两次；401/403、空输出和畸形输出不重试并失败关闭。
- 质量门禁要求模型精确匹配、12/12 非空、12/12 faithfulness 实测、平均 faithfulness ≥ 0.45、平均 unsupported-claim rate ≤ 0.10。
- 必需语料失败时禁止写入新缓存；官方刷新仅接受监管机构 HTTPS 域名，校验大小、PDF 结构和可读文本后原子替换。
- 图片型官方 PDF 使用本地 OCR，不向第三方上传监管文件。
- CI 提供无人工审批的手动 Demo 验收工作流；原始回答工件 Git 忽略，仅作为短期 CI artifact 保存。

## 4. 语料阻塞解除证据

GitHub Actions 运行 `31400706659` 在可访问 HKMA 的网络中成功执行官方刷新、artifact 上传和完整缓存构建。本地下载 artifact 后进行二次校验，确认只有预期的两个文件：

| 文档 ID | 页数 / 大小 | SHA-256 |
|---|---:|---|
| `hkma_amlcft_surveillance_capability_digitalisation_2024` | 10 页 / 2,426,715 字节 | `575faf16f097ae0382f6f95193f5a1852abfa878707c08042e508020fd6ac8ca` |
| `hkma_svf_amlcft_guideline_2023` | 98 页 / 768,692 字节 | `41dbdb48693faea844871c651248a0b4dd29ab693daa57cc21a54e99f9a40be7` |

两个文件均通过严格 PDF 解析且包含可提取文本。原截断文件仍可从 Git 历史恢复；PCPD 扫描件继续通过本地 OCR 摄取。

## 5. 真实门禁证据

本地使用安全进程环境中的 `DEEPSEEK_API_KEY` 执行：

```bash
python -m app.services.evaluation.live_demo_gate --output-dir ../artifacts/evaluation/live
```

命令退出码为 0，门禁报告显示 `responses=12`、`faithfulness=0.866`、`unsupported=0.092`。运行总计 46,493 tokens，单次延迟范围 2,129～7,653 ms，无鉴权、限流、空响应、畸形响应或选定评测错误。

## 6. 范围声明

这是 Demo 验收，不代表生产合规批准。黄金集人工审核在本轮明确允许延后；Redis 生产部署、长期高阈值、SPO/双图/GraphRAG 等 Phase 4 能力均未纳入或暗示完成。
