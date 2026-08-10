# DeepSeek V4 Flash Demo 验收报告

日期：2026-08-10  
范围：受控 Demo；不要求人工审核黄金集；Phase 4 不在本次范围内  
最终状态：**未通过（外部语料下载阻塞）**

## 1. 验收结论

实现层面的修复已完成，并且真实 `deepseek-v4-flash` 调用成功；但必需语料目前仅 18/20 可解析，完整语料缓存无法通过失败关闭门禁，因此固定 12 条真实回答的质量验收尚未执行。严格口径下不能签发 Demo 通过结论。

## 2. 可复现证据

| 验证项 | 结果 | 判定 |
|---|---:|---|
| 显式运行时 | `deepseek` / `deepseek-v4-flash` / `https://api.deepseek.com` | 通过 |
| 最小真实 LLM 调用 | 返回 `runtime-ok`；存在 usage 元数据 | 通过，但不替代质量门禁 |
| 后端全量测试（候选分支） | 248 passed, 4 skipped；3 个既有 warning | 通过 |
| 新增模块定向测试/覆盖率 | 28 passed；84.13% | 通过（阈值 70%） |
| PCPD 扫描件 OCR | 6 页；10,947 字符；6 分块 | 通过 |
| 全量语料摄取 | 18/20 成功；2 失败；1,094 分块 | **失败** |
| 前端单元测试 | 15 passed；statements 21.88%，lines 23.28% | 通过 |
| 前端 E2E | 6 passed | 通过 |
| 前端 lint / typecheck / config / production build | 全部通过 | 通过 |
| `npm audit --audit-level=high` | 0 high/critical；2 moderate | Demo 接受风险 |
| 固定 12 条真实 DeepSeek 门禁 | 未执行 | **阻断** |
| 人工审核黄金集 | 108 pending | Demo 范围内非阻断 |

后端最终全量回归和完整前端链仍应在候选提交上重新执行；本报告不会把早于最后依赖/CI 调整的结果描述为最终候选验证。

## 3. 已落地控制

- DeepSeek 工厂集中管理 interactive、reasoning、evaluation 三种 profile；缺少密钥立即失败。
- 抓取工件固定记录 provider、model、prompt version、case ID、证据 ID、延迟和 token usage，不记录密钥。
- 仅 HTTP 429 和临时 5xx 最多重试两次；401/403、空输出和畸形输出不重试并失败关闭。
- 质量门禁要求模型精确匹配、12/12 非空、12/12 faithfulness 实测、平均 faithfulness ≥ 0.45、平均 unsupported-claim rate ≤ 0.10。
- 必需语料失败时禁止写入新缓存；官方刷新仅接受监管机构 HTTPS 域名，校验大小、PDF 结构和可读文本后原子替换。
- 图片型官方 PDF 使用本地 OCR，不向第三方上传监管文件。
- CI 提供无人工审批的手动 Demo 验收工作流；原始回答工件 Git 忽略，仅作为短期 CI artifact 保存。

## 4. 阻塞明细

以下两个本地文件均为 1,048,576 字节的截断 PDF，`pypdf` 报 `Cannot find Root object in pdf`：

| 文档 ID | 官方来源 | 本机刷新结果 |
|---|---|---|
| `hkma_amlcft_surveillance_capability_digitalisation_2024` | HKMA `20240207e2a1.pdf` | TLS 握手 60 秒超时 |
| `hkma_svf_amlcft_guideline_2023` | HKMA 2023 SVF AML/CFT Guideline | TLS 握手 60 秒超时 |

刷新脚本在下载成功并通过完整验证前不会覆盖现有文件。PCPD 扫描件已不再是阻塞项。

## 5. 解除阻塞后的自动验收

在可访问 HKMA 的网络或 GitHub Actions 中配置仓库 secret `DEEPSEEK_API_KEY`，手动运行 `DeepSeek Demo Acceptance` 工作流。该工作流依次：

1. 安装锁定依赖并执行定向测试；
2. 从官方 HKMA URL 安全刷新两份截断文件；
3. 构建失败关闭的完整语料缓存；
4. 调用真实 DeepSeek 生成固定 12 条回答；
5. 计算并阻断式检查质量阈值；
6. 上传脱敏响应和门禁报告，保留 14 天。

只有该工作流退出码为 0，且报告显示 20/20 语料、12/12 回答和 12/12 faithfulness 实测后，才可把本报告状态改为“通过”。

## 6. 范围声明

这是 Demo 验收，不代表生产合规批准。黄金集人工审核在本轮明确允许延后；Redis 生产部署、长期高阈值、SPO/双图/GraphRAG 等 Phase 4 能力均未纳入或暗示完成。
