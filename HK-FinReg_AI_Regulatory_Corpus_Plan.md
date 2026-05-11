# HK-FinReg_AI 监管文档下载与 RAG Corpus 构造方案

> 适用项目：`EthanChan04/HK-FinReg_AI`  
> 目标：将当前“单 PDF + SVF 专用 RAG”升级为 **Hong Kong FinReg Corpus**，支持 SVF、AML/CFT、AI/GenAI 治理、交易监控、数据隐私与 DeepResearch 检索。  
> 整理日期：2026-05-08

---

## 1. 总体判断

你现在的 RAG 库只有一份 PDF，且只服务 SVF 模块。下一步应优先补充 **SVF 监管主干文件 + AML/CFT 最新文件 + AI/GenAI 治理文件 + 交易监控/RegTech 文件**。

建议不要一次性堆很多 PDF，而是按 **P0 / P1 / P2** 分批加入：

- **P0：必须加入**，直接支撑 SVF RAG、AML/KYC、AI 合规审查。
- **P1：强烈建议加入**，用于扩展 DeepResearch、RegTech、PEP、交易监控。
- **P2：可选加入**，用于扩展证券、虚拟资产、稳定币、财富管理场景。

最终目标不是“更多 PDF”，而是构造一个带有 metadata、模块标签、监管机构标签、主题标签、时间版本的可路由 RAG corpus。

---

## 2. 推荐目录结构

建议在 `backend/data/regulations/` 下建立如下结构：

```text
backend/data/
├── regulations/
│   ├── hkma_svf/
│   ├── hkma_aml_ai/
│   ├── hkma_aml_recent/
│   ├── hkma_genai_consumer/
│   ├── pcpd_ai_privacy/
│   ├── sfc_aml_vasp_optional/
│   └── stablecoin_optional/
│
├── source_manifest.json
├── indexes/
│   ├── chroma/
│   ├── bm25/
│   └── metadata/
│
└── graph/
    ├── regulatory_graph.json
    └── regulatory_graph.gpickle
```

推荐你先只建立前 5 个目录：

```text
hkma_svf/
hkma_aml_ai/
hkma_aml_recent/
hkma_genai_consumer/
pcpd_ai_privacy/
```

等 SVF + AML + AI 合规主线稳定后，再加入 SFC 与 stablecoin。

---

## 3. P0 必须下载文档

这些文档应当第一批下载并接入 RAG。

| Priority | 文件名建议 | 官方名称 | 监管机构 | 日期 | 下载来源 | 放入目录 | 模块标签 | 用途 |
|---|---|---|---|---|---|---|---|---|
| P0 | `hkma_svf_supervision_guideline_2016.pdf` | Guideline on Supervision of Stored Value Facility Licensees | HKMA | Sep 2016 | https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/Guidelines-on-supervision-of-SVF-licensees_Eng.pdf | `hkma_svf/` | `svf`, `supervision`, `governance`, `risk_management` | SVF 监管主干文件；用于回答 SVF licensee 高层监管要求 |
| P0 | `hkma_svf_practice_note_2025.pdf` | Practice Note on Supervision of Stored Value Facility Licensees | HKMA | Oct 2025 | https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/PN_on_supervision_of_SVF_licensees_eng.pdf | `hkma_svf/` | `svf`, `practice_note`, `supervision`, `controls` | 最新 SVF 实务解释文件；优先级最高 |
| P0 | `hkma_svf_amlcft_guideline_2023.pdf` | Guideline on Anti-Money Laundering and Counter-Financing of Terrorism (For Stored Value Facility Licensees), Revised May 2023 | HKMA | 25 May 2023 / Effective 1 Jun 2023 | https://www.hkma.gov.hk/media/eng/doc/key-information/guidelines-and-circular/2023/20230525e2a1.pdf | `hkma_svf/` | `svf`, `aml`, `cft`, `kyc`, `cdd`, `edd` | 替换或补充旧 AML PDF；SVF AML/KYC/CDD 核心文件 |
| P0 | `hkma_svf_licensing_explanatory_notes_2019.pdf` | Explanatory Notes on Licensing for Stored Value Facilities | HKMA | Jan 2019 | 来源页：https://www.hkma.gov.hk/eng/regulatory-resources/regulatory-guides/by-subject-current/stored-value-facilities-and-retail-payment-systems/ | `hkma_svf/` | `svf`, `licensing`, `application`, `fitness_propriety` | 牌照申请、资格、流程、申请材料分析 |
| P0 | `hkma_tm_ai_thematic_review_2024.pdf` | Thematic Review of Transaction Monitoring Systems and Use of Artificial Intelligence | HKMA | 17 Apr 2024 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20240417-1-EN/20240417-1-EN.pdf | `hkma_aml_ai/` | `aml`, `transaction_monitoring`, `ai`, `regtech` | AML/CFT + AI 交易监控核心文档 |
| P0 | `hkma_tm_systems_insights_2024.pdf` | Insights for Design, Implementation and Optimisation of Transaction Monitoring Systems | HKMA | 17 Apr 2024 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20240417-2-EN/20240417-2-EN.pdf | `hkma_aml_ai/` | `aml`, `transaction_monitoring`, `system_design`, `ai`, `controls` | 交易监控系统设计、优化、治理；非常适合 RAG |
| P0 | `hkma_ai_suspicious_activity_monitoring_2024.pdf` | Use of Artificial Intelligence for Monitoring of Suspicious Activities | HKMA | 9 Sep 2024 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20241122-3-EN/20241122-3-EN.pdf | `hkma_aml_ai/` | `aml`, `ai`, `suspicious_activity`, `monitoring`, `regtech` | 支撑 AI suspicious activity monitoring 方向 |
| P0 | `pcpd_ai_model_personal_data_protection_framework_2024.pdf` | Artificial Intelligence: Model Personal Data Protection Framework | PCPD | 11 Jun 2024 | https://www.pcpd.org.hk/english/resources_centre/publications/files/ai_protection_framework.pdf | `pcpd_ai_privacy/` | `ai_governance`, `privacy`, `personal_data`, `risk_assessment`, `human_oversight` | AI 合规、个人数据保护、人机监督、治理框架 |
| P0 | `hkma_genai_consumer_protection_2024.pdf` | Consumer Protection in respect of Use of Generative Artificial Intelligence | HKMA | 19 Aug 2024 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20241107-1-EN/20241107-1-EN.pdf | `hkma_genai_consumer/` | `genai`, `consumer_protection`, `ai_governance`, `banking` | 银行使用 GenAI 的消费者保护、透明度、治理要求 |

---

## 4. P1 强烈建议下载文档

这些文档用于增强 DeepResearch、PEP、AI adoption、RegTech、风险画像能力。

| Priority | 文件名建议 | 官方名称 | 监管机构 | 日期 | 下载来源 | 放入目录 | 模块标签 | 用途 |
|---|---|---|---|---|---|---|---|---|
| P1 | `hkma_ai_monitoring_mltf_technologies_annex_2024.pdf` | Use of Technologies to improve the Effectiveness and Operational Efficiency of Monitoring for MLTF | HKMA | 9 Sep 2024 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20241122-4-EN/20241122-4-EN.pdf | `hkma_aml_ai/` | `aml`, `mltf`, `ai`, `technology`, `monitoring` | AI suspicious activity monitoring 的附件；实务细节更多 |
| P1 | `hkma_supporting_ai_adoption_amlcft_2025.pdf` | Supporting Artificial Intelligence Adoption in AML/CFT | HKMA | 19 Nov 2025 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20251118-3-EN/20251118-3-EN.pdf | `hkma_aml_ai/` | `aml`, `ai_adoption`, `transaction_monitoring`, `regtech` | 2025 新文件，体现最新 AI in AML/CFT 趋势 |
| P1 | `hkma_pep_risk_based_controls_2025.pdf` | Guidance on risk-based AML/CFT controls for politically exposed persons | HKMA | 21 Nov 2025 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20251118-4-EN/20251118-4-EN.pdf | `hkma_aml_recent/` | `aml`, `cdd`, `edd`, `pep`, `risk_based_approach` | PEP/CDD/EDD 场景非常重要，适合 KYC 模块 |
| P1 | `hkma_high_end_money_laundering_guidance_2025.pdf` | Guidance on combating high-end money laundering | HKMA | 12 Dec 2025 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20251210-1-EN/20251210-1-EN.pdf | `hkma_aml_recent/` | `aml`, `high_end_money_laundering`, `private_banking`, `wealth` | 财富管理、高风险客户、复杂资金来源场景 |
| P1 | `hkma_svf_mltf_risk_assessment_2019.pdf` | Stored Value Facility Sector: Money Laundering and Terrorist Financing Risk Assessment Report | HKMA | 19 Jul 2019 | https://www.hkma.gov.hk/media/eng/doc/key-information/guidelines-and-circular/2019/20190719e1.pdf | `hkma_svf/` | `svf`, `mltf_risk`, `risk_assessment`, `sectoral_risk` | SVF 行业风险画像；适合风险评分与 typology |
| P1 | `hkma_amlcft_surveillance_capability_svf_2024.pdf` | AML/CFT Surveillance Capability Enhancement Project (For SVF Licensees) | HKMA | 7 Feb 2024 | 来源页：https://www.hkma.gov.hk/eng/regulatory-resources/regulatory-guides/circulars/ | `hkma_aml_ai/` | `svf`, `aml`, `surveillance`, `suptech`, `regtech` | SVF-specific AML/CFT surveillance 能力建设 |
| P1 | `pcpd_genai_employee_guidelines_checklist_2025.pdf` | Checklist on Guidelines for the Use of Generative AI by Employees | PCPD | 31 Mar 2025 | https://www.pcpd.org.hk/english/resources_centre/publications/files/guidelines_ai_employees.pdf | `pcpd_ai_privacy/` | `genai`, `employee_use`, `privacy`, `ai_policy`, `data_security` | 企业内部 GenAI 使用政策、敏感数据输入、AI 事件响应 |
| P1 | `pcpd_ethical_ai_guidance_2021.pdf` | Guidance on the Ethical Development and Use of Artificial Intelligence | PCPD | Aug 2021 | https://www.pcpd.org.hk/english/resources_centre/publications/files/guidance_ethical_e.pdf | `pcpd_ai_privacy/` | `ethical_ai`, `ai_governance`, `privacy`, `data_stewardship` | AI 伦理、数据管治、可解释性、责任治理基础文件 |

---

## 5. P2 可选扩展文档

这些文档不建议第一批全部加入，但可以作为后续扩展金融科技、证券、虚拟资产和稳定币监管方向的资料。

| Priority | 文件名建议 | 官方名称 | 监管机构 | 日期 | 下载来源 | 放入目录 | 模块标签 | 用途 |
|---|---|---|---|---|---|---|---|---|
| P2 | `sfc_amlcft_lc_vasp_guideline_2023.pdf` | Guideline on Anti-Money Laundering and Counter-Financing of Terrorism for Licensed Corporations and SFC-licensed VASPs | SFC | 1 Jun 2023 | 来源页：https://apps.sfc.hk/edistributionWeb/api/circular/list-content/circular/aml/doc?lang=EN&refNo=23EC21 | `sfc_aml_vasp_optional/` | `sfc`, `aml`, `vasp`, `licensed_corporation`, `virtual_asset` | 将项目扩展到证券、VASP、虚拟资产 AML |
| P2 | `sfc_amlcft_self_assessment_checklist_2023.pdf` | AML/CFT Self-Assessment Checklist | SFC | 14 Nov 2023 | 来源页：https://apps.sfc.hk/edistributionWeb/api/circular/list-content/circular/aml/doc?lang=EN&refNo=23EC56 | `sfc_aml_vasp_optional/` | `sfc`, `aml`, `checklist`, `self_assessment` | 适合生成 compliance checklist |
| P2 | `sfc_amlcft_webinar_materials_2025.pdf` | AML/CFT Webinar Materials | SFC | 27 Nov 2025 | 来源页：https://apps.sfc.hk/edistributionWeb/api/circular/list-content/circular/aml/doc?lang=EN&refNo=25EC67 | `sfc_aml_vasp_optional/` | `sfc`, `aml`, `training`, `typology`, `supervisory_observation` | 证券 AML 培训、监管观察、typology |
| P2 | `hkma_consumer_protection_alternative_data_2026.pdf` | Consumer Protection in the Use of Alternative Data | HKMA | 26 Mar 2026 | https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20260326-1-EN/20260326-1-EN.pdf | `hkma_genai_consumer/` | `alternative_data`, `consumer_protection`, `ai_credit`, `data_governance` | 如果你扩展到 AI 信贷、替代数据风控，可加入 |
| P2 | `hkma_stablecoin_supervision_guideline_2025.pdf` | Guideline on Supervision of Licensed Stablecoin Issuers | HKMA | 1 Aug 2025 | 来源页：https://www.hkma.gov.hk/eng/news-and-media/press-releases/2025/07/20250729-4/ | `stablecoin_optional/` | `stablecoin`, `licensing`, `supervision`, `digital_assets` | 后续扩展稳定币监管智能体 |
| P2 | `hkma_stablecoin_amlcft_guideline_2025.pdf` | Guideline on AML/CFT for Licensed Stablecoin Issuers | HKMA | 1 Aug 2025 | 来源页：https://www.hkma.gov.hk/eng/key-functions/banking/anti-money-laundering-and-counter-financing-of-terrorism/aml-cft-related-information-for-licensed-stablecoin-issuers/ | `stablecoin_optional/` | `stablecoin`, `aml`, `cft`, `wallet_screening`, `digital_assets` | 后续扩展 stablecoin AML/CFT |
| P2 | `hkma_stablecoin_licensing_explanatory_note_2025.pdf` | Explanatory Note on Licensing for Stablecoin Issuers | HKMA | 1 Aug 2025 | https://www.hkma.gov.hk/media/eng/doc/key-functions/ifc/stablecoin-issuers/Explanatory_Notes_on_Licensing_of_Stablecoin_Issuers_eng.pdf | `stablecoin_optional/` | `stablecoin`, `licensing`, `application`, `digital_assets` | 稳定币发牌申请、准入条件、材料要求 |

---

## 6. 下载优先级建议

### 第一批：最建议你马上下载

```text
1. hkma_svf_supervision_guideline_2016.pdf
2. hkma_svf_practice_note_2025.pdf
3. hkma_svf_amlcft_guideline_2023.pdf
4. hkma_svf_licensing_explanatory_notes_2019.pdf
5. hkma_tm_ai_thematic_review_2024.pdf
6. hkma_tm_systems_insights_2024.pdf
7. hkma_ai_suspicious_activity_monitoring_2024.pdf
8. hkma_ai_monitoring_mltf_technologies_annex_2024.pdf
9. hkma_genai_consumer_protection_2024.pdf
10. pcpd_ai_model_personal_data_protection_framework_2024.pdf
```

这 10 份可以支撑你当前项目的主线：

```text
SVF compliance
+ AML/KYC/CDD
+ transaction monitoring
+ AI in AML/CFT
+ GenAI consumer protection
+ AI privacy governance
```

### 第二批：用于增强研究深度

```text
11. hkma_supporting_ai_adoption_amlcft_2025.pdf
12. hkma_pep_risk_based_controls_2025.pdf
13. hkma_high_end_money_laundering_guidance_2025.pdf
14. hkma_svf_mltf_risk_assessment_2019.pdf
15. pcpd_genai_employee_guidelines_checklist_2025.pdf
16. pcpd_ethical_ai_guidance_2021.pdf
```

这 6 份用于增强：

```text
DeepResearch
+ PEP / EDD
+ high-end ML
+ risk-based approach
+ AI policy
+ ethical AI
```

---

## 7. `source_manifest.json` 设计

建议你不要只把 PDF 放进文件夹，而是建立 `source_manifest.json`。RAG 检索时用 metadata 过滤文档来源。

示例：

```json
[
  {
    "doc_id": "hkma_svf_practice_note_2025",
    "title": "Practice Note on Supervision of Stored Value Facility Licensees",
    "regulator": "HKMA",
    "issue_date": "2025-10",
    "effective_date": null,
    "document_type": "Practice Note",
    "jurisdiction": "Hong Kong",
    "sector": ["SVF", "Payment"],
    "topics": ["supervision", "governance", "risk_management", "internal_controls"],
    "module_tags": ["svf", "licensing", "supervision"],
    "priority": "P0",
    "file_path": "hkma_svf/hkma_svf_practice_note_2025.pdf",
    "source_url": "https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/PN_on_supervision_of_SVF_licensees_eng.pdf",
    "status": "current"
  },
  {
    "doc_id": "hkma_ai_suspicious_activity_monitoring_2024",
    "title": "Use of Artificial Intelligence for Monitoring of Suspicious Activities",
    "regulator": "HKMA",
    "issue_date": "2024-09-09",
    "effective_date": null,
    "document_type": "Circular",
    "jurisdiction": "Hong Kong",
    "sector": ["Banking", "AML/CFT"],
    "topics": ["AI", "AML", "CFT", "suspicious_activity", "transaction_monitoring"],
    "module_tags": ["svf", "aml", "ai_regtech", "deepresearch"],
    "priority": "P0",
    "file_path": "hkma_aml_ai/hkma_ai_suspicious_activity_monitoring_2024.pdf",
    "source_url": "https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20241122-3-EN/20241122-3-EN.pdf",
    "status": "current"
  }
]
```

字段建议：

| 字段 | 说明 |
|---|---|
| `doc_id` | 全局唯一 ID，用于 citation 与 KAG 节点 |
| `title` | 官方标题 |
| `regulator` | HKMA / SFC / PCPD |
| `issue_date` | 发布日期 |
| `effective_date` | 生效日期；没有则设为 `null` |
| `document_type` | Guideline / Practice Note / Circular / Framework / Checklist |
| `jurisdiction` | 一般为 Hong Kong |
| `sector` | SVF / Banking / VASP / Stablecoin / Privacy |
| `topics` | AML, CDD, PEP, AI, GenAI, transaction_monitoring 等 |
| `module_tags` | 对应你的业务模块：svf、bank_account、cross_border、sme_credit、ai_governance |
| `priority` | P0 / P1 / P2 |
| `file_path` | 本地相对路径 |
| `source_url` | 官方下载或来源链接 |
| `status` | current / archive / superseded |

---

## 8. RAG Corpus 构造策略

### 8.1 不要把所有 PDF 混在一个无 metadata 的 Chroma collection

不推荐：

```text
所有 PDF → 直接 split → Chroma collection
```

推荐：

```text
PDF + source_manifest
→ document loader
→ hierarchy-aware parser
→ metadata enrichment
→ chunk-level tags
→ BM25 index
→ vector index
→ source-aware retrieval
```

### 8.2 Chunk metadata 建议

每个 chunk 至少应带：

```json
{
  "chunk_id": "hkma_svf_practice_note_2025_sec_2_1_chunk_001",
  "doc_id": "hkma_svf_practice_note_2025",
  "title": "Practice Note on Supervision of Stored Value Facility Licensees",
  "regulator": "HKMA",
  "issue_date": "2025-10",
  "document_type": "Practice Note",
  "sector": ["SVF", "Payment"],
  "topics": ["supervision", "governance", "controls"],
  "module_tags": ["svf", "supervision"],
  "page": 3,
  "section_title": "2.1 Governance and internal controls",
  "hierarchy_path": "Section 2 > 2.1 Governance and internal controls",
  "source_url": "https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/PN_on_supervision_of_SVF_licensees_eng.pdf"
}
```

### 8.3 检索路由建议

你的系统应按问题类型选择不同文档范围：

| 用户问题 | 推荐过滤条件 |
|---|---|
| SVF 牌照申请 | `module_tags contains svf` + `topics contains licensing` |
| SVF AML/KYC | `module_tags contains svf` + `topics contains aml/cdd/kyc` |
| AI 交易监控 | `topics contains ai` + `topics contains transaction_monitoring` |
| GenAI 消费者保护 | `topics contains genai` + `topics contains consumer_protection` |
| PEP 客户处理 | `topics contains pep` + `topics contains cdd/edd` |
| 数据隐私 / 个人资料输入 LLM | `regulator = PCPD` + `topics contains privacy/ai_governance` |
| DeepResearch 综合报告 | 不先强过滤，先 query decomposition，再对子问题分别过滤 |

### 8.4 建议 Chroma collection 设计

方案 A：一个 collection，靠 metadata filter

```text
collection_name = "hk_finreg_corpus"
```

优点：简单。  
缺点：metadata 过滤要写好，否则容易串库。

方案 B：多个 collection

```text
hkma_svf_collection
hkma_aml_ai_collection
pcpd_ai_privacy_collection
sfc_aml_vasp_collection
```

优点：模块边界清晰。  
缺点：需要 retrieval router 做多 collection 查询。

我建议你现阶段使用 **方案 A**：

```text
一个 Chroma collection + 完整 metadata filter
```

因为你的项目还在作品集/研究原型阶段，方案 A 更容易维护。

---

## 9. 推荐接入流程

### Step 1：下载 PDF

```bash
mkdir -p backend/data/regulations/hkma_svf
mkdir -p backend/data/regulations/hkma_aml_ai
mkdir -p backend/data/regulations/hkma_aml_recent
mkdir -p backend/data/regulations/hkma_genai_consumer
mkdir -p backend/data/regulations/pcpd_ai_privacy
```

### Step 2：按推荐文件名保存

不要保留浏览器下载的乱码文件名。统一命名：

```text
regulator_topic_documenttype_year.pdf
```

例如：

```text
hkma_svf_practice_note_2025.pdf
hkma_svf_amlcft_guideline_2023.pdf
pcpd_ai_model_personal_data_protection_framework_2024.pdf
```

### Step 3：维护 `source_manifest.json`

每下载一份 PDF，就在 manifest 中新增一条记录。

### Step 4：修改当前配置

当前项目中类似单文件路径配置：

```python
PDF_PATH = "../Fintech/AML Guideline for LCs_Eng_30 Sep 2021.pdf"
```

建议替换为：

```python
REG_DOC_DIR = "../data/regulations"
SOURCE_MANIFEST_PATH = "../data/source_manifest.json"
CHROMA_COLLECTION = "hk_finreg_corpus"
```

### Step 5：构建 ingestion pipeline

推荐新增：

```text
backend/app/services/corpus/
├── manifest_loader.py
├── corpus_loader.py
├── metadata_enricher.py
└── index_builder.py
```

### Step 6：重新构建 BM25 + Chroma

每次 corpus 有新增文件时：

```text
load manifest
→ load PDFs
→ parse with hierarchy parser
→ enrich metadata
→ build BM25 index
→ build Chroma vector index
```

---

## 10. 建议的 Retrieval Router

建议在 `retrieval_router.py` 里实现：

```python
def infer_filters(query: str) -> dict:
    text = query.lower()
    filters = {}

    if "svf" in text or "stored value" in text or "儲值" in text or "储值" in text:
        filters["module_tags"] = "svf"

    if "aml" in text or "cft" in text or "money laundering" in text or "洗钱" in text:
        filters["topics"] = ["aml", "cft"]

    if "pep" in text or "politically exposed" in text:
        filters["topics"] = ["pep", "cdd", "edd"]

    if "ai" in text or "artificial intelligence" in text or "genai" in text or "生成式" in text:
        filters["topics"] = ["ai", "genai", "ai_governance"]

    if "personal data" in text or "privacy" in text or "个人资料" in text or "私隐" in text:
        filters["regulator"] = "PCPD"

    return filters
```

后续你可以改成 LLM classifier。

---

## 11. KAG / GraphRAG 的构造建议

如果你要做 KAG，建议先从这些文档抽图谱：

```text
1. hkma_svf_supervision_guideline_2016.pdf
2. hkma_svf_practice_note_2025.pdf
3. hkma_svf_amlcft_guideline_2023.pdf
4. hkma_tm_ai_thematic_review_2024.pdf
5. hkma_genai_consumer_protection_2024.pdf
6. pcpd_ai_model_personal_data_protection_framework_2024.pdf
```

图谱 Schema：

```text
Regulator
Document
Clause
Obligation
Risk
Process
Product
DataCategory
InstitutionType
```

关系：

```text
issued_by(Document, Regulator)
contains(Document, Clause)
requires(Clause, Obligation)
applies_to(Clause, InstitutionType)
related_to(Clause, Risk)
governs(Clause, Process)
protects(Clause, DataCategory)
supported_by(Clause, Chunk)
```

示例：

```json
{
  "source": "hkma_genai_consumer_protection_2024",
  "relation": "contains",
  "target": "governance_and_accountability_principle",
  "evidence_chunk_id": "hkma_genai_consumer_protection_2024_p2_c3"
}
```

---

## 12. DeepResearch 用法建议

DeepResearch 不应默认用于所有问题，只用于复杂研究型问题。

触发条件：

```text
用户问题包含：
- 分析
- 比较
- 生成报告
- 检查清单
- 上线前
- 合规风险
- 多监管机构
- AI 产品落地
```

典型 DeepResearch 问题：

```text
请以香港 SVF licensee 准备上线 AI 交易监控系统为背景，
分析其在 HKMA AML/CFT、AI adoption、交易监控系统设计、个人资料保护方面的主要合规义务，
并生成上线前检查清单。
```

系统应该拆解为：

```text
1. SVF licensee 的 AML/CFT 义务是什么？
2. 交易监控系统设计和优化有哪些监管期望？
3. AI suspicious activity monitoring 有哪些要求或监管观察？
4. 使用 GenAI/AI 时消费者保护和治理要求是什么？
5. 涉及个人数据时 PCPD 有哪些 AI privacy governance 要求？
6. 上线前 checklist 应包含哪些控制点？
```

---

## 13. 最小可行 Corpus

你第一版 corpus 不需要很大。建议先用 10 份文档：

```text
backend/data/regulations/
├── hkma_svf/
│   ├── hkma_svf_supervision_guideline_2016.pdf
│   ├── hkma_svf_practice_note_2025.pdf
│   ├── hkma_svf_amlcft_guideline_2023.pdf
│   └── hkma_svf_licensing_explanatory_notes_2019.pdf
│
├── hkma_aml_ai/
│   ├── hkma_tm_ai_thematic_review_2024.pdf
│   ├── hkma_tm_systems_insights_2024.pdf
│   ├── hkma_ai_suspicious_activity_monitoring_2024.pdf
│   └── hkma_ai_monitoring_mltf_technologies_annex_2024.pdf
│
├── hkma_genai_consumer/
│   └── hkma_genai_consumer_protection_2024.pdf
│
└── pcpd_ai_privacy/
    └── pcpd_ai_model_personal_data_protection_framework_2024.pdf
```

这个 corpus 已经足够支撑你展示：

```text
SVF RAG
+ AML/KYC RAG
+ AI transaction monitoring RAG
+ GenAI governance RAG
+ privacy-aware AI compliance RAG
```

---

## 14. README 中可以这样描述

可以在 README 里新增：

```markdown
## Regulatory Corpus

HK-FinReg AI uses a curated Hong Kong financial regulatory corpus covering:

- HKMA Stored Value Facility supervision and licensing documents
- HKMA AML/CFT Guidelines for SVF licensees
- HKMA thematic review and guidance on AI-enabled transaction monitoring
- HKMA consumer protection circular on Generative AI
- PCPD AI personal data protection framework

Each document is tracked through `source_manifest.json` with metadata including regulator, issue date, document type, sector, topics, module tags, and source URL. The retrieval engine supports metadata-aware hybrid search and reranking.
```

---

## 15. 最终建议

你的下载与构造优先级是：

```text
第一阶段：
SVF + SVF AML/CFT + AI transaction monitoring + GenAI consumer protection + PCPD AI privacy

第二阶段：
PEP + high-end ML + Supporting AI Adoption in AML/CFT + GenAI employee checklist

第三阶段：
SFC AML/VASP + stablecoin + alternative data
```

不要一开始把 SFC、Stablecoin、所有 HKMA circular 全塞进去。  
先让你的 SVF + AI AML/CFT 主线跑通，这样项目会更聚焦，也更适合你未来申请香港银行、FinTech、RegTech 和 AI Product / Business Analyst 岗位。
