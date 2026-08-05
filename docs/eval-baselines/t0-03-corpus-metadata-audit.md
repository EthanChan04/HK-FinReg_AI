# T0-03 语料时效元数据补齐（调查记录）

**日期：** 2026-08-05
**状态：** 调查完成，数据补齐待人工核实（不自动写入未经确认的日期）

## 现状

- 语料清单 20 份文档：HKMA 15 / PCPD 3 / SFC 2
- `source_url` 缺失：**0**（全部完整）
- `status` 缺失：**0**（全部 active）
- `effective_date` 缺失：**17/20**（仅 HKMA AML/CFT Guideline 2023-06-01、SFC FAQ 2016-12-23、SFC VATP 2023-06-01 有值）
- SFC 语料：2 份（suitability FAQ + VATP Guidelines），**已非报告时的"0"**——codex 分支期间已补齐，AI 投顾锚点证据覆盖 87.5%（avg_evidence_regulator_coverage）

## 缺失 effective_date 的文档清单

| doc_id | 监管机构 | 语料内可提取的候选日期 |
|---|---|---|
| hkma_genai_consumer_protection_2024 | HKMA | 2024-08-19（文档含 19 August 2024；另有 2019/2021 引用日期） |
| hkma_tm_ai_thematic_review_2024 | HKMA | 2024-04-17（17 April 2024） |
| hkma_ai_monitoring_mltf_technologies_annex_2024 | HKMA | 待核实 |
| hkma_ai_suspicious_activity_monitoring_2024 | HKMA | 待核实 |
| hkma_amlcft_surveillance_capability_digitalisation_2024 | HKMA | 待核实 |
| hkma_amlcft_surveillance_capability_svf_2024 | HKMA | 待核实 |
| hkma_supporting_ai_adoption_amlcft_2025 | HKMA | 待核实 |
| hkma_tm_systems_insights_2024 | HKMA | 待核实 |
| hkma_high_end_money_laundering_guidance_2025 | HKMA | 待核实 |
| hkma_pep_risk_based_controls_2025 | HKMA | 待核实 |
| hkma_svf_licensing_explanatory_notes_2019 | HKMA | 待核实（2019，具体日期未见） |
| hkma_svf_mltf_risk_assessment_2019 | HKMA | 待核实 |
| hkma_svf_practice_note_2025 | HKMA | 待核实 |
| hkma_svf_supervision_guideline_2016 | HKMA | 待核实 |
| pcpd_ai_model_personal_data_protection_framework_2024 | PCPD | 待核实（文档正文未见日期模式） |
| pcpd_ethical_ai_guidance_2021 | PCPD | 待核实 |
| pcpd_genai_employee_guidelines_checklist_2025 | PCPD | 待核实 |

## 建议处理方式（下一步）

1. **人工核实**：由业务/合规人员对照 HKMA/SFC/PCPD 官网确认每份文档的官方发布日期，补入 `source_manifest.json` 的 `effective_date` 字段。
2. **自动辅助**：对已从语料正文提取到唯一候选日期的文档（如 hkma_tm_ai_thematic_review_2024 → 2024-04-17），可先在 manifest 中标注 `effective_date` 并附 `date_source: "document_body"` 备注，再由人工复核。
3. **验收标准**（R-02）：生产语料 `source_url`/`status`/有效日期完整率 100%，或对不适用的字段记录明确原因；AI 投顾基准证据监管机构覆盖率达到 100%。
4. **注意**：HKMA GenAI 消费者保护通函实际发布于 2024-08-19（正文含 "19 August 2024"），可作为首批人工核实样例。
