# T0-04 KAG 图谱 REFERENCES/SUPERSEDES 关系激活审计报告

- 日期：2026-08-06
- 任务类型：验证 + 最小补缺（T0-04，风险 R-02 辅助项：已失效/被替代文档应能自动降级或排除并展示替代关系）
- 范围：`backend/app/services/kag/`（graph_builder / graph_retriever / ontology）、`backend/data/source_manifest.json`、`backend/tests/`
- 结论摘要：**REFERENCES 已完全激活（代码 + 数据 + 图谱产物）；SUPERSEDES 代码路径已激活并通过单元测试，但 manifest 无真实数据，图谱产物中为 0 条边；检索路径（graph_retriever.py）未利用 SUPERSEDES/REFERENCES 做降级或替代展示（仅记录现状与建议，未实现）。**

---

## 1. 现状验证

### 1a. graph_builder.py 的 SUPERSEDES / REFERENCES 处理逻辑

`backend/app/services/kag/graph_builder.py`（工作区未提交改动，codex 分支）第 126-139 行：

```python
for referenced_doc_id in doc.references:
    if referenced_doc_id in document_ids:
        store.add_edge(
            doc_node,
            referenced_doc_id,
            relation=RelationType.REFERENCES.value,
        )
for superseded_doc_id in doc.supersedes:
    if superseded_doc_id in document_ids:
        store.add_edge(
            doc_node,
            superseded_doc_id,
            relation=RelationType.SUPERSEDES.value,
        )
```

- **REFERENCES 与 SUPERSEDES 均已接入，且完全对称**：均遍历文档字段、校验目标 doc_id 存在于本次构建的文档集合（`document_ids`，避免悬空边）、以 `doc_node -> 目标文档节点` 的有向边写入图谱。
- 全库搜索确认：`RelationType.REFERENCES`（ontology.py L38）与 `RelationType.SUPERSEDES`（ontology.py L39）均已定义，graph_builder 中两者都有对应消费点。**REFERENCES 并未缺失，无需补代码（任务 2a 已满足）。**

### 1b. source_manifest.json 的 supersedes / references 实际内容

| 字段 | 有值的文档 | 说明 |
|---|---|---|
| `references` | 仅 2 份 SFC 文档 | `sfc_suitability_faq_2016` ↔ `sfc_vatp_guidelines_2023` **互相引用**（各 1 条） |
| `supersedes` | **无任何文档有值** | 全部 20 份文档均为 `[]` |

HKMA 与 PCPD 的全部文档 `supersedes`/`references` 均为空数组。

### 1c. 图谱重建与产物统计

重建命令：`cd /f/MyFintech/backend && python -m app.services.kag.build_graph_cache`

- 输出：`Built dual regulatory graph: 1913 nodes, 3115 edges, 9 SPO triples`
- 产物：`backend/data/graph/regulatory_graph.json`（node-link 格式，`nodes`/`links`）

边统计（relation 分布）：

| Relation | 数量 |
|---|---|
| SUPPORTED_BY | 2183 |
| CONTAINS | 734 |
| RELATED_TO | 113 |
| APPLIES_TO | 54 |
| ISSUED_BY | 20 |
| ASSERTS | 9 |
| **REFERENCES** | **2** |
| **SUPERSEDES** | **0** |

实际 REFERENCES 边（双向）：
- `sfc_suitability_faq_2016 --REFERENCES--> sfc_vatp_guidelines_2023`
- `sfc_vatp_guidelines_2023 --REFERENCES--> sfc_suitability_faq_2016`

结构节点存在性：

| 节点类型 | 数量 |
|---|---|
| RegulatoryDocument | 20 |
| Regulator | 3 |
| Topic | 40 |
| Product | 14 |
| Risk | 6 |
| EvidenceChunk | 1087 |
| **Section** | **702** |
| **Chapter** | **21** |
| **Clause** | **11** |
| RegulatoryTriple | 9 |

> 说明：SUPERSEDES 为 0 条是**数据为空**导致（manifest 无 supersedes 值），非代码缺失——单元测试已证明代码路径可正常生成 SUPERSEDES 边（见 1d/2c）。

### 1d. 已有测试覆盖（无需新建 test_kag_relations.py）

`backend/tests/test_kag_graph_store.py` 已有 `test_graph_builder_activates_reference_and_supersedes_relations`（L138-171）：

- 构造 3 个文档：`doc_current`（references=[doc_reference], supersedes=[doc_old]）、`doc_old`（status="superseded"）、`doc_reference`
- 断言：
  - `store.graph.edges["doc_current", "doc_reference"]["relation"] == "REFERENCES"`
  - `store.graph.edges["doc_current", "doc_old"]["relation"] == "SUPERSEDES"`
- 该测试已完整覆盖任务 2c 要求的最小断言（A --SUPERSEDES--> B、REFERENCES 边），**无需新增重复测试文件**。

### 1e. 检索路径现状（graph_retriever.py）

`backend/app/services/kag/graph_retriever.py`（252 行）**未利用 SUPERSEDES/REFERENCES**：

- 全文无 `SUPERSEDES` / `REFERENCES` / `supersedes` / `references` 引用。
- 证据支撑分 `evidence_support_score` 仅检查 `SUPPORTED_BY` 关系（L208-210）。
- `_doc_neighbors` 收集 topics/obligations/risks 与 relation 串，但 confidence 计算（L212-218）不含对 `status=superseded` 文档的惩罚，也不含替代文档（SUPERSEDES 反向）提示或 REFERENCES 相关文档增强。
- 文档 `status` 字段虽然存在于 schema（corpus.py L19：`active/superseded/archived/unknown`）且 manifest_loader 校验合法值，但**检索打分与排序未消费它**。

---

## 2. 补缺动作

| 项 | 动作 | 状态 |
|---|---|---|
| 2a. REFERENCES 接入 graph_builder | 已存在（与 SUPERSEDES 对称），无需改动 | ✅ 无需补 |
| 2b. 检索路径利用 SUPERSEDES 降级 | 现状：未利用。按任务要求**仅记录现状 + 给出最小建议，不实现** | 📌 记录（见下） |
| 2c. 测试覆盖 | `test_kag_graph_store.py::test_graph_builder_activates_reference_and_supersedes_relations` 已覆盖两条关系断言 | ✅ 已满足 |
| 图谱重建 | `build_graph_cache` 已执行，产物更新（REFERENCES 2 条边已入图） | ✅ 完成 |

### 最小建议（2b，未实现）

供后续 T0 或迭代实现（按改动量从小到大）：

1. **最小改动（降级惩罚）**：在 `graph_retriever.py::retrieve_paths` 的 confidence 计算处（L212-218），对 `doc_attrs.get("status") == "superseded"` 的候选文档施加惩罚（如 confidence 乘 0.3 或减 0.2），并在 `explanation` 中标注"该文档已被替代"。改动点集中、不影响现有 200+ 测试语义。
2. **替代展示**：`_doc_neighbors` 中收集出边 relation 为 `SUPERSEDES` 的目标文档标题，加入 `GraphPathResult`（如新增 `replaced_by: list[str]` 字段），供上层展示"现行版本"。
3. **REFERENCES 增强**：检索结果附带的 relation_chain 已天然携带 REFERENCES 边（_path_details 会收集路径上的 relation），可将其作为"相关文档"提示，无需改检索逻辑，仅需展示端利用。
4. **数据补全**（前置条件，影响实际效果）：为 manifest 补充真实 supersedes 值（如 HKMA SVF 指引系列 2016/2019 文档被 2023/2025 文档替代的关系），需领域确认，本任务不臆造。

---

## 3. 验证结果

`cd /f/MyFintech/backend && python -m pytest tests -q`

```
3 failed, 203 passed, 2 warnings in 24.38s
```

- **KAG 相关测试全绿**：`tests/test_kag_graph_store.py` 6 passed（含关系激活测试 1 passed）。
- 3 个失败全部位于 `tests/test_adaptive_chunker.py`（TestAdaptiveChunker 3 例），属于 codex 分支新增的**未跟踪新文件**（adaptive_chunker 分块器功能），与 KAG 关系无关，且**先于本次任务存在**（本次任务未改动任何源代码，仅重建图谱缓存与写报告）。
- 本次任务未修改任何跟踪文件（git status 前后一致）；`backend/data/graph/regulatory_graph.json` 未被 git 跟踪（gitignore），重建不产生 diff。
- 未执行 git add/commit/push（按要求）。

## 4. 结论

1. **REFERENCES：已激活** ✅ — 代码接入（graph_builder L126-132）+ manifest 数据（SFC 两份文档互引）+ 图谱产物（2 条双向边）+ 测试覆盖。
2. **SUPERSEDES：代码已激活，数据未激活** ⚠️ — 代码路径存在且对称（L133-139），单元测试证明可生成边；但 manifest 全部文档 supersedes 为空，图谱产物中 0 条 SUPERSEDES 边。**实际业务效果依赖数据补全**（建议 4）。
3. **检索路径：未利用降级** 📌 — graph_retriever.py 不消费 status/supersedes/references；"已失效文档自动降级或排除"尚未在检索层实现，需按第 2 节建议落地（最小改动为 confidence 惩罚）。
4. **测试：已覆盖** ✅ — 无需新建 test_kag_relations.py。
5. **无关失败**：test_adaptive_chunker.py 3 例失败为 codex 分支遗留，建议单独跟进（不属于 T0-04 范围）。

## 5. 相关文件

- 已读（未改动）：`backend/app/services/kag/graph_builder.py`、`graph_retriever.py`、`ontology.py`、`build_graph_cache.py`、`backend/data/source_manifest.json`、`backend/tests/test_kag_graph_store.py`
- 已重建（数据产物，gitignore）：`backend/data/graph/regulatory_graph.json`
- 新建：本报告 `docs/eval-baselines/t0-04-kag-relations-audit.md`
