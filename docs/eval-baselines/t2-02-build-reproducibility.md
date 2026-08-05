# T2-02 核查报告：构建环境可复现性（R-04）

- 日期：2026-08-05
- 范围：风险 R-04（构建及 Python 环境不可完全复现）
- 性质：核查 + 最小修复（未重构，未 git add/commit/push，codex 分支工作区改动保持原样）
- 基线测试：`cd backend && .venv/Scripts/python -m pytest tests -q` → **168 passed**（28.4s，exit 0；上下文基线 164 为审计前数据，期间其他任务新增 4 个测试）

---

## 1. 多 lockfile / 工作区根推断警告（根 package.json 归属）

### 1.1 现状

| 核查项 | 状态 | 证据 |
|---|---|---|
| 根 `package.json`（131B，git 未跟踪） | 工具包装器 | 内容仅 3 个依赖：`@raindrop-ai/pi-agent@^0.0.4`、`pi-diff-review@^0.1.15`、`pi-web-access@^0.10.7`；无 `name`/`private`/`workspaces` 字段 |
| 根 `package-lock.json`（86KB，git 未跟踪）+ 根 `node_modules` | 存在 | `ls node_modules` 含 `@raindrop-ai`、`@anthropic-ai`、`@aws-sdk` 等工具链依赖 |
| 归属判断 | **不属于项目运行时依赖** | 根包与 `frontend/`（next/react 应用）无任何依赖关系，是独立的 AI 代码评审工具链（pi-* 系列 CLI） |
| `npm ls --depth=0`（frontend） | ✅ 无多 lockfile 警告 | exit 0；npm 在无 `workspaces` 配置时不扫描子目录 lockfile，根与 frontend 互为独立包。仅 2 个 `extraneous`（`@emnapi/runtime@1.11.3`、`@img/sharp-wasm32@0.35.0`，sharp 可选依赖安装残留，`npm ci` 干净安装不会产生） |
| `npm run build`（frontend） | ✅ 无工作区根警告 | 完整 29 行构建日志，grep `warn|workspace|lockfile|infer|trace` 仅命中 "Collecting build traces ..."；"Compiled successfully"，exit 0 |

### 1.2 修复动作

无文件改动（任务约束：不删除根 `package.json`，可能被其他工具使用）。**保留现状即正确**：根包是独立的工具依赖包，无 `workspaces` 配置，npm 不会把两套 lockfile 当作冲突。

### 1.3 剩余问题 / 建议（记录，不执行）

- 若长期保留工具链，风险文档建议二选一：
  - (a) 正式配置 workspace：根 `package.json` 加 `"private": true` + `"workspaces": ["frontend"]`，并把两套 lockfile 合并为一套（`npm install` 语义变化，工具锁文件需迁移，且会让根 `npm install` 牵连 frontend 依赖）；
  - (b) 将工具依赖移出仓库（独立目录或全局安装）。
- 当前两 lockfile 并存**不产生任何 npm/构建警告**（已在 1.1 实证），该项 R-04 验收标准"Next.js 构建不再报告多 lockfile 工作区根警告"已达成。
- 可选清理：`cd frontend && npm prune` 移除 2 个 extraneous 包（仅动 node_modules，不入库）。

## 2. Python 依赖锁定（requirements.txt / pyproject.toml / requirements.lock）

### 2.1 现状

| 核查项 | 状态 | 证据 |
|---|---|---|
| `requirements.txt`（git 已跟踪） | 23 个直接依赖，全部 `>=` 宽松下限 | 本次工作区已补 `cryptography`、`redis`（git diff 确认） |
| `backend/pyproject.toml`（git 未跟踪，新增） | 21 项 dependencies + `[dependency-groups] test` | 无内置 lock 机制配置（无 `[tool.uv]`/`uv.lock`）；lock 生成机制由 README 文档化：`uv pip compile backend/pyproject.toml --group test --output-file backend/requirements.lock`（README 第 209 行） |
| `backend/requirements.lock`（git 未跟踪，新增） | uv pip compile 生成，126 个包全部 `==` 锁定，带来源注释 | 文件头注释：`uv pip compile pyproject.toml --group test --output-file requirements.lock` |
| 包名集合一致性（脚本对比） | ⚠️ 一处分歧 → **已修复** | `requirements.txt` 23 个直接依赖全部出现在 lock（"txt 有而 lock 完全没有的包: 无"）；唯一分歧：txt 声明 `psycopg[binary]>=3.1.0`，而 pyproject 未声明 → lock 中 `psycopg==3.3.4` 仅为 `langgraph-checkpoint-postgres` 的传递依赖，且**缺 `psycopg-binary`** |
| `pip check`（项目 .venv） | ✅ 通过 | `env -u PYTHONPATH .venv/Scripts/python.exe -m pip check` → "No broken requirements found."（exit 0） |
| venv 与 lock 版本一致性 | ✅ 一致 | 剥离 PYTHONPATH 后 `pip list`：fastapi 0.141.1、langchain 1.3.14、langchain-core 1.5.3、langchain-openai 1.4.1、openai 2.52.0、chromadb 1.5.9、uvicorn 0.52.1、redis 8.1.0、pytest 9.1.1 — 与 lock 的 `==` 引脚逐项相同 |

### 2.2 关键发现：pip check 冲突来自宿主环境 PYTHONPATH，非项目 venv

- 直接运行 `.venv/Scripts/python.exe -m pip check` 会报：
  - `hermes-agent 0.18.2 has requirement openai==2.24.0, but you have openai 2.44.0`
  - `insure-rag 0.1.0 requires docling / faiss-cpu / sentence-transformers, which are not installed`
- 根因：Hermes 桌面应用向 shell 注入 `PYTHONPATH=C:\Users\27719\AppData\Local\hermes\hermes-agent;C:\Users\27719\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages`，导致**任何** python 解释器的 `sys.path` 都混入 Hermes 环境（实测 `sys.path` 前两条即这两个路径）。风险文档 R-04 中"pip check 显示 hermes-agent 与 openai 版本冲突"的证据即由此产生。
- 剥离 PYTHONPATH 后 venv 完全干净，`import hermes_agent` 在 venv 中也不存在（ModuleNotFoundError）。**项目 .venv 本身无污染，无需重建**；`hermes-agent`/`insure-rag` 均来自 Hermes 环境。

### 2.3 修复动作（最小改动，共 1 行 + lock 重新生成）

1. `backend/pyproject.toml`：dependencies 增加一行 `"psycopg[binary]>=3.1.0"`（与 requirements.txt 对齐，唯一分歧点）。
2. 按 README 文档化的机制重新生成 lock：`cd backend && uv pip compile pyproject.toml --group test --output-file requirements.lock`。
   - diff（旧→新）仅 5 行、**零版本漂移**：
     - `psycopg==3.3.4` 来源注释更新为 `via hk-finreg-ai-backend (pyproject.toml)` + `langgraph-checkpoint-postgres`
     - 新增 `psycopg-binary==3.3.4`（`# via psycopg`）
3. 环境对齐：`.venv` 补装 `psycopg-binary==3.3.4`（与 lock 引脚一致），`pip check` 复跑通过，`psycopg` + `psycopg_binary` 导入验证 OK。
   - 注：venv 在 .gitignore 内（第 38-39 行），不入库。

### 2.4 剩余问题

- 无文件级剩余问题：requirements.txt / pyproject.toml / requirements.lock 三方现已一致（23 个直接依赖全部在 lock 中锁定，含 `psycopg[binary]` 完整 extra）。
- 后续依赖变更必须用同一命令重新生成 lock（README 第 209 行已文档化）；pyproject.toml 本身未内置 lock 机制配置，可在未来加 `[tool.uv]` 或改 `uv.lock`，当前不做（保持最小改动）。

## 3. CI 一致性（release-gates.yml）

### 3.1 现状与证据

| 核查项 | 状态 | 证据 |
|---|---|---|
| CI 后端安装命令 | .venv + pip check + requirements.txt | 两个 backend job 均为：`python -m venv .venv` → `.venv/bin/python -m pip install --upgrade pip` → `.venv/bin/python -m pip install -r ../requirements.txt` → `.venv/bin/python -m pip check`（git diff 确认本次工作区改动） |
| CI 实际安装的依赖集 | **= backend/requirements.lock（锁定集）** | working-directory=`backend`，`../requirements.txt` = 仓库根 `requirements.txt`（**git 已跟踪**，79B）→ 内容为 `-r backend/requirements.lock` 包装器 → pip 按"相对路径相对于所属 requirements 文件"解析 → `F:\MyFintech\backend\requirements.lock` |
| 本地安装命令 | 与 CI 同源 | README 第 203-206 行：`python -m venv .venv` → `.venv/Scripts/python -m pip install --upgrade pip` → `.venv/Scripts/python -m pip install -r requirements.txt` → `pip check`（仓库根执行，同一包装器） |
| 本地实证 | ✅ 一致 | `env -u PYTHONPATH .venv/Scripts/python.exe -m pip install --dry-run -r requirements.txt` → 全部解析为 `F:\MyFintech\backend\requirements.lock` 的 126 个 `==` 锁定包，全部 "already satisfied"，exit 0 |
| CI pip check | ✅ 必过 | GitHub runner 无 Hermes PYTHONPATH 污染；安装集与本地 venv 相同 |

### 3.2 结论

CI 与本地（README 命令）安装**同一锁定依赖集合**，R-04 验收"本地、CI 与部署环境解析出的核心依赖版本一致"在安装层面成立。CI 链路经过一层"根 requirements.txt 包装器"间接引用 lock，已由 dry-run 实证有效，保持最小改动不重写为直引 `requirements.lock`（可选优化，记录即可）。

## 4. next.config.ts（工作区根显式声明）

### 4.1 现状

| 核查项 | 状态 | 证据 |
|---|---|---|
| `outputFileTracingRoot` | ✅ 已显式设置 | `outputFileTracingRoot: process.cwd()`（git diff 确认为本次工作区改动，原文件仅占位注释 `/* config options here */`） |
| `turbopack.root` | ✅ 已显式设置 | `turbopack: { root: process.cwd() }` |
| 构建实证 | ✅ 无推断警告 | `npm run build`（脚本为 `next build --webpack`）exit 0，日志无 "workspace root inferred"/lockfile 警告 |

### 4.2 说明

- build 脚本使用 `--webpack`，故 `turbopack.root` 在 `npm run build` 路径不生效；`outputFileTracingRoot` 对 webpack 文件追踪生效。两者均为显式声明，消除了 R-04 描述的"根目录被推断为工作区根"警告，该配置保留不动。
- 取值 `process.cwd()` = 构建时所在目录（frontend/），对单应用布局正确（文件追踪只覆盖 frontend，不会把 backend 语料/缓存包进产物）。

## 5. 汇总与基线

| 核查项 | 结论 |
|---|---|
| 1. 多 lockfile 警告 | ✅ 无警告（npm 独立包语义 + next.config 显式根）；根 package.json 为工具包装器，保留并记录 workspace 配置建议 |
| 2. Python 依赖锁定 | ✅ 已修复：pyproject +1 行 `psycopg[binary]`，lock 重新生成（+psycopg-binary==3.3.4，零版本漂移）；`pip check` 通过（剥离 Hermes PYTHONPATH 后 venv 干净） |
| 3. CI 一致性 | ✅ 一致：CI 与本地均经根包装器安装 `backend/requirements.lock` 锁定集（dry-run 实证） |
| 4. next.config.ts | ✅ 已达标：显式 outputFileTracingRoot + turbopack.root，构建无警告 |
| 基线测试 | `cd backend && .venv/Scripts/python -m pytest tests -q` → **168 passed**（28.4s，exit 0；2 个 Starlette 弃用警告，与本次改动无关） |

## 6. 文件改动清单

- `backend/pyproject.toml`：+1 行 `"psycopg[binary]>=3.1.0"`（未跟踪新增文件）
- `backend/requirements.lock`：重新生成（+`psycopg-binary==3.3.4`，psycopg 注释更新；未跟踪新增文件）
- `.venv`（gitignore 内）：补装 `psycopg-binary==3.3.4`
- 新增本报告 `docs/eval-baselines/t2-02-build-reproducibility.md`
- 未 git add/commit/push；其余工作区改动未触碰
