# T2-03 / T2-04 审计报告：分布式限流与健康检查、Pickle 缓存替换

- 日期：2026-08-05
- 范围：风险 R-05（限流与健康检查语义）、R-06（Pickle 反序列化）
- 性质：审计 + 最小补缺（未重构，未 git add/commit/push，codex 分支工作区改动保持原样）
- 基线测试：`cd backend && python -m pytest tests -q` → **168 passed**（审计前 164，新增 4，无破坏）

---

## 1. T2-03 限流（backend/app/core/rate_limit.py + config.py）

### 1.1 现状与证据

| 核查项 | 状态 | 证据 |
|---|---|---|
| (a) Redis 存储（INCR/EXPIRE 滑动窗口） | ✅ 已实现 | `RedisRateLimitStore._SCRIPT`：ZSET 滑动窗口 + Lua 原子脚本（ZREMRANGEBYSCORE 清理过期、ZCOUNT 分钟计数、ZCARD 小时计数、INCR sequence key 保证 ZADD 成员唯一、EXPIRE 3700s 兜底 TTL）。通过 `redis.asyncio` 的 `eval()` 原子执行，多副本共享计数。 |
| (b) 键优先认证身份/API key/tenant | ✅ 已实现 | `rate_limit_identity()`：authorization → x-api-key → x-tenant-id → x-user-id →（TRUSTED_PROXY_HEADERS 时）x-forwarded-for → client host。凭据类键经 SHA-256 摘要，避免原始凭据进入 Redis 键空间。 |
| (c) 无 Redis 优雅降级 | ⚠️ 部分实现 → **已补缺** | `build_rate_limit_store()` 只在**构造期**捕获异常（覆盖 redis 包缺失等），但 `redis.from_url()` 是惰性连接——Redis 宕机时异常发生在首次 `allow()` 的 `eval()` 调用，原实现会让异常逃逸到请求路径变成 HTTP 500，而非降级。 |

### 1.2 修复动作（最小改动，rate_limit.py）

`RedisRateLimitStore` 增加调用期降级：

- `__init__` 预建 `_fallback = InMemoryRateLimitStore()`、`_degraded = False`；
- `allow()` 中 `eval()` 抛任何异常 → 置 `_degraded = True`、`logger.warning` **一次**（明确提示"计数不再跨副本共享"）、转交内存 store 继续服务；
- 降级是进程级永久性的，避免每次请求刷警告日志；进程重启后重新尝试 Redis。

无 Redis 场景的降级语义符合生产要求：多副本共享计数依赖 Redis；Redis 不可用时降级为进程内计数并告警，不 500。

### 1.3 依赖

- `backend/requirements.txt` **已有** `redis>=5.0.0`（宽松下限风格，与文件其余条目一致），无需修改。
- 本机 hermes venv 原未安装 redis-py，已 `python -m pip install "redis>=5.0.0"` 装入（纯 Python 包 redis 8.1.0，安装成功，未影响环境，测试全绿）。`redis` 在代码中是 `RedisRateLimitStore.__init__` 内惰性 import，未配置 `RATE_LIMIT_STORAGE_URL` 时即使未安装也不会报错。

### 1.4 测试

新增 `backend/tests/test_rate_limit_redis_fallback.py`：注入 `eval()` 必失败的 fake client，断言首次调用降级（返回 True）、`_degraded` 置位、降级后限流仍生效（同键超限返回 False、异键放行）。

---

## 2. T2-03 健康检查（backend/app/main.py + core/health.py）

| 核查项 | 状态 | 证据 |
|---|---|---|
| (a) live 只查进程存活 | ✅ | `/api/v1/health/live` 仅返回进程状态 + langsmith 开关 + 本地 tracker 计数，无外部调用；`/api/v1/health` 别名复用 live。 |
| (b) ready 验证依赖且有降级 | ✅ | `/api/v1/health/ready` 调 `_dependency_checks()`：llm 配置非空（COPILOT/ZHIPU/LONGCAT API key）、corpus 索引文件存在（`CORPUS_INDEX_DIR/corpus_documents.json`）、graph store 文件存在（`GRAPH_STORE_PATH` 或 data/graph/regulatory_graph.json）。**全部为本地文件/配置检查，无外部调用，不存在需超时保护的网络依赖**；依赖不全时返回 503 + `degraded` 状态（`readiness_report`）。无缺口，未改动。 |
| (c) 健康端点排除限流 | ✅ | `RateLimitMiddleware.dispatch` 对 `path.startswith("/api/v1/health")` 直接 `call_next`，三个健康端点均豁免；`/api/v1/metrics` 不在豁免之列且受 API Key 保护。 |

---

## 3. T2-04 Pickle 替换（backend/app/services/corpus/cache.py 等）

### 3.1 现状与证据

| 核查项 | 状态 | 证据 |
|---|---|---|
| 生产路径无 pickle | ✅ | `grep -rn "pickle" backend/app/ backend/tests/ backend/scripts/ --include="*.py"` 零命中。 |
| 生产加载走 JSON 版本化缓存 | ✅ | `builder.py:_load_and_split_corpus()` 调用 `read_corpus_cache(cache_path, manifest_digest=manifest_digest(manifest_path), parser_version="hierarchy-v1")`；未命中才 `load_corpus_documents()` 并 `write_corpus_cache()` 回写。`build_cache.py` 为独立构建入口（CLI），同样走 `write_corpus_cache`。 |
| 缓存键含 manifest digest + parser version | ✅ | `cache.py` 写盘 payload 含 `schema_version=1` + `manifest_digest`（source_manifest.json 的 SHA-256）+ `parser_version`；`read_corpus_cache` 三者任一不匹配即拒读返回 `[]`（触发重建）。写入用 `.tmp` 临时文件 + `replace()` 原子替换。 |
| 损坏/版本不匹配测试 | ⚠️ 部分覆盖 → **已补缺** | 原有 `test_risk_controls.py::test_json_corpus_cache_roundtrip_and_schema_validation` 覆盖 roundtrip + schema_version 篡改；缺 manifest_digest 不匹配与损坏 JSON 用例。 |

### 3.2 修复动作

新增 `backend/tests/test_corpus_cache_safety.py`（3 个用例，对照 `read_corpus_cache` 实际行为）：

1. 篡改 `schema_version`（999）→ 返回 `[]`；
2. `manifest_digest` 不匹配（换 digest 读）→ 返回 `[]`；
3. 损坏 JSON（非法内容）→ 返回 `[]`（`json.JSONDecodeError` 被捕获，警告并拒读）。

---

## 4. 测试结果

```
cd /f/MyFintech/backend && python -m pytest tests -q
168 passed, 1 warning in 21.95s
```

- 基线 164 passed 全部保持；新增 4 个用例（cache safety 3 + redis fallback 1）全部通过。
- 唯一 warning 为既有的 langchain-community 弃用提示，与本次改动无关。

## 5. 改动文件清单（均未提交）

| 文件 | 动作 |
|---|---|
| `backend/app/core/rate_limit.py` | 修改：Redis store 调用期降级（连接失败 → 内存 + 单次告警） |
| `backend/tests/test_corpus_cache_safety.py` | 新增：缓存损坏/版本不匹配安全测试（3 用例） |
| `backend/tests/test_rate_limit_redis_fallback.py` | 新增：Redis 降级行为测试（1 用例） |
| `backend/requirements.txt` | 未改动（已含 `redis>=5.0.0`）；venv 已装 redis 8.1.0 |

## 6. 遗留说明

- Redis 的 Lua 脚本正确性依赖真实 Redis 实例验证（本环境无 Redis 服务器）；代码与 redis-py 8.x `eval` 签名兼容性已通过 inspect 确认（`eval(script, numkeys, *keys_and_args)` 旧式调用仍受支持）。
- `RATE_LIMIT_STORAGE_URL` 生产部署时须在 `.env` 配置，否则默认内存 store（单副本计数，符合开发语义）。
- 健康检查的 `_dependency_checks` 为纯本地检查，无需超时保护；未来若引入外部依赖探测（如 LLM 连通性 ping），需补短超时（如 asyncio.wait_for 2s）与降级。
