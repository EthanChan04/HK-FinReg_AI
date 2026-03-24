"""
可观测性与追踪模块 (Monitoring & Tracing)

本模块负责两件事：
1. LangSmith 深度追踪 — 接入 LangChain/LangGraph 的原生 Callback 体系，
   使得每一次 LLM 调用、每一个 Agent 节点的进出、每一次 Conditional Edge 判断
   都能在 LangSmith 面板上形成可视化的完整 Trace。
2. 本地性能日志 — 保留原有的 PerformanceTracker 做本地 JSON 审计日志。

=== LangSmith 接入原理 ===

LangSmith 的追踪依赖于 4 个环境变量：
  - LANGCHAIN_TRACING_V2=true          (全局开关)
  - LANGCHAIN_API_KEY=<your_key>       (LangSmith API Key)
  - LANGCHAIN_PROJECT=<project_name>   (在 LangSmith 面板上显示的项目名)
  - LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  (默认)

当这 4 个变量被正确设置后，LangChain 和 LangGraph 内部的每一次
`.invoke()` / `.stream()` 调用都会自动将以下信息上报到 LangSmith：
  - LLM 的输入 Prompt 和输出 Completion
  - Token 使用量
  - 每个 StateGraph Node 的进入/退出时间
  - Conditional Edge 的路由决策 (APPROVED vs REJECTED)
  - 链式调用的完整父子关系 (Parent-Child Trace Tree)

本模块的 `setup_langsmith()` 函数会在 FastAPI 启动时被 `main.py` 调用，
将 config.py 中读取到的 LangSmith 配置注入到 `os.environ` 中，
从而让 LangChain/LangGraph 的内置 Tracer 自动生效。
无需手动传递 Callback 参数。
"""
import os
import time
import json
from datetime import datetime
from collections import defaultdict
import threading

from app.core.config import get_settings

# ==========================================
# Part 1: LangSmith 环境变量注入
# ==========================================
def setup_langsmith():
    """
    在应用启动时调用。

    关键理解：Pydantic BaseSettings 会从 .env 读取变量到 Python 对象属性中，
    但它 **不会** 自动将这些值设置到 os.environ 里。
    而 LangChain/LangGraph 的内置追踪器是直接读取 os.environ 的。

    所以本函数的核心职责就是：
      Settings 对象 (Pydantic) ==注入==> os.environ (操作系统)
    """
    settings = get_settings()

    # 从 Pydantic Settings 中读取用户在 .env 里配置的 LANGCHAIN_* 值，
    # 显式注入到 os.environ，供 LangChain SDK 的底层 Tracer 自动拾取。
    if settings.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    if settings.LANGCHAIN_TRACING_V2:
        os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
    if settings.LANGCHAIN_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    if settings.LANGCHAIN_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

    # 打印确认
    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    project = os.environ.get("LANGCHAIN_PROJECT", "default")

    if tracing_on:
        print(f"✅ LangSmith tracing ENABLED for project: [{project}]")
    else:
        print("⚠️  LangSmith tracing DISABLED (missing API key or toggle off)")


# ==========================================
# Part 2: 本地性能审计日志 (保留原有逻辑)
# ==========================================
LOG_FILE = "./logs/performance_log.json"
STATS_FILE = "./logs/session_stats.json"
_log_lock = threading.Lock()


def ensure_log_directory():
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)


class PerformanceTracker:
    """性能追踪器 — 本地 JSON 审计日志"""
    def __init__(self):
        self.session_stats = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0,
            "queries_by_module": defaultdict(int),
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "processing_times_by_module": defaultdict(list),
            "errors": []
        }
        ensure_log_directory()

    def log_query(self, module_name: str, processing_time: float,
                  input_length: int, status: str = "success", error_msg: str = None):
        with _log_lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": module_name,
                "processing_time_seconds": round(processing_time, 3),
                "input_length": input_length,
                "status": status,
                "error_message": error_msg
            }
            self.session_stats["total_queries"] += 1
            self.session_stats["queries_by_module"][module_name] += 1
            self.session_stats["total_processing_time"] += processing_time
            self.session_stats["avg_processing_time"] = (
                self.session_stats["total_processing_time"] /
                self.session_stats["total_queries"]
            )
            self.session_stats["processing_times_by_module"][module_name].append(processing_time)
            if status == "error":
                self.session_stats["errors"].append({
                    "timestamp": log_entry["timestamp"],
                    "module": module_name,
                    "error": error_msg
                })
            self._append_to_log(log_entry)
            print(f"📊 [Backend Monitor] {module_name} - {processing_time:.2f}s [{status}]")
            return log_entry

    def _append_to_log(self, log_entry: dict):
        try:
            ensure_log_directory()
            logs = []
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            logs.append(log_entry)
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs[-1000:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Log write failed: {e}")

    def get_session_summary(self) -> dict:
        summary = {
            "session_start": self.session_stats["session_start"],
            "total_queries": self.session_stats["total_queries"],
            "queries_by_module": dict(self.session_stats["queries_by_module"]),
            "avg_processing_time": round(self.session_stats["avg_processing_time"], 3),
            "total_errors": len(self.session_stats["errors"]),
            "module_stats": {}
        }
        for module, times in self.session_stats["processing_times_by_module"].items():
            if times:
                summary["module_stats"][module] = {
                    "count": len(times),
                    "avg_time": round(sum(times) / len(times), 3),
                    "min_time": round(min(times), 3),
                    "max_time": round(max(times), 3)
                }
        return summary


# 全局单例
_global_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    return _global_tracker
