"""
多路由共享工作流工具函数集 (P4.1)

设计原则：组合优于继承。各路由按需组合调用，不强制实现接口。

提供的工具函数：
  1. format_sse_event    — 统一 SSE 事件格式化
  2. create_streaming_response — 流式响应包装（预播报 + 异步图执行 + 逐行 token）
  3. build_review_edges  — 反思循环条件边工厂
  4. create_initial_state — State 初始化工厂
  5. build_format_validator — 格式验证中间件工厂
"""
import asyncio
import json
from typing import AsyncGenerator, Callable

from langgraph.graph import StateGraph, END


# ==========================================
# 1. 统一 SSE 事件格式化
# ==========================================

def format_sse_event(event_type: str, data: dict) -> str:
    """统一 SSE 事件格式化

    Args:
        event_type: 事件类型（agent_state / token / done / confidence）
        data: 事件数据字典

    Returns:
        格式化的 SSE 字符串
    """
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==========================================
# 2. 流式响应包装
# ==========================================

async def create_streaming_response(
    graph_fn: Callable,
    safe_input: str,
    agent_steps: list[tuple[str, str]],
    max_revisions: int = 2,
) -> AsyncGenerator[str, None]:
    """流式响应包装：agent 状态播报 + 异步图执行 + 逐行推送

    采用 asyncio.to_thread 避免同步图阻塞 ASGI 事件循环。

    Args:
        graph_fn: 同步图执行函数（如 _run_vb_graph）
        safe_input: 脱敏后的输入文本
        agent_steps: [(agent_name, message), ...] 状态播报列表
        max_revisions: 最大修订次数（传递给图函数内部逻辑）
    """
    # 1. 播报 agent 状态
    for agent_name, msg in agent_steps:
        yield format_sse_event("agent_state", {
            "agent": agent_name, "status": "running", "message": msg
        })

    # 2. 在线程池中执行同步图（避免阻塞 ASGI）
    report = await asyncio.to_thread(graph_fn, safe_input)

    # 3. 逐行推送报告
    from app.services.utils import format_output
    formatted = format_output(report)
    for line in formatted.split("\n"):
        yield format_sse_event("token", {"text": line + "\n"})

    yield format_sse_event("done", {"status": "complete"})


# ==========================================
# 3. 反思循环条件边工厂
# ==========================================

def build_review_edges(
    workflow: StateGraph,
    reviewer_node: str,
    analyzer_node: str,
    planner_node: str | None = None,
    max_revisions: int = 2,
    max_retrieval_rounds: int = 0,
) -> None:
    """反思循环条件边工厂

    根据 reviewer 节点的输出决定下一步路由：
    - end: 审查通过或超过修订上限 → 结束
    - revise: 质量问题 → 回到 analyzer 修订
    - re_retrieve: 信息不足 → 经过 planner 再检索

    Args:
        workflow: StateGraph 实例
        reviewer_node: Reviewer 节点名
        analyzer_node: Analyzer 节点名（修订目标）
        planner_node: SubQueryPlanner 节点名（None 表示无二次检索）
        max_revisions: 最大修订次数
        max_retrieval_rounds: 最大二次检索次数（0 = 不支持）
    """
    def route_fn(state):
        rev_count = state.get("revision_count", 0)
        feedback = state.get("reviewer_feedback", "")

        # 已通过或超过修订上限
        if not feedback or rev_count >= max_revisions:
            return "end"

        # 如果支持二次检索且未达上限
        if max_retrieval_rounds > 0 and planner_node:
            retrieval_round = state.get("retrieval_round", 0)
            if retrieval_round < max_retrieval_rounds:
                # 结构化 rejection_type 优先
                rejection_type = state.get("rejection_type", "")
                if rejection_type == "insufficient_info":
                    return "re_retrieve"
                # Fallback: 关键词匹配
                if rejection_type == "":
                    needs_retrieval_keywords = [
                        "missing", "insufficient", "lack", "not found",
                        "not available", "inadequate",
                        "缺少", "不足", "未找到", "需要更多信息",
                    ]
                    if any(kw in feedback.lower() for kw in needs_retrieval_keywords):
                        return "re_retrieve"

        return "revise"

    edge_map = {
        "end": END,
        "revise": analyzer_node,
    }
    if planner_node and max_retrieval_rounds > 0:
        edge_map["re_retrieve"] = planner_node

    workflow.add_conditional_edges(reviewer_node, route_fn, edge_map)


# ==========================================
# 4. State 初始化工厂
# ==========================================

def create_initial_state(safe_input: str, **extra) -> dict:
    """统一 State 初始化工厂

    各路由追加自定义字段：

    SVF:
        create_initial_state(safe_input,
            retrieval_round=0, format_retry_count=0,
            accumulated_docs=[], sub_queries=[], rejection_type="")

    其他路由:
        create_initial_state(safe_input)

    Args:
        safe_input: 脱敏后的输入文本
        **extra: 各路由的自定义初始字段

    Returns:
        初始 State 字典
    """
    base = {
        "original_input": safe_input,
        "revision_count": 0,
        "final_report": "",
        "reviewer_feedback": "",
    }
    base.update(extra)
    return base


# ==========================================
# 5. 格式验证中间件工厂
# ==========================================

def build_format_validator(
    parse_fn: Callable,
    schema_cls: type,
    max_retries: int = 2,
) -> Callable:
    """格式验证中间件工厂

    Args:
        parse_fn: 将 Markdown 文本解析为字典的函数
        schema_cls: Pydantic 模型类（如 AnalyzerOutput）
        max_retries: 最大重试次数

    Returns:
        format_validator_node 函数（可直接注册到 StateGraph）
    """
    def format_validator_node(state):
        from pydantic import ValidationError

        draft = state.get("draft_report", "")
        retry_count = state.get("format_retry_count", 0)

        try:
            parsed = parse_fn(draft)
            schema_cls(**parsed)
            return {"format_retry_count": retry_count, "validation_errors": ""}
        except ValidationError as exc:
            if retry_count >= max_retries:
                return {
                    "validation_errors": (
                        "FORMAT_VALIDATION_FAILED: report structure is invalid after max retries."
                    ),
                    "format_retry_count": retry_count,
                }
            error_msg = f"FORMAT_VIOLATION ({len(exc.errors())} errors)\n{exc}"
            return {
                "validation_errors": error_msg,
                "format_retry_count": retry_count + 1,
            }
        except Exception as exc:
            if retry_count >= max_retries:
                return {
                    "validation_errors": f"FORMAT_VALIDATION_FAILED: {exc}",
                    "format_retry_count": retry_count,
                }
            return {
                "validation_errors": f"FORMAT_VIOLATION: {exc}",
                "format_retry_count": retry_count + 1,
            }

    return format_validator_node
