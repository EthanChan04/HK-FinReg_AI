"""
澶氳矾鐢卞叡浜伐浣滄祦宸ュ叿鍑芥暟闆?(P4.1)

璁捐鍘熷垯锛氱粍鍚堜紭浜庣户鎵裤€傚悇璺敱鎸夐渶缁勫悎璋冪敤锛屼笉寮哄埗瀹炵幇鎺ュ彛銆?

鎻愪緵鐨勫伐鍏峰嚱鏁帮細
  1. format_sse_event    鈥?缁熶竴 SSE 浜嬩欢鏍煎紡鍖?
  2. create_streaming_response 鈥?娴佸紡鍝嶅簲鍖呰锛堥鎾姤 + 寮傛鍥炬墽琛?+ 閫愯 token锛?
  3. build_review_edges  鈥?鍙嶆€濆惊鐜潯浠惰竟宸ュ巶
  4. create_initial_state 鈥?State 鍒濆鍖栧伐鍘?
  5. build_format_validator 鈥?鏍煎紡楠岃瘉涓棿浠跺伐鍘?
"""
import asyncio
import json
from typing import AsyncGenerator, Callable

from langgraph.graph import StateGraph, END


# ==========================================
# 1. 缁熶竴 SSE 浜嬩欢鏍煎紡鍖?
# ==========================================

def format_sse_event(event_type: str, data: dict) -> str:
    """缁熶竴 SSE 浜嬩欢鏍煎紡鍖?

    Args:
        event_type: 浜嬩欢绫诲瀷锛坅gent_state / token / done / confidence锛?
        data: 浜嬩欢鏁版嵁瀛楀吀

    Returns:
        鏍煎紡鍖栫殑 SSE 瀛楃涓?
    """
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==========================================
# 2. 娴佸紡鍝嶅簲鍖呰
# ==========================================

async def create_streaming_response(
    graph_fn: Callable,
    safe_input: str,
    agent_steps: list[tuple[str, str]],
    max_revisions: int = 2,
) -> AsyncGenerator[str, None]:
    """娴佸紡鍝嶅簲鍖呰锛歛gent 鐘舵€佹挱鎶?+ 寮傛鍥炬墽琛?+ 閫愯鎺ㄩ€?

    閲囩敤 asyncio.to_thread 閬垮厤鍚屾鍥鹃樆濉?ASGI 浜嬩欢寰幆銆?

    Args:
        graph_fn: 鍚屾鍥炬墽琛屽嚱鏁帮紙濡?_run_vb_graph锛?
        safe_input: 鑴辨晱鍚庣殑杈撳叆鏂囨湰
        agent_steps: [(agent_name, message), ...] 鐘舵€佹挱鎶ュ垪琛?
        max_revisions: 鏈€澶т慨璁㈡鏁帮紙浼犻€掔粰鍥惧嚱鏁板唴閮ㄩ€昏緫锛?
    """
    # 1. 鎾姤 agent 鐘舵€?
    for agent_name, msg in agent_steps:
        yield format_sse_event("agent_state", {
            "agent": agent_name, "status": "running", "message": msg
        })

    # 2. 鍦ㄧ嚎绋嬫睜涓墽琛屽悓姝ュ浘锛堥伩鍏嶉樆濉?ASGI锛?
    try:
        report = await asyncio.to_thread(graph_fn, safe_input)
    except Exception as exc:
        yield format_sse_event("error", {
            "status": "error",
            "message": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
        })
        yield format_sse_event("done", {"status": "error"})
        return

    # 3. 閫愯鎺ㄩ€佹姤鍛?
    from app.services.utils import format_output
    formatted = format_output(report)
    for line in formatted.split("\n"):
        yield format_sse_event("token", {"text": line + "\n"})

    yield format_sse_event("done", {"status": "complete"})


# ==========================================
# 3. 鍙嶆€濆惊鐜潯浠惰竟宸ュ巶
# ==========================================

def build_review_edges(
    workflow: StateGraph,
    reviewer_node: str,
    analyzer_node: str,
    planner_node: str | None = None,
    max_revisions: int = 2,
    max_retrieval_rounds: int = 0,
) -> None:
    """鍙嶆€濆惊鐜潯浠惰竟宸ュ巶

    鏍规嵁 reviewer 鑺傜偣鐨勮緭鍑哄喅瀹氫笅涓€姝ヨ矾鐢憋細
    - end: 瀹℃煡閫氳繃鎴栬秴杩囦慨璁笂闄?鈫?缁撴潫
    - revise: 璐ㄩ噺闂 鈫?鍥炲埌 analyzer 淇
    - re_retrieve: 淇℃伅涓嶈冻 鈫?缁忚繃 planner 鍐嶆绱?

    Args:
        workflow: StateGraph 瀹炰緥
        reviewer_node: Reviewer 鑺傜偣鍚?
        analyzer_node: Analyzer 鑺傜偣鍚嶏紙淇鐩爣锛?
        planner_node: SubQueryPlanner 鑺傜偣鍚嶏紙None 琛ㄧず鏃犱簩娆℃绱級
        max_revisions: 鏈€澶т慨璁㈡鏁?
        max_retrieval_rounds: 鏈€澶т簩娆℃绱㈡鏁帮紙0 = 涓嶆敮鎸侊級
    """
    def route_fn(state):
        rev_count = state.get("revision_count", 0)
        feedback = state.get("reviewer_feedback", "")

        # 宸查€氳繃鎴栬秴杩囦慨璁笂闄?
        if not feedback or rev_count >= max_revisions:
            return "end"

        # 濡傛灉鏀寔浜屾妫€绱笖鏈揪涓婇檺
        if max_retrieval_rounds > 0 and planner_node:
            retrieval_round = state.get("retrieval_round", 0)
            if retrieval_round < max_retrieval_rounds:
                # 缁撴瀯鍖?rejection_type 浼樺厛
                rejection_type = state.get("rejection_type", "")
                if rejection_type == "insufficient_info":
                    return "re_retrieve"
                # Fallback: 鍏抽敭璇嶅尮閰?
                if rejection_type == "":
                    needs_retrieval_keywords = [
                        "missing",
                        "insufficient",
                        "lack",
                        "not found",
                        "not available",
                        "inadequate",
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
# 4. State 鍒濆鍖栧伐鍘?
# ==========================================

def create_initial_state(safe_input: str, **extra) -> dict:
    """缁熶竴 State 鍒濆鍖栧伐鍘?

    鍚勮矾鐢辫拷鍔犺嚜瀹氫箟瀛楁锛?

    SVF:
        create_initial_state(safe_input,
            retrieval_round=0, format_retry_count=0,
            accumulated_docs=[], sub_queries=[], rejection_type="")

    鍏朵粬璺敱:
        create_initial_state(safe_input)

    Args:
        safe_input: 鑴辨晱鍚庣殑杈撳叆鏂囨湰
        **extra: 鍚勮矾鐢辩殑鑷畾涔夊垵濮嬪瓧娈?

    Returns:
        鍒濆 State 瀛楀吀
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
# 5. 鏍煎紡楠岃瘉涓棿浠跺伐鍘?
# ==========================================

def build_format_validator(
    parse_fn: Callable,
    schema_cls: type,
    max_retries: int = 2,
) -> Callable:
    """鏍煎紡楠岃瘉涓棿浠跺伐鍘?

    Args:
        parse_fn: 灏?Markdown 鏂囨湰瑙ｆ瀽涓哄瓧鍏哥殑鍑芥暟
        schema_cls: Pydantic 妯″瀷绫伙紙濡?AnalyzerOutput锛?
        max_retries: 鏈€澶ч噸璇曟鏁?

    Returns:
        format_validator_node 鍑芥暟锛堝彲鐩存帴娉ㄥ唽鍒?StateGraph锛?
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

