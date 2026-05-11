"""
人工审查队列 API 路由 (Phase 1)

提供工作流暂停、人工审查、恢复执行的 REST 接口。

核心语义：
  - interrupt() 暂停后，checkpointer 已保存完整状态
  - resume 使用 Command(resume=...) 从断点真正恢复执行
  - reject 使用 Command(resume=...) 通知节点走驳回路径

接口：
  - GET  /review-queue              — 获取待审查队列
  - GET  /review-queue/{id}         — 获取指定工作流审查详情
  - POST /review-queue/{id}/resume   — 批准并恢复执行
  - POST /review-queue/{id}/reject   — 驳回工作流
"""
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from app.schemas.requests import (
    ReviewQueueItemResponse,
    ReviewRejectRequest,
    ReviewResumeRequest,
)
from app.services.workflow_checkpoint import get_checkpoint_manager

router = APIRouter(prefix="/review-queue", tags=["Review Queue"])


def _ensure_queue_rebuilt():
    """确保内存 review queue 已从持久化存储重建

    首次调用时触发，后续跳过。保证列表端点在后端重启后也能返回完整数据。
    """
    manager = get_checkpoint_manager()
    if not manager._review_queue:
        try:
            from app.api.routers.svf import _build_svf_graph
            checkpointer = manager.get_checkpointer()
            app_graph = _build_svf_graph(checkpointer=checkpointer)
            manager.rebuild_review_queue(app_graph)
        except Exception as e:
            print(f"[ReviewQueue] Queue rebuild failed: {e}")


@router.get("", response_model=List[ReviewQueueItemResponse])
async def list_review_queue():
    """获取全部人工审查队列条目

    内存队列为空时自动从 interrupt registry + checkpointer 重建。
    """
    _ensure_queue_rebuilt()
    manager = get_checkpoint_manager()
    items = manager.get_review_queue()
    return items


@router.get("/pending", response_model=List[ReviewQueueItemResponse])
async def list_pending_reviews():
    """获取待审查条目（status=pending）"""
    _ensure_queue_rebuilt()
    manager = get_checkpoint_manager()
    items = manager.get_pending_reviews()
    return items


@router.get("/{workflow_run_id}", response_model=ReviewQueueItemResponse)
async def get_review_item(workflow_run_id: str):
    """获取指定工作流的审查详情

    如果内存队列中没有（可能因为重启丢失），尝试从 checkpointer 重建。
    """
    _ensure_queue_rebuilt()
    manager = get_checkpoint_manager()
    item = manager.get_review_item(workflow_run_id)
    if item:
        return item

    # 内存队列中不存在，尝试从 checkpointer 状态重建
    try:
        from app.api.routers.svf import _build_svf_graph
        checkpointer = manager.get_checkpointer()
        app_graph = _build_svf_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": workflow_run_id}}
        state_snapshot = app_graph.get_state(config)

        for task in state_snapshot.tasks:
            if task.interrupt:
                interrupt_value = task.interrupt.value
                return ReviewQueueItemResponse(
                    workflow_run_id=workflow_run_id,
                    gate_type=interrupt_value.get("gate_type", "unknown"),
                    checkpoint_created_at=0,
                    evidence_snapshot=interrupt_value.get("evidence_snapshot", {}),
                    latest_draft_report=interrupt_value.get("latest_draft_report", ""),
                    confidence_data=interrupt_value.get("confidence_data", {}),
                    original_input=state_snapshot.values.get("original_input", ""),
                    human_review_status="pending",
                    human_review_notes="",
                    reviewed_at=None,
                    reviewed_by=None,
                )
    except Exception as e:
        print(f"[ReviewQueue] Failed to reconstruct from checkpointer: {e}")

    raise HTTPException(status_code=404, detail=f"Workflow {workflow_run_id} not found in review queue")


@router.post("/{workflow_run_id}/resume")
async def resume_workflow(workflow_run_id: str, req: ReviewResumeRequest):
    """批准并恢复指定工作流执行

    使用 LangGraph Command(resume=...) 从 interrupt() 断点真正恢复执行。
    hitl_gate_node 中 interrupt() 返回 human_decision，节点据此走批准路径。
    """
    manager = get_checkpoint_manager()
    item_dict = manager.get_review_item(workflow_run_id)

    # 如果内存中没有，不阻止恢复（checkpointer 持有真实状态）
    if item_dict and item_dict["human_review_status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Workflow {workflow_run_id} is already {item_dict['human_review_status']}",
        )

    try:
        from langgraph.types import Command
        from app.api.routers.svf import _build_svf_graph

        checkpointer = manager.get_checkpointer()
        app_graph = _build_svf_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": workflow_run_id}}

        # 验证当前状态确实是 interrupt
        state_snapshot = app_graph.get_state(config)
        has_interrupt = any(task.interrupt for task in state_snapshot.tasks)
        if not has_interrupt:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow {workflow_run_id} is not in an interrupt state",
            )

        # 更新审查队列状态
        manager.update_review_status(
            workflow_run_id=workflow_run_id,
            status="approved",
            notes=req.notes,
            reviewed_by=req.reviewed_by or "anonymous",
        )

        # ===== 真正的恢复：Command(resume=...) =====
        # hitl_gate_node 中的 interrupt() 将返回这个 decision
        human_decision = {
            "action": "approve",
            "notes": req.notes,
            "additional_context": req.additional_context,
            "reviewed_by": req.reviewed_by or "anonymous",
        }

        final_state = app_graph.invoke(
            Command(resume=human_decision),
            config=config,
        )

        final_report = final_state.get("final_report", "")
        return {
            "status": "resumed",
            "workflow_run_id": workflow_run_id,
            "message": "工作流已恢复执行并完成",
            "final_report": final_report,
        }

    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "workflow_run_id": workflow_run_id,
            "message": f"恢复执行时出错: {str(e)}",
        }


@router.post("/{workflow_run_id}/reject")
async def reject_workflow(workflow_run_id: str, req: ReviewRejectRequest):
    """驳回工作流

    使用 Command(resume=...) 通知 hitl_gate_node 走驳回路径。
    """
    manager = get_checkpoint_manager()
    item_dict = manager.get_review_item(workflow_run_id)

    if item_dict and item_dict["human_review_status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Workflow {workflow_run_id} is already {item_dict['human_review_status']}",
        )

    try:
        from langgraph.types import Command
        from app.api.routers.svf import _build_svf_graph

        checkpointer = manager.get_checkpointer()
        app_graph = _build_svf_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": workflow_run_id}}

        # 验证当前状态确实是 interrupt
        state_snapshot = app_graph.get_state(config)
        has_interrupt = any(task.interrupt for task in state_snapshot.tasks)
        if not has_interrupt:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow {workflow_run_id} is not in an interrupt state",
            )

        # 更新审查队列状态
        manager.update_review_status(
            workflow_run_id=workflow_run_id,
            status="rejected",
            notes=req.notes,
            reviewed_by=req.reviewed_by or "anonymous",
        )

        # ===== 驳回：Command(resume=...) 走 reject 路径 =====
        human_decision = {
            "action": "reject",
            "notes": req.notes,
            "reviewed_by": req.reviewed_by or "anonymous",
        }

        final_state = app_graph.invoke(
            Command(resume=human_decision),
            config=config,
        )

        final_report = final_state.get("final_report", "")
        return {
            "status": "rejected",
            "workflow_run_id": workflow_run_id,
            "message": "工作流已被驳回",
            "final_report": final_report,
        }

    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "workflow_run_id": workflow_run_id,
            "message": f"驳回时出错: {str(e)}",
        }
