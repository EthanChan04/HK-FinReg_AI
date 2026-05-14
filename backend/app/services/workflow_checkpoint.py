"""
工作流持久化与 Checkpoint 管理 (Phase 1)

职责：
  1. 管理 PostgresSaver / MemorySaver 的初始化与生命周期
  2. 提供统一的 checkpoint 创建/恢复/查询接口
  3. 维护 review queue（人工审查队列），支持持久化重建

设计原则：
  - 优先使用 PostgresSaver，不可用时 fallback 到 MemorySaver
  - Checkpoint 与 thread_id (workflow_run_id) 绑定
  - interrupt() 暂停后状态由 checkpointer 持久化
  - interrupt registry（JSON 文件）记录哪些 thread 处于中断态，重启后可重建完整列表
"""
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from app.core.config import get_settings


# ==========================================
# Interrupt Registry — 持久化的中断注册表
# ==========================================

_REGISTRY_FILENAME = "interrupt_registry.json"


def _get_registry_path() -> str:
    """获取中断注册表文件路径"""
    settings = get_settings()
    # 注册表放在后端工作目录下
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", _REGISTRY_FILENAME)


def _load_registry() -> Dict[str, Dict]:
    """从磁盘加载中断注册表"""
    path = _get_registry_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_registry(registry: Dict[str, Dict]) -> None:
    """将中断注册表写入磁盘"""
    path = _get_registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"[Checkpoint] Failed to save interrupt registry: {e}")


# ==========================================
# Review Queue Item
# ==========================================

class ReviewQueueItem:
    """人工审查队列条目

    内存索引 + interrupt registry + checkpointer 持久化三层保障。
    """

    def __init__(
        self,
        workflow_run_id: str,
        gate_type: str,
        checkpoint_created_at: float,
        evidence_snapshot: Optional[Dict] = None,
        latest_draft_report: Optional[str] = None,
        confidence_data: Optional[Dict] = None,
        original_input: Optional[str] = None,
    ):
        self.workflow_run_id = workflow_run_id
        self.gate_type = gate_type
        self.checkpoint_created_at = checkpoint_created_at
        self.evidence_snapshot = evidence_snapshot or {}
        self.latest_draft_report = latest_draft_report or ""
        self.confidence_data = confidence_data or {}
        self.original_input = original_input or ""
        self.human_review_status: str = "pending"  # pending | approved | rejected
        self.human_review_notes: str = ""
        self.reviewed_at: Optional[float] = None
        self.reviewed_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_run_id": self.workflow_run_id,
            "gate_type": self.gate_type,
            "checkpoint_created_at": self.checkpoint_created_at,
            "evidence_snapshot": self.evidence_snapshot,
            "latest_draft_report": self.latest_draft_report,
            "confidence_data": self.confidence_data,
            "original_input": self.original_input,
            "human_review_status": self.human_review_status,
            "human_review_notes": self.human_review_notes,
            "reviewed_at": self.reviewed_at,
            "reviewed_by": self.reviewed_by,
        }

    @classmethod
    def from_interrupt_value(
        cls, workflow_run_id: str, interrupt_value: Dict, state_values: Dict
    ) -> "ReviewQueueItem":
        """从 interrupt payload 和 state values 重建 ReviewQueueItem"""
        item = cls(
            workflow_run_id=workflow_run_id,
            gate_type=interrupt_value.get("gate_type", "unknown"),
            checkpoint_created_at=0,
            evidence_snapshot=interrupt_value.get("evidence_snapshot", {}),
            latest_draft_report=interrupt_value.get("latest_draft_report", ""),
            confidence_data=interrupt_value.get("confidence_data", {}),
            original_input=state_values.get("original_input", ""),
        )
        return item


# ==========================================
# Checkpoint Manager（单例）
# ==========================================

class CheckpointManager:
    """Checkpoint 管理器

    职责：
      - 初始化 LangGraph checkpointer（PostgresSaver 或 MemorySaver）
      - 管理 review queue（内存索引 + interrupt registry 持久化）
      - 提供 workflow_run_id 生成
      - 支持从 checkpointer + registry 重建完整 review queue
    """

    _instance: Optional["CheckpointManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._checkpointer = None
        self._review_queue: Dict[str, ReviewQueueItem] = {}
        self._checkpointer_initialized = False

    def get_checkpointer(self):
        """获取 LangGraph checkpointer 实例"""
        if self._checkpointer_initialized:
            return self._checkpointer

        settings = get_settings()

        if not settings.WORKFLOW_CHECKPOINT_ENABLED:
            from langgraph.checkpoint.memory import MemorySaver
            self._checkpointer = MemorySaver()
            self._checkpointer_initialized = True
            print("[Checkpoint] Disabled by config, using MemorySaver (non-persistent)")
            return self._checkpointer

        db_url = settings.WORKFLOW_DB_URL
        if db_url:
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                self._checkpointer = PostgresSaver.from_conn_string(db_url)
                self._checkpointer.setup()
                self._checkpointer_initialized = True
                print("[Checkpoint] PostgresSaver initialized (persistent)")
                return self._checkpointer
            except Exception as e:
                print(f"[Checkpoint] PostgresSaver init failed: {e}, falling back to MemorySaver")

        from langgraph.checkpoint.memory import MemorySaver
        self._checkpointer = MemorySaver()
        self._checkpointer_initialized = True
        print("[Checkpoint] Using MemorySaver (non-persistent, dev only)")
        return self._checkpointer

    @property
    def is_persistent(self) -> bool:
        """当前 checkpointer 是否为持久化存储"""
        if not self._checkpointer_initialized:
            return False
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            return isinstance(self._checkpointer, PostgresSaver)
        except ImportError:
            return False

    @staticmethod
    def generate_workflow_run_id() -> str:
        """生成稳定的 workflow_run_id"""
        settings = get_settings()
        prefix = settings.WORKFLOW_THREAD_PREFIX
        ts = int(time.time() * 1000)
        short_id = uuid.uuid4().hex[:8]
        return f"{prefix}-{ts}-{short_id}"

    # ---- Review Queue + Interrupt Registry 操作 ----

    def add_to_review_queue(self, item: ReviewQueueItem) -> None:
        """将工作流加入人工审查队列，同时写入 interrupt registry"""
        self._review_queue[item.workflow_run_id] = item

        # 写入 interrupt registry（持久化）
        registry = _load_registry()
        registry[item.workflow_run_id] = {
            "gate_type": item.gate_type,
            "checkpoint_created_at": item.checkpoint_created_at,
            "added_at": time.time(),
        }
        _save_registry(registry)

        print(f"[Checkpoint] Added to review queue: {item.workflow_run_id} (gate={item.gate_type})")

    def get_review_queue(self) -> List[Dict[str, Any]]:
        """获取全部审查项目"""
        return [item.to_dict() for item in self._review_queue.values()]

    def get_review_item(self, workflow_run_id: str) -> Optional[Dict[str, Any]]:
        """获取指定工作流的审查项目（仅内存索引）"""
        item = self._review_queue.get(workflow_run_id)
        return item.to_dict() if item else None

    def update_review_status(
        self,
        workflow_run_id: str,
        status: str,
        notes: str = "",
        reviewed_by: Optional[str] = None,
    ) -> bool:
        """更新审查状态，已完成的从 registry 移除"""
        item = self._review_queue.get(workflow_run_id)
        if not item:
            return False

        item.human_review_status = status
        item.human_review_notes = notes
        item.reviewed_at = time.time()
        item.reviewed_by = reviewed_by

        # 从 interrupt registry 移除（不再待审）
        if status in ("approved", "rejected"):
            registry = _load_registry()
            registry.pop(workflow_run_id, None)
            _save_registry(registry)

        print(f"[Checkpoint] Review status updated: {workflow_run_id} → {status}")
        return True

    def remove_from_queue(self, workflow_run_id: str) -> bool:
        """从审查队列和 registry 移除"""
        if workflow_run_id in self._review_queue:
            del self._review_queue[workflow_run_id]

            registry = _load_registry()
            registry.pop(workflow_run_id, None)
            _save_registry(registry)

            print(f"[Checkpoint] Removed from review queue: {workflow_run_id}")
            return True
        return False

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """获取所有待审查项目"""
        return [
            item.to_dict()
            for item in self._review_queue.values()
            if item.human_review_status == "pending"
        ]

    def rebuild_review_queue(self, app_graph) -> int:
        """从 interrupt registry + checkpointer 重建内存 review queue

        后端重启后调用。扫描 registry 中所有 workflow_run_id，
        从 checkpointer 的 get_state() 获取 interrupt payload，
        重建完整的 ReviewQueueItem。

        Args:
            app_graph: 编译好的 LangGraph 实例（含 checkpointer）

        Returns:
            重建的条目数量
        """
        registry = _load_registry()
        if not registry:
            return 0

        rebuilt = 0
        stale_workflow_ids: List[str] = []
        for workflow_run_id, meta in list(registry.items()):
            # 跳过已在内存队列中的
            if workflow_run_id in self._review_queue:
                continue

            try:
                config = {"configurable": {"thread_id": workflow_run_id}}
                state_snapshot = app_graph.get_state(config)

                # 查找 interrupt
                for task in state_snapshot.tasks:
                    if task.interrupt:
                        interrupt_value = task.interrupt.value
                        item = ReviewQueueItem.from_interrupt_value(
                            workflow_run_id=workflow_run_id,
                            interrupt_value=interrupt_value,
                            state_values=state_snapshot.values,
                        )
                        item.checkpoint_created_at = meta.get("checkpoint_created_at", 0)
                        self._review_queue[workflow_run_id] = item
                        rebuilt += 1
                        print(f"[Checkpoint] Rebuilt review item: {workflow_run_id}")
                        break
                else:
                    # 没有 interrupt，说明已经被处理但 registry 没清理
                    # 清理 registry
                    stale_workflow_ids.append(workflow_run_id)

            except Exception as e:
                print(f"[Checkpoint] Failed to rebuild {workflow_run_id}: {e}")

        # 清理失效条目
        for workflow_run_id in stale_workflow_ids:
            registry.pop(workflow_run_id, None)
        _save_registry(registry)

        if rebuilt:
            print(f"[Checkpoint] Rebuilt {rebuilt} review items from persistent storage")
        return rebuilt


# ==========================================
# 全局单例
# ==========================================

_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """获取全局 CheckpointManager 单例"""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager
