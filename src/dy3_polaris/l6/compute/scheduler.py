"""算力资源调度器.

管理 5 类算力资源的注册、分配、释放与降级：
- LOCAL_CPU / GPU / SSH_REMOTE / HPC_SLURM / CLOUD_GPU
- 资源注册与注销
- 任务入队（分配）与出队（释放/完成/失败）
- 自动降级（高阶资源不可用时回退到低阶资源）
- 资源状态转换
- DRAINING 排空模式
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable

from ..core.config import get_config
from ..core.exceptions import (
    ComputeDegradationError,
    ComputeNoAvailableError,
    ComputeQueueFullError,
    ComputeResourceNotFoundError,
    ComputeTaskNotFoundError,
)
from ..core.models import ComputeResourceDescriptor, ComputeResourceStatus, ComputeResourceType

logger = logging.getLogger(__name__)


# 降级路径：从高到低
_DEGRADATION_PATH: list[ComputeResourceType] = [
    ComputeResourceType.CLOUD_GPU,
    ComputeResourceType.HPC_SLURM,
    ComputeResourceType.GPU,
    ComputeResourceType.SSH_REMOTE,
    ComputeResourceType.LOCAL_CPU,
]

_DEGRADATION_INDEX = {t: i for i, t in enumerate(_DEGRADATION_PATH)}


class TaskResult(str, Enum):
    """算力任务结果."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ComputeTask:
    """算力任务记录."""

    __slots__ = (
        "task_id", "resource_id", "tool_name", "status",
        "priority", "submitted_at", "started_at", "completed_at",
        "estimated_duration_ms", "metadata",
    )

    def __init__(
        self,
        task_id: str,
        resource_id: str,
        tool_name: str = "",
        priority: int = 0,
        estimated_duration_ms: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.resource_id = resource_id
        self.tool_name = tool_name
        self.status = TaskResult.PENDING
        self.priority = priority
        self.submitted_at: float = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.estimated_duration_ms = estimated_duration_ms
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "priority": self.priority,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_duration_ms": self.estimated_duration_ms,
        }


class ComputeScheduler:
    """算力资源调度器.

    核心能力:
    - 注册/注销算力资源
    - 按策略分配资源（入队）
    - 任务完成/失败/取消（出队）
    - 自动降级：高阶资源不可用时回退低阶资源
    - DRAINING 排空模式

    使用示例:
        scheduler = ComputeScheduler()
        scheduler.register(ComputeResourceDescriptor(
            resource_type=ComputeResourceType.GPU,
            name="nvidia-rtx4090",
            gpu_count=1, gpu_memory_gb=24.0,
        ))
        task = scheduler.allocate("bkt_compute", resource_type=ComputeResourceType.GPU)
        scheduler.release(task.task_id)
    """

    def __init__(self) -> None:
        self._resources: dict[str, ComputeResourceDescriptor] = {}
        self._tasks: dict[str, ComputeTask] = {}
        # 资源类型索引: type -> [resource_id, ...]
        self._type_index: dict[ComputeResourceType, list[str]] = {t: [] for t in ComputeResourceType}
        self._degradation_enabled = True

    # --------------------------------------------------------
    # 资源注册
    # --------------------------------------------------------

    def register(self, resource: ComputeResourceDescriptor) -> str:
        """注册算力资源.

        Args:
            resource: 算力资源描述符

        Returns:
            资源 ID
        """
        rid = resource.resource_id
        self._resources[rid] = resource
        self._type_index[resource.resource_type].append(rid)
        logger.info("算力资源已注册: %s (%s)", rid, resource.name)
        return rid

    def unregister(self, resource_id: str) -> bool:
        """注销算力资源.

        资源必须为空闲状态（队列为空）才能注销。
        """
        resource = self._resources.get(resource_id)
        if resource is None:
            return False
        if resource.queue_depth > 0:
            logger.warning("资源 %s 队列非空，无法注销", resource_id)
            return False
        del self._resources[resource_id]
        self._type_index[resource.resource_type].remove(resource_id)
        logger.info("算力资源已注销: %s", resource_id)
        return True

    def get_resource(self, resource_id: str) -> ComputeResourceDescriptor | None:
        return self._resources.get(resource_id)

    def set_status(self, resource_id: str, status: ComputeResourceStatus) -> None:
        """设置资源状态."""
        resource = self._resources.get(resource_id)
        if resource is None:
            raise ComputeResourceNotFoundError(resource_id)
        resource.status = status
        logger.debug("资源 %s 状态变更: %s", resource_id, status.value)

    # --------------------------------------------------------
    # 资源分配（入队）
    # --------------------------------------------------------

    def allocate(
        self,
        task_id: str,
        resource_type: ComputeResourceType | None = None,
        tool_name: str = "",
        priority: int = 0,
        estimated_duration_ms: int = 0,
        metadata: dict[str, Any] | None = None,
        strategy: Callable[[list[ComputeResourceDescriptor]], ComputeResourceDescriptor | None] | None = None,
    ) -> ComputeTask:
        """分配算力资源.

        按策略从候选资源中选择最优资源，将任务入队。

        Args:
            task_id: 任务 ID
            resource_type: 期望的资源类型（None 则从所有可用资源中选择）
            tool_name: 关联的工具名称
            priority: 优先级
            estimated_duration_ms: 预估执行时长
            metadata: 附加元数据
            strategy: 自定义选择策略

        Returns:
            ComputeTask

        Raises:
            ComputeNoAvailableError: 无可用资源（含降级后仍无）
        """
        if resource_type is not None:
            resource = self._try_allocate(task_id, resource_type, strategy)
            if resource is not None:
                return self._enqueue(task_id, resource, tool_name, priority, estimated_duration_ms, metadata)

        # 降级
        if self._degradation_enabled and resource_type is not None:
            resource = self._degrade_allocate(task_id, resource_type, strategy)
            if resource is not None:
                return self._enqueue(task_id, resource, tool_name, priority, estimated_duration_ms, metadata)
            raise ComputeNoAvailableError(
                f"No available resource for type={resource_type.value} after degradation"
            )

        # 无指定类型，从所有可用资源中选择
        candidates = self._available_resources()
        if not candidates:
            raise ComputeNoAvailableError("No available resources")

        selected = strategy(candidates) if strategy else self._default_strategy(candidates)
        if selected is None:
            raise ComputeNoAvailableError("Strategy returned no resource")
        return self._enqueue(task_id, selected, tool_name, priority, estimated_duration_ms, metadata)

    def _try_allocate(
        self,
        task_id: str,
        resource_type: ComputeResourceType,
        strategy: Callable[[list[ComputeResourceDescriptor]], ComputeResourceDescriptor | None] | None,
    ) -> ComputeResourceDescriptor | None:
        """尝试从指定类型分配."""
        candidates = self._available_by_type(resource_type)
        if not candidates:
            return None
        return strategy(candidates) if strategy else self._default_strategy(candidates)

    def _degrade_allocate(
        self,
        task_id: str,
        original_type: ComputeResourceType,
        strategy: Callable[[list[ComputeResourceDescriptor]], ComputeResourceDescriptor | None] | None,
    ) -> ComputeResourceDescriptor | None:
        """降级分配."""
        start_idx = _DEGRADATION_INDEX.get(original_type, 0)
        for i in range(start_idx + 1, len(_DEGRADATION_PATH)):
            fallback_type = _DEGRADATION_PATH[i]
            candidates = self._available_by_type(fallback_type)
            if candidates:
                selected = strategy(candidates) if strategy else self._default_strategy(candidates)
                if selected is not None:
                    logger.info(
                        "降级分配: %s -> %s, 资源=%s",
                        original_type.value, fallback_type.value, selected.resource_id,
                    )
                    return selected
        return None

    def _enqueue(
        self,
        task_id: str,
        resource: ComputeResourceDescriptor,
        tool_name: str,
        priority: int,
        estimated_duration_ms: int,
        metadata: dict[str, Any] | None,
    ) -> ComputeTask:
        """将任务入队."""
        if resource.queue_depth >= resource.max_queue_depth:
            raise ComputeQueueFullError(resource.resource_id, resource.max_queue_depth)

        resource.current_queue.append(task_id)
        if resource.queue_depth >= resource.max_queue_depth:
            resource.status = ComputeResourceStatus.BUSY

        task = ComputeTask(
            task_id=task_id,
            resource_id=resource.resource_id,
            tool_name=tool_name,
            priority=priority,
            estimated_duration_ms=estimated_duration_ms,
            metadata=metadata,
        )
        self._tasks[task_id] = task
        logger.debug("任务 %s 已分配到资源 %s", task_id, resource.resource_id)
        return task

    # --------------------------------------------------------
    # 任务状态转换
    # --------------------------------------------------------

    def start_task(self, task_id: str) -> ComputeTask:
        """标记任务为运行中."""
        task = self._get_task(task_id)
        task.status = TaskResult.RUNNING
        task.started_at = time.time()
        return task

    def complete_task(self, task_id: str, *, result: dict[str, Any] | None = None) -> ComputeTask:
        """标记任务完成并出队."""
        task = self._get_task(task_id)
        task.status = TaskResult.COMPLETED
        task.completed_at = time.time()
        self._dequeue(task)
        return task

    def fail_task(self, task_id: str, *, error: str = "") -> ComputeTask:
        """标记任务失败并出队."""
        task = self._get_task(task_id)
        task.status = TaskResult.FAILED
        task.completed_at = time.time()
        self._dequeue(task)
        return task

    def cancel_task(self, task_id: str) -> ComputeTask:
        """取消任务并出队."""
        task = self._get_task(task_id)
        task.status = TaskResult.CANCELLED
        task.completed_at = time.time()
        self._dequeue(task)
        return task

    def _dequeue(self, task: ComputeTask) -> None:
        """从资源队列中移除任务."""
        resource = self._resources.get(task.resource_id)
        if resource is None:
            return
        if task.task_id in resource.current_queue:
            resource.current_queue.remove(task.task_id)
        # 如果队列有空位且当前是 BUSY，恢复为 AVAILABLE
        if (resource.status == ComputeResourceStatus.BUSY
                and resource.queue_depth < resource.max_queue_depth):
            resource.status = ComputeResourceStatus.AVAILABLE

    def _get_task(self, task_id: str) -> ComputeTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise ComputeTaskNotFoundError(task_id)
        return task

    # --------------------------------------------------------
    # DRAINING 排空
    # --------------------------------------------------------

    def start_draining(self, resource_id: str) -> None:
        """进入排空模式 — 不再接受新任务，等待队列中的任务完成."""
        resource = self._resources.get(resource_id)
        if resource is None:
            raise ComputeResourceNotFoundError(resource_id)
        resource.status = ComputeResourceStatus.DRAINING
        logger.info("资源 %s 进入 DRAINING 模式", resource_id)

    def is_drained(self, resource_id: str) -> bool:
        """检查资源是否已排空（DRAINING 且队列为空）."""
        resource = self._resources.get(resource_id)
        if resource is None:
            return False
        return resource.status == ComputeResourceStatus.DRAINING and resource.queue_depth == 0

    def set_offline(self, resource_id: str) -> None:
        """将资源设为离线（必须已排空）."""
        resource = self._resources.get(resource_id)
        if resource is None:
            raise ComputeResourceNotFoundError(resource_id)
        if resource.queue_depth > 0:
            raise ComputeQueueFullError(resource_id, resource.queue_depth)
        resource.status = ComputeResourceStatus.OFFLINE

    # --------------------------------------------------------
    # 查询
    # --------------------------------------------------------

    def get_task(self, task_id: str) -> ComputeTask | None:
        return self._tasks.get(task_id)

    def _available_by_type(self, resource_type: ComputeResourceType) -> list[ComputeResourceDescriptor]:
        """获取指定类型的可用资源."""
        rids = self._type_index.get(resource_type, [])
        return [self._resources[rid] for rid in rids if self._resources[rid].is_available]

    def _available_resources(self) -> list[ComputeResourceDescriptor]:
        """获取所有可用资源."""
        return [r for r in self._resources.values() if r.is_available]

    def resources_by_type(self, resource_type: ComputeResourceType) -> list[ComputeResourceDescriptor]:
        """获取指定类型的所有资源（不论状态）."""
        rids = self._type_index.get(resource_type, [])
        return [self._resources[rid] for rid in rids if rid in self._resources]

    @property
    def resource_count(self) -> int:
        return len(self._resources)

    @property
    def available_count(self) -> int:
        return len(self._available_resources())

    @property
    def total_tasks(self) -> int:
        return len(self._tasks)

    @property
    def active_tasks(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status in (TaskResult.PENDING, TaskResult.RUNNING))

    def _default_strategy(
        self, candidates: list[ComputeResourceDescriptor],
    ) -> ComputeResourceDescriptor | None:
        """默认策略：优先级最高 > 队列最短 > 预估延迟最低."""
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r.priority, -r.queue_depth, -r.estimated_latency_ms))

    # --------------------------------------------------------
    # 导出
    # --------------------------------------------------------

    def export_summary(self) -> dict[str, Any]:
        """导出调度器摘要."""
        resources_by_type: dict[str, int] = {}
        for rtype, rids in self._type_index.items():
            resources_by_type[rtype.value] = len(rids)
        return {
            "resource_count": self.resource_count,
            "available_count": self.available_count,
            "total_tasks": self.total_tasks,
            "active_tasks": self.active_tasks,
            "resources_by_type": resources_by_type,
            "degradation_enabled": self._degradation_enabled,
        }

    def export_all(self) -> dict[str, Any]:
        """导出完整数据."""
        return {
            "resources": {rid: r.model_dump(mode="json") for rid, r in self._resources.items()},
            "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()},
            "summary": self.export_summary(),
        }

    def clear(self) -> None:
        """清空所有资源和任务."""
        self._resources.clear()
        self._tasks.clear()
        self._type_index = {t: [] for t in ComputeResourceType}


__all__ = ["ComputeScheduler", "ComputeTask", "TaskResult"]
