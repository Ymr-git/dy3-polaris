"""算力调度策略引擎.

提供多种资源选择策略：
- 优先级优先 (PriorityFirstStrategy)
- 最短队列优先 (ShortestQueueStrategy)
- 加权负载均衡 (WeightedLoadBalanceStrategy)
- 工具亲和性 (AffinityStrategy) — 根据工具历史分配偏好
- 资源类型亲和 — 优先匹配工具标注的 requires_compute 类型

所有策略实现统一接口: select(candidates, context) -> ComputeResourceDescriptor | None
"""

from __future__ import annotations

import time
from typing import Any

from ..core.models import ComputeResourceDescriptor, ComputeResourceType


# 工具名 -> 偏好的资源类型映射（可在运行时更新）
_TOOL_TYPE_AFFINITY: dict[str, ComputeResourceType] = {
    "path_simulation": ComputeResourceType.GPU,
    "thermocalc_phase_diagram": ComputeResourceType.HPC_SLURM,
    "vasp_query_result": ComputeResourceType.HPC_SLURM,
}


class StrategyContext:
    """策略执行的上下文信息."""

    __slots__ = (
        "tool_name", "task_priority", "estimated_duration_ms",
        "resource_history", "preferred_type",
    )

    def __init__(
        self,
        tool_name: str = "",
        task_priority: int = 0,
        estimated_duration_ms: int = 0,
        resource_history: dict[str, int] | None = None,
        preferred_type: ComputeResourceType | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.task_priority = task_priority
        self.estimated_duration_ms = estimated_duration_ms
        self.resource_history = resource_history or {}
        self.preferred_type = preferred_type


def _score_base(resource: ComputeResourceDescriptor) -> float:
    """基础评分: 越空闲越好."""
    max_q = resource.max_queue_depth or 1
    return 1.0 - (resource.queue_depth / max_q)


def _score_latency(resource: ComputeResourceDescriptor) -> float:
    """延迟评分: 延迟越低越好 (归一化到 [0,1])."""
    # 30000ms 为最大参考延迟
    return max(0.0, 1.0 - resource.estimated_latency_ms / 30000)


class PriorityFirstStrategy:
    """优先级优先策略.

    选择优先级最高的资源；优先级相同时选队列最短的。
    """

    def select(
        self,
        candidates: list[ComputeResourceDescriptor],
        context: StrategyContext | None = None,
    ) -> ComputeResourceDescriptor | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: (r.priority, -r.queue_depth),
        )


class ShortestQueueStrategy:
    """最短队列优先策略.

    选择当前队列最短（最空闲）的资源。
    队列相同时选延迟最低的。
    """

    def select(
        self,
        candidates: list[ComputeResourceDescriptor],
        context: StrategyContext | None = None,
    ) -> ComputeResourceDescriptor | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda r: (r.queue_depth, r.estimated_latency_ms),
        )


class WeightedLoadBalanceStrategy:
    """加权负载均衡策略.

    综合空闲度和延迟进行加权评分:
    score = 0.6 * idle_score + 0.4 * latency_score

    同时考虑任务优先级作为附加加分。
    """

    WEIGHT_IDLE = 0.6
    WEIGHT_LATENCY = 0.4

    def select(
        self,
        candidates: list[ComputeResourceDescriptor],
        context: StrategyContext | None = None,
    ) -> ComputeResourceDescriptor | None:
        if not candidates:
            return None

        def score(r: ComputeResourceDescriptor) -> float:
            s = self.WEIGHT_IDLE * _score_base(r) + self.WEIGHT_LATENCY * _score_latency(r)
            # 优先级加分
            if context and context.task_priority > 0:
                s += r.priority * 0.01
            return s

        return max(candidates, key=score)


class AffinityStrategy:
    """亲和性策略.

    1. 优先选择与工具历史分配最多的同一资源（减少冷启动）
    2. 若无历史，按工具-资源类型亲和匹配
    3. 兜底回退到加权负载均衡
    """

    def __init__(self, fallback: WeightedLoadBalanceStrategy | None = None) -> None:
        self._fallback = fallback or WeightedLoadBalanceStrategy()

    def select(
        self,
        candidates: list[ComputeResourceDescriptor],
        context: StrategyContext | None = None,
    ) -> ComputeResourceDescriptor | None:
        if not candidates:
            return None

        # 1. 历史亲和：选择该工具使用次数最多的资源
        if context and context.tool_name and context.resource_history:
            history = context.resource_history
            # 过滤候选中有历史记录的资源
            candidate_ids = {c.resource_id for c in candidates}
            best_rid = max(
                (rid for rid in history if rid in candidate_ids),
                key=lambda rid: history[rid],
                default=None,
            )
            if best_rid is not None:
                for c in candidates:
                    if c.resource_id == best_rid:
                        return c

        # 2. 类型亲和
        if context and context.preferred_type:
            for c in candidates:
                if c.resource_type == context.preferred_type:
                    return c

        # 3. 兜底
        return self._fallback.select(candidates, context)


def get_tool_preferred_type(tool_name: str) -> ComputeResourceType | None:
    """获取工具偏好的资源类型.

    基于 _TOOL_TYPE_AFFINITY 映射表。
    """
    return _TOOL_TYPE_AFFINITY.get(tool_name)


def set_tool_type_affinity(tool_name: str, resource_type: ComputeResourceType) -> None:
    """设置工具与资源类型的亲和关系."""
    _TOOL_TYPE_AFFINITY[tool_name] = resource_type


def build_context(
    tool_name: str = "",
    task_priority: int = 0,
    estimated_duration_ms: int = 0,
    resource_history: dict[str, int] | None = None,
) -> StrategyContext:
    """构建策略上下文，自动注入工具亲和类型."""
    preferred = get_tool_preferred_type(tool_name)
    return StrategyContext(
        tool_name=tool_name,
        task_priority=task_priority,
        estimated_duration_ms=estimated_duration_ms,
        resource_history=resource_history,
        preferred_type=preferred,
    )


__all__ = [
    "StrategyContext",
    "PriorityFirstStrategy",
    "ShortestQueueStrategy",
    "WeightedLoadBalanceStrategy",
    "AffinityStrategy",
    "get_tool_preferred_type",
    "set_tool_type_affinity",
    "build_context",
]
