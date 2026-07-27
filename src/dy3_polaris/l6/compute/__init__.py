"""算力资源协议 - 5 类资源描述、调度、策略、监控、降级.

模块组成:
- scheduler: 算力资源调度器（注册/分配/释放/降级/排空）
- strategy: 调度策略引擎（优先级/最短队列/加权均衡/亲和性）
- metrics: 算力度量与可观测性

支持的 5 类算力资源:
- LOCAL_CPU: 本地 CPU
- GPU: 本地 GPU
- SSH_REMOTE: SSH 远程节点
- HPC_SLURM: HPC 集群 (SLURM)
- CLOUD_GPU: 云端 GPU
"""

from __future__ import annotations

from .scheduler import ComputeScheduler, ComputeTask, TaskResult
from .strategy import (
    StrategyContext,
    PriorityFirstStrategy,
    ShortestQueueStrategy,
    WeightedLoadBalanceStrategy,
    AffinityStrategy,
    get_tool_preferred_type,
    set_tool_type_affinity,
    build_context,
)
from .metrics import ComputeMetrics

__all__ = [
    # 调度器
    "ComputeScheduler",
    "ComputeTask",
    "TaskResult",
    # 策略
    "StrategyContext",
    "PriorityFirstStrategy",
    "ShortestQueueStrategy",
    "WeightedLoadBalanceStrategy",
    "AffinityStrategy",
    "get_tool_preferred_type",
    "set_tool_type_affinity",
    "build_context",
    # 度量
    "ComputeMetrics",
]
