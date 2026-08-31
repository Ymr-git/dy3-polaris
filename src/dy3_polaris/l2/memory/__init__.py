"""L2 memory 子模块 — 工作记忆 / 短期记忆 / 长期记忆三层体系.

融合世界先进方案 (Atkinson-Shiffrin 多重存储模型 + Redis Agent Memory 两层
记忆模型 + Claude Persistent Kernels):

1. ``WorkingMemory``: 工作记忆 — 当前会话的短期信息缓存 (Miller's Law 7±2).
   - ``MemoryChunk``: 信息块数据类 (chunk_id / content / chunk_type /
     timestamp / importance)
   - ``MAX_CHUNKS = 9``: Miller 上限, 超容量时 LRU 淘汰最旧块
2. ``ShortTermMemory``: 短期记忆 — 7 天保留窗口的时间衰减记忆.
   - ``RETENTION_HOURS = 168``: 保留窗口 (7 天), 超期条目自动清理
   - 按 ``learner_id`` 隔离
3. ``LongTermMemory``: 长期记忆 — 委托 ``L2Store`` 的持久化记忆层.
   - 依赖注入 (默认 ``InMemoryL2Store``)
   - 存储维度: 画像快照 / 答题历史 / 追踪状态

三层记忆分工:
- 工作记忆 → 当前会话上下文 (容量受限, 会话结束归档)
- 短期记忆 → 跨会话的近期交互 (7 天衰减, 超期清理)
- 长期记忆 → 持久化学情画像 (委托 store, 供 ProfileBuilder 检索)
"""

from dy3_polaris.l2.memory.long_term_memory import LongTermMemory
from dy3_polaris.l2.memory.short_term_memory import RETENTION_HOURS, ShortTermMemory
from dy3_polaris.l2.memory.tracing_service import (
    DEFAULT_IMPORTANCE_THRESHOLD,
    DEFAULT_MIGRATION_REP_THRESHOLD,
    DECAY,
    FACTOR,
    MemoryOutput,
    MemoryTracingService,
    REQUEST_RETENTION,
)
from dy3_polaris.l2.memory.working_memory import (
    MAX_CHUNKS,
    DEFAULT_IMPORTANCE,
    MemoryChunk,
    WorkingMemory,
)

__all__ = [
    # 工作记忆
    "MemoryChunk",
    "WorkingMemory",
    "MAX_CHUNKS",
    "DEFAULT_IMPORTANCE",
    # 短期记忆
    "ShortTermMemory",
    "RETENTION_HOURS",
    # 长期记忆
    "LongTermMemory",
    # 全链路编排服务
    "MemoryTracingService",
    "MemoryOutput",
    "DECAY",
    "FACTOR",
    "REQUEST_RETENTION",
    "DEFAULT_IMPORTANCE_THRESHOLD",
    "DEFAULT_MIGRATION_REP_THRESHOLD",
]
