"""L7 Artifact 管理系统 — 包入口.

任务拆分 T3 交付物组织:

    models.py           — Artifact 元数据模型 (T1 模型重新导出 + 管理域专属)
    lifecycle.py        — 生命周期状态机 (5 阶段合法转移表 + 子状态)
    version_manager.py  — 版本管理 DAG (多 parent 合并 + 公共祖先 + 冲突检测)
    edit_channel.py     — Artifact-Edit 通道 (增量差异 + L5 broadcast 回调)
    store.py            — 三级存储 (L1 内存 LRU / L2 文件 / L3 服务端 + CAS)
    search.py           — 搜索与过滤 (倒排索引 + 布尔查询 + 学情关联)

设计文档: 02-设计/L7-体验呈现设计/layer7-experience-presentation.html Ch.3
任务拆分: 02-设计/报告/L7体验呈现任务拆分.html Task 3
"""

from __future__ import annotations

from .lifecycle import (
    LifecycleStateMachine,
    StateTransitionError,
    assert_transition,
    get_state_machine,
)
from .models import ArtifactMetadata, EditPermission
from .search import InvertedIndex, SearchEngine, build_index, tokenize
from .store import (
    ArtifactStore,
    ContentStore,
    JsonFileArtifactStore,
    MemoryArtifactStore,
    NoopServerStore,
    ServerArtifactStore,
    TieredArtifactStore,
)
from .version_manager import (
    ArtifactVersionGraph,
    MergeConflictError,
    MergeResult,
    VersionNode,
)

__all__ = [
    # lifecycle
    "LifecycleStateMachine",
    "StateTransitionError",
    "assert_transition",
    "get_state_machine",
    # models
    "ArtifactMetadata",
    "EditPermission",
    # search
    "InvertedIndex",
    "SearchEngine",
    "build_index",
    "tokenize",
    # store
    "ArtifactStore",
    "ContentStore",
    "JsonFileArtifactStore",
    "MemoryArtifactStore",
    "NoopServerStore",
    "ServerArtifactStore",
    "TieredArtifactStore",
    # version_manager
    "ArtifactVersionGraph",
    "MergeConflictError",
    "MergeResult",
    "VersionNode",
]
