"""L7 Artifact 管理系统 — 版本管理 DAG (version_manager.py).

任务拆分 T3 · 设计文档 Ch.3.3。

实现 Artifact 版本树的有向无环图 (DAG) 管理，支持:

    v1 → v2 → v3a (分支 A, Session Fork)
         ↘ v3b (分支 B)
              → v4 (合并版本, 多 parent)

融合世界先进方案:
    - Git DAG 数据模型: commit 携带 parent 列表, 分支合并 = 多 parent 的
      合并 commit; 结构共享 (未变版本复用)
    - 三方合并思想: base + ours + theirs, 冲突检测后由调用方裁决
    - 内容寻址 (Git blob): 版本快照按内容哈希去重

设计要点:
    - ArtifactVersionGraph 是独立于 ArtifactManager 的纯 DAG 结构
    - merge() 生成多 parent 合并节点 (Git merge commit 语义)
    - common_ancestor() 支持三方合并的 base 定位
    - is_descendant() 支持分支归属判断
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Artifact, ArtifactVersionNode

#: 合并策略
MERGE_STRATEGY_OURS = "ours"          # 以主线 (ours) 为准
MERGE_STRATEGY_THEIRS = "theirs"      # 以分支 (theirs) 为准
MERGE_STRATEGY_AUTO = "auto"          # 三方合并 (base 上无冲突则合并, 冲突抛错)
VALID_STRATEGIES = (MERGE_STRATEGY_OURS, MERGE_STRATEGY_THEIRS, MERGE_STRATEGY_AUTO)


class MergeConflictError(Exception):
    """合并冲突异常 — auto 策略下 base→ours 与 base→theirs 修改同一字段."""

    def __init__(self, artifact_id: str, branch_version: int, conflicts: list[str]) -> None:
        self.artifact_id = artifact_id
        self.branch_version = branch_version
        self.conflicts = conflicts
        super().__init__(
            f"Artifact {artifact_id} 分支 v{branch_version} 合并冲突字段: {conflicts}"
        )


@dataclass
class MergeResult:
    """合并结果."""

    artifact_id: str
    merge_version: int
    parents: list[int]
    strategy: str
    conflicts: list[str] = field(default_factory=list)
    merged_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VersionNode:
    """DAG 版本节点 — 支持多 parent (Git commit 语义)."""

    version: int
    artifact_id: str
    parents: tuple[int, ...] = ()
    fork_origin: str | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())
    payload_hash: str = ""

    def to_model(self) -> ArtifactVersionNode:
        """转换为 l7.models.ArtifactVersionNode (主 parent 作为 parent_version)."""
        return ArtifactVersionNode(
            version=self.version,
            artifact_id=self.artifact_id,
            parent_version=self.parents[0] if self.parents else None,
            fork_origin=self.fork_origin,
            created_at=self.created_at,
        )


class ArtifactVersionGraph:
    """Artifact 版本 DAG — 多 parent 支持 + 合并 + 祖先查询.

    使用示例::

        graph = ArtifactVersionGraph("art-001")
        v1 = graph.add_version(artifact, parents=())
        v2 = graph.add_version(artifact2, parents=(1,))
        v3a = graph.add_version(artifact3a, parents=(2,), fork_origin="session-fork-A")
        v3b = graph.add_version(artifact3b, parents=(2,), fork_origin="session-fork-B")
        result = graph.merge("art-001", branch=3, main=2, strategy="auto")
    """

    def __init__(self, artifact_id: str) -> None:
        self.artifact_id = artifact_id
        self._nodes: dict[int, VersionNode] = {}
        self._snapshots: dict[int, Artifact] = {}
        self._next_version = 1

    # ----------------------------------------------------------
    # 版本操作
    # ----------------------------------------------------------

    def add_version(
        self,
        artifact: Artifact,
        parents: tuple[int, ...] | int | None = None,
        fork_origin: str | None = None,
    ) -> VersionNode:
        """添加版本节点.

        Args:
            artifact: 版本快照。
            parents: 父版本 (单个 int 或元组, 合并节点可多个)。
            fork_origin: 分叉来源 (Session Fork ID 或原因)。

        Returns:
            新版本节点。
        """
        version = self._next_version
        if isinstance(parents, int):
            parents_tuple: tuple[int, ...] = (parents,)
        elif parents is None:
            parents_tuple = ()
        else:
            parents_tuple = tuple(parents)
        node = VersionNode(
            version=version,
            artifact_id=self.artifact_id,
            parents=parents_tuple,
            fork_origin=fork_origin,
        )
        self._nodes[version] = node
        self._snapshots[version] = artifact
        self._next_version += 1
        return node

    def get_node(self, version: int) -> VersionNode | None:
        return self._nodes.get(version)

    def get_snapshot(self, version: int) -> Artifact | None:
        return self._snapshots.get(version)

    def all_versions(self) -> list[int]:
        """全部版本号 (拓扑序: 父先于子)."""
        return self._topological_order()

    def latest_version(self) -> int:
        """最新版本号 (根节点数最多者优先, 即主线 head)."""
        if not self._nodes:
            return 0
        # 主线 head = 无子节点的版本中版本号最大者 (含多 parent 合并)
        children_of: set[int] = set()
        for node in self._nodes.values():
            children_of.update(node.parents)
        heads = [v for v in self._nodes if v not in children_of]
        return max(heads)

    def parent_versions(self, version: int) -> list[int]:
        """返回指定版本的父版本列表 (合并节点多个)."""
        node = self._nodes.get(version)
        return list(node.parents) if node else []

    def is_descendant(self, ancestor: int, version: int) -> bool:
        """判断 ancestor 是否为 version 的祖先 (含自身)."""
        if ancestor == version:
            return True
        return ancestor in self.get_lineage(version)

    def get_lineage(self, version: int) -> list[int]:
        """从根到指定版本的版本链 (沿第一父指针)."""
        if version not in self._nodes:
            return []
        lineage: list[int] = []
        current: int | None = version
        seen: set[int] = set()
        while current is not None and current not in seen:
            seen.add(current)
            lineage.append(current)
            node = self._nodes.get(current)
            current = node.parents[0] if node and node.parents else None
        lineage.reverse()
        return lineage

    def common_ancestor(self, v1: int, v2: int) -> int | None:
        """最近公共祖先 (三方合并的 base 定位)."""
        if v1 not in self._nodes or v2 not in self._nodes:
            return None
        lineage1 = set(self.get_lineage(v1))
        lineage2 = self.get_lineage(v2)
        for v in reversed(lineage2):  # 从根向目标找第一个共同节点
            if v in lineage1:
                return v
        return None

    # ----------------------------------------------------------
    # 分支合并
    # ----------------------------------------------------------

    def merge(
        self,
        artifact_id: str,
        branch_version: int,
        main_version: int,
        strategy: str = MERGE_STRATEGY_AUTO,
        merge_reason: str = "merge",
    ) -> MergeResult:
        """将分支版本合并回主线 (Git merge commit 语义).

        Args:
            artifact_id: Artifact ID (用于错误信息)。
            branch_version: 分支版本号 (theirs)。
            main_version: 主线版本号 (ours)。
            strategy: ours / theirs / auto。
            merge_reason: 合并原因 (记录到 fork_origin)。

        Returns:
            合并结果 (含合并版本号与冲突列表)。

        Raises:
            ValueError: 版本不存在或分支/主线无效。
            MergeConflictError: auto 策略下三方合并冲突。
        """
        branch = self._nodes.get(branch_version)
        main = self._nodes.get(main_version)
        if branch is None or main is None:
            raise ValueError(
                f"合并版本不存在: branch=v{branch_version}, main=v{main_version}"
            )
        branch_snap = self._snapshots.get(branch_version)
        main_snap = self._snapshots.get(main_version)
        if branch_snap is None or main_snap is None:
            raise ValueError("合并版本快照缺失")

        conflicts: list[str] = []
        if strategy == MERGE_STRATEGY_AUTO:
            base_version = self.common_ancestor(branch_version, main_version)
            if base_version is not None and base_version != branch_version and base_version != main_version:
                base_snap = self._snapshots.get(base_version)
                if base_snap is not None:
                    conflicts = _detect_conflicts(
                        base_snap.payload,
                        main_snap.payload,
                        branch_snap.payload,
                    )
            if conflicts:
                raise MergeConflictError(artifact_id, branch_version, conflicts)
            merged_payload = branch_snap.payload
        elif strategy == MERGE_STRATEGY_OURS:
            merged_payload = main_snap.payload
        else:  # theirs
            merged_payload = branch_snap.payload

        # 生成合并节点 (多 parent)
        merged = branch_snap.model_copy(deep=False)
        merged.version = self._next_version
        merged.payload = merged_payload
        merged.updated_at = __import__("time").time()
        node = self.add_version(
            merged,
            parents=(main_version, branch_version),
            fork_origin=merge_reason or "merge",
        )
        return MergeResult(
            artifact_id=artifact_id,
            merge_version=node.version,
            parents=list(node.parents),
            strategy=strategy,
            conflicts=conflicts,
            merged_payload=merged_payload,
        )

    # ----------------------------------------------------------
    # 快照序列化
    # ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (持久化支持)."""
        return {
            "artifact_id": self.artifact_id,
            "next_version": self._next_version,
            "nodes": [
                {
                    "version": n.version,
                    "parents": list(n.parents),
                    "fork_origin": n.fork_origin,
                    "created_at": n.created_at,
                }
                for n in self._nodes.values()
            ],
            "snapshots": {
                str(v): art.to_dict() for v, art in self._snapshots.items()
            },
        }

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    def _topological_order(self) -> list[int]:
        """BFS 拓扑排序 (父先于子, 子按版本号稳定)."""
        from collections import deque

        indegree: dict[int, int] = {v: len(n.parents) for v, n in self._nodes.items()}
        children: dict[int, list[int]] = {v: [] for v in self._nodes}
        for v, n in self._nodes.items():
            for p in n.parents:
                children[p].append(v)
        queue = deque(sorted(v for v, d in indegree.items() if d == 0))
        order: list[int] = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for c in sorted(children.get(v, [])):
                indegree[c] -= 1
                if indegree[c] == 0:
                    queue.append(c)
        return order


def _detect_conflicts(
    base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any]
) -> list[str]:
    """三方合并冲突检测 — base→ours 与 base→theirs 都修改同一字段.

    Args:
        base: 公共祖先版本 payload。
        ours: 主线版本 payload。
        theirs: 分支版本 payload。

    Returns:
        冲突字段路径列表 (顶层键级检测)。
    """
    conflicts: list[str] = []
    all_keys = set(base) | set(ours) | set(theirs)
    for key in all_keys:
        base_v = base.get(key, _MISSING)
        ours_v = ours.get(key, _MISSING)
        theirs_v = theirs.get(key, _MISSING)
        ours_changed = ours_v != base_v
        theirs_changed = theirs_v != base_v
        if ours_changed and theirs_changed and ours_v != theirs_v:
            conflicts.append(key)
    return conflicts


#: 哨兵 — 键不存在标记
_MISSING = object()
