"""Artifact 生命周期管理器 — L7 体验呈现层核心组件.

融合方案:
- Jupyter nbformat: Artifact 元数据 + payload 分离
- Git: DAG 版本树管理 (VersionTree)
- IndexedDB: 三级缓存策略 (L1 内存 / L2 本地 / L3 服务端) — 本实现为 L1 内存层
- RFC 6902 JSON Patch: 增量编辑 (ArtifactDiff + apply_edit)

核心能力:
1. 生命周期管理: register / get / update / archive / list_artifacts
2. 版本管理 DAG: update / fork / get_version_history / get_latest_version / get_version
3. 编辑管理: apply_edit (RFC 6902 JSON Patch) / get_diff
4. 搜索过滤: search (全文搜索) / list_artifacts (多维过滤)
5. 统计信息: get_stats (按类型/状态聚合)
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

from .events import (
    ARTIFACT_ARCHIVED,
    ARTIFACT_MERGED,
    ARTIFACT_REGISTERED,
    ARTIFACT_REMOVED,
    ARTIFACT_RESTORED,
    ARTIFACT_REVIEWED,
    ARTIFACT_UPDATED,
    get_global_emitter,
)
from .exceptions import (
    ArtifactNotFoundError,
    ArtifactNotEditableError,
    ArtifactValidationError,
    VersionConflictError,
)
from .models import (
    Artifact,
    ArtifactDiff,
    ArtifactLifecycleState,
    ArtifactType,
    ArtifactVersionNode,
    DiffOp,
    DiffOpType,
)

#: test 操作用于标记 "路径不存在" 的哨兵值 (与任何用户值都不相等).
_TEST_MISSING: Any = object()


# ============================================================
# VersionTree — Artifact 版本树 (DAG 结构)
# ============================================================


class VersionTree:
    """Artifact 版本树 (DAG 结构).

    融合 Git 的版本树设计:
    - _nodes: 版本号 -> ArtifactVersionNode (版本节点)
    - _versions: 版本号 -> Artifact (版本快照)
    - _branches: 父版本号 -> [子版本号列表] (DAG 边)

    特性:
    1. DAG 结构: 支持分支 (fork) 和合并
    2. 版本快照: 每个版本保存完整的 Artifact 快照
    3. 拓扑排序: get_all_versions 返回拓扑序
    4. 溯源链: get_lineage 返回从根到指定版本的路径
    """

    def __init__(self, root_artifact: Artifact) -> None:
        """初始化版本树, 以 root_artifact 为根版本 (v1).

        Args:
            root_artifact: 根版本 Artifact
        """
        self._nodes: dict[int, ArtifactVersionNode] = {}
        self._versions: dict[int, Artifact] = {}
        self._branches: dict[int, list[int]] = {}

        # 添加根版本节点
        root_node = ArtifactVersionNode(
            version=root_artifact.version,
            artifact_id=root_artifact.artifact_id,
            parent_version=None,
            fork_origin=root_artifact.fork_origin,
            created_at=root_artifact.created_at,
        )
        self._nodes[root_artifact.version] = root_node
        self._versions[root_artifact.version] = root_artifact
        self._branches[root_artifact.version] = []

    def add_version(
        self,
        artifact: Artifact,
        parent_version: int | None = None,
    ) -> ArtifactVersionNode:
        """添加新版本到版本树.

        Args:
            artifact: 新版本的 Artifact 快照
            parent_version: 父版本号 (根版本为 None)

        Returns:
            新创建的 ArtifactVersionNode
        """
        node = ArtifactVersionNode(
            version=artifact.version,
            artifact_id=artifact.artifact_id,
            parent_version=parent_version,
            fork_origin=artifact.fork_origin,
            created_at=artifact.created_at,
        )
        self._nodes[artifact.version] = node
        self._versions[artifact.version] = artifact

        # 建立 DAG 边
        if parent_version is not None:
            if parent_version not in self._branches:
                self._branches[parent_version] = []
            self._branches[parent_version].append(artifact.version)

        if artifact.version not in self._branches:
            self._branches[artifact.version] = []

        return node

    def get_node(self, version: int) -> ArtifactVersionNode | None:
        """获取指定版本的节点.

        Args:
            version: 版本号

        Returns:
            ArtifactVersionNode, 不存在则返回 None
        """
        return self._nodes.get(version)

    def get_artifact(self, version: int) -> Artifact | None:
        """获取指定版本的 Artifact 快照.

        Args:
            version: 版本号

        Returns:
            Artifact, 不存在则返回 None
        """
        return self._versions.get(version)

    def get_lineage(self, version: int) -> list[ArtifactVersionNode]:
        """获取从根到指定版本的溯源链.

        沿 parent_version 链向上遍历到根, 然后反转得到从根到目标的顺序.

        Args:
            version: 目标版本号

        Returns:
            从根到目标版本的 ArtifactVersionNode 列表
        """
        chain: list[ArtifactVersionNode] = []
        current: int | None = version
        visited: set[int] = set()
        while current is not None and current not in visited:
            visited.add(current)
            node = self._nodes.get(current)
            if node is None:
                break
            chain.append(node)
            current = node.parent_version
        chain.reverse()
        return chain

    def get_latest_version(self) -> int:
        """获取最高版本号.

        Returns:
            最高版本号, 无版本则返回 0
        """
        if not self._nodes:
            return 0
        return max(self._nodes.keys())

    def get_all_versions(self) -> list[ArtifactVersionNode]:
        """获取所有版本节点 (拓扑排序).

        使用 BFS 从根节点开始遍历, 保证父版本在子版本之前.

        Returns:
            拓扑排序的 ArtifactVersionNode 列表
        """
        # 找到根节点 (parent_version=None)
        root_version: int | None = None
        for v, node in self._nodes.items():
            if node.parent_version is None:
                root_version = v
                break

        if root_version is None:
            # 没有根节点, 按版本号排序返回
            return [self._nodes[v] for v in sorted(self._nodes.keys())]

        # BFS 拓扑排序
        result: list[ArtifactVersionNode] = []
        visited: set[int] = set()
        queue: list[int] = [root_version]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            node = self._nodes.get(current)
            if node is not None:
                result.append(node)
                # 子版本按版本号排序入队
                children = sorted(self._branches.get(current, []))
                queue.extend(children)

        # 补充未访问到的节点 (理论上不应该发生)
        for v in sorted(self._nodes.keys()):
            if v not in visited:
                result.append(self._nodes[v])

        return result


# ============================================================
# ArtifactManager — Artifact 生命周期管理器
# ============================================================


class ArtifactManager:
    """Artifact 生命周期管理器.

    融合方案:
    - Jupyter nbformat: Artifact 元数据 + payload 分离
    - Git: DAG 版本树管理 (VersionTree)
    - IndexedDB: 三级缓存策略 (L1 内存 / L2 本地 / L3 服务端)
    - RFC 6902 JSON Patch: 增量编辑 (ArtifactDiff + apply_edit)

    线程安全: 使用 threading.RLock 保护所有操作。
    存储策略: 本实现为 L1 内存层 (dict-based), 可注入 TieredArtifactStore
    启用 L2/L3 持久化与 CAS 内容寻址 (T3 增强)。

    Attributes:
        _artifacts: artifact_id -> 最新主线版本 (head)
        _version_trees: artifact_id -> VersionTree (DAG 版本树)
        _lock: 可重入线程锁
    """

    def __init__(
        self,
        store: Any | None = None,
        state_machine: Any | None = None,
        emit_events: bool = True,
        guard_archived: bool = True,
    ) -> None:
        """初始化 ArtifactManager.

        Args:
            store: 可选的三级存储 (TieredArtifactStore 或 ArtifactStore),
                启用后 register/update/archive 等写操作同步到存储层。
            state_machine: 可选的生命周期状态机 (LifecycleStateMachine),
                启用后 review/unarchive 等操作校验状态转移。
            emit_events: 是否发射生命周期事件 (默认 True)。
            guard_archived: 是否拒绝在归档状态下编辑 (默认 True)。
        """
        self._artifacts: dict[str, Artifact] = {}
        self._version_trees: dict[str, VersionTree] = {}
        self._lock = threading.RLock()
        self._store = store
        self._state_machine = state_machine
        self._emit_events = emit_events
        self._guard_archived = guard_archived
        self._search_engine: Any = None

    # ----------------------------------------------------------
    # 生命周期管理
    # ----------------------------------------------------------

    def register(self, artifact: Artifact) -> Artifact:
        """注册 Artifact, 创建版本树根节点.

        设置状态为 CREATED, 版本号为 1, 创建 VersionTree。
        重复注册幂等: 已存在的 Artifact 保留原版本树与版本号。

        Args:
            artifact: 待注册的 Artifact

        Returns:
            注册后的 Artifact (状态为 CREATED)

        Raises:
            ArtifactValidationError: Artifact 为 None
        """
        if artifact is None:
            raise ArtifactValidationError(field="artifact", detail="Artifact is None")
        with self._lock:
            existing = self._artifacts.get(artifact.artifact_id)
            if existing is not None:
                # 幂等: 保留已有对象, 不重置版本树
                return existing

            artifact.state = ArtifactLifecycleState.CREATED
            artifact.version = 1
            artifact.updated_at = time.time()

            tree = VersionTree(artifact)
            self._version_trees[artifact.artifact_id] = tree
            self._artifacts[artifact.artifact_id] = artifact

            self._sync_store(artifact)
            self._sync_index(artifact)
            self._emit(ARTIFACT_REGISTERED, artifact, {"version": 1})
            return artifact

    def get(self, artifact_id: str, version: int | None = None) -> Artifact:
        """获取 Artifact (默认最新主线版本, 可指定版本).

        Args:
            artifact_id: Artifact ID
            version: 版本号 (None 表示最新主线版本)

        Returns:
            Artifact 对象

        Raises:
            ArtifactNotFoundError: Artifact 或版本不存在
        """
        with self._lock:
            if version is not None:
                tree = self._version_trees.get(artifact_id)
                if tree is None:
                    raise ArtifactNotFoundError(artifact_id)
                art = tree.get_artifact(version)
                if art is None:
                    raise ArtifactNotFoundError(artifact_id)
                return art

            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)
            return art

    def update(
        self,
        artifact_id: str,
        new_payload: dict[str, Any],
        edit_reason: str = "",
        expected_version: int | None = None,
    ) -> Artifact:
        """更新 Artifact (创建新版本, ADK 模式).

        不可变设计: 不修改原版本, 而是创建新版本。
        新版本继承原 Artifact 的元数据, 更新 payload。
        更新主线 head 指针。

        乐观版本锁 (向后兼容):
        - expected_version=None (默认): 不做版本检查, 行为与原有实现一致。
        - expected_version 为整数: 校验当前主线版本号是否等于该值,
          不匹配时抛出 VersionConflictError (携带实际版本号)。

        Args:
            artifact_id: Artifact ID
            new_payload: 新的 payload 数据
            edit_reason: 编辑原因
            expected_version: 期望的当前版本号 (乐观锁), None 表示不检查

        Returns:
            新版本的 Artifact

        Raises:
            ArtifactNotFoundError: Artifact 不存在
            VersionConflictError: expected_version 与当前版本不匹配
        """
        with self._lock:
            current = self._artifacts.get(artifact_id)
            if current is None:
                raise ArtifactNotFoundError(artifact_id)

            self._guard_archived_state(current)

            if expected_version is not None and current.version != expected_version:
                raise VersionConflictError(
                    artifact_id,
                    current.version,
                    detail=(
                        f"Version conflict: expected {expected_version}, "
                        f"but current version is {current.version}"
                    ),
                )

            tree = self._version_trees[artifact_id]
            new_version = tree.get_latest_version() + 1

            new_artifact = current.model_copy(update={
                "version": new_version,
                "payload": dict(new_payload),
                "updated_at": time.time(),
                "state": ArtifactLifecycleState.EDITED,
            })

            tree.add_version(new_artifact, parent_version=current.version)
            self._artifacts[artifact_id] = new_artifact
            self._sync_store(new_artifact)
            self._sync_index(new_artifact)
            self._emit(ARTIFACT_UPDATED, new_artifact, {"version": new_version, "edit_reason": edit_reason})
            return new_artifact

    def fork(self, artifact_id: str, fork_reason: str = "manual") -> Artifact:
        """Fork Artifact (创建分支版本).

        从当前主线 head 创建分支版本, 不移动 head 指针。
        分支版本带有 fork_origin 标记。

        Args:
            artifact_id: Artifact ID
            fork_reason: Fork 原因 (存储为 fork_origin)

        Returns:
            Fork 后的 Artifact

        Raises:
            ArtifactNotFoundError: Artifact 不存在
        """
        with self._lock:
            current = self._artifacts.get(artifact_id)
            if current is None:
                raise ArtifactNotFoundError(artifact_id)

            self._guard_archived_state(current)

            tree = self._version_trees[artifact_id]
            new_version = tree.get_latest_version() + 1

            forked = current.model_copy(update={
                "version": new_version,
                "fork_origin": fork_reason,
                "updated_at": time.time(),
            })

            tree.add_version(forked, parent_version=current.version)
            # 不更新 head — fork 创建分支, 不移动主线
            return forked

    def archive(self, artifact_id: str) -> bool:
        """归档 Artifact (设置状态为 ARCHIVED).

        设计文档 Ch.3.2: Rendered/Reviewed → Archived (会话结束归档)。
        为兼容既有行为采用宽松模式 (允许任意状态归档)。

        Args:
            artifact_id: Artifact ID

        Returns:
            True 表示归档成功

        Raises:
            ArtifactNotFoundError: Artifact 不存在
        """
        with self._lock:
            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)
            art.state = ArtifactLifecycleState.ARCHIVED
            art.updated_at = time.time()
            self._sync_store(art)
            self._emit(ARTIFACT_ARCHIVED, art, {"archived": True})
            return True

    def list_artifacts(
        self,
        session_id: str | None = None,
        artifact_type: ArtifactType | str | None = None,
        source_agent: str | None = None,
        kp_id: str | None = None,
    ) -> list[Artifact]:
        """列出活跃 (非归档) 的 Artifact, 支持多维过滤.

        Args:
            session_id: 按会话 ID 过滤
            artifact_type: 按类型过滤 (ArtifactType 或字符串)
            source_agent: 按来源 Agent 过滤
            kp_id: 按知识点 ID 过滤

        Returns:
            匹配的 Artifact 列表 (仅活跃, 不含归档)
        """
        with self._lock:
            results: list[Artifact] = []
            for art in self._artifacts.values():
                # 排除归档
                if art.state == ArtifactLifecycleState.ARCHIVED:
                    continue
                # 会话过滤
                if session_id is not None and art.session_id != session_id:
                    continue
                # 类型过滤
                if artifact_type is not None and art.type != artifact_type:
                    continue
                # 来源 Agent 过滤
                if source_agent is not None and art.source_agent != source_agent:
                    continue
                # 知识点过滤
                if kp_id is not None and not self._matches_kp_id(art, kp_id):
                    continue
                results.append(art)
            return results

    # ----------------------------------------------------------
    # 版本管理 (DAG)
    # ----------------------------------------------------------

    def get_version_history(self, artifact_id: str) -> list[ArtifactVersionNode]:
        """获取版本历史 (拓扑排序).

        Args:
            artifact_id: Artifact ID

        Returns:
            拓扑排序的 ArtifactVersionNode 列表

        Raises:
            ArtifactNotFoundError: Artifact 不存在
        """
        with self._lock:
            tree = self._version_trees.get(artifact_id)
            if tree is None:
                raise ArtifactNotFoundError(artifact_id)
            return tree.get_all_versions()

    def get_latest_version(self, artifact_id: str) -> int:
        """获取最高版本号.

        Args:
            artifact_id: Artifact ID

        Returns:
            最高版本号

        Raises:
            ArtifactNotFoundError: Artifact 不存在
        """
        with self._lock:
            tree = self._version_trees.get(artifact_id)
            if tree is None:
                raise ArtifactNotFoundError(artifact_id)
            return tree.get_latest_version()

    def get_version(self, artifact_id: str, version: int) -> Artifact | None:
        """获取指定版本的 Artifact.

        Args:
            artifact_id: Artifact ID
            version: 版本号

        Returns:
            Artifact, 不存在则返回 None

        Raises:
            ArtifactNotFoundError: Artifact 不存在
        """
        with self._lock:
            tree = self._version_trees.get(artifact_id)
            if tree is None:
                raise ArtifactNotFoundError(artifact_id)
            return tree.get_artifact(version)

    # ----------------------------------------------------------
    # 编辑管理 (RFC 6902 JSON Patch)
    # ----------------------------------------------------------

    def apply_edit(self, artifact_id: str, diff: ArtifactDiff) -> Artifact:
        """应用 ArtifactDiff 到 Artifact (RFC 6902 JSON Patch 风格).

        支持的完整 RFC 6902 操作:
        - add:     {"op": "add", "path": "/key", "value": val}
        - remove:  {"op": "remove", "path": "/key"}
        - replace: {"op": "replace", "path": "/key", "value": val}
        - move:    {"op": "move", "path": "/target", "from": "/source"}
        - copy:    {"op": "copy", "path": "/target", "from": "/source"}
        - test:    {"op": "test", "path": "/key", "value": expected}

        路径格式 (向后兼容):
        - JSON Pointer: 以 "/" 开头, 如 "/data/0/name" 解析为
          payload["data"][0]["name"].
        - 扁平 key: 不以 "/" 开头, 如 "content" 直接作为顶层键.

        ops 元素支持 dict (RFC 6902 JSON Patch 风格) 和 DiffOp 对象.
        创建新版本, 更新主线 head.

        Args:
            artifact_id: Artifact ID
            diff: ArtifactDiff 增量差异

        Returns:
            编辑后的新版本 Artifact

        Raises:
            ArtifactNotFoundError: Artifact 不存在
            ArtifactNotEditableError: Artifact 不可编辑 (editable=False)
            ArtifactValidationError: test 失败 / 路径无法解析 / 未知 op
        """
        with self._lock:
            current = self._artifacts.get(artifact_id)
            if current is None:
                raise ArtifactNotFoundError(artifact_id)

            if not current.editable:
                raise ArtifactNotEditableError(artifact_id)

            self._guard_archived_state(current)

            # 深拷贝 payload, 避免嵌套编辑破坏版本树中的历史快照.
            new_payload = copy.deepcopy(current.payload)
            for op in diff.ops:
                self._apply_op(new_payload, self._normalize_op(op))

            tree = self._version_trees[artifact_id]
            new_version = tree.get_latest_version() + 1

            edited = current.model_copy(update={
                "version": new_version,
                "payload": new_payload,
                "updated_at": time.time(),
                "state": ArtifactLifecycleState.EDITED,
            })

            tree.add_version(edited, parent_version=current.version)
            self._artifacts[artifact_id] = edited
            self._sync_store(edited)
            self._sync_index(edited)
            self._emit(ARTIFACT_UPDATED, edited, {"version": new_version, "edit_reason": diff.edit_reason})
            return edited

    def get_diff(
        self,
        artifact_id: str,
        from_version: int,
        to_version: int,
    ) -> ArtifactDiff:
        """获取两个版本之间的差异 (ArtifactDiff).

        对比两个版本的 payload, 生成 RFC 6902 风格的操作列表:
        - add: 新增的键
        - remove: 删除的键
        - replace: 值变更的键

        Args:
            artifact_id: Artifact ID
            from_version: 源版本号
            to_version: 目标版本号

        Returns:
            ArtifactDiff 差异对象

        Raises:
            ArtifactNotFoundError: Artifact 或版本不存在
        """
        with self._lock:
            tree = self._version_trees.get(artifact_id)
            if tree is None:
                raise ArtifactNotFoundError(artifact_id)

            from_art = tree.get_artifact(from_version)
            to_art = tree.get_artifact(to_version)
            if from_art is None or to_art is None:
                raise ArtifactNotFoundError(artifact_id)

            ops = self._compute_diff(from_art.payload, to_art.payload)
            return ArtifactDiff(
                artifact_id=artifact_id,
                ops=ops,
                edit_reason=f"diff v{from_version} -> v{to_version}",
            )

    # ----------------------------------------------------------
    # 搜索与过滤
    # ----------------------------------------------------------

    def search(self, query: str) -> list[Artifact]:
        """全文搜索 Artifact (title + payload).

        在活跃 (非归档) Artifact 的 title 和 payload 中搜索。
        结果按相关性排序 (匹配次数多的在前)。

        Args:
            query: 搜索关键词

        Returns:
            匹配的 Artifact 列表 (按相关性降序)
        """
        with self._lock:
            query_lower = query.lower()
            scored: list[tuple[int, Artifact]] = []

            for art in self._artifacts.values():
                if art.state == ArtifactLifecycleState.ARCHIVED:
                    continue

                score = 0
                # 搜索 title
                if art.title:
                    title_lower = art.title.lower()
                    score += title_lower.count(query_lower)

                # 搜索 payload
                payload_str = str(art.payload).lower()
                score += payload_str.count(query_lower)

                if score > 0:
                    scored.append((score, art))

            # 按相关性降序排序
            scored.sort(key=lambda x: x[0], reverse=True)
            return [art for _, art in scored]

    # ----------------------------------------------------------
    # 统计信息
    # ----------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息.

        返回:
        - total: Artifact 总数
        - by_type: 按类型分组的计数
        - by_state: 按状态分组的计数
        """
        with self._lock:
            by_type: dict[str, int] = {}
            by_state: dict[str, int] = {}

            for art in self._artifacts.values():
                type_key = art.type.value
                state_key = art.state.value

                by_type[type_key] = by_type.get(type_key, 0) + 1
                by_state[state_key] = by_state.get(state_key, 0) + 1

            return {
                "total": len(self._artifacts),
                "by_type": by_type,
                "by_state": by_state,
            }

    # ----------------------------------------------------------
    # T3 增强: 审核 / 恢复 / 删除 / 合并
    # ----------------------------------------------------------

    def review(self, artifact_id: str, reviewer: str = "cc1.actor_critic", review_data: dict[str, Any] | None = None) -> Artifact:
        """审核 Artifact (CC1) — 状态转移 Rendered → Reviewed.

        设计文档 Ch.3.2: 由 CC1 Actor-Critic Reviewer 标记为"需审查"的
        Artifact 进入 Reviewed 状态, 审查通过后才正式呈现。

        Args:
            artifact_id: Artifact ID
            reviewer: 审查者标识。
            review_data: 审查结果 (评分/建议等)。

        Returns:
            审核后的 Artifact (状态 REVIEWED)

        Raises:
            ArtifactNotFoundError: Artifact 不存在
            ArtifactValidationError: 状态不允许审核 (非 Rendered)
        """
        with self._lock:
            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)
            self._transition_state(art, ArtifactLifecycleState.REVIEWED)
            art.updated_at = time.time()
            learner = dict(art.learner_context or {})
            review_meta: dict[str, Any] = {"reviewer": reviewer}
            if review_data:
                review_meta["data"] = review_data
            learner["review"] = review_meta
            art.learner_context = learner
            self._sync_store(art)
            self._emit(ARTIFACT_REVIEWED, art, {"reviewer": reviewer})
            return art

    def unarchive(self, artifact_id: str) -> Artifact:
        """恢复归档 Artifact — 状态转移 Archived → Rendered.

        S3 delete marker 软删语义的逆操作: 归档不删数据, 可恢复。
        设计文档 Archive 阶段: 归档数据可在"历史会话回顾"中重新加载。

        Args:
            artifact_id: Artifact ID

        Returns:
            恢复后的 Artifact (状态 RENDERED)

        Raises:
            ArtifactNotFoundError: Artifact 不存在
            ArtifactValidationError: 非归档状态
        """
        with self._lock:
            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)
            if art.state != ArtifactLifecycleState.ARCHIVED:
                raise ArtifactValidationError(
                    field="state",
                    detail=f"Artifact {artifact_id} 当前状态为 {art.state.value}, 仅归档状态可恢复",
                )
            self._transition_state(art, ArtifactLifecycleState.RENDERED)
            art.updated_at = time.time()
            self._sync_store(art)
            self._sync_index(art)
            self._emit(ARTIFACT_RESTORED, art, {"restored": True})
            return art

    def remove(self, artifact_id: str, hard: bool = False) -> bool:
        """移除 Artifact (软删默认).

        软删 (hard=False): 从活跃列表移除, 保留版本树与历史快照
        (S3 delete marker 语义, 可恢复)。
        硬删 (hard=True): 同时清除版本树。

        Args:
            artifact_id: Artifact ID
            hard: 是否硬删除 (清除版本树)。

        Returns:
            True 表示移除成功。

        Raises:
            ArtifactNotFoundError: Artifact 不存在
        """
        with self._lock:
            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)
            self._artifacts.pop(artifact_id, None)
            if hard:
                self._version_trees.pop(artifact_id, None)
            if self._store is not None:
                try:
                    self._store.delete(artifact_id)
                except Exception:  # noqa: BLE001
                    pass
            self._remove_from_index(artifact_id)
            self._emit(ARTIFACT_REMOVED, art, {"hard": hard})
            return True

    def merge(
        self,
        artifact_id: str,
        branch_version: int,
        strategy: str = "auto",
        merge_reason: str = "merge",
    ) -> Artifact:
        """合并分支版本回主线 (Git merge commit 语义).

        设计文档 Ch.3.3: 分支被合并回主线产生合并版本 v4,
        合并结果成为新主线 head。

        Args:
            artifact_id: Artifact ID
            branch_version: 分支版本号。
            strategy: auto (三方合并冲突检测) / ours / theirs。
            merge_reason: 合并原因。

        Returns:
            合并后的新版本 Artifact (成为新 head)。

        Raises:
            ArtifactNotFoundError: Artifact 不存在。
            ArtifactValidationError: 版本不存在或合并冲突。
        """
        with self._lock:
            head = self._artifacts.get(artifact_id)
            if head is None:
                raise ArtifactNotFoundError(artifact_id)
            tree = self._version_trees.get(artifact_id)
            if tree is None:
                raise ArtifactNotFoundError(artifact_id)
            branch_snap = tree.get_artifact(branch_version)
            if branch_snap is None:
                raise ArtifactValidationError(
                    field="version",
                    detail=f"分支版本 v{branch_version} 不存在",
                )

            # 三方合并冲突检测 (auto 策略)
            conflicts: list[str] = []
            if strategy == "auto":
                branch_node = tree.get_node(branch_version)
                lineage = [n.version for n in tree.get_lineage(branch_version)] if branch_node else []
                # 最近公共祖先: 沿主线 head 与分支共同回溯
                base = self._find_common_ancestor(artifact_id, head.version, branch_version)
                if base is not None:
                    base_snap = tree.get_artifact(base)
                    if base_snap is not None:
                        conflicts = self._detect_payload_conflicts(
                            base_snap.payload, head.payload, branch_snap.payload
                        )
                if conflicts:
                    raise ArtifactValidationError(
                        field="merge",
                        detail=f"合并冲突字段: {conflicts} (可选用 ours/theirs 策略)",
                    )
                merged_payload = dict(branch_snap.payload)
            elif strategy == "ours":
                merged_payload = dict(head.payload)
            else:  # theirs
                merged_payload = dict(branch_snap.payload)

            # 生成合并版本 (新 head)
            new_version = tree.get_latest_version() + 1
            merged = head.model_copy(update={
                "version": new_version,
                "payload": merged_payload,
                "updated_at": time.time(),
                "state": ArtifactLifecycleState.EDITED,
                "fork_origin": f"merge:main-{head.version}<-branch-{branch_version}",
            })
            tree.add_version(merged, parent_version=head.version)
            self._artifacts[artifact_id] = merged
            self._sync_store(merged)
            self._sync_index(merged)
            self._emit(ARTIFACT_MERGED, merged, {
                "branch_version": branch_version,
                "main_version": head.version,
                "strategy": strategy,
            })
            return merged

    # ----------------------------------------------------------
    # T3 增强: 快照持久化
    # ----------------------------------------------------------

    def save_snapshot(self, path: str) -> int:
        """将当前全部 Artifact 保存为快照 (进程重启恢复).

        Args:
            path: 快照文件路径。

        Returns:
            保存的 Artifact 数量。
        """
        with self._lock:
            if self._store is not None and hasattr(self._store, "save_snapshot"):
                return self._store.save_snapshot(path)
            # 无 store 时使用内置 JSON 序列化
            import json
            import os

            payload = {
                "artifacts": {aid: art.to_dict() for aid, art in self._artifacts.items()},
                "version_trees": {
                    aid: {
                        "versions": [v.to_dict() for v in tree.get_all_versions()],
                    }
                    for aid, tree in self._version_trees.items()
                },
            }
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, default=str)
            return len(self._artifacts)

    def load_snapshot(self, path: str) -> int:
        """从快照恢复 Artifact.

        Args:
            path: 快照文件路径。

        Returns:
            恢复的 Artifact 数量。

        Raises:
            FileNotFoundError: 快照文件不存在。
        """
        with self._lock:
            if self._store is not None and hasattr(self._store, "load_snapshot"):
                count = self._store.load_snapshot(path)
                # 同步回内存层
                for art in self._store.list():
                    self._artifacts[art.artifact_id] = art
                    if art.artifact_id not in self._version_trees:
                        tree = VersionTree(art)
                        # 尝试重建版本链 (从 store 中仅存 head, 简化重建)
                        self._version_trees[art.artifact_id] = tree
                    self._sync_index(art)
                return count

            import json
            import os

            if not os.path.exists(path):
                raise FileNotFoundError(f"快照文件不存在: {path}")
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            count = 0
            for aid, data in (payload.get("artifacts") or {}).items():
                try:
                    art = Artifact.model_validate(data)
                except Exception:  # noqa: BLE001 — 单条损坏不阻断恢复
                    continue
                if aid not in self._version_trees:
                    tree = VersionTree(art)
                    self._version_trees[aid] = tree
                self._artifacts[aid] = art
                self._sync_index(art)
                count += 1
            return count

    # ----------------------------------------------------------
    # T3 增强: 全文搜索 (倒排索引)
    # ----------------------------------------------------------

    def fulltext_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        sort_by: str = "-created_at",
    ) -> list[Artifact]:
        """全文搜索 + 结构化过滤 (倒排索引).

        支持布尔查询 (AND/OR/NOT)、引号精确短语、结构化过滤
        (type/source_agent/kp_id/时间/edited_only)。

        Args:
            query: 查询字符串。
            filters: 结构化过滤条件。
            sort_by: 排序字段 ("-" 前缀降序)。

        Returns:
            匹配的 Artifact 列表。
        """
        engine = self._ensure_search_engine()
        return engine.search(query, filters, sort_by)

    def related_by_kp(
        self, kp_id: str, max_depth: int = 2, kp_graph: dict[str, set[str]] | None = None
    ) -> list[Artifact]:
        """学情关联搜索 (设计文档 Ch.3.6).

        输入 KP ID, 找到直接关联 + 知识图谱间接关联的 Artifact。

        Args:
            kp_id: 目标知识点 ID。
            max_depth: 知识图谱间接关联最大跳数。
            kp_graph: 知识图谱邻接表 (提供时支持间接关联)。

        Returns:
            关联 Artifact 列表 (直接关联优先)。
        """
        engine = self._ensure_search_engine()
        return engine.related_by_kp(kp_id, max_depth, kp_graph)

    # ----------------------------------------------------------
    # 内部工具方法 (T3 增强)
    # ----------------------------------------------------------

    def _guard_archived_state(self, artifact: Artifact) -> None:
        """防护: 归档状态下拒绝编辑 (设计文档 Ch.3.2)."""
        if self._guard_archived and artifact.state == ArtifactLifecycleState.ARCHIVED:
            raise ArtifactValidationError(
                field="state",
                detail=f"Artifact {artifact.artifact_id} 已归档, 不可编辑",
            )

    def _transition_state(self, artifact: Artifact, target: ArtifactLifecycleState) -> None:
        """状态转移 (校验合法性, 无状态机时宽松执行)."""
        if self._state_machine is not None:
            self._state_machine.transition(artifact.state, target)
        artifact.state = target

    def _emit(self, event_type: str, artifact: Artifact, data: dict[str, Any]) -> None:
        """发射 Artifact 生命周期事件 (错误隔离)."""
        if not self._emit_events:
            return
        try:
            get_global_emitter().emit(event_type, artifact.artifact_id, **data)
        except Exception:  # noqa: BLE001 — 事件失败不阻断业务
            pass

    def _sync_store(self, artifact: Artifact) -> None:
        """同步写入存储层 (写穿)."""
        if self._store is not None:
            try:
                self._store.save(artifact)
            except Exception:  # noqa: BLE001 — 存储失败不阻断内存操作
                pass

    def _ensure_search_engine(self) -> Any:
        """懒初始化倒排索引搜索引擎."""
        if self._search_engine is None:
            from .artifact.search import SearchEngine

            engine = SearchEngine()
            engine.reindex(self._artifacts.values())
            self._search_engine = engine
        return self._search_engine

    def _sync_index(self, artifact: Artifact) -> None:
        """增量同步搜索索引."""
        if self._search_engine is not None:
            try:
                self._search_engine.add(artifact)
            except Exception:  # noqa: BLE001
                pass

    def _remove_from_index(self, artifact_id: str) -> None:
        """从搜索索引移除."""
        if self._search_engine is not None:
            try:
                self._search_engine.remove(artifact_id)
            except Exception:  # noqa: BLE001
                pass

    def _find_common_ancestor(self, artifact_id: str, v1: int, v2: int) -> int | None:
        """最近公共祖先 (三方合并 base 定位)."""
        tree = self._version_trees.get(artifact_id)
        if tree is None:
            return None
        lineage1 = [n.version for n in tree.get_lineage(v1)]
        lineage2 = [n.version for n in tree.get_lineage(v2)]
        set1 = set(lineage1)
        for v in reversed(lineage2):
            if v in set1:
                return v
        return None

    def _detect_payload_conflicts(
        self, base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any]
    ) -> list[str]:
        """三方合并冲突检测 (顶层键级)."""
        conflicts: list[str] = []
        all_keys = set(base) | set(ours) | set(theirs)
        for key in all_keys:
            bv = base.get(key, _TEST_MISSING)
            ov = ours.get(key, _TEST_MISSING)
            tv = theirs.get(key, _TEST_MISSING)
            ours_changed = ov != bv
            theirs_changed = tv != bv
            if ours_changed and theirs_changed and ov != tv:
                conflicts.append(key)
        return conflicts

    # ----------------------------------------------------------
    # 内部工具方法
    # ----------------------------------------------------------

    @staticmethod
    def _matches_kp_id(artifact: Artifact, kp_id: str) -> bool:
        """检查 Artifact 是否关联指定知识点 ID.

        检查位置:
        - payload["kp_id"]
        - payload["kp_gaps"] (列表)
        - payload["kp_ids"] (列表)
        - learner_context["kp_id"]

        Args:
            artifact: Artifact 对象
            kp_id: 知识点 ID

        Returns:
            True 如果 Artifact 关联该知识点
        """
        if artifact.payload.get("kp_id") == kp_id:
            return True
        if kp_id in artifact.payload.get("kp_gaps", []):
            return True
        if kp_id in artifact.payload.get("kp_ids", []):
            return True
        if artifact.learner_context.get("kp_id") == kp_id:
            return True
        return False

    @staticmethod
    def _compute_diff(
        from_payload: dict[str, Any],
        to_payload: dict[str, Any],
        prefix: str = "",
    ) -> list[dict[str, Any]]:
        """计算两个 payload 之间的差异 (RFC 6902 风格, 支持嵌套).

        递归比较 dict / list, 生成 JSON Pointer 风格的操作列表:
        - add:     新增的键 / 列表元素
        - remove:  删除的键 / 列表元素
        - replace: 值变更 (基本类型或类型变更)

        路径格式 (向后兼容):
        - 顶层 (prefix=""): 扁平 key, 如 "confidence" (保持与旧实现兼容).
        - 嵌套 (prefix 非空): JSON Pointer, 如 "/data/config/enabled".

        Args:
            from_payload: 源 payload
            to_payload: 目标 payload
            prefix: 当前递归路径前缀 (顶层为 "")

        Returns:
            操作列表 (add/remove/replace)
        """
        ops: list[dict[str, Any]] = []

        # dict 比较: 递归每个 key
        if isinstance(from_payload, dict) and isinstance(to_payload, dict):
            all_keys = sorted(set(from_payload.keys()) | set(to_payload.keys()))
            for key in all_keys:
                path, child_prefix = ArtifactManager._build_paths(prefix, key)
                if key not in from_payload:
                    # 新增: 整体添加 (不展开子结构)
                    ops.append({"op": "add", "path": path, "value": to_payload[key]})
                elif key not in to_payload:
                    # 删除: 整体移除 (不展开子结构)
                    ops.append({"op": "remove", "path": path})
                else:
                    old_val = from_payload[key]
                    new_val = to_payload[key]
                    if isinstance(old_val, dict) and isinstance(new_val, dict):
                        ops.extend(
                            ArtifactManager._compute_diff(old_val, new_val, child_prefix)
                        )
                    elif isinstance(old_val, list) and isinstance(new_val, list):
                        ops.extend(
                            ArtifactManager._compute_diff(old_val, new_val, child_prefix)
                        )
                    elif old_val != new_val:
                        # 基本类型变更或类型变更 (dict -> str 等)
                        ops.append({"op": "replace", "path": path, "value": new_val})

        # list 比较: 按索引比较
        elif isinstance(from_payload, list) and isinstance(to_payload, list):
            max_len = max(len(from_payload), len(to_payload))
            for i in range(max_len):
                path, child_prefix = ArtifactManager._build_paths(prefix, i)
                if i >= len(from_payload):
                    ops.append({"op": "add", "path": path, "value": to_payload[i]})
                elif i >= len(to_payload):
                    ops.append({"op": "remove", "path": path})
                else:
                    old_val = from_payload[i]
                    new_val = to_payload[i]
                    if isinstance(old_val, dict) and isinstance(new_val, dict):
                        ops.extend(
                            ArtifactManager._compute_diff(old_val, new_val, child_prefix)
                        )
                    elif isinstance(old_val, list) and isinstance(new_val, list):
                        ops.extend(
                            ArtifactManager._compute_diff(old_val, new_val, child_prefix)
                        )
                    elif old_val != new_val:
                        ops.append({"op": "replace", "path": path, "value": new_val})

        # 类型不匹配 (如 dict -> list) 在父层级以 replace 处理, 此处不产生 ops.
        return ops

    # ----------------------------------------------------------
    # RFC 6902 JSON Pointer 与操作辅助方法
    # ----------------------------------------------------------

    @staticmethod
    def _build_paths(prefix: str, key: Any) -> tuple[str, str]:
        """构建当前层路径与子层前缀.

        - 顶层 (prefix=""): path = 扁平 key (向后兼容),
          child_prefix = "/" + key (子层使用 JSON Pointer).
        - 嵌套 (prefix 非空): path = prefix + "/" + key,
          child_prefix = path.

        Args:
            prefix: 当前前缀
            key: 当前键或索引

        Returns:
            (当前路径, 子层前缀)
        """
        key_str = str(key)
        if prefix == "":
            return key_str, "/" + key_str
        return prefix + "/" + key_str, prefix + "/" + key_str

    @staticmethod
    def _is_pointer(path: Any) -> bool:
        """判断路径是否为 JSON Pointer (以 "/" 开头的字符串)."""
        return isinstance(path, str) and path.startswith("/")

    @staticmethod
    def _parse_pointer(path: str) -> list[str]:
        """解析 JSON Pointer 路径为 token 列表 (RFC 6901).

        转义规则: "~1" -> "/", "~0" -> "~".

        Args:
            path: JSON Pointer 字符串 (以 "/" 开头)

        Returns:
            token 列表
        """
        # 以 "/" 分割, 首段为空 (前导 "/")
        tokens = path.split("/")[1:]
        return [t.replace("~1", "/").replace("~0", "~") for t in tokens]

    @staticmethod
    def _list_index(token: str, path: str) -> int:
        """将 token 转换为非负列表索引.

        Args:
            token: token 字符串
            path: 完整路径 (用于错误信息)

        Returns:
            非负整数索引

        Raises:
            ArtifactValidationError: token 不是合法非负整数
        """
        try:
            idx = int(token)
        except (ValueError, TypeError):
            raise ArtifactValidationError(
                field=path,
                detail=f"Invalid list index '{token}' in pointer '{path}'",
            )
        if idx < 0:
            raise ArtifactValidationError(
                field=path,
                detail=f"Negative index {idx} in pointer '{path}'",
            )
        return idx

    @staticmethod
    def _resolve_pointer(payload: Any, path: str) -> Any:
        """按 JSON Pointer 解析路径, 返回对应值.

        Args:
            payload: 数据根
            path: JSON Pointer 路径

        Returns:
            路径处的值

        Raises:
            ArtifactValidationError: 路径无法解析 (键/索引不存在或类型不匹配)
        """
        tokens = ArtifactManager._parse_pointer(path)
        current: Any = payload
        for token in tokens:
            if isinstance(current, list):
                if token == "-":
                    raise ArtifactValidationError(
                        field=path,
                        detail=f"'-' cannot be resolved in pointer '{path}'",
                    )
                idx = ArtifactManager._list_index(token, path)
                if idx >= len(current):
                    raise ArtifactValidationError(
                        field=path,
                        detail=f"Index {idx} out of range in pointer '{path}'",
                    )
                current = current[idx]
            elif isinstance(current, dict):
                if token not in current:
                    raise ArtifactValidationError(
                        field=path,
                        detail=f"Key '{token}' not found in pointer '{path}'",
                    )
                current = current[token]
            else:
                raise ArtifactValidationError(
                    field=path,
                    detail=f"Cannot traverse '{token}' in pointer '{path}' (not a container)",
                )
        return current

    @staticmethod
    def _set_pointer(payload: Any, path: str, value: Any) -> None:
        """按 JSON Pointer 设置值 (原地修改).

        支持向 dict 添加新键, 向 list 设置/追加元素 ("-" 表示追加).

        Args:
            payload: 数据根
            path: JSON Pointer 路径
            value: 待设置值

        Raises:
            ArtifactValidationError: 中间路径无法遍历或类型不匹配
        """
        tokens = ArtifactManager._parse_pointer(path)
        if not tokens:
            raise ArtifactValidationError(field=path, detail="Empty pointer path")

        current: Any = payload
        for token in tokens[:-1]:
            if isinstance(current, list):
                idx = ArtifactManager._list_index(token, path)
                if idx >= len(current):
                    raise ArtifactValidationError(
                        field=path,
                        detail=f"Index {idx} out of range in pointer '{path}'",
                    )
                current = current[idx]
            elif isinstance(current, dict):
                if token not in current:
                    raise ArtifactValidationError(
                        field=path,
                        detail=f"Key '{token}' not found in pointer '{path}'",
                    )
                current = current[token]
            else:
                raise ArtifactValidationError(
                    field=path,
                    detail=f"Cannot traverse '{token}' in pointer '{path}' (not a container)",
                )

        last = tokens[-1]
        if isinstance(current, list):
            if last == "-":
                current.append(value)
            else:
                idx = ArtifactManager._list_index(last, path)
                # 自动扩展列表 (用 None 填充间隙)
                while len(current) <= idx:
                    current.append(None)
                current[idx] = value
        elif isinstance(current, dict):
            current[last] = value
        else:
            raise ArtifactValidationError(
                field=path,
                detail=f"Cannot set '{last}' in pointer '{path}' (not a container)",
            )

    @staticmethod
    def _remove_pointer(payload: Any, path: str) -> None:
        """按 JSON Pointer 删除值 (原地修改).

        Args:
            payload: 数据根
            path: JSON Pointer 路径

        Raises:
            ArtifactValidationError: 中间路径无法遍历
        """
        tokens = ArtifactManager._parse_pointer(path)
        if not tokens:
            raise ArtifactValidationError(field=path, detail="Empty pointer path")

        current: Any = payload
        for token in tokens[:-1]:
            if isinstance(current, list):
                idx = ArtifactManager._list_index(token, path)
                if idx >= len(current):
                    raise ArtifactValidationError(
                        field=path,
                        detail=f"Index {idx} out of range in pointer '{path}'",
                    )
                current = current[idx]
            elif isinstance(current, dict):
                if token not in current:
                    raise ArtifactValidationError(
                        field=path,
                        detail=f"Key '{token}' not found in pointer '{path}'",
                    )
                current = current[token]
            else:
                raise ArtifactValidationError(
                    field=path,
                    detail=f"Cannot traverse '{token}' in pointer '{path}' (not a container)",
                )

        last = tokens[-1]
        if isinstance(current, list):
            idx = ArtifactManager._list_index(last, path)
            if 0 <= idx < len(current):
                current.pop(idx)
        elif isinstance(current, dict):
            current.pop(last, None)
        else:
            raise ArtifactValidationError(
                field=path,
                detail=f"Cannot remove '{last}' in pointer '{path}' (not a container)",
            )

    # ----------------------------------------------------------
    # 统一读写辅助 (兼容 JSON Pointer 与扁平 key)
    # ----------------------------------------------------------

    @staticmethod
    def _get_value(payload: Any, path: str) -> Any:
        """获取路径处的值 (兼容 JSON Pointer 与扁平 key).

        Raises:
            ArtifactValidationError: 路径不存在
        """
        if ArtifactManager._is_pointer(path):
            return ArtifactManager._resolve_pointer(payload, path)
        if not isinstance(payload, dict) or path not in payload:
            raise ArtifactValidationError(
                field=str(path),
                detail=f"Path not found: {path}",
            )
        return payload[path]

    @staticmethod
    def _set_value(payload: Any, path: str, value: Any) -> None:
        """设置路径处的值 (兼容 JSON Pointer 与扁平 key)."""
        if ArtifactManager._is_pointer(path):
            ArtifactManager._set_pointer(payload, path, value)
        else:
            payload[path] = value

    @staticmethod
    def _remove_value(payload: Any, path: str) -> None:
        """删除路径处的值 (兼容 JSON Pointer 与扁平 key)."""
        if ArtifactManager._is_pointer(path):
            ArtifactManager._remove_pointer(payload, path)
        else:
            payload.pop(path, None)

    # ----------------------------------------------------------
    # 操作归一化与应用
    # ----------------------------------------------------------

    @staticmethod
    def _normalize_op(op: Any) -> dict[str, Any]:
        """将 op 归一化为 dict (兼容 dict 与 DiffOp 对象).

        Args:
            op: dict 或 DiffOp 对象

        Returns:
            包含 op / path / value (及可选 from) 的 dict

        Raises:
            ArtifactValidationError: 不支持的 op 类型
        """
        if isinstance(op, dict):
            return op
        if isinstance(op, DiffOp):
            op_val = op.op.value if isinstance(op.op, DiffOpType) else str(op.op)
            result: dict[str, Any] = {"op": op_val, "path": op.path}
            if hasattr(op, "value"):
                result["value"] = op.value
            if hasattr(op, "from"):
                result["from"] = getattr(op, "from")
            return result
        # 兜底: 尝试属性访问
        op_val = getattr(op, "op", None)
        if op_val is not None:
            if hasattr(op_val, "value"):
                op_val = op_val.value
            result = {"op": op_val, "path": getattr(op, "path", None)}
            for attr in ("value", "from"):
                if hasattr(op, attr):
                    result[attr] = getattr(op, attr)
            return result
        raise ArtifactValidationError(
            detail=f"Unsupported op type: {type(op).__name__}",
        )

    @staticmethod
    def _apply_op(payload: Any, op: dict[str, Any]) -> None:
        """在 payload 上应用单个 RFC 6902 操作 (原地修改).

        Args:
            payload: 数据根 (将被原地修改)
            op: 归一化后的操作 dict

        Raises:
            ArtifactValidationError: test 失败 / 路径无法解析 / 未知 op
        """
        op_type = op.get("op")
        path = op.get("path")

        if op_type == "add":
            ArtifactManager._set_value(payload, path, op.get("value"))
        elif op_type == "replace":
            ArtifactManager._set_value(payload, path, op.get("value"))
        elif op_type == "remove":
            ArtifactManager._remove_value(payload, path)
        elif op_type == "move":
            from_path = op.get("from")
            if from_path is None:
                raise ArtifactValidationError(
                    field=str(path),
                    detail="move op requires 'from' field",
                )
            value = ArtifactManager._get_value(payload, from_path)
            ArtifactManager._remove_value(payload, from_path)
            ArtifactManager._set_value(payload, path, value)
        elif op_type == "copy":
            from_path = op.get("from")
            if from_path is None:
                raise ArtifactValidationError(
                    field=str(path),
                    detail="copy op requires 'from' field",
                )
            # 深拷贝, 保证 target 与 source 相互独立
            value = copy.deepcopy(ArtifactManager._get_value(payload, from_path))
            ArtifactManager._set_value(payload, path, value)
        elif op_type == "test":
            expected = op.get("value")
            try:
                current_value = ArtifactManager._get_value(payload, path)
            except ArtifactValidationError:
                current_value = _TEST_MISSING
            if current_value != expected:
                raise ArtifactValidationError(
                    field=str(path),
                    detail=(
                        f"test op failed at '{path}': "
                        f"expected {expected!r}, got {current_value!r}"
                    ),
                )
        else:
            raise ArtifactValidationError(
                field=str(path),
                detail=f"Unknown op: {op_type}",
            )
