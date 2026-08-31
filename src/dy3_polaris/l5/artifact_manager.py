"""产物管理模块 — L5 Agent Runtime 核心组件.

融合世界先进方案:
- LangGraph: Store API + 层次命名空间 + 向量搜索 + TTL
- OpenAI Agents SDK: RunResult 多面结果对象 + to_state() 快照
- Google ADK: BaseArtifactService + 版本自动递增 + MIME 类型 + session/user 作用域
- CrewAI: Task Output 声明式配置 + 任务链传递
- AutoGen: State 序列化 + version 兼容字段
- Temporal: Event Sourcing + Query/Signal/Update 三态分离
- Claude Science: 五维度溯源 + Execution Log 权威优先 + 版本 diff
- L5 设计文档: Artifact-Edit Channel + 五阶段生命周期 + DAG 版本树 + CC1 审核
- L7 设计文档: 三级缓存存储 + 搜索过滤 + 编辑权限控制

本模块实现:
1. ArtifactType — 8 种产物类型枚举 (text/chart/graph/molecule/table/formula/provenance/interactive)
2. ArtifactState — 5 阶段生命周期状态机 (created → rendered → reviewed → edited → archived)
3. Artifact — 不可变产物载体 (元数据 + payload + 溯源链 + 学习上下文)
4. ArtifactVersion — 版本记录 (版本号 + 内容哈希 + CC1 状态 + 编辑操作)
5. ArtifactStore — 抽象存储接口 (save/load/list/delete/versions)
6. InMemoryArtifactStore — 内存存储实现 (线程安全 + 版本管理)
7. ArtifactManager — 产物管理器 (创建/更新/版本/搜索/归档/溯源)
8. ArtifactEdit — 编辑操作记录 (编辑意图/状态/审核)
9. ArtifactEditState — 编辑状态 (pending/applied/rejected)
10. ArtifactProvenance — 产物溯源记录 (actor_chain + code_hash + 版本变更)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class ArtifactType(str, Enum):
    """产物类型枚举 (L7 设计文档 8 种类型).

    每种类型对应不同的渲染器和 MIME 类型:
    - TEXT: 文本解释、报告、说明 (application/json)
    - CHART: 数据可视化图表 (application/json)
    - GRAPH: 知识图谱、流程图 (application/json)
    - MOLECULE: 分子结构 (chemical/x-mdl-molfile)
    - TABLE: 数据表格 (application/json)
    - FORMULA: 数学/化学公式 (application/json)
    - PROVENANCE: 溯源记录 (application/json)
    - INTERACTIVE: 交互式组件 (application/json)
    """

    TEXT = "text"
    CHART = "chart"
    GRAPH = "graph"
    MOLECULE = "molecule"
    TABLE = "table"
    FORMULA = "formula"
    PROVENANCE = "provenance"
    INTERACTIVE = "interactive"


class ArtifactState(str, Enum):
    """产物生命周期状态 (L7 设计文档 5 阶段状态机).

    状态转换图:
        [*] --> Created: Agent 产出
        Created --> Rendered: L7 Renderer 渲染
        Rendered --> Reviewed: CC1 Actor-Critic 审查
        Reviewed --> Edited: 用户/Agent 修改
        Reviewed --> Archived: 会话结束归档
        Edited --> Rendered: 重新渲染
        Rendered --> Archived: 会话结束归档
        Archived --> [*]
    """

    CREATED = "created"
    RENDERED = "rendered"
    REVIEWED = "reviewed"
    EDITED = "edited"
    ARCHIVED = "archived"


class ArtifactEditState(str, Enum):
    """编辑操作状态 (L5 Artifact-Edit Channel).

    PENDING: 编辑已提交, 等待处理
    APPLIED: 编辑已应用, 生成新版本
    REJECTED: 编辑被拒绝 (CC1 审核不通过)
    """

    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"


# ============================================================
# 合法状态转换表
# ============================================================


_VALID_TRANSITIONS: dict[ArtifactState, set[ArtifactState]] = {
    ArtifactState.CREATED: {ArtifactState.RENDERED, ArtifactState.ARCHIVED},
    ArtifactState.RENDERED: {ArtifactState.REVIEWED, ArtifactState.ARCHIVED},
    ArtifactState.REVIEWED: {ArtifactState.EDITED, ArtifactState.ARCHIVED},
    ArtifactState.EDITED: {ArtifactState.RENDERED, ArtifactState.ARCHIVED},
    ArtifactState.ARCHIVED: set(),  # 终态, 不可转换
}


# ============================================================
# 默认 MIME 类型映射
# ============================================================


_DEFAULT_MIME: dict[ArtifactType, str] = {
    ArtifactType.TEXT: "application/json",
    ArtifactType.CHART: "application/json",
    ArtifactType.GRAPH: "application/json",
    ArtifactType.MOLECULE: "chemical/x-mdl-molfile",
    ArtifactType.TABLE: "application/json",
    ArtifactType.FORMULA: "application/json",
    ArtifactType.PROVENANCE: "application/json",
    ArtifactType.INTERACTIVE: "application/json",
}


# ============================================================
# 异常定义
# ============================================================


class ArtifactError(Exception):
    """产物管理基础错误."""

    pass


class ArtifactNotFoundError(ArtifactError):
    """产物不存在错误."""

    def __init__(self, artifact_id: str) -> None:
        self.artifact_id = artifact_id
        super().__init__(f"Artifact not found: {artifact_id}")


# ============================================================
# Artifact — 不可变产物载体
# ============================================================


class Artifact:
    """不可变产物载体 (融合 LangGraph Store + ADK + L5/L7 设计).

    核心特性:
    1. 不可变: 创建后字段不可修改 (通过 __setattr__ 拦截)
    2. 状态转换: 通过 transition_to() 方法转换生命周期状态
    3. 元数据: artifact_id / type / source_agent / version / mime
    4. 溯源链: provenance_chain 记录产出链路
    5. 学习上下文: learner_context 携带学习者信息
    6. Fork 来源: fork_origin 标记 Fork 会话来源
    7. 编辑权限: editable 控制是否可编辑 (CC2 审批后只读)

    不可变设计原理:
    - 产物是 Agent 产出的不可变记录, 类似 git commit
    - 修改通过创建新版本实现 (ADK 模式)
    - 仅 state 字段可通过 transition_to 变更 (生命周期管理)
    """

    def __init__(
        self,
        artifact_type: ArtifactType,
        source_agent: str,
        payload: dict[str, Any],
        *,
        artifact_id: str | None = None,
        version: int = 1,
        state: ArtifactState | None = None,
        editable: bool = True,
        fork_origin: str | None = None,
        mime: str | None = None,
        provenance_chain: list[str] | None = None,
        learner_context: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> None:
        """初始化产物.

        Args:
            artifact_type: 产物类型
            source_agent: 来源 Agent ID
            payload: 产物数据
            artifact_id: 自定义产物 ID (默认自动生成)
            version: 版本号 (默认 1)
            state: 初始状态 (默认 CREATED)
            editable: 是否可编辑 (默认 True)
            fork_origin: Fork 会话来源 (默认 None)
            mime: MIME 类型 (默认按类型推断)
            provenance_chain: 溯源链 (默认空列表)
            learner_context: 学习者上下文 (默认空字典)
            created_at: 创建时间戳 (默认当前时间)
        """
        # 验证 artifact_type
        if isinstance(artifact_type, str) and not isinstance(artifact_type, ArtifactType):
            try:
                artifact_type = ArtifactType(artifact_type)
            except ValueError:
                raise ValueError(f"Invalid artifact type: {artifact_type}")
        elif not isinstance(artifact_type, ArtifactType):
            raise ValueError(f"Invalid artifact type: {artifact_type}")

        # 验证 source_agent
        if not source_agent:
            raise ValueError("source_agent cannot be empty")

        # 使用 object.__setattr__ 绕过不可变检查 (初始化阶段)
        object.__setattr__(self, "artifact_id", artifact_id or f"art-{uuid.uuid4().hex[:12]}")
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "source_agent", source_agent)
        object.__setattr__(self, "payload", dict(payload) if payload else {})
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "state", state or ArtifactState.CREATED)
        object.__setattr__(self, "editable", editable)
        object.__setattr__(self, "fork_origin", fork_origin)
        object.__setattr__(self, "mime", mime or _DEFAULT_MIME.get(artifact_type, "application/json"))
        object.__setattr__(self, "provenance_chain", list(provenance_chain) if provenance_chain else [])
        object.__setattr__(self, "learner_context", dict(learner_context) if learner_context else {})
        object.__setattr__(self, "created_at", created_at or time.time())
        # 标记为不可变 (初始化完成后)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, key: str, value: Any) -> None:
        """拦截属性赋值, 实现不可变性.

        初始化后, 所有直接赋值都会抛出 AttributeError.
        transition_to() 使用 object.__setattr__ 绕过此检查.
        """
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"Cannot modify field '{key}' on immutable Artifact"
            )
        object.__setattr__(self, key, value)

    def transition_to(self, new_state: ArtifactState) -> None:
        """转换产物生命周期状态.

        合法转换:
        - CREATED → RENDERED, ARCHIVED
        - RENDERED → REVIEWED, ARCHIVED
        - REVIEWED → EDITED, ARCHIVED
        - EDITED → RENDERED, ARCHIVED
        - ARCHIVED → (终态, 不可转换)

        Args:
            new_state: 目标状态

        Raises:
            ArtifactError: 非法状态转换
        """
        valid_targets = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in valid_targets:
            raise ArtifactError(
                f"invalid state transition: {self.state.value} -> {new_state.value}"
            )
        object.__setattr__(self, "state", new_state)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典.

        Returns:
            包含所有字段的字典 (枚举转为字符串值)
        """
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "source_agent": self.source_agent,
            "payload": dict(self.payload),
            "version": self.version,
            "state": self.state.value,
            "editable": self.editable,
            "fork_origin": self.fork_origin,
            "mime": self.mime,
            "provenance_chain": list(self.provenance_chain),
            "learner_context": dict(self.learner_context),
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return (
            f"Artifact(id={self.artifact_id}, type={self.artifact_type.value}, "
            f"v{self.version}, state={self.state.value})"
        )


# ============================================================
# ArtifactVersion — 版本记录
# ============================================================


class ArtifactVersion:
    """产物版本记录 (融合 ADK 版本管理 + Claude Science 内容哈希).

    每个版本记录包含:
    - artifact_id: 所属产物 ID
    - version: 版本号 (从 1 开始递增)
    - content_hash: 内容哈希 (SHA-256, 用于完整性校验)
    - data_hash: 数据哈希 (可选, 区分代码和数据)
    - created_by: 创建者 Agent ID
    - created_at: 创建时间
    - cc1_status: CC1 审核状态 (pending/pass/fail)
    - edit_operation: 编辑操作记录 (L5 设计文档)
    - output_ref: 产物存储引用 (L5 DDL output_ref)

    对应 L5 DDL:
        CREATE TABLE artifact_versions (
            id, artifact_id, version, code_hash, data_hash,
            output_ref, edit_operation, created_by, created_at, cc1_status
        );
    """

    def __init__(
        self,
        artifact_id: str,
        version: int,
        content_hash: str,
        created_by: str,
        *,
        data_hash: str | None = None,
        cc1_status: str = "pending",
        edit_operation: dict[str, Any] | None = None,
        output_ref: str | None = None,
        created_at: float | None = None,
    ) -> None:
        self.artifact_id = artifact_id
        self.version = version
        self.content_hash = content_hash
        self.created_by = created_by
        self.data_hash = data_hash
        self.cc1_status = cc1_status
        self.edit_operation = dict(edit_operation) if edit_operation else None
        self.output_ref = output_ref
        self.created_at = created_at or time.time()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "artifact_id": self.artifact_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "data_hash": self.data_hash,
            "created_by": self.created_by,
            "cc1_status": self.cc1_status,
            "edit_operation": self.edit_operation,
            "output_ref": self.output_ref,
            "created_at": self.created_at,
        }


# ============================================================
# ArtifactStore — 抽象存储接口
# ============================================================


class ArtifactStore(ABC):
    """产物存储抽象接口 (ADK BaseArtifactService 模式).

    核心方法:
    - save: 保存产物, 返回版本号
    - load: 加载产物 (默认最新版本, 可指定版本)
    - list_artifacts: 列出所有产物 ID
    - list_versions: 列出产物的所有版本号
    - delete: 删除产物 (含所有版本)

    扩展点:
    - 子类可实现文件系统/数据库/对象存储等后端
    - 支持 TTL 过期 (LangGraph Store 模式)
    - 支持向量搜索 (LangGraph Store 模式)
    """

    @abstractmethod
    def save(self, artifact: Artifact) -> int:
        """保存产物, 返回版本号.

        Args:
            artifact: 产物对象

        Returns:
            保存的版本号
        """
        ...

    @abstractmethod
    def load(self, artifact_id: str, version: int | None = None) -> Artifact | None:
        """加载产物.

        Args:
            artifact_id: 产物 ID
            version: 版本号 (None 表示最新版本)

        Returns:
            产物对象, 不存在则返回 None
        """
        ...

    @abstractmethod
    def list_artifacts(self) -> list[str]:
        """列出所有产物 ID.

        Returns:
            产物 ID 列表
        """
        ...

    @abstractmethod
    def delete(self, artifact_id: str) -> None:
        """删除产物 (含所有版本).

        Args:
            artifact_id: 产物 ID

        Raises:
            ArtifactNotFoundError: 产物不存在
        """
        ...

    @abstractmethod
    def list_versions(self, artifact_id: str) -> list[int]:
        """列出产物的所有版本号.

        Args:
            artifact_id: 产物 ID

        Returns:
            版本号列表 (升序)
        """
        ...


# ============================================================
# InMemoryArtifactStore — 内存存储实现
# ============================================================


class InMemoryArtifactStore(ArtifactStore):
    """内存产物存储实现 (线程安全 + 版本管理).

    数据结构:
    - _store: {artifact_id: {version: Artifact}}
    - _lock: 读写锁 (线程安全)

    特性:
    1. 线程安全: 使用 RLock 保护所有操作
    2. 版本管理: 按版本号存储, 支持加载特定版本
    3. 自动哈希: 保存时计算内容哈希
    4. 引用语义: 存储对象引用 (状态变更实时可见)
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[int, Artifact]] = {}
        self._lock = threading.RLock()

    def save(self, artifact: Artifact) -> int:
        """保存产物, 返回版本号."""
        with self._lock:
            aid = artifact.artifact_id
            ver = artifact.version
            if aid not in self._store:
                self._store[aid] = {}
            self._store[aid][ver] = artifact
            logger.debug(f"Saved artifact {aid} v{ver}")
            return ver

    def load(self, artifact_id: str, version: int | None = None) -> Artifact | None:
        """加载产物 (默认最新版本)."""
        with self._lock:
            if artifact_id not in self._store:
                return None
            versions = self._store[artifact_id]
            if version is not None:
                return versions.get(version)
            # 返回最新版本
            if not versions:
                return None
            max_ver = max(versions.keys())
            return versions[max_ver]

    def list_artifacts(self) -> list[str]:
        """列出所有产物 ID."""
        with self._lock:
            return list(self._store.keys())

    def delete(self, artifact_id: str) -> None:
        """删除产物 (含所有版本)."""
        with self._lock:
            if artifact_id not in self._store:
                raise ArtifactNotFoundError(artifact_id)
            del self._store[artifact_id]
            logger.debug(f"Deleted artifact {artifact_id}")

    def list_versions(self, artifact_id: str) -> list[int]:
        """列出产物的所有版本号 (升序)."""
        with self._lock:
            if artifact_id not in self._store:
                return []
            return sorted(self._store[artifact_id].keys())


# ============================================================
# ArtifactProvenance — 产物溯源记录
# ============================================================


class ArtifactProvenance:
    """产物溯源记录 (Claude Science 五维度溯源 + L5 Provenance Ledger).

    五维度溯源:
    1. actor_chain: 参与产出的 Agent 链 (含 CC1 审核者)
    2. code_hash: 生成代码哈希 (可复现)
    3. data_hash: 数据哈希 (可选, 区分代码和数据)
    4. edit_summary: 编辑摘要 (人类可读)
    5. version: 版本变更 (from_version → to_version)

    对应 L5 DDL artifact_versions 表的溯源字段.
    """

    def __init__(
        self,
        artifact_id: str,
        actor_chain: list[str],
        edit_summary: str,
        code_hash: str,
        *,
        data_hash: str | None = None,
        from_version: int | None = None,
        to_version: int | None = None,
        timestamp: float | None = None,
    ) -> None:
        self.artifact_id = artifact_id
        self.actor_chain = list(actor_chain)
        self.edit_summary = edit_summary
        self.code_hash = code_hash
        self.data_hash = data_hash
        self.from_version = from_version
        self.to_version = to_version
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "artifact_id": self.artifact_id,
            "actor_chain": list(self.actor_chain),
            "edit_summary": self.edit_summary,
            "code_hash": self.code_hash,
            "data_hash": self.data_hash,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "timestamp": self.timestamp,
        }


# ============================================================
# ArtifactEdit — 编辑操作记录
# ============================================================


class ArtifactEdit:
    """产物编辑操作记录 (L5 Artifact-Edit Channel).

    编辑流程五阶段:
    1. 意图解析: Agent 接收自然语言编辑指令, 解析为结构化 edit_content
    2. 代码/数据读取: Agent 读取生成产物时的源代码和中间数据
    3. 重写生成: 基于编辑操作修改代码/数据, 生成新版本产物
    4. CC1 审核: CC1 Actor-Critic Reviewer 审核新版本质量
    5. 版本输出: 审核通过后输出 artifact_v{n+1}

    编辑类型:
    - 图表编辑: 修改坐标轴/颜色映射/数据标注/图表类型
    - 文本编辑: 修改表述/增删例子/结构调整
    - 数据编辑: 修正数值/更新数据源/重新计算
    """

    def __init__(
        self,
        artifact_id: str,
        learner_id: str,
        edit_content: dict[str, Any],
        *,
        edit_id: str | None = None,
        state: ArtifactEditState | None = None,
        created_at: float | None = None,
    ) -> None:
        self.edit_id = edit_id or f"edit-{uuid.uuid4().hex[:12]}"
        self.artifact_id = artifact_id
        self.learner_id = learner_id
        self.edit_content = dict(edit_content) if edit_content else {}
        self.state = state or ArtifactEditState.PENDING
        self.created_at = created_at or time.time()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "edit_id": self.edit_id,
            "artifact_id": self.artifact_id,
            "learner_id": self.learner_id,
            "edit_content": dict(self.edit_content),
            "state": self.state.value,
            "created_at": self.created_at,
        }


# ============================================================
# 用于区分 "未设置" 和 "None" 的哨兵值
# ============================================================


_UNSET: Any = object()


# ============================================================
# ArtifactManager — 产物管理器
# ============================================================


class ArtifactManager:
    """产物管理器 (融合 LangGraph Store + ADK + L5/L7 设计).

    核心能力:
    1. 创建产物: Agent 产出 → Artifact (CREATED 状态)
    2. 更新产物: 创建新版本 (ADK 模式, 不可变 + 版本递增)
    3. 版本管理: 版本历史查询 + 特定版本加载
    4. 搜索过滤: 按类型/来源/状态/Fork来源 多维度过滤
    5. 生命周期: 渲染/审核/编辑/归档 状态转换
    6. 溯源记录: actor_chain + code_hash + 版本变更
    7. 权限控制: editable 控制编辑权限 (CC2 审批后只读)

    设计原理:
    - 产物不可变: 修改通过创建新版本实现
    - 版本链: 每次更新创建新版本, 保留历史
    - 溯源: 每次变更记录溯源信息 (Claude Science 五维度)
    - 搜索: 支持多维度组合过滤 (L7 结构化过滤)
    """

    def __init__(self, store: ArtifactStore) -> None:
        """初始化产物管理器.

        Args:
            store: 产物存储后端
        """
        self._store = store
        # artifact_id → 最新 Artifact (内存缓存)
        self._artifacts: dict[str, Artifact] = {}
        # artifact_id → 版本记录列表
        self._version_records: dict[str, list[ArtifactVersion]] = {}
        # artifact_id → 溯源记录列表
        self._provenance: dict[str, list[ArtifactProvenance]] = {}
        self._lock = threading.RLock()
        self._reviewer: Any = None
        self._quality_gate: Any = None

    # ----------------------------------------------------------
    # 创建 / 获取
    # ----------------------------------------------------------

    def create(
        self,
        artifact_type: ArtifactType,
        source_agent: str,
        payload: dict[str, Any],
        *,
        artifact_id: str | None = None,
        editable: bool = True,
        fork_origin: str | None = None,
        mime: str | None = None,
        provenance_chain: list[str] | None = None,
        learner_context: dict[str, Any] | None = None,
    ) -> Artifact:
        """创建产物 (Agent 产出).

        Args:
            artifact_type: 产物类型
            source_agent: 来源 Agent ID
            payload: 产物数据
            artifact_id: 自定义产物 ID (默认自动生成)
            editable: 是否可编辑 (默认 True)
            fork_origin: Fork 会话来源
            mime: MIME 类型
            provenance_chain: 溯源链
            learner_context: 学习者上下文

        Returns:
            创建的产物对象

        Raises:
            ValueError: source_agent 为空
            ValueError: 无效产物类型
        """
        with self._lock:
            art = Artifact(
                artifact_type=artifact_type,
                source_agent=source_agent,
                payload=payload,
                artifact_id=artifact_id,
                editable=editable,
                fork_origin=fork_origin,
                mime=mime,
                provenance_chain=provenance_chain,
                learner_context=learner_context,
            )

            # 保存到存储
            self._store.save(art)

            # 缓存到内存
            self._artifacts[art.artifact_id] = art

            # 创建版本记录
            ver = ArtifactVersion(
                artifact_id=art.artifact_id,
                version=art.version,
                content_hash=self._compute_hash(art.payload),
                created_by=source_agent,
            )
            self._version_records.setdefault(art.artifact_id, []).append(ver)

            logger.debug(
                f"Created artifact {art.artifact_id} "
                f"(type={art.artifact_type.value}, agent={source_agent})"
            )
            return art

    def get(self, artifact_id: str) -> Artifact:
        """获取产物 (最新版本).

        Args:
            artifact_id: 产物 ID

        Returns:
            产物对象

        Raises:
            ArtifactNotFoundError: 产物不存在
        """
        with self._lock:
            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)
            return art

    # ----------------------------------------------------------
    # 更新 / 版本管理
    # ----------------------------------------------------------

    def update(
        self,
        artifact_id: str,
        payload: dict[str, Any],
        *,
        source_agent: str | None = None,
        edit_operation: dict[str, Any] | None = None,
    ) -> Artifact:
        """更新产物 (创建新版本, ADK 模式).

        不可变设计: 不修改原产物, 而是创建新版本.
        新版本继承原产物的类型/权限/上下文, 更新 payload.

        Args:
            artifact_id: 产物 ID
            payload: 新的产物数据
            source_agent: 执行更新的 Agent (默认使用原产物的 source_agent)
            edit_operation: 编辑操作记录 (L5 设计文档)

        Returns:
            新版本的产物对象

        Raises:
            ArtifactNotFoundError: 产物不存在
            ArtifactError: 产物不可编辑 (editable=False)
            ArtifactError: 产物已归档 (state=ARCHIVED)
        """
        with self._lock:
            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)

            if not art.editable:
                raise ArtifactError(
                    f"Artifact {artifact_id} is not editable "
                    f"(editable=False, CC2 approved)"
                )

            if art.state == ArtifactState.ARCHIVED:
                raise ArtifactError(
                    f"Artifact {artifact_id} is archived, cannot update"
                )

            new_version = art.version + 1
            new_art = Artifact(
                artifact_type=art.artifact_type,
                source_agent=source_agent or art.source_agent,
                payload=payload,
                artifact_id=artifact_id,
                version=new_version,
                state=art.state,  # 继承当前状态
                editable=art.editable,
                fork_origin=art.fork_origin,
                mime=art.mime,
                provenance_chain=list(art.provenance_chain),
                learner_context=dict(art.learner_context),
                created_at=time.time(),
            )

            # 保存到存储
            self._store.save(new_art)

            # 更新缓存
            self._artifacts[artifact_id] = new_art

            # 创建版本记录
            ver = ArtifactVersion(
                artifact_id=artifact_id,
                version=new_version,
                content_hash=self._compute_hash(payload),
                created_by=source_agent or art.source_agent,
                edit_operation=edit_operation,
            )
            self._version_records[artifact_id].append(ver)

            logger.debug(
                f"Updated artifact {artifact_id} to v{new_version} "
                f"(agent={source_agent or art.source_agent})"
            )
            return new_art

    def get_version(self, artifact_id: str, version: int) -> Artifact | None:
        """获取特定版本的产物.

        Args:
            artifact_id: 产物 ID
            version: 版本号

        Returns:
            产物对象, 不存在则返回 None
        """
        with self._lock:
            return self._store.load(artifact_id, version=version)

    def get_version_history(self, artifact_id: str) -> list[ArtifactVersion]:
        """获取版本历史.

        Args:
            artifact_id: 产物 ID

        Returns:
            版本记录列表 (按版本号升序)

        Raises:
            ArtifactNotFoundError: 产物不存在
        """
        with self._lock:
            if artifact_id not in self._artifacts:
                raise ArtifactNotFoundError(artifact_id)
            return list(self._version_records.get(artifact_id, []))

    # ----------------------------------------------------------
    # 搜索 / 过滤
    # ----------------------------------------------------------

    def search(
        self,
        *,
        artifact_type: ArtifactType | None = None,
        source_agent: str | None = None,
        state: ArtifactState | None = None,
        fork_origin: str | _UNSET = _UNSET,
    ) -> list[Artifact]:
        """搜索产物 (L7 结构化多维度过滤).

        支持的过滤维度:
        - artifact_type: 产物类型
        - source_agent: 来源 Agent
        - state: 生命周期状态
        - fork_origin: Fork 来源 (None 表示主会话产物)

        使用哨兵值 _UNSET 区分 "不过滤" 和 "过滤为 None":
        - search() → 不按 fork_origin 过滤
        - search(fork_origin=None) → 过滤 fork_origin 为 None 的产物
        - search(fork_origin="sess-001") → 过滤 fork_origin 为 "sess-001" 的产物

        Args:
            artifact_type: 产物类型过滤
            source_agent: 来源 Agent 过滤
            state: 状态过滤
            fork_origin: Fork 来源过滤

        Returns:
            匹配的产物列表
        """
        with self._lock:
            results: list[Artifact] = []
            for art in self._artifacts.values():
                if artifact_type is not None and art.artifact_type != artifact_type:
                    continue
                if source_agent is not None and art.source_agent != source_agent:
                    continue
                if state is not None and art.state != state:
                    continue
                if fork_origin is not _UNSET:
                    if fork_origin is None:
                        if art.fork_origin is not None:
                            continue
                    else:
                        if art.fork_origin != fork_origin:
                            continue
                results.append(art)
            return results

    def list_all(self) -> list[Artifact]:
        """列出所有产物 (最新版本).

        Returns:
            所有产物列表
        """
        with self._lock:
            return list(self._artifacts.values())

    # ----------------------------------------------------------
    # 生命周期管理
    # ----------------------------------------------------------

    def archive(self, artifact_id: str) -> None:
        """归档产物 (L7 生命周期终态).

        Args:
            artifact_id: 产物 ID

        Raises:
            ArtifactNotFoundError: 产物不存在
            ArtifactError: 非法状态转换
        """
        with self._lock:
            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)
            art.transition_to(ArtifactState.ARCHIVED)
            logger.debug(f"Archived artifact {artifact_id}")

    def delete(self, artifact_id: str) -> None:
        """删除产物 (含所有版本).

        Args:
            artifact_id: 产物 ID

        Raises:
            ArtifactNotFoundError: 产物不存在
        """
        with self._lock:
            if artifact_id not in self._artifacts:
                raise ArtifactNotFoundError(artifact_id)
            self._store.delete(artifact_id)
            del self._artifacts[artifact_id]
            self._version_records.pop(artifact_id, None)
            self._provenance.pop(artifact_id, None)
            logger.debug(f"Deleted artifact {artifact_id}")

    # ----------------------------------------------------------
    # 溯源管理
    # ----------------------------------------------------------

    def add_provenance(
        self,
        artifact_id: str,
        actor_chain: list[str],
        edit_summary: str,
        code_hash: str,
        *,
        data_hash: str | None = None,
        from_version: int | None = None,
        to_version: int | None = None,
    ) -> ArtifactProvenance:
        """添加溯源记录 (Claude Science 五维度溯源).

        Args:
            artifact_id: 产物 ID
            actor_chain: 参与产出的 Agent 链
            edit_summary: 编辑摘要 (人类可读)
            code_hash: 生成代码哈希
            data_hash: 数据哈希 (可选)
            from_version: 源版本号
            to_version: 目标版本号

        Returns:
            创建的溯源记录

        Raises:
            ArtifactNotFoundError: 产物不存在
        """
        with self._lock:
            if artifact_id not in self._artifacts:
                raise ArtifactNotFoundError(artifact_id)

            prov = ArtifactProvenance(
                artifact_id=artifact_id,
                actor_chain=actor_chain,
                edit_summary=edit_summary,
                code_hash=code_hash,
                data_hash=data_hash,
                from_version=from_version,
                to_version=to_version,
            )
            self._provenance.setdefault(artifact_id, []).append(prov)
            logger.debug(
                f"Added provenance for {artifact_id}: "
                f"{from_version}→{to_version} ({edit_summary})"
            )
            return prov

    def get_provenance_chain(self, artifact_id: str) -> list[ArtifactProvenance]:
        """获取产物溯源链.

        Args:
            artifact_id: 产物 ID

        Returns:
            溯源记录列表 (按时间顺序)

        Raises:
            ArtifactNotFoundError: 产物不存在
        """
        with self._lock:
            if artifact_id not in self._artifacts:
                raise ArtifactNotFoundError(artifact_id)
            return list(self._provenance.get(artifact_id, []))

    # ----------------------------------------------------------
    # CC1 审核 (集成 reflection_quality 模块)
    # ----------------------------------------------------------

    def set_reviewer(self, reviewer: Any) -> None:
        """设置 CC1 审核器 (集成 reflection_quality 模块).

        Args:
            reviewer: CC1Reviewer 实例
        """
        self._reviewer = reviewer

    def set_quality_gate(self, gate: Any) -> None:
        """设置质量门控 (集成 reflection_quality 模块).

        Args:
            gate: QualityGate 实例
        """
        self._quality_gate = gate

    async def review_artifact(
        self,
        artifact_id: str,
        reviewer: Any | None = None,
        gate: Any | None = None,
    ) -> Any:
        """对产物执行 CC1 审核 + 自纠循环 (集成 reflection_quality 模块).

        审核流程 (LangGraph Generator-Critic):
        1. CC1Reviewer 多维度评审
        2. QualityGate 评估分数
        3. ALLOW → 通过; REJECT → 拒绝; REVISE → 自纠后重审; ESCALATE → 升级
        4. 更新 ArtifactVersion.cc1_status
        5. 状态转换: RENDERED → REVIEWED

        Args:
            artifact_id: 产物 ID
            reviewer: CC1Reviewer 实例 (可选, 默认使用 set_reviewer 设置的)
            gate: QualityGate 实例 (可选, 默认使用 set_quality_gate 设置的)

        Returns:
            ReflectionResult 反思结果 (含所有审核记录)

        Raises:
            ArtifactNotFoundError: 产物不存在
            ValueError: 未设置 reviewer 或 quality_gate
        """
        from .reflection_quality import (
            GateAction,
            ReflectionResult,
            ReflectionTrigger,
            Verdict,
        )
        import uuid as _uuid

        use_reviewer = reviewer or self._reviewer
        use_gate = gate or self._quality_gate

        if use_reviewer is None:
            raise ValueError("未设置 reviewer, 请先调用 set_reviewer() 或传入 reviewer 参数")
        if use_gate is None:
            raise ValueError("未设置 quality_gate, 请先调用 set_quality_gate() 或传入 gate 参数")

        with self._lock:
            art = self._artifacts.get(artifact_id)
            if art is None:
                raise ArtifactNotFoundError(artifact_id)

            reviews = []
            iteration = 1
            max_iterations = use_gate.max_revisions
            current_data = dict(art.payload)
            resolved_issues: list[str] = []

            # 自纠循环
            while iteration <= max_iterations:
                # CC1 Reviewer 评审
                review = await use_reviewer.review(
                    artifact_id=artifact_id,
                    artifact_data=current_data,
                    agent_id=art.source_agent,
                    iteration=iteration,
                    history=reviews if reviews else None,
                )
                reviews.append(review)

                # QualityGate 评估
                gate_result = use_gate.evaluate(
                    score=review.weighted_score,
                    iteration=iteration,
                )

                if gate_result.action == GateAction.ALLOW:
                    break
                elif gate_result.action == GateAction.REJECT:
                    break
                elif gate_result.action == GateAction.ESCALATE:
                    break
                elif gate_result.action == GateAction.REVISE:
                    # 自纠改进
                    current_data = self._self_correct_payload(current_data)
                    resolved_issues.append(f"迭代{iteration}: 自纠改进")
                    iteration += 1

            # 更新 cc1_status
            versions = self._version_records.get(artifact_id, [])
            if versions:
                if gate_result.action == GateAction.ALLOW:
                    versions[-1].cc1_status = "pass"
                elif gate_result.action == GateAction.REJECT:
                    versions[-1].cc1_status = "fail"
                else:
                    versions[-1].cc1_status = "pending"

            # 状态转换: RENDERED → REVIEWED
            if art.state == ArtifactState.RENDERED:
                art.transition_to(ArtifactState.REVIEWED)

            # 创建反思结果
            final_verdict = reviews[-1].verdict
            result = ReflectionResult(
                artifact_id=artifact_id,
                agent_id=art.source_agent,
                trigger=ReflectionTrigger.SINGLE_AGENT,
                reviews=reviews,
                final_verdict=final_verdict,
                max_iterations=max_iterations,
                resolved_issues=resolved_issues,
            )

            logger.info(
                f"Reviewed artifact {artifact_id}: "
                f"verdict={final_verdict.value}, "
                f"score={reviews[-1].weighted_score:.4f}, "
                f"iterations={len(reviews)}"
            )

            return result

    @staticmethod
    def _self_correct_payload(data: dict[str, Any]) -> dict[str, Any]:
        """自纠改进产物 payload (L5 设计文档 7.1.2 warn 级别).

        根据审核反馈自动改进:
        1. 提升置信度 (如果偏低)
        2. 补充引用 (如果缺失)
        3. 填充空缺字段 (如 report_id / kp_gaps)
        """
        import uuid as _uuid
        corrected = dict(data)
        if "confidence" in corrected and corrected["confidence"] < 0.85:
            corrected["confidence"] = min(0.95, corrected["confidence"] + 0.15)
        if not corrected.get("references"):
            corrected["references"] = ["auto-generated-ref"]
        if not corrected.get("report_id"):
            corrected["report_id"] = f"auto-corrected-{_uuid.uuid4().hex[:8]}"
        if not corrected.get("kp_gaps") and "kp_gaps" not in corrected:
            corrected["kp_gaps"] = ["auto-detected-KP"]
        return corrected

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------

    @staticmethod
    def _compute_hash(payload: dict[str, Any]) -> str:
        """计算 payload 的内容哈希 (SHA-256).

        Args:
            payload: 产物数据

        Returns:
            哈希字符串 (格式: sha256:<16位hex>)
        """
        try:
            payload_str = json.dumps(payload, sort_keys=True, default=str)
            hash_hex = hashlib.sha256(payload_str.encode()).hexdigest()[:16]
            return f"sha256:{hash_hex}"
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to compute hash: {e}")
            return f"sha256:unknown"
