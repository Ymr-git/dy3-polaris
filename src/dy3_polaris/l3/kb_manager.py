"""L3 领域知识层 — 知识库生命周期管理器.

借鉴世界先进方案的知识库管理设计:
- Neo4j DBMS: 数据库创建/打开/关闭/删除 + 数据库状态管理
- Elasticsearch ILM (Index Lifecycle Management): 索引生命周期阶段流转
  (hot -> warm -> cold -> frozen -> delete)
- 企业知识平台管理: 保留策略 + 垃圾回收 + 健康监控 + 备份恢复编排

核心特性:
1. 知识库全生命周期管理: 创建 (CREATING) -> 活跃 (ACTIVE) -> 只读 (READONLY)
   -> 弃用 (DEPRECATED) -> 归档 (ARCHIVED) -> 删除 (DELETED)
2. 保留策略执行: 过期归档 / 低质量归档 / 容量限制 / 宽限期删除
3. 垃圾回收: 孤儿实体 (不被任何三元组引用的实体) 自动清理
4. 健康监控: 实体/三元组/切片统计 + 孤儿检测 + 低质量检测 + 存储估算
5. 备份/恢复编排: 基于快照的备份管理 + 版本化恢复 + 旧快照清理
6. 统计与报告: 多维度统计信息 + 知识库列表 + 配置管理

线程安全: 所有共享状态通过 threading.RLock 保护。
无外部依赖: 仅使用标准库 + pydantic v2。

Usage::

    from dy3_polaris.l3.kb_manager import (
        KnowledgeBaseManager, KBConfig, RetentionPolicy, KBLifecycleState,
    )

    manager = KnowledgeBaseManager()

    # 创建知识库
    config = KBConfig(kb_id="kb-001", name="化学知识库", domain="chemistry")
    manager.create_kb(config)

    # 获取存储并写入数据
    store = manager.open_kb("kb-001")
    store.add_entity(entity)

    # 生命周期管理
    manager.transition_state("kb-001", KBLifecycleState.READONLY)

    # 备份与恢复
    snapshot = manager.backup("kb-001", description="定期备份")
    manager.restore("kb-001", snapshot.snapshot_id)

    # 维护操作
    manager.run_gc("kb-001")           # 垃圾回收
    manager.enforce_retention("kb-001")  # 保留策略
    health = manager.check_health("kb-001")  # 健康检查

    # 删除知识库
    manager.transition_state("kb-001", KBLifecycleState.ARCHIVED)
    manager.delete_kb("kb-001")
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
import uuid
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import L3Error
from .models import KnowledgeBaseStats, KnowledgeStatus
from .persistence import PersistenceManager
from .store import KnowledgeStore

logger = logging.getLogger(__name__)


# ============================================================
# 模块常量
# ============================================================

# 知识库管理器版本号
_KB_MANAGER_VERSION: str = "1.0"

# 默认快照基础目录 (位于系统临时目录下)
_DEFAULT_BASE_DIR: Path = Path(tempfile.gettempdir()) / "dy3_kb_manager"

# 一天的秒数
_DAY_SECONDS: float = 86400.0

# 存储大小估算的采样上限 (避免超大知识库遍历耗时过长)
_SIZE_ESTIMATE_SAMPLE_LIMIT: int = 100000


# ============================================================
# 异常定义
# ============================================================


class KBManagerError(L3Error):
    """知识库管理器异常 (借鉴 Neo4j DBMS 管理错误 + Elasticsearch ILM 异常).

    当知识库生命周期管理操作失败时抛出此异常。
    继承 L3Error 异常体系, 集成 JSON-RPC 错误码 (-32416)。

    Attributes:
        kb_id: 相关的知识库 ID
    """

    def __init__(
        self,
        kb_id: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.kb_id = kb_id
        ctx: dict[str, Any] = {"kb_id": kb_id}
        ctx.update(context or {})
        super().__init__(
            "L3_KB_MANAGER",
            detail or f"kb_id={kb_id}",
            ctx,
        )

    def _jsonrpc_code(self) -> int:
        return -32416


# ============================================================
# 生命周期状态枚举
# ============================================================


class KBLifecycleState(str, Enum):
    """知识库生命周期状态 (借鉴 Elasticsearch ILM 阶段 + Neo4j DBMS 状态).

    状态流转:
        CREATING -> ACTIVE (创建完成)
        ACTIVE -> READONLY (转为只读)
        ACTIVE -> DEPRECATED (标记弃用)
        READONLY -> ACTIVE (恢复活跃)
        READONLY -> DEPRECATED (标记弃用)
        DEPRECATED -> ARCHIVED (归档)
        DEPRECATED -> READONLY (恢复只读)
        ARCHIVED -> DELETED (永久删除)
        DELETED -> (终态, 不可流转)

    Attributes:
        CREATING: 创建中, 尚未就绪
        ACTIVE: 活跃状态, 可读可写
        READONLY: 只读状态, 不可写入
        DEPRECATED: 已弃用, 不推荐使用但仍可访问
        ARCHIVED: 已归档, 仅保留历史快照
        DELETED: 已删除, 终态
    """

    CREATING = "creating"
    ACTIVE = "active"
    READONLY = "readonly"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DELETED = "deleted"


# 允许的状态流转映射 (借鉴 Elasticsearch ILM 阶段流转 + Neo4j DBMS 状态机)
# CREATING -> ACTIVE
# ACTIVE -> READONLY, DEPRECATED
# READONLY -> ACTIVE, DEPRECATED
# DEPRECATED -> ARCHIVED, READONLY
# ARCHIVED -> DELETED
# DELETED -> (终态, 不允许任何流转)
_ALLOWED_TRANSITIONS: dict[KBLifecycleState, set[KBLifecycleState]] = {
    KBLifecycleState.CREATING: {KBLifecycleState.ACTIVE},
    KBLifecycleState.ACTIVE: {KBLifecycleState.READONLY, KBLifecycleState.DEPRECATED},
    KBLifecycleState.READONLY: {KBLifecycleState.ACTIVE, KBLifecycleState.DEPRECATED},
    KBLifecycleState.DEPRECATED: {KBLifecycleState.ARCHIVED, KBLifecycleState.READONLY},
    KBLifecycleState.ARCHIVED: {KBLifecycleState.DELETED},
    KBLifecycleState.DELETED: set(),
}


# ============================================================
# 保留策略
# ============================================================


class RetentionPolicy(BaseModel):
    """知识保留策略 (借鉴 Elasticsearch ILM 保留策略 + 企业数据治理保留规则).

    控制知识库中条目的生命周期:
    - 过期归档: 超过 max_age_days 的条目自动归档
    - 容量限制: 超过 max_entries 时删除最旧条目
    - 低质量归档: 质量分数低于 quality_threshold 的条目自动归档
    - 宽限期删除: 归档超过 deletion_grace_days 后永久删除

    Attributes:
        max_age_days: 条目最大保留天数 (超过则归档)
        max_entries: 最大条目数量 (超过则删除最旧)
        auto_archive: 是否自动归档过期/低质量条目
        auto_delete: 是否自动删除超期归档条目
        deletion_grace_days: 永久删除前的宽限期天数
        quality_threshold: 自动归档的质量分数阈值 [0.0, 1.0]
    """

    max_age_days: int = Field(default=365, ge=1, description="条目最大保留天数")
    max_entries: int = Field(default=1000000, ge=1, description="最大条目数量")
    auto_archive: bool = Field(default=True, description="是否自动归档过期/低质量条目")
    auto_delete: bool = Field(default=False, description="是否自动删除超期归档条目")
    deletion_grace_days: int = Field(default=30, ge=1, description="永久删除前宽限期天数")
    quality_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="自动归档的质量分数阈值"
    )


# ============================================================
# 健康状态
# ============================================================


class HealthStatus(BaseModel):
    """知识库健康状态 (借鉴 Elasticsearch cluster health + Neo4j DBMS 健康检查).

    Attributes:
        state: 健康等级 ("healthy" / "warning" / "critical")
        total_entities: 实体总数
        total_triples: 三元组总数
        total_chunks: 切片总数
        orphaned_entities: 孤儿实体数 (不被任何三元组引用)
        low_quality_entities: 低质量实体数 (质量分数 < 阈值)
        storage_size_mb: 估算存储大小 (MB)
        last_backup_time: 最后备份时间戳, None 表示从未备份
        issues: 发现的问题列表
    """

    state: str = Field(default="healthy", description="健康等级")
    total_entities: int = Field(default=0, ge=0, description="实体总数")
    total_triples: int = Field(default=0, ge=0, description="三元组总数")
    total_chunks: int = Field(default=0, ge=0, description="切片总数")
    orphaned_entities: int = Field(default=0, ge=0, description="孤儿实体数")
    low_quality_entities: int = Field(default=0, ge=0, description="低质量实体数")
    storage_size_mb: float = Field(default=0.0, ge=0.0, description="估算存储大小 (MB)")
    last_backup_time: float | None = Field(default=None, description="最后备份时间戳")
    issues: list[str] = Field(default_factory=list, description="问题列表")


# ============================================================
# 快照信息
# ============================================================


class KBSnapshot(BaseModel):
    """知识库快照信息 (借鉴 Redis RDB 快照元数据 + Neo4j 备份元信息).

    Attributes:
        snapshot_id: 快照唯一标识
        timestamp: 快照创建时间戳
        version: 快照格式版本
        total_entities: 快照中实体总数
        total_triples: 快照中三元组总数
        total_chunks: 快照中切片总数
        size_bytes: 快照大小 (字节)
        checksum: SHA-256 校验和
        description: 快照描述
    """

    snapshot_id: str = Field(..., description="快照唯一标识")
    timestamp: float = Field(default_factory=time.time, description="快照创建时间戳")
    version: str = Field(default=_KB_MANAGER_VERSION, description="快照格式版本")
    total_entities: int = Field(default=0, ge=0, description="快照中实体总数")
    total_triples: int = Field(default=0, ge=0, description="快照中三元组总数")
    total_chunks: int = Field(default=0, ge=0, description="快照中切片总数")
    size_bytes: int = Field(default=0, ge=0, description="快照大小 (字节)")
    checksum: str = Field(default="", description="SHA-256 校验和")
    description: str = Field(default="", description="快照描述")


# ============================================================
# 知识库配置
# ============================================================


class KBConfig(BaseModel):
    """知识库配置 (借鉴 Neo4j 数据库配置 + Elasticsearch 索引设置).

    Attributes:
        kb_id: 知识库唯一标识
        name: 知识库名称
        description: 知识库描述
        domain: 所属领域 (如 "chemistry", "general")
        retention: 保留策略
        auto_backup: 是否启用自动备份
        backup_interval_hours: 自动备份间隔 (小时)
        max_snapshots: 保留的最大快照数量
        created_at: 创建时间戳
        created_by: 创建者标识
    """

    kb_id: str = Field(..., min_length=1, description="知识库唯一标识")
    name: str = Field(..., min_length=1, description="知识库名称")
    description: str = Field(default="", description="知识库描述")
    domain: str = Field(default="general", description="所属领域")
    retention: RetentionPolicy = Field(
        default_factory=RetentionPolicy, description="保留策略"
    )
    auto_backup: bool = Field(default=True, description="是否启用自动备份")
    backup_interval_hours: int = Field(
        default=24, ge=1, description="自动备份间隔 (小时)"
    )
    max_snapshots: int = Field(default=10, ge=1, description="保留的最大快照数量")
    created_at: float = Field(default=0.0, description="创建时间戳")
    created_by: str = Field(default="", description="创建者标识")


# ============================================================
# 知识库生命周期管理器
# ============================================================


class KnowledgeBaseManager:
    """知识库生命周期管理器 (借鉴 Neo4j DBMS + ES ILM + 企业知识平台管理).

    功能:
    1. 知识库创建/打开/关闭/删除
    2. 生命周期状态机 (CREATING -> ACTIVE -> READONLY -> DEPRECATED -> ARCHIVED -> DELETED)
    3. 保留策略执行 (过期归档/低质量归档/容量限制)
    4. 垃圾回收 (孤儿实体清理)
    5. 健康监控 (状态检查/问题检测)
    6. 备份/恢复编排 (快照管理)
    7. 统计与报告

    线程安全: 所有操作通过 RLock 保护。
    """

    def __init__(self) -> None:
        """初始化知识库管理器."""
        # 知识库注册表: {kb_id: (config, store, state)}
        self._kbs: dict[str, tuple[KBConfig, KnowledgeStore, KBLifecycleState]] = {}
        # 快照注册表: {kb_id: [KBSnapshot, ...]}
        self._snapshots: dict[str, list[KBSnapshot]] = {}
        # 持久化管理器: {kb_id: PersistenceManager}
        self._persistence_managers: dict[str, PersistenceManager] = {}
        # 持久化基础路径: {kb_id: Path}
        self._base_paths: dict[str, Path] = {}
        # 全局基础目录
        self._base_dir: Path = _DEFAULT_BASE_DIR
        # 线程安全锁 (可重入)
        self._lock: RLock = RLock()

    # ================================================================
    # 知识库生命周期管理
    # ================================================================

    def create_kb(self, config: KBConfig) -> str:
        """创建新的知识库.

        创建流程:
        1. 检查 kb_id 是否已存在
        2. 创建 KnowledgeStore 实例
        3. 创建 PersistenceManager 实例
        4. 设置状态为 CREATING, 然后立即流转为 ACTIVE

        Args:
            config: 知识库配置

        Returns:
            知识库 ID (kb_id)

        Raises:
            KBManagerError: 知识库已存在
        """
        with self._lock:
            if config.kb_id in self._kbs:
                raise KBManagerError(
                    kb_id=config.kb_id,
                    detail=f"知识库已存在: {config.kb_id}",
                )

            # 填充创建时间
            if config.created_at == 0.0:
                config = config.model_copy(update={"created_at": time.time()})

            # 创建知识存储引擎
            store = KnowledgeStore()

            # 创建持久化管理器 (每个知识库独立目录)
            base_path = self._base_dir / config.kb_id
            base_path.mkdir(parents=True, exist_ok=True)
            pm = PersistenceManager(store, base_path)

            # 注册知识库 (初始状态: CREATING)
            self._kbs[config.kb_id] = (config, store, KBLifecycleState.CREATING)
            self._snapshots[config.kb_id] = []
            self._persistence_managers[config.kb_id] = pm
            self._base_paths[config.kb_id] = base_path

            # 立即流转为 ACTIVE
            self._kbs[config.kb_id] = (config, store, KBLifecycleState.ACTIVE)

            logger.info(
                "创建知识库: %s (name=%s, domain=%s)",
                config.kb_id, config.name, config.domain,
            )
            return config.kb_id

    def open_kb(self, kb_id: str) -> KnowledgeStore:
        """打开知识库并返回其存储引擎.

        Args:
            kb_id: 知识库 ID

        Returns:
            知识存储引擎实例

        Raises:
            KBManagerError: 知识库不存在或已删除
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)

            if state == KBLifecycleState.DELETED:
                raise KBManagerError(
                    kb_id=kb_id,
                    detail=f"知识库已删除, 无法打开: {kb_id}",
                )
            if state == KBLifecycleState.ARCHIVED:
                raise KBManagerError(
                    kb_id=kb_id,
                    detail=f"知识库已归档, 需先恢复才能打开: {kb_id}",
                )

            logger.debug("打开知识库: %s (state=%s)", kb_id, state.value)
            return store

    def close_kb(self, kb_id: str) -> None:
        """关闭知识库.

        关闭操作:
        1. 验证知识库存在
        2. 若启用自动备份且处于 ACTIVE 状态, 保存快照
        3. 记录关闭日志

        注意: 关闭不会改变知识库的生命周期状态, 知识库仍可通过
        open_kb 再次打开。

        Args:
            kb_id: 知识库 ID

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)

            if state == KBLifecycleState.DELETED:
                raise KBManagerError(
                    kb_id=kb_id,
                    detail=f"知识库已删除, 无法关闭: {kb_id}",
                )

            # 若启用自动备份且处于活跃状态, 保存快照确保数据持久化
            if config.auto_backup and state == KBLifecycleState.ACTIVE:
                try:
                    self._backup_internal(kb_id, description="关闭时自动备份")
                except Exception as exc:
                    logger.warning("关闭知识库时自动备份失败: %s (%s)", kb_id, exc)

            logger.info("关闭知识库: %s (state=%s)", kb_id, state.value)

    def delete_kb(self, kb_id: str, *, force: bool = False) -> None:
        """删除知识库.

        删除条件:
        - 知识库必须处于 ARCHIVED 状态, 或 force=True 强制删除
        - 删除后清除存储、快照和持久化数据

        Args:
            kb_id: 知识库 ID
            force: 是否强制删除 (跳过 ARCHIVED 状态检查)

        Raises:
            KBManagerError: 知识库不存在或状态不允许删除
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)

            if not force and state != KBLifecycleState.ARCHIVED:
                raise KBManagerError(
                    kb_id=kb_id,
                    detail=(
                        f"知识库状态为 {state.value}, 删除前必须先转为 "
                        f"{KBLifecycleState.ARCHIVED.value} 状态"
                    ),
                )

            if state != KBLifecycleState.DELETED:
                # 流转为 DELETED 终态
                self._kbs[kb_id] = (config, store, KBLifecycleState.DELETED)

            # 清空存储数据
            store.clear()

            # 清理持久化目录
            base_path = self._base_paths.get(kb_id)
            if base_path and base_path.exists():
                shutil.rmtree(base_path, ignore_errors=True)

            # 清理快照记录
            self._snapshots.pop(kb_id, None)
            self._persistence_managers.pop(kb_id, None)
            self._base_paths.pop(kb_id, None)

            logger.info("删除知识库: %s (force=%s)", kb_id, force)

    # ================================================================
    # 状态管理
    # ================================================================

    def transition_state(
        self, kb_id: str, new_state: KBLifecycleState
    ) -> KBLifecycleState:
        """流转知识库生命周期状态.

        验证状态流转是否允许, 若允许则更新状态。

        Args:
            kb_id: 知识库 ID
            new_state: 目标状态

        Returns:
            流转后的新状态

        Raises:
            KBManagerError: 知识库不存在或状态流转不允许
        """
        with self._lock:
            config, store, current_state = self._get_kb(kb_id)

            if current_state == new_state:
                logger.debug("知识库状态未变化: %s (%s)", kb_id, new_state.value)
                return current_state

            if not self._validate_transition(current_state, new_state):
                raise KBManagerError(
                    kb_id=kb_id,
                    detail=(
                        f"不允许的状态流转: {current_state.value} -> "
                        f"{new_state.value}"
                    ),
                    context={
                        "current_state": current_state.value,
                        "new_state": new_state.value,
                    },
                )

            self._kbs[kb_id] = (config, store, new_state)
            logger.info(
                "知识库状态流转: %s (%s -> %s)",
                kb_id, current_state.value, new_state.value,
            )
            return new_state

    def get_state(self, kb_id: str) -> KBLifecycleState:
        """获取知识库当前生命周期状态.

        Args:
            kb_id: 知识库 ID

        Returns:
            当前生命周期状态

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            _, _, state = self._get_kb(kb_id)
            return state

    # ================================================================
    # 备份与恢复
    # ================================================================

    def backup(self, kb_id: str, *, description: str = "") -> KBSnapshot:
        """创建知识库备份快照.

        使用 PersistenceManager 保存全量快照, 并记录快照元信息。

        Args:
            kb_id: 知识库 ID
            description: 快照描述 (可选)

        Returns:
            快照信息

        Raises:
            KBManagerError: 知识库不存在或备份失败
        """
        with self._lock:
            return self._backup_internal(kb_id, description=description)

    def restore(self, kb_id: str, snapshot_id: str) -> dict[str, Any]:
        """从快照恢复知识库.

        恢复流程:
        1. 查找指定快照
        2. 使用 PersistenceManager 加载快照 (覆盖当前数据)
        3. 返回恢复后的统计信息

        Args:
            kb_id: 知识库 ID
            snapshot_id: 快照 ID

        Returns:
            恢复结果字典, 包含恢复后的统计信息

        Raises:
            KBManagerError: 知识库不存在或快照不存在
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)
            pm = self._persistence_managers.get(kb_id)

            if pm is None:
                raise KBManagerError(
                    kb_id=kb_id,
                    detail="持久化管理器不可用",
                )

            # 查找快照
            snapshot = self._find_snapshot(kb_id, snapshot_id)
            if snapshot is None:
                raise KBManagerError(
                    kb_id=kb_id,
                    detail=f"快照不存在: {snapshot_id}",
                    context={"snapshot_id": snapshot_id},
                )

            # 构建快照路径并加载
            base_path = self._base_paths[kb_id]
            snapshot_dir = base_path / f"snapshot_{snapshot.snapshot_id}"

            if not snapshot_dir.exists():
                raise KBManagerError(
                    kb_id=kb_id,
                    detail=f"快照目录不存在: {snapshot_dir}",
                    context={"snapshot_id": snapshot_id},
                )

            # 加载快照 (会清空当前存储并重新加载)
            pm.load_snapshot(snapshot_dir)

            # 获取恢复后的统计
            stats = store.get_stats()

            logger.info(
                "恢复知识库: %s (snapshot=%s, entities=%d, triples=%d, chunks=%d)",
                kb_id, snapshot_id,
                stats.total_entities, stats.total_triples, stats.total_chunks,
            )

            return {
                "kb_id": kb_id,
                "snapshot_id": snapshot_id,
                "restored": True,
                "stats": stats.model_dump(mode="json"),
            }

    def list_snapshots(self, kb_id: str) -> list[KBSnapshot]:
        """列出知识库的所有快照.

        Args:
            kb_id: 知识库 ID

        Returns:
            快照列表 (按时间戳降序排列)

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            self._get_kb(kb_id)
            snapshots = self._snapshots.get(kb_id, [])
            # 返回副本, 按时间戳降序排列
            return sorted(
                list(snapshots), key=lambda s: s.timestamp, reverse=True
            )

    def cleanup_old_snapshots(self, kb_id: str) -> int:
        """清理超出最大保留数量的旧快照.

        保留最新的 max_snapshots 个快照, 删除其余快照及其磁盘文件。

        Args:
            kb_id: 知识库 ID

        Returns:
            删除的快照数量

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)
            max_snapshots = config.max_snapshots
            snapshots = self._snapshots.get(kb_id, [])

            if len(snapshots) <= max_snapshots:
                return 0

            # 按时间戳降序排列, 保留最新的 max_snapshots 个
            sorted_snaps = sorted(
                snapshots, key=lambda s: s.timestamp, reverse=True
            )
            to_keep = sorted_snaps[:max_snapshots]
            to_remove = sorted_snaps[max_snapshots:]

            # 删除旧快照的磁盘文件
            base_path = self._base_paths.get(kb_id)
            for snap in to_remove:
                if base_path:
                    snap_dir = base_path / f"snapshot_{snap.snapshot_id}"
                    if snap_dir.exists():
                        shutil.rmtree(snap_dir, ignore_errors=True)

            self._snapshots[kb_id] = to_keep
            removed_count = len(to_remove)

            logger.info(
                "清理旧快照: %s (removed=%d, remaining=%d)",
                kb_id, removed_count, len(to_keep),
            )
            return removed_count

    # ================================================================
    # 维护操作
    # ================================================================

    def run_gc(self, kb_id: str) -> dict[str, Any]:
        """执行垃圾回收, 清理孤儿实体.

        孤儿实体: 不被任何三元组引用 (既不是主语也不是宾语) 的实体。
        清理孤儿实体可以释放存储空间并提高查询效率。

        Args:
            kb_id: 知识库 ID

        Returns:
            GC 结果字典, 包含清理的实体数量

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)

            # 查找孤儿实体
            orphaned_ids = self._find_orphaned_entities(store)
            removed_count = 0

            for entity_id in orphaned_ids:
                # 跳过非 ACTIVE 状态的实体 (保留草稿/归档等)
                entity = store.get_entity(entity_id)
                if entity is None:
                    continue
                if entity.status != KnowledgeStatus.ACTIVE:
                    continue
                store.remove_entity(entity_id)
                removed_count += 1

            logger.info("垃圾回收: %s (orphaned=%d, removed=%d)",
                        kb_id, len(orphaned_ids), removed_count)

            return {
                "kb_id": kb_id,
                "orphaned_found": len(orphaned_ids),
                "removed": removed_count,
                "remaining_entities": store.entity_count(),
            }

    def enforce_retention(self, kb_id: str) -> dict[str, Any]:
        """执行保留策略, 归档/删除过期条目.

        执行步骤:
        1. 过期归档: 超过 max_age_days 的活跃实体转为 ARCHIVED
        2. 低质量归档: 质量分数低于 quality_threshold 的活跃实体转为 ARCHIVED
        3. 容量限制: 超过 max_entries 时删除最旧的实体
        4. 宽限期删除: 归档超过 deletion_grace_days 的实体永久删除

        Args:
            kb_id: 知识库 ID

        Returns:
            保留策略执行结果字典

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)
            retention = config.retention
            now = time.time()

            archived_count = 0
            deleted_count = 0

            # 步骤 1: 过期归档
            if retention.auto_archive and retention.max_age_days > 0:
                age_cutoff = now - retention.max_age_days * _DAY_SECONDS
                entities = store.entity_store.list_entities(
                    limit=_SIZE_ESTIMATE_SAMPLE_LIMIT
                )
                for entity in entities:
                    if (
                        entity.status == KnowledgeStatus.ACTIVE
                        and entity.created_at < age_cutoff
                    ):
                        store.update_entity(
                            entity.entity_id,
                            status=KnowledgeStatus.ARCHIVED,
                            reason=f"保留策略: 超过 {retention.max_age_days} 天",
                        )
                        archived_count += 1

            # 步骤 2: 低质量归档
            if retention.auto_archive and retention.quality_threshold > 0:
                low_quality_ids = self._find_low_quality_entities(
                    store, retention.quality_threshold
                )
                for eid in low_quality_ids:
                    entity = store.get_entity(eid)
                    if entity and entity.status == KnowledgeStatus.ACTIVE:
                        store.update_entity(
                            eid,
                            status=KnowledgeStatus.ARCHIVED,
                            reason=f"保留策略: 质量分数低于 {retention.quality_threshold}",
                        )
                        archived_count += 1

            # 步骤 3: 容量限制 — 超出 max_entries 时删除最旧实体
            total = store.entity_count()
            if total > retention.max_entries:
                excess = total - retention.max_entries
                entities = sorted(
                    store.entity_store.list_entities(
                        limit=_SIZE_ESTIMATE_SAMPLE_LIMIT
                    ),
                    key=lambda e: e.created_at,
                )
                for entity in entities[:excess]:
                    store.remove_entity(entity.entity_id)
                    deleted_count += 1

            # 步骤 4: 宽限期删除 — 归档超过 grace period 的实体永久删除
            if retention.auto_delete and retention.deletion_grace_days > 0:
                delete_cutoff = now - retention.deletion_grace_days * _DAY_SECONDS
                entities = store.entity_store.list_entities(
                    limit=_SIZE_ESTIMATE_SAMPLE_LIMIT
                )
                for entity in entities:
                    if (
                        entity.status == KnowledgeStatus.ARCHIVED
                        and entity.updated_at < delete_cutoff
                    ):
                        store.remove_entity(entity.entity_id)
                        deleted_count += 1

            remaining = store.entity_count()

            logger.info(
                "执行保留策略: %s (archived=%d, deleted=%d, remaining=%d)",
                kb_id, archived_count, deleted_count, remaining,
            )

            return {
                "kb_id": kb_id,
                "archived": archived_count,
                "deleted": deleted_count,
                "remaining_entities": remaining,
                "retention_policy": retention.model_dump(mode="json"),
            }

    def check_health(self, kb_id: str) -> HealthStatus:
        """检查知识库健康状态.

        检查维度:
        - 实体/三元组/切片总数
        - 孤儿实体数量及比例
        - 低质量实体数量及比例
        - 估算存储大小
        - 最后备份时间
        - 综合健康等级 (healthy / warning / critical)

        Args:
            kb_id: 知识库 ID

        Returns:
            健康状态信息

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)

            stats = store.get_stats()
            total_entities = stats.total_entities
            total_triples = stats.total_triples
            total_chunks = stats.total_chunks

            # 孤儿实体检测
            orphaned_ids = self._find_orphaned_entities(store)
            orphaned_count = len(orphaned_ids)

            # 低质量实体检测
            low_quality_ids = self._find_low_quality_entities(
                store, config.retention.quality_threshold
            )
            low_quality_count = len(low_quality_ids)

            # 估算存储大小
            storage_bytes = self._estimate_storage_size(store)
            storage_size_mb = round(storage_bytes / (1024 * 1024), 2)

            # 最后备份时间
            snapshots = self._snapshots.get(kb_id, [])
            last_backup_time: float | None = None
            if snapshots:
                last_backup_time = max(s.timestamp for s in snapshots)

            # 收集问题并判定健康等级
            issues: list[str] = []
            health_state = "healthy"

            orphaned_ratio = orphaned_count / total_entities if total_entities > 0 else 0.0
            low_quality_ratio = (
                low_quality_count / total_entities if total_entities > 0 else 0.0
            )

            if orphaned_ratio > 0.3:
                issues.append(
                    f"孤儿实体比例过高: {orphaned_count}/{total_entities} "
                    f"({orphaned_ratio:.1%})"
                )
            elif orphaned_ratio > 0.1:
                issues.append(
                    f"存在孤儿实体: {orphaned_count}/{total_entities} "
                    f"({orphaned_ratio:.1%})"
                )

            if low_quality_ratio > 0.5:
                issues.append(
                    f"低质量实体比例过高: {low_quality_count}/{total_entities} "
                    f"({low_quality_ratio:.1%})"
                )
            elif low_quality_ratio > 0.2:
                issues.append(
                    f"存在低质量实体: {low_quality_count}/{total_entities} "
                    f"({low_quality_ratio:.1%})"
                )

            # 备份检查
            if total_entities > 0 and last_backup_time is None:
                issues.append("知识库从未备份")
            elif last_backup_time is not None:
                backup_age_hours = (time.time() - last_backup_time) / 3600
                if backup_age_hours > config.backup_interval_hours * 3:
                    issues.append(
                        f"备份已过期: 距上次备份 {backup_age_hours:.1f} 小时"
                    )

            # 状态检查
            if state == KBLifecycleState.DEPRECATED:
                issues.append("知识库处于 DEPRECATED 状态, 不推荐使用")
            if state == KBLifecycleState.READONLY:
                issues.append("知识库处于 READONLY 状态, 不可写入")

            # 判定健康等级
            critical = (
                orphaned_ratio > 0.3
                or low_quality_ratio > 0.5
                or (total_entities > 0 and last_backup_time is None)
            )
            warning = (
                orphaned_ratio > 0.1
                or low_quality_ratio > 0.2
                or (last_backup_time is not None
                    and (time.time() - last_backup_time) / 3600
                    > config.backup_interval_hours * 3)
                or state in (KBLifecycleState.DEPRECATED, KBLifecycleState.READONLY)
            )

            if critical:
                health_state = "critical"
            elif warning:
                health_state = "warning"
            else:
                health_state = "healthy"

            if not issues:
                issues.append("无异常问题")

            return HealthStatus(
                state=health_state,
                total_entities=total_entities,
                total_triples=total_triples,
                total_chunks=total_chunks,
                orphaned_entities=orphaned_count,
                low_quality_entities=low_quality_count,
                storage_size_mb=storage_size_mb,
                last_backup_time=last_backup_time,
                issues=issues,
            )

    # ================================================================
    # 配置管理
    # ================================================================

    def get_config(self, kb_id: str) -> KBConfig | None:
        """获取知识库配置.

        Args:
            kb_id: 知识库 ID

        Returns:
            知识库配置, 若不存在返回 None
        """
        with self._lock:
            if kb_id not in self._kbs:
                return None
            config, _, _ = self._kbs[kb_id]
            return config

    def update_config(self, kb_id: str, updates: dict[str, Any]) -> KBConfig:
        """更新知识库配置.

        支持更新的字段: name, description, domain, retention,
        auto_backup, backup_interval_hours, max_snapshots, created_by。
        不可更新字段: kb_id, created_at。

        retention 字段支持两种更新方式:
        - 传入完整的 RetentionPolicy 对象: 直接替换
        - 传入字典: 与当前保留策略合并更新

        Args:
            kb_id: 知识库 ID
            updates: 配置更新字典

        Returns:
            更新后的配置

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)

            # 不可更新的字段
            protected = {"kb_id", "created_at"}
            new_values: dict[str, Any] = {}

            for key, value in updates.items():
                if key in protected:
                    logger.debug("跳过受保护字段: %s", key)
                    continue
                if not hasattr(config, key):
                    logger.debug("跳过未知字段: %s", key)
                    continue

                # 特殊处理 retention 字段
                if key == "retention":
                    if isinstance(value, dict):
                        # 与当前保留策略合并
                        new_values[key] = config.retention.model_copy(update=value)
                    elif isinstance(value, RetentionPolicy):
                        new_values[key] = value
                    else:
                        logger.warning("retention 字段类型不支持: %s", type(value))
                    continue

                new_values[key] = value

            # 创建更新后的配置
            updated_config = config.model_copy(update=new_values)
            self._kbs[kb_id] = (updated_config, store, state)

            logger.info("更新知识库配置: %s (fields=%s)", kb_id, list(new_values.keys()))
            return updated_config

    # ================================================================
    # 统计与报告
    # ================================================================

    def list_kbs(self) -> list[dict[str, Any]]:
        """列出所有知识库及其状态.

        Returns:
            知识库信息列表, 每项包含 kb_id, name, domain, state 和统计数据
        """
        with self._lock:
            result: list[dict[str, Any]] = []
            for kb_id, (config, store, state) in self._kbs.items():
                stats = store.get_stats()
                result.append({
                    "kb_id": kb_id,
                    "name": config.name,
                    "domain": config.domain,
                    "state": state.value,
                    "total_entities": stats.total_entities,
                    "total_triples": stats.total_triples,
                    "total_chunks": stats.total_chunks,
                    "created_at": config.created_at,
                    "snapshot_count": len(self._snapshots.get(kb_id, [])),
                })
            return result

    def get_stats(self, kb_id: str) -> dict[str, Any]:
        """获取知识库详细统计信息.

        Args:
            kb_id: 知识库 ID

        Returns:
            统计信息字典, 包含配置、状态、统计数据和快照数量

        Raises:
            KBManagerError: 知识库不存在
        """
        with self._lock:
            config, store, state = self._get_kb(kb_id)
            stats = store.get_stats()

            return {
                "kb_id": kb_id,
                "name": config.name,
                "domain": config.domain,
                "state": state.value,
                "stats": stats.model_dump(mode="json"),
                "snapshot_count": len(self._snapshots.get(kb_id, [])),
                "max_snapshots": config.max_snapshots,
                "auto_backup": config.auto_backup,
            }

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _get_kb(
        self, kb_id: str
    ) -> tuple[KBConfig, KnowledgeStore, KBLifecycleState]:
        """获取知识库组件 (内部方法, 调用者须持有锁).

        Args:
            kb_id: 知识库 ID

        Returns:
            (config, store, state) 三元组

        Raises:
            KBManagerError: 知识库不存在
        """
        if kb_id not in self._kbs:
            raise KBManagerError(
                kb_id=kb_id,
                detail=f"知识库不存在: {kb_id}",
            )
        return self._kbs[kb_id]

    def _validate_transition(
        self, current: KBLifecycleState, new: KBLifecycleState
    ) -> bool:
        """验证状态流转是否允许.

        允许的流转路径见 _ALLOWED_TRANSITIONS 常量。

        Args:
            current: 当前状态
            new: 目标状态

        Returns:
            是否允许流转
        """
        allowed = _ALLOWED_TRANSITIONS.get(current, set())
        return new in allowed

    def _find_orphaned_entities(self, store: KnowledgeStore) -> list[str]:
        """查找孤儿实体 (不被任何三元组引用的实体).

        孤儿实体: 既不是任何三元组的主语, 也不是任何三元组的宾语
        (非字面值宾语) 的实体。

        Args:
            store: 知识存储引擎

        Returns:
            孤儿实体 ID 列表
        """
        # 收集所有实体 ID
        entities = store.entity_store.list_entities(limit=_SIZE_ESTIMATE_SAMPLE_LIMIT)
        all_entity_ids: set[str] = {e.entity_id for e in entities}

        # 收集所有三元组引用的实体 ID (主语 + 非字面值宾语)
        # 注: 直接访问 triple_store 内部字典, 与 persistence.py 保持一致
        referenced_ids: set[str] = set()
        for triple in store.triple_store._triples.values():
            referenced_ids.add(triple.subject_id)
            if not triple.object_is_literal and triple.object_id:
                referenced_ids.add(triple.object_id)

        # 孤儿 = 全部实体 - 被引用实体
        orphaned = all_entity_ids - referenced_ids
        return sorted(orphaned)

    def _find_low_quality_entities(
        self, store: KnowledgeStore, threshold: float
    ) -> list[str]:
        """查找质量分数低于阈值的实体.

        仅检查已评分实体 (quality 非 None), 未评分实体不计入。

        Args:
            store: 知识存储引擎
            threshold: 质量分数阈值 [0.0, 1.0]

        Returns:
            低质量实体 ID 列表
        """
        entities = store.entity_store.list_entities(limit=_SIZE_ESTIMATE_SAMPLE_LIMIT)
        low_quality: list[str] = []
        for entity in entities:
            if entity.quality is not None:
                try:
                    if entity.quality.overall() < threshold:
                        low_quality.append(entity.entity_id)
                except Exception:
                    # 质量分数计算异常时跳过
                    logger.debug("质量分数计算异常: %s", entity.entity_id)
        return low_quality

    def _compute_checksum(self, data: bytes) -> str:
        """计算数据的 SHA-256 校验和.

        Args:
            data: 待计算的字节数据

        Returns:
            十六进制校验和字符串
        """
        return hashlib.sha256(data).hexdigest()

    def _estimate_storage_size(self, store: KnowledgeStore) -> int:
        """估算知识库存储大小 (字节).

        通过序列化实体、三元组和切片的 JSON 表示来估算实际存储大小。
        对于超大知识库, 采样上限为 _SIZE_ESTIMATE_SAMPLE_LIMIT。

        Args:
            store: 知识存储引擎

        Returns:
            估算的存储字节数
        """
        total_bytes = 0

        # 实体大小
        entities = store.entity_store.list_entities(
            limit=_SIZE_ESTIMATE_SAMPLE_LIMIT
        )
        for entity in entities:
            try:
                data = json.dumps(
                    entity.model_dump(mode="json"), ensure_ascii=False
                ).encode("utf-8")
                total_bytes += len(data)
            except Exception:
                total_bytes += 512  # 估算默认大小

        # 三元组大小
        # 注: 直接访问 triple_store 内部字典, 与 persistence.py 保持一致
        for triple in store.triple_store._triples.values():
            try:
                data = json.dumps(
                    triple.model_dump(mode="json"), ensure_ascii=False
                ).encode("utf-8")
                total_bytes += len(data)
            except Exception:
                total_bytes += 256

        # 切片大小
        for chunk in store.chunk_store._chunks.values():
            try:
                data = json.dumps(
                    chunk.model_dump(mode="json"), ensure_ascii=False
                ).encode("utf-8")
                total_bytes += len(data)
            except Exception:
                total_bytes += 1024

        return total_bytes

    def _backup_internal(
        self, kb_id: str, *, description: str = ""
    ) -> KBSnapshot:
        """创建备份快照 (内部方法, 调用者须持有锁).

        Args:
            kb_id: 知识库 ID
            description: 快照描述

        Returns:
            快照信息

        Raises:
            KBManagerError: 知识库不存在或备份失败
        """
        config, store, state = self._get_kb(kb_id)
        pm = self._persistence_managers.get(kb_id)

        if pm is None:
            raise KBManagerError(
                kb_id=kb_id,
                detail="持久化管理器不可用",
            )

        # 生成快照 ID 和路径
        snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"
        base_path = self._base_paths[kb_id]
        snapshot_dir = base_path / f"snapshot_{snapshot_id}"

        # 保存全量快照
        pm.save_snapshot(snapshot_dir)

        # 读取 manifest 获取元信息
        manifest_path = snapshot_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

        counts = manifest.get("counts", {})
        checksum = manifest.get("manifest_checksum", "")

        # 计算快照目录大小
        size_bytes = 0
        for f in snapshot_dir.rglob("*"):
            if f.is_file():
                size_bytes += f.stat().st_size

        # 创建快照信息
        snapshot = KBSnapshot(
            snapshot_id=snapshot_id,
            timestamp=manifest.get("created_at", time.time()),
            version=manifest.get("format_version", _KB_MANAGER_VERSION),
            total_entities=counts.get(
                "entities", store.entity_count()
            ),
            total_triples=counts.get(
                "triples", store.triple_count()
            ),
            total_chunks=counts.get(
                "chunks", store.chunk_count()
            ),
            size_bytes=size_bytes,
            checksum=checksum,
            description=description,
        )

        # 记录快照
        self._snapshots.setdefault(kb_id, []).append(snapshot)

        logger.info(
            "创建快照: %s (snapshot=%s, entities=%d, triples=%d, chunks=%d, "
            "size=%d bytes)",
            kb_id, snapshot_id,
            snapshot.total_entities, snapshot.total_triples,
            snapshot.total_chunks, size_bytes,
        )

        return snapshot

    def _find_snapshot(
        self, kb_id: str, snapshot_id: str
    ) -> KBSnapshot | None:
        """查找指定快照 (内部方法, 调用者须持有锁).

        Args:
            kb_id: 知识库 ID
            snapshot_id: 快照 ID

        Returns:
            快照信息, 若不存在返回 None
        """
        for snap in self._snapshots.get(kb_id, []):
            if snap.snapshot_id == snapshot_id:
                return snap
        return None


__all__ = [
    "KBManagerError",
    "KBLifecycleState",
    "RetentionPolicy",
    "HealthStatus",
    "KBSnapshot",
    "KBConfig",
    "KnowledgeBaseManager",
]
