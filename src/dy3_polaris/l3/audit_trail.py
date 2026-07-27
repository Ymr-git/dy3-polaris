"""L3 领域知识层 — 不可变审计轨迹.

借鉴世界先进方案的审计日志设计:
- Neo4j 事务事件日志: 事务级操作记录 + 前后状态快照
- Wikidata 编辑历史: 实体级完整编辑历史 + 差异追踪
- 企业审计标准 (SOC 2 / ISO 27001): 不可篡改 + 追溯链 + 完整性校验
- Git 提交日志: append-only + 可重放 + 可压缩归档
- Elasticsearch audit log: 多维度查询 + 结构化导出

核心特性:
1. Append-only: 日志只能追加，不可修改或删除
2. 不可变: 已写入的条目不可篡改
3. 完整性: 记录操作前后的完整状态 + 字段级差异
4. 可查询: 支持多维度过滤查询 (用户/操作/资源/时间/链路)
5. 可导出: 支持 JSON/CSV 格式导出
6. 可重放: 支持从审计日志按时间顺序重放操作
7. 压缩: 支持历史日志压缩归档 (保留每资源最新 N 条)
8. 完整性校验: 校验条目 ID 唯一性 + 时间戳单调性
9. 密封: 支持密封审计轨迹，密封后不可再追加

线程安全: 所有共享状态通过 threading.RLock 保护。
无外部依赖: 仅使用标准库 + pydantic v2。
"""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from enum import Enum
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import L3Error


# ============================================================
# 模块常量
# ============================================================

# 压缩后每个资源保留的最新条目数 (借鉴 Git gc 保留策略)
_COMPRESS_KEEP_PER_RESOURCE: int = 20

# CSV 导出列定义
_CSV_COLUMNS: list[str] = [
    "entry_id",
    "timestamp",
    "operation",
    "resource_type",
    "resource_id",
    "user_id",
    "trace_id",
]


# ============================================================
# 枚举定义
# ============================================================


class OperationType(str, Enum):
    """操作类型 (借鉴 Neo4j 事务操作 + Wikidata 编辑类型).

    覆盖知识存储引擎的全部操作语义:
        CREATE:        创建资源
        UPDATE:        更新资源
        DELETE:        删除资源
        MERGE:         合并资源 (实体去重/融合)
        IMPORT:        批量导入资源
        EXPORT:        批量导出资源
        RESTORE:       从快照/归档恢复资源
        SCHEMA_CHANGE: 本体/索引结构变更
    """

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    IMPORT = "import"
    EXPORT = "export"
    RESTORE = "restore"
    SCHEMA_CHANGE = "schema_change"


class ResourceType(str, Enum):
    """资源类型 (借鉴 Neo4j 子图资源分类).

    审计轨迹追踪的资源类别:
        ENTITY:   知识实体
        TRIPLE:   三元组 (SPO)
        CHUNK:    文档切片
        ONTOLOGY: 本体定义
        INDEX:    索引结构
    """

    ENTITY = "entity"
    TRIPLE = "triple"
    CHUNK = "chunk"
    ONTOLOGY = "ontology"
    INDEX = "index"


# ============================================================
# 异常定义
# ============================================================


class AuditError(L3Error):
    """审计轨迹异常.

    当审计轨迹操作失败时抛出，包括:
    - 密封后尝试追加条目
    - 条目数超过最大限制
    - 完整性校验失败

    继承 L3Error 异常体系，集成 JSON-RPC 错误码 (-32416)。

    Attributes:
        reason: 错误原因标识
    """

    def __init__(
        self,
        reason: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        super().__init__(
            "L3_AUDIT_ERROR",
            detail or f"reason={reason}",
            {"reason": reason, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32416


# ============================================================
# 数据模型 (pydantic v2)
# ============================================================


class ChangeDiff(BaseModel):
    """字段级变更差异 (借鉴 Git diff + Wikidata 编辑差异).

    表示单个字段在一次操作中的变更:
        field:       变更的字段名
        old_value:   变更前的值 (新增字段时为 None)
        new_value:   变更后的值 (删除字段时为 None)
        change_type: 变更类型 ("added" / "removed" / "modified")
    """

    field: str = Field(..., description="变更的字段名")
    old_value: Any = Field(default=None, description="变更前的值")
    new_value: Any = Field(default=None, description="变更后的值")
    change_type: str = Field(..., description="变更类型: added/removed/modified")


class AuditEntry(BaseModel):
    """审计条目 (借鉴 Neo4j 事务事件 + Wikidata 编辑记录).

    一条审计条目记录一次资源操作的完整信息:
    - 操作类型与资源标识
    - 操作前后的完整状态快照
    - 字段级变更差异
    - 操作者、链路追踪 ID 和扩展元数据

    Attributes:
        entry_id:      条目唯一标识 (UUID)
        timestamp:     操作时间戳 (Unix epoch 秒)
        operation:     操作类型
        resource_type: 资源类型
        resource_id:   资源唯一标识
        user_id:       操作者用户 ID
        before_state:  操作前的序列化状态
        after_state:   操作后的序列化状态
        diffs:         字段级变更差异列表
        metadata:      扩展元数据
        trace_id:      链路追踪 ID
    """

    entry_id: str = Field(..., description="条目唯一标识 (UUID)")
    timestamp: float = Field(..., description="操作时间戳 (Unix epoch 秒)")
    operation: OperationType = Field(..., description="操作类型")
    resource_type: ResourceType = Field(..., description="资源类型")
    resource_id: str = Field(..., description="资源唯一标识")
    user_id: str = Field(..., description="操作者用户 ID")
    before_state: dict[str, Any] = Field(
        default_factory=dict, description="操作前的序列化状态"
    )
    after_state: dict[str, Any] = Field(
        default_factory=dict, description="操作后的序列化状态"
    )
    diffs: list[ChangeDiff] = Field(
        default_factory=list, description="字段级变更差异列表"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="扩展元数据"
    )
    trace_id: str = Field(default="", description="链路追踪 ID")


class AuditQuery(BaseModel):
    """审计查询条件 (借鉴 Elasticsearch query DSL).

    支持多维度过滤:
        user_id:       按操作者过滤
        operation:     按操作类型过滤
        resource_type: 按资源类型过滤
        resource_id:   按资源 ID 过滤
        start_time:    起始时间 (包含)
        end_time:      结束时间 (包含)
        trace_id:      按链路追踪 ID 过滤
        limit:         返回条目上限 (默认 100)
        offset:        分页偏移量 (默认 0)
    """

    user_id: str | None = Field(default=None, description="按操作者过滤")
    operation: OperationType | None = Field(default=None, description="按操作类型过滤")
    resource_type: ResourceType | None = Field(
        default=None, description="按资源类型过滤"
    )
    resource_id: str | None = Field(default=None, description="按资源 ID 过滤")
    start_time: float | None = Field(default=None, description="起始时间 (包含)")
    end_time: float | None = Field(default=None, description="结束时间 (包含)")
    trace_id: str | None = Field(default=None, description="按链路追踪 ID 过滤")
    limit: int = Field(default=100, ge=0, description="返回条目上限")
    offset: int = Field(default=0, ge=0, description="分页偏移量")


class AuditStats(BaseModel):
    """审计轨迹统计信息.

    汇总审计轨迹的整体状况:
        total_entries:        活跃条目总数
        entries_by_operation: 按操作类型分组的条目数
        entries_by_resource:  按资源类型分组的条目数
        entries_by_user:      按用户分组的条目数
        time_range:           时间范围 (最早时间戳, 最晚时间戳)
    """

    total_entries: int = Field(..., description="活跃条目总数")
    entries_by_operation: dict[str, int] = Field(
        default_factory=dict, description="按操作类型分组的条目数"
    )
    entries_by_resource: dict[str, int] = Field(
        default_factory=dict, description="按资源类型分组的条目数"
    )
    entries_by_user: dict[str, int] = Field(
        default_factory=dict, description="按用户分组的条目数"
    )
    time_range: tuple[float, float] | None = Field(
        default=None, description="时间范围 (最早, 最晚)"
    )


# ============================================================
# 审计轨迹 — 不可变操作日志
# ============================================================


class AuditTrail:
    """不可变审计轨迹 (借鉴 Neo4j 事务事件日志 + Wikidata 编辑历史 + 企业审计标准).

    特性:
    1. Append-only: 日志只能追加，不可修改或删除
    2. 不可变: 已写入的条目不可篡改
    3. 完整性: 记录操作前后的完整状态
    4. 可查询: 支持多维度过滤查询
    5. 可导出: 支持 JSON/CSV 导出
    6. 可重放: 支持从审计日志重放操作
    7. 压缩: 支持历史日志压缩归档

    设计借鉴:
        - Neo4j 事务事件日志: 事务级操作记录 + 前后状态快照
        - Wikidata 编辑历史: 实体级完整编辑历史 + 差异追踪
        - 企业审计标准 (SOC 2 / ISO 27001): 不可篡改 + 追溯链
        - Git 提交日志: append-only + 可重放 + 可压缩
        - Elasticsearch audit log: 多维度查询 + 结构化导出

    线程安全: 所有共享状态通过 threading.RLock 保护。

    Attributes:
        _entries: 审计条目列表 (按追加顺序)
        _index_by_resource: 资源 ID -> 条目索引列表
        _index_by_user: 用户 ID -> 条目索引列表
        _max_entries: 最大条目数限制
        _auto_compress_threshold: 自动压缩阈值
        _lock: 线程安全锁 (可重入)
        _sealed: 密封标志 (密封后不可追加)
        _compressed_entries: 已压缩归档的条目列表
    """

    def __init__(
        self,
        *,
        max_entries: int = 100000,
        auto_compress_threshold: int = 50000,
    ) -> None:
        """初始化审计轨迹.

        Args:
            max_entries: 最大条目数限制，超过则拒绝追加 (默认 100000)
            auto_compress_threshold: 自动压缩阈值，达到时触发压缩 (默认 50000)
        """
        self._entries: list[AuditEntry] = []
        self._index_by_resource: dict[str, list[int]] = {}  # resource_id -> 条目索引列表
        self._index_by_user: dict[str, list[int]] = {}  # user_id -> 条目索引列表
        self._max_entries = max_entries
        self._auto_compress_threshold = auto_compress_threshold
        self._lock = RLock()
        self._sealed: bool = False  # 密封标志: 密封后不可追加新条目
        self._compressed_entries: list[AuditEntry] = []  # 已压缩归档的条目

    # ============================================================
    # 日志追加
    # ============================================================

    def log(
        self,
        operation: OperationType,
        resource_type: ResourceType,
        resource_id: str,
        user_id: str,
        *,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> AuditEntry:
        """记录一条审计条目 (借鉴 Neo4j 事务事件 + Wikidata 编辑记录).

        创建审计条目，计算字段级差异，追加到日志并更新索引。
        当条目数达到自动压缩阈值时触发自动压缩。

        Args:
            operation:     操作类型
            resource_type: 资源类型
            resource_id:   资源唯一标识
            user_id:       操作者用户 ID
            before:        操作前的序列化状态 (CREATE 操作时为 None)
            after:         操作后的序列化状态 (DELETE 操作时为 None)
            metadata:      扩展元数据
            trace_id:      链路追踪 ID

        Returns:
            创建的审计条目

        Raises:
            AuditError: 审计轨迹已密封或条目数超过最大限制
        """
        with self._lock:
            # 检查密封状态
            if self._sealed:
                raise AuditError(
                    reason="sealed",
                    detail="审计轨迹已密封，不可追加新条目",
                    context={"entry_count": len(self._entries)},
                )

            # 检查最大条目数限制
            if len(self._entries) >= self._max_entries:
                raise AuditError(
                    reason="max_entries_exceeded",
                    detail=f"审计条目数已达上限 {self._max_entries}",
                    context={
                        "max_entries": self._max_entries,
                        "current_count": len(self._entries),
                    },
                )

            # 准备前后状态 (None 转为空字典)
            before_state: dict[str, Any] = dict(before) if before else {}
            after_state: dict[str, Any] = dict(after) if after else {}

            # 计算字段级差异
            diffs = self.compute_diff(before_state, after_state)

            # 创建审计条目
            entry = AuditEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=time.time(),
                operation=operation,
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id,
                before_state=before_state,
                after_state=after_state,
                diffs=diffs,
                metadata=dict(metadata) if metadata else {},
                trace_id=trace_id,
            )

            # 追加到日志列表
            entry_index = len(self._entries)
            self._entries.append(entry)

            # 更新索引
            self._index_by_resource.setdefault(resource_id, []).append(entry_index)
            self._index_by_user.setdefault(user_id, []).append(entry_index)

            # 自动压缩检查: 达到阈值时触发压缩
            if len(self._entries) >= self._auto_compress_threshold:
                try:
                    self.compress()
                except Exception:
                    # 压缩失败不应阻塞日志记录
                    pass

            return entry

    # ============================================================
    # 差异计算
    # ============================================================

    def compute_diff(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> list[ChangeDiff]:
        """计算两个状态之间的字段级差异 (借鉴 Git diff + Wikidata 编辑差异).

        逐字段比较前后状态，检测三种变更:
        - added:    字段仅存在于新状态
        - removed:  字段仅存在于旧状态
        - modified: 字段在两个状态中都存在但值不同

        Args:
            before: 操作前的状态字典
            after:  操作后的状态字典

        Returns:
            变更差异列表
        """
        diffs: list[ChangeDiff] = []
        before = before or {}
        after = after or {}

        # 合并所有字段名 (保持排序以确保输出稳定)
        all_fields = set(before.keys()) | set(after.keys())

        for field_name in sorted(all_fields):
            in_before = field_name in before
            in_after = field_name in after

            if in_before and not in_after:
                # 字段被删除
                diffs.append(
                    ChangeDiff(
                        field=field_name,
                        old_value=before[field_name],
                        new_value=None,
                        change_type="removed",
                    )
                )
            elif not in_before and in_after:
                # 字段被新增
                diffs.append(
                    ChangeDiff(
                        field=field_name,
                        old_value=None,
                        new_value=after[field_name],
                        change_type="added",
                    )
                )
            elif in_before and in_after:
                # 字段已存在，检查值是否变更
                old_val = before[field_name]
                new_val = after[field_name]
                if old_val != new_val:
                    diffs.append(
                        ChangeDiff(
                            field=field_name,
                            old_value=old_val,
                            new_value=new_val,
                            change_type="modified",
                        )
                    )

        return diffs

    # ============================================================
    # 查询
    # ============================================================

    def query(self, query: AuditQuery) -> list[AuditEntry]:
        """多维度过滤查询审计条目 (借鉴 Elasticsearch query DSL).

        支持按用户、操作类型、资源类型、资源 ID、时间范围、
        链路追踪 ID 过滤，结果按时间戳降序排列。

        Args:
            query: 查询条件

        Returns:
            匹配的审计条目列表 (按时间戳降序)
        """
        with self._lock:
            # 收集所有匹配的条目
            matched: list[AuditEntry] = []

            for entry in self._entries:
                if not self._matches_query(entry, query):
                    continue
                matched.append(entry)

            # 按时间戳降序排列 (最新优先)
            matched.sort(key=lambda e: e.timestamp, reverse=True)

            # 应用分页 (offset + limit)
            start = query.offset
            end = start + query.limit
            return matched[start:end]

    def get_resource_history(
        self, resource_id: str, *, limit: int = 50
    ) -> list[AuditEntry]:
        """获取指定资源的完整操作历史 (借鉴 Wikidata 实体编辑历史).

        Args:
            resource_id: 资源唯一标识
            limit:       返回条目上限 (默认 50)

        Returns:
            该资源的审计条目列表 (按时间戳降序)
        """
        with self._lock:
            indices = self._index_by_resource.get(resource_id, [])
            entries = [self._entries[i] for i in indices]
            # 按时间戳降序排列
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            return entries[:limit]

    def get_user_activity(
        self, user_id: str, *, limit: int = 50
    ) -> list[AuditEntry]:
        """获取指定用户的所有操作活动 (借鉴 GitHub 用户活动流).

        Args:
            user_id: 用户 ID
            limit:   返回条目上限 (默认 50)

        Returns:
            该用户的审计条目列表 (按时间戳降序)
        """
        with self._lock:
            indices = self._index_by_user.get(user_id, [])
            entries = [self._entries[i] for i in indices]
            # 按时间戳降序排列
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            return entries[:limit]

    # ============================================================
    # 重放
    # ============================================================

    def replay(
        self,
        *,
        from_timestamp: float | None = None,
        to_timestamp: float | None = None,
        resource_id: str | None = None,
    ) -> list[AuditEntry]:
        """获取用于重放的审计条目 (借鉴 Git log + 数据库重做日志).

        按时间戳升序返回条目，支持时间范围和资源 ID 过滤。
        重放时按此顺序依次应用操作即可重建状态。

        Args:
            from_timestamp: 起始时间戳 (包含, 默认无限制)
            to_timestamp:   结束时间戳 (包含, 默认无限制)
            resource_id:    限定资源 ID (默认无限制)

        Returns:
            审计条目列表 (按时间戳升序)
        """
        with self._lock:
            matched: list[AuditEntry] = []

            for entry in self._entries:
                # 时间范围过滤
                if from_timestamp is not None and entry.timestamp < from_timestamp:
                    continue
                if to_timestamp is not None and entry.timestamp > to_timestamp:
                    continue
                # 资源 ID 过滤
                if resource_id is not None and entry.resource_id != resource_id:
                    continue
                matched.append(entry)

            # 按时间戳升序排列 (重放顺序: 从最早到最晚)
            matched.sort(key=lambda e: e.timestamp)
            return matched

    # ============================================================
    # 导出
    # ============================================================

    def export_json(self, query: AuditQuery | None = None) -> str:
        """导出审计条目为 JSON 字符串 (借鉴 Elasticsearch _search 导出).

        Args:
            query: 查询条件 (None 表示导出全部活跃条目)

        Returns:
            JSON 格式的审计条目列表字符串 (indent=2)
        """
        with self._lock:
            if query is not None:
                entries = self.query(query)
            else:
                entries = list(self._entries)

            # 使用 pydantic 的 model_dump(mode="json") 确保枚举等类型可序列化
            data = [entry.model_dump(mode="json") for entry in entries]
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def export_csv(self, query: AuditQuery | None = None) -> str:
        """导出审计条目为 CSV 字符串.

        CSV 列: entry_id, timestamp, operation, resource_type,
                resource_id, user_id, trace_id

        Args:
            query: 查询条件 (None 表示导出全部活跃条目)

        Returns:
            CSV 格式的审计条目字符串
        """
        with self._lock:
            if query is not None:
                entries = self.query(query)
            else:
                entries = list(self._entries)

            output = io.StringIO()
            writer = csv.writer(output)

            # 写入表头
            writer.writerow(_CSV_COLUMNS)

            # 写入数据行
            for entry in entries:
                writer.writerow(
                    [
                        entry.entry_id,
                        entry.timestamp,
                        entry.operation.value,
                        entry.resource_type.value,
                        entry.resource_id,
                        entry.user_id,
                        entry.trace_id,
                    ]
                )

            return output.getvalue()

    # ============================================================
    # 统计
    # ============================================================

    def get_stats(self) -> AuditStats:
        """获取审计轨迹统计信息 (借鉴 Prometheus 统计指标).

        Returns:
            AuditStats 统计信息对象
        """
        with self._lock:
            total = len(self._entries)
            by_operation: dict[str, int] = {}
            by_resource: dict[str, int] = {}
            by_user: dict[str, int] = {}
            timestamps: list[float] = []

            for entry in self._entries:
                # 按操作类型统计
                op_key = entry.operation.value
                by_operation[op_key] = by_operation.get(op_key, 0) + 1

                # 按资源类型统计
                res_key = entry.resource_type.value
                by_resource[res_key] = by_resource.get(res_key, 0) + 1

                # 按用户统计
                by_user[entry.user_id] = by_user.get(entry.user_id, 0) + 1

                timestamps.append(entry.timestamp)

            # 计算时间范围
            time_range: tuple[float, float] | None = None
            if timestamps:
                time_range = (min(timestamps), max(timestamps))

            return AuditStats(
                total_entries=total,
                entries_by_operation=by_operation,
                entries_by_resource=by_resource,
                entries_by_user=by_user,
                time_range=time_range,
            )

    # ============================================================
    # 生命周期管理
    # ============================================================

    def seal(self) -> None:
        """密封审计轨迹 (借鉴 Git tag + 区块链 finality).

        密封后不可再追加新条目，但查询、导出、统计等功能仍可使用。
        密封操作不可逆，用于确保审计轨迹的最终性。
        """
        with self._lock:
            self._sealed = True

    def compress(self) -> int:
        """压缩归档旧条目 (借鉴 Git gc + 数据库日志压缩).

        对每个资源，仅保留最新的 N 条审计条目
        (N = _COMPRESS_KEEP_PER_RESOURCE)，
        较旧的条目移动到归档列表 (_compressed_entries) 中。
        归档条目不再参与常规查询，但可通过 compressed_count 属性统计。

        压缩后重建索引以确保索引与条目列表一致。

        Returns:
            被压缩归档的条目数
        """
        with self._lock:
            if not self._entries:
                return 0

            keep_count = _COMPRESS_KEEP_PER_RESOURCE

            # 按资源 ID 分组条目 (保留原始索引)
            by_resource: dict[str, list[tuple[int, AuditEntry]]] = {}
            for i, entry in enumerate(self._entries):
                by_resource.setdefault(entry.resource_id, []).append((i, entry))

            # 确定需要压缩的条目索引集合
            compress_indices: set[int] = set()
            for resource_id, items in by_resource.items():
                if len(items) > keep_count:
                    # 按时间戳升序排列，保留最新的 N 条
                    items.sort(key=lambda x: x[1].timestamp)
                    for idx, _entry in items[:-keep_count]:
                        compress_indices.add(idx)

            if not compress_indices:
                return 0

            # 分离活跃条目和压缩条目 (条目本身不被修改，仅移动位置)
            new_entries: list[AuditEntry] = []
            compressed: list[AuditEntry] = []
            for i, entry in enumerate(self._entries):
                if i in compress_indices:
                    compressed.append(entry)
                else:
                    new_entries.append(entry)

            # 更新存储: 压缩条目移入归档列表
            self._compressed_entries.extend(compressed)
            self._entries = new_entries

            # 重建索引 (因为条目位置发生了变化)
            self._rebuild_indices()

            return len(compressed)

    # ============================================================
    # 完整性校验
    # ============================================================

    def verify_integrity(self) -> bool:
        """校验审计轨迹完整性 (借鉴区块链校验 + Merkle 树验证).

        校验内容:
        1. 所有条目 ID 唯一 (无重复)
        2. 时间戳单调非递减 (按追加顺序)

        Returns:
            完整性校验是否通过
        """
        with self._lock:
            seen_ids: set[str] = set()
            prev_timestamp: float | None = None

            for entry in self._entries:
                # 检查条目 ID 唯一性
                if entry.entry_id in seen_ids:
                    return False
                seen_ids.add(entry.entry_id)

                # 检查时间戳单调非递减
                if prev_timestamp is not None and entry.timestamp < prev_timestamp:
                    return False
                prev_timestamp = entry.timestamp

            return True

    # ============================================================
    # 辅助方法
    # ============================================================

    def count(self) -> int:
        """获取当前活跃审计条目总数.

        Returns:
            活跃条目数 (不含已压缩归档的条目)
        """
        with self._lock:
            return len(self._entries)

    @property
    def sealed(self) -> bool:
        """审计轨迹是否已密封."""
        with self._lock:
            return self._sealed

    @property
    def compressed_count(self) -> int:
        """已压缩归档的条目数."""
        with self._lock:
            return len(self._compressed_entries)

    def __len__(self) -> int:
        """返回活跃条目数 (支持 len(audit_trail))."""
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        """返回审计轨迹的简要描述."""
        with self._lock:
            return (
                f"AuditTrail(entries={len(self._entries)}, "
                f"compressed={len(self._compressed_entries)}, "
                f"sealed={self._sealed})"
            )

    # ============================================================
    # 内部方法
    # ============================================================

    def _matches_query(self, entry: AuditEntry, query: AuditQuery) -> bool:
        """检查条目是否匹配查询条件 (内部方法).

        Args:
            entry: 待检查的审计条目
            query: 查询条件

        Returns:
            是否匹配所有过滤条件
        """
        # 用户 ID 过滤
        if query.user_id is not None and entry.user_id != query.user_id:
            return False

        # 操作类型过滤
        if query.operation is not None and entry.operation != query.operation:
            return False

        # 资源类型过滤
        if (
            query.resource_type is not None
            and entry.resource_type != query.resource_type
        ):
            return False

        # 资源 ID 过滤
        if query.resource_id is not None and entry.resource_id != query.resource_id:
            return False

        # 时间范围过滤 (包含边界)
        if query.start_time is not None and entry.timestamp < query.start_time:
            return False

        if query.end_time is not None and entry.timestamp > query.end_time:
            return False

        # 链路追踪 ID 过滤
        if query.trace_id is not None and entry.trace_id != query.trace_id:
            return False

        return True

    def _rebuild_indices(self) -> None:
        """重建资源索引和用户索引 (内部方法).

        在压缩操作后调用，确保索引与条目列表一致。
        压缩会改变条目在列表中的位置，因此需要重建所有索引。
        """
        self._index_by_resource.clear()
        self._index_by_user.clear()

        for i, entry in enumerate(self._entries):
            self._index_by_resource.setdefault(entry.resource_id, []).append(i)
            self._index_by_user.setdefault(entry.user_id, []).append(i)


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    # 枚举
    "OperationType",
    "ResourceType",
    # 异常
    "AuditError",
    # 数据模型
    "ChangeDiff",
    "AuditEntry",
    "AuditQuery",
    "AuditStats",
    # 核心类
    "AuditTrail",
]
