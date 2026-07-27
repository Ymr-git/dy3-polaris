"""L3 领域知识层 — 数据源适配器框架.

融合世界先进方案的数据源适配器框架设计:
- Airbyte Protocol: spec/check/discover/read 四阶段标准化协议 + JSON 消息总线
- Apache SeaTunnel: 引擎无关 API + Split 级并行 + 两阶段提交 (exactly-once)
- MCP (Model Context Protocol): 能力协商 + 三原语 (Resources/Tools/Prompts)
- Limerence: 共享生命周期契约 + 窄幅恢复 (Recoverer 链) + 多层防御
- LangChain Document Loader: 统一 Document 抽象 + 多格式适配
- LlamaIndex BaseReader: 连接器继承体系 + LlamaHub 统一仓库
- Netflix Hystrix: 熔断器三态 (CLOSED/OPEN/HALF_OPEN) — 复用 connector.py
- Kong API Gateway: 分级限流 (PUBLIC/INDUSTRY/PRIVATE 三档) — 复用 connector.py
- Stripe API: 指数退避健康探测 — 复用 connector.py
- Apache SeaTunnel CDC: Snapshot + Incremental 混合同步模型
- Debezium: 变更数据捕获 (CDC) + 日志位置偏移量

框架五层架构:
1. 协议层 (Protocol): spec → check → discover → read 标准化生命周期
2. 适配器层 (Adapter): REST/GraphQL/Database/File/MCP 五种协议适配器基类
3. Schema 层 (Schema): 字段映射 + 类型转换 + Schema 演化
4. 同步层 (Sync): 全量刷新 / 增量游标 / CDC 流式 三种同步模式
5. 治理层 (Governance): 注册发现 + 健康监控 + 熔断保护 + 审计日志

适配器生命周期 (借鉴 Limerence + Airbyte):
    SPEC → CHECK → DISCOVER → READ → TRANSFORM → VALIDATE → PERSIST
    每个阶段可被 Recoverer 链中断和恢复

线程安全: DataAdapterBase 和 DataAdapterRegistry 通过 threading.RLock 保护。
所有适配器均为抽象实现，接口设计支持未来替换为具体协议后端 (HTTP/gRPC/MCP)。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, Flag, auto
from typing import Any, Callable, Iterator

from pydantic import BaseModel, Field

from .connector import (
    CircuitBreaker,
    ConnectorConfig,
    ConnectorHealth,
    ConnectorProtocol,
    ConnectorResponse,
    ConnectorStatus,
    ConnectorTier,
    KnowledgeConnector,
)

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class DataSourceType(str, Enum):
    """数据源类型 (借鉴 Airbyte Source Type + LangChain Loader 分类).

    覆盖七大类数据源:
    - REST_API: RESTful HTTP API (NIST, PubChem, Crossref, arXiv)
    - GRAPHQL: GraphQL 查询接口 (OpenAlex, Reaxys)
    - DATABASE: 关系/文档/图数据库 (PostgreSQL, MongoDB, Neo4j)
    - FILE: 文件系统 (PDF, CSV, JSON, XML, Markdown)
    - MCP_SERVER: MCP 协议服务端 (CIE, ICDD 本地 MCP Server)
    - STREAMING: 流式数据源 (Kafka, CDC binlog, WebSocket)
    - BATCH: 批处理数据源 (S3 批量文件, HDFS, FTP)
    """

    REST_API = "rest_api"
    GRAPHQL = "graphql"
    DATABASE = "database"
    FILE = "file"
    MCP_SERVER = "mcp_server"
    STREAMING = "streaming"
    BATCH = "batch"


class SyncMode(str, Enum):
    """同步模式 (借鉴 Airbyte Sync Mode + SeaTunnel CDC).

    - FULL_REFRESH: 全量刷新，每次读取完整数据集
    - INCREMENTAL: 增量同步，基于游标字段只读取变更
    - CDC: 变更数据捕获，基于日志位置实时流式读取 (binlog/WAL)
    - SNAPSHOT_THEN_INCREMENTAL: 先快照后增量 (SeaTunnel CDC 混合模型)
    """

    FULL_REFRESH = "full_refresh"
    INCREMENTAL = "incremental"
    CDC = "cdc"
    SNAPSHOT_THEN_INCREMENTAL = "snapshot_then_incremental"


class AdapterCapability(Flag):
    """适配器能力标志 (借鉴 MCP 能力协商 + SeaTunnel Source 能力声明).

    使用 Flag 支持组合能力，适配器在 spec 阶段声明自身能力集合。
    """
    SEARCH = auto()           # 支持搜索查询
    FETCH = auto()            # 支持按 ID 获取单条资源
    LIST = auto()             # 支持列出资源目录
    STREAM = auto()           # 支持流式读取 (逐条产出)
    BATCH = auto()            # 支持批量读取 (分页/分块)
    DISCOVER = auto()         # 支持 Schema 自动发现
    SCHEMA_EVOLUTION = auto() # 支持 Schema 动态演化
    INCREMENTAL = auto()      # 支持增量同步 (游标)
    CDC = auto()              # 支持 CDC 流式同步 (日志位置)
    HEALTH_CHECK = auto()     # 支持健康检查
    AUTHENTICATE = auto()     # 需要认证
    RATE_LIMITED = auto()     # 受限流约束
    CACHEABLE = auto()        # 结果可缓存
    SUBSCRIBE = auto()        # 支持订阅变更通知

    # 常见能力组合
    BASIC = SEARCH | FETCH | LIST
    STANDARD = SEARCH | FETCH | LIST | BATCH | DISCOVER | HEALTH_CHECK
    FULL = (
        SEARCH | FETCH | LIST | STREAM | BATCH | DISCOVER
        | SCHEMA_EVOLUTION | INCREMENTAL | CDC | HEALTH_CHECK
    )


class RecoveryAction(str, Enum):
    """恢复动作 (借鉴 Limerence Recoverer 链).

    窄幅恢复策略，仅针对已知失败形态:
    - RETRY: 重试当前操作 (指数退避)
    - RECONNECT: 重新建立连接后重试
    - SKIP: 跳过当前记录，继续处理下一条
    - RESTART: 重启整个同步流程
    - ABORT: 放弃操作，上报错误
    """

    RETRY = "retry"
    RECONNECT = "reconnect"
    SKIP = "skip"
    RESTART = "restart"
    ABORT = "abort"


class LifecyclePhase(str, Enum):
    """适配器生命周期阶段 (借鉴 Airbyte Protocol + Limerence 生命周期).

    SPEC → CHECK → DISCOVER → READ → TRANSFORM → VALIDATE → PERSIST
    每个阶段有明确的输入/输出和可恢复点。
    """

    SPEC = "spec"               # 声明配置规范和能力
    CHECK = "check"             # 验证连通性和认证
    DISCOVER = "discover"       # 发现数据源 Schema
    READ = "read"               # 读取数据 (全量/增量/CDC)
    TRANSFORM = "transform"     # 字段映射和类型转换
    VALIDATE = "validate"       # 数据质量校验
    PERSIST = "persist"         # 持久化到知识库


# ============================================================
# Schema 层数据模型
# ============================================================


class SchemaField(BaseModel):
    """Schema 字段定义 (借鉴 SeaTunnel PhysicalColumn + Airbyte Field).

    描述数据源中单个字段的元信息，支持类型映射和约束声明。

    Attributes:
        name: 字段名 (数据源中的原始名称)
        data_type: 字段数据类型 (string/integer/float/boolean/array/object/datetime/binary)
        nullable: 是否允许 NULL 值
        primary_key: 是否为主键
        description: 字段描述
        default_value: 默认值
        enum_values: 枚举约束 (若为枚举类型)
        format: 格式约束 (如 "date-time", "uri", "email")
        max_length: 最大长度约束
    """

    name: str = Field(..., min_length=1, description="字段名")
    data_type: str = Field(default="string", description="数据类型")
    nullable: bool = Field(default=True, description="是否允许 NULL")
    primary_key: bool = Field(default=False, description="是否为主键")
    description: str = Field(default="", description="字段描述")
    default_value: Any = Field(default=None, description="默认值")
    enum_values: list[str] = Field(default_factory=list, description="枚举约束")
    format: str = Field(default="", description="格式约束")
    max_length: int = Field(default=0, ge=0, description="最大长度")


class DataSourceSchema(BaseModel):
    """数据源 Schema (借鉴 SeaTunnel CatalogTable + Airbyte AirbyteCatalog).

    描述数据源的完整结构信息，支持多流 (Stream) 概念。

    Attributes:
        stream_name: 流名称 (如 "papers", "compounds", "spectra")
        fields: 字段列表
        primary_keys: 主键字段名列表
        cursor_field: 增量同步游标字段 (如 "updated_at", "doi")
        description: 流描述
        metadata: 扩展元数据 (如 {"source": "nist", "format": "json"})
    """

    stream_name: str = Field(..., min_length=1, description="流名称")
    fields: list[SchemaField] = Field(default_factory=list, description="字段列表")
    primary_keys: list[str] = Field(default_factory=list, description="主键字段")
    cursor_field: str = Field(default="", description="增量游标字段")
    description: str = Field(default="", description="流描述")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    def get_field(self, name: str) -> SchemaField | None:
        """按名称获取字段."""
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def field_names(self) -> list[str]:
        """获取所有字段名."""
        return [f.name for f in self.fields]

    def to_json_schema(self) -> dict[str, Any]:
        """转换为 JSON Schema (借鉴 OpenAPI/JSON Schema 规范)."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        type_map = {
            "string": "string",
            "integer": "integer",
            "float": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
            "datetime": "string",
            "binary": "string",
        }
        for f in self.fields:
            prop: dict[str, Any] = {"type": type_map.get(f.data_type, "string")}
            if f.description:
                prop["description"] = f.description
            if f.format:
                prop["format"] = f.format
            if f.enum_values:
                prop["enum"] = f.enum_values
            if f.max_length > 0 and f.data_type == "string":
                prop["maxLength"] = f.max_length
            if f.default_value is not None:
                prop["default"] = f.default_value
            properties[f.name] = prop
            if not f.nullable:
                required.append(f.name)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema


class FieldMapping(BaseModel):
    """字段映射规则 (借鉴 SeaTunnel Schema 映射 + Airbyte 字段同步).

    定义源字段到目标字段的映射关系。

    Attributes:
        source_field: 源字段名
        target_field: 目标字段名
        transform: 转换函数名 (如 "to_lower", "parse_int", "iso_datetime")
        default_value: 源字段缺失时的默认值
        required: 映射是否必须 (源字段不存在则报错)
    """

    source_field: str = Field(..., min_length=1, description="源字段名")
    target_field: str = Field(..., min_length=1, description="目标字段名")
    transform: str = Field(default="", description="转换函数名")
    default_value: Any = Field(default=None, description="默认值")
    required: bool = Field(default=False, description="是否必须")


class SchemaMapper:
    """Schema 映射器 (借鉴 SeaTunnel Schema 映射 + Airbyte Catalog 同步).

    将数据源原始字段映射到 L3 知识实体的标准字段。
    支持字段重命名、类型转换、默认值填充和缺失字段处理。

    内置转换函数:
    - to_lower / to_upper: 大小写转换
    - parse_int / parse_float / parse_bool: 类型解析
    - iso_datetime: ISO 8601 日期时间标准化
    - trim: 去除首尾空白
    - json_parse: JSON 字符串解析
    - split_comma: 逗号分隔转列表
    """

    # 内置转换函数注册表
    _TRANSFORMS: dict[str, Callable[[Any], Any]] = {
        "to_lower": lambda v: str(v).lower() if v is not None else None,
        "to_upper": lambda v: str(v).upper() if v is not None else None,
        "parse_int": lambda v: int(float(v)) if v is not None and v != "" else None,
        "parse_float": lambda v: float(v) if v is not None and v != "" else None,
        "parse_bool": lambda v: (
            True if str(v).lower() in ("true", "1", "yes", "y")
            else False if str(v).lower() in ("false", "0", "no", "n")
            else bool(v)
        ) if v is not None else None,
        "iso_datetime": lambda v: _parse_datetime_to_iso(v),
        "trim": lambda v: str(v).strip() if v is not None else None,
        "json_parse": lambda v: (
            __import__("json").loads(v) if isinstance(v, str) else v
        ) if v is not None else None,
        "split_comma": lambda v: (
            [s.strip() for s in str(v).split(",") if s.strip()]
            if v is not None else None
        ),
        "to_list": lambda v: ([v] if not isinstance(v, list) else v) if v is not None else None,
    }

    def __init__(self, mappings: list[FieldMapping] | None = None) -> None:
        """初始化 Schema 映射器.

        Args:
            mappings: 字段映射规则列表
        """
        self._mappings: dict[str, FieldMapping] = {}
        if mappings:
            for m in mappings:
                self._mappings[m.target_field] = m

    def add_mapping(self, mapping: FieldMapping) -> None:
        """添加字段映射规则."""
        self._mappings[mapping.target_field] = mapping

    def map(self, source_record: dict[str, Any]) -> dict[str, Any]:
        """将源记录映射到目标格式.

        Args:
            source_record: 源数据记录

        Returns:
            映射后的目标记录

        Raises:
            ValueError: 必须字段缺失且无默认值
        """
        result: dict[str, Any] = {}
        for target_field, mapping in self._mappings.items():
            value = source_record.get(mapping.source_field)
            if value is None:
                if mapping.default_value is not None:
                    value = mapping.default_value
                elif mapping.required:
                    raise ValueError(
                        f"必须字段缺失: {mapping.source_field}"
                    )
            if value is not None and mapping.transform:
                transform_fn = self._TRANSFORMS.get(mapping.transform)
                if transform_fn:
                    value = transform_fn(value)
            result[target_field] = value
        # 保留未映射的源字段 (以 _raw_ 前缀)
        for k, v in source_record.items():
            if k not in {m.source_field for m in self._mappings.values()}:
                result[f"_raw_{k}"] = v
        return result

    def map_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """批量映射记录."""
        return [self.map(r) for r in records]

    @classmethod
    def register_transform(cls, name: str, fn: Callable[[Any], Any]) -> None:
        """注册自定义转换函数.

        Args:
            name: 转换函数名
            fn: 转换函数 (接收任意值，返回转换后的值)
        """
        cls._TRANSFORMS[name] = fn

    @property
    def mappings(self) -> dict[str, FieldMapping]:
        """所有映射规则."""
        return dict(self._mappings)


def _parse_datetime_to_iso(value: Any) -> str | None:
    """尝试将各种日期时间格式转为 ISO 8601."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
    s = str(value).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
    ):
        try:
            import datetime as dt
            parsed = dt.datetime.strptime(s, fmt)
            return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            continue
    return s


# ============================================================
# 同步层数据模型
# ============================================================


class SyncCheckpoint(BaseModel):
    """同步检查点 (借鉴 SeaTunnel 检查点 + Airbyte STATE).

    记录增量同步的进度状态，支持故障恢复。

    Attributes:
        adapter_id: 适配器 ID
        stream_name: 流名称
        sync_mode: 同步模式
        cursor_value: 游标值 (如最后一条记录的 updated_at)
        offset: 偏移量 (如 binlog position, Kafka offset)
        records_read: 已读取记录数
        records_written: 已写入记录数
        last_sync_time: 最后同步时间戳
        error: 最后一次错误信息 (若有)
    """

    adapter_id: str = Field(..., description="适配器 ID")
    stream_name: str = Field(default="", description="流名称")
    sync_mode: SyncMode = Field(default=SyncMode.FULL_REFRESH, description="同步模式")
    cursor_value: str = Field(default="", description="游标值")
    offset: str = Field(default="", description="偏移量")
    records_read: int = Field(default=0, ge=0, description="已读记录数")
    records_written: int = Field(default=0, ge=0, description="已写记录数")
    last_sync_time: float = Field(default=0.0, description="最后同步时间戳")
    error: str = Field(default="", description="最后错误信息")

    def is_fresh(self, max_age: float = 3600.0) -> bool:
        """检查点是否仍在有效期内."""
        if self.last_sync_time == 0.0:
            return False
        return (time.time() - self.last_sync_time) < max_age


class ReadResult(BaseModel):
    """读取结果 (借鉴 Airbyte AirbyteRecordMessage + SeaTunnel SeaTunnelRow).

    封装适配器读取操作返回的数据。

    Attributes:
        records: 读取的数据记录列表
        checkpoint: 当前检查点 (用于增量同步恢复)
        has_more: 是否还有更多数据可读
        next_cursor: 下一页游标 (分页读取)
        schema: 数据 Schema (若 discover 阶段获取)
        metadata: 扩展元数据
    """

    records: list[dict[str, Any]] = Field(default_factory=list, description="数据记录")
    checkpoint: SyncCheckpoint | None = Field(default=None, description="检查点")
    has_more: bool = Field(default=False, description="是否还有更多")
    next_cursor: str = Field(default="", description="下一页游标")
    schema: DataSourceSchema | None = Field(default=None, description="数据 Schema")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")


class DiscoverResult(BaseModel):
    """发现结果 (借鉴 Airbyte AirbyteCatalog).

    discover 阶段返回的数据源 Schema 信息。

    Attributes:
        streams: 发现的流列表 (每个流有独立 Schema)
        adapter_id: 适配器 ID
        discovered_at: 发现时间戳
    """

    streams: list[DataSourceSchema] = Field(
        default_factory=list, description="流列表"
    )
    adapter_id: str = Field(default="", description="适配器 ID")
    discovered_at: float = Field(default_factory=time.time, description="发现时间戳")


class AdapterSpec(BaseModel):
    """适配器规范 (借鉴 Airbyte ConnectorSpecification).

    spec 阶段返回的适配器配置规范，描述适配器需要哪些配置参数。

    Attributes:
        adapter_type: 适配器类型
        capabilities: 能力集合 (AdapterCapability Flag)
        config_schema: 配置参数的 JSON Schema
        default_sync_mode: 默认同步模式
        supported_sync_modes: 支持的同步模式列表
        version: 适配器版本
        documentation_url: 文档 URL
        changelog: 变更日志
    """

    adapter_type: DataSourceType = Field(..., description="适配器类型")
    capabilities: int = Field(default=0, description="能力标志位")
    config_schema: dict[str, Any] = Field(
        default_factory=dict, description="配置参数 JSON Schema"
    )
    default_sync_mode: SyncMode = Field(
        default=SyncMode.FULL_REFRESH, description="默认同步模式"
    )
    supported_sync_modes: list[SyncMode] = Field(
        default_factory=lambda: [SyncMode.FULL_REFRESH],
        description="支持的同步模式",
    )
    version: str = Field(default="1.0.0", description="适配器版本")
    documentation_url: str = Field(default="", description="文档 URL")
    changelog: dict[str, str] = Field(default_factory=dict, description="变更日志")

    def has_capability(self, cap: AdapterCapability) -> bool:
        """检查是否具备指定能力."""
        return bool(self.capability_flags() & cap)

    def capability_flags(self) -> AdapterCapability:
        """获取能力标志枚举."""
        return AdapterCapability(self.capabilities)


class Recoverer(ABC):
    """恢复器接口 (借鉴 Limerence Recoverer 链).

    针对特定已知失败形态的窄幅恢复策略。
    恢复器按优先级排列，第一个返回恢复动作的胜出。
    """

    @abstractmethod
    def can_recover(self, error: Exception, phase: LifecyclePhase) -> bool:
        """判断此恢复器是否能处理该错误.

        Args:
            error: 发生的错误
            phase: 生命周期阶段

        Returns:
            是否可以恢复
        """
        ...

    @abstractmethod
    def recommend_action(self, error: Exception, phase: LifecyclePhase) -> RecoveryAction:
        """推荐恢复动作.

        Args:
            error: 发生的错误
            phase: 生命周期阶段

        Returns:
            恢复动作
        """
        ...


class DefaultRecoverer(Recoverer):
    """默认恢复器 (借鉴 Limerence 默认恢复策略).

    提供基于错误类型的通用恢复建议:
    - 连接错误 → RECONNECT
    - 超时错误 → RETRY
    - 数据格式错误 → SKIP
    - 认证错误 → ABORT
    - 其他 → RETRY (有限次)
    """

    _CONNECTION_ERRORS = (
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        TimeoutError,
        OSError,
    )

    def can_recover(self, error: Exception, phase: LifecyclePhase) -> bool:
        """连接/超时/格式错误可恢复，认证错误不可恢复."""
        if isinstance(error, (PermissionError, AuthenticationError)):
            return False
        return True

    def recommend_action(self, error: Exception, phase: LifecyclePhase) -> RecoveryAction:
        """根据错误类型推荐恢复动作."""
        if isinstance(error, self._CONNECTION_ERRORS):
            return RecoveryAction.RECONNECT
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return RecoveryAction.SKIP
        if isinstance(error, (PermissionError, AuthenticationError)):
            return RecoveryAction.ABORT
        return RecoveryAction.RETRY


# ============================================================
# 异常定义
# ============================================================


class AdapterError(Exception):
    """适配器基础异常."""

    def __init__(self, message: str, *, adapter_id: str = "", phase: LifecyclePhase | None = None) -> None:
        super().__init__(message)
        self.adapter_id = adapter_id
        self.phase = phase


class AuthenticationError(AdapterError):
    """认证失败异常."""


class SchemaDiscoveryError(AdapterError):
    """Schema 发现失败异常."""


class SyncError(AdapterError):
    """同步失败异常."""

    def __init__(self, message: str, *, checkpoint: SyncCheckpoint | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.checkpoint = checkpoint


class RecoveryExhaustedError(AdapterError):
    """恢复策略耗尽异常 (借鉴 Limerence RepairExhausted).

    所有 Recoverer 都无法处理该错误，或重试次数已达上限。
    """

    def __init__(self, message: str, *, original_error: Exception | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.original_error = original_error


# ============================================================
# 数据适配器抽象基类
# ============================================================


class DataAdapterBase(KnowledgeConnector):
    """数据适配器抽象基类.

    融合 Airbyte spec/check/discover/read 协议 + Limerence 共享生命周期 + MCP 能力协商.

    在 KnowledgeConnector 基础上扩展:
    - spec(): 声明配置规范和能力 (Airbyte spec 命令)
    - check(): 验证连通性和认证 (Airbyte check 命令)
    - discover(): 发现数据源 Schema (Airbyte discover 命令)
    - read(): 读取数据，支持全量/增量/CDC (Airbyte read 命令)
    - transform(): 字段映射和类型转换
    - validate(): 数据质量校验

    内置恢复机制 (Limerence Recoverer 链):
    - 每个操作阶段可注册 Recoverer
    - 失败时按优先级遍历 Recoverer 链
    - 第一个声明确恢复动作的 Recoverer 胜出

    子类需实现:
    - _do_spec() → AdapterSpec
    - _do_check() → bool
    - _do_discover() → DiscoverResult
    - _do_read() → ReadResult (替代 search/fetch/list_resources)

    或选择实现简化的:
    - search() / fetch() / list_resources() (兼容 KnowledgeConnector 接口)

    Attributes:
        _schema_mapper: Schema 映射器
        _sync_state: 同步状态 (检查点)
        _recoverers: 恢复器链
        _lifecycle_phase: 当前生命周期阶段
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        schema_mapper: SchemaMapper | None = None,
        sync_mode: SyncMode = SyncMode.FULL_REFRESH,
        recoverers: list[Recoverer] | None = None,
    ) -> None:
        """初始化数据适配器.

        Args:
            config: 连接器配置
            schema_mapper: Schema 映射器 (可选，默认空映射)
            sync_mode: 同步模式 (默认全量刷新)
            recoverers: 恢复器链 (可选，默认 [DefaultRecoverer()])
        """
        super().__init__(config)
        self._schema_mapper: SchemaMapper = schema_mapper or SchemaMapper()
        self._sync_mode: SyncMode = sync_mode
        self._sync_state: SyncCheckpoint | None = None
        self._recoverers: list[Recoverer] = recoverers or [DefaultRecoverer()]
        self._lifecycle_phase: LifecyclePhase = LifecyclePhase.SPEC
        self._spec_cache: AdapterSpec | None = None
        self._schema_cache: DiscoverResult | None = None

    # ---- 生命周期阶段 (Airbyte 协议) ----

    def spec(self) -> AdapterSpec:
        """spec 阶段: 声明配置规范和能力.

        Returns:
            适配器规范
        """
        if self._spec_cache is not None:
            return self._spec_cache
        self._lifecycle_phase = LifecyclePhase.SPEC
        try:
            spec = self._do_spec()
            self._spec_cache = spec
            return spec
        except Exception as e:
            self._handle_error(e, LifecyclePhase.SPEC)
            raise

    def check(self) -> bool:
        """check 阶段: 验证连通性和认证.

        Returns:
            连通性是否正常
        """
        self._lifecycle_phase = LifecyclePhase.CHECK
        try:
            result = self._do_check()
            if result:
                logger.info("适配器 %s 连通性检查通过", self.config.id)
            else:
                logger.warning("适配器 %s 连通性检查失败", self.config.id)
            return result
        except Exception as e:
            self._handle_error(e, LifecyclePhase.CHECK)
            return False

    def discover(self, *, force_refresh: bool = False) -> DiscoverResult:
        """discover 阶段: 发现数据源 Schema.

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            发现结果
        """
        if self._schema_cache is not None and not force_refresh:
            return self._schema_cache
        self._lifecycle_phase = LifecyclePhase.DISCOVER
        try:
            result = self._do_discover()
            self._schema_cache = result
            logger.info(
                "适配器 %s 发现 %d 个流",
                self.config.id,
                len(result.streams),
            )
            return result
        except Exception as e:
            self._handle_error(e, LifecyclePhase.DISCOVER)
            raise

    def read(
        self,
        *,
        stream_name: str = "",
        sync_mode: SyncMode | None = None,
        checkpoint: SyncCheckpoint | None = None,
        limit: int = 0,
    ) -> ReadResult:
        """read 阶段: 读取数据.

        支持三种同步模式:
        - FULL_REFRESH: 从头读取全部数据
        - INCREMENTAL: 从检查点继续读取变更
        - CDC: 流式读取变更日志

        Args:
            stream_name: 流名称 (空则读取默认流)
            sync_mode: 同步模式 (None 则使用适配器默认模式)
            checkpoint: 检查点 (增量同步恢复点)
            limit: 最大读取记录数 (0=无限制)

        Returns:
            读取结果
        """
        self._lifecycle_phase = LifecyclePhase.READ
        mode = sync_mode or self._sync_mode
        try:
            result = self._do_read(
                stream_name=stream_name,
                sync_mode=mode,
                checkpoint=checkpoint,
                limit=limit,
            )
            # 更新同步状态
            if result.checkpoint:
                self._sync_state = result.checkpoint
            logger.info(
                "适配器 %s 读取 %d 条记录 (模式=%s, 流=%s)",
                self.config.id,
                len(result.records),
                mode.value,
                stream_name or "default",
            )
            return result
        except Exception as e:
            self._handle_error(e, LifecyclePhase.READ)
            raise

    def transform(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """transform 阶段: 字段映射和类型转换.

        Args:
            records: 原始记录列表

        Returns:
            映射后的记录列表
        """
        self._lifecycle_phase = LifecyclePhase.TRANSFORM
        try:
            return self._schema_mapper.map_batch(records)
        except Exception as e:
            self._handle_error(e, LifecyclePhase.TRANSFORM)
            raise

    def validate(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
        """validate 阶段: 数据质量校验.

        Args:
            records: 待校验记录列表

        Returns:
            (有效记录列表, 无效记录及原因列表)
        """
        self._lifecycle_phase = LifecyclePhase.VALIDATE
        valid: list[dict[str, Any]] = []
        invalid: list[tuple[dict[str, Any], str]] = []
        for record in records:
            error = self._validate_record(record)
            if error:
                invalid.append((record, error))
            else:
                valid.append(record)
        if invalid:
            logger.warning(
                "适配器 %s 校验: %d 有效, %d 无效",
                self.config.id,
                len(valid),
                len(invalid),
            )
        return valid, invalid

    # ---- 完整同步流程 ----

    def sync(
        self,
        *,
        stream_name: str = "",
        sync_mode: SyncMode | None = None,
        checkpoint: SyncCheckpoint | None = None,
        limit: int = 0,
    ) -> ReadResult:
        """执行完整同步流程: discover → read → transform → validate.

        Args:
            stream_name: 流名称
            sync_mode: 同步模式
            checkpoint: 检查点
            limit: 最大读取数

        Returns:
            处理后的读取结果 (已映射和校验)
        """
        # 1. discover (如果未缓存)
        if self._schema_cache is None:
            try:
                self.discover()
            except Exception as e:
                logger.warning("discover 失败，跳过 Schema 发现: %s", e)

        # 2. read
        result = self.read(
            stream_name=stream_name,
            sync_mode=sync_mode,
            checkpoint=checkpoint,
            limit=limit,
        )

        # 3. transform
        if result.records:
            result.records = self.transform(result.records)

        # 4. validate
        if result.records:
            valid, invalid = self.validate(result.records)
            result.records = valid
            if invalid:
                result.metadata["validation_errors"] = len(invalid)
                result.metadata["invalid_samples"] = invalid[:5]

        return result

    # ---- 恢复机制 (Limerence Recoverer 链) ----

    def _handle_error(self, error: Exception, phase: LifecyclePhase) -> None:
        """错误处理: 遍历恢复器链寻找恢复策略.

        借鉴 Limerence: 按序遍历 Recoverer，第一个 can_recover=True 的胜出。
        """
        for recoverer in self._recoverers:
            if recoverer.can_recover(error, phase):
                action = recoverer.recommend_action(error, phase)
                logger.warning(
                    "适配器 %s 在 %s 阶段出错: %s → 恢复动作: %s",
                    self.config.id,
                    phase.value,
                    str(error),
                    action.value,
                )
                if action == RecoveryAction.ABORT:
                    raise RecoveryExhaustedError(
                        f"恢复器建议终止: {str(error)}",
                        original_error=error,
                        adapter_id=self.config.id,
                        phase=phase,
                    ) from error
                return
        # 无恢复器可处理
        raise RecoveryExhaustedError(
            f"无恢复器可处理错误: {str(error)}",
            original_error=error,
            adapter_id=self.config.id,
            phase=phase,
        ) from error

    def add_recoverer(self, recoverer: Recoverer, *, prepend: bool = False) -> None:
        """添加恢复器到链中.

        Args:
            recoverer: 恢复器实例
            prepend: 是否插入到链首 (高优先级)
        """
        if prepend:
            self._recoverers.insert(0, recoverer)
        else:
            self._recoverers.append(recoverer)

    # ---- 同步状态管理 ----

    @property
    def sync_state(self) -> SyncCheckpoint | None:
        """当前同步检查点."""
        return self._sync_state

    @property
    def sync_mode(self) -> SyncMode:
        """当前同步模式."""
        return self._sync_mode

    @sync_mode.setter
    def sync_mode(self, mode: SyncMode) -> None:
        self._sync_mode = mode

    @property
    def schema_mapper(self) -> SchemaMapper:
        """Schema 映射器."""
        return self._schema_mapper

    @schema_mapper.setter
    def schema_mapper(self, mapper: SchemaMapper) -> None:
        self._schema_mapper = mapper

    @property
    def lifecycle_phase(self) -> LifecyclePhase:
        """当前生命周期阶段."""
        return self._lifecycle_phase

    # ---- 子类必须实现的抽象方法 ----

    @abstractmethod
    def _do_spec(self) -> AdapterSpec:
        """子类实现: 返回适配器规范."""
        ...

    @abstractmethod
    def _do_check(self) -> bool:
        """子类实现: 验证连通性."""
        ...

    @abstractmethod
    def _do_discover(self) -> DiscoverResult:
        """子类实现: 发现 Schema."""
        ...

    @abstractmethod
    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        """子类实现: 读取数据."""
        ...

    def _validate_record(self, record: dict[str, Any]) -> str:
        """子类可重写: 校验单条记录. 返回错误信息，空字符串表示有效."""
        return ""

    # ---- KnowledgeConnector 接口兼容 ----
    # 如果子类不实现 search/fetch/list_resources，则用 read() 作为后备

    def search(self, query: str, **kwargs: Any) -> ConnectorResponse:
        """搜索知识 (兼容 KnowledgeConnector 接口).

        默认实现: 将 query 作为过滤条件调用 read()。
        子类可重写以提供更高效的搜索实现。
        """
        try:
            result = self.read(sync_mode=SyncMode.FULL_REFRESH, limit=kwargs.get("limit", 50))
            # 简单文本过滤
            filtered = [
                r for r in result.records
                if any(query.lower() in str(v).lower() for v in r.values())
            ] if query else result.records
            return ConnectorResponse(
                success=True,
                data=filtered,
                source=self.config.id,
                latency_ms=0.0,
            )
        except Exception as e:
            return ConnectorResponse(
                success=False,
                error=str(e),
                source=self.config.id,
            )

    def fetch(self, resource_id: str) -> ConnectorResponse:
        """获取指定资源 (兼容 KnowledgeConnector 接口).

        默认实现: 读取全部数据后按 ID 过滤。
        子类应重写以提供精确获取。
        """
        try:
            result = self.read(sync_mode=SyncMode.FULL_REFRESH, limit=0)
            for record in result.records:
                # 检查常见 ID 字段
                for id_field in ("id", "entity_id", "resource_id", "doi", "cas"):
                    if record.get(id_field) == resource_id:
                        return ConnectorResponse(
                            success=True,
                            data=record,
                            source=self.config.id,
                        )
            return ConnectorResponse(
                success=False,
                error=f"资源未找到: {resource_id}",
                source=self.config.id,
            )
        except Exception as e:
            return ConnectorResponse(
                success=False,
                error=str(e),
                source=self.config.id,
            )

    def list_resources(self, **kwargs: Any) -> ConnectorResponse:
        """列出可用资源 (兼容 KnowledgeConnector 接口)."""
        try:
            result = self.read(
                sync_mode=SyncMode.FULL_REFRESH,
                limit=kwargs.get("limit", 100),
            )
            return ConnectorResponse(
                success=True,
                data=result.records,
                source=self.config.id,
                latency_ms=0.0,
            )
        except Exception as e:
            return ConnectorResponse(
                success=False,
                error=str(e),
                source=self.config.id,
            )

    # ---- 内部辅助方法 ----

    def _make_checkpoint(
        self,
        stream_name: str,
        records_read: int,
        cursor_value: str = "",
        offset: str = "",
    ) -> SyncCheckpoint:
        """创建检查点."""
        return SyncCheckpoint(
            adapter_id=self.config.id,
            stream_name=stream_name,
            sync_mode=self._sync_mode,
            cursor_value=cursor_value,
            offset=offset,
            records_read=records_read,
            records_written=0,
            last_sync_time=time.time(),
        )

    def get_stats(self) -> dict[str, Any]:
        """获取适配器统计信息."""
        base_stats = {
            "adapter_id": self.config.id,
            "adapter_type": self.__class__.__name__,
            "tier": self.config.tier.value,
            "sync_mode": self._sync_mode.value,
            "lifecycle_phase": self._lifecycle_phase.value,
            "is_connected": self._is_connected,
            "health_status": self._health.status.value,
            "recoverers": len(self._recoverers),
            "schema_mappings": len(self._schema_mapper.mappings),
        }
        if self._sync_state:
            base_stats["sync_state"] = {
                "records_read": self._sync_state.records_read,
                "cursor_value": self._sync_state.cursor_value,
                "last_sync_time": self._sync_state.last_sync_time,
            }
        return base_stats


# ============================================================
# 增强注册中心
# ============================================================


class DataAdapterRegistry:
    """数据适配器注册中心.

    融合 Airbyte Connection Registry + SeaTunnel SPI + MCP 能力协商.

    在 ConnectorRegistry 基础上扩展:
    - 按 DataSourceType 分类查询
    - 按 AdapterCapability 能力查询
    - 按 SyncMode 同步模式查询
    - Schema 发现缓存
    - 同步状态追踪
    - 批量健康检查

    Attributes:
        _adapters: 适配器字典 {id: DataAdapterBase}
        _schemas: Schema 缓存 {adapter_id: DiscoverResult}
        _checkpoints: 检查点缓存 {adapter_id: SyncCheckpoint}
        _lock: 线程安全锁
    """

    def __init__(self) -> None:
        """初始化注册中心."""
        self._adapters: dict[str, DataAdapterBase] = {}
        self._schemas: dict[str, DiscoverResult] = {}
        self._checkpoints: dict[str, SyncCheckpoint] = {}
        self._lock = threading.RLock()

    def register(self, adapter: DataAdapterBase) -> str:
        """注册适配器.

        Args:
            adapter: 数据适配器实例

        Returns:
            适配器 ID

        Raises:
            ValueError: 适配器 ID 已存在
        """
        with self._lock:
            adapter_id = adapter.config.id
            if adapter_id in self._adapters:
                raise ValueError(f"适配器已存在: {adapter_id}")
            self._adapters[adapter_id] = adapter
            logger.info("注册适配器: %s (%s)", adapter_id, adapter.__class__.__name__)
            return adapter_id

    def unregister(self, adapter_id: str) -> bool:
        """注销适配器."""
        with self._lock:
            if adapter_id not in self._adapters:
                return False
            del self._adapters[adapter_id]
            self._schemas.pop(adapter_id, None)
            self._checkpoints.pop(adapter_id, None)
            return True

    def get(self, adapter_id: str) -> DataAdapterBase | None:
        """获取适配器."""
        with self._lock:
            return self._adapters.get(adapter_id)

    def list_all(self) -> list[DataAdapterBase]:
        """列出所有适配器."""
        with self._lock:
            return list(self._adapters.values())

    def list_by_tier(self, tier: ConnectorTier) -> list[DataAdapterBase]:
        """按层级列出适配器."""
        with self._lock:
            return [
                a for a in self._adapters.values()
                if a.config.tier == tier
            ]

    def list_by_type(self, source_type: DataSourceType) -> list[DataAdapterBase]:
        """按数据源类型列出适配器."""
        with self._lock:
            result: list[DataAdapterBase] = []
            for adapter in self._adapters.values():
                try:
                    spec = adapter.spec()
                    if spec.adapter_type == source_type:
                        result.append(adapter)
                except Exception:
                    continue
            return result

    def list_by_capability(self, capability: AdapterCapability) -> list[DataAdapterBase]:
        """按能力列出适配器 (借鉴 MCP 能力协商)."""
        with self._lock:
            result: list[DataAdapterBase] = []
            for adapter in self._adapters.values():
                try:
                    spec = adapter.spec()
                    if spec.has_capability(capability):
                        result.append(adapter)
                except Exception:
                    continue
            return result

    def list_by_sync_mode(self, sync_mode: SyncMode) -> list[DataAdapterBase]:
        """按支持的同步模式列出适配器."""
        with self._lock:
            result: list[DataAdapterBase] = []
            for adapter in self._adapters.values():
                try:
                    spec = adapter.spec()
                    if sync_mode in spec.supported_sync_modes:
                        result.append(adapter)
                except Exception:
                    continue
            return result

    def discover_all(self, *, force_refresh: bool = False) -> dict[str, DiscoverResult]:
        """对所有适配器执行 Schema 发现."""
        with self._lock:
            results: dict[str, DiscoverResult] = {}
            for adapter_id, adapter in self._adapters.items():
                if not force_refresh and adapter_id in self._schemas:
                    results[adapter_id] = self._schemas[adapter_id]
                    continue
                try:
                    result = adapter.discover(force_refresh=force_refresh)
                    self._schemas[adapter_id] = result
                    results[adapter_id] = result
                except Exception as e:
                    logger.error("适配器 %s Schema 发现失败: %s", adapter_id, e)
            return results

    def check_all(self) -> dict[str, bool]:
        """对所有适配器执行连通性检查."""
        with self._lock:
            results: dict[str, bool] = {}
            for adapter_id, adapter in self._adapters.items():
                try:
                    results[adapter_id] = adapter.check()
                except Exception as e:
                    logger.error("适配器 %s 检查失败: %s", adapter_id, e)
                    results[adapter_id] = False
            return results

    def search_all(
        self,
        query: str,
        *,
        tiers: list[ConnectorTier] | None = None,
        limit_per_adapter: int = 20,
    ) -> dict[str, ConnectorResponse]:
        """在所有适配器上搜索 (借鉴 Airbyte 全局搜索).

        Args:
            query: 搜索查询
            tiers: 限定层级 (None=所有)
            limit_per_adapter: 每个适配器最大结果数

        Returns:
            {adapter_id: ConnectorResponse}
        """
        with self._lock:
            results: dict[str, ConnectorResponse] = {}
            for adapter_id, adapter in self._adapters.items():
                if tiers and adapter.config.tier not in tiers:
                    continue
                if not adapter.is_connected:
                    continue
                try:
                    response = adapter.search(query, limit=limit_per_adapter)
                    results[adapter_id] = response
                except Exception as e:
                    results[adapter_id] = ConnectorResponse(
                        success=False,
                        error=str(e),
                        source=adapter_id,
                    )
            return results

    def save_checkpoint(self, adapter_id: str, checkpoint: SyncCheckpoint) -> None:
        """保存同步检查点."""
        with self._lock:
            self._checkpoints[adapter_id] = checkpoint

    def get_checkpoint(self, adapter_id: str) -> SyncCheckpoint | None:
        """获取同步检查点."""
        with self._lock:
            return self._checkpoints.get(adapter_id)

    def get_all_checkpoints(self) -> dict[str, SyncCheckpoint]:
        """获取所有检查点."""
        with self._lock:
            return dict(self._checkpoints)

    def get_stats(self) -> dict[str, Any]:
        """获取注册中心统计信息."""
        with self._lock:
            by_tier: dict[str, int] = defaultdict(int)
            by_status: dict[str, int] = defaultdict(int)
            for adapter in self._adapters.values():
                by_tier[adapter.config.tier.value] += 1
                by_status[adapter._health.status.value] += 1
            return {
                "total_adapters": len(self._adapters),
                "by_tier": dict(by_tier),
                "by_status": dict(by_status),
                "cached_schemas": len(self._schemas),
                "cached_checkpoints": len(self._checkpoints),
            }


# ============================================================
# 同步协调器
# ============================================================


class SyncCoordinator:
    """同步协调器.

    融合 SeaTunnel 协调器 + Airbyte 同步编排.

    功能:
    1. 编排多适配器并行同步
    2. 管理检查点和故障恢复
    3. 全量/增量/CDC 模式切换
    4. 同步进度追踪和报告
    5. 限流和资源调度

    借鉴 SeaTunnel:
    - SourceSplitEnumerator: 将数据源分为可并行处理的 Split
    - 协调与执行分离: Coordinator 分配任务, Adapter 执行读取
    - 检查点驱动的状态恢复

    Attributes:
        _registry: 适配器注册中心
        _checkpoints: 检查点存储
        _sync_history: 同步历史记录
        _lock: 线程安全锁
    """

    def __init__(self, registry: DataAdapterRegistry) -> None:
        """初始化同步协调器.

        Args:
            registry: 适配器注册中心
        """
        self._registry = registry
        self._sync_history: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def sync_adapter(
        self,
        adapter_id: str,
        *,
        stream_name: str = "",
        sync_mode: SyncMode | None = None,
        limit: int = 0,
    ) -> ReadResult:
        """同步单个适配器.

        Args:
            adapter_id: 适配器 ID
            stream_name: 流名称
            sync_mode: 同步模式 (None=使用适配器默认)
            limit: 最大读取数

        Returns:
            读取结果

        Raises:
            ValueError: 适配器不存在
        """
        adapter = self._registry.get(adapter_id)
        if not adapter:
            raise ValueError(f"适配器不存在: {adapter_id}")

        # 获取检查点 (增量同步恢复)
        checkpoint = self._registry.get_checkpoint(adapter_id)

        start_time = time.time()
        try:
            result = adapter.sync(
                stream_name=stream_name,
                sync_mode=sync_mode,
                checkpoint=checkpoint,
                limit=limit,
            )
            # 保存检查点
            if result.checkpoint:
                self._registry.save_checkpoint(adapter_id, result.checkpoint)

            elapsed = time.time() - start_time
            self._record_sync(
                adapter_id=adapter_id,
                stream_name=stream_name,
                sync_mode=sync_mode or adapter.sync_mode,
                records=len(result.records),
                elapsed=elapsed,
                success=True,
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            self._record_sync(
                adapter_id=adapter_id,
                stream_name=stream_name,
                sync_mode=sync_mode or adapter.sync_mode,
                records=0,
                elapsed=elapsed,
                success=False,
                error=str(e),
            )
            raise

    def sync_all(
        self,
        *,
        tiers: list[ConnectorTier] | None = None,
        sync_mode: SyncMode | None = None,
        limit_per_adapter: int = 0,
    ) -> dict[str, ReadResult | Exception]:
        """同步所有适配器 (或指定层级).

        Args:
            tiers: 限定层级 (None=所有)
            sync_mode: 同步模式
            limit_per_adapter: 每个适配器最大读取数

        Returns:
            {adapter_id: ReadResult | Exception}
        """
        adapters = self._registry.list_all()
        if tiers:
            adapters = [a for a in adapters if a.config.tier in tiers]

        results: dict[str, ReadResult | Exception] = {}
        for adapter in adapters:
            try:
                result = self.sync_adapter(
                    adapter.config.id,
                    sync_mode=sync_mode,
                    limit=limit_per_adapter,
                )
                results[adapter.config.id] = result
            except Exception as e:
                results[adapter.config.id] = e
        return results

    def sync_tier(
        self,
        tier: ConnectorTier,
        *,
        sync_mode: SyncMode | None = None,
        limit_per_adapter: int = 0,
    ) -> dict[str, ReadResult | Exception]:
        """同步指定层级的所有适配器."""
        return self.sync_all(
            tiers=[tier],
            sync_mode=sync_mode,
            limit_per_adapter=limit_per_adapter,
        )

    def get_sync_history(self, *, adapter_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """获取同步历史.

        Args:
            adapter_id: 限定适配器 ID (空=所有)
            limit: 最大返回条数

        Returns:
            同步历史记录列表
        """
        with self._lock:
            history = self._sync_history
            if adapter_id:
                history = [h for h in history if h.get("adapter_id") == adapter_id]
            return list(reversed(history))[:limit]

    def _record_sync(
        self,
        *,
        adapter_id: str,
        stream_name: str,
        sync_mode: SyncMode,
        records: int,
        elapsed: float,
        success: bool,
        error: str = "",
    ) -> None:
        """记录同步历史."""
        with self._lock:
            self._sync_history.append({
                "adapter_id": adapter_id,
                "stream_name": stream_name,
                "sync_mode": sync_mode.value,
                "records": records,
                "elapsed_ms": elapsed * 1000,
                "success": success,
                "error": error,
                "timestamp": time.time(),
            })
            # 限制历史记录数量
            if len(self._sync_history) > 1000:
                self._sync_history = self._sync_history[-500:]

    def get_progress_report(self) -> dict[str, Any]:
        """获取同步进度报告."""
        with self._lock:
            total = len(self._sync_history)
            successes = sum(1 for h in self._sync_history if h["success"])
            failures = total - successes
            total_records = sum(h["records"] for h in self._sync_history)
            total_time = sum(h["elapsed_ms"] for h in self._sync_history)
            checkpoints = self._registry.get_all_checkpoints()
            return {
                "total_syncs": total,
                "successful": successes,
                "failed": failures,
                "success_rate": successes / total if total > 0 else 0.0,
                "total_records_synced": total_records,
                "total_sync_time_ms": total_time,
                "active_checkpoints": len(checkpoints),
                "checkpoint_details": {
                    aid: {
                        "cursor_value": cp.cursor_value,
                        "records_read": cp.records_read,
                        "last_sync_time": cp.last_sync_time,
                        "sync_mode": cp.sync_mode.value,
                    }
                    for aid, cp in checkpoints.items()
                },
            }
