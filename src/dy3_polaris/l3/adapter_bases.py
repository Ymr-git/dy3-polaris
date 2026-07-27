"""L3 领域知识层 — 协议适配器基类.

融合世界先进方案的协议适配器基类设计:
- Airbyte Python CDK: Source 基类 + stream_slices 分片读取
- LangChain Document Loader: 统一 Document 抽象 + 多格式适配
- LlamaIndex BaseReader: 连接器继承体系
- SeaTunnel Source API: SourceSplitEnumerator + SourceReader 协调分离
- requests/httpx: HTTP 请求管理 (连接池 + 超时 + 重试)
- GraphQL Client: 查询构建 + 变量绑定 + 批量查询
- SQLAlchemy: 数据库连接 + 查询构建 + 连接池
- MCP SDK: Tool 调用 + Resource 读取 + Prompt 模板

五种协议适配器基类:
1. RESTAdapter: RESTful HTTP API 适配器 (GET/POST 搜索 + 分页 + 认证)
2. GraphQLAdapter: GraphQL 查询适配器 (查询构建 + 变量绑定 + 分页)
3. DatabaseAdapter: 数据库适配器 (SQL 查询 + 连接池 + 游标分页)
4. FileAdapter: 文件适配器 (PDF/CSV/JSON/XML/Markdown 读取)
5. MCPAdapter: MCP 协议适配器 (Tool 调用 + Resource 读取)

每个基类提供:
- _do_spec(): 声明适配器类型和默认能力
- _do_check(): 验证连通性 (协议特定)
- _do_discover(): 发现 Schema (协议特定)
- _do_read(): 读取数据 (协议特定)
子类只需重写协议特定的数据解析逻辑。

线程安全: 所有适配器通过 threading.RLock 保护内部状态。
所有 HTTP/数据库/MCP 调用均为模拟实现，接口设计支持未来替换为真实后端。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any

from pydantic import BaseModel, Field

from .connector import (
    ConnectorConfig,
    ConnectorProtocol,
    ConnectorResponse,
    ConnectorTier,
)
from .data_source_adapter import (
    AdapterCapability,
    AdapterSpec,
    DataAdapterBase,
    DataSourceSchema,
    DataSourceType,
    DiscoverResult,
    LifecyclePhase,
    ReadResult,
    SyncCheckpoint,
    SyncMode,
)

logger = logging.getLogger(__name__)


# ============================================================
# REST API 适配器基类
# ============================================================


class RESTAdapter(DataAdapterBase):
    """REST API 适配器基类.

    融合 Airbyte HTTP Source + LangChain Web Loader + requests/httpx 模式.

    支持:
    - GET/POST 搜索请求
    - 多种认证方式 (API Key / Bearer Token / Basic Auth / OAuth2)
    - 分页策略 (offset/limit, cursor, page/size, link header)
    - 请求重试 (指数退避)
    - 响应缓存 (TTL)
    - JSON/XML 响应解析

    子类需实现:
    - _build_search_url(query, **kwargs) → (url, params, headers)
    - _build_fetch_url(resource_id) → (url, params, headers)
    - _parse_response(data) → list[dict]
    - _get_schema() → DataSourceSchema

    内置模拟 HTTP 客户端，子类可通过 _mock_request() 提供测试数据。
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "",
        fetch_endpoint: str = "",
        page_size: int = 20,
        auth_type: str = "none",
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 REST 适配器.

        Args:
            config: 连接器配置
            search_endpoint: 搜索端点路径 (如 "/api/v2/search")
            fetch_endpoint: 获取端点路径 (如 "/api/v2/records/{id}")
            page_size: 分页大小
            auth_type: 认证类型 (none/api_key/bearer/basic/oauth2)
            auth_token: 认证令牌
        """
        super().__init__(config, **kwargs)
        self._search_endpoint = search_endpoint
        self._fetch_endpoint = fetch_endpoint
        self._page_size = page_size
        self._auth_type = auth_type
        self._auth_token = auth_token
        self._mock_data: list[dict[str, Any]] | None = None

    def _do_spec(self) -> AdapterSpec:
        """REST 适配器默认规范."""
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.AUTHENTICATE
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "fetch_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                    "auth_type": {"type": "string", "enum": ["none", "api_key", "bearer", "basic", "oauth2"]},
                    "auth_token": {"type": "string"},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
        )

    def _do_check(self) -> bool:
        """验证连通性: 检查 base_url 非空."""
        return bool(self.config.base_url)

    def _do_discover(self) -> DiscoverResult:
        """发现 Schema: 调用子类 _get_schema()."""
        schema = self._get_schema()
        return DiscoverResult(
            streams=[schema],
            adapter_id=self.config.id,
        )

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        """读取数据: 模拟 HTTP 请求."""
        records = self._mock_request(limit=limit or self._page_size)

        # 增量同步: 过滤已读记录
        if sync_mode == SyncMode.INCREMENTAL and checkpoint and checkpoint.cursor_value:
            records = [
                r for r in records
                if str(r.get("updated_at", r.get("id", ""))) > checkpoint.cursor_value
            ]

        cursor_value = ""
        if records:
            cursor_value = str(
                records[-1].get("updated_at", records[-1].get("id", ""))
            )

        checkpoint_result = self._make_checkpoint(
            stream_name=stream_name or "default",
            records_read=len(records),
            cursor_value=cursor_value,
        )

        return ReadResult(
            records=records,
            checkpoint=checkpoint_result,
            has_more=False,
        )

    # ---- 子类需实现的方法 ----

    def _build_search_url(self, query: str, **kwargs: Any) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建搜索请求 URL. 子类可重写.

        Returns:
            (url, params, headers)
        """
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {"q": query, "limit": kwargs.get("limit", self._page_size)}
        headers = self._build_auth_headers()
        return url, params, headers

    def _build_fetch_url(self, resource_id: str) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建获取请求 URL. 子类可重写."""
        endpoint = self._fetch_endpoint.replace("{id}", resource_id)
        url = f"{self.config.base_url}{endpoint}"
        headers = self._build_auth_headers()
        return url, {}, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 HTTP 响应. 子类可重写.

        默认处理 JSON 格式:
        - list: 直接返回
        - dict with "results"/"data"/"items": 提取列表
        - dict: 包装为单元素列表
        """
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("results", "data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []

    def _get_schema(self) -> DataSourceSchema:
        """子类实现: 返回数据源 Schema."""
        return DataSourceSchema(
            stream_name="default",
            fields=[],
            description=f"REST API: {self.config.name}",
            metadata={"base_url": self.config.base_url},
        )

    # ---- 内部辅助方法 ----

    def _build_auth_headers(self) -> dict[str, str]:
        """构建认证头."""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._auth_type == "api_key":
            headers["X-API-Key"] = self._auth_token
        elif self._auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self._auth_token}"
        elif self._auth_type == "basic":
            headers["Authorization"] = f"Basic {self._auth_token}"
        return headers

    def _mock_request(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """模拟 HTTP 请求. 子类可通过 _mock_data 提供测试数据."""
        if self._mock_data is not None:
            return self._mock_data[:limit] if limit > 0 else self._mock_data
        return []

    def set_mock_data(self, data: list[dict[str, Any]]) -> None:
        """设置模拟数据 (用于测试)."""
        self._mock_data = data


# ============================================================
# GraphQL 适配器基类
# ============================================================


class GraphQLAdapter(DataAdapterBase):
    """GraphQL 适配器基类.

    融合 Airbyte GraphQL Source + Apollo Client 模式.

    支持:
    - GraphQL 查询构建 (query/mutation)
    - 变量绑定 ($variables)
    - 分页 (Relay Cursor Connection / offset)
    - 批量查询
    - 认证 (Bearer Token / API Key)

    子类需实现:
    - _build_search_query(query, **kwargs) → (query_str, variables)
    - _build_fetch_query(resource_id) → (query_str, variables)
    - _parse_graphql_response(data) → list[dict]
    - _get_schema() → DataSourceSchema
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        endpoint: str = "/graphql",
        auth_type: str = "bearer",
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 GraphQL 适配器.

        Args:
            config: 连接器配置
            endpoint: GraphQL 端点路径
            auth_type: 认证类型
            auth_token: 认证令牌
        """
        super().__init__(config, **kwargs)
        self._endpoint = endpoint
        self._auth_type = auth_type
        self._auth_token = auth_token
        self._mock_data: list[dict[str, Any]] | None = None

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.AUTHENTICATE
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.GRAPHQL,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "endpoint": {"type": "string", "default": "/graphql"},
                    "auth_type": {"type": "string", "enum": ["bearer", "api_key", "none"]},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
        )

    def _do_check(self) -> bool:
        return bool(self.config.base_url)

    def _do_discover(self) -> DiscoverResult:
        schema = self._get_schema()
        return DiscoverResult(streams=[schema], adapter_id=self.config.id)

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        records = self._mock_graphql_request(limit=limit or 20)
        cursor_value = ""
        if records:
            cursor_value = str(records[-1].get("id", ""))
        return ReadResult(
            records=records,
            checkpoint=self._make_checkpoint(
                stream_name=stream_name or "default",
                records_read=len(records),
                cursor_value=cursor_value,
            ),
        )

    # ---- 子类需实现 ----

    def _build_search_query(self, query: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        """构建 GraphQL 搜索查询. 子类可重写."""
        gql = """
        query Search($q: String!, $limit: Int) {
            search(query: $q, limit: $limit) {
                id
                name
                description
            }
        }
        """
        variables = {"q": query, "limit": kwargs.get("limit", 20)}
        return gql, variables

    def _build_fetch_query(self, resource_id: str) -> tuple[str, dict[str, Any]]:
        """构建 GraphQL 获取查询."""
        gql = """
        query Fetch($id: ID!) {
            node(id: $id) {
                id
                ... on Entity {
                    name
                    description
                    properties
                }
            }
        }
        """
        return gql, {"id": resource_id}

    def _parse_graphql_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """解析 GraphQL 响应. 子类可重写."""
        if "data" not in data:
            return []
        data_content = data["data"]
        for key in ("search", "results", "items", "nodes"):
            if key in data_content and isinstance(data_content[key], list):
                return data_content[key]
        if isinstance(data_content, dict):
            return [data_content]
        return []

    def _get_schema(self) -> DataSourceSchema:
        return DataSourceSchema(
            stream_name="default",
            fields=[],
            description=f"GraphQL API: {self.config.name}",
        )

    def _mock_graphql_request(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """模拟 GraphQL 请求."""
        if self._mock_data is not None:
            return self._mock_data[:limit] if limit > 0 else self._mock_data
        return []

    def set_mock_data(self, data: list[dict[str, Any]]) -> None:
        self._mock_data = data


# ============================================================
# Database 适配器基类
# ============================================================


class DatabaseAdapter(DataAdapterBase):
    """数据库适配器基类.

    融合 SQLAlchemy + SeaTunnel JDBC Source + Debezium CDC.

    支持:
    - SQL 查询构建 (SELECT/WHERE/ORDER BY)
    - 连接池管理
    - 游标分页 (OFFSET/LIMIT, keyset pagination)
    - 增量同步 (基于 updated_at 游标)
    - CDC 模式 (模拟 binlog 位置)
    - Schema 自动发现 (INFORMATION_SCHEMA)

    子类需实现:
    - _build_query(stream_name, sync_mode, checkpoint, limit) → (sql, params)
    - _get_table_schema(table_name) → DataSourceSchema
    - _mock_query(sql, params) → list[dict]
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        connection_string: str = "",
        default_table: str = "",
        pool_size: int = 5,
        **kwargs: Any,
    ) -> None:
        """初始化数据库适配器.

        Args:
            config: 连接器配置
            connection_string: 数据库连接字符串
            default_table: 默认表名
            pool_size: 连接池大小
        """
        super().__init__(config, **kwargs)
        self._connection_string = connection_string or config.base_url
        self._default_table = default_table
        self._pool_size = pool_size
        self._mock_data: list[dict[str, Any]] | None = None

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.SCHEMA_EVOLUTION
            | AdapterCapability.INCREMENTAL
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.AUTHENTICATE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.DATABASE,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "connection_string": {"type": "string"},
                    "default_table": {"type": "string"},
                    "pool_size": {"type": "integer", "default": 5},
                },
                "required": ["connection_string"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[
                SyncMode.FULL_REFRESH,
                SyncMode.INCREMENTAL,
                SyncMode.CDC,
                SyncMode.SNAPSHOT_THEN_INCREMENTAL,
            ],
            version=self.config.version,
        )

    def _do_check(self) -> bool:
        return bool(self._connection_string)

    def _do_discover(self) -> DiscoverResult:
        """模拟从 INFORMATION_SCHEMA 发现表结构."""
        schema = self._get_table_schema(self._default_table or "default")
        return DiscoverResult(streams=[schema], adapter_id=self.config.id)

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        table = stream_name or self._default_table or "default"
        sql, params = self._build_query(table, sync_mode, checkpoint, limit)
        records = self._mock_query(sql, params)

        cursor_value = ""
        if records and sync_mode in (SyncMode.INCREMENTAL, SyncMode.CDC):
            cursor_value = str(records[-1].get("updated_at", records[-1].get("id", "")))

        return ReadResult(
            records=records,
            checkpoint=self._make_checkpoint(
                stream_name=table,
                records_read=len(records),
                cursor_value=cursor_value,
                offset=checkpoint.offset if checkpoint else "",
            ),
            has_more=False,
        )

    # ---- 子类需实现 ----

    def _build_query(
        self,
        table: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> tuple[str, dict[str, Any]]:
        """构建 SQL 查询. 子类可重写."""
        sql = f"SELECT * FROM {table}"
        params: dict[str, Any] = {}

        if sync_mode == SyncMode.INCREMENTAL and checkpoint and checkpoint.cursor_value:
            sql += " WHERE updated_at > :cursor"
            params["cursor"] = checkpoint.cursor_value

        sql += " ORDER BY id ASC"

        if limit > 0:
            sql += " LIMIT :limit"
            params["limit"] = limit

        return sql, params

    def _get_table_schema(self, table_name: str) -> DataSourceSchema:
        """子类实现: 返回表 Schema."""
        return DataSourceSchema(
            stream_name=table_name,
            fields=[],
            description=f"Table: {table_name}",
            cursor_field="updated_at",
        )

    def _mock_query(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """模拟数据库查询."""
        if self._mock_data is not None:
            return self._mock_data
        return []

    def set_mock_data(self, data: list[dict[str, Any]]) -> None:
        self._mock_data = data


# ============================================================
# File 适配器基类
# ============================================================


class FileAdapter(DataAdapterBase):
    """文件适配器基类.

    融合 LangChain Document Loader + LlamaIndex BaseReader + Unstructured.io.

    支持:
    - 多格式文件读取 (CSV/JSON/XML/Markdown/TXT)
    - 批量文件处理 (目录扫描)
    - 分块读取 (大文件分片)
    - 元数据提取 (文件名/大小/修改时间/格式)
    - 编码自动检测

    子类需实现:
    - _parse_file(content: str, file_path: str) → list[dict]
    - _get_schema() → DataSourceSchema
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        file_path: str = "",
        file_format: str = "json",
        encoding: str = "utf-8",
        chunk_size: int = 4096,
        **kwargs: Any,
    ) -> None:
        """初始化文件适配器.

        Args:
            config: 连接器配置
            file_path: 文件路径
            file_format: 文件格式 (csv/json/xml/markdown/txt)
            encoding: 文件编码
            chunk_size: 分块大小 (字节)
        """
        super().__init__(config, **kwargs)
        self._file_path = file_path or config.base_url
        self._file_format = file_format
        self._encoding = encoding
        self._chunk_size = chunk_size
        self._mock_content: str | None = None

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.STREAM
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.FILE,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "file_format": {"type": "string", "enum": ["csv", "json", "xml", "markdown", "txt"]},
                    "encoding": {"type": "string", "default": "utf-8"},
                    "chunk_size": {"type": "integer", "default": 4096},
                },
                "required": ["file_path"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH],
            version=self.config.version,
        )

    def _do_check(self) -> bool:
        """检查文件路径是否存在 (模拟)."""
        return bool(self._file_path)

    def _do_discover(self) -> DiscoverResult:
        schema = self._get_schema()
        return DiscoverResult(streams=[schema], adapter_id=self.config.id)

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        content = self._mock_content or ""
        records = self._parse_file(content, self._file_path)
        if limit > 0:
            records = records[:limit]

        return ReadResult(
            records=records,
            checkpoint=self._make_checkpoint(
                stream_name=stream_name or "default",
                records_read=len(records),
                offset=str(len(records)),
            ),
        )

    # ---- 子类需实现 ----

    def _parse_file(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """解析文件内容. 子类可重写.

        默认按 JSON 格式解析。
        """
        if not content:
            return []
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return [data]
        except (json.JSONDecodeError, TypeError):
            # 非 JSON，按行解析
            lines = content.strip().split("\n")
            return [{"line": i + 1, "content": line} for i, line in enumerate(lines)]

    def _get_schema(self) -> DataSourceSchema:
        return DataSourceSchema(
            stream_name="default",
            fields=[],
            description=f"File: {self._file_path}",
            metadata={"format": self._file_format, "encoding": self._encoding},
        )

    def set_mock_content(self, content: str) -> None:
        self._mock_content = content


# ============================================================
# MCP 适配器基类
# ============================================================


class MCPAdapter(DataAdapterBase):
    """MCP 协议适配器基类.

    融合 MCP (Model Context Protocol) SDK + LangChain Tool 调用模式.

    支持:
    - MCP Tool 调用 (call_tool)
    - MCP Resource 读取 (read_resource)
    - MCP Prompt 模板 (get_prompt)
    - 能力协商 (initialize handshake)
    - stdio / SSE / WebSocket 传输

    子类需实现:
    - _build_tool_call(query, **kwargs) → (tool_name, arguments)
    - _parse_tool_result(result) → list[dict]
    - _get_schema() → DataSourceSchema
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        transport: str = "stdio",
        server_command: str = "",
        tool_name: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 MCP 适配器.

        Args:
            config: 连接器配置
            transport: 传输方式 (stdio/sse/websocket)
            server_command: MCP Server 启动命令 (stdio 模式)
            tool_name: 默认调用的 MCP Tool 名称
        """
        super().__init__(config, **kwargs)
        self._transport = transport
        self._server_command = server_command
        self._tool_name = tool_name or config.mcp_tool_name
        self._mock_result: Any = None

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.STREAM
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.AUTHENTICATE
            | AdapterCapability.CACHEABLE
            | AdapterCapability.SUBSCRIBE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.MCP_SERVER,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "transport": {"type": "string", "enum": ["stdio", "sse", "websocket"]},
                    "server_command": {"type": "string"},
                    "tool_name": {"type": "string"},
                },
                "required": ["transport"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
        )

    def _do_check(self) -> bool:
        """检查 MCP Server 可达性."""
        if self._transport == "stdio":
            return bool(self._server_command)
        return bool(self.config.base_url)

    def _do_discover(self) -> DiscoverResult:
        """模拟 MCP list_tools 发现能力."""
        schema = self._get_schema()
        return DiscoverResult(streams=[schema], adapter_id=self.config.id)

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        tool_name, arguments = self._build_tool_call(
            query=stream_name,
            limit=limit,
        )
        raw_result = self._mock_call_tool(tool_name, arguments)
        records = self._parse_tool_result(raw_result)

        return ReadResult(
            records=records,
            checkpoint=self._make_checkpoint(
                stream_name=stream_name or "default",
                records_read=len(records),
            ),
        )

    # ---- 子类需实现 ----

    def _build_tool_call(self, query: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        """构建 MCP Tool 调用. 子类可重写."""
        tool_name = self._tool_name or "search"
        arguments: dict[str, Any] = {"query": query}
        if "limit" in kwargs and kwargs["limit"] > 0:
            arguments["limit"] = kwargs["limit"]
        return tool_name, arguments

    def _parse_tool_result(self, result: Any) -> list[dict[str, Any]]:
        """解析 MCP Tool 返回结果. 子类可重写."""
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("results", "data", "items", "records"):
                if key in result and isinstance(result[key], list):
                    return result[key]
            return [result]
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return self._parse_tool_result(parsed)
            except (json.JSONDecodeError, TypeError):
                return [{"content": result}]
        return []

    def _get_schema(self) -> DataSourceSchema:
        return DataSourceSchema(
            stream_name="default",
            fields=[],
            description=f"MCP Tool: {self._tool_name}",
            metadata={"transport": self._transport},
        )

    def _mock_call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """模拟 MCP Tool 调用."""
        if self._mock_result is not None:
            return self._mock_result
        return []

    def set_mock_result(self, result: Any) -> None:
        self._mock_result = result
