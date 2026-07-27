"""L6 配置管理.

基于 pydantic-settings 实现分层配置，支持环境变量覆盖。
所有配置均可通过 DY3_L6_ 前缀的环境变量设置。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 传输类型
# ============================================================

class TransportType(str, Enum):
    """MCP 传输类型."""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


# ============================================================
# 传输层配置
# ============================================================

class StdioTransportConfig(BaseSettings):
    """stdio 传输配置."""
    model_config = SettingsConfigDict(env_prefix="DY3_L6_STDIO_")

    command: str = Field(default="python", description="启动 MCP Server 的命令")
    args: list[str] = Field(default_factory=list, description="命令行参数")
    env: dict[str, str] | None = Field(default=None, description="环境变量")
    cwd: str | None = Field(default=None, description="工作目录")
    request_timeout: float = Field(default=60.0, ge=1.0, description="请求超时（秒）")


class SSETransportConfig(BaseSettings):
    """SSE 传输配置."""
    model_config = SettingsConfigDict(env_prefix="DY3_L6_SSE_")

    url: str = Field(default="http://localhost:8000/sse", description="SSE endpoint URL")
    heartbeat_interval: float = Field(default=30.0, ge=5.0, description="心跳间隔（秒）")
    request_timeout: float = Field(default=120.0, ge=1.0, description="请求超时（秒）")
    max_retries: int = Field(default=5, ge=0, description="最大重试次数")
    base_delay: float = Field(default=1.0, ge=0.1, description="指数退避基础延迟（秒）")
    max_delay: float = Field(default=30.0, ge=1.0, description="指数退避最大延迟（秒）")


class WebSocketTransportConfig(BaseSettings):
    """WebSocket 传输配置."""
    model_config = SettingsConfigDict(env_prefix="DY3_L6_WS_")

    url: str = Field(default="ws://localhost:8000/ws", description="WebSocket endpoint URL")
    heartbeat_interval: float = Field(default=30.0, ge=5.0, description="心跳间隔（秒）")
    request_timeout: float = Field(default=120.0, ge=1.0, description="请求超时（秒）")
    max_retries: int = Field(default=5, ge=0, description="最大重试次数")
    base_delay: float = Field(default=1.0, ge=0.1, description="指数退避基础延迟（秒）")
    max_delay: float = Field(default=30.0, ge=1.0, description="指数退避最大延迟（秒）")
    ping_interval: float = Field(default=20.0, ge=1.0, description="WebSocket ping 间隔（秒）")
    ping_timeout: float = Field(default=10.0, ge=1.0, description="ping 超时（秒）")


# ============================================================
# MCP 协议配置
# ============================================================

class MCPProtocolConfig(BaseSettings):
    """MCP 协议核心配置."""
    model_config = SettingsConfigDict(env_prefix="DY3_L6_MCP_")

    protocol_version: str = Field(default="2024-11-05", description="MCP 协议版本")
    server_name: str = Field(default="dy3-polaris", description="MCP Server 名称")
    server_version: str = Field(default="0.1.0", description="MCP Server 版本")
    instructions: str = Field(
        default="Dy3+ Polaris 多智能体协同决策系统 MCP Server",
        description="Server 使用说明",
    )
    json_response: bool = Field(default=False, description="是否启用 JSON 响应模式")
    max_tool_timeout: float = Field(default=120.0, ge=1.0, description="工具执行最大超时（秒）")
    progress_reporting: bool = Field(default=True, description="是否启用进度通知")


# ============================================================
# 全局 L6 配置
# ============================================================

class L6Config(BaseSettings):
    """L6 协议基础设施全局配置.

    加载优先级：环境变量 > .env 文件 > 默认值
    环境变量前缀：DY3_L6_
    """

    model_config = SettingsConfigDict(
        env_prefix="DY3_L6_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 全局 ----
    debug: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")

    # ---- MCP 协议 ----
    mcp: MCPProtocolConfig = Field(default_factory=MCPProtocolConfig)

    # ---- 传输层 ----
    default_transport: TransportType = Field(
        default=TransportType.STDIO,
        description="默认传输类型",
    )
    stdio: StdioTransportConfig = Field(default_factory=StdioTransportConfig)
    sse: SSETransportConfig = Field(default_factory=SSETransportConfig)
    websocket: WebSocketTransportConfig = Field(default_factory=WebSocketTransportConfig)

    # ---- 溯源 ----
    provenance_enabled: bool = Field(default=True, description="是否启用溯源")
    provenance_auto_attach: bool = Field(default=True, description="是否自动附加溯源包到 MCP 调用")

    # ---- A2A ----
    a2a_enabled: bool = Field(default=True, description="是否启用 A2A 协议")
    a2a_heartbeat_interval: float = Field(default=30.0, ge=5.0, description="A2A 心跳间隔")

    # ---- 限流 ----
    default_rate_limit: int = Field(default=100, ge=1, description="默认工具限流（次/分钟）")
    rate_limit_window: int = Field(default=60, ge=1, description="限流窗口（秒）")

    # ---- 算力 ----
    compute_degradation_enabled: bool = Field(default=True, description="是否启用自动降级")

    # ---- 广播 ----
    broadcast_enabled: bool = Field(default=True, description="是否启用学情广播")
    broadcast_max_subscribers_per_topic: int = Field(default=100, ge=1, description="每主题最大订阅者数")
    broadcast_event_log_enabled: bool = Field(default=False, description="是否启用事件日志")
    broadcast_event_log_max_size: int = Field(default=1000, ge=1, description="事件日志最大条数")

    # ---- 记忆图谱 ----
    memory_graph_enabled: bool = Field(default=True, description="是否启用记忆图谱")
    memory_graph_decay_factor: float = Field(default=0.95, ge=0.0, le=1.0, description="衰减因子")
    memory_graph_min_strength: float = Field(default=0.01, ge=0.0, le=1.0, description="最小强度阈值")
    memory_graph_spreading_depth: int = Field(default=2, ge=1, description="扩散激活深度")
    memory_graph_spreading_decay: float = Field(default=0.5, ge=0.0, le=1.0, description="扩散激活衰减率")

    # ---- REST API ----
    rest_api_enabled: bool = Field(default=True, description="是否启用 REST API")
    rest_api_host: str = Field(default="0.0.0.0", description="REST API 监听地址")
    rest_api_port: int = Field(default=8000, ge=1, le=65535, description="REST API 监听端口")
    rest_api_cors_origins: list[str] = Field(default_factory=lambda: ["*"], description="CORS 允许的源")

    # ---- Legacy ----
    legacy_mode: bool = Field(default=False, description="是否启用 Legacy 兼容模式")
    legacy_endpoint_prefix: str = Field(default="/api/v1", description="Legacy API 前缀")

    # ---- L0 治理 ----
    governance_enabled: bool = Field(default=True, description="是否启用 L0 治理层")
    governance_default_action: str = Field(default="allow", description="治理默认动作")
    governance_violation_log_max: int = Field(default=2000, ge=100, description="违规日志最大条数")
    governance_event_log_max: int = Field(default=500, ge=50, description="治理事件日志最大条数")
    governance_compliance_threshold: float = Field(default=60.0, ge=0.0, le=100.0, description="合规评分告警阈值")

    def get_governance_config(self) -> dict[str, object]:
        """获取治理层配置字典."""
        return {
            "enabled": self.governance_enabled,
            "default_action": self.governance_default_action,
            "violation_log_max": self.governance_violation_log_max,
            "event_log_max": self.governance_event_log_max,
            "compliance_threshold": self.governance_compliance_threshold,
        }

    def get_rest_config(self) -> dict[str, object]:
        """获取 REST API 配置字典."""
        return {
            "enabled": self.rest_api_enabled,
            "host": self.rest_api_host,
            "port": self.rest_api_port,
            "cors_origins": self.rest_api_cors_origins,
        }

    def get_legacy_config(self) -> dict[str, object]:
        """获取 Legacy 兼容配置字典."""
        return {
            "enabled": self.legacy_mode,
            "endpoint_prefix": self.legacy_endpoint_prefix,
        }

    def get_transport_config(self, transport_type: TransportType | None = None) -> StdioTransportConfig | SSETransportConfig | WebSocketTransportConfig:
        """获取指定传输类型的配置."""
        t = transport_type or self.default_transport
        return {
            TransportType.STDIO: self.stdio,
            TransportType.SSE: self.sse,
            TransportType.WEBSOCKET: self.websocket,
        }[t]


# ============================================================
# 全局配置单例
# ============================================================

_global_config: L6Config | None = None


def get_config() -> L6Config:
    """获取全局 L6 配置单例.

    首次调用时从环境变量/.env 加载，后续调用返回缓存实例。
    """
    global _global_config
    if _global_config is None:
        _global_config = L6Config()
    return _global_config


def reset_config() -> None:
    """重置全局配置（仅用于测试）."""
    global _global_config
    _global_config = None