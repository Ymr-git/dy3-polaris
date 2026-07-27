"""L6 异常体系.

分层异常层级，覆盖传输层、消息层、协议层、业务层。
所有异常均继承自 L6Error，支持结构化错误信息。
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any


# ============================================================
# 基础异常
# ============================================================

class L6Error(Exception):
    """L6 层所有异常的基类.

    Attributes:
        code: 机器可读的错误码 (e.g. "TRANSPORT_TIMEOUT")
        detail: 人类可读的错误详情
        context: 结构化上下文信息，用于日志和调试
    """

    def __init__(
        self,
        code: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.context = context or {}
        parts = [f"[{code}]"]
        if detail:
            parts.append(detail)
        super().__init__(" ".join(parts))

    def to_json_rpc_error(self) -> dict[str, Any]:
        """转换为 JSON-RPC 2.0 错误对象."""
        return {
            "code": self._jsonrpc_code(),
            "message": self.code,
            "data": {"detail": self.detail, **self.context} if self.detail or self.context else None,
        }

    def _jsonrpc_code(self) -> int:
        """映射到 JSON-RPC 标准错误码."""
        code_map: dict[str, int] = {
            "TRANSPORT_TIMEOUT": -32001,
            "TRANSPORT_CLOSED": -32002,
            "TRANSPORT_RECONNECT_EXHAUSTED": -32003,
            "JSONRPC_PARSE": -32700,
            "JSONRPC_INVALID_REQUEST": -32600,
            "JSONRPC_METHOD_NOT_FOUND": -32601,
            "JSONRPC_INVALID_PARAMS": -32602,
            "JSONRPC_INTERNAL": -32603,
        }
        return code_map.get(self.code, -32000)


# ============================================================
# 传输层异常
# ============================================================

class TransportError(L6Error):
    """传输层错误基类."""
    pass


class TransportTimeoutError(TransportError):
    """传输层超时.

    Attributes:
        timeout_seconds: 超时时间（秒）
    """

    def __init__(
        self,
        transport_type: str,
        timeout_seconds: float,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.transport_type = transport_type
        self.timeout_seconds = timeout_seconds
        ctx = {"transport": transport_type, "timeout_s": timeout_seconds}
        ctx.update(context or {})
        super().__init__("TRANSPORT_TIMEOUT", detail or f"{transport_type} timeout after {timeout_seconds}s", ctx)


class TransportClosedError(TransportError):
    """传输连接已关闭."""

    def __init__(
        self,
        transport_type: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"transport": transport_type}
        ctx.update(context or {})
        super().__init__("TRANSPORT_CLOSED", detail or f"{transport_type} connection closed", ctx)


class ReconnectExhaustedError(TransportError):
    """重连次数耗尽."""

    def __init__(
        self,
        transport_type: str,
        attempts: int,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.attempts = attempts
        ctx = {"transport": transport_type, "attempts": attempts}
        ctx.update(context or {})
        super().__init__(
            "TRANSPORT_RECONNECT_EXHAUSTED",
            detail or f"{transport_type} reconnect failed after {attempts} attempts",
            ctx,
        )


# ============================================================
# JSON-RPC 消息层异常
# ============================================================

class JSONRPCError(L6Error):
    """JSON-RPC 消息层错误基类."""
    pass


class ParseError(JSONRPCError):
    """JSON-RPC 解析错误 (-32700)."""

    def __init__(self, detail: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("JSONRPC_PARSE", detail or "Parse error", context)


class InvalidRequestError(JSONRPCError):
    """JSON-RPC 无效请求 (-32600)."""

    def __init__(self, detail: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("JSONRPC_INVALID_REQUEST", detail or "Invalid Request", context)


class MethodNotFoundError(JSONRPCError):
    """JSON-RPC 方法未找到 (-32601)."""

    def __init__(self, method: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("JSONRPC_METHOD_NOT_FOUND", f"Method not found: {method}", {"method": method, **(context or {})})


class InvalidParamsError(JSONRPCError):
    """JSON-RPC 参数无效 (-32602)."""

    def __init__(self, detail: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("JSONRPC_INVALID_PARAMS", detail or "Invalid params", context)


class InternalError(JSONRPCError):
    """JSON-RPC 内部错误 (-32603)."""

    def __init__(self, detail: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("JSONRPC_INTERNAL", detail or "Internal error", context)


# ============================================================
# MCP 协议层异常
# ============================================================

class MCPProtocolError(L6Error):
    """MCP 协议层错误基类."""
    pass


class MCPToolNotFoundError(MCPProtocolError):
    """MCP 工具未找到."""

    def __init__(self, tool_name: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(
            "MCP_TOOL_NOT_FOUND",
            f"Tool not found: {tool_name}",
            {"tool_name": tool_name, **(context or {})},
        )


class MCPToolExecutionError(MCPProtocolError):
    """MCP 工具执行错误.

    Attributes:
        tool_name: 工具名
        is_error: 是否标记为 MCP isError=true
    """

    def __init__(
        self,
        tool_name: str,
        detail: str = "",
        is_error: bool = True,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.is_error = is_error
        super().__init__(
            "MCP_TOOL_EXECUTION_ERROR",
            detail or f"Tool execution failed: {tool_name}",
            {"tool_name": tool_name, "is_error": is_error, **(context or {})},
        )


class MCPResourceNotFoundError(MCPProtocolError):
    """MCP 资源未找到."""

    def __init__(self, uri: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(
            "MCP_RESOURCE_NOT_FOUND",
            f"Resource not found: {uri}",
            {"uri": uri, **(context or {})},
        )


class MCPCapabilityNegotiationError(MCPProtocolError):
    """MCP 能力协商失败."""

    def __init__(
        self,
        client_version: str,
        server_version: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "MCP_CAPABILITY_NEGOTIATION_FAILED",
            detail or f"Version mismatch: client={client_version}, server={server_version}",
            {"client_version": client_version, "server_version": server_version, **(context or {})},
        )


# ============================================================
# Schema 校验异常
# ============================================================

class SchemaValidationError(L6Error):
    """JSON Schema 校验错误."""

    def __init__(
        self,
        path: str,
        message: str,
        value: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.message = message
        self.value = value
        super().__init__(
            "SCHEMA_VALIDATION_ERROR",
            f"{path}: {message}",
            {"path": path, "message": message, **(context or {})},
        )


# ============================================================
# 限流异常
# ============================================================

class RateLimitError(L6Error):
    """MCP 工具限流错误."""

    def __init__(
        self,
        tool_name: str,
        limit: int,
        window_seconds: int,
        retry_after: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            "RATE_LIMIT_EXCEEDED",
            f"Tool {tool_name} rate limit {limit}/{window_seconds}s exceeded",
            {
                "tool_name": tool_name,
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after,
                **(context or {}),
            },
        )


# ============================================================
# A2A 协议异常
# ============================================================

class A2AError(L6Error):
    """A2A 协议层错误基类."""
    pass


class A2AAgentNotFoundError(A2AError):
    """A2A Agent 未找到."""

    def __init__(self, agent_id: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("A2A_AGENT_NOT_FOUND", f"Agent not found: {agent_id}", {"agent_id": agent_id, **(context or {})})


class A2AHandshakeError(A2AError):
    """A2A 握手失败."""

    def __init__(self, reason: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("A2A_HANDSHAKE_FAILED", f"Handshake failed: {reason}", {"reason": reason, **(context or {})})


class A2ACapabilityMismatchError(A2AError):
    """A2A 能力不匹配."""

    def __init__(self, requested: str, available: list[str], context: dict[str, Any] | None = None) -> None:
        super().__init__(
            "A2A_CAPABILITY_MISMATCH",
            f"Requested capability '{requested}' not available. Available: {available}",
            {"requested": requested, "available": available, **(context or {})},
        )


class A2ATaskError(A2AError):
    """A2A 任务执行错误."""

    def __init__(self, task_id: str, detail: str = "", context: dict[str, Any] | None = None) -> None:
        self.task_id = task_id
        super().__init__("A2A_TASK_ERROR", detail or f"Task {task_id} failed", {"task_id": task_id, **(context or {})})


class A2ATimeoutError(A2AError):
    """A2A 任务超时."""

    def __init__(self, task_id: str, timeout_ms: float, context: dict[str, Any] | None = None) -> None:
        self.task_id = task_id
        super().__init__("A2A_TASK_TIMEOUT", f"Task {task_id} timed out after {timeout_ms}ms", {"task_id": task_id, "timeout_ms": timeout_ms, **(context or {})})


class A2ASessionError(A2AError):
    """A2A 会话错误."""

    def __init__(self, session_id: str, detail: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("A2A_SESSION_ERROR", detail or f"Session {session_id} error", {"session_id": session_id, **(context or {})})


class A2ACancelError(A2AError):
    """A2A 任务取消失败."""

    def __init__(self, task_id: str, reason: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("A2A_CANCEL_FAILED", reason or f"Failed to cancel task {task_id}", {"task_id": task_id, **(context or {})})


# ============================================================
# 溯源协议异常
# ============================================================

class ProvenanceError(L6Error):
    """溯源协议层错误基类."""
    pass


class KPAChainBrokenError(ProvenanceError):
    """KPA 链断裂 — prev_hash 不匹配."""

    def __init__(self, kpa_id: str, expected_hash: str, actual_hash: str, context: dict[str, Any] | None = None) -> None:
        self.kpa_id = kpa_id
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            "PROVENANCE_CHAIN_BROKEN",
            f"Chain broken at KPA {kpa_id}: expected prev_hash={expected_hash[:16]}..., got={actual_hash[:16]}...",
            {"kpa_id": kpa_id, "expected_hash": expected_hash, "actual_hash": actual_hash, **(context or {})},
        )


class KPAValidationError(ProvenanceError):
    """KPA 数据校验失败."""

    def __init__(self, kpa_id: str, detail: str, context: dict[str, Any] | None = None) -> None:
        self.kpa_id = kpa_id
        super().__init__("PROVENANCE_VALIDATION_ERROR", f"KPA {kpa_id}: {detail}", {"kpa_id": kpa_id, **(context or {})})


class KPANotFoundError(ProvenanceError):
    """KPA 不存在."""

    def __init__(self, kpa_id: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("PROVENANCE_KPA_NOT_FOUND", f"KPA not found: {kpa_id}", {"kpa_id": kpa_id, **(context or {})})


class KPAImmutableError(ProvenanceError):
    """KPA 不可变 — 尝试修改已封存的 KPA."""

    def __init__(self, kpa_id: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("PROVENANCE_KPA_IMMUTABLE", f"KPA {kpa_id} is sealed and cannot be modified", {"kpa_id": kpa_id, **(context or {})})


# ============================================================
# 算力资源异常
# ============================================================

class ComputeError(L6Error):
    """算力资源层错误基类."""
    pass


class ComputeResourceNotFoundError(ComputeError):
    """算力资源不存在."""

    def __init__(self, resource_id: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("COMPUTE_RESOURCE_NOT_FOUND", f"Resource not found: {resource_id}", {"resource_id": resource_id, **(context or {})})


class ComputeNoAvailableError(ComputeError):
    """无可用算力资源."""

    def __init__(self, reason: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("COMPUTE_NO_AVAILABLE", reason or "No available compute resources", context)


class ComputeQueueFullError(ComputeError):
    """算力资源队列已满."""

    def __init__(self, resource_id: str, max_depth: int, context: dict[str, Any] | None = None) -> None:
        self.resource_id = resource_id
        self.max_depth = max_depth
        super().__init__("COMPUTE_QUEUE_FULL", f"Resource {resource_id} queue full ({max_depth})", {"resource_id": resource_id, "max_depth": max_depth, **(context or {})})


class ComputeTaskNotFoundError(ComputeError):
    """算力任务不存在."""

    def __init__(self, task_id: str, resource_id: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("COMPUTE_TASK_NOT_FOUND", f"Task {task_id} not found" + (f" on {resource_id}" if resource_id else ""), {"task_id": task_id, **(context or {})})


class ComputeDegradationError(ComputeError):
    """算力降级失败 — 无降级候选."""

    def __init__(self, original_type: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("COMPUTE_DEGRADATION_FAILED", f"Degradation failed from {original_type}: no fallback candidates", {"original_type": original_type, **(context or {})})


# ============================================================
# 广播协议异常
# ============================================================

class BroadcastError(L6Error):
    """广播协议层错误基类."""
    pass


class TopicNotFoundError(BroadcastError):
    """主题不存在."""

    def __init__(self, topic: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("BROADCAST_TOPIC_NOT_FOUND", f"Topic not found: {topic}", {"topic": topic, **(context or {})})


class SubscriberNotFoundError(BroadcastError):
    """订阅者不存在."""

    def __init__(self, subscriber_id: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("BROADCAST_SUBSCRIBER_NOT_FOUND", f"Subscriber not found: {subscriber_id}", {"subscriber_id": subscriber_id, **(context or {})})


class BroadcastDeliveryError(BroadcastError):
    """广播投递失败."""

    def __init__(self, topic: str, failed_count: int, context: dict[str, Any] | None = None) -> None:
        super().__init__("BROADCAST_DELIVERY_FAILED", f"Delivery failed for topic '{topic}': {failed_count} subscribers", {"topic": topic, "failed_count": failed_count, **(context or {})})


# ============================================================
# 记忆图谱异常
# ============================================================

class MemoryGraphError(L6Error):
    """记忆图谱层错误基类."""
    pass


class NodeNotFoundError(MemoryGraphError):
    """图谱节点不存在."""

    def __init__(self, node_id: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("MEMORY_NODE_NOT_FOUND", f"Node not found: {node_id}", {"node_id": node_id, **(context or {})})


class EdgeNotFoundError(MemoryGraphError):
    """图谱边不存在."""

    def __init__(self, source_id: str, target_id: str, context: dict[str, Any] | None = None) -> None:
        super().__init__("MEMORY_EDGE_NOT_FOUND", f"Edge not found: {source_id} -> {target_id}", {"source_id": source_id, "target_id": target_id, **(context or {})})


class GraphCycleError(MemoryGraphError):
    """图谱循环依赖."""

    def __init__(self, cycle_path: list[str], context: dict[str, Any] | None = None) -> None:
        self.cycle_path = cycle_path
        super().__init__("MEMORY_GRAPH_CYCLE", f"Cycle detected: {' -> '.join(cycle_path)}", {"cycle_path": cycle_path, **(context or {})})


# ============================================================
# 工具类
# ============================================================

class ErrorCode(str, Enum):
    """标准化错误码枚举，方便跨模块引用."""

    # 传输层
    TRANSPORT_TIMEOUT = "TRANSPORT_TIMEOUT"
    TRANSPORT_CLOSED = "TRANSPORT_CLOSED"
    RECONNECT_EXHAUSTED = "TRANSPORT_RECONNECT_EXHAUSTED"

    # JSON-RPC
    JSONRPC_PARSE = "JSONRPC_PARSE"
    JSONRPC_INVALID_REQUEST = "JSONRPC_INVALID_REQUEST"
    JSONRPC_METHOD_NOT_FOUND = "JSONRPC_METHOD_NOT_FOUND"
    JSONRPC_INVALID_PARAMS = "JSONRPC_INVALID_PARAMS"
    JSONRPC_INTERNAL = "JSONRPC_INTERNAL"

    # MCP 协议
    MCP_TOOL_NOT_FOUND = "MCP_TOOL_NOT_FOUND"
    MCP_TOOL_EXECUTION_ERROR = "MCP_TOOL_EXECUTION_ERROR"
    MCP_RESOURCE_NOT_FOUND = "MCP_RESOURCE_NOT_FOUND"
    MCP_CAPABILITY_NEGOTIATION = "MCP_CAPABILITY_NEGOTIATION_FAILED"

    # 校验
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION_ERROR"
    RATE_LIMIT = "RATE_LIMIT_EXCEEDED"