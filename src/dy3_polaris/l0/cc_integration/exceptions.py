"""CC4 三横切集成 — 异常定义.

继承 L6Error 异常体系, 集成 JSON-RPC 错误码.
错误码范围 -32400 ~ -32406.
"""

from __future__ import annotations

from typing import Any

from ...l6.core.exceptions import L6Error


class CC4Error(L6Error):
    """CC4 三横切集成基础异常."""

    def __init__(
        self,
        message: str,
        *,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, detail=detail, context=context or {})

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32400


class BridgeConnectionError(CC4Error):
    """桥接连接错误 — 桥接器无法连接到目标模块."""

    def __init__(
        self,
        source: str,
        target: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"桥接连接失败: {source} → {target}: {reason}",
            detail=f"Source={source}, Target={target}, Reason={reason}",
            context={"source": source, "target": target, "reason": reason},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32401


class FeedbackLoopError(CC4Error):
    """反馈飞轮错误 — 反馈循环执行失败."""

    def __init__(
        self,
        signal_type: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"反馈飞轮执行失败: {signal_type}: {reason}",
            detail=f"SignalType={signal_type}, Reason={reason}",
            context={"signal_type": signal_type, "reason": reason},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32402


class GatewayRoutingError(CC4Error):
    """网关路由错误 — 统一网关无法路由请求."""

    def __init__(
        self,
        path: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"网关路由失败: {path}: {reason}",
            detail=f"Path={path}, Reason={reason}",
            context={"path": path, "reason": reason},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32403


class HealthCheckError(CC4Error):
    """健康检查错误 — 健康检查执行失败."""

    def __init__(
        self,
        module: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"健康检查失败: {module}: {reason}",
            detail=f"Module={module}, Reason={reason}",
            context={"module": module, "reason": reason},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32404


class CircuitBreakerOpenError(CC4Error):
    """断路器开启错误 — 目标模块断路器处于 OPEN 状态."""

    def __init__(
        self,
        module: str,
        retry_after: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"断路器开启: 模块 {module} 暂时不可用, {retry_after:.1f}s 后重试",
            detail=f"Module={module}, RetryAfter={retry_after}s",
            context={"module": module, "retry_after": retry_after},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32405


class GovernancePolicyError(CC4Error):
    """治理策略错误 — 治理策略验证或执行失败."""

    def __init__(
        self,
        policy: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"治理策略错误: {policy}: {reason}",
            detail=f"Policy={policy}, Reason={reason}",
            context={"policy": policy, "reason": reason},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32406
