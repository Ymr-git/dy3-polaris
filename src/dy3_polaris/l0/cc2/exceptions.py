"""CC2 人机协作层 — 异常定义.

继承 L6Error 异常体系，集成 JSON-RPC 错误码。
错误码范围 -32300 ~ -32306。
"""

from __future__ import annotations

from typing import Any

from ...l6.core.exceptions import L6Error


class CC2Error(L6Error):
    """CC2 人机协作层基础异常."""

    def __init__(
        self,
        code: str = "CC2_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32300


class InterventionTimeoutError(CC2Error):
    """干预超时."""

    def __init__(
        self,
        request_id: str,
        timeout_seconds: float,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.request_id = request_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            "CC2_INTERVENTION_TIMEOUT",
            detail or f"request_id={request_id}, timeout={timeout_seconds}s",
            {"request_id": request_id, "timeout_seconds": timeout_seconds, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32301


class NegotiationExhaustedError(CC2Error):
    """协商轮次耗尽."""

    def __init__(
        self,
        session_id: str,
        rounds: int,
        max_rounds: int,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.rounds = rounds
        self.max_rounds = max_rounds
        super().__init__(
            "CC2_NEGOTIATION_EXHAUSTED",
            detail or f"session_id={session_id}, rounds={rounds}, max={max_rounds}",
            {"session_id": session_id, "rounds": rounds, "max_rounds": max_rounds, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32302


class ProfileNotFoundError(CC2Error):
    """Agent 协作配置未找到."""

    def __init__(
        self,
        agent_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        super().__init__(
            "CC2_PROFILE_NOT_FOUND",
            detail or f"agent_id={agent_id}",
            {"agent_id": agent_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32303


class ModeSwitchError(CC2Error):
    """模式切换错误."""

    def __init__(
        self,
        agent_id: str,
        from_mode: str,
        to_mode: str,
        reason: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.from_mode = from_mode
        self.to_mode = to_mode
        super().__init__(
            "CC2_MODE_SWITCH_ERROR",
            reason or f"from={from_mode}, to={to_mode}",
            {"agent_id": agent_id, "from_mode": from_mode, "to_mode": to_mode, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32304


class InterventionConflictError(CC2Error):
    """干预冲突."""

    def __init__(
        self,
        request_id: str,
        reason: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.request_id = request_id
        super().__init__(
            "CC2_INTERVENTION_CONFLICT",
            reason,
            {"request_id": request_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32305


class EscalationTargetError(CC2Error):
    """升级目标错误."""

    def __init__(
        self,
        target: str,
        agent_id: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.target = target
        self.agent_id = agent_id
        super().__init__(
            "CC2_ESCALATION_TARGET_ERROR",
            detail or f"target={target}, agent_id={agent_id}",
            {"target": target, "agent_id": agent_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32306
