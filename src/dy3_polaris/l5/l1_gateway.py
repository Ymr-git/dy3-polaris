"""L1 Agent 网关 — 把 L1 认证/权限/审计/上下文接入 L5 Agent 调用."""

from __future__ import annotations

import logging
import time
from typing import Any

from dy3_polaris.l1.access_control import ResourceType
from dy3_polaris.l1.context_broker import FrontendEvent
from dy3_polaris.l1.models import (
    AuditAction,
    AuditResult,
    DataLevel,
    Permission,
    User,
)

logger = logging.getLogger("dy3_polaris.l5.l1_gateway")


class L1AgentGateway:
    """将 L1 能力包装为 Agent 可调用的安全网关."""

    AGENT_PERMISSIONS: dict[str, Permission] = {
        "agent.learning.diagnosis": Permission.AGENT_DIAGNOSIS,
        "agent.knowledge.generation": Permission.AGENT_KNOWLEDGE_GEN,
        "agent.quality.review": Permission.AGENT_REVIEW,
        "agent.guidance.decision": Permission.AGENT_GUIDE,
    }

    def __init__(self, l1_router: Any) -> None:
        self._router = l1_router
        self._auth_mw = l1_router._auth_mw
        self._acm = l1_router._acm
        self._audit_mw = l1_router._audit_mw
        self._users = l1_router._users_by_id
        self._context_broker = l1_router._context_broker
        self._context_collector: Any = None
        try:
            from dy3_polaris.l1.context_broker import ContextCollector

            self._context_collector = ContextCollector()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ContextCollector 初始化失败: %s", exc)

    def authenticate(self, request: Any) -> User | None:
        """从请求中提取并验证 L1 JWT 用户."""
        token = self._auth_mw.extract_bearer_token(
            request.headers.get("Authorization")
        )
        if token is None:
            return None
        payload = self._auth_mw.authenticate(token)
        if payload is None:
            return None
        return self._users.get(payload.user_id)

    def check_agent_permission(
        self,
        user: User,
        agent_id: str,
    ) -> tuple[bool, str]:
        """按 L1 RBAC/ABAC 权限矩阵校验 Agent 调用."""
        permission = self.AGENT_PERMISSIONS.get(agent_id)
        if permission is None:
            return False, f"AGENT_NOT_FOUND: {agent_id}"
        result = self._acm.check_access(
            user=user,
            permission=permission,
            resource_type=ResourceType.AGENT,
            context={"agent_id": agent_id},
        )
        return bool(getattr(result, "allowed", False)), str(
            getattr(result, "reason", "DENIED")
        )

    def audit_agent_call(
        self,
        user: User,
        agent_id: str,
        *,
        success: bool,
        detail: str = "",
    ) -> None:
        """记录 Agent 调用的 L1 审计日志."""
        try:
            self._audit_mw.record(
                user=user,
                action=AuditAction.AGENT_INVOKE,
                resource=f"agent:{agent_id}",
                data_level=DataLevel.L2_INTERNAL,
                result=AuditResult.SUCCESS if success else AuditResult.DENIED,
                purpose=f"run agent {agent_id} {detail}".strip(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent 审计写入失败: %s", exc)

    def collect_agent_call(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        summary: str,
    ) -> None:
        """通过 L1 上下文采集器记录 Agent 调用事件."""
        if self._context_collector is None:
            return
        try:
            self._context_collector.collect_frontend_event(
                FrontendEvent(
                    event_type="agent_call",
                    actor_id=user_id,
                    target_resource=agent_id,
                    timestamp=int(time.time() * 1000),
                    result={
                        "session_id": session_id,
                        "summary": str(summary or "")[:300],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("上下文采集失败: %s", exc)


__all__ = ["L1AgentGateway"]
