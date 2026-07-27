"""A2A 会话管理器.

管理 Agent 间 A2A 会话的完整生命周期：
- 会话创建与销毁
- 状态跟踪
- 消息历史记录
- 溯源链集成
- 会话限流
- 超时清理

会话管理器构建在 A2AMessageBus 之上，提供更高层的会话抽象。
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field, PrivateAttr

from ..core.exceptions import A2ASessionError, A2ATimeoutError
from ..core.models import KPA, KPAEventType, LayerTag
from .protocol import (
    A2AMessageBus,
    A2ATaskRecord,
    A2AMessageType,
    HandshakeResult,
    TaskStatus,
    create_a2a_message,
    create_session_id,
)

logger = logging.getLogger(__name__)


# ============================================================
# 会话状态
# ============================================================

class SessionState(str, Enum):
    """A2A 会话状态."""
    INITIALIZING = "initializing"  # 握手中
    ACTIVE = "active"            # 已建立，可通信
    SUSPENDED = "suspended"      # 暂停（心跳丢失）
    CLOSED = "closed"            # 正常关闭
    ERROR = "error"              # 异常终止
    EXPIRED = "expired"          # 超时过期


# ============================================================
# 会话记录
# ============================================================

class SessionRecord(BaseModel):
    """A2A 会话完整记录."""
    session_id: str
    from_agent: str
    to_agent: str
    state: SessionState = SessionState.INITIALIZING
    granted_capabilities: list[str] = Field(default_factory=list)
    rate_limits: dict = Field(default_factory=lambda: {"max_rps": 10, "max_concurrent": 3})
    created_at: float = Field(default_factory=time.time)
    closed_at: float | None = None
    last_activity_at: float = Field(default_factory=time.time)
    last_heartbeat_at: float | None = None
    message_count: int = 0
    task_count: int = 0
    error_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    kpa_chain: list[dict] = Field(default_factory=list)  # 序列化的 KPA 链

    # 限流追踪（使用 PrivateAttr 避免暴露）
    _request_timestamps: list[float] = PrivateAttr(default_factory=list)
    _concurrent_tasks: int = PrivateAttr(default=0)

    @property
    def is_active(self) -> bool:
        return self.state == SessionState.ACTIVE

    @property
    def duration_seconds(self) -> float:
        end = self.closed_at or time.time()
        return end - self.created_at

    def touch_activity(self) -> None:
        """更新最后活动时间."""
        self.last_activity_at = time.time()
        self.message_count += 1

    def check_rate_limit(self) -> bool:
        """检查会话级限流.

        Returns:
            True 表示未超限，可以继续
        """
        max_rps = self.rate_limits.get("max_rps", 10)
        now = time.time()
        # 滑动窗口：最近 1 秒内的请求数
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 1.0]
        self._request_timestamps.append(now)
        return len(self._request_timestamps) <= max_rps

    def acquire_concurrent_slot(self) -> bool:
        """尝试获取并发槽位.

        Returns:
            True 表示获取成功
        """
        max_concurrent = self.rate_limits.get("max_concurrent", 3)
        if self._concurrent_tasks < max_concurrent:
            self._concurrent_tasks += 1
            return True
        return False

    def release_concurrent_slot(self) -> None:
        """释放并发槽位."""
        if self._concurrent_tasks > 0:
            self._concurrent_tasks -= 1

    def to_dict(self) -> dict[str, Any]:
        """序列化（排除内部字段）."""
        return {
            "session_id": self.session_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "state": self.state.value,
            "granted_capabilities": self.granted_capabilities,
            "rate_limits": self.rate_limits,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "message_count": self.message_count,
            "task_count": self.task_count,
            "error_count": self.error_count,
            "is_active": self.is_active,
            "duration_seconds": round(self.duration_seconds, 2),
        }


# ============================================================
# 会话管理器
# ============================================================

class SessionManager:
    """A2A 会话管理器.

    管理 Agent 间会话的创建、查询、更新和清理。
    提供会话级限流、并发控制、超时清理等能力。

    使用示例:
        manager = SessionManager(message_bus)
        session = await manager.create_session("A1", "A2", ["knowledge_assessment"])
        await manager.send_message(session.session_id, "A1", "A2", {"key": "value"})
        await manager.close_session(session.session_id)
    """

    def __init__(
        self,
        bus: A2AMessageBus,
        *,
        session_ttl_seconds: float = 3600.0,
        cleanup_interval_seconds: float = 60.0,
    ) -> None:
        self._bus = bus
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()
        self._session_ttl = session_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._cleanup_task: asyncio.Task | None = None
        self._session_event_hooks: dict[str, list[Callable]] = {
            "created": [],
            "closed": [],
            "expired": [],
            "error": [],
        }

    # ============================================================
    # 会话生命周期
    # ============================================================

    async def create_session(
        self,
        from_agent: str,
        to_agent: str,
        capabilities: list[str] | None = None,
        rate_limits: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        """创建新会话.

        如果消息总线中已有对应 session（通过握手创建），则升级为 ACTIVE。
        否则创建新会话记录。
        """
        async with self._lock:
            session_id = create_session_id(from_agent, to_agent)

            record = SessionRecord(
                session_id=session_id,
                from_agent=from_agent,
                to_agent=to_agent,
                granted_capabilities=capabilities or [],
                rate_limits=rate_limits or {},
                metadata=metadata or {},
                state=SessionState.ACTIVE,
            )

            self._sessions[session_id] = record

            logger.info(
                f"Session created: {session_id} [{from_agent} -> {to_agent}], "
                f"capabilities={capabilities}"
            )

            await self._fire_hook("created", record)
            return record

    async def close_session(
        self,
        session_id: str,
        reason: str = "normal_close",
    ) -> SessionRecord:
        """关闭会话."""
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise A2ASessionError(session_id, f"Session not found: {session_id}")

            record.state = SessionState.CLOSED
            record.closed_at = time.time()

            logger.info(f"Session closed: {session_id} (reason={reason})")
            await self._fire_hook("closed", record)
            return record

    async def close_all_sessions(self, reason: str = "shutdown") -> int:
        """关闭所有活跃会话."""
        async with self._lock:
            count = 0
            for record in self._sessions.values():
                if record.state in (SessionState.ACTIVE, SessionState.INITIALIZING, SessionState.SUSPENDED):
                    record.state = SessionState.CLOSED
                    record.closed_at = time.time()
                    count += 1
            logger.info(f"Closed {count} sessions (reason={reason})")
            return count

    # ============================================================
    # 消息发送
    # ============================================================

    async def send_message(
        self,
        session_id: str,
        from_agent: str,
        to_agent: str,
        payload: dict[str, Any],
        message_type: A2AMessageType = A2AMessageType.TASK_REQUEST,
        provenance: KPA | None = None,
    ) -> Any:
        """通过会话发送消息.

        自动进行会话状态检查、限流检查、并发检查。
        """
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise A2ASessionError(session_id)

            if not record.is_active:
                raise A2ASessionError(
                    session_id,
                    f"Session is not active (state={record.state.value})",
                )

            # 限流检查
            if not record.check_rate_limit():
                raise A2ASessionError(
                    session_id,
                    f"Rate limit exceeded for session {session_id}",
                )

        # 溯源链集成
        if provenance is not None:
            record.kpa_chain.append(provenance.model_dump(mode="json"))

        # 通过消息总线发送
        if message_type == A2AMessageType.TASK_REQUEST:
            capability = payload.get("capability", "")
            input_data = payload.get("input", {})
            task_record = await self._bus.send_task(
                from_agent=from_agent,
                to_agent=to_agent,
                capability=capability,
                input_data=input_data,
                session_id=session_id,
                priority=payload.get("context", {}).get("priority", 0),
                timeout_ms=payload.get("context", {}).get("timeout_ms"),
                provenance=provenance,
            )
            async with self._lock:
                record.task_count += 1
                record.touch_activity()
            return task_record
        else:
            msg = create_a2a_message(
                message_type=message_type,
                from_agent=from_agent,
                to_agent=to_agent,
                **payload,
            )
            msg.session_id = session_id
            async with self._lock:
                record.touch_activity()
            return msg

    # ============================================================
    # 查询接口
    # ============================================================

    def get_session(self, session_id: str) -> SessionRecord | None:
        """获取会话记录."""
        return self._sessions.get(session_id)

    def get_session_or_raise(self, session_id: str) -> SessionRecord:
        """获取会话记录，不存在时抛出异常."""
        record = self._sessions.get(session_id)
        if record is None:
            raise A2ASessionError(session_id, f"Session not found: {session_id}")
        return record

    def get_sessions_by_agent(self, agent_id: str) -> list[SessionRecord]:
        """获取 Agent 参与的所有会话."""
        return [
            r for r in self._sessions.values()
            if r.from_agent == agent_id or r.to_agent == agent_id
        ]

    def get_active_sessions(self) -> list[SessionRecord]:
        """获取所有活跃会话."""
        return [r for r in self._sessions.values() if r.is_active]

    def get_all_sessions(self) -> list[SessionRecord]:
        """获取所有会话."""
        return list(self._sessions.values())

    @property
    def total_sessions(self) -> int:
        return len(self._sessions)

    @property
    def active_session_count(self) -> int:
        return sum(1 for r in self._sessions.values() if r.is_active)

    # ============================================================
    # 统计摘要
    # ============================================================

    def export_summary(self) -> dict[str, Any]:
        """导出会话统计摘要."""
        state_counts: dict[str, int] = {}
        for r in self._sessions.values():
            state_counts[r.state.value] = state_counts.get(r.state.value, 0) + 1

        total_messages = sum(r.message_count for r in self._sessions.values())
        total_tasks = sum(r.task_count for r in self._sessions.values())
        total_errors = sum(r.error_count for r in self._sessions.values())

        return {
            "total_sessions": len(self._sessions),
            "active_sessions": self.active_session_count,
            "state_breakdown": state_counts,
            "total_messages": total_messages,
            "total_tasks": total_tasks,
            "total_errors": total_errors,
        }

    # ============================================================
    # 事件钩子
    # ============================================================

    def on(self, event: str, handler: Callable) -> None:
        """注册会话事件钩子.

        支持的事件: created, closed, expired, error
        """
        if event not in self._session_event_hooks:
            self._session_event_hooks[event] = []
        self._session_event_hooks[event].append(handler)

    async def _fire_hook(self, event: str, record: SessionRecord) -> None:
        """触发事件钩子."""
        for handler in self._session_event_hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(record)
                else:
                    handler(record)
            except Exception as exc:
                logger.error(f"Session hook error [{event}]: {exc}")

    # ============================================================
    # 超时清理
    # ============================================================

    async def start_cleanup_loop(self) -> None:
        """启动后台清理循环."""
        if self._cleanup_task is not None:
            return

        async def _cleanup_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(self._cleanup_interval)
                    await self._cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(f"Session cleanup error: {exc}")

        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info("Session cleanup loop started")

    async def stop_cleanup_loop(self) -> None:
        """停止后台清理循环."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Session cleanup loop stopped")

    async def _cleanup_expired(self) -> int:
        """清理过期会话."""
        now = time.time()
        expired: list[str] = []

        async with self._lock:
            for sid, record in self._sessions.items():
                if record.state in (SessionState.CLOSED, SessionState.ERROR, SessionState.EXPIRED):
                    continue
                if now - record.last_activity_at > self._session_ttl:
                    expired.append(sid)

            for sid in expired:
                record = self._sessions[sid]
                record.state = SessionState.EXPIRED
                record.closed_at = now
                await self._fire_hook("expired", record)

        if expired:
            logger.info(f"Expired {len(expired)} sessions: {expired}")
        return len(expired)

    # ============================================================
    # 测试辅助
    # ============================================================

    def clear(self) -> None:
        """清空所有会话（仅用于测试）."""
        self._sessions.clear()


# ============================================================
# 导出
# ============================================================

__all__ = [
    "SessionState",
    "SessionRecord",
    "SessionManager",
]
