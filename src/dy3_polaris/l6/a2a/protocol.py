"""A2A 协议引擎.

实现 Agent-to-Agent 通信协议的核心组件，包括：
- 协议版本管理
- Agent 身份标识
- 握手协商
- 任务记录与生命周期
- 消息总线（路由、分发、会话管理）
- 心跳监控

所有公共接口均使用 asyncio，支持高并发 Agent 协作。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from enum import Enum
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field

from .metrics import A2AMetrics
from ..core.config import get_config
from ..core.exceptions import (
    A2AAgentNotFoundError,
    A2ACancelError,
    A2ACapabilityMismatchError,
    A2AError,
    A2AHandshakeError,
    A2ASessionError,
    A2ATaskError,
    A2ATimeoutError,
)
from ..core.models import (
    A2ACapability,
    A2AMessage,
    A2AMessageType,
    KPA,
    LayerTag,
)

logger = logging.getLogger(__name__)


# ============================================================
# 日志容量上限
# ============================================================

_MESSAGE_LOG_CAP = 1000  # 消息日志最大条数


# ============================================================
# 协议版本
# ============================================================

class A2AProtocolVersion:
    """A2A 协议版本管理.

    定义当前协议版本号及所有支持的版本列表。
    用于握手阶段的版本协商。
    """

    CURRENT: str = "a2a/1.0"
    SUPPORTED: list[str] = ["a2a/1.0"]


# ============================================================
# Agent 身份标识
# ============================================================

class AgentIdentity(BaseModel):
    """Agent 身份标识模型.

    描述一个 Agent 在 A2A 网络中的身份信息，
    包括 ID、租户、DID 标识及通信端点。

    Attributes:
        agent_id: Agent 唯一标识，如 "dy3+tutor-agent"
        tenant_id: 租户 ID（可选，多租户场景使用）
        did: 去中心化身份标识（可选）
        endpoint: Agent 的 A2A 通信端点（可选），如 "a2a://dy3+system/agents/tutor"
    """

    agent_id: str
    tenant_id: str | None = None
    did: str | None = None
    endpoint: str | None = None


# ============================================================
# 握手结果
# ============================================================

class HandshakeResult(BaseModel):
    """握手协商结果.

    记录握手是否成功、授予的能力集合、会话 ID 及限流配置。

    Attributes:
        status: 握手状态，"accepted" 或 "rejected"
        granted_capabilities: 对方同意授予的能力列表
        session_id: 握手成功后分配的会话 ID
        rate_limits: 限流配置，默认最大 10 RPS、3 并发
        rejection_reason: 拒绝原因（仅在 rejected 时有值）
    """

    status: str  # "accepted" | "rejected"
    granted_capabilities: list[str] = []
    session_id: str = ""
    rate_limits: dict = {"max_rps": 10, "max_concurrent": 3}
    rejection_reason: str = ""


# ============================================================
# 任务状态
# ============================================================

class TaskStatus(str, Enum):
    """A2A 任务状态枚举.

    任务在生命周期中的所有可能状态。
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# ============================================================
# 任务记录
# ============================================================

class A2ATaskRecord(BaseModel):
    """A2A 任务记录.

    完整记录一个跨 Agent 任务的生命周期信息，
    包括状态、输入输出、时间戳及溯源链。

    Attributes:
        task_id: 任务唯一 ID
        session_id: 所属会话 ID（可选）
        from_agent: 发起方 Agent ID
        to_agent: 接收方 Agent ID
        capability: 请求的能力名称
        status: 当前任务状态
        input_data: 任务输入数据
        result: 任务执行结果（完成后填写）
        error: 错误信息（失败时填写）
        created_at: 创建时间（Unix 时间戳）
        updated_at: 最后更新时间
        started_at: 开始执行时间
        completed_at: 完成时间
        timeout_ms: 超时时间（毫秒）
        priority: 优先级，数值越大优先级越高
        kpa_chain: 序列化的 KPA 溯源链
    """

    task_id: str
    session_id: str | None = None
    from_agent: str
    to_agent: str
    capability: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    input_data: dict = {}
    result: dict | None = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    timeout_ms: float | None = None
    priority: int = 0
    kpa_chain: list[dict] = []  # 序列化的 KPA 链


# ============================================================
# 辅助函数
# ============================================================

def create_task_id() -> str:
    """生成唯一任务 ID.

    基于 UUID v4 生成，前缀为 'task_' 以便于识别。

    Returns:
        唯一任务 ID 字符串
    """
    return f"task-{uuid.uuid4().hex[:20]}"


def create_session_id(from_agent: str, to_agent: str) -> str:
    """根据 Agent 对生成确定性会话 ID.

    对两个 Agent ID 排序后拼接并取 SHA-256 哈希，
    确保同一对 Agent 无论顺序如何都能得到相同的会话 ID。

    Args:
        from_agent: 发起方 Agent ID
        to_agent: 接收方 Agent ID

    Returns:
        确定性的会话 ID
    """
    pair = sorted([from_agent, to_agent])
    raw = f"{pair[0]}:{pair[1]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def create_a2a_message(
    message_type: A2AMessageType,
    from_agent: str,
    to_agent: str,
    **payload: Any,
) -> A2AMessage:
    """创建 A2A 协议消息.

    快捷构造函数，自动生成 message_id 和 timestamp。

    Args:
        message_type: 消息类型
        from_agent: 发送方 Agent ID
        to_agent: 接收方 Agent ID
        **payload: 消息载荷键值对

    Returns:
        构造完成的 A2AMessage 实例
    """
    return A2AMessage(
        message_type=message_type,
        from_agent=from_agent,
        to_agent=to_agent,
        payload=payload,
    )


# ============================================================
# 消息总线
# ============================================================

class A2AMessageBus:
    """A2A 消息总线.

    核心消息路由与分发组件，负责：
    - Agent 注册与发现
    - 握手协商
    - 任务创建、路由与取消
    - 心跳维护
    - 消息日志记录

    所有公共方法在内部使用 asyncio.Lock 保证线程安全。
    """

    def __init__(self) -> None:
        """初始化消息总线.

        创建内部注册表、会话存储、任务存储和消息日志。
        """
        # Agent 能力注册表：agent_id -> A2ACapability
        self._agents: dict[str, A2ACapability] = {}

        # 会话存储：session_id -> 会话数据字典
        self._sessions: dict[str, dict[str, Any]] = {}

        # 任务存储：task_id -> A2ATaskRecord
        self._tasks: dict[str, A2ATaskRecord] = {}

        # 消息日志（有容量上限）
        self._message_log: list[A2AMessage] = []

        # 异步锁，保证线程安全
        self._lock = asyncio.Lock()

        # 任务处理器回调（可选，用于实际的任务分发）
        self._task_handler: Callable[
            [A2ATaskRecord], Coroutine[Any, Any, dict]
        ] | None = None

        # 可观测性指标
        self.metrics = A2AMetrics()

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _append_log(self, message: A2AMessage) -> None:
        """将消息追加到日志中，超过容量上限时截断旧消息.

        Args:
            message: 要记录的 A2A 消息
        """
        self._message_log.append(message)
        # 超出容量时移除最旧的消息
        if len(self._message_log) > _MESSAGE_LOG_CAP:
            self._message_log = self._message_log[-_MESSAGE_LOG_CAP:]
        # 指标记录
        self.metrics.on_message_sent(message.message_type.value, message.from_agent, message.to_agent)

    # --------------------------------------------------------
    # Agent 注册与发现
    # --------------------------------------------------------

    def register_agent(self, agent_id: str, capability: A2ACapability) -> None:
        """注册 Agent 及其能力声明.

        将 Agent 的能力信息存入注册表，供后续发现和握手使用。

        Args:
            agent_id: Agent 唯一标识
            capability: Agent 的能力声明
        """
        self._agents[agent_id] = capability
        logger.debug("Agent 已注册: %s (%s)", agent_id, capability.agent_name)

    def unregister_agent(self, agent_id: str) -> A2ACapability | None:
        """注销 Agent.

        从注册表中移除指定 Agent。

        Args:
            agent_id: 要移除的 Agent ID

        Returns:
            被移除的 Agent 能力声明，若不存在则返回 None
        """
        cap = self._agents.pop(agent_id, None)
        if cap is not None:
            logger.debug("Agent 已注销: %s", agent_id)
        return cap

    def discover_agents(
        self,
        *,
        capability: str | None = None,
        domain: str | None = None,
    ) -> list[A2ACapability]:
        """发现 Agent.

        根据能力和领域范围筛选已注册的 Agent。
        若未指定筛选条件，则返回所有已注册 Agent。

        Args:
            capability: 按能力名称筛选（可选）
            domain: 按领域范围筛选（可选）

        Returns:
        匹配的 Agent 能力声明列表
        """
        results: list[A2ACapability] = []
        for cap in self._agents.values():
            # 按能力筛选
            if capability is not None:
                if capability not in cap.supported_methods and capability not in cap.supported_tools:
                    continue
            # 按领域筛选
            if domain is not None:
                if domain not in cap.domain_scope:
                    continue
            results.append(cap)
        return results

    def get_agent(self, agent_id: str) -> A2ACapability | None:
        """获取指定 Agent 的能力声明.

        Args:
            agent_id: Agent ID

        Returns:
            Agent 的能力声明，若不存在则返回 None
        """
        return self._agents.get(agent_id)

    # --------------------------------------------------------
    # Discovery 消息
    # --------------------------------------------------------

    async def send_discovery(self, capability: A2ACapability) -> A2AMessage:
        """发送 Discovery 消息.

        构造并发送一条 DISCOVERY 类型的消息，
        用于向网络广播自身的存在和能力。

        Args:
            capability: 发送方的能力声明

        Returns:
            已发出的 Discovery 消息
        """
        msg = create_a2a_message(
            message_type=A2AMessageType.DISCOVERY,
            from_agent=capability.agent_id,
            to_agent="*",  # 广播
            capability=capability.model_dump(),
            protocol_version=A2AProtocolVersion.CURRENT,
        )
        self._append_log(msg)
        logger.debug("已发送 Discovery 消息: %s -> *", capability.agent_id)
        return msg

    # --------------------------------------------------------
    # 握手
    # --------------------------------------------------------

    async def initiate_handshake(
        self,
        from_agent: str,
        to_agent: str,
        requested_capabilities: list[str],
        *,
        auth_token: str = "",
    ) -> HandshakeResult:
        """发起握手协商.

        完整的握手流程：
        1. 验证目标 Agent 是否存在
        2. 验证请求的能力是否至少有一个被支持
        3. 创建或复用会话
        4. 返回握手结果

        Args:
            from_agent: 发起方 Agent ID
            to_agent: 目标 Agent ID
            requested_capabilities: 请求的能力列表
            auth_token: 认证令牌（可选）

        Returns:
            握手结果，包含状态、授予的能力和会话 ID

        Raises:
            A2AAgentNotFoundError: 目标 Agent 不存在
            A2ACapabilityMismatchError: 请求的能力无交集
        """
        async with self._lock:
            # 构造握手请求消息
            request_msg = create_a2a_message(
                message_type=A2AMessageType.HANDSHAKE_REQUEST,
                from_agent=from_agent,
                to_agent=to_agent,
                requested_capabilities=requested_capabilities,
                protocol_version=A2AProtocolVersion.CURRENT,
                auth_token=auth_token,
            )
            self._append_log(request_msg)

            # 步骤 1: 验证目标 Agent 是否存在
            target_cap = self._agents.get(to_agent)
            if target_cap is None:
                logger.warning("握手失败: 目标 Agent 不存在 %s", to_agent)
                result = HandshakeResult(
                    status="rejected",
                    rejection_reason=f"目标 Agent 不存在: {to_agent}",
                )
                # 记录握手响应消息
                response_msg = create_a2a_message(
                    message_type=A2AMessageType.HANDSHAKE_RESPONSE,
                    from_agent=to_agent,
                    to_agent=from_agent,
                    status="rejected",
                    reason=result.rejection_reason,
                )
                self._append_log(response_msg)
                raise A2AAgentNotFoundError(to_agent)

            # 步骤 2: 验证能力交集
            target_all = set(target_cap.supported_methods + target_cap.supported_tools)
            requested_set = set(requested_capabilities)
            granted = list(requested_set & target_all)

            if not granted:
                logger.warning(
                    "握手失败: 能力不匹配. 请求=%s, 可用=%s",
                    requested_capabilities,
                    list(target_all),
                )
                result = HandshakeResult(
                    status="rejected",
                    rejection_reason=f"能力不匹配: 请求的能力 {requested_capabilities} 无交集",
                )
                response_msg = create_a2a_message(
                    message_type=A2AMessageType.HANDSHAKE_RESPONSE,
                    from_agent=to_agent,
                    to_agent=from_agent,
                    status="rejected",
                    reason=result.rejection_reason,
                )
                self._append_log(response_msg)
                raise A2ACapabilityMismatchError(
                    requested=", ".join(requested_capabilities),
                    available=list(target_all),
                )

            # 步骤 3: 创建会话
            session_id = create_session_id(from_agent, to_agent)
            rate_limits = {"max_rps": 10, "max_concurrent": min(target_cap.max_concurrent_tasks, 3)}

            self._sessions[session_id] = {
                "session_id": session_id,
                "agents": (from_agent, to_agent),
                "granted_capabilities": granted,
                "rate_limits": rate_limits,
                "created_at": time.time(),
                "last_heartbeat": time.time(),
                "last_heartbeat_at": time.time(),
            }

            # 构造成功结果
            result = HandshakeResult(
                status="accepted",
                granted_capabilities=granted,
                session_id=session_id,
                rate_limits=rate_limits,
            )

            # 记录握手响应消息
            response_msg = create_a2a_message(
                message_type=A2AMessageType.HANDSHAKE_RESPONSE,
                from_agent=to_agent,
                to_agent=from_agent,
                status="accepted",
                session_id=session_id,
                granted_capabilities=granted,
                rate_limits=rate_limits,
            )
            self._append_log(response_msg)

            self.metrics.on_session_created()

            logger.info(
                "握手成功: %s <-> %s, 会话=%s, 授予能力=%s",
                from_agent,
                to_agent,
                session_id[:8],
                granted,
            )
            return result

    # --------------------------------------------------------
    # 任务管理
    # --------------------------------------------------------

    async def send_task(
        self,
        from_agent: str,
        to_agent: str,
        capability: str,
        input_data: dict[str, Any],
        *,
        session_id: str = "",
        priority: int = 0,
        timeout_ms: float | None = None,
        provenance: KPA | None = None,
    ) -> A2ATaskRecord:
        """发送任务请求.

        创建任务记录，路由到目标 Agent 并执行。
        若已注册任务处理器则调用处理器，否则返回桩结果。

        Args:
            from_agent: 发起方 Agent ID
            to_agent: 接收方 Agent ID
            capability: 请求的能力名称
            input_data: 任务输入数据
            session_id: 会话 ID（可选）
            priority: 优先级（默认 0）
            timeout_ms: 超时时间（毫秒，可选）
            provenance: 溯源 KPA（可选）

        Returns:
            任务记录

        Raises:
            A2AAgentNotFoundError: 目标 Agent 不存在
        """
        async with self._lock:
            # 验证目标 Agent 存在
            if to_agent not in self._agents:
                raise A2AAgentNotFoundError(to_agent)

            # 创建任务记录
            task_id = create_task_id()
            now = time.time()
            task = A2ATaskRecord(
                task_id=task_id,
                session_id=session_id or None,
                from_agent=from_agent,
                to_agent=to_agent,
                capability=capability,
                status=TaskStatus.PENDING,
                input_data=input_data,
                created_at=now,
                updated_at=now,
                timeout_ms=timeout_ms,
                priority=priority,
                kpa_chain=[provenance.model_dump()] if provenance else [],
            )
            self._tasks[task_id] = task

            # 指标记录
            self.metrics.on_task_created(task_id, from_agent, to_agent)
            if capability:
                self.metrics.on_capability_requested(capability)

            # 构造并发送任务请求消息
            msg = create_a2a_message(
                message_type=A2AMessageType.TASK_REQUEST,
                from_agent=from_agent,
                to_agent=to_agent,
                task_id=task_id,
                capability=capability,
                input_data=input_data,
                session_id=task.session_id,
                priority=priority,
                timeout_ms=timeout_ms,
            )
            self._append_log(msg)

            # 标记任务为运行中
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.updated_at = time.time()

        # --- 锁外执行 handler（允许 cancel_task 在 handler 运行时获取锁）---
        try:
            if self._task_handler is not None:
                result_data = await self._task_handler(task)
            else:
                result_data = {"stub": True, "message": "未注册任务处理器，返回桩结果"}

            async with self._lock:
                if task.status == TaskStatus.CANCELLED:
                    # handler 返回后任务已被取消
                    self.metrics.on_task_cancelled()
                    return task
                task.status = TaskStatus.COMPLETED
                task.result = result_data
                task.completed_at = time.time()
                task.updated_at = time.time()

                # 记录指标
                latency_ms = (task.completed_at - (task.started_at or task.created_at)) * 1000
                self.metrics.on_task_completed(latency_ms=latency_ms)

                resp_msg = create_a2a_message(
                    message_type=A2AMessageType.TASK_RESPONSE,
                    from_agent=to_agent,
                    to_agent=from_agent,
                    task_id=task_id,
                    status="completed",
                    result=result_data,
                    session_id=task.session_id,
                )
                self._append_log(resp_msg)

        except Exception as exc:
            async with self._lock:
                if task.status == TaskStatus.CANCELLED:
                    self.metrics.on_task_cancelled()
                    return task
                task.status = TaskStatus.FAILED
                self.metrics.on_task_failed()
                task.error = str(exc)
                task.completed_at = time.time()
                task.updated_at = time.time()

                err_msg = create_a2a_message(
                    message_type=A2AMessageType.TASK_ERROR,
                    from_agent=to_agent,
                    to_agent=from_agent,
                    task_id=task_id,
                    status="failed",
                    error=str(exc),
                    session_id=task.session_id,
                )
                self._append_log(err_msg)
                logger.error("任务执行失败: %s, 错误: %s", task_id, exc)

        logger.debug("任务已完成: %s, 状态=%s", task_id, task.status.value)
        return task

    async def cancel_task(self, task_id: str, reason: str = "") -> A2ATaskRecord:
        """取消任务.

        仅允许取消处于 PENDING 或 RUNNING 状态的任务。

        Args:
            task_id: 要取消的任务 ID
            reason: 取消原因

        Returns:
        更新后的任务记录

        Raises:
            A2ACancelError: 任务不存在或状态不允许取消
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise A2ACancelError(task_id, f"任务不存在: {task_id}")

            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                raise A2ACancelError(
                    task_id,
                    f"无法取消状态为 '{task.status.value}' 的任务",
                )

            # 更新任务状态
            task.status = TaskStatus.CANCELLED
            task.error = reason or "用户取消"
            task.completed_at = time.time()
            task.updated_at = time.time()

            # 发送取消消息
            msg = create_a2a_message(
                message_type=A2AMessageType.CANCEL,
                from_agent=task.from_agent,
                to_agent=task.to_agent,
                task_id=task_id,
                reason=reason or "用户取消",
                session_id=task.session_id,
            )
            self._append_log(msg)

            logger.info("任务已取消: %s, 原因: %s", task_id, reason or "用户取消")
            return task

    # --------------------------------------------------------
    # 心跳
    # --------------------------------------------------------

    async def send_heartbeat(self, agent_id: str, *, session_id: str = "") -> A2AMessage:
        """发送心跳消息.

        向指定会话或所有与 agent_id 相关的会话发送心跳，
        并更新会话的最后心跳时间戳。

        Args:
            agent_id: 发送心跳的 Agent ID
            session_id: 指定会话 ID（可选，若为空则广播所有关联会话）

        Returns:
            心跳消息
        """
        async with self._lock:
            msg = create_a2a_message(
                message_type=A2AMessageType.HEARTBEAT,
                from_agent=agent_id,
                to_agent="*",
                session_id=session_id or None,
            )
            self._append_log(msg)

            now = time.time()
            for sid, session in self._sessions.items():
                # 若指定了 session_id，只更新该会话
                if session_id and sid != session_id:
                    continue
                agents_pair = session.get("agents", ())
                if agent_id in agents_pair:
                    session["last_heartbeat"] = now
                    session["last_heartbeat_at"] = now

                    # 发送心跳确认消息
                    ack_msg = create_a2a_message(
                        message_type=A2AMessageType.HEARTBEAT_ACK,
                        from_agent=agent_id,
                        to_agent="*",
                        session_id=sid,
                    )
                    self._append_log(ack_msg)

            logger.debug("心跳已发送: %s", agent_id)
            return msg

    # --------------------------------------------------------
    # 查询接口
    # --------------------------------------------------------

    def get_task(self, task_id: str) -> A2ATaskRecord | None:
        """获取指定任务记录.

        Args:
            task_id: 任务 ID

        Returns:
            任务记录，若不存在则返回 None
        """
        return self._tasks.get(task_id)

    def get_tasks_by_session(self, session_id: str) -> list[A2ATaskRecord]:
        """获取指定会话下的所有任务.

        Args:
            session_id: 会话 ID

        Returns:
            该会话下的任务记录列表
        """
        return [
            t for t in self._tasks.values()
            if t.session_id == session_id
        ]

    def get_tasks_by_agent(self, agent_id: str) -> list[A2ATaskRecord]:
        """获取与指定 Agent 相关的所有任务.

        包括该 Agent 作为发起方或接收方的所有任务。

        Args:
            agent_id: Agent ID

        Returns:
            相关任务记录列表
        """
        return [
            t for t in self._tasks.values()
            if t.from_agent == agent_id or t.to_agent == agent_id
        ]

    def get_all_tasks(self) -> list[A2ATaskRecord]:
        """获取所有任务记录.

        Returns:
            全部任务记录列表
        """
        return list(self._tasks.values())

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取指定会话信息.

        Args:
            session_id: 会话 ID

        Returns:
            会话数据字典，若不存在则返回 None
        """
        return self._sessions.get(session_id)

    def register_task_handler(
        self,
        handler: Callable[[A2ATaskRecord], Coroutine[Any, Any, dict]],
    ) -> None:
        """注册任务处理器.

        设置用于处理传入任务的异步回调函数。
        处理器接收 A2ATaskRecord，返回结果字典。

        Args:
            handler: 异步任务处理函数
        """
        self._task_handler = handler
        logger.debug("任务处理器已注册")

    def export_message_log(self) -> list[dict[str, Any]]:
        """导出消息日志.

        将内部消息日志序列化为字典列表，
        用于调试和测试。

        Returns:
            消息字典列表
        """
        return [msg.model_dump() for msg in self._message_log]

    def reset(self) -> None:
        """重置消息总线.

        清除所有内部状态，包括注册表、会话、任务和消息日志。
        主要用于测试场景。
        """
        self._agents.clear()
        self._sessions.clear()
        self._tasks.clear()
        self._message_log.clear()
        self._task_handler = None
        self.metrics.reset()
        logger.debug("消息总线已重置")


# ============================================================
# 心跳监控器
# ============================================================

class HeartbeatMonitor:
    """A2A 心跳监控器.

    后台定期检查所有会话的心跳状态，
    识别超过阈值的过期会话。

    过期判定条件：当前时间 - 最后心跳时间 > 3 * 检测间隔
    """

    def __init__(
        self,
        bus: A2AMessageBus,
        interval: float | None = None,
    ) -> None:
        """初始化心跳监控器.

        Args:
            bus: 关联的消息总线实例
            interval: 检测间隔（秒），默认从配置读取，最低 5 秒
        """
        self._bus = bus
        self._interval: float = interval or get_config().a2a_heartbeat_interval
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False

    async def start(self) -> None:
        """启动心跳监控.

        创建后台 asyncio 任务，定期检查会话心跳状态。
        若监控已在运行，则不做任何操作。
        """
        if self._running:
            logger.debug("心跳监控已在运行")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("心跳监控已启动，间隔=%.1f秒", self._interval)

    async def stop(self) -> None:
        """停止心跳监控.

        取消后台任务并等待其结束。
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("心跳监控已停止")

    async def _monitor_loop(self) -> None:
        """后台监控循环.

        每隔 _interval 秒检查一次所有会话的心跳状态，
        对过期会话记录警告日志。
        """
        try:
            while self._running:
                await asyncio.sleep(self._interval)
                stale = self.get_stale_sessions()
                for sid in stale:
                    logger.warning(
                        "会话心跳过期: %s",
                        sid,
                    )
        except asyncio.CancelledError:
            # 正常退出
            pass

    async def check_session(self, session_id: str) -> bool:
        """检查指定会话的心跳是否存活.

        Args:
            session_id: 会话 ID

        Returns:
            True 表示心跳正常，False 表示已过期
        """
        session = self._bus.get_session(session_id)
        if session is None:
            return False

        last_hb = session.get("last_heartbeat", 0)
        threshold = 3 * self._interval
        return (time.time() - last_hb) <= threshold

    def get_stale_sessions(self) -> list[str]:
        """获取所有心跳过期的会话 ID.

        过期判定：当前时间 - 最后心跳时间 > 3 * 检测间隔。

        Returns:
            过期会话 ID 列表
        """
        stale: list[str] = []
        now = time.time()
        threshold = 3 * self._interval

        for sid, session in self._bus._sessions.items():
            last_hb = session.get("last_heartbeat", 0)
            if (now - last_hb) > threshold:
                stale.append(sid)

        return stale

    def reset(self) -> None:
        """重置监控器状态.

        停止监控并清理内部状态。
        """
        # 使用同步方式标记停止，不等待 asyncio 任务
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        logger.debug("心跳监控器已重置")


# ============================================================
# 导出
# ============================================================

__all__ = [
    "A2AProtocolVersion",
    "AgentIdentity",
    "HandshakeResult",
    "TaskStatus",
    "A2ATaskRecord",
    "A2AMessageBus",
    "HeartbeatMonitor",
    "create_a2a_message",
    "create_task_id",
    "create_session_id",
]
