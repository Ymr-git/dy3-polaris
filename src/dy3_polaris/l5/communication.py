"""通信与状态传递模块 — L5 Agent Runtime 核心组件.

融合世界先进方案:
- LangGraph: Channel + Reducer + BSP 屏障同步 + checkpoint
- OpenAI Agents SDK: Handoff 单向移交 + 全量上下文传递
- Google ADK: Session/State/Memory 三层 + output_key + 分层作用域
- AutoGen: Topic 发布-订阅 + GroupChat Manager
- Temporal: Signal/Query + 事件历史 + 确定性重放
- Kafka/RabbitMQ: 发布-订阅 + 消费确认 + 分区日志
- CrewAI: 任务上下文链 + 委派工具

本模块实现:
1. Message — 不可变消息载体 (统一消息格式, frozen dataclass)
2. MessageBus — 消息总线 (Pub/Sub 引擎 + 频道管理 + 消息历史)
3. ChannelSubscription — 频道订阅管理 (活跃/取消)
4. StatePropagator — 状态传播器 (共享状态 + 5 种 Reducer 聚合)
5. StateUpdate — 状态更新事件 (含 Reducer 类型 + 作用域 + 溯源)
6. HandoffContext — 移交上下文 (对话历史 + 状态快照)
7. HandoffManager — Agent 控制权移交 (OpenAI Handoff 单向移交)
8. MessageRouter — 消息路由 (Fork 前缀隔离 + 优先级批量路由)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class MessagePriority(str, Enum):
    """消息优先级 (Kafka 分区优先级模式).

    LOW:    低优先级 (日志/监控类消息)
    NORMAL: 默认优先级 (常规学情消息)
    HIGH:   高优先级 (干预指令/告警消息)
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ReducerType(str, Enum):
    """状态 Reducer 类型 (LangGraph Channel Reducer 模式).

    LAST_VALUE:      新值覆盖旧值 (LangGraph 默认)
    ACCUMULATE_LIST: 列表追加 (LangGraph add_messages 模式)
    SUM:             数值累加 (BinaryOperatorAggregate 模式)
    MAX:             取最大值
    MERGE_DICT:      字典合并 (深合并)
    """

    LAST_VALUE = "last_value"
    ACCUMULATE_LIST = "accumulate_list"
    SUM = "sum"
    MAX = "max"
    MERGE_DICT = "merge_dict"


class StateScope(str, Enum):
    """状态作用域 (Google ADK 前缀模式: session/user/app/temp).

    SESSION: 会话级 (会话结束后清除)
    USER:    用户级 (跨会话保留, 同一用户共享)
    APP:     应用级 (全局配置, 所有用户共享)
    TEMP:    临时级 (单次交互, 立即清除)
    """

    SESSION = "session"
    USER = "user"
    APP = "app"
    TEMP = "temp"


class HandoffState(str, Enum):
    """Handoff 状态 (OpenAI Agents SDK Handoff 生命周期).

    PENDING → IN_PROGRESS → COMPLETED
                       ↘ FAILED
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# 异常定义
# ============================================================


class CommunicationError(Exception):
    """通信错误 (频道不存在/已关闭/Reducer 不匹配等)."""

    pass


# ============================================================
# Message — 不可变消息载体
# ============================================================


@dataclass(frozen=True)
class Message:
    """不可变消息载体 (统一消息格式).

    融合 Kafka 消息模型 + Redis Streams 消费位点 + LangGraph Channel 消息.

    字段:
    - channel: 频道名称 (如 "learning.diagnosis.report")
    - publisher: 发布者 Agent ID
    - payload: 消息内容 (字典)
    - message_id: 自动生成的唯一 ID (msg-xxxxxxxxxxxx)
    - timestamp: 自动生成的时间戳
    - stream_id: Redis Streams 消费位点 (发布时自动分配)
    - priority: 消息优先级 (默认 NORMAL)
    """

    channel: str
    publisher: str
    payload: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    stream_id: str | None = None
    priority: MessagePriority = MessagePriority.NORMAL

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "message_id": self.message_id,
            "channel": self.channel,
            "publisher": self.publisher,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "stream_id": self.stream_id,
            "priority": self.priority.value
            if isinstance(self.priority, MessagePriority)
            else self.priority,
        }


# ============================================================
# ChannelSubscription — 频道订阅
# ============================================================


@dataclass
class ChannelSubscription:
    """频道订阅记录.

    字段:
    - channel: 频道名称
    - subscriber_id: 订阅者 Agent ID
    - callback: 消息回调函数 (可选, 同步或异步)
    - active: 订阅是否活跃
    """

    channel: str
    subscriber_id: str
    callback: Callable[[Message], Any] | None = None
    active: bool = True

    def unsubscribe(self) -> None:
        """取消订阅."""
        self.active = False


# ============================================================
# StateUpdate — 状态更新事件
# ============================================================


@dataclass(frozen=True)
class StateUpdate:
    """状态更新事件 (含 Reducer 类型 + 作用域 + 溯源).

    融合 Temporal 事件历史 + LangGraph Channel Reducer + ADK 分层作用域.

    字段:
    - key: 状态键
    - value: 状态值
    - source_agent: 来源 Agent ID (溯源)
    - reducer_type: Reducer 类型 (默认 LAST_VALUE)
    - scope: 作用域 (默认 SESSION)
    - timestamp: 时间戳
    """

    key: str
    value: Any
    source_agent: str = ""
    reducer_type: ReducerType = ReducerType.LAST_VALUE
    scope: StateScope = StateScope.SESSION
    timestamp: float = field(default_factory=time.time)


# ============================================================
# MessageBus — 消息总线
# ============================================================


class MessageBus:
    """消息总线 (Pub/Sub 引擎 + 频道管理 + 消息历史).

    融合 Redis Pub/Sub + Kafka 分区日志 + LangGraph Channel.

    核心能力:
    1. 频道管理: 创建/关闭/查询频道
    2. 发布-订阅: 消息发布到频道, 所有订阅者接收
    3. 消息历史: 保留消息日志 (Kafka 日志模式, 支持限制查询)
    4. Stream ID: 自动分配消费位点 (Redis Streams 模式)
    5. 异步回调: 支持同步和异步回调函数
    6. 异常隔离: 单个订阅者回调异常不影响其他订阅者
    """

    def __init__(self) -> None:
        self._channels: dict[str, list[ChannelSubscription]] = {}
        self._history: dict[str, list[Message]] = {}
        self._closed: set[str] = set()
        self._publish_lock = threading.RLock()
        self._stream_counters: dict[str, int] = {}

    @property
    def channels(self) -> dict[str, list[ChannelSubscription]]:
        """所有活跃频道."""
        return self._channels

    @property
    def closed_channels(self) -> set[str]:
        """已关闭的频道."""
        return self._closed

    def create_channel(self, channel: str) -> None:
        """创建频道.

        Raises:
            CommunicationError: 频道已存在
        """
        if channel in self._channels:
            raise CommunicationError(f"Channel '{channel}' already exists")
        self._channels[channel] = []
        self._history[channel] = []

    def subscribe(
        self,
        channel: str,
        subscriber_id: str,
        callback: Callable[[Message], Any] | None = None,
    ) -> ChannelSubscription:
        """订阅频道.

        Args:
            channel: 频道名称
            subscriber_id: 订阅者 Agent ID
            callback: 消息回调函数 (可选)

        Returns:
            ChannelSubscription 订阅记录

        Raises:
            CommunicationError: 频道不存在或已关闭
        """
        if channel not in self._channels:
            raise CommunicationError(f"Channel '{channel}' not found")
        if channel in self._closed:
            raise CommunicationError(f"Channel '{channel}' is closed")

        sub = ChannelSubscription(
            channel=channel,
            subscriber_id=subscriber_id,
            callback=callback,
        )
        self._channels[channel].append(sub)
        return sub

    def publish(self, msg: Message) -> None:
        """发布消息到频道.

        自动分配 stream_id (如果未设置), 存入消息历史, 分发给所有活跃订阅者.
        频道未注册时懒创建 (支持按需广播, 避免 ChannelNotFound).

        Raises:
            CommunicationError: 频道已关闭
        """
        if msg.channel in self._closed:
            raise CommunicationError(f"Channel '{msg.channel}' is closed")
        if msg.channel not in self._channels:
            self.create_channel(msg.channel)

        # 多线程安全: 序列化 stream_id 分配与历史写入 (多进程部署需外部锁)
        with self._publish_lock:
            # 自动分配 stream_id (Redis Streams 消费位点)
            if msg.stream_id is None:
                counter = self._stream_counters.get(msg.channel, 0)
                self._stream_counters[msg.channel] = counter + 1
                stream_id = f"{int(time.time() * 1000)}-{counter}"
                object.__setattr__(msg, "stream_id", stream_id)

            # 存入消息历史 (Kafka 日志模式)
            self._history[msg.channel].append(msg)

        # 分发给所有活跃订阅者 (异常隔离)
        for sub in list(self._channels[msg.channel]):
            if not sub.active or sub.callback is None:
                continue
            try:
                result = sub.callback(msg)
                # 处理异步回调
                if asyncio.iscoroutine(result):
                    try:
                        asyncio.ensure_future(result)
                    except RuntimeError:
                        # 没有运行中的事件循环, 关闭协程避免警告
                        result.close()
            except Exception as e:
                logger.warning(
                    "Callback error for subscriber '%s' on channel '%s': %s",
                    sub.subscriber_id,
                    msg.channel,
                    e,
                )

    def get_subscribers(self, channel: str) -> list[str]:
        """获取频道活跃订阅者列表."""
        return [
            sub.subscriber_id
            for sub in self._channels.get(channel, [])
            if sub.active
        ]

    def get_history(
        self, channel: str, limit: int | None = None
    ) -> list[Message]:
        """获取频道消息历史 (Kafka 日志模式).

        Args:
            channel: 频道名称
            limit: 返回最近 N 条消息 (None 返回全部)

        Returns:
            消息列表 (按时间顺序)
        """
        history = self._history.get(channel, [])
        if limit is not None:
            return list(history[-limit:])
        return list(history)

    def close_channel(self, channel: str) -> None:
        """关闭频道 (关闭后无法发布和订阅)."""
        if channel in self._channels:
            self._closed.add(channel)


# ============================================================
# StatePropagator — 状态传播器
# ============================================================


# 优先级排序映射 (用于 route_batch)
_PRIORITY_ORDER: dict[MessagePriority, int] = {
    MessagePriority.LOW: 0,
    MessagePriority.NORMAL: 1,
    MessagePriority.HIGH: 2,
}


class StatePropagator:
    """状态传播器 (共享状态 + Reducer 聚合 + 检查点).

    融合 LangGraph Channel Reducer + Google ADK 分层作用域 + Temporal 事件历史.

    核心能力:
    1. 状态管理: 键值对存储, 支持 5 种 Reducer 聚合策略
    2. 分层作用域: SESSION / USER / APP / TEMP (ADK 前缀模式)
    3. 事件溯源: 每次更新生成 StateUpdate 事件 (Temporal 事件历史)
    4. 检查点: 保存/恢复状态快照 (LangGraph checkpoint 模式)
    5. 作用域清除: 按作用域批量清除状态 (ADK temp: 前缀模式)
    6. 状态快照: 完整状态导出 (用于 Handoff 上下文传递)
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._updates: list[StateUpdate] = []
        self._reducers: dict[str, ReducerType] = {}
        self._scopes: dict[str, StateScope] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._cp_reducers: dict[str, dict[str, ReducerType]] = {}
        self._cp_scopes: dict[str, dict[str, StateScope]] = {}

    @property
    def state(self) -> dict[str, Any]:
        """当前状态字典."""
        return self._state

    @property
    def updates(self) -> list[StateUpdate]:
        """状态更新历史."""
        return self._updates

    def set(
        self,
        key: str,
        value: Any,
        source_agent: str = "",
        reducer: ReducerType = ReducerType.LAST_VALUE,
        scope: StateScope = StateScope.SESSION,
    ) -> None:
        """设置状态 (应用 Reducer 聚合).

        Args:
            key: 状态键
            value: 新值
            source_agent: 来源 Agent ID (溯源)
            reducer: Reducer 类型
            scope: 作用域

        Raises:
            CommunicationError: Reducer 类型不匹配
        """
        if key in self._state:
            existing_reducer = self._reducers.get(key, ReducerType.LAST_VALUE)

            # 检查 Reducer 类型一致性
            if reducer != existing_reducer:
                raise CommunicationError(
                    f"reducer type mismatch for key '{key}': "
                    f"expected {existing_reducer.value}, got {reducer.value}"
                )

            # 应用 Reducer 聚合
            if reducer == ReducerType.LAST_VALUE:
                self._state[key] = value

            elif reducer == ReducerType.ACCUMULATE_LIST:
                if not isinstance(self._state[key], list):
                    self._state[key] = [self._state[key]]
                if isinstance(value, list):
                    self._state[key] = self._state[key] + list(value)
                else:
                    self._state[key] = self._state[key] + [value]

            elif reducer == ReducerType.SUM:
                if not isinstance(value, (int, float)) or not isinstance(
                    self._state[key], (int, float)
                ):
                    raise CommunicationError(
                        f"reducer type mismatch: SUM requires numeric values, "
                        f"got {type(value).__name__}"
                    )
                self._state[key] = self._state[key] + value

            elif reducer == ReducerType.MAX:
                if not isinstance(value, (int, float)) or not isinstance(
                    self._state[key], (int, float)
                ):
                    raise CommunicationError(
                        f"reducer type mismatch: MAX requires numeric values"
                    )
                self._state[key] = max(self._state[key], value)

            elif reducer == ReducerType.MERGE_DICT:
                if not isinstance(self._state[key], dict) or not isinstance(
                    value, dict
                ):
                    raise CommunicationError(
                        f"reducer type mismatch: MERGE_DICT requires dict values"
                    )
                self._state[key] = {**self._state[key], **value}
        else:
            # 首次设置: 直接赋值
            self._state[key] = value

        self._reducers[key] = reducer
        self._scopes[key] = scope

        # 记录状态更新事件 (Temporal 事件历史)
        update = StateUpdate(
            key=key,
            value=value,
            source_agent=source_agent,
            reducer_type=reducer,
            scope=scope,
        )
        self._updates.append(update)

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值."""
        return self._state.get(key, default)

    def get_updates(self) -> list[StateUpdate]:
        """获取状态更新历史 (Temporal 事件重放)."""
        return list(self._updates)

    def checkpoint(self) -> str:
        """创建状态检查点 (LangGraph checkpoint 模式).

        Returns:
            检查点 ID (state-cp-xxxxxxxxxxxx)
        """
        cp_id = f"state-cp-{uuid.uuid4().hex[:12]}"
        self._checkpoints[cp_id] = dict(self._state)
        self._cp_reducers[cp_id] = dict(self._reducers)
        self._cp_scopes[cp_id] = dict(self._scopes)
        return cp_id

    def restore(self, cp_id: str) -> bool:
        """恢复状态检查点.

        Args:
            cp_id: 检查点 ID

        Returns:
            True 如果检查点存在并恢复成功, False 如果检查点不存在
        """
        if cp_id not in self._checkpoints:
            return False
        self._state = dict(self._checkpoints[cp_id])
        self._reducers = dict(self._cp_reducers.get(cp_id, {}))
        self._scopes = dict(self._cp_scopes.get(cp_id, {}))
        return True

    def clear_scope(self, scope: StateScope) -> None:
        """清除指定作用域的所有状态 (ADK temp: 前缀模式)."""
        keys_to_remove = [
            k for k, s in self._scopes.items() if s == scope
        ]
        for k in keys_to_remove:
            self._state.pop(k, None)
            self._reducers.pop(k, None)
            self._scopes.pop(k, None)

    def snapshot(self) -> dict[str, Any]:
        """导出完整状态快照 (用于 Handoff 上下文传递)."""
        return dict(self._state)


# ============================================================
# HandoffContext — 移交上下文
# ============================================================


@dataclass
class HandoffContext:
    """Agent 控制权移交上下文 (OpenAI Agents SDK Handoff 模式).

    融合 OpenAI Handoff 全量上下文传递 + ADK 状态快照.

    字段:
    - from_agent: 移交方 Agent ID
    - to_agent: 接收方 Agent ID
    - conversation_history: 对话历史 (全量传递)
    - state_snapshot: 状态快照 (StatePropagator.snapshot())
    - state: Handoff 状态 (PENDING → IN_PROGRESS → COMPLETED/FAILED)
    """

    from_agent: str
    to_agent: str
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    state: HandoffState = HandoffState.PENDING


# ============================================================
# HandoffManager — Agent 控制权移交管理器
# ============================================================


class HandoffManager:
    """Agent 控制权移交管理器 (OpenAI Agents SDK Handoff 模式).

    核心能力:
    1. Agent 注册: 注册目标 Agent 的处理函数
    2. Handoff 执行: 单向移交控制权, 传递完整上下文
    3. 状态转换: PENDING → IN_PROGRESS → COMPLETED/FAILED
    4. Handoff 链: 支持多级移交 (A → B → C)
    5. 异常处理: 处理器异常标记为 FAILED 并传播
    """

    def __init__(self) -> None:
        self._agents: dict[
            str, Callable[[HandoffContext], Awaitable[dict[str, Any]]]
        ] = {}

    def register_agent(
        self,
        agent_id: str,
        handler: Callable[[HandoffContext], Awaitable[dict[str, Any]]],
    ) -> None:
        """注册目标 Agent 的处理函数.

        Args:
            agent_id: Agent ID
            handler: 异步处理函数, 接收 HandoffContext, 返回结果字典
        """
        self._agents[agent_id] = handler

    async def execute_handoff(self, ctx: HandoffContext) -> dict[str, Any]:
        """执行 Handoff (单向移交控制权).

        Args:
            ctx: 移交上下文

        Returns:
            目标 Agent 处理结果

        Raises:
            CommunicationError: 目标 Agent 未注册
            Exception: 处理器异常 (状态标记为 FAILED)
        """
        if ctx.to_agent not in self._agents:
            raise CommunicationError(
                f"Agent '{ctx.to_agent}' is not registered"
            )

        ctx.state = HandoffState.IN_PROGRESS
        handler = self._agents[ctx.to_agent]

        try:
            result = await handler(ctx)
            ctx.state = HandoffState.COMPLETED
            return result
        except Exception:
            ctx.state = HandoffState.FAILED
            raise


# ============================================================
# MessageRouter — 消息路由器
# ============================================================


class MessageRouter:
    """消息路由器 (Fork 前缀隔离 + 优先级批量路由).

    融合 LangGraph Channel 路由 + Fork 前缀隔离 + Kafka 分区优先级.

    核心能力:
    1. 路由注册: 注册频道-订阅者-回调映射
    2. 消息路由: 将消息路由到频道订阅者
    3. Fork 前缀隔离: fork.xxx.* 消息不路由到主会话订阅者
    4. 优先级批量路由: route_batch 按优先级排序后分发
    5. 异常隔离: 单个回调异常不影响其他订阅者
    6. 路由注销: 注销后不再收到消息
    """

    def __init__(self) -> None:
        self._routes: dict[str, dict[str, Callable[[Message], Any] | None]] = {}

    def register_route(
        self,
        channel: str,
        subscriber_id: str,
        callback: Callable[[Message], Any] | None = None,
    ) -> None:
        """注册路由 (频道-订阅者-回调).

        Args:
            channel: 频道名称 (支持 Fork 前缀: fork.f123.learning.*)
            subscriber_id: 订阅者 Agent ID
            callback: 消息回调函数 (可选)
        """
        if channel not in self._routes:
            self._routes[channel] = {}
        self._routes[channel][subscriber_id] = callback

    def unregister_route(self, channel: str, subscriber_id: str) -> None:
        """注销路由."""
        if channel in self._routes:
            self._routes[channel].pop(subscriber_id, None)
            if not self._routes[channel]:
                del self._routes[channel]

    def route(self, msg: Message) -> None:
        """路由消息到频道订阅者 (同步分发).

        无订阅者的消息静默丢弃 (不报错).
        单个回调异常不影响其他订阅者 (异常隔离).
        """
        subscribers = self._routes.get(msg.channel, {})
        for sub_id, callback in list(subscribers.items()):
            if callback is None:
                continue
            try:
                result = callback(msg)
                if asyncio.iscoroutine(result):
                    try:
                        asyncio.ensure_future(result)
                    except RuntimeError:
                        result.close()
            except Exception as e:
                logger.warning(
                    "Route callback error for subscriber '%s' on channel '%s': %s",
                    sub_id,
                    msg.channel,
                    e,
                )

    def route_batch(self, messages: list[Message]) -> None:
        """批量路由消息 (按优先级排序后分发, Kafka 分区优先级模式).

        高优先级消息先路由, 同优先级按原始顺序.

        Args:
            messages: 待路由的消息列表
        """
        sorted_msgs = sorted(
            messages,
            key=lambda m: _PRIORITY_ORDER.get(m.priority, 1),
            reverse=True,
        )
        for msg in sorted_msgs:
            self.route(msg)

    def list_channels(self) -> list[str]:
        """列出所有已注册的频道."""
        return list(self._routes.keys())
