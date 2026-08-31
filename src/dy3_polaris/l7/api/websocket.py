"""L7 API — WebSocket API (websocket.py).

任务拆分 T6 · 设计文档 Ch.9.4。

WebSocket 三通道 + 韧性机制:

| 通道 | 事件 |
|---|---|
| /ws/stream | artifact_created/updated/archived |
| /ws/broadcast | bkt_update |
| /ws/debate | debate_start/speech/convergence/end |

韧性: 心跳 30s, 重连指数退避 1→2→4→8→16→30s, 单用户最多 3 连接。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from starlette.websockets import WebSocket

#: 心跳间隔 (秒, Ch.9.8)
HEARTBEAT_INTERVAL: float = 30.0
#: 单用户最大连接数
MAX_CONNECTIONS_PER_USER: int = 3
#: 重连指数退避序列 (秒)
RECONNECT_BACKOFF: list[float] = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]

#: 通道定义
CHANNELS: dict[str, dict[str, Any]] = {
    "stream": {
        "path": "/ws/stream",
        "events": ["artifact_created", "artifact_updated", "artifact_archived"],
        "description": "Agent 产出 Artifact 实时推送",
    },
    "broadcast": {
        "path": "/ws/broadcast",
        "events": ["bkt_update"],
        "description": "L2 BKT 学情变更广播 (每次答题/交互后)",
    },
    "debate": {
        "path": "/ws/debate",
        "events": ["debate_start", "speech", "convergence", "end"],
        "description": "L4 辩论范式实时状态",
    },
}


def reconnect_delay(attempt: int) -> float:
    """指数退避重连延迟 (1→2→4→8→16→30s 上限).

    Args:
        attempt: 第 N 次重连尝试 (0 起始)。

    Returns:
        延迟秒数。
    """
    if attempt < 0:
        attempt = 0
    if attempt >= len(RECONNECT_BACKOFF):
        return RECONNECT_BACKOFF[-1]
    return RECONNECT_BACKOFF[attempt]


class ConnectionManager:
    """WebSocket 连接管理器 (单用户连接数限制 + 心跳).

    线程安全: RLock 保护连接表。

    Attributes:
        max_per_user: 单用户连接上限。
        heartbeat: 心跳间隔秒。
    """

    def __init__(
        self,
        max_per_user: int = MAX_CONNECTIONS_PER_USER,
        heartbeat: float = HEARTBEAT_INTERVAL,
    ) -> None:
        import threading

        self.max_per_user = max_per_user
        self.heartbeat = heartbeat
        self._connections: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def register(self, user_id: str, conn_id: str) -> bool:
        """注册连接, 超限拒绝.

        Returns:
            True 注册成功; False 超限。
        """
        with self._lock:
            conns = self._connections.setdefault(user_id, [])
            if len(conns) >= self.max_per_user:
                return False
            conns.append({
                "conn_id": conn_id,
                "registered_at": time.time(),
                "last_heartbeat": time.time(),
            })
            return True

    def heartbeat_tick(self, user_id: str, conn_id: str) -> bool:
        """心跳更新.

        Returns:
            True 连接存在; False 连接已断开。
        """
        with self._lock:
            for conn in self._connections.get(user_id, []):
                if conn["conn_id"] == conn_id:
                    conn["last_heartbeat"] = time.time()
                    return True
            return False

    def unregister(self, user_id: str, conn_id: str) -> None:
        """注销连接."""
        with self._lock:
            conns = self._connections.get(user_id, [])
            self._connections[user_id] = [c for c in conns if c["conn_id"] != conn_id]
            if not self._connections[user_id]:
                self._connections.pop(user_id, None)

    def count(self, user_id: str) -> int:
        """用户当前连接数."""
        with self._lock:
            return len(self._connections.get(user_id, []))

    def expired_connections(self, timeout: float | None = None) -> list[str]:
        """检测心跳超时的连接 (连接 ID 列表)."""
        timeout = timeout or self.heartbeat * 2
        now = time.time()
        expired: list[str] = []
        with self._lock:
            for conns in self._connections.values():
                for conn in conns:
                    if now - conn["last_heartbeat"] > timeout:
                        expired.append(conn["conn_id"])
        return expired


def build_message(event_type: str, payload: dict[str, Any], channel: str = "") -> dict[str, Any]:
    """构建 WebSocket 消息 (统一格式).

    Args:
        event_type: 事件类型 (如 artifact_created / bkt_update / speech)。
        payload: 事件载荷。
        channel: 所属通道。

    Returns:
        {event_type, channel, timestamp, payload}。
    """
    return {
        "event_type": event_type,
        "channel": channel,
        "timestamp": time.time(),
        "payload": payload,
    }


# ============================================================
# 三通道 WebSocket 端点 (M-F5)
# ============================================================


class WebSocketHub:
    """三通道 WebSocket 事件枢纽.

    提供:
    - 连接注册/注销 (单用户 3 连接限制)
    - 通道事件广播 (stream/broadcast/debate)
    - token 握手认证
    """

    def __init__(
        self,
        token_manager: Any | None = None,
        max_connections: int = MAX_CONNECTIONS_PER_USER,
        heartbeat: float = HEARTBEAT_INTERVAL,
    ) -> None:
        self._token_manager = token_manager
        self._connections: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.max_connections = max_connections
        self.heartbeat = heartbeat

    @property
    def token_manager(self) -> Any | None:
        """当前 token 验证器 (兼容 UnifiedApp 注入)."""
        return self._token_manager

    @token_manager.setter
    def token_manager(self, value: Any | None) -> None:
        """设置 token 验证器."""
        self._token_manager = value

    # ---------- 连接管理 ----------

    def register(self, user_id: str, conn_id: str, channel: str) -> bool:
        """注册连接, 超过上限返回 False."""
        with self._lock:
            conns = self._connections.setdefault(user_id, [])
            if len(conns) >= self.max_connections:
                return False
            conns.append({"conn_id": conn_id, "channel": channel, "last_heartbeat": time.time()})
            return True

    def unregister(self, user_id: str, conn_id: str) -> None:
        """注销连接."""
        with self._lock:
            conns = self._connections.get(user_id, [])
            self._connections[user_id] = [c for c in conns if c["conn_id"] != conn_id]
            if not self._connections[user_id]:
                self._connections.pop(user_id, None)

    def count(self, user_id: str) -> int:
        with self._lock:
            return len(self._connections.get(user_id, []))

    def heartbeat_tick(self, user_id: str, conn_id: str) -> bool:
        """心跳续期, 连接不存在返回 False."""
        with self._lock:
            for conn in self._connections.get(user_id, []):
                if conn["conn_id"] == conn_id:
                    conn["last_heartbeat"] = time.time()
                    return True
            return False

    def authenticate(self, token: str | None) -> dict[str, Any] | None:
        """验证握手 token, 返回用户信息 (user_id/role).

        支持鸭子类型验证器:
        - ``verify(token)`` 返回 dict 或含 user_id/sub 的对象
        - ``verify_token(token)`` (L1 JWTManager) 返回 TokenPayload
        """
        if not token:
            return None
        mgr = self._token_manager
        if mgr is None:
            # 回退: 内置 TokenManager (dev-secret)
            from ..api.auth import TokenManager

            mgr = TokenManager()
        try:
            if hasattr(mgr, "verify"):
                result = mgr.verify(token)
            elif hasattr(mgr, "verify_token"):
                result = mgr.verify_token(token)
            else:
                return None
        except Exception:
            return None
        if result is None:
            return None
        if isinstance(result, dict):
            return result
        # TokenPayload / 任意对象
        user_id = getattr(result, "user_id", None) or getattr(result, "sub", None)
        role = getattr(result, "role", None)
        return {"user_id": user_id, "role": role}

    # ---------- 事件广播 ----------

    def broadcast(self, channel: str, event_type: str, payload: dict[str, Any]) -> None:
        """向指定通道的所有连接广播事件 (由连接的发送队列承接).

        Args:
            channel: 通道名 (stream/broadcast/debate)
            event_type: 事件类型
            payload: 载荷
        """
        msg = build_message(event_type, payload, channel)
        for user_id, conns in list(self._connections.items()):
            for conn in conns:
                if conn["channel"] == channel:
                    queue = conn.get("queue")
                    if queue is not None:
                        try:
                            queue.put_nowait(msg)
                        except Exception:
                            pass


#: 全局枢纽实例 (由 UnifiedApp 注入 token_manager)
HUB = WebSocketHub()


def ws_authenticate(websocket: WebSocket, hub: WebSocketHub | None = None) -> dict[str, Any] | None:
    """从握手 query 解析 token 并认证.

    Args:
        websocket: Starlette WebSocket
        hub: 枢纽实例 (默认全局 HUB)

    Returns:
        claims (user_id/role) 或 None。
    """
    hub = hub or HUB
    token = websocket.query_params.get("token")
    return hub.authenticate(token)


async def _ws_loop(
    websocket: WebSocket,
    user_id: str,
    channel: str,
    conn_id: str,
    hub: WebSocketHub,
    queue: asyncio.Queue,
) -> None:
    """通道主循环: 消费队列推送事件 + 接收 ping 心跳.

    协议:
    - 客户端发送 {"type":"ping"} → 服务端心跳续期并回 {"type":"pong"}
    - 服务端事件帧: {event_type, channel, timestamp, payload}
    """
    try:
        while True:
            # 优先推送队列事件
            try:
                msg = queue.get_nowait()
                await websocket.send_json(msg)
                continue
            except asyncio.QueueEmpty:
                pass

            # 等待客户端消息 (心跳/关闭) 或队列事件
            get_task = asyncio.ensure_future(queue.get())
            recv_task = asyncio.ensure_future(websocket.receive_json())
            done, pending = await asyncio.wait({get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()

            for task in done:
                try:
                    result = task.result()
                except Exception:
                    return  # 客户端断开

                if task is get_task:
                    await websocket.send_json(result)
                else:
                    # 客户端消息
                    if isinstance(result, dict) and result.get("type") == "ping":
                        hub.heartbeat_tick(user_id, conn_id)
                        await websocket.send_json({"type": "pong", "timestamp": time.time()})
                    elif isinstance(result, dict) and result.get("type") == "close":
                        return
    finally:
        hub.unregister(user_id, conn_id)


def make_ws_endpoint(channel: str, hub: WebSocketHub | None = None) -> Callable:
    """生成指定通道的 WebSocket 端点处理器.

    Args:
        channel: 通道名 (stream/broadcast/debate)
        hub: 枢纽实例 (默认全局 HUB)

    Returns:
        async 端点函数 (websocket) -> None
    """
    hub = hub or HUB

    async def endpoint(websocket: WebSocket) -> None:
        # 1. token 认证
        claims = ws_authenticate(websocket, hub)
        if claims is None:
            await websocket.close(code=4401, reason="unauthorized")
            return
        user_id = claims.get("user_id") or claims.get("sub") or "unknown"

        # 2. 连接数限制
        conn_id = f"{channel}-{uuid4().hex[:8]}"
        if not hub.register(user_id, conn_id, channel):
            await websocket.close(code=4429, reason="too_many_connections")
            return

        # 3. 建立连接 + 事件队列
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        with hub._lock:
            for conn in hub._connections.get(user_id, []):
                if conn["conn_id"] == conn_id:
                    conn["queue"] = queue

        await _ws_loop(websocket, user_id, channel, conn_id, hub, queue)

    return endpoint


def websocket_routes(hub: WebSocketHub | None = None) -> list[Any]:
    """构建三通道路由表 (挂载于根级 /ws/*).

    Returns:
        [Route("/ws/stream", ...), Route("/ws/broadcast", ...), Route("/ws/debate", ...)]
    """
    from starlette.routing import WebSocketRoute

    return [
        WebSocketRoute("/ws/stream", make_ws_endpoint("stream", hub)),
        WebSocketRoute("/ws/broadcast", make_ws_endpoint("broadcast", hub)),
        WebSocketRoute("/ws/debate", make_ws_endpoint("debate", hub)),
    ]
