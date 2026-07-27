"""A2A (Agent-to-Agent) 协议实现.

Dy3+ Polaris L6 层的 Agent 间高层协商协议，覆盖：
- 能力发现 (Discovery)
- 身份协商 (Handshake)
- 任务分发 (Task)
- 任务取消 (Cancel)
- 心跳保活 (Heartbeat)
- 会话管理 (Session)
- 能力注册 (Capability Registry)

协议版本: a2a/1.0
与 MCP 的关系: MCP 负责工具/数据源底层调用，A2A 负责跨系统/跨域 Agent 高层互操作。

快速使用:
    from dy3_polaris.l6.a2a import A2AMessageBus, SessionManager, CapabilityRegistry

    # 创建消息总线
    bus = A2AMessageBus()

    # 注册 Agent 能力
    cap = A2ACapability(agent_id="tutor", agent_name="Tutor Agent", ...)
    bus.register_agent("tutor", cap)

    # 握手
    result = await bus.initiate_handshake("tutor", "assess", ["knowledge_assessment"])

    # 发送任务
    task = await bus.send_task("tutor", "assess", "knowledge_assessment", {...})

    # 会话管理
    session_mgr = SessionManager(bus)
    session = await session_mgr.create_session("tutor", "assess")
"""

from __future__ import annotations

from .protocol import (
    A2AProtocolVersion,
    AgentIdentity,
    HandshakeResult,
    TaskStatus,
    A2ATaskRecord,
    A2AMessageBus,
    HeartbeatMonitor,
    create_a2a_message,
    create_task_id,
    create_session_id,
)
from .session_manager import (
    SessionState,
    SessionRecord,
    SessionManager,
)
from .metrics import A2AMetrics
from .health import AgentHealthTracker
from .auth import TokenStore, agent_fingerprint, verify_fingerprint


# ============================================================
# 能力注册表（轻量，直接在此实现）
# ============================================================

from ..core.models import A2ACapability


class CapabilityRegistry:
    """A2A 能力注册表.

    集中管理所有 Agent 的能力声明，支持按能力名称、领域、
    工具等多维度查询，为 A2A 能力发现提供索引服务。

    使用示例:
        registry = CapabilityRegistry()
        registry.register(capability)
        agents = registry.find_by_capability("knowledge_assessment")
    """

    def __init__(self) -> None:
        self._agents: dict[str, A2ACapability] = {}
        # 倒排索引
        self._capability_index: dict[str, set[str]] = {}
        self._tool_index: dict[str, set[str]] = {}
        self._domain_index: dict[str, set[str]] = {}
        self._method_index: dict[str, set[str]] = {}

    def register(self, cap: A2ACapability) -> None:
        """注册 Agent 能力."""
        aid = cap.agent_id
        # 先清理旧索引
        if aid in self._agents:
            self._remove_from_indices(aid)

        self._agents[aid] = cap
        self._add_to_indices(cap)

    def unregister(self, agent_id: str) -> A2ACapability | None:
        """注销 Agent."""
        cap = self._agents.pop(agent_id, None)
        if cap:
            self._remove_from_indices(agent_id)
        return cap

    def get(self, agent_id: str) -> A2ACapability | None:
        """获取 Agent 能力."""
        return self._agents.get(agent_id)

    def find_by_capability(self, capability: str) -> list[A2ACapability]:
        """按能力名查找."""
        ids = self._capability_index.get(capability, set())
        return [self._agents[aid] for aid in ids if aid in self._agents]

    def find_by_domain(self, domain: str) -> list[A2ACapability]:
        """按领域查找."""
        ids = self._domain_index.get(domain, set())
        return [self._agents[aid] for aid in ids if aid in self._agents]

    def find_by_tool(self, tool_name: str) -> list[A2ACapability]:
        """按 MCP 工具名查找."""
        ids = self._tool_index.get(tool_name, set())
        return [self._agents[aid] for aid in ids if aid in self._agents]

    def find_by_method(self, method: str) -> list[A2ACapability]:
        """按 MCP method 查找."""
        ids = self._method_index.get(method, set())
        return [self._agents[aid] for aid in ids if aid in self._agents]

    def all_capabilities(self) -> list[str]:
        """获取所有已注册的能力名."""
        return list(self._capability_index.keys())

    def all_agents(self) -> list[A2ACapability]:
        """获取所有已注册的 Agent 能力."""
        return list(self._agents.values())

    @property
    def size(self) -> int:
        return len(self._agents)

    def export_summary(self) -> dict:
        """导出统计摘要."""
        return {
            "total_agents": self.size,
            "unique_capabilities": len(self._capability_index),
            "unique_tools": len(self._tool_index),
            "unique_domains": len(self._domain_index),
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "agent_name": a.agent_name,
                    "version": a.version,
                    "capabilities": a.supported_methods,
                    "domain_scope": a.domain_scope,
                }
                for a in self._agents.values()
            ],
        }

    def clear(self) -> None:
        """清空（仅用于测试）."""
        self._agents.clear()
        self._capability_index.clear()
        self._tool_index.clear()
        self._domain_index.clear()
        self._method_index.clear()

    def _add_to_indices(self, cap: A2ACapability) -> None:
        aid = cap.agent_id
        for m in cap.supported_methods:
            self._capability_index.setdefault(m, set()).add(aid)
            self._method_index.setdefault(m, set()).add(aid)
        for t in cap.supported_tools:
            self._tool_index.setdefault(t, set()).add(aid)
        for d in cap.domain_scope:
            self._domain_index.setdefault(d, set()).add(aid)

    def _remove_from_indices(self, agent_id: str) -> None:
        cap = self._agents.get(agent_id)
        if not cap:
            return
        for m in cap.supported_methods:
            self._capability_index.setdefault(m, set()).discard(agent_id)
            self._method_index.setdefault(m, set()).discard(agent_id)
        for t in cap.supported_tools:
            self._tool_index.setdefault(t, set()).discard(agent_id)
        for d in cap.domain_scope:
            self._domain_index.setdefault(d, set()).discard(agent_id)


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 协议引擎
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
    # 会话管理
    "SessionState",
    "SessionRecord",
    "SessionManager",
    # 能力注册
    "CapabilityRegistry",
    # 可观测性
    "A2AMetrics",
    # 健康评分
    "AgentHealthTracker",
    # 认证
    "TokenStore",
    "agent_fingerprint",
    "verify_fingerprint",
]
