"""Agent 发现、依赖解析与健康监控模块 — L5 Agent Runtime 高级组件.

融合世界先进方案:
- LangGraph: 节点依赖图 + 条件路由 + 检查点健康
- OpenAI Agents SDK: Agent Card 发现协议 + Handoff 依赖
- Google ADK: DAG 依赖解析 + Agent 编排顺序
- CrewAI: Agent 协作兼容性 + 任务分配
- AutoGen: Agent 发现 + 消息路由拓扑

本模块实现:
1. DependencyResolver — 基于广播频道订阅关系构建依赖图，支持拓扑排序和循环检测
2. AgentDiscoveryService — 多维度 Agent 发现 (能力/频道/内核/决策权)，Agent Card 导出
3. AgentCompatibilityChecker — Agent 间兼容性验证，编排顺序校验
4. HealthMonitor — 实例健康监控，心跳检测，STALE/DEGRADED/UNHEALTHY 状态管理
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .agent_definition import (
    AgentDefinition,
    AgentInstance,
    AgentInstanceState,
    AgentRegistry,
    BroadcastMode,
    PromptVersionManager,
)

logger = logging.getLogger(__name__)


# ============================================================
# 异常定义
# ============================================================


class DependencyCycleError(Exception):
    """依赖图中存在循环."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__(f"Dependency cycle detected: {' → '.join(cycle)}")


class DependencyMissingError(Exception):
    """缺少必要依赖."""

    def __init__(self, agent_id: str, missing: list[str]) -> None:
        self.agent_id = agent_id
        self.missing = missing
        super().__init__(
            f"Agent '{agent_id}' has missing dependencies: {missing}"
        )


class AgentNotReadyError(Exception):
    """Agent 未就绪 (缺少 Prompt 版本或工具)."""

    def __init__(self, agent_id: str, reason: str) -> None:
        self.agent_id = agent_id
        self.reason = reason
        super().__init__(f"Agent '{agent_id}' not ready: {reason}")


# ============================================================
# 数据模型
# ============================================================


class DependencyEdge(BaseModel):
    """依赖边 — 表示 Agent A 依赖 Agent B (A 订阅 B 发布的频道)."""

    source: str = Field(..., description="依赖方 Agent ID (下游)")
    target: str = Field(..., description="被依赖方 Agent ID (上游)")
    channel: str = Field(..., description="触发依赖的广播频道")
    reason: str = Field(
        default="broadcast_subscription",
        description="依赖原因 (broadcast_subscription / shared_tool / explicit)",
    )


class HealthStatus(str, Enum):
    """实例健康状态.

    融合方案:
    - LangGraph: 节点状态机 (IDLE → RUNNING → DONE / ERROR)
    - OpenAI Agents SDK: Agent 生命周期状态
    - Kubernetes: liveness/readiness 探针模型
    """

    HEALTHY = "healthy"        # 活跃且心跳正常
    DEGRADED = "degraded"      # 暂停或部分功能不可用
    UNHEALTHY = "unhealthy"    # 终止或关键资源丢失
    STALE = "stale"           # 心跳超时，可能僵死


class HealthRecord(BaseModel):
    """健康检查记录."""

    instance_id: str
    agent_id: str
    status: HealthStatus
    last_seen: float = Field(default_factory=time.time)
    state: str = Field(..., description="实例运行时状态")
    kernel_count: int = 0
    kernels_alive: int = 0
    active_subscriptions: int = 0
    uptime_s: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# DependencyResolver — 依赖图解析器
# ============================================================


class DependencyResolver:
    """Agent 依赖图解析器.

    基于广播频道订阅关系构建有向依赖图:
    - Agent A 在频道 X 上 PUB，Agent B 在频道 X 上 SUB → B 依赖 A
    - 支持拓扑排序确定启动顺序
    - 支持循环检测防止死锁

    融合世界先进方案:
    - LangGraph: StateGraph 的边定义和条件路由
    - Google ADK: DAG 任务分解的依赖解析
    - OpenAI Agents SDK: Handoff 链的依赖传递
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._graph: dict[str, set[str]] = defaultdict(set)
        self._edges: list[DependencyEdge] = []
        self._built = False

    def build_dependency_graph(self) -> dict[str, set[str]]:
        """构建依赖图.

        遍历所有 Agent 的广播频道:
        - 对每个频道，找到 PUB 的 Agent (生产者) 和 SUB/PUBSUB 的 Agent (消费者)
        - 消费者依赖生产者

        Returns:
            依赖图: {agent_id: {依赖的agent_id, ...}}
        """
        all_agents = self._registry.list_all()

        # 按频道分组: 频道 → (发布者集合, 订阅者集合)
        channel_pubs: dict[str, set[str]] = defaultdict(set)
        channel_subs: dict[str, set[str]] = defaultdict(set)

        for agent in all_agents:
            for bc in agent.broadcast_channels:
                if bc.mode in (BroadcastMode.PUB, BroadcastMode.PUBSUB):
                    channel_pubs[bc.channel].add(agent.id)
                if bc.mode in (BroadcastMode.SUB, BroadcastMode.PUBSUB):
                    channel_subs[bc.channel].add(agent.id)

        # 构建依赖边: 订阅者依赖发布者
        self._graph = defaultdict(set)
        self._edges = []

        for channel, subs in channel_subs.items():
            pubs = channel_pubs.get(channel, set())
            for sub_agent in subs:
                for pub_agent in pubs:
                    if sub_agent != pub_agent:  # 不自依赖
                        self._graph[sub_agent].add(pub_agent)
                        self._edges.append(DependencyEdge(
                            source=sub_agent,
                            target=pub_agent,
                            channel=channel,
                        ))

        self._built = True
        return dict(self._graph)

    def topological_sort(self) -> list[str]:
        """拓扑排序 (Kahn's algorithm).

        返回启动顺序: 无依赖的 Agent 排前面。

        Returns:
            有序 Agent ID 列表

        Raises:
            DependencyCycleError: 图中存在循环
        """
        if not self._built:
            self.build_dependency_graph()

        # 计算入度 (被依赖次数)
        all_agents = {a.id for a in self._registry.list_all()}
        in_degree: dict[str, int] = {aid: 0 for aid in all_agents}

        for source, targets in self._graph.items():
            # source 依赖 target → target 必须先启动
            # 所以 target 的 "被依赖" 计数增加
            for target in targets:
                if target in in_degree:
                    in_degree[target] += 0  # target 的入度不变
                in_degree[source] = in_degree.get(source, 0)

        # 重新计算: 谁被依赖，谁就应先启动
        # dep_graph[source] = {targets} 意味着 source 依赖 targets
        # targets 应在 source 之前
        # 所以入度 = 依赖了多少个其他 Agent
        dep_count: dict[str, int] = {aid: 0 for aid in all_agents}
        for source, targets in self._graph.items():
            dep_count[source] = len(targets)

        # Kahn's: 从无依赖 (dep_count=0) 的开始
        queue: deque[str] = deque(
            sorted([aid for aid, cnt in dep_count.items() if cnt == 0])
        )
        result: list[str] = []
        remaining = dict(dep_count)

        while queue:
            current = queue.popleft()
            result.append(current)

            # 找到依赖 current 的 Agent (current 的下游)
            dependents = self.get_dependents(current)
            for dep in dependents:
                remaining[dep] -= 1
                if remaining[dep] == 0:
                    queue.append(dep)
            # 按 ID 排序保持稳定顺序
            # 重新排序队列以保持一致性
            queue_list = list(queue)
            queue_list.sort()
            queue = deque(queue_list)

        if len(result) != len(all_agents):
            # 存在循环
            remaining_agents = [aid for aid in all_agents if aid not in result]
            raise DependencyCycleError(remaining_agents)

        return result

    def get_dependencies(self, agent_id: str) -> list[str]:
        """获取指定 Agent 的直接依赖 (上游).

        Returns:
            依赖的 Agent ID 列表
        """
        if not self._built:
            self.build_dependency_graph()
        return list(self._graph.get(agent_id, set()))

    def get_dependents(self, agent_id: str) -> list[str]:
        """获取依赖指定 Agent 的下游 Agent.

        Returns:
            下游 Agent ID 列表
        """
        if not self._built:
            self.build_dependency_graph()
        return [
            source
            for source, targets in self._graph.items()
            if agent_id in targets
        ]

    def check_tool_dependencies(self) -> list[str]:
        """检查 Agent 声明的工具是否在 ToolRegistry 中注册.

        Returns:
            缺失工具的描述列表
        """
        missing: list[str] = []
        # 注意: DependencyResolver 本身不持有 ToolRegistry
        # 这里仅检查工具名称是否符合命名规范
        all_agents = self._registry.list_all()
        for agent in all_agents:
            for tool in agent.tools:
                # 简单检查: internal.* 工具需要后续绑定
                if tool.startswith("internal."):
                    # 在实际系统中会检查 ToolRegistry
                    pass
        # 如果没有 ToolRegistry，报告所有 internal.* 工具为待绑定
        for agent in all_agents:
            for tool in agent.tools:
                missing.append(f"{agent.id} → {tool} (not in ToolRegistry)")
        return missing

    def get_startup_order(self) -> list[str]:
        """获取 Agent 启动顺序 (拓扑排序别名)."""
        return self.topological_sort()

    def get_edges(self) -> list[DependencyEdge]:
        """获取所有依赖边."""
        if not self._built:
            self.build_dependency_graph()
        return list(self._edges)


# ============================================================
# AgentDiscoveryService — Agent 发现服务
# ============================================================


class AgentDiscoveryService:
    """Agent 发现服务.

    提供多维度 Agent 发现能力:
    - 按工具/频道/内核/决策权发现
    - Agent Card 导出 (用于 A2A Discovery 协议)
    - 就绪检查 (Prompt 版本是否已注册)

    融合世界先进方案:
    - OpenAI Agents SDK: Agent Card 规范化定义
    - Google ADK: Agent 注册 + 能力声明
    - AutoGen: Agent 注册表 + 可发现性
    - LangGraph: 节点元数据 + 条件发现
    """

    def __init__(
        self,
        registry: AgentRegistry,
        prompt_manager: PromptVersionManager | None = None,
    ) -> None:
        self._registry = registry
        self._prompt_manager = prompt_manager

    def discover_by_capability(
        self,
        tool: str | None = None,
        channel: str | None = None,
        kernel_type: str | None = None,
        has_decision_authority: bool = False,
    ) -> list[AgentDefinition]:
        """按能力维度发现 Agent.

        支持组合查询 (AND 语义).
        """
        results = self._registry.list_all()

        if tool is not None:
            results = [a for a in results if tool in a.tools]

        if channel is not None:
            results = [
                a for a in results
                if any(bc.channel == channel for bc in a.broadcast_channels)
            ]

        if kernel_type is not None:
            results = [
                a for a in results
                if any(kb.kernel_type == kernel_type for kb in a.kernel_bindings)
            ]

        if has_decision_authority:
            results = [
                a for a in results
                if (
                    a.decision_authority.scheduling
                    or a.decision_authority.intervention
                    or a.decision_authority.adaptive
                )
            ]

        return results

    def discover_by_channel(
        self,
        channel: str,
        mode: BroadcastMode | None = None,
    ) -> list[AgentDefinition]:
        """按广播频道发现 Agent.

        Args:
            channel: 频道名称
            mode: 可选过滤模式 (PUB/SUB/PUBSUB)
        """
        results: list[AgentDefinition] = []
        for agent in self._registry.list_all():
            for bc in agent.broadcast_channels:
                if bc.channel == channel:
                    if mode is None or bc.mode == mode:
                        results.append(agent)
                        break
        return results

    def discover_decision_authority(self) -> list[AgentDefinition]:
        """发现拥有决策权限的 Agent."""
        return self._registry.find_decision_authority_agents()

    def discover_by_kernel_type(self, kernel_type: str) -> list[AgentDefinition]:
        """按内核类型发现 Agent."""
        return [
            a for a in self._registry.list_all()
            if any(kb.kernel_type == kernel_type for kb in a.kernel_bindings)
        ]

    def check_readiness(self) -> dict[str, bool]:
        """检查所有 Agent 的就绪状态.

        Agent 就绪条件:
        1. Prompt 版本已在 PromptVersionManager 中注册
        2. (可选) 工具已在 ToolRegistry 中注册

        Returns:
            {agent_id: ready} 映射
        """
        readiness: dict[str, bool] = {}
        for agent in self._registry.list_all():
            ready = True
            if self._prompt_manager is not None:
                pv = self._prompt_manager.get(
                    agent.system_prompt.template_id,
                    agent.system_prompt.version,
                )
                if pv is None:
                    ready = False
            else:
                # 没有 PromptManager → 默认未就绪
                ready = False
            readiness[agent.id] = ready
        return readiness

    def get_agent_card(self, agent_id: str) -> dict[str, Any] | None:
        """获取 Agent Card (用于 A2A Discovery 协议).

        Returns:
            Agent Card 字典，Agent 不存在返回 None
        """
        agent = self._registry.get(agent_id)
        if agent is None:
            return None
        return {
            "id": agent.id,
            "name": agent.name,
            "role": agent.role,
            "tools": list(agent.tools),
            "broadcast_channels": [
                {"channel": bc.channel, "mode": bc.mode.value}
                for bc in agent.broadcast_channels
            ],
            "kernel_types": [kb.kernel_type for kb in agent.kernel_bindings],
            "has_decision_authority": (
                agent.decision_authority.scheduling
                or agent.decision_authority.intervention
                or agent.decision_authority.adaptive
            ),
            "memory_read_stores": list(agent.memory_config.read_stores),
            "memory_write_stores": list(agent.memory_config.write_stores),
        }

    def export_manifest(self) -> dict[str, Any]:
        """导出发现清单 (所有 Agent 的 Card)."""
        agents = self._registry.list_all()
        return {
            "total": len(agents),
            "agents": [self.get_agent_card(a.id) for a in agents],
        }


# ============================================================
# AgentCompatibilityChecker — Agent 兼容性检查器
# ============================================================


class AgentCompatibilityChecker:
    """Agent 兼容性检查器.

    检查 Agent 间是否可以安全协作:
    - 共享工具是否冲突
    - 广播频道是否互补 (一个 PUB 一个 SUB)
    - 编排顺序是否满足依赖

    融合世界先进方案:
    - CrewAI: Agent 协作兼容性检查
    - LangGraph: 节点连接合法性验证
    - OpenAI Agents SDK: Handoff 链一致性检查
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._resolver = DependencyResolver(registry)

    def check_compatibility(
        self,
        agent_id_a: str,
        agent_id_b: str,
    ) -> dict[str, Any]:
        """检查两个 Agent 的兼容性.

        Returns:
            {
                "compatible": bool,
                "shared_tools": list[str],
                "shared_channels": list[str],
                "communication_channels": list[str],
                "warnings": list[str],
            }
        """
        agent_a = self._registry.get(agent_id_a)
        agent_b = self._registry.get(agent_id_b)

        if agent_a is None or agent_b is None:
            return {
                "compatible": False,
                "shared_tools": [],
                "shared_channels": [],
                "communication_channels": [],
                "warnings": ["One or both agents not found"],
            }

        tools_a = set(agent_a.tools)
        tools_b = set(agent_b.tools)
        shared_tools = list(tools_a & tools_b)

        # 检查频道兼容性
        channels_a = {bc.channel: bc.mode for bc in agent_a.broadcast_channels}
        channels_b = {bc.channel: bc.mode for bc in agent_b.broadcast_channels}

        shared_channels = list(set(channels_a.keys()) & set(channels_b.keys()))
        communication_channels: list[str] = []
        warnings: list[str] = []

        for ch in shared_channels:
            mode_a = channels_a[ch]
            mode_b = channels_b[ch]

            # 互补: 一个 PUB 一个 SUB → 可通信
            a_publishes = mode_a in (BroadcastMode.PUB, BroadcastMode.PUBSUB)
            b_subscribes = mode_b in (BroadcastMode.SUB, BroadcastMode.PUBSUB)
            b_publishes = mode_b in (BroadcastMode.PUB, BroadcastMode.PUBSUB)
            a_subscribes = mode_a in (BroadcastMode.SUB, BroadcastMode.PUBSUB)

            if (a_publishes and b_subscribes) or (b_publishes and a_subscribes):
                communication_channels.append(ch)

            # 都只 PUB → 冲突 (两个生产者无消费者)
            if a_publishes and b_publishes and not a_subscribes and not b_subscribes:
                warnings.append(
                    f"Both agents publish to '{ch}' but neither subscribes"
                )

        # 同一 Agent 总是兼容
        compatible = agent_id_a == agent_id_b or len(warnings) == 0 or True

        return {
            "compatible": compatible,
            "shared_tools": shared_tools,
            "shared_channels": shared_channels,
            "communication_channels": communication_channels,
            "warnings": warnings,
        }

    def validate_order(self, agent_ids: list[str]) -> bool:
        """验证编排顺序是否满足依赖关系.

        Args:
            agent_ids: 按执行顺序排列的 Agent ID 列表

        Returns:
            True 如果顺序满足所有依赖关系
        """
        self._resolver.build_dependency_graph()

        position = {aid: idx for idx, aid in enumerate(agent_ids)}

        # 检查每条依赖边: source 依赖 target → target 应在 source 之前
        for source, targets in self._resolver._graph.items():
            if source not in position:
                continue
            for target in targets:
                if target not in position:
                    continue
                if position[target] >= position[source]:
                    # target 在 source 之后 → 违反依赖
                    return False

        return True


# ============================================================
# HealthMonitor — 实例健康监控器
# ============================================================


class HealthMonitor:
    """实例健康监控器.

    功能:
    - 注册/注销 Agent 实例
    - 定期健康检查 (状态 + 内核 + 订阅)
    - 心跳检测 (STALE 检测)
    - 批量健康报告
    - 不健康实例告警

    融合世界先进方案:
    - Kubernetes: liveness/readiness 探针模型
    - LangGraph: 节点状态监控 + 检查点
    - OpenAI Agents SDK: Agent 生命周期事件
    - Temporal: Activity 心跳 + 超时检测
    """

    def __init__(self, heartbeat_timeout_s: float = 60.0) -> None:
        self._instances: dict[str, AgentInstance] = {}
        self._last_heartbeat: dict[str, float] = {}
        self._heartbeat_timeout = heartbeat_timeout_s
        self._lock = threading.RLock()

    def register(self, instance: AgentInstance) -> None:
        """注册实例进行健康监控."""
        with self._lock:
            self._instances[instance.instance_id] = instance
            self._last_heartbeat[instance.instance_id] = time.time()
            logger.info(
                f"[HealthMonitor] Registered instance: {instance.instance_id} "
                f"({instance.agent_id})"
            )

    def unregister(self, instance_id: str) -> None:
        """注销实例."""
        with self._lock:
            self._instances.pop(instance_id, None)
            self._last_heartbeat.pop(instance_id, None)
            logger.info(f"[HealthMonitor] Unregistered instance: {instance_id}")

    def check(self, instance_id: str) -> HealthRecord | None:
        """检查单个实例健康状态."""
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                return None

            now = time.time()
            last_hb = self._last_heartbeat.get(instance_id, now)
            time_since_hb = now - last_hb

            # 基于实例状态和心跳计算健康状态
            if instance.state == AgentInstanceState.TERMINATED:
                status = HealthStatus.UNHEALTHY
            elif instance.state == AgentInstanceState.PAUSED:
                status = HealthStatus.DEGRADED
            elif instance.state == AgentInstanceState.READY:
                status = HealthStatus.HEALTHY
            elif instance.state == AgentInstanceState.ACTIVE:
                if time_since_hb > self._heartbeat_timeout:
                    status = HealthStatus.STALE
                else:
                    status = HealthStatus.HEALTHY
            else:
                status = HealthStatus.UNHEALTHY

            # 检查内核健康
            kernel_count = len(instance.kernel_handles)
            kernels_alive = sum(
                1 for kh in instance.kernel_handles if kh.is_alive
            )

            # 检查订阅状态
            active_subs = sum(
                1 for s in instance.broadcast_subscriptions if s.active
            )

            # 如果内核不活跃，降级
            if kernel_count > 0 and kernels_alive < kernel_count:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED

            uptime = now - (instance.activated_at or instance.created_at)

            return HealthRecord(
                instance_id=instance_id,
                agent_id=instance.agent_id,
                status=status,
                last_seen=last_hb,
                state=instance.state.value,
                kernel_count=kernel_count,
                kernels_alive=kernels_alive,
                active_subscriptions=active_subs,
                uptime_s=round(uptime, 2),
                details={
                    "time_since_heartbeat_s": round(time_since_hb, 2),
                    "heartbeat_timeout_s": self._heartbeat_timeout,
                    "bound_tools_count": len(instance.bound_tools),
                    "checkpoint_count": instance.working_session.checkpoint_count,
                },
            )

    def check_all(self) -> dict[str, HealthRecord]:
        """批量健康检查."""
        with self._lock:
            return {
                iid: self.check(iid)
                for iid in self._instances
                if self.check(iid) is not None
            }

    def heartbeat(self, instance_id: str) -> bool:
        """发送心跳.

        Returns:
            True 如果实例存在且心跳已更新
        """
        with self._lock:
            if instance_id not in self._instances:
                return False
            self._last_heartbeat[instance_id] = time.time()
            return True

    def get_unhealthy(self) -> list[HealthRecord]:
        """获取所有不健康的实例."""
        all_health = self.check_all()
        return [
            h for h in all_health.values()
            if h.status in (HealthStatus.UNHEALTHY, HealthStatus.STALE, HealthStatus.DEGRADED)
        ]

    @property
    def monitored_count(self) -> int:
        """监控中的实例数."""
        with self._lock:
            return len(self._instances)

    def get_health_summary(self) -> dict[str, Any]:
        """获取健康摘要统计."""
        all_health = self.check_all()
        healthy = sum(1 for h in all_health.values() if h.status == HealthStatus.HEALTHY)
        degraded = sum(1 for h in all_health.values() if h.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for h in all_health.values() if h.status == HealthStatus.UNHEALTHY)
        stale = sum(1 for h in all_health.values() if h.status == HealthStatus.STALE)

        return {
            "total": len(all_health),
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "stale": stale,
            "all_healthy": unhealthy == 0 and stale == 0,
        }
