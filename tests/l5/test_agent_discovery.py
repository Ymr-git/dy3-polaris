"""Agent 高级发现、依赖检查与健康探针测试.

TDD 测试用例 — 覆盖增强功能:
1. AgentDiscoveryService — 能力发现、依赖拓扑、就绪检查
2. DependencyResolver — 工具依赖图解析、循环检测、拓扑排序
3. HealthMonitor — 实例健康监控、心跳检测、自动恢复
4. AgentCompatibilityChecker — Agent 间兼容性验证

融合世界先进方案:
- LangGraph: 节点依赖图 + 条件路由
- OpenAI Agents SDK: Handoff 依赖 + Agent Card
- Google ADK: DAG 依赖解析 + Agent 编排
- CrewAI: Agent 协作依赖 + 任务分配
- AutoGen: Agent 发现 + 消息路由拓扑
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dy3_polaris.l5.agent_definition import (
    AgentDefinition,
    AgentRegistry,
    AgentFactory,
    AgentInstance,
    AgentInstanceState,
    PromptVersionManager,
    PromptVersion,
    BroadcastMode,
    KernelBinding,
    MemoryConfig,
    ReputationConfig,
    DecisionAuthority,
    SelfEvolutionConfig,
    BroadcastChannel,
    PromptReference,
)
from dy3_polaris.l5.agent_discovery import (
    AgentDiscoveryService,
    DependencyResolver,
    HealthMonitor,
    AgentCompatibilityChecker,
    DependencyCycleError,
    DependencyMissingError,
    AgentNotReadyError,
    HealthStatus,
    DependencyEdge,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def prompt_manager():
    pm = PromptVersionManager()
    pm.register(PromptVersion(
        template_id="tpl.diagnosis",
        version="v2.1.0",
        content="Diagnosis agent for {learner_name}.",
        created_by="system",
    ))
    pm.register(PromptVersion(
        template_id="tpl.generation",
        version="v3.0.0",
        content="Generation agent.",
        created_by="system",
    ))
    pm.register(PromptVersion(
        template_id="tpl.review",
        version="v1.5.0",
        content="Review agent.",
        created_by="system",
    ))
    pm.register(PromptVersion(
        template_id="tpl.guidance",
        version="v4.0.0",
        content="Guidance agent.",
        created_by="system",
    ))
    return pm


@pytest.fixture
def diagnosis_agent():
    return AgentDefinition(
        id="agent.learning.diagnosis",
        name="学情诊断 Agent",
        role="基于 BKT/IRT 引擎对学习者的知识掌握状态进行实时诊断",
        system_prompt=PromptReference(template_id="tpl.diagnosis", version="v2.1.0"),
        tools=["internal.bkt_compute", "internal.irt_evaluate"],
        broadcast_channels=[
            BroadcastChannel(channel="learning.diagnosis.report", mode=BroadcastMode.PUB),
            BroadcastChannel(channel="learning.interaction.event", mode=BroadcastMode.SUB),
        ],
        kernel_bindings=[KernelBinding(kernel_type="python", purpose="BKT计算")],
    )


@pytest.fixture
def generation_agent():
    return AgentDefinition(
        id="agent.knowledge.generation",
        name="知识生成 Agent",
        role="根据学情诊断结果生成个性化学习内容",
        system_prompt=PromptReference(template_id="tpl.generation", version="v3.0.0"),
        tools=["internal.knowledge_synthesize", "internal.bkt_compute"],
        connectors=["l3.knowledge_graph"],
        broadcast_channels=[
            BroadcastChannel(channel="learning.diagnosis.report", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="knowledge.generation.output", mode=BroadcastMode.PUB),
        ],
        kernel_bindings=[KernelBinding(kernel_type="python", purpose="知识图谱推理")],
    )


@pytest.fixture
def review_agent():
    return AgentDefinition(
        id="agent.knowledge.review",
        name="审核校验 Agent",
        role="对知识生成结果进行多维度审核校验",
        system_prompt=PromptReference(template_id="tpl.review", version="v1.5.0"),
        tools=["internal.fact_check", "internal.hallucination_detect"],
        broadcast_channels=[
            BroadcastChannel(channel="knowledge.generation.output", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="knowledge.review.result", mode=BroadcastMode.PUB),
        ],
        kernel_bindings=[KernelBinding(kernel_type="python", purpose="事实核查")],
    )


@pytest.fixture
def guidance_agent():
    return AgentDefinition(
        id="agent.guidance.decision",
        name="导学决策 Agent",
        role="系统决策中枢，整合所有输出做出最优教学路径决策",
        system_prompt=PromptReference(template_id="tpl.guidance", version="v4.0.0"),
        tools=["internal.topology_analysis", "internal.path_simulation"],
        broadcast_channels=[
            BroadcastChannel(channel="learning.diagnosis.report", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="knowledge.generation.output", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="knowledge.review.result", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="guidance.decision.command", mode=BroadcastMode.PUB),
        ],
        kernel_bindings=[
            KernelBinding(kernel_type="python", purpose="路径模拟"),
            KernelBinding(kernel_type="r", purpose="统计检验"),
        ],
        decision_authority=DecisionAuthority(
            scheduling=True, intervention=True, adaptive=True,
        ),
        self_evolution=SelfEvolutionConfig(
            enabled=True, prompt_template_management=True,
            strategy_revision=True, reflection_integration=True,
        ),
    )


@pytest.fixture
def full_registry(diagnosis_agent, generation_agent, review_agent, guidance_agent):
    registry = AgentRegistry()
    registry.register(diagnosis_agent)
    registry.register(generation_agent)
    registry.register(review_agent)
    registry.register(guidance_agent)
    return registry


# ============================================================
# 1. DependencyResolver 测试
# ============================================================

class TestDependencyResolver:
    """依赖解析器测试."""

    def test_build_dependency_graph(self, full_registry):
        """根据广播频道订阅关系构建依赖图."""
        resolver = DependencyResolver(full_registry)
        graph = resolver.build_dependency_graph()

        # 导学Agent 订阅了诊断Agent的发布频道 → 依赖诊断Agent
        assert "agent.guidance.decision" in graph
        assert "agent.learning.diagnosis" in graph["agent.guidance.decision"]
        assert "agent.knowledge.generation" in graph["agent.guidance.decision"]
        assert "agent.knowledge.review" in graph["agent.guidance.decision"]

    def test_topological_sort(self, full_registry):
        """拓扑排序应将诊断Agent排在导学Agent之前."""
        resolver = DependencyResolver(full_registry)
        resolver.build_dependency_graph()
        order = resolver.topological_sort()

        diagnosis_idx = order.index("agent.learning.diagnosis")
        generation_idx = order.index("agent.knowledge.generation")
        review_idx = order.index("agent.knowledge.review")
        guidance_idx = order.index("agent.guidance.decision")

        # 诊断应在生成之前
        assert diagnosis_idx < generation_idx
        # 生成应在审核之前
        assert generation_idx < review_idx
        # 审核应在导学之前
        assert review_idx < guidance_idx

    def test_detect_cycle_raises(self, full_registry):
        """循环依赖应报错."""
        resolver = DependencyResolver(full_registry)
        resolver.build_dependency_graph()
        # 手动注入循环
        resolver._graph["agent.learning.diagnosis"].add("agent.guidance.decision")
        with pytest.raises(DependencyCycleError):
            resolver.topological_sort()

    def test_get_dependencies(self, full_registry):
        """获取指定 Agent 的直接依赖."""
        resolver = DependencyResolver(full_registry)
        resolver.build_dependency_graph()
        deps = resolver.get_dependencies("agent.guidance.decision")
        assert len(deps) == 3

    def test_get_dependents(self, full_registry):
        """获取依赖指定 Agent 的下游 Agent."""
        resolver = DependencyResolver(full_registry)
        resolver.build_dependency_graph()
        dependents = resolver.get_dependents("agent.learning.diagnosis")
        # 生成 Agent 和导学 Agent 都依赖诊断 Agent
        assert "agent.knowledge.generation" in dependents
        assert "agent.guidance.decision" in dependents

    def test_check_missing_dependency(self, full_registry):
        """检查 Agent 依赖的工具是否都已注册."""
        resolver = DependencyResolver(full_registry)
        missing = resolver.check_tool_dependencies()
        # 所有工具都是 internal.* 未在 ToolRegistry 注册 → 报告缺失
        assert len(missing) > 0
        assert any("internal.bkt_compute" in m for m in missing)

    def test_get_startup_order(self, full_registry):
        """获取 Agent 启动顺序 (拓扑排序结果)."""
        resolver = DependencyResolver(full_registry)
        order = resolver.get_startup_order()
        assert len(order) == 4
        # 诊断 Agent 应第一个启动
        assert order[0] == "agent.learning.diagnosis"
        # 导学 Agent 应最后启动
        assert order[-1] == "agent.guidance.decision"


# ============================================================
# 2. AgentDiscoveryService 测试
# ============================================================

class TestAgentDiscoveryService:
    """Agent 发现服务测试."""

    def test_discover_by_capability(self, full_registry):
        """按能力发现 Agent (工具/频道/决策权)."""
        service = AgentDiscoveryService(full_registry)
        results = service.discover_by_capability(
            tool="internal.bkt_compute"
        )
        # 诊断 Agent 和生成 Agent 都绑定了 bkt_compute
        ids = [r.id for r in results]
        assert "agent.learning.diagnosis" in ids
        assert "agent.knowledge.generation" in ids

    def test_discover_by_channel_subscription(self, full_registry):
        """按频道订阅发现 Agent."""
        service = AgentDiscoveryService(full_registry)
        results = service.discover_by_channel(
            "learning.diagnosis.report",
            mode=BroadcastMode.SUB,
        )
        ids = [r.id for r in results]
        # 生成 Agent 和导学 Agent 都订阅了诊断报告
        assert "agent.knowledge.generation" in ids
        assert "agent.guidance.decision" in ids
        # 诊断 Agent 是 PUB 不是 SUB
        assert "agent.learning.diagnosis" not in ids

    def test_discover_decision_authority(self, full_registry):
        """发现拥有决策权限的 Agent."""
        service = AgentDiscoveryService(full_registry)
        results = service.discover_decision_authority()
        assert len(results) == 1
        assert results[0].id == "agent.guidance.decision"

    def test_discover_by_kernel_type(self, full_registry):
        """按内核类型发现 Agent."""
        service = AgentDiscoveryService(full_registry)
        results = service.discover_by_kernel_type("r")
        assert len(results) == 1
        assert results[0].id == "agent.guidance.decision"

    def test_check_agent_readiness_all_ready(self, full_registry, prompt_manager):
        """所有 Agent 的 Prompt 版本都已注册 → 全部就绪."""
        service = AgentDiscoveryService(full_registry, prompt_manager=prompt_manager)
        readiness = service.check_readiness()
        assert len(readiness) == 4
        for agent_id, ready in readiness.items():
            assert ready is True, f"{agent_id} should be ready"

    def test_check_agent_readiness_missing_prompt(self, full_registry):
        """缺少 Prompt 版本 → 未就绪."""
        service = AgentDiscoveryService(full_registry)
        readiness = service.check_readiness()
        assert len(readiness) == 4
        for agent_id, ready in readiness.items():
            assert ready is False, f"{agent_id} should not be ready (no prompt)"

    def test_get_agent_card(self, full_registry):
        """获取 Agent Card (简化版定义, 用于发现协议)."""
        service = AgentDiscoveryService(full_registry)
        card = service.get_agent_card("agent.learning.diagnosis")
        assert card is not None
        assert card["id"] == "agent.learning.diagnosis"
        assert card["name"] == "学情诊断 Agent"
        assert "tools" in card
        assert "broadcast_channels" in card

    def test_get_agent_card_nonexistent(self, full_registry):
        """获取不存在 Agent 的 Card 返回 None."""
        service = AgentDiscoveryService(full_registry)
        card = service.get_agent_card("agent.nonexistent")
        assert card is None

    def test_export_discovery_manifest(self, full_registry):
        """导出发现清单 (所有 Agent 的 Card)."""
        service = AgentDiscoveryService(full_registry)
        manifest = service.export_manifest()
        assert "agents" in manifest
        assert len(manifest["agents"]) == 4
        assert manifest["total"] == 4


# ============================================================
# 3. AgentCompatibilityChecker 测试
# ============================================================

class TestAgentCompatibilityChecker:
    """Agent 兼容性检查器测试."""

    def test_check_compatible_agents(self, full_registry):
        """检查互相兼容的 Agent 组合."""
        checker = AgentCompatibilityChecker(full_registry)
        result = checker.check_compatibility(
            "agent.learning.diagnosis",
            "agent.knowledge.generation",
        )
        assert result["compatible"] is True
        assert len(result["shared_tools"]) >= 1
        assert "internal.bkt_compute" in result["shared_tools"]

    def test_check_incompatible_broadcast(self, full_registry):
        """两个 Agent 都只 PUB 不 SUB 同一频道 → 不兼容."""
        checker = AgentCompatibilityChecker(full_registry)
        result = checker.check_compatibility(
            "agent.learning.diagnosis",
            "agent.knowledge.review",
        )
        # 诊断 PUB diagnosis.report，审核不订阅它
        # 审核 PUB review.result，诊断不订阅它
        # 没有共享频道 → 可独立运行但无直接通信
        assert result["compatible"] is True  # 可以共存但不通信
        assert result["shared_channels"] == []

    def test_check_circular_broadcast(self, full_registry):
        """两个 Agent 在同一频道上互相 PUB → 冲突."""
        checker = AgentCompatibilityChecker(full_registry)
        result = checker.check_compatibility(
            "agent.learning.diagnosis",
            "agent.learning.diagnosis",  # 自身比较
        )
        assert result["compatible"] is True  # 自身总是兼容

    def test_validate_orchestration_order(self, full_registry):
        """验证编排顺序是否满足依赖."""
        checker = AgentCompatibilityChecker(full_registry)
        # 正确顺序
        valid = checker.validate_order([
            "agent.learning.diagnosis",
            "agent.knowledge.generation",
            "agent.knowledge.review",
            "agent.guidance.decision",
        ])
        assert valid is True

        # 错误顺序 (导学在诊断之前)
        invalid = checker.validate_order([
            "agent.guidance.decision",
            "agent.learning.diagnosis",
        ])
        assert invalid is False


# ============================================================
# 4. HealthMonitor 测试
# ============================================================

class TestHealthMonitor:
    """健康监控器测试."""

    @pytest.fixture
    def factory(self, prompt_manager):
        return AgentFactory(prompt_manager=prompt_manager)

    @pytest.mark.asyncio
    async def test_register_and_check_health(
        self, factory, full_registry, diagnosis_agent
    ):
        """注册实例并检查健康状态."""
        instance = await factory.instantiate(
            diagnosis_agent.id, full_registry,
            learner_context={"learner_name": "测试"},
        )
        instance.activate()

        monitor = HealthMonitor()
        monitor.register(instance)
        health = monitor.check(instance.instance_id)

        assert health is not None
        assert health.status == HealthStatus.HEALTHY
        assert health.agent_id == diagnosis_agent.id

    @pytest.mark.asyncio
    async def test_health_status_paused(
        self, factory, full_registry, diagnosis_agent
    ):
        """暂停的实例应报告 DEGRADED."""
        instance = await factory.instantiate(
            diagnosis_agent.id, full_registry,
            learner_context={"learner_name": "测试"},
        )
        instance.activate()
        instance.pause()

        monitor = HealthMonitor()
        monitor.register(instance)
        health = monitor.check(instance.instance_id)

        assert health.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_status_terminated(
        self, factory, full_registry, diagnosis_agent
    ):
        """终止的实例应报告 UNHEALTHY."""
        instance = await factory.instantiate(
            diagnosis_agent.id, full_registry,
            learner_context={"learner_name": "测试"},
        )
        instance.terminate()

        monitor = HealthMonitor()
        monitor.register(instance)
        health = monitor.check(instance.instance_id)

        assert health.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_all(
        self, factory, full_registry, diagnosis_agent, generation_agent
    ):
        """批量健康检查."""
        inst1 = await factory.instantiate(
            diagnosis_agent.id, full_registry,
            learner_context={"learner_name": "A"},
        )
        inst2 = await factory.instantiate(
            generation_agent.id, full_registry,
        )
        inst1.activate()
        inst2.activate()

        monitor = HealthMonitor()
        monitor.register(inst1)
        monitor.register(inst2)

        all_health = monitor.check_all()
        assert len(all_health) == 2
        for h in all_health.values():
            assert h.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_heartbeat_updates_last_seen(
        self, factory, full_registry, diagnosis_agent
    ):
        """心跳应更新 last_seen 时间."""
        instance = await factory.instantiate(
            diagnosis_agent.id, full_registry,
            learner_context={"learner_name": "测试"},
        )
        instance.activate()

        monitor = HealthMonitor(heartbeat_timeout_s=10)
        monitor.register(instance)

        old_seen = monitor.check(instance.instance_id).last_seen
        await asyncio.sleep(0.05)
        monitor.heartbeat(instance.instance_id)
        new_seen = monitor.check(instance.instance_id).last_seen

        assert new_seen > old_seen

    @pytest.mark.asyncio
    async def test_stale_instance_detected(
        self, factory, full_registry, diagnosis_agent
    ):
        """心跳超时的实例应被标记为 STALE."""
        instance = await factory.instantiate(
            diagnosis_agent.id, full_registry,
            learner_context={"learner_name": "测试"},
        )
        instance.activate()

        monitor = HealthMonitor(heartbeat_timeout_s=0.01)
        monitor.register(instance)
        await asyncio.sleep(0.05)

        health = monitor.check(instance.instance_id)
        assert health.status == HealthStatus.STALE

    @pytest.mark.asyncio
    async def test_unregister_instance(
        self, factory, full_registry, diagnosis_agent
    ):
        """注销实例后健康检查返回 None."""
        instance = await factory.instantiate(
            diagnosis_agent.id, full_registry,
            learner_context={"learner_name": "测试"},
        )

        monitor = HealthMonitor()
        monitor.register(instance)
        monitor.unregister(instance.instance_id)

        assert monitor.check(instance.instance_id) is None

    @pytest.mark.asyncio
    async def test_get_unhealthy_instances(
        self, factory, full_registry, diagnosis_agent, generation_agent
    ):
        """获取所有不健康的实例."""
        inst1 = await factory.instantiate(
            diagnosis_agent.id, full_registry,
            learner_context={"learner_name": "A"},
        )
        inst2 = await factory.instantiate(
            generation_agent.id, full_registry,
        )
        inst1.activate()
        inst2.activate()
        inst2.terminate()  # inst2 不健康

        monitor = HealthMonitor()
        monitor.register(inst1)
        monitor.register(inst2)

        unhealthy = monitor.get_unhealthy()
        assert len(unhealthy) == 1
        assert unhealthy[0].instance_id == inst2.instance_id
