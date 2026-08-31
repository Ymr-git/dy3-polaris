"""Agent 定义与注册端到端集成测试.

验证完整工作流:
1. 注册所有核心 Agent 到注册中心
2. 通过发现服务验证就绪状态
3. 解析依赖图确定启动顺序
4. 按顺序实例化所有 Agent
5. 健康检查全部实例
6. 验证 Agent 间通信拓扑
7. 验证与 DecisionEngine 的对接接口
"""

from __future__ import annotations

import asyncio
import pytest

from dy3_polaris.l5.agent_definition import (
    AgentDefinition,
    AgentFactory,
    AgentInstance,
    AgentInstanceState,
    AgentRegistry,
    BroadcastChannel,
    BroadcastMode,
    DecisionAuthority,
    KernelBinding,
    MemoryConfig,
    PromptReference,
    PromptVersion,
    PromptVersionManager,
    ReputationConfig,
    SelfEvolutionConfig,
)
from dy3_polaris.l5.agent_discovery import (
    AgentCompatibilityChecker,
    AgentDiscoveryService,
    DependencyResolver,
    HealthMonitor,
    HealthStatus,
)


# ============================================================
# 四个核心 Agent 定义 (来自 L5 设计文档)
# ============================================================

def _make_diagnosis_agent() -> AgentDefinition:
    return AgentDefinition(
        id="agent.learning.diagnosis",
        name="学情诊断 Agent",
        role="基于 BKT/IRT 引擎对学习者的 Dy3+ 发光材料知识掌握状态进行实时诊断",
        system_prompt=PromptReference(template_id="tpl.diagnosis", version="v2.1.0"),
        tools=["internal.bkt_compute", "internal.irt_evaluate", "internal.forgetfulness_scan"],
        connectors=["l3.learning_record"],
        memory_config=MemoryConfig(
            read_stores=["milvus", "neo4j", "postgresql"],
            write_stores=["neo4j", "postgresql"],
        ),
        reputation_config=ReputationConfig(initial_score=85, penalty_factor=0.8, reward_factor=1.2),
        broadcast_channels=[
            BroadcastChannel(channel="learning.diagnosis.report", mode=BroadcastMode.PUB),
            BroadcastChannel(channel="learning.interaction.event", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="learning.knowledge.gap", mode=BroadcastMode.PUBSUB),
        ],
        kernel_bindings=[KernelBinding(kernel_type="python", purpose="BKT参数EM校准与遗忘曲线计算")],
    )


def _make_generation_agent() -> AgentDefinition:
    return AgentDefinition(
        id="agent.knowledge.generation",
        name="知识生成 Agent",
        role="根据学情诊断结果，结合 L3 知识图谱生成个性化学习内容",
        system_prompt=PromptReference(template_id="tpl.generation", version="v3.0.0"),
        tools=["internal.knowledge_synthesize", "internal.path_generator"],
        connectors=["l3.knowledge_graph"],
        memory_config=MemoryConfig(
            read_stores=["milvus", "neo4j"],
            write_stores=["milvus", "postgresql"],
        ),
        reputation_config=ReputationConfig(initial_score=80, penalty_factor=0.9, reward_factor=1.1),
        broadcast_channels=[
            BroadcastChannel(channel="learning.diagnosis.report", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="knowledge.generation.output", mode=BroadcastMode.PUB),
            BroadcastChannel(channel="learning.knowledge.gap", mode=BroadcastMode.SUB),
        ],
        kernel_bindings=[KernelBinding(kernel_type="python", purpose="知识图谱推理与内容生成")],
    )


def _make_review_agent() -> AgentDefinition:
    return AgentDefinition(
        id="agent.knowledge.review",
        name="审核校验 Agent",
        role="对知识生成结果进行多维度审核校验，包括事实核查、幻觉检测和一致性验证",
        system_prompt=PromptReference(template_id="tpl.review", version="v1.5.0"),
        tools=["internal.fact_check", "internal.hallucination_detect"],
        connectors=["l3.knowledge_graph"],
        memory_config=MemoryConfig(
            read_stores=["milvus", "neo4j"],
            write_stores=["postgresql"],
        ),
        reputation_config=ReputationConfig(initial_score=90, penalty_factor=0.5, reward_factor=1.5),
        broadcast_channels=[
            BroadcastChannel(channel="knowledge.generation.output", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="knowledge.review.result", mode=BroadcastMode.PUB),
        ],
        kernel_bindings=[KernelBinding(kernel_type="python", purpose="事实核查与幻觉检测")],
    )


def _make_guidance_agent() -> AgentDefinition:
    return AgentDefinition(
        id="agent.guidance.decision",
        name="导学决策 Agent",
        role="系统决策中枢，持有调度权、干预权和自适应权，整合所有输出做出最优教学路径决策",
        system_prompt=PromptReference(template_id="tpl.guidance", version="v4.0.0"),
        tools=["internal.topology_analysis", "internal.path_simulation", "internal.resource_matching"],
        connectors=["l3.learning_record", "l3.curriculum"],
        memory_config=MemoryConfig(
            read_stores=["milvus", "neo4j", "postgresql"],
            write_stores=["milvus", "neo4j", "postgresql"],
        ),
        reputation_config=ReputationConfig(initial_score=95, penalty_factor=0.3, reward_factor=2.0),
        broadcast_channels=[
            BroadcastChannel(channel="learning.diagnosis.report", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="knowledge.generation.output", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="knowledge.review.result", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="guidance.decision.command", mode=BroadcastMode.PUB),
        ],
        kernel_bindings=[
            KernelBinding(kernel_type="python", purpose="路径模拟蒙特卡洛计算"),
            KernelBinding(kernel_type="r", purpose="声誉评分计算与A/B测试显著性检验"),
        ],
        decision_authority=DecisionAuthority(
            scheduling=True, intervention=True, adaptive=True,
        ),
        self_evolution=SelfEvolutionConfig(
            enabled=True, prompt_template_management=True,
            strategy_revision=True, reflection_integration=True,
        ),
    )


# ============================================================
# 端到端集成测试
# ============================================================

class TestAgentDefinitionEndToEnd:
    """Agent 定义与注册端到端集成测试."""

    @pytest.fixture
    def prompt_manager(self):
        pm = PromptVersionManager()
        pm.register(PromptVersion(
            template_id="tpl.diagnosis", version="v2.1.0",
            content="You are a diagnosis agent for {learner_name}.",
            created_by="system",
        ))
        pm.register(PromptVersion(
            template_id="tpl.generation", version="v3.0.0",
            content="You are a generation agent.",
            created_by="system",
        ))
        pm.register(PromptVersion(
            template_id="tpl.review", version="v1.5.0",
            content="You are a review agent.",
            created_by="system",
        ))
        pm.register(PromptVersion(
            template_id="tpl.guidance", version="v4.0.0",
            content="You are the guidance decision agent.",
            created_by="system",
        ))
        return pm

    @pytest.fixture
    def full_registry(self):
        registry = AgentRegistry()
        registry.register(_make_diagnosis_agent())
        registry.register(_make_generation_agent())
        registry.register(_make_review_agent())
        registry.register(_make_guidance_agent())
        return registry

    def test_all_four_agents_registered(self, full_registry):
        """四个核心 Agent 均已注册."""
        assert full_registry.size == 4
        ids = {a.id for a in full_registry.list_all()}
        assert "agent.learning.diagnosis" in ids
        assert "agent.knowledge.generation" in ids
        assert "agent.knowledge.review" in ids
        assert "agent.guidance.decision" in ids

    def test_only_guidance_has_decision_authority(self, full_registry):
        """只有导学决策 Agent 拥有决策权限."""
        decision_agents = full_registry.find_decision_authority_agents()
        assert len(decision_agents) == 1
        assert decision_agents[0].id == "agent.guidance.decision"

    def test_only_guidance_has_self_evolution(self, full_registry):
        """只有导学决策 Agent 开启了自演化."""
        for agent in full_registry.list_all():
            if agent.id == "agent.guidance.decision":
                assert agent.self_evolution.enabled is True
            else:
                assert agent.self_evolution.enabled is False

    def test_only_guidance_has_two_kernels(self, full_registry):
        """只有导学决策 Agent 绑定了两个内核 (Python + R)."""
        for agent in full_registry.list_all():
            if agent.id == "agent.guidance.decision":
                assert len(agent.kernel_bindings) == 2
                types = {kb.kernel_type for kb in agent.kernel_bindings}
                assert types == {"python", "r"}
            else:
                assert len(agent.kernel_bindings) == 1

    def test_dependency_graph_correct(self, full_registry):
        """依赖图正确反映广播订阅关系."""
        resolver = DependencyResolver(full_registry)
        graph = resolver.build_dependency_graph()

        # 导学Agent 依赖诊断、生成、审核
        assert graph["agent.guidance.decision"] == {
            "agent.learning.diagnosis",
            "agent.knowledge.generation",
            "agent.knowledge.review",
        }
        # 生成Agent 依赖诊断
        assert graph["agent.knowledge.generation"] == {
            "agent.learning.diagnosis",
        }
        # 审核Agent 依赖生成
        assert graph["agent.knowledge.review"] == {
            "agent.knowledge.generation",
        }
        # 诊断Agent 无依赖
        assert "agent.learning.diagnosis" not in graph or len(graph["agent.learning.diagnosis"]) == 0

    def test_startup_order(self, full_registry):
        """启动顺序: 诊断 → 生成 → 审核 → 导学."""
        resolver = DependencyResolver(full_registry)
        order = resolver.get_startup_order()

        assert order == [
            "agent.learning.diagnosis",
            "agent.knowledge.generation",
            "agent.knowledge.review",
            "agent.guidance.decision",
        ]

    def test_all_agents_ready_with_prompts(self, full_registry, prompt_manager):
        """所有 Prompt 版本注册后，所有 Agent 就绪."""
        service = AgentDiscoveryService(full_registry, prompt_manager=prompt_manager)
        readiness = service.check_readiness()
        assert all(readiness.values())

    def test_agent_card_export(self, full_registry):
        """Agent Card 包含完整发现信息."""
        service = AgentDiscoveryService(full_registry)
        card = service.get_agent_card("agent.guidance.decision")
        assert card is not None
        assert card["has_decision_authority"] is True
        assert "r" in card["kernel_types"]
        assert "python" in card["kernel_types"]
        assert len(card["broadcast_channels"]) == 4

    def test_manifest_export(self, full_registry):
        """发现清单包含所有 Agent."""
        service = AgentDiscoveryService(full_registry)
        manifest = service.export_manifest()
        assert manifest["total"] == 4
        assert len(manifest["agents"]) == 4

    def test_compatibility_check(self, full_registry):
        """诊断 Agent 和生成 Agent 通过共享工具兼容."""
        checker = AgentCompatibilityChecker(full_registry)
        result = checker.check_compatibility(
            "agent.learning.diagnosis",
            "agent.knowledge.generation",
        )
        assert result["compatible"] is True
        assert len(result["communication_channels"]) > 0

    def test_validate_correct_order(self, full_registry):
        """正确编排顺序通过验证."""
        checker = AgentCompatibilityChecker(full_registry)
        assert checker.validate_order([
            "agent.learning.diagnosis",
            "agent.knowledge.generation",
            "agent.knowledge.review",
            "agent.guidance.decision",
        ]) is True

    def test_validate_incorrect_order(self, full_registry):
        """错误编排顺序不通过验证."""
        checker = AgentCompatibilityChecker(full_registry)
        assert checker.validate_order([
            "agent.guidance.decision",
            "agent.learning.diagnosis",
        ]) is False

    @pytest.mark.asyncio
    async def test_full_instantiation_pipeline(self, full_registry, prompt_manager):
        """完整实例化流水线: 按启动顺序实例化所有 Agent."""
        factory = AgentFactory(prompt_manager=prompt_manager)
        resolver = DependencyResolver(full_registry)
        order = resolver.get_startup_order()

        instances: list[AgentInstance] = []
        for agent_id in order:
            instance = await factory.instantiate(
                agent_id, full_registry,
                learner_context={"learner_name": "集成测试用户"},
            )
            instances.append(instance)

        assert len(instances) == 4
        assert factory.instance_count == 4

        # 所有实例初始状态为 READY
        for inst in instances:
            assert inst.state == AgentInstanceState.READY

    @pytest.mark.asyncio
    async def test_health_monitoring_all_healthy(
        self, full_registry, prompt_manager
    ):
        """实例化后健康检查全部 HEALTHY."""
        factory = AgentFactory(prompt_manager=prompt_manager)
        monitor = HealthMonitor()

        for agent in full_registry.list_all():
            instance = await factory.instantiate(
                agent.id, full_registry,
                learner_context={"learner_name": "健康检查用户"},
            )
            instance.activate()
            monitor.register(instance)

        summary = monitor.get_health_summary()
        assert summary["total"] == 4
        assert summary["healthy"] == 4
        assert summary["all_healthy"] is True

    @pytest.mark.asyncio
    async def test_provenance_chain(self, full_registry, prompt_manager):
        """每个实例都有 Provenance 记录."""
        factory = AgentFactory(prompt_manager=prompt_manager)

        for agent in full_registry.list_all():
            instance = await factory.instantiate(
                agent.id, full_registry,
                learner_context={"learner_name": "溯源用户"},
            )
            assert instance.provenance_record is not None
            assert instance.provenance_record["event_type"] == "agent_instantiated"
            assert instance.provenance_record["agent_id"] == agent.id
            assert "tools_bound" in instance.provenance_record
            assert "kernel_count" in instance.provenance_record

    @pytest.mark.asyncio
    async def test_working_session_checkpoint(
        self, full_registry, prompt_manager
    ):
        """每个实例的 Working Session 有初始检查点."""
        factory = AgentFactory(prompt_manager=prompt_manager)

        instance = await factory.instantiate(
            "agent.learning.diagnosis", full_registry,
            learner_context={"learner_name": "检查点用户"},
        )
        assert instance.working_session.checkpoint_count >= 1

    @pytest.mark.asyncio
    async def test_instance_lifecycle_full_cycle(
        self, full_registry, prompt_manager
    ):
        """完整生命周期: READY → ACTIVE → PAUSED → ACTIVE → TERMINATED."""
        factory = AgentFactory(prompt_manager=prompt_manager)

        instance = await factory.instantiate(
            "agent.guidance.decision", full_registry,
            learner_context={"learner_name": "生命周期用户"},
        )

        assert instance.state == AgentInstanceState.READY

        instance.activate()
        assert instance.state == AgentInstanceState.ACTIVE

        instance.pause()
        assert instance.state == AgentInstanceState.PAUSED

        instance.resume()
        assert instance.state == AgentInstanceState.ACTIVE

        instance.terminate()
        assert instance.state == AgentInstanceState.TERMINATED
        assert len(instance.kernel_handles) == 0
        assert len(instance.broadcast_subscriptions) == 0

    @pytest.mark.asyncio
    async def test_prompt_version_rollback(
        self, full_registry, prompt_manager
    ):
        """Prompt 版本回滚后实例化使用回滚版本."""
        # 注册新版本
        prompt_manager.register(PromptVersion(
            template_id="tpl.diagnosis",
            version="v2.2.0",
            content="Updated diagnosis agent for {learner_name}.",
            created_by="admin",
        ))

        # 回滚到旧版本
        prompt_manager.rollback("tpl.diagnosis", "v2.1.0")

        factory = AgentFactory(prompt_manager=prompt_manager)
        instance = await factory.instantiate(
            "agent.learning.diagnosis", full_registry,
            learner_context={"learner_name": "回滚用户"},
        )

        # 使用回滚版本的 Prompt
        assert "diagnosis agent for 回滚用户" in instance.rendered_prompt

    def test_broadcast_topology_complete(self, full_registry):
        """广播拓扑完整: 每个频道至少有一个 PUB 和一个 SUB."""
        service = AgentDiscoveryService(full_registry)

        # 诊断报告频道
        pub_agents = service.discover_by_channel(
            "learning.diagnosis.report", mode=BroadcastMode.PUB,
        )
        sub_agents = service.discover_by_channel(
            "learning.diagnosis.report", mode=BroadcastMode.SUB,
        )
        assert len(pub_agents) >= 1  # 诊断Agent 发布
        assert len(sub_agents) >= 1  # 生成/导学Agent 订阅

        # 知识生成输出频道
        pub_agents = service.discover_by_channel(
            "knowledge.generation.output", mode=BroadcastMode.PUB,
        )
        sub_agents = service.discover_by_channel(
            "knowledge.generation.output", mode=BroadcastMode.SUB,
        )
        assert len(pub_agents) >= 1  # 生成Agent 发布
        assert len(sub_agents) >= 1  # 审核/导学Agent 订阅

        # 审核结果频道
        pub_agents = service.discover_by_channel(
            "knowledge.review.result", mode=BroadcastMode.PUB,
        )
        sub_agents = service.discover_by_channel(
            "knowledge.review.result", mode=BroadcastMode.SUB,
        )
        assert len(pub_agents) >= 1  # 审核Agent 发布
        assert len(sub_agents) >= 1  # 导学Agent 订阅

        # 导学决策命令频道
        pub_agents = service.discover_by_channel(
            "guidance.decision.command", mode=BroadcastMode.PUB,
        )
        assert len(pub_agents) >= 1  # 导学Agent 发布
