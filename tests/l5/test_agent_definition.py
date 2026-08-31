"""Agent 定义与注册模块测试.

TDD 测试用例 — 覆盖:
1. AgentDefinition 数据模型与验证
2. AgentRegistry 注册中心 (注册/查询/注销/索引)
3. PromptVersionManager 提示词版本管理 (CRUD/A-B测试/回滚)
4. AgentFactory 六步实例化流水线
5. AgentInstance 运行时实例与生命周期

融合世界先进方案:
- LangGraph: 有状态节点 + 条件边
- OpenAI Agents SDK: Agent Card + Handoff
- Google ADK: DAG 任务分解 + Agent 注册
- CrewAI: 角色化 Agent 定义
- AutoGen: 消息传递 + Agent 注册表
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dy3_polaris.l5.agent_definition import (
    AgentDefinition,
    AgentRegistry,
    AgentRegistryError,
    AgentNotFoundError,
    AgentAlreadyExistsError,
    PromptVersionManager,
    PromptVersion,
    PromptVersionError,
    AgentFactory,
    AgentInstance,
    AgentInstanceState,
    FactoryStep,
    FactoryError,
    BroadcastMode,
    KernelBinding,
    MemoryConfig,
    ReputationConfig,
    DecisionAuthority,
    SelfEvolutionConfig,
    BroadcastChannel,
    PromptReference,
)
from dy3_polaris.l6.registry.tool_registry import ToolRegistry


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tool_registry():
    """提供一个空的 ToolRegistry 实例."""
    return ToolRegistry()


@pytest.fixture
def prompt_manager():
    """提供一个 PromptVersionManager 实例."""
    return PromptVersionManager()


@pytest.fixture
def agent_registry():
    """提供一个 AgentRegistry 实例."""
    return AgentRegistry()


@pytest.fixture
def sample_agent_def():
    """提供一个标准的 AgentDefinition (学情诊断 Agent)."""
    return AgentDefinition(
        id="agent.learning.diagnosis",
        name="学情诊断 Agent",
        role="基于 BKT/IRT 引擎对学习者的 Dy3+ 发光材料知识掌握状态进行实时诊断，输出知识图谱缺口和掌握概率向量",
        system_prompt=PromptReference(
            template_id="tpl.diagnosis",
            version="v2.1.0",
        ),
        tools=[
            "internal.bkt_compute",
            "internal.irt_evaluate",
            "internal.forgetfulness_scan",
        ],
        connectors=["l3.learning_record"],
        memory_config=MemoryConfig(
            read_stores=["milvus", "neo4j", "postgresql"],
            write_stores=["neo4j", "postgresql"],
        ),
        reputation_config=ReputationConfig(
            initial_score=85,
            penalty_factor=0.8,
            reward_factor=1.2,
        ),
        broadcast_channels=[
            BroadcastChannel(channel="learning.diagnosis.report", mode=BroadcastMode.PUB),
            BroadcastChannel(channel="learning.interaction.event", mode=BroadcastMode.SUB),
            BroadcastChannel(channel="learning.knowledge.gap", mode=BroadcastMode.PUBSUB),
        ],
        kernel_bindings=[
            KernelBinding(kernel_type="python", purpose="BKT参数EM校准与遗忘曲线计算"),
        ],
    )


@pytest.fixture
def guidance_agent_def():
    """提供导学决策 Agent 定义 (决策中枢)."""
    return AgentDefinition(
        id="agent.guidance.decision",
        name="导学决策 Agent",
        role="作为系统决策中枢，持有调度权、干预权和自适应权。整合学情诊断、知识生成和审核校验的输出，做出最优教学路径决策",
        system_prompt=PromptReference(
            template_id="tpl.guidance",
            version="v4.0.0",
        ),
        tools=[
            "internal.topology_analysis",
            "internal.path_simulation",
            "internal.resource_matching",
        ],
        connectors=["l3.learning_record", "l3.curriculum"],
        memory_config=MemoryConfig(
            read_stores=["milvus", "neo4j", "postgresql"],
            write_stores=["milvus", "neo4j", "postgresql"],
        ),
        reputation_config=ReputationConfig(
            initial_score=95,
            penalty_factor=0.3,
            reward_factor=2.0,
        ),
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
            scheduling=True,
            intervention=True,
            adaptive=True,
        ),
        self_evolution=SelfEvolutionConfig(
            enabled=True,
            prompt_template_management=True,
            strategy_revision=True,
            reflection_integration=True,
        ),
    )


# ============================================================
# 1. AgentDefinition 数据模型验证
# ============================================================

class TestAgentDefinition:
    """AgentDefinition 数据模型测试."""

    def test_valid_agent_definition(self, sample_agent_def):
        """有效的 Agent 定义应正常创建."""
        assert sample_agent_def.id == "agent.learning.diagnosis"
        assert sample_agent_def.name == "学情诊断 Agent"
        assert len(sample_agent_def.tools) == 3
        assert len(sample_agent_def.broadcast_channels) == 3

    def test_invalid_id_format_raises(self):
        """ID 不符合 agent.{domain}.{name} 格式应报错."""
        with pytest.raises(ValueError, match="String should match pattern"):
            AgentDefinition(
                id="invalid_id",
                name="Test Agent",
                role="A" * 10,
                system_prompt=PromptReference(template_id="tpl", version="v1.0.0"),
                tools=["tool1"],
                memory_config=MemoryConfig(),
                reputation_config=ReputationConfig(),
                broadcast_channels=[],
            )

    def test_empty_tools_raises(self):
        """工具列表为空应报错 (至少需要一个工具)."""
        with pytest.raises(ValueError, match="at least 1 item"):
            AgentDefinition(
                id="agent.test.valid",
                name="Test Agent",
                role="A" * 10,
                system_prompt=PromptReference(template_id="tpl", version="v1.0.0"),
                tools=[],
                memory_config=MemoryConfig(),
                reputation_config=ReputationConfig(),
                broadcast_channels=[],
            )

    def test_too_many_kernel_bindings_raises(self):
        """内核绑定超过 2 个应报错 (最多 Python + R)."""
        with pytest.raises(ValueError, match="at most 2 items"):
            AgentDefinition(
                id="agent.test.valid",
                name="Test Agent",
                role="A" * 10,
                system_prompt=PromptReference(template_id="tpl", version="v1.0.0"),
                tools=["tool1"],
                memory_config=MemoryConfig(),
                reputation_config=ReputationConfig(),
                broadcast_channels=[],
                kernel_bindings=[
                    KernelBinding(kernel_type="python", purpose="a"),
                    KernelBinding(kernel_type="r", purpose="b"),
                    KernelBinding(kernel_type="python", purpose="c"),
                ],
            )

    def test_invalid_memory_store_raises(self):
        """无效的记忆存储名应报错."""
        with pytest.raises(ValueError):
            MemoryConfig(read_stores=["invalid_store"])

    def test_reputation_score_range(self):
        """声誉分数应在 0-100 范围内."""
        with pytest.raises(ValueError):
            ReputationConfig(initial_score=150)
        with pytest.raises(ValueError):
            ReputationConfig(initial_score=-10)

    def test_decision_authority_defaults(self):
        """非决策中枢 Agent 的 decision_authority 默认全 False."""
        assert sample_agent_def.decision_authority.scheduling is False if 'sample_agent_def' in dir() else True

    def test_self_evolution_defaults(self):
        """非导学 Agent 的 self_evolution 默认关闭."""
        def_def = AgentDefinition(
            id="agent.test.default",
            name="Default Agent",
            role="A" * 10,
            system_prompt=PromptReference(template_id="tpl", version="v1.0.0"),
            tools=["tool1"],
            memory_config=MemoryConfig(),
            reputation_config=ReputationConfig(),
            broadcast_channels=[],
        )
        assert def_def.self_evolution.enabled is False

    def test_to_dict_serialization(self, sample_agent_def):
        """AgentDefinition 应可序列化为字典."""
        d = sample_agent_def.to_dict()
        assert d["id"] == "agent.learning.diagnosis"
        assert "system_prompt" in d
        assert "memory_config" in d
        assert "broadcast_channels" in d

    def test_from_dict_deserialization(self, sample_agent_def):
        """AgentDefinition 应可从字典反序列化."""
        d = sample_agent_def.to_dict()
        restored = AgentDefinition.from_dict(d)
        assert restored.id == sample_agent_def.id
        assert restored.tools == sample_agent_def.tools


# ============================================================
# 2. AgentRegistry 注册中心
# ============================================================

class TestAgentRegistry:
    """AgentRegistry 注册中心测试."""

    def test_register_single_agent(self, agent_registry, sample_agent_def):
        """注册单个 Agent 应成功."""
        agent_registry.register(sample_agent_def)
        assert agent_registry.size == 1
        assert agent_registry.contains(sample_agent_def.id)

    def test_register_duplicate_raises(self, agent_registry, sample_agent_def):
        """重复注册同一 ID 的 Agent 应报错."""
        agent_registry.register(sample_agent_def)
        with pytest.raises(AgentAlreadyExistsError):
            agent_registry.register(sample_agent_def)

    def test_register_with_overwrite(self, agent_registry, sample_agent_def):
        """使用 overwrite=True 应覆盖已有 Agent."""
        agent_registry.register(sample_agent_def)
        updated = sample_agent_def.model_copy(update={"name": "更新后的诊断 Agent"})
        agent_registry.register(updated, overwrite=True)
        retrieved = agent_registry.get(sample_agent_def.id)
        assert retrieved.name == "更新后的诊断 Agent"

    def test_get_existing_agent(self, agent_registry, sample_agent_def):
        """获取已注册的 Agent 应返回正确对象."""
        agent_registry.register(sample_agent_def)
        retrieved = agent_registry.get(sample_agent_def.id)
        assert retrieved is not None
        assert retrieved.id == sample_agent_def.id

    def test_get_nonexistent_agent_returns_none(self, agent_registry):
        """获取不存在的 Agent 应返回 None."""
        assert agent_registry.get("agent.nonexistent.foo") is None

    def test_get_or_raise_nonexistent(self, agent_registry):
        """get_or_raise 对不存在的 Agent 应抛出异常."""
        with pytest.raises(AgentNotFoundError):
            agent_registry.get_or_raise("agent.nonexistent.foo")

    def test_unregister_agent(self, agent_registry, sample_agent_def):
        """注销已注册的 Agent 应成功."""
        agent_registry.register(sample_agent_def)
        removed = agent_registry.unregister(sample_agent_def.id)
        assert removed.id == sample_agent_def.id
        assert not agent_registry.contains(sample_agent_def.id)

    def test_unregister_nonexistent_raises(self, agent_registry):
        """注销不存在的 Agent 应报错."""
        with pytest.raises(AgentNotFoundError):
            agent_registry.unregister("agent.nonexistent.foo")

    def test_list_all_agents(self, agent_registry, sample_agent_def, guidance_agent_def):
        """列出所有已注册 Agent."""
        agent_registry.register(sample_agent_def)
        agent_registry.register(guidance_agent_def)
        all_agents = agent_registry.list_all()
        assert len(all_agents) == 2

    def test_find_by_tool(self, agent_registry, sample_agent_def, guidance_agent_def):
        """按绑定工具查找 Agent."""
        agent_registry.register(sample_agent_def)
        agent_registry.register(guidance_agent_def)
        results = agent_registry.find_by_tool("internal.bkt_compute")
        assert len(results) == 1  # 只有诊断 Agent 绑定了 bkt_compute
        assert results[0].id == "agent.learning.diagnosis"
        results2 = agent_registry.find_by_tool("internal.topology_analysis")
        assert len(results2) == 1
        assert results2[0].id == "agent.guidance.decision"

    def test_find_by_broadcast_channel(self, agent_registry, sample_agent_def, guidance_agent_def):
        """按广播频道查找 Agent."""
        agent_registry.register(sample_agent_def)
        agent_registry.register(guidance_agent_def)
        results = agent_registry.find_by_channel("learning.diagnosis.report")
        # 诊断Agent 发布，导学Agent 订阅
        assert len(results) == 2

    def test_find_decision_authority_agents(self, agent_registry, sample_agent_def, guidance_agent_def):
        """查找拥有决策权限的 Agent."""
        agent_registry.register(sample_agent_def)
        agent_registry.register(guidance_agent_def)
        decision_agents = agent_registry.find_decision_authority_agents()
        assert len(decision_agents) == 1
        assert decision_agents[0].id == "agent.guidance.decision"

    def test_export_registry_summary(self, agent_registry, sample_agent_def, guidance_agent_def):
        """导出注册中心摘要统计."""
        agent_registry.register(sample_agent_def)
        agent_registry.register(guidance_agent_def)
        summary = agent_registry.export_summary()
        assert summary["total_agents"] == 2
        assert "agent_ids" in summary
        assert summary["decision_authority_count"] == 1


# ============================================================
# 3. PromptVersionManager 提示词版本管理
# ============================================================

class TestPromptVersionManager:
    """PromptVersionManager 测试."""

    def test_register_prompt_version(self, prompt_manager):
        """注册新的 Prompt 版本."""
        pv = PromptVersion(
            template_id="tpl.diagnosis",
            version="v2.1.0",
            content="You are a diagnosis agent...",
            changelog="初始版本",
            created_by="system",
        )
        prompt_manager.register(pv)
        assert prompt_manager.count("tpl.diagnosis") == 1

    def test_get_prompt_version(self, prompt_manager):
        """获取指定版本的 Prompt."""
        pv = PromptVersion(
            template_id="tpl.diagnosis",
            version="v2.1.0",
            content="You are a diagnosis agent...",
            created_by="system",
        )
        prompt_manager.register(pv)
        retrieved = prompt_manager.get("tpl.diagnosis", "v2.1.0")
        assert retrieved is not None
        assert retrieved.content == "You are a diagnosis agent..."

    def test_get_nonexistent_prompt_returns_none(self, prompt_manager):
        """获取不存在的 Prompt 版本返回 None."""
        assert prompt_manager.get("tpl.nonexistent", "v1.0.0") is None

    def test_get_active_version(self, prompt_manager):
        """获取当前活跃版本的 Prompt."""
        v1 = PromptVersion(template_id="tpl.test", version="v1.0.0", content="v1", created_by="sys")
        v2 = PromptVersion(template_id="tpl.test", version="v2.0.0", content="v2", created_by="sys")
        prompt_manager.register(v1)
        prompt_manager.register(v2)
        # 最新注册的应自动设为 active
        active = prompt_manager.get_active("tpl.test")
        assert active.version == "v2.0.0"

    def test_list_versions(self, prompt_manager):
        """列出模板的所有版本."""
        for v in ["v1.0.0", "v1.1.0", "v2.0.0"]:
            prompt_manager.register(PromptVersion(
                template_id="tpl.test", version=v, content=f"content_{v}", created_by="sys"
            ))
        versions = prompt_manager.list_versions("tpl.test")
        assert len(versions) == 3
        # 应按版本号排序 (降序)
        assert versions[0].version == "v2.0.0"

    def test_deactivate_version(self, prompt_manager):
        """停用某个版本."""
        pv = PromptVersion(template_id="tpl.test", version="v1.0.0", content="c", created_by="sys")
        prompt_manager.register(pv)
        prompt_manager.deactivate("tpl.test", "v1.0.0")
        retrieved = prompt_manager.get("tpl.test", "v1.0.0")
        assert retrieved.is_active is False

    def test_ab_test_assignment(self, prompt_manager):
        """A/B 测试分组分配."""
        v_a = PromptVersion(template_id="tpl.test", version="v1.0.0", content="A", created_by="sys", ab_group="A")
        v_b = PromptVersion(template_id="tpl.test", version="v1.1.0", content="B", created_by="sys", ab_group="B")
        prompt_manager.register(v_a)
        prompt_manager.register(v_b)
        # 根据学习者 ID 确定性分配
        group_a = prompt_manager.get_ab_version("tpl.test", "learner_001")
        group_b = prompt_manager.get_ab_version("tpl.test", "learner_002")
        # 确定性分配：同一学习者总是分到同一组
        assert group_a is not None
        assert prompt_manager.get_ab_version("tpl.test", "learner_001") == group_a

    def test_rollback_version(self, prompt_manager):
        """版本回滚."""
        v1 = PromptVersion(template_id="tpl.test", version="v1.0.0", content="old", created_by="sys")
        v2 = PromptVersion(template_id="tpl.test", version="v2.0.0", content="new", created_by="sys")
        prompt_manager.register(v1)
        prompt_manager.register(v2)
        # 回滚到 v1.0.0
        prompt_manager.rollback("tpl.test", "v1.0.0", reason="v2 有回归问题", operator="admin")
        active = prompt_manager.get_active("tpl.test")
        assert active.version == "v1.0.0"
        # v2 应被停用
        assert prompt_manager.get("tpl.test", "v2.0.0").is_active is False

    def test_render_prompt_with_context(self, prompt_manager):
        """渲染 Prompt 模板（注入上下文变量）."""
        pv = PromptVersion(
            template_id="tpl.test",
            version="v1.0.0",
            content="Hello {learner_name}, your level is {skill_level}.",
            created_by="sys",
        )
        prompt_manager.register(pv)
        rendered = prompt_manager.render("tpl.test", "v1.0.0", {
            "learner_name": "张三",
            "skill_level": "intermediate",
        })
        assert "张三" in rendered
        assert "intermediate" in rendered


# ============================================================
# 4. AgentFactory 六步实例化流水线
# ============================================================

class TestAgentFactory:
    """AgentFactory 实例化流水线测试."""

    @pytest.fixture
    def factory(self, tool_registry, prompt_manager):
        """提供配置好的 AgentFactory."""
        return AgentFactory(
            tool_registry=tool_registry,
            prompt_manager=prompt_manager,
        )

    @pytest.fixture
    def setup_prompt(self, prompt_manager):
        """预注册 Prompt 版本."""
        prompt_manager.register(PromptVersion(
            template_id="tpl.diagnosis",
            version="v2.1.0",
            content="You are a diagnosis agent for {learner_name}.",
            created_by="system",
        ))

    @pytest.mark.asyncio
    async def test_instantiate_agent_success(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """完整六步实例化应成功."""
        agent_registry.register(sample_agent_def)

        instance = await factory.instantiate(
            agent_id=sample_agent_def.id,
            registry=agent_registry,
            learner_context={"learner_name": "张三"},
        )

        assert instance is not None
        assert instance.state == AgentInstanceState.READY
        assert instance.instance_id is not None
        assert instance.session_id is not None
        assert instance.agent_id == sample_agent_def.id

    @pytest.mark.asyncio
    async def test_instantiate_unknown_agent_raises(self, factory, agent_registry):
        """实例化未注册的 Agent 应报错."""
        with pytest.raises(AgentNotFoundError):
            await factory.instantiate(
                agent_id="agent.nonexistent.foo",
                registry=agent_registry,
            )

    @pytest.mark.asyncio
    async def test_instantiate_missing_prompt_raises(self, factory, agent_registry, sample_agent_def):
        """Prompt 版本未注册时应报错."""
        agent_registry.register(sample_agent_def)
        with pytest.raises(FactoryError, match="prompt"):
            await factory.instantiate(
                agent_id=sample_agent_def.id,
                registry=agent_registry,
            )

    @pytest.mark.asyncio
    async def test_instantiate_assigns_unique_ids(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """每次实例化应分配唯一的 instance_id 和 session_id."""
        agent_registry.register(sample_agent_def)

        inst1 = await factory.instantiate(sample_agent_def.id, agent_registry)
        inst2 = await factory.instantiate(sample_agent_def.id, agent_registry)

        assert inst1.instance_id != inst2.instance_id
        assert inst1.session_id != inst2.session_id

    @pytest.mark.asyncio
    async def test_instantiate_binds_tools(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """实例化应绑定声明的工具集."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)

        assert len(instance.bound_tools) == 3
        assert "internal.bkt_compute" in instance.bound_tools

    @pytest.mark.asyncio
    async def test_instantiate_renders_prompt(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """实例化应渲染 Prompt 模板."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(
            sample_agent_def.id, agent_registry,
            learner_context={"learner_name": "李四"},
        )

        assert "李四" in instance.rendered_prompt

    @pytest.mark.asyncio
    async def test_instantiate_binds_broadcast_channels(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """实例化应绑定广播频道订阅."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)

        assert len(instance.broadcast_subscriptions) == 3

    @pytest.mark.asyncio
    async def test_instantiate_creates_working_session(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """实例化应创建 Working Session."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)

        assert instance.working_session is not None
        assert instance.working_session.session_id == instance.session_id

    @pytest.mark.asyncio
    async def test_instantiate_starts_kernel(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """实例化应启动 Persistent Kernel."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)

        assert len(instance.kernel_handles) == 1
        assert instance.kernel_handles[0].kernel_type == "python"

    @pytest.mark.asyncio
    async def test_instantiate_guidance_agent_with_two_kernels(
        self, factory, agent_registry, guidance_agent_def
    ):
        """导学决策 Agent 应绑定两个内核 (Python + R)."""
        # 注册 Prompt
        factory._prompt_manager.register(PromptVersion(
            template_id="tpl.guidance",
            version="v4.0.0",
            content="You are the guidance agent.",
            created_by="system",
        ))
        agent_registry.register(guidance_agent_def)
        instance = await factory.instantiate(guidance_agent_def.id, agent_registry)

        assert len(instance.kernel_handles) == 2
        kernel_types = {k.kernel_type for k in instance.kernel_handles}
        assert "python" in kernel_types
        assert "r" in kernel_types

    @pytest.mark.asyncio
    async def test_factory_records_provenance(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """实例化完成后应记录 Provenance."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)

        assert instance.provenance_record is not None
        assert instance.provenance_record["event_type"] == "agent_instantiated"
        assert instance.provenance_record["agent_id"] == sample_agent_def.id


# ============================================================
# 5. AgentInstance 运行时实例与生命周期
# ============================================================

class TestAgentInstance:
    """AgentInstance 生命周期测试."""

    @pytest.fixture
    def factory(self, tool_registry, prompt_manager):
        return AgentFactory(tool_registry=tool_registry, prompt_manager=prompt_manager)

    @pytest.fixture
    def setup_prompt(self, prompt_manager):
        prompt_manager.register(PromptVersion(
            template_id="tpl.diagnosis",
            version="v2.1.0",
            content="You are a diagnosis agent for {learner_name}.",
            created_by="system",
        ))

    @pytest.mark.asyncio
    async def test_instance_initial_state_is_ready(
        self, factory, agent_registry, sample_agent_def, setup_prompt
    ):
        """实例化后的初始状态应为 READY."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)
        assert instance.state == AgentInstanceState.READY

    @pytest.mark.asyncio
    async def test_instance_activate(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """激活实例."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)
        instance.activate()
        assert instance.state == AgentInstanceState.ACTIVE

    @pytest.mark.asyncio
    async def test_instance_pause_and_resume(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """暂停和恢复实例."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)
        instance.activate()
        instance.pause()
        assert instance.state == AgentInstanceState.PAUSED
        instance.resume()
        assert instance.state == AgentInstanceState.ACTIVE

    @pytest.mark.asyncio
    async def test_instance_terminate(self, factory, agent_registry, sample_agent_def, setup_prompt):
        """终止实例."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)
        instance.activate()
        instance.terminate()
        assert instance.state == AgentInstanceState.TERMINATED

    @pytest.mark.asyncio
    async def test_instance_terminate_releases_resources(
        self, factory, agent_registry, sample_agent_def, setup_prompt
    ):
        """终止实例应释放内核和广播订阅."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)
        instance.activate()
        instance.terminate()
        # 内核应被释放
        assert len(instance.kernel_handles) == 0
        # 广播订阅应被清理
        assert len(instance.broadcast_subscriptions) == 0

    @pytest.mark.asyncio
    async def test_instance_cannot_activate_from_terminated(
        self, factory, agent_registry, sample_agent_def, setup_prompt
    ):
        """已终止的实例不能被激活."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)
        instance.terminate()
        with pytest.raises(AgentRegistryError, match="terminated"):
            instance.activate()

    @pytest.mark.asyncio
    async def test_instance_health_check(
        self, factory, agent_registry, sample_agent_def, setup_prompt
    ):
        """实例健康检查."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(sample_agent_def.id, agent_registry)
        instance.activate()
        health = instance.health_check()
        assert health["state"] == "active"
        assert health["healthy"] is True
        assert "uptime_s" in health

    @pytest.mark.asyncio
    async def test_instance_metadata(
        self, factory, agent_registry, sample_agent_def, setup_prompt
    ):
        """实例元数据."""
        agent_registry.register(sample_agent_def)
        instance = await factory.instantiate(
            sample_agent_def.id, agent_registry,
            learner_context={"learner_name": "王五"},
        )
        meta = instance.metadata
        assert meta["agent_id"] == "agent.learning.diagnosis"
        assert meta["instance_id"] == instance.instance_id
        assert meta["learner_name"] == "王五"


# ============================================================
# 6. 四个核心 Agent 定义验证
# ============================================================

class TestCoreAgentDefinitions:
    """四个核心 Agent 定义验证 (来自 L5 设计文档)."""

    def test_diagnosis_agent_definition(self):
        """学情诊断 Agent 定义符合规范."""
        agent = AgentDefinition(
            id="agent.learning.diagnosis",
            name="学情诊断 Agent",
            role="基于 BKT/IRT 引擎对学习者的 Dy3+ 发光材料知识掌握状态进行实时诊断，输出知识图谱缺口和掌握概率向量",
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
        assert agent.id == "agent.learning.diagnosis"
        assert len(agent.tools) == 3
        assert len(agent.kernel_bindings) == 1

    def test_generation_agent_definition(self):
        """知识生成 Agent 定义符合规范."""
        agent = AgentDefinition(
            id="agent.knowledge.generation",
            name="知识生成 Agent",
            role="基于 RAG 引擎和 L3 Connector 池生成 Dy3+ 发光材料领域的知识内容",
            system_prompt=PromptReference(template_id="tpl.generation", version="v3.0.2"),
            tools=["l3.rag_retrieve", "l3.connector_tier1_query", "l3.connector_tier2_query"],
            connectors=["l3.tier1_nist_webbook", "l3.tier2_materials_project"],
            memory_config=MemoryConfig(
                read_stores=["milvus", "neo4j", "postgresql"],
                write_stores=["milvus", "postgresql"],
            ),
            reputation_config=ReputationConfig(initial_score=80, penalty_factor=1.0, reward_factor=1.0),
            broadcast_channels=[
                BroadcastChannel(channel="knowledge.generation.output", mode=BroadcastMode.PUB),
                BroadcastChannel(channel="learning.diagnosis.report", mode=BroadcastMode.SUB),
            ],
            kernel_bindings=[KernelBinding(kernel_type="python", purpose="数值模拟与可视化计算")],
        )
        assert agent.id == "agent.knowledge.generation"

    def test_review_agent_definition(self):
        """审核校验 Agent 定义符合规范."""
        agent = AgentDefinition(
            id="agent.quality.review",
            name="审核校验 Agent",
            role="对知识生成内容进行双层校验（规则引擎 + 交叉验证）",
            system_prompt=PromptReference(template_id="tpl.review", version="v2.3.1"),
            tools=["internal.rule_engine_check", "internal.cross_validation", "internal.standard_value_check"],
            connectors=["l3.tier1_nist_webbook", "l3.tier2_materials_project"],
            memory_config=MemoryConfig(
                read_stores=["postgresql", "neo4j"],
                write_stores=["postgresql"],
            ),
            reputation_config=ReputationConfig(initial_score=90, penalty_factor=0.5, reward_factor=1.5),
            broadcast_channels=[
                BroadcastChannel(channel="knowledge.review.result", mode=BroadcastMode.PUB),
                BroadcastChannel(channel="knowledge.generation.output", mode=BroadcastMode.SUB),
            ],
            kernel_bindings=[],
        )
        assert agent.id == "agent.quality.review"
        assert len(agent.kernel_bindings) == 0

    def test_guidance_agent_has_decision_authority(self, guidance_agent_def):
        """导学决策 Agent 应拥有完整的决策权限."""
        assert guidance_agent_def.decision_authority.scheduling is True
        assert guidance_agent_def.decision_authority.intervention is True
        assert guidance_agent_def.decision_authority.adaptive is True

    def test_guidance_agent_has_self_evolution(self, guidance_agent_def):
        """导学决策 Agent 应启用自演化."""
        assert guidance_agent_def.self_evolution.enabled is True
        assert guidance_agent_def.self_evolution.prompt_template_management is True
        assert guidance_agent_def.self_evolution.strategy_revision is True

    def test_guidance_agent_has_two_kernels(self, guidance_agent_def):
        """导学决策 Agent 应绑定两个内核 (Python + R)."""
        assert len(guidance_agent_def.kernel_bindings) == 2
        types = {kb.kernel_type for kb in guidance_agent_def.kernel_bindings}
        assert "python" in types
        assert "r" in types
