"""L5 默认 Agent 引导 — 四个核心 Agent 的运行时默认定义.

从 L5 设计文档与既有测试夹具沉淀为可注册的生产默认体:
学情诊断 / 知识生成 / 审核校验 / 导学决策。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("dy3_polaris.l5.default_agents")

from .agent_definition import (
    AgentDefinition,
    AgentFactory,
    AgentInstance,
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
from .agent_workers import AgentDependencies, build_agent_workers
from .interaction_recorder import (
    InteractionPhase,
    InteractionRecorder,
    get_recorder,
    set_recorder,
)

DECISION_AGENT_ID = "agent.guidance.decision"


def build_default_agents() -> list[AgentDefinition]:
    """返回四个核心 Agent 的默认定义."""
    return [
        AgentDefinition(
            id="agent.learning.diagnosis",
            name="学情诊断 Agent",
            role=(
                "基于 BKT/IRT 引擎对学习者知识掌握状态进行实时诊断，"
                "输出知识图谱缺口和掌握概率向量"
            ),
            system_prompt=PromptReference(
                template_id="tpl.diagnosis", version="v2.1.0"
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
                initial_score=85, penalty_factor=0.8, reward_factor=1.2
            ),
            broadcast_channels=[
                BroadcastChannel(
                    channel="learning.diagnosis.report", mode=BroadcastMode.PUB
                ),
                BroadcastChannel(
                    channel="learning.interaction.event", mode=BroadcastMode.SUB
                ),
                BroadcastChannel(
                    channel="learning.knowledge.gap", mode=BroadcastMode.PUBSUB
                ),
            ],
            kernel_bindings=[
                KernelBinding(
                    kernel_type="python",
                    purpose="BKT参数EM校准与遗忘曲线计算",
                )
            ],
        ),
        AgentDefinition(
            id="agent.knowledge.generation",
            name="知识生成 Agent",
            role=(
                "根据学情诊断结果，结合 L3 知识图谱生成个性化学习内容"
            ),
            system_prompt=PromptReference(
                template_id="tpl.generation", version="v3.0.2"
            ),
            tools=[
                "l3.rag_retrieve",
                "l3.connector_tier1_query",
                "l3.connector_tier2_query",
                "internal.canvas_generation",
            ],
            connectors=["l3.tier1_nist_webbook", "l3.tier2_materials_project"],
            memory_config=MemoryConfig(
                read_stores=["milvus", "neo4j", "postgresql"],
                write_stores=["milvus", "postgresql"],
            ),
            reputation_config=ReputationConfig(
                initial_score=80, penalty_factor=1.0, reward_factor=1.0
            ),
            broadcast_channels=[
                BroadcastChannel(
                    channel="learning.diagnosis.report", mode=BroadcastMode.SUB
                ),
                BroadcastChannel(
                    channel="learning.knowledge.gap", mode=BroadcastMode.SUB
                ),
                BroadcastChannel(
                    channel="knowledge.generation.output", mode=BroadcastMode.PUB
                ),
            ],
            kernel_bindings=[
                KernelBinding(
                    kernel_type="python",
                    purpose="知识图谱推理与内容生成",
                )
            ],
        ),
        AgentDefinition(
            id="agent.quality.review",
            name="审核校验 Agent",
            role=(
                "对知识生成内容进行双层校验（规则引擎 + 交叉验证），"
                "包含事实核查与幻觉检测"
            ),
            system_prompt=PromptReference(
                template_id="tpl.review", version="v2.3.1"
            ),
            tools=[
                "internal.rule_engine_check",
                "internal.cross_validation",
                "internal.standard_value_check",
            ],
            connectors=["l3.tier1_nist_webbook", "l3.tier2_materials_project"],
            memory_config=MemoryConfig(
                read_stores=["postgresql", "neo4j"],
                write_stores=["postgresql"],
            ),
            reputation_config=ReputationConfig(
                initial_score=90, penalty_factor=0.5, reward_factor=1.5
            ),
            broadcast_channels=[
                BroadcastChannel(
                    channel="knowledge.review.result", mode=BroadcastMode.PUB
                ),
                BroadcastChannel(
                    channel="knowledge.generation.output", mode=BroadcastMode.SUB
                ),
            ],
        ),
        AgentDefinition(
            id=DECISION_AGENT_ID,
            name="导学决策 Agent",
            role=(
                "系统决策中枢，整合诊断/生成/审核结果做出最优教学路径决策，"
                "并对不确定结果向提问者发起确认"
            ),
            system_prompt=PromptReference(
                template_id="tpl.guidance", version="v4.0.0"
            ),
            tools=[
                "internal.topology_analysis",
                "internal.path_simulation",
                "internal.uncertainty_confirm",
            ],
            connectors=["l3.knowledge_graph"],
            memory_config=MemoryConfig(
                read_stores=["milvus", "neo4j", "postgresql"],
                write_stores=["neo4j", "postgresql"],
            ),
            reputation_config=ReputationConfig(
                initial_score=88, penalty_factor=0.8, reward_factor=1.2
            ),
            broadcast_channels=[
                BroadcastChannel(
                    channel="learning.diagnosis.report", mode=BroadcastMode.SUB
                ),
                BroadcastChannel(
                    channel="knowledge.generation.output", mode=BroadcastMode.SUB
                ),
                BroadcastChannel(
                    channel="knowledge.review.result", mode=BroadcastMode.SUB
                ),
                BroadcastChannel(
                    channel="guidance.decision.command", mode=BroadcastMode.PUB
                ),
            ],
            kernel_bindings=[
                KernelBinding(kernel_type="python", purpose="教学路径模拟"),
                KernelBinding(kernel_type="r", purpose="统计检验"),
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
        ),
    ]


def build_default_prompt_manager() -> PromptVersionManager:
    """注册四个核心 Agent 的默认 Prompt 版本."""
    manager = PromptVersionManager()
    prompts = [
        PromptVersion(
            template_id="tpl.diagnosis",
            version="v2.1.0",
            content=(
                "你是学情诊断 Agent。基于 BKT/IRT 引擎实时诊断学习者状态，"
                "输出掌握概率、知识缺口与遗忘风险。"
            ),
            created_by="system",
        ),
        PromptVersion(
            template_id="tpl.generation",
            version="v3.0.2",
            content=(
                "你是知识生成 Agent。结合 L3 知识图谱与连接器，"
                "为学习者生成个性化、可追溯的学习内容。"
            ),
            created_by="system",
        ),
        PromptVersion(
            template_id="tpl.review",
            version="v2.3.1",
            content=(
                "你是审核校验 Agent。对知识内容执行事实核查、"
                "交叉验证与标准值校验，识别幻觉与冲突。"
            ),
            created_by="system",
        ),
        PromptVersion(
            template_id="tpl.guidance",
            version="v4.0.0",
            content=(
                "你是导学决策 Agent，也是系统决策中枢。整合学情诊断、"
                "知识生成与审核结果，选择最优教学路径；当置信度不足或"
                "验证出现风险时，必须向提问者发起确认后再输出。"
            ),
            created_by="system",
        ),
    ]
    for prompt in prompts:
        manager.register(prompt)
    return manager


class AgentRuntime:
    """默认 Agent 运行时 — 注册表 + 工厂 + 懒实例化 + 广播订阅 + 交互记录."""

    def __init__(
        self,
        registry: AgentRegistry,
        factory: AgentFactory,
        workers: dict[str, Any] | None = None,
        message_bus: Any | None = None,
        recorder: InteractionRecorder | None = None,
    ) -> None:
        self._registry = registry
        self._factory = factory
        self._workers = workers or {}
        self._instances: dict[str, AgentInstance] = {}
        self._message_bus = message_bus
        # 广播订阅: agent_id -> inbox 消息列表 (按需协作消费)
        self._inboxes: dict[str, list[dict[str, Any]]] = {}
        self._subscriptions: list[Any] = []
        self._subscribed = False
        # 交互记录器
        self._recorder = recorder or get_recorder()
        set_recorder(self._recorder)

    def bind_message_bus(self, message_bus: Any) -> None:
        """绑定消息总线并注册各 Agent 的广播订阅 (SUB 频道)."""
        self._message_bus = message_bus
        if message_bus is None or self._subscribed:
            return
        try:
            from dy3_polaris.l5.communication import Message

            for definition in self._registry.list_all():
                agent_id = definition.id
                self._inboxes.setdefault(agent_id, [])
                for channel in definition.broadcast_channels:
                    if channel.mode.value not in ("sub", "pubsub"):
                        continue
                    sub = message_bus.subscribe(
                        channel.channel,
                        agent_id,
                        self._make_inbox_callback(agent_id),
                    )
                    self._subscriptions.append(sub)
            self._subscribed = True
            logger.info("Agent 广播订阅完成: %d 个 Agent 订阅频道", len(self._subscriptions))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent 广播订阅失败: %s", exc)

    def _make_inbox_callback(self, agent_id: str):
        from dy3_polaris.l5.communication import Message

        def callback(msg: Message) -> None:
            inbox = self._inboxes.setdefault(agent_id, [])
            inbox.append({
                "channel": msg.channel,
                "publisher": msg.publisher,
                "payload": msg.payload,
                "timestamp": msg.timestamp,
                "message_id": msg.message_id,
            })
            if len(inbox) > 200:
                del inbox[: len(inbox) - 200]

        return callback

    def get_inbox(self, agent_id: str) -> list[dict[str, Any]]:
        """返回 Agent 收到的广播消息 (按需协作消费)."""
        return list(self._inboxes.get(agent_id, []))

    def clear_inbox(self, agent_id: str) -> int:
        """清空 Agent inbox, 返回清除条数."""
        inbox = self._inboxes.setdefault(agent_id, [])
        n = len(inbox)
        inbox.clear()
        return n

    @property
    def registry(self) -> AgentRegistry:
        """返回 Agent 注册中心."""
        return self._registry

    @property
    def factory(self) -> AgentFactory:
        """返回 Agent 工厂."""
        return self._factory

    async def ensure_instances(
        self, agent_ids: list[str] | None = None
    ) -> dict[str, AgentInstance]:
        """为已注册 Agent 创建并激活运行时实例（幂等）."""
        ids = agent_ids or [agent.id for agent in self._registry.list_all()]
        for agent_id in ids:
            if agent_id in self._instances:
                continue
            instance = await self._factory.instantiate(
                agent_id,
                self._registry,
                learner_context={"learner_id": "demo-learner"},
            )
            instance.activate()
            self._instances[agent_id] = instance
        return dict(self._instances)

    async def list_status(self) -> dict[str, Any]:
        """返回注册 Agent 与实例状态列表."""
        await self.ensure_instances()
        agents: list[dict[str, Any]] = []
        for definition in self._registry.list_all():
            item = definition.to_dict()
            instance = self._instances.get(definition.id)
            item["instance"] = instance.health_check() if instance else None
            agents.append(item)
        return {
            "total": len(agents),
            "decision_agent": DECISION_AGENT_ID,
            "agents": agents,
        }

    def get_instance(self, agent_id: str) -> AgentInstance | None:
        """按 Agent ID 获取已创建实例."""
        return self._instances.get(agent_id)

    def get_recorder(self) -> InteractionRecorder:
        """获取交互记录器实例."""
        return self._recorder

    async def run(
        self,
        agent_id: str,
        input_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行指定 Agent 的 worker 逻辑，并记录交互."""
        # 查找 Agent 定义
        definition = self._registry.get(agent_id)
        agent_name = definition.name if definition else agent_id.split(".")[-1]

        # 确定阶段
        phase = self._detect_phase(agent_id)

        # 记录开始
        start = time.time()
        self._recorder.start_chain(
            learner_id=str(input_data.get("learner_id", "")),
            query=str(input_data.get("query", input_data.get("learner_id", ""))),
        )
        self._recorder.record_agent_execution(
            agent_id=agent_id,
            agent_name=agent_name,
            action=f"Agent 执行: {agent_name}",
            input_data=input_data or {},
            status="running",
            phase=phase,
        )

        # 执行
        worker = self._workers.get(agent_id)
        if worker is None:
            self._recorder.record_agent_execution(
                agent_id=agent_id,
                agent_name=agent_name,
                action=f"Agent 执行失败: 未注册 worker",
                status="failed",
                phase=phase,
            )
            raise ValueError(f"Agent 未注册执行 worker: {agent_id}")

        try:
            result = worker(input_data or {})
            if asyncio.iscoroutine(result):
                result = await result

            # 记录完成
            elapsed = (time.time() - start) * 1000
            self._recorder.record_agent_execution(
                agent_id=agent_id,
                agent_name=agent_name,
                action=f"Agent 完成: {agent_name}",
                input_data=input_data or {},
                output_data=result if isinstance(result, dict) else {"result": result},
                duration_ms=elapsed,
                status="completed",
                phase=phase,
            )
            self._recorder.end_chain(
                final_answer=str(result.get("summary", result.get("answer", ""))),
            )
            return result
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            self._recorder.record_agent_execution(
                agent_id=agent_id,
                agent_name=agent_name,
                action=f"Agent 异常: {agent_name}",
                status="failed",
                duration_ms=elapsed,
                phase=phase,
            )
            self._recorder.end_chain(status="failed")
            raise

    def _detect_phase(self, agent_id: str) -> InteractionPhase:
        """根据 Agent ID 检测交互阶段."""
        if "diagnosis" in agent_id:
            return InteractionPhase.DIAGNOSIS
        if "generation" in agent_id:
            return InteractionPhase.GENERATION
        if "review" in agent_id:
            return InteractionPhase.REVIEW
        if "decision" in agent_id:
            return InteractionPhase.DECISION
        return InteractionPhase.SYSTEM


def build_default_agent_runtime(
    dependencies: AgentDependencies | None = None,
    message_bus: Any | None = None,
    recorder: InteractionRecorder | None = None,
) -> AgentRuntime:
    """构建注册四个核心 Agent 的默认运行时."""
    registry = AgentRegistry()
    for definition in build_default_agents():
        registry.register(definition)
    factory = AgentFactory(prompt_manager=build_default_prompt_manager())
    workers = build_agent_workers(dependencies)
    runtime = AgentRuntime(
        registry=registry,
        factory=factory,
        workers=workers,
        message_bus=message_bus,
        recorder=recorder,
    )
    runtime.bind_message_bus(message_bus)
    return runtime


__all__ = [
    "DECISION_AGENT_ID",
    "AgentRuntime",
    "build_default_agent_runtime",
    "build_default_agents",
    "build_default_prompt_manager",
]
