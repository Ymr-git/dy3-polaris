"""跨层集成桥接 — L2↔L4↔L5 事件驱动通信与状态同步.

融合世界先进方案的集成架构:
- Knewton SOA: 服务编排 + 事件驱动 + 松耦合
- ALEKS KST: 知识状态 → 决策上下文传递
- Duolingo EDA: 事件驱动异步通信 + 消息总线
- xAPI/Caliper: 标准化学习事件格式
- LangGraph: 跨节点状态传递 + Channel 通信
- Temporal: Signal/Query 跨工作流通信

核心职责:
1. assemble_decision_context: 从 L2 画像/记忆组装 L4 决策上下文
2. process_query_with_profile: 端到端查询处理 (画像 → 决策 → 输出)
3. publish_learner_event: 跨层事件发布到消息总线
4. feedback_to_profile: L5 Agent 执行结果反馈到 L2 画像
5. get_cross_layer_health: 跨层健康检查聚合

使用示例::

    from dy3_polaris.l5.integration_bridge import IntegrationBridge

    bridge = IntegrationBridge(
        irt_service=irt_service,
        profile_service=profile_service,
        memory_service=memory_service,
        decision_engine=decision_engine,
        orchestration_engine=orchestration_engine,
        session_manager=session_manager,
        message_bus=message_bus,
    )

    # 组装决策上下文
    context = bridge.assemble_decision_context(learner_id="learner1")

    # 端到端查询处理
    result = await bridge.process_query_with_profile(
        learner_id="learner1",
        query="Dy3+ 的激发态波长是多少？",
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from dy3_polaris.l2.models import ProfileConflictError

logger = logging.getLogger(__name__)


class IntegrationBridge:
    """跨层集成桥接器 — 连接 L2 个性化层、L4 决策引擎层、L5 Agent 运行时层.

    融合世界先进方案的集成架构:
    - Knewton SOA: 服务编排 + 事件驱动 + 松耦合
    - Duolingo EDA: 事件驱动异步通信
    - LangGraph: 跨节点状态传递
    - xAPI/Caliper: 标准化学习事件

    核心数据流:
    1. L2 → L4: 画像/记忆/能力数据传入决策引擎作为上下文
    2. L4 → L5: 决策结果触发 Agent 编排
    3. L5 → L2: Agent 执行结果反馈到画像/记忆
    4. 跨层事件: 通过 MessageBus 发布订阅实现异步通信

    Args:
        irt_service: L2 IRT 能力评估服务.
        profile_service: L2 画像构建服务.
        memory_service: L2 记忆管理服务.
        decision_engine: L4 决策引擎.
        orchestration_engine: L5 编排引擎.
        session_manager: L5 会话管理器.
        message_bus: L5 消息总线.
    """

    def __init__(
        self,
        irt_service: Any | None = None,
        profile_service: Any | None = None,
        memory_service: Any | None = None,
        decision_engine: Any | None = None,
        orchestration_engine: Any | None = None,
        session_manager: Any | None = None,
        message_bus: Any | None = None,
        governance_router: Any | None = None,
        l1_router: Any | None = None,
        l3_router: Any | None = None,
        l6_router: Any | None = None,
    ) -> None:
        """初始化跨层集成桥接器.

        Args:
            irt_service: L2 IRT 能力评估服务.
            profile_service: L2 画像构建服务.
            memory_service: L2 记忆管理服务.
            decision_engine: L4 决策引擎.
            orchestration_engine: L5 编排引擎.
            session_manager: L5 会话管理器.
            message_bus: L5 消息总线.
            governance_router: L0 治理路由器 (可选).
            l1_router: L1 用户域路由器 (可选).
            l3_router: L3 知识层路由器 (可选).
            l6_router: L6 协议基础设施路由器 (可选).
        """
        self.irt_service = irt_service
        self.profile_service = profile_service
        self.memory_service = memory_service
        self.decision_engine = decision_engine
        self.orchestration_engine = orchestration_engine
        self.session_manager = session_manager
        self.message_bus = message_bus

        # 扩展层引用 (L0/L1/L3/L6)
        self.governance_router = governance_router
        self.l1_router = l1_router
        self.l3_router = l3_router
        self.l6_router = l6_router

        # 事件频道名称 (xAPI/Caliper 标准化)
        self._channel_learner_events = "learner_events"
        self._channel_decision_events = "decision_events"
        self._channel_agent_events = "agent_events"

        # 初始化频道 (如果消息总线可用)
        if self.message_bus is not None:
            for ch in [
                self._channel_learner_events,
                self._channel_decision_events,
                self._channel_agent_events,
            ]:
                try:
                    if hasattr(self.message_bus, "create_channel"):
                        self.message_bus.create_channel(ch)
                except Exception:
                    logger.debug("频道 %s 可能已存在", ch)

    # ============================================================
    # L2 → L4: 决策上下文组装
    # ============================================================

    def assemble_decision_context(self, learner_id: str) -> dict[str, Any]:
        """从 L2 画像/记忆/能力数据组装 L4 决策上下文.

        融合 Knewton 式上下文信封: 将 L2 全部学习者状态打包为
        决策引擎可消费的标准化上下文.

        Args:
            learner_id: 学习者 ID.

        Returns:
            决策上下文字典, 包含:
            - learner_id: 学习者 ID
            - theta: IRT 能力值
            - level: 能力等级
            - kp_mastery: 知识点掌握度映射
            - weak_kps: 薄弱知识点列表
            - confidence: 画像置信度
            - memory_state: 记忆状态
            - timestamp: 上下文组装时间戳
        """
        context: dict[str, Any] = {
            "learner_id": learner_id,
            "timestamp": time.time(),
        }

        # 从画像服务获取画像快照
        if self.profile_service is not None:
            try:
                snapshot = self.profile_service.get_profile_snapshot(learner_id)
                if snapshot is not None:
                    if hasattr(snapshot, "to_dict"):
                        profile_data = snapshot.to_dict()
                    elif isinstance(snapshot, dict):
                        profile_data = snapshot
                    else:
                        profile_data = {}

                    context["theta"] = profile_data.get("theta", 0.0)
                    context["level"] = profile_data.get("level", "beginner")
                    context["kp_mastery"] = profile_data.get("kp_mastery", {})
                    context["weak_kps"] = profile_data.get("weak_kps", [])
                    context["confidence"] = profile_data.get("confidence", 0.0)
                    context["learning_style"] = profile_data.get("learning_style", "multimodal")
                    context["bloom_target"] = profile_data.get("bloom_target", "understand")
                else:
                    context["theta"] = 0.0
                    context["level"] = "beginner"
                    context["kp_mastery"] = {}
                    context["weak_kps"] = []
                    context["confidence"] = 0.0
            except Exception as e:
                logger.warning("获取画像快照失败: %s", e)
                context["theta"] = 0.0
                context["level"] = "beginner"
                context["kp_mastery"] = {}
                context["weak_kps"] = []
                context["confidence"] = 0.0
        else:
            context["theta"] = 0.0
            context["level"] = "beginner"
            context["kp_mastery"] = {}
            context["weak_kps"] = []
            context["confidence"] = 0.0

        # 从 IRT 服务获取能力快照
        if self.irt_service is not None:
            try:
                ability = self.irt_service.get_ability_snapshot(learner_id)
                if ability and isinstance(ability, dict):
                    # 如果画像中没有 theta, 用 IRT 的
                    if "theta" not in context or context.get("theta", 0.0) == 0.0:
                        context["theta"] = ability.get("theta", 0.0)
                    context["se"] = ability.get("se", 0.5)
                    context["response_count"] = ability.get("response_count", 0)
            except Exception as e:
                logger.debug("获取 IRT 快照失败: %s", e)

        # 从记忆服务获取记忆状态
        if self.memory_service is not None:
            try:
                memory_state = self.memory_service.get_memory_state(learner_id)
                if memory_state is None:
                    context["memory_state"] = {"kp_retentions": {}, "stale_kps": []}
                elif isinstance(memory_state, dict):
                    context["memory_state"] = memory_state
                else:
                    context["memory_state"] = {"kp_retentions": {}, "stale_kps": []}
            except Exception as e:
                logger.debug("获取记忆状态失败: %s", e)
                context["memory_state"] = {"kp_retentions": {}, "stale_kps": []}
        else:
            context["memory_state"] = {"kp_retentions": {}, "stale_kps": []}

        return context

    # ============================================================
    # L2 → L4 → 输出: 端到端查询处理
    # ============================================================

    async def process_query_with_profile(
        self,
        learner_id: str,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """端到端查询处理: 画像 → 决策引擎 → 输出.

        融合 Knewton 三引擎架构: 评估 → 策略 → 反馈.
        将 L2 画像数据作为上下文传入 L4 决策引擎, 获取行动记录.

        Args:
            learner_id: 学习者 ID.
            query: 用户查询文本.
            **kwargs: 额外参数 (context_id, query_vector 等).

        Returns:
            行动记录字典, 包含:
            - action_type: 行动类型
            - confidence: 置信度
            - response_payload: 响应载荷
            - plan_id: 计划 ID
            - learner_profile: 使用的画像上下文
        """
        # 1. 组装决策上下文 (L2 → L4)
        context = self.assemble_decision_context(learner_id)

        # 2. 调用决策引擎处理查询 (L4)
        if self.decision_engine is None:
            return {
                "action_type": "fallback",
                "confidence": 0.0,
                "response_payload": {"error": "决策引擎不可用"},
                "plan_id": "",
                "learner_profile": context,
            }

        context_id = kwargs.get("context_id", str(uuid.uuid4()))

        action_record = await self.decision_engine.process_query(
            query=query,
            context_id=context_id,
            learner_profile=context,
        )

        # 3. 序列化行动记录
        if hasattr(action_record, "to_dict"):
            result = action_record.to_dict()
        elif hasattr(action_record, "__dict__"):
            result = {}
            for k, v in action_record.__dict__.items():
                if not k.startswith("_"):
                    if hasattr(v, "value"):
                        result[k] = v.value
                    else:
                        result[k] = v
        else:
            result = {"action_type": str(action_record)}

        result["learner_profile"] = context

        # 4. 发布决策事件 (跨层事件驱动)
        self.publish_learner_event(
            learner_id=learner_id,
            event_type="query_processed",
            payload={
                "query": query[:200],
                "action_type": result.get("action_type", "unknown"),
                "confidence": result.get("confidence", 0.0),
            },
        )

        return result

    # ============================================================
    # 跨层事件驱动通信
    # ============================================================

    def publish_learner_event(
        self,
        learner_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> str | None:
        """发布学习者事件到消息总线 (跨层事件驱动).

        融合 xAPI/Caliper 标准化事件格式:
        - actor: learner_id
        - verb: event_type
        - object: payload

        Args:
            learner_id: 学习者 ID.
            event_type: 事件类型 (answer_submitted/query_processed 等).
            payload: 事件载荷.

        Returns:
            消息 ID, 消息总线不可用时返回 None.
        """
        if self.message_bus is None:
            logger.debug("消息总线不可用, 事件未发布: %s/%s", learner_id, event_type)
            return None

        # xAPI 标准化事件格式
        event = {
            "actor": learner_id,
            "verb": event_type,
            "object": payload or {},
            "timestamp": time.time(),
            "event_id": str(uuid.uuid4()),
        }

        try:
            from dy3_polaris.l5.communication import Message

            message = Message(
                channel=self._channel_learner_events,
                publisher="l5.integration_bridge",
                payload=event,
            )
            self.message_bus.publish(message)
            return message.message_id
        except Exception as e:
            logger.warning("事件发布失败: %s", e)
            return None

    # ============================================================
    # L5 → L2: Agent 执行结果反馈到画像
    # ============================================================

    def feedback_to_profile(
        self,
        learner_id: str,
        agent_result: dict[str, Any],
    ) -> None:
        """将 L5 Agent 执行结果反馈到 L2 画像系统.

        融合 Duolingo 式反馈闭环: Agent 执行结果 → 画像更新.
        根据 Agent 执行的置信度和行动类型, 生成隐式反馈信号.

        Args:
            learner_id: 学习者 ID.
            agent_result: Agent 执行结果, 包含:
                - action_type: 行动类型
                - confidence: 置信度
                - response_payload: 响应载荷
        """
        # 发布 Agent 事件到消息总线
        if self.message_bus is not None:
            try:
                from dy3_polaris.l5.communication import Message

                self.message_bus.publish(Message(
                    channel=self._channel_agent_events,
                    publisher="l5.integration_bridge",
                    payload={
                        "learner_id": learner_id,
                        "action_type": agent_result.get("action_type", "unknown"),
                        "confidence": agent_result.get("confidence", 0.0),
                        "timestamp": time.time(),
                    },
                ))
            except Exception as e:
                logger.debug("Agent 事件发布失败: %s", e)

        # 行为反馈 → 画像: 隐式反馈写回 extras.feedback_log (走 L2 唯一写方 + 乐观锁)
        if self.profile_service is not None:
            try:
                profile = self.profile_service.get_profile_snapshot(learner_id)
                if profile is not None:
                    extras = dict(getattr(profile, "extras", {}) or {})
                    fb_log = list(extras.get("feedback_log", []) or [])
                    fb_log.append({
                        "ts": time.time(),
                        "source": "agent",
                        # 统一反馈类型 (l2.models.FeedbackType.AGENT_OUTCOME)
                        "feedback_type": "agent_outcome",
                        "rating": float(agent_result.get("confidence", 0.5)),
                        "action_type": agent_result.get("action_type", "respond"),
                    })
                    updates = {"extras": dict(extras, feedback_log=fb_log[-50:])}
                    try:
                        self.profile_service.apply_update(
                            learner_id,
                            updates=updates,
                            expected_version=profile.version,
                        )
                    except ProfileConflictError:
                        # 乐观锁冲突: 重拉最新后重试一次
                        latest = self.profile_service.get_profile_snapshot(learner_id)
                        if latest is not None:
                            self.profile_service.apply_update(
                                learner_id,
                                updates=updates,
                                expected_version=latest.version,
                            )
            except Exception as e:
                logger.debug("画像反馈记录失败: %s", e)

    # ============================================================
    # 跨层健康检查
    # ============================================================

    def get_cross_layer_health(self) -> dict[str, dict[str, Any]]:
        """跨层健康检查聚合 (真实探活版).

        对每层执行功能性探针 (非纯存在性检查), 记录延迟:
        - L1: JWT 签发→验签往返
        - L2: IRT/画像/记忆服务可用性
        - L3: 知识存储存在性 + 数据计数
        - L4: 决策引擎存在性
        - L5: 编排/会话/消息总线 + 频道计数
        - L6: 引擎 + 工具注册表工具数
        - L0/L7: 路由器存在性

        Returns:
            各层健康状态字典 (status/services; 每服务含 latency_ms 与 probe).
        """
        import time

        health: dict[str, dict[str, Any]] = {}

        def _probe(label: str, fn) -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                result = fn()
                ok = result is not False
                detail = result if isinstance(result, str) else "ok"
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail = str(exc)[:60]
            return {
                "state": "available" if ok else "unavailable",
                "probe": detail,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
            }

        def _layer(router_key: str | None, services: dict[str, dict[str, Any]]) -> dict[str, Any]:
            ok = all(s["state"] == "available" for s in services.values())
            return {
                "status": "healthy" if ok else "degraded",
                "services": services,
            }

        # L0 治理
        if self.governance_router is not None:
            health["l0"] = _layer("l0", {
                "governance_router": _probe("governance", lambda: True),
            })
        else:
            health["l0"] = _layer("l0", {
                "governance_router": {"state": "unavailable", "probe": "not mounted", "latency_ms": 0.0},
            })

        # L1 认证 (JWT 验签探针: 无效 token 必须被拒)
        jwt = getattr(getattr(self, "l1_router", None), "_jwt", None)
        if jwt is not None:
            def _jwt_probe() -> Any:
                try:
                    jwt.verify_token("invalid.token.value")
                    return False  # 无效 token 被接受 → 探针失败
                except Exception:
                    return "verifier-ok"  # 预期拒绝 → 验签路径正常

            health["l1"] = _layer("l1", {"jwt_verifier": _probe("jwt", _jwt_probe)})
        else:
            health["l1"] = _layer("l1", {
                "jwt_verifier": {"state": "unavailable", "probe": "jwt missing", "latency_ms": 0.0},
            })

        # L2 学习域
        l2_services = {
            "irt": _probe("irt", lambda: self.irt_service is not None),
            "profile": _probe("profile", lambda: self.profile_service is not None),
            "memory": _probe("memory", lambda: self.memory_service is not None),
        }
        health["l2"] = _layer("l2", l2_services)

        # L3 知识层 (存储 + 计数)
        if self.l3_router is not None:
            def _l3_probe() -> Any:
                store = getattr(self.l3_router, "_store", None)
                if store is None:
                    return False
                count = store.count() if hasattr(store, "count") else len(getattr(store, "entities", {}) or {})
                return f"entities={count}"

            health["l3"] = _layer("l3", {"knowledge_store": _probe("store", _l3_probe)})
        else:
            health["l3"] = _layer("l3", {
                "knowledge_store": {"state": "unavailable", "probe": "not mounted", "latency_ms": 0.0},
            })

        # L4 决策引擎
        if self.decision_engine is not None:
            health["l4"] = _layer("l4", {
                "decision_engine": _probe("engine", lambda: self.decision_engine is not None),
            })
        else:
            health["l4"] = _layer("l4", {
                "decision_engine": {"state": "unavailable", "probe": "not mounted", "latency_ms": 0.0},
            })

        # L5 编排/会话/总线
        l5_services = {
            "orchestration": _probe("orch", lambda: self.orchestration_engine is not None),
            "session": _probe("session", lambda: self.session_manager is not None),
        }
        if self.message_bus is not None:
            def _bus_probe() -> Any:
                chans = len(getattr(self.message_bus, "channels", {}) or {})
                return f"channels={chans}"

            l5_services["message_bus"] = _probe("bus", _bus_probe)
        else:
            l5_services["message_bus"] = {"state": "unavailable", "probe": "not mounted", "latency_ms": 0.0}
        health["l5"] = _layer("l5", l5_services)

        # L6 协议引擎 (工具注册表)
        if self.l6_router is not None:
            def _l6_probe() -> Any:
                engine = getattr(self.l6_router, "_engine", None)
                if engine is None or getattr(engine, "tool_registry", None) is None:
                    return False
                tools = getattr(engine.tool_registry, "_tools", {}) or {}
                return f"tools={len(tools)}"

            health["l6"] = _layer("l6", {"tool_registry": _probe("tools", _l6_probe)})
        else:
            health["l6"] = _layer("l6", {
                "tool_registry": {"state": "unavailable", "probe": "not mounted", "latency_ms": 0.0},
            })

        # L7 体验层
        l7_router = getattr(self, "_l7_router", None)
        if l7_router is not None:
            health["l7"] = _layer("l7", {"l7_router": _probe("l7", lambda: True)})
        else:
            health["l7"] = _layer("l7", {
                "l7_router": {"state": "unavailable", "probe": "not mounted", "latency_ms": 0.0},
            })

        return health


__all__ = ["IntegrationBridge"]
