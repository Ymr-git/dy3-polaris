"""CC2 人机协作引擎.

融合八大世界级方案的核心编排引擎：
- REACT 五维评分 → 动态自主等级分配
- LangGraph 式 interrupt（创建干预请求）+ Command resume（响应后恢复）
- AutoGen 式 max_consecutive_auto_reply 渐进自主
- Swarm 式 escalate_to_human 紧急升级
- GAIA 三阶段协商协议
- Chaos Engineering 混沌感知升级
- 持续质量评估自动降级

引擎作为 CC2 层的唯一入口，所有 Agent 的人机协作交互
均通过此引擎进行。
"""

from __future__ import annotations

import logging
from typing import Any

from .models import (
    AgentCollaborationProfile,
    CollaborationConfig,
    CollaborationMode,
    HumanDecision,
    HumanResponse,
    InterventionRecord,
    InterventionRequest,
    InterventionStatus,
    InterventionType,
    ModeSwitchEvent,
    NegotiationSession,
    REACTScore,
    ReviewOutcome,
    SwitchTrigger,
)
from .exceptions import (
    CC2Error,
    EscalationTargetError,
    InterventionConflictError,
    InterventionTimeoutError,
    ModeSwitchError,
    NegotiationExhaustedError,
    ProfileNotFoundError,
)

logger = logging.getLogger(__name__)

# 协作模式的数值顺序，用于相邻级切换校验
_MODE_ORDER: dict[CollaborationMode, int] = {
    CollaborationMode.AUTONOMOUS: 0,
    CollaborationMode.MONITORED: 1,
    CollaborationMode.CONDITIONAL: 2,
    CollaborationMode.SUPERVISED: 3,
}


class CollaborationEngine:
    """人机协作引擎.

    CC2 层的核心编排器，管理 Agent 协作配置、
    干预请求生命周期、模式动态切换和协商会话。

    使用示例::

        engine = CollaborationEngine()
        engine.register_profile(AgentCollaborationProfile(agent_id="tutor"))

        # Agent 请求人类审批（LangGraph interrupt 启发）
        request = engine.create_intervention(
            agent_id="tutor",
            intervention_type=InterventionType.CHECKPOINT,
            reason="评分决策需教师确认",
            payload={"student_answer": "...", "score": 85},
            proposed_action="提交评分",
            confidence=0.75,
        )

        # 人类响应（LangGraph Command resume 启发）
        record = engine.respond_to_intervention(
            request_id=request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
            feedback="评分合理",
        )
    """

    def __init__(self, config: CollaborationConfig | None = None) -> None:
        self.config = config or CollaborationConfig()
        self._profiles: dict[str, AgentCollaborationProfile] = {}
        self._interventions: dict[str, InterventionRecord] = {}
        self._negotiations: dict[str, NegotiationSession] = {}
        self._switch_events: list[ModeSwitchEvent] = []
        self._lock = __import__("threading").RLock()

        # 统计
        self._total_interventions = 0
        self._resolved_count = 0
        self._expired_count = 0
        self._escalation_count = 0
        self._negotiation_count = 0
        self._switch_count = 0

    # ==========================================================
    # Agent 协作配置管理
    # ==========================================================

    def register_profile(self, profile: AgentCollaborationProfile) -> None:
        """注册 Agent 协作配置."""
        with self._lock:
            self._profiles[profile.agent_id] = profile
            logger.info("注册协作配置: agent=%s, mode=%s", profile.agent_id, profile.mode.value)

    def get_profile(self, agent_id: str) -> AgentCollaborationProfile:
        """获取 Agent 协作配置."""
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None:
                raise ProfileNotFoundError(agent_id)
            return profile

    def update_profile(self, agent_id: str, **kwargs: Any) -> AgentCollaborationProfile:
        """更新 Agent 协作配置字段."""
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None:
                raise ProfileNotFoundError(agent_id)
            for k, v in kwargs.items():
                if hasattr(profile, k):
                    setattr(profile, k, v)
            return profile

    def list_profiles(self) -> list[AgentCollaborationProfile]:
        """列出所有已注册的协作配置."""
        with self._lock:
            return list(self._profiles.values())

    # ==========================================================
    # REACT 评分 → 模式映射
    # ==========================================================

    def evaluate_react(self, agent_id: str, score: REACTScore) -> CollaborationMode:
        """根据 REACT 评分计算建议模式.

        Returns:
            REACT 评分映射的建议协作模式
        """
        return score.to_mode()

    # ==========================================================
    # 模式切换（相邻级约束）
    # ==========================================================

    def switch_mode(
        self,
        agent_id: str,
        to_mode: CollaborationMode,
        trigger: SwitchTrigger,
        reason: str = "",
        react_score: REACTScore | None = None,
        confidence: float | None = None,
        allow_skip: bool = False,
    ) -> ModeSwitchEvent:
        """切换 Agent 协作模式.

        默认仅允许相邻级切换（AutoGen 渐进自主启发）。
        设置 allow_skip=True 可跳级（如混沌感知紧急升级）。

        Args:
            agent_id: Agent ID
            to_mode: 目标模式
            trigger: 触发条件
            reason: 切换原因
            react_score: 触发时的 REACT 评分
            confidence: 触发时的 Agent 置信度
            allow_skip: 是否允许跳级切换

        Returns:
            模式切换事件

        Raises:
            ProfileNotFoundError: Agent 未注册
            ModeSwitchError: 非法切换
          """
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None:
                raise ProfileNotFoundError(agent_id)

            from_mode = profile.mode
            if from_mode == to_mode:
                # 同模式不产生事件
                return ModeSwitchEvent(
                    agent_id=agent_id,
                    from_mode=from_mode,
                    to_mode=to_mode,
                    trigger=trigger,
                    reason="无变化" if not reason else reason,
                    react_score=react_score,
                    confidence_at_time=confidence,
                )

            # 相邻级约束
            if not allow_skip:
                from_order = _MODE_ORDER[from_mode]
                to_order = _MODE_ORDER[to_mode]
                if abs(to_order - from_order) > 1:
                    raise ModeSwitchError(
                        agent_id, from_mode.value, to_mode.value,
                        reason=f"不允许跳级切换 (order {from_order}→{to_order})",
                    )

            # 执行切换
            profile.mode = to_mode
            event = ModeSwitchEvent(
                agent_id=agent_id,
                from_mode=from_mode,
                to_mode=to_mode,
                trigger=trigger,
                reason=reason,
                react_score=react_score,
                confidence_at_time=confidence,
            )
            self._switch_events.append(event)
            self._switch_count += 1

            logger.info(
                "模式切换: agent=%s %s→%s trigger=%s",
                agent_id, from_mode.value, to_mode.value, trigger.value,
            )
            return event

    # ==========================================================
    # 干预请求（LangGraph interrupt 启发）
    # ==========================================================

    def create_intervention(
        self,
        agent_id: str,
        intervention_type: InterventionType | str = InterventionType.CHECKPOINT,
        reason: str = "",
        payload: dict[str, Any] | None = None,
        proposed_action: str = "",
        confidence: float = 0.5,
        context: dict[str, Any] | None = None,
        priority: int = 50,
    ) -> InterventionRecord:
        """创建干预请求（LangGraph interrupt 启发）.

        在关键决策点暂停 Agent 执行，生成待审核的干预请求。

        Args:
            agent_id: 请求干预的 Agent ID
            intervention_type: 干预类型（支持字符串和枚举）
            reason: 干预原因
            payload: 审核载荷
            proposed_action: Agent 提议的动作
            confidence: Agent 置信度
            context: 额外上下文
            priority: 优先级 0-100

        Returns:
            干预记录（状态为 PENDING）
        """
        # G6 路由层兼容: 字符串转换为枚举
        if isinstance(intervention_type, str):
            intervention_type = InterventionType(intervention_type)
        with self._lock:
            request = InterventionRequest(
                agent_id=agent_id,
                intervention_type=intervention_type,
                reason=reason,
                payload=payload or {},
                proposed_action=proposed_action,
                confidence=confidence,
                context=context or {},
                priority=priority,
                timeout_seconds=self.config.intervention_timeout_seconds,
            )
            record = InterventionRecord(request=request)
            self._interventions[request.request_id] = record
            self._total_interventions += 1

            logger.info(
                "创建干预: request=%s agent=%s type=%s reason=%s",
                request.request_id, agent_id, intervention_type.value, reason,
            )
            return record

    def respond_to_intervention(
        self,
        request_id: str,
        human_id: str,
        decision: HumanDecision,
        feedback: str = "",
        modified_action: str = "",
        counteroffer: dict[str, Any] | None = None,
        delegate_target: str = "",
    ) -> InterventionRecord:
        """响应干预请求（LangGraph Command resume 启发）.

        人类对干预请求做出决策，恢复 Agent 执行流。

        Args:
            request_id: 干预请求 ID
            human_id: 人类操作者 ID
            decision: 决策类型
            feedback: 反馈文本
            modified_action: 修改后动作（modify 时）
            counteroffer: 反提案（counteroffer 时）
            delegate_target: 委托目标（delegate 时）

        Returns:
            已解决的干预记录

        Raises:
            InterventionConflictError: 请求已解决或不存在
        """
        # G6 路由层兼容: 字符串转换为枚举
        if isinstance(decision, str):
            decision = HumanDecision(decision)
        with self._lock:
            record = self._interventions.get(request_id)
            if record is None:
                raise InterventionConflictError(request_id, reason="干预请求不存在")
            if record.status != InterventionStatus.PENDING:
                raise InterventionConflictError(
                    request_id,
                    reason=f"干预请求已处于 {record.status.value} 状态",
                )

            response = HumanResponse(
                request_id=request_id,
                human_id=human_id,
                decision=decision,
                feedback=feedback,
                modified_action=modified_action,
                counteroffer=counteroffer or {},
                delegate_target=delegate_target,
            )

            summary = f"{decision.value}: {feedback}" if feedback else decision.value
            record.resolve(response, summary)

            # 更新 Agent 配置
            profile = self._profiles.get(record.request.agent_id)
            if profile is not None:
                if decision == HumanDecision.APPROVE:
                    profile.reset_auto_steps()
                elif decision in (HumanDecision.REJECT, HumanDecision.MODIFY):
                    profile.record_override()
                    profile.reset_auto_steps()

            self._resolved_count += 1
            return record

    def expire_intervention(self, request_id: str) -> InterventionRecord | None:
        """将干预标记为超时过期."""
        with self._lock:
            record = self._interventions.get(request_id)
            if record is not None and record.status == InterventionStatus.PENDING:
                record.expire()
                self._expired_count += 1
                return record
            return None

    def cancel_intervention(self, request_id: str) -> InterventionRecord | None:
        """取消干预."""
        with self._lock:
            record = self._interventions.get(request_id)
            if record is not None and record.status == InterventionStatus.PENDING:
                record.cancel()
                return record
            return None

    # ==========================================================
    # Swarm 式升级 (escalate_to_human)
    # ==========================================================

    def escalate_to_human(
        self,
        agent_id: str,
        reason: str,
        target: str = "human_operator",
        payload: dict[str, Any] | None = None,
        priority: int = 80,
    ) -> InterventionRecord:
        """紧急升级到人类 (Swarm escalate_to_human 启发).

        创建高优先级升级干预。如果配置了自动升级且
        Agent 当前不是 SUPERVISED 模式，自动切换模式。
        """
        with self._lock:
            # 验证升级目标
            if target != "human_operator":
                profile = self._profiles.get(target)
                if profile is None:
                    raise EscalationTargetError(target, agent_id=agent_id)

            # 自动模式升级
            if self.config.enable_auto_escalation:
                profile = self._profiles.get(agent_id)
                if profile is not None and profile.mode != CollaborationMode.SUPERVISED:
                    current = profile.mode
                    # 升级到 SUPERVISED（混沌感知允许跳级）
                    profile.mode = CollaborationMode.SUPERVISED
                    self._switch_events.append(ModeSwitchEvent(
                        agent_id=agent_id,
                        from_mode=current,
                        to_mode=CollaborationMode.SUPERVISED,
                        trigger=SwitchTrigger.ANOMALY_DETECTED,
                        reason=reason,
                        confidence_at_time=None,
                    ))
                    self._switch_count += 1
                    self._escalation_count += 1

            record = self.create_intervention(
                agent_id=agent_id,
                intervention_type=InterventionType.ESCALATION,
                reason=reason,
                payload=payload or {},
                priority=priority,
            )
            record.request.context["escalation_target"] = target
            return record

    # ==========================================================
    # AutoGen 式自主步数检查
    # ==========================================================

    def check_auto_step(self, agent_id: str, confidence: float = 1.0) -> InterventionRecord | None:
        """检查自主步数和置信度 (AutoGen + CC1 联动启发).

        在每步 Agent 执行前调用：
        1. 如果达到 max_auto_steps 上限，创建干预
        2. 如果置信度低于阈值，创建升级干预
        3. 否则递增步数计数器

        Returns:
            如果需要人类干预则返回干预记录，否则返回 None
        """
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None or not profile.enabled:
                return None

            # SUPERVISED 模式：每步都需要审批
            if profile.mode == CollaborationMode.SUPERVISED:
                return self.create_intervention(
                    agent_id=agent_id,
                    intervention_type=InterventionType.CHECKPOINT,
                    reason="SUPERVISED 模式要求每步审批",
                    confidence=confidence,
                )

            # 置信度检查
            if confidence < profile.confidence_threshold:
                return self.create_intervention(
                    agent_id=agent_id,
                    intervention_type=InterventionType.ESCALATION,
                    reason=f"置信度 {confidence:.3f} 低于阈值 {profile.confidence_threshold}",
                    confidence=confidence,
                    priority=70,
                )

            # 自主步数检查
            reached = profile.increment_auto_step()
            if reached:
                return self.create_intervention(
                    agent_id=agent_id,
                    intervention_type=InterventionType.CHECKPOINT,
                    reason=f"连续自主步数已达上限 {profile.max_auto_steps}",
                    confidence=confidence,
                    priority=60,
                )

            return None

    # ==========================================================
    # GAIA 三阶段协商
    # ==========================================================

    def start_negotiation(
        self,
        agent_id: str,
        human_id: str,
        topic: str,
        initial_proposal: dict[str, Any],
        initial_confidence: float = 0.5,
        max_rounds: int | None = None,
    ) -> NegotiationSession:
        """启动 GAIA 协商会话.

        进入 Screening 阶段，Agent 提交初始提案。
        """
        if not self.config.enable_negotiation:
            raise CC2Error("CC2_NEGOTIATION_DISABLED", detail="协商模式未启用")

        session = NegotiationSession(
            agent_id=agent_id,
            human_id=human_id,
            topic=topic,
            max_rounds=max_rounds or self.config.max_negotiation_rounds,
        )
        session.add_round(
            proposer="agent",
            proposal=initial_proposal,
            confidence=initial_confidence,
            reasoning="初始提案",
        )

        with self._lock:
            self._negotiations[session.session_id] = session
            self._negotiation_count += 1

        return session

    def add_negotiation_round(
        self,
        session_id: str,
        proposer: str,
        proposal: dict[str, Any],
        confidence: float = 0.5,
        reasoning: str = "",
    ) -> NegotiationRound:
        """添加协商回合.

        Args:
            session_id: 协商会话 ID
            proposer: 提议方 ("agent" 或 "human")
            proposal: 提案内容
            confidence: 置信度
            reasoning: 推理依据

        Returns:
            新添加的协商回合

        Raises:
            NegotiationExhaustedError: 轮次已耗尽
        """
        with self._lock:
            session = self._negotiations.get(session_id)
            if session is None:
                raise CC2Error("CC2_NEGOTIATION_NOT_FOUND", detail=f"协商会话 '{session_id}' 不存在")
            if session.is_exhausted:
                raise NegotiationExhaustedError(
                    session_id, session.current_round, session.max_rounds,
                )
            if session.status != InterventionStatus.ACTIVE:
                raise CC2Error("CC2_NEGOTIATION_ENDED", detail=f"协商会话 '{session_id}' 已结束")

            return session.add_round(
                proposer=proposer,  # type: ignore[arg-type]
                proposal=proposal,
                confidence=confidence,
                reasoning=reasoning,
            )

    def finalize_negotiation(
        self,
        session_id: str,
        decision: HumanDecision,
    ) -> NegotiationSession:
        """结束协商并记录最终决策."""
        # G6 路由层兼容: 字符串转换为枚举
        if isinstance(decision, str):
            decision = HumanDecision(decision)
        with self._lock:
            session = self._negotiations.get(session_id)
            if session is None:
                raise CC2Error("CC2_NEGOTIATION_NOT_FOUND", detail=f"协商会话 '{session_id}' 不存在")
            session.finalize(decision)
            return session

    # ==========================================================
    # 持续质量评估 → 自动降级
    # ==========================================================

    def evaluate_sustained_quality(
        self,
        agent_id: str,
        accuracy: float,
        window_seconds: float | None = None,
    ) -> ModeSwitchEvent | None:
        """评估持续质量并可能触发自动降级.

        当 Agent 在时间窗口内持续保持高质量输出时，
        可自动降级到更高自主等级（需要更少人类干预）。

        Args:
            agent_id: Agent ID
            accuracy: 当前准确率
            window_seconds: 评估窗口秒数

        Returns:
            如果触发降级则返回模式切换事件
        """
        with self._lock:
            profile = self._profiles.get(agent_id)
            if profile is None or profile.mode == CollaborationMode.AUTONOMOUS:
                return None

            window = window_seconds or self.config.sustained_quality_window
            threshold = self.config.sustained_quality_threshold

            if accuracy >= threshold and profile.mode != CollaborationMode.AUTONOMOUS:
                # 降一级
                mode_order = _MODE_ORDER
                current_order = mode_order[profile.mode]
                if current_order > 0:
                    target = [m for m, o in mode_order.items() if o == current_order - 1][0]
                    return self.switch_mode(
                        agent_id=agent_id,
                        to_mode=target,
                        trigger=SwitchTrigger.SUSTAINED_QUALITY,
                        reason=f"持续质量达标 accuracy={accuracy:.3f} >= {threshold}",
                        confidence=accuracy,
                    )
            return None

    # ==========================================================
    # 查询
    # ==========================================================

    def get_intervention(self, request_id: str) -> InterventionRecord | None:
        """获取干预记录."""
        with self._lock:
            return self._interventions.get(request_id)

    def get_negotiation(self, session_id: str) -> NegotiationSession | None:
        """获取协商会话."""
        with self._lock:
            return self._negotiations.get(session_id)

    def query_interventions(
        self,
        *,
        agent_id: str | None = None,
        status: InterventionStatus | None = None,
        intervention_type: InterventionType | None = None,
        limit: int = 100,
    ) -> list[InterventionRecord]:
        """查询干预记录."""
        with self._lock:
            results = list(self._interventions.values())

        if agent_id is not None:
            results = [r for r in results if r.request.agent_id == agent_id]
        if status is not None:
            results = [r for r in results if r.status == status]
        if intervention_type is not None:
            results = [r for r in results if r.request.intervention_type == intervention_type]

        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    def get_switch_events(
        self,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[ModeSwitchEvent]:
        """获取模式切换事件."""
        with self._lock:
            events = list(self._switch_events)
        if agent_id is not None:
            events = [e for e in events if e.agent_id == agent_id]
        return events[-limit:]

    # ==========================================================
    # 统计
    # ==========================================================

    def get_stats(self) -> dict[str, Any]:
        """获取引擎统计信息."""
        with self._lock:
            by_decision: dict[str, int] = {}
            for record in self._interventions.values():
                if record.response:
                    key = record.response.decision.value
                    by_decision[key] = by_decision.get(key, 0) + 1

            by_mode: dict[str, int] = {}
            for profile in self._profiles.values():
                key = profile.mode.value
                by_mode[key] = by_mode.get(key, 0) + 1

            return {
                "registered_agents": len(self._profiles),
                "total_interventions": self._total_interventions,
                "resolved": self._resolved_count,
                "expired": self._expired_count,
                "escalations": self._escalation_count,
                "negotiations": self._negotiation_count,
                "mode_switches": self._switch_count,
                "pending_interventions": sum(
                    1 for r in self._interventions.values()
                    if r.status == InterventionStatus.PENDING
                ),
                "active_negotiations": sum(
                    1 for n in self._negotiations.values()
                    if n.status == InterventionStatus.ACTIVE
                ),
                "by_decision": by_decision,
                "agents_by_mode": by_mode,
            }

    def clear(self) -> None:
        """清空所有数据."""
        with self._lock:
            self._interventions.clear()
            self._negotiations.clear()
            self._switch_events.clear()
            self._total_interventions = 0
            self._resolved_count = 0
            self._expired_count = 0
            self._escalation_count = 0
            self._negotiation_count = 0
            self._switch_count = 0
