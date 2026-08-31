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
from .routing_engine import (
    RoutingEngine,
    RoutingContext,
    RoutingResult,
    CollaborationLayer,
    RiskLevel,
    Reversibility,
    UserRole,
    ApprovalMode,
    TimeoutAction,
)
from .approval_workflow import (
    ApprovalWorkflowManager,
    ApprovalRequest,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalStatus,
    TrustModeWindow,
)
from .anti_fatigue import (
    AntiFatigueManager,
    FatigueConfig,
    FatigueLevel,
    FatigueState,
    BatchApprovalGroup,
    ProgressiveTrustRecord,
)
from .intervention_manager import (
    InterventionManager,
    EmergencyPauseRequest,
    ManualOverrideRequest,
    CorrectionFeedback,
    CorrectionType,
    CorrectionSeverity,
    CreativeRequest,
    InterventionEvent,
)
from .kpi_metrics import (
    KPIMetricsEngine,
    KPISummary,
    KPIStatus,
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

        # CC2 增强组件 (可选集成)
        self._routing_engine: RoutingEngine | None = None
        self._approval_manager: ApprovalWorkflowManager | None = None
        self._anti_fatigue: AntiFatigueManager | None = None
        self._intervention_manager: InterventionManager | None = None
        self._kpi_engine: KPIMetricsEngine | None = None
        self._cc1_callback: Any = None  # CC1 审查回调

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

    # ==========================================================
    # CC2 增强组件挂载
    # ==========================================================

    def attach_routing_engine(self, engine: RoutingEngine) -> None:
        """挂载六维决策路由引擎实例.

        Args:
            engine: RoutingEngine 实例
        """
        with self._lock:
            self._routing_engine = engine
            logger.info("已挂载路由引擎 (RoutingEngine)")

    def attach_approval_manager(self, manager: ApprovalWorkflowManager) -> None:
        """挂载 L3 审批工作流管理器实例.

        Args:
            manager: ApprovalWorkflowManager 实例
        """
        with self._lock:
            self._approval_manager = manager
            logger.info("已挂载审批工作流管理器 (ApprovalWorkflowManager)")

    def attach_anti_fatigue(self, manager: AntiFatigueManager) -> None:
        """挂载审批抗疲劳管理器实例.

        Args:
            manager: AntiFatigueManager 实例
        """
        with self._lock:
            self._anti_fatigue = manager
            logger.info("已挂载抗疲劳管理器 (AntiFatigueManager)")

    def attach_intervention_manager(self, manager: InterventionManager) -> None:
        """挂载 L4 干预层管理器实例.

        Args:
            manager: InterventionManager 实例
        """
        with self._lock:
            self._intervention_manager = manager
            logger.info("已挂载干预管理器 (InterventionManager)")

    def attach_kpi_engine(self, engine: KPIMetricsEngine) -> None:
        """挂载 KPI 指标引擎实例.

        Args:
            engine: KPIMetricsEngine 实例
        """
        with self._lock:
            self._kpi_engine = engine
            logger.info("已挂载 KPI 指标引擎 (KPIMetricsEngine)")

    def attach_cc1_callback(self, callback: Any) -> None:
        """挂载 CC1 审查回调函数.

        当 CC1 评审结论产生或纠错触发重新审查时,
        通过此回调向 CC1 层传递联动信号.

        Args:
            callback: CC1 审查回调函数, 接收结果字典
        """
        with self._lock:
            self._cc1_callback = callback
            logger.info("已挂载 CC1 审查回调")

    # ==========================================================
    # 统一路由决策
    # ==========================================================

    def route_decision(self, ctx: RoutingContext) -> RoutingResult:
        """统一路由决策入口.

        使用路由引擎进行六维决策路由, 同时:
        1. 检查抗疲劳调整建议, 注入疲劳降级信号
        2. 记录 KPI 采样 (干预触发率 / CC1 联动率)
        3. 如有 CC1 联动信号, 传递到路由上下文供规则拾取

        Args:
            ctx: 六维路由上下文

        Returns:
            路由决策结果; 若路由引擎未挂载则返回默认 L1 隐性层结果
        """
        with self._lock:
            if self._routing_engine is None:
                logger.warning("路由引擎未挂载, 返回默认 L1 隐性层结果")
                return RoutingResult(
                    recommended_layer=CollaborationLayer.L1_IMPLICIT,
                    reasoning="路由引擎未挂载, 返回默认隐性层",
                )

            # 1. 抗疲劳调整建议 — 注入到上下文元数据
            if self._anti_fatigue is not None and ctx.user_id:
                try:
                    adjustment = self._anti_fatigue.get_fatigue_adjustment(
                        ctx.user_id,
                    )
                    recommendations = adjustment.get("recommendations", [])
                    if "downgrade_to_l2_prompt" in recommendations:
                        ctx.metadata["fatigue_downgrade"] = True
                    ctx.metadata["fatigue_level"] = adjustment.get(
                        "fatigue_level",
                    )
                    ctx.metadata["trust_score"] = adjustment.get(
                        "trust_score",
                    )
                except Exception:
                    logger.exception("抗疲劳调整建议查询失败")

            # 2. 执行六维路由决策
            result = self._routing_engine.route(ctx)

            # 3. KPI 采样
            if self._kpi_engine is not None:
                try:
                    is_intervention = (
                        result.recommended_layer
                        == CollaborationLayer.L4_INTERVENTION
                    )
                    cc1_triggered = (
                        ctx.metadata.get("cc1_verdict") == "block"
                    )
                    self._kpi_engine.ingest_from_routing_engine(
                        intervention_count=1 if is_intervention else 0,
                        cc1_triggered=cc1_triggered,
                        context={
                            "operation_type": ctx.operation_type,
                            "user_id": ctx.user_id,
                            "layer": result.recommended_layer.value,
                        },
                    )
                except Exception:
                    logger.exception("KPI 采样记录失败")

            logger.info(
                "路由决策: op=%s layer=%s score=%.1f rule=%s",
                ctx.operation_type,
                result.recommended_layer.value,
                result.score,
                result.rule_id or "-",
            )
            return result

    # ==========================================================
    # 审批工作流集成
    # ==========================================================

    def request_approval(
        self, operation: str, **kwargs: Any,
    ) -> ApprovalRequest | None:
        """创建审批请求 (集成审批工作流+抗疲劳+KPI).

        流程:
        1. 检查抗疲劳智能预批 — 历史高批准率操作可自动批准
        2. 创建审批请求 (审批工作流管理器)
        3. 如需等待人工决策, 检查批量审批聚合
        4. 记录 KPI 采样 (自动批准率)

        Args:
            operation: 操作类型
            **kwargs: 透传给 ApprovalWorkflowManager.create_request
                (risk_level / reversibility / approval_mode /
                 requester / user_id / timeout_seconds 等)

        Returns:
            审批请求; 若审批管理器未挂载则返回 None
        """
        with self._lock:
            if self._approval_manager is None:
                logger.warning("审批工作流管理器未挂载, 无法创建审批请求")
                return None

            user_id = kwargs.get("user_id", "")
            risk_level = kwargs.get("risk_level", RiskLevel.MEDIUM)
            risk_str = (
                risk_level.value
                if hasattr(risk_level, "value")
                else str(risk_level)
            )

            # 1. 抗疲劳智能预批检查
            smart_preapproved = False
            if self._anti_fatigue is not None and user_id:
                try:
                    smart_preapproved = (
                        self._anti_fatigue.should_smart_preapprove(
                            user_id, operation, risk_str,
                        )
                    )
                except Exception:
                    logger.exception("抗疲劳智能预批检查失败")

            # 2. 创建审批请求
            request = self._approval_manager.create_request(
                operation=operation, **kwargs,
            )

            # 智能预批通过 → 立即自动批准
            if smart_preapproved:
                record = self._approval_manager.get_record(
                    request.request_id,
                )
                if (
                    record is not None
                    and record.status == ApprovalStatus.PENDING
                ):
                    try:
                        self._approval_manager.make_decision(
                            request.request_id,
                            decision=ApprovalStatus.AUTO_APPROVED,
                            decided_by="anti_fatigue_smart_preapprove",
                            comment="抗疲劳智能预批自动批准",
                        )
                        # 追踪抗疲劳决策
                        if self._anti_fatigue is not None and user_id:
                            self._anti_fatigue.track_decision(
                                user_id, operation, "auto_approved",
                                0.0, risk_str,
                            )
                    except Exception:
                        logger.exception("智能预批自动批准失败")

            # 3. 如需等待, 检查批量审批聚合
            record = self._approval_manager.get_record(request.request_id)
            if (
                self._anti_fatigue is not None
                and user_id
                and record is not None
                and record.status == ApprovalStatus.PENDING
            ):
                try:
                    self._anti_fatigue.add_to_batch(
                        user_id,
                        operation,
                        {
                            "request_id": request.request_id,
                            "risk_level": risk_str,
                            "operation": operation,
                        },
                    )
                except Exception:
                    logger.exception("批量审批聚合失败")

            # 4. KPI 采样
            if self._kpi_engine is not None:
                try:
                    final_record = self._approval_manager.get_record(
                        request.request_id,
                    )
                    auto_approved = (
                        final_record is not None
                        and final_record.status
                        == ApprovalStatus.AUTO_APPROVED
                    )
                    self._kpi_engine.ingest_from_approval_workflow(
                        auto_approved=auto_approved,
                        context={
                            "operation": operation,
                            "user_id": user_id,
                        },
                    )
                except Exception:
                    logger.exception("KPI 采样记录失败")

            logger.info(
                "创建审批请求: op=%s request=%s risk=%s smart_preapprove=%s",
                operation, request.request_id, risk_str, smart_preapproved,
            )
            return request

    # ==========================================================
    # CC1 审查集成
    # ==========================================================

    def process_cc1_result(
        self,
        agent_id: str,
        cc1_verdict: str,
        cc1_score: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """处理 CC1 审查结果.

        根据 CC1 评审结论路由到对应协同层级:
        - pass: 正常执行, 可能触发 L1/L2
        - warn: 降级到 L2 提示层
        - block: 升级到 L3 审批层 (人工仲裁)

        同时更新路由上下文的置信度, 并通过 CC1 回调
        向 CC1 层传递联动信号.

        Args:
            agent_id: Agent ID
            cc1_verdict: CC1 评审结论 (pass/warn/block)
            cc1_score: CC1 综合评分 (0-1 或 0-100, 自动归一化)
            **kwargs: 附加路由上下文字段 (operation_type / user_id 等)

        Returns:
            处理结果字典, 包含动作、路由层级等信息
        """
        with self._lock:
            # 置信度归一化到 [0, 1]
            normalized_score = cc1_score
            if normalized_score > 1.0:
                normalized_score = normalized_score / 100.0
            normalized_score = max(0.0, min(1.0, normalized_score))

            result: dict[str, Any] = {
                "agent_id": agent_id,
                "cc1_verdict": cc1_verdict,
                "cc1_score": cc1_score,
                "normalized_confidence": normalized_score,
                "action": "proceed",
                "routing_layer": None,
            }

            # 构建路由上下文, 注入 CC1 信号与置信度
            metadata = dict(kwargs)
            metadata["cc1_verdict"] = cc1_verdict
            metadata["cc1_score"] = cc1_score
            ctx = RoutingContext(
                confidence=normalized_score,
                operation_type=kwargs.get("operation_type", ""),
                user_id=kwargs.get("user_id", ""),
                metadata=metadata,
            )

            # 路由决策 (CC1 block 规则 RR-005 会拾取 metadata)
            if self._routing_engine is not None:
                try:
                    routing_result = self._routing_engine.route(ctx)
                    result["routing_layer"] = (
                        routing_result.recommended_layer.value
                    )
                    result["routing_result_id"] = routing_result.result_id
                    result["routing_reasoning"] = routing_result.reasoning
                except Exception:
                    logger.exception("CC1 结果路由决策失败")

            # 根据 verdict 决定动作
            if cc1_verdict == "pass":
                result["action"] = "proceed"
            elif cc1_verdict == "warn":
                result["action"] = "downgrade_to_l2_prompt"
            elif cc1_verdict == "block":
                result["action"] = "escalate_to_l3_approval"
                # 设置元数据供路由引擎拾取 (人工仲裁)
                result["metadata"] = {
                    "cc1_verdict": "block",
                    "cc1_score": cc1_score,
                }
            else:
                result["action"] = "unknown_verdict"

            # KPI 记录 CC1 联动率
            if self._kpi_engine is not None:
                try:
                    self._kpi_engine.ingest_from_routing_engine(
                        cc1_triggered=(cc1_verdict == "block"),
                        context={
                            "agent_id": agent_id,
                            "cc1_verdict": cc1_verdict,
                        },
                    )
                except Exception:
                    logger.exception("KPI 采样记录失败")

            # CC1 回调通知
            if self._cc1_callback is not None:
                try:
                    self._cc1_callback(result)
                except Exception:
                    logger.exception("CC1 审查回调执行失败")

            logger.info(
                "CC1 结果处理: agent=%s verdict=%s score=%.3f action=%s layer=%s",
                agent_id, cc1_verdict, cc1_score,
                result["action"], result["routing_layer"],
            )
            return result

    # ==========================================================
    # L4 干预层委托
    # ==========================================================

    def emergency_pause(
        self, user_id: str, reason: str, **kwargs: Any,
    ) -> EmergencyPauseRequest | None:
        """紧急暂停 (委托给 InterventionManager).

        立即阻塞全部相关 Agent 执行, 自动通知教师,
        并记录 KPI 干预触发率.

        Args:
            user_id: 触发用户 ID (通常为学生)
            reason: 暂停原因
            **kwargs: 透传给 InterventionManager.initiate_emergency_pause
                (scope / agent_ids / auto_notify_teacher 等)

        Returns:
            紧急暂停请求; 若干预管理器未挂载则返回 None
        """
        with self._lock:
            if self._intervention_manager is None:
                logger.warning("干预管理器未挂载, 无法发起紧急暂停")
                return None

            pause = self._intervention_manager.initiate_emergency_pause(
                user_id=user_id, reason=reason, **kwargs,
            )

            # KPI 记录干预触发率
            if self._kpi_engine is not None:
                try:
                    self._kpi_engine.ingest_from_routing_engine(
                        intervention_count=1,
                        context={
                            "type": "emergency_pause",
                            "user_id": user_id,
                            "pause_id": pause.pause_id,
                        },
                    )
                except Exception:
                    logger.exception("KPI 采样记录失败")

            logger.warning(
                "紧急暂停委托: pause=%s user=%s reason=%s",
                pause.pause_id, user_id, reason,
            )
            return pause

    def manual_override(
        self, operator_id: str, target_agent: str, **kwargs: Any,
    ) -> ManualOverrideRequest | None:
        """人工接管 (委托给 InterventionManager).

        人类操作者从 AI Agent 接管控制权,
        并记录 KPI 干预触发率.

        Args:
            operator_id: 操作者 ID (教师 / 管理员)
            target_agent: 被接管 Agent ID
            **kwargs: 透传给 InterventionManager.initiate_manual_override
                (override_level / instructions / duration_seconds 等)

        Returns:
            人工接管请求; 若干预管理器未挂载则返回 None
        """
        with self._lock:
            if self._intervention_manager is None:
                logger.warning("干预管理器未挂载, 无法发起人工接管")
                return None

            override = self._intervention_manager.initiate_manual_override(
                operator_id=operator_id,
                target_agent=target_agent,
                **kwargs,
            )

            # KPI 记录干预触发率
            if self._kpi_engine is not None:
                try:
                    self._kpi_engine.ingest_from_routing_engine(
                        intervention_count=1,
                        context={
                            "type": "manual_override",
                            "operator_id": operator_id,
                            "target_agent": target_agent,
                        },
                    )
                except Exception:
                    logger.exception("KPI 采样记录失败")

            logger.info(
                "人工接管委托: override=%s operator=%s target=%s",
                override.override_id, operator_id, target_agent,
            )
            return override

    def submit_correction(
        self,
        corrector_id: str,
        target_content_id: str,
        original: str,
        corrected: str,
        **kwargs: Any,
    ) -> CorrectionFeedback | None:
        """纠错反馈 (委托给 InterventionManager).

        人类纠正 AI 输出并提供反馈. 如果纠正严重度为
        major/critical, 自动应用纠正并触发 CC1 重新审查,
        形成在线学习闭环.

        Args:
            corrector_id: 纠正者 ID
            target_content_id: 被纠正内容 ID
            original: 原始内容
            corrected: 纠正后内容
            **kwargs: 透传给 InterventionManager.submit_correction
                (correction_type / severity / feedback /
                 target_agent_id / corrector_role 等)

        Returns:
            纠正反馈对象; 若干预管理器未挂载则返回 None
        """
        with self._lock:
            if self._intervention_manager is None:
                logger.warning("干预管理器未挂载, 无法提交纠错反馈")
                return None

            # correction_type 为必填, 默认事实性纠正; 兼容字符串与枚举
            correction_type = kwargs.pop("correction_type", CorrectionType.FACTUAL)
            if not isinstance(correction_type, CorrectionType):
                correction_type = CorrectionType(correction_type)

            # severity 兼容字符串与枚举 (默认值由 InterventionManager 处理)
            if "severity" in kwargs:
                sev = kwargs["severity"]
                if not isinstance(sev, CorrectionSeverity):
                    kwargs["severity"] = CorrectionSeverity(sev)

            correction = self._intervention_manager.submit_correction(
                corrector_id=corrector_id,
                target_content_id=target_content_id,
                original=original,
                corrected=corrected,
                correction_type=correction_type,
                **kwargs,
            )

            # 检查严重度, major/critical 自动应用并触发 CC1 重新审查
            severity = correction.severity
            severity_val = (
                severity.value if hasattr(severity, "value") else str(severity)
            )
            if severity_val in ("major", "critical"):
                try:
                    self._intervention_manager.apply_correction(
                        correction.correction_id,
                        applied_by=corrector_id,
                    )
                    # 通过 CC1 回调触发重新审查
                    if self._cc1_callback is not None:
                        self._cc1_callback({
                            "type": "cc1_re_review",
                            "correction_id": correction.correction_id,
                            "severity": severity_val,
                            "target_content_id": target_content_id,
                            "target_agent_id": correction.target_agent_id,
                        })
                except Exception:
                    logger.exception("纠错自动应用 / CC1 重新审查触发失败")

            # KPI 记录纠错反馈率
            if self._kpi_engine is not None:
                try:
                    self._kpi_engine.ingest_from_routing_engine(
                        correction_count=1,
                        context={
                            "correction_id": correction.correction_id,
                            "severity": severity_val,
                        },
                    )
                except Exception:
                    logger.exception("KPI 采样记录失败")

            logger.info(
                "纠错反馈委托: correction=%s corrector=%s severity=%s cc1=%s",
                correction.correction_id, corrector_id, severity_val,
                correction.cc1_re_review_triggered,
            )
            return correction

    # ==========================================================
    # KPI 仪表盘与增强统计
    # ==========================================================

    def get_kpi_dashboard(self) -> dict[str, Any]:
        """获取 KPI 仪表盘数据.

        汇集 9 项 KPI 的状态、整体健康分、分类得分、
        告警与统计信息, 支持仪表盘可视化集成.

        Returns:
            仪表盘数据字典; 若 KPI 引擎未挂载则返回空字典
        """
        with self._lock:
            if self._kpi_engine is None:
                return {}
            try:
                return self._kpi_engine.get_dashboard_data()
            except Exception:
                logger.exception("KPI 仪表盘数据获取失败")
                return {}

    def get_enhanced_stats(self) -> dict[str, Any]:
        """获取增强统计信息 (集成所有组件).

        在基础引擎统计之上, 聚合路由引擎、审批工作流、
        抗疲劳、干预管理器和 KPI 引擎的统计信息,
        并标注各组件挂载状态.

        Returns:
            增强统计信息字典
        """
        with self._lock:
            stats = self.get_stats()

            if self._routing_engine is not None:
                try:
                    stats["routing"] = (
                        self._routing_engine.get_statistics()
                    )
                except Exception:
                    logger.exception("路由引擎统计获取失败")

            if self._approval_manager is not None:
                try:
                    stats["approval"] = (
                        self._approval_manager.get_statistics()
                    )
                except Exception:
                    logger.exception("审批工作流统计获取失败")

            if self._anti_fatigue is not None:
                try:
                    stats["anti_fatigue"] = (
                        self._anti_fatigue.get_statistics()
                    )
                except Exception:
                    logger.exception("抗疲劳统计获取失败")

            if self._intervention_manager is not None:
                try:
                    stats["intervention"] = (
                        self._intervention_manager.get_statistics()
                    )
                except Exception:
                    logger.exception("干预管理器统计获取失败")

            if self._kpi_engine is not None:
                try:
                    stats["kpi"] = self._kpi_engine.get_statistics()
                except Exception:
                    logger.exception("KPI 引擎统计获取失败")

            stats["components_attached"] = {
                "routing_engine": self._routing_engine is not None,
                "approval_manager": self._approval_manager is not None,
                "anti_fatigue": self._anti_fatigue is not None,
                "intervention_manager": self._intervention_manager is not None,
                "kpi_engine": self._kpi_engine is not None,
                "cc1_callback": self._cc1_callback is not None,
            }
            return stats

    def clear(self) -> None:
        """清空所有数据.

        清空引擎内置数据, 并清空已挂载的增强组件数据
        (路由历史 / 审批记录 / 抗疲劳状态 / 干预事件 / KPI 采样).
        增强组件实例与挂载关系不受影响.
        """
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

            # 清空已挂载的增强组件数据
            if self._routing_engine is not None:
                try:
                    self._routing_engine.clear_history()
                except Exception:
                    logger.exception("路由引擎历史清空失败")
            if self._approval_manager is not None:
                try:
                    self._approval_manager.clear()
                except Exception:
                    logger.exception("审批工作流数据清空失败")
            if self._anti_fatigue is not None:
                try:
                    self._anti_fatigue.clear()
                except Exception:
                    logger.exception("抗疲劳数据清空失败")
            if self._intervention_manager is not None:
                try:
                    self._intervention_manager.clear()
                except Exception:
                    logger.exception("干预管理器数据清空失败")
            if self._kpi_engine is not None:
                try:
                    self._kpi_engine.clear()
                except Exception:
                    logger.exception("KPI 引擎数据清空失败")
