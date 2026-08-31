"""CC2 计划审批门 — 增强组件完整测试.

覆盖 CC2 Plan-Approval Gate 系统的全部新增模块:
1. 决策路由引擎 (RoutingEngine) — 六维输入→四层协同
2. 审批工作流 (ApprovalWorkflowManager) — 四种审批模式+超时策略+信任窗口
3. 抗疲劳机制 (AntiFatigueManager) — 频率控制+批量审批+智能预批+渐进信任
4. 干预管理器 (InterventionManager) — 紧急暂停+手动接管+纠错反馈+创意请求
5. KPI 指标引擎 (KPIMetricsEngine) — 9项KPI追踪+动态阈值调整
6. 引擎集成 (CollaborationEngine) — 增强组件挂载与联动
7. REST API (CC2APIRouter) — 端点全覆盖
"""
from __future__ import annotations

import enum
import time
from typing import Any

import pytest

from dy3_polaris.l0.cc2 import (
    # 路由引擎
    RoutingEngine,
    RoutingContext,
    RoutingResult,
    RoutingRule,
    CollaborationLayer,
    RiskLevel,
    Reversibility,
    UserRole,
    ApprovalMode,
    TimeoutAction,
    InterventionTypeL4,
    Priority,
    RecoveryMode,
    DEFAULT_ROUTING_RULES,
    # 审批工作流
    ApprovalWorkflowManager,
    ApprovalRequest,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalStatus,
    TrustModeWindow,
    # 抗疲劳
    AntiFatigueManager,
    FatigueConfig,
    FatigueLevel,
    FatigueState,
    ApprovalPattern,
    BatchApprovalGroup,
    BatchStatus,
    ProgressiveTrustRecord,
    # 干预管理器
    InterventionManager,
    EmergencyPauseRequest,
    ManualOverrideRequest,
    CorrectionFeedback,
    CreativeRequest,
    InterventionEvent,
    InterventionAction,
    PauseScope,
    OverrideLevel,
    CorrectionType,
    CorrectionSeverity,
    CreativeRequestType,
    InterventionEventStatus,
    # KPI 指标
    KPIMetricsEngine,
    KPISample,
    KPIStatus,
    KPICategory,
    TrendDirection,
    KPISummary,
    KPIThreshold,
    KPITrend,
    # REST API
    CC2APIRouter,
    # 引擎
    CollaborationEngine,
    CollaborationConfig,
    AgentCollaborationProfile,
    CollaborationMode,
)
from dy3_polaris.l0.cc2.kpi_metrics import KPI_NAMES


# ============================================================
# 辅助函数
# ============================================================


def _make_routing_context(**kwargs: Any) -> RoutingContext:
    """创建测试用路由上下文."""
    return RoutingContext(**kwargs)


def _make_routing_engine() -> RoutingEngine:
    """创建默认路由引擎."""
    return RoutingEngine()


def _make_approval_manager() -> ApprovalWorkflowManager:
    """创建默认审批工作流管理器."""
    return ApprovalWorkflowManager()


def _make_anti_fatigue() -> AntiFatigueManager:
    """创建默认抗疲劳管理器."""
    return AntiFatigueManager()


def _make_intervention_manager() -> InterventionManager:
    """创建默认干预管理器."""
    return InterventionManager()


def _make_kpi_engine() -> KPIMetricsEngine:
    """创建默认 KPI 指标引擎."""
    return KPIMetricsEngine()


# ============================================================
# 1. 测试决策路由引擎
# ============================================================


class TestRoutingEngine:
    """六维决策路由引擎完整测试."""

    # --- 枚举值 ---

    def test_协同层级_四个值(self) -> None:
        assert len(CollaborationLayer) == 4

    def test_协同层级_字符串值(self) -> None:
        assert CollaborationLayer.L1_IMPLICIT.value == "l1_implicit"
        assert CollaborationLayer.L2_PROMPT.value == "l2_prompt"
        assert CollaborationLayer.L3_APPROVAL.value == "l3_approval"
        assert CollaborationLayer.L4_INTERVENTION.value == "l4_intervention"

    def test_协同层级_继承str和Enum(self) -> None:
        assert issubclass(CollaborationLayer, str)
        assert issubclass(CollaborationLayer, enum.Enum)

    def test_风险等级_四个值(self) -> None:
        assert len(RiskLevel) == 4
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_可逆性_三个值(self) -> None:
        assert len(Reversibility) == 3
        assert Reversibility.REVERSIBLE.value == "reversible"
        assert Reversibility.PARTIALLY_REVERSIBLE.value == "partially_reversible"
        assert Reversibility.IRREVERSIBLE.value == "irreversible"

    def test_用户角色_四个值(self) -> None:
        assert len(UserRole) == 4
        assert UserRole.STUDENT.value == "student"
        assert UserRole.TEACHER.value == "teacher"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.SYSTEM.value == "system"

    def test_审批模式_四个值(self) -> None:
        assert len(ApprovalMode) == 4
        assert ApprovalMode.QUICK_CONFIRM.value == "quick_confirm"
        assert ApprovalMode.DETAILED_REVIEW.value == "detailed_review"
        assert ApprovalMode.NEGOTIATED_APPROVAL.value == "negotiated_approval"
        assert ApprovalMode.RULE_PRESET.value == "rule_preset"

    def test_超时策略_四个值(self) -> None:
        assert len(TimeoutAction) == 4
        assert TimeoutAction.ABORT.value == "abort"
        assert TimeoutAction.AUTO_APPROVE.value == "auto_approve"
        assert TimeoutAction.DOWNGRADE_TO_PROMPT.value == "downgrade_to_prompt"
        assert TimeoutAction.ESCALATE.value == "escalate"

    # --- 数据模型 ---

    def test_路由上下文_默认值(self) -> None:
        ctx = RoutingContext()
        assert ctx.operation_type == ""
        assert ctx.risk_level == RiskLevel.LOW
        assert ctx.confidence == 0.95
        assert ctx.trust_score == 0.90
        assert ctx.reversibility == Reversibility.REVERSIBLE
        assert ctx.user_role == UserRole.STUDENT
        assert ctx.cognitive_load == 0.45
        assert ctx.metadata == {}

    def test_路由上下文_自定义值(self) -> None:
        ctx = RoutingContext(
            operation_type="test_op",
            risk_level=RiskLevel.HIGH,
            confidence=0.5,
            trust_score=0.3,
            reversibility=Reversibility.IRREVERSIBLE,
            user_role=UserRole.ADMIN,
            cognitive_load=0.8,
            user_id="u1",
            session_id="s1",
            metadata={"key": "val"},
        )
        assert ctx.operation_type == "test_op"
        assert ctx.risk_level == RiskLevel.HIGH
        assert ctx.confidence == 0.5
        assert ctx.trust_score == 0.3
        assert ctx.reversibility == Reversibility.IRREVERSIBLE
        assert ctx.user_role == UserRole.ADMIN
        assert ctx.cognitive_load == 0.8
        assert ctx.user_id == "u1"
        assert ctx.metadata["key"] == "val"

    def test_路由结果_默认值(self) -> None:
        result = RoutingResult()
        assert result.recommended_layer == CollaborationLayer.L1_IMPLICIT
        assert result.approval_mode is None
        assert result.timeout_seconds == 300.0
        assert result.timeout_action == TimeoutAction.ABORT
        assert result.score == 0.0
        assert result.rule_id == ""
        assert result.alternatives == []

    def test_路由结果_自定义值(self) -> None:
        result = RoutingResult(
            recommended_layer=CollaborationLayer.L3_APPROVAL,
            approval_mode=ApprovalMode.DETAILED_REVIEW,
            score=55.0,
            rule_id="RR-005",
        )
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.DETAILED_REVIEW
        assert result.score == 55.0
        assert result.rule_id == "RR-005"

    def test_路由结果_result_id自动生成(self) -> None:
        r1 = RoutingResult()
        r2 = RoutingResult()
        assert r1.result_id != r2.result_id
        assert r1.result_id.startswith("rt-")

    # --- 默认路由规则 ---

    def test_默认规则_共12条(self) -> None:
        assert len(DEFAULT_ROUTING_RULES) == 12

    def test_默认规则_包含所有rule_id(self) -> None:
        rule_ids = {r.rule_id for r in DEFAULT_ROUTING_RULES}
        expected = {f"RR-{i:03d}" for i in range(1, 13)}
        assert rule_ids == expected

    def test_默认规则_按优先级排序(self) -> None:
        engine = _make_routing_engine()
        priorities = [r.priority for r in engine.rules]
        assert priorities == sorted(priorities, reverse=True)

    # --- 规则匹配测试 ---

    def test_规则匹配_紧急暂停_RR001(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(operation_type="emergency_pause")
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L4_INTERVENTION
        assert result.rule_id == "RR-001"

    def test_规则匹配_安全内容_RR002(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(metadata={"safety_related": True})
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L4_INTERVENTION
        assert result.rule_id == "RR-002"

    def test_规则匹配_连续错误_RR003(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(metadata={"consecutive_errors": 10})
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L4_INTERVENTION
        assert result.rule_id == "RR-003"

    def test_规则匹配_认知过载_RR004(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(cognitive_load=0.95)
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L4_INTERVENTION
        assert result.rule_id == "RR-004"

    def test_规则匹配_CC1Block_RR005(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(metadata={"cc1_verdict": "block"})
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.DETAILED_REVIEW
        assert result.rule_id == "RR-005"

    def test_规则匹配_Prompt模板修改_RR006(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(operation_type="prompt_template_modify")
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.NEGOTIATED_APPROVAL
        assert result.rule_id == "RR-006"

    def test_规则匹配_发布内容_RR007(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            operation_type="publish_content",
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.NEGOTIATED_APPROVAL
        assert result.rule_id == "RR-007"

    def test_规则匹配_学习路径重置_RR008(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            operation_type="learning_path_reset",
            reversibility=Reversibility.IRREVERSIBLE,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.DETAILED_REVIEW
        assert result.rule_id == "RR-008"

    def test_规则匹配_数据覆写_RR009(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            operation_type="data_overwrite",
            reversibility=Reversibility.IRREVERSIBLE,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.rule_id == "RR-009"

    def test_规则匹配_外部API_RR010(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            operation_type="external_api_call",
            risk_level=RiskLevel.HIGH,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.rule_id == "RR-010"

    def test_规则匹配_低置信度内容_RR011(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            confidence=0.65,
            risk_level=RiskLevel.MEDIUM,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L2_PROMPT
        assert result.rule_id == "RR-011"

    def test_规则匹配_信任模式_RR012(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            metadata={"trust_mode_active": True},
            risk_level=RiskLevel.LOW,
            reversibility=Reversibility.REVERSIBLE,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L1_IMPLICIT
        assert result.rule_id == "RR-012"
        assert result.timeout_action == TimeoutAction.AUTO_APPROVE

    # --- 评分路由 ---

    def test_评分路由_低分L1(self) -> None:
        """默认上下文 → 低分 → L1."""
        engine = _make_routing_engine()
        ctx = RoutingContext()  # 默认值: 低风险高置信高信任
        result = engine.route(ctx)
        assert result.score < 25.0
        assert result.recommended_layer == CollaborationLayer.L1_IMPLICIT
        assert result.approval_mode is None

    def test_评分路由_中低分L2(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.MEDIUM,
            confidence=0.5,
            trust_score=0.5,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            cognitive_load=0.5,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert 25.0 <= result.score < 50.0
        assert result.recommended_layer == CollaborationLayer.L2_PROMPT

    def test_评分路由_中高分L3(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.HIGH,
            confidence=0.5,
            trust_score=0.5,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            cognitive_load=0.5,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert 50.0 <= result.score < 75.0
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL

    def test_评分路由_高分L4(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.HIGH,
            confidence=0.3,
            trust_score=0.2,
            reversibility=Reversibility.IRREVERSIBLE,
            cognitive_load=0.9,
            user_role=UserRole.ADMIN,
        )
        result = engine.route(ctx)
        assert result.score >= 75.0
        assert result.recommended_layer == CollaborationLayer.L4_INTERVENTION

    # --- 审批模式选择 ---

    def test_审批模式_L3低风险_快速确认(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.LOW,
            confidence=0.2,
            trust_score=0.2,
            reversibility=Reversibility.IRREVERSIBLE,
            cognitive_load=0.5,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.QUICK_CONFIRM

    def test_审批模式_L3极高_协商审批(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.CRITICAL,
            confidence=0.5,
            trust_score=0.5,
            reversibility=Reversibility.IRREVERSIBLE,
            cognitive_load=0.5,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.NEGOTIATED_APPROVAL

    def test_审批模式_L3部分可逆_协商审批(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.HIGH,
            confidence=0.5,
            trust_score=0.5,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            cognitive_load=0.5,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.NEGOTIATED_APPROVAL

    def test_审批模式_L3默认_详细审批(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.HIGH,
            confidence=0.5,
            trust_score=0.5,
            reversibility=Reversibility.IRREVERSIBLE,
            cognitive_load=0.5,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.approval_mode == ApprovalMode.DETAILED_REVIEW

    # --- 超时选择 ---

    def test_超时_L1为零(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext()
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L1_IMPLICIT
        assert result.timeout_seconds == 0.0

    def test_超时_L3低风险30秒(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.LOW,
            confidence=0.2,
            trust_score=0.2,
            reversibility=Reversibility.IRREVERSIBLE,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.timeout_seconds == 30.0

    def test_超时_L3中风险120秒(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.MEDIUM,
            confidence=0.2,
            reversibility=Reversibility.IRREVERSIBLE,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.timeout_seconds == 120.0

    def test_超时_L3高风险300秒(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.HIGH,
            confidence=0.5,
            reversibility=Reversibility.IRREVERSIBLE,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.timeout_seconds == 300.0

    def test_超时_L3极高风险600秒(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.CRITICAL,
            confidence=0.5,
            reversibility=Reversibility.IRREVERSIBLE,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.timeout_seconds == 600.0

    # --- 超时策略 ---

    def test_超时策略_L1自动批准(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext()
        result = engine.route(ctx)
        assert result.timeout_action == TimeoutAction.AUTO_APPROVE

    def test_超时策略_L2降级提示(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.MEDIUM,
            confidence=0.5,
            trust_score=0.5,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            cognitive_load=0.5,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L2_PROMPT
        assert result.timeout_action == TimeoutAction.DOWNGRADE_TO_PROMPT

    def test_超时策略_L3低风险自动批准(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.LOW,
            confidence=0.2,
            trust_score=0.2,
            reversibility=Reversibility.IRREVERSIBLE,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
        assert result.timeout_action == TimeoutAction.AUTO_APPROVE

    def test_超时策略_L3非低风险中止(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(
            risk_level=RiskLevel.HIGH,
            confidence=0.5,
            reversibility=Reversibility.IRREVERSIBLE,
            user_role=UserRole.TEACHER,
        )
        result = engine.route(ctx)
        assert result.timeout_action == TimeoutAction.ABORT

    def test_超时策略_L4中止(self) -> None:
        engine = _make_routing_engine()
        ctx = RoutingContext(operation_type="emergency_pause")
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L4_INTERVENTION
        assert result.timeout_action == TimeoutAction.ABORT

    # --- 自定义规则与统计 ---

    def test_添加自定义规则(self) -> None:
        engine = _make_routing_engine()
        custom_rule = RoutingRule(
            rule_id="CUSTOM-001",
            name="自定义规则",
            description="测试自定义规则",
            matcher=lambda ctx: ctx.operation_type == "custom_op",
            layer=CollaborationLayer.L2_PROMPT,
            priority=200,
        )
        engine.add_rule(custom_rule)
        ctx = RoutingContext(operation_type="custom_op")
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L2_PROMPT
        assert result.rule_id == "CUSTOM-001"

    def test_路由历史记录(self) -> None:
        engine = _make_routing_engine()
        engine.route(RoutingContext())
        engine.route(RoutingContext(operation_type="emergency_pause"))
        assert len(engine.routing_history) == 2

    def test_路由统计(self) -> None:
        engine = _make_routing_engine()
        engine.route(RoutingContext())
        engine.route(RoutingContext(operation_type="emergency_pause"))
        stats = engine.get_statistics()
        assert stats["total"] == 2
        assert "by_layer" in stats
        assert "by_rule" in stats
        assert "avg_score" in stats

    def test_路由统计_空历史(self) -> None:
        engine = _make_routing_engine()
        stats = engine.get_statistics()
        assert stats == {"total": 0}

    def test_清空历史(self) -> None:
        engine = _make_routing_engine()
        engine.route(RoutingContext())
        engine.clear_history()
        assert len(engine.routing_history) == 0


# ============================================================
# 2. 测试审批工作流
# ============================================================


class TestApprovalWorkflow:
    """L3 审批工作流完整测试."""

    # --- 枚举 ---

    def test_审批状态_枚举值(self) -> None:
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.MODIFIED.value == "modified"
        assert ApprovalStatus.TIMEOUT.value == "timeout"
        assert ApprovalStatus.AUTO_APPROVED.value == "auto_approved"
        assert ApprovalStatus.CANCELLED.value == "cancelled"
        assert ApprovalStatus.ABORTED.value == "aborted"

    # --- 数据模型 ---

    def test_审批请求_创建与过期时间(self) -> None:
        req = ApprovalRequest(
            operation="test_op",
            timeout_seconds=300.0,
        )
        assert req.operation == "test_op"
        assert req.expires_at > 0
        assert req.expires_at == req.created_at + 300.0

    def test_审批请求_零超时无过期时间(self) -> None:
        req = ApprovalRequest(operation="test_op", timeout_seconds=0.0)
        assert req.expires_at == 0.0

    def test_审批决策_创建(self) -> None:
        dec = ApprovalDecision(
            request_id="req-001",
            decision=ApprovalStatus.APPROVED,
            decided_by="teacher-001",
            comment="同意",
        )
        assert dec.request_id == "req-001"
        assert dec.decision == ApprovalStatus.APPROVED
        assert dec.decided_by == "teacher-001"

    def test_审批记录_创建(self) -> None:
        req = ApprovalRequest(operation="test_op")
        record = ApprovalRecord(request=req)
        assert record.request == req
        assert record.status == ApprovalStatus.PENDING
        assert record.decision is None

    # --- 信任模式窗口 ---

    def test_信任窗口_激活与有效(self) -> None:
        window = TrustModeWindow("user-001", duration_seconds=1800.0)
        assert window.active is False
        assert window.is_valid() is False
        window.activate()
        assert window.active is True
        assert window.is_valid() is True

    def test_信任窗口_停用(self) -> None:
        window = TrustModeWindow("user-001")
        window.activate()
        window.deactivate()
        assert window.active is False
        assert window.is_valid() is False

    def test_信任窗口_剩余时间(self) -> None:
        window = TrustModeWindow("user-001", duration_seconds=1800.0)
        window.activate()
        remaining = window.remaining_seconds()
        assert 0 < remaining <= 1800.0

    def test_信任窗口_过期后无效(self) -> None:
        window = TrustModeWindow("user-001", duration_seconds=0.01)
        window.activate()
        time.sleep(0.02)
        assert window.is_valid() is False
        assert window.remaining_seconds() == 0.0

    # --- 审批工作流管理器 ---

    def test_创建审批请求_正常(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(
            operation="learning_path_reset",
            risk_level=RiskLevel.HIGH,
            approval_mode=ApprovalMode.DETAILED_REVIEW,
            requester="agent-001",
            timeout_seconds=300,
        )
        record = mgr.get_record(req.request_id)
        assert record is not None
        assert record.status == ApprovalStatus.PENDING

    def test_创建审批请求_信任模式自动批准(self) -> None:
        mgr = _make_approval_manager()
        mgr.activate_trust_mode("user-001", duration_seconds=1800.0)
        req = mgr.create_request(
            operation="quiz_submit",
            risk_level=RiskLevel.LOW,
            reversibility=Reversibility.REVERSIBLE,
            user_id="user-001",
        )
        record = mgr.get_record(req.request_id)
        assert record is not None
        assert record.status == ApprovalStatus.AUTO_APPROVED
        assert record.metadata["auto_approved_reason"] == "trust_mode_window"

    def test_创建审批请求_安全操作不受信任模式影响(self) -> None:
        mgr = _make_approval_manager()
        mgr.activate_trust_mode("user-001", duration_seconds=1800.0)
        req = mgr.create_request(
            operation="data_overwrite",
            risk_level=RiskLevel.LOW,
            reversibility=Reversibility.REVERSIBLE,
            user_id="user-001",
        )
        record = mgr.get_record(req.request_id)
        assert record is not None
        assert record.status == ApprovalStatus.PENDING

    def test_创建审批请求_规则预设自动批准(self) -> None:
        mgr = _make_approval_manager()
        mgr.add_rule_preset("preset-001", "quiz_submit", RiskLevel.LOW)
        req = mgr.create_request(
            operation="quiz_submit",
            risk_level=RiskLevel.LOW,
        )
        record = mgr.get_record(req.request_id)
        assert record is not None
        assert record.status == ApprovalStatus.AUTO_APPROVED
        assert record.metadata["auto_approved_reason"] == "rule_preset"

    def test_审批决策_批准(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(operation="test_op")
        record = mgr.make_decision(
            req.request_id,
            decision=ApprovalStatus.APPROVED,
            decided_by="teacher-001",
        )
        assert record.status == ApprovalStatus.APPROVED
        assert record.decision is not None
        assert record.decision.decided_by == "teacher-001"

    def test_审批决策_拒绝(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(operation="test_op")
        record = mgr.make_decision(
            req.request_id,
            decision=ApprovalStatus.REJECTED,
        )
        assert record.status == ApprovalStatus.REJECTED

    def test_审批决策_修改(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(operation="test_op")
        record = mgr.make_decision(
            req.request_id,
            decision=ApprovalStatus.MODIFIED,
            modified_parameters={"param": "new_val"},
        )
        assert record.status == ApprovalStatus.MODIFIED
        assert record.decision.modified_parameters["param"] == "new_val"

    def test_审批决策_已决策报错(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(operation="test_op")
        mgr.make_decision(req.request_id, decision=ApprovalStatus.APPROVED)
        with pytest.raises(ValueError):
            mgr.make_decision(req.request_id, decision=ApprovalStatus.REJECTED)

    def test_审批决策_不存在报错(self) -> None:
        mgr = _make_approval_manager()
        with pytest.raises(KeyError):
            mgr.make_decision("nonexistent", decision=ApprovalStatus.APPROVED)

    def test_超时检查_中止策略(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(
            operation="test_op",
            timeout_seconds=0.01,
            timeout_action=TimeoutAction.ABORT,
        )
        time.sleep(0.02)
        record = mgr.check_timeout(req.request_id)
        assert record is not None
        assert record.status == ApprovalStatus.TIMEOUT
        assert record.metadata["timeout_action"] == "abort"

    def test_超时检查_自动批准策略(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(
            operation="test_op",
            timeout_seconds=0.01,
            timeout_action=TimeoutAction.AUTO_APPROVE,
        )
        time.sleep(0.02)
        record = mgr.check_timeout(req.request_id)
        assert record is not None
        assert record.status == ApprovalStatus.TIMEOUT
        assert record.metadata["timeout_action"] == "auto_approve"

    def test_超时检查_未超时返回None(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(operation="test_op", timeout_seconds=300.0)
        record = mgr.check_timeout(req.request_id)
        assert record is None

    def test_取消审批请求(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(operation="test_op")
        record = mgr.cancel_request(req.request_id)
        assert record is not None
        assert record.status == ApprovalStatus.CANCELLED

    def test_取消已决策请求返回None(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(operation="test_op")
        mgr.make_decision(req.request_id, decision=ApprovalStatus.APPROVED)
        record = mgr.cancel_request(req.request_id)
        assert record is None

    def test_列出审批记录_过滤(self) -> None:
        mgr = _make_approval_manager()
        mgr.create_request(operation="op_a", requester="r1")
        mgr.create_request(operation="op_b", requester="r2")
        results = mgr.list_records(operation="op_a")
        assert len(results) == 1
        assert results[0].request.operation == "op_a"

    def test_审批统计(self) -> None:
        mgr = _make_approval_manager()
        req = mgr.create_request(operation="test_op")
        mgr.make_decision(req.request_id, decision=ApprovalStatus.APPROVED)
        stats = mgr.get_statistics()
        assert stats["total"] == 1
        assert stats["by_status"]["approved"] == 1

    def test_信任模式管理(self) -> None:
        mgr = _make_approval_manager()
        mgr.activate_trust_mode("user-001")
        window = mgr.get_trust_mode("user-001")
        assert window is not None
        assert window.is_valid() is True
        mgr.deactivate_trust_mode("user-001")
        assert mgr.get_trust_mode("user-001").is_valid() is False

    def test_规则预设管理(self) -> None:
        mgr = _make_approval_manager()
        mgr.add_rule_preset("p1", "test_op", RiskLevel.LOW)
        req = mgr.create_request(operation="test_op", risk_level=RiskLevel.LOW)
        record = mgr.get_record(req.request_id)
        assert record.status == ApprovalStatus.AUTO_APPROVED
        mgr.remove_rule_preset("p1")
        req2 = mgr.create_request(operation="test_op", risk_level=RiskLevel.LOW)
        record2 = mgr.get_record(req2.request_id)
        assert record2.status == ApprovalStatus.PENDING

    def test_清空(self) -> None:
        mgr = _make_approval_manager()
        mgr.create_request(operation="test_op")
        mgr.activate_trust_mode("user-001")
        mgr.clear()
        assert len(mgr.records) == 0


# ============================================================
# 3. 测试抗疲劳机制
# ============================================================


class TestAntiFatigue:
    """审批抗疲劳机制完整测试."""

    # --- 枚举 ---

    def test_疲劳等级_四个值(self) -> None:
        assert len(FatigueLevel) == 4
        assert FatigueLevel.NONE.value == "none"
        assert FatigueLevel.MILD.value == "mild"
        assert FatigueLevel.MODERATE.value == "moderate"
        assert FatigueLevel.SEVERE.value == "severe"

    def test_批量状态_六个值(self) -> None:
        assert len(BatchStatus) == 6
        assert BatchStatus.COLLECTING.value == "collecting"
        assert BatchStatus.READY.value == "ready"
        assert BatchStatus.APPROVED.value == "approved"
        assert BatchStatus.REJECTED.value == "rejected"
        assert BatchStatus.PARTIAL.value == "partial"
        assert BatchStatus.EXPIRED.value == "expired"

    # --- 数据模型 ---

    def test_疲劳状态_默认值(self) -> None:
        state = FatigueState(user_id="user-001")
        assert state.approval_count == 0
        assert state.rejection_count == 0
        assert state.fatigue_score == 0.0
        assert state.fatigue_level == FatigueLevel.NONE

    def test_疲劳状态_总决策数(self) -> None:
        state = FatigueState(user_id="u1")
        state.approval_count = 3
        state.rejection_count = 1
        state.modification_count = 1
        state.timeout_count = 1
        assert state.total_decisions == 6

    def test_疲劳状态_自动批准率(self) -> None:
        state = FatigueState(user_id="u1")
        state.approval_count = 8
        state.rejection_count = 2
        assert state.auto_approve_rate == 0.8

    def test_疲劳状态_零决策自动批准率为零(self) -> None:
        state = FatigueState(user_id="u1")
        assert state.auto_approve_rate == 0.0

    def test_审批模式_批准率与拒绝率(self) -> None:
        pattern = ApprovalPattern(operation="test_op")
        pattern.total_count = 10
        pattern.approved_count = 8
        pattern.rejected_count = 2
        assert pattern.approval_rate == 0.8
        assert pattern.rejection_rate == 0.2

    def test_审批模式_高批准率判断(self) -> None:
        pattern = ApprovalPattern(operation="test_op")
        pattern.total_count = 5
        pattern.approved_count = 5
        assert pattern.is_high_approval is True

    def test_审批模式_非高批准率_样本不足(self) -> None:
        pattern = ApprovalPattern(operation="test_op")
        pattern.total_count = 3
        pattern.approved_count = 3
        assert pattern.is_high_approval is False

    def test_审批模式_非高批准率_率不足(self) -> None:
        pattern = ApprovalPattern(operation="test_op")
        pattern.total_count = 10
        pattern.approved_count = 8
        assert pattern.is_high_approval is False

    def test_批量审批组_过期判断(self) -> None:
        batch = BatchApprovalGroup(
            user_id="u1",
            operation="test_op",
            items=[{"a": 1}, {"b": 2}],
        )
        assert batch.is_expired is False
        assert batch.item_count == 2

    def test_批量审批组_已过期(self) -> None:
        batch = BatchApprovalGroup(
            user_id="u1",
            operation="test_op",
            created_at=time.time() - 400,
            expires_at=time.time() - 100,
        )
        assert batch.is_expired is True

    def test_渐进信任记录_默认值(self) -> None:
        trust = ProgressiveTrustRecord(user_id="u1")
        assert trust.trust_score == 0.5
        assert trust.base_trust == 0.5
        assert trust.consecutive_approvals == 0
        assert trust.promotions == 0
        assert trust.demotions == 0

    def test_疲劳配置_默认值(self) -> None:
        config = FatigueConfig()
        assert config.window_seconds == 3600.0
        assert config.max_approvals_per_window == 20
        assert config.batch_min_size == 3
        assert config.progressive_trust_threshold == 10
        assert config.smart_preapproval_min_samples == 5

    # --- 疲劳状态追踪 ---

    def test_追踪决策_批准(self) -> None:
        mgr = _make_anti_fatigue()
        state = mgr.track_decision("u1", "quiz_submit", "approved", 2.0)
        assert state.approval_count == 1
        assert state.fatigue_score >= 0

    def test_追踪决策_拒绝(self) -> None:
        mgr = _make_anti_fatigue()
        state = mgr.track_decision("u1", "quiz_submit", "rejected", 2.0)
        assert state.rejection_count == 1

    def test_追踪决策_修改(self) -> None:
        mgr = _make_anti_fatigue()
        state = mgr.track_decision("u1", "quiz_submit", "modified", 3.0)
        assert state.modification_count == 1

    def test_追踪决策_超时(self) -> None:
        mgr = _make_anti_fatigue()
        state = mgr.track_decision("u1", "quiz_submit", "timeout", 300.0)
        assert state.timeout_count == 1

    def test_追踪决策_自动批准(self) -> None:
        mgr = _make_anti_fatigue()
        mgr.track_decision("u1", "quiz_submit", "auto_approved", 0.0)
        pattern = mgr.get_approval_pattern("u1", "quiz_submit")
        assert pattern is not None
        assert pattern.auto_approved_count == 1
        assert pattern.approved_count == 1

    def test_获取疲劳状态(self) -> None:
        mgr = _make_anti_fatigue()
        state = mgr.get_fatigue_state("u1")
        assert state.user_id == "u1"
        assert state.fatigue_level == FatigueLevel.NONE

    def test_获取疲劳等级(self) -> None:
        mgr = _make_anti_fatigue()
        level = mgr.get_fatigue_level("u1")
        assert level == FatigueLevel.NONE

    def test_疲劳评分随审批增加(self) -> None:
        mgr = _make_anti_fatigue()
        mgr.track_decision("u1", "op", "approved", 1.0)
        score_after_1 = mgr.get_fatigue_state("u1").fatigue_score
        for _ in range(5):
            mgr.track_decision("u1", "op", "approved", 1.0)
        score_after_6 = mgr.get_fatigue_state("u1").fatigue_score
        assert score_after_6 >= score_after_1

    def test_疲劳评分_超时增加更多(self) -> None:
        mgr1 = _make_anti_fatigue()
        mgr2 = _make_anti_fatigue()
        for _ in range(3):
            mgr1.track_decision("u1", "op", "approved", 1.0)
            mgr2.track_decision("u1", "op", "timeout", 100.0)
        s1 = mgr1.get_fatigue_state("u1").fatigue_score
        s2 = mgr2.get_fatigue_state("u1").fatigue_score
        assert s2 > s1

    # --- 批量审批 ---

    def test_批量审批_未达最小值返回None(self) -> None:
        mgr = _make_anti_fatigue()
        result = mgr.add_to_batch("u1", "quiz_submit", {"id": 1})
        assert result is None

    def test_批量审批_达到最小值返回组(self) -> None:
        mgr = _make_anti_fatigue()
        mgr.add_to_batch("u1", "quiz_submit", {"id": 1})
        mgr.add_to_batch("u1", "quiz_submit", {"id": 2})
        batch = mgr.add_to_batch("u1", "quiz_submit", {"id": 3})
        assert batch is not None
        assert batch.status == BatchStatus.READY
        assert batch.item_count == 3

    def test_批量审批_获取就绪组(self) -> None:
        mgr = _make_anti_fatigue()
        for i in range(3):
            mgr.add_to_batch("u1", "quiz_submit", {"id": i})
        batches = mgr.get_ready_batches("u1")
        assert len(batches) == 1

    def test_批量审批_解决组_批准(self) -> None:
        mgr = _make_anti_fatigue()
        for i in range(3):
            mgr.add_to_batch("u1", "quiz_submit", {"id": i})
        batches = mgr.get_ready_batches("u1")
        batch_id = batches[0].batch_id
        result = mgr.resolve_batch(batch_id, "approved")
        assert result is not None
        assert result.status == BatchStatus.APPROVED

    def test_批量审批_解决组_拒绝(self) -> None:
        mgr = _make_anti_fatigue()
        for i in range(3):
            mgr.add_to_batch("u1", "quiz_submit", {"id": i})
        batches = mgr.get_ready_batches("u1")
        batch_id = batches[0].batch_id
        result = mgr.resolve_batch(batch_id, "rejected")
        assert result is not None
        assert result.status == BatchStatus.REJECTED

    def test_批量审批_过期清理(self) -> None:
        mgr = _make_anti_fatigue()
        for i in range(3):
            mgr.add_to_batch("u1", "quiz_submit", {"id": i})
        batches = mgr.get_ready_batches("u1")
        batches[0].expires_at = time.time() - 1
        count = mgr.expire_batches()
        assert count == 1

    # --- 智能预批 ---

    def test_智能预批_高风险返回False(self) -> None:
        mgr = _make_anti_fatigue()
        assert mgr.should_smart_preapprove("u1", "op", "high") is False

    def test_智能预批_样本不足返回False(self) -> None:
        mgr = _make_anti_fatigue()
        for _ in range(3):
            mgr.track_decision("u1", "op", "approved", 1.0, "low")
        assert mgr.should_smart_preapprove("u1", "op", "low") is False

    def test_智能预批_高批准率低风险高信任返回True(self) -> None:
        mgr = _make_anti_fatigue()
        # 手动提升信任到 0.7+
        for _ in range(4):
            mgr.promote_trust("u1")
        # 追踪 5 次批准
        for _ in range(5):
            mgr.track_decision("u1", "op", "approved", 1.0, "low")
        assert mgr.should_smart_preapprove("u1", "op", "low") is True

    def test_智能预批_信任不足返回False(self) -> None:
        mgr = _make_anti_fatigue()
        for _ in range(5):
            mgr.track_decision("u1", "op", "approved", 1.0, "low")
        # trust_score 默认 0.5 < 0.7
        assert mgr.should_smart_preapprove("u1", "op", "low") is False

    def test_获取审批模式(self) -> None:
        mgr = _make_anti_fatigue()
        mgr.track_decision("u1", "op", "approved", 1.0, "low")
        pattern = mgr.get_approval_pattern("u1", "op")
        assert pattern is not None
        assert pattern.operation == "op"
        assert pattern.total_count == 1

    # --- 渐进信任 ---

    def test_获取信任分(self) -> None:
        mgr = _make_anti_fatigue()
        assert mgr.get_trust_score("u1") == 0.5

    def test_获取信任记录(self) -> None:
        mgr = _make_anti_fatigue()
        trust = mgr.get_trust_record("u1")
        assert trust.user_id == "u1"
        assert trust.trust_score == 0.5

    def test_手动提升信任(self) -> None:
        mgr = _make_anti_fatigue()
        new_score = mgr.promote_trust("u1", reason="test")
        assert new_score == 0.55
        trust = mgr.get_trust_record("u1")
        assert trust.promotions == 1

    def test_手动降低信任(self) -> None:
        mgr = _make_anti_fatigue()
        new_score = mgr.demote_trust("u1", reason="test")
        assert new_score == 0.4
        trust = mgr.get_trust_record("u1")
        assert trust.demotions == 1

    def test_自动信任提升_连续批准(self) -> None:
        mgr = _make_anti_fatigue()
        for _ in range(10):
            mgr.track_decision("u1", "op", "approved", 1.0, "low")
        trust = mgr.get_trust_record("u1")
        assert trust.trust_score > 0.5
        assert trust.promotions >= 1

    def test_自动信任降级_拒绝后(self) -> None:
        mgr = _make_anti_fatigue()
        mgr.track_decision("u1", "op", "rejected", 1.0, "low")
        trust = mgr.get_trust_record("u1")
        assert trust.trust_score < 0.5
        assert trust.demotions >= 1

    def test_信任提升上限(self) -> None:
        mgr = _make_anti_fatigue()
        for _ in range(20):
            mgr.promote_trust("u1")
        assert mgr.get_trust_score("u1") <= 0.95

    # --- 疲劳调整建议 ---

    def test_疲劳调整_无疲劳(self) -> None:
        mgr = _make_anti_fatigue()
        adj = mgr.get_fatigue_adjustment("u1")
        assert adj["fatigue_level"] == "none"
        assert adj["recommendations"] == []

    def test_疲劳调整_轻度(self) -> None:
        mgr = _make_anti_fatigue()
        state = mgr.get_fatigue_state("u1")
        state.fatigue_score = 30.0
        state.fatigue_level = FatigueLevel.MILD
        adj = mgr.get_fatigue_adjustment("u1")
        assert "enable_batch_approval" in adj["recommendations"]

    def test_疲劳调整_重度(self) -> None:
        mgr = _make_anti_fatigue()
        state = mgr.get_fatigue_state("u1")
        state.fatigue_score = 80.0
        state.fatigue_level = FatigueLevel.SEVERE
        adj = mgr.get_fatigue_adjustment("u1")
        assert "downgrade_to_l2_prompt" in adj["recommendations"]

    # --- 统计与清理 ---

    def test_抗疲劳统计(self) -> None:
        mgr = _make_anti_fatigue()
        mgr.track_decision("u1", "op", "approved", 1.0)
        mgr.track_decision("u2", "op", "rejected", 1.0)
        stats = mgr.get_statistics()
        assert stats["total_users"] == 2

    def test_抗疲劳统计_空(self) -> None:
        mgr = _make_anti_fatigue()
        stats = mgr.get_statistics()
        assert stats == {"total_users": 0}

    def test_清空(self) -> None:
        mgr = _make_anti_fatigue()
        mgr.track_decision("u1", "op", "approved", 1.0)
        mgr.clear()
        stats = mgr.get_statistics()
        assert stats["total_users"] == 0


# ============================================================
# 4. 测试干预管理器
# ============================================================


class TestInterventionManager:
    """L4 干预层管理器完整测试."""

    # --- 枚举 ---

    def test_干预动作_六个值(self) -> None:
        assert len(InterventionAction) == 6
        assert InterventionAction.PAUSE.value == "pause"
        assert InterventionAction.RESUME.value == "resume"
        assert InterventionAction.OVERRIDE.value == "override"
        assert InterventionAction.CORRECT.value == "correct"
        assert InterventionAction.REDIRECT.value == "redirect"
        assert InterventionAction.TERMINATE.value == "terminate"

    def test_暂停范围_四个值(self) -> None:
        assert len(PauseScope) == 4
        assert PauseScope.AGENT.value == "agent"
        assert PauseScope.SESSION.value == "session"
        assert PauseScope.MODULE.value == "module"
        assert PauseScope.GLOBAL.value == "global"

    def test_接管级别_三个值(self) -> None:
        assert len(OverrideLevel) == 3
        assert OverrideLevel.ADVISORY.value == "advisory"
        assert OverrideLevel.EXECUTIVE.value == "executive"
        assert OverrideLevel.ABSOLUTE.value == "absolute"

    def test_纠正类型_四个值(self) -> None:
        assert len(CorrectionType) == 4
        assert CorrectionType.FACTUAL.value == "factual"
        assert CorrectionType.PROCEDURAL.value == "procedural"
        assert CorrectionType.CONCEPTUAL.value == "conceptual"
        assert CorrectionType.PEDAGOGICAL.value == "pedagogical"

    def test_纠正严重度_四个值(self) -> None:
        assert len(CorrectionSeverity) == 4
        assert CorrectionSeverity.MINOR.value == "minor"
        assert CorrectionSeverity.MODERATE.value == "moderate"
        assert CorrectionSeverity.MAJOR.value == "major"
        assert CorrectionSeverity.CRITICAL.value == "critical"

    def test_创意请求类型_四个值(self) -> None:
        assert len(CreativeRequestType) == 4
        assert CreativeRequestType.BRAINSTORM.value == "brainstorm"
        assert CreativeRequestType.ALTERNATIVE.value == "alternative"
        assert CreativeRequestType.EXAMPLE.value == "example"
        assert CreativeRequestType.METAPHOR.value == "metaphor"

    def test_干预事件状态_四个值(self) -> None:
        assert len(InterventionEventStatus) == 4
        assert InterventionEventStatus.INITIATED.value == "initiated"
        assert InterventionEventStatus.ACTIVE.value == "active"
        assert InterventionEventStatus.RESOLVED.value == "resolved"
        assert InterventionEventStatus.CANCELLED.value == "cancelled"

    # --- 紧急暂停 ---

    def test_发起紧急暂停(self) -> None:
        mgr = _make_intervention_manager()
        pause = mgr.initiate_emergency_pause(
            user_id="student-001",
            reason="感到困惑",
            scope=PauseScope.SESSION,
            agent_ids=["tutor-agent"],
        )
        assert pause.user_id == "student-001"
        assert pause.reason == "感到困惑"
        assert pause.status == InterventionEventStatus.ACTIVE
        assert pause.is_active is True

    def test_解决紧急暂停(self) -> None:
        mgr = _make_intervention_manager()
        pause = mgr.initiate_emergency_pause("student-001", "困惑")
        time.sleep(0.01)
        resolved = mgr.resolve_emergency_pause(
            pause.pause_id, "已解决", "teacher-001"
        )
        assert resolved.status == InterventionEventStatus.RESOLVED
        assert resolved.is_active is False
        assert resolved.duration_seconds > 0

    def test_解决紧急暂停_不存在报错(self) -> None:
        mgr = _make_intervention_manager()
        with pytest.raises(KeyError):
            mgr.resolve_emergency_pause("nonexistent", "ok", "t1")

    def test_解决紧急暂停_已解决报错(self) -> None:
        mgr = _make_intervention_manager()
        pause = mgr.initiate_emergency_pause("s1", "r")
        mgr.resolve_emergency_pause(pause.pause_id, "ok", "t1")
        with pytest.raises(ValueError):
            mgr.resolve_emergency_pause(pause.pause_id, "ok2", "t1")

    # --- 人工接管 ---

    def test_发起人工接管(self) -> None:
        mgr = _make_intervention_manager()
        override = mgr.initiate_manual_override(
            operator_id="teacher-001",
            target_agent="tutor-agent",
            override_level=OverrideLevel.EXECUTIVE,
            instructions="切换教学模式",
        )
        assert override.operator_id == "teacher-001"
        assert override.target_agent == "tutor-agent"
        assert override.status == InterventionEventStatus.ACTIVE

    def test_释放人工接管(self) -> None:
        mgr = _make_intervention_manager()
        override = mgr.initiate_manual_override("t1", "agent-1")
        released = mgr.release_override(override.override_id, "完成", "t1")
        assert released.status == InterventionEventStatus.RESOLVED
        assert released.released is True

    def test_释放人工接管_不存在报错(self) -> None:
        mgr = _make_intervention_manager()
        with pytest.raises(KeyError):
            mgr.release_override("nonexistent")

    # --- 纠正反馈 ---

    def test_提交纠正反馈(self) -> None:
        mgr = _make_intervention_manager()
        correction = mgr.submit_correction(
            corrector_id="teacher-001",
            target_content_id="content-001",
            original="原始内容",
            corrected="纠正内容",
            correction_type=CorrectionType.FACTUAL,
            feedback="事实错误",
        )
        assert correction.corrector_id == "teacher-001"
        assert correction.applied is False

    def test_应用纠正反馈(self) -> None:
        mgr = _make_intervention_manager()
        correction = mgr.submit_correction(
            corrector_id="t1",
            target_content_id="c1",
            original="old",
            corrected="new",
            correction_type=CorrectionType.FACTUAL,
            severity=CorrectionSeverity.MAJOR,
        )
        applied = mgr.apply_correction(correction.correction_id, "t1")
        assert applied.applied is True
        assert applied.cc1_re_review_triggered is True

    def test_纠正_严重度触发CC1(self) -> None:
        mgr = _make_intervention_manager()
        for sev in [CorrectionSeverity.CRITICAL, CorrectionSeverity.MAJOR]:
            c = mgr.submit_correction(
                corrector_id="t1",
                target_content_id="c1",
                original="o",
                corrected="n",
                correction_type=CorrectionType.FACTUAL,
                severity=sev,
            )
            assert c.should_trigger_cc1 is True

    def test_纠正_中等严重度事实性触发CC1(self) -> None:
        mgr = _make_intervention_manager()
        c = mgr.submit_correction(
            corrector_id="t1",
            target_content_id="c1",
            original="o",
            corrected="n",
            correction_type=CorrectionType.FACTUAL,
            severity=CorrectionSeverity.MODERATE,
        )
        assert c.should_trigger_cc1 is True

    def test_纠正_轻微不触发CC1(self) -> None:
        mgr = _make_intervention_manager()
        c = mgr.submit_correction(
            corrector_id="t1",
            target_content_id="c1",
            original="o",
            corrected="n",
            correction_type=CorrectionType.PEDAGOGICAL,
            severity=CorrectionSeverity.MINOR,
        )
        assert c.should_trigger_cc1 is False

    def test_应用纠正_已应用报错(self) -> None:
        mgr = _make_intervention_manager()
        c = mgr.submit_correction("t1", "c1", "o", "n", CorrectionType.FACTUAL)
        mgr.apply_correction(c.correction_id, "t1")
        with pytest.raises(ValueError):
            mgr.apply_correction(c.correction_id, "t1")

    # --- 创意请求 ---

    def test_发起创意请求(self) -> None:
        mgr = _make_intervention_manager()
        req = mgr.request_creative_input(
            requester_id="teacher-001",
            request_type=CreativeRequestType.BRAINSTORM,
            topic="量子力学",
            constraints=["面向高中生"],
        )
        assert req.requester_id == "teacher-001"
        assert req.topic == "量子力学"
        assert req.status == InterventionEventStatus.ACTIVE

    def test_响应创意请求(self) -> None:
        mgr = _make_intervention_manager()
        req = mgr.request_creative_input(
            requester_id="t1",
            request_type=CreativeRequestType.METAPHOR,
            topic="叠加态",
        )
        responded = mgr.respond_creative_request(
            req.request_id, "tutor-agent", "想象旋转的硬币..."
        )
        assert responded.status == InterventionEventStatus.RESOLVED
        assert responded.is_responded is True
        assert responded.response_content == "想象旋转的硬币..."

    def test_响应创意请求_不存在报错(self) -> None:
        mgr = _make_intervention_manager()
        with pytest.raises(KeyError):
            mgr.respond_creative_request("nonexistent", "a1", "content")

    # --- 查询与统计 ---

    def test_获取活跃干预(self) -> None:
        mgr = _make_intervention_manager()
        mgr.initiate_emergency_pause("s1", "r")
        mgr.initiate_manual_override("t1", "a1")
        active = mgr.get_active_interventions()
        assert len(active) == 2

    def test_获取干预历史(self) -> None:
        mgr = _make_intervention_manager()
        mgr.initiate_emergency_pause("s1", "r")
        history = mgr.get_intervention_history()
        assert len(history) == 1

    def test_获取干预历史_按类型过滤(self) -> None:
        mgr = _make_intervention_manager()
        mgr.initiate_emergency_pause("s1", "r")
        mgr.initiate_manual_override("t1", "a1")
        history = mgr.get_intervention_history(
            event_type=InterventionTypeL4.EMERGENCY_PAUSE
        )
        assert len(history) == 1

    def test_干预统计(self) -> None:
        mgr = _make_intervention_manager()
        mgr.initiate_emergency_pause("s1", "r")
        stats = mgr.get_statistics()
        assert stats["total_events"] == 1
        assert stats["active_count"] == 1
        assert stats["total_initiated"] == 1

    def test_干预统计_空(self) -> None:
        mgr = _make_intervention_manager()
        stats = mgr.get_statistics()
        assert stats["total_events"] == 0

    # --- 通知回调 ---

    def test_通知回调_紧急暂停(self) -> None:
        mgr = _make_intervention_manager()
        notifications: list[dict] = []
        mgr.register_notification_callback(lambda p: notifications.append(p))
        mgr.initiate_emergency_pause("s1", "r", auto_notify_teacher=True)
        assert len(notifications) == 1
        assert notifications[0]["type"] == "emergency_pause"

    def test_通知回调_不通知(self) -> None:
        mgr = _make_intervention_manager()
        notifications: list[dict] = []
        mgr.register_notification_callback(lambda p: notifications.append(p))
        mgr.initiate_emergency_pause("s1", "r", auto_notify_teacher=False)
        assert len(notifications) == 0

    def test_注销通知回调(self) -> None:
        mgr = _make_intervention_manager()
        notifications: list[dict] = []
        cb = lambda p: notifications.append(p)
        mgr.register_notification_callback(cb)
        mgr.unregister_notification_callback(cb)
        mgr.initiate_emergency_pause("s1", "r")
        assert len(notifications) == 0

    # --- 清理 ---

    def test_清空(self) -> None:
        mgr = _make_intervention_manager()
        mgr.initiate_emergency_pause("s1", "r")
        mgr.clear()
        assert len(mgr.get_active_interventions()) == 0
        stats = mgr.get_statistics()
        assert stats["total_events"] == 0


# ============================================================
# 5. 测试 KPI 指标引擎
# ============================================================


class TestKPIMetrics:
    """KPI 指标引擎完整测试."""

    # --- 枚举 ---

    def test_KPI分类_四个值(self) -> None:
        assert len(KPICategory) == 4
        assert KPICategory.EFFICIENCY.value == "efficiency"
        assert KPICategory.SAFETY.value == "safety"
        assert KPICategory.QUALITY.value == "quality"
        assert KPICategory.TRUST.value == "trust"

    def test_KPI状态_三个值(self) -> None:
        assert len(KPIStatus) == 3
        assert KPIStatus.GREEN.value == "green"
        assert KPIStatus.YELLOW.value == "yellow"
        assert KPIStatus.RED.value == "red"

    def test_趋势方向_三个值(self) -> None:
        assert len(TrendDirection) == 3
        assert TrendDirection.UP.value == "up"
        assert TrendDirection.DOWN.value == "down"
        assert TrendDirection.STABLE.value == "stable"

    # --- 数据模型 ---

    def test_KPI采样_创建(self) -> None:
        sample = KPISample(kpi_name="fatigue_index", value=30.0)
        assert sample.kpi_name == "fatigue_index"
        assert sample.value == 30.0

    def test_KPI阈值_创建(self) -> None:
        threshold = KPIThreshold(
            kpi_name="test", green_max=10, yellow_max=20, red_min=20
        )
        assert threshold.green_max == 10
        assert threshold.is_inverse is False

    def test_KPI趋势_创建(self) -> None:
        trend = KPITrend(
            kpi_name="test",
            direction=TrendDirection.UP,
            change_percent=15.0,
            window_samples=10,
            confidence=0.8,
        )
        assert trend.direction == TrendDirection.UP

    def test_KPI汇总_创建(self) -> None:
        summary = KPISummary(
            kpi_name="test",
            category=KPICategory.EFFICIENCY,
            target_value=30.0,
            threshold_yellow=120.0,
            threshold_red=120.0,
        )
        assert summary.kpi_name == "test"
        assert summary.status == KPIStatus.GREEN

    # --- 9 项 KPI ---

    def test_九项KPI全部注册(self) -> None:
        engine = _make_kpi_engine()
        assert len(engine.tracked_kpis) == 9
        assert len(KPI_NAMES) == 9

    def test_获取所有汇总(self) -> None:
        engine = _make_kpi_engine()
        summaries = engine.get_all_summaries()
        assert len(summaries) == 9
        for name in KPI_NAMES:
            assert name in summaries

    # --- 采样与汇总 ---

    def test_记录采样与汇总(self) -> None:
        engine = _make_kpi_engine()
        engine.record_sample("fatigue_index", 10.0)
        summary = engine.get_kpi_summary("fatigue_index")
        assert summary.current_value == 10.0
        assert summary.samples_count == 1
        assert summary.status == KPIStatus.GREEN

    def test_记录采样_未知KPI报错(self) -> None:
        engine = _make_kpi_engine()
        with pytest.raises(KeyError):
            engine.record_sample("unknown_kpi", 1.0)

    def test_获取历史(self) -> None:
        engine = _make_kpi_engine()
        engine.record_sample("fatigue_index", 10.0)
        engine.record_sample("fatigue_index", 20.0)
        history = engine.get_kpi_history("fatigue_index")
        assert len(history) == 2
        assert history[0].value == 10.0
        assert history[1].value == 20.0

    # --- 阈值检查 ---

    def test_阈值检查_逆指标_绿色(self) -> None:
        engine = _make_kpi_engine()
        # fatigue_index: inverse, green_max=25, yellow_max=50
        assert engine.check_threshold("fatigue_index", 10.0) == KPIStatus.GREEN

    def test_阈值检查_逆指标_黄色(self) -> None:
        engine = _make_kpi_engine()
        assert engine.check_threshold("fatigue_index", 30.0) == KPIStatus.YELLOW

    def test_阈值检查_逆指标_红色(self) -> None:
        engine = _make_kpi_engine()
        assert engine.check_threshold("fatigue_index", 60.0) == KPIStatus.RED

    def test_阈值检查_正向指标_绿色(self) -> None:
        engine = _make_kpi_engine()
        # auto_approval_rate: non-inverse, green_max=60, yellow_max=30
        assert engine.check_threshold("auto_approval_rate", 70.0) == KPIStatus.GREEN

    def test_阈值检查_正向指标_黄色(self) -> None:
        engine = _make_kpi_engine()
        assert engine.check_threshold("auto_approval_rate", 40.0) == KPIStatus.YELLOW

    def test_阈值检查_正向指标_红色(self) -> None:
        engine = _make_kpi_engine()
        assert engine.check_threshold("auto_approval_rate", 20.0) == KPIStatus.RED

    # --- 趋势分析 ---

    def test_趋势_上升(self) -> None:
        engine = _make_kpi_engine()
        for v in [10, 20, 30, 40, 50, 60]:
            engine.record_sample("fatigue_index", float(v))
        trend = engine.compute_trend("fatigue_index")
        assert trend.direction == TrendDirection.UP

    def test_趋势_下降(self) -> None:
        engine = _make_kpi_engine()
        for v in [60, 50, 40, 30, 20, 10]:
            engine.record_sample("fatigue_index", float(v))
        trend = engine.compute_trend("fatigue_index")
        assert trend.direction == TrendDirection.DOWN

    def test_趋势_稳定(self) -> None:
        engine = _make_kpi_engine()
        for _ in range(6):
            engine.record_sample("fatigue_index", 25.0)
        trend = engine.compute_trend("fatigue_index")
        assert trend.direction == TrendDirection.STABLE

    def test_趋势_样本不足返回稳定(self) -> None:
        engine = _make_kpi_engine()
        trend = engine.compute_trend("fatigue_index")
        assert trend.direction == TrendDirection.STABLE
        assert trend.confidence == 0.0

    # --- 动态阈值调整 ---

    def test_调整阈值_手动(self) -> None:
        engine = _make_kpi_engine()
        new_threshold = engine.adjust_threshold(
            "fatigue_index", green_max=20.0, yellow_max=40.0
        )
        assert new_threshold.green_max == 20.0
        assert new_threshold.yellow_max == 40.0

    def test_调整阈值_自动计算(self) -> None:
        engine = _make_kpi_engine()
        for v in range(10, 30):
            engine.record_sample("fatigue_index", float(v))
        new_threshold = engine.adjust_threshold("fatigue_index")
        assert new_threshold.green_max > 0
        assert new_threshold.yellow_max > 0

    def test_重置阈值(self) -> None:
        engine = _make_kpi_engine()
        engine.adjust_threshold("fatigue_index", green_max=999, yellow_max=999)
        engine.reset_thresholds()
        threshold = engine.get_threshold("fatigue_index")
        assert threshold.green_max == 25.0

    # --- 仪表盘 ---

    def test_仪表盘数据(self) -> None:
        engine = _make_kpi_engine()
        engine.record_sample("fatigue_index", 10.0)
        data = engine.get_dashboard_data()
        assert "overall_health_score" in data
        assert "category_scores" in data
        assert "alerts" in data
        assert "statistics" in data
        assert "kpis" in data

    def test_仪表盘_健康分范围(self) -> None:
        engine = _make_kpi_engine()
        data = engine.get_dashboard_data()
        assert 0 <= data["overall_health_score"] <= 100

    # --- 告警 ---

    def test_获取告警_无告警(self) -> None:
        engine = _make_kpi_engine()
        engine.record_sample("fatigue_index", 10.0)
        alerts = engine.get_alerts()
        assert len(alerts) == 0

    def test_获取告警_有告警(self) -> None:
        engine = _make_kpi_engine()
        engine.record_sample("fatigue_index", 60.0)
        alerts = engine.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["status"] == "red"

    # --- 统计 ---

    def test_统计信息(self) -> None:
        engine = _make_kpi_engine()
        engine.record_sample("fatigue_index", 10.0)
        stats = engine.get_statistics()
        assert stats["total_samples"] == 1
        assert stats["tracked_kpis"] == 9
        assert "status_distribution" in stats

    # --- 集成接口 ---

    def test_集成_从抗疲劳接收(self) -> None:
        engine = _make_kpi_engine()
        sample = engine.ingest_from_anti_fatigue(30.0)
        assert sample.kpi_name == "fatigue_index"
        assert sample.value == 30.0

    def test_集成_从审批工作流接收(self) -> None:
        engine = _make_kpi_engine()
        samples = engine.ingest_from_approval_workflow(
            response_time=25.0, auto_approved=True, rejected=False
        )
        assert len(samples) == 3

    def test_集成_从路由引擎接收(self) -> None:
        engine = _make_kpi_engine()
        samples = engine.ingest_from_routing_engine(
            intervention_count=1, cc1_triggered=True, trust_score=0.8
        )
        assert len(samples) == 3

    # --- 清理 ---

    def test_清空(self) -> None:
        engine = _make_kpi_engine()
        engine.record_sample("fatigue_index", 10.0)
        engine.clear()
        summary = engine.get_kpi_summary("fatigue_index")
        assert summary.samples_count == 0

    def test_清空单个KPI(self) -> None:
        engine = _make_kpi_engine()
        engine.record_sample("fatigue_index", 10.0)
        engine.clear_kpi("fatigue_index")
        summary = engine.get_kpi_summary("fatigue_index")
        assert summary.samples_count == 0


# ============================================================
# 6. 测试引擎集成
# ============================================================


class TestEngineIntegration:
    """CollaborationEngine 增强组件集成测试."""

    def _make_full_engine(self) -> CollaborationEngine:
        """创建挂载全部增强组件的引擎."""
        engine = CollaborationEngine()
        engine.attach_routing_engine(RoutingEngine())
        engine.attach_approval_manager(ApprovalWorkflowManager())
        engine.attach_anti_fatigue(AntiFatigueManager())
        engine.attach_intervention_manager(InterventionManager())
        engine.attach_kpi_engine(KPIMetricsEngine())
        return engine

    # --- 组件挂载 ---

    def test_挂载路由引擎(self) -> None:
        engine = CollaborationEngine()
        engine.attach_routing_engine(RoutingEngine())
        stats = engine.get_enhanced_stats()
        assert stats["components_attached"]["routing_engine"] is True

    def test_挂载审批管理器(self) -> None:
        engine = CollaborationEngine()
        engine.attach_approval_manager(ApprovalWorkflowManager())
        stats = engine.get_enhanced_stats()
        assert stats["components_attached"]["approval_manager"] is True

    def test_挂载抗疲劳管理器(self) -> None:
        engine = CollaborationEngine()
        engine.attach_anti_fatigue(AntiFatigueManager())
        stats = engine.get_enhanced_stats()
        assert stats["components_attached"]["anti_fatigue"] is True

    def test_挂载干预管理器(self) -> None:
        engine = CollaborationEngine()
        engine.attach_intervention_manager(InterventionManager())
        stats = engine.get_enhanced_stats()
        assert stats["components_attached"]["intervention_manager"] is True

    def test_挂载KPI引擎(self) -> None:
        engine = CollaborationEngine()
        engine.attach_kpi_engine(KPIMetricsEngine())
        stats = engine.get_enhanced_stats()
        assert stats["components_attached"]["kpi_engine"] is True

    def test_挂载CC1回调(self) -> None:
        engine = CollaborationEngine()
        engine.attach_cc1_callback(lambda r: None)
        stats = engine.get_enhanced_stats()
        assert stats["components_attached"]["cc1_callback"] is True

    # --- 路由决策 ---

    def test_路由决策_未挂载返回默认L1(self) -> None:
        engine = CollaborationEngine()
        ctx = RoutingContext()
        result = engine.route_decision(ctx)
        assert result.recommended_layer == CollaborationLayer.L1_IMPLICIT

    def test_路由决策_已挂载执行路由(self) -> None:
        engine = self._make_full_engine()
        ctx = RoutingContext(operation_type="emergency_pause")
        result = engine.route_decision(ctx)
        assert result.recommended_layer == CollaborationLayer.L4_INTERVENTION

    # --- 审批请求 ---

    def test_审批请求_未挂载返回None(self) -> None:
        engine = CollaborationEngine()
        result = engine.request_approval("test_op")
        assert result is None

    def test_审批请求_已挂载创建请求(self) -> None:
        engine = self._make_full_engine()
        req = engine.request_approval(
            "test_op", risk_level=RiskLevel.LOW, requester="agent-1"
        )
        assert req is not None
        assert req.operation == "test_op"

    def test_审批请求_智能预批(self) -> None:
        engine = self._make_full_engine()
        # 提升信任 + 追踪历史
        for _ in range(4):
            engine._anti_fatigue.promote_trust("user-1")
        for _ in range(5):
            engine._anti_fatigue.track_decision(
                "user-1", "quiz_submit", "approved", 1.0, "low"
            )
        req = engine.request_approval(
            "quiz_submit",
            risk_level=RiskLevel.LOW,
            requester="agent-1",
            user_id="user-1",
        )
        record = engine._approval_manager.get_record(req.request_id)
        assert record.status == ApprovalStatus.AUTO_APPROVED

    # --- CC1 结果处理 ---

    def test_CC1结果_pass正常执行(self) -> None:
        engine = self._make_full_engine()
        result = engine.process_cc1_result("agent-1", "pass", 0.9)
        assert result["action"] == "proceed"

    def test_CC1结果_warn降级L2(self) -> None:
        engine = self._make_full_engine()
        result = engine.process_cc1_result("agent-1", "warn", 0.6)
        assert result["action"] == "downgrade_to_l2_prompt"

    def test_CC1结果_block升级L3(self) -> None:
        engine = self._make_full_engine()
        result = engine.process_cc1_result("agent-1", "block", 0.3)
        assert result["action"] == "escalate_to_l3_approval"
        assert result["routing_layer"] == "l3_approval"

    # --- L4 干预委托 ---

    def test_紧急暂停委托(self) -> None:
        engine = self._make_full_engine()
        pause = engine.emergency_pause("student-1", "困惑")
        assert pause is not None
        assert pause.user_id == "student-1"

    def test_紧急暂停_未挂载返回None(self) -> None:
        engine = CollaborationEngine()
        assert engine.emergency_pause("s1", "r") is None

    def test_人工接管委托(self) -> None:
        engine = self._make_full_engine()
        override = engine.manual_override("teacher-1", "agent-1")
        assert override is not None
        assert override.operator_id == "teacher-1"

    def test_纠错反馈委托_CC1触发(self) -> None:
        engine = self._make_full_engine()
        cc1_calls: list[dict] = []
        engine.attach_cc1_callback(lambda r: cc1_calls.append(r))
        correction = engine.submit_correction(
            corrector_id="teacher-1",
            target_content_id="c1",
            original="old",
            corrected="new",
            correction_type=CorrectionType.FACTUAL,
            severity=CorrectionSeverity.CRITICAL,
        )
        assert correction is not None
        assert correction.applied is True
        assert correction.cc1_re_review_triggered is True
        assert len(cc1_calls) >= 1

    # --- KPI 仪表盘与统计 ---

    def test_KPI仪表盘(self) -> None:
        engine = self._make_full_engine()
        dashboard = engine.get_kpi_dashboard()
        assert "overall_health_score" in dashboard

    def test_KPI仪表盘_未挂载返回空(self) -> None:
        engine = CollaborationEngine()
        assert engine.get_kpi_dashboard() == {}

    def test_增强统计(self) -> None:
        engine = self._make_full_engine()
        stats = engine.get_enhanced_stats()
        assert "components_attached" in stats
        assert "routing" in stats
        assert "approval" in stats
        assert "anti_fatigue" in stats
        assert "intervention" in stats
        assert "kpi" in stats

    # --- 清理 ---

    def test_清空所有组件(self) -> None:
        engine = self._make_full_engine()
        engine.emergency_pause("s1", "r")
        engine.route_decision(RoutingContext())
        engine.clear()
        stats = engine.get_enhanced_stats()
        assert stats["routing"]["total"] == 0
        assert stats["intervention"]["total_events"] == 0


# ============================================================
# 7. 测试 REST API
# ============================================================


class TestAPI:
    """CC2APIRouter REST API 完整测试."""

    def _make_router(self) -> CC2APIRouter:
        """创建测试用 API 路由器."""
        engine = CollaborationEngine()
        return CC2APIRouter(engine)

    def _make_full_router(self) -> CC2APIRouter:
        """创建挂载全部组件的 API 路由器."""
        engine = CollaborationEngine()
        engine.attach_routing_engine(RoutingEngine())
        engine.attach_approval_manager(ApprovalWorkflowManager())
        engine.attach_anti_fatigue(AntiFatigueManager())
        engine.attach_intervention_manager(InterventionManager())
        engine.attach_kpi_engine(KPIMetricsEngine())
        return CC2APIRouter(
            engine,
            routing_engine=engine._routing_engine,
            approval_manager=engine._approval_manager,
            intervention_manager=engine._intervention_manager,
            anti_fatigue_manager=engine._anti_fatigue,
            kpi_engine=engine._kpi_engine,
        )

    # --- 健康检查 ---

    def test_健康检查(self) -> None:
        router = self._make_router()
        result = router.health()
        assert result["code"] == 200
        assert result["data"]["status"] == "healthy"

    def test_就绪检查(self) -> None:
        router = self._make_router()
        result = router.health_ready()
        assert result["code"] == 200
        assert "subsystems" in result["data"]
        assert result["data"]["ready"] is True

    # --- 路由决策 ---

    def test_路由端点(self) -> None:
        router = self._make_router()
        result = router.route({"operation_type": "emergency_pause"})
        assert result["code"] == 200
        assert result["data"]["recommended_layer"] == "l4_intervention"

    def test_路由端点_错误输入(self) -> None:
        router = self._make_router()
        result = router.route({"confidence": 999})
        assert result["code"] == 400

    def test_路由统计端点(self) -> None:
        router = self._make_router()
        router.route({"operation_type": "emergency_pause"})
        result = router.get_route_statistics()
        assert result["code"] == 200
        assert result["data"]["total"] == 1

    def test_路由规则端点(self) -> None:
        router = self._make_router()
        result = router.get_route_rules()
        assert result["code"] == 200
        assert len(result["data"]) == 12

    # --- 审批工作流 ---

    def test_创建审批端点(self) -> None:
        router = self._make_router()
        result = router.create_approval({"operation": "test_op"})
        assert result["code"] == 200
        assert result["data"]["operation"] == "test_op"

    def test_审批决策端点(self) -> None:
        router = self._make_router()
        create_result = router.create_approval({"operation": "test_op"})
        req_id = create_result["data"]["request_id"]
        result = router.make_approval_decision(
            req_id, {"decision": "approved", "decided_by": "t1"}
        )
        assert result["code"] == 200
        assert result["data"]["status"] == "approved"

    def test_获取审批端点_不存在(self) -> None:
        router = self._make_router()
        result = router.get_approval("nonexistent")
        assert result["code"] == 404

    def test_审批统计端点(self) -> None:
        router = self._make_router()
        router.create_approval({"operation": "test_op"})
        result = router.get_approval_statistics()
        assert result["code"] == 200
        assert result["data"]["total"] == 1

    def test_信任模式激活端点(self) -> None:
        router = self._make_router()
        result = router.activate_trust_mode({"user_id": "u1"})
        assert result["code"] == 200
        assert result["data"]["active"] is True

    def test_信任模式激活_缺少user_id(self) -> None:
        router = self._make_router()
        result = router.activate_trust_mode({})
        assert result["code"] == 400

    # --- 干预管理 ---

    def test_紧急暂停端点(self) -> None:
        router = self._make_router()
        result = router.initiate_emergency_pause({
            "user_id": "s1",
            "reason": "困惑",
        })
        assert result["code"] == 200
        assert result["data"]["user_id"] == "s1"

    def test_紧急暂停_缺少user_id(self) -> None:
        router = self._make_router()
        result = router.initiate_emergency_pause({"reason": "r"})
        assert result["code"] == 400

    def test_解决紧急暂停端点(self) -> None:
        router = self._make_router()
        pause_result = router.initiate_emergency_pause({"user_id": "s1"})
        pause_id = pause_result["data"]["pause_id"]
        result = router.resolve_emergency_pause(
            pause_id, {"resolution": "ok", "resolved_by": "t1"}
        )
        assert result["code"] == 200
        assert result["data"]["status"] == "resolved"

    def test_人工接管端点(self) -> None:
        router = self._make_router()
        result = router.initiate_manual_override({
            "operator_id": "t1",
            "target_agent": "a1",
        })
        assert result["code"] == 200

    def test_纠错端点(self) -> None:
        router = self._make_router()
        result = router.submit_correction({
            "corrector_id": "t1",
            "target_content_id": "c1",
            "correction_type": "factual",
            "original": "old",
            "corrected": "new",
        })
        assert result["code"] == 200

    def test_创意请求端点(self) -> None:
        router = self._make_router()
        result = router.request_creative_input({
            "requester_id": "t1",
            "request_type": "brainstorm",
            "topic": "AI",
        })
        assert result["code"] == 200

    def test_活跃干预端点(self) -> None:
        router = self._make_router()
        router.initiate_emergency_pause({"user_id": "s1"})
        result = router.get_active_interventions()
        assert result["code"] == 200
        assert len(result["data"]) == 1

    def test_干预历史端点(self) -> None:
        router = self._make_router()
        router.initiate_emergency_pause({"user_id": "s1"})
        result = router.get_intervention_history()
        assert result["code"] == 200
        assert len(result["data"]) == 1

    def test_干预统计端点(self) -> None:
        router = self._make_router()
        router.initiate_emergency_pause({"user_id": "s1"})
        result = router.get_intervention_statistics()
        assert result["code"] == 200
        assert result["data"]["total_events"] == 1

    # --- 抗疲劳 ---

    def test_疲劳状态端点(self) -> None:
        router = self._make_router()
        result = router.get_fatigue_state("u1")
        assert result["code"] == 200
        assert result["data"]["user_id"] == "u1"

    def test_疲劳调整建议端点(self) -> None:
        router = self._make_router()
        result = router.get_fatigue_adjustment("u1")
        assert result["code"] == 200

    def test_批量审批添加端点(self) -> None:
        router = self._make_router()
        result = router.add_to_batch({
            "user_id": "u1",
            "operation": "quiz_submit",
            "request_data": {"id": 1},
        })
        assert result["code"] == 200

    def test_信任记录端点(self) -> None:
        router = self._make_router()
        result = router.get_trust_record("u1")
        assert result["code"] == 200

    def test_提升信任端点(self) -> None:
        router = self._make_router()
        result = router.promote_trust("u1", {"reason": "test"})
        assert result["code"] == 200
        assert result["data"]["trust_score"] == 0.55

    def test_降低信任端点(self) -> None:
        router = self._make_router()
        result = router.demote_trust("u1", {"reason": "test"})
        assert result["code"] == 200
        assert result["data"]["trust_score"] == 0.4

    def test_抗疲劳统计端点(self) -> None:
        router = self._make_router()
        result = router.get_fatigue_statistics()
        assert result["code"] == 200

    # --- KPI 指标 ---

    def test_KPI仪表盘端点(self) -> None:
        router = self._make_router()
        result = router.get_kpi_dashboard()
        assert result["code"] == 200
        assert "overall_health_score" in result["data"]

    def test_KPI汇总端点(self) -> None:
        router = self._make_router()
        result = router.get_kpi_summary("fatigue_index")
        assert result["code"] == 200

    def test_KPI汇总_不存在(self) -> None:
        router = self._make_router()
        result = router.get_kpi_summary("nonexistent")
        assert result["code"] == 404

    def test_KPI告警端点(self) -> None:
        router = self._make_router()
        result = router.get_kpi_alerts()
        assert result["code"] == 200

    def test_KPI阈值调整端点(self) -> None:
        router = self._make_router()
        result = router.adjust_kpi_threshold(
            "fatigue_index", {"green_max": 20.0, "yellow_max": 40.0}
        )
        assert result["code"] == 200
        assert result["data"]["green_max"] == 20.0

    # --- Agent 管理 ---

    def test_注册Agent端点(self) -> None:
        router = self._make_router()
        result = router.register_agent({"agent_id": "agent-1"})
        assert result["code"] == 200

    def test_注册Agent_缺少ID(self) -> None:
        router = self._make_router()
        result = router.register_agent({})
        assert result["code"] == 400

    def test_获取Agent端点(self) -> None:
        router = self._make_router()
        router.register_agent({"agent_id": "agent-1"})
        result = router.get_agent("agent-1")
        assert result["code"] == 200

    def test_获取Agent_不存在(self) -> None:
        router = self._make_router()
        result = router.get_agent("nonexistent")
        assert result["code"] == 404

    def test_列出Agent端点(self) -> None:
        router = self._make_router()
        router.register_agent({"agent_id": "a1"})
        router.register_agent({"agent_id": "a2"})
        result = router.list_agents()
        assert result["code"] == 200
        assert len(result["data"]) == 2

    # --- CC1 联动 ---

    def test_CC1结果处理端点_pass(self) -> None:
        router = self._make_full_router()
        result = router.process_cc1_result({
            "verdict": "pass",
            "confidence": 0.9,
            "operation_type": "test_op",
        })
        assert result["code"] == 200
        assert result["data"]["cc1_verdict"] == "pass"

    def test_CC1结果处理端点_block(self) -> None:
        router = self._make_full_router()
        result = router.process_cc1_result({
            "verdict": "block",
            "confidence": 0.3,
            "operation_type": "test_op",
        })
        assert result["code"] == 200
        assert result["data"]["approval_created"] is True

    def test_CC1结果处理_缺少字段(self) -> None:
        router = self._make_router()
        result = router.process_cc1_result({"verdict": "pass"})
        assert result["code"] == 400

    # --- 路由定义 ---

    def test_获取路由定义(self) -> None:
        router = self._make_router()
        routes = router.get_routes()
        assert len(routes) > 0
        # 验证路由结构
        for route in routes:
            assert "path" in route
            assert "methods" in route
            assert "handler" in route
