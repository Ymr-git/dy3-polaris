"""CC2 计划审批门 — 六维决策路由引擎.

基于设计文档 CC2-Plan-Approval-Gate 第六章的决策路由引擎设计,
实现六维输入 → 四层协同的动态路由.

六维输入:
1. 操作风险等级 (RiskLevel): 低/中/高/极高
2. 系统置信度 (confidence): 0.00-1.00 — 来自 CC1 四层评审
3. 用户信任度 (trust_score): 0.00-1.00 — 来自 L2 个性化层
4. 操作可逆性 (Reversibility): 可逆/有限可逆/不可逆
5. 用户角色 (UserRole): 学生/教师/管理员
6. 当前认知负荷 (cognitive_load): 0.00-1.00 — 来自 L2 BKT

四层协同 (CollaborationLayer):
- L1_IMPLICIT: 隐性层 — 系统主导, 无显式交互 (自主性 95-100%)
- L2_PROMPT: 提示层 — 系统建议, 轻量提示 (自主性 70-95%)
- L3_APPROVAL: 审批层 — 系统申请, 阻塞式确认 (自主性 30-70%)
- L4_INTERVENTION: 干预层 — 人为主导, 主动介入 (自主性 0-30%)

路由策略: 加权评分 + 规则覆盖的混合策略
- 基础分由六维加权计算
- 场景化规则可覆盖基础路由结果
- 支持动态降级 (审批疲劳) 和动态升级 (异常信号)

融合世界先进方案:
- NIST AI RMF: Govern-Map-Measure-Manage 四功能核心
- REACT Framework: 五维评分驱动自治级别
- Enterprise Approval Workflows: 四种审批门放置模式
- GAIA: 信息门控 + 承诺检测 + 安全不变量
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ============================================================
# 枚举定义
# ============================================================


class CollaborationLayer(str, enum.Enum):
    """四层人机协同层级 (设计文档定义).

    L1 → L4 自主性递减, 人类控制递增.

    递进关系: Implicit → Prompt → Approval → Intervention
    当系统置信度降低、操作风险升高、用户信任度不足或操作不可逆时,
    协同自动向更高层级升级.
    """

    L1_IMPLICIT = "l1_implicit"
    L2_PROMPT = "l2_prompt"
    L3_APPROVAL = "l3_approval"
    L4_INTERVENTION = "l4_intervention"


class RiskLevel(str, enum.Enum):
    """操作风险等级 (L0 Policy Engine 定义)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Reversibility(str, enum.Enum):
    """操作可逆性."""

    REVERSIBLE = "reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"


class UserRole(str, enum.Enum):
    """用户角色 (L1 用户域 RBAC)."""

    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    SYSTEM = "system"


class ApprovalMode(str, enum.Enum):
    """审批模式 (L3 审批层四种模式).

    - quick_confirm: 快速确认 — 一键 Approve, 低风险常规操作
    - detailed_review: 详细审批 — 完整信息+替代方案+风险分析
    - negotiated_approval: 协商审批 — 用户可修改参数后批准
    - rule_preset: 规则预设 — 批量设置审批规则
    """

    QUICK_CONFIRM = "quick_confirm"
    DETAILED_REVIEW = "detailed_review"
    NEGOTIATED_APPROVAL = "negotiated_approval"
    RULE_PRESET = "rule_preset"


class TimeoutAction(str, enum.Enum):
    """超时处理策略."""

    ABORT = "abort"
    AUTO_APPROVE = "auto_approve"
    DOWNGRADE_TO_PROMPT = "downgrade_to_prompt"
    ESCALATE = "escalate"


class InterventionTypeL4(str, enum.Enum):
    """L4 干预层干预类型."""

    EMERGENCY_PAUSE = "emergency_pause"
    MANUAL_OVERRIDE = "manual_override"
    CORRECTION_FEEDBACK = "correction_feedback"
    CREATIVE_REQUEST = "creative_request"


class Priority(str, enum.Enum):
    """干预优先级."""

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"


class RecoveryMode(str, enum.Enum):
    """恢复模式."""

    RESUME_FROM_CHECKPOINT = "resume_from_checkpoint"
    RESTART_FROM_NEW_STATE = "restart_from_new_state"


# ============================================================
# 数据模型
# ============================================================


class RoutingContext(BaseModel):
    """六维路由上下文.

    封装决策路由引擎的全部输入维度.

    Attributes:
        operation_type: 操作类型标识 (如 "learning_path_reset")
        target: 操作目标 (如知识点 ID)
        risk_level: 操作风险等级
        confidence: 系统置信度 (0-1), 来自 CC1
        trust_score: 用户信任度 (0-1), 来自 L2
        reversibility: 操作可逆性
        user_role: 用户角色
        cognitive_load: 当前认知负荷 (0-1)
        user_id: 用户 ID
        session_id: 会话 ID
        metadata: 附加元数据
    """

    operation_type: str = Field(default="", description="操作类型")
    target: str = Field(default="", description="操作目标")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    confidence: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="系统置信度 (CC1 综合评分)",
    )
    trust_score: float = Field(
        default=0.90, ge=0.0, le=1.0,
        description="用户信任度 (L2 维护)",
    )
    reversibility: Reversibility = Field(
        default=Reversibility.REVERSIBLE,
    )
    user_role: UserRole = Field(default=UserRole.STUDENT)
    cognitive_load: float = Field(
        default=0.45, ge=0.0, le=1.0,
        description="当前认知负荷",
    )
    user_id: str = Field(default="")
    session_id: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingResult(BaseModel):
    """路由决策结果.

    Attributes:
        result_id: 结果 ID
        recommended_layer: 推荐协同层级
        approval_mode: 审批模式 (仅 L3 时有效)
        timeout_seconds: 超时时间 (秒)
        timeout_action: 超时策略
        reasoning: 路由理由
        policy_reference: 策略引用
        score: 路由评分 (0-100, 越高越需人类介入)
        rule_id: 匹配的规则 ID (如有)
        alternatives: 备选方案
        created_at: 创建时间
    """

    result_id: str = Field(
        default_factory=lambda: f"rt-{uuid.uuid4().hex[:10]}"
    )
    recommended_layer: CollaborationLayer = Field(
        default=CollaborationLayer.L1_IMPLICIT,
    )
    approval_mode: ApprovalMode | None = Field(default=None)
    timeout_seconds: float = Field(default=300.0)
    timeout_action: TimeoutAction = Field(default=TimeoutAction.ABORT)
    reasoning: str = Field(default="")
    policy_reference: str = Field(default="")
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    rule_id: str = Field(default="")
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


# ============================================================
# 场景化路由规则
# ============================================================


@dataclass
class RoutingRule:
    """场景化路由规则.

    当上下文匹配特定条件时, 覆盖基础评分路由结果.

    Attributes:
        rule_id: 规则 ID
        name: 规则名称
        description: 规则描述
        matcher: 匹配函数 (RoutingContext -> bool)
        layer: 匹配时强制路由到的层级
        approval_mode: 匹配时的审批模式
        timeout_seconds: 超时时间
        timeout_action: 超时策略
        priority: 规则优先级 (越高越先匹配)
    """

    rule_id: str
    name: str
    description: str
    matcher: Any  # callable: (RoutingContext) -> bool
    layer: CollaborationLayer
    approval_mode: ApprovalMode | None = None
    timeout_seconds: float = 300.0
    timeout_action: TimeoutAction = TimeoutAction.ABORT
    priority: int = 0


# ============================================================
# 默认路由规则表 (设计文档第 6.3 节)
# ============================================================


def _rule_learning_path_reset(ctx: RoutingContext) -> bool:
    """学习路径重置 — 不可逆高风险."""
    return (
        ctx.operation_type == "learning_path_reset"
        and ctx.reversibility == Reversibility.IRREVERSIBLE
    )


def _rule_consecutive_errors(ctx: RoutingContext) -> bool:
    """连续 10 题错误 — 自动干预."""
    return (
        ctx.metadata.get("consecutive_errors", 0) >= 10
    )


def _rule_external_api(ctx: RoutingContext) -> bool:
    """外部 API 调用 — 高风险审批."""
    return (
        ctx.operation_type in ("external_api_call", "paid_api_call")
        and ctx.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    )


def _rule_publish_content(ctx: RoutingContext) -> bool:
    """发布新教学内容 — 极高协商审批."""
    return (
        ctx.operation_type == "publish_content"
        and ctx.user_role == UserRole.TEACHER
    )


def _rule_prompt_template(ctx: RoutingContext) -> bool:
    """修改导学 Agent Prompt — 极高协商审批."""
    return ctx.operation_type == "prompt_template_modify"


def _rule_cc1_block(ctx: RoutingContext) -> bool:
    """CC1 Block 人工仲裁."""
    return ctx.metadata.get("cc1_verdict") == "block"


def _rule_emergency_pause(ctx: RoutingContext) -> bool:
    """学生点击紧急暂停 — 强制 L4."""
    return ctx.operation_type == "emergency_pause"


def _rule_safety_content(ctx: RoutingContext) -> bool:
    """安全相关内容 — 强制 L4."""
    return ctx.metadata.get("safety_related", False) is True


def _rule_cognitive_overload(ctx: RoutingContext) -> bool:
    """认知负荷过高 — 自动干预."""
    return ctx.cognitive_load >= 0.95


def _rule_low_confidence_content(ctx: RoutingContext) -> bool:
    """置信度 0.65 内容呈现 — L2 提示+警告."""
    return (
        0.60 <= ctx.confidence < 0.70
        and ctx.risk_level == RiskLevel.MEDIUM
    )


def _rule_data_overwrite(ctx: RoutingContext) -> bool:
    """学情数据覆写 — L3 详细审批."""
    return (
        ctx.operation_type == "data_overwrite"
        and ctx.reversibility == Reversibility.IRREVERSIBLE
    )


def _rule_trust_mode(ctx: RoutingContext) -> bool:
    """信任模式窗口内 — 低风险自动通过."""
    return (
        ctx.metadata.get("trust_mode_active", False) is True
        and ctx.risk_level == RiskLevel.LOW
        and ctx.reversibility == Reversibility.REVERSIBLE
    )


#: 默认路由规则表 (按优先级排序)
DEFAULT_ROUTING_RULES: list[RoutingRule] = [
    RoutingRule(
        rule_id="RR-001",
        name="紧急暂停",
        description="学生点击紧急暂停 → 强制 L4 干预",
        matcher=_rule_emergency_pause,
        layer=CollaborationLayer.L4_INTERVENTION,
        timeout_seconds=5.0,
        timeout_action=TimeoutAction.ABORT,
        priority=100,
    ),
    RoutingRule(
        rule_id="RR-002",
        name="安全相关内容",
        description="化学试剂等安全内容 → 强制 L4 干预",
        matcher=_rule_safety_content,
        layer=CollaborationLayer.L4_INTERVENTION,
        timeout_seconds=5.0,
        timeout_action=TimeoutAction.ABORT,
        priority=99,
    ),
    RoutingRule(
        rule_id="RR-003",
        name="连续10题错误",
        description="连续错误≥10 → 自动紧急暂停+教师通知",
        matcher=_rule_consecutive_errors,
        layer=CollaborationLayer.L4_INTERVENTION,
        timeout_seconds=5.0,
        timeout_action=TimeoutAction.ABORT,
        priority=98,
    ),
    RoutingRule(
        rule_id="RR-004",
        name="认知负荷过载",
        description="认知负荷≥0.95 → 自动暂停建议休息",
        matcher=_rule_cognitive_overload,
        layer=CollaborationLayer.L4_INTERVENTION,
        timeout_seconds=5.0,
        timeout_action=TimeoutAction.ABORT,
        priority=97,
    ),
    RoutingRule(
        rule_id="RR-005",
        name="CC1 Block 仲裁",
        description="CC1 四层评审 Block → 人工仲裁",
        matcher=_rule_cc1_block,
        layer=CollaborationLayer.L3_APPROVAL,
        approval_mode=ApprovalMode.DETAILED_REVIEW,
        timeout_seconds=300.0,
        timeout_action=TimeoutAction.ABORT,
        priority=90,
    ),
    RoutingRule(
        rule_id="RR-006",
        name="Prompt 模板修改",
        description="修改导学 Agent 核心 Prompt → 协商审批",
        matcher=_rule_prompt_template,
        layer=CollaborationLayer.L3_APPROVAL,
        approval_mode=ApprovalMode.NEGOTIATED_APPROVAL,
        timeout_seconds=172800.0,  # 48 小时
        timeout_action=TimeoutAction.ABORT,
        priority=88,
    ),
    RoutingRule(
        rule_id="RR-007",
        name="发布教学内容",
        description="教师发布新内容 → 协商审批+CC1 校验",
        matcher=_rule_publish_content,
        layer=CollaborationLayer.L3_APPROVAL,
        approval_mode=ApprovalMode.NEGOTIATED_APPROVAL,
        timeout_seconds=259200.0,  # 72 小时
        timeout_action=TimeoutAction.ABORT,
        priority=85,
    ),
    RoutingRule(
        rule_id="RR-008",
        name="学习路径重置",
        description="不可逆高风险路径重置 → 详细审批",
        matcher=_rule_learning_path_reset,
        layer=CollaborationLayer.L3_APPROVAL,
        approval_mode=ApprovalMode.DETAILED_REVIEW,
        timeout_seconds=300.0,
        timeout_action=TimeoutAction.ABORT,
        priority=80,
    ),
    RoutingRule(
        rule_id="RR-009",
        name="学情数据覆写",
        description="不可逆数据覆写 → 详细审批",
        matcher=_rule_data_overwrite,
        layer=CollaborationLayer.L3_APPROVAL,
        approval_mode=ApprovalMode.DETAILED_REVIEW,
        timeout_seconds=60.0,
        timeout_action=TimeoutAction.ABORT,
        priority=78,
    ),
    RoutingRule(
        rule_id="RR-010",
        name="外部 API 调用",
        description="高风险外部 API → 详细审批",
        matcher=_rule_external_api,
        layer=CollaborationLayer.L3_APPROVAL,
        approval_mode=ApprovalMode.DETAILED_REVIEW,
        timeout_seconds=120.0,
        timeout_action=TimeoutAction.ABORT,
        priority=75,
    ),
    RoutingRule(
        rule_id="RR-011",
        name="低置信度内容",
        description="置信度 0.60-0.70 → L2 提示+警告标签",
        matcher=_rule_low_confidence_content,
        layer=CollaborationLayer.L2_PROMPT,
        timeout_seconds=0.0,
        timeout_action=TimeoutAction.ABORT,
        priority=50,
    ),
    RoutingRule(
        rule_id="RR-012",
        name="信任模式窗口",
        description="信任模式+低风险可逆 → L1 隐性",
        matcher=_rule_trust_mode,
        layer=CollaborationLayer.L1_IMPLICIT,
        timeout_seconds=0.0,
        timeout_action=TimeoutAction.AUTO_APPROVE,
        priority=40,
    ),
]


# ============================================================
# 决策路由引擎
# ============================================================


class RoutingEngine:
    """六维决策路由引擎.

    基于六维输入和场景化规则, 动态路由到四层协同层级.

    路由策略: 加权评分 + 规则覆盖的混合策略
    1. 规则匹配: 按优先级依次检查场景化规则, 首个匹配的规则覆盖路由结果
    2. 基础评分: 无规则匹配时, 用六维加权评分计算路由层级

    融合方案:
    - NIST AI RMF: Govern(角色矩阵) → Map(能力边界) → Measure(评分) → Manage(路由)
    - REACT Framework: 多维评分驱动自治级别
    - Enterprise Approval Workflows: 四种审批门放置模式
    - GAIA: 承诺检测(不可逆操作)→强制升级

    使用示例::

        engine = RoutingEngine()
        ctx = RoutingContext(
            operation_type="learning_path_reset",
            risk_level=RiskLevel.HIGH,
            confidence=0.75,
            reversibility=Reversibility.IRREVERSIBLE,
        )
        result = engine.route(ctx)
        assert result.recommended_layer == CollaborationLayer.L3_APPROVAL
    """

    #: 六维权重 (总和=1.0)
    DEFAULT_WEIGHTS: dict[str, float] = {
        "risk": 0.30,
        "confidence": 0.25,
        "trust": 0.15,
        "reversibility": 0.15,
        "cognitive_load": 0.10,
        "role": 0.05,
    }

    #: 风险等级 → 数值映射
    RISK_SCORES: dict[RiskLevel, float] = {
        RiskLevel.LOW: 10.0,
        RiskLevel.MEDIUM: 35.0,
        RiskLevel.HIGH: 70.0,
        RiskLevel.CRITICAL: 90.0,
    }

    #: 可逆性 → 数值映射
    REVERSIBILITY_SCORES: dict[Reversibility, float] = {
        Reversibility.REVERSIBLE: 10.0,
        Reversibility.PARTIALLY_REVERSIBLE: 50.0,
        Reversibility.IRREVERSIBLE: 90.0,
    }

    #: 角色 → 附加分 (管理员操作更需审批)
    ROLE_SCORES: dict[UserRole, float] = {
        UserRole.STUDENT: 30.0,
        UserRole.TEACHER: 50.0,
        UserRole.ADMIN: 70.0,
        UserRole.SYSTEM: 20.0,
    }

    def __init__(
        self,
        rules: list[RoutingRule] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._rules = sorted(
            rules or DEFAULT_ROUTING_RULES,
            key=lambda r: r.priority,
            reverse=True,
        )
        self._weights = weights or dict(self.DEFAULT_WEIGHTS)
        self._routing_history: list[RoutingResult] = []

    @property
    def rules(self) -> list[RoutingRule]:
        return list(self._rules)

    @property
    def routing_history(self) -> list[RoutingResult]:
        return list(self._routing_history)

    def add_rule(self, rule: RoutingRule) -> None:
        """添加自定义路由规则."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def route(self, ctx: RoutingContext) -> RoutingResult:
        """执行路由决策.

        Args:
            ctx: 六维路由上下文

        Returns:
            路由决策结果
        """
        # 1. 规则匹配 (按优先级)
        for rule in self._rules:
            try:
                if rule.matcher(ctx):
                    result = RoutingResult(
                        recommended_layer=rule.layer,
                        approval_mode=rule.approval_mode,
                        timeout_seconds=rule.timeout_seconds,
                        timeout_action=rule.timeout_action,
                        reasoning=f"规则匹配: {rule.name} ({rule.rule_id})",
                        rule_id=rule.rule_id,
                        score=self._compute_score(ctx),
                        policy_reference=rule.rule_id,
                    )
                    self._routing_history.append(result)
                    return result
            except Exception:
                continue

        # 2. 基础评分路由
        score = self._compute_score(ctx)
        layer = self._score_to_layer(score)
        approval_mode = self._select_approval_mode(ctx, layer)
        timeout = self._select_timeout(ctx, layer)
        timeout_action = self._select_timeout_action(ctx, layer)

        result = RoutingResult(
            recommended_layer=layer,
            approval_mode=approval_mode,
            timeout_seconds=timeout,
            timeout_action=timeout_action,
            reasoning=self._generate_reasoning(ctx, score, layer),
            score=score,
        )
        self._routing_history.append(result)
        return result

    def _compute_score(self, ctx: RoutingContext) -> float:
        """计算六维加权评分 (0-100, 越高越需人类介入)."""
        w = self._weights

        risk_score = self.RISK_SCORES.get(ctx.risk_level, 35.0)
        # 置信度越低 → 分数越高 (需更多人类介入)
        confidence_score = (1.0 - ctx.confidence) * 100.0
        # 信任度越低 → 分数越高
        trust_score = (1.0 - ctx.trust_score) * 100.0
        reversibility_score = self.REVERSIBILITY_SCORES.get(
            ctx.reversibility, 50.0
        )
        cognitive_score = ctx.cognitive_load * 100.0
        role_score = self.ROLE_SCORES.get(ctx.user_role, 30.0)

        score = (
            risk_score * w["risk"]
            + confidence_score * w["confidence"]
            + trust_score * w["trust"]
            + reversibility_score * w["reversibility"]
            + cognitive_score * w["cognitive_load"]
            + role_score * w["role"]
        )
        return round(min(max(score, 0.0), 100.0), 2)

    def _score_to_layer(self, score: float) -> CollaborationLayer:
        """评分 → 协同层级映射.

        0-25: L1 隐性层 (系统主导)
        25-50: L2 提示层 (系统建议)
        50-75: L3 审批层 (系统申请)
        75-100: L4 干预层 (人为主导)
        """
        if score < 25.0:
            return CollaborationLayer.L1_IMPLICIT
        elif score < 50.0:
            return CollaborationLayer.L2_PROMPT
        elif score < 75.0:
            return CollaborationLayer.L3_APPROVAL
        else:
            return CollaborationLayer.L4_INTERVENTION

    def _select_approval_mode(
        self, ctx: RoutingContext, layer: CollaborationLayer
    ) -> ApprovalMode | None:
        """选择审批模式 (仅 L3 时有效)."""
        if layer != CollaborationLayer.L3_APPROVAL:
            return None

        if ctx.risk_level == RiskLevel.LOW:
            return ApprovalMode.QUICK_CONFIRM
        elif ctx.risk_level == RiskLevel.CRITICAL:
            return ApprovalMode.NEGOTIATED_APPROVAL
        elif ctx.reversibility == Reversibility.PARTIALLY_REVERSIBLE:
            return ApprovalMode.NEGOTIATED_APPROVAL
        else:
            return ApprovalMode.DETAILED_REVIEW

    def _select_timeout(
        self, ctx: RoutingContext, layer: CollaborationLayer
    ) -> float:
        """选择超时时间."""
        if layer == CollaborationLayer.L1_IMPLICIT:
            return 0.0
        elif layer == CollaborationLayer.L2_PROMPT:
            return 0.0
        elif layer == CollaborationLayer.L3_APPROVAL:
            if ctx.risk_level == RiskLevel.LOW:
                return 30.0
            elif ctx.risk_level == RiskLevel.MEDIUM:
                return 120.0
            elif ctx.risk_level == RiskLevel.HIGH:
                return 300.0
            else:
                return 600.0
        else:  # L4_INTERVENTION
            return 5.0

    def _select_timeout_action(
        self, ctx: RoutingContext, layer: CollaborationLayer
    ) -> TimeoutAction:
        """选择超时策略."""
        if layer == CollaborationLayer.L1_IMPLICIT:
            return TimeoutAction.AUTO_APPROVE
        elif layer == CollaborationLayer.L2_PROMPT:
            return TimeoutAction.DOWNGRADE_TO_PROMPT
        elif layer == CollaborationLayer.L3_APPROVAL:
            if ctx.risk_level == RiskLevel.LOW:
                return TimeoutAction.AUTO_APPROVE
            else:
                return TimeoutAction.ABORT
        else:
            return TimeoutAction.ABORT

    def _generate_reasoning(
        self, ctx: RoutingContext, score: float, layer: CollaborationLayer
    ) -> str:
        """生成路由理由."""
        layer_names = {
            CollaborationLayer.L1_IMPLICIT: "L1 隐性层",
            CollaborationLayer.L2_PROMPT: "L2 提示层",
            CollaborationLayer.L3_APPROVAL: "L3 审批层",
            CollaborationLayer.L4_INTERVENTION: "L4 干预层",
        }
        return (
            f"六维评分 {score:.1f} → {layer_names[layer]}, "
            f"风险={ctx.risk_level.value}, 置信度={ctx.confidence:.2f}, "
            f"信任度={ctx.trust_score:.2f}, "
            f"可逆性={ctx.reversibility.value}"
        )

    def get_statistics(self) -> dict[str, Any]:
        """获取路由统计."""
        total = len(self._routing_history)
        if total == 0:
            return {"total": 0}

        by_layer: dict[str, int] = {}
        by_rule: dict[str, int] = {}
        for r in self._routing_history:
            layer_key = r.recommended_layer.value
            by_layer[layer_key] = by_layer.get(layer_key, 0) + 1
            if r.rule_id:
                by_rule[r.rule_id] = by_rule.get(r.rule_id, 0) + 1

        avg_score = sum(r.score for r in self._routing_history) / total

        return {
            "total": total,
            "by_layer": by_layer,
            "by_rule": by_rule,
            "avg_score": round(avg_score, 2),
        }

    def clear_history(self) -> None:
        """清空路由历史."""
        self._routing_history.clear()
