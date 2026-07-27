"""L0 治理基础模型.

定义治理层的核心数据模型，涵盖四治理域：
- 策略域：声明式策略、匹配规则、作用域
- 审计域：违规记录、合规报告、合规模板
- 声誉域：Agent 声誉快照（预留）
- 溯源域：治理事件类型（与 L6 KPA 体系对接）

所有模型基于 pydantic v2，枚举采用 (str, Enum) 风格与 L6 保持一致。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# 枚举定义
# ============================================================


class PolicyAction(str, Enum):
    """策略动作.

    当策略匹配时执行的动作类型。
    策略引擎按以下优先级处理：deny > transform > escalate > log > allow。
    """

    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"
    ESCALATE = "escalate"
    TRANSFORM = "transform"


class PolicyScope(str, Enum):
    """策略作用域.

    定义策略的适用范围。global 作用于所有请求，
    其余按维度缩小匹配范围。
    """

    GLOBAL = "global"
    AGENT = "agent"
    TOOL = "tool"
    LAYER = "layer"
    DOMAIN = "domain"


class MatchOperator(str, Enum):
    """匹配操作符.

    定义规则匹配的方式：
    - exact: 精确匹配
    - glob:  shell 风格通配符 (* 匹配任意字符, ** 匹配路径分隔)
    - regex: 正则表达式匹配
    """

    EXACT = "exact"
    GLOB = "glob"
    REGEX = "regex"


class SeverityLevel(str, Enum):
    """违规严重级别.

    用于违规记录和告警分级。
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GovernanceEventType(str, Enum):
    """治理事件类型.

    与 L6 KPA 事件体系对接，扩展治理专属事件。
    """

    POLICY_CREATED = "policy_created"
    POLICY_UPDATED = "policy_updated"
    POLICY_DELETED = "policy_deleted"
    POLICY_EVALUATED = "policy_evaluated"
    VIOLATION_DETECTED = "violation_detected"
    VIOLATION_RESOLVED = "violation_resolved"
    COMPLIANCE_CHECK = "compliance_check"
    REPUTATION_UPDATED = "reputation_updated"
    AUDIT_REPORT_GENERATED = "audit_report_generated"
    ESCALATION_TRIGGERED = "escalation_triggered"
    ESCALATION_RESOLVED = "escalation_resolved"


class ViolationStatus(str, Enum):
    """违规记录状态.

    跟踪违规的生命周期：检测 → 确认 → 已解决 / 已忽略。
    """

    DETECTED = "detected"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class ComplianceTemplate(str, Enum):
    """合规模板类型.

    六类标准合规模板包，对应竞赛评审要求的合规维度。
    """

    ACADEMIC_INTEGRITY = "academic_integrity"
    DATA_PRIVACY = "data_privacy"
    CONTENT_SAFETY = "content_safety"
    PLATFORM_OPS = "platform_ops"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    COPYRIGHT_PROTECTION = "copyright_protection"


class GovernanceDomain(str, Enum):
    """治理域.

    四治理域的枚举标识，用于策略分类和审计报告分域。
    """

    PROVENANCE = "provenance"
    POLICY = "policy"
    REPUTATION = "reputation"
    AUDIT = "audit"


# ============================================================
# 匹配规则
# ============================================================


class PolicyMatchRule(BaseModel):
    """策略匹配规则.

    单条匹配规则，由字段名、操作符和期望值组成。
    支持精确匹配、通配符和正则三种匹配方式。

    示例::

        PolicyMatchRule(field="agent_id", operator="glob", value="tutor-*")
        PolicyMatchRule(field="layer", operator="exact", value="L4")
        PolicyMatchRule(field="tool_name", operator="regex", value="bkt_.+")
    """

    field: str = Field(description="匹配字段名，如 agent_id, tool_name, layer, domain, action")
    operator: MatchOperator = Field(default=MatchOperator.EXACT, description="匹配操作符")
    value: str = Field(description="期望值")
    negate: bool = Field(default=False, description="是否取反，true 表示不匹配时命中")


class PolicyCondition(BaseModel):
    """策略条件组合.

    多条规则通过 AND/OR 逻辑组合。
    空条件列表视为全匹配（始终命中）。
    """

    logic: Literal["and", "or"] = Field(default="and", description="逻辑组合方式")
    rules: list[PolicyMatchRule] = Field(default_factory=list, description="匹配规则列表")


class TransformSpec(BaseModel):
    """动作转换规范.

    当策略动作为 transform 时，定义如何修改请求/响应。
    - mask_fields: 将指定字段值替换为掩码
    - strip_fields: 从请求中移除指定字段
    - inject_fields: 注入额外字段
    - max_value: 限制字段最大值
    """

    mask_fields: list[str] = Field(default_factory=list, description="要掩码的字段列表")
    strip_fields: list[str] = Field(default_factory=list, description="要移除的字段列表")
    inject_fields: dict[str, Any] = Field(default_factory=dict, description="要注入的字段")
    max_value: dict[str, float] = Field(default_factory=dict, description="字段最大值限制")


class EscalationSpec(BaseModel):
    """升级处理规范.

    当策略动作为 escalate 时，定义升级目标和约束。
    """

    target: str = Field(description="升级目标，如 agent_id 或 human_review")
    timeout_seconds: float = Field(default=300.0, ge=1.0, description="升级超时时间（秒）")
    auto_resolve: bool = Field(default=False, description="超时后是否自动解决（默认拒绝）")
    reason_template: str = Field(default="策略触发升级: {policy_id}", description="升级原因模板")


# ============================================================
# 核心策略模型
# ============================================================


class GovernancePolicy(BaseModel):
    """治理策略.

    治理层的核心声明式策略单元。策略通过条件匹配请求上下文，
    匹配成功时执行指定动作。

    Attributes:
        policy_id: 策略唯一标识
        name: 策略名称
        description: 策略描述
        domain: 所属治理域
        scope: 策略作用域
        priority: 优先级，数值越大优先级越高
        enabled: 是否启用
        action: 匹配时执行的动作
        condition: 匹配条件
        transform: 转换规范（仅 action=transform 时使用）
        escalation: 升级规范（仅 action=escalate 时使用）
        compliance_templates: 关联的合规模板
        tags: 策略标签
        created_at: 创建时间
        updated_at: 最后更新时间
        created_by: 创建者
    """

    policy_id: str = Field(default_factory=lambda: f"pol-{uuid.uuid4().hex[:12]}", description="策略唯一 ID")
    name: str = Field(description="策略名称")
    description: str = Field(default="", description="策略描述")
    domain: GovernanceDomain = Field(default=GovernanceDomain.POLICY, description="所属治理域")
    scope: PolicyScope = Field(default=PolicyScope.GLOBAL, description="策略作用域")
    priority: int = Field(default=0, ge=-1000, le=1000, description="优先级，数值越大越高")
    enabled: bool = Field(default=True, description="是否启用")
    action: PolicyAction = Field(default=PolicyAction.ALLOW, description="匹配时执行的动作")
    condition: PolicyCondition = Field(default_factory=PolicyCondition, description="匹配条件")
    transform: TransformSpec | None = Field(default=None, description="转换规范")
    escalation: EscalationSpec | None = Field(default=None, description="升级规范")
    compliance_templates: list[ComplianceTemplate] = Field(default_factory=list, description="关联合规模板")
    tags: list[str] = Field(default_factory=list, description="策略标签")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")
    created_by: str = Field(default="system", description="创建者")

    @model_validator(mode="after")
    def _validate_action_specs(self) -> GovernancePolicy:
        """校验动作与附加规范的合法性."""
        if self.action == PolicyAction.TRANSFORM and self.transform is None:
            self.transform = TransformSpec()
        if self.action == PolicyAction.ESCALATE and self.escalation is None:
            self.escalation = EscalationSpec(target="human_review")
        if self.action not in (PolicyAction.TRANSFORM, PolicyAction.ESCALATE):
            # 非动作类型不应有附加规范（静默清理而非报错）
            pass
        return self

    def touch(self) -> None:
        """更新 updated_at 时间戳."""
        self.updated_at = time.time()


class EvalRequest(BaseModel):
    """策略评估请求.

    由 G2 策略引擎接收的评估输入，
    描述一个待检查的操作上下文。

    Attributes:
        actor: 执行者标识（agent_id 或 user_id）
        action: 请求执行的动作名（如 tool_call, message_send, decision_route）
        resource: 操作目标资源标识（如 tool_name, knowledge_id）
        layer: 所属架构层
        domain: 所属领域
        context: 额外上下文字典（供策略匹配使用）
    """

    actor: str = Field(default="", description="执行者标识")
    action: str = Field(default="", description="请求执行的动作名")
    resource: str = Field(default="", description="操作目标资源标识")
    layer: str = Field(default="", description="所属架构层")
    domain: str = Field(default="", description="所属领域")
    context: dict[str, Any] = Field(default_factory=dict, description="额外上下文")


class EvalResult(BaseModel):
    """策略评估结果.

    策略引擎对单次评估请求的决策输出。

    Attributes:
        decision: 最终动作决策
        matched_policy_id: 命中的策略 ID（若有）
        matched_policy_name: 命中的策略名称（若有）
        reason: 决策原因
        transform: 转换规范（decision=transform 时填充）
        escalation: 升级规范（decision=escalate 时填充）
        evaluated_at: 评估时间戳
    """

    decision: PolicyAction = Field(description="最终动作决策")
    matched_policy_id: str | None = Field(default=None, description="命中的策略 ID")
    matched_policy_name: str | None = Field(default=None, description="命中的策略名称")
    reason: str = Field(default="", description="决策原因")
    transform: TransformSpec | None = Field(default=None, description="转换规范")
    escalation: EscalationSpec | None = Field(default=None, description="升级规范")
    evaluated_at: float = Field(default_factory=time.time, description="评估时间")


# ============================================================
# 违规记录
# ============================================================


class ViolationRecord(BaseModel):
    """违规记录.

    当策略评估结果为 deny 或 escalate 时产生的违规事件记录。
    不可变：创建后仅允许更新 status 字段。

    Attributes:
        violation_id: 违规记录唯一 ID
        policy_id: 触发违规的策略 ID
        policy_name: 触发违规的策略名称
        severity: 严重级别
        status: 违规状态
        actor: 违规执行者
        action: 违规动作
        resource: 违规目标资源
        layer: 所属层
        detail: 违规详情
        eval_request: 原始评估请求快照
        eval_result: 原始评估结果快照
        resolved_at: 解决时间
        resolved_by: 解决者
        resolution_note: 解决备注
        created_at: 记录创建时间
    """

    violation_id: str = Field(default_factory=lambda: f"vio-{uuid.uuid4().hex[:12]}", description="违规记录 ID")
    policy_id: str = Field(description="触发策略 ID")
    policy_name: str = Field(default="", description="触发策略名称")
    severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM, description="严重级别")
    status: ViolationStatus = Field(default=ViolationStatus.DETECTED, description="违规状态")
    actor: str = Field(default="", description="违规执行者")
    action: str = Field(default="", description="违规动作")
    resource: str = Field(default="", description="违规目标")
    layer: str = Field(default="", description="所属层")
    detail: str = Field(default="", description="违规详情")
    eval_request: dict[str, Any] = Field(default_factory=dict, description="评估请求快照")
    eval_result: dict[str, Any] = Field(default_factory=dict, description="评估结果快照")
    resolved_at: float | None = Field(default=None, description="解决时间")
    resolved_by: str | None = Field(default=None, description="解决者")
    resolution_note: str | None = Field(default=None, description="解决备注")
    created_at: float = Field(default_factory=time.time, description="创建时间")

    def resolve(self, by: str, note: str = "") -> None:
        """标记违规为已解决."""
        self.status = ViolationStatus.RESOLVED
        self.resolved_at = time.time()
        self.resolved_by = by
        self.resolution_note = note

    def confirm(self) -> None:
        """确认违规（从 detected → confirmed）."""
        if self.status == ViolationStatus.DETECTED:
            self.status = ViolationStatus.CONFIRMED

    def ignore(self, note: str = "") -> None:
        """忽略违规."""
        self.status = ViolationStatus.IGNORED
        self.resolved_at = time.time()
        self.resolution_note = note


# ============================================================
# 合规报告
# ============================================================


class DimensionScore(BaseModel):
    """单维度合规评分.

    Attributes:
        dimension: 维度名称
        score: 0-100 评分
        weight: 权重（0.0-1.0）
        details: 评分说明
    """

    dimension: str = Field(description="维度名称")
    score: float = Field(ge=0.0, le=100.0, description="0-100 评分")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="权重")
    details: str = Field(default="", description="评分说明")


class ComplianceReport(BaseModel):
    """合规报告.

    聚合多维度合规评分的综合报告。

    Attributes:
        report_id: 报告 ID
        period_start: 报告周期起始时间
        period_end: 报告周期结束时间
        overall_score: 综合评分（0-100）
        dimensions: 各维度评分
        violation_summary: 违规统计
        policy_summary: 策略统计
        recommendations: 改进建议
        generated_at: 报告生成时间
    """

    report_id: str = Field(default_factory=lambda: f"rpt-{uuid.uuid4().hex[:12]}", description="报告 ID")
    period_start: float = Field(description="周期起始时间")
    period_end: float = Field(description="周期结束时间")
    overall_score: float = Field(default=100.0, ge=0.0, le=100.0, description="综合评分")
    dimensions: list[DimensionScore] = Field(default_factory=list, description="各维度评分")
    violation_summary: dict[str, int] = Field(
        default_factory=lambda: {
            "total": 0, "resolved": 0, "active": 0,
            "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
        },
        description="违规统计",
    )
    policy_summary: dict[str, int] = Field(
        default_factory=lambda: {"total": 0, "enabled": 0, "disabled": 0, "evaluations": 0},
        description="策略统计",
    )
    recommendations: list[str] = Field(default_factory=list, description="改进建议")
    generated_at: float = Field(default_factory=time.time, description="生成时间")

    def compute_overall_score(self) -> float:
        """根据维度评分加权计算综合分.

        Returns:
            加权综合评分（0-100）
        """
        if not self.dimensions:
            return self.overall_score
        total_weight = sum(d.weight for d in self.dimensions)
        if total_weight == 0:
            return self.overall_score
        weighted_sum = sum(d.score * d.weight for d in self.dimensions)
        score = round(weighted_sum / total_weight, 2)
        self.overall_score = score
        return score


class ReputationSnapshot(BaseModel):
    """Agent 声誉快照（G3 预留）.

    基于 Shapley 值贡献度模型的多维度声誉评分。
    当前为占位模型，G3 声誉域实现时扩展。

    Attributes:
        agent_id: Agent 标识
        overall_trust: 综合信任度（0-1）
        accuracy: 准确性评分
        reliability: 可靠性评分
        compliance_rate: 合规率
        contribution: 贡献度（Shapley 值）
        sample_count: 评估样本数
        updated_at: 最后更新时间
    """

    agent_id: str = Field(description="Agent 标识")
    overall_trust: float = Field(default=0.5, ge=0.0, le=1.0, description="综合信任度")
    accuracy: float = Field(default=0.5, ge=0.0, le=1.0, description="准确性评分")
    reliability: float = Field(default=0.5, ge=0.0, le=1.0, description="可靠性评分")
    compliance_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="合规率")
    contribution: float = Field(default=0.0, ge=0.0, le=1.0, description="贡献度")
    sample_count: int = Field(default=0, ge=0, description="评估样本数")
    updated_at: float = Field(default_factory=time.time, description="最后更新时间")


class GovernanceEvent(BaseModel):
    """治理事件.

    治理层产生的结构化事件，可被 L6 KPA 体系引用。

    Attributes:
        event_id: 事件 ID
        event_type: 事件类型
        actor: 触发者
        domain: 所属治理域
        detail: 事件详情
        references: 关联 ID 列表（策略 ID、违规 ID 等）
        payload: 事件载荷
        timestamp: 时间戳
    """

    event_id: str = Field(default_factory=lambda: f"gevt-{uuid.uuid4().hex[:12]}", description="事件 ID")
    event_type: GovernanceEventType = Field(description="事件类型")
    actor: str = Field(default="system", description="触发者")
    domain: GovernanceDomain = Field(default=GovernanceDomain.POLICY, description="所属治理域")
    detail: str = Field(default="", description="事件详情")
    references: list[str] = Field(default_factory=list, description="关联 ID")
    payload: dict[str, Any] = Field(default_factory=dict, description="事件载荷")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
