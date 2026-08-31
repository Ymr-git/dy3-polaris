"""CC2 计划审批门 — 审批抗疲劳机制.

实现设计文档第五章"抗疲劳设计"的完整机制:
1. 频率控制 — 滑动窗口内审批次数限制
2. 批量审批 — 相似操作聚合一键处理
3. 智能预批 — 历史模式预测自动批准
4. 渐进信任 — 连续批准提升信任度
5. 疲劳检测 — 响应时间+决策模式识别疲劳信号
6. 动态调整 — 疲劳等级驱动自动降级

融合世界先进方案:
- NIST AI RMF Measure: 持续度量人类审批负荷
- Enterprise Approval Matrix: 批量审批降低认知负荷
- Progressive Trust (Anthropic Claude): 渐进信任模型
- Adaptive Automation (Parasuraman): 动态自动化级别调整
- Human Factors Engineering: 疲劳检测+认知负荷管理
- LangGraph State Channels: 状态通道管理疲劳状态
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .routing_engine import RiskLevel, Reversibility
from .approval_workflow import ApprovalStatus


# ============================================================
# 枚举定义
# ============================================================


class FatigueLevel(str, Enum):
    """疲劳等级.

    基于疲劳评分的四等级分类:
    - none: 无疲劳 (0-25) — 正常审批
    - mild: 轻度疲劳 (25-50) — 启用批量审批
    - moderate: 中度疲劳 (50-75) — 启用智能预批
    - severe: 重度疲劳 (75-100) — 强制降级到 L2 提示
    """

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class BatchStatus(str, Enum):
    """批量审批状态."""

    COLLECTING = "collecting"
    READY = "ready"
    APPROVED = "approved"
    REJECTED = "rejected"
    PARTIAL = "partial"
    EXPIRED = "expired"


# ============================================================
# 数据模型
# ============================================================


class FatigueState(BaseModel):
    """用户疲劳状态 (滑动窗口追踪).

    Attributes:
        user_id: 用户 ID
        window_start: 当前窗口起始时间
        approval_count: 窗口内审批次数
        rejection_count: 窗口内拒绝次数
        modification_count: 窗口内修改次数
        timeout_count: 窗口内超时次数
        response_times: 最近响应时间队列
        avg_response_time: 平均响应时间
        fatigue_score: 计算的疲劳评分 (0-100)
        fatigue_level: 疲劳等级
        last_updated: 最后更新时间
    """

    user_id: str = Field(description="用户 ID")
    window_start: float = Field(default_factory=time.time)
    approval_count: int = Field(default=0, ge=0)
    rejection_count: int = Field(default=0, ge=0)
    modification_count: int = Field(default=0, ge=0)
    timeout_count: int = Field(default=0, ge=0)
    response_times: list[float] = Field(default_factory=list)
    avg_response_time: float = Field(default=0.0, ge=0.0)
    fatigue_score: float = Field(default=0.0, ge=0.0, le=100.0)
    fatigue_level: FatigueLevel = Field(default=FatigueLevel.NONE)
    last_updated: float = Field(default_factory=time.time)

    @property
    def total_decisions(self) -> int:
        """窗口内总决策次数."""
        return (
            self.approval_count
            + self.rejection_count
            + self.modification_count
            + self.timeout_count
        )

    @property
    def auto_approve_rate(self) -> float:
        """自动批准率 (越高说明信任度越高)."""
        total = self.total_decisions
        if total == 0:
            return 0.0
        return round(self.approval_count / total, 3)


class ApprovalPattern(BaseModel):
    """操作审批模式 (历史学习).

    追踪特定操作类型的历史审批模式,
    用于智能预批和渐进信任.

    Attributes:
        operation: 操作类型
        total_count: 总请求次数
        approved_count: 批准次数
        rejected_count: 拒绝次数
        modified_count: 修改次数
        auto_approved_count: 自动批准次数
        last_decision: 最后一次决策
        last_decision_time: 最后决策时间
        consecutive_approvals: 连续批准次数 (用于渐进信任)
        avg_risk_level: 平均风险等级数值
    """

    operation: str = Field(description="操作类型")
    total_count: int = Field(default=0, ge=0)
    approved_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    modified_count: int = Field(default=0, ge=0)
    auto_approved_count: int = Field(default=0, ge=0)
    last_decision: str = Field(default="")
    last_decision_time: float = Field(default=0.0)
    consecutive_approvals: int = Field(default=0, ge=0)
    avg_risk_level: float = Field(default=0.0, ge=0.0, le=3.0)

    @property
    def approval_rate(self) -> float:
        """批准率."""
        if self.total_count == 0:
            return 0.0
        return round(self.approved_count / self.total_count, 3)

    @property
    def rejection_rate(self) -> float:
        """拒绝率."""
        if self.total_count == 0:
            return 0.0
        return round(self.rejected_count / self.total_count, 3)

    @property
    def is_high_approval(self) -> bool:
        """是否为高批准率操作 (≥0.9 且至少 5 次)."""
        return self.total_count >= 5 and self.approval_rate >= 0.9


class BatchApprovalGroup(BaseModel):
    """批量审批组.

    将相似操作聚合为一个批量审批组,
    用户可一键批准/拒绝全部.

    Attributes:
        batch_id: 批次 ID
        user_id: 用户 ID
        operation: 操作类型
        items: 待批量审批的请求列表
        status: 批次状态
        created_at: 创建时间
        expires_at: 过期时间
        risk_summary: 风险摘要
    """

    batch_id: str = Field(
        default_factory=lambda: f"batch-{uuid.uuid4().hex[:10]}"
    )
    user_id: str = Field(default="")
    operation: str = Field(default="")
    items: list[dict[str, Any]] = Field(default_factory=list)
    status: BatchStatus = Field(default=BatchStatus.COLLECTING)
    created_at: float = Field(default_factory=time.time)
    expires_at: float = Field(default=0.0)
    risk_summary: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """设置过期时间."""
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + 300.0  # 5 分钟

    @property
    def is_expired(self) -> bool:
        """是否已过期."""
        return time.time() >= self.expires_at

    @property
    def item_count(self) -> int:
        """待审批项数."""
        return len(self.items)


class ProgressiveTrustRecord(BaseModel):
    """渐进信任记录.

    追踪用户的信任度演进, 基于连续批准行为
    逐步提升自动批准权限.

    Attributes:
        user_id: 用户 ID
        trust_score: 当前信任度 (0.0-1.0)
        base_trust: 基础信任度
        consecutive_approvals: 全局连续批准次数
        max_consecutive: 历史最高连续批准次数
        trust_history: 信任度变化历史
        last_violation_time: 最后一次违规时间
        promotions: 信任提升次数
        demotions: 信任降级次数
    """

    user_id: str = Field(description="用户 ID")
    trust_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="当前信任度",
    )
    base_trust: float = Field(default=0.5, ge=0.0, le=1.0)
    consecutive_approvals: int = Field(default=0, ge=0)
    max_consecutive: int = Field(default=0, ge=0)
    trust_history: list[dict[str, Any]] = Field(default_factory=list)
    last_violation_time: float = Field(default=0.0)
    promotions: int = Field(default=0, ge=0)
    demotions: int = Field(default=0, ge=0)


# ============================================================
# 配置
# ============================================================


class FatigueConfig(BaseModel):
    """抗疲劳配置.

    Attributes:
        window_seconds: 疲劳追踪窗口 (秒)
        max_approvals_per_window: 窗口内最大审批数 (超过则触发疲劳)
        max_response_time_seconds: 响应时间上限 (超过则增加疲劳分)
        fatigue_threshold_mild: 轻度疲劳阈值
        fatigue_threshold_moderate: 中度疲劳阈值
        fatigue_threshold_severe: 重度疲劳阈值
        batch_min_size: 批量审批最小项数
        batch_window_seconds: 批量收集窗口
        batch_max_size: 批量审批最大项数
        progressive_trust_threshold: 连续批准次数阈值 (达到则提升信任)
        progressive_trust_increment: 信任度增量
        progressive_trust_max: 信任度上限
        progressive_trust_demote_threshold: 连续拒绝降级阈值
        progressive_trust_demote_decrement: 信任度降级量
        smart_preapproval_min_samples: 智能预批最少历史样本
        smart_preapproval_approval_rate: 智能预批批准率阈值
        fatigue_score_per_approval: 每次审批的疲劳增量
        fatigue_score_per_timeout: 每次超时的疲劳增量
        fatigue_decay_per_second: 每秒疲劳衰减率
    """

    window_seconds: float = Field(default=3600.0, ge=60.0)
    max_approvals_per_window: int = Field(default=20, ge=1)
    max_response_time_seconds: float = Field(default=120.0, ge=1.0)
    fatigue_threshold_mild: float = Field(default=25.0, ge=0.0, le=100.0)
    fatigue_threshold_moderate: float = Field(default=50.0, ge=0.0, le=100.0)
    fatigue_threshold_severe: float = Field(default=75.0, ge=0.0, le=100.0)
    batch_min_size: int = Field(default=3, ge=2)
    batch_window_seconds: float = Field(default=60.0, ge=10.0)
    batch_max_size: int = Field(default=20, ge=2)
    progressive_trust_threshold: int = Field(default=10, ge=1)
    progressive_trust_increment: float = Field(default=0.05, ge=0.01, le=0.5)
    progressive_trust_max: float = Field(default=0.95, ge=0.0, le=1.0)
    progressive_trust_demote_threshold: int = Field(default=3, ge=1)
    progressive_trust_demote_decrement: float = Field(default=0.10, ge=0.01, le=0.5)
    smart_preapproval_min_samples: int = Field(default=5, ge=1)
    smart_preapproval_approval_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    fatigue_score_per_approval: float = Field(default=5.0, ge=0.0)
    fatigue_score_per_timeout: float = Field(default=10.0, ge=0.0)
    fatigue_decay_per_second: float = Field(default=0.01, ge=0.0)


# ============================================================
# 抗疲劳管理器
# ============================================================


class AntiFatigueManager:
    """审批抗疲劳机制管理器.

    综合管理用户审批疲劳状态, 提供批量审批、
    智能预批和渐进信任能力.

    融合方案:
    - NIST AI RMF Measure: 持续度量人类审批负荷
    - Enterprise Approval Matrix: 批量审批降低认知负荷
    - Progressive Trust: 渐进信任模型
    - Adaptive Automation: 动态自动化级别调整
    - Human Factors: 疲劳检测+认知负荷管理

    使用示例::

        manager = AntiFatigueManager()
        # 追踪审批决策
        manager.track_decision("user-001", "quiz_submit", "approved", 2.5)
        # 检查疲劳等级
        state = manager.get_fatigue_state("user-001")
        if state.fatigue_level == FatigueLevel.SEVERE:
            # 启用批量审批或降级
            ...
        # 智能预批
        if manager.should_smart_preapprove("user-001", "quiz_submit"):
            # 自动批准
            ...
    """

    #: 风险等级 → 数值映射
    _RISK_VALUES: dict[str, float] = {
        "low": 0.0,
        "medium": 1.0,
        "high": 2.0,
        "critical": 3.0,
    }

    def __init__(self, config: FatigueConfig | None = None) -> None:
        self._config = config or FatigueConfig()
        self._fatigue_states: dict[str, FatigueState] = {}
        self._approval_patterns: dict[str, dict[str, ApprovalPattern]] = defaultdict(dict)
        self._trust_records: dict[str, ProgressiveTrustRecord] = {}
        self._batch_groups: dict[str, BatchApprovalGroup] = {}
        self._pending_batch_items: dict[str, list[dict[str, Any]]] = defaultdict(list)

    @property
    def config(self) -> FatigueConfig:
        return self._config

    # ==========================================================
    # 疲劳状态追踪
    # ==========================================================

    def track_decision(
        self,
        user_id: str,
        operation: str,
        decision: str,
        response_time: float = 0.0,
        risk_level: str = "medium",
    ) -> FatigueState:
        """追踪用户审批决策.

        每次用户做出审批决策时调用, 更新疲劳状态和审批模式.

        Args:
            user_id: 用户 ID
            operation: 操作类型
            decision: 决策结果 (approved/rejected/modified/timeout/auto_approved)
            response_time: 响应时间 (秒)
            risk_level: 风险等级

        Returns:
            更新后的疲劳状态
        """
        state = self._get_or_create_state(user_id)
        pattern = self._get_or_create_pattern(user_id, operation)
        trust = self._get_or_create_trust(user_id)

        # 更新窗口内计数
        if decision == "approved":
            state.approval_count += 1
            pattern.approved_count += 1
            pattern.consecutive_approvals += 1
            trust.consecutive_approvals += 1
        elif decision == "rejected":
            state.rejection_count += 1
            pattern.rejected_count += 1
            pattern.consecutive_approvals = 0
            trust.consecutive_approvals = 0
            trust.last_violation_time = time.time()
            self._check_trust_demotion(trust)
        elif decision == "modified":
            state.modification_count += 1
            pattern.modified_count += 1
            pattern.consecutive_approvals = 0
            trust.consecutive_approvals = 0
        elif decision == "timeout":
            state.timeout_count += 1
            state.fatigue_score += self._config.fatigue_score_per_timeout
            pattern.consecutive_approvals = 0
            trust.consecutive_approvals = 0
        elif decision == "auto_approved":
            pattern.auto_approved_count += 1
            pattern.approved_count += 1

        # 更新响应时间
        if response_time > 0:
            state.response_times.append(response_time)
            # 保持最近 50 条
            if len(state.response_times) > 50:
                state.response_times = state.response_times[-50:]
            state.avg_response_time = round(
                sum(state.response_times) / len(state.response_times), 3
            )

        # 更新操作模式
        pattern.total_count += 1
        pattern.last_decision = decision
        pattern.last_decision_time = time.time()
        risk_val = self._RISK_VALUES.get(risk_level, 1.0)
        if pattern.total_count > 0:
            pattern.avg_risk_level = round(
                (pattern.avg_risk_level * (pattern.total_count - 1) + risk_val)
                / pattern.total_count,
                3,
            )

        # 更新渐进信任
        self._check_trust_promotion(trust)

        # 计算疲劳评分
        self._compute_fatigue_score(state)

        state.last_updated = time.time()
        return state

    def get_fatigue_state(self, user_id: str) -> FatigueState:
        """获取用户疲劳状态."""
        return self._get_or_create_state(user_id)

    def get_fatigue_level(self, user_id: str) -> FatigueLevel:
        """获取用户疲劳等级."""
        return self._get_or_create_state(user_id).fatigue_level

    def _get_or_create_state(self, user_id: str) -> FatigueState:
        """获取或创建疲劳状态."""
        if user_id not in self._fatigue_states:
            self._fatigue_states[user_id] = FatigueState(user_id=user_id)
        state = self._fatigue_states[user_id]
        # 检查窗口是否过期
        if time.time() - state.window_start > self._config.window_seconds:
            self._reset_window(state)
        return state

    def _reset_window(self, state: FatigueState) -> None:
        """重置疲劳窗口."""
        state.window_start = time.time()
        state.approval_count = 0
        state.rejection_count = 0
        state.modification_count = 0
        state.timeout_count = 0
        state.response_times = []
        state.avg_response_time = 0.0

    def _compute_fatigue_score(self, state: FatigueState) -> None:
        """计算疲劳评分 (0-100).

        评分因素:
        1. 审批频率: 窗口内审批次数占比
        2. 响应时间: 平均响应时间递增表示疲劳
        3. 超时率: 超时次数占比
        4. 修改率: 修改次数高表示注意力下降
        5. 时间衰减: 长时间无操作疲劳降低
        """
        score = 0.0

        # 1. 审批频率 (最高 40 分)
        freq_ratio = min(
            state.total_decisions / max(self._config.max_approvals_per_window, 1),
            1.0,
        )
        score += freq_ratio * 40.0

        # 2. 响应时间 (最高 25 分)
        if state.avg_response_time > 0:
            time_ratio = min(
                state.avg_response_time / self._config.max_response_time_seconds,
                1.0,
            )
            score += time_ratio * 25.0

        # 3. 超时率 (最高 20 分)
        if state.total_decisions > 0:
            timeout_ratio = state.timeout_count / state.total_decisions
            score += timeout_ratio * 20.0

        # 4. 修改率 (最高 15 分)
        if state.total_decisions > 0:
            mod_ratio = state.modification_count / state.total_decisions
            score += mod_ratio * 15.0

        # 叠加超时直接增量
        score += state.fatigue_score * 0.3  # 保留部分历史超时影响

        # 5. 时间衰减
        elapsed = time.time() - state.last_updated
        if elapsed > 0:
            score -= elapsed * self._config.fatigue_decay_per_second

        score = max(0.0, min(100.0, score))
        state.fatigue_score = round(score, 2)

        # 确定疲劳等级
        if score < self._config.fatigue_threshold_mild:
            state.fatigue_level = FatigueLevel.NONE
        elif score < self._config.fatigue_threshold_moderate:
            state.fatigue_level = FatigueLevel.MILD
        elif score < self._config.fatigue_threshold_severe:
            state.fatigue_level = FatigueLevel.MODERATE
        else:
            state.fatigue_level = FatigueLevel.SEVERE

    # ==========================================================
    # 批量审批
    # ==========================================================

    def add_to_batch(
        self,
        user_id: str,
        operation: str,
        request_data: dict[str, Any],
    ) -> BatchApprovalGroup | None:
        """将审批请求添加到批量组.

        相同操作类型的请求在批量窗口内聚合,
        达到最小批量大小后返回批量组.

        Args:
            user_id: 用户 ID
            operation: 操作类型
            request_data: 请求数据

        Returns:
            就绪的批量组 (如果达到最小批量大小), 否则 None
        """
        key = f"{user_id}:{operation}"
        self._pending_batch_items[key].append(request_data)

        # 检查是否达到最小批量大小
        if len(self._pending_batch_items[key]) >= self._config.batch_min_size:
            batch = BatchApprovalGroup(
                user_id=user_id,
                operation=operation,
                items=list(self._pending_batch_items[key]),
                status=BatchStatus.READY,
                risk_summary=self._compute_batch_risk_summary(
                    self._pending_batch_items[key]
                ),
            )
            self._batch_groups[batch.batch_id] = batch
            self._pending_batch_items[key].clear()
            return batch

        return None

    def get_ready_batches(self, user_id: str) -> list[BatchApprovalGroup]:
        """获取用户就绪的批量审批组."""
        return [
            b for b in self._batch_groups.values()
            if b.user_id == user_id
            and b.status == BatchStatus.READY
            and not b.is_expired
        ]

    def resolve_batch(
        self,
        batch_id: str,
        decision: str,
        decided_by: str = "",
    ) -> BatchApprovalGroup | None:
        """解决批量审批组.

        Args:
            batch_id: 批次 ID
            decision: "approved" 或 "rejected"
            decided_by: 决策人

        Returns:
            更新后的批量组
        """
        batch = self._batch_groups.get(batch_id)
        if batch is None or batch.status != BatchStatus.READY:
            return None

        if decision == "approved":
            batch.status = BatchStatus.APPROVED
        elif decision == "rejected":
            batch.status = BatchStatus.REJECTED
        else:
            batch.status = BatchStatus.PARTIAL

        return batch

    def expire_batches(self) -> int:
        """过期所有过期的批量组.

        Returns:
            过期的批量组数量
        """
        count = 0
        for batch in self._batch_groups.values():
            if batch.status == BatchStatus.READY and batch.is_expired:
                batch.status = BatchStatus.EXPIRED
                count += 1
        return count

    def _compute_batch_risk_summary(
        self, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """计算批量组风险摘要."""
        risk_counts: dict[str, int] = defaultdict(int)
        for item in items:
            risk = item.get("risk_level", "medium")
            risk_counts[risk] += 1
        return {
            "total_items": len(items),
            "risk_distribution": dict(risk_counts),
            "max_risk": max(
                risk_counts.keys(),
                key=lambda r: self._RISK_VALUES.get(r, 1.0),
                default="medium",
            ),
        }

    # ==========================================================
    # 智能预批
    # ==========================================================

    def should_smart_preapprove(
        self,
        user_id: str,
        operation: str,
        risk_level: str = "low",
    ) -> bool:
        """判断是否应智能预批.

        基于历史审批模式预测当前操作是否可自动批准.

        条件:
        1. 历史样本数 ≥ smart_preapproval_min_samples
        2. 批准率 ≥ smart_preapproval_approval_rate
        3. 风险等级为低
        4. 用户当前无重度疲劳
        5. 渐进信任分 ≥ 0.7

        Args:
            user_id: 用户 ID
            operation: 操作类型
            risk_level: 风险等级

        Returns:
            True 如果应智能预批
        """
        # 安全操作不预批
        if risk_level in ("high", "critical"):
            return False

        pattern = self._approval_patterns.get(user_id, {}).get(operation)
        if pattern is None:
            return False

        # 样本数检查
        if pattern.total_count < self._config.smart_preapproval_min_samples:
            return False

        # 批准率检查
        if pattern.approval_rate < self._config.smart_preapproval_approval_rate:
            return False

        # 疲劳检查
        state = self._get_or_create_state(user_id)
        if state.fatigue_level == FatigueLevel.SEVERE:
            return False

        # 信任检查
        trust = self._get_or_create_trust(user_id)
        if trust.trust_score < 0.7:
            return False

        return True

    def get_approval_pattern(
        self, user_id: str, operation: str
    ) -> ApprovalPattern | None:
        """获取用户的操作审批模式."""
        return self._approval_patterns.get(user_id, {}).get(operation)

    # ==========================================================
    # 渐进信任
    # ==========================================================

    def get_trust_score(self, user_id: str) -> float:
        """获取用户渐进信任分."""
        return self._get_or_create_trust(user_id).trust_score

    def get_trust_record(self, user_id: str) -> ProgressiveTrustRecord:
        """获取用户渐进信任记录."""
        return self._get_or_create_trust(user_id)

    def promote_trust(self, user_id: str, reason: str = "") -> float:
        """手动提升用户信任度.

        Returns:
            提升后的信任度
        """
        trust = self._get_or_create_trust(user_id)
        old = trust.trust_score
        trust.trust_score = min(
            self._config.progressive_trust_max,
            trust.trust_score + self._config.progressive_trust_increment,
        )
        trust.promotions += 1
        trust.trust_history.append({
            "action": "promote",
            "old": old,
            "new": trust.trust_score,
            "reason": reason,
            "timestamp": time.time(),
        })
        return trust.trust_score

    def demote_trust(self, user_id: str, reason: str = "") -> float:
        """手动降低用户信任度.

        Returns:
            降低后的信任度
        """
        trust = self._get_or_create_trust(user_id)
        old = trust.trust_score
        trust.trust_score = max(
            0.0,
            trust.trust_score - self._config.progressive_trust_demote_decrement,
        )
        trust.demotions += 1
        trust.last_violation_time = time.time()
        trust.trust_history.append({
            "action": "demote",
            "old": old,
            "new": trust.trust_score,
            "reason": reason,
            "timestamp": time.time(),
        })
        return trust.trust_score

    def _check_trust_promotion(self, trust: ProgressiveTrustRecord) -> None:
        """检查并执行渐进信任提升."""
        if trust.consecutive_approvals >= self._config.progressive_trust_threshold:
            if trust.trust_score < self._config.progressive_trust_max:
                old = trust.trust_score
                trust.trust_score = min(
                    self._config.progressive_trust_max,
                    trust.trust_score + self._config.progressive_trust_increment,
                )
                trust.promotions += 1
                trust.max_consecutive = max(
                    trust.max_consecutive, trust.consecutive_approvals
                )
                trust.consecutive_approvals = 0  # 重置计数器
                trust.trust_history.append({
                    "action": "auto_promote",
                    "old": old,
                    "new": trust.trust_score,
                    "reason": "consecutive_approvals_threshold",
                    "timestamp": time.time(),
                })

    def _check_trust_demotion(self, trust: ProgressiveTrustRecord) -> None:
        """检查并执行渐进信任降级."""
        if trust.trust_score > 0.0:
            old = trust.trust_score
            trust.trust_score = max(
                0.0,
                trust.trust_score - self._config.progressive_trust_demote_decrement,
            )
            trust.demotions += 1
            trust.trust_history.append({
                "action": "auto_demote",
                "old": old,
                "new": trust.trust_score,
                "reason": "rejection_violation",
                "timestamp": time.time(),
            })

    def _get_or_create_trust(self, user_id: str) -> ProgressiveTrustRecord:
        """获取或创建渐进信任记录."""
        if user_id not in self._trust_records:
            self._trust_records[user_id] = ProgressiveTrustRecord(
                user_id=user_id,
                trust_score=0.5,
                base_trust=0.5,
            )
        return self._trust_records[user_id]

    # ==========================================================
    # 疲劳驱动的降级建议
    # ==========================================================

    def get_fatigue_adjustment(
        self, user_id: str
    ) -> dict[str, Any]:
        """获取疲劳驱动的调整建议.

        根据用户疲劳等级返回调整建议:
        - NONE: 无调整
        - MILD: 建议启用批量审批
        - MODERATE: 建议启用智能预批
        - SEVERE: 建议降级到 L2 提示层

        Returns:
            调整建议字典
        """
        state = self._get_or_create_state(user_id)
        trust = self._get_or_create_trust(user_id)

        adjustments: dict[str, Any] = {
            "fatigue_level": state.fatigue_level.value,
            "fatigue_score": state.fatigue_score,
            "trust_score": trust.trust_score,
            "recommendations": [],
        }

        if state.fatigue_level == FatigueLevel.MILD:
            adjustments["recommendations"].append("enable_batch_approval")
        elif state.fatigue_level == FatigueLevel.MODERATE:
            adjustments["recommendations"].extend([
                "enable_batch_approval",
                "enable_smart_preapproval",
            ])
        elif state.fatigue_level == FatigueLevel.SEVERE:
            adjustments["recommendations"].extend([
                "enable_batch_approval",
                "enable_smart_preapproval",
                "downgrade_to_l2_prompt",
            ])

        # 信任度高 + 疲劳高 → 建议扩大自动批准范围
        if trust.trust_score >= 0.8 and state.fatigue_level != FatigueLevel.NONE:
            adjustments["recommendations"].append("expand_auto_approval_scope")

        return adjustments

    # ==========================================================
    # 审批模式辅助
    # ==========================================================

    def _get_or_create_pattern(
        self, user_id: str, operation: str
    ) -> ApprovalPattern:
        """获取或创建审批模式."""
        if operation not in self._approval_patterns[user_id]:
            self._approval_patterns[user_id][operation] = ApprovalPattern(
                operation=operation
            )
        return self._approval_patterns[user_id][operation]

    # ==========================================================
    # 统计
    # ==========================================================

    def get_statistics(self) -> dict[str, Any]:
        """获取抗疲劳全局统计."""
        total_users = len(self._fatigue_states)
        if total_users == 0:
            return {"total_users": 0}

        fatigue_levels: dict[str, int] = defaultdict(int)
        total_trust = 0.0
        total_batch_groups = len(self._batch_groups)
        ready_batches = sum(
            1 for b in self._batch_groups.values()
            if b.status == BatchStatus.READY
        )

        for state in self._fatigue_states.values():
            fatigue_levels[state.fatigue_level.value] += 1

        for trust in self._trust_records.values():
            total_trust += trust.trust_score

        avg_trust = total_trust / total_users if total_users > 0 else 0.0

        # 统计操作模式
        total_patterns = 0
        high_approval_patterns = 0
        for user_patterns in self._approval_patterns.values():
            for pattern in user_patterns.values():
                total_patterns += 1
                if pattern.is_high_approval:
                    high_approval_patterns += 1

        return {
            "total_users": total_users,
            "fatigue_distribution": dict(fatigue_levels),
            "avg_trust_score": round(avg_trust, 3),
            "total_batch_groups": total_batch_groups,
            "ready_batches": ready_batches,
            "total_approval_patterns": total_patterns,
            "high_approval_patterns": high_approval_patterns,
        }

    def clear(self) -> None:
        """清空所有状态."""
        self._fatigue_states.clear()
        self._approval_patterns.clear()
        self._trust_records.clear()
        self._batch_groups.clear()
        self._pending_batch_items.clear()
