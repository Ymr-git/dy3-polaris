"""CC2 计划审批门 — L3 审批工作流.

实现设计文档第四章 L3 审批层的完整工作流:
- 四种审批模式 (快速确认/详细审批/协商审批/规则预设)
- 超时处理策略 (中止/自动批准/降级为提示)
- 信任模式窗口 (30分钟自动批准)
- 审批状态机 (Pending → Approved/Rejected/Modified/Timeout)
- 审批决策矩阵 (角色 × 场景 × 风险等级)

融合世界先进方案:
- LangGraph interrupt: 审批请求作为中断点, 阻塞执行
- LangGraph Command: 审批决策作为恢复命令, 恢复执行
- CrewAI human_input: 任务级审批回调
- Enterprise Approval Workflows: 四种审批门放置模式
- GAIA: 承诺检测 — 不可逆操作强制升级
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .routing_engine import (
    ApprovalMode,
    TimeoutAction,
    RiskLevel,
    Reversibility,
    UserRole,
)


# ============================================================
# 审批状态枚举
# ============================================================


class ApprovalStatus(str, Enum):
    """审批状态机.

    [*] → Pending → (Approved/Rejected/Modified/Timeout)
                    → (Executed/Cancelled/Aborted/AutoApproved/Downgraded)
                    → Logged → [*]
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMEOUT = "timeout"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    AUTO_APPROVED = "auto_approved"
    DOWNGRADED = "downgraded"
    LOGGED = "logged"


# ============================================================
# 审批数据模型
# ============================================================


class ApprovalRequest(BaseModel):
    """审批请求 (LangGraph interrupt 启发).

    当路由引擎决定 L3 审批时, 创建标准化审批请求.

    Attributes:
        request_id: 审批请求 ID
        operation: 操作类型
        target: 操作目标
        risk_level: 风险等级
        reversibility: 可逆性
        approval_mode: 审批模式
        cost_estimate: 成本估算
        risk_assessment: 风险评估
        alternatives: 备选方案列表
        policy_reference: 策略引用
        requester: 请求方 (通常是 Agent ID)
        approver_roles: 审批人角色列表
        context: 上下文信息
        timeout_seconds: 超时时间
        timeout_action: 超时策略
        created_at: 创建时间
        expires_at: 过期时间
    """

    request_id: str = Field(
        default_factory=lambda: f"apr-{uuid.uuid4().hex[:10]}"
    )
    operation: str = Field(description="操作类型")
    target: str = Field(default="")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    reversibility: Reversibility = Field(
        default=Reversibility.PARTIALLY_REVERSIBLE
    )
    approval_mode: ApprovalMode = Field(
        default=ApprovalMode.DETAILED_REVIEW
    )
    cost_estimate: dict[str, Any] = Field(default_factory=dict)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    policy_reference: str = Field(default="")
    requester: str = Field(default="")
    approver_roles: list[UserRole] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=300.0)
    timeout_action: TimeoutAction = Field(default=TimeoutAction.ABORT)
    created_at: float = Field(default_factory=time.time)
    expires_at: float = Field(default=0.0)

    def model_post_init(self, __context: Any) -> None:
        """计算过期时间."""
        if self.expires_at == 0.0 and self.timeout_seconds > 0:
            self.expires_at = self.created_at + self.timeout_seconds


class ApprovalDecision(BaseModel):
    """审批决策 (LangGraph Command resume 启发).

    Attributes:
        decision_id: 决策 ID
        request_id: 关联的审批请求 ID
        decision: 决策结果 (approved/rejected/modified)
        selected_alternative: 选择的备选方案 ID (如有)
        comment: 审批备注
        modified_parameters: 修改后的参数 (mode=modified 时)
        decided_by: 决策人
        decided_at: 决策时间
    """

    decision_id: str = Field(
        default_factory=lambda: f"dec-{uuid.uuid4().hex[:10]}"
    )
    request_id: str = Field(description="关联审批请求 ID")
    decision: ApprovalStatus = Field(default=ApprovalStatus.APPROVED)
    selected_alternative: str = Field(default="")
    comment: str = Field(default="")
    modified_parameters: dict[str, Any] = Field(default_factory=dict)
    decided_by: str = Field(default="")
    decided_at: float = Field(default_factory=time.time)


class ApprovalRecord(BaseModel):
    """完整审批记录 (审计轨迹).

    Attributes:
        record_id: 记录 ID
        request: 审批请求
        decision: 审批决策 (如有)
        status: 当前状态
        response_time_seconds: 响应时间 (秒)
        metadata: 附加元数据
    """

    record_id: str = Field(
        default_factory=lambda: f"rec-{uuid.uuid4().hex[:12]}"
    )
    request: ApprovalRequest
    decision: ApprovalDecision | None = None
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    response_time_seconds: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 信任模式窗口
# ============================================================


class TrustModeWindow:
    """信任模式窗口 (设计文档 4.4 节).

    学生可自主选择 30 分钟自动批准窗口, 窗口期内
    低风险可逆操作自动通过. 安全相关操作始终需手动审批.

    Attributes:
        user_id: 用户 ID
        duration_seconds: 窗口持续时间 (默认 1800 = 30 分钟)
        start_time: 窗口开始时间
        active: 是否激活
    """

    def __init__(
        self,
        user_id: str,
        duration_seconds: float = 1800.0,
    ) -> None:
        self.user_id = user_id
        self.duration_seconds = duration_seconds
        self.start_time: float = 0.0
        self.active: bool = False

    def activate(self) -> None:
        """激活信任模式窗口."""
        self.start_time = time.time()
        self.active = True

    def deactivate(self) -> None:
        """停用信任模式窗口."""
        self.active = False

    def is_valid(self) -> bool:
        """检查窗口是否仍有效 (未过期)."""
        if not self.active:
            return False
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration_seconds:
            self.active = False
            return False
        return True

    def remaining_seconds(self) -> float:
        """剩余有效时间."""
        if not self.is_valid():
            return 0.0
        elapsed = time.time() - self.start_time
        return max(0.0, self.duration_seconds - elapsed)


# ============================================================
# 审批工作流管理器
# ============================================================


class ApprovalWorkflowManager:
    """L3 审批工作流管理器.

    管理审批请求的完整生命周期:
    创建 → 等待决策 → 超时处理 → 记录审计

    融合方案:
    - LangGraph interrupt: 审批请求作为中断点
    - LangGraph Command: 决策作为恢复命令
    - CrewAI human_input: 回调式审批
    - Enterprise Workflows: 超时策略矩阵
    - GAIA: 不可逆操作承诺检测

    使用示例::

        manager = ApprovalWorkflowManager()
        request = manager.create_request(
            operation="learning_path_reset",
            risk_level=RiskLevel.HIGH,
            approval_mode=ApprovalMode.DETAILED_REVIEW,
            timeout_seconds=300,
        )
        # ... 等待人工决策 ...
        decision = manager.make_decision(
            request.request_id,
            decision=ApprovalStatus.APPROVED,
            decided_by="teacher_001",
        )
    """

    #: 安全相关操作 (不受信任模式影响)
    SAFETY_OPERATIONS: set[str] = {
        "data_delete",
        "data_overwrite",
        "prompt_template_modify",
        "policy_change",
        "user_data_export",
    }

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}
        self._trust_windows: dict[str, TrustModeWindow] = {}
        self._rule_presets: dict[str, dict[str, Any]] = {}

    @property
    def records(self) -> dict[str, ApprovalRecord]:
        return dict(self._records)

    def create_request(
        self,
        operation: str,
        target: str = "",
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        reversibility: Reversibility = Reversibility.PARTIALLY_REVERSIBLE,
        approval_mode: ApprovalMode = ApprovalMode.DETAILED_REVIEW,
        requester: str = "",
        approver_roles: list[UserRole] | None = None,
        timeout_seconds: float = 300.0,
        timeout_action: TimeoutAction = TimeoutAction.ABORT,
        alternatives: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        policy_reference: str = "",
        user_id: str = "",
    ) -> ApprovalRequest:
        """创建审批请求.

        如果用户有有效信任模式窗口且操作非安全相关,
        则自动批准 (返回 AUTO_APPROVED 状态的请求).

        Args:
            operation: 操作类型
            target: 操作目标
            risk_level: 风险等级
            reversibility: 可逆性
            approval_mode: 审批模式
            requester: 请求方
            approver_roles: 审批人角色列表
            timeout_seconds: 超时时间
            timeout_action: 超时策略
            alternatives: 备选方案
            context: 上下文
            policy_reference: 策略引用
            user_id: 用户 ID (用于信任模式检查)

        Returns:
            审批请求
        """
        request = ApprovalRequest(
            operation=operation,
            target=target,
            risk_level=risk_level,
            reversibility=reversibility,
            approval_mode=approval_mode,
            requester=requester,
            approver_roles=approver_roles or [UserRole.STUDENT],
            timeout_seconds=timeout_seconds,
            timeout_action=timeout_action,
            alternatives=alternatives or [],
            context=context or {},
            policy_reference=policy_reference,
        )

        # 信任模式检查
        if user_id and self._check_trust_mode(user_id, operation, risk_level, reversibility):
            record = ApprovalRecord(
                request=request,
                status=ApprovalStatus.AUTO_APPROVED,
                response_time_seconds=0.0,
                metadata={"auto_approved_reason": "trust_mode_window"},
            )
            self._records[request.request_id] = record
            return request

        # 规则预设检查
        if self._check_rule_preset(operation, risk_level):
            record = ApprovalRecord(
                request=request,
                status=ApprovalStatus.AUTO_APPROVED,
                response_time_seconds=0.0,
                metadata={"auto_approved_reason": "rule_preset"},
            )
            self._records[request.request_id] = record
            return request

        # 创建待审批记录
        record = ApprovalRecord(
            request=request,
            status=ApprovalStatus.PENDING,
        )
        self._records[request.request_id] = record
        return request

    def make_decision(
        self,
        request_id: str,
        decision: ApprovalStatus = ApprovalStatus.APPROVED,
        decided_by: str = "",
        comment: str = "",
        selected_alternative: str = "",
        modified_parameters: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        """对审批请求做出决策.

        Args:
            request_id: 审批请求 ID
            decision: 决策结果
            decided_by: 决策人
            comment: 审批备注
            selected_alternative: 选择的备选方案
            modified_parameters: 修改后的参数

        Returns:
            更新后的审批记录

        Raises:
            KeyError: 请求不存在
            ValueError: 请求已决策
        """
        if request_id not in self._records:
            raise KeyError(f"审批请求不存在: {request_id}")

        record = self._records[request_id]
        if record.status not in (
            ApprovalStatus.PENDING,
            ApprovalStatus.AUTO_APPROVED,
        ):
            raise ValueError(
                f"审批请求已处理: {request_id}, 当前状态: {record.status.value}"
            )

        decision_obj = ApprovalDecision(
            request_id=request_id,
            decision=decision,
            decided_by=decided_by,
            comment=comment,
            selected_alternative=selected_alternative,
            modified_parameters=modified_parameters or {},
        )

        record.decision = decision_obj
        record.status = decision
        record.response_time_seconds = time.time() - record.request.created_at

        return record

    def check_timeout(self, request_id: str) -> ApprovalRecord | None:
        """检查审批请求是否超时, 并执行超时策略.

        Args:
            request_id: 审批请求 ID

        Returns:
            更新后的记录 (如果超时), 否则 None
        """
        if request_id not in self._records:
            return None

        record = self._records[request_id]
        if record.status != ApprovalStatus.PENDING:
            return None

        request = record.request
        if request.timeout_seconds <= 0:
            return None

        elapsed = time.time() - request.created_at
        if elapsed < request.timeout_seconds:
            return None

        # 执行超时策略
        action = request.timeout_action
        if action == TimeoutAction.ABORT:
            record.status = ApprovalStatus.ABORTED
        elif action == TimeoutAction.AUTO_APPROVE:
            record.status = ApprovalStatus.AUTO_APPROVED
        elif action == TimeoutAction.DOWNGRADE_TO_PROMPT:
            record.status = ApprovalStatus.DOWNGRADED
        elif action == TimeoutAction.ESCALATE:
            record.status = ApprovalStatus.ABORTED
            record.metadata["escalated"] = True

        record.status = ApprovalStatus.TIMEOUT
        record.metadata["timeout_action"] = action.value
        record.response_time_seconds = elapsed

        return record

    def cancel_request(self, request_id: str) -> ApprovalRecord | None:
        """取消审批请求."""
        if request_id not in self._records:
            return None
        record = self._records[request_id]
        if record.status != ApprovalStatus.PENDING:
            return None
        record.status = ApprovalStatus.CANCELLED
        return record

    def get_record(self, request_id: str) -> ApprovalRecord | None:
        """获取审批记录."""
        return self._records.get(request_id)

    def list_records(
        self,
        *,
        status: ApprovalStatus | None = None,
        operation: str | None = None,
        requester: str | None = None,
        limit: int = 100,
    ) -> list[ApprovalRecord]:
        """列出审批记录, 支持过滤."""
        results = list(self._records.values())
        if status:
            results = [r for r in results if r.status == status]
        if operation:
            results = [r for r in results if r.request.operation == operation]
        if requester:
            results = [r for r in results if r.request.requester == requester]
        results.sort(
            key=lambda r: r.request.created_at, reverse=True
        )
        return results[:limit]

    def get_statistics(self) -> dict[str, Any]:
        """获取审批统计."""
        total = len(self._records)
        if total == 0:
            return {"total": 0}

        by_status: dict[str, int] = {}
        by_mode: dict[str, int] = {}
        response_times: list[float] = []

        for record in self._records.values():
            status_key = record.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1
            mode_key = record.request.approval_mode.value
            by_mode[mode_key] = by_mode.get(mode_key, 0) + 1
            if record.response_time_seconds > 0:
                response_times.append(record.response_time_seconds)

        avg_response = (
            sum(response_times) / len(response_times)
            if response_times else 0.0
        )

        return {
            "total": total,
            "by_status": by_status,
            "by_mode": by_mode,
            "avg_response_time": round(avg_response, 2),
            "auto_approved_count": by_status.get("auto_approved", 0),
            "timeout_count": by_status.get("timeout", 0),
        }

    # --------------------------------------------------------
    # 信任模式管理
    # --------------------------------------------------------

    def activate_trust_mode(
        self, user_id: str, duration_seconds: float = 1800.0
    ) -> TrustModeWindow:
        """为用户激活信任模式窗口."""
        window = TrustModeWindow(user_id, duration_seconds)
        window.activate()
        self._trust_windows[user_id] = window
        return window

    def deactivate_trust_mode(self, user_id: str) -> None:
        """停用用户的信任模式窗口."""
        if user_id in self._trust_windows:
            self._trust_windows[user_id].deactivate()

    def get_trust_mode(self, user_id: str) -> TrustModeWindow | None:
        """获取用户的信任模式窗口."""
        return self._trust_windows.get(user_id)

    def _check_trust_mode(
        self,
        user_id: str,
        operation: str,
        risk_level: RiskLevel,
        reversibility: Reversibility,
    ) -> bool:
        """检查信任模式是否允许自动批准."""
        window = self._trust_windows.get(user_id)
        if not window or not window.is_valid():
            return False

        # 安全相关操作不受信任模式影响
        if operation in self.SAFETY_OPERATIONS:
            return False

        # 仅低风险可逆操作可自动批准
        if risk_level != RiskLevel.LOW:
            return False
        if reversibility != Reversibility.REVERSIBLE:
            return False

        return True

    # --------------------------------------------------------
    # 规则预设管理
    # --------------------------------------------------------

    def add_rule_preset(
        self,
        preset_id: str,
        operation: str,
        risk_level: RiskLevel = RiskLevel.LOW,
        action: str = "auto_approve",
    ) -> None:
        """添加审批规则预设.

        用户可预设规则, 如"总是批准此类操作".
        """
        self._rule_presets[preset_id] = {
            "operation": operation,
            "risk_level": risk_level.value,
            "action": action,
        }

    def remove_rule_preset(self, preset_id: str) -> None:
        """移除审批规则预设."""
        self._rule_presets.pop(preset_id, None)

    def _check_rule_preset(
        self, operation: str, risk_level: RiskLevel
    ) -> bool:
        """检查是否有匹配的规则预设允许自动批准."""
        for preset in self._rule_presets.values():
            if (
                preset["operation"] == operation
                and preset["risk_level"] == risk_level.value
                and preset["action"] == "auto_approve"
            ):
                return True
        return False

    def clear(self) -> None:
        """清空所有审批记录和信任窗口."""
        self._records.clear()
        self._trust_windows.clear()
        self._rule_presets.clear()
