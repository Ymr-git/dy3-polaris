"""CC2 计划审批门 — L4 干预层管理器.

实现设计文档 CC2-Plan-Approval-Gate 第四章 L4 干预层的完整干预管理:
1. 紧急暂停 (Emergency Pause) — 学生触发紧急停止, 系统立即停机
2. 人工接管 (Manual Override) — 人类操作者从 AI Agent 接管控制权
3. 纠正反馈 (Correction Feedback) — 人类纠正 AI 输出并提供反馈
4. 创意请求 (Creative Request) — 人类请求 AI 提供创意/发散性输入

四种干预类型对应不同的人机协作范式:
- 紧急暂停: 高优先级安全阀门, 阻塞全部 Agent 执行
- 人工接管: 控制权转移, Agent 暂停自动决策
- 纠正反馈: 在线学习闭环, 纠正结果可触发 CC1 重新评审
- 创意请求: 人机共创, 人类引导 AI 发散思维

融合世界先进方案:
- LangGraph interrupt(): 紧急暂停作为图级中断, 阻塞执行流直至恢复
- Swarm handoff(): 人工接管作为 Agent 间移交, 控制权从 AI 转向人类
- CrewAI human_input(): 纠正反馈作为任务级人类输入, 修正 Agent 输出
- GAIA 协议: 创意请求作为协商回合, 人机交替提议
- NIST AI RMF: 全审计轨迹支撑 Govern-Map-Measure-Manage 治理功能

设计原则:
- 线程安全: 所有共享状态使用 threading.RLock() 保护
- 完整审计: 所有干预事件全程记录, 不可篡改
- 路由引擎集成: 复用 RiskLevel / Reversibility / UserRole 等共享枚举
- CC1 集成: 纠正反馈可触发 CC1 四层重新评审
- 通知支持: 紧急暂停自动通知教师
- 恢复模式: 支持 resume_from_checkpoint / restart_from_new_state
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import uuid
from typing import Any, Callable

from pydantic import BaseModel, Field

from .routing_engine import (
    InterventionTypeL4,
    Priority,
    RecoveryMode,
    Reversibility,
    RiskLevel,
    UserRole,
)

logger = logging.getLogger(__name__)

#: 通知回调类型签名 — 接收事件载荷字典, 无返回值
NotificationCallback = Callable[[dict[str, Any]], None]


# ============================================================
# 枚举定义
# ============================================================


class InterventionAction(str, enum.Enum):
    """干预动作类型.

    定义干预生命周期中可执行的动作, 用于时间线追踪.
    完整生命周期: pause → (override / correct / redirect) → resume / terminate

    动作语义:
    - pause: 暂停执行 (紧急暂停触发)
    - resume: 恢复执行 (暂停/接管解除后恢复)
    - override: 接管控制权 (人工接管触发)
    - correct: 纠正输出 (纠正反馈触发)
    - redirect: 重定向思维 (创意请求触发)
    - terminate: 终止流程 (彻底中止当前任务)
    """

    PAUSE = "pause"
    RESUME = "resume"
    OVERRIDE = "override"
    CORRECT = "correct"
    REDIRECT = "redirect"
    TERMINATE = "terminate"


class PauseScope(str, enum.Enum):
    """紧急暂停作用范围.

    定义紧急暂停影响的系统范围, 从局部到全局递进:
    - agent: 单个 Agent 暂停 — 仅影响指定 Agent
    - session: 会话级暂停 — 影响当前学习会话全部 Agent
    - module: 模块级暂停 — 影响特定功能模块 (如测验模块)
    - global: 全局暂停 — 影响整个系统全部 Agent
    """

    AGENT = "agent"
    SESSION = "session"
    MODULE = "module"
    GLOBAL = "global"


class OverrideLevel(str, enum.Enum):
    """人工接管级别 (Swarm handoff 启发).

    定义人类操作者接管 Agent 的控制强度, 从建议到绝对控制:
    - advisory: 建议级 — 人类提供指导, Agent 可选择采纳
    - executive: 执行级 — 人类直接控制 Agent 行为, Agent 遵照执行
    - absolute: 绝对级 — 人类完全接管, Agent 暂停所有自主决策
    """

    ADVISORY = "advisory"
    EXECUTIVE = "executive"
    ABSOLUTE = "absolute"


class CorrectionType(str, enum.Enum):
    """纠正类型 (CrewAI human_input 启发).

    定义人类纠正 AI 输出的类型分类, 用于 CC1 重新评审路由:
    - factual: 事实性纠正 — 知识点事实错误 (如公式、定义、日期)
    - procedural: 程序性纠正 — 步骤/流程错误 (如解题步骤顺序)
    - conceptual: 概念性纠正 — 概念理解偏差 (如概念混淆)
    - pedagogical: 教学法纠正 — 教学策略不当 (如难度不匹配)
    """

    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    PEDAGOGICAL = "pedagogical"


class CorrectionSeverity(str, enum.Enum):
    """纠正严重程度.

    定义纠正的严重性等级, 影响是否触发 CC1 重新评审:
    - minor: 轻微 — 表述优化, 不触发 CC1
    - moderate: 中等 — 内容修正, 可选触发 CC1
    - major: 严重 — 核心内容错误, 触发 CC1 重新评审
    - critical: 致命 — 严重误导性错误, 强制触发 CC1 + 人工仲裁
    """

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class CreativeRequestType(str, enum.Enum):
    """创意请求类型 (GAIA 协商协议启发).

    定义人类向 AI 请求创意输入的类型:
    - brainstorm: 头脑风暴 — 请求发散性思维, 生成多种可能
    - alternative: 替代方案 — 请求不同于当前思路的备选方案
    - example: 示例生成 — 请求具体示例说明概念
    - metaphor: 类比隐喻 — 请求用类比方式解释抽象概念
    """

    BRAINSTORM = "brainstorm"
    ALTERNATIVE = "alternative"
    EXAMPLE = "example"
    METAPHOR = "metaphor"


class InterventionEventStatus(str, enum.Enum):
    """干预事件状态.

    完整生命周期: initiated → active → resolved / cancelled

    状态流转:
    - initiated: 已发起 — 干预请求已创建, 等待激活
    - active: 活跃 — 干预正在进行, Agent 已暂停或被接管
    - resolved: 已解决 — 干预正常结束, 恢复执行
    - cancelled: 已取消 — 干预被中途取消
    """

    INITIATED = "initiated"
    ACTIVE = "active"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


# ============================================================
# 数据模型
# ============================================================


class TimelineEntry(BaseModel):
    """干预时间线条目.

    记录干预生命周期中的每个关键事件, 构成完整审计轨迹.

    Attributes:
        timestamp: 时间戳
        action: 干预动作类型
        actor: 执行者 ID (用户 ID / Agent ID / system)
        description: 动作描述
        metadata: 附加元数据
    """

    timestamp: float = Field(default_factory=time.time)
    action: InterventionAction = Field(description="动作类型")
    actor: str = Field(default="", description="执行者 ID")
    description: str = Field(default="", description="动作描述")
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterventionOutcome(BaseModel):
    """干预结果.

    记录干预解决后的结构化结果.

    Attributes:
        success: 是否成功解决
        resolution: 解决方式描述
        impact: 影响评估 (如受影响的 Agent 数量)
        follow_up: 后续动作 (如触发 CC1 重新评审)
        notes: 备注
    """

    success: bool = Field(default=True, description="是否成功解决")
    resolution: str = Field(default="", description="解决方式")
    impact: dict[str, Any] = Field(
        default_factory=dict, description="影响评估",
    )
    follow_up: list[str] = Field(
        default_factory=list, description="后续动作",
    )
    notes: str = Field(default="", description="备注")


class EmergencyPauseRequest(BaseModel):
    """紧急暂停请求 (LangGraph interrupt 启发).

    当学生触发紧急停止或系统检测到危险信号时,
    立即创建紧急暂停请求, 阻塞全部相关 Agent 执行.

    紧急暂停是最高优先级的干预类型:
    - 响应时间 < 5 秒 (设计文档要求)
    - 自动通知教师 (可配置)
    - 支持作用范围: 单 Agent / 会话 / 模块 / 全局
    - 恢复模式: 从检查点恢复 或 从新状态重启

    Attributes:
        pause_id: 暂停请求 ID
        user_id: 触发用户 ID (通常为学生)
        reason: 暂停原因
        scope: 暂停作用范围
        agent_ids: 受影响 Agent ID 列表
        auto_notify_teacher: 是否自动通知教师
        risk_level: 风险等级 (默认 HIGH)
        recovery_mode: 恢复模式
        created_at: 创建时间
        resolved_at: 解决时间
        duration_seconds: 暂停持续秒数
        status: 当前状态
        resolution: 解决描述
        resolved_by: 解决者 ID
        notify_payload: 通知载荷
    """

    pause_id: str = Field(
        default_factory=lambda: f"epause-{uuid.uuid4().hex[:10]}",
        description="暂停请求 ID",
    )
    user_id: str = Field(description="触发用户 ID")
    reason: str = Field(default="", description="暂停原因")
    scope: PauseScope = Field(
        default=PauseScope.SESSION, description="暂停作用范围",
    )
    agent_ids: list[str] = Field(
        default_factory=list, description="受影响 Agent ID 列表",
    )
    auto_notify_teacher: bool = Field(
        default=True, description="是否自动通知教师",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.HIGH, description="风险等级",
    )
    recovery_mode: RecoveryMode = Field(
        default=RecoveryMode.RESUME_FROM_CHECKPOINT,
        description="恢复模式",
    )
    created_at: float = Field(default_factory=time.time)
    resolved_at: float | None = Field(default=None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    status: InterventionEventStatus = Field(
        default=InterventionEventStatus.ACTIVE,
        description="当前状态",
    )
    resolution: str = Field(default="", description="解决描述")
    resolved_by: str = Field(default="", description="解决者 ID")
    notify_payload: dict[str, Any] = Field(
        default_factory=dict, description="通知载荷",
    )

    @property
    def is_active(self) -> bool:
        """是否仍处于活跃状态."""
        return self.status in (
            InterventionEventStatus.INITIATED,
            InterventionEventStatus.ACTIVE,
        )

    def resolve(
        self,
        resolution: str,
        resolved_by: str,
    ) -> None:
        """解决紧急暂停.

        Args:
            resolution: 解决描述
            resolved_by: 解决者 ID
        """
        self.status = InterventionEventStatus.RESOLVED
        self.resolution = resolution
        self.resolved_by = resolved_by
        self.resolved_at = time.time()
        self.duration_seconds = round(
            self.resolved_at - self.created_at, 3,
        )

    def cancel(self) -> None:
        """取消紧急暂停."""
        self.status = InterventionEventStatus.CANCELLED
        self.resolved_at = time.time()
        self.duration_seconds = round(
            self.resolved_at - self.created_at, 3,
        )


class ManualOverrideRequest(BaseModel):
    """人工接管请求 (Swarm handoff 启发).

    当人类操作者需要从 AI Agent 接管控制权时创建.
    支持三个接管级别: 建议级 / 执行级 / 绝对级.

    接管期间:
    - advisory: Agent 接收人类指导但保持自主决策权
    - executive: Agent 遵照人类指令执行, 不自主决策
    - absolute: Agent 完全暂停, 人类直接操作系统

    Attributes:
        override_id: 接管请求 ID
        operator_id: 操作者 ID (教师 / 管理员)
        operator_role: 操作者角色
        target_agent: 被接管 Agent ID
        override_level: 接管级别
        duration_seconds: 接管持续时间 (秒), 0=无限
        instructions: 接管指令
        context: 上下文信息
        risk_level: 风险等级
        recovery_mode: 恢复模式
        status: 当前状态
        summary: 接管总结
        released: 是否已释放
        released_at: 释放时间
        released_by: 释放者 ID
        created_at: 创建时间
        actual_duration_seconds: 实际接管持续秒数
    """

    override_id: str = Field(
        default_factory=lambda: f"ovr-{uuid.uuid4().hex[:10]}",
        description="接管请求 ID",
    )
    operator_id: str = Field(description="操作者 ID")
    operator_role: UserRole = Field(
        default=UserRole.TEACHER, description="操作者角色",
    )
    target_agent: str = Field(description="被接管 Agent ID")
    override_level: OverrideLevel = Field(
        default=OverrideLevel.EXECUTIVE, description="接管级别",
    )
    duration_seconds: float = Field(
        default=300.0, ge=0.0, description="接管持续时间, 0=无限",
    )
    instructions: str = Field(default="", description="接管指令")
    context: dict[str, Any] = Field(
        default_factory=dict, description="上下文信息",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.MEDIUM, description="风险等级",
    )
    recovery_mode: RecoveryMode = Field(
        default=RecoveryMode.RESUME_FROM_CHECKPOINT,
        description="恢复模式",
    )
    status: InterventionEventStatus = Field(
        default=InterventionEventStatus.ACTIVE,
        description="当前状态",
    )
    summary: str = Field(default="", description="接管总结")
    released: bool = Field(default=False, description="是否已释放")
    released_at: float | None = Field(default=None)
    released_by: str = Field(default="", description="释放者 ID")
    created_at: float = Field(default_factory=time.time)
    actual_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="实际接管持续秒数",
    )

    @property
    def is_active(self) -> bool:
        """是否仍处于接管状态."""
        return self.status in (
            InterventionEventStatus.INITIATED,
            InterventionEventStatus.ACTIVE,
        )

    def release(
        self,
        summary: str = "",
        released_by: str = "",
    ) -> None:
        """释放人工接管.

        Args:
            summary: 接管总结
            released_by: 释放者 ID
        """
        self.status = InterventionEventStatus.RESOLVED
        self.released = True
        self.released_at = time.time()
        self.released_by = released_by
        self.summary = summary
        self.actual_duration_seconds = round(
            self.released_at - self.created_at, 3,
        )

    def cancel(self) -> None:
        """取消接管."""
        self.status = InterventionEventStatus.CANCELLED
        self.released_at = time.time()
        self.actual_duration_seconds = round(
            self.released_at - self.created_at, 3,
        )


class CorrectionFeedback(BaseModel):
    """纠正反馈 (CrewAI human_input 启发).

    当人类纠正 AI 输出并提供反馈时创建.
    纠正结果可触发 CC1 四层重新评审, 形成在线学习闭环.

    纠正类型与 CC1 联动:
    - factual + critical → 强制 CC1 重新评审 + 人工仲裁
    - conceptual + major → 触发 CC1 重新评审
    - procedural + moderate → 可选 CC1 重新评审
    - pedagogical + minor → 不触发 CC1

    Attributes:
        correction_id: 纠正 ID
        corrector_id: 纠正者 ID (教师 / 管理员 / 专家)
        corrector_role: 纠正者角色
        target_content_id: 被纠正内容 ID
        target_agent_id: 产出内容的 Agent ID
        original_content: 原始内容
        corrected_content: 纠正后内容
        correction_type: 纠正类型
        severity: 严重程度
        feedback_text: 反馈文本
        learning_value: 学习价值评分 (0-1)
        applied: 是否已应用
        applied_by: 应用者 ID
        applied_at: 应用时间
        cc1_re_review_triggered: 是否触发 CC1 重新评审
        cc1_re_review_id: CC1 重新评审 ID (如已触发)
        created_at: 创建时间
        metadata: 附加元数据
    """

    correction_id: str = Field(
        default_factory=lambda: f"corr-{uuid.uuid4().hex[:10]}",
        description="纠正 ID",
    )
    corrector_id: str = Field(description="纠正者 ID")
    corrector_role: UserRole = Field(
        default=UserRole.TEACHER, description="纠正者角色",
    )
    target_content_id: str = Field(description="被纠正内容 ID")
    target_agent_id: str = Field(
        default="", description="产出内容的 Agent ID",
    )
    original_content: str = Field(
        default="", description="原始内容",
    )
    corrected_content: str = Field(
        default="", description="纠正后内容",
    )
    correction_type: CorrectionType = Field(
        description="纠正类型",
    )
    severity: CorrectionSeverity = Field(
        default=CorrectionSeverity.MODERATE,
        description="严重程度",
    )
    feedback_text: str = Field(
        default="", description="反馈文本",
    )
    learning_value: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="学习价值评分",
    )
    applied: bool = Field(default=False, description="是否已应用")
    applied_by: str = Field(default="", description="应用者 ID")
    applied_at: float | None = Field(default=None)
    cc1_re_review_triggered: bool = Field(
        default=False, description="是否触发 CC1 重新评审",
    )
    cc1_re_review_id: str = Field(
        default="", description="CC1 重新评审 ID",
    )
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_critical(self) -> bool:
        """是否为致命级别纠正."""
        return self.severity == CorrectionSeverity.CRITICAL

    @property
    def should_trigger_cc1(self) -> bool:
        """是否应触发 CC1 重新评审.

        规则:
        - critical 严重度 → 始终触发
        - major 严重度 → 始终触发
        - moderate + (factual / conceptual) → 触发
        - 其他 → 不触发
        """
        if self.severity in (
            CorrectionSeverity.CRITICAL,
            CorrectionSeverity.MAJOR,
        ):
            return True
        if (
            self.severity == CorrectionSeverity.MODERATE
            and self.correction_type
            in (CorrectionType.FACTUAL, CorrectionType.CONCEPTUAL)
        ):
            return True
        return False

    def mark_applied(
        self,
        applied_by: str,
        trigger_cc1: bool | None = None,
        cc1_review_id: str = "",
    ) -> None:
        """标记纠正为已应用.

        Args:
            applied_by: 应用者 ID
            trigger_cc1: 是否触发 CC1 重新评审
                         (None 时自动判断 should_trigger_cc1)
            cc1_review_id: CC1 重新评审 ID
        """
        self.applied = True
        self.applied_by = applied_by
        self.applied_at = time.time()
        if trigger_cc1 is None:
            trigger_cc1 = self.should_trigger_cc1
        self.cc1_re_review_triggered = trigger_cc1
        if cc1_review_id:
            self.cc1_re_review_id = cc1_review_id


class CreativeRequest(BaseModel):
    """创意请求 (GAIA 协商协议启发).

    当人类请求 AI 提供创意/发散性输入时创建.
    作为人机共创的协商回合, 人类引导 AI 发散思维.

    请求类型:
    - brainstorm: 头脑风暴 — 生成多种可能性
    - alternative: 替代方案 — 提供不同于当前的思路
    - example: 示例生成 — 用具体示例说明
    - metaphor: 类比隐喻 — 用类比解释抽象概念

    Attributes:
        request_id: 创意请求 ID
        requester_id: 请求者 ID
        requester_role: 请求者角色
        request_type: 请求类型
        topic: 请求主题
        constraints: 约束条件列表
        desired_output_format: 期望输出格式
        context: 上下文信息
        status: 当前状态
        response_content: AI 响应内容
        responder_id: 响应 Agent ID
        responded_at: 响应时间
        response_time_seconds: 响应耗时 (秒)
        rating: 响应评分 (0-5, 由请求者评价)
        created_at: 创建时间
        metadata: 附加元数据
    """

    request_id: str = Field(
        default_factory=lambda: f"cr-{uuid.uuid4().hex[:10]}",
        description="创意请求 ID",
    )
    requester_id: str = Field(description="请求者 ID")
    requester_role: UserRole = Field(
        default=UserRole.TEACHER, description="请求者角色",
    )
    request_type: CreativeRequestType = Field(
        description="请求类型",
    )
    topic: str = Field(default="", description="请求主题")
    constraints: list[str] = Field(
        default_factory=list, description="约束条件列表",
    )
    desired_output_format: str = Field(
        default="", description="期望输出格式",
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="上下文信息",
    )
    status: InterventionEventStatus = Field(
        default=InterventionEventStatus.ACTIVE,
        description="当前状态",
    )
    response_content: str = Field(
        default="", description="AI 响应内容",
    )
    responder_id: str = Field(
        default="", description="响应 Agent ID",
    )
    responded_at: float | None = Field(default=None)
    response_time_seconds: float = Field(
        default=0.0, ge=0.0, description="响应耗时",
    )
    rating: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description="响应评分 (0-5)",
    )
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_responded(self) -> bool:
        """是否已响应."""
        return self.status == InterventionEventStatus.RESOLVED

    def respond(
        self,
        responder_id: str,
        content: str,
    ) -> None:
        """提交创意响应.

        Args:
            responder_id: 响应 Agent ID
            content: 响应内容
        """
        self.status = InterventionEventStatus.RESOLVED
        self.response_content = content
        self.responder_id = responder_id
        self.responded_at = time.time()
        self.response_time_seconds = round(
            self.responded_at - self.created_at, 3,
        )

    def cancel(self) -> None:
        """取消创意请求."""
        self.status = InterventionEventStatus.CANCELLED
        self.responded_at = time.time()


class InterventionEvent(BaseModel):
    """统一干预事件模型.

    追踪所有四种干预类型的完整生命周期, 作为审计轨迹的核心载体.
    每个干预请求都会创建一个对应的 InterventionEvent.

    NIST AI RMF 治理要求: 所有干预事件全程记录, 不可篡改,
    支撑 Govern (角色矩阵) → Map (能力边界) → Measure (度量)
    → Manage (路由) 四功能核心.

    Attributes:
        event_id: 事件 ID
        event_type: 干预类型 (四种 L4 干预类型)
        status: 事件状态
        priority: 优先级 (P0/P1/P2)
        recovery_mode: 恢复模式
        payload: 事件载荷 (存储对应的请求对象快照)
        timeline: 时间线 (关键事件序列)
        outcome: 干预结果
        metadata: 附加元数据
        related_event_ids: 关联事件 ID 列表
        created_at: 创建时间
        resolved_at: 解决时间
        duration_seconds: 持续秒数
    """

    event_id: str = Field(
        default_factory=lambda: f"intv-evt-{uuid.uuid4().hex[:10]}",
        description="事件 ID",
    )
    event_type: InterventionTypeL4 = Field(
        description="干预类型",
    )
    status: InterventionEventStatus = Field(
        default=InterventionEventStatus.INITIATED,
        description="事件状态",
    )
    priority: Priority = Field(
        default=Priority.P1, description="优先级",
    )
    recovery_mode: RecoveryMode = Field(
        default=RecoveryMode.RESUME_FROM_CHECKPOINT,
        description="恢复模式",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="事件载荷 (请求对象快照)",
    )
    timeline: list[TimelineEntry] = Field(
        default_factory=list, description="时间线",
    )
    outcome: dict[str, Any] = Field(
        default_factory=dict, description="干预结果",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    related_event_ids: list[str] = Field(
        default_factory=list, description="关联事件 ID 列表",
    )
    created_at: float = Field(default_factory=time.time)
    resolved_at: float | None = Field(default=None)
    duration_seconds: float = Field(
        default=0.0, ge=0.0, description="持续秒数",
    )

    @property
    def is_active(self) -> bool:
        """是否处于活跃状态."""
        return self.status in (
            InterventionEventStatus.INITIATED,
            InterventionEventStatus.ACTIVE,
        )

    @property
    def current_duration(self) -> float:
        """当前持续秒数 (实时计算)."""
        end = self.resolved_at if self.resolved_at else time.time()
        return round(end - self.created_at, 3)

    def add_timeline_entry(
        self,
        action: InterventionAction,
        actor: str = "",
        description: str = "",
        **metadata: Any,
    ) -> TimelineEntry:
        """添加时间线条目.

        Args:
            action: 干预动作类型
            actor: 执行者 ID
            description: 动作描述
            **metadata: 附加元数据

        Returns:
            新创建的时间线条目
        """
        entry = TimelineEntry(
            action=action,
            actor=actor,
            description=description,
            metadata=metadata,
        )
        self.timeline.append(entry)
        return entry

    def activate(self) -> None:
        """激活事件 (initiated → active)."""
        self.status = InterventionEventStatus.ACTIVE
        self.add_timeline_entry(
            InterventionAction.OVERRIDE,
            actor="system",
            description="事件激活",
        )

    def resolve(self, outcome: dict[str, Any] | None = None) -> None:
        """解决事件.

        Args:
            outcome: 干预结果字典
        """
        self.status = InterventionEventStatus.RESOLVED
        self.resolved_at = time.time()
        self.duration_seconds = round(
            self.resolved_at - self.created_at, 3,
        )
        if outcome:
            self.outcome.update(outcome)
        self.add_timeline_entry(
            InterventionAction.RESUME,
            actor="system",
            description="事件解决",
        )

    def cancel(self) -> None:
        """取消事件."""
        self.status = InterventionEventStatus.CANCELLED
        self.resolved_at = time.time()
        self.duration_seconds = round(
            self.resolved_at - self.created_at, 3,
        )
        self.add_timeline_entry(
            InterventionAction.TERMINATE,
            actor="system",
            description="事件取消",
        )


# ============================================================
# L4 干预层管理器
# ============================================================


class InterventionManager:
    """L4 干预层管理器.

    统一管理四种干预类型的完整生命周期, 提供线程安全的
    创建、查询、解决和统计能力.

    融合方案:
    - LangGraph interrupt(): 紧急暂停作为图级中断, 阻塞执行流
    - Swarm handoff(): 人工接管作为 Agent 移交, 控制权转移
    - CrewAI human_input(): 纠正反馈作为任务级人类输入
    - GAIA 协议: 创意请求作为协商回合
    - NIST AI RMF: 全审计轨迹支撑治理

    设计原则:
    - 线程安全: 所有共享状态使用 threading.RLock() 保护
    - 完整审计: 所有干预事件全程记录于 _events 审计轨迹
    - 路由引擎集成: 复用 RiskLevel / Reversibility / UserRole 等枚举
    - CC1 集成: 纠正反馈根据严重度自动触发 CC1 重新评审
    - 通知支持: 紧急暂停通过注册的回调自动通知教师
    - 恢复模式: 支持 resume_from_checkpoint / restart_from_new_state

    使用示例::

        manager = InterventionManager()

        # 1. 紧急暂停 (LangGraph interrupt)
        pause = manager.initiate_emergency_pause(
            user_id="student-001",
            reason="学生感到困惑, 请求暂停",
            scope=PauseScope.SESSION,
            agent_ids=["tutor-agent"],
        )

        # 2. 人工接管 (Swarm handoff)
        override = manager.initiate_manual_override(
            operator_id="teacher-001",
            target_agent="tutor-agent",
            override_level=OverrideLevel.EXECUTIVE,
            instructions="切换到引导式教学模式",
        )

        # 3. 纠正反馈 (CrewAI human_input)
        correction = manager.submit_correction(
            corrector_id="teacher-001",
            target_content_id="content-0042",
            original="水分子由 2 个氢原子和 1 个氧原子组成",
            corrected="水分子由 2 个氢原子和 1 个氧原子组成, 化学式 H2O",
            correction_type=CorrectionType.FACTUAL,
            feedback="补充化学式更完整",
        )
        manager.apply_correction(correction.correction_id, "teacher-001")

        # 4. 创意请求 (GAIA 协商)
        creative = manager.request_creative_input(
            requester_id="teacher-001",
            request_type=CreativeRequestType.METAPHOR,
            topic="解释量子叠加态",
            constraints=["面向高中生", "不超过 200 字"],
        )
        manager.respond_creative_request(
            creative.request_id,
            responder_id="tutor-agent",
            content="想象一枚旋转中的硬币...",
        )

        # 查询活跃干预
        active = manager.get_active_interventions()

        # 获取统计
        stats = manager.get_statistics()
    """

    #: 紧急暂停默认超时 (秒) — 超过此时间自动提醒教师
    DEFAULT_PAUSE_TIMEOUT: float = 300.0

    #: 人工接管默认持续时间 (秒)
    DEFAULT_OVERRIDE_DURATION: float = 300.0

    #: 严重程度 → 是否强制 CC1 重新评审
    _CC1_FORCE_SEVERITY: set[CorrectionSeverity] = {
        CorrectionSeverity.CRITICAL,
        CorrectionSeverity.MAJOR,
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # 四种干预请求存储 (ID → 请求对象)
        self._pauses: dict[str, EmergencyPauseRequest] = {}
        self._overrides: dict[str, ManualOverrideRequest] = {}
        self._corrections: dict[str, CorrectionFeedback] = {}
        self._creative_requests: dict[str, CreativeRequest] = {}

        # 统一审计轨迹 (event_id → InterventionEvent)
        self._events: dict[str, InterventionEvent] = {}

        # 请求 ID → 事件 ID 映射 (快速查找关联事件)
        self._request_to_event: dict[str, str] = {}

        # 通知回调列表
        self._notification_callbacks: list[NotificationCallback] = []

        # 统计计数器
        self._total_initiated = 0
        self._total_resolved = 0
        self._total_cancelled = 0

    # ==========================================================
    # 通知回调注册
    # ==========================================================

    def register_notification_callback(
        self,
        callback: NotificationCallback,
    ) -> None:
        """注册通知回调.

        当紧急暂停且 auto_notify_teacher=True 时, 所有已注册的
        回调将被调用, 传入事件载荷字典.

        Args:
            callback: 通知回调函数
        """
        with self._lock:
            self._notification_callbacks.append(callback)
            logger.info("注册通知回调, 当前共 %d 个", len(self._notification_callbacks))

    def unregister_notification_callback(
        self,
        callback: NotificationCallback,
    ) -> None:
        """注销通知回调."""
        with self._lock:
            if callback in self._notification_callbacks:
                self._notification_callbacks.remove(callback)

    def _notify(self, payload: dict[str, Any]) -> None:
        """触发通知回调 (内部方法, 不加锁以避免死锁)."""
        for callback in self._notification_callbacks:
            try:
                callback(payload)
            except Exception:
                logger.exception("通知回调执行失败")

    # ==========================================================
    # 紧急暂停 (LangGraph interrupt)
    # ==========================================================

    def initiate_emergency_pause(
        self,
        user_id: str,
        reason: str,
        scope: PauseScope = PauseScope.SESSION,
        agent_ids: list[str] | None = None,
        auto_notify_teacher: bool = True,
        risk_level: RiskLevel = RiskLevel.HIGH,
        recovery_mode: RecoveryMode = RecoveryMode.RESUME_FROM_CHECKPOINT,
    ) -> EmergencyPauseRequest:
        """发起紧急暂停 (LangGraph interrupt 启发).

        立即创建紧急暂停请求, 阻塞全部相关 Agent 执行.
        如果 auto_notify_teacher=True, 触发所有注册的通知回调.

        紧急暂停是最高优先级干预:
        - 优先级 P0
        - 响应时间 < 5 秒
        - 默认恢复模式: 从检查点恢复

        Args:
            user_id: 触发用户 ID (通常为学生)
            reason: 暂停原因
            scope: 暂停作用范围
            agent_ids: 受影响 Agent ID 列表
            auto_notify_teacher: 是否自动通知教师
            risk_level: 风险等级
            recovery_mode: 恢复模式

        Returns:
            紧急暂停请求 (状态为 ACTIVE)
        """
        with self._lock:
            pause = EmergencyPauseRequest(
                user_id=user_id,
                reason=reason,
                scope=scope,
                agent_ids=agent_ids or [],
                auto_notify_teacher=auto_notify_teacher,
                risk_level=risk_level,
                recovery_mode=recovery_mode,
                status=InterventionEventStatus.ACTIVE,
            )
            self._pauses[pause.pause_id] = pause

            # 创建审计事件
            event = InterventionEvent(
                event_type=InterventionTypeL4.EMERGENCY_PAUSE,
                status=InterventionEventStatus.ACTIVE,
                priority=Priority.P0,
                recovery_mode=recovery_mode,
                payload={
                    "pause_id": pause.pause_id,
                    "user_id": user_id,
                    "reason": reason,
                    "scope": scope.value,
                    "agent_ids": agent_ids or [],
                    "risk_level": risk_level.value,
                },
            )
            event.add_timeline_entry(
                InterventionAction.PAUSE,
                actor=user_id,
                description=f"紧急暂停: {reason}",
                scope=scope.value,
                agent_count=len(agent_ids or []),
            )
            self._events[event.event_id] = event
            self._request_to_event[pause.pause_id] = event.event_id
            self._total_initiated += 1

            logger.warning(
                "紧急暂停: pause=%s user=%s scope=%s reason=%s",
                pause.pause_id, user_id, scope.value, reason,
            )

            # 自动通知教师
            if auto_notify_teacher:
                notify_payload = {
                    "type": "emergency_pause",
                    "pause_id": pause.pause_id,
                    "user_id": user_id,
                    "reason": reason,
                    "scope": scope.value,
                    "agent_ids": agent_ids or [],
                    "risk_level": risk_level.value,
                    "timestamp": pause.created_at,
                    "event_id": event.event_id,
                }
                pause.notify_payload = notify_payload
                # 通知回调在锁外执行, 避免死锁
                callbacks = list(self._notification_callbacks)

            # 如果无通知需求, callbacks 为空列表
            if not auto_notify_teacher:
                callbacks = []

        # 锁外执行通知回调
        for cb in callbacks:
            try:
                cb(notify_payload)  # type: ignore[possibly-undefined]
            except Exception:
                logger.exception("通知回调执行失败")

        return pause

    def resolve_emergency_pause(
        self,
        pause_id: str,
        resolution: str,
        resolved_by: str,
    ) -> EmergencyPauseRequest:
        """解决紧急暂停.

        解除暂停状态, 恢复 Agent 执行. 根据恢复模式:
        - resume_from_checkpoint: 从暂停点恢复
        - restart_from_new_state: 从新状态重启

        Args:
            pause_id: 暂停请求 ID
            resolution: 解决描述
            resolved_by: 解决者 ID

        Returns:
            已解决的紧急暂停请求

        Raises:
            KeyError: 暂停请求不存在
            ValueError: 暂停请求已解决或已取消
        """
        with self._lock:
            pause = self._pauses.get(pause_id)
            if pause is None:
                raise KeyError(f"紧急暂停请求不存在: {pause_id}")
            if not pause.is_active:
                raise ValueError(
                    f"紧急暂停请求已处理: {pause_id}, "
                    f"当前状态: {pause.status.value}"
                )

            pause.resolve(resolution, resolved_by)

            # 更新审计事件
            event_id = self._request_to_event.get(pause_id)
            if event_id:
                event = self._events.get(event_id)
                if event:
                    event.resolve(
                        outcome={
                            "resolution": resolution,
                            "resolved_by": resolved_by,
                            "duration_seconds": pause.duration_seconds,
                            "recovery_mode": pause.recovery_mode.value,
                        },
                    )

            self._total_resolved += 1
            logger.info(
                "紧急暂停解决: pause=%s resolved_by=%s duration=%.1fs",
                pause_id, resolved_by, pause.duration_seconds,
            )
            return pause

    # ==========================================================
    # 人工接管 (Swarm handoff)
    # ==========================================================

    def initiate_manual_override(
        self,
        operator_id: str,
        target_agent: str,
        override_level: OverrideLevel = OverrideLevel.EXECUTIVE,
        instructions: str = "",
        duration_seconds: float | None = None,
        context: dict[str, Any] | None = None,
        operator_role: UserRole = UserRole.TEACHER,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        recovery_mode: RecoveryMode = RecoveryMode.RESUME_FROM_CHECKPOINT,
    ) -> ManualOverrideRequest:
        """发起人工接管 (Swarm handoff 启发).

        人类操作者从 AI Agent 接管控制权. 接管期间:
        - advisory: Agent 接收指导但保持自主
        - executive: Agent 遵照人类指令执行
        - absolute: Agent 完全暂停, 人类直接操作

        Args:
            operator_id: 操作者 ID
            target_agent: 被接管 Agent ID
            override_level: 接管级别
            instructions: 接管指令
            duration_seconds: 接管持续时间 (秒), None=默认值, 0=无限
            context: 上下文信息
            operator_role: 操作者角色
            risk_level: 风险等级
            recovery_mode: 恢复模式

        Returns:
            人工接管请求 (状态为 ACTIVE)
        """
        with self._lock:
            if duration_seconds is None:
                duration_seconds = self.DEFAULT_OVERRIDE_DURATION

            override = ManualOverrideRequest(
                operator_id=operator_id,
                operator_role=operator_role,
                target_agent=target_agent,
                override_level=override_level,
                duration_seconds=duration_seconds,
                instructions=instructions,
                context=context or {},
                risk_level=risk_level,
                recovery_mode=recovery_mode,
                status=InterventionEventStatus.ACTIVE,
            )
            self._overrides[override.override_id] = override

            # 确定优先级
            priority = Priority.P0 if override_level == OverrideLevel.ABSOLUTE else Priority.P1

            # 创建审计事件
            event = InterventionEvent(
                event_type=InterventionTypeL4.MANUAL_OVERRIDE,
                status=InterventionEventStatus.ACTIVE,
                priority=priority,
                recovery_mode=recovery_mode,
                payload={
                    "override_id": override.override_id,
                    "operator_id": operator_id,
                    "target_agent": target_agent,
                    "override_level": override_level.value,
                    "instructions": instructions,
                    "duration_seconds": duration_seconds,
                },
            )
            event.add_timeline_entry(
                InterventionAction.OVERRIDE,
                actor=operator_id,
                description=f"人工接管: {target_agent} ({override_level.value})",
                target_agent=target_agent,
                override_level=override_level.value,
            )
            self._events[event.event_id] = event
            self._request_to_event[override.override_id] = event.event_id
            self._total_initiated += 1

            logger.info(
                "人工接管: override=%s operator=%s target=%s level=%s",
                override.override_id, operator_id,
                target_agent, override_level.value,
            )
            return override

    def release_override(
        self,
        override_id: str,
        summary: str = "",
        released_by: str = "",
    ) -> ManualOverrideRequest:
        """释放人工接管.

        将控制权交还给 Agent, 恢复自主执行.

        Args:
            override_id: 接管请求 ID
            summary: 接管总结
            released_by: 释放者 ID

        Returns:
            已释放的人工接管请求

        Raises:
            KeyError: 接管请求不存在
            ValueError: 接管请求已释放或已取消
        """
        with self._lock:
            override = self._overrides.get(override_id)
            if override is None:
                raise KeyError(f"人工接管请求不存在: {override_id}")
            if not override.is_active:
                raise ValueError(
                    f"人工接管请求已处理: {override_id}, "
                    f"当前状态: {override.status.value}"
                )

            override.release(summary=summary, released_by=released_by)

            # 更新审计事件
            event_id = self._request_to_event.get(override_id)
            if event_id:
                event = self._events.get(event_id)
                if event:
                    event.resolve(
                        outcome={
                            "summary": summary,
                            "released_by": released_by,
                            "actual_duration": override.actual_duration_seconds,
                            "recovery_mode": override.recovery_mode.value,
                        },
                    )

            self._total_resolved += 1
            logger.info(
                "人工接管释放: override=%s released_by=%s duration=%.1fs",
                override_id, released_by, override.actual_duration_seconds,
            )
            return override

    # ==========================================================
    # 纠正反馈 (CrewAI human_input)
    # ==========================================================

    def submit_correction(
        self,
        corrector_id: str,
        target_content_id: str,
        original: str,
        corrected: str,
        correction_type: CorrectionType,
        feedback: str = "",
        severity: CorrectionSeverity = CorrectionSeverity.MODERATE,
        target_agent_id: str = "",
        corrector_role: UserRole = UserRole.TEACHER,
        learning_value: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionFeedback:
        """提交纠正反馈 (CrewAI human_input 启发).

        人类纠正 AI 输出并提供反馈. 纠正创建后处于待应用状态,
        需调用 apply_correction 应用并可能触发 CC1 重新评审.

        CC1 联动规则:
        - critical / major → 强制触发 CC1 重新评审
        - moderate + (factual / conceptual) → 触发 CC1 重新评审
        - 其他 → 不触发

        Args:
            corrector_id: 纠正者 ID
            target_content_id: 被纠正内容 ID
            original: 原始内容
            corrected: 纠正后内容
            correction_type: 纠正类型
            feedback: 反馈文本
            severity: 严重程度
            target_agent_id: 产出内容的 Agent ID
            corrector_role: 纠正者角色
            learning_value: 学习价值评分 (0-1)
            metadata: 附加元数据

        Returns:
            纠正反馈对象 (状态为待应用)
        """
        with self._lock:
            correction = CorrectionFeedback(
                corrector_id=corrector_id,
                corrector_role=corrector_role,
                target_content_id=target_content_id,
                target_agent_id=target_agent_id,
                original_content=original,
                corrected_content=corrected,
                correction_type=correction_type,
                severity=severity,
                feedback_text=feedback,
                learning_value=learning_value,
                metadata=metadata or {},
            )
            self._corrections[correction.correction_id] = correction

            # 确定优先级
            if severity == CorrectionSeverity.CRITICAL:
                priority = Priority.P0
            elif severity == CorrectionSeverity.MAJOR:
                priority = Priority.P0
            elif severity == CorrectionSeverity.MODERATE:
                priority = Priority.P1
            else:
                priority = Priority.P2

            # 创建审计事件
            event = InterventionEvent(
                event_type=InterventionTypeL4.CORRECTION_FEEDBACK,
                status=InterventionEventStatus.INITIATED,
                priority=priority,
                payload={
                    "correction_id": correction.correction_id,
                    "corrector_id": corrector_id,
                    "target_content_id": target_content_id,
                    "target_agent_id": target_agent_id,
                    "correction_type": correction_type.value,
                    "severity": severity.value,
                    "should_trigger_cc1": correction.should_trigger_cc1,
                },
            )
            event.add_timeline_entry(
                InterventionAction.CORRECT,
                actor=corrector_id,
                description=(
                    f"纠正提交: type={correction_type.value}, "
                    f"severity={severity.value}"
                ),
                target_content_id=target_content_id,
                correction_type=correction_type.value,
            )
            self._events[event.event_id] = event
            self._request_to_event[correction.correction_id] = event.event_id
            self._total_initiated += 1

            logger.info(
                "纠正反馈: correction=%s corrector=%s type=%s severity=%s",
                correction.correction_id, corrector_id,
                correction_type.value, severity.value,
            )
            return correction

    def apply_correction(
        self,
        correction_id: str,
        applied_by: str,
        trigger_cc1: bool | None = None,
        cc1_review_id: str = "",
    ) -> CorrectionFeedback:
        """应用纠正反馈.

        将纠正标记为已应用, 并根据严重度决定是否触发 CC1 重新评审.

        CC1 重新评审触发逻辑:
        - trigger_cc1=None: 自动判断 (should_trigger_cc1)
        - trigger_cc1=True: 强制触发
        - trigger_cc1=False: 强制不触发

        Args:
            correction_id: 纠正 ID
            applied_by: 应用者 ID
            trigger_cc1: 是否触发 CC1 重新评审 (None=自动判断)
            cc1_review_id: CC1 重新评审 ID (如已创建)

        Returns:
            已应用的纠正反馈对象

        Raises:
            KeyError: 纠正不存在
            ValueError: 纠正已应用
        """
        with self._lock:
            correction = self._corrections.get(correction_id)
            if correction is None:
                raise KeyError(f"纠正不存在: {correction_id}")
            if correction.applied:
                raise ValueError(f"纠正已应用: {correction_id}")

            # 判断是否触发 CC1
            if trigger_cc1 is None:
                trigger_cc1 = correction.should_trigger_cc1

            correction.mark_applied(
                applied_by=applied_by,
                trigger_cc1=trigger_cc1,
                cc1_review_id=cc1_review_id,
            )

            # 更新审计事件
            event_id = self._request_to_event.get(correction_id)
            if event_id:
                event = self._events.get(event_id)
                if event:
                    event.resolve(
                        outcome={
                            "applied_by": applied_by,
                            "cc1_re_review_triggered": trigger_cc1,
                            "cc1_re_review_id": cc1_review_id,
                            "learning_value": correction.learning_value,
                        },
                    )
                    event.add_timeline_entry(
                        InterventionAction.RESUME,
                        actor=applied_by,
                        description=f"纠正已应用, CC1 重新评审: {trigger_cc1}",
                        applied=True,
                        cc1_triggered=trigger_cc1,
                    )

            self._total_resolved += 1
            logger.info(
                "纠正应用: correction=%s applied_by=%s cc1_triggered=%s",
                correction_id, applied_by, trigger_cc1,
            )
            return correction

    # ==========================================================
    # 创意请求 (GAIA 协商协议)
    # ==========================================================

    def request_creative_input(
        self,
        requester_id: str,
        request_type: CreativeRequestType,
        topic: str,
        constraints: list[str] | None = None,
        desired_output_format: str = "",
        context: dict[str, Any] | None = None,
        requester_role: UserRole = UserRole.TEACHER,
        metadata: dict[str, Any] | None = None,
    ) -> CreativeRequest:
        """发起创意请求 (GAIA 协商协议启发).

        人类请求 AI 提供创意/发散性输入, 作为人机共创的协商回合.

        Args:
            requester_id: 请求者 ID
            request_type: 请求类型 (brainstorm/alternative/example/metaphor)
            topic: 请求主题
            constraints: 约束条件列表
            desired_output_format: 期望输出格式
            context: 上下文信息
            requester_role: 请求者角色
            metadata: 附加元数据

        Returns:
            创意请求对象 (状态为 ACTIVE, 等待 AI 响应)
        """
        with self._lock:
            request = CreativeRequest(
                requester_id=requester_id,
                requester_role=requester_role,
                request_type=request_type,
                topic=topic,
                constraints=constraints or [],
                desired_output_format=desired_output_format,
                context=context or {},
                status=InterventionEventStatus.ACTIVE,
                metadata=metadata or {},
            )
            self._creative_requests[request.request_id] = request

            # 创建审计事件
            event = InterventionEvent(
                event_type=InterventionTypeL4.CREATIVE_REQUEST,
                status=InterventionEventStatus.ACTIVE,
                priority=Priority.P2,
                payload={
                    "request_id": request.request_id,
                    "requester_id": requester_id,
                    "request_type": request_type.value,
                    "topic": topic,
                    "constraints": constraints or [],
                },
            )
            event.add_timeline_entry(
                InterventionAction.REDIRECT,
                actor=requester_id,
                description=f"创意请求: {request_type.value} - {topic}",
                request_type=request_type.value,
                topic=topic,
            )
            self._events[event.event_id] = event
            self._request_to_event[request.request_id] = event.event_id
            self._total_initiated += 1

            logger.info(
                "创意请求: request=%s requester=%s type=%s topic=%s",
                request.request_id, requester_id,
                request_type.value, topic,
            )
            return request

    def respond_creative_request(
        self,
        request_id: str,
        responder_id: str,
        content: str,
    ) -> CreativeRequest:
        """响应创意请求.

        AI Agent 提交对创意请求的响应, 完成协商回合.

        Args:
            request_id: 创意请求 ID
            responder_id: 响应 Agent ID
            content: 响应内容

        Returns:
            已响应的创意请求对象

        Raises:
            KeyError: 创意请求不存在
            ValueError: 创意请求已响应或已取消
        """
        with self._lock:
            request = self._creative_requests.get(request_id)
            if request is None:
                raise KeyError(f"创意请求不存在: {request_id}")
            if request.status != InterventionEventStatus.ACTIVE:
                raise ValueError(
                    f"创意请求已处理: {request_id}, "
                    f"当前状态: {request.status.value}"
                )

            request.respond(responder_id=responder_id, content=content)

            # 更新审计事件
            event_id = self._request_to_event.get(request_id)
            if event_id:
                event = self._events.get(event_id)
                if event:
                    event.resolve(
                        outcome={
                            "responder_id": responder_id,
                            "response_time_seconds": request.response_time_seconds,
                            "content_length": len(content),
                        },
                    )

            self._total_resolved += 1
            logger.info(
                "创意响应: request=%s responder=%s response_time=%.1fs",
                request_id, responder_id, request.response_time_seconds,
            )
            return request

    # ==========================================================
    # 查询接口
    # ==========================================================

    def get_emergency_pause(self, pause_id: str) -> EmergencyPauseRequest | None:
        """获取紧急暂停请求."""
        with self._lock:
            return self._pauses.get(pause_id)

    def get_manual_override(self, override_id: str) -> ManualOverrideRequest | None:
        """获取人工接管请求."""
        with self._lock:
            return self._overrides.get(override_id)

    def get_correction(self, correction_id: str) -> CorrectionFeedback | None:
        """获取纠正反馈."""
        with self._lock:
            return self._corrections.get(correction_id)

    def get_creative_request(self, request_id: str) -> CreativeRequest | None:
        """获取创意请求."""
        with self._lock:
            return self._creative_requests.get(request_id)

    def get_event(self, event_id: str) -> InterventionEvent | None:
        """获取干预事件."""
        with self._lock:
            return self._events.get(event_id)

    def get_active_interventions(self) -> list[InterventionEvent]:
        """获取所有活跃干预事件.

        返回状态为 INITIATED 或 ACTIVE 的事件, 按创建时间降序排列.

        Returns:
            活跃干预事件列表
        """
        with self._lock:
            active = [
                e for e in self._events.values()
                if e.is_active
            ]
        active.sort(key=lambda e: e.created_at, reverse=True)
        return active

    def get_intervention_history(
        self,
        limit: int = 100,
        event_type: InterventionTypeL4 | None = None,
    ) -> list[InterventionEvent]:
        """获取干预历史记录.

        返回所有干预事件 (含已解决和已取消), 按创建时间降序排列.

        Args:
            limit: 返回最大数量
            event_type: 过滤干预类型 (None=全部)

        Returns:
            干预事件列表
        """
        with self._lock:
            events = list(self._events.values())

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        events.sort(key=lambda e: e.created_at, reverse=True)
        return events[:limit]

    # ==========================================================
    # 统计
    # ==========================================================

    def get_statistics(self) -> dict[str, Any]:
        """获取干预统计信息.

        返回按类型、状态分组的计数, 以及平均解决时间等指标.

        Returns:
            统计信息字典, 包含:
            - total_events: 总事件数
            - active_count: 活跃事件数
            - by_type: 按类型分组计数
            - by_status: 按状态分组计数
            - avg_resolution_time: 平均解决时间 (秒)
            - total_initiated / resolved / cancelled: 累计计数
            - emergency_pauses / manual_overrides / corrections / creative_requests: 各类型详细统计
        """
        with self._lock:
            events = list(self._events.values())

        total = len(events)
        if total == 0:
            return {
                "total_events": 0,
                "active_count": 0,
                "by_type": {},
                "by_status": {},
                "avg_resolution_time": 0.0,
                "total_initiated": 0,
                "total_resolved": 0,
                "total_cancelled": 0,
            }

        # 按类型分组
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        resolution_times: list[float] = []

        for event in events:
            type_key = event.event_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            status_key = event.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            if event.duration_seconds > 0:
                resolution_times.append(event.duration_seconds)

        active_count = sum(
            1 for e in events if e.is_active
        )
        avg_resolution = (
            sum(resolution_times) / len(resolution_times)
            if resolution_times else 0.0
        )

        # 各类型详细统计
        with self._lock:
            pause_stats = self._compute_type_stats(
                list(self._pauses.values()),
            )
            override_stats = self._compute_type_stats(
                list(self._overrides.values()),
            )
            correction_stats = self._compute_correction_stats(
                list(self._corrections.values()),
            )
            creative_stats = self._compute_creative_stats(
                list(self._creative_requests.values()),
            )

        return {
            "total_events": total,
            "active_count": active_count,
            "by_type": by_type,
            "by_status": by_status,
            "avg_resolution_time": round(avg_resolution, 3),
            "total_initiated": self._total_initiated,
            "total_resolved": self._total_resolved,
            "total_cancelled": self._total_cancelled,
            "emergency_pauses": pause_stats,
            "manual_overrides": override_stats,
            "corrections": correction_stats,
            "creative_requests": creative_stats,
        }

    def _compute_type_stats(
        self,
        items: list[Any],
    ) -> dict[str, Any]:
        """计算通用类型统计."""
        total = len(items)
        if total == 0:
            return {"total": 0}

        active = sum(1 for i in items if i.is_active)
        resolved = sum(
            1 for i in items
            if i.status == InterventionEventStatus.RESOLVED
        )
        cancelled = sum(
            1 for i in items
            if i.status == InterventionEventStatus.CANCELLED
        )

        durations = [
            i.duration_seconds
            for i in items
            if hasattr(i, "duration_seconds")
            and i.duration_seconds > 0
        ] + [
            i.actual_duration_seconds
            for i in items
            if hasattr(i, "actual_duration_seconds")
            and i.actual_duration_seconds > 0
        ]
        avg_duration = (
            sum(durations) / len(durations) if durations else 0.0
        )

        return {
            "total": total,
            "active": active,
            "resolved": resolved,
            "cancelled": cancelled,
            "avg_duration": round(avg_duration, 3),
        }

    def _compute_correction_stats(
        self,
        corrections: list[CorrectionFeedback],
    ) -> dict[str, Any]:
        """计算纠正反馈详细统计."""
        total = len(corrections)
        if total == 0:
            return {"total": 0}

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        applied_count = 0
        cc1_triggered = 0
        total_learning_value = 0.0

        for c in corrections:
            type_key = c.correction_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            sev_key = c.severity.value
            by_severity[sev_key] = by_severity.get(sev_key, 0) + 1

            if c.applied:
                applied_count += 1
            if c.cc1_re_review_triggered:
                cc1_triggered += 1
            total_learning_value += c.learning_value

        return {
            "total": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "applied": applied_count,
            "cc1_re_review_triggered": cc1_triggered,
            "avg_learning_value": round(
                total_learning_value / total, 3,
            ),
        }

    def _compute_creative_stats(
        self,
        requests: list[CreativeRequest],
    ) -> dict[str, Any]:
        """计算创意请求详细统计."""
        total = len(requests)
        if total == 0:
            return {"total": 0}

        by_type: dict[str, int] = {}
        responded = 0
        response_times: list[float] = []
        ratings: list[float] = []

        for r in requests:
            type_key = r.request_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            if r.is_responded:
                responded += 1
                if r.response_time_seconds > 0:
                    response_times.append(r.response_time_seconds)
            if r.rating > 0:
                ratings.append(r.rating)

        avg_response = (
            sum(response_times) / len(response_times)
            if response_times else 0.0
        )
        avg_rating = (
            sum(ratings) / len(ratings) if ratings else 0.0
        )

        return {
            "total": total,
            "by_type": by_type,
            "responded": responded,
            "avg_response_time": round(avg_response, 3),
            "avg_rating": round(avg_rating, 2),
        }

    # ==========================================================
    # 清理
    # ==========================================================

    def clear(self) -> None:
        """清空所有干预数据.

        清除全部干预请求、审计事件和统计计数.
        通知回调不受影响 (保持注册).
        """
        with self._lock:
            self._pauses.clear()
            self._overrides.clear()
            self._corrections.clear()
            self._creative_requests.clear()
            self._events.clear()
            self._request_to_event.clear()
            self._total_initiated = 0
            self._total_resolved = 0
            self._total_cancelled = 0
            logger.info("干预管理器数据已清空")
