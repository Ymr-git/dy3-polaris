"""L1 用户域人机协同 (Human-in-the-Loop, HiTL) — 核心引擎.

设计依据:
- L1 设计文档第四章 4.1-4.5: 四类协同场景、置信度门控、紧急干预、交互模式、反馈回路
- L1 设计文档第八章 8.4: JSON-RPC 接口契约 (approval/feedback)
- L1 设计文档第二章 2.2: HITL_CONFIRM 权限

融合世界先进方案:
- LangGraph Human-in-the-Loop: interrupt_before / interrupt_after 节点级中断
- OpenAI Constitutional AI: 多维度置信度评估 + 自我修正循环
- Anthropic Claude 置信度校准: 三级门控 (PASS/WARNING/BLOCK)
- DeepMind AlphaFold pLDDT: 置信度驱动的渐进式呈现
- Duolingo 学情信号: 认知负荷动态监测 + 挫败感检测
- Khan Academy 教师仪表盘: 紧急干预通知 + 升级机制
- Google PAIR 指南: 人机协作模式分类 (被动/主动/强制/可选)
- Microsoft Guidelines for Human-AI Interaction: 交互模式推荐引擎
- Stanford HAI: 反馈闭环 (feedback loop) 分类路由
- ROS (Robot Operating System) 安全模式: 紧急停止 + 降级运行

模块组成:
1. 异常体系: L1HiTLError 层级 (JSON-RPC -32400 范围)
2. 置信度门控: ConfidenceGate (PASS/WARNING/BLOCK 三级)
3. 紧急检测: EmergencyDetector (认知负荷/连续错误/异常速度/BKT偏差)
4. 反馈回路: FeedbackLoop (提交/分类/路由/历史)
5. 交互模式: InteractionMode (被动/主动/强制/可选)
6. 核心管理器: HiTLManager (四类协同场景编排)
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dy3_polaris.l1.models import (
    AlertType,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    ConfidenceGateResult,
    EmergencyAlert,
    FeedbackCategory,
    FeedbackReport,
    FeedbackType,
    HiTLPriority,
    HiTLType,
    BLOCK_THRESHOLD,
    WARNING_THRESHOLD,
    EMERGENCY_THRESHOLD,
    CONSECUTIVE_ERROR_THRESHOLD,
    FAST_ANSWER_THRESHOLD_MS,
    BKT_DEVIATION_THRESHOLD,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 1. 常量定义
# ============================================================

# --- 审批超时 ---
APPROVAL_TIMEOUT_SECONDS: int = 300        # 审批默认超时: 5 分钟
APPROVAL_TIMEOUT_MS: int = 300 * 1000      # 审批默认超时 (毫秒)

# --- 纠错型自纠限制 ---
MAX_CORRECTION_RETRIES: int = 3            # 最大自纠次数, 超过后升级教师

# --- 紧急响应 ---
EMERGENCY_RESPONSE_MS: int = 2000          # 紧急响应时间 (< 2 秒)
EMERGENCY_COOLDOWN_MS: int = 30_000        # 紧急冷却期 (30 秒内不重复触发)

# --- 反馈历史 ---
FEEDBACK_HISTORY_LIMIT: int = 100          # 单会话反馈历史上限

# --- 警报历史 ---
ALERT_HISTORY_LIMIT: int = 50              # 警报历史上限


# ============================================================
# 2. 异常体系 (JSON-RPC -32400 范围)
# ============================================================


class L1HiTLError(L6Error):
    """L1 人机协同层基础异常 (JSON-RPC -32400).

    所有 HiTL 相关异常的基类, 继承自 L6Error.
    """

    def __init__(
        self,
        code: str = "L1_HITL_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32400


class ConfidenceGateError(L1HiTLError):
    """置信度门控错误 (JSON-RPC -32401).

    置信度值非法 (超出 [0.0, 1.0] 范围) 等.
    """

    def __init__(
        self,
        detail: str = "置信度门控评估失败",
        confidence: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if confidence is not None:
            ctx["confidence"] = confidence
        if context:
            ctx.update(context)
        super().__init__("CONFIDENCE_GATE_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32401


class ApprovalError(L1HiTLError):
    """审批错误 (JSON-RPC -32402).

    请求不存在、已处理、超时等.
    """

    def __init__(
        self,
        detail: str = "审批处理失败",
        request_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if request_id:
            ctx["request_id"] = request_id
        if context:
            ctx.update(context)
        super().__init__("APPROVAL_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32402


class FeedbackError(L1HiTLError):
    """反馈错误 (JSON-RPC -32403).

    反馈报告非法、分类失败、路由失败等.
    """

    def __init__(
        self,
        detail: str = "反馈处理失败",
        report_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if report_id:
            ctx["report_id"] = report_id
        if context:
            ctx.update(context)
        super().__init__("FEEDBACK_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32403


class EmergencyError(L1HiTLError):
    """紧急干预错误 (JSON-RPC -32404).

    紧急检测参数非法、警报解决失败等.
    """

    def __init__(
        self,
        detail: str = "紧急干预处理失败",
        alert_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if alert_id:
            ctx["alert_id"] = alert_id
        if context:
            ctx.update(context)
        super().__init__("EMERGENCY_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32404


# ============================================================
# 3. 交互模式 (InteractionMode)
# ============================================================


class InteractionMode(str, Enum):
    """人机交互模式 (设计文档 4.3, 融合 Google PAIR 指南).

    四种交互模式覆盖不同协同场景:
    - PASSIVE_CONFIRMATION: 被动确认 — 内容底部附加确认控件, 用户自主确认
    - PROACTIVE_SUGGESTION: 主动建议 — 检测学习瓶颈时主动弹出建议
    - MANDATORY_BLOCK: 强制阻断 — 紧急情况自动暂停, 必须人工干预
    - OPTIONAL_NEGOTIATION: 可选协商 — 教师与系统双向协商, 可选择性采纳
    """

    PASSIVE_CONFIRMATION = "passive_confirmation"
    PROACTIVE_SUGGESTION = "proactive_suggestion"
    MANDATORY_BLOCK = "mandatory_block"
    OPTIONAL_NEGOTIATION = "optional_negotiation"


# ============================================================
# 4. 置信度门控 (ConfidenceGate)
# ============================================================


class GateDecision(str, Enum):
    """门控决策动作 (设计文档 4.4).

    决定 Agent 输出的呈现策略:
    - PRESENT: 直接呈现 (PASS)
    - PRESENT_WITH_LABEL: 附标签呈现 (WARNING)
    - HOLD_FOR_REVIEW: 暂停审核 (BLOCK)
    """

    PRESENT = "present"
    PRESENT_WITH_LABEL = "present_with_label"
    HOLD_FOR_REVIEW = "hold_for_review"


@dataclass
class GateResult:
    """置信度门控评估结果.

    借鉴 Anthropic Claude 置信度校准: 三级门控 + 交互模式推荐.

    Attributes:
        confidence: 评估的置信度值
        gate_result: 门控结果 (PASS/WARNING/BLOCK)
        decision: 决策动作 (PRESENT/PRESENT_WITH_LABEL/HOLD_FOR_REVIEW)
        recommended_mode: 推荐交互模式
        provenance: 来源追踪 (artifact_id, agent_id, timestamp)
        evaluated_at: 评估时间戳 (毫秒)
    """

    confidence: float
    gate_result: ConfidenceGateResult
    decision: GateDecision
    recommended_mode: InteractionMode
    provenance: dict[str, Any] = field(default_factory=dict)
    evaluated_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "confidence": self.confidence,
            "gate_result": self.gate_result.value,
            "decision": self.decision.value,
            "recommended_mode": self.recommended_mode.value,
            "provenance": self.provenance,
            "evaluated_at": self.evaluated_at,
        }


class ConfidenceGate:
    """置信度门控引擎 (设计文档 4.4).

    借鉴世界先进方案:
    - Anthropic Claude: 三级门控 (PASS >= 0.85 / WARNING >= 0.4 / BLOCK < 0.4)
    - DeepMind AlphaFold pLDDT: 置信度驱动的渐进式呈现
    - OpenAI Constitutional AI: 多维度置信度评估

    门控逻辑:
    - confidence >= WARNING_THRESHOLD (0.85) → PASS → 直接呈现
    - BLOCK_THRESHOLD (0.4) <= confidence < WARNING_THRESHOLD → WARNING → 附标签
    - confidence < BLOCK_THRESHOLD → BLOCK → 暂停审核

    线程安全: 无共享状态, 纯计算方法.
    """

    def evaluate(
        self,
        confidence: float,
        artifact_id: str | None = None,
        agent_id: str | None = None,
    ) -> GateResult:
        """评估置信度并返回门控结果.

        Args:
            confidence: Agent 输出置信度 [0.0, 1.0]
            artifact_id: 关联产出物 ID (可选, 用于溯源)
            agent_id: 生成 Agent ID (可选, 用于溯源)

        Returns:
            GateResult 包含门控结果、决策动作、推荐交互模式

        Raises:
            ConfidenceGateError: 置信度超出 [0.0, 1.0] 范围
        """
        # 参数校验
        if confidence < 0.0 or confidence > 1.0:
            raise ConfidenceGateError(
                detail=f"置信度必须在 [0.0, 1.0] 范围内, 实际值: {confidence}",
                confidence=confidence,
            )

        # 三级门控评估
        gate_result = ConfidenceGateResult.evaluate(confidence)

        # 决策动作映射
        decision = self._map_decision(gate_result)

        # 推荐交互模式
        recommended_mode = self._map_interaction_mode(gate_result)

        # 来源追踪 (Provenance)
        provenance: dict[str, Any] = {
            "timestamp": int(time.time() * 1000),
        }
        if artifact_id is not None:
            provenance["artifact_id"] = artifact_id
        if agent_id is not None:
            provenance["agent_id"] = agent_id

        return GateResult(
            confidence=confidence,
            gate_result=gate_result,
            decision=decision,
            recommended_mode=recommended_mode,
            provenance=provenance,
        )

    @staticmethod
    def _map_decision(gate_result: ConfidenceGateResult) -> GateDecision:
        """映射门控结果到决策动作."""
        if gate_result == ConfidenceGateResult.PASS:
            return GateDecision.PRESENT
        elif gate_result == ConfidenceGateResult.WARNING:
            return GateDecision.PRESENT_WITH_LABEL
        else:
            return GateDecision.HOLD_FOR_REVIEW

    @staticmethod
    def _map_interaction_mode(
        gate_result: ConfidenceGateResult,
    ) -> InteractionMode:
        """映射门控结果到推荐交互模式.

        借鉴 Microsoft Guidelines for Human-AI Interaction:
        - PASS → 被动确认 (用户自主确认)
        - WARNING → 主动建议 (系统主动提示)
        - BLOCK → 强制阻断 (必须人工干预)
        """
        if gate_result == ConfidenceGateResult.PASS:
            return InteractionMode.PASSIVE_CONFIRMATION
        elif gate_result == ConfidenceGateResult.WARNING:
            return InteractionMode.PROACTIVE_SUGGESTION
        else:
            return InteractionMode.MANDATORY_BLOCK


# ============================================================
# 5. 紧急干预检测 (EmergencyDetector)
# ============================================================


class EmergencyDetector:
    """紧急干预检测器 (设计文档 4.2, 4.3).

    借鉴世界先进方案:
    - Duolingo 学情信号: 认知负荷动态监测 + 挫败感检测
    - Khan Academy 教师仪表盘: 紧急干预通知
    - ROS 安全模式: 紧急停止 + 降级运行
    - Affective Computing (MIT Media Lab): 情绪状态识别

    检测条件 (任一满足即触发, 按严重度排序):
    1. 认知负荷 >= EMERGENCY_THRESHOLD (0.95) → HIGH_COGNITIVE_LOAD (最严重)
    2. 连续错误 >= CONSECUTIVE_ERROR_THRESHOLD (10) → CONSECUTIVE_ERRORS
    3. 平均答题时间 < FAST_ANSWER_THRESHOLD_MS (5000ms) → FAST_ANSWERING
    4. BKT 预测偏差 > BKT_DEVIATION_THRESHOLD (0.3) → BKT_DEVIATION (纠错型, 非紧急)

    线程安全: threading.RLock 保护警报存储.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._alerts: list[EmergencyAlert] = []

    def check(
        self,
        session_id: str,
        user_id: str,
        cognitive_load: float,
        consecutive_errors: int,
        avg_answer_time_ms: int,
        bkt_deviation: float = 0.0,
    ) -> EmergencyAlert | None:
        """检测紧急情况并生成警报.

        多条件同时满足时, 按严重度优先返回最严重的警报.
        严重度排序: HIGH_COGNITIVE_LOAD > CONSECUTIVE_ERRORS > FAST_ANSWERING > BKT_DEVIATION

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            cognitive_load: 当前认知负荷 [0.0, 1.0]
            consecutive_errors: 连续错误次数
            avg_answer_time_ms: 平均答题时间 (毫秒)
            bkt_deviation: BKT 预测偏差 [0.0, 1.0]

        Returns:
            EmergencyAlert 如果触发, 否则 None

        Raises:
            EmergencyError: 认知负荷值非法
        """
        # 参数校验
        if cognitive_load < 0.0 or cognitive_load > 1.0:
            raise EmergencyError(
                detail=f"认知负荷必须在 [0.0, 1.0] 范围内, 实际值: {cognitive_load}",
            )

        # 按严重度顺序检测
        alert = self._detect(
            session_id,
            user_id,
            cognitive_load,
            consecutive_errors,
            avg_answer_time_ms,
            bkt_deviation,
        )

        if alert is not None:
            with self._lock:
                self._alerts.append(alert)

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """解决紧急警报.

        Args:
            alert_id: 警报 ID

        Returns:
            True 如果成功解决, False 如果警报不存在
        """
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.is_resolved = True
                    return True
            return False

    def get_active_alerts(self) -> list[EmergencyAlert]:
        """获取所有未解决的活跃警报.

        Returns:
            活跃警报列表 (防御性拷贝)
        """
        with self._lock:
            return [a for a in self._alerts if not a.is_resolved]

    def get_alert_by_id(self, alert_id: str) -> EmergencyAlert | None:
        """按 ID 获取警报."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    return alert
            return None

    def get_all_alerts(self) -> list[EmergencyAlert]:
        """获取所有警报 (含已解决)."""
        with self._lock:
            return list(self._alerts)

    def _detect(
        self,
        session_id: str,
        user_id: str,
        cognitive_load: float,
        consecutive_errors: int,
        avg_answer_time_ms: int,
        bkt_deviation: float,
    ) -> EmergencyAlert | None:
        """执行检测逻辑, 按严重度排序返回最严重的警报."""

        # 1. 认知负荷过高 (最严重, P0)
        if cognitive_load >= EMERGENCY_THRESHOLD:
            return EmergencyAlert(
                session_id=session_id,
                user_id=user_id,
                trigger_reason=f"认知负荷过高: {cognitive_load:.2f} >= {EMERGENCY_THRESHOLD}",
                trigger_value=cognitive_load,
                alert_type=AlertType.HIGH_COGNITIVE_LOAD,
                cognitive_load=cognitive_load,
                error_count=consecutive_errors,
            )

        # 2. 连续错误过多 (严重, P0)
        if consecutive_errors >= CONSECUTIVE_ERROR_THRESHOLD:
            return EmergencyAlert(
                session_id=session_id,
                user_id=user_id,
                trigger_reason=(
                    f"连续错误过多: {consecutive_errors} >= "
                    f"{CONSECUTIVE_ERROR_THRESHOLD}"
                ),
                trigger_value=float(consecutive_errors),
                alert_type=AlertType.CONSECUTIVE_ERRORS,
                cognitive_load=cognitive_load,
                error_count=consecutive_errors,
            )

        # 3. 异常答题速度 (中等, P1)
        if avg_answer_time_ms < FAST_ANSWER_THRESHOLD_MS:
            return EmergencyAlert(
                session_id=session_id,
                user_id=user_id,
                trigger_reason=(
                    f"异常答题速度: {avg_answer_time_ms}ms < "
                    f"{FAST_ANSWER_THRESHOLD_MS}ms"
                ),
                trigger_value=float(avg_answer_time_ms),
                alert_type=AlertType.FAST_ANSWERING,
                cognitive_load=cognitive_load,
                error_count=consecutive_errors,
            )

        # 4. BKT 预测偏差过大 (纠错型, 非紧急, P2)
        if bkt_deviation > BKT_DEVIATION_THRESHOLD:
            return EmergencyAlert(
                session_id=session_id,
                user_id=user_id,
                trigger_reason=(
                    f"BKT 预测偏差过大: {bkt_deviation:.2f} > "
                    f"{BKT_DEVIATION_THRESHOLD}"
                ),
                trigger_value=bkt_deviation,
                alert_type=AlertType.BKT_DEVIATION,
                cognitive_load=cognitive_load,
                error_count=consecutive_errors,
            )

        return None


# ============================================================
# 6. 反馈回路 (FeedbackLoop)
# ============================================================


@dataclass
class FeedbackRoutingResult:
    """反馈路由结果 (设计文档 4.5).

    借鉴 Stanford HAI 反馈闭环: 分类 → 路由 → 追踪.

    Attributes:
        report_id: 关联的反馈报告 ID
        feedback_type: 原始反馈类型
        category: 分类结果 (FACTUAL/ADAPTIVE/SAFETY/None)
        routing_target: 路由目标 (knowledge_base/abac_policy/governance/None)
        severity: 严重度 [0.0, 1.0]
        source_envelope_id: 来源信封 ID (可溯源)
        routed_at: 路由时间戳 (毫秒)
    """

    report_id: str
    feedback_type: FeedbackType
    category: FeedbackCategory | None = None
    routing_target: str | None = None
    severity: float = 0.0
    source_envelope_id: str | None = None
    routed_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "report_id": self.report_id,
            "feedback_type": self.feedback_type.value,
            "category": self.category.value if self.category else None,
            "routing_target": self.routing_target,
            "severity": self.severity,
            "source_envelope_id": self.source_envelope_id,
            "routed_at": self.routed_at,
        }


class FeedbackLoop:
    """反馈回路引擎 (设计文档 4.5).

    借鉴世界先进方案:
    - Stanford HAI: 反馈闭环 (提交 → 分类 → 路由 → 追踪)
    - Google PAIR: 人机协作反馈分类
    - OpenAI Feedback API: 结构化反馈收集

    反馈分类与路由:
    - INCORRECT → FACTUAL → knowledge_base (L3 知识库修正)
    - NEED_MORE → ADAPTIVE → abac_policy (ABAC 策略调整)
    - REPORT → SAFETY → governance (L0 治理升级)
    - UNDERSTOOD → None → None (无需路由, 记录即可)

    线程安全: threading.RLock 保护反馈历史.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._history: list[FeedbackReport] = []

    def submit_feedback(self, report: FeedbackReport) -> FeedbackRoutingResult:
        """提交反馈并进行分类路由.

        Args:
            report: 反馈报告

        Returns:
            FeedbackRoutingResult 包含分类和路由信息

        Raises:
            FeedbackError: 反馈报告为 None 或非法
        """
        if report is None:
            raise FeedbackError(detail="反馈报告不能为 None")

        # 分类
        category = self._classify(report)

        # 路由
        routing_target = self._route(category)

        # 记录历史
        with self._lock:
            self._history.append(report)
            # 限制历史长度
            if len(self._history) > FEEDBACK_HISTORY_LIMIT * 10:
                self._history = self._history[-FEEDBACK_HISTORY_LIMIT * 10:]

        return FeedbackRoutingResult(
            report_id=report.report_id,
            feedback_type=report.feedback_type,
            category=category,
            routing_target=routing_target,
            severity=report.severity,
            source_envelope_id=report.source_envelope_id,
        )

    def get_feedback_history(
        self,
        session_id: str,
        limit: int = FEEDBACK_HISTORY_LIMIT,
    ) -> list[FeedbackReport]:
        """获取指定会话的反馈历史.

        Args:
            session_id: 会话 ID
            limit: 返回上限

        Returns:
            反馈报告列表 (按时间倒序, 最多 limit 条)
        """
        with self._lock:
            filtered = [
                r for r in self._history if r.session_id == session_id
            ]
            # 按创建时间倒序, 取最近 limit 条
            filtered.sort(key=lambda r: r.created_at, reverse=True)
            return filtered[:limit]

    def get_all_history(self) -> list[FeedbackReport]:
        """获取所有反馈历史 (防御性拷贝)."""
        with self._lock:
            return list(self._history)

    def clear(self) -> None:
        """清空反馈历史."""
        with self._lock:
            self._history.clear()

    @staticmethod
    def _classify(report: FeedbackReport) -> FeedbackCategory | None:
        """分类反馈报告.

        分类规则:
        - INCORRECT → FACTUAL (事实性错误)
        - NEED_MORE → ADAPTIVE (适应性不足)
        - REPORT → SAFETY (安全问题)
        - UNDERSTOOD → None (无需分类)
        """
        if report.feedback_type == FeedbackType.INCORRECT:
            return FeedbackCategory.FACTUAL
        elif report.feedback_type == FeedbackType.NEED_MORE:
            return FeedbackCategory.ADAPTIVE
        elif report.feedback_type == FeedbackType.REPORT:
            return FeedbackCategory.SAFETY
        else:
            return None

    @staticmethod
    def _route(category: FeedbackCategory | None) -> str | None:
        """路由反馈到目标系统.

        路由规则:
        - FACTUAL → knowledge_base (L3 知识库修正)
        - ADAPTIVE → abac_policy (ABAC 策略调整)
        - SAFETY → governance (L0 治理升级)
        - None → None (无需路由)
        """
        if category == FeedbackCategory.FACTUAL:
            return "knowledge_base"
        elif category == FeedbackCategory.ADAPTIVE:
            return "abac_policy"
        elif category == FeedbackCategory.SAFETY:
            return "governance"
        else:
            return None


# ============================================================
# 7. 纠错型结果 (CorrectionResult)
# ============================================================


@dataclass
class CorrectionResult:
    """纠错型处理结果 (设计文档 4.1).

    记录纠错自纠循环的状态:
    - request_id: 关联的审批请求 ID
    - retry_count: 当前重试次数
    - escalated: 是否已升级 (超过 MAX_CORRECTION_RETRIES)
    - escalation_target: 升级目标 ("teacher" 或 None)
    - resolved: 是否已解决 (学生确认通过)
    - result_response: 最终审批响应
    - processed_at: 处理时间戳 (毫秒)
    """

    request_id: str
    retry_count: int = 0
    escalated: bool = False
    escalation_target: str | None = None
    resolved: bool = False
    result_response: ApprovalResponse | None = None
    processed_at: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def decision(self) -> ApprovalDecision | None:
        """获取最终决策."""
        if self.result_response is not None:
            return self.result_response.decision
        return None

    @property
    def modifications(self) -> list[dict[str, Any]]:
        """获取修改建议."""
        if self.result_response is not None:
            return self.result_response.modifications
        return []

    def is_approved(self) -> bool:
        """是否最终批准."""
        if self.result_response is not None:
            return self.result_response.is_approved()
        return False


# ============================================================
# 8. HiTL 核心管理器 (HiTLManager)
# ============================================================


class HiTLManager:
    """HiTL 协同核心管理器 (设计文档 4.1-4.5).

    借鉴世界先进方案:
    - LangGraph Human-in-the-Loop: 节点级中断 + 状态恢复
    - OpenAI Constitutional AI: 自我修正循环
    - Anthropic Claude: 置信度门控驱动交互模式选择
    - Khan Academy 教师仪表盘: 紧急干预通知 + 升级机制
    - Google PAIR: 人机协作模式分类
    - Microsoft Guidelines: 交互模式推荐引擎

    四类协同场景编排:
    1. 确认型 (CONFIRMATION): 学生确认"已理解" → PASS/REJECT
    2. 纠错型 (CORRECTION): 学生标记"不理解" → Agent 自纠 → 最多 3 次 → 升级教师
    3. 创造型 (CREATIVE): 教师创建内容 → 审核 Agent 校验 → APPROVE/REJECT/MODIFY
    4. 紧急干预 (EMERGENCY): 自动检测 → 暂停 + 通知教师 → 人工解决

    线程安全: threading.RLock 保护审批请求存储.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._requests: dict[str, ApprovalRequest] = {}
        self._retry_counts: dict[str, int] = {}
        self._confidence_gate = ConfidenceGate()
        self._emergency_detector = EmergencyDetector()
        self._feedback_loop = FeedbackLoop()

    # --- 属性 ---

    @property
    def confidence_gate(self) -> ConfidenceGate:
        """置信度门控引擎."""
        return self._confidence_gate

    @property
    def emergency_detector(self) -> EmergencyDetector:
        """紧急检测器."""
        return self._emergency_detector

    @property
    def feedback_loop(self) -> FeedbackLoop:
        """反馈回路引擎."""
        return self._feedback_loop

    # --- 审批请求管理 ---

    def create_approval_request(
        self,
        user_id: str,
        session_id: str,
        hitl_type: HiTLType,
        content: str,
        confidence: float = 1.0,
        priority: HiTLPriority = HiTLPriority.P2,
        deadline: int | None = None,
    ) -> ApprovalRequest:
        """创建 HiTL 审批请求.

        Args:
            user_id: 请求目标用户 ID
            session_id: 关联会话 ID
            hitl_type: 协同场景类型
            content: 需确认的内容
            confidence: Agent 输出置信度 [0.0, 1.0]
            priority: 优先级
            deadline: 截止时间戳 (毫秒), None 表示永不过期

        Returns:
            创建的 ApprovalRequest
        """
        request = ApprovalRequest(
            user_id=user_id,
            session_id=session_id,
            hitl_type=hitl_type,
            content=content,
            confidence=confidence,
            priority=priority,
            deadline=deadline,
        )
        with self._lock:
            self._requests[request.request_id] = request
            self._retry_counts[request.request_id] = 0
        return request

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """按 ID 获取审批请求.

        Args:
            request_id: 请求 ID

        Returns:
            ApprovalRequest 或 None (不存在)
        """
        with self._lock:
            return self._requests.get(request_id)

    def get_pending_requests(
        self,
        user_id: str | None = None,
    ) -> list[ApprovalRequest]:
        """获取待处理请求列表.

        自动过滤已过期请求 (标记为 expired).

        Args:
            user_id: 可选, 按用户过滤

        Returns:
            待处理请求列表
        """
        with self._lock:
            result: list[ApprovalRequest] = []
            now_ms = int(time.time() * 1000)
            for request in self._requests.values():
                # 跳过已处理的
                if request.status != "pending":
                    continue
                # 检查过期
                if request.deadline is not None and now_ms > request.deadline:
                    request.status = "expired"
                    continue
                # 按用户过滤
                if user_id is not None and request.user_id != user_id:
                    continue
                result.append(request)
            return result

    # --- 四类协同场景处理 ---

    def handle_confirmation(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> ApprovalResponse:
        """处理确认型协同 (设计文档 4.1).

        学生确认"已理解"或"不理解":
        - APPROVE → request.status = "approved"
        - REJECT → request.status = "rejected"

        Args:
            request: 审批请求
            response: 审批响应

        Returns:
            更新后的审批响应

        Raises:
            ApprovalError: 请求不存在或已处理
        """
        self._validate_request(request, response)

        if response.decision == ApprovalDecision.APPROVE:
            request.status = "approved"
        else:
            request.status = "rejected"

        return response

    def handle_correction(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> CorrectionResult:
        """处理纠错型协同 (设计文档 4.1).

        学生标记"不理解" → Agent 自纠循环:
        - 每次拒绝增加重试计数
        - 超过 MAX_CORRECTION_RETRIES → 升级教师
        - 学生确认通过 → 解决

        Args:
            request: 审批请求
            response: 审批响应

        Returns:
            CorrectionResult 包含重试次数和升级状态

        Raises:
            ApprovalError: 请求不存在或已处理
        """
        self._validate_request(request, response)

        with self._lock:
            retry_count = self._retry_counts.get(request.request_id, 0)

            if response.decision == ApprovalDecision.APPROVE:
                # 学生确认通过, 解决
                request.status = "approved"
                result = CorrectionResult(
                    request_id=request.request_id,
                    retry_count=retry_count + 1,
                    escalated=False,
                    resolved=True,
                    result_response=response,
                )
            else:
                # 学生拒绝, 增加重试
                retry_count += 1
                self._retry_counts[request.request_id] = retry_count

                if retry_count >= MAX_CORRECTION_RETRIES:
                    # 超过最大重试, 升级教师
                    request.status = "escalated"
                    result = CorrectionResult(
                        request_id=request.request_id,
                        retry_count=retry_count,
                        escalated=True,
                        escalation_target="teacher",
                        resolved=False,
                        result_response=response,
                    )
                else:
                    # 未超过, 继续 Agent 自纠
                    request.status = "pending"
                    result = CorrectionResult(
                        request_id=request.request_id,
                        retry_count=retry_count,
                        escalated=False,
                        resolved=False,
                        result_response=response,
                    )

            return result

    def handle_creative(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> ApprovalResponse:
        """处理创造型协同 (设计文档 4.1).

        教师创建内容 → 审核 Agent 校验:
        - APPROVE → request.status = "approved"
        - REJECT → request.status = "rejected"
        - MODIFY → request.status = "modify" (退回修改)

        Args:
            request: 审批请求
            response: 审批响应

        Returns:
            更新后的审批响应

        Raises:
            ApprovalError: 请求不存在或已处理
        """
        self._validate_request(request, response)

        if response.decision == ApprovalDecision.APPROVE:
            request.status = "approved"
        elif response.decision == ApprovalDecision.REJECT:
            request.status = "rejected"
        elif response.decision == ApprovalDecision.MODIFY:
            request.status = "modify"
        else:
            request.status = "rejected"

        return response

    def handle_emergency(
        self,
        session_id: str,
        user_id: str,
        cognitive_load: float,
        consecutive_errors: int,
        avg_answer_time_ms: int,
        bkt_deviation: float = 0.0,
    ) -> EmergencyAlert | None:
        """处理紧急干预 (设计文档 4.2, 4.3).

        自动检测 + 暂停 + 通知教师:
        - 检测紧急条件
        - 生成警报
        - (生产环境: 通知教师 + 暂停 Agent)

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            cognitive_load: 认知负荷 [0.0, 1.0]
            consecutive_errors: 连续错误次数
            avg_answer_time_ms: 平均答题时间 (毫秒)
            bkt_deviation: BKT 预测偏差

        Returns:
            EmergencyAlert 如果触发, 否则 None
        """
        return self._emergency_detector.check(
            session_id=session_id,
            user_id=user_id,
            cognitive_load=cognitive_load,
            consecutive_errors=consecutive_errors,
            avg_answer_time_ms=avg_answer_time_ms,
            bkt_deviation=bkt_deviation,
        )

    def resolve_emergency(self, alert_id: str) -> bool:
        """解决紧急警报.

        Args:
            alert_id: 警报 ID

        Returns:
            True 如果成功解决
        """
        return self._emergency_detector.resolve_alert(alert_id)

    # --- 交互模式推荐 ---

    def get_interaction_mode(
        self,
        hitl_type: HiTLType,
        gate_result: ConfidenceGateResult,
    ) -> InteractionMode:
        """根据协同场景和门控结果推荐交互模式.

        借鉴 Microsoft Guidelines for Human-AI Interaction:
        交互模式矩阵 (HiTLType × GateResult):

        | HiTLType      | PASS    | WARNING             | BLOCK             |
        |---------------|---------|---------------------|-------------------|
        | CONFIRMATION  | 被动确认 | 主动建议            | 强制阻断          |
        | CORRECTION    | 被动确认 | 主动建议            | 强制阻断          |
        | CREATIVE      | 被动确认 | 可选协商            | 强制阻断          |
        | EMERGENCY     | 强制阻断 | 强制阻断            | 强制阻断          |

        Args:
            hitl_type: 协同场景类型
            gate_result: 置信度门控结果

        Returns:
            推荐的 InteractionMode
        """
        # 紧急干预 → 始终强制阻断
        if hitl_type == HiTLType.EMERGENCY:
            return InteractionMode.MANDATORY_BLOCK

        # BLOCK → 始终强制阻断
        if gate_result == ConfidenceGateResult.BLOCK:
            return InteractionMode.MANDATORY_BLOCK

        # PASS → 被动确认
        if gate_result == ConfidenceGateResult.PASS:
            return InteractionMode.PASSIVE_CONFIRMATION

        # WARNING → 按场景区分
        if gate_result == ConfidenceGateResult.WARNING:
            if hitl_type == HiTLType.CREATIVE:
                return InteractionMode.OPTIONAL_NEGOTIATION
            else:
                return InteractionMode.PROACTIVE_SUGGESTION

        # 默认: 被动确认
        return InteractionMode.PASSIVE_CONFIRMATION

    # --- 内部方法 ---

    def _validate_request(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> None:
        """验证审批请求和响应的合法性.

        Args:
            request: 审批请求
            response: 审批响应

        Raises:
            ApprovalError: 请求不存在、已处理、或响应不匹配
        """
        with self._lock:
            # 检查请求是否在管理器中注册
            if request.request_id not in self._requests:
                raise ApprovalError(
                    detail=f"审批请求不存在: {request.request_id}",
                    request_id=request.request_id,
                )

            # 检查请求是否已处理
            registered = self._requests[request.request_id]
            if registered.status not in ("pending",):
                raise ApprovalError(
                    detail=(
                        f"审批请求已处理: {request.request_id}, "
                        f"当前状态: {registered.status}"
                    ),
                    request_id=request.request_id,
                )

            # 检查响应的 request_id 是否匹配
            if response.request_id != request.request_id:
                raise ApprovalError(
                    detail=(
                        f"响应的 request_id ({response.request_id}) "
                        f"与请求 ({request.request_id}) 不匹配"
                    ),
                    request_id=request.request_id,
                )

            # 同步外部 request 对象的状态 (确保一致性)
            request.status = registered.status
            request.created_at = registered.created_at
