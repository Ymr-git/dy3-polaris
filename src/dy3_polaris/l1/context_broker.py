"""L1 用户域学习上下文经纪 (Learning Context Broker, LCB) — 核心引擎.

设计依据:
- L1 设计文档第三章 3.1-3.6: 上下文定义、采集机制、传递协议、衰减与刷新
- L1 设计文档第八章 8.2: 与 L2 的接口 (学情画像输入/输出、BKT 参数更新)
- L1 设计文档第七章 7.3: API /api/v1/context/{session_id}

融合世界先进方案:
- ContextFlow: 三区分层上下文架构 (Fixed/Working/History Zone)
- xAPI (IEEE 9274.1.1): Actor-Verb-Object 标准化事件信封
- Redis Agent Memory: 两层记忆模型 (Session + Long-term)
- FSRS: 幂律遗忘曲线 (stability/difficulty/retrievability)
- Khan Academy: 教育场景动态难度调节 + 学习路径追踪
- Duolingo: 百万级数据点构建学习者知识信号
- OSCOI 模式: BKT 离线校准 + 在线推断

模块组成:
1. 异常体系: L1ContextError 层级 (JSON-RPC -32300 范围)
2. 事件采集: FrontendEvent / AgentOutputEvent / UserDeclaration (xAPI Statement)
3. ContextCollector: 三渠道采集器 (前端埋点 / Agent 输出 / 用户声明)
4. DecayEngine: Ebbinghaus 遗忘衰减引擎
5. ContextCache: TTL 缓存分层 (会话级 + 持久层)
6. LearningContextBroker: 核心引擎 (构建/获取/更新/刷新/传递)
"""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from dy3_polaris.l1.models import (
    BKTParams,
    CognitiveLoadBreakdown,
    ContextEnvelope,
    LearningGoal,
    LearningPhase,
    LearningState,
    MasterySnapshot,
    MasteryTrajectory,
    MasteryTrajectoryPoint,
    ResourceItem,
    TimeConstraint,
    calculate_decay,
    DEFAULT_TTL,
    DEFAULT_COGNITIVE_LOAD,
    MIN_STABILITY,
    STABILITY_GAIN,
    PRIOR_PROB,
    MS_PER_HOUR,
    MS_PER_SEC,
    WEAK_THRESHOLD,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 1. 常量定义
# ============================================================

# --- 缓存 TTL (设计文档 3.4 TTL 机制表) ---
CACHE_TTL_SESSION: int = 7200          # 会话级: 2 小时 (活跃会话热数据)
CACHE_TTL_MASTERY: int = 3600          # 知识掌握快照: 1 小时
CACHE_TTL_GOAL: int = 24 * 3600        # 学习目标: 24 小时
CACHE_TTL_COGNITIVE: int = 1800        # 认知负荷: 30 分钟

# --- 认知负荷计算权重 (设计文档 3.1) ---
# 公式: load = base + error_rate*error_w + slow_rate*slow_w + help_rate*help_w
# 低负荷 (全对+快答+无求助): base = 0.2 (低于 0.5 阈值)
# 高负荷 (全错+慢答+全求助): 0.2 + 0.4 + 0.25 + 0.15 = 1.0
COGNITIVE_LOAD_BASE: float = 0.2          # 基础认知负荷 (低负荷起点)
COGNITIVE_LOAD_FAST_ANSWER_MS: int = 5000 # 异常快答题阈值 (毫秒)
COGNITIVE_LOAD_SLOW_ANSWER_MS: int = 8000 # 慢答题阈值 (毫秒)
COGNITIVE_LOAD_ERROR_WEIGHT: float = 0.4  # 错误率权重 (最大影响因子)
COGNITIVE_LOAD_SLOW_WEIGHT: float = 0.25  # 慢响应权重
COGNITIVE_LOAD_HELP_WEIGHT: float = 0.15  # 求助频率权重

# --- 复习紧急度 ---
REVIEW_URGENCY_THRESHOLD: float = 0.5     # 有效掌握度低于此值需复习

# --- 隐私过滤: 禁止采集的事件类型 (设计文档 3.2) ---
_BLOCKED_EVENT_TYPES: frozenset[str] = frozenset({
    "mouse_move",
    "mouse_track",
    "heatmap_click",
    "heatmap_view",
    "cross_domain_request",
    "cross_domain_tracking",
})

# --- 隐私过滤: 禁止采集的资源模式 ---
_BLOCKED_RESOURCE_PATTERNS: tuple[str, ...] = (
    "external-site",
    "third-party",
    "cross-domain",
)


# ============================================================
# 2. 异常体系 (JSON-RPC -32300 范围)
# ============================================================


class L1ContextError(L6Error):
    """L1 上下文经纪层基础异常 (JSON-RPC -32300)."""

    def __init__(
        self,
        code: str = "L1_CONTEXT_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32300


class ContextNotFoundError(L1ContextError):
    """上下文未找到 (JSON-RPC -32301).

    会话不存在或上下文已被清除.
    """

    def __init__(
        self,
        session_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {"session_id": session_id}
        if context:
            ctx.update(context)
        super().__init__(
            "CONTEXT_NOT_FOUND",
            detail or f"会话上下文未找到: {session_id}",
            ctx,
        )

    def _jsonrpc_code(self) -> int:
        return -32301


class ContextExpiredError(L1ContextError):
    """上下文已过期 (JSON-RPC -32302).

    TTL 超时, 上下文需要刷新.
    """

    def __init__(
        self,
        session_id: str,
        expired_at: int = 0,
        detail: str = "",
    ) -> None:
        super().__init__(
            "CONTEXT_EXPIRED",
            detail or f"会话上下文已过期: {session_id}",
            {"session_id": session_id, "expired_at": expired_at},
        )

    def _jsonrpc_code(self) -> int:
        return -32302


class ContextValidationError(L1ContextError):
    """上下文验证失败 (JSON-RPC -32303).

    参数非法、必填字段缺失等.
    """

    def __init__(
        self,
        detail: str = "上下文验证失败",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("CONTEXT_VALIDATION_ERROR", detail, context)

    def _jsonrpc_code(self) -> int:
        return -32303


class DecayError(L1ContextError):
    """衰减计算错误 (JSON-RPC -32304)."""

    def __init__(
        self,
        detail: str = "衰减计算失败",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("DECAY_ERROR", detail, context)

    def _jsonrpc_code(self) -> int:
        return -32304


# ============================================================
# 3. 事件数据结构 (xAPI Actor-Verb-Object 模型)
# ============================================================


@dataclass
class FrontendEvent:
    """前端埋点事件 (设计文档 3.2 渠道 1).

    借鉴 xAPI Statement: Actor-Verb-Object 三元组 + Result 扩展.

    采集对象: 页面浏览、答题响应、资源完成度、按钮点击.
    隐私约束: 不采集鼠标轨迹、热力图、跨域数据.

    Attributes:
        event_type: 事件类型 (page_view / answer_submit / resource_complete / button_click)
        actor_id: 学习者 ID
        target_resource: 目标资源 ID
        timestamp: 毫秒时间戳
        result: 事件结果 (如 is_correct, response_time_ms, duration_ms)
    """

    event_type: str
    actor_id: str
    target_resource: str
    timestamp: int
    result: dict[str, Any] = field(default_factory=dict)

    def to_xapi_statement(self) -> dict[str, Any]:
        """转换为 xAPI Statement 格式 (Actor-Verb-Object).

        Returns:
            xAPI 语句字典: {actor, verb, object, result, timestamp}
        """
        return {
            "actor": self.actor_id,
            "verb": self.event_type,
            "object": self.target_resource,
            "result": self.result,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentOutputEvent:
    """Agent 输出事件 (设计文档 3.2 渠道 2).

    借鉴 LOOM Dynamic Learner Memory Graph: 从 Agent 对话中提取长期掌握度.

    采集对象: BKT 参数更新、学习路径推荐、内容难度评估.
    时效性: 异步、认知层 (每次 Agent 调用后, 每会话 3-8 次).

    Attributes:
        agent_id: Agent 实例 ID
        output_type: 输出类型 (bkt_update / path_recommend / difficulty_assess)
        kc_id: 知识组件 ID
        p_know: BKT 后验掌握概率
        is_correct: 答题是否正确
        timestamp: 毫秒时间戳
    """

    agent_id: str
    output_type: str
    kc_id: str
    p_know: float
    is_correct: bool
    timestamp: int

    def to_xapi_statement(self) -> dict[str, Any]:
        """转换为 xAPI Statement 格式."""
        return {
            "actor": self.agent_id,
            "verb": self.output_type,
            "object": self.kc_id,
            "result": {
                "p_know": self.p_know,
                "is_correct": self.is_correct,
            },
            "timestamp": self.timestamp,
        }


@dataclass
class UserDeclaration:
    """用户显式声明 (设计文档 3.2 渠道 3).

    借鉴 Khan Academy: 用户自报告学习偏好和困惑点.

    采集对象: 可用学习时间、学习偏好、困惑点、目标优先级.
    时效性: 低频、高可信 (首次登录 + 每周一次).

    Attributes:
        user_id: 用户 ID
        available_minutes: 可用学习时间 (分钟)
        preferred_phase: 偏好学习阶段
        confusion_points: 困惑点列表
        timestamp: 毫秒时间戳
    """

    user_id: str
    available_minutes: int = 45
    preferred_phase: LearningPhase = LearningPhase.PRACTICE
    confusion_points: list[str] = field(default_factory=list)
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_xapi_statement(self) -> dict[str, Any]:
        """转换为 xAPI Statement 格式."""
        return {
            "actor": self.user_id,
            "verb": "declared",
            "object": "learning_preference",
            "result": {
                "available_minutes": self.available_minutes,
                "preferred_phase": self.preferred_phase.value,
                "confusion_points": self.confusion_points,
            },
            "timestamp": self.timestamp,
        }


# ============================================================
# 4. 上下文采集器 (ContextCollector)
# ============================================================


class ContextCollector:
    """上下文采集器 — 三渠道采集 + 隐私过滤 (设计文档 3.2).

    借鉴 ContextFlow 三区分层架构:
    - 前端埋点 → Working Zone (实时行为数据)
    - Agent 输出 → Working Zone (认知层数据)
    - 用户声明 → Fixed Zone (高可信偏好)

    隐私约束 (设计文档 3.2):
    - 不采集鼠标轨迹 (mouse_move / mouse_track)
    - 不采集热力图 (heatmap_click / heatmap_view)
    - 不采集跨域数据 (cross_domain_request)

    线程安全: threading.RLock 保护事件缓冲.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: list[FrontendEvent | AgentOutputEvent | UserDeclaration] = []

    def collect_frontend_event(self, event: FrontendEvent) -> list[FrontendEvent]:
        """采集前端事件 (渠道 1: 实时埋点).

        隐私过滤: 鼠标轨迹、热力图、跨域数据被自动过滤.

        Args:
            event: 前端埋点事件

        Returns:
            采集成功的事件列表 (空列表表示被过滤)
        """
        if self._is_blocked(event.event_type, event.target_resource):
            return []
        with self._lock:
            self._events.append(event)
            return [event]

    def collect_agent_output(self, event: AgentOutputEvent) -> list[AgentOutputEvent]:
        """采集 Agent 输出 (渠道 2: 异步认知层).

        Agent 输出不经隐私过滤 (认知层数据已脱敏).

        Args:
            event: Agent 输出事件

        Returns:
            采集成功的事件列表
        """
        with self._lock:
            self._events.append(event)
            return [event]

    def collect_user_declaration(self, decl: UserDeclaration) -> list[UserDeclaration]:
        """采集用户声明 (渠道 3: 低频高可信).

        用户声明不经隐私过滤 (用户主动提供).

        Args:
            decl: 用户声明

        Returns:
            采集成功的声明列表
        """
        with self._lock:
            self._events.append(decl)
            return [decl]

    def get_all_events(self) -> list[FrontendEvent | AgentOutputEvent | UserDeclaration]:
        """获取所有已采集事件 (防御性拷贝)."""
        with self._lock:
            return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[FrontendEvent]:
        """按事件类型查询前端事件."""
        with self._lock:
            return [
                e for e in self._events
                if isinstance(e, FrontendEvent) and e.event_type == event_type
            ]

    def clear(self) -> None:
        """清空采集缓冲."""
        with self._lock:
            self._events.clear()

    @staticmethod
    def _is_blocked(event_type: str, target_resource: str) -> bool:
        """检查事件是否应被隐私过滤阻止.

        Args:
            event_type: 事件类型
            target_resource: 目标资源

        Returns:
            True 如果事件应被阻止
        """
        # 检查事件类型黑名单
        if event_type in _BLOCKED_EVENT_TYPES:
            return True
        # 检查资源模式黑名单
        for pattern in _BLOCKED_RESOURCE_PATTERNS:
            if pattern in target_resource.lower():
                return True
        return False


# ============================================================
# 5. 遗忘衰减引擎 (DecayEngine)
# ============================================================


class DecayEngine:
    """遗忘衰减引擎 — Ebbinghaus 遗忘曲线 + 间隔重复修正 (设计文档 3.4).

    借鉴世界先进方案:
    - FSRS (Free Spaced Repetition Scheduler): 幂律遗忘曲线 R=(1+FACTOR*t/S)^d
    - ContextFlow: 多维相关性评分 (Recency 30% + Frequency 20% + Similarity 30% + Task 20%)
    - Ebbinghaus 经典模型: stability = MIN_STABILITY + reps * STABILITY_GAIN

    衰减公式:
    - stability = MIN_STABILITY + repetitions * STABILITY_GAIN
    - decay = exp(-elapsed_hours / stability)
    - effective_mastery = p_know * decay (不低于 PRIOR_PROB)

    线程安全: 纯静态方法, 无共享状态.
    """

    @staticmethod
    def calculate_decay(
        p_know: float,
        last_practiced: int,
        repetitions: int,
        current_ts: int,
    ) -> float:
        """计算单个 KC 的遗忘衰减后有效掌握度.

        委托给 models.calculate_decay (已实现并测试).

        Args:
            p_know: BKT 后验掌握概率 [0.0, 1.0]
            last_practiced: 上次练习时间戳 (毫秒)
            repetitions: 累计练习次数
            current_ts: 当前时间戳 (毫秒)

        Returns:
            衰减后的有效掌握度 [PRIOR_PROB, p_know]

        Raises:
            DecayError: 计算失败
        """
        try:
            return calculate_decay(
                p_know=p_know,
                last_practiced=last_practiced,
                repetitions=repetitions,
                current_ts=current_ts,
            )
        except ValueError as e:
            raise DecayError(
                f"衰减计算失败: {e}",
                context={
                    "p_know": p_know,
                    "repetitions": repetitions,
                },
            ) from e

    @staticmethod
    def refresh_all_decay(envelope: ContextEnvelope, current_ts: int) -> None:
        """批量刷新信封中所有 KC 的衰减系数.

        遍历 mastery_snapshot, 重新计算每个 MasterySnapshot 的 decay_factor:
        - stability = MIN_STABILITY + repetitions * STABILITY_GAIN
        - decay = exp(-elapsed_hours / stability)
        - decay_factor = decay (纯衰减乘数, 不含 p_know)

        Args:
            envelope: 待刷新的上下文信封
            current_ts: 当前时间戳 (毫秒)
        """
        for snap in envelope.mastery_snapshot:
            elapsed_hours = max(
                0, (current_ts - snap.last_practiced_at) / MS_PER_HOUR
            )
            stability = MIN_STABILITY + snap.repetitions * STABILITY_GAIN
            snap.decay_factor = math.exp(-elapsed_hours / stability)

    @staticmethod
    def get_review_urgency(
        envelope: ContextEnvelope,
        current_ts: int,
        threshold: float = REVIEW_URGENCY_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """获取复习紧急度排序 (借鉴 FSRS 可提取性 R + ContextFlow Recency 评分).

        计算每个 KC 的有效掌握度, 低于阈值的需要复习.
        按紧急度降序排列 (掌握度最低的最紧急).

        Args:
            envelope: 上下文信封
            current_ts: 当前时间戳 (毫秒)
            threshold: 复习阈值 (有效掌握度低于此值需复习)

        Returns:
            紧急复习列表, 每项包含:
            - kc_id: 知识组件 ID
            - effective_mastery: 有效掌握度
            - urgency_score: 紧急度分数 (1.0 - effective_mastery, 越高越紧急)
            - last_practiced_hours_ago: 距上次练习小时数
        """
        result: list[dict[str, Any]] = []
        for snap in envelope.mastery_snapshot:
            # 重算衰减系数
            elapsed_hours = max(
                0, (current_ts - snap.last_practiced_at) / MS_PER_HOUR
            )
            stability = MIN_STABILITY + snap.repetitions * STABILITY_GAIN
            decay = math.exp(-elapsed_hours / stability)
            effective = snap.p_know * decay

            if effective < threshold:
                result.append({
                    "kc_id": snap.kc_id,
                    "effective_mastery": round(effective, 4),
                    "urgency_score": round(1.0 - effective, 4),
                    "last_practiced_hours_ago": round(elapsed_hours, 2),
                })

        # 按紧急度降序排列
        result.sort(key=lambda x: x["urgency_score"], reverse=True)
        return result


# ============================================================
# 6. 上下文缓存 (ContextCache)
# ============================================================


class ContextCache:
    """上下文缓存 — TTL 分层缓存 (设计文档 3.4, 3.5).

    借鉴 Redis Agent Memory 两层记忆模型:
    - Session Memory (会话级): 热数据缓存, 毫秒级读取, TTL 控制
    - Long-term Memory (持久层): 冷数据备份, 会话级缓存失效后可恢复

    TTL 分层 (设计文档 3.4):
    - 学习状态: 会话级 (随会话存活)
    - 知识掌握快照: 1 小时 (CACHE_TTL_MASTERY)
    - 学习目标: 24 小时 (CACHE_TTL_GOAL)
    - 认知负荷: 30 分钟 (CACHE_TTL_COGNITIVE)

    线程安全: threading.RLock 保护缓存层.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 会话级缓存: session_id → (envelope, expires_at)
        self._session_cache: dict[str, tuple[ContextEnvelope, float]] = {}
        # 持久层备份: session_id → envelope (冷数据)
        self._persistent_store: dict[str, ContextEnvelope] = {}

    def get(self, session_id: str) -> ContextEnvelope | None:
        """从会话级缓存获取上下文.

        检查 TTL 过期, 过期则返回 None.

        Args:
            session_id: 会话 ID

        Returns:
            上下文信封, 不存在或已过期返回 None
        """
        with self._lock:
            entry = self._session_cache.get(session_id)
            if entry is None:
                return None
            envelope, expires_at = entry
            # expires_at=0 表示永久不过期; 否则检查是否过期
            if expires_at != 0 and time.time() > expires_at:
                # 过期, 清理
                del self._session_cache[session_id]
                return None
            return envelope

    def set(
        self,
        session_id: str,
        envelope: ContextEnvelope,
        ttl: int = CACHE_TTL_SESSION,
    ) -> None:
        """设置上下文到会话级缓存.

        Args:
            session_id: 会话 ID
            envelope: 上下文信封
            ttl: TTL (秒), 0 表示立即过期, 负数表示永久不过期
        """
        with self._lock:
            if ttl > 0:
                expires_at = time.time() + ttl
            elif ttl == 0:
                # TTL=0: 立即过期 (用于测试或显式过期场景)
                expires_at = time.time() - 1  # 设为过去时间
            else:
                # ttl < 0: 永久不过期
                expires_at = 0.0
            # 防御性拷贝, 避免外部修改影响缓存
            self._session_cache[session_id] = (copy.deepcopy(envelope), expires_at)

    def invalidate(self, session_id: str) -> None:
        """失效会话级缓存."""
        with self._lock:
            self._session_cache.pop(session_id, None)

    def backup_to_persistent(self, session_id: str) -> None:
        """备份上下文到持久层 (冷数据存储).

        借鉴 Redis Long-term Memory: 异步晋升机制.
        会话级缓存失效后, 可从持久层恢复.

        Args:
            session_id: 会话 ID
        """
        with self._lock:
            entry = self._session_cache.get(session_id)
            if entry is not None:
                self._persistent_store[session_id] = copy.deepcopy(entry[0])

    def restore_from_persistent(self, session_id: str) -> ContextEnvelope | None:
        """从持久层恢复上下文.

        借鉴 Redis 冷加载恢复机制.

        Args:
            session_id: 会话 ID

        Returns:
            恢复的上下文信封, 不存在返回 None
        """
        with self._lock:
            envelope = self._persistent_store.get(session_id)
            if envelope is not None:
                # 恢复到会话级缓存
                self._session_cache[session_id] = (copy.deepcopy(envelope), 0.0)
                return copy.deepcopy(envelope)
            return None

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息."""
        with self._lock:
            return {
                "total_entries": len(self._session_cache),
                "persistent_entries": len(self._persistent_store),
            }

    def clear_all(self) -> None:
        """清空所有缓存 (会话级 + 持久层)."""
        with self._lock:
            self._session_cache.clear()
            self._persistent_store.clear()


# ============================================================
# 7. 学习上下文经纪核心引擎 (LearningContextBroker)
# ============================================================


class LearningContextBroker:
    """学习上下文经纪 (LCB) 核心引擎 (设计文档 第三章).

    LCB 是本系统相较于通用 LLM 架构的最大差异化创新:
    将"计算状态持久化"改造为"认知状态持久化".

    核心职责:
    1. 构建标准化上下文信封 (ContextEnvelope)
    2. 管理知识掌握快照 (BKT 更新集成)
    3. 计算认知负荷 (响应时间 + 错误率 + 求助频率)
    4. 遗忘衰减刷新 (Ebbinghaus 曲线)
    5. 跨会话上下文传递 (掌握度继承 + 目标继承)

    借鉴世界先进方案:
    - ContextFlow: Working Zone 存储活跃学习上下文
    - Khan Academy: 学习路径追踪 + 动态难度
    - Duolingo: 多课程数据点聚合知识信号
    - xAPI: 标准化事件采集与审计
    - OSCOI: BKT 离线校准 + 在线推断

    线程安全: threading.RLock 保护所有共享状态.
    """

    def __init__(self, cache: ContextCache | None = None) -> None:
        self._cache = cache or ContextCache()
        self._lock = threading.RLock()

    # ================================================================
    # 上下文构建
    # ================================================================

    def build_envelope(
        self,
        user_id: str,
        session_id: str,
        initial_mastery: list[MasterySnapshot] | None = None,
        initial_goals: list[LearningGoal] | None = None,
        ttl: int = DEFAULT_TTL,
    ) -> ContextEnvelope:
        """构建标准化上下文信封 (设计文档 3.3).

        ContextEnvelope 是 L1 向下层传递数据的唯一载体.
        构建后自动写入缓存.

        Args:
            user_id: 用户 ID (脱敏后)
            session_id: 会话 ID
            initial_mastery: 初始掌握快照列表 (可选)
            initial_goals: 初始学习目标列表 (可选)
            ttl: 信封 TTL (秒)

        Returns:
            构建的 ContextEnvelope

        Raises:
            ContextValidationError: user_id 或 session_id 为空
        """
        if not user_id:
            raise ContextValidationError("user_id 不能为空")
        if not session_id:
            raise ContextValidationError("session_id 不能为空")

        envelope = ContextEnvelope(
            user_id=user_id,
            session_id=session_id,
            mastery_snapshot=initial_mastery or [],
            goals=initial_goals or [],
            ttl=ttl,
        )

        with self._lock:
            self._cache.set(session_id, envelope)
        return envelope

    def get_envelope(self, session_id: str) -> ContextEnvelope:
        """获取当前上下文 (Redis 热数据优先, 设计文档 3.5).

        Args:
            session_id: 会话 ID

        Returns:
            上下文信封

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self._cache.get(session_id)
            if envelope is None:
                # 尝试从持久层恢复
                envelope = self._cache.restore_from_persistent(session_id)
            if envelope is None:
                raise ContextNotFoundError(session_id)
            return envelope

    # ================================================================
    # 知识掌握度更新 (BKT 集成, 设计文档 8.2)
    # ================================================================

    def update_mastery(
        self,
        session_id: str,
        kc_id: str,
        p_know: float,
        is_correct: bool,
    ) -> None:
        """更新知识掌握快照 (BKT 贝叶斯更新, 设计文档 8.2).

        借鉴 OSCOI 模式: 在线推断 + Redis 状态缓存.
        L2 BKT 引擎更新参数后推送给 L1, L1 实时更新掌握度快照.

        更新逻辑:
        1. 查找已有 MasterySnapshot (按 kc_id)
        2. 使用 BKTParams.bayesian_update() 更新 p_know
        3. 递增 repetitions / attempts / correct_count
        4. 更新 last_practiced_at 为当前时间
        5. 写回缓存

        Args:
            session_id: 会话 ID
            kc_id: 知识组件 ID
            p_know: 当前 BKT 后验掌握概率 (更新前)
            is_correct: 本次答题是否正确

        Raises:
            ContextNotFoundError: 会话不存在
            ValueError: p_know 不在 [0.0, 1.0]
        """
        with self._lock:
            envelope = self.get_envelope(session_id)

            # 查找已有快照
            existing: MasterySnapshot | None = None
            for snap in envelope.mastery_snapshot:
                if snap.kc_id == kc_id:
                    existing = snap
                    break

            now_ms = int(time.time() * 1000)

            if existing is not None:
                # BKT 贝叶斯更新
                updated_bkt = existing.bkt_params.bayesian_update(is_correct)
                existing.p_know = updated_bkt.p_know
                existing.bkt_params = updated_bkt
                existing.repetitions += 1
                existing.attempts += 1
                if is_correct:
                    existing.correct_count += 1
                existing.last_practiced_at = now_ms
            else:
                # 新增 KC
                new_bkt = BKTParams(p_know=p_know)
                updated_bkt = new_bkt.bayesian_update(is_correct)
                new_snap = MasterySnapshot(
                    kc_id=kc_id,
                    p_know=updated_bkt.p_know,
                    last_practiced_at=now_ms,
                    repetitions=1,
                    bkt_params=updated_bkt,
                    correct_count=1 if is_correct else 0,
                    attempts=1,
                )
                envelope.mastery_snapshot.append(new_snap)

            # 自动记录掌握度轨迹 (MasteryTrajectory)
            self._record_mastery_trajectory(envelope, kc_id, now_ms)

            # 写回缓存
            self._cache.set(session_id, envelope)

    def _record_mastery_trajectory(
        self,
        envelope: ContextEnvelope,
        kc_id: str,
        timestamp: int,
    ) -> None:
        """记录掌握度轨迹点 (内部辅助方法).

        在 update_mastery 更新快照后调用, 将当前 p_know 作为
        新的 MasteryTrajectoryPoint 追加到对应 KC 的轨迹中.

        Args:
            envelope: 上下文信封
            kc_id: 知识组件 ID
            timestamp: 时间戳 (毫秒)
        """
        # 查找对应 KC 的快照
        snap: MasterySnapshot | None = None
        for s in envelope.mastery_snapshot:
            if s.kc_id == kc_id:
                snap = s
                break
        if snap is None:
            return

        # 获取或创建轨迹
        if kc_id not in envelope.mastery_trajectories:
            envelope.mastery_trajectories[kc_id] = MasteryTrajectory(kc_id=kc_id)

        # 追加轨迹点 (记录更新后的 p_know)
        point = MasteryTrajectoryPoint(
            kc_id=kc_id,
            timestamp=timestamp,
            p_know=snap.p_know,
        )
        envelope.mastery_trajectories[kc_id].add_point(point)

    # ================================================================
    # 认知负荷计算 (设计文档 3.1)
    # ================================================================

    def update_cognitive_load(
        self,
        session_id: str,
        interactions: list[dict[str, Any]],
    ) -> None:
        """计算认知负荷 (设计文档 3.1).

        认知负荷 = 基础值 + 错误率×权重 + 慢响应率×权重 + 求助率×权重

        三维度信号:
        - 响应时间: 超过 SLOW_ANSWER_MS 视为慢响应
        - 错误率: 答错比例
        - 求助频率: 请求帮助比例

        Args:
            session_id: 会话 ID
            interactions: 交互列表, 每项含:
                - response_time_ms: 响应时间 (毫秒)
                - is_correct: 是否正确
                - asked_help: 是否求助

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)

            if not interactions:
                # 空交互保持默认值
                return

            n = len(interactions)
            error_count = sum(1 for i in interactions if not i.get("is_correct", False))
            slow_count = sum(
                1 for i in interactions
                if i.get("response_time_ms", 0) > COGNITIVE_LOAD_SLOW_ANSWER_MS
            )
            help_count = sum(1 for i in interactions if i.get("asked_help", False))

            error_rate = error_count / n
            slow_rate = slow_count / n
            help_rate = help_count / n

            load = (
                COGNITIVE_LOAD_BASE
                + error_rate * COGNITIVE_LOAD_ERROR_WEIGHT
                + slow_rate * COGNITIVE_LOAD_SLOW_WEIGHT
                + help_rate * COGNITIVE_LOAD_HELP_WEIGHT
            )

            # 限制在 [0.0, 1.0]
            load = max(0.0, min(1.0, load))
            envelope.learning_state.cognitive_load = load
            envelope.learning_state.interaction_count += n

            # 写回缓存
            self._cache.set(session_id, envelope)

    def update_cognitive_load_breakdown(
        self,
        session_id: str,
        interactions: list[dict[str, Any]],
    ) -> None:
        """计算认知负荷三分模型 (Sweller Cognitive Load Theory).

        将认知负荷分解为三个维度:
        - ICL (内在负荷): 与任务复杂度 / 错误率正相关
        - ECL (外在负荷): 与呈现方式 / 慢响应率 / 求助率正相关
        - GCL (生成性负荷): 与主动学习行为正相关

        各维度计算:
        - ICL = error_rate * 0.4 + slow_rate * 0.2, clamp [0, 0.5]
        - ECL = slow_rate * 0.3 + help_rate * 0.3, clamp [0, 0.3]
        - GCL = active_rate * 0.5, clamp [0, 0.2]
        - total_load = ICL + ECL + GCL <= 1.0

        Args:
            session_id: 会话 ID
            interactions: 交互列表, 每项含:
                - response_time_ms: 响应时间 (毫秒)
                - is_correct: 是否正确
                - asked_help: 是否求助
                - content_type: 内容类型 (video/text/quiz/interactive 等)

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)

            if not interactions:
                return

            n = len(interactions)
            error_count = sum(
                1 for i in interactions if not i.get("is_correct", False)
            )
            slow_count = sum(
                1 for i in interactions
                if i.get("response_time_ms", 0) > COGNITIVE_LOAD_SLOW_ANSWER_MS
            )
            help_count = sum(
                1 for i in interactions if i.get("asked_help", False)
            )
            active_count = sum(
                1 for i in interactions if self._is_active_learning(i)
            )

            error_rate = error_count / n
            slow_rate = slow_count / n
            help_rate = help_count / n
            active_rate = active_count / n

            # ICL (内在负荷): 错误率 * 0.4 + 慢响应率 * 0.2
            icl = error_rate * 0.4 + slow_rate * 0.2
            icl = max(0.0, min(0.5, icl))

            # ECL (外在负荷): 慢响应率 * 0.3 + 求助率 * 0.3
            ecl = slow_rate * 0.3 + help_rate * 0.3
            ecl = max(0.0, min(0.3, ecl))

            # GCL (生成性负荷): 主动学习行为率 * 0.5
            gcl = active_rate * 0.5
            gcl = max(0.0, min(0.2, gcl))

            envelope.cognitive_load_breakdown = CognitiveLoadBreakdown(
                intrinsic_load=icl,
                extraneous_load=ecl,
                germane_load=gcl,
            )

            # 写回缓存
            self._cache.set(session_id, envelope)

    @staticmethod
    def _is_active_learning(interaction: dict[str, Any]) -> bool:
        """判断交互是否为主动学习行为.

        主动学习 content_type: interactive/quiz/practice/exercise/simulation/experiment.
        被动学习 content_type: video/text/reading/lecture/audio.
        """
        content_type = str(interaction.get("content_type", "")).lower().strip()
        active_types = {
            "interactive", "quiz", "practice", "exercise",
            "simulation", "experiment", "hands_on",
        }
        return content_type in active_types

    # ================================================================
    # 学习状态更新
    # ================================================================

    def update_learning_phase(
        self,
        session_id: str,
        phase: LearningPhase,
    ) -> None:
        """更新学习阶段.

        Args:
            session_id: 会话 ID
            phase: 新学习阶段

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)
            envelope.learning_state.phase = phase
            self._cache.set(session_id, envelope)

    def update_goals(
        self,
        session_id: str,
        goals: list[LearningGoal],
    ) -> None:
        """更新学习目标.

        Args:
            session_id: 会话 ID
            goals: 新目标列表

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)
            envelope.goals = list(goals)
            self._cache.set(session_id, envelope)

    def add_resource(
        self,
        session_id: str,
        resource: ResourceItem,
    ) -> None:
        """添加可用资源.

        Args:
            session_id: 会话 ID
            resource: 资源项

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)
            envelope.resources.append(resource)
            self._cache.set(session_id, envelope)

    def set_time_constraint(
        self,
        session_id: str,
        constraint: TimeConstraint,
    ) -> None:
        """设置时间约束.

        Args:
            session_id: 会话 ID
            constraint: 时间约束

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)
            envelope.time_constraint = constraint
            self._cache.set(session_id, envelope)

    # ================================================================
    # 遗忘衰减刷新 (设计文档 3.4)
    # ================================================================

    def refresh_context(self, session_id: str) -> None:
        """刷新上下文: 重算所有 KC 的遗忘衰减系数 (设计文档 3.4).

        借鉴 FSRS: 基于当前时间重新计算每个 MasterySnapshot 的 decay_factor.
        刷新后更新信封时间戳.

        Args:
            session_id: 会话 ID

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)
            now_ms = int(time.time() * 1000)
            DecayEngine.refresh_all_decay(envelope, current_ts=now_ms)
            envelope.timestamp = now_ms
            self._cache.set(session_id, envelope)

    # ================================================================
    # 薄弱知识点查询
    # ================================================================

    def get_weak_kcs(
        self,
        session_id: str,
        threshold: float = WEAK_THRESHOLD,
    ) -> list[str]:
        """获取薄弱知识点 ID 列表 (设计文档 3.3).

        有效掌握度 = p_know × decay_factor, 低于阈值视为薄弱.

        Args:
            session_id: 会话 ID
            threshold: 薄弱阈值 (默认 0.5)

        Returns:
            薄弱知识点 kc_id 列表

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)
            return envelope.get_weak_kcs(threshold)

    # ================================================================
    # 跨会话上下文传递 (设计文档 3.5)
    # ================================================================

    def transfer_context(
        self,
        source_session_id: str,
        target_session_id: str,
        user_id: str,
    ) -> ContextEnvelope:
        """跨会话上下文传递 (设计文档 3.5).

        借鉴 Redis Long-term Memory: 会话间继承认知状态.

        传递内容:
        1. 知识掌握度: 深拷贝 mastery_snapshot (含 BKT 参数)
        2. 学习目标: 未完成目标自动继承
        3. 学习阶段: 回到 PREVIEW (新会话从头开始)

        Args:
            source_session_id: 源会话 ID
            target_session_id: 目标会话 ID
            user_id: 用户 ID

        Returns:
            目标会话的 ContextEnvelope

        Raises:
            ContextNotFoundError: 源会话不存在
        """
        with self._lock:
            source_envelope = self.get_envelope(source_session_id)

            # 构建目标信封
            target_envelope = ContextEnvelope(
                user_id=user_id,
                session_id=target_session_id,
                # 深拷贝掌握快照 (含 BKT 参数)
                mastery_snapshot=copy.deepcopy(source_envelope.mastery_snapshot),
                # 继承未完成目标
                goals=copy.deepcopy(source_envelope.goals),
                ttl=source_envelope.ttl,
            )
            # 学习阶段回到 PREVIEW (新会话从头开始)
            target_envelope.learning_state = LearningState()

            self._cache.set(target_session_id, target_envelope)
            return target_envelope

    # ================================================================
    # 会话管理
    # ================================================================

    def remove_session(self, session_id: str) -> None:
        """移除会话上下文 (清理缓存).

        Args:
            session_id: 会话 ID
        """
        with self._lock:
            self._cache.invalidate(session_id)

    def get_all_sessions(self) -> list[str]:
        """获取所有活跃会话 ID.

        Returns:
            会话 ID 列表
        """
        with self._lock:
            stats = self._cache.get_stats()
            # 从缓存中提取会话列表 (通过 stats 间接获取)
            # 由于 ContextCache 不直接暴露 keys, 使用内部接口
            return list(self._cache._session_cache.keys())

    def get_envelope_summary(self, session_id: str) -> dict[str, Any]:
        """获取脱敏摘要 (设计文档 3.3).

        用于 API 响应和日志记录, 不含原始学号等敏感信息.

        Args:
            session_id: 会话 ID

        Returns:
            摘要字典: {phase, cognitive_load, weak_kc_count, resource_count, has_time_constraint}

        Raises:
            ContextNotFoundError: 会话不存在
        """
        with self._lock:
            envelope = self.get_envelope(session_id)
            return envelope.to_summary()


# ============================================================
# __all__
# ============================================================

__all__ = [
    # 常量
    "CACHE_TTL_SESSION",
    "CACHE_TTL_MASTERY",
    "CACHE_TTL_GOAL",
    "CACHE_TTL_COGNITIVE",
    "COGNITIVE_LOAD_BASE",
    "COGNITIVE_LOAD_FAST_ANSWER_MS",
    "COGNITIVE_LOAD_ERROR_WEIGHT",
    "COGNITIVE_LOAD_HELP_WEIGHT",
    "REVIEW_URGENCY_THRESHOLD",
    # 异常
    "L1ContextError",
    "ContextNotFoundError",
    "ContextExpiredError",
    "ContextValidationError",
    "DecayError",
    # 事件
    "FrontendEvent",
    "AgentOutputEvent",
    "UserDeclaration",
    # 采集器
    "ContextCollector",
    # 衰减引擎
    "DecayEngine",
    # 缓存
    "ContextCache",
    # 核心引擎
    "LearningContextBroker",
]
