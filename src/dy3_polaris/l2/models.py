"""L2 个性化层内部数据模型.

定义 L2 层 (学情画像 / 知识追踪 / 能力评估 / 记忆 / 会话) 使用的核心 dataclass,
参考 L1 models.py 的实现风格: ``from __future__ import annotations`` + ``@dataclass``,
每个模型提供 ``to_dict()`` / ``from_dict()`` 往返序列化方法.

跨层对齐:
- AnswerRecord.kp_id          ↔ L1 MasterySnapshot.kc_id / L3 KPMastery.kp_id
- TracingState.bkt_params      ↔ L1 BKTParams (此处使用 p_l0/p_t/p_g/p_s 命名)
- LearnerSnapshot              ↔ L1 ContextEnvelope (L2 向 L1 回传画像的载体)
- IRTState.theta               ↔ L1 IRTAbility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProfileConflictError(Exception):
    """画像乐观锁冲突 (L2 唯一写方 + 版本校验).

    当调用方基于陈旧 version 写入画像时抛出:
    - current_version: 存储中的最新版本
    - 调用方应重新拉取最新画像后重试
    """

    def __init__(self, learner_id: str, expected: int, current: int) -> None:
        self.learner_id = learner_id
        self.expected_version = expected
        self.current_version = current
        super().__init__(
            f"画像乐观锁冲突: learner={learner_id} expected_version={expected} "
            f"current_version={current} (请重新拉取最新画像后重试)"
        )


class FeedbackType(str, Enum):
    """统一反馈类型 (全系统单点, 收敛 L1/L4/L2/L5 四套枚举).

    跨通道聚合依据: 所有反馈通道 (HiTL/L4 决策/L5 Agent/学情行为) 使用同一枚举,
    便于画像 extras.feedback_log 统一消费.

    取值语义:
    - EXPLICIT_RATING : 用户显式评分
    - HUMAN_FEEDBACK  : 内容级人工反馈 (四态: understood/need_more/incorrect/report,
                        细分见 category 字段)
    - IMPLICIT_RESULT : 隐式行为/结果信号 (答对答错/停留/跳过等)
    - AGENT_OUTCOME   : Agent 执行结果反馈 (confidence/action_type)
    - SKIP            : 跳过/忽略
    """

    EXPLICIT_RATING = "explicit_rating"
    HUMAN_FEEDBACK = "human_feedback"
    IMPLICIT_RESULT = "implicit_result"
    AGENT_OUTCOME = "agent_outcome"
    SKIP = "skip"


# ============================================================
# 1. BKT 默认四参数 (TracingState.bkt_params 默认值)
# ============================================================

# p_l0: 先验掌握概率 P(Know)        (未学习前已掌握的概率)
# p_t : 学习转移概率 P(Transit)     (从未掌握到掌握的单次转移)
# p_g : 猜测概率   P(Guess)         (未掌握但答对)
# p_s : 失误概率   P(Slip)          (已掌握但答错)
DEFAULT_BKT_PARAMS: dict[str, float] = {
    "p_l0": 0.5,
    "p_t": 0.1,
    "p_g": 0.2,
    "p_s": 0.1,
}


# ============================================================
# 1.1 IRT 能力估计默认参数 (全系统单一事实来源)
# ============================================================

# 冷启动能力先验: θ ~ N(0, SE²).
# SE 需区分两个语义 (曾错误统一成 0.5, 导致冷启动 SE 不再随数据递减、单条默认漂移):
# - DEFAULT_INITIAL_SE = 0.3: 单条 IRTState 默认 SE + 空事件回退 + 冷启动观测端基准
#   (已有一个合理估计时的标准误; 过窄会过早锁定, 过宽会让前几题过度摆动).
# - 群体先验 SE = 0.5: 见 profile_builder.cold_start.POPULATION_SE (0 记录、能力完全未知时的较大不确定性).
DEFAULT_INITIAL_THETA: float = 0.0
DEFAULT_INITIAL_SE: float = 0.3

# 题目参数默认值 (difficulty [0,1] -> IRT b [-3,3] 的映射系数由各调用点给出).
DEFAULT_IRT_A: float = 1.2    # 区分度
DEFAULT_IRT_C: float = 0.25   # 猜测下限 (4 选 1)

# 能力估计 θ 的合理边界 (网格搜索范围外钳制, 防止极端异常值漂移).
IRT_THETA_MIN: float = -4.0
IRT_THETA_MAX: float = 4.0


# ============================================================
# 2. 答题记录 (AnswerRecord)
# ============================================================


@dataclass
class AnswerRecord:
    """单次答题记录 (L2 知识追踪 / IRT 估计的输入信号).

    Attributes:
        learner_id: 学习者 ID
        kp_id: 知识点 ID
        correct: 是否答对 (严格 bool 类型)
        timestamp: 答题时间戳 (秒, float)
        difficulty: 题目难度 [0.0, 1.0], 默认 0.5
        question_id: 题目 ID (可选)
        response_time: 答题响应时间 (秒, float, 可选); 用于 DKT 启发的序列特征工程,
            未采集时为 None. 新增字段, 默认 None, 向后兼容旧调用.
    """

    learner_id: str
    kp_id: str
    correct: bool
    timestamp: float
    difficulty: float = 0.5
    question_id: str | None = None
    response_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "learner_id": self.learner_id,
            "kp_id": self.kp_id,
            "correct": self.correct,
            "timestamp": self.timestamp,
            "difficulty": self.difficulty,
            "question_id": self.question_id,
            "response_time": self.response_time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnswerRecord:
        """从字典反序列化 (response_time 缺失时回退 None, 向后兼容)."""
        return cls(
            learner_id=d["learner_id"],
            kp_id=d["kp_id"],
            correct=d["correct"],
            timestamp=d["timestamp"],
            difficulty=d.get("difficulty", 0.5),
            question_id=d.get("question_id"),
            response_time=d.get("response_time"),
        )


# ============================================================
# 3. 知识追踪状态 (TracingState)
# ============================================================


@dataclass
class TracingState:
    """单个知识点的 BKT 追踪状态.

    Attributes:
        kp_id: 知识点 ID
        mastery_prob: 当前掌握概率 P(Know) [0.0, 1.0]
        attempts: 累计作答次数
        correct_count: 累计答对次数
        last_attempt_time: 上次作答时间戳 (秒, float)
        bkt_params: BKT 四参数字典, 默认含 p_l0/p_t/p_g/p_s
    """

    kp_id: str
    mastery_prob: float
    attempts: int = 0
    correct_count: int = 0
    last_attempt_time: float = 0.0
    bkt_params: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_BKT_PARAMS)
    )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (bkt_params 浅拷贝避免共享引用)."""
        return {
            "kp_id": self.kp_id,
            "mastery_prob": self.mastery_prob,
            "attempts": self.attempts,
            "correct_count": self.correct_count,
            "last_attempt_time": self.last_attempt_time,
            "bkt_params": dict(self.bkt_params),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TracingState:
        """从字典反序列化."""
        return cls(
            kp_id=d["kp_id"],
            mastery_prob=d["mastery_prob"],
            attempts=d.get("attempts", 0),
            correct_count=d.get("correct_count", 0),
            last_attempt_time=d.get("last_attempt_time", 0.0),
            bkt_params=d.get("bkt_params", dict(DEFAULT_BKT_PARAMS)),
        )


# ============================================================
# 4. IRT 能力状态 (IRTState)
# ============================================================


@dataclass
class IRTState:
    """学习者的 IRT (项目反应理论) 能力估计状态.

    Attributes:
        theta: 能力参数 θ (标准分尺度, 可正可负)
        se: 估计标准误 (standard error), 默认 0.3
        response_count: 已纳入估计的作答次数
        last_update_time: 上次更新时间戳 (秒, float)
    """

    theta: float
    se: float = DEFAULT_INITIAL_SE
    response_count: int = 0
    last_update_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "theta": self.theta,
            "se": self.se,
            "response_count": self.response_count,
            "last_update_time": self.last_update_time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IRTState:
        """从字典反序列化."""
        return cls(
            theta=d["theta"],
            se=d.get("se", DEFAULT_INITIAL_SE),
            response_count=d.get("response_count", 0),
            last_update_time=d.get("last_update_time", 0.0),
        )


# ============================================================
# 5. 学情画像快照 (LearnerSnapshot)
# ============================================================


@dataclass
class LearnerSnapshot:
    """学习者画像快照 (L2 → L1 回传 / 跨会话继承的载体).

    Attributes:
        learner_id: 学习者 ID
        snapshot_ts: 快照时间戳 (秒, float)
        kp_mastery: 知识点掌握度映射 {kp_id: mastery_prob} (已应用遗忘衰减)
        theta: IRT 能力参数 θ (画像初始化前可为 None)
        level: 学习者等级标签 (beginner/intermediate/advanced/novice ...)
        learning_style: VARK 学习风格 (visual/aural/reading/kinesthetic/multimodal)
        bloom_target: Bloom 认知目标层次 (remember/understand/apply/analyze/evaluate/create)
        weak_kps: 薄弱知识点列表 (mastery < weak_kps_threshold 的 kp_id)
        confidence: 画像置信度 [0.0, 1.0], 基于 IRT 标准误 SE 计算 (1/(1+se))
    """

    learner_id: str
    snapshot_ts: float
    kp_mastery: dict[str, float] = field(default_factory=dict)
    theta: float | None = None
    level: str = "beginner"
    learning_style: str = "reading"
    bloom_target: str = "understand"
    weak_kps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    #: 扩展记录 (Agent 写回: 决策轨迹/审核结果/考核记录/学习目标等)
    extras: dict[str, Any] = field(default_factory=dict)
    #: 版本号 (乐观并发控制: 合并写时递增, 检测丢失更新)
    version: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (可变字段浅拷贝避免共享引用)."""
        d: dict[str, Any] = {
            "learner_id": self.learner_id,
            "snapshot_ts": self.snapshot_ts,
            "kp_mastery": dict(self.kp_mastery),
            "theta": self.theta,
            "level": self.level,
            "learning_style": self.learning_style,
            "bloom_target": self.bloom_target,
            "weak_kps": list(self.weak_kps),
            "confidence": self.confidence,
            "extras": dict(self.extras),
            "version": self.version,
        }
        # 五维能力雷达: A理论/B应用/C合成/D表征 + E行为(态度), 补齐验收缺口
        try:
            from dy3_polaris.l2.kp_catalog import KP_DOMAIN_IDS
            dims: dict[str, float] = {}
            for dom, ids in KP_DOMAIN_IDS.items():
                vals = [self.kp_mastery.get(kp, 0.0) for kp in ids]
                dims[dom] = round(sum(vals) / len(vals), 4) if vals else 0.0
            activity = 0.0
            for key in ("query_log", "assess_log", "decision_log", "review_log"):
                logs = self.extras.get(key) or []
                activity += min(1.0, len(logs) / 10.0)
            dims["E"] = round(min(1.0, activity / 4.0), 4)
            d["dimensions"] = dims
        except Exception:  # noqa: BLE001
            pass
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearnerSnapshot:
        """从字典反序列化."""
        return cls(
            learner_id=d["learner_id"],
            snapshot_ts=d["snapshot_ts"],
            kp_mastery=d.get("kp_mastery", {}),
            theta=d.get("theta"),
            level=d.get("level", "beginner"),
            learning_style=d.get("learning_style", "reading"),
            bloom_target=d.get("bloom_target", "understand"),
            weak_kps=list(d.get("weak_kps", [])),
            confidence=d.get("confidence", 0.0),
            extras=dict(d.get("extras", {})),
            version=int(d.get("version", 0) or 0),
        )


# ============================================================
# 6. 会话记录 (SessionRecord)
# ============================================================


@dataclass
class SessionRecord:
    """L2 个性化会话记录.

    记录一次个性化学习会话的状态与检查点, 用于会话恢复与上下文继承.

    Attributes:
        session_id: 会话 ID
        learner_id: 学习者 ID
        started_at: 会话开始时间戳 (秒, float)
        status: 会话状态 (active/paused/closed ...), 默认 "active"
        context_envelope: 关联的上下文信封 (可选, 可为 None)
        checkpoints: 检查点列表, 默认空列表
    """

    session_id: str
    learner_id: str
    started_at: float
    status: str = "active"
    context_envelope: dict[str, Any] | None = None
    checkpoints: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (可变字段浅拷贝避免共享引用)."""
        return {
            "session_id": self.session_id,
            "learner_id": self.learner_id,
            "started_at": self.started_at,
            "status": self.status,
            "context_envelope": (
                dict(self.context_envelope)
                if self.context_envelope is not None
                else None
            ),
            "checkpoints": list(self.checkpoints),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionRecord:
        """从字典反序列化."""
        env = d.get("context_envelope")
        return cls(
            session_id=d["session_id"],
            learner_id=d["learner_id"],
            started_at=d["started_at"],
            status=d.get("status", "active"),
            context_envelope=dict(env) if env is not None else None,
            checkpoints=list(d.get("checkpoints", [])),
        )


# ============================================================
# __all__
# ============================================================

__all__ = [
    "DEFAULT_BKT_PARAMS",
    "DEFAULT_INITIAL_THETA",
    "DEFAULT_INITIAL_SE",
    "DEFAULT_IRT_A",
    "DEFAULT_IRT_C",
    "IRT_THETA_MIN",
    "IRT_THETA_MAX",
    "AnswerRecord",
    "TracingState",
    "IRTState",
    "LearnerSnapshot",
    "SessionRecord",
]
