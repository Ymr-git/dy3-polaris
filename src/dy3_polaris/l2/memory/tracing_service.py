"""三记忆 + 遗忘全链路编排服务 (增强版).

融合世界先进方案:
- Atkinson-Shiffrin (1968): 多重存储模型 — 感觉记忆 → 短期记忆 → 长期记忆
- Miller (1956): 工作记忆容量 7±2 (MAX_CHUNKS=9)
- Ebbinghaus (1885): 遗忘曲线 — 幂律衰减
- FSRS-6 (Free Spaced Repetition Scheduler v6): 完整 21 参数间隔重复调度
  - 幂律检索: R(t,S) = (1 + factor * t / S)^(-decay)
  - 稳定性更新: S' = S * (11-D)^w9 * S^(-w10) * (e^((1-R)*w11)-1) * hard_penalty * easy_bonus
  - 难度更新: D' = D - (grade-3)*(10-D)/9, 均值回归后 clamp [1, 10]
- Duolingo HLR (Settles & Meeder 2016): Half-Life Regression — 指数衰减 p = 2^(-Δt/h)
  - 技能级遗忘建模: h = 2^(θ·x), x 为特征向量
  - 技能强度计: strength = f(h, accuracy, reps)
- 记忆巩固 (Stickgold 2005, Diekelmann & Born 2010): 睡眠后弱记忆增强
  - 巩固增益与睡眠时长正相关, 与记忆强度负相关
  - 巩固增益随醒后时间衰减
- 干扰理论 (McGeoch 1932, Wixted 2004): 相似知识点间的 retroactive interference
  - 干扰强度与相似度成正比
  - 干扰随时间间隔增大而衰减
- PSI-KT (Qiu et al. ICLR 2024): 状态空间知识追踪 + 遗忘一体化
  - 状态转移: m' = exp(-α·Δt)·m + (1-exp(-α·Δt))·μ
  - 保持率: r = exp(-α·τ)
  - 自适应遗忘率: α = f(difficulty)
- SSP-MMC (Tabibian et al. NeurIPS 2017): 最优复习调度
  - 成本函数: cost = review_cost + forgetting_cost
  - 最优间隔平衡复习频率与遗忘风险
- BKT × 遗忘融合: P(known|t) = P(known) × R(t)

全链路处理流程:
1. 工作记忆写入: 从 AnswerEvent 创建 MemoryChunk 并添加到 WorkingMemory
2. LRU 迁移: 工作记忆超容量时淘汰最久未访问块, 迁移到 ShortTermMemory
3. FSRS 评分映射: correct + difficulty → grade (1-4: Again/Hard/Good/Easy)
4. FSRS 状态更新: 更新 stability / difficulty / retrievability (完整 FSRS-6 公式)
5. 干扰应用: 相似知识点的稳定性因 retroactive interference 降低
6. 长期迁移: 高重要度或重复曝光的条目持久化到 LongTermMemory (委托 L2Store)
7. BKT 融合: 计算遗忘修正后的掌握度 P(known|t) = P(known) × R(t)
8. 输出构建: 封装为 MemoryOutput 供下游 T2/T3/T5 消费

幂律遗忘模型 (FSRS-6):
    R(t) = (1 + factor * t / S)^(-decay)
    - decay = 0.5 (FSRS 衰减指数 Dw)
    - factor = 0.9^(-1/decay) - 1 (确保 R(t=S) = 0.9, 即 FSRS request_retention)
    - t: 自上次复习以来的天数 (elapsed_days)
    - S: 记忆稳定性 (stability)

指数遗忘模型 (HLR):
    p(Δt) = 2^(-Δt / h)
    - h: 半衰期 (half-life), 即 p=0.5 时的 Δt
    - Δt: 自上次练习以来的天数
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.memory.long_term_memory import LongTermMemory
from dy3_polaris.l2.memory.short_term_memory import ShortTermMemory
from dy3_polaris.l2.memory.working_memory import MAX_CHUNKS, MemoryChunk, WorkingMemory
from dy3_polaris.l2.models import AnswerRecord, TracingState
from dy3_polaris.l2.store import InMemoryL2Store, L2Store


# ============================================================
# 模块级 logger
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

# --- 幂律遗忘模型参数 ---
# 衰减指数 decay (FSRS 标准值 0.5, 对应负指数 -0.5)
DECAY: float = 0.5

# 因子 factor: 确保 R(t=S) = 0.9 (FSRS request_retention)
# 由 R = (1 + factor * t/S)^(-decay), 令 t=S 得 R = (1+factor)^(-decay) = 0.9
# => 1+factor = 0.9^(-1/decay) => factor = 0.9^(-1/decay) - 1
FACTOR: float = 0.9 ** (-1.0 / DECAY) - 1.0  # ≈ 0.2346

# 期望保留率 (FSRS request_retention)
REQUEST_RETENTION: float = 0.9

# 秒 -> 天换算
_SECONDS_PER_DAY: float = 86400.0

# --- FSRS 评分等级 ---
# 1 = Again (遗忘), 2 = Hard (困难), 3 = Good (正常), 4 = Easy (轻松)
GRADE_AGAIN: int = 1
GRADE_HARD: int = 2
GRADE_GOOD: int = 3
GRADE_EASY: int = 4

# --- 初始稳定性 (按 grade) ---
# 新卡片首次复习时的初始记忆稳定性 (天)
_INIT_STABILITY: dict[int, float] = {
    GRADE_AGAIN: 0.4,
    GRADE_HARD: 1.2,
    GRADE_GOOD: 3.0,
    GRADE_EASY: 7.0,
}

# --- 初始难度 (按 grade) ---
# 新卡片首次复习时的初始难度 [1, 10]
_INIT_DIFFICULTY: dict[int, float] = {
    GRADE_AGAIN: 8.5,
    GRADE_HARD: 7.0,
    GRADE_GOOD: 5.5,
    GRADE_EASY: 4.0,
}

# --- 难度调整系数 (FSRS w6 简化) ---
_DIFFICULTY_ADJUST: float = 1.0  # (grade-3) * (10-D)/9 的系数

# --- 难度均值回归目标 (FSRS initial_difficulty(4)) ---
_DIFFICULTY_MEAN_REVERT_TARGET: float = 5.5
_DIFFICULTY_MEAN_REVERT_WEIGHT: float = 0.1  # w7

# --- 遗忘后稳定性缩减因子 ---
_LAPSE_STABILITY_RATIO: float = 0.3

# --- 成功回忆的稳定性增益系数 ---
_HARD_PENALTY: float = 0.8  # grade=2 时的稳定性惩罚
_EASY_BONUS: float = 1.3  # grade=4 时的稳定性奖励

# --- 迁移阈值 ---
# 重要度阈值: importance >= 该值时迁移到长期记忆
DEFAULT_IMPORTANCE_THRESHOLD: float = 0.7

# 重复曝光阈值: reps >= 该值时迁移到长期记忆
DEFAULT_MIGRATION_REP_THRESHOLD: int = 3

# 长期记忆答题历史上限 (每 learner 最多保留最近 N 条, 避免无界增长)
_MAX_ANSWER_HISTORY: int = 500

# --- HLR 默认权重 (Duolingo HLR 简化版) ---
_HLR_WEIGHTS: dict[str, float] = {
    "correct_count": 0.10,
    "incorrect_count": -0.05,
    "total_reps": 0.02,
    "bias": 0.5,
}

# --- PSI-KT 默认参数 ---
_PSI_KT_DEFAULT_ALPHA: float = 0.1  # 默认遗忘率
_PSI_KT_ALPHA_MIN: float = 0.05    # 最小遗忘率
_PSI_KT_ALPHA_MAX: float = 0.25    # 最大遗忘率

# --- 干扰模型参数 ---
_INTERFERENCE_MAX_REDUCTION: float = 0.2   # 最大稳定性缩减比例
_INTERFERENCE_TIME_SCALE_DAYS: float = 30.0  # 干扰时间衰减尺度 (天)

# --- 巩固模型参数 ---
_CONSOLIDATION_SLEEP_SCALE: float = 4.0    # 睡眠饱和参数
_CONSOLIDATION_BOOST_SCALE: float = 0.1    # 巩固增益缩放
_CONSOLIDATION_DECAY_HOURS: float = 24.0   # 巩固增益衰减时间尺度 (小时)

# --- SSP-MMC 调度参数 ---
_SCHEDULING_REVIEW_COST_WEIGHT: float = 1.0   # 复习成本权重
_SCHEDULING_FORGETTING_COST_WEIGHT: float = 10.0  # 遗忘成本权重


# ============================================================
# FSRS-6 完整 21 参数模型
# ============================================================

# FSRS-6 参数 (w0-w20), 参考 open-spaced-repetition/fsrs4anki
_FSRS6_PARAMS: dict[str, float] = {
    # w0-w3: 初始稳定性 (按 grade 1-4)
    "w0": 0.4,    # Again 初始稳定性
    "w1": 1.2,    # Hard 初始稳定性
    "w2": 3.0,    # Good 初始稳定性
    "w3": 7.0,    # Easy 初始稳定性
    # w4-w5: 初始难度与难度衰减
    "w4": 5.5,    # 初始难度 (D0 = w4 - e^((G-1)*w5) + 1)
    "w5": 0.5,    # 难度衰减系数
    # w6-w7: 难度调整与均值回归
    "w6": 1.0,    # 难度调整系数 (grade-3)*(10-D)/9
    "w7": 0.1,    # 难度均值回归权重
    # w8-w11: 稳定性更新公式系数
    "w8": 1.0,    # 稳定性基础乘子
    "w9": 0.5,    # 稳定性难度指数: (11-D)^w9
    "w10": 0.5,   # 稳定性活动指数: S^(-w10)
    "w11": 0.5,   # 稳定性可提取性因子: (e^((1-R)*w11)-1)
    # w12-w13: Hard penalty / Easy bonus
    "w12": 0.8,   # Hard penalty
    "w13": 1.3,   # Easy bonus
    # w14-w16: 遗忘 (lapse) 处理
    "w14": 0.3,   # 遗忘后稳定性缩减比例
    "w15": 0.2,   # 遗忘稳定性基础
    "w16": 0.3,   # 遗忘稳定性衰减
    # w17-w20: 约束与配置
    "w17": 1.0,   # 难度下限
    "w18": 10.0,  # 难度上限
    "w19": 0.9,   # request_retention
    "w20": 0.5,   # decay
    # 便捷别名
    "decay": DECAY,
    "request_retention": REQUEST_RETENTION,
}


# ============================================================
# MemoryOutput — 下游输出标准化契约
# ============================================================


@dataclass
class MemoryOutput:
    """三记忆全链路输出 — 标准化记忆状态契约.

    供下游 T2 (知识追踪) / T3 (上下文代理) / T5 (会话管理) 消费.

    Attributes:
        learner_id: 学习者 ID
        working_memory_size: 当前工作记忆中的信息块数量
        short_term_count: 该学习者在短期记忆中的条目数
        long_term_persisted: 本次处理是否将数据持久化到长期记忆
        fsrs_next_review_days: FSRS 调度的下次复习间隔 (天)
        retrievability: 当前记忆可提取性 R [0.0, 1.0]
        stability: 记忆稳定性 S (天)
        difficulty: 条目难度 D [1.0, 10.0]
        review_grade: FSRS 评分 1-4 (Again/Hard/Good/Easy)
        migration_events: 本次处理发生的迁移事件列表
        last_updated_ts: 最后更新时间戳 (秒, float)
        mastery_with_forgetting: 遗忘修正后的掌握度 P(known|t) = P(known)×R(t) [0.0, 1.0]
    """

    learner_id: str
    working_memory_size: int
    short_term_count: int
    long_term_persisted: bool
    fsrs_next_review_days: int
    retrievability: float
    stability: float
    difficulty: float
    review_grade: int
    migration_events: list[str] = field(default_factory=list)
    last_updated_ts: float = 0.0
    mastery_with_forgetting: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (migration_events 浅拷贝)."""
        return {
            "learner_id": self.learner_id,
            "working_memory_size": self.working_memory_size,
            "short_term_count": self.short_term_count,
            "long_term_persisted": self.long_term_persisted,
            "fsrs_next_review_days": self.fsrs_next_review_days,
            "retrievability": self.retrievability,
            "stability": self.stability,
            "difficulty": self.difficulty,
            "review_grade": self.review_grade,
            "migration_events": list(self.migration_events),
            "last_updated_ts": self.last_updated_ts,
            "mastery_with_forgetting": self.mastery_with_forgetting,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryOutput:
        """从字典反序列化."""
        return cls(
            learner_id=d["learner_id"],
            working_memory_size=d["working_memory_size"],
            short_term_count=d["short_term_count"],
            long_term_persisted=d["long_term_persisted"],
            fsrs_next_review_days=d["fsrs_next_review_days"],
            retrievability=d["retrievability"],
            stability=d["stability"],
            difficulty=d["difficulty"],
            review_grade=d["review_grade"],
            migration_events=list(d.get("migration_events", [])),
            last_updated_ts=d.get("last_updated_ts", 0.0),
            mastery_with_forgetting=d.get("mastery_with_forgetting", 0.0),
        )

    def to_api_response(self) -> dict[str, Any]:
        """转换为 API 响应格式 (RESTful JSON 兼容).

        Returns:
            包含核心记忆状态的字典, 适合直接序列化为 JSON.
        """
        return {
            "learner_id": self.learner_id,
            "retrievability": round(self.retrievability, 6),
            "stability": round(self.stability, 4),
            "difficulty": round(self.difficulty, 4),
            "next_review_days": self.fsrs_next_review_days,
            "mastery": round(self.mastery_with_forgetting, 6),
            "review_grade": self.review_grade,
            "last_updated_ts": self.last_updated_ts,
        }

    @classmethod
    def from_bkt_output(cls, bkt_output: Any) -> MemoryOutput:
        """从 BKT MasteryOutput 构建 MemoryOutput.

        将 BKT 掌握度通过遗忘曲线修正, 生成记忆感知的输出.

        Args:
            bkt_output: BKT 全链路输出 (MasteryOutput 实例).

        Returns:
            MemoryOutput 实例 (含遗忘修正掌握度).
        """
        # 从 BKT 输出提取掌握度 (兼容 p_mastery 和 mastery_prob 两种字段名)
        mastery = getattr(bkt_output, "p_mastery", None)
        if mastery is None:
            mastery = getattr(bkt_output, "mastery_prob", 0.5)

        last_ts = getattr(bkt_output, "last_updated_ts", time.time())

        # 使用默认稳定性计算遗忘衰减
        default_S = 3.0
        elapsed_days = max(0.0, (time.time() - last_ts) / _SECONDS_PER_DAY)
        r = _compute_retrievability_static(elapsed_days, default_S)
        mastery_with_forgetting = mastery * r

        return cls(
            learner_id=bkt_output.learner_id,
            working_memory_size=0,
            short_term_count=0,
            long_term_persisted=False,
            fsrs_next_review_days=max(1, round(default_S)),
            retrievability=r,
            stability=default_S,
            difficulty=5.5,
            review_grade=GRADE_GOOD,
            migration_events=[],
            last_updated_ts=last_ts,
            mastery_with_forgetting=mastery_with_forgetting,
        )


# ============================================================
# 静态工具函数
# ============================================================


def _compute_retrievability_static(elapsed_days: float, stability: float) -> float:
    """计算可提取性 R (幂律遗忘曲线) — 静态函数版本.

        R(t) = (1 + factor * t / S)^(-decay)

    Args:
        elapsed_days: 自上次复习以来的天数.
        stability: 记忆稳定性 (天).

    Returns:
        可提取性 [0.0, 1.0].
    """
    if elapsed_days <= 0.0 or stability <= 0.0:
        return 1.0
    r = (1.0 + FACTOR * elapsed_days / stability) ** (-DECAY)
    return max(0.0, min(1.0, r))


# ============================================================
# MemoryTracingService — 全链路编排器
# ============================================================


class MemoryTracingService:
    """三记忆 + 遗忘全链路编排服务 (增强版).

    编排全链路处理流程:
    1. 工作记忆写入 (MemoryChunk → WorkingMemory)
    2. LRU 淘汰迁移 (WorkingMemory 溢出 → ShortTermMemory)
    3. FSRS 评分映射 (AnswerEvent → grade 1-4)
    4. FSRS-6 状态更新 (stability / difficulty / retrievability)
    5. 干扰应用 (相似 KP 的 retroactive interference)
    6. 长期记忆迁移 (高重要度 / 重复曝光 → LongTermMemory 持久化)
    7. BKT 融合 (遗忘修正掌握度 P(known|t) = P(known)×R(t))
    8. 输出构建 (MemoryOutput)

    增强特性:
    - 完整 FSRS-6: 21 参数模型, 幂律检索, 稳定性更新
    - Duolingo HLR: Half-Life Regression 技能级遗忘
    - 记忆巩固: 睡眠/静息后弱记忆增强
    - 干扰建模: 相似知识点间的 retroactive interference
    - PSI-KT: 状态空间知识追踪 + 遗忘一体化
    - SSP-MMC: 最优复习调度
    - BKT 融合: 遗忘修正后的掌握度

    Args:
        store: L2 存储层. 为 None 时使用内部 InMemoryL2Store.
        working_memory: 工作记忆实例 (可选, 默认新建).
        short_term_memory: 短期记忆实例 (可选, 默认新建).
        long_term_memory: 长期记忆实例 (可选, 默认新建并委托 store).
        importance_threshold: 迁移到长期记忆的重要度阈值, 默认 0.7.
        migration_rep_threshold: 迁移到长期记忆的重复曝光阈值, 默认 3.
    """

    def __init__(
        self,
        store: L2Store | None = None,
        working_memory: WorkingMemory | None = None,
        short_term_memory: ShortTermMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        importance_threshold: float = DEFAULT_IMPORTANCE_THRESHOLD,
        migration_rep_threshold: int = DEFAULT_MIGRATION_REP_THRESHOLD,
    ) -> None:
        self.store: L2Store = store if store is not None else InMemoryL2Store()
        self.working_memory: WorkingMemory = (
            working_memory if working_memory is not None else WorkingMemory()
        )
        self.short_term_memory: ShortTermMemory = (
            short_term_memory if short_term_memory is not None else ShortTermMemory()
        )
        self.long_term_memory: LongTermMemory = (
            long_term_memory if long_term_memory is not None else LongTermMemory(store=self.store)
        )
        self.importance_threshold: float = importance_threshold
        self.migration_rep_threshold: int = migration_rep_threshold

        # FSRS 状态: (learner_id, kp_id) -> {stability, difficulty, reps, lapses,
        #                                     last_review_ts, state, correct_count}
        self._fsrs_states: dict[tuple[str, str], dict[str, Any]] = {}

        # 线程安全锁: 保护 _fsrs_states/_psi_kt_states/_similarities/_sleep_records
        # (异步 memory_update 端点并发调用 process 时避免 read-modify-write 竞态)
        self._lock = threading.RLock()

        # 信息块 ID 计数器 (保证唯一性)
        self._chunk_counter: int = 0

        # --- 增强功能: 干扰模型 ---
        # 知识点相似度注册: (kp_a, kp_b) -> similarity [0, 1]
        self._similarities: dict[tuple[str, str], float] = {}

        # --- 增强功能: PSI-KT 状态 ---
        # (learner_id, kp_id) -> {m: float, alpha: float, last_update_ts: float}
        self._psi_kt_states: dict[tuple[str, str], dict[str, Any]] = {}

        # --- 增强功能: 巩固模型 ---
        # (learner_id,) -> {last_sleep_ts: float, sleep_hours: float}
        self._sleep_records: dict[str, dict[str, float]] = {}

    # ============================================================
    # 公开接口 — 核心处理
    # ============================================================

    # --- 单事件处理 ---

    def process(self, event: AnswerEvent) -> MemoryOutput:
        """处理单条答题事件, 返回完整 MemoryOutput.

        全链路流程:
        1. 创建 MemoryChunk 并添加到工作记忆
        2. 工作记忆超容量时 LRU 淘汰, 迁移到短期记忆
        3. 从事件确定 FSRS 评分 (grade 1-4)
        4. 更新 FSRS-6 卡片状态 (stability / difficulty / retrievability)
        5. 应用 retroactive interference (相似 KP 稳定性降低)
        6. 更新 PSI-KT 状态
        7. 检查是否迁移到长期记忆 (重要度或重复曝光)
        8. 计算 BKT 融合掌握度 P(known|t) = P(known)×R(t)
        9. 构建 MemoryOutput

        Args:
            event: 答题事件.

        Returns:
            MemoryOutput 标准化输出.
        """
        # 线程安全: 全链路状态读写加锁, 避免异步并发竞态
        with self._lock:
            return self._process_locked(event)

    def _process_locked(self, event: AnswerEvent) -> MemoryOutput:
        learner_id = event.learner_id
        kp_id = event.kp_id
        migration_events: list[str] = []

        # --- 1. 创建信息块并添加到工作记忆 ---
        difficulty_clamped = max(0.0, min(1.0, event.difficulty))
        chunk = MemoryChunk(
            chunk_id=f"chunk_{self._chunk_counter}",
            content=f"{kp_id}:{event.correct}:{difficulty_clamped:.2f}",
            chunk_type="answer",
            timestamp=event.timestamp,
            importance=difficulty_clamped,
        )
        self._chunk_counter += 1
        migration_events.extend(
            self._add_to_working_memory(chunk, learner_id)
        )

        # --- 2. 确定 FSRS 评分 ---
        grade = self._determine_grade(event)

        # --- 3. 更新 FSRS 状态 ---
        key = (learner_id, kp_id)
        existing = self._fsrs_states.get(key)

        if existing is None:
            # 新卡片
            fsrs_state = self._init_fsrs_state(grade, event.timestamp)
            retrievability = 1.0  # 刚学习, 完全可提取
        else:
            # 已有卡片: 先计算衰减后的可提取性 (pre-update)
            elapsed_days = max(
                0.0,
                (event.timestamp - existing["last_review_ts"]) / _SECONDS_PER_DAY,
            )
            retrievability = self._compute_retrievability(
                elapsed_days, existing["stability"]
            )
            fsrs_state = self._update_fsrs_state(
                existing, grade, event.timestamp, retrievability
            )

        # 累计作答次数与答对次数
        fsrs_state["reps"] = fsrs_state.get("reps", 0) + 1
        if event.correct:
            fsrs_state["correct_count"] = fsrs_state.get("correct_count", 0) + 1

        self._fsrs_states[key] = fsrs_state

        # --- 4. 应用 retroactive interference ---
        self._apply_interference(learner_id, kp_id, event.timestamp)

        # --- 5. 更新 PSI-KT 状态 ---
        self._update_psi_kt_state(learner_id, kp_id, event, fsrs_state)

        # --- 6. 检查迁移到长期记忆 ---
        long_term_persisted = False
        if (
            chunk.importance >= self.importance_threshold
            or fsrs_state["reps"] >= self.migration_rep_threshold
        ):
            self._persist_to_long_term(learner_id, kp_id, fsrs_state, event)
            long_term_persisted = True
            migration_events.append("short_term_to_long_term")

        # --- 7. 计算 BKT 融合掌握度 ---
        mastery_raw = fsrs_state["stability"] / (fsrs_state["stability"] + 1.0)
        mastery_with_forgetting = mastery_raw * retrievability

        # --- 8. 构建输出 ---
        next_review_days = self._compute_interval(fsrs_state["stability"])

        return MemoryOutput(
            learner_id=learner_id,
            working_memory_size=self.working_memory.get_size(),
            short_term_count=len(self.short_term_memory.get_entries(learner_id)),
            long_term_persisted=long_term_persisted,
            fsrs_next_review_days=next_review_days,
            retrievability=retrievability,
            stability=fsrs_state["stability"],
            difficulty=fsrs_state["difficulty"],
            review_grade=grade,
            migration_events=migration_events,
            last_updated_ts=event.timestamp,
            mastery_with_forgetting=mastery_with_forgetting,
        )

    # --- 批量处理 ---

    def batch_process(
        self,
        events: list[AnswerEvent],
    ) -> list[MemoryOutput]:
        """批量处理答题事件 (按时间戳升序).

        Args:
            events: 答题事件列表.

        Returns:
            每个事件对应的 MemoryOutput 列表.
        """
        if not events:
            return []
        ordered = sorted(events, key=lambda e: e.timestamp)
        return [self.process(ev) for ev in ordered]

    # ============================================================
    # 公开接口 — FSRS-6 完整模型
    # ============================================================

    def compute_retrievability(
        self, elapsed_days: float, stability: float
    ) -> float:
        """计算可提取性 R (FSRS-6 幂律遗忘曲线).

            R(t) = (1 + factor * t / S)^(-decay)

        - t=0 时 R=1.0 (刚复习, 完全可提取)
        - t=S 时 R=0.9 (FSRS request_retention)
        - t→∞ 时 R→0 (完全遗忘)

        Args:
            elapsed_days: 自上次复习以来的天数.
            stability: 记忆稳定性 (天).

        Returns:
            可提取性 [0.0, 1.0].
        """
        return self._compute_retrievability(elapsed_days, stability)

    def get_fsrs_parameters(self) -> dict[str, float]:
        """获取 FSRS-6 完整 21 参数.

        Returns:
            参数字典, 包含 w0-w20 及便捷别名 (decay, request_retention).
        """
        return dict(_FSRS6_PARAMS)

    def compute_interval_from_retention(
        self, stability: float, desired_retention: float
    ) -> int:
        """从目标保持率反推复习间隔.

            t = S * (R^(-1/decay) - 1) / factor

        Args:
            stability: 记忆稳定性 (天).
            desired_retention: 目标保持率 [0.0, 1.0].

        Returns:
            复习间隔 (天), 最小为 1.
        """
        if desired_retention <= 0.0 or desired_retention >= 1.0:
            return max(1, round(stability))
        interval_factor = desired_retention ** (-1.0 / DECAY) - 1.0
        interval = stability * interval_factor / FACTOR
        return max(1, round(interval))

    # ============================================================
    # 公开接口 — Duolingo HLR
    # ============================================================

    def compute_hlr_retrievability(
        self, elapsed_days: float, half_life: float
    ) -> float:
        """计算 HLR 可提取性 (指数衰减).

            p(Δt) = 2^(-Δt / h)

        - Δt=0 时 p=1.0
        - Δt=h 时 p=0.5 (半衰期定义)
        - Δt=2h 时 p=0.25

        Args:
            elapsed_days: 自上次练习以来的天数.
            half_life: 半衰期 (天).

        Returns:
            可提取性 [0.0, 1.0].
        """
        if elapsed_days <= 0.0 or half_life <= 0.0:
            return 1.0
        p = 2.0 ** (-elapsed_days / half_life)
        return max(0.0, min(1.0, p))

    def estimate_half_life(self, learner_id: str, kp_id: str) -> float:
        """从答题序列估计 half-life.

        基于 FSRS 稳定性映射: 当幂律 R=0.5 时的天数即为 half-life.
            (1 + factor*t/S)^(-decay) = 0.5
            => t = S * (0.5^(-1/decay) - 1) / factor

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.

        Returns:
            半衰期 (天). 未知 KP 返回 1.0.
        """
        state = self._fsrs_states.get((learner_id, kp_id))
        if state is None:
            return 1.0
        S = max(state["stability"], 0.1)
        # 幂律 R=0.5 时的天数
        t_half = S * (0.5 ** (-1.0 / DECAY) - 1.0) / FACTOR
        return max(0.1, t_half)

    def predict_half_life(self, features: dict[str, Any]) -> float:
        """HLR 特征向量预测 half-life.

            h = 2^(θ·x)

        使用预定义权重对特征向量加权求和, 然后指数化.

        Args:
            features: 特征字典, 支持的键:
                correct_count, incorrect_count, total_reps.

        Returns:
            预测的半衰期 (天).
        """
        dot_product = _HLR_WEIGHTS.get("bias", 0.5)
        for key, weight in _HLR_WEIGHTS.items():
            if key == "bias":
                continue
            dot_product += weight * features.get(key, 0)
        return max(0.01, 2.0 ** dot_product)

    def compute_skill_strength(
        self, learner_id: str, kp_id: str
    ) -> float:
        """计算技能强度计 (Duolingo skill strength meter).

            strength = (h / (h + 5)) * 0.7 + accuracy * 0.3

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.

        Returns:
            技能强度 [0.0, 1.0].
        """
        state = self._fsrs_states.get((learner_id, kp_id))
        if state is None:
            return 0.0
        h = self.estimate_half_life(learner_id, kp_id)
        reps = state.get("reps", 0)
        correct = state.get("correct_count", 0)
        accuracy = correct / reps if reps > 0 else 0.0
        strength = (h / (h + 5.0)) * 0.7 + accuracy * 0.3
        return max(0.0, min(1.0, strength))

    # ============================================================
    # 公开接口 — 记忆巩固
    # ============================================================

    def compute_consolidation_boost(
        self,
        stability: float,
        sleep_hours: float,
        hours_since_sleep: float = 0.0,
    ) -> float:
        """计算记忆巩固增益.

        基于 Stickgold (2005) 和 Diekelmann & Born (2010) 的睡眠巩固研究:
        - 增益与睡眠时长正相关 (饱和效应)
        - 弱记忆 (低稳定性) 获得更大比例的增益
        - 增益随醒后时间衰减

            boost = S * sleep_factor * weak_factor * decay_factor * scale

        其中:
            sleep_factor = 1 - exp(-sleep_hours / scale_sleep)
            weak_factor = 1 / (1 + S * 0.1)
            decay_factor = exp(-hours_since_sleep / scale_decay)

        Args:
            stability: 记忆稳定性 (天).
            sleep_hours: 睡眠时长 (小时).
            hours_since_sleep: 距上次睡眠的小时数, 默认 0.

        Returns:
            巩固增益 (稳定性增量, 天).
        """
        if sleep_hours <= 0.0:
            return 0.0
        sleep_factor = 1.0 - math.exp(-sleep_hours / _CONSOLIDATION_SLEEP_SCALE)
        weak_factor = 1.0 / (1.0 + stability * 0.1)
        decay_factor = math.exp(-hours_since_sleep / _CONSOLIDATION_DECAY_HOURS)
        boost = (
            stability
            * sleep_factor
            * weak_factor
            * decay_factor
            * _CONSOLIDATION_BOOST_SCALE
        )
        return boost

    def apply_consolidation(
        self,
        learner_id: str,
        kp_id: str,
        sleep_hours: float,
        current_time: float | None = None,
    ) -> None:
        """应用睡眠巩固增益到指定知识点的稳定性.

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.
            sleep_hours: 睡眠时长 (小时).
            current_time: 当前时间戳 (秒), 默认 time.time().
        """
        if current_time is None:
            current_time = time.time()

        state = self._fsrs_states.get((learner_id, kp_id))
        if state is None:
            return

        S = state["stability"]
        boost = self.compute_consolidation_boost(S, sleep_hours)
        state["stability"] = S + boost
        # 修复: 睡眠巩固只提升稳定性, 不刷新 last_review_ts.
        # 原实现把睡眠时间当作复习时间, 会导致复习被无限推迟 (retrievability 被错误拉高).

        # 记录睡眠信息
        self._sleep_records[learner_id] = {
            "last_sleep_ts": current_time,
            "sleep_hours": sleep_hours,
        }

    # ============================================================
    # 公开接口 — 干扰模型
    # ============================================================

    def register_similarity(
        self, kp_a: str, kp_b: str, similarity: float
    ) -> None:
        """注册两个知识点之间的相似度 (双向).

        Args:
            kp_a: 知识点 A.
            kp_b: 知识点 B.
            similarity: 相似度 [0.0, 1.0].
        """
        sim = max(0.0, min(1.0, similarity))
        self._similarities[(kp_a, kp_b)] = sim
        self._similarities[(kp_b, kp_a)] = sim

    def _apply_interference(
        self,
        learner_id: str,
        new_kp_id: str,
        current_ts: float,
    ) -> None:
        """应用 retroactive interference: 学习新 KP 时降低相似 KP 的稳定性.

        干扰公式:
            interference = similarity * exp(-Δt / time_scale)
            S'[similar_kp] = S[similar_kp] * (1 - interference * max_reduction)

        其中 Δt 是相似 KP 最后复习到当前的时间差.

        Args:
            learner_id: 学习者 ID.
            new_kp_id: 新学习的知识点 ID.
            current_ts: 当前时间戳 (秒).
        """
        # 找到所有与新 KP 相似的已有 KP
        for (kp_a, kp_b), sim in self._similarities.items():
            if kp_a != new_kp_id:
                continue
            similar_kp = kp_b
            key = (learner_id, similar_kp)
            existing = self._fsrs_states.get(key)
            if existing is None:
                continue

            # 计算时间差 (天)
            delta_t_days = max(
                0.0,
                (current_ts - existing["last_review_ts"]) / _SECONDS_PER_DAY,
            )
            # 干扰随时间衰减
            time_decay = math.exp(-delta_t_days / _INTERFERENCE_TIME_SCALE_DAYS)
            interference = sim * time_decay
            reduction = interference * _INTERFERENCE_MAX_REDUCTION

            if reduction > 0:
                existing["stability"] = max(
                    0.1, existing["stability"] * (1.0 - reduction)
                )

    # ============================================================
    # 公开接口 — PSI-KT 状态空间知识追踪
    # ============================================================

    def compute_psi_kt_transition(
        self,
        m_current: float,
        alpha: float,
        delta_t: float,
        mu_target: float,
    ) -> float:
        """PSI-KT 状态转移.

            m' = exp(-α·Δt)·m + (1-exp(-α·Δt))·μ

        - 当 Δt→0 时 m'→m (无变化)
        - 当 Δt→∞ 时 m'→μ (趋向目标状态)

        Args:
            m_current: 当前知识状态 [0, 1].
            alpha: 遗忘率.
            delta_t: 时间间隔 (天).
            mu_target: 目标状态 (学习后的状态).

        Returns:
            转移后的知识状态 [0, 1].
        """
        decay = math.exp(-alpha * delta_t)
        m_new = decay * m_current + (1.0 - decay) * mu_target
        return max(0.0, min(1.0, m_new))

    def compute_psi_kt_retention(
        self, alpha: float, tau: float
    ) -> float:
        """PSI-KT 保持率.

            r = exp(-α·τ)

        Args:
            alpha: 遗忘率.
            tau: 时间间隔 (天).

        Returns:
            保持率 (0, 1].
        """
        return math.exp(-alpha * tau)

    def compute_adaptive_forgetting_rate(
        self, difficulty: float
    ) -> float:
        """计算自适应遗忘率 (难度高→遗忘快).

            α = α_min + difficulty * (α_max - α_min)

        Args:
            difficulty: 难度 [0.0, 1.0].

        Returns:
            遗忘率 α ∈ [α_min, α_max].
        """
        d = max(0.0, min(1.0, difficulty))
        return _PSI_KT_DEFAULT_ALPHA + d * (_PSI_KT_ALPHA_MAX - _PSI_KT_ALPHA_MIN)

    def compute_psi_kt_predict(
        self, m: float, a: float, b: float
    ) -> float:
        """PSI-KT 预测答对概率.

            p = sigmoid(a·(m - b)) = 1 / (1 + e^(-a·(m-b)))

        Args:
            m: 知识状态 [0, 1].
            a: 区分度.
            b: 难度.

        Returns:
            答对概率 [0, 1].
        """
        return 1.0 / (1.0 + math.exp(-a * (m - b)))

    def get_psi_kt_state(
        self, learner_id: str, kp_id: str
    ) -> float:
        """获取 PSI-KT 知识状态.

        从 FSRS 稳定性映射: m = S / (S + 1) ∈ [0, 1).

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.

        Returns:
            知识状态 m ∈ [0, 1). 未知 KP 返回 0.0.
        """
        psi_state = self._psi_kt_states.get((learner_id, kp_id))
        if psi_state is not None:
            return psi_state["m"]

        # 回退到 FSRS 稳定性映射
        fsrs_state = self._fsrs_states.get((learner_id, kp_id))
        if fsrs_state is None:
            return 0.0
        S = fsrs_state["stability"]
        return S / (S + 1.0)

    def _update_psi_kt_state(
        self,
        learner_id: str,
        kp_id: str,
        event: AnswerEvent,
        fsrs_state: dict[str, Any],
    ) -> None:
        """更新 PSI-KT 状态.

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.
            event: 答题事件.
            fsrs_state: 当前 FSRS 状态.
        """
        key = (learner_id, kp_id)
        existing = self._psi_kt_states.get(key)

        # 从 FSRS 稳定性映射当前状态
        S = fsrs_state["stability"]
        m_current = S / (S + 1.0)

        # 计算遗忘率
        alpha = self.compute_adaptive_forgetting_rate(event.difficulty)

        if existing is not None:
            # 状态转移
            delta_t = max(
                0.0,
                (event.timestamp - existing["last_update_ts"]) / _SECONDS_PER_DAY,
            )
            mu_target = m_current if event.correct else m_current * 0.5
            m_new = self.compute_psi_kt_transition(
                existing["m"], alpha, delta_t, mu_target
            )
        else:
            m_new = m_current

        self._psi_kt_states[key] = {
            "m": m_new,
            "alpha": alpha,
            "last_update_ts": event.timestamp,
        }

    # ============================================================
    # 公开接口 — SSP-MMC 最优调度
    # ============================================================

    def compute_optimal_interval(
        self, stability: float, difficulty: float
    ) -> int:
        """计算 SSP-MMC 最优复习间隔.

        基于随机最短路径 - 最小均成本 (SSP-MMC) 策略:
        - 高稳定性 → 更长间隔 (降低复习成本)
        - 高难度 → 更短间隔 (降低遗忘风险)

            interval = S * (1 + (10 - D) / 20)

        Args:
            stability: 记忆稳定性 (天).
            difficulty: 难度 [1, 10].

        Returns:
            最优复习间隔 (天), 最小为 1.
        """
        S = max(0.1, stability)
        D = max(1.0, min(10.0, difficulty))
        # 难度因子: D=10 → 1.0, D=1 → 1.45
        difficulty_factor = 1.0 + (10.0 - D) / 20.0
        interval = S * difficulty_factor
        return max(1, round(interval))

    def compute_scheduling_cost(
        self,
        stability: float,
        difficulty: float,
        interval: float,
    ) -> float:
        """计算 SSP-MMC 调度成本函数.

            cost = review_cost + forgetting_cost

        - review_cost = weight_review / interval (更频繁复习 → 更高成本)
        - forgetting_cost = weight_forgetting * (1 - R) * D

        Args:
            stability: 记忆稳定性 (天).
            difficulty: 难度 [1, 10].
            interval: 待评估的复习间隔 (天).

        Returns:
            总成本 (> 0).
        """
        R = self._compute_retrievability(interval, stability)
        review_cost = _SCHEDULING_REVIEW_COST_WEIGHT / max(interval, 0.01)
        forgetting_cost = (
            _SCHEDULING_FORGETTING_COST_WEIGHT * (1.0 - R) * difficulty
        )
        return review_cost + forgetting_cost

    def get_review_queue(
        self,
        learner_id: str,
        current_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """获取最优复习队列 (按紧急度排序).

        紧急度 = 1 - retrievability (可提取性越低越紧急).

        Args:
            learner_id: 学习者 ID.
            current_time: 当前时间戳 (秒), 默认 time.time().

        Returns:
            复习队列列表, 每项含 kp_id / retrievability / urgency /
            next_review_days / stability / difficulty.
            按可提取性升序排列 (最低的在前 = 最紧急).
        """
        if current_time is None:
            current_time = time.time()

        queue: list[dict[str, Any]] = []
        for (lid, kpid), state in self._fsrs_states.items():
            if lid != learner_id:
                continue
            elapsed_days = max(
                0.0,
                (current_time - state["last_review_ts"]) / _SECONDS_PER_DAY,
            )
            r = self._compute_retrievability(
                elapsed_days, state["stability"]
            )
            # 修复: 使用 SSP-MMC 最优间隔 (含难度因子), 而非退化到 stability 的 _compute_interval
            next_review = self.compute_optimal_interval(
                state["stability"], state["difficulty"]
            )
            # 调度成本 (SSP-MMC): 复习成本 + 遗忘成本, 用于更准确的紧急度排序
            scheduling_cost = self.compute_scheduling_cost(
                state["stability"], state["difficulty"], max(next_review, 0.01)
            )
            queue.append({
                "kp_id": kpid,
                "retrievability": r,
                "urgency": 1.0 - r,
                "next_review_days": next_review,
                "stability": state["stability"],
                "difficulty": state["difficulty"],
                "scheduling_cost": round(scheduling_cost, 6),
            })

        # 按可提取性升序 (最紧急的在前) — 若可提取性相同, 按调度成本降序 (遗忘风险高者优先)
        queue.sort(key=lambda x: (x["retrievability"], -x["scheduling_cost"]))
        return queue

    # ============================================================
    # 公开接口 — BKT × 遗忘融合
    # ============================================================

    def get_mastery_with_forgetting(
        self,
        learner_id: str,
        kp_id: str,
        current_time: float,
    ) -> float:
        """获取遗忘修正后的掌握度.

            P(known|t) = P(known) × R(t)

        其中 P(known) = S / (S + 1) (从稳定性映射),
        R(t) 为幂律遗忘曲线计算的可提取性.

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.
            current_time: 当前时间戳 (秒).

        Returns:
            遗忘修正后的掌握度 [0.0, 1.0]. 未知 KP 返回 0.0.
        """
        state = self.get_fsrs_state(learner_id, kp_id, current_time)
        if state is None:
            return 0.0
        S = state["stability"]
        R = state["retrievability"]
        mastery_raw = S / (S + 1.0)
        return mastery_raw * R

    def fusion_mastery_from_bkt(
        self,
        bkt_state: TracingState,
        current_time: float,
    ) -> float:
        """从 BKT TracingState 构建遗忘修正掌握度.

            P(known|t) = P(known) × R(t)

        使用默认稳定性 (3.0 天) 计算遗忘衰减.

        Args:
            bkt_state: BKT 追踪状态.
            current_time: 当前时间戳 (秒).

        Returns:
            遗忘修正后的掌握度 [0.0, 1.0].
        """
        mastery = bkt_state.mastery_prob
        elapsed_days = max(
            0.0,
            (current_time - bkt_state.last_attempt_time) / _SECONDS_PER_DAY,
        )
        # 使用默认稳定性
        default_S = 3.0
        R = self._compute_retrievability(elapsed_days, default_S)
        return mastery * R

    # ============================================================
    # 公开接口 — 质量度量
    # ============================================================

    def compute_coverage(self, all_kps: list[str]) -> float:
        """计算知识点覆盖率.

        Args:
            all_kps: 所有知识点 ID 列表.

        Returns:
            覆盖率 [0.0, 1.0] = 已追踪数量 / 总数量.
        """
        if not all_kps:
            return 0.0
        tracked = sum(
            1
            for kp in all_kps
            if any(kpid == kp for (_, kpid) in self._fsrs_states)
        )
        return tracked / len(all_kps)

    # ============================================================
    # 公开接口 — API 接口
    # ============================================================

    def get_review_schedule_api(
        self, learner_id: str
    ) -> list[dict[str, Any]]:
        """获取复习计划 API 接口.

        Args:
            learner_id: 学习者 ID.

        Returns:
            复习计划列表, 每项含 kp_id / next_review_days /
            retrievability / urgency.
        """
        queue = self.get_review_queue(learner_id)
        return [
            {
                "kp_id": item["kp_id"],
                "next_review_days": item["next_review_days"],
                "retrievability": round(item["retrievability"], 6),
                "urgency": round(item["urgency"], 6),
            }
            for item in queue
        ]

    def get_memory_snapshot_api(
        self, learner_id: str
    ) -> dict[str, Any]:
        """获取记忆快照 API 接口.

        Args:
            learner_id: 学习者 ID.

        Returns:
            含 working_memory_size / short_term_count / tracked_kps /
            avg_retrievability / review_queue_length 的字典.
        """
        snapshot = self.get_memory_snapshot(learner_id)
        fsrs_states = snapshot.get("fsrs_states", {})

        # 计算平均可提取性
        if fsrs_states:
            r_values = [s["retrievability"] for s in fsrs_states.values()]
            avg_r = sum(r_values) / len(r_values)
        else:
            avg_r = 0.0

        # 复习队列长度
        review_queue = self.get_review_queue(learner_id)

        return {
            "working_memory_size": snapshot["working_memory_size"],
            "short_term_count": snapshot["short_term_count"],
            "tracked_kps": list(fsrs_states.keys()),
            "avg_retrievability": round(avg_r, 6),
            "review_queue_length": len(review_queue),
        }

    # ============================================================
    # 公开接口 — 记忆快照 / FSRS 状态查询 / 复习调度
    # ============================================================

    def get_memory_snapshot(self, learner_id: str) -> dict[str, Any]:
        """获取学习者当前记忆状态快照.

        Args:
            learner_id: 学习者 ID.

        Returns:
            含 working_memory_size / short_term_count / fsrs_states / long_term_kps,
            以及前端记忆面板所需 kp_retentions / retentions 的字典.
        """
        fsrs_states: dict[str, dict[str, Any]] = {}
        for (lid, kpid), _ in self._fsrs_states.items():
            if lid == learner_id:
                state = self.get_fsrs_state(lid, kpid)
                if state is not None:
                    fsrs_states[kpid] = state

        # 前端记忆面板契约 (mf7-assistant.runMemory 读取 kp_retentions):
        # 每条含 retention (可提取性) 与 next_review_at (下次复习时间戳)
        kp_retentions: dict[str, dict[str, Any]] = {}
        for kpid, state in fsrs_states.items():
            stability = float(state.get("stability", 0.0))
            retrievability = float(state.get("retrievability", 0.0))
            last_review_ts = float(state.get("last_review_ts", 0.0) or 0.0)
            interval_days = self._compute_interval(stability) if stability > 0 else 1
            next_review_at = (
                last_review_ts + interval_days * _SECONDS_PER_DAY
                if last_review_ts > 0
                else 0.0
            )
            kp_retentions[kpid] = {
                "retention": round(retrievability, 4),
                "retrievability": round(retrievability, 4),
                "stability": round(stability, 4),
                "difficulty": round(float(state.get("difficulty", 0.0)), 4),
                "reps": int(state.get("reps", 0)),
                "lapses": int(state.get("lapses", 0)),
                "state": state.get("state", ""),
                "next_review_at": next_review_at,
                "next_review_days": interval_days,
            }

        return {
            "working_memory_size": self.working_memory.get_size(),
            "short_term_count": len(self.short_term_memory.get_entries(learner_id)),
            "fsrs_states": fsrs_states,
            "long_term_kps": list(fsrs_states.keys()),
            "kp_retentions": kp_retentions,
            "retentions": kp_retentions,
        }

    def get_memory_state(self, learner_id: str) -> dict[str, Any]:
        """集成桥接兼容别名 — 返回记忆快照."""
        return self.get_memory_snapshot(learner_id)

    def get_fsrs_state(
        self,
        learner_id: str,
        kp_id: str,
        current_time: float | None = None,
    ) -> dict[str, Any] | None:
        """获取指定知识点的 FSRS 状态.

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.
            current_time: 当前时间戳 (秒), 默认 time.time().

        Returns:
            FSRS 状态字典 (含 stability / difficulty / reps / lapses /
            last_review_ts / state / retrievability / correct_count),
            不存在返回 None.
        """
        state = self._fsrs_states.get((learner_id, kp_id))
        if state is None:
            return None

        if current_time is None:
            current_time = time.time()

        elapsed_days = max(
            0.0, (current_time - state["last_review_ts"]) / _SECONDS_PER_DAY
        )
        r = self._compute_retrievability(elapsed_days, state["stability"])

        return {
            "stability": state["stability"],
            "difficulty": state["difficulty"],
            "reps": state["reps"],
            "lapses": state["lapses"],
            "last_review_ts": state["last_review_ts"],
            "state": state["state"],
            "retrievability": r,
            "correct_count": state.get("correct_count", 0),
        }

    def schedule_review(self, learner_id: str, kp_id: str) -> int:
        """获取下次复习间隔 (天).

        基于当前记忆稳定性计算: 当可提取性降至 REQUEST_RETENTION (0.9) 时的天数.

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.

        Returns:
            下次复习间隔 (天), 最小为 1. 未知 KP 返回 1.
        """
        state = self._fsrs_states.get((learner_id, kp_id))
        if state is None:
            return 1
        return self._compute_interval(state["stability"])

    # ============================================================
    # 内部方法
    # ============================================================

    # --- 工作记忆写入 + LRU 迁移 ---

    def _add_to_working_memory(
        self,
        chunk: MemoryChunk,
        learner_id: str,
    ) -> list[str]:
        """添加信息块到工作记忆, 超容量时迁移 LRU 块到短期记忆.

        Args:
            chunk: 待添加的信息块.
            learner_id: 学习者 ID (用于短期记忆隔离).

        Returns:
            迁移事件列表 (可能包含 "working_to_short_term").
        """
        migration_events: list[str] = []

        if self.working_memory.is_full():
            # 捕获即将被淘汰的 LRU 块 (列表首部 = 最久未访问)
            context = self.working_memory.get_context()
            evicted = context[0] if context else None

            # 添加新块 (触发 LRU 淘汰)
            self.working_memory.add_chunk(chunk)

            # 迁移被淘汰的块到短期记忆
            if evicted is not None:
                self.short_term_memory.add({
                    "learner_id": learner_id,
                    "chunk_id": evicted.chunk_id,
                    "content": evicted.content,
                    "chunk_type": evicted.chunk_type,
                    "importance": evicted.importance,
                    "timestamp": evicted.timestamp,
                })
                migration_events.append("working_to_short_term")
        else:
            self.working_memory.add_chunk(chunk)

        return migration_events

    # --- FSRS 评分映射 ---

    @staticmethod
    def _determine_grade(event: AnswerEvent) -> int:
        """从答题事件确定 FSRS 评分 (1-4).

        映射规则:
        - correct=True,  difficulty<0.3  → grade 4 (Easy)
        - correct=True,  difficulty>=0.3 → grade 3 (Good)
        - correct=False, difficulty<0.7  → grade 2 (Hard)
        - correct=False, difficulty>=0.7 → grade 1 (Again)

        Args:
            event: 答题事件.

        Returns:
            FSRS 评分 1-4.
        """
        difficulty = max(0.0, min(1.0, event.difficulty))
        if event.correct:
            if difficulty < 0.3:
                return GRADE_EASY
            return GRADE_GOOD
        else:
            if difficulty < 0.7:
                return GRADE_HARD
            return GRADE_AGAIN

    # --- FSRS 初始化 ---

    @staticmethod
    def _init_fsrs_state(grade: int, timestamp: float) -> dict[str, Any]:
        """初始化新卡片的 FSRS 状态.

        Args:
            grade: FSRS 评分 1-4.
            timestamp: 当前时间戳 (秒).

        Returns:
            初始 FSRS 状态字典.
        """
        return {
            "stability": _INIT_STABILITY[grade],
            "difficulty": _INIT_DIFFICULTY[grade],
            "reps": 0,  # 将在 process() 中递增
            "lapses": 0,
            "last_review_ts": timestamp,
            "state": "learning",
            "correct_count": 0,
        }

    # --- FSRS 状态更新 ---

    @staticmethod
    def _update_fsrs_state(
        existing: dict[str, Any],
        grade: int,
        current_ts: float,
        retrievability: float,
    ) -> dict[str, Any]:
        """更新已有卡片的 FSRS 状态 (FSRS-6 公式).

        更新规则:
        - 难度: D' = D - (grade-3)*(10-D)/9, 均值回归后 clamp [1, 10]
        - 遗忘 (grade=1): S' = S * 0.3, lapses += 1
        - 成功 (grade>=2): S' = S * (1 + (11-D)/10 * R) * hard_penalty * easy_bonus

        Args:
            existing: 当前 FSRS 状态.
            grade: FSRS 评分 1-4.
            current_ts: 当前时间戳 (秒).
            retrievability: 更新前的可提取性 R [0, 1].

        Returns:
            更新后的新 FSRS 状态字典.
        """
        S = max(existing["stability"], 0.1)
        D = existing["difficulty"]

        # --- 难度更新 ---
        # D' = D - (grade-3) * (10-D) / 9
        next_d = D - _DIFFICULTY_ADJUST * (grade - 3) * (10.0 - D) / 9.0
        # 均值回归: D' = w7*target + (1-w7)*D'
        next_d = (
            _DIFFICULTY_MEAN_REVERT_WEIGHT * _DIFFICULTY_MEAN_REVERT_TARGET
            + (1.0 - _DIFFICULTY_MEAN_REVERT_WEIGHT) * next_d
        )
        next_d = max(1.0, min(10.0, next_d))

        # --- 稳定性更新 ---
        if grade == GRADE_AGAIN:
            # 遗忘: 稳定性大幅缩减
            new_s = S * _LAPSE_STABILITY_RATIO
            new_state = "relearning"
            lapses = existing["lapses"] + 1
        else:
            # 成功回忆: 稳定性增长
            hard_penalty = _HARD_PENALTY if grade == GRADE_HARD else 1.0
            easy_bonus = _EASY_BONUS if grade == GRADE_EASY else 1.0
            # S' = S * (1 + (11-D)/10 * R) * penalty * bonus
            new_s = S * (1.0 + (11.0 - next_d) / 10.0 * retrievability) * hard_penalty * easy_bonus
            new_state = "review"
            lapses = existing["lapses"]

        new_s = max(0.1, new_s)

        return {
            "stability": new_s,
            "difficulty": next_d,
            "reps": existing["reps"],  # 将在 process() 中递增
            "lapses": lapses,
            "last_review_ts": current_ts,
            "state": new_state,
            "correct_count": existing.get("correct_count", 0),
        }

    # --- 可提取性计算 (幂律遗忘) ---

    @staticmethod
    def _compute_retrievability(elapsed_days: float, stability: float) -> float:
        """计算可提取性 R (幂律遗忘曲线).

            R(t) = (1 + factor * t / S)^(-decay)

        - t=0 时 R=1.0 (刚复习, 完全可提取)
        - t=S 时 R=0.9 (FSRS request_retention)
        - t→∞ 时 R→0 (完全遗忘)

        Args:
            elapsed_days: 自上次复习以来的天数.
            stability: 记忆稳定性 (天).

        Returns:
            可提取性 [0.0, 1.0].
        """
        if elapsed_days <= 0.0 or stability <= 0.0:
            return 1.0
        r = (1.0 + FACTOR * elapsed_days / stability) ** (-DECAY)
        return max(0.0, min(1.0, r))

    # --- 复习间隔计算 ---

    @staticmethod
    def _compute_interval(stability: float) -> int:
        """计算下次复习间隔 (天).

        基于可提取性降至 REQUEST_RETENTION 的时间:
            t = S * (DR^(-1/decay) - 1) / factor

        由于 factor = DR^(-1/decay) - 1, 该式简化为 t = S.

        Args:
            stability: 记忆稳定性 (天).

        Returns:
            复习间隔 (天), 最小为 1.
        """
        interval_factor = REQUEST_RETENTION ** (-1.0 / DECAY) - 1.0
        interval = stability * interval_factor / FACTOR
        return max(1, round(interval))

    # --- 长期记忆持久化 ---

    def _persist_to_long_term(
        self,
        learner_id: str,
        kp_id: str,
        fsrs_state: dict[str, Any],
        event: AnswerEvent,
    ) -> None:
        """将 FSRS 状态和答题记录持久化到长期记忆 (委托 L2Store).

        Args:
            learner_id: 学习者 ID.
            kp_id: 知识点 ID.
            fsrs_state: 当前 FSRS 状态.
            event: 触发持久化的答题事件.
        """
        # 保存答题记录到 store (保留最近 _MAX_HISTORY 条, 避免无界增长)
        record = event.to_answer_record()
        history = self.long_term_memory.get_answer_history(learner_id)
        if history is None:
            history = []
        history.append(record)
        # 上限裁剪: 每个 learner 最多保留最近 _MAX_ANSWER_HISTORY 条答题记录
        if len(history) > _MAX_ANSWER_HISTORY:
            history = history[-_MAX_ANSWER_HISTORY:]
        self.long_term_memory.save_answer_history(learner_id, history)

        # 将 FSRS 状态映射为 TracingState 并持久化
        # mastery_prob 用稳定性映射: S/(S+1) ∈ (0, 1)
        mastery = fsrs_state["stability"] / (fsrs_state["stability"] + 1.0)
        tracing_state = TracingState(
            kp_id=kp_id,
            mastery_prob=mastery,
            attempts=fsrs_state["reps"],
            correct_count=fsrs_state.get("correct_count", 0),
            last_attempt_time=fsrs_state["last_review_ts"],
        )
        self.long_term_memory.save_tracing_state(learner_id, kp_id, tracing_state)


# ============================================================
# __all__
# ============================================================

__all__ = [
    "MemoryTracingService",
    "MemoryOutput",
    "DECAY",
    "FACTOR",
    "REQUEST_RETENTION",
    "DEFAULT_IMPORTANCE_THRESHOLD",
    "DEFAULT_MIGRATION_REP_THRESHOLD",
]
