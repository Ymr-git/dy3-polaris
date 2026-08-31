"""艾宾浩斯遗忘曲线模型.

融合世界先进方案:
- 艾宾浩斯遗忘曲线 (Ebbinghaus 1885): 记忆保持量随时间指数衰减
    m(t) = m * exp(-lambda * delta_t)
- 记忆稳定性 (stability): 借鉴 FSRS 的记忆稳定性概念,
    stability 越大遗忘越慢: lambda = base_lambda / stability.
- Khan Academy / KSS 调度: 短期内 (<= 7 天) 视为无显著遗忘,
    仅在超过 7 天 (168 小时) 后才触发衰减.

衰减规则:
- delta_t_hours <= 168 (7 天): 不衰减, 返回原掌握度
- delta_t_hours > 168: m(t) = mastery * exp(-lambda * (delta_t - 168))

其中 ``lambda = base_lambda / stability``, ``base_lambda = 0.007``.

复习判定 (should_review):
- 计算自上次作答以来的时间差, 应用衰减得到当前有效掌握度;
- 若有效掌握度低于阈值 (默认 0.5), 则判定需要复习.
"""

from __future__ import annotations

import math

from dy3_polaris.l2.models import TracingState


# ============================================================
# 1. 常量定义
# ============================================================

# 基础遗忘率 lambda (stability=1.0 时的指数衰减系数)
BASE_LAMBDA: float = 0.007

# 衰减触发阈值 (小时): 超过 7 天 (168 小时) 才触发衰减
DECAY_THRESHOLD_HOURS: float = 168.0

# --- 动态记忆稳定性 (借鉴 L1 models.py / FSRS 间隔重复) ---
# 最小记忆稳定性: 0 次作答时的初始稳定性
MIN_STABILITY: float = 1.0
# 每次作答增加的稳定性增益 (练习越多遗忘越慢)
STABILITY_GAIN: float = 0.5

# 秒 -> 小时换算系数
_SECONDS_PER_HOUR: float = 3600.0

# should_review 默认掌握度阈值
DEFAULT_REVIEW_THRESHOLD: float = 0.5


# ============================================================
# 2. ForgettingModel 无状态引擎类
# ============================================================


class ForgettingModel:
    """艾宾浩斯遗忘曲线模型 (无状态).

    提供掌握度衰减计算与复习需求判定:
    1. ``decay``: 给定时间间隔计算衰减后的掌握度.
    2. ``should_review``: 综合掌握度与时间差, 判定是否需要复习.

    Attributes:
        base_lambda: 基础遗忘率 (默认 0.007).
        decay_threshold_hours: 衰减触发阈值 (默认 168 小时 = 7 天).
        default_threshold: should_review 默认掌握度阈值 (默认 0.5).

    该类为无状态引擎, 不持有学习者状态, 适合并发复用.
    """

    def __init__(
        self,
        base_lambda: float = BASE_LAMBDA,
        decay_threshold_hours: float = DECAY_THRESHOLD_HOURS,
        default_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    ) -> None:
        """初始化遗忘模型.

        Args:
            base_lambda: 基础遗忘率 lambda, 默认 0.007.
            decay_threshold_hours: 衰减触发阈值 (小时), 默认 168 (7 天).
            default_threshold: should_review 默认掌握度阈值, 默认 0.5.
        """
        self.base_lambda = base_lambda
        self.decay_threshold_hours = decay_threshold_hours
        self.default_threshold = default_threshold

    # --- 衰减计算 ---

    def decay(
        self,
        mastery: float,
        delta_t_hours: float,
        stability: float = 1.0,
    ) -> float:
        """计算给定时间间隔后的衰减掌握度 (艾宾浩斯遗忘曲线).

        规则:
        - 若 ``delta_t_hours <= 168`` (7 天内): 不衰减, 返回原 ``mastery``.
        - 否则: ``m(t) = mastery * exp(-lambda * max(0, delta_t - 168))``

        其中 ``lambda = base_lambda / stability``, ``stability`` 越大遗忘越慢.

        Args:
            mastery: 原始掌握度 [0.0, 1.0].
            delta_t_hours: 距上次学习的时间间隔 (小时).
            stability: 记忆稳定性, 默认 1.0; 越大遗忘越慢.

        Returns:
            衰减后的掌握度 (不超过原值, 落在 [0.0, mastery]).
        """
        # 7 天内不衰减
        if delta_t_hours <= self.decay_threshold_hours:
            return mastery

        # 避免除零 (stability 应为正数)
        stability = stability if stability > 0.0 else 1.0
        lam = self.base_lambda / stability

        # 仅对超出阈值部分的时间施加衰减
        effective_delta = max(0.0, delta_t_hours - self.decay_threshold_hours)
        return mastery * math.exp(-lam * effective_delta)

    # --- 动态记忆稳定性 ---

    def compute_stability(
        self,
        attempts: int,
        correct_count: int,
    ) -> float:
        """计算动态记忆稳定性 — 综合练习次数与正确率 (借鉴 FSRS 间隔重复).

        公式:
            base = MIN_STABILITY + attempts * STABILITY_GAIN
            if attempts > 0:
                correct_rate = correct_count / attempts
                accuracy_bonus = max(0.0, (correct_rate - 0.5) * 2.0) * STABILITY_GAIN * attempts
                stability = base + accuracy_bonus
            else:
                stability = base

        - ``MIN_STABILITY`` = 1.0 (新学知识的初始稳定性);
        - ``STABILITY_GAIN`` = 0.5 (每次作答增加的稳定性);
        - attempts (累计练习次数) 越多, stability 越大, 遗忘越慢;
        - 正确率 > 0.5 时额外给予 accuracy_bonus (正确率越高记忆越稳固),
          正确率 <= 0.5 时无 bonus (避免奖励低质量练习).

        Args:
            attempts: 累计作答次数 (>= 0).
            correct_count: 累计答对次数 (>= 0, 与 attempts 共同决定正确率).

        Returns:
            记忆稳定性 (>= MIN_STABILITY).
        """
        attempts = max(0, int(attempts))
        correct_count = max(0, int(correct_count))
        # 基础稳定性: 由练习次数决定
        base = MIN_STABILITY + attempts * STABILITY_GAIN
        if attempts > 0:
            # 正确率 > 0.5 时给予额外稳定性奖励 (高正确率 -> 记忆更稳固)
            correct_rate = correct_count / attempts
            accuracy_bonus = max(0.0, (correct_rate - 0.5) * 2.0) * STABILITY_GAIN * attempts
            return base + accuracy_bonus
        return base

    # --- 平滑保留率 (连续指数衰减, 无硬阈值) ---

    def compute_retention(
        self,
        mastery: float,
        delta_t_hours: float,
        stability: float = 1.0,
    ) -> float:
        """计算连续指数衰减下的记忆保留率 (平滑, 无 7 天硬阈值).

        与 ``decay`` 的硬阈值模型不同, ``compute_retention`` 对全部时间间隔
        连续施加指数衰减 (适合需要平滑保留率曲线的场景, 如复习调度排序):

            retention = mastery * exp(-(base_lambda / stability) * delta_t)

        Args:
            mastery: 原始掌握度 [0.0, 1.0].
            delta_t_hours: 距上次学习的时间间隔 (小时); 负值视为 0 (无衰减).
            stability: 记忆稳定性, 默认 1.0; 越大衰减越慢.

        Returns:
            保留率 [0.0, 1.0] (不超过原 mastery).
        """
        mastery = max(0.0, min(1.0, mastery))
        # 时间倒流视为无衰减
        if delta_t_hours <= 0.0:
            return mastery
        # 避免除零 (stability 应为正数)
        stability = stability if stability > 0.0 else 1.0
        lam = self.base_lambda / stability
        retention = mastery * math.exp(-lam * delta_t_hours)
        return max(0.0, min(1.0, retention))

    # --- 复习判定 ---

    def should_review(
        self,
        state: TracingState,
        current_time: float,
        threshold: float | None = None,
    ) -> bool:
        """判定知识点是否需要复习.

        根据自上次作答以来的时间差, 使用 ``compute_retention`` (连续指数衰减,
        无 7 天硬阈值) 计算当前有效掌握度; 若有效掌握度低于阈值, 则判定需要复习.

        采用连续衰减 (而非 ``decay`` 的硬阈值) 使得保留率曲线随时间平滑下降,
        避免 168 小时边界处的判定跳变, 更适合复习调度排序.

        时间差由 ``state.last_attempt_time`` (秒) 与 ``current_time`` (秒) 计算,
        并换算为小时后调用 ``compute_retention``.

        Args:
            state: 知识点追踪状态 (含 mastery_prob 与 last_attempt_time).
            current_time: 当前时间戳 (秒, float).
            threshold: 掌握度阈值, 默认使用 ``self.default_threshold`` (0.5).

        Returns:
            True 表示有效掌握度低于阈值, 需要复习; 否则 False.
        """
        if threshold is None:
            threshold = self.default_threshold

        # 时间差 (秒) -> 小时
        delta_t_seconds = current_time - state.last_attempt_time
        delta_t_hours = delta_t_seconds / _SECONDS_PER_HOUR

        # 动态稳定性: 由累计作答次数与正确率决定 (练习越多/正确率越高遗忘越慢)
        stability = self.compute_stability(state.attempts, state.correct_count)

        # 连续衰减后的有效掌握度 (平滑, 无 168h 硬阈值跳变)
        decayed = self.compute_retention(
            state.mastery_prob, delta_t_hours, stability=stability
        )

        return decayed < threshold

    # --- 推荐复习间隔 ---

    def compute_review_interval(
        self,
        mastery: float,
        stability: float,
        target_retention: float = 0.9,
    ) -> float:
        """计算推荐复习间隔 (小时).

        基于连续遗忘模型, 求解掌握度从当前 ``mastery`` 衰减到 ``target_retention``
        所需的时间. 连续衰减模型为:

            retention(t) = mastery * exp(-(base_lambda / stability) * t)

        令 retention(t) = target_retention, 解得:

            t = -ln(target_retention / mastery) / (base_lambda / stability)

        Args:
            mastery: 当前掌握度 [0.0, 1.0].
            stability: 记忆稳定性 (越大衰减越慢, 间隔越长).
            target_retention: 目标保留率, 默认 0.9 (掌握度降至该值前应复习).

        Returns:
            推荐复习间隔 (小时). 当 mastery <= 0、target_retention >= 1.0、
            或 target_retention >= mastery (已低于目标) 时返回 0.0.
        """
        if mastery <= 0.0 or target_retention >= 1.0:
            return 0.0
        # 避免除零 (stability 应为正数)
        stability = max(stability, 0.1)
        # 已低于目标保留率 -> 无需等待, 立即复习
        if target_retention >= mastery:
            return 0.0
        lam = self.base_lambda / stability
        return -math.log(target_retention / mastery) / lam


# ============================================================
# __all__
# ============================================================

__all__ = [
    "ForgettingModel",
    "BASE_LAMBDA",
    "DECAY_THRESHOLD_HOURS",
    "DEFAULT_REVIEW_THRESHOLD",
    "MIN_STABILITY",
    "STABILITY_GAIN",
]
