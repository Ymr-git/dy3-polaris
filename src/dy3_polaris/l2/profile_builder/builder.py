"""学情画像构建器 — 组装 BKT/IRT/VARK/Bloom 的综合画像.

融合世界先进方案:
- Knewton: 三引擎架构 (评估/策略/反馈)
- ALEKS: 知识状态集合 + 画像
- Khan Academy: 综合学情画像

组装流程:
1. 加载 BKT TracingState -> 计算 avg_mastery, 提取 kp_mastery
2. 加载 IRTState -> 获取 theta
3. LevelEstimator.estimate(theta, avg_mastery) -> level
4. StyleInferrer.infer -> learning_style
5. BloomSetter.set_target(level) -> bloom_target
6. 提取 weak_kps (mastery < 0.5)
7. 构建 LearnerSnapshot

设计说明:
- ProfileBuilder 通过依赖注入接收 store (可选).
  若提供 store, build 后自动保存画像快照.
- 三个子引擎 (LevelEstimator / StyleInferrer / BloomSetter) 均为无状态类,
  在 __init__ 中实例化, 可通过实例属性访问.
- build 方法返回 L2 LearnerSnapshot (不是 L1 ContextEnvelope).
"""

from __future__ import annotations

import time
from typing import Any

from dy3_polaris.l2.knowledge_tracer.forgetting import ForgettingModel
from dy3_polaris.l2.models import (
    AnswerRecord,
    IRTState,
    LearnerSnapshot,
    TracingState,
)
from dy3_polaris.l2.profile_builder.bloom_setter import BloomSetter
from dy3_polaris.l2.profile_builder.level_estimator import LevelEstimator
from dy3_polaris.l2.profile_builder.style_inferrer import StyleInferrer
from dy3_polaris.l2.store import L2Store


# ============================================================
# 1. 常量定义
# ============================================================

# 薄弱知识点阈值: mastery < 此值视为薄弱 (默认 0.5, 与 practice.WEAK_KP_THRESHOLD 一致)
_WEAK_KP_THRESHOLD: float = 0.5

# 遗忘衰减门控阈值 (小时): 仅当距上次作答超过此时间才施加衰减
_DECAY_GATE_HOURS: float = 1.0

# 秒 -> 小时换算系数
_SECONDS_PER_HOUR: float = 3600.0

# 学习者等级 -> Bloom 当前层次映射
# (ProfileBuilder 将能力等级映射到 Bloom 认知层次后设定目标)
_LEVEL_TO_BLOOM: dict[str, str] = {
    "beginner": "remember",
    "intermediate": "apply",
    "advanced": "analyze",
}


# ============================================================
# 2. ProfileBuilder 画像构建器
# ============================================================


class ProfileBuilder:
    """学情画像构建器 — 组装 BKT/IRT/VARK/Bloom 综合画像.

    组装流程:
    1. 从 ``tracing_states`` 提取 kp_mastery 并计算 avg_mastery.
    2. 从 ``irt_state`` 获取 theta.
    3. ``LevelEstimator.estimate(theta, avg_mastery)`` -> level.
    4. ``StyleInferrer.infer_from_behavior`` (从 interaction_history) -> learning_style.
    5. ``BloomSetter.set_target`` (level 映射到 Bloom 层次) -> bloom_target.
    6. 提取 weak_kps (mastery < 0.5).
    7. 构建 ``LearnerSnapshot`` (若提供 store 则自动保存).

    依赖注入:
    - ``store``: 可选的 L2Store 实例. 提供时 build 后自动保存画像快照.
    - ``forgetting_model``: 可选的 ForgettingModel 实例. 未提供时内部创建默认实例.

    Attributes:
        level_estimator: 能力等级估计器 (无状态).
        style_inferrer: 学习风格推断器 (无状态).
        bloom_setter: Bloom 目标设定器 (无状态).
        forgetting_model: 遗忘衰减模型 (用于对掌握度施加时间衰减).
    """

    def __init__(
        self,
        store: L2Store | None = None,
        forgetting_model: ForgettingModel | None = None,
    ) -> None:
        """初始化画像构建器.

        Args:
            store: 可选的 L2Store 实例, 提供时 build 后自动保存画像快照.
            forgetting_model: 可选的 ForgettingModel 实例; None 时内部创建默认实例.
        """
        self._store: L2Store | None = store
        self.level_estimator: LevelEstimator = LevelEstimator()
        self.style_inferrer: StyleInferrer = StyleInferrer()
        self.bloom_setter: BloomSetter = BloomSetter()
        self.forgetting_model: ForgettingModel = (
            forgetting_model if forgetting_model is not None else ForgettingModel()
        )

    def build(
        self,
        learner_id: str,
        tracing_states: dict[str, TracingState],
        irt_state: IRTState,
        interaction_history: list[AnswerRecord] | None = None,
        weak_kps_threshold: float = _WEAK_KP_THRESHOLD,
        *,
        prev_level: str | None = None,
        response_count: int | None = None,
    ) -> LearnerSnapshot:
        """组装 BKT/IRT/VARK/Bloom 综合画像.

        组装步骤:
        1. 提取 kp_mastery: 对每个 TracingState 施加遗忘衰减得到有效掌握度.
        2. 计算 avg_mastery: 按 attempts 加权平均; 全零 attempts 时回退简单平均.
        3. 获取 theta: irt_state.theta.
        4. 估计等级: LevelEstimator.estimate(theta, avg_mastery).
        5. 推断风格: StyleInferrer.infer_from_behavior (从 interaction_history).
        6. 设定 Bloom 目标: 等级映射到 Bloom 层次后 BloomSetter.set_target.
        7. 提取薄弱知识点: decayed mastery < weak_kps_threshold 的 kp_id 列表.
        8. 计算置信度: confidence = 1 / (1 + se).
        9. 构建 LearnerSnapshot (若提供 store 则保存).

        遗忘衰减规则:
        - 仅当 ``last_attempt_time > 0`` (有过作答) 且距上次作答超过 1 小时
          才调用 ForgettingModel.decay; 否则保持原始掌握度.
        - 稳定性由 ``state.attempts`` 经 ``compute_stability`` 计算
          (练习次数越多, 记忆越稳固, 衰减越慢).

        加权平均规则:
        - 权重 = ``state.attempts`` (练习越多, 估计越可靠).
        - 若所有 attempts 之和为 0, 回退到简单算术平均.

        Args:
            learner_id: 学习者 ID.
            tracing_states: 知识点追踪状态 {kp_id: TracingState}.
            irt_state: IRT 能力估计状态.
            interaction_history: 交互答题历史 (可选, 用于风格推断).
            weak_kps_threshold: 薄弱知识点掌握度阈值, 默认 0.5.

        Returns:
            学习者画像快照 LearnerSnapshot.
        """
        now: float = time.time()

        # --- 1. 提取 kp_mastery (施加遗忘衰减) ---
        kp_mastery: dict[str, float] = {}
        for kp_id, state in tracing_states.items():
            kp_mastery[kp_id] = self._decay_mastery(state, now)

        # --- 2. 计算 avg_mastery (按 attempts 加权) ---
        avg_mastery: float = self._weighted_avg_mastery(
            tracing_states, kp_mastery
        )

        # --- 3. 获取 theta ---
        theta: float = irt_state.theta

        # --- 4. 估计能力等级 (含滞回 + 置信门控, 抑制单题波动导致的标签横跳) ---
        level: str = self.level_estimator.estimate(
            theta,
            avg_mastery,
            prev_level=prev_level,
            se=irt_state.se,
            response_count=response_count,
        )

        # --- 5. 推断学习风格 ---
        events: list[dict[str, Any]] = self._records_to_events(
            interaction_history
        )
        learning_style: str = self.style_inferrer.infer_from_behavior(events)

        # --- 6. 设定 Bloom 目标 ---
        bloom_current: str = _LEVEL_TO_BLOOM.get(level, "remember")
        bloom_target: str = self.bloom_setter.set_target(bloom_current)

        # --- 7. 提取薄弱知识点 (基于衰减后掌握度) ---
        weak_kps: list[str] = [
            kp_id
            for kp_id, mastery in kp_mastery.items()
            if mastery < weak_kps_threshold
        ]

        # --- 8. 计算置信度 (基于 IRT 标准误) ---
        confidence: float = 1.0 / (1.0 + irt_state.se)

        # --- 9. 构建 LearnerSnapshot ---
        snapshot_ts: float = time.time()
        snapshot = LearnerSnapshot(
            learner_id=learner_id,
            snapshot_ts=snapshot_ts,
            kp_mastery=kp_mastery,
            theta=theta,
            level=level,
            learning_style=learning_style,
            bloom_target=bloom_target,
            weak_kps=weak_kps,
            confidence=confidence,
        )

        # 若提供 store, 自动保存画像快照
        if self._store is not None:
            self._store.save_profile(learner_id, snapshot)

        return snapshot

    # --- 内部方法 ---

    def _decay_mastery(
        self,
        state: TracingState,
        now: float,
    ) -> float:
        """对单个 TracingState 施加遗忘衰减, 返回有效掌握度.

        衰减门控规则:
        - ``last_attempt_time <= 0`` (从未作答): 不衰减, 返回原始掌握度.
        - 距上次作答 <= 1 小时: 不衰减 (短期记忆窗口内).
        - 距上次作答 > 1 小时: 调用 ForgettingModel.decay 施加衰减,
          稳定性由 attempts 经 compute_stability 计算.

        Args:
            state: 知识点追踪状态.
            now: 当前时间戳 (秒).

        Returns:
            衰减后的有效掌握度 [0.0, 1.0].
        """
        # 从未作答 -> 不衰减
        if state.last_attempt_time <= 0.0:
            return state.mastery_prob

        delta_t_seconds: float = now - state.last_attempt_time
        delta_t_hours: float = delta_t_seconds / _SECONDS_PER_HOUR

        # 1 小时门控: 短期内不衰减
        if delta_t_hours <= _DECAY_GATE_HOURS:
            return state.mastery_prob

        # 计算记忆稳定性 (练习次数越多, 稳定性越高, 衰减越慢)
        stability: float = self.forgetting_model.compute_stability(
            state.attempts, state.correct_count
        )

        # 施加艾宾浩斯遗忘曲线衰减
        return self.forgetting_model.decay(
            state.mastery_prob, delta_t_hours, stability=stability
        )

    @staticmethod
    def _weighted_avg_mastery(
        tracing_states: dict[str, TracingState],
        kp_mastery: dict[str, float],
    ) -> float:
        """计算加权平均掌握度 (按 attempts 加权).

        规则:
        - 权重 = ``state.attempts`` (练习次数越多, 估计越可靠).
        - 若所有 attempts 之和为 0, 回退到简单算术平均.
        - 空集返回 0.0.

        Args:
            tracing_states: 原始追踪状态 (提供 attempts 权重).
            kp_mastery: 衰减后的掌握度字典 (提供数值).

        Returns:
            加权平均掌握度 [0.0, 1.0].
        """
        if not kp_mastery:
            return 0.0

        total_weight: int = sum(
            tracing_states[kp_id].attempts for kp_id in kp_mastery
        )

        if total_weight == 0:
            # 全零 attempts -> 回退简单平均
            return sum(kp_mastery.values()) / len(kp_mastery)

        weighted_sum: float = sum(
            kp_mastery[kp_id] * tracing_states[kp_id].attempts
            for kp_id in kp_mastery
        )
        return weighted_sum / total_weight

    @staticmethod
    def _records_to_events(
        history: list[AnswerRecord] | None,
    ) -> list[dict[str, Any]]:
        """将 AnswerRecord 列表转换为行为事件字典列表.

        转换后的事件包含 modality / content_type 字段, 用于 StyleInferrer
        的模态推断. AnswerRecord 本身不含模态字段, 故使用 getattr 回退到
        默认值 (reading / text), 保证下游推断总能获得有效模态信息.

        Args:
            history: 答题记录列表 (可为 None).

        Returns:
            行为事件字典列表 (可为空).
        """
        if not history:
            return []
        events: list[dict[str, Any]] = []
        for rec in history:
            events.append(
                {
                    "event_type": "answer",
                    "learner_id": rec.learner_id,
                    "kp_id": rec.kp_id,
                    "correct": rec.correct,
                    "difficulty": rec.difficulty,
                    "timestamp": rec.timestamp,
                    "modality": getattr(rec, "modality", None) or "reading",
                    "content_type": getattr(rec, "content_type", None) or "text",
                }
            )
        return events


# ============================================================
# __all__
# ============================================================

__all__ = [
    "ProfileBuilder",
]
