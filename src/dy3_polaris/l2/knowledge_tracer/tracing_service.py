"""BKT 全链路编排服务.

融合世界先进方案:
- Corbett & Anderson (1995): 标准 BKT 四参数 + 前向算法
- Yudelson-Koedinger-Gordon (CMU 2013): 个体化 BKT (BPT) 参数覆盖
- Knewton: KG 驱动掌握度传播
- Ebbinghaus: 遗忘曲线衰减
- OSCOI 模式: 离线标定 + 在线推理分离

全链路处理流程 (设计文档要求顺序):
1. 遗忘衰减: 对学习者全部知识点施加基于时间间隔的掌握度衰减
2. BKT 更新: 对本次事件的知识点执行前向算法 (后验 + 转移)
3. KG 传播: 将更新后的掌握度沿知识图谱前置关系传播提升
4. 输出构建: 计算预测正确率 / 掌握标志 / 置信区间, 封装为 MasteryOutput

MasteryOutput 契约字段 (供下游 CAT / 推荐 / 画像消费):
- p_mastery          : 当前掌握概率 [0, 1] (画像着色)
- p_correct_next     : 下一次答对预测概率 [0, 1] (CAT 选题)
- mastery_flag       : 是否达到掌握阈值 (推荐决策)
- confidence_interval: 掌握度 95% 置信区间 [lower, upper] (预警置信)
- attempts           : 累计作答次数 (停滞检测)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.knowledge_tracer.bkt import BKTTracer
from dy3_polaris.l2.knowledge_tracer.forgetting import ForgettingModel
from dy3_polaris.l2.knowledge_tracer.mastery_propagator import MasteryPropagator
from dy3_polaris.l2.models import AnswerRecord, DEFAULT_BKT_PARAMS, TracingState
from dy3_polaris.l2.store import InMemoryL2Store, L2Store


# ============================================================
# 模块级 logger
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

# 默认掌握阈值: 掌握概率 >= 该值时 mastery_flag = True
DEFAULT_MASTERY_THRESHOLD: float = 0.85

# 置信区间计算常数: half_width = CI_CONSTANT / (1 + log(1 + n))
# n=1 时 half_width ≈ 0.177 (宽), n=30 时 ≈ 0.068 (窄)
CI_CONSTANT: float = 0.3

# 秒 -> 小时换算系数
_SECONDS_PER_HOUR: float = 3600.0

# 学习者级参数 -> 技能级参数键映射 (覆盖式, 非 logit 融合)
_LEARNER_PARAM_MAPPING: dict[str, str] = {
    "learner_p_t": "p_t",
    "learner_p_g": "p_g",
    "learner_p_s": "p_s",
}


# ============================================================
# MasteryOutput — 下游输出标准化契约
# ============================================================


@dataclass
class MasteryOutput:
    """BKT 全链路输出 — 标准化掌握度契约.

    供下游 CAT 选题 / 推荐决策 / 画像着色 / 预警置信 消费.

    Attributes:
        learner_id: 学习者 ID
        kp_id: 知识点 ID
        p_mastery: 当前掌握概率 P(Know) [0.0, 1.0]
        p_correct_next: 下一次答对预测概率 [0.0, 1.0]
        mastery_flag: 是否达到掌握阈值
        attempts: 累计作答次数
        last_updated_ts: 最后更新时间戳 (秒, float)
        confidence_interval: 掌握度 95% 置信区间 [lower, upper]
    """

    learner_id: str
    kp_id: str
    p_mastery: float
    p_correct_next: float
    mastery_flag: bool
    attempts: int
    last_updated_ts: float
    confidence_interval: list[float] = field(default_factory=lambda: [0.0, 1.0])

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (confidence_interval 浅拷贝)."""
        return {
            "learner_id": self.learner_id,
            "kp_id": self.kp_id,
            "p_mastery": self.p_mastery,
            "p_correct_next": self.p_correct_next,
            "mastery_flag": self.mastery_flag,
            "attempts": self.attempts,
            "last_updated_ts": self.last_updated_ts,
            "confidence_interval": list(self.confidence_interval),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MasteryOutput:
        """从字典反序列化."""
        return cls(
            learner_id=d["learner_id"],
            kp_id=d["kp_id"],
            p_mastery=d["p_mastery"],
            p_correct_next=d["p_correct_next"],
            mastery_flag=d["mastery_flag"],
            attempts=d["attempts"],
            last_updated_ts=d["last_updated_ts"],
            confidence_interval=list(d.get("confidence_interval", [0.0, 1.0])),
        )


# ============================================================
# BKTTracingService — 全链路编排器
# ============================================================


class BKTTracingService:
    """BKT 全链路编排服务 — 事件→遗忘→BKT→KG→输出.

    编排全链路处理流程:
    1. 遗忘衰减 (全知识点, 基于时间间隔)
    2. BKT 前向更新 (标准或个体化)
    3. KG 掌握度传播 (前置→后继)
    4. 输出构建 (预测正确率 / 掌握标志 / 置信区间)

    支持特性:
    - 个体化 BKT (BPT): 学习者级参数覆盖技能级参数
    - KP 级参数覆盖: EM 标定参数注入
    - KG 传播: 知识图谱前置关系驱动的掌握度提升
    - 遗忘衰减: 艾宾浩斯曲线, 超 7 天触发
    - 置信区间: 基于观测次数的自适应宽度
    - 优雅降级: store=None 时使用内部内存存储

    Args:
        store: L2 存储层. 为 None 时使用内部 InMemoryL2Store.
        mastery_threshold: 掌握阈值, 默认 0.85.
        bkt_tracer: BKT 引擎 (可选, 默认创建 BKTTracer).
        forgetting_model: 遗忘模型 (可选, 默认创建 ForgettingModel).
        mastery_propagator: 掌握度传播器 (可选, 默认创建 MasteryPropagator).

    Attributes:
        bkt_tracer: BKT 追踪引擎
        forgetting_model: 遗忘衰减模型
        mastery_propagator: KG 掌握度传播器
        store: L2 存储层
        mastery_threshold: 掌握阈值
    """

    def __init__(
        self,
        store: L2Store | None = None,
        mastery_threshold: float = DEFAULT_MASTERY_THRESHOLD,
        bkt_tracer: BKTTracer | None = None,
        forgetting_model: ForgettingModel | None = None,
        mastery_propagator: MasteryPropagator | None = None,
    ) -> None:
        self.bkt_tracer = bkt_tracer or BKTTracer()
        self.forgetting_model = forgetting_model or ForgettingModel()
        self.mastery_propagator = mastery_propagator or MasteryPropagator()
        self.store: L2Store = store if store is not None else InMemoryL2Store()
        self.mastery_threshold = mastery_threshold
        # 学习者级参数 (个体化 BKT): learner_id -> {learner_p_t, ...}
        self._learner_params: dict[str, dict[str, float]] = {}
        # KP 级参数覆盖 (EM 标定注入): kp_id -> {p_l0, p_t, p_g, p_s}
        self._kp_params: dict[str, dict[str, float]] = {}

    # --- KG 图谱设置 ---

    def set_kg_graph(
        self,
        graph: dict[str, list[tuple[str, float]]],
    ) -> None:
        """设置知识图谱邻接表 (委托 mastery_propagator).

        Args:
            graph: ``{kp_id: [(prereq_id, weight), ...]}``
        """
        self.mastery_propagator.set_kg_graph(graph)

    # --- 个体化参数设置 ---

    def set_learner_params(
        self,
        learner_id: str,
        params: dict[str, float],
    ) -> None:
        """设置学习者级 BKT 参数 (个体化 BPT).

        学习者级参数覆盖技能级参数 (非 logit 融合, 直接替换):
        - learner_p_t -> 覆盖 p_t (学习转移)
        - learner_p_g -> 覆盖 p_g (猜测)
        - learner_p_s -> 覆盖 p_s (失误)

        Args:
            learner_id: 学习者 ID.
            params: 学习者级参数, 可选键 learner_p_t / learner_p_g / learner_p_s.
        """
        self._learner_params[learner_id] = dict(params)

    def set_kp_params(
        self,
        kp_id: str,
        params: dict[str, float],
    ) -> None:
        """设置知识点级 BKT 参数 (EM 标定注入).

        覆盖默认 / 难度映射的 BKT 参数, 用于 OSCOI 模式的在线推理阶段.

        Args:
            kp_id: 知识点 ID.
            params: BKT 参数, 含 p_l0 / p_t / p_g / p_s.
        """
        self._kp_params[kp_id] = dict(params)

    # --- 单事件处理 ---

    def process(self, event: AnswerEvent) -> MasteryOutput:
        """处理单条答题事件, 返回完整 MasteryOutput.

        全链路流程:
        1. 遗忘衰减: 对学习者全部知识点施加基于时间间隔的掌握度衰减
        2. BKT 更新: 对事件知识点执行前向算法 (标准或个体化)
        3. KG 传播: 将更新后掌握度沿前置关系传播提升
        4. 输出构建: 计算预测正确率 / 掌握标志 / 置信区间

        Args:
            event: 答题事件.

        Returns:
            MasteryOutput 标准化输出.
        """
        runtime_started = time.monotonic()
        # 1. 遗忘衰减 (全知识点, 包括当前)
        self._apply_forgetting_all(event.learner_id, event.timestamp)

        # 2. 持久化答题记录
        answer_write_started = time.monotonic()
        self._persist_answer(event)
        answer_record_write_ms = (time.monotonic() - answer_write_started) * 1000.0

        # 3. 获取或初始化追踪状态
        state = self._get_or_init_state(event)

        # 4. BKT 前向更新 (标准或个体化)
        new_state = self._update_bkt(state, event)

        # 5. 持久化新状态
        self.store.save_tracing_state(event.learner_id, event.kp_id, new_state)

        # 6. KG 传播
        self.mastery_propagator.propagate_mastery(
            event.learner_id, event.kp_id, new_state.mastery_prob, self.store
        )

        # 7. 读取最终状态 (可能被传播修改)
        final_state = self.store.get_tracing_state(event.learner_id, event.kp_id)
        if final_state is None:
            final_state = new_state

        # 8. 构建输出
        output = self._build_output(event, final_state)
        self._last_runtime_metrics = {
            "answer_record_write_ms": round(answer_record_write_ms, 3),
            "bkt_update_ms": round((time.monotonic() - runtime_started) * 1000.0, 3),
        }
        return output

    def record_observation(self, event: AnswerEvent) -> int:
        """Persist a scored answer without changing BKT/IRT/profile state.

        This is the explicit policy path for route verification: the answer is
        real observed evidence, but one navigation check is not evidence of a
        mastery transition.  It deliberately does not apply forgetting,
        initialize tracing state, propagate mastery, or rebuild the profile.
        """

        started = time.monotonic()
        self._persist_answer(event)
        history = self.store.get_answer_history(event.learner_id) or []
        self._last_runtime_metrics = {
            "answer_record_write_ms": round((time.monotonic() - started) * 1000.0, 3),
            "bkt_update_ms": 0.0,
        }
        return sum(1 for item in history if str(getattr(item, "kp_id", "")) == event.kp_id)

    # --- 批量处理 ---

    def batch_process(
        self,
        events: list[AnswerEvent],
    ) -> list[MasteryOutput]:
        """批量处理答题事件 (按时间戳升序).

        Args:
            events: 答题事件列表.

        Returns:
            每个事件对应的 MasteryOutput 列表.
        """
        if not events:
            return []
        ordered = sorted(events, key=lambda e: e.timestamp)
        return [self.process(ev) for ev in ordered]

    # --- 掌握度快照 ---

    def get_mastery_snapshot(
        self,
        learner_id: str,
    ) -> dict[str, float]:
        """获取学习者全部知识点掌握度快照.

        Args:
            learner_id: 学习者 ID.

        Returns:
            ``{kp_id: mastery_prob}`` 映射.
        """
        states = self.store.get_all_tracing_states(learner_id)
        return {kp_id: state.mastery_prob for kp_id, state in states.items()}

    def get_detailed_snapshot(
        self,
        learner_id: str,
    ) -> dict[str, dict[str, Any]]:
        """获取学习者详细掌握度快照 (含置信区间和预测正确率).

        Args:
            learner_id: 学习者 ID.

        Returns:
            ``{kp_id: {p_mastery, p_correct_next, confidence_interval, attempts, mastery_flag}}``
        """
        states = self.store.get_all_tracing_states(learner_id)
        snapshot: dict[str, dict[str, Any]] = {}
        for kp_id, state in states.items():
            p_correct = self.bkt_tracer.predict_correct_prob(state)
            ci = self._compute_confidence_interval(state)
            snapshot[kp_id] = {
                "p_mastery": state.mastery_prob,
                "p_correct_next": p_correct,
                "confidence_interval": ci,
                "attempts": state.attempts,
                "mastery_flag": state.mastery_prob >= self.mastery_threshold,
            }
        return snapshot

    # ============================================================
    # 内部方法
    # ============================================================

    # --- 遗忘衰减 (全知识点) ---

    def _apply_forgetting_all(
        self,
        learner_id: str,
        current_time: float,
    ) -> None:
        """对学习者全部知识点施加艾宾浩斯遗忘衰减.

        遍历学习者全部追踪状态, 依据自上次作答以来的时间间隔计算衰减后掌握度.
        衰减门控由 forgetting_model.decay 自身实现 (默认 <= 168h 不衰减).

        Args:
            learner_id: 学习者 ID.
            current_time: 当前时间戳 (秒).
        """
        states = self.store.get_all_tracing_states(learner_id)
        if not states:
            return

        for kp_id, state in states.items():
            if state.last_attempt_time <= 0.0:
                continue
            delta_t_hours = (current_time - state.last_attempt_time) / _SECONDS_PER_HOUR
            if delta_t_hours <= 0.0:
                continue

            stability = self.forgetting_model.compute_stability(
                state.attempts, state.correct_count
            )
            decayed = self.forgetting_model.decay(
                state.mastery_prob, delta_t_hours, stability=stability
            )
            if decayed == state.mastery_prob:
                continue

            new_state = TracingState(
                kp_id=state.kp_id,
                mastery_prob=decayed,
                attempts=state.attempts,
                correct_count=state.correct_count,
                last_attempt_time=state.last_attempt_time,
                bkt_params=dict(state.bkt_params),
            )
            self.store.save_tracing_state(learner_id, kp_id, new_state)

    # --- 持久化答题记录 ---

    def _persist_answer(self, event: AnswerEvent) -> None:
        """将答题事件转为 AnswerRecord 并追加到 store.answer_history."""
        record = event.to_answer_record()
        history = self.store.get_answer_history(event.learner_id)
        if history is None:
            history = []
        history.append(record)
        self.store.save_answer_history(event.learner_id, history)

    # --- 获取或初始化追踪状态 ---

    def _get_or_init_state(self, event: AnswerEvent) -> TracingState:
        """获取当前追踪状态, 无记录时根据难度初始化.

        若 store 中存在该 (learner_id, kp_id) 的状态, 返回之;
        否则使用 BKTTracer.init_state 初始化 (难度→p_l0 映射).
        若该 KP 有 EM 标定参数, 使用标定参数初始化.

        Args:
            event: 答题事件.

        Returns:
            当前 (或初始化的) TracingState.
        """
        state = self.store.get_tracing_state(event.learner_id, event.kp_id)
        if state is not None:
            return state

        # 初始化: 使用难度映射或 EM 标定参数
        state = self.bkt_tracer.init_state(event.kp_id, event.difficulty)

        # 若有 KP 级参数覆盖 (EM 标定), 合并到初始状态
        kp_params = self._kp_params.get(event.kp_id)
        if kp_params:
            merged = dict(state.bkt_params)
            merged.update(kp_params)
            state = TracingState(
                kp_id=state.kp_id,
                mastery_prob=float(kp_params.get("p_l0", state.mastery_prob)),
                attempts=state.attempts,
                correct_count=state.correct_count,
                last_attempt_time=state.last_attempt_time,
                bkt_params=merged,
            )

        return state

    # --- BKT 前向更新 ---

    def _update_bkt(
        self,
        state: TracingState,
        event: AnswerEvent,
    ) -> TracingState:
        """执行 BKT 前向更新 (标准或个体化).

        若该学习者有个体化参数, 覆盖技能级参数后执行标准 update;
        否则直接执行标准 update.

        Args:
            state: 当前追踪状态.
            event: 答题事件.

        Returns:
            更新后的新 TracingState.
        """
        learner_params = self._learner_params.get(event.learner_id)
        if learner_params:
            # 个体化: 覆盖技能级参数
            modified_state = self._apply_learner_params(state, learner_params)
            return self.bkt_tracer.update(
                modified_state, event.correct, event.timestamp
            )
        return self.bkt_tracer.update(state, event.correct, event.timestamp)

    @staticmethod
    def _apply_learner_params(
        state: TracingState,
        learner_params: dict[str, float],
    ) -> TracingState:
        """将学习者级参数覆盖到状态的 bkt_params.

        覆盖式 (非 logit 融合): learner_p_t -> p_t, 等.
        保留 p_l0 不变 (先验不随个体化改变).

        Args:
            state: 原始追踪状态.
            learner_params: 学习者级参数.

        Returns:
            参数覆盖后的新 TracingState.
        """
        merged = dict(state.bkt_params)
        for learner_key, skill_key in _LEARNER_PARAM_MAPPING.items():
            if learner_key in learner_params and learner_params[learner_key] is not None:
                merged[skill_key] = float(learner_params[learner_key])
        return TracingState(
            kp_id=state.kp_id,
            mastery_prob=state.mastery_prob,
            attempts=state.attempts,
            correct_count=state.correct_count,
            last_attempt_time=state.last_attempt_time,
            bkt_params=merged,
        )

    # --- 置信区间计算 ---

    @staticmethod
    def _compute_confidence_interval(state: TracingState) -> list[float]:
        """计算掌握度的 95% 置信区间.

        基于观测次数的自适应宽度:
            half_width = CI_CONSTANT / (1 + log(1 + n))

        - n=1 (首次): half_width ≈ 0.177, 宽区间
        - n=30 (多次): half_width ≈ 0.068, 窄区间
        - n→∞: half_width → 0, 点估计

        结果 clamp 到 [0, 1], 且保证 lower <= p_mastery <= upper.

        Args:
            state: 追踪状态.

        Returns:
            [lower, upper] 置信区间.
        """
        n = max(state.attempts, 1)
        p = state.mastery_prob
        half_width = CI_CONSTANT / (1.0 + math.log(1.0 + n))
        lower = max(0.0, p - half_width)
        upper = min(1.0, p + half_width)
        return [lower, upper]

    # --- 输出构建 ---

    def _build_output(
        self,
        event: AnswerEvent,
        state: TracingState,
    ) -> MasteryOutput:
        """构建 MasteryOutput 标准化输出.

        Args:
            event: 原始答题事件 (提供 learner_id / kp_id / timestamp).
            state: 最终追踪状态 (BKT + 传播后).

        Returns:
            MasteryOutput.
        """
        p_correct = self.bkt_tracer.predict_correct_prob(state)
        ci = self._compute_confidence_interval(state)
        mastery_flag = state.mastery_prob >= self.mastery_threshold

        return MasteryOutput(
            learner_id=event.learner_id,
            kp_id=event.kp_id,
            p_mastery=state.mastery_prob,
            p_correct_next=p_correct,
            mastery_flag=mastery_flag,
            attempts=state.attempts,
            last_updated_ts=event.timestamp,
            confidence_interval=ci,
        )


# ============================================================
# __all__
# ============================================================

__all__ = [
    "BKTTracingService",
    "MasteryOutput",
    "DEFAULT_MASTERY_THRESHOLD",
]
