"""L2 更新管道 — 事件驱动的实时画像更新.

设计依据:
- L2 规划文档 5.2 节: EventCollector -> BKTTracer -> IRTModel -> ForgettingModel -> ProfileBuilder
- Khan Academy: 事件驱动 BKT 实时更新 (<1ms 延迟)
- Knewton: 三引擎架构 (评估/策略/反馈)

处理流程:
1. 接收事件 (AnswerEvent / QueryEvent / BehaviorEvent)
2. 验证事件 (委托内部 EventCollector.validate)
3. 持久化到 Store (answer_history) —— 仅 AnswerEvent
4. 调用 BKTTracer.update 更新追踪状态  —— 仅 AnswerEvent 且 bkt_tracer 已注入
5. 调用 IRTEstimator.update_theta 更新能力估计 —— 仅 AnswerEvent 且 irt_estimator 已注入
6. 返回更新摘要 {learner_id, kp_id, event_type, updated, new_mastery, new_theta, ...}

依赖注入与优雅降级:
- bkt_tracer / irt_estimator / store 任一为 None 时, 跳过对应步骤, 不抛异常.
- 仅当至少一个引擎实际完成更新时, 摘要 updated=True.

依赖的鸭子类型接口 (L2 knowledge_tracer / ability_assessor 对接):
- bkt_tracer.update(state: TracingState, correct: bool, timestamp: float) -> TracingState
- irt_estimator.update_theta(state: IRTState, item_params: dict, correct: bool) -> IRTState
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from dy3_polaris.l2.interaction.collector import EventCollector
from dy3_polaris.l2.interaction.event_types import (
    AnswerEvent,
    BehaviorEvent,
    QueryEvent,
)
from dy3_polaris.l2.models import (
    DEFAULT_INITIAL_SE,
    DEFAULT_INITIAL_THETA,
    DEFAULT_IRT_A,
    DEFAULT_IRT_C,
    IRT_THETA_MAX,
    IRT_THETA_MIN,
    IRTState,
    TracingState,
)
from dy3_polaris.l2.store import L2Store


# ============================================================
# 模块级 logger
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

# 默认初始 BKT 追踪状态掌握概率 (无历史记录时)
DEFAULT_INITIAL_MASTERY: float = 0.5
# 默认初始 IRT 能力 theta / 估计标准误: 由 models.py 统一常量提供
# (DEFAULT_INITIAL_THETA / DEFAULT_INITIAL_SE 见上方 import), 此处不再重复定义,
# 消除与 tracing_service.py 的 0.3 vs 0.5 不一致.


# ============================================================
# UpdatePipeline
# ============================================================


class UpdatePipeline:
    """L2 更新管道 — 事件驱动的实时画像更新.

    通过依赖注入接收 BKT 追踪器 / IRT 估计器 / 存储层, 任一缺省则优雅降级.

    Args:
        bkt_tracer: BKT 知识追踪器 (鸭子类型, 需实现 update(state, correct, timestamp)).
            为 None 时跳过掌握度更新.
        irt_estimator: IRT 能力估计器 (鸭子类型, 需实现 update_theta(state, item_params, correct)).
            为 None 时跳过能力估计更新.
        store: L2 存储层 (L2Store 实现). 为 None 时跳过持久化.
        forgetting_model: 遗忘衰减模型 (鸭子类型, 需实现 decay / compute_stability).
            为 None 时跳过遗忘衰减. 注入后, process() 会对学习者其他知识点
            施加基于时间间隔的掌握度衰减.
        mastery_propagator: 掌握度传播器 (鸭子类型, 需实现
            propagate_mastery(learner_id, kp_id, mastery, store)).
            为 None 时跳过传播. 注入后, process() 在 BKT 更新成功后调用其
            propagate_mastery, 将掌握度传播到依赖知识点.

    Attributes:
        bkt_tracer: 注入的 BKT 追踪器 (可能为 None)
        irt_estimator: 注入的 IRT 估计器 (可能为 None)
        store: 注入的存储层 (可能为 None)
        forgetting_model: 注入的遗忘衰减模型 (可能为 None)
        mastery_propagator: 注入的掌握度传播器 (可能为 None)
        collector: 内部事件采集器 (负责验证与单调性维护)
    """

    def __init__(
        self,
        bkt_tracer: Any | None = None,
        irt_estimator: Any | None = None,
        store: L2Store | None = None,
        forgetting_model: Any | None = None,
        mastery_propagator: Any | None = None,
    ) -> None:
        self.bkt_tracer = bkt_tracer
        self.irt_estimator = irt_estimator
        self.store = store
        self.forgetting_model = forgetting_model
        self.mastery_propagator = mastery_propagator
        # 内部采集器: 负责事件验证 + 时间戳单调性维护
        self.collector = EventCollector()
        # 保护 _persist_answer 的 read-modify-write (get -> append -> save) 竞态
        self._persist_lock = threading.RLock()

    # --- 单事件处理 ---

    def process(self, event: Any) -> dict[str, Any]:
        """处理单个事件, 返回更新结果摘要.

        处理步骤:
        1. 校验事件 (非法则 updated=False, 附 reason).
        2. 若为 AnswerEvent:
           a. 持久化 AnswerRecord 到 store.answer_history (store 可用时).
           b. 调用 bkt_tracer.update 更新 TracingState (bkt_tracer 可用时).
           c. 调用 irt_estimator.update_theta 更新 IRTState (irt_estimator 可用时).
        3. QueryEvent / BehaviorEvent 不触发 BKT/IRT, updated=False.

        Args:
            event: AnswerEvent / QueryEvent / BehaviorEvent.

        Returns:
            更新摘要字典, 含键:
            - learner_id  : 学习者 ID
            - kp_id       : 知识点 ID (QueryEvent 无则 None)
            - event_type  : "answer" / "query" / "behavior"
            - updated     : 是否实际完成 BKT/IRT 更新
            - new_mastery : 更新后掌握概率 (未更新则 None)
            - new_theta   : 更新后能力 theta (未更新则 None)
            - reason      : 未更新原因 (仅当 updated=False 时出现)
        """
        summary: dict[str, Any] = {
            "learner_id": getattr(event, "learner_id", None),
            "kp_id": getattr(event, "kp_id", None),
            "event_type": _event_type_name(event),
            "updated": False,
            "new_mastery": None,
            "new_theta": None,
        }

        # 1. 验证事件 (维护时间戳单调性)
        if not self.collector.validate(event):
            summary["reason"] = "validation_failed"
            return summary

        # 2. 仅 AnswerEvent 触发 BKT/IRT 更新
        if not isinstance(event, AnswerEvent):
            summary["reason"] = "no_update_for_event_type"
            return summary

        # 2a. 持久化答题记录
        self._persist_answer(event)

        # 2b. BKT 追踪更新
        new_mastery: float | None = None
        if self.bkt_tracer is not None:
            new_mastery = self._update_bkt(event)
            if new_mastery is not None:
                summary["new_mastery"] = new_mastery
                summary["updated"] = True

        # 2c. IRT 能力估计更新
        if self.irt_estimator is not None:
            new_theta = self._update_irt(event)
            if new_theta is not None:
                summary["new_theta"] = new_theta
                summary["updated"] = True

        # 2d. 遗忘衰减: 对学习者其他知识点施加基于时间间隔的掌握度衰减
        if self.forgetting_model is not None and self.store is not None:
            self._apply_forgetting(event)

        # 2e. 掌握度传播: 将本次更新的掌握度传播到依赖知识点
        if self.mastery_propagator is not None and new_mastery is not None:
            self._propagate_mastery(event, new_mastery)

        if not summary["updated"]:
            summary["reason"] = "no_engine_injected"

        return summary

    # --- 批量处理 ---

    def batch_process(self, events: list[Any]) -> list[dict[str, Any]]:
        """批量处理事件 — 按时间戳升序排序后逐个处理.

        排序目的: 保证时间戳单调递增, 避免因乱序输入导致 validate 失败.

        Args:
            events: 事件列表.

        Returns:
            每个事件对应的更新摘要列表 (按时间戳升序排列).
        """
        if not events:
            return []
        # 按时间戳升序排序后逐个处理
        ordered = sorted(events, key=lambda e: getattr(e, "timestamp", 0.0))
        return [self.process(ev) for ev in ordered]

    # --- 内部: 持久化答题记录 ---

    def _persist_answer(self, event: AnswerEvent) -> None:
        """将 AnswerEvent 转为 AnswerRecord 并追加到 store.answer_history.

        store 为 None 时静默跳过 (优雅降级).

        线程安全: 使用 _persist_lock 保护 read-modify-write
        (get_answer_history -> append -> save_answer_history), 避免并发追加
        丢失记录.
        """
        if self.store is None:
            return
        record = event.to_answer_record()
        with self._persist_lock:
            history = self.store.get_answer_history(event.learner_id)
            if history is None:
                history = []
            history.append(record)
            self.store.save_answer_history(event.learner_id, history)

    # --- 内部: BKT 追踪更新 ---

    def _update_bkt(self, event: AnswerEvent) -> float | None:
        """调用 bkt_tracer.update 更新 TracingState 并持久化.

        流程:
        1. 从 store 读取当前 TracingState (无则构造默认 mastery=0.5).
        2. 调用 bkt_tracer.update(state, correct, timestamp) 得到新状态.
        3. 持久化新状态到 store (store 可用时).
        4. 返回新状态的 mastery_prob.

        Returns:
            更新后的 mastery_prob; bkt_tracer 异常时返回 None.
        """
        try:
            current = self._get_tracing_state(event)
            new_state = self.bkt_tracer.update(
                current, event.correct, event.timestamp
            )
            if self.store is not None and new_state is not None:
                self.store.save_tracing_state(
                    event.learner_id, event.kp_id, new_state
                )
            return new_state.mastery_prob if new_state is not None else None
        except Exception as e:
            # 优雅降级: 引擎异常不影响整体流程, 但记录日志便于诊断
            logger.warning(
                "BKT update failed for learner=%s kp=%s: %s",
                event.learner_id,
                event.kp_id,
                e,
            )
            return None

    # --- 内部: IRT 能力估计更新 ---

    def _update_irt(self, event: AnswerEvent) -> float | None:
        """调用 irt_estimator.update_theta 更新 IRTState 并持久化.

        流程:
        1. 从 store 读取当前 IRTState (无则构造默认 theta=0.0, se=0.3).
        2. 将 difficulty [0,1] 转换为 IRT 题目参数 {a, b, c}:
           - b = (difficulty - 0.5) * 6  (映射到 [-3, 3] IRT 难度尺度)
           - a = 1.0 (默认区分度)
           - c = 0.25 (默认猜测下限, 4 选 1)
        3. 调用 irt_estimator.update_theta(state, item_params, correct) 得到新状态.
        4. 持久化新状态到 store (store 可用时).
        5. 返回新状态的 theta.

        Returns:
            更新后的 theta; irt_estimator 异常时返回 None.
        """
        try:
            current = self._get_irt_state(event)
            # difficulty [0,1] -> IRT b [-3,3], a/c 使用统一默认常量
            item_params = {
                "a": DEFAULT_IRT_A,
                "b": (event.difficulty - 0.5) * 6.0,
                "c": DEFAULT_IRT_C,
            }
            new_state = self.irt_estimator.update_theta(
                current, item_params, event.correct
            )
            # 能力估计边界钳制: 防止异常输入导致 θ 漂移到合理范围外
            if new_state is not None:
                new_state.theta = max(
                    IRT_THETA_MIN, min(IRT_THETA_MAX, new_state.theta)
                )
            if self.store is not None and new_state is not None:
                self.store.save_irt_state(event.learner_id, new_state)
            return new_state.theta if new_state is not None else None
        except Exception as e:
            # 优雅降级: 引擎异常不影响整体流程, 但记录日志便于诊断
            logger.warning(
                "IRT update failed for learner=%s: %s",
                event.learner_id,
                e,
            )
            return None

    # --- 内部: 状态读取 (带默认值回退) ---

    def _get_tracing_state(self, event: AnswerEvent) -> TracingState:
        """读取当前 TracingState, 无记录时返回默认状态 (mastery=0.5)."""
        if self.store is not None:
            state = self.store.get_tracing_state(event.learner_id, event.kp_id)
            if state is not None:
                return state
        return TracingState(
            kp_id=event.kp_id,
            mastery_prob=DEFAULT_INITIAL_MASTERY,
            last_attempt_time=event.timestamp,
        )

    def _get_irt_state(self, event: AnswerEvent) -> IRTState:
        """读取当前 IRTState, 无记录时返回默认状态 (theta=0.0, se=0.3)."""
        if self.store is not None:
            state = self.store.get_irt_state(event.learner_id)
            if state is not None:
                return state
        return IRTState(
            theta=DEFAULT_INITIAL_THETA,
            se=DEFAULT_INITIAL_SE,
            last_update_time=event.timestamp,
        )

    # --- 内部: 遗忘衰减集成 ---

    # 秒 -> 小时换算系数
    _SECONDS_PER_HOUR: float = 3600.0

    def _apply_forgetting(self, event: AnswerEvent) -> None:
        """对学习者其他知识点施加艾宾浩斯遗忘衰减.

        遍历学习者全部追踪状态 (排除本次事件刚更新的知识点), 依据自上次作答
        以来的时间间隔 (以事件时间戳为基准), 调用 forgetting_model.decay 计算
        衰减后掌握度并持久化.

        衰减门控由 forgetting_model.decay 自身实现 (默认 <= 168 小时不衰减).
        无作答记录 (last_attempt_time <= 0) 或时间未推移 (delta_t <= 0) 的
        知识点跳过.

        store 不支持 get_all_tracing_states (返回空) 时静默无操作.
        """
        if self.forgetting_model is None or self.store is None:
            return
        # 鸭子类型获取学习者全部追踪状态; 不支持时返回空字典 (无操作)
        getter = getattr(self.store, "get_all_tracing_states", None)
        if getter is None:
            return
        try:
            states = getter(event.learner_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Forgetting apply failed to load states for learner=%s: %s",
                event.learner_id,
                e,
            )
            return
        if not states:
            return
        now = event.timestamp
        for kp_id, state in states.items():
            # 跳过本次事件刚更新的知识点 (已由 BKT 更新)
            if kp_id == event.kp_id:
                continue
            # 从未作答 -> 不衰减
            if state.last_attempt_time <= 0.0:
                continue
            delta_t_hours = (now - state.last_attempt_time) / self._SECONDS_PER_HOUR
            # 时间未推移 (含时间倒流) -> 不衰减
            if delta_t_hours <= 0.0:
                continue
            stability = self.forgetting_model.compute_stability(
                state.attempts, state.correct_count
            )
            decayed = self.forgetting_model.decay(
                state.mastery_prob, delta_t_hours, stability=stability
            )
            # 仅当衰减实际改变掌握度时持久化 (避免无意义写入)
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
            self.store.save_tracing_state(event.learner_id, kp_id, new_state)

    # --- 内部: 掌握度传播集成 ---

    def _propagate_mastery(
        self, event: AnswerEvent, new_mastery: float
    ) -> None:
        """将本次更新的掌握度传播到依赖知识点.

        委托 mastery_propagator.propagate_mastery(learner_id, kp_id, mastery,
        store) 完成 KG 驱动的掌握度传播 (具体传播逻辑与前置关系由传播器实现
        持有, 与本管道解耦).

        传播器未实现 propagate_mastery 方法时静默无操作.
        """
        if self.mastery_propagator is None:
            return
        propagate_fn = getattr(self.mastery_propagator, "propagate_mastery", None)
        if propagate_fn is None:
            return
        try:
            propagate_fn(event.learner_id, event.kp_id, new_mastery, self.store)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Mastery propagation failed for learner=%s kp=%s: %s",
                event.learner_id,
                event.kp_id,
                e,
            )


# ============================================================
# 辅助函数
# ============================================================


def _event_type_name(event: Any) -> str:
    """推断事件类型名称 (用于摘要 event_type 字段).

    Args:
        event: 事件对象.

    Returns:
        "answer" / "query" / "behavior" / "unknown".
    """
    if isinstance(event, AnswerEvent):
        return "answer"
    if isinstance(event, QueryEvent):
        return "query"
    if isinstance(event, BehaviorEvent):
        return "behavior"
    return "unknown"


# ============================================================
# __all__
# ============================================================

__all__ = [
    "UpdatePipeline",
    "DEFAULT_INITIAL_MASTERY",
    "DEFAULT_INITIAL_THETA",
    "DEFAULT_INITIAL_SE",
]
