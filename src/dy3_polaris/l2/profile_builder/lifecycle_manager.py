"""学习者生命周期管理器 + 漂移触发自动重训练.

融合世界先进方案:
- Knewton: 学习者状态生命周期管理 (冷启动 → 稳态 → 漂移)
- ALEKS: 知识状态转换与课程进度追踪
- Duolingo: 学习者技能树成长轨迹
- FuMoE-csKT (2026): 冷启动 → 稳态 → 漂移全生命周期

生命周期五阶段:
1. cold_start    : 0 条记录, 使用群体平均参数
2. warming       : 1 ~ threshold-1 条, 部分个性化
3. stable        : >= threshold 且无漂移, 全量个性化
4. drifting      : 检测到概念漂移, 触发重训练
5. recalibrating : 正在重训练, 等待收敛

设计要点:
- ``LearnerLifecycleManager`` 为每个学习者维护独立漂移检测器
  (以传入检测器的配置为模板), 支持并发调用 (内部 ``threading.RLock``).
- ``DriftAwareRetrainer`` 封装 ``BKTTracer.fit_params`` 的重训练流程,
  带最小改善阈值与回滚机制 (改善不足则回滚旧参数).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from dy3_polaris.l2.ability_assessor.zpd import ZPDCalculator
from dy3_polaris.l2.knowledge_tracer.bkt import BKTTracer
from dy3_polaris.l2.models import DEFAULT_BKT_PARAMS, AnswerRecord
from dy3_polaris.l2.profile_builder.drift_detector import LearnerDriftDetector


# ============================================================
# 1. 学习者生命周期状态
# ============================================================


@dataclass
class LearnerLifecycleState:
    """学习者生命周期状态快照.

    Attributes:
        learner_id: 学习者 ID
        phase: 生命周期阶段 ("cold_start"|"warming"|"stable"|"drifting"|"recalibrating")
        record_count: 累计答题记录数
        last_drift_time: 上次检测到漂移的时间戳 (秒), 无漂移时为 None
        drift_count: 累计检测到的漂移次数
        recalibration_count: 累计成功重训练次数
        current_theta: 当前 IRT 能力值
        current_mastery: 当前平均掌握度
        zpd_zone: 当前 ZPD 区域 ("independent"|"zpd"|"frustration")
        confidence: 估计置信度 [0.0, 1.0]
    """

    learner_id: str
    phase: str = "cold_start"
    record_count: int = 0
    last_drift_time: float | None = None
    drift_count: int = 0
    recalibration_count: int = 0
    current_theta: float = 0.0
    current_mastery: float = 0.5
    zpd_zone: str = "zpd"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "learner_id": self.learner_id,
            "phase": self.phase,
            "record_count": self.record_count,
            "last_drift_time": self.last_drift_time,
            "drift_count": self.drift_count,
            "recalibration_count": self.recalibration_count,
            "current_theta": self.current_theta,
            "current_mastery": self.current_mastery,
            "zpd_zone": self.zpd_zone,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearnerLifecycleState:
        """从字典反序列化."""
        return cls(
            learner_id=d["learner_id"],
            phase=d.get("phase", "cold_start"),
            record_count=d.get("record_count", 0),
            last_drift_time=d.get("last_drift_time"),
            drift_count=d.get("drift_count", 0),
            recalibration_count=d.get("recalibration_count", 0),
            current_theta=d.get("current_theta", 0.0),
            current_mastery=d.get("current_mastery", 0.5),
            zpd_zone=d.get("zpd_zone", "zpd"),
            confidence=d.get("confidence", 0.0),
        )


# ============================================================
# 2. 漂移感知重训练器
# ============================================================


class DriftAwareRetrainer:
    """漂移感知重训练器 — BKT 参数重训练 + 最小改善阈值回滚.

    融合世界先进方案:
    - 概念漂移触发的模型重训练 (Knewton / ALEKS 增量学习)
    - best-tracking + 改善阈值: 仅当重训练显著改善对数似然才接受,
      否则回滚旧参数, 避免漂移误检 / 噪声导致参数劣化.
    """

    def __init__(
        self,
        bkt_tracer: BKTTracer,
        min_improvement: float = 0.01,
        max_retraining_interval: int = 100,
    ):
        """初始化重训练器.

        Args:
            bkt_tracer: BKT 追踪引擎 (用于 fit_params / log_likelihood).
            min_improvement: 接受重训练所需的最小对数似然改善量.
            max_retraining_interval: 无漂移时两次重训练间的最大记录间隔.
        """
        self.bkt_tracer = bkt_tracer
        self.min_improvement = min_improvement
        self.max_retraining_interval = max_retraining_interval

    def retrain(
        self,
        records: list[AnswerRecord],
        current_params: dict[str, Any],
    ) -> dict[str, Any]:
        """执行重训练并依据改善阈值决定接受 / 回滚.

        流程:
        1. 计算旧参数下的对数似然 ``ll_before``;
        2. 调用 ``BKTTracer.fit_params`` 从默认参数重新拟合得到新参数;
        3. 计算新参数下的对数似然 ``ll_after`` 与改善量;
        4. 若改善 >= ``min_improvement``, 接受新参数; 否则回滚旧参数
           (``new_params = old_params``, ``ll_after = ll_before``,
           ``improvement = 0.0``), 保证 ``improvement == ll_after - ll_before``
           始终成立.

        Args:
            records: 答题记录列表 (单个技能的作答历史).
            current_params: 当前 BKT 参数字典.

        Returns:
            重训练结果字典: {new_params, old_params, ll_before, ll_after,
            improvement, accepted}.
        """
        old_params: dict[str, Any] = {
            k: float(current_params.get(k, DEFAULT_BKT_PARAMS[k]))
            for k in ("p_l0", "p_t", "p_g", "p_s")
        }

        # 空记录: 无法重训练, 直接回滚
        if not records:
            return {
                "new_params": old_params,
                "old_params": old_params,
                "ll_before": 0.0,
                "ll_after": 0.0,
                "improvement": 0.0,
                "accepted": False,
            }

        ll_before = self.bkt_tracer.log_likelihood(records, old_params)
        fitted = self.bkt_tracer.fit_params(records)
        ll_fitted = self.bkt_tracer.log_likelihood(records, fitted)
        fitted_improvement = ll_fitted - ll_before

        if fitted_improvement >= self.min_improvement:
            return {
                "new_params": fitted,
                "old_params": old_params,
                "ll_before": ll_before,
                "ll_after": ll_fitted,
                "improvement": fitted_improvement,
                "accepted": True,
            }

        # 改善不足: 回滚旧参数 (保持 ll_after / improvement 一致性)
        return {
            "new_params": old_params,
            "old_params": old_params,
            "ll_before": ll_before,
            "ll_after": ll_before,
            "improvement": 0.0,
            "accepted": False,
        }

    def should_retrain(
        self,
        last_retrain_count: int,
        current_count: int,
        drift_detected: bool,
    ) -> bool:
        """判断是否应该触发重训练.

        触发条件 (任一满足):
        1. 检测到概念漂移 (``drift_detected=True``);
        2. 距上次重训练的记录数超过 ``max_retraining_interval``.

        Args:
            last_retrain_count: 上次重训练时的记录计数.
            current_count: 当前记录计数.
            drift_detected: 是否检测到漂移.

        Returns:
            True 表示应触发重训练.
        """
        if drift_detected:
            return True
        return (current_count - last_retrain_count) >= self.max_retraining_interval


# ============================================================
# 3. 学习者生命周期管理器
# ============================================================


class LearnerLifecycleManager:
    """学习者生命周期管理器.

    融合世界先进方案:
    - Knewton: 学习者状态生命周期管理
    - ALEKS: 知识状态转换与课程进度追踪
    - Duolingo: 学习者技能树成长轨迹
    - FuMoE-csKT (2026): 冷启动 → 稳态 → 漂移全生命周期

    为每个学习者维护独立漂移检测器 (以传入检测器配置为模板), 并通过
    ``threading.RLock`` 保障并发安全. 漂移触发时调用 ``handle_drift``
    经 ``DriftAwareRetrainer`` 重训练 BKT 参数 (带回滚机制).
    """

    # 各生命周期阶段对应的推荐动作
    PHASE_ACTIONS: dict[str, str] = {
        "cold_start": "use_population_average",
        "warming": "partial_personalization",
        "stable": "continue_monitoring",
        "drifting": "trigger_retraining",
        "recalibrating": "await_convergence",
    }

    def __init__(
        self,
        cold_start_threshold: int = 10,
        drift_detector: LearnerDriftDetector | None = None,
        zpd_calculator: ZPDCalculator | None = None,
    ):
        """初始化生命周期管理器.

        Args:
            cold_start_threshold: 冷启动阈值记录数 (>= 该值视为脱离冷启动).
            drift_detector: 漂移检测器模板 (其配置用于为每个学习者创建独立
                检测器); 为 None 时使用默认配置.
            zpd_calculator: ZPD 计算器 (用于判定 zpd_zone); 为 None 时
                使用默认 ZPDCalculator.
        """
        self.cold_start_threshold = cold_start_threshold
        # 漂移检测器模板 (仅取其配置, 为每个学习者创建独立实例)
        self._drift_template = (
            drift_detector if drift_detector is not None else LearnerDriftDetector()
        )
        self.zpd_calculator = (
            zpd_calculator if zpd_calculator is not None else ZPDCalculator()
        )

        # 每个学习者的内部状态 (受 _lock 保护)
        self._states: dict[str, LearnerLifecycleState] = {}
        self._drift_detectors: dict[str, LearnerDriftDetector] = {}
        self._bkt_params: dict[str, dict[str, float]] = {}
        self._recalibrating: set[str] = set()
        self._lock = threading.RLock()

    # --- 漂移检测器管理 (调用方持锁) ---

    def _get_drift_detector(self, learner_id: str) -> LearnerDriftDetector:
        """获取 (或创建) 学习者专属漂移检测器.

        以 ``_drift_template`` 的配置 (adwin_delta / ddm_warning_level /
        ddm_drift_level) 创建独立实例, 确保各学习者漂移检测互不干扰.

        调用方须持有 ``_lock``.

        Args:
            learner_id: 学习者 ID.

        Returns:
            该学习者的 LearnerDriftDetector 实例.
        """
        det = self._drift_detectors.get(learner_id)
        if det is None:
            det = LearnerDriftDetector(
                adwin_delta=self._drift_template.adwin_delta,
                ddm_warning_level=self._drift_template.ddm_warning_level,
                ddm_drift_level=self._drift_template.ddm_drift_level,
            )
            self._drift_detectors[learner_id] = det
        return det

    def _estimate_se(self, record_count: int) -> float:
        """根据记录数启发式估计 IRT 标准误 (记录越多 SE 越小).

        Args:
            record_count: 累计记录数.

        Returns:
            估计标准误, 落在 [0.1, 0.5].
        """
        if record_count <= 0:
            return 0.5
        return max(0.1, 0.5 / math.sqrt(record_count))

    # --- 阶段判定 ---

    def get_phase(self, record_count: int, has_drift: bool = False) -> str:
        """根据记录数与漂移标志获取当前生命周期阶段.

        判定优先级:
        1. 检测到漂移 -> "drifting" (漂移优先于冷启动 / 暖启动 / 稳态);
        2. 0 条记录 -> "cold_start";
        3. 1 ~ threshold-1 条 -> "warming";
        4. >= threshold 条 -> "stable".

        注: "recalibrating" 阶段由管理器在重训练期间内部标记
        (``update`` 检测到该学习者处于重训练集合时返回 "recalibrating"),
        不由本方法直接返回.

        Args:
            record_count: 累计记录数.
            has_drift: 是否检测到概念漂移.

        Returns:
            生命周期阶段字符串.
        """
        if has_drift:
            return "drifting"
        if record_count == 0:
            return "cold_start"
        if record_count < self.cold_start_threshold:
            return "warming"
        return "stable"

    # --- 置信度 ---

    def get_confidence(
        self,
        record_count: int,
        se: float,
        has_drift: bool,
    ) -> float:
        """计算估计置信度.

        公式:
            confidence = (1 - min(1, se)) * min(1, record_count / threshold)
                         * (0.5 if has_drift else 1.0)

        三个因子:
        - SE 因子: 标准误越小置信度越高 (se>=1 时为 0);
        - 数据因子: 记录数达阈值时为 1, 不足时线性增长;
        - 漂移因子: 检测到漂移时置信度折半.

        Args:
            record_count: 累计记录数.
            se: IRT 估计标准误.
            has_drift: 是否检测到漂移.

        Returns:
            置信度, 落在 [0.0, 1.0].
        """
        se_factor = 1.0 - min(1.0, se)
        if self.cold_start_threshold > 0:
            data_factor = min(1.0, record_count / self.cold_start_threshold)
        else:
            data_factor = 1.0
        drift_factor = 0.5 if has_drift else 1.0
        return se_factor * data_factor * drift_factor

    # --- 推荐动作 ---

    def recommend_action(self, state: LearnerLifecycleState) -> str:
        """根据当前生命周期阶段推荐动作.

        映射:
        - cold_start     -> "use_population_average"
        - warming        -> "partial_personalization"
        - stable         -> "continue_monitoring"
        - drifting       -> "trigger_retraining"
        - recalibrating  -> "await_convergence"

        未知阶段回退为 "continue_monitoring".

        Args:
            state: 学习者生命周期状态.

        Returns:
            推荐动作字符串.
        """
        return self.PHASE_ACTIONS.get(state.phase, "continue_monitoring")

    # --- 状态更新 ---

    def update(
        self,
        learner_id: str,
        record_count: int,
        observation: float,
        theta: float = 0.0,
        mastery: float = 0.5,
        difficulty: float = 0.5,
    ) -> LearnerLifecycleState:
        """更新学习者生命周期状态.

        步骤:
        1. 向学习者专属漂移检测器添加观测值, 检测漂移;
        2. 判定阶段 (若该学习者正在重训练则为 "recalibrating");
        3. 基于 theta / difficulty 判定 ZPD 区域;
        4. 依据记录数启发式估计 SE, 计算置信度;
        5. 更新累计漂移计数 / 上次漂移时间, 写入状态并返回.

        线程安全: 全程持有 ``_lock``.

        Args:
            learner_id: 学习者 ID.
            record_count: 当前累计记录数.
            observation: 本次观测值 (如正确率 0/1).
            theta: 当前 IRT 能力值, 默认 0.0.
            mastery: 当前平均掌握度, 默认 0.5.
            difficulty: 本次题目难度, 默认 0.5 (用于 ZPD 判定).

        Returns:
            更新后的 LearnerLifecycleState.
        """
        with self._lock:
            det = self._get_drift_detector(learner_id)
            drift_result = det.add_observation(observation)
            has_drift = bool(drift_result["drift_detected"])

            # 阶段: 重训练中优先标记为 recalibrating
            if learner_id in self._recalibrating:
                phase = "recalibrating"
            else:
                phase = self.get_phase(record_count, has_drift)

            # ZPD 区域
            zpd_zone = self.zpd_calculator.classify_item(theta, difficulty)

            # 置信度
            se = self._estimate_se(record_count)
            confidence = self.get_confidence(record_count, se, has_drift)

            # 累计漂移计数 / 上次漂移时间
            prev = self._states.get(learner_id)
            drift_count = (prev.drift_count if prev else 0) + (1 if has_drift else 0)
            recalibration_count = prev.recalibration_count if prev else 0
            if has_drift:
                last_drift_time = time.time()
            else:
                last_drift_time = prev.last_drift_time if prev else None

            state = LearnerLifecycleState(
                learner_id=learner_id,
                phase=phase,
                record_count=record_count,
                last_drift_time=last_drift_time,
                drift_count=drift_count,
                recalibration_count=recalibration_count,
                current_theta=theta,
                current_mastery=mastery,
                zpd_zone=zpd_zone,
                confidence=confidence,
            )
            self._states[learner_id] = state
            return state

    # --- 漂移处理 (重训练) ---

    def handle_drift(
        self,
        learner_id: str,
        records: list[AnswerRecord],
        bkt_tracer: BKTTracer,
    ) -> dict[str, Any]:
        """漂移处理: 触发 BKT 参数重训练, 依据改善决定接受 / 回滚.

        流程:
        1. 取该学习者当前 BKT 参数 (默认 DEFAULT_BKT_PARAMS);
        2. 标记该学习者为 "recalibrating" (期间并发 update 返回该阶段);
        3. 经 ``DriftAwareRetrainer.retrain`` 重训练;
        4. 若被接受, 更新该学习者的 BKT 参数并递增 recalibration_count;
        5. 清除 "recalibrating" 标记.

        Args:
            learner_id: 学习者 ID.
            records: 答题记录列表 (用于重训练).
            bkt_tracer: BKT 追踪引擎.

        Returns:
            重训练结果: {old_params, new_params, ll_improvement, accepted}.
            其中 ll_improvement 为重训练带来的对数似然改善 (回滚时为 0.0).
        """
        with self._lock:
            current_params = self._bkt_params.get(
                learner_id, dict(DEFAULT_BKT_PARAMS)
            )
            self._recalibrating.add(learner_id)
            try:
                retrainer = DriftAwareRetrainer(bkt_tracer)
                retrain_result = retrainer.retrain(records, current_params)
                accepted = bool(retrain_result["accepted"])

                if accepted:
                    self._bkt_params[learner_id] = dict(retrain_result["new_params"])
                    prev = self._states.get(learner_id)
                    if prev is not None:
                        prev.recalibration_count += 1

                return {
                    "old_params": retrain_result["old_params"],
                    "new_params": retrain_result["new_params"],
                    "ll_improvement": retrain_result["improvement"],
                    "accepted": accepted,
                }
            finally:
                self._recalibrating.discard(learner_id)

    # --- 摘要 ---

    def get_lifecycle_summary(self, learner_id: str) -> dict[str, Any]:
        """返回学习者生命周期摘要.

        Args:
            learner_id: 学习者 ID.

        Returns:
            摘要字典. 已知学习者含全部状态字段 + recommended_action + exists=True;
            未知学习者返回 ``{"learner_id": ..., "exists": False}``.
        """
        with self._lock:
            state = self._states.get(learner_id)
            if state is None:
                return {"learner_id": learner_id, "exists": False}
            summary = state.to_dict()
            summary["recommended_action"] = self.recommend_action(state)
            summary["exists"] = True
            return summary


# ============================================================
# __all__
# ============================================================

__all__ = [
    "LearnerLifecycleState",
    "LearnerLifecycleManager",
    "DriftAwareRetrainer",
]
