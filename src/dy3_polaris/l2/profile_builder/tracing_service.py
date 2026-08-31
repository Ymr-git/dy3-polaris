"""画像全链路编排服务 — 冷启动 → 画像构建 → 风格推断 → Bloom设定 → 漂移检测 → 重训练.

融合世界先进方案:
- Knewton: 三引擎架构 (评估/策略/反馈) + 冷启动降级
- ALEKS: 知识状态集合 + 画像
- Khan Academy: 综合学情画像
- ADWIN/DDM: 概念漂移检测 (Gama 2010)
- FuMoE-csKT (2026): 冷启动 → 稳态 → 漂移全生命周期

全链路处理流程:
1. 获取/初始化 IRT 状态 (冷启动默认 theta=0.0, se=0.5)
2. 递增记录计数
3. 冷启动检查: 冷启动时使用群体先验与观测值的加权混合
4. 持久化答题记录
5. 获取全部追踪状态
6. 通过 ProfileBuilder 构建画像 (冷启动时覆盖默认风格为 multimodal)
7. 保存画像快照到 store
8. 向漂移检测器添加观测值 (correct=1.0, wrong=0.0)
9. 更新生命周期管理器
10. 构建并返回 ProfileOutput

ProfileOutput 契约字段 (供下游 T2/T3/T4 消费):
- learner_id         : 学习者 ID
- phase              : 生命周期阶段 (cold_start/warming/stable/drifting/recalibrating)
- theta              : IRT 能力值 (冷启动时为群体先验与观测值的加权混合)
- level              : 能力等级 (beginner/intermediate/advanced)
- learning_style     : 学习风格 (visual/aural/reading/kinesthetic/multimodal)
- bloom_target       : Bloom 认知目标层次
- kp_mastery         : 知识点掌握度映射
- weak_kps           : 薄弱知识点列表
- confidence         : 画像置信度 [0, 1]
- drift_detected     : 本次事件是否检测到漂移
- drift_count        : 累计漂移次数
- recommended_action : 生命周期推荐动作
- snapshot_ts        : 快照时间戳
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dy3_polaris.l2.interaction.event_types import AnswerEvent, BehaviorEvent
from dy3_polaris.l2.knowledge_tracer.bkt import BKTTracer
from dy3_polaris.l2.models import (
    DEFAULT_INITIAL_SE,
    DEFAULT_INITIAL_THETA,
    DEFAULT_IRT_A,
    DEFAULT_IRT_C,
    IRT_THETA_MAX,
    IRT_THETA_MIN,
    AnswerRecord,
    IRTState,
    LearnerSnapshot,
)
from dy3_polaris.l2.profile_builder.builder import ProfileBuilder
from dy3_polaris.l2.profile_builder.cold_start import LearnerColdStartManager
from dy3_polaris.l2.profile_builder.drift_detector import LearnerDriftDetector
from dy3_polaris.l2.profile_builder.lifecycle_manager import LearnerLifecycleManager
from dy3_polaris.l2.store import InMemoryL2Store, L2Store


# ============================================================
# 常量定义
# ============================================================

# 冷启动默认 IRT 参数 — 统一到 models.py 单一事实来源
_DEFAULT_IRT_THETA: float = DEFAULT_INITIAL_THETA
_DEFAULT_IRT_SE: float = DEFAULT_INITIAL_SE

# 学习行为 → 掌握度信号 (学习是「弱正信号」: 学习≠展示掌握, 故增益小且有上限)
_STUDY_MASTERY_GAIN: float = 0.04   # 单次高质量学习事件的最大掌握度增益
_STUDY_MASTERY_CAP: float = 0.75    # 仅靠学习能达到的掌握度上限 (< 展示掌握 0.85)
_STUDY_FULL_DURATION: float = 600.0  # 达到满增益所需学习时长 (秒)

# 掌握度平滑 (缓存分数): 单次作答最多移动剩余差距的比例.
# 直击「多次测试后才确认」诉求 —— 一次答对/答错不再让掌握度 0.18↔0.92 大幅跳变,
# 需多次一致证据才逐步收敛到 BKT 后验, 从而抑制能力等级横跳并诚实化掌握度.
_BKT_SMOOTH_ALPHA: float = 0.3


# ============================================================
# ProfileOutput — 下游输出标准化契约
# ============================================================


@dataclass
class ProfileOutput:
    """画像全链路输出 — 标准化画像契约.

    供下游 T2/T3/T4 (CAT 选题 / 推荐决策 / 画像着色 / 预警) 消费.

    Attributes:
        learner_id: 学习者 ID.
        phase: 生命周期阶段
            (cold_start/warming/stable/drifting/recalibrating).
        theta: IRT 能力值 (冷启动时为群体先验与观测值的加权混合).
        level: 能力等级 (beginner/intermediate/advanced).
        learning_style: 学习风格
            (visual/aural/reading/kinesthetic/multimodal).
        bloom_target: Bloom 认知目标层次
            (remember/understand/apply/analyze/evaluate/create).
        kp_mastery: 知识点掌握度映射 {kp_id: mastery_prob}.
        weak_kps: 薄弱知识点 ID 列表.
        confidence: 画像置信度 [0.0, 1.0].
        drift_detected: 本次事件是否检测到概念漂移.
        drift_count: 累计检测到的漂移次数.
        recommended_action: 生命周期推荐动作.
        snapshot_ts: 画像快照时间戳 (秒, float).
    """

    learner_id: str
    phase: str
    theta: float
    level: str
    learning_style: str
    bloom_target: str
    kp_mastery: dict[str, float] = field(default_factory=dict)
    weak_kps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    drift_detected: bool = False
    drift_count: int = 0
    recommended_action: str = "continue_monitoring"
    snapshot_ts: float = 0.0
    # --- 增强字段 (T5 增强) ---
    score_overall: float | None = None
    bottleneck_nodes: list[dict[str, Any]] = field(default_factory=list)
    forgetting_alerts: list[dict[str, Any]] = field(default_factory=list)
    kp_centrality: dict[str, float] = field(default_factory=dict)
    mastery_threshold: float | None = None
    profile_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (可变字段浅拷贝避免共享引用)."""
        return {
            "learner_id": self.learner_id,
            "phase": self.phase,
            "theta": self.theta,
            "level": self.level,
            "learning_style": self.learning_style,
            "bloom_target": self.bloom_target,
            "kp_mastery": dict(self.kp_mastery),
            "weak_kps": list(self.weak_kps),
            "confidence": self.confidence,
            "drift_detected": self.drift_detected,
            "drift_count": self.drift_count,
            "recommended_action": self.recommended_action,
            "snapshot_ts": self.snapshot_ts,
            "score_overall": self.score_overall,
            "bottleneck_nodes": list(self.bottleneck_nodes),
            "forgetting_alerts": list(self.forgetting_alerts),
            "kp_centrality": dict(self.kp_centrality),
            "mastery_threshold": self.mastery_threshold,
            "profile_version": self.profile_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProfileOutput:
        """从字典反序列化."""
        return cls(
            learner_id=d["learner_id"],
            phase=d["phase"],
            theta=d["theta"],
            level=d["level"],
            learning_style=d["learning_style"],
            bloom_target=d["bloom_target"],
            kp_mastery=dict(d.get("kp_mastery", {})),
            weak_kps=list(d.get("weak_kps", [])),
            confidence=d.get("confidence", 0.0),
            drift_detected=d.get("drift_detected", False),
            drift_count=d.get("drift_count", 0),
            recommended_action=d.get("recommended_action", "continue_monitoring"),
            snapshot_ts=d.get("snapshot_ts", 0.0),
            score_overall=d.get("score_overall"),
            bottleneck_nodes=list(d.get("bottleneck_nodes", [])),
            forgetting_alerts=list(d.get("forgetting_alerts", [])),
            kp_centrality=dict(d.get("kp_centrality", {})),
            mastery_threshold=d.get("mastery_threshold"),
            profile_version=d.get("profile_version"),
        )

    def to_api_response(self) -> dict[str, Any]:
        """转换为 API 响应格式 (JSON 可序列化).

        Returns:
            包含所有字段 (含增强字段) 的字典.
        """
        return self.to_dict()

    @classmethod
    def from_irt_output(cls, irt_output: Any) -> ProfileOutput:
        """从 IRT AbilityOutput 构建 ProfileOutput (跨模块互操作).

        Args:
            irt_output: IRT 输出对象 (需有 learner_id / theta / se / level 属性).

        Returns:
            ProfileOutput 实例.
        """
        return cls(
            learner_id=getattr(irt_output, "learner_id", "unknown"),
            phase="stable",
            theta=getattr(irt_output, "theta", 0.0),
            level=getattr(irt_output, "level", "beginner"),
            learning_style="multimodal",
            bloom_target="understand",
            confidence=1.0 / (1.0 + getattr(irt_output, "se", 0.5)),
        )


# ============================================================
# ProfileTracingService — 画像全链路编排器
# ============================================================


class ProfileTracingService:
    """画像全链路编排服务 — 冷启动 → 画像构建 → 风格推断 → Bloom设定 → 漂移检测 → 重训练.

    编排全链路处理流程:
    1. 获取/初始化 IRT 状态 (冷启动默认 theta=0.0, se=0.5)
    2. 递增记录计数
    3. 冷启动检查: 使用群体先验与观测值的加权混合 (Knewton 式降级)
    4. 持久化答题记录
    5. 获取全部追踪状态
    6. 通过 ProfileBuilder 构建画像 (冷启动时覆盖默认风格为 multimodal)
    7. 保存画像快照到 store
    8. 向漂移检测器添加观测值 (correct=1.0, wrong=0.0)
    9. 更新生命周期管理器 (阶段转换 / 漂移计数 / ZPD / 置信度)
    10. 构建并返回 ProfileOutput

    支持特性:
    - 依赖注入: store / profile_builder / cold_start_manager / drift_detector /
      lifecycle_manager 均可外部注入, 为 None 时使用默认实例.
    - 冷启动降级: 0 条记录使用群体平均, 1-9 条使用加权混合, 10+ 条全量个性化.
    - 漂移检测: 每学习者独立漂移检测器 (ADWIN + DDM), 支持重训练回调.
    - 生命周期管理: 冷启动 → 暖启动 → 稳态 → 漂移 → 重训练.
    - 优雅降级: store=None 时使用内部内存存储.

    Args:
        store: L2 存储层. 为 None 时使用内部 InMemoryL2Store.
        profile_builder: 画像构建器. 为 None 时使用 ProfileBuilder(store=None) (纯构建, 不自动持久化).
        cold_start_manager: 冷启动策略管理器. 为 None 时使用默认实例.
        drift_detector: 漂移检测器模板 (配置源, 为每学习者创建独立实例).
            为 None 时使用默认实例.
        lifecycle_manager: 生命周期管理器. 为 None 时使用默认实例.

    Attributes:
        store: L2 存储层.
        profile_builder: 画像构建器.
        cold_start_manager: 冷启动策略管理器.
        drift_detector: 漂移检测器模板 (配置源).
        lifecycle_manager: 生命周期管理器.
    """

    def __init__(
        self,
        store: L2Store | None = None,
        profile_builder: ProfileBuilder | None = None,
        cold_start_manager: LearnerColdStartManager | None = None,
        drift_detector: LearnerDriftDetector | None = None,
        lifecycle_manager: LearnerLifecycleManager | None = None,
        enable_enhanced: bool = False,
    ) -> None:
        """初始化画像全链路编排服务.

        Args:
            store: L2 存储层. 为 None 时使用内部 InMemoryL2Store.
            profile_builder: 画像构建器. 为 None 时使用 ProfileBuilder(store=None) (纯构建, 不自动持久化).
            cold_start_manager: 冷启动策略管理器. 为 None 时使用默认实例.
            drift_detector: 漂移检测器模板. 为 None 时使用默认实例.
            lifecycle_manager: 生命周期管理器. 为 None 时使用默认实例.
            enable_enhanced: 是否启用增强功能 (多维融合/KST/置信度/遗忘预警).
        """
        self.store: L2Store = store if store is not None else InMemoryL2Store()
        # 注意: 这里不向 ProfileBuilder 注入 store —— build() 会在 store 非 None 时
        # 自动 save_profile, 而本服务 process()/apply_update() 已显式 save_profile
        # (含乐观锁 CAS / 版本递增)。二者叠加会造成双写, 版本号每次 +2 且 apply_update
        # 的 CAS 恒冲突。故服务侧由本类统一负责持久化, build() 仅纯构建。
        self.profile_builder: ProfileBuilder = (
            profile_builder
            if profile_builder is not None
            else ProfileBuilder(store=None)
        )
        self.cold_start_manager: LearnerColdStartManager = (
            cold_start_manager
            if cold_start_manager is not None
            else LearnerColdStartManager()
        )
        self.drift_detector: LearnerDriftDetector = (
            drift_detector
            if drift_detector is not None
            else LearnerDriftDetector()
        )
        self.lifecycle_manager: LearnerLifecycleManager = (
            lifecycle_manager
            if lifecycle_manager is not None
            else LearnerLifecycleManager()
        )
        # BKT 引擎 (用于 handle_drift 重训练)
        self._bkt_tracer: BKTTracer = BKTTracer()
        # 每学习者记录计数
        self._record_counts: dict[str, int] = {}
        # 每学习者学习行为统计 (study/review 视为学习信号, skip 视为脱离信号)
        self._study_stats: dict[str, dict[str, float]] = {}
        # 每学习者漂移检测器 (以 drift_detector 模板配置创建)
        self._drift_detectors: dict[str, LearnerDriftDetector] = {}

        # --- 增强组件 (T5 增强) ---
        self.enable_enhanced: bool = enable_enhanced
        if enable_enhanced:
            from dy3_polaris.l2.profile_builder.enhanced import (
                MultiDimensionalFuser,
                KSTAnalyzer,
                ProfileConfidenceEstimator,
                DynamicMasteryThreshold,
                ForgettingAlertGenerator,
                ProfileVersionManager,
            )
            self._fuser = MultiDimensionalFuser()
            self._kst_analyzer = KSTAnalyzer()
            self._confidence_estimator = ProfileConfidenceEstimator()
            self._mastery_thresholder = DynamicMasteryThreshold()
            self._forgetting_generator = ForgettingAlertGenerator()
            self._version_manager = ProfileVersionManager()
        else:
            self._fuser = None
            self._kst_analyzer = None
            self._confidence_estimator = None
            self._mastery_thresholder = None
            self._forgetting_generator = None
            self._version_manager = None

    # ============================================================
    # 漂移检测器管理
    # ============================================================

    def _get_drift_detector(self, learner_id: str) -> LearnerDriftDetector:
        """获取 (或创建) 学习者专属漂移检测器.

        以 ``self.drift_detector`` 模板的配置 (adwin_delta /
        ddm_warning_level / ddm_drift_level) 创建独立实例,
        确保各学习者漂移检测互不干扰.

        Args:
            learner_id: 学习者 ID.

        Returns:
            该学习者的 LearnerDriftDetector 实例.
        """
        det = self._drift_detectors.get(learner_id)
        if det is None:
            det = LearnerDriftDetector(
                adwin_delta=self.drift_detector.adwin_delta,
                ddm_warning_level=self.drift_detector.ddm_warning_level,
                ddm_drift_level=self.drift_detector.ddm_drift_level,
            )
            self._drift_detectors[learner_id] = det
        return det

    def set_retraining_callback(
        self,
        learner_id: str,
        callback: Callable[[dict], Any] | None,
    ) -> None:
        """设置漂移触发时的重训练回调函数 (按学习者).

        设置后, 每次 ``process`` 中检测到漂移时会自动调用该回调,
        回调签名: ``callback(drift_info: dict) -> Any``.

        传 None 可清除已设置的回调.

        Args:
            learner_id: 学习者 ID.
            callback: 漂移触发回调; None 表示清除回调.
        """
        det = self._get_drift_detector(learner_id)
        det.set_retraining_callback(callback)

    def get_drift_history(self, learner_id: str) -> list[dict[str, Any]]:
        """获取学习者漂移事件历史.

        Args:
            learner_id: 学习者 ID.

        Returns:
            漂移事件字典列表 (按发生顺序); 无漂移或未知学习者时为空列表.
        """
        det = self._drift_detectors.get(learner_id)
        if det is None:
            return []
        return det.get_drift_history()

    # ============================================================
    # 画像快照与生命周期查询
    # ============================================================

    def get_profile_snapshot(self, learner_id: str) -> LearnerSnapshot | None:
        """获取学习者当前画像快照.

        Args:
            learner_id: 学习者 ID.

        Returns:
            LearnerSnapshot 或 None (不存在时).
        """
        return self.store.get_profile(learner_id)

    def get_lifecycle_summary(self, learner_id: str) -> dict[str, Any]:
        """获取学习者生命周期摘要.

        Args:
            learner_id: 学习者 ID.

        Returns:
            摘要字典. 已知学习者含全部状态字段 + recommended_action + exists=True;
            未知学习者返回 ``{"learner_id": ..., "exists": False}``.
        """
        return self.lifecycle_manager.get_lifecycle_summary(learner_id)

    # ============================================================
    # 漂移处理 (重训练)
    # ============================================================

    def handle_drift(
        self,
        learner_id: str,
        records: list[AnswerRecord],
    ) -> dict[str, Any]:
        """触发漂移处理: BKT 参数重训练.

        委托 ``LearnerLifecycleManager.handle_drift``, 内部经
        ``DriftAwareRetrainer`` 重训练并依据改善阈值决定接受 / 回滚.

        Args:
            learner_id: 学习者 ID.
            records: 答题记录列表 (用于重训练).

        Returns:
            重训练结果: {old_params, new_params, ll_improvement, accepted}.
        """
        return self.lifecycle_manager.handle_drift(
            learner_id, records, self._bkt_tracer
        )

    # ============================================================
    # 批量处理
    # ============================================================

    def batch_process(
        self,
        events: list[AnswerEvent],
    ) -> list[ProfileOutput]:
        """批量处理答题事件 (按时间戳升序).

        Args:
            events: 答题事件列表.

        Returns:
            每个事件对应的 ProfileOutput 列表; 空输入返回空列表.
        """
        if not events:
            return []
        ordered = sorted(events, key=lambda e: e.timestamp)
        return [self.process(ev) for ev in ordered]

    # ============================================================
    # 单事件处理 (全链路)
    # ============================================================

    def process(self, event: AnswerEvent, *, persist_history: bool = True, skip_bkt_update: bool = False) -> ProfileOutput:
        """处理单条答题事件, 返回完整 ProfileOutput.

        全链路流程:
        1. 获取/初始化 IRT 状态 (冷启动默认 theta=0.0, se=0.5)
        2. 递增记录计数
        3. 冷启动检查: 冷启动时使用群体先验与观测值的加权混合
        4. 持久化答题记录
        5. 获取全部追踪状态
        6. 通过 ProfileBuilder 构建画像
        7. 冷启动时覆盖学习风格为 multimodal
        8. 保存画像快照到 store
        9. 向漂移检测器添加观测值
        10. 更新生命周期管理器
        11. 构建并返回 ProfileOutput

        Args:
            event: 答题事件.

        Returns:
            ProfileOutput 标准化输出.
        """
        runtime_started = time.monotonic()
        learner_id = event.learner_id

        # --- 1. 获取或初始化 IRT 状态 ---
        irt_state = self.store.get_irt_state(learner_id)
        if irt_state is None:
            irt_state = IRTState(
                theta=_DEFAULT_IRT_THETA,
                se=_DEFAULT_IRT_SE,
            )

        # --- 1.5 用答题结果更新 IRT 能力 (θ/SE) — 修复: process 此前从不更新 IRT ---
        if getattr(self, "_irt_estimator", None) is None:
            from dy3_polaris.l2.ability_assessor.irt import IRTEstimator

            self._irt_estimator = IRTEstimator()
        _b = max(-3.0, min(3.0, event.difficulty * 6.0 - 3.0))
        _item = {"a": DEFAULT_IRT_A, "b": _b, "c": DEFAULT_IRT_C}
        irt_state = self._irt_estimator.update_theta(
            irt_state, _item, event.correct
        )
        irt_state.theta = max(IRT_THETA_MIN, min(IRT_THETA_MAX, irt_state.theta))
        self.store.save_irt_state(learner_id, irt_state)

        # --- 2. 递增记录计数 ---
        count = self._record_counts.get(learner_id, 0) + 1
        self._record_counts[learner_id] = count

        # --- 3. 冷启动检查与 theta 混合 ---
        is_cold = self.cold_start_manager.is_cold_start(count)
        if is_cold:
            blended_theta, blended_se = (
                self.cold_start_manager.estimate_initial_theta(
                    irt_state.theta, count, observed_se=irt_state.se
                )
            )
            effective_irt = IRTState(
                theta=blended_theta,
                se=blended_se,
                response_count=irt_state.response_count,
                last_update_time=event.timestamp,
            )
        else:
            effective_irt = irt_state

        # --- 4. 持久化答题记录 (practice 路径 BKT 已写, 传 persist_history=False 避免双写) ---
        history = self.store.get_answer_history(learner_id)
        if history is None:
            history = []
        if persist_history:
            history = list(history)
            history.append(event.to_answer_record())
            self.store.save_answer_history(learner_id, history)

        # --- 5. 获取全部追踪状态 (practice 路径已由 BKTTracingService 更新, 这里可跳过避免双重 BKT) ---
        if not skip_bkt_update:
            _cur = self.store.get_tracing_state(learner_id, event.kp_id)
            if _cur is None:
                _cur = self._bkt_tracer.init_state(event.kp_id, event.difficulty)
            _new = self._bkt_tracer.update(_cur, event.correct, event.timestamp)
            # 缓存分数: 掌握度平滑 (多次一致作答才确认), 抑制单题振荡与等级横跳
            _new.mastery_prob = self._smooth_mastery(_cur.mastery_prob, _new.mastery_prob)
            self.store.save_tracing_state(learner_id, event.kp_id, _new)
        tracing_states = self.store.get_all_tracing_states(learner_id)

        # --- 6. 构建画像 (透传历史等级 + 作答次数, 供等级滞回/置信门控) ---
        _prev_snapshot = self.store.get_profile(learner_id)
        _prev_level = _prev_snapshot.level if _prev_snapshot is not None else None
        snapshot = self.profile_builder.build(
            learner_id=learner_id,
            tracing_states=tracing_states,
            irt_state=effective_irt,
            interaction_history=history,
            prev_level=_prev_level,
            response_count=count,
        )

        # --- 7. 冷启动时覆盖学习风格为 multimodal ---
        if is_cold:
            snapshot = LearnerSnapshot(
                learner_id=snapshot.learner_id,
                snapshot_ts=snapshot.snapshot_ts,
                kp_mastery=snapshot.kp_mastery,
                theta=snapshot.theta,
                level=snapshot.level,
                learning_style=self.cold_start_manager.get_default_learning_style(),
                bloom_target=snapshot.bloom_target,
                weak_kps=snapshot.weak_kps,
                confidence=snapshot.confidence,
            )

        # --- 8. 保存画像快照 ---
        self.store.save_profile(learner_id, snapshot)

        # --- 9. 漂移检测 ---
        det = self._get_drift_detector(learner_id)
        observation = 1.0 if event.correct else 0.0
        drift_result = det.add_observation(observation)
        drift_detected = bool(drift_result["drift_detected"])

        # --- 10. 更新生命周期管理器 ---
        avg_mastery = (
            sum(snapshot.kp_mastery.values()) / len(snapshot.kp_mastery)
            if snapshot.kp_mastery
            else 0.5
        )
        lifecycle_state = self.lifecycle_manager.update(
            learner_id=learner_id,
            record_count=count,
            observation=observation,
            theta=effective_irt.theta,
            mastery=avg_mastery,
            difficulty=event.difficulty,
        )

        # --- 11. 构建并返回 ProfileOutput ---
        recommended_action = self.lifecycle_manager.recommend_action(
            lifecycle_state
        )

        # --- 12. 增强字段填充 (T5 增强) ---
        score_overall = None
        bottleneck_nodes: list[dict[str, Any]] = []
        forgetting_alerts: list[dict[str, Any]] = []
        kp_centrality: dict[str, float] = {}
        mastery_threshold = None
        profile_version = None

        if self.enable_enhanced:
            # 多维融合评分
            score_overall = self._fuser.fuse(
                kp_mastery=snapshot.kp_mastery,
                subject_background={},
                behavior_features={},
                theta=effective_irt.theta,
            )
            # 动态掌握阈值
            mastery_threshold = self._mastery_thresholder.get_threshold(
                snapshot.level
            )
            # 遗忘预警
            forgetting_alerts = self._forgetting_generator.generate_alerts(
                tracing_states=tracing_states,
                bloom_level=snapshot.bloom_target,
            )
            # 版本管理
            profile_version = self._version_manager.save(learner_id, snapshot)

        output = ProfileOutput(
            learner_id=learner_id,
            phase=lifecycle_state.phase,
            theta=effective_irt.theta,
            level=snapshot.level,
            learning_style=snapshot.learning_style,
            bloom_target=snapshot.bloom_target,
            kp_mastery=snapshot.kp_mastery,
            weak_kps=snapshot.weak_kps,
            confidence=snapshot.confidence,
            drift_detected=drift_detected,
            drift_count=lifecycle_state.drift_count,
            recommended_action=recommended_action,
            snapshot_ts=snapshot.snapshot_ts,
            score_overall=score_overall,
            bottleneck_nodes=bottleneck_nodes,
            forgetting_alerts=forgetting_alerts,
            kp_centrality=kp_centrality,
            mastery_threshold=mastery_threshold,
            profile_version=profile_version,
        )
        self._last_runtime_metrics = {
            "profile_update_ms": round((time.monotonic() - runtime_started) * 1000.0, 3),
        }
        return output

    # ============================================================
    # 学习行为事件处理 (study/review/skip) — 画像维度补齐
    # ============================================================

    @staticmethod
    def _smooth_mastery(prev: float, raw: float) -> float:
        """掌握度平滑 (缓存分数): 单次作答只移动剩余差距的 α 比例.

        直击「多次测试后才确认」诉求 —— 一次答对/答错不再让掌握度大幅跳变,
        需多次一致证据才逐步收敛到 BKT 后验, 从而:
        - 抑制单题引起的掌握度振荡 (0.18 ↔ 0.92) 与能力等级横跳;
        - 诚实化: 「学了但学不进去」者即便短期答对易题, 掌握度也不会瞬间飙升.

        Args:
            prev: 上一次展示/持久化的掌握度 (作为缓存分数).
            raw: 本次 BKT 后验 (未平滑).

        Returns:
            平滑后的掌握度, 落在 [prev, raw] 之间 (向 raw 靠近 α 比例).
        """
        alpha = _BKT_SMOOTH_ALPHA
        return prev + alpha * (raw - prev)

    def process_behavior(self, event: BehaviorEvent) -> ProfileOutput:
        """处理学习行为事件 (study/review/skip), 让「学习」信号进入画像.

        此前 BehaviorEvent 是画像 no-op, 导致「好好学习但不测试」的学习者完全不可见:
        能力 θ 只由答题更新, 不测试则画像永远停留在冷启动。本方法补齐行为维度:

        - study/review: 对关联知识点施加「有界正增益」 (学习≠展示掌握, 增益小且封顶
          `_STUDY_MASTERY_CAP`), 并按学习时长缩放质量 (时长短 → 增益小, 对应「学不进去」).
        - skip: 记录脱离次数 (不改变掌握度), 供下游诊断「挫败/放弃」信号.
        - 不改变 IRT θ (能力仍以作答为准, 保持诚实: 学习不能伪造展示出的能力).

        Args:
            event: 学习行为事件.

        Returns:
            ProfileOutput 标准化输出 (画像已反映学习行为).
        """
        learner_id = event.learner_id
        stats = self._study_stats.setdefault(
            learner_id, {"study_count": 0, "review_count": 0, "skip_count": 0,
                         "study_duration": 0.0}
        )

        action = getattr(event, "action", "")
        if action in ("study", "review"):
            stats["study_count" if action == "study" else "review_count"] += 1
            stats["study_duration"] += max(0.0, getattr(event, "duration", 0.0))
        elif action == "skip":
            stats["skip_count"] += 1
        else:
            return self._build_profile_output(learner_id)

        # study/review 对关联知识点施加有界掌握度增益 (按时长缩放质量)
        if action in ("study", "review") and getattr(event, "kp_id", None):
            kp_id = event.kp_id
            cur = self.store.get_tracing_state(learner_id, kp_id)
            if cur is None:
                cur = self._bkt_tracer.init_state(kp_id, 0.5)
            duration = max(0.0, getattr(event, "duration", 0.0))
            quality = min(1.0, duration / _STUDY_FULL_DURATION)
            gain = _STUDY_MASTERY_GAIN * quality
            new_mastery = min(_STUDY_MASTERY_CAP, cur.mastery_prob + gain)
            cur.mastery_prob = new_mastery
            cur.last_attempt_time = event.timestamp  # 学习刷新记忆, 避免立刻遗忘
            self.store.save_tracing_state(learner_id, kp_id, cur)

        return self._build_profile_output(learner_id)

    def _build_profile_output(self, learner_id: str) -> ProfileOutput:
        """基于当前 store 状态重建画像快照并返回 ProfileOutput (供行为事件/旁路复用)."""
        irt_state = self.store.get_irt_state(learner_id)
        if irt_state is None:
            irt_state = IRTState(theta=_DEFAULT_IRT_THETA, se=_DEFAULT_IRT_SE)
        tracing_states = self.store.get_all_tracing_states(learner_id)
        history = self.store.get_answer_history(learner_id) or []
        count = self._record_counts.get(learner_id, len(history))

        _prev_snapshot = self.store.get_profile(learner_id)
        _prev_level = _prev_snapshot.level if _prev_snapshot is not None else None
        snapshot = self.profile_builder.build(
            learner_id=learner_id,
            tracing_states=tracing_states,
            irt_state=irt_state,
            interaction_history=history,
            prev_level=_prev_level,
            response_count=count,
        )
        # 学习行为统计写入快照 extras, 供下游诊断读取
        snapshot.extras = dict(snapshot.extras or {})
        snapshot.extras["study_stats"] = dict(self._study_stats.get(learner_id, {}))
        self.store.save_profile(learner_id, snapshot)

        avg_mastery = (
            sum(snapshot.kp_mastery.values()) / len(snapshot.kp_mastery)
            if snapshot.kp_mastery
            else 0.5
        )
        return ProfileOutput(
            learner_id=learner_id,
            phase=self.lifecycle_manager.get_lifecycle_summary(learner_id).get("phase", "stable"),
            theta=irt_state.theta,
            level=snapshot.level,
            learning_style=snapshot.learning_style,
            bloom_target=snapshot.bloom_target,
            kp_mastery=snapshot.kp_mastery,
            weak_kps=snapshot.weak_kps,
            confidence=snapshot.confidence,
            snapshot_ts=snapshot.snapshot_ts,
        )
    # ============================================================
    # 增强查询方法 (T5 增强)
    # ============================================================

    def apply_update(
        self,
        learner_id: str,
        updates: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> LearnerSnapshot:
        """L2 唯一写方入口: 全量重算画像并合并更新 (乐观锁).

        规则:
        - 校验 expected_version (CAS): 不匹配抛 ProfileConflictError
        - 从 store 的 BKT 追踪状态 + IRT 状态经 ProfileBuilder 全量重建
          (含遗忘衰减/等级/薄弱点重算)
        - 合并 updates: {kp_mastery: {...覆盖}, extras: {...合并}, confidence: float}
        - 无追踪数据时保留现有快照基础字段, 仅应用 updates

        Args:
            learner_id: 学习者 ID.
            updates: 要合并的更新 (kp_mastery/extras/confidence).
            expected_version: 调用方持有的版本号 (乐观锁), None 表示不校验.

        Returns:
            合并后的最新 LearnerSnapshot.

        Raises:
            ProfileConflictError: version 不匹配 (调用方需重新拉取后重试).
        """
        existing = self.store.get_profile(learner_id)
        if existing is not None and expected_version is None:
            expected_version = existing.version

        # 1. 全量重算: 从 BKT 追踪状态 + IRT 重建 (若无状态则保留现有 kp_mastery)
        irt_state = self.store.get_irt_state(learner_id)
        if irt_state is None:
            irt_state = IRTState(theta=_DEFAULT_IRT_THETA, se=_DEFAULT_IRT_SE)
        tracing_states = self.store.get_all_tracing_states(learner_id)
        history = self.store.get_answer_history(learner_id) or []
        base = None
        if tracing_states:
            # 透传历史等级 + 作答次数, 让等级估计走同一套滞回/置信门控,
            # 避免 L5 Agent 写入 / profile_mastery_update 等旁路全量重算时
            # 因丢失 prev_level 而让能力等级标签重新横跳 (与 process() 保持一致).
            _prev_level = existing.level if existing is not None else None
            _response_count = (
                getattr(irt_state, "response_count", None) or len(history)
            )
            base = self.profile_builder.build(
                learner_id=learner_id,
                tracing_states=tracing_states,
                irt_state=irt_state,
                interaction_history=history,
                prev_level=_prev_level,
                response_count=_response_count,
            )
        elif existing is not None:
            base = existing

        if base is None:
            base = LearnerSnapshot(learner_id=learner_id, snapshot_ts=time.time())

        # 2. 合并 updates
        updates = updates or {}
        km = dict(base.kp_mastery)
        km.update(updates.get("kp_mastery", {}) or {})
        base.kp_mastery = km
        base.weak_kps = sorted(
            k for k, m in km.items()
            if m < getattr(self.profile_builder, "_WEAK_KP_THRESHOLD", 0.5)
        )
        extras = dict(base.extras or {})
        for k, v in (updates.get("extras", {}) or {}).items():
            if k in extras and isinstance(extras[k], list) and isinstance(v, list):
                merged_list = list(extras[k])
                seen = {repr(i) for i in merged_list}
                for item in v:
                    key = repr(item)
                    if key not in seen:
                        merged_list.append(item)
                        seen.add(key)
                extras[k] = merged_list[-200:]
            else:
                extras[k] = v
        base.extras = extras
        if "confidence" in updates and updates["confidence"] is not None:
            base.confidence = float(updates["confidence"])
        if "learning_style" in updates:
            base.learning_style = updates["learning_style"]
        if "bloom_target" in updates:
            base.bloom_target = updates["bloom_target"]

        # 3. CAS 保存 (乐观锁)
        return self.store.save_profile(
            learner_id, base, expected_version=expected_version
        )

    def get_weak_points(self, learner_id: str) -> dict[str, Any]:
        """获取薄弱知识点分析 — 含瓶颈与中心性.

        Args:
            learner_id: 学习者 ID.

        Returns:
            含 weak_kps / bottleneck_nodes / kp_centrality_map 的字典.
        """
        snapshot = self.store.get_profile(learner_id)
        if snapshot is None:
            return {
                "weak_kps": [],
                "bottleneck_nodes": [],
                "kp_centrality_map": {},
            }

        weak_kps = list(snapshot.weak_kps)

        if self.enable_enhanced and self._kst_analyzer:
            # 瓶颈检测 (需要 KG 结构, 此处用空结构降级)
            kp_mastery = dict(snapshot.kp_mastery)
            bottlenecks = self._kst_analyzer.detect_bottlenecks(
                kp_mastery, {"nodes": [], "edges": []}
            )
            centrality = self._kst_analyzer.compute_centrality(
                {"nodes": [], "edges": []}
            )
        else:
            bottlenecks = []
            centrality = {}

        return {
            "weak_kps": weak_kps,
            "bottleneck_nodes": bottlenecks,
            "kp_centrality_map": centrality,
        }

    def get_confidence(self, learner_id: str) -> dict[str, Any]:
        """获取置信度报告.

        Args:
            learner_id: 学习者 ID.

        Returns:
            含 overall_confidence / kp_confidence / data_sufficiency 的字典.
        """
        snapshot = self.store.get_profile(learner_id)
        record_count = self._record_counts.get(learner_id, 0)
        irt_state = self.store.get_irt_state(learner_id)
        se = irt_state.se if irt_state else 0.5

        if self.enable_enhanced and self._confidence_estimator:
            overall = self._confidence_estimator.estimate(
                record_count=record_count,
                se=se,
                has_drift=False,
                kp_count=len(snapshot.kp_mastery) if snapshot else 0,
            )
            tracing_states = self.store.get_all_tracing_states(learner_id)
            kp_confidence = {
                kp_id: self._confidence_estimator.estimate_kp_confidence(state)
                for kp_id, state in tracing_states.items()
            } if tracing_states else {}
            data_sufficiency = self._confidence_estimator.estimate_data_sufficiency(
                record_count=record_count,
                kp_count=len(snapshot.kp_mastery) if snapshot else 0,
            )
        else:
            overall = snapshot.confidence if snapshot else 0.0
            kp_confidence = {}
            data_sufficiency = 0.0

        return {
            "overall_confidence": overall,
            "kp_confidence": kp_confidence,
            "data_sufficiency": data_sufficiency,
        }

    def get_skillbook(self, learner_id: str) -> dict[str, Any]:
        """获取技能树 (增强版).

        Args:
            learner_id: 学习者 ID.

        Returns:
            含 global_ability / nodes / edges 的技能树字典.
        """
        from dy3_polaris.l2.skillbook.skill_mapper import SkillMapper
        from dy3_polaris.l2.models import IRTState, TracingState

        tracing_states = self.store.get_all_tracing_states(learner_id)

        # 降级 1: 如果 tracing_states 为空, 从画像快照的 kp_mastery 构建
        if not tracing_states:
            snapshot = self.store.get_profile(learner_id)
            if snapshot and snapshot.kp_mastery:
                tracing_states = {
                    kp_id: TracingState(
                        kp_id=kp_id,
                        mastery_prob=mastery,
                        attempts=1,
                        correct_count=1 if mastery >= 0.5 else 0,
                        last_attempt_time=snapshot.snapshot_ts,
                    )
                    for kp_id, mastery in snapshot.kp_mastery.items()
                }

        # 降级 2: 如果仍为空, 从答题历史构建
        if not tracing_states:
            history = self.store.get_answer_history(learner_id)
            if history:
                kp_seen: dict[str, TracingState] = {}
                for record in history:
                    kp_id = record.kp_id
                    if kp_id not in kp_seen:
                        kp_seen[kp_id] = TracingState(
                            kp_id=kp_id,
                            mastery_prob=0.5,
                            attempts=0,
                            correct_count=0,
                            last_attempt_time=record.timestamp,
                        )
                    state = kp_seen[kp_id]
                    state.attempts += 1
                    if record.correct:
                        state.correct_count += 1
                    state.mastery_prob = state.correct_count / state.attempts if state.attempts > 0 else 0.5
                    state.last_attempt_time = record.timestamp
                tracing_states = kp_seen

        irt_state = self.store.get_irt_state(learner_id)
        if irt_state is None:
            irt_state = IRTState(theta=0.0, se=0.5)

        mapper = SkillMapper()
        skill_tree = mapper.to_skill_tree(
            tracing_states=tracing_states if tracing_states else {},
            irt_state=irt_state,
        )

        return skill_tree


# ============================================================
# __all__
# ============================================================

__all__ = [
    "ProfileTracingService",
    "ProfileOutput",
]
