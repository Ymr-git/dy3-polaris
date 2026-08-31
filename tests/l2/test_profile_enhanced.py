"""T5 画像构建器增强测试 — 世界先进方案融合.

测试覆盖:
1. 多维画像融合 (MultiDimensionalProfileFuser)
   - 五维加权融合: Score_overall = 0.60×μ(M) + 0.20×f(B) + 0.10×g(behavior) + 0.10×θ_norm
   - 学科背景加权: f(B) = weighted_mean(B, [0.3, 0.3, 0.25, 0.15])
   - 行为特征评分: g(behavior) 综合 session/streak/accuracy
   - θ 归一化: sigmoid 变换
2. KST 知识空间理论集成 (KSTAnalyzer)
   - 内/外边界 (inner/outer fringe) 计算
   - 瓶颈节点检测 (bottleneck_nodes)
   - 前置依赖传播 (Bayesian proficiency propagation)
   - KP 中心性图 (kp_centrality_map)
3. 画像置信度估计 (ProfileConfidenceEstimator)
   - overall_confidence / kp_confidence / data_sufficiency
   - 中位数 ≥ 0.75 验证
4. 动态掌握阈值 (DynamicMasteryThreshold / CC1)
   - beginner: 0.80, intermediate: 0.85, advanced: 0.90, teacher: 0.95
5. 遗忘预警生成 (ForgettingAlertGenerator)
   - days_since_review / decay_factor / urgency
   - 按 Bloom 分级阈值: L1=0.70, L2=0.60, L3=0.50
6. 在线风格适配 (OnlineStyleAdapter)
   - 指数平滑风格分数 (alpha=0.3)
   - 行为驱动的风格在线修正
7. 自适应 Bloom 目标 (AdaptiveBloomSetter)
   - 基于 ZPD + 掌握度轨迹设定目标
   - 不再仅 +1 级, 考虑 ZPD 可达性
8. 画像版本管理 (ProfileVersionManager)
   - 版本号 / 时间戳 / 变更追踪
9. 画像一致性校验 (ProfileConsistencyValidator)
   - 内部一致性检测 (theta vs mastery, level vs mastery)
   - 异常检测 (不可能时间戳, mastery 越界)
10. ProfileOutput 增强
    - 新增 score_overall / kp_centrality / bottleneck_nodes / forgetting_alerts
    - to_api_response / from_irt_output 跨模块互操作
11. ProfileTracingService 增强
    - enable_enhanced 开关
    - 全链路集成多维融合 / KST / 置信度 / 遗忘预警
12. REST API 端点暴露
    - /l2/profile/{id} / /l2/profile/{id}/weak-points / /l2/profile/{id}/confidence
"""

from __future__ import annotations

import math
import time
from typing import Any

import pytest

from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.models import (
    AnswerRecord,
    IRTState,
    LearnerSnapshot,
    TracingState,
)
from dy3_polaris.l2.profile_builder.tracing_service import (
    ProfileOutput,
    ProfileTracingService,
)


# ============================================================
# 测试辅助工厂函数
# ============================================================


def make_tracing_state(
    kp_id: str = "kp1",
    mastery: float = 0.5,
    attempts: int = 1,
    correct: int = 0,
    last_time: float | None = None,
) -> TracingState:
    """创建 TracingState 测试实例."""
    return TracingState(
        kp_id=kp_id,
        mastery_prob=mastery,
        attempts=attempts,
        correct_count=correct,
        last_attempt_time=last_time if last_time is not None else time.time(),
    )


def make_answer_event(
    learner_id: str = "learner1",
    kp_id: str = "kp1",
    correct: bool = True,
    difficulty: float = 0.5,
    timestamp: float | None = None,
) -> AnswerEvent:
    """创建 AnswerEvent 测试实例."""
    return AnswerEvent(
        learner_id=learner_id,
        kp_id=kp_id,
        correct=correct,
        difficulty=difficulty,
        timestamp=timestamp if timestamp is not None else time.time(),
        question_id=f"q_{kp_id}",
    )


def make_kg_structure() -> dict[str, Any]:
    """创建测试用知识图谱结构 (含前置依赖)."""
    return {
        "nodes": [
            {"kp_id": "A-01", "name": "能级跃迁基础"},
            {"kp_id": "A-02", "name": "能级图绘制"},
            {"kp_id": "A-03", "name": "发射光谱分析"},
            {"kp_id": "A-04", "name": "激发态寿命"},
            {"kp_id": "A-05", "name": "Stokes位移"},
        ],
        "edges": [
            {"from": "A-01", "to": "A-02", "type": "prerequisite"},
            {"from": "A-02", "to": "A-03", "type": "prerequisite"},
            {"from": "A-01", "to": "A-04", "type": "prerequisite"},
            {"from": "A-03", "to": "A-05", "type": "prerequisite"},
        ],
    }


# ============================================================
# 1. 多维画像融合 (MultiDimensionalProfileFuser)
# ============================================================


class TestMultiDimensionalFuser:
    """多维画像融合器测试 — 五维加权融合."""

    def test_fusion_score_formula(self):
        """Score_overall = 0.60×μ(M) + 0.20×f(B) + 0.10×g(behavior) + 0.10×θ_norm."""
        from dy3_polaris.l2.profile_builder.enhanced import MultiDimensionalFuser

        fuser = MultiDimensionalFuser()
        kp_mastery = {"A-01": 0.8, "A-02": 0.6, "A-03": 0.7}
        subject_background = {"physics": 0.8, "chemistry": 0.6, "materials": 0.7, "engineering": 0.5}
        behavior_features = {"avg_session_duration": 30.0, "streak_days": 7, "accuracy_trend": [0.6, 0.7, 0.8]}
        theta = 1.0

        score = fuser.fuse(
            kp_mastery=kp_mastery,
            subject_background=subject_background,
            behavior_features=behavior_features,
            theta=theta,
        )

        # μ(M) = (0.8 + 0.6 + 0.7) / 3 = 0.7
        # f(B) = 0.8*0.3 + 0.6*0.3 + 0.7*0.25 + 0.5*0.15 = 0.24 + 0.18 + 0.175 + 0.075 = 0.67
        # g(behavior) ≈ 0.8 (good session/streak/accuracy)
        # θ_norm = sigmoid(1.0) = 1/(1+e^-1) ≈ 0.731
        # Score = 0.6*0.7 + 0.2*0.67 + 0.1*0.8 + 0.1*0.731 ≈ 0.42 + 0.134 + 0.08 + 0.073 ≈ 0.707
        assert 0.6 < score < 0.8

    def test_theta_normalization_sigmoid(self):
        """θ 归一化使用 sigmoid: θ_norm = 1/(1+e^(-θ))."""
        from dy3_polaris.l2.profile_builder.enhanced import MultiDimensionalFuser

        fuser = MultiDimensionalFuser()
        # θ=0 → sigmoid=0.5
        assert fuser.normalize_theta(0.0) == pytest.approx(0.5, abs=0.01)
        # θ=3 → sigmoid≈0.953
        assert fuser.normalize_theta(3.0) == pytest.approx(0.953, abs=0.01)
        # θ=-3 → sigmoid≈0.047
        assert fuser.normalize_theta(-3.0) == pytest.approx(0.047, abs=0.01)

    def test_subject_background_weighted_mean(self):
        """f(B) = weighted_mean(B, [0.3, 0.3, 0.25, 0.15]) — 物理/化学/材料/工程."""
        from dy3_polaris.l2.profile_builder.enhanced import MultiDimensionalFuser

        fuser = MultiDimensionalFuser()
        bg = {"physics": 0.9, "chemistry": 0.5, "materials": 0.7, "engineering": 0.3}
        score = fuser.score_subject_background(bg)
        # 0.9*0.3 + 0.5*0.3 + 0.7*0.25 + 0.3*0.15 = 0.27 + 0.15 + 0.175 + 0.045 = 0.64
        assert score == pytest.approx(0.64, abs=0.01)

    def test_behavior_scoring(self):
        """g(behavior) 综合活跃度/连续天数/正确率趋势."""
        from dy3_polaris.l2.profile_builder.enhanced import MultiDimensionalFuser

        fuser = MultiDimensionalFuser()
        # 优秀行为: 长会话 + 高连续 + 上升趋势
        good = {"avg_session_duration": 45.0, "streak_days": 14, "accuracy_trend": [0.5, 0.7, 0.9]}
        good_score = fuser.score_behavior(good)
        assert good_score > 0.7

        # 差行为: 短会话 + 低连续 + 下降趋势
        bad = {"avg_session_duration": 3.0, "streak_days": 1, "accuracy_trend": [0.8, 0.6, 0.4]}
        bad_score = fuser.score_behavior(bad)
        assert bad_score < 0.4

    def test_empty_inputs_safe(self):
        """空输入安全降级 — 不抛异常, 返回合理默认值."""
        from dy3_polaris.l2.profile_builder.enhanced import MultiDimensionalFuser

        fuser = MultiDimensionalFuser()
        score = fuser.fuse(
            kp_mastery={},
            subject_background={},
            behavior_features={},
            theta=0.0,
        )
        assert 0.0 <= score <= 1.0


# ============================================================
# 2. KST 知识空间理论集成 (KSTAnalyzer)
# ============================================================


class TestKSTAnalyzer:
    """KST 知识空间理论分析器测试."""

    def test_inner_fringe(self):
        """内边界 = 最近学到的概念 (可遗忘的最深层)."""
        from dy3_polaris.l2.profile_builder.enhanced import KSTAnalyzer

        analyzer = KSTAnalyzer()
        kg = make_kg_structure()
        mastered_kps = {"A-01", "A-02"}
        inner = analyzer.compute_inner_fringe(mastered_kps, kg)
        # A-02 的前置 A-01 已掌握, A-02 是内边界
        assert "A-02" in inner

    def test_outer_fringe(self):
        """外边界 = 下一个可学概念 (前置全满足)."""
        from dy3_polaris.l2.profile_builder.enhanced import KSTAnalyzer

        analyzer = KSTAnalyzer()
        kg = make_kg_structure()
        mastered_kps = {"A-01", "A-02"}
        outer = analyzer.compute_outer_fringe(mastered_kps, kg)
        # A-03 的前置 A-02 已掌握 → 外边界
        assert "A-03" in outer
        # A-04 的前置 A-01 已掌握 → 外边界
        assert "A-04" in outer
        # A-05 的前置 A-03 未掌握 → 不在外边界
        assert "A-05" not in outer

    def test_bottleneck_detection(self):
        """瓶颈节点 = 低掌握度 + 高依赖权重 (阻塞多个后继)."""
        from dy3_polaris.l2.profile_builder.enhanced import KSTAnalyzer

        analyzer = KSTAnalyzer()
        kg = make_kg_structure()
        kp_mastery = {"A-01": 0.9, "A-02": 0.3, "A-03": 0.8, "A-04": 0.7, "A-05": 0.6}

        bottlenecks = analyzer.detect_bottlenecks(kp_mastery, kg)
        # A-02 掌握度低(0.3) 且阻塞 A-03 和 A-05
        assert len(bottlenecks) > 0
        bn_ids = [b["kp_id"] for b in bottlenecks]
        assert "A-02" in bn_ids

        # 瓶颈节点应包含 blocked_kps 和 dependency_weight
        a02_bn = next(b for b in bottlenecks if b["kp_id"] == "A-02")
        assert "blocked_kps" in a02_bn
        assert "dependency_weight" in a02_bn
        assert len(a02_bn["blocked_kps"]) > 0

    def test_kp_centrality_map(self):
        """KP 中心性 = 基于图结构的重要性分数."""
        from dy3_polaris.l2.profile_builder.enhanced import KSTAnalyzer

        analyzer = KSTAnalyzer()
        kg = make_kg_structure()
        centrality = analyzer.compute_centrality(kg)

        # 所有节点都应有中心性分数
        assert len(centrality) == 5
        # A-01 是根节点 (多条路径经过), 中心性应较高
        assert centrality["A-01"] > centrality["A-05"]
        # 中心性值在 [0, 1] 范围内
        for v in centrality.values():
            assert 0.0 <= v <= 1.0

    def test_bayesian_propagation(self):
        """贝叶斯熟练度传播 — 掌握后继推断前置也掌握 (Knewton 式)."""
        from dy3_polaris.l2.profile_builder.enhanced import KSTAnalyzer

        analyzer = KSTAnalyzer()
        kg = make_kg_structure()
        # 直接观测: A-03 掌握度高, 但 A-02 无直接观测
        observed = {"A-01": 0.9, "A-03": 0.85}
        propagated = analyzer.propagate_proficiency(observed, kg)

        # A-03 掌握 → 前置 A-02 应被传播提高
        assert propagated["A-02"] > 0.5
        # A-01 有直接观测, 传播后应保持或提高
        assert propagated["A-01"] >= 0.9


# ============================================================
# 3. 画像置信度估计 (ProfileConfidenceEstimator)
# ============================================================


class TestProfileConfidenceEstimator:
    """画像置信度估计器测试."""

    def test_overall_confidence_formula(self):
        """overall_confidence = se_factor × data_factor × (1 - drift_penalty)."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConfidenceEstimator

        estimator = ProfileConfidenceEstimator()
        confidence = estimator.estimate(
            record_count=20,
            se=0.2,
            has_drift=False,
            kp_count=10,
        )
        # 20 条记录 + 低 SE + 无漂移 → 高置信度
        assert confidence > 0.7

    def test_low_confidence_cold_start(self):
        """冷启动 (少记录) → 低置信度."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConfidenceEstimator

        estimator = ProfileConfidenceEstimator()
        confidence = estimator.estimate(
            record_count=2,
            se=0.5,
            has_drift=False,
            kp_count=10,
        )
        assert confidence < 0.3

    def test_drift_reduces_confidence(self):
        """漂移检测降低置信度."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConfidenceEstimator

        estimator = ProfileConfidenceEstimator()
        no_drift = estimator.estimate(20, 0.2, False, 10)
        with_drift = estimator.estimate(20, 0.2, True, 10)
        assert with_drift < no_drift

    def test_kp_level_confidence(self):
        """单 KP 置信度 — 基于 attempts 和 last_attempt_time."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConfidenceEstimator

        estimator = ProfileConfidenceEstimator()
        state = make_tracing_state("kp1", mastery=0.7, attempts=10, correct=8)
        kp_conf = estimator.estimate_kp_confidence(state)
        # 10 次练习 → 高置信度
        assert kp_conf > 0.7

    def test_data_sufficiency(self):
        """数据充分度 — 基于记录数与 KP 数比值."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConfidenceEstimator

        estimator = ProfileConfidenceEstimator()
        # 42 个 KP, 100 条记录 → 充分度 ~2.4 per KP
        sufficiency = estimator.estimate_data_sufficiency(record_count=100, kp_count=42)
        assert 0.5 < sufficiency < 1.0

    def test_median_confidence_above_threshold(self):
        """中位数 ≥ 0.75 (设计文档要求)."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConfidenceEstimator

        estimator = ProfileConfidenceEstimator()
        # 模拟稳态学习者
        confidences = [
            estimator.estimate(50, 0.15, False, 10),
            estimator.estimate(30, 0.2, False, 10),
            estimator.estimate(100, 0.1, False, 10),
        ]
        median = sorted(confidences)[len(confidences) // 2]
        assert median >= 0.75


# ============================================================
# 4. 动态掌握阈值 (DynamicMasteryThreshold / CC1)
# ============================================================


class TestDynamicMasteryThreshold:
    """动态掌握阈值测试 — CC1 按能力等级调整."""

    def test_beginner_threshold(self):
        """beginner → 0.80."""
        from dy3_polaris.l2.profile_builder.enhanced import DynamicMasteryThreshold

        thresholder = DynamicMasteryThreshold()
        assert thresholder.get_threshold("beginner") == 0.80

    def test_intermediate_threshold(self):
        """intermediate → 0.85."""
        from dy3_polaris.l2.profile_builder.enhanced import DynamicMasteryThreshold

        thresholder = DynamicMasteryThreshold()
        assert thresholder.get_threshold("intermediate") == 0.85

    def test_advanced_threshold(self):
        """advanced → 0.90."""
        from dy3_polaris.l2.profile_builder.enhanced import DynamicMasteryThreshold

        thresholder = DynamicMasteryThreshold()
        assert thresholder.get_threshold("advanced") == 0.90

    def test_teacher_threshold(self):
        """teacher → 0.95."""
        from dy3_polaris.l2.profile_builder.enhanced import DynamicMasteryThreshold

        thresholder = DynamicMasteryThreshold()
        assert thresholder.get_threshold("teacher") == 0.95

    def test_is_mastered_uses_dynamic_threshold(self):
        """掌握判定使用动态阈值而非固定 0.85."""
        from dy3_polaris.l2.profile_builder.enhanced import DynamicMasteryThreshold

        thresholder = DynamicMasteryThreshold()
        # mastery=0.82: beginner 阈值 0.80 → 已掌握; intermediate 阈值 0.85 → 未掌握
        assert thresholder.is_mastered(0.82, "beginner") is True
        assert thresholder.is_mastered(0.82, "intermediate") is False


# ============================================================
# 5. 遗忘预警生成 (ForgettingAlertGenerator)
# ============================================================


class TestForgettingAlertGenerator:
    """遗忘预警生成器测试."""

    def test_generate_alert_for_stale_kp(self):
        """长时间未复习的 KP → 生成遗忘预警."""
        from dy3_polaris.l2.profile_builder.enhanced import ForgettingAlertGenerator

        generator = ForgettingAlertGenerator()
        old_time = time.time() - 10 * 86400  # 10 天前
        state = make_tracing_state("A-01", mastery=0.65, attempts=5, correct=3, last_time=old_time)

        alerts = generator.generate_alerts(
            tracing_states={"A-01": state},
            bloom_level="apply",  # L2 → 阈值 0.60
        )
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["kp_id"] == "A-01"
        assert alert["days_since_review"] >= 9
        assert "decay_factor" in alert
        assert alert["urgency"] in ("low", "medium", "high")

    def test_no_alert_for_recent_kp(self):
        """刚复习的 KP → 无预警."""
        from dy3_polaris.l2.profile_builder.enhanced import ForgettingAlertGenerator

        generator = ForgettingAlertGenerator()
        state = make_tracing_state("A-01", mastery=0.9, attempts=10, correct=9, last_time=time.time())

        alerts = generator.generate_alerts(
            tracing_states={"A-01": state},
            bloom_level="remember",
        )
        assert len(alerts) == 0

    def test_bloom_level_thresholds(self):
        """Bloom 分级阈值: L1=0.70, L2=0.60, L3=0.50."""
        from dy3_polaris.l2.profile_builder.enhanced import ForgettingAlertGenerator

        generator = ForgettingAlertGenerator()
        old_time = time.time() - 5 * 86400

        # mastery=0.65: L1阈值0.70 → 预警; L2阈值0.60 → 无预警
        state = make_tracing_state("kp1", mastery=0.65, last_time=old_time)

        l1_alerts = generator.generate_alerts({"kp1": state}, "remember")
        l2_alerts = generator.generate_alerts({"kp1": state}, "apply")

        assert len(l1_alerts) == 1  # 0.65 < 0.70 → 预警
        assert len(l2_alerts) == 0  # 0.65 >= 0.60 → 无预警

    def test_urgency_levels(self):
        """紧急度分级: high (mastery < 0.3), medium (0.3-0.5), low (0.5-阈值)."""
        from dy3_polaris.l2.profile_builder.enhanced import ForgettingAlertGenerator

        generator = ForgettingAlertGenerator()
        old_time = time.time() - 15 * 86400

        high_alert = generator.generate_alerts(
            {"kp1": make_tracing_state("kp1", mastery=0.2, last_time=old_time)},
            "remember",
        )
        medium_alert = generator.generate_alerts(
            {"kp2": make_tracing_state("kp2", mastery=0.4, last_time=old_time)},
            "remember",
        )

        assert high_alert[0]["urgency"] == "high"
        assert medium_alert[0]["urgency"] == "medium"


# ============================================================
# 6. 在线风格适配 (OnlineStyleAdapter)
# ============================================================


class TestOnlineStyleAdapter:
    """在线风格适配器测试 — 指数平滑."""

    def test_exponential_smoothing_update(self):
        """指数平滑: new_score = α×observed + (1-α)×old_score, α=0.3."""
        from dy3_polaris.l2.profile_builder.enhanced import OnlineStyleAdapter

        adapter = OnlineStyleAdapter(alpha=0.3)
        # 初始: visual=0.5, aural=0.3, reading=0.1, kinesthetic=0.1
        adapter.initialize({"V": 0.5, "A": 0.3, "R": 0.1, "K": 0.1})

        # 观测到 visual 行为
        adapter.update("visual")
        # new_V = 0.3*1.0 + 0.7*0.5 = 0.65
        scores = adapter.get_scores()
        assert scores["V"] == pytest.approx(0.65, abs=0.01)

    def test_style_changes_with_behavior(self):
        """持续 visual 行为 → 风格趋向 visual."""
        from dy3_polaris.l2.profile_builder.enhanced import OnlineStyleAdapter

        adapter = OnlineStyleAdapter(alpha=0.3)
        adapter.initialize({"V": 0.25, "A": 0.25, "R": 0.25, "K": 0.25})

        for _ in range(10):
            adapter.update("visual")

        style = adapter.infer_style()
        assert style == "visual"

    def test_multimodal_preserved(self):
        """多维接近时保持 multimodal."""
        from dy3_polaris.l2.profile_builder.enhanced import OnlineStyleAdapter

        adapter = OnlineStyleAdapter(alpha=0.3)
        adapter.initialize({"V": 0.30, "A": 0.28, "R": 0.22, "K": 0.20})

        style = adapter.infer_style()
        assert style == "multimodal"

    def test_no_initialization_defaults(self):
        """未初始化 → 均匀分布."""
        from dy3_polaris.l2.profile_builder.enhanced import OnlineStyleAdapter

        adapter = OnlineStyleAdapter()
        scores = adapter.get_scores()
        assert all(abs(v - 0.25) < 0.01 for v in scores.values())


# ============================================================
# 7. 自适应 Bloom 目标 (AdaptiveBloomSetter)
# ============================================================


class TestAdaptiveBloomSetter:
    """自适应 Bloom 目标设定器测试."""

    def test_zpd_aware_target(self):
        """基于 ZPD 设定目标 — 掌握度高 → 跳级; 低 → 循序渐进."""
        from dy3_polaris.l2.profile_builder.enhanced import AdaptiveBloomSetter

        setter = AdaptiveBloomSetter()

        # 高掌握度 + ZPD 支持 → 跳级 (2 级以上)
        target = setter.set_adaptive_target(
            current_level="understand",
            avg_mastery=0.9,
            zpd_zone="independent",
        )
        assert target in ("evaluate", "create")  # 跳 2-3 级

    def test_conservative_target_low_mastery(self):
        """低掌握度 → 仅 +1 级 (保守)."""
        from dy3_polaris.l2.profile_builder.enhanced import AdaptiveBloomSetter

        setter = AdaptiveBloomSetter()

        target = setter.set_adaptive_target(
            current_level="remember",
            avg_mastery=0.3,
            zpd_zone="frustration",
        )
        assert target == "understand"  # 仅 +1

    def test_zpd_frustration_stays_current(self):
        """挫败区 → 保持当前级别 (不冒进)."""
        from dy3_polaris.l2.profile_builder.enhanced import AdaptiveBloomSetter

        setter = AdaptiveBloomSetter()
        target = setter.set_adaptive_target(
            current_level="apply",
            avg_mastery=0.4,
            zpd_zone="frustration",
        )
        assert target == "apply"  # 保持

    def test_max_level_capped(self):
        """最高级 create → 保持."""
        from dy3_polaris.l2.profile_builder.enhanced import AdaptiveBloomSetter

        setter = AdaptiveBloomSetter()
        target = setter.set_adaptive_target(
            current_level="create",
            avg_mastery=0.95,
            zpd_zone="independent",
        )
        assert target == "create"


# ============================================================
# 8. 画像版本管理 (ProfileVersionManager)
# ============================================================


class TestProfileVersionManager:
    """画像版本管理器测试."""

    def test_version_increment(self):
        """每次保存画像 → 版本号递增."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileVersionManager

        manager = ProfileVersionManager()
        snap1 = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            theta=0.5,
            level="intermediate",
        )
        v1 = manager.save("l1", snap1)
        assert v1 == 1

        snap2 = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            theta=0.6,
            level="advanced",
        )
        v2 = manager.save("l1", snap2)
        assert v2 == 2

    def test_history_retrieval(self):
        """获取历史画像列表."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileVersionManager

        manager = ProfileVersionManager()
        for theta in [0.1, 0.2, 0.3]:
            snap = LearnerSnapshot(
                learner_id="l1",
                snapshot_ts=time.time(),
                theta=theta,
            )
            manager.save("l1", snap)

        history = manager.get_history("l1")
        assert len(history) == 3
        # 按版本号排序
        assert history[0]["version"] == 1
        assert history[-1]["version"] == 3

    def test_diff_between_versions(self):
        """版本差异比较 — 检测 theta/level 变化."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileVersionManager

        manager = ProfileVersionManager()
        snap1 = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            theta=0.3,
            level="beginner",
            learning_style="reading",
        )
        manager.save("l1", snap1)

        snap2 = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            theta=0.6,
            level="intermediate",
            learning_style="reading",
        )
        manager.save("l1", snap2)

        diff = manager.diff("l1", 1, 2)
        assert diff["theta_changed"] is True
        assert diff["level_changed"] is True
        assert diff["style_changed"] is False


# ============================================================
# 9. 画像一致性校验 (ProfileConsistencyValidator)
# ============================================================


class TestProfileConsistencyValidator:
    """画像一致性校验器测试."""

    def test_valid_profile_passes(self):
        """一致画像 → 无异常."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConsistencyValidator

        validator = ProfileConsistencyValidator()
        snap = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            kp_mastery={"A-01": 0.8, "A-02": 0.6},
            theta=0.5,
            level="intermediate",
            weak_kps=["A-02"],
            confidence=0.8,
        )
        issues = validator.validate(snap)
        assert len(issues) == 0

    def test_theta_mastery_inconsistency(self):
        """theta 高但 mastery 全低 → 不一致."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConsistencyValidator

        validator = ProfileConsistencyValidator()
        snap = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            kp_mastery={"A-01": 0.2, "A-02": 0.1},
            theta=2.0,  # 高能力
            level="advanced",  # 高等级
            confidence=0.9,
        )
        issues = validator.validate(snap)
        assert len(issues) > 0
        assert any("mastery" in i.lower() or "theta" in i.lower() for i in issues)

    def test_mastery_out_of_range(self):
        """mastery 越界 [0, 1] → 异常."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConsistencyValidator

        validator = ProfileConsistencyValidator()
        snap = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            kp_mastery={"A-01": 1.5},  # 越界
            theta=0.0,
        )
        issues = validator.validate(snap)
        assert len(issues) > 0
        assert any("range" in i.lower() or "越界" in i for i in issues)

    def test_weak_kps_consistency(self):
        """weak_kps 应与 mastery < 0.5 的 KP 一致."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConsistencyValidator

        validator = ProfileConsistencyValidator()
        snap = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            kp_mastery={"A-01": 0.8, "A-02": 0.3, "A-03": 0.4},
            theta=0.0,
            weak_kps=["A-01"],  # 错误: A-01 mastery=0.8 不应出现在 weak_kps
        )
        issues = validator.validate(snap)
        assert any("weak" in i.lower() for i in issues)

    def test_confidence_range(self):
        """confidence 应在 [0, 1]."""
        from dy3_polaris.l2.profile_builder.enhanced import ProfileConsistencyValidator

        validator = ProfileConsistencyValidator()
        snap = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=time.time(),
            confidence=1.5,  # 越界
        )
        issues = validator.validate(snap)
        assert any("confidence" in i.lower() for i in issues)


# ============================================================
# 10. ProfileOutput 增强
# ============================================================


class TestProfileOutputEnhanced:
    """ProfileOutput 增强字段测试."""

    def test_new_fields_exist(self):
        """增强字段: score_overall / bottleneck_nodes / forgetting_alerts / kp_centrality."""
        output = ProfileOutput(
            learner_id="l1",
            phase="stable",
            theta=0.5,
            level="intermediate",
            learning_style="visual",
            bloom_target="apply",
        )
        # 增强字段应有默认值
        assert hasattr(output, "score_overall")
        assert hasattr(output, "bottleneck_nodes")
        assert hasattr(output, "forgetting_alerts")
        assert hasattr(output, "kp_centrality")
        assert hasattr(output, "mastery_threshold")
        assert hasattr(output, "profile_version")

    def test_to_api_response_includes_enhanced_fields(self):
        """to_api_response 包含增强字段."""
        output = ProfileOutput(
            learner_id="l1",
            phase="stable",
            theta=0.5,
            level="intermediate",
            learning_style="visual",
            bloom_target="apply",
            score_overall=0.75,
            mastery_threshold=0.85,
        )
        api = output.to_api_response()
        assert "score_overall" in api
        assert "mastery_threshold" in api
        assert api["score_overall"] == pytest.approx(0.75, abs=0.01)

    def test_from_irt_output_interop(self):
        """从 IRT AbilityOutput 构建 ProfileOutput."""
        from dy3_polaris.l2.profile_builder.tracing_service import ProfileOutput

        class MockIRTOutput:
            learner_id = "l1"
            theta = 0.8
            se = 0.2
            level = "advanced"

        output = ProfileOutput.from_irt_output(MockIRTOutput())
        assert output.theta == pytest.approx(0.8, abs=0.01)
        assert output.level == "advanced"


# ============================================================
# 11. ProfileTracingService 增强
# ============================================================


class TestProfileTracingServiceEnhanced:
    """ProfileTracingService 增强集成测试."""

    @pytest.fixture
    def enhanced_service(self):
        """启用增强功能的 ProfileTracingService."""
        return ProfileTracingService(enable_enhanced=True)

    def test_enable_enhanced_flag(self, enhanced_service):
        """enable_enhanced=True → 增强功能激活."""
        assert enhanced_service.enable_enhanced is True

    def test_enhanced_output_contains_score(self, enhanced_service):
        """增强模式 → ProfileOutput 包含 score_overall."""
        event = make_answer_event(learner_id="enh1", correct=True)
        output = enhanced_service.process(event)
        assert hasattr(output, "score_overall")
        assert output.score_overall > 0.0

    def test_enhanced_output_contains_mastery_threshold(self, enhanced_service):
        """增强模式 → 包含动态掌握阈值."""
        event = make_answer_event(learner_id="enh2", correct=True)
        output = enhanced_service.process(event)
        assert output.mastery_threshold is not None
        assert output.mastery_threshold in (0.80, 0.85, 0.90, 0.95)

    def test_enhanced_forgetting_alerts(self, enhanced_service):
        """增强模式 → 生成遗忘预警 (无历史 → 空列表)."""
        event = make_answer_event(learner_id="enh3", correct=True)
        output = enhanced_service.process(event)
        assert hasattr(output, "forgetting_alerts")
        assert isinstance(output.forgetting_alerts, list)

    def test_backward_compatibility(self):
        """enable_enhanced=False → 完全向后兼容."""
        service = ProfileTracingService(enable_enhanced=False)
        event = make_answer_event(learner_id="compat1", correct=True)
        output = service.process(event)
        # 基本字段存在
        assert output.learner_id == "compat1"
        assert output.theta is not None
        assert output.level is not None

    def test_get_weak_points_analysis(self, enhanced_service):
        """get_weak_points 返回薄弱 KP + 瓶颈分析."""
        # 先处理几个事件建立画像
        for i in range(5):
            enhanced_service.process(
                make_answer_event(learner_id="wp1", kp_id=f"kp{i}", correct=False)
            )
        result = enhanced_service.get_weak_points("wp1")
        assert "weak_kps" in result
        assert "bottleneck_nodes" in result
        assert "kp_centrality_map" in result

    def test_get_confidence_report(self, enhanced_service):
        """get_confidence 返回置信度报告."""
        for i in range(10):
            enhanced_service.process(
                make_answer_event(learner_id="conf1", correct=True)
            )
        result = enhanced_service.get_confidence("conf1")
        assert "overall_confidence" in result
        assert "kp_confidence" in result
        assert "data_sufficiency" in result

    def test_get_skillbook_enhanced(self, enhanced_service):
        """get_skillbook 返回增强技能树."""
        for i in range(3):
            enhanced_service.process(
                make_answer_event(learner_id="sb1", kp_id=f"kp{i}", correct=True)
            )
        result = enhanced_service.get_skillbook("sb1")
        assert "global_ability" in result
        assert "nodes" in result
        assert "edges" in result


# ============================================================
# 12. REST API 端点暴露
# ============================================================


class TestProfileAPIEndpoints:
    """画像 REST API 端点测试."""

    @pytest.fixture
    def service(self):
        """初始化带数据的增强服务."""
        svc = ProfileTracingService(enable_enhanced=True)
        for i in range(5):
            svc.process(
                make_answer_event(learner_id="api1", kp_id=f"kp{i}", correct=True)
            )
        return svc

    def test_to_api_response_serializable(self, service):
        """ProfileOutput.to_api_response 可 JSON 序列化."""
        import json

        event = make_answer_event(learner_id="api1", correct=True)
        output = service.process(event)
        api = output.to_api_response()
        # 确保可序列化
        json_str = json.dumps(api, default=str)
        assert "learner_id" in json_str

    def test_weak_points_api_format(self, service):
        """weak-points 端点格式: {weak_kps, bottleneck_nodes, kp_centrality_map}."""
        result = service.get_weak_points("api1")
        assert isinstance(result["weak_kps"], list)
        assert isinstance(result["bottleneck_nodes"], list)
        assert isinstance(result["kp_centrality_map"], dict)

    def test_confidence_api_format(self, service):
        """confidence 端点格式: {overall_confidence, kp_confidence, data_sufficiency}."""
        result = service.get_confidence("api1")
        assert 0.0 <= result["overall_confidence"] <= 1.0
        assert isinstance(result["kp_confidence"], dict)
        assert 0.0 <= result["data_sufficiency"] <= 1.0

    def test_profile_api_format(self, service):
        """profile 端点格式: 完整 ProfileOutput + 增强字段."""
        snap = service.get_profile_snapshot("api1")
        assert snap is not None
        assert snap.learner_id == "api1"

    def test_skillbook_api_format(self, service):
        """skillbook 端点格式: {global_ability, nodes, edges}."""
        result = service.get_skillbook("api1")
        assert 0.0 <= result["global_ability"] <= 1.0
        assert len(result["nodes"]) > 0
