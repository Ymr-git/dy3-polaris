"""L2 profile_builder 子模块测试 — LevelEstimator / StyleInferrer / BloomSetter / ProfileBuilder.

测试覆盖 (TDD):
1. LevelEstimator:
   - estimate(theta, avg_mastery): BKT + IRT 融合能力分级
   - beginner: theta < -0.5 或 avg_mastery < 0.4
   - intermediate: -0.5 <= theta < 0.5 且 0.4 <= avg_mastery < 0.7
   - advanced: theta >= 0.5 且 avg_mastery >= 0.7
   - 边界值测试
2. StyleInferrer:
   - infer_from_vark(vark_profile): 从 VARK 四维分数推断主要风格
   - infer_from_behavior(events): 从行为事件推断风格
   - 无数据时默认 "reading"
   - 多维接近 (差 < 0.05) 返回 "multimodal"
3. BloomSetter:
   - set_target(current_level, goal_level): 设定 Bloom 目标
   - 默认目标比当前高一级
   - 支持 6 级: remember/understand/apply/analyze/evaluate/create
   - 已到最高级则保持
4. ProfileBuilder:
   - __init__(store=None): 依赖注入
   - build(...): 组装 BKT/IRT/VARK/Bloom 综合画像
   - 返回 LearnerSnapshot (learner_id, snapshot_ts, kp_mastery, theta, level, ...)
"""

import time

import pytest

from dy3_polaris.l2.knowledge_tracer.forgetting import ForgettingModel
from dy3_polaris.l2.models import (
    AnswerRecord,
    IRTState,
    LearnerSnapshot,
    TracingState,
)
from dy3_polaris.l2.profile_builder import (
    BloomSetter,
    LevelEstimator,
    ProfileBuilder,
    StyleInferrer,
)
from dy3_polaris.l2.store import InMemoryL2Store


# ============================================================
# 1. LevelEstimator
# ============================================================


class TestLevelEstimator:
    """LevelEstimator 测试 — BKT + IRT 融合的能力分级."""

    def test_beginner_low_theta(self):
        """theta < -0.5 -> beginner (即使掌握度高)."""
        est = LevelEstimator()
        assert est.estimate(theta=-1.0, avg_mastery=0.8) == "beginner"

    def test_beginner_low_mastery(self):
        """avg_mastery < 0.4 -> beginner (即使 theta 高)."""
        est = LevelEstimator()
        assert est.estimate(theta=0.5, avg_mastery=0.3) == "beginner"

    def test_beginner_both_low(self):
        """theta 低且掌握度低 -> beginner."""
        est = LevelEstimator()
        assert est.estimate(theta=-1.0, avg_mastery=0.2) == "beginner"

    def test_intermediate_mid_range(self):
        """-0.5 <= theta < 0.5 且 0.4 <= avg_mastery < 0.7 -> intermediate."""
        est = LevelEstimator()
        assert est.estimate(theta=0.0, avg_mastery=0.5) == "intermediate"

    def test_intermediate_low_boundary_theta(self):
        """theta = -0.5 (边界) 且掌握度中等 -> intermediate."""
        est = LevelEstimator()
        assert est.estimate(theta=-0.5, avg_mastery=0.5) == "intermediate"

    def test_intermediate_low_boundary_mastery(self):
        """avg_mastery = 0.4 (边界) 且 theta 中等 -> intermediate."""
        est = LevelEstimator()
        assert est.estimate(theta=0.0, avg_mastery=0.4) == "intermediate"

    def test_intermediate_upper_boundary_theta(self):
        """theta 接近 0.5 (但 < 0.5) 且掌握度中等 -> intermediate."""
        est = LevelEstimator()
        assert est.estimate(theta=0.49, avg_mastery=0.5) == "intermediate"

    def test_intermediate_upper_boundary_mastery(self):
        """avg_mastery 接近 0.7 (但 < 0.7) 且 theta 中等 -> intermediate."""
        est = LevelEstimator()
        assert est.estimate(theta=0.0, avg_mastery=0.69) == "intermediate"

    def test_advanced_high_both(self):
        """theta >= 0.5 且 avg_mastery >= 0.7 -> advanced."""
        est = LevelEstimator()
        assert est.estimate(theta=1.0, avg_mastery=0.8) == "advanced"

    def test_advanced_boundary(self):
        """theta = 0.5 且 avg_mastery = 0.7 (边界) -> advanced."""
        est = LevelEstimator()
        assert est.estimate(theta=0.5, avg_mastery=0.7) == "advanced"

    def test_advanced_high_theta_high_mastery(self):
        """theta 和 mastery 都很高 -> advanced."""
        est = LevelEstimator()
        assert est.estimate(theta=2.0, avg_mastery=0.95) == "advanced"

    # --- 边界值 / 间隙区域 ---

    def test_boundary_theta_neg_0_5_mastery_0_4(self):
        """边界: theta=-0.5, mastery=0.4 -> intermediate (恰不满足 beginner)."""
        est = LevelEstimator()
        assert est.estimate(theta=-0.5, avg_mastery=0.4) == "intermediate"

    def test_gap_high_theta_mid_mastery(self):
        """间隙区域: theta=0.6 (>= 0.5) 但 mastery=0.5 (< 0.7) -> intermediate."""
        est = LevelEstimator()
        assert est.estimate(theta=0.6, avg_mastery=0.5) == "intermediate"

    def test_gap_mid_theta_high_mastery(self):
        """间隙区域: theta=0.3 (< 0.5) 但 mastery=0.8 (>= 0.7) -> intermediate."""
        est = LevelEstimator()
        assert est.estimate(theta=0.3, avg_mastery=0.8) == "intermediate"

    def test_estimate_returns_valid_level(self):
        """estimate 始终返回 beginner/intermediate/advanced 之一."""
        est = LevelEstimator()
        for theta in (-2.0, -0.5, 0.0, 0.5, 2.0):
            for mastery in (0.0, 0.4, 0.5, 0.7, 1.0):
                level = est.estimate(theta, mastery)
                assert level in ("beginner", "intermediate", "advanced")


# ============================================================
# 2. StyleInferrer - infer_from_vark
# ============================================================


class TestStyleInferrerFromVARK:
    """StyleInferrer.infer_from_vark 测试 — 从 VARK 四维分数推断风格."""

    def test_visual_dominant(self):
        """视觉维度最高 -> "visual"."""
        inf = StyleInferrer()
        profile = {
            "visual_score": 0.8,
            "aural_score": 0.1,
            "read_write_score": 0.05,
            "kinesthetic_score": 0.05,
        }
        assert inf.infer_from_vark(profile) == "visual"

    def test_aural_dominant(self):
        """听觉维度最高 -> "aural"."""
        inf = StyleInferrer()
        profile = {
            "visual_score": 0.1,
            "aural_score": 0.7,
            "read_write_score": 0.1,
            "kinesthetic_score": 0.1,
        }
        assert inf.infer_from_vark(profile) == "aural"

    def test_reading_dominant(self):
        """读写维度最高 -> "reading"."""
        inf = StyleInferrer()
        profile = {
            "visual_score": 0.1,
            "aural_score": 0.1,
            "read_write_score": 0.7,
            "kinesthetic_score": 0.1,
        }
        assert inf.infer_from_vark(profile) == "reading"

    def test_kinesthetic_dominant(self):
        """动觉维度最高 -> "kinesthetic"."""
        inf = StyleInferrer()
        profile = {
            "visual_score": 0.1,
            "aural_score": 0.1,
            "read_write_score": 0.1,
            "kinesthetic_score": 0.7,
        }
        assert inf.infer_from_vark(profile) == "kinesthetic"

    def test_multimodal_close_scores(self):
        """两维接近 (差 < 0.05) -> "multimodal"."""
        inf = StyleInferrer()
        profile = {
            "visual_score": 0.40,
            "aural_score": 0.38,
            "read_write_score": 0.12,
            "kinesthetic_score": 0.10,
        }
        assert inf.infer_from_vark(profile) == "multimodal"

    def test_not_multimodal_at_boundary(self):
        """差值恰好 0.05 (不 < 0.05) -> 不返回 multimodal."""
        inf = StyleInferrer()
        profile = {
            "visual_score": 0.40,
            "aural_score": 0.35,
            "read_write_score": 0.15,
            "kinesthetic_score": 0.10,
        }
        assert inf.infer_from_vark(profile) == "visual"

    def test_multimodal_three_close(self):
        """三维接近 -> "multimodal"."""
        inf = StyleInferrer()
        profile = {
            "visual_score": 0.34,
            "aural_score": 0.33,
            "read_write_score": 0.33,
            "kinesthetic_score": 0.0,
        }
        assert inf.infer_from_vark(profile) == "multimodal"

    def test_empty_dict_returns_reading(self):
        """空字典 -> 默认 "reading"."""
        inf = StyleInferrer()
        assert inf.infer_from_vark({}) == "reading"

    def test_all_zeros_returns_reading(self):
        """全零分数 -> 默认 "reading"."""
        inf = StyleInferrer()
        profile = {
            "visual_score": 0.0,
            "aural_score": 0.0,
            "read_write_score": 0.0,
            "kinesthetic_score": 0.0,
        }
        assert inf.infer_from_vark(profile) == "reading"

    def test_partial_keys(self):
        """缺失的维度默认 0.0."""
        inf = StyleInferrer()
        profile = {"visual_score": 0.9}
        result = inf.infer_from_vark(profile)
        assert result == "visual"

    def test_supports_short_keys(self):
        """支持短键名 (visual/aural/read_write/kinesthetic)."""
        inf = StyleInferrer()
        profile = {
            "visual": 0.8,
            "aural": 0.1,
            "read_write": 0.05,
            "kinesthetic": 0.05,
        }
        assert inf.infer_from_vark(profile) == "visual"


# ============================================================
# 3. StyleInferrer - infer_from_behavior
# ============================================================


class TestStyleInferrerFromBehavior:
    """StyleInferrer.infer_from_behavior 测试 — 从行为事件推断风格."""

    def test_empty_events_returns_reading(self):
        """空事件列表 -> 默认 "reading"."""
        inf = StyleInferrer()
        assert inf.infer_from_behavior([]) == "reading"

    def test_visual_events(self):
        """视觉类事件 -> "visual"."""
        inf = StyleInferrer()
        events = [
            {"modality": "video"},
            {"modality": "image"},
            {"modality": "chart"},
        ]
        assert inf.infer_from_behavior(events) == "visual"

    def test_aural_events(self):
        """听觉类事件 -> "aural"."""
        inf = StyleInferrer()
        events = [
            {"modality": "audio"},
            {"modality": "lecture"},
        ]
        assert inf.infer_from_behavior(events) == "aural"

    def test_reading_events(self):
        """读写类事件 -> "reading"."""
        inf = StyleInferrer()
        events = [
            {"modality": "text"},
            {"modality": "article"},
        ]
        assert inf.infer_from_behavior(events) == "reading"

    def test_kinesthetic_events(self):
        """动觉类事件 -> "kinesthetic"."""
        inf = StyleInferrer()
        events = [
            {"modality": "simulation"},
            {"modality": "practice"},
            {"modality": "quiz"},
        ]
        assert inf.infer_from_behavior(events) == "kinesthetic"

    def test_multimodal_mixed_close(self):
        """两类事件数量接近 -> "multimodal"."""
        inf = StyleInferrer()
        events = [
            {"modality": "video"},
            {"modality": "audio"},
        ]
        assert inf.infer_from_behavior(events) == "multimodal"

    def test_event_type_keyword_matching(self):
        """通过 event_type 关键词匹配模态."""
        inf = StyleInferrer()
        events = [
            {"event_type": "watch_video"},
            {"event_type": "view_diagram"},
        ]
        assert inf.infer_from_behavior(events) == "visual"

    def test_content_type_keyword_matching(self):
        """通过 content_type 关键词匹配模态."""
        inf = StyleInferrer()
        events = [
            {"content_type": "interactive_simulation"},
        ]
        assert inf.infer_from_behavior(events) == "kinesthetic"

    def test_unknown_modality_defaults_to_reading(self):
        """无法识别模态的事件默认归为 "reading"."""
        inf = StyleInferrer()
        events = [
            {"event_type": "unknown_action"},
            {"event_type": "something_else"},
        ]
        assert inf.infer_from_behavior(events) == "reading"

    def test_dominant_style_with_minority(self):
        """多数事件为视觉, 少数为听觉 -> "visual"."""
        inf = StyleInferrer()
        events = [
            {"modality": "video"},
            {"modality": "image"},
            {"modality": "chart"},
            {"modality": "audio"},
        ]
        assert inf.infer_from_behavior(events) == "visual"


# ============================================================
# 4. BloomSetter
# ============================================================


class TestBloomSetter:
    """BloomSetter 测试 — Bloom 认知层次目标设定."""

    def test_default_target_one_level_above(self):
        """默认目标比当前高一级."""
        setter = BloomSetter()
        assert setter.set_target("remember") == "understand"
        assert setter.set_target("understand") == "apply"
        assert setter.set_target("apply") == "analyze"
        assert setter.set_target("analyze") == "evaluate"
        assert setter.set_target("evaluate") == "create"

    def test_highest_level_stays(self):
        """已到最高级 (create) 则保持."""
        setter = BloomSetter()
        assert setter.set_target("create") == "create"

    def test_explicit_goal_level(self):
        """显式指定 goal_level 时直接返回该目标."""
        setter = BloomSetter()
        assert setter.set_target("remember", goal_level="evaluate") == "evaluate"
        assert setter.set_target("apply", goal_level="create") == "create"

    def test_explicit_goal_same_as_current(self):
        """显式 goal_level 等于 current_level 时返回该层级."""
        setter = BloomSetter()
        assert setter.set_target("apply", goal_level="apply") == "apply"

    def test_explicit_goal_lower_than_current(self):
        """显式 goal_level 低于 current_level 时仍返回 goal_level."""
        setter = BloomSetter()
        assert setter.set_target("evaluate", goal_level="understand") == "understand"

    def test_all_six_levels(self):
        """支持 6 级: remember/understand/apply/analyze/evaluate/create."""
        setter = BloomSetter()
        levels = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
        for i, level in enumerate(levels):
            target = setter.set_target(level)
            if i < len(levels) - 1:
                assert target == levels[i + 1]
            else:
                assert target == level

    def test_bloom_levels_constant(self):
        """BLOOM_LEVELS 常量有序: remember -> ... -> create."""
        assert BloomSetter.BLOOM_LEVELS == [
            "remember",
            "understand",
            "apply",
            "analyze",
            "evaluate",
            "create",
        ]

    def test_invalid_current_level_raises(self):
        """无效的 current_level 抛出 ValueError."""
        setter = BloomSetter()
        with pytest.raises(ValueError):
            setter.set_target("invalid_level")

    def test_invalid_goal_level_raises(self):
        """无效的 goal_level 抛出 ValueError."""
        setter = BloomSetter()
        with pytest.raises(ValueError):
            setter.set_target("remember", goal_level="invalid")


# ============================================================
# 5. ProfileBuilder
# ============================================================


class TestProfileBuilderInit:
    """ProfileBuilder 初始化测试 — 依赖注入."""

    def test_init_no_store(self):
        """无 store 时正常初始化."""
        builder = ProfileBuilder()
        assert builder is not None

    def test_init_with_store(self):
        """传入 store 时正常初始化."""
        store = InMemoryL2Store()
        builder = ProfileBuilder(store=store)
        assert builder is not None

    def test_init_engines_initialized(self):
        """初始化后内部引擎就绪."""
        builder = ProfileBuilder()
        assert isinstance(builder.level_estimator, LevelEstimator)
        assert isinstance(builder.style_inferrer, StyleInferrer)
        assert isinstance(builder.bloom_setter, BloomSetter)


class TestProfileBuilderBuild:
    """ProfileBuilder.build 测试 — 组装 BKT/IRT/VARK/Bloom 综合画像."""

    def _make_tracing_states(
        self, masteries: dict[str, float]
    ) -> dict[str, TracingState]:
        """构造 TracingState 字典 {kp_id: TracingState}."""
        return {
            kp_id: TracingState(kp_id=kp_id, mastery_prob=m)
            for kp_id, m in masteries.items()
        }

    def test_build_returns_learner_snapshot(self):
        """build 返回 LearnerSnapshot 实例."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.8, "kp-2": 0.6})
        irt = IRTState(theta=0.3, se=0.2)
        snapshot = builder.build("learner-001", tracing, irt)
        assert isinstance(snapshot, LearnerSnapshot)

    def test_build_learner_id(self):
        """build 返回的 snapshot.learner_id 正确."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.5})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("learner-001", tracing, irt)
        assert snapshot.learner_id == "learner-001"

    def test_build_kp_mastery_extracted(self):
        """kp_mastery 从 tracing_states 正确提取."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.8, "kp-2": 0.6})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.kp_mastery == {"kp-1": 0.8, "kp-2": 0.6}

    def test_build_theta_from_irt_state(self):
        """theta 从 irt_state 提取."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.5})
        irt = IRTState(theta=1.2, se=0.15)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.theta == pytest.approx(1.2)

    def test_build_beginner_level(self):
        """低 theta + 低掌握度 -> level = beginner."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.2, "kp-2": 0.3})
        irt = IRTState(theta=-1.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.level == "beginner"

    def test_build_intermediate_level(self):
        """中 theta + 中掌握度 -> level = intermediate."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.5, "kp-2": 0.6})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.level == "intermediate"

    def test_build_advanced_level(self):
        """高 theta + 高掌握度 -> level = advanced."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.8, "kp-2": 0.9})
        irt = IRTState(theta=1.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.level == "advanced"

    def test_build_snapshot_ts_is_float(self):
        """snapshot_ts 为有效时间戳 (float)."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.5})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert isinstance(snapshot.snapshot_ts, float)
        assert snapshot.snapshot_ts > 0

    def test_build_avg_mastery_computed_correctly(self):
        """avg_mastery = mean(kp_mastery) 用于等级判定."""
        builder = ProfileBuilder()
        # avg = (0.2 + 0.4 + 0.6) / 3 = 0.4, theta = 0.0 -> intermediate
        tracing = self._make_tracing_states({"kp-1": 0.2, "kp-2": 0.4, "kp-3": 0.6})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.level == "intermediate"

    def test_build_empty_tracing_states(self):
        """空 tracing_states -> avg_mastery=0.0 -> beginner."""
        builder = ProfileBuilder()
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", {}, irt)
        assert snapshot.kp_mastery == {}
        assert snapshot.level == "beginner"

    def test_build_weak_kps_extracted(self):
        """weak_kps 提取 mastery < 0.5 的知识点."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states(
            {"kp-1": 0.8, "kp-2": 0.3, "kp-3": 0.49, "kp-4": 0.5}
        )
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert "kp-2" in snapshot.weak_kps
        assert "kp-3" in snapshot.weak_kps
        assert "kp-1" not in snapshot.weak_kps
        assert "kp-4" not in snapshot.weak_kps  # 0.5 不 < 0.5

    def test_build_learning_style_default_reading(self):
        """无交互历史时 learning_style 默认 "reading"."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.5})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.learning_style == "reading"

    def test_build_bloom_target_for_beginner(self):
        """beginner -> bloom_target = "understand"."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.2})
        irt = IRTState(theta=-1.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.bloom_target == "understand"

    def test_build_bloom_target_for_intermediate(self):
        """intermediate -> bloom_target = "analyze"."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.5})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.bloom_target == "analyze"

    def test_build_bloom_target_for_advanced(self):
        """advanced -> bloom_target = "evaluate"."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.8})
        irt = IRTState(theta=1.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.bloom_target == "evaluate"

    def test_build_with_interaction_history(self):
        """传入 interaction_history 时不报错."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.5})
        irt = IRTState(theta=0.0)
        history = [
            AnswerRecord("l1", "kp-1", correct=True, timestamp=1.0),
            AnswerRecord("l1", "kp-1", correct=False, timestamp=2.0),
        ]
        snapshot = builder.build("l1", tracing, irt, interaction_history=history)
        assert isinstance(snapshot, LearnerSnapshot)

    def test_build_saves_to_store(self):
        """提供 store 时 build 后自动保存画像."""
        store = InMemoryL2Store()
        builder = ProfileBuilder(store=store)
        tracing = self._make_tracing_states({"kp-1": 0.5})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        loaded = store.get_profile("l1")
        assert loaded is not None
        assert loaded.learner_id == snapshot.learner_id
        assert loaded.level == snapshot.level

    def test_build_no_store_does_not_error(self):
        """无 store 时 build 正常完成 (不保存)."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states({"kp-1": 0.5})
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot is not None

    def test_build_full_assembly(self):
        """完整组装流程验证: 所有字段正确."""
        builder = ProfileBuilder()
        tracing = self._make_tracing_states(
            {"kp-A": 0.9, "kp-B": 0.3, "kp-C": 0.7}
        )
        irt = IRTState(theta=0.8, se=0.1)
        snapshot = builder.build("learner-X", tracing, irt)
        # 核心字段
        assert snapshot.learner_id == "learner-X"
        assert snapshot.kp_mastery == {"kp-A": 0.9, "kp-B": 0.3, "kp-C": 0.7}
        assert snapshot.theta == pytest.approx(0.8)
        # avg_mastery = (0.9 + 0.3 + 0.7) / 3 ≈ 0.633, theta=0.8 >= 0.5 但 mastery < 0.7
        # -> intermediate (间隙区域)
        assert snapshot.level == "intermediate"
        # weak_kps
        assert "kp-B" in snapshot.weak_kps
        assert "kp-A" not in snapshot.weak_kps
        assert "kp-C" not in snapshot.weak_kps
        # learning_style / bloom_target
        assert snapshot.learning_style == "reading"
        assert snapshot.bloom_target == "analyze"


# ============================================================
# 6. ProfileBuilder - 遗忘衰减 (ForgettingModel)
# ============================================================


class TestProfileBuilderForgettingModel:
    """ProfileBuilder 遗忘衰减测试 — forgetting_model 注入 / 衰减门控."""

    def test_init_accepts_forgetting_model(self):
        """__init__ 接受 forgetting_model 参数并存储."""
        fm = ForgettingModel()
        builder = ProfileBuilder(forgetting_model=fm)
        assert builder.forgetting_model is fm

    def test_init_default_creates_forgetting_model(self):
        """未传 forgetting_model 时内部创建默认 ForgettingModel."""
        builder = ProfileBuilder()
        assert builder.forgetting_model is not None
        assert isinstance(builder.forgetting_model, ForgettingModel)

    def test_init_forgetting_model_none_creates_default(self):
        """forgetting_model=None 时内部创建默认 ForgettingModel."""
        builder = ProfileBuilder(forgetting_model=None)
        assert isinstance(builder.forgetting_model, ForgettingModel)

    def test_init_store_and_forgetting_model_together(self):
        """store 与 forgetting_model 可同时注入."""
        store = InMemoryL2Store()
        fm = ForgettingModel()
        builder = ProfileBuilder(store=store, forgetting_model=fm)
        assert builder.forgetting_model is fm

    def test_decay_not_applied_for_recent_attempt(self):
        """最近作答 (1 小时内) 不衰减, 掌握度保持原值."""
        builder = ProfileBuilder()
        now = time.time()
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.8, attempts=3,
                last_attempt_time=now,
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.kp_mastery["kp-1"] == pytest.approx(0.8)

    def test_decay_not_applied_when_never_attempted(self):
        """从未作答 (last_attempt_time=0) 不衰减, 掌握度保持原值."""
        builder = ProfileBuilder()
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.8, attempts=0,
                last_attempt_time=0.0,
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.kp_mastery["kp-1"] == pytest.approx(0.8)

    def test_decay_applied_for_old_attempt_default_model(self):
        """作答距今超过 168 小时 (默认模型阈值) -> 掌握度衰减下降."""
        builder = ProfileBuilder()
        now = time.time()
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.8, attempts=1,
                last_attempt_time=now - 200 * 3600,  # 200 小时前
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.kp_mastery["kp-1"] < 0.8
        assert snapshot.kp_mastery["kp-1"] > 0.0

    def test_decay_stability_from_attempts(self):
        """attempts 越多 stability 越大, 衰减越少 (掌握度更高)."""
        builder = ProfileBuilder()
        now = time.time()
        old = now - 200 * 3600
        tracing = {
            "kp-few": TracingState(
                kp_id="kp-few", mastery_prob=0.8, attempts=1,
                last_attempt_time=old,
            ),
            "kp-many": TracingState(
                kp_id="kp-many", mastery_prob=0.8, attempts=10,
                last_attempt_time=old,
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        # 作答次数多 -> stability 大 -> 衰减慢 -> 衰减后掌握度更高
        assert snapshot.kp_mastery["kp-many"] > snapshot.kp_mastery["kp-few"]

    def test_decay_one_hour_gate_with_aggressive_model(self):
        """1 小时门控: 注入激进模型, 30 分钟不衰减, 2 小时衰减."""
        # 阈值设为 0 -> 模型本身对任意正时间都衰减; 由 build 的 1 小时门控控制
        aggressive = ForgettingModel(base_lambda=1.0, decay_threshold_hours=0.0)
        builder = ProfileBuilder(forgetting_model=aggressive)
        now = time.time()
        tracing = {
            "kp-recent": TracingState(
                kp_id="kp-recent", mastery_prob=0.8, attempts=1,
                last_attempt_time=now - 0.5 * 3600,  # 30 分钟 (< 1 小时)
            ),
            "kp-old": TracingState(
                kp_id="kp-old", mastery_prob=0.8, attempts=1,
                last_attempt_time=now - 2 * 3600,  # 2 小时 (> 1 小时)
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        # 30 分钟: 门控拦截 -> 不衰减
        assert snapshot.kp_mastery["kp-recent"] == pytest.approx(0.8)
        # 2 小时: 门控放行 -> 激进模型衰减 -> 掌握度下降
        assert snapshot.kp_mastery["kp-old"] < 0.8

    def test_decay_uses_injected_model(self):
        """注入自定义模型参数影响衰减结果."""
        # 大 base_lambda -> 衰减更剧烈
        steep = ForgettingModel(base_lambda=0.5, decay_threshold_hours=0.0)
        builder_steep = ProfileBuilder(forgetting_model=steep)
        now = time.time()
        old = now - 200 * 3600
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.8, attempts=1,
                last_attempt_time=old,
            ),
        }
        irt = IRTState(theta=0.0)
        snap_steep = builder_steep.build("l1", tracing, irt)

        builder_default = ProfileBuilder()
        snap_default = builder_default.build("l1", tracing, irt)
        # 陡峭模型 (大 lambda, 阈值 0) 衰减更剧烈 -> 掌握度更低
        assert snap_steep.kp_mastery["kp-1"] < snap_default.kp_mastery["kp-1"]


# ============================================================
# 7. ProfileBuilder - 加权平均掌握度
# ============================================================


class TestProfileBuilderWeightedAverage:
    """ProfileBuilder 加权平均掌握度测试 — 按 attempts 加权."""

    def test_weighted_average_uses_attempts_weights(self):
        """按 attempts 加权: 高掌握度+多次作答会拉高均值 (高于简单平均)."""
        builder = ProfileBuilder()
        now = time.time()
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.5, attempts=10,
                last_attempt_time=now,
            ),
            "kp-2": TracingState(
                kp_id="kp-2", mastery_prob=0.1, attempts=1,
                last_attempt_time=now,
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        # 加权均值 = (0.5*10 + 0.1*1)/11 ≈ 0.4636 >= 0.4 -> intermediate
        # 简单均值 = (0.5+0.1)/2 = 0.3 < 0.4 -> beginner
        assert snapshot.level == "intermediate"

    def test_simple_average_fallback_all_zero_attempts(self):
        """所有 attempts=0 时回退简单平均."""
        builder = ProfileBuilder()
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.5, attempts=0,
            ),
            "kp-2": TracingState(
                kp_id="kp-2", mastery_prob=0.1, attempts=0,
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        # 简单均值 = 0.3 < 0.4 -> beginner
        assert snapshot.level == "beginner"

    def test_weighted_average_boundary(self):
        """加权均值精确跨越掌握度边界 (beginner -> intermediate)."""
        builder = ProfileBuilder()
        now = time.time()
        # 加权均值 = (0.9*1 + 0.3*9)/10 = (0.9+2.7)/10 = 0.36 < 0.4 -> beginner
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.9, attempts=1,
                last_attempt_time=now,
            ),
            "kp-2": TracingState(
                kp_id="kp-2", mastery_prob=0.3, attempts=9,
                last_attempt_time=now,
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.level == "beginner"

    def test_weighted_average_mixed_zero_and_nonzero_attempts(self):
        """部分 attempts=0 仍按非零权重加权 (非全零)."""
        builder = ProfileBuilder()
        now = time.time()
        # 加权均值 = (0.8*4 + 0.2*0)/(4+0) = 3.2/4 = 0.8
        # 简单均值 = (0.8+0.2)/2 = 0.5
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.8, attempts=4,
                last_attempt_time=now,
            ),
            "kp-2": TracingState(
                kp_id="kp-2", mastery_prob=0.2, attempts=0,
            ),
        }
        irt = IRTState(theta=0.5)  # theta=0.5 满足 advanced 的 theta 条件
        snapshot = builder.build("l1", tracing, irt)
        # 加权均值 0.8 >= 0.7 且 theta>=0.5 -> advanced
        # 简单均值 0.5 < 0.7 -> intermediate
        assert snapshot.level == "advanced"

    def test_empty_tracing_states_avg_zero(self):
        """空 tracing_states -> avg_mastery=0.0."""
        builder = ProfileBuilder()
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", {}, irt)
        assert snapshot.kp_mastery == {}
        assert snapshot.level == "beginner"


# ============================================================
# 8. ProfileBuilder - weak_kps_threshold 参数
# ============================================================


class TestProfileBuilderWeakKpsThreshold:
    """ProfileBuilder.weak_kps_threshold 测试 — 可配置薄弱阈值."""

    def test_weak_kps_threshold_default_0_5(self):
        """默认阈值 0.5: mastery < 0.5 视为薄弱."""
        builder = ProfileBuilder()
        tracing = {
            "kp-1": TracingState(kp_id="kp-1", mastery_prob=0.8),
            "kp-2": TracingState(kp_id="kp-2", mastery_prob=0.49),
            "kp-3": TracingState(kp_id="kp-3", mastery_prob=0.5),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert "kp-2" in snapshot.weak_kps
        assert "kp-1" not in snapshot.weak_kps
        assert "kp-3" not in snapshot.weak_kps  # 0.5 不 < 0.5

    def test_weak_kps_threshold_custom(self):
        """自定义阈值 0.7: mastery < 0.7 视为薄弱."""
        builder = ProfileBuilder()
        tracing = {
            "kp-1": TracingState(kp_id="kp-1", mastery_prob=0.8),
            "kp-2": TracingState(kp_id="kp-2", mastery_prob=0.6),
            "kp-3": TracingState(kp_id="kp-3", mastery_prob=0.5),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt, weak_kps_threshold=0.7)
        assert "kp-2" in snapshot.weak_kps  # 0.6 < 0.7
        assert "kp-3" in snapshot.weak_kps  # 0.5 < 0.7
        assert "kp-1" not in snapshot.weak_kps  # 0.8 不 < 0.7

    def test_weak_kps_threshold_zero(self):
        """阈值 0.0: 无薄弱知识点 (mastery >= 0)."""
        builder = ProfileBuilder()
        tracing = {
            "kp-1": TracingState(kp_id="kp-1", mastery_prob=0.1),
            "kp-2": TracingState(kp_id="kp-2", mastery_prob=0.0),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt, weak_kps_threshold=0.0)
        # mastery < 0.0 -> 无 (0.0 不 < 0.0)
        assert snapshot.weak_kps == []

    def test_weak_kps_threshold_high(self):
        """高阈值 0.9: 大部分知识点视为薄弱."""
        builder = ProfileBuilder()
        tracing = {
            "kp-1": TracingState(kp_id="kp-1", mastery_prob=0.95),
            "kp-2": TracingState(kp_id="kp-2", mastery_prob=0.5),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt, weak_kps_threshold=0.9)
        assert "kp-2" in snapshot.weak_kps
        assert "kp-1" not in snapshot.weak_kps  # 0.95 不 < 0.9

    def test_weak_kps_threshold_uses_decayed_mastery(self):
        """薄弱判定使用衰减后的掌握度."""
        builder = ProfileBuilder()
        now = time.time()
        # 原始 0.8, 但作答于 300 小时前 -> 衰减后低于 0.5 -> 薄弱
        tracing = {
            "kp-1": TracingState(
                kp_id="kp-1", mastery_prob=0.8, attempts=1,
                last_attempt_time=now - 300 * 3600,
            ),
        }
        irt = IRTState(theta=0.0)
        snapshot = builder.build("l1", tracing, irt, weak_kps_threshold=0.5)
        # 衰减后 kp-1 掌握度 < 0.5 -> 薄弱
        assert "kp-1" in snapshot.weak_kps


# ============================================================
# 9. ProfileBuilder - confidence 字段
# ============================================================


class TestProfileBuilderConfidence:
    """ProfileBuilder confidence 字段测试 — 基于 IRT SE 的置信度."""

    def test_confidence_field_present(self):
        """返回的 LearnerSnapshot 含 confidence 字段."""
        builder = ProfileBuilder()
        tracing = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        irt = IRTState(theta=0.0, se=0.3)
        snapshot = builder.build("l1", tracing, irt)
        assert hasattr(snapshot, "confidence")
        assert snapshot.confidence is not None

    def test_confidence_from_se(self):
        """confidence = 1 / (1 + se)."""
        builder = ProfileBuilder()
        tracing = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        irt = IRTState(theta=0.0, se=0.3)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.confidence == pytest.approx(1.0 / 1.3)

    def test_confidence_max_when_se_zero(self):
        """se=0 -> confidence=1.0 (最大置信度)."""
        builder = ProfileBuilder()
        tracing = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        irt = IRTState(theta=0.0, se=0.0)
        snapshot = builder.build("l1", tracing, irt)
        assert snapshot.confidence == pytest.approx(1.0)

    def test_confidence_decreases_with_se(self):
        """se 越大 confidence 越低."""
        builder = ProfileBuilder()
        tracing = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        snap_low_se = builder.build(
            "l1", tracing, IRTState(theta=0.0, se=0.1)
        )
        snap_high_se = builder.build(
            "l1", tracing, IRTState(theta=0.0, se=1.0)
        )
        assert snap_low_se.confidence > snap_high_se.confidence

    def test_confidence_is_float(self):
        """confidence 为 float 类型."""
        builder = ProfileBuilder()
        tracing = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        irt = IRTState(theta=0.0, se=0.3)
        snapshot = builder.build("l1", tracing, irt)
        assert isinstance(snapshot.confidence, float)

    def test_confidence_in_unit_range(self):
        """confidence 落在 (0, 1] 区间."""
        builder = ProfileBuilder()
        tracing = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        for se in (0.0, 0.1, 0.3, 0.5, 1.0, 2.0):
            snap = builder.build("l1", tracing, IRTState(theta=0.0, se=se))
            assert 0.0 < snap.confidence <= 1.0

    def test_confidence_persisted_to_store(self):
        """confidence 随画像保存到 store."""
        store = InMemoryL2Store()
        builder = ProfileBuilder(store=store)
        tracing = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        irt = IRTState(theta=0.0, se=0.4)
        snapshot = builder.build("l1", tracing, irt)
        loaded = store.get_profile("l1")
        assert loaded is not None
        assert loaded.confidence == pytest.approx(snapshot.confidence)

    def test_confidence_roundtrip_serialization(self):
        """confidence 经 to_dict/from_dict 往返保持一致."""
        builder = ProfileBuilder()
        tracing = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        irt = IRTState(theta=0.0, se=0.25)
        snapshot = builder.build("l1", tracing, irt)
        restored = LearnerSnapshot.from_dict(snapshot.to_dict())
        assert restored.confidence == pytest.approx(snapshot.confidence)
