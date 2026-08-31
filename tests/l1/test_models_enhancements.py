"""L1 核心数据模型增强模型测试 — ZPD / KnowledgeComponent / MasteryTrajectory / StudyPlan / LearningEfficiency.

验证 5 个高竞争力增强模型的:
- 构造与字段验证
- 业务逻辑方法
- 序列化往返一致性 (to_dict → from_dict → 等价)
- 边界条件与异常处理
"""

import pytest
import time


# ============================================================
# ZoneOfProximalDevelopment 测试
# ============================================================


class TestZoneOfProximalDevelopment:
    """最近发展区模型测试."""

    def test_construction_default_deltas(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        zpd = ZoneOfProximalDevelopment(learner_theta=0.5)
        assert zpd.zpd_lower == pytest.approx(0.0)
        assert zpd.zpd_upper == pytest.approx(1.0)

    def test_construction_custom_deltas(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        zpd = ZoneOfProximalDevelopment(
            learner_theta=1.0, delta_lower=0.8, delta_upper=0.6
        )
        assert zpd.zpd_lower == pytest.approx(0.2)
        assert zpd.zpd_upper == pytest.approx(1.6)

    def test_clamping_at_boundaries(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        # theta = -3, lower should clamp to -3
        zpd_low = ZoneOfProximalDevelopment(learner_theta=-3.0, delta_lower=1.0)
        assert zpd_low.zpd_lower == pytest.approx(-3.0)

        # theta = 3, upper should clamp to 3
        zpd_high = ZoneOfProximalDevelopment(learner_theta=3.0, delta_upper=1.0)
        assert zpd_high.zpd_upper == pytest.approx(3.0)

    def test_invalid_theta_raises(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        with pytest.raises(ValueError, match="learner_theta"):
            ZoneOfProximalDevelopment(learner_theta=3.5)
        with pytest.raises(ValueError, match="learner_theta"):
            ZoneOfProximalDevelopment(learner_theta=-3.5)

    def test_invalid_delta_raises(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        with pytest.raises(ValueError, match="delta_lower"):
            ZoneOfProximalDevelopment(learner_theta=0.0, delta_lower=0.0)
        with pytest.raises(ValueError, match="delta_lower"):
            ZoneOfProximalDevelopment(learner_theta=0.0, delta_lower=-0.5)
        with pytest.raises(ValueError, match="delta_upper"):
            ZoneOfProximalDevelopment(learner_theta=0.0, delta_upper=0.0)

    def test_is_in_zpd(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        zpd = ZoneOfProximalDevelopment(learner_theta=0.5, delta_lower=0.5, delta_upper=0.5)
        assert zpd.is_in_zpd(0.3) is True
        assert zpd.is_in_zpd(0.0) is True  # boundary
        assert zpd.is_in_zpd(1.0) is True  # boundary
        assert zpd.is_in_zpd(-0.1) is False
        assert zpd.is_in_zpd(1.1) is False

    def test_recommended_difficulty(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        zpd = ZoneOfProximalDevelopment(learner_theta=1.0, delta_lower=0.5, delta_upper=0.5)
        assert zpd.recommended_difficulty() == pytest.approx(1.0)

    def test_adjustment_direction(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        zpd = ZoneOfProximalDevelopment(learner_theta=0.0, delta_lower=0.5, delta_upper=0.5)
        assert zpd.adjustment_direction(-1.0) == "increase"
        assert zpd.adjustment_direction(1.0) == "decrease"
        assert zpd.adjustment_direction(0.3) == "optimal"

    def test_serialization_roundtrip(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        original = ZoneOfProximalDevelopment(
            learner_theta=0.8, delta_lower=0.6, delta_upper=0.4
        )
        d = original.to_dict()
        restored = ZoneOfProximalDevelopment.from_dict(d)
        assert restored.learner_theta == original.learner_theta
        assert restored.delta_lower == original.delta_lower
        assert restored.delta_upper == original.delta_upper
        assert restored.zpd_lower == original.zpd_lower
        assert restored.zpd_upper == original.zpd_upper

    def test_serialization_preserves_clamping(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        original = ZoneOfProximalDevelopment(learner_theta=2.8, delta_upper=1.0)
        d = original.to_dict()
        restored = ZoneOfProximalDevelopment.from_dict(d)
        assert restored.zpd_upper == pytest.approx(3.0)


# ============================================================
# KnowledgeComponent 测试
# ============================================================


class TestKnowledgeComponent:
    """知识点元数据模型测试."""

    def test_construction_minimal(self):
        from dy3_polaris.l1.models import KnowledgeComponent

        kc = KnowledgeComponent(kc_id="kc-001", name="二叉搜索树")
        assert kc.kc_id == "kc-001"
        assert kc.name == "二叉搜索树"
        assert kc.bloom_tag is None
        assert kc.estimated_difficulty == 0.5
        assert kc.prerequisite_kcs == []

    def test_construction_full(self):
        from dy3_polaris.l1.models import KnowledgeComponent, BloomTag, KnowledgeType
        from dy3_polaris.l3.api_models import BloomLevel

        kc = KnowledgeComponent(
            kc_id="kc-002",
            name="动态规划",
            bloom_tag=BloomTag(
                cognitive_level=BloomLevel.APPLY,
                knowledge_type=KnowledgeType.PROCEDURAL,
            ),
            estimated_difficulty=0.8,
            prerequisite_kcs=["kc-001", "kc-003"],
            estimated_time_minutes=45,
            kc_description="动态规划基础概念与应用",
        )
        assert kc.bloom_tag is not None
        assert kc.estimated_difficulty == 0.8
        assert len(kc.prerequisite_kcs) == 2

    def test_invalid_difficulty_raises(self):
        from dy3_polaris.l1.models import KnowledgeComponent

        with pytest.raises(ValueError, match="estimated_difficulty"):
            KnowledgeComponent(kc_id="kc-1", name="test", estimated_difficulty=1.5)
        with pytest.raises(ValueError, match="estimated_difficulty"):
            KnowledgeComponent(kc_id="kc-1", name="test", estimated_difficulty=-0.1)

    def test_invalid_time_raises(self):
        from dy3_polaris.l1.models import KnowledgeComponent

        with pytest.raises(ValueError, match="estimated_time_minutes"):
            KnowledgeComponent(kc_id="kc-1", name="test", estimated_time_minutes=-5)

    def test_has_prerequisites(self):
        from dy3_polaris.l1.models import KnowledgeComponent

        kc_with = KnowledgeComponent(kc_id="kc-1", name="A", prerequisite_kcs=["kc-0"])
        kc_without = KnowledgeComponent(kc_id="kc-2", name="B")
        assert kc_with.has_prerequisites() is True
        assert kc_without.has_prerequisites() is False

    def test_serialization_without_bloom_tag(self):
        from dy3_polaris.l1.models import KnowledgeComponent

        original = KnowledgeComponent(
            kc_id="kc-001",
            name="二叉搜索树",
            estimated_difficulty=0.6,
            prerequisite_kcs=["kc-000"],
            estimated_time_minutes=30,
        )
        d = original.to_dict()
        assert d["bloom_tag"] is None
        restored = KnowledgeComponent.from_dict(d)
        assert restored.kc_id == original.kc_id
        assert restored.name == original.name
        assert restored.estimated_difficulty == original.estimated_difficulty
        assert restored.prerequisite_kcs == original.prerequisite_kcs
        assert restored.bloom_tag is None

    def test_serialization_with_bloom_tag(self):
        from dy3_polaris.l1.models import KnowledgeComponent, BloomTag, KnowledgeType
        from dy3_polaris.l3.api_models import BloomLevel

        original = KnowledgeComponent(
            kc_id="kc-002",
            name="动态规划",
            bloom_tag=BloomTag(
                cognitive_level=BloomLevel.ANALYZE,
                knowledge_type=KnowledgeType.CONCEPTUAL,
            ),
            estimated_difficulty=0.8,
            prerequisite_kcs=["kc-001"],
            estimated_time_minutes=45,
        )
        d = original.to_dict()
        assert d["bloom_tag"] is not None
        restored = KnowledgeComponent.from_dict(d)
        assert restored.kc_id == original.kc_id
        assert restored.bloom_tag is not None
        assert restored.bloom_tag.matrix_cell() == original.bloom_tag.matrix_cell()


# ============================================================
# MasteryTrajectoryPoint 测试
# ============================================================


class TestMasteryTrajectoryPoint:
    """掌握度轨迹点测试."""

    def test_construction(self):
        from dy3_polaris.l1.models import MasteryTrajectoryPoint

        point = MasteryTrajectoryPoint(
            kc_id="kc-001", timestamp=1000, p_know=0.7, decay_factor=0.8
        )
        assert point.kc_id == "kc-001"
        assert point.p_know == 0.7
        assert point.decay_factor == 0.8
        assert point.interaction_type == "practice"

    def test_invalid_p_know_raises(self):
        from dy3_polaris.l1.models import MasteryTrajectoryPoint

        with pytest.raises(ValueError, match="p_know"):
            MasteryTrajectoryPoint(kc_id="kc-1", timestamp=1, p_know=1.5)
        with pytest.raises(ValueError, match="p_know"):
            MasteryTrajectoryPoint(kc_id="kc-1", timestamp=1, p_know=-0.1)

    def test_invalid_decay_raises(self):
        from dy3_polaris.l1.models import MasteryTrajectoryPoint

        with pytest.raises(ValueError, match="decay_factor"):
            MasteryTrajectoryPoint(kc_id="kc-1", timestamp=1, p_know=0.5, decay_factor=1.5)

    def test_effective_mastery(self):
        from dy3_polaris.l1.models import MasteryTrajectoryPoint

        point = MasteryTrajectoryPoint(
            kc_id="kc-1", timestamp=1, p_know=0.8, decay_factor=0.75
        )
        assert point.effective_mastery() == pytest.approx(0.6)

    def test_serialization_roundtrip(self):
        from dy3_polaris.l1.models import MasteryTrajectoryPoint

        original = MasteryTrajectoryPoint(
            kc_id="kc-001", timestamp=1234567890,
            p_know=0.65, decay_factor=0.9, interaction_type="quiz"
        )
        d = original.to_dict()
        restored = MasteryTrajectoryPoint.from_dict(d)
        assert restored.kc_id == original.kc_id
        assert restored.timestamp == original.timestamp
        assert restored.p_know == original.p_know
        assert restored.decay_factor == original.decay_factor
        assert restored.interaction_type == original.interaction_type


# ============================================================
# MasteryTrajectory 测试
# ============================================================


class TestMasteryTrajectory:
    """掌握度轨迹测试."""

    def test_empty_trajectory(self):
        from dy3_polaris.l1.models import MasteryTrajectory

        traj = MasteryTrajectory(kc_id="kc-001")
        assert traj.latest() is None
        assert traj.earliest() is None
        assert traj.trend() == "insufficient_data"
        assert traj.mastery_delta() == 0.0

    def test_add_point_and_sorting(self):
        from dy3_polaris.l1.models import MasteryTrajectory, MasteryTrajectoryPoint

        traj = MasteryTrajectory(kc_id="kc-001")
        p1 = MasteryTrajectoryPoint(kc_id="kc-001", timestamp=3000, p_know=0.6)
        p2 = MasteryTrajectoryPoint(kc_id="kc-001", timestamp=1000, p_know=0.3)
        p3 = MasteryTrajectoryPoint(kc_id="kc-001", timestamp=2000, p_know=0.5)

        traj.add_point(p1)
        traj.add_point(p2)
        traj.add_point(p3)

        # Points should be sorted by timestamp
        assert traj.points[0].timestamp == 1000
        assert traj.points[1].timestamp == 2000
        assert traj.points[2].timestamp == 3000

    def test_latest_and_earliest(self):
        from dy3_polaris.l1.models import MasteryTrajectory, MasteryTrajectoryPoint

        traj = MasteryTrajectory(kc_id="kc-001")
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=1000, p_know=0.3))
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=2000, p_know=0.6))

        assert traj.earliest().p_know == 0.3
        assert traj.latest().p_know == 0.6

    def test_trend_improving(self):
        from dy3_polaris.l1.models import MasteryTrajectory, MasteryTrajectoryPoint

        traj = MasteryTrajectory(kc_id="kc-001")
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=1000, p_know=0.3))
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=2000, p_know=0.5))
        assert traj.trend() == "improving"

    def test_trend_declining(self):
        from dy3_polaris.l1.models import MasteryTrajectory, MasteryTrajectoryPoint

        traj = MasteryTrajectory(kc_id="kc-001")
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=1000, p_know=0.7))
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=2000, p_know=0.4))
        assert traj.trend() == "declining"

    def test_trend_stable(self):
        from dy3_polaris.l1.models import MasteryTrajectory, MasteryTrajectoryPoint

        traj = MasteryTrajectory(kc_id="kc-001")
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=1000, p_know=0.5))
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=2000, p_know=0.52))
        assert traj.trend() == "stable"

    def test_mastery_delta(self):
        from dy3_polaris.l1.models import MasteryTrajectory, MasteryTrajectoryPoint

        traj = MasteryTrajectory(kc_id="kc-001")
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=1000, p_know=0.3))
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=2000, p_know=0.7))
        assert traj.mastery_delta() == pytest.approx(0.4)

    def test_serialization_roundtrip(self):
        from dy3_polaris.l1.models import MasteryTrajectory, MasteryTrajectoryPoint

        original = MasteryTrajectory(kc_id="kc-001")
        original.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=1000, p_know=0.3))
        original.add_point(MasteryTrajectoryPoint(kc_id="kc-001", timestamp=2000, p_know=0.6))

        d = original.to_dict()
        restored = MasteryTrajectory.from_dict(d)
        assert restored.kc_id == original.kc_id
        assert len(restored.points) == 2
        assert restored.points[0].p_know == 0.3
        assert restored.points[1].p_know == 0.6

    def test_serialization_empty(self):
        from dy3_polaris.l1.models import MasteryTrajectory

        original = MasteryTrajectory(kc_id="kc-empty")
        d = original.to_dict()
        restored = MasteryTrajectory.from_dict(d)
        assert restored.kc_id == "kc-empty"
        assert len(restored.points) == 0


# ============================================================
# StudyBlock 测试
# ============================================================


class TestStudyBlock:
    """学习时间块测试."""

    def test_construction(self):
        from dy3_polaris.l1.models import StudyBlock, LearningPhase

        block = StudyBlock(
            kc_id="kc-001", start_time=1000000, duration_minutes=30,
            phase=LearningPhase.PRACTICE,
        )
        assert block.kc_id == "kc-001"
        assert block.duration_minutes == 30
        assert block.phase == LearningPhase.PRACTICE

    def test_invalid_duration_raises(self):
        from dy3_polaris.l1.models import StudyBlock

        with pytest.raises(ValueError, match="duration_minutes"):
            StudyBlock(kc_id="kc-1", start_time=0, duration_minutes=0)
        with pytest.raises(ValueError, match="duration_minutes"):
            StudyBlock(kc_id="kc-1", start_time=0, duration_minutes=-5)

    def test_end_time(self):
        from dy3_polaris.l1.models import StudyBlock

        block = StudyBlock(kc_id="kc-1", start_time=1000000, duration_minutes=30)
        # 30 minutes = 30 * 60 * 1000 = 1,800,000 ms
        assert block.end_time() == 1000000 + 30 * 60 * 1000

    def test_serialization_roundtrip(self):
        from dy3_polaris.l1.models import StudyBlock, LearningPhase

        original = StudyBlock(
            kc_id="kc-001", start_time=1234567890000,
            duration_minutes=45, phase=LearningPhase.REVIEW,
        )
        d = original.to_dict()
        restored = StudyBlock.from_dict(d)
        assert restored.kc_id == original.kc_id
        assert restored.start_time == original.start_time
        assert restored.duration_minutes == original.duration_minutes
        assert restored.phase == original.phase


# ============================================================
# StudyPlan 测试
# ============================================================


class TestStudyPlan:
    """学习计划测试."""

    def test_construction_empty(self):
        from dy3_polaris.l1.models import StudyPlan

        plan = StudyPlan(user_id="u-001")
        assert plan.user_id == "u-001"
        assert plan.block_count() == 0
        assert plan.total_estimated_minutes == 0

    def test_construction_with_blocks(self):
        from dy3_polaris.l1.models import StudyPlan, StudyBlock

        plan = StudyPlan(
            user_id="u-001",
            blocks=[
                StudyBlock(kc_id="kc-1", start_time=1000, duration_minutes=30),
                StudyBlock(kc_id="kc-2", start_time=2000, duration_minutes=45),
            ],
        )
        assert plan.block_count() == 2
        assert plan.total_estimated_minutes == 75

    def test_add_block(self):
        from dy3_polaris.l1.models import StudyPlan, StudyBlock

        plan = StudyPlan(user_id="u-001")
        plan.add_block(StudyBlock(kc_id="kc-1", start_time=1000, duration_minutes=30))
        assert plan.total_estimated_minutes == 30
        plan.add_block(StudyBlock(kc_id="kc-2", start_time=2000, duration_minutes=20))
        assert plan.total_estimated_minutes == 50

    def test_serialization_roundtrip(self):
        from dy3_polaris.l1.models import StudyPlan, StudyBlock, LearningGoal

        original = StudyPlan(
            user_id="u-001",
            blocks=[
                StudyBlock(kc_id="kc-1", start_time=1000, duration_minutes=30),
                StudyBlock(kc_id="kc-2", start_time=2000, duration_minutes=45),
            ],
            goals=[
                LearningGoal(description="掌握二叉树", priority=4),
            ],
        )
        d = original.to_dict()
        restored = StudyPlan.from_dict(d)
        assert restored.user_id == original.user_id
        assert restored.block_count() == 2
        assert restored.total_estimated_minutes == 75
        assert len(restored.goals) == 1
        assert restored.goals[0].description == "掌握二叉树"

    def test_serialization_empty_plan(self):
        from dy3_polaris.l1.models import StudyPlan

        original = StudyPlan(user_id="u-empty")
        d = original.to_dict()
        restored = StudyPlan.from_dict(d)
        assert restored.user_id == "u-empty"
        assert restored.block_count() == 0


# ============================================================
# LearningEfficiency 测试
# ============================================================


class TestLearningEfficiency:
    """学习效率指标测试."""

    def test_construction_defaults(self):
        from dy3_polaris.l1.models import LearningEfficiency

        eff = LearningEfficiency()
        assert eff.mastery_gain == 0.0
        assert eff.time_spent_ms == 0
        assert eff.interactions == 0

    def test_invalid_time_raises(self):
        from dy3_polaris.l1.models import LearningEfficiency

        with pytest.raises(ValueError, match="time_spent_ms"):
            LearningEfficiency(time_spent_ms=-1)

    def test_invalid_interactions_raises(self):
        from dy3_polaris.l1.models import LearningEfficiency

        with pytest.raises(ValueError, match="interactions"):
            LearningEfficiency(interactions=-1)

    def test_time_efficiency(self):
        from dy3_polaris.l1.models import LearningEfficiency, MS_PER_HOUR

        # 1 hour, 0.3 gain → 0.3 per hour
        eff = LearningEfficiency(
            mastery_gain=0.3, time_spent_ms=MS_PER_HOUR, interactions=10
        )
        assert eff.time_efficiency() == pytest.approx(0.3)

    def test_time_efficiency_zero_time(self):
        from dy3_polaris.l1.models import LearningEfficiency

        eff = LearningEfficiency(mastery_gain=0.5, time_spent_ms=0)
        assert eff.time_efficiency() == 0.0

    def test_interaction_efficiency(self):
        from dy3_polaris.l1.models import LearningEfficiency

        eff = LearningEfficiency(mastery_gain=0.4, interactions=20)
        assert eff.interaction_efficiency() == pytest.approx(0.02)

    def test_interaction_efficiency_zero_interactions(self):
        from dy3_polaris.l1.models import LearningEfficiency

        eff = LearningEfficiency(mastery_gain=0.5, interactions=0)
        assert eff.interaction_efficiency() == 0.0

    def test_efficiency_rating_high(self):
        from dy3_polaris.l1.models import LearningEfficiency, MS_PER_HOUR

        # 0.5 gain in 1 hour → 0.5 per hour > 0.3 → high
        eff = LearningEfficiency(
            mastery_gain=0.5, time_spent_ms=MS_PER_HOUR
        )
        assert eff.efficiency_rating() == "high"

    def test_efficiency_rating_medium(self):
        from dy3_polaris.l1.models import LearningEfficiency, MS_PER_HOUR

        # 0.2 gain in 1 hour → 0.2 per hour → medium
        eff = LearningEfficiency(
            mastery_gain=0.2, time_spent_ms=MS_PER_HOUR
        )
        assert eff.efficiency_rating() == "medium"

    def test_efficiency_rating_low(self):
        from dy3_polaris.l1.models import LearningEfficiency, MS_PER_HOUR

        # 0.05 gain in 1 hour → 0.05 per hour < 0.1 → low
        eff = LearningEfficiency(
            mastery_gain=0.05, time_spent_ms=MS_PER_HOUR
        )
        assert eff.efficiency_rating() == "low"

    def test_efficiency_rating_zero_time(self):
        from dy3_polaris.l1.models import LearningEfficiency

        eff = LearningEfficiency(mastery_gain=0.5, time_spent_ms=0)
        assert eff.efficiency_rating() == "low"

    def test_serialization_roundtrip(self):
        from dy3_polaris.l1.models import LearningEfficiency

        original = LearningEfficiency(
            mastery_gain=0.35,
            time_spent_ms=3_600_000,
            interactions=15,
            kc_id="kc-001",
            session_id="s-001",
        )
        d = original.to_dict()
        restored = LearningEfficiency.from_dict(d)
        assert restored.mastery_gain == original.mastery_gain
        assert restored.time_spent_ms == original.time_spent_ms
        assert restored.interactions == original.interactions
        assert restored.kc_id == original.kc_id
        assert restored.session_id == original.session_id

    def test_serialization_includes_computed_fields(self):
        from dy3_polaris.l1.models import LearningEfficiency, MS_PER_HOUR

        eff = LearningEfficiency(
            mastery_gain=0.4, time_spent_ms=MS_PER_HOUR, interactions=10
        )
        d = eff.to_dict()
        assert "time_efficiency" in d
        assert "interaction_efficiency" in d
        assert "efficiency_rating" in d
        assert d["efficiency_rating"] == "high"


# ============================================================
# 跨模型集成测试
# ============================================================


class TestCrossModelIntegration:
    """跨模型集成测试 — 验证新模型与已有模型的协作."""

    def test_zpd_with_irt_ability(self):
        """ZPD 与 IRTAbility 的集成: 使用 IRTAbility.theta 构造 ZPD."""
        from dy3_polaris.l1.models import IRTAbility, ZoneOfProximalDevelopment

        ability = IRTAbility(user_id="u-001", theta=0.8, standard_error=0.2)
        zpd = ZoneOfProximalDevelopment(learner_theta=ability.theta)
        assert zpd.is_in_zpd(0.8) is True
        assert zpd.recommended_difficulty() == pytest.approx(0.8)

    def test_knowledge_component_with_bloom_tag(self):
        """KnowledgeComponent 与 BloomTag 的集成."""
        from dy3_polaris.l1.models import KnowledgeComponent, BloomTag, KnowledgeType
        from dy3_polaris.l3.api_models import BloomLevel

        kc = KnowledgeComponent(
            kc_id="kc-001",
            name="傅里叶变换",
            bloom_tag=BloomTag(
                cognitive_level=BloomLevel.ANALYZE,
                knowledge_type=KnowledgeType.CONCEPTUAL,
            ),
            estimated_difficulty=0.75,
        )
        assert kc.bloom_tag.matrix_cell() == "analyze×conceptual"

    def test_mastery_trajectory_with_mastery_snapshot(self):
        """MasteryTrajectory 与 MasterySnapshot 的集成."""
        from dy3_polaris.l1.models import (
            MasterySnapshot, MasteryTrajectory, MasteryTrajectoryPoint,
        )

        snapshot = MasterySnapshot(kc_id="kc-001", p_know=0.7, last_practiced_at=1000)
        traj = MasteryTrajectory(kc_id="kc-001")
        traj.add_point(MasteryTrajectoryPoint(
            kc_id="kc-001", timestamp=1000, p_know=snapshot.p_know,
            decay_factor=snapshot.decay_factor,
        ))
        assert traj.latest().p_know == 0.7

    def test_study_plan_with_learning_path_nodes(self):
        """StudyPlan 与 LearningPath 的集成: 从 PathNode 创建 StudyBlock."""
        from dy3_polaris.l1.models import StudyPlan, StudyBlock, PathNode, LearningPhase

        path_nodes = [
            PathNode(kc_id="kc-1", order=1, estimated_time_minutes=30),
            PathNode(kc_id="kc-2", order=2, estimated_time_minutes=45),
        ]
        base_time = 1_000_000
        plan = StudyPlan(user_id="u-001")
        for i, node in enumerate(path_nodes):
            plan.add_block(StudyBlock(
                kc_id=node.kc_id,
                start_time=base_time + i * 3_600_000,
                duration_minutes=node.estimated_time_minutes,
                phase=LearningPhase.PRACTICE,
            ))
        assert plan.block_count() == 2
        assert plan.total_estimated_minutes == 75

    def test_learning_efficiency_with_engagement_metrics(self):
        """LearningEfficiency 与 EngagementMetrics 的集成."""
        from dy3_polaris.l1.models import LearningEfficiency, EngagementMetrics

        metrics = EngagementMetrics(
            session_duration_ms=3_600_000,
            login_frequency=5,
            completion_rate=0.8,
            accuracy_rate=0.75,
            avg_response_time_ms=30000,
        )
        eff = LearningEfficiency(
            mastery_gain=0.35,
            time_spent_ms=metrics.session_duration_ms,
            interactions=15,
        )
        assert eff.time_efficiency() == pytest.approx(0.35)
        assert eff.efficiency_rating() == "high"
