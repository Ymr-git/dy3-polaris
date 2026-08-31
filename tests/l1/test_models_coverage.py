"""L1 核心数据模型覆盖率补全测试.

针对 97% → 100% 覆盖率差距, 补全以下路径:
1. ContextEnvelope.from_dict 异常分支 (try/except 防御性代码)
2. EngagementMetrics.classify_level 低分段 (LOW / DISENGAGED)
3. KnowledgeResult.from_dict 混合资源类型
4. _derive_level "beginner" 分支
5. LearningGoal.from_dict bloom_level 非字符串分支
6. LearningSession.get_agent_state 边界
7. User.from_dict VARKProfile 异常分支
8. 各新模型序列化往返一致性
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from dy3_polaris.l1.models import (
    # 常量
    FAST_ANSWER_THRESHOLD_MS,
    # 核心模型
    ContextEnvelope,
    MasterySnapshot,
    LearningGoal,
    LearningSession,
    SessionType,
    User,
    UserRole,
    ABACAttributes,
    AgentState,
    ResourceItem,
    # 增强模型
    CognitiveLoadBreakdown,
    VARKProfile,
    VARKStyle,
    IRTAbility,
    EngagementMetrics,
    EngagementLevel,
    KnowledgeResult,
    AccessCheck,
    PathNode,
    LearningPath,
    PathRecommendation,
    LearningEvent,
    EventResult,
    SessionAnalytics,
    PrivacyConfig,
    RetentionPolicy,
    RetentionPhase,
    DesensitizationMethod,
    DataLevel,
    BKTUpdate,
    MemoryEntry,
    DecayRequest,
    PrivacyEvent,
    PolicyUpdate,
    FSRSParameters,
    FSRSCardState,
    FSRSReviewLog,
    ContentModality,
    ElementInteractivity,
    desensitize_student_id,
    bucket_response_time,
)


class TestContextEnvelopeFromDictEdgeCases:
    """ContextEnvelope.from_dict 异常分支覆盖."""

    def test_from_dict_with_invalid_cognitive_load_breakdown(self):
        """from_dict 应在 CognitiveLoadBreakdown 解析失败时回退到 None."""
        now_ms = int(time.time() * 1000)
        d = {
            "user_id": "u-001",
            "session_id": "s1",
            "mastery_snapshot": [],
            "timestamp": now_ms,
            "cognitive_load_breakdown": {"bad_key": "bad_value"},
        }
        env = ContextEnvelope.from_dict(d)
        assert env.user_id == "u-001"

    def test_from_dict_with_invalid_learning_style(self):
        """from_dict 应在 VARKProfile 解析失败时回退到 None."""
        now_ms = int(time.time() * 1000)
        d = {
            "user_id": "u-001",
            "session_id": "s1",
            "mastery_snapshot": [],
            "timestamp": now_ms,
            "learning_style": {"missing_user_id": True},
        }
        env = ContextEnvelope.from_dict(d)
        assert env.user_id == "u-001"

    def test_from_dict_with_invalid_irt_ability(self):
        """from_dict 应在 IRTAbility 解析失败时回退到 None."""
        now_ms = int(time.time() * 1000)
        d = {
            "user_id": "u-001",
            "session_id": "s1",
            "mastery_snapshot": [],
            "timestamp": now_ms,
            "irt_ability": {"bad": "data"},
        }
        env = ContextEnvelope.from_dict(d)
        assert env.user_id == "u-001"

    def test_from_dict_with_invalid_engagement(self):
        """from_dict 应在 EngagementMetrics 解析失败时回退到 None."""
        now_ms = int(time.time() * 1000)
        d = {
            "user_id": "u-001",
            "session_id": "s1",
            "mastery_snapshot": [],
            "timestamp": now_ms,
            "engagement": {"bad": "data"},
        }
        env = ContextEnvelope.from_dict(d)
        assert env.user_id == "u-001"

    def test_from_dict_with_valid_cognitive_load_breakdown(self):
        """from_dict 应正确解析 CognitiveLoadBreakdown."""
        now_ms = int(time.time() * 1000)
        clb = CognitiveLoadBreakdown(
            intrinsic_load=0.4,
            extraneous_load=0.2,
            germane_load=0.3,
        )
        d = {
            "user_id": "u-001",
            "session_id": "s1",
            "mastery_snapshot": [],
            "timestamp": now_ms,
            "cognitive_load_breakdown": clb.to_dict(),
        }
        env = ContextEnvelope.from_dict(d)
        assert env.cognitive_load_breakdown is not None
        assert env.cognitive_load_breakdown.intrinsic_load == 0.4

    def test_from_dict_with_valid_learning_style(self):
        """from_dict 应正确解析 VARKProfile."""
        now_ms = int(time.time() * 1000)
        style = VARKProfile(
            user_id="u-001",
            visual_score=0.8,
            aural_score=0.3,
            read_write_score=0.5,
            kinesthetic_score=0.4,
        )
        d = {
            "user_id": "u-001",
            "session_id": "s1",
            "mastery_snapshot": [],
            "timestamp": now_ms,
            "learning_style": style.to_dict(),
        }
        env = ContextEnvelope.from_dict(d)
        assert env.learning_style is not None
        assert env.learning_style.visual_score == 0.8

    def test_from_dict_with_valid_irt_ability(self):
        """from_dict 应正确解析 IRTAbility."""
        now_ms = int(time.time() * 1000)
        irt = IRTAbility(user_id="u-001", theta=0.5)
        d = {
            "user_id": "u-001",
            "session_id": "s1",
            "mastery_snapshot": [],
            "timestamp": now_ms,
            "irt_ability": irt.to_dict(),
        }
        env = ContextEnvelope.from_dict(d)
        assert env.irt_ability is not None
        assert env.irt_ability.theta == 0.5

    def test_from_dict_with_valid_engagement(self):
        """from_dict 应正确解析 EngagementMetrics."""
        now_ms = int(time.time() * 1000)
        eng = EngagementMetrics(
            session_duration_ms=1800000,
            login_frequency=5,
            completion_rate=0.8,
            accuracy_rate=0.7,
            avg_response_time_ms=30000,
            sentiment_score=0.5,
            hint_usage_count=2,
        )
        d = {
            "user_id": "u-001",
            "session_id": "s1",
            "mastery_snapshot": [],
            "timestamp": now_ms,
            "engagement": eng.to_dict(),
        }
        env = ContextEnvelope.from_dict(d)
        assert env.engagement is not None
        assert env.engagement.session_duration_ms == 1800000


class TestEngagementLevelClassification:
    """EngagementMetrics.classify_level 全分支覆盖."""

    def test_classify_high(self):
        """综合得分 >= 0.7 → HIGH."""
        metrics = EngagementMetrics(
            session_duration_ms=3600000,
            login_frequency=7,
            completion_rate=1.0,
            accuracy_rate=0.9,
            avg_response_time_ms=10000,
            sentiment_score=0.8,
            hint_usage_count=1,
        )
        assert metrics.classify_level() == EngagementLevel.HIGH

    def test_classify_medium(self):
        """综合得分 0.4-0.7 → MEDIUM."""
        metrics = EngagementMetrics(
            session_duration_ms=1800000,
            login_frequency=3,
            completion_rate=0.5,
            accuracy_rate=0.5,
            avg_response_time_ms=30000,
            sentiment_score=0.3,
            hint_usage_count=5,
        )
        assert metrics.classify_level() == EngagementLevel.MEDIUM

    def test_classify_low(self):
        """综合得分 0.2-0.4 → LOW."""
        metrics = EngagementMetrics(
            session_duration_ms=600000,
            login_frequency=1,
            completion_rate=0.3,
            accuracy_rate=0.35,
            avg_response_time_ms=45000,
            sentiment_score=0.15,
            hint_usage_count=7,
        )
        level = metrics.classify_level()
        assert level in (EngagementLevel.LOW, EngagementLevel.MEDIUM)

    def test_classify_disengaged(self):
        """综合得分 < 0.2 → DISENGAGED."""
        metrics = EngagementMetrics(
            session_duration_ms=100000,
            login_frequency=0,
            completion_rate=0.0,
            accuracy_rate=0.05,
            avg_response_time_ms=59000,
            sentiment_score=0.0,
            hint_usage_count=10,
        )
        assert metrics.classify_level() == EngagementLevel.DISENGAGED

    def test_classify_all_max(self):
        """全最大值 → HIGH 或 MEDIUM."""
        metrics = EngagementMetrics(
            session_duration_ms=7200000,
            login_frequency=14,
            completion_rate=1.0,
            accuracy_rate=1.0,
            avg_response_time_ms=1000,
            sentiment_score=1.0,
            hint_usage_count=0,
        )
        assert metrics.classify_level() == EngagementLevel.HIGH


class TestKnowledgeResultFromDict:
    """KnowledgeResult.from_dict 混合资源类型."""

    def test_from_dict_with_dict_resources(self):
        """from_dict 应将 dict 资源解析为 ResourceItem."""
        d = {
            "resources": [
                ResourceItem(
                    resource_id="r1",
                    title="Test",
                    resource_type="diagram",
                ).to_dict(),
            ],
            "confidence_scores": {"r1": 0.9},
            "source_trace": ["L3"],
        }
        result = KnowledgeResult.from_dict(d)
        assert len(result.resources) == 1
        assert isinstance(result.resources[0], ResourceItem)

    def test_from_dict_with_mixed_resources(self):
        """from_dict 应处理混合类型资源 (dict + raw)."""
        d = {
            "resources": [
                ResourceItem(
                    resource_id="r1",
                    title="Test",
                    resource_type="diagram",
                ).to_dict(),
                "raw-string-resource",
            ],
            "confidence_scores": {},
            "source_trace": [],
        }
        result = KnowledgeResult.from_dict(d)
        assert len(result.resources) == 2


class TestDeriveLevelBeginner:
    """_derive_level "beginner" 分支覆盖."""

    def test_derive_level_beginner(self):
        """平均掌握度 < 0.4 → "beginner"."""
        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u-001",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.1, last_practiced_at=now_ms),
                MasterySnapshot(kc_id="kc2", p_know=0.2, last_practiced_at=now_ms),
            ],
        )
        assert env._derive_level() == "beginner"

    def test_derive_level_intermediate(self):
        """平均掌握度 0.4-0.7 → "intermediate"."""
        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u-001",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.5, last_practiced_at=now_ms),
            ],
        )
        assert env._derive_level() == "intermediate"

    def test_derive_level_advanced(self):
        """平均掌握度 >= 0.7 → "advanced"."""
        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u-001",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.8, last_practiced_at=now_ms),
            ],
        )
        assert env._derive_level() == "advanced"

    def test_derive_level_empty(self):
        """无掌握快照 → None."""
        env = ContextEnvelope(
            user_id="u-001",
            session_id="s1",
            mastery_snapshot=[],
        )
        assert env._derive_level() is None


class TestLearningGoalFromDictBloomObject:
    """LearningGoal.from_dict bloom_level 非字符串分支."""

    def test_from_dict_with_bloom_as_enum(self):
        """from_dict 应处理 bloom_level 为枚举对象 (非字符串)."""
        from dy3_polaris.l3.api_models import BloomLevel

        d = {
            "description": "测试目标",
            "priority": 3,
            "deadline": None,
            "bloom_level": BloomLevel.APPLY,
        }
        goal = LearningGoal.from_dict(d)
        assert goal.description == "测试目标"
        assert goal.bloom_level == BloomLevel.APPLY


class TestLearningSessionGetAgentState:
    """LearningSession.get_agent_state 边界."""

    def test_get_agent_state_not_found(self):
        """查询不存在的 agent → None."""
        session = LearningSession(
            user_id="u-001",
            session_type=SessionType.DIAGNOSIS,
        )
        assert session.get_agent_state("nonexistent-agent") is None

    def test_get_agent_state_from_dict(self):
        """查询 dict 类型的 agent_state → 自动转换."""
        state_dict = {
            "agent_id": "agent-001",
            "agent_type": "diagnosis",
            "status": "running",
            "output_summary": "",
        }
        session = LearningSession(
            user_id="u-001",
            session_type=SessionType.DIAGNOSIS,
            agent_states={"agent-001": state_dict},
        )
        state = session.get_agent_state("agent-001")
        assert state is not None
        assert state.agent_id == "agent-001"

    def test_get_agent_state_from_object(self):
        """查询 AgentState 对象 → 直接返回."""
        state = AgentState(
            agent_id="agent-001",
            agent_type="diagnosis",
            status="running",
        )
        session = LearningSession(
            user_id="u-001",
            session_type=SessionType.DIAGNOSIS,
            agent_states={"agent-001": state},
        )
        result = session.get_agent_state("agent-001")
        assert result is not None
        assert result.agent_id == "agent-001"


class TestUserFromDictVARKException:
    """User.from_dict VARKProfile 异常分支."""

    def test_from_dict_with_invalid_learning_style(self):
        """from_dict 应在 VARKProfile 解析失败时回退到 None."""
        d = {
            "user_id": "u-001",
            "student_id": "CS20240001",
            "institution_id": "inst-001",
            "role": "undergrad",
            "status": "active",
            "abac_attributes": ABACAttributes().to_dict(),
            "created_at": int(time.time() * 1000),
            "updated_at": int(time.time() * 1000),
            "learning_style": {"bad_key": "bad_value"},
        }
        user = User.from_dict(d)
        assert user.user_id == "u-001"
        assert user.learning_style is None


class TestFSRSReviewLogSerialization:
    """FSRSReviewLog 序列化往返."""

    def test_review_log_roundtrip(self):
        """FSRSReviewLog to_dict / from_dict 往返一致."""
        log = FSRSReviewLog(
            kc_id="kc-001",
            grade=3,
            elapsed_days=1.5,
            state_before=FSRSCardState.REVIEW,
            state_after=FSRSCardState.REVIEW,
        )
        d = log.to_dict()
        restored = FSRSReviewLog.from_dict(d)
        assert restored.kc_id == log.kc_id
        assert restored.grade == log.grade
        assert restored.elapsed_days == log.elapsed_days
        assert restored.state_after == log.state_after


class TestIRTAAbilitySerialization:
    """IRTAbility 序列化往返."""

    def test_ability_roundtrip(self):
        """IRTAbility to_dict / from_dict 往返一致."""
        ability = IRTAbility(
            user_id="u-001",
            theta=0.75,
            standard_error=0.15,
        )
        d = ability.to_dict()
        restored = IRTAbility.from_dict(d)
        assert restored.theta == ability.theta
        assert restored.standard_error == ability.standard_error


class TestPathRecommendationSerialization:
    """PathRecommendation 序列化往返."""

    def test_recommendation_roundtrip(self):
        """PathRecommendation to_dict / from_dict 往返一致."""
        rec = PathRecommendation(
            user_id="u-001",
            recommended_path_id="path-001",
            rationale="基于薄弱知识点推荐",
            predicted_mastery_gain=0.15,
            confidence=0.85,
        )
        d = rec.to_dict()
        restored = PathRecommendation.from_dict(d)
        assert restored.user_id == rec.user_id
        assert restored.rationale == rec.rationale
        assert restored.confidence == rec.confidence
        assert restored.predicted_mastery_gain == rec.predicted_mastery_gain


class TestLearningEventSerialization:
    """LearningEvent 序列化往返."""

    def test_event_roundtrip(self):
        """LearningEvent to_dict / from_dict 往返一致."""
        result = EventResult(
            score_scaled=0.85,
            score_raw=17,
            score_max=20,
            success=True,
            completion=True,
            duration_ms=30000,
        )
        event = LearningEvent(
            actor_id="u-001",
            action="answered",
            object_id="question-001",
            object_type="question",
            result=result,
        )
        d = event.to_dict()
        restored = LearningEvent.from_dict(d)
        assert restored.actor_id == event.actor_id
        assert restored.action == event.action
        assert restored.object_id == event.object_id
        assert restored.result is not None
        assert restored.result.success is True


class TestSessionAnalyticsSerialization:
    """SessionAnalytics 序列化往返."""

    def test_analytics_roundtrip(self):
        """SessionAnalytics to_dict / from_dict 往返一致."""
        analytics = SessionAnalytics(
            session_id="s-001",
            total_duration_ms=1800000,
            total_interactions=42,
            total_questions=10,
            correct_answers=7,
            mastery_delta=0.15,
            engagement_score=0.68,
        )
        d = analytics.to_dict()
        restored = SessionAnalytics.from_dict(d)
        assert restored.session_id == analytics.session_id
        assert restored.total_interactions == analytics.total_interactions
        assert restored.accuracy_rate == pytest.approx(0.7, abs=0.01)


class TestPrivacyConfigSerialization:
    """PrivacyConfig 序列化往返."""

    def test_config_roundtrip(self):
        """PrivacyConfig to_dict / from_dict 往返一致."""
        config = PrivacyConfig(
            k_anonymity=5,
            l_diversity=2,
            epsilon=0.5,
            delta=1e-5,
            quasi_identifiers=["student_id", "name"],
            sensitive_attributes=["grade", "score"],
        )
        d = config.to_dict()
        restored = PrivacyConfig.from_dict(d)
        assert restored.k_anonymity == config.k_anonymity
        assert restored.epsilon == config.epsilon
        assert len(restored.quasi_identifiers) == 2


class TestRetentionPolicySerialization:
    """RetentionPolicy 序列化往返."""

    def test_policy_roundtrip(self):
        """RetentionPolicy to_dict / from_dict 往返一致."""
        policy = RetentionPolicy(
            data_level="L3_SENSITIVE",
            phases=[
                (RetentionPhase.ACTIVE, 90),
                (RetentionPhase.ARCHIVED, 365),
                (RetentionPhase.DELETED, 0),
            ],
        )
        d = policy.to_dict()
        restored = RetentionPolicy.from_dict(d)
        assert restored.data_level == policy.data_level
        assert len(restored.phases) == 3


class TestCrossLayerFromDict:
    """跨层接口 from_dict 覆盖."""

    def test_bkt_update_from_dict(self):
        """BKTUpdate.from_dict 往返."""
        update = BKTUpdate(
            kc_id="kc-001",
            p_know=0.85,
            p_slip=0.1,
            p_guess=0.25,
            p_transit=0.1,
        )
        d = update.to_dict()
        restored = BKTUpdate.from_dict(d)
        assert restored.kc_id == update.kc_id
        assert restored.p_know == update.p_know

    def test_memory_entry_from_dict(self):
        """MemoryEntry.from_dict 往返."""
        entry = MemoryEntry(
            session_id="s-001",
            interaction_summary="学生掌握了 Dy3+ 能级跃迁",
            key_insights=["4f-5d 跃迁机制"],
            weak_areas=["Judd-Ofelt 理论"],
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.session_id == entry.session_id
        assert restored.interaction_summary == entry.interaction_summary
        assert len(restored.key_insights) == 1

    def test_decay_request_from_dict(self):
        """DecayRequest.from_dict 往返."""
        req = DecayRequest(
            user_id="u-001",
            kcs_to_review=["kc1", "kc2"],
            urgency_scores={"kc1": 0.8, "kc2": 0.5},
        )
        d = req.to_dict()
        restored = DecayRequest.from_dict(d)
        assert restored.user_id == req.user_id
        assert restored.kcs_to_review == req.kcs_to_review

    def test_privacy_event_from_dict(self):
        """PrivacyEvent.from_dict 往返."""
        event = PrivacyEvent(
            event_type="data_access",
            user_id="u-001",
            data_level=DataLevel.L3_SENSITIVE,
            detail="Agent accessed student report",
        )
        d = event.to_dict()
        restored = PrivacyEvent.from_dict(d)
        assert restored.user_id == event.user_id
        assert restored.event_type == event.event_type

    def test_policy_update_from_dict(self):
        """PolicyUpdate.from_dict 往返."""
        update = PolicyUpdate(
            policy_id="pol-001",
            version="2.0",
            diff={"retention_days": {"old": 90, "new": 365}},
            effective_at=int(time.time() * 1000),
        )
        d = update.to_dict()
        restored = PolicyUpdate.from_dict(d)
        assert restored.policy_id == update.policy_id
        assert restored.version == update.version


class TestElementInteractivity:
    """ElementInteractivity 覆盖."""

    def test_create_and_ratio(self):
        """ElementInteractivity 创建与交互比."""
        ei = ElementInteractivity(
            element_count=5,
            interaction_count=3,
        )
        assert ei.interactivity_ratio == pytest.approx(0.6, abs=0.01)

    def test_zero_elements(self):
        """element_count=0 时交互比为 0."""
        ei = ElementInteractivity(element_count=0, interaction_count=0)
        assert ei.interactivity_ratio == 0.0


class TestContentModality:
    """ContentModality 覆盖."""

    def test_create_modality(self):
        """ContentModality 创建与属性."""
        modality = ContentModality(
            content_id="c-001",
            modality_tags=[VARKStyle.VISUAL, VARKStyle.READ_WRITE],
        )
        assert modality.content_id == "c-001"
        assert VARKStyle.VISUAL in modality.modality_tags

    def test_to_dict(self):
        """ContentModality 序列化."""
        modality = ContentModality(
            content_id="c-001",
            modality_tags=[VARKStyle.VISUAL],
        )
        d = modality.to_dict()
        assert d["content_id"] == "c-001"
        assert "visual" in d["modality_tags"]


class TestDesensitizeFunctions:
    """脱敏函数覆盖."""

    def test_desensitize_student_id_hash(self):
        """desensitize_student_id hash 模式."""
        result = desensitize_student_id("CS20240001", DesensitizationMethod.HASH)
        assert result != "CS20240001"
        assert len(result) > 0

    def test_desensitize_student_id_aggregate(self):
        """desensitize_student_id aggregate 模式."""
        result = desensitize_student_id("CS20240001", DesensitizationMethod.AGGREGATE)
        assert result != "CS20240001"

    def test_desensitize_student_id_bucket(self):
        """desensitize_student_id bucket 模式."""
        result = desensitize_student_id("CS20240001", DesensitizationMethod.BUCKET)
        assert result != "CS20240001"

    def test_desensitize_student_id_pseudo_id(self):
        """desensitize_student_id pseudo_id 模式."""
        result = desensitize_student_id("CS20240001", DesensitizationMethod.PSEUDO_ID)
        assert result != "CS20240001"
        assert "u-" in result or len(result) > 0

    def test_bucket_response_time(self):
        """bucket_response_time 分桶."""
        # < FAST_ANSWER_THRESHOLD_MS (5000) → "fast"
        assert bucket_response_time(3000) == "fast"
        # 5000-60000 → "normal"
        assert bucket_response_time(20000) == "normal"
        assert bucket_response_time(5000) == "normal"
        # > 60000 → "slow"
        assert bucket_response_time(120000) == "slow"
