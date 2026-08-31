"""L1 核心数据模型序列化往返一致性测试 (to_dict → from_dict → 等价).

验证所有 dataclass 模型的 to_dict/from_dict 往返一致性,
确保序列化不会丢失数据或产生不一致。
"""

import pytest
import time


class TestSerializationRoundTrip:
    """所有 dataclass 模型的 to_dict → from_dict 往返一致性."""

    def test_bkt_params_roundtrip(self):
        from dy3_polaris.l1.models import BKTParams

        obj = BKTParams(p_know=0.7, p_slip=0.15, p_guess=0.2, p_transit=0.12)
        d = obj.to_dict()
        restored = BKTParams.from_dict(d)
        assert restored.p_know == obj.p_know
        assert restored.p_slip == obj.p_slip

    def test_role_roundtrip(self):
        from dy3_polaris.l1.models import Role, Permission

        role = Role(
            role_code="undergrad",
            role_name="本科生",
            base_permissions=[Permission.KB_PUBLIC_READ, Permission.AGENT_DIAGNOSIS],
        )
        d = role.to_dict()
        restored = Role.from_dict(d)
        assert restored.role_code == role.role_code
        assert restored.role_name == role.role_name
        assert len(restored.base_permissions) == 2

    def test_abac_attributes_roundtrip(self):
        from dy3_polaris.l1.models import ABACAttributes, GradeLevel, MajorDirection, LabAccessTier

        attr = ABACAttributes(
            grade_level=GradeLevel.SOPHOMORE,
            major_direction=MajorDirection.PHYSICS,
            lab_access_tier=LabAccessTier.TIER1,
            daily_agent_calls=5,
        )
        d = attr.to_dict()
        restored = ABACAttributes.from_dict(d)
        assert restored.grade_level == attr.grade_level
        assert restored.major_direction == attr.major_direction
        assert restored.daily_agent_calls == attr.daily_agent_calls

    def test_user_roundtrip(self):
        from dy3_polaris.l1.models import User, UserRole

        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        d = user.to_dict()
        restored = User.from_dict(d)
        assert restored.student_id == user.student_id
        assert restored.role == user.role

    def test_mastery_snapshot_roundtrip(self):
        from dy3_polaris.l1.models import MasterySnapshot

        snap = MasterySnapshot(
            kc_id="kc-001",
            p_know=0.85,
            last_practiced_at=int(time.time() * 1000),
            decay_factor=0.72,
            repetitions=5,
        )
        d = snap.to_dict()
        restored = MasterySnapshot.from_dict(d)
        assert restored.kc_id == snap.kc_id
        assert restored.p_know == snap.p_know
        assert restored.decay_factor == snap.decay_factor
        assert restored.repetitions == snap.repetitions

    def test_learning_goal_roundtrip(self):
        from dy3_polaris.l1.models import LearningGoal

        goal = LearningGoal(
            description="掌握二叉搜索树",
            priority=4,
            deadline=float(time.time() + 86400),
        )
        d = goal.to_dict()
        restored = LearningGoal.from_dict(d)
        assert restored.description == goal.description
        assert restored.priority == goal.priority

    def test_learning_state_roundtrip(self):
        from dy3_polaris.l1.models import LearningState, LearningPhase

        state = LearningState(
            phase=LearningPhase.PRACTICE,
            session_duration_ms=1800000,
            interaction_count=15,
            cognitive_load=0.6,
        )
        d = state.to_dict()
        restored = LearningState.from_dict(d)
        assert restored.phase == state.phase
        assert restored.interaction_count == state.interaction_count

    def test_resource_item_roundtrip(self):
        from dy3_polaris.l1.models import ResourceItem

        res = ResourceItem(
            resource_id="res-001",
            title="二叉搜索树教程",
            resource_type="video",
            difficulty=0.6,
        )
        d = res.to_dict()
        restored = ResourceItem.from_dict(d)
        assert restored.resource_id == res.resource_id
        assert restored.title == res.title

    def test_time_constraint_roundtrip(self):
        from dy3_polaris.l1.models import TimeConstraint, LearningPhase

        tc = TimeConstraint(
            available_minutes=60,
            recommended_phase=LearningPhase.REVIEW,
        )
        d = tc.to_dict()
        restored = TimeConstraint.from_dict(d)
        assert restored.available_minutes == tc.available_minutes

    def test_context_envelope_roundtrip(self):
        from dy3_polaris.l1.models import ContextEnvelope, MasterySnapshot

        env = ContextEnvelope(
            user_id="u-001",
            session_id="s-001",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc-1", p_know=0.7, last_practiced_at=int(time.time() * 1000)),
            ],
        )
        d = env.to_dict()
        restored = ContextEnvelope.from_dict(d)
        assert restored.user_id == env.user_id
        assert len(restored.mastery_snapshot) == 1
        assert restored.mastery_snapshot[0].kc_id == "kc-1"

    def test_learning_context_roundtrip(self):
        from dy3_polaris.l1.models import LearningContext

        ctx = LearningContext(session_id="s-001", user_id="u-001")
        d = ctx.to_dict()
        restored = LearningContext.from_dict(d)
        assert restored.user_id == ctx.user_id
        assert restored.session_id == ctx.session_id

    def test_agent_state_roundtrip(self):
        from dy3_polaris.l1.models import AgentState

        state = AgentState(
            agent_id="agent-001",
            agent_type="diagnosis",
            status="running",
            output_summary="正在分析...",
        )
        d = state.to_dict()
        restored = AgentState.from_dict(d)
        assert restored.agent_id == state.agent_id
        assert restored.status == state.status

    def test_interaction_roundtrip(self):
        from dy3_polaris.l1.models import Interaction

        inter = Interaction(
            interaction_type="qa",
            content="什么是二叉搜索树?",
            response="二叉搜索树是一种...",
        )
        d = inter.to_dict()
        restored = Interaction.from_dict(d)
        assert restored.interaction_type == inter.interaction_type
        assert restored.content == inter.content

    def test_session_artifact_roundtrip(self):
        from dy3_polaris.l1.models import SessionArtifact

        art = SessionArtifact(
            artifact_type="summary",
            title="会话总结",
            content="本次会话学习了二叉搜索树的基本概念",
        )
        d = art.to_dict()
        restored = SessionArtifact.from_dict(d)
        assert restored.artifact_type == art.artifact_type
        assert restored.title == art.title

    def test_learning_session_roundtrip(self):
        from dy3_polaris.l1.models import LearningSession, SessionType

        sess = LearningSession(
            user_id="u-001",
            session_type=SessionType.LEARNING,
        )
        d = sess.to_dict()
        restored = LearningSession.from_dict(d)
        assert restored.user_id == sess.user_id
        assert restored.session_type == sess.session_type

    def test_session_fork_roundtrip(self):
        from dy3_polaris.l1.models import SessionFork, ContextEnvelope

        fork = SessionFork(
            source_session_id="s-001",
            fork_point_seq=5,
            fork_reason="A/B测试",
            branch_label="路径A-先理论",
            snapshot_at_fork=ContextEnvelope(user_id="u-001", session_id="s-001"),
        )
        d = fork.to_dict()
        restored = SessionFork.from_dict(d)
        assert restored.source_session_id == fork.source_session_id
        assert restored.fork_point_seq == fork.fork_point_seq

    def test_session_checkpoint_roundtrip(self):
        from dy3_polaris.l1.models import SessionCheckpoint

        cp = SessionCheckpoint(session_id="s-001", seq=1)
        d = cp.to_dict()
        restored = SessionCheckpoint.from_dict(d)
        assert restored.session_id == cp.session_id
        assert restored.seq == cp.seq

    def test_audit_log_entry_roundtrip(self):
        from dy3_polaris.l1.models import (
            AuditLogEntry, DataLevel, AuditAction, AuditResult, UserRole,
        )

        entry = AuditLogEntry(
            actor_id="u-001",
            actor_role=UserRole.UNDERGRAD,
            action=AuditAction.VIEW,
            target_resource="kb:dy3_energy",
            target_data_level=DataLevel.L2_INTERNAL,
            purpose="学习查阅",
            result=AuditResult.SUCCESS,
        )
        d = entry.to_dict()
        restored = AuditLogEntry.from_dict(d)
        assert restored.actor_id == entry.actor_id
        assert restored.action == entry.action

    def test_approval_request_roundtrip(self):
        from dy3_polaris.l1.models import ApprovalRequest, HiTLType, HiTLPriority

        req = ApprovalRequest(
            user_id="u-001",
            session_id="s-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="需要确认的内容",
            priority=HiTLPriority.P1,
        )
        d = req.to_dict()
        restored = ApprovalRequest.from_dict(d)
        assert restored.user_id == req.user_id
        assert restored.hitl_type == req.hitl_type

    def test_approval_response_roundtrip(self):
        from dy3_polaris.l1.models import ApprovalResponse, ApprovalDecision

        resp = ApprovalResponse(
            request_id="req-001",
            responder_id="teacher-001",
            decision=ApprovalDecision.APPROVE,
            comment="内容审核通过",
        )
        d = resp.to_dict()
        restored = ApprovalResponse.from_dict(d)
        assert restored.request_id == resp.request_id
        assert restored.decision == resp.decision

    def test_feedback_report_roundtrip(self):
        from dy3_polaris.l1.models import FeedbackReport, FeedbackType

        report = FeedbackReport(
            user_id="u-001",
            session_id="s-001",
            feedback_type=FeedbackType.UNDERSTOOD,
            content="内容讲解很清晰",
        )
        d = report.to_dict()
        restored = FeedbackReport.from_dict(d)
        assert restored.user_id == report.user_id
        assert restored.feedback_type == report.feedback_type

    def test_emergency_alert_roundtrip(self):
        from dy3_polaris.l1.models import EmergencyAlert, AlertType

        alert = EmergencyAlert(
            session_id="s-001",
            user_id="u-001",
            trigger_reason="认知负荷过高",
            trigger_value=0.97,
            alert_type=AlertType.HIGH_COGNITIVE_LOAD,
        )
        d = alert.to_dict()
        restored = EmergencyAlert.from_dict(d)
        assert restored.session_id == alert.session_id
        assert restored.alert_type == alert.alert_type

    def test_provenance_record_roundtrip(self):
        from dy3_polaris.l1.models import ProvenanceRecord

        prov = ProvenanceRecord(
            artifact_id="art-001",
            actor_chain=["agent-001", "agent-002"],
            code_hash="sha256:abc123",
            env_hash="sha256:def456",
        )
        d = prov.to_dict()
        restored = ProvenanceRecord.from_dict(d)
        assert restored.artifact_id == prov.artifact_id
        assert restored.actor_chain == prov.actor_chain

    # --- FSRS 模型 ---

    def test_fsrs_parameters_roundtrip(self):
        from dy3_polaris.l1.models import FSRSParameters

        params = FSRSParameters()
        d = params.to_dict()
        restored = FSRSParameters.from_dict(d)
        assert restored.weights == params.weights

    def test_fsrs_card_state_roundtrip(self):
        from dy3_polaris.l1.models import FSRSCardState

        card = FSRSCardState(kc_id="kc-001", stability=5.0, difficulty=3.5, state="review")
        d = card.to_dict()
        restored = FSRSCardState.from_dict(d)
        assert restored.kc_id == card.kc_id
        assert restored.stability == card.stability

    def test_fsrs_review_log_roundtrip(self):
        from dy3_polaris.l1.models import FSRSReviewLog

        log = FSRSReviewLog(kc_id="kc-001", grade=3, elapsed_days=2.5)
        d = log.to_dict()
        restored = FSRSReviewLog.from_dict(d)
        assert restored.kc_id == log.kc_id
        assert restored.grade == log.grade

    # --- IRT 模型 ---

    def test_irt_item_roundtrip(self):
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(item_id="q-001", model_type=IRTModel.TWO_PL, difficulty_b=0.5, discrimination_a=1.2)
        d = item.to_dict()
        restored = IRTItem.from_dict(d)
        assert restored.item_id == item.item_id
        assert restored.model_type == item.model_type

    def test_irt_ability_roundtrip(self):
        from dy3_polaris.l1.models import IRTAbility

        ability = IRTAbility(user_id="u-001", theta=0.8, standard_error=0.25)
        d = ability.to_dict()
        restored = IRTAbility.from_dict(d)
        assert restored.user_id == ability.user_id
        assert restored.theta == ability.theta

    # --- VARK 模型 ---

    def test_vark_profile_roundtrip(self):
        from dy3_polaris.l1.models import VARKProfile

        profile = VARKProfile(
            user_id="u-001", visual_score=0.8, aural_score=0.3,
            read_write_score=0.5, kinesthetic_score=0.4, confidence=0.85,
        )
        d = profile.to_dict()
        restored = VARKProfile.from_dict(d)
        assert restored.user_id == profile.user_id
        assert restored.visual_score == profile.visual_score

    def test_content_modality_roundtrip(self):
        from dy3_polaris.l1.models import ContentModality, VARKStyle

        modality = ContentModality(
            content_id="res-001",
            modality_tags=[VARKStyle.VISUAL, VARKStyle.AURAL],
        )
        d = modality.to_dict()
        restored = ContentModality.from_dict(d)
        assert restored.content_id == modality.content_id
        assert len(restored.modality_tags) == 2

    # --- 认知负荷模型 ---

    def test_cognitive_load_breakdown_roundtrip(self):
        from dy3_polaris.l1.models import CognitiveLoadBreakdown

        breakdown = CognitiveLoadBreakdown(intrinsic_load=0.4, extraneous_load=0.2, germane_load=0.3)
        d = breakdown.to_dict()
        restored = CognitiveLoadBreakdown.from_dict(d)
        assert restored.intrinsic_load == breakdown.intrinsic_load

    def test_element_interactivity_roundtrip(self):
        from dy3_polaris.l1.models import ElementInteractivity

        ei = ElementInteractivity(element_count=5, interaction_count=10)
        d = ei.to_dict()
        restored = ElementInteractivity.from_dict(d)
        assert restored.element_count == ei.element_count
        assert restored.interactivity_ratio == pytest.approx(ei.interactivity_ratio)

    # --- Bloom 2D ---

    def test_bloom_tag_roundtrip(self):
        from dy3_polaris.l1.models import BloomTag, KnowledgeType
        from dy3_polaris.l3.api_models import BloomLevel

        tag = BloomTag(cognitive_level=BloomLevel.APPLY, knowledge_type=KnowledgeType.PROCEDURAL)
        d = tag.to_dict()
        restored = BloomTag.from_dict(d)
        assert restored.knowledge_type == tag.knowledge_type
        assert restored.matrix_cell() == tag.matrix_cell()

    # --- 跨层接口模型 ---

    def test_bkt_update_roundtrip(self):
        from dy3_polaris.l1.models import BKTUpdate

        update = BKTUpdate(kc_id="kc-1", p_know=0.8)
        d = update.to_dict()
        restored = BKTUpdate.from_dict(d)
        assert restored.kc_id == update.kc_id
        assert restored.p_know == update.p_know

    def test_memory_entry_roundtrip(self):
        from dy3_polaris.l1.models import MemoryEntry

        entry = MemoryEntry(session_id="s-001", interaction_summary="学生学习了BST")
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.session_id == entry.session_id

    def test_decay_request_roundtrip(self):
        from dy3_polaris.l1.models import DecayRequest

        req = DecayRequest(user_id="u-001", kcs_to_review=["kc-1", "kc-2"])
        d = req.to_dict()
        restored = DecayRequest.from_dict(d)
        assert restored.user_id == req.user_id
        assert restored.kcs_to_review == req.kcs_to_review

    def test_access_check_roundtrip(self):
        from dy3_polaris.l1.models import AccessCheck

        check = AccessCheck(user_id="u-001", resource_id="res-001", access_level="read")
        d = check.to_dict()
        restored = AccessCheck.from_dict(d)
        assert restored.user_id == check.user_id

    def test_resource_request_roundtrip(self):
        from dy3_polaris.l1.models import ResourceRequest

        req = ResourceRequest(weak_kcs=["kc-1"], count_limit=5)
        d = req.to_dict()
        restored = ResourceRequest.from_dict(d)
        assert restored.weak_kcs == req.weak_kcs

    def test_knowledge_result_roundtrip(self):
        from dy3_polaris.l1.models import KnowledgeResult, ResourceItem

        result = KnowledgeResult(
            resources=[ResourceItem(resource_id="r1", title="T", resource_type="video")],
            confidence_scores={"r1": 0.9},
        )
        d = result.to_dict()
        restored = KnowledgeResult.from_dict(d)
        assert len(restored.resources) == 1

    def test_privacy_event_roundtrip(self):
        from dy3_polaris.l1.models import PrivacyEvent, DataLevel

        event = PrivacyEvent(
            event_type="data_access", user_id="u-001",
            data_level=DataLevel.L3_SENSITIVE, detail="导出",
        )
        d = event.to_dict()
        restored = PrivacyEvent.from_dict(d)
        assert restored.event_type == event.event_type

    def test_policy_update_roundtrip(self):
        from dy3_polaris.l1.models import PolicyUpdate

        update = PolicyUpdate(policy_id="pol-001", version="2.0", diff={"k": "v"})
        d = update.to_dict()
        restored = PolicyUpdate.from_dict(d)
        assert restored.policy_id == update.policy_id

    # --- 隐私保护模型 ---

    def test_privacy_config_roundtrip(self):
        from dy3_polaris.l1.models import PrivacyConfig

        config = PrivacyConfig(k_anonymity=10, epsilon=0.5, quasi_identifiers=["age"])
        d = config.to_dict()
        restored = PrivacyConfig.from_dict(d)
        assert restored.k_anonymity == config.k_anonymity

    def test_retention_policy_roundtrip(self):
        from dy3_polaris.l1.models import RetentionPolicy, RetentionPhase

        policy = RetentionPolicy(
            data_level="L3",
            phases=[(RetentionPhase.ACTIVE, 90), (RetentionPhase.DELETED, 0)],
        )
        d = policy.to_dict()
        restored = RetentionPolicy.from_dict(d)
        assert restored.data_level == policy.data_level
        assert len(restored.phases) == 2

    # --- 学习分析事件 ---

    def test_event_result_roundtrip(self):
        from dy3_polaris.l1.models import EventResult

        result = EventResult(score_scaled=0.85, success=True, completion=True)
        d = result.to_dict()
        restored = EventResult.from_dict(d)
        assert restored.score_scaled == result.score_scaled

    def test_learning_event_roundtrip(self):
        from dy3_polaris.l1.models import LearningEvent, EventResult

        event = LearningEvent(
            actor_id="u-001", action="answered", object_id="q-001",
            result=EventResult(score_scaled=0.9, success=True),
        )
        d = event.to_dict()
        restored = LearningEvent.from_dict(d)
        assert restored.actor_id == event.actor_id
        assert restored.result is not None
        assert restored.result.score_scaled == 0.9

    # --- 参与度模型 ---

    def test_engagement_metrics_roundtrip(self):
        from dy3_polaris.l1.models import EngagementMetrics

        metrics = EngagementMetrics(
            session_duration_ms=1800000, login_frequency=5, completion_rate=0.8,
        )
        d = metrics.to_dict()
        restored = EngagementMetrics.from_dict(d)
        assert restored.session_duration_ms == metrics.session_duration_ms

    def test_session_analytics_roundtrip(self):
        from dy3_polaris.l1.models import SessionAnalytics

        analytics = SessionAnalytics(session_id="s-001", total_questions=10, correct_answers=7)
        d = analytics.to_dict()
        restored = SessionAnalytics.from_dict(d)
        assert restored.session_id == analytics.session_id

    # --- 学习路径模型 ---

    def test_path_node_roundtrip(self):
        from dy3_polaris.l1.models import PathNode

        node = PathNode(kc_id="kc-1", order=1, estimated_difficulty=0.6, prerequisite_kcs=["kc-0"])
        d = node.to_dict()
        restored = PathNode.from_dict(d)
        assert restored.kc_id == node.kc_id
        assert restored.prerequisite_kcs == node.prerequisite_kcs

    def test_learning_path_roundtrip(self):
        from dy3_polaris.l1.models import LearningPath, PathNode

        path = LearningPath(
            user_id="u-001",
            nodes=[PathNode(kc_id="kc-1", order=1), PathNode(kc_id="kc-2", order=2)],
        )
        d = path.to_dict()
        restored = LearningPath.from_dict(d)
        assert restored.user_id == path.user_id
        assert len(restored.nodes) == 2

    def test_path_recommendation_roundtrip(self):
        from dy3_polaris.l1.models import PathRecommendation

        rec = PathRecommendation(
            user_id="u-001", recommended_path_id="path-001",
            rationale="基于薄弱知识点", confidence=0.85,
        )
        d = rec.to_dict()
        restored = PathRecommendation.from_dict(d)
        assert restored.user_id == rec.user_id
        assert restored.confidence == rec.confidence

    # --- 增强模型 ---

    def test_zpd_roundtrip(self):
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        zpd = ZoneOfProximalDevelopment(learner_theta=0.8, delta_lower=0.6, delta_upper=0.4)
        d = zpd.to_dict()
        restored = ZoneOfProximalDevelopment.from_dict(d)
        assert restored.learner_theta == zpd.learner_theta
        assert restored.zpd_lower == zpd.zpd_lower
        assert restored.zpd_upper == zpd.zpd_upper

    def test_knowledge_component_roundtrip(self):
        from dy3_polaris.l1.models import KnowledgeComponent

        kc = KnowledgeComponent(
            kc_id="kc-001", name="二叉搜索树", estimated_difficulty=0.6,
            prerequisite_kcs=["kc-000"], estimated_time_minutes=30,
        )
        d = kc.to_dict()
        restored = KnowledgeComponent.from_dict(d)
        assert restored.kc_id == kc.kc_id
        assert restored.name == kc.name

    def test_mastery_trajectory_point_roundtrip(self):
        from dy3_polaris.l1.models import MasteryTrajectoryPoint

        point = MasteryTrajectoryPoint(
            kc_id="kc-1", timestamp=1000, p_know=0.7, decay_factor=0.8
        )
        d = point.to_dict()
        restored = MasteryTrajectoryPoint.from_dict(d)
        assert restored.kc_id == point.kc_id
        assert restored.p_know == point.p_know

    def test_mastery_trajectory_roundtrip(self):
        from dy3_polaris.l1.models import MasteryTrajectory, MasteryTrajectoryPoint

        traj = MasteryTrajectory(kc_id="kc-1")
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-1", timestamp=1000, p_know=0.3))
        traj.add_point(MasteryTrajectoryPoint(kc_id="kc-1", timestamp=2000, p_know=0.6))
        d = traj.to_dict()
        restored = MasteryTrajectory.from_dict(d)
        assert restored.kc_id == traj.kc_id
        assert len(restored.points) == 2

    def test_study_block_roundtrip(self):
        from dy3_polaris.l1.models import StudyBlock, LearningPhase

        block = StudyBlock(
            kc_id="kc-1", start_time=1000000, duration_minutes=30,
            phase=LearningPhase.REVIEW,
        )
        d = block.to_dict()
        restored = StudyBlock.from_dict(d)
        assert restored.kc_id == block.kc_id
        assert restored.duration_minutes == block.duration_minutes

    def test_study_plan_roundtrip(self):
        from dy3_polaris.l1.models import StudyPlan, StudyBlock

        plan = StudyPlan(
            user_id="u-001",
            blocks=[StudyBlock(kc_id="kc-1", start_time=1000, duration_minutes=30)],
        )
        d = plan.to_dict()
        restored = StudyPlan.from_dict(d)
        assert restored.user_id == plan.user_id
        assert restored.block_count() == 1

    def test_learning_efficiency_roundtrip(self):
        from dy3_polaris.l1.models import LearningEfficiency

        eff = LearningEfficiency(
            mastery_gain=0.35, time_spent_ms=3_600_000, interactions=15,
            kc_id="kc-001", session_id="s-001",
        )
        d = eff.to_dict()
        restored = LearningEfficiency.from_dict(d)
        assert restored.mastery_gain == eff.mastery_gain
        assert restored.time_spent_ms == eff.time_spent_ms


class TestMissingFromDict:
    """验证 3 个缺失 from_dict 的模型在补全后能正确往返."""

    def test_content_modality_from_dict(self):
        from dy3_polaris.l1.models import ContentModality, VARKStyle

        d = {"content_id": "res-001", "modality_tags": ["visual", "aural"]}
        modality = ContentModality.from_dict(d)
        assert modality.content_id == "res-001"
        assert VARKStyle.VISUAL in modality.modality_tags

    def test_content_modality_roundtrip_preserves_tags(self):
        from dy3_polaris.l1.models import ContentModality, VARKStyle

        original = ContentModality(
            content_id="res-002",
            modality_tags=[VARKStyle.VISUAL, VARKStyle.READ_WRITE, VARKStyle.KINESTHETIC],
        )
        d = original.to_dict()
        restored = ContentModality.from_dict(d)
        assert len(restored.modality_tags) == 3
        for tag in original.modality_tags:
            assert tag in restored.modality_tags

    def test_element_interactivity_from_dict(self):
        from dy3_polaris.l1.models import ElementInteractivity

        d = {"element_count": 5, "interaction_count": 10}
        ei = ElementInteractivity.from_dict(d)
        assert ei.element_count == 5
        assert ei.interaction_count == 10

    def test_element_interactivity_roundtrip_preserves_ratio(self):
        from dy3_polaris.l1.models import ElementInteractivity

        original = ElementInteractivity(element_count=4, interaction_count=8)
        d = original.to_dict()
        restored = ElementInteractivity.from_dict(d)
        assert restored.interactivity_ratio == pytest.approx(original.interactivity_ratio)

    def test_bloom_tag_from_dict(self):
        from dy3_polaris.l1.models import BloomTag, KnowledgeType

        d = {"cognitive_level": "apply", "knowledge_type": "procedural"}
        tag = BloomTag.from_dict(d)
        assert tag.knowledge_type == KnowledgeType.PROCEDURAL
        assert tag.matrix_cell() is not None

    def test_bloom_tag_roundtrip_preserves_matrix_cell(self):
        from dy3_polaris.l1.models import BloomTag, KnowledgeType
        from dy3_polaris.l3.api_models import BloomLevel

        original = BloomTag(
            cognitive_level=BloomLevel.ANALYZE,
            knowledge_type=KnowledgeType.CONCEPTUAL,
        )
        d = original.to_dict()
        restored = BloomTag.from_dict(d)
        assert restored.matrix_cell() == original.matrix_cell()
