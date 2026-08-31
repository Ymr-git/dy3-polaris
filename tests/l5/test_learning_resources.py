"""T1 truth-preserving learning-resource and feedback-loop tests."""

from __future__ import annotations

from copy import deepcopy

from starlette.testclient import TestClient

from dy3_polaris.l5.agent_contracts import (
    QualityReleaseDecision,
    QualityReleaseStatus,
)
from dy3_polaris.l5.knowledge_learning_fusion import (
    ConceptLearningPath,
    KnowledgeLearningContext,
)
from dy3_polaris.l5.learner_foundation import AdaptiveTeachingDecision
from dy3_polaris.l5.learning_resources import (
    ResourceFamily,
    ResourceInteractionAction,
    ResourceSourceType,
    build_learning_resource_plan,
    build_resource_interaction_event,
    public_resource_projection,
)
from dy3_polaris.l5.teaching_memory import (
    commit_resource_interaction,
    interpret_teaching_memory,
    load_teaching_memory_view,
)
from dy3_polaris.l5 import task_state
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_learner_memory import _ProfileService


def _decision(*, research: bool = False) -> AdaptiveTeachingDecision:
    return AdaptiveTeachingDecision(
        content_depth="research" if research else "foundation",
        explanation_strategy=(
            "evidence_first_mechanism" if research else "scaffolded_mechanism"
        ),
        representation_modes=("concept_relation", "evidence_summary"),
        difficulty_strategy="increase" if research else "maintain",
        resource_modes=("research_task",) if research else ("concept_card",),
        next_focus="concept:concentration-quenching",
        diagnostic_needed=False,
        rationale=("Diagnosis interpreted observed and inferred learner state.",),
        source_refs=("DiagnosisContribution", "R06:concept:concentration-quenching"),
        confidence=0.82,
    )


def _knowledge() -> KnowledgeLearningContext:
    return KnowledgeLearningContext(
        learner_id="learner-resource",
        learning_goal=("understand concentration quenching",),
        target_concepts=("concept:concentration-quenching",),
        concept_mastery={},
        active_misconception_concepts=(),
        evidence_available_concepts=("concept:concentration-quenching",),
        concept_to_kps={
            "concept:concentration-quenching": ("kp:D-01",),
        },
        concept_names={
            "concept:concentration-quenching": "浓度猝灭",
        },
        learning_path=ConceptLearningPath(
            current_position="concept:energy-transfer",
            next_concept="concept:concentration-quenching",
            reason="prerequisite relation and current learner model",
            prerequisite_gap=(),
            expected_outcome="explain the mechanism with evidence boundaries",
            confidence=0.78,
        ),
        trace=("R06D:derived",),
    )


def _release() -> QualityReleaseDecision:
    return QualityReleaseDecision(
        task_id="task-resource",
        status=QualityReleaseStatus.FULL_RELEASE,
        eligible=True,
        public_answer="Reviewed scientific explanation.",
        reason_codes=(),
        review_status="completed",
        review_verdict="approved",
        answer_identity="answer-id",
        evidence_versions=(1,),
        correction_count=0,
        message="approved",
    )


def test_resource_plan_has_three_real_families_and_truthful_sources() -> None:
    plan = build_learning_resource_plan(
        task_id="task-resource",
        learner_id="learner-resource",
        teaching_decision=_decision(research=True),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
    )

    assert tuple(item.resource_family for item in plan.resources) == (
        ResourceFamily.KNOWLEDGE,
        ResourceFamily.PRACTICAL,
        ResourceFamily.ASSESSMENT,
    )
    assert tuple(item.source_type for item in plan.resources) == (
        ResourceSourceType.DERIVED,
        ResourceSourceType.DERIVED,
        ResourceSourceType.RETRIEVED,
    )
    practical = plan.resources[1]
    assert practical.title == "当前任务证据分析工作单"
    assert practical.payload["task_binding"] == "task-resource"
    assert practical.review_status == "derived_from_released_task"
    assert practical.evidence_refs == ()
    assert all(step["source"] != "template" for step in practical.payload["steps"])
    assert practical.payload["parameter_status"] == "not_prescribed"
    assert "1400" not in repr(practical.payload)
    assert "7 mol%" not in repr(practical.payload)
    assessment = plan.resources[2]
    assert assessment.payload["question_source"] == "local_practice_bank"
    assert assessment.payload["stage_selection"] == "challenge"
    assert tuple(item["attempt_purpose"] for item in assessment.payload["stages"]) == (
        "DIAGNOSTIC", "REQUIRED_PRACTICE", "STAGED_ASSESSMENT",
    )
    assert assessment.completion_signal == "submitted_answer_record"
    knowledge = plan.resources[0]
    assert knowledge.resource_form == "guided_long_read"
    assert knowledge.payload["guided_document"]["document_mode"] == "GUIDED_LONG_READ"
    sections = knowledge.payload["guided_document"]["sections"]
    assert next(item for item in sections if item["section_id"] == "reviewed-explanation")["content"] == _release().public_answer
    assert ResourceInteractionAction.ASK_FOLLOW_UP in knowledge.interaction_actions
    assert knowledge.payload["guided_questions"]
    public = public_resource_projection(plan)
    assert all("source_type" in item and "provenance" in item for item in public)
    assert "_contract_candidate" not in repr(public)


def test_explicit_resource_feedback_changes_teaching_memory_not_mastery() -> None:
    service = _ProfileService()
    profile = service.ensure("learner-resource")
    profile.kp_mastery = {"kp:D-01": 0.42}
    before_mastery = deepcopy(profile.kp_mastery)
    resource = public_resource_projection(build_learning_resource_plan(
        task_id="task-resource",
        learner_id="learner-resource",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
    ))[0]
    event = build_resource_interaction_event(
        learner_id="learner-resource",
        task_id="task-resource",
        resource=resource,
        action=ResourceInteractionAction.STILL_CONFUSED.value,
    )

    assert commit_resource_interaction(service, event) is True
    assert profile.kp_mastery == before_mastery
    view = load_teaching_memory_view(service, "learner-resource")
    interpretation = interpret_teaching_memory(
        view,
        ("concept:concentration-quenching",),
    )
    assert interpretation.available is True
    assert interpretation.strategy == "repair_with_evidence"
    assert event.event_id in interpretation.source_event_ids
    raw = profile.extras["learner_memory"]["teaching_memory"]
    assert raw["resource_interactions"][-1]["source_class"] == "OBSERVED"


def test_guided_follow_up_is_observed_but_never_updates_mastery() -> None:
    service = _ProfileService()
    profile = service.ensure("learner-follow-up")
    profile.kp_mastery = {"kp:D-01": 0.36}
    before_mastery = deepcopy(profile.kp_mastery)
    resource = public_resource_projection(build_learning_resource_plan(
        task_id="task-follow-up",
        learner_id="learner-follow-up",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
    ))[0]
    event = build_resource_interaction_event(
        learner_id="learner-follow-up",
        task_id="task-follow-up",
        resource=resource,
        action=ResourceInteractionAction.ASK_FOLLOW_UP.value,
    )

    assert commit_resource_interaction(service, event) is True
    assert profile.kp_mastery == before_mastery
    view = load_teaching_memory_view(service, "learner-follow-up")
    assert view.strategies[-1].effect == "guided_follow_up_started"


def test_explicit_example_and_deepen_feedback_keep_distinct_strategy_meaning() -> None:
    service = _ProfileService()
    resource = public_resource_projection(build_learning_resource_plan(
        task_id="task-feedback-strategy",
        learner_id="learner-feedback-strategy",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
    ))[0]
    for action, expected in (
        (ResourceInteractionAction.REQUEST_EXAMPLE, "example_then_mechanism"),
        (ResourceInteractionAction.DEEPEN, "evidence_first_mechanism"),
    ):
        learner_id = f"learner-{action.value}"
        service.ensure(learner_id)
        event = build_resource_interaction_event(
            learner_id=learner_id,
            task_id="task-feedback-strategy",
            resource=resource,
            action=action.value,
        )
        assert commit_resource_interaction(service, event) is True
        interpretation = interpret_teaching_memory(
            load_teaching_memory_view(service, learner_id),
            ("concept:concentration-quenching",),
        )
        assert interpretation.strategy == expected


def test_withheld_review_issues_no_resources_or_practice() -> None:
    withheld = QualityReleaseDecision(
        task_id="task-withheld",
        status=QualityReleaseStatus.WITHHOLD,
        eligible=False,
        public_answer="",
        reason_codes=("review_not_approved",),
        review_status="completed",
        review_verdict="needs_review",
        answer_identity="",
        evidence_versions=(1,),
        correction_count=2,
        message="withheld",
    )
    plan = build_learning_resource_plan(
        task_id="task-withheld",
        learner_id="learner-resource",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=withheld,
    )

    assert plan.resources == ()
    assert public_resource_projection(plan) == []


def test_resource_interaction_api_accepts_only_server_issued_resource() -> None:
    service = _ProfileService()
    service.ensure("learner-resource")
    builder = UnifiedApp.create_full_app_builder()
    builder.bridge.profile_service = service
    resource = public_resource_projection(build_learning_resource_plan(
        task_id="task-resource-api",
        learner_id="learner-resource",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
    ))[0]
    task_context = task_state.create_task_context(
        "task-resource-api", learner_id="learner-resource",
    )
    builder._handlers._task_store.create_task(
        task_context=task_context,
        learner_id="learner-resource",
        session_id="resource-test",
        query="resource interaction test",
    )
    builder._handlers._task_store.update_task(
        "task-resource-api",
        "learner-resource",
        {
            "answer": {"identity": "answer-id", "version": 1, "text": "Reviewed scientific explanation."},
            "reviewer": {
                "challenges": [],
                "revisions": [],
                "review": {
                    "agent_id": "agent.quality.review",
                    "status": "completed",
                    "verdict": "approved",
                },
                "release": {
                    "status": "FULL_RELEASE",
                    "eligible": True,
                },
            },
            "resource_plan": {"resources": [resource]},
        },
    )
    client = TestClient(builder.create_app())

    accepted = client.post("/api/learning/resources/interact", json={
        "learner_id": "learner-resource",
        "task_id": "task-resource-api",
        "resource_id": resource["resource_id"],
        "action": "understood",
    })
    assert accepted.status_code == 200
    assert accepted.json()["data"]["mastery_updated"] is False
    rejected = client.post("/api/learning/resources/interact", json={
        "learner_id": "learner-resource",
        "task_id": "task-resource-api",
        "resource_id": "forged-resource",
        "action": "understood",
    })
    assert rejected.status_code == 404


def test_legacy_withheld_resource_is_not_public_or_actionable() -> None:
    builder = UnifiedApp.create_full_app_builder()
    context = task_state.create_task_context(
        "task-withheld-legacy", learner_id="learner-resource",
    )
    builder._handlers._task_store.create_task(
        task_context=context,
        learner_id="learner-resource",
        session_id="resource-test",
        query="withheld legacy resource",
    )
    resource = {"resource_id": "resource-withheld", "interaction_actions": ["open"]}
    builder._handlers._task_store.update_task(
        "task-withheld-legacy",
        "learner-resource",
        {
            "reviewer": {
                "review": {
                    "agent_id": "agent.quality.review",
                    "status": "completed",
                    "verdict": "needs_review",
                },
                "release": {"status": "WITHHOLD", "eligible": False},
            },
            "resource_plan": {"resources": [resource]},
            "public_result": {"learning_resources": [resource]},
        },
    )
    client = TestClient(builder.create_app())

    detail = client.get(
        "/api/learning-tasks/learner-resource/task-withheld-legacy"
    ).json()["data"]["task"]
    assert detail["resource_plan"]["resources"] == []
    assert detail["public_result"]["learning_resources"] == []
    interaction = client.post("/api/learning/resources/interact", json={
        "learner_id": "learner-resource",
        "task_id": "task-withheld-legacy",
        "resource_id": "resource-withheld",
        "action": "open",
    })
    assert interaction.status_code == 404
