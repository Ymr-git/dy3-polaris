"""R-07A private teaching-process event tests."""
from __future__ import annotations

import json

from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.interaction_recorder import InteractionRecorder
from dy3_polaris.l5.knowledge_learning_fusion import KnowledgeLearningContext
from dy3_polaris.l5.learning_event import (
    TeachingActionType,
    TeachingLearningEvent,
)
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_learner_memory import _ProfileService, _run_task
from tests.l5.test_private_runtime_carrier import (
    _review_candidate,
    _selected_generation,
)


_QUERY = "4f-4f跃迁如何产生Dy³⁺可见发射？"
_ANSWER = "Dy³⁺可见发射来自经证据审核的4f能级跃迁。"


def _event(monkeypatch):
    service = _ProfileService()
    result, diagnosis_inputs, _generation_inputs = _run_task(
        monkeypatch,
        service=service,
        learner_id="learner-r07a",
        task_id="task-r07a",
        query=_QUERY,
        answer=_ANSWER,
    )
    event = result._contract_candidate.teaching_learning_event
    assert isinstance(event, TeachingLearningEvent)
    return event, result, diagnosis_inputs[0], service


def test_complete_query_generates_one_private_teaching_learning_event(monkeypatch) -> None:
    event, result, _diagnosis_input, _service = _event(monkeypatch)

    assert event.event_id.startswith("teaching-event-")
    assert event.task_id == "task-r07a"
    assert event.learner_id == "learner-r07a"
    assert event.timestamp > 0
    assert event not in result.values()
    assert "teaching_learning_event" not in result


def test_event_contains_task_and_concept_context_without_second_learner_state(
    monkeypatch,
) -> None:
    event, _result, diagnosis_input, _service = _event(monkeypatch)
    view = diagnosis_input["_learner_intelligence_view"]
    fused = view.value("derived_context", "knowledge_learning_context")

    assert isinstance(fused, KnowledgeLearningContext)
    assert event.task_mode == "EXPLAIN"
    assert event.learning_goal
    assert "concept:dy3:four-f-four-f-transition" in event.related_concepts
    assert event.knowledge_context.source == "R06D KnowledgeLearningContext"
    assert event.knowledge_context.relation_refs
    assert event.before_state.mastery_projection is fused.concept_mastery
    assert event.before_state.learning_gap == fused.learning_path.prerequisite_gap
    assert event.before_state.source_ref.endswith("@request")


def test_event_records_teaching_actions_as_delivered_or_planned_facts(monkeypatch) -> None:
    event, _result, _diagnosis_input, _service = _event(monkeypatch)
    actions = {item.action_type: item for item in event.teaching_process}

    assert actions[TeachingActionType.EXPLANATION].status == "delivered"
    assert actions[TeachingActionType.CONCEPT_INTRODUCTION].status == "delivered"
    assert actions[TeachingActionType.PREREQUISITE_REPAIR].status == "planned"
    assert actions[TeachingActionType.PREREQUISITE_REPAIR].concept_refs
    assert event.outcome.confidence_change is None
    assert event.outcome.next_learning_target


def test_event_contains_bounded_reviewer_and_guidance_results(monkeypatch) -> None:
    event, _result, _diagnosis_input, _service = _event(monkeypatch)
    agent_ids = {item.agent_id for item in event.agent_contributions}

    assert event.review.producer == "agent.quality.review/run_review"
    assert event.review.status == "completed"
    assert event.review.verdict == "approved"
    assert event.review.real_reviewer_executed is True
    assert event.guidance.decision_type
    assert event.guidance.next_action
    assert agent_ids == {
        agent_workers.DIAGNOSIS_AGENT_ID,
        agent_workers.GENERATION_AGENT_ID,
        agent_workers.REVIEW_AGENT_ID,
        agent_workers.GUIDANCE_AGENT_ID,
    }
    assert all(item.role_summary for item in event.agent_contributions)


def test_event_never_enters_public_mapping_or_task_events_and_only_bounded_memory_is_persisted(
    monkeypatch,
) -> None:
    event, result, _diagnosis_input, service = _event(monkeypatch)
    public = json.dumps(dict(result), ensure_ascii=False, default=str)
    task_events = json.dumps(result["task_events"], ensure_ascii=False, default=str)
    profile = service.get_profile_snapshot("learner-r07a")
    persisted = json.dumps(profile.extras, ensure_ascii=False, default=str)

    for serialized in (public, task_events):
        assert event.event_id not in serialized
        assert "TeachingLearningEvent" not in serialized
        assert "teaching_learning_event" not in serialized
    # R-07B may persist the event identity as a source reference, but never the
    # R-07A object, raw answer, or private request state.
    assert event.event_id in persisted
    assert "TeachingLearningEvent" not in persisted
    assert "teaching_learning_event" not in persisted
    assert _ANSWER not in persisted
    assert {item["event_type"] for item in result["task_events"]} <= {
        "TaskCreated", "StateChanged", "RetrievalCompleted",
        "AgentStarted", "AgentContributionRecorded", "AgentFinished",
        "ReviewCompleted", "ReleaseDecided", "ResourceIssued",
        "TaskCompleted", "ReviewerChallengeRaised", "RevisionApplied",
    }
    private_repr = repr(event)
    assert _ANSWER not in private_repr
    assert "prompt" not in private_repr.lower()
    assert "chain_of_thought" not in private_repr.lower()


def test_event_generation_does_not_update_bkt_irt_or_mastery(monkeypatch) -> None:
    class _ReadOnlyIRT:
        def __init__(self) -> None:
            self.write_count = 0

        def get_ability_snapshot(self, learner_id: str):
            return {
                "learner_id": learner_id,
                "theta": 0.0,
                "se": 0.5,
                "response_count": 0,
                "last_update_time": 0.0,
            }

        def update(self, *_args, **_kwargs):
            self.write_count += 1

    class _ReadOnlyBKT:
        def __init__(self) -> None:
            self.write_count = 0

        def update(self, *_args, **_kwargs):
            self.write_count += 1

    service = _ProfileService()
    profile = service.ensure("learner-r07a-model-guard")
    profile.kp_mastery = {"2.1.1": 0.61}
    mastery_before = dict(profile.kp_mastery)
    level_before = profile.level
    irt = _ReadOnlyIRT()
    bkt = _ReadOnlyBKT()
    task_id = "task-r07a-model-guard"
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: _selected_generation(_ANSWER, task_id),
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda *_args, **_kwargs: _review_candidate(task_id, _ANSWER, "approved"),
    )
    monkeypatch.setattr(agent_workers, "get_recorder", lambda: InteractionRecorder())
    task_context = task_state_runtime.create_task_context(task_id)
    task_state_runtime.set_task_state(task_context, "UNDERSTANDING")

    result = agent_workers.run_guidance(
        {
            "task_id": task_id,
            "task_context": task_context,
            "query": _QUERY,
            "learner_id": "learner-r07a-model-guard",
        },
        AgentDependencies(
            profile_service=service,
            irt_service=irt,
            bkt_service=bkt,
        ),
    )

    assert isinstance(
        result._contract_candidate.teaching_learning_event,
        TeachingLearningEvent,
    )
    assert irt.write_count == 0
    assert bkt.write_count == 0
    assert profile.kp_mastery == mastery_before
    assert profile.level == level_before


def test_api_response_does_not_expose_teaching_event_or_private_context(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    response = client.post(
        "/api/query",
        json={"query": _QUERY, "learner_id": "learner-r07a-api"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    serialized = json.dumps(data, ensure_ascii=False, default=str)
    assert {
        "answer", "review", "evidence", "recommended_path",
        "task_id", "task_state", "task_events",
    }.issubset(data)
    for forbidden in (
        "TeachingLearningEvent",
        "teaching_learning_event",
        "BeforeTeachingState",
        "mastery_projection",
        "TeachingAction",
    ):
        assert forbidden not in serialized
