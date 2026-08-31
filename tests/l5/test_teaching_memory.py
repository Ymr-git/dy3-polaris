"""R-07B Teaching Memory Intelligence runtime and boundary tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.learning_event import TeachingLearningEvent
from dy3_polaris.l5.teaching_memory import (
    ConceptLearningMemory,
    MisconceptionLifecycle,
    TeachingMemoryInterpretation,
    TeachingMemoryView,
    commit_teaching_memory,
    interpret_teaching_memory,
    load_teaching_memory_view,
)
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_learner_memory import _ProfileService, _run_task


_QUERY = "4f-4f跃迁如何产生Dy³⁺可见发射？"
_ANSWER = "Dy³⁺可见发射来自经证据审核的4f能级跃迁。"


def _run(
    monkeypatch,
    service: _ProfileService,
    learner_id: str,
    task_id: str,
):
    return _run_task(
        monkeypatch,
        service=service,
        learner_id=learner_id,
        task_id=task_id,
        query=_QUERY,
        answer=_ANSWER,
    )


def test_private_models_are_frozen_slotted_and_first_request_has_no_history(
    monkeypatch,
) -> None:
    service = _ProfileService()
    result, diagnosis_inputs, _generation_inputs = _run(
        monkeypatch, service, "learner-r07b-first", "task-r07b-first"
    )

    first_view = diagnosis_inputs[0]["_learner_intelligence_view"]
    assert first_view.metadata["teaching_memory_available"] is False
    event = result._contract_candidate.teaching_learning_event
    assert isinstance(event, TeachingLearningEvent)

    memory = load_teaching_memory_view(service, "learner-r07b-first")
    assert memory.available is True
    assert memory.concept_learning
    assert memory.strategies
    assert memory.experiences
    assert not hasattr(memory, "__dict__")
    with pytest.raises(FrozenInstanceError):
        memory.updated_at = 0.0  # type: ignore[misc]

    persisted = service.get_profile_snapshot(
        "learner-r07b-first"
    ).extras["learner_memory"]["teaching_memory"]
    serialized = json.dumps(persisted, ensure_ascii=False)
    assert _QUERY not in serialized
    assert _ANSWER not in serialized
    for forbidden in ("prompt", "chain_of_thought", "agent_contributions"):
        assert forbidden not in serialized.lower()


def test_second_related_task_routes_memory_through_learner_intelligence_and_diagnosis(
    monkeypatch,
) -> None:
    service = _ProfileService()
    first, first_diagnosis, first_generation = _run(
        monkeypatch, service, "learner-r07b-loop", "task-r07b-loop-1"
    )
    profile = service.get_profile_snapshot("learner-r07b-loop")
    teaching_memory = profile.extras["learner_memory"]["teaching_memory"]
    # Isolate R-07B from the older history projection: only Teaching Memory is
    # retained, so any second-request signal must come through its View.
    profile.extras["learner_memory"] = {"teaching_memory": teaching_memory}

    second, second_diagnosis, second_generation = _run(
        monkeypatch, service, "learner-r07b-loop", "task-r07b-loop-2"
    )
    first_view = first_diagnosis[0]["_learner_intelligence_view"]
    second_view = second_diagnosis[0]["_learner_intelligence_view"]
    teaching_context = second_view.value(
        "derived_context", "teaching_memory_context"
    )

    assert first_view.metadata["teaching_memory_available"] is False
    assert second_view.metadata["teaching_memory_available"] is True
    assert isinstance(teaching_context, TeachingMemoryInterpretation)
    assert teaching_context.available is True
    assert teaching_context.strategy == "build_on_prior_exposure"
    assert teaching_context.prior_attempts > 0
    assert second_view.value("derived_context", "adaptive_strategy") == (
        "build_on_prior_exposure"
    )
    assert second_generation[0]["_agent_input"].learner_context[
        "teaching_strategy"
    ] == "build_on_prior_exposure"
    assert "TeachingMemoryView" not in repr(
        second_generation[0]["_agent_input"].learner_context
    )
    assert first_generation[0]["_agent_input"].subtask.goal == (
        second_generation[0]["_agent_input"].subtask.goal
    )
    assert first["review"] == second["review"]


def test_source_backed_misconception_lifecycle_is_addressed_not_auto_resolved(
    monkeypatch,
) -> None:
    service = _ProfileService()
    result, _diagnosis_inputs, _generation_inputs = _run(
        monkeypatch, service, "learner-r07b-misconception", "task-r07b-misconception"
    )
    event = result._contract_candidate.teaching_learning_event
    concept_id = event.related_concepts[0]
    addressed_event = replace(
        event,
        before_state=replace(
            event.before_state,
            misconception_state=(concept_id,),
        ),
        outcome=replace(
            event.outcome,
            misconception_addressed=(concept_id,),
        ),
        event_id=f"{event.event_id}-misconception",
        task_id=f"{event.task_id}-misconception",
    )
    source = {
        "misconception_id": "misconception-source-backed",
        "status": "ACTIVE",
        "confidence": 0.8,
        "source_events": ["challenge-real-1"],
        "belief": "this text must not be copied",
    }

    assert commit_teaching_memory(
        service,
        "learner-r07b-misconception",
        addressed_event,
        source_misconceptions=(source,),
    )
    view = load_teaching_memory_view(service, "learner-r07b-misconception")
    stored = next(
        item for item in view.misconceptions
        if item.misconception_id == "misconception-source-backed"
    )

    assert stored.status is MisconceptionLifecycle.ADDRESSED
    assert stored.source_event_ids == ("challenge-real-1",)
    assert addressed_event.event_id in stored.intervention_event_ids
    persisted = json.dumps(
        service.get_profile_snapshot(
            "learner-r07b-misconception"
        ).extras["learner_memory"]["teaching_memory"],
        ensure_ascii=False,
    )
    assert "this text must not be copied" not in persisted
    assert "RESOLVED" not in persisted


def test_memory_interpretation_never_changes_mastery_or_creates_model_state() -> None:
    concept = ConceptLearningMemory(
        learner_id="learner-r07b-model-guard",
        concept_id="concept:dy3:four-f-four-f-transition",
        learning_attempts=2,
        first_seen=1.0,
        last_seen=2.0,
        learning_event_ids=("event-1", "event-2"),
        teaching_effect="delivered_reviewed_unverified",
        confidence=0.8,
    )
    view = TeachingMemoryView(
        learner_id="learner-r07b-model-guard",
        concept_learning=(concept,),
    )

    interpretation = interpret_teaching_memory(
        view, ("concept:dy3:four-f-four-f-transition",)
    )

    assert interpretation.strategy == "build_on_prior_exposure"
    assert not hasattr(interpretation, "mastery")
    assert not hasattr(interpretation, "theta")
    assert not hasattr(interpretation, "task_plan")


def test_different_learners_have_isolated_teaching_memory(monkeypatch) -> None:
    service = _ProfileService()
    _run(monkeypatch, service, "learner-r07b-a", "task-r07b-a")

    view_a = load_teaching_memory_view(service, "learner-r07b-a")
    view_b = load_teaching_memory_view(service, "learner-r07b-b")

    assert view_a.available is True
    assert view_b.available is False
    assert view_b.concept_learning == ()
    assert all(item.learner_id == "learner-r07b-a" for item in view_a.experiences)


def test_private_teaching_memory_does_not_leak_or_change_current_response_keys(
    monkeypatch,
) -> None:
    service = _ProfileService()
    first, _, _ = _run(
        monkeypatch, service, "learner-r07b-api", "task-r07b-api-1"
    )
    second, _, _ = _run(
        monkeypatch, service, "learner-r07b-api", "task-r07b-api-2"
    )

    assert set(first) == set(second)
    public = json.dumps(dict(second), ensure_ascii=False, default=str)
    for forbidden in (
            "TeachingMemory",
            "teaching_memory",
            "learning_attempts",
        "experience-",
        "_contract_candidate",
    ):
        assert forbidden not in public


def test_api_keeps_current_schema_and_never_projects_teaching_memory(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )

    response = client.post(
        "/api/query",
        json={"query": _QUERY, "learner_id": "learner-r07b-real-api"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert {
        "answer", "evidence", "review", "confidence", "action_type",
        "recommended_path", "task_id", "task_state", "task_events",
    }.issubset(data)
    serialized = json.dumps(data, ensure_ascii=False, default=str)
    for forbidden in (
        "TeachingMemoryView",
            "TeachingMemoryInterpretation",
            "teaching_memory",
            "learning_attempts",
        "experience-",
    ):
        assert forbidden not in serialized
