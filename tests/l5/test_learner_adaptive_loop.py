"""R-05C historical learning signal to teaching-decision loop tests."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_memory import (
    LearningEventClassification,
    LearningEventType,
    build_memory_views,
    create_learning_event,
)
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.learner_intelligence import build_learner_intelligence_view
from dy3_polaris.l5.task_understanding import understand_task
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_learner_memory import _ProfileService, _run_task


def test_learning_event_has_fixed_source_semantics_without_model_state() -> None:
    event = create_learning_event(
        learner_id="learner-r05c-event",
        task_id="task-r05c-event",
        event_type=LearningEventType.QUERY,
        source="user_interaction",
        topics=("dy3_emission",),
        content="Dy³⁺为什么发光？",
        timestamp=100.0,
    )

    assert event.classification is LearningEventClassification.OBSERVED
    assert event.event_type is LearningEventType.QUERY
    value = event.to_dict()
    assert value["topics"] == ["dy3_emission"]
    assert not {"mastery", "theta", "level"} & set(value)


def test_memory_projection_cannot_directly_mutate_plan_agent_input_or_decision() -> None:
    data = {
        "task_id": "task-r05c-boundary",
        "query": "为什么浓度升高反而降低发光？",
        "learner_id": "learner-r05c-boundary",
        "learner_level": "beginner",
    }
    context = initialize_collaboration_context(
        data,
        intent_resolver=lambda query, **_kwargs: understand_task(
            query, use_llm=False
        ),
    )
    plan_before = context.task_plan
    learner_context_before = dict(context.learner_context)
    decisions_before = tuple(context.decisions)
    views = {
        agent_workers.DIAGNOSIS_AGENT_ID: {
            "memory_available": True,
            "related_to_query": True,
            "prior_exposure_topics": ("dy3_emission", "energy_transition"),
        },
        agent_workers.GENERATION_AGENT_ID: {
            "memory_available": True,
            "legacy_projection": "isolated",
        },
        agent_workers.REVIEW_AGENT_ID: {
            "memory_available": True,
            "legacy_projection": "isolated",
        },
        agent_workers.GUIDANCE_AGENT_ID: {
            "memory_available": True,
            "legacy_projection": "isolated",
        },
    }

    agent_workers._apply_memory_to_collaboration_context(context, views)
    diagnosis_input = agent_workers._start_contract_agent(
        context, agent_workers.DIAGNOSIS_AGENT_ID
    )

    assert context.task_plan is plan_before
    assert context.learner_context == learner_context_before
    assert tuple(context.decisions) == decisions_before
    assert diagnosis_input is not None
    assert dict(diagnosis_input.learner_context) == learner_context_before


def test_learning_events_do_not_promote_depth_for_a_new_concept(
    monkeypatch,
) -> None:
    service = _ProfileService()
    learner_id = "learner-r05c-view"
    _run_task(
        monkeypatch,
        service=service,
        learner_id=learner_id,
        task_id="task-r05c-view-first",
        query="Dy³⁺为什么发光？",
        answer="Dy³⁺发光来自已审核的能级跃迁解释。",
    )
    query = "为什么浓度升高反而降低发光？"
    views = build_memory_views(service, learner_id, query)
    view = build_learner_intelligence_view(
        {"learner_id": learner_id, "query": query, "learner_level": "beginner"},
        AgentDependencies(profile_service=service),
        learner_memory_view=views[agent_workers.DIAGNOSIS_AGENT_ID],
    )

    observed = view.value("facts", "observed_learning_events", ())
    historical = view.value("derived_context", "historical_learning_signals", ())
    assert any(item["event_type"] == "query" for item in observed)
    assert any(item["event_type"] == "topic_explained" for item in historical)
    assert view.value("derived_context", "adaptive_strategy") == "establish_foundation"
    assert view.value("derived_context", "recommended_depth") == "beginner"
    assert view.value("models", "mastery", {}) == {}


def test_two_turn_exposure_without_assessment_does_not_inflate_depth(
    monkeypatch,
) -> None:
    service = _ProfileService()
    learner_id = "learner-r05c-loop"
    first, first_diagnosis, _ = _run_task(
        monkeypatch,
        service=service,
        learner_id=learner_id,
        task_id="task-r05c-loop-first",
        query="Dy³⁺为什么发光？",
        answer="Dy³⁺发光来自已审核的能级跃迁解释。",
    )
    follow_up = "为什么浓度升高反而降低发光？"
    answer = "浓度升高可能增强离子间能量迁移并形成非辐射损失通道。"
    second, second_diagnosis, second_generation = _run_task(
        monkeypatch,
        service=service,
        learner_id=learner_id,
        task_id="task-r05c-loop-second",
        query=follow_up,
        answer=answer,
    )
    new_user, new_diagnosis, new_generation = _run_task(
        monkeypatch,
        service=_ProfileService(),
        learner_id="learner-r05c-new",
        task_id="task-r05c-loop-new",
        query=follow_up,
        answer=answer,
    )

    first_view = first_diagnosis[0]["_learner_intelligence_view"]
    second_view = second_diagnosis[0]["_learner_intelligence_view"]
    new_view = new_diagnosis[0]["_learner_intelligence_view"]
    assert first_view.value("derived_context", "recommended_depth") == "beginner"
    assert second_view.value("derived_context", "recommended_depth") == "beginner"
    assert new_view.value("derived_context", "recommended_depth") == "beginner"
    assert second_generation[0]["learner_level"] == "beginner"
    assert new_generation[0]["learner_level"] == "beginner"

    first_decision = first._contract_candidate.final_collaboration_result.decision
    second_decision = second._contract_candidate.final_collaboration_result.decision
    new_decision = new_user._contract_candidate.final_collaboration_result.decision
    assert first_decision.learner_depth == "beginner"
    assert second_decision.learner_depth == "beginner"
    assert new_decision.learner_depth == "beginner"
    assert second_decision.next_action == new_decision.next_action
    assert second["answer"] == new_user["answer"] == answer


def test_current_api_does_not_expose_adaptive_loop_internals(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    response = client.post(
        "/api/query",
        json={
            "query": "Dy³⁺为什么发光？",
            "learner_id": "learner-r05c-api",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    serialized = json.dumps(data, ensure_ascii=False, default=str)
    for forbidden in (
        "LearningEvent",
        "learning_events",
        "learner_memory",
        "AdaptiveContext",
        "LearnerState",
        "adaptive_strategy",
        "historical_learning_signals",
    ):
        assert forbidden not in serialized
    assert {
        "answer",
        "evidence",
        "review",
        "confidence",
        "action_type",
        "recommended_path",
        "task_id",
        "task_state",
    }.issubset(data)
