"""T1 product-result tests for the authoritative quality and release loop."""

from __future__ import annotations

import json
from dataclasses import replace

from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers, task_state as task_state_runtime
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.agent_contracts import DecisionType
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.interaction_recorder import InteractionRecorder
from dy3_polaris.l5.task_understanding import understand_task
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_private_runtime_carrier import (
    _review_candidate,
    _selected_generation,
)


UNSAFE = "THIS_UNSAFE_CLAIM_MUST_NEVER_REACH_PUBLIC_RESPONSE"


def _run_guidance(monkeypatch, *, task_id, query, generations, reviews):
    generation_iter = iter(generations)
    review_iter = iter(reviews)
    monkeypatch.setattr(agent_workers, "get_recorder", lambda: InteractionRecorder())
    monkeypatch.setattr(
        agent_workers,
        "run_diagnosis",
        lambda _payload, _deps: {
            "agent_id": agent_workers.DIAGNOSIS_AGENT_ID,
            "status": "completed",
            "weak_kps": [],
            "confidence": 0.8,
        },
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: next(generation_iter),
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda *_args, **_kwargs: next(review_iter),
    )
    task_context = task_state_runtime.create_task_context(task_id)
    task_state_runtime.set_task_state(task_context, "UNDERSTANDING")
    return agent_workers.run_guidance(
        {
            "task_id": task_id,
            "task_context": task_context,
            "query": query,
            "learner_id": "learner-t1",
        },
        AgentDependencies(),
    )


def test_first_pass_approved_is_full_release(monkeypatch) -> None:
    task_id = "task-t1-full"
    answer = "Reviewed Dy3+ explanation"
    result = _run_guidance(
        monkeypatch,
        task_id=task_id,
        query="Why does Dy3+ emit blue and yellow light?",
        generations=[_selected_generation(answer, task_id)],
        reviews=[_review_candidate(task_id, answer, "approved")],
    )

    assert result["quality_release"]["status"] == "FULL_RELEASE"
    assert result["quality_release"]["eligible"] is True
    assert result["answer"] == answer
    private = result._contract_candidate
    assert private.answer_correlation.correlation is True
    assert private.evidence_candidate.evidence_versions == (1,)


def test_approved_evaluation_with_uncertainty_is_limited_release(monkeypatch) -> None:
    task_id = "task-t1-limited"
    answer = "3000 K alone cannot establish health safety without SPD and exposure conditions."
    original_synthesis = agent_workers._synthesize_guidance_decision

    def _limited_synthesis(*args, **kwargs):
        decision, final = original_synthesis(*args, **kwargs)
        limited_decision = replace(
            decision,
            decision_type=DecisionType.ANSWER_WITH_UNCERTAINTY,
            answer_policy="SHOW_REVIEWED_WITH_UNCERTAINTY",
        )
        return limited_decision, replace(
            final,
            decision=limited_decision,
        )

    monkeypatch.setattr(
        agent_workers, "_synthesize_guidance_decision", _limited_synthesis,
    )
    result = _run_guidance(
        monkeypatch,
        task_id=task_id,
        query="Is 3000 K always safe for healthy lighting?",
        generations=[_selected_generation(answer, task_id)],
        reviews=[_review_candidate(task_id, answer, "approved")],
    )

    assert result["quality_release"]["status"] == "LIMITED_RELEASE"
    assert result["quality_release"]["eligible"] is True
    assert result["answer"] == answer


def test_revision_must_replace_unsafe_version_before_release(monkeypatch) -> None:
    task_id = "task-t1-revise"
    safe = "Revised answer with bounded scientific claim"
    result = _run_guidance(
        monkeypatch,
        task_id=task_id,
        query="Explain Dy3+ concentration quenching.",
        generations=[
            _selected_generation(UNSAFE, task_id),
            _selected_generation(safe, task_id),
        ],
        reviews=[
            _review_candidate(task_id, UNSAFE, "needs_review"),
            _review_candidate(task_id, safe, "approved"),
        ],
    )

    assert result["quality_release"]["status"] == "FULL_RELEASE"
    assert result["quality_release"]["correction_count"] == 1
    assert result["answer"] == safe
    serialized = json.dumps(dict(result), ensure_ascii=False)
    assert UNSAFE not in serialized


def test_rejected_answer_and_runtime_projections_never_leak_marker(monkeypatch) -> None:
    task_id = "task-t1-reject"
    result = _run_guidance(
        monkeypatch,
        task_id=task_id,
        query="Claim this unsupported statement is certain.",
        generations=[_selected_generation(UNSAFE, task_id)],
        reviews=[_review_candidate(task_id, UNSAFE, "rejected")],
    )

    assert result["quality_release"]["status"] == "REFUSE"
    assert result["answer"] == ""
    assert result["candidates"] == []
    assert result["debate"] is None
    assert UNSAFE not in json.dumps(dict(result), ensure_ascii=False)


def test_no_progress_withholds_original_answer(monkeypatch) -> None:
    task_id = "task-t1-no-progress"
    unchanged = _selected_generation(UNSAFE, task_id)
    result = _run_guidance(
        monkeypatch,
        task_id=task_id,
        query="Explain this mechanism.",
        generations=[unchanged, _selected_generation(UNSAFE, task_id)],
        reviews=[
            _review_candidate(task_id, UNSAFE, "needs_review"),
            _review_candidate(task_id, UNSAFE, "needs_review"),
        ],
    )

    assert result["quality_release"]["status"] == "WITHHOLD"
    assert result["answer"] == ""
    assert UNSAFE not in json.dumps(dict(result), ensure_ascii=False)


def test_fact_find_has_one_real_correction_budget_and_global_cap_is_20() -> None:
    fact = initialize_collaboration_context(
        {"task_id": "task-fact-budget", "query": "Dy3+ definition"},
        intent_resolver=lambda value, **_kwargs: understand_task(value, use_llm=False),
    )
    assert fact.collaboration_budget.max_expensive_iterations == 1
    assert fact.can_resolve("REVISE") is True
    fact.consume_resolution_budget("REVISE")
    assert fact.can_resolve("REVISE") is False

    complex_context = initialize_collaboration_context(
        {"task_id": "task-global-budget", "query": "判断 3000 K 是否一定安全"},
        intent_resolver=lambda value, **_kwargs: understand_task(value, use_llm=False),
    )
    for _ in range(10):
        complex_context.consume_resolution_budget("RE_RETRIEVE")
    for _ in range(10):
        complex_context.consume_resolution_budget("REVISE")
    assert complex_context.iteration_state["global_corrections_used"] == 20
    assert complex_context.can_resolve("REVISE") is False


class _UnsafeAskUserRuntime:
    async def run(self, _agent_id, input_data):
        context = input_data["task_context"]
        for state in ("RETRIEVING", "COLLABORATING", "REVIEWING", "ANSWERING"):
            task_state_runtime.set_task_state(context, state)
        return {
            "answer": UNSAFE,
            "confidence": 0.2,
            "review": {
                "agent_id": agent_workers.REVIEW_AGENT_ID,
                "status": "completed",
                "verdict": "needs_review",
            },
            "quality_release": {
                "status": "ASK_USER",
                "eligible": False,
                "message": "Need the material host and measurement conditions.",
                "reason_codes": ["clarification_required"],
                "review_status": "completed",
                "review_verdict": "needs_review",
                "correction_count": 0,
                "evidence_versions": [1],
            },
            "requires_confirmation": True,
            "action_type": "clarify",
            "clarify": {"question": "Which material host?", "options": ["phosphate", "silicate"]},
            "evidence": [],
            "recommended_path": [],
            "agent_trace": [],
            "collab_lines": [],
            "flow_events": [],
        }


def test_ask_user_and_l4_fallback_are_fail_closed_at_api() -> None:
    builder = UnifiedApp.create_full_app_builder()
    builder._handlers._agents = _UnsafeAskUserRuntime()
    client = TestClient(builder.create_app())
    ask = client.post("/api/query", json={
        "query": "Is this always safe?",
        "learner_id": "learner-t1",
    })
    assert ask.status_code == 200
    ask_data = ask.json()["data"]
    assert ask_data["quality_release"]["status"] == "ASK_USER"
    assert ask_data["answer"] == ""
    assert UNSAFE not in ask.text

    builder._handlers._agents = None
    degraded = client.post("/api/query", json={
        "query": "Give an unreviewed fallback answer",
        "learner_id": "learner-t1",
    })
    assert degraded.status_code in {200, 503}
    if degraded.status_code == 200:
        data = degraded.json()["data"]
        assert data["quality_release"]["status"] == "DEGRADED"
        assert data["answer"] == ""
