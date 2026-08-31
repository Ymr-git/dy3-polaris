"""P1-16 private candidate set and TaskResult readiness gate tests."""

from __future__ import annotations

import hashlib
import json

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5 import unified_app as unified_app_runtime
from dy3_polaris.l5.agent_workers import (
    GUIDANCE_AGENT_ID,
    _EvidenceCandidate,
    _FinalPrivateCandidateSet,
    _PrivateRuntimeCarrier,
    _ReviewCandidate,
    _correlate_final_answer,
)
from dy3_polaris.l5.interaction_recorder import InteractionRecorder
from dy3_polaris.l5.unified_app import (
    UnifiedApp,
    _evaluate_task_result_readiness,
)


def _task_context(task_id: str) -> dict[str, object]:
    context = task_state_runtime.create_task_context(task_id)
    for state in (
        "UNDERSTANDING", "PLANNING", "RETRIEVING",
        "COLLABORATING", "REVIEWING", "ANSWERING",
    ):
        task_state_runtime.set_task_state(context, state)
    return context


def _guidance_carrier(
    *,
    task_id: str,
    final_answer: str,
    evidence_answer: str | None = None,
    review_answer: str | None = None,
    review_producer: str = "agent.quality.review/run_review",
    review_status: str = "completed",
    review_verdict: str = "approved",
    real_reviewer_executed: bool = True,
    mapping_refused_reason: str = "",
) -> _PrivateRuntimeCarrier:
    evidence_answer = (
        final_answer if evidence_answer is None else evidence_answer
    )
    review_answer = final_answer if review_answer is None else review_answer
    evidence = _EvidenceCandidate(
        task_id=task_id,
        producer=(
            "agent.knowledge.generation/"
            "_run_multi_candidate_generation"
        ),
        stage="selected",
        answer_identity=hashlib.sha256(
            f"{task_id}{evidence_answer}".encode("utf-8")
        ).hexdigest(),
        context_chunks=("real context chunk",),
        citations=("real citation",),
        sources=({"chunk_id": "chunk-1"},),
        knowledge_unavailable=False,
        honest_unavailable=False,
    )
    review = _ReviewCandidate(
        task_id=task_id,
        producer=review_producer,
        reviewed_answer_identity=(
            hashlib.sha256(
                f"{task_id}{review_answer}".encode("utf-8")
            ).hexdigest()
            if real_reviewer_executed and review_answer
            else ""
        ),
        raw_status=review_status,
        raw_verdict=review_verdict,
        raw_reason="raw review reason",
        raw_fact_check={"passed": True, "checked": 1, "failed": 0},
        raw_anti_hallucination={
            "action": "",
            "score": 1.0,
            "hallucination_detected": False,
        },
        raw_confidence=1.0,
        real_reviewer_executed=real_reviewer_executed,
        mapping_refused_reason=mapping_refused_reason,
    )
    correlation = _correlate_final_answer(
        task_id=task_id,
        final_answer=final_answer,
        evidence_candidate=evidence,
        review_candidate=review,
    )
    guidance = _PrivateRuntimeCarrier({"answer": final_answer})
    guidance._contract_candidate = _FinalPrivateCandidateSet(
        evidence_candidate=evidence,
        review_candidate=review,
        answer_correlation=correlation,
    )
    return guidance


def _readiness(guidance, task_id: str) -> dict[str, object]:
    return _evaluate_task_result_readiness(
        guidance,
        _task_context(task_id),
    )


def test_final_private_candidate_set_stays_out_of_mapping_and_json() -> None:
    guidance = _guidance_carrier(
        task_id="task-ready-private",
        final_answer="ready answer",
    )

    assert isinstance(
        guidance._contract_candidate,
        _FinalPrivateCandidateSet,
    )
    assert "_contract_candidate" not in guidance
    assert dict(guidance) == {"answer": "ready answer"}
    serialized = json.dumps(guidance, ensure_ascii=False)
    assert "FinalPrivateCandidateSet" not in serialized
    assert "answer_correlation" not in serialized
    assert "readiness" not in serialized


def test_normal_agent_private_facts_are_ready() -> None:
    result = _readiness(
        _guidance_carrier(
            task_id="task-ready-normal",
            final_answer="normal answer",
        ),
        "task-ready-normal",
    )

    assert result == {"ready": True, "reasons": ()}


@pytest.mark.parametrize(
    ("producer", "status", "verdict", "real", "refusal"),
    (
        (
            "synthetic_review",
            "completed",
            "approved",
            False,
            "real reviewer not executed",
        ),
        (
            "skipped",
            "skipped",
            "skipped",
            False,
            "no reviewed content",
        ),
    ),
)
def test_synthetic_and_skipped_reviews_are_not_ready(
    producer,
    status,
    verdict,
    real,
    refusal,
) -> None:
    result = _readiness(
        _guidance_carrier(
            task_id="task-review-refused",
            final_answer="candidate answer",
            review_producer=producer,
            review_status=status,
            review_verdict=verdict,
            real_reviewer_executed=real,
            mapping_refused_reason=refusal,
        ),
        "task-review-refused",
    )

    assert result["ready"] is False
    assert "review_producer_invalid" in result["reasons"]
    assert "review_candidate_refused" in result["reasons"]


def test_self_correction_adopted_is_ready() -> None:
    result = _readiness(
        _guidance_carrier(
            task_id="task-self-adopted",
            final_answer="generation2 answer",
            evidence_answer="generation2 answer",
            review_answer="generation2 answer",
        ),
        "task-self-adopted",
    )

    assert result["ready"] is True


def test_self_correction_rejected_mismatch_is_not_ready() -> None:
    result = _readiness(
        _guidance_carrier(
            task_id="task-self-rejected",
            final_answer="initial answer",
            evidence_answer="initial answer",
            review_answer="rejected generation2 answer",
        ),
        "task-self-rejected",
    )

    assert result["ready"] is False
    assert "answer_identity_mismatch" in result["reasons"]
    assert "answer_correlation_refused" in result["reasons"]


@pytest.mark.parametrize(
    "answer",
    ("critic adopted answer", "critic kept current answer"),
)
def test_critic_adopted_and_not_adopted_use_final_identity(answer) -> None:
    result = _readiness(
        _guidance_carrier(
            task_id="task-critic-readiness",
            final_answer=answer,
        ),
        "task-critic-readiness",
    )

    assert result["ready"] is True


def test_l4_fallback_without_agent_candidates_is_not_ready() -> None:
    result = _readiness(None, "task-l4-fallback")

    assert result["ready"] is False
    assert result["reasons"] == ("missing_agent_private_candidates",)


class _ReadyGuidanceRuntime:
    def __init__(self) -> None:
        self._recorder = InteractionRecorder()

    async def run(self, agent_id, input_data):
        assert agent_id == GUIDANCE_AGENT_ID
        task_context = input_data["task_context"]
        for state in (
            "RETRIEVING",
            "COLLABORATING",
            "REVIEWING",
            "ANSWERING",
        ):
            task_state_runtime.set_task_state(task_context, state)
        guidance = _guidance_carrier(
            task_id=input_data["task_id"],
            final_answer="ready API answer",
        )
        guidance.update({
            "agent_id": GUIDANCE_AGENT_ID,
            "status": "completed",
            "task_id": input_data["task_id"],
            "confidence": 0.9,
            "review": {"verdict": "approved"},
            "evidence": [{"content": "public evidence", "source": "runtime"}],
            "recommended_path": [],
            "action_type": "answer",
            "requires_confirmation": False,
        })
        return guidance

    def get_recorder(self):
        return self._recorder


def test_api_executes_gate_without_changing_current_response_keys(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    original_gate = unified_app_runtime._evaluate_task_result_readiness

    def observing_gate(guidance, task_context):
        result = original_gate(guidance, task_context)
        calls.append(result)
        return result

    monkeypatch.setattr(
        unified_app_runtime,
        "_evaluate_task_result_readiness",
        observing_gate,
    )
    builder = UnifiedApp.create_full_app_builder()
    builder._handlers._agents = _ReadyGuidanceRuntime()
    client = TestClient(builder.create_app())

    response = client.post("/api/query", json={"query": "readiness gate"})

    assert response.status_code == 200
    assert calls == [{"ready": True, "reasons": ()}]
    data = response.json()["data"]
    assert set(data) == {
        "task_id", "task_state", "task_events", "action_type",
        "confidence", "recommended_path", "safety_level", "pipeline",
        "evidence", "sources", "session", "learner", "viz",
        "total_elapsed_ms", "requires_confirmation",
        "knowledge_unavailable", "review", "agent_trace", "collab_lines",
        "self_correction", "reasoning_loop", "flow_events",
        "broadcast_events", "consensus_score", "consensus_reached",
        "candidate_count", "candidates", "divergence_matrix", "debate",
        "needs_adjudication", "consensus_threshold", "question_type",
        "confirmation_questions", "confirmation_reason", "plan_id",
            "clarify", "answer",
            "quality_release", "teaching_strategy", "learning_resources",
            "learner_context", "knowledge_context",
        }
    serialized = response.text
    assert "readiness" not in serialized
    assert "FinalPrivateCandidateSet" not in serialized
    assert "answer_correlation" not in serialized
