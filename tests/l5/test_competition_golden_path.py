"""Day1 Golden Path facts: final answer, evidence, and review must stay aligned."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_workers import (
    AgentDependencies,
    GENERATION_AGENT_ID,
    REVIEW_AGENT_ID,
    _FinalPrivateCandidateSet,
)
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_private_runtime_carrier import (
    _answer_correlation,
    _guidance_input,
    _no_critic_adoption,
    _review_candidate,
    _selected_generation,
    _stub_guidance_edges,
)
from tests.l5.test_readiness_gate import (
    _ReadyGuidanceRuntime,
    _guidance_carrier,
)


MAIN_QUERY = (
    "Dy³⁺ 的黄蓝双发射为什么能够形成近白光？"
    "黄蓝比、色温与蓝光健康风险之间有什么关系？"
)


def test_case1_dy3_golden_answer_evidence_review_use_real_reviewer(
    monkeypatch,
) -> None:
    """A terminal main-case answer must not substitute synthetic approval."""
    _stub_guidance_edges(monkeypatch)
    task_id = "task-day1-dy3-main"
    answer = "Dy³⁺ 黄蓝发射的相对强度共同影响综合色度与近白光表现。"
    terminal = _selected_generation(answer, task_id, stage="terminal")
    terminal["query"] = MAIN_QUERY
    terminal["plain_language"] = True
    review_calls: list[str] = []

    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: terminal,
    )

    def observed_review(input_data, _deps):
        reviewed = str(input_data.get("content") or "")
        review_calls.append(reviewed)
        return _review_candidate(task_id, reviewed)

    monkeypatch.setattr(agent_workers, "run_review", observed_review)
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )
    guidance_input = _guidance_input(task_id)
    guidance_input["query"] = MAIN_QUERY

    result = agent_workers.run_guidance(
        guidance_input,
        AgentDependencies(),
    )

    private_set = result._contract_candidate
    assert isinstance(private_set, _FinalPrivateCandidateSet)
    correlation = private_set.answer_correlation
    review_candidate = private_set.review_candidate
    evidence_candidate = private_set.evidence_candidate
    observed = {
        "review_calls": review_calls,
        "review_producer": review_candidate.producer if review_candidate else "",
        "real_reviewer_executed": (
            review_candidate.real_reviewer_executed if review_candidate else False
        ),
        "final_identity": correlation.final_answer_identity,
        "evidence_identity": (
            evidence_candidate.answer_identity if evidence_candidate else ""
        ),
        "review_identity": (
            review_candidate.reviewed_answer_identity if review_candidate else ""
        ),
        "correlation": correlation.correlation,
    }
    assert observed == {
        "review_calls": [answer],
        "review_producer": "agent.quality.review/run_review",
        "real_reviewer_executed": True,
        "final_identity": correlation.final_answer_identity,
        "evidence_identity": correlation.final_answer_identity,
        "review_identity": correlation.final_answer_identity,
        "correlation": True,
    }


def test_case2_self_correction_adopted_keeps_generation2_review2_pair(
    monkeypatch,
) -> None:
    _stub_guidance_edges(monkeypatch)
    task_id = "task-day1-self-adopted"
    generation1 = _selected_generation("generation1", task_id)
    generation2 = _selected_generation("generation2", task_id)
    review1 = _review_candidate(task_id, "generation1", "needs_review")
    review2 = _review_candidate(task_id, "generation2", "approved")
    generations = iter((generation1, generation2))
    reviews = iter((review1, review2))

    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: next(generations),
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda *_args, **_kwargs: next(reviews),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )

    result = agent_workers.run_guidance(
        _guidance_input(task_id),
        AgentDependencies(),
    )

    correlation = _answer_correlation(result)
    private_set = result._contract_candidate
    assert private_set.evidence_candidate is generation2._contract_candidate
    assert private_set.review_candidate is review2._contract_candidate
    assert result["answer"] == "generation2"
    assert correlation.correlation is True
    assert (
        correlation.final_answer_identity
        == generation2._contract_candidate.answer_identity
        == review2._contract_candidate.reviewed_answer_identity
    )


def test_case3_self_correction_not_adopted_keeps_generation1_review1_pair(
    monkeypatch,
) -> None:
    _stub_guidance_edges(monkeypatch)
    task_id = "task-day1-self-not-adopted"
    generation1 = _selected_generation("generation1", task_id)
    generation2 = _selected_generation("generation2 rejected", task_id)
    review1 = _review_candidate(task_id, "generation1", "needs_review")
    review2 = _review_candidate(task_id, "generation2 rejected", "rejected")
    generations = iter((generation1, generation2))
    reviews = iter((review1, review2))

    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: next(generations),
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda *_args, **_kwargs: next(reviews),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )

    result = agent_workers.run_guidance(
        _guidance_input(task_id),
        AgentDependencies(),
    )

    correlation = _answer_correlation(result)
    private_set = result._contract_candidate
    assert private_set.evidence_candidate is generation1._contract_candidate
    assert private_set.review_candidate is review1._contract_candidate
    assert result["answer"] == ""
    assert correlation.correlation is False
    assert (
        generation1._contract_candidate.answer_identity
        == review1._contract_candidate.reviewed_answer_identity
    )


class _NonCompletingReviewRuntime(_ReadyGuidanceRuntime):
    """Reuse the existing API runtime fixture with a non-final review fact."""

    def __init__(self, public_review: dict[str, object], private_review: dict[str, object]):
        super().__init__()
        self._public_review = public_review
        self._private_review = private_review

    async def run(self, agent_id, input_data):
        guidance = await super().run(agent_id, input_data)
        private = _guidance_carrier(
            task_id=input_data["task_id"],
            final_answer=str(guidance["answer"]),
            **self._private_review,
        )
        guidance._contract_candidate = private._contract_candidate
        guidance["review"] = dict(self._public_review)
        return guidance


@pytest.mark.parametrize(
    ("public_review", "private_review"),
    [
        pytest.param(
            {"verdict": "approved", "confidence": 0.9},
            {
                "review_producer": "synthetic_review",
                "real_reviewer_executed": False,
                "mapping_refused_reason": "real reviewer not executed",
            },
            id="synthetic-approved",
        ),
        pytest.param(
            {
                "agent_id": REVIEW_AGENT_ID,
                "status": "skipped",
                "verdict": "skipped",
                "reason": "内容为空，跳过审核",
                "confidence": 1.0,
            },
            {
                "review_producer": "skipped",
                "review_status": "skipped",
                "review_verdict": "skipped",
                "real_reviewer_executed": False,
                "mapping_refused_reason": "no reviewed content",
            },
            id="skipped",
        ),
        pytest.param(
            {
                "agent_id": REVIEW_AGENT_ID,
                "status": "completed",
                "verdict": "needs_review",
                "reason": "仍需复核",
                "confidence": 0.6,
            },
            {
                "review_producer": "agent.quality.review/run_review",
                "review_status": "completed",
                "review_verdict": "needs_review",
                "real_reviewer_executed": True,
            },
            id="unresolved-needs-review",
        ),
    ],
)
def test_case4_synthetic_skipped_or_unresolved_review_cannot_complete(
    public_review,
    private_review,
) -> None:
    builder = UnifiedApp.create_full_app_builder()
    builder._handlers._agents = _NonCompletingReviewRuntime(
        public_review,
        private_review,
    )
    client = TestClient(builder.create_app())

    response = client.post("/api/query", json={"query": MAIN_QUERY})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["task_state"] != "COMPLETED"
    assert data["review"] == public_review
    serialized = json.dumps(response.json(), ensure_ascii=False)
    for private_name in (
        "EvidenceCandidate",
        "ReviewCandidate",
        "answer_identity",
        "correlation",
        "_contract_candidate",
    ):
        assert private_name not in serialized
