"""R-05D structured misconception intelligence tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_contracts import (
    Challenge,
    ChallengeSeverity,
    ChallengeType,
    Claim,
    ClaimType,
    RequestedAction,
    ResolutionAction,
)
from dy3_polaris.l5.agent_memory import (
    LearningEventOutcome,
    LearningEventType,
    MisconceptionStatus,
    build_memory_views,
    commit_learner_memory,
    commit_learning_event,
    create_learning_event,
    extract_memory_candidate,
    load_learner_memory,
)
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.learner_intelligence import build_learner_intelligence_view
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_learner_memory import _ProfileService


def _structured_context(
    *,
    task_id: str,
    belief: str,
    reason: str,
    missing: tuple[str, ...],
    challenge_type: ChallengeType,
) -> SimpleNamespace:
    claim = Claim(
        claim_id=f"claim-{task_id}",
        statement=belief,
        claim_type=ClaimType.INFERENCE,
        evidence_refs=(f"evidence-{task_id}",),
        confidence=0.6,
    )
    contribution = SimpleNamespace(
        contribution_id=f"contrib-{task_id}",
        claims=(claim,),
    )
    challenge = Challenge(
        challenge_id=f"challenge-{task_id}",
        task_id=task_id,
        subtask_id=f"subtask-{task_id}",
        reviewer_agent_id=agent_workers.REVIEW_AGENT_ID,
        target_contribution_id=contribution.contribution_id,
        target_claim_ids=(claim.claim_id,),
        challenge_type=challenge_type,
        reason=reason,
        severity=ChallengeSeverity.HIGH,
        missing_information=missing,
        evidence_refs=claim.evidence_refs,
        requested_action=ResolutionAction.REVISE,
        status="OPEN",
        iteration=1,
    )
    return SimpleNamespace(
        contributions=[contribution],
        challenges=[challenge],
    )


def _candidate(context: SimpleNamespace, task_id: str, question: str):
    return extract_memory_candidate(
        context=context,
        final_result=SimpleNamespace(
            task_id=task_id,
            task_mode=SimpleNamespace(value="EVALUATE"),
            answer_identity=f"answer-{task_id}",
            completion_eligibility=True,
            next_action="按审核意见核对条件与证据",
            provenance_refs=(f"doc-{task_id}",),
        ),
        question=question,
        learner_id="learner-misconception",
    )


@pytest.mark.parametrize(
    ("belief", "question", "reason", "missing", "challenge_type"),
    (
        (
            "低色温在所有光谱和暴露条件下都必然更健康",
            "3000K低色温是否一定更健康？",
            "结论忽略光谱功率分布和暴露条件",
            ("SPD", "暴露条件"),
            ChallengeType.SAFETY_OVERCLAIM,
        ),
        (
            "提高掺杂浓度在任何基质中都会持续增强发光",
            "浓度猝灭为什么会降低发光？",
            "结论忽略能量迁移和临界浓度",
            ("浓度猝灭证据", "基质条件"),
            ChallengeType.CONDITION_MISMATCH,
        ),
    ),
)
def test_structured_challenges_create_domain_general_misconceptions(
    belief: str,
    question: str,
    reason: str,
    missing: tuple[str, ...],
    challenge_type: ChallengeType,
) -> None:
    task_id = f"task-{challenge_type.value.lower()}"
    candidate = _candidate(
        _structured_context(
            task_id=task_id,
            belief=belief,
            reason=reason,
            missing=missing,
            challenge_type=challenge_type,
        ),
        task_id,
        question,
    )

    assert candidate.valid is True
    assert len(candidate.misconceptions) == 1
    misconception = candidate.misconceptions[0]
    assert misconception.belief == belief
    assert misconception.status is MisconceptionStatus.ACTIVE
    assert misconception.source_events == (f"challenge-{task_id}",)
    assert misconception.evidence == (f"evidence-{task_id}",)
    assert missing[0] in misconception.correction_strategy
    assert candidate.error_patterns[0]["misconception"] == belief


def test_observed_correction_updates_confidence_then_resolves_after_reverification() -> None:
    service = _ProfileService()
    task_id = "task-misconception-lifecycle"
    candidate = _candidate(
        _structured_context(
            task_id=task_id,
            belief="单一色温数值足以证明全部健康风险",
            reason="评价边界缺少光谱和暴露条件",
            missing=("SPD", "暴露条件"),
            challenge_type=ChallengeType.SAFETY_OVERCLAIM,
        ),
        task_id,
        "3000K色温能否单独证明健康性？",
    )
    assert commit_learner_memory(service, "learner-misconception", candidate)
    initial = load_learner_memory(service, "learner-misconception")["misconceptions"][0]

    first = create_learning_event(
        learner_id="learner-misconception",
        task_id="task-feedback-1",
        event_type=LearningEventType.FEEDBACK,
        source="user_feedback",
        reference=initial["misconception_id"],
        outcome=LearningEventOutcome.CORRECTION_ACCEPTED,
        timestamp=200.0,
    )
    assert commit_learning_event(service, "learner-misconception", first)
    after_first = load_learner_memory(service, "learner-misconception")["misconceptions"][0]
    assert after_first["confidence"] < initial["confidence"]
    assert after_first["status"] == "UNCERTAIN"

    second = create_learning_event(
        learner_id="learner-misconception",
        task_id="task-feedback-2",
        event_type=LearningEventType.PRACTICE_RESULT,
        source="practice_result",
        reference=initial["misconception_id"],
        outcome=LearningEventOutcome.CORRECT_RESPONSE,
        timestamp=300.0,
    )
    assert commit_learning_event(service, "learner-misconception", second)
    resolved = load_learner_memory(service, "learner-misconception")["misconceptions"][0]
    assert resolved["status"] == "RESOLVED"
    assert resolved["confidence"] <= 0.3


def test_misconception_enters_view_then_diagnosis_only() -> None:
    service = _ProfileService()
    task_id = "task-misconception-view"
    candidate = _candidate(
        _structured_context(
            task_id=task_id,
            belief="色温数值可以替代完整光谱安全评价",
            reason="缺少SPD和暴露条件",
            missing=("SPD",),
            challenge_type=ChallengeType.SAFETY_OVERCLAIM,
        ),
        task_id,
        "色温能否替代健康照明的完整评价？",
    )
    assert commit_learner_memory(service, "learner-misconception", candidate)
    query = "如何评价健康照明灯具？"
    views = build_memory_views(service, "learner-misconception", query)
    view = build_learner_intelligence_view(
        {"learner_id": "learner-misconception", "query": query},
        AgentDependencies(profile_service=service),
        learner_memory_view=views[agent_workers.DIAGNOSIS_AGENT_ID],
    )
    diagnosis = agent_workers.run_diagnosis(
        {"learner_id": "learner-misconception", "query": query,
         "_learner_intelligence_view": view},
        AgentDependencies(profile_service=service),
    )

    assert view.value("models", "misconceptions")
    assert view.value("derived_context", "misconception_focus")
    assert "错误认知假设" in diagnosis["summary"]
    assert "misconceptions" not in diagnosis


def test_api_does_not_leak_misconception_models(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    response = client.post(
        "/api/query",
        json={"query": "如何评价照明光谱的健康边界？", "learner_id": "learner-r05d-api"},
    )
    assert response.status_code == 200
    serialized = json.dumps(response.json()["data"], ensure_ascii=False, default=str)
    for forbidden in (
        "Misconception", "misconception_id", "misconception_focus",
        "LearnerMemory", "LearningPath",
    ):
        assert forbidden not in serialized
