"""R-03F authoritative GuidanceDecision and FinalCollaborationResult tests."""

from __future__ import annotations

import json

import pytest

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_contracts import (
    ClaimFinalState,
    DecisionType,
    FinalCollaborationResult,
)
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.interaction_recorder import InteractionRecorder
from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5.task_understanding import TaskMode, understand_task
from tests.l5.test_collaboration_loop import _review
from tests.l5.test_private_runtime_carrier import (
    _FinalPrivateCandidateSet,
    _review_candidate,
    _selected_generation,
)


def _request(task_id: str, query: str, *, learner_level: str = "beginner"):
    task_context = task_state_runtime.create_task_context(task_id)
    task_state_runtime.set_task_state(task_context, "UNDERSTANDING")
    data = {
        "task_id": task_id,
        "query": query,
        "learner_id": f"learner-{task_id}",
        "learner_level": learner_level,
        "task_context": task_context,
    }
    context = initialize_collaboration_context(
        data,
        intent_resolver=lambda value, **_kwargs: understand_task(value, use_llm=False),
    )
    return data, context


def _stub_runtime(monkeypatch, *, level: str = "beginner") -> None:
    monkeypatch.setattr(
        agent_workers,
        "run_diagnosis",
        lambda input_data, _deps: {
            "agent_id": agent_workers.DIAGNOSIS_AGENT_ID,
            "status": "completed",
            "learner_id": input_data.get("learner_id"),
            "level": level,
            "summary": f"learner depth is {level}",
            "weak_kps": ["energy levels"] if level == "beginner" else [],
            "confidence": 0.9,
        },
    )
    monkeypatch.setattr(agent_workers, "get_recorder", lambda: InteractionRecorder())


def _run(monkeypatch, data, generations, reviews, *, level="beginner", deps=None):
    _stub_runtime(monkeypatch, level=level)
    generation_values = iter(generations)
    review_values = iter(reviews)
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: next(generation_values),
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda *_args, **_kwargs: next(review_values),
    )
    return agent_workers.run_guidance(data, deps or AgentDependencies())


def _final(result) -> FinalCollaborationResult:
    private = result._contract_candidate
    assert isinstance(private, _FinalPrivateCandidateSet)
    assert isinstance(private.final_collaboration_result, FinalCollaborationResult)
    return private.final_collaboration_result


def test_approved_result_binds_final_generation_review_task_and_evidence(monkeypatch) -> None:
    task_id = "task-r03f-approved"
    data, context = _request(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    answer = "Dy³⁺ 的蓝、黄发射来自已标注的 4F9/2 能级跃迁。"
    generation = _selected_generation(answer, task_id)
    result = _run(
        monkeypatch,
        data,
        [generation],
        [_review_candidate(task_id, answer, "approved")],
    )

    final = _final(result)
    assert final.task_id == task_id
    assert final.answer == answer == result["answer"]
    assert final.answer_identity == final.review.artifact_identity
    assert final.review is [c for c in context.contributions if c.agent_id == agent_workers.REVIEW_AGENT_ID][-1]
    assert final.evidence == tuple(agent_workers._active_evidence_packs(context))
    assert final.accepted_claims
    assert not final.rejected_claims
    assert all(item.state is ClaimFinalState.ACCEPTED for item in final.decision.claim_decisions)


def test_post_review_mutation_is_forbidden_even_when_adjudication_flag_is_set(monkeypatch) -> None:
    task_id = "task-r03f-post-review"
    data, _context = _request(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    answer_a = "Reviewer 审核的 Answer A"
    generation = _selected_generation(answer_a, task_id)
    generation["needs_adjudication"] = True
    result = _run(
        monkeypatch,
        data,
        [generation],
        [_review_candidate(task_id, answer_a, "approved")],
    )

    final = _final(result)
    assert result["answer"] == answer_a
    assert final.answer == answer_a
    assert final.answer_identity == final.review.artifact_identity
    assert result._contract_candidate.answer_correlation.correlation is True


def test_diagnosis_changes_learning_decision_but_not_scientific_answer(monkeypatch) -> None:
    query = "为什么 Dy³⁺ 会产生黄蓝双发射？"
    answer = "审核后的科学事实保持不变。"
    beginner_data, _ = _request("task-r03f-beginner", query)
    beginner = _run(
        monkeypatch,
        beginner_data,
        [_selected_generation(answer, "task-r03f-beginner")],
        [_review_candidate("task-r03f-beginner", answer)],
        level="beginner",
    )
    advanced_data, _ = _request("task-r03f-advanced", query, learner_level="advanced")
    advanced = _run(
        monkeypatch,
        advanced_data,
        [_selected_generation(answer, "task-r03f-advanced")],
        [_review_candidate("task-r03f-advanced", answer)],
        level="advanced",
    )

    beginner_final = _final(beginner)
    advanced_final = _final(advanced)
    assert beginner["answer"] == advanced["answer"] == answer
    assert beginner_final.decision.next_action != advanced_final.decision.next_action
    assert beginner_final.decision.recommended_path != advanced_final.decision.recommended_path


def test_rejected_claim_is_withheld_and_cannot_be_restored_by_guidance(monkeypatch) -> None:
    task_id = "task-r03f-reject"
    data, _ = _request(task_id, "3000 K 的 Dy³⁺ 白光是否一定更健康？")
    unsafe = "3000 K 一定更安全。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(unsafe, task_id)],
        [_review(task_id, unsafe, "rejected", "低 CCT 不等于一定安全")],
    )

    final = _final(result)
    assert final.task_mode is TaskMode.EVALUATE
    assert final.decision.decision_type is DecisionType.REFUSE_CONCLUSION
    assert final.rejected_claims
    assert final.answer == ""
    assert result["answer"] == ""
    assert unsafe not in result["answer"]


def test_unresolved_review_preserves_uncertainty_and_caps_confidence(monkeypatch) -> None:
    task_id = "task-r03f-uncertain"
    data, _ = _request(task_id, "Dy³⁺ 的蓝光发射波长是多少？")
    answer = "当前证据仅支持一个条件性结论。"
    generation = _selected_generation(answer, task_id)
    generation["confidence"] = 0.99
    revised = _selected_generation("复核后仍只有条件性结论。", task_id)
    revised["confidence"] = 0.99
    result = _run(
        monkeypatch,
        data,
        [generation, revised],
        [
            _review(task_id, answer, "needs_review", "缺少直接证据"),
            _review(task_id, revised["answer"], "needs_review", "缺少直接证据"),
        ],
    )

    final = _final(result)
    assert final.uncertain_claims
    assert not final.accepted_claims
    assert final.decision.decision_type in {
        DecisionType.ANSWER_WITH_UNCERTAINTY,
        DecisionType.PARTIAL_ANSWER,
    }
    assert final.decision.confidence <= 0.55
    assert any(item.state is ClaimFinalState.UNCERTAIN for item in final.decision.claim_decisions)
    assert all(
        item.state in {ClaimFinalState.UNCERTAIN, ClaimFinalState.REJECTED}
        for item in final.decision.claim_decisions
    )


def test_compare_knowledge_gap_does_not_create_ranking(monkeypatch) -> None:
    task_id = "task-r03f-gq07"
    data, _ = _request(task_id, "如何比较 YSZ:Dy³⁺ 和 YAG:Dy³⁺ 的量子效率？")
    generations = [
        _selected_generation(f"第 {index} 次复核仍缺少同条件数据，不能排名。", task_id)
        for index in range(6)
    ]
    result = _run(
        monkeypatch,
        data,
        generations,
        [
            _review(task_id, item["answer"], "needs_review", "同条件证据不足")
            for item in generations
        ],
    )

    final = _final(result)
    assert final.task_mode is TaskMode.COMPARE
    assert final.decision.decision_type is DecisionType.KNOWLEDGE_GAP
    assert result["answer"] == ""
    assert result["quality_release"]["status"] == "WITHHOLD"
    assert final.knowledge_gaps
    assert "补充同条件测试数据" in final.next_action


def test_3000k_safety_overclaim_remains_rejected_and_uncertainty_propagates(monkeypatch) -> None:
    task_id = "task-r03f-3000k"
    data, context = _request(task_id, "3000 K 的 Dy³⁺ 白光是否一定更健康？")
    unsafe = _selected_generation("3000 K 一定更安全。", task_id)
    conditional = _selected_generation(
        "不能仅凭 CCT 判断；还需 SPD、暴露条件和风险加权指标。",
        task_id,
    )
    result = _run(
        monkeypatch,
        data,
        [unsafe, conditional],
        [
            _review(task_id, unsafe["answer"], "needs_review", "安全结论过度"),
            _review(task_id, conditional["answer"], "approved", "条件化结论通过"),
        ],
    )

    final = _final(result)
    assert final.task_mode is TaskMode.EVALUATE
    assert final.decision.decision_type is DecisionType.ANSWER_WITH_UNCERTAINTY
    assert final.rejected_claims
    assert any("一定更安全" in claim.statement for claim in final.rejected_claims)
    assert final.uncertain_claims
    assert result["answer"] == conditional["answer"]
    assert "一定更安全" not in result["answer"]
    assert final.decision.confidence <= 0.65
    assert context.challenges


def test_ask_user_maps_to_existing_clarify_semantics(monkeypatch) -> None:
    task_id = "task-r03f-ask"
    data, _ = _request(task_id, "YSZ:Dy³⁺ 和 YAG:Dy³⁺ 哪个更好？")
    answer = "没有评价标准无法直接比较。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(answer, task_id)],
        [_review(task_id, answer, "needs_review", "缺少评价目标")],
    )

    final = _final(result)
    assert final.decision.decision_type is DecisionType.ASK_USER
    assert result["answer"] == ""
    assert result["requires_confirmation"] is True
    assert result["action_type"] == "clarify"
    assert result["clarify"]["options"]


@pytest.mark.parametrize(
    ("query", "expected_mode", "expected_type"),
    [
        ("Dy³⁺ 的蓝光发射波长是多少？", TaskMode.FACT_FIND, DecisionType.ANSWER),
        ("为什么 Dy³⁺ 会发生浓度猝灭？", TaskMode.EXPLAIN, DecisionType.ANSWER),
        ("如何设计实验验证 Dy³⁺ 基质的热猝灭机制？", TaskMode.RESEARCH_GUIDE, DecisionType.LEARNING_GUIDANCE),
    ],
)
def test_task_modes_produce_bounded_distinct_decisions(
    monkeypatch, query, expected_mode, expected_type
) -> None:
    task_id = f"task-r03f-{expected_mode.value.lower()}"
    data, _ = _request(task_id, query)
    answer = "已审核结论。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(answer, task_id)],
        [_review_candidate(task_id, answer)],
    )
    final = _final(result)
    assert final.task_mode is expected_mode
    assert final.decision.decision_type is expected_type
    if expected_mode is TaskMode.FACT_FIND:
        assert final.recommended_path == ()


def test_l4_is_only_a_candidate_and_cannot_override_authoritative_fact_decision(monkeypatch) -> None:
    class _L4:
        def next_action_sync(self, *_args, **_kwargs):
            return {
                "action_type": "practice",
                "recommended_path": [{"action": "unsafe_override"}],
                "confidence": 1.0,
            }

    task_id = "task-r03f-l4"
    data, _ = _request(task_id, "Dy³⁺ 的蓝光发射波长是多少？")
    answer = "已审核事实。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(answer, task_id)],
        [_review_candidate(task_id, answer)],
        deps=AgentDependencies(decision_engine=_L4()),
    )
    final = _final(result)
    assert final.decision.decision_type is DecisionType.ANSWER
    assert final.recommended_path == ()
    assert result["action_type"] == "answer"
    assert result["recommended_path"] == []


def test_final_result_is_private_non_serialized_and_not_a_task_state_authority(monkeypatch) -> None:
    task_id = "task-r03f-private"
    data, _ = _request(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    answer = "已审核答案。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(answer, task_id)],
        [_review_candidate(task_id, answer)],
    )
    final = _final(result)
    public_json = json.dumps(dict(result), ensure_ascii=False, default=str)

    assert "FinalCollaborationResult" not in public_json
    assert "accepted_claims" not in result
    assert "rejected_claims" not in result
    assert "collaboration_context" not in result
    assert not hasattr(final, "__dict__")
    assert final.completion_eligibility is True
    assert task_state_runtime.get_task_state(data["task_context"]) == "ANSWERING"
