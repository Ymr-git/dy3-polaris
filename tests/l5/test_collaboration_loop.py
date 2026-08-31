"""R-03E authoritative Reviewer Challenge / Resolution loop tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_contracts import (
    Challenge,
    ChallengeSeverity,
    ChallengeType,
    ResolutionAction,
)
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.interaction_recorder import InteractionRecorder
from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5.task_understanding import understand_task
from tests.l5.test_private_runtime_carrier import (
    _answer_correlation,
    _review_candidate,
    _selected_generation,
)


def _input(task_id: str, query: str):
    task_context = task_state_runtime.create_task_context(task_id)
    task_state_runtime.set_task_state(task_context, "UNDERSTANDING")
    data = {"task_id": task_id, "query": query, "task_context": task_context}
    context = initialize_collaboration_context(
        data,
        intent_resolver=lambda value, **_kwargs: understand_task(value, use_llm=False),
    )
    return data, context


def _review(task_id: str, content: str, verdict: str, reason: str):
    return agent_workers._attach_review_candidate(
        {
            "agent_id": agent_workers.REVIEW_AGENT_ID,
            "status": "completed",
            "verdict": verdict,
            "reason": reason,
            "confidence": 0.8,
            "fact_check": {"passed": verdict == "approved", "checked": 1, "failed": 0},
            "anti_hallucination": {"action": "", "score": 0.8, "hallucination_detected": False},
        },
        {"task_id": task_id},
        content=content,
        producer="agent.quality.review/run_review",
        real_reviewer_executed=True,
    )


def _stub_common(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_workers,
        "run_diagnosis",
        lambda *_args, **_kwargs: {
            "agent_id": agent_workers.DIAGNOSIS_AGENT_ID,
            "status": "completed",
            "weak_kps": [],
            "confidence": 0.9,
        },
    )
    monkeypatch.setattr(agent_workers, "get_recorder", lambda: InteractionRecorder())


def _run_with_sequences(monkeypatch, data, generations, reviews, deps=None):
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


class _Retriever:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[str] = []

    def retrieve(self, query, **_kwargs):
        self.calls.append(query)
        item = {
            "chunk_id": f"chunk-{len(self.calls)}",
            "document_id": "doc-r03e",
            "content": self.content,
            "metadata": {"entity": "Dy3+"},
        }
        return SimpleNamespace(results=[item], scores=[0.5])


class _Reranker:
    def rerank_result(self, _query, result, top_k=10):
        return SimpleNamespace(results=result.results[:top_k], scores=result.scores[:top_k])


def test_approved_review_accepts_without_loop_or_critic(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-accept"
    data, context = _input(task_id, "Dy³⁺主要黄色发射对应什么跃迁？")
    generation = _selected_generation("黄色发射对应 4F9/2→6H13/2。", task_id)
    review = _review_candidate(task_id, generation["answer"], "approved")
    monkeypatch.setattr(agent_workers, "_run_critic_loop", lambda *_a, **_k: pytest.fail("critic must be frozen"))

    result = _run_with_sequences(monkeypatch, data, [generation], [review])

    assert result["self_correction"] is None
    assert context.challenges == []
    assert context.runtime_metadata["r03e_call_counts"] == {"generation": 1, "retrieval": 1, "review": 1}


def test_revise_changes_contribution_without_retrieval(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-revise"
    data, context = _input(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    first = _selected_generation("该机制确定由单一过程造成。", task_id)
    revised = _selected_generation("现有证据支持能级跃迁解释，但具体强度还受基质影响。", task_id)
    reviews = [
        _review(task_id, first["answer"], "needs_review", "把可能机制写成确定事实，需增加限定"),
        _review(task_id, revised["answer"], "approved", "限定后通过"),
    ]

    result = _run_with_sequences(monkeypatch, data, [first, revised], reviews)

    challenge = context.challenges[0]
    assert challenge.challenge_type is ChallengeType.FACT_INFERENCE_CONFUSION
    assert challenge.requested_action is ResolutionAction.REVISE
    assert len(context.retrieval_history) == 1
    assert context.revision_history[0]["action"] is ResolutionAction.REVISE
    generations = [item for item in context.contributions if item.agent_id == agent_workers.GENERATION_AGENT_ID]
    assert generations[-1].parent_contribution_id == generations[-2].contribution_id
    assert generations[-1].revision_reason
    assert result["answer"] == revised["answer"]
    assert _answer_correlation(result).correlation is True


def test_re_retrieve_changes_query_and_versions_evidence(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-reretrieve"
    data, context = _input(task_id, "3000 K是否一定更加健康？")
    first = _selected_generation("3000 K 一定更健康。", task_id)
    revised = _selected_generation("不能仅凭 CCT 判断；还需 SPD、暴露和风险加权。", task_id)
    retriever = _Retriever("SPD blue light hazard spectral weighting exposure CCT limitations")
    deps = AgentDependencies(hybrid_retriever=retriever, reranker=_Reranker())
    reviews = [
        _review(task_id, first["answer"], "needs_review", "安全结论过度"),
        _review(task_id, revised["answer"], "approved", "条件化结论通过"),
    ]

    result = _run_with_sequences(monkeypatch, data, [first, revised], reviews, deps)

    challenge = context.challenges[0]
    assert challenge.challenge_type is ChallengeType.SAFETY_OVERCLAIM
    assert challenge.requested_action is ResolutionAction.RE_RETRIEVE
    assert [entry["version"] for entry in context.retrieval_history] == [1, 2]
    first_queries = tuple(q for p in context.retrieval_history[0]["plans"] for q in p.rewritten_queries)
    second_queries = tuple(q for p in context.retrieval_history[1]["plans"] for q in p.rewritten_queries)
    assert first_queries != second_queries
    assert any("exposure" in query.lower() for query in second_queries)
    assert {pack.version for pack in context.evidence_pool} == {1, 2}
    assert result["answer"] == revised["answer"]


def test_budget_exhaustion_stops_unapproved_loop(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-budget"
    data, context = _input(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    generations = [
        _selected_generation(f"第 {index} 版仍把可能机制写成事实。", task_id)
        for index in range(6)
    ]
    reviews = [
        _review(task_id, item["answer"], "needs_review", "可能机制写成确定事实")
        for item in generations
    ]

    result = _run_with_sequences(monkeypatch, data, generations, reviews)

    assert context.iteration_state["expensive_iterations_used"] == 2
    assert context.challenges[-1].status == "BUDGET_EXHAUSTED"
    assert result["review"]["verdict"] == "needs_review"
    assert context.runtime_metadata["r03e_call_counts"]["generation"] == 3
    assert result["answer"] == ""


def test_rejected_revision_keeps_parent_pair_and_history(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-rejected-revision"
    data, context = _input(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    first = _selected_generation("初稿", task_id)
    rejected = _selected_generation("被拒修订", task_id)
    result = _run_with_sequences(
        monkeypatch,
        data,
        [first, rejected],
        [
            _review(task_id, "初稿", "needs_review", "可能机制写成确定事实"),
            _review(task_id, "被拒修订", "rejected", "unsupported"),
        ],
    )
    private_set = result._contract_candidate
    assert private_set.evidence_candidate is first._contract_candidate
    assert result["pipeline"][2]["verdict"] == "needs_review"
    assert context.revision_history
    assert any(item.get("action") == "REVISION_REJECTED_KEEP_PARENT_PAIR" for item in context.decisions)
    assert private_set.review_candidate.reviewed_answer_identity == first._contract_candidate.answer_identity
    assert _answer_correlation(result).correlation is False
    assert result["answer"] == ""


def test_vague_comparison_asks_user_without_generation_retry(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-ask"
    data, context = _input(task_id, "YSZ:Dy³⁺ 和 YAG:Dy³⁺ 哪个更好？")
    first = _selected_generation("无法脱离指标直接比较。", task_id)
    result = _run_with_sequences(
        monkeypatch,
        data,
        [first],
        [_review(task_id, first["answer"], "needs_review", "缺少评价目标")],
    )
    assert context.challenges[0].requested_action is ResolutionAction.ASK_USER
    assert context.runtime_metadata["r03e_call_counts"]["generation"] == 1
    assert result["review"]["verdict"] == "needs_review"


def test_gq07_knowledge_gap_is_bounded_and_not_approved(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-gq07"
    data, context = _input(task_id, "如何公平比较两种不同 Dy³⁺ 发光材料体系？")
    generations = [
        _selected_generation(f"第 {index} 次复核后仍缺少同条件数据，不能给出排名。", task_id)
        for index in range(6)
    ]
    reviews = [
        _review(task_id, item["answer"], "needs_review", "同条件比较证据不足")
        for item in generations
    ]
    result = _run_with_sequences(monkeypatch, data, generations, reviews)
    assert len(context.revision_history) == 2
    assert context.iteration_state["expensive_iterations_used"] == 2
    assert result["review"]["verdict"] != "approved"
    assert result["answer"] == ""
    assert result["quality_release"]["status"] == "WITHHOLD"


def test_challenge_contract_is_private_slotted_and_not_serialized(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-private"
    data, context = _input(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    first = _selected_generation("可能机制被写成事实。", task_id)
    second = _selected_generation("机制结论带有限定。", task_id)
    result = _run_with_sequences(
        monkeypatch,
        data,
        [first, second],
        [
            _review(task_id, first["answer"], "needs_review", "可能机制写成确定事实"),
            _review(task_id, second["answer"], "approved", "通过"),
        ],
    )
    challenge = context.challenges[0]
    assert isinstance(challenge, Challenge)
    assert not hasattr(challenge, "__dict__")
    with pytest.raises(TypeError):
        json.dumps(challenge)
    public = json.dumps(dict(result), ensure_ascii=False)
    assert "ChallengeType" not in public
    assert "ReviewerChallengeRaised" in public
    assert challenge.challenge_id in public
    assert "ResolutionAction" not in public


def test_contribution_influence_is_observable(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-influence"
    data, _context = _input(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    first = _selected_generation("该结论一定成立。", task_id)
    revised = _selected_generation("在当前证据和条件下，该解释较为合理。", task_id)
    result = _run_with_sequences(
        monkeypatch,
        data,
        [first, revised],
        [
            _review(task_id, first["answer"], "needs_review", "可能推断写成确定事实"),
            _review(task_id, revised["answer"], "approved", "通过"),
        ],
    )
    assert result["answer"] != first["answer"]
    assert "当前证据" in result["answer"]


def test_challenge_and_resolution_enums_are_closed() -> None:
    assert len(ChallengeType) == 11
    assert {item.value for item in ResolutionAction} == {
        "ACCEPT", "REVISE", "RE_RETRIEVE", "ASK_USER", "REJECT"
    }
    assert ChallengeSeverity.CRITICAL.value == "CRITICAL"


def test_task_mode_budget_is_actually_enforced() -> None:
    _, fact = _input("task-r03e-fact-budget", "Dy³⁺主要黄色发射对应什么跃迁？")
    _, evaluate = _input("task-r03e-eval-budget", "3000 K是否一定更加健康？")
    assert fact.can_resolve("REVISE") is True
    fact.consume_resolution_budget("REVISE")
    assert fact.can_resolve("REVISE") is False
    assert evaluate.can_resolve("REVISE") is True
    for _ in range(evaluate.collaboration_budget.review_revision_budget):
        evaluate.consume_resolution_budget("REVISE")
    assert evaluate.can_resolve("REVISE") is False


def test_rejected_review_does_not_retry_or_disguise_approval(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-reject"
    data, context = _input(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    first = _selected_generation("unsupported", task_id)
    result = _run_with_sequences(
        monkeypatch,
        data,
        [first],
        [_review(task_id, first["answer"], "rejected", "防幻觉拒绝")],
    )
    assert context.challenges[0].requested_action is ResolutionAction.REJECT
    assert context.runtime_metadata["r03e_call_counts"]["generation"] == 1
    assert result["review"]["verdict"] == "rejected"


def test_rereview_binds_new_generation_identity_and_active_evidence(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-bind"
    data, context = _input(task_id, "3000 K是否一定更加健康？")
    first = _selected_generation("一定安全", task_id)
    revised = _selected_generation("需结合 SPD 与暴露条件判断", task_id)
    retriever = _Retriever("SPD exposure blue light hazard")
    result = _run_with_sequences(
        monkeypatch,
        data,
        [first, revised],
        [
            _review(task_id, first["answer"], "needs_review", "安全结论过度"),
            _review(task_id, revised["answer"], "approved", "通过"),
        ],
        AgentDependencies(hybrid_retriever=retriever, reranker=_Reranker()),
    )
    generations = [item for item in context.contributions if item.agent_id == agent_workers.GENERATION_AGENT_ID]
    reviews = [item for item in context.contributions if item.agent_id == agent_workers.REVIEW_AGENT_ID]
    assert generations[-1].artifact_identity == reviews[-1].artifact_identity
    assert reviews[-1].parent_contribution_id == reviews[-2].contribution_id
    assert all(pack.version == 2 for pack in context.tool_results["active_evidence_packs"])
    assert _answer_correlation(result).correlation is True


def test_identical_challenge_without_material_progress_stops(monkeypatch) -> None:
    _stub_common(monkeypatch)
    task_id = "task-r03e-progress"
    data, context = _input(task_id, "3000 K是否一定更加健康？")
    answer = "3000 K 一定安全"
    generations = [_selected_generation(answer, task_id) for _ in range(3)]
    reviews = [
        _review(task_id, answer, "needs_review", "安全结论过度")
        for _ in range(3)
    ]
    retriever = _Retriever("same evidence")
    _run_with_sequences(
        monkeypatch,
        data,
        generations,
        reviews,
        AgentDependencies(hybrid_retriever=retriever, reranker=_Reranker()),
    )
    assert context.challenges[-1].status in {"NO_PROGRESS", "BUDGET_EXHAUSTED"}
    assert context.iteration_state["expensive_iterations_used"] <= 2
    assert context.runtime_metadata["r03e_call_counts"]["generation"] <= 3


def test_cc1_fix_action_is_not_mapped_to_approved() -> None:
    class _Pipeline:
        def verify(self, _request):
            return SimpleNamespace(
                action=SimpleNamespace(value="fix"),
                overall_score=0.7,
                hallucination_detected=False,
            )

    result = agent_workers.run_review(
        {"task_id": "task-r03e-fix", "content": "claim", "context_chunks": ["evidence"]},
        AgentDependencies(anti_hallucination_pipeline=_Pipeline()),
    )
    assert result["verdict"] == "needs_review"
    assert result["verdict"] != "approved"
