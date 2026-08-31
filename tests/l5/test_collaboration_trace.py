"""R-03G real collaboration trace, evaluation, and safe projection tests."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_contracts import CollaborationTrace, DecisionType
from dy3_polaris.l5.agent_workers import AgentDependencies, _MultiAgentEvaluation
from dy3_polaris.l5.task_understanding import TaskMode
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_collaboration_loop import _Retriever, _Reranker, _review
from tests.l5.test_guidance_decision import _final, _request, _run
from tests.l5.test_private_runtime_carrier import (
    _review_candidate,
    _selected_generation,
    _stub_guidance_edges,
)


def _trace(result) -> CollaborationTrace:
    value = result._contract_candidate.collaboration_trace
    assert isinstance(value, CollaborationTrace)
    return value


def _evaluation(result) -> _MultiAgentEvaluation:
    value = result._contract_candidate.multi_agent_evaluation
    assert isinstance(value, _MultiAgentEvaluation)
    return value


def test_trace_is_real_monotonic_and_task_bound(monkeypatch) -> None:
    task_id = "task-r03g-basic"
    data, context = _request(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    answer = "Dy³⁺ 的黄蓝发射来自已审核的能级跃迁解释。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(answer, task_id)],
        [_review_candidate(task_id, answer)],
    )
    trace = _trace(result)

    assert trace.task_id == task_id
    assert [event.sequence for event in trace.events] == list(
        range(1, len(trace.events) + 1)
    )
    assert all(event.task_id == task_id for event in trace.events)
    assert [event.timestamp for event in trace.events] == sorted(
        event.timestamp for event in trace.events
    )
    contribution_ids = {item.contribution_id for item in context.contributions}
    traced_refs = {ref for event in trace.events for ref in event.artifact_refs}
    assert contribution_ids <= traced_refs
    assert {"TASK_UNDERSTOOD", "TASK_DECOMPOSED", "GUIDANCE_DECIDED"} <= {
        event.event_type for event in trace.events
    }


def test_challenge_revision_and_reretrieval_have_causal_parents(monkeypatch) -> None:
    task_id = "task-r03g-causal"
    data, context = _request(task_id, "3000 K 的 Dy³⁺ 白光是否一定更健康？")
    first = _selected_generation("3000 K 一定更安全。", task_id)
    revised = _selected_generation(
        "不能仅凭 CCT 判断；还需 SPD、暴露和风险加权。", task_id
    )
    deps = AgentDependencies(
        hybrid_retriever=_Retriever(
            "SPD blue light hazard exposure CCT limitations Dy3+"
        ),
        reranker=_Reranker(),
    )
    result = _run(
        monkeypatch,
        data,
        [first, revised],
        [
            _review(task_id, first["answer"], "needs_review", "安全结论过度"),
            _review(task_id, revised["answer"], "approved", "条件化结论通过"),
        ],
        deps=deps,
    )
    trace = _trace(result)
    by_id = {event.event_id: event for event in trace.events}
    challenge = next(event for event in trace.events if event.event_type == "CHALLENGE_RAISED")
    reretrieve = next(
        event for event in trace.events if event.event_type == "RE_RETRIEVAL_REQUESTED"
    )
    revised_event = next(
        event
        for event in trace.events
        if event.event_type == "CONTRIBUTION_REVISED"
        and event.actor == agent_workers.GENERATION_AGENT_ID
    )

    assert challenge.caused_by in {
        item.contribution_id
        for item in context.contributions
        if item.agent_id == agent_workers.GENERATION_AGENT_ID
    }
    assert by_id[challenge.parent_event_id].actor == agent_workers.GENERATION_AGENT_ID
    assert reretrieve.caused_by == challenge.artifact_refs[0]
    assert reretrieve.parent_event_id == challenge.event_id
    assert revised_event.caused_by == challenge.artifact_refs[0]
    assert revised_event.parent_event_id == challenge.event_id


def test_evidence_pack_v2_and_guidance_trace_to_real_sources(monkeypatch) -> None:
    task_id = "task-r03g-evidence-v2"
    data, context = _request(task_id, "3000 K 的 Dy³⁺ 白光是否一定更健康？")
    first = _selected_generation("3000 K 一定更健康。", task_id)
    second = _selected_generation("需结合 SPD 和暴露条件评价。", task_id)
    result = _run(
        monkeypatch,
        data,
        [first, second],
        [
            _review(task_id, first["answer"], "needs_review", "安全结论过度"),
            _review(task_id, second["answer"], "approved", "条件化结论通过"),
        ],
        deps=AgentDependencies(
            hybrid_retriever=_Retriever("SPD exposure blue-light hazard CCT"),
            reranker=_Reranker(),
        ),
    )
    trace = _trace(result)
    by_id = {event.event_id: event for event in trace.events}
    pack_v2 = next(
        event
        for event in trace.events
        if event.event_type == "EVIDENCE_RETRIEVED"
        and any(ref.endswith(":v2") for ref in event.artifact_refs)
    )
    guidance = next(event for event in trace.events if event.event_type == "GUIDANCE_DECIDED")

    assert pack_v2.parent_event_id
    assert by_id[pack_v2.parent_event_id].event_type == "RETRIEVAL_REQUESTED"
    assert guidance.caused_by == _final(result).review.contribution_id
    assert by_id[guidance.parent_event_id].actor == agent_workers.REVIEW_AGENT_ID
    assert [item["version"] for item in context.retrieval_history] == [1, 2]
    queries_v1 = {
        q for plan in context.retrieval_history[0]["plans"] for q in plan.rewritten_queries
    }
    queries_v2 = {
        q for plan in context.retrieval_history[1]["plans"] for q in plan.rewritten_queries
    }
    assert queries_v1 != queries_v2


def test_public_projection_contains_no_private_or_fake_agent_data(monkeypatch) -> None:
    task_id = "task-r03g-safe"
    data, _ = _request(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    answer = "已审核答案。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(answer, task_id)],
        [_review_candidate(task_id, answer)],
    )
    public = json.dumps(dict(result), ensure_ascii=False, default=str)

    for forbidden in (
        "CollaborationTrace",
        "AgentInput",
        "CollaborationContext",
        "FinalCollaborationResult",
        "_contract_candidate",
        "readiness",
        "system prompt",
        "api_key",
    ):
        assert forbidden not in public
    actors = {
        item.get("agent_id")
        for item in result["agent_trace"]
    } | {
        step.get("agent")
        for line in result["collab_lines"]
        for step in line.get("steps", ())
    }
    assert "debate.pro" not in actors
    assert "debate.con" not in actors
    assert "debate.vote" not in actors
    assert "agent.adjudicator" not in actors
    assert not any("parallel" in line.lower() or "并行" in line for line in public.splitlines())
    assert result["broadcast_events"] == []
    assert result["reasoning_loop"] is None


@pytest.mark.parametrize(
    ("query", "mode"),
    [
        ("Dy³⁺ 的黄色发射对应什么跃迁？", TaskMode.FACT_FIND),
        ("为什么 Dy³⁺ 会产生黄蓝双发射？", TaskMode.EXPLAIN),
        ("如何比较 YSZ:Dy³⁺ 和 YAG:Dy³⁺ 的量子效率？", TaskMode.COMPARE),
        ("3000 K 的 Dy³⁺ 白光是否更健康？", TaskMode.EVALUATE),
        ("如何设计实验验证 Dy³⁺ 浓度猝灭机制？", TaskMode.RESEARCH_GUIDE),
    ],
)
def test_task_modes_expose_real_plan_path_signatures(monkeypatch, query, mode) -> None:
    task_id = f"task-r03g-{mode.value.lower()}"
    data, _ = _request(task_id, query)
    answer = "模式对应的已审核答案。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(answer, task_id)],
        [_review_candidate(task_id, answer)],
    )
    trace = _trace(result)
    assert trace.task_mode is mode
    assert trace.path_signature
    assert any(item.startswith("READY:") for item in trace.path_signature)
    assert trace.path_signature[-1].startswith("GUIDE")


def test_fact_find_is_shorter_than_evaluate_challenge_path(monkeypatch) -> None:
    fact_id = "task-r03g-short"
    fact_data, _ = _request(fact_id, "Dy³⁺ 的黄色发射对应什么跃迁？")
    fact_answer = "4F9/2→6H13/2。"
    fact = _run(
        monkeypatch,
        fact_data,
        [_selected_generation(fact_answer, fact_id)],
        [_review_candidate(fact_id, fact_answer)],
    )

    eval_id = "task-r03g-long"
    eval_data, _ = _request(eval_id, "3000 K 的 Dy³⁺ 白光是否一定更健康？")
    first = _selected_generation("3000 K 一定更健康。", eval_id)
    second = _selected_generation("还需 SPD 与暴露条件。", eval_id)
    evaluated = _run(
        monkeypatch,
        eval_data,
        [first, second],
        [
            _review(eval_id, first["answer"], "needs_review", "安全结论过度"),
            _review(eval_id, second["answer"], "approved", "修订通过"),
        ],
        deps=AgentDependencies(
            hybrid_retriever=_Retriever("SPD exposure CCT risk"),
            reranker=_Reranker(),
        ),
    )

    assert len(_trace(fact).events) < len(_trace(evaluated).events)
    assert "CHALLENGE" not in _trace(fact).path_signature
    assert "CHALLENGE" in _trace(evaluated).path_signature
    assert "RETRIEVE_AGAIN" in _trace(evaluated).path_signature
    assert _evaluation(fact).costs["generation_count"] == 1
    assert _evaluation(evaluated).costs["generation_count"] == 2


def test_evaluation_proves_diagnosis_reviewer_and_evidence_influence(monkeypatch) -> None:
    task_id = "task-r03g-influence"
    data, _ = _request(task_id, "3000 K 的 Dy³⁺ 白光是否一定更健康？")
    first = _selected_generation("3000 K 一定更健康。", task_id)
    second = _selected_generation("需结合 SPD 与暴露条件。", task_id)
    result = _run(
        monkeypatch,
        data,
        [first, second],
        [
            _review(task_id, first["answer"], "needs_review", "安全结论过度"),
            _review(task_id, second["answer"], "approved", "修订通过"),
        ],
        level="advanced",
        deps=AgentDependencies(
            hybrid_retriever=_Retriever("SPD exposure blue-light risk"),
            reranker=_Reranker(),
        ),
    )
    evaluation = _evaluation(result)

    assert evaluation.collaboration_intelligence["diagnosis_influence"] is True
    assert evaluation.collaboration_intelligence["reviewer_influence"] is True
    assert evaluation.collaboration_intelligence["evidence_influence"] is True
    assert evaluation.collaboration_intelligence["challenge_target_validity"] == 1.0
    assert evaluation.collaboration_intelligence["budget_compliant"] is True
    assert evaluation.trust["unsupported_claim_rejected"] is True
    assert evaluation.trust["uncertainty_preserved"] is True
    assert evaluation.educational_intelligence["learner_depth"] == "advanced"


def test_knowledge_gap_trace_is_honest_and_does_not_rank(monkeypatch) -> None:
    task_id = "task-r03g-gq07"
    data, _ = _request(task_id, "如何公平比较两种不同 Dy³⁺ 发光材料体系？")
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

    assert _final(result).decision.decision_type is DecisionType.KNOWLEDGE_GAP
    assert result["answer"] == ""
    assert result["quality_release"]["status"] == "WITHHOLD"
    assert _evaluation(result).trust["knowledge_gap_honest"] is True
    assert any(
        event.event_type == "CHALLENGE_RAISED" for event in _trace(result).events
    )


def test_private_trace_and_evaluation_are_slotted_and_not_mapping_keys(monkeypatch) -> None:
    task_id = "task-r03g-private"
    data, _ = _request(task_id, "为什么 Dy³⁺ 会产生黄蓝双发射？")
    answer = "已审核答案。"
    result = _run(
        monkeypatch,
        data,
        [_selected_generation(answer, task_id)],
        [_review_candidate(task_id, answer)],
    )

    trace = _trace(result)
    evaluation = _evaluation(result)
    assert not hasattr(trace, "__dict__")
    assert not hasattr(trace.events[0], "__dict__")
    assert not hasattr(evaluation, "__dict__")
    assert "collaboration_trace" not in result
    assert "multi_agent_evaluation" not in result


def test_five_modes_have_distinct_trace_signatures(monkeypatch) -> None:
    cases = (
        ("fact", "Dy³⁺ 的黄色发射对应什么跃迁？"),
        ("explain", "为什么 Dy³⁺ 会产生黄蓝双发射？"),
        ("compare", "如何比较 YSZ:Dy³⁺ 和 YAG:Dy³⁺ 的量子效率？"),
        ("evaluate", "3000 K 的 Dy³⁺ 白光是否更健康？"),
        ("research", "如何设计实验验证 Dy³⁺ 浓度猝灭机制？"),
    )
    signatures = []
    for label, query in cases:
        task_id = f"task-r03g-signature-{label}"
        data, _ = _request(task_id, query)
        answer = f"{label} reviewed answer"
        result = _run(
            monkeypatch,
            data,
            [_selected_generation(answer, task_id)],
            [_review_candidate(task_id, answer)],
        )
        signatures.append(_trace(result).path_signature)
    assert len(set(signatures)) == 5


def test_api_query_uses_current_request_safe_trace_projection(monkeypatch) -> None:
    _stub_guidance_edges(monkeypatch)

    def selected(input_data, *_args, **_kwargs):
        return _selected_generation(
            "R-03G API reviewed answer",
            str(input_data.get("task_id") or ""),
        )

    monkeypatch.setattr(agent_workers, "_run_multi_candidate_generation", selected)
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda input_data, _deps: _review_candidate(
            str(input_data.get("task_id") or ""),
            str(input_data.get("content") or ""),
        ),
    )
    client = TestClient(UnifiedApp.create_full_app_builder().create_app())
    response = client.post(
        "/api/query",
        json={"query": "为什么 Dy³⁺ 会产生黄蓝双发射？"},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["agent_trace"]
    assert any(
        item["agent_id"] == agent_workers.GUIDANCE_AGENT_ID
        for item in data["agent_trace"]
    )
    serialized = json.dumps(data, ensure_ascii=False)
    assert "CollaborationTrace" not in serialized
    assert "_contract_candidate" not in serialized
    assert "debate.pro" not in serialized
    assert "agent.adjudicator" not in serialized
