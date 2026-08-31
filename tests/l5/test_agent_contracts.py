"""R-03C private AgentInput/Contribution and compatibility scheduler tests."""

from __future__ import annotations

import json

import pytest

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_contracts import (
    AgentContribution,
    AgentInput,
    Claim,
    ClaimType,
    RequestedAction,
    agent_for_capability,
    build_agent_input,
    make_contribution,
)
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.interaction_recorder import InteractionRecorder
from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5.task_understanding import TaskMode, understand_task
from tests.l5.test_private_runtime_carrier import (
    _no_critic_adoption,
    _review_candidate,
    _selected_generation,
)


EXPLAIN_QUERY = "为什么Dy³⁺会产生黄蓝双发射？"
EVALUATE_QUERY = "3000 K是否一定更加健康？"
COMPARE_QUERY = "比较 YSZ:Dy³⁺ 和 YAG:Dy³⁺ 的发光表现。"


def _context(query: str, task_id: str = "task-r03c"):
    data = {"task_id": task_id, "query": query, "learner_id": "learner-r03c"}
    return initialize_collaboration_context(
        data,
        intent_resolver=lambda value, **_kwargs: understand_task(
            value, use_llm=False
        ),
    )


@pytest.mark.parametrize(
    ("capability", "agent_id"),
    [
        ("learner_context_analysis", agent_workers.DIAGNOSIS_AGENT_ID),
        ("domain_knowledge_generation", agent_workers.GENERATION_AGENT_ID),
        ("scientific_review", agent_workers.REVIEW_AGENT_ID),
        ("learning_guidance", agent_workers.GUIDANCE_AGENT_ID),
    ],
)
def test_capability_routes_to_only_existing_agents(capability, agent_id) -> None:
    assert agent_for_capability(capability) == agent_id


def test_unknown_capability_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown required_capability"):
        agent_for_capability("invented_capability")


def test_agent_input_is_subtask_scoped_private_and_slotted() -> None:
    context = _context(EXPLAIN_QUERY)
    value = build_agent_input(context, agent_workers.DIAGNOSIS_AGENT_ID)

    assert isinstance(value, AgentInput)
    assert value.task_id == context.task_id
    assert value.subtask.subtask_id == "establish_learner_context"
    assert value.user_query == EXPLAIN_QUERY
    assert value.intent is context.intent_result
    assert all(item is not context for item in value.prior_contributions)
    assert value.task_mode is TaskMode.EXPLAIN
    assert not hasattr(value, "__dict__")
    assert not isinstance(value, dict)
    with pytest.raises(TypeError):
        json.dumps(value)


def test_contribution_binds_task_subtask_and_context() -> None:
    context = _context(EXPLAIN_QUERY)
    agent_input = build_agent_input(context, agent_workers.DIAGNOSIS_AGENT_ID)
    assert agent_input is not None
    context.begin_subtasks((agent_input.subtask.subtask_id,))
    contribution = make_contribution(
        context,
        agent_input,
        conclusion="需先理解能级与跃迁。",
        claims=(
            Claim(
                "claim-prerequisite",
                "学习者需要能级跃迁前置知识",
                ClaimType.INFERENCE,
                confidence=0.8,
            ),
        ),
        requested_actions=(RequestedAction.ACCEPT,),
        confidence=0.8,
    )
    context.record_contribution(contribution)
    context.complete_subtasks((agent_input.subtask.subtask_id,))

    assert isinstance(contribution, AgentContribution)
    assert contribution.task_id == context.task_id
    assert contribution.subtask_id == agent_input.subtask.subtask_id
    assert context.contributions == [contribution]
    assert context.subtask_states[agent_input.subtask.subtask_id] == "completed"


def test_contribution_rejects_wrong_task_binding() -> None:
    context = _context(EXPLAIN_QUERY)
    agent_input = build_agent_input(context, agent_workers.DIAGNOSIS_AGENT_ID)
    assert agent_input is not None
    contribution = make_contribution(
        context, agent_input, conclusion="x", confidence=0.5
    )
    wrong = AgentContribution(
        contribution.contribution_id,
        "task-other",
        contribution.subtask_id,
        contribution.agent_id,
        contribution.conclusion,
        contribution.claims,
        contribution.evidence_refs,
        contribution.assumptions,
        contribution.uncertainty,
        contribution.challenges,
        contribution.requested_actions,
        contribution.tool_usage,
        contribution.confidence,
        contribution.status,
        contribution.produced_at,
        contribution.sequence,
        contribution.covered_subtask_ids,
        contribution.artifact_identity,
    )
    with pytest.raises(ValueError, match="task_id mismatch"):
        context.record_contribution(wrong)


def test_diagnosis_contribution_observably_changes_generation_input() -> None:
    context = _context(EXPLAIN_QUERY)
    before = build_agent_input(context, agent_workers.GENERATION_AGENT_ID)
    assert before is not None
    assert before.prior_contributions == ()

    diagnosis_input = build_agent_input(context, agent_workers.DIAGNOSIS_AGENT_ID)
    assert diagnosis_input is not None
    context.begin_subtasks((diagnosis_input.subtask.subtask_id,))
    diagnosis = make_contribution(
        context,
        diagnosis_input,
        conclusion="学习者缺少能级前置知识",
        assumptions=("missing prerequisite: energy levels",),
        confidence=0.8,
    )
    context.record_contribution(diagnosis)
    context.complete_subtasks((diagnosis_input.subtask.subtask_id,))
    after = build_agent_input(context, agent_workers.GENERATION_AGENT_ID)

    assert after is not None
    assert diagnosis in after.prior_contributions
    assert after.prior_contributions != before.prior_contributions


def test_compare_branches_become_ready_together_without_parallel_claim() -> None:
    context = _context(COMPARE_QUERY)
    diagnosis_input = build_agent_input(context, agent_workers.DIAGNOSIS_AGENT_ID)
    assert diagnosis_input is not None
    context.begin_subtasks((diagnosis_input.subtask.subtask_id,))
    context.complete_subtasks((diagnosis_input.subtask.subtask_id,))
    context.begin_subtasks(("define_comparison_criteria",))
    context.complete_subtasks(("define_comparison_criteria",))

    ready = {item.subtask_id for item in context.ready_subtasks()}
    assert {
        "collect_first_material_evidence",
        "collect_second_material_evidence",
    }.issubset(ready)
    assert context.runtime_metadata["current_execution_model"] == (
        "compatibility_scheduler"
    )
    assert "parallel" not in context.runtime_metadata


def test_all_planned_capabilities_are_deterministically_routable() -> None:
    for query in (
        "Dy³⁺黄光跃迁是哪一个？",
        EXPLAIN_QUERY,
        COMPARE_QUERY,
        EVALUATE_QUERY,
        "如何研究Dy³⁺浓度猝灭？",
    ):
        context = _context(query, task_id=f"task-{len(query)}")
        for item in context.task_plan.subtasks:
            assert agent_for_capability(item.required_capability) in {
                agent_workers.DIAGNOSIS_AGENT_ID,
                agent_workers.GENERATION_AGENT_ID,
                agent_workers.REVIEW_AGENT_ID,
                agent_workers.GUIDANCE_AGENT_ID,
            }


def test_subtask_state_history_contains_each_required_transition() -> None:
    context = _context(EXPLAIN_QUERY)
    agent_input = build_agent_input(context, agent_workers.DIAGNOSIS_AGENT_ID)
    assert agent_input is not None
    context.begin_subtasks((agent_input.subtask.subtask_id,))
    context.complete_subtasks((agent_input.subtask.subtask_id,))
    transitions = [
        (before, after)
        for subtask_id, before, after, _timestamp in context.runtime_metadata[
            "subtask_state_history"
        ]
        if subtask_id == agent_input.subtask.subtask_id
    ]
    assert transitions == [
        ("pending", "ready"),
        ("ready", "running"),
        ("running", "completed"),
    ]


def test_evaluate_generation_contract_preserves_assumption_and_uncertainty() -> None:
    context = _context(EVALUATE_QUERY)
    diagnosis_input = agent_workers._start_contract_agent(
        context, agent_workers.DIAGNOSIS_AGENT_ID
    )
    assert diagnosis_input is not None
    diagnosis_result = {
        "summary": "中等学习起点",
        "weak_kps": [],
        "level": "中",
        "ability": {"theta": 0.0},
        "confidence": 0.8,
    }
    diagnosis = agent_workers._adapt_diagnosis_contribution(
        context, diagnosis_input, diagnosis_result
    )
    agent_workers._finish_contract_agent(context, diagnosis_input, diagnosis)
    generation_input = agent_workers._start_contract_agent(
        context, agent_workers.GENERATION_AGENT_ID
    )
    assert generation_input is not None
    generation = _selected_generation(
        "3000 K不能单独决定健康性。",
        context.task_id,
    )
    contribution = agent_workers._adapt_generation_contribution(
        context, generation_input, generation
    )

    assert contribution.claims[0].claim_type is ClaimType.INFERENCE
    assert contribution.evidence_refs
    assert "absolute claim depends on unstated conditions" in contribution.uncertainty


def test_generation_contribution_reuses_private_answer_identity() -> None:
    context = _context(EXPLAIN_QUERY)
    diagnosis_input = agent_workers._start_contract_agent(
        context, agent_workers.DIAGNOSIS_AGENT_ID
    )
    assert diagnosis_input is not None
    diagnosis = agent_workers._adapt_diagnosis_contribution(
        context,
        diagnosis_input,
        {"summary": "x", "ability": {}, "confidence": 0.5},
    )
    agent_workers._finish_contract_agent(context, diagnosis_input, diagnosis)
    generation_input = agent_workers._start_contract_agent(
        context, agent_workers.GENERATION_AGENT_ID
    )
    assert generation_input is not None
    generation = _selected_generation("identity answer", context.task_id)
    contribution = agent_workers._adapt_generation_contribution(
        context, generation_input, generation
    )
    assert contribution.artifact_identity == (
        generation._contract_candidate.answer_identity
    )
    assert contribution.claims[0].claim_id != contribution.artifact_identity


@pytest.mark.parametrize(
    ("verdict", "action"),
    [
        ("approved", RequestedAction.ACCEPT),
        ("needs_review", RequestedAction.REQUEST_REVISION),
        ("rejected", RequestedAction.REFUSE_CONCLUSION),
        ("skipped", RequestedAction.DECLARE_UNCERTAINTY),
    ],
)
def test_review_adapter_records_raw_decision_as_controlled_action(
    verdict, action
) -> None:
    context = _context(EXPLAIN_QUERY)
    diagnosis_input = agent_workers._start_contract_agent(
        context, agent_workers.DIAGNOSIS_AGENT_ID
    )
    assert diagnosis_input is not None
    diagnosis = agent_workers._adapt_diagnosis_contribution(
        context,
        diagnosis_input,
        {"summary": "x", "ability": {}, "confidence": 0.5},
    )
    agent_workers._finish_contract_agent(context, diagnosis_input, diagnosis)
    generation_input = agent_workers._start_contract_agent(
        context, agent_workers.GENERATION_AGENT_ID
    )
    assert generation_input is not None
    generation = _selected_generation("review me", context.task_id)
    generation_fact = agent_workers._adapt_generation_contribution(
        context, generation_input, generation
    )
    agent_workers._finish_contract_agent(
        context, generation_input, generation_fact
    )
    review_input = agent_workers._start_contract_agent(
        context, agent_workers.REVIEW_AGENT_ID
    )
    assert review_input is not None
    review = _review_candidate(context.task_id, "review me", verdict)
    contribution = agent_workers._adapt_review_contribution(
        context, review_input, review
    )
    assert contribution.requested_actions == (action,)
    assert contribution.artifact_identity == (
        review._contract_candidate.reviewed_answer_identity
    )


def test_real_guidance_uses_structured_edges_and_completes_plan(monkeypatch) -> None:
    task_id = "task-r03c-runtime"
    captured: dict[str, AgentInput] = {}
    observed_contexts = []
    original_start = agent_workers._start_contract_agent

    def observe_start(context, agent_id):
        observed_contexts.append(context)
        value = original_start(context, agent_id)
        if value is not None:
            captured[agent_id] = value
        return value

    def diagnosis(input_data, _deps):
        value = input_data.get("_agent_input")
        assert isinstance(value, AgentInput)
        return {
            "agent_id": agent_workers.DIAGNOSIS_AGENT_ID,
            "status": "completed",
            "learner_id": "learner-r03c",
            "ability": {"theta": -0.5},
            "weak_kps": ["energy-level"],
            "level": "beginner",
            "summary": "需要能级前置知识",
            "confidence": 0.8,
        }

    generation_calls = 0

    def generation(input_data, _deps, **_kwargs):
        nonlocal generation_calls
        generation_calls += 1
        value = input_data.get("_agent_input")
        assert isinstance(value, AgentInput)
        assert input_data["learner_level"] == "beginner"
        assert any(
            item.agent_id == agent_workers.DIAGNOSIS_AGENT_ID
            for item in value.prior_contributions
        )
        return _selected_generation(
            "Dy³⁺黄蓝发射来自激发态向不同低能级的跃迁。",
            task_id,
        )

    def review(input_data, _deps):
        value = input_data.get("_agent_input")
        assert isinstance(value, AgentInput)
        generation_facts = [
            item
            for item in value.prior_contributions
            if item.agent_id == agent_workers.GENERATION_AGENT_ID
        ]
        assert generation_facts
        assert input_data["content"] == generation_facts[-1].conclusion
        assert input_data["context_chunks"] == agent_workers._review_evidence_texts(
            value,
            generation_facts[-1],
        )
        return _review_candidate(task_id, input_data["content"], "approved")

    monkeypatch.setattr(agent_workers, "_start_contract_agent", observe_start)
    monkeypatch.setattr(agent_workers, "run_diagnosis", diagnosis)
    monkeypatch.setattr(agent_workers, "_run_multi_candidate_generation", generation)
    monkeypatch.setattr(agent_workers, "run_review", review)
    monkeypatch.setattr(agent_workers, "_run_critic_loop", _no_critic_adoption)
    monkeypatch.setattr(agent_workers, "get_recorder", lambda: InteractionRecorder())
    task_context = task_state_runtime.create_task_context(task_id)
    task_state_runtime.set_task_state(task_context, "UNDERSTANDING")
    result = agent_workers.run_guidance(
        {
            "task_id": task_id,
            "task_context": task_context,
            "query": EXPLAIN_QUERY,
            "learner_id": "learner-r03c",
        },
        AgentDependencies(),
    )

    generation_input = captured[agent_workers.GENERATION_AGENT_ID]
    review_input = captured[agent_workers.REVIEW_AGENT_ID]
    guidance_input = captured[agent_workers.GUIDANCE_AGENT_ID]
    assert any(
        item.agent_id == agent_workers.DIAGNOSIS_AGENT_ID
        for item in generation_input.prior_contributions
    )
    assert any(
        item.agent_id == agent_workers.GENERATION_AGENT_ID
        for item in review_input.prior_contributions
    )
    assert {
        item.agent_id for item in guidance_input.prior_contributions
    }.issuperset(
        {
            agent_workers.DIAGNOSIS_AGENT_ID,
            agent_workers.GENERATION_AGENT_ID,
            agent_workers.REVIEW_AGENT_ID,
        }
    )
    context = observed_contexts[0]
    assert all(item is context for item in observed_contexts)
    assert context.task_id == task_id
    assert set(context.subtask_states.values()) == {"completed"}
    assert [item.agent_id for item in context.contributions] == [
        agent_workers.DIAGNOSIS_AGENT_ID,
        agent_workers.GENERATION_AGENT_ID,
        agent_workers.REVIEW_AGENT_ID,
        agent_workers.GUIDANCE_AGENT_ID,
    ]
    assert generation_calls == 1
    assert result["answer"]
    serialized = json.dumps(result, ensure_ascii=False)
    for private_name in (
        "AgentInput",
        "ClaimType",
        "RequestedAction",
        "_agent_input",
        "subtask_id",
    ):
        assert private_name not in serialized
    event_text = json.dumps(
        task_state_runtime.get_task_events(task_context), ensure_ascii=False
    )
    assert "AgentInput" not in event_text
    assert "_agent_input" not in event_text


def test_claim_types_remain_distinct() -> None:
    assert {item.value for item in ClaimType} == {
        "FACT",
        "INFERENCE",
        "RECOMMENDATION",
        "UNCERTAIN",
    }


def test_contract_enum_is_closed_and_does_not_execute_actions() -> None:
    assert {item.value for item in RequestedAction} == {
        "ACCEPT",
        "USE_EXISTING_EVIDENCE",
        "REQUEST_RETRIEVAL",
        "REQUEST_RE_RANK",
        "REQUEST_REVISION",
        "CHALLENGE",
        "ASK_USER",
        "DECLARE_UNCERTAINTY",
        "REFUSE_CONCLUSION",
    }
    assert not hasattr(RequestedAction, "execute")
