"""R-03B mode-specific TaskPlan and private CollaborationContext tests."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.collaboration_context import (
    CollaborationContext,
    Subtask,
    TaskPlan,
    build_task_plan,
    initialize_collaboration_context,
)
from dy3_polaris.l5.task_understanding import IntentResult, TaskMode, understand_task
from dy3_polaris.l5.unified_app import UnifiedApp


PLAN_CASES = (
    (
        TaskMode.FACT_FIND,
        "Dy³⁺主要黄色发射对应哪一能级跃迁？",
    ),
    (
        TaskMode.EXPLAIN,
        "为什么Dy³⁺会产生黄蓝双发射？",
    ),
    (
        TaskMode.COMPARE,
        "比较 YSZ:Dy³⁺ 和 YAG:Dy³⁺ 在温度传感中的发光表现。",
    ),
    (
        TaskMode.EVALUATE,
        "3000 K是否一定更加健康？",
    ),
    (
        TaskMode.RESEARCH_GUIDE,
        "如果我要研究Dy³⁺浓度猝灭，下一步应该重点看什么？",
    ),
)


def _plan(mode: TaskMode, query: str, task_id: str = "task-r03b") -> TaskPlan:
    intent = understand_task(query, use_llm=False)
    assert intent.task_mode is mode
    return build_task_plan(task_id, query, intent)


@pytest.mark.parametrize(("mode", "query"), PLAN_CASES)
def test_each_task_mode_builds_a_valid_bound_plan(mode: TaskMode, query: str) -> None:
    plan = _plan(mode, query, task_id=f"task-{mode.value.lower()}")

    assert plan.task_id == f"task-{mode.value.lower()}"
    assert plan.task_mode is mode
    assert plan.intent.task_mode is mode
    assert plan.subtasks
    assert len(plan.topological_order()) == len(plan.subtasks)
    assert plan.required_capabilities
    assert plan.collaboration_budget.retrieval_budget >= 1


def test_five_modes_produce_materially_different_subtask_graphs() -> None:
    plans = [_plan(mode, query) for mode, query in PLAN_CASES]
    fingerprints = {
        (
            tuple(subtask.type for subtask in plan.subtasks),
            plan.dependencies,
            plan.collaboration_budget,
        )
        for plan in plans
    }

    assert len(fingerprints) == len(PLAN_CASES)
    assert [len(plan.subtasks) for plan in plans] == [4, 5, 7, 7, 6]


def test_subtasks_describe_problems_not_agent_invocations() -> None:
    forbidden = ("agent.", "run agent", "运行agent", "运行 agent", "调用agent", "调用 agent")
    for mode, query in PLAN_CASES:
        plan = _plan(mode, query)
        for subtask in plan.subtasks:
            searchable = f"{subtask.type} {subtask.goal}".lower()
            assert not any(token in searchable for token in forbidden)


def test_compare_graph_expresses_two_evidence_branches_before_synthesis() -> None:
    plan = _plan(TaskMode.COMPARE, PLAN_CASES[2][1])
    first = plan.get_subtask("collect_first_material_evidence")
    second = plan.get_subtask("collect_second_material_evidence")
    synthesis = plan.get_subtask("synthesize_comparison")

    assert first is not None and second is not None and synthesis is not None
    assert first.dependencies == ("define_comparison_criteria",)
    assert second.dependencies == ("define_comparison_criteria",)
    assert {
        "collect_first_material_evidence",
        "collect_second_material_evidence",
    }.issubset(set(synthesis.dependencies))
    assert "YSZ:Dy³⁺" in first.goal
    assert "YAG:Dy³⁺" in second.goal


def test_budget_is_mode_specific_and_evaluation_has_highest_iteration_allowance() -> None:
    budgets = {
        mode: _plan(mode, query).collaboration_budget
        for mode, query in PLAN_CASES
    }

    assert budgets[TaskMode.FACT_FIND].level == "low"
    assert budgets[TaskMode.FACT_FIND].max_expensive_iterations == 1
    assert budgets[TaskMode.EXPLAIN].level == "low-medium"
    assert budgets[TaskMode.COMPARE].level == "medium"
    assert budgets[TaskMode.EVALUATE].level == "medium-high"
    assert budgets[TaskMode.EVALUATE].max_expensive_iterations == 20
    assert all(item.global_correction_limit <= 20 for item in budgets.values())
    assert all(item.per_challenge_limit <= 5 for item in budgets.values())
    assert budgets[TaskMode.RESEARCH_GUIDE].level == "medium"


def _copy_plan_with(
    base: TaskPlan,
    *,
    subtasks: tuple[Subtask, ...],
    dependencies: tuple[tuple[str, str], ...],
) -> TaskPlan:
    return TaskPlan(
        task_id=base.task_id,
        intent=base.intent,
        task_mode=base.task_mode,
        goal=base.goal,
        success_criteria=base.success_criteria,
        subtasks=subtasks,
        dependencies=dependencies,
        required_capabilities=base.required_capabilities,
        risk_level=base.risk_level,
        evidence_requirement=base.evidence_requirement,
        collaboration_budget=base.collaboration_budget,
    )


def test_plan_rejects_missing_dependency_reference() -> None:
    base = _plan(TaskMode.FACT_FIND, PLAN_CASES[0][1])
    invalid = Subtask(
        subtask_id="invalid",
        type="test",
        goal="验证缺失依赖会被拒绝",
        input_requirements=("query",),
        required_capability="test_validation",
        dependencies=("missing",),
        evidence_need="none",
    )

    with pytest.raises(ValueError, match="missing dependency"):
        _copy_plan_with(
            base,
            subtasks=(invalid,),
            dependencies=(("missing", "invalid"),),
        )


def test_plan_rejects_dependency_cycle() -> None:
    base = _plan(TaskMode.FACT_FIND, PLAN_CASES[0][1])
    first = Subtask(
        "first", "test", "第一个问题", ("query",), "test", ("second",), "none"
    )
    second = Subtask(
        "second", "test", "第二个问题", ("query",), "test", ("first",), "none"
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        _copy_plan_with(
            base,
            subtasks=(first, second),
            dependencies=(("second", "first"), ("first", "second")),
        )


def test_collaboration_context_initializes_once_and_reuses_same_intent() -> None:
    calls = 0

    def resolver(query: str, **_kwargs) -> IntentResult:
        nonlocal calls
        calls += 1
        return understand_task(query, use_llm=False)

    input_data = {
        "task_id": "task-context-once",
        "query": PLAN_CASES[1][1],
        "learner_id": "learner-r03b",
    }
    first = initialize_collaboration_context(input_data, intent_resolver=resolver)
    second = initialize_collaboration_context(input_data, intent_resolver=resolver)

    assert calls == 1
    assert first is second
    assert input_data["_collaboration_context"] is first
    assert input_data["_intent_result"] is first.intent_result
    assert first.task_plan.intent is first.intent_result
    assert first.task_plan.task_id == first.task_id == "task-context-once"
    assert first.subtask_states["establish_learner_context"] == "ready"
    assert first.subtask_states["gather_mechanism_evidence"] == "ready"
    assert all(
        state == "pending"
        for subtask_id, state in first.subtask_states.items()
        if subtask_id not in {
            "establish_learner_context",
            "gather_mechanism_evidence",
        }
    )
    assert tuple(item.subtask_id for item in first.ready_subtasks()) == (
        "establish_learner_context",
        "gather_mechanism_evidence",
    )


def test_existing_context_with_different_task_id_is_rejected() -> None:
    input_data = {
        "task_id": "task-original",
        "query": PLAN_CASES[0][1],
    }
    initialize_collaboration_context(input_data, intent_resolver=lambda query, **kwargs: understand_task(query, use_llm=False))
    input_data["task_id"] = "task-other"

    with pytest.raises(ValueError, match="task_id mismatch"):
        initialize_collaboration_context(input_data)


def test_task_plan_and_context_are_private_non_mapping_contracts() -> None:
    input_data = {"task_id": "task-private-plan", "query": PLAN_CASES[3][1]}
    context = initialize_collaboration_context(
        input_data,
        intent_resolver=lambda query, **kwargs: understand_task(query, use_llm=False),
    )

    assert isinstance(context, CollaborationContext)
    assert not isinstance(context, dict)
    assert not isinstance(context.task_plan, dict)
    assert not hasattr(context, "__dict__")
    assert not hasattr(context.task_plan, "__dict__")
    with pytest.raises(TypeError):
        json.dumps(context)


def test_real_api_initializes_one_context_before_diagnosis_without_public_leakage(
    monkeypatch,
) -> None:
    intent_calls = 0
    observed: list[CollaborationContext] = []
    real_understand = understand_task
    real_diagnosis = agent_workers.run_diagnosis

    def count_understanding(query: str, **_kwargs) -> IntentResult:
        nonlocal intent_calls
        intent_calls += 1
        return real_understand(query, use_llm=False)

    def observe_diagnosis(input_data, deps):
        context = input_data.get("_collaboration_context")
        assert isinstance(context, CollaborationContext)
        assert input_data["_intent_result"] is context.intent_result
        observed.append(context)
        return real_diagnosis(input_data, deps)

    monkeypatch.setattr(agent_workers, "understand_task", count_understanding)
    monkeypatch.setattr(agent_workers, "run_diagnosis", observe_diagnosis)
    client = TestClient(UnifiedApp.create_full_app_builder().create_app())

    response = client.post(
        "/api/query",
        json={"query": PLAN_CASES[3][1], "learner_id": "r03b-api"},
    )

    assert response.status_code == 200
    assert intent_calls == 1
    assert len(observed) == 1
    data = response.json()["data"]
    context = observed[0]
    assert context.task_id == data["task_id"]
    assert context.task_plan.task_mode is TaskMode.EVALUATE
    serialized = json.dumps(data, ensure_ascii=False)
    for forbidden in (
        "TaskPlan",
        "Subtask",
        "CollaborationContext",
        "_collaboration_context",
        "_intent_result",
        "collaboration_budget",
        "max_expensive_iterations",
    ):
        assert forbidden not in serialized
    assert "task_mode" not in data
