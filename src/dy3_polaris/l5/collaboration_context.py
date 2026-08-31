"""R-03B private task plan and collaboration workspace.

The contracts in this module are request-local runtime facts.  They do not
replace the P0 task lifecycle context, the public confirmation ``plan_id``, the
L4 ``DecisionPlan``, or the standalone L5 ``OrchestrationPlan``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, MutableMapping

from dy3_polaris.l5.task_understanding import IntentResult, TaskMode, understand_task


@dataclass(frozen=True, slots=True)
class CollaborationBudget:
    """Planning budget only; R-03B does not execute these iterations."""

    level: str
    max_expensive_iterations: int
    retrieval_budget: int
    review_revision_budget: int
    global_correction_limit: int = 20
    per_challenge_limit: int = 5

    def __post_init__(self) -> None:
        if self.level not in {"low", "low-medium", "medium", "medium-high"}:
            raise ValueError(f"unsupported collaboration budget level: {self.level}")
        for name in (
            "max_expensive_iterations",
            "retrieval_budget",
            "review_revision_budget",
            "global_correction_limit",
            "per_challenge_limit",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.global_correction_limit > 20:
            raise ValueError("global correction limit cannot exceed 20")
        if self.per_challenge_limit > 5:
            raise ValueError("per-challenge correction limit cannot exceed 5")


@dataclass(frozen=True, slots=True)
class Subtask:
    """One problem to solve, independent from the Agent that may execute it."""

    subtask_id: str
    type: str
    goal: str
    input_requirements: tuple[str, ...]
    required_capability: str
    dependencies: tuple[str, ...]
    evidence_need: str
    status: str = "pending"

    def __post_init__(self) -> None:
        if not self.subtask_id:
            raise ValueError("subtask_id is required")
        if not self.type or not self.goal or not self.required_capability:
            raise ValueError("subtask type, goal, and required_capability are required")
        if self.subtask_id in self.dependencies:
            raise ValueError(f"subtask cannot depend on itself: {self.subtask_id}")
        if self.evidence_need not in {"none", "low", "medium", "high"}:
            raise ValueError(f"unsupported evidence need: {self.evidence_need}")
        if self.status != "pending":
            raise ValueError("R-03B subtasks must be initialized as pending")


@dataclass(frozen=True, slots=True)
class TaskPlan:
    """Immutable decomposition of one IntentResult into a validated DAG."""

    task_id: str
    intent: IntentResult
    task_mode: TaskMode
    goal: str
    success_criteria: tuple[str, ...]
    subtasks: tuple[Subtask, ...]
    dependencies: tuple[tuple[str, str], ...]
    required_capabilities: tuple[str, ...]
    risk_level: str
    evidence_requirement: str
    collaboration_budget: CollaborationBudget

    def __post_init__(self) -> None:
        if self.task_mode is not self.intent.task_mode:
            raise ValueError("task plan mode must match IntentResult")
        if not self.goal or not self.success_criteria or not self.subtasks:
            raise ValueError("task plan goal, success criteria, and subtasks are required")
        ids = [subtask.subtask_id for subtask in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate subtask_id")
        id_set = set(ids)
        expected_edges: list[tuple[str, str]] = []
        for subtask in self.subtasks:
            for dependency in subtask.dependencies:
                if dependency not in id_set:
                    raise ValueError(
                        f"missing dependency: {subtask.subtask_id} -> {dependency}"
                    )
                expected_edges.append((dependency, subtask.subtask_id))
        if tuple(expected_edges) != self.dependencies:
            raise ValueError("task plan dependency edges do not match subtasks")
        self.topological_order()

    def get_subtask(self, subtask_id: str) -> Subtask | None:
        return next(
            (subtask for subtask in self.subtasks if subtask.subtask_id == subtask_id),
            None,
        )

    def topological_order(self) -> tuple[str, ...]:
        """Return a stable topological order and reject dependency cycles."""
        ids = [subtask.subtask_id for subtask in self.subtasks]
        in_degree = {subtask_id: 0 for subtask_id in ids}
        adjacency = {subtask_id: [] for subtask_id in ids}
        for source, target in self.dependencies:
            adjacency[source].append(target)
            in_degree[target] += 1
        queue = deque(subtask_id for subtask_id in ids if in_degree[subtask_id] == 0)
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for target in adjacency[current]:
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)
        if len(order) != len(ids):
            raise ValueError("task plan contains a dependency cycle")
        return tuple(order)

    def ready_subtasks(self, completed: set[str] | None = None) -> tuple[Subtask, ...]:
        """R-03C-facing read interface; it does not execute or change status."""
        done = set(completed or ())
        return tuple(
            subtask
            for subtask in self.subtasks
            if subtask.subtask_id not in done
            and all(dependency in done for dependency in subtask.dependencies)
        )


@dataclass(slots=True)
class CollaborationContext:
    """The single authoritative private collaboration workspace for one task."""

    task_id: str
    query: str
    intent_result: IntentResult
    task_plan: TaskPlan
    learner_context: dict[str, Any]
    collaboration_budget: CollaborationBudget
    subtask_states: dict[str, str] = field(default_factory=dict)
    evidence_pool: list[Any] = field(default_factory=list)
    contributions: list[Any] = field(default_factory=list)
    challenges: list[Any] = field(default_factory=list)
    revision_history: list[Any] = field(default_factory=list)
    retrieval_history: list[Any] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    decisions: list[Any] = field(default_factory=list)
    iteration_state: dict[str, int] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_plan.task_id != self.task_id:
            raise ValueError("CollaborationContext and TaskPlan task_id mismatch")
        if self.task_plan.intent is not self.intent_result:
            raise ValueError("CollaborationContext must own the TaskPlan IntentResult")
        if self.task_plan.collaboration_budget is not self.collaboration_budget:
            raise ValueError("CollaborationContext must use the TaskPlan budget")
        expected_states = {
            subtask.subtask_id: subtask.status for subtask in self.task_plan.subtasks
        }
        if self.subtask_states and self.subtask_states != expected_states:
            raise ValueError("subtask states must be initialized from the TaskPlan")
        self.subtask_states.update(expected_states)
        self.iteration_state.update(
            {
                "expensive_iterations_used": 0,
                "global_corrections_used": 0,
                "retrievals_used": 0,
                "review_revisions_used": 0,
            }
        )
        self.runtime_metadata.update(
            {
                "created_at": time.time(),
                "context_owner": "run_guidance",
                "current_execution_model": "compatibility_scheduler",
                "task_plan_scheduling_active": True,
                "subtask_state_history": [],
            }
        )
        self.refresh_ready_subtasks()

    def ready_subtasks(self) -> tuple[Subtask, ...]:
        """Return only runtime tasks that have reached READY."""
        return tuple(
            item
            for item in self.task_plan.subtasks
            if self.subtask_states[item.subtask_id] == "ready"
        )

    def _transition(self, subtask_id: str, target: str) -> None:
        current = self.subtask_states.get(subtask_id)
        allowed = {
            "pending": {"ready"},
            "ready": {"running"},
            "running": {"completed", "failed"},
        }
        if target not in allowed.get(str(current), set()):
            raise ValueError(
                f"illegal subtask transition: {subtask_id} {current} -> {target}"
            )
        self.subtask_states[subtask_id] = target
        history = self.runtime_metadata.setdefault("subtask_state_history", [])
        history.append((subtask_id, current, target, time.time()))

    def refresh_ready_subtasks(self) -> tuple[Subtask, ...]:
        completed = {
            key for key, value in self.subtask_states.items() if value == "completed"
        }
        for item in self.task_plan.subtasks:
            if (
                self.subtask_states[item.subtask_id] == "pending"
                and all(dep in completed for dep in item.dependencies)
            ):
                self._transition(item.subtask_id, "ready")
        return self.ready_subtasks()

    def begin_subtasks(self, subtask_ids: tuple[str, ...]) -> None:
        """Start one compatibility batch in stable topological order."""
        batch = set(subtask_ids)
        completed = {
            key for key, value in self.subtask_states.items() if value == "completed"
        }
        for subtask_id in self.task_plan.topological_order():
            if subtask_id not in batch:
                continue
            item = self.task_plan.get_subtask(subtask_id)
            if item is None:
                raise ValueError(f"unknown subtask: {subtask_id}")
            if self.subtask_states[subtask_id] == "pending":
                if not all(dep in completed or dep in batch for dep in item.dependencies):
                    raise ValueError(f"subtask dependencies are not ready: {subtask_id}")
                self._transition(subtask_id, "ready")
            self._transition(subtask_id, "running")

    def complete_subtasks(self, subtask_ids: tuple[str, ...]) -> None:
        for subtask_id in subtask_ids:
            self._transition(subtask_id, "completed")
        self.refresh_ready_subtasks()

    def fail_subtasks(self, subtask_ids: tuple[str, ...]) -> None:
        for subtask_id in subtask_ids:
            self._transition(subtask_id, "failed")

    def record_contribution(self, contribution: Any) -> None:
        if getattr(contribution, "task_id", None) != self.task_id:
            raise ValueError("contribution task_id mismatch")
        if self.task_plan.get_subtask(
            str(getattr(contribution, "subtask_id", ""))
        ) is None:
            raise ValueError("contribution subtask_id is not in TaskPlan")
        self.contributions.append(contribution)

    def record_challenge(self, challenge: Any) -> None:
        if getattr(challenge, "task_id", None) != self.task_id:
            raise ValueError("challenge task_id mismatch")
        if not any(
            getattr(item, "contribution_id", None)
            == getattr(challenge, "target_contribution_id", None)
            for item in self.contributions
        ):
            raise ValueError("challenge target contribution is unknown")
        self.challenges.append(challenge)
        self.runtime_metadata.setdefault("challenge_recorded_at", {})[
            str(getattr(challenge, "challenge_id", ""))
        ] = time.time()

    def can_resolve(self, action: str) -> bool:
        if self.iteration_state["global_corrections_used"] >= min(
            self.collaboration_budget.global_correction_limit,
            self.collaboration_budget.max_expensive_iterations,
        ):
            return False
        if action == "REVISE":
            return (
                self.iteration_state["review_revisions_used"]
                < self.collaboration_budget.review_revision_budget
            )
        if action == "RE_RETRIEVE":
            return (
                self.iteration_state["retrievals_used"]
                < self.collaboration_budget.retrieval_budget
            )
        return False

    def consume_resolution_budget(self, action: str) -> None:
        if not self.can_resolve(action):
            raise ValueError(f"collaboration budget exhausted for {action}")
        self.iteration_state["expensive_iterations_used"] += 1
        self.iteration_state["global_corrections_used"] += 1
        if action == "REVISE":
            self.iteration_state["review_revisions_used"] += 1
        elif action == "RE_RETRIEVE":
            self.iteration_state["retrievals_used"] += 1


_BUDGETS = {
    # FACT_FIND remains cheap, but a real Reviewer objection gets one chance
    # to revise or refresh evidence instead of leaking an unresolved answer.
    TaskMode.FACT_FIND: CollaborationBudget("low", 1, 1, 1),
    TaskMode.EXPLAIN: CollaborationBudget("low-medium", 8, 4, 6),
    TaskMode.COMPARE: CollaborationBudget("medium", 12, 6, 8),
    TaskMode.EVALUATE: CollaborationBudget("medium-high", 20, 10, 14),
    TaskMode.RESEARCH_GUIDE: CollaborationBudget("medium", 16, 8, 10),
}


def _subtask(
    subtask_id: str,
    type_: str,
    goal: str,
    capability: str,
    *,
    dependencies: tuple[str, ...] = (),
    inputs: tuple[str, ...] = ("query", "intent_result"),
    evidence_need: str = "medium",
) -> Subtask:
    return Subtask(
        subtask_id=subtask_id,
        type=type_,
        goal=goal,
        input_requirements=inputs,
        required_capability=capability,
        dependencies=dependencies,
        evidence_need=evidence_need,
    )


def _fact_find_subtasks(query: str) -> tuple[Subtask, ...]:
    return (
        _subtask(
            "establish_learner_context",
            "learner_prerequisite_analysis",
            "确认事实回答所需的学习起点与表达深度",
            "learner_context_analysis",
            evidence_need="none",
        ),
        _subtask(
            "identify_fact",
            "fact_identification",
            f"识别问题中需要确认的具体事实：{query}",
            "domain_question_analysis",
            dependencies=("establish_learner_context",),
            evidence_need="low",
        ),
        _subtask(
            "verify_fact_evidence",
            "evidence_verification",
            "找到并核验直接支持该事实的领域证据",
            "evidence_validation",
            dependencies=("identify_fact",),
            inputs=("identified_fact", "intent_result"),
            evidence_need="high",
        ),
        _subtask(
            "form_concise_response",
            "learning_synthesis",
            "在证据边界内形成简洁且可学习的事实回答",
            "learning_response_synthesis",
            dependencies=("verify_fact_evidence",),
            inputs=("verified_fact", "learner_context"),
            evidence_need="low",
        ),
    )


def _explain_subtasks(query: str) -> tuple[Subtask, ...]:
    return (
        _subtask(
            "establish_learner_context",
            "learner_prerequisite_analysis",
            "确认理解该机制所需的学习起点与前置概念",
            "learner_context_analysis",
            evidence_need="none",
        ),
        _subtask(
            "explain_mechanism",
            "mechanism_explanation",
            f"解释问题涉及的材料与物理机制：{query}",
            "domain_knowledge_generation",
            dependencies=("establish_learner_context",),
            inputs=("query", "learner_context", "intent_result"),
            evidence_need="medium",
        ),
        _subtask(
            "gather_mechanism_evidence",
            "evidence_support",
            "收集支持关键机制链条的真实领域证据",
            "evidence_retrieval",
            evidence_need="high",
        ),
        _subtask(
            "review_explanation",
            "scientific_review",
            "审核机制解释与证据是否一致并识别不确定性",
            "scientific_review",
            dependencies=("explain_mechanism", "gather_mechanism_evidence"),
            inputs=("mechanism_explanation", "evidence"),
            evidence_need="high",
        ),
        _subtask(
            "synthesize_learning_explanation",
            "learning_synthesis",
            "按学习者起点综合机制、证据、限制和下一步学习建议",
            "learning_guidance",
            dependencies=("establish_learner_context", "review_explanation"),
            inputs=("learner_context", "reviewed_explanation"),
            evidence_need="medium",
        ),
    )


def _comparison_targets(intent: IntentResult) -> tuple[str, str]:
    materials = [
        entity.text
        for entity in intent.domain_entities
        if entity.entity_type == "material"
    ]
    if len(materials) < 2:
        materials.extend(
            entity.text
            for entity in intent.domain_entities
            if entity.entity_type in {"formula", "ion"}
            and entity.text not in materials
        )
    if len(materials) < 2:
        materials.extend(("第一个比较对象", "第二个比较对象"))
    return materials[0], materials[1]


def _compare_subtasks(query: str, intent: IntentResult) -> tuple[Subtask, ...]:
    target_a, target_b = _comparison_targets(intent)
    return (
        _subtask(
            "establish_learner_context",
            "learner_prerequisite_analysis",
            "确认比较任务所需的学习起点与评价概念",
            "learner_context_analysis",
            evidence_need="none",
        ),
        _subtask(
            "define_comparison_criteria",
            "comparison_criteria",
            f"为比较问题定义同条件、可核验的评价标准：{query}",
            "comparison_criteria_definition",
            dependencies=("establish_learner_context",),
            evidence_need="medium",
        ),
        _subtask(
            "collect_first_material_evidence",
            "material_evidence",
            f"收集 {target_a} 在统一评价标准下的证据",
            "material_evidence_retrieval",
            dependencies=("define_comparison_criteria",),
            inputs=("comparison_criteria", "intent_result"),
            evidence_need="high",
        ),
        _subtask(
            "collect_second_material_evidence",
            "material_evidence",
            f"收集 {target_b} 在统一评价标准下的证据",
            "material_evidence_retrieval",
            dependencies=("define_comparison_criteria",),
            inputs=("comparison_criteria", "intent_result"),
            evidence_need="high",
        ),
        _subtask(
            "synthesize_comparison",
            "comparison_synthesis",
            "并列比较两个对象，明确条件差异、证据缺口与不可直接外推部分",
            "evidence_comparison",
            dependencies=(
                "define_comparison_criteria",
                "collect_first_material_evidence",
                "collect_second_material_evidence",
            ),
            inputs=("comparison_criteria", "first_evidence", "second_evidence"),
            evidence_need="high",
        ),
        _subtask(
            "review_comparison",
            "scientific_review",
            "审核比较是否使用同口径证据且没有把条件性结论绝对化",
            "scientific_review",
            dependencies=("synthesize_comparison",),
            inputs=("comparison", "evidence"),
            evidence_need="high",
        ),
        _subtask(
            "form_comparison_guidance",
            "learning_synthesis",
            "综合审核后的比较结论与学习提示",
            "learning_guidance",
            dependencies=("review_comparison",),
            inputs=("reviewed_comparison", "learner_context"),
            evidence_need="medium",
        ),
    )


def _evaluate_subtasks(query: str) -> tuple[Subtask, ...]:
    return (
        _subtask(
            "establish_learner_context",
            "learner_prerequisite_analysis",
            "确认评价结论所需的学习起点与风险概念",
            "learner_context_analysis",
            evidence_need="none",
        ),
        _subtask(
            "define_claim",
            "claim_definition",
            f"把待评价问题转化为有边界、可检验的主张：{query}",
            "claim_analysis",
            dependencies=("establish_learner_context",),
            evidence_need="medium",
        ),
        _subtask(
            "define_evaluation_criteria",
            "evaluation_criteria",
            "明确评价指标、适用条件与不能混同的概念",
            "evaluation_criteria_definition",
            dependencies=("define_claim",),
            inputs=("bounded_claim", "intent_result"),
            evidence_need="medium",
        ),
        _subtask(
            "collect_evaluation_evidence",
            "required_evidence",
            "收集支持、限制或反驳该主张所需的证据",
            "evidence_retrieval",
            dependencies=("define_evaluation_criteria",),
            inputs=("bounded_claim", "evaluation_criteria"),
            evidence_need="high",
        ),
        _subtask(
            "analyze_risk_assumptions",
            "risk_and_assumption_analysis",
            "分析健康、安全、条件依赖和证据缺失带来的不确定性",
            "risk_uncertainty_analysis",
            dependencies=("define_evaluation_criteria", "collect_evaluation_evidence"),
            inputs=("criteria", "evidence", "intent_ambiguity"),
            evidence_need="high",
        ),
        _subtask(
            "review_evaluation",
            "scientific_review",
            "挑战评价中的绝对化结论、证据越界与遗漏条件",
            "scientific_review",
            dependencies=("analyze_risk_assumptions",),
            inputs=("claim_evaluation", "risk_analysis", "evidence"),
            evidence_need="high",
        ),
        _subtask(
            "decide_under_uncertainty",
            "uncertainty_decision",
            "在审核后的证据边界内形成条件化判断并表达不确定性",
            "guidance_decision_under_uncertainty",
            dependencies=("review_evaluation",),
            inputs=("reviewed_evaluation", "learner_context"),
            evidence_need="high",
        ),
    )


def _research_guide_subtasks(query: str) -> tuple[Subtask, ...]:
    return (
        _subtask(
            "establish_research_start",
            "learner_research_state",
            "确认学习者或研究者的当前起点、已有条件和目标",
            "learner_context_analysis",
            evidence_need="none",
        ),
        _subtask(
            "identify_knowledge_gaps",
            "knowledge_gap_analysis",
            f"识别开展该学习或研究任务前必须补齐的问题：{query}",
            "knowledge_gap_diagnosis",
            dependencies=("establish_research_start",),
            inputs=("learner_context", "intent_result"),
            evidence_need="medium",
        ),
        _subtask(
            "gather_required_concepts_evidence",
            "concept_and_evidence_support",
            "收集关键概念、变量、可观察指标及其证据基础",
            "evidence_retrieval",
            dependencies=("identify_knowledge_gaps",),
            inputs=("knowledge_gaps", "query"),
            evidence_need="high",
        ),
        _subtask(
            "form_research_learning_steps",
            "research_learning_guidance",
            "形成按依赖排序、可验证且不越过现有证据的下一步路径",
            "research_guidance",
            dependencies=(
                "establish_research_start",
                "gather_required_concepts_evidence",
            ),
            inputs=("starting_state", "knowledge_gaps", "evidence"),
            evidence_need="medium",
        ),
        _subtask(
            "review_unsupported_recommendations",
            "scientific_review",
            "审核路径中的无证据推荐、隐含假设与不可执行步骤",
            "scientific_review",
            dependencies=("form_research_learning_steps",),
            inputs=("research_learning_steps", "evidence"),
            evidence_need="high",
        ),
        _subtask(
            "finalize_research_guidance",
            "learning_synthesis",
            "综合形成适合当前起点的研究学习建议及验证检查点",
            "learning_guidance",
            dependencies=("review_unsupported_recommendations",),
            inputs=("reviewed_steps", "learner_context"),
            evidence_need="medium",
        ),
    )


def _success_criteria(mode: TaskMode) -> tuple[str, ...]:
    return {
        TaskMode.FACT_FIND: (
            "目标事实被明确识别",
            "事实有直接证据支持",
            "回答简洁且不超出证据",
        ),
        TaskMode.EXPLAIN: (
            "机制链条与学习者起点匹配",
            "关键机制有证据支持",
            "科学审核与限制被保留",
        ),
        TaskMode.COMPARE: (
            "两个对象使用统一评价标准",
            "双方证据均被独立收集",
            "比较结论经过科学审核并保留条件",
        ),
        TaskMode.EVALUATE: (
            "待评价主张和指标边界明确",
            "风险、假设与缺失证据被识别",
            "最终判断表达条件和不确定性",
        ),
        TaskMode.RESEARCH_GUIDE: (
            "当前起点与知识缺口明确",
            "下一步按前置依赖排序且可验证",
            "无证据建议经过审核并被标记",
        ),
    }[mode]


def build_task_plan(task_id: str, query: str, intent: IntentResult) -> TaskPlan:
    """Build one mode-specific immutable plan; no Agent is selected here."""
    mode = intent.task_mode
    if mode is TaskMode.FACT_FIND:
        subtasks = _fact_find_subtasks(query)
    elif mode is TaskMode.EXPLAIN:
        subtasks = _explain_subtasks(query)
    elif mode is TaskMode.COMPARE:
        subtasks = _compare_subtasks(query, intent)
    elif mode is TaskMode.EVALUATE:
        subtasks = _evaluate_subtasks(query)
    elif mode is TaskMode.RESEARCH_GUIDE:
        subtasks = _research_guide_subtasks(query)
    else:  # pragma: no cover - TaskMode is a closed enum
        raise ValueError(f"unsupported TaskMode: {mode}")

    dependencies = tuple(
        (dependency, subtask.subtask_id)
        for subtask in subtasks
        for dependency in subtask.dependencies
    )
    capabilities = tuple(
        dict.fromkeys(
            (*intent.required_capabilities, *(item.required_capability for item in subtasks))
        )
    )
    return TaskPlan(
        task_id=str(task_id or ""),
        intent=intent,
        task_mode=mode,
        goal=intent.learner_goal,
        success_criteria=_success_criteria(mode),
        subtasks=subtasks,
        dependencies=dependencies,
        required_capabilities=capabilities,
        risk_level=intent.risk_level,
        evidence_requirement=intent.evidence_need,
        collaboration_budget=_BUDGETS[mode],
    )


IntentResolver = Callable[..., IntentResult]


def initialize_collaboration_context(
    input_data: MutableMapping[str, Any],
    *,
    intent_resolver: IntentResolver = understand_task,
) -> CollaborationContext:
    """Create or reuse the one request-local authoritative collaboration context."""
    existing = input_data.get("_collaboration_context")
    if existing is not None:
        if not isinstance(existing, CollaborationContext):
            raise TypeError("_collaboration_context must be CollaborationContext")
        input_task_id = str(input_data.get("task_id") or "")
        if input_task_id and existing.task_id != input_task_id:
            raise ValueError("existing CollaborationContext task_id mismatch")
        input_data["_intent_result"] = existing.intent_result
        return existing

    intent = input_data.get("_intent_result")
    if not isinstance(intent, IntentResult):
        intent = intent_resolver(
            str(input_data.get("query") or ""),
            learner_context={
                "learner_id": input_data.get("learner_id")
                or input_data.get("student_id"),
                "learner_level": input_data.get("learner_level"),
            },
        )
    task_id = str(input_data.get("task_id") or "")
    plan = build_task_plan(task_id, str(input_data.get("query") or ""), intent)
    context = CollaborationContext(
        task_id=task_id,
        query=str(input_data.get("query") or ""),
        intent_result=intent,
        task_plan=plan,
        learner_context={
            "learner_id": input_data.get("learner_id")
            or input_data.get("student_id"),
            "learner_level": input_data.get("learner_level"),
        },
        collaboration_budget=plan.collaboration_budget,
    )
    input_data["_collaboration_context"] = context
    # R-03A compatibility alias: exact same object, not a second truth source.
    input_data["_intent_result"] = context.intent_result
    return context


def get_collaboration_context(
    input_data: Mapping[str, Any],
) -> CollaborationContext | None:
    """Minimal R-03C-facing read interface."""
    value = input_data.get("_collaboration_context")
    return value if isinstance(value, CollaborationContext) else None


__all__ = [
    "CollaborationBudget",
    "CollaborationContext",
    "Subtask",
    "TaskPlan",
    "build_task_plan",
    "get_collaboration_context",
    "initialize_collaboration_context",
]
