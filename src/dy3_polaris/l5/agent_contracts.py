"""R-03C private contracts connecting TaskPlan to the four existing Agents.

These objects are request-local runtime facts.  They are intentionally not
mapping-compatible and must never become public response, recorder, audit, or
TaskEvent payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Mapping

from dy3_polaris.l5.collaboration_context import CollaborationContext, Subtask
from dy3_polaris.l5.task_understanding import IntentResult, TaskMode


DIAGNOSIS_AGENT_ID = "agent.learning.diagnosis"
GENERATION_AGENT_ID = "agent.knowledge.generation"
REVIEW_AGENT_ID = "agent.quality.review"
GUIDANCE_AGENT_ID = "agent.guidance.decision"


class ClaimType(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    RECOMMENDATION = "RECOMMENDATION"
    UNCERTAIN = "UNCERTAIN"


class EvidenceSupportLevel(str, Enum):
    """Scientific support semantics; retrieval relevance is not support."""

    MENTION = "MENTION"
    CANDIDATE = "CANDIDATE"
    SUPPORTS = "SUPPORTS"
    CONFLICTS = "CONFLICTS"
    INSUFFICIENT = "INSUFFICIENT"


class RequestedAction(str, Enum):
    ACCEPT = "ACCEPT"
    USE_EXISTING_EVIDENCE = "USE_EXISTING_EVIDENCE"
    REQUEST_RETRIEVAL = "REQUEST_RETRIEVAL"
    REQUEST_RE_RANK = "REQUEST_RE_RANK"
    REQUEST_REVISION = "REQUEST_REVISION"
    CHALLENGE = "CHALLENGE"
    ASK_USER = "ASK_USER"
    DECLARE_UNCERTAINTY = "DECLARE_UNCERTAINTY"
    REFUSE_CONCLUSION = "REFUSE_CONCLUSION"


class ChallengeType(str, Enum):
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    EVIDENCE_MISMATCH = "EVIDENCE_MISMATCH"
    ENTITY_MISMATCH = "ENTITY_MISMATCH"
    CONDITION_MISMATCH = "CONDITION_MISMATCH"
    OVERGENERALIZATION = "OVERGENERALIZATION"
    FACT_INFERENCE_CONFUSION = "FACT_INFERENCE_CONFUSION"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    MISSING_ASSUMPTION = "MISSING_ASSUMPTION"
    AMBIGUOUS_USER_REQUIREMENT = "AMBIGUOUS_USER_REQUIREMENT"
    SAFETY_OVERCLAIM = "SAFETY_OVERCLAIM"


class ChallengeSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ResolutionAction(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    RE_RETRIEVE = "RE_RETRIEVE"
    ASK_USER = "ASK_USER"
    REJECT = "REJECT"


class DecisionType(str, Enum):
    ANSWER = "ANSWER"
    ANSWER_WITH_UNCERTAINTY = "ANSWER_WITH_UNCERTAINTY"
    ASK_USER = "ASK_USER"
    KNOWLEDGE_GAP = "KNOWLEDGE_GAP"
    REFUSE_CONCLUSION = "REFUSE_CONCLUSION"
    LEARNING_GUIDANCE = "LEARNING_GUIDANCE"
    PARTIAL_ANSWER = "PARTIAL_ANSWER"


class QualityReleaseStatus(str, Enum):
    """Public-delivery disposition after the authoritative Reviewer loop."""

    FULL_RELEASE = "FULL_RELEASE"
    LIMITED_RELEASE = "LIMITED_RELEASE"
    ASK_USER = "ASK_USER"
    REFUSE = "REFUSE"
    WITHHOLD = "WITHHOLD"
    DEGRADED = "DEGRADED"


class ClaimFinalState(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class FinalClaimDecision:
    claim_id: str
    state: ClaimFinalState
    reason: str


@dataclass(frozen=True, slots=True)
class GuidanceDecision:
    task_id: str
    task_mode: TaskMode
    source_contribution_id: str
    source_review_id: str
    review_identity: str
    decision_type: DecisionType
    claim_decisions: tuple[FinalClaimDecision, ...]
    accepted_claim_ids: tuple[str, ...]
    rejected_claim_ids: tuple[str, ...]
    uncertain_claim_ids: tuple[str, ...]
    answer_policy: str
    learner_depth: str
    next_action: str
    recommended_path: tuple[Any, ...]
    clarification_needed: bool
    knowledge_gap: tuple[str, ...]
    confidence: float
    reasoning_summary: str
    status: str


@dataclass(frozen=True, slots=True)
class QualityReleaseDecision:
    """Fail-closed decision over already-produced runtime facts.

    This is not a second review verdict and cannot change scientific content.
    It only decides whether the reviewed artifact is eligible for public
    delivery and records a bounded, non-CoT explanation.
    """

    task_id: str
    status: QualityReleaseStatus
    eligible: bool
    public_answer: str
    reason_codes: tuple[str, ...]
    review_status: str
    review_verdict: str
    answer_identity: str
    evidence_versions: tuple[int, ...]
    correction_count: int
    message: str


@dataclass(frozen=True, slots=True)
class FinalCollaborationResult:
    task_id: str
    task_mode: TaskMode
    answer: str
    answer_identity: str
    accepted_claims: tuple[Claim, ...]
    rejected_claims: tuple[Claim, ...]
    uncertain_claims: tuple[Claim, ...]
    evidence: tuple[Any, ...]
    review: AgentContribution
    decision: GuidanceDecision
    next_action: str
    recommended_path: tuple[Any, ...]
    learner_context_summary: tuple[str, ...]
    knowledge_gaps: tuple[str, ...]
    completion_eligibility: bool
    provenance_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CollaborationTraceEvent:
    """One private, causally linked observation of actual runtime behavior."""

    event_id: str
    task_id: str
    sequence: int
    event_type: str
    actor: str
    subtask_id: str
    timestamp: float
    summary: str
    artifact_refs: tuple[str, ...] = ()
    parent_event_id: str = ""
    caused_by: str = ""


@dataclass(frozen=True, slots=True)
class CollaborationTrace:
    """Request-local trace; never a raw reasoning or prompt transcript."""

    task_id: str
    task_mode: TaskMode
    events: tuple[CollaborationTraceEvent, ...]
    path_signature: tuple[str, ...]
    cost_summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Challenge:
    """One bounded Reviewer objection against a concrete contribution."""

    challenge_id: str
    task_id: str
    subtask_id: str
    reviewer_agent_id: str
    target_contribution_id: str
    target_claim_ids: tuple[str, ...]
    challenge_type: ChallengeType
    reason: str
    severity: ChallengeSeverity
    missing_information: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    requested_action: ResolutionAction
    status: str
    iteration: int


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    statement: str
    claim_type: ClaimType
    evidence_refs: tuple[str, ...] = ()
    confidence: float = 0.0
    status: str = "current"
    concept_ids: tuple[str, ...] = ()
    scope: str = ""
    conditions: tuple[tuple[str, str], ...] = ()
    support_status: EvidenceSupportLevel = EvidenceSupportLevel.INSUFFICIENT
    source_refs: tuple[str, ...] = ()
    reviewer_status: str = "not_reviewed"
    provenance_refs: tuple[str, ...] = ()
    answer_identity: str = ""
    evidence_version: int = 0


@dataclass(frozen=True, slots=True)
class AgentInput:
    task_id: str
    agent_id: str
    subtask: Subtask
    intent: IntentResult
    task_mode: TaskMode
    user_query: str
    learner_context: Mapping[str, Any]
    domain_context: Mapping[str, Any]
    prior_contributions: tuple["AgentContribution", ...]
    evidence_pack: tuple[Any, ...]
    constraints: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    iteration_state: Mapping[str, int]
    runtime_metadata: Mapping[str, Any]
    related_subtasks: tuple[Subtask, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentContribution:
    contribution_id: str
    task_id: str
    subtask_id: str
    agent_id: str
    conclusion: str
    claims: tuple[Claim, ...]
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...]
    uncertainty: tuple[str, ...]
    challenges: tuple[str, ...]
    requested_actions: tuple[RequestedAction, ...]
    tool_usage: tuple[str, ...]
    confidence: float
    status: str
    produced_at: float
    sequence: int
    covered_subtask_ids: tuple[str, ...] = ()
    artifact_identity: str = ""
    parent_contribution_id: str = ""
    revision_reason: str = ""
    iteration: int = 0


_CAPABILITY_TO_AGENT = {
    "learner_context_analysis": DIAGNOSIS_AGENT_ID,
    "knowledge_gap_diagnosis": DIAGNOSIS_AGENT_ID,
    "domain_question_analysis": GENERATION_AGENT_ID,
    "domain_knowledge_generation": GENERATION_AGENT_ID,
    "evidence_retrieval": GENERATION_AGENT_ID,
    "material_evidence_retrieval": GENERATION_AGENT_ID,
    "comparison_criteria_definition": GENERATION_AGENT_ID,
    "evidence_comparison": GENERATION_AGENT_ID,
    "claim_analysis": GENERATION_AGENT_ID,
    "evaluation_criteria_definition": GENERATION_AGENT_ID,
    "research_guidance": GENERATION_AGENT_ID,
    "evidence_validation": REVIEW_AGENT_ID,
    "scientific_review": REVIEW_AGENT_ID,
    "risk_uncertainty_analysis": REVIEW_AGENT_ID,
    "learning_response_synthesis": GUIDANCE_AGENT_ID,
    "learning_guidance": GUIDANCE_AGENT_ID,
    "guidance_decision_under_uncertainty": GUIDANCE_AGENT_ID,
}


_TOOLS_BY_AGENT = {
    DIAGNOSIS_AGENT_ID: ("irt", "learner_profile", "learning_memory"),
    GENERATION_AGENT_ID: ("hybrid_retrieval", "reranker", "llm_synthesizer"),
    REVIEW_AGENT_ID: ("fact_checker", "anti_hallucination"),
    GUIDANCE_AGENT_ID: ("learning_decision",),
}


def agent_for_capability(required_capability: str) -> str:
    """Resolve a frozen capability to one of the existing four Agents."""
    try:
        return _CAPABILITY_TO_AGENT[required_capability]
    except KeyError as exc:
        raise ValueError(
            f"unknown required_capability: {required_capability}"
        ) from exc


def _compatible_batch(
    context: CollaborationContext,
    agent_id: str,
) -> tuple[Subtask, ...]:
    """Select ready work plus same-Agent descendants for one legacy call."""
    ready = [
        item
        for item in context.ready_subtasks()
        if agent_for_capability(item.required_capability) == agent_id
    ]
    if not ready:
        return ()
    selected = {item.subtask_id for item in ready}
    completed = {
        key for key, value in context.subtask_states.items() if value == "completed"
    }
    changed = True
    while changed:
        changed = False
        for subtask_id in context.task_plan.topological_order():
            item = context.task_plan.get_subtask(subtask_id)
            if item is None or item.subtask_id in selected:
                continue
            if agent_for_capability(item.required_capability) != agent_id:
                continue
            if all(dep in completed or dep in selected for dep in item.dependencies):
                selected.add(item.subtask_id)
                changed = True
    return tuple(
        item
        for item in context.task_plan.subtasks
        if item.subtask_id in selected
    )


def build_agent_input(
    context: CollaborationContext,
    agent_id: str,
) -> AgentInput | None:
    """Build a least-information input from currently schedulable subtasks."""
    batch = _compatible_batch(context, agent_id)
    if not batch:
        return None
    intent = context.intent_result
    return AgentInput(
        task_id=context.task_id,
        agent_id=agent_id,
        subtask=batch[0],
        related_subtasks=batch[1:],
        intent=intent,
        task_mode=intent.task_mode,
        user_query=context.query,
        learner_context=dict(context.learner_context),
        domain_context={
            "entities": intent.domain_entities,
            "evidence_need": intent.evidence_need,
            "risk_level": intent.risk_level,
        },
        prior_contributions=tuple(context.contributions),
        evidence_pack=tuple(context.evidence_pool),
        constraints=tuple(intent.ambiguity),
        allowed_tools=_TOOLS_BY_AGENT[agent_id],
        iteration_state=dict(context.iteration_state),
        runtime_metadata={
            "scheduler": "compatibility_ready_subtasks",
            "batch_size": len(batch),
        },
    )


def contribution_sequence(context: CollaborationContext) -> int:
    return len(context.contributions) + 1


def make_contribution(
    context: CollaborationContext,
    agent_input: AgentInput,
    *,
    conclusion: str,
    claims: tuple[Claim, ...] = (),
    evidence_refs: tuple[str, ...] = (),
    assumptions: tuple[str, ...] = (),
    uncertainty: tuple[str, ...] = (),
    challenges: tuple[str, ...] = (),
    requested_actions: tuple[RequestedAction, ...] = (),
    tool_usage: tuple[str, ...] = (),
    confidence: float = 0.0,
    status: str = "completed",
    artifact_identity: str = "",
    parent_contribution_id: str = "",
    revision_reason: str = "",
    iteration: int = 0,
) -> AgentContribution:
    sequence = contribution_sequence(context)
    covered = (agent_input.subtask, *agent_input.related_subtasks)
    return AgentContribution(
        contribution_id=(
            f"contrib-{context.task_id}-{agent_input.subtask.subtask_id}-{sequence}"
        ),
        task_id=context.task_id,
        subtask_id=agent_input.subtask.subtask_id,
        agent_id=agent_input.agent_id,
        conclusion=conclusion,
        claims=claims,
        evidence_refs=evidence_refs,
        assumptions=assumptions,
        uncertainty=uncertainty,
        challenges=challenges,
        requested_actions=requested_actions,
        tool_usage=tool_usage,
        confidence=max(0.0, min(1.0, float(confidence or 0.0))),
        status=status,
        produced_at=time.time(),
        sequence=sequence,
        covered_subtask_ids=tuple(item.subtask_id for item in covered),
        artifact_identity=artifact_identity,
        parent_contribution_id=parent_contribution_id,
        revision_reason=revision_reason,
        iteration=max(0, int(iteration)),
    )


__all__ = [
    "AgentContribution",
    "AgentInput",
    "Claim",
    "ClaimType",
    "EvidenceSupportLevel",
    "Challenge",
    "ChallengeSeverity",
    "ChallengeType",
    "ClaimFinalState",
    "CollaborationTrace",
    "CollaborationTraceEvent",
    "DecisionType",
    "FinalClaimDecision",
    "FinalCollaborationResult",
    "GuidanceDecision",
    "QualityReleaseDecision",
    "QualityReleaseStatus",
    "RequestedAction",
    "ResolutionAction",
    "agent_for_capability",
    "build_agent_input",
    "make_contribution",
]
