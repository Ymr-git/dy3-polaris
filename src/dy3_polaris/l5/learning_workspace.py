"""Public-safe learning workspace projection over existing learner truth.

The workspace is a read-only product projection.  It does not create another
learner state, mastery model, curriculum, resource system, or task engine.
Eligibility is computed from the existing R05 learner view, R06 canonical
Concept mappings, the authored L2 PracticeBank, and server-observed task or
resource activity.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from dy3_polaris.l3.concept_foundation import canonical_concepts
from dy3_polaris.l5.knowledge_learning_fusion import KnowledgeLearningContext
from dy3_polaris.l5.learner_intelligence import LearnerIntelligenceView


class ActionEligibilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NEEDS_DIAGNOSTIC = "NEEDS_DIAGNOSTIC"
    NEEDS_PREREQUISITE = "NEEDS_PREREQUISITE"
    EVIDENCE_LIMITED = "EVIDENCE_LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


class SequenceStatus(str, Enum):
    REQUIRED = "REQUIRED"
    READY = "READY"
    DONE = "DONE"
    VERIFY_FIRST = "VERIFY_FIRST"
    NOT_NEEDED_THIS_ROUND = "NOT_NEEDED_THIS_ROUND"
    LOCKED_BY_PREREQUISITE = "LOCKED_BY_PREREQUISITE"


@dataclass(frozen=True, slots=True)
class CapabilityCoverage:
    concept_id: str
    name: str
    concept_available: bool
    evidence_status: str
    authored_practice_available: bool
    practice_kps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EligibleAction:
    action_type: str
    label: str
    status: ActionEligibilityStatus
    target: str
    reason: str
    route: str
    context: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class LearningSequenceStep:
    order: int
    concept_id: str
    label: str
    status: SequenceStatus
    action: EligibleAction
    reason: str


@dataclass(frozen=True, slots=True)
class LearningWorkspaceView:
    """Request-local product view; all fields have explicit public projection."""

    learner_id: str
    identity_scope: str
    lifecycle_stage: str
    learner_summary: Mapping[str, Any]
    current_focus: Mapping[str, Any]
    blocking_prerequisites: tuple[Mapping[str, str], ...]
    current_challenge_decision: Mapping[str, str]
    teaching_adaptation_summary: Mapping[str, Any]
    capability_coverage: tuple[CapabilityCoverage, ...]
    learning_sequence: tuple[LearningSequenceStep, ...]
    recent_changes: tuple[Mapping[str, Any], ...]
    quick_actions: tuple[EligibleAction, ...]
    resume_action: EligibleAction | None
    data_freshness: Mapping[str, Any]


def _concept_catalog() -> dict[str, Any]:
    return {item.concept_id: item for item in canonical_concepts()}


def _practice_kps(practice_bank: Any | None) -> set[str]:
    return set(str(item) for item in getattr(practice_bank, "by_kp", {}) or {})


def _coverage_for(
    concept_id: str,
    *,
    knowledge: KnowledgeLearningContext | None,
    practice_kps: set[str],
    released_evidence_concepts: set[str],
) -> CapabilityCoverage | None:
    concept = _concept_catalog().get(concept_id)
    if concept is None:
        return None
    mapped_kps = tuple(str(item) for item in concept.related_kps)
    authored = tuple(item for item in mapped_kps if item in practice_kps)
    if concept_id in released_evidence_concepts:
        evidence_status = "RELEASED_TASK_EVIDENCE"
    elif knowledge is not None and concept_id in knowledge.evidence_available_concepts:
        # R06 currently proves a chunk mention candidate, not claim support.
        evidence_status = "MENTION_CANDIDATE_ONLY"
    else:
        evidence_status = "NONE"
    return CapabilityCoverage(
        concept_id=concept_id,
        name=str(concept.canonical_name),
        concept_available=True,
        evidence_status=evidence_status,
        authored_practice_available=bool(authored),
        practice_kps=authored,
    )


def _learn_action(coverage: CapabilityCoverage, *, required: bool = False) -> EligibleAction:
    return EligibleAction(
        action_type="LEARN_CONCEPT",
        label=f"学习{coverage.name}",
        status=(
            ActionEligibilityStatus.NEEDS_PREREQUISITE
            if required
            else ActionEligibilityStatus.AVAILABLE
        ),
        target=coverage.concept_id,
        reason=(
            "该 Concept 是当前目标的真实先修节点。"
            if required
            else "该主题存在于 R06 Canonical Concept 目录。"
        ),
        route="query",
        context={"concept_id": coverage.concept_id},
    )


def _practice_action(coverage: CapabilityCoverage) -> EligibleAction:
    if coverage.authored_practice_available:
        return EligibleAction(
            action_type="PRACTICE",
            label=f"练习{coverage.name}",
            status=ActionEligibilityStatus.AVAILABLE,
            target=coverage.concept_id,
            reason="L2 PracticeBank 存在与该 Concept 映射 KP 对应的真实题目。",
            route="practice",
            context={
                "concept_id": coverage.concept_id,
                "kp_ids": ",".join(coverage.practice_kps),
                "attempt_purpose": "DIAGNOSTIC",
            },
        )
    return EligibleAction(
        action_type="PRACTICE",
        label=f"练习{coverage.name}",
        status=ActionEligibilityStatus.UNAVAILABLE,
        target=coverage.concept_id,
        reason="当前题库没有覆盖该 Concept 的已编写题目。",
        route="",
        context={},
    )


def _public_action(action: EligibleAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type,
        "label": action.label,
        "status": action.status.value,
        "target": action.target,
        "reason": action.reason,
        "route": action.route,
        "context": dict(action.context),
    }


def build_learning_workspace_view(
    *,
    learner_view: LearnerIntelligenceView,
    learner_report: Mapping[str, Any],
    practice_bank: Any | None = None,
    recent_task: Mapping[str, Any] | None = None,
    released_evidence_concepts: Iterable[str] = (),
) -> LearningWorkspaceView:
    """Build the single learner-facing workspace truth projection."""

    if not isinstance(learner_view, LearnerIntelligenceView):
        raise TypeError("learner_view must be LearnerIntelligenceView")
    report = dict(learner_report or {})
    knowledge = learner_view.value("derived_context", "knowledge_learning_context")
    if not isinstance(knowledge, KnowledgeLearningContext):
        knowledge = None
    practice_ids = _practice_kps(practice_bank)
    released = {str(item) for item in released_evidence_concepts if str(item)}
    path = dict((report.get("learning_path") or {}))
    next_action = dict((report.get("next_action") or {}))
    next_concept = str(path.get("next_concept") or next_action.get("target") or "unknown")
    current_position = str(path.get("current_position") or "unknown")
    prerequisites = tuple(
        dict.fromkeys(str(item) for item in path.get("prerequisites_to_learn") or () if str(item))
    )

    ordered_ids: list[str] = []
    for concept_id in (*prerequisites, next_concept, current_position):
        if concept_id != "unknown" and concept_id not in ordered_ids:
            ordered_ids.append(concept_id)
    # Product topic choices are discovered from the canonical catalogue and
    # real authored PracticeBank coverage.  They are not canned answers or a
    # hard-coded learning route.
    for concept in canonical_concepts():
        if len(ordered_ids) >= 8:
            break
        if concept.concept_id in ordered_ids:
            continue
        if any(str(kp_id) in practice_ids for kp_id in concept.related_kps):
            ordered_ids.append(concept.concept_id)
    coverages = tuple(
        item
        for item in (
            _coverage_for(
                concept_id,
                knowledge=knowledge,
                practice_kps=practice_ids,
                released_evidence_concepts=released,
            )
            for concept_id in ordered_ids
        )
        if item is not None
    )
    coverage_by_id = {item.concept_id: item for item in coverages}
    model_mastery = dict(learner_view.value("models", "mastery", {}) or {})

    sequence: list[LearningSequenceStep] = []
    if current_position in coverage_by_id:
        coverage = coverage_by_id[current_position]
        sequence.append(LearningSequenceStep(
            order=len(sequence) + 1,
            concept_id=current_position,
            label=coverage.name,
            status=(
                SequenceStatus.DONE
                if any(float(model_mastery.get(kp_id, 0.0) or 0.0) >= 0.7
                       for kp_id in coverage.practice_kps)
                else SequenceStatus.VERIFY_FIRST
            ),
            action=_learn_action(coverage),
            reason=(
                "已有模型证据支持当前位置。"
                if any(float(model_mastery.get(kp_id, 0.0) or 0.0) >= 0.7
                       for kp_id in coverage.practice_kps)
                else "路径投影指向该位置，但尚需真实作答验证。"
            ),
        ))
    for concept_id in prerequisites:
        coverage = coverage_by_id.get(concept_id)
        if coverage is None:
            continue
        sequence.append(LearningSequenceStep(
            order=len(sequence) + 1,
            concept_id=concept_id,
            label=coverage.name,
            status=SequenceStatus.REQUIRED,
            action=_learn_action(coverage, required=True),
            reason="R06 prerequisite_of 关系要求先补齐该节点。",
        ))
    if next_concept in coverage_by_id and next_concept not in prerequisites:
        coverage = coverage_by_id[next_concept]
        sequence.append(LearningSequenceStep(
            order=len(sequence) + 1,
            concept_id=next_concept,
            label=coverage.name,
            status=(
                SequenceStatus.LOCKED_BY_PREREQUISITE
                if prerequisites else SequenceStatus.READY
            ),
            action=_learn_action(coverage),
            reason=str(next_action.get("reason") or "当前 Learner Intelligence 推荐节点。"),
        ))

    challenge = dict(report.get("difficulty_decision") or {})
    decision = str(challenge.get("decision") or "DIAGNOSE_FIRST")
    teaching = learner_view.value("derived_context", "adaptive_teaching_decision")
    teaching_summary = {
        "content_depth": str(getattr(teaching, "content_depth", "foundation")),
        "explanation_strategy": str(getattr(teaching, "explanation_strategy", "baseline_explanation")),
        "representation_modes": list(getattr(teaching, "representation_modes", ()) or ()),
        "source_class": "DECISION",
    }

    focus_coverage = coverage_by_id.get(next_concept)
    focus = {
        "concept_id": next_concept,
        "name": focus_coverage.name if focus_coverage else "尚未确定",
        "state": "READY" if focus_coverage else "UNKNOWN",
        "reason": str(next_action.get("reason") or "尚无足够事实确定学习焦点。"),
    }
    blockers = tuple(
        {
            "concept_id": concept_id,
            "name": coverage_by_id[concept_id].name,
            "source_class": "DERIVED_FROM_R06_RELATIONS",
        }
        for concept_id in prerequisites
        if concept_id in coverage_by_id
    )

    quick_actions: list[EligibleAction] = [
        EligibleAction(
            action_type="ASK",
            label="直接提问",
            status=ActionEligibilityStatus.AVAILABLE,
            target="",
            reason="UNKNOWN 不阻止提问；Diagnosis 会在任务中解释现有学习信息。",
            route="query",
            context={},
        )
    ]
    if focus_coverage is not None:
        quick_actions.append(_learn_action(focus_coverage, required=bool(prerequisites)))
        quick_actions.append(_practice_action(focus_coverage))
        quick_actions.append(EligibleAction(
            action_type="VIEW_EVIDENCE",
            label=f"核对{focus_coverage.name}证据",
            status=(
                ActionEligibilityStatus.AVAILABLE
                if focus_coverage.evidence_status == "RELEASED_TASK_EVIDENCE"
                else ActionEligibilityStatus.EVIDENCE_LIMITED
            ),
            target=focus_coverage.concept_id,
            reason=(
                "最近真实任务已发布与该 Concept 关联的公开 Evidence。"
                if focus_coverage.evidence_status == "RELEASED_TASK_EVIDENCE"
                else "当前仅有术语提及候选或无 Claim 支持，不能宣称已具备科学证据。"
            ),
            route=("kb" if focus_coverage.evidence_status == "RELEASED_TASK_EVIDENCE" else ""),
            context={"concept_id": focus_coverage.concept_id},
        ))

    resume_action: EligibleAction | None = None
    recent = dict(recent_task or {})
    if recent.get("task_id") and recent.get("resume_route"):
        resume_action = EligibleAction(
            action_type="RESUME",
            label="继续上次学习",
            status=ActionEligibilityStatus.AVAILABLE,
            target=str(recent.get("task_id")),
            reason="服务端仍保留该任务产生的真实资源计划。",
            route=str(recent.get("resume_route")),
            context={"task_id": str(recent.get("task_id"))},
        )

    timeline = tuple(
        {
            "event_type": str(item.get("event_type") or ""),
            "timestamp": float(item.get("timestamp", 0.0) or 0.0),
            "reference": str(item.get("reference") or ""),
            "outcome": str(item.get("outcome") or ""),
            "source_class": str(item.get("source_class") or "OBSERVED"),
        }
        for item in list(report.get("growth_timeline") or ())[-8:]
        if isinstance(item, Mapping)
    )
    updated_at = float(report.get("updated_at", 0.0) or 0.0)
    identity_scope = "DEVICE_LOCAL_GUEST" if learner_view.learner_id.startswith("guest-") else "AUTHENTICATED"
    declared = dict(learner_view.value("facts", "declared_background", {}) or {})
    observed_count = len(learner_view.value("facts", "observed_records", ()) or ())
    return LearningWorkspaceView(
        learner_id=learner_view.learner_id,
        identity_scope=identity_scope,
        lifecycle_stage=str(learner_view.metadata.get("learner_lifecycle_stage") or "UNKNOWN_LEARNER"),
        learner_summary={
            "declared_background": declared,
            "observed_record_count": observed_count,
            "modelled_kp_count": len(model_mastery),
            "evidence_status": str(report.get("status") or "UNKNOWN"),
        },
        current_focus=focus,
        blocking_prerequisites=blockers,
        current_challenge_decision={
            "decision": decision,
            "reason": str(challenge.get("reason") or "当前没有足够作答证据。"),
            "source_class": "DECISION",
        },
        teaching_adaptation_summary=teaching_summary,
        capability_coverage=coverages,
        learning_sequence=tuple(sequence),
        recent_changes=timeline,
        quick_actions=tuple(quick_actions),
        resume_action=resume_action,
        data_freshness={
            "updated_at": updated_at if updated_at > 0 else None,
            "status": "OBSERVED" if timeline else "NO_OBSERVED_CHANGE",
            "request_local_projection": True,
        },
    )


def public_learning_workspace_projection(view: LearningWorkspaceView) -> dict[str, Any]:
    """Explicit plain-data DTO.  Private source objects are never traversed."""

    if not isinstance(view, LearningWorkspaceView):
        raise TypeError("view must be LearningWorkspaceView")
    return {
        "learner_id": view.learner_id,
        "identity_scope": view.identity_scope,
        "continuity": "SAME_DEVICE" if view.identity_scope == "DEVICE_LOCAL_GUEST" else "ACCOUNT",
        "cross_device_continuity": "PARTIAL" if view.identity_scope == "DEVICE_LOCAL_GUEST" else "SUPPORTED",
        "lifecycle_stage": view.lifecycle_stage,
        "learner_summary": dict(view.learner_summary),
        "current_focus": dict(view.current_focus),
        "blocking_prerequisites": [dict(item) for item in view.blocking_prerequisites],
        "current_challenge_decision": dict(view.current_challenge_decision),
        "teaching_adaptation_summary": dict(view.teaching_adaptation_summary),
        "capability_coverage": [
            {
                "concept_id": item.concept_id,
                "name": item.name,
                "concept_available": item.concept_available,
                "evidence_status": item.evidence_status,
                "authored_practice_available": item.authored_practice_available,
                "practice_kps": list(item.practice_kps),
            }
            for item in view.capability_coverage
        ],
        "learning_sequence": [
            {
                "order": item.order,
                "concept_id": item.concept_id,
                "label": item.label,
                "status": item.status.value,
                "action": _public_action(item.action),
                "reason": item.reason,
            }
            for item in view.learning_sequence
        ],
        "recent_changes": [dict(item) for item in view.recent_changes],
        "quick_actions": [_public_action(item) for item in view.quick_actions],
        "resume_action": _public_action(view.resume_action) if view.resume_action else None,
        "data_freshness": dict(view.data_freshness),
    }


__all__ = [
    "ActionEligibilityStatus",
    "CapabilityCoverage",
    "EligibleAction",
    "LearningSequenceStep",
    "LearningWorkspaceView",
    "SequenceStatus",
    "build_learning_workspace_view",
    "public_learning_workspace_projection",
]
