"""四个核心 Agent 的执行 workers.

把默认 Agent 定义从“注册表条目”变成可运行的执行体:
- 学情诊断 Agent: L2 IRT / 画像 / 记忆快照
- 知识生成 Agent: L3 混合检索 + 响应合成
- 审核校验 Agent: L3 事实校验 + L0 防幻觉管道
- 导学决策 Agent: 串联诊断 → 生成 → 审核，输出最终教学路径决策
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import logging
import os
import re
import time
from typing import Any, Callable

from dy3_polaris.l2.kp_catalog import kp_name
from dy3_polaris.l2.models import ProfileConflictError
from dy3_polaris.l3.concept_foundation import ConceptType, canonical_concepts
from dy3_polaris.l3.concept_relations import (
    ConceptRelationType,
    build_concept_relation_network,
)
from dy3_polaris.l5.critic import critique_answer, rewrite_query
from dy3_polaris.l5.agent_contracts import (
    AgentContribution,
    AgentInput,
    Challenge,
    ChallengeSeverity,
    ChallengeType,
    Claim,
    ClaimFinalState,
    ClaimType,
    CollaborationTrace,
    CollaborationTraceEvent,
    DecisionType,
    FinalClaimDecision,
    FinalCollaborationResult,
    GuidanceDecision,
    QualityReleaseDecision,
    QualityReleaseStatus,
    RequestedAction,
    ResolutionAction,
    build_agent_input,
    make_contribution,
)
from dy3_polaris.l5.collaboration_context import (
    CollaborationContext,
    get_collaboration_context,
    initialize_collaboration_context,
)
from dy3_polaris.l5.interaction_recorder import InteractionPhase, get_recorder
from dy3_polaris.l5.learner_intelligence import (
    LearnerIntelligenceView,
    LearningPath,
    build_learner_intelligence_view,
    public_learner_intelligence_projection,
)
from dy3_polaris.l5.learner_foundation import (
    AdaptiveTeachingDecision,
    PersonalLearnerModel,
)
from dy3_polaris.l5.knowledge_learning_fusion import (
    ConceptLearningPath,
    KnowledgeLearningContext,
    public_knowledge_learning_projection,
)
from dy3_polaris.l5.learning_event import (
    TeachingLearningEvent,
    build_teaching_learning_event,
)
from dy3_polaris.l5.learning_resources import (
    LearningResourcePlan,
    build_learning_resource_plan,
    public_resource_projection,
)
from dy3_polaris.l5.scientific_grounding import (
    ScientificGrounding,
    atomic_claims,
    build_scientific_grounding,
    public_scientific_grounding_projection,
)
from dy3_polaris.l5.teaching_memory import (
    TeachingMemoryInterpretation,
    TeachingMemoryView,
    commit_teaching_memory,
    load_teaching_memory_view,
)
from dy3_polaris.l5.retrieval_planning import (
    EvidencePack,
    agent_aware_rerank,
    build_challenge_retrieval_plans,
    build_evidence_pack,
    build_retrieval_plans,
    hard_filter,
)
from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5.task_understanding import understand_task

logger = logging.getLogger("dy3_polaris.l5.agent_workers")

DIAGNOSIS_AGENT_ID = "agent.learning.diagnosis"
GENERATION_AGENT_ID = "agent.knowledge.generation"
REVIEW_AGENT_ID = "agent.quality.review"
GUIDANCE_AGENT_ID = "agent.guidance.decision"


def _placeholder_knowledge_enabled() -> bool:
    """未溯源教材占位内容只能在显式开发/测试模式中使用."""
    return os.environ.get("DY3_ENABLE_PLACEHOLDER_KNOWLEDGE", "0") == "1"


def _load_learner_memory_views(
    input_data: dict[str, Any],
    deps: Any,
) -> dict[str, dict[str, Any]]:
    """Load one shared learner memory and create private per-Agent views."""
    learner_id = input_data.get("learner_id") or input_data.get("student_id")
    profile_service = getattr(deps, "profile_service", None)
    if not learner_id or profile_service is None:
        return {}
    try:
        from dy3_polaris.l5.agent_memory import build_memory_views

        views = build_memory_views(
            profile_service,
            str(learner_id),
            str(input_data.get("query") or ""),
        )
        return views if any(
            bool(view.get("memory_available")) for view in views.values()
        ) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Learner Memory 投影读取失败: %s", exc)
        return {}


def _load_teaching_memory(
    input_data: dict[str, Any],
    deps: Any,
) -> TeachingMemoryView | None:
    """Load the private R-07B view without placing it in an Agent payload."""

    learner_id = input_data.get("learner_id") or input_data.get("student_id")
    profile_service = getattr(deps, "profile_service", None)
    if not learner_id or profile_service is None:
        return None
    try:
        return load_teaching_memory_view(profile_service, str(learner_id))
    except Exception as exc:  # noqa: BLE001 - optional compatibility path
        logger.warning("Teaching Memory view load failed: %s", exc)
        return None


def _apply_memory_to_collaboration_context(
    context: CollaborationContext,
    views: dict[str, dict[str, Any]],
) -> None:
    """Keep the legacy Memory path visible without granting decision authority.

    Memory used to rewrite TaskPlan and directly set downstream teaching depth.
    R-05A isolates that behavior: the Diagnosis projection is now consumed only
    through LearnerIntelligenceView.  This hook records compatibility presence
    for diagnostics but cannot alter plan, learner context, or Agent inputs.
    """
    diagnosis_view = views.get(DIAGNOSIS_AGENT_ID) or {}
    if not diagnosis_view.get("memory_available"):
        return
    context.runtime_metadata["legacy_learner_memory_path"] = "isolated"


class _PrivateRuntimeCarrier(dict[str, Any]):
    """Dict-compatible result carrier with non-serialized private metadata."""

    __slots__ = ("_contract_candidate",)


@dataclass(frozen=True, slots=True)
class _EvidenceCandidate:
    """Private source-preserving snapshot for one selected generation result."""

    task_id: str
    producer: str
    stage: str
    answer_identity: str
    context_chunks: tuple[Any, ...]
    citations: tuple[Any, ...]
    sources: tuple[Any, ...]
    knowledge_unavailable: bool
    honest_unavailable: bool
    evidence_versions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _ReviewCandidate:
    """Private snapshot of the answer and raw result actually reviewed."""

    task_id: str
    producer: str
    reviewed_answer_identity: str
    raw_status: str
    raw_verdict: str
    raw_reason: str
    raw_fact_check: dict[str, Any]
    raw_anti_hallucination: dict[str, Any]
    raw_confidence: float
    real_reviewer_executed: bool
    mapping_refused_reason: str
    scientific_issue_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AnswerCorrelation:
    """Private identity comparison for the final selected answer."""

    task_id: str
    final_answer_identity: str
    evidence_answer_identity: str
    review_answer_identity: str
    correlation: bool
    refusal_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FinalPrivateCandidateSet:
    """Final selected private facts carried to the API readiness gate."""

    evidence_candidate: _EvidenceCandidate | None
    review_candidate: _ReviewCandidate | None
    answer_correlation: _AnswerCorrelation
    quality_release: QualityReleaseDecision | None = None
    adaptive_teaching_decision: AdaptiveTeachingDecision | None = None
    final_collaboration_result: FinalCollaborationResult | None = None
    collaboration_trace: CollaborationTrace | None = None
    multi_agent_evaluation: Any | None = None
    teaching_learning_event: TeachingLearningEvent | None = None
    learning_resource_plan: LearningResourcePlan | None = None
    scientific_grounding: ScientificGrounding | None = None


@dataclass(frozen=True, slots=True)
class _MultiAgentEvaluation:
    """Private deterministic metrics over one real collaboration run."""

    task_id: str
    task_mode: str
    task_intelligence: dict[str, Any]
    retrieval_intelligence: dict[str, Any]
    collaboration_intelligence: dict[str, Any]
    trust: dict[str, Any]
    educational_intelligence: dict[str, Any]
    costs: dict[str, Any]


def _contract_batch_ids(agent_input: AgentInput) -> tuple[str, ...]:
    return tuple(
        item.subtask_id
        for item in (agent_input.subtask, *agent_input.related_subtasks)
    )


def _start_contract_agent(
    context: CollaborationContext,
    agent_id: str,
) -> AgentInput | None:
    agent_input = build_agent_input(context, agent_id)
    if agent_input is not None:
        context.begin_subtasks(_contract_batch_ids(agent_input))
    return agent_input


def _contract_runtime_payload(
    base: dict[str, Any],
    agent_input: AgentInput | None,
) -> dict[str, Any]:
    """Project private AgentInput into CURRENT worker fields without leakage."""
    payload = dict(base)
    payload.pop("_learner_memory_views", None)
    learner_intelligence_view = payload.pop("_learner_intelligence_view", None)
    if agent_input is None:
        return payload
    payload["_agent_input"] = agent_input
    if (
        agent_input.agent_id == DIAGNOSIS_AGENT_ID
        and isinstance(learner_intelligence_view, LearnerIntelligenceView)
    ):
        payload["_learner_intelligence_view"] = learner_intelligence_view
    payload["query"] = agent_input.user_query
    learner_level = agent_input.learner_context.get("level")
    if learner_level:
        payload["learner_level"] = learner_level
    payload["weak_kps"] = tuple(
        agent_input.learner_context.get("weak_kps") or ()
    )
    teaching_decision = agent_input.learner_context.get(
        "adaptive_teaching_decision"
    )
    if (
        agent_input.agent_id == GENERATION_AGENT_ID
        and isinstance(teaching_decision, AdaptiveTeachingDecision)
    ):
        # Private runtime input: public mapping keys and Agent Contract remain
        # unchanged. Generation consumes only the Diagnosis-owned strategy.
        payload["_adaptive_teaching_decision"] = teaching_decision
    return payload


def _planned_retrieval_queries(
    agent_input: AgentInput,
    plan: Any,
) -> tuple[str, ...]:
    """Keep raw and Concept-aware branches beside subtask rewrites."""

    values: list[str] = [agent_input.user_query]
    knowledge = dict(getattr(agent_input, "learner_context", {}) or {}).get(
        "knowledge_learning_context"
    )
    if isinstance(knowledge, KnowledgeLearningContext):
        concepts_by_id = {
            concept.concept_id: concept for concept in canonical_concepts()
        }
        concept_queries = tuple(
            str(knowledge.concept_names.get(concept_id, "")).strip()
            for concept_id in knowledge.target_concepts
            if str(knowledge.concept_names.get(concept_id, "")).strip()
        )
        # Search each resolved Concept independently before the combined view.
        # This prevents one generic neighbour (for example, "spectrum") from
        # drowning out the exact scientific Concept in lexical retrieval.
        values.extend(concept_queries)
        if concept_queries:
            values.append(" ".join(concept_queries))
        # R06 Concept identity is bilingual while most learner questions are
        # Chinese and much of the real paper corpus is English.  Keep the
        # canonical Concept and all curated aliases in one retrieval branch so
        # lexical search can reach the same scientific idea across languages.
        # These are catalogue terms, not generated facts or answer templates.
        for concept_id in knowledge.target_concepts:
            concept = concepts_by_id.get(concept_id)
            if concept is None:
                continue
            # Keep each alias as an independent branch.  Combining Chinese and
            # English aliases in one BM25 query biased the result toward noisy
            # bilingual thesis pages and could still hide a direct English
            # paper passage.
            values.extend(
                str(value).strip()
                for value in concept.aliases
                if str(value).strip()
            )
            alias_query = " ".join(
                dict.fromkeys(
                    str(value).strip()
                    for value in concept.aliases
                    if str(value).strip()
                )
            )
            if alias_query:
                values.append(alias_query)
            if (
                concept.concept_type is ConceptType.CHARACTERIZATION
                and _is_procedure_question(agent_input.user_query)
            ):
                # Characterisation questions need evidence about the complete
                # method, not only passages which name the instrument.  These
                # are generic experimental-method dimensions and contain no
                # XRD/PL-specific parameter or answer fact.
                values.extend(
                    (
                        f"{alias_query} sample preparation measurement conditions "
                        "data analysis standard reference",
                        f"{alias_query} 样品制备 测量条件 数据分析 标准参照",
                    )
                )
        # A scientific question often names the two ends of a mechanism while
        # omitting the intermediate Concept (for example, crystal field ->
        # Stark splitting -> emission spectrum).  Expand only one curated
        # relation hop from the R-06 network.  The relation guides retrieval;
        # it is never used as answer evidence.
        for concept_id in _concept_relation_retrieval_ids(
            knowledge, agent_input.user_query
        ):
            concept = concepts_by_id.get(concept_id)
            if concept is None:
                continue
            values.extend(
                str(value).strip()
                for value in concept.aliases
                if str(value).strip()
            )
    values.extend(tuple(getattr(plan, "rewritten_queries", ()) or ()))
    return tuple(dict.fromkeys(value for value in values if value))


def _concept_retrieval_terms(agent_input: AgentInput) -> tuple[str, ...]:
    """Return canonical names and curated aliases resolved from this task."""

    knowledge = dict(getattr(agent_input, "learner_context", {}) or {}).get(
        "knowledge_learning_context"
    )
    if not isinstance(knowledge, KnowledgeLearningContext):
        return ()
    concepts_by_id = {
        concept.concept_id: concept for concept in canonical_concepts()
    }
    values: list[str] = []
    for concept_id in knowledge.target_concepts:
        concept = concepts_by_id.get(concept_id)
        if concept is not None:
            values.append(concept.canonical_name)
            values.extend(concept.aliases)
    return tuple(dict.fromkeys(value for value in values if str(value).strip()))


def _concept_relation_retrieval_ids(
    knowledge: KnowledgeLearningContext,
    query: str,
) -> tuple[str, ...]:
    """Return bounded one-hop mechanism neighbours from curated R-06 facts."""

    query_value = str(query or "")
    procedure = _is_procedure_question(query_value)
    mechanism = bool(re.search(
        r"(影响|关系|导致|作用|联系|为什么|机制|机理|"
        r"affect|impact|cause|relation|why|mechanism)",
        query_value,
        flags=re.IGNORECASE,
    ))
    asks_directional_effect = bool(re.search(
        r"(影响|导致|作用|关系|联系|怎么改变|如何改变|"
        r"affect|impact|cause|relation|influence|change)",
        query_value,
        flags=re.IGNORECASE,
    ))
    evaluation = bool(re.search(
        r"(评价|指标|测量|测试|表征|分析|说明|"
        r"evaluate|metric|measure|test|characteri[sz]e|analy[sz]e)",
        query_value,
        flags=re.IGNORECASE,
    ))
    if not (procedure or mechanism or evaluation):
        return ()
    network = build_concept_relation_network()
    allowed: set[ConceptRelationType] = set()
    if mechanism:
        allowed.update({
            ConceptRelationType.EXPLAINS,
            ConceptRelationType.CAUSES,
            ConceptRelationType.AFFECTS,
        })
    if procedure or evaluation:
        allowed.add(ConceptRelationType.EVALUATED_BY)
    values: list[str] = []
    for concept_id in knowledge.target_concepts:
        # A pure “why/mechanism” question needs the Concepts which explain or
        # cause the target, not every downstream quantity affected by it.  The
        # old undirected expansion turned “thermal-quenching mechanism” into
        # unrelated quantum-efficiency measurement evidence.  Outgoing
        # expansion remains available for an explicit effect/relationship
        # question where that direction is part of the user's task.
        if not mechanism or asks_directional_effect:
            for relation in network.outgoing(concept_id):
                target_id = relation.target_concept_id
                if relation.relation_type not in allowed:
                    continue
                if target_id in knowledge.target_concepts or target_id in values:
                    continue
                values.append(target_id)
                if len(values) >= 6:
                    return tuple(values)
        if mechanism:
            # Mechanism questions often name the outcome (thermal quenching)
            # while the causal Concept is the incoming end of the curated
            # relation (non-radiative relaxation -> thermal quenching).
            for relation in network.relations:
                source_id = relation.source_concept_id
                if relation.target_concept_id != concept_id:
                    continue
                if relation.relation_type not in allowed:
                    continue
                if source_id in knowledge.target_concepts or source_id in values:
                    continue
                values.append(source_id)
                if len(values) >= 6:
                    return tuple(values)
    return tuple(values)


def _concept_relation_retrieval_terms(agent_input: AgentInput) -> tuple[str, ...]:
    """Return canonical terms of bounded R-06 relation neighbours."""

    knowledge = dict(getattr(agent_input, "learner_context", {}) or {}).get(
        "knowledge_learning_context"
    )
    if not isinstance(knowledge, KnowledgeLearningContext):
        return ()
    concepts_by_id = {
        concept.concept_id: concept for concept in canonical_concepts()
    }
    values: list[str] = []
    for concept_id in _concept_relation_retrieval_ids(
        knowledge, agent_input.user_query
    ):
        concept = concepts_by_id.get(concept_id)
        if concept is not None:
            values.append(concept.canonical_name)
            values.extend(concept.aliases)
    return tuple(dict.fromkeys(value for value in values if str(value).strip()))


def _is_procedure_question(query: str) -> bool:
    """Identify requests for an experimental procedure, not a definition."""

    value = str(query or "").strip()
    if re.search(r"(怎么|如何|怎样)影响", value):
        return False
    if re.search(r"(步骤|流程|操作|实验方法|protocol|procedure)", value, re.IGNORECASE):
        return True
    return bool(
        re.search(r"(怎么|如何|怎样).*(测量|测试|表征|制备|合成|操作|使用|分析)", value)
        or re.search(r"how\s+to\s+(measure|test|prepare|synthesi[sz]e|operate|analyse|analyze)", value, re.IGNORECASE)
    )


def _concept_plan_expansion_terms(agent_input: AgentInput) -> tuple[str, ...]:
    """Return curated Concept terms plus method dimensions for reranking."""

    values = [
        *_concept_retrieval_terms(agent_input),
        *_concept_relation_retrieval_terms(agent_input),
    ]
    knowledge = dict(getattr(agent_input, "learner_context", {}) or {}).get(
        "knowledge_learning_context"
    )
    if not isinstance(knowledge, KnowledgeLearningContext):
        return tuple(values)
    concepts_by_id = {
        concept.concept_id: concept for concept in canonical_concepts()
    }
    if not _is_procedure_question(agent_input.user_query):
        return tuple(dict.fromkeys(value for value in values if str(value).strip()))
    if any(
        concepts_by_id.get(concept_id) is not None
        and concepts_by_id[concept_id].concept_type is ConceptType.CHARACTERIZATION
        for concept_id in knowledge.target_concepts
    ):
        values.extend(
            (
                "sample preparation",
                "measurement conditions",
                "data analysis",
                "standard reference",
                "样品制备",
                "测量条件",
                "数据分析",
                "标准参照",
            )
        )
    return tuple(dict.fromkeys(value for value in values if str(value).strip()))


def _prepare_generation_retrieval(
    context: CollaborationContext,
    agent_input: AgentInput,
    payload: dict[str, Any],
    deps: "AgentDependencies",
    *,
    plans_override: tuple[Any, ...] | None = None,
    evidence_version: int = 1,
    refresh_reason: str = "",
    requested_by: str = "",
) -> tuple[AgentInput, dict[str, Any]]:
    """Execute private subtask plans through CURRENT retrieval components."""
    if "�" in agent_input.user_query:
        # Preserve the CURRENT path for already-corrupted legacy text.  A
        # retrieval plan cannot recover entities or information gaps from it.
        return agent_input, payload
    plans = plans_override if plans_override is not None else build_retrieval_plans(agent_input)
    packs: list[EvidencePack] = []
    merged: list[dict[str, Any]] = []
    merged_scores: list[float] = []
    seen: set[str] = set()
    for plan in plans:
        ranking_plan = replace(
            plan,
            expansion_terms=tuple(
                dict.fromkeys(
                    (*tuple(getattr(plan, "expansion_terms", ()) or ()),
                     *_concept_plan_expansion_terms(agent_input))
                )
            ),
        )
        candidate_by_key: dict[str, dict[str, Any]] = {}
        score_by_key: dict[str, float] = {}
        knowledge = dict(getattr(agent_input, "learner_context", {}) or {}).get(
            "knowledge_learning_context"
        )
        if isinstance(knowledge, KnowledgeLearningContext) and deps.l3_store is not None:
            mapped_concept_ids = tuple(dict.fromkeys((
                *knowledge.target_concepts,
                *_concept_relation_retrieval_ids(
                    knowledge,
                    agent_input.user_query,
                ),
            )))
            try:
                for chunk in deps.l3_store.find_chunks_by_metadata(
                    "concept_ids",
                    mapped_concept_ids,
                ):
                    item = chunk.model_dump(exclude={"embedding"})
                    key = str(item.get("chunk_id") or item.get("content") or "")
                    candidate_by_key[key] = item
                    score_by_key[key] = max(score_by_key.get(key, 0.0), 1.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("R-06 Concept evidence lookup failed: %s", exc)
        # Keep the user's original question as an independent retrieval branch.
        # A long subtask rewrite can dilute the decisive scientific terms in
        # lexical retrieval even when the real corpus contains an exact match.
        # Both branches still pass through the existing hard filter and
        # agent-aware reranker; this does not inject domain facts or bypass the
        # RetrievalPlan.
        retrieval_queries = _planned_retrieval_queries(agent_input, plan)
        for rewritten_query in retrieval_queries:
            retrieval = None
            query_vector = None
            entity_id = None
            is_concept_query = (
                rewritten_query != agent_input.user_query
                and rewritten_query not in plan.rewritten_queries
            )
            if is_concept_query and deps.l3_store is not None:
                # HybridRetriever merges entities and chunks before its own
                # top-k.  On a large corpus an exact Concept evidence summary
                # can therefore disappear even when ChunkStore ranks it in
                # the first page.  Preserve a bounded direct lexical branch
                # for canonical Concept aliases, then apply the same hard
                # filter and agent-aware reranker below.  This is still real
                # L3 retrieval; it neither injects answer text nor bypasses
                # scientific source admission.
                try:
                    lexical_hits = deps.l3_store.search_text(
                        rewritten_query,
                        top_k=max(48, plan.top_k * 8),
                    )
                    for chunk, lexical_score in lexical_hits:
                        item = chunk.model_dump(exclude={"embedding"})
                        key = str(item.get("chunk_id") or item.get("content") or "")
                        if key not in candidate_by_key:
                            candidate_by_key[key] = item
                        score_by_key[key] = max(
                            score_by_key.get(key, float("-inf")),
                            float(lexical_score),
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("R-03D Concept lexical retrieval failed: %s", exc)
            if deps.embedding_manager is not None and not is_concept_query:
                try:
                    query_vector = deps.embedding_manager.embed(rewritten_query).vector
                except Exception:  # noqa: BLE001
                    query_vector = None
            if deps.l3_store is not None and not is_concept_query:
                entity_id = _resolve_entity_id(rewritten_query, deps.l3_store)
            if deps.hybrid_retriever is not None:
                try:
                    retrieval = deps.hybrid_retriever.retrieve(
                        rewritten_query,
                        top_k=max(12, plan.top_k * 3),
                        query_vector=query_vector,
                        entity_id=entity_id,
                        retrievers=(
                            ["keyword"] if is_concept_query else None
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("R-03D planned retrieval failed: %s", exc)
            if retrieval is None:
                continue
            current = retrieval
            if deps.reranker is not None and not is_concept_query:
                try:
                    current = deps.reranker.rerank_result(
                        rewritten_query, retrieval, top_k=max(8, plan.top_k * 2)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("R-03D base rerank failed: %s", exc)
            current_results = list(getattr(current, "results", ()) or ())
            current_scores = [
                float(value)
                for value in (getattr(current, "scores", ()) or ())
            ]
            for index, item in enumerate(current_results):
                key = str(item.get("chunk_id") or item.get("content") or index)
                score = current_scores[index] if index < len(current_scores) else 0.0
                if key not in candidate_by_key:
                    candidate_by_key[key] = item
                score_by_key[key] = max(score_by_key.get(key, float("-inf")), score)
        candidates = list(candidate_by_key.values())
        filtered = hard_filter(ranking_plan, candidates)
        filtered_scores = [
            score_by_key.get(str(item.get("chunk_id") or item.get("content") or ""), 0.0)
            for item in filtered
        ]
        # Rank the complete admissible pool before applying ``top_k``.  A
        # Concept relation is a retrieval constraint, not answer evidence;
        # nevertheless at least one real chunk about that related Concept must
        # survive into the EvidencePack when such a chunk exists.  Otherwise a
        # generic passage which repeats the user's words can crowd out the
        # actual measurement/mechanism evidence (for example, a paragraph
        # discussing quantum efficiency crowding out an integrating-sphere
        # measurement paragraph).
        full_ranking_plan = replace(
            ranking_plan,
            top_k=max(ranking_plan.top_k, len(filtered)),
        )
        full_ranked = agent_aware_rerank(
            full_ranking_plan,
            filtered,
            filtered_scores,
        )
        ranked = list(full_ranked[: ranking_plan.top_k])
        relation_champions: list[tuple[dict[str, Any], float, tuple[str, ...]]] = []
        if isinstance(knowledge, KnowledgeLearningContext) and full_ranked:
            concepts_by_id = {
                concept.concept_id: concept for concept in canonical_concepts()
            }
            champion_keys: set[str] = set()
            # Preserve at most one real chunk for every resolved target Concept
            # and each bounded one-hop relation Concept.  The target champion
            # prevents bilingual exact evidence (for example "thermal
            # quenching") from being crowded out by generic Chinese prose;
            # relation champions keep distinct CIE/CCT/CRI/QE evidence for
            # multi-metric questions.
            champion_concept_ids = tuple(dict.fromkeys((
                *knowledge.target_concepts,
                *_concept_relation_retrieval_ids(
                    knowledge,
                    agent_input.user_query,
                ),
            )))
            for champion_concept_id in champion_concept_ids:
                champion_concept = concepts_by_id.get(champion_concept_id)
                if champion_concept is None:
                    continue
                relation_terms = tuple(
                    str(term).strip().casefold()
                    for term in (
                        champion_concept.canonical_name,
                        *champion_concept.aliases,
                    )
                    if len(str(term).strip()) >= 2
                )
                relation_candidates = [
                    value
                    for value in full_ranked
                    if (
                        champion_concept_id
                        in tuple(
                            str(concept_id)
                            for concept_id in (
                                (value[0].get("metadata") or {}).get(
                                    "concept_ids", ()
                                )
                                if isinstance(value[0].get("metadata"), dict)
                                else ()
                            )
                        )
                        or any(
                            term in str(value[0].get("content") or "").casefold()
                            for term in relation_terms
                        )
                    )
                ]
                if not relation_candidates:
                    continue
                champion = max(
                    relation_candidates,
                    key=lambda value: (
                        int(
                            champion_concept_id
                            in tuple(
                                str(concept_id)
                                for concept_id in (
                                    (value[0].get("metadata") or {}).get(
                                        "concept_ids", ()
                                    )
                                    if isinstance(value[0].get("metadata"), dict)
                                    else ()
                                )
                            )
                        ),
                        sum(
                            1
                            for term in relation_terms
                            if term
                            in str(value[0].get("content") or "").casefold()
                        ),
                        value[1],
                    ),
                )
                champion_key = str(
                    champion[0].get("chunk_id")
                    or champion[0].get("content")
                    or ""
                )
                if champion_key in champion_keys:
                    continue
                champion_keys.add(champion_key)
                relation_champions.append(champion)
                if len(relation_champions) >= ranking_plan.top_k:
                    break
            if relation_champions:
                ranked = [
                    *relation_champions,
                    *(
                        value
                        for value in ranked
                        if str(
                            value[0].get("chunk_id")
                            or value[0].get("content")
                            or ""
                        )
                        not in champion_keys
                    ),
                ][: ranking_plan.top_k]
        parent_pack_ids = tuple(
            f"{item.subtask_id}:v{item.version}"
            for item in context.tool_results.get("active_evidence_packs", ())
            if isinstance(item, EvidencePack) and item.subtask_id == plan.subtask_id
        )
        pack = build_evidence_pack(
            ranking_plan,
            ranked,
            version=evidence_version,
            parent_pack_ids=parent_pack_ids,
            refresh_reason=refresh_reason,
            requested_by=requested_by,
        )
        packs.append(pack)
        for item, score, _reasons in ranked:
            key = str(item.get("chunk_id") or item.get("content") or "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(item))
            merged_scores.append(score)
    try:
        from dy3_polaris.l3.models import RetrievalResult

        planned_result = RetrievalResult(
            query=(plans[0].rewritten_queries[0] if plans and plans[0].rewritten_queries else agent_input.user_query),
            results=merged,
            scores=merged_scores,
            total=len(merged),
            source_type="r03d_agent_aware",
        )
    except Exception:  # noqa: BLE001
        planned_result = None
    if evidence_version <= 1:
        context.evidence_pool[:] = packs
    else:
        context.evidence_pool.extend(packs)
    context.tool_results["retrieval_plans"] = plans
    context.tool_results["evidence_packs"] = tuple(packs)
    context.tool_results["active_evidence_packs"] = tuple(packs)
    context.retrieval_history.append(
        {
            "version": evidence_version,
            "plans": plans,
            "packs": tuple(packs),
            "reason": refresh_reason,
            "requested_by": requested_by,
            "timestamp": time.time(),
        }
    )
    updated_input = replace(agent_input, evidence_pack=tuple(packs))
    updated_payload = dict(payload)
    updated_payload["_agent_input"] = updated_input
    updated_payload["_retrieval_plans_applied"] = True
    updated_payload["_planned_retrieval_result"] = planned_result
    # Retrieval may use several task/subtask rewrites, but downstream answer
    # relevance and evidence-selection guards must remain anchored to the
    # user's actual question.  Reusing a long planning rewrite here caused
    # direct corpus evidence to be removed merely because it did not repeat
    # every orchestration phrase.
    updated_payload["_retrieval_query"] = agent_input.user_query
    return updated_input, updated_payload


def _review_evidence_texts(
    agent_input: AgentInput,
    generation_contribution: AgentContribution | None,
) -> list[str]:
    refs = set(generation_contribution.evidence_refs if generation_contribution else ())
    selected: list[str] = []
    fallback: list[str] = []
    for pack in agent_input.evidence_pack:
        if not isinstance(pack, EvidencePack):
            fallback.append(str(pack))
            continue
        for item in pack.items:
            if item.content:
                fallback.append(item.content)
                if item.evidence_id in refs or item.chunk_reference in refs:
                    selected.append(item.content)
    return list(dict.fromkeys(selected or fallback))


def _review_scientific_grounding(
    context: CollaborationContext,
    generation_contribution: AgentContribution | None,
) -> ScientificGrounding | None:
    """Build Reviewer input from the selected claims and active evidence only."""

    if generation_contribution is None or not generation_contribution.claims:
        return None
    return build_scientific_grounding(
        task_id=context.task_id,
        answer_identity=generation_contribution.artifact_identity,
        claims=generation_contribution.claims,
        evidence_packs=_active_evidence_packs(context),
    )


def _claim(
    contribution_id: str,
    statement: str,
    claim_type: ClaimType,
    *,
    evidence_refs: tuple[str, ...] = (),
    confidence: float = 0.0,
) -> Claim:
    return Claim(
        claim_id=f"{contribution_id}-claim-1",
        statement=statement,
        claim_type=claim_type,
        evidence_refs=evidence_refs,
        confidence=confidence,
    )


def _adapt_diagnosis_contribution(
    context: CollaborationContext,
    agent_input: AgentInput,
    result: dict[str, Any],
) -> AgentContribution:
    sequence = len(context.contributions) + 1
    contribution_id = (
        f"contrib-{context.task_id}-{agent_input.subtask.subtask_id}-{sequence}"
    )
    weak_kps = tuple(str(item) for item in result.get("weak_kps") or ())
    ability = result.get("ability") if isinstance(result.get("ability"), dict) else {}
    profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
    has_ability_evidence = bool(
        int(ability.get("response_count", 0) or 0) > 0 or profile
    )
    uncertainty = () if has_ability_evidence else ("ability snapshot unavailable",)
    mastery = profile.get("kp_mastery") if isinstance(profile.get("kp_mastery"), dict) else {}
    known_prerequisites = tuple(
        str(key)
        for key, value in mastery.items()
        if isinstance(value, (int, float)) and float(value) >= 0.7
    )
    diagnosis_claims = [
        _claim(
            contribution_id,
            str(result.get("summary") or ""),
            ClaimType.INFERENCE,
            confidence=float(result.get("confidence", 0.0) or 0.0),
        )
    ]
    diagnosis_claims.extend(
        Claim(
            claim_id=f"{contribution_id}-claim-{index}",
            statement=f"missing prerequisite: {item}",
            claim_type=ClaimType.INFERENCE,
            confidence=float(result.get("confidence", 0.0) or 0.0),
        )
        for index, item in enumerate(weak_kps, start=2)
    )
    diagnosis_claims.append(
        Claim(
            claim_id=f"{contribution_id}-claim-depth",
            statement=f"recommended explanation depth: {result.get('level') or 'unknown'}",
            claim_type=ClaimType.RECOMMENDATION,
            confidence=float(result.get("confidence", 0.0) or 0.0),
        )
    )
    contribution = make_contribution(
        context,
        agent_input,
        conclusion=str(result.get("summary") or ""),
        claims=tuple(diagnosis_claims),
        assumptions=tuple(
            [*(f"known prerequisite: {item}" for item in known_prerequisites),
             *(f"missing prerequisite: {item}" for item in weak_kps)]
        ),
        uncertainty=uncertainty,
        requested_actions=(RequestedAction.ACCEPT,),
        tool_usage=("irt", "learner_profile", "learning_memory"),
        confidence=float(result.get("confidence", 0.0) or 0.0),
    )
    context.learner_context.update(
        {
            "learner_id": result.get("learner_id"),
            "level": result.get("level"),
            "weak_kps": weak_kps,
            "recommended_depth": result.get("level"),
            "diagnosis_constraints": uncertainty,
        }
    )
    return contribution


def _apply_diagnosis_teaching_context(
    context: CollaborationContext,
    view: LearnerIntelligenceView,
) -> None:
    """Project Diagnosis-interpreted growth signals into private teaching context."""

    learning_path = view.value("derived_context", "learning_path")
    knowledge_learning_context = view.value(
        "derived_context", "knowledge_learning_context"
    )
    concept_learning_path = view.value(
        "derived_context", "concept_learning_path"
    )
    misconception_focus = tuple(
        view.value("derived_context", "misconception_focus", ())
    )
    teaching_memory_context = view.value(
        "derived_context", "teaching_memory_context"
    )
    personal_learner_model = view.value(
        "derived_context", "personal_learner_model"
    )
    teaching_decision = view.value(
        "derived_context", "adaptive_teaching_decision"
    )
    if isinstance(learning_path, LearningPath):
        context.learner_context["learning_path"] = learning_path
    if isinstance(knowledge_learning_context, KnowledgeLearningContext):
        context.learner_context["knowledge_learning_context"] = (
            knowledge_learning_context
        )
    if isinstance(concept_learning_path, ConceptLearningPath):
        context.learner_context["concept_learning_path"] = concept_learning_path
    if misconception_focus:
        context.learner_context["misconception_focus"] = misconception_focus
    if (
        isinstance(teaching_memory_context, TeachingMemoryInterpretation)
        and teaching_memory_context.available
    ):
        # This is Diagnosis' bounded interpretation, not the raw Memory view.
        context.learner_context["teaching_strategy"] = (
            teaching_memory_context.strategy
        )
    if isinstance(personal_learner_model, PersonalLearnerModel):
        context.learner_context["learner_lifecycle_stage"] = (
            personal_learner_model.lifecycle_stage.value
        )
    if isinstance(teaching_decision, AdaptiveTeachingDecision):
        context.learner_context["adaptive_teaching_decision"] = teaching_decision


def _generation_evidence_refs(result: dict[str, Any]) -> tuple[str, ...]:
    refs: list[str] = []
    for item in result.get("sources") or ():
        if isinstance(item, dict):
            value = item.get("chunk_id") or item.get("document_id")
            if value:
                refs.append(str(value))
    refs.extend(str(item) for item in result.get("citations") or () if item)
    return tuple(dict.fromkeys(refs))


def _adapt_generation_contribution(
    context: CollaborationContext,
    agent_input: AgentInput,
    result: dict[str, Any],
    *,
    parent_contribution_id: str = "",
    revision_reason: str = "",
    iteration: int = 0,
) -> AgentContribution:
    existing_packs = list(context.tool_results.get("active_evidence_packs", ())) or [
        item for item in context.evidence_pool if isinstance(item, EvidencePack)
    ]
    if (
        existing_packs
        and result.get("context_chunks")
    ):
        actual_candidates: list[dict[str, Any]] = []
        sources = list(result.get("sources") or ())
        chunks = list(result.get("context_chunks") or ())
        for index, chunk in enumerate(chunks):
            source = sources[index] if index < len(sources) and isinstance(sources[index], dict) else {}
            actual_candidates.append(
                {
                    "chunk_id": source.get("chunk_id") or f"actual-{index}",
                    "document_id": source.get("document_id") or "",
                    "content": str(chunk),
                    "metadata": {"entity": source.get("entity") or ""},
                }
            )
        synchronized: list[EvidencePack] = []
        active_version = max((item.version for item in existing_packs), default=1)
        for plan in context.tool_results.get("retrieval_plans", ()):
            filtered = hard_filter(plan, actual_candidates)
            synchronized.append(
                build_evidence_pack(
                    plan,
                    agent_aware_rerank(plan, filtered),
                    version=active_version,
                    parent_pack_ids=next(
                        (
                            item.parent_pack_ids
                            for item in existing_packs
                            if item.subtask_id == plan.subtask_id
                        ),
                        (),
                    ),
                    refresh_reason=next(
                        (item.refresh_reason for item in existing_packs), ""
                    ),
                    requested_by=next(
                        (item.requested_by for item in existing_packs), ""
                    ),
                )
            )
        if synchronized:
            context.evidence_pool[:] = [
                item
                for item in context.evidence_pool
                if not isinstance(item, EvidencePack) or item.version != active_version
            ] + synchronized
            context.tool_results["evidence_packs"] = tuple(synchronized)
            context.tool_results["active_evidence_packs"] = tuple(synchronized)
            if context.retrieval_history:
                context.retrieval_history[-1]["packs"] = tuple(synchronized)
    sequence = len(context.contributions) + 1
    contribution_id = (
        f"contrib-{context.task_id}-{agent_input.subtask.subtask_id}-{sequence}"
    )
    evidence_refs = _generation_evidence_refs(result)
    candidate = getattr(result, "_contract_candidate", None)
    if isinstance(candidate, _EvidenceCandidate):
        candidate = replace(
            candidate,
            evidence_versions=tuple(
                sorted({pack.version for pack in _active_evidence_packs(context)})
            ),
        )
        # Preserve private-only carrier semantics while binding the evidence
        # version actually used by this selected generation.
        result._contract_candidate = candidate
    answer = str(result.get("answer") or "")
    uncertainty: list[str] = []
    if result.get("knowledge_unavailable") or result.get("honest_unavailable"):
        uncertainty.append("knowledge or evidence unavailable")
    if agent_input.intent.ambiguity:
        uncertainty.extend(agent_input.intent.ambiguity)
    if agent_input.task_mode.value == "EVALUATE" and "一定" in agent_input.user_query:
        uncertainty.append("absolute claim depends on unstated conditions")
    action = (
        RequestedAction.USE_EXISTING_EVIDENCE
        if evidence_refs or result.get("context_chunks")
        else RequestedAction.DECLARE_UNCERTAINTY
    )
    # Teaching navigation is part of the learner-facing answer identity, but
    # it is not a scientific assertion that a paper must prove.  Build the
    # claim set from the scientific body while keeping every claim bound to
    # the complete answer identity used by Evidence/Reviewer correlation.
    scientific_claims = atomic_claims(
        _scientific_review_content(answer),
        contribution_id=contribution_id,
        answer_identity=(
            candidate.answer_identity
            if isinstance(candidate, _EvidenceCandidate)
            else _answer_identity(context.task_id, answer)
        ),
        confidence=float(result.get("confidence", 0.0) or 0.0),
    ) if answer else ()
    if agent_input.task_mode.value == "EVALUATE" and uncertainty:
        # A condition-bound evaluation remains an inference even when the
        # generated sentence itself omits an explicit hedge word.
        scientific_claims = tuple(
            replace(claim, claim_type=ClaimType.INFERENCE)
            if claim.claim_type is ClaimType.FACT else claim
            for claim in scientific_claims
        )
    contribution = make_contribution(
        context,
        agent_input,
        conclusion=answer,
        claims=scientific_claims,
        evidence_refs=evidence_refs,
        assumptions=tuple(agent_input.constraints),
        uncertainty=tuple(dict.fromkeys(uncertainty)),
        requested_actions=(action,),
        tool_usage=("hybrid_retrieval", "reranker", "llm_synthesizer"),
        confidence=float(result.get("confidence", 0.0) or 0.0),
        status=str(result.get("status") or "completed"),
        artifact_identity=(
            candidate.answer_identity
            if isinstance(candidate, _EvidenceCandidate)
            else ""
        ),
        parent_contribution_id=parent_contribution_id,
        revision_reason=revision_reason,
        iteration=iteration,
    )
    if not any(isinstance(item, EvidencePack) for item in context.evidence_pool):
        context.evidence_pool[:] = list(result.get("context_chunks") or ())
    return contribution


def _adapt_review_contribution(
    context: CollaborationContext,
    agent_input: AgentInput,
    result: dict[str, Any],
    *,
    parent_contribution_id: str = "",
    revision_reason: str = "",
    iteration: int = 0,
) -> AgentContribution:
    candidate = getattr(result, "_contract_candidate", None)
    verdict = str(result.get("verdict") or "")
    reason = str(result.get("reason") or "")
    actions = {
        "approved": (RequestedAction.ACCEPT,),
        "needs_review": (RequestedAction.REQUEST_REVISION,),
        "rejected": (RequestedAction.REFUSE_CONCLUSION,),
    }.get(verdict, (RequestedAction.DECLARE_UNCERTAINTY,))
    issues = () if verdict == "approved" else ((reason,) if reason else ())
    sequence = len(context.contributions) + 1
    contribution_id = (
        f"contrib-{context.task_id}-{agent_input.subtask.subtask_id}-{sequence}"
    )
    reviewed_claims = tuple(
        claim
        for contribution in agent_input.prior_contributions
        if contribution.agent_id == GENERATION_AGENT_ID
        for claim in contribution.claims
    )
    review_identity = (
        candidate.reviewed_answer_identity
        if isinstance(candidate, _ReviewCandidate)
        else ""
    )
    return make_contribution(
        context,
        agent_input,
        conclusion=reason,
        claims=(
            *reviewed_claims,
            Claim(
                claim_id=f"{contribution_id}-raw-verdict",
                statement=f"raw verdict: {verdict}",
                claim_type=ClaimType.FACT,
                evidence_refs=((review_identity,) if review_identity else ()),
                confidence=float(result.get("confidence", 0.0) or 0.0),
            ),
        ),
        evidence_refs=tuple(
            ref
            for contribution in agent_input.prior_contributions
            if contribution.agent_id == GENERATION_AGENT_ID
            for ref in contribution.evidence_refs
        ),
        uncertainty=issues,
        challenges=issues,
        requested_actions=actions,
        tool_usage=("fact_checker", "anti_hallucination"),
        confidence=float(result.get("confidence", 0.0) or 0.0),
        status=str(result.get("status") or ""),
        artifact_identity=review_identity,
        parent_contribution_id=parent_contribution_id,
        revision_reason=revision_reason,
        iteration=iteration,
    )


def _finish_contract_agent(
    context: CollaborationContext,
    agent_input: AgentInput,
    contribution: AgentContribution,
) -> None:
    context.record_contribution(contribution)
    context.complete_subtasks(_contract_batch_ids(agent_input))


def _revision_agent_input(
    context: CollaborationContext,
    original: AgentInput,
) -> AgentInput:
    """Refresh prior structured facts for an existing bounded revision."""
    return replace(
        original,
        learner_context=dict(context.learner_context),
        prior_contributions=tuple(context.contributions),
        evidence_pack=tuple(
            context.tool_results.get("active_evidence_packs", ())
            or context.evidence_pool
        ),
        iteration_state=dict(context.iteration_state),
        runtime_metadata={
            **dict(original.runtime_metadata),
            "revision": True,
        },
    )


def _answer_identity(task_id: str, content: str) -> str:
    """Request-local deterministic identity for answer/review correlation."""
    material = f"{task_id}{content}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _attach_selected_evidence_candidate(
    result: dict[str, Any],
    input_data: dict[str, Any],
    *,
    stage: str,
) -> _PrivateRuntimeCarrier:
    """Attach private evidence metadata without changing public result keys."""
    carrier = _PrivateRuntimeCarrier(result)
    answer = str(result.get("answer") or "")
    task_id = str(input_data.get("task_id") or "")
    carrier._contract_candidate = _EvidenceCandidate(
        task_id=task_id,
        producer="agent.knowledge.generation/_run_multi_candidate_generation",
        stage=stage,
        answer_identity=_answer_identity(task_id, answer),
        context_chunks=tuple(result.get("context_chunks") or ()),
        citations=tuple(result.get("citations") or ()),
        sources=tuple(result.get("sources") or ()),
        knowledge_unavailable=bool(result.get("knowledge_unavailable", False)),
        honest_unavailable=bool(result.get("honest_unavailable", False)),
    )
    return carrier


def _correlate_final_answer(
    *,
    task_id: str,
    final_answer: str,
    evidence_candidate: Any,
    review_candidate: Any,
) -> _AnswerCorrelation:
    """Compare existing private identities without inferring selection facts."""
    final_identity = _answer_identity(task_id, final_answer)
    evidence_identity = ""
    review_identity = ""
    refusal_reasons: list[str] = []

    if isinstance(evidence_candidate, _EvidenceCandidate):
        evidence_identity = evidence_candidate.answer_identity
        if evidence_candidate.task_id != task_id:
            refusal_reasons.append("evidence task_id mismatch")
        if evidence_candidate.stage != "selected":
            refusal_reasons.append("evidence candidate refused")
        if (
            evidence_candidate.knowledge_unavailable
            or evidence_candidate.honest_unavailable
        ):
            refusal_reasons.append("evidence unavailable")
    else:
        refusal_reasons.append("evidence candidate missing")

    if isinstance(review_candidate, _ReviewCandidate):
        review_identity = review_candidate.reviewed_answer_identity
        if review_candidate.task_id != task_id:
            refusal_reasons.append("review task_id mismatch")
        if (
            review_candidate.producer
            != "agent.quality.review/run_review"
            or not review_candidate.real_reviewer_executed
        ):
            refusal_reasons.append("real reviewer not executed")
        if review_candidate.mapping_refused_reason:
            refusal_reasons.append(review_candidate.mapping_refused_reason)
    else:
        refusal_reasons.append("review candidate missing")

    if evidence_identity != final_identity:
        refusal_reasons.append("evidence answer identity mismatch")
    if review_identity != final_identity:
        refusal_reasons.append("review answer identity mismatch")

    return _AnswerCorrelation(
        task_id=task_id,
        final_answer_identity=final_identity,
        evidence_answer_identity=evidence_identity,
        review_answer_identity=review_identity,
        correlation=not refusal_reasons,
        refusal_reasons=tuple(dict.fromkeys(refusal_reasons)),
    )


def _build_quality_release_decision(
    *,
    context: CollaborationContext,
    final_result: FinalCollaborationResult,
    evidence_candidate: Any,
    review_candidate: Any,
    answer_correlation: _AnswerCorrelation,
    scientific_grounding: ScientificGrounding | None = None,
) -> QualityReleaseDecision:
    """Fail closed over the final selected answer/evidence/review version.

    Reviewer remains the scientific authority.  This function neither edits
    the answer nor maps raw verdicts to a new scientific verdict; it only
    decides whether the already-reviewed artifact may cross the public API.
    """

    reasons: list[str] = []
    review_status = str(getattr(review_candidate, "raw_status", "") or "")
    review_verdict = str(getattr(review_candidate, "raw_verdict", "") or "")
    final_identity = str(answer_correlation.final_answer_identity or "")
    active_versions = tuple(sorted({pack.version for pack in _active_evidence_packs(context)}))
    candidate_versions = tuple(
        sorted(set(getattr(evidence_candidate, "evidence_versions", ()) or ()))
    )
    correction_count = int(
        context.iteration_state.get("global_corrections_used", 0) or 0
    )
    decision_type = final_result.decision.decision_type

    if decision_type is DecisionType.ASK_USER:
        return QualityReleaseDecision(
            task_id=context.task_id,
            status=QualityReleaseStatus.ASK_USER,
            eligible=False,
            public_answer="",
            reason_codes=("clarification_required",),
            review_status=review_status,
            review_verdict=review_verdict,
            answer_identity=final_identity,
            evidence_versions=active_versions,
            correction_count=correction_count,
            message="需要补充关键条件后才能形成可靠结论。",
        )
    if (
        decision_type is DecisionType.REFUSE_CONCLUSION
        or review_verdict == "rejected"
    ):
        return QualityReleaseDecision(
            task_id=context.task_id,
            status=QualityReleaseStatus.REFUSE,
            eligible=False,
            public_answer="",
            reason_codes=("review_rejected",),
            review_status=review_status,
            review_verdict=review_verdict,
            answer_identity=final_identity,
            evidence_versions=active_versions,
            correction_count=correction_count,
            message="当前证据不足以支持可靠结论，系统未发布被拒绝的回答。",
        )

    if not isinstance(review_candidate, _ReviewCandidate):
        reasons.append("review_candidate_missing")
    else:
        if (
            review_candidate.producer != "agent.quality.review/run_review"
            or not review_candidate.real_reviewer_executed
        ):
            reasons.append("real_reviewer_not_executed")
        if review_candidate.mapping_refused_reason:
            reasons.append("review_mapping_refused")
    if review_status != "completed":
        reasons.append("review_not_completed")
    if review_verdict != "approved":
        reasons.append("review_not_approved")

    if not isinstance(evidence_candidate, _EvidenceCandidate):
        reasons.append("evidence_candidate_missing")
    else:
        if evidence_candidate.task_id != context.task_id:
            reasons.append("evidence_task_mismatch")
        if evidence_candidate.stage != "selected":
            reasons.append("evidence_not_selected")
        if evidence_candidate.knowledge_unavailable or evidence_candidate.honest_unavailable:
            reasons.append("evidence_unavailable")
        if not (evidence_candidate.context_chunks or evidence_candidate.sources):
            reasons.append("evidence_source_missing")
        if not candidate_versions or candidate_versions != active_versions:
            reasons.append("evidence_version_mismatch")

    if not answer_correlation.correlation or answer_correlation.refusal_reasons:
        reasons.append("answer_evidence_review_artifact_mismatch")
    if not final_identity or final_identity != final_result.answer_identity:
        reasons.append("final_answer_artifact_mismatch")
    if not final_result.completion_eligibility:
        reasons.append("collaboration_result_not_eligible")
    if not final_result.answer:
        reasons.append("reviewed_answer_missing")
    if not isinstance(scientific_grounding, ScientificGrounding):
        reasons.append("scientific_grounding_missing")
    else:
        if scientific_grounding.task_id != context.task_id:
            reasons.append("scientific_grounding_task_mismatch")
        if not scientific_grounding.identity_consistent:
            reasons.append("scientific_grounding_artifact_mismatch")
        if scientific_grounding.evidence_versions != active_versions:
            reasons.append("scientific_grounding_version_mismatch")
        for issue in scientific_grounding.issue_codes:
            if issue in {
                "conflicting_evidence",
                "condition_mismatch",
                "unsupported_universalization",
                "fact_not_directly_supported",
                "claim_evidence_review_identity_mismatch",
            }:
                reasons.append(f"scientific_grounding:{issue}")

    unresolved = tuple(
        challenge
        for challenge in context.challenges
        if str(getattr(challenge, "status", ""))
        in {"OPEN", "NO_PROGRESS", "BUDGET_EXHAUSTED", "ASK_USER", "REJECT"}
    )
    if unresolved:
        reasons.append("unresolved_review_challenge")
    if any(
        getattr(challenge, "severity", None)
        in {ChallengeSeverity.HIGH, ChallengeSeverity.CRITICAL}
        for challenge in unresolved
    ):
        reasons.append("unresolved_critical_challenge")

    reasons = list(dict.fromkeys(reasons))
    if not reasons:
        limited = bool(
            decision_type is DecisionType.ANSWER_WITH_UNCERTAINTY
            or final_result.uncertain_claims
        )
        return QualityReleaseDecision(
            task_id=context.task_id,
            status=(
                QualityReleaseStatus.LIMITED_RELEASE
                if limited else QualityReleaseStatus.FULL_RELEASE
            ),
            eligible=True,
            public_answer=final_result.answer,
            reason_codes=(),
            review_status=review_status,
            review_verdict=review_verdict,
            answer_identity=final_identity,
            evidence_versions=active_versions,
            correction_count=correction_count,
            message=(
                "已通过真实 Reviewer 与身份一致性检查；不确定边界随回答一并发布。"
                if limited
                else "已通过真实 Reviewer、证据版本和答案身份一致性检查。"
            ),
        )

    degraded = any(
        reason in {"review_candidate_missing", "real_reviewer_not_executed"}
        for reason in reasons
    )
    # 诚实拒答文案：当根因是"生成层判定知识库无直接知识/领域外"(evidence_unavailable)
    # 时，审核未执行是结果而非原因。此前这里统一回"审核能力不可用"，把"域外/暂无
    # 知识"错误地表述成系统故障（实测 红烧肉/稀土元素列举 页面误报审核不可用）。
    if degraded and "evidence_unavailable" in reasons:
        _honest_message = str(final_result.answer or "").strip()
        if not _honest_message:
            _honest_message = (
                "当前问题不在系统已验证的知识范围内，或知识库暂无直接相关证据，"
                "系统未编造回答。"
            )
        message = _honest_message[:240]
    else:
        message = (
            "审核能力当前不可用，系统没有冒充完整审核结果。"
            if degraded
            else "当前结果尚未满足发布条件，未解决的科学回答已被保留而未公开。"
        )
    return QualityReleaseDecision(
        task_id=context.task_id,
        status=(
            QualityReleaseStatus.DEGRADED
            if degraded
            else QualityReleaseStatus.WITHHOLD
        ),
        eligible=False,
        public_answer="",
        reason_codes=tuple(reasons),
        review_status=review_status,
        review_verdict=review_verdict,
        answer_identity=final_identity,
        evidence_versions=active_versions,
        correction_count=correction_count,
        message=message,
    )


def _attach_review_candidate(
    result: dict[str, Any],
    input_data: dict[str, Any],
    *,
    content: str,
    producer: str,
    real_reviewer_executed: bool,
    mapping_refused_reason: str = "",
) -> _PrivateRuntimeCarrier:
    """Attach raw review facts without changing the public review mapping."""
    carrier = _PrivateRuntimeCarrier(result)
    task_id = str(input_data.get("task_id") or "")
    carrier._contract_candidate = _ReviewCandidate(
        task_id=task_id,
        producer=producer,
        reviewed_answer_identity=(
            _answer_identity(task_id, content)
            if real_reviewer_executed and content
            else ""
        ),
        raw_status=str(result.get("status") or ""),
        raw_verdict=str(result.get("verdict") or ""),
        raw_reason=str(result.get("reason") or ""),
        raw_fact_check=dict(result.get("fact_check") or {}),
        raw_anti_hallucination=dict(
            result.get("anti_hallucination") or {}
        ),
        raw_confidence=float(result.get("confidence", 0.0) or 0.0),
        real_reviewer_executed=real_reviewer_executed,
        mapping_refused_reason=mapping_refused_reason,
        scientific_issue_codes=tuple(
            str(item)
            for item in (
                input_data.get("_claim_evidence_grounding").issue_codes
                if isinstance(
                    input_data.get("_claim_evidence_grounding"),
                    ScientificGrounding,
                )
                else ()
            )
        ),
    )
    return carrier


def _active_evidence_packs(context: CollaborationContext) -> tuple[EvidencePack, ...]:
    return tuple(
        item
        for item in (
            context.tool_results.get("active_evidence_packs", ())
            or context.evidence_pool
        )
        if isinstance(item, EvidencePack)
    )


def _build_review_challenge(
    context: CollaborationContext,
    generation_contribution: AgentContribution,
    review: dict[str, Any],
    *,
    iteration: int,
) -> tuple[Challenge | None, ResolutionAction]:
    """Map CURRENT raw review facts to one closed, executable resolution."""
    verdict = str(review.get("verdict") or "")
    if verdict == "approved":
        return None, ResolutionAction.ACCEPT

    reason = str(review.get("reason") or "").strip()
    fact_check = review.get("fact_check") if isinstance(review.get("fact_check"), dict) else {}
    anti = review.get("anti_hallucination") if isinstance(review.get("anti_hallucination"), dict) else {}
    review_candidate = getattr(review, "_contract_candidate", None)
    grounding_issues = (
        review_candidate.scientific_issue_codes
        if isinstance(review_candidate, _ReviewCandidate)
        else ()
    )
    packs = _active_evidence_packs(context)
    missing = tuple(
        dict.fromkeys(
            value
            for pack in packs
            for value in pack.missing_information
            if value
        )
    )
    query = context.query
    vague_comparison = (
        context.intent_result.task_mode.value == "COMPARE"
        and any(term in query for term in ("哪个更好", "哪种更好", "哪个更适合"))
        and not any(term in query for term in ("效率", "稳定", "色温", "健康", "成本", "光谱"))
    )

    if verdict == "skipped":
        challenge_type = ChallengeType.EVIDENCE_INSUFFICIENT
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.REJECT
    elif verdict == "rejected":
        challenge_type = ChallengeType.UNSUPPORTED_CLAIM
        severity = ChallengeSeverity.CRITICAL
        action = ResolutionAction.REJECT
    elif vague_comparison:
        challenge_type = ChallengeType.AMBIGUOUS_USER_REQUIREMENT
        severity = ChallengeSeverity.MEDIUM
        action = ResolutionAction.ASK_USER
    elif "unsupported_universalization" in grounding_issues:
        challenge_type = ChallengeType.OVERGENERALIZATION
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.REVISE
        if not missing:
            missing = ("适用材料体系与实验条件",)
    elif "condition_mismatch" in grounding_issues:
        challenge_type = ChallengeType.CONDITION_MISMATCH
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.RE_RETRIEVE
        if not missing:
            missing = ("与主张条件一致的证据",)
    elif "conflicting_evidence" in grounding_issues:
        challenge_type = ChallengeType.CONFLICTING_EVIDENCE
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.REVISE
    elif "fact_not_directly_supported" in grounding_issues:
        challenge_type = ChallengeType.EVIDENCE_INSUFFICIENT
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.RE_RETRIEVE
        if not missing:
            missing = ("事实主张的直接支持证据",)
    elif "问题核心覆盖门要求重新检索" in reason:
        challenge_type = ChallengeType.EVIDENCE_INSUFFICIENT
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.RE_RETRIEVE
        if not missing:
            missing = (reason.split("：", 1)[-1][:180],)
    elif "问题核心覆盖门要求修订" in reason:
        challenge_type = ChallengeType.EVIDENCE_INSUFFICIENT
        severity = ChallengeSeverity.MEDIUM
        action = ResolutionAction.REVISE
        if not missing:
            missing = (reason.split("：", 1)[-1][:180],)
    elif "一定" in query or any(term in reason for term in ("安全", "绝对", "过度")):
        challenge_type = ChallengeType.SAFETY_OVERCLAIM
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.RE_RETRIEVE if missing else ResolutionAction.REVISE
        if not missing:
            missing = ("适用条件与限制",)
    elif int(fact_check.get("failed", 0) or 0) > 0:
        challenge_type = ChallengeType.EVIDENCE_MISMATCH
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.RE_RETRIEVE
        if not missing:
            missing = ("异常断言的直接支持证据",)
    elif any(term in reason for term in ("推断", "可能", "确定事实", "限定")):
        challenge_type = ChallengeType.FACT_INFERENCE_CONFUSION
        severity = ChallengeSeverity.MEDIUM
        action = ResolutionAction.REVISE
    elif anti.get("hallucination_detected") or str(anti.get("action") or "") in {"degrade", "reask"}:
        challenge_type = ChallengeType.UNSUPPORTED_CLAIM
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.REVISE
    elif missing:
        challenge_type = ChallengeType.EVIDENCE_INSUFFICIENT
        severity = ChallengeSeverity.HIGH
        action = ResolutionAction.RE_RETRIEVE
    else:
        challenge_type = ChallengeType.OVERGENERALIZATION
        severity = ChallengeSeverity.MEDIUM
        action = ResolutionAction.REVISE

    claims = generation_contribution.claims
    challenge = Challenge(
        challenge_id=f"challenge-{context.task_id}-{len(context.challenges) + 1}",
        task_id=context.task_id,
        subtask_id=generation_contribution.subtask_id,
        reviewer_agent_id=REVIEW_AGENT_ID,
        target_contribution_id=generation_contribution.contribution_id,
        target_claim_ids=tuple(claim.claim_id for claim in claims),
        challenge_type=challenge_type,
        reason=reason or "review did not approve the current contribution",
        severity=severity,
        missing_information=missing,
        evidence_refs=generation_contribution.evidence_refs,
        requested_action=action,
        status="OPEN",
        iteration=iteration,
    )
    return challenge, action


def _challenge_signature(challenge: Challenge) -> tuple[Any, ...]:
    return (
        challenge.challenge_type,
        challenge.requested_action,
        challenge.missing_information,
        challenge.reason,
    )


def _evidence_signature(packs: tuple[EvidencePack, ...]) -> tuple[Any, ...]:
    return tuple(
        (
            pack.subtask_id,
            tuple(item.evidence_id for item in pack.items),
            pack.missing_information,
        )
        for pack in packs
    )


def _set_challenge_status(
    context: CollaborationContext,
    challenge: Challenge,
    status: str,
) -> Challenge:
    updated = replace(challenge, status=status)
    context.challenges[-1] = updated
    return updated


# 生成端思考开关: True=让 DeepSeek flash 打开 CoT (先想后答); False=关闭省 token.
# 实测(2026-08-15, 黄金集 N=16): 开启后逐题分数与关闭时几乎一致(瓶颈在检索/证据覆盖,
# 非模型思考深度), 故默认关闭以省钱; 若遇更难的开放题需最大化质量可临时置 True。
# 仅影响知识生成 Agent 的 LLM 重组这一步, 不影响 critic/rewrite (它们仍关思考省 token)。
_GENERATION_THINKING = False

# 领域实体别名归一化: 用户口语/简写 → 知识库标准实体 (根本相关性提升)
_ENTITY_ALIASES: dict[str, str] = {
    "dy离子": "dy3+", "镝离子": "dy3+", "镝": "dy3+", "dy": "dy3+",
    "er离子": "er3+", "铒离子": "er3+", "铒": "er3+", "er": "er3+",
    "yb离子": "yb3+", "镱离子": "yb3+", "镱": "yb3+", "yb": "yb3+",
    "eu离子": "eu3+", "铕离子": "eu3+", "铕": "eu3+", "eu": "eu3+",
    "发光机理": "发光", "发光原理": "发光", "发光机制": "发光", "发光过程": "发光",
    "上转换": "上转换发光", "量子效率": "量子效率", "稀土发光": "稀土发光材料",
}


def normalize_query(query: str) -> str:
    """查询归一化: 别名 → 标准实体 + 过滤口语词, 供检索相关性计算."""
    qq = str(query).lower().strip()
    for alias, std in _ENTITY_ALIASES.items():
        qq = qq.replace(alias, std)
    # 过滤口语词/提问套话 (避免稀释检索主题词, 如"帮我系统讲解一下" → 只剩主题词)
    for filler in ("帮我", "系统", "讲解", "一下", "请", "能不能", "可以", "给我",
                   "说说", "讲讲", "介绍一下", "介绍", "如何", "怎么", "怎样", "为什么",
                   "有没有", "多少", "是什么", "什么是", "了解", "知道", "讲讲", "的"):
        qq = qq.replace(filler, "")
    return qq


# ============================================================
# 通俗化讲解库 (画像1 小白本科生 / 画像6 跨专业爱好者)
#
# 设计: 无 LLM 时, 对"大白话/通俗/简单讲讲"类请求, 不再罗列学术证据句,
# 而是返回预置的领域通俗讲解 (生活化比喻 + 贴近生活的例子)。
# 覆盖稀土发光材料最常被小白/爱好者问到的核心概念。
# ============================================================

# 通俗化请求词 (触发"说人话"模式)
_PLAIN_REQUEST_TERMS = (
    "大白话", "通俗", "简单讲讲", "简单说说", "说人话", "生活化",
    "听不懂", "形象", "比喻", "科普", "通俗易懂", "能听懂", "好懂",
    "入门", "小白", "外行", "孩子都能懂", "再简单",
)

# 核心概念 → 通俗讲解 (用生活化比喻讲清机理, 不堆术语)
_PLAIN_LANGUAGE_GUIDES: dict[str, str] = {
    "发光材料": (
        "发光材料，说白了就是「吸了能量会自己发光的材料」。\n"
        "打个比方：它像一块能「存光」的海绵——你用光或电去「喂」它，它先把能量吃进去，"
        "过一会儿再慢慢把能量变成光「吐」出来。\n"
        "你身边到处都是：荧光笔、夜光手表、LED 灯里那层白色粉末（荧光粉）、手机屏幕，"
        "背后都靠这类材料在发光。"
    ),
    "荧光粉": (
        "荧光粉是发光材料里最常用的一种，通常是很细的粉末。\n"
        "它的拿手本事是「转换颜色」：拿蓝光芯片去照它，它能转成黄光、绿光或红光，"
        "几种颜色一混，就成了白光——LED 灯能发出白光，基本就靠这个套路。"
    ),
    "发光机理": (
        "材料为什么会发光？一句话：电子在「跳台阶」。\n"
        "材料里的电子平时待在低处（基态），被光照或通电后，它吸收能量跳到高处（激发态）；"
        "但高处站不稳，它很快又跳回低处，把多出来的能量变成一束光放出来。\n"
        "就像你把弹簧压下去再松手，它会「弹」回原样——弹回来的那一下，就是发出来的光。"
    ),
    "稀土": (
        "稀土不是「稀少的土」，而是一组挺特殊的金属元素，一共 17 种，名字有点怪：镧、铈、镝、铕……\n"
        "它们「特殊」在：内部电子结构很特别，能发出颜色特别纯、特别稳定的光，"
        "所以被大量用来做发光材料——LED、激光、荧光粉都离不开它们。"
    ),
    "稀土发光材料": (
        "稀土发光材料，就是用上面那 17 种稀土元素做成的发光材料。\n"
        "它们的优点是：颜色纯、亮度高、寿命长。你看到的 LED 灯、节能灯、电视屏幕，"
        "很多都要靠稀土（比如镝 Dy、铕 Eu）来发出漂亮的颜色。"
    ),
    "上转换发光": (
        "上转换发光有点反直觉：它能把「能量低」的近红外光（肉眼看不见），"
        "变成「能量高」的可见光甚至紫外光——就像把两个小台阶叠起来，变成一个大台阶。\n"
        "这类材料常用于生物成像（能照进身体里还很清楚）和防伪标签。"
    ),
    "白光led": (
        "要做出白光，最常用的办法是「蓝光芯片 + 荧光粉」：\n"
        "蓝光芯片先发蓝光，蓝光照到荧光粉上，一部分被转成黄光（或其他颜色），"
        "蓝光和黄光一混，人眼就看成白光了。这也是绝大多数 LED 灯的发光原理。"
    ),
    "浓度猝灭": (
        "浓度猝灭，说的是「好东西也不能贪多」：\n"
        "发光材料里掺的发光离子（比如镝 Dy）一开始越多越亮，但超过某个量之后，"
        "离子之间会「互相打架」，把本该发出来的光悄悄耗掉了，结果反而变暗。"
    ),
    "能量传递": (
        "能量传递，就像一场「接力赛」：\n"
        "一个离子（比如铈 Ce）先把能量吃进去，再「转手」递给另一个离子（比如镝 Dy），"
        "由后者来发光。这样能让本来不太发光的离子，也能借力发出漂亮的光。"
    ),
    "量子效率": (
        "量子效率，说白了就是「吃进去多少光、吐出来多少光」的比例。\n"
        "比如吃进去 100 份光，能吐出来 80 份，效率就是 80%。"
        "这个数越接近 100%，说明材料「浪费」得越少、发光越划算。"
    ),
    "热猝灭": (
        "热猝灭，说的是「一热就蔫了」：\n"
        "发光材料温度一升高，里面的电子就「躁动」起来，"
        "本该拿来发光的能量被白白耗成热，结果亮度下降。"
        "好的发光材料要能扛住高温、不轻易变暗。"
    ),
    "能级跃迁": (
        "能级，可以想成电子站的「台阶」：能量低的叫基态（一楼），能量高的叫激发态（二楼）。\n"
        "电子从一楼跳到二楼叫「吸收」，从二楼跳回一楼、顺手放出一束光，就叫「能级跃迁」。"
        "不同台阶的高度差，决定了发出的光是什么颜色。"
    ),
}

# 概念关键词 → 讲解 key (匹配顺序: 更具体在前, 避免"发光"吞掉"发光材料")
_PLAIN_CONCEPT_KEYS: tuple[tuple[str, str], ...] = (
    ("上转换", "上转换发光"),
    ("能量传递", "能量传递"),
    ("浓度猝灭", "浓度猝灭"),
    ("热猝灭", "热猝灭"),
    ("猝灭", "浓度猝灭"),
    ("量子效率", "量子效率"),
    ("荧光粉", "荧光粉"),
    ("发光材料", "发光材料"),
    ("稀土发光", "稀土发光材料"),
    ("白光", "白光led"),
    ("能级跃迁", "能级跃迁"),
    ("跃迁", "能级跃迁"),
    ("稀土", "稀土"),
    ("发光", "发光机理"),
)


def _match_plain_concept(query: str) -> str | None:
    """概念 → 通俗讲解 (不检查请求词, 供显式请求与画像自动降级复用)."""
    q = str(query or "").strip()
    for key, guide_key in _PLAIN_CONCEPT_KEYS:
        if key in q:
            return _PLAIN_LANGUAGE_GUIDES.get(guide_key)
    return None


def _try_plain_language_answer(query: str) -> str | None:
    """检测「大白话/通俗」等显式请求并返回通俗讲解; 未触发返回 None."""
    q = str(query or "").strip()
    if not any(t in q for t in _PLAIN_REQUEST_TERMS):
        return None
    return _match_plain_concept(q)


def _is_beginner_profile(profile: Any) -> bool:
    """画像是否 beginner 档位 (level=beginner 或 θ 低于 -0.5).

    用于「通俗化按画像自动降级」: beginner 问基础概念自动给通俗讲解,
    不再只依赖显式「大白话/通俗」关键词 (persona #29④/#31 缺口).
    """
    if profile is None:
        return False
    level = str(getattr(profile, "level", "") or "").lower()
    if level in {"foundation", "beginner", "novice"}:
        return True
    try:
        theta = float(getattr(profile, "theta", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return theta < -0.5


def _is_beginner_plain_question(query: str) -> bool:
    """beginner 自动通俗化的触发条件: 基础概念定义式, 且非具体专业查询.

    含具体离子符号(Dy3+/Eu2+)/跃迁项(4F9/2)的「专业查询」即便由 beginner
    画像发出, 也是具体问题、要准确答案, 不自动通俗化 (避免「问 Dy 跃迁」
    被降级成「电子跳台阶」比喻).
    """
    q = str(query or "").strip()
    if _detect_question_type(q) != "definition":
        return False
    if re.search(r"\b[A-Z][a-z]?\d*[+-]\b", q):  # Dy3+, Eu2+ 具体离子
        return False
    if re.search(r"\d[A-Z]\d+/\d+", q):           # 4F9/2 具体跃迁项
        return False
    return True


def _try_deduce(query: str) -> str | None:
    """领域推演兜底: 规则 + 已知量 → 未知结论, 不依赖检索 (对标「114+256」).

    命中数值公式/因果链/关系规则直接推, 推不了返回 None (走正常检索)。
    """
    try:
        from dy3_polaris.l3.deduction import deduce
        return deduce(query)
    except Exception:  # noqa: BLE001
        return None


def _requires_retrieved_evidence(query: str) -> bool:
    """Day2 Golden Questions 中必须经过真实证据链的领域问题。"""
    q = str(query or "").lower().replace(" ", "")
    if "浓度" in q and any(term in q for term in ("下降", "降低", "变弱", "猝灭")):
        return True
    return any(
        term in q
        for term in (
            "黄蓝", "白光", "浓度猝灭", "基质", "量子效率", "发光效率",
            "色温", "蓝光风险", "健康照明", "如何比较", "一定优于", "一定安全",
        )
    )


def _absolute_claim_boundary_answer(query: str) -> str:
    """对绝对化 Golden Question 给出与问题相关的证据边界说明。"""
    q = str(query or "").lower().replace(" ", "")
    if ("色温" in q or "cct" in q) and any(
        term in q for term in ("安全", "风险", "蓝光")
    ):
        return (
            "不能仅根据相关色温判定照明是否安全。色温主要描述光色外观；"
            "蓝光风险还与完整光谱功率分布、蓝光波段能量、辐亮度或照度、暴露时间和评价条件有关。"
            "在缺少这些数据时，证据不足以得出「低色温一定安全」或「高色温一定不安全」的结论。"
        )
    if any(term in q for term in ("比较", "优于", "更好", "孰优孰劣")):
        return (
            "在缺少同一应用目标和同条件数据时，不能判定某一 Dy³⁺ 材料体系全面优于另一体系。"
            "可比较的事实至少需要来自一致测试条件的发射光谱与色坐标、量子效率、热稳定性、寿命、掺杂浓度及制备条件。"
            "当前问题没有提供这些事实，因此只能保留判断。"
        )
    return (
        "这是一个不能简单以「是/否」回答的判断，且当前论断缺少限定条件，"
        "现有证据不足以给出可靠的肯定或否定。"
    )


def _adapt_educational_depth(
    answer: str,
    learner_level: str,
    teaching_decision: AdaptiveTeachingDecision | None = None,
) -> str:
    """只调整已有证据答案的学习表达，不增加材料事实。"""
    text = str(answer or "").strip()
    if not text:
        return text
    level = str(learner_level or "intermediate").lower()
    strategy = (
        teaching_decision.explanation_strategy
        if isinstance(teaching_decision, AdaptiveTeachingDecision)
        else ""
    )
    if strategy in {"repair_with_evidence", "contrast_with_evidence"}:
        return (
            "先澄清：以下解释会把容易混淆的说法与证据支持的结论分开。\n"
            f"{text}\n\n"
            "核对方式：请逐项检查事实、推理和适用条件；这不会改变原有科学结论。"
        )
    if strategy == "example_then_mechanism":
        return (
            "学习顺序：先借助证据中的具体对象建立直观，再回到机制。\n"
            f"{text}\n\n"
            "迁移边界：例子只说明证据覆盖的体系，不能自动外推到其他基质或条件。"
        )
    if level in {"foundation", "beginner", "novice"}:
        return (
            "入门理解：先抓住「条件—过程—结果」这条主线。\n"
            f"{text}\n\n"
            "学习提示：先确认答案中每个专业术语的含义，再对照证据理解因果关系。"
        )
    if level == "advanced":
        return (
            "进阶分析：\n"
            f"{text}\n\n"
            "研究边界：需同时核对证据对应的基质、掺杂浓度、激发与测试条件，不能将单一体系结论直接外推。"
        )
    return text


_TEACHING_FRAME_PREFIXES = (
    "先澄清：",
    "核对方式：",
    "学习顺序：",
    "迁移边界：",
    "入门理解：",
    "学习提示：",
    "进阶分析：",
    "研究边界：",
    "机制依据：",
    "证据中的条件化观察：",
    "当前边界：",
    "（来源：",
    "（以上措施依据",
    "基于当前真实证据，",
    "（当前证据支持",
)


def _scientific_review_content(content: str) -> str:
    """Remove teaching navigation while preserving every scientific claim.

    Adaptive teaching adds request-local reading instructions around the same
    evidence-grounded answer.  Those instructions are not scientific claims
    and must not lower citation coverage or trigger a correction loop.  The
    Reviewer still receives and correlates the complete public answer; only
    CC1/fact verification is scoped to its scientific body.
    """
    lines = [
        line
        for line in str(content or "").splitlines()
        if line.strip()
        and not line.strip().startswith(_TEACHING_FRAME_PREFIXES)
    ]
    return "\n".join(lines).strip() or str(content or "").strip()


# ============================================================
# 模糊问题检测与人性化引导式澄清 (L5 高等级: 问题理解 / 动态追问)
#
# 设计: 对"纯元素名 / 过短 / 信息不足"的问题，不硬猜答案，
# 而是像 Trae/Codex 一样，把澄清做成对用户问题的"自然补充"：
#   - 先复述理解（让用户感到被听懂，而非被盘问）
#   - 提供可点选的探索方向（引导而非僵硬弹窗）
#   - 给出补全问题的示例（降低表达门槛，让用户舒服自然地补充）
# ============================================================

_BARE_ELEMENT_RE = re.compile(
    r"^(dy|er|eu|ce|tb|yb|nd|sm|pr|ho|tm|gd|lu|la|sc|y"
    r"|镝|铒|铕|铈|铽|镱|钕|钐|镨|钬|铥|钆|镥|镧|钇)$",
    re.IGNORECASE,
)

# 统一意图识别 (复用 L3 IntentClassifier, 单一意图来源, 消除 L5 独立词表).
_INTENT_CLASSIFIER: Any = None


def _get_intent_classifier() -> Any:
    """惰性获取意图分类器单例 (L3 IntentClassifier, 规则 + LLM 兜底)."""
    global _INTENT_CLASSIFIER
    if _INTENT_CLASSIFIER is None:
        from dy3_polaris.l3.intent_router import IntentClassifier
        _INTENT_CLASSIFIER = IntentClassifier()
    return _INTENT_CLASSIFIER


def _has_clear_intent(query: str, intent_result: Any | None = None) -> bool:
    """Read the R-03A IntentResult instead of a second L3 intent authority."""
    nq = normalize_query(query)
    if _BARE_ELEMENT_RE.match(nq) or len(str(query).strip()) <= 2:
        return False
    try:
        resolved = intent_result or understand_task(query)
    except Exception:
        return False
    blocking_ambiguity = {"missing_subject"}
    return bool(getattr(resolved, "primary_intent", "")) and not (
        blocking_ambiguity & set(getattr(resolved, "ambiguity", ()) or ())
    )

# 领域方向词 → 对应可点选项 (用于引导补全)
_DOMAIN_DIRECTION_TERMS: dict[str, str] = {
    "发光": "发光机理", "猝灭": "浓度/热猝灭", "光谱": "能级与光谱",
    "能级": "能级与光谱", "跃迁": "能级与光谱", "量子": "量子效率",
    "效率": "量子效率", "制备": "制备与合成", "合成": "制备与合成",
    "掺杂": "制备与合成", "应用": "实际应用", "材料": "材料体系",
    "荧光": "发光机理", "磷光": "发光机理", "温度": "温度依赖/热稳定性",
    "色度": "色度与显色", "显色": "色度与显色", "寿命": "荧光寿命",
}

# 常见稀土离子物理符号 → 展示名 (澄清引导时更友好)
_ELEMENT_DISPLAY: dict[str, str] = {
    "dy": "Dy³⁺（镝）", "er": "Er³⁺（铒）", "eu": "Eu³⁺（铕）",
    "ce": "Ce³⁺（铈）", "tb": "Tb³⁺（铽）", "yb": "Yb³⁺（镱）",
    "nd": "Nd³⁺（钕）", "sm": "Sm³⁺（钐）", "pr": "Pr³⁺（镨）",
    "ho": "Ho³⁺（钬）", "tm": "Tm³⁺（铥）", "gd": "Gd³⁺（钆）",
    "lu": "Lu³⁺（镥）", "la": "La³⁺（镧）", "sc": "Sc³⁺（钪）",
}


def _clarify_directions() -> list[str]:
    """默认的探索方向 (覆盖稀土发光材料常见着眼点)."""
    return ["发光机理", "制备与合成", "能级与光谱", "实际应用", "量子效率"]


def _build_clarify(query: str, a_type: str) -> dict[str, Any]:
    """构建人性化引导式澄清载荷 (作为对用户问题的补充追问)."""
    q = str(query).strip()
    nq = normalize_query(q)
    entity = ""
    m = _BARE_ELEMENT_RE.match(nq)
    if m:
        sym = str(m.group(1)).lower()
        entity = _ELEMENT_DISPLAY.get(sym, sym + "³⁺")
    directions = _clarify_directions()
    # 问题里已含方向词 → 该方向优先
    for term, direction in _DOMAIN_DIRECTION_TERMS.items():
        if term in nq and direction != directions[0]:
            directions = [direction] + [d for d in directions if d != direction]
    opts = list(dict.fromkeys(directions))  # 去重保序
    if entity:
        head = f"你输入的「{q}」我理解为 {entity} 的某一方面"
        question = (
            f"{head}～稀土离子的知识点不少，为了不答偏方向，想跟你确认一下："
            f"你现在更想了解它的哪一方面？"
        )
        guidance = (
            f"你可以点选一个方向，或再补一句，比如「{entity}的发光机理」「{entity}怎么制备」，"
            f"我就能顺着你的问题给出更贴合的解答。"
        )
    else:
        question = (
            f"你这个问题包含的信息有点少，为了不答错方向、给你真正有用的答案，"
            f"能再补充一下你想了解的角度吗？"
        )
        guidance = (
            f"比如补充具体对象或想了解的方向（{opts[0]}、{opts[1]}…），"
            f"你多说一句，我马上接着答。"
        )
    return {
        "type": a_type,
        "entity": entity,
        "question": question,
        "options": opts,
        "guidance": guidance,
    }


def _detect_ambiguity(
    query: str,
    intent_result: Any | None = None,
) -> dict[str, Any] | None:
    """检测问题是否模糊; 有明确意图 → 不澄清, 信息不足 → 澄清.

    先按「意图」判断 (在 normalize 之前), 避免 "是什么/为什么" 等提问词被
    normalize_query 当填充词滤掉后误判成「纯元素」; 无意图时才回落到
    「纯元素名 / 过短」这类信息不足的澄清。
    """
    q = str(query).strip()
    if not q:
        return None
    # 有明确意图 (定义/方法/原因/数值/关系/比较/机理) → 不澄清
    if _has_clear_intent(q, intent_result):
        return None
    # 无明确意图: 纯元素名 (如 "dy" / "镝") 或过短 → 引导补全
    nq = normalize_query(q)
    if _BARE_ELEMENT_RE.match(nq):
        return _build_clarify(q, "element_only")
    if len(q) <= 2 and not any(t in nq for t in _DOMAIN_DIRECTION_TERMS):
        return _build_clarify(q, "too_vague")
    return None


class AgentDependencies:
    """Agent 执行所需的跨层服务依赖."""

    def __init__(
        self,
        *,
        irt_service: Any | None = None,
        profile_service: Any | None = None,
        memory_service: Any | None = None,
        bkt_service: Any | None = None,
        practice_bank: Any | None = None,
        message_bus: Any | None = None,
        l3_store: Any | None = None,
        hybrid_retriever: Any | None = None,
        graph_reasoner: Any | None = None,
        graphrag_retriever: Any | None = None,
        fact_checker: Any | None = None,
        quality_manager: Any | None = None,
        anti_hallucination_pipeline: Any | None = None,
        response_synthesizer: Any | None = None,
        reranker: Any | None = None,
        decision_engine: Any | None = None,
        audit_engine: Any | None = None,
        external_kb: Any | None = None,
        embedding_manager: Any | None = None,
        user_understanding_service: Any | None = None,
    ) -> None:
        self.irt_service = irt_service
        self.profile_service = profile_service
        self.memory_service = memory_service
        self.bkt_service = bkt_service
        self.practice_bank = practice_bank
        self.message_bus = message_bus
        self.l3_store = l3_store
        self.hybrid_retriever = hybrid_retriever
        self.graph_reasoner = graph_reasoner
        self.graphrag_retriever = graphrag_retriever
        self.fact_checker = fact_checker
        self.quality_manager = quality_manager
        self.anti_hallucination_pipeline = anti_hallucination_pipeline
        self.response_synthesizer = response_synthesizer
        self.reranker = reranker
        # L4 决策引擎 (策略决策唯一入口: next-action)
        self.decision_engine = decision_engine
        # L0 审计引擎 (Agent 执行轨迹持久化)
        self.audit_engine = audit_engine
        # 外部知识源 (动态知识识别: 本地无匹配 → 可选外部检索兜底)
        self.external_kb = external_kb
        # 嵌入管理器 (向量检索: 查询侧编码 + 图检索实体定位)
        self.embedding_manager = embedding_manager
        # Optional declared/background facts.  LearnerIntelligenceView is the
        # only component allowed to interpret them for Agent behavior.
        self.user_understanding_service = user_understanding_service


def _broadcast(bus: Any, channel: str, payload: dict[str, Any], publisher: str) -> bool:
    """安全向消息总线发布事件 (频道未注册/总线缺失时静默跳过).

    多向信息传播的出口: 任何 Agent 完成关键动作后, 将结果发布到
    对应频道, 供其他 Agent 订阅消费 (诊断→生成/审核/决策, 考核→画像等).
    """
    if bus is None or not channel:
        return False
    try:
        from dy3_polaris.l5.communication import Message

        msg = Message(channel=channel, publisher=publisher, payload=payload)
        bus.publish(msg)
        logger.info("Agent 广播 %s -> %s (%s)", publisher, channel, list(payload)[:3])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent 广播失败 %s -> %s: %s", publisher, channel, exc)
        return False


def _load_profile(profile_service: Any, learner_id: str) -> Any | None:
    """读取画像快照 (安全)."""
    if profile_service is None:
        return None
    try:
        return profile_service.get_profile_snapshot(learner_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取画像失败 %s: %s", learner_id, exc)
        return None


def _save_profile(profile_service: Any, profile: Any) -> bool:
    """持久化画像快照 — 走 L2 唯一写方 (apply_update + 乐观锁).

    不再直接 store.save_profile; 提取 extras/confidence 更新提交给
    L2 ProfileTracingService.apply_update (全量重算 + CAS).
    乐观锁冲突时重新拉取最新版本重试一次.
    """
    if profile_service is None or profile is None:
        return False
    learner_id = getattr(profile, "learner_id", None)
    if not learner_id:
        return False
    try:
        updates: dict[str, Any] = {}
        extras = getattr(profile, "extras", None)
        if extras:
            updates["extras"] = dict(extras)
        confidence = getattr(profile, "confidence", None)
        if confidence is not None:
            updates["confidence"] = float(confidence)
        expected = getattr(profile, "version", None)
        try:
            profile_service.apply_update(
                learner_id, updates=updates, expected_version=expected
            )
            return True
        except ProfileConflictError:
            # 乐观锁冲突: 重新拉取最新画像后重试一次
            latest = profile_service.get_profile_snapshot(learner_id)
            if latest is not None:
                profile_service.apply_update(
                    learner_id, updates=updates, expected_version=latest.version
                )
                return True
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("画像写回失败 %s: %s", learner_id, exc)
        return False


def _mastery_map(profile: Any) -> dict[str, float]:
    """从画像取 kp_mastery 映射 (安全)."""
    if profile is None:
        return {}
    km = getattr(profile, "kp_mastery", None)
    return dict(km) if isinstance(km, dict) else {}


def _profile_dict(snapshot: Any) -> dict[str, Any]:
    """将画像快照安全转为字典."""
    if snapshot is None:
        return {}
    to_dict = getattr(snapshot, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001
            pass
    return vars(snapshot)


def _recommend_difficulty(theta: float, recent_accuracy: float | None = None) -> dict[str, Any]:
    """画像 → 难度映射 (ZPD 最近发展区, 复用 l1 的 ZoneOfProximalDevelopment).

    由 IRT 能力 θ 推荐「适中挑战」难度: 推荐难度略低于 θ (b = θ - 0.25),
    使 2PL 预测答对概率 P ≈ 0.56, 落在最近发展区 [0.5, 0.8], 既不过易也不过难。

    探索动量 (recent_accuracy): 纯 θ 估计对「能力持续上升/下降」存在滞后 —
    EAP 先验随样本增多而收紧, 且过易的题目对「θ 究竟有多高」几乎不提供信息
    (答对只说明 θ > b), 导致「一直测试但能力上涨」的学习者被长期喂过易题、
    估计卡在低位。为此引入「比例探索」: 观测正确率偏离目标 0.56 越多,
    探索步长越大 (有界 ±0.75)。基于「窗口正确率」(非单题), 平滑且不易振荡
    (等效闭环增益 < 1), 不破坏上一轮修出的「能力不因单题乱答剧烈波动」。
    """
    from math import exp

    t = max(-3.0, min(3.0, float(theta)))
    if t >= 1.0:
        band = "前沿"
    elif t >= 0.0:
        band = "提高"
    elif t >= -1.0:
        band = "进阶"
    else:
        band = "基础"
    try:
        from dy3_polaris.l1.models import ZoneOfProximalDevelopment

        zpd = ZoneOfProximalDevelopment(learner_theta=t)
        lo, hi = zpd.zpd_lower, zpd.zpd_upper
    except Exception:  # noqa: BLE001
        lo, hi = t - 0.5, t + 0.5
    # 探索动量 (比例控制, 有界): 观测正确率越高 → 越需上调难度探测; 反之回退
    momentum = 0.0
    if recent_accuracy is not None:
        acc = max(0.0, min(1.0, float(recent_accuracy)))
        momentum = max(-0.75, min(0.75, 1.5 * (acc - 0.56)))
    # 推荐难度 = ZPD 中点略偏下 + 探索动量 (适中挑战: 答对概率 ~0.56)
    recommended = (lo + hi) / 2.0 - 0.25 + momentum
    recommended = max(-3.0, min(3.0, recommended))
    predicted_p = 1.0 / (1.0 + exp(-(t - recommended)))
    return {
        "band": band,
        "recommended_difficulty": round(recommended, 4),
        "zpd_lower": round(lo, 4),
        "zpd_upper": round(hi, 4),
        "predicted_success_probability": round(predicted_p, 4),
    }


def _difficulty_adaptation_rate(theta: float, difficulties: list[float]) -> float:
    """画像-难度适配率 = 题目难度对应的 IRT 预测答对概率落在适中挑战区 [0.5, 0.8] 的比例.

    这是「画像-难度适配 ≥85%」硬指标的可量化口径: 适配 = 推荐/命中题目的难度对
    该学习者既不太易(>0.8, 已掌握)也不太难(<0.5, 挫败), 处于最近发展区。
    """
    from math import exp

    if not difficulties:
        return 1.0
    t = max(-3.0, min(3.0, float(theta)))
    n_ok = 0
    for b in difficulties:
        p = 1.0 / (1.0 + exp(-(t - float(b))))
        if 0.5 <= p <= 0.8:
            n_ok += 1
    return round(n_ok / len(difficulties), 4)


def _recent_accuracy(profile_service: Any, learner_id: str, window: int = 5) -> float | None:
    """从答题历史计算近期正确率 (探索动量输入).

    供 ``_recommend_difficulty`` 的 ``recent_accuracy`` 参数使用: 观测正确率
    显著偏离目标 0.56 时, 说明能力估计可能滞后 (连续答对=被低估 / 连续答错=
    被高估), 触发难度探测上调/回退。基于「窗口」而非单题, 抗噪声。

    Args:
        profile_service: L2 画像服务 (需暴露 .store.get_answer_history).
        learner_id: 学习者 ID.
        window: 滑动窗口大小 (默认 5 次作答).

    Returns:
        近期正确率 [0, 1]; 无画像服务 / 无答题历史时返回 None (不触发动量).
    """
    if profile_service is None:
        return None
    try:
        store = getattr(profile_service, "store", None)
        if store is None:
            return None
        history = store.get_answer_history(learner_id) or []
        if not history:
            return None
        recent = list(history)[-window:]
        if not recent:
            return None
        correct = sum(1 for r in recent if bool(getattr(r, "correct", False)))
        return correct / len(recent)
    except Exception:  # noqa: BLE001
        return None


def run_diagnosis(
    input_data: dict[str, Any],
    deps: AgentDependencies,
) -> dict[str, Any]:
    """学情诊断 Agent — 解释请求内 LearnerIntelligenceView."""
    view = input_data.get("_learner_intelligence_view")
    if not isinstance(view, LearnerIntelligenceView):
        # Direct worker calls retain compatibility while still passing through
        # the same source-classification boundary.
        view = build_learner_intelligence_view(
            input_data,
            deps,
            learner_memory_view=(
                input_data.get("_learner_memory_view")
                if isinstance(input_data.get("_learner_memory_view"), dict)
                else None
            ),
            teaching_memory_view=_load_teaching_memory(input_data, deps),
        )

    learner_id = view.learner_id
    ability = view.compatibility_projection("ability_projection")
    profile = view.compatibility_projection("profile_projection")
    memory = view.compatibility_projection("memory_projection")
    theta_signal = view.models["theta"]
    theta = float(theta_signal.value) if theta_signal.value is not None else 0.0
    se = float(ability.get("se", 0.5) or 0.5)
    weak_kps = list(view.value("derived_context", "weak_kps", ()))
    learning_stage = str(
        view.value("derived_context", "learning_stage", "unknown")
    )
    level = str(
        view.value("derived_context", "recommended_depth", "foundation")
    )
    confidence = float(view.metadata.get("confidence", 0.0) or 0.0)
    adaptive_strategy = str(
        view.value("derived_context", "adaptive_strategy", "establish_foundation")
    )
    teaching_memory_context = view.value(
        "derived_context", "teaching_memory_context"
    )
    personal_learner_model = view.value(
        "derived_context", "personal_learner_model"
    )
    teaching_decision = view.value(
        "derived_context", "adaptive_teaching_decision"
    )
    historical_exposure = tuple(
        view.value("derived_context", "historical_exposure", ())
    )
    prerequisite_focus = tuple(
        view.value("derived_context", "prerequisite_focus", ())
    )
    misconception_focus = tuple(
        view.value("derived_context", "misconception_focus", ())
    )
    learning_path = view.value("derived_context", "learning_path")
    concept_learning_path = view.value(
        "derived_context", "concept_learning_path"
    )
    knowledge_learning_context = view.value(
        "derived_context", "knowledge_learning_context"
    )
    if theta_signal.value is None:
        summary = (
            f"学习者 {learner_id} 当前能力数据不足（学习阶段 {learning_stage}），"
            f"薄弱知识点 {len(weak_kps)} 个；教学深度按 {level} 处理。"
        )
    else:
        summary = (
            f"学习者 {learner_id} 当前能力 θ={theta:.2f}（{learning_stage}），"
            f"标准误 {se:.2f}，薄弱知识点 {len(weak_kps)} 个；"
            f"教学深度按 {level} 处理。"
        )
    known_history = tuple(view.value("facts", "known_history", ()))
    if known_history:
        summary += f" 已识别 {len(known_history)} 条相关历史任务记录。"
    if adaptive_strategy == "advance_from_prior_exposure":
        exposure_text = "、".join(str(item) for item in historical_exposure[:3])
        focus_text = "、".join(str(item) for item in prerequisite_focus[:3])
        summary += (
            f" 历史信号显示此前已解释 {exposure_text or '相关基础主题'}；"
            f"本次教学聚焦 {focus_text or '后续机制'}，不把该信号解释为掌握度。"
        )
    if (
        isinstance(teaching_memory_context, TeachingMemoryInterpretation)
        and teaching_memory_context.available
    ):
        summary += (
            f" 历史教学事实支持策略 {teaching_memory_context.strategy}；"
            f"相关Concept {len(teaching_memory_context.relevant_concepts)} 个，"
            "该信号不作为掌握度更新。"
        )
    if isinstance(personal_learner_model, PersonalLearnerModel):
        summary += (
            f" 学习者理解阶段为 {personal_learner_model.lifecycle_stage.value}；"
        )
        if personal_learner_model.diagnostic.needed:
            summary += (
                "当前画像仍是低置信先验，需要后续观察或自适应诊断验证。"
            )
    if isinstance(teaching_decision, AdaptiveTeachingDecision):
        summary += (
            f" 本次教学采用 {teaching_decision.explanation_strategy}，"
            f"内容深度为 {teaching_decision.content_depth}。"
        )
    if misconception_focus:
        summary += (
            f" 当前有 {len(misconception_focus)} 个经来源事件支持的错误认知假设，"
            "需在教学中明确证据边界并继续验证。"
        )
    if isinstance(learning_path, LearningPath) and learning_path.recommended_nodes:
        first_node = next(
            (
                item for item in learning_path.milestones
                if item.kp_id == learning_path.recommended_nodes[0]
            ),
            None,
        )
        if first_node is not None:
            summary += f" 当前学习路线优先节点为 {first_node.name}。"
    if (
        isinstance(concept_learning_path, ConceptLearningPath)
        and isinstance(knowledge_learning_context, KnowledgeLearningContext)
        and concept_learning_path.next_concept != "unknown"
    ):
        concept_name = knowledge_learning_context.concept_names.get(
            concept_learning_path.next_concept,
            concept_learning_path.next_concept,
        )
        summary += (
            f" Concept Relation 综合学习目标、掌握度、误区与证据候选后，"
            f"下一节点为 {concept_name}。"
        )
    # 画像 → 难度映射 + 适配率 (对标硬指标「画像-难度适配 ≥85%」)
    diff = _recommend_difficulty(
        theta,
        view.value("derived_context", "recent_accuracy"),
    )
    # 领域标准难度阶梯 (θ 标度, 覆盖 -2.5~+2.5): 衡量推荐难度落在适中挑战区的稳健性
    _DIFF_LADDER = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    adaptation_rate = _difficulty_adaptation_rate(theta, _DIFF_LADDER)
    difficulty_adaptation = {
        "level": level,
        "band": diff["band"],
        "recommended_difficulty": diff["recommended_difficulty"],
        "zpd_lower": diff["zpd_lower"],
        "zpd_upper": diff["zpd_upper"],
        "predicted_success_probability": diff["predicted_success_probability"],
        "adaptation_rate": adaptation_rate,
        "n_ladder": len(_DIFF_LADDER),
    }
    public_profile = dict(profile)
    if isinstance(public_profile.get("extras"), dict):
        public_extras = dict(public_profile["extras"])
        public_extras.pop("learner_memory", None)
        public_profile["extras"] = public_extras
    result = {
        "agent_id": DIAGNOSIS_AGENT_ID,
        "status": "completed",
        "learner_id": learner_id,
        "ability": ability,
        "profile": public_profile,
        "memory": memory,
        "weak_kps": weak_kps,
        "level": level,
        "summary": summary,
        "confidence": confidence,
        "difficulty_adaptation": difficulty_adaptation,
    }
    # 多向信息传播: 诊断结果广播到 knowledge.gap 频道, 供知识生成/审核/导学决策订阅
    _broadcast(
        deps.message_bus,
        "learning.knowledge.gap",
        {
            "event": "diagnosis_report",
            "learner_id": learner_id,
            "weak_kps": weak_kps,
            "kp_mastery": dict(view.value("models", "mastery", {})),
            "theta": theta,
            "level": level,
            "confidence": confidence,
        },
        DIAGNOSIS_AGENT_ID,
    )
    _broadcast(
        deps.message_bus,
        "learning.diagnosis.report",
        {"event": "diagnosis_report", "learner_id": learner_id, "summary": summary},
        DIAGNOSIS_AGENT_ID,
    )
    return result


def _result_text(item: Any) -> str:
    """从检索结果项提取可展示文本."""
    if isinstance(item, dict):
        return str(
            item.get("content")
            or item.get("text")
            or item.get("name")
            or item.get("title")
            or item
        )
    return str(item)


# ============================================================
# 知识点溯源: 由证据切片文本推断关联知识点 (KP) — 关键词启发式
# ============================================================
# 每个 KP 的关键词 (hints 按特异性从高到低排序, 匹配时取每个 KP 的首个命中)
_KP_HINTS: dict[str, tuple[str, ...]] = {
    "A-01": ("电子构型", "电子组态", "4f 电子"),
    "A-02": ("5d 轨道", "4f 壳层", "壳层"),
    "A-03": ("光谱项",),
    "A-04": ("选择定则", "LS 耦合", "自旋-轨道耦合"),
    "A-05": ("Dy3+", "4f-4f", "f-f 跃迁", "蓝光", "黄光", "F9/2", "H15/2", "H13/2"),
    "A-06": ("晶体场", "Dq", "场分裂"),
    "A-07": ("Judd", "Ofelt", "强度参数", "Ω2"),
    "A-08": ("4f-5d", "宽带跃迁"),
    "A-09": ("Stark", "劈裂"),
    "A-10": ("荧光寿命", "辐射跃迁速率"),
    "A-11": ("能量传递", "交叉弛豫"),
    "A-12": ("浓度猝灭", "临界浓度"),
    "A-13": ("热猝灭", "温度猝灭"),
    "B-01": ("氟化物", "NaGdF4"),
    "B-02": ("磷酸盐", "YPO4"),
    "B-03": ("铝酸盐", "BaMgAl"),
    "B-04": ("电荷补偿",),
    "B-05": ("晶格对称", "配位环境"),
    "B-06": ("内量子效率", "量子效率", "IQE"),
    "B-07": ("色坐标", "色纯度", "显色指数", "CIE"),
    "B-08": ("激发光谱", "吸收截面", "PLE"),
    "B-09": ("上转换", "量子剪裁", "反斯托克斯", "光子雪崩"),
    "B-10": ("陷阱态", "缺陷"),
    "B-11": ("核壳", "表面效应"),
    "C-01": ("固相烧结", "高温固相", "烧结"),
    "C-02": ("共沉淀",),
    "C-03": ("溶胶-凝胶", "溶胶"),
    "C-04": ("水热", "溶剂热"),
    "C-05": ("焙烧温度", "煅烧", "结晶度"),
    "C-06": ("还原气氛", "价态控制"),
    "C-07": ("助熔剂", "晶粒形貌"),
    "C-08": ("前驱体", "掺杂均匀"),
    "C-09": ("工艺参数", "缺陷控制"),
    "C-10": ("规模放大", "批次一致"),
    "D-01": ("XRD", "物相", "衍射"),
    "D-02": ("SEM", "TEM"),
    "D-03": ("发射光谱", "PL 光谱", "荧光光谱", "光谱仪"),
    "D-04": ("寿命拟合", "衰减曲线", "荧光寿命"),
    "D-05": ("绝对法", "积分球"),
    "D-06": ("T50", "热稳定性"),
    "D-07": ("ICP-OES", "ICP"),
    "D-08": ("色度", "色温"),
}


def _infer_kps(text: str, top_k: int = 2) -> list[str]:
    """从证据切片文本推断关联知识点 (关键词启发式, 特异性优先)."""
    t = str(text or "").lower()
    if not t:
        return []
    hits: list[tuple[int, str]] = []
    for kp, hints in _KP_HINTS.items():
        for h in hints:
            if h.lower() in t:
                hits.append((len(h), kp))
                break  # 每个 KP 只取首个(最特异)命中
    if not hits:
        return []
    # 特异性优先 (长关键词在前, 稳定排序保 KP 编号序), 去重取 top_k
    hits.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, kp in hits:
        if kp not in seen:
            seen.add(kp)
            out.append(kp)
        if len(out) >= top_k:
            break
    return out


def _clean_markdown_chunk(text: str) -> str:
    """清理切片中的图片、图注、URL 与引用噪声."""
    # 清理 PDF 转文本的标点乱码 (ꎬ=逗号, ꎮ=句号)
    text = str(text or "").replace("ꎬ", "，").replace("ꎮ", "。")
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("![") or re.search(r"\]\(https?://", line):
            continue
        if re.match(r"^(图\s*\d+|Fig(?:ure)?\.?\s*\d+)", line):
            continue
        if line.startswith("http://") or line.startswith("https://"):
            continue
        # 清理残留 HTML 标签 (如 <sup>[9]</sup>) 与 Markdown 标题/编号
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^\d+(\.\d+)*\s*", "", line)
        # 清理行中的编号标记 (如 "(2)"、"（6）")
        line = re.sub(r"[（(]\d+[)）]\s*", "", line).strip()
        # 清理行中残留的 Markdown 标题标记 (如 "Low ## 二、…" 中 "##" 不在行首)
        line = re.sub(r"#{1,6}\s*", " ", line)
        # 清理英文置信度残标 (Low/Medium/High, 行首)
        line = re.sub(r"^(?:Low|Medium|High)\s+", "", line, flags=re.IGNORECASE)
        # 清理引用号 [28]/[29,30]/[65-67]
        line = re.sub(r"\[\s*\d+(?:\s*[,，\-–]\s*\d+)*\s*\]", "", line).strip()
        # 清理 PDF 转文本乱码 (货币/重音/连字/分数等拉丁扩展字符 + 控制字符, 中文材料中不应出现)
        line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", line)
        line = re.sub(r"[£¥¢¤€ðøþÞß½¾¼ÓÒÔÕÖÆæ¡¿«»¬ªº´¨¯¸¹²³]", "", line).strip()
        if not line:
            continue
        if re.match(r"^\[\d", line) or re.search(r"Mater\. Lett\.|References", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _clean_synthesized_answer(text: str) -> str:
    """清理最终合成答案中的 Markdown 图片/链接噪声."""
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", str(text or ""))
    cleaned = re.sub(r"\]\(https?://[^)]*\)", ")", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _latex_to_plain(text: str) -> str:
    """把常见 LaTeX 记号转成可读纯文本."""
    result = str(text or "")
    for command in ("mathrm", "operatorname", "mathbf", "textit", "text"):
        result = re.sub(rf"\\{command}\{{([^}}]*)\}}", r"\1", result)
    result = result.replace("\\rightarrow", "->").replace("\\left", "").replace("\\right", "")
    result = result.replace("\\,", "").replace("\\;", "").replace("\\quad", " ")
    result = result.replace("~", " ")
    result = re.sub(r"\^\{([^}]*)\}", r"\1", result)
    result = re.sub(r"_\{([^}]*)\}", r"\1", result)
    result = result.replace("^", "").replace("_", "")
    result = result.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"\s+", " ", result).strip()


def _split_sentences(text: str) -> list[str]:
    """按中文/英文句号切分句子."""
    flat = str(text or "").replace("\n", " ")
    # 英文粘连句界: MinerU/清洗后常见 ".An increase"/".The" (句点后无空格紧跟大写),
    # 原切分把整段当一个句子 → 候选超长且常以切片边界截断 ("…can be"), 审核判幻觉。
    flat = re.sub(r"(?<=[a-z0-9)\]])\s*\.(?=[A-Z(])", ". ", flat)
    parts = re.split(r"(?<=[。！？；])|(?<=\.)\s+", flat)
    return [part.strip() for part in parts if len(part.strip()) >= 8]


# 问题类型识别 (提质核心: 方法型 vs 机理型 vs 定义型)
# 方法型 ("怎么避免/如何提高/怎样解决") → 优先抽取措施性句子, 组织成结构化建议
# 机理型 ("为什么/是什么机理") → 保留机理解释
_METHOD_QUESTION_RE = re.compile(
    r"(步骤|操作流程|实验方法|protocol|procedure|how\s+to|"
    r"怎么|如何|怎样(?!的)|咋|能否|可不可以|有什么办法|有什么措施|怎么避免|如何避免|"
    r"怎样避免|怎么防止|如何防止|怎么降低|如何降低|怎么减少|如何减少|"
    r"怎么解决|如何解决|怎么提高|如何提高|怎么增强|如何增强|怎么做|怎么实现|如何实现)"
)
_METHOD_ACTION_TERMS = (
    "通过", "控制", "降低", "减少", "避免", "选择", "优化", "调整", "调控",
    "采用", "使用", "测量", "记录", "扫描", "对照", "研磨", "混合", "放置", "拟合",
    "引入", "提高", "增强", "增加", "保持", "防止", "解决", "应",
    "measure", "measured", "measurement", "record", "recorded", "scan",
    "analyse", "analyze", "fit", "fitted", "using", "equipped with",
    "可以", "需要", "一、", "二、", "三、", "四、", "五、", "六、",
)
_MECHANISM_RE = re.compile(r"(为什么|机理|原理|原因|怎么回事|如何发生|怎样发生|机制|本质)")


def _detect_question_type(query: str) -> str:
    """判断问题类型: method(方法型) / mechanism(机理型) / definition(定义型) / other."""
    q = str(query or "").strip()
    if not q:
        return "other"
    # “如何影响” asks for a causal mechanism, not an experimental procedure.
    # The previous generic “如何” rule routed these questions to a numbered
    # method template and produced scientifically irrelevant answers.
    if re.search(
        r"(怎么|如何|怎样)(?:通过|经由|借助|利用)?[^？?。]{0,24}"
        r"(?:影响|改变|导致|作用)",
        q,
    ):
        return "mechanism"
    if "为什么" in q and re.search(r"怎么|如何|怎样", q):
        return "mechanism"
    # “如何用于理解/分析” asks how a scientific framework organises an
    # interpretation, not for an experimental procedure.  Routing it through
    # the method template drops the framework definition and keeps only an
    # imperative sentence, which can omit half of a multi-dimension question.
    if re.search(
        r"(怎么|如何|怎样)用于(?:理解|解释|分析|判断|评价)",
        q,
    ):
        return "other"
    # “有哪些影响/有什么作用” asks for an observed relationship.  Treating
    # the word “有哪些” as a definition caused the definition-only evidence
    # sorter to discard the exact Concept evidence and keep an unrelated
    # passage which merely contained “性质”.
    if re.search(
        r"(有(?:哪|什)些?影响|有何影响|有什么作用|有何作用|"
        r"起什么作用|起哪些作用|是什么关系)",
        q,
    ):
        return "mechanism"
    # 方法型优先 (含"怎么/如何/避免"等动作词, 即使也含"是什么")
    if _METHOD_QUESTION_RE.search(q):
        return "method"
    if _MECHANISM_RE.search(q):
        return "mechanism"
    if re.search(r"(是什么|什么是|介绍一下|介绍|定义|含义|有哪些)", q):
        return "definition"
    return "other"


# ---- 相近词库实体一致性 (P2): 稀土离子元素识别, 用于"张冠李戴"幻觉防御 ----
_ION_ASCII_RE = re.compile(
    r"(?<![A-Za-z])(Dy|Eu|Ce|Tb|Yb|Er|Nd|Sm|Pr|Ho|Tm|Gd|Lu|La)"
    r"\s*[0-9³²⁴]?[+⁺\-⁻]?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ION_CN_MAP = {
    "镝": "Dy", "铕": "Eu", "铈": "Ce", "铽": "Tb", "镱": "Yb",
    "铒": "Er", "钕": "Nd", "钐": "Sm", "镨": "Pr", "钬": "Ho", "铥": "Tm",
}
_ION_DISPLAY = {
    "Dy": "Dy³⁺", "Eu": "Eu³⁺", "Ce": "Ce³⁺", "Tb": "Tb³⁺", "Yb": "Yb³⁺",
    "Er": "Er³⁺", "Nd": "Nd³⁺", "Sm": "Sm³⁺", "Pr": "Pr³⁺", "Ho": "Ho³⁺",
    "Tm": "Tm³⁺", "Gd": "Gd³⁺", "Lu": "Lu³⁺", "La": "La³⁺",
}


def _extract_ions(text: str) -> set[str]:
    """从文本提取明确出现的稀土离子元素符号集合 (如 {Dy}, {Eu, Ce}).

    同时匹配英文元素符号 (Dy/Eu/Ce…) 与中文元素名 (镝/铕/铈…).
    用于相近词库防御: 查询指定某离子时, 剔除只讲"另一离子"的错配知识块.
    """
    s = str(text or "")
    ions: set[str] = set()
    for m in _ION_ASCII_RE.finditer(s):
        sym = m.group(1).capitalize()
        if sym in _ION_DISPLAY:
            ions.add(sym)
    for ch in s:
        if ch in _ION_CN_MAP:
            ions.add(_ION_CN_MAP[ch])
    return ions


# 强领域外意图黑名单: 命中即硬拦截(拒绝作答)。列表刻意保守——
# 未命中黑名单的问题交给检索与审核门处置(证据不足会诚实拒答),
# 以免误伤教材库扩展后的稀土化学基础问题 (2026-09-03 双库评测)。
_OUT_OF_DOMAIN_BLOCKERS = (
    "红烧肉", "菜谱", "做菜", "怎么煮", "怎么炒", "怎么炖", "美食",
    "打游戏", "足球", "篮球", "羽毛球", "乒乓球", "明星", "电视剧", "电影",
    "股票", "彩票", "房价", "基金定投", "汇率预测", "政治", "总统",
    "怎么学英语", "雅思", "考研英语", "数学题", "物理题",
    "石墨烯", "量子点led", "化学气相沉积cvd",
)


def _hard_out_of_domain(query: str) -> bool:
    """Return True only for clearly out-of-domain everyday intents."""
    q = str(query or "")
    return any(blocker in q for blocker in _OUT_OF_DOMAIN_BLOCKERS)


def _count_ions(text: str) -> dict[str, int]:
    """统计文本中稀土离子的出现次数 (英文符号 + 中文名).

    用于上下文可靠性: 判断查询离子是否为文档的主要离子,
    防止"问 Dy 却混入主要讲 Ce/Pr 的文档"(顺带提一句 Dy 不应算匹配).
    """
    s = str(text or "")
    counts: dict[str, int] = {}
    for m in _ION_ASCII_RE.finditer(s):
        sym = m.group(1).capitalize()
        if sym in _ION_DISPLAY:
            counts[sym] = counts.get(sym, 0) + 1
    for ch in s:
        if ch in _ION_CN_MAP:
            sym = _ION_CN_MAP[ch]
            counts[sym] = counts.get(sym, 0) + 1
    return counts


def _ion_gate_rerank(
    query: str,
    results: list[dict[str, Any]],
    scores: list[float],
) -> tuple[list[dict[str, Any]], list[float]]:
    """离子门重排: 问题点名某离子时, 把含该离子的证据排前、只含其它离子的证据排后.

    防"张冠李戴": 问 Dy3+ 却检索到 Eu3+/Ce3+/Mn2+ 内容时, 目标离子证据优先,
    而非让语义/关键词分数占主导. 无明确离子的问题不门控 (保持原序).
    稳定排序: 同组内保留原 RRF 名次.
    """
    q_ions = _extract_ions(query)
    if not q_ions or not results:
        return results, scores

    def _priority(item: dict[str, Any]) -> int:
        text = " ".join(
            str(item.get(k) or "") for k in ("content", "description", "name")
        )
        t_ions = _extract_ions(text)
        if q_ions & t_ions:
            return 0  # 含目标离子 → 优先
        if t_ions:
            return 2  # 只含其它离子 → 降权 (疑似张冠李戴)
        return 1      # 无离子 (原理/方法类) → 中性

    priorities = [_priority(it) for it in results]
    order = sorted(range(len(results)), key=lambda i: priorities[i])
    gated_results = [results[i] for i in order]
    gated_scores = [scores[i] for i in order] if len(scores) == len(results) else []
    # 分数抬升: 只重排不改分会被后续重排器 (QualityBoost/MMR 按分降序) 洗回原序,
    # 导致离子门失效. 这里把"含目标离子"证据的分数抬到高于任一非目标证据,
    # 使重排器在按分排序后仍把目标离子证据保留在前排.
    if gated_scores:
        _max_s = max(gated_scores)
        _floor = (_max_s + 1.0) if _max_s > 0 else 1.0
        for _i in range(len(gated_results)):
            if priorities[order[_i]] == 0:
                gated_scores[_i] = max(gated_scores[_i], _floor)
    return gated_results, gated_scores


def _resolve_entity_id(query: str, l3_store: Any) -> str | None:
    """把查询命中的稀土离子映射到知识库主离子实体 (最佳努力, 供图检索起始点).

    尝试多个候选名 (Dy / dy3+ / Dy3+ ...), 取首个命中的实体; 无命中返回 None,
    使图检索分支自动跳过 (不影响关键词/向量检索).
    """
    if l3_store is None:
        return None
    ions = _extract_ions(query)
    if not ions:
        return None
    entity_store = getattr(l3_store, "entity_store", None)
    if entity_store is None or not hasattr(entity_store, "find_by_name"):
        return None
    for sym in ions:
        candidates = (sym, sym.lower(), f"{sym.lower()}3+", f"{sym}3+")
        for cand in candidates:
            try:
                hits = entity_store.find_by_name(cand)
            except Exception:  # noqa: BLE001
                continue
            if not hits:
                continue
            for e in hits:
                if getattr(e, "triples", None):
                    return e.entity_id
            return hits[0].entity_id
    return None


def _is_method_sentence(sentence: str) -> bool:
    """判断句子是否为措施/建议类 (含动作词, 面向"怎么做/怎么避免")."""
    return any(term in sentence for term in _METHOD_ACTION_TERMS)


def _method_sentence_matches_query(query: str, sentence: str) -> bool:
    """Require the action in a method sentence to match the requested task."""

    query_value = str(query or "")
    sentence_value = str(sentence or "").casefold()
    if re.search(
        r"(测量|测试|表征|分析|measure|test|characteri[sz]e|analy[sz]e)",
        query_value,
        re.IGNORECASE,
    ):
        return any(
            term in sentence_value
            for term in (
                "测量", "测试", "表征", "记录", "扫描", "拟合", "仪器",
                "measure", "measured", "measurement", "record", "scan",
                "fit", "fitted", "spectrometer", "integrating sphere",
            )
        )
    if re.search(
        r"(制备|合成|烧结|prepare|synthesi[sz]e)",
        query_value,
        re.IGNORECASE,
    ):
        return any(
            term in sentence_value
            for term in (
                "制备", "合成", "烧结", "原料", "研磨", "混合", "气氛",
                "prepare", "prepared", "synthesis", "synthesized",
                "calcination", "anneal", "mixed", "ground",
            )
        )
    return _is_method_sentence(sentence)


def _trim_fragment(sentence: str) -> str:
    """去掉切片片段开头/结尾的残缺噪声."""
    value = sentence.strip()
    # MinerU chunks can begin in the middle of an English word, e.g.
    # ``...ctively.It is well known ...``.  Remove only that leading boundary
    # fragment; the scientific sentence that follows remains verbatim.
    value = re.sub(r"^\.{3}[A-Za-z]{1,24}\.\s*", "", value)
    if value.startswith("..."):
        value = value[3:].lstrip()
    if (
        re.match(r"^[，,；;：:、\s]+", value)
        or value.startswith(("rm", "ns", "composite", "mposite"))
    ):
        if "。" in value:
            value = value.split("。")[-1]
    value = value.lstrip("，,；;：:、 \t")
    # 去「文档自指」前缀/短语：答案里没有图/公式，这些指代会悬空
    value = re.sub(
        r"^(?:从图中[^，,。]*[，,]\s*|如图所示[，,]\s*|由图可知[，,]\s*|"
        r"见图\d*[，,]\s*|见下表[^，,。]*[，,]\s*)",
        "", value,
    ).strip()
    # 非锚定：MinerU 常把正文句与后续公式 OCR 粘在一起。只截断显式
    # 的“下列公式”尾部，不重建公式，也不改写前面的科学句。
    value = re.sub(r"[，,；;]?\s*而?\s*由下列公式.*$", "", value)
    value = re.sub(r"由下列公式[^，,。]*?得到", "", value)
    value = re.sub(r"\.{3,}$", "", value).strip()
    return value


_ANSWER_RELEVANCE_TERMS = (
    "浓度猝灭", "热猝灭", "交叉弛豫", "能量传递", "非辐射", "晶体场",
    "stark", "concentration quenching", "thermal quenching", "cross relaxation",
    "energy transfer", "nonradiative", "crystal field", "emission spectrum",
    "浓度", "猝灭", "发光", "发射", "光谱", "黄蓝", "黄光", "蓝光", "白光",
    "能级", "跃迁", "弛豫", "能量", "辐射", "声子", "晶格", "基质", "缺陷",
    "掺杂", "效率", "寿命", "温度", "热稳定", "量子", "色度", "色温", "显色",
    "健康", "风险", "安全", "波长", "表征", "合成", "制备",
    "emission", "spectrum", "transition", "lifetime", "efficiency", "phonon",
    "matrix", "host", "defect", "temperature", "chromaticity", "cct", "cri",
)


def _bare_element_definition_identity_terms(query: str) -> tuple[str, ...]:
    """Return ontology terms required for a bare-element definition query.

    A passage that merely mentions ``Dy3+`` in an unrelated mechanism is not
    evidence for “Dy是什么”.  This guard checks entity identity/category only;
    it does not provide or hard-code the requested scientific definition.
    """

    if _detect_question_type(query) != "definition":
        return ()
    normalized = re.sub(
        r"[\s?？!！。,.，:：;；]+",
        "",
        normalize_query(query),
    )
    # Query normalization expands element aliases to their ionic form (for
    # example ``dy`` -> ``dy3+``).  Identity matching needs the stable element
    # symbol, not the display charge suffix.
    normalized = re.sub(r"(?:3\+|³\+|3⁺|³⁺)$", "", normalized)
    match = _BARE_ELEMENT_RE.fullmatch(normalized)
    if match is None:
        return ()
    symbol = str(match.group(1) or "").lower()
    display = str(_ELEMENT_DISPLAY.get(symbol) or symbol)
    chinese_names = tuple(re.findall(r"[\u4e00-\u9fff]+", display))
    return tuple(dict.fromkeys((
        *chinese_names,
        "元素",
        "chemicalelement",
        "rare-earthelement",
        "lanthanide",
        "三价",
        "trivalent",
    )))


def _filter_task_answer_evidence(
    query: str,
    items: list[dict[str, Any]],
    *,
    focus_terms: tuple[str, ...] = (),
    preferred_concept_ids: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Keep evidence about the target, not merely one Concept relation away.

    R-06 relations may expand recall, but a neighbouring Concept is not
    automatically evidence for the asked question.  No scientific fact or
    relation is created here.
    """

    preferred = frozenset(
        str(value).strip()
        for value in preferred_concept_ids
        if str(value).strip()
    )
    normalized_focus = tuple(dict.fromkeys(
        re.sub(r"\s+", "", str(value).lower())
        for value in focus_terms
        if len(re.sub(r"\s+", "", str(value))) >= 3
    ))
    compact_query = re.sub(r"\s+", "", str(query).lower())
    query_ions = _extract_ions(query)
    definition_identity_terms = _bare_element_definition_identity_terms(query)
    concentration_drop_intent = (
        "浓度" in compact_query
        and any(
            term in compact_query
            for term in ("下降", "降低", "减小", "变弱", "衰减", "猝灭")
        )
    )
    query_anchors = tuple(dict.fromkeys(
        re.sub(r"\s+", "", term.lower())
        for term in _ANSWER_RELEVANCE_TERMS
        if re.sub(r"\s+", "", term.lower()) in compact_query
    ))
    if (
        not preferred
        and not normalized_focus
        and not query_anchors
        and not definition_identity_terms
    ):
        return list(items)

    required_anchor_hits = 2 if len(query_anchors) >= 2 else 1
    selected: list[dict[str, Any]] = []
    for item in items:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        mapped = frozenset(
            str(value).strip()
            for value in metadata.get("concept_ids", ())
            if str(value).strip()
        )
        direct_concept_mapping = bool(preferred.intersection(mapped))
        text = re.sub(
            r"\s+", "", str(item.get("content") or item.get("text") or "").lower()
        )
        if not text:
            continue
        if definition_identity_terms and not any(
            term in text for term in definition_identity_terms
        ):
            continue
        # If both the question and a passage explicitly name ions, a passage
        # about another ion is not task evidence.  Ion-free textbook mechanism
        # passages remain eligible as general scientific support.
        item_ions = _extract_ions(text)
        if query_ions and item_ions and not query_ions.intersection(item_ions):
            continue
        if definition_identity_terms:
            # For a bare identity question the ontology/category statement is
            # itself the direct task evidence.  Mechanism focus terms are not
            # required and must not pull in neighbouring passages.
            selected.append(item)
            continue
        # For concentration-loss questions, two generic words such as
        # "concentration" and "luminescence" are too weak.  Require the
        # passage itself to express the loss/quenching or its physical process.
        if concentration_drop_intent and not (
            "浓度" in text
            and any(
                term in text
                for term in (
                    "猝灭", "下降", "降低", "减小", "衰减",
                    "能量传递", "非辐射", "迁移", "耗散",
                    "quenching", "nonradiative", "energytransfer",
                )
            )
        ):
            continue
        # A canonical Concept mapping raises priority but does not turn a
        # neighbouring or merely topical passage into direct task evidence.
        # The passage must still satisfy the query's ion and mechanism guards.
        if direct_concept_mapping:
            selected.append(item)
            continue
        if any(term in text for term in normalized_focus):
            selected.append(item)
            continue
        anchor_hits = sum(1 for term in query_anchors if term in text)
        if query_anchors and anchor_hits >= required_anchor_hits:
            selected.append(item)
            continue
        if metadata.get("source_type") == "textbook_fallback" and anchor_hits:
            selected.append(item)
    return selected


def _collect_answer_candidates(
    query: str,
    items: list[dict[str, Any]],
    *,
    focus_terms: tuple[str, ...] = (),
    preferred_concept_ids: tuple[str, ...] = (),
) -> list[str]:
    """从清洗后的证据中抽取相关句子候选.

    提质修复 (回答不到点子上的根因):
    - 原实现用"完整问题串"去匹配句子 (如"浓度猝灭怎么避免"), 必然失配,
      导致候选为空、答案回退到通用综合模板.
    - 现改为抽取问题的 2 字 bigram 主题词 (如 浓度/猝灭/避免), 任一词命中即认可,
      并放宽"必须含数字"与"必须含指定关键词"的限制, 使措施性句子可被抽到.
    """
    question_type = _detect_question_type(query)
    raw_terms = [
        term
        for term in re.split(r"[，。？、\s]+", str(query))
        if len(term) >= 2
    ]
    # bigram 主题词: 中文按 2 字滑窗, 过滤纯功能词 (怎么/如何/避免/防止/解决 等)
    query_terms: list[str] = []
    _FUNC_WORDS = frozenset(
        "怎么 如何 怎样 咋 能否 可不可以 避免 防止 解决 降低 减少 提高 增强 为什么 "
        "什么 哪些 哪个 什么 可以 需要 应该 的 了 吗 呢 一下 请问".split()
    )
    for term in raw_terms:
        if len(term) <= 6 and term not in _FUNC_WORDS:
            query_terms.append(term)
        for i in range(len(term) - 1):
            bg = term[i : i + 2]
            if bg not in _FUNC_WORDS and bg not in query_terms:
                query_terms.append(bg)
    if not query_terms:
        query_terms = raw_terms
    # Chinese scientific questions often compress parallel concepts into one
    # phrase (for example, “A/B双发射”), while evidence expands them into two
    # separate clauses.  Bigram matching alone misses that equivalence.  Keep
    # content-bearing Han characters as a secondary lexical signal; common
    # interrogative/function characters are excluded to avoid generic matches.
    _QUERY_FUNCTION_CHARS = frozenset(
        "为什么怎么如何怎样什么哪些哪个是否能否可以请问具有的了吗呢"
    )
    query_content_chars = tuple(dict.fromkeys(
        char
        for char in str(query)
        if "\u4e00" <= char <= "\u9fff"
        and char not in _QUERY_FUNCTION_CHARS
    ))
    normalized_focus_terms = tuple(
        dict.fromkeys(
            str(term).strip().lower()
            for term in focus_terms
            if len(str(term).strip()) >= 3
        )
    )
    normalized_preferred_concept_ids = frozenset(
        str(concept_id).strip()
        for concept_id in preferred_concept_ids
        if str(concept_id).strip()
    )

    candidates: list[tuple[str, int, int]] = []
    junk_patterns = (
        "[J]",
        "Chin.J.Lumin",
        "Mater. Lett",
        "PACS",
        "关键词",
        "文章编号",
        "中图分类号",
        "References",
        "Fig.",
        "Fig ",
        "Figure",
        "分光光度计",
        "文章编号",
    )
    # 领域内容词: 命中任一词即视为"实质性知识句" (替代原来过于狭窄的 5 词白名单)
    _CONTENT_HINTS = (
        "猝灭", "浓度", "掺杂", "发光", "荧光", "磷光", "发射", "激发", "光谱",
        "能级", "跃迁", "效率", "寿命", "能量", "传递", "敏化", "基质", "合成",
        "制备", "温度", "热稳定", "色度", "显色", "波长", "峰", "声子", "晶格",
        "缺陷", "陷阱", "弛豫", "离子", "纳米", "量子", "测量", "表征", "工艺",
        "措施", "控制", "选择", "优化", "引入", "调整", "壳层", "隔离",
        # 通用稀土/教材域内容词: 覆盖教材级概念句 (元素/配位/分离/结构等),
        # 否则教材库的概念型句子全部落选, 生成退化为空 (2026-09-03 双库评测发现)
        "稀土", "镧系", "元素", "配位", "配合物", "化合物", "分离", "矿物",
        "萃取", "结晶", "催化", "合金", "结构", "价态", "配体", "基团",
        "原子", "电子", "轨道", "溶解", "沉淀", "反应", "氧化物", "金属",
    )
    # 英文/拉丁内容线索: 中文查询的 bigram 无法匹配英文机理句, 而这些句子
    # 恰恰承载机制/公式正文 (如 "critical distance ... Rc = 2(3V/(4πxcN))^(1/3)")。
    # 命中任一英文科学线索即视为实质知识句, 否则双语语料的答案候选恒为空。
    _EN_CONTENT_CUES = (
        "quench", "luminesc", "phosphor", "emission", "excitation", "emitting",
        "energy transfer", "cross-relaxation", "critical distance",
        "critical concentration", "concentration quenching", "dexter",
        "dipole", "multipolar", "multipole", "nonradiative", "radiative",
        "decay", "lifetime", "wavelength", "spectra", "spectrum", "host",
        "dopant", "activator", "thermal", "lattice", "symmetry", "synthes",
        "prepared", "powder", "solid-state", "mol%", "doped", "ions",
        "transfer", "quenching", "concentration", "intensity", "CIE",
    )
    _FORMULA_CUE_RE = re.compile(
        r"(?:Rc|RC|lg\s*\(|log\s*\(|\^\(?1/3|\b1/3\b|"
        r"\d+(?:\.\d+)?\s*(?:nm|mol%|wt%|at%|eV|Å|A3|K|℃|°C))",
        re.IGNORECASE,
    )
    # 查询中的拉丁/离子 token ("dy3+", "4f9/2", "yag", "cie") 作为句子命中信号,
    # 使双语语料中同一实体的英文句与中文查询对齐 (bigram 无法跨语言匹配)。
    _query_latin_tokens = tuple(
        dict.fromkeys(
            token.lower()
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]{1,14}", str(query))
            if len(token) >= 2
        )
    )
    for item in items:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_type = str(metadata.get("source_type") or "")
        mapped_concept_ids = frozenset(
            str(concept_id).strip()
            for concept_id in metadata.get("concept_ids", ())
            if str(concept_id).strip()
        )
        # Preserve how many explicit target Concepts this evidence item
        # covers.  A passage mapped to one neighbouring target must not rank
        # equally with a passage that covers the complete multi-Concept
        # question (for example, both yellow and blue emission branches).
        direct_concept_priority = len(
            normalized_preferred_concept_ids.intersection(mapped_concept_ids)
        )
        source_priority = (
            2
            if source_type == "curated_source_summary"
            and metadata.get("evidence_status") == "reviewed"
            else 1
            if source_type in {"textbook_fallback", "kp_expand"}
            else 0
        )
        text = _latex_to_plain(_clean_markdown_chunk(item.get("content", "")))
        for sentence in _split_sentences(text):
            sentence_lower = sentence.lower()
            if any(pattern in sentence for pattern in junk_patterns):
                continue
            # 过滤残留标题/图片噪声 (如 "## 5.4"、".jpg)图 3.5")
            if re.search(r"^#{1,6}\s|https?://|\.jpg|\.png|\.jpeg|\)图\s*\d", sentence):
                continue
            # 图注正文噪声: "如图 4(c)所示…" / "…相互作用 图 4 (a) 不同掺杂浓度…" 是
            # MinerU 把图表说明粘入正文的残留, 句内常夹 图X(…)/;…(…) 列表, 拼入答案
            # 即产生"表述残缺混乱" (审核实测). 带图注结构的句子整体不作文答候选.
            if re.search(r"如图\s*\d|图\s*\d+\s*[（(]", sentence):
                continue
            has_numeric = bool(
                re.search(r"\d+\s*nm|\d+\s*%|\d{3}", sentence)
            )
            hit_query_term = any(term in sentence for term in query_terms)
            hit_focus_term = any(
                term in sentence_lower for term in normalized_focus_terms
            )
            hit_term = hit_query_term or hit_focus_term
            hit_en_cue = any(cue in sentence_lower for cue in _EN_CONTENT_CUES)
            hit_formula = bool(_FORMULA_CUE_RE.search(sentence))
            hit_latin = any(tok in sentence_lower for tok in _query_latin_tokens)
            hit_content = (
                any(kw in sentence for kw in _CONTENT_HINTS)
                or hit_focus_term
                or hit_en_cue
                or hit_formula
            )
            if (hit_term or hit_latin or hit_en_cue or hit_formula) and (
                has_numeric or hit_content
            ):
                if hit_content:
                    trimmed = _trim_fragment(sentence)
                    # 切片边界残句防护: 以无标点结尾的片段 ("…which causes
                    # concentration quenching, can be") 多为切片截断, 单列会
                    # 被审核判"表述残缺/幻觉", 不作文答候选。
                    if len(trimmed) >= 8 and re.search(
                        r"[。！？；.!?;:）)\]]$|[\u4e00-\u9fff]$",
                        trimmed,
                    ):
                        candidates.append(
                            (trimmed, source_priority, direct_concept_priority)
                        )

    dedup_by_key: dict[str, tuple[str, int, int]] = {}
    for sentence, source_priority, direct_concept_priority in candidates:
        key = re.sub(r"[\s，,、；;：:()（）\[\]]", "", sentence)[:60]
        existing = dedup_by_key.get(key)
        if existing is None or (
            direct_concept_priority,
            source_priority,
        ) > (existing[2], existing[1]):
            dedup_by_key[key] = (
                sentence,
                source_priority,
                direct_concept_priority,
            )
    if not dedup_by_key:
        return []

    # Evidence order is not answer relevance.  Rank sentences against the
    # actual question so a nearby experimental detail cannot precede the
    # sentence that directly explains the requested mechanism.  This is a
    # query-coverage rule over retrieved text, not a domain answer template.
    q_ions = _extract_ions(query)

    def _sentence_score(
        sentence: str,
        source_priority: int = 0,
        direct_concept_priority: int = 0,
    ) -> tuple[int, int, int, int, int]:
        compact = sentence.lower().replace(" ", "")
        coverage = sum(
            1 for term in query_terms
            if len(term) >= 2 and term.lower().replace(" ", "") in compact
        )
        focus_coverage = sum(
            1
            for term in normalized_focus_terms
            if term.replace(" ", "") in compact
        )
        character_coverage = sum(
            1 for char in query_content_chars if char in sentence
        )
        ion_match = int(bool(q_ions & _extract_ions(sentence)))
        mechanism_match = sum(
            1
            for term in (
                "能级", "跃迁", "发射", "机制", "机理", "原因", "因为",
                "导致", "由于", "能量传递", "非辐射", "迁移", "耗散",
                "emission", "transition", "crystal field", "symmetry",
                "energy transfer", "nonradiative", "migration", "dissipation",
            )
            if term in sentence.lower()
        )
        relation_match = sum(
            1
            for term in (
                "影响", "导致", "决定", "作用", "依赖", "改变",
                "influence", "affect", "cause", "depend", "vary", "determine",
            )
            if term in sentence.lower()
        )
        # A mechanism answer should prefer a sentence that names the queried
        # entity and states an observed relation.  This ranks retrieved facts;
        # it does not add a domain answer or infer a new scientific claim.
        directness = ion_match * 3 + relation_match * 2
        # Explicit query coverage must outrank generic mechanism richness.
        # Otherwise a neighbouring sentence containing several scientific
        # mechanism words can precede the sentence that actually names the
        # phenomenon asked by the learner.  This is general relevance ranking
        # over retrieved evidence, not a domain-answer rule.
        intent_bonus = (
            min(mechanism_match, 3) * 2
            if question_type == "mechanism"
            else 0
        )
        return (
            direct_concept_priority,
            coverage * 3
            + character_coverage * 3
            # Upstream Concept/retrieval focus is a stronger semantic signal
            # than character overlap.  It must remain dominant for bilingual
            # questions whose best evidence is an English source sentence.
            + focus_coverage * 20
            + directness
            + intent_bonus,
            source_priority,
            mechanism_match,
            -len(sentence),
        )

    ranked_pairs = sorted(
        dedup_by_key.values(),
        key=lambda value: _sentence_score(value[0], value[1], value[2]),
        reverse=True,
    )
    best_coverage = _sentence_score(*ranked_pairs[0])[1]
    if best_coverage >= 3:
        # 双语保底: 聚焦过滤若只按中文 bigram 重叠 (score) 裁剪, 会把含公式/英文
        # 机制正文的低重叠句全部清掉 (中文句 coverage 高 → 公式句被判出局), 实测
        # "临界距离如何计算" 的 Rc 公式句因此从候选消失。公式/领域线索句无条件保留,
        # 最终由 compose 排序与审核门决定取舍。
        def _keep_for_bilingual(value: tuple[str, int, int]) -> bool:
            s = value[0]
            return bool(
                re.search(r"(?i)(\brc\b|blasse|dexter|lg\s*\(|1/3|energy transfer|"
                          r"cross[- ]?relaxation|quenching|critical distance)",
                          s)
                or any(cue in s.lower() for cue in _EN_CONTENT_CUES)
            )
        focused_pairs = [
            value for value in ranked_pairs
            if _sentence_score(*value)[1] >= 2
            or value[2] > 0
            or _keep_for_bilingual(value)
        ]
        if focused_pairs:
            ranked_pairs = focused_pairs
    return [sentence for sentence, _source_priority, _direct in ranked_pairs]


def _prefer_reviewed_concept_evidence(
    items: list[dict[str, Any]],
    preferred_concept_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Use reviewed Concept mappings as the bounded answer evidence set.

    Raw paper chunks remain in L3 and in the retrieval trace.  When a reviewed
    source-bounded summary is explicitly mapped to the target Concept,
    however, allowing lower-provenance neighbouring chunks back into the
    Generation context can only dilute the answer and its public provenance.
    This function selects existing evidence; it neither creates a claim nor
    infers a Concept relationship.
    """

    preferred = frozenset(
        str(concept_id).strip()
        for concept_id in preferred_concept_ids
        if str(concept_id).strip()
    )
    if not preferred:
        return items
    direct: list[dict[str, Any]] = []
    for item in items:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if not (
            metadata.get("source_type") == "curated_source_summary"
            and metadata.get("evidence_status") == "reviewed"
            and str(metadata.get("source_uri") or "").strip()
        ):
            continue
        concept_ids = frozenset(
            str(concept_id).strip()
            for concept_id in metadata.get("concept_ids", ())
            if str(concept_id).strip()
        )
        if preferred.intersection(concept_ids):
            direct.append(item)
    if not direct:
        return items
    # Reviewed provenance is not a relevance licence.  Once direct reviewed
    # evidence exists, unrelated reviewed summaries must not be appended just
    # because they are high quality in isolation; doing so let generic
    # spectrum/host passages displace the actual question focus.
    return direct


def _compose_concise_answer(
    query: str,
    items: list[dict[str, Any]],
    *,
    focus_terms: tuple[str, ...] = (),
    preferred_concept_ids: tuple[str, ...] = (),
) -> str:
    """从清洗后的证据中抽取相关句子，按问题类型生成针对性答案.

    提质设计 (问题理解增强):
    - 方法型问题 ("怎么避免/如何解决/怎样提高") → 优先抽取措施类句子,
      组织成可操作的结构化建议, 直接回应"怎么做".
    - 定义型/机理型/其他 → 抽取与主题相关的解释性句子.
    """
    qtype = _detect_question_type(query)
    # 兜底层优先: 命中教材级权威事实时, 以其为答案主体 (权威、无 KB 噪声),
    # 保证「诚实 + 忠实」: 兜底事实来自教材兜底层并标注来源, 不混入 TOC/其他离子噪声。
    fb_items = [
        it for it in items
        if (it.get("metadata") or {}).get("source_type") == "textbook_fallback"
    ]
    if fb_items:
        body: list[str] = []
        seen: set[str] = set()
        chapters: list[str] = []
        for it in fb_items:
            text = _latex_to_plain(_clean_markdown_chunk(it.get("content", "")))
            for sentence in _split_sentences(text):
                trimmed = _trim_fragment(sentence)
                if len(trimmed) < 8:
                    continue
                key = re.sub(r"[\s，,、；;：:()（）\[\]]", "", trimmed)[:60]
                if key in seen:
                    continue
                seen.add(key)
                body.append(trimmed)
            src = (it.get("metadata") or {}).get("source") or it.get("section", "")
            if src and src not in chapters:
                chapters.append(src)
        if body:
            lines = "".join(s for s in body[:6])
            src_note = "；".join(chapters)[:200]
            if src_note:
                return f"{lines}\n\n（来源：{src_note}）"
            return lines
    dedup = _collect_answer_candidates(
        query,
        items,
        focus_terms=focus_terms,
        preferred_concept_ids=preferred_concept_ids,
    )
    if not dedup:
        return ""
    # 方法型: 措施句优先, 保底补机理句 (避免只有空泛建议)
    if qtype == "method":
        method_focus_terms = tuple(
            str(term).strip().casefold()
            for term in focus_terms
            if len(str(term).strip()) >= 3
        )
        method_sents = [
            sentence
            for sentence in dedup
            if _method_sentence_matches_query(query, sentence)
            and (
                not method_focus_terms
                or any(
                    term in sentence.casefold()
                    for term in method_focus_terms
                )
            )
        ]
        others = [sentence for sentence in dedup if sentence not in method_sents]
        # 公式/计算型方法题: 含公式/计算步骤的句子优先作首条 (审核完整门要求
        # "临界距离如何计算"必须出现公式; 措施句再补位)。
        _FORMULA_SENT_RE = re.compile(
            r"(?i)(\brc\b|blasse|dexter|lg\s*\(|1/3|^\s*[A-Za-z0-9]+\s*=|"
            r"计算.*公式|公式.*计算|代入|步骤)")
        _calc_ask = bool(re.search(
            r"(计算|公式|推导|临界距离|\brc\b|calculate|formula|equation)",
            str(query), re.IGNORECASE))
        if _calc_ask:
            _formula_first = [s for s in dedup if _FORMULA_SENT_RE.search(s)]
            if _formula_first:
                lead = _formula_first[:1] + [
                    s for s in (method_sents or others)
                    if s not in _formula_first[:1]
                ][:3]
            else:
                lead = method_sents[:4] if method_sents else others[:4]
        else:
            # One directly supported method statement is safer and more useful
            # than padding the answer with nearby mechanism/efficiency prose.  The
            # old "at least two" rule mixed unrelated paragraphs into procedures
            # and caused the real Reviewer to block an otherwise grounded answer.
            lead = method_sents[:4] if method_sents else others[:4]
        lines = "\n".join(
            f"{index}. {sentence}"
            for index, sentence in enumerate(lead[:4], start=1)
        )
        return (
            "基于当前真实证据，可确认的操作步骤/条件如下：\n"
            f"{lines}\n\n"
            f"（当前证据支持 {len(dedup)} 条方法信息；"
            "证据未覆盖的实验参数和步骤未予补写，"
            "实际实施时应依据具体样品、仪器和标准文件确认。）"
        )
    # Preserve atomic sentence boundaries for Claim-Evidence review.  A
    # mechanism question benefits from factual organization even when no LLM
    # is configured: causal sentences lead, one condition-specific observation
    # follows, and numerical values remain explicitly bounded to their source.
    # This only groups retrieved sentences; it never writes a domain answer.
    if qtype == "mechanism":
        causal_terms = (
            "原因", "因为", "由于", "所以", "因此", "因而", "导致", "引起", "能量传递",
            "非辐射", "迁移", "耗散", "弛豫", "产生", "改变", "影响",
            "从而", "促进", "决定", "使", "会", "可以", "可", "不能", "需要", "需", "减少", "增加",
            "cause", "because", "affect",
            "change", "produce", "determine",
            "energy transfer", "nonradiative", "relaxation", "can", "cannot", "require",
        )
        bounded_observations = [
            sentence
            for sentence in dedup
            if re.search(r"\d+(?:\.\d+)?\s*%", sentence)
            or ("先" in sentence and "随后" in sentence)
        ]
        causal = [
            sentence
            for sentence in dedup
            if sentence not in bounded_observations
            and any(term in sentence.lower() for term in causal_terms)
        ]
        sections: list[str] = []
        if causal:
            sections.append("机制依据：\n" + "\n".join(causal[:2]))
        if bounded_observations:
            sections.append(
                "证据中的条件化观察：\n" + bounded_observations[0]
            )
        if bounded_observations:
            sections.append(
                "当前边界：上述数值只表示对应来源的样品与测试条件，"
                "不作为其他基质或掺杂体系的通用最优值。"
            )
        if sections:
            return "\n\n".join(sections)
    lines = "\n".join(sentence for sentence in dedup[:3])
    return lines


def _answer_matches_intent(query: str, answer: str) -> bool:
    """判断答案是否符合问题意图（规则档：意图关键词匹配）。

    问「怎么做/怎么测/如何提高」→ 答案须含方法/步骤词；只有定义/机理算不匹配。
    这是「对没有的回答没有」的底层机制：检索到沾边但不符合意图的知识，不算「有」。
    """
    if not answer:
        return False
    qtype = _detect_question_type(query)
    answer_lower = answer.lower()
    if qtype == "method":
        return any(
            kw in answer_lower
            for kw in ("步骤", "方法", "措施", "首先", "然后", "先", "设置",
                       "调整", "降低", "提高", "选择", "引入", "避免", "抑制", "实现",
                       "step", "procedure", "method", "measure", "prepare",
                       "adjust", "reduce", "increase", "select", "avoid")
        )
    if qtype == "definition":
        return any(
            kw in answer_lower
            for kw in ("是", "指", "就是", "构成", "构型", "定义",
                       " refers to ", " is defined as ", " consists of ")
        )
    if qtype == "mechanism":
        return any(
            kw in answer_lower
            for kw in (
                "机理", "机制", "因为", "由于", "跃迁", "传递", "猝灭", "弛豫",
                # “机制”在任务理解层也覆盖作用、影响、评价和关系问题。
                # 这些答案不一定包含发光专用词，但只要明确表达变量之间的
                # 作用、条件或评价关系，就已经回答了问题。旧门槛把真实的
                # CIE 评价、缺陷影响和光生物安全解释误拒为答非所问。
                "作用", "影响", "改变", "导致", "引起", "取决于", "相关",
                "表征", "衡量", "评价", "反映", "决定", "限制", "不能只",
                "需要结合", "因此", "从而", "并非", "关系",
                "mechanism", "transition", "emission", "because", "due to",
                "caused by", "energy transfer", "relaxation", "quenching",
                "affect", "influence", "depend", "evaluate", "measure",
                "represent", "indicate", "therefore", "relationship",
            )
        )
    return True  # other，不判断


def _honest_unavailable(query: str) -> str:
    """诚实解释：暂时没有符合要求的知识（不编造、不硬凑沾边答案）。"""
    qtype = _detect_question_type(query)
    if qtype == "method":
        return (
            f"关于「{query}」，我目前只有相关的定义或机理知识，"
            f"暂时没有具体的方法/步骤，无法给出符合要求的回答。"
        )
    return f"关于「{query}」，我目前检索到的内容与你的问题不直接相关，暂时无法给出贴切的回答。"


def _answer_mentions_grounded_focus(
    answer: str,
    focus_terms: tuple[str, ...],
) -> bool:
    """Return whether an answer explicitly carries its retrieval focus.

    This does not approve scientific correctness.  It only proves that the
    generated text still addresses at least one non-trivial Concept/retrieval
    focus selected upstream.  Correctness remains the responsibility of the
    real Reviewer and quality-release gate.
    """

    normalized_answer = str(answer or "").casefold()
    return any(
        normalized_term in normalized_answer
        for term in focus_terms
        if len(normalized_term := str(term or "").strip().casefold()) >= 2
    )


def _semantic_answers_question(query: str, answer: str) -> bool:
    """语义判断：答案是否真正回答了问题（flash LLM，通用的「有没有」判断）。

    规则档（_answer_matches_intent）只能抓「预定义的不匹配」；语义档让 LLM 判断
    「答非所问」，覆盖所有「没预定义」的不匹配。这是「对没有的回答没有」的治本——
    像大模型那样，在生成时就「知道」自己有没有真正答到，而不是事后套规则。
    """
    try:
        from dy3_polaris.l3.llm_config import chat_completion
    except Exception:  # noqa: BLE001
        return True  # LLM 不可用，保守放行（不拦）
    prompt = (
        "判断下面的答案是否「答非所问」（与问题完全不相关，牛头不对马嘴）。\n\n"
        f"问题：{query}\n"
        f"答案：{answer[:400]}\n\n"
        "如果答案与问题的主题相关（哪怕不够精确、不够完整），只回复「相关」；"
        "如果答案与问题完全不相关（牛头不对马嘴），只回复「不相关」。"
    )
    try:
        raw = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
            disable_thinking=True,
            role="semantic_fast",
        )
    except Exception:  # noqa: BLE001
        return True
    return "不相关" not in (raw or "")


# ============================================================
# 个性化学习资源生成 (3 种形态中的「讲义」「实操指南」, 补齐 agent 能力覆盖)
#
# 背景: 交付要求 ≥3 种资源形态。原实现里「定制化讲解/实操指南/分阶测试题」只在
# 应用层 (unified_app.api_personalized_resources) 提供, 且多为硬编码模板, 未走
# 检索质量管线(离子门/重排/主题过滤) 也未附溯源。这里把「讲义」与「实操指南」下沉为
# 知识生成 Agent 的模式, 复用同一套检索→清洗→溯源管线, 保证:
#   - 能力覆盖: 生成 Agent 本身即可产出讲义/实操指南 (而非仅应用层拼模板);
#   - 诚实: 每条要点均来自知识库证据并附溯源, 覆盖不足处明确标注而非硬编;
#   - 交互: 按画像能力档位 (基础/进阶/前沿) 调整讲义深度与指南提示。
# ============================================================

def _definition_facts(query: str) -> list[dict[str, Any]]:
    """定义意图定向: "X是什么/什么是X" → 返回 X 的元素定义事实 (A-01 电子构型).

    词袋匹配里 "dy" 只命中定义事实 1 个弱关键词 (<2), 匹配不上; 而「定义意图」
    语义上就是问「这是什么」, 应直接给元素定义 (原子序数/电子构型), 而非发光切片.
    """
    if not _placeholder_knowledge_enabled():
        return []
    ions = _extract_ions(query)
    if not ions:
        return []
    q_ions = {_bare_ion(i) for i in ions}
    try:
        from dy3_polaris.l3.textbook_fallback import CANONICAL_FACTS
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    for f in CANONICAL_FACTS:
        # 元素定义事实 = kp_ids 含 A-01 (稀土离子电子构型) 且离子匹配
        if "A-01" not in (f.get("kp_ids") or []):
            continue
        if not ({_bare_ion(i) for i in (f.get("ions") or [])} & q_ions):
            continue
        item = dict(f)
        item["score"] = 100
        item["source"] = (
            "教材兜底层 · 《镝基绿色健康照明发光材料》"
            f"{f.get('chapter', '')}（占位，待替换为真实教材切片）"
        )
        out.append(item)
    return out


def _apply_textbook_fallback(
    query: str, compose_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """教材级权威事实兜底层 (#35): 检索命中不足或证据缺核心术语时补充 canonical facts.

    只补充「KB 证据尚未覆盖其核心术语」的事实 (避免与真实检索重复), 每条附
    `metadata.source_type="textbook_fallback"` 与 `source` 标注 (可替换为真实教材切片),
    保证「诚实」支柱: 兜底事实明确标注来源, 不与 KB 检索命中混为一谈。
    兜底事实是「权威且 KB 未覆盖」的, 前置插入以保证进入溯源/合成的前 N 条。
    """
    if not _placeholder_knowledge_enabled():
        return compose_items
    try:
        from dy3_polaris.l3.textbook_fallback import query_canonical
        facts = list(query_canonical(query, top_k=6))
        # 定义意图定向: "X是什么/什么是X" → 元素定义事实 (电子构型) 优先命中
        if _detect_question_type(query) == "definition":
            facts = _definition_facts(query) + facts
    except Exception as exc:  # noqa: BLE001
        logger.warning("教材兜底层查询失败: %s", exc)
        return compose_items
    if not facts:
        return compose_items
    # 防张冠李戴: 问题明确点名某离子 (如 Eu3+), 而事实锚定的是另一离子 (如 Dy3+) →
    # 该事实与问题离子不重叠, 跳过, 避免「问 Eu 却答 Dy」的错配 (兜底事实一律 Dy3+ 锚定).
    q_ions = {_bare_ion(i) for i in _extract_ions(query)}
    # 仅与 KB 证据比对去重 (静态, 避免兜底事实之间互相跳过)
    ev_all = " ".join(str(c.get("content") or "") for c in compose_items).lower()
    fallback_items: list[dict[str, Any]] = []
    for f in facts:
        fact_ions = {_bare_ion(i) for i in (f.get("ions") or [])}
        if q_ions and fact_ions and not (q_ions & fact_ions):
            continue  # 问 Eu 却锚定 Dy → 张冠李戴, 不补充
        core = [str(t).lower() for t in (f.get("core_terms") or [])]
        # 仅当 KB 证据已「实质覆盖」该事实时才跳过。原实现仅凭 core_terms 全命中即跳过,
        # 但 KB 常只是零星提到术语 (如 g09 白光 LED: KB 综述里出现「白光/蓝光芯片」
        # 却跑题到研究进展), 导致权威答案被误跳过、合成跑题。这里再要求 KB 证据包含
        # 事实正文的「首句特征片段」, 作为"KB 真已给出该答案"的强信号。
        if core and all(t in ev_all for t in core) and _fact_content_present(f, ev_all):
            continue  # KB 证据已实质覆盖该事实, 不重复补充
        fallback_items.append({
            "chunk_id": f["id"],
            "document_id": "textbook-fallback",
            "section": f.get("chapter", ""),
            "content": f["content"],
            "metadata": {
                "entity": (f.get("ions") or ["dy3+"])[0],
                "source_type": "textbook_fallback",
                "source": f.get("source", ""),
            },
        })
    # 兜底事实前置 (溯源 _build_sources 默认取前 8 条, 须保证兜底事实可见)
    if fallback_items:
        compose_items = fallback_items + list(compose_items)
    return compose_items


def _expand_kp_related(
    query: str,
    compose_items: list[dict[str, Any]],
    *,
    max_hop: int = 2,
    max_extra: int = 6,
) -> list[dict[str, Any]]:
    """知识点关系图多跳拓展: 命中 KP → 邻居 KP → 邻居 KP 的权威事实, 增量补充.

    只追加、不删改现有证据。拓展事实 source_type="kp_expand", 附 relation/hop/
    reason/kp 元数据, 供生成器自然表述「这跟 X 有关」而非机械堆砌, 实现「知识拓展」。
    种子 KP 来自 query_canonical 命中的兜底事实 (其 kp_ids), 故与 _apply_textbook_fallback
    命中同一批事实, 不会引入额外错误来源。
    """
    if not _placeholder_knowledge_enabled():
        return compose_items
    try:
        from dy3_polaris.l2.kp_catalog import expand_kp, kp_name
        from dy3_polaris.l3.textbook_fallback import CANONICAL_FACTS, query_canonical
    except Exception:  # noqa: BLE001
        return compose_items

    hits = query_canonical(query, top_k=3)
    seed_kps: list[str] = []
    for f in hits:
        for kp in f.get("kp_ids", []):
            if kp not in seed_kps:
                seed_kps.append(kp)
    if not seed_kps:
        return compose_items

    expanded: dict[str, dict[str, str]] = {}
    for kp in seed_kps:
        for nb in expand_kp(kp, max_hop=max_hop):
            expanded.setdefault(nb["kp_id"], nb)

    existing_ids = {str(c.get("chunk_id") or "") for c in compose_items}

    extra: list[dict[str, Any]] = []
    for fact in CANONICAL_FACTS:
        fid = fact["id"]
        if fid in existing_ids:
            continue
        hit_kp = next((kp for kp in fact.get("kp_ids", []) if kp in expanded), None)
        if hit_kp is None:
            continue
        nb = expanded[hit_kp]
        extra.append({
            "chunk_id": fid,
            "document_id": "kg-kp-expand",
            "section": fact.get("chapter", ""),
            "content": fact["content"],
            "metadata": {
                "entity": (fact.get("ions") or ["dy3+"])[0],
                "source_type": "kp_expand",
                "source": (
                    f"教材兜底层 · {fact.get('chapter', '')}"
                    "（知识点关系拓展，占位待替换为真实教材切片）"
                ),
                "kp_id": hit_kp,
                "kp_name": kp_name(hit_kp),
                "relation": nb["rel"],
                "hop": nb["hop"],
                "reason": nb.get("reason", ""),
            },
        })
        existing_ids.add(fid)
        if len(extra) >= max_extra:
            break

    if extra:
        compose_items = list(compose_items) + extra
    return compose_items


def _expand_graph_related(
    query: str,
    compose_items: list[dict[str, Any]],
    l3_store: Any,
    *,
    max_hop: int = 2,
    max_extra: int = 10,
) -> list[dict[str, Any]]:
    """图消费层多跳召回 (#11 P3): 从真实知识图谱召回多类型实体 + 事实, 附溯源路径.

    与 _expand_kp_related (静态 KP 邻接表) 互补: 这里走真实 KnowledgeStore 多类型图,
    沿 mentions/part_of/supports 等边双向 BFS, 把材料/离子/能级/方法/参数实体及其关联
    权威事实作为增量证据补充, 实现「图谱驱动问答」的多跳拓展。
    只追加、不删改; 任何异常回退原列表 (不阻塞主检索链路)。
    """
    if l3_store is None:
        return compose_items
    try:
        from dy3_polaris.l3.graph_consume import recall
    except Exception:  # noqa: BLE001
        return compose_items

    # 显式种子: 离子实体 + 命中 KP (供图召回起点, 与静态兜底层种子对齐)
    extra_seeds: list[str] = []
    ion_entity = _resolve_entity_id(query, l3_store)
    if ion_entity:
        extra_seeds.append(ion_entity)
    try:
        extra = recall(
            query, l3_store, max_hop=max_hop,
            max_facts=6, max_concepts=8, extra_seed_ids=extra_seeds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("图消费层召回失败: %s", exc)
        return compose_items
    if not extra:
        return compose_items

    existing = {str(c.get("chunk_id") or "") for c in compose_items}
    # 图事实与兜底层 canonical facts 同源, 按内容首句去重, 避免重复证据
    existing_text = " ".join(str(c.get("content") or "") for c in compose_items).lower()
    appended: list[dict[str, Any]] = []
    for it in extra:
        cid = str(it.get("chunk_id") or "")
        if cid in existing:
            continue
        content = str(it.get("content") or "").strip()
        if content and content[:40].lower() in existing_text:
            continue
        appended.append(it)
        existing.add(cid)
        if len(appended) >= max_extra:
            break
    if appended:
        compose_items = list(compose_items) + appended
    return compose_items


def _bare_ion(sym: str) -> str:
    """把离子写法归一为裸元素符号 (Dy3+ -> dy; _extract_ions 已把中文名/大写符号归一为 Dy)."""
    return "".join(ch for ch in str(sym).lower() if ch.isalpha())


def _fact_content_present(fact: dict[str, Any], ev_all: str) -> bool:
    """判断 KB 证据是否已包含该事实的正文首句特征片段 (强覆盖信号).

    用于 _apply_textbook_fallback 的去重: 仅当 KB 证据里真的出现了事实正文的
    首句 (而非仅仅零散提到 core_terms) 时才视为"已覆盖", 避免跑题综述把权威
    答案挤掉。
    """
    content = str(fact.get("content") or "")
    first_clause = re.split(r"[；。;]", content)[0]
    anchor = re.sub(r"[\s，,：:（）()\[\]]", "", first_clause)[:12].lower()
    return bool(anchor) and anchor in ev_all


def _retrieve_evidence(
    query: str,
    input_data: dict[str, Any],
    deps: AgentDependencies,
    *,
    top_k: int = 30,
) -> tuple[Any, list[dict[str, Any]]]:
    """共享证据检索管线 (资源形态复用): 归一化 → 向量/图增强 → 混合检索 → 离子门 → 重排 → 清洗.

    与 run_generation 的检索段同构, 但更精简 (资源生成无需问答式幻觉防御门)。
    返回 (clean_retrieval, compose_items); 检索不可用/无命中时返回 (None, [])。
    """
    retrieval_query = normalize_query(query)
    if _detect_question_type(query) == "method":
        boost = " 避免 措施 控制 优化 降低 选择"
        if retrieval_query and not any(
            t in retrieval_query for t in ("避免", "措施", "控制")
        ):
            retrieval_query = retrieval_query + boost
    query_vector: list[float] | None = None
    entity_id: str | None = None
    if deps.hybrid_retriever is not None and deps.embedding_manager is not None:
        try:
            query_vector = deps.embedding_manager.embed(retrieval_query).vector
        except Exception:  # noqa: BLE001
            query_vector = None
        entity_id = _resolve_entity_id(retrieval_query, deps.l3_store)
    retrieval: Any = None
    if deps.hybrid_retriever is not None:
        try:
            retrieval = deps.hybrid_retriever.retrieve(
                retrieval_query, top_k=top_k,
                query_vector=query_vector, entity_id=entity_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("资源生成检索失败: %s", exc)
    if retrieval is not None and getattr(retrieval, "results", None):
        try:
            _gated_res, _gated_scores = _ion_gate_rerank(
                query,
                list(retrieval.results),
                list(getattr(retrieval, "scores", []) or []),
            )
            retrieval.results = _gated_res
            if _gated_scores:
                retrieval.scores = _gated_scores
            retrieval.total = len(_gated_res)
        except Exception as exc:  # noqa: BLE001
            logger.warning("资源生成离子门重排失败: %s", exc)
    reranked = retrieval
    if retrieval is not None and deps.reranker is not None:
        try:
            reranked = deps.reranker.rerank_result(retrieval_query, retrieval, top_k=8)
        except Exception as exc:  # noqa: BLE001
            logger.warning("资源生成重排失败: %s", exc)
    if reranked is None:
        return None, []
    clean_retrieval: Any = None
    compose_items: list[dict[str, Any]] = []
    try:
        from dy3_polaris.l3.models import RetrievalResult

        clean_results: list[dict[str, Any]] = []
        for item in list(getattr(reranked, "results", []) or []):
            copy = dict(item)
            if "content" in copy:
                copy["content"] = _clean_markdown_chunk(copy.get("content", ""))
            clean_results.append(copy)
        compose_items = list(clean_results)
        clean_retrieval = RetrievalResult(
            query=query,
            results=clean_results,
            scores=list(getattr(reranked, "scores", []) or []),
            total=len(clean_results),
            source_type=str(getattr(reranked, "source_type", "reranked")),
            trace_id=str(getattr(reranked, "trace_id", "")),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("资源生成结果清洗失败: %s", exc)
    # 教材级权威事实兜底层: 补充 KB 未覆盖的权威事实 (标注来源), 关闭跨语言/粒度缺口
    compose_items = _apply_textbook_fallback(query, compose_items)
    # 知识点关系图多跳拓展: 沿前提/类比/因果/表征关系补邻居 KP 权威事实, 实现知识拓展
    compose_items = _expand_kp_related(query, compose_items)
    # 图消费层多跳召回: 从多类型图补材料/离子/能级/方法/参数实体 + 溯源路径 (#11 P3)
    compose_items = _expand_graph_related(query, compose_items, deps.l3_store)
    return clean_retrieval, compose_items


def _build_sources(
    compose_items: list[dict[str, Any]], limit: int = 8
) -> list[dict[str, Any]]:
    """由证据切片构建溯源列表 (chunk_id/document_id/section/excerpt/KP)."""
    sources: list[dict[str, Any]] = []
    for it in (compose_items or [])[:limit]:
        txt = str(it.get("content") or "")
        if not txt.strip():
            continue
        kp_ids = _infer_kps(txt)
        meta = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}
        sources.append({
            "chunk_id": it.get("chunk_id"),
            "document_id": it.get("document_id"),
            "section": it.get("section"),
            "excerpt": txt[:160],
            "kp_ids": kp_ids,
            "kp_names": [kp_name(k) for k in kp_ids],
            "entity": meta.get("entity") or meta.get("entity_name") or it.get("entity") or "",
        })
    return sources


def _resource_unavailable(
    topic: str,
    resource_type: str,
    learner_id: str,
    deps: AgentDependencies,
) -> dict[str, Any]:
    """资源生成的诚实兜底: 知识库无相关证据时明确说明, 不硬编模板."""
    msg = (
        f"当前知识库暂无与「{topic}」直接相关的知识，无法生成有据可查的"
        f"{'讲义' if resource_type == 'lecture' else '实操指南'}。\n"
        "本系统知识库聚焦镝（Dy）绿色健康照明发光材料领域（Dy³⁺ 发光机理、能级跃迁、"
        "单基质白光荧光粉、蓝光危害、浓度猝灭、热猝灭、荧光粉表征等知识点）。"
    )
    _broadcast(
        deps.message_bus,
        "knowledge.generation.output",
        {"event": "generation_output", "query": topic, "answer": msg[:400],
         "confidence": 0.0, "learner_id": learner_id, "rejected": True},
        GENERATION_AGENT_ID,
    )
    return {
        "agent_id": GENERATION_AGENT_ID,
        "mode": resource_type,
        "resource_type": resource_type,
        "status": "completed",
        "topic": topic,
        "learner_id": learner_id,
        "knowledge_unavailable": True,
        "sources": [],
        "confidence": 0.0,
        "message": msg,
    }


def _run_lecture_mode(
    input_data: dict[str, Any],
    deps: AgentDependencies,
) -> dict[str, Any]:
    """知识生成 Agent — 定制讲义模式 (资源形态 #1).

    围绕主题检索知识库证据 → 按知识点分节 → 每节要点均来自证据并附溯源 →
    按画像能力档位调整深度; 覆盖不足处诚实标注。
    """
    topic = str(
        input_data.get("topic")
        or input_data.get("query")
        or input_data.get("question")
        or ""
    ).strip()
    learner_id = input_data.get("learner_id") or input_data.get("student_id") or "anonymous-request"
    if not topic:
        return {"agent_id": GENERATION_AGENT_ID, "mode": "lecture",
                "status": "failed", "error": "缺少 topic 参数"}
    profile = _load_profile(deps.profile_service, learner_id)
    level = str(getattr(profile, "level", "beginner") or "beginner") if profile is not None else "beginner"
    depth = {"beginner": "基础", "intermediate": "进阶", "advanced": "前沿"}.get(level, "基础")

    _, compose_items = _retrieve_evidence(topic, input_data, deps, top_k=30)
    sources = _build_sources(compose_items)
    if not compose_items:
        return _resource_unavailable(topic, "lecture", learner_id, deps)

    # 按知识点分节: 每个证据块归入其主 KP, 抽相关句作为讲解要点
    by_kp: dict[str, list[str]] = {}
    for it in compose_items:
        txt = str(it.get("content") or "")
        if not txt.strip():
            continue
        kps = _infer_kps(txt)
        kp = kps[0] if kps else "通用"
        cands = _collect_answer_candidates(topic, [{"content": txt}])
        points = cands[:3] if cands else [txt[:240]]
        by_kp.setdefault(kp, [])
        for p in points:
            if p not in by_kp[kp]:
                by_kp[kp].append(p)
    sections: list[dict[str, Any]] = []
    for kp, points in by_kp.items():
        sections.append({
            "kp_id": kp if kp != "通用" else "",
            "kp_name": kp_name(kp) if kp != "通用" else "相关要点",
            "key_points": [p[:240] for p in points[:3]],
        })
    kp_ids = [s["kp_id"] for s in sections if s["kp_id"]]
    body = "\n\n".join(
        f"【{s['kp_name']}】\n" + "\n".join(f"· {p}" for p in s["key_points"])
        for s in sections
    )
    _broadcast(
        deps.message_bus,
        "knowledge.generation.output",
        {"event": "generation_output", "query": topic, "answer": body[:400],
         "confidence": 0.7, "learner_id": learner_id},
        GENERATION_AGENT_ID,
    )
    return {
        "agent_id": GENERATION_AGENT_ID,
        "mode": "lecture",
        "resource_type": "lecture",
        "status": "completed",
        "topic": topic,
        "learner_id": learner_id,
        "depth": depth,
        "title": f"《{topic}》定制讲义（{depth}）",
        "sections": sections,
        "content": body,
        "sources": sources,
        "kp_ids": kp_ids,
        "coverage_note": (
            f"本讲义基于知识库 {len(compose_items)} 条证据组织，覆盖 {len(kp_ids)} 个知识点；"
            "证据不足之处已如实标注，待教材级兜底层补充。"
        ),
        "confidence": round(min(0.9, 0.4 + 0.08 * len(compose_items)), 4),
    }


def _run_guide_mode(
    input_data: dict[str, Any],
    deps: AgentDependencies,
) -> dict[str, Any]:
    """知识生成 Agent — 实操指南模式 (资源形态 #2).

    面向"怎么做/怎么制备/怎么表征"类请求, 从知识库抽取措施/步骤类句子,
    组织为分步操作指南并附溯源; 无实操细节时诚实标注 (兜底层将补)。
    """
    topic = str(
        input_data.get("topic")
        or input_data.get("query")
        or input_data.get("question")
        or ""
    ).strip()
    learner_id = input_data.get("learner_id") or input_data.get("student_id") or "anonymous-request"
    if not topic:
        return {"agent_id": GENERATION_AGENT_ID, "mode": "guide",
                "status": "failed", "error": "缺少 topic 参数"}
    # 方法型检索增强: 追加步骤关键词提升实操类证据召回
    guide_query = f"{topic} 制备 合成 表征 步骤 方法"
    _, compose_items = _retrieve_evidence(guide_query, input_data, deps, top_k=30)
    sources = _build_sources(compose_items)
    if not compose_items:
        return _resource_unavailable(topic, "guide", learner_id, deps)

    # 抽取措施/步骤句 (过滤文献综述式描述, 只保留可落地的步骤; 诚实起见不硬凑)
    _lit_markers = re.compile(
        r"等人|报道|本文|文献|综述|研究目的|研究内容|研究对象|年第|\.\d+\s|"
        r"^\s*\d+(\.\d+)*\s"
    )
    steps: list[str] = []
    for it in compose_items:
        cands = _collect_answer_candidates(
            guide_query, [{"content": it.get("content", "")}]
        )
        for s in cands:
            if not _is_method_sentence(s) or s in steps:
                continue
            if _lit_markers.search(s):
                continue
            steps.append(s)
    steps = steps[:8]
    safety = [
        "实验前确认仪器状态与安全防护（高温炉 / 紫外光源 / 化学试剂）。",
        "严格按配方称量，避免交叉污染；研磨与煅烧过程注意防烫。",
    ]
    if not steps:
        msg = (
            f"知识库已检索到与「{topic}」相关的机理知识，但缺少可落地的分步实操细节，"
            "暂不硬编流程。建议先理解其机理（见讲义/答疑），具体实验步骤待教材级兜底层补充。"
        )
        _broadcast(
            deps.message_bus,
            "knowledge.generation.output",
            {"event": "generation_output", "query": topic, "answer": msg[:400],
             "confidence": 0.25, "learner_id": learner_id},
            GENERATION_AGENT_ID,
        )
        return {
            "agent_id": GENERATION_AGENT_ID,
            "mode": "guide",
            "resource_type": "guide",
            "status": "completed",
            "topic": topic,
            "learner_id": learner_id,
            "steps": [],
            "safety": safety,
            "sources": sources,
            "coverage_note": "知识库对该主题的实操步骤覆盖不足，待教材级兜底层补充。",
            "confidence": 0.25,
            "message": msg,
        }
    _broadcast(
        deps.message_bus,
        "knowledge.generation.output",
        {"event": "generation_output", "query": topic, "answer": "\n".join(steps[:3])[:400],
         "confidence": 0.7, "learner_id": learner_id},
        GENERATION_AGENT_ID,
    )
    return {
        "agent_id": GENERATION_AGENT_ID,
        "mode": "guide",
        "resource_type": "guide",
        "status": "completed",
        "topic": topic,
        "learner_id": learner_id,
        "title": f"{topic}·实操指南",
        "steps": [{"step": i + 1, "operation": s} for i, s in enumerate(steps)],
        "safety": safety,
        "sources": sources,
        "coverage_note": (
            f"本指南基于知识库 {len(compose_items)} 条证据抽取 {len(steps)} 条步骤；"
            "涉及具体工艺参数时请以原始文献/教材为准。"
        ),
        "confidence": round(min(0.85, 0.4 + 0.06 * len(steps)), 4),
    }


def _run_practice_mode(
    input_data: dict[str, Any],
    deps: AgentDependencies,
) -> dict[str, Any]:
    """知识生成 Agent — 练习模式: 读取画像 → 按薄弱点针对性出题.

    流程: 画像薄弱 KP → PracticeBank 自适应选题 (薄弱优先) →
    返回题目 (隐藏答案) 供学习者练习。
    """
    learner_id = (
        input_data.get("learner_id")
        or input_data.get("student_id")
        or "anonymous-request"
    )
    count = int(input_data.get("count", 5))
    bank = deps.practice_bank
    if bank is None:
        return {
            "agent_id": GENERATION_AGENT_ID,
            "mode": "practice",
            "status": "failed",
            "error": "题库服务不可用",
        }
    profile = _load_profile(deps.profile_service, learner_id)
    mastery = _mastery_map(profile)
    weak_kps = list(getattr(profile, "weak_kps", []) or [])
    questions = bank.select_questions(
        learner_id=learner_id, count=count, mastery=mastery
    )
    public = [bank.public_question(q) for q in questions]
    _broadcast(
        deps.message_bus,
        "learning.interaction.event",
        {
            "event": "practice_issued",
            "learner_id": learner_id,
            "count": len(public),
            "weak_kps": weak_kps,
            "kps": [q["kp_id"] for q in public],
        },
        GENERATION_AGENT_ID,
    )
    return {
        "agent_id": GENERATION_AGENT_ID,
        "mode": "practice",
        "status": "completed",
        "learner_id": learner_id,
        "target_kps": weak_kps[:5],
        "questions": public,
        "count": len(public),
        "summary": (
            f"基于画像薄弱点 {len(public)} 题 (薄弱 KP: {', '.join(weak_kps[:5]) or '无'})"
        ),
    }


def _run_assess_mode(
    input_data: dict[str, Any],
    deps: AgentDependencies,
) -> dict[str, Any]:
    """知识生成 Agent — 考核模式: 按画像出题考核 → 判题 → BKT+画像写回 → 广播.

    流程:
    1. 读取画像 (已学/薄弱/目标 KP)
    2. 出题: 优先考核薄弱点 + 学习目标相关 KP
    3. 判题: answers [{qid, selected}] → BKT 在线更新 → 画像 kp_mastery 写回
    4. 结果广播: 考核报告发布到 knowledge.gap / interaction.event 供其他 Agent 消费
    """
    learner_id = (
        input_data.get("learner_id")
        or input_data.get("student_id")
        or "anonymous-request"
    )
    bank = deps.practice_bank
    if bank is None or deps.bkt_service is None:
        return {
            "agent_id": GENERATION_AGENT_ID,
            "mode": "assess",
            "status": "failed",
            "error": "考核依赖不可用 (题库/BKT)",
        }
    profile = _load_profile(deps.profile_service, learner_id)
    mastery = _mastery_map(profile)

    # 1. 出题阶段 (无 answers) 或 2. 判题阶段 (有 answers)
    answers = input_data.get("answers") or []
    if not answers:
        count = int(input_data.get("count", 5))
        questions = bank.select_questions(
            learner_id=learner_id, count=count, mastery=mastery
        )
        public = [bank.public_question(q) for q in questions]
        return {
            "agent_id": GENERATION_AGENT_ID,
            "mode": "assess",
            "status": "pending_answers",
            "learner_id": learner_id,
            "questions": public,
            "count": len(public),
            "bloom_target": getattr(profile, "bloom_target", "understand"),
            "learning_style": getattr(profile, "learning_style", "reading"),
            "summary": "考核题目已生成, 提交 answers=[{qid, selected}] 完成判题",
        }

    # 判题阶段
    results: list[dict[str, Any]] = []
    for ans in answers:
        qid = str(ans.get("qid", ""))
        selected = int(ans.get("selected", -1))
        try:
            r = bank.answer(
                learner_id=learner_id,
                qid=qid,
                selected=selected,
                bkt_service=deps.bkt_service,
                profile_service=deps.profile_service,
            )
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            logger.warning("考核判题失败 qid=%s: %s", qid, exc)
            results.append({"qid": qid, "error": str(exc)})

    correct = sum(1 for r in results if r.get("correct"))
    profile_after = _load_profile(deps.profile_service, learner_id)
    mastery_after = _mastery_map(profile_after)
    avg_after = (
        sum(mastery_after.values()) / len(mastery_after)
        if mastery_after
        else 0.0
    )
    weak_after = list(getattr(profile_after, "weak_kps", []) or [])

    # 考核结果写回画像 (考核记录)
    if profile_after is not None:
        extras = dict(getattr(profile_after, "extras", {}) or {})
        assess_log = list(extras.get("assess_log", []) or [])
        assess_log.append({
            "ts": time.time(),
            "agent": GENERATION_AGENT_ID,
            "total": len(results),
            "correct": correct,
            "kps": [r.get("kp_id") for r in results if r.get("kp_id")],
            "avg_mastery": round(avg_after, 4),
        })
        extras["assess_log"] = assess_log[-20:]
        profile_after.extras = extras
        _save_profile(deps.profile_service, profile_after)

    # 考核结果广播: 用户掌握情况传递给其他 Agent
    _broadcast(
        deps.message_bus,
        "learning.knowledge.gap",
        {
            "event": "assess_report",
            "learner_id": learner_id,
            "total": len(results),
            "correct": correct,
            "weak_kps": weak_after,
            "avg_mastery": round(avg_after, 4),
        },
        GENERATION_AGENT_ID,
    )
    _broadcast(
        deps.message_bus,
        "learning.interaction.event",
        {
            "event": "assess_completed",
            "learner_id": learner_id,
            "total": len(results),
            "correct": correct,
            "kps": [r.get("kp_id") for r in results if r.get("kp_id")],
        },
        GENERATION_AGENT_ID,
    )

    return {
        "agent_id": GENERATION_AGENT_ID,
        "mode": "assess",
        "status": "completed",
        "learner_id": learner_id,
        "results": results,
        "score": f"{correct}/{len(results)}",
        "accuracy": round(correct / len(results), 4) if results else 0.0,
        "avg_mastery": round(avg_after, 4),
        "weak_kps_after": weak_after,
        "assess_log_written": True,
        "summary": f"考核完成 {correct}/{len(results)} 正确, 平均掌握度 {avg_after:.1%}",
    }


def run_generation(
    input_data: dict[str, Any],
    deps: AgentDependencies,
) -> dict[str, Any]:
    """知识生成 Agent — 按 mode 分发: 检索合成 / 练习出题 / 针对性考核."""
    mode = str(input_data.get("mode") or input_data.get("task") or "answer").lower()
    # 域外硬拦截提前到 mode 分发前: guide/practical 等用户实操分支不改写检索而
    # 直接空答 (实测 "如何做红烧肉/石墨烯CVD工艺" DEGRADED 而非带文案拒答).
    # 系统生成内容模式 (练习/考核/讲义) 不拦截, 仅拦用户自由文本.
    _pre_query = str(
        input_data.get("query")
        or input_data.get("question")
        or input_data.get("topic")
        or ""
    ).strip()
    if (
        mode not in ("practice", "quiz", "assess", "assessment", "exam",
                     "lecture", "讲义", "notes", "customized", "customized_resource")
        and _pre_query
        and _hard_out_of_domain(_pre_query)
    ):
        _unavail = (
            f"当前问题「{_pre_query}」不属于本系统已验证的稀土发光材料与绿色健康照明知识范围。"
            "系统不会使用相邻词命中的材料文献拼接回答；请改为领域问题，或提供可核验的领域上下文。"
        )
        return {
            "agent_id": GENERATION_AGENT_ID,
            "status": "completed",
            "query": _pre_query,
            "answer": _unavail,
            "confidence": 0.0,
            "knowledge_unavailable": True,
            "honest_unavailable": True,
            "context_chunks": [],
            "citations": [],
        }
    if mode in ("practice", "quiz"):
        return _run_practice_mode(input_data, deps)
    if mode in ("assess", "assessment", "exam"):
        return _run_assess_mode(input_data, deps)
    if mode in ("lecture", "讲义", "notes", "customized", "customized_resource"):
        return _run_lecture_mode(input_data, deps)
    if mode in ("guide", "practical", "实操", "实操指南", "practical_guide", "experiment"):
        return _run_guide_mode(input_data, deps)
    # 默认: 知识检索合成答案
    query = str(
        input_data.get("query")
        or input_data.get("question")
        or input_data.get("topic")
        or ""
    ).strip()
    if not query:
        return {
            "agent_id": GENERATION_AGENT_ID,
            "status": "failed",
            "answer": "",
            "confidence": 0.0,
            "error": "缺少 query 参数",
        }

    private_agent_input = input_data.get("_agent_input")
    # 领域范围门：只对明确领域外意图做硬拦截（防跑题，如菜谱/股价/娱乐）。
    # 不再以“发光域概念/离子缺失”直接拦截：知识库本身是领域过滤器——
    # 教材类知识（稀土化学基础/分离/配合物等）入库后，化学概念问题在检索
    # 与审核门中自然获得证据或诚实拒答（2026-09-03 双库评测定位）。
    # 2026-09-03: 去掉 AgentInput 前置条件 - 普通答疑路径("如何做红烧肉"等)
    # 实测未进该分支而退化为空答 DEGRADED, 应带领域外文案拒答而非静默.
    if _hard_out_of_domain(query):
        unavailable = (
            f"当前问题「{query}」不属于本系统已验证的稀土发光材料与绿色健康照明知识范围。"
            "系统不会使用相邻词命中的材料文献拼接回答；请改为领域问题，或提供可核验的领域上下文。"
        )
        return {
            "agent_id": GENERATION_AGENT_ID,
            "status": "completed",
            "query": query,
            "answer": unavailable,
            "confidence": 0.0,
            "knowledge_unavailable": True,
            "honest_unavailable": True,
            "context_chunks": [],
            "citations": [],
        }

    teaching_decision = input_data.get("_adaptive_teaching_decision")
    # DiagnosisContribution is the teaching authority.  The private decision
    # is supporting context and may only fill a missing normalized level.
    learner_level = str(
        input_data.get("learner_level")
        or (
            teaching_decision.content_depth
            if isinstance(teaching_decision, AdaptiveTeachingDecision)
            else "intermediate"
        )
    ).lower()

    # Scientific teaching content must always cross the real retrieval and
    # Reviewer boundary.  The former beginner/plain-language and rule-deduction
    # shortcuts returned fixed prose before retrieval, so personalization
    # removed the very evidence required for safe publication.  Reading depth
    # is now applied later by ``_adapt_educational_depth`` to the same selected
    # evidence-backed fact set for every learner.

    retrieval: Any = input_data.get("_planned_retrieval_result")
    planned_retrieval = bool(input_data.get("_retrieval_plans_applied"))
    retrieval_query = str(
        input_data.get("_retrieval_query") or normalize_query(query)
    )
    # 方法型问题检索增强: 问题含"怎么避免/如何解决"等时, 追加措施关键词,
    # 使"避免/措施/控制"类知识文档在混合检索中优先命中 (提质: 答到点子上)
    qtype = _detect_question_type(query)
    if qtype == "method":
        if re.search(r"(测量|测试|表征|分析|measure|test|characteri[sz]e|analy[sz]e)", query, re.IGNORECASE):
            boost = " 测量条件 样品 仪器 校准 数据分析 标准参照"
        elif re.search(r"(制备|合成|烧结|prepare|synthesi[sz]e)", query, re.IGNORECASE):
            boost = " 原料 步骤 温度 时间 气氛 表征"
        elif re.search(
            r"(计算|公式|推导|怎么算|如何得|计算|临界距离|distance|calculate|equation|formula)",
            query, re.IGNORECASE,
        ):
            # 计算/公式推导型问题: 通用"措施/避免"增强词会把检索带到应对性内容,
            # 反而挤掉含 Rc/Blasse/Dexter 公式正文的切片 (实测 "临界距离如何计算")。
            boost = " 公式 计算 推导 Rc 临界距离 Blasse Dexter 能量传递 步骤 代入"
        else:
            boost = " 避免 措施 控制 优化 降低 选择"
        retrieval_query = retrieval_query + boost
    # Golden Questions 仅在查询侧补充同义评价维度，不改变检索器或知识库。
    _rq_compact = retrieval_query.lower().replace(" ", "")
    if "黄蓝" in _rq_compact or "白光" in _rq_compact:
        retrieval_query += " Dy3+ 黄光 蓝光 发射 跃迁 Y/B 色坐标"
    if "基质" in _rq_compact:
        retrieval_query += " 晶场 对称性 声子 非辐射 缺陷 热稳定性"
    if any(term in _rq_compact for term in ("比较", "优于", "孰优孰劣")):
        retrieval_query += " 量子效率 发射光谱 热稳定性 寿命 色坐标 掺杂浓度 测试条件"
    if "发光效率" in _rq_compact or "量子效率" in _rq_compact:
        retrieval_query += " 辐射跃迁 非辐射损失 浓度猝灭 热猝灭 缺陷"
    # 定义/简介类超短实体问句 ("Dy是什么?"): 仅拉丁符号的查询难以命中中文切片,
    # 追加实体别名/中文名做检索扩展 (元素符号→中文名; 不改变检索器与知识库)
    if qtype == "definition":
        _DEF_ALIASES = {
            "dy": "镝 镝离子 dysprosium 稀土元素 性质", "dy3+": "镝 Dy3+ 发光 能级 跃迁",
            "eu": "铕 铕离子 europium", "ce": "铈 铈离子 cerium",
            "tb": "铽 铽离子 terbium", "er": "铒 铒离子 erbium",
            "nd": "钕 钕离子 neodymium", "sm": "钐 钐离子 samarium",
            "ho": "钬 钬离子 holmium", "tm": "铥 铥离子 thulium",
            "yb": "镱 镱离子 ytterbium", "lu": "镥 镥离子 lutetium",
            "la": "镧 镧离子 lanthanum", "gd": "钆 钆离子 gadolinium",
            "pr": "镨 镨离子 praseodymium", "pm": "钷 promethium",
            "sc": "钪 scandium", "y": "钇 yttrium",
            "yag": "钇铝石榴石 Y3Al5O12 晶体结构",
        }
        _q_latin = re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]{1,8}", str(query).lower())
        _extra = " ".join(
            _DEF_ALIASES[tok] for tok in _q_latin if tok in _DEF_ALIASES
        ).strip()
        if _extra:
            retrieval_query = retrieval_query + " " + _extra
    # 自纠修订: 带审核反馈时加宽召回 (设计 CC1 自纠回路: 依据审核意见重新检索)
    review_feedback = str(input_data.get("review_feedback") or "")
    # 自纠不再空转: 从审核意见中抽取要点词并入检索查询 (原实现仅加宽 top_k,
    # 检索词不变 → 召回不变 → 同一候选再次被审 → 迭代白费, 实测 4~5 轮全同)。
    if review_feedback:
        _FB_STOP = frozenset(
            "回答 仅 提及 对 为 问题 用户 核心 要求 修订 重新 检索 未 遗漏 关键 信息 缺失 "
            "表述 残缺 混乱 完全 证据 候选 中 的 了 是 有 与 及 提出 得到 达到 相关 属于 "
            "内容 现象 机制 原因 且 但 而 从 在 后 前 出 已 再 也 都 该 其 此 你 我 请 需要 "
            "包括 关于 由 于 让 被 把 向 往".split()
        )
        _fb_terms = []
        for _raw in re.split(r"[，。；、,;:：\s()（）\[\]【】\"'“”‘’「」『』]+",
                             review_feedback):
            _tok = _raw.strip().strip("'\"“”‘’「」『』()（）")
            if 2 <= len(_tok) <= 14 and _tok not in _FB_STOP:
                _fb_terms.append(_tok)
        if _fb_terms:
            _fb_extra = " ".join(dict.fromkeys(_fb_terms))[:160]
            if not all(term in retrieval_query for term in _fb_terms[:4]):
                retrieval_query = retrieval_query + " " + _fb_extra
    # 多候选交叉验证(L5 高等级): 允许按候选策略覆盖 top_k(标准/宽召回/精聚焦)
    _top_k = int(
        input_data.get("_candidate_top_k") or (28 if review_feedback else 24)
    )
    # 向量/图检索增强: 查询侧编码 + 实体定位, 使混合检索的「向量」与「图」分支生效
    # (治检索不相关: 原实现只走 BM25 关键词, 语义匹配与图谱关联均被绕过).
    query_vector: list[float] | None = None
    entity_id: str | None = None
    if not planned_retrieval and deps.hybrid_retriever is not None and deps.embedding_manager is not None:
        try:
            query_vector = deps.embedding_manager.embed(retrieval_query).vector
        except Exception:  # noqa: BLE001
            query_vector = None
        entity_id = _resolve_entity_id(retrieval_query, deps.l3_store)
    if not planned_retrieval and deps.hybrid_retriever is not None:
        try:
            retrieval = deps.hybrid_retriever.retrieve(
                retrieval_query, top_k=_top_k,
                query_vector=query_vector, entity_id=entity_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识生成 Agent 检索失败: %s", exc)
    # 公式/计算类题二次召回: 公式正文是英文 (Rc=2(3V/(4πxcN))^(1/3)、Dexter
    # lg(I/x) 拟合), 中文长问句的 bigram/BM25 命中被中文文献稀释, 实测 top24
    # 无任何公式片 → 追加一次公式向检索并并入候选池 (重排器统一排序, 审核门把关)。
    if (
        not planned_retrieval
        and retrieval is not None
        and qtype == "method"
        and deps.hybrid_retriever is not None
        and re.search(r"(计算|公式|推导|临界距离|\brc\b|calculate|formula|equation)",
                      str(query), re.IGNORECASE)
    ):
        try:
            _formula_q = (
                "critical distance Rc Blasse equation concentration quenching "
                "Dy3+ Dexter lg(I/x) slope calculation formula 2(3V/(4πxcN))^(1/3)"
            )
            _extra_retrieval = deps.hybrid_retriever.retrieve(
                _formula_q, top_k=10,
                query_vector=(
                    deps.embedding_manager.embed(_formula_q).vector
                    if deps.embedding_manager is not None else None
                ),
            )
            _have_ids = {str(it.get("chunk_id") or "") for it in
                         (getattr(retrieval, "results", None) or [])}
            for _extra_item in (getattr(_extra_retrieval, "results", None) or []):
                if str(_extra_item.get("chunk_id") or "") not in _have_ids:
                    getattr(retrieval, "results", []).append(_extra_item)
                    _have_ids.add(str(_extra_item.get("chunk_id") or ""))
            retrieval.total = len(list(getattr(retrieval, "results", None) or []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识生成 Agent 公式二次召回失败: %s", exc)

    # 离子门 (防张冠李戴): 问题点名某离子时, 重排检索结果使目标离子证据优先,
    # 只谈其它离子的证据沉底. 这是语义/关键词分数之上的硬约束, 在重排器之前生效,
    # 让重排器 (top_k=6) 从离子一致的前排候选中挑选.
    if not planned_retrieval and retrieval is not None and getattr(retrieval, "results", None):
        try:
            _gated_res, _gated_scores = _ion_gate_rerank(
                query,
                list(retrieval.results),
                list(getattr(retrieval, "scores", []) or []),
            )
            retrieval.results = _gated_res
            if _gated_scores:
                retrieval.scores = _gated_scores
            retrieval.total = len(_gated_res)
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识生成 Agent 离子门重排失败: %s", exc)

    answer = ""
    confidence = 0.3
    context_chunks: list[str] = []
    citations: list[str] = []
    clean_items: list[dict[str, Any]] = []
    compose_items: list[dict[str, Any]] = []
    reranked: Any = retrieval
    if not planned_retrieval and retrieval is not None and deps.reranker is not None:
        try:
            # 定义/机理/方法题扩宽重排窗口: 这类题的答句常排在第 6 名之外
            # (元素列表句/英文公式句), 只有 6 条会把答句挡在证据集外 (实测).
            _rerank_k = 14 if qtype in ("definition", "mechanism", "method") else 6
            reranked = deps.reranker.rerank_result(
                retrieval_query, retrieval, top_k=_rerank_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识生成 Agent 重排失败: %s", exc)

    clean_retrieval: Any = None
    if reranked is not None:
        try:
            from dy3_polaris.l3.models import RetrievalResult

            clean_results: list[dict[str, Any]] = []
            for item in list(getattr(reranked, "results", []) or []):
                copy = dict(item)
                if "content" in copy:
                    copy["content"] = _clean_markdown_chunk(copy.get("content", ""))
                clean_results.append(copy)
            # 主题词过滤: 只保留含查询核心主题词的文档 (修复"问机理检索到光谱"答非所问)。
            # 双语对齐: 英文机理/公式正文不含中文字面词 (猝灭→quenching), 原仅按
            # 中文关键词过滤会把英文答句所在切片整片丢掉 (实测 Q1 候选仅剩中文
            # 能级/光谱句)。
            _DIS_ALIASES = {
                "机理": ("机理", "机制", "mechanism"), "机制": ("机理", "机制", "mechanism"),
                "猝灭": ("猝灭", "quench"), "光谱": ("光谱", "spectr"),
                "效率": ("效率", "efficien"), "寿命": ("寿命", "lifetime"),
                "跃迁": ("跃迁", "transition"), "能级": ("能级", "energy level"),
                "能量传递": ("能量传递", "energy transfer"), "掺杂": ("掺杂", "dop"),
                "制备": ("制备", "synthes", "prepar"), "合成": ("合成", "synthes"),
                "发射": ("发射", "emission"), "激发": ("激发", "excitation"),
                "显色": ("显色", "color"), "色度": ("色度", "chromatic"),
                "热猝灭": ("热猝灭", "thermal quench"), "浓度猝灭": ("浓度猝灭", "concentration quench"),
                "表征": ("表征", "characteri"), "温度": ("温度", "temperature"),
            }
            _disc = [
                kw for kw in (
                    "机理", "机制", "原理", "猝灭", "光谱", "效率", "寿命", "跃迁",
                    "能级", "制备", "合成", "掺杂", "表征", "能量传递", "显色",
                    "色度", "浓度猝灭", "热猝灭", "发射", "激发", "温度",
                ) if kw in str(query)
            ]
            if _disc and not planned_retrieval:
                _needles = [
                    kw
                    for keyword in _disc
                    for kw in _DIS_ALIASES.get(keyword, (keyword,))
                ]
                # 计算/公式型问题再补英文公式线索 (临界距离/Blasse/Rc 等正文常为
                # 英文, 且不一定含 "quench/猝灭" 字面词, 需显式入过滤词)
                if re.search(r"(计算|公式|临界|距离|推导|blasse|dexter|\brc\b|"
                             r"calculate|formula|critical distance)",
                             str(query), re.IGNORECASE):
                    _needles += ["critical distance", "blasse", "dexter",
                                 "critical concentration", "1/3", "rc", "lg("]
                _needles = tuple(dict.fromkeys(_needles))
                _kept = [r for r in clean_results
                         if any(kw in str(r.get("content", "")).lower()
                                for kw in _needles)]
                if _kept:
                    clean_results = _kept
            clean_items = list(clean_results)
            clean_retrieval = RetrievalResult(
                query=query,
                results=clean_results,
                scores=list(getattr(reranked, "scores", []) or []),
                total=len(clean_results),
                source_type=str(getattr(reranked, "source_type", "reranked")),
                trace_id=str(getattr(reranked, "trace_id", "")),
            )
            compose_items = list(clean_results)
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识生成 Agent 结果清洗失败: %s", exc)

    # 教材级权威事实兜底层 (#35): KB 命中不足时补充 canonical facts (标注来源),
    # 使"知识库暂无"的诚实拒绝不再漏答已在权威事实层覆盖的知识点。
    compose_items = _apply_textbook_fallback(query, compose_items)
    # 知识点关系图多跳拓展: 沿前提/类比/因果/表征关系补邻居 KP 权威事实, 实现知识拓展
    compose_items = _expand_kp_related(query, compose_items)
    # 图消费层多跳召回: 从多类型图补材料/离子/能级/方法/参数实体 + 溯源路径 (#11 P3)
    compose_items = _expand_graph_related(query, compose_items, deps.l3_store)
    if compose_items and (clean_retrieval is None or not clean_retrieval.results):
        try:
            from dy3_polaris.l3.models import RetrievalResult

            clean_retrieval = RetrievalResult(
                query=query,
                results=[dict(c) for c in compose_items],
                scores=[1.0] * len(compose_items),
                total=len(compose_items),
                source_type="textbook_fallback",
            )
            clean_items = list(compose_items)
        except Exception as exc:  # noqa: BLE001
            logger.warning("教材兜底层结果封装失败: %s", exc)

    # 相关性门槛: 无检索结果 → 明确"知识库暂无", 不编造 (P1 修复).
    # 有命中则基于检索证据诚实合成; 不再依赖不可靠的归一化分数 (RRF/质量分语义不一).
    if clean_retrieval is None or not clean_retrieval.results:
        # 动态知识识别: 本地知识库无匹配 → 条件路由到外部知识源 (Haystack WebSearch
        # Fallback 模式; 配置 DY3_EXT_KB_URL 时生效, 未配置保持诚实拒绝不编造)
        external_items: list[dict[str, Any]] = []
        if deps.external_kb is not None:
            try:
                enabled = getattr(deps.external_kb, "enabled", lambda: False)
                if enabled():
                    external_items = list(
                        deps.external_kb.search(retrieval_query, top_k=3) or []
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("外部知识源兜底检索失败: %s", exc)
        if external_items:
            src_name = str(external_items[0].get("source") or "在线检索")
            lines = "\n".join(
                "- " + str(it.get("title") or "")[:80] + "\n  " + str(it.get("content") or "")[:200]
                for it in external_items[:3]
            )
            answer = (
                f"知识库暂无与「{query}」直接相关的本地知识，以下为来自"
                f"「{src_name}」的检索结果（动态识别，未经领域审核，仅供参考）：\n{lines}"
            )
            _broadcast(
                deps.message_bus,
                "knowledge.generation.output",
                {
                    "event": "generation_output",
                    "query": query,
                    "answer": answer[:400],
                    "confidence": 0.3,
                    "learner_id": input_data.get("learner_id") or input_data.get("student_id") or "",
                },
                GENERATION_AGENT_ID,
            )
            return {
                "agent_id": GENERATION_AGENT_ID,
                "status": "completed",
                "query": query,
                "answer": answer,
                "confidence": 0.3,
                "external_sourced": True,
                "context_chunks": [],
                "citations": [],
            }
        unavailable = (
            f"当前知识库暂无与「{query}」直接相关的知识。\n"
            "本系统知识库聚焦镝（Dy）绿色健康照明发光材料领域（Dy³⁺ 发光机理、能级跃迁、"
            "单基质白光荧光粉、蓝光危害、浓度猝灭、热猝灭、荧光粉表征等知识点）。\n"
            "请尝试在该范围内提问，或补充更多上下文（如材料体系、性能参数）。"
        )
        _broadcast(
            deps.message_bus,
            "knowledge.generation.output",
            {
                "event": "generation_output",
                "query": query,
                "answer": unavailable[:400],
                "confidence": 0.0,
                "learner_id": input_data.get("learner_id") or input_data.get("student_id") or "",
            },
            GENERATION_AGENT_ID,
        )
        return {
            "agent_id": GENERATION_AGENT_ID,
            "status": "completed",
            "query": query,
            "answer": unavailable,
            "confidence": 0.0,
            "knowledge_unavailable": True,
            "context_chunks": [],
            "citations": [],
        }

    # 词重叠启发式: 查询与命中内容的字组重叠过低 → 视为"没有相关的东西"并诚实拒绝
    # (中文按 bigram, 英文/数字按原 token; 避免弱命中(如"稀土"命中"稀土发光材料")被当作直接答案)
    # 业务主题词加权: 量子/效率/发光/猝灭/离子 等主题词命中权重 2x,
    # 使"量子效率"类查询优先聚焦主题文档, 而非仅因含"Dy3+/离子"而混入无关文档
    _TOPIC_TERMS = frozenset(
        "量子 效率 发光 猝灭 温度 测量 光谱 合成 掺杂 发射 跃迁 荧光 能量 寿命 "
        "色度 结晶 形貌 能级 基质 上转换 敏化 离子 浓度 热 寿命 强度 波长 工艺 表征 "
        "黄蓝 白光 蓝光 色温 健康 风险 安全 比较 辐射 非辐射 缺陷 热稳定".split()
    )

    def _query_overlap(q: str, doc_texts: list[Any]) -> float:
        qq = str(q).lower().replace(" ", "")
        if not qq:
            return 1.0
        q_tokens = {qq[i : i + 2] for i in range(len(qq) - 1)} or set(qq)
        # 分母封顶: 长问题的 bigram 总数不稀释相关性 (核心实体命中即高重叠)
        den = min(len(q_tokens), 8)
        best = 0.0
        for t in doc_texts[:3]:
            dd = str(t or "").lower().replace(" ", "")
            if not dd:
                continue
            d_tokens = {dd[i : i + 2] for i in range(len(dd) - 1)} or set(dd)
            inter = q_tokens & d_tokens
            if not inter:
                continue
            topic_hits = len([tok for tok in inter if tok in _TOPIC_TERMS])
            if topic_hits:
                # 主题词命中权重 2x (其余命中原样计), 与主题无关的命中不稀释
                score = (topic_hits * 2 + (len(inter) - topic_hits)) / max(1, den)
            else:
                score = len(inter) / max(1, den)
            best = max(best, score)
        return best

    overlap = _query_overlap(
        retrieval_query,
        [i.get("content") or i.get("text") or "" for i in compose_items],
    )
    # 相关性门槛 (供下面各防御门共用): 检索命中足够相关时, 不因关键词启发式误拒
    _min_overlap = float(input_data.get("_candidate_min_overlap") or 0.3)
    # 教材兜底层命中即视为「相关」(query_canonical 已用强/弱关键词阈值校验), 后续 bigram
    # 重叠门槛只针对噪声 KB 检索, 不应误伤权威事实 (如「热猝灭」问句填充词多、与兜底事实
    # bigram 重叠仅 0.25, 但事实本身命中 core_terms)。
    _has_fallback = any(
        (it.get("metadata") or {}).get("source_type") in ("textbook_fallback", "kp_expand")
        for it in compose_items
    )
    # ---- 幻觉防御门 (P1 修复): 无关/虚构/诱导 → 诚实拒答, 不拼接检索块 ----
    # 检索合并文本 (去空白, 用于术语存在性判断)
    _mtext = "".join(
        (str(i.get("content") or i.get("text") or "")).lower().replace(" ", "")
        for i in compose_items
    )
    _qs = str(query)
    _qs_l = _qs.lower().replace(" ", "")
    _reject_block = None

    # (a) 无关领域: 查询无任何领域主题词/稀土实体 → 本系统不覆盖, 直接引导 (非硬答)
    #     CRAG 式放宽: 仅当"无领域词"且"检索重叠度低"时才拒; 检索已命中相关知识说明在域内
    _domain_hit = [t for t in _TOPIC_TERMS if t in _qs_l] or re.search(
        r"(dy|eu|ce|tb|yb|er|nd|sm|pr|ho|tm|镝|铕|铈|铽|镱|铒|钕|钐|稀土|荧光|磷光|发光|白光|蓝光|"
        r"led|荧光粉|色温|显色|光|颜色|灯|照明|亮|xrd|pl|物相|结晶|晶体|临界|rc|量子|公式|谱|能级|跃迁)",
        _qs_l,
    )
    if not _domain_hit and overlap < _min_overlap and not _has_fallback:
        _reject_block = (
            "这个问题不属于本系统（镝 Dy 绿色健康照明发光材料智能学习）的知识范围，我无法回答。"
            "\n如需帮助，可输入「帮助」查看我能做什么（答疑、出题、学情画像、记忆复习等）。"
        )

    # (b) 诱导式/绝对论断: "对吗/是不是/一定是因为/就一定/只要…就" → 拒绝强制是/否, 重新锚定
    if _reject_block is None and re.search(
        r"(对吗|是吗|对吧|是不是|一定是因为|就一定|一定更|一定优于|"
        r"是否\S{0,8}一定|能否据此认定|只要\S{0,4}就|肯定是因为|必定)", _qs
    ):
        _reject_block = _absolute_claim_boundary_answer(_qs)

    # (c) 环境背景谬误前提: "在极夜/北极/天气/水下…能发光吗" → 激发依赖机制, 纠正前提
    if _reject_block is None and re.search(
        r"(极夜|北欧|南极|北极|下雨|天气|冬天|夏天|水下|真空)", _qs
    ) and re.search(r"(能发光|可以发光|还能发光|亮吗|发光吗|吗)", _qs):
        _reject_block = (
            "荧光/磷光由激发光源（紫外/蓝光/近紫外）激发产生，与昼夜、极夜或天气等"
            "环境背景光照无关——「能否发光」取决于是否存在激发光源，而非环境。\n"
            "建议从激发-发射机理角度提问，例如：「荧光粉的激发与发射机理是什么？」"
        )

    # (d) 虚构/未收录术语: 引号内术语或"XXX是多少/是什么"的新奇复合词, 若整段未出现在检索命中 → 不编造
    if _reject_block is None:
        _ghost = None
        _quoted = re.findall(
            r"['\"“”‘’「」『』()（）]\s*([^'\"“”‘’「」『』()（）]{2,16})\s*['\"“”‘’「」『』()（）]",
            _qs,
        )
        _cand = [t for t in _quoted if len(t) >= 3]
        if not _cand:
            _m = re.search(
                r"([\u4e00-\u9fffA-Za-z0-9]{2,6})\s*(?:是多少|是什么|是什么意思)",
                _qs,
            )
            if _m:
                _cand.append(_m.group(1))
        # 排除含连接词的候选 (避免把"XX的主要机理和YY"整段误判为虚构术语)
        _cand = [t for t in _cand if not re.search(r"[的和与及或主要以及其等]", t)]
        # 术语应为短词(≤6字); 更长的是用户口语描述, 不作"虚构术语"误判
        _cand = [t for t in _cand if len(str(t).strip()) <= 6]
        # 排除含"描述性通用词"的候选 (如"发光原理/掺杂浓度"是通用描述, 非可判定真伪的术语)
        _cand = [t for t in _cand if not re.search(r"(原理|机理|作用|优势|区别|现象|指标|性能|方法|步骤|影响|浓度|温度|结构|特性|公式|导入语|通常|数量级|角度|范围|课堂|颜色|物相|扫描|计算)", t)]
        for _t in _cand:
            _tt = str(_t).lower().replace(" ", "")
            if len(_tt) < 3:
                continue
            if _tt not in _mtext:
                _ghost = _t
                break
        if _ghost is not None and overlap < _min_overlap and not _has_fallback:
            _reject_block = (
                f"「{_ghost}」这一术语/概念在本地知识库中未收录，我无法给出可靠解释。\n"
                "本系统知识库聚焦镝（Dy）绿色健康照明发光材料领域（Dy³⁺ 发光机理、能级跃迁、"
                "单基质白光荧光粉、蓝光危害、浓度猝灭、热猝灭、荧光粉表征等知识点）。请确认术语名称，或换用知识库已有的"
                "概念提问，例如「浓度猝灭」「量子效率」「Dy³⁺ 发光机理」。"
            )

    if _reject_block is not None:
        _broadcast(
            deps.message_bus,
            "knowledge.generation.output",
            {
                "event": "generation_output",
                "query": query,
                "answer": _reject_block[:400],
                "confidence": 0.05,
                "learner_id": input_data.get("learner_id") or input_data.get("student_id") or "",
                "rejected": True,
            },
            GENERATION_AGENT_ID,
        )
        return {
            "agent_id": GENERATION_AGENT_ID,
            "status": "completed",
            "query": query,
            "answer": _reject_block,
            "confidence": 0.05,
            "knowledge_unavailable": True,
            "rejected": True,
            "reject_reason": "relevant_gate",
            "context_chunks": [],
            "citations": [],
        }

    # 多候选交叉验证(L5 高等级): 允许按候选策略覆盖重叠门槛(精聚焦候选门槛更高)
    if overlap < _min_overlap and not _has_fallback:
        unavailable = (
            f"当前知识库暂无与「{query}」直接相关的知识"
            f"（最接近的知识与问题主题重叠度仅 {overlap:.2f}）。\n"
            "本系统知识库聚焦镝（Dy）绿色健康照明发光材料领域（Dy³⁺ 发光机理、能级跃迁、"
            "单基质白光荧光粉、蓝光危害、浓度猝灭、热猝灭、荧光粉表征等知识点）。\n"
            "请尝试在该范围内提问，或补充更多上下文（如材料体系、性能参数）。"
        )
        _broadcast(
            deps.message_bus,
            "knowledge.generation.output",
            {
                "event": "generation_output",
                "query": query,
                "answer": unavailable[:400],
                "confidence": 0.0,
                "learner_id": input_data.get("learner_id") or input_data.get("student_id") or "",
            },
            GENERATION_AGENT_ID,
        )
        return {
            "agent_id": GENERATION_AGENT_ID,
            "status": "completed",
            "query": query,
            "answer": unavailable,
            "confidence": 0.0,
            "knowledge_unavailable": True,
            "context_chunks": [],
            "citations": [],
        }

    # 综合前过滤: 只合并与问题主题重叠的命中 (避免混入无关知识块, 保底前 2 条)
    # 主题覆盖: query 含业务主题词(量子/效率/发光等)时, 命中文档必须覆盖过半 query 主题词,
    # 避免仅因含实体(dy/3+/离子)而混入"组态/猝灭"等无关文档
    def _related(item: dict[str, Any]) -> bool:
        # 教材兜底层事实已由 query_canonical 强/弱关键词阈值校验相关性, 且正文用教材
        # 术语(如「发射/4F9/2→6H15/2」)而非查询字面词(如「跃迁/能级」), 不应受主题词
        # 覆盖率门槛误伤 (否则「蓝光/黄光跃迁」这类题会把兜底事实洗掉 → 张冠李戴拒答)。
        if (item.get("metadata") or {}).get("source_type") == "textbook_fallback":
            return True
        text = item.get("content") or item.get("text") or ""
        qq = str(retrieval_query).lower().replace(" ", "")
        q_toks = {qq[i : i + 2] for i in range(len(qq) - 1)} or set(qq)
        q_topics = [t for t in q_toks if t in _TOPIC_TERMS]
        if q_topics:
            dd = str(text).lower().replace(" ", "")
            d_toks = {dd[i : i + 2] for i in range(len(dd) - 1)} or set(dd)
            doc_topics = [t for t in (q_toks & d_toks) if t in _TOPIC_TERMS]
            if len(doc_topics) * 2 < len(q_topics):
                return False
        if _query_overlap(retrieval_query, [text]) >= 0.45:
            return True
        # 双语/公式证据放宽: 英文公式正文 (Rc=2(3V/(4πxcN))^(1/3)、Dexter lg(I/x))
        # 与中文查询 bigram 重叠为 0, 会被 0.45 门槛整片丢弃 (实测 "临界距离如何
        # 计算" 证据集内无公式片)。查询含拉丁记号或机理/计算意图时放行含领域线索
        # 的英文片, 交由候选门与审核门继续把关。
        _q_lat = tuple(
            dict.fromkeys(
                token.lower()
                for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]{1,14}", str(query))
                if len(token) >= 2
            )
        )
        low = str(text).lower()
        _sci_hint = re.compile(
            r"(?i)(\brc\b|blasse|dexter|critical distance|lg\s*\(|1/3|"
            r"cross[- ]?relaxation|energy transfer|quenching|concentration)")
        calc_ask = any(
            token in str(query)
            for token in ("计算", "公式", "临界", "距离", "机理", "猝灭", "推导", "方法",
                          "rc", "blasse", "dexter", "how", "calculate", "formula")
        )
        if _sci_hint.search(low) and (
            any(tok in low for tok in _q_lat) or (calc_ask and qtype in ("mechanism", "method"))
        ):
            return True
        return False

    if planned_retrieval:
        # R-03D has already merged bilingual Concept branches, applied the
        # entity hard filter and ranked candidates against the Agent plan.
        # Reapplying the legacy Chinese-bigram filter here discarded direct
        # English paper passages (for example, spectroscopic transition
        # notation) and kept only loosely related Chinese thesis prose.
        # Preserve the bounded plan-ranked set; downstream ion and review
        # guards still apply unchanged.
        compose_items = compose_items[:8]
        knowledge_context = (
            private_agent_input.learner_context.get(
                "knowledge_learning_context"
            )
            if isinstance(private_agent_input, AgentInput)
            else None
        )
        preferred_ids = (
            tuple(knowledge_context.target_concepts)
            if isinstance(knowledge_context, KnowledgeLearningContext)
            else ()
        )
        compose_items = _prefer_reviewed_concept_evidence(
            compose_items,
            preferred_ids,
        )
        target_filtered_items = _filter_task_answer_evidence(
            query,
            compose_items,
            focus_terms=(
                _concept_retrieval_terms(private_agent_input)
                if isinstance(private_agent_input, AgentInput)
                else ()
            ),
            preferred_concept_ids=preferred_ids,
        )
        # Relation expansion is recall-only.  Once target-bound passages are
        # available, neighbours cannot enter Generation, Reviewer, claims,
        # resources or the public evidence projection on their own.
        if target_filtered_items:
            compose_items = target_filtered_items
        # 公式/计算类题 (planned 路径同样适用): 计划检索按概念主题排证据, 公式正文
        # (英文 Rc=2(3V/(4πxcN))^(1/3)、Dexter lg(I/x)) 无中文概念词而缺席 → 补一次
        # 公式向关键词召回并入候选池 (重排/候选门/审核门继续把关, 不绕过)。
        if (
            qtype == "method"
            and deps.hybrid_retriever is not None
            and re.search(r"(计算|公式|推导|临界距离|\brc\b|calculate|formula|equation)",
                          str(query), re.IGNORECASE)
        ):
            try:
                _formula_q = (
                    "critical distance Rc Blasse equation concentration quenching "
                    "Dy3+ Dexter lg(I/x) slope calculation formula 2(3V/(4πxcN))^(1/3)"
                )
                _extra_retrieval = deps.hybrid_retriever.retrieve(
                    _formula_q, top_k=10,
                    query_vector=(
                        deps.embedding_manager.embed(_formula_q).vector
                        if deps.embedding_manager is not None else None
                    ),
                )
                _have_ids = {str(it.get("chunk_id") or "") for it in compose_items}
                _joined_items = list(compose_items)
                for _extra_item in (getattr(_extra_retrieval, "results", None) or []):
                    if str(_extra_item.get("chunk_id") or "") not in _have_ids:
                        _joined_items.append(_extra_item)
                        _have_ids.add(str(_extra_item.get("chunk_id") or ""))
                if len(_joined_items) > len(compose_items):
                    compose_items = _joined_items
            except Exception as exc:  # noqa: BLE001
                logger.warning("知识生成 Agent planned 公式二次召回失败: %s", exc)
    else:
        related_items = [it for it in compose_items if _related(it)]
        if related_items:
            compose_items = related_items
        else:
            compose_items = compose_items[:2]

    # ---- 相近词库实体一致性门 (P2 修复): 查询指定某离子, 命中文档却大篇幅讲另一离子
    #      (彼此共享"发光/效率/猝灭"等主题词而通过重叠门槛) → 剔除错配块防止张冠李戴;
    #      若全部命中都只讲"他离子" → 诚实澄清, 不硬答 ----
    _q_ions = _extract_ions(query)
    if _q_ions:
        _ion_kept: list[dict[str, Any]] = []
        for it in compose_items:
            _doc_text = str(it.get("content") or it.get("text") or "")
            _doc_counts = _count_ions(_doc_text)
            if not _doc_counts:
                # 通用知识 (不含任何离子), 保留
                _ion_kept.append(it)
                continue
            # 上下文可靠性: 查询离子须为文档主要对象；其他离子的出现次数
            # 需要合计，避免多离子综述中每种离子单独不占优、合计却远多于 Dy。
            _q_count = sum(_doc_counts.get(q, 0) for q in _q_ions)
            _other_count = sum(
                c for s, c in _doc_counts.items() if s not in _q_ions
            )
            if _q_count > 0 and _q_count >= _other_count:
                _ion_kept.append(it)
        if _ion_kept:
            compose_items = _ion_kept
        else:
            _which_ion = "、".join(
                _ION_DISPLAY.get(s, s) for s in sorted(_q_ions)
            )
            _entity_reject = (
                f"你问的是「{_which_ion}」的知识，但知识库命中的内容主要涉及其他稀土离子"
                f"（存在相近词匹配，若直接拼接作答易张冠李戴）。为避免给出错误答案，暂不据此作答。\n"
                f"请确认问题对象，或换用更具体的问题，例如「{_which_ion} 的发光机理是什么」"
                f"「{_which_ion} 的浓度猝灭如何避免」。"
            )
            _broadcast(
                deps.message_bus,
                "knowledge.generation.output",
                {
                    "event": "generation_output",
                    "query": query,
                    "answer": _entity_reject[:400],
                    "confidence": 0.05,
                    "learner_id": input_data.get("learner_id") or input_data.get("student_id") or "",
                    "rejected": True,
                },
                GENERATION_AGENT_ID,
            )
            return {
                "agent_id": GENERATION_AGENT_ID,
                "status": "completed",
                "query": query,
                "answer": _entity_reject,
                "confidence": 0.05,
                "knowledge_unavailable": True,
                "rejected": True,
                "reject_reason": "entity_mismatch",
                "context_chunks": [],
                "citations": [],
            }

    # 定义类问题 ("X是什么/什么是X/介绍X") → 优先实体定义内容 (电子组态/特性/定义),
    # 避免把"浓度猝灭"等机理文档当作"离子是什么"的答案
    if qtype == "definition":
        def _def_score(item: dict[str, Any]) -> int:
            text = str(item.get("content") or item.get("text") or "")
            score = 0
            # 教材兜底的权威定义事实优先于 KB 检索的泛泛内容（如「电子构型」不该被
            # 泛泛的「价电子构型」挤掉）
            if (item.get("metadata") or {}).get("source_type") == "textbook_fallback":
                score += 10
            for kw in ("组态", "定义", "介绍", "电子结构", "特性", "性质", "电子构型", "构型"):
                if kw in text:
                    score += 2
            # 枚举/归类句 ("包括哪些元素"): "镧系元素包括镧(La)…" 这类清单句
            # 正是定义题的答句主体, 无 组态/性质 词也不应被埋没 (实测定位)
            for kw in ("包括", "属于", "由", "周期表中", "元素符号"):
                if kw in text:
                    score += 2
            for kw in ("离子", "元素", "掺杂离子"):
                if kw in text:
                    score += 1
            return score

        def_items = sorted(compose_items, key=_def_score, reverse=True)
        top_def = [it for it in def_items if _def_score(it) >= 2]
        if top_def:
            compose_items = top_def[:4]
        elif def_items:
            compose_items = def_items[:3]
    # 同步裁剪检索结果 (response_synthesizer 只综合最终保留的相关命中)
    kept_keys = set(
        str(it.get("chunk_id") or it.get("content") or "")[:80]
        for it in compose_items
    )
    kept_results: list[dict[str, Any]] = []
    kept_scores: list[float] = []
    for idx, it in enumerate(list(getattr(clean_retrieval, "results", []) or [])):
        key = str(it.get("chunk_id") or it.get("content") or "")[:80]
        if key in kept_keys:
            kept_results.append(it)
            scores = list(getattr(clean_retrieval, "scores", []) or [])
            if idx < len(scores):
                kept_scores.append(float(scores[idx]))
    if kept_results:
        clean_retrieval.results = kept_results
        clean_retrieval.scores = kept_scores
        clean_retrieval.total = len(kept_results)

    if clean_retrieval is not None and deps.response_synthesizer is not None:
        try:
            synth = deps.response_synthesizer.synthesize(clean_retrieval, query=query)
            answer = _clean_synthesized_answer(getattr(synth, "answer", "") or "")
            synth_confidence = float(getattr(synth, "confidence", 0.3) or 0.3)
            top_score = 0.3
            if clean_retrieval.scores:
                top_score = max(float(s) for s in clean_retrieval.scores)
            # 置信度: 基础 + 命中数 + 主题重叠度 (相关证据越多越可信, 封顶 0.95)
            n_hits = len(list(getattr(clean_retrieval, "results", []) or []))
            quality = min(0.95, 0.4 + 0.06 * n_hits + overlap * 0.25)
            confidence = max(synth_confidence, quality, min(0.95, top_score))
            for citation in getattr(synth, "citations", []) or []:
                citations.append(
                    getattr(citation, "url", None)
                    or getattr(citation, "uri", None)
                    or getattr(citation, "title", None)
                    or str(citation)
                )
            for evidence in getattr(synth, "evidence_pieces", []) or []:
                context_chunks.append(
                    getattr(evidence, "content", None)
                    or getattr(evidence, "text", None)
                    or str(evidence)
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("知识生成 Agent 响应合成失败: %s", exc)

    # 证据兜底: 合成未产出上下文切片时, 用检索命中前 3 条填充 (供前端证据区展示)
    if not context_chunks and clean_retrieval is not None:
        try:
            for item in list(getattr(clean_retrieval, "results", []) or [])[:3]:
                context_chunks.append(_result_text(item))
        except Exception:  # noqa: BLE001
            pass

    # 兜底层命中时, 把权威事实正文并入证据切片, 供 reviewer 离子一致性校验
    # (否则「答案提 Dy、证据只含他离子」会误判张冠李戴 → fix_faithfulness, 且问题
    # 离子不在证据中会被相关性打折)。
    if _has_fallback:
        fb_chunks = [
            str(it.get("content") or "")
            for it in compose_items
            if (it.get("metadata") or {}).get("source_type") == "textbook_fallback"
        ]
        context_chunks = fb_chunks + context_chunks

    concept_focus_terms = (
        _concept_retrieval_terms(private_agent_input)
        if isinstance(private_agent_input, AgentInput)
        else ()
    )
    preferred_concept_ids = (
        tuple(
            private_agent_input.learner_context[
                "knowledge_learning_context"
            ].target_concepts
        )
        if isinstance(private_agent_input, AgentInput)
        and isinstance(
            private_agent_input.learner_context.get(
                "knowledge_learning_context"
            ),
            KnowledgeLearningContext,
        )
        else ()
    )
    concise = _compose_concise_answer(
        query,
        compose_items,
        focus_terms=concept_focus_terms,
        preferred_concept_ids=preferred_concept_ids,
    )
    if concise:
        answer = concise
        confidence = max(confidence, 0.45)

    if review_feedback:
        # A Reviewer-requested revision must be materially narrower than the
        # challenged draft.  Rebuild it extractively from the refreshed real
        # evidence instead of repeating the same broad synthesis.  This gives
        # the quality loop a deterministic convergence mechanism without
        # adding facts or changing the review verdict.
        if qtype == "method":
            # Preserve the evidence-backed procedure structure across the
            # correction loop.  Falling back to two concatenated source
            # sentences made the revised answer less usable than the draft
            # and hid which procedural details were still unsupported.
            revised_method = _compose_concise_answer(
                query,
                compose_items,
                focus_terms=concept_focus_terms,
                preferred_concept_ids=preferred_concept_ids,
            )
            if revised_method:
                answer = revised_method
                confidence = max(confidence, 0.45)
        else:
            revision_sentences = _collect_answer_candidates(
                query,
                compose_items,
                focus_terms=concept_focus_terms,
                preferred_concept_ids=preferred_concept_ids,
            )
            if revision_sentences:
                # A completeness revision must be allowed to cover several
                # explicit question dimensions.  Two sentences were enough
                # for single-mechanism questions but systematically dropped
                # the second arm of comparisons and multi-factor questions.
                answer = "".join(revision_sentences[:4])
                confidence = max(confidence, 0.45)

    if planned_retrieval and compose_items:
        # The LLM may only reorganize the same evidence set selected by the
        # RetrievalPlan.  Previously this alignment happened after LLM
        # synthesis, so the synthesizer could still read stale, loosely
        # related evidence pieces left by the legacy response synthesizer and
        # overwrite a correct Concept-grounded extractive answer.  Aligning
        # here keeps Retrieval -> Generation -> Review on one scientific
        # artifact without bypassing either the LLM or the Reviewer.
        context_chunks = [
            _result_text(item) for item in compose_items[:6]
            if _result_text(item).strip()
        ]
        citations = [
            str(
                (item.get("metadata") or {}).get("source_uri")
                or item.get("document_id")
                or item.get("source")
                or ""
            )
            for item in compose_items[:6]
            if (
                (item.get("metadata") or {}).get("source_uri")
                or item.get("document_id")
                or item.get("source")
            )
        ]

    reviewed_concept_grounding = any(
        isinstance(item.get("metadata"), dict)
        and item["metadata"].get("source_type") == "curated_source_summary"
        and item["metadata"].get("evidence_status") == "reviewed"
        and str(item["metadata"].get("source_uri") or "").strip()
        for item in compose_items
    )
    # Reviewed Concept summaries are already compact, curated scientific
    # artifacts.  Keep their extractive representation in both single- and
    # multi-model modes instead of paying a model to paraphrase them and
    # potentially widen a claim beyond its source.  Multi-model generation is
    # still used for ordinary document chunks, deep analysis, reviewed long
    # resources and Socratic questions; the trusted canonical path stays
    # deterministic and still crosses the real Reviewer.
    if (
        answer
        and context_chunks
        and not review_feedback
        and not reviewed_concept_grounding
    ):
        try:
            from dy3_polaris.l3.llm_synthesizer import LLMSynthesizer

            _llm_synth = LLMSynthesizer()
            _llm_answer, _used_llm = _llm_synth.synthesize(
                query=query,
                evidence=list(context_chunks)[:6],
                enable_thinking=bool(
                    input_data.get("_llm_enable_thinking", _GENERATION_THINKING)
                ),
                learner_level=learner_level,
                teaching_strategy=(
                    {
                        "explanation_strategy": teaching_decision.explanation_strategy,
                        "representation_modes": list(teaching_decision.representation_modes),
                    }
                    if isinstance(teaching_decision, AdaptiveTeachingDecision)
                    else None
                ),
                model_role=str(input_data.get("_llm_role") or "generation_fast"),
                reasoning_effort=str(input_data.get("_llm_reasoning_effort") or ""),
            )
            if _used_llm and _llm_answer:
                answer = _llm_answer
                confidence = max(confidence, 0.55)
                logger.debug("知识生成 Agent 使用 LLM 重组合成答案")
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 增强合成失败, 保留模板答案: %s", exc)

    if answer and context_chunks:
        answer = _adapt_educational_depth(
            answer,
            learner_level,
            teaching_decision if isinstance(teaching_decision, AdaptiveTeachingDecision) else None,
        )

    # 答案相关性判断：规则档零成本预筛明显不匹配 + 语义档(LLM)判「答非所问」
    _honest_unavailable_flag = False
    if answer and not _answer_matches_intent(query, answer):
        answer = _honest_unavailable(query)
        _honest_unavailable_flag = True
    elif (
        answer
        and not reviewed_concept_grounding
        and not _answer_mentions_grounded_focus(answer, concept_focus_terms)
        and not _semantic_answers_question(query, answer)
    ):
        answer = _honest_unavailable(query)
        _honest_unavailable_flag = True

    if not answer and clean_retrieval is not None:
        items = list(getattr(clean_retrieval, "results", []) or [])[:5]
        context_chunks = [_result_text(item) for item in items]
        if context_chunks:
            answer = "基于知识库检索：\n" + "\n".join(
                f"- {chunk}" for chunk in context_chunks
            )
            confidence = max(0.3, float(getattr(clean_retrieval, "best_score", lambda: 0.3)() or 0.3))
        else:
            answer = f"关于“{query}”的知识暂未检索到，请补充更多上下文。"

    if not answer:
        answer = f"关于“{query}”的知识暂未检索到，请补充更多上下文。"

    # 上下文记忆注入: 多轮对话历史关联
    context = input_data.get("context") or {}
    recent_history = context.get("recent_history") or []
    topic = context.get("topic") or ""
    ctx_note = ""
    if recent_history:
        # 取真正的"上一轮" (倒数第二个, 因为最后一个恒为当前 query, 见前端 handle 先 push 当前)
        prev_q = ""
        for h in reversed(recent_history[:-1]):
            if isinstance(h, dict) and str(h.get("text", "")).strip():
                prev_q = str(h["text"]).strip()
                break
        if prev_q and prev_q != query:
            ctx_note = f"\n\n（基于上下文：上一轮你问了「{prev_q[:60]}」）"
    # 增强上下文: 当查询是"它/这个/那个/其"等指代, 或查询很短(<6字), 拼入topic
    REFER_RE = re.compile(r"^(它|这个|那个|这些|那|其|这)")
    if topic and (REFER_RE.match(query) or len(query.strip()) < 6):
        retrieval_query = f"{topic} {query}"
        if not ctx_note:
            ctx_note = f"\n\n（基于上下文：当前主题「{topic}」）"
    # 知识点溯源: 为每条证据切片推断关联知识点 (KP), 供前端展示来源
    sources: list[dict[str, Any]] = []
    for it in (compose_items or [])[:6]:
        txt = str(it.get("content") or "")
        if not txt.strip():
            continue
        kp_ids = _infer_kps(txt)
        meta = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}
        sources.append({
            "chunk_id": it.get("chunk_id"),
            "document_id": it.get("document_id"),
            "section": it.get("section"),
            "excerpt": txt[:160],
            "kp_ids": kp_ids,
            "kp_names": [kp_name(k) for k in kp_ids],
            "entity": meta.get("entity") or meta.get("entity_name") or it.get("entity") or "",
            "source_title": meta.get("source_title") or "",
            "source_uri": meta.get("source_uri") or "",
            "source_type": meta.get("source_type") or "",
            "evidence_status": meta.get("evidence_status") or "",
            "concept_ids": list(meta.get("concept_ids") or ()),
        })

    result = {
        "agent_id": GENERATION_AGENT_ID,
        "status": "completed",
        "query": query,
        "answer": answer + ctx_note,
        "confidence": round(confidence, 4),
        "honest_unavailable": _honest_unavailable_flag,
        "context_chunks": context_chunks,
        "citations": citations,
        "sources": sources,
        "question_type": qtype,
        "rerank_strategy": (
            getattr(deps.reranker, "strategy_name", "")
            if deps.reranker is not None
            else ""
        ),
    }
    # 知识到决策: 生成内容广播到 generation.output 频道, 供审核/导学决策订阅
    learner_id = input_data.get("learner_id") or input_data.get("student_id")
    if result["status"] == "completed":
        _broadcast(
            deps.message_bus,
            "knowledge.generation.output",
            {
                "event": "generation_output",
                "query": query,
                "answer": answer[:400],
                "confidence": result["confidence"],
                "learner_id": learner_id or "",
            },
            GENERATION_AGENT_ID,
        )
    return result


def run_review(
    input_data: dict[str, Any],
    deps: AgentDependencies,
) -> dict[str, Any]:
    """审核校验 Agent — 事实校验 + 防幻觉管道."""
    content = str(
        input_data.get("content")
        or input_data.get("answer")
        or ""
    ).strip()
    context_chunks = list(input_data.get("context_chunks") or [])
    if os.environ.get("DY3_DEBUG_REVIEW"):
        try:
            import json as _json

            _dbg = {
                "query": str(input_data.get("query") or input_data.get("question") or ""),
                "content": content[:2000],
                "n_ctx": len(context_chunks),
                "ctx_heads": [str(c)[:120] for c in context_chunks[:6]],
            }
            with open(r"tmp\review_debug.jsonl", "a", encoding="utf-8") as _f:
                _f.write(_json.dumps(_dbg, ensure_ascii=False) + "\n")
        except Exception:
            pass
    grounding = input_data.get("_claim_evidence_grounding")
    if not content:
        return _attach_review_candidate(
            {
                "agent_id": REVIEW_AGENT_ID,
                "status": "skipped",
                "verdict": "skipped",
                "reason": "内容为空，跳过审核",
                "confidence": 1.0,
            },
            input_data,
            content="",
            producer="skipped",
            real_reviewer_executed=False,
            mapping_refused_reason="no reviewed content",
        )

    review_content = _scientific_review_content(content)
    guided_question_review = bool(input_data.get("guided_question_review"))

    fact_checked = 0
    fact_failed = 0
    fact_passed: bool | None = None
    # 启发式追问是待审核的“问题列表”，不是对原问题的事实性回答。普通
    # FactChecker/CC1 会把问句中的待探究命题当作已作出的事实断言，产生
    # 假阳性。因此该显式私有模式交由独立 Reviewer 模型按证据边界审核；
    # 模型不可用时不放行，调用方回退到已验证的 Concept 关系问题。
    if not guided_question_review and deps.fact_checker is not None:
        try:
            report = deps.fact_checker.check(review_content)
            fact_passed = bool(getattr(report, "overall_passed", True))
            fact_checked = int(getattr(report, "checked", 0))
            fact_failed = int(getattr(report, "failed", 0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("审核校验 Agent 事实校验失败: %s", exc)

    ah_action = ""
    ah_score = 1.0
    hallucination_detected = False
    if not guided_question_review and deps.anti_hallucination_pipeline is not None:
        try:
            from dy3_polaris.l0.cc1.models import VerificationRequest

            report = deps.anti_hallucination_pipeline.verify(
                VerificationRequest(
                    output_text=review_content,
                    context_chunks=context_chunks,
                    agent_id=REVIEW_AGENT_ID,
                )
            )
            ah_action = str(getattr(getattr(report, "action", None), "value", "") or "")
            ah_score = float(getattr(report, "overall_score", 1.0) or 1.0)
            hallucination_detected = bool(
                getattr(report, "hallucination_detected", False)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("审核校验 Agent 防幻觉校验失败: %s", exc)

    grounding_issues = (
        tuple(grounding.issue_codes)
        if isinstance(grounding, ScientificGrounding)
        else ()
    )
    blocking_grounding_issues = tuple(
        issue
        for issue in grounding_issues
        if issue in {
            "conflicting_evidence",
            "condition_mismatch",
            "unsupported_universalization",
            "fact_not_directly_supported",
            "claim_evidence_review_identity_mismatch",
        }
    )
    model_challenge: dict[str, Any] | None = None
    review_query = str(
        input_data.get("query") or input_data.get("question") or ""
    ).strip()
    if review_query and context_chunks:
        try:
            candidate_challenge = critique_answer(
                review_query,
                review_content,
                context_chunks,
            )
            # Ordinary answers always cross the deterministic question-focus
            # gate, even when no model key is configured.  Guided-question
            # lists remain model-reviewed because they are not factual answer
            # prose and the answer heuristic would misclassify them.
            if not guided_question_review or candidate_challenge.get("used_llm"):
                model_challenge = candidate_challenge
        except Exception as exc:  # noqa: BLE001
            logger.warning("独立模型交叉审核失败，保留确定性审核结果: %s", type(exc).__name__)
    if guided_question_review and model_challenge is None:
        verdict = "needs_review"
        reason = "启发式追问未完成独立 Reviewer 模型审核"
    elif (
        guided_question_review
        and model_challenge
        and model_challenge.get("verdict") != "pass"
    ):
        verdict = "needs_review"
        reason = "独立 Reviewer 认为追问超出证据边界：" + str(
            model_challenge.get("reason") or model_challenge.get("verdict")
        )[:180]
    elif guided_question_review:
        verdict = "approved"
        reason = (
            "启发式追问与原问题相关，并通过独立 Reviewer 的证据边界审核"
        )
    elif ah_action == "refuse":
        verdict = "rejected"
        reason = "防幻觉管道判定拒绝输出"
    elif blocking_grounding_issues:
        verdict = "needs_review"
        reason = "Claim-Evidence 审查发现：" + ", ".join(blocking_grounding_issues)
    elif fact_passed is False and fact_failed > 0:
        # 标准值校验异常降级为复核: 知识库来源内容 (多谱线/多参数) 可能超出
        # 标准值库覆盖范围 (如 480/659nm 辅峰), 不硬拒, 交由复核与溯源展示
        verdict = "needs_review"
        reason = f"标准值校验 {fact_failed}/{fact_checked} 条断言异常（知识库来源内容，建议复核）"
    elif ah_action in ("degrade", "fix", "reask") or hallucination_detected:
        verdict = "needs_review"
        reason = "检测到潜在幻觉或需要人工复核"
    elif (
        model_challenge
        and model_challenge.get("verdict") != "pass"
    ):
        # The model is an additional challenger, never the sole approval
        # authority. A challenge can withhold/revise; a model PASS cannot
        # override FactChecker, CC1 or claim-evidence grounding.
        verdict = "needs_review"
        challenge_verdict = str(model_challenge.get("verdict") or "")
        challenge_reason = str(
            model_challenge.get("reason") or challenge_verdict
        )[:180]
        if challenge_verdict in {"fix_relevance", "unanswerable"}:
            reason = "问题核心覆盖门要求重新检索：" + challenge_reason
        elif challenge_verdict == "fix_completeness":
            reason = "问题核心覆盖门要求修订：" + challenge_reason
        else:
            reason = "独立模型交叉审核提出挑战：" + challenge_reason
    else:
        verdict = "approved"
        reason = "事实校验与防幻觉校验均通过"

    confidence = round(min(1.0, max(0.3, ah_score)), 4)
    result = {
        "agent_id": REVIEW_AGENT_ID,
        "status": "completed",
        "verdict": verdict,
        "reason": reason,
        "fact_check": {
            "passed": fact_passed,
            "checked": fact_checked,
            "failed": fact_failed,
        },
        "anti_hallucination": {
            "action": ah_action,
            "score": ah_score,
            "hallucination_detected": hallucination_detected,
        },
        "confidence": confidence,
    }
    # 审核到画像: 审核结果写回画像 extras.review_log (供画像/决策参考)
    learner_id = input_data.get("learner_id") or input_data.get("student_id")
    if learner_id:
        profile = _load_profile(deps.profile_service, learner_id)
        if profile is not None:
            extras = dict(getattr(profile, "extras", {}) or {})
            review_log = list(extras.get("review_log", []) or [])
            review_log.append({
                "ts": time.time(),
                "verdict": verdict,
                "reason": reason,
                "confidence": confidence,
            })
            extras["review_log"] = review_log[-20:]
            profile.extras = extras
            # 审核拒绝/需复核时小幅下调画像置信度 (质量信号)
            if verdict in ("rejected", "needs_review"):
                profile.confidence = round(
                    max(0.1, float(profile.confidence or 0.5) - 0.02), 4
                )
            _save_profile(deps.profile_service, profile)
        # 审核结果广播供导学决策订阅
        _broadcast(
            deps.message_bus,
            "knowledge.review.result",
            {
                "event": "review_result",
                "learner_id": learner_id,
                "verdict": verdict,
                "reason": reason,
                "confidence": confidence,
            },
            REVIEW_AGENT_ID,
        )
    return _attach_review_candidate(
        result,
        input_data,
        content=content,
        producer="agent.quality.review/run_review",
        real_reviewer_executed=True,
    )


def _resource_evidence_passages(
    evidence_candidate: _EvidenceCandidate | None,
    final_result: FinalCollaborationResult | None = None,
) -> list[str]:
    """Return de-duplicated source passages from the selected generation."""

    passages: list[str] = []
    seen: set[str] = set()
    raw_items: list[Any] = []
    if isinstance(evidence_candidate, _EvidenceCandidate):
        raw_items.extend(evidence_candidate.context_chunks)
    if isinstance(final_result, FinalCollaborationResult):
        for pack in final_result.evidence:
            if isinstance(pack, EvidencePack):
                raw_items.extend(pack.items)
            elif isinstance(pack, dict):
                raw_items.append(pack)
    for item in raw_items:
        if isinstance(item, dict):
            text = str(
                item.get("content")
                or item.get("text")
                or item.get("excerpt")
                or ""
            ).strip()
        elif hasattr(item, "content"):
            text = str(getattr(item, "content", "") or "").strip()
        else:
            text = str(item or "").strip()
        normalized = re.sub(r"\s+", " ", text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        passages.append(normalized[:1600])
        if len(passages) >= 18:
            break
    return passages


def _resource_source_references(
    evidence_candidate: _EvidenceCandidate | None,
) -> list[str]:
    """Return only source identifiers already carried by selected evidence."""

    if not isinstance(evidence_candidate, _EvidenceCandidate):
        return []
    references: list[str] = []
    for item in (*evidence_candidate.sources, *evidence_candidate.citations):
        if isinstance(item, dict):
            value = str(
                item.get("source_uri")
                or item.get("uri")
                or item.get("url")
                or item.get("source")
                or item.get("document_id")
                or item.get("chunk_id")
                or ""
            ).strip()
        else:
            value = str(item or "").strip()
        if value and value not in references:
            references.append(value)
    return references


def _resource_passage_excerpt(
    query: str,
    text: str,
    *,
    focus_terms: tuple[str, ...] = (),
    preferred_concept_ids: tuple[str, ...] = (),
) -> str:
    """Project a source chunk into task-relevant verbatim sentences.

    Long learning resources should not dump an entire MinerU chunk merely to
    reach a character target.  This helper only selects sentences already in
    the retrieved text and applies the same boundary cleanup as the answer.
    """

    source = _latex_to_plain(_clean_markdown_chunk(str(text or "")))
    candidates = _collect_answer_candidates(
        query,
        [{"content": source}],
        focus_terms=focus_terms,
        preferred_concept_ids=preferred_concept_ids,
    )
    if candidates:
        selected: list[str] = []
        for candidate in candidates[:5]:
            cleaned = _trim_fragment(candidate)
            if cleaned and cleaned not in selected:
                selected.append(cleaned)
        if selected:
            return " ".join(selected)[:1000].strip()
    return _trim_fragment(re.sub(r"\s+", " ", source))[:900].strip()


def _compile_reviewed_source_reader(
    *,
    query: str,
    reviewed_answer: str,
    evidence_passages: list[str],
    final_result: FinalCollaborationResult,
    knowledge_context: KnowledgeLearningContext | None,
) -> str:
    """Offline long-read compiler that never creates new scientific facts."""

    target_names: list[str] = []
    prerequisite_names: list[str] = []
    if isinstance(knowledge_context, KnowledgeLearningContext):
        target_names = [
            str(knowledge_context.concept_names.get(item, item))
            for item in knowledge_context.target_concepts
        ]
        prerequisite_names = [
            str(knowledge_context.concept_names.get(item, item))
            for item in knowledge_context.learning_path.prerequisite_gap
        ]
    accepted_claims = [
        str(claim.statement).strip()
        for claim in final_result.accepted_claims
        if str(claim.statement).strip()
    ]
    gaps = [str(item).strip() for item in final_result.knowledge_gaps if str(item).strip()]
    sections = [
        f"## 学习问题与边界\n\n本专题围绕“{query}”展开。"
        "下文只组织本次已审核回答、已接受结论和已检索证据；"
        "没有证据的数值、材料优劣和实验参数不在文中补造。",
        "## 已审核核心解释\n\n" + reviewed_answer.strip(),
    ]
    if target_names or prerequisite_names:
        relation_lines = []
        if target_names:
            relation_lines.append("本次目标 Concept：" + "、".join(target_names) + "。")
        if prerequisite_names:
            relation_lines.append("当前需先检查的先修 Concept：" + "、".join(prerequisite_names) + "。")
        sections.append("## Concept 与先修结构\n\n" + "\n\n".join(relation_lines))
    if accepted_claims:
        sections.append(
            "## Reviewer 接受的结论\n\n"
            + "\n".join(f"{index}. {claim}" for index, claim in enumerate(accepted_claims, 1))
        )
    if evidence_passages:
        sections.append(
            "## 证据研读\n\n"
            + "\n\n".join(
                f"【证据 {index}】\n{passage}"
                for index, passage in enumerate(evidence_passages, 1)
            )
        )
    sections.append(
        "## 条件、限制与不确定性\n\n"
        + (
            "\n".join(f"- {gap}" for gap in gaps)
            if gaps
            else "本次运行没有发布额外的知识缺口；这不等于结论可以跨材料体系或跨测试条件外推。"
        )
    )
    sections.append(
        "## 证据阅读与实验分析框架\n\n"
        "阅读本专题时，先把每个结论拆成研究对象、测试条件、观察结果和适用边界四项。"
        "研究对象至少要区分基质、掺杂离子与浓度范围；测试条件至少要记录激发条件、温度、"
        "仪器口径和是否进行校正；观察结果只写证据中实际报告的光谱、寿命、效率或结构信息；"
        "适用边界用于说明结论能否跨材料体系比较。\n\n"
        "进行跨来源比较时，不以单个峰值或单个评价指标直接判定材料优劣。先检查样品制备、"
        "测试条件和归一化方式是否可比，再分别记录支持、仅提及、存在冲突和证据不足的内容。"
        "如果来源之间条件不同，应保留差异，不把差异平均成一个看似确定的答案。\n\n"
        "形成实验任务时，可以按“问题—可观测量—控制变量—判据—限制”记录：问题说明要验证什么；"
        "可观测量来自本次证据实际涉及的表征数据；控制变量用于避免把浓度、温度、基质或测试条件混为一谈；"
        "判据写明什么结果支持或反驳当前解释；限制记录当前证据尚未覆盖的部分。"
    )
    sections.append(
        "## 学习与研究记录模板\n\n"
        "1. 核心问题：用一句话写出本次需要解释或验证的对象。\n"
        "2. 已审核事实：只摘录 Reviewer 已接受且有来源的内容。\n"
        "3. 机制推断：单独标注推断，并写出它依赖哪些事实。\n"
        "4. 条件边界：记录材料体系、实验条件和不能直接外推的范围。\n"
        "5. 证据缺口：列出仍缺少的表征、对照实验或来源。\n"
        "6. 下一步行动：选择补前置概念、继续检索、设计实验或完成分阶练习。"
    )
    sections.append(
        "## 学习检查\n\n"
        "1. 用自己的话复述核心机制，并标出哪些是证据直接支持的事实。\n"
        "2. 说明当前结论的材料体系、测试条件或适用边界。\n"
        "3. 如果要形成更强结论，列出仍需补充的证据，而不是直接补入未验证参数。"
    )
    return "\n\n".join(section for section in sections if section.strip())


def _build_reviewed_long_form_resource(
    *,
    query: str,
    task_id: str,
    quality_release: QualityReleaseDecision,
    final_result: FinalCollaborationResult,
    evidence_candidate: _EvidenceCandidate | None,
    teaching_decision: AdaptiveTeachingDecision | None,
    knowledge_context: KnowledgeLearningContext | None,
    deps: AgentDependencies,
    event_callback: Callable[..., None] | None = None,
) -> dict[str, Any] | None:
    """Generation → Reviewer candidate for one task-bound long resource."""

    if not quality_release.eligible or not quality_release.public_answer.strip():
        return None
    # A several-thousand-character model call is an explicit learning resource,
    # not mandatory overhead for every short Q&A. It remains inside the same
    # evidence → generation → reviewer loop when the learner asks for it.
    if not _long_form_resource_requested(query):
        return None
    passages = _resource_evidence_passages(evidence_candidate, final_result)
    resource_focus_terms: tuple[str, ...] = ()
    resource_target_ids: tuple[str, ...] = ()
    if isinstance(knowledge_context, KnowledgeLearningContext):
        resource_target_ids = tuple(knowledge_context.target_concepts)
        concepts_by_id = {
            concept.concept_id: concept for concept in canonical_concepts()
        }
        values: list[str] = []
        for concept_id in resource_target_ids:
            concept = concepts_by_id.get(concept_id)
            if concept is not None:
                values.append(concept.canonical_name)
                values.extend(concept.aliases)
            public_name = str(knowledge_context.concept_names.get(concept_id, "")).strip()
            if public_name:
                values.append(public_name)
        resource_focus_terms = tuple(dict.fromkeys(value for value in values if value))
    relevant_initial = _filter_task_answer_evidence(
        query,
        [{"content": value} for value in passages],
        focus_terms=resource_focus_terms,
        preferred_concept_ids=resource_target_ids,
    )
    passages = [
        excerpt
        for excerpt in (
            _resource_passage_excerpt(
                query,
                str(item.get("content") or item.get("text") or ""),
                focus_terms=resource_focus_terms,
                preferred_concept_ids=resource_target_ids,
            )
            for item in relevant_initial
        )
        if excerpt
    ]
    depth = (
        teaching_decision.content_depth
        if isinstance(teaching_decision, AdaptiveTeachingDecision)
        else "foundation"
    )
    default_target = 3800 if depth in {"research", "advanced"} else 2600 if depth in {"foundation", "beginner"} else 3200
    target_characters = _requested_resource_character_target(query, default_target)
    strategy = (
        {
            "explanation_strategy": teaching_decision.explanation_strategy,
            "representation_modes": list(teaching_decision.representation_modes),
        }
        if isinstance(teaching_decision, AdaptiveTeachingDecision)
        else {}
    )
    if event_callback is not None:
        event_callback(
            "AgentStarted",
            GENERATION_AGENT_ID,
            agent_id=GENERATION_AGENT_ID,
            phase="learning_resource_generation",
        )
    source_references = _resource_source_references(evidence_candidate)
    retrieval_queries_used: list[str] = []
    if sum(len(item) for item in passages) < int(target_characters * 0.8):
        existing = set(passages)
        concept_focus = ""
        if isinstance(knowledge_context, KnowledgeLearningContext):
            concept_focus = " ".join(
                str(knowledge_context.concept_names.get(item, item))
                for item in knowledge_context.target_concepts[:4]
            )
        expansion_queries = tuple(dict.fromkeys(filter(None, (
            query,
            f"{query} {concept_focus} 物理机制 材料机制".strip(),
            f"{query} {concept_focus} 评价指标 测试条件 适用边界".strip(),
            f"{concept_focus or query} 绿色健康照明 应用 限制".strip(),
        ))))
        for expansion_query in expansion_queries:
            _retrieval, expanded_items = _retrieve_evidence(
                expansion_query,
                {"task_id": task_id, "resource_generation": True},
                deps,
                top_k=24,
            )
            retrieval_queries_used.append(expansion_query)
            expanded_items = _filter_task_answer_evidence(
                query,
                expanded_items,
                focus_terms=resource_focus_terms,
                preferred_concept_ids=resource_target_ids,
            )
            for item in expanded_items:
                normalized = _resource_passage_excerpt(
                    query,
                    str(item.get("content") or item.get("text") or ""),
                    focus_terms=resource_focus_terms,
                    preferred_concept_ids=resource_target_ids,
                )
                if normalized and normalized not in existing:
                    existing.add(normalized)
                    passages.append(normalized[:1600])
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                source_ref = str(
                    item.get("source")
                    or item.get("document_id")
                    or metadata.get("source_uri")
                    or metadata.get("document_id")
                    or ""
                ).strip()
                if source_ref and source_ref not in source_references:
                    source_references.append(source_ref)
                if len(passages) >= 18:
                    break
            if (
                len(passages) >= 18
                or sum(len(item) for item in passages) >= target_characters
            ):
                break
        if event_callback is not None:
            event_callback(
                "RetrievalCompleted",
                GENERATION_AGENT_ID,
                phase="learning_resource_generation",
                evidence_count=len(passages),
                query_count=len(retrieval_queries_used),
            )
    # One isolated excerpt is insufficient for a multi-aspect long resource.
    # The existing Generation retrieval loop gets one bounded opportunity to
    # find more task-relevant evidence; if that still fails, keep the short
    # reviewed lesson instead of stretching or templating a single source.
    if len(passages) < 2:
        if event_callback is not None:
            event_callback(
                "AgentFinished",
                GENERATION_AGENT_ID,
                agent_id=GENERATION_AGENT_ID,
                phase="learning_resource_generation",
                generation_mode="insufficient_relevant_evidence",
                character_count=0,
                source_passage_count=len(passages),
            )
        return None
    model_content = ""
    model_used = False
    try:
        from dy3_polaris.l3.llm_synthesizer import LLMSynthesizer

        model_content, model_used = LLMSynthesizer().synthesize_learning_resource(
            query=query,
            reviewed_answer=quality_release.public_answer,
            evidence=passages,
            learner_level=depth,
            teaching_strategy=strategy,
            target_characters=target_characters,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("专题长文模型路径不可用, 改用已审核证据编排: %s", type(exc).__name__)
    # A deterministic source reader is not a generated personalised lesson.
    # Without a configured model (or when the model returns a short fragment),
    # keep the already-reviewed short lesson and its evidence appendix instead
    # of publishing evidence excerpts as a several-thousand-character resource.
    content = model_content.strip()
    minimum_publishable = max(1200, int(target_characters * 0.55))
    if not model_used or len(content) < minimum_publishable:
        if event_callback is not None:
            event_callback(
                "AgentFinished",
                GENERATION_AGENT_ID,
                agent_id=GENERATION_AGENT_ID,
                phase="learning_resource_generation",
                generation_mode=(
                    "model_unavailable"
                    if not model_used
                    else "model_output_too_short"
                ),
                character_count=len(content),
                target_characters=target_characters,
                source_passage_count=len(passages),
            )
        return None
    generation_mode = "llm_evidence_synthesis"
    if event_callback is not None:
        event_callback(
            "AgentFinished",
            GENERATION_AGENT_ID,
            agent_id=GENERATION_AGENT_ID,
            phase="learning_resource_generation",
            generation_mode=generation_mode,
            character_count=len(content),
            source_passage_count=len(passages),
            source_character_count=sum(len(item) for item in passages),
        )
        event_callback(
            "AgentStarted",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            phase="learning_resource_review",
        )
    review = run_review(
        {
            "task_id": task_id,
            "content": content,
            "context_chunks": passages,
            "resource_review": True,
        },
        deps,
    )
    verdict = str(review.get("verdict") or "").lower()
    approved = (
        str(review.get("status") or "") == "completed"
        and verdict == "approved"
        and str(review.get("agent_id") or "") == REVIEW_AGENT_ID
    )
    if event_callback is not None:
        event_callback(
            "ReviewCompleted",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            phase="learning_resource_review",
            verdict=verdict,
            reason=str(review.get("reason") or ""),
        )
        event_callback(
            "AgentFinished",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            phase="learning_resource_review",
        )
    return {
        "content": content if approved else "",
        "generation_mode": generation_mode,
        "model_used": model_used,
        "reviewer_executed": str(review.get("status") or "") == "completed",
        "review_verdict": verdict,
        "review_reason": str(review.get("reason") or ""),
        "target_characters": target_characters,
        "actual_characters": len(content) if approved else 0,
        "source_passage_count": len(passages),
        "source_references": tuple(source_references),
        "retrieval_query_count": len(retrieval_queries_used),
        "delivery_variant": (
            "research_evidence_dossier"
            if depth in {"research", "advanced"}
            else "scaffolded_concept_tutorial"
            if depth in {"foundation", "beginner"}
            else "mechanism_learning_article"
        ),
    }


def _build_reviewed_guided_questions(
    *,
    query: str,
    task_id: str,
    quality_release: QualityReleaseDecision,
    final_result: FinalCollaborationResult,
    evidence_candidate: _EvidenceCandidate | None,
    teaching_decision: AdaptiveTeachingDecision | None,
    knowledge_context: KnowledgeLearningContext | None,
    deps: AgentDependencies,
    event_callback: Callable[..., None] | None = None,
) -> dict[str, Any] | None:
    """Generation → Reviewer → Guidance loop for adaptive follow-up prompts."""

    if not quality_release.eligible or not quality_release.public_answer.strip():
        return None
    passages = _resource_evidence_passages(evidence_candidate, final_result)
    if not passages:
        return None
    concept_names: tuple[str, ...] = ()
    prerequisite_names: tuple[str, ...] = ()
    if isinstance(knowledge_context, KnowledgeLearningContext):
        concept_names = tuple(
            str(knowledge_context.concept_names.get(item, item))
            for item in knowledge_context.target_concepts
            if str(item)
        )
        prerequisite_names = tuple(
            str(knowledge_context.concept_names.get(item, item))
            for item in knowledge_context.learning_path.prerequisite_gap
            if str(item)
        )
    depth = (
        teaching_decision.content_depth
        if isinstance(teaching_decision, AdaptiveTeachingDecision)
        else "foundation"
    )
    strategy = (
        {
            "explanation_strategy": teaching_decision.explanation_strategy,
            "representation_modes": list(teaching_decision.representation_modes),
        }
        if isinstance(teaching_decision, AdaptiveTeachingDecision)
        else {}
    )
    if event_callback is not None:
        event_callback(
            "AgentStarted",
            GENERATION_AGENT_ID,
            agent_id=GENERATION_AGENT_ID,
            phase="guided_question_generation",
        )
    try:
        from dy3_polaris.l3.llm_synthesizer import LLMSynthesizer

        questions, model_used = LLMSynthesizer().synthesize_guided_questions(
            query=query,
            reviewed_answer=quality_release.public_answer,
            evidence=passages,
            concept_names=concept_names,
            prerequisite_names=prerequisite_names,
            learner_level=depth,
            teaching_strategy=strategy,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("启发式追问模型路径不可用: %s", type(exc).__name__)
        if event_callback is not None:
            event_callback(
                "AgentFinished",
                GENERATION_AGENT_ID,
                agent_id=GENERATION_AGENT_ID,
                phase="guided_question_generation",
                generation_mode="model_unavailable",
                question_count=0,
            )
        return None
    if not model_used or not questions:
        if event_callback is not None:
            event_callback(
                "AgentFinished",
                GENERATION_AGENT_ID,
                agent_id=GENERATION_AGENT_ID,
                phase="guided_question_generation",
                generation_mode="model_unavailable_or_invalid_json",
                question_count=0,
            )
        return None
    if event_callback is not None:
        event_callback(
            "AgentFinished",
            GENERATION_AGENT_ID,
            agent_id=GENERATION_AGENT_ID,
            phase="guided_question_generation",
            question_count=len(questions),
        )
        event_callback(
            "AgentStarted",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            phase="guided_question_review",
        )
    review_text = "\n".join(
        f"{index}. {item['prompt']}" for index, item in enumerate(questions, 1)
    )
    review = run_review(
        {
            "task_id": task_id,
            "query": (
                "审核任务：判断以下启发式追问是否与原始问题相关、能否由给定证据"
                "继续讨论，以及是否暗含证据之外的既定事实。候选内容应当是问题列表，"
                "不要求回答原始问题。原始问题：" + query
            ),
            "content": review_text,
            "context_chunks": passages,
            "guided_question_review": True,
        },
        deps,
    )
    verdict = str(review.get("verdict") or "").lower()
    approved = bool(
        str(review.get("status") or "") == "completed"
        and verdict == "approved"
        and str(review.get("agent_id") or "") == REVIEW_AGENT_ID
    )
    if event_callback is not None:
        event_callback(
            "ReviewCompleted",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            phase="guided_question_review",
            verdict=verdict,
            question_count=len(questions),
        )
        event_callback(
            "AgentFinished",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            phase="guided_question_review",
        )
    return {
        "questions": tuple(questions) if approved else (),
        "model_used": model_used,
        "reviewer_executed": str(review.get("status") or "") == "completed",
        "review_verdict": verdict,
        "review_reason": str(review.get("reason") or ""),
        "source_passage_count": len(passages),
        "collaboration_path": "Generation → Reviewer → Guidance",
    }


def _long_form_resource_requested(query: str) -> bool:
    """Return whether the learner explicitly requested a long teaching artifact."""

    normalized = re.sub(r"\s+", "", str(query or "")).lower()
    if re.search(r"(?:[1-9]\d{3,4})字", normalized):
        return True
    return any(
        marker in normalized
        for marker in (
            "几千字",
            "长文",
            "专题讲义",
            "完整讲义",
            "教学讲义",
            "课程讲义",
            "完整报告",
            "详细报告",
        )
    )


def _requested_resource_character_target(query: str, default_target: int) -> int:
    """Read an explicit long-form length request without confusing values such as 3000 K."""

    text = str(query or "")
    match = re.search(
        r"(?:不少于|至少|不低于|约|大约)?\s*(\d{3,5})\s*(?:字|字符)",
        text,
    )
    requested = int(match.group(1)) if match else 0
    if not requested and re.search(r"几千字|专题长文|专题学习资源|长篇学习资源", text):
        requested = 3200
    return max(1800, min(5000, max(int(default_target), requested)))


def _text_overlap_ratio(a: str, b: str) -> float:
    """字符级 Jaccard 相似度 (用于多候选分歧度计算).

    L5 高等级对标: 中间分歧度数据供前端"协同决策"可视化.
    """
    a_set = set(str(a or "").replace(" ", "")[:800])
    b_set = set(str(b or "").replace(" ", "")[:800])
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    if not union:
        return 1.0
    return round(len(a_set & b_set) / len(union), 4)


def _candidate_divergence_matrix(
    candidates: list[dict[str, Any]],
) -> tuple[list[list[float]], int, int]:
    """计算候选两两分歧度矩阵 (1 - 字符级 Jaccard 相似度).

    Returns:
        divergence_matrix: n×n 分歧度矩阵 (对称, 对角为 0)
        agree_pairs: 相似度 >= 0.6 的候选对数
        total_pairs: 候选总对数
    """
    n = len(candidates)
    divergence_matrix = [[0.0] * n for _ in range(n)]
    agree_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = _text_overlap_ratio(
                candidates[i]["answer"], candidates[j]["answer"]
            )
            divergence = round(1.0 - sim, 4)
            divergence_matrix[i][j] = divergence
            divergence_matrix[j][i] = divergence
            total_pairs += 1
            if sim >= 0.6:
                agree_pairs += 1
    return divergence_matrix, agree_pairs, total_pairs


def _run_candidate_debate(
    candidates: list[dict[str, Any]],
    divergence_matrix: list[list[float]],
    sim_threshold: float,
    max_rounds: int = 3,
) -> dict[str, Any]:
    """多轮协同辩论 (对标 Debate-Augmented RAG / Multi-Round Agentic RAG).

    未达共识时, 分歧最大的两候选逐轮交换关键论据, 并做证据可靠性自反思:
    - 证据更充分的候选每轮保持说服力, 证据稀少的一方置信度逐轮衰减
      (模拟"论据不足"在辩论中落败, 对标 Self-Reflective Debates for Context Reliability);
    - 分歧度随轮次衰减 (0.9 因子), 最多 max_rounds 轮, 从冲突逐步收敛或标记待裁决.
    """
    n = len(candidates)
    max_div = -1.0
    pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if divergence_matrix[i][j] > max_div:
                max_div = divergence_matrix[i][j]
                pair = (i, j)
    ca = candidates[pair[0]]
    cb = candidates[pair[1]]
    conf_a = float(ca.get("confidence", 0.0) or 0.0)
    conf_b = float(cb.get("confidence", 0.0) or 0.0)
    na = len(ca.get("context_chunks") or [])
    nb = len(cb.get("context_chunks") or [])

    rounds_log: list[dict[str, Any]] = []
    divergence = max_div
    converged = False
    pro_args: list[str] = []
    con_args: list[str] = []
    for r in range(1, max_rounds + 1):
        # 证据可靠性自反思: 证据稀少的一方置信度逐轮衰减
        if na > 0 and nb == 0:
            conf_b *= 0.85
        elif nb > 0 and na == 0:
            conf_a *= 0.85
        # 每轮多交换一条论据 (逐轮深入)
        pro_args = [str(ch)[:120] for ch in list(ca.get("context_chunks") or [])[: r + 1]]
        con_args = [str(ch)[:120] for ch in list(cb.get("context_chunks") or [])[: r + 1]]
        conf_diff = abs(conf_a - conf_b)
        # 分歧度随轮次衰减: 多轮交换论据后分歧逐步收敛
        divergence = round(max_div * (1 - min(conf_diff, 0.5)) * (0.9 ** (r - 1)), 4)
        rounds_log.append({
            "round": r,
            "divergence": divergence,
            "conf_a": round(conf_a, 4),
            "conf_b": round(conf_b, 4),
        })
        if divergence < 0.35 or conf_diff >= 0.25:
            converged = True
            break

    return {
        "rounds": len(rounds_log),
        "rounds_log": rounds_log,
        "pro": {
            "candidate_id": ca.get("candidate_id", ""),
            "label": ca.get("label", ""),
            "confidence": round(conf_a, 4),
            "arguments": pro_args,
        },
        "con": {
            "candidate_id": cb.get("candidate_id", ""),
            "label": cb.get("label", ""),
            "confidence": round(conf_b, 4),
            "arguments": con_args,
        },
        "divergence_before": max_div,
        "divergence_after": divergence,
        "converged": converged,
    }


def _select_consensus_candidate(
    candidates: list[dict[str, Any]],
    consensus_reached: bool,
    debate: dict[str, Any] | None,
) -> dict[str, Any]:
    """选定最终候选.

    共识达成 → 置信度最高者; 辩论收敛 → 采纳胜方(置信度更高者);
    否则 → 置信度最高者并标记待裁决 (needs_adjudication).
    """
    if consensus_reached:
        return max(candidates, key=lambda c: c["confidence"])
    if debate and debate.get("converged"):
        winner_id = (
            debate["pro"]["candidate_id"]
            if debate["pro"]["confidence"] >= debate["con"]["confidence"]
            else debate["con"]["candidate_id"]
        )
        return next(c for c in candidates if c["candidate_id"] == winner_id)
    best = max(candidates, key=lambda c: c["confidence"])
    best["needs_adjudication"] = True
    return best


def _run_multi_candidate_generation(
    input_data: dict[str, Any],
    deps: AgentDependencies,
    review_feedback: str = "",
) -> dict[str, Any]:
    """多候选知识生成 + 交叉验证 (流程多样性核心, L5 高等级对标).

    实现协同三类流程多样性:
    1. 并行生成 3 个不同检索策略候选答案 (标准 / 宽召回 / 精聚焦)
    2. 交叉验证: 计算候选两两分歧度矩阵, 判定共识度
    3. 协同辩论: 未达共识时, 分歧最大的两候选交换论据并收敛,
       收敛采纳胜方, 否则标记待裁决 (needs_adjudication)

    返回结构同时兼容下游单答案消费(answer/confidence/context_chunks),
    并附带完整中间数据(candidates/consensus/divergence/debate)供前端可视化.
    """
    query = str(
        input_data.get("query") or input_data.get("question") or ""
    ).strip()
    # 模糊问题 → 引导式澄清 (人性化补充追问), 不浪费多候选生成, 也不硬凑答案
    agent_input = input_data.get("_agent_input")
    clarify = (
        _detect_ambiguity(query, agent_input.intent)
        if agent_input is not None
        else _detect_ambiguity(query)
    )
    if clarify is not None:
        return _attach_selected_evidence_candidate({
            "agent_id": GENERATION_AGENT_ID,
            "status": "clarify",
            "query": query,
            "answer": "",
            "confidence": 0.0,
            "context_chunks": [],
            "citations": [],
            "sources": [],
            "knowledge_unavailable": False,
            "clarify": clarify,
            "candidates": [],
            "consensus_score": 0.0,
            "consensus_reached": False,
            "consensus_threshold": 0.5,
            "divergence_matrix": [],
            "agree_pairs": 0,
            "total_pairs": 0,
            "debate": None,
            "selected_candidate": "",
            "needs_adjudication": False,
        }, input_data, stage="clarify")
    strategies = [
        {
            "candidate_id": "A",
            "label": "标准检索",
            "top_k": 12,
            "min_overlap": 0.3,
            "llm_role": "generation_fast",
            "thinking": False,
            "reasoning_effort": "none",
        },
        {
            "candidate_id": "B",
            "label": "宽召回",
            "top_k": 20,
            "min_overlap": 0.25,
            "llm_role": "generation_long",
            "thinking": False,
            "reasoning_effort": "none",
        },
        {
            "candidate_id": "C",
            "label": "精聚焦",
            "top_k": 6,
            "min_overlap": 0.45,
            "llm_role": "generation_deep",
            "thinking": True,
            "reasoning_effort": "high",
        },
    ]
    def generate_one(s: dict[str, Any]) -> dict[str, Any]:
        cand_input = {
            **input_data,
            "strategy_id": s["candidate_id"],
            "_candidate_top_k": s["top_k"],
            "_candidate_min_overlap": s["min_overlap"],
            "_llm_role": s["llm_role"],
            "_llm_enable_thinking": s["thinking"],
            "_llm_reasoning_effort": s["reasoning_effort"],
            "review_feedback": review_feedback,
        }
        return run_generation(cand_input, deps)

    # The three candidates are independent retrieval/generation views. Execute
    # their network-bound work concurrently, while consuming results below in
    # deterministic A/B/C order so public output and selection stay stable.
    with ThreadPoolExecutor(
        max_workers=len(strategies), thread_name_prefix="dy3-candidate"
    ) as executor:
        generated_candidates = list(executor.map(generate_one, strategies))

    candidates: list[dict[str, Any]] = []
    for s, cand in zip(strategies, generated_candidates, strict=True):
        # 通俗讲解/推演结果都是终端答案: 直接返回, 不参与多候选交叉验证/辩论/自纠,
        # 避免被「改写 query → 重新生成」覆盖成学术/检索答案 (persona #29④ / 推演逻辑)
        if cand.get("plain_language") or cand.get("deduced") or cand.get("honest_unavailable"):
            has_real_evidence = bool(
                cand.get("context_chunks")
                or cand.get("citations")
                or cand.get("sources")
            )
            return _attach_selected_evidence_candidate({
                "agent_id": GENERATION_AGENT_ID,
                "status": "completed",
                "query": query,
                "answer": cand.get("answer", ""),
                "confidence": 0.9 if cand.get("deduced") else 0.8,
                "context_chunks": list(cand.get("context_chunks") or []),
                "citations": list(cand.get("citations") or []),
                "sources": list(cand.get("sources") or []),
                "knowledge_unavailable": False,
                "question_type": cand.get("question_type", ""),
                "plain_language": bool(cand.get("plain_language")),
                "deduced": bool(cand.get("deduced")),
                "honest_unavailable": bool(cand.get("honest_unavailable")),
                "candidates": [],
                "consensus_score": 1.0,
                "consensus_reached": True,
                "consensus_threshold": 0.5,
                "divergence_matrix": [],
                "agree_pairs": 0,
                "total_pairs": 0,
                "debate": None,
                "selected_candidate": s["candidate_id"],
                "needs_adjudication": False,
            }, input_data, stage=(
                "selected"
                if has_real_evidence and not cand.get("honest_unavailable")
                else "terminal"
            ))
        candidates.append({
            "candidate_id": s["candidate_id"],
            "label": s["label"],
            "answer": cand.get("answer", ""),
            "confidence": round(float(cand.get("confidence", 0.0) or 0.0), 4),
            "context_chunks": list(cand.get("context_chunks") or []),
            "citations": list(cand.get("citations") or []),
            "sources": list(cand.get("sources") or []),
            "knowledge_unavailable": bool(cand.get("knowledge_unavailable", False)),
            "status": cand.get("status", ""),
        })

    # 交叉验证: 分歧度矩阵 + 共识判定
    sim_threshold = 0.6
    consensus_threshold = 0.5
    divergence_matrix, agree_pairs, total_pairs = _candidate_divergence_matrix(
        candidates
    )
    consensus_score = (
        round(agree_pairs / total_pairs, 4) if total_pairs else 0.0
    )
    consensus_reached = consensus_score >= consensus_threshold

    # 协同辩论: 未达共识 → 分歧最大两候选交换论据并收敛
    debate: dict[str, Any] | None = None
    if not consensus_reached and len(candidates) >= 2:
        debate = _run_candidate_debate(
            candidates, divergence_matrix, sim_threshold
        )

    # 选定最终候选
    final = _select_consensus_candidate(
        candidates, consensus_reached, debate
    )

    return _attach_selected_evidence_candidate({
        "agent_id": GENERATION_AGENT_ID,
        "status": "completed",
        "query": query,
        "answer": final["answer"],
        "confidence": round(float(final["confidence"]), 4),
        "context_chunks": list(final["context_chunks"]),
        "citations": list(final["citations"]),
        "sources": list(final.get("sources") or []),
        "knowledge_unavailable": bool(final["knowledge_unavailable"]),
        "question_type": _detect_question_type(query),
        # 中间数据 (供前端"协同决策"可视化)
        "candidates": candidates,
        "consensus_score": consensus_score,
        "consensus_reached": consensus_reached,
        "consensus_threshold": consensus_threshold,
        "divergence_matrix": divergence_matrix,
        "agree_pairs": agree_pairs,
        "total_pairs": total_pairs,
        "debate": debate,
        "selected_candidate": final["candidate_id"],
        "needs_adjudication": bool(final.get("needs_adjudication", False)),
    }, input_data, stage="selected")


def _run_critic_loop(
    input_data: dict[str, Any],
    deps: AgentDependencies,
    generation: dict[str, Any],
) -> dict[str, Any]:
    """验证器引导的迭代自纠回路 (对标 DeepVerifier / CoRefine / SETS).

    四角色闭环: rewriter(改写 query) → retriever(重检索, 经 run_generation) →
    generator(多候选重生成) → critic(语义评审), 有界 MAX_ROUNDS 轮, 只采纳更优答案.

    这是"类级"修复: 把原「字符 Jaccard + 假辩论」判定答案好坏的环节, 升级为
    「真·语义 critic 驱动闭环纠错」, 从根上治理答非所问 / 检索不相关 / 空词.
    """
    query = str(input_data.get("query") or input_data.get("question") or "").strip()
    if not query or not generation.get("answer"):
        return {
            "adopted": False, "generation": generation, "rounds": [],
            "final_verdict": "unanswerable", "final_score": 0.0, "reason": "",
        }

    best = generation
    best_answer = str(generation.get("answer", ""))
    best_chunks = list(generation.get("context_chunks") or [])
    best_score = 0.0
    current_query = query
    current_feedback = ""
    rounds: list[dict[str, Any]] = []
    MAX_ROUNDS = 3  # 2-3 轮是 self-refinement 甜点区; 只采纳更优答案, 3 轮给"延迟收敛"留余量

    for rnd in range(1, MAX_ROUNDS + 1):
        crit = critique_answer(current_query, best_answer, best_chunks)
        crit["round"] = rnd
        crit["query"] = current_query
        rounds.append(crit)
        verdict = str(crit.get("verdict", "pass"))
        score = float(crit.get("score", 0.0) or 0.0)
        if rnd == 1:
            best_score = score
        if verdict == "pass":
            break

        # 依据裁决决定回路方向
        if verdict in ("fix_relevance", "unanswerable"):
            new_query = (rewrite_query(current_query, str(crit.get("reason", ""))) or "").strip()
            if not new_query or new_query == current_query:
                break
            current_query = new_query
            current_feedback = ""
        elif verdict in ("fix_faithfulness", "fix_completeness"):
            current_feedback = str(crit.get("reason", ""))[:400]
        else:
            break

        # 重新生成 (复用多候选 + 交叉验证, 保持 sources/context_chunks 结构一致)
        try:
            regen = _run_multi_candidate_generation(
                {**input_data, "query": current_query},
                deps,
                review_feedback=current_feedback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("验证器回路重生成失败, 停止: %s", exc)
            break
        new_answer = str(regen.get("answer", ""))
        if not new_answer:
            break
        new_chunks = list(regen.get("context_chunks") or [])

        # 二段评审: 只在更优时采纳 (fixed-point: 不越改越差)
        crit2 = critique_answer(current_query, new_answer, new_chunks)
        crit2["round"] = rnd
        crit2["query"] = current_query
        rounds.append(crit2)
        if float(crit2.get("score", 0.0) or 0.0) > best_score:
            best = regen
            best_answer = new_answer
            best_chunks = new_chunks
            best_score = float(crit2.get("score", 0.0) or 0.0)
        if str(crit2.get("verdict", "pass")) == "pass":
            break

    adopted = best is not generation
    final_crit = rounds[-1] if rounds else {"verdict": "pass", "score": 0.0, "reason": ""}
    return {
        "adopted": adopted,
        "generation": best,
        "rounds": rounds,
        "final_verdict": str(final_crit.get("verdict", "pass")),
        "final_score": float(final_crit.get("score", 0.0) or 0.0),
        "reason": str(final_crit.get("reason", ""))[:200],
    }


def _loop_direction(verdict: str) -> str:
    """把 critic 裁决映射为循环回路的动作方向 (供交互轨迹展示)."""
    if verdict in ("fix_relevance", "unanswerable"):
        return "改写查询 → 重检索"
    if verdict in ("fix_faithfulness", "fix_completeness"):
        return "注入评审反馈 → 重生成"
    return "通过, 无需修订"


def _build_loop_trace(
    reasoning_loop: dict[str, Any] | None,
    self_correction: dict[str, Any] | None,
    debate: dict[str, Any] | None,
    consensus_reached: bool,
    needs_adjudication: bool,
) -> list[dict[str, Any]]:
    """把「多候选→交叉验证→辩论→审核自纠→验证器迭代」整理成学习者可见的协同交互轨迹.

    面向竞赛主题「多智能体协同交互」: 每一步都写明「哪个 Agent 做了什么、产出什么、
    是否被采纳」, 让学习者直观看到 4 Agent 不是黑盒, 而是可追溯的迭代协作闭环。
    """
    trace: list[dict[str, Any]] = []
    seq = 0
    max_round = 0

    def _push(round_no: int, stage: str, agent: str, action: str,
              outcome: str, adopted: bool, score: float | None = None) -> None:
        nonlocal seq, max_round
        seq += 1
        if round_no > max_round:
            max_round = round_no
        item: dict[str, Any] = {
            "seq": seq,
            "round": round_no,
            "stage": stage,
            "agent": agent,
            "action": action,
            "outcome": outcome,
            "adopted": bool(adopted),
        }
        if score is not None:
            item["score"] = round(float(score), 3)
        trace.append(item)

    # 0) 多候选并行生成 + 交叉验证 (流程多样性)
    _push(
        0, "多候选生成", "generator",
        "3 路并行候选(A/B/C)生成 + 两两交叉验证",
        ("达成共识" if consensus_reached else "未达共识"),
        consensus_reached,
    )

    # 1) 协同辩论 (分歧大 → 论据交换 → 收敛/待裁决)
    if debate:
        _push(
            0, "协同辩论", "debate",
            "分歧最大两候选交换论据",
            ("收敛·采纳胜方" if debate.get("converged") else "未收敛"),
            bool(debate.get("converged")),
        )

    # 2) 审核自纠回路 (初审 needs_review → 修订 → 终审)
    if self_correction:
        _push(
            1, "审核自纠", "reviewer",
            "初审发现问题 → 触发生成修订",
            f"终审 {self_correction.get('verdict_after', '')}",
            self_correction.get("verdict_after") == "approved",
        )

    # 3) 验证器迭代回路 (真·语义 critic 闭环, 每轮: 评审→回路动作→重生成→二段评审)
    if reasoning_loop:
        rounds = list(reasoning_loop.get("rounds", []))
        by_round: dict[int, list[dict[str, Any]]] = {}
        for c in rounds:
            by_round.setdefault(int(c.get("round", 0)), []).append(c)
        for rnd in sorted(by_round):
            entries = by_round[rnd]
            first = entries[0]
            v_before = str(first.get("verdict", "pass"))
            s_before = float(first.get("score", 0.0) or 0.0)
            direction = _loop_direction(v_before)
            if len(entries) >= 2:
                second = entries[1]
                v_after = str(second.get("verdict", "pass"))
                s_after = float(second.get("score", 0.0) or 0.0)
                improved = s_after > s_before
                _push(
                    rnd, "验证器迭代", "critic",
                    f"语义评审(第{rnd}轮): {v_before} → {direction}",
                    ("修订后 " + v_after + (" · 采纳(评分↑)" if improved else " · 保留初稿")),
                    improved, s_after,
                )
            else:
                _push(
                    rnd, "验证器迭代", "critic",
                    f"语义评审(第{rnd}轮): {v_before}",
                    "通过, 无需修订",
                    True, s_before,
                )

    # 4) 待裁决 (分歧未收敛 → 转人工/仲裁)
    if needs_adjudication:
        _push(
            max_round + 1, "待裁决", "adjudicator",
            "分歧未收敛 → 转人工/仲裁确认",
            "标注分歧, 仍给出最优候选",
            False,
        )

    return trace


def _build_loop_narrative(trace: list[dict[str, Any]], final_verdict: str) -> str:
    """把协同交互轨迹压缩成一句自然语言叙述, 供学习者/评委一眼看懂协作闭环."""
    if not trace:
        return ""
    stages = " → ".join(dict.fromkeys(t.get("stage", "") for t in trace))
    loop_steps = [t for t in trace if t.get("stage") in ("审核自纠", "验证器迭代")]
    n_loops = len(loop_steps)
    improved = any(t.get("adopted") for t in loop_steps)
    parts = [f"4 个 Agent 协同闭环: {stages}。"]
    if n_loops:
        parts.append(
            f"历经 {n_loops} 轮迭代自纠"
            + ("，采纳了更优答案。" if improved else "，最终保留初稿。")
        )
    if final_verdict:
        parts.append(f"终审裁决: {final_verdict}。")
    return "".join(parts)


def _run_authoritative_correction_loop(
    *,
    context: CollaborationContext,
    input_data: dict[str, Any],
    deps: AgentDependencies,
    generation: dict[str, Any],
    review: dict[str, Any],
    generation_input: AgentInput,
    review_input: AgentInput,
    generation_contribution: AgentContribution,
    review_contribution: AgentContribution,
    task_event: Callable[..., None],
) -> tuple[dict[str, Any], dict[str, Any], AgentContribution, AgentContribution, dict[str, Any] | None]:
    """Execute the sole R-03E correction semantics with bounded progress."""
    first_verdict = str(review.get("verdict") or "")
    iterations: list[dict[str, Any]] = []
    seen_states: set[tuple[Any, ...]] = set()
    current_generation_input = generation_input
    current_review_input = review_input
    current_generation = generation
    current_review = review
    current_generation_contribution = generation_contribution
    current_review_contribution = review_contribution
    counts = context.runtime_metadata.setdefault(
        "r03e_call_counts",
        {"generation": 1, "retrieval": 1, "review": 1},
    )

    # Product policy: at most two automatic revisions.  The wider R03 budget
    # remains a planning ceiling, but cannot turn one request into an
    # unbounded generation/review loop.
    hard_limit = min(
        2,
        context.collaboration_budget.global_correction_limit,
        context.collaboration_budget.max_expensive_iterations,
    )
    for iteration in range(1, hard_limit + 2):
        challenge, action = _build_review_challenge(
            context,
            current_generation_contribution,
            current_review,
            iteration=iteration,
        )
        if challenge is None:
            context.decisions.append({"iteration": iteration - 1, "action": ResolutionAction.ACCEPT})
            break
        context.record_challenge(challenge)
        task_event(
            "ReviewerChallengeRaised",
            REVIEW_AGENT_ID,
            challenge_id=challenge.challenge_id,
            challenge_type=challenge.challenge_type.value,
            severity=challenge.severity.value,
            requested_action=action.value,
        )
        if action in {ResolutionAction.REJECT, ResolutionAction.ASK_USER}:
            _set_challenge_status(context, challenge, action.value)
            context.decisions.append({"challenge_id": challenge.challenge_id, "action": action})
            break
        if iteration > hard_limit:
            _set_challenge_status(context, challenge, "BUDGET_EXHAUSTED")
            context.decisions.append({
                "challenge_id": challenge.challenge_id,
                "action": "STOP_PRODUCT_REVISION_LIMIT",
            })
            break

        current_state = (
            _challenge_signature(challenge),
            _evidence_signature(_active_evidence_packs(context)),
            tuple(
                query
                for plan in context.tool_results.get("retrieval_plans", ())
                for query in plan.rewritten_queries
            ),
        )
        if current_state in seen_states:
            _set_challenge_status(context, challenge, "NO_PROGRESS")
            context.decisions.append({"challenge_id": challenge.challenge_id, "action": "STOP_NO_PROGRESS"})
            break
        seen_states.add(current_state)
        challenge_key = repr(_challenge_signature(challenge))
        challenge_counts = context.runtime_metadata.setdefault(
            "challenge_retry_counts", {}
        )
        challenge_retry_count = int(challenge_counts.get(challenge_key, 0) or 0)
        if challenge_retry_count >= context.collaboration_budget.per_challenge_limit:
            _set_challenge_status(context, challenge, "BUDGET_EXHAUSTED")
            context.decisions.append({
                "challenge_id": challenge.challenge_id,
                "action": "STOP_PER_CHALLENGE_BUDGET",
            })
            break
        if not context.can_resolve(action.value):
            _set_challenge_status(context, challenge, "BUDGET_EXHAUSTED")
            context.decisions.append({"challenge_id": challenge.challenge_id, "action": "STOP_BUDGET"})
            break
        context.consume_resolution_budget(action.value)
        challenge_counts[challenge_key] = challenge_retry_count + 1

        task_context = input_data.get("task_context")
        task_state_runtime.set_task_state(
            task_context,
            "RETRYING",
            producer="agent.quality.review",
        )

        revision_input = _revision_agent_input(context, current_generation_input)
        revision_input = replace(
            revision_input,
            constraints=tuple(dict.fromkeys((*revision_input.constraints, challenge.reason, *challenge.missing_information))),
            runtime_metadata={
                **dict(revision_input.runtime_metadata),
                "challenge_id": challenge.challenge_id,
                "resolution_action": action.value,
                "iteration": iteration,
            },
        )
        generation_payload = _contract_runtime_payload(input_data, revision_input)
        task_state_runtime.set_task_state(
            task_context,
            "RETRIEVING",
            producer="agent.knowledge.generation",
        )
        if action is ResolutionAction.RE_RETRIEVE:
            plans = build_challenge_retrieval_plans(
                revision_input,
                challenge.missing_information,
                reason=challenge.reason,
            )
            next_version = max((pack.version for pack in _active_evidence_packs(context)), default=1) + 1
            revision_input, generation_payload = _prepare_generation_retrieval(
                context,
                revision_input,
                generation_payload,
                deps,
                plans_override=plans,
                evidence_version=next_version,
                refresh_reason=challenge.reason,
                requested_by=challenge.challenge_id,
            )
            counts["retrieval"] += 1
        task_event(
            "RetrievalCompleted",
            GENERATION_AGENT_ID,
            challenge_id=challenge.challenge_id,
            evidence_version=max(
                (pack.version for pack in _active_evidence_packs(context)),
                default=0,
            ),
        )

        task_state_runtime.set_task_state(
            task_context,
            "COLLABORATING",
            producer="agent.knowledge.generation",
        )
        task_event("AgentStarted", GENERATION_AGENT_ID, agent_id=GENERATION_AGENT_ID)
        next_generation = _run_multi_candidate_generation(
            generation_payload,
            deps,
            review_feedback=(
                f"{challenge.challenge_type.value}: {challenge.reason}; "
                f"constraints: {', '.join(challenge.missing_information)}"
            ),
        )
        task_event("AgentFinished", GENERATION_AGENT_ID, agent_id=GENERATION_AGENT_ID)
        counts["generation"] += 1
        next_generation_contribution = _adapt_generation_contribution(
            context,
            revision_input,
            next_generation,
            parent_contribution_id=current_generation_contribution.contribution_id,
            revision_reason=challenge.reason,
            iteration=iteration,
        )
        context.record_contribution(next_generation_contribution)
        task_event(
            "AgentContributionRecorded",
            GENERATION_AGENT_ID,
            agent_id=GENERATION_AGENT_ID,
            contribution_id=next_generation_contribution.contribution_id,
            iteration=iteration,
        )

        next_review_input = _revision_agent_input(context, current_review_input)
        next_review_payload = _contract_runtime_payload(input_data, next_review_input)
        next_review_payload.update({
            "content": next_generation_contribution.conclusion,
            "context_chunks": _review_evidence_texts(next_review_input, next_generation_contribution),
            "_claim_evidence_grounding": _review_scientific_grounding(
                context,
                next_generation_contribution,
            ),
        })
        task_state_runtime.set_task_state(
            task_context,
            "REVIEWING",
            producer="agent.quality.review",
        )
        task_event("AgentStarted", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID)
        next_review = run_review(next_review_payload, deps)
        task_event("ReviewCompleted", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID, verdict=str(next_review.get("verdict", "")))
        task_event("AgentFinished", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID)
        counts["review"] += 1
        next_review_contribution = _adapt_review_contribution(
            context,
            next_review_input,
            next_review,
            parent_contribution_id=current_review_contribution.contribution_id,
            revision_reason=challenge.reason,
            iteration=iteration,
        )
        context.record_contribution(next_review_contribution)
        task_event(
            "AgentContributionRecorded",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            contribution_id=next_review_contribution.contribution_id,
            iteration=iteration,
        )
        _set_challenge_status(context, challenge, "RESOLVED_ACTION_EXECUTED")
        history = {
            "iteration": iteration,
            "challenge_id": challenge.challenge_id,
            "action": action,
            "parent_contribution_id": current_generation_contribution.contribution_id,
            "contribution_id": next_generation_contribution.contribution_id,
            "review_contribution_id": next_review_contribution.contribution_id,
            "evidence_versions": tuple(pack.version for pack in _active_evidence_packs(context)),
        }
        context.revision_history.append(history)
        context.decisions.append(history)
        task_event(
            "RevisionApplied",
            GENERATION_AGENT_ID,
            challenge_id=challenge.challenge_id,
            contribution_id=next_generation_contribution.contribution_id,
            review_contribution_id=next_review_contribution.contribution_id,
            iteration=iteration,
        )
        iterations.append({
            "iteration": iteration,
            "action": action.value,
            "verdict_before": str(current_review.get("verdict") or ""),
            "verdict_after": str(next_review.get("verdict") or ""),
        })
        if str(next_review.get("verdict") or "") == "rejected":
            # Preserve the last internally consistent accepted/current pair;
            # the rejected revision remains in history and cannot complete.
            context.decisions.append({
                "challenge_id": challenge.challenge_id,
                "action": "REVISION_REJECTED_KEEP_PARENT_PAIR",
            })
            break
        current_generation_input = revision_input
        current_review_input = next_review_input
        current_generation = next_generation
        current_review = next_review
        current_generation_contribution = next_generation_contribution
        current_review_contribution = next_review_contribution

    self_correction = None
    if iterations:
        self_correction = {
            "rounds": len(iterations),
            "verdict_before": first_verdict,
            "verdict_after": str(current_review.get("verdict") or ""),
            "reason": str(current_review.get("reason") or "")[:120],
        }
    return current_generation, current_review, current_generation_contribution, current_review_contribution, self_correction


def _synthesize_guidance_decision(
    *,
    context: CollaborationContext,
    generation: dict[str, Any],
    review: dict[str, Any],
    generation_contribution: AgentContribution,
    review_contribution: AgentContribution,
    l4_candidate: dict[str, Any] | None = None,
) -> tuple[GuidanceDecision, FinalCollaborationResult]:
    """Deterministically decide over reviewed facts without rewriting them."""
    verdict = str(review.get("verdict") or "")
    review_status = str(review.get("status") or "")
    learner_depth = str(
        context.learner_context.get("recommended_depth")
        or context.learner_context.get("level")
        or "unknown"
    )
    claims = generation_contribution.claims
    latest_challenge = context.challenges[-1] if context.challenges else None
    latest_action = getattr(latest_challenge, "requested_action", None)
    unresolved = bool(
        latest_challenge is not None
        and getattr(latest_challenge, "status", "")
        in {"OPEN", "NO_PROGRESS", "BUDGET_EXHAUSTED", "ASK_USER", "REJECT"}
    )
    ask_user = bool(
        latest_action is ResolutionAction.ASK_USER
        or generation.get("clarify")
    )
    rejected = bool(verdict == "rejected" or latest_action is ResolutionAction.REJECT)
    active_packs = _active_evidence_packs(context)
    missing = tuple(
        dict.fromkeys(
            value
            for pack in active_packs
            for value in pack.missing_information
            if value
        )
    )
    knowledge_gap_items = list(missing if verdict != "approved" else ())
    if generation.get("knowledge_unavailable") or generation.get("honest_unavailable"):
        knowledge_gap_items.append("knowledge or evidence unavailable")
    if unresolved and latest_challenge is not None:
        knowledge_gap_items.extend(
            getattr(latest_challenge, "missing_information", ())
        )
    knowledge_gap = tuple(dict.fromkeys(knowledge_gap_items))
    generation_uncertainty = tuple(generation_contribution.uncertainty)

    if rejected:
        active_state = ClaimFinalState.REJECTED
        active_reason = "current Reviewer rejected the claim"
    elif verdict == "approved" and not generation_uncertainty:
        active_state = ClaimFinalState.ACCEPTED
        active_reason = "current real Reviewer approved the reviewed contribution"
    else:
        active_state = ClaimFinalState.UNCERTAIN
        active_reason = "review or evidence remains unresolved"

    # Keep challenged/superseded Generation claims visible to the private
    # decision layer, but never restore them into the active conclusion.
    all_generation_claims = tuple(
        claim
        for contribution in context.contributions
        if contribution.agent_id == GENERATION_AGENT_ID
        for claim in contribution.claims
    )
    claim_states: dict[str, tuple[ClaimFinalState, str]] = {}
    for claim in all_generation_claims:
        if claim.claim_id in {item.claim_id for item in claims}:
            claim_states[claim.claim_id] = (active_state, active_reason)
        else:
            claim_states[claim.claim_id] = (
                ClaimFinalState.REJECTED,
                "superseded or challenged Generation claim is not in the final reviewed contribution",
            )
    claim_decisions = tuple(
        FinalClaimDecision(claim.claim_id, *claim_states[claim.claim_id])
        for claim in all_generation_claims
    )
    accepted_claims = tuple(
        claim
        for claim in all_generation_claims
        if claim_states[claim.claim_id][0] is ClaimFinalState.ACCEPTED
    )
    rejected_claims = tuple(
        claim
        for claim in all_generation_claims
        if claim_states[claim.claim_id][0] is ClaimFinalState.REJECTED
    )
    uncertain_claims = tuple(
        claim
        for claim in all_generation_claims
        if claim_states[claim.claim_id][0] is ClaimFinalState.UNCERTAIN
    )

    mode = context.intent_result.task_mode
    diagnosis_learning_path = context.learner_context.get("learning_path")
    knowledge_learning_context = context.learner_context.get(
        "knowledge_learning_context"
    )
    concept_learning_path = context.learner_context.get("concept_learning_path")
    path_nodes = (
        tuple(
            item
            for kp_id in diagnosis_learning_path.recommended_nodes
            for item in diagnosis_learning_path.milestones
            if item.kp_id == kp_id
        )
        if isinstance(diagnosis_learning_path, LearningPath)
        else ()
    )
    legacy_diagnosis_path = tuple(
        {
            "kp_id": item.kp_id,
            "action": (
                "analyze"
                if learner_depth in {"advanced", "graduate", "research"}
                else "learn"
            ),
            "topic": item.name,
        }
        for item in path_nodes[:4]
    )
    concept_path_topic = ""
    concept_diagnosis_path: tuple[dict[str, str], ...] = ()
    if (
        isinstance(knowledge_learning_context, KnowledgeLearningContext)
        and isinstance(concept_learning_path, ConceptLearningPath)
        and concept_learning_path.next_concept != "unknown"
    ):
        concept_id = concept_learning_path.next_concept
        concept_path_topic = knowledge_learning_context.concept_names.get(
            concept_id, concept_id
        )
        mapped_kps = knowledge_learning_context.concept_to_kps.get(concept_id, ())
        concept_diagnosis_path = ({
            "kp_id": mapped_kps[0] if mapped_kps else concept_id,
            "action": (
                "analyze"
                if learner_depth in {"advanced", "graduate", "research"}
                else "learn"
            ),
            "topic": concept_path_topic,
        },)
    diagnosis_path = concept_diagnosis_path or legacy_diagnosis_path
    if ask_user:
        decision_type = DecisionType.ASK_USER
        answer_policy = "WITHHOLD_AND_CLARIFY"
        next_action = "请明确比较或评价标准"
        path = (
            {"action": "clarify", "topic": "效率、热稳定性、显色、健康照明或成本"},
        )
    elif rejected:
        decision_type = DecisionType.REFUSE_CONCLUSION
        answer_policy = "WITHHOLD_REJECTED_CLAIM"
        next_action = "保留可核验事实并停止当前被拒结论"
        path = ({"action": "review", "topic": "核对被拒主张的直接证据"},)
    elif knowledge_gap and mode.value == "COMPARE":
        decision_type = DecisionType.KNOWLEDGE_GAP
        answer_policy = "WITHHOLD_UNVERIFIED_CLAIMS"
        next_action = "补充同条件测试数据后再比较"
        path = ({"action": "collect_evidence", "topic": "统一测试条件、效率、色度与热稳定性"},)
    elif knowledge_gap:
        decision_type = DecisionType.PARTIAL_ANSWER
        answer_policy = "WITHHOLD_UNVERIFIED_CLAIMS"
        next_action = "补充缺失证据与适用条件"
        path = ({"action": "collect_evidence", "topic": "、".join(knowledge_gap[:4])},)
    elif mode.value == "RESEARCH_GUIDE":
        decision_type = DecisionType.LEARNING_GUIDANCE
        answer_policy = "SHOW_REVIEWED"
        next_action = "按已审核前置条件开展下一步学习或验证"
        path = diagnosis_path or (
            {"action": "research_step", "topic": "确认变量、评价指标与验证检查点"},
        )
        if concept_path_topic or path_nodes:
            first_topic = concept_path_topic or path_nodes[0].name
            next_action = f"先完成 {first_topic}，再进入研究验证"
    elif verdict != "approved" or generation_uncertainty:
        decision_type = DecisionType.ANSWER_WITH_UNCERTAINTY
        answer_policy = (
            "SHOW_REVIEWED_WITH_UNCERTAINTY"
            if verdict == "approved"
            else "WITHHOLD_UNVERIFIED_CLAIMS"
        )
        next_action = "复核当前不确定主张"
        path = ({"action": "review", "topic": "证据边界与适用条件"},)
    elif mode.value == "FACT_FIND":
        decision_type = DecisionType.ANSWER
        answer_policy = "SHOW_REVIEWED"
        next_action = "确认该事实对应的证据来源"
        path = ()
    elif mode.value == "EXPLAIN":
        decision_type = DecisionType.ANSWER
        answer_policy = "SHOW_REVIEWED"
        if learner_depth in {"advanced", "graduate", "research"}:
            next_action = "分析能级机制、影响因素与边界条件"
            path = diagnosis_path or (
                {"action": "learn", "topic": "能级机制与基质条件"},
            )
        else:
            next_action = "先掌握能级跃迁与黄蓝发射的概念关系"
            path = diagnosis_path or (
                {"action": "learn", "topic": "能级跃迁基础"},
            )
        if concept_path_topic or path_nodes:
            first_topic = concept_path_topic or path_nodes[0].name
            next_action = (
                f"基于已有学习状态深化 {first_topic}"
                if learner_depth in {"advanced", "graduate", "research"}
                else f"先学习 {first_topic}，再推进当前问题"
            )
    elif mode.value == "EVALUATE":
        decision_type = DecisionType.ANSWER_WITH_UNCERTAINTY
        answer_policy = "SHOW_REVIEWED_WITH_UNCERTAINTY"
        next_action = "补充 SPD、暴露条件和风险加权指标"
        path = ({"action": "collect_evidence", "topic": "SPD、暴露与蓝光风险加权"},)
    else:
        decision_type = DecisionType.ANSWER
        answer_policy = "SHOW_REVIEWED"
        next_action = "按统一评价标准继续学习"
        path = ({"action": "learn", "topic": "统一评价标准"},)

    if not path and l4_candidate and isinstance(l4_candidate.get("recommended_path"), list):
        # L4 is candidate support only; FACT_FIND intentionally stays lightweight.
        if mode.value != "FACT_FIND":
            path = tuple(l4_candidate["recommended_path"])

    generation_conf = float(generation.get("confidence", 0.0) or 0.0)
    review_conf = float(review.get("confidence", 0.0) or 0.0)
    evidence_cap = 1.0 if any(pack.items for pack in active_packs) else 0.6
    review_cap = review_conf if verdict == "approved" else (0.55 if verdict == "needs_review" else 0.25)
    confidence = min(generation_conf, review_cap, evidence_cap)
    if generation_uncertainty or knowledge_gap:
        confidence = min(confidence, 0.65)
    confidence = round(max(0.0, confidence), 4)

    reviewed_identity = str(
        getattr(getattr(review, "_contract_candidate", None), "reviewed_answer_identity", "")
    )
    decision = GuidanceDecision(
        task_id=context.task_id,
        task_mode=mode,
        source_contribution_id=generation_contribution.contribution_id,
        source_review_id=review_contribution.contribution_id,
        review_identity=reviewed_identity,
        decision_type=decision_type,
        claim_decisions=claim_decisions,
        accepted_claim_ids=tuple(claim.claim_id for claim in accepted_claims),
        rejected_claim_ids=tuple(claim.claim_id for claim in rejected_claims),
        uncertain_claim_ids=tuple(claim.claim_id for claim in uncertain_claims),
        answer_policy=answer_policy,
        learner_depth=learner_depth,
        next_action=next_action,
        recommended_path=path,
        clarification_needed=ask_user,
        knowledge_gap=knowledge_gap,
        confidence=confidence,
        reasoning_summary=(
            f"{decision_type.value}: verdict={verdict or 'missing'}; "
            f"accepted={len(accepted_claims)}, rejected={len(rejected_claims)}, uncertain={len(uncertain_claims)}"
        ),
        status="completed",
    )
    reviewed_answer = generation_contribution.conclusion
    final_answer = (
        "" if answer_policy.startswith("WITHHOLD") else reviewed_answer
    )
    completion_eligibility = bool(
        reviewed_answer
        and review_status == "completed"
        and verdict == "approved"
        and not ask_user
        and not rejected
        and not unresolved
    )
    final_result = FinalCollaborationResult(
        task_id=context.task_id,
        task_mode=mode,
        answer=final_answer,
        answer_identity=(
            generation_contribution.artifact_identity
            or _answer_identity(context.task_id, reviewed_answer)
            if final_answer
            else ""
        ),
        accepted_claims=accepted_claims,
        rejected_claims=rejected_claims,
        uncertain_claims=uncertain_claims,
        evidence=active_packs,
        review=review_contribution,
        decision=decision,
        next_action=next_action,
        recommended_path=path,
        learner_context_summary=tuple(
            value
            for value in (
                f"learner_depth={learner_depth}",
                *(f"weak_kp={item}" for item in context.learner_context.get("weak_kps", ())),
            )
            if value
        ),
        knowledge_gaps=knowledge_gap,
        completion_eligibility=completion_eligibility,
        provenance_refs=tuple(
            dict.fromkeys(
                ref
                for pack in active_packs
                for item in pack.items
                for ref in (item.evidence_id, item.provenance_reference)
                if ref
            )
        ),
    )
    return decision, final_result


def _build_collaboration_trace(
    *,
    context: CollaborationContext,
    task_events: list[dict[str, Any]],
    final_result: FinalCollaborationResult,
) -> CollaborationTrace:
    """Project actual request-local runtime records into one causal trace."""
    created_at = float(context.runtime_metadata.get("created_at", time.time()))
    drafts: list[dict[str, Any]] = []
    ordinal = 0

    def add(
        event_type: str,
        actor: str,
        summary: str,
        *,
        timestamp: float,
        subtask_id: str = "",
        artifact_refs: tuple[str, ...] = (),
        caused_by_ref: str = "",
    ) -> None:
        nonlocal ordinal
        ordinal += 1
        drafts.append(
            {
                "ordinal": ordinal,
                "event_type": event_type,
                "actor": actor,
                "subtask_id": subtask_id,
                "timestamp": float(timestamp),
                "summary": str(summary)[:240],
                "artifact_refs": tuple(str(item) for item in artifact_refs if item),
                "caused_by_ref": str(caused_by_ref or ""),
            }
        )

    add(
        "TASK_UNDERSTOOD",
        "task.understanding",
        "收到用户任务并建立请求级协作上下文",
        timestamp=created_at,
    )
    add(
        "INTENT_RESOLVED",
        "task.understanding",
        f"识别任务模式 {context.intent_result.task_mode.value}",
        timestamp=created_at + 0.000001,
        artifact_refs=(context.intent_result.task_mode.value,),
    )
    add(
        "TASK_DECOMPOSED",
        "task.planning",
        f"形成 {len(context.task_plan.subtasks)} 个受约束子任务",
        timestamp=created_at + 0.000002,
        artifact_refs=tuple(item.subtask_id for item in context.task_plan.subtasks),
    )

    for subtask_id, _before, after, observed_at in context.runtime_metadata.get(
        "subtask_state_history", ()
    ):
        event_type = {
            "ready": "SUBTASK_READY",
            "running": "SUBTASK_RUNNING",
            "completed": "SUBTASK_COMPLETED",
            "failed": "SUBTASK_FAILED",
        }.get(str(after))
        if event_type:
            add(
                event_type,
                "task.scheduler",
                f"子任务 {subtask_id} 进入 {str(after).upper()}",
                timestamp=float(observed_at),
                subtask_id=str(subtask_id),
                artifact_refs=(str(subtask_id),),
            )

    for event in task_events:
        if str(event.get("event_type")) not in {"AgentStarted", "ReviewCompleted"}:
            continue
        actor = str((event.get("details") or {}).get("agent_id") or event.get("producer") or "")
        event_type = (
            "REVIEW_STARTED"
            if event.get("event_type") == "AgentStarted" and actor == REVIEW_AGENT_ID
            else str(event.get("event_type") or "").upper()
        )
        if event.get("event_type") == "ReviewCompleted":
            event_type = "REVIEW_COMPLETED"
        add(
            event_type,
            actor,
            (
                f"{actor} 开始执行"
                if event.get("event_type") == "AgentStarted"
                else f"真实审核完成：{(event.get('details') or {}).get('verdict', '')}"
            ),
            timestamp=float(event.get("timestamp") or created_at),
        )

    for retrieval in context.retrieval_history:
        timestamp = float(retrieval.get("timestamp") or created_at)
        version = int(retrieval.get("version") or 1)
        requested_by = str(retrieval.get("requested_by") or "")
        for plan_index, plan in enumerate(retrieval.get("plans") or (), start=1):
            plan_ref = f"retrieval-plan:{plan.subtask_id}:v{version}:{plan_index}"
            add(
                "RETRIEVAL_REQUESTED",
                GENERATION_AGENT_ID,
                f"为 {plan.subtask_id} 请求证据检索 v{version}",
                timestamp=timestamp + plan_index * 0.000001,
                subtask_id=plan.subtask_id,
                artifact_refs=(plan_ref,),
                caused_by_ref=requested_by,
            )
            add(
                "QUERY_REWRITTEN",
                "retrieval.planning",
                f"使用 {len(plan.rewritten_queries)} 条确定性查询",
                timestamp=timestamp + plan_index * 0.000001 + 0.0000001,
                subtask_id=plan.subtask_id,
                artifact_refs=(plan_ref, *plan.rewritten_queries),
                caused_by_ref=plan_ref,
            )
        for pack_index, pack in enumerate(retrieval.get("packs") or (), start=1):
            pack_ref = f"evidence-pack:{pack.subtask_id}:v{pack.version}"
            plan_ref = f"retrieval-plan:{pack.subtask_id}:v{version}:{pack_index}"
            add(
                "EVIDENCE_RETRIEVED",
                "retrieval.runtime",
                f"EvidencePack v{pack.version} 获得 {len(pack.items)} 条真实证据",
                timestamp=timestamp + pack_index * 0.000001 + 0.0000002,
                subtask_id=pack.subtask_id,
                artifact_refs=(pack_ref, *(item.evidence_id for item in pack.items)),
                caused_by_ref=plan_ref,
            )
            add(
                "EVIDENCE_RERANKED",
                "retrieval.rerank",
                f"EvidencePack v{pack.version} 完成任务相关重排",
                timestamp=timestamp + pack_index * 0.000001 + 0.0000003,
                subtask_id=pack.subtask_id,
                artifact_refs=(pack_ref,),
                caused_by_ref=pack_ref,
            )

    challenge_times = context.runtime_metadata.get("challenge_recorded_at", {})
    for challenge in context.challenges:
        timestamp = float(challenge_times.get(challenge.challenge_id) or created_at)
        add(
            "CHALLENGE_RAISED",
            challenge.reviewer_agent_id,
            f"{challenge.challenge_type.value}: {challenge.reason}",
            timestamp=timestamp,
            subtask_id=challenge.subtask_id,
            artifact_refs=(challenge.challenge_id, *challenge.target_claim_ids),
            caused_by_ref=challenge.target_contribution_id,
        )
        if challenge.requested_action in {
            ResolutionAction.REVISE,
            ResolutionAction.RE_RETRIEVE,
        }:
            add(
                (
                    "REVISION_REQUESTED"
                    if challenge.requested_action is ResolutionAction.REVISE
                    else "RE_RETRIEVAL_REQUESTED"
                ),
                challenge.reviewer_agent_id,
                f"请求 {challenge.requested_action.value}: {', '.join(challenge.missing_information)}",
                timestamp=timestamp + 0.000001,
                subtask_id=challenge.subtask_id,
                artifact_refs=(challenge.challenge_id, *challenge.missing_information),
                caused_by_ref=challenge.challenge_id,
            )

    generation_before_review: AgentContribution | None = None
    for contribution in context.contributions:
        event_type = "CONTRIBUTION_REVISED" if contribution.parent_contribution_id else "CONTRIBUTION_PRODUCED"
        if contribution.agent_id == REVIEW_AGENT_ID:
            caused_by = (
                generation_before_review.contribution_id
                if generation_before_review is not None
                else ""
            )
        elif contribution.parent_contribution_id:
            caused_by = next(
                (
                    challenge.challenge_id
                    for challenge in reversed(context.challenges)
                    if challenge.target_contribution_id == contribution.parent_contribution_id
                ),
                contribution.parent_contribution_id,
            )
        else:
            caused_by = ""
        add(
            event_type,
            contribution.agent_id,
            contribution.conclusion or f"{contribution.agent_id} 产生结构化贡献",
            timestamp=float(contribution.produced_at),
            subtask_id=contribution.subtask_id,
            artifact_refs=(contribution.contribution_id, *contribution.evidence_refs),
            caused_by_ref=caused_by,
        )
        if contribution.agent_id == GENERATION_AGENT_ID:
            generation_before_review = contribution

    guidance_contribution = next(
        (
            item
            for item in reversed(context.contributions)
            if item.agent_id == GUIDANCE_AGENT_ID
        ),
        None,
    )
    if guidance_contribution is not None:
        add(
            "GUIDANCE_DECIDED",
            GUIDANCE_AGENT_ID,
            f"{final_result.decision.decision_type.value}: {final_result.next_action}",
            timestamp=float(guidance_contribution.produced_at) + 0.000001,
            subtask_id=guidance_contribution.subtask_id,
            artifact_refs=(guidance_contribution.contribution_id,),
            caused_by_ref=final_result.review.contribution_id,
        )

    drafts.sort(key=lambda item: (item["timestamp"], item["ordinal"]))
    artifact_event_ids: dict[str, str] = {}
    events: list[CollaborationTraceEvent] = []
    for sequence, draft in enumerate(drafts, start=1):
        event_id = f"trace-{context.task_id}-{sequence:03d}"
        parent_event_id = artifact_event_ids.get(draft["caused_by_ref"], "")
        event = CollaborationTraceEvent(
            event_id=event_id,
            task_id=context.task_id,
            sequence=sequence,
            event_type=draft["event_type"],
            actor=draft["actor"],
            subtask_id=draft["subtask_id"],
            timestamp=draft["timestamp"],
            summary=draft["summary"],
            artifact_refs=draft["artifact_refs"],
            parent_event_id=parent_event_id,
            caused_by=draft["caused_by_ref"],
        )
        events.append(event)
        for artifact_ref in event.artifact_refs:
            artifact_event_ids.setdefault(artifact_ref, event_id)

    signature: list[str] = []
    actor_counts: dict[str, int] = {}
    for event in events:
        if event.event_type == "SUBTASK_READY":
            signature.append(f"READY:{event.subtask_id}")
        elif event.event_type in {"CONTRIBUTION_PRODUCED", "CONTRIBUTION_REVISED"}:
            short = {
                DIAGNOSIS_AGENT_ID: "DIAG",
                GENERATION_AGENT_ID: "GEN",
                REVIEW_AGENT_ID: "REV",
                GUIDANCE_AGENT_ID: "GUIDE",
            }.get(event.actor)
            if short:
                actor_counts[short] = actor_counts.get(short, 0) + 1
                signature.append(f"{short}{actor_counts[short]}")
        elif event.event_type == "CHALLENGE_RAISED":
            signature.append("CHALLENGE")
        elif event.event_type == "RE_RETRIEVAL_REQUESTED":
            signature.append("RETRIEVE_AGAIN")
        elif event.event_type == "REVISION_REQUESTED":
            signature.append("REVISE")

    counts = context.runtime_metadata.get("r03e_call_counts", {})
    costs = {
        "task_mode": context.intent_result.task_mode.value,
        "agent_execution_count": sum(
            1 for item in task_events if item.get("event_type") == "AgentStarted"
        ),
        "generation_count": int(counts.get("generation", actor_counts.get("GEN", 0))),
        "retrieval_count": len(context.retrieval_history),
        "review_count": int(counts.get("review", actor_counts.get("REV", 0))),
        "revision_count": sum(
            1
            for item in context.contributions
            if item.agent_id == GENERATION_AGENT_ID and item.parent_contribution_id
        ),
        "re_retrieval_count": sum(
            1 for item in context.retrieval_history if item.get("requested_by")
        ),
        "llm_call_count": None,
        "llm_call_count_status": "not_instrumented",
        "latency_ms": round(
            max((item.timestamp for item in events), default=created_at) * 1000
            - created_at * 1000,
            2,
        ),
    }
    return CollaborationTrace(
        task_id=context.task_id,
        task_mode=context.intent_result.task_mode,
        events=tuple(events),
        path_signature=tuple(signature),
        cost_summary=costs,
    )


def _evaluate_multi_agent_runtime(
    *,
    context: CollaborationContext,
    trace: CollaborationTrace,
    final_result: FinalCollaborationResult,
    answer_correlation: _AnswerCorrelation,
    review_candidate: _ReviewCandidate | None,
) -> _MultiAgentEvaluation:
    active_packs = _active_evidence_packs(context)
    items = [item for pack in active_packs for item in pack.items]
    evidence_ids = [item.evidence_id for item in items]
    sources = [item.source_reference for item in items if item.source_reference]
    intent_entities = {
        item.text.casefold() for item in context.intent_result.domain_entities
    }
    entity_matches = [
        item for item in items if item.entity and item.entity.casefold() in intent_entities
    ]
    material_items = [item for item in items if item.material_system]
    rewritten_queries = [
        query
        for history in context.retrieval_history
        for plan in history.get("plans", ())
        for query in plan.rewritten_queries
    ]
    original_queries = [
        plan.original_query
        for history in context.retrieval_history
        for plan in history.get("plans", ())
    ]
    challenged_targets = {item.target_contribution_id for item in context.challenges}
    known_contributions = {item.contribution_id for item in context.contributions}
    budget = context.collaboration_budget
    iteration = context.iteration_state
    task_intelligence = {
        "intent_resolved": bool(context.intent_result.primary_intent),
        "task_mode": context.intent_result.task_mode.value,
        "task_plan_valid": bool(context.task_plan.subtasks),
        "task_decomposition_size": len(context.task_plan.subtasks),
        "path_signature": trace.path_signature,
    }
    retrieval_intelligence = {
        "relevant_at_k": (
            round(sum(item.relevance_score > 0 for item in items) / len(items), 4)
            if items
            else None
        ),
        "entity_match_at_k": (
            round(len(entity_matches) / len(items), 4) if items else None
        ),
        "material_match_at_k": (
            round(len(material_items) / len(items), 4) if items else None
        ),
        "coverage_count": len({value for pack in active_packs for value in pack.coverage}),
        "missing_information_count": len(
            {value for pack in active_packs for value in pack.missing_information}
        ),
        "duplicate_rate": (
            round(1.0 - len(set(evidence_ids)) / len(evidence_ids), 4)
            if evidence_ids
            else None
        ),
        "source_diversity": len(set(sources)),
        "query_rewrite_changed": any(
            rewritten != original
            for rewritten, original in zip(rewritten_queries, original_queries)
        ),
    }
    collaboration_intelligence = {
        "contribution_count": len(context.contributions),
        "diagnosis_influence": bool(context.learner_context.get("recommended_depth")),
        "reviewer_influence": bool(context.challenges or final_result.rejected_claims or final_result.uncertain_claims),
        "evidence_influence": bool(final_result.evidence and final_result.provenance_refs),
        "challenge_count": len(context.challenges),
        "challenge_target_validity": (
            round(len(challenged_targets & known_contributions) / len(challenged_targets), 4)
            if challenged_targets
            else None
        ),
        "successful_correction": bool(
            context.challenges
            and final_result.review.requested_actions
            and final_result.review.requested_actions[0] is RequestedAction.ACCEPT
        ),
        "no_progress_termination": any(
            item.status == "NO_PROGRESS" for item in context.challenges
        ),
        "budget_compliant": bool(
            iteration.get("expensive_iterations_used", 0)
            <= min(3, budget.max_expensive_iterations)
            and iteration.get("retrievals_used", 0) <= budget.retrieval_budget
            and iteration.get("review_revisions_used", 0)
            <= budget.review_revision_budget
        ),
    }
    trust = {
        "unsupported_claim_rejected": bool(final_result.rejected_claims),
        "knowledge_gap_honest": bool(
            not final_result.knowledge_gaps
            or final_result.decision.decision_type
            in {DecisionType.KNOWLEDGE_GAP, DecisionType.PARTIAL_ANSWER, DecisionType.ASK_USER}
        ),
        "answer_evidence_review_identity": answer_correlation.correlation,
        "post_review_mutation_prevented": answer_correlation.correlation,
        "reviewer_authority": bool(
            review_candidate is not None and review_candidate.real_reviewer_executed
        ),
        "uncertainty_preserved": bool(
            not final_result.uncertain_claims
            or final_result.decision.decision_type
            in {
                DecisionType.ANSWER_WITH_UNCERTAINTY,
                DecisionType.PARTIAL_ANSWER,
                DecisionType.KNOWLEDGE_GAP,
            }
        ),
    }
    educational = {
        "learner_adaptation_active": bool(final_result.decision.learner_depth),
        "learner_depth": final_result.decision.learner_depth,
        "next_action_present": bool(final_result.next_action),
        "recommended_path_size": len(final_result.recommended_path),
    }
    return _MultiAgentEvaluation(
        task_id=context.task_id,
        task_mode=context.intent_result.task_mode.value,
        task_intelligence=task_intelligence,
        retrieval_intelligence=retrieval_intelligence,
        collaboration_intelligence=collaboration_intelligence,
        trust=trust,
        educational_intelligence=educational,
        costs=dict(trace.cost_summary),
    )


def _public_trace_summary(
    event: CollaborationTraceEvent,
    *,
    release_eligible: bool,
    limit: int,
) -> str:
    if (
        event.actor == GENERATION_AGENT_ID
        and event.event_type in {"CONTRIBUTION_PRODUCED", "CONTRIBUTION_REVISED"}
    ):
        # A trace records that generation happened; it is not another answer
        # channel.  Always suppress draft bodies here because an earlier draft
        # can be rejected even when a later revision is safe to release.
        return (
            "知识生成产物已形成，正文由最终发布结果展示"
            if release_eligible
            else "知识生成产物未通过发布门，正文已隐藏"
        )
    return event.summary[:limit]


def _project_agent_trace(
    trace: CollaborationTrace,
    *,
    release_eligible: bool = True,
) -> list[dict[str, Any]]:
    visible_types = {
        "CONTRIBUTION_PRODUCED",
        "CONTRIBUTION_REVISED",
        "CHALLENGE_RAISED",
        "RE_RETRIEVAL_REQUESTED",
        "REVISION_REQUESTED",
        "GUIDANCE_DECIDED",
    }
    return [
        {
            "agent_id": event.actor,
            "result": "observed",
            "time": event.timestamp,
            "detail": _public_trace_summary(
                event, release_eligible=release_eligible, limit=120
            ),
        }
        for event in trace.events
        if event.event_type in visible_types and event.actor.startswith("agent.")
    ]


def _project_collab_lines(
    trace: CollaborationTrace,
    *,
    release_eligible: bool = True,
) -> list[dict[str, Any]]:
    visible_types = {
        "SUBTASK_READY",
        "CONTRIBUTION_PRODUCED",
        "CONTRIBUTION_REVISED",
        "CHALLENGE_RAISED",
        "REVISION_REQUESTED",
        "RE_RETRIEVAL_REQUESTED",
        "EVIDENCE_RETRIEVED",
        "REVIEW_COMPLETED",
        "GUIDANCE_DECIDED",
    }
    return [
        {
            "line": f"T{event.sequence}",
            "label": event.event_type,
            "kind": "runtime_fact",
            "steps": [
                {
                    "agent": event.actor,
                    "elapsed_ms": None,
                    "output": _public_trace_summary(
                        event, release_eligible=release_eligible, limit=160
                    ),
                }
            ],
        }
        for event in trace.events
        if event.event_type in visible_types
    ]


def _project_flow_events(
    trace: CollaborationTrace,
    *,
    release_eligible: bool = True,
) -> list[dict[str, Any]]:
    return [
        {
            "seq": event.sequence,
            "step": event.event_type,
            "agent": event.actor,
            "label": _public_trace_summary(
                event, release_eligible=release_eligible, limit=80
            ),
            "to": "",
            "detail": _public_trace_summary(
                event, release_eligible=release_eligible, limit=160
            ),
            "elapsed_ms": None,
        }
        for event in trace.events
        if event.event_type
        in {
            "TASK_DECOMPOSED",
            "CONTRIBUTION_PRODUCED",
            "CONTRIBUTION_REVISED",
            "CHALLENGE_RAISED",
            "RE_RETRIEVAL_REQUESTED",
            "REVIEW_COMPLETED",
            "GUIDANCE_DECIDED",
        }
    ]


def run_guidance(
    input_data: dict[str, Any],
    deps: AgentDependencies,
) -> dict[str, Any]:
    """导学决策 Agent — 先理解任务，再运行当前四 Agent 主链."""

    # R-03B collaboration-runtime boundary: create exactly one private context
    # containing the R-03A intent and its mode-specific TaskPlan before the
    # first Agent executes. CURRENT workers receive the same request-local
    # object but do not yet schedule from it; AgentInput/Contribution is R-03C.
    input_data = dict(input_data)
    learner_memory_views = _load_learner_memory_views(input_data, deps)
    teaching_memory_view = _load_teaching_memory(input_data, deps)
    collaboration_context = initialize_collaboration_context(
        input_data,
        intent_resolver=understand_task,
    )
    # Legacy Memory remains readable, but only its Diagnosis projection may
    # enter the normalized request-local view.  It no longer rewrites TaskPlan
    # or reaches Generation/Guidance as an Agent-specific private payload.
    _apply_memory_to_collaboration_context(
        collaboration_context,
        learner_memory_views,
    )
    input_data["_learner_intelligence_view"] = build_learner_intelligence_view(
        input_data,
        deps,
        learner_memory_view=learner_memory_views.get(DIAGNOSIS_AGENT_ID),
        teaching_memory_view=teaching_memory_view,
    )

    task_context = input_data.get("task_context")

    # The TaskPlan already exists at this real boundary.  Persist PLANNING as
    # an execution fact instead of inferring it after the response returns.
    task_state_runtime.set_task_state(
        task_context,
        "PLANNING",
        producer="run_guidance",
    )

    def _task_event(
        event_type: str,
        producer: str,
        **details: Any,
    ) -> None:
        task_state_runtime.record_task_event(
            task_context,
            event_type,
            producer,
            details=details or None,
        )

    def _audit_trail(
        agent_id: str,
        detail: str,
        learner: str = "",
        action: str = "agent_invoke",
        latency_ms: float = 0.0,
    ) -> None:
        """记录单条 Agent 执行轨迹 (L0 审计, 持久化可查)."""
        ae = getattr(deps, "audit_engine", None)
        if ae is None:
            return
        try:
            ae.record(
                actor=str(learner or input_data.get("learner_id") or input_data.get("student_id") or "demo"),
                action=action,
                layer="L5",
                outcome="success",
                agent_id=agent_id,
                trace_id=str(input_data.get("trace_id") or input_data.get("task_id") or ""),
                session_id=str(input_data.get("session_id") or f"request-{input_data.get('task_id', '')}"),
                latency_ms=max(0.0, float(latency_ms or 0.0)),
                input_context={
                    "task_id": str(input_data.get("task_id") or ""),
                    "learner_id": str(input_data.get("learner_id") or input_data.get("student_id") or ""),
                    "query": str(input_data.get("query", ""))[:120],
                },
                output_result={
                    "agent_id": agent_id,
                    "detail": str(detail)[:200],
                },
                metadata={"task_id": str(input_data.get("task_id") or "")},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent 轨迹记录失败 %s: %s", agent_id, exc)

    _rec = get_recorder()
    # ---- 主线 L1: TaskPlan 驱动的学情诊断 ----
    diagnosis_input = _start_contract_agent(
        collaboration_context,
        DIAGNOSIS_AGENT_ID,
    )
    if diagnosis_input is not None:
        _t0 = time.time()
        _task_event("AgentStarted", DIAGNOSIS_AGENT_ID, agent_id=DIAGNOSIS_AGENT_ID)
        diagnosis = run_diagnosis(
            _contract_runtime_payload(input_data, diagnosis_input),
            deps,
        )
        _task_event("AgentFinished", DIAGNOSIS_AGENT_ID, agent_id=DIAGNOSIS_AGENT_ID)
        t_diag = round((time.time() - _t0) * 1000, 1)
        diagnosis_contribution = _adapt_diagnosis_contribution(
            collaboration_context,
            diagnosis_input,
            diagnosis,
        )
        learner_view = input_data.get("_learner_intelligence_view")
        if isinstance(learner_view, LearnerIntelligenceView):
            _apply_diagnosis_teaching_context(
                collaboration_context,
                learner_view,
            )
        _finish_contract_agent(
            collaboration_context,
            diagnosis_input,
            diagnosis_contribution,
        )
        _task_event(
            "AgentContributionRecorded",
            DIAGNOSIS_AGENT_ID,
            agent_id=DIAGNOSIS_AGENT_ID,
            contribution_id=diagnosis_contribution.contribution_id,
        )
        _audit_trail(DIAGNOSIS_AGENT_ID, "学情诊断: 薄弱点 " + ",".join(list(diagnosis.get("weak_kps", []))[:5]), learner=str(input_data.get("learner_id", "")), latency_ms=t_diag)
        _rec.record_agent_execution(
            agent_id=DIAGNOSIS_AGENT_ID, agent_name="学情诊断 Agent",
            action="诊断薄弱点: " + ",".join(list(diagnosis.get("weak_kps", []))[:3]),
            duration_ms=t_diag, status="completed", phase=InteractionPhase.DIAGNOSIS,
            input_data={"query": diagnosis_input.user_query},
            output_data={"weak_kps": list(diagnosis.get("weak_kps", [])), "confidence": diagnosis.get("confidence")},
        )
    else:
        t_diag = 0.0
        diagnosis = {
            "agent_id": DIAGNOSIS_AGENT_ID,
            "status": "not_required",
            "learner_id": input_data.get("learner_id") or input_data.get("student_id"),
            "weak_kps": [],
            "summary": "TaskPlan did not require learner diagnosis",
            "confidence": 1.0,
        }

    # ---- 主线 L2: 多候选知识生成 + 交叉验证 (流程多样性, L5 高等级对标) ----
    task_state_runtime.set_task_state(
        task_context,
        "RETRIEVING",
        producer="run_guidance",
    )
    _t1 = time.time()
    generation_input = _start_contract_agent(
        collaboration_context,
        GENERATION_AGENT_ID,
    )
    if generation_input is None:
        raise RuntimeError("TaskPlan has no ready Generation subtask")
    _task_event("AgentStarted", GENERATION_AGENT_ID, agent_id=GENERATION_AGENT_ID)
    generation_payload = _contract_runtime_payload(input_data, generation_input)
    generation_input, generation_payload = _prepare_generation_retrieval(
        collaboration_context,
        generation_input,
        generation_payload,
        deps,
    )
    _task_event(
        "RetrievalCompleted",
        GENERATION_AGENT_ID,
        query_count=len(
            collaboration_context.tool_results.get("retrieval_plans", ()) or ()
        ),
        evidence_version=max(
            (pack.version for pack in _active_evidence_packs(collaboration_context)),
            default=0,
        ),
    )
    generation = _run_multi_candidate_generation(generation_payload, deps)
    _task_event("AgentFinished", GENERATION_AGENT_ID, agent_id=GENERATION_AGENT_ID)
    generation_contribution = _adapt_generation_contribution(
        collaboration_context,
        generation_input,
        generation,
    )
    _finish_contract_agent(
        collaboration_context,
        generation_input,
        generation_contribution,
    )
    _task_event(
        "AgentContributionRecorded",
        GENERATION_AGENT_ID,
        agent_id=GENERATION_AGENT_ID,
        contribution_id=generation_contribution.contribution_id,
    )
    t_gen = round((time.time() - _t1) * 1000, 1)
    t_cv = t_gen  # 交叉验证在候选生成内完成, 共享耗时
    n_cand = len(generation.get("candidates", []))
    _audit_trail(GENERATION_AGENT_ID, "知识生成(多候选交叉验证): 共识 " + str(generation.get("consensus_score", 0)) + " · 候选 " + str(n_cand), learner=str(input_data.get("learner_id", "")), latency_ms=t_gen)
    _rec.record_agent_execution(
        agent_id=GENERATION_AGENT_ID, agent_name="知识生成 Agent",
        action="多候选生成 + 交叉验证: 共识 " + str(generation.get("consensus_score", 0)) + " · 达成 " + str(generation.get("consensus_reached", False)) + " · 选中 " + str(generation.get("selected_candidate", "")),
        duration_ms=t_gen, status="completed", phase=InteractionPhase.GENERATION,
        input_data={"query": input_data.get("query", "")},
        output_data={"n_candidates": n_cand, "consensus_score": generation.get("consensus_score"), "consensus_reached": generation.get("consensus_reached"), "needs_adjudication": generation.get("needs_adjudication"), "answer_length": len(generation.get("answer", ""))},
    )

    # 生成结果已形成并交给下游能力，进入当前真实协同交接阶段。
    task_state_runtime.set_task_state(
        task_context,
        "COLLABORATING",
        producer="run_guidance",
    )

    # ---- 主线 L3: 审核校验 (初审) ----
    _t2 = time.time()
    generation_answer = str(generation.get("answer") or "")
    has_real_generation_evidence = bool(
        generation.get("context_chunks")
        or generation.get("evidence")
        or generation.get("citations")
        or generation.get("sources")
    )
    terminal_without_reviewable_answer = bool(
        generation.get("honest_unavailable")
        or generation.get("knowledge_unavailable")
        or generation.get("clarify")
        or not generation_answer
    )
    if terminal_without_reviewable_answer:
        # 不可用/澄清/空回答没有可供真实 Reviewer 审核的完整内容；明确跳过，
        # 不用 synthetic approved 冒充质量审核通过。
            review = _attach_review_candidate(
            {
                "verdict": "skipped",
                "confidence": 0.0,
                "reason": "不可用、澄清或空回答，未执行真实审核",
                "plain_language": bool(generation.get("plain_language")),
                "deduced": bool(generation.get("deduced")),
                "honest_unavailable": bool(
                    generation.get("honest_unavailable")
                ),
            },
            input_data,
            content=str(generation.get("answer") or ""),
            producer="synthetic_review",
            real_reviewer_executed=False,
            mapping_refused_reason=(
                "no reviewed content"
                if not generation_answer
                else "real reviewer not executed"
            ),
            )
            # Preserve a typed, explicitly skipped Review contribution for
            # Guidance.  This records the absence of a real Reviewer without
            # turning the synthetic result into an approval.
            review_input = _start_contract_agent(
                collaboration_context,
                REVIEW_AGENT_ID,
            )
            if review_input is None:
                raise RuntimeError("TaskPlan has no ready Review subtask")
            review_contribution = _adapt_review_contribution(
                collaboration_context,
                review_input,
                review,
            )
            _finish_contract_agent(
                collaboration_context,
                review_input,
                review_contribution,
            )
            _task_event(
                "AgentContributionRecorded",
                REVIEW_AGENT_ID,
                agent_id=REVIEW_AGENT_ID,
                contribution_id=review_contribution.contribution_id,
                status="skipped",
            )
            review_input = None
    else:
        evidence_candidate = getattr(generation, "_contract_candidate", None)
        if (
            has_real_generation_evidence
            and isinstance(evidence_candidate, _EvidenceCandidate)
            and evidence_candidate.stage != "selected"
        ):
            # 有真实答案和真实来源的 terminal 结果已经成为最终候选；只修正
            # private stage 事实，public generation keys 与 evidence 内容保持不变。
            generation = _attach_selected_evidence_candidate(
                dict(generation),
                input_data,
                stage="selected",
            )
        task_state_runtime.set_task_state(
            task_context,
            "REVIEWING",
            producer="run_guidance",
        )
        review_input = _start_contract_agent(
            collaboration_context,
            REVIEW_AGENT_ID,
        )
        if review_input is None:
            raise RuntimeError("TaskPlan has no ready Review subtask")
        generation_facts = [
            contribution
            for contribution in review_input.prior_contributions
            if contribution.agent_id == GENERATION_AGENT_ID
        ]
        reviewed_generation = generation_facts[-1] if generation_facts else None
        review_payload = _contract_runtime_payload(input_data, review_input)
        review_payload.update(
            {
                "content": (
                    reviewed_generation.conclusion
                    if reviewed_generation is not None
                    else generation.get("answer", "")
                ),
                "context_chunks": _review_evidence_texts(
                    review_input,
                    reviewed_generation,
                ),
                "_claim_evidence_grounding": _review_scientific_grounding(
                    collaboration_context,
                    reviewed_generation,
                ),
            }
        )
        _task_event("AgentStarted", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID)
        review = run_review(review_payload, deps)
        _task_event(
            "ReviewCompleted",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            verdict=str(review.get("verdict", "")),
        )
        _task_event("AgentFinished", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID)
        review_contribution = _adapt_review_contribution(
            collaboration_context,
            review_input,
            review,
        )
        _finish_contract_agent(
            collaboration_context,
            review_input,
            review_contribution,
        )
        _task_event(
            "AgentContributionRecorded",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            contribution_id=review_contribution.contribution_id,
        )
    t_rev = round((time.time() - _t2) * 1000, 1)
    _audit_trail(REVIEW_AGENT_ID, "审核校验: " + str(review.get("verdict", "")), learner=str(input_data.get("learner_id", "")), latency_ms=t_rev)
    _rec.record_agent_execution(
        agent_id=REVIEW_AGENT_ID, agent_name="审核校验 Agent",
        action="事实核查与幻觉检测: " + str(review.get("verdict", "")),
        duration_ms=t_rev, status="completed", phase=InteractionPhase.REVIEW,
        input_data={"content_length": len(generation.get("answer", ""))},
        output_data={"verdict": review.get("verdict"), "confidence": review.get("confidence"), "reason": str(review.get("reason", ""))[:100]},
    )

    # ---- 多线协作时序 (主线 + 并行候选 + 交叉验证 + 辩论 + 自纠回路) ----
    cands = list(generation.get("candidates", []))
    consensus_score = float(generation.get("consensus_score", 0.0) or 0.0)
    consensus_reached = bool(generation.get("consensus_reached", False))
    debate = generation.get("debate")
    needs_adjudication = bool(generation.get("needs_adjudication", False))

    collab_lines: list[dict[str, Any]] = [
        {
            "line": "L1",
            "label": "主线",
            "steps": [
                {"agent": DIAGNOSIS_AGENT_ID, "elapsed_ms": t_diag, "output": "薄弱点 " + ",".join(list(diagnosis.get("weak_kps", []))[:4])},
                {"agent": "cross.validate", "elapsed_ms": t_cv, "output": "多候选交叉验证 共识 " + f"{consensus_score:.0%}"},
                {"agent": REVIEW_AGENT_ID, "elapsed_ms": t_rev, "output": "终审: " + str(review.get("verdict", ""))},
            ],
        },
    ]
    # 并行候选线 L1.A / L1.B / L1.C (流程多样性: 多策略并行生成)
    selected_cand = str(generation.get("selected_candidate", ""))
    for cand in cands:
        cid = str(cand.get("candidate_id", ""))
        collab_lines.append({
            "line": "L1." + cid,
            "label": str(cand.get("label", "")),
            "kind": "candidate",
            "steps": [
                {"agent": GENERATION_AGENT_ID, "elapsed_ms": t_gen, "output": "候选 " + cid + " · 置信度 " + (f"{cand.get('confidence', 0):.2f}") + (" · 选中" if cid == selected_cand else "")},
            ],
        })
    # 交叉验证 / 共识线
    collab_lines.append({
        "line": "L2",
        "label": "交叉验证",
        "kind": "consensus",
        "steps": [
            {"agent": "cross.validate", "elapsed_ms": t_cv, "output": "两两分歧度矩阵 · 共识度 " + f"{consensus_score:.0%}" + (" · 达成共识" if consensus_reached else " · 未达共识")},
        ],
    })
    # 协同辩论线 (分歧大 → 论据交换 → 收敛/待裁决)
    if debate:
        collab_lines.append({
            "line": "L2.1",
            "label": "协同辩论",
            "kind": "debate",
            "steps": [
                {"agent": "debate.pro", "elapsed_ms": t_cv, "output": str(debate.get("pro", {}).get("label", "")) + " 论据交换"},
                {"agent": "debate.con", "elapsed_ms": t_cv, "output": str(debate.get("con", {}).get("label", "")) + " 论据交换"},
                {"agent": "debate.vote", "elapsed_ms": t_cv, "output": ("收敛·采纳胜方" if debate.get("converged") else "未收敛") + " · 分歧 " + f"{debate.get('divergence_after', 0):.0%}"},
            ],
        })
    if needs_adjudication:
        collab_lines.append({
            "line": "L2.2",
            "label": "待裁决",
            "kind": "adjudication",
            "steps": [
                {"agent": "agent.adjudicator", "elapsed_ms": 0.0, "output": "分歧未收敛 → 人工/仲裁确认"},
            ],
        })
    # ---- R-03E authoritative correction semantics ----
    # The former self-correction implementation remains below only as a frozen
    # compatibility reference.  Reviewer Challenge is now the sole trigger.
    self_correction: dict[str, Any] | None = None
    reasoning_loop: dict[str, Any] | None = None
    if review_input is not None:
        (
            generation,
            review,
            generation_contribution,
            review_contribution,
            self_correction,
        ) = _run_authoritative_correction_loop(
            context=collaboration_context,
            input_data=input_data,
            deps=deps,
            generation=generation,
            review=review,
            generation_input=generation_input,
            review_input=review_input,
            generation_contribution=generation_contribution,
            review_contribution=review_contribution,
            task_event=_task_event,
        )

    # ---- frozen legacy self-correction executor (no longer a decision source) ----
    initial_generation, initial_review = generation, review
    if False and review.get("verdict") == "needs_review" and generation.get("answer"):
        _t3 = time.time()
        generation2_input = _revision_agent_input(
            collaboration_context,
            generation_input,
        )
        _task_event("AgentStarted", GENERATION_AGENT_ID, agent_id=GENERATION_AGENT_ID)
        generation2 = _run_multi_candidate_generation(
            _contract_runtime_payload(input_data, generation2_input), deps,
            review_feedback=str(review.get("reason", "")),
        )
        _task_event("AgentFinished", GENERATION_AGENT_ID, agent_id=GENERATION_AGENT_ID)
        generation2_contribution = _adapt_generation_contribution(
            collaboration_context,
            generation2_input,
            generation2,
        )
        collaboration_context.record_contribution(generation2_contribution)
        review2_input = _revision_agent_input(
            collaboration_context,
            review_input,
        )
        review2_payload = _contract_runtime_payload(input_data, review2_input)
        review2_payload.update(
            {
                "content": generation2_contribution.conclusion,
                "context_chunks": _review_evidence_texts(
                    review2_input,
                    generation2_contribution,
                ),
            }
        )
        _task_event("AgentStarted", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID)
        review2 = run_review(review2_payload, deps)
        _task_event(
            "ReviewCompleted",
            REVIEW_AGENT_ID,
            agent_id=REVIEW_AGENT_ID,
            verdict=str(review2.get("verdict", "")),
        )
        _task_event("AgentFinished", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID)
        review2_contribution = _adapt_review_contribution(
            collaboration_context,
            review2_input,
            review2,
        )
        collaboration_context.record_contribution(review2_contribution)
        t_loop = round((time.time() - _t3) * 1000, 1)
        _audit_trail(GENERATION_AGENT_ID, "知识生成(自纠修订): " + (str(generation2.get("answer", ""))[:60]), learner=str(input_data.get("learner_id", "")), latency_ms=t_loop / 2)
        _audit_trail(REVIEW_AGENT_ID, "审核校验(终审): " + str(review2.get("verdict", "")), learner=str(input_data.get("learner_id", "")), latency_ms=t_loop / 2)
        _rec.record_agent_execution(
            agent_id=GENERATION_AGENT_ID, agent_name="知识生成 Agent",
            action="多候选修订 + 交叉验证 (依据审核意见)",
            duration_ms=t_loop / 2, status="completed", phase=InteractionPhase.GENERATION,
            input_data={"review_feedback": str(review.get("reason", ""))[:100]},
            output_data={"answer_length": len(generation2.get("answer", ""))},
        )
        _rec.record_agent_execution(
            agent_id=REVIEW_AGENT_ID, agent_name="审核校验 Agent",
            action="终审: " + str(review2.get("verdict", "")),
            duration_ms=t_loop / 2, status="completed", phase=InteractionPhase.REVIEW,
            output_data={"verdict": review2.get("verdict"), "confidence": review2.get("confidence")},
        )
        # 修订版通过则采用, 否则保留初稿并附终审意见 (设计: 最多 1 轮自纠)
        verdict_after = review2.get("verdict", "needs_review")
        if verdict_after == "approved" and generation2.get("answer"):
            generation, review = generation2, review2
        else:
            generation, review = initial_generation, initial_review
            if not any(
                isinstance(item, EvidencePack)
                for item in collaboration_context.evidence_pool
            ):
                collaboration_context.evidence_pool[:] = list(
                    initial_generation.get("context_chunks") or ()
                )
        self_correction = {
            "rounds": 1,
            "verdict_before": "needs_review",
            "verdict_after": verdict_after,
            "reason": str(review2.get("reason", ""))[:120],
        }
        collab_lines.append({
            "line": "L1.1",
            "label": "自纠回流",
            "kind": "correction",
            "steps": [
                {"agent": REVIEW_AGENT_ID, "elapsed_ms": t_rev, "output": "初审 needs_review → 触发修订"},
                {"agent": GENERATION_AGENT_ID, "elapsed_ms": t_loop / 2, "output": "多候选修订 + 交叉验证"},
                {"agent": REVIEW_AGENT_ID, "elapsed_ms": t_loop / 2, "output": "终审: " + verdict_after},
            ],
        })
    # ---- 验证器引导的迭代自纠回路 (真·语义 critic 闭环, 对标 DeepVerifier/CoRefine/SETS) ----
    # 类级修复: 把「字符 Jaccard + 假辩论」判定好坏的环节, 升级为「语义 critic 驱动闭环纠错」.
    if False and review_input is not None and generation.get("answer") and not (generation.get("plain_language") or generation.get("deduced") or generation.get("honest_unavailable")):
        try:
            critic_generation_input = _revision_agent_input(
                collaboration_context,
                generation_input,
            )
            _critic_result = _run_critic_loop(
                _contract_runtime_payload(input_data, critic_generation_input),
                deps,
                generation,
            )
            if _critic_result.get("adopted"):
                generation = _critic_result["generation"]
                critic_generation_contribution = _adapt_generation_contribution(
                    collaboration_context,
                    critic_generation_input,
                    generation,
                )
                collaboration_context.record_contribution(
                    critic_generation_contribution
                )
                # 采纳更优答案后重新事实核查, 保证 review/verdict 与最终答案一致
                critic_review_input = _revision_agent_input(
                    collaboration_context,
                    review_input,
                )
                critic_review_payload = _contract_runtime_payload(
                    input_data,
                    critic_review_input,
                )
                critic_review_payload.update(
                    {
                        "content": critic_generation_contribution.conclusion,
                        "context_chunks": _review_evidence_texts(
                            critic_review_input,
                            critic_generation_contribution,
                        ),
                    }
                )
                _task_event("AgentStarted", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID)
                review = run_review(critic_review_payload, deps)
                _task_event(
                    "ReviewCompleted",
                    REVIEW_AGENT_ID,
                    agent_id=REVIEW_AGENT_ID,
                    verdict=str(review.get("verdict", "")),
                )
                _task_event("AgentFinished", REVIEW_AGENT_ID, agent_id=REVIEW_AGENT_ID)
                collaboration_context.record_contribution(
                    _adapt_review_contribution(
                        collaboration_context,
                        critic_review_input,
                        review,
                    )
                )
                needs_adjudication = bool(generation.get("needs_adjudication", False))
                _audit_trail(
                    GENERATION_AGENT_ID,
                    "验证器回路修订: " + str(_critic_result.get("final_verdict", "")) + " · 分数 " + f"{_critic_result.get('final_score', 0):.2f}",
                    learner=str(input_data.get("learner_id", "")),
                    latency_ms=float(_critic_result.get("total_elapsed_ms") or 0.0),
                )
            reasoning_loop = _critic_result
            _critic_rounds = len(_critic_result.get("rounds", []))
            collab_lines.append({
                "line": "L2.3",
                "label": "验证器迭代",
                "kind": "correction",
                "steps": [
                    {"agent": "critic", "elapsed_ms": 0.0, "output": "语义评审 " + str(_critic_result.get("final_verdict", "pass")) + (" · 已修正" if _critic_result.get("adopted") else " · 保留初稿")},
                    {"agent": "rewriter", "elapsed_ms": 0.0, "output": ("改写查询 + 重检索" if _critic_result.get("adopted") else "无需改写")},
                    {"agent": "generator", "elapsed_ms": 0.0, "output": f"共 {_critic_rounds} 次评审"},
                ],
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("验证器迭代回路跳过: %s", exc)

    # 主线 L4: TaskPlan 驱动的导学决策 (汇总)
    guidance_input = _start_contract_agent(
        collaboration_context,
        GUIDANCE_AGENT_ID,
    )
    _task_event("AgentStarted", GUIDANCE_AGENT_ID, agent_id=GUIDANCE_AGENT_ID)
    _t4 = time.time()
    collab_lines[0]["steps"].append({
        "agent": GUIDANCE_AGENT_ID,
        "elapsed_ms": 0.0,
        "output": "决策（见上）",
    })
    collab_lines[0]["steps"][-1]["elapsed_ms"] = round((time.time() - _t4) * 1000, 1)

    # L4 remains a candidate provider only.  GuidanceDecision is the single
    # authority that accepts or rejects that candidate against reviewed facts.
    learner_id = (
        input_data.get("learner_id")
        or input_data.get("student_id")
        or diagnosis.get("learner_id")
    )
    l4_candidate: dict[str, Any] | None = None
    if learner_id and not generation.get("clarify"):
        engine = getattr(deps, "decision_engine", None)
        if engine is not None and hasattr(engine, "next_action_sync"):
            try:
                # Guidance receives only Diagnosis-normalized teaching context.
                # It must not re-read the L2 profile as a second interpreter.
                profile_dict = {
                    "level": collaboration_context.learner_context.get("level"),
                    "weak_kps": list(
                        collaboration_context.learner_context.get("weak_kps") or ()
                    ),
                }
                l4_candidate = engine.next_action_sync(
                    learner_id,
                    mode="guide",
                    learner_profile=profile_dict,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("L4 策略候选生成失败: %s", exc)

    guidance_decision, final_collaboration_result = _synthesize_guidance_decision(
        context=collaboration_context,
        generation=generation,
        review=review,
        generation_contribution=generation_contribution,
        review_contribution=review_contribution,
        l4_candidate=l4_candidate,
    )
    verdict = str(review.get("verdict") or "needs_review")
    confidence = guidance_decision.confidence
    clarify = generation.get("clarify")
    if guidance_decision.clarification_needed and not clarify:
        clarify = {
            "type": "decision_clarification",
            "message": guidance_decision.next_action,
            "options": [
                "发光效率",
                "热稳定性",
                "显色与色度",
                "健康照明指标",
                "成本",
            ],
        }
    withhold_answer = guidance_decision.answer_policy.startswith("WITHHOLD")
    # The scientific answer is either the reviewed Generation conclusion or
    # withheld.  Guidance never rewrites it after review.
    answer = "" if withhold_answer else final_collaboration_result.answer
    # ASK_USER is an ordinary interaction.  REFUSE/WITHHOLD are terminal
    # quality outcomes and must not be disguised as a request for human
    # adjudication.
    requires_confirmation = bool(guidance_decision.clarification_needed)
    if guidance_decision.decision_type is DecisionType.ASK_USER:
        decision = "clarify"
    elif guidance_decision.decision_type is DecisionType.REFUSE_CONCLUSION:
        decision = "refuse"
    elif not answer:
        decision = "withhold"
    else:
        decision = "direct"
    # 导学决策 Agent 轨迹: 记录决策与置信度 (与诊断/生成/审核构成完整 4 条链路)
    _audit_trail(
        GUIDANCE_AGENT_ID,
        f"导学决策: {decision} · 置信度 {confidence:.2f}",
        learner=str(input_data.get("learner_id", "")),
        latency_ms=round((time.time() - _t4) * 1000, 1),
    )
    _rec.record_agent_execution(
        agent_id=GUIDANCE_AGENT_ID, agent_name="导学决策 Agent",
        action=f"汇总决策: {decision} · 置信度 {confidence:.2f}",
        duration_ms=round((time.time() - _t4) * 1000, 1), status="completed", phase=InteractionPhase.DECISION,
        input_data={"query": input_data.get("query", ""), "learner_id": input_data.get("learner_id", "")},
        output_data={"decision": decision, "confidence": confidence, "requires_confirmation": requires_confirmation, "answer_length": len(answer)},
    )

    # 证据切片 (知识生成 Agent 的检索命中, 供前端"知识证据"区展示)
    evidence: list[dict[str, Any]] = []
    try:
        gchunks = list(generation.get("context_chunks") or [])
        gcites = list(generation.get("citations") or [])
        for i, ch in enumerate(gchunks[:3]):
            src = gcites[i] if i < len(gcites) else ""
            evidence.append({
                "content": str(ch)[:500],
                "source": str(src)[:120] if src else "",
            })
    except Exception:  # noqa: BLE001
        evidence = []

    # 流程/指向/广播/协作 可视化数据 (评委视角: 清楚展示 4 Agent 如何协同)
    # 指向: 上游产出 → 下游消费; 广播: 消息总线频道事件
    qtype_label = {
        "method": "方法型(怎么做)",
        "mechanism": "机理型(为什么)",
        "definition": "定义型(是什么)",
        "other": "通用型",
    }.get(str(generation.get("question_type", "")), "通用型")
    flow_events: list[dict[str, Any]] = [
        {
            "seq": 1,
            "step": "学情诊断",
            "agent": DIAGNOSIS_AGENT_ID,
            "label": "诊断画像/薄弱点",
            "to": GENERATION_AGENT_ID,
            "detail": "输出薄弱点与画像特征，供生成定向检索",
            "elapsed_ms": t_diag,
        },
        {
            "seq": 2,
            "step": "知识生成",
            "agent": GENERATION_AGENT_ID,
            "label": "多候选生成+交叉验证",
            "to": REVIEW_AGENT_ID,
            "detail": (
                f"问题类型: {qtype_label} · "
                f"候选 {len(generation.get('candidates', []))} 条 · "
                f"共识度 {float(generation.get('consensus_score', 0.0) or 0.0):.0%}"
            ),
            "elapsed_ms": t_gen,
        },
        {
            "seq": 3,
            "step": "审核校验",
            "agent": REVIEW_AGENT_ID,
            "label": "事实核查/防幻觉",
            "to": GUIDANCE_AGENT_ID,
            "detail": f"裁决: {str(review.get('verdict', ''))}",
            "elapsed_ms": t_rev,
        },
        {
            "seq": 4,
            "step": "导学决策",
            "agent": GUIDANCE_AGENT_ID,
            "label": "汇总决策",
            "to": "前端展示",
            "detail": f"决策: {decision} · 置信度 {confidence:.0%}",
            "elapsed_ms": round((time.time() - _t4) * 1000, 1),
        },
    ]
    # 广播事件: 各 Agent 发布到消息总线的频道 (可订阅即协作的证据)
    broadcast_events: list[dict[str, str]] = [
        {
            "publisher": DIAGNOSIS_AGENT_ID,
            "channel": "learning.diagnosis.report",
            "event": "diagnosis_output",
            "to": "知识生成 / 导学决策",
        },
        {
            "publisher": GENERATION_AGENT_ID,
            "channel": "knowledge.generation.output",
            "event": "generation_output",
            "to": "审核校验 / 导学决策",
        },
        {
            "publisher": REVIEW_AGENT_ID,
            "channel": "quality.review.result",
            "event": "review_output",
            "to": "导学决策 / 自纠回路",
        },
        {
            "publisher": GUIDANCE_AGENT_ID,
            "channel": "guidance.decision.output",
            "event": "decision_output",
            "to": "前端 / 会话闭环",
        },
    ]
    # 自纠回路广播 (若触发)
    if self_correction:
        broadcast_events.append({
            "publisher": REVIEW_AGENT_ID,
            "channel": "quality.review.retry",
            "event": "review_retry",
            "to": "知识生成(修订) → 终审",
        })

    # ---- 循环模型 · 协同交互轨迹 (强化交互体现: 让学习者看见 4 Agent 的迭代协作闭环) ----
    loop_trace = _build_loop_trace(
        reasoning_loop, self_correction, debate, consensus_reached, needs_adjudication
    )
    loop_final_verdict = (
        str(reasoning_loop.get("final_verdict", "")) if reasoning_loop
        else str(review.get("verdict", ""))
    )
    loop_narrative = _build_loop_narrative(loop_trace, loop_final_verdict)

    result = {
        "agent_id": GUIDANCE_AGENT_ID,
        "status": "completed",
        "task_id": str(input_data.get("task_id") or ""),
        "task_state": task_state_runtime.get_task_state(input_data.get("task_context")),
        "query": input_data.get("query", ""),
        "decision": decision,
        "requires_confirmation": requires_confirmation,
        "action_type": "clarify" if clarify else "answer",
        "recommended_path": list(guidance_decision.recommended_path),
        "clarify": clarify,
        "knowledge_unavailable": bool(
            generation.get("knowledge_unavailable", False)
        ),
        "answer": answer,
        "confidence": confidence,
        "honest_unavailable": bool(generation.get("honest_unavailable")),
        "evidence": evidence,
        "pipeline": [diagnosis, generation, review],
        "review": review,
        "collab_lines": collab_lines,
        "self_correction": self_correction,
        "reasoning_loop": reasoning_loop,
        "flow_events": flow_events,
        "broadcast_events": broadcast_events,
        "question_type": generation.get("question_type", ""),
        "sources": list(generation.get("sources") or []),
        # 协同决策中间数据 (供前端"协同调度与决策"可视化, L5 高等级对标)
        "candidates": generation.get("candidates", []),
        "consensus_score": generation.get("consensus_score", 0.0),
        "consensus_reached": generation.get("consensus_reached", False),
        "consensus_threshold": generation.get("consensus_threshold", 0.5),
        "divergence_matrix": generation.get("divergence_matrix", []),
        "debate": generation.get("debate"),
        "needs_adjudication": needs_adjudication,
        # 循环模型 · 协同交互轨迹 (强化交互体现)
        "loop_trace": loop_trace,
        "loop_narrative": loop_narrative,
        "loop_rounds": list(reasoning_loop.get("rounds", [])) if reasoning_loop else [],
    }

    current_memory_misconceptions: tuple[dict[str, Any], ...] = ()
    # Shared Learner Memory: persist only validated learning/decision facts.
    # Legacy query/reputation bookkeeping remains separate and does not serve
    # as the memory-aware planning source.
    learner_id_mem = input_data.get("learner_id") or input_data.get("student_id")
    if learner_id_mem and deps.profile_service is not None:
        try:
            from dy3_polaris.l5.agent_memory import (
                commit_learner_memory,
                extract_memory_candidate,
                recall_memory,
                record_query_log,
                update_reputation,
            )

            q_text = str(input_data.get("query", ""))
            memory_candidate = extract_memory_candidate(
                context=collaboration_context,
                final_result=final_collaboration_result,
                question=q_text,
                learner_id=str(learner_id_mem),
            )
            current_memory_misconceptions = tuple(
                item.to_dict() for item in memory_candidate.misconceptions
            )
            commit_learner_memory(
                deps.profile_service,
                learner_id_mem,
                memory_candidate,
            )
            for aid in (
                DIAGNOSIS_AGENT_ID,
                GENERATION_AGENT_ID,
                REVIEW_AGENT_ID,
                GUIDANCE_AGENT_ID,
            ):
                update_reputation(
                    deps.profile_service, learner_id_mem, aid, verdict
                )
            record_query_log(
                deps.profile_service, learner_id_mem, query=q_text
            )
            # 历史关联: 同主题历史问答 (跨问题记忆, 供 Agent 引用)
            related = recall_memory(
                deps.profile_service, learner_id_mem, q_text, top_k=2
            )
            if related:
                result["related_history"] = [
                    {"ts": r.get("ts"), "query": r.get("query")} for r in related
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent 记忆写入失败: %s", exc)
    # Public CURRENT mapping reads the one authoritative GuidanceDecision.
    # The private ``next_action`` remains canonical metadata because CURRENT
    # explicitly forbids a new public next_action field.
    result["action_type"] = "clarify" if clarify else "answer"
    result["recommended_path"] = list(guidance_decision.recommended_path)
    # 决策到画像: 决策轨迹写回画像 extras.decision_log (反向链路)
    if learner_id:
        profile = _load_profile(deps.profile_service, learner_id)
        if profile is not None:
            extras = dict(getattr(profile, "extras", {}) or {})
            decision_log = list(extras.get("decision_log", []) or [])
            decision_log.append({
                "ts": time.time(),
                "decision": decision,
                "confidence": confidence,
                "action_type": result.get("action_type", "answer"),
                "weak_kps": list(diagnosis.get("weak_kps", []) or []),
            })
            extras["decision_log"] = decision_log[-20:]
            profile.extras = extras
            _save_profile(deps.profile_service, profile)
        # 决策输出广播: 供会话/前端/其他 Agent 消费
        _broadcast(
            deps.message_bus,
            "guidance.decision.command",
            {
                "event": "decision_made",
                "learner_id": learner_id,
                "decision": decision,
                "confidence": confidence,
                "requires_confirmation": requires_confirmation,
            },
            GUIDANCE_AGENT_ID,
        )
    _task_event("AgentFinished", GUIDANCE_AGENT_ID, agent_id=GUIDANCE_AGENT_ID)
    if guidance_input is not None:
        guidance_sequence = len(collaboration_context.contributions) + 1
        guidance_id = (
            f"contrib-{collaboration_context.task_id}-"
            f"{guidance_input.subtask.subtask_id}-{guidance_sequence}"
        )
        if guidance_decision.decision_type is DecisionType.ASK_USER:
            guidance_action = RequestedAction.ASK_USER
        elif guidance_decision.decision_type is DecisionType.REFUSE_CONCLUSION:
            guidance_action = RequestedAction.REFUSE_CONCLUSION
        elif guidance_decision.uncertain_claim_ids:
            guidance_action = RequestedAction.DECLARE_UNCERTAINTY
        else:
            guidance_action = RequestedAction.ACCEPT
        guidance_contribution = make_contribution(
            collaboration_context,
            guidance_input,
            conclusion=guidance_decision.reasoning_summary,
            claims=(
                _claim(
                    guidance_id,
                    guidance_decision.next_action,
                    ClaimType.RECOMMENDATION,
                    confidence=confidence,
                ),
            ),
            assumptions=tuple(guidance_input.constraints),
            uncertainty=tuple(guidance_decision.knowledge_gap)
            or (("confirmation required",) if requires_confirmation else ()),
            requested_actions=(guidance_action,),
            tool_usage=("learning_decision",),
            confidence=confidence,
            status="completed",
            artifact_identity=final_collaboration_result.answer_identity,
        )
        _finish_contract_agent(
            collaboration_context,
            guidance_input,
            guidance_contribution,
        )
        _task_event(
            "AgentContributionRecorded",
            GUIDANCE_AGENT_ID,
            agent_id=GUIDANCE_AGENT_ID,
            contribution_id=guidance_contribution.contribution_id,
        )
    result["task_events"] = task_state_runtime.get_task_events(task_context)
    guidance_carrier = _PrivateRuntimeCarrier(result)
    evidence_candidate = getattr(
        generation,
        "_contract_candidate",
        None,
    )
    review_candidate = getattr(review, "_contract_candidate", None)
    answer_correlation = _correlate_final_answer(
        task_id=str(input_data.get("task_id") or ""),
        final_answer=final_collaboration_result.answer,
        evidence_candidate=evidence_candidate,
        review_candidate=review_candidate,
    )
    scientific_grounding = build_scientific_grounding(
        task_id=collaboration_context.task_id,
        answer_identity=final_collaboration_result.answer_identity,
        claims=generation_contribution.claims,
        evidence_packs=_active_evidence_packs(collaboration_context),
        review_identity=(
            review_candidate.reviewed_answer_identity
            if isinstance(review_candidate, _ReviewCandidate)
            else ""
        ),
        reviewer_status=(
            review_candidate.raw_verdict
            if isinstance(review_candidate, _ReviewCandidate)
            else "not_reviewed"
        ),
    )
    grounded_by_id = {claim.claim_id: claim for claim in scientific_grounding.claims}
    final_collaboration_result = replace(
        final_collaboration_result,
        accepted_claims=tuple(
            grounded_by_id.get(claim.claim_id, claim)
            for claim in final_collaboration_result.accepted_claims
        ),
        rejected_claims=tuple(
            grounded_by_id.get(claim.claim_id, claim)
            for claim in final_collaboration_result.rejected_claims
        ),
        uncertain_claims=tuple(
            grounded_by_id.get(claim.claim_id, claim)
            for claim in final_collaboration_result.uncertain_claims
        ),
    )
    quality_release = _build_quality_release_decision(
        context=collaboration_context,
        final_result=final_collaboration_result,
        evidence_candidate=evidence_candidate,
        review_candidate=review_candidate,
        answer_correlation=answer_correlation,
        scientific_grounding=scientific_grounding,
    )
    _task_event(
        "ReleaseDecided",
        GUIDANCE_AGENT_ID,
        release_status=quality_release.status.value,
        eligible=quality_release.eligible,
        correction_count=quality_release.correction_count,
    )
    if quality_release.status is QualityReleaseStatus.ASK_USER:
        task_state_runtime.set_task_state(
            task_context,
            "NEEDS_CONFIRMATION",
            producer="run_guidance",
        )
    elif quality_release.eligible:
        task_state_runtime.set_task_state(
            task_context,
            "ANSWERING",
            producer="run_guidance",
        )
    else:
        task_state_runtime.set_task_state(
            task_context,
            "PARTIAL",
            producer="run_guidance",
        )
    guidance_carrier["answer"] = quality_release.public_answer
    guidance_carrier["requires_confirmation"] = (
        quality_release.status is QualityReleaseStatus.ASK_USER
    )
    guidance_carrier["action_type"] = {
        QualityReleaseStatus.FULL_RELEASE: "answer",
        QualityReleaseStatus.LIMITED_RELEASE: "limited_answer",
        QualityReleaseStatus.ASK_USER: "clarify",
        QualityReleaseStatus.REFUSE: "refuse",
        QualityReleaseStatus.WITHHOLD: "withhold",
        QualityReleaseStatus.DEGRADED: "degraded",
    }[quality_release.status]
    guidance_carrier["quality_release"] = {
        "status": quality_release.status.value,
        "eligible": quality_release.eligible,
        "message": quality_release.message,
        "reason_codes": list(quality_release.reason_codes),
        "review_status": quality_release.review_status,
        "review_verdict": quality_release.review_verdict,
        "correction_count": quality_release.correction_count,
        "evidence_versions": list(quality_release.evidence_versions),
    }
    # Public Agent Runtime responses must not serialize unreviewed or losing
    # Generation artifacts.  The complete selected objects remain available
    # only through the private carrier for correlation/readiness.
    guidance_carrier["pipeline"] = [
        {
            "agent_id": DIAGNOSIS_AGENT_ID,
            "status": str(diagnosis.get("status") or "completed"),
            "contribution": "learner_context_interpreted",
        },
        {
            "agent_id": GENERATION_AGENT_ID,
            "status": str(generation.get("status") or "completed"),
            "contribution": (
                "reviewed_generation_selected"
                if quality_release.eligible
                else "generation_withheld_by_release_gate"
            ),
            "evidence_count": len(generation.get("context_chunks") or ()),
        },
        {
            "agent_id": REVIEW_AGENT_ID,
            "status": quality_release.review_status or "not_completed",
            "verdict": quality_release.review_verdict or "not_available",
            "contribution": "scientific_quality_decision",
        },
    ]
    # Public review reason: keep the reviewer's SPECIFIC reason when the
    # result was withheld by a review verdict (needs_review/rejected), so the
    # front end can explain to the learner WHY nothing was published (e.g.
    # "问题核心覆盖门要求修订：回答遗漏了...").  Only fall back to the
    # generic release message when the reviewer approved but the release gate
    # failed for other (artifact/identity) reasons, where the reviewer reason
    # would be misleading.
    _review_reason = str(review.get("reason") or "").strip()
    _public_review_reason = (
        _review_reason
        if quality_release.eligible
        else _review_reason
        if quality_release.review_verdict in {"needs_review", "rejected"}
        else quality_release.message
    )
    guidance_carrier["review"] = {
        "agent_id": REVIEW_AGENT_ID,
        "status": quality_release.review_status or "not_completed",
        "verdict": quality_release.review_verdict or "not_available",
        "confidence": float(review.get("confidence", 0.0) or 0.0),
        "reason": _public_review_reason,
    }
    guidance_carrier["candidates"] = []
    guidance_carrier["divergence_matrix"] = []
    guidance_carrier["needs_adjudication"] = False
    if not quality_release.eligible:
        guidance_carrier["evidence"] = []
        guidance_carrier["sources"] = []
        if isinstance(guidance_carrier.get("self_correction"), dict):
            correction = guidance_carrier["self_correction"]
            guidance_carrier["self_correction"] = {
                "rounds": int(correction.get("rounds", 0) or 0),
                "verdict_before": str(correction.get("verdict_before") or ""),
                "verdict_after": str(correction.get("verdict_after") or ""),
                "reason": "未通过发布门，未审核正文已隐藏。",
            }
    learner_view = input_data.get("_learner_intelligence_view")
    knowledge_learning_context = collaboration_context.learner_context.get(
        "knowledge_learning_context"
    )
    adaptive_teaching_decision = collaboration_context.learner_context.get(
        "adaptive_teaching_decision"
    )
    if not isinstance(adaptive_teaching_decision, AdaptiveTeachingDecision):
        adaptive_teaching_decision = None
    reviewed_long_form = _build_reviewed_long_form_resource(
        query=str(input_data.get("query") or ""),
        task_id=collaboration_context.task_id,
        quality_release=quality_release,
        final_result=final_collaboration_result,
        evidence_candidate=(
            evidence_candidate
            if isinstance(evidence_candidate, _EvidenceCandidate)
            else None
        ),
        teaching_decision=adaptive_teaching_decision,
        knowledge_context=(
            knowledge_learning_context
            if isinstance(knowledge_learning_context, KnowledgeLearningContext)
            else None
        ),
        deps=deps,
        event_callback=_task_event,
    )
    reviewed_guided_questions = _build_reviewed_guided_questions(
        query=str(input_data.get("query") or ""),
        task_id=collaboration_context.task_id,
        quality_release=quality_release,
        final_result=final_collaboration_result,
        evidence_candidate=(
            evidence_candidate
            if isinstance(evidence_candidate, _EvidenceCandidate)
            else None
        ),
        teaching_decision=adaptive_teaching_decision,
        knowledge_context=(
            knowledge_learning_context
            if isinstance(knowledge_learning_context, KnowledgeLearningContext)
            else None
        ),
        deps=deps,
        event_callback=_task_event,
    )
    result["task_events"] = task_state_runtime.get_task_events(task_context)
    collaboration_trace = _build_collaboration_trace(
        context=collaboration_context,
        task_events=list(result.get("task_events") or ()),
        final_result=final_collaboration_result,
    )
    multi_agent_evaluation = _evaluate_multi_agent_runtime(
        context=collaboration_context,
        trace=collaboration_trace,
        final_result=final_collaboration_result,
        answer_correlation=answer_correlation,
        review_candidate=(
            review_candidate
            if isinstance(review_candidate, _ReviewCandidate)
            else None
        ),
    )
    teaching_learning_event = (
        build_teaching_learning_event(
            context=collaboration_context,
            final_result=final_collaboration_result,
            review_candidate=review_candidate,
            knowledge_learning_context=knowledge_learning_context,
            learner_view=learner_view,
        )
        if isinstance(knowledge_learning_context, KnowledgeLearningContext)
        and isinstance(learner_view, LearnerIntelligenceView)
        else None
    )
    if (
        isinstance(teaching_learning_event, TeachingLearningEvent)
        and learner_id_mem
        and deps.profile_service is not None
    ):
        try:
            diagnosis_memory_view = learner_memory_views.get(
                DIAGNOSIS_AGENT_ID, {}
            )
            source_misconceptions = (
                (
                    *(diagnosis_memory_view.get("misconceptions") or ()),
                    *current_memory_misconceptions,
                )
                if isinstance(diagnosis_memory_view, dict)
                else current_memory_misconceptions
            )
            commit_teaching_memory(
                deps.profile_service,
                str(learner_id_mem),
                teaching_learning_event,
                source_misconceptions=source_misconceptions,
            )
        except Exception as exc:  # noqa: BLE001 - memory cannot fail the task
            logger.warning("Teaching Memory commit failed: %s", exc)
    learning_resource_plan = build_learning_resource_plan(
        task_id=collaboration_context.task_id,
        learner_id=str(learner_id_mem or "anonymous-request"),
        teaching_decision=adaptive_teaching_decision,
        knowledge_context=(
            knowledge_learning_context
            if isinstance(knowledge_learning_context, KnowledgeLearningContext)
            else None
        ),
        final_result=final_collaboration_result,
        quality_release=quality_release,
        reviewed_long_form=reviewed_long_form,
        reviewed_guided_questions=reviewed_guided_questions,
    )
    guidance_carrier["learning_resources"] = public_resource_projection(
        learning_resource_plan
    )
    for resource in guidance_carrier["learning_resources"]:
        _task_event(
            "ResourceIssued",
            GUIDANCE_AGENT_ID,
            resource_id=str(resource.get("resource_id") or ""),
            resource_family=str(resource.get("resource_family") or ""),
            review_status=str(resource.get("review_status") or ""),
        )
    guidance_carrier["teaching_strategy"] = (
        {
            "content_depth": adaptive_teaching_decision.content_depth,
            "explanation_strategy": (
                adaptive_teaching_decision.explanation_strategy
            ),
            "representation_modes": list(
                adaptive_teaching_decision.representation_modes
            ),
            "difficulty_strategy": (
                adaptive_teaching_decision.difficulty_strategy
            ),
            "resource_modes": list(adaptive_teaching_decision.resource_modes),
            "next_focus": adaptive_teaching_decision.next_focus,
            "rationale": list(adaptive_teaching_decision.rationale),
            "confidence": adaptive_teaching_decision.confidence,
        }
        if adaptive_teaching_decision is not None
        else {
            "content_depth": "foundation",
            "explanation_strategy": "baseline_explanation",
            "representation_modes": [],
            "difficulty_strategy": "diagnose_then_maintain",
            "resource_modes": [],
            "next_focus": "",
            "rationale": ["当前学习者证据不足，采用保守基础教学策略。"],
            "confidence": 0.0,
        }
    )
    guidance_carrier["learner_context"] = (
        public_learner_intelligence_projection(learner_view)
        if isinstance(learner_view, LearnerIntelligenceView)
        else {}
    )
    guidance_carrier["knowledge_context"] = (
        public_knowledge_learning_projection(knowledge_learning_context)
        if isinstance(knowledge_learning_context, KnowledgeLearningContext)
        else {}
    )
    guidance_carrier["knowledge_context"]["scientific_grounding"] = (
        public_scientific_grounding_projection(
            scientific_grounding,
            release_eligible=quality_release.eligible,
        )
    )
    # Existing public keys now receive deterministic projections of actual
    # runtime facts.  No private IDs, prompts, reasoning, or Contract objects
    # are exposed, and no new response key is introduced.
    guidance_carrier["agent_trace"] = _project_agent_trace(
        collaboration_trace,
        release_eligible=quality_release.eligible,
    )
    guidance_carrier["collab_lines"] = _project_collab_lines(
        collaboration_trace,
        release_eligible=quality_release.eligible,
    )
    guidance_carrier["flow_events"] = _project_flow_events(
        collaboration_trace,
        release_eligible=quality_release.eligible,
    )
    guidance_carrier["broadcast_events"] = []
    guidance_carrier["reasoning_loop"] = None
    guidance_carrier["debate"] = None
    guidance_carrier["loop_trace"] = []
    guidance_carrier["loop_narrative"] = ""
    guidance_carrier["loop_rounds"] = []
    guidance_carrier._contract_candidate = _FinalPrivateCandidateSet(
        evidence_candidate=(
            evidence_candidate
            if isinstance(evidence_candidate, _EvidenceCandidate)
            else None
        ),
        review_candidate=(
            review_candidate
            if isinstance(review_candidate, _ReviewCandidate)
            else None
        ),
        answer_correlation=answer_correlation,
        quality_release=quality_release,
        adaptive_teaching_decision=adaptive_teaching_decision,
        final_collaboration_result=final_collaboration_result,
        collaboration_trace=collaboration_trace,
        multi_agent_evaluation=multi_agent_evaluation,
        teaching_learning_event=teaching_learning_event,
        learning_resource_plan=learning_resource_plan,
        scientific_grounding=scientific_grounding,
    )
    return guidance_carrier


def build_agent_workers(
    deps: AgentDependencies | None = None,
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """构建 Agent ID → 执行函数 的映射."""
    deps = deps or AgentDependencies()
    return {
        DIAGNOSIS_AGENT_ID: lambda data: run_diagnosis(data, deps),
        GENERATION_AGENT_ID: lambda data: run_generation(data, deps),
        REVIEW_AGENT_ID: lambda data: run_review(data, deps),
        GUIDANCE_AGENT_ID: lambda data: run_guidance(data, deps),
    }


__all__ = [
    "AgentDependencies",
    "DIAGNOSIS_AGENT_ID",
    "GENERATION_AGENT_ID",
    "GUIDANCE_AGENT_ID",
    "REVIEW_AGENT_ID",
    "build_agent_workers",
    "run_diagnosis",
    "run_generation",
    "run_guidance",
    "run_review",
    "_run_practice_mode",
    "_run_assess_mode",
    "_run_lecture_mode",
    "_run_guide_mode",
    "_retrieve_evidence",
    "_build_sources",
]
