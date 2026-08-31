"""R09A learner lifecycle, persona prior and adaptive teaching validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import SimpleNamespace
from typing import Any

from starlette.testclient import TestClient

from dy3_polaris.l2.models import LearnerSnapshot
from dy3_polaris.l2.profile_builder.tracing_service import ProfileTracingService
from dy3_polaris.l2.store import InMemoryL2Store
from dy3_polaris.l2.user_understanding.service import UserUnderstandingService
from dy3_polaris.l3.concept_foundation import build_concept_foundation
from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.learner_foundation import (
    AdaptiveTeachingDecision,
    LearnerLifecycleStage,
    LearnerPersonaPrototype,
    PersonalLearnerModel,
    build_persona_prototype,
)
from dy3_polaris.l5.learner_intelligence import build_learner_intelligence_view
from dy3_polaris.l5.task_understanding import understand_task
from dy3_polaris.l5.unified_app import UnifiedApp


@dataclass
class _Profile:
    learner_id: str
    theta: float | None = None
    level: str = "unknown"
    weak_kps: list[str] = field(default_factory=list)
    kp_mastery: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.8
    snapshot_ts: float = 100.0
    extras: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "theta": self.theta,
            "level": self.level,
            "weak_kps": list(self.weak_kps),
            "kp_mastery": dict(self.kp_mastery),
            "confidence": self.confidence,
            "snapshot_ts": self.snapshot_ts,
            "extras": dict(self.extras),
            "version": self.version,
        }


class _ProfileService:
    def __init__(self, profile: _Profile | None) -> None:
        self.profile = profile
        self.store = SimpleNamespace(get_answer_history=lambda _learner_id: [])

    def get_profile_snapshot(self, _learner_id: str) -> _Profile | None:
        return self.profile

    def get_weak_points(self, _learner_id: str) -> dict[str, Any]:
        return {"weak_kps": list(self.profile.weak_kps) if self.profile else []}


class _IRTService:
    def __init__(self, theta: float, response_count: int) -> None:
        self.theta = theta
        self.response_count = response_count

    def get_ability_snapshot(self, learner_id: str) -> dict[str, Any]:
        return {
            "learner_id": learner_id,
            "theta": self.theta,
            "se": 0.15,
            "response_count": self.response_count,
            "last_update_time": 200.0,
        }


def _declare(
    service: UserUnderstandingService,
    learner_id: str,
    **values: str,
) -> None:
    for slot_key, value in values.items():
        service.answer(learner_id, {"slot_key": slot_key, "value": value})


def _view(payload: dict[str, Any], deps: AgentDependencies):
    return build_learner_intelligence_view(payload, deps)


def test_unknown_learner_has_no_fake_persona_or_mastery() -> None:
    view = _view(
        {"learner_id": "r09-unknown", "query": "Dy³⁺为什么有黄蓝双发射？"},
        AgentDependencies(),
    )

    persona = view.value("derived_context", "persona_prototype")
    model = view.value("derived_context", "personal_learner_model")
    decision = view.value("derived_context", "adaptive_teaching_decision")

    assert isinstance(persona, LearnerPersonaPrototype)
    assert isinstance(model, PersonalLearnerModel)
    assert isinstance(decision, AdaptiveTeachingDecision)
    assert persona.source_refs == ()
    assert persona.confidence == 0.0
    assert model.lifecycle_stage is LearnerLifecycleStage.UNKNOWN
    assert model.model_state_available is False
    assert model.diagnostic.needed is True
    assert decision.content_depth == "foundation"
    assert view.models["mastery"].value == {}


def test_optional_declared_profile_reuses_existing_l2_snapshot_storage(tmp_path) -> None:
    store = InMemoryL2Store(persist_dir=tmp_path)
    profile_service = ProfileTracingService(store=store)
    understanding = UserUnderstandingService(profile_service=profile_service)
    learner_id = "r09-persisted"

    _declare(
        understanding,
        learner_id,
        learning_stage="研究生阶段",
        learning_goal="阅读科研证据",
        professional_background="材料科学",
        domain_experience="有科研经历",
    )

    snapshot = profile_service.get_profile_snapshot(learner_id)
    assert isinstance(snapshot, LearnerSnapshot)
    assert snapshot.theta is None
    assert snapshot.level == "unknown"
    assert snapshot.kp_mastery == {}
    persisted = snapshot.extras["user_profile"]["declared_background"]
    assert persisted["learning_stage"] == "研究生阶段"
    assert persisted["learning_goal"] == "阅读科研证据"

    restored = UserUnderstandingService(profile_service=profile_service).get_profile(
        learner_id
    )
    assert restored is not None
    assert restored.declared_background == persisted


def test_frontend_stage_options_map_to_real_persona_priors() -> None:
    expected = {
        "本科阶段": "undergraduate",
        "研究生阶段": "graduate",
        "科研人员": "researcher",
        "行业从业者": "professional",
    }

    for declared, canonical in expected.items():
        persona = build_persona_prototype({
            "declared_background": {"learning_stage": declared},
            "confidence": 0.25,
        })
        assert persona.background["learning_stage"] == canonical
        assert persona.knowledge_priors["material_foundation"] > 0.0


def test_initial_questions_are_optional_bounded_and_do_not_repeat_skips() -> None:
    understanding = UserUnderstandingService(max_ask_per_session=3)
    learner_id = "r09-optional"

    first = understanding.ask(learner_id, {"initial_profile": True})
    assert first is not None
    assert first["slot_key"] == "learning_stage"
    assert first["optional"] is True
    understanding.answer(
        learner_id, {"slot_key": first["slot_key"], "value": "跳过"}
    )

    second = understanding.ask(learner_id, {"initial_profile": True})
    assert second is not None
    assert second["slot_key"] == "learning_goal"
    understanding.answer(
        learner_id,
        {"slot_key": second["slot_key"], "value": "理解基础概念"},
    )
    third = understanding.ask(learner_id, {"initial_profile": True})
    assert third is not None
    assert third["slot_key"] == "professional_background"
    assert understanding.ask(learner_id, {"initial_profile": True}) is None

    profile = understanding.get_profile(learner_id)
    assert profile is not None
    assert profile.declared_background["learning_stage"] == "skipped"
    assert profile.confidence < 0.5


def test_persona_prior_and_model_evidence_produce_different_teaching_decisions() -> None:
    query = "Dy³⁺为什么可以调节白光？"
    foundation_understanding = UserUnderstandingService()
    _declare(
        foundation_understanding,
        "r09-foundation",
        learning_stage="本科阶段",
        learning_goal="理解基础概念",
        professional_background="跨专业",
        domain_experience="刚开始了解",
    )
    foundation_view = _view(
        {"learner_id": "r09-foundation", "query": query},
        AgentDependencies(user_understanding_service=foundation_understanding),
    )

    all_kps = {
        kp_id
        for concept in build_concept_foundation().concepts.values()
        for kp_id in concept.related_kps
    }
    research_profile = _Profile(
        learner_id="r09-research",
        theta=1.6,
        level="advanced",
        kp_mastery={kp_id: 0.94 for kp_id in all_kps},
        confidence=0.94,
    )
    research_understanding = UserUnderstandingService()
    _declare(
        research_understanding,
        "r09-research",
        learning_stage="科研人员",
        learning_goal="阅读科研证据",
        professional_background="材料科学",
        domain_experience="有科研经历",
        representation_preference="论文证据",
    )
    research_view = _view(
        {"learner_id": "r09-research", "query": query},
        AgentDependencies(
            profile_service=_ProfileService(research_profile),
            irt_service=_IRTService(theta=1.6, response_count=12),
            user_understanding_service=research_understanding,
        ),
    )

    foundation_decision = foundation_view.value(
        "derived_context", "adaptive_teaching_decision"
    )
    research_decision = research_view.value(
        "derived_context", "adaptive_teaching_decision"
    )
    assert foundation_decision.content_depth == "foundation"
    assert foundation_decision.explanation_strategy in {
        "foundation_conceptual",
        "prerequisite_scaffolding",
    }
    assert research_decision.content_depth == "advanced"
    assert research_decision.explanation_strategy == "evidence_first_mechanism"
    assert "evidence" in research_decision.representation_modes
    assert foundation_view.value(
        "derived_context", "knowledge_learning_context"
    ).target_concepts == research_view.value(
        "derived_context", "knowledge_learning_context"
    ).target_concepts


def test_observed_weak_model_prevents_declared_research_prior_from_forcing_advanced() -> None:
    foundation = build_concept_foundation()
    all_kps = {
        kp_id
        for concept in foundation.concepts.values()
        for kp_id in concept.related_kps
    }
    profile = _Profile(
        learner_id="r09-prior-conflict",
        theta=-0.5,
        level="beginner",
        weak_kps=sorted(all_kps),
        kp_mastery={kp_id: 0.2 for kp_id in all_kps},
        confidence=0.9,
    )
    understanding = UserUnderstandingService()
    _declare(
        understanding,
        "r09-prior-conflict",
        learning_stage="科研人员",
        professional_background="材料科学",
        domain_experience="有科研经历",
    )
    view = _view(
        {"learner_id": "r09-prior-conflict", "query": "解释Dy³⁺黄蓝双发射"},
        AgentDependencies(
            profile_service=_ProfileService(profile),
            irt_service=_IRTService(theta=-0.5, response_count=8),
            user_understanding_service=understanding,
        ),
    )

    decision = view.value("derived_context", "adaptive_teaching_decision")
    assert decision.content_depth != "advanced"
    assert view.models["mastery"].value
    assert any(
        "weak" in reason or "prerequisite" in reason
        for reason in decision.rationale
    )


def test_diagnosis_is_the_only_agent_interpretation_boundary() -> None:
    understanding = UserUnderstandingService()
    _declare(
        understanding,
        "r09-boundary",
        learning_stage="本科阶段",
        learning_goal="理解基础概念",
    )
    payload = {
        "task_id": "task-r09-boundary",
        "learner_id": "r09-boundary",
        "query": "为什么Dy³⁺产生黄蓝双发射？",
    }
    deps = AgentDependencies(user_understanding_service=understanding)
    view = _view(payload, deps)
    context = initialize_collaboration_context(
        payload,
        intent_resolver=lambda query, **_kwargs: understand_task(query, use_llm=False),
    )
    diagnosis_input = agent_workers._start_contract_agent(
        context, agent_workers.DIAGNOSIS_AGENT_ID
    )
    assert diagnosis_input is not None
    diagnosis = agent_workers.run_diagnosis(
        {**payload, "_learner_intelligence_view": view}, deps
    )
    contribution = agent_workers._adapt_diagnosis_contribution(
        context, diagnosis_input, diagnosis
    )
    agent_workers._apply_diagnosis_teaching_context(context, view)
    agent_workers._finish_contract_agent(
        context, diagnosis_input, contribution
    )
    generation_input = agent_workers._start_contract_agent(
        context, agent_workers.GENERATION_AGENT_ID
    )
    assert generation_input is not None
    generation_payload = agent_workers._contract_runtime_payload(
        payload, generation_input
    )

    decision = view.value("derived_context", "adaptive_teaching_decision")
    assert diagnosis["level"] == decision.content_depth
    assert generation_payload["_adaptive_teaching_decision"] is decision
    assert generation_payload["learner_level"] == decision.content_depth
    assert "declared_background" not in generation_payload
    assert "persona_prototype" not in generation_payload


def test_current_query_schema_is_stable_and_private_foundation_does_not_leak(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    baseline = client.post(
        "/api/query",
        json={
            "query": "为什么Dy³⁺会产生黄蓝双发射？",
            "learner_id": "r09-api-baseline",
        },
    )
    assert baseline.status_code == 200

    for slot_key, value in (
        ("learning_stage", "研究生阶段"),
        ("learning_goal", "阅读科研证据"),
        ("professional_background", "材料科学"),
    ):
        saved = client.post(
            "/api/user-understanding/answer",
            json={
                "learner_id": "r09-api-profiled",
                "payload": {"slot_key": slot_key, "value": value},
            },
        )
        assert saved.status_code == 200

    saved_profile = saved.json()["data"]["profile"]
    assert saved_profile["declared_background"]["learning_stage"] == "研究生阶段"

    profiled = client.post(
        "/api/query",
        json={
            "query": "为什么Dy³⁺会产生黄蓝双发射？",
            "learner_id": "r09-api-profiled",
        },
    )
    assert profiled.status_code == 200
    baseline_data = baseline.json()["data"]
    profiled_data = profiled.json()["data"]
    assert set(profiled_data) == set(baseline_data)
    serialized = json.dumps(profiled_data, ensure_ascii=False, default=str)
    for forbidden in (
        "LearnerPersonaPrototype",
        "PersonalLearnerModel",
        "AdaptiveTeachingDecision",
        "_adaptive_teaching_decision",
        "persona_prototype",
        "learner_lifecycle_stage",
    ):
        assert forbidden not in serialized
