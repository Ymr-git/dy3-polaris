"""R-05A Learner Intelligence boundary and runtime validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from types import SimpleNamespace
from typing import Any

from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.learner_intelligence import (
    DERIVED,
    INFERRED,
    OBSERVED,
    LearnerIntelligenceView,
    build_learner_intelligence_view,
)
from dy3_polaris.l5.task_understanding import understand_task
from dy3_polaris.l5.unified_app import UnifiedApp


@dataclass
class _Profile:
    learner_id: str
    theta: float | None = None
    level: str = "unknown"
    weak_kps: list[str] = field(default_factory=list)
    kp_mastery: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.7
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
        return {
            "weak_kps": list(self.profile.weak_kps) if self.profile else []
        }


class _IRTService:
    def __init__(self, *, theta: float, se: float, response_count: int) -> None:
        self.snapshot = {
            "theta": theta,
            "se": se,
            "response_count": response_count,
            "last_update_time": 200.0 if response_count else 0.0,
        }

    def get_ability_snapshot(self, learner_id: str) -> dict[str, Any]:
        return {"learner_id": learner_id, **self.snapshot}


def _deps_with_profile() -> AgentDependencies:
    profile = _Profile(
        learner_id="learner-known",
        theta=1.2,
        level="advanced",
        weak_kps=["A-05"],
        kp_mastery={"A-05": 0.35, "A-06": 0.82},
        confidence=0.91,
    )
    return AgentDependencies(
        profile_service=_ProfileService(profile),
        irt_service=_IRTService(theta=1.2, se=0.2, response_count=8),
    )


def test_unknown_learner_remains_unknown_and_never_auto_advanced() -> None:
    deps = AgentDependencies()
    payload = {"learner_id": "learner-unknown", "query": "Dy³⁺是什么？"}

    view = build_learner_intelligence_view(payload, deps)
    diagnosis = agent_workers.run_diagnosis(
        {**payload, "_learner_intelligence_view": view}, deps
    )

    assert isinstance(view, LearnerIntelligenceView)
    assert view.value("derived_context", "learning_stage") == "unknown"
    assert view.value("derived_context", "recommended_depth") == "foundation"
    assert view.models["theta"].value is None
    assert diagnosis["level"] == "foundation"
    assert diagnosis["level"] != "advanced"


def test_profile_and_irt_weak_points_reach_diagnosis_as_normalized_context() -> None:
    deps = _deps_with_profile()
    payload = {"learner_id": "learner-known", "query": "解释Dy³⁺跃迁"}

    view = build_learner_intelligence_view(payload, deps)
    diagnosis = agent_workers.run_diagnosis(
        {**payload, "_learner_intelligence_view": view}, deps
    )

    assert view.models["mastery"].source_type == "profile_snapshot_bkt_projection"
    assert view.models["theta"].source_type == "irt_service"
    assert view.value("derived_context", "weak_kps") == ("A-05",)
    assert view.value("derived_context", "learning_stage") == "advanced"
    assert view.value("derived_context", "recommended_depth") == "intermediate"
    assert diagnosis["weak_kps"] == ["A-05"]
    assert diagnosis["level"] == "intermediate"


def test_memory_is_legacy_input_to_view_not_direct_agent_authority() -> None:
    data = {
        "task_id": "task-r05a-memory-isolation",
        "query": "如何调控Dy³⁺白光？",
        "learner_id": "learner-memory",
    }
    context = initialize_collaboration_context(
        data,
        intent_resolver=lambda query, **_kwargs: understand_task(
            query, use_llm=False
        ),
    )
    plan_before = context.task_plan
    learner_context_before = dict(context.learner_context)
    views = {
        agent_workers.DIAGNOSIS_AGENT_ID: {
            "memory_available": True,
            "recent_tasks": ("task-earlier",),
            "remaining_gap_labels": ("CIE 色度与色坐标",),
        },
        agent_workers.GENERATION_AGENT_ID: {
            "memory_available": True,
            "explanation_strategy": "advance_from_memory",
        },
        agent_workers.GUIDANCE_AGENT_ID: {
            "memory_available": True,
            "focus_topic_labels": ("相关色温 CCT",),
        },
    }

    agent_workers._apply_memory_to_collaboration_context(context, views)

    assert context.task_plan is plan_before
    assert context.learner_context == learner_context_before
    assert context.runtime_metadata["legacy_learner_memory_path"] == "isolated"

    view = build_learner_intelligence_view(
        data,
        AgentDependencies(),
        learner_memory_view=views[agent_workers.DIAGNOSIS_AGENT_ID],
    )
    base = {**data, "_learner_memory_views": views, "_learner_intelligence_view": view}
    diagnosis_input = agent_workers._start_contract_agent(
        context, agent_workers.DIAGNOSIS_AGENT_ID
    )
    assert diagnosis_input is not None
    diagnosis_payload = agent_workers._contract_runtime_payload(base, diagnosis_input)
    assert diagnosis_payload["_learner_intelligence_view"] is view
    assert "_learner_memory_view" not in diagnosis_payload

    for agent_id in (
        agent_workers.GENERATION_AGENT_ID,
        agent_workers.GUIDANCE_AGENT_ID,
    ):
        downstream = replace(diagnosis_input, agent_id=agent_id)
        projected = agent_workers._contract_runtime_payload(base, downstream)
        assert "_learner_intelligence_view" not in projected
        assert "_learner_memory_view" not in projected
        assert "_learner_memory_views" not in projected


def test_signal_classification_keeps_observed_inferred_and_derived_separate() -> None:
    view = build_learner_intelligence_view(
        {"learner_id": "learner-known", "query": "解释Dy³⁺跃迁"},
        _deps_with_profile(),
        learner_memory_view={
            "memory_available": True,
            "recent_tasks": ("task-observed",),
            "remaining_gap_labels": ("CIE 色度与色坐标",),
        },
    )

    assert {signal.classification for signal in view.facts.values()} == {OBSERVED}
    assert {signal.classification for signal in view.models.values()} == {INFERRED}
    assert {
        signal.classification for signal in view.derived_context.values()
    } == {DERIVED}
    assert view.models["confidence"].source_type == "profile_snapshot_mixed_semantics"
    assert view.models["confidence"].decision_eligible is False
    assert view.metadata["request_local"] is True


def test_api_response_keys_remain_current_and_private_view_does_not_leak(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )

    response = client.post(
        "/api/query",
        json={
            "query": "为什么Dy³⁺会产生黄蓝双发射？",
            "learner_id": "learner-r05a-api",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert {
        "answer",
        "evidence",
        "review",
        "confidence",
        "action_type",
        "recommended_path",
        "task_id",
        "task_state",
    }.issubset(data)
    serialized = json.dumps(data, ensure_ascii=False, default=str)
    for forbidden in (
        "LearnerIntelligenceView",
        "LearnerIntelligenceSignal",
        "_learner_intelligence_view",
        "legacy_learner_memory_path",
    ):
        assert forbidden not in serialized
