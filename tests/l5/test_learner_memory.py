"""R05 shared Learner Memory and memory-aware collaboration tests."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import SimpleNamespace
from typing import Any

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_memory import (
    build_memory_views,
    extract_memory_candidate,
    load_learner_memory,
)
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.interaction_recorder import InteractionRecorder
from dy3_polaris.l5.learner_intelligence import LearnerIntelligenceView
from dy3_polaris.l5 import task_state as task_state_runtime
from tests.l5.test_private_runtime_carrier import (
    _review_candidate,
    _selected_generation,
)


@dataclass
class _Profile:
    learner_id: str
    extras: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    confidence: float = 0.5
    level: str = "beginner"
    weak_kps: list[str] = field(default_factory=list)
    kp_mastery: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "extras": dict(self.extras),
            "version": self.version,
            "confidence": self.confidence,
            "level": self.level,
            "weak_kps": list(self.weak_kps),
            "kp_mastery": dict(self.kp_mastery),
        }


class _ProfileService:
    def __init__(self) -> None:
        self.profiles: dict[str, _Profile] = {}

    def ensure(self, learner_id: str) -> _Profile:
        return self.profiles.setdefault(learner_id, _Profile(learner_id))

    def get_profile_snapshot(self, learner_id: str) -> _Profile:
        return self.ensure(learner_id)

    def get_weak_points(self, _learner_id: str) -> dict[str, Any]:
        return {"weak_kps": []}

    def apply_update(
        self,
        learner_id: str,
        *,
        updates: dict[str, Any],
        expected_version: int | None = None,
    ) -> _Profile:
        profile = self.ensure(learner_id)
        assert expected_version in {None, profile.version}
        if "extras" in updates:
            profile.extras.update(dict(updates["extras"]))
        if "confidence" in updates:
            profile.confidence = float(updates["confidence"])
        profile.version += 1
        return profile


def _run_task(
    monkeypatch,
    *,
    service: _ProfileService,
    learner_id: str,
    task_id: str,
    query: str,
    answer: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    diagnosis_inputs: list[dict[str, Any]] = []
    generation_inputs: list[dict[str, Any]] = []
    original_diagnosis = agent_workers.run_diagnosis

    def diagnosis(payload, deps):
        diagnosis_inputs.append(dict(payload))
        return original_diagnosis(payload, deps)

    def generation(payload, _deps):
        generation_inputs.append(dict(payload))
        return _selected_generation(answer, task_id)

    monkeypatch.setattr(agent_workers, "run_diagnosis", diagnosis)
    monkeypatch.setattr(agent_workers, "_run_multi_candidate_generation", generation)
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda *_args, **_kwargs: _review_candidate(task_id, answer, "approved"),
    )
    monkeypatch.setattr(agent_workers, "get_recorder", lambda: InteractionRecorder())
    task_context = task_state_runtime.create_task_context(task_id)
    task_state_runtime.set_task_state(task_context, "UNDERSTANDING")
    result = agent_workers.run_guidance(
        {
            "task_id": task_id,
            "query": query,
            "learner_id": learner_id,
            "learner_level": "beginner",
            "task_context": task_context,
        },
        AgentDependencies(profile_service=service),
    )
    return result, diagnosis_inputs, generation_inputs


def test_first_task_has_no_memory_and_persists_filtered_learning_facts(monkeypatch) -> None:
    service = _ProfileService()
    answer = "Dy³⁺ 黄蓝双发射来自 4F9/2 到 6H15/2、6H13/2 的跃迁。"
    result, diagnosis_inputs, generation_inputs = _run_task(
        monkeypatch,
        service=service,
        learner_id="learner-memory-demo",
        task_id="task-memory-first",
        query="Dy³⁺为什么能够产生黄蓝双发射？",
        answer=answer,
    )

    assert "_learner_memory_view" not in diagnosis_inputs[0]
    assert "_learner_memory_view" not in generation_inputs[0]
    memory = load_learner_memory(service, "learner-memory-demo")
    assert memory["knowledge_state"] == {}
    events = memory["learning_events"]
    assert {item["classification"] for item in events} == {"OBSERVED", "DERIVED"}
    assert any(item["event_type"] == "query" for item in events)
    assert any(item["event_type"] == "topic_explained" for item in events)
    assert all("mastery" not in item and "theta" not in item for item in events)
    assert memory["learning_history"][-1]["task_id"] == "task-memory-first"
    assert answer not in json.dumps(memory, ensure_ascii=False)
    assert result["answer"] == answer


def test_second_related_task_routes_memory_through_diagnosis_only(monkeypatch) -> None:
    service = _ProfileService()
    first_answer = "Dy³⁺ 黄蓝双发射来自已审核的能级跃迁。"
    first, _, _ = _run_task(
        monkeypatch,
        service=service,
        learner_id="learner-memory-demo",
        task_id="task-memory-first",
        query="Dy³⁺为什么能够产生黄蓝双发射？",
        answer=first_answer,
    )
    second_answer = "白光调控需联合考察 Y/B、CIE 色坐标、CCT 与 CRI。"
    second, diagnosis_inputs, generation_inputs = _run_task(
        monkeypatch,
        service=service,
        learner_id="learner-memory-demo",
        task_id="task-memory-second",
        query="如何调控Dy³⁺实现更好的白光？",
        answer=second_answer,
    )

    intelligence_view = diagnosis_inputs[0]["_learner_intelligence_view"]
    assert isinstance(intelligence_view, LearnerIntelligenceView)
    assert intelligence_view.metadata["memory_available"] is True
    assert "task-memory-first" in intelligence_view.value(
        "facts", "known_history", ()
    )
    assert intelligence_view.value("derived_context", "weak_kps", ()) == ()
    focus = intelligence_view.value("derived_context", "prerequisite_focus", ())
    assert focus
    assert any("白光" in item for item in focus)
    assert "_learner_memory_view" not in diagnosis_inputs[0]
    assert "_learner_memory_view" not in generation_inputs[0]
    assert "_learner_intelligence_view" not in generation_inputs[0]
    # A prior explanation is teaching history, not observed mastery.  Memory
    # changes the Diagnosis focus above, but cannot promote the learner to an
    # advanced level without AnswerRecord/BKT/IRT evidence.
    assert generation_inputs[0]["learner_level"] == "beginner"

    generation_agent_input = generation_inputs[0]["_agent_input"]
    planned_goals = [
        generation_agent_input.subtask.goal,
        *(item.goal for item in generation_agent_input.related_subtasks),
    ]
    assert all("跳过已掌握基础内容" not in goal for goal in planned_goals)
    assert all("黄蓝发射强度比 Y/B" not in goal for goal in planned_goals)
    path_topics = [item.get("topic") for item in second["recommended_path"]]
    assert "黄蓝发射强度比 Y/B" not in path_topics
    assert "相关色温 CCT" not in path_topics
    assert set(first) == set(second)


def test_same_follow_up_for_new_user_keeps_foundation_strategy(monkeypatch) -> None:
    service = _ProfileService()
    _, diagnosis_inputs, generation_inputs = _run_task(
        monkeypatch,
        service=service,
        learner_id="learner-new",
        task_id="task-new-user",
        query="如何调控Dy³⁺实现更好的白光？",
        answer="白光调控需联合考察 Y/B、CIE、CCT 与 CRI。",
    )

    assert "_learner_memory_view" not in diagnosis_inputs[0]
    assert "_learner_memory_view" not in generation_inputs[0]
    assert generation_inputs[0]["learner_level"] != "advanced"


def test_private_memory_does_not_leak_to_current_response(monkeypatch) -> None:
    service = _ProfileService()
    _run_task(
        monkeypatch,
        service=service,
        learner_id="learner-private",
        task_id="task-private-first",
        query="Dy³⁺为什么能够产生黄蓝双发射？",
        answer="Dy³⁺ 黄蓝发射来自已审核的能级跃迁。",
    )
    result, _, _ = _run_task(
        monkeypatch,
        service=service,
        learner_id="learner-private",
        task_id="task-private-second",
        query="如何调控Dy³⁺实现更好的白光？",
        answer="白光调控需联合考察 Y/B、CIE、CCT 与 CRI。",
    )

    public = json.dumps(dict(result), ensure_ascii=False, default=str)
    for forbidden in (
        "learner_memory",
        "_learner_memory_view",
        "memory_focus_topics",
        "LearnerMemoryCandidate",
        "LearningEvent",
        "adaptive_strategy",
        "_contract_candidate",
    ):
        assert forbidden not in public
    assert set(build_memory_views(service, "learner-private", "如何调控Dy³⁺实现更好的白光？")) == {
        agent_workers.DIAGNOSIS_AGENT_ID,
        agent_workers.GENERATION_AGENT_ID,
        agent_workers.REVIEW_AGENT_ID,
        agent_workers.GUIDANCE_AGENT_ID,
    }


def test_unstructured_challenge_cannot_create_a_canned_error_pattern() -> None:
    candidate = extract_memory_candidate(
        context=SimpleNamespace(challenges=[object()]),
        final_result=SimpleNamespace(
            task_id="task-memory-challenge",
            task_mode=SimpleNamespace(value="EVALUATE"),
            answer_identity="answer-hash",
            completion_eligibility=True,
            next_action="核对 SPD 与暴露条件",
            provenance_refs=("doc-health-01",),
        ),
        question="3000K低色温是否一定没有蓝光风险？",
    )

    assert candidate.valid is True
    assert candidate.error_patterns == ()
    assert candidate.misconceptions == ()
