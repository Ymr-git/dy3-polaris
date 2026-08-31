"""R-03A task-understanding contract and real /api/query boundary tests."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_workers import (
    DIAGNOSIS_AGENT_ID,
    GENERATION_AGENT_ID,
    REVIEW_AGENT_ID,
    _attach_review_candidate,
    _attach_selected_evidence_candidate,
)
from dy3_polaris.l5.task_understanding import (
    IntentResult,
    TaskMode,
    understand_task,
)
from dy3_polaris.l5.unified_app import UnifiedApp


MODE_CASES = (
    ("Dy3+ 的蓝光发射波长是多少？", TaskMode.FACT_FIND),
    ("Dy3+ 为什么会发生浓度猝灭？", TaskMode.EXPLAIN),
    ("如何比较 YAG:Dy3+ 和 Y2O3:Dy3+ 的量子效率？", TaskMode.COMPARE),
    ("3000 K 的 Dy3+ 白光是否一定更健康？", TaskMode.EVALUATE),
    ("如何设计实验验证 Dy3+ 基质的热猝灭机制？", TaskMode.RESEARCH_GUIDE),
)


@pytest.mark.parametrize(("query", "expected"), MODE_CASES)
def test_deterministic_task_modes(query: str, expected: TaskMode) -> None:
    result = understand_task(query, use_llm=False)

    assert result.task_mode is expected
    assert result.primary_intent
    assert result.learner_goal
    assert result.evidence_need in {"low", "medium", "high"}
    assert result.risk_level in {"low", "medium", "high"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.required_capabilities


def test_domain_entities_cover_scientific_and_lighting_signals() -> None:
    result = understand_task(
        "比较 YAG:Dy³⁺ 在 575 nm、7 mol% 和 3000 K 下的 CCT 与 CRI",
        use_llm=False,
    )
    entity_types = {entity.entity_type for entity in result.domain_entities}
    entity_text = {entity.text.upper() for entity in result.domain_entities}

    assert result.task_mode is TaskMode.COMPARE
    assert "ion" in entity_types
    assert "material" in entity_types
    assert "measurement" in entity_types
    assert {"CCT", "CRI"}.issubset(entity_text)


def test_safety_language_sets_evidence_and_risk_without_claiming_certainty() -> None:
    result = understand_task(
        "3000 K 的 Dy3+ 白光是否一定更健康？",
        use_llm=False,
    )

    assert result.task_mode is TaskMode.EVALUATE
    assert result.evidence_need == "high"
    assert result.risk_level == "high"
    assert "health_criterion_unspecified" in result.ambiguity


def test_rule_llm_disagreement_keeps_strong_safety_signal_and_lowers_confidence() -> None:
    query = "这种 Dy3+ 白光是否一定安全？"
    deterministic = understand_task(query, use_llm=False)

    result = understand_task(
        query,
        semantic_interpreter=lambda _query, _context: {
            "task_mode": "EXPLAIN",
            "confidence": 0.98,
        },
    )

    assert result.task_mode is TaskMode.EVALUATE
    assert result.confidence < deterministic.confidence
    assert "semantic_disagreement" in result.ambiguity
    assert result.semantic_source == "deterministic+llm_disagreement"


def test_llm_can_resolve_only_a_weak_unclassified_request() -> None:
    result = understand_task(
        "帮我看看这个现象",
        semantic_interpreter=lambda _query, _context: {
            "task_mode": "RESEARCH_GUIDE",
            "confidence": 0.8,
        },
    )

    assert result.task_mode is TaskMode.RESEARCH_GUIDE
    assert result.semantic_source == "llm_resolved"


def _stub_current_workers(monkeypatch, observed: list[IntentResult]) -> None:
    def diagnosis(input_data, _deps):
        observed.append(input_data["_intent_result"])
        return {
            "agent_id": DIAGNOSIS_AGENT_ID,
            "status": "completed",
            "weak_kps": [],
            "profile": {},
            "confidence": 0.9,
        }

    def generation(input_data, _deps, review_feedback=None):
        del review_feedback
        return _attach_selected_evidence_candidate(
            {
                "agent_id": GENERATION_AGENT_ID,
                "status": "completed",
                "answer": "基于当前证据形成的教学解释。",
                "confidence": 0.9,
                "context_chunks": [
                    "Dy3+ 发光材料证据片段：基于当前证据形成的教学解释。"
                ],
                "citations": ["test-source"],
                "sources": [
                    {
                        "source": "test-source",
                        "document_id": "test-document",
                        "chunk_id": "test-chunk",
                    }
                ],
                "candidates": [],
                "consensus_score": 1.0,
                "consensus_reached": True,
                "selected_candidate": "",
                "needs_adjudication": False,
                "question_type": "other",
            },
            input_data,
            stage="selected",
        )

    def review(input_data, _deps):
        return _attach_review_candidate(
            {
                "agent_id": REVIEW_AGENT_ID,
                "status": "completed",
                "verdict": "approved",
                "confidence": 0.9,
                "reason": "测试审核通过",
                "fact_check": {},
                "anti_hallucination": {},
            },
            input_data,
            content=str(input_data.get("content") or ""),
            producer="agent.quality.review/run_review",
            real_reviewer_executed=True,
        )

    monkeypatch.setattr(agent_workers, "run_diagnosis", diagnosis)
    monkeypatch.setattr(agent_workers, "_run_multi_candidate_generation", generation)
    monkeypatch.setattr(agent_workers, "run_review", review)
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        lambda *_args, **_kwargs: {
            "adopted": False,
            "rounds": [],
            "final_verdict": "pass",
            "final_score": 0.9,
        },
    )
    real_understanding = understand_task
    monkeypatch.setattr(
        agent_workers,
        "understand_task",
        lambda query, **_kwargs: real_understanding(query, use_llm=False),
    )


def test_real_api_query_resolves_all_modes_before_first_agent_without_leakage(
    monkeypatch,
) -> None:
    observed: list[IntentResult] = []
    _stub_current_workers(monkeypatch, observed)
    client = TestClient(UnifiedApp.create_full_app_builder().create_app())

    for query, expected in MODE_CASES:
        response = client.post("/api/query", json={"query": query})
        assert response.status_code == 200
        data = response.json()["data"]
        serialized = json.dumps(data, ensure_ascii=False)
        assert "_intent_result" not in serialized
        assert "IntentResult" not in serialized
        assert "task_mode" not in data
        assert data["task_id"].startswith("task-")
        assert data["answer"]
        assert observed[-1].task_mode is expected

    assert len(observed) == len(MODE_CASES)


def test_intent_result_is_slotted_and_not_a_public_mapping() -> None:
    result = understand_task("Dy3+ 为什么有黄蓝双发射？", use_llm=False)

    assert not hasattr(result, "__dict__")
    assert not isinstance(result, dict)
