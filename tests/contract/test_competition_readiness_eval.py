"""Contract tests for the evidence-backed competition evaluation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "competition_eval.py"
_SPEC = importlib.util.spec_from_file_location("competition_eval", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
competition_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(competition_eval)


def test_benchmark_contains_three_profiles_and_at_least_fifty_real_cases() -> None:
    benchmark = competition_eval.load_benchmark()

    assert len(benchmark["learner_profiles"]) >= 3
    assert competition_eval.benchmark_case_count(benchmark) >= 50
    assert competition_eval.scientific_domain_case_count(benchmark) >= 50
    assert len(benchmark["scientific_cases"]) >= 51
    assert len(benchmark["adaptation_topics"]) >= 8
    assert len(benchmark["job_scenarios"]) >= 3
    assert len(benchmark["feedback_actions"]) >= 4
    for case in benchmark["scientific_cases"]:
        if case["behavior"] == "honest_refusal":
            assert case["concept_ids"] == []
        else:
            assert case["concept_ids"]


def test_refuse_everything_cannot_pass_scientific_readiness() -> None:
    benchmark = competition_eval.load_benchmark()
    cases = benchmark["scientific_cases"][:3]
    details = [
        competition_eval.classify_scientific_result(case, {
            "answer": "",
            "evidence": [],
            "review": {"status": "skipped", "verdict": "skipped"},
            "quality_release": {"eligible": False, "status": "REFUSE"},
            "knowledge_unavailable": True,
        })
        for case in cases
    ]

    result = competition_eval.summarize_scientific_results(
        details, benchmark["metric_contract"]
    )

    assert result["hallucination_rate"] == 0.0
    assert result["answer_availability_rate"] == 0.0
    assert result["pass"] is False


def test_grounded_release_requires_review_evidence_and_domain_terms() -> None:
    case = competition_eval.load_benchmark()["scientific_cases"][0]
    data = {
        "task_id": "task-eval",
        "answer": "Dy3+由4F9/2能级向6H15/2和6H13/2跃迁形成蓝、黄发射。",
        "evidence": [{"chunk_id": "real-chunk"}],
        "review": {"status": "completed", "verdict": "approved"},
        "quality_release": {"eligible": True, "status": "FULL_RELEASE"},
    }

    result = competition_eval.classify_scientific_result(case, data)

    assert result["unsafe_published"] is False
    assert result["term_groups_ok"] is True
    assert result["domain_correct"] is True


def test_coverage_uses_loaded_chunks_and_new_curriculum_not_placeholder_nodes() -> None:
    from dy3_polaris.l3.persistence import PersistenceManager
    from dy3_polaris.l3.store import KnowledgeStore
    from dy3_polaris.l5.unified_app import _l3_snapshot_dir

    store = KnowledgeStore()
    PersistenceManager(store, base_path=_l3_snapshot_dir()).load_snapshot()
    builder = SimpleNamespace(_l3_router=SimpleNamespace(_store=store))
    benchmark = competition_eval.load_benchmark()

    result = competition_eval.evaluate_real_knowledge_coverage(
        builder, benchmark["metric_contract"]
    )

    assert result["loaded_chunk_count"] > 0
    assert result["curriculum_kps"] == 48
    assert result["source_backed_curriculum_kps"] == 45
    assert result["curriculum_source_candidate_coverage"] >= 0.90
    assert "placeholder" in result["coverage_semantics"]

    grounding = competition_eval.evaluate_scientific_case_grounding(
        builder, benchmark["scientific_cases"]
    )
    assert grounding["domain_cases"] >= 50
    assert grounding["grounded_cases"] == len(benchmark["scientific_cases"])
    assert grounding["pass"] is True


def test_job_task_fit_requires_real_four_agent_public_trace() -> None:
    benchmark = competition_eval.load_benchmark()
    scenario = benchmark["job_scenarios"][1]
    data = {
        "answer": "蓝光光生物危害不能只看CCT色温，还要核对光谱辐亮度与暴露条件。",
        "evidence": [{"chunk_id": "real-chunk", "source": "real-source"}],
        "review": {"status": "completed", "verdict": "approved"},
        "quality_release": {"eligible": True, "status": "FULL_RELEASE"},
        "learning_resources": [
            {"resource_family": "knowledge_understanding"},
            {"resource_family": "research_practice"},
            {"resource_family": "assessment_practice"},
        ],
        "agent_trace": [
            {"agent_id": "agent.learning.diagnosis"},
            {"agent_id": "agent.knowledge.generation"},
            {"agent_id": "agent.quality.review"},
        ],
    }

    missing_guidance = competition_eval.classify_job_task_result(
        scenario, data, profile_persisted=True
    )
    assert missing_guidance["collaboration_ok"] is False
    assert missing_guidance["pass"] is False

    data["agent_trace"].append({"agent_id": "agent.guidance.decision"})
    complete = competition_eval.classify_job_task_result(
        scenario, data, profile_persisted=True
    )
    assert complete["collaboration_ok"] is True
    assert complete["resources_ok"] is True
    assert complete["pass"] is True
