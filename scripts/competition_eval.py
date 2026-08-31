"""Evidence-backed competition readiness evaluation for DY3 Polaris.

The former ten-case script mixed scientific safety, answer availability,
personalization and placeholder graph coverage.  This evaluator measures each
claim independently and never counts ``tf-dy-*`` placeholder nodes as corpus
coverage or submitted correct answers as personalization accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
_BENCHMARK_PATH = _PROJECT_ROOT / "data" / "competition_benchmark.json"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def load_benchmark(path: Path = _BENCHMARK_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "metric_contract", "learner_profiles", "scientific_cases",
        "adaptation_topics", "job_scenarios", "feedback_actions",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"benchmark missing sections: {sorted(missing)}")
    return payload


def benchmark_case_count(benchmark: Mapping[str, Any]) -> int:
    return (
        len(benchmark.get("scientific_cases") or ())
        + len(benchmark.get("learner_profiles") or ())
        * len(benchmark.get("adaptation_topics") or ())
        + len(benchmark.get("job_scenarios") or ())
        + len(benchmark.get("feedback_actions") or ())
    )


def _data(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("code") == 0 and isinstance(payload.get("data"), dict):
        return dict(payload["data"])
    return payload


def build_runtime() -> tuple[Any, Any, tempfile.TemporaryDirectory[str]]:
    """Create isolated learner/task storage over the shipped L3 corpus."""

    from starlette.testclient import TestClient
    from dy3_polaris.l5.unified_app import UnifiedApp

    temporary = tempfile.TemporaryDirectory(prefix="dy3-competition-eval-")
    builder = UnifiedApp.create_full_app_builder(data_dir=temporary.name)
    return TestClient(builder.create_app()), builder, temporary


def _contains_all_groups(text: str, groups: Iterable[Iterable[str]]) -> bool:
    lowered = text.casefold()
    return all(
        any(str(term).casefold() in lowered for term in group)
        for group in groups
    )


def _is_honest_refusal(data: Mapping[str, Any]) -> bool:
    answer = str(data.get("answer") or "").strip()
    release = data.get("quality_release") or {}
    return bool(
        not answer
        or data.get("knowledge_unavailable")
        or (
            isinstance(release, Mapping)
            and not bool(release.get("eligible", False))
            and str(release.get("status") or "")
            not in {"FULL_RELEASE", "LIMITED_RELEASE"}
        )
    )


def classify_scientific_result(
    case: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one public response without reading private runtime state."""

    answer = str(data.get("answer") or "").strip()
    release = data.get("quality_release") or {}
    review = data.get("review") or {}
    evidence = list(data.get("evidence") or ())
    status = str(release.get("status") or "") if isinstance(release, Mapping) else ""
    verdict = str(review.get("verdict") or "").lower() if isinstance(review, Mapping) else ""
    review_status = str(review.get("status") or "").lower() if isinstance(review, Mapping) else ""
    published = bool(answer)
    release_ok = bool(
        isinstance(release, Mapping)
        and release.get("eligible") is True
        and status in {"FULL_RELEASE", "LIMITED_RELEASE"}
    )
    review_ok = review_status == "completed" and verdict in {"approved", "pass"}
    evidence_ok = bool(evidence)
    forbidden = [
        phrase for phrase in case.get("forbidden_phrases") or ()
        if str(phrase).casefold() in answer.casefold()
    ]
    term_groups_ok = _contains_all_groups(
        answer, case.get("required_any_groups") or ()
    )
    unsafe_published = bool(
        published and (
            not release_ok or not review_ok or not evidence_ok or bool(forbidden)
        )
    )
    behavior = str(case.get("behavior") or "answer")
    honest_refusal = _is_honest_refusal(data)
    if behavior == "answer":
        behavior_ok = published and release_ok and review_ok and evidence_ok
    elif behavior == "boundary":
        behavior_ok = honest_refusal or (
            published and release_ok and review_ok and evidence_ok
            and term_groups_ok and not forbidden
        )
    elif behavior == "honest_refusal":
        behavior_ok = honest_refusal
    else:
        behavior_ok = False
    domain_correct = bool(
        behavior_ok
        and (behavior != "answer" or term_groups_ok)
        and not forbidden
    )
    return {
        "case_id": str(case.get("case_id") or ""),
        "behavior": behavior,
        "task_id": str(data.get("task_id") or ""),
        "published": published,
        "honest_refusal": honest_refusal,
        "release_status": status,
        "release_ok": release_ok,
        "review_status": review_status,
        "review_verdict": verdict,
        "review_ok": review_ok,
        "evidence_count": len(evidence),
        "evidence_sources": list(dict.fromkeys(
            str(item.get("source") or item.get("document_id") or "")
            for item in evidence
            if isinstance(item, Mapping)
            and (item.get("source") or item.get("document_id"))
        )),
        "answer_excerpt": answer[:800],
        "term_groups_ok": term_groups_ok,
        "forbidden_phrases_found": forbidden,
        "unsafe_published": unsafe_published,
        "domain_correct": domain_correct,
    }


def summarize_scientific_results(
    details: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    total = len(details)
    answer_cases = [item for item in details if item.get("behavior") == "answer"]
    answer_count = sum(bool(item.get("published")) for item in answer_cases)
    unsafe_count = sum(bool(item.get("unsafe_published")) for item in details)
    correct_count = sum(bool(item.get("domain_correct")) for item in details)
    hallucination_rate = unsafe_count / total if total else 1.0
    answer_availability = answer_count / len(answer_cases) if answer_cases else 0.0
    domain_accuracy = correct_count / total if total else 0.0
    hallucination_target = float(contract["hallucination_rate_target"])
    availability_target = float(contract["answer_availability_target"])
    return {
        "total_cases": total,
        "answer_expected_cases": len(answer_cases),
        "published_answer_cases": answer_count,
        "unsafe_published_cases": unsafe_count,
        "domain_correct_cases": correct_count,
        "hallucination_rate": round(hallucination_rate, 4),
        "answer_availability_rate": round(answer_availability, 4),
        "domain_behavior_accuracy": round(domain_accuracy, 4),
        "targets": {
            "hallucination_rate_lt": hallucination_target,
            "answer_availability_gte": availability_target,
            "domain_behavior_accuracy_gte": availability_target,
        },
        "pass": bool(
            hallucination_rate < hallucination_target
            and answer_availability >= availability_target
            and domain_accuracy >= availability_target
        ),
        "details": list(details),
    }


def evaluate_scientific_quality(
    client: Any,
    cases: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for case in cases:
        response = client.post(
            "/api/query",
            json={
                "query": str(case["query"]),
                "learner_id": f"eval-science-{case['case_id']}",
            },
        )
        result = classify_scientific_result(case, _data(response))
        result["http_status"] = response.status_code
        details.append(result)
    return summarize_scientific_results(details, contract)


def _declare_profile(client: Any, learner_id: str, declared: Mapping[str, Any]) -> bool:
    for slot_key, value in declared.items():
        response = client.post(
            "/api/user-understanding/answer",
            json={
                "learner_id": learner_id,
                "payload": {"slot_key": str(slot_key), "value": value},
            },
        )
        if response.status_code != 200 or not _data(response):
            return False
    return True


def _resource_families(data: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("resource_family") or "")
        for item in data.get("learning_resources") or ()
        if isinstance(item, Mapping)
    }


_EXPECTED_AGENT_IDS = {
    "agent.learning.diagnosis",
    "agent.knowledge.generation",
    "agent.quality.review",
    "agent.guidance.decision",
}


def _public_agent_ids(data: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("agent_id") or "")
        for item in data.get("agent_trace") or ()
        if isinstance(item, Mapping) and item.get("agent_id")
    }


def classify_job_task_result(
    scenario: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    profile_persisted: bool,
) -> dict[str, Any]:
    answer = str(data.get("answer") or "")
    release = data.get("quality_release") or {}
    review = data.get("review") or {}
    evidence = [
        item for item in data.get("evidence") or () if isinstance(item, Mapping)
    ]
    families = _resource_families(data)
    required_families = {
        str(item) for item in scenario.get("required_resource_families") or ()
    }
    agent_ids = _public_agent_ids(data)
    release_ok = bool(
        isinstance(release, Mapping)
        and release.get("eligible") is True
        and release.get("status") in {"FULL_RELEASE", "LIMITED_RELEASE"}
    )
    review_ok = bool(
        isinstance(review, Mapping)
        and str(review.get("status") or "").lower() == "completed"
        and str(review.get("verdict") or "").lower() in {"approved", "pass"}
    )
    term_groups_ok = _contains_all_groups(
        answer, scenario.get("required_any_groups") or ()
    )
    resources_ok = required_families.issubset(families)
    collaboration_ok = _EXPECTED_AGENT_IDS.issubset(agent_ids)
    passed = bool(
        profile_persisted
        and answer.strip()
        and release_ok
        and review_ok
        and evidence
        and term_groups_ok
        and resources_ok
        and collaboration_ok
    )
    return {
        "case_id": str(scenario.get("case_id") or ""),
        "role": str(scenario.get("role") or ""),
        "profile_id": str(scenario.get("profile_id") or ""),
        "profile_persisted": profile_persisted,
        "answer_excerpt": answer[:1200],
        "release_ok": release_ok,
        "review_ok": review_ok,
        "evidence_count": len(evidence),
        "term_groups_ok": term_groups_ok,
        "resource_families": sorted(families),
        "required_resource_families": sorted(required_families),
        "resources_ok": resources_ok,
        "agent_ids": sorted(agent_ids),
        "collaboration_ok": collaboration_ok,
        "pass": passed,
    }


def evaluate_job_task_fit(
    client: Any,
    scenarios: list[Mapping[str, Any]],
    profiles: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    profiles_by_id = {
        str(profile.get("profile_id") or ""): profile for profile in profiles
    }
    details: list[dict[str, Any]] = []
    for scenario in scenarios:
        profile_id = str(scenario.get("profile_id") or "")
        profile = profiles_by_id.get(profile_id) or {}
        learner_id = f"eval-job-{scenario['case_id']}"
        profile_persisted = bool(
            profile and _declare_profile(client, learner_id, profile.get("declared") or {})
        )
        response = client.post(
            "/api/query",
            json={"query": str(scenario["query"]), "learner_id": learner_id},
        )
        result = classify_job_task_result(
            scenario, _data(response), profile_persisted=profile_persisted
        )
        result["http_status"] = response.status_code
        details.append(result)
    passed = sum(bool(item["pass"]) for item in details)
    accuracy = passed / len(details) if details else 0.0
    target = float(contract["job_task_fit_target"])
    return {
        "measurement_scope": (
            "authored role-task scenarios over the public runtime; "
            "not employer certification or field-outcome validation"
        ),
        "total_cases": len(details),
        "passed_cases": passed,
        "job_task_fit_accuracy": round(accuracy, 4),
        "target": target,
        "pass": bool(details and accuracy >= target),
        "details": details,
    }


def evaluate_learner_resource_fit(
    client: Any,
    profiles: list[Mapping[str, Any]],
    topics: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    by_topic: dict[str, set[tuple[str, str]]] = {}
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        learner_id = f"eval-profile-{profile_id}"
        declared_ok = _declare_profile(client, learner_id, profile["declared"])
        for topic in topics:
            response = client.post(
                "/api/query",
                json={"query": str(topic["query"]), "learner_id": learner_id},
            )
            data = _data(response)
            decision = data.get("teaching_strategy") or {}
            depth = str(decision.get("content_depth") or "")
            strategy = str(decision.get("explanation_strategy") or "")
            representations = {
                str(item) for item in decision.get("representation_modes") or ()
            }
            families = _resource_families(data)
            required_families = {
                "knowledge_understanding", "research_practice", "assessment_practice"
            }
            release = data.get("quality_release") or {}
            release_ok = bool(
                release.get("eligible") is True
                and release.get("status") in {"FULL_RELEASE", "LIMITED_RELEASE"}
            )
            fit = bool(
                declared_ok
                and depth in set(profile.get("allowed_depths") or ())
                and str(profile.get("required_representation") or "") in representations
                and required_families.issubset(families)
                and release_ok
            )
            details.append({
                "case_id": f"{topic['case_id']}:{profile_id}",
                "profile_id": profile_id,
                "declared_profile_persisted": declared_ok,
                "depth": depth,
                "allowed_depths": list(profile.get("allowed_depths") or ()),
                "explanation_strategy": strategy,
                "representation_modes": sorted(representations),
                "required_representation": profile.get("required_representation"),
                "resource_families": sorted(families),
                "release_ok": release_ok,
                "fit": fit,
            })
            by_topic.setdefault(str(topic["case_id"]), set()).add((depth, strategy))
    fit_count = sum(bool(item["fit"]) for item in details)
    accuracy = fit_count / len(details) if details else 0.0
    differentiated = sum(len(values) >= 2 for values in by_topic.values())
    differentiated_rate = differentiated / len(by_topic) if by_topic else 0.0
    target = float(contract["learner_resource_fit_target"])
    return {
        "total_cases": len(details),
        "fit_cases": fit_count,
        "learner_resource_fit_accuracy": round(accuracy, 4),
        "differentiated_topics": differentiated,
        "topic_count": len(by_topic),
        "differentiated_topic_rate": round(differentiated_rate, 4),
        "target": target,
        "pass": bool(accuracy >= target and differentiated_rate >= target),
        "details": details,
    }


def _resource_for_action(data: Mapping[str, Any], action: str) -> Mapping[str, Any] | None:
    for resource in data.get("learning_resources") or ():
        if isinstance(resource, Mapping) and action in set(
            str(item) for item in resource.get("interaction_actions") or ()
        ):
            return resource
    return None


def evaluate_feedback_loop(
    client: Any,
    cases: list[Mapping[str, Any]],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    query = "Dy³⁺浓度猝灭为什么发生？"
    for case in cases:
        learner_id = f"eval-feedback-{case['case_id']}"
        _declare_profile(client, learner_id, {
            "learning_stage": "本科阶段",
            "learning_goal": "理解基础概念",
            "professional_background": "材料科学",
            "domain_experience": "刚开始了解",
        })
        first = _data(client.post(
            "/api/query", json={"query": query, "learner_id": learner_id}
        ))
        action = str(case["action"])
        resource = _resource_for_action(first, action)
        interaction_data: dict[str, Any] = {}
        status = 0
        if resource is not None:
            interaction = client.post(
                "/api/learning/resources/interact",
                json={
                    "learner_id": learner_id,
                    "task_id": first.get("task_id"),
                    "resource_id": resource.get("resource_id"),
                    "action": action,
                },
            )
            status = interaction.status_code
            interaction_data = _data(interaction)
        expected_strategy = str(case.get("expected_next_strategy") or "")
        observed_strategy = ""
        if expected_strategy and interaction_data:
            follow_up = _data(client.post(
                "/api/query", json={"query": query, "learner_id": learner_id}
            ))
            observed_strategy = str(
                (follow_up.get("teaching_strategy") or {}).get(
                    "explanation_strategy"
                ) or ""
            )
        passed = bool(
            resource is not None
            and status == 200
            and interaction_data.get("source_class") == "OBSERVED"
            and interaction_data.get("mastery_updated") is False
            and (not expected_strategy or observed_strategy == expected_strategy)
        )
        if action == "start_practice":
            passed = bool(passed and interaction_data.get("practice_endpoint"))
        details.append({
            "case_id": case["case_id"],
            "action": action,
            "resource_found": resource is not None,
            "http_status": status,
            "source_class": interaction_data.get("source_class"),
            "mastery_updated": interaction_data.get("mastery_updated"),
            "expected_next_strategy": expected_strategy,
            "observed_next_strategy": observed_strategy,
            "pass": passed,
        })
    passed = sum(bool(item["pass"]) for item in details)
    return {
        "total_cases": len(details),
        "passed_cases": passed,
        "pass": passed == len(details) and bool(details),
        "details": details,
    }


def evaluate_real_knowledge_coverage(
    builder: Any,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure source-candidate coverage over the loaded 48-KP curriculum."""

    from dy3_polaris.l2.kp_catalog import NEW_ALL_KP_IDS, NEW_KP_NAMES
    from dy3_polaris.l3.concept_foundation import build_concept_foundation

    store = getattr(getattr(builder, "_l3_router", None), "_store", None)
    if store is None:
        raise RuntimeError("loaded L3 store is unavailable")
    foundation = build_concept_foundation(store)
    source_backed_concepts = {
        item.concept_id for item in foundation.evidence_mappings
    }
    source_backed_kps = {
        kp_id
        for concept_id in source_backed_concepts
        for kp_id in foundation.concepts[concept_id].related_kps
        if kp_id in NEW_ALL_KP_IDS
    }
    uncovered = [kp_id for kp_id in NEW_ALL_KP_IDS if kp_id not in source_backed_kps]
    curriculum_coverage = (
        len(source_backed_kps) / len(NEW_ALL_KP_IDS) if NEW_ALL_KP_IDS else 0.0
    )
    concept_coverage = (
        len(source_backed_concepts) / len(foundation.concepts)
        if foundation.concepts else 0.0
    )
    target = float(contract["curriculum_source_coverage_target"])
    return {
        "coverage_semantics": (
            "term-mention candidates in real loaded DocumentChunks; "
            "not direct claim support and not placeholder KG nodes"
        ),
        "loaded_chunk_count": int(store.chunk_count()),
        "canonical_concepts": len(foundation.concepts),
        "source_backed_concepts": len(source_backed_concepts),
        "concept_source_candidate_coverage": round(concept_coverage, 4),
        "curriculum_kps": len(NEW_ALL_KP_IDS),
        "source_backed_curriculum_kps": len(source_backed_kps),
        "curriculum_source_candidate_coverage": round(curriculum_coverage, 4),
        "uncovered_kps": [
            {"kp_id": kp_id, "name": NEW_KP_NAMES[kp_id]}
            for kp_id in uncovered
        ],
        "target": target,
        "pass": curriculum_coverage >= target,
        "foundation_stats": foundation.stats(),
    }


def evaluate_case_volume(
    benchmark: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    count = benchmark_case_count(benchmark)
    minimum = int(contract["minimum_case_count"])
    return {
        "case_count": count,
        "minimum": minimum,
        "scientific_cases": len(benchmark["scientific_cases"]),
        "adaptation_cases": len(benchmark["learner_profiles"])
        * len(benchmark["adaptation_topics"]),
        "job_task_cases": len(benchmark["job_scenarios"]),
        "feedback_loop_cases": len(benchmark["feedback_actions"]),
        "pass": count >= minimum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DY3 Polaris competition readiness evaluation")
    parser.add_argument("--out", default="competition_eval_report.json")
    parser.add_argument(
        "--quick", action="store_true",
        help="run a smoke subset; quick output never counts as the official 50-case report",
    )
    args = parser.parse_args()

    benchmark = load_benchmark()
    contract = benchmark["metric_contract"]
    official_volume = evaluate_case_volume(benchmark, contract)
    scientific_cases = list(benchmark["scientific_cases"])
    adaptation_topics = list(benchmark["adaptation_topics"])
    job_scenarios = list(benchmark["job_scenarios"])
    feedback_cases = list(benchmark["feedback_actions"])
    if args.quick:
        scientific_cases = scientific_cases[:4]
        adaptation_topics = adaptation_topics[:1]
        job_scenarios = job_scenarios[:1]
        feedback_cases = feedback_cases[:1]

    started = time.time()
    client, builder, temporary = build_runtime()
    try:
        scientific = evaluate_scientific_quality(client, scientific_cases, contract)
        adaptation = evaluate_learner_resource_fit(
            client, list(benchmark["learner_profiles"]), adaptation_topics, contract
        )
        job_fit = evaluate_job_task_fit(
            client,
            job_scenarios,
            list(benchmark["learner_profiles"]),
            contract,
        )
        feedback = evaluate_feedback_loop(client, feedback_cases)
        coverage = evaluate_real_knowledge_coverage(builder, contract)
    finally:
        client.close()
        temporary.cleanup()

    official = not args.quick
    overall_pass = bool(
        official
        and official_volume["pass"]
        and scientific["pass"]
        and adaptation["pass"]
        and job_fit["pass"]
        and feedback["pass"]
        and coverage["pass"]
    )
    report = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_version": benchmark["version"],
        "generated_at": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "official_full_run": official,
        "overall_pass": overall_pass,
        "case_volume": official_volume,
        "metrics": {
            "scientific_quality": scientific,
            "learner_resource_fit": adaptation,
            "job_task_fit": job_fit,
            "feedback_decision_loop": feedback,
            "knowledge_coverage": coverage,
        },
        "limitations": [
            "Concept source coverage means a real chunk mentions the term; it is not direct claim support.",
            "Declared background is a low-confidence prior and does not masquerade as demonstrated mastery.",
            "Resource self-report changes future teaching strategy but never updates BKT/IRT mastery.",
            "A quick run is smoke evidence only and cannot satisfy the 50-case requirement.",
            "Job-task fit uses authored role scenarios; it is not employer certification or workplace outcome evidence.",
        ],
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "official_full_run": official,
        "case_count": official_volume["case_count"],
        "scientific_pass": scientific["pass"],
        "learner_resource_fit": adaptation["learner_resource_fit_accuracy"],
        "feedback_loop_pass": feedback["pass"],
        "curriculum_source_coverage": coverage["curriculum_source_candidate_coverage"],
        "overall_pass": overall_pass,
        "report": str(out_path),
    }, ensure_ascii=False, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
