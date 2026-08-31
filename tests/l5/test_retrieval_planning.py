"""R-03D subtask retrieval planning, rerank, and EvidencePack tests."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_contracts import build_agent_input
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.retrieval_planning import (
    EvidencePack,
    agent_aware_rerank,
    build_evidence_pack,
    build_retrieval_need,
    build_retrieval_plan,
    build_retrieval_plans,
    hard_filter,
    retrieval_metrics,
)
from dy3_polaris.l5.task_understanding import understand_task
from dy3_polaris.l3.concept_foundation import load_curated_concept_evidence
from dy3_polaris.l3.models import DocumentChunk
from dy3_polaris.l3.store import KnowledgeStore
from dy3_polaris.l5.knowledge_learning_fusion import build_knowledge_learning_context


def _context(query: str, task_id: str = "task-r03d"):
    data = {"task_id": task_id, "query": query, "learner_id": "r03d"}
    return initialize_collaboration_context(
        data,
        intent_resolver=lambda value, **_kwargs: understand_task(value, use_llm=False),
    )


def _generation_input(query: str, task_id: str = "task-r03d"):
    context = _context(query, task_id)
    diagnosis = agent_workers._start_contract_agent(
        context, agent_workers.DIAGNOSIS_AGENT_ID
    )
    assert diagnosis is not None
    fact = agent_workers._adapt_diagnosis_contribution(
        context,
        diagnosis,
        {"summary": "learner context", "ability": {"theta": 0}, "confidence": 0.8},
    )
    agent_workers._finish_contract_agent(context, diagnosis, fact)
    value = build_agent_input(context, agent_workers.GENERATION_AGENT_ID)
    assert value is not None
    return context, value


def _candidate(chunk_id, content, source="doc-1", entity="Dy3+"):
    return {
        "chunk_id": chunk_id,
        "document_id": source,
        "content": content,
        "metadata": {"entity": entity},
    }


def test_fact_find_has_one_precise_query() -> None:
    _, value = _generation_input("Dy³⁺主要黄色发射对应什么跃迁？")
    plans = build_retrieval_plans(value)
    assert plans
    assert all(len(plan.rewritten_queries) == 1 for plan in plans)


def test_different_subtasks_have_different_plans() -> None:
    _, value = _generation_input("为什么Dy³⁺会产生黄蓝双发射？")
    plans = build_retrieval_plans(value)
    assert len({plan.subtask_id for plan in plans}) == len(plans)
    assert len({plan.purpose for plan in plans}) == len(plans)


def test_compare_material_branches_keep_independent_plans() -> None:
    _, value = _generation_input(
        "比较 YSZ:Dy³⁺ 和 YAG:Dy³⁺ 在白光中的表现"
    )
    plans = {plan.subtask_id: plan for plan in build_retrieval_plans(value)}
    first = plans["collect_first_material_evidence"]
    second = plans["collect_second_material_evidence"]
    assert first is not second
    assert "YSZ" in first.purpose
    assert "YAG" in second.purpose
    assert first.rewritten_queries != second.rewritten_queries


def test_evaluate_rewrite_targets_information_gaps() -> None:
    _, value = _generation_input("3000 K是否一定更加健康？")
    plan = build_retrieval_plans(value)[0]
    joined = " ".join(plan.rewritten_queries).lower()
    assert len(plan.rewritten_queries) == 3
    assert "spectral power distribution" in joined
    assert "blue light hazard" in joined
    assert "exposure" in joined


def test_rewrite_preserves_entities_and_marks_expansions() -> None:
    _, value = _generation_input("基质材料如何影响 Dy³⁺ 发光性能？")
    plan = build_retrieval_plans(value)[0]
    assert any("Dy" in entity for entity in plan.entities)
    assert "声子能量" in plan.expansion_terms
    assert all(term not in plan.entities for term in plan.expansion_terms)


def test_retrieval_need_is_not_a_boolean_only() -> None:
    _, value = _generation_input("3000 K是否一定更加健康？")
    need = build_retrieval_need(value, value.subtask)
    assert need.required
    assert need.information_gap
    assert need.target_parameters
    assert need.coverage_requirement == "multi-concept"


def test_explicit_activator_mismatch_is_filtered() -> None:
    _, value = _generation_input("为什么 Dy³⁺ 会发光？")
    plan = build_retrieval_plan(value, value.subtask)
    candidates = [
        _candidate("dy", "Dy3+ 能级跃迁发光"),
        _candidate("eu", "Eu3+ 红色发光", entity="Eu3+"),
    ]
    assert [item["chunk_id"] for item in hard_filter(plan, candidates)] == ["dy"]


def test_english_words_do_not_become_false_luminescent_ions() -> None:
    _, value = _generation_input("为什么 Dy³⁺ 会发光？")
    plan = build_retrieval_plan(value, value.subtask)
    candidate = _candidate(
        "dy-review",
        "Dy3+ evidence for Reviewer checked Answer A",
    )

    assert hard_filter(plan, [candidate]) == [candidate]


def test_project_authored_outline_is_not_scientific_evidence() -> None:
    _, value = _generation_input("Dy³⁺黄蓝发射强度比会影响什么？")
    plan = build_retrieval_plan(value, value.subtask)
    derived_outline = _candidate(
        "outline",
        "黄蓝比决定白光色温",
        source="dy-稀土化学导论-教材知识大纲",
    )
    paper = _candidate(
        "paper",
        "Adjustment of the yellow-to-blue ratio produces white light.",
        source="dy-MinerU_markdown_peer_reviewed_paper",
    )

    assert hard_filter(plan, [derived_outline, paper]) == [paper]


def test_project_authored_outline_in_nested_metadata_is_not_evidence() -> None:
    _, value = _generation_input("Dy³⁺黄蓝发射强度比会影响什么？")
    plan = build_retrieval_plan(value, value.subtask)
    derived_outline = _candidate("outline", "黄蓝比决定白光色温", source="")
    derived_outline["metadata"]["source"] = "dy-稀土化学导论-教材知识大纲"

    assert hard_filter(plan, [derived_outline]) == []


def test_curated_summary_requires_reviewed_traceable_source() -> None:
    _, value = _generation_input("热猝灭的物理机制是什么？")
    plan = build_retrieval_plan(value, value.subtask)
    valid = _candidate("valid", "热猝灭由非辐射通道增强导致。", source="reviewed-summary")
    valid["metadata"].update({
        "source_type": "curated_source_summary",
        "source_uri": "https://doi.org/10.1039/D2TC04439K",
        "evidence_status": "reviewed",
    })
    missing_source = _candidate("missing", "热猝灭由非辐射通道增强导致。")
    missing_source["metadata"].update({
        "source_type": "curated_source_summary",
        "evidence_status": "reviewed",
    })
    unreviewed = _candidate("unreviewed", "热猝灭由非辐射通道增强导致。")
    unreviewed["metadata"].update({
        "source_type": "curated_source_summary",
        "source_uri": "https://doi.org/10.1039/D2TC04439K",
        "evidence_status": "candidate",
    })

    assert hard_filter(plan, [valid, missing_source, unreviewed]) == [valid]


def test_correct_material_candidate_ranks_first() -> None:
    _, value = _generation_input("比较 YSZ:Dy³⁺ 和 YAG:Dy³⁺ 的发光表现")
    plan = next(
        item for item in build_retrieval_plans(value)
        if item.subtask_id == "collect_first_material_evidence"
    )
    candidates = [
        _candidate("yag", "YAG:Dy3+ 发射光谱", entity="YAG:Dy3+"),
        _candidate("ysz", "YSZ:Dy3+ 发射 色度 效率 热稳定性", entity="YSZ:Dy3+"),
    ]
    ranked = agent_aware_rerank(plan, candidates, [0.5, 0.4])
    assert ranked[0][0]["chunk_id"] == "ysz"


def test_duplicate_evidence_receives_penalty() -> None:
    _, value = _generation_input("哪些参数决定 Dy³⁺ 材料的发光效率？")
    plan = build_retrieval_plans(value)[0]
    duplicate = "Dy3+ 发光效率受辐射跃迁和非辐射损失影响"
    ranked = agent_aware_rerank(
        plan,
        [_candidate("a", duplicate), _candidate("b", duplicate)],
        [0.5, 0.5],
    )
    assert any("duplicate_penalty=0.35" in reason for reason in ranked[1][2])


def test_agent_aware_rerank_changes_base_order() -> None:
    _, value = _generation_input("哪些参数决定 Dy³⁺ 材料的发光效率？")
    plan = build_retrieval_plans(value)[0]
    candidates = [
        _candidate("irrelevant", "LED PN结形貌与芯片封装"),
        _candidate("relevant", "Dy3+ 量子效率由辐射跃迁与非辐射损失竞争，受缺陷和温度影响"),
    ]
    ranked = agent_aware_rerank(plan, candidates, [0.8, 0.2])
    assert ranked[0][0]["chunk_id"] == "relevant"
    assert any(reason.startswith("final=") for reason in ranked[0][2])


def test_evidence_pack_binds_task_and_subtask() -> None:
    _, value = _generation_input("基质材料如何影响 Dy³⁺ 发光性能？")
    plan = build_retrieval_plans(value)[0]
    ranked = agent_aware_rerank(
        plan,
        [_candidate("host", "Dy3+ 基质局域晶场 声子非辐射 缺陷占位 热稳定性")],
        [0.5],
    )
    pack = build_evidence_pack(plan, ranked)
    assert pack.task_id == plan.task_id
    assert pack.subtask_id == plan.subtask_id
    assert pack.items[0].rerank_reasons


def test_no_candidates_produces_missing_information_not_fake_evidence() -> None:
    _, value = _generation_input("3000 K是否一定更加健康？")
    plan = build_retrieval_plans(value)[0]
    pack = build_evidence_pack(plan, [])
    assert pack.items == ()
    assert pack.missing_information == plan.required_evidence_types


def test_metrics_capture_coverage_duplicate_and_source_diversity() -> None:
    _, value = _generation_input("哪些参数决定 Dy³⁺ 材料的发光效率？")
    plan = build_retrieval_plans(value)[0]
    ranked = agent_aware_rerank(
        plan,
        [
            _candidate("a", "辐射跃迁 非辐射损失", "doc-a"),
            _candidate("b", "浓度猝灭 热猝灭 缺陷", "doc-b"),
        ],
        [0.5, 0.4],
    )
    metrics = retrieval_metrics(build_evidence_pack(plan, ranked))
    assert metrics["duplicate_rate"] == 0.0
    assert metrics["source_diversity"] == 1.0
    assert metrics["relevant_at_k"] == 1.0


class _Retriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, **_kwargs):
        self.calls.append(query)
        return SimpleNamespace(results=list(self.results), scores=[0.2] * len(self.results))


class _Reranker:
    def rerank_result(self, _query, result, top_k=10):
        return SimpleNamespace(results=result.results[:top_k], scores=result.scores[:top_k])


def test_runtime_attaches_separate_packs_to_generation_input() -> None:
    context, value = _generation_input("比较 YSZ:Dy³⁺ 和 YAG:Dy³⁺ 的发光表现")
    retriever = _Retriever([
        _candidate("ysz", "YSZ:Dy3+ 发射 色度 效率", "doc-ysz", "YSZ:Dy3+"),
        _candidate("yag", "YAG:Dy3+ 发射 色度 热稳定", "doc-yag", "YAG:Dy3+"),
    ])
    deps = AgentDependencies(hybrid_retriever=retriever, reranker=_Reranker())
    updated, payload = agent_workers._prepare_generation_retrieval(
        context, value, {"query": value.user_query}, deps
    )
    assert updated.evidence_pack
    assert all(isinstance(pack, EvidencePack) for pack in updated.evidence_pack)
    assert len({pack.subtask_id for pack in updated.evidence_pack}) == len(updated.evidence_pack)
    assert payload["_retrieval_plans_applied"] is True
    assert context.tool_results["evidence_packs"] == updated.evidence_pack


def test_runtime_preserves_reviewed_concept_evidence_from_large_lexical_pool() -> None:
    context, value = _generation_input("热猝灭的物理机制是什么？")
    store = KnowledgeStore()
    for index in range(60):
        store.add_chunk(DocumentChunk(
            chunk_id=f"noise-{index}",
            document_id=f"paper-noise-{index}",
            content=f"文献{index}提到热猝灭，但这个片段没有机制证据。",
        ))
    load_curated_concept_evidence(store)
    knowledge = build_knowledge_learning_context(
        learner_id="r03d",
        query=value.user_query,
        l3_store=store,
    )
    value = replace(
        value,
        learner_context={
            **dict(value.learner_context),
            "knowledge_learning_context": knowledge,
        },
    )
    deps = AgentDependencies(
        l3_store=store,
        hybrid_retriever=_Retriever([]),
        reranker=_Reranker(),
    )

    updated, payload = agent_workers._prepare_generation_retrieval(
        context, value, {"query": value.user_query}, deps
    )

    evidence_ids = {
        item.chunk_reference
        for pack in updated.evidence_pack
        for item in pack.items
    }
    curated_ids = {
        chunk.chunk_id
        for chunk in load_curated_concept_evidence(store)
        if chunk.document_id == "curated-concept-evidence:thermal-quenching-routes"
    }
    assert evidence_ids.intersection(curated_ids)
    planned = payload["_planned_retrieval_result"]
    assert planned.results
    assert planned.results[0]["chunk_id"] in curated_ids


def test_runtime_uses_explicit_concept_mapping_across_synonyms() -> None:
    context, value = _generation_input("宿主晶格如何影响Dy³⁺发光性能？")
    store = KnowledgeStore()
    load_curated_concept_evidence(store)
    knowledge = build_knowledge_learning_context(
        learner_id="r03d-host",
        query=value.user_query,
        l3_store=store,
    )
    value = replace(
        value,
        learner_context={
            **dict(value.learner_context),
            "knowledge_learning_context": knowledge,
        },
    )
    deps = AgentDependencies(
        l3_store=store,
        hybrid_retriever=_Retriever([]),
        reranker=_Reranker(),
    )

    _updated, payload = agent_workers._prepare_generation_retrieval(
        context,
        value,
        {"query": value.user_query},
        deps,
    )

    planned = payload["_planned_retrieval_result"]
    assert planned.results
    assert planned.results[0]["document_id"] == (
        "curated-concept-evidence:host-local-environment"
    )


def test_mechanism_question_is_not_reduced_by_definition_filter() -> None:
    items = [
        {
            "chunk_id": "mechanism",
            "content": (
                "热猝灭是温度升高时激发态通过非辐射通道失活，"
                "从而导致发光强度下降的现象。"
            ),
        },
        {
            "chunk_id": "generic-definition",
            "content": "电子构型的定义用于介绍稀土离子的基本性质。",
        },
    ]

    answer = agent_workers._compose_concise_answer(
        "热猝灭的物理机制是什么？",
        items,
        focus_terms=("热猝灭", "非辐射弛豫"),
    )

    assert "热猝灭" in answer
    assert "非辐射" in answer


def test_reviewed_direct_concept_evidence_excludes_unreviewed_neighbours() -> None:
    direct = {
        "chunk_id": "reviewed-direct",
        "content": "缺陷态可通过非辐射复合削弱发光。",
        "metadata": {
            "source_type": "curated_source_summary",
            "evidence_status": "reviewed",
            "source_uri": "kb://dy3/chunks/defect",
            "concept_ids": ["concept:dy3:defects-traps"],
        },
    }
    related = {
        "chunk_id": "reviewed-related",
        "content": "多声子弛豫是非辐射过程。",
        "metadata": {
            "source_type": "curated_source_summary",
            "evidence_status": "reviewed",
            "source_uri": "https://doi.org/10.0000/example",
            "concept_ids": ["concept:dy3:nonradiative-relaxation"],
        },
    }
    neighbour = {
        "chunk_id": "unreviewed-neighbour",
        "content": "只因关键词相邻而召回的旧论文段落。",
    }

    selected = agent_workers._prefer_reviewed_concept_evidence(
        [neighbour, related, direct],
        ("concept:dy3:defects-traps",),
    )

    assert [item["chunk_id"] for item in selected] == [
        "reviewed-direct",
        "reviewed-related",
    ]


def test_reviewer_uses_generation_referenced_pack_evidence() -> None:
    context, value = _generation_input("为什么 Dy³⁺ 会产生黄蓝双发射？")
    plan = build_retrieval_plans(value)[0]
    pack = build_evidence_pack(
        plan,
        agent_aware_rerank(plan, [_candidate("used", "Dy3+ 黄蓝跃迁")], [0.5]),
    )
    generation = SimpleNamespace(evidence_refs=("used",))
    reviewer_input = SimpleNamespace(evidence_pack=(pack,))
    assert agent_workers._review_evidence_texts(reviewer_input, generation) == ["Dy3+ 黄蓝跃迁"]


def test_private_contracts_do_not_json_serialize() -> None:
    _, value = _generation_input("基质材料如何影响 Dy³⁺ 发光性能？")
    plan = build_retrieval_plans(value)[0]
    pack = build_evidence_pack(plan, [])
    with pytest.raises(TypeError):
        json.dumps(plan)
    with pytest.raises(TypeError):
        json.dumps(pack)
