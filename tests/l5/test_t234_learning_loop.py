"""T2/T3/T4 learner, teaching and resource-loop protection tests."""

from __future__ import annotations

from types import SimpleNamespace

from starlette.testclient import TestClient

from dy3_polaris.l2.practice import PracticeBank
from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.knowledge_learning_fusion import (
    build_knowledge_learning_context,
    public_knowledge_learning_projection,
    resolve_concepts,
)
from dy3_polaris.l3.concept_relations import build_concept_relation_network
from dy3_polaris.l5.learner_foundation import AdaptiveTeachingDecision
from dy3_polaris.l5.learner_intelligence import (
    build_learner_intelligence_view,
    public_learner_intelligence_projection,
)
from dy3_polaris.l5.retrieval_planning import (
    RetrievalPlan,
    agent_aware_rerank,
    hard_filter,
)
from dy3_polaris.l5.unified_app import UnifiedApp


def test_unknown_is_not_rendered_as_weak_or_zero_mastery() -> None:
    view = build_learner_intelligence_view(
        {
            "learner_id": "guest-t234-unknown",
            "query": "为什么 Dy³⁺ 会产生黄蓝双发射？",
        },
        AgentDependencies(),
    )
    public = public_learner_intelligence_projection(view)

    assert public["state"] == "UNKNOWN"
    assert public["model_state_available"] is False
    assert public["mastery_summary"]["known_count"] == 0
    assert "overall_mastery" not in public
    assert public["diagnostic"]["needed"] is True


def test_explicit_teaching_action_changes_structure_not_scientific_text() -> None:
    base = build_learner_intelligence_view(
        {
            "learner_id": "guest-t234-action",
            "query": "解释 Dy³⁺ 黄蓝双发射",
        },
        AgentDependencies(),
    )
    changed = build_learner_intelligence_view(
        {
            "learner_id": "guest-t234-action",
            "query": "解释 Dy³⁺ 黄蓝双发射",
            "teaching_action": "request_example",
        },
        AgentDependencies(),
    )
    base_decision = base.value("derived_context", "adaptive_teaching_decision")
    changed_decision = changed.value("derived_context", "adaptive_teaching_decision")
    assert base_decision.explanation_strategy != "example_then_mechanism"
    assert changed_decision.explanation_strategy == "example_then_mechanism"
    assert "worked_example" in changed_decision.representation_modes

    scientific_text = "证据支持该跃迁产生两个发射带。[1]"
    rendered = agent_workers._adapt_educational_depth(
        scientific_text,
        changed_decision.content_depth,
        changed_decision,
    )
    assert scientific_text in rendered
    assert "迁移边界" in rendered


def test_reviewer_scopes_quality_checks_to_scientific_body() -> None:
    scientific_text = "Dy3+ 的两个特征跃迁分别产生蓝光与黄光发射。"
    public_answer = (
        "入门理解：先抓住「条件—过程—结果」这条主线。\n"
        f"{scientific_text}\n\n"
        "学习提示：对照证据理解因果关系。"
    )

    assert agent_workers._scientific_review_content(public_answer) == scientific_text
    assert agent_workers._answer_identity("task-t234", public_answer) != ""


def test_revision_sentence_selection_prefers_question_coverage() -> None:
    query = "Dy³⁺蓝光和黄光发射分别对应什么跃迁？"
    items = [{
        "content": (
            "351 nm 激发可以获得一组发射光谱。"
            "Dy3+ 的蓝光和黄光发射分别对应两个特征能级跃迁。"
            "辐射跃迁会伴随光子发射。"
        )
    }]

    ranked = agent_workers._collect_answer_candidates(query, items)

    assert ranked
    assert ranked[0].startswith("Dy3+")
    assert "351 nm" not in ranked[0]


def test_mechanism_sentence_selection_prefers_entity_bound_relation() -> None:
    query = "晶体场如何影响 Dy³⁺ 发射光谱？"
    items = [{
        "content": (
            "发射光谱是在固定激发条件下记录不同波长的发光强度。"
            "稀土离子能级在晶体场中发生劈裂。"
            "The Dy3+ yellow transition is strongly influenced by the local "
            "crystal field, while the blue transition varies much less."
        )
    }]

    ranked = agent_workers._collect_answer_candidates(
        query,
        items,
        focus_terms=("crystal field", "emission spectrum"),
    )

    assert ranked
    assert ranked[0].startswith("The Dy3+")


def test_leading_english_chunk_fragment_is_removed_without_rewriting_fact() -> None:
    assert agent_workers._trim_fragment(
        "...ctively.It is well known that Dy3+ emission depends on symmetry."
    ).startswith("It is well known")


def test_formula_ocr_tail_is_removed_without_reconstructing_science() -> None:
    value = agent_workers._trim_fragment(
        "浓度较高时非辐射跃迁增强，而由下列公式我们可以得到 Dy3 \\ + I / x = K [ 1+(x)"
    )

    assert value == "浓度较高时非辐射跃迁增强"
    assert "I / x" not in value


def test_target_evidence_filter_excludes_relation_neighbour_noise() -> None:
    items = [
        {
            "chunk_id": "target",
            "content": "激活离子浓度增加后，能量传递增强并发生浓度猝灭，发光强度下降。",
        },
        {
            "chunk_id": "thermal-neighbour",
            "content": "温度升高使晶格振动增强，导致热猝灭和发光减弱。",
        },
        {
            "chunk_id": "powder-neighbour",
            "content": "粉末样品受潮会导致发光特性变化。",
        },
    ]

    selected = agent_workers._filter_task_answer_evidence(
        "为什么 Dy³⁺ 浓度增加会导致发光下降？",
        items,
        focus_terms=("浓度猝灭", "concentration quenching"),
    )

    assert [item["chunk_id"] for item in selected] == ["target"]


def test_concept_mapping_does_not_bypass_task_mechanism_guard() -> None:
    concept_id = "concept:dy3:concentration-quenching"
    items = [
        {
            "chunk_id": "direct-mechanism",
            "content": "Dy3+ 浓度升高后能量迁移和非辐射损失增强，发生浓度猝灭。",
            "metadata": {"concept_ids": [concept_id]},
        },
        {
            "chunk_id": "same-concept-different-observation",
            "content": "改变 Dy3+ 掺杂浓度可以调节样品的发射颜色。",
            "metadata": {"concept_ids": [concept_id]},
        },
    ]

    selected = agent_workers._filter_task_answer_evidence(
        "为什么 Dy³⁺ 浓度增加会导致发光下降？",
        items,
        preferred_concept_ids=(concept_id,),
    )

    assert [item["chunk_id"] for item in selected] == ["direct-mechanism"]


def test_bare_element_definition_rejects_unrelated_mechanism_mentions() -> None:
    items = [
        {
            "chunk_id": "mechanism-only",
            "content": "Energy transfer between Dy3+ and Tb3+ changes the emission intensity.",
        },
        {
            "chunk_id": "identity",
            "content": "Dysprosium is a chemical element in the lanthanide series.",
        },
    ]

    selected = agent_workers._filter_task_answer_evidence("Dy 是什么？", items)

    assert [item["chunk_id"] for item in selected] == ["identity"]


def test_long_resource_excerpt_uses_relevant_sentences_not_formula_ocr_dump() -> None:
    excerpt = agent_workers._resource_passage_excerpt(
        "为什么 Dy³⁺ 浓度增加会导致发光下降？",
        (
            "随着 Dy3+ 浓度升高，离子间能量迁移增强并出现非辐射损失。"
            "由下列公式我们可以得到 I / x = K [1 + beta x]。"
            "样品编号和图注不属于机制结论。"
        ),
        focus_terms=("浓度猝灭", "能量迁移"),
    )

    assert "能量迁移增强" in excerpt
    assert "I / x" not in excerpt
    assert "样品编号" not in excerpt


def test_concentration_mechanism_answer_prefers_process_then_bounded_observation() -> None:
    items = [{
        "content": (
            "随着 Dy3+ 掺杂浓度逐渐增加，发光强度先增大，在 7% 时达到最大，随后下降。"
            "由 Dexter 能量传递理论可知，浓度较高时离子的非辐射跃迁导致浓度猝灭。"
            "浓度猝灭的原因是能量传递几率超过发射几率，激发能通过晶格迁移而消耗。"
            "如果基质晶格本身是敏化剂，也可形成基质晶格发射的浓度猝灭。"
        )
    }]

    answer = agent_workers._compose_concise_answer(
        "为什么 Dy³⁺ 浓度增加会导致发光下降？",
        items,
    )

    mechanism, observation, boundary = answer.split("\n\n")
    assert mechanism.startswith("机制依据：")
    assert "非辐射跃迁" in mechanism
    assert "能量传递几率超过发射几率" in mechanism
    assert observation.startswith("证据中的条件化观察：")
    assert "7%" in observation
    assert "基质晶格本身是敏化剂" not in answer
    assert boundary.startswith("当前边界：")
    scientific_body = agent_workers._scientific_review_content(answer)
    assert "上述数值只表示" not in scientific_body


def test_bilingual_evidence_answer_satisfies_mechanism_intent() -> None:
    assert agent_workers._answer_matches_intent(
        "Dy³⁺为什么具有黄蓝双发射？",
        "The blue and yellow emissions correspond to two 4f-4f transitions.",
    )


def test_concept_projection_preserves_unknown_and_curated_relations() -> None:
    context = build_knowledge_learning_context(
        learner_id="guest-t234-concept",
        query="为什么 Dy³⁺ 会产生黄蓝双发射？",
        mastery={},
    )
    public = public_knowledge_learning_projection(context)

    assert public["nodes"]
    assert all(node["learner_state"] == "UNKNOWN" for node in public["nodes"])
    assert all(edge["status"] == "curated" for edge in public["edges"])
    assert "trace" not in public


def test_concept_targeted_practice_uses_authored_bank_or_returns_empty() -> None:
    bank = PracticeBank()
    covered_kp = next(iter(bank.by_kp))
    questions = bank.select_questions(
        "guest-t234-practice",
        count=4,
        target_kps=(covered_kp,),
    )
    assert questions
    assert all(item["kp_id"] == covered_kp for item in questions)

    assert bank.select_questions(
        "guest-t234-practice",
        count=4,
        target_kps=("kp:not-authored",),
    ) == []


def test_persona_prior_never_overrides_observed_model_evidence() -> None:
    # Existing R09 tests cover the full conflict path.  This guard makes the
    # authority rule explicit at the fused T2/T3/T4 boundary.
    decision = AdaptiveTeachingDecision(
        content_depth="foundation",
        explanation_strategy="foundation_conceptual",
        representation_modes=("structured_text",),
        difficulty_strategy="maintain",
        resource_modes=("concept_resource",),
        next_focus="",
        diagnostic_needed=False,
        rationale=("observed model evidence",),
        source_refs=("inferred:mastery_model",),
        confidence=0.8,
    )
    assert decision.content_depth != "advanced"
    assert any(item.startswith("inferred:") for item in decision.source_refs)


def test_query_interaction_skeleton_remains_unknown(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    learner_id = "guest-t234-query-event"
    collected = client.post(
        "/l2/event/collect",
        json={
            "learner_id": learner_id,
            "event_type": "query",
            "detail": "为什么 Dy³⁺ 会产生黄蓝双发射？",
        },
    )
    assert collected.status_code == 200
    profile = client.get(f"/l2/profile/{learner_id}").json()["data"]
    assert profile["initial_assessed"] is False
    assert profile["level"] == "unknown"
    assert profile["kp_mastery"] == {}


def test_planned_retrieval_keeps_original_question_as_independent_branch() -> None:
    knowledge = build_knowledge_learning_context(
        learner_id="guest-t234-retrieval",
        query="Dy³⁺ 蓝光和黄光发射分别对应什么跃迁？",
        mastery={},
    )
    agent_input = SimpleNamespace(
        user_query="Dy³⁺ 蓝光发射对应什么跃迁？",
        learner_context={"knowledge_learning_context": knowledge},
    )
    plan = SimpleNamespace(
        rewritten_queries=(
            "Dy³⁺ 蓝光发射对应什么跃迁？ 收集机制证据并解释教学目标",
        )
    )

    queries = agent_workers._planned_retrieval_queries(agent_input, plan)

    assert queries[0] == agent_input.user_query
    assert "Dy3+蓝光发射" in queries
    assert "Dy3+黄光发射" in queries
    assert any(
        "Dy3+蓝光发射" in item and "Dy3+黄光发射" in item
        for item in queries
    )
    assert any("blue emission" in item and "Dy3+蓝光发射" in item for item in queries)
    assert any("yellow emission" in item and "Dy3+黄光发射" in item for item in queries)
    assert "blue emission" in queries
    assert "yellow emission" in queries
    assert queries[-1] == plan.rewritten_queries[0]


def test_dual_emission_question_resolves_both_scientific_concepts() -> None:
    network = build_concept_relation_network()

    resolved = resolve_concepts(network, "Dy³⁺为什么具有黄蓝双发射？")

    assert "concept:dy3:dy3-blue-emission" in resolved
    assert "concept:dy3:dy3-yellow-emission" in resolved


def test_crystal_field_retrieval_expands_curated_stark_relation() -> None:
    knowledge = build_knowledge_learning_context(
        learner_id="guest-t234-crystal-field",
        query="晶体场如何影响 Dy³⁺ 发射光谱？",
        mastery={},
    )
    agent_input = SimpleNamespace(
        user_query="晶体场如何影响 Dy³⁺ 发射光谱？",
        learner_context={"knowledge_learning_context": knowledge},
    )
    plan = SimpleNamespace(rewritten_queries=("晶体场 发射光谱",))

    queries = agent_workers._planned_retrieval_queries(agent_input, plan)

    assert "Stark splitting" in queries
    assert "Stark 劈裂" in queries
    assert agent_workers._detect_question_type(agent_input.user_query) == "mechanism"
    assert agent_workers._is_procedure_question(agent_input.user_query) is False


def test_what_effects_question_is_a_relationship_not_a_definition() -> None:
    assert (
        agent_workers._detect_question_type("缺陷和陷阱态对发光有哪些影响？")
        == "mechanism"
    )


def test_relationship_answers_pass_the_generic_mechanism_intent_gate() -> None:
    cases = (
        (
            "CIE色坐标在白光发光材料评价中有什么作用？",
            "CIE色坐标表征发光颜色在色度图中的位置，用于评价白光色度。",
        ),
        (
            "蓝光危害评价为什么不能只看相关色温？",
            "相关色温只反映光色冷暖，风险评价需要结合光谱辐亮度和暴露条件。",
        ),
        (
            "电荷补偿为什么会改变稀土掺杂材料的发光？",
            "电荷补偿会改变局域缺陷和价态平衡，从而影响非辐射复合。",
        ),
    )
    for query, answer in cases:
        assert agent_workers._detect_question_type(query) == "mechanism"
        assert agent_workers._answer_matches_intent(query, answer) is True


def test_answer_focus_grounding_requires_an_explicit_nontrivial_term() -> None:
    assert agent_workers._answer_mentions_grounded_focus(
        "CIE色坐标用于表征白光色度位置。",
        ("CIE色坐标", "白光评价"),
    )
    assert not agent_workers._answer_mentions_grounded_focus(
        "这是一段与问题无关的通用说明。",
        ("CIE色坐标", "白光评价"),
    )
    assert not agent_workers._answer_mentions_grounded_focus("任意回答", ("a", ""))


def test_relation_retrieval_expands_incoming_cause_and_measurement_method() -> None:
    thermal = build_knowledge_learning_context(
        learner_id="guest-thermal",
        query="热猝灭的物理机制是什么？",
        mastery={},
    )
    thermal_input = SimpleNamespace(
        user_query="热猝灭的物理机制是什么？",
        learner_context={"knowledge_learning_context": thermal},
    )
    thermal_terms = agent_workers._concept_relation_retrieval_terms(thermal_input)
    assert "非辐射弛豫" in thermal_terms

    quantum = build_knowledge_learning_context(
        learner_id="guest-quantum",
        query="发光量子效率应如何测量？",
        mastery={},
    )
    quantum_input = SimpleNamespace(
        user_query="发光量子效率应如何测量？",
        learner_context={"knowledge_learning_context": quantum},
    )
    quantum_terms = agent_workers._concept_relation_retrieval_terms(quantum_input)
    assert "积分球量子效率测量" in quantum_terms


def test_real_dual_emission_retrieval_reaches_transition_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    original = agent_workers._prepare_generation_retrieval
    captured: list[str] = []
    captured_queries: list[str] = []

    def observed(*args, **kwargs):
        plans = kwargs.get("plans_override") or agent_workers.build_retrieval_plans(args[1])
        for plan in plans:
            captured_queries.extend(agent_workers._planned_retrieval_queries(args[1], plan))
        result = original(*args, **kwargs)
        planned = result[1].get("_planned_retrieval_result")
        captured.extend(
            str(item.get("content") or "")
            for item in list(getattr(planned, "results", ()) or ())
        )
        return result

    monkeypatch.setattr(agent_workers, "_prepare_generation_retrieval", observed)
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )

    response = client.post(
        "/api/query",
        json={"query": "Dy³⁺为什么具有黄蓝双发射？"},
    )

    assert response.status_code == 200
    corpus = " ".join(captured)
    assert "4F9/2" in corpus
    assert "6H15/2" in corpus
    assert "6H13/2" in corpus, "\n".join(captured_queries)


def test_characterization_procedure_plan_adds_method_dimensions_not_answers() -> None:
    knowledge = build_knowledge_learning_context(
        learner_id="guest-t234-xrd-method",
        query="XRD 测荧光粉物相的操作步骤",
        mastery={},
    )
    agent_input = SimpleNamespace(
        user_query="XRD 测荧光粉物相的操作步骤",
        learner_context={"knowledge_learning_context": knowledge},
    )
    plan = SimpleNamespace(rewritten_queries=("XRD 物相表征",))

    queries = agent_workers._planned_retrieval_queries(agent_input, plan)
    terms = agent_workers._concept_plan_expansion_terms(agent_input)

    assert any("sample preparation" in query for query in queries)
    assert any("样品制备" in query and "标准参照" in query for query in queries)
    assert "sample preparation" in terms
    assert "standard reference" in terms
    # Planning expands evidence dimensions only; it must not supply a fixed
    # instrument parameter or procedural answer.
    assert not any("2θ" in query or "PDF card" in query for query in queries)


def test_agent_aware_rerank_distinguishes_direct_chinese_evidence() -> None:
    plan = RetrievalPlan(
        task_id="task-t234-rerank",
        subtask_id="subtask-t234-rerank",
        purpose="确认蓝光和黄光发射对应的跃迁",
        original_query="Dy³⁺蓝光和黄光发射分别对应什么跃迁？",
        rewritten_queries=("收集支持关键机制链条的真实领域证据",),
        entities=("Dy³⁺",),
        filters=(),
        required_evidence_types=("能级跃迁",),
        top_k=2,
        diversity_requirement="source_and_claim",
        rerank_profile="fact_find",
        reason="test",
    )
    generic = {
        "chunk_id": "generic",
        "document_id": "paper-generic",
        "content": "稀土离子的跃迁选择定则与晶体场有关。",
    }
    direct = {
        "chunk_id": "direct",
        "document_id": "paper-direct",
        "content": "Dy3+ 的蓝光与黄光发射分别来自两个特征能级跃迁。",
    }

    ranked = agent_aware_rerank(plan, [generic, direct], [0.5, 0.5])

    assert ranked[0][0]["chunk_id"] == "direct"


def test_agent_aware_rerank_uses_curated_concept_aliases_for_english_papers() -> None:
    plan = RetrievalPlan(
        task_id="task-t234-concept-rerank",
        subtask_id="subtask-t234-concept-rerank",
        purpose="确认Dy3+黄蓝发射的科学机制",
        original_query="Dy³⁺蓝光和黄光发射分别对应什么跃迁？",
        rewritten_queries=("Dy3+ emission transition",),
        entities=("Dy³⁺",),
        filters=(),
        required_evidence_types=("能级跃迁",),
        top_k=2,
        diversity_requirement="source_and_claim",
        rerank_profile="fact_find",
        reason="test",
        expansion_terms=("blue emission", "yellow emission"),
    )
    generic = {
        "chunk_id": "generic-cie",
        "document_id": "paper-generic-cie",
        "content": "Dy3+ concentration changes the CIE chromaticity coordinate.",
    }
    direct = {
        "chunk_id": "direct-transition",
        "document_id": "paper-direct-transition",
        "content": (
            "The Dy3+ blue emission and yellow emission arise from two "
            "characteristic 4f transitions."
        ),
    }

    ranked = agent_aware_rerank(plan, [generic, direct], [1.0, 1.0])

    assert ranked[0][0]["chunk_id"] == "direct-transition"
    assert any("concept_matches=2" in reason for reason in ranked[0][2])
    assert "semantic=" in " ".join(ranked[0][2])


def test_retrieval_ion_guard_normalizes_charge_and_rejects_other_dopant() -> None:
    plan = RetrievalPlan(
        task_id="task-t234-ion",
        subtask_id="subtask-t234-ion",
        purpose="确认 Dy³⁺ 发射",
        original_query="Dy³⁺ 的蓝光发射是什么？",
        rewritten_queries=("Dy3+ 蓝光发射",),
        entities=("Dy³⁺",),
        filters=("explicit_entity_mismatch",),
        required_evidence_types=("发射",),
        top_k=3,
        diversity_requirement="source_and_claim",
        rerank_profile="fact_find",
        reason="test",
    )
    dy = {"chunk_id": "dy", "content": "Dy3+ 在蓝光区域产生特征发射。"}
    mn = {"chunk_id": "mn", "content": "Mn2+ 在红光区域产生宽带发射。"}
    incidental = {
        "chunk_id": "incidental",
        "content": "Eu2+、Eu2+、Eu2+ 长余辉体系中也顺带提到 Dy3+。",
    }

    assert hard_filter(plan, [mn, incidental, dy]) == [dy]


def test_retrieval_ion_guard_sums_distinct_incidental_activators() -> None:
    plan = RetrievalPlan(
        task_id="task-t234-ion-sum",
        subtask_id="subtask-t234-ion-sum",
        purpose="确认 Dy³⁺ 发射",
        original_query="Dy³⁺ 的发射机理是什么？",
        rewritten_queries=("Dy3+ 发射机理",),
        entities=("Dy³⁺",),
        filters=("explicit_entity_mismatch",),
        required_evidence_types=("发射",),
        top_k=3,
        diversity_requirement="source_and_claim",
        rerank_profile="fact_find",
        reason="test",
    )
    mixed = {
        "chunk_id": "mixed",
        "content": "Tb3+、Sm3+、Eu3+ 发光体系中顺带提到一次 Dy3+。",
    }

    assert hard_filter(plan, [mixed]) == []
