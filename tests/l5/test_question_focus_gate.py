"""Question-focus retrieval and Reviewer completeness regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from dy3_polaris.l3.concept_foundation import load_curated_concept_evidence
from dy3_polaris.l3.concept_relations import build_concept_relation_network
from dy3_polaris.l3.store import KnowledgeStore
from dy3_polaris.l5 import critic
from dy3_polaris.l5.agent_workers import _detect_question_type
from dy3_polaris.l5.knowledge_learning_fusion import resolve_concepts


def test_priority_questions_resolve_their_canonical_focus() -> None:
    network = build_concept_relation_network()

    yellow_blue = set(resolve_concepts(
        network,
        "Dy³⁺为什么具有黄蓝双发射？",
        limit=6,
    ))
    host = set(resolve_concepts(
        network,
        "宿主晶格如何影响Dy³⁺发光性能？",
        limit=6,
    ))

    assert {
        "concept:dy3:dy3-blue-emission",
        "concept:dy3:dy3-yellow-emission",
    }.issubset(yellow_blue)
    assert "concept:dy3:host-lattice" in host


def test_correct_mechanism_is_linked_to_question_concept(monkeypatch) -> None:
    monkeypatch.setattr(critic, "_llm_critique", lambda *_args, **_kwargs: None)
    answer = (
        "在多声子弛豫模型中，较高的基质声子能量使跨越同一能隙所需的"
        "声子数减少，因此会提高非辐射弛豫概率并削弱发光。"
    )

    result = critic.critique_answer(
        "基质声子能量为什么会影响稀土发光？",
        answer,
        [answer],
    )

    assert result["verdict"] == "pass"


def test_neighbouring_method_explanation_cannot_satisfy_core_question(monkeypatch) -> None:
    monkeypatch.setattr(critic, "_llm_critique", lambda *_args, **_kwargs: None)
    answer = (
        "共沉淀法是一种湿化学制备方法。"
        "溶胶-凝胶法通过分子尺度分散使组分混合均匀。"
    )
    evidence = (
        "共沉淀法把多种金属离子置于同一溶液并共同形成前驱体，"
        "因而有利于得到均匀的多组分前驱体。"
    )

    result = critic.critique_answer(
        "共沉淀法为什么有利于多组分发光材料前驱体的均匀混合？",
        answer,
        [evidence],
    )

    assert result["verdict"] in {"fix_completeness", "fix_relevance"}
    assert any(term in result["reason"] for term in ("问题核心", "检索焦点"))


def test_complete_coprecipitation_mechanism_passes(monkeypatch) -> None:
    monkeypatch.setattr(critic, "_llm_critique", lambda *_args, **_kwargs: None)
    answer = (
        "共沉淀法把多种金属离子置于同一溶液，并通过pH和沉淀剂共同控制"
        "前驱体形成；各组分在溶液尺度共同沉淀，因而有利于均匀混合。"
    )

    result = critic.critique_answer(
        "共沉淀法为什么有利于多组分发光材料前驱体的均匀混合？",
        answer,
        [answer],
    )

    assert result["verdict"] == "pass"


def test_necessity_explanation_counts_as_direct_mechanism(monkeypatch) -> None:
    monkeypatch.setattr(critic, "_llm_critique", lambda *_args, **_kwargs: None)
    answer = (
        "绿色健康照明不能由单一参数代表：能效、光品质和光生物安全"
        "约束不同目标，因此需要并行评价。"
    )

    result = critic.critique_answer(
        "绿色健康照明为什么需要同时考虑能效、光品质和光生物安全？",
        answer,
        [answer],
    )

    assert result["verdict"] == "pass"


def test_nested_parallel_wording_only_requires_explicit_objects(monkeypatch) -> None:
    monkeypatch.setattr(critic, "_llm_critique", lambda *_args, **_kwargs: None)
    answer = (
        "SEM用于观察颗粒表面形貌和晶粒尺寸；TEM用于观察纳米颗粒的"
        "内部微结构、晶格条纹和选区电子衍射。"
    )

    result = critic.critique_answer(
        "SEM与TEM在发光材料形貌和微结构表征中分别能提供什么信息？",
        answer,
        [answer],
    )

    assert result["verdict"] == "pass"


def test_analytical_use_and_compound_why_how_are_not_procedure_only() -> None:
    assert _detect_question_type(
        "L-S耦合与跃迁选择定则如何用于理解Dy³⁺光谱项？"
    ) == "other"
    assert _detect_question_type(
        "纳米发光材料的表面态为什么会造成损失，核壳结构可以如何缓解？"
    ) == "mechanism"


def test_governed_concept_evidence_keeps_real_local_source_ids() -> None:
    root = Path(__file__).resolve().parents[2]
    snapshot = (
        root
        / "src"
        / "dy3_polaris"
        / "l3"
        / "data"
        / "snapshots"
        / "snapshot_final"
        / "chunks.jsonl"
    )
    source_ids = {
        str(json.loads(line)["chunk_id"])
        for line in snapshot.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    store = KnowledgeStore()
    loaded = load_curated_concept_evidence(store)

    assert len(loaded) >= 31
    for chunk in loaded:
        for source_id in chunk.metadata.get("source_chunk_ids", ()):
            assert source_id in source_ids
