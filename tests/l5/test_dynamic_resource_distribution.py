"""Dynamic, source-bounded long-form resource tests."""

from __future__ import annotations

from dataclasses import dataclass

from dy3_polaris.l3.llm_synthesizer import LLMSynthesizer
from dy3_polaris.l5.learning_resources import (
    ResourceFamily,
    ResourceSourceType,
    build_learning_resource_plan,
)
from dy3_polaris.l5.agent_workers import _requested_resource_character_target
from tests.l5.test_learning_resources import _decision, _knowledge, _release


@dataclass
class _ReadyConfig:
    temperature: float = 0.6
    max_tokens: int = 2048

    @staticmethod
    def is_ready() -> bool:
        return True


def test_explicit_long_form_length_is_understood_without_confusing_cct() -> None:
    assert _requested_resource_character_target("请生成不少于4200字的专题资源", 3200) == 4200
    assert _requested_resource_character_target("分析3000K照明", 2600) == 2600
    assert _requested_resource_character_target("写一份几千字专题长文", 2600) == 3200


def test_model_long_resource_uses_large_budget_and_source_bounded_prompt(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _chat(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return "## 专题\n\n" + "基于证据的机制解释。" * 260

    monkeypatch.setattr("dy3_polaris.l3.llm_config.chat_completion", _chat)
    content, used_model = LLMSynthesizer(_ReadyConfig()).synthesize_learning_resource(
        query="蓝光危害评价为什么不能只看相关色温？",
        reviewed_answer="已审核结论：相关色温不能单独代表蓝光危害。",
        evidence=["证据片段一", "证据片段二"],
        learner_level="research",
        teaching_strategy={"explanation_strategy": "evidence_first_mechanism"},
        target_characters=3800,
    )

    assert used_model is True
    assert len(content) >= 2000
    assert captured["max_tokens"] == 6144
    assert captured["disable_thinking"] is True
    user_prompt = captured["messages"][1]["content"]
    assert "3800" in user_prompt
    assert "已审核核心回答" in user_prompt
    assert "给定证据" in user_prompt


def test_reviewed_long_form_is_published_inside_existing_resource_family() -> None:
    long_text = "## 专题长文\n\n" + "已审核证据支持的学习内容。" * 260
    plan = build_learning_resource_plan(
        task_id="task-long-form",
        learner_id="learner-long-form",
        teaching_decision=_decision(research=True),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
        reviewed_long_form={
            "content": long_text,
            "generation_mode": "llm_evidence_synthesis",
            "model_used": True,
            "reviewer_executed": True,
            "review_verdict": "approved",
            "review_reason": "resource review passed",
            "target_characters": 3800,
            "source_passage_count": 6,
            "source_references": ("kb://dy3/chunks/source-1",),
            "retrieval_query_count": 3,
            "delivery_variant": "research_evidence_dossier",
        },
    )

    knowledge, practical, assessment = plan.resources
    assert knowledge.resource_family is ResourceFamily.KNOWLEDGE
    assert knowledge.source_type is ResourceSourceType.GENERATED
    assert knowledge.payload["guided_document"]["actual_characters"] == len(long_text)
    assert knowledge.payload["guided_document"]["review_verdict"] == "approved"
    assert knowledge.payload["guided_document"]["source_references"] == (
        "kb://dy3/chunks/source-1",
    )
    assert knowledge.payload["guided_document"]["retrieval_query_count"] == 3
    assert knowledge.payload["guided_document"]["collaboration_path"] == "Generation → Reviewer → Guidance"
    assert any(
        section["section_id"] == "reviewed-topic-long-form"
        for section in knowledge.payload["guided_document"]["sections"]
    )
    assert practical.payload["recommended"] is True
    assert knowledge.payload["recommended"] is False
    assert assessment.payload["recommended"] is False


def test_unreviewed_long_form_never_enters_public_resource() -> None:
    plan = build_learning_resource_plan(
        task_id="task-unreviewed-long-form",
        learner_id="learner-long-form",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
        reviewed_long_form={
            "content": "unreviewed model prose",
            "generation_mode": "llm_evidence_synthesis",
            "model_used": True,
            "reviewer_executed": True,
            "review_verdict": "needs_review",
        },
    )

    knowledge = plan.resources[0]
    assert knowledge.source_type is ResourceSourceType.DERIVED
    assert "unreviewed model prose" not in repr(knowledge.payload)
    assert all(
        section["section_id"] != "reviewed-topic-long-form"
        for section in knowledge.payload["guided_document"]["sections"]
    )


def test_deterministic_source_reader_is_not_published_as_generated_long_form() -> None:
    source_reader = "## 证据汇编\n\n" + "来源片段只可作为佐证。" * 260
    plan = build_learning_resource_plan(
        task_id="task-source-reader",
        learner_id="learner-source-reader",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
        reviewed_long_form={
            "content": source_reader,
            "generation_mode": "deterministic_reviewed_source_reader",
            "model_used": False,
            "reviewer_executed": True,
            "review_verdict": "approved",
        },
    )

    knowledge = plan.resources[0]
    assert knowledge.source_type is ResourceSourceType.DERIVED
    assert source_reader not in repr(knowledge.payload)
    assert all(
        section["section_id"] != "reviewed-topic-long-form"
        for section in knowledge.payload["guided_document"]["sections"]
    )
