"""Dynamic, source-bounded long-form resource tests."""

from __future__ import annotations

from dataclasses import dataclass

from dy3_polaris.l3.llm_synthesizer import LLMSynthesizer
from dy3_polaris.l5.learning_resources import (
    ResourceFamily,
    ResourceSourceType,
    build_learning_resource_plan,
)
from dy3_polaris.l5.agent_workers import (
    _long_form_resource_requested,
    _requested_resource_character_target,
)
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


def test_long_form_model_work_is_explicit_not_paid_on_every_short_query() -> None:
    assert _long_form_resource_requested("Dy³⁺为什么具有黄蓝双发射？") is False
    assert _long_form_resource_requested("请写一份完整讲义解释浓度猝灭") is True
    assert _long_form_resource_requested("请生成不少于4200字的专题资源") is True


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


def test_model_guided_questions_are_json_bounded_and_question_only(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _chat(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return (
            '{"questions":['
            '{"prompt":"交叉弛豫如何把浓度变化连接到非辐射损失？",'
            '"purpose":"SOCRATIC_MECHANISM"},'
            '{"prompt":"现有证据尚不能排除哪些测试条件差异？",'
            '"purpose":"EVIDENCE_BOUNDARY"}]}'
        )

    monkeypatch.setattr("dy3_polaris.l3.llm_config.chat_completion", _chat)
    questions, used_model = LLMSynthesizer(_ReadyConfig()).synthesize_guided_questions(
        query="为什么Dy³⁺浓度升高会发生猝灭？",
        reviewed_answer="已审核回答。",
        evidence=["交叉弛豫会随离子间距缩短而增强。"],
        concept_names=("浓度猝灭", "交叉弛豫"),
        prerequisite_names=("能量传递",),
        learner_level="intermediate",
        teaching_strategy={"explanation_strategy": "mechanism_with_context"},
    )

    assert used_model is True
    assert len(questions) == 2
    assert all(item["prompt"].endswith("？") for item in questions)
    assert captured["role"] == "generation_fast"
    assert "只生成问题" in captured["messages"][0]["content"]
    assert "给定证据" in captured["messages"][1]["content"]


def test_guided_question_parser_accepts_json_envelope_without_inventing_content() -> None:
    raw = (
        "以下为问题：\n```json\n"
        '{"questions":[{"prompt":"当前证据能支持哪一步机制判断？",'
        '"purpose":"EVIDENCE_BOUNDARY"}]}\n```\n请继续。'
    )

    parsed = LLMSynthesizer._parse_guided_questions(raw)

    assert parsed == ({
        "question_id": "model-guided-1",
        "prompt": "当前证据能支持哪一步机制判断？",
        "purpose": "EVIDENCE_BOUNDARY",
    },)


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


def test_only_reviewer_approved_model_questions_replace_relation_fallback() -> None:
    reviewed_questions = {
        "questions": (
            {
                "question_id": "model-guided-1",
                "prompt": "如果基质声子能量改变，当前机制判断需要补充什么证据？",
                "purpose": "TRANSFER_CHALLENGE",
            },
        ),
        "model_used": True,
        "reviewer_executed": True,
        "review_verdict": "approved",
        "review_reason": "question is evidence bounded",
        "source_passage_count": 2,
        "collaboration_path": "Generation → Reviewer → Guidance",
    }
    plan = build_learning_resource_plan(
        task_id="task-guided-question",
        learner_id="learner-guided-question",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
        reviewed_guided_questions=reviewed_questions,
    )

    knowledge = plan.resources[0]
    assert knowledge.payload["guided_question_mode"] == "llm_reviewed_socratic"
    assert knowledge.payload["guided_questions"][0]["source_class"] == (
        "GENERATION_REVIEWED_GUIDANCE"
    )
    assert knowledge.payload["guided_question_review"]["reviewer_executed"] is True

    rejected = build_learning_resource_plan(
        task_id="task-guided-question-rejected",
        learner_id="learner-guided-question",
        teaching_decision=_decision(),
        knowledge_context=_knowledge(),
        final_result=None,
        quality_release=_release(),
        reviewed_guided_questions={**reviewed_questions, "review_verdict": "needs_review"},
    )
    rejected_knowledge = rejected.resources[0]
    assert rejected_knowledge.payload["guided_question_mode"] == (
        "relation_backed_fallback"
    )
    assert all(
        item["source_class"] == "RELATION_BACKED_PROMPT"
        for item in rejected_knowledge.payload["guided_questions"]
    )


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
