"""Day2 Golden Questions: educational depth and evidence-boundary checks."""

from __future__ import annotations

from dy3_polaris.l3.llm_synthesizer import _SYSTEM_PROMPT, _build_prompt
from dy3_polaris.l5.agent_workers import (
    _absolute_claim_boundary_answer,
    _adapt_educational_depth,
    _requires_retrieved_evidence,
)


def test_generation_prompt_preserves_teaching_and_fact_boundaries() -> None:
    prompt = _SYSTEM_PROMPT + _build_prompt(
        "Dy³⁺ 为什么具有黄蓝双发射？",
        ["Dy³⁺ 可见发射来自 4F9/2 能级向低能级的跃迁。"],
    )

    for required in (
        "直接回答核心问题",
        "物理/材料机制",
        "证据支持",
        "应用意义",
        "限制与不确定性",
        "下一步学习建议",
        "事实",
        "推理",
        "建议",
    ):
        assert required in prompt


def test_beginner_and_advanced_prompts_change_depth_not_evidence() -> None:
    query = "Dy³⁺ 浓度猝灭为什么发生？"
    evidence = ["激活离子间距缩小会增强能量迁移与非辐射损失。"]
    beginner = _build_prompt(query, evidence, "beginner")
    advanced = _build_prompt(query, evidence, "advanced")

    assert "本科入门" in beginner
    assert "必要前置概念" in beginner
    assert "研究生进阶" in advanced
    assert "参数权衡" in advanced
    assert evidence[0] in beginner and evidence[0] in advanced
    assert "不得降低事实标准" in beginner
    assert "保留相同事实边界" in advanced


def test_golden_material_questions_require_retrieved_evidence() -> None:
    for query in (
        "Dy³⁺为什么具有黄蓝双发射？",
        "Dy³⁺浓度猝灭为什么发生？",
        "基质材料如何影响发光性能？",
        "哪些参数决定发光效率？",
        "低色温是否一定安全？",
    ):
        assert _requires_retrieved_evidence(query)

    assert not _requires_retrieved_evidence("2+3 等于多少？")


def test_low_cct_boundary_does_not_claim_safety() -> None:
    answer = _absolute_claim_boundary_answer("低色温照明是否一定安全？")

    assert "不能仅根据" in answer
    assert "光谱功率分布" in answer
    assert "暴露时间" in answer
    assert "一定安全" in answer


def test_material_comparison_boundary_refuses_unconditioned_ranking() -> None:
    answer = _absolute_claim_boundary_answer("没有数据时，A材料是否一定优于B材料？")

    assert "不能判定" in answer
    assert "同条件" in answer
    assert "量子效率" in answer
    assert "热稳定性" in answer


def test_educational_depth_changes_explanation_not_facts() -> None:
    fact = "Dy3+ 黄、蓝发射的相对强度会影响综合色度。"
    beginner = _adapt_educational_depth(fact, "beginner")
    advanced = _adapt_educational_depth(fact, "advanced")

    assert fact in beginner and fact in advanced
    assert "入门理解" in beginner
    assert "专业术语" in beginner
    assert "进阶分析" in advanced
    assert "测试条件" in advanced
    assert beginner != advanced
