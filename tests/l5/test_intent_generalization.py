"""意图识别泛化测试 — 边界模糊 + 领域外 + 词表外变体.

针对「只会 1+1 不会 114+256」的底层改造验证: 澄清判断必须识别「意图」而非「词」,
用规则覆盖不到的表达 (LLM 兜底) 也能泛化, 同时信息不足/纯元素仍正确澄清。
"""
from __future__ import annotations

import pytest

from dy3_polaris.l5.agent_workers import _detect_ambiguity, _has_clear_intent
from dy3_polaris.l5.task_understanding import understand_task

# 规则可识别的明确意图 (不依赖 LLM, 离线)
_RULE_INTENTS = [
    # 定义
    "dy是什么", "什么是镝", "Dy3+是什么", "镝是什么元素",
    # 方法 / 制备
    "dy怎么制备", "如何合成荧光粉",
    # 原因 / 机理
    "dy为什么发光", "为何会猝灭",
    # 比较
    "Dy和Eu的区别", "这俩有啥区别",
    # 数值
    "4F9/2跃迁波长是多少",
    # 复合
    "浓度猝灭是什么", "dy的发光机理",
]

# 信息不足 (纯元素 / 过短), 应澄清
_BARE_OR_SHORT = ["dy", "镝", "er", "铒", "嗯", "啊"]

# 词表外变体 (规则覆盖不了, 依赖 LLM 兜底语义识别)
_LLM_VARIANTS = [
    "dy是何方神圣", "镝是干嘛的", "dy啥意思", "镝指的是什么", "这俩有啥不同",
]


class TestRuleBasedIntent:
    """规则能识别的明确意图, 一律不澄清 (不依赖 LLM)."""

    @pytest.mark.parametrize("q", _RULE_INTENTS)
    def test_clear_intent_not_ambiguous(self, q: str) -> None:
        assert _has_clear_intent(q), f"{q!r} 应有明确意图, 不应澄清"


class TestBareOrShortClarifies:
    """纯元素 / 过短 → 信息不足, 应澄清 (零成本, 不调 LLM)."""

    @pytest.mark.parametrize("q", _BARE_OR_SHORT)
    def test_bare_or_short_clarifies(self, q: str) -> None:
        assert _detect_ambiguity(q) is not None, f"{q!r} 信息不足应澄清"


class TestAuthoritativeIntentGeneralization:
    """词表外变体由 R-03A authoritative IntentResult 统一解释。"""

    @pytest.mark.parametrize("q", _LLM_VARIANTS)
    def test_authoritative_intent_recognizes_colloquial_query(self, q: str) -> None:
        intent = understand_task(q, use_llm=False)
        assert intent.primary_intent
        assert "missing_subject" not in intent.ambiguity
        assert _has_clear_intent(q, intent)
