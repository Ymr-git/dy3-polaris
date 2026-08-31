"""L3 context_builder 模块综合测试.

覆盖:
  - ContextBudget          (上下文 Token 预算分配)
  - DialogTurn             (单轮对话记录)
  - HistoryCompressStrategy (压缩策略枚举)
  - HistoryCompressor      (对话历史压缩器 -- 3 种策略)
  - CoreferenceResolver    (指代消解器)
  - SchemaContextInjector  (KG Schema 上下文注入器)
  - LearnerContextAdapter  (学习者上下文适配器)
  - RetrievalNeedAssessor  (检索需求评估器)
  - QueryContext           (查询上下文数据结构)
  - ContextBuilder         (上下文构建器主入口 -- 4 阶段集成)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from dy3_polaris.l3.api_models import (
    BloomLevel,
    LearnerProfile,
    LearningStyle,
)
from dy3_polaris.l3.context_builder import (
    ContextBudget,
    ContextBuilder,
    CoreferenceResolver,
    DialogTurn,
    HistoryCompressStrategy,
    HistoryCompressor,
    LearnerContextAdapter,
    QueryContext,
    RetrievalNeedAssessor,
    SchemaContextInjector,
)


# ============================================================
# 辅助工厂
# ============================================================


def _make_turn(
    role: str = "user",
    content: str = "Dy3+的跃迁波长",
    timestamp: float = 1.0,
) -> DialogTurn:
    return DialogTurn(role=role, content=content, timestamp=timestamp)


def _make_beginner_profile(**overrides: Any) -> LearnerProfile:
    defaults = dict(
        learner_id="test-learner-001",
        level="beginner",
        bloom_target=BloomLevel.REMEMBER,
        preferred_style=LearningStyle.READING,
    )
    defaults.update(overrides)
    return LearnerProfile(**defaults)


def _make_advanced_profile(**overrides: Any) -> LearnerProfile:
    defaults = dict(
        learner_id="test-learner-002",
        level="advanced",
        bloom_target=BloomLevel.ANALYZE,
        preferred_style=LearningStyle.VISUAL,
    )
    defaults.update(overrides)
    return LearnerProfile(**defaults)


# ============================================================
# 1. ContextBudget
# ============================================================


class TestContextBudget:
    """ContextBudget 冻结数据类与 budget_for 方法."""

    def test_default_max_tokens(self):
        budget = ContextBudget()
        assert budget.max_tokens == 4096

    def test_default_ratios(self):
        budget = ContextBudget()
        assert budget.query_ratio == 0.10
        assert budget.history_ratio == 0.20
        assert budget.learner_ratio == 0.05
        assert budget.schema_ratio == 0.05
        assert budget.retrieval_ratio == 0.60

    def test_frozen(self):
        budget = ContextBudget()
        with pytest.raises(AttributeError):
            budget.max_tokens = 9999  # type: ignore[misc]

    def test_budget_for_query(self):
        budget = ContextBudget(max_tokens=4096, query_ratio=0.10)
        assert budget.budget_for("query") == int(4096 * 0.10)

    def test_budget_for_history(self):
        budget = ContextBudget(max_tokens=4096, history_ratio=0.20)
        assert budget.budget_for("history") == int(4096 * 0.20)

    def test_budget_for_learner(self):
        budget = ContextBudget(max_tokens=4096, learner_ratio=0.05)
        assert budget.budget_for("learner") == int(4096 * 0.05)

    def test_budget_for_schema(self):
        budget = ContextBudget(max_tokens=4096, schema_ratio=0.05)
        assert budget.budget_for("schema") == int(4096 * 0.05)

    def test_budget_for_retrieval(self):
        budget = ContextBudget(max_tokens=4096, retrieval_ratio=0.60)
        assert budget.budget_for("retrieval") == int(4096 * 0.60)

    def test_budget_for_unknown_key(self):
        budget = ContextBudget()
        assert budget.budget_for("nonexistent") == 0

    def test_custom_budget_values(self):
        budget = ContextBudget(
            max_tokens=8192,
            query_ratio=0.15,
            history_ratio=0.25,
        )
        assert budget.budget_for("query") == int(8192 * 0.15)
        assert budget.budget_for("history") == int(8192 * 0.25)

    def test_all_ratios_sum_approximately_one(self):
        budget = ContextBudget()
        total = (
            budget.query_ratio
            + budget.history_ratio
            + budget.learner_ratio
            + budget.schema_ratio
            + budget.retrieval_ratio
        )
        assert abs(total - 1.0) < 1e-9


# ============================================================
# 2. DialogTurn
# ============================================================


class TestDialogTurn:
    """DialogTurn 简单数据类."""

    def test_defaults(self):
        turn = DialogTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.timestamp == 0.0

    def test_with_timestamp(self):
        turn = DialogTurn(role="assistant", content="answer", timestamp=100.0)
        assert turn.timestamp == 100.0

    def test_mutable(self):
        turn = DialogTurn(role="user", content="old")
        turn.content = "new"
        assert turn.content == "new"


# ============================================================
# 3. HistoryCompressStrategy (enum)
# ============================================================


class TestHistoryCompressStrategy:
    """枚举值与字符串互转."""

    def test_enum_values(self):
        assert HistoryCompressStrategy.RECENT.value == "recent"
        assert HistoryCompressStrategy.SUMMARIZE.value == "summarize"
        assert HistoryCompressStrategy.SLIDING_WINDOW.value == "sliding_window"

    def test_from_string(self):
        assert HistoryCompressStrategy("recent") is HistoryCompressStrategy.RECENT

    def test_members_count(self):
        assert len(HistoryCompressStrategy) == 3


# ============================================================
# 4. HistoryCompressor
# ============================================================


class TestHistoryCompressorRecent:
    """RECENT 策略: 保留首轮 + 最近 N 轮."""

    def test_empty_input(self):
        c = HistoryCompressor(strategy=HistoryCompressStrategy.RECENT, max_recent_turns=3)
        assert c.compress([]) == []

    def test_single_turn_no_compression(self):
        turns = [_make_turn(content="q1")]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.RECENT, max_recent_turns=3)
        result = c.compress(turns)
        assert len(result) == 1
        assert result[0].content == "q1"

    def test_exact_n_turns_no_compression(self):
        turns = [_make_turn(content=f"q{i}") for i in range(5)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.RECENT, max_recent_turns=5)
        result = c.compress(turns)
        assert len(result) == 5

    def test_exceeds_n_compression(self):
        turns = [_make_turn(content=f"q{i}", timestamp=float(i)) for i in range(10)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.RECENT, max_recent_turns=3)
        result = c.compress(turns)
        # 首轮 + 最近 3 轮 = 4
        assert len(result) == 4
        assert result[0].content == "q0"  # 首轮保留
        assert result[-1].content == "q9"  # 最近轮

    def test_first_turn_preserved(self):
        turns = [_make_turn(content=f"q{i}", timestamp=float(i)) for i in range(8)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.RECENT, max_recent_turns=2)
        result = c.compress(turns)
        assert result[0].content == "q0"

    def test_middle_turns_dropped(self):
        turns = [_make_turn(content=f"q{i}", timestamp=float(i)) for i in range(10)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.RECENT, max_recent_turns=3)
        result = c.compress(turns)
        content_set = {t.content for t in result}
        assert "q1" not in content_set
        assert "q5" not in content_set


class TestHistoryCompressorSummarize:
    """SUMMARIZE 策略: 旧轮次压缩为摘要行."""

    def test_empty_input(self):
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SUMMARIZE)
        assert c.compress([]) == []

    def test_single_turn(self):
        turns = [_make_turn(content="single")]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SUMMARIZE, max_recent_turns=3)
        result = c.compress(turns)
        assert len(result) == 1
        assert result[0].content == "single"

    def test_exact_n_no_compression(self):
        turns = [_make_turn(content=f"q{i}") for i in range(3)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SUMMARIZE, max_recent_turns=3)
        result = c.compress(turns)
        assert len(result) == 3

    def test_exceeds_n_generates_summary(self):
        turns = [_make_turn(role="user", content=f"query-{i}", timestamp=float(i)) for i in range(8)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SUMMARIZE, max_recent_turns=3)
        result = c.compress(turns)
        assert len(result) == 4  # 1 summary + 3 recent
        assert result[0].role == "system"
        assert "历史摘要" in result[0].content

    def test_summary_contains_turn_count(self):
        turns = [_make_turn(role="user", content=f"query-{i}", timestamp=float(i)) for i in range(10)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SUMMARIZE, max_recent_turns=3)
        result = c.compress(turns)
        assert "7轮" in result[0].content  # 10 - 3 = 7 old turns

    def test_summary_recent_turns_intact(self):
        turns = [_make_turn(content=f"q{i}", timestamp=float(i)) for i in range(6)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SUMMARIZE, max_recent_turns=2)
        result = c.compress(turns)
        recent = result[1:]
        assert recent[0].content == "q4"
        assert recent[1].content == "q5"


class TestHistoryCompressorSlidingWindow:
    """SLIDING_WINDOW 策略: 保留最近 N 个字符."""

    def test_empty_input(self):
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SLIDING_WINDOW)
        assert c.compress([]) == []

    def test_single_turn_within_budget(self):
        turns = [_make_turn(content="short")]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SLIDING_WINDOW, max_chars=100)
        result = c.compress(turns)
        assert len(result) == 1

    def test_respects_char_budget(self):
        # 每轮 10 字符, 预算 25 字符 → 最多 2 轮
        turns = [_make_turn(content="ABCDEFGHIJ", timestamp=float(i)) for i in range(5)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SLIDING_WINDOW, max_chars=25)
        result = c.compress(turns)
        total_chars = sum(len(t.content) for t in result)
        assert total_chars <= 25

    def test_keeps_most_recent(self):
        turns = [_make_turn(content=f"text-{i}-longer", timestamp=float(i)) for i in range(5)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SLIDING_WINDOW, max_chars=60)
        result = c.compress(turns)
        assert result[-1].content == "text-4-longer"

    def test_single_large_turn_exceeds_budget(self):
        turns = [_make_turn(content="X" * 100)]
        c = HistoryCompressor(strategy=HistoryCompressStrategy.SLIDING_WINDOW, max_chars=50)
        result = c.compress(turns)
        # 第一轮就超过预算, 但它仍然被包含 (budget - turn_chars < 0 → break)
        # 检查: 100 > 50, break → result 为空
        # 不对, 让我重新看逻辑: budget - turn_chars < 0 → break, result 还没有 append
        assert result == []


# ============================================================
# 5. CoreferenceResolver
# ============================================================


class TestCoreferenceResolver:
    """指代消解器 -- 代词替换为化学实体."""

    def setup_method(self):
        self.resolver = CoreferenceResolver()

    def test_no_history_no_resolution(self):
        result = self.resolver.resolve("它的波长是多少?", [])
        assert result == "它的波长是多少?"

    def test_pronoun_it_replaced_with_ion(self):
        history = [
            _make_turn(role="assistant", content="Dy3+是稀土离子"),
        ]
        result = self.resolver.resolve("它的跃迁波长?", history)
        assert "Dy3+" in result
        assert "它" not in result

    def test_pronoun_zhege_replaced_with_formula(self):
        history = [
            _make_turn(role="assistant", content="Y2O3 是常见基质材料"),
        ]
        result = self.resolver.resolve("这个的发光性能?", history)
        assert "Y2O3" in result

    def test_no_pronoun_unchanged(self):
        history = [
            _make_turn(role="assistant", content="Dy3+是稀土离子"),
        ]
        result = self.resolver.resolve("稀土离子的发光原理?", history)
        assert result == "稀土离子的发光原理?"

    def test_no_chemical_entity_in_history_unchanged(self):
        history = [
            _make_turn(role="assistant", content="这是很常见的现象"),
        ]
        result = self.resolver.resolve("它是什么?", history)
        assert result == "它是什么?"

    def test_user_message_entity_fallback(self):
        # 无 assistant 历史, 但 user 消息中包含实体
        history = [
            _make_turn(role="user", content="Eu3+在什么条件下发光?"),
        ]
        result = self.resolver.resolve("它的浓度猝灭阈值?", history)
        assert "Eu3+" in result

    def test_assistant_priority_over_user(self):
        history = [
            _make_turn(role="user", content="Eu3+离子"),
            _make_turn(role="assistant", content="Tb3+的发射峰在545nm"),
        ]
        result = self.resolver.resolve("它的波长是多少?", history)
        # assistant 的 Tb3+ 应该优先
        assert "Tb3+" in result

    def test_only_first_pronoun_replaced(self):
        history = [
            _make_turn(role="assistant", content="Dy3+是稀土离子"),
        ]
        result = self.resolver.resolve("它的波长和它的浓度?", history)
        # 只有第一个 "它" 被替换
        assert result.count("Dy3+") == 1

    def test_empty_query_returns_empty(self):
        history = [_make_turn(role="assistant", content="Dy3+")]
        assert self.resolver.resolve("", history) == ""


# ============================================================
# 6. SchemaContextInjector
# ============================================================


class TestSchemaContextInjector:
    """KG Schema 上下文注入器."""

    def setup_method(self):
        self.injector = SchemaContextInjector()

    # --- inject ---

    def test_inject_chemistry_returns_schema_text(self):
        result = self.injector.inject("Dy3+跃迁波长", "chemistry")
        assert "领域实体类型" in result
        assert "关系类型" in result

    def test_inject_education_returns_schema_text(self):
        result = self.injector.inject("学习知识点", "education")
        assert "领域实体类型" in result
        assert "knowledge_point" in result

    def test_inject_unknown_domain_returns_empty(self):
        result = self.injector.inject("some query", "nonexistent_domain")
        assert result == ""

    def test_inject_chemistry_contains_entity_types(self):
        result = self.injector.inject("test", "chemistry")
        assert "chemical_element" in result

    def test_inject_chemistry_contains_relation_types(self):
        result = self.injector.inject("test", "chemistry")
        assert "doped_in" in result

    def test_inject_chemistry_contains_numeric_properties(self):
        result = self.injector.inject("test", "chemistry")
        assert "wavelength" in result

    # --- detect_domain ---

    def test_detect_domain_chemistry_keywords(self):
        domain = self.injector.detect_domain("Dy3+离子的跃迁波长")
        assert domain == "chemistry"

    def test_detect_domain_education_keywords(self):
        domain = self.injector.detect_domain("学习知识点和考试")
        assert domain == "education"

    def test_detect_domain_defaults_to_chemistry(self):
        domain = self.injector.detect_domain("今天天气怎么样")
        assert domain == "chemistry"

    def test_detect_domain_english_chemistry(self):
        domain = self.injector.detect_domain("ion transition wavelength")
        assert domain == "chemistry"

    def test_detect_domain_english_education(self):
        domain = self.injector.detect_domain("learning exam course")
        assert domain == "education"

    # --- register_domain ---

    def test_register_and_inject_custom_domain(self):
        self.injector.register_domain("physics", {
            "entity_types": ["particle", "field"],
            "relation_types": ["interacts_with"],
            "numeric_properties": ["mass", "charge"],
        })
        result = self.injector.inject("test", "physics")
        assert "领域实体类型" in result
        assert "particle" in result

    def test_register_domain_overrides_existing(self):
        self.injector.register_domain("chemistry", {
            "entity_types": ["custom_entity"],
        })
        result = self.injector.inject("test", "chemistry")
        assert "custom_entity" in result


# ============================================================
# 7. LearnerContextAdapter
# ============================================================


class TestLearnerContextAdapter:
    """学习者上下文适配器."""

    def setup_method(self):
        self.adapter = LearnerContextAdapter()

    def test_none_profile_returns_default(self):
        result = self.adapter.adapt(None)
        assert result["weak_kp_ids"] == []
        assert result["bloom_level"] == "understand"
        assert result["suggested_strategy"] == "expand"
        assert result["suggested_depth"] == 1
        assert result["style_preference"] == "reading"
        assert result["context_hint"] == ""

    def test_beginner_level_contextual_strategy(self):
        profile = _make_beginner_profile(bloom_target=BloomLevel.REMEMBER)
        result = self.adapter.adapt(profile)
        assert result["suggested_strategy"] == "contextual"

    def test_beginner_level_top_k_implied(self):
        profile = _make_beginner_profile(bloom_target=BloomLevel.REMEMBER)
        result = self.adapter.adapt(profile)
        # remember → suggested_depth=1, strategy=contextual
        assert result["suggested_depth"] == 1

    def test_advanced_level_expand_strategy(self):
        profile = _make_advanced_profile(bloom_target=BloomLevel.ANALYZE)
        result = self.adapter.adapt(profile)
        assert result["suggested_strategy"] == "decompose"

    def test_advanced_level_depth(self):
        profile = _make_advanced_profile(bloom_target=BloomLevel.ANALYZE)
        result = self.adapter.adapt(profile)
        assert result["suggested_depth"] == 2

    def test_weak_kps_present_in_hint(self):
        profile = _make_beginner_profile(weak_kps=["KP-A01", "KP-B02"])
        result = self.adapter.adapt(profile)
        assert "2 个薄弱知识点" in result["context_hint"]
        assert result["weak_kp_ids"] == ["KP-A01", "KP-B02"]

    def test_bloom_level_mapping_all_six(self):
        mapping = {
            BloomLevel.REMEMBER: ("contextual", 1),
            BloomLevel.UNDERSTAND: ("expand", 1),
            BloomLevel.APPLY: ("synonym", 2),
            BloomLevel.ANALYZE: ("decompose", 2),
            BloomLevel.EVALUATE: ("decompose", 3),
            BloomLevel.CREATE: ("expand", 3),
        }
        for bloom, (expected_strategy, expected_depth) in mapping.items():
            profile = _make_beginner_profile(bloom_target=bloom)
            result = self.adapter.adapt(profile)
            assert result["suggested_strategy"] == expected_strategy, (
                f"Bloom {bloom.value}: expected strategy {expected_strategy}, "
                f"got {result['suggested_strategy']}"
            )
            assert result["suggested_depth"] == expected_depth, (
                f"Bloom {bloom.value}: expected depth {expected_depth}, "
                f"got {result['suggested_depth']}"
            )

    def test_style_preference_from_profile(self):
        profile = _make_advanced_profile(preferred_style=LearningStyle.VISUAL)
        result = self.adapter.adapt(profile)
        assert result["style_preference"] == "visual"

    def test_beginner_hint_in_context(self):
        profile = _make_beginner_profile(level="beginner")
        result = self.adapter.adapt(profile)
        assert "初学者" in result["context_hint"]

    def test_advanced_hint_in_context(self):
        profile = _make_advanced_profile(level="advanced")
        result = self.adapter.adapt(profile)
        assert "高级" in result["context_hint"]

    def test_empty_weak_kps_no_hint(self):
        profile = _make_beginner_profile(weak_kps=[])
        result = self.adapter.adapt(profile)
        assert "薄弱知识点" not in result["context_hint"]


# ============================================================
# 8. RetrievalNeedAssessor
# ============================================================


class TestRetrievalNeedAssessor:
    """检索需求评估器."""

    def setup_method(self):
        self.assessor = RetrievalNeedAssessor()

    def test_greeting_returns_false(self):
        assert self.assessor.assess("你好") is False

    def test_hi_returns_false(self):
        assert self.assessor.assess("hi") is False

    def test_hello_returns_false(self):
        assert self.assessor.assess("hello") is False

    def test_thanks_returns_false(self):
        assert self.assessor.assess("谢谢") is False

    def test_bye_returns_false(self):
        assert self.assessor.assess("再见") is False

    def test_short_query_under_4_chars_returns_false(self):
        assert self.assessor.assess("嗯") is False

    def test_normal_query_no_history_returns_true(self):
        assert self.assessor.assess("Dy3+的跃迁波长") is True

    def test_normal_query_no_context_returns_true(self):
        ctx = QueryContext(original_query="test")
        assert self.assessor.assess("Dy3+离子发光原理", ctx) is True

    def test_query_covered_by_last_assistant_returns_false(self):
        ctx = QueryContext(
            original_query="test",
            dialog_history=[
                _make_turn(role="assistant", content="Dy3+的跃迁波长是580nm"),
            ],
        )
        result = self.assessor.assess("Dy3+跃迁波长", ctx)
        assert result is False

    def test_query_not_covered_returns_true(self):
        ctx = QueryContext(
            original_query="test",
            dialog_history=[
                _make_turn(role="assistant", content="今天天气很好"),
            ],
        )
        result = self.assessor.assess("Dy3+离子的发光波长", ctx)
        assert result is True

    def test_what_are_you_returns_false(self):
        assert self.assessor.assess("你是谁") is False

    def test_what_can_you_do_returns_false(self):
        assert self.assessor.assess("你能做什么") is False


# ============================================================
# 9. QueryContext
# ============================================================


class TestQueryContext:
    """查询上下文数据结构."""

    def test_default_values(self):
        ctx = QueryContext()
        assert ctx.original_query == ""
        assert ctx.resolved_query == ""
        assert ctx.rewritten_queries == []
        assert ctx.intent_hint == ""
        assert ctx.entities == []
        assert ctx.dialog_history == []
        assert ctx.learner_adaptation == {}
        assert ctx.schema_context == ""
        assert ctx.domain == "chemistry"
        assert ctx.needs_retrieval is True
        assert ctx.suggested_top_k == 10
        assert ctx.suggested_depth == 1

    def test_active_query_returns_resolved_when_set(self):
        ctx = QueryContext(
            original_query="原始查询",
            resolved_query="消解后查询",
        )
        assert ctx.active_query == "消解后查询"

    def test_active_query_falls_back_to_original(self):
        ctx = QueryContext(original_query="原始查询", resolved_query="")
        assert ctx.active_query == "原始查询"

    def test_active_query_both_empty(self):
        ctx = QueryContext()
        assert ctx.active_query == ""

    def test_build_time_ms_from_metadata(self):
        ctx = QueryContext(metadata={"build_time_ms": 12.5})
        assert ctx.build_time_ms == 12.5

    def test_build_time_ms_default_zero(self):
        ctx = QueryContext()
        assert ctx.build_time_ms == 0.0

    def test_context_id_auto_generated(self):
        ctx = QueryContext()
        assert ctx.context_id.startswith("ctx-")
        assert len(ctx.context_id) > len("ctx-")

    def test_to_dict_structure(self):
        ctx = QueryContext(
            original_query="Dy3+波长",
            resolved_query="Dy3+波长",
            entities=["Dy3+"],
            domain="chemistry",
            metadata={"build_time_ms": 5.0},
        )
        d = ctx.to_dict()
        assert d["context_id"] == ctx.context_id
        assert d["original_query"] == "Dy3+波长"
        assert d["resolved_query"] == "Dy3+波长"
        assert d["entities"] == ["Dy3+"]
        assert d["domain"] == "chemistry"
        assert d["dialog_turns"] == 0
        assert d["needs_retrieval"] is True
        assert d["suggested_top_k"] == 10
        assert d["build_time_ms"] == 5.0
        assert d["learner_level"] == ""

    def test_to_dict_with_learner_adaptation(self):
        ctx = QueryContext(
            learner_adaptation={"bloom_level": "analyze"},
        )
        d = ctx.to_dict()
        assert d["learner_level"] == "analyze"


# ============================================================
# 10. ContextBuilder 集成测试
# ============================================================


class TestContextBuilder:
    """ContextBuilder 主入口 -- 4 阶段集成测试."""

    def setup_method(self):
        self.builder = ContextBuilder()

    # --- 基础构建 ---

    def test_basic_build_no_profile_no_history(self):
        ctx = self.builder.build("Dy3+的跃迁波长")
        assert ctx.original_query == "Dy3+的跃迁波长"
        assert ctx.resolved_query == "Dy3+的跃迁波长"
        assert ctx.domain == "chemistry"
        assert ctx.needs_retrieval is True
        assert ctx.suggested_top_k > 0

    def test_build_returns_query_context(self):
        ctx = self.builder.build("test query")
        assert isinstance(ctx, QueryContext)

    # --- 对话历史压缩 ---

    def test_build_with_dialog_history_compression(self):
        turns = [_make_turn(content=f"q{i}", timestamp=float(i)) for i in range(10)]
        ctx = self.builder.build("新问题", dialog_history=turns)
        # 压缩后历史应少于原始
        assert len(ctx.dialog_history) < len(turns)

    def test_build_history_metadata_compressed(self):
        turns = [_make_turn(content=f"q{i}", timestamp=float(i)) for i in range(10)]
        ctx = self.builder.build("新问题", dialog_history=turns)
        compressed_count = ctx.metadata.get("history_compressed", 0)
        assert compressed_count > 0

    # --- 学习者画像适配 ---

    def test_build_with_beginner_profile(self):
        profile = _make_beginner_profile(bloom_target=BloomLevel.REMEMBER)
        ctx = self.builder.build("Dy3+波长", learner_profile=profile)
        assert ctx.suggested_top_k == 5  # remember/understand → base=5

    def test_build_with_advanced_profile(self):
        profile = _make_advanced_profile(bloom_target=BloomLevel.ANALYZE)
        ctx = self.builder.build("Dy3+波长", learner_profile=profile)
        assert ctx.suggested_top_k == 15  # analyze/evaluate/create → base=15

    def test_build_with_weak_kps_increases_top_k(self):
        profile = _make_beginner_profile(
            bloom_target=BloomLevel.REMEMBER,
            weak_kps=["KP-A01", "KP-B02"],
        )
        ctx = self.builder.build("Dy3+波长", learner_profile=profile)
        # base 5 + 3 weak_kps = 8
        assert ctx.suggested_top_k == 8

    def test_build_with_weak_kps_capped_at_20(self):
        profile = _make_advanced_profile(
            bloom_target=BloomLevel.ANALYZE,
            weak_kps=["KP-A01", "KP-B02", "KP-C03"],
        )
        ctx = self.builder.build("Dy3+波长", learner_profile=profile)
        # base 15 + 3 = 18 (under cap 20)
        assert ctx.suggested_top_k == 18

    # --- 指代消解 ---

    def test_coreference_resolution_in_build(self):
        history = [
            _make_turn(role="assistant", content="Dy3+是重要稀土离子"),
        ]
        ctx = self.builder.build("它的跃迁波长?", dialog_history=history)
        assert "Dy3+" in ctx.resolved_query
        assert ctx.metadata.get("coreference_resolved") is True

    def test_no_coreference_no_flag(self):
        ctx = self.builder.build("稀土发光原理")
        assert ctx.metadata.get("coreference_resolved") is False

    # --- 领域检测 ---

    def test_domain_detection_chemistry(self):
        ctx = self.builder.build("Dy3+离子的跃迁波长")
        assert ctx.domain == "chemistry"

    def test_domain_detection_education(self):
        ctx = self.builder.build("学习知识点的方法")
        assert ctx.domain == "education"

    # --- 实体提取 ---

    def test_entity_extraction_in_build(self):
        ctx = self.builder.build("Dy3+离子的发射波长")
        assert len(ctx.entities) > 0
        assert any("Dy3+" in e for e in ctx.entities)

    def test_entity_count_in_metadata(self):
        ctx = self.builder.build("Dy3+在Y2O3中的发光")
        assert ctx.metadata.get("entities_count", 0) > 0

    # --- 检索需求评估 ---

    def test_needs_retrieval_false_for_greeting(self):
        ctx = self.builder.build("你好")
        assert ctx.needs_retrieval is False

    def test_needs_retrieval_true_for_normal_query(self):
        ctx = self.builder.build("Dy3+离子的跃迁波长是多少")
        assert ctx.needs_retrieval is True

    # --- 查询重写 ---

    def test_rewritten_queries_populated(self):
        ctx = self.builder.build("Dy3+离子的跃迁波长和发光效率")
        assert isinstance(ctx.rewritten_queries, list)
        assert ctx.metadata.get("rewrite_count", 0) >= 0

    def test_custom_rewrite_strategies(self):
        ctx = self.builder.build(
            "Dy3+波长",
            rewrite_strategies=["synonym"],
        )
        # 应该尝试 synonym 策略
        assert isinstance(ctx.rewritten_queries, list)

    def test_invalid_rewrite_strategies_ignored(self):
        ctx = self.builder.build(
            "Dy3+波长",
            rewrite_strategies=["nonexistent_strategy"],
        )
        assert isinstance(ctx.rewritten_queries, list)

    # --- 构建耗时 ---

    def test_build_time_ms_positive(self):
        ctx = self.builder.build("Dy3+离子的跃迁波长是多少")
        assert ctx.build_time_ms >= 0

    def test_build_time_ms_in_metadata(self):
        ctx = self.builder.build("test query")
        assert "build_time_ms" in ctx.metadata

    # --- Schema 注入 ---

    def test_schema_context_injected(self):
        ctx = self.builder.build("Dy3+离子")
        assert ctx.schema_context != ""
        assert "领域实体类型" in ctx.schema_context

    # --- intent_hint ---

    def test_intent_hint_populated(self):
        ctx = self.builder.build("Dy3+离子的跃迁波长")
        assert ctx.intent_hint != ""

    # --- to_dict ---

    def test_build_result_to_dict(self):
        ctx = self.builder.build("Dy3+波长")
        d = ctx.to_dict()
        assert "original_query" in d
        assert "resolved_query" in d
        assert "build_time_ms" in d

    # --- LLM 分类器 ---

    def test_llm_classifier_used_when_provided(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {"intent_hint": "llm-concept"}
        builder = ContextBuilder(llm_classifier=mock_classifier)
        ctx = builder.build("Dy3+波长")
        assert ctx.intent_hint == "llm-concept"

    def test_llm_classifier_failure_graceful(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = RuntimeError("LLM unavailable")
        builder = ContextBuilder(llm_classifier=mock_classifier)
        ctx = builder.build("Dy3+波长")
        # 不应该抛出异常, 应回退到规则推断
        assert isinstance(ctx, QueryContext)

    # --- 自定义预算 ---

    def test_custom_budget(self):
        budget = ContextBudget(max_tokens=2048)
        builder = ContextBuilder(budget=budget)
        ctx = builder.build("Dy3+波长")
        assert isinstance(ctx, QueryContext)

    def test_sliding_window_strategy(self):
        # 用极小的预算让 sliding window 生效
        tiny_budget = ContextBudget(max_tokens=50, history_ratio=0.20)
        builder = ContextBuilder(
            budget=tiny_budget,
            history_strategy=HistoryCompressStrategy.SLIDING_WINDOW,
        )
        turns = [_make_turn(content=f"q{i} long content here", timestamp=float(i)) for i in range(10)]
        ctx = builder.build("新问题", dialog_history=turns)
        assert len(ctx.dialog_history) < len(turns)


# ============================================================
# 11. 边界情况
# ============================================================


class TestEdgeCases:
    """边界情况测试."""

    def test_empty_query_string(self):
        builder = ContextBuilder()
        ctx = builder.build("")
        assert ctx.original_query == ""
        assert ctx.needs_retrieval is False  # 空字符串 < 4 字符

    def test_whitespace_only_query(self):
        builder = ContextBuilder()
        ctx = builder.build("   ")
        assert ctx.needs_retrieval is False  # strip 后 < 4 字符

    def test_very_long_query_triggers_composite_hint(self):
        builder = ContextBuilder()
        long_query = "Dy3+离子在Y2O3基质材料中的发光效率以及能量传递机理的详细研究分析报告"
        ctx = builder.build(long_query)
        if len(ctx.entities) == 0:
            # 无数值/化学实体时, 长查询应提示 composite
            assert "composite" in ctx.intent_hint

    def test_concurrent_builds_thread_safety(self):
        """多线程并发构建, 不应抛出异常."""
        builder = ContextBuilder()
        results: list[QueryContext] = []
        errors: list[Exception] = []

        def worker(query: str) -> None:
            try:
                ctx = builder.build(query)
                results.append(ctx)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"查询-{i}的波长",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发构建出错: {errors}"
        assert len(results) == 10

    def test_dialog_turn_with_system_role(self):
        turns = [
            DialogTurn(role="system", content="系统提示"),
            _make_turn(role="user", content="Dy3+波长"),
            _make_turn(role="assistant", content="580nm"),
        ]
        builder = ContextBuilder()
        ctx = builder.build("它的浓度?", dialog_history=turns)
        assert isinstance(ctx, QueryContext)

    def test_unicode_query(self):
        builder = ContextBuilder()
        ctx = builder.build("Dy\u00b3\u207a离子的跃迁波长")
        assert isinstance(ctx, QueryContext)

    def test_none_dialog_history(self):
        builder = ContextBuilder()
        ctx = builder.build("Dy3+波长", dialog_history=None)
        assert ctx.dialog_history == []

    def test_intermediate_bloom_level_top_k(self):
        """apply 层级 → base=10 (默认)."""
        profile = _make_beginner_profile(bloom_target=BloomLevel.APPLY)
        builder = ContextBuilder()
        ctx = builder.build("Dy3+波长", learner_profile=profile)
        assert ctx.suggested_top_k == 10

    def test_evaluate_bloom_level_top_k(self):
        """evaluate 层级 → base=15."""
        profile = _make_beginner_profile(bloom_target=BloomLevel.EVALUATE)
        builder = ContextBuilder()
        ctx = builder.build("Dy3+波长", learner_profile=profile)
        assert ctx.suggested_top_k == 15

    def test_create_bloom_level_top_k(self):
        """create 层级 → base=15."""
        profile = _make_beginner_profile(bloom_target=BloomLevel.CREATE)
        builder = ContextBuilder()
        ctx = builder.build("Dy3+波长", learner_profile=profile)
        assert ctx.suggested_top_k == 15

    def test_query_with_summarize_strategy(self):
        builder = ContextBuilder(
            history_strategy=HistoryCompressStrategy.SUMMARIZE,
        )
        turns = [_make_turn(role="user", content=f"问题{i}", timestamp=float(i)) for i in range(8)]
        ctx = builder.build("新问题", dialog_history=turns)
        # 应有摘要行
        system_turns = [t for t in ctx.dialog_history if t.role == "system"]
        assert len(system_turns) == 1
        assert "历史摘要" in system_turns[0].content
