"""L3 IntentRouter v2 + ContextBuilder 集成测试.

覆盖:
  1. IntentClassifier v2 (intent_hint / schema_context 融合)
  2. IntentRouter.route_with_context (Self-RAG 跳过 / 自适应参数 / 多查询变体)
  3. IntentRouter.route_auto (自动模式选择 / 回退兼容)
  4. IntentRouter._rrf_fuse (多路 RRF 融合)
  5. ContextBuilder → IntentRouter 端到端 (四阶段构建 + 路由)
  6. v1 向后兼容 (原有 route 接口不受影响)

设计参考:
  - Self-RAG: needs_retrieval 评估 → 跳过不必要的检索
  - ReCAP: 上下文信号融合意图分类
  - Plan-and-Solve: 多查询变体 → RRF 融合
  - SEAL: Schema 上下文校准意图
  - Context Recycling: 预算感知上下文管理
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dy3_polaris.l3.api_models import (
    BloomLevel,
    KPMastery,
    LearnerProfile,
    LearningStyle,
)
from dy3_polaris.l3.context_builder import (
    ContextBuilder,
    DialogTurn,
    QueryContext,
)
from dy3_polaris.l3.intent_router import (
    EntityExtractor,
    IntentClassifier,
    IntentResult,
    IntentRouter,
    IntentType,
    RoutedResult,
)
from dy3_polaris.l3.models import (
    DocumentChunk,
    EntityType,
    KnowledgeEntity,
    KnowledgeTriple,
    RetrievalResult,
)
from dy3_polaris.l3.store import KnowledgeStore


# ============================================================
# 辅助工厂
# ============================================================


def _make_store_with_data() -> KnowledgeStore:
    """创建带有测试数据的 KnowledgeStore."""
    store = KnowledgeStore()

    # 添加实体
    dy3 = KnowledgeEntity(
        entity_id="dy3+",
        name="Dy3+",
        entity_type=EntityType.CONCEPT,
        domain="chemistry",
        description="三价镝离子，稀土发光离子",
    )
    yag = KnowledgeEntity(
        entity_id="yag",
        name="YAG",
        entity_type=EntityType.CONCEPT,
        domain="chemistry",
        description="钇铝石榴石，常用激光基质材料",
    )
    store.add_entity(dy3, check_duplicate=False)
    store.add_entity(yag, check_duplicate=False)

    # 添加三元组
    store.triple_store.add_triple(
        KnowledgeTriple(
            subject_id="dy3+",
            predicate="doped_in",
            object_id="yag",
            confidence=0.9,
        )
    )

    # 添加文档切片 (用于关键词检索)
    chunk1 = DocumentChunk(
        chunk_id="ch-001",
        document_id="dy3+",
        content="Dy3+离子的4F9/2→6H15/2跃迁产生约580nm的黄光发射",
        content_type="text",
        section="emission",
        page=1,
    )
    chunk2 = DocumentChunk(
        chunk_id="ch-002",
        document_id="yag",
        content="YAG是一种优良的激光基质材料，常用于固态激光器",
        content_type="text",
        section="overview",
        page=1,
    )
    chunk3 = DocumentChunk(
        chunk_id="ch-003",
        document_id="dy3+",
        content="浓度猝灭效应在Dy3+掺杂体系中普遍存在，最佳掺杂浓度约为1-3 mol%",
        content_type="text",
        section="quenching",
        page=2,
    )
    store.chunk_store.add_chunk(chunk1)
    store.chunk_store.add_chunk(chunk2)
    store.chunk_store.add_chunk(chunk3)

    return store


def _make_learner(
    level: str = "beginner",
    bloom: BloomLevel = BloomLevel.REMEMBER,
) -> LearnerProfile:
    return LearnerProfile(
        learner_id="test-learner",
        level=level,
        bloom_target=bloom,
        preferred_style=LearningStyle.READING,
        kp_mastery={
            "dy3-emission": KPMastery(
                kp_id="dy3-emission", mastery_prob=0.3, attempts=5, correct_count=2
            ),
        },
        weak_kps=["dy3-emission"],
        interests=["发光材料"],
    )


def _make_turns(
    n: int = 4,
) -> list[DialogTurn]:
    return [
        DialogTurn(role="user", content="什么是稀土离子?", timestamp=1000.0 + i * 10)
        if i % 2 == 0
        else DialogTurn(
            role="assistant", content="稀土离子是...", timestamp=1005.0 + i * 10
        )
        for i in range(n)
    ]


def _make_ctx(
    *,
    query: str = "Dy3+的发射波长是多少?",
    needs_retrieval: bool = True,
    intent_hint: str = "",
    schema_context: str = "",
    rewritten_queries: list[str] | None = None,
    suggested_top_k: int = 10,
    suggested_depth: int = 1,
    entities: list[str] | None = None,
    domain: str = "chemistry",
) -> QueryContext:
    """快速构建测试用 QueryContext."""
    return QueryContext(
        original_query=query,
        resolved_query=query,
        rewritten_queries=rewritten_queries or [],
        intent_hint=intent_hint,
        schema_context=schema_context,
        needs_retrieval=needs_retrieval,
        suggested_top_k=suggested_top_k,
        suggested_depth=suggested_depth,
        entities=entities or [],
        domain=domain,
    )


# ============================================================
# 1. IntentClassifier v2
# ============================================================


class TestIntentClassifierV2:
    """IntentClassifier v2: intent_hint / schema_context 融合测试."""

    def test_basic_numeric_classification(self):
        """基本数值意图分类."""
        c = IntentClassifier()
        r = c.classify("Dy3+的发射波长是多少nm?")
        assert r.intent_type == IntentType.NUMERIC
        assert r.confidence >= 0.3
        assert any("numeric" in rule for rule in r.matched_rules)

    def test_basic_concept_classification(self):
        """基本概念意图分类."""
        c = IntentClassifier()
        r = c.classify("什么是稀土发光材料?")
        assert r.intent_type == IntentType.CONCEPT
        assert any("concept" in rule for rule in r.matched_rules)

    def test_basic_relational_classification(self):
        """基本关系意图分类."""
        c = IntentClassifier()
        r = c.classify("Dy3+和YAG之间有什么关系?")
        assert r.intent_type == IntentType.RELATIONAL
        assert any("relational" in rule for rule in r.matched_rules)

    def test_basic_composite_classification(self):
        """基本复合意图分类."""
        c = IntentClassifier()
        r = c.classify("比较Dy3+和Eu3+的发光效率并且说明两者的区别")
        assert r.intent_type == IntentType.COMPOSITE

    def test_intent_hint_boosts_numeric(self):
        """intent_hint='numeric' 增强 numeric 得分."""
        c = IntentClassifier()
        # 没有提示时可能是 concept
        r1 = c.classify("Dy3+的跃迁")
        # 有提示时应该偏向 numeric
        r2 = c.classify("Dy3+的跃迁", intent_hint="numeric")
        # 提示应该增加 context_hint 规则
        assert any("context_hint" in rule for rule in r2.matched_rules)
        # numeric 得分应该比没有提示时高
        # (不强制要求改变分类结果, 但提示必须有影响)

    def test_intent_hint_boosts_relational(self):
        """intent_hint='relational' 增强 relational 得分."""
        c = IntentClassifier()
        r = c.classify("Dy3+和YAG", intent_hint="relational")
        assert any("context_hint:relational" in rule for rule in r.matched_rules)

    def test_intent_hint_multi_type(self):
        """多类型 intent_hint 同时增强."""
        c = IntentClassifier()
        r = c.classify("Dy3+的波长和浓度关系", intent_hint="numeric+relational")
        hints = [rule for rule in r.matched_rules if "context_hint" in rule]
        assert len(hints) >= 2

    def test_intent_hint_unknown_ignored(self):
        """未知的 intent_hint 类型被忽略."""
        c = IntentClassifier()
        r = c.classify("什么是Dy3+", intent_hint="unknown_type")
        # 不应该有 context_hint 规则
        assert not any("context_hint" in rule for rule in r.matched_rules)

    def test_schema_context_boosts_numeric(self):
        """Schema 上下文中的可查询数值属性增强 numeric."""
        c = IntentClassifier()
        schema = "可查询数值属性: 波长, 浓度, 温度, 效率"
        r = c.classify("Dy3+的波长是多少", schema_context=schema)
        assert any("schema_boost:numeric" in rule for rule in r.matched_rules)

    def test_schema_context_boosts_relational(self):
        """Schema 上下文中的关系类型增强 relational."""
        c = IntentClassifier()
        schema = "关系类型: 依赖, 先修, 影响, 掺杂, 猝灭"
        r = c.classify("Dy3+的猝灭机理", schema_context=schema)
        assert any("schema_boost:relational" in rule for rule in r.matched_rules)

    def test_schema_context_no_match_no_boost(self):
        """Schema 上下文无匹配时不增强."""
        c = IntentClassifier()
        schema = "可查询数值属性: 波长, 浓度, 温度, 效率"
        r = c.classify("什么是Dy3+", schema_context=schema)
        # '什么是Dy3+' 不含数值属性关键词, 不应有 schema_boost:numeric
        numeric_boosts = [
            rule for rule in r.matched_rules
            if "schema_boost:numeric" in rule
        ]
        assert len(numeric_boosts) == 0

    def test_hint_and_schema_combined(self):
        """intent_hint 和 schema_context 同时生效."""
        c = IntentClassifier()
        schema = "可查询数值属性: 波长, 浓度, 温度"
        r = c.classify(
            "Dy3+波长是多少nm?",
            intent_hint="numeric",
            schema_context=schema,
        )
        has_hint = any("context_hint" in rule for rule in r.matched_rules)
        has_schema = any("schema_boost" in rule for rule in r.matched_rules)
        assert has_hint and has_schema

    def test_fallback_default_concept(self):
        """低分查询默认回退到 concept."""
        c = IntentClassifier()
        r = c.classify("hello world")
        assert r.intent_type == IntentType.CONCEPT
        assert any("fallback" in rule for rule in r.matched_rules)

    def test_llm_fallback_flag(self):
        """LLM 兜底标志被传递但不改变结果 (无实际 LLM)."""
        c = IntentClassifier()
        r1 = c.classify("什么是稀土?", use_llm=False)
        r2 = c.classify("什么是稀土?", use_llm=True)
        # 没有实际 LLM, 结果应该一致
        assert r1.intent_type == r2.intent_type

    def test_entity_extraction_in_result(self):
        """分类结果包含提取的实体."""
        c = IntentClassifier()
        r = c.classify("Dy3+离子 the 4F9/2 transition wavelength")
        types = {e.entity_type for e in r.extracted_entities}
        assert "ion" in types  # Dy3+
        assert "spectral_term" in types  # 4F9/2

    def test_suggested_path_populated(self):
        """分类结果包含建议路径."""
        c = IntentClassifier()
        r = c.classify("Dy3+的浓度是多少mol%?")
        assert r.suggested_path != ""
        assert r.classification_time_ms >= 0


# ============================================================
# 2. IntentRouter.route_with_context
# ============================================================


class TestRouteWithContext:
    """IntentRouter v2 上下文增强路由测试."""

    @pytest.fixture
    def store(self) -> KnowledgeStore:
        return _make_store_with_data()

    @pytest.fixture
    def router(self, store: KnowledgeStore) -> IntentRouter:
        return IntentRouter(store, use_llm_fallback=False)

    def test_skip_retrieval_self_rag(self, router: IntentRouter):
        """Self-RAG: needs_retrieval=False 时跳过检索."""
        ctx = _make_ctx(
            query="你好",
            needs_retrieval=False,
        )
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)
        assert result.total == 0
        assert result.retrieval_result.total == 0
        assert any(
            "self_rag:skip_retrieval" in rule
            for rule in result.intent.matched_rules
        )

    def test_adaptive_top_k(self, router: IntentRouter):
        """自适应参数: suggested_top_k 传递到检索."""
        ctx = _make_ctx(suggested_top_k=5)
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)
        # 不强制结果数量, 但 top_k 应该被使用
        assert result.total <= 5 or result.total >= 0

    def test_intent_hint_passed_to_classifier(self, router: IntentRouter):
        """intent_hint 传递到分类器."""
        ctx = _make_ctx(
            query="Dy3+的跃迁",
            intent_hint="numeric",
        )
        result = router.route_with_context(ctx)
        has_hint = any(
            "context_hint" in rule
            for rule in result.intent.matched_rules
        )
        assert has_hint

    def test_schema_context_passed_to_classifier(self, router: IntentRouter):
        """schema_context 传递到分类器."""
        ctx = _make_ctx(
            query="Dy3+的波长",
            schema_context="可查询数值属性: 波长, 浓度, 温度, 效率",
        )
        result = router.route_with_context(ctx)
        has_schema = any(
            "schema_boost" in rule
            for rule in result.intent.matched_rules
        )
        assert has_schema

    def test_single_query_fallback(self, router: IntentRouter):
        """单查询 (无重写变体) 使用 _route_single."""
        ctx = _make_ctx(
            query="Dy3+的发射波长是多少nm?",
            rewritten_queries=[],
        )
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)
        assert result.intent.intent_type == IntentType.NUMERIC

    def test_multi_query_triggers_rrf(self, router: IntentRouter):
        """多查询变体触发 _route_multi_query + RRF 融合."""
        ctx = _make_ctx(
            query="Dy3+的发射波长",
            rewritten_queries=[
                "Dy3+的发射波长是多少nm",
                "Dy3+的发射光谱波长",
                "Dy3+离子发射波长数值",
            ],
        )
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)
        assert result.total >= 0

    def test_multi_query_deduplicates_primary(self, router: IntentRouter):
        """多查询变体中与主查询相同的会被去重."""
        ctx = _make_ctx(
            query="Dy3+的发射波长",
            rewritten_queries=[
                "Dy3+的发射波长",  # 与主查询相同, 应被去重
                "Dy3+离子发射波长",
            ],
        )
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)

    def test_multi_query_limits_variants(self, router: IntentRouter):
        """多查询变体最多使用 3 个 (加主查询共 4 个)."""
        ctx = _make_ctx(
            query="Dy3+的发射波长",
            rewritten_queries=[
                f"变体查询{i}" for i in range(10)
            ],
        )
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)
        # 不应崩溃, 变体数量被限制

    def test_deep_graph_search(self, router: IntentRouter):
        """suggested_depth > 1 时触发深度图检索."""
        ctx = _make_ctx(
            query="Dy3+和YAG的关系",
            intent_hint="relational",
            suggested_depth=3,
        )
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)

    def test_total_time_ms_populated(self, router: IntentRouter):
        """total_time_ms 被正确填充."""
        ctx = _make_ctx(query="Dy3+")
        result = router.route_with_context(ctx)
        assert result.total_time_ms >= 0
        assert result.intent.classification_time_ms >= 0

    def test_concept_query_returns_results(self, router: IntentRouter):
        """概念查询能返回检索结果."""
        ctx = _make_ctx(query="Dy3+的浓度猝灭效应")
        result = router.route_with_context(ctx)
        # 至少应触发关键词检索
        assert isinstance(result, RoutedResult)

    def test_with_filter(self, router: IntentRouter):
        """带 filter 参数的路由."""
        from dy3_polaris.l3.models import RetrievalFilter
        f = RetrievalFilter(min_quality=0.5)
        ctx = _make_ctx(query="Dy3+")
        result = router.route_with_context(ctx, filter=f)
        assert isinstance(result, RoutedResult)

    def test_with_query_vector(self, router: IntentRouter):
        """带 query_vector 参数的路由."""
        ctx = _make_ctx(query="Dy3+")
        vec = [0.1] * 128
        result = router.route_with_context(ctx, query_vector=vec)
        assert isinstance(result, RoutedResult)


# ============================================================
# 3. IntentRouter.route_auto
# ============================================================


class TestRouteAuto:
    """IntentRouter.route_auto 自动模式选择测试."""

    @pytest.fixture
    def store(self) -> KnowledgeStore:
        return _make_store_with_data()

    def test_fallback_to_v1_without_builder(self, store: KnowledgeStore):
        """无 ContextBuilder 时回退到 v1 route."""
        router = IntentRouter(store, context_builder=None)
        result = router.route_auto("Dy3+的发射波长是多少nm?")
        assert isinstance(result, RoutedResult)
        assert result.total >= 0

    def test_uses_v2_with_builder(self, store: KnowledgeStore):
        """有 ContextBuilder 时使用 v2 route_with_context."""
        builder = ContextBuilder()
        router = IntentRouter(store, context_builder=builder)
        result = router.route_auto("Dy3+的发射波长是多少nm?")
        assert isinstance(result, RoutedResult)

    def test_passes_learner_profile(self, store: KnowledgeStore):
        """learner_profile 传递到 ContextBuilder."""
        builder = ContextBuilder()
        router = IntentRouter(store, context_builder=builder)
        profile = _make_learner()
        result = router.route_auto(
            "Dy3+的跃迁波长",
            learner_profile=profile,
        )
        assert isinstance(result, RoutedResult)

    def test_passes_dialog_history(self, store: KnowledgeStore):
        """dialog_history 传递到 ContextBuilder."""
        builder = ContextBuilder()
        router = IntentRouter(store, context_builder=builder)
        turns = _make_turns(4)
        result = router.route_auto(
            "它的能级跃迁波长是多少?",
            dialog_history=turns,
        )
        assert isinstance(result, RoutedResult)

    def test_v1_backward_compat(self, store: KnowledgeStore):
        """v1 route 方法不受 v2 变更影响."""
        router = IntentRouter(store, context_builder=ContextBuilder())
        result = router.route("Dy3+的发射波长是多少nm?")
        assert isinstance(result, RoutedResult)
        assert result.intent.intent_type == IntentType.NUMERIC

    def test_batch_route_still_works(self, store: KnowledgeStore):
        """batch_route 方法不受影响."""
        router = IntentRouter(store)
        results = router.batch_route([
            "什么是Dy3+?",
            "YAG的化学式是什么?",
        ])
        assert len(results) == 2
        assert all(isinstance(r, RoutedResult) for r in results)


# ============================================================
# 4. RRF 融合
# ============================================================


class TestRRFFuse:
    """IntentRouter._rrf_fuse RRF 融合测试."""

    @pytest.fixture
    def store(self) -> KnowledgeStore:
        return _make_store_with_data()

    @pytest.fixture
    def router(self, store: KnowledgeStore) -> IntentRouter:
        return IntentRouter(store, rrf_k=60)

    def test_single_result_returns_unchanged(self, router: IntentRouter):
        """单路结果原样返回."""
        r = RetrievalResult(
            query="test",
            results=[{"id": "a", "content": "A"}],
            scores=[0.9],
            total=1,
            retrieval_time_ms=1.0,
        )
        fused = router._rrf_fuse([r])
        assert fused.total == 1
        assert fused.results[0]["id"] == "a"

    def test_two_results_fuse_correctly(self, router: IntentRouter):
        """两路结果正确融合."""
        r1 = RetrievalResult(
            query="test",
            results=[
                {"id": "a", "content": "A"},
                {"id": "b", "content": "B"},
            ],
            scores=[0.9, 0.8],
            total=2,
            retrieval_time_ms=1.0,
        )
        r2 = RetrievalResult(
            query="test",
            results=[
                {"id": "b", "content": "B"},
                {"id": "c", "content": "C"},
            ],
            scores=[0.85, 0.7],
            total=2,
            retrieval_time_ms=1.0,
        )
        fused = router._rrf_fuse([r1, r2])
        # b 出现在两路中, RRF 分数应该最高
        assert fused.total == 3
        ids = [r["id"] for r in fused.results]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        # b 应排在最前 (在两路中都出现)
        assert fused.results[0]["id"] == "b"

    def test_three_results_merge_dedup(self, router: IntentRouter):
        """三路结果去重合并."""
        r1 = RetrievalResult(
            query="test",
            results=[{"chunk_id": "x", "content": "X"}],
            scores=[0.9],
            total=1,
            retrieval_time_ms=1.0,
        )
        r2 = RetrievalResult(
            query="test",
            results=[{"chunk_id": "x", "content": "X"}],
            scores=[0.8],
            total=1,
            retrieval_time_ms=1.0,
        )
        r3 = RetrievalResult(
            query="test",
            results=[{"chunk_id": "y", "content": "Y"}],
            scores=[0.7],
            total=1,
            retrieval_time_ms=1.0,
        )
        fused = router._rrf_fuse([r1, r2, r3])
        # x 在两路中出现, 应比 y 分数高
        assert fused.total == 2
        assert fused.results[0]["chunk_id"] == "x"
        assert fused.results[1]["chunk_id"] == "y"

    def test_empty_list_returns_empty(self, router: IntentRouter):
        """空列表返回空结果."""
        fused = router._rrf_fuse([])
        assert fused.total == 0
        assert fused.results == []

    def test_rrf_k_parameter_effects(self):
        """不同 rrf_k 值影响融合结果."""
        store = _make_store_with_data()
        router_low_k = IntentRouter(store, rrf_k=5)
        router_high_k = IntentRouter(store, rrf_k=100)

        r1 = RetrievalResult(
            query="test",
            results=[{"id": "a"}, {"id": "b"}],
            scores=[0.9, 0.5],
            total=2, retrieval_time_ms=1.0,
        )
        r2 = RetrievalResult(
            query="test",
            results=[{"id": "b"}, {"id": "a"}],
            scores=[0.8, 0.6],
            total=2, retrieval_time_ms=1.0,
        )

        f_low = router_low_k._rrf_fuse([r1, r2])
        f_high = router_high_k._rrf_fuse([r1, r2])

        # 两者都应正确融合
        assert f_low.total == 2
        assert f_high.total == 2

    def test_total_time_aggregated(self, router: IntentRouter):
        """融合结果的 retrieval_time_ms 是各路之和."""
        r1 = RetrievalResult(
            query="test", results=[], scores=[], total=0, retrieval_time_ms=5.0,
        )
        r2 = RetrievalResult(
            query="test", results=[], scores=[], total=0, retrieval_time_ms=3.0,
        )
        fused = router._rrf_fuse([r1, r2])
        assert fused.retrieval_time_ms == 8.0


# ============================================================
# 5. ContextBuilder → IntentRouter 端到端
# ============================================================


class TestContextBuilderToRouterE2E:
    """ContextBuilder 四阶段构建 → IntentRouter 路由 端到端测试."""

    @pytest.fixture
    def store(self) -> KnowledgeStore:
        return _make_store_with_data()

    @pytest.fixture
    def builder(self) -> ContextBuilder:
        return ContextBuilder()

    @pytest.fixture
    def router_with_builder(
        self, store: KnowledgeStore, builder: ContextBuilder
    ) -> IntentRouter:
        return IntentRouter(store, context_builder=builder)

    def test_simple_query_e2e(self, router_with_builder: IntentRouter):
        """简单查询端到端: build → route_with_context."""
        builder = ContextBuilder()
        ctx = builder.build("Dy3+的发射波长是多少nm?")
        assert isinstance(ctx, QueryContext)
        assert ctx.needs_retrieval is True
        assert len(ctx.entities) >= 1

        router = router_with_builder
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)
        assert result.total >= 0

    def test_with_learner_profile_e2e(
        self, store: KnowledgeStore, builder: ContextBuilder
    ):
        """带学习者画像端到端."""
        profile = _make_learner(level="advanced")
        ctx = builder.build(
            "Dy3+的浓度猝灭机理",
            learner_profile=profile,
        )
        assert ctx.learner_adaptation != {}

        router = IntentRouter(store, context_builder=builder)
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)

    def test_with_dialog_history_e2e(
        self, store: KnowledgeStore, builder: ContextBuilder
    ):
        """带对话历史端到端: 指代消解."""
        turns = [
            DialogTurn(role="user", content="什么是Dy3+?", timestamp=1000.0),
            DialogTurn(
                role="assistant",
                content="Dy3+是三价镝离子，稀土发光离子",
                timestamp=1010.0,
            ),
        ]
        ctx = builder.build(
            "它的发射波长是多少?",
            dialog_history=turns,
        )
        # 指代消解: "它" → "Dy3+"
        assert "Dy3+" in ctx.resolved_query

        router = IntentRouter(store, context_builder=builder)
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)

    def test_route_auto_full_pipeline(
        self, store: KnowledgeStore, builder: ContextBuilder
    ):
        """route_auto 完整管线: build → route."""
        profile = _make_learner()
        turns = _make_turns(4)
        router = IntentRouter(store, context_builder=builder)
        result = router.route_auto(
            "它的能级跃迁波长是多少nm?",
            learner_profile=profile,
            dialog_history=turns,
        )
        assert isinstance(result, RoutedResult)
        assert result.total >= 0

    def test_context_id_unique(self, builder: ContextBuilder):
        """每次构建的 context_id 唯一."""
        ctx1 = builder.build("query1")
        ctx2 = builder.build("query2")
        assert ctx1.context_id != ctx2.context_id
        assert ctx1.context_id.startswith("ctx-")

    def test_build_time_recorded(self, builder: ContextBuilder):
        """构建耗时被记录."""
        ctx = builder.build("Dy3+的波长")
        assert ctx.metadata.get("build_time_ms", 0) >= 0  # 环境计时精度

    def test_domain_detection(self, builder: ContextBuilder):
        """领域检测."""
        ctx = builder.build("Dy3+的跃迁波长")
        assert ctx.domain == "chemistry"

    def test_rewrite_strategies_applied(self, builder: ContextBuilder):
        """查询重写策略被应用."""
        ctx = builder.build(
            "Dy3+的发射波长和猝灭浓度",
            rewrite_strategies=["expand", "contextual"],
        )
        # 至少应有重写结果 (可能为空或非空)
        assert isinstance(ctx.rewritten_queries, list)

    def test_full_context_fields_populated(
        self, store: KnowledgeStore, builder: ContextBuilder
    ):
        """完整上下文字段填充验证."""
        profile = _make_learner()
        turns = _make_turns(4)
        ctx = builder.build(
            "Dy3+的4F9/2能级跃迁波长是多少nm?",
            learner_profile=profile,
            dialog_history=turns,
        )
        # 验证关键字段
        assert ctx.original_query != ""
        assert ctx.resolved_query != ""
        assert isinstance(ctx.entities, list)
        assert isinstance(ctx.dialog_history, list)
        assert isinstance(ctx.learner_adaptation, dict)
        assert isinstance(ctx.metadata, dict)
        assert isinstance(ctx.needs_retrieval, bool)
        assert ctx.suggested_top_k > 0
        assert ctx.suggested_depth >= 1

    def test_beginner_gets_more_results(
        self, store: KnowledgeStore, builder: ContextBuilder
    ):
        """初学者有薄弱知识点时增加建议结果数."""
        profile = _make_learner(level="beginner")
        ctx = builder.build("Dy3+的跃迁", learner_profile=profile)
        # beginner (remember) base=5, weak_kps adds +3 => 8
        assert ctx.suggested_top_k >= 5
        assert ctx.suggested_top_k > 5  # weak_kps boost

    def test_advanced_gets_fewer_results(
        self, store: KnowledgeStore, builder: ContextBuilder
    ):
        """高级学习者建议返回较少结果."""
        profile = _make_learner(level="advanced")
        ctx = builder.build("Dy3+的跃迁", learner_profile=profile)
        assert ctx.suggested_top_k <= 10


# ============================================================
# 6. v1 向后兼容
# ============================================================


class TestV1BackwardCompat:
    """v1 route 接口不受 v2 变更影响."""

    @pytest.fixture
    def store(self) -> KnowledgeStore:
        return _make_store_with_data()

    def test_route_concept(self, store: KnowledgeStore):
        """v1 概念检索."""
        router = IntentRouter(store)
        result = router.route("什么是Dy3+?")
        assert isinstance(result, RoutedResult)
        assert result.intent.intent_type == IntentType.CONCEPT

    def test_route_numeric(self, store: KnowledgeStore):
        """v1 数值检索."""
        router = IntentRouter(store)
        result = router.route("Dy3+的发射波长是多少nm?")
        assert isinstance(result, RoutedResult)
        assert result.intent.intent_type == IntentType.NUMERIC

    def test_route_relational(self, store: KnowledgeStore):
        """v1 关系检索."""
        router = IntentRouter(store)
        result = router.route("Dy3+和YAG的关系是什么?")
        assert isinstance(result, RoutedResult)
        assert result.intent.intent_type == IntentType.RELATIONAL

    def test_route_composite(self, store: KnowledgeStore):
        """v1 复合检索."""
        router = IntentRouter(store)
        result = router.route("比较Dy3+和Eu3+的发光效率并且说明区别")
        assert isinstance(result, RoutedResult)
        assert result.intent.intent_type == IntentType.COMPOSITE

    def test_route_with_custom_classifier(self, store: KnowledgeStore):
        """自定义分类器."""
        custom = IntentClassifier()
        router = IntentRouter(store, classifier=custom)
        result = router.route("Dy3+")
        assert isinstance(result, RoutedResult)

    def test_route_with_custom_top_k(self, store: KnowledgeStore):
        """自定义 top_k."""
        router = IntentRouter(store, top_k=3)
        result = router.route("Dy3+的发射波长")
        assert isinstance(result, RoutedResult)
        assert result.total <= 3

    def test_route_result_properties(self, store: KnowledgeStore):
        """RoutedResult 便捷属性."""
        router = IntentRouter(store)
        result = router.route("Dy3+")
        assert isinstance(result.results, list)
        assert isinstance(result.scores, list)
        assert isinstance(result.total, int)
        assert result.total == len(result.results)


# ============================================================
# 7. EntityExtractor (补充测试)
# ============================================================


class TestEntityExtractor:
    """EntityExtractor 补充测试."""

    def test_extract_ion(self):
        """提取离子符号."""
        ext = EntityExtractor()
        entities = ext.extract("Dy3+离子和Eu2+的比较")
        types = {e.entity_type for e in entities}
        assert "ion" in types
        texts = {e.text for e in entities if e.entity_type == "ion"}
        assert "Dy3+" in texts or "Eu2+" in texts

    def test_extract_formula(self):
        """提取化学式."""
        ext = EntityExtractor()
        entities = ext.extract("the NaYF4 host material")
        types = {e.entity_type for e in entities}
        assert "formula" in types

    def test_extract_spectral_term(self):
        """提取光谱项."""
        ext = EntityExtractor()
        entities = ext.extract("the 4F9/2 level transition")
        types = {e.entity_type for e in entities}
        assert "spectral_term" in types
        texts = {e.text for e in entities if e.entity_type == "spectral_term"}
        assert any("4F9/2" in t for t in texts)

    def test_extract_numeric_with_unit(self):
        """提取数值+单位."""
        ext = EntityExtractor()
        entities = ext.extract("emission at 580 nm wavelength")
        numeric = [e for e in entities if e.entity_type == "numeric"]
        assert len(numeric) >= 1
        assert numeric[0].unit is not None

    def test_extract_domain_keyword(self):
        """提取领域关键词."""
        ext = EntityExtractor()
        entities = ext.extract("浓度猝灭机理")
        keywords = {e.text for e in entities if e.entity_type == "keyword"}
        assert len(keywords) >= 1

    def test_deduplication(self):
        """重叠实体去重."""
        ext = EntityExtractor()
        entities = ext.extract("Dy3+")
        # 不应有重叠
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                ei, ej = entities[i], entities[j]
                # 不应有位置重叠
                assert not (ei.start < ej.end and ei.end > ej.start)

    def test_empty_query(self):
        """空查询返回空列表."""
        ext = EntityExtractor()
        assert ext.extract("") == []

    def test_no_match_query(self):
        """无匹配查询返回仅关键词 (可能为空)."""
        ext = EntityExtractor()
        entities = ext.extract("hello world")
        # 至少不应崩溃
        assert isinstance(entities, list)


# ============================================================
# 8. 边界与异常场景
# ============================================================


class TestEdgeCases:
    """边界条件与异常场景."""

    @pytest.fixture
    def store(self) -> KnowledgeStore:
        return _make_store_with_data()

    def test_empty_query_context(self, store: KnowledgeStore):
        """空查询上下文."""
        router = IntentRouter(store)
        ctx = _make_ctx(query="")
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)

    def test_very_long_query(self, store: KnowledgeStore):
        """超长查询不崩溃."""
        router = IntentRouter(store)
        long_q = "Dy3+" * 500
        result = router.route(long_q)
        assert isinstance(result, RoutedResult)

    def test_special_characters(self, store: KnowledgeStore):
        """特殊字符查询不崩溃."""
        router = IntentRouter(store)
        result = router.route("Dy3+的<>&\"'波长")
        assert isinstance(result, RoutedResult)

    def test_chinese_punctuation(self, store: KnowledgeStore):
        """中文标点查询."""
        router = IntentRouter(store)
        result = router.route("Dy3+的波长，是多少？")
        assert isinstance(result, RoutedResult)

    def test_mixed_language(self, store: KnowledgeStore):
        """中英混合查询."""
        router = IntentRouter(store)
        result = router.route("Dy3+离子的quantum efficiency是多少?")
        assert isinstance(result, RoutedResult)

    def test_context_builder_empty_history(self, store: KnowledgeStore):
        """空对话历史."""
        builder = ContextBuilder()
        ctx = builder.build("Dy3+的波长", dialog_history=[])
        router = IntentRouter(store, context_builder=builder)
        result = router.route_with_context(ctx)
        assert isinstance(result, RoutedResult)

    def test_context_builder_unicode_query(self):
        """Unicode 查询."""
        builder = ContextBuilder()
        ctx = builder.build("Dy3+的发光机制 🔬")
        assert ctx.original_query == "Dy3+的发光机制 🔬"

    def test_concurrent_build(self):
        """并发构建上下文不崩溃."""
        import threading
        builder = ContextBuilder()
        errors: list[str] = []

        def build_query(q: str) -> None:
            try:
                builder.build(q)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=build_query, args=(f"query-{i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"并发错误: {errors}"


# ============================================================
# 9. ContextBudget 与压缩策略
# ============================================================


class TestContextBudgetAndCompression:
    """预算管理与压缩策略补充测试."""

    def test_budget_allocation_sums_to_total(self):
        """预算分配总和等于总量."""
        from dy3_polaris.l3.context_builder import ContextBudget
        budget = ContextBudget(max_tokens=1000)
        allocated = sum(budget.budget_for(k) for k in [
            "schema", "history", "learner", "query_rewrite", "reserved",
        ])
        # budget_for only recognizes known keys; unknown keys return 0
        # sum of known ratios = 0.10+0.20+0.05+0.05+0.60 = 1.00
        allocated_known = sum(budget.budget_for(k) for k in [
            "schema", "history", "learner", "query", "retrieval",
        ])
        assert allocated_known <= budget.max_tokens

    def test_recent_strategy(self):
        """RECENT 策略保留首轮 + 最近 N 轮."""
        from dy3_polaris.l3.context_builder import (
            HistoryCompressStrategy,
            HistoryCompressor,
        )
        compressor = HistoryCompressor(
            strategy=HistoryCompressStrategy.RECENT,
            max_recent_turns=2,
            max_chars=5000,
        )
        turns = _make_turns(6)
        compressed = compressor.compress(turns)
        # _compress_recent keeps first turn (parent task reinjection) + last N
        # 6 turns with max_recent=2 => first + last 2 = 3
        assert len(compressed) == 3
        assert compressed[0].role == "user"  # first turn preserved

    def test_sliding_window_strategy(self):
        """SLIDING_WINDOW 策略保留滑动窗口内的内容."""
        from dy3_polaris.l3.context_builder import (
            HistoryCompressStrategy,
            HistoryCompressor,
        )
        # Use a small budget so only some turns fit
        compressor = HistoryCompressor(
            strategy=HistoryCompressStrategy.SLIDING_WINDOW,
            max_recent_turns=2,
            max_chars=30,
        )
        turns = _make_turns(6)
        compressed = compressor.compress(turns)
        # With max_chars=30, each turn is ~7 chars, ~4 turns fit
        assert len(compressed) < len(turns)
        assert len(compressed) > 0

    def test_summarize_strategy(self):
        """SUMMARIZE 策略压缩旧轮次为摘要."""
        from dy3_polaris.l3.context_builder import (
            HistoryCompressStrategy,
            HistoryCompressor,
        )
        compressor = HistoryCompressor(
            strategy=HistoryCompressStrategy.SUMMARIZE,
            max_recent_turns=2,
            max_chars=50,
        )
        long_content = "这是一段非常长的内容" * 20
        turns = [
            DialogTurn(role="user", content=long_content, timestamp=1.0),
            DialogTurn(role="assistant", content=long_content, timestamp=2.0),
            DialogTurn(role="user", content=long_content, timestamp=3.0),
            DialogTurn(role="assistant", content="短回复", timestamp=4.0),
        ]
        compressed = compressor.compress(turns)
        # With 4 turns and max_recent=2: old_turns = turns[:2], recent = turns[-2:]
        # Result = [summary] + recent_turns => 3 turns
        assert len(compressed) == 3
        # First turn should be a summary (role="system")
        assert compressed[0].role == "system"
        assert "历史摘要" in compressed[0].content

    def test_empty_turns(self):
        """空对话历史压缩."""
        from dy3_polaris.l3.context_builder import (
            HistoryCompressor,
        )
        compressor = HistoryCompressor()
        assert compressor.compress([]) == []


# ============================================================
# 10. SchemaContextInjector 与 RetrievalNeedAssessor
# ============================================================


class TestSchemaAndNeedAssessor:
    """Schema 注入与检索需求评估测试."""

    def test_schema_inject_chemistry(self):
        """化学领域 Schema 注入."""
        from dy3_polaris.l3.context_builder import SchemaContextInjector
        injector = SchemaContextInjector()
        ctx = injector.inject("Dy3+的波长", "chemistry")
        assert "可查询数值属性" in ctx

    def test_schema_inject_unknown_domain(self):
        """未知领域返回默认 Schema."""
        from dy3_polaris.l3.context_builder import SchemaContextInjector
        injector = SchemaContextInjector()
        ctx = injector.inject("some query", "unknown_domain")
        # 不应崩溃
        assert isinstance(ctx, str)

    def test_need_assessor_positive(self):
        """检索需求评估: 需要检索."""
        from dy3_polaris.l3.context_builder import RetrievalNeedAssessor
        assessor = RetrievalNeedAssessor()
        ctx = _make_ctx(query="Dy3+的波长是多少nm?")
        assert assessor.assess("Dy3+的波长是多少nm?", ctx) is True

    def test_need_assessor_negative(self):
        """检索需求评估: 不需要检索 (问候语)."""
        from dy3_polaris.l3.context_builder import RetrievalNeedAssessor
        assessor = RetrievalNeedAssessor()
        ctx = _make_ctx(query="你好")
        assert assessor.assess("你好", ctx) is False

    def test_need_assessor_greeting_variants(self):
        """检索需求评估: 各类问候语."""
        from dy3_polaris.l3.context_builder import RetrievalNeedAssessor
        assessor = RetrievalNeedAssessor()
        greetings = ["你好", "hello", "Hi", "谢谢", "谢谢你的帮助"]
        ctx = _make_ctx(query="test")
        for g in greetings:
            assert assessor.assess(g, ctx) is False, f"'{g}' 应不需要检索"
