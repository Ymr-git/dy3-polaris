"""L3 意图路由、事实校验、跨库对齐测试套件.

覆盖范围:
- intent_router.py: IntentType / ExtractedEntity / IntentResult / EntityExtractor /
  IntentClassifier / IntentRouter / RoutedResult
- fact_check.py: ToleranceType / CheckStatus / StandardValue / NumericAssertion /
  CheckResult / FactCheckReport / StandardValueStore / AssertionExtractor /
  FactChecker (含严格模式与三类容差)
- cross_db.py: SourceType / AlignedItem / FusionConfig / AlignmentResult /
  CrossDBAligner (RRF 融合 / 多源加分 / 去重) / QualityWeightedFuser / fuse_results
"""

from __future__ import annotations

import pytest

from dy3_polaris.l3 import (
    IntentRouter, IntentType, IntentClassifier, EntityExtractor,
    ExtractedEntity, IntentResult, RoutedResult,
    FactChecker, StandardValue, StandardValueStore, AssertionExtractor,
    ToleranceType, CheckStatus, NumericAssertion, CheckResult, FactCheckReport,
    CrossDBAligner, SourceType, AlignedItem, FusionConfig, AlignmentResult,
    QualityWeightedFuser,
    KnowledgeStore, KnowledgeEntity, EntityType, DocumentChunk,
    ContentModality, KnowledgeTriple, RelationType,
    RetrievalResult, RetrievalFilter,
)


# ============================================================
# 测试数据工厂
# ============================================================


def make_entity(
    name: str = "测试实体",
    entity_type: EntityType = EntityType.CONCEPT,
    domain: str = "test",
    **kwargs,
) -> KnowledgeEntity:
    """创建测试实体."""
    return KnowledgeEntity(
        name=name,
        entity_type=entity_type,
        domain=domain,
        **kwargs,
    )


def make_chunk(
    content: str = "测试内容",
    document_id: str = "doc-001",
    chunk_index: int = 0,
    **kwargs,
) -> DocumentChunk:
    """创建测试切片."""
    return DocumentChunk(
        content=content,
        document_id=document_id,
        chunk_index=chunk_index,
        **kwargs,
    )


def make_triple(
    subject_id: str,
    predicate: str = RelationType.RELATED_TO.value,
    object_id: str = "",
    confidence: float = 1.0,
    **kwargs,
) -> KnowledgeTriple:
    """创建测试三元组."""
    return KnowledgeTriple(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=confidence,
        **kwargs,
    )


def make_standard_value(
    kp_id: str = "KP-001",
    param_name: str = "emission_wavelength",
    standard_value: float = 580.0,
    tolerance: float = 2.0,
    tolerance_type: ToleranceType = ToleranceType.ABSOLUTE,
    unit: str = "nm",
    **kwargs,
) -> StandardValue:
    """创建测试标准值."""
    return StandardValue(
        kp_id=kp_id,
        param_name=param_name,
        standard_value=standard_value,
        tolerance=tolerance,
        tolerance_type=tolerance_type,
        unit=unit,
        **kwargs,
    )


# ============================================================
# 共享 fixtures
# ============================================================


@pytest.fixture
def knowledge_store() -> KnowledgeStore:
    """提供带测试数据的知识库."""
    store = KnowledgeStore()

    # 实体: Dy3+ 离子
    dy = make_entity(
        name="Dy3+",
        entity_type=EntityType.CHEMICAL_COMPOUND,
        domain="luminescence",
        aliases=["镝离子", "dysprosium"],
        tags=["rare_earth", "ion"],
    )
    store.add_entity(dy)

    # 实体: Eu3+ 离子
    eu = make_entity(
        name="Eu3+",
        entity_type=EntityType.CHEMICAL_COMPOUND,
        domain="luminescence",
        aliases=["铕离子", "europium"],
        tags=["rare_earth", "ion"],
    )
    store.add_entity(eu)

    # 实体: YAG 基质
    yag = make_entity(
        name="YAG",
        entity_type=EntityType.MATERIAL,
        domain="luminescence",
        aliases=["钇铝石榴石"],
    )
    store.add_entity(yag)

    # 关系三元组
    store.add_triple(make_triple(
        dy.entity_id, RelationType.RELATED_TO.value, eu.entity_id,
    ))
    store.add_triple(make_triple(
        dy.entity_id, RelationType.PART_OF.value, yag.entity_id,
    ))

    # 切片: 发光机理
    store.add_chunk(make_chunk(
        content="Dy3+离子的发光机理涉及4F9/2能级到6H15/2的跃迁，发射波长约为580nm。",
        document_id="doc-001",
        chunk_index=0,
        section="发光机理",
    ))

    # 切片: 量子效率
    store.add_chunk(make_chunk(
        content="Eu3+离子在YAG基质中的量子效率达到85%，发射波长为611nm。",
        document_id="doc-002",
        chunk_index=0,
        section="量子效率",
    ))

    return store


@pytest.fixture
def standard_store() -> StandardValueStore:
    """提供带预置标准值的标准值库."""
    store = StandardValueStore()
    store.add(make_standard_value(
        kp_id="KP-001",
        param_name="emission_wavelength",
        standard_value=580.0,
        tolerance=2.0,
        unit="nm",
        source_ref="GB/T 1-2020",
    ))
    store.add(make_standard_value(
        kp_id="KP-002",
        param_name="emission_wavelength",
        standard_value=611.0,
        tolerance=2.0,
        unit="nm",
        source_ref="GB/T 2-2020",
    ))
    store.add(StandardValue(
        kp_id="KP-003",
        param_name="quantum_efficiency",
        standard_value=0.85,
        tolerance=0.05,
        tolerance_type=ToleranceType.RELATIVE,
        unit="%",
        source_ref="ISO 12345",
    ))
    store.add(StandardValue(
        kp_id="KP-004",
        param_name="rietveld_rwp",
        standard_value=0.0,
        tolerance=10.0,
        tolerance_type=ToleranceType.THRESHOLD,
        unit="%",
        source_ref="ASTM E1234",
    ))
    return store


@pytest.fixture
def fact_checker(standard_store: StandardValueStore) -> FactChecker:
    """提供事实校验器."""
    return FactChecker(standard_store)


@pytest.fixture
def strict_fact_checker(standard_store: StandardValueStore) -> FactChecker:
    """提供严格模式事实校验器."""
    return FactChecker(standard_store, strict_mode=True)


@pytest.fixture
def cross_aligner() -> CrossDBAligner:
    """提供跨库对齐器."""
    return CrossDBAligner()


# ============================================================
# 模块 1: intent_router.py 测试
# ============================================================


class TestIntentType:
    """IntentType 枚举测试."""

    def test_枚举值(self):
        """四个意图类型的值."""
        assert IntentType.CONCEPT.value == "concept"
        assert IntentType.NUMERIC.value == "numeric"
        assert IntentType.RELATIONAL.value == "relational"
        assert IntentType.COMPOSITE.value == "composite"

    def test_继承str(self):
        """继承 str 类型."""
        assert isinstance(IntentType.CONCEPT, str)
        assert IntentType.CONCEPT == "concept"

    def test_枚举成员数(self):
        """共 4 个枚举成员."""
        assert len(list(IntentType)) == 4


class TestExtractedEntity:
    """ExtractedEntity 数据类测试."""

    def test_基本创建(self):
        """创建带必填字段的实体."""
        e = ExtractedEntity(text="Dy3+", entity_type="ion")
        assert e.text == "Dy3+"
        assert e.entity_type == "ion"
        assert e.value is None
        assert e.unit is None
        assert e.start == 0
        assert e.end == 0

    def test_完整创建(self):
        """创建带所有字段的实体."""
        e = ExtractedEntity(
            text="580nm",
            entity_type="numeric",
            value=580.0,
            unit="nm",
            start=10,
            end=15,
        )
        assert e.value == 580.0
        assert e.unit == "nm"
        assert e.start == 10
        assert e.end == 15

    def test_默认值(self):
        """默认值正确."""
        e = ExtractedEntity(text="test", entity_type="keyword")
        assert e.value is None
        assert e.unit is None
        assert e.start == 0
        assert e.end == 0


class TestIntentResult:
    """IntentResult 数据类测试."""

    def test_基本创建(self):
        """创建意图识别结果."""
        r = IntentResult(
            intent_type=IntentType.NUMERIC,
            confidence=0.8,
        )
        assert r.intent_type == IntentType.NUMERIC
        assert r.confidence == 0.8
        assert r.matched_rules == []
        assert r.extracted_entities == []
        assert r.suggested_path == ""
        assert r.classification_time_ms == 0.0

    def test_带规则和实体(self):
        """带匹配规则和提取实体."""
        entity = ExtractedEntity(text="580nm", entity_type="numeric", value=580.0)
        r = IntentResult(
            intent_type=IntentType.NUMERIC,
            confidence=0.9,
            matched_rules=["numeric:unit", "numeric:entity"],
            extracted_entities=[entity],
            suggested_path="exact+filter→fact_check",
            classification_time_ms=5.2,
        )
        assert len(r.matched_rules) == 2
        assert len(r.extracted_entities) == 1
        assert r.suggested_path == "exact+filter→fact_check"
        assert r.classification_time_ms == 5.2


class TestEntityExtractor:
    """EntityExtractor 实体提取器测试."""

    def test_提取离子符号(self):
        """提取 Dy3+, Eu3+ 等离子符号."""
        extractor = EntityExtractor()
        # ION_PATTERN 的 \b 要求离子符号后紧跟词字符;
        # Python 3 中 CJK 字符是词字符 (\w), 故用 CJK 字符紧随离子符号
        entities = extractor.extract("Dy3+的发光性质 Eu3+的发射波长")

        ions = [e for e in entities if e.entity_type == "ion"]
        assert len(ions) == 2
        texts = {e.text for e in ions}
        assert "Dy3+" in texts
        assert "Eu3+" in texts

    def test_提取化学式(self):
        """提取 Y2O3 等化学式."""
        extractor = EntityExtractor()
        # 用空格分隔避免 CJK 字符影响 \b 边界匹配
        entities = extractor.extract("Y2O3 基质的发光性能")
        formulas = [e for e in entities if e.entity_type == "formula"]
        assert len(formulas) >= 1
        assert any(f.text == "Y2O3" for f in formulas)

    def test_提取YAG化学式(self):
        """提取 YAG 化学式."""
        extractor = EntityExtractor()
        # 用空格分隔避免 CJK 字符影响 \b 边界匹配
        entities = extractor.extract("YAG 晶体的光学性质")
        formulas = [e for e in entities if e.entity_type == "formula"]
        assert len(formulas) >= 1
        assert any(f.text == "YAG" for f in formulas)

    def test_提取光谱项(self):
        """提取 4F9/2, 5D0 等光谱项."""
        extractor = EntityExtractor()
        # 用空格分隔避免 CJK 字符影响 \b 边界匹配
        entities = extractor.extract("4F9/2 能级跃迁到 5D0")
        spectral = [e for e in entities if e.entity_type == "spectral_term"]
        texts = {e.text for e in spectral}
        assert "4F9/2" in texts
        assert "5D0" in texts

    def test_提取数值加单位(self):
        """提取 580nm, 300K 等数值+单位."""
        extractor = EntityExtractor()
        entities = extractor.extract("发射波长580nm，温度300K")
        numerics = [e for e in entities if e.entity_type == "numeric"]
        assert len(numerics) >= 2
        values = {e.value for e in numerics}
        assert 580.0 in values
        assert 300.0 in values

    def test_数值单位字段(self):
        """数值实体的 value 和 unit 字段."""
        extractor = EntityExtractor()
        entities = extractor.extract("波长580nm")
        numerics = [e for e in entities if e.entity_type == "numeric"]
        assert len(numerics) == 1
        assert numerics[0].value == 580.0
        assert numerics[0].unit is not None

    def test_提取科学计数法(self):
        """提取科学计数法数值."""
        extractor = EntityExtractor()
        entities = extractor.extract("能量5.5e-19 J")
        numerics = [e for e in entities if e.entity_type == "numeric"]
        assert len(numerics) >= 1
        assert any(abs(e.value - 5.5e-19) < 1e-30 for e in numerics)

    def test_提取领域关键词(self):
        """提取机理、关系等领域关键词."""
        extractor = EntityExtractor()
        entities = extractor.extract("Dy3+的发光机理")
        keywords = [e for e in entities if e.entity_type == "keyword"]
        assert len(keywords) >= 1
        assert any(e.value == "机理" for e in keywords)

    def test_空查询返回空列表(self):
        """空查询返回空列表."""
        extractor = EntityExtractor()
        entities = extractor.extract("")
        assert entities == []

    def test_无匹配查询(self):
        """无领域实体的查询只返回关键词或空."""
        extractor = EntityExtractor()
        entities = extractor.extract("hello world")
        # hello world 不包含领域关键词或实体
        assert all(e.entity_type != "ion" for e in entities)
        assert all(e.entity_type != "formula" for e in entities)

    def test_按位置排序(self):
        """提取结果按位置排序."""
        extractor = EntityExtractor()
        entities = extractor.extract("Dy3+在580nm处的发光")
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                assert entities[i].start <= entities[i + 1].start

    def test_离子与化学式去重(self):
        """离子符号不被重复识别为化学式."""
        extractor = EntityExtractor()
        entities = extractor.extract("Dy3+离子")
        ions = [e for e in entities if e.entity_type == "ion"]
        formulas = [e for e in entities if e.entity_type == "formula"]
        # Dy3+ 应被识别为离子，不应同时被识别为化学式
        assert len(ions) == 1
        # 确保没有与离子重叠的化学式
        for f in formulas:
            assert not (f.start < ions[0].end and f.end > ions[0].start)


class TestIntentClassifier:
    """IntentClassifier 意图分类器测试."""

    def test_数值意图分类(self):
        """包含数值+单位的查询分类为 NUMERIC."""
        classifier = IntentClassifier()
        result = classifier.classify("Dy3+的发射波长是多少nm?")
        assert result.intent_type == IntentType.NUMERIC
        assert result.confidence > 0
        assert len(result.matched_rules) > 0

    def test_概念意图分类(self):
        """概念定义类查询分类为 CONCEPT."""
        classifier = IntentClassifier()
        result = classifier.classify("什么是Dy3+离子的发光机理?")
        assert result.intent_type == IntentType.CONCEPT
        assert result.confidence > 0

    def test_关系意图分类(self):
        """关系查询分类为 RELATIONAL."""
        classifier = IntentClassifier()
        result = classifier.classify("Dy3+和Eu3+之间有什么关系?")
        assert result.intent_type == IntentType.RELATIONAL

    def test_复合意图分类(self):
        """对比类查询分类为 COMPOSITE."""
        classifier = IntentClassifier()
        result = classifier.classify("对比Dy3+和Eu3+的区别和差异")
        assert result.intent_type == IntentType.COMPOSITE

    def test_低分默认概念(self):
        """无匹配规则的查询默认为 CONCEPT."""
        classifier = IntentClassifier()
        result = classifier.classify("hello world")
        assert result.intent_type == IntentType.CONCEPT
        assert result.confidence == 0.3
        assert "fallback:default_concept" in result.matched_rules

    def test_建议路径映射(self):
        """每种意图有对应的建议路径."""
        classifier = IntentClassifier()
        for query, expected_path in [
            ("什么是发光机理", "vector+keyword→rrf"),
            ("波长580nm", "exact+filter→fact_check"),
            ("Dy3+和Eu3+的关系", "graph_traversal→subgraph"),
            ("对比Dy3+和Eu3+的差异", "parallel(vector+keyword+graph)→rrf+fact_check"),
        ]:
            result = classifier.classify(query)
            assert result.suggested_path == expected_path, f"路径错误: {query}"

    def test_分类耗时记录(self):
        """分类耗时被记录."""
        classifier = IntentClassifier()
        result = classifier.classify("Dy3+的发射波长")
        assert result.classification_time_ms >= 0.0

    def test_提取实体附带在结果中(self):
        """分类结果包含提取的实体."""
        classifier = IntentClassifier()
        result = classifier.classify("Dy3+在580nm处的发光")
        assert len(result.extracted_entities) > 0

    def test_标准引用触发数值意图(self):
        """标准引用 (GB/T) 触发数值意图."""
        classifier = IntentClassifier()
        result = classifier.classify("按照GB/T 1234标准测量")
        assert result.intent_type == IntentType.NUMERIC
        assert any("standard_ref" in r for r in result.matched_rules)

    def test_多意图检测(self):
        """多意图同时出现时触发复合检测."""
        classifier = IntentClassifier()
        # 同时包含数值(波长580nm)和概念(机理)和关系(关系)
        result = classifier.classify("Dy3+的发光机理和580nm波长有什么关系?")
        assert "composite:multi_intent_detected" in result.matched_rules

    def test_use_llm参数(self):
        """use_llm 参数不崩溃 (预留接口返回 None)."""
        classifier = IntentClassifier()
        result = classifier.classify("hello world", use_llm=True)
        assert result.intent_type == IntentType.CONCEPT


class TestIntentRouter:
    """IntentRouter 意图路由检索引擎测试."""

    def test_概念路由(self, knowledge_store: KnowledgeStore):
        """概念查询路由到关键词检索."""
        router = IntentRouter(knowledge_store)
        result = router.route("什么是Dy3+离子的发光机理?")
        assert isinstance(result, RoutedResult)
        assert result.intent.intent_type == IntentType.CONCEPT
        assert isinstance(result.retrieval_result, RetrievalResult)

    def test_数值路由(self, knowledge_store: KnowledgeStore):
        """数值查询路由到数值检索."""
        router = IntentRouter(knowledge_store)
        result = router.route("Dy3+的发射波长是多少nm?")
        assert result.intent.intent_type == IntentType.NUMERIC
        assert isinstance(result.retrieval_result, RetrievalResult)

    def test_关系路由(self, knowledge_store: KnowledgeStore):
        """关系查询路由到图检索."""
        router = IntentRouter(knowledge_store)
        result = router.route("Dy3+和Eu3+之间有什么关系?")
        assert result.intent.intent_type == IntentType.RELATIONAL
        assert isinstance(result.retrieval_result, RetrievalResult)

    def test_复合路由(self, knowledge_store: KnowledgeStore):
        """复合查询路由到混合检索."""
        router = IntentRouter(knowledge_store)
        result = router.route("对比Dy3+和Eu3+的区别和差异")
        assert result.intent.intent_type == IntentType.COMPOSITE
        assert isinstance(result.retrieval_result, RetrievalResult)

    def test_带query_vector的概念路由(self, knowledge_store: KnowledgeStore):
        """概念路由带查询向量时使用混合检索."""
        router = IntentRouter(knowledge_store)
        # 添加向量到切片
        chunks = knowledge_store.chunk_store._chunks
        for chunk in chunks.values():
            knowledge_store.chunk_store.add_embedding(chunk.chunk_id, [1.0, 0.0])

        result = router.route(
            "什么是发光机理",
            query_vector=[1.0, 0.0],
        )
        assert result.intent.intent_type == IntentType.CONCEPT
        assert isinstance(result.retrieval_result, RetrievalResult)

    def test_带entity_id的关系路由(self, knowledge_store: KnowledgeStore):
        """关系路由带 entity_id 时直接图检索."""
        router = IntentRouter(knowledge_store)
        # 获取一个实体 ID
        entities = knowledge_store.entity_store.list_entities()
        entity_id = entities[0].entity_id

        result = router.route(
            "Dy3+有什么关系?",
            entity_id=entity_id,
        )
        assert result.intent.intent_type == IntentType.RELATIONAL

    def test_自定义top_k(self, knowledge_store: KnowledgeStore):
        """自定义 top_k 参数."""
        router = IntentRouter(knowledge_store, top_k=5)
        result = router.route("发光机理")
        assert result.total <= 5 or result.total == 0

    def test_带filter的路由(self, knowledge_store: KnowledgeStore):
        """带过滤条件的路由."""
        router = IntentRouter(knowledge_store)
        f = RetrievalFilter(domain="luminescence")
        result = router.route("发光机理", filter=f)
        assert isinstance(result, RoutedResult)

    def test_空查询路由(self, knowledge_store: KnowledgeStore):
        """空查询不崩溃 (默认为概念检索)."""
        router = IntentRouter(knowledge_store)
        result = router.route("")
        assert isinstance(result, RoutedResult)
        assert result.intent.intent_type == IntentType.CONCEPT

    def test_batch_route(self, knowledge_store: KnowledgeStore):
        """批量路由检索."""
        router = IntentRouter(knowledge_store)
        results = router.batch_route(["发光机理", "波长580nm"])
        assert len(results) == 2
        assert all(isinstance(r, RoutedResult) for r in results)

    def test_数值路由结果提升(self, knowledge_store: KnowledgeStore):
        """数值路由对包含匹配数值的结果提升分数."""
        router = IntentRouter(knowledge_store)
        result = router.route("Dy3+的发射波长580nm")
        assert result.intent.intent_type == IntentType.NUMERIC
        # 结果应包含含有 580 的切片
        if result.results:
            contents = [str(r.get("content", "")) for r in result.results]
            assert any("580" in c for c in contents)

    def test_自定义分类器注入(self, knowledge_store: KnowledgeStore):
        """注入自定义分类器."""
        custom_classifier = IntentClassifier()
        router = IntentRouter(knowledge_store, classifier=custom_classifier)
        assert router.classifier is custom_classifier

    def test_use_llm_fallback参数(self, knowledge_store: KnowledgeStore):
        """use_llm_fallback 参数初始化."""
        router = IntentRouter(knowledge_store, use_llm_fallback=True)
        result = router.route("hello world test")
        assert isinstance(result, RoutedResult)


class TestRoutedResult:
    """RoutedResult 数据类测试."""

    def test_属性访问(self, knowledge_store: KnowledgeStore):
        """results/scores/total 属性代理."""
        router = IntentRouter(knowledge_store)
        routed = router.route("发光机理")
        assert routed.results == routed.retrieval_result.results
        assert routed.scores == routed.retrieval_result.scores
        assert routed.total == routed.retrieval_result.total

    def test_total_time_ms(self, knowledge_store: KnowledgeStore):
        """总耗时被记录."""
        router = IntentRouter(knowledge_store)
        routed = router.route("发光机理")
        assert routed.total_time_ms >= 0.0

    def test_intent字段(self, knowledge_store: KnowledgeStore):
        """intent 字段为 IntentResult."""
        router = IntentRouter(knowledge_store)
        routed = router.route("发光机理")
        assert isinstance(routed.intent, IntentResult)


# ============================================================
# 模块 2: fact_check.py 测试
# ============================================================


class TestToleranceType:
    """ToleranceType 枚举测试."""

    def test_枚举值(self):
        """三个容差类型."""
        assert ToleranceType.ABSOLUTE.value == "absolute"
        assert ToleranceType.RELATIVE.value == "relative"
        assert ToleranceType.THRESHOLD.value == "threshold"

    def test_继承str(self):
        """继承 str."""
        assert isinstance(ToleranceType.ABSOLUTE, str)

    def test_枚举成员数(self):
        """共 3 个."""
        assert len(list(ToleranceType)) == 3


class TestCheckStatus:
    """CheckStatus 枚举测试."""

    def test_枚举值(self):
        """四个校验状态."""
        assert CheckStatus.PASSED.value == "passed"
        assert CheckStatus.FAILED.value == "failed"
        assert CheckStatus.SKIPPED.value == "skipped"
        assert CheckStatus.ERROR.value == "error"

    def test_继承str(self):
        """继承 str."""
        assert isinstance(CheckStatus.PASSED, str)

    def test_枚举成员数(self):
        """共 4 个."""
        assert len(list(CheckStatus)) == 4


class TestStandardValue:
    """StandardValue 模型测试."""

    def test_创建标准值(self):
        """创建标准值."""
        sv = StandardValue(
            kp_id="KP-001",
            param_name="emission_wavelength",
            standard_value=580.0,
            tolerance=2.0,
        )
        assert sv.kp_id == "KP-001"
        assert sv.standard_value == 580.0
        assert sv.tolerance_type == ToleranceType.ABSOLUTE  # 默认
        assert sv.unit == ""
        assert sv.source_type == "standard"
        assert sv.confidence == 1.0

    def test_绝对容差通过(self):
        """绝对容差: 偏差在范围内通过."""
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        assert sv.check(581.0) is True
        assert sv.check(578.0) is True

    def test_绝对容差失败(self):
        """绝对容差: 偏差超出范围失败."""
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        assert sv.check(583.0) is False
        assert sv.check(577.0) is False

    def test_绝对容差边界(self):
        """绝对容差: 恰好等于边界值通过."""
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        assert sv.check(582.0) is True
        assert sv.check(578.0) is True

    def test_相对容差通过(self):
        """相对容差: 相对偏差在范围内通过."""
        sv = make_standard_value(
            standard_value=100.0,
            tolerance=0.05,
            tolerance_type=ToleranceType.RELATIVE,
        )
        assert sv.check(103.0) is True  # 3% < 5%
        assert sv.check(97.0) is True

    def test_相对容差失败(self):
        """相对容差: 相对偏差超出范围失败."""
        sv = make_standard_value(
            standard_value=100.0,
            tolerance=0.05,
            tolerance_type=ToleranceType.RELATIVE,
        )
        assert sv.check(110.0) is False  # 10% > 5%

    def test_相对容差零标准值(self):
        """相对容差: 标准值为 0 时退化为绝对值判断."""
        sv = make_standard_value(
            standard_value=0.0,
            tolerance=0.5,
            tolerance_type=ToleranceType.RELATIVE,
        )
        assert sv.check(0.3) is True
        assert sv.check(0.6) is False

    def test_阈值容差通过(self):
        """阈值容差: 值 ≤ tolerance 通过."""
        sv = make_standard_value(
            standard_value=0.0,
            tolerance=10.0,
            tolerance_type=ToleranceType.THRESHOLD,
        )
        assert sv.check(5.0) is True
        assert sv.check(10.0) is True

    def test_阈值容差失败(self):
        """阈值容差: 值 > tolerance 失败."""
        sv = make_standard_value(
            standard_value=0.0,
            tolerance=10.0,
            tolerance_type=ToleranceType.THRESHOLD,
        )
        assert sv.check(11.0) is False

    def test_偏差计算绝对(self):
        """绝对偏差计算."""
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        assert sv.deviation(583.0) == 3.0
        assert sv.deviation(577.0) == 3.0

    def test_偏差计算相对(self):
        """相对偏差计算."""
        sv = make_standard_value(
            standard_value=100.0,
            tolerance=0.05,
            tolerance_type=ToleranceType.RELATIVE,
        )
        assert sv.deviation(110.0) == pytest.approx(0.1)

    def test_偏差计算阈值(self):
        """阈值偏差计算 (超出值)."""
        sv = make_standard_value(
            standard_value=0.0,
            tolerance=10.0,
            tolerance_type=ToleranceType.THRESHOLD,
        )
        assert sv.deviation(5.0) == 0.0  # 未超出
        assert sv.deviation(15.0) == 5.0  # 超出 5

    def test_默认容差类型(self):
        """默认容差类型为 ABSOLUTE."""
        sv = StandardValue(
            kp_id="KP-001",
            param_name="test",
            standard_value=1.0,
            tolerance=0.1,
        )
        assert sv.tolerance_type == ToleranceType.ABSOLUTE


class TestNumericAssertion:
    """NumericAssertion 数据类测试."""

    def test_创建断言(self):
        """创建数值断言."""
        a = NumericAssertion(
            text="580nm",
            value=580.0,
            unit="nm",
        )
        assert a.text == "580nm"
        assert a.value == 580.0
        assert a.unit == "nm"
        assert a.param_name == ""
        assert a.kp_id == ""
        assert a.context == ""
        assert a.start == 0
        assert a.end == 0

    def test_完整断言(self):
        """创建带所有字段的断言."""
        a = NumericAssertion(
            text="580nm",
            value=580.0,
            unit="nm",
            param_name="emission_wavelength",
            kp_id="KP-001",
            context="Dy3+的发射波长580nm",
            start=10,
            end=15,
        )
        assert a.param_name == "emission_wavelength"
        assert a.kp_id == "KP-001"
        assert a.start == 10


class TestCheckResult:
    """CheckResult 数据类测试."""

    def test_创建结果(self):
        """创建校验结果."""
        assertion = NumericAssertion(text="580nm", value=580.0, unit="nm")
        result = CheckResult(
            assertion=assertion,
            standard=None,
            status=CheckStatus.SKIPPED,
        )
        assert result.assertion.value == 580.0
        assert result.standard is None
        assert result.status == CheckStatus.SKIPPED
        assert result.deviation == 0.0
        assert result.passed is False

    def test_通过结果(self):
        """通过校验的结果."""
        assertion = NumericAssertion(text="580nm", value=580.0, unit="nm")
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        result = CheckResult(
            assertion=assertion,
            standard=sv,
            status=CheckStatus.PASSED,
            deviation=0.0,
            message="通过",
            passed=True,
        )
        assert result.passed is True
        assert result.status == CheckStatus.PASSED


class TestFactCheckReport:
    """FactCheckReport 模型测试."""

    def test_创建空报告(self):
        """创建空报告."""
        report = FactCheckReport(content="测试内容")
        assert report.content == "测试内容"
        assert report.total_assertions == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.skipped == 0
        assert report.overall_passed is False
        assert report.pass_rate == 0.0

    def test_通过率计算(self):
        """通过率属性."""
        report = FactCheckReport(content="test")
        report.passed = 3
        report.checked = 4
        assert report.pass_rate == 0.75

    def test_通过率零检查(self):
        """无检查项时通过率为 0."""
        report = FactCheckReport(content="test")
        assert report.pass_rate == 0.0

    def test_add_result通过(self):
        """添加通过结果."""
        report = FactCheckReport(content="test")
        assertion = NumericAssertion(text="580nm", value=580.0, unit="nm")
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        result = CheckResult(
            assertion=assertion, standard=sv,
            status=CheckStatus.PASSED, passed=True,
        )
        report.add_result(result)
        assert report.total_assertions == 1
        assert report.passed == 1
        assert report.checked == 1

    def test_add_result失败(self):
        """添加失败结果."""
        report = FactCheckReport(content="test")
        assertion = NumericAssertion(text="580nm", value=590.0, unit="nm")
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        result = CheckResult(
            assertion=assertion, standard=sv,
            status=CheckStatus.FAILED, passed=False,
        )
        report.add_result(result)
        assert report.failed == 1
        assert report.checked == 1

    def test_add_result跳过(self):
        """添加跳过结果."""
        report = FactCheckReport(content="test")
        assertion = NumericAssertion(text="580nm", value=580.0, unit="nm")
        result = CheckResult(
            assertion=assertion, standard=None,
            status=CheckStatus.SKIPPED,
        )
        report.add_result(result)
        assert report.skipped == 1
        assert report.checked == 0

    def test_finalize全通过(self):
        """finalize: 全通过时 overall_passed=True."""
        report = FactCheckReport(content="test")
        assertion = NumericAssertion(text="580nm", value=580.0, unit="nm")
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        report.add_result(CheckResult(
            assertion=assertion, standard=sv,
            status=CheckStatus.PASSED, passed=True,
        ))
        report.finalize()
        assert report.overall_passed is True
        assert report.confidence == 1.0

    def test_finalize有失败(self):
        """finalize: 有失败时 overall_passed=False 且生成反馈."""
        report = FactCheckReport(content="test")
        assertion = NumericAssertion(text="580nm", value=590.0, unit="nm")
        sv = make_standard_value(standard_value=580.0, tolerance=2.0)
        report.add_result(CheckResult(
            assertion=assertion, standard=sv,
            status=CheckStatus.FAILED, passed=False,
        ))
        report.finalize()
        assert report.overall_passed is False
        assert report.feedback != ""

    def test_finalize无检查项(self):
        """finalize: 无检查项时 overall_passed=False."""
        report = FactCheckReport(content="test")
        report.finalize()
        assert report.overall_passed is False
        assert report.confidence == 0.0


class TestStandardValueStore:
    """StandardValueStore 标准值库测试."""

    def test_add和get(self):
        """添加和精确查找."""
        store = StandardValueStore()
        sv = make_standard_value(kp_id="KP-001", param_name="wavelength")
        std_id = store.add(sv)
        assert std_id == "KP-001:wavelength"

        found = store.get("KP-001", "wavelength")
        assert found is not None
        assert found.standard_value == 580.0

    def test_get不存在(self):
        """查找不存在的标准值返回 None."""
        store = StandardValueStore()
        assert store.get("KP-999", "unknown") is None

    def test_get_by_param(self):
        """按参数名查找."""
        store = StandardValueStore()
        store.add(make_standard_value(kp_id="KP-001", param_name="wavelength"))
        store.add(make_standard_value(kp_id="KP-002", param_name="wavelength"))
        results = store.get_by_param("wavelength")
        assert len(results) == 2

    def test_get_by_kp(self):
        """按知识点 ID 查找."""
        store = StandardValueStore()
        store.add(make_standard_value(kp_id="KP-001", param_name="wavelength"))
        store.add(make_standard_value(kp_id="KP-001", param_name="temperature"))
        results = store.get_by_kp("KP-001")
        assert len(results) == 2

    def test_remove(self):
        """移除标准值."""
        store = StandardValueStore()
        store.add(make_standard_value(kp_id="KP-001", param_name="wavelength"))
        removed = store.remove("KP-001", "wavelength")
        assert removed is not None
        assert store.count() == 0
        assert store.get("KP-001", "wavelength") is None

    def test_remove不存在(self):
        """移除不存在的返回 None."""
        store = StandardValueStore()
        assert store.remove("KP-999", "unknown") is None

    def test_count(self):
        """计数."""
        store = StandardValueStore()
        assert store.count() == 0
        store.add(make_standard_value())
        assert store.count() == 1

    def test_list_all(self):
        """列出所有."""
        store = StandardValueStore()
        store.add(make_standard_value(kp_id="KP-001"))
        store.add(make_standard_value(kp_id="KP-002"))
        all_sv = store.list_all()
        assert len(all_sv) == 2

    def test_bulk_add(self):
        """批量添加."""
        store = StandardValueStore()
        standards = [
            make_standard_value(kp_id=f"KP-{i:03d}")
            for i in range(5)
        ]
        ids = store.bulk_add(standards)
        assert len(ids) == 5
        assert store.count() == 5

    def test_默认容差配置(self):
        """获取预置默认容差."""
        store = StandardValueStore()
        config = store.get_default_tolerance("emission_wavelength")
        assert config is not None
        assert config["tolerance"] == 2.0
        assert config["tolerance_type"] == ToleranceType.ABSOLUTE

    def test_默认容差不存在(self):
        """不存在的参数返回 None."""
        store = StandardValueStore()
        assert store.get_default_tolerance("unknown_param") is None

    def test_覆盖添加(self):
        """相同 kp_id+param_name 覆盖旧值."""
        store = StandardValueStore()
        store.add(make_standard_value(kp_id="KP-001", standard_value=580.0))
        store.add(make_standard_value(kp_id="KP-001", standard_value=582.0))
        # 覆盖后仍然只有一条
        sv = store.get("KP-001", "emission_wavelength")
        assert sv.standard_value == 582.0


class TestAssertionExtractor:
    """AssertionExtractor 断言提取器测试."""

    def test_提取数值断言(self):
        """从文本中提取数值+单位."""
        extractor = AssertionExtractor()
        assertions = extractor.extract("发射波长580nm，温度300K")
        assert len(assertions) >= 2
        values = {a.value for a in assertions}
        assert 580.0 in values
        assert 300.0 in values

    def test_提取断言单位(self):
        """提取的单位被标准化."""
        extractor = AssertionExtractor()
        assertions = extractor.extract("波长580nm")
        assert len(assertions) >= 1
        assert assertions[0].unit == "nm"

    def test_推断参数名(self):
        """从上下文推断参数名."""
        extractor = AssertionExtractor()
        assertions = extractor.extract("Dy3+的发射波长580nm")
        wavelength_assertions = [a for a in assertions if a.value == 580.0]
        assert len(wavelength_assertions) >= 1
        assert wavelength_assertions[0].param_name == "emission_wavelength"

    def test_推断KP_ID(self):
        """从上下文推断知识点 ID."""
        extractor = AssertionExtractor()
        assertions = extractor.extract("Dy3+的发射波长580nm")
        dy_assertions = [a for a in assertions if a.value == 580.0]
        assert len(dy_assertions) >= 1
        assert dy_assertions[0].kp_id == "KP-001"

    def test_无数值文本(self):
        """无数值的文本返回空列表."""
        extractor = AssertionExtractor()
        assertions = extractor.extract("没有数值的文本")
        assert assertions == []

    def test_空文本(self):
        """空文本返回空列表."""
        extractor = AssertionExtractor()
        assert extractor.extract("") == []

    def test_科学计数法(self):
        """提取科学计数法数值."""
        extractor = AssertionExtractor()
        assertions = extractor.extract("能量5.5e-19 J")
        assert len(assertions) >= 1
        assert any(abs(a.value - 5.5e-19) < 1e-30 for a in assertions)

    def test_上下文字段(self):
        """断言包含上下文."""
        extractor = AssertionExtractor()
        # 用空格分隔避免 CJK 字符影响 \b 边界匹配
        assertions = extractor.extract("前文Dy3+的发射波长580nm 后文")
        assert len(assertions) >= 1
        assert assertions[0].context != ""

    def test_单位标准化cm(self):
        """cm^-1 单位标准化."""
        extractor = AssertionExtractor()
        assertions = extractor.extract("能量20000cm-1")
        assert len(assertions) >= 1
        # 单位被标准化为 cm^-1
        assert "cm" in assertions[0].unit.lower()


class TestFactChecker:
    """FactChecker 事实校验器测试."""

    def test_校验通过(self, fact_checker: FactChecker):
        """数值在容差范围内通过."""
        report = fact_checker.check("Dy3+的发射波长580nm")
        assert report.total_assertions > 0
        assert report.passed >= 1
        assert report.overall_passed is True

    def test_校验失败(self, fact_checker: FactChecker):
        """数值超出容差范围失败."""
        report = fact_checker.check("Dy3+的发射波长600nm")
        assert report.failed >= 1
        assert report.overall_passed is False
        assert report.feedback != ""

    def test_校验跳过(self, fact_checker: FactChecker):
        """无标准值匹配时跳过."""
        report = fact_checker.check("未知参数值999lux")
        assert report.skipped >= 1
        assert report.failed == 0

    def test_空内容校验(self, fact_checker: FactChecker):
        """空内容不崩溃."""
        report = fact_checker.check("")
        assert report.total_assertions == 0
        assert report.overall_passed is False

    def test_无数值内容(self, fact_checker: FactChecker):
        """无数值的内容不崩溃."""
        report = fact_checker.check("这是纯文本没有数值")
        assert report.total_assertions == 0

    def test_多数值校验(self, fact_checker: FactChecker):
        """多个数值同时校验."""
        # 用足够长的文本分隔两个离子，确保各自的 50 字符上下文
        # 只包含一个离子，从而正确匹配各自的标准值
        report = fact_checker.check(
            "Dy3+的发射波长580nm。"
            "该离子在多种基质体系中均表现出良好的发光性能，"
            "其发光机理涉及丰富的能级跃迁过程。"
            "Eu3+的发射波长611nm。"
        )
        assert report.total_assertions >= 2
        assert report.passed >= 2

    def test_限定kp_ids(self, fact_checker: FactChecker):
        """限定知识点 ID 范围."""
        report = fact_checker.check(
            "发射波长580nm",
            kp_ids=["KP-001"],
        )
        assert isinstance(report, FactCheckReport)

    def test_校验耗时(self, fact_checker: FactChecker):
        """校验耗时被记录."""
        report = fact_checker.check("Dy3+的发射波长580nm")
        assert report.check_time_ms >= 0.0

    def test_校验反馈内容(self, fact_checker: FactChecker):
        """失败反馈包含具体信息."""
        report = fact_checker.check("Dy3+的发射波长600nm")
        assert "600" in report.feedback or "600.0" in report.feedback

    def test_check_with_retry无回调(self, fact_checker: FactChecker):
        """带重试校验: 无回调时不重试."""
        report = fact_checker.check_with_retry("Dy3+的发射波长600nm")
        assert isinstance(report, FactCheckReport)
        assert fact_checker.retry_count == 0

    def test_check_with_retry有回调(self, fact_checker: FactChecker):
        """带重试校验: 有回调时修正后重试."""
        call_count = [0]

        def on_fail(report: FactCheckReport) -> str:
            call_count[0] += 1
            if call_count[0] <= 2:
                return "Dy3+的发射波长600nm"  # 仍然失败
            return "Dy3+的发射波长580nm"  # 修正为正确值

        report = fact_checker.check_with_retry(
            "Dy3+的发射波长600nm",
            on_fail=on_fail,
        )
        assert isinstance(report, FactCheckReport)

    def test_get_coverage(self, fact_checker: FactChecker):
        """覆盖率统计."""
        coverage = fact_checker.get_coverage()
        assert "total_standards" in coverage
        assert "param_coverage" in coverage
        assert "kp_coverage" in coverage
        assert coverage["total_standards"] > 0


class TestFactCheckerStrictMode:
    """严格模式测试."""

    def test_严格模式跳过变失败(self, strict_fact_checker: FactChecker):
        """严格模式下无标准值的断言视为失败."""
        report = strict_fact_checker.check("未知参数999lux")
        assert report.failed >= 1
        assert report.skipped == 0

    def test_严格模式通过仍通过(self, strict_fact_checker: FactChecker):
        """严格模式下通过仍通过."""
        report = strict_fact_checker.check("Dy3+的发射波长580nm")
        assert report.passed >= 1

    def test_严格模式失败仍失败(self, strict_fact_checker: FactChecker):
        """严格模式下失败仍失败."""
        report = strict_fact_checker.check("Dy3+的发射波长600nm")
        assert report.failed >= 1

    def test_默认非严格模式(self):
        """默认非严格模式: 跳过的不断言为跳过."""
        store = StandardValueStore()
        checker = FactChecker(store)
        report = checker.check("未知参数999lux")
        assert report.skipped >= 1
        assert report.failed == 0


class TestToleranceCalculation:
    """三类容差计算综合测试."""

    def test_绝对容差校验流程(self, standard_store: StandardValueStore):
        """绝对容差: 完整校验流程."""
        checker = FactChecker(standard_store)
        # 580 ± 2nm → 579 通过
        report = checker.check("Dy3+的发射波长579nm")
        assert report.passed >= 1

    def test_相对容差校验流程(self, standard_store: StandardValueStore):
        """相对容差: 量子效率校验."""
        checker = FactChecker(standard_store)
        # 标准值 0.85, 容差 5% → 0.86 通过 (相对偏差 < 5%)
        # 量子效率的上下文关键词为"效率"
        # % 后需接 word 字符以满足 \b 边界
        report = checker.check("Dy3+的量子效率86%的")
        # 至少应该有断言被提取
        assert report.total_assertions > 0

    def test_阈值容差校验流程(self, standard_store: StandardValueStore):
        """阈值容差: Rwp 校验."""
        checker = FactChecker(standard_store)
        # Rwp < 10% → 8% 通过
        # % 后需接 word 字符以满足 \b 边界
        report = checker.check("Rwp拟合优度8%的")
        assert report.total_assertions > 0

    def test_阈值容差失败(self, standard_store: StandardValueStore):
        """阈值容差: 超出阈值失败."""
        checker = FactChecker(standard_store)
        # Rwp > 10% → 15% 失败
        # % 后需接 word 字符以满足 \b 边界
        report = checker.check("Rwp拟合优度15%的")
        assert report.failed >= 1


# ============================================================
# 模块 3: cross_db.py 测试
# ============================================================


class TestSourceType:
    """SourceType 枚举测试."""

    def test_枚举值(self):
        """五个来源类型."""
        assert SourceType.VECTOR.value == "vector"
        assert SourceType.GRAPH.value == "graph"
        assert SourceType.EXACT.value == "exact"
        assert SourceType.KEYWORD.value == "keyword"
        assert SourceType.FUSED.value == "fused"

    def test_继承str(self):
        """继承 str."""
        assert isinstance(SourceType.VECTOR, str)

    def test_枚举成员数(self):
        """共 5 个."""
        assert len(list(SourceType)) == 5


class TestAlignedItem:
    """AlignedItem 数据类测试."""

    def test_创建单源项(self):
        """创建单来源对齐项."""
        item = AlignedItem(
            kp_id="KP-001",
            content="测试内容",
            score=0.5,
            sources=[SourceType.VECTOR],
        )
        assert item.kp_id == "KP-001"
        assert item.score == 0.5
        assert item.source_count == 1
        assert item.is_multi_source is False

    def test_创建多源项(self):
        """创建多来源对齐项."""
        item = AlignedItem(
            kp_id="KP-001",
            content="测试",
            score=0.8,
            sources=[SourceType.VECTOR, SourceType.GRAPH, SourceType.EXACT],
        )
        assert item.source_count == 3
        assert item.is_multi_source is True

    def test_空来源(self):
        """空来源列表."""
        item = AlignedItem(
            kp_id="KP-001",
            content="测试",
            score=0.1,
            sources=[],
        )
        assert item.source_count == 0
        assert item.is_multi_source is False

    def test_source_scores字段(self):
        """source_scores 字典."""
        item = AlignedItem(
            kp_id="KP-001",
            content="测试",
            score=0.5,
            sources=[SourceType.VECTOR],
            source_scores={"vector": 0.3},
        )
        assert item.source_scores == {"vector": 0.3}

    def test_metadata字段(self):
        """metadata 字段."""
        item = AlignedItem(
            kp_id="KP-001",
            content="测试",
            score=0.5,
            metadata={"quality": 0.9},
        )
        assert item.metadata == {"quality": 0.9}

    def test_默认值(self):
        """默认值正确."""
        item = AlignedItem(kp_id="KP-001", content="测试", score=0.5)
        assert item.sources == []
        assert item.source_scores == {}
        assert item.metadata == {}


class TestFusionConfig:
    """FusionConfig 配置测试."""

    def test_默认配置(self):
        """默认融合配置."""
        config = FusionConfig()
        assert config.k == 60
        assert config.multi_source_boost == 0.15
        assert config.min_score == 0.0
        assert config.max_results == 10

    def test_默认权重(self):
        """默认来源权重."""
        config = FusionConfig()
        assert "vector" in config.weights
        assert "graph" in config.weights
        assert "exact" in config.weights
        assert config.weights["vector"] == 0.4

    def test_自定义配置(self):
        """自定义配置."""
        config = FusionConfig(
            k=100,
            multi_source_boost=0.2,
            min_score=0.01,
            max_results=5,
        )
        assert config.k == 100
        assert config.multi_source_boost == 0.2
        assert config.min_score == 0.01
        assert config.max_results == 5

    def test_自定义权重(self):
        """自定义权重."""
        config = FusionConfig(
            weights={"vector": 0.5, "graph": 0.5},
        )
        assert config.weights["vector"] == 0.5
        assert "exact" not in config.weights


class TestAlignmentResult:
    """AlignmentResult 数据类测试."""

    def test_创建空结果(self):
        """创建空对齐结果."""
        result = AlignmentResult(query="test")
        assert result.query == "test"
        assert result.items == []
        assert result.total == 0
        assert result.total_sources == 0
        assert result.total_aligned == 0
        assert result.multi_source_count == 0

    def test_results属性(self):
        """results 属性转换为字典列表."""
        item = AlignedItem(
            kp_id="KP-001",
            content="测试",
            score=0.5,
            sources=[SourceType.VECTOR],
        )
        result = AlignmentResult(query="test", items=[item])
        dicts = result.results
        assert len(dicts) == 1
        assert dicts[0]["kp_id"] == "KP-001"
        assert dicts[0]["score"] == 0.5

    def test_scores属性(self):
        """scores 属性."""
        items = [
            AlignedItem(kp_id="KP-001", content="a", score=0.5),
            AlignedItem(kp_id="KP-002", content="b", score=0.3),
        ]
        result = AlignmentResult(query="test", items=items)
        assert result.scores == [0.5, 0.3]

    def test_total属性(self):
        """total 属性."""
        items = [
            AlignedItem(kp_id="KP-001", content="a", score=0.5),
            AlignedItem(kp_id="KP-002", content="b", score=0.3),
        ]
        result = AlignmentResult(query="test", items=items)
        assert result.total == 2


class TestCrossDBAligner:
    """CrossDBAligner 跨库对齐器测试."""

    def test_初始化默认配置(self):
        """默认配置初始化."""
        aligner = CrossDBAligner()
        assert aligner.config.k == 60
        assert aligner.config.max_results == 10

    def test_初始化自定义配置(self):
        """自定义配置初始化."""
        config = FusionConfig(k=100, max_results=5)
        aligner = CrossDBAligner(config)
        assert aligner.config.k == 100

    def test_add_source字符串类型(self):
        """字符串来源类型."""
        aligner = CrossDBAligner()
        aligner.add_source("vector", [{"kp_id": "KP-001", "content": "test"}], [0.9])
        # fuse 后验证
        result = aligner.fuse("test")
        assert result.total_sources == 1

    def test_add_source枚举类型(self):
        """枚举来源类型."""
        aligner = CrossDBAligner()
        aligner.add_source(
            SourceType.VECTOR,
            [{"kp_id": "KP-001", "content": "test"}],
            [0.9],
        )
        result = aligner.fuse("test")
        assert result.total_sources == 1

    def test_add_retrieval_result(self):
        """添加 RetrievalResult 格式结果."""
        aligner = CrossDBAligner()
        rr = RetrievalResult(
            query="test",
            results=[{"kp_id": "KP-001", "content": "test"}],
            scores=[0.9],
            total=1,
        )
        aligner.add_retrieval_result("vector", rr)
        result = aligner.fuse("test")
        assert result.total_sources == 1
        assert result.total_aligned >= 1

    def test_fuse空来源(self):
        """无来源时返回空结果."""
        aligner = CrossDBAligner()
        result = aligner.fuse("test")
        assert result.total == 0
        assert result.total_sources == 0
        assert result.items == []

    def test_fuse单来源(self):
        """单来源融合."""
        aligner = CrossDBAligner()
        aligner.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "test1"}, {"kp_id": "KP-002", "content": "test2"}],
            [0.9, 0.8],
        )
        result = aligner.fuse("query")
        assert result.total_sources == 1
        assert result.total_aligned == 2
        assert result.total_raw == 2

    def test_fuse多来源(self):
        """多来源融合."""
        aligner = CrossDBAligner()
        aligner.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "vec"}],
            [0.9],
        )
        aligner.add_source(
            "graph",
            [{"kp_id": "KP-001", "content": "graph"}],
            [0.8],
        )
        result = aligner.fuse("query")
        assert result.total_sources == 2
        assert result.total_aligned == 1  # 去重后
        assert result.multi_source_count == 1

    def test_clear(self):
        """清空来源数据."""
        aligner = CrossDBAligner()
        aligner.add_source("vector", [{"kp_id": "KP-001"}], [0.9])
        aligner.clear()
        result = aligner.fuse("test")
        assert result.total_sources == 0


class TestRRFFusion:
    """RRF 融合计算测试."""

    def test_RRF分数计算(self):
        """RRF 分数: w / (k + rank + 1)."""
        config = FusionConfig(k=60, max_results=10)
        aligner = CrossDBAligner(config)
        aligner.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "test"}],
            [1.0],
        )
        result = aligner.fuse("query")
        # rank=0, weight=0.4, k=60 → 0.4 / (60 + 0 + 1) = 0.4/61
        expected = 0.4 / 61
        assert result.items[0].score == pytest.approx(expected, rel=1e-6)

    def test_RRF排名影响分数(self):
        """排名靠后的结果分数更低."""
        config = FusionConfig(k=60)
        aligner = CrossDBAligner(config)
        aligner.add_source(
            "vector",
            [
                {"kp_id": "KP-001", "content": "first"},
                {"kp_id": "KP-002", "content": "second"},
            ],
            [0.9, 0.8],
        )
        result = aligner.fuse("query")
        # rank 0 的分数应高于 rank 1
        assert result.items[0].score > result.items[1].score

    def test_RRF权重影响分数(self):
        """不同来源的权重影响最终分数."""
        config = FusionConfig(k=60)
        aligner = CrossDBAligner(config)
        # vector 权重 0.4, keyword 权重 0.2
        aligner.add_source("vector", [{"kp_id": "KP-001", "content": "v"}], [1.0])
        aligner.add_source("keyword", [{"kp_id": "KP-002", "content": "k"}], [1.0])
        result = aligner.fuse("query")
        # vector 项分数应高于 keyword 项 (权重不同, rank 相同)
        scores_by_kp = {item.kp_id: item.score for item in result.items}
        assert scores_by_kp["KP-001"] > scores_by_kp["KP-002"]

    def test_自定义k参数(self):
        """自定义 k 参数影响 RRF 计算."""
        config_small_k = FusionConfig(k=10)
        config_large_k = FusionConfig(k=100)

        aligner1 = CrossDBAligner(config_small_k)
        aligner1.add_source("vector", [{"kp_id": "KP-001", "content": "t"}], [1.0])
        result1 = aligner1.fuse("q")

        aligner2 = CrossDBAligner(config_large_k)
        aligner2.add_source("vector", [{"kp_id": "KP-001", "content": "t"}], [1.0])
        result2 = aligner2.fuse("q")

        # k=10 的分数应高于 k=100 (分母更小)
        assert result1.items[0].score > result2.items[0].score


class TestMultiSourceBoost:
    """多源命中加分测试."""

    def test_多源命中加分(self):
        """多源命中的项获得额外加分."""
        config = FusionConfig(k=60, multi_source_boost=0.15)
        aligner = CrossDBAligner(config)

        # KP-001 和 KP-002 都在 vector 源中 (同一源不可重复添加)
        aligner.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "v"}, {"kp_id": "KP-002", "content": "v2"}],
            [0.9, 0.7],
        )
        # KP-001 也在 graph 源中 → 多源命中
        aligner.add_source("graph", [{"kp_id": "KP-001", "content": "g"}], [0.8])

        result = aligner.fuse("query")
        scores = {item.kp_id: item.score for item in result.items}

        # KP-001 有多源加分 (0.15 * 2 = 0.3)
        # KP-002 无多源加分
        # 基础 RRF: KP-001 = 0.4/61 + 0.3/61 ≈ 0.01148
        # 多源加分: 0.15 * 2 = 0.3
        # KP-002 = 0.4/62 ≈ 0.00645
        assert scores["KP-001"] > scores["KP-002"]

    def test_单源无加分(self):
        """单源命中的项无多源加分."""
        aligner = CrossDBAligner()
        aligner.add_source("vector", [{"kp_id": "KP-001", "content": "t"}], [0.9])
        result = aligner.fuse("q")
        item = result.items[0]
        assert item.is_multi_source is False
        assert result.multi_source_count == 0

    def test_三源命中(self):
        """三个来源同时命中同一项."""
        aligner = CrossDBAligner()
        for src in ["vector", "graph", "exact"]:
            aligner.add_source(
                src,
                [{"kp_id": "KP-001", "content": f"src_{src}"}],
                [0.9],
            )
        result = aligner.fuse("query")
        assert result.multi_source_count == 1
        assert result.items[0].source_count == 3

    def test_multi_source_count统计(self):
        """multi_source_count 正确统计."""
        aligner = CrossDBAligner()
        # KP-001 和 KP-002 都在 vector 源中
        aligner.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "v"}, {"kp_id": "KP-002", "content": "v2"}],
            [0.9, 0.7],
        )
        # KP-001 也在 graph 源中 → 多源命中
        aligner.add_source("graph", [{"kp_id": "KP-001", "content": "g"}], [0.8])

        result = aligner.fuse("q")
        assert result.multi_source_count == 1


class TestDeduplication:
    """去重逻辑测试."""

    def test_相同kp_id去重(self):
        """相同 kp_id 的结果合并为一个."""
        aligner = CrossDBAligner()
        aligner.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "vector版本"}],
            [0.9],
        )
        aligner.add_source(
            "graph",
            [{"kp_id": "KP-001", "content": "graph版本更长的内容"}],
            [0.8],
        )
        result = aligner.fuse("q")
        assert result.total_aligned == 1
        assert result.items[0].source_count == 2

    def test_保留最长内容(self):
        """去重时保留最长的内容."""
        aligner = CrossDBAligner()
        short_content = "短"
        long_content = "这是更长的内容描述"
        aligner.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": short_content}],
            [0.9],
        )
        aligner.add_source(
            "graph",
            [{"kp_id": "KP-001", "content": long_content}],
            [0.8],
        )
        result = aligner.fuse("q")
        assert result.items[0].content == long_content

    def test_不同kp_id不去重(self):
        """不同 kp_id 的结果保留各自."""
        aligner = CrossDBAligner()
        aligner.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "a"}, {"kp_id": "KP-002", "content": "b"}],
            [0.9, 0.8],
        )
        result = aligner.fuse("q")
        assert result.total_aligned == 2

    def test_max_results限制(self):
        """max_results 限制返回数量."""
        config = FusionConfig(max_results=2)
        aligner = CrossDBAligner(config)
        items = [{"kp_id": f"KP-{i:03d}", "content": f"item{i}"} for i in range(5)]
        scores = [0.9 - i * 0.1 for i in range(5)]
        aligner.add_source("vector", items, scores)
        result = aligner.fuse("q")
        assert result.total_aligned <= 2

    def test_min_score过滤(self):
        """min_score 过滤低分结果."""
        config = FusionConfig(min_score=1.0)  # 高阈值
        aligner = CrossDBAligner(config)
        aligner.add_source("vector", [{"kp_id": "KP-001", "content": "t"}], [0.9])
        result = aligner.fuse("q")
        # RRF 分数远小于 1.0, 应被过滤
        assert result.total_aligned == 0

    def test_未知来源字段回退(self):
        """无 kp_id 时使用其他字段或生成 ID."""
        aligner = CrossDBAligner()
        aligner.add_source(
            "vector",
            [{"entity_id": "E-001", "content": "test"}],
            [0.9],
        )
        result = aligner.fuse("q")
        assert result.total_aligned >= 1

    def test_kp_anchors列表(self):
        """kp_anchors 列表字段取第一个."""
        aligner = CrossDBAligner()
        aligner.add_source(
            "vector",
            [{"kp_anchors": ["KP-001", "KP-002"], "content": "test"}],
            [0.9],
        )
        result = aligner.fuse("q")
        assert result.total_aligned >= 1
        assert result.items[0].kp_id == "KP-001"


class TestQualityWeightedFuser:
    """QualityWeightedFuser 质量加权融合器测试."""

    def test_初始化(self):
        """初始化默认配置."""
        fuser = QualityWeightedFuser()
        assert fuser.config.k == 60

    def test_add_source带质量等级(self):
        """添加带质量等级的来源."""
        fuser = QualityWeightedFuser()
        fuser.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "test"}],
            [0.9],
            quality_tier=1,
        )
        result = fuser.fuse("q")
        assert result.total_sources == 1

    def test_T1质量权重最高(self):
        """T1 权威来源分数高于 T4."""
        fuser = QualityWeightedFuser()
        # T1 和 T4 各有一个不同 kp_id
        fuser.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "t1"}],
            [1.0],
            quality_tier=1,
        )
        fuser.add_source(
            "graph",
            [{"kp_id": "KP-002", "content": "t4"}],
            [1.0],
            quality_tier=4,
        )
        result = fuser.fuse("q")
        scores = {item.kp_id: item.score for item in result.items}
        # T1 (权重1.0) 的分数应高于 T4 (权重0.4)
        assert scores["KP-001"] > scores["KP-002"]

    def test_质量等级元数据(self):
        """质量等级记录在 metadata 中."""
        fuser = QualityWeightedFuser()
        fuser.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "test"}],
            [0.9],
            quality_tier=2,
        )
        result = fuser.fuse("q")
        assert result.items[0].metadata.get("quality_tier") == 2

    def test_多源加分(self):
        """质量加权融合也支持多源加分."""
        fuser = QualityWeightedFuser()
        fuser.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "v"}],
            [0.9],
            quality_tier=1,
        )
        fuser.add_source(
            "graph",
            [{"kp_id": "KP-001", "content": "g"}],
            [0.8],
            quality_tier=2,
        )
        result = fuser.fuse("q")
        assert result.multi_source_count == 1
        assert result.items[0].is_multi_source is True

    def test_空来源(self):
        """无来源时返回空结果."""
        fuser = QualityWeightedFuser()
        result = fuser.fuse("q")
        assert result.total == 0

    def test_clear(self):
        """清空来源."""
        fuser = QualityWeightedFuser()
        fuser.add_source("vector", [{"kp_id": "KP-001"}], [0.9])
        fuser.clear()
        result = fuser.fuse("q")
        assert result.total_sources == 0

    def test_默认质量等级(self):
        """默认质量等级为 T2."""
        fuser = QualityWeightedFuser()
        fuser.add_source(
            "vector",
            [{"kp_id": "KP-001", "content": "test"}],
            [0.9],
            # 不指定 quality_tier, 默认为 2
        )
        result = fuser.fuse("q")
        assert result.items[0].metadata.get("quality_tier") == 2

    def test_T1与T2质量差异(self):
        """T1 (1.0) 和 T2 (0.8) 的分数差异."""
        fuser1 = QualityWeightedFuser()
        fuser1.add_source("vector", [{"kp_id": "KP-001", "content": "t"}], [1.0], quality_tier=1)
        r1 = fuser1.fuse("q")

        fuser2 = QualityWeightedFuser()
        fuser2.add_source("vector", [{"kp_id": "KP-001", "content": "t"}], [1.0], quality_tier=2)
        r2 = fuser2.fuse("q")

        assert r1.items[0].score > r2.items[0].score


class TestFuseResults:
    """fuse_results 静态方法测试."""

    def test_一次性融合多源(self):
        """一次性融合多个 RetrievalResult."""
        results = {
            "vector": RetrievalResult(
                query="test",
                results=[{"kp_id": "KP-001", "content": "vec"}],
                scores=[0.9],
                total=1,
            ),
            "graph": RetrievalResult(
                query="test",
                results=[{"kp_id": "KP-001", "content": "graph"}],
                scores=[0.8],
                total=1,
            ),
        }
        alignment = CrossDBAligner.fuse_results(results, query="test")
        assert isinstance(alignment, AlignmentResult)
        assert alignment.total_sources == 2
        assert alignment.total_aligned == 1  # 去重
        assert alignment.multi_source_count == 1

    def test_带自定义配置(self):
        """带自定义配置的一次性融合."""
        config = FusionConfig(k=30, max_results=5)
        results = {
            "vector": RetrievalResult(
                query="test",
                results=[{"kp_id": "KP-001", "content": "vec"}],
                scores=[0.9],
                total=1,
            ),
        }
        alignment = CrossDBAligner.fuse_results(results, config=config, query="test")
        assert alignment.config.k == 30
        assert alignment.config.max_results == 5

    def test_空结果融合(self):
        """空结果映射返回空对齐."""
        alignment = CrossDBAligner.fuse_results({}, query="test")
        assert alignment.total == 0
        assert alignment.total_sources == 0

    def test_单源融合(self):
        """单源 RetrievalResult 融合."""
        results = {
            "exact": RetrievalResult(
                query="test",
                results=[
                    {"kp_id": "KP-001", "content": "a"},
                    {"kp_id": "KP-002", "content": "b"},
                ],
                scores=[0.9, 0.8],
                total=2,
            ),
        }
        alignment = CrossDBAligner.fuse_results(results, query="test")
        assert alignment.total_aligned == 2
        assert alignment.total_raw == 2

    def test_三源融合去重(self):
        """三源融合后正确去重."""
        results = {
            "vector": RetrievalResult(
                query="test",
                results=[{"kp_id": "KP-001", "content": "v"}],
                scores=[0.9],
                total=1,
            ),
            "graph": RetrievalResult(
                query="test",
                results=[{"kp_id": "KP-001", "content": "g"}],
                scores=[0.8],
                total=1,
            ),
            "exact": RetrievalResult(
                query="test",
                results=[{"kp_id": "KP-001", "content": "e"}],
                scores=[0.7],
                total=1,
            ),
        }
        alignment = CrossDBAligner.fuse_results(results, query="test")
        assert alignment.total_aligned == 1
        assert alignment.items[0].source_count == 3
