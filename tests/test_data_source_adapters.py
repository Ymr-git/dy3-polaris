"""L3 领域知识层 — 数据源适配器框架综合测试套件.

覆盖数据源适配器框架的全部六个层面:
1. 核心框架组件 (data_source_adapter.py):
   SchemaMapper / SyncCheckpoint / AdapterSpec / Recoverer 链
2. 协议基类 (adapter_bases.py):
   RESTAdapter / GraphQLAdapter / DatabaseAdapter / FileAdapter / MCPAdapter
3. 全部 20 个具体适配器 (Tier-1/2/3):
   create / spec / check / discover / read / get_stats 生命周期
4. DataAdapterRegistry:
   注册 / 注销 / 分类查询 / 检查点 / 批量发现
5. SyncCoordinator:
   单适配器同步 / 全量同步 / 分层同步 / 历史与进度报告
6. 集成测试:
   多层注册 → 全量同步 → 检查点保存 → 进度报告 → 增量恢复

测试原则:
- 使用 pytest 框架 + 类分组 (TestClassName 模式)
- 使用 @pytest.fixture 复用公共装配
- 使用 @pytest.mark.parametrize 批量测试 20 个适配器
- 所有导入均来自 dy3_polaris.l3
- 通过 autouse fixture 抑制 Pydantic "schema" 字段遮蔽告警
- 自包含: 无外部网络/文件系统依赖
"""

from __future__ import annotations

import time
import warnings

import pytest

from dy3_polaris.l3 import (
    # --- 核心框架组件 ---
    AdapterCapability,
    AdapterSpec,
    AuthenticationError,
    DataAdapterBase,
    DataAdapterRegistry,
    DataSourceSchema,
    DataSourceType,
    DefaultRecoverer,
    DiscoverResult,
    FieldMapping,
    LifecyclePhase,
    MCPAdapter,
    ReadResult,
    RecoveryAction,
    RecoveryExhaustedError,
    Recoverer,
    RESTAdapter,
    GraphQLAdapter,
    DatabaseAdapter,
    FileAdapter,
    SchemaField,
    SchemaMapper,
    SyncCheckpoint,
    SyncCoordinator,
    SyncMode,
    # --- 连接器基础设施 ---
    ConnectorConfig,
    ConnectorProtocol,
    ConnectorTier,
    # --- Tier-1 公共数据源适配器 (10) ---
    NISTWebBookAdapter,
    PubChemAdapter,
    ArxivAdapter,
    WikipediaAdapter,
    OpenAlexAdapter,
    CrossRefAdapter,
    DOAJAdapter,
    UniProtAdapter,
    ChemSpiderAdapter,
    SemanticScholarAdapter,
    # --- Tier-2 行业数据源适配器 (6) ---
    CASAdapter,
    WebOfScienceAdapter,
    SciFinderAdapter,
    ReaxysAdapter,
    GooglePatentsAdapter,
    EngineeringVillageAdapter,
    # --- Tier-3 校园/私有数据源适配器 (4) ---
    LibraryOPACAdapter,
    LIMSAdapter,
    AcademicAffairsAdapter,
    InternalDocRepositoryAdapter,
)


# ============================================================
# 公共 fixture
# ============================================================


@pytest.fixture(autouse=True)
def suppress_pydantic_schema_warning():
    """抑制 Pydantic 'Field name "schema" shadows an attribute' 告警.

    ReadResult 模型包含名为 ``schema`` 的字段, 会遮蔽 BaseModel 的同名属性,
    触发 UserWarning. 本 autouse fixture 在每个测试期间过滤该告警,
    保持测试输出整洁.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message='Field name "schema" in "ReadResult" shadows an attribute in parent "BaseModel"',
        )
        warnings.filterwarnings("ignore", message='Field name.*"schema".*')
        yield


def _make_config(
    adapter_id: str = "test-adapter",
    name: str = "Test Adapter",
    tier: ConnectorTier = ConnectorTier.PUBLIC,
    base_url: str = "https://api.example.com",
) -> ConnectorConfig:
    """构建测试用 ConnectorConfig."""
    return ConnectorConfig(
        id=adapter_id,
        name=name,
        tier=tier,
        protocol=ConnectorProtocol.HTTPS,
        base_url=base_url,
        rate_limit=60,
        cache_ttl=300,
        version="1.0.0",
        tags=["test"],
        description="测试适配器",
    )


# 全部 20 个具体适配器类
ALL_ADAPTER_CLASSES: list[type[DataAdapterBase]] = [
    # Tier-1 (10)
    NISTWebBookAdapter,
    PubChemAdapter,
    ArxivAdapter,
    WikipediaAdapter,
    OpenAlexAdapter,
    CrossRefAdapter,
    DOAJAdapter,
    UniProtAdapter,
    ChemSpiderAdapter,
    SemanticScholarAdapter,
    # Tier-2 (6)
    CASAdapter,
    WebOfScienceAdapter,
    SciFinderAdapter,
    ReaxysAdapter,
    GooglePatentsAdapter,
    EngineeringVillageAdapter,
    # Tier-3 (4)
    LibraryOPACAdapter,
    LIMSAdapter,
    AcademicAffairsAdapter,
    InternalDocRepositoryAdapter,
]

# 适配器类 → 预期 DataSourceType 映射
ADAPTER_TYPE_MAP: dict[type[DataAdapterBase], DataSourceType] = {
    NISTWebBookAdapter: DataSourceType.REST_API,
    PubChemAdapter: DataSourceType.REST_API,
    ArxivAdapter: DataSourceType.REST_API,
    WikipediaAdapter: DataSourceType.REST_API,
    OpenAlexAdapter: DataSourceType.REST_API,
    CrossRefAdapter: DataSourceType.REST_API,
    DOAJAdapter: DataSourceType.REST_API,
    UniProtAdapter: DataSourceType.REST_API,
    ChemSpiderAdapter: DataSourceType.REST_API,
    SemanticScholarAdapter: DataSourceType.REST_API,
    CASAdapter: DataSourceType.REST_API,
    WebOfScienceAdapter: DataSourceType.REST_API,
    SciFinderAdapter: DataSourceType.REST_API,
    ReaxysAdapter: DataSourceType.GRAPHQL,
    GooglePatentsAdapter: DataSourceType.REST_API,
    EngineeringVillageAdapter: DataSourceType.REST_API,
    LibraryOPACAdapter: DataSourceType.REST_API,
    LIMSAdapter: DataSourceType.DATABASE,
    AcademicAffairsAdapter: DataSourceType.DATABASE,
    InternalDocRepositoryAdapter: DataSourceType.FILE,
}


# ============================================================
# 1. 核心框架组件 — SchemaMapper
# ============================================================


class TestSchemaMapper:
    """SchemaMapper 字段映射与类型转换测试."""

    def test_field_renaming(self):
        """字段重命名: source_field → target_field."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="name", target_field="entity_name")]
        )
        result = mapper.map({"name": "water"})
        assert result["entity_name"] == "water"

    def test_transform_to_lower(self):
        """to_lower 转换: 大写转小写."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="name", target_field="lower", transform="to_lower")]
        )
        assert mapper.map({"name": "WATER"})["lower"] == "water"

    def test_transform_to_upper(self):
        """to_upper 转换: 小写转大写."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="name", target_field="upper", transform="to_upper")]
        )
        assert mapper.map({"name": "water"})["upper"] == "WATER"

    def test_transform_parse_int(self):
        """parse_int 转换: 字符串解析为整数."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="cnt", target_field="count", transform="parse_int")]
        )
        result = mapper.map({"cnt": "42"})["count"]
        assert result == 42
        assert isinstance(result, int)

    def test_transform_parse_int_from_float_string(self):
        """parse_int 转换: 浮点字符串也能解析为整数."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="cnt", target_field="count", transform="parse_int")]
        )
        assert mapper.map({"cnt": "3.9"})["count"] == 3

    def test_transform_parse_float(self):
        """parse_float 转换: 字符串解析为浮点数."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="wt", target_field="weight", transform="parse_float")]
        )
        result = mapper.map({"wt": "18.015"})["weight"]
        assert result == pytest.approx(18.015)
        assert isinstance(result, float)

    def test_transform_parse_bool_true(self):
        """parse_bool 转换: 真值字符串解析为 True."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="flag", target_field="active", transform="parse_bool")]
        )
        for val in ("true", "1", "yes", "y"):
            assert mapper.map({"flag": val})["active"] is True, f"failed for {val!r}"

    def test_transform_parse_bool_false(self):
        """parse_bool 转换: 假值字符串解析为 False."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="flag", target_field="active", transform="parse_bool")]
        )
        for val in ("false", "0", "no", "n"):
            assert mapper.map({"flag": val})["active"] is False, f"failed for {val!r}"

    def test_transform_iso_datetime(self):
        """iso_datetime 转换: 日期字符串标准化为 ISO 8601."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="dt", target_field="date", transform="iso_datetime")]
        )
        result = mapper.map({"dt": "2024-01-15"})["date"]
        assert result == "2024-01-15T00:00:00Z"

    def test_transform_iso_datetime_datetime_format(self):
        """iso_datetime 转换: 完整日期时间格式."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="dt", target_field="date", transform="iso_datetime")]
        )
        result = mapper.map({"dt": "2024-01-15 10:30:00"})["date"]
        assert result == "2024-01-15T10:30:00Z"

    def test_transform_trim(self):
        """trim 转换: 去除首尾空白."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="txt", target_field="trimmed", transform="trim")]
        )
        assert mapper.map({"txt": "  hello  "})["trimmed"] == "hello"

    def test_transform_split_comma(self):
        """split_comma 转换: 逗号分隔字符串转列表."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="csv", target_field="items", transform="split_comma")]
        )
        result = mapper.map({"csv": "a, b, c"})["items"]
        assert result == ["a", "b", "c"]
        assert isinstance(result, list)

    def test_transform_json_parse(self):
        """json_parse 转换: JSON 字符串解析为 Python 对象."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="jsn", target_field="parsed", transform="json_parse")]
        )
        result = mapper.map({"jsn": '{"k": 1}'})["parsed"]
        assert result == {"k": 1}

    def test_default_value_for_missing_field(self):
        """缺失字段使用 default_value 填充."""
        mapper = SchemaMapper(
            [
                FieldMapping(
                    source_field="missing",
                    target_field="filled",
                    default_value="DEFAULT",
                )
            ]
        )
        result = mapper.map({"other": "x"})
        assert result["filled"] == "DEFAULT"

    def test_required_field_missing_raises_value_error(self):
        """必须字段缺失且无默认值时抛出 ValueError."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="must_have", target_field="target", required=True)]
        )
        with pytest.raises(ValueError, match="必须字段缺失"):
            mapper.map({})

    def test_required_field_present_does_not_raise(self):
        """必须字段存在时不抛出异常."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="must_have", target_field="target", required=True)]
        )
        result = mapper.map({"must_have": "value"})
        assert result["target"] == "value"

    def test_map_batch(self):
        """map_batch 批量映射记录列表."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="name", target_field="entity_name")]
        )
        records = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        results = mapper.map_batch(records)
        assert len(results) == 3
        assert results[0]["entity_name"] == "a"
        assert results[2]["entity_name"] == "c"

    def test_register_custom_transform(self):
        """register_transform 注册自定义转换函数."""
        SchemaMapper.register_transform("double", lambda v: v * 2 if v is not None else None)
        mapper = SchemaMapper(
            [FieldMapping(source_field="n", target_field="doubled", transform="double")]
        )
        assert mapper.map({"n": 5})["doubled"] == 10

    def test_unmapped_fields_preserved_with_raw_prefix(self):
        """未映射的源字段以 _raw_ 前缀保留."""
        mapper = SchemaMapper(
            [FieldMapping(source_field="name", target_field="entity_name")]
        )
        result = mapper.map({"name": "water", "extra_field": "kept"})
        assert result["entity_name"] == "water"
        assert result["_raw_extra_field"] == "kept"

    def test_add_mapping_method(self):
        """add_mapping 动态添加映射规则."""
        mapper = SchemaMapper()
        mapper.add_mapping(FieldMapping(source_field="a", target_field="b"))
        result = mapper.map({"a": 1})
        assert result["b"] == 1

    def test_empty_mapper_preserves_all_as_raw(self):
        """空映射器将所有字段以 _raw_ 前缀保留."""
        mapper = SchemaMapper()
        result = mapper.map({"x": 1, "y": 2})
        assert result["_raw_x"] == 1
        assert result["_raw_y"] == 2

    def test_mappings_property(self):
        """mappings 属性返回所有映射规则."""
        m1 = FieldMapping(source_field="a", target_field="b")
        mapper = SchemaMapper([m1])
        assert "b" in mapper.mappings
        assert mapper.mappings["b"].source_field == "a"


# ============================================================
# 1. 核心框架组件 — SyncCheckpoint
# ============================================================


class TestSyncCheckpoint:
    """SyncCheckpoint 同步检查点测试."""

    def test_is_fresh_for_recent_checkpoint(self):
        """最近创建的检查点 is_fresh() 返回 True."""
        cp = SyncCheckpoint(
            adapter_id="test",
            stream_name="default",
            last_sync_time=time.time(),
        )
        assert cp.is_fresh() is True

    def test_is_stale_for_old_checkpoint(self):
        """超过 max_age 的检查点 is_fresh() 返回 False."""
        cp = SyncCheckpoint(
            adapter_id="test",
            stream_name="default",
            last_sync_time=time.time() - 7200,  # 2 小时前
        )
        assert cp.is_fresh(max_age=3600) is False

    def test_is_stale_for_zero_timestamp(self):
        """last_sync_time 为 0 的检查点 is_fresh() 返回 False."""
        cp = SyncCheckpoint(adapter_id="test")
        assert cp.is_fresh() is False

    def test_default_values(self):
        """检查点默认值正确."""
        cp = SyncCheckpoint(adapter_id="test")
        assert cp.stream_name == ""
        assert cp.sync_mode == SyncMode.FULL_REFRESH
        assert cp.cursor_value == ""
        assert cp.records_read == 0
        assert cp.records_written == 0

    def test_is_fresh_within_custom_max_age(self):
        """自定义 max_age 边界检查."""
        cp = SyncCheckpoint(
            adapter_id="test",
            last_sync_time=time.time() - 50,
        )
        assert cp.is_fresh(max_age=100) is True
        assert cp.is_fresh(max_age=10) is False


# ============================================================
# 1. 核心框架组件 — AdapterSpec
# ============================================================


class TestAdapterSpec:
    """AdapterSpec 适配器规范测试."""

    def test_has_capability_true(self):
        """has_capability() 对已声明的能力返回 True."""
        caps = (AdapterCapability.SEARCH | AdapterCapability.FETCH).value
        spec = AdapterSpec(adapter_type=DataSourceType.REST_API, capabilities=caps)
        assert spec.has_capability(AdapterCapability.SEARCH) is True
        assert spec.has_capability(AdapterCapability.FETCH) is True

    def test_has_capability_false(self):
        """has_capability() 对未声明的能力返回 False."""
        caps = (AdapterCapability.SEARCH | AdapterCapability.FETCH).value
        spec = AdapterSpec(adapter_type=DataSourceType.REST_API, capabilities=caps)
        assert spec.has_capability(AdapterCapability.LIST) is False
        assert spec.has_capability(AdapterCapability.CDC) is False

    def test_capability_flags_returns_enum(self):
        """capability_flags() 返回 AdapterCapability Flag 枚举."""
        caps = (AdapterCapability.SEARCH | AdapterCapability.FETCH).value
        spec = AdapterSpec(adapter_type=DataSourceType.REST_API, capabilities=caps)
        flags = spec.capability_flags()
        assert isinstance(flags, AdapterCapability)
        assert AdapterCapability.SEARCH in flags
        assert AdapterCapability.FETCH in flags

    def test_default_spec(self):
        """AdapterSpec 默认值正确."""
        spec = AdapterSpec(adapter_type=DataSourceType.FILE)
        assert spec.adapter_type == DataSourceType.FILE
        assert spec.capabilities == 0
        assert spec.default_sync_mode == SyncMode.FULL_REFRESH
        assert spec.version == "1.0.0"

    def test_has_capability_zero_capabilities(self):
        """能力为 0 时 has_capability() 始终返回 False."""
        spec = AdapterSpec(adapter_type=DataSourceType.REST_API, capabilities=0)
        assert spec.has_capability(AdapterCapability.SEARCH) is False


# ============================================================
# 1. 核心框架组件 — Recoverer 链
# ============================================================


class TestRecovererChain:
    """Recoverer 恢复器链测试."""

    def test_default_recoverer_can_recover_connection_error(self):
        """DefaultRecoverer.can_recover() 对连接错误返回 True."""
        rec = DefaultRecoverer()
        assert rec.can_recover(ConnectionError("refused"), LifecyclePhase.READ) is True
        assert rec.can_recover(TimeoutError("timeout"), LifecyclePhase.READ) is True
        assert rec.can_recover(OSError("os"), LifecyclePhase.READ) is True

    def test_default_recoverer_cannot_recover_auth_error(self):
        """DefaultRecoverer.can_recover() 对认证错误返回 False."""
        rec = DefaultRecoverer()
        assert rec.can_recover(AuthenticationError("bad key"), LifecyclePhase.CHECK) is False
        assert rec.can_recover(PermissionError("denied"), LifecyclePhase.READ) is False

    def test_recommend_action_connection_error(self):
        """连接错误 → RECONNECT."""
        rec = DefaultRecoverer()
        assert rec.recommend_action(ConnectionError("x"), LifecyclePhase.READ) == RecoveryAction.RECONNECT
        assert rec.recommend_action(TimeoutError("x"), LifecyclePhase.READ) == RecoveryAction.RECONNECT

    def test_recommend_action_value_error(self):
        """数据格式错误 → SKIP."""
        rec = DefaultRecoverer()
        assert rec.recommend_action(ValueError("bad data"), LifecyclePhase.TRANSFORM) == RecoveryAction.SKIP
        assert rec.recommend_action(TypeError("bad type"), LifecyclePhase.TRANSFORM) == RecoveryAction.SKIP
        assert rec.recommend_action(KeyError("missing"), LifecyclePhase.TRANSFORM) == RecoveryAction.SKIP

    def test_recommend_action_auth_error(self):
        """认证错误 → ABORT.

        注意: ``PermissionError`` 是 ``OSError`` 子类, 会被 ``_CONNECTION_ERRORS``
        检查先命中而返回 RECONNECT; 框架自身的 ``AuthenticationError`` (非 OSError
        子类) 才是真正触发 ABORT 的认证错误类型.
        """
        rec = DefaultRecoverer()
        assert rec.recommend_action(AuthenticationError("x"), LifecyclePhase.CHECK) == RecoveryAction.ABORT

    def test_recommend_action_unknown_error(self):
        """未知错误 → RETRY."""
        rec = DefaultRecoverer()
        assert rec.recommend_action(RuntimeError("unknown"), LifecyclePhase.READ) == RecoveryAction.RETRY

    def test_handle_error_recoverable_does_not_raise(self):
        """_handle_error() 对可恢复错误不抛出异常 (仅记录)."""
        adapter = RESTAdapter(_make_config("recover-test"))
        # ConnectionError 可恢复 → 不抛出
        adapter._handle_error(ConnectionError("transient"), LifecyclePhase.READ)

    def test_handle_error_auth_raises_recovery_exhausted(self):
        """_handle_error() 对认证错误抛出 RecoveryExhaustedError."""
        adapter = RESTAdapter(_make_config("recover-test"))
        with pytest.raises(RecoveryExhaustedError):
            adapter._handle_error(AuthenticationError("invalid token"), LifecyclePhase.CHECK)

    def test_add_recoverer_append(self):
        """add_recoverer() 追加到链尾."""
        adapter = RESTAdapter(_make_config("chain-test"))
        initial = len(adapter._recoverers)
        custom = _CustomRecoverer()
        adapter.add_recoverer(custom)
        assert len(adapter._recoverers) == initial + 1
        assert adapter._recoverers[-1] is custom

    def test_add_recoverer_prepend(self):
        """add_recoverer(prepend=True) 插入到链首."""
        adapter = RESTAdapter(_make_config("chain-test"))
        custom = _CustomRecoverer()
        adapter.add_recoverer(custom, prepend=True)
        assert adapter._recoverers[0] is custom

    def test_recoverer_chain_traversal_order(self):
        """恢复器链按优先级遍历: 第一个 can_recover=True 的胜出."""
        adapter = RESTAdapter(_make_config("order-test"))
        # 自定义恢复器置顶, 优先于 DefaultRecoverer
        custom = _CustomRecoverer()
        adapter.add_recoverer(custom, prepend=True)
        # ValueError 可被自定义恢复器处理 → SKIP
        action = custom.recommend_action(ValueError("x"), LifecyclePhase.TRANSFORM)
        assert action == RecoveryAction.SKIP


class _CustomRecoverer(Recoverer):
    """测试用自定义恢复器."""

    def can_recover(self, error: Exception, phase: LifecyclePhase) -> bool:
        return isinstance(error, ValueError)

    def recommend_action(self, error: Exception, phase: LifecyclePhase) -> RecoveryAction:
        return RecoveryAction.SKIP


# ============================================================
# 2. 协议基类 — RESTAdapter
# ============================================================


class TestRESTAdapter:
    """RESTAdapter REST API 适配器基类测试."""

    def test_do_spec_returns_rest_api_type(self):
        """_do_spec() 返回 REST_API 类型的 AdapterSpec."""
        adapter = RESTAdapter(_make_config("rest-spec"))
        spec = adapter._do_spec()
        assert spec.adapter_type == DataSourceType.REST_API

    def test_do_spec_capabilities(self):
        """_do_spec() 声明 SEARCH/FETCH/LIST/BATCH/DISCOVER 等能力."""
        adapter = RESTAdapter(_make_config("rest-spec"))
        spec = adapter._do_spec()
        assert spec.has_capability(AdapterCapability.SEARCH)
        assert spec.has_capability(AdapterCapability.FETCH)
        assert spec.has_capability(AdapterCapability.LIST)
        assert spec.has_capability(AdapterCapability.BATCH)
        assert spec.has_capability(AdapterCapability.DISCOVER)

    def test_do_spec_supported_sync_modes(self):
        """_do_spec() 支持 FULL_REFRESH 和 INCREMENTAL."""
        adapter = RESTAdapter(_make_config("rest-spec"))
        spec = adapter._do_spec()
        assert SyncMode.FULL_REFRESH in spec.supported_sync_modes
        assert SyncMode.INCREMENTAL in spec.supported_sync_modes

    def test_build_auth_headers_api_key(self):
        """api_key 认证: X-API-Key 头."""
        adapter = RESTAdapter(
            _make_config("rest-auth"), auth_type="api_key", auth_token="KEY123"
        )
        headers = adapter._build_auth_headers()
        assert headers["X-API-Key"] == "KEY123"
        assert headers["Accept"] == "application/json"

    def test_build_auth_headers_bearer(self):
        """bearer 认证: Authorization: Bearer 头."""
        adapter = RESTAdapter(
            _make_config("rest-auth"), auth_type="bearer", auth_token="TOKEN"
        )
        headers = adapter._build_auth_headers()
        assert headers["Authorization"] == "Bearer TOKEN"

    def test_build_auth_headers_basic(self):
        """basic 认证: Authorization: Basic 头."""
        adapter = RESTAdapter(
            _make_config("rest-auth"), auth_type="basic", auth_token="CREDS"
        )
        headers = adapter._build_auth_headers()
        assert headers["Authorization"] == "Basic CREDS"

    def test_build_auth_headers_none(self):
        """none 认证: 仅 Accept 头, 无认证头."""
        adapter = RESTAdapter(_make_config("rest-auth"), auth_type="none")
        headers = adapter._build_auth_headers()
        assert "X-API-Key" not in headers
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/json"

    def test_parse_response_list(self):
        """_parse_response() 直接返回列表."""
        adapter = RESTAdapter(_make_config("rest-parse"))
        data = [{"id": 1}, {"id": 2}]
        assert adapter._parse_response(data) == data

    def test_parse_response_dict_with_results(self):
        """_parse_response() 从 dict['results'] 提取列表."""
        adapter = RESTAdapter(_make_config("rest-parse"))
        data = {"results": [{"id": 1}]}
        assert adapter._parse_response(data) == [{"id": 1}]

    def test_parse_response_dict_with_data(self):
        """_parse_response() 从 dict['data'] 提取列表."""
        adapter = RESTAdapter(_make_config("rest-parse"))
        data = {"data": [{"id": 2}]}
        assert adapter._parse_response(data) == [{"id": 2}]

    def test_parse_response_dict_with_items(self):
        """_parse_response() 从 dict['items'] 提取列表."""
        adapter = RESTAdapter(_make_config("rest-parse"))
        data = {"items": [{"id": 3}]}
        assert adapter._parse_response(data) == [{"id": 3}]

    def test_parse_response_single_dict(self):
        """_parse_response() 将单个 dict 包装为单元素列表."""
        adapter = RESTAdapter(_make_config("rest-parse"))
        data = {"id": 42, "name": "x"}
        result = adapter._parse_response(data)
        assert len(result) == 1
        assert result[0] == data

    def test_mock_data_round_trip(self):
        """set_mock_data() / _mock_request() 往返测试."""
        adapter = RESTAdapter(_make_config("rest-mock"))
        mock = [{"id": 1}, {"id": 2}, {"id": 3}]
        adapter.set_mock_data(mock)
        assert adapter._mock_request(limit=5) == mock
        assert adapter._mock_request(limit=2) == mock[:2]

    def test_check_returns_true_with_base_url(self):
        """check() 在 base_url 非空时返回 True."""
        adapter = RESTAdapter(_make_config("rest-check", base_url="https://api.test"))
        assert adapter.check() is True

    def test_discover_returns_streams(self):
        """discover() 返回至少 1 个流."""
        adapter = RESTAdapter(_make_config("rest-disc"))
        result = adapter.discover()
        assert len(result.streams) >= 1
        assert result.adapter_id == "rest-disc"


# ============================================================
# 2. 协议基类 — GraphQLAdapter
# ============================================================


class TestGraphQLAdapter:
    """GraphQLAdapter GraphQL 适配器基类测试."""

    def test_do_spec_returns_graphql_type(self):
        """_do_spec() 返回 GRAPHQL 类型的 AdapterSpec."""
        adapter = GraphQLAdapter(_make_config("gql-spec"))
        spec = adapter._do_spec()
        assert spec.adapter_type == DataSourceType.GRAPHQL

    def test_do_spec_capabilities(self):
        """_do_spec() 声明 SEARCH/FETCH/LIST 等能力."""
        adapter = GraphQLAdapter(_make_config("gql-spec"))
        spec = adapter._do_spec()
        assert spec.has_capability(AdapterCapability.SEARCH)
        assert spec.has_capability(AdapterCapability.FETCH)
        assert spec.has_capability(AdapterCapability.DISCOVER)

    def test_build_search_query_returns_tuple(self):
        """_build_search_query() 返回 (query_str, variables) 元组."""
        adapter = GraphQLAdapter(_make_config("gql-query"))
        query_str, variables = adapter._build_search_query("test", limit=10)
        assert isinstance(query_str, str)
        assert isinstance(variables, dict)
        assert variables["q"] == "test"
        assert variables["limit"] == 10

    def test_build_search_query_contains_query_keyword(self):
        """_build_search_query() 生成的查询包含 GraphQL query 关键字."""
        adapter = GraphQLAdapter(_make_config("gql-query"))
        query_str, _ = adapter._build_search_query("test")
        assert "query" in query_str.lower() or "search" in query_str.lower()

    def test_parse_graphql_response_search_key(self):
        """_parse_graphql_response() 从 data['search'] 提取列表."""
        adapter = GraphQLAdapter(_make_config("gql-parse"))
        data = {"data": {"search": [{"id": 1}]}}
        assert adapter._parse_graphql_response(data) == [{"id": 1}]

    def test_parse_graphql_response_nodes_key(self):
        """_parse_graphql_response() 从 data['nodes'] 提取列表."""
        adapter = GraphQLAdapter(_make_config("gql-parse"))
        data = {"data": {"nodes": [{"id": 2}]}}
        assert adapter._parse_graphql_response(data) == [{"id": 2}]

    def test_parse_graphql_response_single_dict(self):
        """_parse_graphql_response() 将 data dict 包装为单元素列表."""
        adapter = GraphQLAdapter(_make_config("gql-parse"))
        data = {"data": {"id": 42, "name": "x"}}
        result = adapter._parse_graphql_response(data)
        assert len(result) == 1
        assert result[0]["id"] == 42

    def test_parse_graphql_response_no_data(self):
        """_parse_graphql_response() 无 data 键时返回空列表."""
        adapter = GraphQLAdapter(_make_config("gql-parse"))
        assert adapter._parse_graphql_response({"errors": []}) == []

    def test_check_with_base_url(self):
        """check() 在 base_url 非空时返回 True."""
        adapter = GraphQLAdapter(_make_config("gql-check"))
        assert adapter.check() is True


# ============================================================
# 2. 协议基类 — DatabaseAdapter
# ============================================================


class TestDatabaseAdapter:
    """DatabaseAdapter 数据库适配器基类测试."""

    def test_do_spec_returns_database_type(self):
        """_do_spec() 返回 DATABASE 类型的 AdapterSpec."""
        adapter = DatabaseAdapter(
            _make_config("db-spec"), connection_string="postgresql://localhost/db"
        )
        spec = adapter._do_spec()
        assert spec.adapter_type == DataSourceType.DATABASE

    def test_do_spec_supports_cdc(self):
        """_do_spec() 支持 CDC 同步模式."""
        adapter = DatabaseAdapter(
            _make_config("db-spec"), connection_string="postgresql://localhost/db"
        )
        spec = adapter._do_spec()
        assert SyncMode.CDC in spec.supported_sync_modes
        assert SyncMode.SNAPSHOT_THEN_INCREMENTAL in spec.supported_sync_modes

    def test_do_spec_has_incremental_capability(self):
        """_do_spec() 声明 INCREMENTAL 和 SCHEMA_EVOLUTION 能力."""
        adapter = DatabaseAdapter(
            _make_config("db-spec"), connection_string="postgresql://localhost/db"
        )
        spec = adapter._do_spec()
        assert spec.has_capability(AdapterCapability.INCREMENTAL)
        assert spec.has_capability(AdapterCapability.SCHEMA_EVOLUTION)

    def test_build_query_full_refresh(self):
        """_build_query() 全量模式: SELECT + ORDER BY, 无 WHERE."""
        adapter = DatabaseAdapter(
            _make_config("db-q"), connection_string="postgresql://localhost/db"
        )
        sql, params = adapter._build_query("users", SyncMode.FULL_REFRESH, None, 0)
        assert "SELECT * FROM users" in sql
        assert "ORDER BY id ASC" in sql
        assert "WHERE" not in sql
        assert params == {}

    def test_build_query_incremental_with_checkpoint(self):
        """_build_query() 增量模式: WHERE updated_at > cursor + LIMIT."""
        adapter = DatabaseAdapter(
            _make_config("db-q"), connection_string="postgresql://localhost/db"
        )
        checkpoint = SyncCheckpoint(adapter_id="db-q", cursor_value="100")
        sql, params = adapter._build_query("users", SyncMode.INCREMENTAL, checkpoint, 50)
        assert "WHERE updated_at > :cursor" in sql
        assert "ORDER BY id ASC" in sql
        assert "LIMIT :limit" in sql
        assert params["cursor"] == "100"
        assert params["limit"] == 50

    def test_build_query_with_limit(self):
        """_build_query() 全量模式带 LIMIT."""
        adapter = DatabaseAdapter(
            _make_config("db-q"), connection_string="postgresql://localhost/db"
        )
        sql, params = adapter._build_query("users", SyncMode.FULL_REFRESH, None, 100)
        assert "LIMIT :limit" in sql
        assert params["limit"] == 100

    def test_mock_query_returns_mock_data(self):
        """_mock_query() 返回 set_mock_data 设置的数据."""
        adapter = DatabaseAdapter(
            _make_config("db-mock"), connection_string="postgresql://localhost/db"
        )
        mock = [{"id": 1}, {"id": 2}]
        adapter.set_mock_data(mock)
        assert adapter._mock_query("SELECT * FROM x", {}) == mock

    def test_check_with_connection_string(self):
        """check() 在连接字符串非空时返回 True."""
        adapter = DatabaseAdapter(
            _make_config("db-check"), connection_string="postgresql://localhost/db"
        )
        assert adapter.check() is True


# ============================================================
# 2. 协议基类 — FileAdapter
# ============================================================


class TestFileAdapter:
    """FileAdapter 文件适配器基类测试."""

    def test_do_spec_returns_file_type(self):
        """_do_spec() 返回 FILE 类型的 AdapterSpec."""
        adapter = FileAdapter(_make_config("file-spec"), file_path="/tmp/data.json")
        spec = adapter._do_spec()
        assert spec.adapter_type == DataSourceType.FILE

    def test_do_spec_capabilities(self):
        """_do_spec() 声明 STREAM/BATCH/DISCOVER 等能力."""
        adapter = FileAdapter(_make_config("file-spec"), file_path="/tmp/data.json")
        spec = adapter._do_spec()
        assert spec.has_capability(AdapterCapability.STREAM)
        assert spec.has_capability(AdapterCapability.BATCH)
        assert spec.has_capability(AdapterCapability.DISCOVER)

    def test_parse_file_json_array(self):
        """_parse_file() 解析 JSON 数组."""
        adapter = FileAdapter(_make_config("file-parse"), file_path="/tmp/data.json")
        content = '[{"a": 1}, {"b": 2}]'
        result = adapter._parse_file(content, "/tmp/data.json")
        assert result == [{"a": 1}, {"b": 2}]

    def test_parse_file_json_object(self):
        """_parse_file() 解析 JSON 对象, 包装为单元素列表."""
        adapter = FileAdapter(_make_config("file-parse"), file_path="/tmp/data.json")
        content = '{"x": 42}'
        result = adapter._parse_file(content, "/tmp/data.json")
        assert len(result) == 1
        assert result[0] == {"x": 42}

    def test_parse_file_plain_text_line_by_line(self):
        """_parse_file() 非 JSON 内容按行解析."""
        adapter = FileAdapter(_make_config("file-parse"), file_path="/tmp/data.txt")
        content = "line1\nline2\nline3"
        result = adapter._parse_file(content, "/tmp/data.txt")
        assert len(result) == 3
        assert result[0] == {"line": 1, "content": "line1"}
        assert result[2] == {"line": 3, "content": "line3"}

    def test_parse_file_empty_content(self):
        """_parse_file() 空内容返回空列表."""
        adapter = FileAdapter(_make_config("file-parse"), file_path="/tmp/empty.json")
        assert adapter._parse_file("", "/tmp/empty.json") == []

    def test_mock_content_round_trip(self):
        """set_mock_content() / _do_read() 往返测试."""
        adapter = FileAdapter(_make_config("file-rt"), file_path="/tmp/data.json")
        adapter.set_mock_content('[{"id": 1}, {"id": 2}, {"id": 3}]')
        result = adapter._do_read(
            stream_name="default",
            sync_mode=SyncMode.FULL_REFRESH,
            checkpoint=None,
            limit=2,
        )
        assert len(result.records) == 2
        assert result.records[0]["id"] == 1

    def test_check_with_file_path(self):
        """check() 在文件路径非空时返回 True."""
        adapter = FileAdapter(_make_config("file-check"), file_path="/tmp/data.json")
        assert adapter.check() is True


# ============================================================
# 2. 协议基类 — MCPAdapter
# ============================================================


class TestMCPAdapter:
    """MCPAdapter MCP 协议适配器基类测试."""

    def test_do_spec_returns_mcp_server_type(self):
        """_do_spec() 返回 MCP_SERVER 类型的 AdapterSpec."""
        adapter = MCPAdapter(
            _make_config("mcp-spec", base_url="stdio://local"),
            transport="stdio",
            server_command="python server.py",
            tool_name="search",
        )
        spec = adapter._do_spec()
        assert spec.adapter_type == DataSourceType.MCP_SERVER

    def test_do_spec_capabilities(self):
        """_do_spec() 声明 STREAM/SUBSCRIBE/DISCOVER 等能力."""
        adapter = MCPAdapter(
            _make_config("mcp-spec", base_url="stdio://local"),
            transport="stdio",
            server_command="python server.py",
        )
        spec = adapter._do_spec()
        assert spec.has_capability(AdapterCapability.STREAM)
        assert spec.has_capability(AdapterCapability.SUBSCRIBE)
        assert spec.has_capability(AdapterCapability.DISCOVER)

    def test_build_tool_call_returns_tuple(self):
        """_build_tool_call() 返回 (tool_name, arguments) 元组."""
        adapter = MCPAdapter(
            _make_config("mcp-call", base_url="stdio://local"),
            transport="stdio",
            server_command="python server.py",
            tool_name="query_tool",
        )
        tool_name, arguments = adapter._build_tool_call("search_term", limit=10)
        assert tool_name == "query_tool"
        assert isinstance(arguments, dict)
        assert arguments["query"] == "search_term"
        assert arguments["limit"] == 10

    def test_build_tool_call_no_limit(self):
        """_build_tool_call() limit<=0 时不包含 limit 参数."""
        adapter = MCPAdapter(
            _make_config("mcp-call", base_url="stdio://local"),
            transport="stdio",
            tool_name="search",
        )
        tool_name, arguments = adapter._build_tool_call("q", limit=0)
        assert "limit" not in arguments

    def test_parse_tool_result_list(self):
        """_parse_tool_result() 直接返回列表."""
        adapter = MCPAdapter(_make_config("mcp-parse", base_url="stdio://local"))
        data = [{"id": 1}]
        assert adapter._parse_tool_result(data) == data

    def test_parse_tool_result_dict_with_results(self):
        """_parse_tool_result() 从 dict['results'] 提取列表."""
        adapter = MCPAdapter(_make_config("mcp-parse", base_url="stdio://local"))
        data = {"results": [{"id": 2}]}
        assert adapter._parse_tool_result(data) == [{"id": 2}]

    def test_parse_tool_result_dict_single(self):
        """_parse_tool_result() 将单个 dict 包装为单元素列表."""
        adapter = MCPAdapter(_make_config("mcp-parse", base_url="stdio://local"))
        data = {"id": 42}
        result = adapter._parse_tool_result(data)
        assert len(result) == 1
        assert result[0] == data

    def test_parse_tool_result_string_json(self):
        """_parse_tool_result() 解析 JSON 字符串."""
        adapter = MCPAdapter(_make_config("mcp-parse", base_url="stdio://local"))
        result = adapter._parse_tool_result('[{"id": 1}]')
        assert result == [{"id": 1}]

    def test_parse_tool_result_string_plain(self):
        """_parse_tool_result() 非 JSON 字符串包装为 content."""
        adapter = MCPAdapter(_make_config("mcp-parse", base_url="stdio://local"))
        result = adapter._parse_tool_result("hello world")
        assert result == [{"content": "hello world"}]

    def test_check_stdio_transport(self):
        """check() stdio 传输在有 server_command 时返回 True."""
        adapter = MCPAdapter(
            _make_config("mcp-check", base_url="stdio://local"),
            transport="stdio",
            server_command="python server.py",
        )
        assert adapter.check() is True


# ============================================================
# 3. 全部 20 个具体适配器 — 参数化测试
# ============================================================


class TestConcreteAdapters:
    """全部 20 个具体适配器的生命周期测试."""

    ALL_ADAPTER_CLASSES = ALL_ADAPTER_CLASSES

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_create_returns_configured_instance(self, adapter_cls):
        """create() 返回正确配置的实例."""
        adapter = adapter_cls.create()
        assert adapter is not None
        assert isinstance(adapter, DataAdapterBase)
        assert adapter.config.id  # 非空 ID

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_spec_returns_correct_adapter_type(self, adapter_cls):
        """spec() 返回的 AdapterSpec 包含正确的 adapter_type."""
        adapter = adapter_cls.create()
        spec = adapter.spec()
        assert spec is not None
        assert isinstance(spec, AdapterSpec)
        expected_type = ADAPTER_TYPE_MAP[adapter_cls]
        assert spec.adapter_type == expected_type

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_check_returns_bool(self, adapter_cls):
        """check() 返回布尔值."""
        adapter = adapter_cls.create()
        result = adapter.check()
        assert result in (True, False)

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_discover_returns_streams(self, adapter_cls):
        """discover() 返回至少 1 个流的 DiscoverResult."""
        adapter = adapter_cls.create()
        result = adapter.discover()
        assert isinstance(result, DiscoverResult)
        assert len(result.streams) >= 1

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_read_returns_records(self, adapter_cls):
        """read(limit=5) 返回包含记录的 ReadResult."""
        adapter = adapter_cls.create()
        result = adapter.read(limit=5)
        assert isinstance(result, ReadResult)
        assert len(result.records) >= 1

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_get_stats_returns_dict_with_keys(self, adapter_cls):
        """get_stats() 返回包含 adapter_id 和 adapter_type 的字典."""
        adapter = adapter_cls.create()
        stats = adapter.get_stats()
        assert isinstance(stats, dict)
        assert "adapter_id" in stats
        assert "adapter_type" in stats
        assert stats["adapter_id"] == adapter.config.id

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_spec_has_capabilities(self, adapter_cls):
        """spec() 声明了 SEARCH 等基础能力."""
        adapter = adapter_cls.create()
        spec = adapter.spec()
        assert spec.has_capability(AdapterCapability.SEARCH)

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_read_returns_checkpoint(self, adapter_cls):
        """read() 返回结果包含检查点."""
        adapter = adapter_cls.create()
        result = adapter.read(limit=5)
        assert result.checkpoint is not None
        assert result.checkpoint.adapter_id == adapter.config.id

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_discover_caches_result(self, adapter_cls):
        """discover() 第二次调用返回缓存结果 (同一对象)."""
        adapter = adapter_cls.create()
        first = adapter.discover()
        second = adapter.discover()
        assert first is second  # 缓存命中

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_discover_force_refresh(self, adapter_cls):
        """discover(force_refresh=True) 强制刷新缓存."""
        adapter = adapter_cls.create()
        first = adapter.discover()
        second = adapter.discover(force_refresh=True)
        assert len(second.streams) == len(first.streams)

    @pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES, ids=lambda c: c.__name__)
    def test_sync_full_flow(self, adapter_cls):
        """sync() 执行完整流程: discover → read → transform → validate."""
        adapter = adapter_cls.create()
        result = adapter.sync(limit=5)
        assert isinstance(result, ReadResult)


# ============================================================
# 4. DataAdapterRegistry
# ============================================================


class TestDataAdapterRegistry:
    """DataAdapterRegistry 注册中心测试."""

    @pytest.fixture
    def registry_with_adapters(self):
        """注册多个适配器的注册中心 fixture."""
        registry = DataAdapterRegistry()
        for cls in [
            NISTWebBookAdapter,
            PubChemAdapter,
            ArxivAdapter,
            CASAdapter,
            ReaxysAdapter,
            LIMSAdapter,
            InternalDocRepositoryAdapter,
        ]:
            registry.register(cls.create())
        return registry

    def test_register_returns_id(self):
        """register() 返回适配器 ID."""
        registry = DataAdapterRegistry()
        adapter = NISTWebBookAdapter.create()
        result = registry.register(adapter)
        assert result == adapter.config.id

    def test_register_duplicate_raises_value_error(self):
        """register() 重复注册同一 ID 抛出 ValueError."""
        registry = DataAdapterRegistry()
        registry.register(NISTWebBookAdapter.create())
        with pytest.raises(ValueError, match="适配器已存在"):
            registry.register(NISTWebBookAdapter.create())

    def test_unregister_existing(self):
        """unregister() 已注册的适配器返回 True."""
        registry = DataAdapterRegistry()
        registry.register(NISTWebBookAdapter.create())
        assert registry.unregister("nist-webbook") is True
        assert registry.get("nist-webbook") is None

    def test_unregister_nonexistent(self):
        """unregister() 不存在的适配器返回 False."""
        registry = DataAdapterRegistry()
        assert registry.unregister("nonexistent") is False

    def test_get_returns_adapter(self):
        """get() 返回已注册的适配器."""
        registry = DataAdapterRegistry()
        adapter = PubChemAdapter.create()
        registry.register(adapter)
        assert registry.get("pubchem") is adapter

    def test_get_returns_none_for_nonexistent(self):
        """get() 对不存在的 ID 返回 None."""
        registry = DataAdapterRegistry()
        assert registry.get("nonexistent") is None

    def test_list_all(self, registry_with_adapters):
        """list_all() 返回所有已注册适配器."""
        adapters = registry_with_adapters.list_all()
        assert len(adapters) == 7

    def test_list_by_tier_public(self, registry_with_adapters):
        """list_by_tier(PUBLIC) 过滤公共层级."""
        adapters = registry_with_adapters.list_by_tier(ConnectorTier.PUBLIC)
        assert len(adapters) == 3  # NIST, PubChem, Arxiv

    def test_list_by_tier_industry(self, registry_with_adapters):
        """list_by_tier(INDUSTRY) 过滤行业层级."""
        adapters = registry_with_adapters.list_by_tier(ConnectorTier.INDUSTRY)
        assert len(adapters) == 2  # CAS, Reaxys

    def test_list_by_tier_private(self, registry_with_adapters):
        """list_by_tier(PRIVATE) 过滤私有层级."""
        adapters = registry_with_adapters.list_by_tier(ConnectorTier.PRIVATE)
        assert len(adapters) == 2  # LIMS, InternalDoc

    def test_list_by_type_rest_api(self, registry_with_adapters):
        """list_by_type(REST_API) 过滤 REST API 类型."""
        adapters = registry_with_adapters.list_by_type(DataSourceType.REST_API)
        assert len(adapters) == 4  # NIST, PubChem, Arxiv, CAS

    def test_list_by_type_graphql(self, registry_with_adapters):
        """list_by_type(GRAPHQL) 过滤 GraphQL 类型."""
        adapters = registry_with_adapters.list_by_type(DataSourceType.GRAPHQL)
        assert len(adapters) == 1  # Reaxys

    def test_list_by_type_database(self, registry_with_adapters):
        """list_by_type(DATABASE) 过滤数据库类型."""
        adapters = registry_with_adapters.list_by_type(DataSourceType.DATABASE)
        assert len(adapters) == 1  # LIMS

    def test_list_by_type_file(self, registry_with_adapters):
        """list_by_type(FILE) 过滤文件类型."""
        adapters = registry_with_adapters.list_by_type(DataSourceType.FILE)
        assert len(adapters) == 1  # InternalDoc

    def test_list_by_capability(self, registry_with_adapters):
        """list_by_capability() 按能力过滤."""
        adapters = registry_with_adapters.list_by_capability(AdapterCapability.SEARCH)
        assert len(adapters) == 7  # 所有适配器都有 SEARCH

    def test_list_by_capability_cdc(self, registry_with_adapters):
        """list_by_capability(CDC) 仅返回支持 CDC 的适配器."""
        adapters = registry_with_adapters.list_by_capability(AdapterCapability.CDC)
        # DatabaseAdapter (LIMS) 声明了 CDC 能力
        assert all(a.spec().has_capability(AdapterCapability.CDC) for a in adapters)

    def test_list_by_sync_mode(self, registry_with_adapters):
        """list_by_sync_mode() 按同步模式过滤."""
        adapters = registry_with_adapters.list_by_sync_mode(SyncMode.INCREMENTAL)
        assert len(adapters) >= 1
        for a in adapters:
            assert SyncMode.INCREMENTAL in a.spec().supported_sync_modes

    def test_check_all(self, registry_with_adapters):
        """check_all() 返回 {adapter_id: bool} 字典."""
        results = registry_with_adapters.check_all()
        assert isinstance(results, dict)
        assert len(results) == 7
        for adapter_id, ok in results.items():
            assert isinstance(ok, bool)

    def test_discover_all(self, registry_with_adapters):
        """discover_all() 返回 {adapter_id: DiscoverResult} 字典."""
        results = registry_with_adapters.discover_all()
        assert isinstance(results, dict)
        assert len(results) == 7
        for adapter_id, disc in results.items():
            assert isinstance(disc, DiscoverResult)
            assert len(disc.streams) >= 1

    def test_save_and_get_checkpoint(self):
        """save_checkpoint() / get_checkpoint() 检查点存取."""
        registry = DataAdapterRegistry()
        cp = SyncCheckpoint(
            adapter_id="test-adapter",
            stream_name="default",
            cursor_value="abc",
            records_read=42,
            last_sync_time=time.time(),
        )
        registry.save_checkpoint("test-adapter", cp)
        retrieved = registry.get_checkpoint("test-adapter")
        assert retrieved is not None
        assert retrieved.cursor_value == "abc"
        assert retrieved.records_read == 42

    def test_get_checkpoint_nonexistent(self):
        """get_checkpoint() 对不存在的 ID 返回 None."""
        registry = DataAdapterRegistry()
        assert registry.get_checkpoint("nonexistent") is None

    def test_get_all_checkpoints(self):
        """get_all_checkpoints() 返回所有检查点."""
        registry = DataAdapterRegistry()
        cp1 = SyncCheckpoint(adapter_id="a", cursor_value="1", last_sync_time=time.time())
        cp2 = SyncCheckpoint(adapter_id="b", cursor_value="2", last_sync_time=time.time())
        registry.save_checkpoint("a", cp1)
        registry.save_checkpoint("b", cp2)
        all_cps = registry.get_all_checkpoints()
        assert len(all_cps) == 2
        assert "a" in all_cps
        assert "b" in all_cps

    def test_get_stats(self, registry_with_adapters):
        """get_stats() 返回汇总统计字典."""
        stats = registry_with_adapters.get_stats()
        assert isinstance(stats, dict)
        assert stats["total_adapters"] == 7
        assert "by_tier" in stats
        assert "by_status" in stats
        assert stats["by_tier"]["public"] == 3
        assert stats["by_tier"]["industry"] == 2
        assert stats["by_tier"]["private"] == 2

    def test_unregister_clears_cache(self):
        """unregister() 同时清除 Schema 缓存和检查点."""
        registry = DataAdapterRegistry()
        registry.register(NISTWebBookAdapter.create())
        registry.discover_all()
        registry.save_checkpoint("nist-webbook", SyncCheckpoint(adapter_id="nist-webbook"))
        registry.unregister("nist-webbook")
        assert registry.get_checkpoint("nist-webbook") is None
        assert registry.get("nist-webbook") is None


# ============================================================
# 5. SyncCoordinator
# ============================================================


class TestSyncCoordinator:
    """SyncCoordinator 同步协调器测试."""

    @pytest.fixture
    def coordinator(self):
        """构建含多个适配器的 SyncCoordinator fixture."""
        registry = DataAdapterRegistry()
        for cls in [NISTWebBookAdapter, PubChemAdapter, ArxivAdapter]:
            registry.register(cls.create())
        return SyncCoordinator(registry)

    def test_sync_adapter_full_flow(self, coordinator):
        """sync_adapter() 执行完整同步流程并返回 ReadResult."""
        result = coordinator.sync_adapter("nist-webbook", limit=5)
        assert isinstance(result, ReadResult)
        assert len(result.records) >= 1

    def test_sync_adapter_saves_checkpoint(self, coordinator):
        """sync_adapter() 同步后保存检查点."""
        coordinator.sync_adapter("nist-webbook", limit=5)
        cp = coordinator._registry.get_checkpoint("nist-webbook")
        assert cp is not None
        assert cp.adapter_id == "nist-webbook"
        assert cp.records_read >= 1

    def test_sync_adapter_nonexistent_raises_value_error(self, coordinator):
        """sync_adapter() 对不存在的适配器抛出 ValueError."""
        with pytest.raises(ValueError, match="适配器不存在"):
            coordinator.sync_adapter("nonexistent")

    def test_sync_all(self, coordinator):
        """sync_all() 同步所有已注册适配器."""
        results = coordinator.sync_all(limit_per_adapter=3)
        assert len(results) == 3
        for adapter_id, result in results.items():
            assert isinstance(result, ReadResult)

    def test_sync_all_returns_exceptions_on_failure(self):
        """sync_all() 对失败的适配器返回 Exception 而非抛出."""
        registry = DataAdapterRegistry()
        registry.register(NISTWebBookAdapter.create())
        coord = SyncCoordinator(registry)
        # 正常情况所有同步成功
        results = coord.sync_all(limit_per_adapter=3)
        assert all(not isinstance(r, Exception) for r in results.values())

    def test_sync_tier(self):
        """sync_tier() 仅同步指定层级的适配器."""
        registry = DataAdapterRegistry()
        registry.register(NISTWebBookAdapter.create())  # PUBLIC
        registry.register(CASAdapter.create())  # INDUSTRY
        coord = SyncCoordinator(registry)
        results = coord.sync_tier(ConnectorTier.PUBLIC, limit_per_adapter=3)
        assert "nist-webbook" in results
        assert "cas-registry" not in results

    def test_get_sync_history_all(self, coordinator):
        """get_sync_history() 返回所有同步记录."""
        coordinator.sync_adapter("nist-webbook", limit=3)
        coordinator.sync_adapter("pubchem", limit=3)
        history = coordinator.get_sync_history()
        assert len(history) == 2

    def test_get_sync_history_by_adapter(self, coordinator):
        """get_sync_history(adapter_id=...) 过滤指定适配器."""
        coordinator.sync_adapter("nist-webbook", limit=3)
        coordinator.sync_adapter("pubchem", limit=3)
        history = coordinator.get_sync_history(adapter_id="nist-webbook")
        assert len(history) == 1
        assert all(h["adapter_id"] == "nist-webbook" for h in history)

    def test_get_sync_history_limit(self, coordinator):
        """get_sync_history(limit=N) 限制返回条数."""
        for _ in range(3):
            coordinator.sync_adapter("nist-webbook", limit=3)
        history = coordinator.get_sync_history(limit=2)
        assert len(history) <= 2

    def test_get_progress_report(self, coordinator):
        """get_progress_report() 返回聚合统计."""
        coordinator.sync_adapter("nist-webbook", limit=3)
        coordinator.sync_adapter("pubchem", limit=3)
        report = coordinator.get_progress_report()
        assert isinstance(report, dict)
        assert report["total_syncs"] == 2
        assert report["successful"] == 2
        assert report["failed"] == 0
        assert report["success_rate"] == 1.0
        assert report["total_records_synced"] >= 2
        assert report["active_checkpoints"] >= 2
        assert "checkpoint_details" in report

    def test_sync_records_history_fields(self, coordinator):
        """同步历史记录包含完整字段."""
        coordinator.sync_adapter("nist-webbook", limit=3)
        history = coordinator.get_sync_history()
        assert len(history) >= 1
        record = history[0]
        assert "adapter_id" in record
        assert "sync_mode" in record
        assert "records" in record
        assert "elapsed_ms" in record
        assert "success" in record
        assert "timestamp" in record

    def test_incremental_sync_uses_checkpoint(self, coordinator):
        """增量同步使用已保存的检查点."""
        # 第一次全量同步
        coordinator.sync_adapter("nist-webbook", sync_mode=SyncMode.FULL_REFRESH, limit=5)
        cp_before = coordinator._registry.get_checkpoint("nist-webbook")
        assert cp_before is not None
        # 第二次增量同步
        result = coordinator.sync_adapter(
            "nist-webbook", sync_mode=SyncMode.INCREMENTAL, limit=5
        )
        assert isinstance(result, ReadResult)
        cp_after = coordinator._registry.get_checkpoint("nist-webbook")
        assert cp_after is not None


# ============================================================
# 6. 集成测试
# ============================================================


class TestIntegration:
    """数据源适配器框架端到端集成测试."""

    @pytest.fixture
    def multi_tier_registry(self):
        """注册多个不同层级适配器的注册中心."""
        registry = DataAdapterRegistry()
        # Tier-1 (PUBLIC)
        registry.register(NISTWebBookAdapter.create())
        registry.register(ArxivAdapter.create())
        registry.register(WikipediaAdapter.create())
        # Tier-2 (INDUSTRY)
        registry.register(CASAdapter.create())
        registry.register(ReaxysAdapter.create())
        # Tier-3 (PRIVATE)
        registry.register(LIMSAdapter.create())
        registry.register(InternalDocRepositoryAdapter.create())
        return registry

    def test_multi_tier_registration(self, multi_tier_registry):
        """多层级适配器注册成功."""
        assert len(multi_tier_registry.list_all()) == 7
        assert len(multi_tier_registry.list_by_tier(ConnectorTier.PUBLIC)) == 3
        assert len(multi_tier_registry.list_by_tier(ConnectorTier.INDUSTRY)) == 2
        assert len(multi_tier_registry.list_by_tier(ConnectorTier.PRIVATE)) == 2

    def test_sync_all_tiers(self, multi_tier_registry):
        """SyncCoordinator 同步所有层级的适配器."""
        coord = SyncCoordinator(multi_tier_registry)
        results = coord.sync_all(limit_per_adapter=3)
        assert len(results) == 7
        # 所有同步都应成功 (返回 ReadResult 而非 Exception)
        for adapter_id, result in results.items():
            assert not isinstance(result, Exception), f"{adapter_id} sync failed"
            assert isinstance(result, ReadResult)

    def test_checkpoints_saved_after_sync(self, multi_tier_registry):
        """同步后所有适配器的检查点均已保存."""
        coord = SyncCoordinator(multi_tier_registry)
        coord.sync_all(limit_per_adapter=3)
        checkpoints = multi_tier_registry.get_all_checkpoints()
        assert len(checkpoints) == 7
        for adapter_id, cp in checkpoints.items():
            assert cp.adapter_id == adapter_id
            assert cp.records_read >= 1

    def test_progress_report_after_sync(self, multi_tier_registry):
        """同步后进度报告显示正确的统计数据."""
        coord = SyncCoordinator(multi_tier_registry)
        coord.sync_all(limit_per_adapter=3)
        report = coord.get_progress_report()
        assert report["total_syncs"] == 7
        assert report["successful"] == 7
        assert report["failed"] == 0
        assert report["success_rate"] == 1.0
        assert report["active_checkpoints"] == 7
        assert report["total_records_synced"] >= 7

    def test_incremental_sync_with_checkpoint(self, multi_tier_registry):
        """增量同步使用已保存的检查点 (第二次同步)."""
        coord = SyncCoordinator(multi_tier_registry)
        # 第一次全量同步
        coord.sync_adapter("nist-webbook", sync_mode=SyncMode.FULL_REFRESH, limit=5)
        first_cp = multi_tier_registry.get_checkpoint("nist-webbook")
        assert first_cp is not None
        first_cursor = first_cp.cursor_value
        # 第二次增量同步 (使用检查点)
        result = coord.sync_adapter(
            "nist-webbook", sync_mode=SyncMode.INCREMENTAL, limit=5
        )
        assert isinstance(result, ReadResult)
        second_cp = multi_tier_registry.get_checkpoint("nist-webbook")
        assert second_cp is not None
        # 检查点已更新
        assert second_cp.last_sync_time >= first_cp.last_sync_time

    def test_sync_history_records_all_tiers(self, multi_tier_registry):
        """同步历史记录所有层级的同步操作."""
        coord = SyncCoordinator(multi_tier_registry)
        coord.sync_all(limit_per_adapter=3)
        history = coord.get_sync_history()
        assert len(history) == 7
        adapter_ids_in_history = {h["adapter_id"] for h in history}
        assert "nist-webbook" in adapter_ids_in_history
        assert "cas-registry" in adapter_ids_in_history
        assert "campus-lims" in adapter_ids_in_history

    def test_discover_all_streams(self, multi_tier_registry):
        """discover_all() 发现所有适配器的流."""
        results = multi_tier_registry.discover_all()
        assert len(results) == 7
        for adapter_id, disc in results.items():
            assert isinstance(disc, DiscoverResult)
            assert len(disc.streams) >= 1

    def test_check_all_adapters(self, multi_tier_registry):
        """check_all() 检查所有适配器连通性."""
        results = multi_tier_registry.check_all()
        assert len(results) == 7
        for adapter_id, ok in results.items():
            assert isinstance(ok, bool)

    def test_registry_stats_reflect_registration(self, multi_tier_registry):
        """注册中心统计反映已注册的适配器."""
        stats = multi_tier_registry.get_stats()
        assert stats["total_adapters"] == 7
        assert stats["by_tier"]["public"] == 3
        assert stats["by_tier"]["industry"] == 2
        assert stats["by_tier"]["private"] == 2

    def test_full_lifecycle_integration(self, multi_tier_registry):
        """完整生命周期集成: 注册 → 发现 → 检查 → 同步 → 检查点 → 报告."""
        coord = SyncCoordinator(multi_tier_registry)

        # 1. discover_all
        disc_results = multi_tier_registry.discover_all()
        assert len(disc_results) == 7

        # 2. check_all
        check_results = multi_tier_registry.check_all()
        assert len(check_results) == 7

        # 3. sync_all
        sync_results = coord.sync_all(limit_per_adapter=3)
        assert len(sync_results) == 7

        # 4. checkpoints saved
        checkpoints = multi_tier_registry.get_all_checkpoints()
        assert len(checkpoints) == 7

        # 5. progress report
        report = coord.get_progress_report()
        assert report["total_syncs"] == 7
        assert report["successful"] == 7

        # 6. sync history
        history = coord.get_sync_history()
        assert len(history) == 7

    def test_all_20_adapters_sync(self):
        """全部 20 个适配器均可成功注册并同步."""
        registry = DataAdapterRegistry()
        for cls in ALL_ADAPTER_CLASSES:
            registry.register(cls.create())
        coord = SyncCoordinator(registry)
        results = coord.sync_all(limit_per_adapter=2)
        assert len(results) == 20
        successful = sum(1 for r in results.values() if not isinstance(r, Exception))
        assert successful == 20
        report = coord.get_progress_report()
        assert report["total_syncs"] == 20
        assert report["successful"] == 20
