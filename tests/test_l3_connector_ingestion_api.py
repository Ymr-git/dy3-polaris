"""L3 领域知识层测试套件 — 连接器、摄入管道、层间接口模型.

覆盖三个模块:
1. connector.py — 知识连接器框架 (枚举/模型/熔断器/抽象基类/注册中心)
2. ingestion.py — 知识摄入管道 (枚举/模型/分块引擎/分类引擎/摄入管道)
3. api_models.py — 层间接口模型 (枚举/模型/适配器函数/学习者画像过滤)
"""

from __future__ import annotations

import logging
import time

logging.disable(logging.CRITICAL)

import pytest
from pydantic import ValidationError

from dy3_polaris.l3 import (
    KnowledgeConnector,
    ConnectorRegistry,
    CircuitBreaker,
    ConnectorTier,
    ConnectorStatus,
    ConnectorProtocol,
    CircuitState,
    ConnectorConfig,
    ConnectorHealth,
    ConnectorResponse,
    IngestionPipeline,
    ChunkingEngine,
    ClassificationEngine,
    KnowledgeDomain,
    KnowledgeLevel,
    ContentType,
    AuthorityTier,
    ChunkMetadata,
    ChunkingConfig,
    ClassificationResult,
    IngestionResult,
    LearningStyle,
    BloomLevel,
    KPMastery,
    LearnerProfile,
    KnowledgeHit,
    FactCheckSummary,
    KnowledgeRetrievalResult,
    ProvenanceEventType,
    ProvenanceEvent,
    ProvenanceMetadata,
    MCPToolDescriptor,
    MCPToolCall,
    MCPToolResult,
    to_knowledge_hit,
    to_retrieval_result,
    to_provenance_event,
    apply_learner_filter,
    KnowledgeStore,
)
from dy3_polaris.l3.exceptions import ChunkingError, IngestError
from dy3_polaris.l3.models import (
    ChunkingStrategy,
    ContentModality,
    DocumentChunk,
    EntityType,
    KnowledgeEntity,
)


# ============================================================
# Mock 连接器实现 (用于测试 KnowledgeConnector 抽象基类)
# ============================================================


class MockConnector(KnowledgeConnector):
    """用于测试的模拟连接器，实现 search/fetch/list_resources."""

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._data: dict[str, str] = {
            "res-001": "钙钛矿太阳能电池效率突破25%",
            "res-002": "有机-无机杂化材料的光电性质研究",
        }
        self._connect_should_fail = False
        self._health_should_fail = False

    def search(self, query: str, **kwargs) -> ConnectorResponse:
        results = [
            {"id": k, "text": v}
            for k, v in self._data.items()
            if query.lower() in v.lower()
        ]
        return ConnectorResponse(
            success=True,
            data=results,
            source=self._config.id,
        )

    def fetch(self, resource_id: str) -> ConnectorResponse:
        if resource_id in self._data:
            return ConnectorResponse(
                success=True,
                data=self._data[resource_id],
                source=self._config.id,
            )
        return ConnectorResponse(
            success=False,
            error=f"资源不存在: {resource_id}",
            source=self._config.id,
        )

    def list_resources(self, **kwargs) -> ConnectorResponse:
        return ConnectorResponse(
            success=True,
            data=list(self._data.keys()),
            source=self._config.id,
        )

    def _do_connect(self) -> bool:
        if self._connect_should_fail:
            return False
        return bool(self._config.base_url)

    def _do_health_check(self) -> bool:
        if self._health_should_fail:
            return False
        return self._is_connected


class FailingConnectConnector(MockConnector):
    """连接总是失败的连接器."""

    def _do_connect(self) -> bool:
        return False


class ExceptionConnector(MockConnector):
    """连接时抛出异常的连接器."""

    def _do_connect(self) -> bool:
        raise ConnectionError("连接被拒绝")


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def connector_config() -> ConnectorConfig:
    """基础连接器配置."""
    return ConnectorConfig(
        id="test-connector",
        name="测试连接器",
        tier=ConnectorTier.PUBLIC,
        protocol=ConnectorProtocol.HTTPS,
        base_url="https://api.example.com",
        rate_limit=60,
        cache_ttl=300.0,
    )


@pytest.fixture
def mock_connector(connector_config: ConnectorConfig) -> MockConnector:
    """模拟连接器实例."""
    return MockConnector(connector_config)


@pytest.fixture
def registry() -> ConnectorRegistry:
    """空注册中心."""
    return ConnectorRegistry()


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """分块配置 (小阈值便于测试)."""
    return ChunkingConfig(
        min_chunk_size=50,
        max_chunk_size=500,
        overlap=0,
        strategy=ChunkingStrategy.SEMANTIC_PARAGRAPH,
    )


@pytest.fixture
def chunking_engine(chunking_config: ChunkingConfig) -> ChunkingEngine:
    """分块引擎."""
    return ChunkingEngine(chunking_config)


@pytest.fixture
def classification_engine() -> ClassificationEngine:
    """分类引擎."""
    return ClassificationEngine()


@pytest.fixture
def knowledge_store() -> KnowledgeStore:
    """知识存储."""
    return KnowledgeStore()


@pytest.fixture
def ingestion_pipeline(
    knowledge_store: KnowledgeStore,
    chunking_engine: ChunkingEngine,
    classification_engine: ClassificationEngine,
) -> IngestionPipeline:
    """摄入管道."""
    return IngestionPipeline(
        store=knowledge_store,
        chunker=chunking_engine,
        classifier=classification_engine,
    )


@pytest.fixture
def sample_text() -> str:
    """用于摄入测试的样本文本 (含领域关键词)."""
    return (
        "钙钛矿太阳能电池是一种基于有机-无机杂化钙钛矿材料的光伏器件。"
        "近年来，钙钛矿太阳能电池的效率从 3.8% 迅速提升至超过 25%，"
        "成为光伏领域的研究热点。钙钛矿材料具有优异的光吸收系数和长载流子扩散长度，"
        "这使得其在太阳能电池应用中表现出色。\n\n"
        "钙钛矿薄膜的制备通常采用旋涂法或蒸镀法。旋涂法通过旋涂仪将前驱体溶液"
        "均匀涂覆在基底上，然后通过退火处理形成钙钛矿晶体。退火温度和退火时间"
        "对薄膜形貌和器件效率有显著影响。掺杂和界面工程也是提高器件性能的重要手段。\n\n"
        "在器件结构方面，钙钛矿太阳能电池通常采用 p-i-n 或 n-i-p 结构。"
        "电极材料的选择、界面层的优化以及异质结的构建都对开路电压和短路电流有重要影响。"
        "I-V 曲线测量是评估器件性能的标准方法，填充因子和 EQE 是关键性能指标。"
    )


@pytest.fixture
def learner_profile() -> LearnerProfile:
    """学习者画像."""
    return LearnerProfile(
        learner_id="learner-001",
        level="beginner",
        kp_mastery={
            "KP-C-001": KPMastery(
                kp_id="KP-C-001",
                mastery_prob=0.3,
                attempts=5,
                correct_count=1,
            ),
        },
        preferred_style=LearningStyle.VISUAL,
        bloom_target=BloomLevel.UNDERSTAND,
        weak_kps=["KP-C-001"],
        interests=["钙钛矿", "太阳能电池"],
    )


# ============================================================
# 1. connector.py — 枚举测试
# ============================================================


class TestConnectorTierEnum:
    """ConnectorTier 枚举测试."""

    def test_枚举值正确(self):
        assert ConnectorTier.PUBLIC.value == "public"
        assert ConnectorTier.INDUSTRY.value == "industry"
        assert ConnectorTier.PRIVATE.value == "private"

    def test_枚举数量为三(self):
        assert len(list(ConnectorTier)) == 3

    def test_枚举是字符串子类(self):
        assert isinstance(ConnectorTier.PUBLIC, str)
        assert ConnectorTier.PUBLIC == "public"


class TestConnectorStatusEnum:
    """ConnectorStatus 枚举测试."""

    def test_枚举值正确(self):
        assert ConnectorStatus.REGISTERED.value == "registered"
        assert ConnectorStatus.RUNNING.value == "running"
        assert ConnectorStatus.OFFLINE.value == "offline"
        assert ConnectorStatus.DEGRADED.value == "degraded"
        assert ConnectorStatus.CIRCUIT_BREAKING.value == "circuit_breaking"

    def test_枚举数量为八(self):
        assert len(list(ConnectorStatus)) == 8


class TestConnectorProtocolEnum:
    """ConnectorProtocol 枚举测试."""

    def test_枚举值正确(self):
        assert ConnectorProtocol.HTTP.value == "http"
        assert ConnectorProtocol.HTTPS.value == "https"
        assert ConnectorProtocol.GRPC.value == "grpc"
        assert ConnectorProtocol.MCP.value == "mcp"
        assert ConnectorProtocol.GRAPHQL.value == "graphql"
        assert ConnectorProtocol.REST.value == "rest"

    def test_枚举数量为六(self):
        assert len(list(ConnectorProtocol)) == 6


class TestCircuitStateEnum:
    """CircuitState 枚举测试."""

    def test_枚举值正确(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_枚举数量为三(self):
        assert len(list(CircuitState)) == 3


# ============================================================
# 2. connector.py — 数据模型测试
# ============================================================


class TestConnectorConfig:
    """ConnectorConfig 模型测试."""

    def test_默认值正确(self):
        config = ConnectorConfig(id="c1", name="连接器1", base_url="https://a.com")
        assert config.tier == ConnectorTier.PUBLIC
        assert config.protocol == ConnectorProtocol.HTTPS
        assert config.rate_limit == 60
        assert config.cache_ttl == 300.0
        assert config.version == "1.0.0"
        assert config.owner == "system"
        assert config.tags == []
        assert config.auth_config == {}
        assert config.metadata == {}

    def test_必填字段缺失抛出验证错误(self):
        with pytest.raises(ValidationError):
            ConnectorConfig(id="", name="test", base_url="https://a.com")
        with pytest.raises(ValidationError):
            ConnectorConfig(id="c1", name="", base_url="https://a.com")
        with pytest.raises(ValidationError):
            ConnectorConfig(id="c1", name="test", base_url="")

    def test_限流和缓存TTL不能为负(self):
        with pytest.raises(ValidationError):
            ConnectorConfig(
                id="c1", name="t", base_url="https://a.com", rate_limit=-1
            )
        with pytest.raises(ValidationError):
            ConnectorConfig(
                id="c1", name="t", base_url="https://a.com", cache_ttl=-1.0
            )

    def test_自定义字段赋值(self):
        config = ConnectorConfig(
            id="nist",
            name="NIST WebBook",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.REST,
            base_url="https://webbook.nist.gov",
            rate_limit=120,
            cache_ttl=600.0,
            tags=["chemistry", "reference"],
            mcp_tool_name="nist_query",
        )
        assert config.tier == ConnectorTier.PUBLIC
        assert config.protocol == ConnectorProtocol.REST
        assert config.rate_limit == 120
        assert config.cache_ttl == 600.0
        assert "chemistry" in config.tags
        assert config.mcp_tool_name == "nist_query"


class TestConnectorHealth:
    """ConnectorHealth 模型测试."""

    def test_默认值正确(self):
        health = ConnectorHealth()
        assert health.status == ConnectorStatus.REGISTERED
        assert health.last_check_time == 0.0
        assert health.response_time_ms == 0.0
        assert health.success_rate == 1.0
        assert health.error_count == 0
        assert health.last_error == ""

    def test_成功率范围约束(self):
        with pytest.raises(ValidationError):
            ConnectorHealth(success_rate=1.5)
        with pytest.raises(ValidationError):
            ConnectorHealth(success_rate=-0.1)

    def test_自定义健康状态(self):
        health = ConnectorHealth(
            status=ConnectorStatus.RUNNING,
            response_time_ms=42.5,
            success_rate=0.95,
            error_count=3,
            last_error="timeout",
        )
        assert health.status == ConnectorStatus.RUNNING
        assert health.response_time_ms == 42.5
        assert health.success_rate == 0.95
        assert health.error_count == 3


class TestConnectorResponse:
    """ConnectorResponse 模型测试."""

    def test_默认值正确(self):
        resp = ConnectorResponse()
        assert resp.success is False
        assert resp.data is None
        assert resp.error == ""
        assert resp.source == ""
        assert resp.latency_ms == 0.0
        assert resp.cached is False

    def test_成功响应构造(self):
        resp = ConnectorResponse(
            success=True,
            data={"key": "value"},
            source="test-connector",
            latency_ms=12.3,
            cached=True,
        )
        assert resp.success is True
        assert resp.data == {"key": "value"}
        assert resp.cached is True
        assert resp.latency_ms == 12.3

    def test_延迟不能为负(self):
        with pytest.raises(ValidationError):
            ConnectorResponse(latency_ms=-1.0)


# ============================================================
# 3. connector.py — CircuitBreaker 熔断器测试
# ============================================================


class TestCircuitBreaker:
    """CircuitBreaker 熔断器测试."""

    def test_初始状态为关闭(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_关闭态放行所有请求(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_连续失败达阈值触发熔断(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_熔断态拒绝所有请求(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_恢复超时后进入半开状态(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # 模拟恢复超时已过
        cb._last_failure_time = time.time() - 61.0
        assert cb.state == CircuitState.HALF_OPEN

    def test_半开态限制探测请求数量(self):
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=60.0, half_open_max_calls=2
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb._last_failure_time = time.time() - 61.0
        # 半开态放行 2 个探测请求
        assert cb.allow_request() is True
        assert cb.allow_request() is True
        # 第 3 个被拒绝
        assert cb.allow_request() is False

    def test_半开态成功恢复为关闭(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        cb._last_failure_time = time.time() - 61.0
        assert cb.state == CircuitState.HALF_OPEN
        cb.allow_request()  # 消耗一个探测名额
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_半开态失败重新熔断(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        cb._last_failure_time = time.time() - 61.0
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb._failure_count == cb.failure_threshold

    def test_重置恢复初始状态(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._half_open_calls == 0
        assert cb._last_failure_time == 0.0

    def test_关闭态成功重置失败计数(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 3
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_获取统计信息(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        cb.record_failure()
        stats = cb.get_stats()
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["failure_threshold"] == 3
        assert stats["recovery_timeout"] == 30.0
        assert "half_open_calls" in stats
        assert "last_failure_time" in stats

    def test_完整状态转换闭环(self):
        """CLOSED → OPEN → HALF_OPEN → CLOSED 完整闭环."""
        cb = CircuitBreaker(
            failure_threshold=2, recovery_timeout=0.1, half_open_max_calls=1
        )
        # CLOSED → OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # OPEN → HALF_OPEN (等待恢复)
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        # HALF_OPEN → CLOSED (探测成功)
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


# ============================================================
# 4. connector.py — KnowledgeConnector 抽象基类测试
# ============================================================


class TestKnowledgeConnector:
    """KnowledgeConnector 抽象基类测试."""

    def test_不能直接实例化抽象类(self):
        with pytest.raises(TypeError):
            KnowledgeConnector(ConnectorConfig(id="x", name="x", base_url="https://a.com"))

    def test_连接成功(self, mock_connector: MockConnector):
        assert mock_connector.is_connected is False
        result = mock_connector.connect()
        assert result is True
        assert mock_connector.is_connected is True
        assert mock_connector.health.status == ConnectorStatus.RUNNING

    def test_连接失败(self, connector_config: ConnectorConfig):
        connector = FailingConnectConnector(connector_config)
        result = connector.connect()
        assert result is False
        assert connector.is_connected is False
        assert connector.health.status == ConnectorStatus.OFFLINE

    def test_连接异常被捕获(self, connector_config: ConnectorConfig):
        connector = ExceptionConnector(connector_config)
        result = connector.connect()
        assert result is False
        assert connector.is_connected is False
        assert connector.health.status == ConnectorStatus.OFFLINE
        assert connector.health.error_count == 1
        assert "连接被拒绝" in connector.health.last_error

    def test_断开连接清理状态(self, mock_connector: MockConnector):
        mock_connector.connect()
        assert mock_connector.is_connected is True
        mock_connector._cache_set("key", "value")
        assert mock_connector._cache_get("key") == "value"
        mock_connector.disconnect()
        assert mock_connector.is_connected is False
        assert mock_connector.health.status == ConnectorStatus.OFFLINE
        # 缓存应被清空
        assert mock_connector._cache_get("key") is None

    def test_健康检查成功(self, mock_connector: MockConnector):
        mock_connector.connect()
        health = mock_connector.health_check()
        assert health.status == ConnectorStatus.RUNNING
        assert health.response_time_ms > 0
        assert health.last_check_time > 0

    def test_健康检查失败返回降级状态(self, mock_connector: MockConnector):
        mock_connector.connect()
        mock_connector._health_should_fail = True
        health = mock_connector.health_check()
        assert health.status == ConnectorStatus.DEGRADED
        assert health.error_count >= 1

    def test_限流检查(self, connector_config: ConnectorConfig):
        connector_config.rate_limit = 3
        connector = MockConnector(connector_config)
        # 前 3 次允许
        for _ in range(3):
            assert connector._check_rate_limit() is True
        # 第 4 次被限流
        assert connector._check_rate_limit() is False

    def test_限流为零表示不限流(self, connector_config: ConnectorConfig):
        connector_config.rate_limit = 0
        connector = MockConnector(connector_config)
        for _ in range(100):
            assert connector._check_rate_limit() is True

    def test_缓存读写(self, mock_connector: MockConnector):
        mock_connector._cache_set("key1", "value1")
        assert mock_connector._cache_get("key1") == "value1"
        assert mock_connector._cache_get("nonexistent") is None

    def test_缓存过期返回None(self, mock_connector: MockConnector):
        mock_connector._cache_set("temp", "data", ttl=0.001)
        time.sleep(0.01)
        assert mock_connector._cache_get("temp") is None

    def test_缓存键生成确定性(self, mock_connector: MockConnector):
        key1 = mock_connector._make_cache_key("search", "query", {"k": 1})
        key2 = mock_connector._make_cache_key("search", "query", {"k": 1})
        key3 = mock_connector._make_cache_key("search", "other", {"k": 1})
        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64  # SHA-256 hex

    def test_执行保护_成功操作(self, mock_connector: MockConnector):
        mock_connector.connect()
        resp = mock_connector._execute_with_protection("search", "钙钛矿")
        assert resp.success is True
        assert resp.cached is False
        assert resp.source == mock_connector.config.id
        assert isinstance(resp.data, list)

    def test_执行保护_缓存命中(self, mock_connector: MockConnector):
        mock_connector.connect()
        # 第一次执行 (未缓存)
        resp1 = mock_connector._execute_with_protection("search", "钙钛矿")
        assert resp1.cached is False
        # 第二次相同参数 (缓存命中)
        resp2 = mock_connector._execute_with_protection("search", "钙钛矿")
        assert resp2.cached is True
        assert resp2.success is True

    def test_执行保护_熔断器开启时拒绝(self, mock_connector: MockConnector):
        mock_connector.connect()
        # 手动触发熔断
        cb = mock_connector._circuit_breaker
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        resp = mock_connector._execute_with_protection("search", "钙钛矿")
        assert resp.success is False
        assert "熔断器" in resp.error

    def test_执行保护_限流时拒绝(self, connector_config: ConnectorConfig):
        connector_config.rate_limit = 1
        connector = MockConnector(connector_config)
        connector.connect()
        # 第一次允许 (消耗限流配额)
        resp1 = connector._execute_with_protection("search", "钙钛矿")
        assert resp1.success is True
        # 第二次被限流 (缓存命中不算限流，所以用不同参数)
        resp2 = connector._execute_with_protection("search", "有机")
        assert resp2.success is False
        assert "限" in resp2.error

    def test_执行保护_未知操作类型(self, mock_connector: MockConnector):
        mock_connector.connect()
        resp = mock_connector._execute_with_protection("unknown_op", "arg")
        assert resp.success is False
        assert "未知操作" in resp.error

    def test_属性访问(self, mock_connector: MockConnector):
        assert mock_connector.config.id == "test-connector"
        assert isinstance(mock_connector.health, ConnectorHealth)
        assert mock_connector.is_connected is False


# ============================================================
# 5. connector.py — ConnectorRegistry 注册中心测试
# ============================================================


class TestConnectorRegistry:
    """ConnectorRegistry 注册中心测试."""

    def test_注册连接器(self, registry: ConnectorRegistry, mock_connector: MockConnector):
        cid = registry.register(mock_connector)
        assert cid == "test-connector"
        assert registry.get("test-connector") is mock_connector

    def test_重复注册抛出异常(self, registry: ConnectorRegistry, mock_connector: MockConnector):
        registry.register(mock_connector)
        with pytest.raises(ValueError, match="已注册"):
            registry.register(mock_connector)

    def test_注销连接器(self, registry: ConnectorRegistry, mock_connector: MockConnector):
        registry.register(mock_connector)
        mock_connector.connect()
        result = registry.unregister("test-connector")
        assert result is True
        assert registry.get("test-connector") is None
        # 注销后连接器应已断开
        assert mock_connector.is_connected is False

    def test_注销不存在的连接器返回False(self, registry: ConnectorRegistry):
        result = registry.unregister("nonexistent")
        assert result is False

    def test_获取不存在的连接器返回None(self, registry: ConnectorRegistry):
        assert registry.get("nonexistent") is None

    def test_按层级查询(self, registry: ConnectorRegistry):
        public_cfg = ConnectorConfig(id="pub", name="pub", base_url="https://a.com", tier=ConnectorTier.PUBLIC)
        private_cfg = ConnectorConfig(id="priv", name="priv", base_url="https://b.com", tier=ConnectorTier.PRIVATE)
        industry_cfg = ConnectorConfig(id="ind", name="ind", base_url="https://c.com", tier=ConnectorTier.INDUSTRY)
        registry.register(MockConnector(public_cfg))
        registry.register(MockConnector(private_cfg))
        registry.register(MockConnector(industry_cfg))

        public_list = registry.list_by_tier(ConnectorTier.PUBLIC)
        assert len(public_list) == 1
        assert public_list[0].config.id == "pub"

        private_list = registry.list_by_tier(ConnectorTier.PRIVATE)
        assert len(private_list) == 1

        assert len(registry.list_by_tier(ConnectorTier.INDUSTRY)) == 1

    def test_列出全部连接器(self, registry: ConnectorRegistry):
        cfg1 = ConnectorConfig(id="c1", name="c1", base_url="https://a.com")
        cfg2 = ConnectorConfig(id="c2", name="c2", base_url="https://b.com")
        registry.register(MockConnector(cfg1))
        registry.register(MockConnector(cfg2))
        all_connectors = registry.list_all()
        assert len(all_connectors) == 2

    def test_获取统计信息(self, registry: ConnectorRegistry):
        cfg = ConnectorConfig(id="c1", name="c1", base_url="https://a.com", tier=ConnectorTier.PUBLIC)
        connector = MockConnector(cfg)
        connector.connect()
        registry.register(connector)
        stats = registry.get_stats()
        assert stats["total"] == 1
        assert "public" in stats["by_tier"]
        assert "by_status" in stats
        assert "by_protocol" in stats
        assert "avg_success_rate" in stats

    def test_全局搜索仅运行中连接器(self, registry: ConnectorRegistry):
        cfg = ConnectorConfig(id="c1", name="c1", base_url="https://a.com")
        connector = MockConnector(cfg)
        connector.connect()
        registry.register(connector)
        results = registry.search_all("钙钛矿")
        assert len(results) == 1
        assert results[0].success is True

    def test_全局搜索跳过离线连接器(self, registry: ConnectorRegistry):
        cfg = ConnectorConfig(id="c1", name="c1", base_url="https://a.com")
        connector = MockConnector(cfg)
        # 不调用 connect，状态为 REGISTERED
        registry.register(connector)
        results = registry.search_all("钙钛矿")
        assert len(results) == 0


# ============================================================
# 6. ingestion.py — 枚举测试
# ============================================================


class TestKnowledgeDomainEnum:
    """KnowledgeDomain 枚举测试."""

    def test_枚举值正确(self):
        assert KnowledgeDomain.PHYSICS.value == "physics"
        assert KnowledgeDomain.CHEMISTRY.value == "chemistry"
        assert KnowledgeDomain.MATERIALS.value == "materials"
        assert KnowledgeDomain.DEVICE.value == "device"
        assert KnowledgeDomain.APPLICATION.value == "application"
        assert KnowledgeDomain.METHODOLOGY.value == "methodology"

    def test_枚举数量为六(self):
        assert len(list(KnowledgeDomain)) == 6


class TestKnowledgeLevelEnum:
    """KnowledgeLevel 枚举测试."""

    def test_枚举值正确(self):
        assert KnowledgeLevel.BASIC.value == "basic"
        assert KnowledgeLevel.INTERMEDIATE.value == "intermediate"
        assert KnowledgeLevel.ADVANCED.value == "advanced"
        assert KnowledgeLevel.TOOL.value == "tool"

    def test_枚举数量为四(self):
        assert len(list(KnowledgeLevel)) == 4


class TestContentTypeEnum:
    """ContentType 枚举测试."""

    def test_枚举值正确(self):
        assert ContentType.LITERATURE.value == "literature"
        assert ContentType.TEXTBOOK.value == "textbook"
        assert ContentType.CONCEPT.value == "concept"
        assert ContentType.EXPERIMENT_DATA.value == "experiment_data"
        assert ContentType.INTERACTION_HISTORY.value == "interaction_history"

    def test_枚举数量为五(self):
        assert len(list(ContentType)) == 5


class TestAuthorityTierEnum:
    """AuthorityTier 枚举测试."""

    def test_枚举值正确(self):
        assert AuthorityTier.T1 == 1
        assert AuthorityTier.T2 == 2
        assert AuthorityTier.T3 == 3
        assert AuthorityTier.T4 == 4

    def test_是整数枚举(self):
        assert isinstance(AuthorityTier.T1, int)

    def test_权威度数值越小越高(self):
        assert AuthorityTier.T1 < AuthorityTier.T2 < AuthorityTier.T3 < AuthorityTier.T4


# ============================================================
# 7. ingestion.py — 数据模型测试
# ============================================================


class TestChunkMetadata:
    """ChunkMetadata 模型测试."""

    def test_默认值正确(self):
        meta = ChunkMetadata()
        assert meta.knowledge_domain == KnowledgeDomain.MATERIALS
        assert meta.knowledge_level == KnowledgeLevel.INTERMEDIATE
        assert meta.content_type == ContentType.CONCEPT
        assert meta.authority_tier == AuthorityTier.T3
        assert meta.kp_anchors == []
        assert meta.material_system == ""

    def test_自定义元数据(self):
        meta = ChunkMetadata(
            knowledge_domain=KnowledgeDomain.PHYSICS,
            material_system="钙钛矿",
            knowledge_level=KnowledgeLevel.ADVANCED,
            content_type=ContentType.LITERATURE,
            authority_tier=AuthorityTier.T1,
            kp_anchors=["KP-C-001", "KP-M-042"],
            key_concepts=["带隙", "载流子"],
        )
        assert meta.knowledge_domain == KnowledgeDomain.PHYSICS
        assert meta.material_system == "钙钛矿"
        assert meta.knowledge_level == KnowledgeLevel.ADVANCED
        assert len(meta.kp_anchors) == 2


class TestChunkingConfig:
    """ChunkingConfig 模型测试."""

    def test_默认值正确(self):
        config = ChunkingConfig()
        assert config.min_chunk_size == 200
        assert config.max_chunk_size == 2000
        assert config.overlap == 100
        assert config.strategy == ChunkingStrategy.SEMANTIC_PARAGRAPH
        assert "L1" in config.level_config
        assert "L2" in config.level_config
        assert "L3" in config.level_config

    def test_最小分块大小约束(self):
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=10)  # < 50
        ChunkingConfig(min_chunk_size=50)  # 边界值合法

    def test_自定义配置(self):
        config = ChunkingConfig(
            min_chunk_size=100,
            max_chunk_size=1000,
            overlap=50,
            strategy=ChunkingStrategy.STRUCTURED_HEADING,
        )
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 1000
        assert config.overlap == 50
        assert config.strategy == ChunkingStrategy.STRUCTURED_HEADING


class TestClassificationResult:
    """ClassificationResult 模型测试."""

    def test_默认值正确(self):
        result = ClassificationResult()
        assert result.domain == KnowledgeDomain.MATERIALS
        assert result.level == KnowledgeLevel.INTERMEDIATE
        assert result.content_type == ContentType.CONCEPT
        assert result.authority_tier == AuthorityTier.T3
        assert result.confidence == 0.5

    def test_置信度范围约束(self):
        with pytest.raises(ValidationError):
            ClassificationResult(confidence=1.5)
        with pytest.raises(ValidationError):
            ClassificationResult(confidence=-0.1)


class TestIngestionResult:
    """IngestionResult 模型测试."""

    def test_默认值正确(self):
        result = IngestionResult()
        assert result.total_chunks == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.skipped == 0
        assert result.errors == []
        assert result.chunk_ids == []

    def test_自定义结果(self):
        result = IngestionResult(
            total_chunks=10,
            successful=8,
            failed=1,
            skipped=1,
            processing_time_ms=123.45,
            errors=["块 3 存储失败"],
            chunk_ids=["c-001", "c-002"],
        )
        assert result.total_chunks == 10
        assert result.successful == 8
        assert result.processing_time_ms == 123.45
        assert len(result.chunk_ids) == 2


# ============================================================
# 8. ingestion.py — ChunkingEngine 分块引擎测试
# ============================================================


class TestChunkingEngine:
    """ChunkingEngine 分块引擎测试."""

    def test_段落分块_多段落(self, chunking_engine: ChunkingEngine):
        text = "第一段内容足够长。\n\n第二段内容也足够长。\n\n第三段内容同样足够长。"
        chunks = chunking_engine.chunk(text, "doc-001")
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        for chunk in chunks:
            assert chunk.document_id == "doc-001"
            assert len(chunk.content) > 0

    def test_章节分块_带标题(self):
        config = ChunkingConfig(
            min_chunk_size=50,
            max_chunk_size=500,
            overlap=0,
            strategy=ChunkingStrategy.STRUCTURED_HEADING,
        )
        engine = ChunkingEngine(config)
        # 每个章节内容超过 max_chunk_size 以触发按标题分块
        section1 = "钙钛矿材料具有优异的光电性质，其带隙约为1.5电子伏特，非常适合太阳能电池应用。" * 10
        section2 = "太阳能电池的效率不断提升，目前钙钛矿器件效率已超过百分之二十五，这是光伏领域的重要突破。" * 10
        text = (
            "# 第一章 钙钛矿\n"
            f"{section1}\n\n"
            "# 第二章 器件\n"
            f"{section2}"
        )
        chunks = engine.chunk(text, "doc-002")
        assert len(chunks) >= 2
        # 第一个块应包含第一章标题
        assert "钙钛矿" in chunks[0].content or "第一章" in chunks[0].content

    def test_句子分块(self):
        config = ChunkingConfig(
            min_chunk_size=50,
            max_chunk_size=500,
            overlap=0,
            strategy=ChunkingStrategy.RECURSIVE_CHAR,
        )
        engine = ChunkingEngine(config)
        text = "这是第一句话，讲述钙钛矿的基本性质。这是第二句话，讨论太阳能电池效率。这是第三句话，总结器件性能。"
        chunks = engine.chunk(text, "doc-003")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_固定长度分块(self):
        config = ChunkingConfig(
            min_chunk_size=50,
            max_chunk_size=500,
            overlap=0,
            strategy=ChunkingStrategy.FIXED_LENGTH,
        )
        engine = ChunkingEngine(config)
        text = "A" * 1100  # 1100 个字符，max=500 → 3 块
        chunks = engine.chunk(text, "doc-004")
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.content) <= 500

    def test_空文本抛出异常(self, chunking_engine: ChunkingEngine):
        with pytest.raises(ChunkingError):
            chunking_engine.chunk("", "doc-empty")
        with pytest.raises(ChunkingError):
            chunking_engine.chunk("   \n  \t  ", "doc-whitespace")

    def test_重叠区域添加(self):
        config = ChunkingConfig(
            min_chunk_size=50,
            max_chunk_size=500,
            overlap=100,
            strategy=ChunkingStrategy.FIXED_LENGTH,
        )
        engine = ChunkingEngine(config)
        text = "A" * 1100  # 1100 chars, max=500 → 2+ blocks
        chunks = engine.chunk(text, "doc-overlap")
        if len(chunks) > 1:
            # 第二块应包含来自第一块末尾的重叠文本
            assert chunks[1].overlap_prev == 100
            assert len(chunks[1].content) > 100  # 重叠 + 自身内容

    def test_切片元数据包含分块层级(self, chunking_engine: ChunkingEngine):
        text = "这是一段足够长的文本用于测试分块引擎的基本功能。"
        chunks = chunking_engine.chunk(text, "doc-level")
        for chunk in chunks:
            assert "chunk_level" in chunk.metadata
            assert chunk.metadata["chunk_level"] in ("L1", "L2", "L3")

    def test_切片内容模态为文本(self, chunking_engine: ChunkingEngine):
        text = "这是一段足够长的文本内容用于测试分块引擎。"
        chunks = chunking_engine.chunk(text, "doc-modality")
        for chunk in chunks:
            assert chunk.content_type == ContentModality.TEXT

    def test_配置属性访问(self, chunking_config: ChunkingConfig):
        engine = ChunkingEngine(chunking_config)
        assert engine.config is chunking_config


# ============================================================
# 9. ingestion.py — ClassificationEngine 分类引擎测试
# ============================================================


class TestClassificationEngine:
    """ClassificationEngine 六维分类引擎测试."""

    def test_知识域分类_物理学(self, classification_engine: ClassificationEngine):
        text = "能带结构和带隙是半导体物理的核心概念。载流子在费米能级附近的态密度决定了导电性。"
        result = classification_engine.classify(text)
        assert result.domain == KnowledgeDomain.PHYSICS

    def test_知识域分类_化学(self, classification_engine: ClassificationEngine):
        text = "通过化学合成方法制备分子，研究化学键和晶体结构。溶剂选择对反应和结晶至关重要。"
        result = classification_engine.classify(text)
        assert result.domain == KnowledgeDomain.CHEMISTRY

    def test_知识域分类_材料(self, classification_engine: ClassificationEngine):
        text = "钙钛矿薄膜的制备采用旋涂和退火工艺，掺杂和缺陷影响形貌和热稳定性。"
        result = classification_engine.classify(text)
        assert result.domain == KnowledgeDomain.MATERIALS

    def test_知识域分类_无匹配返回默认(self, classification_engine: ClassificationEngine):
        text = "今天天气很好，适合出门散步。"
        result = classification_engine.classify(text)
        assert result.domain == KnowledgeDomain.MATERIALS  # 默认域

    def test_材料体系分类_钙钛矿(self, classification_engine: ClassificationEngine):
        text = "钙钛矿材料在太阳能电池中应用广泛。"
        result = classification_engine.classify(text)
        assert "钙钛矿" in result.material_system

    def test_材料体系分类_多体系匹配(self, classification_engine: ClassificationEngine):
        text = "钙钛矿和硅基材料的异质结结构。"
        result = classification_engine.classify(text)
        assert "钙钛矿" in result.material_system
        assert "硅基" in result.material_system

    def test_材料体系分类_无匹配返回空(self, classification_engine: ClassificationEngine):
        text = "普通金属材料的常规性能测试。"
        result = classification_engine.classify(text)
        assert result.material_system == ""

    def test_知识层级分类_工具级(self, classification_engine: ClassificationEngine):
        text = "实验操作步骤如下：首先准备样品，然后按照流程进行测试。"
        result = classification_engine.classify(text)
        assert result.level == KnowledgeLevel.TOOL

    def test_知识层级分类_基础级(self, classification_engine: ClassificationEngine):
        text = "这是一个简单的介绍。"
        result = classification_engine.classify(text)
        assert result.level == KnowledgeLevel.BASIC

    def test_内容类型分类_文献(self, classification_engine: ClassificationEngine):
        text = "Abstract: 本研究报道了新方法。参考文献中引用了 DOI。"
        result = classification_engine.classify(text)
        assert result.content_type == ContentType.LITERATURE

    def test_内容类型分类_概念(self, classification_engine: ClassificationEngine):
        text = "带隙是指价带顶到导带底的能量差。"
        result = classification_engine.classify(text)
        assert result.content_type == ContentType.CONCEPT

    def test_KP锚点提取(self, classification_engine: ClassificationEngine):
        text = "本节涉及知识点 KP-C-001 和 KP-M-042 的内容。KP-C-001 是核心概念。"
        result = classification_engine.classify(text)
        assert "KP-C-001" in result.kp_anchors
        assert "KP-M-042" in result.kp_anchors

    def test_KP锚点去重(self, classification_engine: ClassificationEngine):
        text = "KP-C-001 是重要概念。再次提到 KP-C-001。"
        result = classification_engine.classify(text)
        assert result.kp_anchors.count("KP-C-001") == 1

    def test_权威度评估_顶级期刊(self, classification_engine: ClassificationEngine):
        result = classification_engine.classify("内容", {"journal": "nature"})
        assert result.authority_tier == AuthorityTier.T1

    def test_权威度评估_权威数据库(self, classification_engine: ClassificationEngine):
        result = classification_engine.classify("内容", {"source_type": "nist"})
        assert result.authority_tier == AuthorityTier.T1

    def test_权威度评估_同行评审期刊(self, classification_engine: ClassificationEngine):
        result = classification_engine.classify("内容", {"peer_reviewed": True})
        assert result.authority_tier == AuthorityTier.T2

    def test_权威度评估_用户输入(self, classification_engine: ClassificationEngine):
        result = classification_engine.classify("内容", {"source_type": "user_input"})
        assert result.authority_tier == AuthorityTier.T4

    def test_关键概念提取(self, classification_engine: ClassificationEngine):
        text = "钙钛矿薄膜的带隙和载流子扩散长度是关键参数。"
        result = classification_engine.classify(text)
        assert len(result.key_concepts) > 0
        assert "钙钛矿" in result.key_concepts

    def test_置信度在合理范围(self, classification_engine: ClassificationEngine):
        text = "钙钛矿太阳能电池的带隙和载流子性质是光伏器件效率的关键因素。"
        result = classification_engine.classify(text)
        assert 0.0 <= result.confidence <= 1.0

    def test_完整分类结果结构(self, classification_engine: ClassificationEngine):
        text = "钙钛矿薄膜的制备步骤如下。KP-C-001 是核心概念。"
        result = classification_engine.classify(text, {"source_type": "textbook"})
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.domain, KnowledgeDomain)
        assert isinstance(result.level, KnowledgeLevel)
        assert isinstance(result.content_type, ContentType)
        assert isinstance(result.authority_tier, AuthorityTier)


# ============================================================
# 10. ingestion.py — IngestionPipeline 摄入管道测试
# ============================================================


class TestIngestionPipeline:
    """IngestionPipeline 摄入管道测试."""

    def test_单文档摄入成功(
        self, ingestion_pipeline: IngestionPipeline, sample_text: str
    ):
        result = ingestion_pipeline.ingest(sample_text, "doc-001")
        assert isinstance(result, IngestionResult)
        assert result.total_chunks > 0
        assert result.successful > 0
        assert result.failed == 0
        assert len(result.chunk_ids) == result.successful
        assert result.processing_time_ms > 0

    def test_单文档摄入_带元数据(
        self, ingestion_pipeline: IngestionPipeline, sample_text: str
    ):
        result = ingestion_pipeline.ingest(
            sample_text, "doc-002", {"source_type": "textbook", "journal": "nature"}
        )
        assert result.successful > 0

    def test_批量摄入(
        self, ingestion_pipeline: IngestionPipeline, sample_text: str
    ):
        items = [
            {"content": sample_text, "document_id": "batch-1"},
            {"content": sample_text + " 额外内容用于区分。", "document_id": "batch-2"},
        ]
        result = ingestion_pipeline.ingest_batch(items)
        assert isinstance(result, IngestionResult)
        assert result.successful > 0
        assert result.total_chunks > 0

    def test_去重逻辑_重复内容被跳过(
        self, ingestion_pipeline: IngestionPipeline, sample_text: str
    ):
        # 第一次摄入
        result1 = ingestion_pipeline.ingest(sample_text, "doc-dedup-1")
        assert result1.successful > 0
        # 第二次摄入相同内容 → 全部跳过
        result2 = ingestion_pipeline.ingest(sample_text, "doc-dedup-2")
        assert result2.successful == 0
        assert result2.skipped > 0

    def test_空文本抛出IngestError(self, ingestion_pipeline: IngestionPipeline):
        with pytest.raises(IngestError):
            ingestion_pipeline.ingest("", "doc-empty")

    def test_纯空白文本抛出IngestError(self, ingestion_pipeline: IngestionPipeline):
        with pytest.raises(IngestError):
            ingestion_pipeline.ingest("   \n\t  ", "doc-whitespace")

    def test_摄入结果统计(
        self, ingestion_pipeline: IngestionPipeline, sample_text: str
    ):
        ingestion_pipeline.ingest(sample_text, "doc-stats")
        stats = ingestion_pipeline.get_stats()
        assert stats["total_ingested"] > 0
        assert stats["total_documents"] == 1
        assert stats["unique_hashes"] > 0
        assert stats["store_chunks"] > 0
        assert "store_entities" in stats

    def test_切片验证_过短内容跳过(self, ingestion_pipeline: IngestionPipeline):
        short_chunk = DocumentChunk(document_id="doc", content="短")
        assert ingestion_pipeline._validate_chunk(short_chunk) is False

        empty_chunk = DocumentChunk(document_id="doc", content="   ")
        assert ingestion_pipeline._validate_chunk(empty_chunk) is False

        valid_chunk = DocumentChunk(
            document_id="doc", content="这是一段足够长的有效内容。"
        )
        assert ingestion_pipeline._validate_chunk(valid_chunk) is True

    def test_内容哈希计算(self, ingestion_pipeline: IngestionPipeline):
        hash1 = IngestionPipeline._compute_hash("test content")
        hash2 = IngestionPipeline._compute_hash("test content")
        hash3 = IngestionPipeline._compute_hash("different content")
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex

    def test_批量摄入_空内容文档记为失败(
        self, ingestion_pipeline: IngestionPipeline, sample_text: str
    ):
        items = [
            {"content": sample_text, "document_id": "ok-doc"},
            {"content": "", "document_id": "empty-doc"},
        ]
        result = ingestion_pipeline.ingest_batch(items)
        assert result.successful > 0
        assert result.failed >= 1
        assert any("empty-doc" in err for err in result.errors)

    def test_摄入后存储中有切片(
        self, ingestion_pipeline: IngestionPipeline, sample_text: str
    ):
        result = ingestion_pipeline.ingest(sample_text, "doc-store")
        for chunk_id in result.chunk_ids:
            chunk = ingestion_pipeline._store.get_chunk(chunk_id)
            assert chunk is not None
            assert chunk.document_id == "doc-store"

    def test_摄入后切片含分类元数据(
        self, ingestion_pipeline: IngestionPipeline, sample_text: str
    ):
        result = ingestion_pipeline.ingest(sample_text, "doc-meta")
        if result.chunk_ids:
            chunk = ingestion_pipeline._store.get_chunk(result.chunk_ids[0])
            assert chunk is not None
            assert "classification" in chunk.metadata
            assert "classification_confidence" in chunk.metadata


# ============================================================
# 11. api_models.py — 枚举测试
# ============================================================


class TestLearningStyleEnum:
    """LearningStyle 枚举测试."""

    def test_枚举值正确(self):
        assert LearningStyle.VISUAL.value == "visual"
        assert LearningStyle.AUDITORY.value == "auditory"
        assert LearningStyle.READING.value == "reading"
        assert LearningStyle.KINESTHETIC.value == "kinesthetic"

    def test_枚举数量为四(self):
        assert len(list(LearningStyle)) == 4


class TestBloomLevelEnum:
    """BloomLevel 枚举测试."""

    def test_枚举值正确(self):
        assert BloomLevel.REMEMBER.value == "remember"
        assert BloomLevel.UNDERSTAND.value == "understand"
        assert BloomLevel.APPLY.value == "apply"
        assert BloomLevel.ANALYZE.value == "analyze"
        assert BloomLevel.EVALUATE.value == "evaluate"
        assert BloomLevel.CREATE.value == "create"

    def test_枚举数量为六(self):
        assert len(list(BloomLevel)) == 6


class TestProvenanceEventTypeEnum:
    """ProvenanceEventType 枚举测试."""

    def test_枚举值正确(self):
        assert ProvenanceEventType.QUERY.value == "query"
        assert ProvenanceEventType.INGEST.value == "ingest"
        assert ProvenanceEventType.UPDATE.value == "update"
        assert ProvenanceEventType.FACT_CHECK.value == "fact_check"
        assert ProvenanceEventType.VERSION_RESTORE.value == "version_restore"

    def test_枚举数量为五(self):
        assert len(list(ProvenanceEventType)) == 5


# ============================================================
# 12. api_models.py — L2→L3 接口模型测试
# ============================================================


class TestKPMastery:
    """KPMastery 知识点掌握度测试."""

    def test_默认值和必填字段(self):
        m = KPMastery(kp_id="KP-C-001", mastery_prob=0.5)
        assert m.kp_id == "KP-C-001"
        assert m.mastery_prob == 0.5
        assert m.attempts == 0
        assert m.correct_count == 0
        assert m.last_attempt_time == 0.0

    def test_掌握概率范围约束(self):
        with pytest.raises(ValidationError):
            KPMastery(kp_id="KP-1", mastery_prob=1.5)
        with pytest.raises(ValidationError):
            KPMastery(kp_id="KP-1", mastery_prob=-0.1)

    def test_正确率计算(self):
        m = KPMastery(kp_id="KP-1", mastery_prob=0.5, attempts=10, correct_count=7)
        assert m.accuracy() == 0.7

    def test_零作答正确率为零(self):
        m = KPMastery(kp_id="KP-1", mastery_prob=0.5, attempts=0)
        assert m.accuracy() == 0.0

    def test_薄弱知识点判断(self):
        weak = KPMastery(kp_id="KP-1", mastery_prob=0.3)
        strong = KPMastery(kp_id="KP-2", mastery_prob=0.8)
        assert weak.is_weak() is True
        assert strong.is_weak() is False
        assert weak.is_weak(threshold=0.2) is False


class TestLearnerProfile:
    """LearnerProfile 学习者画像测试."""

    def test_默认值正确(self):
        p = LearnerProfile(learner_id="L001")
        assert p.learner_id == "L001"
        assert p.level == "beginner"
        assert p.kp_mastery == {}
        assert p.preferred_style == LearningStyle.READING
        assert p.bloom_target == BloomLevel.UNDERSTAND
        assert p.weak_kps == []
        assert p.interests == []

    def test_获取知识点掌握状态(self):
        mastery = KPMastery(kp_id="KP-1", mastery_prob=0.5)
        p = LearnerProfile(learner_id="L001", kp_mastery={"KP-1": mastery})
        assert p.get_mastery("KP-1") is mastery
        assert p.get_mastery("KP-2") is None

    def test_判断薄弱知识点(self):
        p = LearnerProfile(learner_id="L001", weak_kps=["KP-1", "KP-2"])
        assert p.is_weak_kp("KP-1") is True
        assert p.is_weak_kp("KP-3") is False

    def test_薄弱知识点数量(self):
        p = LearnerProfile(learner_id="L001", weak_kps=["KP-1", "KP-2", "KP-3"])
        assert p.weak_kp_count() == 3


# ============================================================
# 13. api_models.py — L3→L4 接口模型测试
# ============================================================


class TestKnowledgeHit:
    """KnowledgeHit 知识命中结果测试."""

    def test_默认值和必填字段(self):
        hit = KnowledgeHit(kp_id="KP-1", content="测试内容")
        assert hit.kp_id == "KP-1"
        assert hit.content == "测试内容"
        assert hit.score == 0.0
        assert hit.source == "vector"
        assert hit.is_bottleneck is False
        assert hit.confidence == 0.0

    def test_内容不能为空(self):
        with pytest.raises(ValidationError):
            KnowledgeHit(kp_id="KP-1", content="")

    def test_分数和置信度范围约束(self):
        with pytest.raises(ValidationError):
            KnowledgeHit(kp_id="KP-1", content="x", score=1.5)
        with pytest.raises(ValidationError):
            KnowledgeHit(kp_id="KP-1", content="x", confidence=-0.1)

    def test_高置信度判断(self):
        high = KnowledgeHit(kp_id="KP-1", content="x", confidence=0.9)
        low = KnowledgeHit(kp_id="KP-1", content="x", confidence=0.5)
        assert high.is_high_confidence() is True
        assert low.is_high_confidence() is False
        assert high.is_high_confidence(threshold=0.95) is False


class TestFactCheckSummary:
    """FactCheckSummary 事实校验摘要测试."""

    def test_默认值正确(self):
        s = FactCheckSummary()
        assert s.checked == 0
        assert s.passed == 0
        assert s.failed == 0
        assert s.skipped == 0
        assert s.overall_passed is True
        assert s.failed_items == []

    def test_通过率计算(self):
        s = FactCheckSummary(checked=10, passed=8, failed=2)
        assert s.pass_rate() == 0.8

    def test_零校验通过率为一(self):
        s = FactCheckSummary()
        assert s.pass_rate() == 1.0

    def test_整体通过判断(self):
        passed = FactCheckSummary(checked=5, passed=5, failed=0)
        failed = FactCheckSummary(checked=5, passed=3, failed=2)
        assert passed.overall_passed is True
        assert failed.overall_passed is True  # 默认值，需手动设置


class TestKnowledgeRetrievalResult:
    """KnowledgeRetrievalResult 知识检索结果测试."""

    def test_默认值正确(self):
        r = KnowledgeRetrievalResult(query="测试")
        assert r.query == "测试"
        assert r.intent_type == "concept"
        assert r.hits == []
        assert r.fact_check is None
        assert r.latency_ms == 0.0
        assert r.total == 0
        assert r.query_id.startswith("q-")

    def test_结果为空判断(self):
        empty = KnowledgeRetrievalResult(query="q")
        assert empty.is_empty() is True
        non_empty = KnowledgeRetrievalResult(
            query="q", hits=[KnowledgeHit(kp_id="KP-1", content="x")]
        )
        assert non_empty.is_empty() is False

    def test_获取前K个命中(self):
        hits = [
            KnowledgeHit(kp_id=f"KP-{i}", content=f"c{i}", score=float(i) / 10)
            for i in range(5)
        ]
        r = KnowledgeRetrievalResult(query="q", hits=hits)
        top3 = r.top_k(3)
        assert len(top3) == 3
        # 按分数降序
        assert top3[0].score >= top3[1].score >= top3[2].score

    def test_最高命中分数(self):
        hits = [
            KnowledgeHit(kp_id="KP-1", content="a", score=0.3),
            KnowledgeHit(kp_id="KP-2", content="b", score=0.9),
            KnowledgeHit(kp_id="KP-3", content="c", score=0.5),
        ]
        r = KnowledgeRetrievalResult(query="q", hits=hits)
        assert r.best_score() == 0.9

    def test_无命中时最高分数为零(self):
        r = KnowledgeRetrievalResult(query="q")
        assert r.best_score() == 0.0


# ============================================================
# 14. api_models.py — 溯源接口模型测试
# ============================================================


class TestProvenanceEvent:
    """ProvenanceEvent 溯源事件测试."""

    def test_默认值正确(self):
        e = ProvenanceEvent(event_type=ProvenanceEventType.QUERY)
        assert e.event_type == ProvenanceEventType.QUERY
        assert e.learner_id == "system"
        assert e.query == ""
        assert e.latency_ms == 0.0
        assert e.results == []
        assert e.model_versions == {}
        assert e.timestamp > 0

    def test_系统事件判断(self):
        sys_event = ProvenanceEvent(event_type=ProvenanceEventType.INGEST)
        user_event = ProvenanceEvent(
            event_type=ProvenanceEventType.QUERY, learner_id="user-001"
        )
        assert sys_event.is_system_event() is True
        assert user_event.is_system_event() is False

    def test_自定义事件(self):
        e = ProvenanceEvent(
            event_type=ProvenanceEventType.FACT_CHECK,
            learner_id="user-001",
            query="钙钛矿带隙",
            intent="numeric",
            retrieval_path="vector→rerank→fact_check",
            latency_ms=42.5,
            model_versions={"embedding": "v2", "reranker": "v1"},
        )
        assert e.event_type == ProvenanceEventType.FACT_CHECK
        assert e.learner_id == "user-001"
        assert e.retrieval_path == "vector→rerank→fact_check"
        assert e.model_versions["embedding"] == "v2"


class TestProvenanceMetadata:
    """ProvenanceMetadata 溯源元数据测试."""

    def test_默认值正确(self):
        m = ProvenanceMetadata(source_doc_id="doc-001")
        assert m.source_doc_id == "doc-001"
        assert m.doi is None
        assert m.standard_ref is None
        assert m.textbook_page is None
        assert m.confidence == 0.0
        assert m.fact_checked is False

    def test_有外部参考判断(self):
        with_doi = ProvenanceMetadata(source_doc_id="d1", doi="10.1234/test")
        with_ref = ProvenanceMetadata(source_doc_id="d1", standard_ref="GB/T 1234")
        neither = ProvenanceMetadata(source_doc_id="d1")
        assert with_doi.has_external_ref() is True
        assert with_ref.has_external_ref() is True
        assert neither.has_external_ref() is False

    def test_置信度范围约束(self):
        with pytest.raises(ValidationError):
            ProvenanceMetadata(source_doc_id="d1", confidence=1.5)

    def test_页码不能为负(self):
        with pytest.raises(ValidationError):
            ProvenanceMetadata(source_doc_id="d1", textbook_page=-1)


# ============================================================
# 15. api_models.py — MCP 工具接口模型测试
# ============================================================


class TestMCPToolDescriptor:
    """MCPToolDescriptor MCP 工具描述符测试."""

    def test_默认值和必填字段(self):
        d = MCPToolDescriptor(name="knowledge_search")
        assert d.name == "knowledge_search"
        assert d.description == ""
        assert d.input_schema == {}
        assert d.output_schema == {}
        assert d.tags == []

    def test_名称不能为空(self):
        with pytest.raises(ValidationError):
            MCPToolDescriptor(name="")

    def test_标签判断(self):
        d = MCPToolDescriptor(name="tool1", tags=["L3", "retrieval"])
        assert d.has_tag("L3") is True
        assert d.has_tag("L4") is False


class TestMCPToolCall:
    """MCPToolCall MCP 工具调用请求测试."""

    def test_默认值和必填字段(self):
        c = MCPToolCall(tool_name="search")
        assert c.tool_name == "search"
        assert c.arguments == {}
        assert c.call_id.startswith("call-")
        assert c.timeout_ms == 30000

    def test_工具名不能为空(self):
        with pytest.raises(ValidationError):
            MCPToolCall(tool_name="")

    def test_超时必须为正(self):
        with pytest.raises(ValidationError):
            MCPToolCall(tool_name="t", timeout_ms=0)

    def test_自定义参数(self):
        c = MCPToolCall(
            tool_name="search",
            arguments={"query": "钙钛矿", "top_k": 5},
            call_id="custom-id",
            timeout_ms=5000,
        )
        assert c.arguments["query"] == "钙钛矿"
        assert c.call_id == "custom-id"
        assert c.timeout_ms == 5000


class TestMCPToolResult:
    """MCPToolResult MCP 工具调用结果测试."""

    def test_默认值正确(self):
        r = MCPToolResult(call_id="call-1")
        assert r.call_id == "call-1"
        assert r.success is True
        assert r.result is None
        assert r.error is None
        assert r.latency_ms == 0.0

    def test_成功结果工厂方法(self):
        r = MCPToolResult.ok("call-1", {"hits": 5}, latency_ms=12.3)
        assert r.success is True
        assert r.result == {"hits": 5}
        assert r.error is None
        assert r.latency_ms == 12.3

    def test_失败结果工厂方法(self):
        r = MCPToolResult.fail("call-1", "超时", latency_ms=5000.0)
        assert r.success is False
        assert r.result is None
        assert r.error == "超时"
        assert r.latency_ms == 5000.0

    def test_延迟不能为负(self):
        with pytest.raises(ValidationError):
            MCPToolResult(call_id="c1", latency_ms=-1.0)


# ============================================================
# 16. api_models.py — 适配器函数测试
# ============================================================


class TestToKnowledgeHitAdapter:
    """to_knowledge_hit 适配器函数测试."""

    def test_基本字段映射(self):
        chunk_dict = {
            "kp_id": "KP-C-001",
            "content": "钙钛矿带隙为1.5eV",
            "source_doc_id": "doc-001",
        }
        hit = to_knowledge_hit(chunk_dict, 0.85, "vector")
        assert hit.kp_id == "KP-C-001"
        assert hit.content == "钙钛矿带隙为1.5eV"
        assert hit.score == 0.85
        assert hit.source == "vector"
        assert hit.source_doc_id == "doc-001"

    def test_字段名兼容映射(self):
        chunk_dict = {
            "entity_id": "E-001",
            "text": "内容文本",
            "document_id": "doc-002",
        }
        hit = to_knowledge_hit(chunk_dict, 0.5, "keyword")
        assert hit.kp_id == "E-001"
        assert hit.content == "内容文本"
        assert hit.source_doc_id == "doc-002"

    def test_chunk_id作为后备kp_id(self):
        chunk_dict = {"chunk_id": "c-abc123", "content": "内容"}
        hit = to_knowledge_hit(chunk_dict, 0.3, "graph")
        assert hit.kp_id == "c-abc123"

    def test_provenance来源引用提取(self):
        chunk_dict = {
            "content": "内容",
            "provenance": {"primary_source": "DOI:10.1234/test"},
        }
        hit = to_knowledge_hit(chunk_dict, 0.5, "hybrid")
        assert "DOI:10.1234/test" in hit.source_refs

    def test_quality子维度分数提取(self):
        chunk_dict = {
            "content": "内容",
            "quality": {
                "accuracy": 0.9,
                "trustworthiness": 0.8,
                "overall": 0.85,
            },
        }
        hit = to_knowledge_hit(chunk_dict, 0.7, "vector")
        assert hit.sub_scores["accuracy"] == 0.9
        assert hit.sub_scores["trustworthiness"] == 0.8
        assert hit.confidence == 0.85  # 回退到 quality.overall

    def test_置信度回退到score(self):
        chunk_dict = {"content": "内容"}
        hit = to_knowledge_hit(chunk_dict, 0.6, "vector")
        assert hit.confidence == 0.6

    def test_分数截断到合法范围(self):
        chunk_dict = {"content": "内容"}
        hit = to_knowledge_hit(chunk_dict, 1.5, "vector")
        assert hit.score == 1.0
        hit_neg = to_knowledge_hit(chunk_dict, -0.5, "vector")
        assert hit_neg.score == 0.0

    def test_瓶颈标记提取(self):
        chunk_dict = {"content": "内容", "is_bottleneck": True}
        hit = to_knowledge_hit(chunk_dict, 0.5, "vector")
        assert hit.is_bottleneck is True

    def test_显式sub_scores优先于quality(self):
        chunk_dict = {
            "content": "内容",
            "sub_scores": {"visual": 0.8, "difficulty": 0.3},
            "quality": {"accuracy": 0.9},
        }
        hit = to_knowledge_hit(chunk_dict, 0.5, "vector")
        assert hit.sub_scores == {"visual": 0.8, "difficulty": 0.3}

    def test_空内容抛出验证错误(self):
        with pytest.raises(ValidationError):
            to_knowledge_hit({}, 0.5, "vector")


class TestToRetrievalResultAdapter:
    """to_retrieval_result 适配器函数测试."""

    def test_基本转换(self):
        results = [
            {"kp_id": "KP-1", "content": "内容1"},
            {"kp_id": "KP-2", "content": "内容2"},
        ]
        scores = [0.9, 0.7]
        r = to_retrieval_result("查询", results, scores, "concept", 15.5)
        assert r.query == "查询"
        assert r.intent_type == "concept"
        assert r.latency_ms == 15.5
        assert len(r.hits) == 2
        assert r.hits[0].score == 0.9
        assert r.hits[1].score == 0.7
        assert r.total == 2

    def test_分数列表不足时补零(self):
        results = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        scores = [0.5]  # 只有1个分数
        r = to_retrieval_result("q", results, scores, "concept", 10.0)
        assert len(r.hits) == 3
        assert r.hits[0].score == 0.5
        assert r.hits[1].score == 0.0
        assert r.hits[2].score == 0.0

    def test_分数列表超出时截断(self):
        results = [{"content": "a"}]
        scores = [0.5, 0.3, 0.1]
        r = to_retrieval_result("q", results, scores, "concept", 10.0)
        assert len(r.hits) == 1
        assert r.hits[0].score == 0.5

    def test_来源类型推断_concept(self):
        r = to_retrieval_result("q", [{"content": "x"}], [0.5], "concept", 1.0)
        assert r.hits[0].source == "hybrid"

    def test_来源类型推断_numeric(self):
        r = to_retrieval_result("q", [{"content": "x"}], [0.5], "numeric", 1.0)
        assert r.hits[0].source == "keyword"

    def test_来源类型推断_relational(self):
        r = to_retrieval_result("q", [{"content": "x"}], [0.5], "relational", 1.0)
        assert r.hits[0].source == "graph"

    def test_显式source_type优先(self):
        r = to_retrieval_result(
            "q", [{"content": "x", "source_type": "custom"}], [0.5], "concept", 1.0
        )
        assert r.hits[0].source == "custom"


class TestToProvenanceEventAdapter:
    """to_provenance_event 适配器函数测试."""

    def test_基本转换(self):
        e = to_provenance_event("query", "钙钛矿带隙", [], 15.0)
        assert e.event_type == ProvenanceEventType.QUERY
        assert e.query == "钙钛矿带隙"
        assert e.latency_ms == 15.0
        assert e.learner_id == "system"
        assert e.is_system_event() is True

    def test_自定义kwargs覆盖(self):
        e = to_provenance_event(
            "ingest",
            "",
            [{"chunk_id": "c-1"}],
            30.0,
            learner_id="user-001",
            intent="concept",
            retrieval_path="vector→rerank",
            model_versions={"embedding": "v2"},
        )
        assert e.event_type == ProvenanceEventType.INGEST
        assert e.learner_id == "user-001"
        assert e.intent == "concept"
        assert e.retrieval_path == "vector→rerank"
        assert e.model_versions == {"embedding": "v2"}
        assert len(e.results) == 1

    def test_自定义时间戳(self):
        custom_ts = 1000000.0
        e = to_provenance_event("query", "q", [], 1.0, timestamp=custom_ts)
        assert e.timestamp == custom_ts

    def test_无效事件类型抛出异常(self):
        with pytest.raises(ValueError, match="无效的事件类型"):
            to_provenance_event("invalid_type", "q", [], 1.0)

    def test_所有合法事件类型可转换(self):
        for et in ProvenanceEventType:
            e = to_provenance_event(et.value, "q", [], 1.0)
            assert e.event_type == et


# ============================================================
# 17. api_models.py — apply_learner_filter 测试
# ============================================================


class TestApplyLearnerFilter:
    """apply_learner_filter 学习者画像过滤测试."""

    def test_空结果返回空列表(self, learner_profile: LearnerProfile):
        result = apply_learner_filter(learner_profile, [])
        assert result == []

    def test_弱项KP加权提升(self):
        """弱项 KP 的命中分数提升 1.5 倍."""
        profile = LearnerProfile(
            learner_id="L1",
            level="intermediate",
            preferred_style=LearningStyle.READING,
            bloom_target=BloomLevel.CREATE,  # 高目标，不受难度惩罚
            weak_kps=["KP-weak"],
        )
        hits = [
            KnowledgeHit(kp_id="KP-weak", content="弱项内容", score=0.4),
            KnowledgeHit(kp_id="KP-strong", content="强项内容", score=0.4),
        ]
        result = apply_learner_filter(profile, hits)
        # 弱项得分 = 0.4 * 1.5 = 0.6 (READING 无匹配 sub_scores → *0.9 → 0.54)
        # 强项得分 = 0.4 * 0.9 = 0.36
        weak_hit = next(h for h in result if h.kp_id == "KP-weak")
        strong_hit = next(h for h in result if h.kp_id == "KP-strong")
        assert weak_hit.score > strong_hit.score

    def test_学习风格匹配加权(self):
        """匹配学习风格的命中分数提升."""
        profile = LearnerProfile(
            learner_id="L1",
            level="intermediate",
            preferred_style=LearningStyle.VISUAL,
            bloom_target=BloomLevel.CREATE,
        )
        hits = [
            KnowledgeHit(
                kp_id="KP-1",
                content="视觉内容",
                score=0.5,
                sub_scores={"visual": 0.8},
            ),
            KnowledgeHit(
                kp_id="KP-2",
                content="文本内容",
                score=0.5,
                sub_scores={},
            ),
        ]
        result = apply_learner_filter(profile, hits)
        visual_hit = next(h for h in result if h.kp_id == "KP-1")
        text_hit = next(h for h in result if h.kp_id == "KP-2")
        # visual 匹配: 0.5 * 1.2 = 0.6
        # text 不匹配: 0.5 * 0.9 = 0.45
        assert visual_hit.score > text_hit.score

    def test_学习风格不匹配降权(self):
        profile = LearnerProfile(
            learner_id="L1",
            level="intermediate",
            preferred_style=LearningStyle.AUDITORY,
            bloom_target=BloomLevel.CREATE,
        )
        hit = KnowledgeHit(kp_id="KP-1", content="x", score=0.8, sub_scores={})
        result = apply_learner_filter(profile, [hit])
        # 不匹配: 0.8 * 0.9 = 0.72
        assert result[0].score == pytest.approx(0.72, abs=0.01)

    def test_难度超过目标层级降权(self):
        """难度超过学习者目标层级的命中分数降低."""
        profile = LearnerProfile(
            learner_id="L1",
            level="beginner",
            preferred_style=LearningStyle.READING,
            bloom_target=BloomLevel.REMEMBER,  # 最低目标 (rank=1)
        )
        hit = KnowledgeHit(
            kp_id="KP-1",
            content="高难度内容",
            score=0.8,
            sub_scores={"difficulty": 0.9},  # difficulty_rank = int(0.9*6)+1 = 6 > 1
        )
        result = apply_learner_filter(profile, [hit])
        # 0.8 * 0.9(READING不匹配) * 0.5(难度) * 0.8(beginner) = 0.288
        assert result[0].score < 0.5

    def test_结果按分数降序排列(self):
        profile = LearnerProfile(
            learner_id="L1",
            level="intermediate",
            preferred_style=LearningStyle.VISUAL,
            bloom_target=BloomLevel.CREATE,
            weak_kps=["KP-1"],
        )
        hits = [
            KnowledgeHit(kp_id="KP-2", content="b", score=0.3, sub_scores={}),
            KnowledgeHit(kp_id="KP-1", content="a", score=0.3, sub_scores={}),
            KnowledgeHit(kp_id="KP-3", content="c", score=0.9, sub_scores={}),
        ]
        result = apply_learner_filter(profile, hits)
        scores = [h.score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_分数截断到合法范围(self):
        profile = LearnerProfile(
            learner_id="L1",
            level="intermediate",
            preferred_style=LearningStyle.VISUAL,
            bloom_target=BloomLevel.CREATE,
            weak_kps=["KP-1"],
        )
        hit = KnowledgeHit(
            kp_id="KP-1",
            content="x",
            score=0.9,
            sub_scores={"visual": 1.0},
        )
        result = apply_learner_filter(profile, [hit])
        # 0.9 * 1.5(weak) * 1.2(visual) = 1.62 → 截断到 1.0
        assert result[0].score == 1.0

    def test_原始结果不被修改(self):
        """apply_learner_filter 不应修改原始 hit 对象."""
        profile = LearnerProfile(
            learner_id="L1",
            level="intermediate",
            preferred_style=LearningStyle.VISUAL,
            bloom_target=BloomLevel.CREATE,
            weak_kps=["KP-1"],
        )
        original_score = 0.5
        hit = KnowledgeHit(
            kp_id="KP-1", content="x", score=original_score, sub_scores={"visual": 0.8}
        )
        apply_learner_filter(profile, [hit])
        assert hit.score == original_score  # 原始对象分数不变

    def test_初学者对高难度额外惩罚(self):
        """beginner 对高难度内容有额外 0.8x 惩罚."""
        beginner = LearnerProfile(
            learner_id="L1",
            level="beginner",
            preferred_style=LearningStyle.READING,
            bloom_target=BloomLevel.REMEMBER,
        )
        advanced = LearnerProfile(
            learner_id="L2",
            level="advanced",
            preferred_style=LearningStyle.READING,
            bloom_target=BloomLevel.REMEMBER,
        )
        hit_data = {
            "kp_id": "KP-1",
            "content": "高难度",
            "score": 0.8,
            "sub_scores": {"difficulty": 0.9},
        }
        hit_b = KnowledgeHit(**hit_data)
        hit_a = KnowledgeHit(**hit_data)
        result_b = apply_learner_filter(beginner, [hit_b])
        result_a = apply_learner_filter(advanced, [hit_a])
        # beginner: 0.8 * 0.9 * 0.5 * 0.8 = 0.288
        # advanced: 0.8 * 0.9 * 0.5 * 1.0 = 0.36
        assert result_b[0].score < result_a[0].score
