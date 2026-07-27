"""L3 领域知识层 — 知识连接器.

融合世界先进方案的外部知识连接器设计:
- MCP (Model Context Protocol): 统一工具调用协议 + 标准化资源发现
- LangChain Tools: 工具抽象 + 异步执行 + 错误重试 + 结构化输出
- LlamaIndex Readers: 多格式数据读取器 (PDF/网页/API/数据库)
- Airbyte Connectors: ELT 管道 + 增量同步 + Schema 演化
- Stripe API health check pattern: 指数退避健康探测 + 熔断恢复
- Netflix Hystrix: 熔断器模式 (CLOSED/OPEN/HALF_OPEN 三态)
- Kong API Gateway: 分级限流 (PUBLIC/INDUSTRY/PRIVATE 三档)
- Datadog: 健康指标采集 + 成功率追踪 + 错误计数

连接器分层架构 (三档分级):
1. PUBLIC  — 公共数据源 (NIST, PubChem, arXiv, Wikipedia, OpenAlex)
2. INDUSTRY — 行业数据源 (CAS, WoS, SciFinder, Reaxys, ThermoCalc)
3. PRIVATE — 校园/私有数据源 (图书馆, LIMS, 教务, 内部文档库)

连接器生命周期:
    REGISTERED → HEALTH_CHECKING → RUNNING → (DEGRADED | CIRCUIT_BREAKING)
                                          → VERSION_UPDATING → RECOVERING → RUNNING
                                          → OFFLINE

熔断器三态 (借鉴 Netflix Hystrix):
    CLOSED:     正常运行，请求全量放行
    OPEN:       连续失败达阈值，熔断拒绝所有请求
    HALF_OPEN:  恢复期半开探测，放行少量请求验证恢复

线程安全: ConnectorRegistry 和 CircuitBreaker 通过 threading.RLock 保护。
所有连接器均为抽象实现，接口设计支持未来替换为具体协议后端 (HTTP/gRPC/MCP)。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class ConnectorTier(str, Enum):
    """连接器层级 (借鉴 Kong API Gateway 分级路由).

    三档分级对应不同的访问权限、限流策略和数据可信度:
    - PUBLIC: 公共数据源，免费开放，宽松限流，权威度 T1
    - INDUSTRY: 行业数据源，付费授权，严格限流，权威度 T2
    - PRIVATE: 校园/私有数据源，内网访问，自定义限流，权威度 T3/T4
    """

    PUBLIC = "public"
    INDUSTRY = "industry"
    PRIVATE = "private"


class ConnectorStatus(str, Enum):
    """连接器运行状态 (8 种生命周期状态).

    状态流转:
        REGISTERED → HEALTH_CHECKING → RUNNING
        RUNNING → DEGRADED (性能下降)
        RUNNING → CIRCUIT_BREAKING (熔断触发)
        RUNNING → VERSION_UPDATING (版本更新)
        DEGRADED/CIRCUIT_BREAKING → RECOVERING → RUNNING
        任意状态 → OFFLINE (下线)
    """

    REGISTERED = "registered"
    HEALTH_CHECKING = "health_checking"
    RUNNING = "running"
    DEGRADED = "degraded"
    CIRCUIT_BREAKING = "circuit_breaking"
    VERSION_UPDATING = "version_updating"
    RECOVERING = "recovering"
    OFFLINE = "offline"


class ConnectorProtocol(str, Enum):
    """连接器通信协议 (借鉴 MCP 协议 + LangChain Tool adapters).

    - HTTP: 标准 HTTP/1.1 REST API
    - HTTPS: 加密 HTTP/2 REST API
    - GRPC: gRPC 二进制协议 (高性能场景)
    - MCP: Model Context Protocol (AI 工具标准化协议)
    - GRAPHQL: GraphQL 查询语言
    - REST: RESTful API (HTTPS 超集，含 OpenAPI 规范)
    """

    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    MCP = "mcp"
    GRAPHQL = "graphql"
    REST = "rest"


class CircuitState(str, Enum):
    """熔断器状态 (借鉴 Netflix Hystrix Circuit Breaker).

    - CLOSED: 关闭态，正常放行所有请求
    - OPEN: 开启态，熔断拒绝所有请求
    - HALF_OPEN: 半开态，放行探测请求验证恢复
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ============================================================
# 数据模型 (Pydantic v2)
# ============================================================


class ConnectorConfig(BaseModel):
    """连接器配置 (借鉴 Airbyte SourceConfiguration + LangChain Tool args).

    定义一个外部知识连接器的完整配置，包括连接参数、认证、
    限流策略、缓存策略、健康检查端点和元数据。

    Attributes:
        id: 连接器唯一标识 (如 "nist-webbook")
        name: 连接器显示名称
        tier: 连接器层级 (PUBLIC/INDUSTRY/PRIVATE)
        protocol: 通信协议
        base_url: 基础 URL
        auth_config: 认证配置 (如 {"type": "api_key", "header": "X-API-Key"})
        rate_limit: 每分钟请求上限
        cache_ttl: 缓存生存时间 (秒)
        health_check_url: 健康检查端点 URL
        version: 连接器版本号
        owner: 负责人/团队
        tags: 标签列表 (便于分类检索)
        mcp_tool_name: 对应的 MCP 工具名称 (如 "nist_query_spectrum")
        description: 连接器描述
        created_at: 创建时间戳
        metadata: 扩展元数据
    """

    id: str = Field(..., min_length=1, description="连接器唯一标识")
    name: str = Field(..., min_length=1, description="连接器显示名称")
    tier: ConnectorTier = Field(default=ConnectorTier.PUBLIC, description="连接器层级")
    protocol: ConnectorProtocol = Field(
        default=ConnectorProtocol.HTTPS, description="通信协议"
    )
    base_url: str = Field(..., min_length=1, description="基础 URL")
    auth_config: dict[str, Any] = Field(default_factory=dict, description="认证配置")
    rate_limit: int = Field(default=60, ge=0, description="每分钟请求上限")
    cache_ttl: float = Field(default=300.0, ge=0.0, description="缓存生存时间 (秒)")
    health_check_url: str = Field(default="", description="健康检查端点 URL")
    version: str = Field(default="1.0.0", description="连接器版本号")
    owner: str = Field(default="system", description="负责人/团队")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    mcp_tool_name: str = Field(default="", description="对应的 MCP 工具名称")
    description: str = Field(default="", description="连接器描述")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")


class ConnectorHealth(BaseModel):
    """连接器健康状态 (借鉴 Stripe API health check + Datadog 指标采集).

    记录连接器的实时健康指标，用于熔断决策和监控告警。

    Attributes:
        status: 当前运行状态
        last_check_time: 最后一次健康检查时间戳
        response_time_ms: 最近请求响应时间 (毫秒)
        success_rate: 成功率 [0.0, 1.0] (滑动窗口统计)
        error_count: 错误累计次数
        last_error: 最近一次错误信息
    """

    status: ConnectorStatus = Field(
        default=ConnectorStatus.REGISTERED, description="当前运行状态"
    )
    last_check_time: float = Field(default=0.0, description="最后健康检查时间戳")
    response_time_ms: float = Field(default=0.0, ge=0.0, description="响应时间 (毫秒)")
    success_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="成功率 [0,1]"
    )
    error_count: int = Field(default=0, ge=0, description="错误累计次数")
    last_error: str = Field(default="", description="最近一次错误信息")


class ConnectorResponse(BaseModel):
    """连接器响应 (借鉴 LangChain ToolMessage + MCP result).

    统一封装所有连接器的返回结果，支持缓存标记和延迟追踪。

    Attributes:
        success: 请求是否成功
        data: 返回数据 (任意类型)
        error: 错误信息 (失败时填充)
        source: 数据来源标识 (连接器 ID)
        latency_ms: 请求延迟 (毫秒)
        cached: 是否来自缓存
        timestamp: 响应时间戳
    """

    success: bool = Field(default=False, description="请求是否成功")
    data: Any = Field(default=None, description="返回数据")
    error: str = Field(default="", description="错误信息")
    source: str = Field(default="", description="数据来源标识")
    latency_ms: float = Field(default=0.0, ge=0.0, description="请求延迟 (毫秒)")
    cached: bool = Field(default=False, description="是否来自缓存")
    timestamp: float = Field(default_factory=time.time, description="响应时间戳")


# ============================================================
# 熔断器 (借鉴 Netflix Hystrix Circuit Breaker)
# ============================================================


class CircuitBreaker:
    """熔断器 (借鉴 Netflix Hystrix + resilience4j).

    通过连续失败计数和状态机实现自动熔断与恢复:
    - CLOSED → OPEN: 连续失败达 failure_threshold 次
    - OPEN → HALF_OPEN: 熔断恢复时间 (recovery_timeout) 到期后
    - HALF_OPEN → CLOSED: 探测请求成功
    - HALF_OPEN → OPEN: 探测请求失败

    Attributes:
        failure_threshold: 连续失败触发熔断的阈值
        recovery_timeout: 熔断恢复等待时间 (秒)
        half_open_max_calls: 半开态最大探测请求数
        _state: 当前熔断状态
        _failure_count: 连续失败计数
        _last_failure_time: 最后失败时间戳
        _half_open_calls: 半开态已放行的探测请求数
        _lock: 线程安全锁
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        """初始化熔断器.

        Args:
            failure_threshold: 连续失败触发熔断的阈值 (默认 5 次)
            recovery_timeout: 熔断恢复等待时间 (默认 60 秒)
            half_open_max_calls: 半开态最大探测请求数 (默认 3 次)
        """
        self.failure_threshold: int = failure_threshold
        self.recovery_timeout: float = recovery_timeout
        self.half_open_max_calls: int = half_open_max_calls

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls: int = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """当前熔断状态 (自动检查恢复超时)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否已过恢复超时
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("熔断器进入半开状态，开始探测")
            return self._state

    def allow_request(self) -> bool:
        """是否允许请求通过.

        Returns:
            True 如果请求被放行，False 如果被熔断拒绝
        """
        with self._lock:
            current_state = self.state  # 触发恢复检查
            if current_state == CircuitState.CLOSED:
                return True
            if current_state == CircuitState.OPEN:
                return False
            # HALF_OPEN: 限制探测请求数量
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        """记录一次成功请求."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # 半开态成功 → 恢复为关闭态
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info("熔断器恢复为关闭状态")
            elif self._state == CircuitState.CLOSED:
                # 关闭态成功 → 重置失败计数
                self._failure_count = 0

    def record_failure(self) -> None:
        """记录一次失败请求."""
        with self._lock:
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                # 半开态失败 → 重新熔断
                self._state = CircuitState.OPEN
                self._failure_count = self.failure_threshold
                logger.warning("半开态探测失败，熔断器重新开启")
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "连续失败 %d 次达到阈值，熔断器开启",
                        self._failure_count,
                    )

    def reset(self) -> None:
        """重置熔断器到初始关闭状态."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._half_open_calls = 0

    def get_stats(self) -> dict[str, Any]:
        """获取熔断器统计信息."""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
            }


# ============================================================
# KnowledgeConnector 抽象基类
# ============================================================


class KnowledgeConnector(ABC):
    """知识连接器抽象基类 (借鉴 LangChain BaseTool + LlamaIndex BaseReader).

    定义外部知识源连接的统一接口，包括连接管理、健康检查、
    搜索/获取/列表三大核心操作，以及限流和缓存支持。

    子类需实现三个抽象方法:
    - search: 搜索知识
    - fetch: 获取指定资源
    - list_resources: 列出可用资源

    内置功能:
    - 限流检查 (_check_rate_limit): 基于令牌桶的请求限流
    - 缓存管理 (_cache_get/_cache_set): TTL 缓存减少重复请求
    - 熔断保护: 集成 CircuitBreaker 自动熔断与恢复
    - 健康检查: 定期探测连接器可用性

    Attributes:
        _config: 连接器配置
        _health: 健康状态
        _is_connected: 连接状态标志
        _cache: 本地缓存 {key: (value, expire_time)}
        _rate_limit_window: 限流时间窗口记录
        _circuit_breaker: 熔断器实例
        _lock: 线程安全锁
    """

    def __init__(self, config: ConnectorConfig) -> None:
        """初始化知识连接器.

        Args:
            config: 连接器配置
        """
        self._config: ConnectorConfig = config
        self._health: ConnectorHealth = ConnectorHealth(
            status=ConnectorStatus.REGISTERED,
        )
        self._is_connected: bool = False
        self._cache: dict[str, tuple[Any, float]] = {}
        self._rate_limit_window: list[float] = []
        self._circuit_breaker: CircuitBreaker = CircuitBreaker()
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------

    @property
    def config(self) -> ConnectorConfig:
        """连接器配置."""
        return self._config

    @property
    def health(self) -> ConnectorHealth:
        """连接器健康状态."""
        return self._health

    @property
    def is_connected(self) -> bool:
        """是否已连接."""
        return self._is_connected

    # --------------------------------------------------------
    # 连接管理
    # --------------------------------------------------------

    def connect(self) -> bool:
        """建立连接.

        执行连接器初始化，设置连接状态并触发首次健康检查。

        Returns:
            True 如果连接成功
        """
        try:
            self._health.status = ConnectorStatus.HEALTH_CHECKING
            success = self._do_connect()
            if success:
                self._is_connected = True
                self._health.status = ConnectorStatus.RUNNING
                self._health.last_check_time = time.time()
                logger.info("连接器 [%s] 连接成功", self._config.id)
            else:
                self._health.status = ConnectorStatus.OFFLINE
                logger.warning("连接器 [%s] 连接失败", self._config.id)
            return success
        except Exception as exc:
            self._health.status = ConnectorStatus.OFFLINE
            self._health.last_error = str(exc)
            self._health.error_count += 1
            logger.exception("连接器 [%s] 连接异常", self._config.id)
            return False

    def disconnect(self) -> None:
        """断开连接.

        释放连接器资源，清理缓存，重置连接状态。
        """
        try:
            self._do_disconnect()
        finally:
            self._is_connected = False
            self._health.status = ConnectorStatus.OFFLINE
            with self._lock:
                self._cache.clear()
            logger.info("连接器 [%s] 已断开", self._config.id)

    def health_check(self) -> ConnectorHealth:
        """执行健康检查 (借鉴 Stripe API health check pattern).

        探测连接器可用性，更新健康状态指标。
        子类可重写 _do_health_check 实现具体探测逻辑。

        Returns:
            更新后的健康状态
        """
        start_time = time.time()
        try:
            self._health.status = ConnectorStatus.HEALTH_CHECKING
            is_healthy = self._do_health_check()
            elapsed_ms = (time.time() - start_time) * 1000

            self._health.last_check_time = time.time()
            self._health.response_time_ms = elapsed_ms

            if is_healthy:
                self._health.status = ConnectorStatus.RUNNING
                self._circuit_breaker.record_success()
            else:
                self._health.status = ConnectorStatus.DEGRADED
                self._health.error_count += 1
                self._circuit_breaker.record_failure()
                if self._circuit_breaker.state == CircuitState.OPEN:
                    self._health.status = ConnectorStatus.CIRCUIT_BREAKING

        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            self._health.last_check_time = time.time()
            self._health.response_time_ms = elapsed_ms
            self._health.status = ConnectorStatus.DEGRADED
            self._health.last_error = str(exc)
            self._health.error_count += 1
            self._circuit_breaker.record_failure()
            if self._circuit_breaker.state == CircuitState.OPEN:
                self._health.status = ConnectorStatus.CIRCUIT_BREAKING
            logger.exception("连接器 [%s] 健康检查异常", self._config.id)

        return self._health

    # --------------------------------------------------------
    # 抽象方法 — 子类必须实现
    # --------------------------------------------------------

    @abstractmethod
    def search(self, query: str, **kwargs: Any) -> ConnectorResponse:
        """搜索知识 (抽象方法).

        Args:
            query: 搜索查询字符串
            **kwargs: 额外搜索参数 (如 top_k, filters, language)

        Returns:
            连接器响应
        """

    @abstractmethod
    def fetch(self, resource_id: str) -> ConnectorResponse:
        """获取指定资源 (抽象方法).

        Args:
            resource_id: 资源唯一标识 (如 CAS 号, DOI, 条目 ID)

        Returns:
            连接器响应
        """

    @abstractmethod
    def list_resources(self, **kwargs: Any) -> ConnectorResponse:
        """列出可用资源 (抽象方法).

        Args:
            **kwargs: 列表参数 (如 limit, offset, category)

        Returns:
            连接器响应
        """

    # --------------------------------------------------------
    # 内部辅助方法
    # --------------------------------------------------------

    def _check_rate_limit(self) -> bool:
        """限流检查 (借鉴 Kong API Gateway 令牌桶限流).

        基于滑动窗口算法检查当前请求是否在限流范围内。

        Returns:
            True 如果请求被允许，False 如果被限流
        """
        if self._config.rate_limit <= 0:
            return True

        now = time.time()
        window_start = now - 60.0  # 1 分钟滑动窗口

        with self._lock:
            # 清理过期记录
            self._rate_limit_window = [
                t for t in self._rate_limit_window if t > window_start
            ]
            if len(self._rate_limit_window) >= self._config.rate_limit:
                logger.warning(
                    "连接器 [%s] 触发限流: %d/%d",
                    self._config.id,
                    len(self._rate_limit_window),
                    self._config.rate_limit,
                )
                return False
            self._rate_limit_window.append(now)
            return True

    def _cache_get(self, key: str) -> Any | None:
        """缓存获取.

        Args:
            key: 缓存键

        Returns:
            缓存值，未命中或已过期返回 None
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, expire_time = entry
            if time.time() > expire_time:
                del self._cache[key]
                return None
            return value

    def _cache_set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """缓存设置.

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间 (秒)，None 时使用配置默认值
        """
        effective_ttl = ttl if ttl is not None else self._config.cache_ttl
        expire_time = time.time() + effective_ttl
        with self._lock:
            self._cache[key] = (value, expire_time)

    def _make_cache_key(self, *parts: Any) -> str:
        """生成缓存键 (借鉴 Elasticsearch request cache key).

        Args:
            *parts: 缓存键组成部分

        Returns:
            SHA-256 哈希缓存键
        """
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _execute_with_protection(
        self, operation: str, *args: Any, **kwargs: Any
    ) -> ConnectorResponse:
        """在熔断器和限流保护下执行操作.

        Args:
            operation: 操作类型 ("search"/"fetch"/"list_resources")
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            连接器响应
        """
        # 熔断器检查
        if not self._circuit_breaker.allow_request():
            return ConnectorResponse(
                success=False,
                error="熔断器开启，请求被拒绝",
                source=self._config.id,
            )

        # 限流检查
        if not self._check_rate_limit():
            return ConnectorResponse(
                success=False,
                error="请求频率超限",
                source=self._config.id,
            )

        start_time = time.time()
        try:
            # 缓存检查
            cache_key = self._make_cache_key(operation, args, kwargs)
            cached = self._cache_get(cache_key)
            if cached is not None:
                return ConnectorResponse(
                    success=True,
                    data=cached,
                    source=self._config.id,
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True,
                )

            # 执行实际操作
            if operation == "search":
                result = self.search(*args, **kwargs)
            elif operation == "fetch":
                result = self.fetch(*args, **kwargs)
            elif operation == "list_resources":
                result = self.list_resources(*args, **kwargs)
            else:
                return ConnectorResponse(
                    success=False,
                    error=f"未知操作类型: {operation}",
                    source=self._config.id,
                )

            # 记录结果
            if result.success:
                self._circuit_breaker.record_success()
                if result.data is not None:
                    self._cache_set(cache_key, result.data)
            else:
                self._circuit_breaker.record_failure()
                self._health.error_count += 1
                self._health.last_error = result.error

            result.latency_ms = (time.time() - start_time) * 1000
            return result

        except Exception as exc:
            self._circuit_breaker.record_failure()
            self._health.error_count += 1
            self._health.last_error = str(exc)
            logger.exception("连接器 [%s] 执行 %s 异常", self._config.id, operation)
            return ConnectorResponse(
                success=False,
                error=str(exc),
                source=self._config.id,
                latency_ms=(time.time() - start_time) * 1000,
            )

    # --------------------------------------------------------
    # 子类可重写的钩子方法
    # --------------------------------------------------------

    def _do_connect(self) -> bool:
        """实际连接逻辑 (子类可重写).

        默认实现: 验证 base_url 非空即视为连接成功。
        """
        return bool(self._config.base_url)

    def _do_disconnect(self) -> None:
        """实际断开逻辑 (子类可重写)."""
        # 默认空实现

    def _do_health_check(self) -> bool:
        """实际健康检查逻辑 (子类可重写).

        默认实现: 已连接即视为健康。
        """
        return self._is_connected


# ============================================================
# ConnectorRegistry — 连接器注册中心
# ============================================================


class ConnectorRegistry:
    """连接器注册中心 (借鉴 Airbyte Connection Registry + Kong Service Registry).

    管理所有知识连接器的注册、发现、健康监控和全局搜索。

    内置指数退避健康检查调度 (借鉴 Stripe API retry pattern):
    - 基础间隔: 30 秒
    - 最大间隔: 5 分钟 (300 秒)
    - 退避因子: 2.0 (每次失败翻倍)
    - 成功后重置为基础间隔

    Attributes:
        _connectors: 已注册连接器 {connector_id: KnowledgeConnector}
        _health_intervals: 健康检查间隔 {connector_id: next_check_time}
        _base_interval: 基础健康检查间隔 (30 秒)
        _max_interval: 最大健康检查间隔 (300 秒)
        _lock: 线程安全锁
    """

    def __init__(
        self,
        base_interval: float = 30.0,
        max_interval: float = 300.0,
    ) -> None:
        """初始化连接器注册中心.

        Args:
            base_interval: 基础健康检查间隔 (默认 30 秒)
            max_interval: 最大健康检查间隔 (默认 300 秒)
        """
        self._connectors: dict[str, KnowledgeConnector] = {}
        self._health_intervals: dict[str, float] = {}  # connector_id -> 当前间隔
        self._next_check: dict[str, float] = {}  # connector_id -> 下次检查时间
        self._base_interval: float = base_interval
        self._max_interval: float = max_interval
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 注册管理
    # --------------------------------------------------------

    def register(self, connector: KnowledgeConnector) -> str:
        """注册连接器.

        Args:
            connector: 知识连接器实例

        Returns:
            连接器 ID

        Raises:
            ValueError: 连接器 ID 已存在
        """
        connector_id = connector.config.id
        with self._lock:
            if connector_id in self._connectors:
                raise ValueError(f"连接器已注册: {connector_id}")
            self._connectors[connector_id] = connector
            self._health_intervals[connector_id] = self._base_interval
            self._next_check[connector_id] = time.time() + self._base_interval
            logger.info("注册连接器: %s (层级=%s, 协议=%s)",
                        connector_id,
                        connector.config.tier.value,
                        connector.config.protocol.value)
        return connector_id

    def unregister(self, connector_id: str) -> bool:
        """注销连接器.

        Args:
            connector_id: 连接器 ID

        Returns:
            True 如果注销成功，False 如果连接器不存在
        """
        with self._lock:
            connector = self._connectors.pop(connector_id, None)
            if connector is None:
                return False
            connector.disconnect()
            self._health_intervals.pop(connector_id, None)
            self._next_check.pop(connector_id, None)
            logger.info("注销连接器: %s", connector_id)
            return True

    def get(self, connector_id: str) -> KnowledgeConnector | None:
        """获取连接器.

        Args:
            connector_id: 连接器 ID

        Returns:
            连接器实例，不存在返回 None
        """
        with self._lock:
            return self._connectors.get(connector_id)

    # --------------------------------------------------------
    # 列表查询
    # --------------------------------------------------------

    def list_by_tier(self, tier: ConnectorTier) -> list[KnowledgeConnector]:
        """按层级列出连接器.

        Args:
            tier: 连接器层级

        Returns:
            匹配层级的连接器列表
        """
        with self._lock:
            return [
                c for c in self._connectors.values() if c.config.tier == tier
            ]

    def list_by_status(self, status: ConnectorStatus) -> list[KnowledgeConnector]:
        """按状态列出连接器.

        Args:
            status: 连接器状态

        Returns:
            匹配状态的连接器列表
        """
        with self._lock:
            return [
                c for c in self._connectors.values() if c.health.status == status
            ]

    def list_all(self) -> list[KnowledgeConnector]:
        """列出全部连接器.

        Returns:
            所有已注册的连接器列表
        """
        with self._lock:
            return list(self._connectors.values())

    # --------------------------------------------------------
    # 健康检查 (指数退避)
    # --------------------------------------------------------

    def health_check_all(self) -> dict[str, ConnectorHealth]:
        """对所有连接器执行健康检查 (指数退避调度).

        只对到达下次检查时间的连接器执行健康检查，
        根据检查结果调整下次检查间隔 (指数退避)。

        Returns:
            所有连接器的健康状态 {connector_id: ConnectorHealth}
        """
        now = time.time()
        results: dict[str, ConnectorHealth] = {}

        with self._lock:
            connectors_to_check = [
                (cid, c) for cid, c in self._connectors.items()
                if self._next_check.get(cid, 0) <= now
            ]

        for connector_id, connector in connectors_to_check:
            health = connector.health_check()
            results[connector_id] = health

            # 指数退避调整
            with self._lock:
                if health.status in (
                    ConnectorStatus.RUNNING,
                ):
                    # 健康 → 重置为基础间隔
                    self._health_intervals[connector_id] = self._base_interval
                else:
                    # 不健康 → 翻倍间隔 (上限 max_interval)
                    current = self._health_intervals.get(
                        connector_id, self._base_interval
                    )
                    self._health_intervals[connector_id] = min(
                        current * 2.0, self._max_interval
                    )

                self._next_check[connector_id] = (
                    now + self._health_intervals[connector_id]
                )

        # 未检查的连接器也返回当前健康状态
        with self._lock:
            for cid, connector in self._connectors.items():
                if cid not in results:
                    results[cid] = connector.health

        return results

    def _get_health_interval(self, connector_id: str) -> float:
        """获取连接器当前的健康检查间隔."""
        with self._lock:
            return self._health_intervals.get(connector_id, self._base_interval)

    # --------------------------------------------------------
    # 全局搜索
    # --------------------------------------------------------

    def search_all(
        self, query: str, **kwargs: Any
    ) -> list[ConnectorResponse]:
        """在所有运行中的连接器上执行搜索.

        只对处于 RUNNING 或 DEGRADED 状态的连接器执行搜索，
        跳过熔断中和离线的连接器。

        Args:
            query: 搜索查询字符串
            **kwargs: 额外搜索参数

        Returns:
            各连接器的搜索响应列表
        """
        results: list[ConnectorResponse] = []

        with self._lock:
            active_connectors = [
                (cid, c) for cid, c in self._connectors.items()
                if c.health.status in (
                    ConnectorStatus.RUNNING,
                    ConnectorStatus.DEGRADED,
                )
            ]

        for connector_id, connector in active_connectors:
            try:
                response = connector._execute_with_protection(
                    "search", query, **kwargs
                )
                results.append(response)
            except Exception as exc:
                logger.exception("连接器 [%s] 搜索异常", connector_id)
                results.append(
                    ConnectorResponse(
                        success=False,
                        error=str(exc),
                        source=connector_id,
                    )
                )

        return results

    # --------------------------------------------------------
    # 统计信息
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取注册中心统计信息.

        Returns:
            统计信息字典，包括:
            - total: 连接器总数
            - by_tier: 按层级统计
            - by_status: 按状态统计
            - by_protocol: 按协议统计
            - avg_success_rate: 平均成功率
            - avg_response_time_ms: 平均响应时间
        """
        with self._lock:
            connectors = list(self._connectors.values())

        total = len(connectors)
        by_tier: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_protocol: dict[str, int] = {}

        success_rates: list[float] = []
        response_times: list[float] = []

        for c in connectors:
            tier_key = c.config.tier.value
            by_tier[tier_key] = by_tier.get(tier_key, 0) + 1

            status_key = c.health.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            proto_key = c.config.protocol.value
            by_protocol[proto_key] = by_protocol.get(proto_key, 0) + 1

            success_rates.append(c.health.success_rate)
            if c.health.response_time_ms > 0:
                response_times.append(c.health.response_time_ms)

        avg_success = (
            sum(success_rates) / len(success_rates) if success_rates else 0.0
        )
        avg_response = (
            sum(response_times) / len(response_times) if response_times else 0.0
        )

        return {
            "total": total,
            "by_tier": by_tier,
            "by_status": by_status,
            "by_protocol": by_protocol,
            "avg_success_rate": round(avg_success, 4),
            "avg_response_time_ms": round(avg_response, 2),
        }


__all__ = [
    # 枚举
    "ConnectorTier",
    "ConnectorStatus",
    "ConnectorProtocol",
    "CircuitState",
    # 数据模型
    "ConnectorConfig",
    "ConnectorHealth",
    "ConnectorResponse",
    # 核心类
    "CircuitBreaker",
    "KnowledgeConnector",
    "ConnectorRegistry",
]
