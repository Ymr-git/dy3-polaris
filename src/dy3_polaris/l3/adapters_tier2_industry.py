"""L3 领域知识层 — Tier-2 行业数据源适配器.

Tier-2 行业数据源策略 (权威度 T2):
=================================================

Tier-2 (INDUSTRY) 行业数据源是 dy3-polaris L3 知识层中权威度排名第二的数据来源,
介于 Tier-1 公共数据源 (NIST / PubChem / arXiv, 免费、宽松限流) 与 Tier-3/4
校园私有数据源之间。本模块封装 6 个高价值付费 / 授权行业数据源的具体适配器实现。

核心特征:
1. 付费授权访问 — 需要机构订阅或商业 API Key, 非免费开放
2. 严格限流策略 — 5~50 req/min, 远低于公共数据源, 节省 API 配额
3. 高权威度 (T2) — 数据经专业审核, 引用价值高于公共数据源
4. 长缓存周期 — cache_ttl 通常 7200s (2h), 减少重复付费请求
5. 增量同步优先 — 默认 INCREMENTAL 模式, 基于游标字段只拉取变更
6. 熔断保护 — 连续失败自动熔断, 避免浪费昂贵的 API 配额

数据源覆盖:
- 化学物质注册: CAS Registry (3 亿+ 物质, 黄金标准 CAS RN)
- 引文索引: Clarivate Web of Science (1900 年至今, 21K+ 期刊)
- 化学文献检索: CAS SciFinder (文献 / 物质 / 反应)
- 反应数据库: Elsevier Reaxys (物质 / 反应 / 文档 / 性质, GraphQL)
- 专利数据库: Google Patents (1.2 亿+ 专利, 100+ 专利局)
- 工程文献: Elsevier Engineering Village (Compendex / Inspec / GEOBASE)

架构说明:
- CASAdapter / WebOfScienceAdapter / SciFinderAdapter /
  GooglePatentsAdapter / EngineeringVillageAdapter 继承 RESTAdapter (REST/HTTPS)
- ReaxysAdapter 继承 GraphQLAdapter (GraphQL/HTTPS)
- 每个适配器在 __init__ 中配置 SchemaMapper, 将源字段映射到 L3 标准字段
  (entity_id / entity_name / entity_type / identifiers / properties /
   source_uri / description + 领域专属字段)
- 每个适配器提供 create() 类方法返回预配置实例 (含模拟数据)
- 所有 HTTP/GraphQL 调用均为模拟实现, 通过 set_mock_data() 注入测试数据,
  接口设计支持未来替换为真实请求后端 (requests / httpx / httpx-gql)

线程安全: 继承自 DataAdapterBase, 通过 threading.RLock 保护内部状态。
"""

from __future__ import annotations

import logging
from typing import Any

from .adapter_bases import DatabaseAdapter, GraphQLAdapter, RESTAdapter
from .connector import ConnectorConfig, ConnectorProtocol, ConnectorTier
from .data_source_adapter import (
    AdapterCapability,
    AdapterSpec,
    DataAdapterBase,
    DataSourceSchema,
    DataSourceType,
    DiscoverResult,
    FieldMapping,
    LifecyclePhase,
    ReadResult,
    SchemaField,
    SchemaMapper,
    SyncCheckpoint,
    SyncMode,
)

logger = logging.getLogger(__name__)


# ============================================================
# 通用辅助
# ============================================================


def _rest_caps(*extra: AdapterCapability) -> int:
    """构建 REST 行业适配器的标准能力集合 (含 RATE_LIMITED + INCREMENTAL).

    Args:
        *extra: 额外能力标志

    Returns:
        能力标志位的整数值
    """
    caps = (
        AdapterCapability.SEARCH
        | AdapterCapability.FETCH
        | AdapterCapability.LIST
        | AdapterCapability.BATCH
        | AdapterCapability.DISCOVER
        | AdapterCapability.HEALTH_CHECK
        | AdapterCapability.AUTHENTICATE
        | AdapterCapability.RATE_LIMITED
        | AdapterCapability.CACHEABLE
        | AdapterCapability.INCREMENTAL
    )
    for cap in extra:
        caps |= cap
    # AdapterCapability 是 Flag (非 IntFlag), 需通过 .value 取整数值
    return caps.value


# ============================================================
# 适配器 1: CASAdapter — CAS 化学物质注册库
# ============================================================


class CASAdapter(RESTAdapter):
    """CAS (Chemical Abstracts Service) 化学物质注册库适配器.

    数据源: CAS SciFinder API (https://api.cas.org)
    协议: REST / HTTPS, 认证 OAuth2 (client_id + client_secret)
    限流: 10 req/min (极其受限的商业配额)
    缓存: 7200s (2 小时)

    CAS Registry 是全球化学物质注册的黄金标准, 收录 3 亿+ 物质, 每条物质
    分配唯一的 CAS Registry Number (CAS RN)。CAS RN 是化学领域最权威的物质
    标识符, 广泛用于化学品监管、供应链追踪和学术引用。

    支持的流 (streams):
    - "substances": 化学物质记录 (CAS RN / 名称 / 分子式 / 结构 / 性质)
    - "reactions": 化学反应记录 (反应物 / 产物 / 条件 / 产率)
    - "references": 文献引用记录 (关联 CAS 物质 / 反应的原始文献)

    搜索维度:
    - 按 CAS RN 精确查询 (如 "7732-18-5")
    - 按物质名称查询 (如 "ethanol")
    - 按分子式查询 (如 "C2H6O")
    - 按结构查询 (SMILES / InChI)

    OAuth2 认证流程:
    1. POST /oauth/token (client_id + client_secret) → access_token
    2. 后续请求携带 Authorization: Bearer <access_token>
    本适配器的 auth_token 参数即为步骤 1 获取的 access_token。

    SchemaMapper 映射 (源字段 → L3 标准字段):
    - cas_rn → entity_id (主键, 必须存在)
    - substance_name → entity_name
    - record_type → entity_type (默认 "chemical_substance")
    - cas_rn → identifiers (CAS RN 作为标识符)
    - property_data → properties (物理化学性质字典)
    - structure_smiles → source_uri (结构标识符)
    - substance_name → description
    - molecular_formula / molecular_weight → 同名领域字段
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        auth_token: str = "",
        client_id: str = "",
        client_secret: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 CAS 适配器.

        Args:
            config: 连接器配置 (tier=INDUSTRY, protocol=REST)
            auth_token: OAuth2 access_token (token exchange 后获得)
            client_id: OAuth2 客户端 ID (用于 token exchange)
            client_secret: OAuth2 客户端密钥 (用于 token exchange)
            **kwargs: 传递给 RESTAdapter 的额外参数
        """
        super().__init__(
            config,
            search_endpoint=kwargs.pop(
                "search_endpoint", "/api/substances/search"
            ),
            fetch_endpoint=kwargs.pop(
                "fetch_endpoint", "/api/substances/{id}"
            ),
            page_size=kwargs.pop("page_size", 10),
            auth_type="oauth2",
            auth_token=auth_token,
            **kwargs,
        )
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token = auth_token

        # SchemaMapper: 源字段 → L3 标准字段
        mapper = self._schema_mapper
        mapper.add_mapping(
            FieldMapping(
                source_field="cas_rn",
                target_field="entity_id",
                required=True,
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="substance_name",
                target_field="entity_name",
                required=True,
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="record_type",
                target_field="entity_type",
                default_value="chemical_substance",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="cas_rn",
                target_field="identifiers",
                transform="to_upper",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="property_data", target_field="properties")
        )
        mapper.add_mapping(
            FieldMapping(source_field="structure_smiles", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="substance_name", target_field="description")
        )
        # 领域专属字段 (identity 映射以保留在输出中)
        mapper.add_mapping(
            FieldMapping(
                source_field="cas_rn",
                target_field="cas_rn",
                transform="to_upper",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="molecular_formula", target_field="molecular_formula"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="molecular_weight",
                target_field="molecular_weight",
                transform="parse_float",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="structure_smiles", target_field="structure_smiles"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="structure_inchi", target_field="structure_inchi"
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="synonyms", target_field="synonyms")
        )

    def _do_spec(self) -> AdapterSpec:
        """声明 CAS 适配器规范 (REST_API + INCREMENTAL + RATE_LIMITED)."""
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=_rest_caps(),
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "client_id": {"type": "string"},
                    "client_secret": {"type": "string"},
                    "auth_token": {"type": "string"},
                    "page_size": {"type": "integer", "default": 10},
                },
                "required": ["base_url", "client_id", "client_secret"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://www.cas.org/support/documentation",
            changelog={"1.0.0": "初始版本, 支持 substances 流"},
        )

    def _get_schema(self) -> DataSourceSchema:
        """返回 substances 流的 Schema."""
        return DataSourceSchema(
            stream_name="substances",
            fields=[
                SchemaField(
                    name="cas_rn",
                    data_type="string",
                    nullable=False,
                    primary_key=True,
                    description="CAS Registry Number (唯一标识)",
                    format="cas-rn",
                ),
                SchemaField(
                    name="substance_name",
                    data_type="string",
                    nullable=False,
                    description="物质标准名称",
                ),
                SchemaField(
                    name="molecular_formula",
                    data_type="string",
                    description="分子式 (Hill 表示法)",
                ),
                SchemaField(
                    name="molecular_weight",
                    data_type="float",
                    description="分子量 (g/mol)",
                ),
                SchemaField(
                    name="structure_smiles",
                    data_type="string",
                    description="SMILES 结构表示",
                ),
                SchemaField(
                    name="structure_inchi",
                    data_type="string",
                    description="InChI 国际化学标识符",
                ),
                SchemaField(
                    name="synonyms",
                    data_type="array",
                    description="同义词列表",
                ),
                SchemaField(
                    name="commercial_sources",
                    data_type="array",
                    description="商业供应商列表",
                ),
                SchemaField(
                    name="regulatory_info",
                    data_type="object",
                    description="监管信息 (EPA / REACH 等)",
                ),
                SchemaField(
                    name="property_data",
                    data_type="object",
                    description="物理化学性质字典",
                ),
            ],
            primary_keys=["cas_rn"],
            cursor_field="cas_rn",
            description="CAS Registry 化学物质记录 (3 亿+ 物质)",
            metadata={"source": "CAS", "authority_tier": "T2"},
        )

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 CAS 物质搜索 URL.

        支持按 CAS RN / 名称 / 分子式 / 结构搜索, 通过 search_type 参数区分。

        Returns:
            (url, params, headers)
        """
        url = f"{self.config.base_url}{self._search_endpoint}"
        search_type = kwargs.get("search_type", "name")
        params: dict[str, Any] = {
            "query": query,
            "type": search_type,
            "limit": kwargs.get("limit", self._page_size),
            "offset": kwargs.get("offset", 0),
        }
        return url, params, self._build_auth_headers()

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 CAS 物质获取 URL (按 CAS RN)."""
        endpoint = self._fetch_endpoint.replace("{id}", resource_id)
        url = f"{self.config.base_url}{endpoint}"
        return url, {}, self._build_auth_headers()

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 CAS API 响应.

        CAS 返回格式: {"substances": [...], "total_count": N, "query_id": "..."}
        """
        if isinstance(data, dict):
            for key in ("substances", "results", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            return data
        return []

    def _build_auth_headers(self) -> dict[str, str]:
        """构建 OAuth2 认证头.

        生产环境应先调用 /oauth/token 进行 client_credentials 授权换取
        access_token, 此处直接使用注入的 auth_token 作为 Bearer 令牌。
        """
        headers: dict[str, str] = {"Accept": "application/json"}
        token = self._access_token or self._auth_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @classmethod
    def create(
        cls,
        auth_token: str = "",
        *,
        client_id: str = "",
        client_secret: str = "",
    ) -> CASAdapter:
        """创建预配置的 CAS 适配器实例 (含模拟数据).

        Args:
            auth_token: OAuth2 access_token
            client_id: OAuth2 客户端 ID
            client_secret: OAuth2 客户端密钥

        Returns:
            预配置的 CASAdapter 实例
        """
        config = ConnectorConfig(
            id="cas-registry",
            name="CAS Chemical Registry",
            tier=ConnectorTier.INDUSTRY,
            protocol=ConnectorProtocol.REST,
            base_url="https://api.cas.org",
            auth_config={"type": "oauth2"},
            rate_limit=10,
            cache_ttl=7200,
            version="1.0.0",
            tags=["chemistry", "registry", "cas-rn", "paid"],
            description="CAS SciFinder 化学物质注册库 (3 亿+ 物质)",
        )
        instance = cls(
            config,
            auth_token=auth_token,
            client_id=client_id,
            client_secret=client_secret,
        )
        instance.set_mock_data(
            [
                {
                    "cas_rn": "7732-18-5",
                    "substance_name": "Water",
                    "record_type": "chemical_substance",
                    "molecular_formula": "H2O",
                    "molecular_weight": 18.015,
                    "structure_smiles": "O",
                    "structure_inchi": "InChI=1S/H2O/h1H2",
                    "synonyms": ["water", "dihydrogen oxide", "aqua"],
                    "commercial_sources": ["Sigma-Aldrich", "Fisher Scientific"],
                    "regulatory_info": {"EPA": "not_regulated", "REACH": "registered"},
                    "property_data": {
                        "melting_point": 0.0,
                        "boiling_point": 100.0,
                        "density": 0.997,
                    },
                },
                {
                    "cas_rn": "64-17-5",
                    "substance_name": "Ethanol",
                    "record_type": "chemical_substance",
                    "molecular_formula": "C2H6O",
                    "molecular_weight": 46.069,
                    "structure_smiles": "CCO",
                    "structure_inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                    "synonyms": ["ethyl alcohol", "alcohol", "ethanol"],
                    "commercial_sources": ["Sigma-Aldrich", "Merck"],
                    "regulatory_info": {"EPA": "regulated", "REACH": "registered"},
                    "property_data": {
                        "melting_point": -114.1,
                        "boiling_point": 78.37,
                        "density": 0.789,
                    },
                },
            ]
        )
        return instance


# ============================================================
# 适配器 2: WebOfScienceAdapter — Clarivate 引文索引
# ============================================================


class WebOfScienceAdapter(RESTAdapter):
    """Clarivate Web of Science 引文索引适配器.

    数据源: Clarivate Web of Science API (https://api.clarivate.com/api/wos)
    协议: REST / HTTPS, 认证 Bearer Token (API Key)
    限流: 15 req/min
    缓存: 7200s (2 小时)

    Web of Science 是全球最权威的学术引文索引数据库, 收录 1900 年至今的
    21,000+ 高质量期刊, 提供引文追踪、影响因子、研究前沿分析等功能。
    其收录标准严格 (SCI / SSCI / AHCI), 被引数据是科研评价的核心指标。

    支持的流 (streams):
    - "records": 文献记录 (标题 / 作者 / 摘要 / 关键词 / 引文)
    - "citations": 施引文献 (引用指定文献的后续论文)
    - "references": 参考文献 (指定文献引用的前序论文)

    搜索维度:
    - 按 topic (主题关键词)
    - 按 author (作者)
    - 按 DOI
    - 按 WOS ID (UT 唯一标识符)

    SchemaMapper 映射 (源字段 → L3 标准字段):
    - wos_id → entity_id (主键)
    - title → entity_name
    - record_type → entity_type (默认 "academic_paper")
    - doi → identifiers (DOI 标识符)
    - abstract → properties (摘要作为属性)
    - doi → source_uri (DOI URL)
    - title → description
    - doi / title / authors / abstract / publication_year → 同名领域字段
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 Web of Science 适配器.

        Args:
            config: 连接器配置
            auth_token: Clarivate API Key (Bearer Token)
            **kwargs: 传递给 RESTAdapter 的额外参数
        """
        super().__init__(
            config,
            search_endpoint=kwargs.pop(
                "search_endpoint", "/api/wos/search"
            ),
            fetch_endpoint=kwargs.pop(
                "fetch_endpoint", "/api/wos/records/{id}"
            ),
            page_size=kwargs.pop("page_size", 15),
            auth_type="bearer",
            auth_token=auth_token,
            **kwargs,
        )

        mapper = self._schema_mapper
        mapper.add_mapping(
            FieldMapping(
                source_field="wos_id",
                target_field="entity_id",
                required=True,
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="title", target_field="entity_name", required=True
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="record_type",
                target_field="entity_type",
                default_value="academic_paper",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="identifiers")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="properties")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="doi",
                target_field="source_uri",
                transform="to_lower",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="description")
        )
        # 论文领域专属字段
        mapper.add_mapping(FieldMapping(source_field="doi", target_field="doi"))
        mapper.add_mapping(FieldMapping(source_field="title", target_field="title"))
        mapper.add_mapping(
            FieldMapping(source_field="authors", target_field="authors")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="publication_year",
                target_field="publication_date",
                transform="iso_datetime",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="keywords", target_field="keywords"
            )
        )

    def _do_spec(self) -> AdapterSpec:
        """声明 Web of Science 适配器规范."""
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=_rest_caps(),
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "auth_token": {"type": "string"},
                    "page_size": {"type": "integer", "default": 15},
                },
                "required": ["base_url", "auth_token"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://clarivate.com/developer/wos-api",
            changelog={"1.0.0": "初始版本, 支持 records / citations / references 流"},
        )

    def _get_schema(self) -> DataSourceSchema:
        """返回 records 流的 Schema."""
        return DataSourceSchema(
            stream_name="records",
            fields=[
                SchemaField(
                    name="wos_id",
                    data_type="string",
                    nullable=False,
                    primary_key=True,
                    description="Web of Science UT 唯一标识符",
                ),
                SchemaField(
                    name="ut",
                    data_type="string",
                    description="UT accession number (WOS ID 别名)",
                ),
                SchemaField(
                    name="title",
                    data_type="string",
                    nullable=False,
                    description="文献标题",
                ),
                SchemaField(
                    name="authors",
                    data_type="array",
                    description="作者列表",
                ),
                SchemaField(
                    name="source_title",
                    data_type="string",
                    description="期刊 / 来源名称",
                ),
                SchemaField(
                    name="publication_year",
                    data_type="integer",
                    description="出版年份",
                ),
                SchemaField(
                    name="doi",
                    data_type="string",
                    description="DOI 标识符",
                    format="doi",
                ),
                SchemaField(
                    name="abstract",
                    data_type="string",
                    description="文献摘要",
                ),
                SchemaField(
                    name="keywords",
                    data_type="array",
                    description="作者关键词",
                ),
                SchemaField(
                    name="categories",
                    data_type="array",
                    description="学科分类",
                ),
                SchemaField(
                    name="citation_count",
                    data_type="integer",
                    description="被引次数",
                ),
                SchemaField(
                    name="reference_count",
                    data_type="integer",
                    description="参考文献数",
                ),
                SchemaField(
                    name="times_cited",
                    data_type="integer",
                    description="总被引次数",
                ),
                SchemaField(
                    name="journal_impact_factor",
                    data_type="float",
                    description="期刊影响因子 (JCR)",
                ),
                SchemaField(
                    name="research_areas",
                    data_type="array",
                    description="研究领域",
                ),
            ],
            primary_keys=["wos_id"],
            cursor_field="publication_year",
            description="Web of Science 文献记录 (1900+, 21K+ 期刊)",
            metadata={"source": "Clarivate", "authority_tier": "T2"},
        )

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 WoS 搜索 URL.

        支持按 topic / author / doi / wos_id 搜索, 通过 search_field 区分。
        """
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "q": query,
            "search_field": kwargs.get("search_field", "topic"),
            "count": kwargs.get("limit", self._page_size),
            "first_record": kwargs.get("offset", 1),
        }
        return url, params, self._build_auth_headers()

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 WoS 记录获取 URL (按 WOS ID / UT)."""
        endpoint = self._fetch_endpoint.replace("{id}", resource_id)
        url = f"{self.config.base_url}{endpoint}"
        return url, {}, self._build_auth_headers()

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 WoS API 响应.

        WoS 返回格式: {"Data": {"Records": {"records": [...]}}}
        或简化格式: {"records": [...]}
        """
        if isinstance(data, dict):
            # 标准 WoS 嵌套结构
            records = (
                data.get("Data", {})
                .get("Records", {})
                .get("records", {})
                .get("REC", [])
            )
            if records:
                return records
            # 简化结构
            for key in ("records", "Data", "results", "items"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    inner = val.get("records") or val.get("REC")
                    if isinstance(inner, list):
                        return inner
            return [data]
        if isinstance(data, list):
            return data
        return []

    @classmethod
    def create(cls, auth_token: str = "") -> WebOfScienceAdapter:
        """创建预配置的 Web of Science 适配器实例 (含模拟数据).

        Args:
            auth_token: Clarivate API Key

        Returns:
            预配置的 WebOfScienceAdapter 实例
        """
        config = ConnectorConfig(
            id="wos-clarivate",
            name="Web of Science (Clarivate)",
            tier=ConnectorTier.INDUSTRY,
            protocol=ConnectorProtocol.REST,
            base_url="https://api.clarivate.com",
            auth_config={"type": "bearer"},
            rate_limit=15,
            cache_ttl=7200,
            version="1.0.0",
            tags=["citation", "academic", "impact-factor", "paid"],
            description="Clarivate Web of Science 引文索引 (1900+, 21K+ 期刊)",
        )
        instance = cls(config, auth_token=auth_token)
        instance.set_mock_data(
            [
                {
                    "wos_id": "WOS:000123456789",
                    "ut": "000123456789",
                    "title": "Rare-earth-doped upconversion nanomaterials"
                    " for bioimaging applications",
                    "authors": ["Zhang, Y.", "Li, X.", "Wang, H."],
                    "source_title": "Nature Photonics",
                    "publication_year": 2023,
                    "doi": "10.1038/s41566-023-01234-5",
                    "abstract": "Rare-earth-doped upconversion nanoparticles"
                    " (UCNPs) convert near-infrared light to visible emission"
                    " via multiphoton processes...",
                    "keywords": ["upconversion", "rare earth", "nanomaterials"],
                    "categories": ["Physics", "Optics"],
                    "citation_count": 45,
                    "reference_count": 62,
                    "times_cited": 45,
                    "journal_impact_factor": 35.5,
                    "research_areas": ["Physics", "Materials Science"],
                },
                {
                    "wos_id": "WOS:000987654321",
                    "ut": "000987654321",
                    "title": "Dy3+ ion luminescence properties in fluoride"
                    " glass ceramics",
                    "authors": ["Chen, L.", "Liu, J."],
                    "source_title": "Journal of Luminescence",
                    "publication_year": 2022,
                    "doi": "10.1016/j.jlumin.2022.119876",
                    "abstract": "The spectroscopic properties of Dy3+ ions in"
                    " fluoride glass ceramics were investigated...",
                    "keywords": ["Dy3+", "luminescence", "fluoride glass"],
                    "categories": ["Physics", "Materials Science"],
                    "citation_count": 12,
                    "reference_count": 38,
                    "times_cited": 12,
                    "journal_impact_factor": 3.6,
                    "research_areas": ["Physics", "Chemistry"],
                },
            ]
        )
        return instance


# ============================================================
# 适配器 3: SciFinderAdapter — CAS SciFinder 文献检索
# ============================================================


class SciFinderAdapter(RESTAdapter):
    """CAS SciFinder 文献检索适配器.

    数据源: CAS SciFinder (封装 SciFinder-n 接口, 后端复用 CAS API)
    协议: REST / HTTPS, 认证 Bearer Token
    限流: 5 req/min (极其受限, 配额最严格的行业源之一)
    缓存: 7200s (2 小时)

    SciFinder 是化学领域最权威的文献检索工具, 由 CAS (American Chemical
    Society 旗下) 提供。它整合了 CAplus 文献数据库 (1.6 亿+ 文献) 和 CAS
    Registry 物质数据库, 支持按研究主题、物质名称、反应方案进行检索。
    SciFinder-n 是其新一代 Web 界面, 本适配器封装其 API 后端。

    与 CASAdapter 的区别:
    - CASAdapter 聚焦物质注册数据 (CAS RN / 结构 / 性质)
    - SciFinderAdapter 聚焦文献检索 (论文 / 专利 / 反应方案关联文献)

    支持的流 (streams):
    - "literature": 文献记录 (论文 / 专利 / 图书章节)
    - "substances": 物质记录 (关联文献的物质)
    - "reactions": 反应记录 (反应方案 + 关联文献)

    搜索维度:
    - 按 research_topic (研究主题)
    - 按 substance_name (物质名称)
    - 按 reaction_scheme (反应方案)

    SchemaMapper 映射 (源字段 → L3 标准字段):
    - record_id → entity_id (主键)
    - title → entity_name
    - record_type → entity_type (默认 "scientific_literature")
    - doi → identifiers
    - abstract → properties
    - doi → source_uri
    - title → description
    - cas_rn / title / authors / abstract → 同名领域字段
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 SciFinder 适配器.

        Args:
            config: 连接器配置
            auth_token: SciFinder API Bearer Token
            **kwargs: 传递给 RESTAdapter 的额外参数
        """
        super().__init__(
            config,
            search_endpoint=kwargs.pop(
                "search_endpoint", "/api/scifinder/literature/search"
            ),
            fetch_endpoint=kwargs.pop(
                "fetch_endpoint", "/api/scifinder/literature/{id}"
            ),
            page_size=kwargs.pop("page_size", 5),
            auth_type="bearer",
            auth_token=auth_token,
            **kwargs,
        )

        mapper = self._schema_mapper
        mapper.add_mapping(
            FieldMapping(
                source_field="record_id",
                target_field="entity_id",
                required=True,
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="title", target_field="entity_name", required=True
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="record_type",
                target_field="entity_type",
                default_value="scientific_literature",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="identifiers")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="properties")
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="description")
        )
        # 文献领域专属字段
        mapper.add_mapping(
            FieldMapping(source_field="record_id", target_field="record_id")
        )
        mapper.add_mapping(FieldMapping(source_field="title", target_field="title"))
        mapper.add_mapping(
            FieldMapping(source_field="authors", target_field="authors")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="publication_date",
                target_field="publication_date",
                transform="iso_datetime",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="source_title", target_field="source_title"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="cas_rn",
                target_field="cas_rn",
                transform="to_upper",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="patent_info", target_field="patent_info"
            )
        )

    def _do_spec(self) -> AdapterSpec:
        """声明 SciFinder 适配器规范 (限流最严格 5/min)."""
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=_rest_caps(),
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "auth_token": {"type": "string"},
                    "page_size": {"type": "integer", "default": 5},
                },
                "required": ["base_url", "auth_token"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://scifinder.cas.org/documentation",
            changelog={
                "1.0.0": "初始版本, 支持 literature / substances / reactions 流",
            },
        )

    def _get_schema(self) -> DataSourceSchema:
        """返回 literature 流的 Schema."""
        return DataSourceSchema(
            stream_name="literature",
            fields=[
                SchemaField(
                    name="record_id",
                    data_type="string",
                    nullable=False,
                    primary_key=True,
                    description="SciFinder 文献记录 ID",
                ),
                SchemaField(
                    name="title",
                    data_type="string",
                    nullable=False,
                    description="文献标题",
                ),
                SchemaField(
                    name="authors",
                    data_type="array",
                    description="作者列表",
                ),
                SchemaField(
                    name="abstract",
                    data_type="string",
                    description="文献摘要",
                ),
                SchemaField(
                    name="publication_date",
                    data_type="datetime",
                    description="出版日期",
                    format="date",
                ),
                SchemaField(
                    name="source_title",
                    data_type="string",
                    description="期刊 / 来源名称",
                ),
                SchemaField(
                    name="cas_rn",
                    data_type="string",
                    description="关联 CAS Registry Number",
                    format="cas-rn",
                ),
                SchemaField(
                    name="reaction_details",
                    data_type="object",
                    description="反应详情 (若为反应文献)",
                ),
                SchemaField(
                    name="substance_details",
                    data_type="object",
                    description="物质详情 (若关联物质)",
                ),
                SchemaField(
                    name="patent_info",
                    data_type="object",
                    description="专利信息 (若为专利文献)",
                ),
                SchemaField(
                    name="cited_references",
                    data_type="array",
                    description="引用的参考文献列表",
                ),
            ],
            primary_keys=["record_id"],
            cursor_field="publication_date",
            description="CAS SciFinder 文献记录 (1.6 亿+ 文献, CAplus)",
            metadata={"source": "CAS-SciFinder", "authority_tier": "T2"},
        )

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 SciFinder 文献搜索 URL.

        支持按 research_topic / substance_name / reaction_scheme 搜索。
        """
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "query": query,
            "search_type": kwargs.get("search_type", "research_topic"),
            "limit": kwargs.get("limit", self._page_size),
            "offset": kwargs.get("offset", 0),
        }
        return url, params, self._build_auth_headers()

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 SciFinder 文献获取 URL (按 record_id)."""
        endpoint = self._fetch_endpoint.replace("{id}", resource_id)
        url = f"{self.config.base_url}{endpoint}"
        return url, {}, self._build_auth_headers()

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 SciFinder API 响应.

        SciFinder 返回格式: {"results": {"literature": [...]}}
        或简化格式: {"literature": [...]} / {"results": [...]}
        """
        if isinstance(data, dict):
            results_obj = data.get("results", data)
            if isinstance(results_obj, dict):
                for key in ("literature", "records", "items"):
                    if key in results_obj and isinstance(results_obj[key], list):
                        return results_obj[key]
            if isinstance(results_obj, list):
                return results_obj
            return [data]
        if isinstance(data, list):
            return data
        return []

    @classmethod
    def create(cls, auth_token: str = "") -> SciFinderAdapter:
        """创建预配置的 SciFinder 适配器实例 (含模拟数据).

        Args:
            auth_token: SciFinder API Bearer Token

        Returns:
            预配置的 SciFinderAdapter 实例
        """
        config = ConnectorConfig(
            id="scifinder-cas",
            name="CAS SciFinder",
            tier=ConnectorTier.INDUSTRY,
            protocol=ConnectorProtocol.REST,
            base_url="https://scifinder.cas.org",
            auth_config={"type": "bearer"},
            rate_limit=5,
            cache_ttl=7200,
            version="1.0.0",
            tags=["chemistry", "literature", "scifinder", "paid"],
            description="CAS SciFinder 文献检索 (1.6 亿+ 文献, CAplus)",
        )
        instance = cls(config, auth_token=auth_token)
        instance.set_mock_data(
            [
                {
                    "record_id": "SF-2023-00123456",
                    "title": "Spectroscopic analysis of Dy3+-doped"
                    " lanthanide complexes",
                    "record_type": "scientific_literature",
                    "authors": ["Wang, Q.", "Zhao, M.", "Sun, Y."],
                    "abstract": "The photoluminescence properties of Dy3+-doped"
                    " lanthanide complexes were systematically studied using"
                    " excitation and emission spectroscopy...",
                    "publication_date": "2023-06-15",
                    "source_title": "Journal of Alloys and Compounds",
                    "cas_rn": "10025-74-8",
                    "reaction_details": {},
                    "substance_details": {
                        "dopant": "Dy3+",
                        "host": "Y2O3",
                    },
                    "patent_info": {},
                    "cited_references": ["10.1016/j.jallcom.2022.01.001"],
                },
                {
                    "record_id": "SF-2022-00789101",
                    "title": "Synthesis and characterization of"
                    " Dy3+-activated phosphors for white LEDs",
                    "record_type": "scientific_literature",
                    "authors": ["Kim, S.", "Park, J."],
                    "abstract": "A series of Dy3+-activated phosphor materials"
                    " were synthesized via solid-state reaction method and"
                    " their luminescence properties evaluated...",
                    "publication_date": "2022-11-20",
                    "source_title": "Optical Materials",
                    "cas_rn": "7440-58-6",
                    "reaction_details": {
                        "method": "solid-state reaction",
                        "temperature": "1200 C",
                    },
                    "substance_details": {
                        "dopant": "Dy3+",
                        "host": "BaMgAl10O17",
                    },
                    "patent_info": {},
                    "cited_references": [],
                },
            ]
        )
        return instance


# ============================================================
# 适配器 4: ReaxysAdapter — Elsevier Reaxys 反应数据库 (GraphQL)
# ============================================================


class ReaxysAdapter(GraphQLAdapter):
    """Elsevier Reaxys 反应数据库适配器 (GraphQL).

    数据源: Reaxys API (https://www.reaxys.com/api) — Elsevier
    协议: GraphQL / HTTPS, 认证 Bearer Token (Elsevier API Key)
    限流: 10 req/min
    缓存: 7200s (2 小时)

    Reaxys 是 Elsevier 旗下的化学反应数据库, 整合了 Beilstein (有机化学) 和
    Gmelin (无机化学) 两大经典数据库, 收录 1.2 亿+ 物质、5000 万+ 反应和
    5 亿+ 性质数据点。其 GraphQL 接口支持灵活的物质 / 反应 / 性质查询。

    支持的流 (streams):
    - "substances": 物质记录 (Reaxys ID / 分子式 / CAS RN / 性质)
    - "reactions": 反应记录 (反应方程 / 条件 / 产率 / 文档引用)
    - "documents": 文献记录 (关联物质 / 反应的原始文献)
    - "properties": 性质记录 (物理化学性质数据点)

    搜索维度:
    - 按 substance (物质查询)
    - 按 reaction (反应查询)
    - 按 property (性质查询)

    GraphQL 查询示例 (物质搜索):
        query SearchSubstances($query: String!, $limit: Int, $after: String) {
            substances(search: $query, first: $limit, after: $after) {
                edges { node { reaxysId name molecularFormula casRn } }
                pageInfo { endCursor hasNextPage }
            }
        }

    SchemaMapper 映射 (源字段 → L3 标准字段):
    - reaxys_id → entity_id (主键)
    - substance_name → entity_name
    - record_type → entity_type (默认 "chemical_substance")
    - cas_rn → identifiers
    - property_value → properties
    - document_reference → source_uri
    - substance_name → description
    - cas_rn / molecular_formula → 同名领域字段
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 Reaxys 适配器.

        Args:
            config: 连接器配置
            auth_token: Elsevier API Key (Bearer Token)
            **kwargs: 传递给 GraphQLAdapter 的额外参数
        """
        super().__init__(
            config,
            endpoint=kwargs.pop("endpoint", "/api/graphql"),
            auth_type="bearer",
            auth_token=auth_token,
            **kwargs,
        )

        mapper = self._schema_mapper
        mapper.add_mapping(
            FieldMapping(
                source_field="reaxys_id",
                target_field="entity_id",
                required=True,
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="substance_name",
                target_field="entity_name",
                required=True,
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="record_type",
                target_field="entity_type",
                default_value="chemical_substance",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="cas_rn",
                target_field="identifiers",
                transform="to_upper",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="property_value", target_field="properties"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="document_reference", target_field="source_uri"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="substance_name", target_field="description"
            )
        )
        # 化学物质领域专属字段
        mapper.add_mapping(
            FieldMapping(
                source_field="reaxys_id", target_field="reaxys_id"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="substance_name", target_field="substance_name"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="molecular_formula",
                target_field="molecular_formula",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="cas_rn",
                target_field="cas_rn",
                transform="to_upper",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="reaction_equation",
                target_field="reaction_equation",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="yield_percent",
                target_field="yield_percent",
                transform="parse_float",
            )
        )

    def _do_spec(self) -> AdapterSpec:
        """声明 Reaxys 适配器规范 (GRAPHQL + INCREMENTAL)."""
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.AUTHENTICATE
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
            | AdapterCapability.INCREMENTAL
        )
        return AdapterSpec(
            adapter_type=DataSourceType.GRAPHQL,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "endpoint": {"type": "string", "default": "/api/graphql"},
                    "auth_token": {"type": "string"},
                },
                "required": ["base_url", "auth_token"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://www.reaxys.com/reaxys/api",
            changelog={
                "1.0.0": "初始版本, 支持 substances / reactions / documents /"
                " properties 流",
            },
        )

    def _get_schema(self) -> DataSourceSchema:
        """返回 substances 流的 Schema."""
        return DataSourceSchema(
            stream_name="substances",
            fields=[
                SchemaField(
                    name="reaxys_id",
                    data_type="string",
                    nullable=False,
                    primary_key=True,
                    description="Reaxys 物质唯一标识符",
                ),
                SchemaField(
                    name="substance_name",
                    data_type="string",
                    nullable=False,
                    description="物质名称",
                ),
                SchemaField(
                    name="molecular_formula",
                    data_type="string",
                    description="分子式",
                ),
                SchemaField(
                    name="cas_rn",
                    data_type="string",
                    description="CAS Registry Number",
                    format="cas-rn",
                ),
                SchemaField(
                    name="reaction_equation",
                    data_type="string",
                    description="反应方程式",
                ),
                SchemaField(
                    name="reaction_conditions",
                    data_type="string",
                    description="反应条件 (温度 / 催化剂 / 溶剂)",
                ),
                SchemaField(
                    name="yield_percent",
                    data_type="float",
                    description="反应产率 (%)",
                ),
                SchemaField(
                    name="document_reference",
                    data_type="string",
                    description="文献引用标识",
                ),
                SchemaField(
                    name="property_name",
                    data_type="string",
                    description="性质名称 (如 melting_point)",
                ),
                SchemaField(
                    name="property_value",
                    data_type="string",
                    description="性质数值",
                ),
                SchemaField(
                    name="property_unit",
                    data_type="string",
                    description="性质单位",
                ),
            ],
            primary_keys=["reaxys_id"],
            cursor_field="reaxys_id",
            description="Reaxys 物质 / 反应记录 (1.2 亿+ 物质, 5000 万+ 反应)",
            metadata={"source": "Elsevier-Reaxys", "authority_tier": "T2"},
        )

    def _build_search_query(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        """构建 Reaxys GraphQL 物质搜索查询.

        使用 Relay Cursor Connection 分页规范。

        Returns:
            (query_str, variables)
        """
        gql = """
        query SearchSubstances(
            $query: String!
            $limit: Int
            $after: String
        ) {
            substances(search: $query, first: $limit, after: $after) {
                edges {
                    node {
                        reaxysId
                        name
                        molecularFormula
                        casRn
                        molecularWeight
                        reactions {
                            equation
                            conditions
                            yield
                            documentReference
                        }
                        properties {
                            name
                            value
                            unit
                        }
                    }
                }
                pageInfo {
                    endCursor
                    hasNextPage
                }
                totalCount
            }
        }
        """
        variables: dict[str, Any] = {
            "query": query,
            "limit": kwargs.get("limit", 20),
            "after": kwargs.get("after", None),
        }
        return gql, variables

    def _build_fetch_query(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any]]:
        """构建 Reaxys GraphQL 单物质获取查询."""
        gql = """
        query FetchSubstance($id: ID!) {
            substance(reaxysId: $id) {
                reaxysId
                name
                molecularFormula
                casRn
                molecularWeight
                reactions {
                    equation
                    conditions
                    yield
                    documentReference
                }
                properties {
                    name
                    value
                    unit
                }
            }
        }
        """
        return gql, {"id": resource_id}

    def _parse_graphql_response(
        self, data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """解析 Reaxys GraphQL 响应.

        Reaxys GraphQL 返回格式:
            {"data": {"substances": {"edges": [{"node": {...}}, ...]}}}

        本方法提取 edges[].node 并将 camelCase 字段转为 snake_case。
        """
        if "data" not in data or data["data"] is None:
            return []
        data_content = data["data"]

        # 物质搜索: data.substances.edges[].node
        substances = data_content.get("substances")
        if isinstance(substances, dict):
            edges = substances.get("edges", [])
            records: list[dict[str, Any]] = []
            for edge in edges:
                node = edge.get("node", {}) if isinstance(edge, dict) else {}
                records.append(self._normalize_substance_node(node))
            return records

        # 单物质获取: data.substance
        substance = data_content.get("substance")
        if isinstance(substance, dict):
            return [self._normalize_substance_node(substance)]

        # 通用回退
        for key in ("search", "results", "items", "nodes"):
            val = data_content.get(key)
            if isinstance(val, list):
                return val
        if isinstance(data_content, dict):
            return [data_content]
        return []

    @staticmethod
    def _normalize_substance_node(
        node: dict[str, Any]
    ) -> dict[str, Any]:
        """将 GraphQL camelCase 节点转换为 snake_case 记录.

        Args:
            node: GraphQL node (camelCase 字段)

        Returns:
            snake_case 字段的标准记录
        """
        record: dict[str, Any] = {
            "reaxys_id": node.get("reaxysId", ""),
            "substance_name": node.get("name", ""),
            "molecular_formula": node.get("molecularFormula", ""),
            "cas_rn": node.get("casRn", ""),
            "record_type": "chemical_substance",
            # 反应信息 (默认空, 确保输出 Schema 一致)
            "reaction_equation": "",
            "reaction_conditions": "",
            "yield_percent": None,
            "document_reference": "",
            # 性质信息 (默认空, 确保输出 Schema 一致)
            "property_name": "",
            "property_value": "",
            "property_unit": "",
        }
        # 反应信息: 取第一条反应记录填充
        reactions = node.get("reactions", [])
        if reactions and isinstance(reactions, list):
            first_rxn = reactions[0] if reactions else {}
            record["reaction_equation"] = first_rxn.get("equation", "")
            record["reaction_conditions"] = first_rxn.get("conditions", "")
            record["yield_percent"] = first_rxn.get("yield")
            record["document_reference"] = first_rxn.get(
                "documentReference", ""
            )
        # 性质信息: 取第一条性质记录填充
        properties = node.get("properties", [])
        if properties and isinstance(properties, list):
            first_prop = properties[0] if properties else {}
            record["property_name"] = first_prop.get("name", "")
            record["property_value"] = first_prop.get("value", "")
            record["property_unit"] = first_prop.get("unit", "")
        return record

    @classmethod
    def create(cls, auth_token: str = "") -> ReaxysAdapter:
        """创建预配置的 Reaxys 适配器实例 (含模拟数据).

        Args:
            auth_token: Elsevier API Key

        Returns:
            预配置的 ReaxysAdapter 实例
        """
        config = ConnectorConfig(
            id="reaxys-elsevier",
            name="Reaxys (Elsevier)",
            tier=ConnectorTier.INDUSTRY,
            protocol=ConnectorProtocol.GRAPHQL,
            base_url="https://www.reaxys.com",
            auth_config={"type": "bearer"},
            rate_limit=10,
            cache_ttl=7200,
            version="1.0.0",
            tags=["chemistry", "reactions", "graphql", "paid"],
            description="Elsevier Reaxys 反应数据库 (1.2 亿+ 物质)",
        )
        instance = cls(config, auth_token=auth_token)
        instance.set_mock_data(
            [
                {
                    "reaxys_id": "RX-0001234567",
                    "substance_name": "Dysprosium(III) chloride",
                    "record_type": "chemical_substance",
                    "molecular_formula": "Cl3Dy",
                    "cas_rn": "10025-74-8",
                    "reaction_equation": "Dy2O3 + 6 HCl -> 2 DyCl3 + 3 H2O",
                    "reaction_conditions": "200 C, HCl gas, anhydrous",
                    "yield_percent": 95.0,
                    "document_reference": "Beilstein:IV-1234",
                    "property_name": "melting_point",
                    "property_value": "718",
                    "property_unit": "C",
                },
                {
                    "reaxys_id": "RX-0007654321",
                    "substance_name": "Dysprosium oxide",
                    "record_type": "chemical_substance",
                    "molecular_formula": "Dy2O3",
                    "cas_rn": "1308-87-8",
                    "reaction_equation": "2 Dy + 3/2 O2 -> Dy2O3",
                    "reaction_conditions": "room temperature, air",
                    "yield_percent": 100.0,
                    "document_reference": "Gmelin:5678",
                    "property_name": "density",
                    "property_value": "7.81",
                    "property_unit": "g/cm3",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 5: GooglePatentsAdapter — Google Patents 专利数据库
# ============================================================


class GooglePatentsAdapter(RESTAdapter):
    """Google Patents 专利数据库适配器.

    数据源: Google Patents (https://patents.google.com)
    协议: REST / HTTPS, 认证 "none" (公开) 或 "api_key" (Google Patents
          Public Datasets, 更高限额)
    限流: 50 req/min (公开访问, 较宽松)
    缓存: 3600s (1 小时)

    Google Patents 覆盖 1.2 亿+ 专利文献, 来自 100+ 专利局 (USPTO / EPO /
    JPO / SIPO 等), 时间跨度从 1782 年至今。它提供全文搜索、引文网络、
    法律状态追踪和分类码 (IPC / CPC) 检索。公开接口免费, 通过 Google Cloud
    BigQuery 的 Google Patents Public Datasets 可获得更高限额的结构化访问。

    支持的流 (streams):
    - "patents": 已授权专利记录
    - "applications": 专利申请记录

    搜索维度:
    - 按 patent_number (专利号, 如 "US-12345678-A")
    - 按 inventor (发明人)
    - 按 assignee (受让人 / 专利权人)
    - 按 classification (IPC / CPC 分类码)
    - 按 keyword (全文关键词)

    SchemaMapper 映射 (源字段 → L3 标准字段):
    - patent_number → entity_id (主键)
    - title → entity_name
    - record_type → entity_type (默认 "patent")
    - patent_number → identifiers
    - claims → properties
    - patent_number → source_uri (Google Patents URL)
    - abstract → description
    - patent_number / inventors / assignees / classification_codes → 同名字段
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        auth_token: str = "",
        auth_type: str = "none",
        **kwargs: Any,
    ) -> None:
        """初始化 Google Patents 适配器.

        Args:
            config: 连接器配置
            auth_token: API Key (可选, 用于 Public Datasets 高限额访问)
            auth_type: 认证类型 ("none" 公开 / "api_key" 高限额)
            **kwargs: 传递给 RESTAdapter 的额外参数
        """
        super().__init__(
            config,
            search_endpoint=kwargs.pop(
                "search_endpoint", "/api/patents/search"
            ),
            fetch_endpoint=kwargs.pop(
                "fetch_endpoint", "/api/patents/{id}"
            ),
            page_size=kwargs.pop("page_size", 50),
            auth_type=auth_type,
            auth_token=auth_token,
            **kwargs,
        )

        mapper = self._schema_mapper
        mapper.add_mapping(
            FieldMapping(
                source_field="patent_number",
                target_field="entity_id",
                required=True,
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="title", target_field="entity_name", required=True
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="record_type",
                target_field="entity_type",
                default_value="patent",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="patent_number", target_field="identifiers"
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="claims", target_field="properties")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="patent_number", target_field="source_uri"
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="description")
        )
        # 专利领域专属字段
        mapper.add_mapping(
            FieldMapping(
                source_field="patent_number", target_field="patent_number"
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="title")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="inventors", target_field="inventors"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="assignees", target_field="assignees"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="filing_date",
                target_field="publication_date",
                transform="iso_datetime",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="classification_codes",
                target_field="classification_codes",
                transform="split_comma",
            )
        )

    def _do_spec(self) -> AdapterSpec:
        """声明 Google Patents 适配器规范 (公开 / 可选 API Key)."""
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=_rest_caps(),
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "auth_token": {
                        "type": "string",
                        "description": "可选 API Key (Public Datasets)",
                    },
                    "auth_type": {
                        "type": "string",
                        "enum": ["none", "api_key"],
                        "default": "none",
                    },
                    "page_size": {"type": "integer", "default": 50},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://patents.google.com",
            changelog={
                "1.0.0": "初始版本, 支持 patents / applications 流",
            },
        )

    def _get_schema(self) -> DataSourceSchema:
        """返回 patents 流的 Schema."""
        return DataSourceSchema(
            stream_name="patents",
            fields=[
                SchemaField(
                    name="patent_number",
                    data_type="string",
                    nullable=False,
                    primary_key=True,
                    description="专利号 (如 US-12345678-A)",
                ),
                SchemaField(
                    name="title",
                    data_type="string",
                    nullable=False,
                    description="专利标题",
                ),
                SchemaField(
                    name="abstract",
                    data_type="string",
                    description="专利摘要",
                ),
                SchemaField(
                    name="inventors",
                    data_type="array",
                    description="发明人列表",
                ),
                SchemaField(
                    name="assignees",
                    data_type="array",
                    description="受让人 / 专利权人列表",
                ),
                SchemaField(
                    name="filing_date",
                    data_type="datetime",
                    description="申请日期",
                    format="date",
                ),
                SchemaField(
                    name="publication_date",
                    data_type="datetime",
                    description="公开日期",
                    format="date",
                ),
                SchemaField(
                    name="grant_date",
                    data_type="datetime",
                    description="授权日期",
                    format="date",
                ),
                SchemaField(
                    name="classification_codes",
                    data_type="array",
                    description="IPC / CPC 分类码列表",
                ),
                SchemaField(
                    name="cited_by",
                    data_type="array",
                    description="施引专利列表",
                ),
                SchemaField(
                    name="references",
                    data_type="array",
                    description="引用的在前专利列表",
                ),
                SchemaField(
                    name="claims",
                    data_type="string",
                    description="权利要求文本",
                ),
                SchemaField(
                    name="description",
                    data_type="string",
                    description="说明书全文",
                ),
                SchemaField(
                    name="legal_status",
                    data_type="string",
                    description="法律状态 (active / expired / lapsed)",
                    enum_values=["active", "expired", "lapsed", "pending"],
                ),
                SchemaField(
                    name="country_code",
                    data_type="string",
                    description="国家代码 (US / EP / JP / CN ...)",
                    max_length=2,
                ),
            ],
            primary_keys=["patent_number"],
            cursor_field="publication_date",
            description="Google Patents 专利记录 (1.2 亿+ 专利, 100+ 专利局)",
            metadata={"source": "Google-Patents", "authority_tier": "T2"},
        )

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 Google Patents 搜索 URL.

        支持按 patent_number / inventor / assignee / classification /
        keyword 搜索, 通过 search_field 区分。API Key 作为查询参数传递。
        """
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "q": query,
            "search_field": kwargs.get("search_field", "keyword"),
            "num": kwargs.get("limit", self._page_size),
            "page": kwargs.get("page", 1),
        }
        # API Key 通过查询参数传递 (Google Patents Public Datasets 约定)
        if self._auth_type == "api_key" and self._auth_token:
            params["key"] = self._auth_token
        return url, params, self._build_auth_headers()

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 Google Patents 获取 URL (按 patent_number)."""
        endpoint = self._fetch_endpoint.replace("{id}", resource_id)
        url = f"{self.config.base_url}{endpoint}"
        params: dict[str, Any] = {}
        if self._auth_type == "api_key" and self._auth_token:
            params["key"] = self._auth_token
        return url, params, self._build_auth_headers()

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 Google Patents API 响应.

        返回格式: {"patents": [...], "total_results": N} 或 [...]
        """
        if isinstance(data, dict):
            for key in ("patents", "results", "items", "data"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            return data
        return []

    @classmethod
    def create(cls, auth_token: str = "") -> GooglePatentsAdapter:
        """创建预配置的 Google Patents 适配器实例 (含模拟数据).

        Args:
            auth_token: 可选 API Key (空则使用公开访问)

        Returns:
            预配置的 GooglePatentsAdapter 实例
        """
        auth_type = "api_key" if auth_token else "none"
        config = ConnectorConfig(
            id="google-patents",
            name="Google Patents",
            tier=ConnectorTier.INDUSTRY,
            protocol=ConnectorProtocol.REST,
            base_url="https://patents.google.com",
            auth_config={"type": auth_type},
            rate_limit=50,
            cache_ttl=3600,
            version="1.0.0",
            tags=["patent", "public", "free-tier"],
            description="Google Patents 专利数据库 (1.2 亿+ 专利)",
        )
        instance = cls(
            config, auth_token=auth_token, auth_type=auth_type
        )
        instance.set_mock_data(
            [
                {
                    "patent_number": "US-11234567-B2",
                    "title": "Rare earth doped phosphor for LED"
                    " lighting devices",
                    "abstract": "A phosphor material doped with rare earth"
                    " ions including Dy3+ for use in white LED lighting...",
                    "record_type": "patent",
                    "inventors": ["Yamamoto, T.", "Tanaka, K."],
                    "assignees": ["Nichia Corporation"],
                    "filing_date": "2020-03-15",
                    "publication_date": "2022-01-18",
                    "grant_date": "2022-02-01",
                    "classification_codes": [
                        "C09K11/77",
                        "H01L33/50",
                        "F21K9/60",
                    ],
                    "cited_by": ["US-11345678-B2", "US-11456789-B2"],
                    "references": ["US-99887766-B1", "US-99887765-B1"],
                    "claims": "1. A phosphor composition comprising a host"
                    " material and Dy3+ as an activator...",
                    "description": "The present invention relates to rare"
                    " earth doped phosphor materials...",
                    "legal_status": "active",
                    "country_code": "US",
                },
                {
                    "patent_number": "CN-109876543-A",
                    "title": "Preparation method of Dy3+ doped"
                    " luminescent glass",
                    "abstract": "The invention discloses a preparation"
                    " method of Dy3+ doped luminescent glass ceramic...",
                    "record_type": "patent",
                    "inventors": ["Wang, L.", "Chen, H.", "Zhao, B."],
                    "assignees": ["Chinese Academy of Sciences"],
                    "filing_date": "2019-05-20",
                    "publication_date": "2019-09-10",
                    "grant_date": "",
                    "classification_codes": [
                        "C03C4/12",
                        "C03C10/02",
                    ],
                    "cited_by": [],
                    "references": ["CN-108765432-A"],
                    "claims": "1. A preparation method of Dy3+ doped"
                    " luminescent glass, comprising the steps of...",
                    "description": "The invention belongs to the technical"
                    " field of luminescent materials...",
                    "legal_status": "pending",
                    "country_code": "CN",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 6: EngineeringVillageAdapter — Elsevier 工程文献
# ============================================================


class EngineeringVillageAdapter(RESTAdapter):
    """Elsevier Engineering Village 工程文献适配器.

    数据源: Elsevier Engineering Village API
            (https://api.elsevier.com/content/ev)
    协议: REST / HTTPS, 认证 API Key (Elsevier API Key)
    限流: 20 req/min
    缓存: 7200s (2 小时)

    Engineering Village 是 Elsevier 旗下的工程领域综合文献平台, 整合了
    Compendex (工程综合)、Inspec (物理 / 电子 / 计算机)、GEOBASE (地球科学)、
    GeoRef (地质学) 等权威工程数据库。收录 1884 年至今的 2 亿+ 工程文献记录,
    是工程技术领域最权威的检索工具。

    支持的流 (streams):
    - "records": 期刊论文 / 技术报告记录
    - "conference_papers": 会议论文记录
    - "standards": 技术标准记录

    搜索维度:
    - 按 title (标题)
    - 按 author (作者)
    - 按 subject (主题 / 受控词)
    - 按 ISBN / ISSN

    SchemaMapper 映射 (源字段 → L3 标准字段):
    - accession_number → entity_id (主键)
    - title → entity_name
    - record_type → entity_type (默认 "engineering_literature")
    - doi → identifiers
    - abstract → properties
    - doi → source_uri
    - title → description
    - doi / title / authors / abstract / publication_year → 同名字段
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 Engineering Village 适配器.

        Args:
            config: 连接器配置
            auth_token: Elsevier API Key
            **kwargs: 传递给 RESTAdapter 的额外参数
        """
        super().__init__(
            config,
            search_endpoint=kwargs.pop(
                "search_endpoint", "/content/ev/search"
            ),
            fetch_endpoint=kwargs.pop(
                "fetch_endpoint", "/content/ev/records/{id}"
            ),
            page_size=kwargs.pop("page_size", 20),
            auth_type="api_key",
            auth_token=auth_token,
            **kwargs,
        )

        mapper = self._schema_mapper
        mapper.add_mapping(
            FieldMapping(
                source_field="accession_number",
                target_field="entity_id",
                required=True,
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="title", target_field="entity_name", required=True
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="record_type",
                target_field="entity_type",
                default_value="engineering_literature",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="identifiers")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="properties")
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="description")
        )
        # 工程文献领域专属字段
        mapper.add_mapping(
            FieldMapping(
                source_field="accession_number",
                target_field="accession_number",
            )
        )
        mapper.add_mapping(FieldMapping(source_field="title", target_field="title"))
        mapper.add_mapping(
            FieldMapping(source_field="authors", target_field="authors")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="source_title", target_field="source_title"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="publication_year",
                target_field="publication_date",
                transform="iso_datetime",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="doi")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="controlled_terms", target_field="controlled_terms"
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="classification_codes",
                target_field="classification_codes",
                transform="split_comma",
            )
        )

    def _do_spec(self) -> AdapterSpec:
        """声明 Engineering Village 适配器规范."""
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=_rest_caps(),
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "auth_token": {
                        "type": "string",
                        "description": "Elsevier API Key",
                    },
                    "page_size": {"type": "integer", "default": 20},
                },
                "required": ["base_url", "auth_token"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://dev.elsevier.com/documentation/EngineeringVillageAPI",
            changelog={
                "1.0.0": "初始版本, 支持 records / conference_papers /"
                " standards 流",
            },
        )

    def _get_schema(self) -> DataSourceSchema:
        """返回 records 流的 Schema."""
        return DataSourceSchema(
            stream_name="records",
            fields=[
                SchemaField(
                    name="accession_number",
                    data_type="string",
                    nullable=False,
                    primary_key=True,
                    description="Engineering Village 唯一检索号",
                ),
                SchemaField(
                    name="title",
                    data_type="string",
                    nullable=False,
                    description="文献标题",
                ),
                SchemaField(
                    name="authors",
                    data_type="array",
                    description="作者列表",
                ),
                SchemaField(
                    name="abstract",
                    data_type="string",
                    description="文献摘要",
                ),
                SchemaField(
                    name="source_title",
                    data_type="string",
                    description="期刊 / 来源名称",
                ),
                SchemaField(
                    name="publication_year",
                    data_type="integer",
                    description="出版年份",
                ),
                SchemaField(
                    name="doi",
                    data_type="string",
                    description="DOI 标识符",
                    format="doi",
                ),
                SchemaField(
                    name="issn",
                    data_type="string",
                    description="ISSN (期刊)",
                ),
                SchemaField(
                    name="isbn",
                    data_type="string",
                    description="ISBN (图书)",
                ),
                SchemaField(
                    name="conference_info",
                    data_type="object",
                    description="会议信息 (会议论文专用)",
                ),
                SchemaField(
                    name="language",
                    data_type="string",
                    description="文献语言",
                    max_length=3,
                ),
                SchemaField(
                    name="controlled_terms",
                    data_type="array",
                    description="受控词 (Compendex / Inspec 主题词)",
                ),
                SchemaField(
                    name="classification_codes",
                    data_type="array",
                    description="分类码列表",
                ),
                SchemaField(
                    name="cited_by_count",
                    data_type="integer",
                    description="被引次数",
                ),
            ],
            primary_keys=["accession_number"],
            cursor_field="publication_year",
            description="Engineering Village 工程文献 (Compendex / Inspec /"
            " GEOBASE, 2 亿+ 记录)",
            metadata={"source": "Elsevier-EV", "authority_tier": "T2"},
        )

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 Engineering Village 搜索 URL.

        支持按 title / author / subject / isbn-issn 搜索。
        API Key 通过 X-API-Key 头传递 (由 _build_auth_headers 处理)。
        """
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "query": query,
            "search_field": kwargs.get("search_field", "title"),
            "count": kwargs.get("limit", self._page_size),
            "start": kwargs.get("offset", 1),
            "database": kwargs.get("database", "cpx"),  # cpx=Compendex
        }
        return url, params, self._build_auth_headers()

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 EV 记录获取 URL (按 accession_number)."""
        endpoint = self._fetch_endpoint.replace("{id}", resource_id)
        url = f"{self.config.base_url}{endpoint}"
        return url, {}, self._build_auth_headers()

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 Engineering Village API 响应.

        EV 返回格式:
            {"search-results": {"entry": [...], "opensearch:totalResults": N}}
        或简化格式: {"records": [...]}
        """
        if isinstance(data, dict):
            # 标准 Elsevier 嵌套结构
            search_results = data.get("search-results", {})
            if isinstance(search_results, dict):
                entry = search_results.get("entry", [])
                if isinstance(entry, list):
                    return entry
            # 简化结构
            for key in ("records", "results", "items", "entry"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            return data
        return []

    @classmethod
    def create(cls, auth_token: str = "") -> EngineeringVillageAdapter:
        """创建预配置的 Engineering Village 适配器实例 (含模拟数据).

        Args:
            auth_token: Elsevier API Key

        Returns:
            预配置的 EngineeringVillageAdapter 实例
        """
        config = ConnectorConfig(
            id="engineering-village",
            name="Engineering Village (Elsevier)",
            tier=ConnectorTier.INDUSTRY,
            protocol=ConnectorProtocol.REST,
            base_url="https://api.elsevier.com",
            auth_config={"type": "api_key"},
            rate_limit=20,
            cache_ttl=7200,
            version="1.0.0",
            tags=["engineering", "compendex", "inspec", "paid"],
            description="Elsevier Engineering Village (Compendex / Inspec /"
            " GEOBASE, 2 亿+ 记录)",
        )
        instance = cls(config, auth_token=auth_token)
        instance.set_mock_data(
            [
                {
                    "accession_number": "EV-2023000123456",
                    "title": "Optical properties of Dy3+ doped"
                    " lithium borate glasses for white LEDs",
                    "record_type": "engineering_literature",
                    "authors": ["Prasad, R.", "Kumar, S.", "Singh, A."],
                    "abstract": "Dy3+ doped lithium borate glass samples"
                    " were prepared by melt quenching technique and their"
                    " optical properties investigated...",
                    "source_title": "Optical Materials",
                    "publication_year": 2023,
                    "doi": "10.1016/j.optmat.2023.113456",
                    "issn": "0925-3467",
                    "isbn": "",
                    "conference_info": {},
                    "language": "eng",
                    "controlled_terms": [
                        "rare earth doped materials",
                        "luminescence",
                        "optical glass",
                    ],
                    "classification_codes": ["741.1", "931.3"],
                    "cited_by_count": 8,
                },
                {
                    "accession_number": "EV-2022000987654",
                    "title": "Photoluminescence and energy transfer"
                    " in Dy3+/Eu3+ co-doped phosphors",
                    "record_type": "engineering_literature",
                    "authors": ["Li, M.", "Zhang, Y."],
                    "abstract": "A series of Dy3+/Eu3+ co-doped phosphor"
                    " samples were synthesized and their energy transfer"
                    " mechanisms studied...",
                    "source_title": "Journal of Physics D: Applied Physics",
                    "publication_year": 2022,
                    "doi": "10.1088/1361-6463/ac4567",
                    "issn": "0022-3727",
                    "isbn": "",
                    "conference_info": {},
                    "language": "eng",
                    "controlled_terms": [
                        "phosphors",
                        "energy transfer",
                        "white light emission",
                    ],
                    "classification_codes": ["741.1", "743.2"],
                    "cited_by_count": 23,
                },
            ]
        )
        return instance


# ============================================================
# 模块导出
# ============================================================


__all__ = [
    # 基类再导出 (供下游模块统一导入)
    "DataAdapterBase",
    "DatabaseAdapter",
    "GraphQLAdapter",
    "RESTAdapter",
    # Tier-2 行业适配器 (6 个)
    "CASAdapter",
    "WebOfScienceAdapter",
    "SciFinderAdapter",
    "ReaxysAdapter",
    "GooglePatentsAdapter",
    "EngineeringVillageAdapter",
]
