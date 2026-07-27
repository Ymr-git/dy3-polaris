"""L3 领域知识层 — Tier-1 公共数据源适配器集合.

本模块实现 10 个面向公开 (PUBLIC) 数据源的具体 REST 适配器, 属于连接器分层架构中的
第一档 (Tier-1) 公共数据源. 这些数据源免费开放、权威度高 (T1 可信度)、限流宽松,
是 L3 知识层"事实底座"的首要来源.

=== Tier-1 公共数据源策略 ===

1. 权威度分级: 所有适配器均标记为 ``ConnectorTier.PUBLIC``, 在跨库对齐融合
   (:mod:`dy3_polaris.l3.cross_db`) 中享有最高质量权重, 作为冲突解决
   (:mod:`dy3_polaris.l3.models.ConflictResolutionStrategy`) 的事实基准.

2. 零认证优先: 尽量使用无需认证的公开端点 (NIST/PubChem/arXiv/Wikipedia/
   OpenAlex/CrossRef/DOAJ/UniProt), 仅有 ChemSpider (API Key 必需) 与
   Semantic Scholar (API Key 可选, 提升限流) 需要凭据.

3. 礼貌爬取 (Polite Pool): 对 OpenAlex/CrossRef 等"礼貌池"服务, 在请求头注入
   ``User-Agent`` (含 mailto), 以获得更稳定的服务质量与更高限流配额.

4. 限流约束: 每个适配器按官方文档设置 ``rate_limit`` (次/分钟), 由基类
   :class:`~dy3_polaris.l3.connector.KnowledgeConnector` 的滑动窗口限流器强制执行,
   并受 :class:`~dy3_polaris.l3.connector.CircuitBreaker` 熔断保护.

5. 统一 L3 标准字段: 每个适配器通过 :class:`SchemaMapper` 将异构源字段映射到
   L3 知识实体的标准字段 (``entity_id`` / ``entity_name`` / ``entity_type`` /
   ``identifiers`` / ``properties`` / ``source_uri`` / ``description``),
   以及领域专属字段 (化学: ``cas_number`` / ``molecular_formula`` /
   ``molecular_weight``; 文献: ``doi`` / ``title`` / ``authors`` / ``abstract`` /
   ``publication_date``).

6. 真实响应解析: ``_parse_response()`` 按各 API 真实返回格式实现
   (NIST HTML 表 / PubChem ``PC_Compounds`` / arXiv Atom Feed / OpenAlex-CrossRef-
   DOAJ-SemanticScholar 嵌套 JSON / UniProt JSON / ChemSpider JSON),
   支持未来接入真实 HTTP 后端时直接复用.

7. 可测试性: 每个适配器的 ``create()`` 类方法注入真实样例 mock 数据, 使
   ``read()`` / ``discover()`` 在无网络环境下也可端到端验证.

适配器清单 (10 个):
    1. :class:`NISTWebBookAdapter`        — NIST Chemistry WebBook (化学/热物性/光谱)
    2. :class:`PubChemAdapter`            — PubChem PUG REST (化合物/物质/生物活性)
    3. :class:`ArxivAdapter`              — arXiv API (预印本文献)
    4. :class:`WikipediaAdapter`          — Wikipedia API (百科条目/分类)
    5. :class:`OpenAlexAdapter`           — OpenAlex ( scholarly works/authors/institutions )
    6. :class:`CrossRefAdapter`           — CrossRef REST (DOI/文献/基金)
    7. :class:`DOAJAdapter`               — DOAJ (开放获取期刊/文章)
    8. :class:`UniProtAdapter`           — UniProt REST (蛋白质序列/注释)
    9. :class:`ChemSpiderAdapter`         — ChemSpider API (化合物, 需 API Key)
   10. :class:`SemanticScholarAdapter`    — Semantic Scholar Graph API (文献图谱)

所有适配器均继承 :class:`~dy3_polaris.l3.adapter_bases.RESTAdapter` (经
:class:`Tier1PublicAdapterBase` 中间基类), 复用其 spec/check/discover/read 协议、
熔断、限流与缓存能力.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .adapter_bases import RESTAdapter
from .connector import (
    ConnectorConfig,
    ConnectorProtocol,
    ConnectorTier,
)
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
# 模块级辅助函数
# ============================================================


def _project_to_fields(
    records: list[dict[str, Any]],
    field_names: list[str],
) -> list[dict[str, Any]]:
    """将记录投影到指定字段集合 (按流 Schema 裁剪)."""
    if not field_names:
        return records
    return [{k: r.get(k) for k in field_names} for r in records]


def _apply_incremental_filter(
    records: list[dict[str, Any]],
    cursor_field: str,
    cursor_value: str,
) -> list[dict[str, Any]]:
    """按游标字段过滤出大于检查点游标值的记录 (增量同步)."""
    if not cursor_field or not cursor_value:
        return records
    return [r for r in records if str(r.get(cursor_field, "")) > cursor_value]


def describe_adapter(adapter: DataAdapterBase) -> dict[str, Any]:
    """返回适配器的可读摘要 (用于注册中心展示与诊断).

    Args:
        adapter: 数据适配器实例

    Returns:
        包含 id/name/tier/phase 的摘要字典
    """
    phase = (
        adapter.lifecycle_phase
        if hasattr(adapter, "lifecycle_phase")
        else LifecyclePhase.SPEC
    )
    return {
        "id": adapter.config.id,
        "name": adapter.config.name,
        "tier": adapter.config.tier.value,
        "protocol": adapter.config.protocol.value,
        "phase": phase.value,
    }


# ============================================================
# Tier-1 公共数据源 REST 适配器中间基类
# ============================================================


class Tier1PublicAdapterBase(RESTAdapter):
    """Tier-1 公共数据源 REST 适配器通用基类 (内部使用).

    在 :class:`RESTAdapter` 之上为所有 Tier-1 公共适配器提供统一行为:

    - **流感知读取**: ``_do_read()`` 根据请求的 ``stream_name`` 选择对应流的
      Schema, 据其字段列表投影记录、据其 ``cursor_field`` 执行增量过滤,
      从而正确支持多流 (multi-stream) 适配器 (如 NIST 的 compounds/thermo/spectra)。
    - **Schema 发现**: 多流子类通过设置 ``self._STREAMS`` 并重写 ``_do_discover()``
      暴露全部流; 单流适配器沿用基类 ``_do_discover()`` (返回 ``_get_schema()``)。
    - **统一游标检查点**: 读取结果附带基于流游标字段的 :class:`SyncCheckpoint`。

    子类 (具体适配器) 须实现:
        - ``__init__``: 构建 :class:`SchemaMapper` 字段映射后调用 ``super().__init__``
        - :meth:`_do_spec`: 声明适配器规范与能力
        - :meth:`_get_schema`: 返回主 (默认) 流 Schema
        - :meth:`_build_search_url` / :meth:`_build_fetch_url` / :meth:`_parse_response`
        - ``create()`` 类方法: 返回预配置实例并注入 mock 数据

    Attributes:
        _STREAMS: 多流适配器的全部流 Schema 映射 (单流适配器保持 ``None``)
        _PRIMARY_STREAM: 多流适配器的主 (默认) 流名称
    """

    _STREAMS: dict[str, DataSourceSchema] | None = None
    _PRIMARY_STREAM: str = ""

    def _resolve_schema(self, stream_name: str = "") -> DataSourceSchema:
        """根据流名称解析对应的 Schema.

        多流适配器优先从 ``_STREAMS`` 取; 单流适配器回退到 ``_get_schema()``。
        """
        streams = self._STREAMS
        if streams:
            if stream_name and stream_name in streams:
                return streams[stream_name]
            if self._PRIMARY_STREAM and self._PRIMARY_STREAM in streams:
                return streams[self._PRIMARY_STREAM]
            return next(iter(streams.values()))
        return self._get_schema()

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        """读取数据 (流感知 + 增量过滤 + 字段投影)."""
        schema = self._resolve_schema(stream_name)
        cursor_field = schema.cursor_field if schema else ""

        records = self._mock_request(limit=limit or self._page_size)

        # 增量同步: 仅保留游标值大于检查点的记录
        if (
            sync_mode == SyncMode.INCREMENTAL
            and checkpoint
            and checkpoint.cursor_value
            and cursor_field
        ):
            records = _apply_incremental_filter(
                records, cursor_field, checkpoint.cursor_value
            )

        # 按流 Schema 字段投影
        field_names = schema.field_names() if schema else []
        if field_names:
            records = _project_to_fields(records, field_names)

        cursor_value = (
            str(records[-1].get(cursor_field, ""))
            if records and cursor_field
            else ""
        )

        return ReadResult(
            records=records,
            checkpoint=self._make_checkpoint(
                stream_name=stream_name or (schema.stream_name if schema else "default"),
                records_read=len(records),
                cursor_value=cursor_value,
            ),
            has_more=False,
        )


# ============================================================
# 适配器 1: NIST Chemistry WebBook
# ============================================================


class NISTWebBookAdapter(Tier1PublicAdapterBase):
    """NIST Chemistry WebBook 适配器.

    数据源: `NIST Chemistry WebBook <https://webbook.nist.gov/chemistry/>`_
    (美国国家标准与技术研究院化学手册). 提供化合物的热力学数据、
    热物性参数 (熔点/沸点/密度) 与多种光谱 (IR/质谱/UV), 是化学领域
    事实校验与物性查询的权威基准之一.

    特性:
    - 协议: REST/HTTPS, 无认证 (公开)
    - 限流: 30 次/分钟 (NIST 建议保守爬取)
    - 流: ``compounds`` (化合物主信息), ``thermo`` (热力学), ``spectra`` (光谱)
    - 检索维度: 名称 / 分子式 / CAS 号 / InChI
    - 响应格式: HTML 表格 (本适配器简化为 dict 解析)

    L3 标准映射: ``cas_number`` → ``entity_id``, ``name`` → ``entity_name``,
    ``formula`` → ``molecular_formula``, ``molecular_weight`` → ``molecular_weight``.
    """

    _PRIMARY_STREAM = "compounds"

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/cgi/cbook",
        fetch_endpoint: str = "/cgi/cbook",
        page_size: int = 20,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化 NIST WebBook 适配器.

        Args:
            config: 连接器配置
            search_endpoint: 搜索端点 (默认 ``/cgi/cbook``)
            fetch_endpoint: 获取端点 (默认 ``/cgi/cbook``)
            page_size: 分页大小
            auth_token: 未使用 (NIST 公开, 保留以统一接口)
            **kwargs: 透传给 ``RESTAdapter``
        """
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="cas_number", target_field="entity_id")
        )
        mapper.add_mapping(
            FieldMapping(source_field="name", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type",
                target_field="entity_type",
                default_value="chemical_compound",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="cas_number", target_field="cas_number")
        )
        mapper.add_mapping(
            FieldMapping(source_field="formula", target_field="molecular_formula")
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
                source_field="melting_point", target_field="melting_point",
                transform="parse_float",
            )
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="boiling_point", target_field="boiling_point",
                transform="parse_float",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="density", target_field="density",
                         transform="parse_float")
        )
        mapper.add_mapping(
            FieldMapping(source_field="source_uri", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="name", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )
        self._STREAMS = self._build_schemas()

    def _build_schemas(self) -> dict[str, DataSourceSchema]:
        """构建 NIST 三个流的 Schema."""
        compounds = DataSourceSchema(
            stream_name="compounds",
            fields=[
                SchemaField(name="cas_number", data_type="string", nullable=False,
                            primary_key=True, description="CAS 注册号"),
                SchemaField(name="name", data_type="string", nullable=False,
                            description="化合物名称"),
                SchemaField(name="formula", data_type="string", nullable=True,
                            description="分子式"),
                SchemaField(name="molecular_weight", data_type="float", nullable=True,
                            description="分子量 (g/mol)"),
                SchemaField(name="melting_point", data_type="float", nullable=True,
                            description="熔点 (K)"),
                SchemaField(name="boiling_point", data_type="float", nullable=True,
                            description="沸点 (K)"),
                SchemaField(name="density", data_type="float", nullable=True,
                            description="密度 (g/cm^3)"),
                SchemaField(name="source_uri", data_type="string", nullable=True,
                            format="uri", description="NIST 条目 URL"),
            ],
            primary_keys=["cas_number"],
            cursor_field="cas_number",
            description="NIST 化合物主信息流",
            metadata={"source": "nist-webbook"},
        )
        thermo = DataSourceSchema(
            stream_name="thermo",
            fields=[
                SchemaField(name="cas_number", data_type="string", nullable=False,
                            primary_key=True, description="CAS 注册号"),
                SchemaField(name="name", data_type="string", nullable=True,
                            description="化合物名称"),
                SchemaField(name="thermo_data", data_type="object", nullable=True,
                            description="热力学数据 (热容/熵/焓)"),
            ],
            primary_keys=["cas_number"],
            cursor_field="cas_number",
            description="NIST 热力学数据流",
        )
        spectra = DataSourceSchema(
            stream_name="spectra",
            fields=[
                SchemaField(name="cas_number", data_type="string", nullable=False,
                            primary_key=True, description="CAS 注册号"),
                SchemaField(name="ir_spectrum", data_type="object", nullable=True,
                            description="红外光谱数据"),
                SchemaField(name="mass_spectrum", data_type="object", nullable=True,
                            description="质谱数据"),
                SchemaField(name="uv_spectrum", data_type="object", nullable=True,
                            description="紫外-可见光谱数据"),
            ],
            primary_keys=["cas_number"],
            cursor_field="cas_number",
            description="NIST 光谱数据流",
        )
        return {"compounds": compounds, "thermo": thermo, "spectra": spectra}

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://webbook.nist.gov/chemistry/",
            changelog={"1.0.0": "初始实现, 支持化合物/热力学/光谱三流"},
        )

    def _do_discover(self) -> DiscoverResult:
        return DiscoverResult(
            streams=list(self._STREAMS.values()),
            adapter_id=self.config.id,
        )

    def _get_schema(self) -> DataSourceSchema:
        return self._STREAMS[self._PRIMARY_STREAM]

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 NIST 搜索 URL (按名称/分子式/CAS/InChI)."""
        search_type = kwargs.get("search_type", "Name")
        units = kwargs.get("units", "SI")
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            search_type: query,
            "Units": units,
            "cIE": "on",
        }
        headers = {"Accept": "text/html", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 CAS 号获取化合物详情的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}"
        params: dict[str, Any] = {"CAS": resource_id, "Units": "SI"}
        headers = {"Accept": "text/html", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 NIST 响应 (HTML 表格简化为 dict).

        真实响应为 HTML 表格, 本方法处理模拟的已解析结构:
        - ``{"results": [...]}`` / ``{"compounds": [...]}``: 提取列表
        - ``list``: 直接返回
        - 单条 ``dict``: 包装为单元素列表
        """
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("results", "compounds", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []

    @classmethod
    def create(cls, auth_token: str = "") -> NISTWebBookAdapter:
        """创建预配置的 NIST WebBook 适配器实例 (含 mock 数据)."""
        config = ConnectorConfig(
            id="nist-webbook",
            name="NIST Chemistry WebBook",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTPS,
            base_url="https://webbook.nist.gov",
            rate_limit=30,
            cache_ttl=3600,
            version="1.0.0",
            tags=["chemistry", "thermodynamics", "spectra"],
            description="NIST 化学手册: 热物性与光谱权威数据",
        )
        instance = cls(
            config,
            search_endpoint="/cgi/cbook",
            fetch_endpoint="/cgi/cbook",
            page_size=20,
        )
        instance.set_mock_data(
            [
                {
                    "cas_number": "7732-18-5",
                    "name": "Water",
                    "formula": "H2O",
                    "molecular_weight": 18.015,
                    "melting_point": 273.15,
                    "boiling_point": 373.15,
                    "density": 0.997,
                    "source_uri": "https://webbook.nist.gov/cgi/cbook.cgi?CAS=7732-18-5",
                    "thermo_data": {"cp": 75.29, "s": 69.91, "dhf": -285.83},
                    "ir_spectrum": {"peaks": [{"x": 3400, "y": 0.9}]},
                    "mass_spectrum": {"peaks": [{"mz": 18, "intensity": 100.0}]},
                    "uv_spectrum": {"peaks": []},
                },
                {
                    "cas_number": "64-17-5",
                    "name": "Ethanol",
                    "formula": "C2H6O",
                    "molecular_weight": 46.069,
                    "melting_point": 159.05,
                    "boiling_point": 351.45,
                    "density": 0.789,
                    "source_uri": "https://webbook.nist.gov/cgi/cbook.cgi?CAS=64-17-5",
                    "thermo_data": {"cp": 112.3, "s": 160.7, "dhf": -277.6},
                    "ir_spectrum": {"peaks": [{"x": 1050, "y": 0.8}, {"x": 2970, "y": 0.7}]},
                    "mass_spectrum": {"peaks": [{"mz": 31, "intensity": 100.0},
                                               {"mz": 46, "intensity": 60.0}]},
                    "uv_spectrum": {"peaks": []},
                },
                {
                    "cas_number": "71-43-2",
                    "name": "Benzene",
                    "formula": "C6H6",
                    "molecular_weight": 78.114,
                    "melting_point": 278.65,
                    "boiling_point": 353.25,
                    "density": 0.879,
                    "source_uri": "https://webbook.nist.gov/cgi/cbook.cgi?CAS=71-43-2",
                    "thermo_data": {"cp": 134.8, "s": 173.3, "dhf": 49.0},
                    "ir_spectrum": {"peaks": [{"x": 670, "y": 0.9}, {"x": 3030, "y": 0.6}]},
                    "mass_spectrum": {"peaks": [{"mz": 78, "intensity": 100.0},
                                                {"mz": 77, "intensity": 75.0}]},
                    "uv_spectrum": {"peaks": [{"x": 254, "y": 0.95}]},
                },
            ]
        )
        return instance


# ============================================================
# 适配器 2: PubChem PUG REST
# ============================================================


class PubChemAdapter(Tier1PublicAdapterBase):
    """PubChem PUG REST 适配器.

    数据源: `PubChem PUG REST <https://pubchem.ncbi.nlm.nih.gov/rest/pug>`_
    (NCBI 公共化学数据库). 汇集化合物 (Compounds)、物质 (Substances) 与
    生物活性测定 (Assays) 三类数据, 通过 CID/名称/SMILES/InChIKey 检索,
    是化学实体对齐与标识解析的核心公共源.

    特性:
    - 协议: REST/HTTPS, 无认证
    - 限流: 200 次/分钟 (5 次/秒, 无密钥下PubChem 官方建议)
    - 流: ``compounds`` / ``substances`` / ``assays``
    - 响应格式: JSON, 化合物流含 ``PC_Compounds`` 结构
    """

    _PRIMARY_STREAM = "compounds"

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/compound",
        fetch_endpoint: str = "/compound/cid",
        page_size: int = 20,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="cid", target_field="entity_id",
                         transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="name", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="chemical_compound",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="cid", target_field="cid", transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="canonical_smiles", target_field="canonical_smiles")
        )
        mapper.add_mapping(
            FieldMapping(source_field="isomeric_smiles", target_field="isomeric_smiles")
        )
        mapper.add_mapping(
            FieldMapping(source_field="inchi", target_field="inchi")
        )
        mapper.add_mapping(
            FieldMapping(source_field="inchikey", target_field="inchikey")
        )
        mapper.add_mapping(
            FieldMapping(source_field="molecular_formula", target_field="molecular_formula")
        )
        mapper.add_mapping(
            FieldMapping(source_field="molecular_weight", target_field="molecular_weight",
                         transform="parse_float")
        )
        mapper.add_mapping(
            FieldMapping(source_field="iupac_name", target_field="iupac_name")
        )
        mapper.add_mapping(
            FieldMapping(source_field="cas_number", target_field="cas_number")
        )
        mapper.add_mapping(
            FieldMapping(source_field="synonyms", target_field="synonyms")
        )
        mapper.add_mapping(
            FieldMapping(source_field="source_uri", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="iupac_name", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )
        self._STREAMS = self._build_schemas()

    def _build_schemas(self) -> dict[str, DataSourceSchema]:
        compounds = DataSourceSchema(
            stream_name="compounds",
            fields=[
                SchemaField(name="cid", data_type="integer", nullable=False,
                            primary_key=True, description="PubChem CID"),
                SchemaField(name="name", data_type="string", nullable=True,
                            description="化合物名称"),
                SchemaField(name="canonical_smiles", data_type="string", nullable=True,
                            description="规范 SMILES"),
                SchemaField(name="isomeric_smiles", data_type="string", nullable=True,
                            description="异构 SMILES"),
                SchemaField(name="inchi", data_type="string", nullable=True,
                            description="InChI"),
                SchemaField(name="inchikey", data_type="string", nullable=True,
                            description="InChIKey"),
                SchemaField(name="molecular_formula", data_type="string", nullable=True,
                            description="分子式"),
                SchemaField(name="molecular_weight", data_type="float", nullable=True,
                            description="分子量"),
                SchemaField(name="iupac_name", data_type="string", nullable=True,
                            description="IUPAC 名称"),
                SchemaField(name="cas_number", data_type="string", nullable=True,
                            description="CAS 号"),
                SchemaField(name="synonyms", data_type="array", nullable=True,
                            description="同义名列表"),
                SchemaField(name="source_uri", data_type="string", nullable=True,
                            format="uri", description="PubChem 条目 URL"),
            ],
            primary_keys=["cid"],
            cursor_field="cid",
            description="PubChem 化合物流",
            metadata={"source": "pubchem"},
        )
        substances = DataSourceSchema(
            stream_name="substances",
            fields=[
                SchemaField(name="sid", data_type="integer", nullable=False,
                            primary_key=True, description="PubChem SID"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="source_name", data_type="string", nullable=True),
            ],
            primary_keys=["sid"],
            cursor_field="sid",
            description="PubChem 物质流",
        )
        assays = DataSourceSchema(
            stream_name="assays",
            fields=[
                SchemaField(name="aid", data_type="integer", nullable=False,
                            primary_key=True, description="PubChem AID"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="outcome", data_type="string", nullable=True),
            ],
            primary_keys=["aid"],
            cursor_field="aid",
            description="PubChem 生物活性测定流",
        )
        return {"compounds": compounds, "substances": substances, "assays": assays}

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            changelog={"1.0.0": "初始实现, 支持化合物/物质/测定三流"},
        )

    def _do_discover(self) -> DiscoverResult:
        return DiscoverResult(
            streams=list(self._STREAMS.values()),
            adapter_id=self.config.id,
        )

    def _get_schema(self) -> DataSourceSchema:
        return self._STREAMS[self._PRIMARY_STREAM]

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 PubChem PUG 搜索 URL (按 name/SMILES/InChIKey)."""
        namespace = kwargs.get("namespace", "name")
        url = f"{self.config.base_url}{self._search_endpoint}/{namespace}/{query}/JSON"
        params: dict[str, Any] = {
            "MaxRecords": kwargs.get("limit", self._page_size),
        }
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 CID 获取化合物属性 JSON 的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}/{resource_id}/JSON"
        params: dict[str, Any] = {
            "property": "MolecularFormula,MolecularWeight,IUPACName,InChI,InChIKey",
        }
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 PubChem PUG 响应.

        处理 ``PC_Compounds`` 结构: 将 ``{"PC_Compounds": [{"CID": ..., "props": [...]}]}``
        展平为字段字典; 同时兼容 ``PropertyTable`` 与 ``{results: [...]}`` 形式.
        """
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        # PC_Compounds 结构
        pc = data.get("PC_Compounds")
        if isinstance(pc, list):
            out: list[dict[str, Any]] = []
            for item in pc:
                record: dict[str, Any] = {"cid": item.get("CID")}
                for prop in item.get("props", []):
                    urn = prop.get("urn", {})
                    label = urn.get("label", "")
                    value = prop.get("value", {}).get("sval")
                    key_map = {
                        "Molecular Formula": "molecular_formula",
                        "Molecular Weight": "molecular_weight",
                        "IUPAC Name": "iupac_name",
                        "InChI": "inchi",
                        "InChIKey": "inchikey",
                        "Canonical SMILES": "canonical_smiles",
                        "Isomeric SMILES": "isomeric_smiles",
                    }
                    if label in key_map:
                        record[key_map[label]] = value
                out.append(record)
            return out
        # PropertyTable 结构
        pt = data.get("PropertyTable")
        if isinstance(pt, dict) and isinstance(pt.get("Properties"), list):
            return pt["Properties"]
        # 通用列表字段
        for key in ("results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    @classmethod
    def create(cls, auth_token: str = "") -> PubChemAdapter:
        """创建预配置的 PubChem 适配器实例 (含 mock 数据)."""
        config = ConnectorConfig(
            id="pubchem",
            name="PubChem PUG REST",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.REST,
            base_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            rate_limit=200,
            cache_ttl=3600,
            version="1.0.0",
            tags=["chemistry", "compounds", "bioassay"],
            description="NCBI PubChem 公共化学数据库 (PUG REST)",
        )
        instance = cls(
            config,
            search_endpoint="/compound",
            fetch_endpoint="/compound/cid",
            page_size=20,
        )
        instance.set_mock_data(
            [
                {
                    "cid": 962,
                    "name": "Water",
                    "canonical_smiles": "O",
                    "isomeric_smiles": "O",
                    "inchi": "InChI=1S/H2O/h1H2",
                    "inchikey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N",
                    "molecular_formula": "H2O",
                    "molecular_weight": 18.015,
                    "iupac_name": "oxidane",
                    "cas_number": "7732-18-5",
                    "synonyms": ["Water", "Dihydrogen oxide", "H2O"],
                    "source_uri": "https://pubchem.ncbi.nlm.nih.gov/compound/962",
                },
                {
                    "cid": 702,
                    "name": "Ethanol",
                    "canonical_smiles": "CCO",
                    "isomeric_smiles": "CCO",
                    "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                    "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                    "molecular_formula": "C2H6O",
                    "molecular_weight": 46.069,
                    "iupac_name": "ethanol",
                    "cas_number": "64-17-5",
                    "synonyms": ["Ethanol", "Ethyl alcohol", "Alcohol"],
                    "source_uri": "https://pubchem.ncbi.nlm.nih.gov/compound/702",
                },
                {
                    "cid": 2244,
                    "name": "Aspirin",
                    "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "isomeric_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
                    "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                    "molecular_formula": "C9H8O4",
                    "molecular_weight": 180.158,
                    "iupac_name": "2-acetyloxybenzoic acid",
                    "cas_number": "50-78-2",
                    "synonyms": ["Aspirin", "Acetylsalicylic acid", "2-acetoxybenzoic acid"],
                    "source_uri": "https://pubchem.ncbi.nlm.nih.gov/compound/2244",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 3: arXiv
# ============================================================


class ArxivAdapter(Tier1PublicAdapterBase):
    """arXiv API 适配器.

    数据源: `arXiv API <http://export.arxiv.org/api/query>`_ (康奈尔大学预印本仓库).
    提供物理学/数学/计算机科学/定量生物学等领域的预印本文献, 以 Atom XML Feed
    格式返回 (本适配器简化为已解析 dict). 是科研文献时效性补充与 DOI 互链的
    重要公开源.

    特性:
    - 协议: REST/HTTP (arXiv 仅 HTTP 导出端点)
    - 限流: 10 次/分钟, 建议请求间隔 3 秒
    - 流: ``papers`` (单流)
    - 检索维度: 标题 / 作者 / 分类 / 摘要
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/api/query",
        fetch_endpoint: str = "/api/query",
        page_size: int = 20,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="arxiv_id", target_field="entity_id")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="academic_paper",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="title")
        )
        mapper.add_mapping(
            FieldMapping(source_field="authors", target_field="authors")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(source_field="categories", target_field="categories")
        )
        mapper.add_mapping(
            FieldMapping(source_field="published_date", target_field="publication_date",
                         transform="iso_datetime")
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="doi")
        )
        mapper.add_mapping(
            FieldMapping(source_field="pdf_url", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://info.arxiv.org/help/api/index.html",
            changelog={"1.0.0": "初始实现, Atom Feed 简化解析"},
        )

    def _get_schema(self) -> DataSourceSchema:
        return DataSourceSchema(
            stream_name="papers",
            fields=[
                SchemaField(name="arxiv_id", data_type="string", nullable=False,
                            primary_key=True, description="arXiv 标识 (如 2301.00001)"),
                SchemaField(name="title", data_type="string", nullable=False,
                            description="论文标题"),
                SchemaField(name="authors", data_type="array", nullable=True,
                            description="作者列表"),
                SchemaField(name="abstract", data_type="string", nullable=True,
                            description="摘要"),
                SchemaField(name="categories", data_type="array", nullable=True,
                            description="arXiv 分类列表"),
                SchemaField(name="published_date", data_type="datetime", nullable=True,
                            format="date-time", description="发布日期"),
                SchemaField(name="updated_date", data_type="datetime", nullable=True,
                            format="date-time", description="更新日期"),
                SchemaField(name="doi", data_type="string", nullable=True,
                            description="关联 DOI"),
                SchemaField(name="pdf_url", data_type="string", nullable=True,
                            format="uri", description="PDF 链接"),
                SchemaField(name="comments", data_type="string", nullable=True,
                            description="作者注释"),
                SchemaField(name="journal_ref", data_type="string", nullable=True,
                            description="期刊引用"),
            ],
            primary_keys=["arxiv_id"],
            cursor_field="published_date",
            description="arXiv 预印本文献流",
            metadata={"source": "arxiv", "format": "atom"},
        )

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 arXiv 搜索 URL (search_query 表达式)."""
        field = kwargs.get("field", "all")
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "search_query": f"{field}:{query}",
            "start": kwargs.get("start", 0),
            "max_results": kwargs.get("limit", self._page_size),
            "sortBy": kwargs.get("sort_by", "submittedDate"),
            "sortOrder": "descending",
        }
        headers = {"Accept": "application/atom+xml", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 arXiv ID 获取条目的 URL (id_list)."""
        url = f"{self.config.base_url}{self._fetch_endpoint}"
        params: dict[str, Any] = {"id_list": resource_id}
        headers = {"Accept": "application/atom+xml", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 arXiv Atom XML Feed (简化为已解析 dict).

        真实响应为 Atom XML; 本方法处理模拟的已解析结构:
        ``{"feed": {"entry": [...]}}`` 或 ``{"entries": [...]}``。
        """
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        feed = data.get("feed")
        if isinstance(feed, dict):
            entries = feed.get("entry", [])
            if isinstance(entries, dict):
                return [entries]
            if isinstance(entries, list):
                return entries
        if "entries" in data and isinstance(data["entries"], list):
            return data["entries"]
        if "entry" in data:
            entry = data["entry"]
            return entry if isinstance(entry, list) else [entry]
        for key in ("results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    @classmethod
    def create(cls, auth_token: str = "") -> ArxivAdapter:
        """创建预配置的 arXiv 适配器实例 (含 mock 数据)."""
        config = ConnectorConfig(
            id="arxiv",
            name="arXiv API",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTP,
            base_url="http://export.arxiv.org",
            rate_limit=10,
            cache_ttl=7200,
            version="1.0.0",
            tags=["literature", "preprint", "physics"],
            description="arXiv 预印本文献 API (Atom Feed)",
        )
        instance = cls(
            config,
            search_endpoint="/api/query",
            fetch_endpoint="/api/query",
            page_size=20,
        )
        instance.set_mock_data(
            [
                {
                    "arxiv_id": "2301.00001",
                    "title": "Attention Is All You Need",
                    "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                    "abstract": "The dominant sequence transduction models are based on "
                                "complex recurrent or convolutional neural networks.",
                    "categories": ["cs.CL", "cs.AI"],
                    "published_date": "2017-06-12T17:59:00Z",
                    "updated_date": "2023-01-15T00:00:00Z",
                    "doi": "10.48550/arXiv.1706.03762",
                    "pdf_url": "https://arxiv.org/pdf/1706.03762",
                    "comments": "15 pages, 5 figures",
                    "journal_ref": "NeurIPS 2017",
                },
                {
                    "arxiv_id": "2303.08774",
                    "title": "GPT-4 Technical Report",
                    "authors": ["OpenAI"],
                    "abstract": "We report the development of GPT-4, a large multimodal "
                                "model accepting image and text inputs.",
                    "categories": ["cs.CL", "cs.AI", "cs.LG"],
                    "published_date": "2023-03-15T17:00:00Z",
                    "updated_date": "2023-03-15T17:00:00Z",
                    "doi": "",
                    "pdf_url": "https://arxiv.org/pdf/2303.08774",
                    "comments": "100 pages",
                    "journal_ref": "",
                },
                {
                    "arxiv_id": "2010.11929",
                    "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition",
                    "authors": ["Alexey Dosovitskiy", "Lucas Beyer"],
                    "abstract": "While the Transformer architecture has become the de-facto "
                                "standard for natural language processing tasks, its applications "
                                "to computer vision remain limited.",
                    "categories": ["cs.CV", "cs.AI"],
                    "published_date": "2020-10-22T17:00:00Z",
                    "updated_date": "2020-10-22T17:00:00Z",
                    "doi": "10.48550/arXiv.2010.11929",
                    "pdf_url": "https://arxiv.org/pdf/2010.11929",
                    "comments": "",
                    "journal_ref": "ICLR 2021",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 4: Wikipedia
# ============================================================


class WikipediaAdapter(Tier1PublicAdapterBase):
    """Wikipedia API 适配器.

    数据源: `Wikipedia API <https://en.wikipedia.org/w/api.php>`_
    (维基百科 MediaWiki Action API). 提供百科条目与分类检索, 是通用知识、
    概念释义与跨语言对齐的基础公开源.

    特性:
    - 协议: REST/HTTPS, 无认证
    - 限流: 200 次/分钟
    - 流: ``articles`` (条目), ``categories`` (分类)
    - 检索维度: 标题 / 全文搜索
    """

    _PRIMARY_STREAM = "articles"

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/w/api.php",
        fetch_endpoint: str = "/w/api.php",
        page_size: int = 20,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="page_id", target_field="entity_id",
                         transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="encyclopedia_article",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="title")
        )
        mapper.add_mapping(
            FieldMapping(source_field="extract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(source_field="extract", target_field="description")
        )
        mapper.add_mapping(
            FieldMapping(source_field="url", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="categories", target_field="categories")
        )
        mapper.add_mapping(
            FieldMapping(source_field="coordinates", target_field="coordinates")
        )
        mapper.add_mapping(
            FieldMapping(source_field="last_modified", target_field="last_modified",
                         transform="iso_datetime")
        )
        mapper.add_mapping(
            FieldMapping(source_field="language", target_field="language")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )
        self._STREAMS = self._build_schemas()

    def _build_schemas(self) -> dict[str, DataSourceSchema]:
        articles = DataSourceSchema(
            stream_name="articles",
            fields=[
                SchemaField(name="page_id", data_type="integer", nullable=False,
                            primary_key=True, description="MediaWiki 页面 ID"),
                SchemaField(name="title", data_type="string", nullable=False,
                            description="条目标题"),
                SchemaField(name="extract", data_type="string", nullable=True,
                            description="条目摘要 (纯文本)"),
                SchemaField(name="url", data_type="string", nullable=True,
                            format="uri", description="条目 URL"),
                SchemaField(name="categories", data_type="array", nullable=True,
                            description="分类列表"),
                SchemaField(name="coordinates", data_type="object", nullable=True,
                            description="地理坐标"),
                SchemaField(name="last_modified", data_type="datetime", nullable=True,
                            format="date-time", description="最后修改时间"),
                SchemaField(name="thumbnail_url", data_type="string", nullable=True,
                            format="uri", description="缩略图 URL"),
                SchemaField(name="language", data_type="string", nullable=True,
                            description="语言代码"),
            ],
            primary_keys=["page_id"],
            cursor_field="last_modified",
            description="Wikipedia 百科条目流",
            metadata={"source": "wikipedia"},
        )
        categories = DataSourceSchema(
            stream_name="categories",
            fields=[
                SchemaField(name="category_id", data_type="integer", nullable=False,
                            primary_key=True, description="分类页 ID"),
                SchemaField(name="title", data_type="string", nullable=False,
                            description="分类名"),
                SchemaField(name="pages", data_type="array", nullable=True,
                            description="成员页面 ID 列表"),
            ],
            primary_keys=["category_id"],
            cursor_field="title",
            description="Wikipedia 分类流",
        )
        return {"articles": articles, "categories": categories}

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://www.mediawiki.org/wiki/API:Main_page",
            changelog={"1.0.0": "初始实现, 条目+分类双流"},
        )

    def _do_discover(self) -> DiscoverResult:
        return DiscoverResult(
            streams=list(self._STREAMS.values()),
            adapter_id=self.config.id,
        )

    def _get_schema(self) -> DataSourceSchema:
        return self._STREAMS[self._PRIMARY_STREAM]

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 Wikipedia 全文搜索 URL (list=search)."""
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": kwargs.get("limit", self._page_size),
            "format": "json",
        }
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 pageid 获取条目 (extract+categories) 的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}"
        params: dict[str, Any] = {
            "action": "query",
            "prop": "extracts|categories|coordinates|info",
            "pageids": resource_id,
            "explaintext": 1,
            "inprop": "url|talkid|timestamp",
            "format": "json",
        }
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 Wikipedia API 响应.

        处理两种结构:
        - 搜索: ``{"query": {"search": [...]}}``
        - 获取: ``{"query": {"pages": {"<id>": {...}}}}`` (dict of pages)
        """
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        query = data.get("query")
        if not isinstance(query, dict):
            for key in ("results", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        search = query.get("search")
        if isinstance(search, list):
            return search
        pages = query.get("pages")
        if isinstance(pages, dict):
            return list(pages.values())
        return [query]

    @classmethod
    def create(cls, auth_token: str = "") -> WikipediaAdapter:
        """创建预配置的 Wikipedia 适配器实例 (含 mock 数据)."""
        config = ConnectorConfig(
            id="wikipedia",
            name="Wikipedia API",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTPS,
            base_url="https://en.wikipedia.org",
            rate_limit=200,
            cache_ttl=3600,
            version="1.0.0",
            tags=["encyclopedia", "general-knowledge"],
            description="Wikipedia MediaWiki Action API",
        )
        instance = cls(
            config,
            search_endpoint="/w/api.php",
            fetch_endpoint="/w/api.php",
            page_size=20,
        )
        instance.set_mock_data(
            [
                {
                    "page_id": 534366,
                    "title": "Dysprosium",
                    "extract": "Dysprosium is a chemical element; it has symbol Dy and "
                               "atomic number 66. It is a rare-earth element of the "
                               "lanthanide series.",
                    "url": "https://en.wikipedia.org/wiki/Dysprosium",
                    "categories": ["Chemical elements", "Lanthanides",
                                   "Rare-earth elements"],
                    "coordinates": None,
                    "last_modified": "2024-01-10T14:22:33Z",
                    "thumbnail_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Dy.jpg",
                    "language": "en",
                },
                {
                    "page_id": 45678,
                    "title": "Lanthanide",
                    "extract": "The lanthanide or lanthanoid series comprises the 15 metallic "
                               "chemical elements with atomic numbers 57-71.",
                    "url": "https://en.wikipedia.org/wiki/Lanthanide",
                    "categories": ["Periodic table", "Lanthanides"],
                    "coordinates": None,
                    "last_modified": "2024-02-05T09:11:00Z",
                    "thumbnail_url": "",
                    "language": "en",
                },
                {
                    "page_id": 33344,
                    "title": "Photon",
                    "extract": "A photon is an elementary particle that is a quantum of the "
                               "electromagnetic field, including electromagnetic radiation "
                               "such as light and radio waves.",
                    "url": "https://en.wikipedia.org/wiki/Photon",
                    "categories": ["Elementary particles", "Bosons", "Quantum mechanics"],
                    "coordinates": None,
                    "last_modified": "2024-03-12T20:45:18Z",
                    "thumbnail_url": "",
                    "language": "en",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 5: OpenAlex
# ============================================================


class OpenAlexAdapter(Tier1PublicAdapterBase):
    """OpenAlex API 适配器.

    数据源: `OpenAlex API <https://api.openalex.org>`_ (取代 Microsoft Academic Graph
    的开放学术图谱). 覆盖 works/authors/institutions/concepts/sources 五大实体,
    提供引用计数、开放获取链接与概念聚类, 是科研影响力量化与文献图谱构建的
    首选公开源.

    特性:
    - 协议: REST/HTTPS, 无认证 (礼貌池: 注入 mailto 获得稳定配额)
    - 限流: 100 次/分钟 (礼貌池), 10 次/秒
    - 流: ``works`` / ``authors`` / ``institutions`` / ``concepts`` / ``sources``
    - 游标字段: ``publication_date`` (works 流增量同步基准)
    """

    _PRIMARY_STREAM = "works"
    _POLITE_MAILTO = "dy3-polaris@example.org"

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/works",
        fetch_endpoint: str = "/works",
        page_size: int = 25,
        auth_token: str = "",
        mailto: str = _POLITE_MAILTO,
        **kwargs: Any,
    ) -> None:
        self._mailto = mailto
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="work_id", target_field="entity_id")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="academic_paper",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="title")
        )
        mapper.add_mapping(
            FieldMapping(source_field="authors", target_field="authors")
        )
        mapper.add_mapping(
            FieldMapping(source_field="publication_date", target_field="publication_date",
                         transform="iso_datetime")
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="doi")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(source_field="concepts", target_field="concepts")
        )
        mapper.add_mapping(
            FieldMapping(source_field="cited_by_count", target_field="cited_by_count",
                         transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="open_access_url", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="venue", target_field="venue")
        )
        mapper.add_mapping(
            FieldMapping(source_field="type", target_field="type")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )
        self._STREAMS = self._build_schemas()

    def _build_schemas(self) -> dict[str, DataSourceSchema]:
        works = DataSourceSchema(
            stream_name="works",
            fields=[
                SchemaField(name="work_id", data_type="string", nullable=False,
                            primary_key=True, description="OpenAlex Work ID (W...)"),
                SchemaField(name="title", data_type="string", nullable=True),
                SchemaField(name="authors", data_type="array", nullable=True),
                SchemaField(name="publication_date", data_type="datetime", nullable=True,
                            format="date-time"),
                SchemaField(name="doi", data_type="string", nullable=True),
                SchemaField(name="abstract", data_type="string", nullable=True),
                SchemaField(name="concepts", data_type="array", nullable=True),
                SchemaField(name="cited_by_count", data_type="integer", nullable=True),
                SchemaField(name="open_access_url", data_type="string", nullable=True,
                            format="uri"),
                SchemaField(name="venue", data_type="string", nullable=True),
                SchemaField(name="type", data_type="string", nullable=True),
            ],
            primary_keys=["work_id"],
            cursor_field="publication_date",
            description="OpenAlex 学术成果流",
            metadata={"source": "openalex"},
        )
        authors = DataSourceSchema(
            stream_name="authors",
            fields=[
                SchemaField(name="author_id", data_type="string", nullable=False,
                            primary_key=True, description="OpenAlex Author ID (A...)"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="works_count", data_type="integer", nullable=True),
            ],
            primary_keys=["author_id"],
            cursor_field="author_id",
            description="OpenAlex 作者流",
        )
        institutions = DataSourceSchema(
            stream_name="institutions",
            fields=[
                SchemaField(name="institution_id", data_type="string", nullable=False,
                            primary_key=True, description="OpenAlex Institution ID (I...)"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="country", data_type="string", nullable=True),
            ],
            primary_keys=["institution_id"],
            cursor_field="institution_id",
            description="OpenAlex 机构流",
        )
        concepts = DataSourceSchema(
            stream_name="concepts",
            fields=[
                SchemaField(name="concept_id", data_type="string", nullable=False,
                            primary_key=True, description="OpenAlex Concept ID (C...)"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="level", data_type="integer", nullable=True),
            ],
            primary_keys=["concept_id"],
            cursor_field="concept_id",
            description="OpenAlex 概念流",
        )
        sources = DataSourceSchema(
            stream_name="sources",
            fields=[
                SchemaField(name="source_id", data_type="string", nullable=False,
                            primary_key=True, description="OpenAlex Source ID (S...)"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="issn", data_type="string", nullable=True),
            ],
            primary_keys=["source_id"],
            cursor_field="source_id",
            description="OpenAlex 来源流",
        )
        return {
            "works": works,
            "authors": authors,
            "institutions": institutions,
            "concepts": concepts,
            "sources": sources,
        }

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
            | AdapterCapability.INCREMENTAL
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 25},
                    "mailto": {"type": "string", "format": "email"},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://docs.openalex.org/",
            changelog={"1.0.0": "初始实现, works 增量同步"},
        )

    def _do_discover(self) -> DiscoverResult:
        return DiscoverResult(
            streams=list(self._STREAMS.values()),
            adapter_id=self.config.id,
        )

    def _get_schema(self) -> DataSourceSchema:
        return self._STREAMS[self._PRIMARY_STREAM]

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 OpenAlex works 搜索 URL (礼貌池: mailto)."""
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "search": query,
            "per-page": kwargs.get("limit", self._page_size),
            "page": kwargs.get("page", 1),
            "mailto": self._mailto,
        }
        ua = f"dy3-polaris/1.0 (mailto:{self._mailto})"
        headers = {"Accept": "application/json", "User-Agent": ua}
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 OpenAlex Work ID 获取成果的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}/{resource_id}"
        params: dict[str, Any] = {"mailto": self._mailto}
        ua = f"dy3-polaris/1.0 (mailto:{self._mailto})"
        headers = {"Accept": "application/json", "User-Agent": ua}
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 OpenAlex 响应 (``{"meta": ..., "results": [...]}``)."""
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        if "meta" in data and isinstance(data.get("results"), list):
            return data["results"]
        for key in ("results", "data", "items", "works"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    @classmethod
    def create(cls, auth_token: str = "") -> OpenAlexAdapter:
        """创建预配置的 OpenAlex 适配器实例 (含 mock 数据)."""
        config = ConnectorConfig(
            id="openalex",
            name="OpenAlex API",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTPS,
            base_url="https://api.openalex.org",
            rate_limit=100,
            cache_ttl=3600,
            version="1.0.0",
            tags=["scholarly", "citation", "open-access"],
            description="OpenAlex 开放学术图谱 (礼貌池)",
        )
        instance = cls(
            config,
            search_endpoint="/works",
            fetch_endpoint="/works",
            page_size=25,
        )
        instance.set_mock_data(
            [
                {
                    "work_id": "W2741809807",
                    "title": "Attention Is All You Need",
                    "authors": ["Ashish Vaswani", "Noam Shazeer"],
                    "publication_date": "2017-06-12T00:00:00Z",
                    "doi": "10.48550/arXiv.1706.03762",
                    "abstract": "The dominant sequence transduction models are based on "
                                "complex recurrent or convolutional neural networks.",
                    "concepts": ["Natural language processing", "Transformer", "Attention"],
                    "cited_by_count": 120000,
                    "open_access_url": "https://arxiv.org/abs/1706.03762",
                    "venue": "NeurIPS",
                    "type": "conference-paper",
                },
                {
                    "work_id": "W4239281923",
                    "title": "ImageNet Classification with Deep CNNs",
                    "authors": ["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey Hinton"],
                    "publication_date": "2012-06-01T00:00:00Z",
                    "doi": "10.1145/3065386",
                    "abstract": "We trained a large deep convolutional neural network to "
                                "classify the 1.2 million images in ImageNet.",
                    "concepts": ["Computer vision", "Deep learning",
                                 "Convolutional neural network"],
                    "cited_by_count": 95000,
                    "open_access_url": "",
                    "venue": "Communications of the ACM",
                    "type": "journal-article",
                },
                {
                    "work_id": "W1993498823",
                    "title": "Dy3+ luminescence in fluoride crystals",
                    "authors": ["A. Smith", "B. Jones"],
                    "publication_date": "2020-05-15T00:00:00Z",
                    "doi": "10.1016/j.jlumin.2020.00001",
                    "abstract": "Spectroscopic properties of Dy3+ doped into fluoride host "
                                "crystals were investigated.",
                    "concepts": ["Luminescence", "Rare earths", "Spectroscopy"],
                    "cited_by_count": 42,
                    "open_access_url": "https://doi.org/10.1016/j.jlumin.2020.00001",
                    "venue": "Journal of Luminescence",
                    "type": "journal-article",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 6: CrossRef
# ============================================================


class CrossRefAdapter(Tier1PublicAdapterBase):
    """CrossRef REST API 适配器.

    数据源: `CrossRef REST API <https://api.crossref.org>`_ (DOI 注册机构官方 API).
    提供权威的 DOI 元数据、引用计数与基金关联, 是 DOI 解析、参考文献核对与
    引用网络构建的基准公开源.

    特性:
    - 协议: REST/HTTPS, 无认证 (礼貌: 注入 mailto 获得更高配额)
    - 限流: 50 次/分钟 (礼貌池可至更高)
    - 流: ``works`` / ``funders`` / ``members`` / ``types``
    - 游标字段: ``published_date`` (works 流)
    """

    _PRIMARY_STREAM = "works"
    _POLITE_MAILTO = "dy3-polaris@example.org"

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/works",
        fetch_endpoint: str = "/works",
        page_size: int = 20,
        auth_token: str = "",
        mailto: str = _POLITE_MAILTO,
        **kwargs: Any,
    ) -> None:
        self._mailto = mailto
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="entity_id")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="academic_paper",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="doi", target_field="doi")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="title")
        )
        mapper.add_mapping(
            FieldMapping(source_field="author", target_field="authors")
        )
        mapper.add_mapping(
            FieldMapping(source_field="container_title", target_field="venue")
        )
        mapper.add_mapping(
            FieldMapping(source_field="publisher", target_field="publisher")
        )
        mapper.add_mapping(
            FieldMapping(source_field="published_date", target_field="publication_date",
                         transform="iso_datetime")
        )
        mapper.add_mapping(
            FieldMapping(source_field="type", target_field="type")
        )
        mapper.add_mapping(
            FieldMapping(source_field="issn", target_field="issn")
        )
        mapper.add_mapping(
            FieldMapping(source_field="isbn", target_field="isbn")
        )
        mapper.add_mapping(
            FieldMapping(source_field="url", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(source_field="references_count", target_field="references_count",
                         transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="is_referenced_by_count",
                         target_field="is_referenced_by_count", transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )
        self._STREAMS = self._build_schemas()

    def _build_schemas(self) -> dict[str, DataSourceSchema]:
        works = DataSourceSchema(
            stream_name="works",
            fields=[
                SchemaField(name="doi", data_type="string", nullable=False,
                            primary_key=True, description="DOI"),
                SchemaField(name="title", data_type="string", nullable=True),
                SchemaField(name="author", data_type="array", nullable=True),
                SchemaField(name="container_title", data_type="string", nullable=True),
                SchemaField(name="publisher", data_type="string", nullable=True),
                SchemaField(name="published_date", data_type="datetime", nullable=True,
                            format="date-time"),
                SchemaField(name="type", data_type="string", nullable=True),
                SchemaField(name="issn", data_type="string", nullable=True),
                SchemaField(name="isbn", data_type="string", nullable=True),
                SchemaField(name="url", data_type="string", nullable=True, format="uri"),
                SchemaField(name="abstract", data_type="string", nullable=True),
                SchemaField(name="references_count", data_type="integer", nullable=True),
                SchemaField(name="is_referenced_by_count", data_type="integer", nullable=True),
            ],
            primary_keys=["doi"],
            cursor_field="published_date",
            description="CrossRef 作品流",
            metadata={"source": "crossref"},
        )
        funders = DataSourceSchema(
            stream_name="funders",
            fields=[
                SchemaField(name="funder_id", data_type="string", nullable=False,
                            primary_key=True, description="CrossRef Funder ID"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="doi_prefix", data_type="string", nullable=True),
            ],
            primary_keys=["funder_id"],
            cursor_field="funder_id",
            description="CrossRef 基金机构流",
        )
        members = DataSourceSchema(
            stream_name="members",
            fields=[
                SchemaField(name="member_id", data_type="string", nullable=False,
                            primary_key=True),
                SchemaField(name="name", data_type="string", nullable=True),
            ],
            primary_keys=["member_id"],
            cursor_field="member_id",
            description="CrossRef 成员流",
        )
        types = DataSourceSchema(
            stream_name="types",
            fields=[
                SchemaField(name="type_id", data_type="string", nullable=False,
                            primary_key=True),
                SchemaField(name="label", data_type="string", nullable=True),
            ],
            primary_keys=["type_id"],
            cursor_field="type_id",
            description="CrossRef 类型流",
        )
        return {
            "works": works, "funders": funders,
            "members": members, "types": types,
        }

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
            | AdapterCapability.INCREMENTAL
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                    "mailto": {"type": "string", "format": "email"},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://api.crossref.org/swagger-ui/",
            changelog={"1.0.0": "初始实现, works 增量同步"},
        )

    def _do_discover(self) -> DiscoverResult:
        return DiscoverResult(
            streams=list(self._STREAMS.values()),
            adapter_id=self.config.id,
        )

    def _get_schema(self) -> DataSourceSchema:
        return self._STREAMS[self._PRIMARY_STREAM]

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 CrossRef works 搜索 URL (礼貌: mailto)."""
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "query": query,
            "rows": kwargs.get("limit", self._page_size),
            "offset": kwargs.get("offset", 0),
            "mailto": self._mailto,
        }
        headers = {"Accept": "application/json",
                   "User-Agent": f"dy3-polaris/1.0 (mailto:{self._mailto})"}
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 DOI 获取作品的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}/{resource_id}"
        params: dict[str, Any] = {"mailto": self._mailto}
        headers = {"Accept": "application/json",
                   "User-Agent": f"dy3-polaris/1.0 (mailto:{self._mailto})"}
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 CrossRef 响应 (``{"message": {"items": [...]}}`` 或单条)."""
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        message = data.get("message")
        if isinstance(message, dict):
            if isinstance(message.get("items"), list):
                return message["items"]
            return [message]
        if isinstance(message, list):
            return message
        for key in ("results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    @classmethod
    def create(cls, auth_token: str = "") -> CrossRefAdapter:
        """创建预配置的 CrossRef 适配器实例 (含 mock 数据)."""
        config = ConnectorConfig(
            id="crossref",
            name="CrossRef REST API",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTPS,
            base_url="https://api.crossref.org",
            rate_limit=50,
            cache_ttl=3600,
            version="1.0.0",
            tags=["scholarly", "doi", "citations"],
            description="CrossRef DOI 元数据与引用统计 (礼貌池)",
        )
        instance = cls(
            config,
            search_endpoint="/works",
            fetch_endpoint="/works",
            page_size=20,
        )
        instance.set_mock_data(
            [
                {
                    "doi": "10.1038/nature12373",
                    "title": "DeepMind's Go program AlphaGo",
                    "author": ["Silver, David", "Huang, Aja"],
                    "container_title": "Nature",
                    "publisher": "Springer Nature",
                    "published_date": "2013-08-22T00:00:00Z",
                    "type": "journal-article",
                    "issn": "0028-0836",
                    "isbn": "",
                    "url": "http://dx.doi.org/10.1038/nature12373",
                    "abstract": "An artificial intelligence program trained to play Go...",
                    "references_count": 35,
                    "is_referenced_by_count": 1500,
                },
                {
                    "doi": "10.1126/science.1151804",
                    "title": "The CRISPR revolution",
                    "author": ["Doudna, Jennifer", "Charpentier, Emmanuelle"],
                    "container_title": "Science",
                    "publisher": "AAAS",
                    "published_date": "2012-09-28T00:00:00Z",
                    "type": "journal-article",
                    "issn": "0036-8075",
                    "isbn": "",
                    "url": "http://dx.doi.org/10.1126/science.1151804",
                    "abstract": "A programmable dual-RNA-guided DNA endonuclease...",
                    "references_count": 28,
                    "is_referenced_by_count": 8000,
                },
                {
                    "doi": "10.48550/arXiv.2010.11929",
                    "title": "An Image is Worth 16x16 Words",
                    "author": ["Dosovitskiy, Alexey"],
                    "container_title": "ICLR",
                    "publisher": "OpenReview",
                    "published_date": "2020-10-22T00:00:00Z",
                    "type": "proceedings-article",
                    "issn": "",
                    "isbn": "",
                    "url": "http://dx.doi.org/10.48550/arXiv.2010.11929",
                    "abstract": "Vision Transformers for image recognition...",
                    "references_count": 12,
                    "is_referenced_by_count": 32000,
                },
            ]
        )
        return instance


# ============================================================
# 适配器 7: DOAJ
# ============================================================


class DOAJAdapter(Tier1PublicAdapterBase):
    """DOAJ (Directory of Open Access Journals) API 适配器.

    数据源: `DOAJ API <https://doaj.org/api>`_ (开放获取期刊与文章目录).
    聚焦开放获取 (OA) 期刊及其全文文章, 提供许可证、ISSN、关键词等元数据,
    是 OA 文献发现与全文链接获取的公共源.

    特性:
    - 协议: REST/HTTPS, 无认证
    - 限流: 100 次/分钟
    - 流: ``articles`` (文章), ``journals`` (期刊)
    - 检索维度: 标题 / 作者 / 主题 / 期刊
    """

    _PRIMARY_STREAM = "articles"

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/search/articles",
        fetch_endpoint: str = "/articles",
        page_size: int = 20,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="article_id", target_field="entity_id")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="academic_paper",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="title")
        )
        mapper.add_mapping(
            FieldMapping(source_field="authors", target_field="authors")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(source_field="journal_title", target_field="venue")
        )
        mapper.add_mapping(
            FieldMapping(source_field="journal_issn", target_field="issn")
        )
        mapper.add_mapping(
            FieldMapping(source_field="publication_date", target_field="publication_date",
                         transform="iso_datetime")
        )
        mapper.add_mapping(
            FieldMapping(source_field="keywords", target_field="keywords")
        )
        mapper.add_mapping(
            FieldMapping(source_field="fulltext_url", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="license", target_field="license")
        )
        mapper.add_mapping(
            FieldMapping(source_field="language", target_field="language")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )
        self._STREAMS = self._build_schemas()

    def _build_schemas(self) -> dict[str, DataSourceSchema]:
        articles = DataSourceSchema(
            stream_name="articles",
            fields=[
                SchemaField(name="article_id", data_type="string", nullable=False,
                            primary_key=True, description="DOAJ 文章 ID"),
                SchemaField(name="title", data_type="string", nullable=True),
                SchemaField(name="authors", data_type="array", nullable=True),
                SchemaField(name="abstract", data_type="string", nullable=True),
                SchemaField(name="journal_title", data_type="string", nullable=True),
                SchemaField(name="journal_issn", data_type="string", nullable=True),
                SchemaField(name="publication_date", data_type="datetime", nullable=True,
                            format="date-time"),
                SchemaField(name="keywords", data_type="array", nullable=True),
                SchemaField(name="fulltext_url", data_type="string", nullable=True,
                            format="uri"),
                SchemaField(name="license", data_type="string", nullable=True),
                SchemaField(name="language", data_type="string", nullable=True),
            ],
            primary_keys=["article_id"],
            cursor_field="publication_date",
            description="DOAJ 文章流",
            metadata={"source": "doaj"},
        )
        journals = DataSourceSchema(
            stream_name="journals",
            fields=[
                SchemaField(name="journal_id", data_type="string", nullable=False,
                            primary_key=True, description="DOAJ 期刊 ID"),
                SchemaField(name="title", data_type="string", nullable=True),
                SchemaField(name="issn", data_type="string", nullable=True),
                SchemaField(name="publisher", data_type="string", nullable=True),
                SchemaField(name="country", data_type="string", nullable=True),
            ],
            primary_keys=["journal_id"],
            cursor_field="journal_id",
            description="DOAJ 期刊流",
        )
        return {"articles": articles, "journals": journals}

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
            | AdapterCapability.INCREMENTAL
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://doaj.org/api/v1/docs",
            changelog={"1.0.0": "初始实现, 文章+期刊双流"},
        )

    def _do_discover(self) -> DiscoverResult:
        return DiscoverResult(
            streams=list(self._STREAMS.values()),
            adapter_id=self.config.id,
        )

    def _get_schema(self) -> DataSourceSchema:
        return self._STREAMS[self._PRIMARY_STREAM]

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 DOAJ 文章搜索 URL (路径式 query)."""
        url = f"{self.config.base_url}{self._search_endpoint}/{query}"
        params: dict[str, Any] = {
            "page": kwargs.get("page", 1),
            "pageSize": kwargs.get("limit", self._page_size),
        }
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按文章 ID 获取的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}/{resource_id}"
        params: dict[str, Any] = {}
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 DOAJ 响应 (``{"results": [...]}``)."""
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        for key in ("results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    @classmethod
    def create(cls, auth_token: str = "") -> DOAJAdapter:
        """创建预配置的 DOAJ 适配器实例 (含 mock 数据)."""
        config = ConnectorConfig(
            id="doaj",
            name="DOAJ API",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTPS,
            base_url="https://doaj.org/api",
            rate_limit=100,
            cache_ttl=3600,
            version="1.0.0",
            tags=["open-access", "journals", "scholarly"],
            description="DOAJ 开放获取期刊与文章目录",
        )
        instance = cls(
            config,
            search_endpoint="/search/articles",
            fetch_endpoint="/articles",
            page_size=20,
        )
        instance.set_mock_data(
            [
                {
                    "article_id": "0001a3c8c1f5456789abcdef0",
                    "title": "Rare-earth luminescence in advanced phosphors",
                    "authors": ["Chen, L.", "Wang, Y."],
                    "abstract": "Recent advances in rare-earth activated phosphors for "
                                "solid-state lighting are reviewed.",
                    "journal_title": "Open Materials Journal",
                    "journal_issn": "1874-0886",
                    "publication_date": "2022-03-10T00:00:00Z",
                    "keywords": ["rare earths", "luminescence", "phosphors"],
                    "fulltext_url": "https://doaj.org/article/0001a3c8c1f5",
                    "license": "CC-BY",
                    "language": "en",
                },
                {
                    "article_id": "0002b4d9d2e6567890abcdef1",
                    "title": "Open science practices in materials research",
                    "authors": ["Müller, K.", "Rossi, P."],
                    "abstract": "We discuss how open data and FAIR principles reshape "
                                "materials science reproducibility.",
                    "journal_title": "Journal of Open Research Software",
                    "journal_issn": "2049-9647",
                    "publication_date": "2023-01-05T00:00:00Z",
                    "keywords": ["open science", "FAIR data", "reproducibility"],
                    "fulltext_url": "https://doaj.org/article/0002b4d9d2e6",
                    "license": "CC-BY-NC",
                    "language": "en",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 8: UniProt
# ============================================================


class UniProtAdapter(Tier1PublicAdapterBase):
    """UniProt REST API 适配器.

    数据源: `UniProt REST API <https://rest.uniprot.org>`_
    (通用蛋白质资源知识库). 提供蛋白质序列、功能注释、亚细胞定位、GO 术语、
    PDB 交叉引用与 EC 编号, 是蛋白质序列与功能注释的事实基准.

    特性:
    - 协议: REST/HTTPS, 无认证
    - 限流: 30 次/分钟 (UniProt 建议保守请求)
    - 流: ``proteins`` (单流)
    - 检索维度: accession / gene_name / organism / keyword
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/uniprotkb/search",
        fetch_endpoint: str = "/uniprotkb",
        page_size: int = 25,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="accession", target_field="entity_id")
        )
        mapper.add_mapping(
            FieldMapping(source_field="protein_name", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="protein",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="accession", target_field="accession")
        )
        mapper.add_mapping(
            FieldMapping(source_field="protein_name", target_field="protein_name")
        )
        mapper.add_mapping(
            FieldMapping(source_field="gene_names", target_field="gene_names")
        )
        mapper.add_mapping(
            FieldMapping(source_field="organism", target_field="organism")
        )
        mapper.add_mapping(
            FieldMapping(source_field="sequence", target_field="sequence")
        )
        mapper.add_mapping(
            FieldMapping(source_field="length", target_field="length",
                         transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="molecular_weight", target_field="molecular_weight",
                         transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="function", target_field="function")
        )
        mapper.add_mapping(
            FieldMapping(source_field="subcellular_location",
                         target_field="subcellular_location")
        )
        mapper.add_mapping(
            FieldMapping(source_field="go_terms", target_field="go_terms")
        )
        mapper.add_mapping(
            FieldMapping(source_field="pdb_references", target_field="pdb_references")
        )
        mapper.add_mapping(
            FieldMapping(source_field="ec_number", target_field="ec_number")
        )
        mapper.add_mapping(
            FieldMapping(source_field="protein_name", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 25},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://www.uniprot.org/help/programmatic_access",
            changelog={"1.0.0": "初始实现, JSON 格式解析"},
        )

    def _get_schema(self) -> DataSourceSchema:
        return DataSourceSchema(
            stream_name="proteins",
            fields=[
                SchemaField(name="accession", data_type="string", nullable=False,
                            primary_key=True, description="UniProt accession"),
                SchemaField(name="protein_name", data_type="string", nullable=True),
                SchemaField(name="gene_names", data_type="array", nullable=True),
                SchemaField(name="organism", data_type="string", nullable=True),
                SchemaField(name="sequence", data_type="string", nullable=True,
                            description="氨基酸序列"),
                SchemaField(name="length", data_type="integer", nullable=True),
                SchemaField(name="molecular_weight", data_type="integer", nullable=True,
                            description="分子量 (Da)"),
                SchemaField(name="function", data_type="string", nullable=True),
                SchemaField(name="subcellular_location", data_type="string", nullable=True),
                SchemaField(name="go_terms", data_type="array", nullable=True),
                SchemaField(name="pdb_references", data_type="array", nullable=True),
                SchemaField(name="ec_number", data_type="string", nullable=True),
            ],
            primary_keys=["accession"],
            cursor_field="accession",
            description="UniProt 蛋白质流",
            metadata={"source": "uniprot"},
        )

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 UniProt 搜索 URL (query 表达式 + facets)."""
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "query": query,
            "size": kwargs.get("limit", self._page_size),
            "format": "json",
        }
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 accession 获取蛋白质条目的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}/{resource_id}"
        params: dict[str, Any] = {"format": "json"}
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 UniProt 响应.

        UniProt 支持 JSON (``{"results": [...]}``) 与 TSV; 本方法处理 JSON 形式,
        并对 TSV 字符串做兼容 (按行解析为 dict)。
        """
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                return data["results"]
            for key in ("data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        if isinstance(data, str):
            # TSV 兼容解析
            lines = [ln for ln in data.strip().split("\n") if ln]
            if not lines:
                return []
            header = lines[0].split("\t")
            records: list[dict[str, Any]] = []
            for line in lines[1:]:
                cells = line.split("\t")
                records.append(dict(zip(header, cells)))
            return records
        return []

    @classmethod
    def create(cls, auth_token: str = "") -> UniProtAdapter:
        """创建预配置的 UniProt 适配器实例 (含 mock 数据)."""
        config = ConnectorConfig(
            id="uniprot",
            name="UniProt REST API",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTPS,
            base_url="https://rest.uniprot.org",
            rate_limit=30,
            cache_ttl=7200,
            version="1.0.0",
            tags=["biology", "protein", "sequence"],
            description="UniProt 蛋白质序列与注释资源",
        )
        instance = cls(
            config,
            search_endpoint="/uniprotkb/search",
            fetch_endpoint="/uniprotkb",
            page_size=25,
        )
        instance.set_mock_data(
            [
                {
                    "accession": "P69905",
                    "protein_name": "Hemoglobin subunit alpha",
                    "gene_names": ["HBA1", "HBA2"],
                    "organism": "Homo sapiens",
                    "sequence": (
                        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQV"
                        "KGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAA"
                        "HLPAEFTPAVHASLDKFLASVSTVLTSKYR"
                    ),
                    "length": 142,
                    "molecular_weight": 15257,
                    "function": (
                        "Involved in oxygen transport from the lung to the "
                        "peripheral tissues."
                    ),
                    "subcellular_location": "Cytoplasm",
                    "go_terms": ["GO:0005833", "GO:0015671"],
                    "pdb_references": ["1A3N", "2DN1"],
                    "ec_number": "",
                },
                {
                    "accession": "P01308",
                    "protein_name": "Insulin",
                    "gene_names": ["INS"],
                    "organism": "Homo sapiens",
                    "sequence": (
                        "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTR"
                        "REAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
                    ),
                    "length": 110,
                    "molecular_weight": 11981,
                    "function": "Insulin decreases blood glucose concentration.",
                    "subcellular_location": "Secreted",
                    "go_terms": ["GO:0030073", "GO:0008286"],
                    "pdb_references": ["1A7F", "3I3Z"],
                    "ec_number": "",
                },
                {
                    "accession": "P00390",
                    "protein_name": "Glutathione reductase",
                    "gene_names": ["GR", "GSR"],
                    "organism": "Homo sapiens",
                    "sequence": (
                        "MVQGQNDNQEAIAEIQKANPELPLLQITELIDVYGVPKAINTQKDFLNSKSNKTVY"
                        "FPTVLLTAAIYDYKIKVVEDDKTYQSLVPQRPDIDRVVDGTTWEDDYPVKVRKAL"
                        "GEVPTYNVNIFPVPQKEPIPVRGATVMPYNYKDAVINVGFTVNAKPYVEAVLKD"
                        "DKWGKPVVLPTAATGEVCDDLRALLRAKASPYGVIFGLSALGNLIAEYVRFGYD"
                        "PKEKLIEAGVSNAAPAVQYVDQDYDTPVPLHADIVRVNDFELAKKGLGAQAPEA"
                        "PDVLETFEDVIKDAIVAKPKDGFIVGSSDDVIKKSVASDFNTQIYVDKVEQADI"
                        "PVLVLPDVEKEGISQEYGVQEGVQHEWYTFDQELVRTVLSDRKKVEQLTEEFTH"
                        "RQFGGVRNWSEARSLYDLQGYLPEGSTVFADLVNNRPGTFVGVEPVTNADGQL"
                        "KVDLYKNGDKVYTLPFKMNGQIAVDVGWNLPGTLPPNLRAQFEAVQDAGFKELK"
                        "AAGAKPFVVESVPYMVPPQMVTGMDNAVMAPTVCNLAVEAVGFRPTFTSEKDIQ"
                        "QEFLSKVAEGTKNLPGAIVPNDVFTTLQKQ"
                    ),
                    "length": 522,
                    "molecular_weight": 56203,
                    "function": "Maintains high levels of reduced glutathione in the cytosol.",
                    "subcellular_location": "Cytoplasm, Mitochondrion",
                    "go_terms": ["GO:0004362", "GO:0050661"],
                    "pdb_references": ["3DJG", "1GRB"],
                    "ec_number": "1.8.1.7",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 9: ChemSpider
# ============================================================


class ChemSpiderAdapter(Tier1PublicAdapterBase):
    """ChemSpider API 适配器.

    数据源: `ChemSpider API <https://www.chemspider.com/api>`_
    (英国皇家化学会化合物数据库). 提供化合物结构、质量与 CAS 交叉引用,
    需 API Key 认证 (免费注册). 是化合物结构与标识补全的辅助公共源.

    特性:
    - 协议: REST/HTTPS, 认证: API Key (必需)
    - 限流: 20 次/分钟
    - 流: ``compounds`` (单流)
    - 检索维度: CSID / 名称 / SMILES / InChI
    """

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/Search",
        fetch_endpoint: str = "/records",
        page_size: int = 20,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="csid", target_field="entity_id", transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="name", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="chemical_compound",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="csid", target_field="csid", transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="name", target_field="name")
        )
        mapper.add_mapping(
            FieldMapping(source_field="smiles", target_field="smiles")
        )
        mapper.add_mapping(
            FieldMapping(source_field="inchi", target_field="inchi")
        )
        mapper.add_mapping(
            FieldMapping(source_field="inchikey", target_field="inchikey")
        )
        mapper.add_mapping(
            FieldMapping(source_field="molecular_formula", target_field="molecular_formula")
        )
        mapper.add_mapping(
            FieldMapping(source_field="molecular_weight", target_field="molecular_weight",
                         transform="parse_float")
        )
        mapper.add_mapping(
            FieldMapping(source_field="average_mass", target_field="average_mass",
                         transform="parse_float")
        )
        mapper.add_mapping(
            FieldMapping(source_field="monoisotopic_mass",
                         target_field="monoisotopic_mass", transform="parse_float")
        )
        mapper.add_mapping(
            FieldMapping(source_field="cas_number", target_field="cas_number")
        )
        mapper.add_mapping(
            FieldMapping(source_field="common_name", target_field="common_name")
        )
        mapper.add_mapping(
            FieldMapping(source_field="source_uri", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="name", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="api_key",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )

    def _do_spec(self) -> AdapterSpec:
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
        )
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                    "auth_token": {"type": "string", "description": "ChemSpider API Key"},
                },
                "required": ["base_url", "auth_token"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://www.chemspider.com/AboutServices.aspx",
            changelog={"1.0.0": "初始实现, API Key 认证"},
        )

    def _get_schema(self) -> DataSourceSchema:
        return DataSourceSchema(
            stream_name="compounds",
            fields=[
                SchemaField(name="csid", data_type="integer", nullable=False,
                            primary_key=True, description="ChemSpider ID"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="smiles", data_type="string", nullable=True),
                SchemaField(name="inchi", data_type="string", nullable=True),
                SchemaField(name="inchikey", data_type="string", nullable=True),
                SchemaField(name="molecular_formula", data_type="string", nullable=True),
                SchemaField(name="molecular_weight", data_type="float", nullable=True),
                SchemaField(name="average_mass", data_type="float", nullable=True),
                SchemaField(name="monoisotopic_mass", data_type="float", nullable=True),
                SchemaField(name="cas_number", data_type="string", nullable=True),
                SchemaField(name="common_name", data_type="string", nullable=True),
                SchemaField(name="source_uri", data_type="string", nullable=True,
                            format="uri"),
            ],
            primary_keys=["csid"],
            cursor_field="csid",
            description="ChemSpider 化合物流",
            metadata={"source": "chemspider", "auth": "api_key"},
        )

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 ChemSpider 搜索 URL (token 查询参数)."""
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "token": self._auth_token,
            "q": query,
            "count": kwargs.get("limit", self._page_size),
        }
        headers = self._build_auth_headers()
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 CSID 获取记录详情的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}/{resource_id}"
        params: dict[str, Any] = {"token": self._auth_token}
        headers = self._build_auth_headers()
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 ChemSpider JSON 响应 (列表或 ``{results: [...]}``)."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("results", "data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []

    @classmethod
    def create(cls, auth_token: str = "") -> ChemSpiderAdapter:
        """创建预配置的 ChemSpider 适配器实例 (含 mock 数据).

        Args:
            auth_token: ChemSpider API Key (生产环境必需; 测试可留空)
        """
        config = ConnectorConfig(
            id="chemspider",
            name="ChemSpider API",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTPS,
            base_url="https://www.chemspider.com/api",
            rate_limit=20,
            cache_ttl=3600,
            version="1.0.0",
            tags=["chemistry", "compounds", "api-key"],
            description="RSC ChemSpider 化合物数据库 (API Key 认证)",
        )
        instance = cls(
            config,
            search_endpoint="/Search",
            fetch_endpoint="/records",
            page_size=20,
            auth_token=auth_token,
        )
        instance.set_mock_data(
            [
                {
                    "csid": 917,
                    "name": "Caffeine",
                    "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                    "inchi": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
                    "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                    "molecular_formula": "C8H10N4O2",
                    "molecular_weight": 194.19,
                    "average_mass": 194.19,
                    "monoisotopic_mass": 194.0804,
                    "cas_number": "58-08-2",
                    "common_name": "Caffeine",
                    "source_uri": "https://www.chemspider.com/Chemical-Structure.917.html",
                },
                {
                    "csid": 13849980,
                    "name": "Aspirin",
                    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
                    "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                    "molecular_formula": "C9H8O4",
                    "molecular_weight": 180.16,
                    "average_mass": 180.16,
                    "monoisotopic_mass": 180.0423,
                    "cas_number": "50-78-2",
                    "common_name": "Acetylsalicylic acid",
                    "source_uri": "https://www.chemspider.com/Chemical-Structure.13849980.html",
                },
                {
                    "csid": 4574,
                    "name": "Glucose",
                    "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                    "inchi": (
                        "InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2"
                        "/h2-11H,1H2/t2-,3-,4+,5-,6?/m1/s1"
                    ),
                    "inchikey": "WQZGKKKJIJFFOK-GASJEMNASA-N",
                    "molecular_formula": "C6H12O6",
                    "molecular_weight": 180.16,
                    "average_mass": 180.16,
                    "monoisotopic_mass": 180.0634,
                    "cas_number": "50-99-7",
                    "common_name": "D-Glucose",
                    "source_uri": "https://www.chemspider.com/Chemical-Structure.4574.html",
                },
            ]
        )
        return instance


# ============================================================
# 适配器 10: Semantic Scholar
# ============================================================


class SemanticScholarAdapter(Tier1PublicAdapterBase):
    """Semantic Scholar Graph API 适配器.

    数据源: `Semantic Scholar Graph API
    <https://api.semanticscholar.org/graph/v1>`_ (AI2 学术搜索引擎).
    提供论文图谱、影响力指标 (citation_count / influential_citation_count)、
    研究领域 (fields_of_study) 与 TLDR 摘要, 是文献影响力分析与
    智能检索的关键公开源.

    特性:
    - 协议: REST/HTTPS, 认证: API Key (可选, 提升 限流至 100/min)
    - 限流: 100 次/分钟 (含 Key) / 10 次/分钟 (无 Key)
    - 流: ``papers`` (论文), ``authors`` (作者)
    - 检索维度: paper_id / 标题 / 作者 / DOI
    """

    _PRIMARY_STREAM = "papers"

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/paper/search",
        fetch_endpoint: str = "/paper",
        page_size: int = 20,
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        mapper = SchemaMapper()
        mapper.add_mapping(
            FieldMapping(source_field="paper_id", target_field="entity_id")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="entity_name")
        )
        mapper.add_mapping(
            FieldMapping(
                source_field="entity_type", target_field="entity_type",
                default_value="academic_paper",
            )
        )
        mapper.add_mapping(
            FieldMapping(source_field="paper_id", target_field="paper_id")
        )
        mapper.add_mapping(
            FieldMapping(source_field="title", target_field="title")
        )
        mapper.add_mapping(
            FieldMapping(source_field="abstract", target_field="abstract")
        )
        mapper.add_mapping(
            FieldMapping(source_field="authors", target_field="authors")
        )
        mapper.add_mapping(
            FieldMapping(source_field="year", target_field="year", transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="venue", target_field="venue")
        )
        mapper.add_mapping(
            FieldMapping(source_field="external_ids", target_field="external_ids")
        )
        mapper.add_mapping(
            FieldMapping(source_field="fields_of_study", target_field="fields_of_study")
        )
        mapper.add_mapping(
            FieldMapping(source_field="citation_count", target_field="citation_count",
                         transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="reference_count", target_field="reference_count",
                         transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="influential_citation_count",
                         target_field="influential_citation_count", transform="parse_int")
        )
        mapper.add_mapping(
            FieldMapping(source_field="open_access_pdf_url", target_field="source_uri")
        )
        mapper.add_mapping(
            FieldMapping(source_field="tldr", target_field="description")
        )
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type="api_key" if auth_token else "none",
            auth_token=auth_token,
            schema_mapper=mapper,
            **kwargs,
        )
        self._STREAMS = self._build_schemas()

    def _build_schemas(self) -> dict[str, DataSourceSchema]:
        papers = DataSourceSchema(
            stream_name="papers",
            fields=[
                SchemaField(name="paper_id", data_type="string", nullable=False,
                            primary_key=True, description="Semantic Scholar Paper ID"),
                SchemaField(name="title", data_type="string", nullable=True),
                SchemaField(name="abstract", data_type="string", nullable=True),
                SchemaField(name="authors", data_type="array", nullable=True),
                SchemaField(name="year", data_type="integer", nullable=True),
                SchemaField(name="venue", data_type="string", nullable=True),
                SchemaField(name="external_ids", data_type="object", nullable=True,
                            description="外部 ID (DOI/arXiv/...)"),
                SchemaField(name="fields_of_study", data_type="array", nullable=True),
                SchemaField(name="citation_count", data_type="integer", nullable=True),
                SchemaField(name="reference_count", data_type="integer", nullable=True),
                SchemaField(name="influential_citation_count",
                            data_type="integer", nullable=True),
                SchemaField(name="open_access_pdf_url", data_type="string", nullable=True,
                            format="uri"),
                SchemaField(name="tldr", data_type="string", nullable=True,
                            description="模型生成的 TLDR"),
            ],
            primary_keys=["paper_id"],
            cursor_field="year",
            description="Semantic Scholar 论文流",
            metadata={"source": "semantic-scholar"},
        )
        authors = DataSourceSchema(
            stream_name="authors",
            fields=[
                SchemaField(name="author_id", data_type="string", nullable=False,
                            primary_key=True, description="Semantic Scholar Author ID"),
                SchemaField(name="name", data_type="string", nullable=True),
                SchemaField(name="paper_count", data_type="integer", nullable=True),
                SchemaField(name="citation_count", data_type="integer", nullable=True),
            ],
            primary_keys=["author_id"],
            cursor_field="author_id",
            description="Semantic Scholar 作者流",
        )
        return {"papers": papers, "authors": authors}

    def _do_spec(self) -> AdapterSpec:
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.RATE_LIMITED
            | AdapterCapability.CACHEABLE
            | AdapterCapability.INCREMENTAL
        )
        if self._auth_token:
            caps |= AdapterCapability.AUTHENTICATE
        return AdapterSpec(
            adapter_type=DataSourceType.REST_API,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "search_endpoint": {"type": "string"},
                    "page_size": {"type": "integer", "default": 20},
                    "auth_token": {"type": "string",
                                   "description": "Semantic Scholar API Key (可选)"},
                },
                "required": ["base_url"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://api.semanticscholar.org/api-docs/graph",
            changelog={"1.0.0": "初始实现, 论文+作者双流"},
        )

    def _do_discover(self) -> DiscoverResult:
        return DiscoverResult(
            streams=list(self._STREAMS.values()),
            adapter_id=self.config.id,
        )

    def _get_schema(self) -> DataSourceSchema:
        return self._STREAMS[self._PRIMARY_STREAM]

    def _build_search_url(
        self, query: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 Semantic Scholar 论文搜索 URL (fields 参数)."""
        url = f"{self.config.base_url}{self._search_endpoint}"
        params: dict[str, Any] = {
            "query": query,
            "limit": kwargs.get("limit", self._page_size),
            "offset": kwargs.get("offset", 0),
            "fields": "title,abstract,authors,year,venue,externalIds,fieldsOfStudy,"
                      "citationCount,referenceCount,influentialCitationCount,"
                      "openAccessPdf,tldr",
        }
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        if self._auth_token:
            headers["x-api-key"] = self._auth_token
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建按 paper_id 获取论文的 URL."""
        url = f"{self.config.base_url}{self._fetch_endpoint}/{resource_id}"
        params: dict[str, Any] = {
            "fields": "title,abstract,authors,year,venue,externalIds,fieldsOfStudy,"
                      "citationCount,referenceCount,influentialCitationCount,"
                      "openAccessPdf,tldr",
        }
        headers = {"Accept": "application/json", "User-Agent": "dy3-polaris/1.0"}
        if self._auth_token:
            headers["x-api-key"] = self._auth_token
        return url, params, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 Semantic Scholar Graph 响应.

        搜索返回 ``{"data": [...]}``; 单条获取返回 ``{"data": {...}}`` 或对象本身。
        字段名采用 camelCase (citationCount 等), 映射到 snake_case 标准字段。
        """
        if isinstance(data, list):
            return self._normalize_records(data)
        if isinstance(data, dict):
            inner = data.get("data")
            if isinstance(inner, list):
                return self._normalize_records(inner)
            if isinstance(inner, dict):
                return self._normalize_records([inner])
            for key in ("results", "items", "papers"):
                if key in data and isinstance(data[key], list):
                    return self._normalize_records(data[key])
            return self._normalize_records([data])
        return []

    def _normalize_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将 Semantic Scholar camelCase 字段规范化为 snake_case 标准字段."""
        key_map = {
            "paperId": "paper_id",
            "externalIds": "external_ids",
            "fieldsOfStudy": "fields_of_study",
            "citationCount": "citation_count",
            "referenceCount": "reference_count",
            "influentialCitationCount": "influential_citation_count",
            "openAccessPdf": "open_access_pdf_url",
            "tldr": "tldr",
        }
        out: list[dict[str, Any]] = []
        for rec in records:
            norm: dict[str, Any] = {}
            for k, v in rec.items():
                norm[key_map.get(k, k)] = v
            # tldr 可能是 {"model": ..., "text": ...}
            tldr = norm.get("tldr")
            if isinstance(tldr, dict):
                norm["tldr"] = tldr.get("text", "")
            # openAccessPdf 可能是 {"url": ...}
            oap = norm.get("open_access_pdf_url")
            if isinstance(oap, dict):
                norm["open_access_pdf_url"] = oap.get("url", "")
            out.append(norm)
        return out

    @classmethod
    def create(cls, auth_token: str = "") -> SemanticScholarAdapter:
        """创建预配置的 Semantic Scholar 适配器实例 (含 mock 数据).

        Args:
            auth_token: Semantic Scholar API Key (可选, 提升限流至 100/min)
        """
        config = ConnectorConfig(
            id="semantic-scholar",
            name="Semantic Scholar Graph API",
            tier=ConnectorTier.PUBLIC,
            protocol=ConnectorProtocol.HTTPS,
            base_url="https://api.semanticscholar.org/graph/v1",
            rate_limit=100 if auth_token else 10,
            cache_ttl=3600,
            version="1.0.0",
            tags=["scholarly", "citation-graph", "ai"],
            description="Semantic Scholar 学术论文图谱 API",
        )
        instance = cls(
            config,
            search_endpoint="/paper/search",
            fetch_endpoint="/paper",
            page_size=20,
            auth_token=auth_token,
        )
        # mock 数据已使用 snake_case 标准字段名 (经 _normalize 后保持不变)
        instance.set_mock_data(
            [
                {
                    "paper_id": "10cf2d4c209b16e54e18f4c5f56f4dba0bf0c956",
                    "title": "Attention Is All You Need",
                    "abstract": "The dominant sequence transduction models are based on "
                                "complex recurrent or convolutional neural networks.",
                    "authors": ["Ashish Vaswani", "Noam Shazeer"],
                    "year": 2017,
                    "venue": "NeurIPS",
                    "external_ids": {"DOI": "10.48550/arXiv.1706.03762",
                                     "ArXiv": "1706.03762"},
                    "fields_of_study": ["Computer Science"],
                    "citation_count": 120000,
                    "reference_count": 35,
                    "influential_citation_count": 8500,
                    "open_access_pdf_url": "https://arxiv.org/pdf/1706.03762",
                    "tldr": (
                        "Introduces the Transformer architecture relying "
                        "entirely on attention."
                    ),
                },
                {
                    "paper_id": "0f7c3c1d8e5a4b2f9c1e7d3a6b8f5e2d1c4a9b7e",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "abstract": "We introduce a new language representation model called BERT.",
                    "authors": ["Jacob Devlin", "Ming-Wei Chang"],
                    "year": 2019,
                    "venue": "NAACL",
                    "external_ids": {"DOI": "10.18653/v1/N19-1423"},
                    "fields_of_study": ["Computer Science"],
                    "citation_count": 95000,
                    "reference_count": 58,
                    "influential_citation_count": 7200,
                    "open_access_pdf_url": "https://arxiv.org/pdf/1810.04805",
                    "tldr": "Pre-trains a bidirectional Transformer for language understanding.",
                },
                {
                    "paper_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
                    "title": "Dy3+ spectroscopy in novel host lattices",
                    "abstract": "A systematic study of Dy3+ luminescence in fluoride and "
                                "oxide host lattices is presented.",
                    "authors": ["R. Gupta", "S. Patel"],
                    "year": 2021,
                    "venue": "Journal of Luminescence",
                    "external_ids": {"DOI": "10.1016/j.jlumin.2021.00099"},
                    "fields_of_study": ["Materials Science", "Physics"],
                    "citation_count": 38,
                    "reference_count": 41,
                    "influential_citation_count": 5,
                    "open_access_pdf_url": "",
                    "tldr": "Investigates Dy3+ emission properties across host crystals.",
                },
            ]
        )
        return instance


# ============================================================
# 模块级注册: Tier-1 公共适配器工厂表
# ============================================================


PUBLIC_TIER1_FACTORIES: dict[str, Callable[[], DataAdapterBase]] = {
    "nist-webbook": NISTWebBookAdapter.create,
    "pubchem": PubChemAdapter.create,
    "arxiv": ArxivAdapter.create,
    "wikipedia": WikipediaAdapter.create,
    "openalex": OpenAlexAdapter.create,
    "crossref": CrossRefAdapter.create,
    "doaj": DOAJAdapter.create,
    "uniprot": UniProtAdapter.create,
    "chemspider": ChemSpiderAdapter.create,
    "semantic-scholar": SemanticScholarAdapter.create,
}


__all__ = [
    # 中间基类与辅助
    "Tier1PublicAdapterBase",
    "describe_adapter",
    "PUBLIC_TIER1_FACTORIES",
    # 10 个具体适配器
    "NISTWebBookAdapter",
    "PubChemAdapter",
    "ArxivAdapter",
    "WikipediaAdapter",
    "OpenAlexAdapter",
    "CrossRefAdapter",
    "DOAJAdapter",
    "UniProtAdapter",
    "ChemSpiderAdapter",
    "SemanticScholarAdapter",
]
