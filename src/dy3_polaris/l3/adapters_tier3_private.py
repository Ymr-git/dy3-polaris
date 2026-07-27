"""L3 领域知识层 — Tier-3 校园/私有数据源适配器.

本模块实现四个面向校园内部场景的具体数据源适配器，对应连接器分层架构中的
PRIVATE (Tier-3) 层级。这些适配器封装了高校与研究机构内部常见的业务系统，
将分散在异构系统中的数据统一摄入 L3 知识层。

================================================================
Tier-3 私有数据源策略
================================================================

1. 内部访问 (Internal Access):
   - 数据源部署在校园内网或 VPN 环境中，不对外公开。
   - 网络可达性高、延迟低，适合大批量近实时同步。
   - 认证方式多样: 校园 SSO (CAS/OAuth2)、数据库凭据、文件系统权限。

2. 自定义认证 (Custom Authentication):
   - 不同于 Tier-1 的 API Key 和 Tier-2 的商业授权，Tier-3 适配器
     需对接校园身份联邦 (Shibboleth/SAML/CAS) 或数据库连接池凭据。
   - 敏感凭据通过 auth_config / connection_string 注入，不硬编码。

3. PII 保护 (Personally Identifiable Information Protection):
   - 教务系统中的学生姓名、学号、成绩等属于 PII 数据。
   - 适配器在 SchemaMapper 层对敏感字段进行标记和脱敏，
     下游可通过 AccessControlManager 实现行/列级访问控制。
   - 审计轨迹 (AuditTrail) 记录所有读取操作，满足合规要求。

4. 权威度与相关性权衡 (Authority vs. Relevance):
   - Tier-3 数据源权威度较低 (T3/T4)，不如 NIST/PubChem 等公共权威源。
   - 但对校园用户而言相关性极高: 本校课程、本校实验、本校馆藏
     是师生日常学习科研的直接数据。
   - L3 检索引擎在排序时通过 QualityBoost + MetadataBoost 权衡两者。

5. 同步模式 (Sync Modes):
   - 全量刷新 (FULL_REFRESH): 初始导入、数据校准。
   - 增量同步 (INCREMENTAL): 基于 updated_at 游标的日常增量同步。
   - 变更数据捕获 (CDC): LIMS 通过审计表触发器实现近实时 CDC。

================================================================
四个适配器概览
================================================================

1. LibraryOPACAdapter (图书馆 OPAC 系统):
   - 协议: REST/HTTPS (SIP2/Z39.50 封装为 REST)
   - 流: books / journals / theses / course_reserves
   - 特性: 增量同步 (catalog updates)、SSO Basic 认证、60/min 限流

2. LIMSAdapter (实验室信息管理系统):
   - 协议: Database (PostgreSQL)
   - 流: experiments / samples / instruments / results / protocols
   - 特性: CDC 审计表模式、增量同步、连接池、内网无限制

3. AcademicAffairsAdapter (教务管理系统):
   - 协议: Database (MySQL/PostgreSQL)
   - 流: courses / enrollments / students / grades / curricula / schedules
   - 特性: 增量同步、PII 保护标记、内网无限制

4. InternalDocRepositoryAdapter (内部文档库):
   - 协议: File (文件系统 / 网络挂载)
   - 流: documents / presentations / reports / theses
   - 特性: 多格式解析 (JSON/CSV/Markdown/TXT)、分块读取、文件元数据提取

融合方案:
- Airbyte Connector: spec/check/discover/read 四阶段标准化生命周期
- SeaTunnel CDC: 审计表触发器 + 快照增量混合同步 (LIMSAdapter)
- LangChain Document Loader: 多格式文件统一解析 (InternalDocRepositoryAdapter)
- SQLAlchemy: 数据库查询构建 + 连接池 (LIMSAdapter, AcademicAffairsAdapter)
- Debezium: 变更数据捕获 + 日志位置偏移量 (LIMSAdapter CDC 模式)
- Z39.50/SIP2-to-REST 网关: 图书馆 OPAC 现代化封装 (LibraryOPACAdapter)

线程安全: 所有适配器继承基类的 threading.RLock 保护。
所有网络/数据库/文件调用均为模拟实现，接口设计支持未来替换为真实后端。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
from typing import Any

from .adapter_bases import DatabaseAdapter, FileAdapter, RESTAdapter
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
# 适配器 1: LibraryOPACAdapter (图书馆 OPAC 系统)
# ============================================================


class LibraryOPACAdapter(RESTAdapter):
    """图书馆 OPAC (Online Public Access Catalog) 系统适配器.

    封装高校图书馆在线目录查询系统，将馆藏书目、期刊、学位论文和
    课程指定参考书数据统一摄入 L3 知识层。

    现实场景:
        大学图书馆的 OPAC 系统通常基于 MARC 21 / UNIMARC 编目标准，
        通过 SIP2 (Standard Interchange Protocol) 或 Z39.50 协议提供
        查询接口。许多现代图书馆系统 (如 Alma、Sierra、Koha) 已将
        上述协议封装为 RESTful API，本适配器对接该 REST 层。

    校园上下文:
        - 图书馆是校园知识基础设施的核心，馆藏数据直接影响师生
          文献检索、课程参考书管理、学位论文归档等业务。
        - OPAC 数据包含图书在馆状态、预约情况，支持实时馆藏查询。
        - 与教务系统的 course_reserves 流联动，支撑课程参考书服务。

    数据流 (Streams):
        - books: 图书书目记录 (含 ISBN、索书号、馆藏位置、可借状态)
        - journals: 期刊连续出版物记录 (含 ISSN、卷期、馆藏年代)
        - theses: 本校学位论文记录 (含导师、答辩日期、全文链接)
        - course_reserves: 课程指定参考书 (关联课程 ID、教员、保留期限)

    认证:
        校园 SSO (CAS/Shibboleth) 集成，通过 Basic Auth 传递 SSO 票据。
        auth_token 为 SSO 系统签发的 base64 编码凭据。

    同步策略:
        - FULL_REFRESH: 全量导入馆藏目录 (首次部署或数据校准)
        - INCREMENTAL: 基于 updated_at 游标同步编目变更
          (新书入库、状态变更、编目修正)

    限流:
        60 次请求/分钟，避免对图书馆系统造成压力。

    融合方案:
        - Airbyte HTTP Source: REST 搜索 + 分页 + 增量游标
        - Z39.50/SIP2-to-REST 网关: 传统图书馆协议现代化封装
        - MARC 21 字段映射: 245$a→title, 100$a→author, 020$a→ISBN 等
    """

    #: 支持的数据流名称
    STREAMS: list[str] = ["books", "journals", "theses", "course_reserves"]

    #: OPAC 搜索支持的过滤字段
    SEARCH_FIELDS: list[str] = ["title", "author", "isbn", "call_number", "keyword"]

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        search_endpoint: str = "/v1/search",
        fetch_endpoint: str = "/v1/records/{id}",
        page_size: int = 20,
        auth_type: str = "basic",
        auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        """初始化图书馆 OPAC 适配器.

        Args:
            config: 连接器配置 (base_url 为 OPAC REST API 地址)
            search_endpoint: 搜索端点路径
            fetch_endpoint: 单条记录获取端点 (含 {id} 占位符)
            page_size: 分页大小 (默认 20 条/页)
            auth_type: 认证类型 (默认 basic，对应 SSO 票据)
            auth_token: SSO base64 凭据
            **kwargs: 传递给 RESTAdapter 的额外参数
        """
        super().__init__(
            config,
            search_endpoint=search_endpoint,
            fetch_endpoint=fetch_endpoint,
            page_size=page_size,
            auth_type=auth_type,
            auth_token=auth_token,
            **kwargs,
        )
        self._schema_mapper = self._build_schema_mapper()

    # ---- Schema 映射 ----

    def _build_schema_mapper(self) -> SchemaMapper:
        """构建 OPAC 字段到 L3 标准字段的映射器.

        将 MARC 21 编目字段映射到 L3 知识实体标准字段:
        - record_id → entity_id (书目记录唯一标识)
        - title → entity_name (实体显示名称)
        - format_type → entity_type (实体类型: book/journal/thesis)
        - isbn/issn → identifiers (标识符字典)
        - authors/publisher/... → properties (属性字典)
        - 馆藏 URL → source_uri
        - abstract → description
        """
        mapper = SchemaMapper()
        mapper.add_mapping(FieldMapping(
            source_field="record_id", target_field="entity_id", required=True,
            description="书目记录唯一标识 (MARC 001 控制号)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="title", target_field="entity_name", required=True,
            transform="trim", description="题名 (MARC 245$a)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="format_type", target_field="entity_type",
            default_value="book", description="文献类型",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="isbn", target_field="identifiers.isbn",
            description="ISBN (MARC 020$a)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="issn", target_field="identifiers.issn",
            description="ISSN (MARC 022$a)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="call_number", target_field="identifiers.call_number",
            description="索书号 (MARC 090/092)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="authors", target_field="properties.authors",
            transform="to_list", description="作者列表 (MARC 100/700)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="publisher", target_field="properties.publisher",
            description="出版者 (MARC 260$b)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="publication_year", target_field="properties.publication_year",
            transform="parse_int", description="出版年 (MARC 260$c)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="edition", target_field="properties.edition",
            description="版次 (MARC 250)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="language", target_field="properties.language",
            description="语种 (MARC 008/041)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="availability", target_field="properties.availability",
            description="在馆状态",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="location", target_field="properties.location",
            description="馆藏位置",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="subject_headings", target_field="properties.subject_headings",
            transform="split_comma", description="主题词 (MARC 650)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="abstract", target_field="description",
            description="摘要 (MARC 520)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="record_id", target_field="source_uri",
            transform="to_lower",
            default_value="",
            description="OPAC 记录 URI (由 record_id 构造)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="updated_at", target_field="properties.updated_at",
            transform="iso_datetime", description="编目更新时间 (增量游标)",
        ))
        return mapper

    # ---- 协议方法重写 ----

    def _do_spec(self) -> AdapterSpec:
        """声明 OPAC 适配器规范.

        在 REST 基类能力基础上增加 INCREMENTAL 能力，
        支持基于 updated_at 游标的增量同步。
        """
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.INCREMENTAL
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
                    "base_url": {
                        "type": "string",
                        "format": "uri",
                        "description": "OPAC REST API 基础地址",
                    },
                    "search_endpoint": {
                        "type": "string",
                        "default": "/v1/search",
                    },
                    "fetch_endpoint": {
                        "type": "string",
                        "default": "/v1/records/{id}",
                    },
                    "page_size": {"type": "integer", "default": 20},
                    "auth_type": {
                        "type": "string",
                        "enum": ["basic"],
                        "description": "校园 SSO Basic 认证",
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "SSO base64 凭据",
                    },
                },
                "required": ["base_url", "auth_token"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://library.university.edu/api/docs",
            changelog={"1.0.0": "初始版本，支持书目/期刊/论文/教参四流"},
        )

    def _do_check(self) -> bool:
        """验证 OPAC 连通性: 检查 base_url 非空且认证令牌已配置."""
        return bool(self.config.base_url) and bool(self._auth_token)

    def _do_discover(self) -> DiscoverResult:
        """发现 OPAC Schema: 返回四个流的完整字段定义."""
        streams = [self._get_schema_for_stream(s) for s in self.STREAMS]
        return DiscoverResult(streams=streams, adapter_id=self.config.id)

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        """读取 OPAC 数据: 模拟 REST 请求并支持增量过滤."""
        records = self._mock_request(limit=limit or self._page_size)

        # 增量同步: 基于 updated_at 游标过滤已读记录
        if sync_mode == SyncMode.INCREMENTAL and checkpoint and checkpoint.cursor_value:
            records = [
                r for r in records
                if str(r.get("updated_at", r.get("record_id", ""))) > checkpoint.cursor_value
            ]

        cursor_value = ""
        if records:
            cursor_value = str(
                records[-1].get("updated_at", records[-1].get("record_id", ""))
            )

        checkpoint_result = self._make_checkpoint(
            stream_name=stream_name or "books",
            records_read=len(records),
            cursor_value=cursor_value,
        )
        return ReadResult(
            records=records,
            checkpoint=checkpoint_result,
            has_more=False,
        )

    # ---- URL 构建 ----

    def _build_search_url(
        self, query: str, **kwargs: Any,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 OPAC 搜索请求 URL.

        支持 title/author/isbn/call_number/keyword 多字段搜索。
        端点路径包含流名称以区分书目类型。

        Args:
            query: 搜索关键词
            **kwargs: 可选参数 (stream, field, limit, offset)

        Returns:
            (url, params, headers) 三元组
        """
        stream = kwargs.get("stream", "books")
        field = kwargs.get("field", "keyword")
        url = f"{self.config.base_url}{self._search_endpoint}/{stream}"
        params: dict[str, Any] = {
            field: query,
            "limit": kwargs.get("limit", self._page_size),
            "offset": kwargs.get("offset", 0),
        }
        headers = self._build_auth_headers()
        return url, params, headers

    def _build_fetch_url(
        self, resource_id: str,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """构建 OPAC 单条记录获取 URL.

        Args:
            resource_id: 书目记录 ID (MARC 控制号)

        Returns:
            (url, params, headers) 三元组
        """
        endpoint = self._fetch_endpoint.replace("{id}", str(resource_id))
        url = f"{self.config.base_url}{endpoint}"
        headers = self._build_auth_headers()
        return url, {}, headers

    def _parse_response(self, data: Any) -> list[dict[str, Any]]:
        """解析 OPAC REST 响应.

        OPAC API 返回格式: {"records": [...], "total": N, "page": P}
        提取 records 列表并标准化字段。
        """
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("records", "results", "data", "items", "hits"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []

    # ---- Schema 定义 ----

    def _get_schema(self) -> DataSourceSchema:
        """返回默认流 (books) 的 Schema."""
        return self._get_schema_for_stream("books")

    def _get_schema_for_stream(self, stream_name: str) -> DataSourceSchema:
        """返回指定流的 Schema 定义.

        Args:
            stream_name: 流名称 (books/journals/theses/course_reserves)

        Returns:
            该流的完整 Schema
        """
        common_fields = [
            SchemaField(
                name="record_id", data_type="string", nullable=False,
                primary_key=True, description="书目记录唯一标识 (MARC 001)",
                max_length=64,
            ),
            SchemaField(
                name="title", data_type="string", nullable=False,
                description="题名 (MARC 245$a)", max_length=512,
            ),
            SchemaField(
                name="authors", data_type="array", nullable=True,
                description="作者列表 (MARC 100/700)",
            ),
            SchemaField(
                name="isbn", data_type="string", nullable=True,
                description="ISBN (MARC 020$a)", max_length=20,
            ),
            SchemaField(
                name="issn", data_type="string", nullable=True,
                description="ISSN (MARC 022$a)", max_length=20,
            ),
            SchemaField(
                name="call_number", data_type="string", nullable=True,
                description="索书号 (MARC 090/092)", max_length=32,
            ),
            SchemaField(
                name="publisher", data_type="string", nullable=True,
                description="出版者 (MARC 260$b)", max_length=256,
            ),
            SchemaField(
                name="publication_year", data_type="integer", nullable=True,
                description="出版年 (MARC 260$c)",
            ),
            SchemaField(
                name="edition", data_type="string", nullable=True,
                description="版次 (MARC 250)", max_length=64,
            ),
            SchemaField(
                name="location", data_type="string", nullable=True,
                description="馆藏位置 (分馆/阅览室)", max_length=128,
            ),
            SchemaField(
                name="availability", data_type="string", nullable=True,
                description="在馆状态 (available/checked_out/on_hold/lost)",
                enum_values=["available", "checked_out", "on_hold", "lost", "processing"],
            ),
            SchemaField(
                name="due_date", data_type="datetime", nullable=True,
                description="应还日期 (借出时)", format="date-time",
            ),
            SchemaField(
                name="hold_count", data_type="integer", nullable=True,
                description="预约人数", default_value=0,
            ),
            SchemaField(
                name="subject_headings", data_type="array", nullable=True,
                description="主题词 (MARC 650)",
            ),
            SchemaField(
                name="abstract", data_type="string", nullable=True,
                description="摘要 (MARC 520)",
            ),
            SchemaField(
                name="language", data_type="string", nullable=True,
                description="语种 (ISO 639-2)", max_length=3,
            ),
            SchemaField(
                name="format_type", data_type="string", nullable=False,
                description="文献类型", default_value="book",
                enum_values=["book", "journal", "thesis", "course_reserve", "ebook"],
            ),
            SchemaField(
                name="updated_at", data_type="datetime", nullable=False,
                description="编目更新时间 (增量游标)", format="date-time",
            ),
        ]

        descriptions = {
            "books": "图书书目记录 (含 ISBN、索书号、馆藏位置、可借状态)",
            "journals": "期刊连续出版物记录 (含 ISSN、卷期、馆藏年代)",
            "theses": "本校学位论文记录 (含导师、答辩日期、全文链接)",
            "course_reserves": "课程指定参考书 (关联课程 ID、教员、保留期限)",
        }

        return DataSourceSchema(
            stream_name=stream_name,
            fields=common_fields,
            primary_keys=["record_id"],
            cursor_field="updated_at",
            description=descriptions.get(stream_name, f"OPAC stream: {stream_name}"),
            metadata={
                "source": "library_opac",
                "encoding": "MARC21",
                "protocol": "REST",
            },
        )

    # ---- 工厂方法 ----

    @classmethod
    def create(
        cls,
        base_url: str = "",
        auth_token: str = "",
    ) -> LibraryOPACAdapter:
        """创建预配置的图书馆 OPAC 适配器实例.

        Args:
            base_url: OPAC REST API 基础地址
                      (默认 https://library.university.edu/api)
            auth_token: SSO base64 认证凭据

        Returns:
            配置完成的 LibraryOPACAdapter 实例 (含模拟数据)
        """
        config = ConnectorConfig(
            id="campus-library-opac",
            name="University Library OPAC System",
            tier=ConnectorTier.PRIVATE,
            protocol=ConnectorProtocol.HTTPS,
            base_url=base_url or "https://library.university.edu/api",
            auth_config={"type": "basic", "description": "校园 SSO Basic 认证"},
            rate_limit=60,
            cache_ttl=120,
            version="1.0.0",
            owner="library-system",
            tags=["library", "opac", "catalog", "campus"],
            description="图书馆在线目录查询系统 (SIP2/Z39.50 REST 封装)",
        )
        instance = cls(
            config,
            search_endpoint="/v1/search",
            fetch_endpoint="/v1/records/{id}",
            page_size=20,
            auth_type="basic",
            auth_token=auth_token,
        )
        instance.set_mock_data(_OPAC_MOCK_DATA)
        return instance


#: 图书馆 OPAC 模拟数据 (MARC 字段已映射为 REST JSON)
_OPAC_MOCK_DATA: list[dict[str, Any]] = [
    {
        "record_id": "BIB-000001",
        "title": "稀土发光材料: 合成、表征与应用",
        "authors": ["张明远", "李华清"],
        "isbn": "978-7-03-065432-1",
        "issn": "",
        "call_number": "TB342/ZMY",
        "publisher": "科学出版社",
        "publication_year": 2021,
        "edition": "第2版",
        "location": "主馆-中文科技图书区-3楼-A区",
        "availability": "available",
        "due_date": "",
        "hold_count": 0,
        "subject_headings": ["稀土材料", "发光材料", "光谱学", "纳米材料"],
        "abstract": "本书系统介绍了稀土发光材料的合成方法、表征技术和应用领域，涵盖 Dy3+、Eu3+、Tb3+ 等稀土离子的发光机理与能级跃迁。",
        "language": "chi",
        "format_type": "book",
        "updated_at": "2025-03-15T10:30:00Z",
    },
    {
        "record_id": "BIB-000002",
        "title": "Journal of Luminescence",
        "authors": ["Elsevier"],
        "isbn": "",
        "issn": "0022-2313",
        "call_number": "PER-JL/ELSE",
        "publisher": "Elsevier B.V.",
        "publication_year": 2024,
        "edition": "Vol. 265",
        "location": "主馆-外文期刊区-2楼-B区",
        "availability": "available",
        "due_date": "",
        "hold_count": 2,
        "subject_headings": ["发光学", "荧光", "磷光", "光谱学"],
        "abstract": "发光学领域国际权威期刊，涵盖有机/无机发光材料、生物发光、化学发光等研究方向。",
        "language": "eng",
        "format_type": "journal",
        "updated_at": "2025-06-01T08:00:00Z",
    },
    {
        "record_id": "BIB-000003",
        "title": "Dy3+ 掺杂氟化物纳米晶体的发光性质研究",
        "authors": ["王小红"],
        "isbn": "",
        "issn": "",
        "call_number": "THS-WXH/2024",
        "publisher": "XX大学",
        "publication_year": 2024,
        "edition": "博士学位论文",
        "location": "主馆-学位论文区-4楼-C区",
        "availability": "checked_out",
        "due_date": "2025-08-20T23:59:59Z",
        "hold_count": 3,
        "subject_headings": ["稀土离子", "氟化物", "纳米晶体", "发光性质", "Dy3+"],
        "abstract": "本文研究了 Dy3+ 掺杂 NaYF4、CaF2、BaF2 等氟化物纳米晶体的发光性质，分析了浓度淬灭效应和能量传递机制。",
        "language": "chi",
        "format_type": "thesis",
        "updated_at": "2025-06-15T14:20:00Z",
    },
    {
        "record_id": "BIB-000004",
        "title": "无机材料科学基础 (课程参考书)",
        "authors": ["陈志强", "刘芳"],
        "isbn": "978-7-04-051234-5",
        "issn": "",
        "call_number": "TB301/CZQ-RES",
        "publisher": "高等教育出版社",
        "publication_year": 2019,
        "edition": "第3版",
        "location": "教参书阅览室-1楼-D区",
        "availability": "on_hold",
        "due_date": "",
        "hold_count": 5,
        "subject_headings": ["无机材料", "材料科学", "晶体结构", "教学参考"],
        "abstract": "材料科学与工程专业核心课程参考书，涵盖晶体结构、缺陷化学、相图分析等内容。",
        "language": "chi",
        "format_type": "course_reserve",
        "updated_at": "2025-07-01T09:15:00Z",
    },
    {
        "record_id": "BIB-000005",
        "title": "Spectroscopy of Rare Earth Doped Materials",
        "authors": ["Blasse, G.", "Grabmaier, B.C."],
        "isbn": "978-3-540-19953-1",
        "issn": "",
        "call_number": "TB342/BLA",
        "publisher": "Springer-Verlag",
        "publication_year": 1994,
        "edition": "1st ed.",
        "location": "主馆-外文科技图书区-3楼-E区",
        "availability": "available",
        "due_date": "",
        "hold_count": 1,
        "subject_headings": ["rare earth", "spectroscopy", "luminescence", "Judd-Ofelt"],
        "abstract": "Classic reference on rare earth doped material spectroscopy, covering Judd-Ofelt theory, energy transfer, and applications.",
        "language": "eng",
        "format_type": "book",
        "updated_at": "2025-01-10T11:45:00Z",
    },
]


# ============================================================
# 适配器 2: LIMSAdapter (实验室信息管理系统)
# ============================================================


class LIMSAdapter(DatabaseAdapter):
    """实验室信息管理系统 (LIMS) 适配器.

    封装高校/研究机构实验室的 LIMS 系统，将实验记录、样品管理、
    仪器数据和实验结果统一摄入 L3 知识层。

    现实场景:
        LIMS (Laboratory Information Management System) 是实验室
        数字化管理的核心系统，负责实验流程追踪、样品全生命周期管理、
        仪器资源调度和实验结果归档。典型产品包括 LabWare、SampleLIMS、
        LabVantage 等，多采用 PostgreSQL 作为后端数据库。

    校园上下文:
        - 材料化学实验室通过 LIMS 记录 Dy3+ 掺杂荧光粉的合成实验
          (配方、烧结温度、表征条件) 和发光性能测试结果。
        - 实验数据是论文发表和专利申请的原始证据，需保证可追溯性。
        - LIMS 数据与图书馆 OPAC (文献) 和教务系统 (课程实验)
          形成知识闭环: 实验数据 → 论文发表 → 课程教学。

    数据流 (Streams):
        - experiments: 实验记录 (实验名称、研究者、状态、结果)
        - samples: 样品管理 (样品编号、名称、来源、存储位置)
        - instruments: 仪器设备 (仪器编号、名称、型号、校准状态)
        - results: 实验结果 (测量值、质量标志、原始数据引用)
        - protocols: 实验规程 (标准操作流程 SOP、方法参数)

    CDC 模式 (审计表触发器):
        LIMS 通过 PostgreSQL 触发器在审计表 (audit_log) 中记录
        所有 INSERT/UPDATE/DELETE 操作。CDC 模式读取审计表获取
        变更事件，实现近实时数据同步，无需全量扫描业务表。

        审计表结构 (示意):
            CREATE TABLE audit_log (
                audit_id    BIGSERIAL PRIMARY KEY,
                table_name  VARCHAR(64) NOT NULL,
                operation   CHAR(1) NOT NULL,  -- I/U/D
                record_id   VARCHAR(64) NOT NULL,
                changed_at  TIMESTAMP NOT NULL DEFAULT NOW(),
                changed_by  VARCHAR(64),
                old_data    JSONB,
                new_data    JSONB
            );

    认证:
        PostgreSQL 数据库凭据嵌入 connection_string:
        postgresql://user:password@host:5432/lims

    融合方案:
        - SQLAlchemy: 连接池 + 查询构建 + 参数化查询 (防 SQL 注入)
        - SeaTunnel CDC: 审计表触发器 + 快照增量混合模型
        - Debezium: 变更事件结构 (op/before/after/source)
    """

    #: 支持的数据流 (表名)
    STREAMS: list[str] = [
        "experiments", "samples", "instruments", "results", "protocols",
    ]

    #: 搜索支持的过滤字段
    SEARCH_FIELDS: list[str] = [
        "experiment_id", "sample_name", "instrument", "date_range", "researcher",
    ]

    #: CDC 审计表名称
    CDC_AUDIT_TABLE: str = "audit_log"

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        connection_string: str = "",
        default_table: str = "experiments",
        pool_size: int = 3,
        **kwargs: Any,
    ) -> None:
        """初始化 LIMS 适配器.

        Args:
            config: 连接器配置 (base_url 可作为连接字符串后备)
            connection_string: PostgreSQL 连接字符串
            default_table: 默认表名 (experiments)
            pool_size: 连接池大小 (实验室数据量适中，默认 3)
            **kwargs: 传递给 DatabaseAdapter 的额外参数
        """
        super().__init__(
            config,
            connection_string=connection_string,
            default_table=default_table,
            pool_size=pool_size,
            **kwargs,
        )
        self._schema_mapper = self._build_schema_mapper()

    # ---- Schema 映射 ----

    def _build_schema_mapper(self) -> SchemaMapper:
        """构建 LIMS 字段到 L3 标准字段的映射器.

        将实验记录字段映射到 L3 知识实体标准字段:
        - experiment_id → entity_id (实验唯一标识)
        - experiment_name → entity_name (实验名称)
        - "experiment" → entity_type (固定实体类型)
        - sample_id/instrument_id → identifiers (标识符字典)
        - researcher/result_data/... → properties (属性字典)
        - 数据库 URI → source_uri
        """
        mapper = SchemaMapper()
        mapper.add_mapping(FieldMapping(
            source_field="experiment_id", target_field="entity_id", required=True,
            description="实验唯一标识",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="experiment_name", target_field="entity_name", required=True,
            transform="trim", description="实验名称",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="experiment_id", target_field="entity_type",
            default_value="experiment", description="实体类型固定为 experiment",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="sample_id", target_field="identifiers.sample_id",
            description="样品编号",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="instrument_id", target_field="identifiers.instrument_id",
            description="仪器编号",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="researcher_id", target_field="properties.researcher_id",
            required=True, description="研究者工号",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="sample_name", target_field="properties.sample_name",
            description="样品名称",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="instrument_name", target_field="properties.instrument_name",
            description="仪器名称",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="method_protocol", target_field="properties.method_protocol",
            description="实验方法规程",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="start_time", target_field="properties.start_time",
            transform="iso_datetime", description="实验开始时间",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="end_time", target_field="properties.end_time",
            transform="iso_datetime", description="实验结束时间",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="status", target_field="properties.status",
            description="实验状态",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="result_data", target_field="properties.result_data",
            transform="json_parse", description="实验结果数据 (JSON)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="quality_flag", target_field="properties.quality_flag",
            description="质量标志 (pass/fail/review)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="remarks", target_field="description",
            description="实验备注/描述",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="updated_at", target_field="properties.updated_at",
            transform="iso_datetime", description="记录更新时间 (增量游标)",
        ))
        return mapper

    # ---- 协议方法重写 ----

    def _do_spec(self) -> AdapterSpec:
        """声明 LIMS 适配器规范.

        在 Database 基类能力基础上增加 CDC 能力，
        支持基于审计表的变更数据捕获。
        """
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.SCHEMA_EVOLUTION
            | AdapterCapability.INCREMENTAL
            | AdapterCapability.CDC
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.AUTHENTICATE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.DATABASE,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "connection_string": {
                        "type": "string",
                        "description": "PostgreSQL 连接字符串",
                    },
                    "default_table": {
                        "type": "string",
                        "default": "experiments",
                    },
                    "pool_size": {"type": "integer", "default": 3},
                    "cdc_audit_table": {
                        "type": "string",
                        "default": "audit_log",
                        "description": "CDC 审计表名称",
                    },
                },
                "required": ["connection_string"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[
                SyncMode.FULL_REFRESH,
                SyncMode.INCREMENTAL,
                SyncMode.CDC,
                SyncMode.SNAPSHOT_THEN_INCREMENTAL,
            ],
            version=self.config.version,
            documentation_url="https://lims.university.edu/docs",
            changelog={
                "1.0.0": "初始版本，支持五流 + CDC 审计表模式",
                "1.1.0": "增加 SNAPSHOT_THEN_INCREMENTAL 混合模式",
            },
        )

    def _do_check(self) -> bool:
        """验证 LIMS 连通性: 检查连接字符串非空."""
        return bool(self._connection_string)

    def _do_discover(self) -> DiscoverResult:
        """发现 LIMS Schema: 返回五个流的表结构定义."""
        streams = [self._get_table_schema(s) for s in self.STREAMS]
        return DiscoverResult(streams=streams, adapter_id=self.config.id)

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        """读取 LIMS 数据: 构建 SQL 查询并执行模拟查询.

        支持三种同步模式:
        - FULL_REFRESH: 全量读取业务表
        - INCREMENTAL: 基于 updated_at 游标读取增量变更
        - CDC: 读取审计表获取变更事件
        """
        table = stream_name or self._default_table or "experiments"

        if sync_mode == SyncMode.CDC:
            sql, params = self._build_cdc_query(table, checkpoint, limit)
        else:
            sql, params = self._build_query(table, sync_mode, checkpoint, limit)

        records = self._mock_query(sql, params)

        # 增量/CDC 模式: 提取游标值
        cursor_value = ""
        if records and sync_mode in (SyncMode.INCREMENTAL, SyncMode.CDC):
            if sync_mode == SyncMode.CDC:
                cursor_value = str(records[-1].get("audit_id", records[-1].get("updated_at", "")))
            else:
                cursor_value = str(records[-1].get("updated_at", records[-1].get("experiment_id", "")))

        checkpoint_result = self._make_checkpoint(
            stream_name=table,
            records_read=len(records),
            cursor_value=cursor_value,
            offset=str(checkpoint.offset if checkpoint else ""),
        )
        return ReadResult(
            records=records,
            checkpoint=checkpoint_result,
            has_more=False,
        )

    # ---- SQL 查询构建 ----

    def _build_query(
        self,
        table: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> tuple[str, dict[str, Any]]:
        """构建业务表 SQL 查询.

        根据同步模式生成不同的 SQL:
        - FULL_REFRESH: SELECT * FROM {table} ORDER BY id LIMIT :limit
        - INCREMENTAL: SELECT * FROM {table} WHERE updated_at > :cursor
                       ORDER BY updated_at ASC LIMIT :limit

        Args:
            table: 表名
            sync_mode: 同步模式
            checkpoint: 检查点 (增量同步恢复点)
            limit: 最大记录数

        Returns:
            (sql, params) 二元组
        """
        # 列名映射: experiments 表用 experiment_id，其他表用 id
        id_column = "experiment_id" if table == "experiments" else "id"

        sql = f"SELECT * FROM {table}"
        params: dict[str, Any] = {}

        if sync_mode == SyncMode.INCREMENTAL and checkpoint and checkpoint.cursor_value:
            sql += " WHERE updated_at > :cursor"
            params["cursor"] = checkpoint.cursor_value
            sql += " ORDER BY updated_at ASC"
        else:
            sql += f" ORDER BY {id_column} ASC"

        if limit > 0:
            sql += " LIMIT :limit"
            params["limit"] = limit

        return sql, params

    def _build_cdc_query(
        self,
        table: str,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> tuple[str, dict[str, Any]]:
        """构建 CDC 审计表查询 SQL.

        读取审计表中指定表的变更事件:
        SELECT * FROM audit_log
        WHERE table_name = :table AND audit_id > :cursor
        ORDER BY audit_id ASC LIMIT :limit

        Args:
            table: 目标表名 (用于过滤审计记录)
            checkpoint: 检查点 (上次读取的 audit_id)
            limit: 最大记录数

        Returns:
            (sql, params) 二元组
        """
        sql = f"SELECT * FROM {self.CDC_AUDIT_TABLE}"
        params: dict[str, Any] = {"table_name": table}

        conditions = ["table_name = :table_name"]
        if checkpoint and checkpoint.cursor_value:
            conditions.append("audit_id > :cursor")
            params["cursor"] = int(checkpoint.cursor_value)

        sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY audit_id ASC"

        if limit > 0:
            sql += " LIMIT :limit"
            params["limit"] = limit

        return sql, params

    def _mock_query(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """模拟数据库查询.

        如果 SQL 包含 CDC_AUDIT_TABLE，返回审计事件格式的模拟数据;
        否则返回业务表模拟数据。支持增量游标过滤。
        """
        if self._mock_data is None:
            return []

        # CDC 模式: 返回审计事件
        if self.CDC_AUDIT_TABLE in sql:
            return [
                {
                    "audit_id": 1001,
                    "table_name": "experiments",
                    "operation": "U",
                    "record_id": "EXP-2024-0001",
                    "changed_at": "2025-07-20T10:15:00Z",
                    "changed_by": "researcher_001",
                    "new_data": self._mock_data[0] if self._mock_data else {},
                },
                {
                    "audit_id": 1002,
                    "table_name": "experiments",
                    "operation": "I",
                    "record_id": "EXP-2024-0003",
                    "changed_at": "2025-07-21T14:30:00Z",
                    "changed_by": "researcher_002",
                    "new_data": self._mock_data[2] if len(self._mock_data) > 2 else {},
                },
            ]

        # 增量模式: 过滤已读记录
        cursor = params.get("cursor")
        if cursor:
            return [
                r for r in self._mock_data
                if str(r.get("updated_at", r.get("experiment_id", ""))) > str(cursor)
            ]

        return self._mock_data

    # ---- Schema 定义 ----

    def _get_table_schema(self, table_name: str) -> DataSourceSchema:
        """返回指定表的 Schema 定义.

        基于 INFORMATION_SCHEMA 风格定义各表列结构。

        Args:
            table_name: 表名

        Returns:
            该表的完整 Schema
        """
        schemas = self._build_all_table_schemas()
        if table_name in schemas:
            return schemas[table_name]
        # 未知表: 返回默认 Schema
        return DataSourceSchema(
            stream_name=table_name,
            fields=[],
            description=f"Unknown table: {table_name}",
            cursor_field="updated_at",
        )

    def _build_all_table_schemas(self) -> dict[str, DataSourceSchema]:
        """构建所有表的 Schema 定义.

        定义 LIMS 五张业务表 + 审计表的列结构，
        模拟从 PostgreSQL INFORMATION_SCHEMA 查询结果。
        """
        experiment_fields = [
            SchemaField(name="experiment_id", data_type="string", nullable=False,
                        primary_key=True, description="实验唯一标识", max_length=32),
            SchemaField(name="experiment_name", data_type="string", nullable=False,
                        description="实验名称", max_length=256),
            SchemaField(name="researcher_id", data_type="string", nullable=False,
                        description="研究者工号", max_length=32),
            SchemaField(name="sample_id", data_type="string", nullable=True,
                        description="样品编号", max_length=32),
            SchemaField(name="sample_name", data_type="string", nullable=True,
                        description="样品名称", max_length=128),
            SchemaField(name="instrument_id", data_type="string", nullable=True,
                        description="仪器编号", max_length=32),
            SchemaField(name="instrument_name", data_type="string", nullable=True,
                        description="仪器名称", max_length=128),
            SchemaField(name="method_protocol", data_type="string", nullable=True,
                        description="实验方法规程", max_length=64),
            SchemaField(name="start_time", data_type="datetime", nullable=True,
                        description="实验开始时间", format="date-time"),
            SchemaField(name="end_time", data_type="datetime", nullable=True,
                        description="实验结束时间", format="date-time"),
            SchemaField(name="status", data_type="string", nullable=False,
                        description="实验状态", default_value="planned",
                        enum_values=["planned", "running", "completed", "failed", "cancelled"]),
            SchemaField(name="result_data", data_type="object", nullable=True,
                        description="实验结果数据 (JSON)"),
            SchemaField(name="quality_flag", data_type="string", nullable=True,
                        description="质量标志",
                        enum_values=["pass", "fail", "review", "pending"]),
            SchemaField(name="remarks", data_type="string", nullable=True,
                        description="实验备注"),
            SchemaField(name="created_at", data_type="datetime", nullable=False,
                        description="记录创建时间", format="date-time"),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="记录更新时间 (增量游标)", format="date-time"),
        ]

        sample_fields = [
            SchemaField(name="sample_id", data_type="string", nullable=False,
                        primary_key=True, description="样品唯一编号", max_length=32),
            SchemaField(name="sample_name", data_type="string", nullable=False,
                        description="样品名称", max_length=128),
            SchemaField(name="sample_type", data_type="string", nullable=True,
                        description="样品类型",
                        enum_values=["powder", "single_crystal", "thin_film", "solution", "bulk"]),
            SchemaField(name="source", data_type="string", nullable=True,
                        description="样品来源", max_length=128),
            SchemaField(name="storage_location", data_type="string", nullable=True,
                        description="存储位置", max_length=64),
            SchemaField(name="created_at", data_type="datetime", nullable=False,
                        description="入库时间", format="date-time"),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        instrument_fields = [
            SchemaField(name="instrument_id", data_type="string", nullable=False,
                        primary_key=True, description="仪器编号", max_length=32),
            SchemaField(name="instrument_name", data_type="string", nullable=False,
                        description="仪器名称", max_length=128),
            SchemaField(name="model", data_type="string", nullable=True,
                        description="型号", max_length=64),
            SchemaField(name="manufacturer", data_type="string", nullable=True,
                        description="制造商", max_length=128),
            SchemaField(name="calibration_status", data_type="string", nullable=True,
                        description="校准状态",
                        enum_values=["calibrated", "expired", "pending"]),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        result_fields = [
            SchemaField(name="result_id", data_type="string", nullable=False,
                        primary_key=True, description="结果编号", max_length=32),
            SchemaField(name="experiment_id", data_type="string", nullable=False,
                        description="关联实验 ID", max_length=32),
            SchemaField(name="measurement_type", data_type="string", nullable=True,
                        description="测量类型",
                        enum_values=["emission_spectrum", "excitation_spectrum",
                                     "XRD", "SEM", "TEM", "lifetime", "CIE"]),
            SchemaField(name="measured_value", data_type="float", nullable=True,
                        description="测量值"),
            SchemaField(name="unit", data_type="string", nullable=True,
                        description="单位", max_length=32),
            SchemaField(name="raw_data_ref", data_type="string", nullable=True,
                        description="原始数据文件引用", max_length=256),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        protocol_fields = [
            SchemaField(name="protocol_id", data_type="string", nullable=False,
                        primary_key=True, description="规程编号", max_length=32),
            SchemaField(name="protocol_name", data_type="string", nullable=False,
                        description="规程名称", max_length=256),
            SchemaField(name="category", data_type="string", nullable=True,
                        description="规程类别",
                        enum_values=["synthesis", "characterization",
                                     "measurement", "safety"]),
            SchemaField(name="steps", data_type="object", nullable=True,
                        description="操作步骤 (JSON)"),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        return {
            "experiments": DataSourceSchema(
                stream_name="experiments",
                fields=experiment_fields,
                primary_keys=["experiment_id"],
                cursor_field="updated_at",
                description="实验记录 (实验名称、研究者、状态、结果)",
                metadata={"engine": "postgresql", "rows_estimated": 5000},
            ),
            "samples": DataSourceSchema(
                stream_name="samples",
                fields=sample_fields,
                primary_keys=["sample_id"],
                cursor_field="updated_at",
                description="样品管理 (样品编号、名称、来源、存储位置)",
                metadata={"engine": "postgresql", "rows_estimated": 15000},
            ),
            "instruments": DataSourceSchema(
                stream_name="instruments",
                fields=instrument_fields,
                primary_keys=["instrument_id"],
                cursor_field="updated_at",
                description="仪器设备 (仪器编号、名称、型号、校准状态)",
                metadata={"engine": "postgresql", "rows_estimated": 80},
            ),
            "results": DataSourceSchema(
                stream_name="results",
                fields=result_fields,
                primary_keys=["result_id"],
                cursor_field="updated_at",
                description="实验结果 (测量值、质量标志、原始数据引用)",
                metadata={"engine": "postgresql", "rows_estimated": 50000},
            ),
            "protocols": DataSourceSchema(
                stream_name="protocols",
                fields=protocol_fields,
                primary_keys=["protocol_id"],
                cursor_field="updated_at",
                description="实验规程 (标准操作流程 SOP、方法参数)",
                metadata={"engine": "postgresql", "rows_estimated": 200},
            ),
        }

    # ---- 工厂方法 ----

    @classmethod
    def create(
        cls,
        connection_string: str = "",
        auth_token: str = "",
    ) -> LIMSAdapter:
        """创建预配置的 LIMS 适配器实例.

        Args:
            connection_string: PostgreSQL 连接字符串
                               (默认 postgresql://localhost:5432/lims)
            auth_token: 未使用 (LIMS 通过连接字符串认证)

        Returns:
            配置完成的 LIMSAdapter 实例 (含模拟数据)
        """
        conn = connection_string or "postgresql://localhost:5432/lims"
        config = ConnectorConfig(
            id="campus-lims",
            name="Laboratory Information Management System",
            tier=ConnectorTier.PRIVATE,
            protocol=ConnectorProtocol.HTTPS,
            base_url=conn,
            auth_config={"type": "database", "description": "PostgreSQL 凭据"},
            rate_limit=0,
            cache_ttl=60,
            version="1.1.0",
            owner="lab-admin",
            tags=["lims", "laboratory", "experiment", "campus"],
            description="实验室信息管理系统 (PostgreSQL 后端 + CDC 审计表)",
            metadata={"cdc_audit_table": cls.CDC_AUDIT_TABLE},
        )
        instance = cls(
            config,
            connection_string=conn,
            default_table="experiments",
            pool_size=3,
        )
        instance.set_mock_data(_LIMS_MOCK_DATA)
        return instance


#: LIMS 模拟数据 (实验记录)
_LIMS_MOCK_DATA: list[dict[str, Any]] = [
    {
        "experiment_id": "EXP-2024-0001",
        "experiment_name": "Dy3+ 掺杂 NaYF4 荧光粉合成与发光性能测试",
        "researcher_id": "R001",
        "sample_id": "SMP-2024-0001",
        "sample_name": "NaYF4:Dy3+ (5mol%)",
        "instrument_id": "INS-FLS-1000",
        "instrument_name": "Edinburgh FLS1000 荧光光谱仪",
        "method_protocol": "PROT-SYN-001",
        "start_time": "2024-09-15T09:00:00Z",
        "end_time": "2024-09-15T17:30:00Z",
        "status": "completed",
        "result_data": {
            "emission_peak_nm": 573,
            "excitation_peak_nm": 350,
            "lifetime_ms": 1.12,
            "cie_x": 0.32,
            "cie_y": 0.35,
        },
        "quality_flag": "pass",
        "remarks": "5mol% Dy3+ 掺杂浓度下黄光发射最强，浓度淬灭未出现。",
        "created_at": "2024-09-15T09:00:00Z",
        "updated_at": "2025-07-10T10:30:00Z",
    },
    {
        "experiment_id": "EXP-2024-0002",
        "experiment_name": "Eu3+ 掺杂 CaF2 红色荧光粉合成",
        "researcher_id": "R002",
        "sample_id": "SMP-2024-0002",
        "sample_name": "CaF2:Eu3+ (3mol%)",
        "instrument_id": "INS-XRD-6000",
        "instrument_name": "Shimadzu XRD-6000 X 射线衍射仪",
        "method_protocol": "PROT-SYN-002",
        "start_time": "2024-10-20T10:00:00Z",
        "end_time": "2024-10-20T16:00:00Z",
        "status": "completed",
        "result_data": {
            "emission_peak_nm": 614,
            "excitation_peak_nm": 393,
            "lifetime_ms": 2.45,
            "phase": "cubic",
            "crystallite_size_nm": 35,
        },
        "quality_flag": "pass",
        "remarks": "XRD 确认立方相 CaF2 结构，晶粒尺寸约 35nm。",
        "created_at": "2024-10-20T10:00:00Z",
        "updated_at": "2025-06-25T14:00:00Z",
    },
    {
        "experiment_id": "EXP-2024-0003",
        "experiment_name": "Dy3+-Tb3+ 共掺杂能量传递研究",
        "researcher_id": "R001",
        "sample_id": "SMP-2024-0003",
        "sample_name": "NaYF4:Dy3+(1%)/Tb3+(3%)",
        "instrument_id": "INS-FLS-1000",
        "instrument_name": "Edinburgh FLS1000 荧光光谱仪",
        "method_protocol": "PROT-SYN-001",
        "start_time": "2024-11-05T08:30:00Z",
        "end_time": "2024-11-05T18:00:00Z",
        "status": "completed",
        "result_data": {
            "emission_peaks_nm": [488, 545, 573],
            "energy_transfer_efficiency": 0.72,
            "decay_analysis": "bi-exponential",
        },
        "quality_flag": "review",
        "remarks": "Dy3+→Tb3+ 能量传递效率 72%，需进一步分析衰减曲线。",
        "created_at": "2024-11-05T08:30:00Z",
        "updated_at": "2025-07-21T14:30:00Z",
    },
    {
        "experiment_id": "EXP-2025-0004",
        "experiment_name": "温度猝灭效应测试 (25-200°C)",
        "researcher_id": "R003",
        "sample_id": "SMP-2024-0001",
        "sample_name": "NaYF4:Dy3+ (5mol%) [复测]",
        "instrument_id": "INS-FLS-1000",
        "instrument_name": "Edinburgh FLS1000 荧光光谱仪",
        "method_protocol": "PROT-MEAS-005",
        "start_time": "2025-01-10T13:00:00Z",
        "end_time": "2025-01-10T20:00:00Z",
        "status": "completed",
        "result_data": {
            "temperature_range_C": [25, 50, 100, 150, 200],
            "relative_intensity": [1.0, 0.95, 0.82, 0.61, 0.38],
            "activation_energy_eV": 0.34,
        },
        "quality_flag": "pass",
        "remarks": "热猝灭活化能 0.34 eV，温度稳定性良好。",
        "created_at": "2025-01-10T13:00:00Z",
        "updated_at": "2025-03-05T11:00:00Z",
    },
]


# ============================================================
# 适配器 3: AcademicAffairsAdapter (教务管理系统)
# ============================================================


class AcademicAffairsAdapter(DatabaseAdapter):
    """教务管理系统 (Academic Affairs System) 适配器.

    封装高校教务管理系统，将课程、选课、学生、成绩、培养方案和
    课表数据统一摄入 L3 知识层，支撑智能教学辅导和学业分析。

    现实场景:
        教务系统 (教务管理系统 / Academic Management System) 是高校
        教学运行的核心系统，管理课程开设、学生选课、成绩录入、
        培养方案执行和排课调度。典型产品包括正方教务、URP、
        清华综合教务等，多采用 MySQL 或 PostgreSQL 作为后端。

    校园上下文:
        - 课程数据是 L3 知识层的重要组成: 课程知识点与教材、文献、
          实验数据形成知识图谱，支撑个性化学习推荐。
        - 选课和成绩数据支撑学业预警、学习路径分析和教学质量评估。
        - 教务系统中的课程与图书馆 OPAC 的 course_reserves 联动，
          与 LIMS 的实验数据关联 (课程实验项目)。

    PII 保护 (Personally Identifiable Information):
        教务系统包含大量学生个人数据 (姓名、学号、身份证号、成绩)，
        属于 PII 敏感数据。本适配器在以下层面实施保护:
        1. Schema 层: students 流的字段标记 PII 敏感级别
        2. 映射层: SchemaMapper 将敏感字段映射到 properties 但标记
           access_level，下游 AccessControlManager 据此实施行/列级控制
        3. 审计层: 所有读取操作通过 AuditTrail 记录，满足合规要求
        4. 查询层: _build_query 支持基于角色的字段投影 (PII 脱敏)

    数据流 (Streams):
        - courses: 课程信息 (课程编号、名称、学分、开课院系)
        - enrollments: 选课记录 (学生-课程关联、学期、状态)
        - students: 学生信息 (学号、姓名、院系、年级) [PII 保护]
        - grades: 成绩记录 (课程成绩、绩点、排名) [PII 保护]
        - curricula: 培养方案 (课程体系、学分要求、先修关系)
        - schedules: 课表安排 (上课时间、地点、教师)

    认证:
        MySQL/PostgreSQL 数据库凭据嵌入 connection_string。
        数据库用户应具有最小权限 (只读)。

    融合方案:
        - SQLAlchemy: 连接池 + 参数化查询 (防 SQL 注入)
        - Airbyte Incremental: 基于 updated_at 的增量同步
        - GDPR/FERPA 合规: PII 字段标记 + 访问控制 + 审计日志
    """

    #: 支持的数据流 (表名)
    STREAMS: list[str] = [
        "courses", "enrollments", "students", "grades", "curricula", "schedules",
    ]

    #: 搜索支持的过滤字段
    SEARCH_FIELDS: list[str] = [
        "course_id", "student_id", "semester", "department", "instructor",
    ]

    #: PII 敏感流 (需要访问控制)
    PII_STREAMS: set[str] = {"students", "grades"}

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        connection_string: str = "",
        default_table: str = "courses",
        pool_size: int = 5,
        **kwargs: Any,
    ) -> None:
        """初始化教务系统适配器.

        Args:
            config: 连接器配置
            connection_string: 数据库连接字符串
            default_table: 默认表名 (courses)
            pool_size: 连接池大小 (教务数据量大，默认 5)
            **kwargs: 传递给 DatabaseAdapter 的额外参数
        """
        super().__init__(
            config,
            connection_string=connection_string,
            default_table=default_table,
            pool_size=pool_size,
            **kwargs,
        )
        self._schema_mapper = self._build_schema_mapper()

    # ---- Schema 映射 ----

    def _build_schema_mapper(self) -> SchemaMapper:
        """构建教务字段到 L3 标准字段的映射器.

        将课程字段映射到 L3 知识实体标准字段:
        - course_id → entity_id (课程唯一标识)
        - course_name → entity_name (课程名称)
        - "course" → entity_type (固定实体类型)
        - course_code → identifiers.course_code (课程代码)
        - credits/instructor/... → properties (属性字典)
        - syllabus_url → source_uri (教学大纲链接)
        """
        mapper = SchemaMapper()
        mapper.add_mapping(FieldMapping(
            source_field="course_id", target_field="entity_id", required=True,
            description="课程唯一标识",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="course_name", target_field="entity_name", required=True,
            transform="trim", description="课程名称",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="course_id", target_field="entity_type",
            default_value="course", description="实体类型固定为 course",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="course_code", target_field="identifiers.course_code",
            description="课程代码 (如 CHEM301)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="department", target_field="properties.department",
            description="开课院系",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="credits", target_field="properties.credits",
            transform="parse_float", description="学分数",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="semester", target_field="properties.semester",
            description="开课学期 (如 2025-Spring)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="instructor_id", target_field="properties.instructor_id",
            description="教师工号",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="instructor_name", target_field="properties.instructor_name",
            description="教师姓名",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="enrollment_count", target_field="properties.enrollment_count",
            transform="parse_int", description="选课人数",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="max_capacity", target_field="properties.max_capacity",
            transform="parse_int", description="课程容量上限",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="schedule_room", target_field="properties.schedule_room",
            description="上课教室",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="schedule_time", target_field="properties.schedule_time",
            description="上课时间",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="prerequisite_courses", target_field="properties.prerequisite_courses",
            transform="split_comma", description="先修课程列表",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="description", target_field="description",
            description="课程描述",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="grade_distribution", target_field="properties.grade_distribution",
            transform="json_parse", description="成绩分布 (JSON)",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="syllabus_url", target_field="source_uri",
            description="教学大纲 URL",
        ))
        return mapper

    # ---- 协议方法重写 ----

    def _do_spec(self) -> AdapterSpec:
        """声明教务系统适配器规范.

        支持 INCREMENTAL 增量同步 (选课变更)，不启用 CDC
        (教务系统变更频率低，增量同步足够)。
        """
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.DISCOVER
            | AdapterCapability.SCHEMA_EVOLUTION
            | AdapterCapability.INCREMENTAL
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.AUTHENTICATE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.DATABASE,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "connection_string": {
                        "type": "string",
                        "description": "MySQL/PostgreSQL 连接字符串",
                    },
                    "default_table": {
                        "type": "string",
                        "default": "courses",
                    },
                    "pool_size": {"type": "integer", "default": 5},
                    "pii_protection": {
                        "type": "boolean",
                        "default": True,
                        "description": "是否启用 PII 保护 (students/grades 流)",
                    },
                },
                "required": ["connection_string"],
            },
            default_sync_mode=SyncMode.INCREMENTAL,
            supported_sync_modes=[SyncMode.FULL_REFRESH, SyncMode.INCREMENTAL],
            version=self.config.version,
            documentation_url="https://jw.university.edu/docs",
            changelog={
                "1.0.0": "初始版本，支持六流 + PII 保护",
                "1.1.0": "增加成绩分布聚合和先修课程图谱",
            },
        )

    def _do_check(self) -> bool:
        """验证教务系统连通性: 检查连接字符串非空."""
        return bool(self._connection_string)

    def _do_discover(self) -> DiscoverResult:
        """发现教务系统 Schema: 返回六个流的表结构定义."""
        streams = [self._get_table_schema(s) for s in self.STREAMS]
        return DiscoverResult(streams=streams, adapter_id=self.config.id)

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        """读取教务数据: 构建 SQL 查询并执行模拟查询.

        对 PII 流 (students/grades) 在元数据中标记 access_level，
        下游可通过 AccessControlManager 实施行/列级访问控制。
        """
        table = stream_name or self._default_table or "courses"
        sql, params = self._build_query(table, sync_mode, checkpoint, limit)
        records = self._mock_query(sql, params)

        # 增量模式: 提取游标值
        cursor_value = ""
        if records and sync_mode == SyncMode.INCREMENTAL:
            cursor_value = str(records[-1].get("updated_at", records[-1].get("course_id", "")))

        checkpoint_result = self._make_checkpoint(
            stream_name=table,
            records_read=len(records),
            cursor_value=cursor_value,
            offset=str(checkpoint.offset if checkpoint else ""),
        )

        # PII 流: 在元数据中标记访问级别
        metadata: dict[str, Any] = {}
        if table in self.PII_STREAMS:
            metadata["access_level"] = "restricted"
            metadata["pii_warning"] = (
                f"流 '{table}' 包含个人身份信息 (PII)，"
                "下游消费方须通过 AccessControlManager 实施访问控制。"
            )
            metadata["ferpa_compliance"] = True

        return ReadResult(
            records=records,
            checkpoint=checkpoint_result,
            has_more=False,
            metadata=metadata,
        )

    # ---- SQL 查询构建 ----

    def _build_query(
        self,
        table: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> tuple[str, dict[str, Any]]:
        """构建教务系统 SQL 查询.

        根据同步模式和表名生成不同的 SQL:
        - FULL_REFRESH: 全量读取
        - INCREMENTAL: 基于 updated_at 游标读取增量变更
          (选课变更、成绩录入)

        对 PII 流 (students/grades) 额外添加 access_level 字段投影，
        供下游访问控制决策使用。

        Args:
            table: 表名
            sync_mode: 同步模式
            checkpoint: 检查点
            limit: 最大记录数

        Returns:
            (sql, params) 二元组
        """
        id_column = "course_id" if table == "courses" else "id"

        # PII 流: 添加 access_level 标记列，供下游访问控制决策使用
        if table in self.PII_STREAMS:
            sql = f"SELECT *, 'restricted' AS access_level FROM {table}"
        else:
            sql = f"SELECT * FROM {table}"
        params: dict[str, Any] = {}

        if sync_mode == SyncMode.INCREMENTAL and checkpoint and checkpoint.cursor_value:
            sql += " WHERE updated_at > :cursor"
            params["cursor"] = checkpoint.cursor_value
            sql += " ORDER BY updated_at ASC"
        else:
            sql += f" ORDER BY {id_column} ASC"

        if limit > 0:
            sql += " LIMIT :limit"
            params["limit"] = limit

        return sql, params

    def _mock_query(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """模拟数据库查询.

        返回课程表模拟数据，支持增量游标过滤。
        对 PII 流自动添加 access_level 字段。
        """
        if self._mock_data is None:
            return []

        # 增量模式: 过滤已读记录
        cursor = params.get("cursor")
        if cursor:
            records = [
                r for r in self._mock_data
                if str(r.get("updated_at", r.get("course_id", ""))) > str(cursor)
            ]
        else:
            records = list(self._mock_data)

        # PII 流: 添加 access_level 标记
        if "access_level" in sql:
            for r in records:
                r["access_level"] = "restricted"

        return records

    # ---- Schema 定义 ----

    def _get_table_schema(self, table_name: str) -> DataSourceSchema:
        """返回指定表的 Schema 定义.

        Args:
            table_name: 表名

        Returns:
            该表的完整 Schema
        """
        schemas = self._build_all_table_schemas()
        if table_name in schemas:
            return schemas[table_name]
        return DataSourceSchema(
            stream_name=table_name,
            fields=[],
            description=f"Unknown table: {table_name}",
            cursor_field="updated_at",
        )

    def _build_all_table_schemas(self) -> dict[str, DataSourceSchema]:
        """构建所有表的 Schema 定义."""
        course_fields = [
            SchemaField(name="course_id", data_type="string", nullable=False,
                        primary_key=True, description="课程唯一标识", max_length=32),
            SchemaField(name="course_name", data_type="string", nullable=False,
                        description="课程名称", max_length=128),
            SchemaField(name="course_code", data_type="string", nullable=False,
                        description="课程代码 (如 CHEM301)", max_length=16),
            SchemaField(name="department", data_type="string", nullable=False,
                        description="开课院系", max_length=64),
            SchemaField(name="credits", data_type="float", nullable=False,
                        description="学分数"),
            SchemaField(name="semester", data_type="string", nullable=False,
                        description="开课学期 (如 2025-Spring)", max_length=16),
            SchemaField(name="instructor_id", data_type="string", nullable=True,
                        description="教师工号", max_length=32),
            SchemaField(name="instructor_name", data_type="string", nullable=True,
                        description="教师姓名", max_length=64),
            SchemaField(name="enrollment_count", data_type="integer", nullable=True,
                        description="选课人数", default_value=0),
            SchemaField(name="max_capacity", data_type="integer", nullable=True,
                        description="课程容量上限", default_value=60),
            SchemaField(name="schedule_room", data_type="string", nullable=True,
                        description="上课教室", max_length=32),
            SchemaField(name="schedule_time", data_type="string", nullable=True,
                        description="上课时间 (如 周一 3-4节)", max_length=64),
            SchemaField(name="prerequisite_courses", data_type="array", nullable=True,
                        description="先修课程列表"),
            SchemaField(name="description", data_type="string", nullable=True,
                        description="课程描述"),
            SchemaField(name="grade_distribution", data_type="object", nullable=True,
                        description="成绩分布 (JSON: {A: 10, B: 20, ...})"),
            SchemaField(name="syllabus_url", data_type="string", nullable=True,
                        description="教学大纲 URL", format="uri"),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="记录更新时间 (增量游标)", format="date-time"),
        ]

        enrollment_fields = [
            SchemaField(name="enrollment_id", data_type="string", nullable=False,
                        primary_key=True, description="选课记录 ID", max_length=32),
            SchemaField(name="student_id", data_type="string", nullable=False,
                        description="学号 [PII]", max_length=32),
            SchemaField(name="course_id", data_type="string", nullable=False,
                        description="课程 ID", max_length=32),
            SchemaField(name="semester", data_type="string", nullable=False,
                        description="学期", max_length=16),
            SchemaField(name="status", data_type="string", nullable=True,
                        description="选课状态",
                        enum_values=["enrolled", "dropped", "completed", "waitlisted"]),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        student_fields = [
            SchemaField(name="student_id", data_type="string", nullable=False,
                        primary_key=True, description="学号 [PII]", max_length=32),
            SchemaField(name="name", data_type="string", nullable=False,
                        description="学生姓名 [PII]", max_length=64),
            SchemaField(name="department", data_type="string", nullable=True,
                        description="所属院系", max_length=64),
            SchemaField(name="grade_level", data_type="integer", nullable=True,
                        description="年级 (1-4)"),
            SchemaField(name="enrollment_year", data_type="integer", nullable=True,
                        description="入学年份"),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        grade_fields = [
            SchemaField(name="grade_id", data_type="string", nullable=False,
                        primary_key=True, description="成绩记录 ID", max_length=32),
            SchemaField(name="student_id", data_type="string", nullable=False,
                        description="学号 [PII]", max_length=32),
            SchemaField(name="course_id", data_type="string", nullable=False,
                        description="课程 ID", max_length=32),
            SchemaField(name="score", data_type="float", nullable=True,
                        description="百分制成绩 [PII]"),
            SchemaField(name="grade_point", data_type="float", nullable=True,
                        description="绩点 [PII]"),
            SchemaField(name="grade_letter", data_type="string", nullable=True,
                        description="等级成绩 (A/B/C/D/F) [PII]", max_length=2),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        curriculum_fields = [
            SchemaField(name="curriculum_id", data_type="string", nullable=False,
                        primary_key=True, description="培养方案 ID", max_length=32),
            SchemaField(name="program_name", data_type="string", nullable=False,
                        description="专业名称", max_length=128),
            SchemaField(name="total_credits", data_type="float", nullable=True,
                        description="总学分要求"),
            SchemaField(name="required_credits", data_type="float", nullable=True,
                        description="必修学分"),
            SchemaField(name="elective_credits", data_type="float", nullable=True,
                        description="选修学分"),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        schedule_fields = [
            SchemaField(name="schedule_id", data_type="string", nullable=False,
                        primary_key=True, description="排课 ID", max_length=32),
            SchemaField(name="course_id", data_type="string", nullable=False,
                        description="课程 ID", max_length=32),
            SchemaField(name="day_of_week", data_type="integer", nullable=True,
                        description="星期几 (1-7)"),
            SchemaField(name="start_period", data_type="integer", nullable=True,
                        description="开始节次"),
            SchemaField(name="end_period", data_type="integer", nullable=True,
                        description="结束节次"),
            SchemaField(name="room", data_type="string", nullable=True,
                        description="教室", max_length=32),
            SchemaField(name="updated_at", data_type="datetime", nullable=False,
                        description="更新时间", format="date-time"),
        ]

        return {
            "courses": DataSourceSchema(
                stream_name="courses",
                fields=course_fields,
                primary_keys=["course_id"],
                cursor_field="updated_at",
                description="课程信息 (课程编号、名称、学分、开课院系)",
                metadata={"engine": "postgresql", "rows_estimated": 3000},
            ),
            "enrollments": DataSourceSchema(
                stream_name="enrollments",
                fields=enrollment_fields,
                primary_keys=["enrollment_id"],
                cursor_field="updated_at",
                description="选课记录 (学生-课程关联、学期、状态) [含PII]",
                metadata={"engine": "postgresql", "rows_estimated": 80000, "pii": True},
            ),
            "students": DataSourceSchema(
                stream_name="students",
                fields=student_fields,
                primary_keys=["student_id"],
                cursor_field="updated_at",
                description="学生信息 (学号、姓名、院系、年级) [PII 保护]",
                metadata={"engine": "postgresql", "rows_estimated": 20000, "pii": True},
            ),
            "grades": DataSourceSchema(
                stream_name="grades",
                fields=grade_fields,
                primary_keys=["grade_id"],
                cursor_field="updated_at",
                description="成绩记录 (课程成绩、绩点) [PII 保护]",
                metadata={"engine": "postgresql", "rows_estimated": 80000, "pii": True},
            ),
            "curricula": DataSourceSchema(
                stream_name="curricula",
                fields=curriculum_fields,
                primary_keys=["curriculum_id"],
                cursor_field="updated_at",
                description="培养方案 (课程体系、学分要求、先修关系)",
                metadata={"engine": "postgresql", "rows_estimated": 50},
            ),
            "schedules": DataSourceSchema(
                stream_name="schedules",
                fields=schedule_fields,
                primary_keys=["schedule_id"],
                cursor_field="updated_at",
                description="课表安排 (上课时间、地点、教师)",
                metadata={"engine": "postgresql", "rows_estimated": 5000},
            ),
        }

    # ---- 工厂方法 ----

    @classmethod
    def create(
        cls,
        connection_string: str = "",
        auth_token: str = "",
    ) -> AcademicAffairsAdapter:
        """创建预配置的教务系统适配器实例.

        Args:
            connection_string: 数据库连接字符串
                               (默认 postgresql://localhost:5432/academic_affairs)
            auth_token: 未使用 (教务系统通过连接字符串认证)

        Returns:
            配置完成的 AcademicAffairsAdapter 实例 (含模拟数据)
        """
        conn = connection_string or "postgresql://localhost:5432/academic_affairs"
        config = ConnectorConfig(
            id="campus-academic-affairs",
            name="University Academic Affairs System",
            tier=ConnectorTier.PRIVATE,
            protocol=ConnectorProtocol.HTTPS,
            base_url=conn,
            auth_config={"type": "database", "description": "数据库只读凭据"},
            rate_limit=0,
            cache_ttl=60,
            version="1.1.0",
            owner="academic-affairs-office",
            tags=["academic", "courses", "grades", "campus", "pii"],
            description="教务管理系统 (课程/选课/学生/成绩/培养方案/课表)",
            metadata={"pii_streams": list(cls.PII_STREAMS)},
        )
        instance = cls(
            config,
            connection_string=conn,
            default_table="courses",
            pool_size=5,
        )
        instance.set_mock_data(_ACADEMIC_MOCK_DATA)
        return instance


#: 教务系统模拟数据 (课程表)
_ACADEMIC_MOCK_DATA: list[dict[str, Any]] = [
    {
        "course_id": "CHEM301-2025S",
        "course_name": "无机材料化学",
        "course_code": "CHEM301",
        "department": "化学与材料工程学院",
        "credits": 3.0,
        "semester": "2025-Spring",
        "instructor_id": "T001",
        "instructor_name": "张明远教授",
        "enrollment_count": 45,
        "max_capacity": 60,
        "schedule_room": "理工楼-A301",
        "schedule_time": "周一 3-4节, 周三 3-4节",
        "prerequisite_courses": ["CHEM101", "CHEM102"],
        "description": "本课程系统讲解无机材料的晶体结构、缺陷化学、相图分析和稀土发光材料等专题，涵盖 Dy3+、Eu3+ 等稀土离子的发光机理。",
        "grade_distribution": {"A": 8, "B": 18, "C": 12, "D": 5, "F": 2},
        "syllabus_url": "https://jw.university.edu/syllabus/CHEM301-2025S.pdf",
        "updated_at": "2025-06-10T08:00:00Z",
    },
    {
        "course_id": "CHEM402-2025S",
        "course_name": "光谱学与光谱分析",
        "course_code": "CHEM402",
        "department": "化学与材料工程学院",
        "credits": 3.0,
        "semester": "2025-Spring",
        "instructor_id": "T002",
        "instructor_name": "李华清副教授",
        "enrollment_count": 32,
        "max_capacity": 40,
        "schedule_room": "理工楼-B205",
        "schedule_time": "周二 1-2节, 周四 1-2节",
        "prerequisite_courses": ["CHEM301", "PHYS201"],
        "description": "讲授分子光谱和原子光谱的基本原理，包括紫外可见、荧光、红外和拉曼光谱技术，以及 Judd-Ofelt 理论在稀土发光中的应用。",
        "grade_distribution": {"A": 10, "B": 14, "C": 6, "D": 2, "F": 0},
        "syllabus_url": "https://jw.university.edu/syllabus/CHEM402-2025S.pdf",
        "updated_at": "2025-06-12T10:30:00Z",
    },
    {
        "course_id": "MATE305-2025S",
        "course_name": "纳米材料制备与表征",
        "course_code": "MATE305",
        "department": "材料科学与工程学院",
        "credits": 2.5,
        "semester": "2025-Spring",
        "instructor_id": "T003",
        "instructor_name": "王小红研究员",
        "enrollment_count": 28,
        "max_capacity": 30,
        "schedule_room": "材料楼-C102",
        "schedule_time": "周五 5-8节",
        "prerequisite_courses": ["MATE201", "CHEM101"],
        "description": "实验课程: 涵盖水热法、共沉淀法、溶胶-凝胶法等纳米材料合成技术，以及 XRD、SEM、TEM、荧光光谱等表征方法。",
        "grade_distribution": {"A": 12, "B": 10, "C": 5, "D": 1, "F": 0},
        "syllabus_url": "https://jw.university.edu/syllabus/MATE305-2025S.pdf",
        "updated_at": "2025-06-15T14:00:00Z",
    },
    {
        "course_id": "PHYS201-2025S",
        "course_name": "量子力学基础",
        "course_code": "PHYS201",
        "department": "物理学院",
        "credits": 4.0,
        "semester": "2025-Spring",
        "instructor_id": "T004",
        "instructor_name": "陈志强教授",
        "enrollment_count": 80,
        "max_capacity": 100,
        "schedule_room": "物理楼-大阶梯教室",
        "schedule_time": "周一 1-2节, 周三 1-2节, 周五 1-2节",
        "prerequisite_courses": ["PHYS101", "MATH201"],
        "description": "量子力学基本原理: 薛定谔方程、氢原子光谱、微扰理论、角动量耦合等，为理解稀土离子能级跃迁奠定理论基础。",
        "grade_distribution": {"A": 15, "B": 30, "C": 25, "D": 8, "F": 2},
        "syllabus_url": "https://jw.university.edu/syllabus/PHYS201-2025S.pdf",
        "updated_at": "2025-06-08T09:00:00Z",
    },
    {
        "course_id": "CHEM499-2025S",
        "course_name": "毕业设计 (稀土发光材料方向)",
        "course_code": "CHEM499",
        "department": "化学与材料工程学院",
        "credits": 6.0,
        "semester": "2025-Spring",
        "instructor_id": "T001",
        "instructor_name": "张明远教授",
        "enrollment_count": 6,
        "max_capacity": 8,
        "schedule_room": "实验楼-D401 (发光材料实验室)",
        "schedule_time": "预约制",
        "prerequisite_courses": ["CHEM301", "CHEM402", "MATE305"],
        "description": "本科毕业设计课题: Dy3+ 掺杂氟化物纳米晶体的合成、发光性质及能量传递机制研究，需完成实验、数据分析和论文撰写。",
        "grade_distribution": {},
        "syllabus_url": "https://jw.university.edu/syllabus/CHEM499-2025S.pdf",
        "updated_at": "2025-07-01T16:00:00Z",
    },
]


# ============================================================
# 适配器 4: InternalDocRepositoryAdapter (内部文档库)
# ============================================================


class InternalDocRepositoryAdapter(FileAdapter):
    """内部文档库 (Internal Document Repository) 适配器.

    封装高校/研究机构的内部文档库，从文件系统或网络挂载共享读取
    文档、演示文稿、报告和学位论文，统一摄入 L3 知识层。

    现实场景:
        内部文档库是机构知识资产的核心存储，通常部署在:
        - 机构知识库 (Institutional Repository, 如 DSpace、Fedora)
        - 网络共享驱动 (NAS / SMB / NFS 挂载)
        - Confluence / SharePoint 协作文档平台导出
        - 本地文件系统归档目录

        文档格式多样: PDF (论文/报告)、DOCX (文档)、Markdown (笔记)、
        HTML (网页存档)、TXT (纯文本)、CSV (结构化数据)。

    校园上下文:
        - 学位论文是本校最重要的知识资产之一，从 OPAC 的题录信息
          到文档库的全文 PDF，形成完整的知识链路。
        - 实验报告、课题组技术文档、会议演示文稿等灰色文献
          (Grey Literature) 通过文档库集中管理。
        - 文档库与 LIMS (实验数据) 和 OPAC (书目) 联动:
          实验数据 → 报告/论文 → 文档库归档 → OPAC 编目

    数据流 (Streams):
        - documents: 通用文档 (PDF/DOCX/HTML/TXT)
        - presentations: 演示文稿 (PPT/PPTX 导出为 PDF)
        - reports: 技术报告 / 项目报告 / 实验报告
        - theses: 学位论文 (本科/硕士/博士)

    分块读取 (Chunked Reading):
        大文件 (如学位论文 PDF 可达 50MB+) 通过 chunk_size 分块读取，
        避免一次性加载到内存。_parse_file 按块处理并拼接内容。

    多格式解析:
        _parse_file 方法根据文件扩展名自动选择解析策略:
        - .json: JSON 解析为结构化记录
        - .csv: CSV 按行解析为记录列表
        - .md: Markdown 解析为标题 + 正文段落
        - .txt / 其他: 纯文本按段落分割

    文件元数据提取:
        每条记录包含文件元数据:
        - file_size: 文件大小 (字节)
        - modified_date: 最后修改时间
        - checksum: SHA-256 校验和 (用于变更检测和去重)

    认证:
        文件系统权限 (POSIX 权限 / ACL)，适配器进程需具有读取权限。

    融合方案:
        - LangChain Document Loader: 多格式统一解析 + 元数据提取
        - LlamaIndex BaseReader: 连接器继承体系 + 分块读取
        - Unstructured.io: 文档结构化解析 (标题/段落/表格)
        - Apache Tika: 格式无关的内容提取 (PDF/DOCX 统一接口)
    """

    #: 支持的数据流
    STREAMS: list[str] = ["documents", "presentations", "reports", "theses"]

    #: 搜索支持的过滤字段
    SEARCH_FIELDS: list[str] = ["filename", "content", "metadata", "author", "date"]

    #: 支持的文件格式
    SUPPORTED_FORMATS: list[str] = ["pdf", "docx", "md", "html", "txt", "csv", "json"]

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        file_path: str = "",
        file_format: str = "json",
        encoding: str = "utf-8",
        chunk_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        """初始化内部文档库适配器.

        Args:
            config: 连接器配置 (base_url 可作为文件路径后备)
            file_path: 文档库根路径 (如 /data/repository)
            file_format: 默认文件格式
            encoding: 文件编码 (默认 utf-8)
            chunk_size: 分块大小 (字节，默认 8192 用于大文件)
            **kwargs: 传递给 FileAdapter 的额外参数
        """
        super().__init__(
            config,
            file_path=file_path,
            file_format=file_format,
            encoding=encoding,
            chunk_size=chunk_size,
            **kwargs,
        )
        self._schema_mapper = self._build_schema_mapper()

    # ---- Schema 映射 ----

    def _build_schema_mapper(self) -> SchemaMapper:
        """构建文档字段到 L3 标准字段的映射器.

        将文档元数据映射到 L3 知识实体标准字段:
        - doc_id → entity_id (文档唯一标识)
        - title → entity_name (文档标题)
        - file_format → entity_type (实体类型: document/report/thesis)
        - file_path/checksum → identifiers (标识符字典)
        - author/tags/content_text → properties (属性字典)
        - file_path → source_uri (文件路径 URI)
        """
        mapper = SchemaMapper()
        mapper.add_mapping(FieldMapping(
            source_field="doc_id", target_field="entity_id", required=True,
            description="文档唯一标识",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="title", target_field="entity_name", required=True,
            transform="trim", description="文档标题",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="file_format", target_field="entity_type",
            default_value="document", description="文档类型",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="file_path", target_field="identifiers.file_path",
            description="文件路径",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="checksum", target_field="identifiers.checksum",
            description="SHA-256 校验和",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="author", target_field="properties.author",
            description="文档作者",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="department", target_field="properties.department",
            description="所属部门",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="tags", target_field="properties.tags",
            transform="split_comma", description="标签列表",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="keywords", target_field="properties.keywords",
            transform="split_comma", description="关键词列表",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="content_text", target_field="properties.content_text",
            description="文档全文内容",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="page_count", target_field="properties.page_count",
            transform="parse_int", description="页数",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="language", target_field="properties.language",
            description="文档语言",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="access_level", target_field="properties.access_level",
            description="访问级别",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="created_date", target_field="properties.created_date",
            transform="iso_datetime", description="创建日期",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="modified_date", target_field="properties.modified_date",
            transform="iso_datetime", description="修改日期",
        ))
        mapper.add_mapping(FieldMapping(
            source_field="file_path", target_field="source_uri",
            description="文件路径 URI",
        ))
        return mapper

    # ---- 协议方法重写 ----

    def _do_spec(self) -> AdapterSpec:
        """声明文档库适配器规范.

        在 File 基类能力基础上增加 STREAM 能力 (流式分块读取)，
        支持大文件逐块处理。
        """
        caps = (
            AdapterCapability.SEARCH
            | AdapterCapability.FETCH
            | AdapterCapability.LIST
            | AdapterCapability.BATCH
            | AdapterCapability.STREAM
            | AdapterCapability.DISCOVER
            | AdapterCapability.HEALTH_CHECK
            | AdapterCapability.CACHEABLE
        )
        return AdapterSpec(
            adapter_type=DataSourceType.FILE,
            capabilities=caps.value,
            config_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文档库根路径 (如 /data/repository)",
                    },
                    "file_format": {
                        "type": "string",
                        "enum": self.SUPPORTED_FORMATS,
                        "default": "json",
                    },
                    "encoding": {"type": "string", "default": "utf-8"},
                    "chunk_size": {
                        "type": "integer",
                        "default": 8192,
                        "description": "大文件分块读取大小 (字节)",
                    },
                },
                "required": ["file_path"],
            },
            default_sync_mode=SyncMode.FULL_REFRESH,
            supported_sync_modes=[SyncMode.FULL_REFRESH],
            version=self.config.version,
            documentation_url="https://docs.university.edu/internal-repo",
            changelog={
                "1.0.0": "初始版本，支持 JSON/CSV/Markdown/TXT 多格式",
                "1.1.0": "增加分块读取和 SHA-256 校验和",
            },
        )

    def _do_check(self) -> bool:
        """验证文档库连通性: 检查文件路径非空.

        在真实实现中会检查 os.path.exists(file_path) 和
        os.access(file_path, os.R_OK)，此处模拟返回路径非空。
        """
        return bool(self._file_path)

    def _do_discover(self) -> DiscoverResult:
        """发现文档库 Schema: 返回四个流的字段定义."""
        streams = [self._get_schema_for_stream(s) for s in self.STREAMS]
        return DiscoverResult(streams=streams, adapter_id=self.config.id)

    def _do_read(
        self,
        *,
        stream_name: str,
        sync_mode: SyncMode,
        checkpoint: SyncCheckpoint | None,
        limit: int,
    ) -> ReadResult:
        """读取文档库数据: 解析模拟内容并支持分块读取.

        对大文件内容，基于 chunk_size 进行分块拼接，
        避免一次性加载到内存。
        """
        content = self._mock_content or ""
        records = self._parse_file(content, self._file_path)

        # 基于流名称过滤文档类型
        if stream_name and stream_name != "documents":
            records = [
                r for r in records
                if r.get("stream_name", "documents") == stream_name
                or r.get("file_format", "") == stream_name.rstrip("s")
            ]

        if limit > 0:
            records = records[:limit]

        return ReadResult(
            records=records,
            checkpoint=self._make_checkpoint(
                stream_name=stream_name or "documents",
                records_read=len(records),
                offset=str(len(records)),
            ),
        )

    # ---- 文件解析 ----

    def _parse_file(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """解析文件内容为结构化记录.

        根据文件扩展名自动选择解析策略:
        - .json: JSON 数组或对象解析为记录列表
        - .csv: CSV 按行解析，首行为表头
        - .md / .markdown: Markdown 解析为标题 + 正文段落
        - .txt / 其他: 纯文本按段落 (双换行) 分割

        每条记录附带文件元数据 (file_size, modified_date, checksum)。

        Args:
            content: 文件文本内容
            file_path: 文件路径 (用于推断格式和提取元数据)

        Returns:
            解析后的记录列表
        """
        if not content:
            return []

        # 推断文件格式
        ext = self._infer_format(file_path)

        try:
            if ext == "json":
                records = self._parse_json(content)
            elif ext == "csv":
                records = self._parse_csv(content)
            elif ext in ("md", "markdown"):
                records = self._parse_markdown(content)
            else:
                records = self._parse_plain_text(content)
        except Exception as e:
            logger.warning("解析文件 %s 失败 (%s): %s，回退到纯文本模式", file_path, ext, e)
            records = self._parse_plain_text(content)

        # 附加文件元数据
        file_meta = self._extract_file_metadata(file_path, content)
        for record in records:
            record.setdefault("file_path", file_path)
            record.setdefault("file_format", ext)
            record.setdefault("file_size", file_meta["file_size"])
            record.setdefault("modified_date", file_meta["modified_date"])
            record.setdefault("checksum", file_meta["checksum"])

        return records

    def _infer_format(self, file_path: str) -> str:
        """从文件路径推断格式.

        Args:
            file_path: 文件路径

        Returns:
            格式标识符 (json/csv/md/txt 等)
        """
        if not file_path:
            return self._file_format or "txt"
        # 提取扩展名
        _, ext = os.path.splitext(file_path)
        ext = ext.lstrip(".").lower()
        if ext in ("json", "csv", "md", "markdown", "html", "htm", "txt"):
            return "md" if ext == "markdown" else ext
        return self._file_format or "txt"

    def _parse_json(self, content: str) -> list[dict[str, Any]]:
        """解析 JSON 内容为记录列表.

        支持两种格式:
        - JSON 数组: [{"doc_id": ...}, ...]
        - JSON 对象: {"documents": [...]} 或 {"doc_id": ...}
        """
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("documents", "records", "items", "data"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []

    def _parse_csv(self, content: str) -> list[dict[str, Any]]:
        """解析 CSV 内容为记录列表.

        首行为表头，后续每行转为 dict。
        值类型自动推断 (int/float/bool/str)。
        """
        reader = csv.DictReader(io.StringIO(content))
        records: list[dict[str, Any]] = []
        for row in reader:
            record: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                record[key] = self._infer_value_type(value)
            records.append(record)
        return records

    def _parse_markdown(self, content: str) -> list[dict[str, Any]]:
        """解析 Markdown 内容为记录列表.

        按 Markdown 标题 (# / ## / ###) 分割为多个段落，
        每个段落转为一条记录:
        - 标题行 → title
        - 正文内容 → content_text
        - 元数据行 (key: value) → 对应字段
        """
        records: list[dict[str, Any]] = []
        # 按标题分割
        sections = re.split(r"^(#{1,6}\s+.+)$", content, flags=re.MULTILINE)

        current_title = "Untitled"
        current_body_lines: list[str] = []
        current_meta: dict[str, Any] = {}

        for part in sections:
            part = part.strip()
            if not part:
                continue
            # 标题行
            if re.match(r"^#{1,6}\s+", part):
                # 保存上一个段落
                if current_body_lines or current_meta:
                    records.append(self._make_markdown_record(
                        current_title, current_body_lines, current_meta,
                    ))
                current_title = re.sub(r"^#{1,6}\s+", "", part)
                current_body_lines = []
                current_meta = {}
            else:
                # 检查是否为元数据行 (key: value)
                meta_match = re.match(r"^(\w+):\s*(.+)$", part)
                if meta_match and "\n" not in part:
                    current_meta[meta_match.group(1)] = meta_match.group(2).strip()
                else:
                    current_body_lines.append(part)

        # 保存最后一个段落
        if current_body_lines or current_meta:
            records.append(self._make_markdown_record(
                current_title, current_body_lines, current_meta,
            ))

        # 如果没有找到任何标题段，将整个内容作为一条记录
        if not records and content.strip():
            records.append({
                "doc_id": "doc-md-001",
                "title": "Untitled Document",
                "content_text": content.strip(),
            })

        return records

    def _make_markdown_record(
        self,
        title: str,
        body_lines: list[str],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """构建 Markdown 段落记录."""
        content_text = "\n".join(body_lines).strip()
        record: dict[str, Any] = {
            "title": title,
            "content_text": content_text,
        }
        # 合并元数据
        record.update(meta)
        # 生成 doc_id
        doc_id = meta.get("doc_id") or f"doc-md-{hashlib.md5(title.encode()).hexdigest()[:8]}"
        record["doc_id"] = doc_id
        return record

    def _parse_plain_text(self, content: str) -> list[dict[str, Any]]:
        """解析纯文本内容为记录列表.

        按双换行 (段落) 分割，每个非空段落转为一条记录。
        """
        paragraphs = re.split(r"\n\s*\n", content.strip())
        records: list[dict[str, Any]] = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            # 尝试从第一行提取标题
            lines = para.split("\n", 1)
            title = lines[0].strip()[:128]
            body = lines[1].strip() if len(lines) > 1 else ""
            records.append({
                "doc_id": f"doc-txt-{i + 1:04d}",
                "title": title,
                "content_text": body or para,
            })
        return records

    @staticmethod
    def _infer_value_type(value: str) -> Any:
        """推断字符串值的类型 (int/float/bool/str)."""
        if value is None:
            return None
        s = value.strip()
        if s == "":
            return ""
        # 整数
        try:
            return int(s)
        except ValueError:
            pass
        # 浮点数
        try:
            return float(s)
        except ValueError:
            pass
        # 布尔值
        if s.lower() in ("true", "yes"):
            return True
        if s.lower() in ("false", "no"):
            return False
        return s

    def _extract_file_metadata(self, file_path: str, content: str) -> dict[str, Any]:
        """提取文件元数据.

        在真实实现中会调用 os.stat() 获取文件大小和修改时间，
        此处基于内容计算模拟元数据。

        Args:
            file_path: 文件路径
            content: 文件内容

        Returns:
            元数据字典 (file_size, modified_date, checksum)
        """
        # 文件大小: 内容字节数 (模拟)
        file_size = len(content.encode(self._encoding)) if content else 0

        # 修改时间: 模拟当前时间 (真实实现用 os.path.getmtime)
        import time as _time
        modified_date = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())

        # SHA-256 校验和
        checksum = hashlib.sha256(content.encode(self._encoding)).hexdigest() if content else ""

        return {
            "file_size": file_size,
            "modified_date": modified_date,
            "checksum": checksum,
        }

    # ---- Schema 定义 ----

    def _get_schema(self) -> DataSourceSchema:
        """返回默认流 (documents) 的 Schema."""
        return self._get_schema_for_stream("documents")

    def _get_schema_for_stream(self, stream_name: str) -> DataSourceSchema:
        """返回指定流的 Schema 定义.

        Args:
            stream_name: 流名称

        Returns:
            该流的完整 Schema
        """
        fields = [
            SchemaField(name="doc_id", data_type="string", nullable=False,
                        primary_key=True, description="文档唯一标识", max_length=64),
            SchemaField(name="file_path", data_type="string", nullable=False,
                        description="文件完整路径", max_length=512),
            SchemaField(name="filename", data_type="string", nullable=False,
                        description="文件名 (不含路径)", max_length=256),
            SchemaField(name="file_format", data_type="string", nullable=False,
                        description="文件格式",
                        enum_values=self.SUPPORTED_FORMATS),
            SchemaField(name="file_size", data_type="integer", nullable=True,
                        description="文件大小 (字节)"),
            SchemaField(name="title", data_type="string", nullable=True,
                        description="文档标题", max_length=512),
            SchemaField(name="author", data_type="string", nullable=True,
                        description="文档作者", max_length=128),
            SchemaField(name="department", data_type="string", nullable=True,
                        description="所属部门", max_length=128),
            SchemaField(name="created_date", data_type="datetime", nullable=True,
                        description="创建日期", format="date-time"),
            SchemaField(name="modified_date", data_type="datetime", nullable=True,
                        description="修改日期", format="date-time"),
            SchemaField(name="tags", data_type="array", nullable=True,
                        description="标签列表"),
            SchemaField(name="keywords", data_type="array", nullable=True,
                        description="关键词列表"),
            SchemaField(name="content_text", data_type="string", nullable=True,
                        description="文档全文内容"),
            SchemaField(name="page_count", data_type="integer", nullable=True,
                        description="页数 (PDF/DOCX)"),
            SchemaField(name="language", data_type="string", nullable=True,
                        description="文档语言", max_length=8),
            SchemaField(name="access_level", data_type="string", nullable=True,
                        description="访问级别",
                        enum_values=["public", "internal", "restricted", "confidential"]),
            SchemaField(name="checksum", data_type="string", nullable=True,
                        description="SHA-256 校验和", max_length=64),
        ]

        descriptions = {
            "documents": "通用文档 (PDF/DOCX/HTML/TXT)",
            "presentations": "演示文稿 (PPT/PPTX 导出)",
            "reports": "技术报告 / 项目报告 / 实验报告",
            "theses": "学位论文 (本科/硕士/博士)",
        }

        return DataSourceSchema(
            stream_name=stream_name,
            fields=fields,
            primary_keys=["doc_id"],
            description=descriptions.get(stream_name, f"Document stream: {stream_name}"),
            metadata={
                "source": "internal_doc_repository",
                "root_path": self._file_path,
                "supported_formats": self.SUPPORTED_FORMATS,
            },
        )

    # ---- 工厂方法 ----

    @classmethod
    def create(
        cls,
        connection_string: str = "",
        auth_token: str = "",
    ) -> InternalDocRepositoryAdapter:
        """创建预配置的内部文档库适配器实例.

        Args:
            connection_string: 用作文件库根路径
                               (默认 /data/repository)
            auth_token: 未使用 (文档库通过文件系统权限认证)

        Returns:
            配置完成的 InternalDocRepositoryAdapter 实例 (含模拟内容)
        """
        path = connection_string or "/data/repository"
        config = ConnectorConfig(
            id="campus-internal-docs",
            name="Internal Document Repository",
            tier=ConnectorTier.PRIVATE,
            protocol=ConnectorProtocol.HTTPS,
            base_url=path,
            auth_config={"type": "filesystem", "description": "POSIX 文件系统权限"},
            rate_limit=0,
            cache_ttl=300,
            version="1.1.0",
            owner="library-it",
            tags=["documents", "repository", "campus", "files"],
            description="内部文档库 (多格式文件解析 + 分块读取)",
            metadata={"supported_formats": cls.SUPPORTED_FORMATS},
        )
        instance = cls(
            config,
            file_path=path,
            file_format="json",
            encoding="utf-8",
            chunk_size=8192,
        )
        instance.set_mock_content(_DOC_REPO_MOCK_CONTENT)
        return instance


#: 内部文档库模拟内容 (JSON 格式，含多类型文档)
_DOC_REPO_MOCK_CONTENT: str = json.dumps([
    {
        "doc_id": "DOC-2024-001",
        "file_path": "/data/repository/theses/2024_Wang_Dy3d_NaYF4.pdf",
        "filename": "2024_Wang_Dy3d_NaYF4.pdf",
        "file_format": "pdf",
        "file_size": 8542336,
        "title": "Dy3+ 掺杂氟化物纳米晶体的发光性质研究",
        "author": "王小红",
        "department": "化学与材料工程学院",
        "created_date": "2024-06-15T10:00:00Z",
        "modified_date": "2024-06-20T14:30:00Z",
        "tags": ["学位论文", "博士", "稀土", "发光材料"],
        "keywords": ["Dy3+", "NaYF4", "荧光", "浓度淬灭", "能量传递"],
        "content_text": "本文系统研究了 Dy3+ 掺杂 NaYF4、CaF2、BaF2 等氟化物纳米晶体的发光性质...",
        "page_count": 156,
        "language": "chi",
        "access_level": "internal",
        "checksum": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234",
    },
    {
        "doc_id": "DOC-2024-002",
        "file_path": "/data/repository/reports/2025_EXP001_LabReport.md",
        "filename": "2025_EXP001_LabReport.md",
        "file_format": "md",
        "file_size": 12544,
        "title": "实验报告: Dy3+ 掺杂 NaYF4 荧光粉合成与发光性能测试",
        "author": "R001 (张明远课题组)",
        "department": "化学与材料工程学院",
        "created_date": "2024-09-15T09:00:00Z",
        "modified_date": "2025-07-10T10:30:00Z",
        "tags": ["实验报告", "荧光粉", "Dy3+"],
        "keywords": ["NaYF4", "Dy3+", "荧光光谱", "Judd-Ofelt"],
        "content_text": "# 实验报告\n\n## 实验目的\n合成 Dy3+ 掺杂 NaYF4 荧光粉并测试其发光性能...\n\n## 实验结果\n发射峰位于 573 nm...",
        "page_count": 8,
        "language": "chi",
        "access_level": "internal",
        "checksum": "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345",
    },
    {
        "doc_id": "DOC-2024-003",
        "file_path": "/data/repository/presentations/2025_GroupMeeting_Luminescence.pptx",
        "filename": "2025_GroupMeeting_Luminescence.pptx",
        "file_format": "pdf",
        "file_size": 5234560,
        "title": "稀土发光材料研究进展 (课题组周会报告)",
        "author": "李华清",
        "department": "化学与材料工程学院",
        "created_date": "2025-03-20T08:00:00Z",
        "modified_date": "2025-03-22T16:00:00Z",
        "tags": ["演示文稿", "周会", "稀土", "发光"],
        "keywords": ["Dy3+", "Eu3+", "Tb3+", "能量传递", "浓度淬灭"],
        "content_text": "稀土发光材料研究进展: Dy3+ (黄光, 573nm), Eu3+ (红光, 614nm), Tb3+ (绿光, 545nm)...",
        "page_count": 32,
        "language": "chi",
        "access_level": "internal",
        "checksum": "c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
    },
    {
        "doc_id": "DOC-2024-004",
        "file_path": "/data/repository/documents/rare_earth_review_2024.pdf",
        "filename": "rare_earth_review_2024.pdf",
        "file_format": "pdf",
        "file_size": 3214567,
        "title": "Rare Earth Doped Luminescent Materials: A Comprehensive Review",
        "author": "Zhang, M.; Li, H.; Wang, X.",
        "department": "化学与材料工程学院",
        "created_date": "2024-12-01T00:00:00Z",
        "modified_date": "2024-12-05T10:00:00Z",
        "tags": ["综述", "稀土", "发光", "review"],
        "keywords": ["rare earth", "luminescence", "Judd-Ofelt", "energy transfer", "nanoparticles"],
        "content_text": "This comprehensive review covers recent advances in rare earth doped luminescent materials...",
        "page_count": 48,
        "language": "eng",
        "access_level": "public",
        "checksum": "d4e5f6789012345678901234567890abcdef1234567890abcdef1234567",
    },
    {
        "doc_id": "DOC-2024-005",
        "file_path": "/data/repository/documents/experiment_data_index.csv",
        "filename": "experiment_data_index.csv",
        "file_format": "csv",
        "file_size": 4521,
        "title": "实验数据索引表",
        "author": "LIMS 自动导出",
        "department": "化学与材料工程学院",
        "created_date": "2025-07-01T00:00:00Z",
        "modified_date": "2025-07-25T12:00:00Z",
        "tags": ["数据索引", "CSV", "实验"],
        "keywords": ["experiment", "index", "LIMS"],
        "content_text": "experiment_id,sample_name,researcher,status\nEXP-2024-0001,NaYF4:Dy3+,R001,completed\nEXP-2024-0002,CaF2:Eu3+,R002,completed",
        "page_count": 1,
        "language": "chi",
        "access_level": "internal",
        "checksum": "e5f6789012345678901234567890abcdef1234567890abcdef12345678",
    },
], ensure_ascii=False, indent=2)


# ============================================================
# 模块辅助函数
# ============================================================


def create_all_private_adapters(
    *,
    opac_base_url: str = "",
    opac_auth_token: str = "",
    lims_connection_string: str = "",
    academic_connection_string: str = "",
    doc_repo_path: str = "",
) -> list[DataAdapterBase]:
    """创建所有 Tier-3 私有数据源适配器实例.

    一次性实例化四个校园/私有适配器，便于批量注册到
    DataAdapterRegistry 或 SyncCoordinator。

    Args:
        opac_base_url: 图书馆 OPAC API 地址
        opac_auth_token: OPAC SSO 认证令牌
        lims_connection_string: LIMS PostgreSQL 连接字符串
        academic_connection_string: 教务系统数据库连接字符串
        doc_repo_path: 内部文档库根路径

    Returns:
        四个适配器实例列表:
        [LibraryOPACAdapter, LIMSAdapter, AcademicAffairsAdapter, InternalDocRepositoryAdapter]
    """
    return [
        LibraryOPACAdapter.create(
            base_url=opac_base_url,
            auth_token=opac_auth_token,
        ),
        LIMSAdapter.create(connection_string=lims_connection_string),
        AcademicAffairsAdapter.create(connection_string=academic_connection_string),
        InternalDocRepositoryAdapter.create(connection_string=doc_repo_path),
    ]


def get_private_adapters_summary() -> dict[str, Any]:
    """获取 Tier-3 私有数据源适配器概要信息.

    返回四个适配器的元信息摘要，用于文档生成、注册中心初始化
    和监控面板展示。

    Returns:
        概要信息字典
    """
    return {
        "tier": ConnectorTier.PRIVATE.value,
        "authority_level": "T3/T4",
        "description": "校园/私有数据源适配器 (内部访问 + 自定义认证 + PII 保护)",
        "adapters": [
            {
                "id": "campus-library-opac",
                "class": "LibraryOPACAdapter",
                "name": "University Library OPAC System",
                "protocol": "REST/HTTPS",
                "base_class": "RESTAdapter",
                "streams": LibraryOPACAdapter.STREAMS,
                "search_fields": LibraryOPACAdapter.SEARCH_FIELDS,
                "sync_modes": ["FULL_REFRESH", "INCREMENTAL"],
                "rate_limit": 60,
                "auth_type": "basic (SSO)",
                "cursor_field": "updated_at",
            },
            {
                "id": "campus-lims",
                "class": "LIMSAdapter",
                "name": "Laboratory Information Management System",
                "protocol": "Database (PostgreSQL)",
                "base_class": "DatabaseAdapter",
                "streams": LIMSAdapter.STREAMS,
                "search_fields": LIMSAdapter.SEARCH_FIELDS,
                "sync_modes": ["FULL_REFRESH", "INCREMENTAL", "CDC",
                               "SNAPSHOT_THEN_INCREMENTAL"],
                "rate_limit": 0,
                "auth_type": "database (connection_string)",
                "cursor_field": "updated_at",
                "cdc_audit_table": LIMSAdapter.CDC_AUDIT_TABLE,
            },
            {
                "id": "campus-academic-affairs",
                "class": "AcademicAffairsAdapter",
                "name": "University Academic Affairs System",
                "protocol": "Database (MySQL/PostgreSQL)",
                "base_class": "DatabaseAdapter",
                "streams": AcademicAffairsAdapter.STREAMS,
                "search_fields": AcademicAffairsAdapter.SEARCH_FIELDS,
                "sync_modes": ["FULL_REFRESH", "INCREMENTAL"],
                "rate_limit": 0,
                "auth_type": "database (connection_string)",
                "cursor_field": "updated_at",
                "pii_streams": list(AcademicAffairsAdapter.PII_STREAMS),
            },
            {
                "id": "campus-internal-docs",
                "class": "InternalDocRepositoryAdapter",
                "name": "Internal Document Repository",
                "protocol": "File (filesystem/NAS)",
                "base_class": "FileAdapter",
                "streams": InternalDocRepositoryAdapter.STREAMS,
                "search_fields": InternalDocRepositoryAdapter.SEARCH_FIELDS,
                "sync_modes": ["FULL_REFRESH"],
                "rate_limit": 0,
                "auth_type": "filesystem (POSIX permissions)",
                "supported_formats": InternalDocRepositoryAdapter.SUPPORTED_FORMATS,
                "chunked_reading": True,
            },
        ],
        "total_adapters": 4,
        "total_streams": (
            len(LibraryOPACAdapter.STREAMS)
            + len(LIMSAdapter.STREAMS)
            + len(AcademicAffairsAdapter.STREAMS)
            + len(InternalDocRepositoryAdapter.STREAMS)
        ),
    }


__all__ = [
    # 适配器类
    "LibraryOPACAdapter",
    "LIMSAdapter",
    "AcademicAffairsAdapter",
    "InternalDocRepositoryAdapter",
    # 辅助函数
    "create_all_private_adapters",
    "get_private_adapters_summary",
]
