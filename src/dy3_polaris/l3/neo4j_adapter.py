"""L3 领域知识层 — Neo4j 图数据库适配器.

融合世界先进方案的图数据库适配设计:
- Neo4j Python Driver: 连接池管理 + Session 复用 + 参数化查询
- Cypher DSL: 流式 API 构建 Cypher 查询 (借鉴 neontology/py2neo)
- Neo4j Graph Database: 节点/关系/属性图模型 (标签 + 类型 + 属性)
- Neo4j MERGE: 幂等写入 (ON CREATE / ON MATCH 语义)
- Neo4j Index: 属性索引 + 全文索引 + 约束管理
- Redis Pipeline: 批量操作 + 事务提交 (借鉴 neo4j UnitOfWork)

核心组件:
1. CypherQueryBuilder — Cypher 查询构建器 (流式 API, 参数化防注入)
2. Neo4jEntityMapper — 实体/三元组 <-> Neo4j 节点/关系 映射器
3. Neo4jAdapter — Neo4j 图数据库适配器 (连接池/同步/查询/子图/统计)

设计原则:
- neo4j 驱动为可选依赖，未安装时给出明确安装提示
- 所有公共方法线程安全 (RLock)
- 查询参数化防止 Cypher 注入
- 上下文管理器支持 (with / async with)

Usage::

    from dy3_polaris.l3.neo4j_adapter import Neo4jAdapter

    adapter = Neo4jAdapter("bolt://localhost:7687", "neo4j", "password")
    adapter.connect()

    # 同步实体到 Neo4j
    result = adapter.sync_entity(entity)

    # 执行 Cypher 查询
    rows = adapter.execute_cypher("MATCH (n:Entity) RETURN n LIMIT 10")

    # 提取子图
    subgraph = adapter.extract_subgraph(["e-abc123"], max_hops=2)

    adapter.close()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from pydantic import BaseModel, Field

from .models import (
    AccessLevel,
    EntityType,
    KnowledgeEntity,
    KnowledgeQualifier,
    KnowledgeSource,
    KnowledgeStatus,
    KnowledgeTriple,
    ProvenanceInfo,
    QualityScore,
    StatementRank,
)

logger = logging.getLogger(__name__)

# 尝试导入 neo4j 驱动 (可选依赖)
try:
    from neo4j import GraphDatabase

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

    # 占位符，避免类型检查报错
    GraphDatabase = None  # type: ignore[assignment]


# ============================================================
# 数据类 — 查询/同步结果
# ============================================================


class SyncResult(BaseModel):
    """单条同步操作的结果.

    Attributes:
        entity_id: 实体或三元组的唯一标识
        status: 操作状态 ("created" / "updated" / "skipped" / "error")
        time_ms: 操作耗时 (毫秒)
        error: 错误信息 (仅 status 为 "error" 时有值)
    """

    entity_id: str = Field(..., description="实体或三元组唯一标识")
    status: str = Field(..., description='操作状态 ("created"/"updated"/"skipped"/"error")')
    time_ms: float = Field(default=0.0, description="操作耗时 (毫秒)")
    error: str = Field(default="", description="错误信息")


class BatchSyncResult(BaseModel):
    """批量同步操作的结果.

    Attributes:
        entities_synced: 成功同步的实体数量
        triples_synced: 成功同步的三元组数量
        errors: 错误列表 (错误信息字符串)
        time_ms: 批量操作总耗时 (毫秒)
    """

    entities_synced: int = Field(default=0, description="成功同步的实体数量")
    triples_synced: int = Field(default=0, description="成功同步的三元组数量")
    errors: list[str] = Field(default_factory=list, description="错误列表")
    time_ms: float = Field(default=0.0, description="批量操作总耗时 (毫秒)")


class SubgraphResult(BaseModel):
    """子图提取结果.

    Attributes:
        entities: 提取到的实体列表
        triples: 提取到的三元组列表 (实体间关系)
        hops: 实际遍历的跳数
    """

    entities: list[KnowledgeEntity] = Field(
        default_factory=list, description="提取到的实体列表"
    )
    triples: list[KnowledgeTriple] = Field(
        default_factory=list, description="提取到的三元组列表"
    )
    hops: int = Field(default=0, description="实际遍历的跳数")


class GraphStats(BaseModel):
    """图数据库统计信息.

    Attributes:
        node_count: 节点总数
        relationship_count: 关系总数
        label_distribution: 标签分布 {标签名: 节点数量}
        relation_type_distribution: 关系类型分布 {类型名: 关系数量}
    """

    node_count: int = Field(default=0, description="节点总数")
    relationship_count: int = Field(default=0, description="关系总数")
    label_distribution: dict[str, int] = Field(
        default_factory=dict, description="标签分布 {标签名: 节点数量}"
    )
    relation_type_distribution: dict[str, int] = Field(
        default_factory=dict, description="关系类型分布 {类型名: 关系数量}"
    )


# ============================================================
# CypherQueryBuilder — Cypher 查询构建器
# ============================================================


class CypherQueryBuilder:
    """Cypher 查询构建器 (借鉴 Neo4j Python Driver API + Cypher DSL).

    提供流式 API 构建 Cypher 查询语句。
    支持: MATCH/WHERE/RETURN/CREATE/MERGE/SET/DELETE/ORDER BY/LIMIT/SKIP。

    所有模式字符串和条件字符串中使用 ``$param`` 占位符,
    通过 ``parameters`` 字典传递实际值，实现参数化查询防止 Cypher 注入。

    Attributes:
        _clauses: 已添加的子句列表 [(子句类型, 子句内容), ...]
        _parameters: 查询参数字典 {参数名: 参数值}

    Usage::

        query = (CypherQueryBuilder()
            .match("(e:Entity {entity_type: $etype})", etype="MATERIAL")
            .where("e.name CONTAINS $name", name="YAG")
            .return_("e.entity_id", "e.name")
            .order_by("e.name")
            .limit(10)
            .build())
        # => MATCH (e:Entity {entity_type: $etype})
        #    WHERE e.name CONTAINS $name
        #    RETURN e.entity_id, e.name
        #    ORDER BY e.name LIMIT 10

        params = query.parameters
        # => {"etype": "MATERIAL", "name": "YAG"}
    """

    def __init__(self) -> None:
        self._clauses: list[tuple[str, str]] = []
        self._parameters: dict[str, Any] = {}

    def match(self, pattern: str, **aliases: Any) -> CypherQueryBuilder:
        """添加 MATCH 子句.

        Args:
            pattern: Cypher 匹配模式, 如 "(e:Entity {entity_type: $etype})"
            **aliases: 查询参数键值对, 如 etype="MATERIAL"

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("MATCH", pattern))
        self._parameters.update(aliases)
        return self

    def where(self, condition: str, **aliases: Any) -> CypherQueryBuilder:
        """添加 WHERE 子句 (可多次调用, 自动用 AND 连接).

        Args:
            condition: 过滤条件, 如 "e.name CONTAINS $name"
            **aliases: 查询参数键值对

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("WHERE", condition))
        self._parameters.update(aliases)
        return self

    def return_(self, *expressions: str) -> CypherQueryBuilder:
        """添加 RETURN 子句.

        Args:
            *expressions: 返回表达式, 如 "e.entity_id", "e.name", "count(*) AS cnt"

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("RETURN", ", ".join(expressions)))
        return self

    def create(self, pattern: str, **aliases: Any) -> CypherQueryBuilder:
        """添加 CREATE 子句.

        Args:
            pattern: 创建模式, 如 "(e:Entity $props)"
            **aliases: 查询参数键值对

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("CREATE", pattern))
        self._parameters.update(aliases)
        return self

    def merge(self, pattern: str, **aliases: Any) -> CypherQueryBuilder:
        """添加 MERGE 子句 (幂等创建/匹配).

        Args:
            pattern: 合并模式, 如 "(e:Entity {entity_id: $eid})"
            **aliases: 查询参数键值对

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("MERGE", pattern))
        self._parameters.update(aliases)
        return self

    def set_(self, *assignments: str) -> CypherQueryBuilder:
        """添加 SET 子句.

        Args:
            *assignments: 赋值表达式, 如 "e.name = $name", "e.updated = true"

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("SET", ", ".join(assignments)))
        return self

    def delete(self, *expressions: str) -> CypherQueryBuilder:
        """添加 DELETE 子句.

        Args:
            *expressions: 待删除的元素, 如 "e", "r"

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("DELETE", ", ".join(expressions)))
        return self

    def order_by(self, *expressions: str) -> CypherQueryBuilder:
        """添加 ORDER BY 子句.

        Args:
            *expressions: 排序表达式, 如 "e.name", "e.created_at DESC"

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("ORDER BY", ", ".join(expressions)))
        return self

    def limit(self, n: int) -> CypherQueryBuilder:
        """添加 LIMIT 子句.

        Args:
            n: 返回结果数量上限

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("LIMIT", str(n)))
        return self

    def skip(self, n: int) -> CypherQueryBuilder:
        """添加 SKIP 子句.

        Args:
            n: 跳过的结果数量

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("SKIP", str(n)))
        return self

    def optional_match(self, pattern: str, **aliases: Any) -> CypherQueryBuilder:
        """添加 OPTIONAL MATCH 子句 (左外连接语义).

        Args:
            pattern: 匹配模式
            **aliases: 查询参数键值对

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("OPTIONAL MATCH", pattern))
        self._parameters.update(aliases)
        return self

    def with_(self, *expressions: str) -> CypherQueryBuilder:
        """添加 WITH 子句 (管道中间结果).

        Args:
            *expressions: 传递表达式, 如 "e", "count(*) AS cnt"

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("WITH", ", ".join(expressions)))
        return self

    def unwind(self, expression: str, as_alias: str) -> CypherQueryBuilder:
        """添加 UNWIND 子句 (列表展开).

        Args:
            expression: 列表表达式, 如 "$items"
            as_alias: 展开后的别名, 如 "item"

        Returns:
            self (支持链式调用)
        """
        self._clauses.append(("UNWIND", f"{expression} AS {as_alias}"))
        return self

    def build(self) -> CypherQuery:
        """构建最终的 Cypher 查询.

        Returns:
            CypherQuery 对象，包含查询字符串和参数字典

        Raises:
            ValueError: 没有添加任何子句时抛出
        """
        if not self._clauses:
            raise ValueError("至少需要添加一个子句才能构建查询")

        parts: list[str] = []
        prev_keyword: str = ""

        for keyword, content in self._clauses:
            # 多个 WHERE 子句用 AND 连接
            if keyword == "WHERE" and prev_keyword == "WHERE":
                parts.append(f"AND {content}")
            else:
                parts.append(f"{keyword} {content}")
            prev_keyword = keyword

        query_str = " ".join(parts)
        return CypherQuery(query=query_str, parameters=dict(self._parameters))


class CypherQuery(BaseModel):
    """构建完成的 Cypher 查询.

    Attributes:
        query: Cypher 查询字符串
        parameters: 查询参数字典 (用于 neo4j driver 的 session.run 参数)
    """

    query: str = Field(..., description="Cypher 查询字符串")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="查询参数字典"
    )


# ============================================================
# Neo4jEntityMapper — 实体/三元组映射器
# ============================================================


class Neo4jEntityMapper:
    """KnowledgeEntity / KnowledgeTriple <-> Neo4j Node / Relationship 映射器.

    负责在内存模型和 Neo4j 图模型之间双向转换:
    - KnowledgeEntity → Neo4j 节点属性 + 标签
    - Neo4j 节点属性 → KnowledgeEntity
    - KnowledgeTriple → Neo4j 关系属性 + 类型
    - Neo4j 关系属性 → KnowledgeTriple

    映射策略:
    - 实体类型 (EntityType) 映射为 Neo4j 标签
    - 实体标识符 (identifiers) 序列化为 JSON 字符串存储在节点属性
    - 嵌套模型 (quality/provenance/source) 序列化为 JSON 字符串
    - 列表字段 (tags/aliases) 序列化为 JSON 字符串
    - 三元组限定符 (qualifiers) 序列化为 JSON 字符串存储在关系属性

    注意:
    - triples 字段不存储到节点属性中 (三元组通过独立关系在 Neo4j 中表达)
    - 反向映射时 triples 字段为空，需通过 extract_subgraph 单独提取
    """

    # ---- 实体 -> 节点属性 ----

    @staticmethod
    def entity_to_node_props(entity: KnowledgeEntity) -> dict[str, Any]:
        """将 KnowledgeEntity 转换为 Neo4j 节点属性字典.

        生成的属性字典不包含标签 (标签通过 entity_type 映射)，
        调用方需自行将 entity_type 设为节点标签。

        Args:
            entity: 知识实体

        Returns:
            Neo4j 节点属性字典, 适合作为 MERGE/CREATE 的属性参数
        """
        props: dict[str, Any] = {
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type.value
            if hasattr(entity.entity_type, "value")
            else str(entity.entity_type),
            "name": entity.name,
            "description": entity.description,
            "identifiers": json.dumps(entity.identifiers, ensure_ascii=False),
            "properties": json.dumps(entity.properties, ensure_ascii=False),
            "domain": entity.domain,
            "access_level": entity.access_level.value
            if hasattr(entity.access_level, "value")
            else str(entity.access_level),
            "version": entity.version,
            "parent_entity_id": entity.parent_entity_id,
            "created_at": entity.created_at,
            "updated_at": entity.updated_at,
            "tags": json.dumps(entity.tags, ensure_ascii=False),
            "aliases": json.dumps(entity.aliases, ensure_ascii=False),
            "language": entity.language,
            "status": entity.status.value
            if hasattr(entity.status, "value")
            else str(entity.status),
            "confidence_score": entity.confidence_score,
            "is_verified": entity.is_verified,
        }

        # 嵌套模型序列化为 JSON
        if entity.source is not None:
            props["source"] = json.dumps(
                entity.source.model_dump(mode="json"), ensure_ascii=False
            )
        if entity.quality is not None:
            props["quality"] = json.dumps(
                entity.quality.model_dump(mode="json"), ensure_ascii=False
            )
        if entity.provenance is not None:
            props["provenance"] = json.dumps(
                entity.provenance.model_dump(mode="json"), ensure_ascii=False
            )

        # 扩展元数据
        if entity.metadata:
            props["metadata"] = json.dumps(entity.metadata, ensure_ascii=False)

        return props

    # ---- 节点属性 -> 实体 ----

    @staticmethod
    def node_props_to_entity(props: dict) -> KnowledgeEntity:
        """将 Neo4j 节点属性字典转换为 KnowledgeEntity.

        Args:
            props: Neo4j 节点属性字典 (来自 session.run 的记录)

        Returns:
            构建完成的知识实体 (triples 字段为空列表)
        """
        # 反序列化 JSON 字段
        identifiers = props.get("identifiers", "{}")
        if isinstance(identifiers, str):
            identifiers = json.loads(identifiers)

        properties = props.get("properties", "{}")
        if isinstance(properties, str):
            properties = json.loads(properties)

        tags = props.get("tags", "[]")
        if isinstance(tags, str):
            tags = json.loads(tags)

        aliases = props.get("aliases", "[]")
        if isinstance(aliases, str):
            aliases = json.loads(aliases)

        # 嵌套模型反序列化
        source = None
        raw_source = props.get("source")
        if raw_source is not None:
            if isinstance(raw_source, str):
                source = KnowledgeSource(**json.loads(raw_source))
            elif isinstance(raw_source, dict):
                source = KnowledgeSource(**raw_source)

        quality = None
        raw_quality = props.get("quality")
        if raw_quality is not None:
            if isinstance(raw_quality, str):
                quality = QualityScore(**json.loads(raw_quality))
            elif isinstance(raw_quality, dict):
                quality = QualityScore(**raw_quality)

        provenance = None
        raw_provenance = props.get("provenance")
        if raw_provenance is not None:
            if isinstance(raw_provenance, str):
                provenance = ProvenanceInfo(**json.loads(raw_provenance))
            elif isinstance(raw_provenance, dict):
                provenance = ProvenanceInfo(**raw_provenance)

        metadata = props.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # 构建实体
        entity = KnowledgeEntity(
            entity_id=props.get("entity_id", ""),
            entity_type=props.get("entity_type", "concept"),
            name=props.get("name", ""),
            description=props.get("description", ""),
            identifiers=identifiers,
            properties=properties,
            domain=props.get("domain", "general"),
            access_level=props.get("access_level", "internal"),
            version=props.get("version", 1),
            parent_entity_id=props.get("parent_entity_id", ""),
            created_at=props.get("created_at", 0.0),
            updated_at=props.get("updated_at", 0.0),
            source=source,
            quality=quality,
            provenance=provenance,
            metadata=metadata,
            tags=tags,
            aliases=aliases,
            language=props.get("language", "zh"),
            status=props.get("status", "active"),
            confidence_score=props.get("confidence_score", 1.0),
            is_verified=props.get("is_verified", False),
        )

        return entity

    # ---- 三元组 -> 关系属性 ----

    @staticmethod
    def triple_to_rel_props(triple: KnowledgeTriple) -> dict[str, Any]:
        """将 KnowledgeTriple 转换为 Neo4j 关系属性字典.

        关系类型由 predicate 字段直接映射。
        主语和宾语通过节点 entity_id 关联。

        Args:
            triple: 知识三元组

        Returns:
            Neo4j 关系属性字典
        """
        props: dict[str, Any] = {
            "triple_id": triple.triple_id,
            "predicate": triple.predicate,
            "object_id": triple.object_id,
            "object_value": json.dumps(triple.object_value, ensure_ascii=False)
            if triple.object_value is not None
            else None,
            "object_is_literal": triple.object_is_literal,
            "qualifiers": json.dumps(
                [q.model_dump(mode="json") for q in triple.qualifiers],
                ensure_ascii=False,
            ),
            "rank": triple.rank.value
            if hasattr(triple.rank, "value")
            else str(triple.rank),
            "confidence": triple.confidence,
            "source_id": triple.source_id,
            "created_at": triple.created_at,
            "valid_from": triple.valid_from,
            "valid_until": triple.valid_until,
        }
        return props

    # ---- 关系属性 -> 三元组 ----

    @staticmethod
    def rel_props_to_triple(
        props: dict, subject_id: str
    ) -> KnowledgeTriple:
        """将 Neo4j 关系属性字典转换为 KnowledgeTriple.

        Args:
            props: Neo4j 关系属性字典
            subject_id: 主语实体 ID (从关系起点节点获取)

        Returns:
            构建完成的知识三元组
        """
        # 反序列化限定符
        qualifiers = props.get("qualifiers", "[]")
        if isinstance(qualifiers, str):
            qualifiers = [KnowledgeQualifier(**q) for q in json.loads(qualifiers)]
        elif isinstance(qualifiers, list):
            qualifiers = [
                KnowledgeQualifier(**q) if isinstance(q, dict) else q
                for q in qualifiers
            ]

        # 反序列化 object_value
        object_value = props.get("object_value")
        if isinstance(object_value, str):
            object_value = json.loads(object_value)

        triple = KnowledgeTriple(
            triple_id=props.get("triple_id", ""),
            subject_id=subject_id,
            predicate=props.get("predicate", ""),
            object_id=props.get("object_id", ""),
            object_value=object_value,
            object_is_literal=props.get("object_is_literal", False),
            qualifiers=qualifiers,
            rank=props.get("rank", "normal"),
            confidence=props.get("confidence", 1.0),
            source_id=props.get("source_id", ""),
            created_at=props.get("created_at", 0.0),
            valid_from=props.get("valid_from", 0.0),
            valid_until=props.get("valid_until", 0.0),
        )

        return triple


# ============================================================
# Neo4jAdapter — Neo4j 图数据库适配器 (核心类)
# ============================================================


class Neo4jAdapter:
    """Neo4j 图数据库适配器 (Neo4j Python Driver 封装).

    提供 KnowledgeStore <-> Neo4j 的双向同步能力。
    设计为可选依赖: 当 neo4j 驱动未安装时, connect() 抛出 ImportError。

    功能:
    1. 实体持久化: 将 KnowledgeEntity 同步到 Neo4j 节点
    2. 关系持久化: 将 KnowledgeTriple 同步到 Neo4j 关系
    3. Cypher 查询: 执行原生 Cypher 查询 (参数化, 防注入)
    4. 子图提取: 从 Neo4j 提取子图返回内存模型
    5. 批量导入: 批量 MERGE 节点和关系
    6. 图谱统计: 节点数/关系数/标签分布
    7. 索引管理: 创建/删除属性索引

    线程安全: 所有公共方法通过 RLock 保护。
    连接管理: 基于 neo4j Python Driver 内置连接池。

    Usage::

        adapter = Neo4jAdapter(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            database="neo4j",
        )
        adapter.connect()

        # 同步单个实体
        result = adapter.sync_entity(entity)

        # 同步整个知识库
        batch_result = adapter.sync_store(knowledge_store)

        # 执行查询
        rows = adapter.execute_cypher(
            "MATCH (e:Entity) WHERE e.entity_type = $etype RETURN e",
            {"etype": "material"}
        )

        # 提取子图
        subgraph = adapter.extract_subgraph(["e-abc123"], max_hops=2)

        # 批量导入
        import_result = adapter.batch_import(entities, triples)

        # 图谱统计
        stats = adapter.get_graph_stats()

        adapter.close()
    """

    def __init__(
        self,
        uri: str,
        user: str = "",
        password: str = "",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
    ) -> None:
        """初始化 Neo4j 适配器.

        Args:
            uri: Neo4j 连接 URI, 如 "bolt://localhost:7687" 或 "neo4j://localhost:7687"
            user: 用户名 (默认为空, 适配无认证场景)
            password: 密码 (默认为空)
            database: 目标数据库名称 (默认 "neo4j")
            max_connection_pool_size: 连接池最大连接数 (默认 50)
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._max_connection_pool_size = max_connection_pool_size

        self._driver = None
        self._lock = threading.RLock()
        self._mapper = Neo4jEntityMapper()

    # --------------------------------------------------------
    # 连接管理
    # --------------------------------------------------------

    def connect(self) -> None:
        """建立 Neo4j 连接.

        使用 neo4j Python Driver 创建连接池。
        如果 neo4j 包未安装, 抛出 ImportError 并给出安装提示。

        Raises:
            ImportError: neo4j 驱动未安装
            Exception: 连接失败 (neo4j 异常透传)
        """
        with self._lock:
            if not _NEO4J_AVAILABLE:
                raise ImportError(
                    "neo4j Python 驱动未安装。请运行: pip install neo4j>=5.0.0"
                )

            if self._driver is not None:
                logger.debug("Neo4j 连接已存在, 跳过重复连接")
                return

            # 构建认证信息
            auth = None
            if self._user and self._password:
                auth = (self._user, self._password)

            self._driver = GraphDatabase.driver(
                self._uri,
                auth=auth,
                max_connection_pool_size=self._max_connection_pool_size,
            )

            # 验证连接
            self._driver.verify_connectivity()
            logger.info(
                "已连接到 Neo4j: %s (database=%s)", self._uri, self._database
            )

    def close(self) -> None:
        """关闭 Neo4j 连接, 释放连接池资源.

        安全关闭: 重复调用不会报错。
        """
        with self._lock:
            if self._driver is not None:
                self._driver.close()
                self._driver = None
                logger.info("已关闭 Neo4j 连接: %s", self._uri)

    def is_connected(self) -> bool:
        """检查是否已连接到 Neo4j.

        Returns:
            True 表示连接已建立且驱动对象存在
        """
        with self._lock:
            return self._driver is not None

    # --------------------------------------------------------
    # 上下文管理器支持
    # --------------------------------------------------------

    def __enter__(self) -> Neo4jAdapter:
        """支持 with 语句, 自动连接."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """支持 with 语句, 自动关闭."""
        self.close()

    # --------------------------------------------------------
    # 实体持久化
    # --------------------------------------------------------

    def sync_entity(self, entity: KnowledgeEntity) -> SyncResult:
        """将单个 KnowledgeEntity 同步到 Neo4j.

        使用 MERGE 语句实现幂等写入:
        - 节点已存在: 更新属性 (ON MATCH SET)
        - 节点不存在: 创建新节点 (ON CREATE SET)

        实体类型映射为 Neo4j 标签。

        Args:
            entity: 知识实体

        Returns:
            SyncResult 包含操作状态和耗时
        """
        if not self.is_connected():
            raise RuntimeError("Neo4j 未连接, 请先调用 connect()")

        start = time.perf_counter()
        entity_id = entity.entity_id

        try:
            props = self._mapper.entity_to_node_props(entity)
            label = entity.entity_type.value if hasattr(
                entity.entity_type, "value"
            ) else str(entity.entity_type)

            # MERGE 实现幂等写入
            cypher = (
                f"MERGE (e:{label} {{entity_id: $entity_id}}) "
                f"ON CREATE SET e += $props "
                f"ON MATCH SET e += $props "
                f"RETURN e.entity_id AS entity_id, "
                f"elementId(e) AS internal_id"
            )

            parameters = {"entity_id": entity_id, "props": props}

            with self._lock:
                with self._driver.session(database=self._database) as session:
                    result = session.run(cypher, parameters)
                    records = list(result)

            elapsed_ms = (time.perf_counter() - start) * 1000

            if records:
                return SyncResult(
                    entity_id=entity_id,
                    status="updated",
                    time_ms=elapsed_ms,
                )
            else:
                # MERGE 在首次创建时也应返回记录
                return SyncResult(
                    entity_id=entity_id,
                    status="created",
                    time_ms=elapsed_ms,
                )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("同步实体失败 [%s]: %s", entity_id, e)
            return SyncResult(
                entity_id=entity_id,
                status="error",
                time_ms=elapsed_ms,
                error=str(e),
            )

    # --------------------------------------------------------
    # 三元组持久化
    # --------------------------------------------------------

    def sync_triple(self, triple: KnowledgeTriple) -> SyncResult:
        """将单个 KnowledgeTriple 同步到 Neo4j 关系.

        使用 MERGE 在主语和宾语之间创建关系:
        - 三元组的 predicate 映射为 Neo4j 关系类型
        - 三元组的限定符、排名、置信度等存储为关系属性
        - 字面值宾语 (object_is_literal=True) 仅创建关系, 不要求宾语节点存在

        Args:
            triple: 知识三元组

        Returns:
            SyncResult 包含操作状态和耗时
        """
        if not self.is_connected():
            raise RuntimeError("Neo4j 未连接, 请先调用 connect()")

        start = time.perf_counter()
        triple_id = triple.triple_id

        try:
            rel_props = self._mapper.triple_to_rel_props(triple)

            if triple.object_is_literal:
                # 字面值宾语: 不要求宾语节点存在
                cypher = (
                    f"MATCH (s {{entity_id: $subject_id}}) "
                    f"MERGE (s)-[r:`{triple.predicate}` {{triple_id: $triple_id}}]->() "
                    f"ON CREATE SET r += $rel_props "
                    f"ON MATCH SET r += $rel_props "
                    f"RETURN r.triple_id AS triple_id"
                )
            else:
                # 实体宾语: 要求宾语节点存在
                cypher = (
                    f"MATCH (s {{entity_id: $subject_id}}) "
                    f"MATCH (o {{entity_id: $object_id}}) "
                    f"MERGE (s)-[r:`{triple.predicate}` {{triple_id: $triple_id}}]->(o) "
                    f"ON CREATE SET r += $rel_props "
                    f"ON MATCH SET r += $rel_props "
                    f"RETURN r.triple_id AS triple_id"
                )

            parameters = {
                "subject_id": triple.subject_id,
                "object_id": triple.object_id,
                "triple_id": triple_id,
                "rel_props": rel_props,
            }

            with self._lock:
                with self._driver.session(database=self._database) as session:
                    result = session.run(cypher, parameters)
                    records = list(result)

            elapsed_ms = (time.perf_counter() - start) * 1000

            return SyncResult(
                entity_id=triple_id,
                status="created" if records else "skipped",
                time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("同步三元组失败 [%s]: %s", triple_id, e)
            return SyncResult(
                entity_id=triple_id,
                status="error",
                time_ms=elapsed_ms,
                error=str(e),
            )

    # --------------------------------------------------------
    # 整体同步
    # --------------------------------------------------------

    def sync_store(
        self,
        store: Any,
        batch_size: int = 500,
    ) -> BatchSyncResult:
        """将 KnowledgeStore 整体同步到 Neo4j.

        分批同步所有实体和三元组, 降低事务压力。
        同步顺序: 先实体后三元组 (确保关系两端节点存在)。

        Args:
            store: KnowledgeStore 实例 (需有 entities/all_entities 和 triples 接口)
            batch_size: 每批处理的数量 (默认 500)

        Returns:
            BatchSyncResult 包含同步统计和错误列表
        """
        if not self.is_connected():
            raise RuntimeError("Neo4j 未连接, 请先调用 connect()")

        start = time.perf_counter()
        errors: list[str] = []
        entities_synced = 0
        triples_synced = 0

        # 收集所有实体
        try:
            all_entities = list(store.entities.values())
        except (AttributeError, TypeError):
            try:
                all_entities = list(store.all_entities())
            except (AttributeError, TypeError):
                errors.append("无法从 store 获取实体列表")
                elapsed_ms = (time.perf_counter() - start) * 1000
                return BatchSyncResult(
                    entities_synced=0,
                    triples_synced=0,
                    errors=errors,
                    time_ms=elapsed_ms,
                )

        # 收集所有三元组
        try:
            all_triples: list[KnowledgeTriple] = []
            for entity in all_entities:
                all_triples.extend(entity.triples)
            # 也尝试从 store 直接获取
            if hasattr(store, "triples"):
                store_triples = list(store.triples.values())
                # 去重: 以 triple_id 为准
                existing_ids = {t.triple_id for t in all_triples}
                for t in store_triples:
                    if t.triple_id not in existing_ids:
                        all_triples.append(t)
        except (AttributeError, TypeError):
            pass

        # 第一阶段: 同步实体 (分批)
        for i in range(0, len(all_entities), batch_size):
            batch = all_entities[i : i + batch_size]
            for entity in batch:
                result = self.sync_entity(entity)
                if result.status == "error":
                    errors.append(f"实体同步失败 [{entity.entity_id}]: {result.error}")
                else:
                    entities_synced += 1

        # 第二阶段: 同步三元组 (分批)
        for i in range(0, len(all_triples), batch_size):
            batch = all_triples[i : i + batch_size]
            for triple in batch:
                result = self.sync_triple(triple)
                if result.status == "error":
                    errors.append(
                        f"三元组同步失败 [{triple.triple_id}]: {result.error}"
                    )
                else:
                    triples_synced += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "同步完成: 实体=%d, 三元组=%d, 错误=%d, 耗时=%.1fms",
            entities_synced,
            triples_synced,
            len(errors),
            elapsed_ms,
        )

        return BatchSyncResult(
            entities_synced=entities_synced,
            triples_synced=triples_synced,
            errors=errors,
            time_ms=elapsed_ms,
        )

    # --------------------------------------------------------
    # Cypher 查询
    # --------------------------------------------------------

    def execute_cypher(
        self,
        query: str,
        parameters: dict | None = None,
    ) -> list[dict]:
        """执行原生 Cypher 查询, 返回字典列表.

        使用参数化查询防止 Cypher 注入。
        所有 Neo4j 记录转换为 Python 字典 (键值对)。

        Args:
            query: Cypher 查询字符串 (使用 $param 占位符)
            parameters: 查询参数字典

        Returns:
            查询结果列表, 每条记录为一个字典

        Raises:
            RuntimeError: 未连接时抛出
            Exception: Neo4j 查询异常透传
        """
        if not self.is_connected():
            raise RuntimeError("Neo4j 未连接, 请先调用 connect()")

        params = parameters or {}

        with self._lock:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, params)
                records = [dict(record) for record in result]

        return records

    # --------------------------------------------------------
    # 子图提取
    # --------------------------------------------------------

    def extract_subgraph(
        self,
        entity_ids: list[str],
        max_hops: int = 2,
    ) -> SubgraphResult:
        """从 Neo4j 提取以指定实体为中心的子图.

        使用可变长度路径模式 ([:R*1..N]) 遍历关系,
        将提取到的节点和关系转换回内存模型。

        Args:
            entity_ids: 起始实体 ID 列表
            max_hops: 最大遍历跳数 (默认 2)

        Returns:
            SubgraphResult 包含提取到的实体和三元组
        """
        if not self.is_connected():
            raise RuntimeError("Neo4j 未连接, 请先调用 connect()")

        if not entity_ids:
            return SubgraphResult(entities=[], triples=[], hops=max_hops)

        try:
            # 提取节点和关系 (单次查询)
            cypher = (
                f"MATCH (e) WHERE e.entity_id IN $entity_ids "
                f"OPTIONAL MATCH path = (e)-[r*1..{max_hops}]-(neighbor) "
                f"RETURN e, r, neighbor, nodes(path) AS path_nodes, "
                f"relationships(path) AS path_rels"
            )

            parameters: dict[str, Any] = {"entity_ids": entity_ids}

            with self._lock:
                with self._driver.session(database=self._database) as session:
                    result = session.run(cypher, parameters)
                    records = list(result)

            # 解析结果: 收集所有唯一节点和关系
            seen_nodes: dict[str, dict] = {}
            seen_rels: dict[str, tuple[str, str, str, dict]] = {}

            for record in records:
                # 提取起始节点
                start_node = record.get("e")
                if start_node is not None:
                    node_dict = dict(start_node)
                    eid = node_dict.get("entity_id", "")
                    if eid and eid not in seen_nodes:
                        seen_nodes[eid] = node_dict

                # 提取路径上的节点和关系
                path_nodes = record.get("path_nodes") or []
                path_rels = record.get("path_rels") or []

                for node in path_nodes:
                    node_dict = dict(node)
                    eid = node_dict.get("entity_id", "")
                    if eid and eid not in seen_nodes:
                        seen_nodes[eid] = node_dict

                for rel in path_rels:
                    rel_dict = dict(rel)
                    triple_id = rel_dict.get("triple_id", "")
                    if triple_id and triple_id not in seen_rels:
                        # 获取关系端点
                        start_node_id = rel_dict.get("entity_id", "")
                        # Neo4j 关系类型
                        predicate = rel_dict.get("predicate", type(rel).__name__)
                        seen_rels[triple_id] = (
                            start_node_id,
                            predicate,
                            rel_dict.get("object_id", ""),
                            rel_dict,
                        )

            # 转换为内存模型
            entities = [
                self._mapper.node_props_to_entity(props)
                for props in seen_nodes.values()
            ]
            triples = [
                self._mapper.rel_props_to_triple(rel_props, subject_id)
                for subject_id, _, _, rel_props in seen_rels.values()
            ]

            return SubgraphResult(
                entities=entities,
                triples=triples,
                hops=max_hops,
            )

        except Exception as e:
            logger.error("提取子图失败: %s", e)
            return SubgraphResult(entities=[], triples=[], hops=max_hops)

    # --------------------------------------------------------
    # 批量导入
    # --------------------------------------------------------

    def batch_import(
        self,
        entities: list[KnowledgeEntity],
        triples: list[KnowledgeTriple],
    ) -> BatchSyncResult:
        """批量导入实体和三元组到 Neo4j.

        使用 UNWIND + MERGE 实现高性能批量导入:
        1. 先批量 MERGE 所有实体节点
        2. 再批量 MERGE 所有关系

        优于逐条 sync_entity/sync_triple 的场景: 大量数据一次性导入。

        Args:
            entities: 实体列表
            triples: 三元组列表

        Returns:
            BatchSyncResult 包含导入统计和错误列表
        """
        if not self.is_connected():
            raise RuntimeError("Neo4j 未连接, 请先调用 connect()")

        start = time.perf_counter()
        errors: list[str] = []
        entities_synced = 0
        triples_synced = 0

        # 第一阶段: 批量 MERGE 实体 (使用 UNWIND)
        if entities:
            try:
                entity_batches: dict[str, list[dict]] = {}
                for entity in entities:
                    label = entity.entity_type.value if hasattr(
                        entity.entity_type, "value"
                    ) else str(entity.entity_type)
                    props = self._mapper.entity_to_node_props(entity)
                    entity_batches.setdefault(label, []).append(props)

                with self._lock:
                    with self._driver.session(database=self._database) as session:
                        for label, batch_props in entity_batches.items():
                            cypher = (
                                f"UNWIND $batch AS props "
                                f"MERGE (e:{label} {{entity_id: props.entity_id}}) "
                                f"ON CREATE SET e += props "
                                f"ON MATCH SET e += props "
                                f"RETURN count(e) AS cnt"
                            )
                            result = session.run(cypher, {"batch": batch_props})
                            for record in result:
                                entities_synced += record["cnt"]

            except Exception as e:
                errors.append(f"批量导入实体失败: {e}")
                logger.error("批量导入实体失败: %s", e)

        # 第二阶段: 批量 MERGE 关系 (使用 UNWIND)
        if triples:
            # 分离字面值三元组和实体三元组
            entity_triples = [t for t in triples if not t.object_is_literal]
            literal_triples = [t for t in triples if t.object_is_literal]

            try:
                with self._lock:
                    with self._driver.session(
                        database=self._database
                    ) as session:
                        # 导入实体间关系 (按 predicate 分组: Cypher 关系类型不能参数化,
                        # 跨谓词合并到单一关系类型会张冠李戴)
                        if entity_triples:
                            entity_grouped: dict[str, list[dict]] = {}
                            for triple in entity_triples:
                                rel_props = self._mapper.triple_to_rel_props(triple)
                                entity_grouped.setdefault(
                                    triple.predicate, []
                                ).append({
                                    "subject_id": triple.subject_id,
                                    "object_id": triple.object_id,
                                    "triple_id": triple.triple_id,
                                    "rel_props": rel_props,
                                })

                            for predicate, rel_data in entity_grouped.items():
                                cypher = (
                                    "UNWIND $batch AS item "
                                    "MATCH (s {entity_id: item.subject_id}) "
                                    "MATCH (o {entity_id: item.object_id}) "
                                    f"MERGE (s)-[r:`{predicate}` "
                                    "{triple_id: item.triple_id}]->(o) "
                                    "ON CREATE SET r += item.rel_props "
                                    "ON MATCH SET r += item.rel_props "
                                    "RETURN count(r) AS cnt"
                                )
                                result = session.run(cypher, {"batch": rel_data})
                                for record in result:
                                    triples_synced += record["cnt"]

                        # 导入字面值关系 (不要求宾语节点, 同样按 predicate 分组)
                        if literal_triples:
                            literal_grouped: dict[str, list[dict]] = {}
                            for triple in literal_triples:
                                rel_props = self._mapper.triple_to_rel_props(triple)
                                literal_grouped.setdefault(
                                    triple.predicate, []
                                ).append({
                                    "subject_id": triple.subject_id,
                                    "triple_id": triple.triple_id,
                                    "rel_props": rel_props,
                                })

                            for predicate, lit_data in literal_grouped.items():
                                cypher = (
                                    "UNWIND $batch AS item "
                                    "MATCH (s {entity_id: item.subject_id}) "
                                    f"MERGE (s)-[r:`{predicate}` "
                                    "{triple_id: item.triple_id}]->() "
                                    "ON CREATE SET r += item.rel_props "
                                    "ON MATCH SET r += item.rel_props "
                                    "RETURN count(r) AS cnt"
                                )
                                result = session.run(cypher, {"batch": lit_data})
                                for record in result:
                                    triples_synced += record["cnt"]

            except Exception as e:
                errors.append(f"批量导入关系失败: {e}")
                logger.error("批量导入关系失败: %s", e)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "批量导入完成: 实体=%d, 三元组=%d, 错误=%d, 耗时=%.1fms",
            entities_synced,
            triples_synced,
            len(errors),
            elapsed_ms,
        )

        return BatchSyncResult(
            entities_synced=entities_synced,
            triples_synced=triples_synced,
            errors=errors,
            time_ms=elapsed_ms,
        )

    # --------------------------------------------------------
    # 图谱统计
    # --------------------------------------------------------

    def get_graph_stats(self) -> GraphStats:
        """获取 Neo4j 图数据库统计信息.

        通过 Cypher 查询统计节点数、关系数、标签分布和关系类型分布。

        Returns:
            GraphStats 包含完整统计信息; 查询失败时返回零值统计
        """
        if not self.is_connected():
            return GraphStats()

        try:
            with self._lock:
                with self._driver.session(database=self._database) as session:
                    # 节点数统计
                    node_result = session.run(
                        "MATCH (n) RETURN count(n) AS cnt"
                    )
                    node_count = node_result.single()["cnt"]

                    # 关系数统计
                    rel_result = session.run(
                        "MATCH ()-[r]->() RETURN count(r) AS cnt"
                    )
                    relationship_count = rel_result.single()["cnt"]

                    # 标签分布
                    label_result = session.run(
                        "MATCH (n) UNWIND labels(n) AS label "
                        "RETURN label, count(n) AS cnt ORDER BY cnt DESC"
                    )
                    label_distribution = {
                        record["label"]: record["cnt"]
                        for record in label_result
                    }

                    # 关系类型分布
                    rel_type_result = session.run(
                        "MATCH ()-[r]->() "
                        "RETURN type(r) AS rel_type, count(r) AS cnt "
                        "ORDER BY cnt DESC"
                    )
                    relation_type_distribution = {
                        record["rel_type"]: record["cnt"]
                        for record in rel_type_result
                    }

            return GraphStats(
                node_count=node_count,
                relationship_count=relationship_count,
                label_distribution=label_distribution,
                relation_type_distribution=relation_type_distribution,
            )

        except Exception as e:
            logger.error("获取图谱统计失败: %s", e)
            return GraphStats()

    # --------------------------------------------------------
    # 索引管理
    # --------------------------------------------------------

    def create_index(self, label: str, property: str) -> None:
        """在指定标签的属性上创建索引.

        使用 Neo4j 5.x 语法: CREATE INDEX index_name FOR (n:Label) ON (n.property)。

        Args:
            label: 节点标签名称, 如 "Entity"
            property: 属性名称, 如 "entity_id"

        Raises:
            RuntimeError: 未连接时抛出
            Exception: Neo4j 索引创建异常透传
        """
        if not self.is_connected():
            raise RuntimeError("Neo4j 未连接, 请先调用 connect()")

        index_name = f"idx_{label.lower()}_{property.lower()}"
        cypher = (
            f"CREATE INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{property})"
        )

        with self._lock:
            with self._driver.session(database=self._database) as session:
                session.run(cypher)

        logger.info("已创建索引: %s (标签=%s, 属性=%s)", index_name, label, property)

    def drop_index(self, label: str, property: str) -> None:
        """删除指定标签属性上的索引.

        Args:
            label: 节点标签名称
            property: 属性名称

        Raises:
            RuntimeError: 未连接时抛出
            Exception: Neo4j 索引删除异常透传
        """
        if not self.is_connected():
            raise RuntimeError("Neo4j 未连接, 请先调用 connect()")

        index_name = f"idx_{label.lower()}_{property.lower()}"
        cypher = f"DROP INDEX {index_name} IF EXISTS"

        with self._lock:
            with self._driver.session(database=self._database) as session:
                session.run(cypher)

        logger.info("已删除索引: %s (标签=%s, 属性=%s)", index_name, label, property)


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 数据类
    "SyncResult",
    "BatchSyncResult",
    "SubgraphResult",
    "GraphStats",
    # 查询构建
    "CypherQueryBuilder",
    "CypherQuery",
    # 映射器
    "Neo4jEntityMapper",
    # 适配器
    "Neo4jAdapter",
]
