"""L3 领域知识层 — 知识存储引擎.

融合世界先进方案的存储引擎设计:
- Neo4j: 节点/关系/属性分离存储 + 原生指针遍历
- LlamaIndex: 多类型 Store 抽象 (KVStore/DocStore/IndexStore)
- Milvus: 向量 + 标量混合存储 + 分区
- Weaviate: 对象 + 向量一体化存储
- GraphRAG: 社区检测 + 层次化知识组织
- ConVer-G: 版本管理 + 变更集追踪
- MACR: 冲突检测 + 解决策略

三层存储架构:
1. EntityStore    — 知识实体存储 (类型/名称/标签/标识符多维索引)
2. TripleStore    — 三元组存储 (主语/谓词/宾语图索引 + 时间有效性)
3. ChunkStore     — 文档切片存储 (全文倒排索引 + 向量索引)
4. KnowledgeStore — 统一知识存储 (编排三层存储 + 版本管理 + 冲突追踪)

所有存储均为内存实现，接口设计支持未来替换为持久化后端。
线程安全：所有写操作通过 RLock 保护。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

from .exceptions import (
    DuplicateEntityError,
    EntityMergeError,
    EntityNotFoundError,
    IngestError,
    VersionConflictError,
)
from .index import HashIndex, InvertedIndex, NameIndex, TypeIndex, VectorIndex
from .models import (
    ChangeRecord,
    ChunkRelationshipType,
    ConflictResolutionStrategy,
    ConflictType,
    DocumentChunk,
    EntityType,
    EvidenceRecord,
    IngestResult,
    KnowledgeBaseStats,
    KnowledgeConflict,
    KnowledgeEntity,
    KnowledgeGraph,
    KnowledgeQuery,
    KnowledgeTriple,
    KnowledgeVersion,
    QueryOperator,
    RelationType,
    RetrievalFilter,
    StatementRank,
    SubgraphConfig,
)

logger = logging.getLogger(__name__)


# ============================================================
# 实体存储 — 知识实体多维索引
# ============================================================


class EntityStore:
    """知识实体存储 (借鉴 Neo4j node store + LlamaIndex KVStore).

    管理知识实体的全生命周期，维护多维度索引以支持快速查询:
    - 类型索引: entity_type -> entity_ids (借鉴 Neo4j 标签索引)
    - 名称索引: name/alias -> entity_ids (借鉴全文索引 + 别名映射)
    - 领域索引: domain -> entity_ids
    - 标签索引: tag -> entity_ids
    - 标识符索引: id_type:id_value -> entity_id (借鉴实体消歧)
    - 状态索引: status -> entity_ids

    支持实体去重 (基于标识符) 和乐观版本控制。

    Attributes:
        _entities: 实体存储 {entity_id: KnowledgeEntity}
        _type_index: 实体类型索引
        _name_index: 名称/别名索引
        _domain_index: 领域索引
        _tag_index: 标签索引
        _identifier_index: 外部标识符索引
        _lock: 线程安全锁
    """

    def __init__(self) -> None:
        self._entities: dict[str, KnowledgeEntity] = {}
        self._type_index: TypeIndex = TypeIndex()
        self._name_index: NameIndex = NameIndex()
        self._domain_index: HashIndex = HashIndex()
        self._tag_index: HashIndex = HashIndex()
        self._identifier_index: HashIndex = HashIndex()
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 索引维护
    # --------------------------------------------------------

    def _index_entity(self, entity: KnowledgeEntity) -> None:
        """为实体建立所有索引 (已持有锁)."""
        # 类型索引
        type_val = entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type)
        self._type_index.add(type_val, entity.entity_id)
        # 名称索引
        self._name_index.add_name(entity.name, entity.entity_id)
        for alias in entity.aliases:
            self._name_index.add_alias(alias, entity.entity_id)
        # 领域索引
        self._domain_index.add(entity.domain, entity.entity_id)
        # 标签索引
        for tag in entity.tags:
            self._tag_index.add(tag, entity.entity_id)
        # 标识符索引
        for id_type, id_value in entity.identifiers.items():
            self._identifier_index.add(f"{id_type}:{id_value}", entity.entity_id)

    def _deindex_entity(self, entity: KnowledgeEntity) -> None:
        """移除实体的所有索引 (已持有锁)."""
        type_val = entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type)
        self._type_index.remove(type_val, entity.entity_id)
        self._name_index.remove_name(entity.name, entity.entity_id)
        for alias in entity.aliases:
            self._name_index.remove_alias(alias, entity.entity_id)
        self._domain_index.remove(entity.domain, entity.entity_id)
        for tag in entity.tags:
            self._tag_index.remove(tag, entity.entity_id)
        for id_type, id_value in entity.identifiers.items():
            self._identifier_index.remove(f"{id_type}:{id_value}", entity.entity_id)

    def _reindex_entity(
        self, entity: KnowledgeEntity, old_snapshot: dict[str, Any]
    ) -> None:
        """用旧快照移除旧索引, 再用当前实体建立新索引 (已持有锁).

        用于版本恢复等场景, 此时 entity 的字段可能已被 setattr 修改,
        需要基于 old_snapshot 移除旧索引.

        Args:
            entity: 当前实体 (已修改)
            old_snapshot: 修改前的快照 (model_dump(mode="json"))
        """
        # 移除旧索引 (基于快照)
        old_type = old_snapshot.get("entity_type", "")
        old_type_val = old_type if isinstance(old_type, str) else str(old_type)
        self._type_index.remove(old_type_val, entity.entity_id)
        old_name = old_snapshot.get("name", "")
        if old_name:
            self._name_index.remove_name(old_name, entity.entity_id)
        for alias in old_snapshot.get("aliases", []):
            self._name_index.remove_alias(alias, entity.entity_id)
        old_domain = old_snapshot.get("domain", "")
        if old_domain:
            self._domain_index.remove(old_domain, entity.entity_id)
        for tag in old_snapshot.get("tags", []):
            self._tag_index.remove(tag, entity.entity_id)
        for id_type, id_value in old_snapshot.get("identifiers", {}).items():
            self._identifier_index.remove(f"{id_type}:{id_value}", entity.entity_id)

        # 建立新索引
        self._index_entity(entity)

    def _find_duplicate(self, entity: KnowledgeEntity) -> str | None:
        """检查是否存在标识符重复的实体 (借鉴 Entity Resolution).

        Returns:
            重复实体的 ID，若无则返回 None
        """
        for id_type, id_value in entity.identifiers.items():
            key = f"{id_type}:{id_value}"
            existing_ids = self._identifier_index.get(key)
            for eid in existing_ids:
                if eid != entity.entity_id:
                    return eid
        return None

    # --------------------------------------------------------
    # CRUD 操作
    # --------------------------------------------------------

    def add_entity(
        self,
        entity: KnowledgeEntity,
        *,
        check_duplicate: bool = True,
    ) -> KnowledgeEntity:
        """添加知识实体.

        Args:
            entity: 要添加的实体
            check_duplicate: 是否检查标识符重复

        Returns:
            存储后的实体

        Raises:
            DuplicateEntityError: 标识符重复且 check_duplicate=True
        """
        with self._lock:
            if entity.entity_id in self._entities:
                raise DuplicateEntityError(
                    entity.entity_id,
                    detail=f"实体已存在: {entity.name}",
                )

            if check_duplicate:
                dup_id = self._find_duplicate(entity)
                if dup_id is not None:
                    raise DuplicateEntityError(
                        entity.entity_id,
                        detail=f"标识符与实体 {dup_id} 重复",
                        context={"duplicate_of": dup_id},
                    )

            self._entities[entity.entity_id] = entity
            self._index_entity(entity)
            logger.debug("添加实体: %s (%s)", entity.entity_id, entity.name)
            return entity

    def get_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """获取实体 (返回引用)."""
        return self._entities.get(entity_id)

    def get_entity_or_raise(self, entity_id: str) -> KnowledgeEntity:
        """获取实体，不存在时抛异常."""
        entity = self._entities.get(entity_id)
        if entity is None:
            raise EntityNotFoundError(entity_id)
        return entity

    def update_entity(
        self,
        entity_id: str,
        *,
        expected_version: int | None = None,
        **updates: Any,
    ) -> KnowledgeEntity:
        """更新实体字段 (乐观版本控制).

        Args:
            entity_id: 实体 ID
            expected_version: 期望的版本号 (乐观锁)，None 表示不检查
            **updates: 要更新的字段

        Returns:
            更新后的实体

        Raises:
            EntityNotFoundError: 实体不存在
            VersionConflictError: 版本冲突
        """
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                raise EntityNotFoundError(entity_id)

            # 乐观版本控制 (借鉴 ConVer-G 乐观并发控制)
            if expected_version is not None and entity.version != expected_version:
                raise VersionConflictError(
                    entity_id=entity_id,
                    expected_version=expected_version,
                    actual_version=entity.version,
                )

            # 先移除旧索引
            self._deindex_entity(entity)

            # 应用更新
            for key, value in updates.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)

            entity.version += 1
            entity.touch()

            # 重建索引
            self._index_entity(entity)
            logger.debug("更新实体: %s (v%d)", entity_id, entity.version)
            return entity

    def remove_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """移除实体并清理索引."""
        with self._lock:
            entity = self._entities.pop(entity_id, None)
            if entity is not None:
                self._deindex_entity(entity)
                logger.debug("移除实体: %s", entity_id)
            return entity

    def exists(self, entity_id: str) -> bool:
        """实体是否存在."""
        return entity_id in self._entities

    # --------------------------------------------------------
    # 查询操作
    # --------------------------------------------------------

    def count(self) -> int:
        """实体总数."""
        return len(self._entities)

    def list_entities(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KnowledgeEntity]:
        """列出实体 (分页)."""
        entities = list(self._entities.values())
        return entities[offset : offset + limit]

    def find_by_type(self, entity_type: EntityType | str) -> list[KnowledgeEntity]:
        """按类型查找实体 (使用类型索引)."""
        type_value = entity_type.value if isinstance(entity_type, EntityType) else entity_type
        ids = self._type_index.get_by_type(type_value)
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def find_by_domain(self, domain: str) -> list[KnowledgeEntity]:
        """按领域查找实体 (使用领域索引)."""
        ids = self._domain_index.get(domain)
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def find_by_name(self, name: str) -> list[KnowledgeEntity]:
        """按名称或别名查找实体 (使用名称索引)."""
        ids = self._name_index.lookup(name)
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def find_by_tag(self, tag: str) -> list[KnowledgeEntity]:
        """按标签查找实体 (使用标签索引)."""
        ids = self._tag_index.get(tag)
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def find_by_identifier(self, id_type: str, id_value: str) -> KnowledgeEntity | None:
        """按外部标识符查找实体 (借鉴实体消歧).

        Args:
            id_type: 标识符类型 (如 "cas", "doi", "isbn")
            id_value: 标识符值

        Returns:
            匹配的实体，无匹配则返回 None
        """
        ids = self._identifier_index.get(f"{id_type}:{id_value}")
        for eid in ids:
            if eid in self._entities:
                return self._entities[eid]
        return None

    def find_by_predicate(self, predicate: str) -> list[KnowledgeEntity]:
        """查找拥有指定谓词三元组的实体."""
        return [
            e for e in self._entities.values()
            if any(t.predicate == predicate for t in e.triples)
        ]

    def search(
        self,
        *,
        entity_type: EntityType | str | None = None,
        domain: str | None = None,
        name: str | None = None,
        tag: str | None = None,
        status: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[KnowledgeEntity]:
        """多条件组合搜索 (AND 关系).

        优先使用索引加速查询，结果在内存中二次过滤。

        Args:
            entity_type: 实体类型过滤
            domain: 领域过滤
            name: 名称/别名过滤
            tag: 标签过滤
            status: 状态过滤
            min_confidence: 最低置信度
            limit: 返回上限

        Returns:
            匹配的实体列表
        """
        with self._lock:
            # 选择最优索引 (选择候选集最小的索引)
            candidates: list[KnowledgeEntity] | None = None

            if entity_type is not None:
                type_value = (
                    entity_type.value
                    if isinstance(entity_type, EntityType)
                    else entity_type
                )
                ids = self._type_index.get_by_type(type_value)
                type_results = [self._entities[eid] for eid in ids if eid in self._entities]
                if candidates is None or len(type_results) < len(candidates):
                    candidates = type_results

            if domain is not None:
                ids = self._domain_index.get(domain)
                domain_results = [self._entities[eid] for eid in ids if eid in self._entities]
                if candidates is None or len(domain_results) < len(candidates):
                    candidates = domain_results

            if name is not None:
                ids = self._name_index.lookup(name)
                name_results = [self._entities[eid] for eid in ids if eid in self._entities]
                if candidates is None or len(name_results) < len(candidates):
                    candidates = name_results

            if tag is not None:
                ids = self._tag_index.get(tag)
                tag_results = [self._entities[eid] for eid in ids if eid in self._entities]
                if candidates is None or len(tag_results) < len(candidates):
                    candidates = tag_results

            if candidates is None:
                candidates = list(self._entities.values())

            # 二次过滤
            results: list[KnowledgeEntity] = []
            for entity in candidates:
                if entity_type is not None:
                    ev = entity_type.value if isinstance(entity_type, EntityType) else entity_type
                    if entity.entity_type.value != ev:
                        continue
                if domain is not None and entity.domain != domain:
                    continue
                if name is not None and not entity.match_name_or_alias(name):
                    continue
                if tag is not None and not entity.has_tag(tag):
                    continue
                if status is not None and entity.status.value != status:
                    continue
                if entity.confidence_score < min_confidence:
                    continue
                results.append(entity)
                if len(results) >= limit:
                    break

            return results

    # --------------------------------------------------------
    # 实体合并 (借鉴 Entity Resolution)
    # --------------------------------------------------------

    def merge_entities(
        self,
        source_id: str,
        target_id: str,
        *,
        strategy: str = "prefer_target",
    ) -> KnowledgeEntity:
        """合并两个实体 (借鉴 Entity Resolution 合并策略).

        将 source 实体的属性、三元组、别名合并到 target 实体中，
        然后移除 source 实体。

        Args:
            source_id: 被合并的实体 ID
            target_id: 合并目标实体 ID
            strategy: 合并策略 ("prefer_target" / "prefer_source" / "merge_all")

        Returns:
            合并后的目标实体

        Raises:
            EntityNotFoundError: 实体不存在
            EntityMergeError: 合并失败
        """
        with self._lock:
            source = self._entities.get(source_id)
            target = self._entities.get(target_id)

            if source is None:
                raise EntityNotFoundError(source_id)
            if target is None:
                raise EntityNotFoundError(target_id)
            if source_id == target_id:
                raise EntityMergeError(
                    source_id, target_id, "不能合并到自身"
                )

            try:
                # 合并别名 (去重)
                merged_aliases = set(target.aliases)
                merged_aliases.update(source.aliases)
                merged_aliases.add(source.name)
                target.aliases = list(merged_aliases - {target.name})

                # 合并标签
                merged_tags = set(target.tags)
                merged_tags.update(source.tags)
                target.tags = list(merged_tags)

                # 合并标识符
                merged_ids = dict(target.identifiers)
                for k, v in source.identifiers.items():
                    if k not in merged_ids:
                        merged_ids[k] = v
                target.identifiers = merged_ids

                # 合并属性
                if strategy == "prefer_source":
                    # source 覆盖 target
                    for k, v in source.properties.items():
                        target.properties[k] = v
                else:
                    # prefer_target / merge_all: target 优先, 补充 source 独有属性
                    for k, v in source.properties.items():
                        if k not in target.properties:
                            target.properties[k] = v

                # 合并三元组
                existing_triple_ids = {t.triple_id for t in target.triples}
                for triple in source.triples:
                    if triple.triple_id not in existing_triple_ids:
                        triple.subject_id = target.entity_id
                        target.triples.append(triple)

                target.version += 1
                target.touch()

                # 移除 source 实体
                self._deindex_entity(source)
                self._entities.pop(source_id, None)

                # 重建 target 索引
                self._deindex_entity(target)
                self._index_entity(target)

                logger.info("合并实体: %s -> %s", source_id, target_id)
                return target

            except Exception as exc:
                raise EntityMergeError(
                    source_id, target_id, str(exc)
                ) from exc

    # --------------------------------------------------------
    # 批量操作
    # --------------------------------------------------------

    def bulk_add(
        self,
        entities: list[KnowledgeEntity],
        *,
        skip_duplicates: bool = True,
    ) -> tuple[int, int, list[str]]:
        """批量添加实体.

        Args:
            entities: 实体列表
            skip_duplicates: 是否跳过重复实体

        Returns:
            (成功数, 跳过数, 成功的 entity_id 列表)
        """
        success = 0
        skipped = 0
        added_ids: list[str] = []

        with self._lock:
            for entity in entities:
                try:
                    if entity.entity_id in self._entities:
                        if skip_duplicates:
                            skipped += 1
                            continue
                        raise DuplicateEntityError(entity.entity_id)

                    if skip_duplicates:
                        dup_id = self._find_duplicate(entity)
                        if dup_id is not None:
                            skipped += 1
                            continue

                    self._entities[entity.entity_id] = entity
                    self._index_entity(entity)
                    success += 1
                    added_ids.append(entity.entity_id)
                except DuplicateEntityError:
                    if skip_duplicates:
                        skipped += 1
                    else:
                        raise

        return success, skipped, added_ids

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取实体存储统计."""
        with self._lock:
            by_type: dict[str, int] = defaultdict(int)
            by_domain: dict[str, int] = defaultdict(int)
            by_status: dict[str, int] = defaultdict(int)

            for entity in self._entities.values():
                by_type[entity.entity_type.value] += 1
                by_domain[entity.domain] += 1
                by_status[entity.status.value] += 1

            quality_scores = [
                e.quality.overall()
                for e in self._entities.values()
                if e.quality is not None
            ]
            avg_quality = (
                sum(quality_scores) / len(quality_scores)
                if quality_scores
                else 0.0
            )

            return {
                "total_entities": len(self._entities),
                "by_type": dict(by_type),
                "by_domain": dict(by_domain),
                "by_status": dict(by_status),
                "avg_quality": round(avg_quality, 4),
                "avg_confidence": round(
                    sum(e.confidence_score for e in self._entities.values())
                    / max(len(self._entities), 1),
                    4,
                ),
            }

    def clear(self) -> None:
        """清空所有实体和索引."""
        with self._lock:
            self._entities.clear()
            self._type_index.clear()
            self._name_index.clear()
            self._domain_index.clear()
            self._tag_index.clear()
            self._identifier_index.clear()


# ============================================================
# 三元组存储 — 图索引 + 时间有效性
# ============================================================


class TripleStore:
    """三元组存储 (借鉴 RDF store + Neo4j relationship store).

    管理知识三元组的存储和图索引，支持高效图遍历。

    维护的索引:
    - 主语索引: subject_id -> triple_ids
    - 宾语索引: object_id -> triple_ids
    - 谓词索引: predicate -> triple_ids
    - 主语+谓词复合索引: subject_id:predicate -> triple_ids
    - 宾语+谓词复合索引: object_id:predicate -> triple_ids

    支持时间有效性查询 (valid_from/valid_until) 和置信度过滤。

    Attributes:
        _triples: 三元组存储 {triple_id: KnowledgeTriple}
        _subject_index: 主语索引
        _object_index: 宾语索引
        _predicate_index: 谓词索引
        _subj_pred_index: 主语+谓词复合索引
        _obj_pred_index: 宾语+谓词复合索引
        _lock: 线程安全锁
    """

    def __init__(self) -> None:
        self._triples: dict[str, KnowledgeTriple] = {}
        self._subject_index: HashIndex = HashIndex()
        self._object_index: HashIndex = HashIndex()
        self._predicate_index: HashIndex = HashIndex()
        self._subj_pred_index: HashIndex = HashIndex()
        self._obj_pred_index: HashIndex = HashIndex()
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 索引维护
    # --------------------------------------------------------

    def _index_triple(self, triple: KnowledgeTriple) -> None:
        """为三元组建立索引 (已持有锁)."""
        tid = triple.triple_id
        self._subject_index.add(triple.subject_id, tid)
        if triple.object_id:
            self._object_index.add(triple.object_id, tid)
        self._predicate_index.add(triple.predicate, tid)
        self._subj_pred_index.add(f"{triple.subject_id}:{triple.predicate}", tid)
        if triple.object_id:
            self._obj_pred_index.add(f"{triple.object_id}:{triple.predicate}", tid)

    def _deindex_triple(self, triple: KnowledgeTriple) -> None:
        """移除三元组索引 (已持有锁)."""
        tid = triple.triple_id
        self._subject_index.remove(triple.subject_id, tid)
        if triple.object_id:
            self._object_index.remove(triple.object_id, tid)
        self._predicate_index.remove(triple.predicate, tid)
        self._subj_pred_index.remove(f"{triple.subject_id}:{triple.predicate}", tid)
        if triple.object_id:
            self._obj_pred_index.remove(f"{triple.object_id}:{triple.predicate}", tid)

    # --------------------------------------------------------
    # CRUD 操作
    # --------------------------------------------------------

    def add_triple(self, triple: KnowledgeTriple) -> KnowledgeTriple:
        """添加三元组."""
        with self._lock:
            if triple.triple_id in self._triples:
                raise DuplicateEntityError(
                    triple.triple_id,
                    detail=f"三元组已存在: {triple.subject_id} -{triple.predicate}-> {triple.object_id}",
                )
            self._triples[triple.triple_id] = triple
            self._index_triple(triple)
            return triple

    def get_triple(self, triple_id: str) -> KnowledgeTriple | None:
        """获取三元组."""
        return self._triples.get(triple_id)

    def remove_triple(self, triple_id: str) -> KnowledgeTriple | None:
        """移除三元组."""
        with self._lock:
            triple = self._triples.pop(triple_id, None)
            if triple is not None:
                self._deindex_triple(triple)
            return triple

    def exists(self, triple_id: str) -> bool:
        """三元组是否存在."""
        return triple_id in self._triples

    def count(self) -> int:
        """三元组总数."""
        return len(self._triples)

    # --------------------------------------------------------
    # 图查询操作
    # --------------------------------------------------------

    def get_by_subject(self, subject_id: str) -> list[KnowledgeTriple]:
        """获取以指定实体为主语的所有三元组 (出边)."""
        ids = self._subject_index.get(subject_id)
        return [self._triples[tid] for tid in ids if tid in self._triples]

    def get_by_object(self, object_id: str) -> list[KnowledgeTriple]:
        """获取以指定实体为宾语的所有三元组 (入边)."""
        ids = self._object_index.get(object_id)
        return [self._triples[tid] for tid in ids if tid in self._triples]

    def get_by_predicate(self, predicate: str) -> list[KnowledgeTriple]:
        """获取指定谓词的所有三元组."""
        ids = self._predicate_index.get(predicate)
        return [self._triples[tid] for tid in ids if tid in self._triples]

    def get_by_subject_predicate(
        self, subject_id: str, predicate: str
    ) -> list[KnowledgeTriple]:
        """获取指定主语和谓词的三元组 (复合索引查询)."""
        ids = self._subj_pred_index.get(f"{subject_id}:{predicate}")
        return [self._triples[tid] for tid in ids if tid in self._triples]

    def get_outgoing(
        self,
        entity_id: str,
        *,
        predicate: str | None = None,
        min_confidence: float = 0.0,
        only_preferred: bool = False,
        exclude_deprecated: bool = True,
        valid_at: float | None = None,
    ) -> list[KnowledgeTriple]:
        """获取实体的出边三元组 (支持多维度过滤).

        Args:
            entity_id: 实体 ID
            predicate: 谓词过滤
            min_confidence: 最低置信度
            only_preferred: 仅返回首选声明
            exclude_deprecated: 排除已弃用声明
            valid_at: 时间有效性过滤 (None=当前时间)

        Returns:
            匹配的三元组列表
        """
        if predicate is not None:
            triples = self.get_by_subject_predicate(entity_id, predicate)
        else:
            triples = self.get_by_subject(entity_id)

        return self._filter_triples(
            triples,
            min_confidence=min_confidence,
            only_preferred=only_preferred,
            exclude_deprecated=exclude_deprecated,
            valid_at=valid_at,
        )

    def get_incoming(
        self,
        entity_id: str,
        *,
        predicate: str | None = None,
        min_confidence: float = 0.0,
        only_preferred: bool = False,
        exclude_deprecated: bool = True,
        valid_at: float | None = None,
    ) -> list[KnowledgeTriple]:
        """获取实体的入边三元组 (支持多维度过滤)."""
        if predicate is not None:
            ids = self._obj_pred_index.get(f"{entity_id}:{predicate}")
            triples = [self._triples[tid] for tid in ids if tid in self._triples]
        else:
            triples = self.get_by_object(entity_id)

        return self._filter_triples(
            triples,
            min_confidence=min_confidence,
            only_preferred=only_preferred,
            exclude_deprecated=exclude_deprecated,
            valid_at=valid_at,
        )

    def get_neighbors(
        self,
        entity_id: str,
        *,
        direction: str = "both",
        min_confidence: float = 0.0,
        exclude_deprecated: bool = True,
        valid_at: float | None = None,
    ) -> list[str]:
        """获取实体的邻居节点 ID (借鉴 Neo4j 邻居查询).

        Args:
            entity_id: 实体 ID
            direction: 方向 ("out"/"in"/"both")
            min_confidence: 最低置信度
            exclude_deprecated: 排除已弃用声明
            valid_at: 时间有效性过滤

        Returns:
            邻居实体 ID 列表 (去重)
        """
        neighbors: set[str] = set()

        if direction in ("out", "both"):
            outgoing = self.get_outgoing(
                entity_id,
                min_confidence=min_confidence,
                exclude_deprecated=exclude_deprecated,
                valid_at=valid_at,
            )
            for t in outgoing:
                if t.object_id:
                    neighbors.add(t.object_id)

        if direction in ("in", "both"):
            incoming = self.get_incoming(
                entity_id,
                min_confidence=min_confidence,
                exclude_deprecated=exclude_deprecated,
                valid_at=valid_at,
            )
            for t in incoming:
                neighbors.add(t.subject_id)

        neighbors.discard(entity_id)
        return list(neighbors)

    def get_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = 5,
        min_confidence: float = 0.0,
    ) -> list[str] | None:
        """查找两个实体之间的最短路径 (BFS).

        Args:
            source_id: 起点实体 ID
            target_id: 终点实体 ID
            max_depth: 最大搜索深度
            min_confidence: 最低置信度

        Returns:
            路径上的实体 ID 列表，若无路径则返回 None
        """
        if source_id == target_id:
            return [source_id]

        visited: set[str] = {source_id}
        queue: list[tuple[str, list[str]]] = [(source_id, [source_id])]

        while queue:
            current, path = queue.pop(0)

            if len(path) - 1 >= max_depth:
                continue

            neighbors = self.get_neighbors(
                current,
                min_confidence=min_confidence,
            )

            for neighbor in neighbors:
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def traverse_bfs(
        self,
        start_id: str,
        *,
        max_depth: int = 2,
        min_confidence: float = 0.0,
        direction: str = "both",
        max_entities: int = 100,
    ) -> tuple[list[str], list[KnowledgeTriple]]:
        """BFS 图遍历 (借鉴 GraphRAG 子图提取).

        Args:
            start_id: 起始实体 ID
            max_depth: 最大遍历深度
            min_confidence: 最低置信度
            direction: 遍历方向
            max_entities: 最大实体数量

        Returns:
            (访问的实体 ID 列表, 遍历的三元组列表)
        """
        visited_entities: set[str] = {start_id}
        visited_triples: list[KnowledgeTriple] = []
        current_level: list[str] = [start_id]

        for depth in range(max_depth):
            if len(visited_entities) >= max_entities:
                break

            next_level: list[str] = []
            for entity_id in current_level:
                outgoing = self.get_outgoing(
                    entity_id,
                    min_confidence=min_confidence,
                )
                incoming = self.get_incoming(
                    entity_id,
                    min_confidence=min_confidence,
                ) if direction in ("in", "both") else []

                all_triples = outgoing + incoming
                for triple in all_triples:
                    if triple not in visited_triples:
                        visited_triples.append(triple)

                    # 收集邻居
                    neighbor = (
                        triple.object_id
                        if triple.subject_id == entity_id and triple.object_id
                        else triple.subject_id
                    )
                    if (
                        neighbor not in visited_entities
                        and len(visited_entities) < max_entities
                    ):
                        visited_entities.add(neighbor)
                        next_level.append(neighbor)

            current_level = next_level
            if not current_level:
                break

        return list(visited_entities), visited_triples

    # --------------------------------------------------------
    # 内部过滤方法
    # --------------------------------------------------------

    @staticmethod
    def _filter_triples(
        triples: list[KnowledgeTriple],
        *,
        min_confidence: float = 0.0,
        only_preferred: bool = False,
        exclude_deprecated: bool = True,
        valid_at: float | None = None,
    ) -> list[KnowledgeTriple]:
        """多维度过滤三元组."""
        results: list[KnowledgeTriple] = []
        for triple in triples:
            if only_preferred and not triple.is_preferred():
                continue
            if exclude_deprecated and triple.is_deprecated():
                continue
            if triple.confidence < min_confidence:
                continue
            if valid_at is not None and not triple.is_valid_at(valid_at):
                continue
            results.append(triple)
        return results

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取三元组存储统计."""
        with self._lock:
            by_predicate: dict[str, int] = defaultdict(int)
            by_rank: dict[str, int] = defaultdict(int)
            confidence_sum = 0.0

            for triple in self._triples.values():
                by_predicate[triple.predicate] += 1
                by_rank[triple.rank.value] += 1
                confidence_sum += triple.confidence

            total = len(self._triples)
            return {
                "total_triples": total,
                "by_predicate": dict(by_predicate),
                "by_rank": dict(by_rank),
                "avg_confidence": round(confidence_sum / total, 4) if total > 0 else 0.0,
                "preferred_count": by_rank.get(StatementRank.PREFERRED.value, 0),
                "deprecated_count": by_rank.get(StatementRank.DEPRECATED.value, 0),
            }

    def clear(self) -> None:
        """清空所有三元组和索引."""
        with self._lock:
            self._triples.clear()
            self._subject_index.clear()
            self._object_index.clear()
            self._predicate_index.clear()
            self._subj_pred_index.clear()
            self._obj_pred_index.clear()


# ============================================================
# 切片存储 — 全文索引 + 向量索引
# ============================================================


class ChunkStore:
    """文档切片存储 (借鉴 LlamaIndex docstore + Pinecone record store).

    管理文档切片的存储和检索，维护:
    - 文档索引: document_id -> chunk_ids
    - 全文倒排索引: BM25 全文检索
    - 向量索引: 向量相似性搜索
    - 模态索引: content_type -> chunk_ids

    支持多模态内容 (text/image/table/equation/code) 和质量过滤。

    Attributes:
        _chunks: 切片存储 {chunk_id: DocumentChunk}
        _doc_index: 文档索引 (document_id -> chunk_ids)
        _modality_index: 模态索引
        _inverted_index: 全文倒排索引
        _vector_index: 向量索引
        _section_index: 章节索引 (document_id:section -> chunk_ids)
        _lock: 线程安全锁
    """

    def __init__(self, vector_dim: int = 0, vector_metric: str = "cosine") -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._doc_index: HashIndex = HashIndex()
        self._modality_index: TypeIndex = TypeIndex()
        self._inverted_index: InvertedIndex = InvertedIndex()
        self._vector_index: VectorIndex = VectorIndex(dim=vector_dim, metric=vector_metric)
        self._section_index: HashIndex = HashIndex()
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # CRUD 操作
    # --------------------------------------------------------

    def add_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """添加文档切片.

        自动将切片内容加入倒排索引，若切片包含向量则加入向量索引。
        """
        with self._lock:
            if chunk.chunk_id in self._chunks:
                raise DuplicateEntityError(
                    chunk.chunk_id,
                    detail=f"切片已存在: document={chunk.document_id}, index={chunk.chunk_index}",
                )

            self._chunks[chunk.chunk_id] = chunk
            self._doc_index.add(chunk.document_id, chunk.chunk_id)
            self._modality_index.add(chunk.content_type.value, chunk.chunk_id)
            if chunk.section:
                self._section_index.add(
                    f"{chunk.document_id}:{chunk.section}", chunk.chunk_id
                )

            # 全文索引
            self._inverted_index.add_document(chunk.chunk_id, chunk.content)

            # 向量索引 (如果切片已有向量)
            if chunk.has_embedding() and chunk.embedding is not None:
                self._vector_index.add(
                    chunk.chunk_id,
                    chunk.embedding.vector,
                    metadata={
                        "document_id": chunk.document_id,
                        "content_type": chunk.content_type.value,
                        "section": chunk.section,
                        "language": chunk.language,
                    },
                )

            return chunk

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """获取切片."""
        return self._chunks.get(chunk_id)

    def get_chunk_or_raise(self, chunk_id: str) -> DocumentChunk:
        """获取切片，不存在时抛异常."""
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            raise EntityNotFoundError(chunk_id)
        return chunk

    def update_chunk(self, chunk_id: str, **updates: Any) -> DocumentChunk:
        """更新切片字段并重建索引."""
        with self._lock:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                raise EntityNotFoundError(chunk_id)

            # 移除旧索引
            self._deindex_chunk(chunk)

            # 应用更新
            for key, value in updates.items():
                if hasattr(chunk, key):
                    setattr(chunk, key, value)

            # 重建索引
            self._index_chunk(chunk)
            return chunk

    def remove_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """移除切片并清理索引."""
        with self._lock:
            chunk = self._chunks.pop(chunk_id, None)
            if chunk is not None:
                self._deindex_chunk(chunk)
            return chunk

    def exists(self, chunk_id: str) -> bool:
        """切片是否存在."""
        return chunk_id in self._chunks

    def count(self) -> int:
        """切片总数."""
        return len(self._chunks)

    def _index_chunk(self, chunk: DocumentChunk) -> None:
        """为切片建立索引 (已持有锁)."""
        self._doc_index.add(chunk.document_id, chunk.chunk_id)
        self._modality_index.add(chunk.content_type.value, chunk.chunk_id)
        if chunk.section:
            self._section_index.add(
                f"{chunk.document_id}:{chunk.section}", chunk.chunk_id
            )
        self._inverted_index.add_document(chunk.chunk_id, chunk.content)
        if chunk.has_embedding() and chunk.embedding is not None:
            self._vector_index.add(
                chunk.chunk_id,
                chunk.embedding.vector,
                metadata={
                    "document_id": chunk.document_id,
                    "content_type": chunk.content_type.value,
                    "section": chunk.section,
                    "language": chunk.language,
                },
            )

    def _deindex_chunk(self, chunk: DocumentChunk) -> None:
        """移除切片索引 (已持有锁)."""
        self._doc_index.remove(chunk.document_id, chunk.chunk_id)
        self._modality_index.remove(chunk.content_type.value, chunk.chunk_id)
        if chunk.section:
            self._section_index.remove(
                f"{chunk.document_id}:{chunk.section}", chunk.chunk_id
            )
        self._inverted_index.remove_document(chunk.chunk_id)
        self._vector_index.remove(chunk.chunk_id)

    # --------------------------------------------------------
    # 查询操作
    # --------------------------------------------------------

    def get_by_document(self, document_id: str) -> list[DocumentChunk]:
        """获取文档的所有切片 (按 chunk_index 排序)."""
        ids = self._doc_index.get(document_id)
        chunks = [self._chunks[cid] for cid in ids if cid in self._chunks]
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def get_by_section(self, document_id: str, section: str) -> list[DocumentChunk]:
        """获取指定章节的切片."""
        ids = self._section_index.get(f"{document_id}:{section}")
        return [self._chunks[cid] for cid in ids if cid in self._chunks]

    def get_by_modality(self, content_type: str) -> list[DocumentChunk]:
        """获取指定模态的切片."""
        ids = self._modality_index.get_by_type(content_type)
        return [self._chunks[cid] for cid in ids if cid in self._chunks]

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 10,
        document_id: str | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """BM25 全文检索.

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            document_id: 限定文档范围 (None=全局搜索)

        Returns:
            [(chunk, score)] 按分数降序排列
        """
        results = self._inverted_index.search(query, top_k=top_k * 3 if document_id else top_k)

        chunk_results: list[tuple[DocumentChunk, float]] = []
        for chunk_id, score in results:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            if document_id is not None and chunk.document_id != document_id:
                continue
            chunk_results.append((chunk, score))
            if len(chunk_results) >= top_k:
                break

        return chunk_results

    def search_vector(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filter_fn: Any = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """向量相似性搜索.

        Args:
            query_vector: 查询向量
            top_k: 返回前 k 个结果
            filter_fn: 预过滤函数 (metadata -> bool)

        Returns:
            [(chunk, score)] 按分数降序排列
        """
        results = self._vector_index.search(query_vector, top_k=top_k, filter_fn=filter_fn)

        chunk_results: list[tuple[DocumentChunk, float]] = []
        for chunk_id, score in results:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            chunk_results.append((chunk, score))

        return chunk_results

    # --------------------------------------------------------
    # 向量管理
    # --------------------------------------------------------

    def add_embedding(self, chunk_id: str, vector: list[float], model: str = "default") -> None:
        """为切片添加向量 (延迟向量化场景).

        Args:
            chunk_id: 切片 ID
            vector: 密集向量
            model: 编码模型名称

        Raises:
            EntityNotFoundError: 切片不存在
        """
        from .models import EmbeddingVector

        with self._lock:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                raise EntityNotFoundError(chunk_id)

            embedding = EmbeddingVector(
                content_id=chunk_id,
                vector=vector,
                model=model,
            )
            chunk.embedding = embedding

            # 添加到向量索引
            self._vector_index.add(
                chunk_id,
                vector,
                metadata={
                    "document_id": chunk.document_id,
                    "content_type": chunk.content_type.value,
                    "section": chunk.section,
                    "language": chunk.language,
                },
            )

    def remove_embedding(self, chunk_id: str) -> bool:
        """移除切片的向量."""
        with self._lock:
            chunk = self._chunks.get(chunk_id)
            if chunk is None or chunk.embedding is None:
                return False
            chunk.embedding = None
            self._vector_index.remove(chunk_id)
            return True

    def has_embedding(self, chunk_id: str) -> bool:
        """切片是否已向量化."""
        chunk = self._chunks.get(chunk_id)
        return chunk is not None and chunk.has_embedding()

    def indexed_vector_count(self) -> int:
        """已索引向量数."""
        return self._vector_index.size()

    # --------------------------------------------------------
    # 文档管理
    # --------------------------------------------------------

    def get_document_ids(self) -> list[str]:
        """获取所有文档 ID."""
        return self._doc_index.keys()

    def document_chunk_count(self, document_id: str) -> int:
        """文档的切片数量."""
        return self._doc_index.size() if document_id in self._doc_index.keys() else 0

    def remove_document(self, document_id: str) -> int:
        """移除文档的所有切片.

        Returns:
            移除的切片数量
        """
        with self._lock:
            chunk_ids = self._doc_index.get(document_id)
            count = 0
            for chunk_id in chunk_ids:
                chunk = self._chunks.pop(chunk_id, None)
                if chunk is not None:
                    self._deindex_chunk(chunk)
                    count += 1
            return count

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取切片存储统计."""
        with self._lock:
            by_modality: dict[str, int] = defaultdict(int)
            by_document: dict[str, int] = defaultdict(int)
            total_chars = 0
            total_tokens = 0
            embedded_count = 0

            for chunk in self._chunks.values():
                by_modality[chunk.content_type.value] += 1
                by_document[chunk.document_id] += 1
                total_chars += chunk.char_count
                total_tokens += chunk.token_count
                if chunk.has_embedding():
                    embedded_count += 1

            total = len(self._chunks)
            return {
                "total_chunks": total,
                "total_documents": len(by_document),
                "by_modality": dict(by_modality),
                "indexed_vectors": self._vector_index.size(),
                "vocabulary_size": self._inverted_index.vocabulary_size(),
                "avg_chunk_chars": round(total_chars / total, 1) if total > 0 else 0,
                "avg_chunk_tokens": round(total_tokens / total, 1) if total > 0 else 0,
                "embedded_ratio": round(embedded_count / total, 4) if total > 0 else 0.0,
            }

    def clear(self) -> None:
        """清空所有切片和索引."""
        with self._lock:
            self._chunks.clear()
            self._doc_index.clear()
            self._modality_index.clear()
            self._inverted_index.clear()
            self._vector_index.clear()
            self._section_index.clear()


# ============================================================
# 统一知识存储 — 编排三层存储 + 版本 + 冲突
# ============================================================


class KnowledgeStore:
    """统一知识存储 (借鉴 Neo4j graph store + LlamaIndex StorageContext).

    编排 EntityStore、TripleStore、ChunkStore 三层存储，
    提供版本管理、冲突追踪、子图提取和统一查询接口。

    设计原则:
    - 单一入口: 所有知识操作通过 KnowledgeStore
    - 索引委托: 各 Store 自维护索引，KnowledgeStore 不重复索引
    - 版本追踪: 实体更新自动记录版本快照 (借鉴 ConVer-G)
    - 冲突管理: 知识冲突检测与解决 (借鉴 MACR)
    - 证据管理: 知识证据追踪 (借鉴 ProVe)

    Attributes:
        entity_store: 实体存储
        triple_store: 三元组存储
        chunk_store: 切片存储
        _versions: 版本历史 {entity_id: [KnowledgeVersion]}
        _conflicts: 冲突记录 {conflict_id: KnowledgeConflict}
        _evidence: 证据记录 {evidence_id: EvidenceRecord}
        _lock: 线程安全锁
    """

    def __init__(
        self,
        *,
        vector_dim: int = 0,
        vector_metric: str = "cosine",
    ) -> None:
        self.entity_store: EntityStore = EntityStore()
        self.triple_store: TripleStore = TripleStore()
        self.chunk_store: ChunkStore = ChunkStore(
            vector_dim=vector_dim, vector_metric=vector_metric
        )
        self._versions: dict[str, list[KnowledgeVersion]] = defaultdict(list)
        self._conflicts: dict[str, KnowledgeConflict] = {}
        self._evidence: dict[str, EvidenceRecord] = {}
        self._lock = threading.RLock()

    # ================================================================
    # 实体操作 (委托 EntityStore + 版本追踪)
    # ================================================================

    def add_entity(
        self,
        entity: KnowledgeEntity,
        *,
        check_duplicate: bool = True,
    ) -> KnowledgeEntity:
        """添加实体并创建初始版本."""
        entity = self.entity_store.add_entity(entity, check_duplicate=check_duplicate)

        # 创建初始版本
        version = KnowledgeVersion(
            entity_id=entity.entity_id,
            revision_number=1,
            changeset=[
                ChangeRecord(
                    change_type="add",
                    entity_id=entity.entity_id,
                    field_path="*",
                    new_value=entity.model_dump(),
                    reason="初始创建",
                )
            ],
            snapshot=entity.model_dump(mode="json"),
            version_note="初始版本",
        )
        with self._lock:
            self._versions[entity.entity_id].append(version)

        return entity

    def get_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """获取实体."""
        return self.entity_store.get_entity(entity_id)

    def get_entity_or_raise(self, entity_id: str) -> KnowledgeEntity:
        """获取实体，不存在时抛异常."""
        return self.entity_store.get_entity_or_raise(entity_id)

    def update_entity(
        self,
        entity_id: str,
        *,
        expected_version: int | None = None,
        track_version: bool = True,
        changed_by: str = "system",
        reason: str = "",
        **updates: Any,
    ) -> KnowledgeEntity:
        """更新实体并记录版本变更.

        Args:
            entity_id: 实体 ID
            expected_version: 期望版本号 (乐观锁)
            track_version: 是否记录版本
            changed_by: 变更者
            reason: 变更原因
            **updates: 更新字段

        Returns:
            更新后的实体
        """
        # 记录旧快照 (用于变更集)
        old_entity = self.entity_store.get_entity_or_raise(entity_id)
        old_snapshot = old_entity.model_dump(mode="json")

        # 执行更新
        updated = self.entity_store.update_entity(
            entity_id, expected_version=expected_version, **updates
        )

        # 记录版本
        if track_version:
            changeset: list[ChangeRecord] = []
            new_snapshot = updated.model_dump(mode="json")

            # 检测变更字段 (借鉴 DBpedia-TKG 三元组级变更追踪)
            for field in updates:
                old_val = old_snapshot.get(field)
                new_val = new_snapshot.get(field)
                if old_val != new_val:
                    changeset.append(
                        ChangeRecord(
                            change_type="modify",
                            entity_id=entity_id,
                            field_path=field,
                            old_value=old_val,
                            new_value=new_val,
                            changed_by=changed_by,
                            reason=reason,
                        )
                    )

            if not changeset:
                changeset.append(
                    ChangeRecord(
                        change_type="modify",
                        entity_id=entity_id,
                        field_path="version",
                        old_value=old_snapshot.get("version"),
                        new_value=updated.version,
                        changed_by=changed_by,
                        reason=reason or "版本递增",
                    )
                )

            version = KnowledgeVersion(
                entity_id=entity_id,
                revision_number=updated.version,
                parent_version_id=self._get_latest_version_id(entity_id),
                changeset=changeset,
                snapshot=new_snapshot,
                version_note=reason or f"更新字段: {', '.join(updates.keys())}",
                created_by=changed_by,
            )

            with self._lock:
                # 标记前一版本为非当前
                versions = self._versions.get(entity_id, [])
                if versions and versions[-1].is_current():
                    versions[-1].valid_until = time.time()
                self._versions[entity_id].append(version)

        return updated

    def remove_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """移除实体 (同时清理关联三元组)."""
        with self._lock:
            # 清理以该实体为主语或宾语的三元组
            outgoing = self.triple_store.get_by_subject(entity_id)
            incoming = self.triple_store.get_by_object(entity_id)
            for triple in outgoing + incoming:
                self.triple_store.remove_triple(triple.triple_id)

            # 清理版本记录
            self._versions.pop(entity_id, None)

            # 清理关联冲突
            conflict_ids = [
                cid for cid, c in self._conflicts.items()
                if c.entity_id == entity_id
            ]
            for cid in conflict_ids:
                del self._conflicts[cid]

            return self.entity_store.remove_entity(entity_id)

    def entity_count(self) -> int:
        """实体总数."""
        return self.entity_store.count()

    # ================================================================
    # 三元组操作 (委托 TripleStore)
    # ================================================================

    def add_triple(self, triple: KnowledgeTriple) -> KnowledgeTriple:
        """添加三元组."""
        return self.triple_store.add_triple(triple)

    def get_triple(self, triple_id: str) -> KnowledgeTriple | None:
        """获取三元组."""
        return self.triple_store.get_triple(triple_id)

    def remove_triple(self, triple_id: str) -> KnowledgeTriple | None:
        """移除三元组."""
        return self.triple_store.remove_triple(triple_id)

    def triple_count(self) -> int:
        """三元组总数."""
        return self.triple_store.count()

    def get_entity_triples(
        self,
        entity_id: str,
        *,
        direction: str = "both",
        min_confidence: float = 0.0,
    ) -> list[KnowledgeTriple]:
        """获取实体的所有关联三元组."""
        outgoing = self.triple_store.get_outgoing(
            entity_id, min_confidence=min_confidence
        )
        if direction == "out":
            return outgoing

        incoming = self.triple_store.get_incoming(
            entity_id, min_confidence=min_confidence
        )
        if direction == "in":
            return incoming

        return outgoing + incoming

    # ================================================================
    # 切片操作 (委托 ChunkStore)
    # ================================================================

    def add_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """添加切片."""
        return self.chunk_store.add_chunk(chunk)

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """获取切片."""
        return self.chunk_store.get_chunk(chunk_id)

    def remove_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """移除切片."""
        return self.chunk_store.remove_chunk(chunk_id)

    def chunk_count(self) -> int:
        """切片总数."""
        return self.chunk_store.count()

    def search_text(
        self, query: str, *, top_k: int = 10, document_id: str | None = None
    ) -> list[tuple[DocumentChunk, float]]:
        """全文检索."""
        return self.chunk_store.search_text(query, top_k=top_k, document_id=document_id)

    def search_vector(
        self, query_vector: list[float], *, top_k: int = 10, filter_fn: Any = None
    ) -> list[tuple[DocumentChunk, float]]:
        """向量检索."""
        return self.chunk_store.search_vector(
            query_vector, top_k=top_k, filter_fn=filter_fn
        )

    # ================================================================
    # 版本管理 (借鉴 ConVer-G + DBpedia-TKG)
    # ================================================================

    def _get_latest_version_id(self, entity_id: str) -> str:
        """获取实体最新版本的 ID."""
        versions = self._versions.get(entity_id, [])
        return versions[-1].version_id if versions else ""

    def get_version_history(self, entity_id: str) -> list[KnowledgeVersion]:
        """获取实体的版本历史."""
        return list(self._versions.get(entity_id, []))

    def get_current_version(self, entity_id: str) -> KnowledgeVersion | None:
        """获取实体当前版本."""
        versions = self._versions.get(entity_id, [])
        return versions[-1] if versions else None

    def get_version_at(self, entity_id: str, timestamp: float) -> KnowledgeVersion | None:
        """获取指定时间点的版本 (时间旅行查询)."""
        versions = self._versions.get(entity_id, [])
        for version in reversed(versions):
            if version.is_valid_at(timestamp):
                return version
        return None

    def restore_version(
        self,
        entity_id: str,
        version_id: str,
        *,
        restored_by: str = "system",
    ) -> KnowledgeEntity:
        """恢复到指定版本 (借鉴 DBpedia 快照恢复).

        Args:
            entity_id: 实体 ID
            version_id: 要恢复的版本 ID
            restored_by: 恢复者

        Returns:
            恢复后的实体

        Raises:
            EntityNotFoundError: 实体或版本不存在
        """
        with self._lock:
            versions = self._versions.get(entity_id, [])
            target_version = None
            for v in versions:
                if v.version_id == version_id:
                    target_version = v
                    break

            if target_version is None:
                raise EntityNotFoundError(
                    version_id, detail=f"版本不存在: entity={entity_id}, version={version_id}"
                )

            entity = self.entity_store.get_entity_or_raise(entity_id)

            # 从快照恢复
            old_snapshot = entity.model_dump(mode="json")
            snapshot = dict(target_version.snapshot)

            # 使用 model_validate 正确重建枚举类型 (避免字符串残留)
            snapshot["entity_id"] = entity_id
            snapshot["version"] = entity.version + 1
            restored_entity = KnowledgeEntity.model_validate(snapshot)
            restored_entity.touch()

            # 替换存储中的实体
            self.entity_store._reindex_entity(restored_entity, old_snapshot)
            self.entity_store._entities[entity_id] = restored_entity
            entity = restored_entity

            # 记录恢复版本
            restore_version = KnowledgeVersion(
                entity_id=entity_id,
                revision_number=entity.version,
                parent_version_id=self._get_latest_version_id(entity_id),
                changeset=[
                    ChangeRecord(
                        change_type="modify",
                        entity_id=entity_id,
                        field_path="*",
                        old_value=old_snapshot,
                        new_value=snapshot,
                        changed_by=restored_by,
                        reason=f"恢复到版本 {version_id}",
                    )
                ],
                snapshot=entity.model_dump(mode="json"),
                version_note=f"恢复到版本 {version_id}",
                created_by=restored_by,
            )

            # 标记前一版本为非当前
            if versions and versions[-1].is_current():
                versions[-1].valid_until = time.time()
            versions.append(restore_version)

            return entity

    # ================================================================
    # 冲突管理 (借鉴 MACR + Detect-Then-Resolve)
    # ================================================================

    def add_conflict(self, conflict: KnowledgeConflict) -> KnowledgeConflict:
        """添加冲突记录."""
        with self._lock:
            self._conflicts[conflict.conflict_id] = conflict
            return conflict

    def get_conflict(self, conflict_id: str) -> KnowledgeConflict | None:
        """获取冲突记录."""
        return self._conflicts.get(conflict_id)

    def get_conflicts(
        self,
        *,
        entity_id: str | None = None,
        status: str | None = None,
        conflict_type: ConflictType | None = None,
    ) -> list[KnowledgeConflict]:
        """查询冲突记录 (多条件 AND)."""
        results: list[KnowledgeConflict] = []
        for conflict in self._conflicts.values():
            if entity_id is not None and conflict.entity_id != entity_id:
                continue
            if status is not None and conflict.status != status:
                continue
            if conflict_type is not None and conflict.conflict_type != conflict_type:
                continue
            results.append(conflict)
        return results

    def get_unresolved_conflicts(self) -> list[KnowledgeConflict]:
        """获取未解决的冲突."""
        return [c for c in self._conflicts.values() if not c.is_resolved()]

    def resolve_conflict(
        self,
        conflict_id: str,
        value: Any,
        *,
        claim_id: str = "",
        explanation: str = "",
        resolved_by: str = "system",
    ) -> KnowledgeConflict:
        """解决冲突."""
        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None:
                raise EntityNotFoundError(conflict_id, detail="冲突记录不存在")
            conflict.resolve(
                value=value,
                claim_id=claim_id,
                explanation=explanation,
                resolved_by=resolved_by,
            )
            return conflict

    def ignore_conflict(
        self,
        conflict_id: str,
        reason: str = "",
        by: str = "system",
    ) -> KnowledgeConflict:
        """忽略冲突."""
        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None:
                raise EntityNotFoundError(conflict_id, detail="冲突记录不存在")
            conflict.ignore(reason=reason, by=by)
            return conflict

    @property
    def conflict_count(self) -> int:
        """冲突记录总数."""
        return len(self._conflicts)

    @property
    def unresolved_conflict_count(self) -> int:
        """未解决冲突数."""
        return sum(1 for c in self._conflicts.values() if not c.is_resolved())

    # ================================================================
    # 证据管理 (借鉴 ProVe)
    # ================================================================

    def add_evidence(self, evidence: EvidenceRecord) -> EvidenceRecord:
        """添加证据记录."""
        with self._lock:
            self._evidence[evidence.evidence_id] = evidence
            return evidence

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        """获取证据记录."""
        return self._evidence.get(evidence_id)

    def get_evidence_for_entity(self, entity_id: str) -> list[EvidenceRecord]:
        """获取实体的所有证据."""
        return [e for e in self._evidence.values() if e.entity_id == entity_id]

    def get_evidence_for_triple(self, triple_id: str) -> list[EvidenceRecord]:
        """获取三元组的所有证据."""
        return [e for e in self._evidence.values() if e.triple_id == triple_id]

    @property
    def evidence_count(self) -> int:
        """证据记录总数."""
        return len(self._evidence)

    # ================================================================
    # 子图提取 (借鉴 GraphRAG)
    # ================================================================

    def extract_subgraph(self, config: SubgraphConfig) -> KnowledgeGraph:
        """提取以指定实体为中心的子图 (借鉴 GraphRAG 实体中心子图提取).

        Args:
            config: 子图提取配置

        Returns:
            包含子图的 KnowledgeGraph
        """
        with self._lock:
            entity_ids, triples = self.triple_store.traverse_bfs(
                config.entity_focus,
                max_depth=config.max_depth,
                min_confidence=config.min_confidence,
                max_entities=config.max_entities,
            )

            # 构建子图
            subgraph = KnowledgeGraph(
                domain="subgraph",
                name=f"子图: {config.entity_focus}",
                description=f"以 {config.entity_focus} 为中心，深度={config.max_depth}",
            )

            # 添加实体
            for eid in entity_ids:
                entity = self.entity_store.get_entity(eid)
                if entity is not None:
                    # 质量过滤
                    if config.min_quality > 0.0:
                        if entity.quality is None or entity.quality.overall() < config.min_quality:
                            continue
                    subgraph.add_entity(entity)

            # 添加三元组
            for triple in triples:
                if config.include_deprecated or not triple.is_deprecated():
                    subgraph.add_triple(triple)

            return subgraph

    def create_graph(
        self,
        *,
        domain: str = "general",
        name: str = "",
        description: str = "",
    ) -> KnowledgeGraph:
        """从当前存储创建知识图谱快照."""
        with self._lock:
            graph = KnowledgeGraph(
                domain=domain,
                name=name,
                description=description,
            )
            for entity in self.entity_store.list_entities(limit=10000):
                graph.add_entity(entity)
            for triple in self.triple_store._triples.values():
                graph.add_triple(triple)
            return graph

    # ================================================================
    # 结构化查询 (借鉴 SPARQL + GraphQL)
    # ================================================================

    def query(self, query: KnowledgeQuery) -> list[KnowledgeEntity]:
        """执行结构化查询 (借鉴 SPARQL 图查询).

        支持多条件 AND 组合查询，可选图遍历。

        Args:
            query: 结构化查询

        Returns:
            匹配的实体列表
        """
        with self._lock:
            # 基础过滤: 使用索引加速
            if not query.conditions:
                results = self.entity_store.list_entities(limit=10000)
            else:
                # 提取可用于索引的条件
                type_condition = None
                domain_condition = None
                name_condition = None
                other_conditions: list = []

                for cond in query.conditions:
                    if cond.field == "entity_type" and cond.operator == QueryOperator.EQ:
                        type_condition = cond
                    elif cond.field == "domain" and cond.operator == QueryOperator.EQ:
                        domain_condition = cond
                    elif cond.field == "name" and cond.operator == QueryOperator.EQ:
                        name_condition = cond
                    else:
                        other_conditions.append(cond)

                # 使用最优索引
                if type_condition:
                    results = self.entity_store.find_by_type(type_condition.value)
                elif domain_condition:
                    results = self.entity_store.find_by_domain(domain_condition.value)
                elif name_condition:
                    results = self.entity_store.find_by_name(name_condition.value)
                else:
                    results = self.entity_store.list_entities(limit=10000)

                # 应用其他条件
                if other_conditions:
                    filtered: list[KnowledgeEntity] = []
                    for entity in results:
                        entity_dict = entity.model_dump(mode="json")
                        match_all = True
                        for cond in other_conditions:
                            field_value = entity_dict.get(cond.field)
                            if not cond.matches(field_value):
                                match_all = False
                                break
                        if match_all:
                            filtered.append(entity)
                    results = filtered

            # 时间戳过滤
            if query.timestamp_filter > 0.0:
                results = [
                    e for e in results
                    if e.created_at <= query.timestamp_filter
                ]

            # 图遍历
            if query.max_hops > 0 and results:
                visited: set[str] = set()
                expanded: list[KnowledgeEntity] = []
                for entity in results:
                    if entity.entity_id not in visited:
                        entity_ids, _ = self.triple_store.traverse_bfs(
                            entity.entity_id,
                            max_depth=query.max_hops,
                            max_entities=500,
                        )
                        for eid in entity_ids:
                            if eid not in visited:
                                visited.add(eid)
                                e = self.entity_store.get_entity(eid)
                                if e is not None:
                                    expanded.append(e)
                results = expanded

            # 排序
            if query.sort_by:
                reverse = query.sort_desc
                try:
                    results.sort(
                        key=lambda e: getattr(e, query.sort_by, 0),
                        reverse=reverse,
                    )
                except (TypeError, AttributeError):
                    pass

            # 分页
            results = results[query.offset : query.offset + query.limit]

            return results

    # ================================================================
    # 批量导入 (借鉴 LlamaIndex ingestion pipeline)
    # ================================================================

    def ingest(
        self,
        *,
        entities: list[KnowledgeEntity] | None = None,
        triples: list[KnowledgeTriple] | None = None,
        chunks: list[DocumentChunk] | None = None,
        source: str = "",
    ) -> IngestResult:
        """批量导入知识 (借鉴 LlamaIndex ingestion pipeline).

        Args:
            entities: 实体列表
            triples: 三元组列表
            chunks: 切片列表
            source: 导入来源标识

        Returns:
            导入结果
        """
        start_time = time.time()
        total = 0
        success = 0
        failed = 0
        skipped = 0
        errors: list[dict[str, Any]] = []
        ingested_ids: list[str] = []

        # 导入实体
        if entities:
            for entity in entities:
                total += 1
                try:
                    self.add_entity(entity, check_duplicate=True)
                    success += 1
                    ingested_ids.append(entity.entity_id)
                except DuplicateEntityError:
                    skipped += 1
                except Exception as exc:
                    failed += 1
                    errors.append({
                        "type": "entity",
                        "id": entity.entity_id,
                        "error": str(exc),
                    })

        # 导入三元组
        if triples:
            for triple in triples:
                total += 1
                try:
                    if self.triple_store.exists(triple.triple_id):
                        skipped += 1
                        continue
                    self.add_triple(triple)
                    success += 1
                    ingested_ids.append(triple.triple_id)
                except Exception as exc:
                    failed += 1
                    errors.append({
                        "type": "triple",
                        "id": triple.triple_id,
                        "error": str(exc),
                    })

        # 导入切片
        if chunks:
            for chunk in chunks:
                total += 1
                try:
                    if self.chunk_store.exists(chunk.chunk_id):
                        skipped += 1
                        continue
                    self.add_chunk(chunk)
                    success += 1
                    ingested_ids.append(chunk.chunk_id)
                except Exception as exc:
                    failed += 1
                    errors.append({
                        "type": "chunk",
                        "id": chunk.chunk_id,
                        "error": str(exc),
                    })

        duration_ms = (time.time() - start_time) * 1000

        return IngestResult(
            source=source,
            total=total,
            success=success,
            failed=failed,
            skipped=skipped,
            errors=errors,
            ingested_ids=ingested_ids,
            duration_ms=round(duration_ms, 2),
        )

    # ================================================================
    # 检索过滤器匹配 (借鉴 Milvus + Pinecone)
    # ================================================================

    def filter_entities(self, filter: RetrievalFilter) -> list[KnowledgeEntity]:
        """使用 RetrievalFilter 过滤实体."""
        results: list[KnowledgeEntity] = []
        for entity in self.entity_store.list_entities(limit=10000):
            if filter.matches_entity(entity):
                results.append(entity)
        return results

    def filter_chunks(self, filter: RetrievalFilter) -> list[DocumentChunk]:
        """使用 RetrievalFilter 过滤切片."""
        results: list[DocumentChunk] = []
        for chunk in self.chunk_store._chunks.values():
            if filter.matches_chunk(chunk):
                results.append(chunk)
        return results

    # ================================================================
    # 统计
    # ================================================================

    def get_stats(self) -> KnowledgeBaseStats:
        """获取统一知识库统计."""
        entity_stats = self.entity_store.get_stats()
        chunk_stats = self.chunk_store.get_stats()
        triple_stats = self.triple_store.get_stats()

        # 统计数据源
        source_ids: set[str] = set()
        for entity in self.entity_store.list_entities(limit=10000):
            if entity.source:
                source_ids.add(entity.source.source_id)

        return KnowledgeBaseStats(
            total_entities=entity_stats["total_entities"],
            total_chunks=chunk_stats["total_chunks"],
            total_triples=triple_stats["total_triples"],
            total_sources=len(source_ids),
            entities_by_type=entity_stats["by_type"],
            chunks_by_modality=chunk_stats["by_modality"],
            avg_quality=entity_stats["avg_quality"],
            indexed_vectors=chunk_stats["indexed_vectors"],
            last_updated=time.time(),
        )

    def get_detailed_stats(self) -> dict[str, Any]:
        """获取详细统计信息 (含版本和冲突)."""
        with self._lock:
            version_count = sum(len(v) for v in self._versions.values())
            entities_with_versions = len(self._versions)

            return {
                "entities": self.entity_store.get_stats(),
                "triples": self.triple_store.get_stats(),
                "chunks": self.chunk_store.get_stats(),
                "versions": {
                    "total_versions": version_count,
                    "entities_with_history": entities_with_versions,
                    "avg_versions_per_entity": round(
                        version_count / max(entities_with_versions, 1), 2
                    ),
                },
                "conflicts": {
                    "total": len(self._conflicts),
                    "unresolved": sum(1 for c in self._conflicts.values() if not c.is_resolved()),
                },
                "evidence": {
                    "total": len(self._evidence),
                    "verified": sum(1 for e in self._evidence.values() if e.is_verified()),
                },
            }

    # ================================================================
    # 清理
    # ================================================================

    def clear(self) -> None:
        """清空所有存储."""
        with self._lock:
            self.entity_store.clear()
            self.triple_store.clear()
            self.chunk_store.clear()
            self._versions.clear()
            self._conflicts.clear()
            self._evidence.clear()


__all__ = [
    "EntityStore",
    "TripleStore",
    "ChunkStore",
    "KnowledgeStore",
]
