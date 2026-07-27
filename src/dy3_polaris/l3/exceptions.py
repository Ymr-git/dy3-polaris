"""L3 领域知识层 — 异常定义.

继承 L6Error 异常体系，集成 JSON-RPC 错误码。
错误码范围 -32400 ~ -32414。

异常层级:
    L6Error
      └─ L3Error (-32400)
            ├─ EntityNotFoundError (-32401)
            ├─ DuplicateEntityError (-32402)
            ├─ OntologyValidationError (-32403)
            ├─ QualityAssessmentError (-32404)
            ├─ ProvenanceError (-32405)
            ├─ ChunkingError (-32406)
            ├─ EmbeddingError (-32407)
            ├─ RetrievalError (-32408)
            ├─ IngestError (-32409)
            ├─ ConflictError (-32410)
            ├─ VersionConflictError (-32411)
            ├─ QueryError (-32412)
            ├─ InferenceError (-32413)
            └─ EntityMergeError (-32414)
"""

from __future__ import annotations

from typing import Any

from ..l6.core.exceptions import L6Error


class L3Error(L6Error):
    """L3 领域知识层基础异常."""

    def __init__(
        self,
        code: str = "L3_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32400


class EntityNotFoundError(L3Error):
    """知识实体未找到."""

    def __init__(
        self,
        entity_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.entity_id = entity_id
        super().__init__(
            "L3_ENTITY_NOT_FOUND",
            detail or f"entity_id={entity_id}",
            {"entity_id": entity_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32401


class DuplicateEntityError(L3Error):
    """知识实体重复."""

    def __init__(
        self,
        entity_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.entity_id = entity_id
        super().__init__(
            "L3_DUPLICATE_ENTITY",
            detail or f"entity_id={entity_id}",
            {"entity_id": entity_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32402


class OntologyValidationError(L3Error):
    """本体验证失败 (实体类型/属性/关系不符合本体约束)."""

    def __init__(
        self,
        entity_type: str = "",
        violation: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.entity_type = entity_type
        self.violation = violation
        super().__init__(
            "L3_ONTOLOGY_VALIDATION",
            detail or f"entity_type={entity_type}, violation={violation}",
            {"entity_type": entity_type, "violation": violation, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32403


class QualityAssessmentError(L3Error):
    """知识质量评估失败."""

    def __init__(
        self,
        dimension: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.dimension = dimension
        super().__init__(
            "L3_QUALITY_ASSESSMENT",
            detail or f"dimension={dimension}",
            {"dimension": dimension, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32404


class ProvenanceError(L3Error):
    """知识溯源链断裂或无效."""

    def __init__(
        self,
        entity_id: str = "",
        chain_break: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.entity_id = entity_id
        self.chain_break = chain_break
        super().__init__(
            "L3_PROVENANCE",
            detail or f"entity_id={entity_id}, break={chain_break}",
            {"entity_id": entity_id, "chain_break": chain_break, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32405


class ChunkingError(L3Error):
    """文档切片失败."""

    def __init__(
        self,
        document_id: str = "",
        strategy: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.document_id = document_id
        self.strategy = strategy
        super().__init__(
            "L3_CHUNKING",
            detail or f"document_id={document_id}, strategy={strategy}",
            {"document_id": document_id, "strategy": strategy, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32406


class EmbeddingError(L3Error):
    """向量化编码失败."""

    def __init__(
        self,
        content_id: str = "",
        model: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.content_id = content_id
        self.model = model
        super().__init__(
            "L3_EMBEDDING",
            detail or f"content_id={content_id}, model={model}",
            {"content_id": content_id, "model": model, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32407


class RetrievalError(L3Error):
    """知识检索失败."""

    def __init__(
        self,
        query: str = "",
        reason: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.query = query
        self.reason = reason
        super().__init__(
            "L3_RETRIEVAL",
            detail or f"query={query}, reason={reason}",
            {"query": query, "reason": reason, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32408


class IngestError(L3Error):
    """知识导入失败."""

    def __init__(
        self,
        source: str = "",
        count: int = 0,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.count = count
        super().__init__(
            "L3_INGEST",
            detail or f"source={source}, count={count}",
            {"source": source, "count": count, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32409


class ConflictError(L3Error):
    """知识冲突错误 (借鉴 MACR 多智能体冲突解决框架).

    当不同来源的知识声明相互矛盾时触发。
    冲突类型包括时间冲突、来源冲突和语义冲突。
    """

    def __init__(
        self,
        conflict_type: str = "",
        entity_id: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.conflict_type = conflict_type
        self.entity_id = entity_id
        super().__init__(
            "L3_CONFLICT",
            detail or f"conflict_type={conflict_type}, entity_id={entity_id}",
            {"conflict_type": conflict_type, "entity_id": entity_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32410


class VersionConflictError(L3Error):
    """版本冲突错误 (借鉴 ConVer-G 并发版本管理).

    当并发修改导致版本不一致，或版本链断裂时触发。
    """

    def __init__(
        self,
        entity_id: str = "",
        expected_version: int = 0,
        actual_version: int = 0,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            "L3_VERSION_CONFLICT",
            detail or f"entity_id={entity_id}, expected=v{expected_version}, actual=v{actual_version}",
            {
                "entity_id": entity_id,
                "expected_version": expected_version,
                "actual_version": actual_version,
                **(context or {}),
            },
        )

    def _jsonrpc_code(self) -> int:
        return -32411


class QueryError(L3Error):
    """知识查询错误.

    当结构化查询语法错误、查询超时或查询条件无效时触发。
    """

    def __init__(
        self,
        query: str = "",
        reason: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.query = query
        self.reason = reason
        super().__init__(
            "L3_QUERY",
            detail or f"query={query}, reason={reason}",
            {"query": query, "reason": reason, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32412


class InferenceError(L3Error):
    """本体推理错误 (借鉴 OWL Reasoner 推理失败场景).

    当推理规则应用失败、推理结果矛盾或推理超时时触发。
    """

    def __init__(
        self,
        rule_type: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.rule_type = rule_type
        super().__init__(
            "L3_INFERENCE",
            detail or f"rule_type={rule_type}",
            {"rule_type": rule_type, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32413


class EntityMergeError(L3Error):
    """实体合并错误 (借鉴 Entity Resolution 合并失败场景).

    当实体去重/合并过程中发现不可调和的冲突时触发。
    """

    def __init__(
        self,
        source_entity_id: str = "",
        target_entity_id: str = "",
        reason: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.source_entity_id = source_entity_id
        self.target_entity_id = target_entity_id
        self.reason = reason
        super().__init__(
            "L3_ENTITY_MERGE",
            detail or f"source={source_entity_id}, target={target_entity_id}, reason={reason}",
            {
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "reason": reason,
                **(context or {}),
            },
        )

    def _jsonrpc_code(self) -> int:
        return -32414


__all__ = [
    "L3Error",
    "EntityNotFoundError",
    "DuplicateEntityError",
    "OntologyValidationError",
    "QualityAssessmentError",
    "ProvenanceError",
    "ChunkingError",
    "EmbeddingError",
    "RetrievalError",
    "IngestError",
    "ConflictError",
    "VersionConflictError",
    "QueryError",
    "InferenceError",
    "EntityMergeError",
]
