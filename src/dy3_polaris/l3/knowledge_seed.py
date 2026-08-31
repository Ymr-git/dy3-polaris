"""L3 领域知识启动播种 — 绿色健康照明发光材料 (Dy 垂直领域).

种子知识库已清空 (旧手写文档由真实文献 wxk 目录替代)。保留本模块
以维持 seed_domain_knowledge 接口与调用方兼容; DOMAIN_KNOWLEDGE 现为空。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("dy3_polaris.l3.knowledge_seed")

DOMAIN_KNOWLEDGE: list[dict[str, Any]] = []



def seed_domain_knowledge(l3_store: Any) -> int:
    """向知识库幂等播种领域知识，返回新摄入文档数.

    按 document_id 去重：已存在的文档跳过。
    """
    seeded = 0
    try:
        from dy3_polaris.l3.ingestion import IngestionPipeline
        from dy3_polaris.l3.models import RetrievalFilter

        existing = {
            c.document_id
            for c in l3_store.filter_chunks(RetrievalFilter())
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("领域知识播种: 初始化失败 %s", exc)
        return 0

    try:
        pipeline = IngestionPipeline(l3_store)
        for doc in DOMAIN_KNOWLEDGE:
            doc_id = doc["document_id"]
            if doc_id in existing:
                continue
            try:
                # kp_id 由 L2 kp_catalog 单点派生 (kg_node → 42 KP 规范 ID)
                metadata = dict(doc.get("metadata") or {})
                node = metadata.get("kg_node", "")
                if node and "kp_id" not in metadata:
                    from dy3_polaris.l2.kp_catalog import NODE_TO_KP

                    kp_id = NODE_TO_KP.get(node)
                    if kp_id:
                        metadata["kp_id"] = kp_id
                pipeline.ingest(
                    content=doc["content"],
                    document_id=doc_id,
                    metadata=metadata,
                )
                seeded += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("领域知识播种失败 %s: %s", doc_id, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("领域知识播种管道失败: %s", exc)
        return 0
    if seeded:
        logger.info("领域知识播种完成: 新增 %d 个文档", seeded)
    return seeded


__all__ = ["DOMAIN_KNOWLEDGE", "seed_domain_knowledge"]
