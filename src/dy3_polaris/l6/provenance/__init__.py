"""溯源协议 - KPA 数据包、Merkle 链、存储查询、完整性验证、审计报告.

模块组成:
- chain: KPA Merkle 链管理器
- store: 多链存储与查询引擎
- validator: 链完整性验证器
- audit: 审计报告生成器

核心概念:
- KPA (Knowledge Provenance Artifact): 知识溯源数据包
- Merkle 链: 通过 prev_hash 形成的防篡改链式结构
- 7 维溯源: 输入快照 / 处理逻辑 / 输出快照 / 上下文引用 / 置信度 / 代码哈希 / 环境哈希
"""

from __future__ import annotations

from .chain import KPAChain
from .store import ProvenanceStore
from .validator import ChainValidator, ValidationResult
from .audit import AuditReportGenerator

__all__ = [
    # 链管理
    "KPAChain",
    # 存储查询
    "ProvenanceStore",
    # 验证
    "ChainValidator",
    "ValidationResult",
    # 审计
    "AuditReportGenerator",
]
