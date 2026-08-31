"""L3 边补全 (P2 · 规则生成部分) — 多类型节点之间的结构性关系.

把实体层 (材料/离子/能级/方法/参数) 与 Topic 层、知识点层用语义关系串起来,
解决旧图「边极稀疏、实体被拍平成字符串」的痛点。此处为**规则生成**边 (source_id="rule"),
确定性、可解释; LLM 生成候选边 (source_id="llm") 单独跑, 经规则校验 + 人工抽查后并入。

关系语义 (见 l3/models.py RelationType):
  - measured_by : 性能参数 → 表征方法   (量子效率 → 积分球绝对法)
  - doped_with  : 材料     → 激活剂离子 (NaGdF4 → Dy3+)
  - has_property: 材料     → 性能参数   (YPO4 → 黄蓝比)
  - applies_to  : 知识点/机理 → 应用场景 (色坐标 → 白光 LED 与色度)

幂等: triple_id 确定性命名, 重复播种自动跳过。
"""
from __future__ import annotations

import logging
from typing import Any

from dy3_polaris.l3.models import KnowledgeTriple

logger = logging.getLogger("dy3_polaris.l3.edge_enrich")

# 规则边来源标记 (区别于手工默认 "" 与后续 LLM "llm")
_SRC = "rule"

# ---- measured_by: (性能参数, 表征方法) ----
MEASURED_BY: list[tuple[str, str]] = [
    ("par:QE", "mth:integrating-sphere"),       # 量子效率 → 积分球绝对法
    ("par:chromaticity", "mth:CIE-1931"),       # 色坐标 → CIE 1931 三刺激值
    ("par:CRI", "mth:CIE-1931"),                # 显色指数 → 光谱经 CIE 计算
    ("par:CCT", "mth:CIE-1931"),                # 色温 → CIE 坐标推导
    ("par:lifetime", "mth:PL"),                 # 荧光寿命 → 发射衰减曲线拟合
]

# ---- doped_with: (材料, 激活剂离子) ----
DOPED_WITH: list[tuple[str, str]] = [
    ("mat:NaGdF4", "ion:Dy3+"),
    ("mat:YPO4", "ion:Dy3+"),
    ("mat:BAM", "ion:Dy3+"),
]

# ---- has_property: (材料, 性能参数) ----
HAS_PROPERTY: list[tuple[str, str]] = [
    ("mat:YPO4", "par:YB-ratio"),   # YPO4 占非反演格位 → Y/B 高 (tf-dy-28)
    ("mat:BAM", "par:T50"),         # BAM 热稳定性好 (tf-dy-29)
    ("mat:NaGdF4", "par:QE"),       # NaGdF4 高效上转换基质
]

# ---- applies_to: (知识点, 健康照明应用主题) 应用主线 ----
APPLIES_TO: list[tuple[str, str]] = [
    ("kp:B-07", "topic:6.1"),   # 色坐标/色纯度/CRI → 白光 LED 与色度
    ("kp:B-07", "topic:6.2"),   # 蓝光危害/显色 → 蓝光危害与光健康
    ("kp:B-06", "topic:6.3"),   # 量子效率 → 健康照明设计
    ("kp:B-08", "topic:6.1"),   # 激发光谱(近紫外匹配) → 白光 LED 与色度
]


def _add_edge(store: Any, predicate: str, src: str, dst: str, prefix: str) -> int:
    """加一条确定性命名边, 返回实际新增数 (0 或 1)."""
    tid = f"{prefix}:{src}:{predicate}:{dst}"
    if store.get_triple(tid) is None:
        store.add_triple(KnowledgeTriple(
            triple_id=tid,
            subject_id=src,
            predicate=predicate,
            object_id=dst,
            confidence=1.0,
            source_id=_SRC,
        ))
        return 1
    return 0


def seed_structural_edges(store: Any) -> dict[str, int]:
    """幂等播种结构性规则边, 返回各关系新增计数."""
    groups: list[tuple[str, list[tuple[str, str]], str]] = [
        ("measured_by", MEASURED_BY, "mb"),
        ("doped_with", DOPED_WITH, "dw"),
        ("has_property", HAS_PROPERTY, "hp"),
        ("applies_to", APPLIES_TO, "at"),
    ]
    counts: dict[str, int] = {}
    for predicate, edges, prefix in groups:
        n = 0
        for src, dst in edges:
            n += _add_edge(store, predicate, src, dst, prefix)
        counts[predicate] = n
    total = sum(counts.values())
    logger.info("结构性规则边播种: %s (total=%d)", counts, total)
    return counts


__all__ = [
    "MEASURED_BY",
    "DOPED_WITH",
    "HAS_PROPERTY",
    "APPLIES_TO",
    "seed_structural_edges",
]
