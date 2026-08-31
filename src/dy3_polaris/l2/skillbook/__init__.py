"""L2 skillbook 子模块 — 技能树映射器 (BKT + IRT → 技能树可视化).

子模块构成:
1. ``SkillNode``: 技能树节点数据类
   - 字段: kp_id / name / mastery / status / level
   - status 由 mastery 自动推导 (not_started / weak / learning / mastered)
   - level 由 mastery 自动推导 (L0 / L1 / L2)
2. ``SkillEdge``: 技能树边数据类
   - 字段: from_kp / to_kp / edge_type (prerequisite / related)
3. ``SkillMapper``: 技能树映射器 (无状态引擎)
   - to_skill_tree: BKT TracingState + IRT IRTState → 技能树字典
   - get_skill_status / get_skill_level: 静态映射方法
   - get_summary: 技能树摘要统计

设计参考:
- Squirrel AI: 纳米级知识分解 + 知识图谱
- ALEKS: 知识空间理论 + 技能树
- Duolingo: 技能依赖图 + 学习路径

依赖 L2 基础设施: ``TracingState`` / ``IRTState`` (来自 ``dy3_polaris.l2.models``).
融合 L2 已实现模块:
- ``knowledge_tracer.BKTTracer`` 产出 ``TracingState`` (mastery_prob)
- ``ability_assessor.IRTEstimator`` 产出 ``IRTState`` (theta)
- 本子模块将二者映射为统一的技能树视图
"""

from __future__ import annotations

from dy3_polaris.l2.skillbook.skill_mapper import (
    SkillEdge,
    SkillMapper,
    SkillNode,
)

__all__ = [
    "SkillNode",
    "SkillEdge",
    "SkillMapper",
]
