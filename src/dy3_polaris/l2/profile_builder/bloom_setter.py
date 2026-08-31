"""Bloom 认知层次目标设定器.

融合世界先进方案:
- Bloom (1956) / Anderson & Krathwohl (2001): 认知六层次
- Mastery Learning: 掌握当前层次后提升目标
- ZPD (Vygotsky): 目标应在最近发展区内

六层次 (由低到高):
remember -> understand -> apply -> analyze -> evaluate -> create

设计说明:
- BloomSetter 为无状态引擎类, 不持有学习者状态.
- 默认目标设定为当前层次的高一级 (Mastery Learning + ZPD 原则).
- 已到最高级 (create) 时保持不变 (不超越六层次框架).
- 支持显式指定 goal_level (如个性化目标设定).
"""

from __future__ import annotations


# ============================================================
# 1. 常量定义
# ============================================================

# Bloom 认知六层次 (由低到高有序)
BLOOM_LEVELS: list[str] = [
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
]

# 层次 -> 索引映射 (快速查找)
_BLOOM_INDEX: dict[str, int] = {level: i for i, level in enumerate(BLOOM_LEVELS)}


# ============================================================
# 2. BloomSetter 无状态引擎类
# ============================================================


class BloomSetter:
    """Bloom 认知层次目标设定器 (无状态引擎).

    根据学习者当前 Bloom 认知层次设定目标层次:
    1. 默认: 目标比当前高一级 (Mastery Learning 原则, 目标在 ZPD 内).
    2. 显式: 通过 goal_level 参数直接指定目标层次.
    3. 封顶: 已到最高级 (create) 时保持不变.

    六层次 (由低到高):
    remember -> understand -> apply -> analyze -> evaluate -> create

    无状态: 相同输入产生相同输出, 可安全多实例并发使用.
    """

    # Bloom 六层次常量 (类属性, 供外部引用)
    BLOOM_LEVELS: list[str] = BLOOM_LEVELS

    def set_target(
        self,
        current_level: str,
        goal_level: str | None = None,
    ) -> str:
        """设定 Bloom 认知目标层次.

        目标设定规则:
        1. 若 ``goal_level`` 显式指定: 校验有效性后直接返回该层次.
        2. 若 ``goal_level`` 为 None (默认):
           - 当前层次非最高级: 返回高一级层次.
           - 当前层次为最高级 (create): 保持不变.

        Args:
            current_level: 当前 Bloom 认知层次 (六层次之一).
            goal_level: 显式指定的目标层次 (可选). None 时自动设定为高一级.

        Returns:
            目标 Bloom 认知层次.

        Raises:
            ValueError: ``current_level`` 或 ``goal_level`` 不在六层次中.
        """
        # 校验当前层次
        if current_level not in _BLOOM_INDEX:
            raise ValueError(
                f"无效的 Bloom 层次: {current_level!r}, "
                f"有效值: {BLOOM_LEVELS}"
            )

        # 显式指定目标层次: 校验后直接返回
        if goal_level is not None:
            if goal_level not in _BLOOM_INDEX:
                raise ValueError(
                    f"无效的 Bloom 目标层次: {goal_level!r}, "
                    f"有效值: {BLOOM_LEVELS}"
                )
            return goal_level

        # 默认: 高一级 (已到最高级则保持)
        idx = _BLOOM_INDEX[current_level]
        if idx >= len(BLOOM_LEVELS) - 1:
            return BLOOM_LEVELS[idx]  # 最高级保持不变
        return BLOOM_LEVELS[idx + 1]


# ============================================================
# __all__
# ============================================================

__all__ = [
    "BloomSetter",
    "BLOOM_LEVELS",
]
