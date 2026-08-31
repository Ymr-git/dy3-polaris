"""能力等级估计器 — BKT + IRT 融合的能力分级.

融合世界先进方案:
- Knewton: IRT 驱动的能力分级
- ALEKS: 知识空间理论 + 掌握阈值 (0.85)
- Khan Academy: 掌握度 + 能力综合分级

分级规则:
- beginner: theta < -0.5 或 avg_mastery < 0.4
- intermediate: -0.5 <= theta < 0.5 且 0.4 <= avg_mastery < 0.7
- advanced: theta >= 0.5 且 avg_mastery >= 0.7

设计说明:
- LevelEstimator 为无状态引擎类, 不持有学习者状态.
- 分级采用 "优先级短路" 策略:
  1. 先判定 beginner (任一条件满足即可, 最宽松);
  2. 再判定 advanced (两个条件均需满足, 最严格);
  3. 其余落入 intermediate (含间隙区域, 作为默认兜底).
- 间隙区域示例: theta=0.6 (>= 0.5) 但 mastery=0.5 (< 0.7),
  既不满足 advanced 也未触发 beginner, 归入 intermediate.
"""

from __future__ import annotations


# ============================================================
# 1. 常量定义
# ============================================================

# 能力等级标签
LEVEL_BEGINNER: str = "beginner"
LEVEL_INTERMEDIATE: str = "intermediate"
LEVEL_ADVANCED: str = "advanced"

# theta 阈值
_BEGINNER_THETA: float = -0.5    # theta < 此值 -> beginner
_INTERMEDIATE_THETA: float = 0.5  # theta >= 此值且 mastery 达标 -> advanced

# 掌握度阈值
_BEGINNER_MASTERY: float = 0.4    # mastery < 此值 -> beginner
_ADVANCED_MASTERY: float = 0.7    # mastery >= 此值 (且 theta 达标) -> advanced

# 浮点边界容差: 避免 0.4 经算术运算得到 0.39999999999999997 时被误判为 beginner
_EPSILON: float = 1e-9

# --- 稳定性增强常量 (Schmitt 滞回 + 置信门控) ---
# 升级/降级需越过的死区半宽: θ 在 ±0.5、mastery 在 0.4/0.7 附近小幅穿越时
# 不再立即翻转等级标签 (解决"认真答/乱答导致能力等级波动大"的离散化抖动).
_HYST_THETA: float = 0.25
_HYST_MASTERY: float = 0.10
# 置信门控: SE 高于此值或作答次数低于阈值时视为数据不足, 不放大极端判定.
_SE_GATE: float = 0.6
_MIN_RESPONSES: int = 5


# ============================================================
# 2. LevelEstimator 无状态引擎类
# ============================================================


class LevelEstimator:
    """能力等级估计器 (无状态引擎).

    融合 IRT 能力参数 theta 与 BKT 平均掌握度 avg_mastery,
    将学习者划分为 beginner / intermediate / advanced 三级.

    分级优先级 (短路求值):
    1. beginner: theta < -0.5 或 avg_mastery < 0.4
    2. advanced: theta >= 0.5 且 avg_mastery >= 0.7
    3. intermediate: 其余情况 (含间隙区域, 作为默认)

    无状态: 相同输入产生相同输出, 可安全多实例并发使用.
    """

    def estimate(
        self,
        theta: float,
        avg_mastery: float,
        *,
        prev_level: str | None = None,
        se: float | None = None,
        response_count: int | None = None,
    ) -> str:
        """融合 theta 与 avg_mastery 估计能力等级 (含滞回 + 置信门控).

        基础分级 (优先级短路):
        1. beginner: theta < -0.5 或 avg_mastery < 0.4
           — 任一条件满足即判定为初学者 (最宽松).
        2. advanced: theta >= 0.5 且 avg_mastery >= 0.7
           — 两个条件均需满足 (最严格).
        3. intermediate: 其余情况 (含间隙区域, 默认兜底).

        稳定性增强 (解决能力等级随单题波动反复横跳):
        - 首次调用 (prev_level=None) 且数据不足 (SE 大 / 作答少):
          不轻易判 beginner/advanced, 保守归入 intermediate.
        - 有历史等级 (prev_level 提供): 施加 Schmitt 滞回 —— 升/降级需越过
          上/下边界带, 避免 θ 在 ±0.5、mastery 在 0.4·0.7 附近小幅穿越就翻转.

        Args:
            theta: IRT 能力参数 θ (标准分尺度, 可正可负).
            avg_mastery: BKT 平均掌握度 [0.0, 1.0].
            prev_level: 上一次能力等级 (用于滞回), None 表示首次估计.
            se: IRT 估计标准误 (用于置信门控), None 表示不门控.
            response_count: 已纳入估计的作答次数 (用于置信门控), None 表示不门控.

        Returns:
            能力等级标签: "beginner" / "intermediate" / "advanced".
        """
        # 无历史等级 -> 基础分级 (可选置信门控)
        if prev_level is None:
            raw = self._raw_level(theta, avg_mastery)
            low_confidence = (
                (se is not None and se > _SE_GATE)
                or (response_count is not None and response_count < _MIN_RESPONSES)
            )
            # 低置信时保守: 不放大到 beginner/advanced 的极端判定
            if raw != LEVEL_INTERMEDIATE and low_confidence:
                return LEVEL_INTERMEDIATE
            return raw

        # 有历史等级 -> 非对称阈值 (Schmitt 滞回) 抑制边界反复横跳
        return self._level_with_hysteresis(theta, avg_mastery, prev_level)

    @staticmethod
    def _raw_level(theta: float, avg_mastery: float) -> str:
        """基础分级 (无滞回), 保留原始阈值逻辑."""
        # 1. beginner: 任一条件满足 (最宽松)
        if (
            theta < _BEGINNER_THETA - _EPSILON
            or avg_mastery < _BEGINNER_MASTERY - _EPSILON
        ):
            return LEVEL_BEGINNER

        # 2. advanced: 两个条件均满足 (最严格)
        if (
            theta >= _INTERMEDIATE_THETA - _EPSILON
            and avg_mastery >= _ADVANCED_MASTERY - _EPSILON
        ):
            return LEVEL_ADVANCED

        # 3. intermediate: 其余情况 (含间隙区域, 默认兜底)
        return LEVEL_INTERMEDIATE

    @staticmethod
    def _level_with_hysteresis(
        theta: float, avg_mastery: float, prev_level: str
    ) -> str:
        """按 prev_level 施加非对称阈值 (Schmitt trigger), 双向抑制边界横跳.

        两个等级边界 (beginner↔intermediate↔advanced) 均带死区带:
        - 升 intermediate (自 beginner): 需越过上边界 (theta >= -0.5+HYST 且
          mastery >= 0.4+HYST).
        - 降 intermediate (自 advanced): 需跌破下边界 (theta < 0.5-HYST 或
          mastery < 0.7-HYST).
        - 自 intermediate: 升 advanced 需越过上边界 (theta >= 0.5+HYST 且
          mastery >= 0.7+HYST), 降 beginner 需跌破下边界 (theta < -0.5-HYST
          或 mastery < 0.4-HYST); 死区内保持 intermediate.
        """
        if prev_level == LEVEL_BEGINNER:
            if (
                theta >= _BEGINNER_THETA + _HYST_THETA
                and avg_mastery >= _BEGINNER_MASTERY + _HYST_MASTERY
            ):
                return LEVEL_INTERMEDIATE
            return LEVEL_BEGINNER

        if prev_level == LEVEL_ADVANCED:
            if (
                theta < _INTERMEDIATE_THETA - _HYST_THETA
                or avg_mastery < _ADVANCED_MASTERY - _HYST_MASTERY
            ):
                return LEVEL_INTERMEDIATE
            return LEVEL_ADVANCED

        # prev_level == intermediate: 双向死区
        if (
            theta >= _INTERMEDIATE_THETA + _HYST_THETA
            and avg_mastery >= _ADVANCED_MASTERY + _HYST_MASTERY
        ):
            return LEVEL_ADVANCED
        if (
            theta < _BEGINNER_THETA - _HYST_THETA
            or avg_mastery < _BEGINNER_MASTERY - _HYST_MASTERY
        ):
            return LEVEL_BEGINNER
        return LEVEL_INTERMEDIATE


# ============================================================
# __all__
# ============================================================

__all__ = [
    "LevelEstimator",
]
