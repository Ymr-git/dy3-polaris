"""冷启动策略管理器 — 新学习者数据不足时的降级方案.

融合世界先进方案:
- FuMoE-csKT (2026): 融合解耦注意力 + 自注意力 + 个性化MoE 冷启动知识追踪
- Knewton: 群体平均参数降级策略
- ALEKS: 知识空间理论初始状态估计

策略:
1. 0 条记录: 使用群体平均参数 (theta=0.0, mastery=0.5, default VARK)
2. 1-9 条记录: 部分个性化 (加权平均群体参数与观测值)
3. 10+ 条记录: 全量个性化 (正常流程)
"""

from __future__ import annotations

import math

from dy3_polaris.l2.models import DEFAULT_INITIAL_SE

# 冷启动阈值
COLD_START_THRESHOLD: int = 10  # 少于此记录数视为冷启动
WARM_START_RATIO: float = 0.3   # 冷启动时观测值权重 (群体参数权重=1-ratio)


class LearnerColdStartManager:
    """冷启动策略管理器 (面向学情画像)."""

    # Population average parameters (group-level priors)
    POPULATION_THETA: float = 0.0
    POPULATION_MASTERY: float = 0.5
    # 群体先验 SE: 0 记录、能力完全未知时的较大不确定性 (0.5),
    # 独立于单条 IRTState 默认 SE (DEFAULT_INITIAL_SE=0.3, 冷启动观测端基准),
    # 这样冷启动混合才随记录数单调下降 (0.5 → 0.3).
    POPULATION_SE: float = 0.5

    def __init__(self, cold_start_threshold: int = COLD_START_THRESHOLD):
        self.cold_start_threshold = cold_start_threshold

    def is_cold_start(self, record_count: int) -> bool:
        """判断是否处于冷启动阶段."""
        return record_count < self.cold_start_threshold

    def get_strategy(self, record_count: int) -> str:
        """获取冷启动策略名称."""
        if record_count == 0:
            return "population_average"
        elif record_count < self.cold_start_threshold:
            return "partial_personalization"
        else:
            return "full_personalization"

    def estimate_initial_theta(
        self,
        observed_theta: float | None,
        record_count: int,
        observed_se: float | None = None,
    ) -> tuple[float, float]:
        """估计初始能力 theta 和标准误 (群体先验 + 观测值的收缩融合).

        冷启动时使用群体先验与观测值的加权平均 (Knewton 式降级).
        SE 采用精度加权 (variance blending): 观测越精确 (SE 越小) 权重越高,
        取代历史上硬编码 0.3 的线性内插, 保证 SE 随数据单调下降且与 IRT
        状态一致.

        Args:
            observed_theta: 观测到的能力估计 (None 表示无观测).
            record_count: 已记录作答次数.
            observed_se: 观测能力估计的标准误 (None 时回退线性内插).

        Returns:
            (theta, standard_error)
        """
        if record_count == 0 or observed_theta is None:
            return self.POPULATION_THETA, self.POPULATION_SE

        # Weight: 0 for 0 records, WARM_START_RATIO for threshold-1 records, 1.0 for threshold+
        weight = min(1.0, record_count / self.cold_start_threshold)

        # Weighted average of population prior and observed value
        theta = (1 - weight) * self.POPULATION_THETA + weight * observed_theta

        # SE: 精度加权 (variance blending), 观测 SE 缺失时回退线性内插
        if observed_se is not None and observed_se > 0.0:
            obs_prec = 1.0 / (observed_se * observed_se)
            pop_prec = 1.0 / (self.POPULATION_SE * self.POPULATION_SE)
            blended_prec = (1 - weight) * pop_prec + weight * obs_prec
            se = 1.0 / math.sqrt(blended_prec)
        else:
            se = self.POPULATION_SE * (1 - weight) + DEFAULT_INITIAL_SE * weight

        return theta, se

    def estimate_initial_mastery(self, observed_mastery: float | None, record_count: int) -> float:
        """估计初始掌握度."""
        if record_count == 0 or observed_mastery is None:
            return self.POPULATION_MASTERY
        weight = min(1.0, record_count / self.cold_start_threshold)
        return (1 - weight) * self.POPULATION_MASTERY + weight * observed_mastery

    def get_default_learning_style(self) -> str:
        """冷启动时返回默认学习风格 (多模态)."""
        return "multimodal"

    def recommend_initial_content(self, record_count: int, available_kps: list[str] | None = None) -> list[str]:
        """推荐初始学习内容.

        冷启动时推荐基础知识点 (难度最低).
        """
        if available_kps is None:
            return []
        # Sort by assumed difficulty (first N kps are usually foundational)
        return available_kps[:min(5, len(available_kps))]


# 向后兼容别名 (已弃用，请使用 LearnerColdStartManager)
ColdStartManager = LearnerColdStartManager

__all__ = [
    "LearnerColdStartManager",
    "ColdStartManager",  # 向后兼容别名 (已弃用)
    "COLD_START_THRESHOLD",
    "WARM_START_RATIO",
]
