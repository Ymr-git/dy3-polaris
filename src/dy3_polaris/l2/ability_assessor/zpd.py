"""最近发展区 (ZPD) 量化计算器.

融合世界先进方案:
- Vygotsky ZPD: 独立水平 → 辅助水平 → 挫败区
- IRT-based ZPD: 使用 IRT 正确率预测确定 ZPD 边界
- 2024 研究: VARK + ZPD 集成模型

ZPD 三区:
1. 独立区 (Independent): P(correct) > 0.9 — 学习者可独立完成
2. ZPD 区: 0.3 < P(correct) <= 0.9 — 需要支架支持
3. 挫败区 (Frustration): P(correct) <= 0.3 — 超出当前能力
"""

from __future__ import annotations

import math
from typing import Any
from dataclasses import dataclass


@dataclass
class ZPDResult:
    """ZPD 计算结果."""
    independent_threshold: float  # 独立区上界 (P>0.9)
    zpd_lower: float              # ZPD 下界
    zpd_upper: float              # ZPD 上界
    frustration_threshold: float  # 挫折区起点 (P<=0.3)
    optimal_difficulty: float     # ZPD 中心难度
    recommended_difficulty: float # 推荐难度 (考虑支架)


class ZPDCalculator:
    """最近发展区 (ZPD) 计算器."""

    INDEPENDENT_THRESHOLD: float = 0.9   # P(correct) > 0.9
    FRUSTRATION_THRESHOLD: float = 0.3   # P(correct) <= 0.3

    def __init__(self, independent_p: float = 0.9, frustration_p: float = 0.3):
        self.independent_p = independent_p
        self.frustration_p = frustration_p

    def calculate_zpd(self, theta: float, item_bank: list[dict[str, Any]]) -> ZPDResult:
        """计算学习者的 ZPD (解析逆解 + 题库扫描细化).

        始终先解析计算 ZPD 边界 (IRT 3PL 逆解), 保证空题库 / 全易 / 全难
        等退化情形下 ZPD 仍有明确定义 (历史上空题库返回 (0.5,0.5) 零宽区间).
        题库可用且非退化时, 用实际题目难度细化边界 (吸附到最近题目 b).

        Args:
            theta: 学习者 IRT 能力值.
            item_bank: 题库列表, 每项含 {item_id, difficulty_b, discrimination_a, guessing_c}.

        Returns:
            ZPDResult 对象.
        """
        # 解析边界 (a=1.0, c=0.0 默认; P 随 b 递减, 故下界对应 P=0.9, 上界对应 P=0.3)
        zpd_lower = self._irt_difficulty_for_p(theta, 1.0, 0.0, self.independent_p)
        zpd_upper = self._irt_difficulty_for_p(theta, 1.0, 0.0, self.frustration_p)

        # 题库可用且非退化时, 用实际题目难度细化边界
        if item_bank:
            scan_lower, scan_upper = self._scan_boundaries(theta, item_bank)
            if scan_upper > scan_lower:
                zpd_lower, zpd_upper = scan_lower, scan_upper

        # 数值安全: 确保下界 < 上界 (避免零宽区间)
        if zpd_upper <= zpd_lower:
            zpd_upper = zpd_lower + 0.5

        # Optimal difficulty = center of ZPD
        optimal = (zpd_lower + zpd_upper) / 2.0

        return ZPDResult(
            independent_threshold=zpd_lower,
            zpd_lower=zpd_lower,
            zpd_upper=zpd_upper,
            frustration_threshold=zpd_upper,
            optimal_difficulty=optimal,
            recommended_difficulty=optimal,
        )

    def _irt_difficulty_for_p(
        self,
        theta: float,
        a: float,
        c: float,
        target_p: float,
    ) -> float:
        """IRT 3PL 逆解: 求解使 P(theta|a,b,c)=target_p 的难度 b.

        由 P = c + (1-c)/(1+exp(-a(theta-b))) 反解:
            b = theta + ln((1-P)/(P-c)) / a

        要求 target_p > c 且 target_p < 1 (保证对数域非负); 不满足时安全回退
        到 theta. 用于保证 ZPD 在空题库 / 退化题库下仍有解析定义.

        Args:
            theta: 能力值.
            a: 区分度.
            c: 猜测下限.
            target_p: 目标正确率 (0, 1).

        Returns:
            难度 b.
        """
        target_p = max(min(target_p, 1.0 - 1e-9), 1e-9)
        c = max(0.0, min(c, 1.0 - 1e-9))
        if target_p <= c or target_p >= 1.0 - 1e-12:
            return theta
        a = a if a > 0.0 else 1.0
        return theta + math.log((1.0 - target_p) / (target_p - c)) / a

    def _scan_boundaries(
        self, theta: float, item_bank: list[dict[str, Any]]
    ) -> tuple[float, float]:
        """扫描题库得到 ZPD 下/上界 (吸附到实际题目难度).

        返回 (zpd_lower, zpd_upper); 无法确定边界 (全易/全难/无独立区) 时返回
        (0.0, 0.0) 表示退化, 由调用方回退到解析边界.
        """
        independent_diff: float | None = None
        zpd_upper: float | None = None
        for item in sorted(item_bank, key=lambda x: x.get('difficulty_b', 0)):
            b = item.get('difficulty_b', 0.0)
            a = item.get('discrimination_a', 1.0)
            c = item.get('guessing_c', 0.0)
            p = self._irt_probability(theta, a, b, c)
            if p > self.independent_p:
                independent_diff = b
            if p > self.frustration_p:
                zpd_upper = b
        if independent_diff is None or zpd_upper is None:
            return 0.0, 0.0
        return independent_diff, zpd_upper

    def _irt_probability(self, theta: float, a: float, b: float, c: float) -> float:
        """3PL IRT 正确率."""
        z = a * (theta - b)
        p = c + (1 - c) / (1 + math.exp(-z))
        return max(0.0, min(1.0, p))

    def recommend_difficulty(self, theta: float, scaffold_level: float = 0.5,
                            item_bank: list[dict] | None = None) -> float:
        """推荐下次学习难度.

        Args:
            theta: 学习者能力
            scaffold_level: 支架水平 [0, 1], 0=独立, 1=最大辅助
            item_bank: 可选题库

        Returns:
            推荐难度 b 值
        """
        if item_bank:
            zpd = self.calculate_zpd(theta, item_bank)
            # Target difficulty in ZPD, adjusted by scaffold level
            target = zpd.zpd_lower + scaffold_level * (zpd.zpd_upper - zpd.zpd_lower)
            return target
        # Fallback: theta + small challenge
        return theta + 0.5 * scaffold_level

    def classify_item(self, theta: float, difficulty_b: float,
                      discrimination_a: float = 1.0, guessing_c: float = 0.0) -> str:
        """分类题目对学习者的难度区域.

        Returns:
            "independent" | "zpd" | "frustration"
        """
        p = self._irt_probability(theta, discrimination_a, difficulty_b, guessing_c)
        if p > self.independent_p:
            return "independent"
        elif p > self.frustration_p:
            return "zpd"
        else:
            return "frustration"

    # --- 置信区间量化 (Confidence Interval ZPD) ---

    def calculate_zpd_ci(
        self,
        theta: float,
        se: float,
        confidence_level: float = 0.95,
    ) -> dict[str, float]:
        """基于置信区间的 ZPD 计算 — Vygotsky + IRT 集成.

        实际发展水平: theta (当前能力估计)
        潜在发展水平: theta + z_{alpha/2} * SE (置信区间上界)
        ZPD 区间: [theta, theta + z * SE]

        Args:
            theta: 当前能力估计 θ.
            se: 能力估计标准误.
            confidence_level: 置信水平, 默认 0.95.

        Returns:
            含 actual_level, potential_level, zpd_lower, zpd_upper, zpd_width 的字典.
        """
        z = self._norm_ppf(1.0 - (1.0 - confidence_level) / 2.0)
        ci_half = z * se
        actual = theta
        potential = theta + ci_half
        return {
            "actual_level": actual,
            "potential_level": potential,
            "zpd_lower": actual,
            "zpd_upper": potential,
            "zpd_width": potential - actual,
            "se": se,
            "confidence_level": confidence_level,
        }

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """标准正态分布分位数 (Beasley-Springer-Moro 近似)."""
        if p <= 0.0:
            return -3.5
        if p >= 1.0:
            return 3.5
        a = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00,
        ]
        b = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01,
        ]
        c = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00,
        ]
        d = [
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00,
        ]
        p_low = 0.02425
        p_high = 1.0 - p_low
        if p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return x

    def zpd_coverage_score(
        self,
        theta: float,
        se: float,
        administered_items: list[dict[str, Any]],
    ) -> float:
        """ZPD 覆盖得分 — 量化已施测题目在 ZPD 三区的分布均匀性.

        得分 = (覆盖区数 / 3) * (1 - 不均衡惩罚)
        三区均覆盖且分布均匀 → 接近 1.0; 仅覆盖一个区 → 接近 0.33.

        Args:
            theta: 当前能力估计 θ.
            se: 能力估计标准误 (未使用, 保留接口).
            administered_items: 已施测题目列表.

        Returns:
            覆盖得分 [0.0, 1.0].
        """
        if not administered_items:
            return 0.0
        n_independent = 0
        n_zpd = 0
        n_frustration = 0
        for item in administered_items:
            b = float(item.get("b", item.get("difficulty_b", 0.0)))
            a = float(item.get("a", item.get("discrimination_a", 1.0)))
            c = float(item.get("c", item.get("guessing_c", 0.0)))
            zone = self.classify_item(theta, b, a, c)
            if zone == "independent":
                n_independent += 1
            elif zone == "zpd":
                n_zpd += 1
            else:
                n_frustration += 1
        zones_covered = sum(1 for n in (n_independent, n_zpd, n_frustration) if n > 0)
        coverage_ratio = zones_covered / 3.0
        total = len(administered_items)
        if total > 0:
            proportions = [n / total for n in (n_independent, n_zpd, n_frustration) if n > 0]
            if len(proportions) > 1:
                mean_prop = sum(proportions) / len(proportions)
                imbalance = sum(abs(p - mean_prop) for p in proportions) / len(proportions)
            else:
                imbalance = 1.0
        else:
            imbalance = 1.0
        return coverage_ratio * (1.0 - imbalance * 0.5)

    def recommend_scaffold_level(
        self,
        theta: float,
        se: float,
        item_bank: list[dict[str, Any]] | None = None,
    ) -> float:
        """自适应支架推荐 — 基于 ZPD 宽度和能力置信度.

        低置信度 (高 SE) → 保守 (低 scaffold_level, 偏独立区)
        高置信度 (低 SE) → 挑战性 (高 scaffold_level, 偏 ZPD 上界)

        scaffold_level = clamp(0.5 + (0.3 - SE) / 0.6, 0.2, 0.9)

        Args:
            theta: 当前能力估计 θ.
            se: 能力估计标准误.
            item_bank: 可选题库 (可选, 用于参考 ZPD 边界).

        Returns:
            推荐支架水平 ∈ [0.2, 0.9].
        """
        # 基础支架: SE=0.3 → 0.5 (ZPD 中心)
        scaffold = 0.5 + (0.3 - se) / 0.6
        return max(0.2, min(0.9, scaffold))

    def recommend_learning_path(
        self,
        theta: float,
        se: float,
        item_bank: list[dict[str, Any]],
        n_steps: int = 5,
    ) -> list[dict[str, Any]]:
        """ZPD 学习路径 — 在 ZPD 区间内推荐递进难度序列.

        从 ZPD 下界 (theta) 到上界 (theta + z*SE) 均匀取 n_steps 个难度点,
        在题库中为每个难度点找最接近的题目. 仅选择难度落在 ZPD 区间内的题目,
        确保学习路径始终在最近发展区内.

        Args:
            theta: 当前能力估计 θ.
            se: 能力估计标准误.
            item_bank: 可选题库.
            n_steps: 推荐步数.

        Returns:
            题目列表 (按难度递增排序), 每项含 item_id, a, b, c.
        """
        if not item_bank:
            return []
        zpd_ci = self.calculate_zpd_ci(theta, se)
        lower = zpd_ci["zpd_lower"]
        upper = zpd_ci["zpd_upper"]
        # 过滤题库: 仅保留难度在 ZPD 区间内的题目
        zpd_items = [
            item for item in item_bank
            if lower - 0.5 <= float(item.get("b", item.get("difficulty_b", 0.0))) <= upper + 0.5
        ]
        if not zpd_items:
            return []
        step = (upper - lower) / max(n_steps, 1)
        path: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for i in range(n_steps):
            target_b = lower + (i + 1) * step
            best_item = None
            best_dist = float("inf")
            for item in zpd_items:
                item_id = item.get("item_id")
                if item_id in used_ids:
                    continue
                b = float(item.get("b", item.get("difficulty_b", 0.0)))
                dist = abs(b - target_b)
                if dist < best_dist:
                    best_dist = dist
                    best_item = item
            if best_item is not None:
                used_ids.add(best_item.get("item_id"))
                path.append({
                    "item_id": best_item.get("item_id"),
                    "a": float(best_item.get("a", best_item.get("discrimination_a", 1.0))),
                    "b": float(best_item.get("b", best_item.get("difficulty_b", 0.0))),
                    "c": float(best_item.get("c", best_item.get("guessing_c", 0.0))),
                })
        path.sort(key=lambda x: x["b"])
        return path

    def classify_item_ci(
        self,
        theta: float,
        se: float,
        difficulty_b: float,
        discrimination_a: float = 1.0,
        guessing_c: float = 0.0,
        confidence_level: float = 0.95,
    ) -> str:
        """含置信区间信息的 ZPD 区分类.

        使用点估计 theta 做主分类, 同时用置信区间宽度做不确定性修正:
        - 当 SE 较大 (置信区间宽) 时, 对边界分类做保守调整
        - 当 SE 较小 (置信区间窄) 时, 点估计分类可靠, 直接采用

        分类规则:
        - independent: P(theta) > independent_p (点估计)
        - frustration: P(theta) <= frustration_p (点估计)
        - zpd: 其他

        置信区间用于 calculate_zpd_ci 中的 ZPD 宽度量化, 此处仅做分类.

        Args:
            theta: 当前能力估计 θ.
            se: 能力估计标准误.
            difficulty_b: 题目难度.
            discrimination_a: 区分度.
            guessing_c: 猜测下限.
            confidence_level: 置信水平 (保留接口, 用于未来扩展).

        Returns:
            "independent" | "zpd" | "frustration"
        """
        p = self._irt_probability(theta, discrimination_a, difficulty_b, guessing_c)
        if p > self.independent_p:
            return "independent"
        if p <= self.frustration_p:
            return "frustration"
        return "zpd"


__all__ = [
    "ZPDCalculator",
    "ZPDResult",
]
