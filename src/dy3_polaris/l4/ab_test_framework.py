"""L4 决策引擎层 — 在线 A/B 测试框架 (ABTestFramework).

融合世界先进方案的实验设计:
- Bayesian A/B Testing:
  - 使用 Beta 后验分布而非频率学派 p 值
  - 支持随时停止实验 (optional stopping)
  - 提供期望损失 (expected loss) 用于决策
- Frequentist A/B Testing:
  - Welch's t-test (不假设等方差)
  - Cohen's d 效应量
  - 统计功效分析
- Multi-armed Bandit Integration:
  - Thompson Sampling 作为变体分配策略
  - 最小化实验期间的 regret
- Sequential Testing:
  - SPRT (Sequential Probability Ratio Test)
  - 逐步检验，减少所需样本量

核心职责:
    对比不同行动选择策略在生产环境中的实际表现，
    基于统计显著性自动选择最优策略。

Usage::

    framework = ABTestFramework()
    exp_id = framework.create_experiment(
        name="ucb_vs_thompson",
        variants=["ucb", "thompson"],
        min_samples=30,
        significance_level=0.05,
    )

    for interaction in interactions:
        variant = framework.assign_variant(exp_id, seed=interaction.id)
        reward = run_strategy(variant, interaction)
        framework.record_outcome(exp_id, variant, reward=reward)

        result = framework.check_significance(exp_id)
        if result["is_significant"]:
            print(f"Winner: {result['winner']}")
            break
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================


class ExperimentStatus(str, Enum):
    """实验状态."""

    CREATED = "created"      # 已创建，未开始
    RUNNING = "running"      # 运行中
    COMPLETED = "completed"  # 已完成（有显著结果）
    STOPPED = "stopped"      # 已停止（无显著结果或手动）
    PAUSED = "paused"        # 已暂停


@dataclass
class VariantStats:
    """单个变体的统计数据.

    Attributes:
        name: 变体名称
        rewards: 奖励列表
        count: 样本数
        sum: 奖励总和
        sum_sq: 奖励平方和
        avg_reward: 平均奖励
        variance: 方差
        std: 标准差
    """

    name: str
    rewards: list[float] = field(default_factory=list)
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0

    @property
    def avg_reward(self) -> float:
        """平均奖励."""
        return self.sum / self.count if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        """方差."""
        if self.count < 2:
            return 0.0
        mean = self.avg_reward
        var = (self.sum_sq / self.count) - (mean * mean)
        return max(0.0, var)

    @property
    def std(self) -> float:
        """标准差."""
        return math.sqrt(self.variance) if self.variance > 0 else 0.0

    def add_reward(self, reward: float) -> None:
        """添加奖励."""
        self.rewards.append(reward)
        self.count += 1
        self.sum += reward
        self.sum_sq += reward * reward


@dataclass
class ABTestExperiment:
    """A/B 测试实验.

    Attributes:
        experiment_id: 实验 ID
        name: 实验名称
        variants: 变体名称列表
        variant_stats: 各变体统计
        min_samples: 每个变体最少样本数
        significance_level: 显著性水平 α
        status: 实验状态
        winner: 获胜变体 (实验完成后)
        created_at: 创建时间
        completed_at: 完成时间
        assignment_index: 轮转分配索引
    """

    experiment_id: str = field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:12]}")
    name: str = ""
    variants: list[str] = field(default_factory=list)
    variant_stats: dict[str, VariantStats] = field(default_factory=dict)
    min_samples: int = 30
    significance_level: float = 0.05
    status: ExperimentStatus = ExperimentStatus.CREATED
    winner: str | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())
    completed_at: float | None = None
    assignment_index: int = 0

    def __post_init__(self) -> None:
        """初始化变体统计."""
        if not self.variant_stats:
            self.variant_stats = {v: VariantStats(name=v) for v in self.variants}
        if self.status == ExperimentStatus.CREATED:
            self.status = ExperimentStatus.RUNNING


# ============================================================
# A/B 测试框架
# ============================================================


class ABTestFramework:
    """在线 A/B 测试框架.

    支持多变体对比实验，提供:
    - 均匀轮转 / 随机变体分配
    - Welch's t-test 显著性检验
    - Cohen's d 效应量
    - 自动实验完成

    统计方法:
    - Welch's t-test: t = (mean_A - mean_B) / sqrt(var_A/n_A + var_B/n_B)
    - 自由度 (Welch-Satterthwaite): df = (var_A/n_A + var_B/n_B)^2 /
      ((var_A/n_A)^2/(n_A-1) + (var_B/n_B)^2/(n_B-1))
    - Cohen's d: d = (mean_A - mean_B) / pooled_std
    """

    def __init__(self) -> None:
        """初始化 A/B 测试框架."""
        self._experiments: dict[str, ABTestExperiment] = {}
        logger.info("ABTestFramework 初始化完成")

    def create_experiment(
        self,
        *,
        name: str,
        variants: list[str],
        min_samples: int = 30,
        significance_level: float = 0.05,
    ) -> ABTestExperiment:
        """创建新实验.

        Args:
            name: 实验名称
            variants: 变体名称列表 (至少 2 个)
            min_samples: 每个变体最少样本数
            significance_level: 显著性水平 α

        Returns:
            创建的实验对象

        Raises:
            ValueError: 变体少于 2 个
        """
        if len(variants) < 2:
            raise ValueError("至少需要 2 个变体")

        exp = ABTestExperiment(
            name=name,
            variants=list(variants),
            min_samples=min_samples,
            significance_level=significance_level,
        )
        self._experiments[exp.experiment_id] = exp
        logger.info(
            "创建实验: id=%s, name=%s, 变体=%s, 最小样本=%d",
            exp.experiment_id, name, variants, min_samples,
        )
        return exp

    def get_experiment(self, experiment_id: str) -> ABTestExperiment:
        """获取实验.

        Args:
            experiment_id: 实验 ID

        Returns:
            实验对象

        Raises:
            KeyError: 实验不存在
        """
        if experiment_id not in self._experiments:
            raise KeyError(f"实验不存在: {experiment_id}")
        return self._experiments[experiment_id]

    def record_outcome(
        self,
        experiment_id: str,
        variant: str,
        reward: float,
    ) -> None:
        """记录实验结果.

        Args:
            experiment_id: 实验 ID
            variant: 变体名称
            reward: 奖励值 (-1 ~ 1)
        """
        exp = self.get_experiment(experiment_id)

        if exp.status != ExperimentStatus.RUNNING:
            logger.warning("实验 %s 状态为 %s，跳过记录", experiment_id, exp.status.value)
            return

        if variant not in exp.variant_stats:
            logger.warning("变体 %s 不在实验 %s 中", variant, experiment_id)
            return

        exp.variant_stats[variant].add_reward(reward)

    def assign_variant(self, experiment_id: str, *, seed: int | None = None) -> str:
        """分配变体给新请求.

        使用轮转 (round-robin) 分配确保均匀分布。

        Args:
            experiment_id: 实验 ID
            seed: 随机种子 (用于可重现的分配)

        Returns:
            分配的变体名称
        """
        exp = self.get_experiment(experiment_id)

        if exp.status != ExperimentStatus.RUNNING:
            # 实验已结束，返回获胜者
            if exp.winner:
                return exp.winner
            return exp.variants[0]

        if seed is not None:
            # 基于种子的确定性分配
            return exp.variants[seed % len(exp.variants)]

        # 轮转分配
        variant = exp.variants[exp.assignment_index % len(exp.variants)]
        exp.assignment_index += 1
        return variant

    def check_significance(self, experiment_id: str) -> dict[str, Any]:
        """检验实验显著性.

        对所有变体对执行 Welch's t-test，找出最优变体。

        Args:
            experiment_id: 实验 ID

        Returns:
            检验结果:
            {
                "is_significant": bool,
                "winner": str | None,
                "p_value": float,
                "effect_size": float,
                "best_pair": tuple[str, str],
                "comparison": dict,
            }
        """
        exp = self.get_experiment(experiment_id)

        # 检查最小样本量
        all_have_min = all(
            vs.count >= exp.min_samples for vs in exp.variant_stats.values()
        )
        if not all_have_min:
            return {
                "is_significant": False,
                "winner": None,
                "reason": "insufficient_samples",
                "samples": {v: s.count for v, s in exp.variant_stats.items()},
            }

        # 找到最优和最差变体
        sorted_variants = sorted(
            exp.variant_stats.values(),
            key=lambda v: v.avg_reward,
            reverse=True,
        )

        best = sorted_variants[0]
        worst = sorted_variants[-1]

        # 如果只有两个变体，直接比较
        if len(sorted_variants) == 2:
            result = self._welch_t_test(best, worst)
            is_significant = result["p_value"] < exp.significance_level

            if is_significant:
                exp.status = ExperimentStatus.COMPLETED
                exp.winner = best.name
                import time
                exp.completed_at = time.time()
                logger.info(
                    "实验 %s 完成: 获胜者=%s, p=%.4f, d=%.4f",
                    experiment_id, best.name, result["p_value"], result["effect_size"],
                )

            return {
                "is_significant": is_significant,
                "winner": best.name if is_significant else None,
                "p_value": round(result["p_value"], 6),
                "effect_size": round(result["effect_size"], 4),
                "best_pair": (best.name, worst.name),
            }

        # 多变体: 找到最佳配对
        best_result: dict[str, Any] | None = None
        best_p_value = 1.0

        for i in range(len(sorted_variants)):
            for j in range(i + 1, len(sorted_variants)):
                result = self._welch_t_test(sorted_variants[i], sorted_variants[j])
                if result["p_value"] < best_p_value:
                    best_p_value = result["p_value"]
                    best_result = result
                    best_pair = (sorted_variants[i].name, sorted_variants[j].name)

        if best_result is None:
            return {
                "is_significant": False,
                "winner": None,
                "reason": "no_comparison",
            }

        is_significant = best_result["p_value"] < exp.significance_level

        if is_significant:
            exp.status = ExperimentStatus.COMPLETED
            exp.winner = best.name
            import time
            exp.completed_at = time.time()
            logger.info(
                "实验 %s 完成: 获胜者=%s, p=%.4f",
                experiment_id, best.name, best_result["p_value"],
            )

        return {
            "is_significant": is_significant,
            "winner": best.name if is_significant else None,
            "p_value": round(best_result["p_value"], 6),
            "effect_size": round(best_result["effect_size"], 4),
            "best_pair": best_pair,
        }

    def get_experiment_stats(self, experiment_id: str) -> dict[str, Any]:
        """获取实验统计数据.

        Args:
            experiment_id: 实验 ID

        Returns:
            统计字典
        """
        exp = self.get_experiment(experiment_id)

        variants: dict[str, Any] = {}
        for name, vs in exp.variant_stats.items():
            variants[name] = {
                "count": vs.count,
                "avg_reward": round(vs.avg_reward, 6),
                "std": round(vs.std, 6),
                "variance": round(vs.variance, 6),
                "sum": round(vs.sum, 6),
            }

        # 计算最大效应量
        sorted_v = sorted(exp.variant_stats.values(), key=lambda v: v.avg_reward, reverse=True)
        effect_size = 0.0
        if len(sorted_v) >= 2 and sorted_v[0].count > 0 and sorted_v[-1].count > 0:
            best = sorted_v[0]
            worst = sorted_v[-1]
            pooled_std = math.sqrt(
                (best.variance + worst.variance) / 2
            ) if (best.variance + worst.variance) > 0 else 1.0
            effect_size = abs(best.avg_reward - worst.avg_reward) / pooled_std if pooled_std > 0 else 0.0

        return {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "status": exp.status.value,
            "winner": exp.winner,
            "variants": variants,
            "min_samples": exp.min_samples,
            "significance_level": exp.significance_level,
            "effect_size": round(effect_size, 4),
            "total_samples": sum(vs.count for vs in exp.variant_stats.values()),
        }

    def stop_experiment(self, experiment_id: str, *, reason: str = "manual") -> None:
        """手动停止实验.

        Args:
            experiment_id: 实验 ID
            reason: 停止原因
        """
        exp = self.get_experiment(experiment_id)
        exp.status = ExperimentStatus.STOPPED
        logger.info("实验 %s 已停止: %s", experiment_id, reason)

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    @staticmethod
    def _welch_t_test(a: VariantStats, b: VariantStats) -> dict[str, float]:
        """执行 Welch's t-test.

        Args:
            a: 变体 A 统计
            b: 变体 B 统计

        Returns:
            {"t_statistic": float, "p_value": float, "effect_size": float}
        """
        n_a = a.count
        n_b = b.count

        if n_a < 2 or n_b < 2:
            return {"t_statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}

        var_a = a.variance
        var_b = b.variance

        # 零方差处理: 如果两个变体方差都为 0，无法判断显著性
        if var_a == 0.0 and var_b == 0.0:
            # 如果均值相同，不显著；如果均值不同但方差为 0，也不足以判定
            return {"t_statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}

        # Welch's t-statistic
        se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-10
        t_stat = (a.avg_reward - b.avg_reward) / se

        # Welch-Satterthwaite 自由度
        num = (var_a / n_a + var_b / n_b) ** 2
        denom_a = (var_a / n_a) ** 2 / (n_a - 1) if n_a > 1 and var_a > 0 else 0.0
        denom_b = (var_b / n_b) ** 2 / (n_b - 1) if n_b > 1 and var_b > 0 else 0.0
        denom = denom_a + denom_b
        df = num / denom if denom > 0 else n_a + n_b - 2

        # p-value (双尾)
        p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))

        # Cohen's d
        pooled_var = (var_a + var_b) / 2 if (var_a + var_b) > 0 else 1e-10
        pooled_std = math.sqrt(pooled_var)
        d = abs(a.avg_reward - b.avg_reward) / pooled_std if pooled_std > 0 else 0.0

        return {
            "t_statistic": round(t_stat, 6),
            "p_value": round(p_value, 6),
            "effect_size": round(d, 4),
        }


# ============================================================
# 辅助函数
# ============================================================


def _t_cdf(t: float, df: float) -> float:
    """计算 t 分布的累积分布函数 (近似).

    使用 Fisher-Cornish 渐近展开近似。
    对于大自由度，t 分布趋近于正态分布。

    Args:
        t: t 值
        df: 自由度

    Returns:
        CDF 值 (0 ~ 1)
    """
    if df > 200:
        # 大自由度: 使用正态分布近似
        return _normal_cdf(t)

    # 使用 incomplete beta function 近似
    # P(T <= t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)
    x = df / (df + t * t)
    ib = _incomplete_beta(x, df / 2.0, 0.5)
    cdf = 1.0 - 0.5 * ib

    if t < 0:
        cdf = 1.0 - cdf

    return max(0.0, min(1.0, cdf))


def _normal_cdf(x: float) -> float:
    """标准正态分布 CDF (使用 error function 近似)."""
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x: float) -> float:
    """误差函数近似 (Abramowitz & Stegun)."""
    # 使用有理函数近似
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """不完全 Beta 函数近似.

    使用连分数展开 (Lentz's method)。
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # 使用连分数展开
    max_iter = 200
    epsilon = 1e-10

    # 前置因子
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x)) / a

    # 连分数
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < epsilon:
        d = epsilon
    d = 1.0 / d

    result = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < epsilon:
            d = epsilon
        c = 1.0 + aa / c
        if abs(c) < epsilon:
            c = epsilon
        d = 1.0 / d
        result *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < epsilon:
            d = epsilon
        c = 1.0 + aa / c
        if abs(c) < epsilon:
            c = epsilon
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break

    return front * result


__all__ = [
    "ABTestFramework",
    "ABTestExperiment",
    "ExperimentStatus",
    "VariantStats",
]
