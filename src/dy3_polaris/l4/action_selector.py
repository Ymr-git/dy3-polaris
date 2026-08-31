"""L4 决策引擎层 — 行动选择器 (ActionSelector).

融合世界先进方案的行动决策设计:
- OLIVIA (2026): 上下文线性赌博机 + UCB 行动选择
  - 将行动选择建模为多臂赌博机问题
  - UCB (Upper Confidence Bound) 平衡探索与利用
  - 上下文感知: 根据验证分数、意图类型、历史反馈选择最优行动
- Thompson Sampling: 贝叶斯后验采样
  - Beta-Bernoulli 共轭先验，后验采样实现自适应探索
  - 收敛速度优于 UCB，适合稳定环境
- LinUCB (Linear UCB): 上下文线性赌博机
  - 利用上下文特征线性建模回报
  - 置信区间基于岭回归，理论 regret 上界保证
- Ensemble Bandit: 多策略投票
  - 融合 UCB + Thompson + LinUCB 三种策略
  - 加权投票机制，动态调整权重
- ReACT: 推理与行动交替，验证后决定下一步
- LangGraph 条件边: 基于状态的条件路由
- TDP 框架: Supervisor 层最终决策

核心职责:
    根据 T4(ValidationReport) 选择最优行动策略，产出 ActionRecord。

四级行动策略:
    1. DIRECT_ANSWER   — 验证通过，直接输出结果
    2. TOOL_ENHANCED   — 需要调用外部工具补充信息
    3. NEGOTIATE       — 验证有警告，需要与用户确认
    4. HUMAN_CONFIRM   — 验证失败，必须转人工
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

from .models import (
    ActionRecord,
    ActionType,
    ExecutionResult,
    TaskType,
    ValidationReport,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


# ============================================================
# UCB 行动选择器
# ============================================================


class UCBActionSelector:
    """UCB 行动选择器 (借鉴 OLIVIA 上下文线性赌博机).

    维护每个行动类型的历史表现，使用 UCB1 公式平衡探索与利用:
        UCB(a) = Q(a) + c * sqrt(ln(N) / N(a))

    其中:
        Q(a): 行动 a 的平均回报
        N: 总选择次数
        N(a): 行动 a 的选择次数
        c: 探索系数 (默认 sqrt(2))
    """

    def __init__(self, exploration_constant: float = 1.414) -> None:
        """初始化 UCB 选择器.

        Args:
            exploration_constant: 探索系数 c
        """
        self._c = exploration_constant
        self._counts: dict[str, int] = {a.value: 0 for a in ActionType}
        self._values: dict[str, float] = {a.value: 0.0 for a in ActionType}
        self._total_count: int = 0

    def select(self, context: dict[str, float]) -> tuple[ActionType, float]:
        """基于 UCB 选择最优行动.

        Args:
            context: 上下文特征 (validation_score, execution_confidence, ...)

        Returns:
            (选择的行动类型, UCB 分数)
        """
        self._total_count += 1

        # 未尝试过的行动优先探索
        for action_type in ActionType:
            if self._counts[action_type.value] == 0:
                return action_type, float("inf")

        # 计算每个行动的 UCB 分数
        best_action = ActionType.DIRECT_ANSWER
        best_score = -float("inf")

        for action_type in ActionType:
            count = self._counts[action_type.value]
            value = self._values[action_type.value]

            # UCB1 公式
            exploration = math.sqrt(math.log(self._total_count) / count) if count > 0 else float("inf")
            ucb_score = value + self._c * exploration

            # 结合上下文调整 (validation_score 高时降低探索)
            validation_score = context.get("validation_score", 0.5)
            adjusted_score = ucb_score * (0.5 + 0.5 * validation_score)

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_action = action_type

        return best_action, best_score

    def update(self, action_type: ActionType, reward: float) -> None:
        """更新行动回报.

        Args:
            action_type: 被执行的行动类型
            reward: 回报 (-1 ~ 1)
        """
        key = action_type.value
        self._counts[key] += 1
        n = self._counts[key]
        # 增量平均
        self._values[key] += (reward - self._values[key]) / n

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """获取选择器统计信息."""
        return {
            a.value: {
                "count": self._counts[a.value],
                "avg_reward": round(self._values[a.value], 4),
            }
            for a in ActionType
        }


# ============================================================
# Thompson Sampling 行动选择器
# ============================================================


class ThompsonSamplingSelector:
    """Thompson Sampling 行动选择器 (贝叶斯后验采样).

    为每个行动维护 Beta(alpha, beta) 后验分布:
    - 正反馈增加 alpha
    - 负反馈增加 beta
    - 从后验采样获取选择概率

    优势:
    - 比 UCB 更快的收敛速度
    - 天然的探索-利用平衡
    - 适合稳定回报分布
    """

    def __init__(self, *, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """初始化 Thompson Sampling 选择器.

        Args:
            prior_alpha: Beta 先验 alpha (默认 1.0 = 均匀先验)
            prior_beta: Beta 先验 beta (默认 1.0 = 均匀先验)
        """
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._alpha: dict[str, float] = {a.value: prior_alpha for a in ActionType}
        self._beta: dict[str, float] = {a.value: prior_beta for a in ActionType}

    def select(self, context: dict[str, float]) -> tuple[ActionType, float]:
        """从 Beta 后验采样选择行动.

        Args:
            context: 上下文特征

        Returns:
            (选择的行动类型, 采样分数)
        """
        best_action = ActionType.DIRECT_ANSWER
        best_sample = -1.0

        for action_type in ActionType:
            # 从 Beta(alpha, beta) 采样
            sample = self._beta_sample(
                self._alpha[action_type.value],
                self._beta[action_type.value],
            )
            if sample > best_sample:
                best_sample = sample
                best_action = action_type

        return best_action, best_sample

    def update(self, action_type: ActionType, reward: float) -> None:
        """更新行动的 Beta 后验参数.

        Args:
            action_type: 被执行的行动类型
            reward: 回报 (-1 ~ 1)，正值增加 alpha，负值增加 beta
        """
        key = action_type.value
        if reward > 0:
            self._alpha[key] += reward
        elif reward < 0:
            self._beta[key] += abs(reward)
        # reward == 0 不更新

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """获取选择器统计信息."""
        return {
            a.value: {
                "alpha": round(self._alpha[a.value], 4),
                "beta": round(self._beta[a.value], 4),
                "expected_value": round(
                    self._alpha[a.value] / (self._alpha[a.value] + self._beta[a.value]), 4
                ),
            }
            for a in ActionType
        }

    @staticmethod
    def _beta_sample(alpha: float, beta: float) -> float:
        """从 Beta 分布采样.

        使用 numpy.random.beta 如果可用，否则使用简化近似。
        """
        try:
            import numpy as np
            return float(np.random.beta(alpha, beta))
        except ImportError:
            # 简化近似: 使用 Gamma 分布
            # Beta(a,b) ~ Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
            x = random.gammavariate(alpha, 1.0)
            y = random.gammavariate(beta, 1.0)
            return x / (x + y) if (x + y) > 0 else 0.5


# ============================================================
# LinUCB 上下文线性赌博机
# ============================================================


class LinUCBSelector:
    """LinUCB 上下文线性赌博机 (Linear Upper Confidence Bound).

    为每个行动维护线性模型:
        theta_a = (A_a + lambda*I)^{-1} * b_a

    选择时计算置信上界:
        score(a) = theta_a^T * x + alpha * sqrt(x^T * A_a^{-1} * x)

    优势:
    - 利用上下文特征（验证分数、执行置信度、异常标志等）
    - 理论 regret 上界 O(sqrt(T))
    - 适合上下文信息丰富的场景
    """

    def __init__(
        self,
        *,
        n_features: int = 3,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
    ) -> None:
        """初始化 LinUCB 选择器.

        Args:
            n_features: 上下文特征维度
            alpha: 探索系数
            lambda_reg: 岭回归正则化参数
        """
        self._n_features = n_features
        self._alpha = alpha
        self._lambda = lambda_reg

        # 每个行动维护 A (协方差矩阵) 和 b (回报向量)
        self._A: dict[str, list[list[float]]] = {}
        self._b: dict[str, list[float]] = {}

        for action_type in ActionType:
            # A = lambda * I (单位矩阵 * 正则化)
            self._A[action_type.value] = [
                [self._lambda if i == j else 0.0 for j in range(n_features)]
                for i in range(n_features)
            ]
            self._b[action_type.value] = [0.0] * n_features

    def select(self, context: dict[str, float]) -> tuple[ActionType, float]:
        """基于 LinUCB 选择行动.

        Args:
            context: 上下文特征

        Returns:
            (选择的行动类型, UCB 分数)
        """
        features = self._context_to_features(context)

        best_action = ActionType.DIRECT_ANSWER
        best_score = -float("inf")

        for action_type in ActionType:
            key = action_type.value
            theta = self._matvec_solve(self._A[key], self._b[key])
            mean = self._dot(theta, features)

            # 置信区间: alpha * sqrt(x^T * A^{-1} * x)
            A_inv_x = self._matvec_solve(self._A[key], features)
            confidence = self._alpha * math.sqrt(max(0.0, self._dot(features, A_inv_x)))

            score = mean + confidence

            if score > best_score:
                best_score = score
                best_action = action_type

        return best_action, best_score

    def update(
        self,
        action_type: ActionType,
        reward: float,
        context: dict[str, float],
    ) -> None:
        """更新行动的线性模型.

        Args:
            action_type: 被执行的行动类型
            reward: 回报 (-1 ~ 1)
            context: 执行时的上下文
        """
        key = action_type.value
        features = self._context_to_features(context)

        # A += x * x^T
        for i in range(self._n_features):
            for j in range(self._n_features):
                self._A[key][i][j] += features[i] * features[j]
            # b += reward * x
            self._b[key][i] += reward * features[i]

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """获取选择器统计信息."""
        stats: dict[str, dict[str, Any]] = {}
        for action_type in ActionType:
            key = action_type.value
            theta = self._matvec_solve(self._A[key], self._b[key])
            stats[key] = {
                "theta": [round(t, 4) for t in theta],
                "b_norm": round(math.sqrt(sum(x * x for x in self._b[key])), 4),
            }
        return stats

    def _context_to_features(self, context: dict[str, float]) -> list[float]:
        """将上下文字典转为特征向量.

        特征顺序: [validation_score, execution_confidence, has_anomalies]
        """
        return [
            context.get("validation_score", 0.5),
            context.get("execution_confidence", 0.5),
            context.get("has_anomalies", 0.0),
        ][: self._n_features]

    @staticmethod
    def _matvec_solve(A: list[list[float]], b: list[float]) -> list[float]:
        """解线性方程组 A * x = b (高斯消元法).

        简化实现，适用于小规模矩阵。
        """
        n = len(b)
        # 增广矩阵
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]

        # 前向消元
        for col in range(n):
            # 选主元
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            if abs(aug[col][col]) < 1e-10:
                continue  # 跳过奇异列

            for row in range(col + 1, n):
                factor = aug[row][col] / aug[col][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

        # 回代
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < 1e-10:
                x[i] = 0.0
                continue
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]

        return x

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        """向量点积."""
        return sum(x * y for x, y in zip(a, b))


# ============================================================
# 多策略投票 (Ensemble Bandit)
# ============================================================


class EnsembleActionSelector:
    """多策略投票行动选择器 (Ensemble Bandit).

    融合三种赌博机策略:
    - UCB: 经典上置信界
    - Thompson Sampling: 贝叶斯后验采样
    - LinUCB: 上下文线性赌博机

    投票机制:
    - 每个策略独立选择并打分
    - 归一化后加权求和
    - 权重根据各策略近期表现动态调整
    """

    def __init__(
        self,
        *,
        ucb_exploration: float = 1.414,
        thompson_prior_alpha: float = 1.0,
        thompson_prior_beta: float = 1.0,
        linucb_alpha: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        """初始化集成选择器.

        Args:
            ucb_exploration: UCB 探索系数
            thompson_prior_alpha: Thompson 先验 alpha
            thompson_prior_beta: Thompson 先验 beta
            linucb_alpha: LinUCB 探索系数
            weights: 策略权重 {ucb, thompson, linucb}
        """
        self._ucb = UCBActionSelector(exploration_constant=ucb_exploration)
        self._thompson = ThompsonSamplingSelector(
            prior_alpha=thompson_prior_alpha,
            prior_beta=thompson_prior_beta,
        )
        self._linucb = LinUCBSelector(alpha=linucb_alpha)

        self._weights: dict[str, float] = weights or {
            "ucb": 0.33,
            "thompson": 0.34,
            "linucb": 0.33,
        }
        self._strategy_rewards: dict[str, float] = {
            "ucb": 0.0,
            "thompson": 0.0,
            "linucb": 0.0,
        }
        self._strategy_counts: dict[str, int] = {
            "ucb": 0,
            "thompson": 0,
            "linucb": 0,
        }
        self._last_votes: dict[str, ActionType] = {}

    def select(
        self, context: dict[str, float]
    ) -> tuple[ActionType, float, str]:
        """多策略投票选择行动.

        Args:
            context: 上下文特征

        Returns:
            (选择的行动类型, 综合分数, 选择理由)
        """
        # 各策略独立选择
        ucb_action, ucb_score = self._ucb.select(context)
        thompson_action, thompson_score = self._thompson.select(context)
        linucb_action, linucb_score = self._linucb.select(context)

        self._last_votes = {
            "ucb": ucb_action,
            "thompson": thompson_action,
            "linucb": linucb_action,
        }

        # 归一化分数
        ucb_norm = self._normalize_score(ucb_score)
        thompson_norm = self._normalize_score(thompson_score)
        linucb_norm = self._normalize_score(linucb_score)

        # 加权投票
        action_scores: dict[str, float] = {a.value: 0.0 for a in ActionType}

        action_scores[ucb_action.value] += self._weights["ucb"] * ucb_norm
        action_scores[thompson_action.value] += self._weights["thompson"] * thompson_norm
        action_scores[linucb_action.value] += self._weights["linucb"] * linucb_norm

        # 历史表现加权: 已有正反馈的行动获得额外加成
        # 这确保充分训练的行动在投票中获得优势
        ucb_stats = self._ucb.get_stats()
        thompson_stats = self._thompson.get_stats()
        for action_type in ActionType:
            key = action_type.value
            ucb_count = ucb_stats[key]["count"]
            ucb_reward = ucb_stats[key]["avg_reward"]
            thompson_ev = thompson_stats[key].get("expected_value", 0.5)
            # 综合加成: UCB 回报 + Thompson 期望值
            if ucb_count > 0 and ucb_reward > 0:
                bonus = min(0.5, ucb_reward * 0.25 + (thompson_ev - 0.5) * 0.3)
                action_scores[key] += bonus

        # 选择最高分行动
        best_action = max(action_scores, key=lambda k: action_scores[k])
        best_score = action_scores[best_action]

        reason = (
            f"Ensemble 投票: UCB→{ucb_action.value}, "
            f"Thompson→{thompson_action.value}, "
            f"LinUCB→{linucb_action.value}, "
            f"胜出={best_action}"
        )

        return ActionType(best_action), best_score, reason

    def update(
        self,
        action_type: ActionType,
        reward: float,
        context: dict[str, float] | None = None,
    ) -> None:
        """更新所有策略.

        Args:
            action_type: 被执行的行动类型
            reward: 回报 (-1 ~ 1)
            context: 执行时的上下文
        """
        self._ucb.update(action_type, reward)
        self._thompson.update(action_type, reward)
        if context is not None:
            self._linucb.update(action_type, reward, context)

        # 更新策略表现追踪
        for strategy, voted_action in self._last_votes.items():
            self._strategy_counts[strategy] += 1
            if voted_action == action_type:
                self._strategy_rewards[strategy] += reward

        # 动态调整权重（每 10 次更新）
        total_updates = sum(self._strategy_counts.values())
        if total_updates > 0 and total_updates % 10 == 0:
            self._adjust_weights()

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """获取集成选择器统计信息."""
        return {
            "weights": dict(self._weights),
            "strategy_performance": {
                s: {
                    "avg_reward": round(
                        self._strategy_rewards[s] / max(1, self._strategy_counts[s]), 4
                    ),
                    "count": self._strategy_counts[s],
                }
                for s in self._strategy_rewards
            },
            "ucb_stats": self._ucb.get_stats(),
            "thompson_stats": self._thompson.get_stats(),
            "linucb_stats": self._linucb.get_stats(),
        }

    @staticmethod
    def _normalize_score(score: float) -> float:
        """将分数归一化到 [0, 1]."""
        if score == float("inf"):
            return 1.0
        if score == float("-inf"):
            return 0.0
        return max(0.0, min(1.0, score))

    def _adjust_weights(self) -> None:
        """根据近期表现调整策略权重."""
        total_reward = sum(
            max(0.01, self._strategy_rewards[s])
            for s in self._strategy_rewards
        )
        if total_reward <= 0:
            return

        new_weights: dict[str, float] = {}
        for strategy in self._weights:
            reward = max(0.01, self._strategy_rewards[strategy])
            new_weights[strategy] = reward / total_reward

        # 平滑更新（避免权重剧烈变化）
        for strategy in self._weights:
            self._weights[strategy] = (
                0.7 * self._weights[strategy] + 0.3 * new_weights[strategy]
            )

        # 归一化
        total = sum(self._weights.values())
        if total > 0:
            for s in self._weights:
                self._weights[s] /= total

        logger.debug("Ensemble 权重调整: %s", self._weights)


# ============================================================
# 规则行动选择器
# ============================================================


class RuleBasedSelector:
    """规则行动选择器 — 基于验证状态的确定性策略.

    作为 UCB 的基准对照和冷启动回退:
    - PASS       → DIRECT_ANSWER
    - INFO       → DIRECT_ANSWER (附注)
    - WARNING    → NEGOTIATE
    - ERROR      → TOOL_ENHANCED (一次) / HUMAN_CONFIRM
    - CRITICAL   → HUMAN_CONFIRM
    """

    @staticmethod
    def select(
        validation_report: ValidationReport,
        execution_result: ExecutionResult,
    ) -> tuple[ActionType, str]:
        """基于规则选择行动.

        Returns:
            (行动类型, 选择理由)
        """
        status = validation_report.overall_status
        score = validation_report.overall_score
        exec_conf = execution_result.confidence

        # CRITICAL: 必须人工确认
        if status == ValidationSeverity.CRITICAL:
            return (
                ActionType.HUMAN_CONFIRM,
                f"验证严重失败 (分数={score:.2f})，存在重大风险，需人工介入",
            )

        # ERROR: 验证失败
        if status == ValidationSeverity.ERROR:
            # 分数很低时直接人工确认 (安全优先)
            if score < 0.4:
                return (
                    ActionType.HUMAN_CONFIRM,
                    f"验证失败且分数极低 (分数={score:.2f})，需人工确认",
                )
            # 若执行置信度尚可，尝试工具增强
            if exec_conf >= 0.5:
                return (
                    ActionType.TOOL_ENHANCED,
                    f"验证未通过 (分数={score:.2f})，尝试工具增强后重试",
                )
            return (
                ActionType.HUMAN_CONFIRM,
                f"验证失败且执行置信度低 ({exec_conf:.2f})，需人工确认",
            )

        # WARNING: 有警告，协商确认
        if status == ValidationSeverity.WARNING:
            return (
                ActionType.NEGOTIATE,
                f"验证有警告 (分数={score:.2f})，建议与用户确认",
            )

        # INFO: 信息级提示，直接回答但附注
        if status == ValidationSeverity.INFO:
            return (
                ActionType.DIRECT_ANSWER,
                f"验证基本通过 (分数={score:.2f})，附信息提示",
            )

        # PASS: 直接回答
        return (
            ActionType.DIRECT_ANSWER,
            f"验证通过 (分数={score:.2f})，直接输出",
        )


# ============================================================
# 行动选择器
# ============================================================


class ActionSelector:
    """行动选择器 — T5 核心模块.

    结合规则引擎和多策略赌博机学习选择最优行动策略:
    - 冷启动/高风险场景: 使用 RuleBasedSelector (确定性)
    - 正常运行: 使用多策略赌博机 (UCB / Thompson / LinUCB / Ensemble)
    - 混合策略: 验证分数低时用规则，分数高时用赌博机

    支持的策略:
    - "ucb": 经典 UCB1 (默认，向后兼容)
    - "thompson": Thompson Sampling (贝叶斯后验采样)
    - "linucb": LinUCB (上下文线性赌博机)
    - "ensemble": 多策略投票 (UCB + Thompson + LinUCB)

    Usage::

        # 默认 UCB (向后兼容)
        selector = ActionSelector()
        record = selector.select(validation_report, execution_result)

        # Thompson Sampling
        selector = ActionSelector(strategy="thompson")

        # 多策略集成
        selector = ActionSelector(strategy="ensemble")
        record = selector.select(validation_report, execution_result)
    """

    def __init__(
        self,
        *,
        use_ucb: bool = True,
        ucb_exploration: float = 1.414,
        rule_threshold: float = 0.5,
        strategy: str = "ucb",
    ) -> None:
        """初始化行动选择器.

        Args:
            use_ucb: 是否启用学习策略 (向后兼容)
            ucb_exploration: UCB 探索系数
            rule_threshold: 强制使用规则选择的验证分数阈值（低于此值用规则）
            strategy: 学习策略 ("ucb" / "thompson" / "linucb" / "ensemble")
        """
        self._use_ucb = use_ucb
        self._rule_threshold = rule_threshold
        self._strategy = strategy
        self._rule_selector = RuleBasedSelector()

        # 根据策略初始化对应的选择器
        if not use_ucb:
            self._ucb_selector = UCBActionSelector(exploration_constant=ucb_exploration)
            self._learning_selector: Any = None
        elif strategy == "thompson":
            self._ucb_selector = UCBActionSelector(exploration_constant=ucb_exploration)
            self._learning_selector = ThompsonSamplingSelector()
        elif strategy == "linucb":
            self._ucb_selector = UCBActionSelector(exploration_constant=ucb_exploration)
            self._learning_selector = LinUCBSelector(n_features=3)
        elif strategy == "ensemble":
            self._ucb_selector = UCBActionSelector(exploration_constant=ucb_exploration)
            self._learning_selector = EnsembleActionSelector(
                ucb_exploration=ucb_exploration,
            )
        else:
            # 默认 UCB (向后兼容)
            self._ucb_selector = UCBActionSelector(exploration_constant=ucb_exploration)
            self._learning_selector = None

        logger.info(
            "ActionSelector 初始化完成 (策略=%s, UCB=%s, 规则阈值=%.2f)",
            strategy, use_ucb, rule_threshold,
        )

    def select(
        self,
        validation_report: ValidationReport,
        execution_result: ExecutionResult,
    ) -> ActionRecord:
        """选择最优行动.

        Args:
            validation_report: T4 产出的验证报告
            execution_result: T3 产出的执行结果

        Returns:
            ActionRecord 行动记录
        """
        # 构建上下文
        context = {
            "validation_score": validation_report.overall_score,
            "execution_confidence": execution_result.confidence,
            "has_anomalies": 1.0 if validation_report.anomalies else 0.0,
        }

        # 决策逻辑
        if validation_report.overall_score < self._rule_threshold:
            # 低分场景: 强制规则选择（安全优先）
            action_type, reason = self._rule_selector.select(validation_report, execution_result)
            confidence = validation_report.overall_score
        elif self._use_ucb and self._learning_selector is not None:
            # 使用指定学习策略
            action_type, confidence, reason = self._select_with_strategy(context)
        elif self._use_ucb:
            # 默认 UCB 选择 (向后兼容)
            action_type, ucb_score = self._ucb_selector.select(context)
            confidence = min(1.0, validation_report.overall_score * 0.8 + 0.2 * (ucb_score / 2.0))
            reason = f"UCB 选择 (分数={ucb_score:.2f})，验证分数={validation_report.overall_score:.2f}"
        else:
            action_type, reason = self._rule_selector.select(validation_report, execution_result)
            confidence = validation_report.overall_score

        # 构建响应载荷
        payload = self._build_payload(action_type, validation_report, execution_result)

        # 构建澄清问题（NEGOTIATE 场景）
        clarification_questions = []
        if action_type == ActionType.NEGOTIATE:
            clarification_questions = self._build_clarification_questions(validation_report)

        # 构建工具调用（TOOL_ENHANCED 场景）
        tool_calls = []
        if action_type == ActionType.TOOL_ENHANCED:
            tool_calls = self._build_tool_calls(validation_report, execution_result)

        record = ActionRecord(
            plan_id=execution_result.plan_id,
            action_type=action_type,
            confidence=round(confidence, 4),
            validation_score=validation_report.overall_score,
            execution_confidence=execution_result.confidence,
            selection_reason=reason,
            response_payload=payload,
            tool_calls=tool_calls,
            clarification_questions=clarification_questions,
        )

        logger.info(
            "行动选择: plan_id=%s, 行动=%s, 置信度=%.4f, 理由=%s",
            record.plan_id, action_type.value, confidence, reason,
        )

        return record

    def _select_with_strategy(
        self, context: dict[str, float]
    ) -> tuple[ActionType, float, str]:
        """使用指定学习策略选择行动."""
        if isinstance(self._learning_selector, EnsembleActionSelector):
            action_type, score, reason = self._learning_selector.select(context)
            confidence = min(1.0, max(0.0, score))
            return action_type, confidence, reason
        elif isinstance(self._learning_selector, ThompsonSamplingSelector):
            action_type, score = self._learning_selector.select(context)
            confidence = min(1.0, max(0.0, score))
            reason = (
                f"Thompson Sampling 选择 (采样分数={score:.4f})，"
                f"验证分数={context['validation_score']:.2f}"
            )
            return action_type, confidence, reason
        elif isinstance(self._learning_selector, LinUCBSelector):
            action_type, score = self._learning_selector.select(context)
            # LinUCB 分数可能超出 [0,1]，需要归一化
            confidence = min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-score))) if score != float("inf") else 1.0)
            reason = (
                f"LinUCB 选择 (UCB分数={score:.4f})，"
                f"验证分数={context['validation_score']:.2f}"
            )
            return action_type, confidence, reason
        else:
            # 回退到 UCB
            action_type, ucb_score = self._ucb_selector.select(context)
            confidence = min(1.0, context.get("validation_score", 0.5) * 0.8 + 0.2 * (ucb_score / 2.0))
            reason = f"UCB 选择 (分数={ucb_score:.2f})"
            return action_type, confidence, reason

    def feedback(self, action_type: ActionType, reward: float) -> None:
        """接收行动反馈，更新所有学习策略.

        Args:
            action_type: 实际执行的行动类型
            reward: 用户反馈 (-1 ~ 1)
        """
        if self._use_ucb:
            self._ucb_selector.update(action_type, reward)
            if self._learning_selector is not None:
                if isinstance(self._learning_selector, EnsembleActionSelector):
                    self._learning_selector.update(action_type, reward, context=None)
                elif isinstance(self._learning_selector, LinUCBSelector):
                    # LinUCB 需要 context，使用默认值
                    self._learning_selector.update(
                        action_type, reward,
                        context={"validation_score": 0.5, "execution_confidence": 0.5, "has_anomalies": 0.0},
                    )
                else:
                    self._learning_selector.update(action_type, reward)
            logger.debug("反馈更新: action=%s, reward=%.2f, 策略=%s", action_type.value, reward, self._strategy)

    def get_ucb_stats(self) -> dict[str, dict[str, Any]]:
        """获取选择器统计信息."""
        if self._learning_selector is not None:
            if isinstance(self._learning_selector, EnsembleActionSelector):
                # Ensemble: 合并底层统计，顶层提供 action -> stats 映射
                stats = self._learning_selector.get_stats()
                # 提取 ucb_stats 中的行动级统计作为顶层 action -> stats
                ucb_stats = stats.get("ucb_stats", {})
                # 合并 thompson_stats 的 alpha/beta 信息
                thompson_stats = stats.get("thompson_stats", {})
                merged: dict[str, dict[str, Any]] = {}
                for action_key in ucb_stats:
                    merged[action_key] = {
                        **ucb_stats[action_key],
                        **{k: v for k, v in thompson_stats.get(action_key, {}).items()},
                    }
                # 保留 ensemble 元信息
                merged["_ensemble"] = {
                    "weights": stats.get("weights", {}),
                    "strategy_performance": stats.get("strategy_performance", {}),
                }
                return merged
            elif isinstance(self._learning_selector, ThompsonSamplingSelector):
                return self._learning_selector.get_stats()
            elif isinstance(self._learning_selector, LinUCBSelector):
                return self._learning_selector.get_stats()
        return self._ucb_selector.get_stats()

    # --------------------------------------------------------
    # 响应构建
    # --------------------------------------------------------

    def _build_payload(
        self,
        action_type: ActionType,
        validation_report: ValidationReport,
        execution_result: ExecutionResult,
    ) -> dict[str, Any]:
        """构建响应载荷."""
        payload: dict[str, Any] = {
            "plan_id": execution_result.plan_id,
            "validation_status": validation_report.overall_status.value,
            "validation_score": validation_report.overall_score,
        }

        # 提取核心答案
        answers: list[Any] = []
        for tr in execution_result.get_results_by_type(TaskType.REASON):
            answers.extend(tr.output.get("answers", []))
        payload["answers"] = answers

        # 提取证据
        payload["evidence"] = execution_result.evidence_set[:10]

        # 提取推理链
        payload["reasoning_chain"] = execution_result.reasoning_chain

        # 根据行动类型添加额外信息
        if action_type == ActionType.DIRECT_ANSWER:
            payload["response_type"] = "direct"
        elif action_type == ActionType.TOOL_ENHANCED:
            payload["response_type"] = "tool_enhanced"
            payload["needs_tool"] = True
        elif action_type == ActionType.NEGOTIATE:
            payload["response_type"] = "negotiate"
            payload["needs_clarification"] = True
            payload["warnings"] = [
                a["message"] for a in validation_report.anomalies
            ]
        elif action_type == ActionType.HUMAN_CONFIRM:
            payload["response_type"] = "human_confirm"
            payload["escalation_reason"] = validation_report.recommendations[:3] if validation_report.recommendations else ["验证未通过，需人工复核"]

        return payload

    @staticmethod
    def _build_clarification_questions(validation_report: ValidationReport) -> list[str]:
        """构建澄清问题."""
        questions: list[str] = []

        for anomaly in validation_report.anomalies:
            msg = anomaly.get("message", "")
            if "事实校验" in msg:
                questions.append("您提到的数值是否有特定来源？我可以帮您核实。")
            elif "冲突" in msg:
                questions.append("不同来源对该问题有不同说法，您倾向参考哪个来源？")
            elif "证据" in msg:
                questions.append("当前证据有限，您能否提供更多背景信息？")

        if not questions:
            questions.append("结果存在一定不确定性，您希望我进一步核实哪些方面？")

        return questions[:3]

    @staticmethod
    def _build_tool_calls(
        validation_report: ValidationReport, execution_result: ExecutionResult
    ) -> list[dict[str, Any]]:
        """构建工具调用列表."""
        tools: list[dict[str, Any]] = []

        # 若事实校验失败，建议调用搜索工具
        if not validation_report.fact_check.get("passed", True):
            tools.append({
                "tool": "web_search",
                "purpose": "补充事实校验所需的外部信息",
                "query": execution_result.source_metadata.get("query", ""),
            })

        # 若证据不足，建议调用检索工具
        if len(execution_result.evidence_set) < 2:
            tools.append({
                "tool": "deep_retrieval",
                "purpose": "获取更深入的证据",
                "params": {"max_depth": 5},
            })

        return tools


__all__ = [
    "ActionSelector",
    "UCBActionSelector",
    "ThompsonSamplingSelector",
    "LinUCBSelector",
    "EnsembleActionSelector",
    "RuleBasedSelector",
]
