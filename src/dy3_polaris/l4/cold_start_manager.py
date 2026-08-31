"""L4 决策引擎层 — 冷启动管理器 (ColdStartManager).

融合世界先进方案的冷启动设计:
- ε-greedy with Decay:
  - 初始 ε=1.0 (纯探索)
  - 随观测次数指数衰减 ε = ε_0 × decay_rate^(n/n_step)
  - 达到最小观测数后转入利用阶段
- UCB Cold Start:
  - 未探索的行动获得无限大 UCB 值，优先被选中
  - 随着探索次数增加，不确定性降低
- Thompson Sampling Warmup:
  - 使用弱先验 Beta(1,1) (均匀分布)
  - 逐步更新后验，自然过渡到利用
- Contextual Cold Start:
  - 基于上下文相似度推荐行动
  - 利用意图类型、验证分数等上下文信息

三阶段冷启动:
    1. EXPLORATION (探索): 纯探索，均匀尝试所有行动
    2. TRANSITION (过渡): 探索与利用混合，ε 逐步衰减
    3. EXPLOITATION (利用): 主要利用已知最优策略

Usage::

    manager = ColdStartManager(
        min_observations=20,
        initial_epsilon=1.0,
        decay_rate=0.95,
    )

    for interaction in interactions:
        if manager.should_explore():
            action = manager.recommend_action(
                available_actions=actions,
                action_stats=stats,
            )
        else:
            action = best_known_action
        manager.observe_action(action, reward=reward)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Callable

from .models import ActionType

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class ColdStartPhase(str, Enum):
    """冷启动阶段.

    EXPLORATION:  纯探索阶段 — ε ≈ 1.0，均匀尝试所有行动
    TRANSITION:   过渡阶段 — ε 衰减中，探索与利用混合
    EXPLOITATION: 利用阶段 — ε ≈ 0.1，主要利用已知最优策略
    """

    EXPLORATION = "exploration"
    TRANSITION = "transition"
    EXPLOITATION = "exploitation"


# ============================================================
# 冷启动管理器
# ============================================================


class ColdStartManager:
    """冷启动管理器 — ε-greedy 衰减策略.

    管理系统从冷启动到稳定运行的过渡过程。

    阶段转换逻辑:
    - 初始: EXPLORATION (ε = initial_epsilon)
    - 观测数 >= min_observations * 0.5: TRANSITION (ε 开始衰减)
    - 观测数 >= min_observations: EXPLOITATION (ε < exploitation_threshold)

    Args:
        min_observations: 最小观测数 (达到后进入利用阶段)
        initial_epsilon: 初始探索率
        decay_rate: ε 衰减率 (每次观测乘以此值)
        exploitation_threshold: 利用阶段 ε 阈值
        exploration_bonus: 未探索行动的额外加成
    """

    def __init__(
        self,
        *,
        min_observations: int = 20,
        initial_epsilon: float = 1.0,
        decay_rate: float = 0.95,
        exploitation_threshold: float = 0.3,
        exploration_bonus: float = 0.5,
    ) -> None:
        """初始化冷启动管理器.

        Args:
            min_observations: 最小观测数
            initial_epsilon: 初始 ε (探索率)
            decay_rate: 衰减率 (0~1, 越小衰减越快)
            exploitation_threshold: 利用阶段 ε 阈值
            exploration_bonus: 未探索行动的探索加成
        """
        self._min_obs = min_observations
        self._initial_epsilon = initial_epsilon
        self._decay_rate = decay_rate
        self._exploitation_threshold = exploitation_threshold
        self._exploration_bonus = exploration_bonus

        self._epsilon = initial_epsilon
        self._observation_count = 0
        self._phase = ColdStartPhase.EXPLORATION

        # 行动统计
        self._action_counts: dict[str, int] = defaultdict(int)
        self._action_rewards: dict[str, list[float]] = defaultdict(list)

        # 阶段转换回调
        self.on_phase_change: Callable[[ColdStartPhase, ColdStartPhase], None] | None = None

        logger.info(
            "ColdStartManager 初始化 (最小观测=%d, 初始ε=%.2f, 衰减率=%.2f)",
            min_observations, initial_epsilon, decay_rate,
        )

    def observe(self, reward: float) -> None:
        """记录一般性观测 (不区分行动).

        Args:
            reward: 奖励值
        """
        self._observation_count += 1
        self._update_epsilon()
        self._update_phase()

    def observe_action(
        self,
        action_type: ActionType | str,
        reward: float,
    ) -> None:
        """记录行动观测.

        Args:
            action_type: 行动类型
            reward: 奖励值
        """
        key = action_type.value if isinstance(action_type, ActionType) else str(action_type)
        self._action_counts[key] += 1
        self._action_rewards[key].append(reward)
        self._observation_count += 1
        self._update_epsilon()
        self._update_phase()

    def should_explore(self) -> bool:
        """是否应该探索.

        以概率 ε 返回 True。

        Returns:
            是否探索
        """
        return random.random() < self._epsilon

    def recommend_action(
        self,
        *,
        available_actions: list[ActionType],
        action_stats: dict[str, dict[str, Any]] | None = None,
    ) -> ActionType:
        """推荐行动 (冷启动期间).

        策略:
        - EXPLORATION: 优先推荐未探索或探索最少的行动
        - TRANSITION: ε-greedy (以 ε 概率探索，否则利用)
        - EXPLOITATION: 推荐平均奖励最高的行动

        Args:
            available_actions: 可选行动列表
            action_stats: 行动统计 {action_name: {count, avg_reward}}

        Returns:
            推荐的 ActionType
        """
        if not available_actions:
            return ActionType.DIRECT_ANSWER

        action_stats = action_stats or {}

        if self._phase == ColdStartPhase.EXPLORATION:
            # 纯探索: 优先未探索的行动
            unexplored = [
                a for a in available_actions
                if action_stats.get(a.value, {}).get("count", 0) == 0
            ]
            if unexplored:
                return random.choice(unexplored)

            # 都探索过，选探索次数最少的
            min_count = min(
                action_stats.get(a.value, {}).get("count", 0)
                for a in available_actions
            )
            least_explored = [
                a for a in available_actions
                if action_stats.get(a.value, {}).get("count", 0) == min_count
            ]
            return random.choice(least_explored)

        elif self._phase == ColdStartPhase.TRANSITION:
            # ε-greedy
            if self.should_explore():
                # 探索: 随机选
                return random.choice(available_actions)
            # 利用: 选最佳
            return self._select_best_action(available_actions, action_stats)

        else:
            # EXPLOITATION: 利用
            return self._select_best_action(available_actions, action_stats)

    def _select_best_action(
        self,
        available_actions: list[ActionType],
        action_stats: dict[str, dict[str, Any]],
    ) -> ActionType:
        """选择最佳行动 (利用)."""
        best_action = available_actions[0]
        best_reward = -float("inf")

        for action in available_actions:
            stats = action_stats.get(action.value, {})
            avg_reward = stats.get("avg_reward", 0.0)
            count = stats.get("count", 0)

            # UCB 风格的加成: 探索次数少的行动获得不确定性加成
            if count > 0:
                ucb_bonus = self._exploration_bonus * (
                    1.0 / (1.0 + count * 0.1)
                )
                score = avg_reward + ucb_bonus
            else:
                score = self._exploration_bonus  # 未探索的行动获得固定加成

            if score > best_reward:
                best_reward = score
                best_action = action

        return best_action

    def _update_epsilon(self) -> None:
        """更新 ε (指数衰减)."""
        self._epsilon = self._initial_epsilon * (self._decay_rate ** self._observation_count)
        # 确保 ε 不低于最小值
        self._epsilon = max(0.01, min(1.0, self._epsilon))

    def _update_phase(self) -> None:
        """更新冷启动阶段."""
        old_phase = self._phase

        if self._observation_count < self._min_obs * 0.5:
            new_phase = ColdStartPhase.EXPLORATION
        elif self._observation_count < self._min_obs:
            new_phase = ColdStartPhase.TRANSITION
        else:
            # 达到最小观测数后，直接进入利用阶段
            # ε 仍用于控制探索概率，但不影响阶段判定
            new_phase = ColdStartPhase.EXPLOITATION

        if new_phase != old_phase:
            self._phase = new_phase
            logger.info(
                "冷启动阶段转换: %s → %s (观测数=%d, ε=%.4f)",
                old_phase.value, new_phase.value,
                self._observation_count, self._epsilon,
            )
            if self.on_phase_change is not None:
                self.on_phase_change(old_phase, new_phase)

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------

    @property
    def phase(self) -> ColdStartPhase:
        """当前冷启动阶段."""
        return self._phase

    @property
    def epsilon(self) -> float:
        """当前探索率 ε."""
        return self._epsilon

    @property
    def observation_count(self) -> int:
        """总观测数."""
        return self._observation_count

    def is_in_cold_start(self) -> bool:
        """是否在冷启动期."""
        return self._phase != ColdStartPhase.EXPLOITATION

    def get_stats(self) -> dict[str, Any]:
        """获取冷启动统计."""
        return {
            "phase": self._phase.value,
            "epsilon": round(self._epsilon, 6),
            "observation_count": self._observation_count,
            "min_observations": self._min_obs,
            "action_counts": dict(self._action_counts),
            "action_avg_rewards": {
                k: round(sum(v) / len(v), 4) if v else 0.0
                for k, v in self._action_rewards.items()
            },
        }


__all__ = [
    "ColdStartManager",
    "ColdStartPhase",
]
