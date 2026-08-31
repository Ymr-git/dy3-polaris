"""FSRS-6 在线参数优化器 — 从复习日志学习最优间隔重复参数.

融合世界先进方案:
- FSRS-6 优化器 (open-spaced-repetition): 最大似然估计 + 有限差分梯度下降.
- FSRS-4 全局参数集: 21 参数权重 (w0-w20) 的全局最优与个性化.
- 在线学习: Follow-the-Regularized-Leader (FTRL) 遗憾最小化, 逐题在线更新.

核心思路:
1. 复习日志 (FSRSReviewLog / dict) 携带评分 (grade) 与自上次复习的间隔
   (elapsed_days). 优化器按卡片 (kc_id) 分组并按时间顺序重放, 使用
   FSRSScheduler 演化记忆稳定性 (stability), 在每次复习前计算预测可提取性 R.
2. 损失函数 (负对数似然, 伯努利模型):
       L = -(1/n) * Σ [ actual * log(R) + (1-actual) * log(1-R) ]
   其中 actual = 1 (grade >= 2, 回忆成功) / 0 (grade == 1, 遗忘).
   首次复习 (NEW 状态无先验稳定性) 不计入损失.
3. 梯度: 中心差分有限差分 ∂L/∂w_i = (L(w+ε) - L(w-ε)) / (2ε).
   重放机制使全部 21 个权重 (含 w20 衰减参数) 均参与损失计算, 梯度有意义.
4. 优化: 梯度下降 w_i -= lr * ∂L/∂w_i, 带权重钳制 (投影梯度) 与最优参数
   追踪 (返回迭代过程中损失最低的参数, 保证非增).

依赖: 仅使用标准库 (math), 不依赖 numpy. 复用 FSRSScheduler 进行稳定性演化,
保证优化目标与调度引擎使用同一套 FSRS-6 公式语义.
"""

from __future__ import annotations

import math
from typing import Any

from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler
from dy3_polaris.l1.models import (
    FSRSCardState,
    FSRSParameters,
    MS_PER_SEC,
)


# ============================================================
# 1. 常量定义
# ============================================================

# 毫秒/天 (与 FSRSScheduler._MS_PER_DAY 对齐)
_MS_PER_DAY: float = float(MS_PER_SEC * 86400)

# 数值稳定性下限 (避免 log(0) / 除零)
_EPS: float = 1e-10

# recommend_params 冷启动所需的最小历史记录数
_MIN_HISTORY: int = 5

# 权重钳制边界 (投影梯度下降的可行域, 防止数值发散)
# w20 (衰减参数): 0.01 <= w20 <= 0.5  =>  decay = -w20 ∈ [-0.5, -0.01]
# w5  (初始难度斜率): 0.001 <= w5 <= 5.0  (避免 exp(w5*3) 溢出)
# 其余权重: -10.0 <= w <= 30.0
_W20_MIN: float = 0.01
_W20_MAX: float = 0.5
_W5_MIN: float = 0.001
_W5_MAX: float = 5.0
_W_MIN: float = -10.0
_W_MAX: float = 30.0


# ============================================================
# 2. FSRSOptimizer 在线参数优化器
# ============================================================


class FSRSOptimizer:
    """FSRS-6 在线参数优化器 — 从复习日志学习最优 FSRS 参数.

    通过重放复习日志、最大化对数似然 (最小化负对数似然损失) 来优化 FSRS-6
    的 21 个权重参数. 使用中心差分有限差分计算梯度, 梯度下降迭代更新,
    并追踪迭代过程中的最优参数 (返回损失最低者).

    Args:
        learning_rate: 梯度下降学习率, 默认 0.01.
        max_iter: 最大迭代次数, 默认 50.
        tol: 收敛阈值 — 单次迭代权重最大变化 < tol 时提前停止, 默认 1e-5.

    属性:
        learning_rate / max_iter / tol: 同构造参数 (公开, 便于检视).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> None:
        """初始化 FSRS 参数优化器.

        Args:
            learning_rate: 梯度下降步长.
            max_iter: 最大迭代轮数.
            tol: 收敛阈值 (权重最大变化量).
        """
        self.learning_rate: float = float(learning_rate)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        # 复用无状态调度器进行稳定性演化 (重放)
        self._scheduler: FSRSScheduler = FSRSScheduler()

    # --- 核心优化 ---

    def optimize(
        self,
        review_logs: list,
        initial_params: FSRSParameters | None = None,
    ) -> FSRSParameters:
        """从复习日志学习最优 FSRS 参数.

        流程:
        1. 以 initial_params (默认全局参数) 为起点.
        2. 每轮迭代: 中心差分计算梯度, 梯度下降更新权重, 钳制到可行域.
        3. 追踪迭代过程中损失最低的参数 (保证 optimized_loss <= initial_loss).
        4. 收敛 (权重最大变化 < tol) 或达 max_iter 时停止.

        Args:
            review_logs: 复习日志列表 (FSRSReviewLog 或 dict, 见 _extract_log).
            initial_params: 初始参数; None 时使用默认 FSRSParameters.

        Returns:
            优化后的 FSRSParameters (迭代过程中损失最低者).
        """
        # 起点
        if initial_params is not None:
            params = FSRSParameters(
                weights=list(initial_params.weights),
                request_retention=initial_params.request_retention,
                maximum_interval=initial_params.maximum_interval,
            )
        else:
            params = FSRSParameters()

        # 无数据直接返回起点
        if not review_logs:
            return params

        best_params = FSRSParameters(
            weights=list(params.weights),
            request_retention=params.request_retention,
            maximum_interval=params.maximum_interval,
        )
        best_loss = self.compute_loss(review_logs, params)

        for _ in range(self.max_iter):
            grad = self.compute_gradient(review_logs, params)
            old_weights = list(params.weights)
            new_weights = [
                w - self.learning_rate * g
                for w, g in zip(old_weights, grad)
            ]
            new_weights = self._clamp_weights(new_weights)
            params = FSRSParameters(
                weights=new_weights,
                request_retention=params.request_retention,
                maximum_interval=params.maximum_interval,
            )
            loss = self.compute_loss(review_logs, params)
            if loss < best_loss:
                best_loss = loss
                best_params = FSRSParameters(
                    weights=list(params.weights),
                    request_retention=params.request_retention,
                    maximum_interval=params.maximum_interval,
                )
            # 收敛检测: 权重最大变化
            max_change = max(
                (abs(nw - ow) for nw, ow in zip(new_weights, old_weights)),
                default=0.0,
            )
            if max_change < self.tol:
                break

        return best_params

    def compute_loss(
        self,
        review_logs: list,
        params: FSRSParameters,
    ) -> float:
        """计算当前参数下的平均负对数似然损失.

        L = -(1/n) * Σ [ actual * log(R) + (1-actual) * log(1-R) ]

        其中 R 为复习前的预测可提取性 (由重放得到的稳定性 + elapsed_days +
        params.decay/factor 计算), actual = 1 (grade>=2) / 0 (grade==1).
        首次复习 (NEW 状态) 不计入.

        Args:
            review_logs: 复习日志列表.
            params: FSRS 参数.

        Returns:
            平均负对数似然损失 (>= 0); 无有效样本返回 0.0.
        """
        predictions = self._compute_predictions(review_logs, params)
        if not predictions:
            return 0.0
        n = len(predictions)
        total = 0.0
        for r, actual in predictions:
            r = max(_EPS, min(1.0 - _EPS, r))
            if actual >= 0.5:
                total += -math.log(r)
            else:
                total += -math.log(1.0 - r)
        return total / n

    def compute_gradient(
        self,
        review_logs: list,
        params: FSRSParameters,
        eps: float = 1e-4,
    ) -> list[float]:
        """有限差分梯度计算 (中心差分).

        ∂L/∂w_i ≈ (L(w + ε·e_i) - L(w - ε·e_i)) / (2ε)

        重放机制使全部 21 个权重均参与损失, 故每个权重都有有意义的梯度
        (未被子序列触及的权重梯度为 0).

        Args:
            review_logs: 复习日志列表.
            params: 当前 FSRS 参数.
            eps: 有限差分步长, 默认 1e-4.

        Returns:
            长度等于 len(params.weights) 的梯度列表.
        """
        n = len(params.weights)
        grad: list[float] = [0.0] * n
        if not review_logs:
            return grad
        base_weights = list(params.weights)
        rr = params.request_retention
        mi = params.maximum_interval
        for i in range(n):
            w_plus = list(base_weights)
            w_minus = list(base_weights)
            w_plus[i] += eps
            w_minus[i] -= eps
            loss_plus = self.compute_loss(
                review_logs, FSRSParameters(weights=w_plus, request_retention=rr, maximum_interval=mi)
            )
            loss_minus = self.compute_loss(
                review_logs, FSRSParameters(weights=w_minus, request_retention=rr, maximum_interval=mi)
            )
            grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
        return grad

    def evaluate_params(
        self,
        review_logs: list,
        params: FSRSParameters,
    ) -> dict[str, float]:
        """评估参数质量.

        Args:
            review_logs: 复习日志列表.
            params: 待评估的 FSRS 参数.

        Returns:
            {"log_likelihood": 总对数似然 (<=0, 越大越好),
             "rmse": 预测保留率与实际正确率的均方根误差,
             "mae": 平均绝对误差};
            无有效样本时各项为 0.0.
        """
        predictions = self._compute_predictions(review_logs, params)
        if not predictions:
            return {"log_likelihood": 0.0, "rmse": 0.0, "mae": 0.0}
        n = len(predictions)
        ll = 0.0
        se = 0.0
        ae = 0.0
        for r, actual in predictions:
            r_c = max(_EPS, min(1.0 - _EPS, r))
            ll += actual * math.log(r_c) + (1.0 - actual) * math.log(1.0 - r_c)
            se += (r - actual) ** 2
            ae += abs(r - actual)
        return {
            "log_likelihood": ll,
            "rmse": math.sqrt(se / n),
            "mae": ae / n,
        }

    def recommend_params(
        self,
        learner_history: list[dict],
        cold_start: bool = False,
    ) -> FSRSParameters:
        """根据学习者历史推荐 FSRS 参数.

        冷启动 (cold_start=True) 或历史记录不足时返回默认全局参数;
        有足够数据时调用 optimize 个性化参数.

        Args:
            learner_history: 学习者复习历史 (dict 列表), 每项含
                kc_id / grade / elapsed_days / reviewed_at (可选) /
                state_before (可选).
            cold_start: 是否强制冷启动.

        Returns:
            推荐的 FSRSParameters.
        """
        if cold_start or not learner_history or len(learner_history) < _MIN_HISTORY:
            return FSRSParameters()
        review_logs: list[dict] = []
        for h in learner_history:
            review_logs.append(
                {
                    "kc_id": h.get("kc_id", "default"),
                    "grade": int(h.get("grade", 3)),
                    "elapsed_days": float(h.get("elapsed_days", 0.0)),
                    "reviewed_at": int(h.get("reviewed_at", 0)),
                    "state_before": h.get("state_before", h.get("state", "new")),
                }
            )
        return self.optimize(review_logs)

    # --- 内部辅助 ---

    @staticmethod
    def _extract_log(log: Any) -> tuple[str, int, float, int, str]:
        """从复习日志提取 (kc_id, grade, elapsed_days, reviewed_at, state_before).

        兼容 dict 与 FSRSReviewLog (及任意带同名属性的对象).

        Args:
            log: 复习日志 (dict 或对象).

        Returns:
            (kc_id, grade, elapsed_days, reviewed_at, state_before) 五元组.
        """
        if isinstance(log, dict):
            kc_id = str(log.get("kc_id", "default"))
            grade = int(log.get("grade", 3))
            elapsed = float(log.get("elapsed_days", 0.0))
            reviewed_at = int(log.get("reviewed_at", 0))
            state_before = str(log.get("state_before", log.get("state", "new")))
        else:
            kc_id = str(getattr(log, "kc_id", "default"))
            grade = int(getattr(log, "grade", 3))
            elapsed = float(getattr(log, "elapsed_days", 0.0))
            reviewed_at = int(getattr(log, "reviewed_at", 0))
            state_before = str(getattr(log, "state_before", "new"))
        return kc_id, grade, elapsed, reviewed_at, state_before

    def _group_logs(self, review_logs: list) -> dict[str, list[Any]]:
        """按 kc_id 分组并按 reviewed_at 排序复习日志.

        Args:
            review_logs: 复习日志列表.

        Returns:
            {kc_id: [log, ...]} (每组按 reviewed_at 升序, reviewed_at 相同保持原序).
        """
        grouped: dict[str, list[Any]] = {}
        for log in review_logs:
            kc_id, _, _, reviewed_at, _ = self._extract_log(log)
            grouped.setdefault(kc_id, []).append(log)
        for kc_id in grouped:
            grouped[kc_id].sort(
                key=lambda lg: self._extract_log(lg)[3]  # reviewed_at
            )
        return grouped

    def _compute_predictions(
        self,
        review_logs: list,
        params: FSRSParameters,
    ) -> list[tuple[float, float]]:
        """重放复习日志, 生成 (预测保留率 R, 实际结果 actual) 序列.

        对每张卡片从 NEW 状态开始, 按 reviewed_at 顺序重放:
        - 复习前若卡片已有稳定性 (非 NEW), 计算预测保留率 R 并记录 (R, actual).
        - 通过 FSRSScheduler.schedule_review 演化卡片稳定性 (依赖全部权重).
        首次复习 (NEW) 无先验稳定性, 不计入预测.

        Args:
            review_logs: 复习日志列表.
            params: FSRS 参数.

        Returns:
            [(R, actual), ...] 列表 (actual ∈ {0.0, 1.0}).
        """
        if not review_logs:
            return []
        scheduler = self._scheduler
        grouped = self._group_logs(review_logs)
        predictions: list[tuple[float, float]] = []
        decay = params.decay
        factor = params.factor
        for kc_id, logs in grouped.items():
            card = FSRSCardState(kc_id=kc_id)  # NEW, stability=0
            current_ts = 0.0
            for log in logs:
                _, grade, elapsed, _, _ = self._extract_log(log)
                current_ts = current_ts + elapsed * _MS_PER_DAY
                # 仅当卡片已有稳定性 (非 NEW) 时计入预测
                if card.state != FSRSCardState.NEW and card.stability > 0.0:
                    r = card.retrievability(current_ts, decay, factor)
                    actual = 1.0 if grade >= 2 else 0.0
                    predictions.append((r, actual))
                # 演化稳定性 (调度器内部使用全部 FSRS-6 权重)
                new_card, _, _ = scheduler.schedule_review(
                    card, grade, params, int(current_ts)
                )
                card = new_card
        return predictions

    @staticmethod
    def _clamp_weights(weights: list[float]) -> list[float]:
        """将权重钳制到可行域 (投影梯度下降).

        - w20 (衰减): [0.01, 0.5]
        - w5 (初始难度斜率): [0.001, 5.0]
        - 其余: [-10.0, 30.0]

        Args:
            weights: 待钳制的权重列表.

        Returns:
            钳制后的权重列表 (新列表).
        """
        clamped: list[float] = []
        for i, w in enumerate(weights):
            if i == 20:
                w = max(_W20_MIN, min(_W20_MAX, w))
            elif i == 5:
                w = max(_W5_MIN, min(_W5_MAX, w))
            else:
                w = max(_W_MIN, min(_W_MAX, w))
            clamped.append(w)
        return clamped


# ============================================================
# 3. RegretMinimizer 在线遗憾最小化 (FTRL)
# ============================================================


class RegretMinimizer:
    """在线遗憾最小化器 (Follow-the-Regularized-Leader).

    采用 FTRL-with-L2 (自适应每坐标步长, AdaGrad 风格) 在线学习参数向量.
    将奖励建模为参数的线性函数 r ≈ <w, a> (a = chosen_params), 以平方损失
    的梯度 g = (pred - reward) * a 在线更新, FTRL 闭式解:

        w_i = -alpha * G_i / (1 + sqrt(G2_i))

    其中 G_i = Σ g_i (累计梯度), G2_i = Σ g_i^2 (累计梯度平方).

    遗憾定义为累计差距: 每轮 regret_t = max_reward_so_far - observed_reward_t
    (相对历史最优奖励的累计差距, 始终非负).

    Args:
        n_params: 参数维度, 默认 21 (对应 FSRS-6 权重数).
    """

    def __init__(self, n_params: int = 21) -> None:
        """初始化遗憾最小化器.

        Args:
            n_params: 参数向量维度.
        """
        self.n_params: int = int(n_params)
        # 当前参数 (FTRL 闭式解)
        self._params: list[float] = [0.0] * self.n_params
        # 累计梯度 G_i
        self._grad_sum: list[float] = [0.0] * self.n_params
        # 累计梯度平方 G2_i (自适应步长)
        self._grad_sq_sum: list[float] = [0.0] * self.n_params
        # FTRL 正则化强度 (L2)
        self._alpha: float = 1.0
        # 累计遗憾
        self._regret: float = 0.0
        # 历史最优观测奖励 (用于遗憾计算)
        self._max_reward: float = float("-inf")
        # 累计奖励 (信息性)
        self._cumulative_reward: float = 0.0
        # 轮次
        self._t: int = 0

    def update(
        self,
        chosen_params: list[float],
        observed_reward: float,
    ) -> None:
        """在线遗憾最小化更新 (Follow-the-Regularized-Leader).

        以线性模型预测奖励 pred = <w, chosen_params>, 计算平方损失梯度
        g_i = (pred - reward) * chosen_params_i, 累计梯度后用 FTRL 闭式解
        更新参数, 并累加遗憾.

        Args:
            chosen_params: 本轮选取的参数向量 (动作).
            observed_reward: 本轮观测到的标量奖励.
        """
        # 对齐维度 (不足补 0, 超出截断)
        a = list(chosen_params[: self.n_params])
        if len(a) < self.n_params:
            a.extend([0.0] * (self.n_params - len(a)))

        pred = 0.0
        for w, x in zip(self._params, a):
            pred += w * x
        error = pred - observed_reward  # ∂(0.5*(pred-reward)^2)/∂w = error * a

        for i in range(self.n_params):
            g_i = error * a[i]
            self._grad_sum[i] += g_i
            self._grad_sq_sum[i] += g_i * g_i
            # FTRL-with-L2 闭式解 (自适应每坐标步长)
            denom = 1.0 + math.sqrt(self._grad_sq_sum[i])
            self._params[i] = -self._alpha * self._grad_sum[i] / denom

        # 遗憾累加 (相对历史最优奖励的差距, 非负)
        if observed_reward > self._max_reward:
            self._max_reward = observed_reward
        self._regret += self._max_reward - observed_reward

        self._cumulative_reward += observed_reward
        self._t += 1

    def get_params(self) -> list[float]:
        """返回当前最优参数 (FTRL 闭式解).

        Returns:
            长度为 n_params 的参数向量副本.
        """
        return list(self._params)

    def get_regret(self) -> float:
        """返回累积遗憾.

        Returns:
            累积遗憾 (>= 0).
        """
        return self._regret


# ============================================================
# __all__
# ============================================================

__all__ = [
    "FSRSOptimizer",
    "RegretMinimizer",
]
