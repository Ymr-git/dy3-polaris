"""IRT 能力估计引擎 — 贝叶斯后验更新与最大似然估计.

融合世界先进方案:
- 贝叶斯项目反应理论 (Bayesian IRT):
    使用数值积分计算后验分布, 通过 EAP (Expected A Posteriori) 估计能力.
    先验分布采用学习者当前能力估计 N(theta, SE), 似然为 IRT 项目反应函数.
- 最大似然估计 (MLE):
    网格搜索 (estimate_mle) 与 Newton-Raphson 迭代 (estimate_mle_newton_raphson).
    Newton-Raphson 使用步长减半保护, 比 网格搜索更快收敛.
    标准误 SE = 1/sqrt(总信息量).
- Gauss-Hermite EAP (estimate_eap_gauss_hermite):
    使用 Gauss-Hermite 正交节点进行更精确的 EAP 估计.
- catR / mirt R 包: 自适应测试能力估计的标准方法参考.

IRT 模型:
- 2PL: P(theta) = 1 / (1 + exp(-a*(theta - b)))
- 3PL: P(theta) = c + (1-c) / (1 + exp(-a*(theta - b)))
- 4PL: P(theta) = c + (d-c) / (1 + exp(-a*(theta - b)))
- 信息函数 I(theta) 因模型类型而异 (2PL/3PL/4PL)
"""

from __future__ import annotations

import math
from typing import Any

from dy3_polaris.l1.models import IRTAbility, IRTItem


class IRTEstimator:
    """IRT 能力估计器.

    提供两种能力估计方法:
    1. update_theta: 贝叶斯后验更新 (单题在线更新, EAP 估计)
    2. estimate_mle: 最大似然估计 (批量离线估计, 网格搜索)
    """

    # theta 网格参数
    _THETA_MIN: float = -3.0
    _THETA_MAX: float = 3.0
    _THETA_STEP: float = 0.1  # 贝叶斯积分步长
    _MLE_STEP: float = 0.01   # MLE 网格搜索步长

    # 数值稳定性下限
    _EPS: float = 1e-10

    def update_theta(
        self,
        ability: IRTAbility,
        item: IRTItem,
        correct: bool,
    ) -> IRTAbility:
        """贝叶斯后验更新 — 使用数值积分计算 EAP 和 SE.

        将学习者当前能力估计作为先验 N(theta, SE),
        结合单题作答结果更新后验分布.

        Args:
            ability: 当前能力估计 (theta + standard_error), 作为先验.
            item: IRT 题目参数.
            correct: 是否答对.

        Returns:
            更新后的 IRTAbility (theta=EAP, standard_error=后验标准差).
        """
        # 构建 theta 网格
        n_points = int(
            round((self._THETA_MAX - self._THETA_MIN) / self._THETA_STEP)
        ) + 1
        thetas: list[float] = [
            self._THETA_MIN + i * self._THETA_STEP for i in range(n_points)
        ]

        # 先验分布: N(ability.theta, ability.standard_error)
        mu = ability.theta
        sigma = max(ability.standard_error, self._EPS)
        coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

        # 似然函数: P(theta)^correct * (1-P(theta))^(1-correct)
        # 后验 = 先验 * 似然
        posterior: list[float] = []
        for t in thetas:
            # 先验 (正态密度)
            prior = coeff * math.exp(-0.5 * ((t - mu) / sigma) ** 2)
            # 似然
            p = item.probability(t)
            p = max(self._EPS, min(1.0 - self._EPS, p))
            likelihood = p if correct else (1.0 - p)
            posterior.append(prior * likelihood)

        # 归一化
        total = sum(posterior)
        if total <= 0.0:
            # 数值下溢, 回退到原值
            return IRTAbility(
                user_id=ability.user_id,
                theta=ability.theta,
                standard_error=ability.standard_error,
            )
        posterior = [p / total for p in posterior]

        # EAP (期望后验估计)
        eap = sum(t * p for t, p in zip(thetas, posterior))

        # 后验标准差 (SE)
        variance = sum((t - eap) ** 2 * p for t, p in zip(thetas, posterior))
        se = math.sqrt(max(0.0, variance))

        # 钳制到有效范围
        eap = max(self._THETA_MIN, min(self._THETA_MAX, eap))

        return IRTAbility(
            user_id=ability.user_id,
            theta=eap,
            standard_error=se,
        )

    def estimate_mle(
        self,
        responses: list[tuple[IRTItem, bool]],
        initial_theta: float = 0.0,
    ) -> IRTAbility:
        """最大似然估计 — 网格搜索最大化对数似然.

        适用于批量响应数据, 无先验信息.
        全对返回正 theta, 全错返回负 theta.
        SE = 1 / sqrt(总信息量).

        Args:
            responses: (IRTItem, correct) 列表.
            initial_theta: 初始 theta (用于空响应回退).

        Returns:
            MLE 估计的 IRTAbility.
        """
        if not responses:
            return IRTAbility(
                user_id="mle",
                theta=max(self._THETA_MIN, min(self._THETA_MAX, initial_theta)),
                standard_error=1.0,
            )

        # 网格搜索: 在 [-3, 3] 上寻找最大化对数似然的 theta
        n_points = int(
            round((self._THETA_MAX - self._THETA_MIN) / self._MLE_STEP)
        ) + 1
        thetas: list[float] = [
            self._THETA_MIN + i * self._MLE_STEP for i in range(n_points)
        ]

        best_theta = max(
            self._THETA_MIN, min(self._THETA_MAX, initial_theta)
        )
        best_ll = float("-inf")

        for theta in thetas:
            ll = 0.0
            for item, correct in responses:
                p = item.probability(theta)
                p = max(self._EPS, min(1.0 - self._EPS, p))
                if correct:
                    ll += math.log(p)
                else:
                    ll += math.log(1.0 - p)
            if ll > best_ll:
                best_ll = ll
                best_theta = theta

        # 钳制到有效范围
        best_theta = max(self._THETA_MIN, min(self._THETA_MAX, best_theta))

        # SE = 1 / sqrt(总信息量)
        total_info = sum(
            item.information(best_theta) for item, _ in responses
        )
        if total_info > 0.0:
            se = 1.0 / math.sqrt(total_info)
        else:
            se = 1.0

        return IRTAbility(
            user_id="mle",
            theta=best_theta,
            standard_error=se,
        )

    def estimate_mle_newton_raphson(
        self,
        responses: list[tuple[IRTItem, bool]],
        initial_theta: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> IRTAbility:
        """Newton-Raphson MLE 估计 — 迭代优化对数似然.

        使用 Newton-Raphson 迭代: theta_{n+1} = theta_n + score / info.
        - score (一阶导数) = sum(dP/dtheta * (u - P) / (P * (1 - P)))
        - info (Fisher 信息量) = sum(item.information(theta))
        - 步长减半保护: 若对数似然不改善, 将步长减半.

        Args:
            responses: (IRTItem, correct) 列表.
            initial_theta: 初始 theta.
            max_iter: 最大迭代次数.
            tol: 收敛阈值 (|Δtheta| < tol).

        Returns:
            MLE 估计的 IRTAbility.
        """
        if not responses:
            return IRTAbility(
                user_id="mle-nr",
                theta=max(self._THETA_MIN, min(self._THETA_MAX, initial_theta)),
                standard_error=1.0,
            )

        theta = max(self._THETA_MIN, min(self._THETA_MAX, initial_theta))

        def log_likelihood(theta_val: float) -> float:
            ll = 0.0
            for item, correct in responses:
                p = item.probability(theta_val)
                p = max(self._EPS, min(1.0 - self._EPS, p))
                if correct:
                    ll += math.log(p)
                else:
                    ll += math.log(1.0 - p)
            return ll

        prev_ll = log_likelihood(theta)

        for _ in range(max_iter):
            # 计算 score (一阶导数) 和 info (Fisher 信息量)
            score = 0.0
            info = 0.0
            for item, correct in responses:
                p = item.probability(theta)
                p = max(self._EPS, min(1.0 - self._EPS, p))
                u = 1.0 if correct else 0.0
                a = item.discrimination_a
                c = item.guessing_c
                d = getattr(item, "upper_d", 1.0)

                # dP/dtheta = a * (P - c) * (d - P) / (d - c)
                # (通用公式: 对 2PL c=0/d=1, 3PL c>0/d=1, 4PL 均适用)
                dc = d - c
                if abs(dc) < self._EPS:
                    dp_dtheta = a * p * (1.0 - p)
                else:
                    dp_dtheta = a * (p - c) * (d - p) / dc

                # score += dp/dtheta * (u - P) / (P * (1 - P))
                score += dp_dtheta * (u - p) / (p * (1.0 - p))

                # info += Fisher 信息量 (使用 item.information)
                info += item.information(theta)

            if info < self._EPS:
                break

            step = score / info
            new_theta = theta + step
            new_theta = max(self._THETA_MIN, min(self._THETA_MAX, new_theta))

            new_ll = log_likelihood(new_theta)

            # 步长减半保护: 若对数似然不改善, 将步长减半
            halving = 0
            while new_ll < prev_ll - 1e-12 and halving < 20:
                step *= 0.5
                new_theta = theta + step
                new_theta = max(self._THETA_MIN, min(self._THETA_MAX, new_theta))
                new_ll = log_likelihood(new_theta)
                halving += 1

            if abs(new_theta - theta) < tol:
                theta = new_theta
                break

            theta = new_theta
            prev_ll = new_ll

        # SE = 1 / sqrt(总信息量)
        total_info = sum(item.information(theta) for item, _ in responses)
        if total_info > 0.0:
            se = 1.0 / math.sqrt(total_info)
        else:
            se = 1.0

        return IRTAbility(
            user_id="mle-nr",
            theta=theta,
            standard_error=se,
        )

    def estimate_eap_gauss_hermite(
        self,
        responses: list[tuple[IRTItem, bool]],
        n_quad: int = 20,
    ) -> IRTAbility:
        """Gauss-Hermite EAP 估计 — 使用 Gauss-Hermite 正交节点.

        使用 Gauss-Hermite 正交近似后验积分:
        EAP = ∫ θ * L(θ) * φ(θ) dθ / ∫ L(θ) * φ(θ) dθ

        其中 φ(θ) 为 N(0, 1) 先验, L(θ) 为似然函数.
        通过 θ = √2 * x 变换, 利用 Gauss-Hermite 节点计算.

        Args:
            responses: (IRTItem, correct) 列表.
            n_quad: 正交节点数 (默认 20).

        Returns:
            EAP 估计的 IRTAbility.
        """
        if not responses:
            return IRTAbility(
                user_id="eap-gh",
                theta=0.0,
                standard_error=1.0,
            )

        # 获取 Gauss-Hermite 节点和权重
        try:
            import numpy as np
            nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
            nodes = [float(x) for x in nodes]
            weights = [float(x) for x in weights]
        except ImportError:
            # 回退: 均匀网格 [-3, 3]
            step = (self._THETA_MAX - self._THETA_MIN) / max(1, n_quad - 1)
            nodes = [self._THETA_MIN + i * step for i in range(n_quad)]
            weights = [1.0 / n_quad] * n_quad

        # 映射到 theta 空间: θ = √2 * x
        # (Gauss-Hermite 近似 ∫ f(x) exp(-x²) dx, 对 N(0,1) 先验需 √2 变换)
        sqrt2 = math.sqrt(2.0)
        thetas = [sqrt2 * x for x in nodes]

        # 计算每个 theta 处的对数似然
        log_likes: list[float] = []
        for theta in thetas:
            ll = 0.0
            for item, correct in responses:
                p = item.probability(theta)
                p = max(self._EPS, min(1.0 - self._EPS, p))
                if correct:
                    ll += math.log(p)
                else:
                    ll += math.log(1.0 - p)
            log_likes.append(ll)

        # 数值稳定: 减去最大值
        max_ll = max(log_likes)
        likes = [math.exp(ll - max_ll) for ll in log_likes]

        # EAP = Σ(w_i * θ_i * L(θ_i)) / Σ(w_i * L(θ_i))
        numerator = sum(w * t * l for w, t, l in zip(weights, thetas, likes))
        denominator = sum(w * l for w, l in zip(weights, likes))

        if denominator <= 0.0:
            return IRTAbility(
                user_id="eap-gh",
                theta=0.0,
                standard_error=1.0,
            )

        eap = numerator / denominator
        eap = max(self._THETA_MIN, min(self._THETA_MAX, eap))

        # SE = sqrt(Σ(w_i * θ_i² * L(θ_i)) / Σ(w_i * L(θ_i)) - EAP²)
        numerator_var = sum(w * t * t * l for w, t, l in zip(weights, thetas, likes))
        variance = numerator_var / denominator - eap ** 2
        se = math.sqrt(max(0.0, variance))

        if se <= 0.0:
            se = 1.0

        return IRTAbility(
            user_id="eap-gh",
            theta=eap,
            standard_error=se,
        )
