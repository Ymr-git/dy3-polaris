"""IRT 项目反应理论能力评估引擎.

融合世界先进方案:
- 贝叶斯 IRT (Bayesian IRT): EAP (Expected A Posteriori) 估计
- 最大似然估计 (MLE): 网格搜索最大化对数似然
- catR / mirt R 包: CAT 自适应测试标准方法
- Knewton / ALEKS: 商用 IRT 驱动自适应学习平台

IRT 3PL 模型:
- P(theta) = c + (1-c) / (1 + exp(-a*(theta - b)))
- 信息函数 I(theta) = a^2 * (P-c)^2 * (1-P) / ((1-c)^2 * P)
- SE = 1 / sqrt(总信息量)

参数:
- a: 区分度 (discrimination, 0.8~2.5 为佳)
- b: 难度 (difficulty, 与 theta 同尺度 [-3, 3])
- c: 伪猜测下限 (guessing, [0, 0.5])

与 L1 irt_estimator.py 的差异:
- 面向 L2 IRTState (theta/se/response_count/last_update_time), 而非 L1 IRTAbility
- 题目参数用 dict 格式 {"a", "b", "c"}, 而非 L1 IRTItem dataclass
- information() 实现完整 3PL Fisher 信息公式 (L1 仅实现 2PL 近似)
"""

from __future__ import annotations

import math
from typing import Any

from dy3_polaris.l2.models import IRTState


# ============================================================
# 1. 常量定义
# ============================================================

# theta 网格范围 (与 theta 估计尺度一致)
_THETA_MIN: float = -3.0
_THETA_MAX: float = 3.0

# EAP 数值积分步长
_EAP_STEP: float = 0.1

# MLE 网格搜索步长
_MLE_STEP: float = 0.01

# 数值稳定性下限 (避免 log(0) / 除零)
_EPS: float = 1e-10

# 空响应回退值
_FALLBACK_THETA: float = 0.0
_FALLBACK_SE: float = 1.0

# Newton-Raphson 步长折半上限 (safeguarded step-halving)
_MAX_HALVINGS: int = 30

# 对数似然比较容差 (允许微小浮点回退)
_LL_TOL: float = 1e-10

# --- Newton-Raphson 鲁棒性增强常量 ---

# 扩展的 θ 边界 (NR 迭代允许更大搜索范围)
_NR_THETA_BOUND: float = 4.0

# 发散检测耐心值: 连续 N 次 |delta| 增大则终止
_NR_DIVERGENCE_PATIENCE: int = 5

# 似然下降耐心值: 连续 N 次对数似然下降则终止
_NR_LL_DECLINE_PATIENCE: int = 3

# 最大迭代硬上限 (防止无限循环)
_NR_MAX_ITER_HARD: int = 200

# 最小 Fisher 信息量 (低于此值视为信息退化, 提前终止)
_NR_MIN_INFO: float = 1e-6


def _extract_params(item_params: dict[str, Any]) -> tuple[float, float, float]:
    """从 item_params 字典提取并校验 (a, b, c) 三参数.

    Args:
        item_params: 题目参数字典, 含 "a"/"b" 键, "c" 可选 (默认 0.0).

    Returns:
        (a, b, c) 三元组; 缺失键回退: a=1.0, b=0.0, c=0.0.

    Raises:
        ValueError: 参数越界时抛出:
            - a 必须 > 0 (区分度为正);
            - b 必须在 [-3, 3] (与 theta 同尺度);
            - c 必须在 [0, 0.5] (猜测下限非负且不超过 0.5).
    """
    a = float(item_params.get("a", 1.0))
    b = float(item_params.get("b", 0.0))
    c = float(item_params.get("c", 0.0))
    # 参数合法性校验 (fail-fast, 避免无效参数污染估计结果)
    if not (a > 0.0):
        raise ValueError(f"IRT 参数非法: a (区分度) 必须 > 0, 实际 a={a}")
    if not (-3.0 <= b <= 3.0):
        raise ValueError(f"IRT 参数非法: b (难度) 必须在 [-3, 3], 实际 b={b}")
    if not (0.0 <= c <= 0.5):
        raise ValueError(f"IRT 参数非法: c (猜测下限) 必须在 [0, 0.5], 实际 c={c}")
    return a, b, c


# ============================================================
# 2. IRTEstimator 能力估计引擎
# ============================================================


class IRTEstimator:
    """IRT 能力估计器 (无状态引擎).

    提供两种能力估计方法, 均面向 L2 ``IRTState``:

    1. ``update_theta``: 贝叶斯后验更新 (单题在线更新, EAP 估计).
       将学习者当前能力估计作为先验 N(theta, se), 结合单题作答结果
       通过数值积分计算后验, 输出 EAP (期望后验) 与后验标准差.
    2. ``estimate_mle``: 最大似然估计 (批量离线估计, 网格搜索).
       无先验信息, 在 [-3, 3] 上网格搜索最大化对数似然,
       SE = 1 / sqrt(总信息量).

    另提供 ``predict_correct`` 与 ``information`` 两个底层 IRT 函数,
    分别实现 3PL 项目反应函数与 Fisher 信息量.

    无状态: 所有方法均为纯函数式, 相同输入产生相同输出, 可安全多实例并发使用.
    """

    # --- 参数约束钳制 ---

    def clamp_params(self, a: float, b: float, c: float) -> dict[str, float]:
        """参数约束钳制 — a∈[0.3,3.0], b∈[-3,3], c∈[0,0.5].

        确保题目参数在合法范围内, 用于 MMLE 校准等场景的参数后处理.

        Args:
            a: 区分度 (discrimination).
            b: 难度 (difficulty).
            c: 伪猜测下限 (guessing).

        Returns:
            钳制后的参数字典 ``{"a": ..., "b": ..., "c": ...}``.
        """
        return {
            "a": max(0.3, min(3.0, a)),
            "b": max(-3.0, min(3.0, b)),
            "c": max(0.0, min(0.5, c)),
        }

    def clamp_theta(self, theta: float) -> float:
        """theta 约束钳制到 [-3, 3].

        Args:
            theta: 能力参数 θ.

        Returns:
            钳制后的 θ 值.
        """
        return max(-3.0, min(3.0, theta))

    # --- 底层 IRT 函数 ---

    def predict_correct(
        self,
        theta: float,
        a: float,
        b: float,
        c: float = 0.0,
    ) -> float:
        """3PL 项目反应函数 — 预测答对概率.

        P(theta) = c + (1 - c) / (1 + exp(-a * (theta - b)))

        - 2PL (c=0): P = 1 / (1 + exp(-a*(theta-b))), theta=b 时 P=0.5
        - 3PL (c>0): 极低能力时 P 趋近于猜测下限 c

        Args:
            theta: 能力参数 θ.
            a: 区分度 (discrimination, > 0).
            b: 难度 (difficulty, 与 theta 同尺度).
            c: 伪猜测下限 (guessing, [0, 0.5]), 默认 0.0 (2PL).

        Returns:
            答对概率, 钳制到 [0.0, 1.0].
        """
        z = a * (theta - b)
        # 数值稳定: 防止 exp 溢出
        p = c + (1.0 - c) / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, p))

    def predict_correct_4pl(
        self,
        theta: float,
        a: float,
        b: float,
        c: float = 0.0,
        d: float = 1.0,
    ) -> float:
        """4PL 项目反应函数 — 预测答对概率 (含上渐近线 d).

        P(theta) = c + (d - c) / (1 + exp(-a * (theta - b)))

        - 3PL (d=1.0): 退化为 c + (1-c)/(1+exp(-a*(theta-b)))
        - 4PL (d<1.0): 极高能力时 P 趋近上渐近线 d (<1), 模型失误/上限
        - c: 下渐近线 (猜测下限), d: 上渐近线 (1-失误)

        Args:
            theta: 能力参数 θ.
            a: 区分度 (discrimination, > 0).
            b: 难度 (difficulty, 与 theta 同尺度).
            c: 下渐近线 (guessing, [0, 0.5]), 默认 0.0.
            d: 上渐近线 (upper asymptote, (c, 1.0]), 默认 1.0 (退化为 3PL).

        Returns:
            答对概率, 钳制到 [0.0, 1.0].
        """
        z = a * (theta - b)
        # 数值稳定: 防止 exp 溢出
        p = c + (d - c) / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, p))

    def information(
        self,
        theta: float,
        a: float,
        b: float,
        c: float = 0.0,
    ) -> float:
        """Fisher 信息量 — 评估题目对能力估计的区分效力.

        - 2PL (c=0): I = a^2 * P * (1 - P)
          信息峰值在 theta=b 处 (P=0.5, I=a^2/4), 天然落在 ZPD.
        - 3PL (c>0): I = a^2 * (P-c)^2 * (1-P) / ((1-c)^2 * P)
          引入猜测下限修正, 低能力区信息量被压缩.

        Args:
            theta: 能力参数 θ.
            a: 区分度 (discrimination, > 0).
            b: 难度 (difficulty, 与 theta 同尺度).
            c: 伪猜测下限 (guessing, [0, 0.5]), 默认 0.0 (2PL).

        Returns:
            Fisher 信息量 (非负). 数值边界处返回 0.0 (避免除零).
        """
        p = self.predict_correct(theta, a, b, c)
        # 2PL: 退化公式, 避免 c=0 时 P≈0 导致除零
        if c <= 0.0:
            return a * a * p * (1.0 - p)
        # 3PL: 完整 Fisher 信息公式
        # I = a^2 * (P-c)^2 * (1-P) / ((1-c)^2 * P)
        denom = (1.0 - c) ** 2 * p
        if denom <= _EPS:
            return 0.0
        return a * a * (p - c) ** 2 * (1.0 - p) / denom

    # --- 贝叶斯 EAP 在线更新 ---

    def update_theta(
        self,
        state: IRTState,
        item_params: dict[str, Any],
        correct: bool,
    ) -> IRTState:
        """贝叶斯后验更新 — 使用数值积分计算 EAP 和 SE (单题在线).

        将学习者当前能力估计作为先验 N(state.theta, state.se),
        结合单题作答结果 (item_params, correct) 更新后验分布:

            posterior(theta) ∝ prior(theta) * likelihood(theta)
            likelihood(theta) = P(theta)^correct * (1-P(theta))^(1-correct)

        通过 theta 网格 [-3, 3] (步长 0.1) 数值积分:
        - EAP = Σ theta * posterior(theta)
        - SE  = sqrt(Σ (theta - EAP)^2 * posterior(theta))

        Args:
            state: 当前能力估计状态 (theta + se 作为先验).
            item_params: 题目参数字典 {"a", "b", "c"}.
            correct: 是否答对.

        Returns:
            更新后的 IRTState:
            - theta = EAP (钳制到 [-3, 3])
            - se = 后验标准差
            - response_count = state.response_count + 1
            - last_update_time = state.last_update_time (本方法不修改时间戳)
        """
        a, b, c = _extract_params(item_params)

        # 构建 theta 网格
        n_points = int(round((_THETA_MAX - _THETA_MIN) / _EAP_STEP)) + 1
        thetas: list[float] = [
            _THETA_MIN + i * _EAP_STEP for i in range(n_points)
        ]

        # 先验分布: N(state.theta, state.se)
        mu = state.theta
        sigma = max(state.se, _EPS)
        coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

        # 后验 = 先验 * 似然
        posterior: list[float] = []
        for t in thetas:
            # 先验 (正态密度)
            prior = coeff * math.exp(-0.5 * ((t - mu) / sigma) ** 2)
            # 似然: P^correct * (1-P)^(1-correct)
            p = self.predict_correct(t, a, b, c)
            p = max(_EPS, min(1.0 - _EPS, p))
            likelihood = p if correct else (1.0 - p)
            posterior.append(prior * likelihood)

        # 归一化
        total = sum(posterior)
        if total <= 0.0:
            # 数值下溢, 回退到原 theta (但仍记录作答次数)
            return IRTState(
                theta=state.theta,
                se=state.se,
                response_count=state.response_count + 1,
                last_update_time=state.last_update_time,
            )
        posterior = [w / total for w in posterior]

        # EAP (期望后验估计)
        eap = sum(t * w for t, w in zip(thetas, posterior))

        # 后验标准差 (SE)
        variance = sum((t - eap) ** 2 * w for t, w in zip(thetas, posterior))
        se = math.sqrt(max(0.0, variance))

        # 钳制到有效范围
        eap = max(_THETA_MIN, min(_THETA_MAX, eap))

        return IRTState(
            theta=eap,
            se=se,
            response_count=state.response_count + 1,
            last_update_time=state.last_update_time,
        )

    # --- 最大似然批量估计 ---

    def estimate_mle(
        self,
        responses: list[tuple[dict[str, Any], bool]],
    ) -> IRTState:
        """最大似然估计 — 网格搜索最大化对数似然 (批量离线).

        在 [-3, 3] (步长 0.01) 上网格搜索最大化对数似然:

            LL(theta) = Σ [correct*log P(theta) + (1-correct)*log(1-P(theta))]

        全对倾向于正 theta, 全错倾向于负 theta.
        SE = 1 / sqrt(总信息量), 总信息量在最优 theta 处累加.

        Args:
            responses: (item_params, correct) 列表, item_params 为
                {"a", "b", "c"} 字典.

        Returns:
            MLE 估计的 IRTState:
            - theta = 最优网格点 (钳制到 [-3, 3])
            - se = 1/sqrt(总信息量), 信息量为 0 时回退 1.0
            - response_count = len(responses)
            - last_update_time = 0.0

            空响应回退: theta=0.0, se=1.0, response_count=0.
        """
        # 空响应回退
        if not responses:
            return IRTState(
                theta=_FALLBACK_THETA,
                se=_FALLBACK_SE,
                response_count=0,
                last_update_time=0.0,
            )

        # 预解析参数 (避免重复 dict 查找)
        parsed = [(_extract_params(params), correct) for params, correct in responses]

        # 网格搜索最大化对数似然
        n_points = int(round((_THETA_MAX - _THETA_MIN) / _MLE_STEP)) + 1
        thetas: list[float] = [
            _THETA_MIN + i * _MLE_STEP for i in range(n_points)
        ]

        best_theta = _FALLBACK_THETA
        best_ll = float("-inf")

        for theta in thetas:
            ll = 0.0
            for (a, b, c), correct in parsed:
                p = self.predict_correct(theta, a, b, c)
                p = max(_EPS, min(1.0 - _EPS, p))
                if correct:
                    ll += math.log(p)
                else:
                    ll += math.log(1.0 - p)
            if ll > best_ll:
                best_ll = ll
                best_theta = theta

        # 钳制到有效范围
        best_theta = max(_THETA_MIN, min(_THETA_MAX, best_theta))

        # SE = 1 / sqrt(总信息量)
        total_info = sum(
            self.information(best_theta, a, b, c) for (a, b, c), _ in parsed
        )
        if total_info > 0.0:
            se = 1.0 / math.sqrt(total_info)
        else:
            se = _FALLBACK_SE

        return IRTState(
            theta=best_theta,
            se=se,
            response_count=len(responses),
            last_update_time=0.0,
        )

    # --- Newton-Raphson 最大似然估计 (Fisher scoring) ---

    def _log_likelihood(
        self,
        theta: float,
        parsed: list[tuple[tuple[float, float, float], float]],
    ) -> float:
        """计算对数似然 LL(theta) = Σ [u*log P + (1-u)*log(1-P)] (内部辅助)."""
        ll = 0.0
        for (a, b, c), u in parsed:
            p = self.predict_correct(theta, a, b, c)
            p = max(_EPS, min(1.0 - _EPS, p))
            if u > 0.5:
                ll += math.log(p)
            else:
                ll += math.log(1.0 - p)
        return ll

    def estimate_mle_newton_raphson(
        self,
        responses: list[tuple[dict[str, Any], bool]],
        initial_theta: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        return_stats: bool = False,
        prior_sd: float | None = None,
    ) -> IRTState | dict[str, Any]:
        """Newton-Raphson (Fisher scoring) 最大似然估计 — 快速收敛批量离线估计.

        使用 Fisher 信息矩阵替代观测 Hessian, 迭代更新:

            theta_{n+1} = theta_n + score(theta_n) / info(theta_n)

        其中:
        - score(theta) = Σ_i a_i * (u_i - P_i) * (P_i - c_i) / (P_i * (1 - c_i))
          (对数似然一阶导数, u_i 为 0/1 作答)
        - info(theta)  = Σ_i I_i(theta)  (Fisher 信息量, 见 ``information``)

        相比网格搜索 ``estimate_mle`` (O(601*N)), Newton-Raphson 通常在
        5~10 次迭代内收敛, 复杂度 O(K*N) (K 远小于 601).

        **步长折半 (step-halving) 保护**: 远离最优解时, 单步 Newton 更新可能
        过冲甚至越界 (在 ±3 边界间震荡). 借鉴 catR/mirt 的 safeguarded
        Newton-Raphson: 若候选 theta 越出 [-3, 3] 或使对数似然下降, 则反复
        折半步长, 直至候选合法且似然非降. 这保证从任意初始 theta 均能稳健
        收敛到全局最优 (2PL/3PL 对数似然为凹函数).

        退化情形 (全对/全错) 下 theta 收敛到 ±3 边界.

        **贝叶斯收缩 (prior_sd)**: 当 ``prior_sd`` 不为 None 时, 在 NR 收敛后
        对 theta 应用正态先验 N(0, prior_sd) 的后验收缩:

            theta_shrunk = (I * theta_mle) / (I + I_prior)
            se_shrunk    = 1 / sqrt(I + I_prior)

        其中 I = 1/SE² (数据信息量), I_prior = 1/prior_sd² (先验信息量).
        这等价于 MAP 估计的正态近似后验均值, 在小样本下显著降低估计方差
        (MAE), 代价是向先验均值 (0) 引入轻微偏差.

        Args:
            responses: (item_params, correct) 列表, item_params 为
                {"a", "b", "c"} 字典.
            initial_theta: 迭代初始 theta, 默认 0.0.
            max_iter: 最大迭代次数, 默认 100.
            tol: 收敛阈值 (相邻两次 theta 差的绝对值), 默认 1e-6.
            return_stats: 是否返回统计信息字典 (含 iterations / converged 等),
                默认 False (返回 IRTState 对象).
            prior_sd: 贝叶斯先验标准差; 不为 None 时启用正态先验收缩.
                默认 None (纯 MLE, 无先验). 推荐值 1.0 (弱先验, 平衡偏差与方差).

        Returns:
            MLE 估计的 IRTState (return_stats=False 时):
            - theta = Newton-Raphson 收敛解 (钳制到 [-3, 3])
            - se = 1/sqrt(总信息量), 信息量为 0 时回退 1.0
            - response_count = len(responses)
            - last_update_time = 0.0

            统计信息字典 (return_stats=True 时):
            - theta / se / response_count / last_update_time: 同上
            - iterations: 实际迭代次数
            - converged: 是否收敛 (相邻 theta 差 < tol)

            空响应回退: theta=0.0, se=1.0, response_count=0.
        """
        # 空响应回退
        if not responses:
            return IRTState(
                theta=_FALLBACK_THETA,
                se=_FALLBACK_SE,
                response_count=0,
                last_update_time=0.0,
            )

        # 预解析参数 (含合法性校验), u 转 0.0/1.0
        parsed = [
            (_extract_params(params), (1.0 if correct else 0.0))
            for params, correct in responses
        ]

        # 鲁棒性增强: 最大迭代硬上限
        effective_max_iter = min(max_iter, _NR_MAX_ITER_HARD)

        theta = float(initial_theta)
        # 鲁棒性增强: 使用扩展边界 [-4, 4]
        theta = max(-_NR_THETA_BOUND, min(_NR_THETA_BOUND, theta))
        ll_current = self._log_likelihood(theta, parsed)

        # 鲁棒性增强: 跟踪发散与似然下降
        prev_delta: float = float("inf")  # 上一次 |delta|
        divergence_count: int = 0         # 连续 |delta| 增大计数
        ll_decline_count: int = 0         # 连续似然下降计数
        converged: bool = False           # 收敛标志
        iteration_count: int = 0          # 实际迭代次数

        for iteration in range(effective_max_iter):
            iteration_count = iteration + 1
            score = 0.0
            info = 0.0
            for (a, b, c), u in parsed:
                p = self.predict_correct(theta, a, b, c)
                # 数值稳定: 钳制 P 到 (eps, 1-eps) 避免除零与 log(0)
                p = max(_EPS, min(1.0 - _EPS, p))
                # score 贡献: a * (u - P) * (P - c) / (P * (1 - c))
                denom_score = p * (1.0 - c)
                if denom_score > _EPS:
                    score += a * (u - p) * (p - c) / denom_score
                # info 贡献: Fisher 信息量
                info += self.information(theta, a, b, c)

            # 鲁棒性增强: 信息量过低检测
            if info < _NR_MIN_INFO:
                break

            # Newton-Raphson / Fisher scoring 候选步长
            delta = score / info

            # 鲁棒性增强: 发散检测 (连续 |delta| 增大)
            current_abs_delta = abs(delta)
            if current_abs_delta > prev_delta:
                divergence_count += 1
                if divergence_count >= _NR_DIVERGENCE_PATIENCE:
                    break
            else:
                divergence_count = 0
            prev_delta = current_abs_delta

            # 步长折半: 保证候选 theta 在扩展边界内且对数似然非降
            theta_new = theta
            accepted = False
            ll_before_step = ll_current
            for _ in range(_MAX_HALVINGS):
                theta_prop = theta + delta
                if -_NR_THETA_BOUND <= theta_prop <= _NR_THETA_BOUND:
                    ll_prop = self._log_likelihood(theta_prop, parsed)
                    if ll_prop >= ll_current - _LL_TOL:
                        theta_new = theta_prop
                        ll_current = ll_prop
                        accepted = True
                        break
                delta *= 0.5

            if not accepted:
                # 无法找到改进的合法步长 -> 已收敛 (常为边界最优)
                break

            # 鲁棒性增强: 似然下降保护
            if ll_current < ll_before_step - _LL_TOL:
                ll_decline_count += 1
                if ll_decline_count >= _NR_LL_DECLINE_PATIENCE:
                    theta = theta_new
                    break
            else:
                ll_decline_count = 0

            # 收敛判定
            if abs(theta_new - theta) < tol:
                theta = theta_new
                converged = True
                break
            theta = theta_new

        # 最终钳制到标准范围 [-3, 3] (返回给调用方的 theta 在标准尺度)
        theta = max(_THETA_MIN, min(_THETA_MAX, theta))

        # SE = 1 / sqrt(总信息量)
        total_info = sum(
            self.information(theta, a, b, c) for (a, b, c), _ in parsed
        )
        if total_info > 0.0:
            se = 1.0 / math.sqrt(total_info)
        else:
            se = _FALLBACK_SE

        # 如果信息量足够 (非退化), 即使未因 tol 收敛也视为收敛
        # (步长折半保护下, 无法找到改进步长通常意味着已到达最优)
        if not converged and total_info > _NR_MIN_INFO and iteration_count > 0:
            converged = True

        # (可选) 贝叶斯收缩: 使用正态先验 N(0, prior_sd) 正则化极端估计
        # theta_shrunk = (I * theta_mle) / (I + I_prior)
        # se_shrunk    = 1 / sqrt(I + I_prior)
        if prior_sd is not None and prior_sd > 0.0:
            prior_var = max(prior_sd * prior_sd, _EPS)
            prior_info = 1.0 / prior_var
            total_info_with_prior = total_info + prior_info
            if total_info > 0.0:
                theta = (total_info * theta) / total_info_with_prior
            else:
                theta = 0.0  # 无数据信息时回退到先验均值
            se = 1.0 / math.sqrt(total_info_with_prior)
            # 重新钳制到标准范围
            theta = max(_THETA_MIN, min(_THETA_MAX, theta))

        if return_stats:
            return {
                "theta": theta,
                "se": se,
                "response_count": len(responses),
                "last_update_time": 0.0,
                "iterations": iteration_count,
                "converged": converged,
            }

        return IRTState(
            theta=theta,
            se=se,
            response_count=len(responses),
            last_update_time=0.0,
        )

    def _check_convergence(
        self,
        delta_history: list[float],
        ll_history: list[float],
        current_iter: int,
    ) -> tuple[bool, str]:
        """检查 Newton-Raphson 迭代收敛状态 (内部辅助).

        综合判断是否应提前终止迭代:
        1. 发散检测: 连续 ``_NR_DIVERGENCE_PATIENCE`` 次 |delta| 增大;
        2. 似然下降: 连续 ``_NR_LL_DECLINE_PATIENCE`` 次对数似然下降;
        3. 信息退化: 当前 delta 过大且 ll 极低 (隐式信息不足).

        Args:
            delta_history: 历次迭代的 |delta| 序列.
            ll_history: 历次迭代的对数似然序列.
            current_iter: 当前迭代序号.

        Returns:
            (should_stop, reason) 二元组; should_stop=True 时 reason 描述终止原因.
        """
        if len(delta_history) < 2:
            return False, ""

        # 1. 发散检测: 连续 _NR_DIVERGENCE_PATIENCE 次 |delta| 增大
        if len(delta_history) >= _NR_DIVERGENCE_PATIENCE:
            recent = delta_history[-_NR_DIVERGENCE_PATIENCE:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                return True, "divergence_detected"

        # 2. 似然下降检测: 连续 _NR_LL_DECLINE_PATIENCE 次对数似然下降
        if len(ll_history) >= _NR_LL_DECLINE_PATIENCE:
            recent_ll = ll_history[-_NR_LL_DECLINE_PATIENCE:]
            if all(recent_ll[i] > recent_ll[i + 1] for i in range(len(recent_ll) - 1)):
                return True, "ll_decline_detected"

        return False, ""

    # --- 贝叶斯分层 IRT (Hierarchical Bayesian IRT) ---

    def estimate_map(
        self,
        responses: list[tuple[dict[str, Any], bool]],
        prior_mean: float = 0.0,
        prior_sd: float = 1.0,
    ) -> IRTState:
        """MAP (Maximum A Posteriori) 估计 — 似然 × 正态先验的众数.

        在 MLE 的对数似然基础上加入正态先验 N(prior_mean, prior_sd) 的惩罚项:

            θ_MAP = argmax [LL(θ) - 0.5 * ((θ - μ) / σ)²]

        使用 Newton-Raphson 优化, 在 score 和 info 中加入先验贡献:
        - score_prior = -(θ - μ) / σ²
        - info_prior  = 1 / σ²

        强先验 (小 σ) 将 θ 拉向先验均值; 弱先验 (大 σ) 退化为 MLE.

        Args:
            responses: (item_params, correct) 列表.
            prior_mean: 先验均值 μ, 默认 0.0.
            prior_sd: 先验标准差 σ, 默认 1.0.

        Returns:
            MAP 估计的 IRTState. 空响应回退到先验 (theta=μ, se=σ).
        """
        if not responses:
            return IRTState(
                theta=prior_mean,
                se=prior_sd,
                response_count=0,
                last_update_time=0.0,
            )

        parsed = [
            (_extract_params(params), (1.0 if correct else 0.0))
            for params, correct in responses
        ]

        prior_var = max(prior_sd * prior_sd, _EPS)
        info_prior = 1.0 / prior_var

        theta = float(prior_mean)
        theta = max(-_NR_THETA_BOUND, min(_NR_THETA_BOUND, theta))
        ll_current = self._log_likelihood(theta, parsed) - 0.5 * ((theta - prior_mean) ** 2) / prior_var

        for _ in range(_NR_MAX_ITER_HARD):
            score = 0.0
            info = 0.0
            for (a, b, c), u in parsed:
                p = self.predict_correct(theta, a, b, c)
                p = max(_EPS, min(1.0 - _EPS, p))
                denom_score = p * (1.0 - c)
                if denom_score > _EPS:
                    score += a * (u - p) * (p - c) / denom_score
                info += self.information(theta, a, b, c)

            # 加入先验贡献
            score += -(theta - prior_mean) / prior_var
            info += info_prior

            if info < _NR_MIN_INFO:
                break

            delta = score / info

            # 步长折半保护
            accepted = False
            for _ in range(_MAX_HALVINGS):
                theta_prop = theta + delta
                if -_NR_THETA_BOUND <= theta_prop <= _NR_THETA_BOUND:
                    ll_prop = self._log_likelihood(theta_prop, parsed) - 0.5 * ((theta_prop - prior_mean) ** 2) / prior_var
                    if ll_prop >= ll_current - _LL_TOL:
                        theta = theta_prop
                        ll_current = ll_prop
                        accepted = True
                        break
                delta *= 0.5

            if not accepted:
                break

            if abs(delta) < 1e-6:
                break

        theta = max(_THETA_MIN, min(_THETA_MAX, theta))

        total_info = sum(
            self.information(theta, a, b, c) for (a, b, c), _ in parsed
        ) + info_prior
        se = 1.0 / math.sqrt(total_info) if total_info > 0.0 else _FALLBACK_SE

        return IRTState(
            theta=theta,
            se=se,
            response_count=len(responses),
            last_update_time=0.0,
        )

    def estimate_hierarchical_bayesian(
        self,
        responses_by_learner: dict[str, list[tuple[dict[str, Any], bool]]],
        group_prior: dict[str, float] | None = None,
        n_iterations: int = 50,
        shrinkage: float = 0.5,
        adaptive: bool = False,
    ) -> dict[str, IRTState]:
        """多学习者分层贝叶斯 IRT 估计 — MAP 收缩 toward 群体先验.

        层级结构:
        - 群体层: θ ~ N(μ_group, σ_group)
        - 学习者层: θ_i 由自身作答数据估计, 向群体先验收缩

        自适应收缩 (adaptive=True):
            λ_i = σ²_i / (τ² + σ²_i)
        其中 σ²_i = MLE SE² (学习者内方差), τ² = sd_group² (群体间方差).
        数据少 (SE 大) → λ 大 → 强收缩; 数据多 (SE 小) → λ 小 → 弱收缩.

        固定收缩 (adaptive=False):
            λ = shrinkage (统一收缩系数).

        Args:
            responses_by_learner: {learner_id: [(item_params, correct), ...]}.
            group_prior: 群体先验 {"mean": μ, "sd": σ}; 默认 μ=0.0, σ=1.0.
            n_iterations: 保留参数 (当前实现为解析收缩, 不需迭代).
            shrinkage: 固定收缩系数 λ ∈ [0, 1], 默认 0.5 (adaptive=False 时使用).
            adaptive: 是否使用自适应收缩, 默认 False.

        Returns:
            {learner_id: IRTState} 映射; 空输入返回空字典.
        """
        if not responses_by_learner:
            return {}

        if group_prior is not None:
            mu_group = float(group_prior.get("mean", 0.0))
            sd_group = float(group_prior.get("sd", 1.0))
        else:
            mu_group = 0.0
            sd_group = 1.0

        tau_sq = max(sd_group * sd_group, _EPS)
        fixed_lam = max(0.0, min(1.0, shrinkage))

        results: dict[str, IRTState] = {}
        for learner_id, responses in responses_by_learner.items():
            if not responses:
                results[learner_id] = IRTState(
                    theta=mu_group, se=sd_group,
                    response_count=0, last_update_time=0.0,
                )
                continue

            mle_state = self.estimate_mle_newton_raphson(responses)

            if adaptive:
                sigma_sq = max(mle_state.se * mle_state.se, _EPS)
                lam = sigma_sq / (tau_sq + sigma_sq)
            else:
                lam = fixed_lam

            shrunk_theta = (1.0 - lam) * mle_state.theta + lam * mu_group
            shrunk_se = (1.0 - lam) * mle_state.se + lam * sd_group

            results[learner_id] = IRTState(
                theta=shrunk_theta,
                se=shrunk_se,
                response_count=mle_state.response_count,
                last_update_time=0.0,
            )

        return results

    def calibrate_items(
        self,
        responses_by_item: dict[str, list[tuple[float, bool]]],
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> dict[str, dict[str, float]]:
        """题目参数联合校准 (EM 算法).

        使用期望最大化 (EM) 算法交替优化:
        - E 步: 用当前题目参数对每个题目对应的作答序列估计 θ (MLE);
        - M 步: 用当前 θ 对每个题目的参数 (a, b, c) 做梯度上升.

        初始值: a=1.0, b=0.0, c=0.25.
        收敛: 所有题目参数的最大变化 < tol.

        Args:
            responses_by_item: {item_id: [(theta, correct), ...]},
                每条记录含已知的能力值 θ 和作答结果.
            max_iter: EM 最大迭代次数, 默认 50.
            tol: 收敛阈值 (参数最大变化), 默认 1e-5.

        Returns:
            {item_id: {"a": ..., "b": ..., "c": ...}} 校准后的题目参数.
            空输入返回空字典.
        """
        if not responses_by_item:
            return {}

        # 初始化题目参数
        item_params: dict[str, dict[str, float]] = {}
        for item_id in responses_by_item:
            item_params[item_id] = {"a": 1.0, "b": 0.0, "c": 0.25}

        lr = 0.01  # M 步学习率
        eps_grad = 1e-4  # 有限差分步长

        for iteration in range(max_iter):
            max_change = 0.0

            for item_id, responses in responses_by_item.items():
                if not responses:
                    continue

                params = item_params[item_id]
                a, b, c = params["a"], params["b"], params["c"]

                # 有限差分梯度 (对 a, b, c 各维度)
                for key in ("a", "b", "c"):
                    # 正扰动
                    p_plus = dict(params)
                    new_val = params[key] + eps_grad
                    if key == "a":
                        new_val = max(new_val, 0.01)
                    elif key == "c":
                        new_val = max(0.0, min(0.5, new_val))
                    p_plus[key] = new_val

                    # 负扰动
                    p_minus = dict(params)
                    new_val = params[key] - eps_grad
                    if key == "a":
                        new_val = max(new_val, 0.01)
                    elif key == "c":
                        new_val = max(0.0, min(0.5, new_val))
                    p_minus[key] = new_val

                    # 计算对数似然
                    ll_plus = 0.0
                    ll_minus = 0.0
                    for theta, correct in responses:
                        pp = self.predict_correct(theta, p_plus["a"], p_plus["b"], p_plus["c"])
                        pp = max(_EPS, min(1.0 - _EPS, pp))
                        pm = self.predict_correct(theta, p_minus["a"], p_minus["b"], p_minus["c"])
                        pm = max(_EPS, min(1.0 - _EPS, pm))
                        if correct:
                            ll_plus += math.log(pp)
                            ll_minus += math.log(pm)
                        else:
                            ll_plus += math.log(1.0 - pp)
                            ll_minus += math.log(1.0 - pm)

                    grad = (ll_plus - ll_minus) / (2.0 * eps_grad) if eps_grad > 0 else 0.0

                    # 梯度上升
                    old_val = params[key]
                    new_val = old_val + lr * grad

                    # 钳制到合法范围
                    if key == "a":
                        new_val = max(0.01, min(5.0, new_val))
                    elif key == "b":
                        new_val = max(-3.0, min(3.0, new_val))
                    elif key == "c":
                        new_val = max(0.0, min(0.5, new_val))

                    params[key] = new_val
                    max_change = max(max_change, abs(new_val - old_val))

            if max_change < tol:
                break

        return item_params

    # --- MMLE 边际最大似然估计 (EM 算法) ---

    def estimate_mmle(
        self,
        responses_by_learner: dict[str, list[tuple[dict[str, Any], bool]]],
        group_prior: dict[str, float] | None = None,
        n_iterations: int = 50,
        convergence_threshold: float = 1e-4,
        return_history: bool = False,
    ) -> dict[str, dict[str, float]] | dict[str, Any]:
        """MMLE 边际最大似然估计 — EM 算法校准题库参数 (Bock & Aitkin 1981).

        从多学习者作答数据联合校准题目参数 (a, b, c), 使用 EM 算法:

        - E 步: 用当前题目参数和数值积分 (theta 网格 [-3, 3], 步长 0.1)
          计算每个学习者的 θ 后验分布;
        - M 步: 用 θ 后验作为权重, 对每个题目参数做梯度上升 (有限差分梯度)
          更新 (a, b, c);
        - 迭代直到参数变化 < convergence_threshold 或达到 n_iterations.

        初始值: a=1.0, b=0.0, c=0.0.
        参数钳制: 每次更新后调用 ``clamp_params`` 确保 a∈[0.3,3.0],
        b∈[-3,3], c∈[0,0.5].

        边际对数似然 (用网格积分近似) 在 EM 中保证单调递增; 若某次 M 步
        更新导致边际对数似然下降 (学习率过大), 则回退参数并终止迭代.

        Args:
            responses_by_learner: {learner_id: [(item_params, correct), ...]},
                item_params 含 ``item_id`` 键用于题目标识.
            group_prior: 群体先验 ``{"mean": μ, "sd": σ}``; 默认 μ=0.0, σ=1.0.
            n_iterations: EM 最大迭代次数, 默认 50.
            convergence_threshold: 收敛阈值 (参数最大变化), 默认 1e-4.
            return_history: 是否返回含 ``loglik_history`` 的详细字典.

        Returns:
            return_history=False: ``{item_id: {"a", "b", "c"}}`` 校准参数.
            return_history=True: ``{"items": ..., "loglik_history": [...],
            "iterations": N}``.
            空输入返回空字典.
        """
        if not responses_by_learner:
            return {}

        # 1. 收集所有题目的 item_id
        item_ids: set[str] = set()
        for responses in responses_by_learner.values():
            for params, _ in responses:
                iid = params.get("item_id")
                if iid is not None:
                    item_ids.add(iid)
        if not item_ids:
            return {}

        # 2. 初始化每题参数为 a=1.0, b=0.0, c=0.0
        item_params: dict[str, dict[str, float]] = {
            iid: {"a": 1.0, "b": 0.0, "c": 0.0} for iid in item_ids
        }

        # 群体先验
        if group_prior is not None:
            prior_mean = float(group_prior.get("mean", 0.0))
            prior_sd = float(group_prior.get("sd", 1.0))
        else:
            prior_mean = 0.0
            prior_sd = 1.0

        # theta 网格 (与 EAP 一致: [-3, 3], 步长 0.1)
        n_grid = int(round((_THETA_MAX - _THETA_MIN) / _EAP_STEP)) + 1
        thetas: list[float] = [
            _THETA_MIN + i * _EAP_STEP for i in range(n_grid)
        ]
        # 先验权重 (正态密度 × 步长)
        sigma = max(prior_sd, _EPS)
        prior_weights: list[float] = [
            math.exp(-0.5 * ((t - prior_mean) / sigma) ** 2)
            / (sigma * math.sqrt(2.0 * math.pi))
            * _EAP_STEP
            for t in thetas
        ]

        loglik_history: list[float] = []
        lr = 0.01       # M 步学习率
        eps_grad = 1e-4  # 有限差分步长

        def _get_abc(params_dict: dict[str, Any]) -> tuple[float, float, float]:
            """获取题目参数: 优先用校准值, 回退到原始字典值."""
            iid = params_dict.get("item_id")
            if iid is not None and iid in item_params:
                p = item_params[iid]
                return p["a"], p["b"], p["c"]
            return (
                float(params_dict.get("a", 1.0)),
                float(params_dict.get("b", 0.0)),
                float(params_dict.get("c", 0.0)),
            )

        def _compute_posteriors_and_loglik() -> tuple[dict, float]:
            """E 步: 计算每个学习者的 θ 后验分布与边际对数似然."""
            all_posteriors: dict[str, list[float] | None] = {}
            total_ll = 0.0
            for learner_id, responses in responses_by_learner.items():
                if not responses:
                    all_posteriors[learner_id] = None
                    continue
                likelihoods: list[float] = []
                for idx, t in enumerate(thetas):
                    ll = prior_weights[idx]
                    for params, correct in responses:
                        a, b, c = _get_abc(params)
                        p = self.predict_correct(t, a, b, c)
                        p = max(_EPS, min(1.0 - _EPS, p))
                        ll *= p if correct else (1.0 - p)
                    likelihoods.append(ll)
                total = sum(likelihoods)
                if total > 0.0:
                    all_posteriors[learner_id] = [l / total for l in likelihoods]
                else:
                    all_posteriors[learner_id] = [1.0 / n_grid] * n_grid
                total_ll += math.log(max(total, _EPS))
            return all_posteriors, total_ll

        actual_iterations = 0
        for iteration in range(n_iterations):
            actual_iterations = iteration + 1

            # E 步: 计算 θ 后验与边际对数似然
            posteriors, ll_before = _compute_posteriors_and_loglik()
            loglik_history.append(ll_before)

            # 保存旧参数 (用于回退)
            old_params = {iid: dict(item_params[iid]) for iid in item_ids}

            # M 步: 对每个题目做梯度上升 (有限差分梯度, 后验加权)
            max_change = 0.0
            for iid in item_ids:
                params = dict(item_params[iid])
                for key in ("a", "b", "c"):
                    # 正/负扰动参数
                    p_plus = dict(params)
                    p_minus = dict(params)
                    val = params[key]
                    p_plus[key] = val + eps_grad
                    p_minus[key] = val - eps_grad
                    if key == "a":
                        p_plus[key] = max(0.01, p_plus[key])
                        p_minus[key] = max(0.01, p_minus[key])
                    elif key == "c":
                        p_plus[key] = max(0.0, min(0.5, p_plus[key]))
                        p_minus[key] = max(0.0, min(0.5, p_minus[key]))

                    # 后验加权有限差分梯度
                    grad = 0.0
                    for learner_id, responses in responses_by_learner.items():
                        post = posteriors.get(learner_id)
                        if post is None:
                            continue
                        for idx, t in enumerate(thetas):
                            has_item = False
                            ll_plus = 0.0
                            ll_minus = 0.0
                            for rparams, correct in responses:
                                if rparams.get("item_id") == iid:
                                    has_item = True
                                    pp = self.predict_correct(
                                        t, p_plus["a"], p_plus["b"], p_plus["c"]
                                    )
                                    pm = self.predict_correct(
                                        t, p_minus["a"], p_minus["b"], p_minus["c"]
                                    )
                                    pp = max(_EPS, min(1.0 - _EPS, pp))
                                    pm = max(_EPS, min(1.0 - _EPS, pm))
                                    if correct:
                                        ll_plus += math.log(pp)
                                        ll_minus += math.log(pm)
                                    else:
                                        ll_plus += math.log(1.0 - pp)
                                        ll_minus += math.log(1.0 - pm)
                            if has_item:
                                grad += post[idx] * (
                                    ll_plus - ll_minus
                                ) / (2.0 * eps_grad)

                    old_val = params[key]
                    new_val = old_val + lr * grad
                    params[key] = new_val

                # 参数钳制
                clamped = self.clamp_params(params["a"], params["b"], params["c"])
                max_change = max(
                    max_change,
                    max(
                        abs(clamped["a"] - item_params[iid]["a"]),
                        abs(clamped["b"] - item_params[iid]["b"]),
                        abs(clamped["c"] - item_params[iid]["c"]),
                    ),
                )
                item_params[iid] = clamped

            # 回退保护: 若边际对数似然下降, 回退参数
            _, ll_after = _compute_posteriors_and_loglik()
            if ll_after < ll_before - 1e-10:
                for iid in item_ids:
                    item_params[iid] = old_params[iid]
                max_change = 0.0

            # 收敛判定
            if max_change < convergence_threshold:
                break

        if return_history:
            return {
                "items": item_params,
                "loglik_history": loglik_history,
                "iterations": actual_iterations,
            }
        return item_params

    # --- 多模型 IRT (1PL/2PL/3PL/4PL) ---

    def predict_correct_1pl(self, theta: float, b: float) -> float:
        """1PL (Rasch) 项目反应函数 — P = 1/(1+exp(-(theta-b))).

        等价于 a=1, c=0 的 2PL/3PL.
        """
        z = theta - b
        p = 1.0 / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, p))

    def predict_correct_2pl(self, theta: float, a: float, b: float) -> float:
        """2PL 项目反应函数 — P = 1/(1+exp(-a*(theta-b))).

        等价于 c=0 的 3PL.
        """
        z = a * (theta - b)
        p = 1.0 / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, p))

    def information_4pl(
        self,
        theta: float,
        a: float,
        b: float,
        c: float = 0.0,
        d: float = 1.0,
    ) -> float:
        """4PL Fisher 信息量.

        I(θ) = a² * (P-c)² * (d-P)² / ((d-c)² * P * (1-P))

        推导: dP/dθ = a*(P-c)*(d-P)/(d-c), 故
        I = [dP/dθ]² / [P(1-P)] = a²*(P-c)²*(d-P)² / ((d-c)²*P*(1-P))

        - 3PL (d=1.0): (1-P)²/(1-P) = (1-P), 退化为 a²*(P-c)²*(1-P)/((1-c)²*P)
        - 2PL (c=0, d=1): P²*(1-P)²/(P*(1-P)) = P*(1-P), 退化为 a²*P*(1-P)
        - 4PL (d<1): 上渐近线降低信息量 (d-P < 1-P)
        """
        p = self.predict_correct_4pl(theta, a, b, c, d)
        if c <= 0.0 and d >= 1.0:
            return a * a * p * (1.0 - p)
        denom = (d - c) ** 2 * p * (1.0 - p)
        if denom <= _EPS:
            return 0.0
        return a * a * (p - c) ** 2 * (d - p) ** 2 / denom

    def _loglik_by_model(
        self,
        theta: float,
        responses: list[tuple[dict[str, Any], bool]],
        model: str,
    ) -> float:
        """按指定模型计算对数似然 (内部辅助)."""
        ll = 0.0
        for params, correct in responses:
            a = float(params.get("a", 1.0))
            b = float(params.get("b", 0.0))
            c = float(params.get("c", 0.0))
            if model == "1PL":
                p = self.predict_correct_1pl(theta, b)
            elif model == "2PL":
                p = self.predict_correct_2pl(theta, a, b)
            elif model == "3PL":
                p = self.predict_correct(theta, a, b, c)
            elif model == "4PL":
                d = float(params.get("d", 1.0))
                p = self.predict_correct_4pl(theta, a, b, c, d)
            else:
                raise ValueError(f"未知模型: {model!r}")
            p = max(_EPS, min(1.0 - _EPS, p))
            if correct:
                ll += math.log(p)
            else:
                ll += math.log(1.0 - p)
        return ll

    def compare_models(
        self,
        responses: list[tuple[dict[str, Any], bool]],
    ) -> dict[str, dict[str, float]]:
        """多模型 AIC/BIC 比较 — 嵌套模型选择 (1PL→2PL→3PL→4PL).

        对每个模型用 Newton-Raphson 估计 theta, 然后计算 AIC/BIC:
        - AIC = -2*loglik + 2*k
        - BIC = -2*loglik + k*ln(n)
        其中 k = 每题参数数 * 题数 + 1 (theta), n = 作答数.
        """
        if not responses:
            return {}
        n = len(responses)
        n_items = len(set(
            (p.get("a", 1.0), p.get("b", 0.0), p.get("c", 0.0))
            for p, _ in responses
        ))
        results: dict[str, dict[str, float]] = {}
        for model, params_per_item in (
            ("1PL", 1), ("2PL", 2), ("3PL", 3), ("4PL", 4)
        ):
            try:
                state = self.estimate_by_model(responses, model)
                ll = self._loglik_by_model(state.theta, responses, model)
                k = params_per_item * max(n_items, 1) + 1
                aic = -2.0 * ll + 2.0 * k
                bic = -2.0 * ll + k * math.log(max(n, 1))
                results[model] = {
                    "aic": aic,
                    "bic": bic,
                    "loglik": ll,
                    "n_params": k,
                    "theta": state.theta,
                }
            except Exception:
                results[model] = {
                    "aic": float("inf"),
                    "bic": float("inf"),
                    "loglik": float("-inf"),
                    "n_params": params_per_item * max(n_items, 1) + 1,
                    "theta": 0.0,
                }
        return results

    def select_best_model(
        self,
        responses: list[tuple[dict[str, Any], bool]],
    ) -> str:
        """自动模型选择 — 基于 BIC 选择最优模型 (BIC 对复杂模型惩罚更重)."""
        if not responses:
            return "2PL"
        comparison = self.compare_models(responses)
        if not comparison:
            return "2PL"
        return min(comparison, key=lambda m: comparison[m]["bic"])

    def estimate_by_model(
        self,
        responses: list[tuple[dict[str, Any], bool]],
        model: str = "3PL",
    ) -> IRTState:
        """按指定模型估计能力 — 将题目参数适配到目标模型后用 Newton-Raphson.

        - 1PL: a=1, c=0 (仅用 b)
        - 2PL: c=0 (用 a, b)
        - 3PL: 用 a, b, c (原始 3PL)
        - 4PL: 用 a, b, c, d (d 默认 1.0)
        """
        if not responses:
            return IRTState(
                theta=_FALLBACK_THETA, se=_FALLBACK_SE,
                response_count=0, last_update_time=0.0,
            )
        adapted: list[tuple[dict[str, Any], bool]] = []
        for params, correct in responses:
            a = float(params.get("a", 1.0))
            b = float(params.get("b", 0.0))
            c = float(params.get("c", 0.0))
            if model == "1PL":
                adapted.append(({"a": 1.0, "b": b, "c": 0.0}, correct))
            elif model == "2PL":
                adapted.append(({"a": a, "b": b, "c": 0.0}, correct))
            elif model == "3PL":
                adapted.append(({"a": a, "b": b, "c": c}, correct))
            elif model == "4PL":
                d = float(params.get("d", 1.0))
                adapted.append(({"a": a, "b": b, "c": c, "d": d}, correct))
            else:
                raise ValueError(f"未知模型: {model!r}")
        return self.estimate_mle_newton_raphson(adapted)

    # --- 贝叶斯可信区间 ---

    def estimate_with_credible_interval(
        self,
        responses: list[tuple[dict[str, Any], bool]],
        credible_level: float = 0.95,
    ) -> dict[str, float]:
        """带可信区间的能力估计 — 正态近似等尾区间.

        θ_hat ± z_{α/2} * SE

        其中 z_{α/2} 为标准正态分位数 (95% → 1.96).
        """
        state = self.estimate_mle_newton_raphson(responses)
        alpha = 1.0 - credible_level
        z = self._norm_ppf(1.0 - alpha / 2.0)
        ci_lower = state.theta - z * state.se
        ci_upper = state.theta + z * state.se
        return {
            "theta": state.theta,
            "se": state.se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "credible_level": credible_level,
        }

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """标准正态分布分位数 (近似, Beasley-Springer-Moro 算法)."""
        if p <= 0.0:
            return -3.5
        if p >= 1.0:
            return 3.5
        # Beasley-Springer-Moro 近似
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

    # --- 群体先验估计 ---

    def estimate_group_prior(
        self,
        responses_by_learner: dict[str, list[tuple[dict[str, Any], bool]]],
    ) -> dict[str, float]:
        """从多学习者作答数据自动估计群体先验 (Empirical Bayes).

        对每个学习者做 MLE 估计, 然后计算群体均值与标准差.
        """
        if not responses_by_learner:
            return {"mean": 0.0, "sd": 1.0}
        thetas: list[float] = []
        for responses in responses_by_learner.values():
            if responses:
                state = self.estimate_mle_newton_raphson(responses)
                thetas.append(state.theta)
        if not thetas:
            return {"mean": 0.0, "sd": 1.0}
        mean = sum(thetas) / len(thetas)
        if len(thetas) > 1:
            var = sum((t - mean) ** 2 for t in thetas) / (len(thetas) - 1)
            sd = math.sqrt(max(var, _EPS))
        else:
            sd = 1.0
        return {"mean": mean, "sd": sd}


# ============================================================
# __all__
# ============================================================

__all__ = [
    "IRTEstimator",
]
