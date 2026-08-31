"""BKT 贝叶斯知识追踪引擎.

融合世界先进方案:
- Corbett & Anderson (1995): 标准 BKT 四参数模型 + 前向算法
- Yudelson-Koedinger-Gordon (CMU 2013): Individualized BKT (BPT) 自适应参数
- Khan Academy: BKT 后验概率 + 遗忘曲线衰减

BKT 四参数:
- p_l0: 先验掌握概率 P(Know)
- p_t: 学习转移概率 P(Transit)
- p_g: 猜测概率 P(Guess)
- p_s: 失误概率 P(Slip)

前向算法 (O(1) 增量更新):
1. 答对后验: P(L|correct) = P(L)*(1-S) / (P(L)*(1-S) + (1-P(L))*G)
2. 答错后验: P(L|wrong) = P(L)*S / (P(L)*S + (1-P(L))*(1-G))
3. 转移: P(L)_next = P(L|obs) + (1-P(L|obs))*T
4. 预测: P(C) = P(L)*(1-S) + (1-P(L))*G

设计说明:
- BKTTracer 为无状态引擎类 (不持有学习者状态), 所有状态显式以
  ``TracingState`` 入参/返回传递, 便于并发与持久化.
- ``update`` 采用函数式风格: 返回新的 ``TracingState``, 不修改入参.
- ``bkt_params`` 中的 ``p_l0`` 为初始先验, 在前向更新过程中保持不变;
  当前掌握概率随观测演化, 存于 ``TracingState.mastery_prob``.
"""

from __future__ import annotations

import math
from typing import Any

from dy3_polaris.l2.models import (
    DEFAULT_BKT_PARAMS,
    AnswerRecord,
    TracingState,
)


# ============================================================
# 1. 常量定义
# ============================================================

# 难度 -> 先验掌握概率 p_l0 的线性映射端点
# difficulty 0.0 (最易) -> p_l0 = 0.7 (高先验)
# difficulty 1.0 (最难) -> p_l0 = 0.3 (低先验)
_EASY_P_L0: float = 0.7
_HARD_P_L0: float = 0.3

# 数值稳定性下限 (避免分母为 0)
_EPS: float = 1e-12

# 概率边界 (logit/sigmoid 数值稳定钳制)
_PROB_MIN: float = 1e-12
_PROB_MAX: float = 1.0 - 1e-12


# ============================================================
# 1.1 BPT 个性化参数融合辅助 (Yudelson-Koedinger-Gordon, CMU 2013)
# ============================================================


def _logit(p: float) -> float:
    """logit 变换: logit(p) = ln(p / (1-p)).

    对 p 进行数值稳定钳制到 (0, 1) 内部, 避免两端奇异.

    Args:
        p: 概率值, 理论上 (0, 1).

    Returns:
        logit 值 (实数域, 可正可负).
    """
    p = max(min(p, _PROB_MAX), _PROB_MIN)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """sigmoid 函数: sigmoid(x) = 1 / (1 + exp(-x)).

    Args:
        x: 实数.

    Returns:
        概率值 (0, 1).
    """
    return 1.0 / (1.0 + math.exp(-x))


def _fuse_params(skill_param: float, learner_param: float) -> float:
    """logit-sigmoid 融合: w = sigmoid(logit(skill) + logit(learner)).

    将技能级参数与学习者级参数在 logit 空间相加后经 sigmoid 映射回概率,
    等价于贝叶斯式的独立证据融合 (noisy-OR 归一化):

        fused = a*b / (a*b + (1-a)*(1-b))

    其中 a=skill_param, b=learner_param. 当学习者级证据与技能级证据
    一致时增强, 不一致时折中, 始终落在 (0, 1).

    Args:
        skill_param: 技能级 (全局) BKT 参数.
        learner_param: 学习者级 BKT 参数.

    Returns:
        融合后的参数值 (0, 1).
    """
    return _sigmoid(_logit(skill_param) + _logit(learner_param))


# ============================================================
# 2. BKTTracer 无状态引擎类
# ============================================================


class BKTTracer:
    """BKT 贝叶斯知识追踪引擎 (无状态).

    提供 BKT 前向算法的初始化 / 增量更新 / 批量重建 / 正确率预测:
    1. ``init_state``: 根据题目难度映射先验 p_l0, 初始化追踪状态.
    2. ``update``: 单次作答的 O(1) 增量更新 (后验 + 转移).
    3. ``batch_trace``: 按时间戳排序逐条更新, 重建历史追踪状态.
    4. ``predict_correct_prob``: 预测下一次作答的正确率.

    该类不持有任何学习者状态 (无状态引擎), 多次实例化行为一致,
    适合作为单例或在多线程/多学习者场景下复用.
    """

    # --- 难度 -> p_l0 映射 ---

    @staticmethod
    def _difficulty_to_p_l0(difficulty: float) -> float:
        """将题目难度线性映射为先验掌握概率 p_l0.

        映射关系: p_l0 = 0.7 - 0.4 * difficulty
        - difficulty 0.0 (最易) -> p_l0 = 0.7
        - difficulty 0.5 (中等) -> p_l0 = 0.5
        - difficulty 1.0 (最难) -> p_l0 = 0.3

        Args:
            difficulty: 题目难度 [0.0, 1.0].

        Returns:
            先验掌握概率 p_l0, clamp 到 [0.0, 1.0].
        """
        p_l0 = _EASY_P_L0 + (_HARD_P_L0 - _EASY_P_L0) * difficulty
        return max(0.0, min(1.0, p_l0))

    # --- 初始化 ---

    def init_state(self, kp_id: str, difficulty: float) -> TracingState:
        """根据题目难度初始化知识点的 BKT 追踪状态.

        难度通过 ``_difficulty_to_p_l0`` 映射为先验 p_l0,
        初始 ``mastery_prob`` 取该先验值, 计数归零.

        Args:
            kp_id: 知识点 ID.
            difficulty: 题目难度 [0.0, 1.0].

        Returns:
            初始化后的 TracingState (mastery_prob=p_l0, attempts=0).
        """
        p_l0 = self._difficulty_to_p_l0(difficulty)
        bkt_params: dict[str, Any] = dict(DEFAULT_BKT_PARAMS)
        bkt_params["p_l0"] = p_l0
        return TracingState(
            kp_id=kp_id,
            mastery_prob=p_l0,
            attempts=0,
            correct_count=0,
            last_attempt_time=0.0,
            bkt_params=bkt_params,
        )

    # --- 单次增量更新 ---

    def update(
        self,
        state: TracingState,
        correct: bool,
        timestamp: float,
    ) -> TracingState:
        """BKT 前向算法: 单次作答的 O(1) 增量更新.

        步骤:
        1. 后验更新 (Bayesian posterior):
           - 答对: P(L|correct) = P(L)*(1-S) / (P(L)*(1-S) + (1-P(L))*G)
           - 答错: P(L|wrong)   = P(L)*S     / (P(L)*S + (1-P(L))*(1-G))
        2. 学习转移 (transit): P(L)' = P(L|obs) + (1-P(L|obs))*T
        3. 计数更新: attempts += 1, correct_count += (1 if correct else 0),
           last_attempt_time = timestamp.

        采用函数式风格: 返回新的 TracingState, 不修改入参 ``state``.
        ``bkt_params`` 中的 ``p_l0`` 保持先验不变 (仅 ``mastery_prob`` 演化).

        Args:
            state: 当前追踪状态 (含 mastery_prob 与 bkt_params).
            correct: 本次是否答对.
            timestamp: 本次作答时间戳 (秒, float).

        Returns:
            更新后的新 TracingState.
        """
        params = state.bkt_params
        p_l = state.mastery_prob
        p_s = float(params.get("p_s", DEFAULT_BKT_PARAMS["p_s"]))
        p_g = float(params.get("p_g", DEFAULT_BKT_PARAMS["p_g"]))
        p_t = float(params.get("p_t", DEFAULT_BKT_PARAMS["p_t"]))

        # --- 1. 后验更新 ---
        if correct:
            # P(L|correct) = P(L)*(1-S) / (P(L)*(1-S) + (1-P(L))*G)
            num = p_l * (1.0 - p_s)
            den = num + (1.0 - p_l) * p_g
        else:
            # P(L|wrong) = P(L)*S / (P(L)*S + (1-P(L))*(1-G))
            num = p_l * p_s
            den = num + (1.0 - p_l) * (1.0 - p_g)

        # 数值稳定性: 分母过小时回退到当前掌握度
        if den <= _EPS:
            p_l_post = p_l
        else:
            p_l_post = num / den

        # --- 2. 学习转移 ---
        p_l_next = p_l_post + (1.0 - p_l_post) * p_t
        # 钳制到 [0, 1]
        p_l_next = max(0.0, min(1.0, p_l_next))

        # --- 3. 计数与时间更新 ---
        return TracingState(
            kp_id=state.kp_id,
            mastery_prob=p_l_next,
            attempts=state.attempts + 1,
            correct_count=state.correct_count + (1 if correct else 0),
            last_attempt_time=timestamp,
            # p_l0 保持先验不变, 浅拷贝避免共享引用
            bkt_params=dict(state.bkt_params),
        )

    # --- 批量历史重建 ---

    def batch_trace(
        self,
        records: list[AnswerRecord],
    ) -> dict[str, TracingState]:
        """批量历史重建: 按时间戳排序后逐条更新.

        对每条答题记录:
        - 若该 ``kp_id`` 首次出现, 使用该条记录的 ``difficulty`` 初始化先验 p_l0;
        - 否则在已有状态上调用 ``update`` 增量更新.

        多个 ``kp_id`` 之间相互隔离, 各自独立追踪.

        Args:
            records: 答题记录列表 (时间戳可乱序, 内部按升序排序).

        Returns:
            ``{kp_id: TracingState}`` 映射; 空记录返回空字典.
        """
        if not records:
            return {}

        # 按时间戳升序排序 (稳定排序, 同时间戳保持原相对顺序)
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        states: dict[str, TracingState] = {}
        for rec in sorted_records:
            kp_id = rec.kp_id
            if kp_id not in states:
                states[kp_id] = self.init_state(kp_id, rec.difficulty)
            states[kp_id] = self.update(
                states[kp_id], rec.correct, rec.timestamp
            )
        return states

    # --- 正确率预测 ---

    def predict_correct_prob(self, state: TracingState) -> float:
        """预测下一次作答的正确率 P(correct).

        公式: P(C) = P(L)*(1-S) + (1-P(L))*G
        - 已掌握答对概率: (1 - S)
        - 未掌握猜对概率: G

        Args:
            state: 当前追踪状态.

        Returns:
            预测正确率 [0.0, 1.0].
        """
        params = state.bkt_params
        p_l = state.mastery_prob
        p_s = float(params.get("p_s", DEFAULT_BKT_PARAMS["p_s"]))
        p_g = float(params.get("p_g", DEFAULT_BKT_PARAMS["p_g"]))
        return p_l * (1.0 - p_s) + (1.0 - p_l) * p_g

    # --- 参数约束校验 ---

    @staticmethod
    def validate_params(params: dict[str, Any]) -> bool:
        """校验 BKT 四参数的合法性与约束.

        约束:
        1. p_l0 / p_t / p_g / p_s 均存在且落在 [0.0, 1.0];
        2. p_g + p_s < 1.0 (猜测与失误概率之和须严格小于 1,
           否则 BKT 观测模型退化为不可识别).

        Args:
            params: BKT 参数字典, 含 p_l0/p_t/p_g/p_s.

        Returns:
            True 表示参数合法.

        Raises:
            ValueError: 参数缺失、越界或违反 p_g + p_s < 1 约束.
        """
        for key in ("p_l0", "p_t", "p_g", "p_s"):
            if key not in params or params[key] is None:
                raise ValueError(f"missing BKT parameter: {key}")
            value = float(params[key])
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"BKT parameter {key}={value} out of range [0, 1]"
                )
        p_g = float(params["p_g"])
        p_s = float(params["p_s"])
        if p_g + p_s >= 1.0:
            raise ValueError(
                f"constraint violated: p_g + p_s = {p_g + p_s} >= 1.0"
            )
        return True

    # --- BPT 个性化更新 (Individualized BKT) ---

    def update_individualized(
        self,
        state: TracingState,
        correct: bool,
        timestamp: float,
        learner_params: dict[str, float] | None = None,
    ) -> TracingState:
        """BPT 个性化 BKT 前向更新 (Yudelson-Koedinger-Gordon, CMU 2013).

        在标准 BKT 前向更新的基础上, 将技能级 (skill-level) 参数与
        学习者级 (learner-level) 参数通过 logit-sigmoid 融合后再做后验/转移:

            fused = sigmoid(logit(skill_param) + logit(learner_param))

        融合仅作用于本次更新所用的 p_t / p_g / p_s (学习者级参数可选提供,
        未提供的维度沿用技能级原值, 不做融合).

        回退策略: 若 ``learner_params`` 为 None 或空字典, 等价于标准
        ``update`` (不融合).

        与 ``update`` 一致采用函数式风格: 返回新 TracingState, 不修改入参;
        返回状态的 ``bkt_params`` 保持技能级原值 (融合为瞬时计算).

        Args:
            state: 当前追踪状态 (含 mastery_prob 与技能级 bkt_params).
            correct: 本次是否答对.
            timestamp: 本次作答时间戳 (秒, float).
            learner_params: 学习者级参数, 可选键:
                ``learner_p_t`` / ``learner_p_g`` / ``learner_p_s``.
                为 None 或空时回退到标准 update.

        Returns:
            更新后的新 TracingState (bkt_params 保持技能级原值).
        """
        # 回退: 无学习者级参数 -> 标准 update
        if not learner_params:
            return self.update(state, correct, timestamp)

        skill = state.bkt_params
        # 工作副本: 在技能级基础上, 对提供了学习者级参数的维度做融合
        work: dict[str, Any] = dict(skill)
        mapping = {
            "learner_p_t": "p_t",
            "learner_p_g": "p_g",
            "learner_p_s": "p_s",
        }
        for learner_key, skill_key in mapping.items():
            if learner_key in learner_params and learner_params[learner_key] is not None:
                skill_val = float(skill.get(skill_key, DEFAULT_BKT_PARAMS[skill_key]))
                learner_val = float(learner_params[learner_key])
                work[skill_key] = _fuse_params(skill_val, learner_val)

        p_l = state.mastery_prob
        p_s = float(work.get("p_s", DEFAULT_BKT_PARAMS["p_s"]))
        p_g = float(work.get("p_g", DEFAULT_BKT_PARAMS["p_g"]))
        p_t = float(work.get("p_t", DEFAULT_BKT_PARAMS["p_t"]))

        # --- 后验更新 (使用融合参数) ---
        if correct:
            num = p_l * (1.0 - p_s)
            den = num + (1.0 - p_l) * p_g
        else:
            num = p_l * p_s
            den = num + (1.0 - p_l) * (1.0 - p_g)
        p_l_post = num / den if den > _EPS else p_l

        # --- 学习转移 ---
        p_l_next = p_l_post + (1.0 - p_l_post) * p_t
        p_l_next = max(0.0, min(1.0, p_l_next))

        return TracingState(
            kp_id=state.kp_id,
            mastery_prob=p_l_next,
            attempts=state.attempts + 1,
            correct_count=state.correct_count + (1 if correct else 0),
            last_attempt_time=timestamp,
            # bkt_params 保持技能级原值 (融合为瞬时计算, 不持久化)
            bkt_params=dict(state.bkt_params),
        )

    # --- 对数似然 (EM/梯度上升的基础) ---

    def log_likelihood(
        self,
        records: list[AnswerRecord],
        params: dict[str, Any],
    ) -> float:
        """计算答题序列在给定 BKT 参数下的前向对数似然.

        按时间戳升序处理序列, 每步:
        1. 预测观测概率 P(C_t) = P(L)*(1-S) + (1-P(L))*G;
        2. 累加 log P(obs_t);
        3. 前向更新 P(L) (后验 + 转移).

        Args:
            records: 答题记录列表 (时间戳可乱序, 内部按升序排序).
            params: BKT 参数字典, 含 p_l0/p_t/p_g/p_s.

        Returns:
            序列对数似然 (负值, 越大越好); 空序列返回 0.0.
        """
        if not records:
            return 0.0

        sorted_records = sorted(records, key=lambda r: r.timestamp)
        p_l = float(params.get("p_l0", DEFAULT_BKT_PARAMS["p_l0"]))
        p_t = float(params.get("p_t", DEFAULT_BKT_PARAMS["p_t"]))
        p_g = float(params.get("p_g", DEFAULT_BKT_PARAMS["p_g"]))
        p_s = float(params.get("p_s", DEFAULT_BKT_PARAMS["p_s"]))

        ll = 0.0
        for rec in sorted_records:
            p_correct = p_l * (1.0 - p_s) + (1.0 - p_l) * p_g
            p_obs = p_correct if rec.correct else (1.0 - p_correct)
            ll += math.log(max(p_obs, _EPS))

            if rec.correct:
                num = p_l * (1.0 - p_s)
                den = num + (1.0 - p_l) * p_g
            else:
                num = p_l * p_s
                den = num + (1.0 - p_l) * (1.0 - p_g)
            p_l_post = num / den if den > _EPS else p_l
            p_l = p_l_post + (1.0 - p_l_post) * p_t
            p_l = min(max(p_l, _EPS), 1.0 - _EPS)

        return ll

    # --- 参数学习 (梯度上升 + 收敛检测 + best-tracking) ---

    def fit_params(
        self,
        records: list[AnswerRecord],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """从答题历史学习 BKT 参数 (梯度上升 + 收敛检测 + best-tracking).

        以 ``DEFAULT_BKT_PARAMS`` 为初值, 对 p_l0 / p_t / p_g / p_s 做有限差分
        梯度上升 (最多 max_iter 步), 每步沿对数似然梯度方向更新并钳制到 (0, 1);
        每步后投影以满足约束 p_g + p_s < 1.

        增强特性 (相对朴素梯度上升):
        - ``p_l0`` 一并学习 (随观测调整先验掌握概率, 而非保持默认);
        - 收敛检测: 相邻两次迭代对数似然变化小于 ``tol`` 时提前停止,
          避免无意义的满迭代开销;
        - best-tracking: 全程记录对数似然最高的参数, 返回最佳参数
          (避免梯度震荡/过冲导致返回劣质最终参数).

        Args:
            records: 答题记录列表 (单个技能的作答历史).
            max_iter: 梯度上升最大迭代次数, 默认 100.
            tol: 收敛容差; 相邻迭代对数似然变化绝对值小于该值时提前停止,
                默认 1e-6.

        Returns:
            优化后的 BKT 参数字典 (p_l0/p_t/p_g/p_s, 满足约束, 最高似然).
        """
        if not records:
            return dict(DEFAULT_BKT_PARAMS)

        sorted_records = sorted(records, key=lambda r: r.timestamp)
        params: dict[str, float] = {k: float(v) for k, v in DEFAULT_BKT_PARAMS.items()}
        # p_l0 一并学习 (随观测调整先验掌握概率)
        learn_keys = ("p_l0", "p_t", "p_g", "p_s")

        lr = 0.05  # 学习率
        eps = 1e-4  # 有限差分步长

        # best-tracking: 记录最高对数似然对应的参数 (初始化为初始参数)
        best_ll = self.log_likelihood(sorted_records, params)
        best_params: dict[str, float] = dict(params)
        prev_ll: float = best_ll

        for iteration in range(max_iter):
            # --- 有限差分梯度 ---
            grads: dict[str, float] = {}
            for key in learn_keys:
                p_plus = dict(params)
                p_plus[key] = min(0.999, params[key] + eps)
                p_minus = dict(params)
                p_minus[key] = max(0.001, params[key] - eps)
                ll_plus = self.log_likelihood(sorted_records, p_plus)
                ll_minus = self.log_likelihood(sorted_records, p_minus)
                h = p_plus[key] - p_minus[key]
                grads[key] = (ll_plus - ll_minus) / h if h > 0.0 else 0.0

            # --- 梯度上升更新 ---
            for key in learn_keys:
                params[key] = params[key] + lr * grads[key]
                params[key] = min(max(params[key], 1e-3), 1.0 - 1e-3)

            # 投影以满足约束 p_g + p_s < 1
            if params["p_g"] + params["p_s"] >= 1.0:
                total = params["p_g"] + params["p_s"]
                params["p_g"] = params["p_g"] / total * 0.99
                params["p_s"] = params["p_s"] / total * 0.99

            # --- best-tracking: 记录最高似然参数 ---
            current_ll = self.log_likelihood(sorted_records, params)
            if current_ll > best_ll:
                best_ll = current_ll
                best_params = dict(params)

            # --- 收敛检测: 相邻迭代似然变化 < tol 则提前停止 ---
            if iteration > 0 and abs(current_ll - prev_ll) < tol:
                break
            prev_ll = current_ll

        # 最终校验 (确保返回合法参数)
        self.validate_params(best_params)
        return best_params

    # --- 在线增量参数更新 (单条记录随机梯度上升) ---

    def _single_record_log_likelihood(
        self,
        state: TracingState,
        record: AnswerRecord,
        params: dict[str, Any],
    ) -> float:
        """计算单条记录在给定参数下的观测对数似然.

        以 ``state.mastery_prob`` 作为观测前的当前掌握概率 P(L) (在线设定:
        当前信念已知), 仅观测模型参与:
            P(correct) = P(L)*(1-p_s) + (1-P(L))*p_g
            ll = log(P(obs))

        注: p_l0 / p_t 描述信念演化, 不直接进入单步观测似然, 因此其单步
        梯度为 0 (由 ``compute_gradient_single`` 以有限差分自然给出).

        Args:
            state: 当前追踪状态 (提供 mastery_prob 作为信念 P(L)).
            record: 单条答题记录.
            params: BKT 参数字典 (使用其中的 p_g / p_s).

        Returns:
            单条记录的观测对数似然 (负值, 越大越好).
        """
        p_l = state.mastery_prob
        p_g = float(params.get("p_g", DEFAULT_BKT_PARAMS["p_g"]))
        p_s = float(params.get("p_s", DEFAULT_BKT_PARAMS["p_s"]))
        p_correct = p_l * (1.0 - p_s) + (1.0 - p_l) * p_g
        p_obs = p_correct if record.correct else (1.0 - p_correct)
        return math.log(max(p_obs, _EPS))

    def compute_gradient_single(
        self,
        state: TracingState,
        record: AnswerRecord,
        params: dict[str, Any],
    ) -> dict[str, float]:
        """计算单条记录的参数梯度 (中心有限差分).

        以 ``state.mastery_prob`` 为当前信念, 对 p_l0 / p_t / p_g / p_s
        各维度做中心有限差分, 得到单步观测对数似然的梯度方向.

        数值说明:
        - p_g / p_s 直接进入观测模型, 梯度通常非零;
        - p_l0 / p_t 在固定信念的单步观测中不出现, 梯度为 0.0
          (这与在线增量更新的设定一致: 先验与转移通过信念演化间接生效,
          不在单步观测中被即时调整).

        Args:
            state: 当前追踪状态 (提供 mastery_prob).
            record: 单条答题记录.
            params: 待求梯度的 BKT 参数字典.

        Returns:
            ``{p_l0, p_t, p_g, p_s}`` 各维度的梯度值.
        """
        eps = 1e-4
        base: dict[str, float] = {
            k: float(params.get(k, DEFAULT_BKT_PARAMS[k]))
            for k in ("p_l0", "p_t", "p_g", "p_s")
        }
        grads: dict[str, float] = {}
        for key in ("p_l0", "p_t", "p_g", "p_s"):
            p_plus = dict(base)
            p_plus[key] = min(0.999, base[key] + eps)
            p_minus = dict(base)
            p_minus[key] = max(0.001, base[key] - eps)
            ll_plus = self._single_record_log_likelihood(state, record, p_plus)
            ll_minus = self._single_record_log_likelihood(state, record, p_minus)
            h = p_plus[key] - p_minus[key]
            grads[key] = (ll_plus - ll_minus) / h if h > 0.0 else 0.0
        return grads

    def incremental_fit(
        self,
        state: TracingState,
        record: AnswerRecord,
        learning_rate: float = 0.01,
    ) -> dict[str, float]:
        """在线增量参数更新: 基于单条记录的随机梯度上升.

        以 ``state.bkt_params`` 为起点, 计算单条记录的参数梯度并沿梯度
        上升方向更新 (params += lr * gradient), 随后:
        1. 将各参数钳制到 (1e-3, 1 - 1e-3);
        2. 投影以满足约束 p_g + p_s < 1 (与 ``fit_params`` 一致).

        采用函数式风格: 不修改入参 ``state`` 的 ``bkt_params``, 返回新的
        参数字典.

        Args:
            state: 当前追踪状态 (提供 mastery_prob 与起始 bkt_params).
            record: 新到达的单条答题记录.
            learning_rate: 随机梯度上升学习率, 默认 0.01.

        Returns:
            更新后的 BKT 参数字典 (满足约束, 落在 (0, 1)).
        """
        params: dict[str, float] = {
            k: float(state.bkt_params.get(k, DEFAULT_BKT_PARAMS[k]))
            for k in ("p_l0", "p_t", "p_g", "p_s")
        }
        grads = self.compute_gradient_single(state, record, params)
        for key in ("p_l0", "p_t", "p_g", "p_s"):
            params[key] = params[key] + learning_rate * grads[key]
            params[key] = min(max(params[key], 1e-3), 1.0 - 1e-3)

        # 投影以满足约束 p_g + p_s < 1
        if params["p_g"] + params["p_s"] >= 1.0:
            total = params["p_g"] + params["p_s"]
            params["p_g"] = params["p_g"] / total * 0.99
            params["p_s"] = params["p_s"] / total * 0.99

        return params


# ============================================================
# __all__
# ============================================================

__all__ = [
    "BKTTracer",
]
