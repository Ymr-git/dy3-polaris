"""FSRS-6 间隔重复调度引擎 — 基于记忆稳定性/难度/可提取性的幂律遗忘模型.

融合世界先进方案:
- FSRS-6 (Free Spaced Repetition Scheduler):
    open-spaced-repetition/fsrs4 全局参数集 (21 参数权重 w0-w20).
    核心三变量: stability (记忆稳定性), difficulty (难度), retrievability (可提取性).
- Anki / SuperMemo: 间隔重复调度与状态机 (New → Learning → Review → Relearning).
- 认知科学: 幂律遗忘曲线 R(t) = (1 + factor * t / S)^(-decay).

算法核心 (FSRS-6 公式):
1. 可提取性 R = (1 + factor * elapsed_days / S)^(-decay)
   - decay = -w20 (FSRS-6 参数化), factor = 0.9^(1/decay) - 1
2. 首次复习 (state=new): S = w[G-1], D = w4 - exp(w5*(G-1)) + 1
3. 同日复习 (elapsed < 1 天): S' = w17*(S^w18 - S)*(1 - w19*S) + S (短期记忆模型)
4. 成功回忆 (grade>=2, 长期):
   S' = S * (1 + e^(w8)*(11-D)*S^(-w9)*e^(-w10*(1-R)) - 1) * hard_penalty * easy_bonus
5. 遗忘 (grade==1, 长期):
   S' = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^((1-R)*w14), 且 S' = min(S', S)
6. 难度更新: D' = D - w6*(G-3)*(10-D)/9; D' = w7*initial_difficulty(4) + (1-w7)*D'; clamp [1, 10]
7. 间隔: interval = max(1, round(S * (DR^(1/decay) - 1) / factor))
"""

from __future__ import annotations

import math
from typing import Any

from dy3_polaris.l1.models import (
    FSRSCardState,
    FSRSParameters,
    FSRSReviewLog,
    MS_PER_SEC,
)


class FSRSScheduler:
    """FSRS-6 间隔重复调度器.

    根据卡片当前状态和用户评分, 计算下一次复习的:
    - 更新后的记忆稳定性 (stability)
    - 更新后的难度 (difficulty)
    - 下次复习间隔 (天数)
    - 复习日志 (review log)

    评分等级 (grade):
    - 1 = Again (遗忘, 需要重新学习)
    - 2 = Hard (回忆困难)
    - 3 = Good (正常回忆)
    - 4 = Easy (轻松回忆)

    衰减参数从 FSRSParameters.decay / .factor 获取 (FSRS-6 参数化).
    """

    # 毫秒/天
    _MS_PER_DAY: float = float(MS_PER_SEC * 86400)

    def schedule_review(
        self,
        card_state: FSRSCardState,
        grade: int,
        params: FSRSParameters,
        current_ts: int,
        enable_fuzzing: bool = False,
    ) -> tuple[FSRSCardState, FSRSReviewLog, int]:
        """调度复习并返回 (新卡片状态, 复习日志, 下次间隔天数).

        Args:
            card_state: 当前卡片状态 (含 stability/difficulty/state 等).
            grade: 评分 1-4 (Again/Hard/Good/Easy).
            params: FSRS 参数 (含 21 个权重, decay/factor 属性).
            current_ts: 当前时间戳 (毫秒).
            enable_fuzzing: 是否对间隔添加 ±5% 随机抖动.

        Returns:
            (new_card_state, review_log, next_interval_days)
        """
        if not (1 <= grade <= 4):
            raise ValueError(f"grade must be in [1, 4], got {grade}")

        w = params.weights
        state_before = card_state.state

        # 计算自上次复习以来的天数
        elapsed_days = 0.0
        if state_before != FSRSCardState.NEW and card_state.last_review_ts > 0:
            elapsed_days = max(
                0.0,
                (current_ts - card_state.last_review_ts) / self._MS_PER_DAY,
            )

        if state_before == FSRSCardState.NEW:
            # --- 首次复习: 使用初始参数 ---
            new_stability = params.initial_stability(grade)
            new_difficulty = params.initial_difficulty(grade)
            # 新卡片首次复习后进入学习阶段
            state_after = FSRSCardState.LEARNING
            new_reps = 1
            new_lapses = 0
        else:
            # --- 复习中卡片: FSRS-6 公式更新 ---
            S = max(card_state.stability, 0.1)
            D = card_state.difficulty

            # 可提取性 R (使用参数化 decay/factor)
            R = card_state.retrievability(
                current_ts, decay=params.decay, factor=params.factor
            )
            if R <= 0.0:
                R = 1.0  # 防御: 无法计算时视为刚复习 (R=1)

            # 难度更新 (先更新 D, 再用新 D 计算 S)
            # D' = D - w6*(grade-3)*(10-D)/9
            next_d = D - w[6] * (grade - 3) * (10.0 - D) / 9.0
            # 均值回归: D' = w7*initial_difficulty(4) + (1-w7)*D'
            mean_revert_target = params.initial_difficulty(4)
            next_d = w[7] * mean_revert_target + (1.0 - w[7]) * next_d
            next_d = max(1.0, min(10.0, next_d))

            if elapsed_days < 1.0:
                # --- 短期记忆模型 (same-day, FSRS-5/6) ---
                w17 = w[17] if len(w) > 17 else 0.51655
                w18 = w[18] if len(w) > 18 else 0.6621
                w19 = w[19] if len(w) > 19 else 0.8285
                new_stability = w17 * (S ** w18 - S) * (1 - w19 * S) + S
                if grade >= 3:
                    new_stability = max(new_stability, S)
                # 状态转换
                if grade == 1:
                    state_after = FSRSCardState.RELEARNING
                    new_lapses = card_state.lapses + 1
                else:
                    state_after = FSRSCardState.REVIEW
                    new_lapses = card_state.lapses
            elif grade == 1:
                # --- 遗忘 (Again, 长期) ---
                # S' = w11 * D^(-w12) * ((S+1)^w13 - 1) * exp((1-R)*w14)
                new_stability = (
                    w[11]
                    * (next_d ** (-w[12]))
                    * ((S + 1.0) ** w[13] - 1.0)
                    * math.exp((1.0 - R) * w[14])
                )
                # 遗忘后稳定性不超过遗忘前
                new_stability = min(new_stability, S)
                state_after = FSRSCardState.RELEARNING
                new_lapses = card_state.lapses + 1
            else:
                # --- 成功回忆 (grade >= 2, 长期) ---
                # S' = S * (1 + exp(w8)*(11-D)*S^(-w9)*exp(-w10*(1-R)) - 1)
                #        * hard_penalty * easy_bonus
                hard_penalty = w[15] if grade == 2 else 1.0
                easy_bonus = w[16] if grade == 4 else 1.0
                new_stability = (
                    S
                    * (
                        1.0
                        + math.exp(w[8])
                        * (11.0 - next_d)
                        * (S ** (-w[9]))
                        * math.exp(-w[10] * (1.0 - R))
                        - 1.0
                    )
                    * hard_penalty
                    * easy_bonus
                )
                state_after = FSRSCardState.REVIEW
                new_lapses = card_state.lapses

            new_difficulty = next_d
            new_reps = card_state.reps + 1

        # 确保稳定性为正数
        new_stability = max(0.1, new_stability)

        # --- 间隔计算 (使用参数化 decay/factor) ---
        # interval = max(1, round(S * (DR^(1/decay) - 1) / factor))
        DR = params.request_retention
        decay = params.decay
        factor = params.factor
        interval_factor = DR ** (1.0 / decay) - 1.0
        next_interval = max(
            1,
            int(round(new_stability * interval_factor / factor)),
        )
        next_interval = min(next_interval, params.maximum_interval)

        # --- Fuzz factor (可选 ±5% 抖动) ---
        if enable_fuzzing:
            import random
            fuzz = 1 + random.uniform(-0.05, 0.05)
            next_interval = max(1, int(round(next_interval * fuzz)))

        # --- 构建新卡片状态 ---
        new_card = FSRSCardState(
            kc_id=card_state.kc_id,
            stability=new_stability,
            difficulty=new_difficulty,
            state=state_after,
            reps=new_reps,
            lapses=new_lapses,
            last_review_ts=current_ts,
        )

        # --- 构建复习日志 ---
        review_log = FSRSReviewLog(
            kc_id=card_state.kc_id,
            grade=grade,
            elapsed_days=elapsed_days,
            state_before=state_before,
            state_after=state_after,
            reviewed_at=current_ts,
        )

        return new_card, review_log, next_interval
