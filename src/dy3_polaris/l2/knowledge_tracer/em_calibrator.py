"""EM 参数标定服务 — 离线 BKT 参数校准.

融合世界先进方案:
- Corbett & Anderson (1995): BKT 参数 EM 估计
- Yudelson-Koedinger-Gordon (CMU 2013): 参数学习收敛检测 + best-tracking
- OSCOI 模式: Offline calibration + Online inference 分离

标定流程:
1. 收集某知识点的全部答题记录 (多学习者)
2. 以 DEFAULT_BKT_PARAMS 为初值, 做梯度上升优化 (有限差分)
3. 收敛检测: 相邻迭代对数似然变化 < tol 时提前停止
4. best-tracking: 全程记录最高似然参数, 避免梯度震荡/过冲
5. 约束投影: p_g + p_s < 1 (BKT 可识别性约束)

版本化标定 (calibrate_versioned):
- 记录版本号 / 标定时间 / 样本量, 供 OSCOI 在线推理阶段追溯
- 版本号自增: 同一 KP 重复标定时 version += 1

批量标定 (calibrate_batch):
- 对多个知识点并行标定, 返回 {kp_id: params} 映射
- 各 KP 独立标定, 互不影响
"""

from __future__ import annotations

import time
from typing import Any

from dy3_polaris.l2.knowledge_tracer.bkt import BKTTracer
from dy3_polaris.l2.models import AnswerRecord, DEFAULT_BKT_PARAMS


# ============================================================
# 常量定义
# ============================================================

# 默认最小记录数: 低于该值不触发标定 (数据不足)
DEFAULT_MIN_RECORDS: int = 30

# 默认最大迭代次数
DEFAULT_MAX_ITER: int = 100

# 默认收敛容差
DEFAULT_TOL: float = 1e-6


# ============================================================
# EMCalibrator — 离线参数标定服务
# ============================================================


class EMCalibrator:
    """EM 算法离线 BKT 参数标定服务.

    使用梯度上升 (有限差分) 从答题历史学习 BKT 四参数:
    - p_l0: 先验掌握概率
    - p_t: 学习转移概率
    - p_g: 猜测概率
    - p_s: 失误概率

    标定结果满足约束 p_g + p_s < 1 (BKT 可识别性).

    Args:
        min_records: 最小记录数, 低于该值 calibrate_if_needed 返回 None.
        max_iter: 梯度上升最大迭代次数.
        tol: 收敛容差.

    Attributes:
        bkt_tracer: BKT 引擎 (提供 fit_params / log_likelihood)
        min_records: 最小记录数阈值
        max_iter: 最大迭代次数
        tol: 收敛容差
    """

    def __init__(
        self,
        min_records: int = DEFAULT_MIN_RECORDS,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
    ) -> None:
        self.bkt_tracer = BKTTracer()
        self.min_records = min_records
        self.max_iter = max_iter
        self.tol = tol
        # 版本计数器: kp_id -> 当前版本号
        self._version_counter: dict[str, int] = {}

    # --- 单知识点标定 ---

    def calibrate(
        self,
        kp_id: str,
        records: list[AnswerRecord],
    ) -> dict[str, float]:
        """标定单个知识点的 BKT 参数.

        使用梯度上升从答题历史学习最优参数:
        1. 以 DEFAULT_BKT_PARAMS 为初值
        2. 有限差分梯度上升 (p_l0 / p_t / p_g / p_s)
        3. 收敛检测 + best-tracking
        4. 约束投影 p_g + p_s < 1

        空记录返回默认参数.

        Args:
            kp_id: 知识点 ID (标识用, 不参与计算).
            records: 该知识点的答题记录列表 (多学习者).

        Returns:
            BKT 参数字典 {p_l0, p_t, p_g, p_s}, 满足约束.
        """
        if not records:
            return dict(DEFAULT_BKT_PARAMS)

        return self.bkt_tracer.fit_params(
            records, max_iter=self.max_iter, tol=self.tol
        )

    # --- 条件标定 (阈值门控) ---

    def calibrate_if_needed(
        self,
        kp_id: str,
        records: list[AnswerRecord],
    ) -> dict[str, float] | None:
        """条件标定: 记录数达标时标定, 否则返回 None.

        Args:
            kp_id: 知识点 ID.
            records: 答题记录列表.

        Returns:
            标定参数字典 (达标时), 或 None (记录数 < min_records).
        """
        if len(records) < self.min_records:
            return None
        return self.calibrate(kp_id, records)

    # --- 批量标定 ---

    def calibrate_batch(
        self,
        all_records: dict[str, list[AnswerRecord]],
    ) -> dict[str, dict[str, float]]:
        """批量标定多个知识点的 BKT 参数.

        各知识点独立标定, 互不影响.

        Args:
            all_records: ``{kp_id: [AnswerRecord, ...]}`` 映射.

        Returns:
            ``{kp_id: {p_l0, p_t, p_g, p_s}}`` 映射.
        """
        return {
            kp_id: self.calibrate(kp_id, records)
            for kp_id, records in all_records.items()
        }

    # --- 版本化标定 ---

    def calibrate_versioned(
        self,
        kp_id: str,
        records: list[AnswerRecord],
    ) -> dict[str, Any]:
        """版本化标定: 标定参数 + 版本号 + 时间戳 + 样本量.

        版本号自增: 同一 KP 重复标定时 version += 1.
        供 OSCOI 在线推理阶段追溯标定历史.

        Args:
            kp_id: 知识点 ID.
            records: 答题记录列表.

        Returns:
            标定结果字典, 含:
            - params: BKT 参数 {p_l0, p_t, p_g, p_s}
            - version: 版本号 (>= 1)
            - calibrated_at: 标定时间戳 (秒)
            - sample_count: 样本量 (记录数)
        """
        params = self.calibrate(kp_id, records)
        version = self._version_counter.get(kp_id, 0) + 1
        self._version_counter[kp_id] = version

        return {
            "params": params,
            "version": version,
            "calibrated_at": time.time(),
            "sample_count": len(records),
        }


# ============================================================
# __all__
# ============================================================

__all__ = [
    "EMCalibrator",
    "DEFAULT_MIN_RECORDS",
]
