"""L2 ability_assessor 子模块 — IRT 能力评估 + CAT 自适应选题.

导出:
- IRTEstimator: IRT 项目反应理论能力评估引擎
    - predict_correct: 3PL 项目反应函数
    - information: Fisher 信息量 (2PL / 3PL)
    - update_theta: 贝叶斯 EAP 后验更新 (单题在线)
    - estimate_mle: 最大似然估计 (网格搜索, 批量离线)
- CATSelector: CAT 计算机化自适应测试选题引擎
    - select_next: 最大 Fisher 信息准则选题
    - should_stop: 终止条件判定 (SE < 阈值 或 题数 >= 上限)
    - estimate_ability: 从作答序列估计能力
- ZPDCalculator: 最近发展区 (ZPD) 量化计算器
    - calculate_zpd: 基于 IRT 正确率预测确定 ZPD 三区边界
    - recommend_difficulty: 结合支架水平推荐下次学习难度
    - classify_item: 分类题目对学习者的难度区域 (independent/zpd/frustration)
- ZPDResult: ZPD 计算结果 dataclass
- IRTTracingService: IRT 能力评估全链路编排器 (答题记录→IRT估计→CAT选题→ZPD校准→能力输出)
    - process: 单事件全链路处理, 返回 AbilityOutput
    - batch_process: 批量处理 (按时间戳升序)
    - select_next_item: 委托 CATSelector 选题 (基于当前 theta)
    - should_stop: 委托 CATSelector 终止条件判定
    - get_ability_snapshot: 获取当前能力快照 (冷启动回退群体先验)
    - set_item_bank: 设置题库 (供 CAT/ZPD)
- AbilityOutput: IRT 全链路输出标准化契约 (供 T2/T4/T5 下游消费)

设计参考:
- 贝叶斯 IRT (EAP) + 最大似然估计 (MLE) + catR/mirt 自适应测试标准
- Vygotsky ZPD + IRT-based ZPD 边界 + VARK+ZPD 集成模型
- 面向 L2 IRTState, 题目参数用 dict {"a", "b", "c"} 格式
- 引擎类无状态, 可安全多实例并发使用
"""

from __future__ import annotations

from dy3_polaris.l2.ability_assessor.irt import IRTEstimator
from dy3_polaris.l2.ability_assessor.cat import CATSelector
from dy3_polaris.l2.ability_assessor.zpd import ZPDCalculator, ZPDResult
from dy3_polaris.l2.ability_assessor.tracing_service import (
    IRTTracingService,
    AbilityOutput,
)


__all__ = [
    "IRTEstimator",
    "CATSelector",
    "ZPDCalculator",
    "ZPDResult",
    "IRTTracingService",
    "AbilityOutput",
]
