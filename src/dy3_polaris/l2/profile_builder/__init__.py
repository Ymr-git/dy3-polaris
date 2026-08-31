"""L2 profile_builder 子模块 — 学情画像构建器.

子模块构成:
1. ``LevelEstimator``: 能力等级估计器
   - 融合 IRT theta + BKT avg_mastery 进行三级分级 (beginner/intermediate/advanced)
   - 优先级短路: beginner (最宽松) -> advanced (最严格) -> intermediate (默认)
2. ``StyleInferrer``: 学习风格推断器
   - infer_from_vark: 从 VARK 四维分数推断主要风格 (问卷优先)
   - infer_from_behavior: 从学习行为事件推断风格 (行为推断)
   - 多维接近 (差 < 0.05) 返回 "multimodal", 无数据默认 "reading"
3. ``BloomSetter``: Bloom 认知层次目标设定器
   - 默认目标比当前高一级 (Mastery Learning + ZPD)
   - 支持六层次: remember/understand/apply/analyze/evaluate/create
   - 已到最高级 (create) 则保持
4. ``ProfileBuilder``: 学情画像构建器
   - 组装 BKT/IRT/VARK/Bloom 综合画像
   - 依赖注入 store (可选), build 后自动保存画像快照
   - 返回 L2 LearnerSnapshot
5. ``LearnerColdStartManager``: 冷启动策略管理器 (面向学情画像)
   - 新学习者数据不足时的降级方案 (群体先验 → 部分个性化 → 全量个性化)
   - estimate_initial_theta / estimate_initial_mastery: 群体先验与观测值加权混合
   - 向后兼容别名 ``ColdStartManager`` (已弃用)
6. ``LearnerDriftDetector``: 概念漂移检测器 (面向学情画像)
   - ADWIN (自适应滑动窗口) + DDM (错误率均值/标准差) 双方法
   - 检测学习者行为模式变化, 建议触发模型重训练
   - 支持漂移触发回调 / 漂移历史记录
   - 向后兼容别名 ``ConceptDriftDetector`` (已弃用)
7. ``LearnerLifecycleManager``: 学习者生命周期管理器
   - 冷启动 → 暖启动 → 稳态 → 漂移 → 重训练 全生命周期管理
   - 每学习者独立漂移检测器 + 线程安全 + 漂移触发自动重训练
8. ``DriftAwareRetrainer``: 漂移感知重训练器
   - BKT 参数重训练 + 最小改善阈值回滚机制
9. ``ProfileTracingService``: 画像全链路编排服务
   - 冷启动 → 画像构建 → 风格推断 → Bloom设定 → 漂移检测 → 重训练
   - 依赖注入 store / profile_builder / cold_start_manager / drift_detector /
     lifecycle_manager, 为 None 时使用默认实例
   - 返回 ``ProfileOutput`` 标准化画像契约 (供下游 T2/T3/T4 消费)
10. ``ProfileOutput``: 画像全链路输出标准化契约
   - 包含 learner_id / phase / theta / level / learning_style / bloom_target /
     kp_mastery / weak_kps / confidence / drift_detected / drift_count /
     recommended_action / snapshot_ts

三个子引擎均为无状态类, 不持有学习者状态, 适合并发复用.
依赖 L2 基础设施: TracingState / IRTState / AnswerRecord / LearnerSnapshot / L2Store.
"""

from __future__ import annotations

from dy3_polaris.l2.profile_builder.bloom_setter import BloomSetter, BLOOM_LEVELS
from dy3_polaris.l2.profile_builder.builder import ProfileBuilder
from dy3_polaris.l2.profile_builder.cold_start import (
    ColdStartManager,
    LearnerColdStartManager,
)
from dy3_polaris.l2.profile_builder.drift_detector import (
    ConceptDriftDetector,
    LearnerDriftDetector,
)
from dy3_polaris.l2.profile_builder.lifecycle_manager import (
    DriftAwareRetrainer,
    LearnerLifecycleManager,
    LearnerLifecycleState,
)
from dy3_polaris.l2.profile_builder.level_estimator import LevelEstimator
from dy3_polaris.l2.profile_builder.style_inferrer import StyleInferrer
from dy3_polaris.l2.profile_builder.tracing_service import (
    ProfileOutput,
    ProfileTracingService,
)


__all__ = [
    "ProfileBuilder",
    "LevelEstimator",
    "StyleInferrer",
    "BloomSetter",
    "BLOOM_LEVELS",
    "LearnerColdStartManager",
    "ColdStartManager",  # 向后兼容别名 (已弃用)
    "LearnerDriftDetector",
    "ConceptDriftDetector",  # 向后兼容别名 (已弃用)
    "LearnerLifecycleManager",
    "LearnerLifecycleState",
    "DriftAwareRetrainer",
    "ProfileTracingService",
    "ProfileOutput",
]
