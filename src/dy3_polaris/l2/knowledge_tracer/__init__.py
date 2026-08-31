"""L2 知识追踪子模块 — BKT 知识追踪 / 掌握度传播 / 遗忘模型 / 序列特征工程.

子模块构成:
1. ``BKTTracer``: 贝叶斯知识追踪引擎
   - 标准 BKT 四参数模型 (Corbett & Anderson, 1995) + 前向算法
   - 难度 -> 先验 p_l0 映射 / O(1) 增量更新 / 批量历史重建 / 正确率预测
2. ``MasteryPropagator``: KG 驱动的掌握度传播器
   - 以前置知识点掌握度加性提升后继先验 (alpha=0.3, clamp [0,1])
   - GNN 式注意力加权多层传播 (propagate_gnn) / 注意力单跳传播
     (propagate_attention) / 异构图传播 (propagate_heterogeneous)
3. ``ForgettingModel``: 艾宾浩斯遗忘曲线模型
   - 指数衰减 (lambda=0.007/stability), 仅超 7 天 (168h) 触发衰减
   - 复习需求判定 (有效掌握度 < 阈值)
4. DKT 启发的序列特征工程 (sequence_features):
   - ``SequenceFeatures``: 序列特征数据类
   - ``SequenceFeatureExtractor``: 滑窗正确率 / 趋势 / 响应时间 /
     连续对错 / 掌握度速度 / 时序模式
   - ``TemporalPatternClassifier``: steady_bloom / late_bloom /
     early_decay / oscillating / stable

引擎均为无状态类, 不持有学习者状态, 适合并发复用.
依赖 L2 基础设施: ``TracingState`` / ``AnswerRecord`` / ``DEFAULT_BKT_PARAMS``.
"""

from dy3_polaris.l2.knowledge_tracer.bkt import BKTTracer
from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
from dy3_polaris.l2.knowledge_tracer.forgetting import ForgettingModel
from dy3_polaris.l2.knowledge_tracer.mastery_propagator import MasteryPropagator
from dy3_polaris.l2.knowledge_tracer.sequence_features import (
    SequenceFeatureExtractor,
    SequenceFeatures,
    TemporalPatternClassifier,
)
from dy3_polaris.l2.knowledge_tracer.tracing_service import (
    BKTTracingService,
    MasteryOutput,
)

__all__ = [
    "BKTTracer",
    "BKTTracingService",
    "EMCalibrator",
    "MasteryOutput",
    "MasteryPropagator",
    "ForgettingModel",
    "SequenceFeatures",
    "SequenceFeatureExtractor",
    "TemporalPatternClassifier",
]
