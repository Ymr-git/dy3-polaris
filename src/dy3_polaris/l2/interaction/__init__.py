"""L2 interaction 子模块 — 交互事件类型 / 事件采集器 / 更新管道.

负责 L2 个性化引擎的事件驱动入口:
1. 事件类型 (event_types): AnswerEvent / QueryEvent / BehaviorEvent
   - AnswerEvent  : 答题事件, 触发 BKT + IRT 实时更新
   - QueryEvent   : 查询事件, 推断学习兴趣
   - BehaviorEvent: 行为事件 (学习/复习/跳过)
2. 事件采集器 (collector): EventCollector
   - collect_answer() / validate() / collect_batch() (收集 / 验证 / 按 question_id 去重)
3. 更新管道 (pipeline): UpdatePipeline
   - process() / batch_process() (事件驱动的 BKT + IRT 实时画像更新, 依赖注入 + 优雅降级)

设计依据:
- xAPI (Experience API) / Caliper Analytics: 标准化学习事件
- Khan Academy: 事件驱动 BKT 实时更新 + 事件去重
- L2 规划文档 5.2 节: EventCollector -> BKTTracer -> IRTModel -> ProfileBuilder
"""

from dy3_polaris.l2.interaction.event_types import (
    VALID_ACTIONS,
    AnswerEvent,
    BehaviorEvent,
    QueryEvent,
)
from dy3_polaris.l2.interaction.collector import EventCollector
from dy3_polaris.l2.interaction.pipeline import UpdatePipeline

__all__ = [
    # 事件类型
    "AnswerEvent",
    "QueryEvent",
    "BehaviorEvent",
    "VALID_ACTIONS",
    # 采集器
    "EventCollector",
    # 更新管道
    "UpdatePipeline",
]
