"""IRT 能力评估全链路编排服务.

融合世界先进方案:
- 贝叶斯 IRT (EAP): 在线单题后验更新 (Lord 1980 / Bock & Aitkin 1981)
- 最大似然估计 (MLE) + Newton-Raphson (Fisher scoring): 批量离线估计
- catR / mirt R 包: CAT 自适应测试标准 (最大 Fisher 信息准则选题)
- Vygotsky ZPD + IRT-based ZPD 边界: 三区分类与支架感知推荐
- Hierarchical Bayesian IRT: 多学习者分层收缩 (Empirical Bayes)
- 多模型 IRT (1PL/2PL/3PL/4PL): AIC/BIC 自动模型选择 (mirt)
- PSER 终止准则: 预测 SE 降低量决定是否继续 (catR)
- 渐进法/a-分层曝光控制: 平衡题目曝光 (Revuelta & Ponsoda 1998)
- ZPD 置信区间量化: Vygotsky + IRT CI 集成

全链路处理流程 (设计文档要求顺序):
1. IRT 估计: 取/初始化 IRT 状态 → 贝叶斯 EAP 后验更新 theta 与 SE
2. CAT 选题: 基于当前 theta 的最大 Fisher 信息准则选题 (可叠加 ZPD 感知)
3. ZPD 校准: 依据 theta 与题库对当前题目做三区分类, 推荐下次难度
4. 能力输出: 封装 theta / SE / 预测正确率 / ZPD 区 / 置信度 / 终止标志

增强模式 (enable_enhanced=True):
- 多模型 IRT: 支持 1PL/2PL/3PL/4PL 指定或自动选择
- 贝叶斯分层: 自适应收缩 toward 群体先验
- 可信区间: theta 的 95% 等尾可信区间
- ZPD 量化: 基于置信区间的 ZPD 宽度计算
- 自适应支架: 基于能力置信度的支架水平推荐
- PSER 终止: 预测 SE 降低量决定是否继续测试
- 模型比较: AIC/BIC 多模型比较报告

AbilityOutput 契约字段 (供下游 T2 画像 / T4 决策 / T5 反思消费):
- theta                : 当前能力估计 θ (画像着色 / 推荐定级)
- se                   : 估计标准误 (能力置信)
- p_correct_next       : 下一题预测答对概率 (CAT 选题)
- zpd_zone             : 当前题目所处 ZPD 区 (independent/zpd/frustration)
- recommended_difficulty: 下次推荐难度 b (推荐决策)
- confidence           : 能力置信度 1/(1+se) (预警置信)
- next_item_id         : CAT 选中的下一题 ID (选题引擎)
- termination_flag     : CAT 是否应终止 (终止控制)
- ci_lower             : 可信区间下界 (增强模式)
- ci_upper             : 可信区间上界 (增强模式)
- zpd_width            : ZPD 宽度 (增强模式)
- scaffold_level       : 推荐支架水平 (增强模式)
- irt_model            : 使用的 IRT 模型 (增强模式)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from dy3_polaris.l2.ability_assessor.cat import CATSelector
from dy3_polaris.l2.ability_assessor.irt import IRTEstimator
from dy3_polaris.l2.ability_assessor.zpd import ZPDCalculator
from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.models import (
    IRTState,
    DEFAULT_INITIAL_SE,
    DEFAULT_INITIAL_THETA,
    DEFAULT_IRT_A,
    DEFAULT_IRT_C,
)
from dy3_polaris.l2.store import InMemoryL2Store, L2Store


# ============================================================
# 模块级 logger
# ============================================================

import logging

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

# 默认先验能力 (群体均值): 无作答数据时回退 (统一到 models.py 单一事实来源)
DEFAULT_PRIOR_THETA: float = DEFAULT_INITIAL_THETA

# 默认先验标准误: 无作答数据时回退 (统一到 models.py 单一事实来源)
DEFAULT_PRIOR_SE: float = DEFAULT_INITIAL_SE

# 默认题目区分度 a (难度→IRT 参数映射, 统一到 models.py)
DEFAULT_DISCRIMINATION: float = DEFAULT_IRT_A

# 默认伪猜测下限 c (3PL 模型, 统一到 models.py)
DEFAULT_GUESSING: float = DEFAULT_IRT_C

# 默认支架水平 (ZPD 中心选题, 0=独立 / 1=最大挑战)
DEFAULT_SCAFFOLD_LEVEL: float = 0.5

# 默认群体先验标准差 (自适应收缩用)
DEFAULT_GROUP_SD: float = 1.0

# 默认可信区间置信水平
DEFAULT_CREDIBLE_LEVEL: float = 0.95


# ============================================================
# AbilityOutput — 下游输出标准化契约
# ============================================================


@dataclass
class AbilityOutput:
    """IRT 全链路输出 — 标准化能力契约.

    供下游 T2 画像着色 / T4 推荐决策 / T5 反思校验 / CAT 选题消费.

    基础字段 (始终可用):
        learner_id: 学习者 ID
        theta: 当前能力估计 θ (标准分尺度, 可正可负)
        se: 估计标准误 (standard error)
        response_count: 已纳入估计的作答次数
        p_correct_next: 下一题预测答对概率 [0.0, 1.0]
        zpd_zone: 当前题目所处 ZPD 区 ("independent" | "zpd" | "frustration")
        recommended_difficulty: 下次推荐难度 b 值
        confidence: 能力置信度 1/(1+se) [0.0, 1.0]
        next_item_id: CAT 选中的下一题 ID (无选题时为 None)
        termination_flag: CAT 是否应终止
        last_updated_ts: 最后更新时间戳 (秒, float)

    增强字段 (enable_enhanced=True 时可用):
        ci_lower: 可信区间下界 θ - z*SE
        ci_upper: 可信区间上界 θ + z*SE
        zpd_width: ZPD 宽度 (潜在发展水平 - 实际发展水平)
        scaffold_level: 推荐支架水平 ∈ [0.2, 0.9]
        irt_model: 使用的 IRT 模型 ("1PL" | "2PL" | "3PL" | "4PL")
    """

    learner_id: str
    theta: float
    se: float
    response_count: int
    p_correct_next: float
    zpd_zone: str
    recommended_difficulty: float
    confidence: float
    next_item_id: str | None
    termination_flag: bool
    last_updated_ts: float = 0.0
    # 增强字段 (默认 None, 增强模式时填充)
    ci_lower: float | None = None
    ci_upper: float | None = None
    zpd_width: float | None = None
    scaffold_level: float | None = None
    irt_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (含增强字段)."""
        return {
            "learner_id": self.learner_id,
            "theta": self.theta,
            "se": self.se,
            "response_count": self.response_count,
            "p_correct_next": self.p_correct_next,
            "zpd_zone": self.zpd_zone,
            "recommended_difficulty": self.recommended_difficulty,
            "confidence": self.confidence,
            "next_item_id": self.next_item_id,
            "termination_flag": self.termination_flag,
            "last_updated_ts": self.last_updated_ts,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "zpd_width": self.zpd_width,
            "scaffold_level": self.scaffold_level,
            "irt_model": self.irt_model,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AbilityOutput:
        """从字典反序列化 (增强字段缺失时回退 None)."""
        return cls(
            learner_id=d["learner_id"],
            theta=d["theta"],
            se=d["se"],
            response_count=d["response_count"],
            p_correct_next=d["p_correct_next"],
            zpd_zone=d["zpd_zone"],
            recommended_difficulty=d["recommended_difficulty"],
            confidence=d["confidence"],
            next_item_id=d.get("next_item_id"),
            termination_flag=d["termination_flag"],
            last_updated_ts=d.get("last_updated_ts", 0.0),
            ci_lower=d.get("ci_lower"),
            ci_upper=d.get("ci_upper"),
            zpd_width=d.get("zpd_width"),
            scaffold_level=d.get("scaffold_level"),
            irt_model=d.get("irt_model"),
        )

    def to_api_response(self) -> dict[str, Any]:
        """转换为 API 响应格式 — 包含能力等级与推荐信息.

        返回所有基础字段, 外加:
        - ability_level: 能力等级描述 (theta < -1 → "低",
          -1 ≤ theta ≤ 1 → "中", theta > 1 → "高")
        - recommendation: 推荐信息字典 (含 recommended_difficulty,
          next_item_id, zpd_zone)
        - 增强字段 (如果存在): ci_lower, ci_upper, zpd_width,
          scaffold_level, irt_model

        Returns:
            API 响应字典 (可 JSON 序列化).
        """
        # 能力等级描述
        if self.theta < -1.0:
            ability_level = "低"
        elif self.theta > 1.0:
            ability_level = "高"
        else:
            ability_level = "中"

        # 推荐信息
        recommendation: dict[str, Any] = {
            "recommended_difficulty": self.recommended_difficulty,
            "next_item_id": self.next_item_id,
            "zpd_zone": self.zpd_zone,
        }

        # 基础字段
        result: dict[str, Any] = {
            "learner_id": self.learner_id,
            "theta": self.theta,
            "se": self.se,
            "response_count": self.response_count,
            "p_correct_next": self.p_correct_next,
            "zpd_zone": self.zpd_zone,
            "recommended_difficulty": self.recommended_difficulty,
            "confidence": self.confidence,
            "next_item_id": self.next_item_id,
            "termination_flag": self.termination_flag,
            "last_updated_ts": self.last_updated_ts,
            "ability_level": ability_level,
            "recommendation": recommendation,
        }

        # 增强字段 (仅当非 None 时包含)
        if self.ci_lower is not None:
            result["ci_lower"] = self.ci_lower
        if self.ci_upper is not None:
            result["ci_upper"] = self.ci_upper
        if self.zpd_width is not None:
            result["zpd_width"] = self.zpd_width
        if self.scaffold_level is not None:
            result["scaffold_level"] = self.scaffold_level
        if self.irt_model is not None:
            result["irt_model"] = self.irt_model

        return result


# ============================================================
# IRTTracingService — 全链路编排器
# ============================================================


class IRTTracingService:
    """IRT 能力评估全链路编排服务 — 答题记录 → IRT 估计 → CAT 选题 → ZPD 校准 → 能力输出.

    编排全链路处理流程:
    1. IRT 估计: 取/初始化 IRT 状态, 贝叶斯 EAP 后验更新 theta 与 SE
    2. CAT 选题: 基于当前 theta 的最大 Fisher 信息准则选题
    3. ZPD 校准: 依据 theta 与题库对当前题目做三区分类, 推荐下次难度
    4. 能力输出: 封装 theta / SE / 预测正确率 / ZPD 区 / 置信度 / 终止标志

    支持特性:
    - 依赖注入: store / irt_estimator / cat_selector / zpd_calculator 均可注入
    - 优雅降级: store=None 时使用内部 InMemoryL2Store
    - 难度映射: 事件难度 [0,1] → IRT b [-3,3] (b = difficulty*6 - 3)
    - ZPD 集成: 默认 cat_selector 注入 zpd_calculator, 支持 zpd_aware 选题
    - 冷启动: 无作答数据回退群体先验 (theta=0.0, se=默认先验)

    增强模式 (enable_enhanced=True):
    - 多模型 IRT: 指定 irt_model 或自动选择最优模型
    - 自适应收缩: 少量数据向群体先验收缩 (Empirical Bayes)
    - 可信区间: 输出 theta 的置信区间
    - ZPD 量化: 基于置信区间的 ZPD 宽度
    - PSER 终止: 预测 SE 降低量决定是否继续
    - 模型比较: AIC/BIC 多模型比较报告

    Args:
        store: L2 存储层. 为 None 时使用内部 InMemoryL2Store.
        irt_estimator: IRT 估计器. 为 None 时创建默认 IRTEstimator.
        cat_selector: CAT 选题器. 为 None 时创建默认 CATSelector.
        zpd_calculator: ZPD 计算器. 为 None 时创建默认 ZPDCalculator.
        enable_enhanced: 是否启用增强模式 (多模型/贝叶斯/PSER/ZPD-CI).
        irt_model: 指定 IRT 模型 ("1PL"/"2PL"/"3PL"/"4PL"), None 时自动选择.
        adaptive_shrinkage: 是否启用自适应收缩 (增强模式).
        cat_termination_criteria: CAT 终止准则列表, 如 ["pser", "precision", "length"].

    Attributes:
        store: L2 存储层
        irt_estimator: IRT 估计引擎
        cat_selector: CAT 选题引擎
        zpd_calculator: ZPD 计算器
    """

    def __init__(
        self,
        store: L2Store | None = None,
        irt_estimator: IRTEstimator | None = None,
        cat_selector: CATSelector | None = None,
        zpd_calculator: ZPDCalculator | None = None,
        enable_enhanced: bool = False,
        irt_model: str | None = None,
        adaptive_shrinkage: bool = False,
        cat_termination_criteria: list[str] | None = None,
        enable_fusion: bool = False,
    ) -> None:
        self.store: L2Store = store if store is not None else InMemoryL2Store()
        self.irt_estimator: IRTEstimator = (
            irt_estimator if irt_estimator is not None else IRTEstimator()
        )
        self.zpd_calculator: ZPDCalculator = (
            zpd_calculator if zpd_calculator is not None else ZPDCalculator()
        )
        # 默认 cat_selector 注入 irt_estimator 与 zpd_calculator, 启用 ZPD 感知选题
        if cat_selector is not None:
            self.cat_selector: CATSelector = cat_selector
        else:
            self.cat_selector = CATSelector(
                estimator=self.irt_estimator,
                zpd_calculator=self.zpd_calculator,
            )
        # 题库 (CAT 标准键格式 {"item_id", "a", "b", "c"})
        self._item_bank: list[dict[str, Any]] = []
        # 当前能力追踪 (供 select_next_item 使用)
        self._current_theta: float = DEFAULT_PRIOR_THETA
        self._current_learner_id: str | None = None

        # --- 增强模式配置 ---
        self._enable_enhanced: bool = enable_enhanced
        self._irt_model: str | None = irt_model
        self._adaptive_shrinkage: bool = adaptive_shrinkage
        self._cat_termination_criteria: list[str] | None = cat_termination_criteria

        # --- 融合模式配置 ---
        self._enable_fusion: bool = enable_fusion

        # 作答历史 (供模型比较报告使用): {learner_id: [(item_params, correct), ...]}
        self._response_history: dict[str, list[tuple[dict[str, Any], bool]]] = {}

    # --- 题库设置 ---

    def set_item_bank(self, items: list[dict[str, Any]]) -> None:
        """设置题库 (供 CAT 选题与 ZPD 校准使用).

        Args:
            items: 题目列表, 每项为 CAT 标准键格式
                {"item_id", "a", "b", "c", "content_area"(可选)}.
        """
        self._item_bank = [dict(item) for item in items]

    # --- 单事件处理 ---

    def process(self, event: AnswerEvent) -> AbilityOutput:
        """处理单条答题事件, 返回完整 AbilityOutput.

        全链路流程:
        1. 取/初始化 IRT 状态 (默认 theta=0.0, se=默认先验)
        2. 事件难度 [0,1] → IRT b [-3,3], 构造 item_params {"a","b","c"}
        3. 贝叶斯 EAP 后验更新 theta 与 SE
        4. (增强) 自适应收缩 toward 群体先验
        5. 持久化更新后的 IRT 状态
        6. ZPD 三区分类 + 推荐下次难度
        7. (增强) 可信区间 + ZPD 量化 + 支架推荐
        8. 构建并返回 AbilityOutput

        Args:
            event: 答题事件.

        Returns:
            AbilityOutput 标准化输出.
        """
        # 1. 取/初始化 IRT 状态
        state = self._get_or_init_state(event.learner_id)

        # 2. 事件难度 → IRT 参数 (b = difficulty*6 - 3, 默认 a=1.2, c=0.25)
        item_params = self._event_to_item_params(event)

        # 3. 贝叶斯 EAP 后验更新
        updated = self.irt_estimator.update_theta(
            state, item_params, event.correct
        )
        # update_theta 不修改时间戳, 这里写入事件时间戳
        updated = replace(updated, last_update_time=event.timestamp)

        # 4. (增强) 自适应收缩
        if self._enable_enhanced and self._adaptive_shrinkage:
            updated = self._apply_adaptive_shrinkage(updated)

        # 5. 持久化
        self.store.save_irt_state(event.learner_id, updated)

        # 记录作答历史 (供模型比较报告)
        if self._enable_enhanced:
            history = self._response_history.setdefault(event.learner_id, [])
            history.append((dict(item_params), event.correct))

        # 更新当前能力追踪
        self._current_theta = updated.theta
        self._current_learner_id = event.learner_id

        # 6. ZPD 校准 (三区分类 + 推荐难度)
        zpd_zone = self.zpd_calculator.classify_item(
            updated.theta,
            item_params["b"],
            item_params["a"],
            item_params["c"],
        )
        recommended_b = self._recommend_difficulty(updated.theta)

        # 7. (增强) 可信区间 + ZPD 量化 + 支架推荐
        enhanced_fields: dict[str, Any] = {}
        if self._enable_enhanced:
            enhanced_fields = self._compute_enhanced_fields(updated)

        # 8. 构建输出
        return self._build_output(
            event, updated, item_params, zpd_zone, recommended_b, enhanced_fields
        )

    # --- 批量处理 ---

    def batch_process(
        self,
        events: list[AnswerEvent],
    ) -> list[AbilityOutput]:
        """批量处理答题事件 (按时间戳升序).

        Args:
            events: 答题事件列表.

        Returns:
            每个事件对应的 AbilityOutput 列表.
        """
        if not events:
            return []
        ordered = sorted(events, key=lambda e: e.timestamp)
        return [self.process(ev) for ev in ordered]

    # --- CAT 选题委托 ---

    def select_next_item(
        self,
        available_items: list[dict[str, Any]],
        administered_ids: set[str],
    ) -> dict[str, Any] | None:
        """委托 CATSelector 选题 (基于当前 theta).

        Args:
            available_items: 可用题目列表.
            administered_ids: 已答题目 ID 集合.

        Returns:
            选中的题目字典; 无可用题目返回 None.
        """
        return self.cat_selector.select_next(
            theta=self._current_theta,
            available_items=available_items,
            administered_ids=administered_ids,
        )

    # --- 融合模式选题 ---

    def select_next_item_fusion(
        self,
        available_items: list[dict[str, Any]],
        administered_ids: set[str],
        mastery_map: dict[str, float],
    ) -> dict[str, Any] | None:
        """融合模式选题 — 注入 BKT 掌握度后委托 CATSelector 选题.

        将 mastery_map 中的 p_mastery 注入到 available_items 中 (按 item_id
        匹配), 然后调用 cat_selector.select_next 进行融合选题.

        要求 cat_selector 的 selection_strategy 为 "bkt_irt_fusion" 才会
        使用融合逻辑; 否则退化为普通选题.

        Args:
            available_items: 可用题目列表.
            administered_ids: 已答题目 ID 集合.
            mastery_map: {item_id: p_mastery} 掌握度映射.

        Returns:
            选中的题目字典; 无可用题目返回 None.
        """
        # 注入 p_mastery 到题目字典
        injected_items: list[dict[str, Any]] = []
        for item in available_items:
            item_copy = dict(item)
            item_id = item_copy.get("item_id")
            if item_id is not None and item_id in mastery_map:
                item_copy["p_mastery"] = mastery_map[item_id]
            else:
                # 无掌握度数据时默认 0.5 (ZPD 区)
                item_copy.setdefault("p_mastery", 0.5)
            injected_items.append(item_copy)

        # 临时切换到融合选题策略
        original_strategy = self.cat_selector.selection_strategy
        self.cat_selector.selection_strategy = "bkt_irt_fusion"
        try:
            chosen = self.cat_selector.select_next(
                theta=self._current_theta,
                available_items=injected_items,
                administered_ids=administered_ids,
            )
        finally:
            self.cat_selector.selection_strategy = original_strategy

        return chosen

    def from_mastery_output(
        self,
        mastery_outputs: list[dict[str, Any]],
    ) -> dict[str, float]:
        """从 BKT MasteryOutput 列表构建 mastery_map.

        输入格式: [{"kp_id": "kp_1", "p_mastery": 0.5}, ...]
        输出: {"kp_1": 0.5, ...}

        Args:
            mastery_outputs: BKT 掌握度输出列表, 每项含 kp_id 和 p_mastery.

        Returns:
            {kp_id: p_mastery} 映射; 空输入返回空字典.
        """
        mastery_map: dict[str, float] = {}
        for output in mastery_outputs:
            kp_id = output.get("kp_id")
            p_mastery = output.get("p_mastery")
            if kp_id is not None and p_mastery is not None:
                mastery_map[kp_id] = float(p_mastery)
        return mastery_map

    # --- 能力估计 API (对应 /l2/irt/estimate) ---

    def estimate_ability(
        self,
        learner_id: str,
        events: list[AnswerEvent],
        mastery_map: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """能力估计 API — 处理事件序列并返回 API 响应格式字典.

        对应 /l2/irt/estimate API 端点. 处理事件序列:
        1. 按 time 排序, 逐条 process 事件更新能力估计
        2. 如果提供了 mastery_map 且启用了融合模式, 使用融合选题推荐下一题
        3. 返回包含 theta/se/zpd_zone/ability_level/recommendation 等的字典

        Args:
            learner_id: 学习者 ID.
            events: 答题事件列表.
            mastery_map: {item_id: p_mastery} 掌握度映射 (可选, 融合模式时使用).

        Returns:
            API 响应字典, 包含:
            - theta, se, response_count, p_correct_next, zpd_zone, confidence,
              termination_flag, last_updated_ts
            - ability_level: 能力等级 ("低"/"中"/"高")
            - recommendation: 推荐信息 (recommended_difficulty, next_item_id, zpd_zone)
            - next_item_id: 下一题 ID (融合选题或普通选题)
            - 增强字段 (如果启用增强模式)
        """
        if not events:
            # 空事件: 返回群体先验
            return {
                "learner_id": learner_id,
                "theta": DEFAULT_PRIOR_THETA,
                "se": DEFAULT_PRIOR_SE,
                "response_count": 0,
                "p_correct_next": 0.5,
                "zpd_zone": "zpd",
                "confidence": 1.0 / (1.0 + DEFAULT_PRIOR_SE),
                "next_item_id": None,
                "termination_flag": False,
                "last_updated_ts": 0.0,
                "ability_level": "中",
                "recommendation": {
                    "recommended_difficulty": DEFAULT_PRIOR_THETA,
                    "next_item_id": None,
                    "zpd_zone": "zpd",
                },
            }

        # 按时间戳排序处理事件
        ordered = sorted(events, key=lambda e: e.timestamp)
        last_output: AbilityOutput | None = None
        administered_ids: set[str] = set()

        for event in ordered:
            last_output = self.process(event)
            # 记录已答题目 ID
            if event.kp_id:
                administered_ids.add(event.kp_id)

        if last_output is None:
            return {
                "learner_id": learner_id,
                "theta": DEFAULT_PRIOR_THETA,
                "se": DEFAULT_PRIOR_SE,
                "response_count": 0,
                "p_correct_next": 0.5,
                "zpd_zone": "zpd",
                "confidence": 1.0 / (1.0 + DEFAULT_PRIOR_SE),
                "next_item_id": None,
                "termination_flag": False,
                "last_updated_ts": 0.0,
                "ability_level": "中",
                "recommendation": {
                    "recommended_difficulty": DEFAULT_PRIOR_THETA,
                    "next_item_id": None,
                    "zpd_zone": "zpd",
                },
            }

        # 获取 API 响应基础格式
        result = last_output.to_api_response()

        # 选题推荐
        if self._enable_fusion and mastery_map is not None:
            # 融合模式选题
            chosen = self.select_next_item_fusion(
                available_items=self._item_bank,
                administered_ids=administered_ids,
                mastery_map=mastery_map,
            )
            next_item_id = chosen.get("item_id") if chosen else None
        elif self._item_bank:
            # 普通选题
            chosen = self.select_next_item(
                available_items=self._item_bank,
                administered_ids=administered_ids,
            )
            next_item_id = chosen.get("item_id") if chosen else None
        else:
            next_item_id = None

        result["next_item_id"] = next_item_id
        # 更新 recommendation 中的 next_item_id
        if "recommendation" in result:
            result["recommendation"]["next_item_id"] = next_item_id

        return result

    # --- 终止条件委托 ---

    def should_stop(self, se: float, count: int) -> bool:
        """委托 CATSelector 判定终止条件.

        增强模式下使用 cat_termination_criteria 多准则组合终止;
        基础模式使用标准 length + precision 终止.

        Args:
            se: 当前能力估计标准误.
            count: 已作答题数.

        Returns:
            True 表示应终止测试, False 表示继续.
        """
        if self._enable_enhanced and self._cat_termination_criteria:
            return self.cat_selector.should_stop_multi(
                current_se=se,
                count=count,
                theta=self._current_theta,
                available_items=self._item_bank,
                administered_ids=set(),
                criteria=self._cat_termination_criteria,
            )
        return self.cat_selector.should_stop(se, count)

    # --- 能力快照 ---

    def get_ability_snapshot(self, learner_id: str) -> dict[str, Any]:
        """获取学习者当前能力状态快照.

        无作答数据时回退群体先验 (theta=0.0, se=默认先验).

        Args:
            learner_id: 学习者 ID.

        Returns:
            ``{learner_id, theta, se, response_count, last_update_time}``.
        """
        state = self.store.get_irt_state(learner_id)
        if state is None:
            state = IRTState(
                theta=DEFAULT_PRIOR_THETA,
                se=DEFAULT_PRIOR_SE,
                response_count=0,
                last_update_time=0.0,
            )
        return {
            "learner_id": learner_id,
            "theta": state.theta,
            "se": state.se,
            "response_count": state.response_count,
            "last_update_time": state.last_update_time,
        }

    # --- 模型比较报告 (增强模式) ---

    def get_model_comparison_report(self, learner_id: str) -> dict[str, dict[str, float]] | None:
        """获取学习者的多模型 IRT 比较报告 (AIC/BIC).

        基于该学习者的作答历史, 对 1PL/2PL/3PL/4PL 四个模型计算 AIC/BIC,
        供模型选择参考.

        Args:
            learner_id: 学习者 ID.

        Returns:
            {model: {aic, bic, loglik, n_params, theta}} 映射;
            无作答历史或非增强模式返回 None.
        """
        if not self._enable_enhanced:
            return None
        history = self._response_history.get(learner_id)
        if not history or len(history) < 2:
            return None
        return self.irt_estimator.compare_models(history)

    # ============================================================
    # 内部方法
    # ============================================================

    def _get_or_init_state(self, learner_id: str) -> IRTState:
        """获取当前 IRT 状态, 无记录时使用群体先验初始化.

        Args:
            learner_id: 学习者 ID.

        Returns:
            当前 (或初始化的) IRTState.
        """
        state = self.store.get_irt_state(learner_id)
        if state is not None:
            return state
        return IRTState(
            theta=DEFAULT_PRIOR_THETA,
            se=DEFAULT_PRIOR_SE,
            response_count=0,
            last_update_time=0.0,
        )

    @staticmethod
    def _event_to_item_params(event: AnswerEvent) -> dict[str, float]:
        """将答题事件难度映射为 IRT 题目参数.

        映射规则:
        - 难度 [0,1] → IRT b [-3,3]: b = difficulty * 6 - 3
        - 默认区分度 a = 1.2
        - 默认伪猜测下限 c = 0.25

        Args:
            event: 答题事件.

        Returns:
            {"a": 1.2, "b": difficulty*6-3, "c": 0.25}.
        """
        b = event.difficulty * 6.0 - 3.0
        # 钳制到 [-3, 3] (IRT b 与 theta 同尺度)
        b = max(-3.0, min(3.0, b))
        return {
            "a": DEFAULT_DISCRIMINATION,
            "b": b,
            "c": DEFAULT_GUESSING,
        }

    def _apply_adaptive_shrinkage(self, state: IRTState) -> IRTState:
        """对 EAP 更新后的状态应用自适应收缩.

        收缩公式 (Empirical Bayes):
            λ = σ² / (τ² + σ²)
            θ_shrunk = (1-λ) * θ_eap + λ * μ_group

        其中 σ² = SE² (学习者内方差), τ² = group_sd² (群体间方差).
        数据少 (SE 大) → λ 大 → 强收缩; 数据多 (SE 小) → λ 小 → 弱收缩.

        **重要**: SE 不被收缩 (不向群体 SD 混合). EAP 后验 SE 已反映数据与
        先验的综合信息量, 应随数据累积单调递减; 若将其向群体 SD (DEFAULT_GROUP_SD=1.0)
        混合, 则少量数据时 SE 会被膨胀至 ~1.0, 阻止 SE 随数据累积下降.
        正确做法是仅收缩 θ (减少极端估计), SE 保持 EAP 后验值不变.

        Args:
            state: EAP 更新后的 IRT 状态.

        Returns:
            收缩后的 IRT 状态 (仅 theta 被收缩, SE 保持 EAP 后验值).
        """
        mu_group = DEFAULT_PRIOR_THETA
        sd_group = DEFAULT_GROUP_SD
        tau_sq = max(sd_group * sd_group, 1e-10)
        sigma_sq = max(state.se * state.se, 1e-10)
        lam = sigma_sq / (tau_sq + sigma_sq)
        shrunk_theta = (1.0 - lam) * state.theta + lam * mu_group
        return replace(state, theta=shrunk_theta)

    def _compute_enhanced_fields(self, state: IRTState) -> dict[str, Any]:
        """计算增强字段: 可信区间, ZPD 量化, 支架推荐, IRT 模型.

        Args:
            state: 更新后的 IRT 状态.

        Returns:
            含 ci_lower, ci_upper, zpd_width, scaffold_level, irt_model 的字典.
        """
        # 可信区间 (正态近似等尾区间)
        ci_result = self.irt_estimator.estimate_with_credible_interval(
            self._response_history.get(self._current_learner_id, []),
            credible_level=DEFAULT_CREDIBLE_LEVEL,
        )
        ci_lower = ci_result["ci_lower"]
        ci_upper = ci_result["ci_upper"]

        # ZPD 置信区间量化
        zpd_ci = self.zpd_calculator.calculate_zpd_ci(
            theta=state.theta,
            se=state.se,
            confidence_level=DEFAULT_CREDIBLE_LEVEL,
        )
        zpd_width = zpd_ci["zpd_width"]

        # 自适应支架推荐
        scaffold_level = self.zpd_calculator.recommend_scaffold_level(
            theta=state.theta,
            se=state.se,
            item_bank=self._item_bank if self._item_bank else None,
        )

        # IRT 模型 (指定或自动选择)
        if self._irt_model is not None:
            irt_model = self._irt_model
        else:
            history = self._response_history.get(self._current_learner_id, [])
            if len(history) >= 4:
                irt_model = self.irt_estimator.select_best_model(history)
            else:
                irt_model = "3PL"  # 默认模型

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "zpd_width": zpd_width,
            "scaffold_level": scaffold_level,
            "irt_model": irt_model,
        }

    def _recommend_difficulty(self, theta: float) -> float:
        """推荐下次学习难度 (结合题库与支架水平).

        若已设置题库, 使用 ZPDCalculator 在 ZPD 区间内按支架水平取目标难度;
        否则回退到 theta + 0.5 * scaffold_level.

        Args:
            theta: 当前能力估计 θ.

        Returns:
            推荐难度 b 值.
        """
        if self._item_bank:
            zpd_bank = CATSelector._to_zpd_items(self._item_bank)
            return self.zpd_calculator.recommend_difficulty(
                theta, DEFAULT_SCAFFOLD_LEVEL, zpd_bank
            )
        return self.zpd_calculator.recommend_difficulty(
            theta, DEFAULT_SCAFFOLD_LEVEL, None
        )

    def _build_output(
        self,
        event: AnswerEvent,
        state: IRTState,
        item_params: dict[str, float],
        zpd_zone: str,
        recommended_b: float,
        enhanced_fields: dict[str, Any] | None = None,
    ) -> AbilityOutput:
        """构建 AbilityOutput 标准化输出.

        Args:
            event: 原始答题事件 (提供 learner_id / timestamp).
            state: 更新后的 IRT 状态.
            item_params: 本次题目 IRT 参数 {"a","b","c"}.
            zpd_zone: 当前题目 ZPD 区分类.
            recommended_b: 推荐下次难度 b.
            enhanced_fields: 增强字段字典 (增强模式时传入).

        Returns:
            AbilityOutput.
        """
        # 下一题预测答对概率 (以推荐难度为下一题难度)
        p_correct_next = self.irt_estimator.predict_correct(
            state.theta,
            DEFAULT_DISCRIMINATION,
            recommended_b,
            DEFAULT_GUESSING,
        )
        # 能力置信度 1/(1+se)
        confidence = 1.0 / (1.0 + max(state.se, 0.0))

        # CAT 终止标志 (增强模式使用多准则)
        if self._enable_enhanced and self._cat_termination_criteria:
            termination_flag = self.cat_selector.should_stop_multi(
                current_se=state.se,
                count=state.response_count,
                theta=state.theta,
                available_items=self._item_bank,
                administered_ids=set(),
                criteria=self._cat_termination_criteria,
            )
        else:
            termination_flag = self.cat_selector.should_stop(
                state.se, state.response_count
            )

        # 构建输出 (合并增强字段)
        output_kwargs: dict[str, Any] = {
            "learner_id": event.learner_id,
            "theta": state.theta,
            "se": state.se,
            "response_count": state.response_count,
            "p_correct_next": p_correct_next,
            "zpd_zone": zpd_zone,
            "recommended_difficulty": recommended_b,
            "confidence": confidence,
            "next_item_id": None,
            "termination_flag": termination_flag,
            "last_updated_ts": event.timestamp,
        }
        if enhanced_fields:
            output_kwargs.update(enhanced_fields)

        return AbilityOutput(**output_kwargs)


# ============================================================
# __all__
# ============================================================

__all__ = [
    "IRTTracingService",
    "AbilityOutput",
    "DEFAULT_PRIOR_THETA",
    "DEFAULT_PRIOR_SE",
    "DEFAULT_DISCRIMINATION",
    "DEFAULT_GUESSING",
]
