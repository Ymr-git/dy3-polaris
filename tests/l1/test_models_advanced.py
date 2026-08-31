"""L1 用户域核心数据模型高级增强测试 — TDD RED 阶段.

基于深度差距分析与世界先进方案研究，覆盖以下增强领域:

A. FSRS 间隔重复调度器 (Free Spaced Repetition Scheduler)
   - FSRSParameters: 19 参数模型 (w0-w18)
   - FSRSCardState: 卡片记忆状态 (stability/difficulty/retrievability/state)
   - FSRSReviewLog: 复习日志 (grade/elapsed_days/state_transition)

B. IRT 项目反应理论 (Item Response Theory)
   - IRTModel: 模型类型枚举 (1PL/2PL/3PL)
   - IRTItem: 题目参数 (difficulty_b/discrimination_a/guessing_c)
   - IRTAbility: 能力参数 (theta/standard_error)
   - 信息函数: I(theta) = a² · P(θ) · (1 - P(θ))

C. VARK 学习风格模型
   - VARKStyle: 四种模态枚举 (V/A/R/K)
   - VARKProfile: 学习风格画像 (四维分数 + 主导风格 + 置信度)
   - ContentModality: 内容模态标签

D. 认知负荷三分模型 (Cognitive Load Theory)
   - CognitiveLoadBreakdown: ICL + ECL + GCL 三分量
   - ElementInteractivity: 元素交互度计算

E. Bloom 2D 分类法 (Anderson & Krathwohl 修订版)
   - KnowledgeType: 四类知识维度 (Factual/Conceptual/Procedural/Metacognitive)
   - BloomTag: 2D 矩阵标签 (认知层级 × 知识类型)

F. 跨层接口模型 (设计文档第八章)
   - BKTUpdate (L2→L1)
   - MemoryEntry (L1→L2)
   - DecayRequest (L1→L2)
   - AccessCheck (L1→L3)
   - ResourceRequest (L1→L3)
   - KnowledgeResult (L3→L1)
   - PrivacyEvent (L1→L0)
   - PolicyUpdate (L0→L1)

G. 隐私保护执行模型
   - DesensitizationMethod: 脱敏方法枚举
   - PrivacyConfig: 隐私配置 (k-anonymity/l-diversity/differential-privacy)
   - RetentionPhase: 数据保留阶段枚举
   - RetentionPolicy: 保留策略

H. 学习分析事件 (xAPI / Caliper 兼容)
   - LearningEvent: 统一学习事件模型
   - EventResult: 事件结果
   - EventContext: 事件上下文

I. 参与度指标
   - EngagementLevel: 参与度等级
   - EngagementMetrics: 三维参与度 (行为/认知/情感)

J. 会话分析
   - SessionAnalytics: 会话聚合分析

K. 学习路径数据结构
   - LearningPath: 学习路径
   - PathNode: 路径节点
   - PathRecommendation: 路径推荐

L. 序列化补全 (from_dict)
   - Role / ApprovalRequest / FeedbackReport / ProvenanceRecord

M. 验证补全
   - 所有边界检查与校验

N. 跨层对齐增强
   - ContextEnvelope.to_l3_learner_profile() 完整字段填充
   - 新增反向转换方法

设计依据:
- L1 设计文档第八章: 跨层接口数据结构
- L1 设计文档第六章: 隐私保护与数据治理
- L1 设计文档第三章: 学习上下文经纪 (VARK/认知负荷)
- FSRS-6 算法规范 (open-spaced-repetition)
- Anderson & Krathwohl Bloom 修订版 (2001)
- IRT 1PL/2PL/3PL 标准模型
- xAPI (IEEE 9274.1.1-2023) / Caliper Analytics v1.2
- 差分隐私 (Dwork, 2006) 在教育数据挖掘中的应用
"""

from __future__ import annotations

import math
import time
from typing import Any

import pytest

from dy3_polaris.l1.models import (
    # 常量
    MS_PER_HOUR,
    MS_PER_SEC,
    MIN_STABILITY,
    STABILITY_GAIN,
    PRIOR_PROB,
    WEAK_THRESHOLD,
    EMERGENCY_THRESHOLD,
    BLOCK_THRESHOLD,
    WARNING_THRESHOLD,
    K_ANONYMITY_MIN,
    L_DIVERSITY_MIN,
    MAX_DAILY_AGENT_CALLS,
    # 现有模型
    UserRole,
    UserStatus,
    Permission,
    Role,
    GradeLevel,
    MajorDirection,
    LabAccessTier,
    ABACAttributes,
    User,
    LearningPhase,
    MasterySnapshot,
    BKTParams,
    LearningGoal,
    LearningState,
    ResourceItem,
    TimeConstraint,
    ContextEnvelope,
    LearningContext,
    SessionType,
    SessionStatus,
    AgentState,
    Interaction,
    SessionArtifact,
    LearningSession,
    SessionFork,
    SessionCheckpoint,
    DataLevel,
    AuditAction,
    AuditResult,
    AuditLogEntry,
    HiTLType,
    HiTLPriority,
    ConfidenceGateResult,
    FeedbackType,
    ApprovalDecision,
    FeedbackCategory,
    AlertType,
    ApprovalRequest,
    ApprovalResponse,
    FeedbackReport,
    EmergencyAlert,
    ProvenanceRecord,
)


# ============================================================
# A. FSRS 间隔重复调度器测试
# ============================================================


class TestFSRSParameters:
    """FSRS 参数模型 — 21 参数权重 (w0-w20, FSRS-6)."""

    def test_create_default_parameters(self):
        """FSRS 默认参数包含 21 个权重."""
        from dy3_polaris.l1.models import FSRSParameters

        params = FSRSParameters()
        assert len(params.weights) == 21
        assert params.request_retention == 0.9
        assert params.maximum_interval > 0

    def test_initial_stability_from_rating(self):
        """S0(G) = w[G-1], 不同评分对应不同初始稳定性."""
        from dy3_polaris.l1.models import FSRSParameters

        params = FSRSParameters()
        s0_again = params.initial_stability(grade=1)
        s0_easy = params.initial_stability(grade=4)
        assert s0_again < s0_easy  # "again" 的初始稳定性低于 "easy"

    def test_initial_difficulty_from_rating(self):
        """D0(G) = w4 - exp(w5*(G-1)) + 1, 评分越高难度越低."""
        from dy3_polaris.l1.models import FSRSParameters

        params = FSRSParameters()
        d0_again = params.initial_difficulty(grade=1)
        d0_easy = params.initial_difficulty(grade=4)
        assert d0_again > d0_easy

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import FSRSParameters

        params = FSRSParameters()
        d = params.to_dict()
        restored = FSRSParameters.from_dict(d)
        assert restored.weights == params.weights
        assert restored.request_retention == params.request_retention


class TestFSRSCardState:
    """FSRS 卡片状态 — 记忆稳定性/难度/可提取性/状态."""

    def test_create_new_card(self):
        """新卡片: stability=0, difficulty=0, state=NEW."""
        from dy3_polaris.l1.models import FSRSCardState, FSRSCardState as State

        card = FSRSCardState(kc_id="kc-001")
        assert card.kc_id == "kc-001"
        assert card.state == State.NEW
        assert card.stability == 0.0
        assert card.difficulty == 0.0
        assert card.reps == 0
        assert card.lapses == 0

    def test_retrievability_calculation(self):
        """可提取性 R = (1 + factor * t/S)^(-DECAY)."""
        from dy3_polaris.l1.models import FSRSCardState, FSRSCardState as State

        card = FSRSCardState(
            kc_id="kc-001",
            stability=10.0,
            state=State.REVIEW,
            last_review_ts=int(time.time() * 1000),
        )
        # 刚复习完, t=0, R 应接近 1.0
        r = card.retrievability(current_ts=card.last_review_ts)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_retrievability_decays_over_time(self):
        """经过时间越长, 可提取性越低."""
        from dy3_polaris.l1.models import FSRSCardState, FSRSCardState as State

        now = int(time.time() * 1000)
        card = FSRSCardState(
            kc_id="kc-001",
            stability=10.0,
            state=State.REVIEW,
            last_review_ts=now,
        )
        r_now = card.retrievability(current_ts=now)
        r_future = card.retrievability(current_ts=now + 7 * 24 * 3600 * 1000)
        assert r_future < r_now

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import FSRSCardState, FSRSCardState as State

        card = FSRSCardState(
            kc_id="kc-001",
            stability=5.5,
            difficulty=7.2,
            state=State.REVIEW,
            reps=3,
            lapses=1,
            last_review_ts=int(time.time() * 1000),
        )
        d = card.to_dict()
        restored = FSRSCardState.from_dict(d)
        assert restored.kc_id == card.kc_id
        assert restored.stability == card.stability
        assert restored.difficulty == card.difficulty
        assert restored.state == card.state
        assert restored.reps == card.reps


class TestFSRSReviewLog:
    """FSRS 复习日志 — 记录每次复习的评分与状态转换."""

    def test_create_review_log(self):
        from dy3_polaris.l1.models import FSRSReviewLog

        log = FSRSReviewLog(
            kc_id="kc-001",
            grade=3,
            elapsed_days=2.5,
            state_before="review",
            state_after="review",
        )
        assert log.kc_id == "kc-001"
        assert log.grade == 3
        assert log.elapsed_days == 2.5
        assert log.review_id  # 自动生成

    def test_grade_range(self):
        """评分范围 1-4 (again/hard/good/easy)."""
        from dy3_polaris.l1.models import FSRSReviewLog

        with pytest.raises(ValueError):
            FSRSReviewLog(kc_id="kc", grade=0, elapsed_days=0)
        with pytest.raises(ValueError):
            FSRSReviewLog(kc_id="kc", grade=5, elapsed_days=0)


# ============================================================
# B. IRT 项目反应理论测试
# ============================================================


class TestIRTModel:
    """IRT 模型类型枚举."""

    def test_three_models(self):
        from dy3_polaris.l1.models import IRTModel

        assert IRTModel.ONE_PL
        assert IRTModel.TWO_PL
        assert IRTModel.THREE_PL


class TestIRTItem:
    """IRT 题目参数 (difficulty_b / discrimination_a / guessing_c)."""

    def test_create_1pl_item(self):
        """1PL (Rasch): 仅 difficulty 参数."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(item_id="q-001", model_type=IRTModel.ONE_PL, difficulty_b=0.5)
        assert item.model_type == IRTModel.ONE_PL
        assert item.difficulty_b == 0.5
        assert item.discrimination_a == 1.0  # 1PL 固定为 1.0
        assert item.guessing_c == 0.0  # 1PL 无猜测参数

    def test_create_2pl_item(self):
        """2PL: difficulty + discrimination."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(
            item_id="q-002",
            model_type=IRTModel.TWO_PL,
            difficulty_b=-0.3,
            discrimination_a=1.5,
        )
        assert item.model_type == IRTModel.TWO_PL
        assert item.guessing_c == 0.0

    def test_create_3pl_item(self):
        """3PL: difficulty + discrimination + guessing."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(
            item_id="q-003",
            model_type=IRTModel.THREE_PL,
            difficulty_b=1.0,
            discrimination_a=1.2,
            guessing_c=0.25,
        )
        assert item.guessing_c == 0.25

    def test_probability_1pl(self):
        """1PL: P(θ) = 1 / (1 + e^(-(θ - b)))."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(item_id="q", model_type=IRTModel.ONE_PL, difficulty_b=0.0)
        # θ = b 时 P = 0.5
        p = item.probability(theta=0.0)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_probability_2pl(self):
        """2PL: P(θ) = 1 / (1 + e^(-a*(θ - b)))."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(
            item_id="q",
            model_type=IRTModel.TWO_PL,
            difficulty_b=0.0,
            discrimination_a=1.0,
        )
        p = item.probability(theta=0.0)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_probability_3pl(self):
        """3PL: P(θ) = c + (1-c) / (1 + e^(-a*(θ - b)))."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(
            item_id="q",
            model_type=IRTModel.THREE_PL,
            difficulty_b=0.0,
            discrimination_a=1.0,
            guessing_c=0.25,
        )
        # θ -> -∞ 时 P -> c (猜测概率下限)
        p_low = item.probability(theta=-10.0)
        assert p_low == pytest.approx(0.25, abs=0.01)

    def test_information_function(self):
        """信息函数 I(θ) = a² · P(θ) · (1 - P(θ))."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(
            item_id="q",
            model_type=IRTModel.TWO_PL,
            difficulty_b=0.0,
            discrimination_a=1.5,
        )
        # θ = b 时信息量最大
        info = item.information(theta=0.0)
        assert info > 0
        # 远离 b 时信息量减小
        info_far = item.information(theta=3.0)
        assert info_far < info

    def test_validation_difficulty_range(self):
        """难度 b 通常在 [-3, +3] 范围."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        with pytest.raises(ValueError):
            IRTItem(item_id="q", model_type=IRTModel.ONE_PL, difficulty_b=5.0)

    def test_validation_discrimination_positive(self):
        """区分度 a 必须为正数."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        with pytest.raises(ValueError):
            IRTItem(
                item_id="q",
                model_type=IRTModel.TWO_PL,
                difficulty_b=0.0,
                discrimination_a=-1.0,
            )

    def test_validation_guessing_range(self):
        """猜测参数 c 在 [0, 0.5] 范围."""
        from dy3_polaris.l1.models import IRTItem, IRTModel

        with pytest.raises(ValueError):
            IRTItem(
                item_id="q",
                model_type=IRTModel.THREE_PL,
                difficulty_b=0.0,
                discrimination_a=1.0,
                guessing_c=0.8,
            )

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import IRTItem, IRTModel

        item = IRTItem(
            item_id="q-003",
            model_type=IRTModel.THREE_PL,
            difficulty_b=1.0,
            discrimination_a=1.2,
            guessing_c=0.25,
        )
        d = item.to_dict()
        restored = IRTItem.from_dict(d)
        assert restored.model_type == item.model_type
        assert restored.difficulty_b == item.difficulty_b


class TestIRTAbility:
    """IRT 能力参数 (theta / standard_error)."""

    def test_create_ability(self):
        from dy3_polaris.l1.models import IRTAbility

        ability = IRTAbility(user_id="u-001", theta=0.5, standard_error=0.3)
        assert ability.user_id == "u-001"
        assert ability.theta == 0.5
        assert ability.standard_error == 0.3

    def test_theta_range(self):
        """能力 θ 通常在 [-3, +3] 范围."""
        from dy3_polaris.l1.models import IRTAbility

        with pytest.raises(ValueError):
            IRTAbility(user_id="u", theta=5.0, standard_error=0.3)

    def test_confidence_interval(self):
        """95% CI = θ ± 1.96 * SE."""
        from dy3_polaris.l1.models import IRTAbility

        ability = IRTAbility(user_id="u", theta=0.5, standard_error=0.2)
        ci = ability.confidence_interval_95()
        assert ci[0] < 0.5 < ci[1]

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import IRTAbility

        ability = IRTAbility(user_id="u-001", theta=1.0, standard_error=0.25)
        d = ability.to_dict()
        restored = IRTAbility.from_dict(d)
        assert restored.theta == ability.theta


# ============================================================
# C. VARK 学习风格模型测试
# ============================================================


class TestVARKStyle:
    """VARK 学习风格枚举."""

    def test_four_styles(self):
        from dy3_polaris.l1.models import VARKStyle

        assert VARKStyle.VISUAL
        assert VARKStyle.AURAL
        assert VARKStyle.READ_WRITE
        assert VARKStyle.KINESTHETIC


class TestVARKProfile:
    """VARK 学习风格画像."""

    def test_create_profile(self):
        from dy3_polaris.l1.models import VARKProfile, VARKStyle

        profile = VARKProfile(
            user_id="u-001",
            visual_score=0.8,
            aural_score=0.3,
            read_write_score=0.6,
            kinesthetic_score=0.4,
        )
        assert profile.user_id == "u-001"
        assert profile.visual_score == 0.8
        assert profile.primary_style == VARKStyle.VISUAL

    def test_primary_style_detection(self):
        """主导风格 = 最高分的模态."""
        from dy3_polaris.l1.models import VARKProfile, VARKStyle

        profile = VARKProfile(
            user_id="u",
            visual_score=0.2,
            aural_score=0.1,
            read_write_score=0.9,
            kinesthetic_score=0.3,
        )
        assert profile.primary_style == VARKStyle.READ_WRITE

    def test_multimodal_detection(self):
        """多模态: 多个维度分数接近最高分时标记为 MULTIMODAL."""
        from dy3_polaris.l1.models import VARKProfile, VARKStyle

        profile = VARKProfile(
            user_id="u",
            visual_score=0.7,
            aural_score=0.72,
            read_write_score=0.3,
            kinesthetic_score=0.1,
        )
        assert profile.primary_style == VARKStyle.MULTIMODAL

    def test_score_range_validation(self):
        """所有分数在 [0.0, 1.0] 范围."""
        from dy3_polaris.l1.models import VARKProfile

        with pytest.raises(ValueError):
            VARKProfile(
                user_id="u",
                visual_score=1.5,
                aural_score=0.3,
                read_write_score=0.3,
                kinesthetic_score=0.3,
            )

    def test_confidence_default(self):
        """默认置信度为 0.0 (未评估)."""
        from dy3_polaris.l1.models import VARKProfile

        profile = VARKProfile(
            user_id="u",
            visual_score=0.5,
            aural_score=0.5,
            read_write_score=0.5,
            kinesthetic_score=0.5,
        )
        assert profile.confidence == 0.0

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import VARKProfile

        profile = VARKProfile(
            user_id="u-001",
            visual_score=0.8,
            aural_score=0.3,
            read_write_score=0.6,
            kinesthetic_score=0.4,
            confidence=0.85,
        )
        d = profile.to_dict()
        restored = VARKProfile.from_dict(d)
        assert restored.visual_score == profile.visual_score
        assert restored.confidence == profile.confidence


class TestContentModality:
    """内容模态标签."""

    def test_create_modality(self):
        from dy3_polaris.l1.models import ContentModality, VARKStyle

        modality = ContentModality(
            content_id="res-001",
            modality_tags=[VARKStyle.VISUAL, VARKStyle.READ_WRITE],
        )
        assert VARKStyle.VISUAL in modality.modality_tags
        assert len(modality.modality_tags) == 2

    def test_to_dict(self):
        from dy3_polaris.l1.models import ContentModality, VARKStyle

        modality = ContentModality(
            content_id="res-001",
            modality_tags=[VARKStyle.VISUAL],
        )
        d = modality.to_dict()
        assert "visual" in d["modality_tags"]


# ============================================================
# D. 认知负荷三分模型测试
# ============================================================


class TestCognitiveLoadBreakdown:
    """认知负荷三分模型: ICL + ECL + GCL."""

    def test_create_breakdown(self):
        from dy3_polaris.l1.models import CognitiveLoadBreakdown

        breakdown = CognitiveLoadBreakdown(
            intrinsic_load=0.4,
            extraneous_load=0.2,
            germane_load=0.3,
        )
        assert breakdown.total_load == pytest.approx(0.9, abs=0.01)

    def test_total_load_additive(self):
        """总负荷 = ICL + ECL + GCL."""
        from dy3_polaris.l1.models import CognitiveLoadBreakdown

        breakdown = CognitiveLoadBreakdown(
            intrinsic_load=0.3,
            extraneous_load=0.1,
            germane_load=0.2,
        )
        assert breakdown.total_load == pytest.approx(0.6, abs=0.01)

    def test_load_range_validation(self):
        """每个分量在 [0.0, 1.0] 范围."""
        from dy3_polaris.l1.models import CognitiveLoadBreakdown

        with pytest.raises(ValueError):
            CognitiveLoadBreakdown(
                intrinsic_load=1.5,
                extraneous_load=0.1,
                germane_load=0.2,
            )

    def test_is_overloaded(self):
        """总负荷 >= EMERGENCY_THRESHOLD 时标记为过载."""
        from dy3_polaris.l1.models import CognitiveLoadBreakdown

        breakdown = CognitiveLoadBreakdown(
            intrinsic_load=0.4,
            extraneous_load=0.35,
            germane_load=0.25,
        )
        assert breakdown.is_overloaded()

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import CognitiveLoadBreakdown

        breakdown = CognitiveLoadBreakdown(
            intrinsic_load=0.4,
            extraneous_load=0.2,
            germane_load=0.3,
        )
        d = breakdown.to_dict()
        restored = CognitiveLoadBreakdown.from_dict(d)
        assert restored.intrinsic_load == breakdown.intrinsic_load
        assert restored.total_load == breakdown.total_load


class TestElementInteractivity:
    """元素交互度计算."""

    def test_create_interactivity(self):
        from dy3_polaris.l1.models import ElementInteractivity

        ei = ElementInteractivity(
            element_count=5,
            interaction_count=10,
        )
        assert ei.interactivity_ratio == pytest.approx(2.0, abs=0.01)

    def test_low_interactivity(self):
        """低交互度 = 元素少且独立."""
        from dy3_polaris.l1.models import ElementInteractivity

        ei = ElementInteractivity(element_count=3, interaction_count=1)
        assert ei.interactivity_ratio < 1.0

    def test_high_interactivity(self):
        """高交互度 = 元素多且相互依赖."""
        from dy3_polaris.l1.models import ElementInteractivity

        ei = ElementInteractivity(element_count=4, interaction_count=12)
        assert ei.interactivity_ratio > 2.0

    def test_to_dict(self):
        from dy3_polaris.l1.models import ElementInteractivity

        ei = ElementInteractivity(element_count=5, interaction_count=10)
        d = ei.to_dict()
        assert d["interactivity_ratio"] == pytest.approx(2.0, abs=0.01)


# ============================================================
# E. Bloom 2D 分类法测试
# ============================================================


class TestKnowledgeType:
    """知识类型枚举 (Anderson & Krathwohl 四类)."""

    def test_four_types(self):
        from dy3_polaris.l1.models import KnowledgeType

        assert KnowledgeType.FACTUAL
        assert KnowledgeType.CONCEPTUAL
        assert KnowledgeType.PROCEDURAL
        assert KnowledgeType.METACOGNITIVE


class TestBloomTag:
    """Bloom 2D 标签: 认知层级 × 知识类型."""

    def test_create_tag(self):
        from dy3_polaris.l1.models import BloomTag, KnowledgeType
        from dy3_polaris.l3.api_models import BloomLevel

        tag = BloomTag(
            cognitive_level=BloomLevel.APPLY,
            knowledge_type=KnowledgeType.PROCEDURAL,
        )
        assert tag.cognitive_level == BloomLevel.APPLY
        assert tag.knowledge_type == KnowledgeType.PROCEDURAL

    def test_matrix_cell(self):
        """2D 矩阵单元格 = (cognitive_level, knowledge_type)."""
        from dy3_polaris.l1.models import BloomTag, KnowledgeType
        from dy3_polaris.l3.api_models import BloomLevel

        tag = BloomTag(
            cognitive_level=BloomLevel.CREATE,
            knowledge_type=KnowledgeType.METACOGNITIVE,
        )
        cell = tag.matrix_cell()
        assert "create" in cell
        assert "metacognitive" in cell

    def test_to_dict(self):
        from dy3_polaris.l1.models import BloomTag, KnowledgeType
        from dy3_polaris.l3.api_models import BloomLevel

        tag = BloomTag(
            cognitive_level=BloomLevel.UNDERSTAND,
            knowledge_type=KnowledgeType.CONCEPTUAL,
        )
        d = tag.to_dict()
        assert d["cognitive_level"] == "understand"
        assert d["knowledge_type"] == "conceptual"


# ============================================================
# F. 跨层接口模型测试 (设计文档第八章)
# ============================================================


class TestBKTUpdate:
    """BKT 参数更新 (L2 → L1)."""

    def test_create_update(self):
        from dy3_polaris.l1.models import BKTUpdate

        update = BKTUpdate(
            kc_id="kc-001",
            p_know=0.85,
            p_slip=0.08,
            p_guess=0.2,
            p_transit=0.12,
        )
        assert update.kc_id == "kc-001"
        assert update.p_know == 0.85

    def test_validation(self):
        from dy3_polaris.l1.models import BKTUpdate

        with pytest.raises(ValueError):
            BKTUpdate(kc_id="kc", p_know=1.5)

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import BKTUpdate

        update = BKTUpdate(kc_id="kc", p_know=0.7, p_slip=0.1, p_guess=0.25, p_transit=0.1)
        d = update.to_dict()
        restored = BKTUpdate.from_dict(d)
        assert restored.p_know == update.p_know


class TestMemoryEntry:
    """学习记忆写入 (L1 → L2)."""

    def test_create_entry(self):
        from dy3_polaris.l1.models import MemoryEntry

        entry = MemoryEntry(
            session_id="sess-001",
            interaction_summary="学生完成了 Dy3+ 能级跃迁练习",
            key_insights=["掌握了 Judd-Ofelt 理论基础"],
            weak_areas=["光谱项计算仍有困难"],
        )
        assert entry.session_id == "sess-001"
        assert len(entry.key_insights) == 1
        assert len(entry.weak_areas) == 1

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import MemoryEntry

        entry = MemoryEntry(
            session_id="s1",
            interaction_summary="test",
            key_insights=["a"],
            weak_areas=["b"],
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.interaction_summary == entry.interaction_summary


class TestDecayRequest:
    """遗忘调度请求 (L1 → L2)."""

    def test_create_request(self):
        from dy3_polaris.l1.models import DecayRequest

        req = DecayRequest(
            user_id="u-001",
            kcs_to_review=["kc-001", "kc-002"],
            urgency_scores={"kc-001": 0.8, "kc-002": 0.3},
        )
        assert req.user_id == "u-001"
        assert len(req.kcs_to_review) == 2

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import DecayRequest

        req = DecayRequest(
            user_id="u",
            kcs_to_review=["kc1"],
            urgency_scores={"kc1": 0.5},
        )
        d = req.to_dict()
        restored = DecayRequest.from_dict(d)
        assert restored.kcs_to_review == req.kcs_to_review


class TestAccessCheck:
    """知识访问检查 (L1 → L3)."""

    def test_create_check(self):
        from dy3_polaris.l1.models import AccessCheck

        check = AccessCheck(
            user_id="u-001",
            resource_id="res-001",
            access_level="read",
        )
        assert check.user_id == "u-001"
        assert check.resource_id == "res-001"

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import AccessCheck

        check = AccessCheck(user_id="u", resource_id="r", access_level="read")
        d = check.to_dict()
        restored = AccessCheck.from_dict(d)
        assert restored.access_level == check.access_level


class TestResourceRequest:
    """资源推荐请求 (L1 → L3)."""

    def test_create_request(self):
        from dy3_polaris.l1.models import ResourceRequest

        req = ResourceRequest(
            weak_kcs=["kc-001", "kc-003"],
            difficulty_range=(0.3, 0.7),
            resource_types=["diagram", "card"],
            count_limit=5,
        )
        assert len(req.weak_kcs) == 2
        assert req.count_limit == 5

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import ResourceRequest

        req = ResourceRequest(
            weak_kcs=["kc1"],
            difficulty_range=(0.2, 0.6),
            resource_types=["video"],
            count_limit=3,
        )
        d = req.to_dict()
        restored = ResourceRequest.from_dict(d)
        assert restored.weak_kcs == req.weak_kcs


class TestKnowledgeResult:
    """知识查询结果 (L3 → L1)."""

    def test_create_result(self):
        from dy3_polaris.l1.models import KnowledgeResult, ResourceItem

        result = KnowledgeResult(
            resources=[
                ResourceItem(resource_id="r1", title="Test", resource_type="card"),
            ],
            confidence_scores={"r1": 0.9},
            source_trace=["kb:dy3_energy"],
        )
        assert len(result.resources) == 1
        assert result.confidence_scores["r1"] == 0.9

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import KnowledgeResult

        result = KnowledgeResult(
            resources=[],
            confidence_scores={},
            source_trace=["kb:test"],
        )
        d = result.to_dict()
        restored = KnowledgeResult.from_dict(d)
        assert restored.source_trace == result.source_trace


class TestPrivacyEvent:
    """隐私事件通知 (L1 → L0)."""

    def test_create_event(self):
        from dy3_polaris.l1.models import PrivacyEvent, DataLevel

        event = PrivacyEvent(
            event_type="data_access",
            user_id="u-001",
            data_level=DataLevel.L3_SENSITIVE,
            detail="Accessed learning report",
        )
        assert event.event_type == "data_access"
        assert event.data_level == DataLevel.L3_SENSITIVE

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import PrivacyEvent, DataLevel

        event = PrivacyEvent(
            event_type="data_export",
            user_id="u",
            data_level=DataLevel.L4_CONFIDENTIAL,
            detail="Export requested",
        )
        d = event.to_dict()
        restored = PrivacyEvent.from_dict(d)
        assert restored.event_type == event.event_type


class TestPolicyUpdate:
    """策略更新通知 (L0 → L1)."""

    def test_create_update(self):
        from dy3_polaris.l1.models import PolicyUpdate

        update = PolicyUpdate(
            policy_id="privacy-retention-v2",
            version="2.0",
            diff={"retention_days": 365},
            effective_at=int(time.time() * 1000),
        )
        assert update.policy_id == "privacy-retention-v2"
        assert update.version == "2.0"

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import PolicyUpdate

        update = PolicyUpdate(
            policy_id="p1",
            version="1.0",
            diff={"key": "val"},
            effective_at=1000,
        )
        d = update.to_dict()
        restored = PolicyUpdate.from_dict(d)
        assert restored.policy_id == update.policy_id


# ============================================================
# G. 隐私保护执行模型测试
# ============================================================


class TestDesensitizationMethod:
    """脱敏方法枚举."""

    def test_methods(self):
        from dy3_polaris.l1.models import DesensitizationMethod

        assert DesensitizationMethod.HASH
        assert DesensitizationMethod.AGGREGATE
        assert DesensitizationMethod.BUCKET
        assert DesensitizationMethod.DP_NOISE
        assert DesensitizationMethod.PSEUDO_ID


class TestPrivacyConfig:
    """隐私配置模型."""

    def test_create_config(self):
        from dy3_polaris.l1.models import PrivacyConfig, DesensitizationMethod

        config = PrivacyConfig(
            k_anonymity=K_ANONYMITY_MIN,
            l_diversity=L_DIVERSITY_MIN,
            epsilon=0.5,
            delta=1e-5,
            quasi_identifiers=["age", "major"],
            sensitive_attributes=["grade", "score"],
        )
        assert config.k_anonymity == K_ANONYMITY_MIN
        assert config.epsilon == 0.5

    def test_epsilon_range(self):
        """ε 应为正数."""
        from dy3_polaris.l1.models import PrivacyConfig

        with pytest.raises(ValueError):
            PrivacyConfig(epsilon=-0.1)

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import PrivacyConfig

        config = PrivacyConfig(k_anonymity=10, l_diversity=3, epsilon=0.8)
        d = config.to_dict()
        restored = PrivacyConfig.from_dict(d)
        assert restored.k_anonymity == config.k_anonymity


class TestRetentionPhase:
    """数据保留阶段枚举."""

    def test_four_phases(self):
        from dy3_polaris.l1.models import RetentionPhase

        assert RetentionPhase.ACTIVE
        assert RetentionPhase.ARCHIVED
        assert RetentionPhase.ANONYMIZED
        assert RetentionPhase.DELETED


class TestRetentionPolicy:
    """数据保留策略."""

    def test_create_policy(self):
        from dy3_polaris.l1.models import RetentionPolicy, RetentionPhase

        policy = RetentionPolicy(
            data_level="L3_SENSITIVE",
            phases=[
                (RetentionPhase.ACTIVE, 180),       # 课程期间: 180 天
                (RetentionPhase.ARCHIVED, 365),     # 毕业后 1 年
                (RetentionPhase.ANONYMIZED, 730),   # 匿名化保留 2 年
                (RetentionPhase.DELETED, 0),         # 3 年后删除
            ],
        )
        assert len(policy.phases) == 4
        assert policy.phases[0][0] == RetentionPhase.ACTIVE

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import RetentionPolicy, RetentionPhase

        policy = RetentionPolicy(
            data_level="L4_CONFIDENTIAL",
            phases=[(RetentionPhase.ACTIVE, 90), (RetentionPhase.DELETED, 0)],
        )
        d = policy.to_dict()
        restored = RetentionPolicy.from_dict(d)
        assert len(restored.phases) == 2


class TestDesensitizeStudentID:
    """学号脱敏函数."""

    def test_hash_student_id(self):
        from dy3_polaris.l1.models import desensitize_student_id

        hashed = desensitize_student_id("CS20240001", salt="test-salt")
        assert hashed != "CS20240001"
        assert len(hashed) > 0
        # 同输入同盐应产生相同哈希
        assert desensitize_student_id("CS20240001", salt="test-salt") == hashed

    def test_different_ids_different_hash(self):
        from dy3_polaris.l1.models import desensitize_student_id

        h1 = desensitize_student_id("CS20240001", salt="s")
        h2 = desensitize_student_id("CS20240002", salt="s")
        assert h1 != h2


class TestBucketResponseTime:
    """答题时间分桶函数."""

    def test_bucket_fast(self):
        from dy3_polaris.l1.models import bucket_response_time

        assert bucket_response_time(3000) == "fast"  # < 5s

    def test_bucket_normal(self):
        from dy3_polaris.l1.models import bucket_response_time

        assert bucket_response_time(30000) == "normal"  # 5-60s

    def test_bucket_slow(self):
        from dy3_polaris.l1.models import bucket_response_time

        assert bucket_response_time(120000) == "slow"  # > 60s


# ============================================================
# H. 学习分析事件测试 (xAPI / Caliper 兼容)
# ============================================================


class TestLearningEvent:
    """统一学习事件模型 (xAPI Actor-Verb-Object / Caliper Event)."""

    def test_create_event(self):
        from dy3_polaris.l1.models import LearningEvent

        event = LearningEvent(
            actor_id="u-001",
            action="completed",
            object_id="res-001",
            object_type="assessment",
        )
        assert event.actor_id == "u-001"
        assert event.action == "completed"
        assert event.event_id  # 自动生成
        assert event.timestamp > 0

    def test_with_result(self):
        from dy3_polaris.l1.models import LearningEvent, EventResult

        result = EventResult(
            score_scaled=0.85,
            success=True,
            completion=True,
            duration_ms=30000,
        )
        event = LearningEvent(
            actor_id="u-001",
            action="scored",
            object_id="quiz-001",
            object_type="assessment",
            result=result,
        )
        assert event.result.score_scaled == 0.85

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import LearningEvent

        event = LearningEvent(
            actor_id="u",
            action="viewed",
            object_id="r1",
            object_type="resource",
        )
        d = event.to_dict()
        restored = LearningEvent.from_dict(d)
        assert restored.actor_id == event.actor_id
        assert restored.action == event.action


class TestEventResult:
    """事件结果."""

    def test_create_result(self):
        from dy3_polaris.l1.models import EventResult

        result = EventResult(
            score_scaled=0.9,
            score_raw=90,
            score_max=100,
            success=True,
            completion=True,
            duration_ms=45000,
        )
        assert result.score_scaled == 0.9
        assert result.success is True

    def test_score_range_validation(self):
        from dy3_polaris.l1.models import EventResult

        with pytest.raises(ValueError):
            EventResult(score_scaled=1.5)

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import EventResult

        result = EventResult(score_scaled=0.7, success=True, completion=True, duration_ms=10000)
        d = result.to_dict()
        restored = EventResult.from_dict(d)
        assert restored.score_scaled == result.score_scaled


# ============================================================
# I. 参与度指标测试
# ============================================================


class TestEngagementLevel:
    """参与度等级枚举."""

    def test_four_levels(self):
        from dy3_polaris.l1.models import EngagementLevel

        assert EngagementLevel.HIGH
        assert EngagementLevel.MEDIUM
        assert EngagementLevel.LOW
        assert EngagementLevel.DISENGAGED


class TestEngagementMetrics:
    """三维参与度指标 (行为/认知/情感)."""

    def test_create_metrics(self):
        from dy3_polaris.l1.models import EngagementMetrics, EngagementLevel

        metrics = EngagementMetrics(
            session_duration_ms=1800000,
            login_frequency=5,
            completion_rate=0.8,
            accuracy_rate=0.75,
            avg_response_time_ms=25000,
            sentiment_score=0.6,
            hint_usage_count=3,
        )
        assert metrics.session_duration_ms == 1800000
        assert metrics.completion_rate == 0.8

    def test_composite_score(self):
        """综合参与度 = 加权平均(行为+认知+情感)."""
        from dy3_polaris.l1.models import EngagementMetrics

        metrics = EngagementMetrics(
            session_duration_ms=1800000,
            login_frequency=5,
            completion_rate=0.8,
            accuracy_rate=0.75,
            avg_response_time_ms=25000,
            sentiment_score=0.6,
            hint_usage_count=3,
        )
        score = metrics.composite_score()
        assert 0.0 <= score <= 1.0

    def test_engagement_level_classification(self):
        """根据综合得分分类参与度等级."""
        from dy3_polaris.l1.models import EngagementMetrics, EngagementLevel

        high = EngagementMetrics(
            session_duration_ms=3600000,
            login_frequency=7,
            completion_rate=0.95,
            accuracy_rate=0.9,
            avg_response_time_ms=20000,
            sentiment_score=0.8,
            hint_usage_count=1,
        )
        assert high.classify_level() == EngagementLevel.HIGH

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import EngagementMetrics

        metrics = EngagementMetrics(
            session_duration_ms=600000,
            login_frequency=2,
            completion_rate=0.5,
            accuracy_rate=0.6,
            avg_response_time_ms=30000,
            sentiment_score=0.3,
            hint_usage_count=5,
        )
        d = metrics.to_dict()
        restored = EngagementMetrics.from_dict(d)
        assert restored.completion_rate == metrics.completion_rate


# ============================================================
# J. 会话分析测试
# ============================================================


class TestSessionAnalytics:
    """会话聚合分析."""

    def test_create_analytics(self):
        from dy3_polaris.l1.models import SessionAnalytics

        analytics = SessionAnalytics(
            session_id="sess-001",
            total_duration_ms=1800000,
            total_interactions=25,
            total_questions=10,
            correct_answers=7,
            mastery_delta=0.15,
            engagement_score=0.75,
        )
        assert analytics.session_id == "sess-001"
        assert analytics.accuracy_rate == pytest.approx(0.7, abs=0.01)

    def test_accuracy_rate_calculation(self):
        from dy3_polaris.l1.models import SessionAnalytics

        analytics = SessionAnalytics(
            session_id="s1",
            total_duration_ms=1000,
            total_interactions=10,
            total_questions=8,
            correct_answers=6,
            mastery_delta=0.1,
            engagement_score=0.5,
        )
        assert analytics.accuracy_rate == pytest.approx(0.75, abs=0.01)

    def test_accuracy_rate_zero_questions(self):
        """无问题时正确率为 0."""
        from dy3_polaris.l1.models import SessionAnalytics

        analytics = SessionAnalytics(
            session_id="s1",
            total_duration_ms=1000,
            total_interactions=5,
            total_questions=0,
            correct_answers=0,
            mastery_delta=0.0,
            engagement_score=0.0,
        )
        assert analytics.accuracy_rate == 0.0

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import SessionAnalytics

        analytics = SessionAnalytics(
            session_id="s1",
            total_duration_ms=1000,
            total_interactions=5,
            total_questions=3,
            correct_answers=2,
            mastery_delta=0.1,
            engagement_score=0.6,
        )
        d = analytics.to_dict()
        restored = SessionAnalytics.from_dict(d)
        assert restored.session_id == analytics.session_id


# ============================================================
# K. 学习路径数据结构测试
# ============================================================


class TestPathNode:
    """学习路径节点."""

    def test_create_node(self):
        from dy3_polaris.l1.models import PathNode

        node = PathNode(
            kc_id="kc-001",
            order=0,
            estimated_difficulty=0.4,
            prerequisite_kcs=[],
        )
        assert node.kc_id == "kc-001"
        assert node.order == 0

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import PathNode

        node = PathNode(kc_id="kc", order=1, estimated_difficulty=0.5, prerequisite_kcs=["kc0"])
        d = node.to_dict()
        restored = PathNode.from_dict(d)
        assert restored.kc_id == node.kc_id


class TestLearningPath:
    """学习路径."""

    def test_create_path(self):
        from dy3_polaris.l1.models import LearningPath, PathNode

        path = LearningPath(
            user_id="u-001",
            nodes=[
                PathNode(kc_id="kc-001", order=0, estimated_difficulty=0.3),
                PathNode(kc_id="kc-002", order=1, estimated_difficulty=0.5, prerequisite_kcs=["kc-001"]),
            ],
        )
        assert path.user_id == "u-001"
        assert len(path.nodes) == 2
        assert path.path_id  # 自动生成

    def test_total_estimated_time(self):
        from dy3_polaris.l1.models import LearningPath, PathNode

        path = LearningPath(
            user_id="u",
            nodes=[
                PathNode(kc_id="kc1", order=0, estimated_difficulty=0.3, estimated_time_minutes=15),
                PathNode(kc_id="kc2", order=1, estimated_difficulty=0.5, estimated_time_minutes=30),
            ],
        )
        assert path.total_estimated_time() == 45

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import LearningPath, PathNode

        path = LearningPath(
            user_id="u",
            nodes=[PathNode(kc_id="kc", order=0, estimated_difficulty=0.3)],
        )
        d = path.to_dict()
        restored = LearningPath.from_dict(d)
        assert len(restored.nodes) == 1


class TestPathRecommendation:
    """路径推荐."""

    def test_create_recommendation(self):
        from dy3_polaris.l1.models import PathRecommendation

        rec = PathRecommendation(
            user_id="u-001",
            recommended_path_id="path-001",
            rationale="基于薄弱知识点 kc-001 和 kc-003 推荐",
            predicted_mastery_gain=0.2,
            confidence=0.85,
        )
        assert rec.user_id == "u-001"
        assert rec.predicted_mastery_gain == 0.2

    def test_confidence_range(self):
        from dy3_polaris.l1.models import PathRecommendation

        with pytest.raises(ValueError):
            PathRecommendation(
                user_id="u",
                recommended_path_id="p",
                rationale="r",
                predicted_mastery_gain=0.1,
                confidence=1.5,
            )

    def test_to_dict_and_from_dict(self):
        from dy3_polaris.l1.models import PathRecommendation

        rec = PathRecommendation(
            user_id="u",
            recommended_path_id="p",
            rationale="r",
            predicted_mastery_gain=0.15,
            confidence=0.8,
        )
        d = rec.to_dict()
        restored = PathRecommendation.from_dict(d)
        assert restored.rationale == rec.rationale


# ============================================================
# L. 序列化补全测试 (from_dict)
# ============================================================


class TestRoleFromDict:
    """Role.from_dict 补全."""

    def test_from_dict_roundtrip(self):
        role = Role(
            role_code="undergrad",
            role_name="本科生",
            base_permissions=[Permission.KB_PUBLIC_READ, Permission.AGENT_DIAGNOSIS],
        )
        d = role.to_dict()
        restored = Role.from_dict(d)
        assert restored.role_code == role.role_code
        assert restored.role_name == role.role_name
        assert Permission.KB_PUBLIC_READ in restored.base_permissions
        assert Permission.AGENT_DIAGNOSIS in restored.base_permissions


class TestApprovalRequestFromDict:
    """ApprovalRequest.from_dict 补全."""

    def test_from_dict_roundtrip(self):
        req = ApprovalRequest(
            user_id="u-001",
            session_id="s-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="请确认理解 Dy3+ 能级跃迁",
            priority=HiTLPriority.P2,
            confidence=0.8,
        )
        d = req.to_dict()
        restored = ApprovalRequest.from_dict(d)
        assert restored.user_id == req.user_id
        assert restored.hitl_type == req.hitl_type
        assert restored.confidence == req.confidence


class TestFeedbackReportFromDict:
    """FeedbackReport.from_dict 补全."""

    def test_from_dict_roundtrip(self):
        report = FeedbackReport(
            user_id="u-001",
            session_id="s-001",
            feedback_type=FeedbackType.INCORRECT,
            content="能级图标注有误",
            artifact_id="art-001",
            severity=0.7,
        )
        d = report.to_dict()
        restored = FeedbackReport.from_dict(d)
        assert restored.user_id == report.user_id
        assert restored.feedback_type == report.feedback_type
        assert restored.severity == report.severity


class TestProvenanceRecordFromDict:
    """ProvenanceRecord.from_dict 补全."""

    def test_from_dict_roundtrip(self):
        record = ProvenanceRecord(
            artifact_id="art-001",
            actor_chain=["agent-diag-001", "agent-review-001"],
            code_hash="sha256:abc123",
            env_hash="sha256:def456",
        )
        d = record.to_dict()
        restored = ProvenanceRecord.from_dict(d)
        assert restored.artifact_id == record.artifact_id
        assert restored.actor_chain == record.actor_chain
        assert restored.code_hash == record.code_hash


# ============================================================
# M. 验证补全测试
# ============================================================


class TestValidationGaps:
    """补全校验逻辑."""

    def test_approval_request_confidence_range(self):
        """ApprovalRequest.confidence 应在 [0.0, 1.0]."""
        with pytest.raises(ValueError):
            ApprovalRequest(
                user_id="u",
                session_id="s",
                hitl_type=HiTLType.CONFIRMATION,
                content="c",
                confidence=1.5,
            )

    def test_feedback_report_severity_range(self):
        """FeedbackReport.severity 应在 [0.0, 1.0]."""
        with pytest.raises(ValueError):
            FeedbackReport(
                user_id="u",
                session_id="s",
                feedback_type=FeedbackType.UNDERSTOOD,
                content="c",
                severity=-0.1,
            )

    def test_session_artifact_confidence_range(self):
        """SessionArtifact.confidence 应在 [0.0, 1.0]."""
        with pytest.raises(ValueError):
            SessionArtifact(
                artifact_type="card",
                title="t",
                confidence=1.5,
            )

    def test_emergency_alert_cognitive_load_range(self):
        """EmergencyAlert.cognitive_load 应在 [0.0, 1.0]."""
        with pytest.raises(ValueError):
            EmergencyAlert(
                session_id="s",
                user_id="u",
                trigger_reason="r",
                trigger_value=0.97,
                cognitive_load=1.5,
            )

    def test_session_fork_fork_point_seq_non_negative(self):
        """SessionFork.fork_point_seq 应 >= 0."""
        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(user_id="u", session_id="s", timestamp=now_ms)
        with pytest.raises(ValueError):
            SessionFork(
                source_session_id="s1",
                fork_point_seq=-1,
                fork_reason="test",
                branch_label="b",
                snapshot_at_fork=env,
            )

    def test_session_checkpoint_seq_non_negative(self):
        """SessionCheckpoint.seq 应 >= 0."""
        with pytest.raises(ValueError):
            SessionCheckpoint(session_id="s", seq=-1)

    def test_calculate_decay_negative_elapsed(self):
        """current_ts < last_practiced 时应处理为 0 elapsed."""
        from dy3_polaris.l1.models import calculate_decay

        # 不应抛出异常, 应将 elapsed 当作 0
        result = calculate_decay(
            p_know=0.8,
            last_practiced=1000000,
            repetitions=2,
            current_ts=500000,  # 比 last_practiced 还小
        )
        assert result == pytest.approx(0.8, abs=0.01)  # decay=1.0, effective=0.8


# ============================================================
# N. 跨层对齐增强测试
# ============================================================


class TestContextEnvelopeEnhancedAlignment:
    """ContextEnvelope 跨层对齐增强."""

    def test_to_l3_learner_profile_with_level(self):
        """to_l3_learner_profile 应填充 level 字段."""
        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u-001",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.8, last_practiced_at=now_ms),
            ],
            timestamp=now_ms,
        )
        profile = env.to_l3_learner_profile()
        assert profile.level is not None

    def test_to_l3_learner_profile_with_bloom_target(self):
        """to_l3_learner_profile 应从 goals 推导 bloom_target."""
        from dy3_polaris.l3.api_models import BloomLevel

        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u-001",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.5, last_practiced_at=now_ms),
            ],
            goals=[
                LearningGoal(
                    description="掌握应用",
                    priority=5,
                    bloom_level=BloomLevel.APPLY,
                ),
            ],
            timestamp=now_ms,
        )
        profile = env.to_l3_learner_profile()
        assert profile.bloom_target == BloomLevel.APPLY

    def test_to_l3_learner_profile_with_preferred_style(self):
        """to_l3_learner_profile 应填充 preferred_style (如果有)."""
        from dy3_polaris.l1.models import VARKProfile, VARKStyle

        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u-001",
            session_id="s1",
            mastery_snapshot=[],
            timestamp=now_ms,
            learning_style=VARKProfile(
                user_id="u-001",
                visual_score=0.8,
                aural_score=0.3,
                read_write_score=0.5,
                kinesthetic_score=0.4,
            ),
        )
        profile = env.to_l3_learner_profile()
        assert profile.preferred_style is not None


class TestContextEnvelopeFromL3:
    """ContextEnvelope 反向转换."""

    def test_from_l3_learner_profile(self):
        """从 L3 LearnerProfile 逆向构建 ContextEnvelope."""
        from dy3_polaris.l3.api_models import LearnerProfile, KPMastery, LearningStyle, BloomLevel

        profile = LearnerProfile(
            learner_id="u-001",
            kp_mastery={
                "kc1": KPMastery(
                    kp_id="kc1",
                    mastery_prob=0.8,
                    attempts=10,
                    correct_count=8,
                    last_attempt_time=time.time(),
                ),
            },
            weak_kps=["kc1"],
            level="intermediate",
            preferred_style=LearningStyle.VISUAL,
            bloom_target=BloomLevel.UNDERSTAND,
        )
        env = ContextEnvelope.from_l3_learner_profile(
            profile, session_id="s-001"
        )
        assert env.user_id == "u-001"
        assert env.session_id == "s-001"
        assert len(env.mastery_snapshot) == 1
        assert env.mastery_snapshot[0].kc_id == "kc1"


class TestContextEnvelopeWithCognitiveLoadBreakdown:
    """ContextEnvelope 集成认知负荷三分模型."""

    def test_envelope_with_cognitive_load_breakdown(self):
        from dy3_polaris.l1.models import CognitiveLoadBreakdown

        now_ms = int(time.time() * 1000)
        breakdown = CognitiveLoadBreakdown(
            intrinsic_load=0.4,
            extraneous_load=0.2,
            germane_load=0.3,
        )
        env = ContextEnvelope(
            user_id="u",
            session_id="s",
            timestamp=now_ms,
            cognitive_load_breakdown=breakdown,
        )
        assert env.cognitive_load_breakdown is not None
        assert env.cognitive_load_breakdown.total_load == pytest.approx(0.9, abs=0.01)


class TestContextEnvelopeWithLearningStyle:
    """ContextEnvelope 集成 VARK 学习风格."""

    def test_envelope_with_learning_style(self):
        from dy3_polaris.l1.models import VARKProfile, VARKStyle

        now_ms = int(time.time() * 1000)
        style = VARKProfile(
            user_id="u",
            visual_score=0.9,
            aural_score=0.2,
            read_write_score=0.4,
            kinesthetic_score=0.3,
        )
        env = ContextEnvelope(
            user_id="u",
            session_id="s",
            timestamp=now_ms,
            learning_style=style,
        )
        assert env.learning_style is not None
        assert env.learning_style.primary_style == VARKStyle.VISUAL


class TestContextEnvelopeWithIRTAbility:
    """ContextEnvelope 集成 IRT 能力参数."""

    def test_envelope_with_irt_ability(self):
        from dy3_polaris.l1.models import IRTAbility

        now_ms = int(time.time() * 1000)
        ability = IRTAbility(user_id="u", theta=0.8, standard_error=0.2)
        env = ContextEnvelope(
            user_id="u",
            session_id="s",
            timestamp=now_ms,
            irt_ability=ability,
        )
        assert env.irt_ability is not None
        assert env.irt_ability.theta == 0.8


class TestContextEnvelopeWithEngagement:
    """ContextEnvelope 集成参与度指标."""

    def test_envelope_with_engagement(self):
        from dy3_polaris.l1.models import EngagementMetrics

        now_ms = int(time.time() * 1000)
        metrics = EngagementMetrics(
            session_duration_ms=1800000,
            login_frequency=5,
            completion_rate=0.8,
            accuracy_rate=0.75,
            avg_response_time_ms=25000,
            sentiment_score=0.6,
            hint_usage_count=3,
        )
        env = ContextEnvelope(
            user_id="u",
            session_id="s",
            timestamp=now_ms,
            engagement=metrics,
        )
        assert env.engagement is not None
        assert env.engagement.completion_rate == 0.8


class TestContextEnvelopeSerializationWithNewFields:
    """ContextEnvelope 序列化包含新字段."""

    def test_to_dict_includes_new_fields(self):
        from dy3_polaris.l1.models import VARKProfile, CognitiveLoadBreakdown

        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u",
            session_id="s",
            timestamp=now_ms,
            learning_style=VARKProfile(
                user_id="u",
                visual_score=0.7,
                aural_score=0.3,
                read_write_score=0.5,
                kinesthetic_score=0.4,
            ),
            cognitive_load_breakdown=CognitiveLoadBreakdown(
                intrinsic_load=0.3,
                extraneous_load=0.2,
                germane_load=0.3,
            ),
        )
        d = env.to_dict()
        assert "learning_style" in d
        assert "cognitive_load_breakdown" in d

    def test_from_dict_restores_new_fields(self):
        from dy3_polaris.l1.models import VARKProfile, CognitiveLoadBreakdown

        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u",
            session_id="s",
            timestamp=now_ms,
            learning_style=VARKProfile(
                user_id="u",
                visual_score=0.7,
                aural_score=0.3,
                read_write_score=0.5,
                kinesthetic_score=0.4,
            ),
        )
        d = env.to_dict()
        restored = ContextEnvelope.from_dict(d)
        assert restored.learning_style is not None
        assert restored.learning_style.visual_score == 0.7


# ============================================================
# O. User 模型增强测试
# ============================================================


class TestUserWithLearningStyle:
    """User 模型集成学习风格."""

    def test_user_with_vark_profile(self):
        from dy3_polaris.l1.models import VARKProfile

        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        user.learning_style = VARKProfile(
            user_id=user.user_id,
            visual_score=0.8,
            aural_score=0.3,
            read_write_score=0.5,
            kinesthetic_score=0.4,
        )
        d = user.to_dict()
        assert "learning_style" in d

    def test_user_from_dict_with_learning_style(self):
        from dy3_polaris.l1.models import VARKProfile

        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        user.learning_style = VARKProfile(
            user_id=user.user_id,
            visual_score=0.8,
            aural_score=0.3,
            read_write_score=0.5,
            kinesthetic_score=0.4,
        )
        d = user.to_dict()
        restored = User.from_dict(d)
        assert restored.learning_style is not None
        assert restored.learning_style.visual_score == 0.8


class TestUserTouch:
    """User.touch() 更新 updated_at."""

    def test_touch_updates_timestamp(self):
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        original = user.updated_at
        time.sleep(0.01)
        user.touch()
        assert user.updated_at > original


# ============================================================
# P. 审计日志溯源链增强测试
# ============================================================


class TestAuditLogWithProvenance:
    """审计日志溯源链字段."""

    def test_audit_log_with_provenance_chain(self):
        entry = AuditLogEntry(
            actor_id="u-001",
            actor_role=UserRole.TEACHER,
            action=AuditAction.MODIFY,
            target_resource="kb:dy3_energy",
            target_data_level=DataLevel.L2_INTERNAL,
            purpose="内容更新",
            result=AuditResult.SUCCESS,
            provenance_chain=["art-001", "prov-001"],
        )
        assert entry.provenance_chain is not None
        assert len(entry.provenance_chain) == 2

    def test_audit_log_to_dict_includes_provenance(self):
        entry = AuditLogEntry(
            actor_id="u",
            actor_role=UserRole.UNDERGRAD,
            action=AuditAction.VIEW,
            target_resource="kb:test",
            target_data_level=DataLevel.L2_INTERNAL,
            purpose="test",
            result=AuditResult.SUCCESS,
            provenance_chain=["art-001"],
        )
        d = entry.to_dict()
        assert "provenance_chain" in d

    def test_audit_log_from_dict_with_provenance(self):
        entry = AuditLogEntry(
            actor_id="u",
            actor_role=UserRole.UNDERGRAD,
            action=AuditAction.VIEW,
            target_resource="kb:test",
            target_data_level=DataLevel.L2_INTERNAL,
            purpose="test",
            result=AuditResult.SUCCESS,
            provenance_chain=["art-001", "art-002"],
        )
        d = entry.to_dict()
        restored = AuditLogEntry.from_dict(d)
        assert restored.provenance_chain == ["art-001", "art-002"]


# ============================================================
# Q. 模块导出完整性测试
# ============================================================


class TestModuleExports:
    """验证所有新增模型在 __all__ 中导出."""

    def test_fsrs_exports(self):
        from dy3_polaris.l1 import models

        assert hasattr(models, "FSRSParameters")
        assert hasattr(models, "FSRSCardState")
        assert hasattr(models, "FSRSReviewLog")
        assert "FSRSParameters" in models.__all__
        assert "FSRSCardState" in models.__all__
        assert "FSRSReviewLog" in models.__all__

    def test_irt_exports(self):
        from dy3_polaris.l1 import models

        assert hasattr(models, "IRTModel")
        assert hasattr(models, "IRTItem")
        assert hasattr(models, "IRTAbility")
        assert "IRTModel" in models.__all__
        assert "IRTItem" in models.__all__
        assert "IRTAbility" in models.__all__

    def test_vark_exports(self):
        from dy3_polaris.l1 import models

        assert hasattr(models, "VARKStyle")
        assert hasattr(models, "VARKProfile")
        assert hasattr(models, "ContentModality")
        assert "VARKStyle" in models.__all__
        assert "VARKProfile" in models.__all__
        assert "ContentModality" in models.__all__

    def test_cognitive_load_exports(self):
        from dy3_polaris.l1 import models

        assert hasattr(models, "CognitiveLoadBreakdown")
        assert hasattr(models, "ElementInteractivity")
        assert "CognitiveLoadBreakdown" in models.__all__
        assert "ElementInteractivity" in models.__all__

    def test_bloom_2d_exports(self):
        from dy3_polaris.l1 import models

        assert hasattr(models, "KnowledgeType")
        assert hasattr(models, "BloomTag")
        assert "KnowledgeType" in models.__all__
        assert "BloomTag" in models.__all__

    def test_cross_layer_exports(self):
        from dy3_polaris.l1 import models

        for name in [
            "BKTUpdate",
            "MemoryEntry",
            "DecayRequest",
            "AccessCheck",
            "ResourceRequest",
            "KnowledgeResult",
            "PrivacyEvent",
            "PolicyUpdate",
        ]:
            assert hasattr(models, name), f"Missing {name}"
            assert name in models.__all__, f"{name} not in __all__"

    def test_privacy_exports(self):
        from dy3_polaris.l1 import models

        for name in [
            "DesensitizationMethod",
            "PrivacyConfig",
            "RetentionPhase",
            "RetentionPolicy",
            "desensitize_student_id",
            "bucket_response_time",
        ]:
            assert hasattr(models, name), f"Missing {name}"
            assert name in models.__all__, f"{name} not in __all__"

    def test_analytics_exports(self):
        from dy3_polaris.l1 import models

        for name in [
            "LearningEvent",
            "EventResult",
            "EngagementLevel",
            "EngagementMetrics",
            "SessionAnalytics",
        ]:
            assert hasattr(models, name), f"Missing {name}"
            assert name in models.__all__, f"{name} not in __all__"

    def test_path_exports(self):
        from dy3_polaris.l1 import models

        for name in ["PathNode", "LearningPath", "PathRecommendation"]:
            assert hasattr(models, name), f"Missing {name}"
            assert name in models.__all__, f"{name} not in __all__"
