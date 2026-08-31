"""L1 执行引擎补充模块测试 — FSRSScheduler / IRTEstimator / VARKCollector / 认知负荷三分 / 掌握度轨迹 / ZPD 便捷接口.

遵循 TDD Red-Green-Refactor:
1. 先写测试 (RED): 每个测试描述期望行为
2. 验证测试失败 (feature missing → ImportError / AttributeError)
3. 最小实现 (GREEN)
4. 重构 (保持绿色)

测试覆盖 (均为尚未实现的 L1 执行引擎补充功能):
- FSRSScheduler: FSRS 间隔重复调度 (schedule_review)
- IRTEstimator: IRT 能力估计 (贝叶斯后验更新 update_theta / MLE 批量估计 estimate_mle)
- VARKSurveyCollector: VARK 学习风格采集 (问卷采集 collect_survey / 行为推断 infer_from_behavior)
- LearningContextBroker.update_cognitive_load_breakdown: 认知负荷三分模型 (ICL/ECL/GCL)
- LearningContextBroker.update_mastery 增强: 自动记录 MasteryTrajectory 轨迹
- ContextEnvelope.get_zpd_recommendation: ZPD 便捷推荐接口

注意:
- 所有 import 路径使用 dy3_polaris.l1.xxx
- 不使用 mock, 使用真实类
- 新增模块 (fsrs_scheduler / irt_estimator / vark_collector) 尚未实现,
  相关测试在实现不存在时会因 ImportError 而失败 (TDD RED 阶段预期行为)
- 已有模型通过 from dy3_polaris.l1.models import ... 导入
"""

from __future__ import annotations

import time

import pytest

from dy3_polaris.l1.models import (
    BKTParams,
    CognitiveLoadBreakdown,
    ContextEnvelope,
    FSRSParameters,
    FSRSCardState,
    FSRSReviewLog,
    IRTAbility,
    IRTItem,
    IRTModel,
    MasterySnapshot,
    MasteryTrajectory,
    MasteryTrajectoryPoint,
    VARKProfile,
    VARKStyle,
    ZoneOfProximalDevelopment,
)
from dy3_polaris.l1.context_broker import LearningContextBroker


# ============================================================
# 辅助函数
# ============================================================


def _make_fsrs_params() -> FSRSParameters:
    """创建默认 FSRSParameters."""
    return FSRSParameters()


def _make_new_card(kc_id: str = "kc-fsrs-001") -> FSRSCardState:
    """创建新卡片 (state=new)."""
    return FSRSCardState(kc_id=kc_id, state=FSRSCardState.NEW)


def _make_review_card(
    kc_id: str = "kc-fsrs-001",
    stability: float = 5.0,
    difficulty: float = 5.0,
    last_review_ts: int | None = None,
) -> FSRSCardState:
    """创建复习中卡片 (state=review), 默认 5 天前复习."""
    if last_review_ts is not None:
        ts = last_review_ts
    else:
        # 默认 5 天前, 确保使用长期记忆公式 (非 same-day 短期模型)
        ts = int(time.time() * 1000) - int(5.0 * 86400 * 1000)
    return FSRSCardState(
        kc_id=kc_id,
        stability=stability,
        difficulty=difficulty,
        state=FSRSCardState.REVIEW,
        reps=3,
        last_review_ts=ts,
    )


def _make_irt_item(
    difficulty_b: float = 0.0,
    discrimination_a: float = 1.0,
    item_id: str = "item-001",
) -> IRTItem:
    """创建 IRTItem (2PL 模型)."""
    return IRTItem(
        item_id=item_id,
        model_type=IRTModel.TWO_PL,
        difficulty_b=difficulty_b,
        discrimination_a=discrimination_a,
    )


def _make_irt_ability(theta: float = 0.0, se: float = 0.5) -> IRTAbility:
    """创建 IRTAbility."""
    return IRTAbility(user_id="user-001", theta=theta, standard_error=se)


def _find_trajectory(envelope: ContextEnvelope, kc_id: str) -> MasteryTrajectory | None:
    """从信封中查找指定 KC 的掌握度轨迹 (兼容 dict / list 存储)."""
    container = getattr(envelope, "mastery_trajectories", None)
    if container is None:
        container = getattr(envelope, "mastery_trajectory", None)
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(kc_id)
    # list-like 容器
    for traj in container:
        if getattr(traj, "kc_id", None) == kc_id:
            return traj
    return None


# ============================================================
# 1. FSRSScheduler 测试
# ============================================================


class TestFSRSScheduler:
    """FSRS 间隔重复调度器测试 (新增模块 dy3_polaris.l1.fsrs_scheduler)."""

    def test_schedule_review_again_decreases_or_resets_stability(self):
        """grade=1 (Again): 稳定性应下降或重置."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = _make_fsrs_params()
        card = _make_review_card(stability=5.0)
        now_ts = int(time.time() * 1000)

        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=1, params=params, current_ts=now_ts
        )
        # Again → 稳定性下降或重置为低值
        assert new_card.stability < 5.0

    def test_schedule_review_good_increases_stability(self):
        """grade=3 (Good): 稳定性应增加."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = _make_fsrs_params()
        card = _make_review_card(stability=2.0)
        now_ts = int(time.time() * 1000)

        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now_ts
        )
        assert new_card.stability > 2.0

    def test_schedule_review_easy_increases_more_than_good(self):
        """grade=4 (Easy) 稳定性增幅应大于 grade=3 (Good)."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = _make_fsrs_params()
        now_ts = int(time.time() * 1000)

        card_good = _make_review_card(stability=2.0, kc_id="kc-good")
        card_easy = _make_review_card(stability=2.0, kc_id="kc-easy")

        new_good, _log_good, _int_good = scheduler.schedule_review(
            card_state=card_good, grade=3, params=params, current_ts=now_ts
        )
        new_easy, _log_easy, _int_easy = scheduler.schedule_review(
            card_state=card_easy, grade=4, params=params, current_ts=now_ts
        )
        # Easy 增幅更大 → 稳定性更高
        assert new_easy.stability > new_good.stability

    def test_schedule_review_returns_positive_interval(self):
        """返回的 next_interval 应 > 0."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = _make_fsrs_params()
        card = _make_review_card(stability=3.0)
        now_ts = int(time.time() * 1000)

        _new_card, _log, next_interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now_ts
        )
        assert next_interval > 0

    def test_schedule_review_new_card_uses_initial_params(self):
        """首次复习 (state=new) 应使用 initial_stability / initial_difficulty."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = _make_fsrs_params()
        card = _make_new_card(kc_id="kc-new")
        now_ts = int(time.time() * 1000)
        grade = 3

        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=grade, params=params, current_ts=now_ts
        )
        # 新卡片稳定性 = initial_stability(grade)
        assert new_card.stability == pytest.approx(params.initial_stability(grade))
        # 新卡片难度 = initial_difficulty(grade)
        assert new_card.difficulty == pytest.approx(params.initial_difficulty(grade))

    def test_schedule_review_review_log_records_correctly(self):
        """ReviewLog 的 grade / state_before / state_after 应正确记录."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = _make_fsrs_params()
        card = _make_review_card(stability=3.0, kc_id="kc-log")
        now_ts = int(time.time() * 1000)

        _new_card, review_log, _interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now_ts
        )
        assert isinstance(review_log, FSRSReviewLog)
        assert review_log.grade == 3
        assert review_log.state_before == FSRSCardState.REVIEW
        # Good 评分后应进入 review 或保持 review 状态
        assert review_log.state_after in (
            FSRSCardState.REVIEW,
            FSRSCardState.LEARNING,
        )

    def test_schedule_review_again_increments_lapses(self):
        """grade=1 (Again) 应增加 lapses (遗忘次数)."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = _make_fsrs_params()
        card = _make_review_card(stability=5.0, kc_id="kc-lapse")
        now_ts = int(time.time() * 1000)
        initial_lapses = card.lapses

        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=1, params=params, current_ts=now_ts
        )
        assert new_card.lapses > initial_lapses

    def test_schedule_review_increments_reps(self):
        """每次复习应递增 reps."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = _make_fsrs_params()
        card = _make_review_card(stability=3.0, kc_id="kc-reps")
        now_ts = int(time.time() * 1000)
        initial_reps = card.reps

        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now_ts
        )
        assert new_card.reps == initial_reps + 1


# ============================================================
# 2. IRTEstimator 测试
# ============================================================


class TestIRTEstimator:
    """IRT 能力估计器测试 (新增模块 dy3_polaris.l1.irt_estimator)."""

    def test_update_theta_correct_easy_item(self):
        """答对难度低的题: θ 应微升或不变."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        ability = _make_irt_ability(theta=0.0)
        easy_item = _make_irt_item(difficulty_b=-1.0, item_id="item-easy")

        updated = estimator.update_theta(ability, easy_item, correct=True)
        # 答对易题 → θ 微升或不变 (>= 初始值)
        assert updated.theta >= 0.0 - 1e-9

    def test_update_theta_correct_hard_item(self):
        """答对难度高的题: θ 应明显上升."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        ability = _make_irt_ability(theta=0.0)
        hard_item = _make_irt_item(difficulty_b=1.5, item_id="item-hard")

        updated = estimator.update_theta(ability, hard_item, correct=True)
        # 答对难题 → θ 明显上升
        assert updated.theta > 0.0

    def test_update_theta_hard_more_than_easy(self):
        """答对难题的 θ 增幅应大于答对易题."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        easy_item = _make_irt_item(difficulty_b=-1.5, item_id="item-easy2")
        hard_item = _make_irt_item(difficulty_b=1.5, item_id="item-hard2")

        updated_easy = estimator.update_theta(
            _make_irt_ability(theta=0.0), easy_item, correct=True
        )
        updated_hard = estimator.update_theta(
            _make_irt_ability(theta=0.0), hard_item, correct=True
        )
        # 难题增幅更大
        assert updated_hard.theta > updated_easy.theta

    def test_update_theta_incorrect_easy_item(self):
        """答错难度低的题: θ 应下降."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        ability = _make_irt_ability(theta=0.0)
        easy_item = _make_irt_item(difficulty_b=-1.0, item_id="item-easy3")

        updated = estimator.update_theta(ability, easy_item, correct=False)
        # 答错易题 → θ 下降
        assert updated.theta < 0.0

    def test_update_theta_decreases_standard_error(self):
        """更新后 standard_error 应减小 (信息量增加)."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        ability = _make_irt_ability(theta=0.0, se=0.5)
        item = _make_irt_item(difficulty_b=0.0, item_id="item-se")

        updated = estimator.update_theta(ability, item, correct=True)
        # 信息量增加 → 标准误减小
        assert updated.standard_error < 0.5

    def test_update_theta_returns_new_ability(self):
        """update_theta 应返回 IRTAbility 对象."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        ability = _make_irt_ability(theta=0.0)
        item = _make_irt_item(difficulty_b=0.0)

        updated = estimator.update_theta(ability, item, correct=True)
        assert isinstance(updated, IRTAbility)

    def test_estimate_mle_all_correct_positive_theta(self):
        """MLE 批量估计: 全对 → θ 应为正."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=-1.0, item_id=f"item-c-{i}")
            for i in range(5)
        ]
        responses = [(item, True) for item in items]

        ability = estimator.estimate_mle(responses, initial_theta=0.0)
        assert ability.theta > 0.0
        assert isinstance(ability, IRTAbility)

    def test_estimate_mle_all_incorrect_negative_theta(self):
        """MLE 批量估计: 全错 → θ 应为负."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=1.0, item_id=f"item-w-{i}")
            for i in range(5)
        ]
        responses = [(item, False) for item in items]

        ability = estimator.estimate_mle(responses, initial_theta=0.0)
        assert ability.theta < 0.0

    def test_estimate_mle_mixed_reasonable_theta(self):
        """MLE 批量估计: 混合作答 → θ 应在合理范围 [-3, 3]."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=-1.0, item_id="item-m-0"),
            _make_irt_item(difficulty_b=0.5, item_id="item-m-1"),
            _make_irt_item(difficulty_b=1.0, item_id="item-m-2"),
            _make_irt_item(difficulty_b=-0.5, item_id="item-m-3"),
        ]
        responses = [
            (items[0], True),
            (items[1], False),
            (items[2], False),
            (items[3], True),
        ]

        ability = estimator.estimate_mle(responses, initial_theta=0.0)
        assert -3.0 <= ability.theta <= 3.0

    def test_estimate_mle_respects_theta_bounds(self):
        """MLE 估计结果应尊重 θ 边界 [-3, 3]."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        estimator = IRTEstimator()
        # 极端: 全对难题 → θ 应接近上界但不越界
        items = [
            _make_irt_item(difficulty_b=2.5, item_id=f"item-x-{i}")
            for i in range(10)
        ]
        responses = [(item, True) for item in items]

        ability = estimator.estimate_mle(responses, initial_theta=0.0)
        assert ability.theta <= 3.0


# ============================================================
# 3. VARKSurveyCollector 测试
# ============================================================


class TestVARKSurveyCollector:
    """VARK 学习风格采集器测试 (新增模块 dy3_polaris.l1.vark_collector)."""

    def test_collect_survey_all_visual(self):
        """问卷采集: 全选 1 (V) → visual_score=1.0, primary_style=VISUAL."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        # 16 题, 每题选 1 → 全部 Visual
        answers = [1] * 16

        profile = collector.collect_survey(user_id="user-vark-001", answers=answers)
        assert isinstance(profile, VARKProfile)
        assert profile.visual_score == pytest.approx(1.0)
        assert profile.primary_style == VARKStyle.VISUAL

    def test_collect_survey_all_aural(self):
        """问卷采集: 全选 2 (A) → aural_score=1.0, primary_style=AURAL."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        answers = [2] * 16

        profile = collector.collect_survey(user_id="user-vark-002", answers=answers)
        assert profile.aural_score == pytest.approx(1.0)
        assert profile.primary_style == VARKStyle.AURAL

    def test_collect_survey_mixed_correct_scores(self):
        """问卷采集: 混合选择应正确计算四维分数."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        # 16 题: 4 个 V(1), 4 个 A(2), 4 个 R(3), 4 个 K(4)
        answers = [1, 2, 3, 4] * 4

        profile = collector.collect_survey(user_id="user-vark-003", answers=answers)
        # 每个维度 4/16 = 0.25
        assert profile.visual_score == pytest.approx(0.25, abs=0.01)
        assert profile.aural_score == pytest.approx(0.25, abs=0.01)
        assert profile.read_write_score == pytest.approx(0.25, abs=0.01)
        assert profile.kinesthetic_score == pytest.approx(0.25, abs=0.01)

    def test_collect_survey_primary_style_detected(self):
        """问卷采集: primary_style 应自动检测为主导风格."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        # 12 个 K(4) + 4 个 V(1) → kinesthetic 主导
        answers = [4] * 12 + [1] * 4

        profile = collector.collect_survey(user_id="user-vark-004", answers=answers)
        assert profile.kinesthetic_score > profile.visual_score
        assert profile.primary_style == VARKStyle.KINESTHETIC

    def test_collect_survey_returns_profile_with_user_id(self):
        """问卷采集: 返回的 VARKProfile 应携带 user_id."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        profile = collector.collect_survey(user_id="user-vark-005", answers=[1] * 16)
        assert profile.user_id == "user-vark-005"

    def test_infer_from_behavior_video_heavy(self):
        """行为推断: 大量视频观看事件 → visual_score 偏高."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        events = [
            {"event_type": "video_watch", "modality": "visual"}
            for _ in range(20)
        ]

        profile = collector.infer_from_behavior(user_id="user-beh-001", events=events)
        assert isinstance(profile, VARKProfile)
        # 视频观看 → 视觉模态分数偏高
        assert profile.visual_score > 0.5

    def test_infer_from_behavior_text_heavy(self):
        """行为推断: 大量文本阅读事件 → read_write_score 偏高."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        events = [
            {"event_type": "text_read", "modality": "read_write"}
            for _ in range(20)
        ]

        profile = collector.infer_from_behavior(user_id="user-beh-002", events=events)
        # 文本阅读 → 读写模态分数偏高
        assert profile.read_write_score > 0.5

    def test_infer_from_behavior_mixed_modalities(self):
        """行为推断: 混合模态事件应正确分配分数."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        events = (
            [{"event_type": "video_watch", "modality": "visual"}] * 10
            + [{"event_type": "text_read", "modality": "read_write"}] * 10
        )

        profile = collector.infer_from_behavior(user_id="user-beh-003", events=events)
        # 视觉和读写分数都应存在
        assert profile.visual_score > 0.0
        assert profile.read_write_score > 0.0
        # 各占一半 → 分数接近
        assert abs(profile.visual_score - profile.read_write_score) < 0.2

    def test_infer_from_behavior_returns_profile_with_user_id(self):
        """行为推断: 返回的 VARKProfile 应携带 user_id."""
        from dy3_polaris.l1.vark_collector import VARKSurveyCollector

        collector = VARKSurveyCollector()
        events = [{"event_type": "video_watch", "modality": "visual"}]

        profile = collector.infer_from_behavior(user_id="user-beh-004", events=events)
        assert profile.user_id == "user-beh-004"


# ============================================================
# 4. CognitiveLoadBreakdown 增强测试
# ============================================================


class TestCognitiveLoadBreakdownEnhanced:
    """认知负荷三分模型增强测试 (LearningContextBroker.update_cognitive_load_breakdown).

    Sweller 认知负荷三分理论:
    - ICL (内在负荷): 与任务本身复杂度 / 错误率正相关
    - ECL (外在负荷): 与呈现方式 / 慢响应率 / 求助率正相关
    - GCL (生成性负荷): 与主动学习行为正相关
    """

    def test_update_cognitive_load_breakdown_creates_object(self):
        """更新后 envelope.cognitive_load_breakdown 应为 CognitiveLoadBreakdown 对象."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-clb-001")
        interactions = [
            {"response_time_ms": 3000, "is_correct": True,
             "asked_help": False, "content_type": "quiz"},
            {"response_time_ms": 6000, "is_correct": False,
             "asked_help": True, "content_type": "video"},
        ]

        broker.update_cognitive_load_breakdown("sess-clb-001", interactions)
        envelope = broker.get_envelope("sess-clb-001")

        assert envelope.cognitive_load_breakdown is not None
        assert isinstance(envelope.cognitive_load_breakdown, CognitiveLoadBreakdown)

    def test_icl_correlates_with_error_rate(self):
        """ICL (内在负荷) 应与错误率正相关."""
        broker = LearningContextBroker()

        # 低错误率场景: 全对
        broker.build_envelope(user_id="user-001", session_id="sess-icl-low")
        low_error_interactions = [
            {"response_time_ms": 3000, "is_correct": True,
             "asked_help": False, "content_type": "quiz"}
            for _ in range(5)
        ]
        broker.update_cognitive_load_breakdown("sess-icl-low", low_error_interactions)
        low_icl = broker.get_envelope("sess-icl-low").cognitive_load_breakdown.intrinsic_load

        # 高错误率场景: 全错
        broker.build_envelope(user_id="user-001", session_id="sess-icl-high")
        high_error_interactions = [
            {"response_time_ms": 3000, "is_correct": False,
             "asked_help": False, "content_type": "quiz"}
            for _ in range(5)
        ]
        broker.update_cognitive_load_breakdown("sess-icl-high", high_error_interactions)
        high_icl = broker.get_envelope("sess-icl-high").cognitive_load_breakdown.intrinsic_load

        # 高错误率 → ICL 更高
        assert high_icl > low_icl

    def test_ecl_correlates_with_slow_and_help(self):
        """ECL (外在负荷) 应与慢响应率和求助率正相关."""
        broker = LearningContextBroker()

        # 低 ECL: 快速响应 + 无求助
        broker.build_envelope(user_id="user-001", session_id="sess-ecl-low")
        low_ecl_interactions = [
            {"response_time_ms": 2000, "is_correct": True,
             "asked_help": False, "content_type": "text"}
            for _ in range(5)
        ]
        broker.update_cognitive_load_breakdown("sess-ecl-low", low_ecl_interactions)
        low_ecl = broker.get_envelope("sess-ecl-low").cognitive_load_breakdown.extraneous_load

        # 高 ECL: 慢响应 + 全求助
        broker.build_envelope(user_id="user-001", session_id="sess-ecl-high")
        high_ecl_interactions = [
            {"response_time_ms": 15000, "is_correct": True,
             "asked_help": True, "content_type": "text"}
            for _ in range(5)
        ]
        broker.update_cognitive_load_breakdown("sess-ecl-high", high_ecl_interactions)
        high_ecl = broker.get_envelope("sess-ecl-high").cognitive_load_breakdown.extraneous_load

        # 慢响应 + 求助 → ECL 更高
        assert high_ecl > low_ecl

    def test_gcl_correlates_with_active_learning(self):
        """GCL (生成性负荷) 应与主动学习行为正相关."""
        broker = LearningContextBroker()

        # 被动学习: 视频观看 (content_type=video)
        broker.build_envelope(user_id="user-001", session_id="sess-gcl-passive")
        passive_interactions = [
            {"response_time_ms": 4000, "is_correct": True,
             "asked_help": False, "content_type": "video"}
            for _ in range(5)
        ]
        broker.update_cognitive_load_breakdown("sess-gcl-passive", passive_interactions)
        passive_gcl = broker.get_envelope(
            "sess-gcl-passive"
        ).cognitive_load_breakdown.germane_load

        # 主动学习: 交互练习/测验 (content_type=interactive/quiz)
        broker.build_envelope(user_id="user-001", session_id="sess-gcl-active")
        active_interactions = [
            {"response_time_ms": 4000, "is_correct": True,
             "asked_help": False, "content_type": "interactive"}
            for _ in range(5)
        ]
        broker.update_cognitive_load_breakdown("sess-gcl-active", active_interactions)
        active_gcl = broker.get_envelope(
            "sess-gcl-active"
        ).cognitive_load_breakdown.germane_load

        # 主动学习 → GCL 更高
        assert active_gcl > passive_gcl

    def test_total_load_not_exceed_one(self):
        """total_load 不超过 1.0 (极端高负荷场景)."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-clb-max")
        # 极端: 全错 + 全慢 + 全求助 + 全主动
        extreme_interactions = [
            {"response_time_ms": 20000, "is_correct": False,
             "asked_help": True, "content_type": "interactive"}
            for _ in range(10)
        ]
        broker.update_cognitive_load_breakdown("sess-clb-max", extreme_interactions)
        breakdown = broker.get_envelope("sess-clb-max").cognitive_load_breakdown

        assert breakdown is not None
        assert breakdown.total_load <= 1.0

    def test_breakdown_values_in_valid_range(self):
        """三分负荷各维度值应在 [0.0, 1.0] 范围内."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-clb-range")
        interactions = [
            {"response_time_ms": 5000, "is_correct": False,
             "asked_help": True, "content_type": "quiz"},
            {"response_time_ms": 2000, "is_correct": True,
             "asked_help": False, "content_type": "text"},
        ]
        broker.update_cognitive_load_breakdown("sess-clb-range", interactions)
        breakdown = broker.get_envelope("sess-clb-range").cognitive_load_breakdown

        assert 0.0 <= breakdown.intrinsic_load <= 1.0
        assert 0.0 <= breakdown.extraneous_load <= 1.0
        assert 0.0 <= breakdown.germane_load <= 1.0


# ============================================================
# 5. MasteryTrajectory 记录测试
# ============================================================


class TestMasteryTrajectoryRecording:
    """掌握度轨迹记录测试 (LearningContextBroker.update_mastery 增强).

    增强后的 update_mastery 应自动记录 MasteryTrajectory:
    - 调用后可在 envelope 中找到对应 KC 的轨迹
    - 多次更新同一 KC → 轨迹点递增
    - MasteryTrajectory.trend() 应反映掌握度变化趋势
    """

    def test_update_mastery_records_trajectory(self):
        """调用 update_mastery 后, 应能在 envelope 中找到对应的 MasteryTrajectory."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-traj-001",
            initial_mastery=[MasterySnapshot(
                kc_id="kc-traj-1", p_know=0.4, last_practiced_at=int(time.time() * 1000),
            )],
        )

        broker.update_mastery("sess-traj-001", "kc-traj-1", p_know=0.4, is_correct=True)
        envelope = broker.get_envelope("sess-traj-001")

        trajectory = _find_trajectory(envelope, "kc-traj-1")
        assert trajectory is not None
        assert isinstance(trajectory, MasteryTrajectory)
        # 至少有 1 个轨迹点
        assert len(trajectory.points) >= 1

    def test_multiple_updates_increment_trajectory_points(self):
        """多次 update_mastery 同一 kc_id 后, 轨迹点应递增."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-traj-002",
            initial_mastery=[MasterySnapshot(
                kc_id="kc-traj-2", p_know=0.3, last_practiced_at=int(time.time() * 1000),
            )],
        )

        # 第一次更新
        broker.update_mastery("sess-traj-002", "kc-traj-2", p_know=0.3, is_correct=True)
        envelope_1 = broker.get_envelope("sess-traj-002")
        traj_1 = _find_trajectory(envelope_1, "kc-traj-2")
        assert traj_1 is not None
        count_after_1 = len(traj_1.points)

        # 第二次更新
        broker.update_mastery("sess-traj-002", "kc-traj-2", p_know=0.5, is_correct=True)
        envelope_2 = broker.get_envelope("sess-traj-002")
        traj_2 = _find_trajectory(envelope_2, "kc-traj-2")
        count_after_2 = len(traj_2.points)

        # 轨迹点应递增
        assert count_after_2 > count_after_1

        # 第三次更新
        broker.update_mastery("sess-traj-002", "kc-traj-2", p_know=0.7, is_correct=True)
        envelope_3 = broker.get_envelope("sess-traj-002")
        traj_3 = _find_trajectory(envelope_3, "kc-traj-2")
        count_after_3 = len(traj_3.points)
        assert count_after_3 > count_after_2

    def test_trajectory_trend_reflects_improvement(self):
        """MasteryTrajectory.trend() 应反映掌握度上升趋势."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-traj-003",
            initial_mastery=[MasterySnapshot(
                kc_id="kc-traj-3", p_know=0.2, last_practiced_at=int(time.time() * 1000),
            )],
        )

        # 连续答对, 掌握度持续上升
        broker.update_mastery("sess-traj-003", "kc-traj-3", p_know=0.2, is_correct=True)
        broker.update_mastery("sess-traj-003", "kc-traj-3", p_know=0.6, is_correct=True)

        envelope = broker.get_envelope("sess-traj-003")
        trajectory = _find_trajectory(envelope, "kc-traj-3")
        assert trajectory is not None
        # 趋势应为 improving
        assert trajectory.trend() == "improving"

    def test_trajectory_trend_reflects_decline(self):
        """MasteryTrajectory.trend() 应反映掌握度下降趋势."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-traj-004",
            initial_mastery=[MasterySnapshot(
                kc_id="kc-traj-4", p_know=0.8, last_practiced_at=int(time.time() * 1000),
            )],
        )

        # 答错, 掌握度下降
        broker.update_mastery("sess-traj-004", "kc-traj-4", p_know=0.8, is_correct=True)
        broker.update_mastery("sess-traj-004", "kc-traj-4", p_know=0.5, is_correct=False)

        envelope = broker.get_envelope("sess-traj-004")
        trajectory = _find_trajectory(envelope, "kc-traj-4")
        assert trajectory is not None
        # 趋势应为 declining 或 stable (取决于降幅)
        trend = trajectory.trend()
        assert trend in ("declining", "stable")

    def test_trajectory_points_are_mastery_trajectory_points(self):
        """轨迹点应为 MasteryTrajectoryPoint 类型."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-traj-005",
            initial_mastery=[MasterySnapshot(
                kc_id="kc-traj-5", p_know=0.5, last_practiced_at=int(time.time() * 1000),
            )],
        )

        broker.update_mastery("sess-traj-005", "kc-traj-5", p_know=0.5, is_correct=True)
        envelope = broker.get_envelope("sess-traj-005")
        trajectory = _find_trajectory(envelope, "kc-traj-5")

        assert trajectory is not None
        for point in trajectory.points:
            assert isinstance(point, MasteryTrajectoryPoint)
            assert point.kc_id == "kc-traj-5"

    def test_trajectory_records_correct_p_know(self):
        """轨迹点应记录更新后的 p_know 值."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-traj-006",
            initial_mastery=[MasterySnapshot(
                kc_id="kc-traj-6", p_know=0.4, last_practiced_at=int(time.time() * 1000),
            )],
        )

        broker.update_mastery("sess-traj-006", "kc-traj-6", p_know=0.4, is_correct=True)
        envelope = broker.get_envelope("sess-traj-006")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-traj-6")
        trajectory = _find_trajectory(envelope, "kc-traj-6")

        assert trajectory is not None
        latest_point = trajectory.latest()
        assert latest_point is not None
        # 最新轨迹点的 p_know 应与快照一致
        assert latest_point.p_know == pytest.approx(snap.p_know, abs=1e-6)


# ============================================================
# 6. ZPD 便捷接口测试
# ============================================================


class TestZPDRecommendation:
    """ZPD 便捷接口测试 (ContextEnvelope.get_zpd_recommendation).

    返回 (recommended_difficulty, adjustment_direction):
    - 当 irt_ability 存在时: 返回基于 θ 的推荐难度和方向
    - 当 irt_ability 为 None 时: 返回默认值 (0.5, "optimal")
    """

    def test_get_zpd_recommendation_without_irt_ability(self):
        """irt_ability 为 None 时返回默认值 (0.5, "optimal")."""
        envelope = ContextEnvelope(user_id="user-001", session_id="sess-zpd-001")
        # 默认 irt_ability 为 None
        assert envelope.irt_ability is None

        result = envelope.get_zpd_recommendation()
        assert result == (0.5, "optimal")

    def test_get_zpd_recommendation_with_irt_ability(self):
        """irt_ability 存在时返回基于 θ 的推荐难度和方向."""
        envelope = ContextEnvelope(user_id="user-001", session_id="sess-zpd-002")
        envelope.irt_ability = _make_irt_ability(theta=1.0)

        result = envelope.get_zpd_recommendation()
        assert isinstance(result, tuple)
        assert len(result) == 2
        recommended_difficulty, adjustment_direction = result
        # 推荐难度应在合理范围 [0.0, 1.0]
        assert 0.0 <= recommended_difficulty <= 1.0
        # 调整方向应为有效字符串
        assert adjustment_direction in ("increase", "decrease", "optimal")

    def test_get_zpd_recommendation_uses_theta(self):
        """推荐难度应基于 θ: 不同 θ 应产生不同推荐难度."""
        envelope_high = ContextEnvelope(user_id="user-001", session_id="sess-zpd-003")
        envelope_high.irt_ability = _make_irt_ability(theta=1.5)

        envelope_low = ContextEnvelope(user_id="user-001", session_id="sess-zpd-004")
        envelope_low.irt_ability = _make_irt_ability(theta=-1.5)

        rec_high, _ = envelope_high.get_zpd_recommendation()
        rec_low, _ = envelope_low.get_zpd_recommendation()

        # 高能力 (θ=1.5) 推荐难度应高于低能力 (θ=-1.5)
        assert rec_high > rec_low

    def test_get_zpd_recommendation_positive_theta_higher_than_default(self):
        """正 θ 能力推荐难度应高于默认 0.5."""
        envelope = ContextEnvelope(user_id="user-001", session_id="sess-zpd-005")
        envelope.irt_ability = _make_irt_ability(theta=2.0)

        recommended, _ = envelope.get_zpd_recommendation()
        # 高能力 → 推荐更难的内容
        assert recommended > 0.5

    def test_get_zpd_recommendation_negative_theta_lower_than_default(self):
        """负 θ 能力推荐难度应低于默认 0.5."""
        envelope = ContextEnvelope(user_id="user-001", session_id="sess-zpd-006")
        envelope.irt_ability = _make_irt_ability(theta=-2.0)

        recommended, _ = envelope.get_zpd_recommendation()
        # 低能力 → 推荐更易的内容
        assert recommended < 0.5

    def test_get_zpd_recommendation_returns_tuple_type(self):
        """返回值应为 tuple[float, str] 类型."""
        envelope = ContextEnvelope(user_id="user-001", session_id="sess-zpd-007")
        envelope.irt_ability = _make_irt_ability(theta=0.0)

        result = envelope.get_zpd_recommendation()
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)
