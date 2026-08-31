"""T3 IRT 能力评估全链路集成测试.

全链路定义: 答题记录 → IRT 估计 → CAT 选题 → ZPD 校准 → 能力输出

测试覆盖:
1. TestIRTTracingService        — 服务初始化与基础处理
2. TestCATIntegration           — CAT 选题集成
3. TestZPDIntegration           — ZPD 校准集成
4. TestAbilityOutput            — AbilityOutput 下游输出契约
5. TestFullLinkIntegration      — 全链路端到端集成
6. TestWorldSchemeIntegration   — 世界先进方案融合验证

遵循 TDD Red-Green-Refactor: 先写测试 (RED) → 验证失败 → 最小实现 (GREEN).
"""

from __future__ import annotations

import time

import pytest

from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.models import IRTState
from dy3_polaris.l2.store import InMemoryL2Store
from dy3_polaris.l2.ability_assessor import (
    CATSelector,
    IRTEstimator,
    ZPDCalculator,
    IRTTracingService,
    AbilityOutput,
)


# ============================================================
# 辅助函数
# ============================================================


def _bank(bs: list[float], a: float = 1.2, c: float = 0.25) -> list[dict]:
    """构造 CAT 标准键格式题库 {"item_id", "a", "b", "c"}."""
    return [{"item_id": f"q{i}", "a": a, "b": b, "c": c} for i, b in enumerate(bs)]


def _event(
    learner_id: str = "learner_001",
    kp_id: str = "kp_math_01",
    correct: bool = True,
    difficulty: float = 0.5,
    ts: float | None = None,
) -> AnswerEvent:
    """构造答题事件."""
    return AnswerEvent(
        learner_id=learner_id,
        kp_id=kp_id,
        correct=correct,
        difficulty=difficulty,
        timestamp=ts if ts is not None else time.time(),
    )


# ============================================================
# 1. TestIRTTracingService — 服务初始化与基础处理
# ============================================================


class TestIRTTracingService:
    """IRT 全链路编排服务 — 答题记录 → IRT 估计 → ZPD 校准 → 能力输出."""

    def test_service_initializes(self):
        """IRTTracingService 可用 store=None 创建 (回退 InMemoryL2Store)."""
        service = IRTTracingService(store=None)
        assert service is not None
        assert service.store is not None
        assert service.irt_estimator is not None
        assert service.cat_selector is not None
        assert service.zpd_calculator is not None

    def test_service_process_answer_event(self):
        """处理单条答题事件返回 AbilityOutput."""
        service = IRTTracingService(store=InMemoryL2Store())
        event = _event(correct=True, difficulty=0.5)
        output = service.process(event)
        assert output is not None
        assert isinstance(output, AbilityOutput)
        assert output.learner_id == "learner_001"
        assert output.response_count == 1
        # theta 在有效范围内
        assert -3.0 <= output.theta <= 3.0
        assert output.se > 0.0

    def test_service_correct_increases_theta(self):
        """连续答对, theta 递增."""
        service = IRTTracingService(store=InMemoryL2Store())
        ts = time.time()
        thetas = []
        for i in range(5):
            output = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
            thetas.append(output.theta)
        # theta 应递增
        for i in range(1, len(thetas)):
            assert thetas[i] > thetas[i - 1]
        # 答对后 theta 应高于初始 0.0
        assert thetas[-1] > 0.0

    def test_service_wrong_decreases_theta(self):
        """连续答错, theta 递减."""
        service = IRTTracingService(store=InMemoryL2Store())
        ts = time.time()
        thetas = []
        for i in range(5):
            output = service.process(_event(correct=False, difficulty=0.5, ts=ts + i))
            thetas.append(output.theta)
        # theta 应递减
        for i in range(1, len(thetas)):
            assert thetas[i] < thetas[i - 1]
        # 答错后 theta 应低于初始 0.0
        assert thetas[-1] < 0.0

    def test_service_se_decreases_with_data(self):
        """数据越多, 标准误 SE 越小."""
        service = IRTTracingService(store=InMemoryL2Store())
        ts = time.time()
        ses = []
        for i in range(12):
            output = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
            ses.append(output.se)
        # SE 应随数据增加而减小 (精度提升)
        assert ses[-1] < ses[0]
        # 最终 SE 低于初始先验 0.3
        assert ses[-1] < 0.3

    def test_service_batch_process(self):
        """批量处理多个事件."""
        service = IRTTracingService(store=InMemoryL2Store())
        ts = time.time()
        events = [
            _event(correct=True, difficulty=0.4, ts=ts),
            _event(correct=True, difficulty=0.4, ts=ts + 1),
            _event(correct=False, difficulty=0.7, ts=ts + 2),
            _event(learner_id="learner_002", correct=True, difficulty=0.5, ts=ts + 3),
        ]
        outputs = service.batch_process(events)
        assert len(outputs) == 4
        assert all(isinstance(o, AbilityOutput) for o in outputs)
        # 顺序按时间戳升序
        assert outputs[0].last_updated_ts <= outputs[-1].last_updated_ts
        # learner_001 答对两次, theta 应高于答错的那次
        assert outputs[1].theta > outputs[2].theta


# ============================================================
# 2. TestCATIntegration — CAT 选题集成
# ============================================================


class TestCATIntegration:
    """CAT 自适应选题在全链路中的集成."""

    def test_select_next_item(self):
        """服务可通过 CAT 选题."""
        service = IRTTracingService()
        items = _bank([-2.0, 0.0, 2.0])
        service.set_item_bank(items)
        chosen = service.select_next_item(items, administered_ids=set())
        assert chosen is not None
        assert "item_id" in chosen

    def test_cat_uses_current_theta(self):
        """CAT 选题使用更新后的 theta."""
        service = IRTTracingService()
        items = _bank([-2.0, 0.0, 2.0])
        service.set_item_bank(items)
        ts = time.time()
        # 连续答对多次, 推高 theta
        for i in range(20):
            service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
        snap = service.get_ability_snapshot("learner_001")
        assert snap["theta"] > 0.5  # theta 已上升
        # CAT 应基于当前 theta 选题 (Fisher 信息最大者)
        est = service.irt_estimator
        theta = snap["theta"]
        chosen = service.select_next_item(items, administered_ids=set())
        assert chosen is not None
        infos = {
            it["item_id"]: est.information(theta, it["a"], it["b"], it["c"])
            for it in items
        }
        assert chosen["item_id"] == max(infos, key=infos.get)

    def test_cat_fisher_info_strategy(self):
        """Fisher 信息策略选择高信息量题目."""
        service = IRTTracingService()  # 默认 fisher_info 策略
        est = service.irt_estimator
        # theta 默认 0.0, b=0 的题 Fisher 信息最大 (2PL 峰值在 theta=b)
        items = [
            {"item_id": "q_low", "a": 1.0, "b": -2.0, "c": 0.0},
            {"item_id": "q_mid", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q_high", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        chosen = service.select_next_item(items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] == "q_mid"
        # 验证 q_mid 信息量最大
        assert est.information(0.0, 1.0, 0.0, 0.0) > est.information(0.0, 1.0, 2.0, 0.0)

    def test_cat_zpd_aware_strategy(self):
        """ZPD 感知策略过滤挫败区题目."""
        zpd = ZPDCalculator()
        cat = CATSelector(
            selection_strategy="zpd_aware",
            zpd_calculator=zpd,
            estimator=IRTEstimator(),
        )
        service = IRTTracingService(cat_selector=cat)
        est = service.irt_estimator
        # theta=0 (默认): b=3 为挫败区 (P=0.047 <= 0.3)
        items = [
            {"item_id": "easy", "a": 1.0, "b": -2.0, "c": 0.0},  # zpd
            {"item_id": "mid", "a": 1.0, "b": 0.0, "c": 0.0},     # zpd
            {"item_id": "hard", "a": 1.0, "b": 3.0, "c": 0.0},    # frustration
        ]
        chosen = service.select_next_item(items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] != "hard"  # 挫败区被过滤
        p = est.predict_correct(0.0, chosen["a"], chosen["b"], chosen["c"])
        assert p > 0.3  # 不在挫败区

    def test_should_stop_termination(self):
        """终止条件判定正确."""
        service = IRTTracingService()
        # SE 达标 (低于阈值 0.3) → 终止
        assert service.should_stop(se=0.1, count=10) is True
        # SE 未达标且未达题量上限 → 不终止
        assert service.should_stop(se=0.5, count=5) is False
        # 题量达上限 (>=20) → 终止
        assert service.should_stop(se=0.5, count=20) is True
        assert service.should_stop(se=0.5, count=25) is True


# ============================================================
# 3. TestZPDIntegration — ZPD 校准集成
# ============================================================


class TestZPDIntegration:
    """ZPD (最近发展区) 校准在全链路中的集成."""

    def test_zpd_calculation(self):
        """ZPD 由 theta 和题库计算得出."""
        service = IRTTracingService()
        items = _bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        service.set_item_bank(items)
        zpd_items = CATSelector._to_zpd_items(items)
        zpd = service.zpd_calculator.calculate_zpd(0.0, zpd_items)
        assert zpd.zpd_lower <= zpd.zpd_upper
        # 通过全链路处理后, 输出含合法 zpd_zone
        out = service.process(_event(correct=True, difficulty=0.5))
        assert out.zpd_zone in ("independent", "zpd", "frustration")

    def test_zpd_classification(self):
        """题目被分类为 independent / zpd / frustration 三区."""
        service = IRTTracingService()
        zpd = service.zpd_calculator
        theta = 0.0
        # b=-3 (易, P>0.9) → independent
        assert zpd.classify_item(theta, -3.0, 1.0, 0.0) == "independent"
        # b=0 (中, 0.3<P<0.9) → zpd
        assert zpd.classify_item(theta, 0.0, 1.0, 0.0) == "zpd"
        # b=3 (难, P<=0.3) → frustration
        assert zpd.classify_item(theta, 3.0, 1.0, 0.0) == "frustration"

    def test_zpd_recommend_difficulty(self):
        """ZPD 推荐合适难度."""
        service = IRTTracingService()
        items = _bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        service.set_item_bank(items)
        zpd_items = CATSelector._to_zpd_items(items)
        rec = service.zpd_calculator.recommend_difficulty(0.0, 0.5, zpd_items)
        assert -3.0 <= rec <= 3.0
        # 全链路输出含合法 recommended_difficulty
        out = service.process(_event(correct=True, difficulty=0.5))
        assert -3.0 <= out.recommended_difficulty <= 3.0

    def test_zpd_scaffold_selection(self):
        """支架感知选题可用."""
        service = IRTTracingService()
        items = _bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
        service.set_item_bank(items)
        out = service.process(_event(correct=True, difficulty=0.5))
        theta = out.theta
        chosen = service.cat_selector.select_with_scaffold(
            theta, items, set(), scaffold_level=0.5
        )
        assert chosen is not None
        assert chosen["item_id"] in {it["item_id"] for it in items}


# ============================================================
# 4. TestAbilityOutput — 下游输出契约
# ============================================================


class TestAbilityOutput:
    """AbilityOutput 标准化输出契约 (供 T2/T4/T5 下游消费)."""

    def test_output_fields(self):
        """AbilityOutput 包含所有必需字段."""
        output = AbilityOutput(
            learner_id="l1",
            theta=1.5,
            se=0.2,
            response_count=10,
            p_correct_next=0.75,
            zpd_zone="zpd",
            recommended_difficulty=0.5,
            confidence=0.83,
            next_item_id="q5",
            termination_flag=False,
            last_updated_ts=1000.0,
        )
        assert output.learner_id == "l1"
        assert output.theta == 1.5
        assert output.se == 0.2
        assert output.response_count == 10
        assert output.p_correct_next == 0.75
        assert output.zpd_zone == "zpd"
        assert output.recommended_difficulty == 0.5
        assert output.confidence == 0.83
        assert output.next_item_id == "q5"
        assert output.termination_flag is False
        assert output.last_updated_ts == 1000.0

    def test_output_to_dict(self):
        """序列化为字典."""
        output = AbilityOutput(
            learner_id="l1",
            theta=1.0,
            se=0.3,
            response_count=5,
            p_correct_next=0.6,
            zpd_zone="zpd",
            recommended_difficulty=0.0,
            confidence=0.77,
            next_item_id=None,
            termination_flag=True,
            last_updated_ts=500.0,
        )
        d = output.to_dict()
        assert d["learner_id"] == "l1"
        assert d["theta"] == 1.0
        assert d["se"] == 0.3
        assert d["response_count"] == 5
        assert d["p_correct_next"] == 0.6
        assert d["zpd_zone"] == "zpd"
        assert d["recommended_difficulty"] == 0.0
        assert d["confidence"] == 0.77
        assert d["next_item_id"] is None
        assert d["termination_flag"] is True
        assert d["last_updated_ts"] == 500.0

    def test_output_from_dict(self):
        """从字典反序列化."""
        d = {
            "learner_id": "l2",
            "theta": -0.5,
            "se": 0.4,
            "response_count": 3,
            "p_correct_next": 0.45,
            "zpd_zone": "frustration",
            "recommended_difficulty": 1.2,
            "confidence": 0.71,
            "next_item_id": "q9",
            "termination_flag": False,
            "last_updated_ts": 250.0,
        }
        output = AbilityOutput.from_dict(d)
        assert output.learner_id == "l2"
        assert output.theta == -0.5
        assert output.se == 0.4
        assert output.response_count == 3
        assert output.p_correct_next == 0.45
        assert output.zpd_zone == "frustration"
        assert output.recommended_difficulty == 1.2
        assert output.confidence == 0.71
        assert output.next_item_id == "q9"
        assert output.termination_flag is False
        assert output.last_updated_ts == 250.0

    def test_output_roundtrip(self):
        """序列化-反序列化往返保持数据一致."""
        output = AbilityOutput(
            learner_id="l1",
            theta=2.1,
            se=0.15,
            response_count=20,
            p_correct_next=0.92,
            zpd_zone="independent",
            recommended_difficulty=-1.0,
            confidence=0.87,
            next_item_id="q12",
            termination_flag=True,
            last_updated_ts=2000.0,
        )
        restored = AbilityOutput.from_dict(output.to_dict())
        assert restored == output

    def test_output_roundtrip_with_none_next_item(self):
        """next_item_id=None 时往返仍一致."""
        output = AbilityOutput(
            learner_id="l1",
            theta=0.0,
            se=0.3,
            response_count=0,
            p_correct_next=0.5,
            zpd_zone="zpd",
            recommended_difficulty=0.0,
            confidence=0.77,
            next_item_id=None,
            termination_flag=False,
            last_updated_ts=0.0,
        )
        restored = AbilityOutput.from_dict(output.to_dict())
        assert restored == output
        assert restored.next_item_id is None


# ============================================================
# 5. TestFullLinkIntegration — 全链路端到端集成
# ============================================================


class TestFullLinkIntegration:
    """答题记录 → IRT 估计 → CAT 选题 → ZPD 校准 → 能力输出 端到端."""

    def test_full_link_single_learner(self):
        """单学习者走完 IRT → CAT → ZPD → 输出 全链路."""
        service = IRTTracingService(store=InMemoryL2Store())
        items = _bank([-2.0, -1.0, 0.0, 1.0, 2.0])
        service.set_item_bank(items)
        ts = time.time()
        outputs = []
        for i in range(10):
            out = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
            outputs.append(out)
        # 全链路输出合法
        assert all(o.learner_id == "learner_001" for o in outputs)
        assert outputs[-1].response_count == 10
        assert outputs[-1].theta > 0.0  # 答对, theta 上升
        # CAT 可基于当前能力选题
        chosen = service.select_next_item(items, administered_ids=set())
        assert chosen is not None
        # 输出含 ZPD 区分类
        assert outputs[-1].zpd_zone in ("independent", "zpd", "frustration")

    def test_full_link_adaptive_progression(self):
        """theta 随作答增加而收敛 (增量递减)."""
        service = IRTTracingService(store=InMemoryL2Store())
        ts = time.time()
        thetas = []
        for i in range(20):
            out = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
            thetas.append(out.theta)
        # theta 上升
        assert thetas[-1] > thetas[0]
        # 增量递减 (收敛): 后段增量 < 前段增量
        deltas = [thetas[i + 1] - thetas[i] for i in range(len(thetas) - 1)]
        assert deltas[-1] < deltas[0]

    def test_full_link_termination_and_output(self):
        """测试正确终止并输出最终能力."""
        service = IRTTracingService(store=InMemoryL2Store())
        service.set_item_bank(_bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
        ts = time.time()
        final_output = None
        for i in range(25):
            out = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
            if out.termination_flag:
                final_output = out
                break
        assert final_output is not None
        assert final_output.termination_flag is True
        assert final_output.learner_id == "learner_001"
        assert final_output.response_count >= 1
        assert final_output.theta > 0.0  # 答对, 能力上升

    def test_full_link_cold_start(self):
        """冷启动 (无数据) 返回群体平均值."""
        service = IRTTracingService(store=InMemoryL2Store())
        snapshot = service.get_ability_snapshot("new_learner")
        # 无数据时回退群体先验: theta=0.0, se=0.3
        assert snapshot["theta"] == pytest.approx(0.0)
        assert snapshot["se"] == pytest.approx(0.3)
        assert snapshot["response_count"] == 0
        assert snapshot["learner_id"] == "new_learner"

    def test_full_link_graceful_degradation(self):
        """服务优雅处理边界情况."""
        # 无 store, 无题库 — 仅 IRT 引擎工作
        service = IRTTracingService(store=None)
        event = _event(correct=True, difficulty=0.5)
        out = service.process(event)
        assert out is not None
        assert -3.0 <= out.theta <= 3.0
        # 极端难度 (0.0 / 1.0 → b=-3 / b=3) 不报错
        out_easy = service.process(_event(correct=True, difficulty=0.0))
        assert out_easy is not None
        out_hard = service.process(_event(correct=False, difficulty=1.0))
        assert out_hard is not None
        # 空题库选题返回 None
        assert service.select_next_item([], administered_ids=set()) is None
        # 空批量返回空列表
        assert service.batch_process([]) == []


# ============================================================
# 6. TestWorldSchemeIntegration — 世界先进方案融合验证
# ============================================================


class TestWorldSchemeIntegration:
    """验证世界先进方案在全链路中的融合."""

    def test_bayesian_eap_update(self):
        """贝叶斯 EAP 后验更新符合预期."""
        service = IRTTracingService()
        est = service.irt_estimator
        state = IRTState(theta=0.0, se=0.3)
        item = {"a": 1.2, "b": 0.0, "c": 0.25}
        # 答对 → theta 上升
        new_state = est.update_theta(state, item, True)
        assert new_state.theta > 0.0
        assert new_state.response_count == 1
        # 答错 → theta 下降
        new_state_wrong = est.update_theta(state, item, False)
        assert new_state_wrong.theta < 0.0
        assert new_state_wrong.response_count == 1

    def test_mle_batch_estimation(self):
        """MLE 批量估计可用."""
        service = IRTTracingService()
        est = service.irt_estimator
        # 全对 (易题) → 高 theta
        responses_correct = [({"a": 1.0, "b": -1.0, "c": 0.0}, True)] * 10
        state_correct = est.estimate_mle(responses_correct)
        assert state_correct.theta > 0.5
        assert state_correct.response_count == 10
        # 全错 → 低 theta
        responses_wrong = [({"a": 1.0, "b": -1.0, "c": 0.0}, False)] * 10
        state_wrong = est.estimate_mle(responses_wrong)
        assert state_wrong.theta < state_correct.theta

    def test_newton_raphson_convergence(self):
        """Newton-Raphson 正确收敛."""
        service = IRTTracingService()
        est = service.irt_estimator
        responses = [
            ({"a": 1.5, "b": 0.5, "c": 0.2}, True),
            ({"a": 1.5, "b": 0.5, "c": 0.2}, False),
        ] * 5
        state = est.estimate_mle_newton_raphson(responses)
        # 收敛到有效范围
        assert -3.0 <= state.theta <= 3.0
        assert state.se > 0.0
        assert state.response_count == 10
        # 与网格 MLE 结果接近
        state_grid = est.estimate_mle(responses)
        assert abs(state.theta - state_grid.theta) < 0.5

    def test_hierarchical_bayesian(self):
        """分层贝叶斯估计可用 (向群体先验收缩)."""
        service = IRTTracingService()
        est = service.irt_estimator
        responses_by_learner = {
            "l_high": [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 10,
            "l_low": [({"a": 1.0, "b": 0.0, "c": 0.0}, False)] * 10,
        }
        results = est.estimate_hierarchical_bayesian(
            responses_by_learner,
            group_prior={"mean": 0.0, "sd": 1.0},
            shrinkage=0.5,
        )
        assert "l_high" in results
        assert "l_low" in results
        # 全对学习者能力高于全错学习者
        assert results["l_high"].theta > results["l_low"].theta
        # 收缩后能力估计比纯 MLE 更接近群体均值 0
        mle_high = est.estimate_mle_newton_raphson(
            responses_by_learner["l_high"]
        ).theta
        assert abs(results["l_high"].theta) < abs(mle_high)

    def test_catr_fisher_info_selection(self):
        """catR 风格 Fisher 信息选题."""
        service = IRTTracingService()
        est = service.irt_estimator
        theta = 0.5
        items = [
            {"item_id": "q_far", "a": 1.0, "b": 2.5, "c": 0.0},
            {"item_id": "q_near", "a": 1.0, "b": 0.5, "c": 0.0},  # b=theta, 信息最大
            {"item_id": "q_low_a", "a": 0.3, "b": 0.5, "c": 0.0},  # 低区分度
        ]
        chosen = service.cat_selector.select_next(
            theta=theta, available_items=items, administered_ids=set()
        )
        assert chosen["item_id"] == "q_near"
        info_near = est.information(theta, 1.0, 0.5, 0.0)
        info_far = est.information(theta, 1.0, 2.5, 0.0)
        assert info_near > info_far

    def test_zpd_vygotsky(self):
        """Vygotsky ZPD 三区分类."""
        service = IRTTracingService()
        zpd = service.zpd_calculator
        theta = 0.0
        assert zpd.classify_item(theta, -3.0, 1.0, 0.0) == "independent"
        assert zpd.classify_item(theta, 0.0, 1.0, 0.0) == "zpd"
        assert zpd.classify_item(theta, 3.0, 1.0, 0.0) == "frustration"

    def test_output_contract_for_downstream(self):
        """输出契约可被 T2/T4/T5 下游消费."""
        service = IRTTracingService(store=InMemoryL2Store())
        service.set_item_bank(_bank([-2.0, -1.0, 0.0, 1.0, 2.0]))
        ts = time.time()
        out = None
        for i in range(5):
            out = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
        assert out is not None
        d = out.to_dict()
        required_fields = [
            "learner_id", "theta", "se", "response_count",
            "p_correct_next", "zpd_zone", "recommended_difficulty",
            "confidence", "next_item_id", "termination_flag", "last_updated_ts",
        ]
        for field in required_fields:
            assert field in d, f"缺失下游契约字段: {field}"
        # 字段值合法
        assert 0.0 <= d["p_correct_next"] <= 1.0
        assert d["zpd_zone"] in ("independent", "zpd", "frustration")
        assert 0.0 <= d["confidence"] <= 1.0
        assert isinstance(d["termination_flag"], bool)
        assert isinstance(d["response_count"], int)
        # 往返序列化稳定 (可被下游持久化/传输)
        restored = AbilityOutput.from_dict(d)
        assert restored == out
