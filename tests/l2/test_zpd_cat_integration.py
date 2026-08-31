"""L2 ZPD-CAT 深度集成测试 — ZPD 感知选题 / 支架感知选题 / ZPD 终止条件.

遵循 TDD Red-Green-Refactor. 测试覆盖:
- ZPD 感知选题 (zpd_aware): 挫败区过滤 / ZPD 优先 / Fisher 信息选优 /
  回退 b_match / 自定义阈值 / 排除已答 / 策略分派.
- 支架感知选题 (select_with_scaffold): 不同 scaffold_level 的选题行为 /
  单调性 / 无 ZPD 计算器回退 / 排除已答 / 曝光记录.
- ZPD 终止条件 (should_stop_zpd): 三区覆盖检查 / SE 终止 / 最低题量 /
  题量上限.

注意:
- 不使用 mock, 使用真实 CATSelector / IRTEstimator / ZPDCalculator.
- 题目字典使用 CAT 标准 {"item_id", "a", "b", "c"} 键.
"""

from __future__ import annotations

import pytest

from dy3_polaris.l2.ability_assessor import CATSelector, IRTEstimator, ZPDCalculator
from dy3_polaris.l2.ability_assessor.cat import _VALID_STRATEGIES


# ============================================================
# 辅助
# ============================================================


def _bank(bs: list[float], a: float = 1.0, c: float = 0.0) -> list[dict]:
    """构造题目库 (CAT 键格式)."""
    return [{"item_id": f"q{i}", "a": a, "b": b, "c": c} for i, b in enumerate(bs)]


def _p(est: IRTEstimator, theta: float, item: dict) -> float:
    return est.predict_correct(theta, item["a"], item["b"], item["c"])


# ============================================================
# 1. ZPD 感知选题策略测试
# ============================================================


class TestZPDAwareStrategy:
    """zpd_aware 策略 — 挫败区过滤 / ZPD 优先 / Fisher 选优 / 回退."""

    def test_zpd_aware_is_valid_strategy(self):
        """zpd_aware 在合法策略集合中."""
        assert "zpd_aware" in _VALID_STRATEGIES

    def test_zpd_aware_filters_frustration(self):
        """存在 ZPD 候选时, 挫败区题目 (P<=0.3) 永不被选中."""
        est = IRTEstimator()
        # theta=0, a=1.0: b=-2(ZPD), b=0(ZPD), b=2(frustration P=0.119)
        items = _bank([-2.0, 0.0, 2.0], a=1.0)
        sel = CATSelector(selection_strategy="zpd_aware")
        chosen = sel.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] != "q2"  # b=2 (frustration) 不应被选
        assert _p(est, 0.0, chosen) > 0.3

    def test_zpd_aware_prefers_zpd_over_independent(self):
        """存在 ZPD 候选时优先 ZPD (而非独立区)."""
        est = IRTEstimator()
        # b=-3 (independent P=0.953), b=0 (ZPD P=0.5)
        items = _bank([-3.0, 0.0], a=1.0)
        sel = CATSelector(selection_strategy="zpd_aware")
        chosen = sel.select_next(theta=0.0, available_items=items, administered_ids=set())
        # 应选 ZPD 题 b=0 (Fisher 最大), 而非独立区 b=-3
        assert chosen["item_id"] == "q1"
        assert 0.3 < _p(est, 0.0, chosen) < 0.9

    def test_zpd_aware_uses_fisher_among_zpd(self):
        """在 ZPD 候选中用 Fisher 信息选优 (与 b_match 不同时验证)."""
        est = IRTEstimator()
        # 两道 ZPD 题 (3PL, c=0.3, a=1.5), theta=0:
        #   b=+0.4: P≈0.548 ; b=-0.4: P≈0.752 (均在 ZPD 区)
        # Fisher 信息: b=-0.4 更大 (3PL 信息峰值偏向较高 P);
        # b_match (距 theta 等距 0.4) 取首个 b=+0.4 → 二者不同, 可区分
        items = [
            {"item_id": "match_pick", "a": 1.5, "b": 0.4, "c": 0.3},
            {"item_id": "fisher_pick", "a": 1.5, "b": -0.4, "c": 0.3},
        ]
        sel = CATSelector(selection_strategy="zpd_aware")
        chosen = sel.select_next(theta=0.0, available_items=items, administered_ids=set())
        # Fisher 选 fisher_pick (b=-0.4), b_match 会选 match_pick (等距取首)
        assert chosen["item_id"] == "fisher_pick"
        info_match = est.information(0.0, 1.5, 0.4, 0.3)
        info_fisher = est.information(0.0, 1.5, -0.4, 0.3)
        assert info_fisher > info_match

    def test_zpd_aware_fallback_bmatch_when_no_zpd(self):
        """无 ZPD 候选 (全独立区) 时回退 b_match."""
        est = IRTEstimator()
        # b=-3, -2.5 均为独立区 (P>0.9), 无 ZPD 候选
        items = _bank([-3.0, -2.5], a=1.0)
        sel = CATSelector(selection_strategy="zpd_aware")
        chosen = sel.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen is not None
        # b_match 选 b 最接近 theta=0 → b=-2.5
        assert chosen["b"] == pytest.approx(-2.5)

    def test_zpd_aware_fallback_when_all_frustration(self):
        """全部候选均在挫败区时回退 b_match (在全部候选上)."""
        items = _bank([2.0, 3.0], a=1.0)  # theta=0: P=0.119, 0.047 全挫败
        sel = CATSelector(selection_strategy="zpd_aware")
        chosen = sel.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen is not None
        # b_match 选 b 最接近 theta=0 → b=2.0
        assert chosen["b"] == pytest.approx(2.0)

    def test_zpd_aware_empty_returns_none(self):
        """无候选返回 None."""
        sel = CATSelector(selection_strategy="zpd_aware")
        assert sel.select_next(theta=0.0, available_items=[], administered_ids=set()) is None

    def test_zpd_aware_respects_custom_thresholds(self):
        """注入自定义阈值的 ZPDCalculator 时, zpd_aware 使用其阈值."""
        # independent_p=0.6, frustration_p=0.4 → ZPD 区间更窄 (0.4<P<0.6)
        zpd = ZPDCalculator(independent_p=0.6, frustration_p=0.4)
        sel = CATSelector(selection_strategy="zpd_aware", zpd_calculator=zpd)
        ind, frus = sel._zpd_thresholds()
        assert ind == pytest.approx(0.6)
        assert frus == pytest.approx(0.4)

    def test_zpd_aware_default_thresholds_without_calculator(self):
        """未注入 zpd_calculator 时使用默认阈值 (0.9, 0.3)."""
        sel = CATSelector(selection_strategy="zpd_aware")
        ind, frus = sel._zpd_thresholds()
        assert ind == pytest.approx(0.9)
        assert frus == pytest.approx(0.3)

    def test_zpd_aware_excludes_administered(self):
        """zpd_aware 排除已答题目."""
        items = _bank([-2.0, 0.0, 0.5], a=1.0)
        sel = CATSelector(selection_strategy="zpd_aware")
        # 标记 b=0 (Fisher 最优 ZPD) 已答 → 应选其他 ZPD 题
        chosen = sel.select_next(theta=0.0, available_items=items, administered_ids={"q1"})
        assert chosen is not None
        assert chosen["item_id"] != "q1"

    def test_zpd_aware_records_exposure(self):
        """zpd_aware 选题后记录曝光."""
        items = _bank([-2.0, 0.0], a=1.0)
        sel = CATSelector(selection_strategy="zpd_aware")
        sel.select_next(theta=0.0, available_items=items, administered_ids=set())
        stats = sel.get_exposure_stats()
        assert sum(stats.values()) == 1


# ============================================================
# 2. 支架感知选题测试
# ============================================================


class TestSelectWithScaffold:
    """select_with_scaffold — 按支架水平在 ZPD 区间选题."""

    def test_scaffold_zero_picks_independent(self):
        """scaffold_level=0 → target_b=zpd_lower (独立区, 简单题)."""
        zpd = ZPDCalculator()
        sel = CATSelector(zpd_calculator=zpd)
        # theta=0, a=1.0: b=-3 (independent), b=-2..0 (ZPD), b>=1 (frustration)
        items = _bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], a=1.0)
        chosen = sel.select_with_scaffold(0.0, items, set(), scaffold_level=0.0)
        z = zpd.calculate_zpd(0.0, sel._to_zpd_items(items))
        # target_b = zpd_lower, 选 b 最接近者
        assert chosen is not None
        assert chosen["b"] == pytest.approx(z.zpd_lower, abs=1.0)

    def test_scaffold_one_picks_upper(self):
        """scaffold_level=1 → target_b=zpd_upper (ZPD 上界, 挑战题)."""
        zpd = ZPDCalculator()
        sel = CATSelector(zpd_calculator=zpd)
        items = _bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], a=1.0)
        chosen = sel.select_with_scaffold(0.0, items, set(), scaffold_level=1.0)
        z = zpd.calculate_zpd(0.0, sel._to_zpd_items(items))
        assert chosen is not None
        assert chosen["b"] == pytest.approx(z.zpd_upper, abs=1.0)

    def test_scaffold_monotonic_difficulty(self):
        """scaffold_level 越高, 选中题目难度 b 越大 (非递减)."""
        zpd = ZPDCalculator()
        items = _bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], a=1.0)
        bs = []
        for sl in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sel = CATSelector(zpd_calculator=zpd)
            chosen = sel.select_with_scaffold(0.0, items, set(), scaffold_level=sl)
            bs.append(chosen["b"])
        # 非递减
        for i in range(len(bs) - 1):
            assert bs[i] <= bs[i + 1] + 1e-9
        # 端点严格递增
        assert bs[0] < bs[-1]

    def test_scaffold_default_level_is_center(self):
        """scaffold_level 默认 0.5 (ZPD 中心)."""
        zpd = ZPDCalculator()
        sel = CATSelector(zpd_calculator=zpd)
        items = _bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], a=1.0)
        chosen = sel.select_with_scaffold(0.0, items, set())  # 默认 scaffold_level
        z = zpd.calculate_zpd(0.0, sel._to_zpd_items(items))
        center = z.zpd_lower + 0.5 * (z.zpd_upper - z.zpd_lower)
        assert chosen is not None
        # 选中的 b 是最接近中心者
        min_dist = min(abs(it["b"] - center) for it in items)
        assert abs(chosen["b"] - center) == pytest.approx(min_dist, abs=1e-9)

    def test_scaffold_no_zpd_calculator_fallback(self):
        """未注入 zpd_calculator 时回退到 target_b=theta."""
        sel = CATSelector()  # 无 zpd_calculator
        items = _bank([-2.0, 0.0, 1.0], a=1.0)
        chosen = sel.select_with_scaffold(0.5, items, set(), scaffold_level=0.5)
        # target_b = theta = 0.5 → 选 b 最接近 0.5 者 (b=1.0, dist 0.5; b=0.0, dist 0.5)
        # 等距取首 (b=0.0 在前? 列表顺序 -2,0,1 → b=0.0 dist 0.5 先命中)
        assert chosen is not None
        assert chosen["b"] in (0.0, 1.0)

    def test_scaffold_excludes_administered(self):
        """支架感知选题排除已答题目."""
        zpd = ZPDCalculator()
        sel = CATSelector(zpd_calculator=zpd)
        items = _bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], a=1.0)
        # 排除最接近 target 的题, 应选次优
        chosen = sel.select_with_scaffold(0.0, items, {"q0"}, scaffold_level=0.0)
        assert chosen is not None
        assert chosen["item_id"] != "q0"

    def test_scaffold_empty_returns_none(self):
        """无可用题目返回 None."""
        zpd = ZPDCalculator()
        sel = CATSelector(zpd_calculator=zpd)
        assert sel.select_with_scaffold(0.0, [], set(), scaffold_level=0.5) is None

    def test_scaffold_records_exposure(self):
        """支架感知选题记录曝光."""
        zpd = ZPDCalculator()
        sel = CATSelector(zpd_calculator=zpd)
        items = _bank([-3.0, -2.0, -1.0, 0.0], a=1.0)
        sel.select_with_scaffold(0.0, items, set(), scaffold_level=0.5)
        assert sum(sel.get_exposure_stats().values()) == 1


# ============================================================
# 3. ZPD 终止条件测试
# ============================================================


class TestShouldStopZPD:
    """should_stop_zpd — 标准 CAT 终止 + ZPD 覆盖检查."""

    def test_stop_at_max_items(self):
        """题量达上限即终止 (无论 SE / 覆盖)."""
        sel = CATSelector()
        administered = _bank([-3.0, 0.0, 3.0], a=1.0)
        assert sel.should_stop_zpd(0.5, 25, 0.0, administered) is True

    def test_stop_when_se_met_and_covered(self):
        """SE 达标 + 最低题量 + 三区覆盖 → 终止."""
        sel = CATSelector()
        # theta=0: b=-3 (independent), b=0 (zpd), b=3 (frustration)
        administered = _bank([-3.0, 0.0, 3.0], a=1.0)
        assert sel.should_stop_zpd(0.2, 6, 0.0, administered) is True

    def test_no_stop_when_se_met_but_not_covered(self):
        """SE 达标但 ZPD 三区未全覆盖 → 不终止."""
        sel = CATSelector()
        # 仅 ZPD 区题目 (b=0), 缺独立区与挫败区
        administered = _bank([0.0, 0.5], a=1.0)
        assert sel.should_stop_zpd(0.2, 6, 0.0, administered) is False

    def test_no_stop_when_se_not_met(self):
        """SE 未达标 (即使覆盖) → 不终止."""
        sel = CATSelector()
        administered = _bank([-3.0, 0.0, 3.0], a=1.0)
        assert sel.should_stop_zpd(0.4, 6, 0.0, administered) is False

    def test_no_stop_when_below_min_items(self):
        """题数不足 min_items (即使 SE 达标 + 覆盖) → 不终止."""
        sel = CATSelector()
        administered = _bank([-3.0, 0.0, 3.0], a=1.0)
        assert sel.should_stop_zpd(0.2, 3, 0.0, administered, min_items=5) is False

    def test_no_stop_default_conditions(self):
        """默认条件 (SE 未达、未达上限、未覆盖) → 不终止."""
        sel = CATSelector()
        administered = _bank([0.0], a=1.0)
        assert sel.should_stop_zpd(0.4, 3, 0.0, administered) is False

    def test_custom_min_items_allows_earlier_stop(self):
        """自定义 min_items 较小时, 满足覆盖即可更早终止."""
        sel = CATSelector()
        administered = _bank([-3.0, 0.0, 3.0], a=1.0)
        # min_items=3, count=3, SE 达标, 覆盖 → 终止
        assert sel.should_stop_zpd(0.2, 3, 0.0, administered, min_items=3) is True


# ============================================================
# 4. ZPD 覆盖检查测试
# ============================================================


class TestZPDCoverageCheck:
    """_zpd_coverage_check — ZPD 三区覆盖判定."""

    def test_three_zones_covered(self):
        """独立/ZPD/挫败三区均有题目 → True."""
        sel = CATSelector()
        administered = _bank([-3.0, 0.0, 3.0], a=1.0)  # theta=0: 0.953 / 0.5 / 0.047
        assert sel._zpd_coverage_check(0.0, administered) is True

    def test_missing_frustration_zone(self):
        """缺挫败区 → False."""
        sel = CATSelector()
        administered = _bank([-3.0, 0.0], a=1.0)  # independent + zpd, 无 frustration
        assert sel._zpd_coverage_check(0.0, administered) is False

    def test_missing_independent_zone(self):
        """缺独立区 → False."""
        sel = CATSelector()
        administered = _bank([0.0, 3.0], a=1.0)  # zpd + frustration, 无 independent
        assert sel._zpd_coverage_check(0.0, administered) is False

    def test_empty_administered_returns_false(self):
        """无已施测题目 → False."""
        sel = CATSelector()
        assert sel._zpd_coverage_check(0.0, []) is False

    def test_coverage_respects_custom_thresholds(self):
        """自定义阈值下覆盖判定相应变化."""
        zpd = ZPDCalculator(independent_p=0.6, frustration_p=0.4)
        sel = CATSelector(zpd_calculator=zpd)
        # theta=0, a=1.0: b=0 → P=0.5 (ZPD, 因 0.4<0.5<0.6)
        #                 b=-1 → P=0.731 (>0.6 → independent)
        #                 b=1 → P=0.269 (<0.4 → frustration)
        administered = _bank([-1.0, 0.0, 1.0], a=1.0)
        assert sel._zpd_coverage_check(0.0, administered) is True


# ============================================================
# 5. 模块导出与集成测试
# ============================================================


class TestZPDCATIntegration:
    """ZPD-CAT 集成 — 端到端选题流程."""

    def test_zpd_aware_strategy_end_to_end(self):
        """zpd_aware 端到端: 多轮选题均在 ZPD 区 (排除挫败)."""
        est = IRTEstimator()
        sel = CATSelector(selection_strategy="zpd_aware", zpd_calculator=ZPDCalculator())
        items = _bank([-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0], a=1.0)
        administered: set[str] = set()
        theta = 0.0
        for _ in range(4):
            chosen = sel.select_next(theta=theta, available_items=items, administered_ids=administered)
            if chosen is None:
                break
            # 选中的题不应在挫败区
            assert _p(est, theta, chosen) > 0.3
            administered.add(chosen["item_id"])

    def test_zpd_calculator_injection_optional(self):
        """zpd_calculator 为可选参数, 默认 None 不影响构造."""
        sel = CATSelector()
        assert sel._zpd_calculator is None
        sel2 = CATSelector(zpd_calculator=ZPDCalculator())
        assert sel2._zpd_calculator is not None

    def test_scaffold_then_zpd_aware_consistency(self):
        """支架选题与 zpd_aware 均在 ZPD 区工作 (一致性)."""
        est = IRTEstimator()
        zpd = ZPDCalculator()
        items = _bank([-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0], a=1.0)
        sel_scaffold = CATSelector(zpd_calculator=zpd)
        chosen = sel_scaffold.select_with_scaffold(0.0, items, set(), scaffold_level=0.5)
        # 中心选题应在 ZPD 区
        assert 0.3 < _p(est, 0.0, chosen) < 0.9
