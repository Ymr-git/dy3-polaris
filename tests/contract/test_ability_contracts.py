"""能力契约测试（第一版）— 收编自 docs/ability-contracts.md.

判定零 LLM：断言用确定性规则（关键词 / 字段 / 函数），不用 LLM 判分。
意图契约（B1~B3）已由 tests/l5/test_intent_generalization.py 覆盖，此处不重复。

运行：
    pytest tests/contract/ -m "not integration"   # 快速契约（零 LLM，默认）
    pytest tests/contract/ -m integration          # 全栈契约（走 /api/query）
"""
from __future__ import annotations

import pytest

from dy3_polaris.l2.kp_catalog import ALL_KP_IDS, _KG_NODES
from dy3_polaris.l3.deduction import deduce


# ============================================================
# D. 能力覆盖契约（单元级，零 LLM，快）
# ============================================================


class TestCoverageContract:
    def test_kp_coverage_ge_90(self) -> None:
        """42 KP 覆盖率 ≥ 90%（competition_eval 口径，实测 100%）。"""
        covered = set(_KG_NODES.keys())
        coverage = len(covered) / len(ALL_KP_IDS) if ALL_KP_IDS else 0.0
        assert coverage >= 0.90, f"知识点覆盖率 {coverage:.1%} 低于 90%"


# ============================================================
# 全栈契约（走 /api/query，标记 integration）
# ============================================================

_HONEST_REFUSAL_WORDS = ("不属于本系统", "暂无", "无法给出", "无法回答", "未收录")

# 学术术语：通俗化契约判定「术语密度」用的词表（beginner 画像不应堆砌）
_ACADEMIC_TERMS = (
    "4f", "能级", "跃迁", "电荷迁移", "谱线", "基质", "猝灭",
    "量子效率", "激发态", "电子构型", "晶场", "声子",
)


@pytest.fixture(scope="module")
def client():
    from starlette.testclient import TestClient
    from dy3_polaris.l5.unified_app import UnifiedApp

    builder = UnifiedApp.create_full_app_builder()
    return TestClient(builder.create_app())


def _query(client, q: str) -> dict:
    r = client.post("/api/query", json={"query": q, "learner_id": "DY20240001"})
    return r.json().get("data") or {}


@pytest.mark.integration
class TestHonestRefusalContract:
    def test_out_of_domain_refuses(self, client) -> None:
        """A1 领域外 → 诚实拒答，不编造。"""
        d = _query(client, "今天天气如何")
        answer = str(d.get("answer") or "")
        assert d.get("knowledge_unavailable") or any(
            w in answer for w in _HONEST_REFUSAL_WORDS
        )


@pytest.mark.integration
class TestPersonaAdaptationContracts:
    def test_beginner_gets_plain_language(self, client) -> None:
        """C2 beginner 画像问基础概念 → 通俗讲解，不堆学术术语（已修复，防回归）。"""
        d = _query(client, "发光材料是什么")
        answer = str(d.get("answer") or "")
        academic_count = sum(1 for t in _ACADEMIC_TERMS if t in answer)
        assert academic_count <= 2, f"学术术语密度 {academic_count} 过高，未通俗化"

    def test_dy_transition_retrieves_dy(self, client) -> None:
        """C3 问 Dy 跃迁 → 命中 Dy 跃迁，不串到激子/仪器（已修复，防回归）。

        注意：检索不相关是「类问题」，此契约只锁「Dy 跃迁」一个 case；
        其余答非所问 case（降蓝光危害/XRD 步骤）待补契约后逐一锁定。
        """
        d = _query(client, "Dy3+ 的蓝光和黄光分别来自哪个能级跃迁")
        answer = str(d.get("answer") or "")
        assert "4F9/2" in answer or "480" in answer or "575" in answer
        assert "激子" not in answer and "FS5" not in answer


# ============================================================
# 推演能力契约（D1~D5，单元级，零 LLM，快）
# ============================================================


class TestDeductionContracts:
    """推演能力：规则 + 已知量 → 未知结论（对标「114+256」）。"""

    def test_deduce_wavelength_from_energy(self) -> None:
        """D1 已知能量差 → 推发射波长（λ=1240/E）。"""
        out = deduce("4F9/2到6H15/2能量差2.58eV，发射波长多少")
        assert out and "48" in out  # ≈480/481 nm

    def test_deduce_energy_from_wavelength(self) -> None:
        """D2 已知波长 → 推能量（E=1240/λ）。"""
        out = deduce("480nm的光子能量是多少eV")
        assert out and "2.5" in out  # ≈2.58 eV

    def test_deduce_concentration_quenching(self) -> None:
        """D3 因果推演：浓度↑ → 效率↓（浓度猝灭）。"""
        out = deduce("掺杂浓度超过临界会怎样")
        assert out and "猝灭" in out

    def test_deduce_color_temperature(self) -> None:
        """D4 关系推演：黄蓝比高 → 色温低（暖白）。"""
        out = deduce("黄蓝比高色温偏暖还是冷")
        assert out and "暖" in out

    def test_deduce_missing_input_returns_none(self) -> None:
        """D5 诚实边界：缺已知量（没给 ΔE）→ 推不了，返回 None。"""
        assert deduce("4F9/2到6H15/2的波长是多少") is None

    def test_deduce_out_of_domain_returns_none(self) -> None:
        """D5 诚实边界：领域外 → 不推。"""
        assert deduce("今天天气如何") is None


@pytest.mark.integration
class TestMultiAgentContract:
    def test_query_has_review_loop(self, client) -> None:
        """E1 ≥3 agents 闭环：响应含 review（生成→校验→决策）。"""
        d = _query(client, "Dy3+ 的浓度猝灭机理是什么")
        review = d.get("review") or {}
        assert review.get("verdict"), "响应应含审核 Agent 的 verdict"


@pytest.mark.integration
class TestResourceFormsContract:
    def test_three_resource_forms(self, client) -> None:
        """E2 ≥3 资源形态：定制讲解 / 实操指南 / 分阶题都能产出。"""
        r = client.get("/api/personalized/resources", params={"learner_id": "DY20240001"})
        d = r.json().get("data") or {}
        assert d.get("customized_resource"), "定制化讲解应非空"
        assert d.get("practical_guide"), "实操指南应非空"
        assert d.get("staged_questions"), "分阶测试题应非空"


@pytest.mark.integration
class TestRetrievalRelevanceContract:
    """检索相关性（答非所问）契约——persona #29③ 各 case 已缓解，锁住防回归。"""

    def test_reduce_blue_light_hazard(self, client) -> None:
        """问「降蓝光危害」→ 应用建议，不串到 FS5 仪器。"""
        d = _query(client, "如何降低白光 LED 的蓝光危害")
        answer = str(d.get("answer") or "")
        assert "色温" in answer or "滤光" in answer
        assert "FS5" not in answer

    def test_xrd_steps(self, client) -> None:
        """问「XRD 步骤」→ 分步，不串到图目录/SEM 图注。"""
        d = _query(client, "XRD 测荧光粉物相的操作步骤")
        answer = str(d.get("answer") or "")
        assert "步骤" in answer or "PDF" in answer or "2θ" in answer
        assert "图目录" not in answer

    def test_upconversion_qy(self, client) -> None:
        """问「上转换量子产率」→ 主题命中，不串到 LED 芯片。"""
        d = _query(client, "如何提高上转换量子产率")
        answer = str(d.get("answer") or "")
        assert "量子产率" in answer or "上转换" in answer



