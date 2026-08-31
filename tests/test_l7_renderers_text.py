"""L7 渲染器 T2 — TextRenderer 与 FormulaRenderer 单元测试.

测试覆盖:
1. TextRenderer 扩展语法: 知识卡片 / callout 折叠 / 术语高亮 / 行内公式 / 块级公式
2. BKT 三档个性化: beginner (平均 P(L)<0.4) / advanced (P(L)>0.7) / 显式 learner_mode
3. Socratic 对话气泡渲染
4. 渲染描述符契约: html/config/assets/metadata
5. FormulaRenderer: 内联/块级公式容器、编号、KaTeX 资源
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.exceptions import ArtifactValidationError
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactType,
    LearnerMode,
    RenderContext,
    RenderDescriptor,
)
from dy3_polaris.l7.renderers.formula_renderer import FormulaRenderer
from dy3_polaris.l7.renderers.text_renderer import TextRenderer


def _text_artifact(content: str, bkt: dict | None = None, payload_extra: dict | None = None) -> Artifact:
    payload = {"content": content}
    if payload_extra:
        payload.update(payload_extra)
    learner_context = {"bkt_state": bkt} if bkt else {}
    return Artifact(
        type=ArtifactType.TEXT,
        mime="text/vnd.dy3+markdown",
        payload=payload,
        learner_context=learner_context,
    )


MARKDOWN_WITH_EXTENSIONS = (
    "# 标题\n"
    ':::kp[A-01]{title="跃迁测试"}:::\n'
    "卡片内容\n"
    ":::\n\n"
    "==晶体场分裂==\n\n"
    "$E=mc^2$\n\n"
    "$$\\int_0^1 x dx$$\n\n"
    "> [!note] 注意\n"
    "折叠内容"
)


class TestTextRendererCore:
    """TextRenderer 基础契约."""

    def test_mime_types(self):
        renderer = TextRenderer()
        assert "text/vnd.dy3+markdown" in renderer.supported_mime_types()
        assert "text/markdown" in renderer.supported_mime_types()

    def test_render_returns_descriptor(self):
        d = TextRenderer().render(
            _text_artifact("# 标题"), RenderContext()
        )
        assert isinstance(d, RenderDescriptor)
        assert d.artifact_id
        assert d.html
        assert d.config["engine"] == "python-markdown"
        assert d.config["katex"]["enabled"] is True
        assert "renderer" in d.metadata

    def test_empty_payload_raises(self):
        with pytest.raises(ArtifactValidationError):
            TextRenderer().render(
                Artifact(type=ArtifactType.TEXT, mime="text/vnd.dy3+markdown", payload={}),
                RenderContext(),
            )

    def test_assets_include_katex(self):
        d = TextRenderer().render(_text_artifact("公式 $x$"), RenderContext())
        assert any("katex" in a for a in d.assets)


class TestTextRendererExtensions:
    """扩展语法: 知识卡片 / callout / 术语高亮 / 公式."""

    def _render_html(self, content: str) -> str:
        return TextRenderer().render(_text_artifact(content), RenderContext()).html

    def test_kp_card(self):
        html = self._render_html(
            ':::kp[A-01]{title="跃迁"}:::\n卡片内容\n:::'
        )
        assert "kp-card" in html
        assert 'data-kp="A-01"' in html
        assert "跃迁" in html
        assert "卡片内容" in html

    def test_kp_card_no_blank_line(self):
        html = self._render_html(
            ':::kp[A-01]:::\n内容\n:::\n\n后续段落'
        )
        assert "kp-card" in html
        assert "内容" in html
        assert "后续段落" in html

    def test_callout(self):
        html = self._render_html("> [!note] 注意\n折叠内容")
        assert "callout" in html
        assert "callout-note" in html
        assert "折叠内容" in html

    def test_callout_warning(self):
        html = self._render_html("> [!warning] 危险\n高温操作")
        assert "callout-warning" in html

    def test_term_highlight(self):
        html = self._render_html("术语 ==晶体场分裂== 高亮")
        assert "term-highlight" in html
        assert 'data-term="晶体场分裂"' in html

    def test_inline_math(self):
        html = self._render_html("行内公式 $E=mc^2$")
        assert "math-inline" in html
        assert 'data-latex="E=mc^2"' in html

    def test_block_math(self):
        html = self._render_html("$$\n\\int_0^1 x dx\n$$")
        assert "math-display" in html

    def test_block_math_single_line(self):
        html = self._render_html("$$\\tau = 1/A_{rad}$$")
        assert "math-display" in html

    def test_kp_refs_metadata(self):
        d = TextRenderer().render(
            _text_artifact("参考 A-01 与 B-02"), RenderContext()
        )
        assert "A-01" in d.metadata["kp_refs"]
        assert "B-02" in d.metadata["kp_refs"]


class TestTextRendererBKTAdaptation:
    """BKT 三档个性化 (设计文档 §2.2.3)."""

    def test_beginner_from_avg_p_l(self):
        bkt = {
            "A-01": {"p_l": 0.2, "p_k_l": 0.3, "p_g": 0.2, "p_s": 0.1},
            "A-02": {"p_l": 0.35, "p_k_l": 0.4, "p_g": 0.2, "p_s": 0.1},
        }
        d = TextRenderer().render(
            _text_artifact("内容", bkt=bkt), RenderContext()
        )
        assert d.config["learner_adaptation"]["mode"] == "beginner"
        assert "term-tooltip" in d.config["learner_adaptation"]["applied_rules"]
        assert "data-adapt=\"beginner-tooltip\"" in d.html or "term-tooltip" in d.config["learner_adaptation"]["applied_rules"]

    def test_advanced_from_avg_p_l(self):
        bkt = {
            "A-01": {"p_l": 0.9, "p_k_l": 0.85, "p_g": 0.1, "p_s": 0.05},
            "A-02": {"p_l": 0.8, "p_k_l": 0.8, "p_g": 0.1, "p_s": 0.05},
        }
        d = TextRenderer().render(
            _text_artifact("内容", bkt=bkt), RenderContext()
        )
        assert d.config["learner_adaptation"]["mode"] == "advanced"
        assert "concise-mode" in d.config["learner_adaptation"]["applied_rules"]

    def test_intermediate_without_bkt(self):
        d = TextRenderer().render(_text_artifact("内容"), RenderContext())
        assert d.config["learner_adaptation"]["mode"] == "intermediate"

    def test_explicit_learner_mode_used_when_no_bkt(self):
        d = TextRenderer().render(
            _text_artifact("内容"),
            RenderContext(learner_mode=LearnerMode.ADVANCED),
        )
        assert d.config["learner_adaptation"]["mode"] == "advanced"
        assert d.config["learner_adaptation"]["source"] == "explicit"


class TestTextRendererSocratic:
    """Socratic 对话气泡 (§2.2.4)."""

    def test_dialogue_rendered(self):
        dialogue = [
            {"role": "teacher", "content": "什么是晶体场分裂？"},
            {"role": "system", "content": "配体场对 d 轨道能级的劈裂。"},
        ]
        art = _text_artifact("教学内容", payload_extra={"dialogue": dialogue})
        d = TextRenderer().render(art, RenderContext())
        assert "socratic-dialogue" in d.html
        assert "socratic-teacher" in d.html
        assert "socratic-system" in d.html
        assert d.config["socratic"]["enabled"] is True

    def test_no_dialogue(self):
        d = TextRenderer().render(_text_artifact("内容"), RenderContext())
        assert d.config["socratic"]["enabled"] is False


# ============================================================
# FormulaRenderer
# ============================================================

class TestFormulaRenderer:
    """FormulaRenderer 单元测试."""

    def _formula_artifact(self, latex: str, **extra):
        payload = {"latex": latex}
        payload.update(extra)
        return Artifact(
            type=ArtifactType.FORMULA,
            mime="application/vnd.dy3.formula+json",
            payload=payload,
        )

    def test_mime_types(self):
        renderer = FormulaRenderer()
        assert "application/vnd.dy3.formula+json" in renderer.supported_mime_types()
        assert "application/vnd.dy3.formula+tex" in renderer.supported_mime_types()

    def test_inline_formula(self):
        d = FormulaRenderer().render(
            self._formula_artifact("量子效率 $\\eta = I_{em}/I_{abs}$"),
            RenderContext(),
        )
        assert "math-inline" in d.html

    def test_block_formula(self):
        d = FormulaRenderer().render(
            self._formula_artifact("$$\\tau = \\frac{1}{A_{rad} + A_{nr}}$$"),
            RenderContext(),
        )
        assert "math-display" in d.html

    def test_numbering(self):
        d = FormulaRenderer().render(
            self._formula_artifact("$$\\tau = 1$$", display_numbers=True),
            RenderContext(),
        )
        assert "formula-number" in d.html

    def test_no_numbering(self):
        d = FormulaRenderer().render(
            self._formula_artifact("$$\\tau = 1$$", display_numbers=False),
            RenderContext(),
        )
        assert "formula-number" not in d.html

    def test_katex_assets(self):
        d = FormulaRenderer().render(
            self._formula_artifact("$x$"), RenderContext()
        )
        assert any("katex" in a for a in d.assets)
        assert d.config["katex"]["interactions"] == ["click-zoom", "copy-source"]

    def test_metadata_counts(self):
        d = FormulaRenderer().render(
            self._formula_artifact("$a$ 与 $b$ 以及 $$c$$"),
            RenderContext(),
        )
        assert d.metadata["inline_count"] >= 2
        assert d.metadata["block_count"] >= 1

    def test_empty_payload_raises(self):
        with pytest.raises(ArtifactValidationError):
            FormulaRenderer().render(
                Artifact(type=ArtifactType.FORMULA, mime="application/vnd.dy3.formula+json", payload={}),
                RenderContext(),
            )
