"""L7 渲染器 T2 — TableRenderer、MoleculeRenderer、ProvenanceRenderer 单元测试.

测试覆盖:
1. TableRenderer: 表格结构、条件格式、42 KP×4 参数微型条形图、瓶颈行着色
2. MoleculeRenderer: 结构解析 (molfile/smiles/structure)、能级动画、三级降级、宿主晶格
3. ProvenanceRenderer: 时间线 (脱敏)、决策树 (深度过滤)、分支合并图
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.exceptions import ArtifactValidationError
from dy3_polaris.l7.models import Artifact, ArtifactType, RenderContext, RenderDescriptor
from dy3_polaris.l7.renderers.molecule_renderer import MoleculeRenderer
from dy3_polaris.l7.renderers.provenance_renderer import ProvenanceRenderer
from dy3_polaris.l7.renderers.table_renderer import TableRenderer


# ============================================================
# TableRenderer
# ============================================================

class TestTableRendererCore:
    """TableRenderer 基础契约."""

    def _table_artifact(self, payload: dict, bkt: dict | None = None) -> Artifact:
        return Artifact(
            type=ArtifactType.TABLE,
            mime="application/vnd.dy3.table+json",
            payload=payload,
            learner_context={"bkt_state": bkt} if bkt else {},
        )

    def test_mime_types(self):
        assert "application/vnd.dy3.table+json" in TableRenderer().supported_mime_types()

    def test_render_basic_table(self):
        d = TableRenderer().render(
            self._table_artifact(
                {"title": "数据", "headers": ["材料", "QE"], "rows": [["NaGdF4", 85]]}
            ),
            RenderContext(),
        )
        assert isinstance(d, RenderDescriptor)
        assert "l7-data-table" in d.html
        assert "NaGdF4" in d.html
        assert d.config["sortable"] is True

    def test_missing_headers_raises(self):
        with pytest.raises(ArtifactValidationError):
            TableRenderer().render(
                self._table_artifact({"rows": [["a"]]}), RenderContext()
            )

    def test_condition_formatting(self):
        d = TableRenderer().render(
            self._table_artifact(
                {
                    "title": "QE",
                    "headers": ["材料", "QE"],
                    "rows": [["A", 90], ["B", 20]],
                    "format_rules": [
                        {"column": "QE", "op": "gte", "threshold": 80, "color": "#16a34a"},
                        {"column": "QE", "op": "lt", "threshold": 30, "color": "#ef4444"},
                    ],
                }
            ),
            RenderContext(),
        )
        assert "color:#16a34a" in d.html
        assert "color:#ef4444" in d.html

    def test_csv_export_config(self):
        d = TableRenderer().render(
            self._table_artifact(
                {"title": "t", "headers": ["a"], "rows": [["1"]]}
            ),
            RenderContext(),
        )
        assert d.config["csv_export"] is True


class TestTableMiniBars(TestTableRendererCore):
    """42 KP × 4 参数微型条形图 (设计文档 §2.6.2)."""

    def _bkt_table_artifact(self):
        return self._table_artifact(
            {
                "title": "学情总览",
                "headers": ["KP", "名称"],
                "rows": [["A-01", "电子构型"], ["A-02", "4f壳层"], ["D-08", "色度学"]],
                "bkt_table": True,
            },
            bkt={
                "A-01": {"p_l": 0.9, "p_k_l": 0.8, "p_g": 0.2, "p_s": 0.1},
                "A-02": {"p_l": 0.75, "p_k_l": 0.25, "p_g": 0.2, "p_s": 0.1},
            },
        )

    def test_mini_bars_enabled(self):
        d = TableRenderer().render(self._bkt_table_artifact(), RenderContext())
        assert d.config["mini_bars"] is True
        assert d.metadata["mini_bars"] is True

    def test_mini_bar_headers(self):
        d = TableRenderer().render(self._bkt_table_artifact(), RenderContext())
        assert "P(L)" in d.html
        assert "P(K|L)" in d.html
        assert "mini-bar-cell" in d.html

    def test_bottleneck_row_pulse(self):
        d = TableRenderer().render(self._bkt_table_artifact(), RenderContext())
        # A-02: P(L)=0.75>0.7 且 P(K|L)=0.25<0.3 → 瓶颈红色脉冲
        assert "bkt-bottleneck-pulse" in d.html

    def test_no_mini_bars_when_disabled(self):
        d = TableRenderer().render(
            self._table_artifact(
                {"title": "t", "headers": ["a"], "rows": [["1"]]}
            ),
            RenderContext(),
        )
        assert d.config["mini_bars"] is False


# ============================================================
# MoleculeRenderer
# ============================================================

class TestMoleculeRenderer:
    """MoleculeRenderer 单元测试."""

    def _mol_artifact(self, payload: dict) -> Artifact:
        return Artifact(
            type=ArtifactType.MOLECULE,
            mime="application/vnd.dy3.molecule+json",
            payload=payload,
        )

    def test_mime_types(self):
        renderer = MoleculeRenderer()
        assert "application/vnd.dy3.molecule+json" in renderer.supported_mime_types()
        assert "chemical/x-mdl-molfile" in renderer.supported_mime_types()

    def test_smiles_structure(self):
        d = MoleculeRenderer().render(
            self._mol_artifact({"smiles": "CCO", "host": "NaGdF4"}),
            RenderContext(),
        )
        assert d.config["structure"]["source"] == "smiles"
        assert d.config["structure"]["content"] == "CCO"
        assert d.config["host_info"]["formula"] == "NaGdF4"
        assert "3Dmol" in d.assets[0]

    def test_molfile_structure(self):
        molfile = "test\n  test\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n"
        d = MoleculeRenderer().render(
            self._mol_artifact({"molfile": molfile}), RenderContext()
        )
        assert d.config["structure"]["source"] == "molfile"
        assert d.config["structure"]["format"] == "mol"

    def test_structure_dict_cif(self):
        d = MoleculeRenderer().render(
            self._mol_artifact({"structure": {"format": "cif", "content": "data_X\n_cell_length_a 5"}}),
            RenderContext(),
        )
        assert d.config["structure"]["source"] == "structure"
        assert d.config["structure"]["format"] == "cif"

    def test_animation_normalized(self):
        d = MoleculeRenderer().render(
            self._mol_artifact(
                {
                    "smiles": "CCO",
                    "animation": {
                        "ground": "^6H_15/2",
                        "excited_5d": "4f^5 5d",
                        "excited_4f": "^4F_9/2",
                        "emission_nm": 575,
                    },
                }
            ),
            RenderContext(),
        )
        anim = d.config["animation"]
        assert anim["emission_nm"] == 575.0
        assert anim["sync_jablonski"] is True

    def test_three_level_fallback(self):
        d = MoleculeRenderer().render(
            self._mol_artifact({"smiles": "CCO"}), RenderContext()
        )
        levels = d.config["levels"]
        assert levels["level0"]["webgl"] == "webgl2"
        assert levels["level1"]["limits"]["face_count"] == 5000
        assert levels["level1"]["features"] == ["rotate", "zoom"]
        assert levels["level2"]["webgl"] is None

    def test_style_switch(self):
        d = MoleculeRenderer().render(
            self._mol_artifact({"smiles": "CCO", "style": "spacefill"}),
            RenderContext(),
        )
        assert "sphere" in d.config["style"]
        assert d.config["interactions"]["style_switch"] == ["stick", "ball", "spacefill"]

    def test_missing_structure_raises(self):
        with pytest.raises(ArtifactValidationError):
            MoleculeRenderer().render(
                Artifact(type=ArtifactType.MOLECULE, mime="application/vnd.dy3.molecule+json", payload={}),
                RenderContext(),
            )


# ============================================================
# ProvenanceRenderer
# ============================================================

class TestProvenanceRenderer:
    """ProvenanceRenderer 单元测试."""

    def _prov_artifact(self, payload: dict, learner_context: dict | None = None) -> Artifact:
        return Artifact(
            type=ArtifactType.PROVENANCE,
            mime="application/vnd.dy3.provenance+json",
            payload=payload,
            learner_context=learner_context or {},
        )

    def test_mime_types(self):
        assert (
            "application/vnd.dy3.provenance+json"
            in ProvenanceRenderer().supported_mime_types()
        )

    def test_timeline_mode(self):
        d = ProvenanceRenderer().render(
            self._prov_artifact(
                {
                    "mode": "timeline",
                    "events": [
                        {"type": "decision", "timestamp": "10:00", "summary": "选择范式", "kp_id": "A-05"},
                        {"type": "test", "timestamp": "10:05", "summary": "答题"},
                    ],
                }
            ),
            RenderContext(),
        )
        assert d.config["mode"] == "timeline"
        assert "prov-timeline" in d.html
        assert "筛选" in d.html

    def test_privacy_masked_by_default(self):
        d = ProvenanceRenderer().render(
            self._prov_artifact(
                {
                    "mode": "timeline",
                    "events": [
                        {"type": "teaching", "timestamp": "10:00", "summary": "讲解", "raw": "用户原始输入"},
                    ],
                }
            ),
            RenderContext(),
        )
        assert d.config["privacy"]["masked"] is True
        assert "已脱敏" in d.html

    def test_full_access_reveals_raw(self):
        d = ProvenanceRenderer().render(
            self._prov_artifact(
                {
                    "mode": "timeline",
                    "events": [{"type": "teaching", "timestamp": "10:00", "summary": "讲解", "raw": "原文内容"}],
                },
                learner_context={"full_access": True},
            ),
            RenderContext(),
        )
        assert d.config["privacy"]["masked"] is False
        assert "原文内容" in d.html

    def test_decision_mode_summary_depth(self):
        d = ProvenanceRenderer().render(
            self._prov_artifact(
                {
                    "mode": "decision",
                    "depth": "summary",
                    "steps": [
                        {"step": "complexity", "title": "复杂度评估"},
                        {"step": "decision", "title": "范式选择"},
                        {"step": "adjudication", "title": "裁决结果"},
                    ],
                }
            ),
            RenderContext(),
        )
        assert d.config["mode"] == "decision"
        assert d.config["depth"] == "summary"
        # summary 深度只保留关键节点
        assert "范式选择" in d.html
        assert "复杂度评估" not in d.html

    def test_decision_mode_standard_depth(self):
        d = ProvenanceRenderer().render(
            self._prov_artifact(
                {
                    "mode": "decision",
                    "steps": [
                        {"step": "complexity", "title": "复杂度评估"},
                        {"step": "decision", "title": "范式选择"},
                    ],
                }
            ),
            RenderContext(),
        )
        assert "复杂度评估" in d.html
        assert "范式选择" in d.html

    def test_branch_merge_mode(self):
        d = ProvenanceRenderer().render(
            self._prov_artifact(
                {
                    "mode": "branch_merge",
                    "mainline": [{"id": "v1", "title": "初始版本"}],
                    "branches": [{"title": "分支A", "reason": "方案探索"}],
                    "merges": [{"title": "合并", "result": "采纳分支A"}],
                }
            ),
            RenderContext(),
        )
        assert d.config["mode"] == "branch_merge"
        assert "prov-branch" in d.html
        assert "方案探索" in d.html
        assert "采纳分支A" in d.html

    def test_invalid_mode_fallback_timeline(self):
        d = ProvenanceRenderer().render(
            self._prov_artifact(
                {"mode": "bogus", "events": [{"type": "test", "summary": "x"}]}
            ),
            RenderContext(),
        )
        assert d.config["mode"] == "timeline"
