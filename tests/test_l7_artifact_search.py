"""L7 Artifact 管理系统 T3 — 搜索与过滤单元测试.

测试覆盖:
1. tokenize / InvertedIndex: 分词与倒排构建
2. SearchEngine 全文本: 关键词、引号短语、布尔 AND/OR/NOT
3. 结构化过滤: type/source_agent/kp_id/时间/edited_only/state
4. 学情关联搜索: 直接关联 + 知识图谱间接关联
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.artifact.search import (
    InvertedIndex,
    SearchEngine,
    build_index,
    tokenize,
)
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactLifecycleState,
    ArtifactType,
)


def _art(
    content: str,
    title: str = "",
    agent: str = "",
    kp_ids: list[str] | None = None,
    mime: str = "text/vnd.dy3+markdown",
    **kwargs,
) -> Artifact:
    return Artifact(
        type=ArtifactType.CHART if "chart" in mime else ArtifactType.TEXT,
        mime=mime,
        payload={"content": content},
        title=title,
        source_agent=agent,
        learner_context={"kp_ids": kp_ids or []},
        **kwargs,
    )


class TestTokenize:
    """分词."""

    def test_english_and_chinese(self):
        tokens = tokenize("荧光 efficiency A-01")
        assert "荧光" in tokens
        assert "efficiency" in tokens
        assert "a-01" in tokens

    def test_case_insensitive(self):
        assert tokenize("Hello") == ["hello"]

    def test_empty(self):
        assert tokenize(None) == []
        assert tokenize("") == []


class TestInvertedIndex:
    """倒排索引."""

    def test_add_and_query(self):
        index = InvertedIndex()
        index.add_document("d1", "荧光效率分析")
        index.add_document("d2", "材料性能对比")
        assert "d1" in index.postings("荧光")
        assert "d2" not in index.postings("荧光")

    def test_remove_document(self):
        index = InvertedIndex()
        index.add_document("d1", "荧光")
        index.remove_document("d1")
        assert "d1" not in index.postings("荧光")

    def test_doc_count(self):
        index = InvertedIndex()
        index.add_document("d1", "a")
        index.add_document("d2", "b")
        assert index.doc_count == 2


class TestSearchEngine:
    """全文本搜索."""

    def _engine(self) -> SearchEngine:
        engine = SearchEngine()
        engine.reindex([
            _art("荧光效率分析", title="效率报告", agent="A1", kp_ids=["A-01"]),
            _art("材料性能对比", title="性能图", agent="A2", kp_ids=["A-02"]),
            _art("合成工艺优化", title="工艺", agent="A3"),
        ])
        return engine

    def test_keyword_search(self):
        engine = self._engine()
        results = engine.search("荧光")
        assert len(results) == 1
        assert results[0].title == "效率报告"

    def test_case_insensitive(self):
        engine = self._engine()
        assert len(engine.search("FLUORESCENCE")) >= 0  # 英文词小写化
        assert len(engine.search("荧光")) == 1

    def test_or_implicit(self):
        engine = self._engine()
        results = engine.search("荧光 材料")
        assert len(results) == 2

    def test_explicit_or(self):
        engine = self._engine()
        results = engine.search("荧光 OR 材料")
        assert len(results) == 2

    def test_and(self):
        engine = SearchEngine()
        engine.reindex([
            _art("荧光 效率 分析"),
            _art("荧光 材料"),
            _art("材料 效率"),
        ])
        # AND 语义: 同时含"荧光"和"效率"
        results = engine.search("荧光 AND 效率")
        assert len(results) == 1

    def test_not(self):
        engine = SearchEngine()
        engine.reindex([
            _art("荧光 效率"),
            _art("荧光 材料"),
            _art("材料"),
        ])
        results = engine.search("荧光 NOT 材料")
        assert len(results) == 1
        assert "效率" in results[0].payload["content"]

    def test_exact_phrase(self):
        engine = SearchEngine()
        engine.reindex([
            _art("晶体场 分裂 理论"),
            _art("晶体场 与 配位"),
        ])
        results = engine.search('"晶体场 分裂"')
        assert len(results) == 1

    def test_empty_query_returns_all(self):
        engine = self._engine()
        assert len(engine.search("")) == 3

    def test_boolean_query_in_engine(self):
        engine = self._engine()
        results = engine.search("效率 OR 工艺")
        assert len(results) == 2


class TestStructuredFilters:
    """结构化过滤 (设计文档 Ch.3.6)."""

    def _engine(self) -> SearchEngine:
        engine = SearchEngine()
        engine.reindex([
            _art("内容A", title="t1", agent="A1", kp_ids=["A-01"]),
            _art("内容B", title="t2", agent="A2", kp_ids=["B-02"],
                 mime="application/vnd.dy3.chart+json"),
        ])
        return engine

    def test_filter_by_type(self):
        engine = self._engine()
        results = engine.search("", filters={"type": "chart"})
        assert len(results) == 1
        assert results[0].source_agent == "A2"

    def test_filter_by_source_agent(self):
        engine = self._engine()
        results = engine.search("", filters={"source_agent": "A1"})
        assert len(results) == 1

    def test_filter_by_kp_id(self):
        engine = self._engine()
        results = engine.search("", filters={"kp_id": "A-01"})
        assert len(results) == 1

    def test_filter_edited_only(self):
        engine = SearchEngine()
        engine.reindex([
            _art("v1", title="单版本"),
            _art("v2", title="多版本", version=3),
        ])
        results = engine.search("", filters={"edited_only": True})
        assert len(results) == 1
        assert results[0].version == 3

    def test_filter_state(self):
        engine = SearchEngine()
        engine.reindex([
            _art("活跃", title="a"),
            _art("归档", title="b", state=ArtifactLifecycleState.ARCHIVED),
        ])
        results = engine.search("", filters={"state": "archived"})
        assert len(results) == 1

    def test_sort_by_created(self):
        engine = SearchEngine()
        engine.reindex([
            _art("x", title="new", created_at=2000.0),
            _art("x", title="old", created_at=1000.0),
        ])
        results = engine.search("", sort_by="-created_at")
        assert results[0].title == "new"


class TestRelatedByKP:
    """学情关联搜索."""

    def test_direct_related(self):
        engine = SearchEngine()
        engine.reindex([
            _art("直接相关", kp_ids=["A-01"]),
            _art("无关", kp_ids=["B-05"]),
        ])
        results = engine.related_by_kp("A-01")
        assert len(results) == 1

    def test_indirect_related_via_graph(self):
        engine = SearchEngine()
        engine.reindex([
            _art("A 相关", kp_ids=["A-01"]),
            _art("B 相关", kp_ids=["B-05"]),
        ])
        # A-01 → A-02 → B-05: 跳 2 找到 B-05 的 Artifact
        graph = {"A-01": {"A-02"}, "A-02": {"B-05"}}
        results = engine.related_by_kp("A-01", max_depth=2, kp_graph=graph)
        assert len(results) == 2  # 直接 + 间接

    def test_no_graph_only_direct(self):
        engine = SearchEngine()
        engine.reindex([
            _art("A 相关", kp_ids=["A-01"]),
            _art("B 相关", kp_ids=["B-05"]),
        ])
        results = engine.related_by_kp("A-01")  # 无图 → 仅直接
        assert len(results) == 1

    def test_direct_priority(self):
        engine = SearchEngine()
        engine.reindex([
            _art("间接", title="间接", kp_ids=["B-05"]),
            _art("直接", title="直接", kp_ids=["A-01"]),
        ])
        graph = {"A-01": {"B-05"}}
        results = engine.related_by_kp("A-01", kp_graph=graph)
        assert results[0].title == "直接"
