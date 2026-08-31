"""产品真实性边界：默认运行不消费占位证据或演示外部结果."""

from __future__ import annotations

from dy3_polaris.l3.kp_graph_seed import seed_kp_graph
from dy3_polaris.l3.store import KnowledgeStore
from dy3_polaris.l5.agent_workers import _apply_textbook_fallback
from dy3_polaris.l6.core.engine import L6CoreEngine
from dy3_polaris.l6.registry import (
    CONNECTOR_TOOL_NAMES,
    EXTERNAL_TOOL_NAMES,
    load_all_tools,
)


def test_placeholder_textbook_facts_are_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DY3_ENABLE_PLACEHOLDER_KNOWLEDGE", raising=False)
    real_chunks = [{"chunk_id": "real-1", "content": "真实导入切片"}]

    result = _apply_textbook_fallback("Dy3+ 黄蓝双发射", real_chunks)

    assert result == real_chunks
    assert not any(
        (item.get("metadata") or {}).get("source_type") == "textbook_fallback"
        for item in result
    )


def test_placeholder_facts_do_not_enter_default_knowledge_graph() -> None:
    store = KnowledgeStore()

    counts = seed_kp_graph(store)

    assert counts["facts"] == 0
    assert counts["fact_kp_edges"] == 0
    assert all(
        "textbook_fallback" not in set(entity.tags or [])
        for entity in store.entity_store.list_entities(limit=1000)
    )


def test_product_registry_fails_closed_for_unimplemented_external_tools() -> None:
    engine = L6CoreEngine()
    engine.initialize()
    load_all_tools(engine.tool_registry)

    for tool_name in (CONNECTOR_TOOL_NAMES[0], EXTERNAL_TOOL_NAMES[0]):
        entry = engine.tool_registry.get(tool_name)
        assert entry is not None
        assert entry.is_stub is True
        outcome = engine.call_tool(tool_name, {})
        assert "error" in outcome
        assert "result" not in outcome

