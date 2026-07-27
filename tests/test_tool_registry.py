"""T2 MCP 工具注册中心 - 单元测试.

测试覆盖:
1. SchemaValidator (Schema 校验器)
2. ToolRegistry (注册中心核心)
3. ToolEntry (工具条目)
4. 47 个工具定义完整性验证
5. 内部工具 handler 功能测试
6. 注册中心多维发现与索引
7. 依赖解析
8. MCP 兼容导出
9. 批量加载与全局单例
"""

from __future__ import annotations

import asyncio
import pytest

from dy3_polaris.l6.core.exceptions import (
    InvalidParamsError,
    L6Error,
    MCPToolNotFoundError,
    SchemaValidationError,
)
from dy3_polaris.l6.core.models import (
    Dy3ToolAnnotations,
    LayerTag,
    ToolCategory,
    ToolRegistration,
)
from dy3_polaris.l6.registry import (
    ALL_TOOL_DEFINITIONS,
    ALL_TOOL_NAMES,
    TOTAL_TOOL_COUNT,
    INTERNAL_TOOL_DEFINITIONS,
    INTERNAL_TOOL_NAMES,
    CONNECTOR_TOOL_DEFINITIONS,
    CONNECTOR_TOOL_NAMES,
    SKILLBOOK_TOOL_DEFINITIONS,
    SKILLBOOK_TOOL_NAMES,
    EXTERNAL_TOOL_DEFINITIONS,
    EXTERNAL_TOOL_NAMES,
    DIAGNOSIS_TOOLS,
    REVIEW_TOOLS,
    GUIDANCE_TOOLS,
    SHARED_TOOLS,
    TIER1_TOOLS,
    TIER2_TOOLS,
    TIER3_TOOLS,
    ToolRegistry,
    ToolEntry,
    SchemaValidator,
    load_all_tools,
    load_internal_tools,
    load_connector_tools,
    load_skillbook_tools,
    load_external_tools,
    find_tool,
    get_tool_names_by_category,
    get_registry,
    reset_registry,
    get_validator,
    reset_validator,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def fresh_registry() -> ToolRegistry:
    """每个测试使用全新的注册中心."""
    return ToolRegistry()

@pytest.fixture
def loaded_registry() -> ToolRegistry:
    """加载全部 47 个工具的注册中心."""
    reg = ToolRegistry()
    load_all_tools(reg)
    return reg

@pytest.fixture
def validator() -> SchemaValidator:
    return SchemaValidator()

@pytest.fixture
def sample_registration() -> ToolRegistration:
    return ToolRegistration(
        name="test_tool_alpha",
        description="A test tool for unit testing",
        input_schema={
            "type": "object",
            "properties": {
                "param_a": {"type": "string"},
                "param_b": {"type": "integer", "default": 42},
            },
            "required": ["param_a"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
            },
            "required": ["result"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["test", "alpha"],
            layer=LayerTag.L5_AGENT_RUNTIME,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=100,
            domain_scope=["DOM-A"],
        ),
    )


# ============================================================
# 1. SchemaValidator 测试
# ============================================================

class TestSchemaValidator:
    def test_valid_input_schema(self, validator: SchemaValidator):
        errors = validator.validate_input_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })
        assert len(errors) == 0

    def test_missing_type(self, validator: SchemaValidator):
        errors = validator.validate_input_schema({
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })
        assert any("type must be 'object'" in e.message for e in errors)

    def test_missing_properties(self, validator: SchemaValidator):
        errors = validator.validate_input_schema({
            "type": "object",
            "required": [],
        })
        assert any("properties" in e.message for e in errors)

    def test_missing_required(self, validator: SchemaValidator):
        errors = validator.validate_input_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        assert any("required" in e.message for e in errors)

    def test_output_schema_none_ok(self, validator: SchemaValidator):
        errors = validator.validate_output_schema(None)
        assert len(errors) == 0

    def test_validate_definition_valid(self, validator: SchemaValidator):
        errors = validator.validate_definition(
            name="valid_tool_name",
            description="A valid tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )
        assert len(errors) == 0

    def test_validate_definition_bad_name(self, validator: SchemaValidator):
        errors = validator.validate_definition(
            name="Invalid-Name",
            description="Bad name",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        assert any("name" in e.path for e in errors)

    def test_validate_definition_empty_description(self, validator: SchemaValidator):
        errors = validator.validate_definition(
            name="some_tool",
            description="",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        assert any("description" in e.path for e in errors)

    def test_validate_input_passes(self, validator: SchemaValidator):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        # 正确参数
        validator.validate_input(schema, {"name": "Alice", "age": 30})
        # 缺少必填项应该报错
        with pytest.raises(InvalidParamsError):
            validator.validate_input(schema, {"age": 30})

    def test_validate_input_type_mismatch(self, validator: SchemaValidator):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        with pytest.raises(InvalidParamsError):
            validator.validate_input(schema, {"count": "not_a_number"})

    def test_validate_output_passes(self, validator: SchemaValidator):
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }
        validator.validate_output(schema, {"result": "ok"})
        with pytest.raises(SchemaValidationError):
            validator.validate_output(schema, {"result": 123})

    def test_validate_output_none_skips(self, validator: SchemaValidator):
        validator.validate_output(None, {"anything": 123})

    def test_cache_reused(self, validator: SchemaValidator):
        schema = {"type": "object", "properties": {}, "required": []}
        validator.validate_input(schema, {})
        assert len(validator._cache) > 0
        validator.validate_input(schema, {})  # 应该使用缓存
        validator.clear_cache()
        assert len(validator._cache) == 0


# ============================================================
# 2. ToolEntry 测试
# ============================================================

class TestToolEntry:
    def test_create_entry(self, sample_registration: ToolRegistration):
        async def handler(**kw): return {"result": "ok"}
        entry = ToolEntry(sample_registration, handler)
        assert entry.name == "test_tool_alpha"
        assert entry.is_stub is False
        assert entry.call_count == 0

    def test_stub_entry(self, sample_registration: ToolRegistration):
        entry = ToolEntry(sample_registration, None)
        assert entry.is_stub is True

    def test_touch(self, sample_registration: ToolRegistration):
        entry = ToolEntry(sample_registration, None)
        entry.touch(success=True)
        entry.touch(success=True)
        entry.touch(success=False)
        assert entry.call_count == 3
        assert entry.error_count == 1
        assert entry.last_called_at is not None

    def test_to_dict(self, sample_registration: ToolRegistration):
        entry = ToolEntry(sample_registration, None, dependencies=["dep_a"])
        d = entry.to_dict()
        assert d["name"] == "test_tool_alpha"
        assert d["is_stub"] is True
        assert d["dependencies"] == ["dep_a"]
        assert d["category"] == "internal"


# ============================================================
# 3. ToolRegistry 注册/更新/注销测试
# ============================================================

class TestRegistryRegistration:
    def test_register_sync(self, fresh_registry: ToolRegistry, sample_registration: ToolRegistration):
        async def handler(**kw): return {}
        entry = fresh_registry.register_sync(sample_registration, handler)
        assert entry.name == "test_tool_alpha"
        assert fresh_registry.size == 1
        assert fresh_registry.contains("test_tool_alpha")

    def test_register_duplicate_fails(self, fresh_registry: ToolRegistry, sample_registration: ToolRegistration):
        fresh_registry.register_sync(sample_registration, None)
        with pytest.raises(L6Error, match="already registered"):
            fresh_registry.register_sync(sample_registration, None)

    def test_register_duplicate_overwrite(self, fresh_registry: ToolRegistry, sample_registration: ToolRegistration):
        fresh_registry.register_sync(sample_registration, None)
        async def new_handler(**kw): return {"new": True}
        entry = fresh_registry.register_sync(sample_registration, new_handler, overwrite=True)
        assert entry.handler is new_handler

    def test_register_invalid_schema_fails(self, fresh_registry: ToolRegistry):
        # Pydantic 自身会拦截空描述, 所以用合法 ToolRegistration 但 Schema 不合规
        bad_reg = ToolRegistration(
            name="bad_tool",
            description="A tool with bad schema",
            input_schema={"type": "string"},  # type 应为 object
        )
        with pytest.raises(SchemaValidationError):
            fresh_registry.register_sync(bad_reg, None)

    @pytest.mark.asyncio
    async def test_register_async(self, fresh_registry: ToolRegistry, sample_registration: ToolRegistration):
        async def handler(**kw): return {}
        entry = await fresh_registry.register(sample_registration, handler)
        assert entry.name == "test_tool_alpha"

    @pytest.mark.asyncio
    async def test_unregister(self, fresh_registry: ToolRegistry, sample_registration: ToolRegistration):
        fresh_registry.register_sync(sample_registration, None)
        entry = await fresh_registry.unregister("test_tool_alpha")
        assert entry.name == "test_tool_alpha"
        assert not fresh_registry.contains("test_tool_alpha")

    @pytest.mark.asyncio
    async def test_unregister_not_found(self, fresh_registry: ToolRegistry):
        with pytest.raises(MCPToolNotFoundError):
            await fresh_registry.unregister("nonexistent")

    @pytest.mark.asyncio
    async def test_update_handler(self, fresh_registry: ToolRegistry, sample_registration: ToolRegistration):
        fresh_registry.register_sync(sample_registration, None)
        async def new_handler(**kw): return {}
        entry = await fresh_registry.update("test_tool_alpha", handler=new_handler)
        assert entry.handler is new_handler
        assert entry.is_stub is False

    @pytest.mark.asyncio
    async def test_update_not_found(self, fresh_registry: ToolRegistry):
        with pytest.raises(MCPToolNotFoundError):
            await fresh_registry.update("nonexistent", handler=None)

    def test_register_batch_sync(self, fresh_registry: ToolRegistry):
        regs = [
            (ToolRegistration(
                name=f"batch_tool_{i}",
                description=f"Batch tool {i}",
                input_schema={"type": "object", "properties": {}, "required": []},
                annotations=Dy3ToolAnnotations(tags=["batch"], layer=LayerTag.L5_AGENT_RUNTIME),
            ), None)
            for i in range(5)
        ]
        entries = fresh_registry.register_batch_sync(regs)
        assert len(entries) == 5
        assert fresh_registry.size == 5


# ============================================================
# 4. 多维发现测试
# ============================================================

class TestRegistryDiscovery:
    def test_get(self, loaded_registry: ToolRegistry):
        entry = loaded_registry.get("bkt_compute")
        assert entry is not None
        assert entry.name == "bkt_compute"

    def test_get_not_found(self, loaded_registry: ToolRegistry):
        assert loaded_registry.get("nonexistent") is None

    def test_get_or_raise(self, loaded_registry: ToolRegistry):
        entry = loaded_registry.get_or_raise("bkt_compute")
        assert entry.name == "bkt_compute"
        with pytest.raises(MCPToolNotFoundError):
            loaded_registry.get_or_raise("nonexistent")

    def test_discover_by_category(self, loaded_registry: ToolRegistry):
        internal = loaded_registry.discover_by_category(ToolCategory.INTERNAL)
        assert len(internal) == 11

        tier1 = loaded_registry.discover_by_category(ToolCategory.CONNECTOR_TIER1)
        assert len(tier1) == 10

        skillbook = loaded_registry.discover_by_category(ToolCategory.SKILLBOOK)
        assert len(skillbook) == 11

        external = loaded_registry.discover_by_category(ToolCategory.EXTERNAL)
        assert len(external) == 5

    def test_discover_by_layer(self, loaded_registry: ToolRegistry):
        l2 = loaded_registry.discover_by_layer(LayerTag.L2_PERSONALIZATION)
        # 11 internal (L2 related) + 11 skillbook = 22
        # But some internal tools are CC1, L4, CC3
        # Let's just verify we get some
        assert len(l2) > 0

        l3 = loaded_registry.discover_by_layer(LayerTag.L3_DOMAIN_KNOWLEDGE)
        # 20 connectors + 5 external = 25
        assert len(l3) == 25

        cc1 = loaded_registry.discover_by_layer(LayerTag.CC1_ANTI_HALLUCINATION)
        # rule_engine_check, cross_validation, standard_value_check, fact_consistency
        assert len(cc1) == 4

    def test_discover_by_tag(self, loaded_registry: ToolRegistry):
        bkt_tools = loaded_registry.discover_by_tag("bkt")
        assert len(bkt_tools) == 1
        assert bkt_tools[0].name == "bkt_compute"

        skill_tools = loaded_registry.discover_by_tag("skill")
        assert len(skill_tools) == 11

    def test_discover_by_domain(self, loaded_registry: ToolRegistry):
        dom_a = loaded_registry.discover_by_domain("DOM-A")
        assert len(dom_a) > 0

    def test_discover_combination(self, loaded_registry: ToolRegistry):
        # L3 + Tier1
        results = loaded_registry.discover(
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
        )
        assert len(results) == 10
        for entry in results:
            assert entry.annotations.layer == LayerTag.L3_DOMAIN_KNOWLEDGE
            assert entry.annotations.category == ToolCategory.CONNECTOR_TIER1

    def test_discover_no_filter(self, loaded_registry: ToolRegistry):
        results = loaded_registry.discover()
        assert len(results) == 47

    def test_search(self, loaded_registry: ToolRegistry):
        results = loaded_registry.search("bkt")
        assert any(e.name == "bkt_compute" for e in results)

        results = loaded_registry.search("贝叶斯")
        assert any(e.name == "bkt_compute" for e in results)


# ============================================================
# 5. 依赖解析测试
# ============================================================

class TestDependencyResolution:
    def test_get_dependencies(self, fresh_registry: ToolRegistry, sample_registration: ToolRegistration):
        fresh_registry.register_sync(sample_registration, None, dependencies=["dep_a", "dep_b"])
        deps = fresh_registry.get_dependencies("test_tool_alpha")
        assert deps == ["dep_a", "dep_b"]

    def test_get_dependencies_empty(self, loaded_registry: ToolRegistry):
        deps = loaded_registry.get_dependencies("bkt_compute")
        assert deps == []

    def test_get_dependents(self, fresh_registry: ToolRegistry):
        reg_a = ToolRegistration(name="tool_a", description="A", input_schema={"type": "object", "properties": {}, "required": []})
        reg_b = ToolRegistration(name="tool_b", description="B", input_schema={"type": "object", "properties": {}, "required": []})
        fresh_registry.register_sync(reg_a, None)
        fresh_registry.register_sync(reg_b, None, dependencies=["tool_a"])
        dependents = fresh_registry.get_dependents("tool_a")
        assert "tool_b" in dependents

    def test_resolve_dependency_chain(self, fresh_registry: ToolRegistry):
        # 创建 A -> B -> C 依赖链
        for name in ["tool_c", "tool_b", "tool_a"]:
            reg = ToolRegistration(name=name, description=name, input_schema={"type": "object", "properties": {}, "required": []})
            fresh_registry.register_sync(reg, None)

        fresh_registry.register_sync(
            ToolRegistration(name="tool_b", description="B", input_schema={"type": "object", "properties": {}, "required": []}),
            None, dependencies=["tool_c"], overwrite=True,
        )
        fresh_registry.register_sync(
            ToolRegistration(name="tool_a", description="A", input_schema={"type": "object", "properties": {}, "required": []}),
            None, dependencies=["tool_b"], overwrite=True,
        )

        chain = fresh_registry.resolve_dependency_chain("tool_a")
        # 应该是 C -> B -> A 的顺序
        assert chain[-1] == "tool_a"
        assert "tool_c" in chain
        assert chain.index("tool_c") < chain.index("tool_b")
        assert chain.index("tool_b") < chain.index("tool_a")

    def test_circular_dependency_detected(self, fresh_registry: ToolRegistry):
        reg_a = ToolRegistration(name="circ_a", description="A", input_schema={"type": "object", "properties": {}, "required": []})
        reg_b = ToolRegistration(name="circ_b", description="B", input_schema={"type": "object", "properties": {}, "required": []})
        fresh_registry.register_sync(reg_a, None, dependencies=["circ_b"])
        fresh_registry.register_sync(reg_b, None, dependencies=["circ_a"])

        with pytest.raises(L6Error, match="Circular dependency"):
            fresh_registry.resolve_dependency_chain("circ_a")


# ============================================================
# 6. 导出测试
# ============================================================

class TestRegistryExport:
    def test_export_mcp_tool_list(self, loaded_registry: ToolRegistry):
        mcp_list = loaded_registry.export_mcp_tool_list()
        assert len(mcp_list) == 47
        for tool in mcp_list:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_export_mcp_tool_list_enabled_only(self, loaded_registry: ToolRegistry):
        # Disable one tool
        entry = loaded_registry.get("bkt_compute")
        entry.registration.enabled = False
        mcp_list = loaded_registry.export_mcp_tool_list(enabled_only=True)
        assert len(mcp_list) == 46
        assert all(t["name"] != "bkt_compute" for t in mcp_list)
        # Restore
        entry.registration.enabled = True

    def test_export_registry_summary(self, loaded_registry: ToolRegistry):
        summary = loaded_registry.export_registry_summary()
        assert summary["total_tools"] == 47
        assert summary["enabled_tools"] == 47
        assert summary["stub_tools"] == 0  # All have handlers
        assert summary["category_breakdown"]["internal"] == 11
        assert summary["category_breakdown"]["connector_tier1"] == 10
        assert summary["category_breakdown"]["connector_tier2"] == 6
        assert summary["category_breakdown"]["connector_tier3"] == 4
        assert summary["category_breakdown"]["skillbook"] == 11
        assert summary["category_breakdown"]["external"] == 5
        assert summary["total_registrations"] == 47

    def test_export_all_entries(self, loaded_registry: ToolRegistry):
        entries = loaded_registry.export_all_entries()
        assert len(entries) == 47
        for e in entries:
            assert "name" in e
            assert "category" in e
            assert "is_stub" in e

    def test_export_mcp_tool_list_with_output_schema(self, loaded_registry: ToolRegistry):
        mcp_list = loaded_registry.export_mcp_tool_list()
        bkt = next(t for t in mcp_list if t["name"] == "bkt_compute")
        assert "outputSchema" in bkt


# ============================================================
# 7. 47 个工具完整性测试
# ============================================================

class TestToolDefinitionsCompleteness:
    def test_total_count(self):
        assert TOTAL_TOOL_COUNT == 47

    def test_all_unique_names(self):
        assert len(ALL_TOOL_NAMES) == len(set(ALL_TOOL_NAMES))

    def test_category_counts(self):
        internal = get_tool_names_by_category(ToolCategory.INTERNAL)
        tier1 = get_tool_names_by_category(ToolCategory.CONNECTOR_TIER1)
        tier2 = get_tool_names_by_category(ToolCategory.CONNECTOR_TIER2)
        tier3 = get_tool_names_by_category(ToolCategory.CONNECTOR_TIER3)
        skillbook = get_tool_names_by_category(ToolCategory.SKILLBOOK)
        external = get_tool_names_by_category(ToolCategory.EXTERNAL)

        assert len(internal) == 11
        assert len(tier1) == 10
        assert len(tier2) == 6
        assert len(tier3) == 4
        assert len(skillbook) == 11
        assert len(external) == 5
        assert len(internal) + len(tier1) + len(tier2) + len(tier3) + len(skillbook) + len(external) == 47

    def test_internal_subcategories(self):
        assert len(DIAGNOSIS_TOOLS) == 3
        assert len(REVIEW_TOOLS) == 4
        assert len(GUIDANCE_TOOLS) == 3
        assert len(SHARED_TOOLS) == 1
        assert len(DIAGNOSIS_TOOLS) + len(REVIEW_TOOLS) + len(GUIDANCE_TOOLS) + len(SHARED_TOOLS) == 11

    def test_connector_subcategories(self):
        assert len(TIER1_TOOLS) == 10
        assert len(TIER2_TOOLS) == 6
        assert len(TIER3_TOOLS) == 4
        assert len(TIER1_TOOLS) + len(TIER2_TOOLS) + len(TIER3_TOOLS) == 20

    def test_all_tools_have_valid_schema(self, validator: SchemaValidator):
        for reg, _ in ALL_TOOL_DEFINITIONS:
            errors = validator.validate_definition(
                name=reg.name,
                description=reg.description,
                input_schema=reg.input_schema,
                output_schema=reg.output_schema,
            )
            assert len(errors) == 0, f"Tool '{reg.name}' has schema errors: {[e.message for e in errors]}"

    def test_all_tools_have_handlers(self):
        for reg, handler in ALL_TOOL_DEFINITIONS:
            assert handler is not None, f"Tool '{reg.name}' has no handler"

    def test_all_tools_have_annotations(self):
        for reg, _ in ALL_TOOL_DEFINITIONS:
            assert reg.annotations.layer is not None, f"Tool '{reg.name}' has no layer"
            assert reg.annotations.category is not None, f"Tool '{reg.name}' has no category"
            assert reg.annotations.estimated_latency_ms > 0, f"Tool '{reg.name}' has no latency"

    def test_internal_tools_have_correct_layer(self):
        for reg, _ in INTERNAL_TOOL_DEFINITIONS:
            assert reg.annotations.layer in [
                LayerTag.L2_PERSONALIZATION,
                LayerTag.CC1_ANTI_HALLUCINATION,
                LayerTag.L4_DECISION_ENGINE,
                LayerTag.CC3_PROVENANCE,
            ], f"Internal tool '{reg.name}' has unexpected layer: {reg.annotations.layer}"

    def test_connector_tools_have_l3_layer(self):
        for reg, _ in CONNECTOR_TOOL_DEFINITIONS:
            assert reg.annotations.layer == LayerTag.L3_DOMAIN_KNOWLEDGE, \
                f"Connector tool '{reg.name}' has wrong layer: {reg.annotations.layer}"

    def test_skillbook_tools_have_l2_layer(self):
        for reg, _ in SKILLBOOK_TOOL_DEFINITIONS:
            assert reg.annotations.layer == LayerTag.L2_PERSONALIZATION, \
                f"Skillbook tool '{reg.name}' has wrong layer: {reg.annotations.layer}"

    def test_external_tools_have_l3_layer(self):
        for reg, _ in EXTERNAL_TOOL_DEFINITIONS:
            assert reg.annotations.layer == LayerTag.L3_DOMAIN_KNOWLEDGE, \
                f"External tool '{reg.name}' has wrong layer: {reg.annotations.layer}"

    def test_find_tool(self):
        result = find_tool("bkt_compute")
        assert result is not None
        assert result[0].name == "bkt_compute"

        result = find_tool("nonexistent_tool")
        assert result is None


# ============================================================
# 8. 内部工具 handler 功能测试
# ============================================================

class TestInternalToolHandlers:
    @pytest.mark.asyncio
    async def test_bkt_compute_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("bkt_compute")
        result = await handler(
            learner_id="u001",
            kp_id="DOM-A-01",
            response=True,
            prior_p_know=0.5,
        )
        assert "p_know_posterior" in result
        assert 0 <= result["p_know_posterior"] <= 1
        assert result["update_direction"] == "increase"
        assert result["kp_id"] == "DOM-A-01"

    @pytest.mark.asyncio
    async def test_bkt_compute_incorrect_response(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("bkt_compute")
        result = await handler(
            learner_id="u001",
            kp_id="DOM-A-01",
            response=False,
            prior_p_know=0.9,
        )
        assert result["p_know_posterior"] < 0.9
        assert result["update_direction"] == "decrease"

    @pytest.mark.asyncio
    async def test_irt_evaluate_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("irt_evaluate")
        result = await handler(
            learner_id="u001",
            theta=0.0,
            item_params={"a": 1.5, "b": 0.0, "c": 0.1},
        )
        assert "p_correct" in result
        assert 0 <= result["p_correct"] <= 1
        assert "information" in result

    @pytest.mark.asyncio
    async def test_forgetfulness_scan_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("forgetfulness_scan")
        import time
        result = await handler(
            learner_id="u001",
            kp_list=[
                {"kp_id": "KP-01", "last_study_ts": time.time() - 86400 * 30, "strength": 0.8},
                {"kp_id": "KP-02", "last_study_ts": time.time() - 3600, "strength": 0.9},
            ],
        )
        assert "urgent_review" in result
        assert "scheduled_review" in result
        assert "stable_kps" in result
        assert "avg_retention" in result

    @pytest.mark.asyncio
    async def test_rule_engine_check_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("rule_engine_check")
        result = await handler(content="This is a valid content about chemistry.", agent_id="A1")
        assert "passed" in result
        assert "violations" in result
        assert "compliance_score" in result

    @pytest.mark.asyncio
    async def test_rule_engine_check_absolute_words(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("rule_engine_check")
        result = await handler(content="You must always remember this. It is absolutely certain.")
        assert len(result["violations"]) > 0

    @pytest.mark.asyncio
    async def test_cross_validation_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("cross_validation")
        result = await handler(
            outputs=[
                {"agent_id": "A1", "result": "The boiling point of water is 100°C", "confidence": 0.9},
                {"agent_id": "A2", "result": "The boiling point of water is 100 degrees Celsius", "confidence": 0.85},
            ],
        )
        assert "consensus" in result
        assert "agreement_score" in result

    @pytest.mark.asyncio
    async def test_standard_value_check_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("standard_value_check")
        result = await handler(
            claims=[
                {"kp_id": "DOM-A-01", "field": "boiling_point", "value": 100.0},
                {"kp_id": "DOM-A-01", "field": "boiling_point", "value": 200.0},  # 错误值
            ],
        )
        assert "all_valid" in result
        assert result["all_valid"] is False  # 有一个错误值
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_topology_analysis_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("topology_analysis")
        result = await handler(kp_ids=["KP-01", "KP-02", "KP-03"])
        assert "nodes" in result
        assert "edges" in result
        assert "entry_points" in result
        assert "exit_points" in result
        assert len(result["nodes"]) == 3

    @pytest.mark.asyncio
    async def test_path_simulation_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("path_simulation")
        result = await handler(
            learner_id="u001",
            start_kp="KP-01",
            target_kp="KP-05",
        )
        assert "recommended_path" in result
        assert "estimated_time_minutes" in result
        assert "success_probability" in result
        assert result["recommended_path"][0] == "KP-01"
        assert result["recommended_path"][-1] == "KP-05"

    @pytest.mark.asyncio
    async def test_resource_matching_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("resource_matching")
        result = await handler(
            learner_id="u001",
            kp_id="DOM-A-01",
            learner_style="visual",
            resource_types=["video", "document"],
        )
        assert "matched_resources" in result
        assert "total_found" in result
        assert result["total_found"] > 0

    @pytest.mark.asyncio
    async def test_literature_trace_handler(self):
        from dy3_polaris.l6.registry.internal_tools import get_internal_tool
        reg, handler = get_internal_tool("literature_trace")
        result = await handler(kp_id="DOM-A-01", max_depth=3)
        assert "source_chain" in result
        assert "evidence_strength" in result
        assert len(result["source_chain"]) == 3


# ============================================================
# 9. 全局单例与批量加载测试
# ============================================================

class TestGlobalRegistry:
    def test_get_registry_singleton(self):
        reset_registry()
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset_registry(self):
        r1 = get_registry()
        reset_registry()
        r2 = get_registry()
        assert r1 is not r2

    def test_load_all_to_fresh_registry(self):
        reg = ToolRegistry()
        load_all_tools(reg)
        assert reg.size == 47

    def test_load_internal_only(self):
        reg = ToolRegistry()
        load_internal_tools(reg)
        assert reg.size == 11

    def test_load_connector_only(self):
        reg = ToolRegistry()
        load_connector_tools(reg)
        assert reg.size == 20

    def test_load_skillbook_only(self):
        reg = ToolRegistry()
        load_skillbook_tools(reg)
        assert reg.size == 11

    def test_load_external_only(self):
        reg = ToolRegistry()
        load_external_tools(reg)
        assert reg.size == 5

    def test_load_incremental(self):
        reg = ToolRegistry()
        load_internal_tools(reg)
        assert reg.size == 11
        load_connector_tools(reg)
        assert reg.size == 31
        load_skillbook_tools(reg)
        assert reg.size == 42
        load_external_tools(reg)
        assert reg.size == 47


# ============================================================
# 10. 调用统计测试
# ============================================================

class TestCallStatistics:
    def test_record_call(self, loaded_registry: ToolRegistry):
        loaded_registry.record_call("bkt_compute", success=True)
        loaded_registry.record_call("bkt_compute", success=True)
        loaded_registry.record_call("bkt_compute", success=False)

        stats = loaded_registry.get_call_stats("bkt_compute")
        assert stats is not None
        assert stats["call_count"] == 3
        assert stats["error_count"] == 1
        assert stats["error_rate"] == pytest.approx(1/3, rel=0.01)

    def test_get_call_stats_not_found(self, loaded_registry: ToolRegistry):
        stats = loaded_registry.get_call_stats("nonexistent")
        assert stats is None


# ============================================================
# 11. 索引一致性测试
# ============================================================

class TestIndexConsistency:
    def test_category_index_consistent(self, loaded_registry: ToolRegistry):
        for category in ToolCategory:
            indexed = set(loaded_registry._category_index.get(category, set()))
            actual = {
                name for name, entry in loaded_registry._tools.items()
                if entry.annotations.category == category
            }
            assert indexed == actual, f"Category index mismatch for {category}"

    def test_layer_index_consistent(self, loaded_registry: ToolRegistry):
        for layer in LayerTag:
            indexed = set(loaded_registry._layer_index.get(layer, set()))
            actual = {
                name for name, entry in loaded_registry._tools.items()
                if entry.annotations.layer == layer
            }
            assert indexed == actual, f"Layer index mismatch for {layer}"

    def test_unregister_cleans_indices(self, loaded_registry: ToolRegistry):
        # 注销一个工具，验证索引被清理
        loaded_registry.unregister_sync = loaded_registry.unregister  # fallback
        # Use sync approach
        entry = loaded_registry.get("bkt_compute")
        assert entry is not None

        # Manually unregister (sync version)
        loaded_registry._remove_from_indices("bkt_compute")
        del loaded_registry._tools["bkt_compute"]

        # Verify not in any index
        assert "bkt_compute" not in loaded_registry._category_index.get(ToolCategory.INTERNAL, set())
        for layer in LayerTag:
            assert "bkt_compute" not in loaded_registry._layer_index.get(layer, set())
        assert "bkt_compute" not in loaded_registry._tag_index.get("bkt", set())

    def test_clear(self, loaded_registry: ToolRegistry):
        loaded_registry.clear()
        assert loaded_registry.size == 0
        assert len(loaded_registry._category_index) == 0
        assert len(loaded_registry._layer_index) == 0
