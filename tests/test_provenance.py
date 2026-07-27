"""溯源协议 - 完整单元测试.

测试覆盖:
1. KPAChain — KPA Merkle 链管理器（追加、封存、回滚、统计等）
2. ProvenanceStore — 多链存储与查询引擎（创建、查询、索引、导出）
3. ChainValidator — 链完整性验证器（prev_hash 连续性、时间戳、置信度等）
4. AuditReportGenerator — 审计报告生成器（摘要、时间线、分析、风险评估）
5. 溯源异常体系（KPAImmutableError、KPANotFoundError、异常继承层级）

所有测试均为同步测试，使用 pytest 框架。
"""

from __future__ import annotations

import json
import time

import pytest

from dy3_polaris.l6.core.exceptions import (
    KPAChainBrokenError,
    KPAImmutableError,
    KPANotFoundError,
    KPAValidationError,
    L6Error,
    ProvenanceError,
)
from dy3_polaris.l6.core.models import KPA, KPAEventType, LayerTag
from dy3_polaris.l6.provenance import (
    AuditReportGenerator,
    ChainValidator,
    KPAChain,
    ProvenanceStore,
    ValidationResult,
)


# ============================================================
# 1. KPAChain 测试
# ============================================================

class TestKPAChain:
    """KPA Merkle 链管理器测试."""

    def test_empty_chain(self):
        """空链的基本属性."""
        chain = KPAChain()
        assert chain.length == 0
        assert chain.is_empty is True
        assert chain.head_hash is None
        assert chain.genesis_hash is None

    def test_append_single(self):
        """追加单个 KPA，验证基本属性."""
        chain = KPAChain()
        kpa = chain.append(
            KPAEventType.TOOL_INVOKED, "test_actor", LayerTag.L6_PROTOCOL
        )
        assert chain.length == 1
        assert chain.is_empty is False
        # 创世 KPA 的 prev_hash 应为 None
        assert kpa.prev_hash is None

    def test_append_chain(self):
        """追加 3 个 KPA，验证 prev_hash 链接关系."""
        chain = KPAChain()
        kpa1 = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        kpa2 = chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        kpa3 = chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        # 创世 KPA 的 prev_hash 为 None
        assert kpa1.prev_hash is None
        # 每个 KPA 的 prev_hash 指向前一个 KPA 的哈希
        assert kpa2.prev_hash == kpa1.compute_hash()
        assert kpa3.prev_hash == kpa2.compute_hash()

    def test_head_and_genesis_hash(self):
        """验证 head_hash 和 genesis_hash."""
        chain = KPAChain()
        kpa1 = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        kpa3 = chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        # head_hash 为链尾（最新）KPA 的哈希
        assert chain.head_hash == kpa3.compute_hash()
        # genesis_hash 为创世（第一个）KPA 的哈希
        assert chain.genesis_hash == kpa1.compute_hash()

    def test_seal(self):
        """封存链后追加应抛 KPAImmutableError."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.seal()
        assert chain.is_sealed is True
        with pytest.raises(KPAImmutableError):
            chain.append(
                KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
            )

    def test_unseal(self):
        """解除封存后可以继续追加."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.seal()
        chain.unseal()
        assert chain.is_sealed is False
        # 解除封存后可以继续追加
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        assert chain.length == 2

    def test_get_by_index(self):
        """按索引获取 KPA."""
        chain = KPAChain()
        kpa1 = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        kpa2 = chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        kpa3 = chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        assert chain.get(0) is kpa1
        assert chain.get(1) is kpa2
        assert chain.get(2) is kpa3
        # 越界索引返回 None
        assert chain.get(3) is None

    def test_get_by_id(self):
        """按 kpa_id 查找 KPA."""
        chain = KPAChain()
        kpa = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        assert chain.get_by_id(kpa.kpa_id) is kpa
        # 不存在的 kpa_id 返回 None
        assert chain.get_by_id("nonexistent") is None

    def test_slice(self):
        """获取链子段."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        chain.append(
            KPAEventType.RESOURCE_READ, "actor4", LayerTag.L3_DOMAIN_KNOWLEDGE
        )
        result = chain.slice(1, 3)
        assert len(result) == 2
        assert result[0].actor == "actor2"
        assert result[1].actor == "actor3"

    def test_snapshot(self):
        """获取链快照（序列化）."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        snapshot = chain.snapshot()
        assert isinstance(snapshot, list)
        assert len(snapshot) == 2
        for item in snapshot:
            assert isinstance(item, dict)
            assert "kpa_id" in item
            assert "event_type" in item

    def test_rollback(self):
        """回滚链到指定索引."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        chain.append(
            KPAEventType.RESOURCE_READ, "actor4", LayerTag.L3_DOMAIN_KNOWLEDGE
        )
        # 回滚到索引 1（保留索引 0 和 1，移除 2 和 3）
        removed = chain.rollback(1)
        assert removed == 2
        assert chain.length == 2

    def test_rollback_sealed(self):
        """封存链后回滚应抛 KPAImmutableError."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        chain.seal()
        with pytest.raises(KPAImmutableError):
            chain.rollback(0)

    def test_stats(self):
        """统计 event_type_counts, actor_counts, layer_counts."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor1", LayerTag.L6_PROTOCOL
        )

        # 按事件类型统计
        etc = chain.event_type_counts()
        assert etc["tool_invoked"] == 2
        assert etc["agent_output"] == 1

        # 按执行者统计
        ac = chain.actor_counts()
        assert ac["actor1"] == 2
        assert ac["actor2"] == 1

        # 按层标签统计
        lc = chain.layer_counts()
        assert lc["L6"] == 2
        assert lc["L5"] == 1

    def test_avg_confidence(self):
        """平均置信度计算."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL,
            confidence=0.8,
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME,
            confidence=0.6,
        )
        # 无 confidence 的 KPA 不参与平均
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        avg = chain.avg_confidence()
        assert avg is not None
        assert avg == pytest.approx(0.7)

    def test_clear(self):
        """清空链."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.seal()
        chain.clear()
        assert chain.length == 0
        assert chain.is_sealed is False


# ============================================================
# 2. ProvenanceStore 测试
# ============================================================

class TestProvenanceStore:
    """溯源存储与查询引擎测试."""

    def test_create_chain(self):
        """创建链."""
        store = ProvenanceStore()
        chain = store.create_chain("c1")
        assert isinstance(chain, KPAChain)
        assert store.chain_count == 1

    def test_create_chain_auto_id(self):
        """不传 ID 时自动生成 chain_id."""
        store = ProvenanceStore()
        chain = store.create_chain()
        assert chain.chain_id  # 非空字符串
        assert chain.chain_id == "chain-0000"

    def test_create_chain_duplicate(self):
        """重复创建返回同一条链."""
        store = ProvenanceStore()
        chain1 = store.create_chain("c1")
        chain2 = store.create_chain("c1")
        assert chain1 is chain2
        assert store.chain_count == 1

    def test_get_chain(self):
        """获取链."""
        store = ProvenanceStore()
        store.create_chain("c1")
        assert store.get_chain("c1") is not None
        assert store.get_chain("nope") is None

    def test_get_chain_or_raise(self):
        """不存在的链抛 KPANotFoundError."""
        store = ProvenanceStore()
        with pytest.raises(KPANotFoundError):
            store.get_chain_or_raise("nope")

    def test_remove_chain(self):
        """移除链."""
        store = ProvenanceStore()
        store.create_chain("c1")
        assert store.remove_chain("c1") is True
        assert store.chain_count == 0

    def test_total_kpa_count(self):
        """跨链 KPA 总数."""
        store = ProvenanceStore()
        chain1 = store.create_chain("c1")
        chain2 = store.create_chain("c2")
        for i in range(3):
            chain1.append(
                KPAEventType.TOOL_INVOKED, f"actor{i}", LayerTag.L6_PROTOCOL
            )
            chain2.append(
                KPAEventType.AGENT_OUTPUT, f"actor{i}", LayerTag.L5_AGENT_RUNTIME
            )
        assert store.total_kpa_count == 6

    def test_get_kpa(self):
        """按 kpa_id 全局查找 KPA."""
        store = ProvenanceStore()
        chain = store.create_chain("c1")
        kpa = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        found = store.get_kpa(kpa.kpa_id)
        assert found is not None
        assert found.kpa_id == kpa.kpa_id

    def test_query_by_actor(self):
        """按执行者查询."""
        store = ProvenanceStore()
        chain = store.create_chain("c1")
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor_a", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor_b", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor_a", LayerTag.L6_PROTOCOL
        )
        results = store.query_by_actor("actor_a")
        assert len(results) == 2
        for kpa in results:
            assert kpa.actor == "actor_a"

    def test_query_multi_conditions(self):
        """多条件 AND 查询."""
        store = ProvenanceStore()
        chain = store.create_chain("c1")
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor_a", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor_a", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor_a", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor_b", LayerTag.L6_PROTOCOL
        )
        # 三条件 AND 查询
        results = store.query(
            actor="actor_a",
            event_type=KPAEventType.TOOL_INVOKED,
            layer=LayerTag.L6_PROTOCOL,
        )
        assert len(results) == 1
        assert results[0].actor == "actor_a"
        assert results[0].event_type == KPAEventType.TOOL_INVOKED
        assert results[0].layer == LayerTag.L6_PROTOCOL

    def test_query_low_confidence(self):
        """查询低置信度 KPA."""
        store = ProvenanceStore()
        chain = store.create_chain("c1")
        kpa1 = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL,
            confidence=0.3,
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME,
            confidence=0.7,
        )
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        # threshold=0.5，只返回 confidence < 0.5 且非 None 的 KPA
        results = store.query_low_confidence(0.5)
        assert len(results) == 1
        assert results[0].kpa_id == kpa1.kpa_id

    def test_find_all_actors(self):
        """获取所有执行者（排序去重）."""
        store = ProvenanceStore()
        chain = store.create_chain("c1")
        chain.append(
            KPAEventType.TOOL_INVOKED, "charlie", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "alice", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.TOOL_INVOKED, "bob", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "alice", LayerTag.L5_AGENT_RUNTIME
        )
        actors = store.find_all_actors()
        # 返回排序后的去重列表
        assert actors == ["alice", "bob", "charlie"]

    def test_export_summary(self):
        """导出摘要."""
        store = ProvenanceStore()
        chain = store.create_chain("c1")
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        summary = store.export_summary()
        assert "chain_count" in summary
        assert "total_kpas" in summary
        assert "chains" in summary
        assert summary["chain_count"] == 1
        assert summary["total_kpas"] == 2


# ============================================================
# 3. ChainValidator 测试
# ============================================================

class TestChainValidator:
    """KPA 链完整性验证器测试."""

    def test_validate_empty_chain(self):
        """空链验证通过."""
        validator = ChainValidator()
        chain = KPAChain()
        result = validator.validate(chain)
        assert result.is_valid is True

    def test_validate_valid_chain(self):
        """有效链验证通过."""
        validator = ChainValidator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        time.sleep(0.01)
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        time.sleep(0.01)
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        result = validator.validate(chain)
        assert result.is_valid is True
        assert result.errors == []

    def test_validate_broken_chain(self):
        """prev_hash 断裂检测."""
        validator = ChainValidator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        kpa2 = chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        # 手动破坏 kpa2 的 prev_hash
        kpa2.prev_hash = "wrong_hash_value"
        result = validator.validate(chain, strict=False)
        assert result.is_valid is False
        checks = [e["check"] for e in result.errors]
        assert "prev_hash_continuity" in checks

    def test_validate_genesis_prev_hash(self):
        """创世 KPA 的 prev_hash 应为 None."""
        validator = ChainValidator()
        chain = KPAChain()
        kpa1 = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        # 手动设置创世 KPA 的 prev_hash 为非 None
        kpa1.prev_hash = "should_be_none"
        result = validator.validate(chain, strict=False)
        assert result.is_valid is False
        checks = [e["check"] for e in result.errors]
        assert "genesis_prev_hash" in checks

    def test_validate_timestamp_disorder(self):
        """时间戳非单调递增检测."""
        validator = ChainValidator()
        chain = KPAChain()
        kpa1 = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        kpa2 = chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        # 手动设置时间戳逆序（仅修改 kpa2，不影响 prev_hash 连续性）
        kpa2.timestamp = kpa1.timestamp - 10
        result = validator.validate(chain, strict=False)
        checks = [w["check"] for w in result.warnings]
        assert "timestamp_order" in checks

    def test_validate_confidence_range(self):
        """置信度超出 [0,1] 范围检测."""
        validator = ChainValidator()
        chain = KPAChain()
        kpa1 = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        # 手动设置置信度超出范围（Pydantic 构造后修改不触发验证）
        kpa1.confidence = 1.5
        result = validator.validate(chain, strict=False)
        assert result.is_valid is False
        checks = [e["check"] for e in result.errors]
        assert "confidence_range" in checks

    def test_validate_strict_mode(self):
        """严格模式下 warning 也视为无效."""
        validator = ChainValidator()
        chain = KPAChain()
        kpa1 = chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        kpa2 = chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        # 制造时间戳逆序（仅产生 warning，无 error）
        kpa2.timestamp = kpa1.timestamp - 10

        # 非严格模式：有 warning 但 is_valid=True
        result_loose = validator.validate(chain, strict=False)
        assert result_loose.is_valid is True
        assert result_loose.warning_count > 0

        # 严格模式：有 warning 则 is_valid=False
        result_strict = validator.validate(chain, strict=True)
        assert result_strict.is_valid is False

    def test_quick_check_valid(self):
        """快速检查有效链."""
        validator = ChainValidator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        assert validator.quick_check(chain) is True

    def test_quick_check_broken(self):
        """快速检查断裂链."""
        validator = ChainValidator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        kpa2 = chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        # 手动破坏 prev_hash
        kpa2.prev_hash = "broken"
        assert validator.quick_check(chain) is False

    def test_validate_kpa_single(self):
        """验证单个正常 KPA."""
        validator = ChainValidator()
        kpa = KPA(
            event_type=KPAEventType.TOOL_INVOKED,
            actor="actor1",
            layer=LayerTag.L6_PROTOCOL,
        )
        result = validator.validate_kpa(kpa)
        assert result.is_valid is True

    def test_validate_kpa_bad_confidence(self):
        """验证置信度异常的单个 KPA."""
        validator = ChainValidator()
        kpa = KPA(
            event_type=KPAEventType.TOOL_INVOKED,
            actor="actor1",
            layer=LayerTag.L6_PROTOCOL,
        )
        # 构造后修改 confidence 为非法值
        kpa.confidence = -0.5
        result = validator.validate_kpa(kpa)
        assert result.is_valid is False
        checks = [e["check"] for e in result.errors]
        assert "confidence_range" in checks

    def test_validation_result_properties(self):
        """ValidationResult 的 error_count, warning_count, to_dict."""
        result = ValidationResult(
            is_valid=False,
            chain_id="test-chain",
            total_kpas=3,
            checked_kpas=3,
            errors=[{"check": "test_error", "message": "测试错误"}],
            warnings=[{"check": "test_warning", "message": "测试警告"}],
        )
        assert result.error_count == 1
        assert result.warning_count == 1
        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["error_count"] == 1
        assert d["warning_count"] == 1
        assert d["chain_id"] == "test-chain"
        assert len(d["errors"]) == 1
        assert len(d["warnings"]) == 1


# ============================================================
# 4. AuditReport 测试
# ============================================================

class TestAuditReport:
    """溯源审计报告生成器测试."""

    def test_generate_basic(self):
        """生成基本审计报告，验证包含所有主要键."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        report = generator.generate(chain)
        # 验证报告包含所有主要键
        assert "report_id" in report
        assert "chain_info" in report
        assert "summary" in report
        assert "event_timeline" in report
        assert "validation" in report
        assert "actor_analysis" in report
        assert "layer_analysis" in report
        assert "confidence_analysis" in report
        assert "risk_assessment" in report

    def test_summary_fields(self):
        """摘要字段完整性."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        report = generator.generate(chain)
        summary = report["summary"]
        assert "total_events" in summary
        assert "is_valid" in summary
        assert "event_types" in summary
        assert "actors_involved" in summary
        assert summary["total_events"] == 2

    def test_event_timeline(self):
        """事件时间线长度与 KPA 数量一致."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        report = generator.generate(chain)
        timeline = report["event_timeline"]
        assert len(timeline) == 3
        for i, item in enumerate(timeline):
            assert "index" in item
            assert "kpa_id" in item
            assert "event_label" in item
            assert item["index"] == i

    def test_actor_analysis(self):
        """执行者分析按事件数降序排列."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor_a", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor_a", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor_a", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor_b", LayerTag.L4_DECISION_ENGINE
        )
        report = generator.generate(chain)
        analysis = report["actor_analysis"]
        assert len(analysis) == 2
        # actor_a 有 3 个事件，actor_b 有 1 个，按事件数降序排列
        assert analysis[0]["actor"] == "actor_a"
        assert analysis[0]["total_events"] == 3
        assert analysis[1]["actor"] == "actor_b"
        assert analysis[1]["total_events"] == 1

    def test_layer_analysis(self):
        """层标签分析按 layer 升序排列."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        chain.append(
            KPAEventType.DECISION_ROUTED, "actor3", LayerTag.L4_DECISION_ENGINE
        )
        report = generator.generate(chain)
        analysis = report["layer_analysis"]
        assert len(analysis) == 3
        # 按 layer 字符串升序排列：L4 < L5 < L6
        assert analysis[0]["layer"] == "L4"
        assert analysis[1]["layer"] == "L5"
        assert analysis[2]["layer"] == "L6"

    def test_confidence_analysis(self):
        """置信度分析：有数据和无数据两种情况."""
        generator = AuditReportGenerator()

        # 情况 1：有置信度数据
        chain1 = KPAChain()
        chain1.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL,
            confidence=0.8,
        )
        chain1.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME,
            confidence=0.4,
        )
        report1 = generator.generate(chain1)
        ca1 = report1["confidence_analysis"]
        assert ca1["has_data"] is True
        assert ca1["count"] == 2
        assert ca1["min"] == 0.4
        assert ca1["max"] == 0.8
        assert ca1["low_confidence_count"] == 1  # 0.4 < 0.5

        # 情况 2：无置信度数据
        chain2 = KPAChain()
        chain2.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain2.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        report2 = generator.generate(chain2)
        ca2 = report2["confidence_analysis"]
        assert ca2["has_data"] is False
        assert ca2["count"] == 0

    def test_risk_assessment_valid(self):
        """有效链的风险等级为 low（未封存的 low risk）."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        report = generator.generate(chain)
        risk = report["risk_assessment"]
        assert risk["level"] == "low"

    def test_risk_assessment_invalid(self):
        """无效链的风险等级为 high."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        kpa2 = chain.append(
            KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
        )
        # 破坏链完整性
        kpa2.prev_hash = "broken"
        report = generator.generate(chain)
        risk = report["risk_assessment"]
        assert risk["level"] == "high"

    def test_to_json(self):
        """导出为有效 JSON 字符串."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        report = generator.generate(chain)
        json_str = generator.to_json(report)
        assert isinstance(json_str, str)
        # 验证是有效 JSON
        parsed = json.loads(json_str)
        assert parsed["summary"]["total_events"] == 1

    def test_to_text(self):
        """导出为人类可读文本."""
        generator = AuditReportGenerator()
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        report = generator.generate(chain)
        text = generator.to_text(report)
        assert isinstance(text, str)
        assert "溯源审计报告" in text


# ============================================================
# 5. 溯源异常测试
# ============================================================

class TestProvenanceExceptions:
    """溯源协议异常体系测试."""

    def test_kpa_immutable_error(self):
        """封存链后追加抛出 KPAImmutableError."""
        chain = KPAChain()
        chain.append(
            KPAEventType.TOOL_INVOKED, "actor1", LayerTag.L6_PROTOCOL
        )
        chain.seal()
        with pytest.raises(KPAImmutableError):
            chain.append(
                KPAEventType.AGENT_OUTPUT, "actor2", LayerTag.L5_AGENT_RUNTIME
            )

    def test_kpa_not_found_error(self):
        """不存在的链抛 KPANotFoundError."""
        store = ProvenanceStore()
        with pytest.raises(KPANotFoundError):
            store.get_chain_or_raise("nonexistent")

    def test_provenance_error_hierarchy(self):
        """所有溯源异常都是 ProvenanceError 的子类，也是 L6Error 的子类."""
        # 所有溯源异常都是 ProvenanceError 的子类
        assert issubclass(KPAChainBrokenError, ProvenanceError)
        assert issubclass(KPAImmutableError, ProvenanceError)
        assert issubclass(KPANotFoundError, ProvenanceError)
        assert issubclass(KPAValidationError, ProvenanceError)
        # ProvenanceError 是 L6Error 的子类
        assert issubclass(ProvenanceError, L6Error)
        # 所有溯源异常也是 L6Error 的子类
        assert issubclass(KPAChainBrokenError, L6Error)
        assert issubclass(KPAImmutableError, L6Error)
        assert issubclass(KPANotFoundError, L6Error)
        assert issubclass(KPAValidationError, L6Error)
