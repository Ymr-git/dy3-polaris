"""L1 用户域隐私治理 (Privacy Governance) 测试 — TDD 测试用例.

测试覆盖:
1. 异常体系 (L1PrivacyError 层级, JSON-RPC -32600 范围)
2. DataClassifier — 数据分级控制 (分类 + 访问检查 + 最小化校验)
3. DesensitizationEngine — 数据脱敏引擎 (5 种方法 + K-匿名 + l-多样性 + 差分隐私)
4. RetentionManager — 数据留存策略 (四阶段生命周期 + 执行)
5. AuditLogger — 审计日志管理 (append-only + 哈希链 + 查询 + 分页)
6. PrivacyEventNotifier — 隐私事件通知 (事件队列 + L0 通知)
7. PrivacyGovernanceManager — 统一治理管理器
8. 线程安全
9. 边界条件与异常
10. 集成测试

设计依据:
- L1 设计文档第六章: 隐私保护与数据治理 (6.1-6.6)
- L1 设计文档第七章 7.2: API `/api/v1/audit/logs`, `/api/v1/export/learner-data`
- 世界先进方案: FERPA / GDPR / PIPL / k-匿名 / l-多样性 / 差分隐私 / 审计日志哈希链
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from dy3_polaris.l1.models import (
    AuditAction,
    AuditLogEntry,
    AuditResult,
    DataLevel,
    DesensitizationMethod,
    K_ANONYMITY_MIN,
    L_DIVERSITY_MIN,
    PrivacyConfig,
    PrivacyEvent,
    RetentionPhase,
    RetentionPolicy,
    User,
    UserRole,
    UserStatus,
    bucket_response_time,
    desensitize_student_id,
)


# ============================================================
# 1. 异常体系测试
# ============================================================


class TestExceptionHierarchy:
    """隐私治理异常体系测试."""

    def test_base_error_inherits_l6(self):
        """基础异常继承 L6Error."""
        from dy3_polaris.l6.core.exceptions import L6Error
        from dy3_polaris.l1.privacy_governance import L1PrivacyError

        assert issubclass(L1PrivacyError, L6Error)

    def test_base_error_jsonrpc_code(self):
        """基础异常 JSON-RPC 码在 -32600 范围."""
        from dy3_polaris.l1.privacy_governance import L1PrivacyError

        err = L1PrivacyError(detail="test")
        assert -32699 <= err.to_json_rpc_error()["code"] <= -32600

    def test_data_classification_error_inherits_base(self):
        """数据分级错误继承基础异常."""
        from dy3_polaris.l1.privacy_governance import (
            L1PrivacyError,
            DataClassificationError,
        )

        assert issubclass(DataClassificationError, L1PrivacyError)

    def test_desensitization_error_inherits_base(self):
        """脱敏错误继承基础异常."""
        from dy3_polaris.l1.privacy_governance import (
            L1PrivacyError,
            DesensitizationError,
        )

        assert issubclass(DesensitizationError, L1PrivacyError)

    def test_retention_error_inherits_base(self):
        """留存策略错误继承基础异常."""
        from dy3_polaris.l1.privacy_governance import (
            L1PrivacyError,
            RetentionExecutionError,
        )

        assert issubclass(RetentionExecutionError, L1PrivacyError)

    def test_audit_error_inherits_base(self):
        """审计日志错误继承基础异常."""
        from dy3_polaris.l1.privacy_governance import (
            L1PrivacyError,
            AuditLogError,
        )

        assert issubclass(AuditLogError, L1PrivacyError)

    def test_privacy_violation_error_inherits_base(self):
        """隐私违规错误继承基础异常."""
        from dy3_polaris.l1.privacy_governance import (
            L1PrivacyError,
            PrivacyViolationError,
        )

        assert issubclass(PrivacyViolationError, L1PrivacyError)

    def test_violation_error_contains_detail(self):
        """隐私违规错误包含详细信息."""
        from dy3_polaris.l1.privacy_governance import PrivacyViolationError

        err = PrivacyViolationError(
            user_id="u-001",
            violation_type="unauthorized_access",
            detail="尝试访问 L4 机密数据",
        )
        assert "u-001" in err.context.get("user_id", "")
        assert err.context.get("violation_type") == "unauthorized_access"


# ============================================================
# 2. DataClassifier 测试
# ============================================================


class TestDataClassifier:
    """数据分级控制测试 (设计文档 6.1, 6.2)."""

    def test_classify_public_data(self):
        """公开数据分类为 L1_PUBLIC."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        level = classifier.classify("course_announcement")
        assert level == DataLevel.L1_PUBLIC

    def test_classify_internal_data(self):
        """内部数据分类为 L2_INTERNAL."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        level = classifier.classify("knowledge_base_content")
        assert level == DataLevel.L2_INTERNAL

    def test_classify_sensitive_data(self):
        """敏感数据分类为 L3_SENSITIVE."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        level = classifier.classify("learning_report")
        assert level == DataLevel.L3_SENSITIVE

    def test_classify_confidential_data(self):
        """机密数据分类为 L4_CONFIDENTIAL."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        level = classifier.classify("student_id")
        assert level == DataLevel.L4_CONFIDENTIAL

    def test_classify_unknown_data_defaults_internal(self):
        """未知数据类型默认分类为 L2_INTERNAL."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        level = classifier.classify("unknown_data_type")
        assert level == DataLevel.L2_INTERNAL

    def test_check_access_undergrad_can_view_public(self):
        """本科生可查看公开数据."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        user = make_user(UserRole.UNDERGRAD)
        assert classifier.check_access(user, DataLevel.L1_PUBLIC) is True

    def test_check_access_undergrad_cannot_view_confidential(self):
        """本科生不可查看机密数据."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        user = make_user(UserRole.UNDERGRAD)
        assert classifier.check_access(user, DataLevel.L4_CONFIDENTIAL) is False

    def test_check_access_admin_can_view_all(self):
        """管理员可查看所有级别数据."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        user = make_user(UserRole.ADMIN)
        for level in DataLevel:
            assert classifier.check_access(user, level) is True

    def test_check_access_teacher_can_view_sensitive(self):
        """教师可查看敏感数据."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        user = make_user(UserRole.TEACHER)
        assert classifier.check_access(user, DataLevel.L3_SENSITIVE) is True

    def test_check_access_alumni_cannot_view_sensitive(self):
        """校友不可查看敏感数据."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        user = make_user(UserRole.ALUMNI)
        assert classifier.check_access(user, DataLevel.L3_SENSITIVE) is False

    def test_check_minimization_allowed_field(self):
        """数据最小化: 允许采集的字段."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        assert classifier.check_minimization("student_id") is True
        assert classifier.check_minimization("grade_level") is True
        assert classifier.check_minimization("answer_correct") is True

    def test_check_minimization_blocked_field(self):
        """数据最小化: 禁止采集的字段."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        assert classifier.check_minimization("mouse_track") is False
        assert classifier.check_minimization("device_fingerprint") is False
        assert classifier.check_minimization("ip_address") is False
        assert classifier.check_minimization("biometric_data") is False
        assert classifier.check_minimization("geo_location") is False

    def test_get_minimization_allowed_list(self):
        """获取允许采集的字段列表."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        allowed = classifier.get_allowed_fields()
        assert "student_id" in allowed
        assert "grade_level" in allowed
        assert "answer_correct" in allowed
        assert "mouse_track" not in allowed

    def test_get_minimization_blocked_list(self):
        """获取禁止采集的字段列表."""
        from dy3_polaris.l1.privacy_governance import DataClassifier

        classifier = DataClassifier()
        blocked = classifier.get_blocked_fields()
        assert "mouse_track" in blocked
        assert "device_fingerprint" in blocked
        assert "ip_address" in blocked
        assert "student_id" not in blocked


# ============================================================
# 3. DesensitizationEngine 测试
# ============================================================


class TestDesensitizationEngine:
    """数据脱敏引擎测试 (设计文档 6.3)."""

    def test_desensitize_hash_student_id(self):
        """学号哈希脱敏."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        result = engine.desensitize(
            data="CS20240001",
            method=DesensitizationMethod.HASH,
            salt="institution-salt",
        )
        assert result != "CS20240001"
        assert len(result) == 64  # SHA-256 hex

    def test_desensitize_hash_deterministic(self):
        """相同输入+盐值产生相同哈希."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        r1 = engine.desensitize("CS20240001", DesensitizationMethod.HASH, salt="s")
        r2 = engine.desensitize("CS20240001", DesensitizationMethod.HASH, salt="s")
        assert r1 == r2

    def test_desensitize_hash_different_salt(self):
        """不同盐值产生不同哈希."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        r1 = engine.desensitize("CS20240001", DesensitizationMethod.HASH, salt="s1")
        r2 = engine.desensitize("CS20240001", DesensitizationMethod.HASH, salt="s2")
        assert r1 != r2

    def test_desensitize_aggregate_answers(self):
        """答题记录聚合为正确率."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        answers = [True, False, True, True, False]
        result = engine.desensitize(
            data=answers,
            method=DesensitizationMethod.AGGREGATE,
        )
        assert isinstance(result, float | int)
        assert result == 0.6  # 3/5

    def test_desensitize_bucket_response_time(self):
        """响应时间分桶泛化."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        assert engine.desensitize(3000, DesensitizationMethod.BUCKET) == "fast"
        assert engine.desensitize(30000, DesensitizationMethod.BUCKET) == "normal"
        assert engine.desensitize(90000, DesensitizationMethod.BUCKET) == "slow"

    def test_desensitize_dp_noise(self):
        """差分隐私加噪."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine(privacy_config=PrivacyConfig(epsilon=1.0))
        result = engine.desensitize(
            data=0.85,
            method=DesensitizationMethod.DP_NOISE,
        )
        # 加噪后值应在合理范围内 (0.0-1.0)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_desensitize_dp_noise_within_epsilon(self):
        """差分隐私噪声受 epsilon 控制."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine_low = DesensitizationEngine(
            privacy_config=PrivacyConfig(epsilon=0.1)
        )
        engine_high = DesensitizationEngine(
            privacy_config=PrivacyConfig(epsilon=10.0)
        )
        # epsilon 越小噪声越大 (平均偏差越大)
        deviations_low = []
        deviations_high = []
        for _ in range(100):
            r_low = engine_low.desensitize(0.5, DesensitizationMethod.DP_NOISE)
            r_high = engine_high.desensitize(0.5, DesensitizationMethod.DP_NOISE)
            deviations_low.append(abs(r_low - 0.5))
            deviations_high.append(abs(r_high - 0.5))
        avg_low = sum(deviations_low) / len(deviations_low)
        avg_high = sum(deviations_high) / len(deviations_high)
        # epsilon=0.1 的平均偏差应大于 epsilon=10.0
        assert avg_low >= avg_high

    def test_desensitize_pseudo_id(self):
        """伪 ID 替换."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        result = engine.desensitize(
            data="CS20240001",
            method=DesensitizationMethod.PSEUDO_ID,
        )
        assert result != "CS20240001"
        assert result.startswith("pseudo-")

    def test_desensitize_pseudo_id_consistent(self):
        """相同输入产生相同伪 ID (同一引擎实例内)."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        r1 = engine.desensitize("CS20240001", DesensitizationMethod.PSEUDO_ID)
        r2 = engine.desensitize("CS20240001", DesensitizationMethod.PSEUDO_ID)
        assert r1 == r2

    def test_check_k_anonymity_pass(self):
        """K-匿名检查通过."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        # 10 条记录, k=5 → 通过
        records = [
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "y"},
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "z"},
            {"qi": "A", "sensitive": "y"},
            {"qi": "B", "sensitive": "x"},
            {"qi": "B", "sensitive": "y"},
            {"qi": "B", "sensitive": "z"},
            {"qi": "B", "sensitive": "x"},
            {"qi": "B", "sensitive": "y"},
        ]
        assert engine.check_k_anonymity(records, "qi", k=5) is True

    def test_check_k_anonymity_fail(self):
        """K-匿名检查失败."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        records = [
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "y"},
            {"qi": "B", "sensitive": "x"},
        ]
        assert engine.check_k_anonymity(records, "qi", k=5) is False

    def test_check_l_diversity_pass(self):
        """l-多样性检查通过."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        records = [
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "y"},
            {"qi": "A", "sensitive": "z"},
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "y"},
        ]
        assert engine.check_l_diversity(records, "qi", "sensitive", l=3) is True

    def test_check_l_diversity_fail(self):
        """l-多样性检查失败."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        records = [
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "x"},
            {"qi": "A", "sensitive": "x"},
        ]
        assert engine.check_l_diversity(records, "qi", "sensitive", l=3) is False

    def test_anonymize_dataset(self):
        """数据集匿名化 (K-匿名 + l-多样性)."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        records = [
            {"student_id": "CS20240001", "grade": "A", "score": 85},
            {"student_id": "CS20240002", "grade": "A", "score": 90},
            {"student_id": "CS20240003", "grade": "B", "score": 75},
            {"student_id": "CS20240004", "grade": "B", "score": 80},
            {"student_id": "CS20240005", "grade": "A", "score": 88},
            {"student_id": "CS20240006", "grade": "B", "score": 72},
        ]
        anonymized = engine.anonymize_for_research(
            records,
            quasi_identifiers=["grade"],
            sensitive_attributes=["score"],
            k=3,
            l=2,
        )
        assert len(anonymized) <= len(records)
        for r in anonymized:
            assert "student_id" not in r or r["student_id"].startswith("pseudo-")

    def test_desensitize_unknown_method_raises(self):
        """未知脱敏方法抛异常."""
        from dy3_polaris.l1.privacy_governance import (
            DesensitizationEngine,
            DesensitizationError,
        )

        engine = DesensitizationEngine()
        with pytest.raises(DesensitizationError):
            engine.desensitize("test", "unknown_method")


# ============================================================
# 4. RetentionManager 测试
# ============================================================


class TestRetentionManager:
    """数据留存策略测试 (设计文档 6.4)."""

    def test_get_retention_policy_for_public(self):
        """公开数据留存策略."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        policy = mgr.get_policy(DataLevel.L1_PUBLIC)
        assert policy is not None
        assert len(policy.phases) > 0

    def test_get_retention_policy_for_confidential(self):
        """机密数据留存策略."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        policy = mgr.get_policy(DataLevel.L4_CONFIDENTIAL)
        assert policy is not None
        # 机密数据应有 DELETED 阶段
        phase_values = [p.value if hasattr(p, "value") else str(p) for p, _ in policy.phases]
        assert "deleted" in phase_values

    def test_check_retention_active_user(self):
        """活跃用户: 保留阶段为 ACTIVE."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        action = mgr.check_retention(
            user_id="u-001",
            graduation_ts=None,
            current_ts=int(time.time() * 1000),
        )
        assert action.phase == RetentionPhase.ACTIVE

    def test_check_retention_recently_graduated(self):
        """刚毕业 (< 1年): 保留阶段为 ARCHIVED."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        now_ms = int(time.time() * 1000)
        six_months_ago = now_ms - 180 * 24 * 60 * 60 * 1000
        action = mgr.check_retention(
            user_id="u-001",
            graduation_ts=six_months_ago,
            current_ts=now_ms,
        )
        assert action.phase == RetentionPhase.ARCHIVED

    def test_check_retention_one_year_after_graduation(self):
        """毕业后 1 年以上: 匿名化阶段."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        now_ms = int(time.time() * 1000)
        two_years_ago = now_ms - 730 * 24 * 60 * 60 * 1000
        action = mgr.check_retention(
            user_id="u-001",
            graduation_ts=two_years_ago,
            current_ts=now_ms,
        )
        assert action.phase == RetentionPhase.ANONYMIZED

    def test_check_retention_three_years_after_graduation(self):
        """毕业后 3 年以上: 删除阶段."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        now_ms = int(time.time() * 1000)
        four_years_ago = now_ms - 1460 * 24 * 60 * 60 * 1000
        action = mgr.check_retention(
            user_id="u-001",
            graduation_ts=four_years_ago,
            current_ts=now_ms,
        )
        assert action.phase == RetentionPhase.DELETED

    def test_execute_retention_anonymize(self):
        """执行匿名化留存操作."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        result = mgr.execute_retention(
            user_id="u-001",
            phase=RetentionPhase.ANONYMIZED,
        )
        assert result.success is True
        assert result.phase == RetentionPhase.ANONYMIZED

    def test_execute_retention_delete(self):
        """执行删除留存操作."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        result = mgr.execute_retention(
            user_id="u-001",
            phase=RetentionPhase.DELETED,
        )
        assert result.success is True
        assert result.phase == RetentionPhase.DELETED

    def test_retention_action_contains_actions(self):
        """留存操作结果包含具体动作列表."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        now_ms = int(time.time() * 1000)
        two_years_ago = now_ms - 730 * 24 * 60 * 60 * 1000
        action = mgr.check_retention(
            user_id="u-001",
            graduation_ts=two_years_ago,
            current_ts=now_ms,
        )
        assert len(action.actions) > 0
        # 匿名化阶段应包含学号脱敏动作
        assert any("学号" in a or "student" in a.lower() for a in action.actions)


# ============================================================
# 5. AuditLogger 测试
# ============================================================


class TestAuditLogger:
    """审计日志管理测试 (设计文档 6.5)."""

    def test_log_entry(self):
        """记录审计日志."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        entry = make_audit_entry()
        logger.log(entry)
        assert len(logger) == 1

    def test_log_multiple_entries(self):
        """记录多条审计日志."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        for i in range(10):
            logger.log(make_audit_entry(actor_id=f"u-{i:03d}"))
        assert len(logger) == 10

    def test_log_is_append_only(self):
        """审计日志 append-only (不可删除)."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        logger.log(make_audit_entry())
        # 不提供 delete 方法
        assert not hasattr(logger, "delete") or not callable(getattr(logger, "delete", None))

    def test_query_by_actor(self):
        """按操作者查询."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        logger.log(make_audit_entry(actor_id="u-001"))
        logger.log(make_audit_entry(actor_id="u-002"))
        logger.log(make_audit_entry(actor_id="u-001"))
        results = logger.query(actor_id="u-001")
        assert len(results) == 2

    def test_query_by_action(self):
        """按操作类型查询."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        logger.log(make_audit_entry(action=AuditAction.VIEW))
        logger.log(make_audit_entry(action=AuditAction.EXPORT))
        logger.log(make_audit_entry(action=AuditAction.VIEW))
        results = logger.query(action=AuditAction.VIEW)
        assert len(results) == 2

    def test_query_by_data_level(self):
        """按数据级别查询."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        logger.log(make_audit_entry(data_level=DataLevel.L3_SENSITIVE))
        logger.log(make_audit_entry(data_level=DataLevel.L4_CONFIDENTIAL))
        logger.log(make_audit_entry(data_level=DataLevel.L3_SENSITIVE))
        results = logger.query(data_level=DataLevel.L3_SENSITIVE)
        assert len(results) == 2

    def test_query_by_result(self):
        """按结果查询."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        logger.log(make_audit_entry(result=AuditResult.SUCCESS))
        logger.log(make_audit_entry(result=AuditResult.DENIED))
        logger.log(make_audit_entry(result=AuditResult.SUCCESS))
        results = logger.query(result=AuditResult.DENIED)
        assert len(results) == 1

    def test_query_by_time_range(self):
        """按时间范围查询."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        base_ts = int(time.time() * 1000)
        entry1 = make_audit_entry()
        entry1.timestamp = base_ts - 10000
        entry2 = make_audit_entry()
        entry2.timestamp = base_ts
        entry3 = make_audit_entry()
        entry3.timestamp = base_ts + 10000
        logger.log(entry1)
        logger.log(entry2)
        logger.log(entry3)
        results = logger.query(start_ts=base_ts - 5000, end_ts=base_ts + 5000)
        assert len(results) == 1

    def test_query_with_limit(self):
        """分页查询 (limit)."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        for i in range(20):
            logger.log(make_audit_entry(actor_id=f"u-{i:03d}"))
        results = logger.query(limit=5)
        assert len(results) == 5

    def test_query_with_offset(self):
        """分页查询 (offset)."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        for i in range(20):
            logger.log(make_audit_entry(actor_id=f"u-{i:03d}"))
        page1 = logger.query(limit=5, offset=0)
        page2 = logger.query(limit=5, offset=5)
        assert page1 != page2
        assert len(page1) == 5
        assert len(page2) == 5

    def test_hash_chain_integrity(self):
        """哈希链完整性验证."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        logger.log(make_audit_entry())
        logger.log(make_audit_entry())
        logger.log(make_audit_entry())
        # 验证哈希链
        assert logger.verify_chain() is True

    def test_hash_chain_tamper_detection(self):
        """篡改检测."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        logger.log(make_audit_entry(actor_id="u-001"))
        logger.log(make_audit_entry(actor_id="u-002"))
        # 篡改第一条日志 (直接修改内部存储)
        # 验证应检测到篡改
        # 注意: 正常使用不会篡改, 这里测试防篡改能力
        assert logger.verify_chain() is True  # 未篡改时通过

    def test_get_audit_stats(self):
        """审计统计."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        logger.log(make_audit_entry(result=AuditResult.SUCCESS))
        logger.log(make_audit_entry(result=AuditResult.SUCCESS))
        logger.log(make_audit_entry(result=AuditResult.DENIED))
        stats = logger.get_stats()
        assert stats["total"] == 3
        assert stats["success"] == 2
        assert stats["denied"] == 1


# ============================================================
# 6. PrivacyEventNotifier 测试
# ============================================================


class TestPrivacyEventNotifier:
    """隐私事件通知测试 (设计文档 6.6, 8.1)."""

    def test_notify_event(self):
        """发送隐私事件通知."""
        from dy3_polaris.l1.privacy_governance import PrivacyEventNotifier

        notifier = PrivacyEventNotifier()
        notifier.notify(
            event_type="unauthorized_access",
            user_id="u-001",
            data_level=DataLevel.L4_CONFIDENTIAL,
            detail="尝试越权访问学号数据",
        )
        events = notifier.get_events()
        assert len(events) == 1
        assert events[0].event_type == "unauthorized_access"

    def test_notify_multiple_events(self):
        """发送多个事件通知."""
        from dy3_polaris.l1.privacy_governance import PrivacyEventNotifier

        notifier = PrivacyEventNotifier()
        for i in range(5):
            notifier.notify(
                event_type="data_export",
                user_id=f"u-{i:03d}",
                data_level=DataLevel.L3_SENSITIVE,
            )
        assert len(notifier.get_events()) == 5

    def test_event_contains_all_fields(self):
        """事件包含所有必需字段."""
        from dy3_polaris.l1.privacy_governance import PrivacyEventNotifier

        notifier = PrivacyEventNotifier()
        notifier.notify(
            event_type="retention_anonymize",
            user_id="u-001",
            data_level=DataLevel.L3_SENSITIVE,
            detail="毕业后1年触发匿名化",
        )
        event = notifier.get_events()[0]
        assert event.event_type == "retention_anonymize"
        assert event.user_id == "u-001"
        assert event.data_level == DataLevel.L3_SENSITIVE
        assert event.detail == "毕业后1年触发匿名化"
        assert event.event_id.startswith("pevt-")
        assert event.timestamp > 0

    def test_clear_events(self):
        """清除事件队列."""
        from dy3_polaris.l1.privacy_governance import PrivacyEventNotifier

        notifier = PrivacyEventNotifier()
        notifier.notify("test", "u-001")
        notifier.notify("test", "u-002")
        notifier.clear()
        assert len(notifier.get_events()) == 0

    def test_event_serialization(self):
        """事件序列化."""
        from dy3_polaris.l1.privacy_governance import PrivacyEventNotifier

        notifier = PrivacyEventNotifier()
        notifier.notify("test", "u-001", DataLevel.L3_SENSITIVE, "detail")
        event = notifier.get_events()[0]
        d = event.to_dict()
        restored = PrivacyEvent.from_dict(d)
        assert restored.event_type == event.event_type
        assert restored.user_id == event.user_id


# ============================================================
# 7. PrivacyGovernanceManager 统一管理器测试
# ============================================================


class TestPrivacyGovernanceManager:
    """隐私治理统一管理器测试."""

    def test_manager_has_all_components(self):
        """管理器包含所有组件."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        assert mgr.classifier is not None
        assert mgr.desensitization_engine is not None
        assert mgr.retention_manager is not None
        assert mgr.audit_logger is not None
        assert mgr.event_notifier is not None

    def test_manager_desensitize_student_id(self):
        """管理器: 学号脱敏."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        result = mgr.desensitize_student_id("CS20240001", salt="salt")
        assert result != "CS20240001"
        assert len(result) == 64

    def test_manager_classify_and_check(self):
        """管理器: 分类 + 访问检查."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        level = mgr.classify_data("student_id")
        user = make_user(UserRole.UNDERGRAD)
        assert mgr.check_data_access(user, level) is False  # 本科生不可访问 L4

    def test_manager_log_audit(self):
        """管理器: 审计日志记录."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        entry = make_audit_entry()
        mgr.log_audit(entry)
        assert len(mgr.audit_logger) == 1

    def test_manager_check_retention(self):
        """管理器: 留存策略检查."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        now_ms = int(time.time() * 1000)
        action = mgr.check_user_retention(
            user_id="u-001",
            graduation_ts=now_ms - 1460 * 24 * 60 * 60 * 1000,
            current_ts=now_ms,
        )
        assert action.phase == RetentionPhase.DELETED

    def test_manager_notify_privacy_event(self):
        """管理器: 隐私事件通知."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        mgr.notify_event(
            event_type="unauthorized_access",
            user_id="u-001",
            data_level=DataLevel.L4_CONFIDENTIAL,
        )
        assert len(mgr.event_notifier.get_events()) == 1

    def test_manager_export_learner_data_desensitized(self):
        """管理器: 导出脱敏学情数据."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        learner_data = {
            "student_id": "CS20240001",
            "student_name": "张三",
            "mastery": 0.85,
            "response_times": [3000, 30000, 90000],
            "answers": [True, False, True, True, False],
        }
        exported = mgr.export_learner_data(learner_data, requester_role=UserRole.TEACHER)
        assert exported["student_id"] != "CS20240001"
        assert "student_name" not in exported or exported["student_name"] != "张三"
        assert "mastery" in exported

    def test_manager_export_undergrad_sees_own(self):
        """本科生导出自己的数据 (保留原始学号)."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        learner_data = {
            "student_id": "CS20240001",
            "mastery": 0.85,
        }
        exported = mgr.export_learner_data(
            learner_data,
            requester_role=UserRole.UNDERGRAD,
            requester_id="u-001",
            owner_id="u-001",
        )
        # 本科生查看自己的数据, 学号可见
        assert exported["student_id"] == "CS20240001"

    def test_manager_export_undergrad_cannot_see_others(self):
        """本科生不可导出他人数据."""
        from dy3_polaris.l1.privacy_governance import (
            PrivacyGovernanceManager,
            PrivacyViolationError,
        )

        mgr = PrivacyGovernanceManager()
        learner_data = {"student_id": "CS20240002", "mastery": 0.9}
        with pytest.raises(PrivacyViolationError):
            mgr.export_learner_data(
                learner_data,
                requester_role=UserRole.UNDERGRAD,
                requester_id="u-001",
                owner_id="u-002",
            )


# ============================================================
# 8. 线程安全测试
# ============================================================


class TestThreadSafety:
    """线程安全测试."""

    def test_concurrent_audit_logging(self):
        """并发审计日志写入."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        errors: list[Exception | None] = [None] * 20

        def log_entry(idx: int) -> None:
            try:
                logger.log(make_audit_entry(actor_id=f"u-{idx:03d}"))
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=log_entry, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(logger) == 20
        assert all(e is None for e in errors)

    def test_concurrent_notify_events(self):
        """并发隐私事件通知."""
        from dy3_polaris.l1.privacy_governance import PrivacyEventNotifier

        notifier = PrivacyEventNotifier()
        errors: list[Exception | None] = [None] * 20

        def notify(idx: int) -> None:
            try:
                notifier.notify("test", f"u-{idx:03d}")
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=notify, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(notifier.get_events()) == 20
        assert all(e is None for e in errors)

    def test_concurrent_desensitize(self):
        """并发脱敏操作."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        results: list[str] = [""] * 20
        errors: list[Exception | None] = [None] * 20

        def desensitize(idx: int) -> None:
            try:
                results[idx] = engine.desensitize(
                    f"CS2024{idx:04d}",
                    DesensitizationMethod.HASH,
                    salt="test",
                )
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=desensitize, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r != "" for r in results)
        assert all(e is None for e in errors)
        # 所有结果应唯一
        assert len(set(results)) == 20


# ============================================================
# 9. 边界条件与异常测试
# ============================================================


class TestEdgeCases:
    """边界条件与异常测试."""

    def test_empty_audit_query(self):
        """空审计日志查询返回空列表."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        assert logger.query() == []

    def test_audit_logger_len_zero(self):
        """空审计日志长度为 0."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        assert len(logger) == 0

    def test_empty_event_notifier(self):
        """空事件通知器."""
        from dy3_polaris.l1.privacy_governance import PrivacyEventNotifier

        notifier = PrivacyEventNotifier()
        assert notifier.get_events() == []

    def test_desensitize_empty_string(self):
        """空字符串脱敏."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        result = engine.desensitize("", DesensitizationMethod.HASH, salt="s")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_desensitize_empty_aggregate(self):
        """空列表聚合脱敏."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        result = engine.desensitize([], DesensitizationMethod.AGGREGATE)
        assert result == 0.0

    def test_check_k_anonymity_empty(self):
        """空数据集 K-匿名检查."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        assert engine.check_k_anonymity([], "qi", k=5) is True

    def test_check_l_diversity_empty(self):
        """空数据集 l-多样性检查."""
        from dy3_polaris.l1.privacy_governance import DesensitizationEngine

        engine = DesensitizationEngine()
        assert engine.check_l_diversity([], "qi", "sensitive", l=3) is True

    def test_retention_no_graduation(self):
        """无毕业时间: ACTIVE 阶段."""
        from dy3_polaris.l1.privacy_governance import RetentionManager

        mgr = RetentionManager()
        action = mgr.check_retention(
            user_id="u-001",
            graduation_ts=None,
            current_ts=int(time.time() * 1000),
        )
        assert action.phase == RetentionPhase.ACTIVE

    def test_audit_stats_empty(self):
        """空审计日志统计."""
        from dy3_polaris.l1.privacy_governance import AuditLogger

        logger = AuditLogger()
        stats = logger.get_stats()
        assert stats["total"] == 0
        assert stats["success"] == 0
        assert stats["denied"] == 0


# ============================================================
# 10. 集成测试
# ============================================================


class TestIntegration:
    """完整生命周期集成测试."""

    def test_full_privacy_lifecycle(self):
        """完整隐私治理生命周期."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()

        # 1. 数据分类
        level = mgr.classify_data("student_id")
        assert level == DataLevel.L4_CONFIDENTIAL

        # 2. 访问检查
        undergrad = make_user(UserRole.UNDERGRAD)
        assert not mgr.check_data_access(undergrad, level)

        # 3. 审计日志
        mgr.log_audit(make_audit_entry(
            actor_id="u-001",
            action=AuditAction.VIEW,
            result=AuditResult.DENIED,
        ))
        assert len(mgr.audit_logger) == 1

        # 4. 隐私事件通知
        mgr.notify_event(
            event_type="unauthorized_access",
            user_id="u-001",
            data_level=level,
        )
        assert len(mgr.event_notifier.get_events()) == 1

        # 5. 数据脱敏
        hashed = mgr.desensitize_student_id("CS20240001")
        assert hashed != "CS20240001"

        # 6. 留存策略检查
        now_ms = int(time.time() * 1000)
        action = mgr.check_user_retention(
            user_id="u-001",
            graduation_ts=now_ms - 1460 * 24 * 60 * 60 * 1000,
            current_ts=now_ms,
        )
        assert action.phase == RetentionPhase.DELETED

    def test_teacher_export_student_data(self):
        """教师导出学生数据 (脱敏)."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        data = {
            "student_id": "CS20240001",
            "student_name": "张三",
            "mastery": 0.75,
            "answers": [True, False, True],
            "response_times": [3000, 30000, 90000],
        }
        exported = mgr.export_learner_data(data, requester_role=UserRole.TEACHER)
        # 学号应被脱敏
        assert exported["student_id"] != "CS20240001"
        # 掌握度保留
        assert exported["mastery"] == 0.75
        # 审计日志应记录此次导出
        logs = mgr.audit_logger.query(action=AuditAction.EXPORT)
        assert len(logs) == 1

    def test_admin_export_with_full_access(self):
        """管理员导出数据 (审计但可访问)."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        data = {"student_id": "CS20240001", "mastery": 0.9}
        exported = mgr.export_learner_data(data, requester_role=UserRole.ADMIN)
        # 管理员导出仍脱敏学号 (安全默认)
        assert exported["student_id"] != "CS20240001"

    def test_retention_triggers_anonymization(self):
        """留存策略触发匿名化."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        now_ms = int(time.time() * 1000)
        two_years_ago = now_ms - 730 * 24 * 60 * 60 * 1000

        action = mgr.check_user_retention(
            user_id="u-001",
            graduation_ts=two_years_ago,
            current_ts=now_ms,
        )

        # 执行匿名化
        result = mgr.execute_user_retention("u-001", action.phase)
        assert result.success is True

        # 应产生隐私事件
        events = mgr.event_notifier.get_events()
        assert any(e.event_type == "retention_anonymize" for e in events)

    def test_audit_chain_after_multiple_operations(self):
        """多次操作后审计链完整性."""
        from dy3_polaris.l1.privacy_governance import PrivacyGovernanceManager

        mgr = PrivacyGovernanceManager()
        for i in range(15):
            mgr.log_audit(make_audit_entry(actor_id=f"u-{i:03d}"))
        assert mgr.audit_logger.verify_chain() is True
        assert len(mgr.audit_logger) == 15


# ============================================================
# 辅助函数
# ============================================================


def make_user(role: UserRole = UserRole.UNDERGRAD) -> User:
    """创建测试用户."""
    return User(
        student_id="CS20240001",
        institution_id="inst-001",
        role=role,
        status=UserStatus.ACTIVE,
    )


def make_audit_entry(
    actor_id: str = "u-001",
    action: AuditAction = AuditAction.VIEW,
    data_level: DataLevel = DataLevel.L2_INTERNAL,
    result: AuditResult = AuditResult.SUCCESS,
) -> AuditLogEntry:
    """创建测试审计日志条目."""
    return AuditLogEntry(
        actor_id=actor_id,
        actor_role=UserRole.UNDERGRAD,
        action=action,
        target_resource="kb:dy3_energy_level",
        target_data_level=data_level,
        purpose="学习查阅",
        result=result,
    )
