"""T3 学习上下文经纪 — 测试套件.

遵循 TDD Red-Green-Refactor:
1. 先写测试 (RED): 每个测试描述期望行为
2. 验证测试失败 (feature missing)
3. 最小实现 (GREEN)
4. 重构 (保持绿色)

测试覆盖:
- ContextCollector: 三渠道采集 (前端埋点/Agent输出/用户声明)
- DecayEngine: Ebbinghaus 遗忘衰减计算
- ContextCache: TTL 缓存分层
- LearningContextBroker: 核心引擎 (构建/获取/更新/刷新/传递)
- 异常体系: JSON-RPC 错误码
- 线程安全: 并发访问
- 边界情况: 空值/过期/非法输入
- 集成测试: 全生命周期
"""

from __future__ import annotations

import threading
import time

import pytest

from dy3_polaris.l1.models import (
    BKTParams,
    ContextEnvelope,
    LearningGoal,
    LearningPhase,
    LearningState,
    MasterySnapshot,
    ResourceItem,
    TimeConstraint,
    calculate_decay,
)
from dy3_polaris.l1.context_broker import (
    # 异常
    ContextExpiredError,
    ContextNotFoundError,
    ContextValidationError,
    DecayError,
    L1ContextError,
    # 采集
    AgentOutputEvent,
    ContextCollector,
    FrontendEvent,
    UserDeclaration,
    # 衰减
    DecayEngine,
    # 缓存
    ContextCache,
    # 核心引擎
    LearningContextBroker,
    # 常量
    CACHE_TTL_SESSION,
    CACHE_TTL_MASTERY,
    CACHE_TTL_GOAL,
    CACHE_TTL_COGNITIVE,
    REVIEW_URGENCY_THRESHOLD,
    COGNITIVE_LOAD_BASE,
    COGNITIVE_LOAD_FAST_ANSWER_MS,
    COGNITIVE_LOAD_ERROR_WEIGHT,
    COGNITIVE_LOAD_HELP_WEIGHT,
)


# ============================================================
# 辅助函数
# ============================================================


def make_mastery(
    kc_id: str = "kc-test-001",
    p_know: float = 0.7,
    last_practiced_offset_ms: int = 0,
    repetitions: int = 3,
    correct: int = 2,
    attempts: int = 3,
) -> MasterySnapshot:
    """创建 MasterySnapshot 测试辅助."""
    now_ms = int(time.time() * 1000)
    return MasterySnapshot(
        kc_id=kc_id,
        p_know=p_know,
        last_practiced_at=now_ms - last_practiced_offset_ms,
        repetitions=repetitions,
        correct_count=correct,
        attempts=attempts,
    )


def make_envelope(
    user_id: str = "user-test-001",
    session_id: str = "sess-test-001",
    mastery_list: list[MasterySnapshot] | None = None,
    goals: list[LearningGoal] | None = None,
    ttl: int = 3600,
) -> ContextEnvelope:
    """创建 ContextEnvelope 测试辅助."""
    return ContextEnvelope(
        user_id=user_id,
        session_id=session_id,
        mastery_snapshot=mastery_list or [make_mastery()],
        goals=goals or [LearningGoal(description="掌握测试知识")],
        ttl=ttl,
    )


# ============================================================
# 1. 异常体系测试
# ============================================================


class TestExceptionHierarchy:
    """异常继承与 JSON-RPC 错误码测试."""

    def test_base_error_inherits_l6(self):
        """L1ContextError 继承 L6Error."""
        from dy3_polaris.l6.core.exceptions import L6Error
        assert issubclass(L1ContextError, L6Error)

    def test_base_error_jsonrpc_code(self):
        """L1ContextError JSON-RPC 码为 -32300."""
        err = L1ContextError("test")
        assert err._jsonrpc_code() == -32300

    def test_not_found_inherits_base(self):
        """ContextNotFoundError 继承 L1ContextError."""
        assert issubclass(ContextNotFoundError, L1ContextError)

    def test_not_found_jsonrpc_code(self):
        """ContextNotFoundError JSON-RPC 码为 -32301."""
        err = ContextNotFoundError("sess-xxx")
        assert err._jsonrpc_code() == -32301

    def test_not_found_contains_session_id(self):
        """异常包含 session_id 上下文."""
        err = ContextNotFoundError("sess-123")
        assert err.context.get("session_id") == "sess-123"

    def test_expired_inherits_base(self):
        """ContextExpiredError 继承 L1ContextError."""
        assert issubclass(ContextExpiredError, L1ContextError)

    def test_expired_jsonrpc_code(self):
        """ContextExpiredError JSON-RPC 码为 -32302."""
        err = ContextExpiredError("sess-xxx", expired_at=123)
        assert err._jsonrpc_code() == -32302

    def test_validation_inherits_base(self):
        """ContextValidationError 继承 L1ContextError."""
        assert issubclass(ContextValidationError, L1ContextError)

    def test_validation_jsonrpc_code(self):
        """ContextValidationError JSON-RPC 码为 -32303."""
        err = ContextValidationError("invalid")
        assert err._jsonrpc_code() == -32303

    def test_decay_inherits_base(self):
        """DecayError 继承 L1ContextError."""
        assert issubclass(DecayError, L1ContextError)

    def test_decay_jsonrpc_code(self):
        """DecayError JSON-RPC 码为 -32304."""
        err = DecayError("calculation failed")
        assert err._jsonrpc_code() == -32304


# ============================================================
# 2. ContextCollector 测试
# ============================================================


class TestFrontendEvent:
    """前端埋点事件测试."""

    def test_create_frontend_event(self):
        """创建前端埋点事件."""
        evt = FrontendEvent(
            event_type="page_view",
            actor_id="user-001",
            target_resource="page-dy3-intro",
            timestamp=int(time.time() * 1000),
            result={"duration_ms": 5000},
        )
        assert evt.event_type == "page_view"
        assert evt.actor_id == "user-001"

    def test_frontend_event_to_xapi_statement(self):
        """前端事件转换为 xAPI Statement (Actor-Verb-Object)."""
        evt = FrontendEvent(
            event_type="answer_submit",
            actor_id="user-001",
            target_resource="question-042",
            timestamp=int(time.time() * 1000),
            result={"is_correct": True, "response_time_ms": 3500},
        )
        stmt = evt.to_xapi_statement()
        assert stmt["actor"] == "user-001"
        assert stmt["verb"] == "answer_submit"
        assert stmt["object"] == "question-042"
        assert stmt["result"]["is_correct"] is True


class TestAgentOutputEvent:
    """Agent 输出事件测试."""

    def test_create_agent_output(self):
        """创建 Agent 输出事件."""
        evt = AgentOutputEvent(
            agent_id="agent-diagnosis-001",
            output_type="bkt_update",
            kc_id="kc-dy3-energy",
            p_know=0.82,
            is_correct=True,
            timestamp=int(time.time() * 1000),
        )
        assert evt.agent_id == "agent-diagnosis-001"
        assert evt.p_know == 0.82

    def test_agent_output_to_xapi_statement(self):
        """Agent 输出转换为 xAPI Statement."""
        evt = AgentOutputEvent(
            agent_id="agent-001",
            output_type="bkt_update",
            kc_id="kc-test",
            p_know=0.85,
            is_correct=True,
            timestamp=int(time.time() * 1000),
        )
        stmt = evt.to_xapi_statement()
        assert stmt["verb"] == "bkt_update"
        assert stmt["object"] == "kc-test"
        assert stmt["result"]["p_know"] == 0.85


class TestUserDeclaration:
    """用户显式声明测试."""

    def test_create_user_declaration(self):
        """创建用户声明."""
        decl = UserDeclaration(
            user_id="user-001",
            available_minutes=30,
            preferred_phase=LearningPhase.PRACTICE,
            confusion_points=["不理解能级简并"],
            timestamp=int(time.time() * 1000),
        )
        assert decl.available_minutes == 30
        assert decl.preferred_phase == LearningPhase.PRACTICE

    def test_user_declaration_to_xapi_statement(self):
        """用户声明转换为 xAPI Statement."""
        decl = UserDeclaration(
            user_id="user-001",
            available_minutes=45,
            preferred_phase=LearningPhase.QUIZ,
            confusion_points=["概念A"],
            timestamp=int(time.time() * 1000),
        )
        stmt = decl.to_xapi_statement()
        assert stmt["actor"] == "user-001"
        assert stmt["verb"] == "declared"
        assert stmt["result"]["available_minutes"] == 45


class TestContextCollector:
    """上下文采集器测试."""

    def test_collect_frontend_event(self):
        """采集前端事件."""
        collector = ContextCollector()
        evt = FrontendEvent(
            event_type="page_view",
            actor_id="user-001",
            target_resource="page-001",
            timestamp=int(time.time() * 1000),
        )
        collected = collector.collect_frontend_event(evt)
        assert len(collected) == 1
        assert collected[0].event_type == "page_view"

    def test_collect_agent_output(self):
        """采集 Agent 输出."""
        collector = ContextCollector()
        evt = AgentOutputEvent(
            agent_id="agent-001",
            output_type="bkt_update",
            kc_id="kc-test",
            p_know=0.8,
            is_correct=True,
            timestamp=int(time.time() * 1000),
        )
        collected = collector.collect_agent_output(evt)
        assert len(collected) == 1

    def test_collect_user_declaration(self):
        """采集用户声明."""
        collector = ContextCollector()
        decl = UserDeclaration(
            user_id="user-001",
            available_minutes=30,
            preferred_phase=LearningPhase.PRACTICE,
            timestamp=int(time.time() * 1000),
        )
        collected = collector.collect_user_declaration(decl)
        assert len(collected) == 1

    def test_collect_multiple_events(self):
        """多事件采集累积."""
        collector = ContextCollector()
        for i in range(5):
            evt = FrontendEvent(
                event_type="page_view",
                actor_id="user-001",
                target_resource=f"page-{i}",
                timestamp=int(time.time() * 1000),
            )
            collector.collect_frontend_event(evt)
        all_events = collector.get_all_events()
        assert len(all_events) == 5

    def test_privacy_filter_blocks_mouse_tracking(self):
        """隐私过滤: 不采集鼠标轨迹."""
        collector = ContextCollector()
        evt = FrontendEvent(
            event_type="mouse_move",
            actor_id="user-001",
            target_resource="page-001",
            timestamp=int(time.time() * 1000),
        )
        collected = collector.collect_frontend_event(evt)
        assert len(collected) == 0  # 被过滤

    def test_privacy_filter_blocks_heatmap(self):
        """隐私过滤: 不采集热力图."""
        collector = ContextCollector()
        evt = FrontendEvent(
            event_type="heatmap_click",
            actor_id="user-001",
            target_resource="page-001",
            timestamp=int(time.time() * 1000),
        )
        collected = collector.collect_frontend_event(evt)
        assert len(collected) == 0

    def test_privacy_filter_blocks_cross_domain(self):
        """隐私过滤: 不采集跨域数据."""
        collector = ContextCollector()
        evt = FrontendEvent(
            event_type="cross_domain_request",
            actor_id="user-001",
            target_resource="external-site.com",
            timestamp=int(time.time() * 1000),
        )
        collected = collector.collect_frontend_event(evt)
        assert len(collected) == 0

    def test_clear_events(self):
        """清空采集缓冲."""
        collector = ContextCollector()
        evt = FrontendEvent(
            event_type="page_view",
            actor_id="user-001",
            target_resource="page-001",
            timestamp=int(time.time() * 1000),
        )
        collector.collect_frontend_event(evt)
        collector.clear()
        assert len(collector.get_all_events()) == 0

    def test_get_events_by_type(self):
        """按事件类型查询."""
        collector = ContextCollector()
        for etype in ["page_view", "answer_submit", "page_view"]:
            evt = FrontendEvent(
                event_type=etype,
                actor_id="user-001",
                target_resource="page-001",
                timestamp=int(time.time() * 1000),
            )
            collector.collect_frontend_event(evt)
        views = collector.get_events_by_type("page_view")
        assert len(views) == 2


# ============================================================
# 3. DecayEngine 测试
# ============================================================


class TestDecayEngine:
    """遗忘衰减引擎测试."""

    def test_calculate_decay_no_time_elapsed(self):
        """无时间流逝时衰减为 1.0 (无衰减)."""
        now = int(time.time() * 1000)
        decay = DecayEngine.calculate_decay(
            p_know=0.8,
            last_practiced=now,
            repetitions=3,
            current_ts=now,
        )
        assert decay == pytest.approx(0.8, abs=0.01)

    def test_calculate_decay_with_time(self):
        """有时间流逝时掌握度下降."""
        now = int(time.time() * 1000)
        one_day_ago = now - 24 * 3_600_000  # 24 小时前
        decayed = DecayEngine.calculate_decay(
            p_know=0.9,
            last_practiced=one_day_ago,
            repetitions=1,
            current_ts=now,
        )
        assert decayed < 0.9  # 衰减后低于原始
        assert decayed >= 0.3  # 不低于先验概率

    def test_calculate_decay_more_repetitions_slower_decay(self):
        """更多练习次数 → 衰减更慢."""
        now = int(time.time() * 1000)
        one_week_ago = now - 7 * 24 * 3_600_000
        low_reps = DecayEngine.calculate_decay(
            p_know=0.8, last_practiced=one_week_ago, repetitions=0, current_ts=now
        )
        high_reps = DecayEngine.calculate_decay(
            p_know=0.8, last_practiced=one_week_ago, repetitions=10, current_ts=now
        )
        assert high_reps > low_reps  # 练习多 → 衰减少 → 掌握度高

    def test_calculate_decay_floor_prior_prob(self):
        """衰减不低于先验概率 PRIOR_PROB (0.3)."""
        now = int(time.time() * 1000)
        long_ago = now - 365 * 24 * 3_600_000  # 1 年前
        decayed = DecayEngine.calculate_decay(
            p_know=0.5, last_practiced=long_ago, repetitions=0, current_ts=now
        )
        assert decayed >= 0.3  # 不低于先验

    def test_calculate_decay_zero_p_know(self):
        """p_know=0 时衰减为 0."""
        now = int(time.time() * 1000)
        decayed = DecayEngine.calculate_decay(
            p_know=0.0, last_practiced=now, repetitions=0, current_ts=now
        )
        assert decayed == 0.0

    def test_refresh_all_decay(self):
        """批量刷新信封中所有 KC 的衰减系数."""
        now = int(time.time() * 1000)
        envelope = make_envelope(
            mastery_list=[
                make_mastery(kc_id="kc-1", p_know=0.8, last_practiced_offset_ms=48 * 3_600_000, repetitions=1),
                make_mastery(kc_id="kc-2", p_know=0.6, last_practiced_offset_ms=2 * 3_600_000, repetitions=5),
            ]
        )
        DecayEngine.refresh_all_decay(envelope, current_ts=now)
        # kc-1 距上次 48h, reps=1, 衰减应该明显
        assert envelope.mastery_snapshot[0].decay_factor < 1.0
        # kc-2 距上次 2h, reps=5, 衰减应该较少
        assert envelope.mastery_snapshot[1].decay_factor < 1.0
        assert envelope.mastery_snapshot[1].decay_factor > envelope.mastery_snapshot[0].decay_factor

    def test_get_review_urgency(self):
        """获取复习紧急度排序."""
        now = int(time.time() * 1000)
        envelope = make_envelope(
            mastery_list=[
                make_mastery(kc_id="kc-urgent", p_know=0.4, last_practiced_offset_ms=72 * 3_600_000, repetitions=0),
                make_mastery(kc_id="kc-okay", p_know=0.9, last_practiced_offset_ms=1 * 3_600_000, repetitions=8),
            ]
        )
        urgency = DecayEngine.get_review_urgency(envelope, current_ts=now)
        assert len(urgency) > 0
        # 紧急度最高的应该是 kc-urgent
        assert urgency[0]["kc_id"] == "kc-urgent"

    def test_get_review_urgency_threshold_filter(self):
        """复习紧急度阈值过滤."""
        now = int(time.time() * 1000)
        envelope = make_envelope(
            mastery_list=[
                make_mastery(kc_id="kc-weak", p_know=0.3, last_practiced_offset_ms=48 * 3_600_000, repetitions=0),
                make_mastery(kc_id="kc-strong", p_know=0.95, last_practiced_offset_ms=1 * 3_600_000, repetitions=10),
            ]
        )
        urgency = DecayEngine.get_review_urgency(
            envelope, current_ts=now, threshold=REVIEW_URGENCY_THRESHOLD
        )
        # 只有 kc-weak 的有效掌握度低于阈值
        kc_ids = [u["kc_id"] for u in urgency]
        assert "kc-weak" in kc_ids
        assert "kc-strong" not in kc_ids


# ============================================================
# 4. ContextCache 测试
# ============================================================


class TestContextCache:
    """上下文缓存测试."""

    def test_set_and_get(self):
        """设置并获取上下文."""
        cache = ContextCache()
        envelope = make_envelope(session_id="sess-001")
        cache.set("sess-001", envelope)
        result = cache.get("sess-001")
        assert result is not None
        assert result.session_id == "sess-001"

    def test_get_nonexistent(self):
        """获取不存在的上下文返回 None."""
        cache = ContextCache()
        assert cache.get("sess-xxx") is None

    def test_invalidate(self):
        """失效缓存."""
        cache = ContextCache()
        envelope = make_envelope()
        cache.set("sess-001", envelope)
        cache.invalidate("sess-001")
        assert cache.get("sess-001") is None

    def test_ttl_expiration(self):
        """TTL 过期后获取返回 None."""
        cache = ContextCache()
        envelope = make_envelope(ttl=0)  # TTL=0 立即过期
        # 手动设置过期的时间戳
        envelope.timestamp = int(time.time() * 1000) - 10_000
        cache.set("sess-001", envelope, ttl=0)
        # TTL=0 意味着立即过期
        result = cache.get("sess-001")
        assert result is None

    def test_custom_ttl(self):
        """自定义 TTL."""
        cache = ContextCache()
        envelope = make_envelope()
        cache.set("sess-001", envelope, ttl=3600)
        result = cache.get("sess-001")
        assert result is not None

    def test_cache_stats(self):
        """缓存统计信息."""
        cache = ContextCache()
        cache.set("sess-001", make_envelope(session_id="sess-001"))
        cache.set("sess-002", make_envelope(session_id="sess-002"))
        stats = cache.get_stats()
        assert stats["total_entries"] == 2

    def test_clear_all(self):
        """清空所有缓存."""
        cache = ContextCache()
        cache.set("sess-001", make_envelope())
        cache.set("sess-002", make_envelope())
        cache.clear_all()
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

    def test_persistent_layer_backup(self):
        """持久层备份: 会话级缓存失效后可从持久层恢复."""
        cache = ContextCache()
        envelope = make_envelope(session_id="sess-001")
        cache.set("sess-001", envelope)
        # 备份到持久层
        cache.backup_to_persistent("sess-001")
        # 失效会话级缓存
        cache.invalidate("sess-001")
        assert cache.get("sess-001") is None
        # 从持久层恢复
        restored = cache.restore_from_persistent("sess-001")
        assert restored is not None
        assert restored.session_id == "sess-001"

    def test_ttl_constants_exist(self):
        """TTL 常量存在且合理."""
        assert CACHE_TTL_SESSION > 0
        assert CACHE_TTL_MASTERY > 0
        assert CACHE_TTL_GOAL > 0
        assert CACHE_TTL_COGNITIVE > 0
        # 掌握快照 TTL (1h) > 认知负荷 TTL (30min)
        assert CACHE_TTL_MASTERY > CACHE_TTL_COGNITIVE
        # 目标 TTL (24h) > 掌握快照 TTL (1h)
        assert CACHE_TTL_GOAL > CACHE_TTL_MASTERY


# ============================================================
# 5. LearningContextBroker 测试
# ============================================================


class TestLearningContextBroker:
    """学习上下文经纪核心引擎测试."""

    def test_build_envelope(self):
        """构建标准化上下文信封."""
        broker = LearningContextBroker()
        envelope = broker.build_envelope(
            user_id="user-001", session_id="sess-001"
        )
        assert envelope.user_id == "user-001"
        assert envelope.session_id == "sess-001"
        assert envelope.envelope_id.startswith("env-")
        assert envelope.ttl > 0

    def test_build_envelope_with_initial_mastery(self):
        """构建信封时传入初始掌握快照."""
        broker = LearningContextBroker()
        mastery = [make_mastery(kc_id="kc-init", p_know=0.5)]
        envelope = broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=mastery,
        )
        assert len(envelope.mastery_snapshot) == 1
        assert envelope.mastery_snapshot[0].kc_id == "kc-init"

    def test_build_envelope_with_goals(self):
        """构建信封时传入学习目标."""
        broker = LearningContextBroker()
        goals = [LearningGoal(description="掌握 Dy3+ 能级", priority=5)]
        envelope = broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_goals=goals,
        )
        assert len(envelope.goals) == 1
        assert envelope.goals[0].priority == 5

    def test_get_envelope_from_cache(self):
        """从缓存获取上下文."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        envelope = broker.get_envelope("sess-001")
        assert envelope is not None
        assert envelope.session_id == "sess-001"

    def test_get_envelope_nonexistent(self):
        """获取不存在的会话上下文抛异常."""
        broker = LearningContextBroker()
        with pytest.raises(ContextNotFoundError):
            broker.get_envelope("sess-xxx")

    def test_update_mastery_new_kc(self):
        """更新掌握度: 新增 KC."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        broker.update_mastery("sess-001", "kc-new", p_know=0.75, is_correct=True)
        envelope = broker.get_envelope("sess-001")
        kc_ids = [s.kc_id for s in envelope.mastery_snapshot]
        assert "kc-new" in kc_ids

    def test_update_mastery_existing_kc(self):
        """更新掌握度: 已有 KC 的 BKT 更新."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.5)],
        )
        # 答对 → p_know 应上升
        broker.update_mastery("sess-001", "kc-1", p_know=0.7, is_correct=True)
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        assert snap.p_know > 0.5  # BKT 更新后上升
        assert snap.repetitions == 4  # 3 + 1

    def test_update_mastery_incorrect(self):
        """答错时 p_know 下降."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.8)],
        )
        broker.update_mastery("sess-001", "kc-1", p_know=0.8, is_correct=False)
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        assert snap.p_know < 0.8  # BKT 更新后下降

    def test_update_mastery_session_not_found(self):
        """更新不存在会话的掌握度抛异常."""
        broker = LearningContextBroker()
        with pytest.raises(ContextNotFoundError):
            broker.update_mastery("sess-xxx", "kc-1", p_know=0.5, is_correct=True)

    def test_update_cognitive_load(self):
        """计算认知负荷."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        interactions = [
            {"response_time_ms": 3000, "is_correct": True, "asked_help": False},
            {"response_time_ms": 8000, "is_correct": False, "asked_help": True},
            {"response_time_ms": 5000, "is_correct": True, "asked_help": False},
        ]
        broker.update_cognitive_load("sess-001", interactions)
        envelope = broker.get_envelope("sess-001")
        # 认知负荷应该在合理范围 [0.0, 1.0]
        assert 0.0 <= envelope.learning_state.cognitive_load <= 1.0

    def test_update_cognitive_load_high_load(self):
        """高错误率 + 慢响应 + 求助 → 高认知负荷."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        interactions = [
            {"response_time_ms": 15000, "is_correct": False, "asked_help": True},
            {"response_time_ms": 12000, "is_correct": False, "asked_help": True},
            {"response_time_ms": 10000, "is_correct": False, "asked_help": True},
        ]
        broker.update_cognitive_load("sess-001", interactions)
        envelope = broker.get_envelope("sess-001")
        assert envelope.learning_state.cognitive_load > 0.7  # 高负荷

    def test_update_cognitive_load_low_load(self):
        """低错误率 + 快响应 → 低认知负荷."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        interactions = [
            {"response_time_ms": 2000, "is_correct": True, "asked_help": False},
            {"response_time_ms": 1500, "is_correct": True, "asked_help": False},
            {"response_time_ms": 2500, "is_correct": True, "asked_help": False},
        ]
        broker.update_cognitive_load("sess-001", interactions)
        envelope = broker.get_envelope("sess-001")
        assert envelope.learning_state.cognitive_load < 0.5  # 低负荷

    def test_update_learning_phase(self):
        """更新学习阶段."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        broker.update_learning_phase("sess-001", LearningPhase.QUIZ)
        envelope = broker.get_envelope("sess-001")
        assert envelope.learning_state.phase == LearningPhase.QUIZ

    def test_get_weak_kcs(self):
        """获取薄弱知识点."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[
                make_mastery(kc_id="kc-weak", p_know=0.2, repetitions=0),
                make_mastery(kc_id="kc-strong", p_know=0.9, repetitions=5),
            ],
        )
        weak = broker.get_weak_kcs("sess-001", threshold=0.5)
        assert "kc-weak" in weak
        assert "kc-strong" not in weak

    def test_refresh_context(self):
        """刷新上下文: 重算衰减系数."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[
                make_mastery(
                    kc_id="kc-1",
                    p_know=0.8,
                    last_practiced_offset_ms=48 * 3_600_000,
                    repetitions=1,
                ),
            ],
        )
        broker.refresh_context("sess-001")
        envelope = broker.get_envelope("sess-001")
        assert envelope.mastery_snapshot[0].decay_factor < 1.0

    def test_refresh_context_nonexistent(self):
        """刷新不存在的会话抛异常."""
        broker = LearningContextBroker()
        with pytest.raises(ContextNotFoundError):
            broker.refresh_context("sess-xxx")

    def test_transfer_context(self):
        """跨会话上下文传递."""
        broker = LearningContextBroker()
        # 源会话有掌握度和目标
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-source",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.7)],
            initial_goals=[LearningGoal(description="目标A", priority=5)],
        )
        # 传递到目标会话
        broker.transfer_context("sess-source", "sess-target", "user-001")
        target = broker.get_envelope("sess-target")
        assert target is not None
        # 知识掌握度应继承
        kc_ids = [s.kc_id for s in target.mastery_snapshot]
        assert "kc-1" in kc_ids
        # 未完成目标应继承
        assert len(target.goals) > 0

    def test_transfer_context_source_not_found(self):
        """源会话不存在时传递抛异常."""
        broker = LearningContextBroker()
        with pytest.raises(ContextNotFoundError):
            broker.transfer_context("sess-xxx", "sess-target", "user-001")

    def test_update_goals(self):
        """更新学习目标."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        new_goals = [
            LearningGoal(description="新目标A", priority=4),
            LearningGoal(description="新目标B", priority=2),
        ]
        broker.update_goals("sess-001", new_goals)
        envelope = broker.get_envelope("sess-001")
        assert len(envelope.goals) == 2

    def test_add_resource(self):
        """添加可用资源."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        resource = ResourceItem(
            resource_id="res-001",
            title="Dy3+ 能级跃迁讲义",
            resource_type="document",
            difficulty=0.6,
        )
        broker.add_resource("sess-001", resource)
        envelope = broker.get_envelope("sess-001")
        assert len(envelope.resources) == 1
        assert envelope.resources[0].resource_id == "res-001"

    def test_set_time_constraint(self):
        """设置时间约束."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        tc = TimeConstraint(available_minutes=30, recommended_phase=LearningPhase.PRACTICE)
        broker.set_time_constraint("sess-001", tc)
        envelope = broker.get_envelope("sess-001")
        assert envelope.time_constraint is not None
        assert envelope.time_constraint.available_minutes == 30

    def test_remove_session(self):
        """移除会话上下文."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        broker.remove_session("sess-001")
        with pytest.raises(ContextNotFoundError):
            broker.get_envelope("sess-001")

    def test_get_envelope_summary(self):
        """获取脱敏摘要."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[
                make_mastery(kc_id="kc-weak", p_know=0.2),
                make_mastery(kc_id="kc-okay", p_know=0.8),
            ],
        )
        summary = broker.get_envelope_summary("sess-001")
        assert "phase" in summary
        assert "cognitive_load" in summary
        assert "weak_kc_count" in summary
        assert summary["weak_kc_count"] == 1

    def test_get_all_sessions(self):
        """获取所有活跃会话 ID."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        broker.build_envelope(user_id="user-002", session_id="sess-002")
        sessions = broker.get_all_sessions()
        assert "sess-001" in sessions
        assert "sess-002" in sessions


# ============================================================
# 6. BKT 更新集成测试
# ============================================================


class TestBKTIntegration:
    """BKT 参数更新集成测试 (设计文档 8.2)."""

    def test_bkt_update_correct_answer(self):
        """答对 → BKT 后验 P(Know) 上升."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.5)],
        )
        broker.update_mastery("sess-001", "kc-1", p_know=0.5, is_correct=True)
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        assert snap.p_know > 0.5
        assert snap.bkt_params is not None
        assert snap.bkt_params.p_know == snap.p_know

    def test_bkt_update_incorrect_answer(self):
        """答错 → BKT 后验 P(Know) 下降."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.8)],
        )
        broker.update_mastery("sess-001", "kc-1", p_know=0.8, is_correct=False)
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        assert snap.p_know < 0.8

    def test_bkt_update_increments_repetitions(self):
        """每次更新增加练习次数."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.5, repetitions=3)],
        )
        broker.update_mastery("sess-001", "kc-1", p_know=0.5, is_correct=True)
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        assert snap.repetitions == 4

    def test_bkt_update_updates_last_practiced(self):
        """更新后 last_practiced_at 更新为当前时间."""
        broker = LearningContextBroker()
        old_ts = int(time.time() * 1000) - 48 * 3_600_000
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[MasterySnapshot(
                kc_id="kc-1", p_know=0.5, last_practiced_at=old_ts, repetitions=3,
            )],
        )
        time.sleep(0.01)
        broker.update_mastery("sess-001", "kc-1", p_know=0.5, is_correct=True)
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        assert snap.last_practiced_at > old_ts

    def test_bkt_update_correct_count(self):
        """答对增加 correct_count."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.5, correct=2, attempts=3)],
        )
        broker.update_mastery("sess-001", "kc-1", p_know=0.5, is_correct=True)
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        assert snap.correct_count == 3
        assert snap.attempts == 4

    def test_bkt_update_incorrect_no_correct_count(self):
        """答错不增加 correct_count 但增加 attempts."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.5, correct=2, attempts=3)],
        )
        broker.update_mastery("sess-001", "kc-1", p_know=0.5, is_correct=False)
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        assert snap.correct_count == 2
        assert snap.attempts == 4


# ============================================================
# 7. 线程安全测试
# ============================================================


class TestThreadSafety:
    """线程安全测试."""

    def test_concurrent_build_envelope(self):
        """并发构建信封不抛异常."""
        broker = LearningContextBroker()
        errors = []

        def worker(i):
            try:
                broker.build_envelope(
                    user_id=f"user-{i}", session_id=f"sess-{i}"
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(broker.get_all_sessions()) == 20

    def test_concurrent_update_mastery(self):
        """并发更新掌握度不抛异常."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery(kc_id="kc-1", p_know=0.5)],
        )
        errors = []

        def worker():
            try:
                for _ in range(50):
                    broker.update_mastery("sess-001", "kc-1", p_know=0.5, is_correct=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        envelope = broker.get_envelope("sess-001")
        snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-1")
        # 10 threads × 50 updates = 500 updates
        assert snap.attempts == 503  # 3 initial + 500

    def test_concurrent_cache_access(self):
        """并发缓存访问不抛异常."""
        cache = ContextCache()
        errors = []

        def worker(i):
            try:
                cache.set(f"sess-{i}", make_envelope(session_id=f"sess-{i}"))
                cache.get(f"sess-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.get_stats()["total_entries"] == 20


# ============================================================
# 8. 边界情况测试
# ============================================================


class TestEdgeCases:
    """边界情况测试."""

    def test_build_envelope_empty_user_id(self):
        """空 user_id 抛异常."""
        broker = LearningContextBroker()
        with pytest.raises(ContextValidationError):
            broker.build_envelope(user_id="", session_id="sess-001")

    def test_build_envelope_empty_session_id(self):
        """空 session_id 抛异常."""
        broker = LearningContextBroker()
        with pytest.raises(ContextValidationError):
            broker.build_envelope(user_id="user-001", session_id="")

    def test_update_mastery_invalid_p_know(self):
        """无效 p_know 抛异常."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        with pytest.raises(ValueError):
            broker.update_mastery("sess-001", "kc-1", p_know=1.5, is_correct=True)

    def test_update_cognitive_load_empty_interactions(self):
        """空交互列表不抛异常."""
        broker = LearningContextBroker()
        broker.build_envelope(user_id="user-001", session_id="sess-001")
        broker.update_cognitive_load("sess-001", [])
        envelope = broker.get_envelope("sess-001")
        # 空交互应保持默认值
        assert envelope.learning_state.cognitive_load == 0.5

    def test_get_weak_kcs_empty_mastery(self):
        """无掌握快照时返回空列表."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[],
        )
        weak = broker.get_weak_kcs("sess-001")
        assert weak == []

    def test_refresh_context_empty_mastery(self):
        """无掌握快照时刷新不抛异常."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[],
        )
        broker.refresh_context("sess-001")  # 不应抛异常

    def test_transfer_context_empty_source(self):
        """源会话无掌握快照时传递不抛异常."""
        broker = LearningContextBroker()
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-source",
            initial_mastery=[],
            initial_goals=[],
        )
        broker.transfer_context("sess-source", "sess-target", "user-001")
        target = broker.get_envelope("sess-target")
        assert target is not None

    def test_envelope_serialization_roundtrip(self):
        """信封序列化/反序列化一致性."""
        broker = LearningContextBroker()
        envelope = broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[make_mastery()],
            initial_goals=[LearningGoal(description="test", priority=3)],
        )
        d = envelope.to_dict()
        restored = ContextEnvelope.from_dict(d)
        assert restored.user_id == envelope.user_id
        assert restored.session_id == envelope.session_id
        assert len(restored.mastery_snapshot) == len(envelope.mastery_snapshot)


# ============================================================
# 9. 集成测试: 全生命周期
# ============================================================


class TestIntegrationLifecycle:
    """全生命周期集成测试."""

    def test_full_learning_session_lifecycle(self):
        """完整学习会话生命周期."""
        broker = LearningContextBroker()
        collector = ContextCollector()

        # 1. 构建初始上下文
        envelope = broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[
                make_mastery(kc_id="kc-dy3-energy", p_know=0.3, repetitions=0),
                make_mastery(kc_id="kc-dy3-transition", p_know=0.5, repetitions=1),
            ],
            initial_goals=[LearningGoal(description="掌握 Dy3+ 能级跃迁", priority=5)],
        )
        assert len(envelope.mastery_snapshot) == 2

        # 2. 采集前端事件
        for i in range(5):
            evt = FrontendEvent(
                event_type="answer_submit",
                actor_id="user-001",
                target_resource=f"q-{i}",
                timestamp=int(time.time() * 1000),
                result={"is_correct": i % 2 == 0, "response_time_ms": 3000 + i * 1000},
            )
            collector.collect_frontend_event(evt)

        # 3. 更新掌握度 (模拟答题)
        broker.update_mastery("sess-001", "kc-dy3-energy", p_know=0.3, is_correct=True)
        broker.update_mastery("sess-001", "kc-dy3-transition", p_know=0.5, is_correct=False)

        # 4. 更新认知负荷
        broker.update_cognitive_load("sess-001", [
            {"response_time_ms": 4000, "is_correct": True, "asked_help": False},
            {"response_time_ms": 9000, "is_correct": False, "asked_help": True},
        ])

        # 5. 更新学习阶段
        broker.update_learning_phase("sess-001", LearningPhase.QUIZ)

        # 6. 刷新上下文
        broker.refresh_context("sess-001")

        # 7. 获取薄弱知识点
        weak = broker.get_weak_kcs("sess-001", threshold=0.5)

        # 8. 验证最终状态
        envelope = broker.get_envelope("sess-001")
        assert envelope.learning_state.phase == LearningPhase.QUIZ
        assert len(envelope.mastery_snapshot) == 2
        # kc-dy3-energy 答对了, p_know 应上升
        energy_snap = next(s for s in envelope.mastery_snapshot if s.kc_id == "kc-dy3-energy")
        assert energy_snap.p_know > 0.3

    def test_cross_session_inheritance(self):
        """跨会话上下文继承."""
        broker = LearningContextBroker()

        # 会话 1: 学习并积累掌握度
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-1",
            initial_mastery=[
                make_mastery(kc_id="kc-1", p_know=0.4, repetitions=1),
                make_mastery(kc_id="kc-2", p_know=0.6, repetitions=2),
            ],
            initial_goals=[LearningGoal(description="目标A", priority=5)],
        )
        broker.update_mastery("sess-1", "kc-1", p_know=0.4, is_correct=True)
        broker.update_mastery("sess-1", "kc-2", p_know=0.6, is_correct=True)

        # 传递到会话 2
        broker.transfer_context("sess-1", "sess-2", "user-001")
        target = broker.get_envelope("sess-2")

        # 验证掌握度继承
        assert len(target.mastery_snapshot) == 2
        kc_ids = {s.kc_id for s in target.mastery_snapshot}
        assert kc_ids == {"kc-1", "kc-2"}

        # 验证目标继承
        assert len(target.goals) >= 1
        assert any(g.description == "目标A" for g in target.goals)

    def test_decay_over_time(self):
        """时间流逝后衰减生效."""
        broker = LearningContextBroker()
        now = int(time.time() * 1000)

        # 创建一个 48 小时前的掌握快照
        old_snap = MasterySnapshot(
            kc_id="kc-old",
            p_know=0.8,
            last_practiced_at=now - 48 * 3_600_000,
            repetitions=1,
        )
        broker.build_envelope(
            user_id="user-001",
            session_id="sess-001",
            initial_mastery=[old_snap],
        )

        # 刷新衰减
        broker.refresh_context("sess-001")
        envelope = broker.get_envelope("sess-001")
        snap = envelope.mastery_snapshot[0]

        # 48 小时后, reps=1, stability=24+6=30h
        # decay = exp(-48/30) ≈ 0.20
        # effective = 0.8 * 0.20 = 0.16
        assert snap.decay_factor < 0.5
        assert snap.effective_mastery() < 0.5  # 已降至薄弱

    def test_collector_broker_integration(self):
        """采集器与经纪器集成."""
        broker = LearningContextBroker()
        collector = ContextCollector()

        broker.build_envelope(user_id="user-001", session_id="sess-001")

        # 采集 Agent 输出 (BKT 更新)
        agent_evt = AgentOutputEvent(
            agent_id="agent-diagnosis-001",
            output_type="bkt_update",
            kc_id="kc-1",
            p_know=0.65,
            is_correct=True,
            timestamp=int(time.time() * 1000),
        )
        collector.collect_agent_output(agent_evt)

        # 采集用户声明
        decl = UserDeclaration(
            user_id="user-001",
            available_minutes=20,
            preferred_phase=LearningPhase.PRACTICE,
            timestamp=int(time.time() * 1000),
        )
        collector.collect_user_declaration(decl)

        # 应用用户声明到上下文
        broker.set_time_constraint(
            "sess-001",
            TimeConstraint(available_minutes=20, recommended_phase=LearningPhase.PRACTICE),
        )

        # 验证
        envelope = broker.get_envelope("sess-001")
        assert envelope.time_constraint is not None
        assert envelope.time_constraint.available_minutes == 20
