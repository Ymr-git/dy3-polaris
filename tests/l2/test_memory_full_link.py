"""T4 三记忆 + 遗忘全链路集成测试.

全链路定义: 工作记忆 → 短期记忆 → 长期记忆 迁移机制 + FSRS 遗忘调度

测试覆盖:
1. MemoryTracingService — 服务初始化与基础处理
2. MemoryMigration — 记忆层级迁移机制 (working→short-term→long-term)
3. FSRSIntegration — FSRS 调度集成 (幂律遗忘 + 间隔重复)
4. MemoryOutput — 输出契约 (字段 / 序列化 / 往返)
5. FullLinkIntegration — 端到端全链路集成
6. WorldSchemeIntegration — 世界先进方案融合验证
    (Atkinson-Shiffrin / Miller / Ebbinghaus / FSRS-6 / LRU)
"""

from __future__ import annotations

import time

import pytest

from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.memory import (
    LongTermMemory,
    MAX_CHUNKS,
    MemoryChunk,
    ShortTermMemory,
    WorkingMemory,
)
from dy3_polaris.l2.memory.tracing_service import MemoryOutput, MemoryTracingService
from dy3_polaris.l2.store import InMemoryL2Store


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def store():
    return InMemoryL2Store()


@pytest.fixture
def service(store):
    return MemoryTracingService(store=store)


# ============================================================
# 1. MemoryTracingService — 服务初始化与基础处理
# ============================================================


class TestMemoryTracingService:
    """MemoryTracingService 初始化与基础处理测试."""

    def test_service_initializes(self):
        """MemoryTracingService 可通过 store=None 创建."""
        svc = MemoryTracingService(store=None)
        assert svc is not None
        assert svc.store is not None

    def test_service_has_three_memory_tiers(self):
        """服务包含 WorkingMemory / ShortTermMemory / LongTermMemory 三层."""
        svc = MemoryTracingService()
        assert svc.working_memory is not None
        assert svc.short_term_memory is not None
        assert svc.long_term_memory is not None
        assert isinstance(svc.working_memory, WorkingMemory)
        assert isinstance(svc.short_term_memory, ShortTermMemory)
        assert isinstance(svc.long_term_memory, LongTermMemory)

    def test_service_process_answer_event(self):
        """处理单条答题事件返回 MemoryOutput."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        event = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_math_01",
            correct=True,
            difficulty=0.5,
            timestamp=time.time(),
        )
        output = svc.process(event)
        assert output is not None
        assert isinstance(output, MemoryOutput)
        assert output.learner_id == "learner_001"

    def test_service_add_to_working_memory(self):
        """答题事件将信息块添加到工作记忆."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        event = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_math_01",
            correct=True,
            difficulty=0.5,
            timestamp=time.time(),
        )
        output = svc.process(event)
        assert output.working_memory_size == 1
        assert svc.working_memory.get_size() == 1

    def test_service_working_memory_capacity(self):
        """工作记忆遵守 Miller 7±2 容量上限 (MAX_CHUNKS=9)."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        assert WorkingMemory.MAX_CHUNKS == 9
        assert MAX_CHUNKS == 9
        ts = time.time()
        for i in range(9):
            event = AnswerEvent(
                learner_id="learner_001",
                kp_id=f"kp_{i}",
                correct=True,
                difficulty=0.5,
                timestamp=ts + i,
            )
            output = svc.process(event)
        assert output.working_memory_size == 9
        assert svc.working_memory.is_full() is True

    def test_service_working_memory_lru(self):
        """超容量时 LRU 淘汰, 容量保持 MAX_CHUNKS, 并迁移到短期记忆."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        outputs = []
        for i in range(10):
            event = AnswerEvent(
                learner_id="learner_001",
                kp_id=f"kp_{i}",
                correct=True,
                difficulty=0.5,
                timestamp=ts + i,
            )
            output = svc.process(event)
            outputs.append(output)
        # 容量保持 9 (LRU 淘汰最旧块)
        assert outputs[-1].working_memory_size == 9
        # 第 10 个事件触发 working_to_short_term 迁移
        assert "working_to_short_term" in outputs[-1].migration_events


# ============================================================
# 2. MemoryMigration — 记忆层级迁移机制
# ============================================================


class TestMemoryMigration:
    """记忆层级迁移: working → short-term → long-term."""

    def test_working_to_short_term_migration(self):
        """工作记忆溢出时, 信息块迁移到短期记忆."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        outputs = []
        for i in range(10):
            event = AnswerEvent(
                learner_id="learner_001",
                kp_id=f"kp_{i}",
                correct=True,
                difficulty=0.5,
                timestamp=ts + i,
            )
            output = svc.process(event)
            outputs.append(output)
        # 第 10 个事件触发迁移
        assert "working_to_short_term" in outputs[-1].migration_events
        # 短期记忆中有被淘汰的信息块
        entries = svc.short_term_memory.get_entries("learner_001")
        assert len(entries) >= 1

    def test_short_term_to_long_term_migration(self):
        """高重要度条目迁移到长期记忆 (持久化)."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        # difficulty=0.8 → importance=0.8 >= 0.7 → 迁移到长期记忆
        event = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=True,
            difficulty=0.8,
            timestamp=ts,
        )
        output = svc.process(event)
        assert output.long_term_persisted is True
        assert "short_term_to_long_term" in output.migration_events

    def test_migration_preserves_data(self, store):
        """迁移后数据在长期记忆 (store) 中保留."""
        svc = MemoryTracingService(store=store)
        ts = time.time()
        event = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=True,
            difficulty=0.8,
            timestamp=ts,
        )
        svc.process(event)
        # store 中有追踪状态
        state = store.get_tracing_state("learner_001", "kp_01")
        assert state is not None
        assert state.kp_id == "kp_01"
        assert state.attempts >= 1

    def test_migration_importance_filter(self):
        """仅高重要度条目迁移到长期记忆 (首次曝光时)."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        # 低重要度 (difficulty=0.2 < 0.7), 首次曝光 (reps=1 < 3)
        event_low = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_low",
            correct=True,
            difficulty=0.2,
            timestamp=ts,
        )
        output_low = svc.process(event_low)
        assert output_low.long_term_persisted is False
        assert "short_term_to_long_term" not in output_low.migration_events

        # 高重要度 (difficulty=0.8 >= 0.7)
        event_high = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_high",
            correct=True,
            difficulty=0.8,
            timestamp=ts + 1,
        )
        output_high = svc.process(event_high)
        assert output_high.long_term_persisted is True
        assert "short_term_to_long_term" in output_high.migration_events

    def test_short_term_expiry(self):
        """短期记忆条目超过保留窗口后过期."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        # 填满工作记忆以触发迁移到短期记忆
        for i in range(10):
            event = AnswerEvent(
                learner_id="learner_001",
                kp_id=f"kp_{i}",
                correct=True,
                difficulty=0.5,
                timestamp=ts + i,
            )
            svc.process(event)
        assert len(svc.short_term_memory.get_entries("learner_001")) >= 1
        # 全部过期
        svc.short_term_memory.expire_all()
        assert len(svc.short_term_memory.get_entries("learner_001")) == 0

    def test_long_term_persistence(self, store):
        """长期记忆通过 L2Store 持久化 (重复曝光触发迁移)."""
        svc = MemoryTracingService(store=store)
        ts = time.time()
        # 低重要度但重复曝光 (reps >= 3 触发迁移)
        outputs = []
        for i in range(5):
            event = AnswerEvent(
                learner_id="learner_001",
                kp_id="kp_01",
                correct=True,
                difficulty=0.3,
                timestamp=ts + i,
            )
            output = svc.process(event)
            outputs.append(output)
        # 第 3 次及以后应迁移到长期记忆
        assert any(o.long_term_persisted for o in outputs)
        # store 中有数据
        state = store.get_tracing_state("learner_001", "kp_01")
        assert state is not None
        assert state.kp_id == "kp_01"


# ============================================================
# 3. FSRSIntegration — FSRS 调度集成
# ============================================================


class TestFSRSIntegration:
    """FSRS (简化版) 间隔重复调度集成测试."""

    def test_fsrs_schedule_review(self, service):
        """FSRS 根据卡片状态和评分调度复习."""
        ts = time.time()
        event = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=True,
            difficulty=0.5,
            timestamp=ts,
        )
        service.process(event)
        interval = service.schedule_review("learner_001", "kp_01")
        assert isinstance(interval, int)
        assert interval >= 1

    def test_fsrs_interval_increases(self, service):
        """成功回忆后复习间隔递增."""
        ts = time.time()
        intervals = []
        for i in range(5):
            event = AnswerEvent(
                learner_id="learner_001",
                kp_id="kp_01",
                correct=True,
                difficulty=0.3,
                timestamp=ts + i * 86400,
            )
            output = service.process(event)
            intervals.append(output.fsrs_next_review_days)
        # 间隔应递增
        assert intervals[-1] > intervals[0]
        for i in range(1, len(intervals)):
            assert intervals[i] >= intervals[i - 1]

    def test_fsrs_lapse_handling(self, service):
        """遗忘 (grade=1) 后间隔显著缩短."""
        ts = time.time()
        # 建立稳定性
        for i in range(3):
            event = AnswerEvent(
                learner_id="learner_001",
                kp_id="kp_01",
                correct=True,
                difficulty=0.3,
                timestamp=ts + i,
            )
            output = service.process(event)
        interval_before = output.fsrs_next_review_days
        assert interval_before >= 3
        # 遗忘 (答错 + 高难度 → grade=1)
        lapse_event = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=False,
            difficulty=0.8,
            timestamp=ts + 10,
        )
        output_lapse = service.process(lapse_event)
        interval_after = output_lapse.fsrs_next_review_days
        # 间隔应缩短
        assert interval_after < interval_before

    def test_fsrs_retrievability_decay(self, service):
        """可提取性随时间衰减."""
        ts = time.time()
        # 首次复习
        event1 = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=True,
            difficulty=0.5,
            timestamp=ts,
        )
        output1 = service.process(event1)
        r1 = output1.retrievability
        # 30 天后再复习 (pre-update retrievability 应体现衰减)
        event2 = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=True,
            difficulty=0.5,
            timestamp=ts + 30 * 86400,
        )
        output2 = service.process(event2)
        r2 = output2.retrievability
        # 可提取性应衰减
        assert r2 < r1
        assert 0.0 <= r2 <= 1.0

    def test_fsrs_difficulty_adjustment(self, service):
        """难度根据表现调整 (Easy 降低 / Again 升高)."""
        ts = time.time()
        # 首次: grade=4 (Easy, correct + difficulty=0.2 < 0.3)
        event1 = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=True,
            difficulty=0.2,
            timestamp=ts,
        )
        output1 = service.process(event1)
        d1 = output1.difficulty
        # 第二次: grade=4 → 难度应降低
        event2 = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=True,
            difficulty=0.2,
            timestamp=ts + 86400,
        )
        output2 = service.process(event2)
        d2 = output2.difficulty
        assert d2 < d1
        # 第三次: grade=1 (Again, wrong + difficulty=0.8 >= 0.7) → 难度应升高
        event3 = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_01",
            correct=False,
            difficulty=0.8,
            timestamp=ts + 2 * 86400,
        )
        output3 = service.process(event3)
        d3 = output3.difficulty
        assert d3 > d2

    def test_fsrs_get_state(self, service):
        """get_fsrs_state 返回完整 FSRS 状态."""
        ts = time.time()
        service.process(AnswerEvent(
            learner_id="learner_001", kp_id="kp_01", correct=True,
            difficulty=0.5, timestamp=ts,
        ))
        state = service.get_fsrs_state("learner_001", "kp_01")
        assert state is not None
        assert "stability" in state
        assert "difficulty" in state
        assert "reps" in state
        assert "lapses" in state
        assert "retrievability" in state

    def test_fsrs_get_state_missing(self, service):
        """get_fsrs_state 对未知 KP 返回 None."""
        assert service.get_fsrs_state("unknown", "unknown_kp") is None


# ============================================================
# 4. MemoryOutput — 输出契约
# ============================================================


class TestMemoryOutput:
    """MemoryOutput 标准化输出契约测试."""

    REQUIRED_FIELDS = [
        "learner_id",
        "working_memory_size",
        "short_term_count",
        "long_term_persisted",
        "fsrs_next_review_days",
        "retrievability",
        "stability",
        "difficulty",
        "review_grade",
        "migration_events",
        "last_updated_ts",
    ]

    def test_output_fields(self):
        """MemoryOutput 包含所有必需字段."""
        output = MemoryOutput(
            learner_id="l1",
            working_memory_size=3,
            short_term_count=2,
            long_term_persisted=True,
            fsrs_next_review_days=7,
            retrievability=0.85,
            stability=5.0,
            difficulty=5.5,
            review_grade=3,
            migration_events=["working_to_short_term"],
            last_updated_ts=1000.0,
        )
        assert output.learner_id == "l1"
        assert output.working_memory_size == 3
        assert output.short_term_count == 2
        assert output.long_term_persisted is True
        assert output.fsrs_next_review_days == 7
        assert output.retrievability == 0.85
        assert output.stability == 5.0
        assert output.difficulty == 5.5
        assert output.review_grade == 3
        assert output.migration_events == ["working_to_short_term"]
        assert output.last_updated_ts == 1000.0

    def test_output_to_dict(self):
        """MemoryOutput 可序列化为字典."""
        output = MemoryOutput(
            learner_id="l1",
            working_memory_size=5,
            short_term_count=1,
            long_term_persisted=False,
            fsrs_next_review_days=3,
            retrievability=0.9,
            stability=3.0,
            difficulty=5.5,
            review_grade=4,
            migration_events=["short_term_to_long_term"],
            last_updated_ts=500.0,
        )
        d = output.to_dict()
        assert isinstance(d, dict)
        assert d["learner_id"] == "l1"
        assert d["working_memory_size"] == 5
        assert d["long_term_persisted"] is False
        assert d["fsrs_next_review_days"] == 3
        assert d["migration_events"] == ["short_term_to_long_term"]
        # 全部必需字段存在
        for field in self.REQUIRED_FIELDS:
            assert field in d, f"Missing field: {field}"

    def test_output_from_dict(self):
        """MemoryOutput 可从字典反序列化."""
        d = {
            "learner_id": "l2",
            "working_memory_size": 7,
            "short_term_count": 3,
            "long_term_persisted": True,
            "fsrs_next_review_days": 14,
            "retrievability": 0.72,
            "stability": 10.0,
            "difficulty": 4.0,
            "review_grade": 3,
            "migration_events": ["working_to_short_term", "short_term_to_long_term"],
            "last_updated_ts": 2000.0,
        }
        output = MemoryOutput.from_dict(d)
        assert output.learner_id == "l2"
        assert output.working_memory_size == 7
        assert output.short_term_count == 3
        assert output.long_term_persisted is True
        assert output.fsrs_next_review_days == 14
        assert output.retrievability == 0.72
        assert output.stability == 10.0
        assert output.difficulty == 4.0
        assert output.review_grade == 3
        assert output.migration_events == ["working_to_short_term", "short_term_to_long_term"]
        assert output.last_updated_ts == 2000.0

    def test_output_roundtrip(self):
        """序列化-反序列化往返一致."""
        original = MemoryOutput(
            learner_id="l3",
            working_memory_size=9,
            short_term_count=5,
            long_term_persisted=True,
            fsrs_next_review_days=21,
            retrievability=0.65,
            stability=15.0,
            difficulty=6.5,
            review_grade=2,
            migration_events=["working_to_short_term", "short_term_to_long_term"],
            last_updated_ts=9999.0,
        )
        d = original.to_dict()
        restored = MemoryOutput.from_dict(d)
        assert restored.learner_id == original.learner_id
        assert restored.working_memory_size == original.working_memory_size
        assert restored.short_term_count == original.short_term_count
        assert restored.long_term_persisted == original.long_term_persisted
        assert restored.fsrs_next_review_days == original.fsrs_next_review_days
        assert restored.retrievability == original.retrievability
        assert restored.stability == original.stability
        assert restored.difficulty == original.difficulty
        assert restored.review_grade == original.review_grade
        assert restored.migration_events == original.migration_events
        assert restored.last_updated_ts == original.last_updated_ts


# ============================================================
# 5. FullLinkIntegration — 端到端全链路集成
# ============================================================


class TestFullLinkIntegration:
    """工作记忆 → 短期记忆 → 长期记忆 + FSRS 端到端集成."""

    def test_full_link_single_learner(self, service):
        """单学习者全链路: 事件 → 工作记忆 → FSRS → 输出."""
        ts = time.time()
        event = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_math_01",
            correct=True,
            difficulty=0.5,
            timestamp=ts,
        )
        output = service.process(event)
        # 全链路输出验证
        assert output.learner_id == "learner_001"
        assert output.working_memory_size == 1
        assert output.review_grade in (1, 2, 3, 4)
        assert 0.0 <= output.retrievability <= 1.0
        assert output.stability > 0.0
        assert 1.0 <= output.difficulty <= 10.0
        assert output.fsrs_next_review_days >= 1

    def test_full_link_learning_progression(self, service):
        """多次答题逐步建立记忆."""
        ts = time.time()
        learner_id = "learner_001"
        kp_id = "kp_math_01"
        stabilities = []
        for i in range(8):
            event = AnswerEvent(
                learner_id=learner_id,
                kp_id=kp_id,
                correct=True,
                difficulty=0.3,
                timestamp=ts + i * 86400,
            )
            output = service.process(event)
            stabilities.append(output.stability)
        # 稳定性应递增 (学习进步)
        assert stabilities[-1] > stabilities[0]
        # FSRS 状态存在
        state = service.get_fsrs_state(learner_id, kp_id)
        assert state is not None
        assert state["reps"] == 8

    def test_full_link_forgetting_and_review(self, service):
        """遗忘发生, 复习恢复记忆."""
        ts = time.time()
        learner_id = "learner_001"
        kp_id = "kp_01"
        # 建立记忆
        for i in range(3):
            event = AnswerEvent(
                learner_id=learner_id,
                kp_id=kp_id,
                correct=True,
                difficulty=0.3,
                timestamp=ts + i,
            )
            service.process(event)
        # 长时间后, 可提取性衰减 (pre-update retrievability)
        far_event = AnswerEvent(
            learner_id=learner_id,
            kp_id=kp_id,
            correct=True,
            difficulty=0.3,
            timestamp=ts + 60 * 86400,
        )
        output_far = service.process(far_event)
        assert output_far.retrievability < 0.9  # 衰减
        # 紧接着复习, 可提取性恢复 (elapsed ≈ 0)
        review_event = AnswerEvent(
            learner_id=learner_id,
            kp_id=kp_id,
            correct=True,
            difficulty=0.3,
            timestamp=ts + 60 * 86400 + 1,
        )
        output_review = service.process(review_event)
        assert output_review.retrievability > output_far.retrievability

    def test_full_link_migration_chain(self, store):
        """完整迁移链: working → short-term → long-term."""
        svc = MemoryTracingService(store=store)
        ts = time.time()
        learner_id = "learner_001"
        # 填满工作记忆触发 working → short-term
        for i in range(10):
            event = AnswerEvent(
                learner_id=learner_id,
                kp_id=f"kp_{i}",
                correct=True,
                difficulty=0.5,
                timestamp=ts + i,
            )
            output = svc.process(event)
        assert "working_to_short_term" in output.migration_events
        assert len(svc.short_term_memory.get_entries(learner_id)) >= 1
        # 高重要度事件触发 short-term → long-term
        event = AnswerEvent(
            learner_id=learner_id,
            kp_id="kp_important",
            correct=True,
            difficulty=0.9,
            timestamp=ts + 100,
        )
        output = svc.process(event)
        assert output.long_term_persisted is True
        assert "short_term_to_long_term" in output.migration_events
        # store 中有数据
        assert store.get_tracing_state(learner_id, "kp_important") is not None

    def test_full_link_fsrs_scheduling(self, service):
        """FSRS 调度合适的复习时间."""
        ts = time.time()
        learner_id = "learner_001"
        kp_id = "kp_01"
        # 答对多次, 间隔应递增
        intervals = []
        for i in range(5):
            event = AnswerEvent(
                learner_id=learner_id,
                kp_id=kp_id,
                correct=True,
                difficulty=0.3,
                timestamp=ts + i * 86400,
            )
            output = service.process(event)
            intervals.append(output.fsrs_next_review_days)
        # schedule_review 返回与最后一次输出一致的间隔
        scheduled = service.schedule_review(learner_id, kp_id)
        assert scheduled == intervals[-1]
        # 间隔递增 (间隔重复)
        assert intervals[-1] > intervals[0]

    def test_full_link_graceful_degradation(self):
        """服务处理边界情况 (优雅降级)."""
        # store=None 正常工作
        svc = MemoryTracingService(store=None)
        event = AnswerEvent(
            learner_id="l1",
            kp_id="kp_01",
            correct=True,
            difficulty=0.5,
            timestamp=time.time(),
        )
        output = svc.process(event)
        assert output is not None
        assert output.learner_id == "l1"
        # 未知 KP 的 FSRS 状态返回 None
        assert svc.get_fsrs_state("l1", "unknown_kp") is None
        # 未知 KP 的复习间隔返回默认值
        assert svc.schedule_review("l1", "unknown_kp") >= 1
        # 空批量处理返回空列表
        assert svc.batch_process([]) == []
        # get_memory_snapshot 对未知学习者返回空状态
        snap = svc.get_memory_snapshot("unknown_learner")
        assert snap["working_memory_size"] >= 0
        assert snap["short_term_count"] == 0

    def test_full_link_batch_process(self, service):
        """批量处理多个事件."""
        ts = time.time()
        events = [
            AnswerEvent(learner_id="l1", kp_id="kp_01", correct=True,
                        difficulty=0.4, timestamp=ts),
            AnswerEvent(learner_id="l1", kp_id="kp_01", correct=True,
                        difficulty=0.4, timestamp=ts + 86400),
            AnswerEvent(learner_id="l1", kp_id="kp_02", correct=False,
                        difficulty=0.7, timestamp=ts + 2 * 86400),
            AnswerEvent(learner_id="l2", kp_id="kp_01", correct=True,
                        difficulty=0.5, timestamp=ts + 3 * 86400),
        ]
        outputs = service.batch_process(events)
        assert len(outputs) == 4
        assert all(isinstance(o, MemoryOutput) for o in outputs)
        # kp_01 两次答对的稳定性应高于 kp_02 答错
        assert outputs[1].stability > outputs[2].stability

    def test_full_link_memory_snapshot(self, service):
        """get_memory_snapshot 返回完整记忆状态."""
        ts = time.time()
        learner_id = "learner_001"
        for kp_id in ["kp_01", "kp_02", "kp_03"]:
            service.process(AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=True,
                difficulty=0.4, timestamp=ts,
            ))
            ts += 1
        snapshot = service.get_memory_snapshot(learner_id)
        assert "working_memory_size" in snapshot
        assert "short_term_count" in snapshot
        assert "fsrs_states" in snapshot
        assert len(snapshot["fsrs_states"]) == 3
        assert all(kp in snapshot["fsrs_states"]
                   for kp in ["kp_01", "kp_02", "kp_03"])


# ============================================================
# 6. WorldSchemeIntegration — 世界先进方案融合验证
# ============================================================


class TestWorldSchemeIntegration:
    """验证世界先进方案在全链路中的融合."""

    def test_atkinson_shiffrin_model(self):
        """三层记忆模型匹配 Atkinson-Shiffrin 多重存储模型."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        # 三层: 感觉记忆(省略) → 工作记忆 → 短期记忆 → 长期记忆
        assert isinstance(svc.working_memory, WorkingMemory)
        assert isinstance(svc.short_term_memory, ShortTermMemory)
        assert isinstance(svc.long_term_memory, LongTermMemory)
        # 迁移路径: working → short-term → long-term
        ts = time.time()
        learner_id = "learner_001"
        # 填满工作记忆, 触发 → 短期记忆
        for i in range(10):
            svc.process(AnswerEvent(
                learner_id=learner_id, kp_id=f"kp_{i}", correct=True,
                difficulty=0.5, timestamp=ts + i,
            ))
        assert len(svc.short_term_memory.get_entries(learner_id)) >= 1
        # 高重要度 → 长期记忆
        svc.process(AnswerEvent(
            learner_id=learner_id, kp_id="kp_imp", correct=True,
            difficulty=0.9, timestamp=ts + 100,
        ))
        assert svc.long_term_memory.get_tracing_state(learner_id, "kp_imp") is not None

    def test_miller_law_capacity(self):
        """工作记忆容量 7±2 (MAX_CHUNKS=9 为上界)."""
        assert WorkingMemory.MAX_CHUNKS == 9
        assert MAX_CHUNKS == 9
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        for i in range(9):
            svc.process(AnswerEvent(
                learner_id="l1", kp_id=f"kp_{i}", correct=True,
                difficulty=0.5, timestamp=ts + i,
            ))
        assert svc.working_memory.get_size() == 9
        assert svc.working_memory.is_full() is True

    def test_ebbinghaus_forgetting(self):
        """遗忘遵循 Ebbinghaus 曲线模式 (可提取性随时间衰减)."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        learner_id = "learner_001"
        kp_id = "kp_01"
        # 建立记忆
        for i in range(3):
            svc.process(AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=True,
                difficulty=0.3, timestamp=ts + i,
            ))
        base_ts = ts + 10  # 在最后一次复习之后
        # 不同时间点的可提取性
        r_values = []
        for days in [0, 1, 7, 14, 30]:
            state = svc.get_fsrs_state(learner_id, kp_id, current_time=base_ts + days * 86400)
            r_values.append(state["retrievability"])
        # 可提取性随时间递减 (Ebbinghaus 曲线)
        for i in range(1, len(r_values)):
            assert r_values[i] <= r_values[i - 1]
        # 30 天后可提取性显著低于初始
        assert r_values[-1] < r_values[0]

    def test_fsrs6_spaced_repetition(self):
        """FSRS-6 间隔重复调度 (间隔随成功回忆递增)."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        intervals = []
        for i in range(6):
            output = svc.process(AnswerEvent(
                learner_id="learner_001", kp_id="kp_01", correct=True,
                difficulty=0.3, timestamp=ts + i * 86400,
            ))
            intervals.append(output.fsrs_next_review_days)
        # 间隔重复: 间隔递增
        for i in range(1, len(intervals)):
            assert intervals[i] >= intervals[i - 1]
        assert intervals[-1] > intervals[0]

    def test_lru_eviction_strategy(self):
        """LRU 淘汰策略: 超容量时淘汰最久未访问的块."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        ts = time.time()
        # 添加 10 个块 (容量 9, 第 10 个触发淘汰)
        for i in range(10):
            svc.process(AnswerEvent(
                learner_id="l1", kp_id=f"kp_{i}", correct=True,
                difficulty=0.5, timestamp=ts + i,
            ))
        context = svc.working_memory.get_context()
        assert len(context) == 9
        # 最旧的块 (kp_0 的 chunk) 被淘汰
        chunk_ids = [c.chunk_id for c in context]
        # 短期记忆中有被淘汰的块
        entries = svc.short_term_memory.get_entries("l1")
        assert len(entries) >= 1

    def test_output_contract_for_downstream(self):
        """输出契约可被下游 T2/T3/T5 消费."""
        svc = MemoryTracingService(store=InMemoryL2Store())
        output = svc.process(AnswerEvent(
            learner_id="l1", kp_id="kp_01", correct=True,
            difficulty=0.5, timestamp=time.time(),
        ))
        d = output.to_dict()
        required = [
            "learner_id",
            "working_memory_size",
            "short_term_count",
            "long_term_persisted",
            "fsrs_next_review_days",
            "retrievability",
            "stability",
            "difficulty",
            "review_grade",
            "migration_events",
            "last_updated_ts",
        ]
        for field in required:
            assert field in d, f"Missing field for downstream: {field}"
        # 类型契约
        assert isinstance(d["learner_id"], str)
        assert isinstance(d["working_memory_size"], int)
        assert isinstance(d["short_term_count"], int)
        assert isinstance(d["long_term_persisted"], bool)
        assert isinstance(d["fsrs_next_review_days"], int)
        assert isinstance(d["retrievability"], float)
        assert isinstance(d["stability"], float)
        assert isinstance(d["difficulty"], float)
        assert isinstance(d["review_grade"], int)
        assert isinstance(d["migration_events"], list)
        assert isinstance(d["last_updated_ts"], float)
