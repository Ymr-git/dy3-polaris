"""L2 个性化层基础设施测试 — 异常体系 / 数据模型 / 存储层 / 缓存层.

测试覆盖 (TDD RED 阶段, 实现尚未存在, 预期因 ImportError 失败):
1. 异常体系: L2Error 基类及 5 个子类, 继承 L6Error, JSON-RPC 码 -32300 范围
   - L2Error(-32300) / ProfileNotFoundError(-32301) / TracingError(-32302)
   - IRTError(-32303) / MemoryError(-32304) / StoreError(-32305)
   - 每个异常: code / detail / context 属性 + to_json_rpc_error() 码值
2. L2 数据模型: AnswerRecord / TracingState / IRTState / LearnerSnapshot / SessionRecord
   - 字段验证 / 默认值 / to_dict()/from_dict() 往返序列化
3. 存储层: L2Store 抽象基类接口 + InMemoryL2Store 实现 (线程安全, threading.RLock)
4. 缓存层: L2Cache 分层 TTL (profile/bkt/memory) + write-through 语义 + clear()
"""

import threading
import time

import pytest

from dy3_polaris.l6.core.exceptions import L6Error
from dy3_polaris.l2.exceptions import (
    IRTError,
    L2Error,
    MemoryError,
    ProfileNotFoundError,
    StoreError,
    TracingError,
)
from dy3_polaris.l2.models import (
    AnswerRecord,
    IRTState,
    LearnerSnapshot,
    SessionRecord,
    TracingState,
)
from dy3_polaris.l2.store import InMemoryL2Store, L2Store
from dy3_polaris.l2.cache import L2Cache


# ============================================================
# 1. 异常体系测试
# ============================================================


class TestL2ExceptionHierarchy:
    """L2 异常体系测试 — 继承层级 + code/detail/context + JSON-RPC 码."""

    # --- 继承层级 ---

    def test_l2_error_is_l6_error(self):
        """L2Error 继承 L6Error."""
        assert issubclass(L2Error, L6Error)

    def test_profile_not_found_is_l2_error(self):
        """ProfileNotFoundError 继承 L2Error."""
        assert issubclass(ProfileNotFoundError, L2Error)

    def test_tracing_error_is_l2_error(self):
        """TracingError 继承 L2Error."""
        assert issubclass(TracingError, L2Error)

    def test_irt_error_is_l2_error(self):
        """IRTError 继承 L2Error."""
        assert issubclass(IRTError, L2Error)

    def test_memory_error_is_l2_error(self):
        """MemoryError 继承 L2Error."""
        assert issubclass(MemoryError, L2Error)

    def test_store_error_is_l2_error(self):
        """StoreError 继承 L2Error."""
        assert issubclass(StoreError, L2Error)

    # --- L2Error 基类 (JSON-RPC -32300) ---

    def test_l2_error_code(self):
        """L2Error code 属性为 L2_ERROR."""
        err = L2Error(detail="L2 基础错误")
        assert err.code == "L2_ERROR"

    def test_l2_error_detail(self):
        """L2Error detail 属性保留传入详情."""
        err = L2Error(detail="L2 基础错误")
        assert err.detail == "L2 基础错误"

    def test_l2_error_context(self):
        """L2Error context 属性保留传入上下文."""
        err = L2Error(detail="测试", context={"module": "profile_builder"})
        assert err.context == {"module": "profile_builder"}

    def test_l2_error_context_default_empty(self):
        """L2Error 无 context 时默认空字典."""
        err = L2Error(detail="测试")
        assert err.context == {}

    def test_l2_error_to_json_rpc_error(self):
        """L2Error to_json_rpc_error() 码值为 -32300."""
        err = L2Error(detail="测试")
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32300

    # --- ProfileNotFoundError (code=PROFILE_NOT_FOUND, -32301) ---

    def test_profile_not_found_error_code(self):
        """ProfileNotFoundError code 属性."""
        err = ProfileNotFoundError(learner_id="learner-001")
        assert err.code == "PROFILE_NOT_FOUND"

    def test_profile_not_found_error_detail(self):
        """ProfileNotFoundError detail 属性."""
        err = ProfileNotFoundError(learner_id="learner-001", detail="画像不存在")
        assert err.detail == "画像不存在"

    def test_profile_not_found_error_context(self):
        """ProfileNotFoundError context 包含 learner_id."""
        err = ProfileNotFoundError(learner_id="learner-001")
        assert err.context["learner_id"] == "learner-001"

    def test_profile_not_found_error_to_json_rpc_error(self):
        """ProfileNotFoundError to_json_rpc_error() 码值为 -32301."""
        err = ProfileNotFoundError(learner_id="learner-001")
        assert err.to_json_rpc_error()["code"] == -32301

    # --- TracingError (code=TRACING_ERROR, -32302) ---

    def test_tracing_error_code(self):
        """TracingError code 属性."""
        err = TracingError(detail="追踪失败")
        assert err.code == "TRACING_ERROR"

    def test_tracing_error_detail(self):
        """TracingError detail 属性."""
        err = TracingError(detail="BKT 更新失败")
        assert err.detail == "BKT 更新失败"

    def test_tracing_error_context(self):
        """TracingError context 保留传入上下文."""
        err = TracingError(detail="测试", context={"kp_id": "kp-001"})
        assert err.context["kp_id"] == "kp-001"

    def test_tracing_error_to_json_rpc_error(self):
        """TracingError to_json_rpc_error() 码值为 -32302."""
        err = TracingError(detail="测试")
        assert err.to_json_rpc_error()["code"] == -32302

    # --- IRTError (code=IRT_ERROR, -32303) ---

    def test_irt_error_code(self):
        """IRTError code 属性."""
        err = IRTError(detail="IRT 估计失败")
        assert err.code == "IRT_ERROR"

    def test_irt_error_detail(self):
        """IRTError detail 属性."""
        err = IRTError(detail="theta 收敛失败")
        assert err.detail == "theta 收敛失败"

    def test_irt_error_context(self):
        """IRTError context 保留传入上下文."""
        err = IRTError(detail="测试", context={"learner_id": "learner-001"})
        assert err.context["learner_id"] == "learner-001"

    def test_irt_error_to_json_rpc_error(self):
        """IRTError to_json_rpc_error() 码值为 -32303."""
        err = IRTError(detail="测试")
        assert err.to_json_rpc_error()["code"] == -32303

    # --- MemoryError (code=MEMORY_ERROR, -32304) ---

    def test_memory_error_code(self):
        """MemoryError code 属性."""
        err = MemoryError(detail="记忆检索失败")
        assert err.code == "MEMORY_ERROR"

    def test_memory_error_detail(self):
        """MemoryError detail 属性."""
        err = MemoryError(detail="记忆图谱写入失败")
        assert err.detail == "记忆图谱写入失败"

    def test_memory_error_context(self):
        """MemoryError context 保留传入上下文."""
        err = MemoryError(detail="测试", context={"session_id": "sess-001"})
        assert err.context["session_id"] == "sess-001"

    def test_memory_error_to_json_rpc_error(self):
        """MemoryError to_json_rpc_error() 码值为 -32304."""
        err = MemoryError(detail="测试")
        assert err.to_json_rpc_error()["code"] == -32304

    # --- StoreError (code=STORE_ERROR, -32305) ---

    def test_store_error_code(self):
        """StoreError code 属性."""
        err = StoreError(detail="存储写入失败")
        assert err.code == "STORE_ERROR"

    def test_store_error_detail(self):
        """StoreError detail 属性."""
        err = StoreError(detail="持久化失败")
        assert err.detail == "持久化失败"

    def test_store_error_context(self):
        """StoreError context 保留传入上下文."""
        err = StoreError(detail="测试", context={"operation": "save_profile"})
        assert err.context["operation"] == "save_profile"

    def test_store_error_to_json_rpc_error(self):
        """StoreError to_json_rpc_error() 码值为 -32305."""
        err = StoreError(detail="测试")
        assert err.to_json_rpc_error()["code"] == -32305

    # --- 异常可抛出与捕获 ---

    def test_l2_error_catch_as_l6_error(self):
        """L2 异常可作为 L6Error 捕获."""
        with pytest.raises(L6Error):
            raise ProfileNotFoundError(learner_id="learner-001")

    def test_subclass_error_catch_as_l2_error(self):
        """子类异常可作为 L2Error 捕获."""
        with pytest.raises(L2Error):
            raise TracingError(detail="测试")


# ============================================================
# 2. L2 数据模型测试
# ============================================================


class TestL2Models:
    """L2 数据模型测试 — 字段验证 / 默认值 / to_dict()/from_dict() 往返."""

    # --- AnswerRecord ---

    def test_answer_record_creation(self):
        """AnswerRecord 全字段创建."""
        rec = AnswerRecord(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            timestamp=1000.0,
            difficulty=0.7,
            question_id="q-001",
        )
        assert rec.learner_id == "learner-001"
        assert rec.kp_id == "kp-001"
        assert rec.correct is True
        assert rec.timestamp == 1000.0
        assert rec.difficulty == 0.7
        assert rec.question_id == "q-001"

    def test_answer_record_defaults(self):
        """AnswerRecord 默认值: difficulty=0.5, question_id=None."""
        rec = AnswerRecord(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=False,
            timestamp=1000.0,
        )
        assert rec.difficulty == 0.5
        assert rec.question_id is None

    def test_answer_record_correct_is_bool(self):
        """AnswerRecord correct 为 bool 类型."""
        rec = AnswerRecord(learner_id="l", kp_id="k", correct=True, timestamp=1.0)
        assert isinstance(rec.correct, bool)

    def test_answer_record_to_dict(self):
        """AnswerRecord to_dict() 返回字典."""
        rec = AnswerRecord(learner_id="l", kp_id="k", correct=True, timestamp=1.0)
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d["learner_id"] == "l"
        assert d["correct"] is True

    def test_answer_record_roundtrip(self):
        """AnswerRecord to_dict()/from_dict() 往返一致."""
        rec = AnswerRecord(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=False,
            timestamp=1234.5,
            difficulty=0.8,
            question_id="q-009",
        )
        restored = AnswerRecord.from_dict(rec.to_dict())
        assert restored.learner_id == rec.learner_id
        assert restored.kp_id == rec.kp_id
        assert restored.correct == rec.correct
        assert restored.timestamp == rec.timestamp
        assert restored.difficulty == rec.difficulty
        assert restored.question_id == rec.question_id

    # --- TracingState ---

    def test_tracing_state_creation(self):
        """TracingState 全字段创建, bkt_params 含 p_l0/p_t/p_g/p_s."""
        bkt_params = {"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1}
        state = TracingState(
            kp_id="kp-001",
            mastery_prob=0.75,
            attempts=3,
            correct_count=2,
            last_attempt_time=1000.0,
            bkt_params=bkt_params,
        )
        assert state.kp_id == "kp-001"
        assert state.mastery_prob == 0.75
        assert state.attempts == 3
        assert state.correct_count == 2
        assert state.last_attempt_time == 1000.0
        assert state.bkt_params == bkt_params

    def test_tracing_state_defaults(self):
        """TracingState 默认值: attempts=0, correct_count=0, last_attempt_time=0.0."""
        state = TracingState(
            kp_id="kp-001",
            mastery_prob=0.3,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
        )
        assert state.attempts == 0
        assert state.correct_count == 0
        assert state.last_attempt_time == 0.0

    def test_tracing_state_bkt_params_keys(self):
        """TracingState bkt_params 包含 BKT 四参数."""
        state = TracingState(
            kp_id="kp-001",
            mastery_prob=0.5,
            bkt_params={"p_l0": 0.4, "p_t": 0.08, "p_g": 0.25, "p_s": 0.15},
        )
        assert "p_l0" in state.bkt_params
        assert "p_t" in state.bkt_params
        assert "p_g" in state.bkt_params
        assert "p_s" in state.bkt_params

    def test_tracing_state_roundtrip(self):
        """TracingState to_dict()/from_dict() 往返一致."""
        state = TracingState(
            kp_id="kp-001",
            mastery_prob=0.66,
            attempts=5,
            correct_count=3,
            last_attempt_time=9999.0,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
        )
        restored = TracingState.from_dict(state.to_dict())
        assert restored.kp_id == state.kp_id
        assert restored.mastery_prob == state.mastery_prob
        assert restored.attempts == state.attempts
        assert restored.correct_count == state.correct_count
        assert restored.last_attempt_time == state.last_attempt_time
        assert restored.bkt_params == state.bkt_params

    # --- IRTState ---

    def test_irt_state_creation(self):
        """IRTState 全字段创建."""
        state = IRTState(theta=0.8, se=0.25, response_count=10, last_update_time=1000.0)
        assert state.theta == 0.8
        assert state.se == 0.25
        assert state.response_count == 10
        assert state.last_update_time == 1000.0

    def test_irt_state_defaults(self):
        """IRTState 默认值: se=0.3, response_count=0, last_update_time=0.0."""
        state = IRTState(theta=0.0)
        assert state.se == 0.3
        assert state.response_count == 0
        assert state.last_update_time == 0.0

    def test_irt_state_roundtrip(self):
        """IRTState to_dict()/from_dict() 往返一致."""
        state = IRTState(theta=-0.5, se=0.4, response_count=7, last_update_time=4321.0)
        restored = IRTState.from_dict(state.to_dict())
        assert restored.theta == state.theta
        assert restored.se == state.se
        assert restored.response_count == state.response_count
        assert restored.last_update_time == state.last_update_time

    def test_irt_state_roundtrip_preserves_defaults(self):
        """IRTState 往返后默认值保持."""
        state = IRTState(theta=1.2)
        restored = IRTState.from_dict(state.to_dict())
        assert restored.se == 0.3
        assert restored.response_count == 0

    # --- LearnerSnapshot ---

    def test_learner_snapshot_creation(self):
        """LearnerSnapshot 全字段创建."""
        snap = LearnerSnapshot(
            learner_id="learner-001",
            snapshot_ts=1000.0,
            kp_mastery={"kp-1": 0.8, "kp-2": 0.6},
            theta=0.7,
            level="intermediate",
        )
        assert snap.learner_id == "learner-001"
        assert snap.snapshot_ts == 1000.0
        assert snap.kp_mastery == {"kp-1": 0.8, "kp-2": 0.6}
        assert snap.theta == 0.7
        assert snap.level == "intermediate"

    def test_learner_snapshot_theta_can_be_none(self):
        """LearnerSnapshot theta 可为 None."""
        snap = LearnerSnapshot(
            learner_id="learner-001",
            snapshot_ts=1000.0,
            kp_mastery={},
            theta=None,
            level="novice",
        )
        assert snap.theta is None

    def test_learner_snapshot_roundtrip(self):
        """LearnerSnapshot to_dict()/from_dict() 往返一致."""
        snap = LearnerSnapshot(
            learner_id="learner-001",
            snapshot_ts=2000.0,
            kp_mastery={"kp-1": 0.9, "kp-2": 0.4},
            theta=0.55,
            level="advanced",
        )
        restored = LearnerSnapshot.from_dict(snap.to_dict())
        assert restored.learner_id == snap.learner_id
        assert restored.snapshot_ts == snap.snapshot_ts
        assert restored.kp_mastery == snap.kp_mastery
        assert restored.theta == snap.theta
        assert restored.level == snap.level

    def test_learner_snapshot_roundtrip_none_theta(self):
        """LearnerSnapshot theta=None 往返保持 None."""
        snap = LearnerSnapshot(
            learner_id="l1",
            snapshot_ts=1.0,
            kp_mastery={"kp": 0.5},
            theta=None,
            level="novice",
        )
        restored = LearnerSnapshot.from_dict(snap.to_dict())
        assert restored.theta is None

    # --- SessionRecord ---

    def test_session_record_creation(self):
        """SessionRecord 全字段创建."""
        sess = SessionRecord(
            session_id="sess-001",
            learner_id="learner-001",
            status="active",
            started_at=1000.0,
            context_envelope={"user_id": "u-001"},
            checkpoints=[{"seq": 1}, {"seq": 2}],
        )
        assert sess.session_id == "sess-001"
        assert sess.learner_id == "learner-001"
        assert sess.status == "active"
        assert sess.started_at == 1000.0
        assert sess.context_envelope == {"user_id": "u-001"}
        assert sess.checkpoints == [{"seq": 1}, {"seq": 2}]

    def test_session_record_defaults(self):
        """SessionRecord 默认值: status=active, context_envelope=None, checkpoints=[]."""
        sess = SessionRecord(
            session_id="sess-001",
            learner_id="learner-001",
            started_at=1000.0,
        )
        assert sess.status == "active"
        assert sess.context_envelope is None
        assert sess.checkpoints == []

    def test_session_record_roundtrip(self):
        """SessionRecord to_dict()/from_dict() 往返一致."""
        sess = SessionRecord(
            session_id="sess-002",
            learner_id="learner-002",
            status="paused",
            started_at=5000.0,
            context_envelope={"session_id": "sess-002"},
            checkpoints=[{"seq": 1, "ts": 100.0}],
        )
        restored = SessionRecord.from_dict(sess.to_dict())
        assert restored.session_id == sess.session_id
        assert restored.learner_id == sess.learner_id
        assert restored.status == sess.status
        assert restored.started_at == sess.started_at
        assert restored.context_envelope == sess.context_envelope
        assert restored.checkpoints == sess.checkpoints

    def test_session_record_roundtrip_defaults(self):
        """SessionRecord 默认值往返后保持."""
        sess = SessionRecord(
            session_id="sess-003",
            learner_id="learner-003",
            started_at=1.0,
        )
        restored = SessionRecord.from_dict(sess.to_dict())
        assert restored.status == "active"
        assert restored.context_envelope is None
        assert restored.checkpoints == []


# ============================================================
# 3. 存储层测试
# ============================================================


class TestL2StoreABC:
    """L2Store 抽象基类接口测试."""

    def test_l2_store_is_abstract(self):
        """L2Store 是抽象基类, 不能直接实例化."""
        with pytest.raises(TypeError):
            L2Store()

    def test_l2_store_defines_save_profile(self):
        """L2Store 定义 save_profile 抽象方法."""
        assert "save_profile" in L2Store.__abstractmethods__

    def test_l2_store_defines_get_profile(self):
        """L2Store 定义 get_profile 抽象方法."""
        assert "get_profile" in L2Store.__abstractmethods__

    def test_l2_store_defines_save_answer_history(self):
        """L2Store 定义 save_answer_history 抽象方法."""
        assert "save_answer_history" in L2Store.__abstractmethods__

    def test_l2_store_defines_get_answer_history(self):
        """L2Store 定义 get_answer_history 抽象方法."""
        assert "get_answer_history" in L2Store.__abstractmethods__

    def test_l2_store_defines_save_tracing_state(self):
        """L2Store 定义 save_tracing_state 抽象方法."""
        assert "save_tracing_state" in L2Store.__abstractmethods__

    def test_l2_store_defines_get_tracing_state(self):
        """L2Store 定义 get_tracing_state 抽象方法."""
        assert "get_tracing_state" in L2Store.__abstractmethods__

    def test_l2_store_defines_save_irt_state(self):
        """L2Store 定义 save_irt_state 抽象方法."""
        assert "save_irt_state" in L2Store.__abstractmethods__

    def test_l2_store_defines_get_irt_state(self):
        """L2Store 定义 get_irt_state 抽象方法."""
        assert "get_irt_state" in L2Store.__abstractmethods__

    def test_l2_store_defines_save_session(self):
        """L2Store 定义 save_session 抽象方法."""
        assert "save_session" in L2Store.__abstractmethods__

    def test_l2_store_defines_get_session(self):
        """L2Store 定义 get_session 抽象方法."""
        assert "get_session" in L2Store.__abstractmethods__

    def test_inmemory_store_is_subclass(self):
        """InMemoryL2Store 是 L2Store 子类."""
        assert issubclass(InMemoryL2Store, L2Store)

    def test_inmemory_store_can_instantiate(self):
        """InMemoryL2Store 可实例化 (实现了全部抽象方法)."""
        store = InMemoryL2Store()
        assert store is not None


class TestInMemoryL2Store:
    """InMemoryL2Store 实现测试 — save/get 往返 + 缺失返回 None."""

    def _make_snapshot(self, learner_id="learner-001") -> LearnerSnapshot:
        return LearnerSnapshot(
            learner_id=learner_id,
            snapshot_ts=1000.0,
            kp_mastery={"kp-1": 0.8},
            theta=0.7,
            level="intermediate",
        )

    def _make_answer_history(self) -> list:
        return [
            AnswerRecord(
                learner_id="learner-001", kp_id="kp-1", correct=True, timestamp=1.0
            ),
            AnswerRecord(
                learner_id="learner-001", kp_id="kp-1", correct=False, timestamp=2.0
            ),
        ]

    def _make_tracing_state(self) -> TracingState:
        return TracingState(
            kp_id="kp-001",
            mastery_prob=0.75,
            attempts=3,
            correct_count=2,
            last_attempt_time=1000.0,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
        )

    def _make_irt_state(self) -> IRTState:
        return IRTState(theta=0.8, se=0.25, response_count=10, last_update_time=1000.0)

    def _make_session(self, session_id="sess-001") -> SessionRecord:
        return SessionRecord(
            session_id=session_id,
            learner_id="learner-001",
            started_at=1000.0,
        )

    # --- profile ---

    def test_save_and_get_profile(self):
        """save_profile 后 get_profile 取回正确数据."""
        store = InMemoryL2Store()
        profile = self._make_snapshot()
        store.save_profile("learner-001", profile)
        got = store.get_profile("learner-001")
        assert got is not None
        assert got.learner_id == "learner-001"
        assert got.level == "intermediate"
        assert got.theta == 0.7

    def test_get_profile_missing_returns_none(self):
        """get_profile 不存在时返回 None."""
        store = InMemoryL2Store()
        assert store.get_profile("nonexistent") is None

    # --- answer_history ---

    def test_save_and_get_answer_history(self):
        """save_answer_history 后 get_answer_history 取回正确数据."""
        store = InMemoryL2Store()
        history = self._make_answer_history()
        store.save_answer_history("learner-001", history)
        got = store.get_answer_history("learner-001")
        assert got is not None
        assert len(got) == 2
        assert got[0].correct is True
        assert got[1].correct is False

    def test_get_answer_history_missing_returns_none(self):
        """get_answer_history 不存在时返回 None."""
        store = InMemoryL2Store()
        assert store.get_answer_history("nonexistent") is None

    # --- tracing_state ---

    def test_save_and_get_tracing_state(self):
        """save_tracing_state 后 get_tracing_state 取回正确数据."""
        store = InMemoryL2Store()
        state = self._make_tracing_state()
        store.save_tracing_state("learner-001", "kp-001", state)
        got = store.get_tracing_state("learner-001", "kp-001")
        assert got is not None
        assert got.kp_id == "kp-001"
        assert got.mastery_prob == 0.75
        assert got.attempts == 3

    def test_get_tracing_state_missing_returns_none(self):
        """get_tracing_state 不存在时返回 None."""
        store = InMemoryL2Store()
        assert store.get_tracing_state("nonexistent", "kp-x") is None

    def test_get_tracing_state_different_kp_isolated(self):
        """同一 learner 不同 kp 的 tracing_state 互不干扰."""
        store = InMemoryL2Store()
        s1 = TracingState(
            kp_id="kp-1",
            mastery_prob=0.5,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
        )
        s2 = TracingState(
            kp_id="kp-2",
            mastery_prob=0.9,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
        )
        store.save_tracing_state("learner-001", "kp-1", s1)
        store.save_tracing_state("learner-001", "kp-2", s2)
        assert store.get_tracing_state("learner-001", "kp-1").mastery_prob == 0.5
        assert store.get_tracing_state("learner-001", "kp-2").mastery_prob == 0.9

    # --- irt_state ---

    def test_save_and_get_irt_state(self):
        """save_irt_state 后 get_irt_state 取回正确数据."""
        store = InMemoryL2Store()
        state = self._make_irt_state()
        store.save_irt_state("learner-001", state)
        got = store.get_irt_state("learner-001")
        assert got is not None
        assert got.theta == 0.8
        assert got.response_count == 10

    def test_get_irt_state_missing_returns_none(self):
        """get_irt_state 不存在时返回 None."""
        store = InMemoryL2Store()
        assert store.get_irt_state("nonexistent") is None

    # --- session ---

    def test_save_and_get_session(self):
        """save_session 后 get_session 取回正确数据."""
        store = InMemoryL2Store()
        sess = self._make_session()
        store.save_session("sess-001", sess)
        got = store.get_session("sess-001")
        assert got is not None
        assert got.session_id == "sess-001"
        assert got.learner_id == "learner-001"
        assert got.status == "active"

    def test_get_session_missing_returns_none(self):
        """get_session 不存在时返回 None."""
        store = InMemoryL2Store()
        assert store.get_session("nonexistent") is None

    # --- 覆盖更新 ---

    def test_save_profile_overwrites(self):
        """重复 save_profile 覆盖旧数据."""
        store = InMemoryL2Store()
        store.save_profile("learner-001", self._make_snapshot())
        updated = LearnerSnapshot(
            learner_id="learner-001",
            snapshot_ts=2000.0,
            kp_mastery={"kp-1": 0.95},
            theta=0.9,
            level="advanced",
        )
        store.save_profile("learner-001", updated)
        got = store.get_profile("learner-001")
        assert got.level == "advanced"
        assert got.theta == 0.9


class TestL2StoreThreadSafety:
    """InMemoryL2Store 线程安全测试 (threading.RLock)."""

    def test_concurrent_save_get_profile(self):
        """并发 save/get profile 不抛异常且数据完整."""
        store = InMemoryL2Store()
        errors = []

        def worker(idx):
            try:
                lid = f"learner-{idx:03d}"
                profile = LearnerSnapshot(
                    learner_id=lid,
                    snapshot_ts=float(idx),
                    kp_mastery={"kp-1": 0.5},
                    theta=float(idx) / 10,
                    level="novice",
                )
                store.save_profile(lid, profile)
                got = store.get_profile(lid)
                assert got is not None
                assert got.learner_id == lid
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        for i in range(20):
            assert store.get_profile(f"learner-{i:03d}") is not None

    def test_concurrent_save_get_session(self):
        """并发 save/get session 不抛异常且数据完整."""
        store = InMemoryL2Store()
        errors = []

        def worker(idx):
            try:
                sid = f"sess-{idx:03d}"
                sess = SessionRecord(
                    session_id=sid,
                    learner_id=f"learner-{idx}",
                    started_at=float(idx),
                )
                store.save_session(sid, sess)
                got = store.get_session(sid)
                assert got is not None
                assert got.session_id == sid
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        for i in range(20):
            assert store.get_session(f"sess-{i:03d}") is not None

    def test_concurrent_mixed_operations(self):
        """并发混合操作 (profile + tracing + irt) 不抛异常."""
        store = InMemoryL2Store()
        errors = []

        def worker(idx):
            try:
                lid = f"learner-{idx:03d}"
                store.save_profile(
                    lid,
                    LearnerSnapshot(
                        learner_id=lid,
                        snapshot_ts=float(idx),
                        kp_mastery={},
                        theta=None,
                        level="novice",
                    ),
                )
                store.save_tracing_state(
                    lid,
                    f"kp-{idx}",
                    TracingState(
                        kp_id=f"kp-{idx}",
                        mastery_prob=0.5,
                        bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
                    ),
                )
                store.save_irt_state(lid, IRTState(theta=float(idx) / 10))
                assert store.get_profile(lid) is not None
                assert store.get_tracing_state(lid, f"kp-{idx}") is not None
                assert store.get_irt_state(lid) is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================
# 4. 缓存层测试
# ============================================================


class TestL2Cache:
    """L2Cache 分层 TTL + write-through 语义 + clear() 测试."""

    def test_cache_default_ttls(self):
        """L2Cache 默认分层 TTL: profile=300, bkt=30, memory=600."""
        cache = L2Cache()
        assert cache.profile_ttl == 300
        assert cache.bkt_ttl == 30
        assert cache.memory_ttl == 600

    def test_cache_custom_ttls(self):
        """L2Cache 自定义分层 TTL."""
        cache = L2Cache(profile_ttl=100, bkt_ttl=20, memory_ttl=200)
        assert cache.profile_ttl == 100
        assert cache.bkt_ttl == 20
        assert cache.memory_ttl == 200

    def test_cache_set_and_get(self):
        """set 后 get 返回写入的值."""
        cache = L2Cache()
        cache.set("key-1", "value-1", layer="profile")
        assert cache.get("key-1") == "value-1"

    def test_cache_get_missing_returns_none(self):
        """get 不存在的 key 返回 None."""
        cache = L2Cache()
        assert cache.get("nonexistent") is None

    def test_cache_set_different_layers(self):
        """不同 layer (profile/bkt/memory) 均可 set/get."""
        cache = L2Cache()
        cache.set("pk", "pv", layer="profile")
        cache.set("bk", "bv", layer="bkt")
        cache.set("mk", "mv", layer="memory")
        assert cache.get("pk") == "pv"
        assert cache.get("bk") == "bv"
        assert cache.get("mk") == "mv"

    def test_cache_overwrite_key(self):
        """重复 set 同一 key 覆盖旧值."""
        cache = L2Cache()
        cache.set("key-1", "old", layer="profile")
        cache.set("key-1", "new", layer="profile")
        assert cache.get("key-1") == "new"

    def test_cache_ttl_expiry(self):
        """TTL 过期后 get 返回 None (无 backing store)."""
        cache = L2Cache(profile_ttl=0.05)
        cache.set("key-1", "value-1", layer="profile")
        assert cache.get("key-1") == "value-1"
        time.sleep(0.1)
        assert cache.get("key-1") is None

    def test_cache_layered_ttl_expiry(self):
        """分层 TTL: bkt 先过期, profile 仍有效."""
        cache = L2Cache(profile_ttl=10, bkt_ttl=0.05, memory_ttl=10)
        cache.set("profile-key", "pv", layer="profile")
        cache.set("bkt-key", "bv", layer="bkt")
        assert cache.get("profile-key") == "pv"
        assert cache.get("bkt-key") == "bv"
        time.sleep(0.1)
        # bkt 已过期
        assert cache.get("bkt-key") is None
        # profile 仍有效
        assert cache.get("profile-key") == "pv"

    def test_cache_write_through_to_backing(self):
        """write-through: set 同时写入 backing store (真实 dict)."""
        backing: dict = {}
        cache = L2Cache(backing_store=backing)
        cache.set("key-1", "value-1", layer="profile")
        # backing store 应包含写入的数据
        assert backing.get("key-1") == "value-1"
        # cache 也能取到
        assert cache.get("key-1") == "value-1"

    def test_cache_write_through_multiple_keys(self):
        """write-through: 多个 key 均写入 backing store."""
        backing: dict = {}
        cache = L2Cache(backing_store=backing)
        cache.set("k1", "v1", layer="profile")
        cache.set("k2", "v2", layer="bkt")
        assert backing.get("k1") == "v1"
        assert backing.get("k2") == "v2"

    def test_cache_clear(self):
        """clear() 清空缓存 (无 backing 时 get 返回 None)."""
        cache = L2Cache()
        cache.set("key-1", "value-1", layer="profile")
        cache.set("key-2", "value-2", layer="bkt")
        cache.clear()
        assert cache.get("key-1") is None
        assert cache.get("key-2") is None

    def test_cache_recovery_from_backing_after_clear(self):
        """clear 后, get 从 backing store 恢复数据 (write-through 持久性)."""
        backing: dict = {}
        cache = L2Cache(backing_store=backing, profile_ttl=10)
        cache.set("key-1", "value-1", layer="profile")
        assert backing.get("key-1") == "value-1"
        # 清空缓存层 (backing store 保留)
        cache.clear()
        # 清空后 get 应从 backing store 恢复
        assert cache.get("key-1") == "value-1"

    def test_cache_clear_does_not_clear_backing(self):
        """clear() 只清缓存层, 不清 backing store."""
        backing: dict = {}
        cache = L2Cache(backing_store=backing)
        cache.set("key-1", "value-1", layer="profile")
        cache.clear()
        # backing store 数据仍在
        assert backing.get("key-1") == "value-1"
