"""L2 个性化会话管理器测试 — SessionManager 生命周期 / 检查点 / 活跃会话查询.

测试覆盖 (TDD, 先写测试再实现):
1. start_session: 创建会话, 返回 session_id (UUID, "sess-" 前缀), 自动保存到 store
   - goal 参数写入 context_envelope
   - started_at = time.time(), status = "active", checkpoints = []
2. get_session: 获取会话, 不存在返回 None
3. end_session: 结束会话, 返回摘要 {session_id, learner_id, duration, status: "closed"}
   - 设置 status="closed", duration = now - started_at
4. pause_session / resume_session: 状态机 active <-> paused
5. add_checkpoint: 追加检查点到 checkpoints 列表
   - 检查点包含 SHA-256 哈希用于完整性验证 (Claude Science: Checkpoint 机制)
   - 不修改调用方传入的 dict
6. get_active_sessions: 过滤 learner_id 匹配且 status="active" 的会话
7. 错误处理: 操作不存在的 session_id 抛出 StoreError
8. 依赖注入: store=None 时内部创建 InMemoryL2Store; 注入 store 时使用注入实例
9. 线程安全: 并发 start_session 生成唯一 ID, 并发混合操作不抛异常
"""

import threading
import time

import pytest

from dy3_polaris.l2.exceptions import StoreError
from dy3_polaris.l2.models import SessionRecord
from dy3_polaris.l2.session import SessionManager
from dy3_polaris.l2.store import InMemoryL2Store, L2Store


# ============================================================
# 1. start_session 测试
# ============================================================


class TestSessionManagerStart:
    """SessionManager.start_session 测试 — 创建会话 / session_id 格式 / 自动保存."""

    def test_start_session_returns_str(self):
        """start_session 返回 str 类型的 session_id."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        assert isinstance(sid, str)

    def test_start_session_id_has_sess_prefix(self):
        """session_id 以 "l2s-" 前缀开头 (统一命名空间: L2 学习会话)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        assert sid.startswith("l2s-")

    def test_start_session_id_has_uuid_hex_suffix(self):
        """session_id 后缀为 UUID hex (12 个十六进制字符, 全系统统一)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        suffix = sid[len("l2s-"):]
        assert len(suffix) == 12
        # 全部为十六进制字符
        int(suffix, 16)

    def test_start_session_generates_unique_ids(self):
        """多次 start_session 生成互不相同的 session_id."""
        mgr = SessionManager()
        sid1 = mgr.start_session("learner-001")
        sid2 = mgr.start_session("learner-001")
        sid3 = mgr.start_session("learner-001")
        assert sid1 != sid2
        assert sid2 != sid3
        assert sid1 != sid3

    def test_start_session_creates_active_session(self):
        """start_session 创建 status="active" 的会话."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.get_session(sid)
        assert sess is not None
        assert sess.status == "active"
        assert sess.learner_id == "learner-001"
        assert sess.session_id == sid

    def test_start_session_sets_started_at(self):
        """start_session 设置 started_at 为当前时间戳."""
        mgr = SessionManager()
        before = time.time()
        sid = mgr.start_session("learner-001")
        after = time.time()
        sess = mgr.get_session(sid)
        assert sess is not None
        assert before <= sess.started_at <= after

    def test_start_session_empty_checkpoints(self):
        """start_session 创建的会话 checkpoints 为空列表."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.get_session(sid)
        assert sess.checkpoints == []

    def test_start_session_auto_saves_to_store(self):
        """start_session 自动将会话保存到 store (依赖注入验证)."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        # 注入的 store 应包含该会话
        assert store.get_session(sid) is not None
        assert store.get_session(sid).learner_id == "learner-001"

    def test_start_session_with_goal_sets_context(self):
        """start_session 携带 goal 时写入 context_envelope["goal"]."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001", goal="掌握线性代数")
        sess = mgr.get_session(sid)
        assert sess is not None
        assert sess.context_envelope is not None
        assert sess.context_envelope["goal"] == "掌握线性代数"

    def test_start_session_without_goal_context_none(self):
        """start_session 不带 goal 时 context_envelope 为 None."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.get_session(sid)
        assert sess is not None
        assert sess.context_envelope is None

    def test_start_session_goal_default_empty(self):
        """start_session goal 默认参数为空字符串, 不写入 context_envelope."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.get_session(sid)
        assert sess.context_envelope is None


# ============================================================
# 2. get_session 测试
# ============================================================


class TestSessionManagerGet:
    """SessionManager.get_session 测试 — 获取会话 / 不存在返回 None."""

    def test_get_session_returns_record(self):
        """get_session 返回已创建的 SessionRecord."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.get_session(sid)
        assert sess is not None
        assert isinstance(sess, SessionRecord)
        assert sess.session_id == sid

    def test_get_session_nonexistent_returns_none(self):
        """get_session 不存在的 session_id 返回 None."""
        mgr = SessionManager()
        assert mgr.get_session("sess-nonexistent") is None

    def test_get_session_returns_same_learner_id(self):
        """get_session 返回的 learner_id 与创建时一致."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-xyz")
        sess = mgr.get_session(sid)
        assert sess.learner_id == "learner-xyz"


# ============================================================
# 3. end_session 测试
# ============================================================


class TestSessionManagerEnd:
    """SessionManager.end_session 测试 — 结束会话 / 摘要字典 / 持续时间."""

    def test_end_session_returns_summary_dict(self):
        """end_session 返回包含 session_id/learner_id/duration/status 的摘要."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        summary = mgr.end_session(sid)
        assert isinstance(summary, dict)
        assert summary["session_id"] == sid
        assert summary["learner_id"] == "learner-001"
        assert summary["status"] == "closed"

    def test_end_session_summary_has_duration(self):
        """end_session 摘要包含 duration (float, >= 0)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        summary = mgr.end_session(sid)
        assert "duration" in summary
        assert isinstance(summary["duration"], float)
        assert summary["duration"] >= 0.0

    def test_end_session_sets_status_closed(self):
        """end_session 将会话 status 设置为 "closed"."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.end_session(sid)
        sess = mgr.get_session(sid)
        assert sess.status == "closed"

    def test_end_session_duration_is_elapsed_time(self):
        """end_session duration 约等于 now - started_at."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        started = mgr.get_session(sid).started_at
        time.sleep(0.05)
        now = time.time()
        summary = mgr.end_session(sid)
        expected = now - started
        # 允许一定误差
        assert abs(summary["duration"] - expected) < 1.0

    def test_end_session_persists_closed_status(self):
        """end_session 后 store 中会话状态持久化为 closed."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        mgr.end_session(sid)
        assert store.get_session(sid).status == "closed"


# ============================================================
# 4. pause_session / resume_session 测试
# ============================================================


class TestSessionManagerPauseResume:
    """SessionManager.pause_session / resume_session 测试 — 状态机 active <-> paused."""

    def test_pause_session_sets_paused(self):
        """pause_session 将 status 设置为 "paused"."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.pause_session(sid)
        assert sess.status == "paused"

    def test_pause_session_persists_paused(self):
        """pause_session 后 store 中状态为 paused."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        assert mgr.get_session(sid).status == "paused"

    def test_pause_session_returns_session_record(self):
        """pause_session 返回 SessionRecord."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.pause_session(sid)
        assert isinstance(sess, SessionRecord)
        assert sess.session_id == sid

    def test_resume_session_sets_active(self):
        """resume_session 将 status 设置为 "active"."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        sess = mgr.resume_session(sid)
        assert sess.status == "active"

    def test_resume_session_persists_active(self):
        """resume_session 后 store 中状态为 active."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        mgr.resume_session(sid)
        assert mgr.get_session(sid).status == "active"

    def test_resume_session_returns_session_record(self):
        """resume_session 返回 SessionRecord."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        sess = mgr.resume_session(sid)
        assert isinstance(sess, SessionRecord)
        assert sess.session_id == sid

    def test_pause_resume_cycle(self):
        """pause/resume 可多次循环 (active <-> paused)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        for _ in range(3):
            mgr.pause_session(sid)
            assert mgr.get_session(sid).status == "paused"
            mgr.resume_session(sid)
            assert mgr.get_session(sid).status == "active"

    def test_paused_session_not_in_active(self):
        """paused 状态的会话不出现在 get_active_sessions 中."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        active = mgr.get_active_sessions("learner-001")
        assert all(s.session_id != sid for s in active)


# ============================================================
# 5. add_checkpoint 测试
# ============================================================


class TestSessionManagerCheckpoint:
    """SessionManager.add_checkpoint 测试 — 检查点追加 / SHA-256 完整性 / seq 递增."""

    def test_add_checkpoint_returns_session_record(self):
        """add_checkpoint 返回更新后的 SessionRecord."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.add_checkpoint(sid, {"step": 1})
        assert isinstance(sess, SessionRecord)
        assert sess.session_id == sid

    def test_add_checkpoint_appends_to_list(self):
        """add_checkpoint 将检查点追加到 checkpoints 列表."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        mgr.add_checkpoint(sid, {"step": 2})
        sess = mgr.get_session(sid)
        assert len(sess.checkpoints) == 2

    def test_add_checkpoint_preserves_original_fields(self):
        """add_checkpoint 保留调用方传入的检查点字段."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1, "note": "首检"})
        cp = mgr.get_session(sid).checkpoints[0]
        assert cp["step"] == 1
        assert cp["note"] == "首检"

    def test_add_checkpoint_includes_sha256(self):
        """检查点包含 SHA-256 哈希 (64 位十六进制) 用于完整性验证."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        cp = mgr.get_session(sid).checkpoints[0]
        assert "sha256" in cp
        assert len(cp["sha256"]) == 64
        int(cp["sha256"], 16)  # 合法十六进制

    def test_add_checkpoint_sha256_differs_for_different_content(self):
        """不同内容的检查点产生不同的 SHA-256 哈希."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        mgr.add_checkpoint(sid, {"step": 2})
        cps = mgr.get_session(sid).checkpoints
        assert cps[0]["sha256"] != cps[1]["sha256"]

    def test_add_checkpoint_seq_increments(self):
        """检查点 seq 从 0 开始递增."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        mgr.add_checkpoint(sid, {"step": 2})
        mgr.add_checkpoint(sid, {"step": 3})
        cps = mgr.get_session(sid).checkpoints
        assert cps[0]["seq"] == 0
        assert cps[1]["seq"] == 1
        assert cps[2]["seq"] == 2

    def test_add_checkpoint_includes_timestamp(self):
        """检查点包含 ts 时间戳."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        before = time.time()
        mgr.add_checkpoint(sid, {"step": 1})
        after = time.time()
        cp = mgr.get_session(sid).checkpoints[0]
        assert "ts" in cp
        assert before <= cp["ts"] <= after

    def test_add_checkpoint_does_not_mutate_input(self):
        """add_checkpoint 不修改调用方传入的 dict."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        original = {"step": 1, "note": "原始"}
        snapshot = dict(original)
        mgr.add_checkpoint(sid, original)
        # 调用方的 dict 应保持不变
        assert original == snapshot

    def test_add_checkpoint_persists_to_store(self):
        """add_checkpoint 将检查点持久化到 store."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        mgr.add_checkpoint(sid, {"step": 2})
        persisted = store.get_session(sid)
        assert len(persisted.checkpoints) == 2


# ============================================================
# 6. get_active_sessions 测试
# ============================================================


class TestSessionManagerActiveSessions:
    """SessionManager.get_active_sessions 测试 — 过滤 learner_id 匹配且 status=active."""

    def test_get_active_sessions_returns_active_only(self):
        """get_active_sessions 仅返回 status="active" 的会话."""
        mgr = SessionManager()
        sid1 = mgr.start_session("learner-001")
        sid2 = mgr.start_session("learner-001")
        sid3 = mgr.start_session("learner-001")
        mgr.pause_session(sid2)
        mgr.end_session(sid3)
        active = mgr.get_active_sessions("learner-001")
        assert len(active) == 1
        assert active[0].session_id == sid1
        assert all(s.status == "active" for s in active)

    def test_get_active_sessions_returns_list(self):
        """get_active_sessions 返回 list 类型."""
        mgr = SessionManager()
        mgr.start_session("learner-001")
        result = mgr.get_active_sessions("learner-001")
        assert isinstance(result, list)

    def test_get_active_sessions_multiple_active(self):
        """同一学习者多个活跃会话全部返回."""
        mgr = SessionManager()
        mgr.start_session("learner-001")
        mgr.start_session("learner-001")
        mgr.start_session("learner-001")
        active = mgr.get_active_sessions("learner-001")
        assert len(active) == 3

    def test_get_active_sessions_isolates_learners(self):
        """get_active_sessions 按 learner_id 隔离, 不返回其他学习者会话."""
        mgr = SessionManager()
        mgr.start_session("learner-001")
        mgr.start_session("learner-001")
        mgr.start_session("learner-002")
        active1 = mgr.get_active_sessions("learner-001")
        active2 = mgr.get_active_sessions("learner-002")
        assert len(active1) == 2
        assert len(active2) == 1
        assert all(s.learner_id == "learner-001" for s in active1)
        assert all(s.learner_id == "learner-002" for s in active2)

    def test_get_active_sessions_empty_for_unknown_learner(self):
        """未知学习者的 get_active_sessions 返回空列表."""
        mgr = SessionManager()
        mgr.start_session("learner-001")
        assert mgr.get_active_sessions("nobody") == []

    def test_get_active_sessions_empty_when_all_closed(self):
        """所有会话关闭后 get_active_sessions 返回空列表."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.end_session(sid)
        assert mgr.get_active_sessions("learner-001") == []

    def test_get_active_sessions_all_are_session_records(self):
        """get_active_sessions 返回的元素均为 SessionRecord."""
        mgr = SessionManager()
        mgr.start_session("learner-001")
        active = mgr.get_active_sessions("learner-001")
        assert all(isinstance(s, SessionRecord) for s in active)


# ============================================================
# 7. 错误处理测试 (StoreError)
# ============================================================


class TestSessionManagerStoreError:
    """SessionManager 错误处理 — 操作不存在的 session 抛出 StoreError."""

    def test_end_session_nonexistent_raises_store_error(self):
        """end_session 不存在的 session_id 抛出 StoreError."""
        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.end_session("sess-nonexistent")

    def test_pause_session_nonexistent_raises_store_error(self):
        """pause_session 不存在的 session_id 抛出 StoreError."""
        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.pause_session("sess-nonexistent")

    def test_resume_session_nonexistent_raises_store_error(self):
        """resume_session 不存在的 session_id 抛出 StoreError."""
        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.resume_session("sess-nonexistent")

    def test_add_checkpoint_nonexistent_raises_store_error(self):
        """add_checkpoint 不存在的 session_id 抛出 StoreError."""
        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.add_checkpoint("sess-nonexistent", {"step": 1})

    def test_store_error_is_l2_error(self):
        """StoreError 可作为 L2 异常体系的一部分被捕获."""
        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.end_session("sess-nonexistent")

    def test_store_error_context_contains_session_id(self):
        """StoreError 的 context 包含 session_id 信息."""
        mgr = SessionManager()
        with pytest.raises(StoreError) as exc_info:
            mgr.end_session("sess-abc123")
        err = exc_info.value
        assert "session_id" in err.context
        assert err.context["session_id"] == "sess-abc123"


# ============================================================
# 8. 依赖注入测试
# ============================================================


class TestSessionManagerDependencyInjection:
    """SessionManager 依赖注入测试 — store=None / 注入 store."""

    def test_default_store_is_inmemory_l2_store(self):
        """无参构造时内部创建 InMemoryL2Store."""
        mgr = SessionManager()
        assert isinstance(mgr.store, InMemoryL2Store)

    def test_store_none_creates_internal_store(self):
        """store=None 时内部创建 store."""
        mgr = SessionManager(store=None)
        assert mgr.store is not None
        assert isinstance(mgr.store, L2Store)

    def test_injected_store_is_used(self):
        """注入的 store 被用于会话持久化."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        assert store.get_session(sid) is not None
        assert mgr.store is store

    def test_injected_store_reflects_mutations(self):
        """注入 store 上的状态变更对 manager 可见."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        assert store.get_session(sid).status == "paused"
        assert mgr.get_session(sid).status == "paused"

    def test_store_property_exposes_store(self):
        """store 属性暴露内部存储实例."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        assert mgr.store is store


# ============================================================
# 9. 线程安全测试
# ============================================================


class TestSessionManagerThreadSafety:
    """SessionManager 线程安全测试 (threading.RLock, 参考 store.py 模式)."""

    def test_concurrent_start_generates_unique_ids(self):
        """并发 start_session 生成互不相同的 session_id."""
        mgr = SessionManager()
        ids: list[str] = []
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                sid = mgr.start_session(f"learner-{idx:03d}")
                ids.append(sid)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(ids) == 30
        assert len(set(ids)) == 30  # 全部唯一

    def test_concurrent_mixed_operations_no_error(self):
        """并发混合操作 (start/pause/resume/end) 不抛异常."""
        mgr = SessionManager()
        errors: list[Exception] = []

        # 预创建一批会话
        sids = [mgr.start_session("learner-001") for _ in range(10)]

        def worker(sid: str) -> None:
            try:
                mgr.pause_session(sid)
                mgr.add_checkpoint(sid, {"step": 1})
                mgr.resume_session(sid)
                mgr.add_checkpoint(sid, {"step": 2})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(sid,)) for sid in sids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        for sid in sids:
            sess = mgr.get_session(sid)
            assert sess is not None
            assert len(sess.checkpoints) == 2

    def test_concurrent_get_active_sessions(self):
        """并发 get_active_sessions 与 start_session 不抛异常且结果一致."""
        mgr = SessionManager()
        errors: list[Exception] = []

        def starter() -> None:
            try:
                for _ in range(5):
                    mgr.start_session("learner-shared")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(5):
                    active = mgr.get_active_sessions("learner-shared")
                    assert isinstance(active, list)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=starter) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 4 个 starter 各创建 5 个, 全部 active
        assert len(mgr.get_active_sessions("learner-shared")) == 20


# ============================================================
# 10. fork_session 测试 (Claude Science: Session Fork)
# ============================================================


class TestSessionManagerFork:
    """SessionManager.fork_session 测试 — 会话分叉 / 状态继承 / 分叉元数据."""

    def test_fork_session_returns_session_record(self):
        """fork_session 返回新的 SessionRecord."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001", goal="掌握微积分")
        forked = mgr.fork_session(sid)
        assert isinstance(forked, SessionRecord)

    def test_fork_session_new_id_differs_from_source(self):
        """分叉会话的 session_id 与源会话不同."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        assert forked.session_id != sid

    def test_fork_session_status_active(self):
        """分叉会话状态为 'active'."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        assert forked.status == "active"

    def test_fork_session_inherits_learner_id(self):
        """分叉会话继承源会话的 learner_id."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-fork")
        forked = mgr.fork_session(sid)
        assert forked.learner_id == "learner-fork"

    def test_fork_session_inherits_context_envelope(self):
        """分叉会话继承源会话的 context_envelope (含 goal)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001", goal="掌握线性代数")
        forked = mgr.fork_session(sid)
        assert forked.context_envelope is not None
        assert forked.context_envelope["goal"] == "掌握线性代数"

    def test_fork_session_inherits_checkpoints(self):
        """分叉会话继承源会话的检查点列表."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        mgr.add_checkpoint(sid, {"step": 2})
        forked = mgr.fork_session(sid)
        assert len(forked.checkpoints) == 2

    def test_fork_session_records_fork_metadata(self):
        """分叉会话的 context_envelope 记录分叉元数据."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001", goal="G")
        forked = mgr.fork_session(sid)
        env = forked.context_envelope
        assert env is not None
        assert "fork_reason" in env
        assert "forked_from" in env
        assert "forked_at" in env

    def test_fork_session_forked_from_is_source(self):
        """forked_from 元数据指向源会话 ID."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        assert forked.context_envelope["forked_from"] == sid

    def test_fork_session_default_fork_reason(self):
        """默认 fork_reason 为 'branch'."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        assert forked.context_envelope["fork_reason"] == "branch"

    def test_fork_session_custom_fork_reason(self):
        """自定义 fork_reason 被记录."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid, fork_reason="experiment")
        assert forked.context_envelope["fork_reason"] == "experiment"

    def test_fork_session_branch_label_recorded(self):
        """branch_label 被记录到分叉元数据."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid, branch_label="exp-A")
        assert forked.context_envelope["branch_label"] == "exp-A"

    def test_fork_session_branch_label_default_none(self):
        """branch_label 默认为 None."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        assert forked.context_envelope.get("branch_label") is None

    def test_fork_session_forked_at_is_timestamp(self):
        """forked_at 为有效时间戳 (float)."""
        mgr = SessionManager()
        before = time.time()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        after = time.time()
        assert isinstance(forked.context_envelope["forked_at"], float)
        assert before <= forked.context_envelope["forked_at"] <= after

    def test_fork_session_persists_to_store(self):
        """分叉会话持久化到 store."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        assert store.get_session(forked.session_id) is not None
        assert store.get_session(forked.session_id).status == "active"

    def test_fork_session_registered_in_learner_index(self):
        """分叉会话注册到 learner 索引, 出现在活跃会话中."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        active = mgr.get_active_sessions("learner-001")
        assert any(s.session_id == forked.session_id for s in active)

    def test_fork_session_nonexistent_raises_store_error(self):
        """fork_session 不存在的 session 抛出 StoreError."""
        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.fork_session("sess-nonexistent")

    def test_fork_session_does_not_mutate_source(self):
        """fork_session 不修改源会话."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001", goal="G")
        source_before = mgr.get_session(sid)
        source_cp_count = len(source_before.checkpoints)
        source_env = dict(source_before.context_envelope)
        mgr.fork_session(sid)
        source_after = mgr.get_session(sid)
        assert source_after.status == source_before.status
        assert len(source_after.checkpoints) == source_cp_count
        assert source_after.context_envelope == source_env

    def test_fork_session_from_source_without_context(self):
        """源会话无 context_envelope 时分叉仍可创建并记录元数据."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")  # 无 goal -> context None
        assert mgr.get_session(sid).context_envelope is None
        forked = mgr.fork_session(sid)
        assert forked.context_envelope is not None
        assert forked.context_envelope["forked_from"] == sid


# ============================================================
# 11. restore_checkpoint 测试 (Temporal: 检查点恢复)
# ============================================================


class TestSessionManagerRestoreCheckpoint:
    """SessionManager.restore_checkpoint 测试 — 检查点恢复 / 截断."""

    def test_restore_checkpoint_returns_session_record(self):
        """restore_checkpoint 返回 SessionRecord."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        mgr.add_checkpoint(sid, {"step": 2})
        restored = mgr.restore_checkpoint(sid, 0)
        assert isinstance(restored, SessionRecord)
        assert restored.session_id == sid

    def test_restore_checkpoint_truncates_after(self):
        """恢复到 seq=1 后, seq>1 的检查点被截断."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})  # seq 0
        mgr.add_checkpoint(sid, {"step": 2})  # seq 1
        mgr.add_checkpoint(sid, {"step": 3})  # seq 2
        mgr.add_checkpoint(sid, {"step": 4})  # seq 3
        restored = mgr.restore_checkpoint(sid, 1)
        seqs = [cp["seq"] for cp in restored.checkpoints]
        assert seqs == [0, 1]

    def test_restore_checkpoint_keeps_target_and_before(self):
        """恢复后保留目标检查点及其之前的所有检查点."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        for i in range(5):
            mgr.add_checkpoint(sid, {"step": i})
        mgr.restore_checkpoint(sid, 2)
        sess = mgr.get_session(sid)
        seqs = [cp["seq"] for cp in sess.checkpoints]
        assert seqs == [0, 1, 2]

    def test_restore_checkpoint_to_first(self):
        """恢复到 seq=0 只保留第一个检查点."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        mgr.add_checkpoint(sid, {"step": 2})
        mgr.restore_checkpoint(sid, 0)
        sess = mgr.get_session(sid)
        assert len(sess.checkpoints) == 1
        assert sess.checkpoints[0]["seq"] == 0

    def test_restore_checkpoint_invalid_seq_raises_value_error(self):
        """恢复到不存在的 seq 抛出 ValueError."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        with pytest.raises(ValueError):
            mgr.restore_checkpoint(sid, 99)

    def test_restore_checkpoint_persists_truncation(self):
        """截断结果持久化到 store."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})
        mgr.add_checkpoint(sid, {"step": 2})
        mgr.add_checkpoint(sid, {"step": 3})
        mgr.restore_checkpoint(sid, 0)
        persisted = store.get_session(sid)
        assert len(persisted.checkpoints) == 1

    def test_restore_checkpoint_nonexistent_session_raises(self):
        """restore_checkpoint 不存在的 session 抛出 StoreError."""
        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.restore_checkpoint("sess-nonexistent", 0)

    def test_restore_checkpoint_no_checkpoints_invalid_seq_raises(self):
        """无检查点时恢复任意 seq 抛出 ValueError."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        with pytest.raises(ValueError):
            mgr.restore_checkpoint(sid, 0)


# ============================================================
# 12. merge_fork 测试 (Claude Science: Fork 合并)
# ============================================================


class TestSessionManagerMergeFork:
    """SessionManager.merge_fork 测试 — 分叉合并 / 标记 merged."""

    def test_merge_fork_marks_fork_as_merged(self):
        """merge_fork 将分叉会话标记为 'merged'."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        mgr.merge_fork(forked.session_id, sid)
        assert mgr.get_session(forked.session_id).status == "merged"

    def test_merge_fork_target_has_merged_checkpoints(self):
        """合并后目标会话包含分叉会话的检查点."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})  # target 1 checkpoint
        forked = mgr.fork_session(sid)
        mgr.add_checkpoint(forked.session_id, {"step": 2})
        mgr.add_checkpoint(forked.session_id, {"step": 3})
        mgr.merge_fork(forked.session_id, sid)
        target = mgr.get_session(sid)
        # target 原有 1 + fork 新增 2 = 3
        assert len(target.checkpoints) == 3

    def test_merge_fork_returns_target_session_record(self):
        """merge_fork 返回目标会话的 SessionRecord."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        result = mgr.merge_fork(forked.session_id, sid)
        assert isinstance(result, SessionRecord)
        assert result.session_id == sid

    def test_merge_fork_persists_target(self):
        """合并结果持久化到目标会话."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        mgr.add_checkpoint(forked.session_id, {"step": 9})
        mgr.merge_fork(forked.session_id, sid)
        assert store.get_session(sid) is not None
        assert len(store.get_session(sid).checkpoints) >= 1

    def test_merge_fork_persists_fork_merged_status(self):
        """分叉会话的 merged 状态持久化到 store."""
        store = InMemoryL2Store()
        mgr = SessionManager(store=store)
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        mgr.merge_fork(forked.session_id, sid)
        assert store.get_session(forked.session_id).status == "merged"

    def test_merge_fork_nonexistent_fork_raises_store_error(self):
        """merge_fork 不存在的 fork session 抛出 StoreError."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        with pytest.raises(StoreError):
            mgr.merge_fork("sess-nonexistent", sid)

    def test_merge_fork_nonexistent_target_raises_store_error(self):
        """merge_fork 不存在的 target session 抛出 StoreError."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        with pytest.raises(StoreError):
            mgr.merge_fork(forked.session_id, "sess-nonexistent")

    def test_merge_fork_renumbers_appended_checkpoints(self):
        """合并后追加的检查点 seq 连续递增 (从目标原有数量续编)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.add_checkpoint(sid, {"step": 1})  # target seq 0
        mgr.add_checkpoint(sid, {"step": 2})  # target seq 1
        forked = mgr.fork_session(sid)
        mgr.add_checkpoint(forked.session_id, {"step": 3})  # fork seq 2
        mgr.merge_fork(forked.session_id, sid)
        target = mgr.get_session(sid)
        seqs = [cp["seq"] for cp in target.checkpoints]
        assert seqs == [0, 1, 2]


# ============================================================
# 13. 状态转移校验测试 (Activity 状态机)
# ============================================================


class TestSessionManagerStateTransition:
    """SessionManager 状态转移校验 — 非法转移抛出 ValueError."""

    def test_valid_transitions_defined(self):
        """合法转移集合包含 created->active 等五条转移."""
        # 合法转移: created->active, active->paused, paused->active,
        #           active->closed, paused->closed
        assert SessionManager.is_valid_transition("created", "active")
        assert SessionManager.is_valid_transition("active", "paused")
        assert SessionManager.is_valid_transition("paused", "active")
        assert SessionManager.is_valid_transition("active", "closed")
        assert SessionManager.is_valid_transition("paused", "closed")

    def test_is_valid_transition_invalid_cases(self):
        """非法转移返回 False."""
        assert not SessionManager.is_valid_transition("closed", "active")
        assert not SessionManager.is_valid_transition("closed", "paused")
        assert not SessionManager.is_valid_transition("active", "active")
        assert not SessionManager.is_valid_transition("paused", "paused")
        assert not SessionManager.is_valid_transition("created", "closed")

    def test_pause_from_paused_raises_value_error(self):
        """对已暂停会话再次暂停 -> ValueError (paused->paused 非法)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        with pytest.raises(ValueError):
            mgr.pause_session(sid)

    def test_pause_from_closed_raises_value_error(self):
        """对已关闭会话暂停 -> ValueError."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.end_session(sid)
        with pytest.raises(ValueError):
            mgr.pause_session(sid)

    def test_resume_from_active_raises_value_error(self):
        """对活跃会话恢复 -> ValueError (active->active 非法)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        with pytest.raises(ValueError):
            mgr.resume_session(sid)

    def test_resume_from_closed_raises_value_error(self):
        """对已关闭会话恢复 -> ValueError."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.end_session(sid)
        with pytest.raises(ValueError):
            mgr.resume_session(sid)

    def test_end_from_closed_raises_value_error(self):
        """对已关闭会话再次关闭 -> ValueError (closed->closed 非法)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.end_session(sid)
        with pytest.raises(ValueError):
            mgr.end_session(sid)

    def test_end_from_paused_is_valid(self):
        """从暂停状态关闭会话合法 (paused->closed)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        summary = mgr.end_session(sid)
        assert summary["status"] == "closed"
        assert mgr.get_session(sid).status == "closed"

    def test_pause_from_active_is_valid(self):
        """从活跃状态暂停合法 (active->paused)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        sess = mgr.pause_session(sid)
        assert sess.status == "paused"

    def test_resume_from_paused_is_valid(self):
        """从暂停状态恢复合法 (paused->active)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        mgr.pause_session(sid)
        sess = mgr.resume_session(sid)
        assert sess.status == "active"

    def test_existing_pause_resume_cycle_still_works(self):
        """现有 pause/resume 循环在加入校验后仍然正常."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        for _ in range(3):
            mgr.pause_session(sid)
            assert mgr.get_session(sid).status == "paused"
            mgr.resume_session(sid)
            assert mgr.get_session(sid).status == "active"

    def test_operation_on_merged_fork_raises_value_error(self):
        """对已合并的分叉会话执行生命周期操作 -> ValueError (merged 为终态)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        forked = mgr.fork_session(sid)
        mgr.merge_fork(forked.session_id, sid)
        with pytest.raises(ValueError):
            mgr.pause_session(forked.session_id)
        with pytest.raises(ValueError):
            mgr.resume_session(forked.session_id)
        with pytest.raises(ValueError):
            mgr.end_session(forked.session_id)
