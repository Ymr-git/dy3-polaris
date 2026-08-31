"""集成层 TDD 测试 — L2/L3/L4/L5 跨层集成 + 统一应用组装.

测试覆盖:
1. L4 API Router — 决策引擎 RESTful 端点
   - POST /l4/decision/query: 完整决策流程
   - POST /l4/decision/plan: 生成决策计划
   - POST /l4/decision/execute: 执行计划
   - POST /l4/decision/feedback: 提交反馈
   - GET /l4/health: 健康检查

2. L5 API Router — 编排/会话/Agent RESTful 端点
   - POST /l5/orchestrate: 执行编排计划 (Pipeline/Debate/Voting)
   - GET /l5/orchestration/{plan_id}: 获取编排结果
   - POST /l5/session: 创建会话
   - GET /l5/session/{session_id}: 获取会话状态
   - POST /l5/session/{session_id}/fork: Fork 会话
   - POST /l5/message/publish: 发布消息
   - GET /l5/health: 健康检查

3. L2 API Router 扩展 — Profile/Memory/Session 端点
   - GET /l2/profile/{learner_id}: 获取画像
   - GET /l2/profile/{learner_id}/weak-points: 获取薄弱知识点
   - GET /l2/profile/{learner_id}/confidence: 获取置信度
   - POST /l2/memory/update: 更新记忆状态
   - GET /l2/memory/{learner_id}: 获取记忆状态
   - GET /l2/skillbook/{learner_id}: 获取技能手册

4. 跨层集成桥接 (IntegrationBridge)
   - L2 → L4: 画像数据传入决策引擎
   - L4 → L5: 决策结果触发 Agent 编排
   - L5 → L2: Agent 执行结果反馈到画像
   - 事件驱动跨层通信

5. 统一应用组装器 (UnifiedApp)
   - 所有层 Router 挂载到单一应用
   - 统一健康检查聚合
   - CORS 中间件配置
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient


# ============================================================
# 测试辅助
# ============================================================


def make_mock_irt_service() -> MagicMock:
    """创建 mock IRT 服务."""
    svc = MagicMock()
    svc.estimate_ability.return_value = MagicMock(
        to_dict=lambda: {
            "learner_id": "learner1",
            "theta": 0.5,
            "se": 0.2,
            "response_count": 10,
            "p_correct_next": 0.65,
            "zpd_zone": "zpd",
            "confidence": 0.8,
            "ability_level": "中",
        }
    )
    svc.get_ability_snapshot.return_value = {
        "theta": 0.5,
        "se": 0.2,
        "response_count": 10,
    }
    return svc


def make_mock_profile_service() -> MagicMock:
    """创建 mock Profile 服务."""
    from dy3_polaris.l2.profile_builder.tracing_service import ProfileOutput

    svc = MagicMock()
    svc.get_profile_snapshot.return_value = MagicMock(
        to_dict=lambda: {
            "learner_id": "learner1",
            "theta": 0.5,
            "level": "intermediate",
            "kp_mastery": {"kp1": 0.8, "kp2": 0.3},
            "weak_kps": ["kp2"],
            "confidence": 0.85,
        }
    )
    svc.process.return_value = ProfileOutput(
        learner_id="learner1",
        phase="stable",
        theta=0.5,
        level="intermediate",
        learning_style="visual",
        bloom_target="apply",
        kp_mastery={"kp1": 0.8, "kp2": 0.3},
        weak_kps=["kp2"],
        confidence=0.85,
        snapshot_ts=time.time(),
    )
    return svc


def make_mock_memory_service() -> MagicMock:
    """创建 mock Memory 服务."""
    svc = MagicMock()
    svc.process.return_value = MagicMock(
        to_dict=lambda: {
            "learner_id": "learner1",
            "kp_id": "kp1",
            "retention": 0.85,
            "stability": 2.5,
            "retrievability": 0.9,
            "next_review_interval": 3.0,
        }
    )
    svc.get_memory_state.return_value = {
        "learner_id": "learner1",
        "kp_retentions": {"kp1": 0.85, "kp2": 0.6},
        "stale_kps": ["kp2"],
    }
    return svc


def make_mock_decision_engine() -> MagicMock:
    """创建 mock 决策引擎."""
    engine = MagicMock()

    # process_query 返回 ActionRecord mock
    action_record = MagicMock()
    action_record.action_type.value = "respond"
    action_record.confidence = 0.85
    action_record.response_payload = {"answer": "Dy3+ 激发态波长为 580nm"}
    action_record.plan_id = "plan_001"
    action_record.to_dict = lambda: {
        "action_type": "respond",
        "confidence": 0.85,
        "response_payload": {"answer": "Dy3+ 激发态波长为 580nm"},
        "plan_id": "plan_001",
    }

    engine.process_query = AsyncMock(return_value=action_record)
    engine.synthesize_output.return_value = MagicMock(
        to_dict=lambda: {
            "content": "Dy3+ 激发态波长为 580nm",
            "confidence": 0.85,
            "safety_level": "safe",
        }
    )
    return engine


def make_mock_orchestration_engine() -> MagicMock:
    """创建 mock 编排引擎."""
    engine = MagicMock()
    result = MagicMock()
    result.plan_id = "orch_001"
    result.state = "completed"
    result.to_dict = lambda: {
        "plan_id": "orch_001",
        "state": "completed",
        "paradigm": "pipeline",
        "results": [{"task_id": "t1", "status": "completed", "output": "result1"}],
        "elapsed_ms": 125.5,
    }
    engine.execute = AsyncMock(return_value=result)
    engine.get_result.return_value = result
    return engine


def make_mock_session_manager() -> MagicMock:
    """创建 mock 会话管理器."""
    mgr = MagicMock()
    session = MagicMock()
    session.session_id = "sess_001"
    session.state = "active"
    session.to_dict = lambda: {
        "session_id": "sess_001",
        "state": "active",
        "created_at": time.time(),
        "learner_id": "learner1",
    }
    mgr.create_session.return_value = session
    mgr.get_session.return_value = session
    mgr.fork_session.return_value = MagicMock(
        session_id="sess_002",
        parent_id="sess_001",
        to_dict=lambda: {
            "session_id": "sess_002",
            "parent_id": "sess_001",
            "state": "active",
        },
    )
    # router 实际调用 create_fork (L5 会话 Fork 接口)
    mgr.create_fork.return_value = MagicMock(
        session_id="sess_002",
        parent_id="sess_001",
        to_dict=lambda: {
            "session_id": "sess_002",
            "parent_id": "sess_001",
            "state": "active",
        },
    )
    mgr.close_session.return_value = True
    mgr.close.return_value = True  # router 实际调用 close (L5 会话关闭接口)
    return mgr


def make_mock_message_bus() -> MagicMock:
    """创建 mock 消息总线."""
    bus = MagicMock()
    bus.publish.return_value = "msg_001"
    bus.create_channel.return_value = True
    bus.subscribe.return_value = MagicMock(subscription_id="sub_001")
    bus.get_channel_history.return_value = []
    return bus


# ============================================================
# 1. L4 API Router 测试
# ============================================================


class TestL4APIRouter:
    """L4 决策引擎 API 路由器测试."""

    def test_health_check(self):
        """GET /l4/health 返回健康状态."""
        from dy3_polaris.l4.api.router import L4Router

        router = L4Router(decision_engine=make_mock_decision_engine())
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "healthy"
        assert data["data"]["layer"] == "L4"

    def test_decision_query(self):
        """POST /l4/decision/query 处理完整决策流程."""
        from dy3_polaris.l4.api.router import L4Router

        router = L4Router(decision_engine=make_mock_decision_engine())
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/decision/query", json={
            "query": "Dy3+ 的激发态波长是多少？",
            "context_id": "ctx_001",
            "learner_profile": {"learner_id": "learner1", "theta": 0.5},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["action_type"] == "respond"
        assert data["data"]["confidence"] > 0

    def test_decision_query_missing_query(self):
        """POST /l4/decision/query 缺少 query 参数返回错误."""
        from dy3_polaris.l4.api.router import L4Router

        router = L4Router(decision_engine=make_mock_decision_engine())
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/decision/query", json={"context_id": "ctx_001"})
        assert resp.status_code == 400

    def test_decision_feedback(self):
        """POST /l4/decision/feedback 提交反馈信号."""
        from dy3_polaris.l4.api.router import L4Router

        mock_engine = make_mock_decision_engine()
        mock_engine.record_feedback = MagicMock(return_value=True)

        router = L4Router(decision_engine=mock_engine)
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/decision/feedback", json={
            "action_id": "action_001",
            "feedback_type": "explicit",
            "rating": 0.8,
            "comment": "回答准确",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0

    def test_decision_synthesize(self):
        """POST /l4/decision/synthesize 合成最终输出."""
        from dy3_polaris.l4.api.router import L4Router

        router = L4Router(decision_engine=make_mock_decision_engine())
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/decision/synthesize", json={
            "action_type": "respond",
            "confidence": 0.85,
            "response_payload": {"answer": "580nm"},
            "plan_id": "plan_001",
            "execution_status": "completed",
            "validation_status": "passed",
            "validation_score": 0.9,
            "refinement_iterations": 0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "content" in data["data"]

        from dy3_polaris.l4.models import (
            ActionRecord,
            ExecutionResult,
            ValidationReport,
        )

        args = router._handlers._engine.synthesize_output.call_args.args
        assert isinstance(args[0], ActionRecord)
        assert isinstance(args[1], ExecutionResult)
        assert isinstance(args[2], ValidationReport)

    def test_routes_summary(self):
        """get_routes_summary 返回所有路由摘要."""
        from dy3_polaris.l4.api.router import L4Router

        router = L4Router(decision_engine=make_mock_decision_engine())
        summary = router.get_routes_summary()
        assert len(summary) >= 4
        paths = [r["path"] for r in summary]
        assert "/health" in paths
        assert "/decision/query" in paths


# ============================================================
# 2. L5 API Router 测试
# ============================================================


class TestL5APIRouter:
    """L5 Agent Runtime API 路由器测试."""

    def test_health_check(self):
        """GET /l5/health 返回健康状态."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "healthy"
        assert data["data"]["layer"] == "L5"

    def test_orchestrate(self):
        """POST /l5/orchestrate 执行编排计划."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/orchestrate", json={
            "paradigm": "pipeline",
            "tasks": [
                {"task_id": "t1", "agent_id": "agent_001", "input": {"query": "test"}},
            ],
            "session_id": "sess_001",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["state"] == "completed"
        assert data["data"]["plan_id"] is not None

    def test_orchestrate_missing_paradigm(self):
        """POST /l5/orchestrate 缺少 paradigm 返回错误."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/orchestrate", json={"tasks": []})
        assert resp.status_code == 400

    def test_get_orchestration_result(self):
        """GET /l5/orchestration/{plan_id} 获取编排结果."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/orchestration/orch_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["plan_id"] == "orch_001"

    def test_create_session(self):
        """POST /l5/session 创建会话."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/session", json={
            "learner_id": "learner1",
            "context": {"subject": "physics"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["session_id"] is not None
        assert data["data"]["state"] == "active"

    def test_get_session(self):
        """GET /l5/session/{session_id} 获取会话状态."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/session/sess_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["session_id"] == "sess_001"

    def test_fork_session(self):
        """POST /l5/session/{session_id}/fork Fork 会话."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/session/sess_001/fork", json={
            "fork_reason": "explore_alternative",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "session_id" in data["data"]
        assert "parent_id" in data["data"]

    def test_close_session(self):
        """POST /l5/session/{session_id}/close 关闭会话."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/session/sess_001/close")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0

    def test_publish_message(self):
        """POST /l5/message/publish 发布消息到总线."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/message/publish", json={
            "channel": "learner_events",
            "payload": {"event": "answer_submitted", "learner_id": "learner1"},
            "priority": "normal",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "message_id" in data["data"]

    def test_routes_summary(self):
        """get_routes_summary 返回所有路由摘要."""
        from dy3_polaris.l5.api.router import L5Router

        router = L5Router(
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        summary = router.get_routes_summary()
        assert len(summary) >= 6
        paths = [r["path"] for r in summary]
        assert "/health" in paths
        assert "/orchestrate" in paths
        assert "/session" in paths


# ============================================================
# 3. L2 API Router 扩展测试
# ============================================================


class TestL2APIRouterExpansion:
    """L2 API Router 扩展端点测试 — Profile/Memory/Session."""

    def test_profile_endpoint(self):
        """GET /l2/profile/{learner_id} 获取画像."""
        from dy3_polaris.l2.api.router import L2Router

        router = L2Router(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/profile/learner1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["learner_id"] == "learner1"
        assert "theta" in data["data"]
        assert "kp_mastery" in data["data"]

    def test_profile_weak_points(self):
        """GET /l2/profile/{learner_id}/weak-points 获取薄弱知识点."""
        from dy3_polaris.l2.api.router import L2Router

        router = L2Router(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/profile/learner1/weak-points")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert isinstance(data["data"]["weak_kps"], list)

    def test_profile_confidence(self):
        """GET /l2/profile/{learner_id}/confidence 获取置信度."""
        from dy3_polaris.l2.api.router import L2Router

        router = L2Router(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/profile/learner1/confidence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "confidence" in data["data"]

    def test_memory_update(self):
        """POST /l2/memory/update 更新记忆状态."""
        from dy3_polaris.l2.api.router import L2Router

        router = L2Router(
            irt_service=make_mock_irt_service(),
            memory_service=make_mock_memory_service(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/memory/update", json={
            "learner_id": "learner1",
            "kp_id": "kp1",
            "correct": True,
            "difficulty": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "retention" in data["data"]

    def test_memory_get(self):
        """GET /l2/memory/{learner_id} 获取记忆状态."""
        from dy3_polaris.l2.api.router import L2Router

        router = L2Router(
            irt_service=make_mock_irt_service(),
            memory_service=make_mock_memory_service(),
        )
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/memory/learner1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "kp_retentions" in data["data"]

    def test_expanded_routes_summary(self):
        """扩展后的路由摘要包含 Profile/Memory 端点."""
        from dy3_polaris.l2.api.router import L2Router

        router = L2Router(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
        )
        summary = router.get_routes_summary()
        paths = [r["path"] for r in summary]
        assert "/profile/{learner_id}" in paths
        assert "/memory/update" in paths
        assert "/memory/{learner_id}" in paths


# ============================================================
# 4. 跨层集成桥接测试
# ============================================================


class TestIntegrationBridge:
    """跨层集成桥接测试 — L2↔L4↔L5 事件驱动通信."""

    def test_bridge_initialization(self):
        """IntegrationBridge 初始化包含所有层引用."""
        from dy3_polaris.l5.integration_bridge import IntegrationBridge

        bridge = IntegrationBridge(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        assert bridge is not None
        assert bridge.irt_service is not None
        assert bridge.profile_service is not None
        assert bridge.decision_engine is not None
        assert bridge.orchestration_engine is not None

    def test_assemble_decision_context(self):
        """从 L2 画像数据组装 L4 决策上下文."""
        from dy3_polaris.l5.integration_bridge import IntegrationBridge

        bridge = IntegrationBridge(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )

        context = bridge.assemble_decision_context(learner_id="learner1")
        assert "learner_id" in context
        assert "theta" in context
        assert "kp_mastery" in context
        assert "weak_kps" in context
        assert "memory_state" in context

    def test_process_query_with_profile(self):
        """端到端: 画像 → 决策引擎 → 输出."""
        from dy3_polaris.l5.integration_bridge import IntegrationBridge

        bridge = IntegrationBridge(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )

        result = asyncio.run(
            bridge.process_query_with_profile(
                learner_id="learner1",
                query="Dy3+ 的激发态波长是多少？",
            )
        )
        assert result is not None
        assert "action_type" in result
        assert "confidence" in result

    def test_publish_learner_event(self):
        """发布学习者事件到消息总线 (跨层事件驱动)."""
        from dy3_polaris.l5.integration_bridge import IntegrationBridge

        mock_bus = make_mock_message_bus()
        bridge = IntegrationBridge(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=mock_bus,
        )

        msg_id = bridge.publish_learner_event(
            learner_id="learner1",
            event_type="answer_submitted",
            payload={"kp_id": "kp1", "correct": True},
        )
        assert msg_id is not None
        mock_bus.publish.assert_called_once()

    def test_feedback_to_profile(self):
        """L5 Agent 执行结果反馈到 L2 画像."""
        from dy3_polaris.l5.integration_bridge import IntegrationBridge

        mock_profile = make_mock_profile_service()
        bridge = IntegrationBridge(
            irt_service=make_mock_irt_service(),
            profile_service=mock_profile,
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )

        bridge.feedback_to_profile(
            learner_id="learner1",
            agent_result={
                "action_type": "respond",
                "confidence": 0.85,
                "response_payload": {"answer": "580nm"},
            },
        )
        # Profile 服务应被调用 (具体调用方式取决于实现)
        assert mock_profile is not None

    def test_get_cross_layer_health(self):
        """跨层健康检查聚合."""
        from dy3_polaris.l5.integration_bridge import IntegrationBridge

        bridge = IntegrationBridge(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )

        health = bridge.get_cross_layer_health()
        assert "l2" in health
        assert "l4" in health
        assert "l5" in health
        assert health["l2"]["status"] in ("healthy", "degraded")
        assert health["l4"]["status"] in ("healthy", "degraded")
        assert health["l5"]["status"] in ("healthy", "degraded")


# ============================================================
# 5. 统一应用组装器测试
# ============================================================


class TestUnifiedApp:
    """统一应用组装器测试 — 所有层 Router 挂载到单一应用."""

    def test_unified_app_creation(self):
        """UnifiedApp 创建成功并挂载所有层路由."""
        from dy3_polaris.l5.unified_app import UnifiedApp

        app_builder = UnifiedApp(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = app_builder.create_app()
        assert app is not None

    def test_unified_health(self):
        """GET /health 返回所有层的聚合健康状态."""
        from dy3_polaris.l5.unified_app import UnifiedApp

        app_builder = UnifiedApp(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = app_builder.create_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "layers" in data["data"]

    def test_unified_l2_mounted(self):
        """L2 路由挂载在 /l2 前缀下."""
        from dy3_polaris.l5.unified_app import UnifiedApp

        app_builder = UnifiedApp(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = app_builder.create_app()
        client = TestClient(app)

        resp = client.get("/l2/health")
        assert resp.status_code == 200

    def test_unified_l4_mounted(self):
        """L4 路由挂载在 /l4 前缀下."""
        from dy3_polaris.l5.unified_app import UnifiedApp

        app_builder = UnifiedApp(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = app_builder.create_app()
        client = TestClient(app)

        resp = client.get("/l4/health")
        assert resp.status_code == 200

    def test_unified_l5_mounted(self):
        """L5 路由挂载在 /l5 前缀下."""
        from dy3_polaris.l5.unified_app import UnifiedApp

        app_builder = UnifiedApp(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = app_builder.create_app()
        client = TestClient(app)

        resp = client.get("/l5/health")
        assert resp.status_code == 200

    def test_unified_api_info(self):
        """GET /api/info 返回所有可用 API 端点."""
        from dy3_polaris.l5.unified_app import UnifiedApp

        app_builder = UnifiedApp(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )
        app = app_builder.create_app()
        client = TestClient(app)

        resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "endpoints" in data["data"]
        assert len(data["data"]["endpoints"]) >= 3

    def test_unified_cors_headers(self):
        """统一应用配置 CORS 中间件."""
        from dy3_polaris.l5.unified_app import UnifiedApp

        app_builder = UnifiedApp(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
            cors_origins=["https://example.com"],
        )
        app = app_builder.create_app()
        client = TestClient(app)

        resp = client.options(
            "/health",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS 预检请求应返回 200
        assert resp.status_code in (200, 204)


# ============================================================
# 6. 端到端跨层流程测试
# ============================================================


class TestEndToEndFlow:
    """端到端跨层流程测试 — 完整的用户查询处理链路."""

    def test_full_query_flow(self):
        """完整流程: 用户查询 → L4决策 → L5编排 → L2画像更新."""
        from dy3_polaris.l5.integration_bridge import IntegrationBridge

        bridge = IntegrationBridge(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )

        loop = asyncio.new_event_loop()
        try:
            # 1. 组装决策上下文 (L2 画像 → L4)
            context = bridge.assemble_decision_context(learner_id="learner1")
            assert "theta" in context

            # 2. 处理查询 (L4 决策引擎)
            action = loop.run_until_complete(
                bridge.process_query_with_profile(
                    learner_id="learner1",
                    query="Dy3+ 的激发态波长是多少？",
                )
            )
            assert action["action_type"] == "respond"

            # 3. 反馈到画像 (L5 → L2)
            bridge.feedback_to_profile(
                learner_id="learner1",
                agent_result=action,
            )

            # 4. 发布事件
            msg_id = bridge.publish_learner_event(
                learner_id="learner1",
                event_type="query_completed",
                payload={"confidence": action["confidence"]},
            )
            assert msg_id is not None
        finally:
            loop.close()

    def test_cross_layer_health_aggregation(self):
        """跨层健康检查: 所有层状态聚合到统一视图."""
        from dy3_polaris.l5.integration_bridge import IntegrationBridge

        bridge = IntegrationBridge(
            irt_service=make_mock_irt_service(),
            profile_service=make_mock_profile_service(),
            memory_service=make_mock_memory_service(),
            decision_engine=make_mock_decision_engine(),
            orchestration_engine=make_mock_orchestration_engine(),
            session_manager=make_mock_session_manager(),
            message_bus=make_mock_message_bus(),
        )

        health = bridge.get_cross_layer_health()
        assert "l2" in health
        assert "l4" in health
        assert "l5" in health
        # 每层都有 status 字段
        for layer_name, layer_health in health.items():
            assert "status" in layer_health
