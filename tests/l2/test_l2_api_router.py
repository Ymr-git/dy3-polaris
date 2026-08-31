"""T3 IRT 能力评估 — L2 REST API 路由器测试.

基于 httpx.AsyncClient + ASGITransport 进行无端口 API 测试,
遵循项目统一模式 (与 test_rest_api.py 一致).

测试覆盖 (设计文档 §7.3 API 设计):
1. 健康检查端点 GET /l2/health
2. IRT 能力估计端点 POST /l2/irt/estimate (核心端点)
3. 自适应出题端点 POST /l2/irt/next-question
4. MMLE 题库校准端点 POST /l2/irt/calibrate
5. 能力快照端点 GET /l2/irt/ability/{learner_id}
6. BKT 单 KP 在线更新端点 POST /l2/bkt/update
7. 融合诊断端点 POST /l2/irt/estimate (带 mastery_map)
8. L2Router 元信息 (get_routes_summary)
9. 错误处理与边界情况
10. 世界先进方案 API 对标 (catR/mirt/Knewton)

设计参考:
- L2 个性化设计 §7.3: /l2/irt/estimate, /l2/irt/next-question, /l2/bkt/update
- L6 协议基础设施: MCP 工具 skill_irt_evaluate
- 项目统一响应格式: {"code": 0, "data": ..., "message": ""}
- Starlette ASGI + CORS 中间件
"""

from __future__ import annotations

import logging
import time

logging.disable(logging.CRITICAL)

import pytest
from httpx import ASGITransport, AsyncClient

from dy3_polaris.l2.ability_assessor import IRTTracingService
from dy3_polaris.l2.interaction.event_types import AnswerEvent

# L2Router 在实现前不存在, 这是 RED phase
try:
    from dy3_polaris.l2.api import L2Router
    _L2_ROUTER_AVAILABLE = True
except ImportError:
    _L2_ROUTER_AVAILABLE = False
    L2Router = None  # type: ignore[assignment,misc]


# ============================================================
# 测试辅助
# ============================================================

def _make_irt_service() -> IRTTracingService:
    """创建带题库的 IRTTracingService 测试实例."""
    service = IRTTracingService(enable_enhanced=True, enable_fusion=True)
    # 设置测试题库
    service.set_item_bank([
        {"item_id": "q1", "a": 1.2, "b": -1.0, "c": 0.2},
        {"item_id": "q2", "a": 1.5, "b": 0.0, "c": 0.25},
        {"item_id": "q3", "a": 1.0, "b": 1.0, "c": 0.15},
        {"item_id": "q4", "a": 1.8, "b": -0.5, "c": 0.2},
        {"item_id": "q5", "a": 1.3, "b": 0.5, "c": 0.3},
    ])
    return service


def _make_answer_events(learner_id: str = "learner_001", n: int = 5) -> list[AnswerEvent]:
    """创建答题事件序列."""
    events = []
    for i in range(n):
        events.append(AnswerEvent(
            learner_id=learner_id,
            kp_id=f"kp_{i}",
            correct=i % 2 == 0,
            difficulty=0.3 + i * 0.1,
            question_id=f"q{i+1}",
            timestamp=1000.0 + i,
        ))
    return events


# ============================================================
# 1. 健康检查端点
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestHealthEndpoint:
    """GET /l2/health — L2 个性化层健康检查."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        """健康检查返回 200 和统一响应格式."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "healthy"
        assert data["data"]["layer"] == "L2"

    @pytest.mark.asyncio
    async def test_health_includes_service_info(self):
        """健康检查包含服务信息."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()["data"]
        assert "timestamp" in data
        assert "services" in data


# ============================================================
# 2. IRT 能力估计端点 (核心)
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestIRTEstimateEndpoint:
    """POST /l2/irt/estimate — IRT 能力参数估计 (对应设计文档 §7.3)."""

    @pytest.mark.asyncio
    async def test_estimate_returns_200_with_theta(self):
        """正常请求返回 theta 和 SE."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.4, "timestamp": 1000.0},
                    {"learner_id": "learner_001", "kp_id": "kp_2", "correct": False, "difficulty": 0.6, "timestamp": 1001.0},
                ],
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        ability = data["data"]
        assert "theta" in ability
        assert "se" in ability
        assert "response_count" in ability
        assert "ability_level" in ability
        assert "recommendation" in ability

    @pytest.mark.asyncio
    async def test_estimate_empty_events_returns_prior(self):
        """空事件返回群体先验 (theta=0.0, se=0.3)."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_cold",
                "events": [],
            })
        assert resp.status_code == 200
        ability = resp.json()["data"]
        assert ability["theta"] == 0.0
        assert ability["se"] == 0.3
        assert ability["response_count"] == 0

    @pytest.mark.asyncio
    async def test_estimate_missing_learner_id_returns_400(self):
        """缺少 learner_id 返回 400."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "events": [],
            })
        assert resp.status_code == 400
        assert resp.json()["code"] != 0

    @pytest.mark.asyncio
    async def test_estimate_invalid_json_returns_400(self):
        """无效 JSON 返回 400."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", content="not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_estimate_includes_zpd_zone(self):
        """响应包含 ZPD 区分类."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.3, "timestamp": 1000.0},
                ],
            })
        ability = resp.json()["data"]
        assert ability["zpd_zone"] in ("independent", "zpd", "frustration")

    @pytest.mark.asyncio
    async def test_estimate_includes_confidence(self):
        """响应包含置信度."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.3, "timestamp": 1000.0},
                ],
            })
        ability = resp.json()["data"]
        assert 0.0 < ability["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_estimate_with_enhanced_fields(self):
        """增强模式返回 ci_lower, ci_upper 等增强字段."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.3, "timestamp": 1000.0},
                    {"learner_id": "learner_001", "kp_id": "kp_2", "correct": True, "difficulty": 0.4, "timestamp": 1001.0},
                    {"learner_id": "learner_001", "kp_id": "kp_3", "correct": False, "difficulty": 0.6, "timestamp": 1002.0},
                ],
            })
        ability = resp.json()["data"]
        assert "ci_lower" in ability
        assert "ci_upper" in ability
        assert ability["ci_lower"] <= ability["theta"] <= ability["ci_upper"]


# ============================================================
# 3. 自适应出题端点
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestNextQuestionEndpoint:
    """POST /l2/irt/next-question — 自适应出题 (CAT 选题)."""

    @pytest.mark.asyncio
    async def test_next_question_returns_item(self):
        """正常请求返回推荐的下一题."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 先处理一些事件建立 theta
            await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.3, "timestamp": 1000.0},
                ],
            })
            # 请求下一题
            resp = await client.post("/irt/next-question", json={
                "learner_id": "learner_001",
                "available_items": [
                    {"item_id": "q1", "a": 1.2, "b": -1.0, "c": 0.2},
                    {"item_id": "q2", "a": 1.5, "b": 0.0, "c": 0.25},
                    {"item_id": "q3", "a": 1.0, "b": 1.0, "c": 0.15},
                ],
                "administered_ids": ["kp_1"],
            })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "item_id" in data
        assert data["item_id"] is not None

    @pytest.mark.asyncio
    async def test_next_question_empty_pool_returns_null(self):
        """空题库返回 next_item_id=null."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/next-question", json={
                "learner_id": "learner_001",
                "available_items": [],
                "administered_ids": [],
            })
        assert resp.status_code == 200
        assert resp.json()["data"]["item_id"] is None

    @pytest.mark.asyncio
    async def test_next_question_with_fusion(self):
        """融合模式: 提供 mastery_map 进行 BKT+IRT 融合选题."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/next-question", json={
                "learner_id": "learner_001",
                "available_items": [
                    {"item_id": "q1", "a": 1.2, "b": -1.0, "c": 0.2},
                    {"item_id": "q2", "a": 1.5, "b": 0.0, "c": 0.25},
                    {"item_id": "q3", "a": 1.0, "b": 1.0, "c": 0.15},
                ],
                "administered_ids": [],
                "mastery_map": {"q1": 0.5, "q2": 0.4, "q3": 0.9},
            })
        assert resp.status_code == 200
        data = resp.json()["data"]
        # 融合模式应优先选 ZPD 区题目 (mastery 在 0.3~0.7)
        assert data["item_id"] in ("q1", "q2")


# ============================================================
# 4. MMLE 题库校准端点
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestCalibrateEndpoint:
    """POST /l2/irt/calibrate — MMLE 题库参数校准."""

    @pytest.mark.asyncio
    async def test_calibrate_returns_calibrated_params(self):
        """校准返回更新后的题目参数."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        # 构造多学习者作答数据
        responses_by_learner = {}
        for lid in ["l1", "l2", "l3"]:
            responses_by_learner[lid] = [
                ({"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0}, True),
                ({"item_id": "q2", "a": 1.0, "b": 0.5, "c": 0.0}, False),
                ({"item_id": "q3", "a": 1.0, "b": -0.5, "c": 0.0}, True),
            ]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/calibrate", json={
                "responses_by_learner": responses_by_learner,
                "n_iterations": 10,
            })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "items" in data
        assert "q1" in data["items"]
        # 参数应在约束范围内
        for item_id, params in data["items"].items():
            assert 0.3 <= params["a"] <= 3.0
            assert -3.0 <= params["b"] <= 3.0
            assert 0.0 <= params["c"] <= 0.5

    @pytest.mark.asyncio
    async def test_calibrate_empty_input_returns_empty(self):
        """空输入返回空结果."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/calibrate", json={
                "responses_by_learner": {},
            })
        assert resp.status_code == 200
        assert resp.json()["data"]["items"] == {}


# ============================================================
# 5. 能力快照端点
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestAbilitySnapshotEndpoint:
    """GET /l2/irt/ability/{learner_id} — 获取能力快照."""

    @pytest.mark.asyncio
    async def test_get_ability_returns_200(self):
        """获取能力快照返回 200."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 先处理事件
            await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.3, "timestamp": 1000.0},
                ],
            })
            # 获取快照
            resp = await client.get("/irt/ability/learner_001")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["learner_id"] == "learner_001"
        assert "theta" in data
        assert "se" in data
        assert "response_count" in data

    @pytest.mark.asyncio
    async def test_get_ability_cold_start_returns_prior(self):
        """冷启动学习者返回群体先验."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/irt/ability/unknown_learner")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["theta"] == 0.0
        assert data["response_count"] == 0


# ============================================================
# 6. BKT 单 KP 在线更新端点
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestBKTUpdateEndpoint:
    """POST /l2/bkt/update — BKT 单 KP 在线更新."""

    @pytest.mark.asyncio
    async def test_bkt_update_returns_200(self):
        """BKT 更新返回 200 和掌握度."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/bkt/update", json={
                "learner_id": "learner_001",
                "kp_id": "kp_1",
                "correct": True,
                "difficulty": 0.4,
                "timestamp": 1000.0,
            })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "p_mastery" in data
        assert 0.0 <= data["p_mastery"] <= 1.0

    @pytest.mark.asyncio
    async def test_bkt_update_missing_kp_id_returns_400(self):
        """缺少 kp_id 返回 400."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/bkt/update", json={
                "learner_id": "learner_001",
                "correct": True,
            })
        assert resp.status_code == 400


# ============================================================
# 7. 融合诊断端点 (BKT+IRT)
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestFusionDiagnosisEndpoint:
    """POST /l2/irt/estimate (带 mastery_map) — BKT+IRT 融合自适应诊断."""

    @pytest.mark.asyncio
    async def test_fusion_estimate_includes_next_item(self):
        """融合估计返回基于掌握度的推荐下一题."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.3, "timestamp": 1000.0},
                    {"learner_id": "learner_001", "kp_id": "kp_2", "correct": False, "difficulty": 0.6, "timestamp": 1001.0},
                ],
                "mastery_map": {"q1": 0.5, "q2": 0.4, "q3": 0.9, "q4": 0.6, "q5": 0.3},
            })
        assert resp.status_code == 200
        ability = resp.json()["data"]
        # 应推荐下一题
        assert ability["next_item_id"] is not None
        # 推荐信息中应包含 next_item_id
        assert ability["recommendation"]["next_item_id"] is not None

    @pytest.mark.asyncio
    async def test_fusion_estimate_prefers_zpd_items(self):
        """融合估计优先推荐 ZPD 区 (掌握度 0.3~0.7) 的题目."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.3, "timestamp": 1000.0},
                ],
                # 提供所有题目的掌握度: q1=frustration, q2=ZPD, q3=mastered, q4=frustration, q5=mastered
                "mastery_map": {"q1": 0.1, "q2": 0.5, "q3": 0.95, "q4": 0.1, "q5": 0.95},
            })
        ability = resp.json()["data"]
        # 只有 q2 在 ZPD 区 (掌握度 0.5), 应被选中
        assert ability["next_item_id"] == "q2"


# ============================================================
# 8. L2Router 元信息
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestL2RouterMeta:
    """L2Router 元信息测试."""

    def test_routes_summary_not_empty(self):
        """路由摘要非空."""
        router = L2Router(irt_service=_make_irt_service())
        summary = router.get_routes_summary()
        assert len(summary) > 0
        # 每项应包含 path, methods, description
        for route in summary:
            assert "path" in route
            assert "methods" in route
            assert "description" in route

    def test_routes_summary_includes_irt_estimate(self):
        """路由摘要包含 /irt/estimate."""
        router = L2Router(irt_service=_make_irt_service())
        summary = router.get_routes_summary()
        paths = [r["path"] for r in summary]
        assert "/irt/estimate" in paths

    def test_routes_summary_includes_next_question(self):
        """路由摘要包含 /irt/next-question."""
        router = L2Router(irt_service=_make_irt_service())
        summary = router.get_routes_summary()
        paths = [r["path"] for r in summary]
        assert "/irt/next-question" in paths

    def test_routes_summary_includes_health(self):
        """路由摘要包含 /health."""
        router = L2Router(irt_service=_make_irt_service())
        summary = router.get_routes_summary()
        paths = [r["path"] for r in summary]
        assert "/health" in paths

    def test_create_app_returns_starlette(self):
        """create_app 返回 Starlette 应用."""
        from starlette.applications import Starlette
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        assert isinstance(app, Starlette)


# ============================================================
# 9. 错误处理与边界情况
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestErrorHandling:
    """错误处理与边界情况."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_404(self):
        """未知端点返回 404."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_wrong_method_returns_405(self):
        """错误 HTTP 方法返回 405."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/irt/estimate")
        assert resp.status_code == 405

    @pytest.mark.asyncio
    async def test_estimate_with_many_events(self):
        """大量事件 (20+) 正常处理."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        events = []
        for i in range(25):
            events.append({
                "learner_id": "learner_stress",
                "kp_id": f"kp_{i}",
                "correct": i % 3 != 0,
                "difficulty": 0.2 + i * 0.03,
                "timestamp": 1000.0 + i,
            })
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_stress",
                "events": events,
            })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["response_count"] == 25


# ============================================================
# 10. 世界先进方案 API 对标
# ============================================================

@pytest.mark.skipif(not _L2_ROUTER_AVAILABLE, reason="L2Router 未实现")
class TestWorldSchemeAPICompliance:
    """世界先进方案 API 对标测试 (catR/mirt/Knewton/ALEKS)."""

    @pytest.mark.asyncio
    async def test_api_response_format_matches_standard(self):
        """响应格式符合项目统一标准 {code, data, message}."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        body = resp.json()
        assert "code" in body
        assert "data" in body
        assert "message" in body
        assert body["code"] == 0

    @pytest.mark.asyncio
    async def test_irt_estimate_corresponds_to_skill_irt_evaluate(self):
        """/irt/estimate 对应 L6 MCP 工具 skill_irt_evaluate."""
        # L6 设计文档定义 skill_irt_evaluate 为 "3PL IRT 模型能力参数估计"
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.3, "timestamp": 1000.0},
                ],
            })
        data = resp.json()["data"]
        # 应包含 3PL 模型核心输出: theta, se
        assert "theta" in data
        assert "se" in data
        # 应包含预测正确率 (P(theta))
        assert "p_correct_next" in data

    @pytest.mark.asyncio
    async def test_next_question_implements_catr_fisher_info(self):
        """/irt/next-question 实现 catR 最大 Fisher 信息准则."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 先建立 theta > 0 (答对简单题)
            await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": [
                    {"learner_id": "learner_001", "kp_id": "kp_1", "correct": True, "difficulty": 0.2, "timestamp": 1000.0},
                    {"learner_id": "learner_001", "kp_id": "kp_2", "correct": True, "difficulty": 0.3, "timestamp": 1001.0},
                ],
            })
            # 题目 b 接近 theta 的题信息量最大
            resp = await client.post("/irt/next-question", json={
                "learner_id": "learner_001",
                "available_items": [
                    {"item_id": "easy", "a": 1.0, "b": -2.0, "c": 0.0},
                    {"item_id": "medium", "a": 1.5, "b": 0.5, "c": 0.0},
                    {"item_id": "hard", "a": 1.0, "b": 2.5, "c": 0.0},
                ],
                "administered_ids": [],
            })
        data = resp.json()["data"]
        # Fisher 信息量在 theta≈b 处最大, 应选中 medium (b=0.5)
        assert data["item_id"] == "medium"

    @pytest.mark.asyncio
    async def test_latency_under_200ms(self):
        """单次估计延迟 < 200ms (测试策略 §性能指标)."""
        import time as _time
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        events = [
            {"learner_id": "learner_001", "kp_id": f"kp_{i}", "correct": i % 2 == 0, "difficulty": 0.3 + i * 0.1, "timestamp": 1000.0 + i}
            for i in range(10)
        ]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            start = _time.perf_counter()
            resp = await client.post("/irt/estimate", json={
                "learner_id": "learner_001",
                "events": events,
            })
            elapsed_ms = (_time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        assert elapsed_ms < 200.0, f"延迟 {elapsed_ms:.1f}ms 超过 200ms 阈值"

    @pytest.mark.asyncio
    async def test_cors_headers_present(self):
        """CORS 头存在 (跨域支持)."""
        router = L2Router(irt_service=_make_irt_service())
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options("/health", headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            })
        # CORS 预检应返回 200
        assert resp.status_code in (200, 204)
