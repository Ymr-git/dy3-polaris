"""评审整改专项测试: 标准值单点 / 契约单点 / 安全网关 / 幂等·Outbox·Persistence·health."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def builder():
    return UnifiedApp.create_full_app_builder()


@pytest.fixture(scope="module")
def client(builder) -> TestClient:
    return TestClient(builder.create_app())


def _login(client, student_id="DY20240001", password="demo123"):
    r = client.post("/l1/api/v1/auth/login", json={"student_id": student_id, "password": password})
    return {"Authorization": "Bearer " + r.json()["data"]["access_token"]}


# ============================================================
# P1-1 领域标准值库单点
# ============================================================

class TestDomainStandards:
    def test_singleton_store(self):
        from dy3_polaris.shared.domain_standards import (
            build_domain_standard_store, domain_standard_count,
        )
        store = build_domain_standard_store()
        assert domain_standard_count() == 0
        assert store.count() == domain_standard_count()

    def test_fact_checker_uses_domain_store(self, builder):
        """运行时不把特定材料波长误当作全局标准值."""
        fc = builder.bridge.fact_checker if hasattr(builder.bridge, "fact_checker") else None
        if fc is None:
            pytest.skip("bridge 无 fact_checker")
        report = fc.check("Dy3+的发射波长580nm")
        assert report.total_assertions >= 1
        assert report.checked == 0
        assert report.skipped >= 1

    def test_reject_unknown(self):
        """没有真实标准时保持未知，不伪造失败结论."""
        from dy3_polaris.shared.domain_standards import build_domain_standard_store
        from dy3_polaris.l3.fact_check import FactChecker
        report = FactChecker(build_domain_standard_store()).check("Dy3+的发射波长999nm")
        assert report.total_assertions == 1
        assert report.passed == 0
        assert report.failed == 0
        assert report.skipped == 1


# ============================================================
# P1-2 错误码注册表 + 响应信封单点
# ============================================================

class TestContractSSOT:
    def test_error_code_duplicate_rejected(self):
        from dy3_polaris.shared.contract import register_error_code
        with pytest.raises(ValueError):
            register_error_code(-32602, "OTHER_OWNER", "dup", 400)

    def test_registry_contains_common_codes(self):
        from dy3_polaris.shared.contract import error_code_info
        assert error_code_info(-32700)["name"] == "PARSE_ERROR"
        assert error_code_info(-32201)["http_status"] == 401
        assert error_code_info(-32310)["name"] == "PROFILE_CONFLICT"

    def test_err_envelope(self):
        from dy3_polaris.shared.contract import err, ok
        assert ok({"x": 1}) == {"code": 0, "data": {"x": 1}, "message": ""}
        e = err(-32400, "boom", "detail-msg")
        assert e["code"] == -32400 and e["message"] == "boom" and e["detail"] == "detail-msg"
        e2 = err(-32310, "conflict", current_version=7)
        assert e2["current_version"] == 7

    def test_layer_reexport_consistency(self, client):
        """各层错误信封同构 (含 trace_id 回填)."""
        r = client.post("/l4/decision/next-action", json={"learner_id": "x", "mode": "guide"})
        body = r.json()
        assert body["code"] != 0 and "message" in body
        assert r.headers.get("x-trace-id", "").startswith("tr-")


# ============================================================
# P1-3 安全网关 (写端点鉴权)
# ============================================================

class TestSecurityGateway:
    def test_write_endpoint_requires_token(self, client):
        """未带 token 的写端点 → 401 (统一错误信封)."""
        r = client.post("/l4/decision/next-action", json={"learner_id": "x", "mode": "guide"})
        assert r.status_code == 401
        body = r.json()
        assert body["code"] == -32201
        assert body["trace_id"].startswith("tr-")

    def test_read_endpoint_open(self, client):
        """GET 只读端点放行."""
        assert client.get("/l2/kp-catalog").status_code == 200
        assert client.get("/l3/stats").status_code == 200

    def test_public_write_whitelist(self, client):
        """学生公开操作白名单放行 (练习/埋点/检索)."""
        assert client.post("/l2/practice/answer",
                           json={"learner_id": "DY20240001", "qid": "x", "selected": 0}).status_code in (200, 400)
        assert client.post("/l3/retrieve/vector",
                           json={"query_vector": [0.1, 0.2, 0.3], "query": "Dy3+", "top_k": 2}).status_code == 200
        assert client.post("/l6/jsonrpc", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 200

    def test_write_with_token(self, client):
        """带 token 的写端点通过 (L4 决策)."""
        h = _login(client)
        r = client.post("/l4/decision/next-action",
                        json={"learner_id": "DY20240001", "mode": "guide"}, headers=h)
        assert r.status_code == 200
        assert r.json()["code"] == 0

    def test_l6_tool_call_guarded(self, client):
        """L6 工具调用写端点需 token (内部工具调用例外放行)."""
        r = client.post("/l6/tools/path_simulation/call",
                        json={"arguments": {"learner_id": "DY20240001"}})
        assert r.status_code == 401


# ============================================================
# P2 幂等键 / Outbox / Persistence / health
# ============================================================

class TestP2:
    def test_idempotency_key_dedup(self, client):
        """同键同路径重复写请求返回首次响应 (不重复执行)."""
        h = _login(client)
        payload = {"learner_id": "DY20240001", "qid": "no-such-q", "selected": 0}
        r1 = client.post("/l2/practice/answer", json=payload,
                         headers={**h, "X-Idempotency-Key": "idem-test-1"})
        r2 = client.post("/l2/practice/answer", json=payload,
                         headers={**h, "X-Idempotency-Key": "idem-test-1"})
        assert r1.status_code == r2.status_code
        assert r1.content == r2.content
        # 不同键 → 独立处理
        r3 = client.post("/l2/practice/answer", json=payload,
                         headers={**h, "X-Idempotency-Key": "idem-test-2"})
        assert r3.status_code == r2.status_code

    def test_outbox_wired_to_message_bus(self, builder):
        """消息发布经 Outbox 入箱并投递 (投递后 pending=0)."""
        outbox = getattr(builder.bridge, "outbox", None)
        if outbox is None:
            pytest.skip("bridge 无 outbox")
        from dy3_polaris.l5.communication import Message

        bus = builder.bridge.message_bus
        bus.publish(Message(channel="test.outbox", payload={"v": 1}, publisher="test"))
        assert outbox.total_count() >= 1
        assert outbox.pending_count() == 0  # 已投递
        history = bus.get_history("test.outbox")
        assert history  # 消息已到达真实总线

    def test_l3_persistence_wired(self, builder):
        """L3 PersistenceManager 接入统一组装 (项目数据目录 + 快照)."""
        l3_router = builder.bridge.l3_router
        pm = getattr(l3_router._handlers, "_persistence", None)
        if pm is None:
            pm = getattr(l3_router, "persistence_manager", None)
        assert pm is not None
        base = getattr(pm, "_base_path", None) or getattr(pm, "base_path", "")
        assert "l3" in str(base) and "tmp" not in str(base).lower()

    def test_health_real_probe(self, client):
        """/health 真实探活: 每层含 latency_ms 与 probe 明细."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()["data"]
        layers = data["layers"] if "layers" in data else data
        for layer in ("l1", "l2", "l3", "l4", "l5", "l6"):
            assert layer in layers
            services = layers[layer]["services"]
            assert services, f"{layer} 无探活明细"
            for svc in services.values():
                assert "latency_ms" in svc
                assert "probe" in svc
        # L1 JWT 验签探针真实执行
        assert layers["l1"]["services"]["jwt_verifier"]["state"] == "available"
        # L3 存储计数探针返回实体数
        assert "entities=" in layers["l3"]["services"]["knowledge_store"]["probe"]
