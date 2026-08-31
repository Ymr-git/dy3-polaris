"""用户理解 API 集成测试."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def client():
    app = UnifiedApp.create_full_app_builder().create_app()
    with TestClient(app) as c:
        yield c


def test_uu_extract(client):
    r = client.post("/api/user-understanding/extract", json={
        "learner_id": "UUTEST01",
        "turns": [{"role": "user", "text": "浓度猝灭怎么避免？我想考研"}],
    })
    assert r.status_code == 200
    data = r.json().get("data") or {}
    assert isinstance(data.get("signals"), list)


def test_uu_ask_observation_first(client):
    """意图清晰时不主动提问 (观察为主)."""
    r = client.post("/api/user-understanding/ask", json={
        "learner_id": "UUTEST01",
        "context": {"view": "overview"},
    })
    assert r.status_code == 200
    data = r.json().get("data") or {}
    assert data.get("question") is None


def test_uu_ask_ambiguous_clarify(client):
    """请求难以理解/意图模糊时, 返回澄清问题."""
    r = client.post("/api/user-understanding/ask", json={
        "learner_id": "UUTEST01",
        "context": {"ambiguous": True, "intent": "query"},
    })
    assert r.status_code == 200
    data = r.json().get("data") or {}
    q = data.get("question")
    assert q is not None
    assert q.get("trigger") in ("ambiguous", "clarify")
    assert q.get("options") and "跳过" in q["options"]


def test_uu_guide(client):
    """引导式咨询: 返回结构化建议 (结合学情画像)."""
    r = client.post("/api/user-understanding/guide", json={
        "learner_id": "UUTEST01",
        "context": {"utterance": "不知道学什么"},
    })
    assert r.status_code == 200
    data = r.json().get("data") or {}
    g = data.get("guidance") or {}
    assert "direction" in g
    assert "reason" in g
    assert "next_steps" in g
    assert g.get("source") in ("learner_snapshot", "interest", "mixed", "fallback")


def test_uu_profile(client):
    r = client.post("/api/user-understanding/profile", json={"learner_id": "UUTEST01"})
    assert r.status_code == 200
    data = r.json().get("data") or {}
    assert "interests" in data


def test_uu_clear(client):
    r = client.delete("/api/user-understanding/profile", params={"learner_id": "UUTEST01"})
    assert r.status_code == 200
    r2 = client.post("/api/user-understanding/profile", json={"learner_id": "UUTEST01"})
    data = r2.json().get("data") or {}
    assert data.get("interests") == []
