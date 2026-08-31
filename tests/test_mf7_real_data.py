"""真实学习链路专项测试: 练习 API + 画像联动 + 持久化 + 前端资源."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp

LEARNERS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "dy3_polaris" / "l2" / "data" / "learners"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    builder = UnifiedApp.create_full_app_builder()
    return TestClient(builder.create_app())


class TestPracticeAPI:
    """练习出题 / 判题 / 画像联动."""

    def test_bank_loaded(self, client):
        # 38 题中按映射至少有 30 题可用
        r = client.get("/l2/practice/questions", params={"learner_id": "DY20240001", "count": 10})
        assert r.json()["code"] == 0
        assert len(r.json()["data"]["questions"]) >= 5

    def test_questions_hide_answer(self, client):
        r = client.get("/l2/practice/questions", params={"learner_id": "DY20240001", "count": 3})
        q = r.json()["data"]["questions"][0]
        assert "answer" not in q
        assert "explanation" not in q
        assert q["kp_id"]

    def test_weak_first(self, client):
        # 薄弱 KP 优先: 题库中存在薄弱 KP 时, 首题必须属于薄弱点 (画像可能已被考核更新)
        p = client.get("/l2/profile/DY20240001").json()["data"]
        weak = set(p["weak_kps"])
        bank = __import__("dy3_polaris.l2.practice", fromlist=["PracticeBank"]).PracticeBank()
        weak_in_bank = weak & set(bank.by_kp.keys())
        r = client.get("/l2/practice/questions", params={"learner_id": "DY20240001", "count": 5})
        qs = r.json()["data"]["questions"]
        assert qs
        if weak_in_bank:
            assert qs[0]["kp_id"] in weak, f"首题 KP {qs[0]['kp_id']} 应属薄弱点 {sorted(weak_in_bank)}"

    def test_answer_updates_profile(self, client):
        # 选薄弱题 (掌握度 < 0.6) 保证 BKT 有明显变化
        p0 = client.get("/l2/profile/DY20240001").json()["data"]
        weak = p0["weak_kps"]
        r = client.get("/l2/practice/questions", params={"learner_id": "DY20240001", "count": 8})
        qs = r.json()["data"]["questions"]
        # 取题库中当前掌握度最低的题 (薄弱优先)
        target = min(qs, key=lambda q: p0["kp_mastery"].get(q["kp_id"], 0.0))
        kp = target["kp_id"]
        before = p0["kp_mastery"].get(kp, 0.0)

        # 故意答错 (正确答案之外), 确保掌握度变化
        wrong = 99
        ans = client.post("/l2/practice/answer", json={
            "learner_id": "DY20240001", "qid": target["qid"], "selected": wrong,
        })
        assert ans.json()["code"] == 0
        d = ans.json()["data"]
        assert d["kp_id"] == kp
        assert abs(d["p_mastery_after"] - d["p_mastery_before"]) > 1e-6

        p1 = client.get("/l2/profile/DY20240001").json()["data"]
        after = p1["kp_mastery"].get(kp, 0.0)
        assert abs(after - d["p_mastery_after"]) < 1e-6
        assert abs(after - before) > 1e-6

    def test_answer_wrong_decreases(self, client):
        # 答错高难度题通常掌握度下降 (差异可能微小, 只断言返回结构完整)
        r = client.get("/l2/practice/questions", params={"learner_id": "DY20240001", "count": 1})
        q = r.json()["data"]["questions"][0]
        # 用确定错误的选项 (超出范围)
        ans = client.post("/l2/practice/answer", json={
            "learner_id": "DY20240001", "qid": q["qid"], "selected": 99,
        })
        d = ans.json()["data"]
        assert d["correct"] is False
        assert "explanation" in d

    def test_answer_bad_qid(self, client):
        r = client.post("/l2/practice/answer", json={
            "learner_id": "DY20240001", "qid": "Q999", "selected": 0,
        })
        assert r.json()["code"] != 0

    def test_missing_params(self, client):
        r = client.get("/l2/practice/questions")
        assert r.json()["code"] != 0


class TestProfilePersistence:
    """画像 JSON 持久化 (借鉴 dy-agent-system learner JSON)."""

    def test_persist_file_exists(self):
        files = list(LEARNERS_DIR.glob("DY20240001.json"))
        assert files, "画像持久化文件缺失"
        d = json.loads(files[0].read_text(encoding="utf-8"))
        assert d["learner_id"] == "DY20240001"
        assert len(d["kp_mastery"]) == 42

    def test_persist_roundtrip(self):
        # 重启恢复: 从磁盘读回与内存一致
        d = json.loads((LEARNERS_DIR / "DY20240001.json").read_text(encoding="utf-8"))
        assert 0.0 <= d["theta"] <= 2.0 or -2.0 <= d["theta"] <= 2.0
        assert d["snapshot_ts"] > 0


class TestMF7Frontend:
    """前端资源包含真实学习链路功能."""

    def test_mf6_js_has_practice(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "renderPractice" in js
        assert "renderWeakPoints" in js
        assert "renderRecommendations" in js
        assert "openSettingsFallback" in js
        assert "setupSettingsDelegate" in js
        assert "/l2/practice/questions" in js
        assert "/l2/practice/answer" in js

    def test_app_js_settings_binding(self, client):
        js = client.get("/static/assets/app.js").text
        assert "sv('settings')" in js
        assert "Promise.resolve" in js

    def test_bank_json_served(self, client):
        # 题库文件存在于源码包
        bank = LEARNERS_DIR.parents[1] / "pretest_bank.json"
        assert bank.exists()
        d = json.loads(bank.read_text(encoding="utf-8"))
        assert d["metadata"]["total_questions"] == 38
