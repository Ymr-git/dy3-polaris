"""Product-loop acceptance for the durable learning-task truth."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5 import task_state
from dy3_polaris.l5.unified_app import UnifiedApp


def _created_store(tmp_path, *, learner_id: str = "learner-a"):
    store = task_state.LearningTaskStore(tmp_path / "tasks")
    context = task_state.create_task_context(
        "task-persistence-1", learner_id=learner_id, session_id="session-1",
    )
    store.create_task(
        task_context=context,
        learner_id=learner_id,
        session_id="session-1",
        query="Dy3+ 为什么有黄蓝双发射？",
    )
    store.bind_context(context)
    return store, context


def test_snapshot_is_atomic_json_and_events_are_idempotent(tmp_path) -> None:
    store, context = _created_store(tmp_path)
    event_id = "task-event-idempotent"
    event = task_state.record_task_event(
        context,
        "AgentStarted",
        "agent.learning.diagnosis",
        event_id=event_id,
    )
    assert event is not None
    assert store.append_event(event) is False

    snapshot_path = store.snapshots_dir / "task-persistence-1.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["task_id"] == "task-persistence-1"
    assert sum(item["event_id"] == event_id for item in snapshot["task_events"]) == 1
    assert not list(store.snapshots_dir.glob("*.tmp"))


def test_restart_restores_same_task_and_learner_isolation(tmp_path) -> None:
    store, context = _created_store(tmp_path)
    task_state.set_task_state(context, "UNDERSTANDING")
    task_state.set_task_state(context, "PLANNING")
    store.update_task(
        "task-persistence-1", "learner-a",
        {"resume_route": "learning", "next_action": {"action_type": "PRACTICE"}},
    )

    restarted = task_state.LearningTaskStore(tmp_path / "tasks")
    restored = restarted.get_task("task-persistence-1", "learner-a")
    assert restored is not None
    assert restored["task_id"] == "task-persistence-1"
    assert restored["state"] == "PLANNING"
    assert restored["next_action"]["action_type"] == "PRACTICE"
    with pytest.raises(PermissionError):
        restarted.get_task("task-persistence-1", "learner-b")
    assert restarted.list_tasks("learner-b") == []


def test_task_api_survives_app_restart_and_resume_keeps_identity(tmp_path) -> None:
    data_dir = str(tmp_path / "app-data")
    first = UnifiedApp.create_full_app_builder(data_dir=data_dir)
    store = first._handlers._task_store
    context = task_state.create_task_context(
        "task-api-restart", learner_id="learner-api", session_id="session-api",
    )
    store.create_task(
        task_context=context,
        learner_id="learner-api",
        session_id="session-api",
        query="浓度猝灭的机制是什么？",
    )
    store.update_task(
        "task-api-restart", "learner-api",
        {"brief": "浓度猝灭", "resume_route": "learning"},
    )

    second = UnifiedApp.create_full_app_builder(data_dir=data_dir)
    client = TestClient(second.create_app())
    listed = client.get("/api/learning-tasks/learner-api")
    assert listed.status_code == 200
    assert listed.json()["data"]["tasks"][0]["task_id"] == "task-api-restart"

    resumed = client.post(
        "/api/learning-tasks/learner-api/task-api-restart/resume", json={},
    )
    assert resumed.status_code == 200
    payload = resumed.json()["data"]
    assert payload["task"]["task_id"] == "task-api-restart"
    assert payload["resume_route"] == "learning"
    assert any(
        item["event_type"] == "TaskResumed"
        for item in payload["task"]["task_events"]
    )

    assert client.get(
        "/api/learning-tasks/learner-other/task-api-restart"
    ).status_code == 404


def test_api_query_persists_the_same_public_task_result(tmp_path) -> None:
    builder = UnifiedApp.create_full_app_builder(data_dir=str(tmp_path / "query-data"))
    client = TestClient(builder.create_app())
    response = client.post(
        "/api/query",
        json={
            "learner_id": "learner-query-persisted",
            "query": "Dy3+ 为什么会产生黄蓝双发射？",
        },
    )
    assert response.status_code == 200
    public = response.json()["data"]
    detail = client.get(
        f"/api/learning-tasks/learner-query-persisted/{public['task_id']}"
    )
    assert detail.status_code == 200
    task = detail.json()["data"]["task"]
    assert task["task_id"] == public["task_id"]
    assert task["state"] == public["task_state"]
    assert task["public_result"]["task_id"] == public["task_id"]
    assert task["public_result"]["answer"] == public["answer"]
    assert task["resource_plan"]["resources"] == public["learning_resources"]
