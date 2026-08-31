"""Authoritative task lifecycle and JSON/JSONL learning-task persistence.

The public helpers keep the original P0 API, while the implementation now
supports the frozen 13-state lifecycle and one durable task fact.  The store is
file based: it is an adapter over the existing project data root, not a second
database, Artifact system, or kernel checkpoint runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid
from typing import Any, Mapping


TASK_STATES = (
    "CREATED", "UNDERSTANDING", "PLANNING", "RETRIEVING",
    "COLLABORATING", "REVIEWING", "RETRYING", "NEEDS_CONFIRMATION",
    "ANSWERING", "COMPLETED", "PARTIAL", "FAILED", "CANCELLED",
)
P0_TASK_STATES = TASK_STATES
TERMINAL_TASK_STATES = frozenset({"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"})

_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "CREATED": frozenset({"UNDERSTANDING", "FAILED", "CANCELLED"}),
    "UNDERSTANDING": frozenset({"PLANNING", "RETRIEVING", "PARTIAL", "FAILED", "CANCELLED"}),
    "PLANNING": frozenset({"RETRIEVING", "PARTIAL", "FAILED", "CANCELLED"}),
    "RETRIEVING": frozenset({"COLLABORATING", "PARTIAL", "FAILED", "CANCELLED"}),
    "COLLABORATING": frozenset({
        "REVIEWING", "NEEDS_CONFIRMATION", "PARTIAL", "FAILED", "CANCELLED",
    }),
    "REVIEWING": frozenset({
        "RETRYING", "NEEDS_CONFIRMATION", "ANSWERING", "PARTIAL", "FAILED", "CANCELLED",
    }),
    "RETRYING": frozenset({"RETRIEVING", "FAILED", "CANCELLED"}),
    "NEEDS_CONFIRMATION": frozenset({"COLLABORATING", "CANCELLED", "FAILED"}),
    "ANSWERING": frozenset({"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}),
    "COMPLETED": frozenset(), "PARTIAL": frozenset(),
    "FAILED": frozenset(), "CANCELLED": frozenset(),
}

TASK_EVENT_TYPES = (
    "TaskCreated", "TaskResumed", "StateChanged", "RetrievalCompleted",
    "AgentStarted", "AgentContributionRecorded", "AgentFinished",
    "ReviewCompleted", "ReviewerChallengeRaised", "RevisionApplied",
    "ReleaseDecided", "ResourceIssued", "PracticeSaved",
    "LearnerProjectionUpdated", "TeachingActionRequested",
    "TaskCompleted", "TaskFailed",
)
P0_TASK_EVENT_TYPES = TASK_EVENT_TYPES


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return str(value)


def _emit_to_sink(task_context: dict[str, Any], event: dict[str, Any]) -> None:
    sink = task_context.get("_event_sink")
    if callable(sink):
        sink(dict(event))


def create_task_context(
    task_id: str,
    producer: str = "task_engine",
    *,
    learner_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Create one request view over the authoritative task identity."""
    if not task_id:
        raise ValueError("task_id is required")
    task_context: dict[str, Any] = {
        "task_id": str(task_id), "learner_id": str(learner_id),
        "session_id": str(session_id), "state": "CREATED", "events": [],
    }
    record_task_event(
        task_context, "StateChanged", producer=producer,
        state_before="", state_after="CREATED",
    )
    record_task_event(
        task_context, "TaskCreated", producer=producer,
        details={"schema_version": 1},
    )
    return task_context


def record_task_event(
    task_context: Any,
    event_type: str,
    producer: str,
    *,
    state_before: str | None = None,
    state_after: str | None = None,
    details: dict[str, Any] | None = None,
    event_id: str | None = None,
) -> dict[str, Any] | None:
    """Append a public, bounded fact at the point where it occurred."""
    if task_context is None:
        return None
    if not isinstance(task_context, dict):
        raise TypeError("task_context must be a dict")
    if event_type not in TASK_EVENT_TYPES:
        raise ValueError(f"unsupported task event type: {event_type}")
    current = get_task_state(task_context)
    before = current if state_before is None else str(state_before)
    after = current if state_after is None else str(state_after)
    event = {
        "event_id": str(event_id or f"task-event-{uuid.uuid4().hex}"),
        "task_id": str(task_context.get("task_id") or ""),
        "event_kind": "state" if event_type == "StateChanged" else "activity",
        "event_type": event_type,
        "timestamp": time.time(),
        "producer": str(producer),
        "state_before": before,
        "state_after": after,
    }
    if details:
        event["details"] = _json_safe(details)
    events = task_context.setdefault("events", [])
    if not isinstance(events, list):
        raise TypeError("task_context events must be a list")
    for item in events:
        if isinstance(item, dict) and item.get("event_id") == event["event_id"]:
            return item
    events.append(event)
    _emit_to_sink(task_context, event)
    return event


def set_task_state(
    task_context: Any,
    state: str,
    producer: str = "task_engine",
) -> str:
    """Apply one legal transition; retry is the only backward execution arc."""
    if task_context is None:
        return ""
    if not isinstance(task_context, dict):
        raise TypeError("task_context must be a dict")
    if state not in TASK_STATES:
        raise ValueError(f"unsupported task state: {state}")
    current = str(task_context.get("state") or "")
    if current not in TASK_STATES:
        raise ValueError(f"invalid current task state: {current}")
    if state == current:
        return state
    if state not in _VALID_TRANSITIONS[current]:
        raise ValueError(f"task state cannot transition: {current} -> {state}")
    task_context["state"] = state
    record_task_event(
        task_context, "StateChanged", producer=producer,
        state_before=current, state_after=state,
    )
    if state == "COMPLETED":
        record_task_event(task_context, "TaskCompleted", producer=producer)
    elif state == "FAILED":
        record_task_event(task_context, "TaskFailed", producer=producer)
    return state


def get_task_state(task_context: Any) -> str:
    if not isinstance(task_context, dict):
        return ""
    state = str(task_context.get("state") or "")
    return state if state in TASK_STATES else ""


def get_task_events(task_context: Any) -> list[dict[str, Any]]:
    if not isinstance(task_context, dict):
        return []
    events = task_context.get("events")
    if not isinstance(events, list):
        return []
    return [dict(event) for event in events if isinstance(event, dict)]


class LearningTaskStore:
    """Single durable source for public learning-task snapshots and events."""

    SCHEMA_VERSION = 1
    _SAFE_ID = re.compile(r"^[A-Za-z0-9_.:@+-]{1,160}$")

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.events_dir = self.root / "events"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._event_ids: dict[str, set[str]] = {}

    def _validate_id(self, value: str, name: str) -> str:
        parsed = str(value or "")
        if not self._SAFE_ID.fullmatch(parsed):
            raise ValueError(f"invalid {name}")
        return parsed

    def _snapshot_path(self, task_id: str) -> Path:
        return self.snapshots_dir / f"{self._validate_id(task_id, 'task_id')}.json"

    def _event_path(self, task_id: str) -> Path:
        return self.events_dir / f"{self._validate_id(task_id, 'task_id')}.jsonl"

    @staticmethod
    def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def create_task(
        self,
        *,
        task_context: dict[str, Any],
        learner_id: str,
        session_id: str,
        query: str,
        goal: str = "",
        brief: str = "",
    ) -> dict[str, Any]:
        task_id = self._validate_id(str(task_context.get("task_id") or ""), "task_id")
        learner_id = self._validate_id(str(learner_id), "learner_id")
        now = time.time()
        with self._lock:
            if self._snapshot_path(task_id).exists():
                existing = self.get_task(task_id, learner_id)
                if existing is None:
                    raise PermissionError("task belongs to another learner")
                return existing
            record = {
                "schema_version": self.SCHEMA_VERSION, "revision": 1,
                "task_id": task_id, "learner_id": learner_id,
                "session_id": str(session_id or ""), "query": str(query),
                "goal": str(goal), "brief": str(brief or query)[:500],
                "state": get_task_state(task_context), "created_at": now,
                "updated_at": now, "task_events": [],
                "agent_contributions": [],
                "reviewer": {"challenges": [], "revisions": [], "release": {}},
                "answer": {"identity": "", "version": 0, "text": ""},
                "claim_evidence_source": [],
                "resource_plan": {"resources": []}, "practice_plan": {},
                "practice_receipts": [], "teaching_decision": {},
                "next_action": {}, "resume_route": "learning",
                "failures": [], "degradation": {}, "learning_blocks": [],
                "public_result": {},
            }
            self._atomic_write(self._snapshot_path(task_id), record)
            self._event_ids[task_id] = set()
            for event in get_task_events(task_context):
                self.append_event(event)
            return self.get_task(task_id, learner_id) or record

    def _known_event_ids(self, task_id: str) -> set[str]:
        cached = self._event_ids.get(task_id)
        if cached is not None:
            return cached
        ids: set[str] = set()
        path = self._event_path(task_id)
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_id = str(value.get("event_id") or "")
                    if event_id:
                        ids.add(event_id)
        self._event_ids[task_id] = ids
        return ids

    def append_event(self, event: Mapping[str, Any]) -> bool:
        task_id = self._validate_id(str(event.get("task_id") or ""), "task_id")
        event_id = self._validate_id(str(event.get("event_id") or ""), "event_id")
        safe_event = _json_safe(event)
        with self._lock:
            known = self._known_event_ids(task_id)
            if event_id in known:
                return False
            snapshot = self._read_snapshot(task_id)
            if snapshot is None:
                raise FileNotFoundError(f"task snapshot not found: {task_id}")
            with self._event_path(task_id).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe_event, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            known.add(event_id)
            events = list(snapshot.get("task_events") or [])
            events.append(safe_event)
            snapshot["task_events"] = events
            if str(event.get("event_type") or "") == "StateChanged":
                snapshot["state"] = str(event.get("state_after") or snapshot.get("state") or "")
            snapshot["revision"] = int(snapshot.get("revision", 0) or 0) + 1
            snapshot["updated_at"] = time.time()
            self._atomic_write(self._snapshot_path(task_id), snapshot)
            return True

    def bind_context(self, task_context: dict[str, Any]) -> None:
        task_context["_event_sink"] = self.append_event

    def record_activity(
        self,
        task_id: str,
        learner_id: str,
        event_type: str,
        producer: str,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.get_task(task_id, learner_id)
        if snapshot is None:
            raise FileNotFoundError(f"task not found: {task_id}")
        if event_type not in TASK_EVENT_TYPES or event_type == "StateChanged":
            raise ValueError(f"unsupported activity event type: {event_type}")
        state = str(snapshot.get("state") or "")
        event = {
            "event_id": f"task-event-{uuid.uuid4().hex}",
            "task_id": task_id,
            "event_kind": "activity",
            "event_type": event_type,
            "timestamp": time.time(),
            "producer": str(producer),
            "state_before": state,
            "state_after": state,
        }
        if details:
            event["details"] = _json_safe(details)
        self.append_event(event)
        return event

    def transition_task(
        self,
        task_id: str,
        learner_id: str,
        state: str,
        producer: str,
    ) -> dict[str, Any]:
        snapshot = self.get_task(task_id, learner_id)
        if snapshot is None:
            raise FileNotFoundError(f"task not found: {task_id}")
        current = str(snapshot.get("state") or "")
        if state not in TASK_STATES:
            raise ValueError(f"unsupported task state: {state}")
        if state == current:
            return snapshot
        if state not in _VALID_TRANSITIONS.get(current, frozenset()):
            raise ValueError(f"task state cannot transition: {current} -> {state}")
        event = {
            "event_id": f"task-event-{uuid.uuid4().hex}",
            "task_id": task_id,
            "event_kind": "state",
            "event_type": "StateChanged",
            "timestamp": time.time(),
            "producer": str(producer),
            "state_before": current,
            "state_after": state,
        }
        self.append_event(event)
        if state == "COMPLETED":
            self.record_activity(task_id, learner_id, "TaskCompleted", producer)
        elif state == "FAILED":
            self.record_activity(task_id, learner_id, "TaskFailed", producer)
        return self.get_task(task_id, learner_id) or snapshot

    def update_task(
        self, task_id: str, learner_id: str, values: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            snapshot = self.get_task(task_id, learner_id)
            if snapshot is None:
                raise FileNotFoundError(f"task not found: {task_id}")
            immutable = {"task_id", "learner_id", "created_at", "schema_version"}
            for key, value in values.items():
                if key in immutable or str(key).startswith("_"):
                    continue
                snapshot[str(key)] = _json_safe(value)
            snapshot["revision"] = int(snapshot.get("revision", 0) or 0) + 1
            snapshot["updated_at"] = time.time()
            self._atomic_write(self._snapshot_path(task_id), snapshot)
            return snapshot

    def _read_snapshot(self, task_id: str) -> dict[str, Any] | None:
        path = self._snapshot_path(task_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def get_task(self, task_id: str, learner_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._read_snapshot(task_id)
            if value is None:
                return None
            if str(value.get("learner_id") or "") != str(learner_id or ""):
                raise PermissionError("task belongs to another learner")
            return _json_safe(value)

    def list_tasks(self, learner_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        learner_id = self._validate_id(str(learner_id), "learner_id")
        values: list[dict[str, Any]] = []
        with self._lock:
            for path in self.snapshots_dir.glob("task-*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if str(item.get("learner_id") or "") == learner_id:
                    values.append(item)
        values.sort(key=lambda item: float(item.get("updated_at", 0.0) or 0.0), reverse=True)
        return [_json_safe(item) for item in values[: max(1, min(int(limit), 200))]]


__all__ = [
    "LearningTaskStore", "P0_TASK_EVENT_TYPES", "P0_TASK_STATES",
    "TASK_EVENT_TYPES", "TASK_STATES", "TERMINAL_TASK_STATES",
    "create_task_context", "get_task_events", "get_task_state",
    "record_task_event", "set_task_state",
]
