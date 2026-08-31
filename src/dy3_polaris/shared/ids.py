"""统一 ID 工厂 — 全系统 ID 命名空间单点 (SSOT).

前缀规范 (分层前缀, 统一 12 位 hex 后缀):
- ``l1``  : ``sess-``  L1 用户会话 (唯一对外会话入口, 前端只持此 ID)
- ``l2``  : ``l2s-``  L2 学习/交互会话 (内部执行上下文)
- ``l5``  : ``ag-``   L5 Agent 执行会话 (经 source_session_id 关联 L1)
- ``ws``  : ``ws-``   L5 WorkingSession (Agent 实例工作会话)
- ``fork``: ``fork-`` L5 会话 Fork 记录
- ``a2a`` : ``a2a-``  L6 Agent 间会话 (确定性哈希)
- ``nego``: ``nego-`` L0 治理协商记录
- ``tr``  : ``tr-``   请求级 trace_id (见 l5/tracing.py)
"""
from __future__ import annotations

import uuid

#: 层 → 前缀
LAYER_PREFIX: dict[str, str] = {
    "l1": "sess",
    "l2": "l2s",
    "l5": "ag",
    "ws": "ws",
    "fork": "fork",
    "a2a": "a2a",
    "nego": "nego",
    "tr": "tr",
}


def new_id(kind: str, *, hex_len: int = 12) -> str:
    """生成带命名空间前缀的随机 ID.

    Args:
        kind: 语义层 (见 LAYER_PREFIX).
        hex_len: 随机段长度 (默认 12 位 hex).

    Returns:
        形如 ``{prefix}-{hex}`` 的 ID; 未知 kind 回退为 ``{kind}-{hex}``.
    """
    prefix = LAYER_PREFIX.get(kind, kind)
    return f"{prefix}-{uuid.uuid4().hex[:hex_len]}"


def new_session_id(layer: str = "l1") -> str:
    """生成会话 ID (统一命名空间, 默认 L1 用户会话)."""
    return new_id(layer)
