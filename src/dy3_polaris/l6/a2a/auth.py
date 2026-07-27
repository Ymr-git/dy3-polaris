"""A2A 认证与安全工具.

提供协议级认证能力：
- 简易 Token 验证（生产环境替换为 JWT/Ed25519）
- Agent 身份指纹生成
- 会话 Token 管理

注意: 当前实现为教学演示级别的简化认证。
生产部署应替换为 DID + JWT + Ed25519 标准方案。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any


# ============================================================
# Token 管理
# ============================================================

class TokenStore:
    """轻量 Token 存储与管理.

    支持生成、验证和吊销会话级 Token。
    Token 格式: hex(secret + ":" + expire_timestamp + ":" + agent_id)
    """

    def __init__(self, secret_key: str = "") -> None:
        self._secret = (secret_key or secrets.token_hex(32)).encode("utf-8")
        self._tokens: dict[str, dict[str, Any]] = {}  # token -> {agent_id, session_id, expires_at}

    def generate(self, agent_id: str, session_id: str = "", ttl_seconds: float = 3600) -> str:
        """生成 Token.

        Args:
            agent_id: Agent ID
            session_id: 关联会话 ID
            ttl_seconds: 有效期（秒）

        Returns:
            Token 字符串
        """
        expires_at = time.time() + ttl_seconds
        raw = f"{secrets.token_hex(16)}:{expires_at:.0f}:{agent_id}"
        token = hmac.new(self._secret, raw.encode("utf-8"), hashlib.sha256).hexdigest()[:32]

        self._tokens[token] = {
            "agent_id": agent_id,
            "session_id": session_id,
            "expires_at": expires_at,
        }
        return token

    def validate(self, token: str) -> dict[str, Any] | None:
        """验证 Token.

        Returns:
            Token 信息字典（验证通过），过期或无效返回 None
        """
        info = self._tokens.get(token)
        if info is None:
            return None

        if time.time() > info["expires_at"]:
            del self._tokens[token]
            return None

        return info

    def revoke(self, token: str) -> bool:
        """吊销 Token."""
        return self._tokens.pop(token, None) is not None

    def cleanup_expired(self) -> int:
        """清理所有过期 Token."""
        now = time.time()
        expired = [t for t, info in self._tokens.items() if now > info["expires_at"]]
        for t in expired:
            del self._tokens[t]
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._tokens)

    def reset(self) -> None:
        self._tokens.clear()


# ============================================================
# Agent 身份指纹
# ============================================================

def agent_fingerprint(agent_id: str, capabilities: list[str], secret: str = "") -> str:
    """生成 Agent 身份指纹.

    基于 Agent ID + 能力列表生成确定性指纹，
    可用于简化认证场景下的身份校验。

    Args:
        agent_id: Agent ID
        capabilities: 能力列表
        secret: 密钥（可选，增加安全性）

    Returns:
        16 字符十六进制指纹
    """
    payload = f"{agent_id}:{','.join(sorted(capabilities))}:{secret}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def verify_fingerprint(
    agent_id: str,
    capabilities: list[str],
    fingerprint: str,
    secret: str = "",
) -> bool:
    """验证 Agent 指纹是否匹配."""
    return agent_fingerprint(agent_id, capabilities, secret) == fingerprint


__all__ = ["TokenStore", "agent_fingerprint", "verify_fingerprint"]
