"""L7 API — 认证与授权 (auth.py).

任务拆分 T6 · 设计文档 Ch.9.6。

JWT 双 Token + RBAC:

| Token | 有效期 | 存储 |
|---|---|---|
| access | 2h | 内存 (防 XSS) |
| refresh | 7d | HttpOnly cookie |

- WebSocket 握手 ?token= 参数
- RBAC: 学生看自己, 教师看所教学生
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

#: Token 有效期 (秒, Ch.9.6)
ACCESS_TTL: int = 7200  # 2h
REFRESH_TTL: int = 604800  # 7d

#: 角色
ROLE_STUDENT = "student"
ROLE_TEACHER = "teacher"
ROLE_ADMIN = "admin"


def _b64url(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign(secret: str, payload_b64: str) -> str:
    sig = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    return _b64url(sig)


class TokenManager:
    """JWT 双 Token 管理器 (HS256).

    简化实现: 自包含签名 (header.payload.signature), 线程安全。
    """

    def __init__(self, secret_key: str = "dev-secret") -> None:
        import threading

        self._secret = secret_key
        self._lock = threading.RLock()
        self._jti_blacklist: set[str] = set()
        self._refresh_store: dict[str, dict[str, Any]] = {}  # refresh_jti -> {user_id, expires}

    def issue_tokens(self, user_id: str, role: str) -> dict[str, str]:
        """颁发 access + refresh token.

        Returns:
            {access_token, refresh_token, token_type, expires_in, user_id, role}
        """
        import uuid

        now = time.time()
        # uuid 保证 jti 全局唯一 (毫秒时间戳在快速连续调用下会冲突)
        access_jti = f"jti-{uuid.uuid4().hex[:16]}"
        refresh_jti = f"rjt-{uuid.uuid4().hex[:16]}"
        access = self._encode({
            "sub": user_id,
            "role": role,
            "jti": access_jti,
            "typ": "access",
            "iat": now,
            "exp": now + ACCESS_TTL,
        })
        refresh = self._encode({
            "sub": user_id,
            "role": role,
            "jti": refresh_jti,
            "typ": "refresh",
            "iat": now,
            "exp": now + REFRESH_TTL,
        })
        with self._lock:
            self._refresh_store[refresh_jti] = {
                "user_id": user_id,
                "role": role,
                "expires": now + REFRESH_TTL,
            }
        return {
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "Bearer",
            "expires_in": ACCESS_TTL,
            "user_id": user_id,
            "role": role,
        }

    def verify(self, token: str) -> dict[str, Any] | None:
        """验证 access token.

        Returns:
            解码 payload; 无效/过期返回 None。
        """
        try:
            header_b64, body_b64, sig = token.split(".")
            signing_input = f"{header_b64}.{body_b64}"
            if not hmac.compare_digest(sig, _sign(self._secret, signing_input)):
                return None
            payload = json.loads(self._decode(body_b64))
        except Exception:  # noqa: BLE001
            return None
        if payload.get("typ") != "access":
            return None
        if payload.get("exp", 0) < time.time():
            return None
        with self._lock:
            if payload.get("jti") in self._jti_blacklist:
                return None
        return payload

    def refresh(self, refresh_token: str) -> dict[str, str] | None:
        """用 refresh token 无感刷新 access.

        Returns:
            新 token 对; refresh 无效/过期返回 None。
        """
        try:
            header_b64, body_b64, sig = refresh_token.split(".")
            signing_input = f"{header_b64}.{body_b64}"
            if not hmac.compare_digest(sig, _sign(self._secret, signing_input)):
                return None
            payload = json.loads(self._decode(body_b64))
        except Exception:  # noqa: BLE001
            return None
        if payload.get("typ") != "refresh":
            return None
        if payload.get("exp", 0) < time.time():
            return None
        jti = payload.get("jti")
        with self._lock:
            record = self._refresh_store.get(jti)
            if record is None or record["expires"] < time.time():
                return None
        return self.issue_tokens(record["user_id"], record["role"])

    def revoke(self, access_token: str) -> None:
        """吊销 access token (登出)."""
        payload = self.verify(access_token)
        if payload and payload.get("jti"):
            with self._lock:
                self._jti_blacklist.add(payload["jti"])

    def _encode(self, payload: dict[str, Any]) -> str:
        header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        body = _b64url(json.dumps(payload, sort_keys=True).encode())
        signing_input = f"{header}.{body}"
        return f"{signing_input}.{_sign(self._secret, signing_input)}"

    @staticmethod
    def _decode(payload_b64: str) -> str:
        import base64

        padding = "=" * (-len(payload_b64) % 4)
        return base64.urlsafe_b64decode(payload_b64 + padding).decode()


def extract_token(authorization: str | None, ws_token: str | None = None) -> str | None:
    """从 Authorization header 或 WebSocket query 提取 token.

    Args:
        authorization: "Bearer {token}"。
        ws_token: WebSocket 握手 ?token= 参数。

    Returns:
        token 字符串; 无返回 None。
    """
    if ws_token:
        return ws_token
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


class AccessControl:
    """RBAC 数据访问控制 (Ch.9.6).

    规则:
    - student: 只能看自己的学情数据
    - teacher: 可看所教学生的学情数据
    - admin: 全量
    """

    def can_view_learner(self, role: str, viewer_id: str, learner_id: str, teacher_students: set[str] | None = None) -> bool:
        """判断角色是否有权查看某学习者数据.

        Args:
            role: 请求者角色。
            viewer_id: 请求者 ID。
            learner_id: 目标学习者 ID。
            teacher_students: 教师所教学生集合。

        Returns:
            True 允许访问。
        """
        if role == ROLE_ADMIN:
            return True
        if role == ROLE_STUDENT:
            return viewer_id == learner_id
        if role == ROLE_TEACHER:
            return teacher_students is not None and learner_id in teacher_students
        return False


#: 并发限制 (Ch.9.8: 渲染 30 次/分钟)
RENDER_RATE_LIMIT_PER_MINUTE: int = 30
