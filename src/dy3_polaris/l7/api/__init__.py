"""L7 体验呈现层 — REST API 路由层."""

from .router import L7Router
from .openapi import (
    generate_openapi_spec,
    openapi_handler,
    swagger_ui_html,
    swagger_ui_handler,
)
from .error_codes import (
    ERROR_CODES,
    all_error_codes,
    error_payload,
    http_status_for,
)
from .artifact_api import get_artifact, list_artifacts, edit_artifact
from .dashboard_api import bkt_dashboard, contribution_for_session, provenance_for_kp
from .websocket import (
    CHANNELS,
    HEARTBEAT_INTERVAL,
    MAX_CONNECTIONS_PER_USER,
    RECONNECT_BACKOFF,
    ConnectionManager,
    build_message,
    reconnect_delay,
)
from .auth import (
    ACCESS_TTL,
    REFRESH_TTL,
    AccessControl,
    TokenManager,
    extract_token,
)

#: 便捷别名 — TokenManager.issue_tokens (供路由复用)
issue_tokens_wrapper = TokenManager().issue_tokens

__all__ = [
    "L7Router",
    "generate_openapi_spec",
    "openapi_handler",
    "swagger_ui_html",
    "swagger_ui_handler",
    # error codes
    "ERROR_CODES",
    "all_error_codes",
    "error_payload",
    "http_status_for",
    # artifact api
    "get_artifact",
    "list_artifacts",
    "edit_artifact",
    # dashboard api
    "bkt_dashboard",
    "contribution_for_session",
    "provenance_for_kp",
    # websocket
    "CHANNELS",
    "HEARTBEAT_INTERVAL",
    "MAX_CONNECTIONS_PER_USER",
    "RECONNECT_BACKOFF",
    "ConnectionManager",
    "build_message",
    "reconnect_delay",
    # auth
    "ACCESS_TTL",
    "REFRESH_TTL",
    "AccessControl",
    "TokenManager",
    "extract_token",
    "issue_tokens_wrapper",
]
