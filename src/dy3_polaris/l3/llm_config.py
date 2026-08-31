"""可插拔、多模型 LLM 配置层.

设计目标:
- 用户只需填写 API Key (环境变量或 .env 文件), 无需改动代码.
- 内置多家 API 预设 (DeepSeek / Claude / OpenAI / 国内兼容服务).
- 按工作负载选择模型，而不是让所有 Agent 共用一个模型。
- 密钥绝不硬编码进源码, 仅从环境变量 / .env 读取, 防止随文件复制泄漏.

使用方式 (任选其一):
1. 环境变量:  export DY3_LLM_PROVIDER=deepseek  &&  export DY3_LLM_API_KEY=sk-xxx
2. .env 文件 (项目根目录):  见 .env.example, 填入密钥后即可.

支持的 provider (开箱即用):
- deepseek     : DeepSeek 官方 (https://api.deepseek.com)
- qwen         : 通义千问 (阿里云 DashScope 兼容 OpenAI 协议)
- zhipu        : 智谱 GLM (https://open.bigmodel.cn)
- anthropic    : Anthropic Claude 官方 Messages API
- openai       : OpenAI 官方 Responses API（GPT-5 系列）
- custom       : 任意 OpenAI 兼容服务 (需同时填 base_url + model)

安全说明:
- 本项目 .gitignore 已忽略 .env 与 .env.local.
- 所有读取只发生在运行时, 密钥不进入任何序列化输出 / 日志 / 报告.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

# ============================================================
# 各家 API 预设 (provider -> 默认 base_url / model)
# ============================================================

_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-5.1",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.6",
    },
    "minimax": {
        "base_url": "https://api.minimax.chat/v1",
        "model": "minimax-m2.7",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.6-terra",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-5",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
    },
    "custom": {
        "base_url": "",
        "model": "",
    },
}


_call_status_lock = RLock()
_last_call_status: dict[str, dict[str, Any]] = {}


def _record_call_status(
    role: str,
    cfg: "LLMConfig",
    *,
    success: bool,
    failure_kind: str = "",
    http_status: int | None = None,
) -> None:
    """Keep only non-secret connection facts; never retain prompts or responses."""

    status = {
        "role": str(role or "default"),
        "provider": cfg.provider,
        "model": cfg.resolve_model(),
        "configured": cfg.is_ready(),
        "success": bool(success),
        "failure_kind": failure_kind,
        "http_status": http_status,
    }
    with _call_status_lock:
        _last_call_status[status["role"]] = status


def last_model_call_status(role: str = "default") -> dict[str, Any]:
    """Return the latest redacted call status for diagnostics and health checks."""

    with _call_status_lock:
        return dict(_last_call_status.get(str(role or "default"), {}))

# 角色默认值只在相应 provider 已有真实密钥时使用。当前 DeepSeek 单密钥也能
# 形成 Flash（快速）+ Pro（高精度）双模型；若配置 Claude/OpenAI 密钥，则长文
# 与深度推理会自动使用各自擅长的模型。Diagnosis/Guidance 仍由确定性契约负责，
# 不为了“多模型”而把可靠状态解释和发布决策交给概率模型。
_ROLE_MODEL_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {
        "semantic_fast": "deepseek-v4-flash",
        "generation_fast": "deepseek-v4-flash",
        "generation_long": "deepseek-v4-pro",
        "generation_deep": "deepseek-v4-pro",
        "review": "deepseek-v4-pro",
    },
    "anthropic": {
        "semantic_fast": "claude-haiku-4-5",
        "generation_fast": "claude-haiku-4-5",
        "generation_long": "claude-sonnet-5",
        "generation_deep": "claude-opus-5",
        "review": "claude-opus-5",
    },
    "openai": {
        "semantic_fast": "gpt-5.6-luna",
        "generation_fast": "gpt-5.6-luna",
        "generation_long": "gpt-5.6-terra",
        "generation_deep": "gpt-5.6-terra",
        "review": "gpt-5.6-sol",
    },
}

_ROLE_PROVIDER_PREFERENCE = {
    "generation_long": "anthropic",
    "generation_deep": "openai",
    "review": "openai",
}

# 环境变量前缀
_ENV_PREFIX = "DY3_LLM_"

#: 运行时配置 (由前端「API 配置」页 POST /api/llm/config 写入, 优先级高于 .env)
_runtime_config: dict[str, str] = {}


def _env(name: str, default: str = "") -> str:
    """读取环境变量 (兼容带前缀与不带前缀两种写法)."""
    return os.environ.get(f"{_ENV_PREFIX}{name}", default)


def _read_dotenv() -> dict[str, str]:
    """极简 .env 解析 (无外部依赖), 返回 {KEY: VALUE}.

    仅读取项目根目录的 .env 与 .env.local, 不覆盖已有环境变量.
    解析规则: 忽略空行与 # 注释, 支持 KEY=VALUE 与 KEY="VALUE".
    """
    result: dict[str, str] = {}
    candidates: list[Path] = []
    # 从当前工作目录找 .env (通常项目根)
    cwd = Path.cwd()
    for p in (cwd / ".env", cwd / ".env.local"):
        if p.is_file():
            candidates.append(p)
    # editable install / 从任意目录启动时, cwd 可能远离项目根: 补查包自身所在的项目根
    try:
        import dy3_polaris as _pkg

        _pkg_dir = Path(_pkg.__file__).resolve().parent  # .../04-编码/src/dy3_polaris
        # parents: src/dy3_polaris -> src -> 04-编码 -> (项目根, 兜底)
        for _base in (_pkg_dir.parent.parent, _pkg_dir.parent.parent.parent):
            for _p in (_base / ".env", _base / ".env.local"):
                if _p.is_file() and _p not in candidates:
                    candidates.append(_p)
    except Exception:  # noqa: BLE001
        pass

    for path in candidates:
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    result[key] = value
        except OSError:
            continue
    return result


@dataclass
class LLMConfig:
    """LLM 接入配置.

    Attributes:
        provider: 服务商标识 (deepseek/qwen/zhipu/openai/custom).
        api_key: API 密钥 (绝不打印/序列化).
        base_url: API 端点 (custom 时必填).
        model: 模型名.
        temperature: 采样温度 (0=确定性, 1=随机).
        max_tokens: 最大生成 token 数.
        timeout_seconds: 请求超时.
        enabled: 是否启用 LLM 增强 (无 api_key 时自动 False).
    """

    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: float = 12.0
    enabled: bool = False

    # 密钥脱敏显示 (日志/报告用)
    def masked_key(self) -> str:
        if not self.api_key:
            return "(未配置)"
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def is_ready(self) -> bool:
        """是否具备调用条件.

        - ollama (本地模型): 无需 api_key, 只要 base_url/model 可解析即可.
        - 其他 provider: 需有 api_key.
        """
        if not self.enabled:
            return False
        if self.provider == "ollama":
            return bool(self.resolve_base_url()) and bool(self.resolve_model())
        if not self.api_key:
            return False
        return bool(self.resolve_base_url()) and bool(self.resolve_model())

    def resolve_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return _PROVIDER_PRESETS.get(self.provider, {}).get("base_url", "")

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        return _PROVIDER_PRESETS.get(self.provider, {}).get("model", "")


def _mask_key(key: str) -> str:
    """密钥脱敏显示."""
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def set_runtime_config(
    provider: str,
    api_key: str,
    base_url: str = "",
    model: str = "",
) -> dict[str, str]:
    """设置运行时 LLM 配置 (由前端「API 配置」页 POST /api/llm/config 调用).

    优先级高于 .env / 环境变量; 内存存储, 服务重启后失效 (需重新设置).
    返回脱敏后的配置摘要 (不含明文密钥).
    """
    global _runtime_config
    _runtime_config = {
        "provider": (provider or "").strip().lower(),
        "api_key": (api_key or "").strip(),
        "base_url": (base_url or "").strip(),
        "model": (model or "").strip(),
    }
    return _runtime_summary()


def _runtime_summary() -> dict[str, str]:
    """运行时配置脱敏摘要 (供 /api/llm/config GET 返回)."""
    if not _runtime_config:
        return {"provider": "", "base_url": "", "model": "", "has_key": False, "masked_key": ""}
    key = _runtime_config.get("api_key", "")
    return {
        "provider": _runtime_config.get("provider", ""),
        "base_url": _runtime_config.get("base_url", ""),
        "model": _runtime_config.get("model", ""),
        "has_key": bool(key),
        "masked_key": _mask_key(key),
    }


def _truthy(value: str, default: bool = True) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    return normalized not in {"0", "false", "no", "off", "disabled"}


def multi_model_enabled() -> bool:
    """Return whether role-aware model routing is enabled."""

    dotenv = _read_dotenv()
    value = os.environ.get(
        "DY3_MULTI_MODEL_ENABLED",
        dotenv.get("DY3_MULTI_MODEL_ENABLED", "true"),
    )
    return _truthy(value, True)


def _provider_api_key(provider: str, dotenv: dict[str, str]) -> str:
    upper = provider.strip().upper()
    for name in (
        f"DY3_LLM_{upper}_API_KEY",
        f"DY3_{upper}_API_KEY",
        f"{upper}_API_KEY",
    ):
        value = os.environ.get(name) or dotenv.get(name, "")
        if value:
            return value.strip()
    return ""


def _role_env_name(role: str, name: str) -> str:
    safe_role = "".join(ch if ch.isalnum() else "_" for ch in role.upper())
    return f"{_ENV_PREFIX}{safe_role}_{name}"


def load_llm_config(role: str = "default") -> LLMConfig:
    """从运行时配置 / 环境变量 / .env 加载 LLM 配置.

    优先级: 运行时配置 (前端 API 配置页) > 环境变量 > .env 文件 > 默认值.
    """
    dotenv = _read_dotenv()

    def pick(name: str, default: str = "") -> str:
        # 运行时配置最高优先 (key 为小写)
        if _runtime_config and _runtime_config.get(name.lower()):
            return _runtime_config[name.lower()]
        if f"{_ENV_PREFIX}{name}" in os.environ:
            return os.environ[f"{_ENV_PREFIX}{name}"]
        return dotenv.get(f"{_ENV_PREFIX}{name}", default)

    default_provider = pick("PROVIDER", "deepseek").strip().lower()
    provider = default_provider
    api_key = pick("API_KEY", "").strip()
    base_url = pick("BASE_URL", "").strip()
    model = pick("MODEL", "").strip()

    normalized_role = str(role or "default").strip().lower()
    if normalized_role != "default" and multi_model_enabled():
        def role_pick(name: str) -> str:
            env_name = _role_env_name(normalized_role, name)
            return str(os.environ.get(env_name) or dotenv.get(env_name, "")).strip()

        explicit_provider = role_pick("PROVIDER").lower()
        preferred_provider = _ROLE_PROVIDER_PREFERENCE.get(normalized_role, "")
        preferred_key = _provider_api_key(preferred_provider, dotenv)
        if explicit_provider:
            provider = explicit_provider
        elif preferred_provider and preferred_key:
            provider = preferred_provider

        role_key = role_pick("API_KEY")
        provider_key = _provider_api_key(provider, dotenv)
        if role_key:
            api_key = role_key
        elif provider_key:
            api_key = provider_key
        elif provider != default_provider:
            # Never reuse one provider's credential against another provider.
            api_key = ""

        role_base_url = role_pick("BASE_URL")
        if role_base_url:
            base_url = role_base_url
        elif provider != default_provider:
            base_url = ""

        role_model = role_pick("MODEL")
        model = (
            role_model
            or _ROLE_MODEL_DEFAULTS.get(provider, {}).get(normalized_role, "")
            or (model if provider == default_provider else "")
        )

    # ollama 本地模型无需 api_key; 其他 provider 需 key 才启用
    enabled = bool(api_key) or provider == "ollama"

    config = LLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=float(pick("TEMPERATURE", "0.3")),
        max_tokens=int(pick("MAX_TOKENS", "2048")),
        timeout_seconds=float(pick("TIMEOUT", "12")),
        enabled=enabled,
    )
    return config


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    disable_thinking: bool = False,
    role: str = "default",
    config: LLMConfig | None = None,
    reasoning_effort: str = "",
) -> str:
    """Call one role-selected model and return only visible answer text.

    DeepSeek uses Chat Completions, Claude uses Messages, and GPT-5 uses
    Responses. Reasoning/CoT is never returned as the user-visible answer.
    Network, authentication and parse failures return an empty string so the
    existing evidence-bounded fallback remains authoritative.
    """
    import httpx

    cfg = config or load_llm_config(role)
    if not cfg.is_ready():
        _record_call_status(
            role,
            cfg,
            success=False,
            failure_kind="missing_configuration",
        )
        return ""
    base_url = cfg.resolve_base_url().rstrip("/")
    model = cfg.resolve_model()
    provider = cfg.provider.strip().lower()

    if provider == "anthropic":
        url = base_url + "/messages"
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        anthropic_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
            if m.get("role") != "system"
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
        }
    elif provider == "openai" and model.startswith("gpt-5"):
        url = base_url + "/responses"
        payload = {
            "model": model,
            "input": list(messages),
            "max_output_tokens": max_tokens,
            "reasoning": {
                "effort": "none" if disable_thinking else (reasoning_effort or "medium")
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        }
    else:
        url = base_url + "/chat/completions"
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if disable_thinking and provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        elif provider == "deepseek" and reasoning_effort:
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = reasoning_effort
        if provider == "ollama":
            payload["options"] = {"num_gpu": 0}
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
    try:
        with httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=cfg.timeout_seconds, write=10.0, pool=5.0)
        ) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code in {401, 403}:
            failure_kind = "authentication"
        elif status_code == 404:
            failure_kind = "model_or_endpoint"
        elif status_code == 429:
            failure_kind = "quota_or_rate_limit"
        elif status_code in {400, 409, 422}:
            failure_kind = "request_validation"
        else:
            failure_kind = "provider_http_error"
        _record_call_status(
            role,
            cfg,
            success=False,
            failure_kind=failure_kind,
            http_status=status_code,
        )
        return ""
    except httpx.TimeoutException:
        _record_call_status(role, cfg, success=False, failure_kind="timeout")
        return ""
    except httpx.RequestError:
        _record_call_status(role, cfg, success=False, failure_kind="network")
        return ""
    except (ValueError, TypeError, KeyError):
        _record_call_status(role, cfg, success=False, failure_kind="invalid_response")
        return ""
    if provider == "anthropic":
        answer = "\n".join(
            str(item.get("text") or "").strip()
            for item in data.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    elif provider == "openai" and model.startswith("gpt-5"):
        if str(data.get("output_text") or "").strip():
            answer = str(data["output_text"]).strip()
        else:
            parts: list[str] = []
            for item in data.get("output", []) or []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for block in item.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        parts.append(str(block.get("text") or ""))
            answer = "\n".join(parts).strip()
    else:
        msg = data.get("choices", [{}])[0].get("message", {}) or {}
        answer = str(msg.get("content") or "").strip()
    _record_call_status(
        role,
        cfg,
        success=bool(answer),
        failure_kind="" if answer else "empty_visible_response",
        http_status=int(getattr(resp, "status_code", 200)),
    )
    return answer


def model_route_summary() -> dict[str, dict[str, Any]]:
    """Return non-secret runtime routing facts for diagnostics and tests."""

    roles = (
        "semantic_fast",
        "generation_fast",
        "generation_long",
        "generation_deep",
        "review",
    )
    summary: dict[str, dict[str, Any]] = {}
    for role in roles:
        cfg = load_llm_config(role)
        summary[role] = {
            "provider": cfg.provider,
            "model": cfg.resolve_model(),
            "ready": cfg.is_ready(),
            "readiness_basis": "configuration_only",
            "last_call": last_model_call_status(role),
        }
    return summary


def available_providers() -> dict[str, dict[str, str]]:
    """返回可用的 provider 预设 (供设置页/文档展示)."""
    return {k: dict(v) for k, v in _PROVIDER_PRESETS.items()}
