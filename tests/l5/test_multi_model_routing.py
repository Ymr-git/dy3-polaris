"""Role-specialized multi-model routing without paid network calls."""

from __future__ import annotations

from typing import Any

import httpx

from dy3_polaris.l3 import llm_config
from dy3_polaris.l3.llm_config import (
    LLMConfig,
    chat_completion,
    last_model_call_status,
    load_llm_config,
)
from dy3_polaris.l3.llm_synthesizer import LLMSynthesizer
from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_workers import AgentDependencies


def _clean_runtime(monkeypatch, dotenv: dict[str, str]) -> None:
    llm_config._runtime_config.clear()
    monkeypatch.setattr(llm_config, "_read_dotenv", lambda: dict(dotenv))
    monkeypatch.setenv("DY3_MULTI_MODEL_ENABLED", "1")
    for name in (
        "DY3_LLM_PROVIDER",
        "DY3_LLM_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_one_deepseek_key_activates_flash_and_pro_roles(monkeypatch) -> None:
    _clean_runtime(
        monkeypatch,
        {"DY3_LLM_PROVIDER": "deepseek", "DY3_LLM_API_KEY": "test-key"},
    )

    assert load_llm_config("generation_fast").resolve_model() == "deepseek-v4-flash"
    assert load_llm_config("generation_long").resolve_model() == "deepseek-v4-pro"
    assert load_llm_config("generation_deep").resolve_model() == "deepseek-v4-pro"
    assert load_llm_config("review").resolve_model() == "deepseek-v4-pro"


def test_optional_provider_keys_activate_cross_provider_strengths(monkeypatch) -> None:
    _clean_runtime(
        monkeypatch,
        {
            "DY3_LLM_PROVIDER": "deepseek",
            "DY3_LLM_API_KEY": "deepseek-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "OPENAI_API_KEY": "openai-key",
        },
    )

    long_cfg = load_llm_config("generation_long")
    deep_cfg = load_llm_config("generation_deep")
    review_cfg = load_llm_config("review")
    assert (long_cfg.provider, long_cfg.resolve_model()) == ("anthropic", "claude-sonnet-5")
    assert (deep_cfg.provider, deep_cfg.resolve_model()) == ("openai", "gpt-5.6-terra")
    assert (review_cfg.provider, review_cfg.resolve_model()) == ("openai", "gpt-5.6-sol")


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, captured: dict[str, Any], response: dict[str, Any], **_: Any) -> None:
        self._captured = captured
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]):
        self._captured.update({"url": url, "payload": json, "headers": headers})
        return _FakeResponse(self._response)


def test_anthropic_route_uses_messages_protocol(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    response = {"content": [{"type": "text", "text": "Claude answer"}]}
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: _FakeClient(captured, response, **kwargs),
    )
    cfg = LLMConfig(
        provider="anthropic",
        api_key="secret",
        model="claude-sonnet-5",
        enabled=True,
    )

    answer = chat_completion(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        config=cfg,
    )

    assert answer == "Claude answer"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["payload"]["system"] == "system"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "question"}]
    assert captured["headers"]["x-api-key"] == "secret"


def test_openai_gpt5_route_uses_responses_protocol(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    response = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "review"}]}
        ]
    }
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: _FakeClient(captured, response, **kwargs),
    )
    cfg = LLMConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.6-sol",
        enabled=True,
    )

    answer = chat_completion(
        [{"role": "user", "content": "audit"}],
        config=cfg,
        reasoning_effort="high",
    )

    assert answer == "review"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert "temperature" not in captured["payload"]


def test_deepseek_never_returns_private_reasoning_as_answer(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    response = {"choices": [{"message": {"content": "", "reasoning_content": "private"}}]}
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: _FakeClient(captured, response, **kwargs),
    )
    cfg = LLMConfig(
        provider="deepseek",
        api_key="secret",
        model="deepseek-v4-pro",
        enabled=True,
    )

    answer = chat_completion(
        [{"role": "user", "content": "reason"}],
        config=cfg,
        reasoning_effort="high",
    )

    assert answer == ""
    assert captured["payload"]["thinking"] == {"type": "enabled"}
    assert captured["payload"]["reasoning_effort"] == "high"


def test_authentication_failure_is_visible_without_secret_or_prompt(monkeypatch) -> None:
    class _UnauthorizedClient:
        def __init__(self, **_: Any) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]):
            request = httpx.Request("POST", url)
            return httpx.Response(401, request=request, json={"error": "invalid key"})

    monkeypatch.setattr(httpx, "Client", _UnauthorizedClient)
    cfg = LLMConfig(
        provider="deepseek",
        api_key="must-never-leak",
        model="deepseek-v4-flash",
        enabled=True,
    )

    assert chat_completion(
        [{"role": "user", "content": "private learner prompt"}],
        config=cfg,
        role="generation_fast",
    ) == ""
    status = last_model_call_status("generation_fast")
    assert status["failure_kind"] == "authentication"
    assert status["http_status"] == 401
    assert "must-never-leak" not in str(status)
    assert "private learner prompt" not in str(status)


def test_synthesizer_forwards_generation_role(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _chat(messages, **kwargs):
        captured.update(kwargs)
        return "evidence-bounded answer"

    monkeypatch.setattr(llm_config, "chat_completion", _chat)
    cfg = LLMConfig(
        provider="deepseek",
        api_key="secret",
        model="deepseek-v4-pro",
        enabled=True,
    )
    answer, used = LLMSynthesizer(cfg).synthesize(
        "query",
        ["evidence"],
        model_role="generation_deep",
        reasoning_effort="high",
        enable_thinking=True,
    )

    assert (answer, used) == ("evidence-bounded answer", True)
    assert captured["role"] == "generation_deep"
    assert captured["reasoning_effort"] == "high"


def test_multi_candidate_generation_dispatches_three_model_profiles(monkeypatch) -> None:
    seen: list[tuple[str, bool, str]] = []

    def _generation(payload, _deps):
        seen.append(
            (
                payload["_llm_role"],
                payload["_llm_enable_thinking"],
                payload["_llm_reasoning_effort"],
            )
        )
        return {
            "status": "completed",
            "answer": "Dy³⁺ evidence answer",
            "confidence": 0.8,
            "context_chunks": ["evidence"],
            "citations": ["source"],
            "sources": ["source"],
        }

    monkeypatch.setattr(agent_workers, "run_generation", _generation)
    result = agent_workers._run_multi_candidate_generation(
        {"query": "为什么Dy³⁺有黄蓝双发射？", "task_id": "task-models"},
        AgentDependencies(),
    )

    assert seen == [
        ("generation_fast", False, "none"),
        ("generation_long", False, "none"),
        ("generation_deep", True, "high"),
    ]
    assert result["answer"] == "Dy³⁺ evidence answer"
    assert all("_llm_role" not in candidate for candidate in result["candidates"])


def test_independent_model_challenge_can_block_but_not_approve(monkeypatch) -> None:
    monkeypatch.setenv("DY3_MULTI_MODEL_ENABLED", "1")
    monkeypatch.setattr(
        agent_workers,
        "critique_answer",
        lambda *_args, **_kwargs: {
            "used_llm": True,
            "verdict": "fix_faithfulness",
            "reason": "claim lacks direct evidence",
        },
    )

    review = agent_workers.run_review(
        {
            "task_id": "task-review-model",
            "query": "为什么Dy³⁺有黄蓝双发射？",
            "content": "candidate answer",
            "context_chunks": ["source evidence"],
        },
        AgentDependencies(),
    )

    assert review["verdict"] == "needs_review"
    assert "独立模型交叉审核提出挑战" in review["reason"]
    assert "model" not in review
