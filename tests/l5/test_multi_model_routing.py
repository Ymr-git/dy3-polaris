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
    llm_config._runtime_provider_configs.clear()
    llm_config._runtime_role_routes.clear()
    monkeypatch.setattr(llm_config, "_read_dotenv", lambda: dict(dotenv))
    monkeypatch.setenv("DY3_MULTI_MODEL_ENABLED", "1")
    for name in (
        "DY3_LLM_PROVIDER",
        "DY3_LLM_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "QWEN_API_KEY",
        "ZHIPU_API_KEY",
        "KIMI_API_KEY",
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


def test_domestic_provider_keys_activate_role_specialization(monkeypatch) -> None:
    _clean_runtime(
        monkeypatch,
        {
            "DY3_LLM_PROVIDER": "deepseek",
            "DY3_LLM_API_KEY": "deepseek-key",
            "QWEN_API_KEY": "qwen-key",
            "ZHIPU_API_KEY": "zhipu-key",
            "KIMI_API_KEY": "kimi-key",
        },
    )

    long_cfg = load_llm_config("generation_long")
    deep_cfg = load_llm_config("generation_deep")
    review_cfg = load_llm_config("review")
    fast_cfg = load_llm_config("semantic_fast")
    assert (fast_cfg.provider, fast_cfg.resolve_model()) == ("qwen", "qwen-max")
    assert (long_cfg.provider, long_cfg.resolve_model()) == ("kimi", "kimi-k2.6")
    assert (deep_cfg.provider, deep_cfg.resolve_model()) == ("deepseek", "deepseek-v4-pro")
    assert (review_cfg.provider, review_cfg.resolve_model()) == ("zhipu", "glm-5.1")


def test_runtime_multi_provider_summary_never_exposes_keys(monkeypatch) -> None:
    _clean_runtime(monkeypatch, {})
    summary = llm_config.set_multi_runtime_config(
        [
            {"provider": "qwen", "api_key": "qwen-secret", "model": "qwen-max"},
            {"provider": "zhipu", "api_key": "zhipu-secret", "model": "glm-5.1"},
        ],
        role_routes={"semantic_fast": "qwen", "review": "zhipu"},
    )

    rendered = str(summary)
    assert "qwen-secret" not in rendered
    assert "zhipu-secret" not in rendered
    assert summary["providers"]["qwen"]["configured"] is True
    assert summary["providers"]["zhipu"]["configured"] is True
    assert load_llm_config("semantic_fast").provider == "qwen"
    assert load_llm_config("review").provider == "zhipu"


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


def test_kimi_k26_uses_provider_required_non_thinking_parameters(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    response = {"choices": [{"message": {"content": "Kimi answer"}}]}
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: _FakeClient(captured, response, **kwargs),
    )
    cfg = LLMConfig(
        provider="kimi",
        api_key="secret",
        model="kimi-k2.6",
        enabled=True,
    )

    answer = chat_completion(
        [{"role": "user", "content": "long teaching resource"}],
        config=cfg,
        temperature=0.3,
        disable_thinking=True,
    )

    assert answer == "Kimi answer"
    assert captured["payload"]["temperature"] == 0.6
    assert captured["payload"]["thinking"] == {"type": "disabled"}


def test_zhipu_review_explicitly_disables_hidden_thinking(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    response = {"choices": [{"message": {"content": '{"verdict":"pass"}'}}]}
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: _FakeClient(captured, response, **kwargs),
    )
    cfg = LLMConfig(
        provider="zhipu",
        api_key="secret",
        model="glm-5.1",
        enabled=True,
    )

    assert chat_completion(
        [{"role": "user", "content": "review"}],
        config=cfg,
        disable_thinking=True,
    )
    assert captured["payload"]["thinking"] == {"type": "disabled"}


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


def test_long_generation_retries_one_different_provider(monkeypatch) -> None:
    primary = LLMConfig(
        provider="kimi", api_key="kimi-key", model="kimi-k2.6", enabled=True
    )
    fallback = LLMConfig(
        provider="qwen", api_key="qwen-key", model="qwen-max", enabled=True
    )
    synthesizer = LLMSynthesizer(primary)
    synthesizer._config_override = False
    monkeypatch.setattr(synthesizer, "_config_for_role", lambda _role: primary)
    monkeypatch.setattr(
        synthesizer,
        "_fallback_config_for_role",
        lambda _role, _current: fallback,
    )
    seen: list[str] = []

    def _call(*_args, **kwargs):
        seen.append(kwargs["route_config"].provider)
        return "" if len(seen) == 1 else "fallback answer"

    monkeypatch.setattr(synthesizer, "_call_llm", _call)

    answer, used = synthesizer.synthesize(
        "query", ["evidence"], model_role="generation_long"
    )

    assert (answer, used) == ("fallback answer", True)
    assert seen == ["kimi", "qwen"]


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

    assert sorted(seen) == sorted([
        ("generation_fast", False, "none"),
        ("generation_long", False, "none"),
        ("generation_deep", True, "high"),
    ])
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


def test_faithful_but_incomplete_answer_is_withheld_for_revision(monkeypatch) -> None:
    monkeypatch.setenv("DY3_MULTI_MODEL_ENABLED", "1")
    monkeypatch.setattr(
        agent_workers,
        "critique_answer",
        lambda *_args, **_kwargs: {
            "used_llm": True,
            "verdict": "fix_completeness",
            "relevance": 1.0,
            "faithfulness": 1.0,
            "completeness": 0.6,
            "score": 0.92,
            "reason": "answer is faithful but could teach one more mechanism",
        },
    )

    review = agent_workers.run_review(
        {
            "task_id": "task-review-completeness-advisory",
            "query": "为什么Dy³⁺有黄蓝双发射？",
            "content": "4F9/2到6H15/2和6H13/2分别产生蓝、黄发射。",
            "context_chunks": [
                "Dy3+的4F9/2到6H15/2和6H13/2跃迁分别产生蓝、黄发射。"
            ],
        },
        AgentDependencies(),
    )

    assert review["verdict"] == "needs_review"
    assert "问题核心覆盖门要求修订" in review["reason"]


def test_guided_question_review_uses_independent_reviewer_not_answer_checks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DY3_MULTI_MODEL_ENABLED", "1")
    monkeypatch.setattr(
        agent_workers,
        "critique_answer",
        lambda *_args, **_kwargs: {
            "used_llm": True,
            "verdict": "pass",
            "relevance": 1.0,
            "faithfulness": 1.0,
            "completeness": 0.8,
            "score": 0.96,
            "reason": "questions stay within the evidence boundary",
        },
    )

    class _MustNotRun:
        def check(self, *_args, **_kwargs):
            raise AssertionError("answer fact checker must not classify questions as claims")

        def verify(self, *_args, **_kwargs):
            raise AssertionError("answer CC1 must not classify questions as claims")

    review = agent_workers.run_review(
        {
            "task_id": "task-guided-review",
            "query": "审核问题列表是否受证据支持；原始问题：为什么Dy³⁺有黄蓝双发射？",
            "content": "1. 两条跃迁分别对应什么发射颜色？\n2. 基质变化可能影响哪些光谱特征？",
            "context_chunks": [
                "Dy3+的4F9/2到6H15/2和6H13/2跃迁分别产生蓝、黄发射。"
            ],
            "guided_question_review": True,
        },
        AgentDependencies(
            fact_checker=_MustNotRun(),
            anti_hallucination_pipeline=_MustNotRun(),
        ),
    )

    assert review["verdict"] == "approved"
    assert "独立 Reviewer" in review["reason"]


def test_guided_question_review_does_not_release_without_model_reviewer(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DY3_MULTI_MODEL_ENABLED", "1")
    monkeypatch.setattr(
        agent_workers,
        "critique_answer",
        lambda *_args, **_kwargs: {
            "used_llm": False,
            "verdict": "pass",
            "relevance": 1.0,
            "faithfulness": 1.0,
            "completeness": 1.0,
            "score": 1.0,
            "reason": "heuristic fallback only",
        },
    )

    review = agent_workers.run_review(
        {
            "task_id": "task-guided-no-model",
            "query": "审核问题列表；原始问题：为什么Dy³⁺有黄蓝双发射？",
            "content": "1. 两条跃迁分别对应什么发射颜色？",
            "context_chunks": [
                "Dy3+的4F9/2到6H15/2和6H13/2跃迁分别产生蓝、黄发射。"
            ],
            "guided_question_review": True,
        },
        AgentDependencies(),
    )

    assert review["verdict"] == "needs_review"
    assert "未完成独立 Reviewer 模型审核" in review["reason"]
