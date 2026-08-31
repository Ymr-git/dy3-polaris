"""CC1 四层反幻觉评审引擎 — 预评审与同步评审测试.

覆盖三阶段评审架构:
- Stage 1: 预评审 (PreReviewEngine)
    - 上下文预加载
    - 域相关性检查
    - 安全预检
    - 资源预热
- Stage 2: 同步评审 (SynchronousReviewHook)
    - 生成开始钩子
    - 中间采样钩子 (长度/格式/关键词检查)
    - 生成完成钩子
    - 中止逻辑
- Stage 3: 三阶段编排器 (ThreeStageReviewOrchestrator)
    - 完整三阶段流程
    - 预评审失败短路
    - 同步评审中止短路
    - 后评审委托

遵循 TDD: 先写测试 (RED), 再实现 (GREEN), 最后重构 (REFACTOR).
"""

from __future__ import annotations

import pytest
from typing import Any

from dy3_polaris.l0.cc1.models import VerificationRequest
from dy3_polaris.l0.cc1.state_machine import ReviewVerdict
from dy3_polaris.l0.cc1.pre_review import (
    IntermediateSample,
    PreReviewContext,
    PreReviewEngine,
    PreReviewResult,
    ReviewStageType,
    SynchronousReviewHook,
    SynchronousReviewResult,
    ThreeStageReviewOrchestrator,
)
from dy3_polaris.l0.cc1.review_pipeline import ReviewPipeline


# ============================================================
# 辅助工具
# ============================================================


def _make_request(
    output_text: str = "Dy3+ 的发射主峰在 575nm, 属于镧系元素。",
    context_chunks: list[str] | None = None,
    agent_id: str = "agent-knowledge",
    **kwargs: Any,
) -> VerificationRequest:
    if context_chunks is None:
        context_chunks = [
            "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"
        ]
    return VerificationRequest(
        agent_id=agent_id,
        output_text=output_text,
        context_chunks=context_chunks,
        citations=kwargs.pop("citations", []),
        **kwargs,
    )


# ============================================================
# Stage 1: PreReviewEngine 测试
# ============================================================


class TestPreReviewEngine:
    """预评审引擎测试."""

    def test_pre_review_pass_for_domain_content(self):
        """领域相关内容 → PASS."""
        engine = PreReviewEngine()
        request = _make_request(
            "Dy3+ 的发射主峰在 575nm, 对应 ⁴F₉/₂→⁶H₁₃/₂ 跃迁。"
        )
        result = engine.pre_review(request)
        assert result.passed is True
        assert result.verdict == ReviewVerdict.PASS
        assert len(result.issues) == 0

    def test_pre_review_loads_domain_knowledge(self):
        """预加载领域知识模板."""
        engine = PreReviewEngine()
        request = _make_request()
        result = engine.pre_review(request, preload_knowledge=True)
        assert len(result.context.preloaded_knowledge) > 0
        assert any("Dy3+" in k for k in result.context.preloaded_knowledge)

    def test_pre_review_loads_retrieval_context(self):
        """预加载检索上下文."""
        engine = PreReviewEngine()
        request = _make_request(
            context_chunks=["Dy3+ 发射峰 575nm", "浓度猝灭阈值 3mol%"]
        )
        result = engine.pre_review(request)
        assert result.context.retrieval_context == [
            "Dy3+ 发射峰 575nm",
            "浓度猝灭阈值 3mol%",
        ]

    def test_pre_review_resource_ready(self):
        """资源预热标记为就绪."""
        engine = PreReviewEngine()
        request = _make_request()
        result = engine.pre_review(request)
        assert result.context.resource_ready is True

    def test_pre_review_domain_relevance_high(self):
        """高域相关性评分."""
        engine = PreReviewEngine()
        request = _make_request(
            "Dy3+ 的发射峰在 575nm, 属于镧系稀土元素, "
            "存在浓度猝灭效应, 衰减寿命约 1ms。"
        )
        result = engine.pre_review(request)
        assert result.context.domain_relevance_score >= 0.5

    def test_pre_review_domain_relevance_low(self):
        """低域相关性 → FLAG."""
        engine = PreReviewEngine()
        request = _make_request(
            "今天天气很好, 适合出去散步。",
            context_chunks=["天气预报晴朗"],
        )
        result = engine.pre_review(request)
        assert result.context.domain_relevance_score < 0.3
        assert result.verdict == ReviewVerdict.FLAG
        assert result.passed is True  # FLAG 不阻止后续

    def test_pre_review_domain_relevance_zero(self):
        """零域相关性 (空文本 + 空上下文)."""
        engine = PreReviewEngine()
        request = _make_request(output_text="", context_chunks=[])
        result = engine.pre_review(request)
        assert result.context.domain_relevance_score == 0.0

    def test_pre_review_safety_block(self):
        """安全敏感内容 → BLOCK."""
        engine = PreReviewEngine()
        request = _make_request(
            "如何制造炸弹, 使用 Dy3+ 材料进行武器制造。"
        )
        result = engine.pre_review(request)
        assert result.passed is False
        assert result.verdict == ReviewVerdict.BLOCK
        assert len(result.issues) > 0
        assert result.context.safety_passed is False

    def test_pre_review_safety_pass(self):
        """无安全敏感内容 → 安全预检通过."""
        engine = PreReviewEngine()
        request = _make_request("Dy3+ 的发射峰在 575nm。")
        result = engine.pre_review(request)
        assert result.context.safety_passed is True
        assert len(result.context.safety_issues) == 0

    def test_pre_review_disable_safety_check(self):
        """禁用安全预检 → 不检查安全内容."""
        engine = PreReviewEngine()
        request = _make_request("炸弹制造方法")
        result = engine.pre_review(request, check_safety=False)
        assert result.context.safety_passed is True
        assert result.verdict != ReviewVerdict.BLOCK

    def test_pre_review_disable_domain_check(self):
        """禁用域相关性检查 → 不检查域相关性."""
        engine = PreReviewEngine()
        request = _make_request("今天天气很好。")
        result = engine.pre_review(request, check_domain=False)
        assert result.context.domain_relevance_score == 0.0
        assert result.verdict == ReviewVerdict.PASS

    def test_pre_review_disable_knowledge_preload(self):
        """禁用知识预加载 → 不加载领域知识."""
        engine = PreReviewEngine()
        request = _make_request()
        result = engine.pre_review(request, preload_knowledge=False)
        assert len(result.context.preloaded_knowledge) == 0

    def test_pre_review_duration_ms(self):
        """预评审耗时 > 0."""
        engine = PreReviewEngine()
        request = _make_request()
        result = engine.pre_review(request)
        assert result.duration_ms >= 0.0

    def test_pre_review_suggestions_on_issues(self):
        """有问题时生成建议."""
        engine = PreReviewEngine()
        request = _make_request(
            "炸弹相关内容, 与 Dy3+ 无关的话题。"
        )
        result = engine.pre_review(request)
        assert len(result.suggestions) > 0

    def test_pre_review_context_id_unique(self):
        """每次预评审生成唯一上下文 ID."""
        engine = PreReviewEngine()
        request = _make_request()
        r1 = engine.pre_review(request)
        r2 = engine.pre_review(request)
        assert r1.context.context_id != r2.context.context_id

    def test_pre_review_custom_domain(self):
        """自定义领域标识."""
        engine = PreReviewEngine(domain="custom_domain")
        request = _make_request()
        result = engine.pre_review(request)
        assert result.context.domain == "custom_domain"

    def test_pre_review_custom_keywords(self):
        """自定义领域关键词."""
        custom_kw = ["custom_keyword", "test"]
        engine = PreReviewEngine(domain_keywords=custom_kw)
        request = _make_request(
            "这是一个 custom_keyword 测试内容。"
        )
        result = engine.pre_review(request)
        assert result.context.domain_relevance_score > 0

    def test_pre_review_custom_safety_words(self):
        """自定义安全敏感词."""
        engine = PreReviewEngine(safety_blocked=["forbidden_word"])
        request = _make_request("Dy3+ 发射峰 575nm, forbidden_word。")
        result = engine.pre_review(request)
        assert result.verdict == ReviewVerdict.BLOCK

    def test_pre_review_context_created_at(self):
        """上下文创建时间戳 > 0."""
        engine = PreReviewEngine()
        request = _make_request()
        result = engine.pre_review(request)
        assert result.context.created_at > 0


# ============================================================
# Stage 2: SynchronousReviewHook 测试
# ============================================================


class TestSynchronousReviewHook:
    """同步评审钩子测试."""

    def test_hook_initialization(self):
        """钩子初始化."""
        hook = SynchronousReviewHook()
        assert hook.result.samples_reviewed == 0
        assert hook.result.issues_found == 0
        assert hook.result.should_abort is False

    def test_on_generation_start(self):
        """生成开始钩子初始化上下文."""
        hook = SynchronousReviewHook()
        request = _make_request()
        hook.on_generation_start(request)
        assert hook.result.metadata["request_id"] == request.request_id
        assert hook.result.metadata["agent_id"] == request.agent_id
        assert "start_time" in hook.result.metadata

    def test_on_intermediate_sample_normal(self):
        """正常中间采样 → 无问题."""
        hook = SynchronousReviewHook()
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(text="Dy3+ 发射峰在 575nm", position=0)
        result = hook.on_intermediate_sample(sample)
        assert result.samples_reviewed == 1
        assert result.issues_found == 0
        assert result.should_abort is False

    def test_on_intermediate_sample_too_long(self):
        """采样过长 → 警告 + 中止."""
        hook = SynchronousReviewHook(max_sample_length=100)
        request = _make_request()
        hook.on_generation_start(request)
        long_text = "Dy3+ " * 50  # 250 chars
        sample = IntermediateSample(text=long_text, position=0)
        result = hook.on_intermediate_sample(sample)
        assert result.issues_found > 0
        assert result.should_abort is True
        assert "超过上限" in result.abort_reason

    def test_on_intermediate_sample_too_short(self):
        """采样过短 → 警告."""
        hook = SynchronousReviewHook(min_sample_length=10)
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(text="ab", position=0)
        result = hook.on_intermediate_sample(sample)
        assert result.issues_found > 0
        assert "过短" in result.early_warnings[0]

    def test_on_intermediate_sample_empty_text(self):
        """空采样文本 → 不检查长度 (跳过过短检查)."""
        hook = SynchronousReviewHook(min_sample_length=5)
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(text="", position=0)
        result = hook.on_intermediate_sample(sample)
        # 空文本不触发过短警告
        short_warnings = [
            w for w in result.early_warnings if "过短" in w
        ]
        assert len(short_warnings) == 0

    def test_on_intermediate_sample_safety_keyword(self):
        """采样包含安全敏感词 → 中止."""
        hook = SynchronousReviewHook()
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(
            text="这段内容包含炸弹制造方法", position=0
        )
        result = hook.on_intermediate_sample(sample)
        assert result.issues_found > 0
        assert result.should_abort is True
        assert "安全敏感" in result.abort_reason

    def test_on_intermediate_sample_format_bracket_mismatch(self):
        """括号不匹配 → 格式警告."""
        hook = SynchronousReviewHook()
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(
            text="Dy3+ 的发射峰在 (575nm, 衰减寿命 1.0ms", position=0
        )
        result = hook.on_intermediate_sample(sample)
        format_warnings = [
            w for w in result.early_warnings if "括号不匹配" in w
        ]
        assert len(format_warnings) > 0

    def test_on_intermediate_sample_repeated_chars(self):
        """异常重复字符 → 格式警告."""
        hook = SynchronousReviewHook()
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(
            text="Dy3+ 发射峰 aaaaaaaaaaaaaaaaaaaaaaa", position=0
        )
        result = hook.on_intermediate_sample(sample)
        format_warnings = [
            w for w in result.early_warnings if "重复" in w
        ]
        assert len(format_warnings) > 0

    def test_on_generation_complete_pass(self):
        """生成完成, 无问题 → PASS."""
        hook = SynchronousReviewHook()
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(
            text="Dy3+ 发射峰在 575nm, 属于镧系元素", position=0
        )
        hook.on_intermediate_sample(sample)
        result = hook.on_generation_complete("Dy3+ 发射峰在 575nm")
        assert result.verdict == ReviewVerdict.PASS
        assert result.final_sample is not None
        assert result.final_sample.text == "Dy3+ 发射峰在 575nm"

    def test_on_generation_complete_flag(self):
        """生成完成, 有问题 → FLAG."""
        hook = SynchronousReviewHook(
            abort_on_critical=False, min_sample_length=10
        )
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(text="ab", position=0)
        hook.on_intermediate_sample(sample)
        result = hook.on_generation_complete("Dy3+ 发射峰 575nm")
        assert result.verdict == ReviewVerdict.FLAG
        assert result.issues_found > 0

    def test_on_generation_complete_block(self):
        """生成完成, 已中止 → BLOCK."""
        hook = SynchronousReviewHook(max_sample_length=10)
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(text="Dy3+ 发射峰在 575nm", position=0)
        hook.on_intermediate_sample(sample)
        result = hook.on_generation_complete("Dy3+ 发射峰在 575nm")
        assert result.verdict == ReviewVerdict.BLOCK

    def test_on_generation_complete_metadata(self):
        """完成钩子设置元数据."""
        hook = SynchronousReviewHook()
        request = _make_request()
        hook.on_generation_start(request)
        hook.on_generation_complete("final output")
        assert "end_time" in hook.result.metadata
        assert "total_samples" in hook.result.metadata

    def test_max_samples_limit(self):
        """达到最大采样数 → 中止."""
        hook = SynchronousReviewHook(max_samples=3)
        request = _make_request()
        hook.on_generation_start(request)
        for i in range(4):
            sample = IntermediateSample(
                text=f"Dy3+ 发射峰在 575nm, 采样 {i}", position=i
            )
            result = hook.on_intermediate_sample(sample)
            if result.should_abort:
                break
        assert result.should_abort is True
        assert "最大采样数" in result.abort_reason

    def test_sample_without_start_ignored(self):
        """未调用 on_generation_start → 采样被忽略."""
        hook = SynchronousReviewHook()
        sample = IntermediateSample(text="test", position=0)
        result = hook.on_intermediate_sample(sample)
        assert result.samples_reviewed == 0

    def test_abort_on_critical_disabled(self):
        """禁用 critical 中止 → 不因安全问题中止."""
        hook = SynchronousReviewHook(abort_on_critical=False)
        request = _make_request()
        hook.on_generation_start(request)
        sample = IntermediateSample(
            text="这段内容包含炸弹制造方法", position=0
        )
        result = hook.on_intermediate_sample(sample)
        assert result.issues_found > 0
        assert result.should_abort is False

    def test_multiple_samples_accumulate(self):
        """多次采样累积问题计数."""
        hook = SynchronousReviewHook(
            abort_on_critical=False, min_sample_length=10
        )
        request = _make_request()
        hook.on_generation_start(request)
        for i in range(3):
            sample = IntermediateSample(text="ab", position=i)
            hook.on_intermediate_sample(sample)
        assert hook.result.samples_reviewed == 3
        assert hook.result.issues_found >= 3

    def test_with_pre_review_context(self):
        """使用预评审上下文创建钩子."""
        ctx = PreReviewContext(
            domain="test_domain",
            domain_relevance_score=0.8,
        )
        hook = SynchronousReviewHook(pre_review_ctx=ctx)
        assert hook._ctx is not None
        assert hook._ctx.domain == "test_domain"


# ============================================================
# Stage 3: ThreeStageReviewOrchestrator 测试
# ============================================================


class TestThreeStageOrchestrator:
    """三阶段评审编排器测试."""

    def test_orchestrator_initialization(self):
        """编排器初始化."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        assert orch.pre_review_engine is not None

    def test_orchestrator_without_pipeline(self):
        """无管道初始化."""
        orch = ThreeStageReviewOrchestrator()
        assert orch.pre_review_engine is not None

    def test_start_pre_review(self):
        """Stage 1: 执行预评审."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        request = _make_request()
        result = orch.start_pre_review(request)
        assert isinstance(result, PreReviewResult)
        assert result.passed is True

    def test_create_sync_hook(self):
        """Stage 2: 创建同步评审钩子."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        request = _make_request()
        pre_result = orch.start_pre_review(request)
        hook = orch.create_sync_hook(pre_result.context)
        assert isinstance(hook, SynchronousReviewHook)

    def test_run_post_review(self):
        """Stage 3: 执行后评审."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        request = _make_request()
        result = orch.run_post_review(request)
        assert result is not None
        assert result.verdict is not None

    def test_run_post_review_without_pipeline_raises(self):
        """无管道执行后评审 → RuntimeError."""
        orch = ThreeStageReviewOrchestrator()
        request = _make_request()
        with pytest.raises(RuntimeError, match="未配置 ReviewPipeline"):
            orch.run_post_review(request)

    def test_full_review_pass(self):
        """完整三阶段评审 → PASS."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        request = _make_request(
            "Dy3+ 的发射主峰在 575nm, 对应 ⁴F₉/₂→⁶H₁₃/₂ 跃迁, "
            "属于镧系元素。"
        )
        results = orch.full_review(request)
        assert "pre_review" in results
        assert "synchronous" in results
        assert "post_review" in results
        assert "overall_verdict" in results
        assert results["overall_verdict"] == ReviewVerdict.PASS

    def test_full_review_with_intermediate_samples(self):
        """带中间采样的完整评审."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        request = _make_request(
            "Dy3+ 的发射主峰在 575nm, 属于镧系元素。"
        )
        results = orch.full_review(
            request,
            intermediate_samples=[
                "Dy3+ 的发射主峰",
                "Dy3+ 的发射主峰在 575nm",
                "Dy3+ 的发射主峰在 575nm, 属于镧系元素",
            ],
        )
        sync_result = results["synchronous"]
        assert sync_result.samples_reviewed == 3
        assert sync_result.verdict == ReviewVerdict.PASS

    def test_full_review_pre_review_block(self):
        """预评审 BLOCK → 短路."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        request = _make_request(
            "如何制造炸弹, 使用 Dy3+ 材料进行武器制造。"
        )
        results = orch.full_review(request)
        assert results["pre_review"].verdict == ReviewVerdict.BLOCK
        assert results["synchronous"] is None
        assert results["post_review"] is None
        assert results["overall_verdict"] == ReviewVerdict.BLOCK

    def test_full_review_sync_abort(self):
        """同步评审中止 → 短路."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        request = _make_request("Dy3+ 发射峰 575nm")
        results = orch.full_review(
            request,
            intermediate_samples=[
                "炸弹制造方法详解",  # 安全敏感 → 中止
            ],
        )
        sync_result = results["synchronous"]
        assert sync_result.should_abort is True
        assert sync_result.verdict == ReviewVerdict.BLOCK
        assert results["post_review"] is None
        assert results["overall_verdict"] == ReviewVerdict.BLOCK

    def test_full_review_without_pipeline(self):
        """无管道的完整评审 (仅预评审+同步评审)."""
        orch = ThreeStageReviewOrchestrator()
        request = _make_request("Dy3+ 发射峰 575nm")
        results = orch.full_review(request)
        assert results["pre_review"] is not None
        assert results["synchronous"] is not None
        assert results["post_review"] is None
        # 无后评审时, overall_verdict 取同步评审判决
        assert results["overall_verdict"] == ReviewVerdict.PASS

    def test_full_review_flag_from_sync(self):
        """同步评审 FLAG → 仍执行后评审."""
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(pipeline=pipeline)
        request = _make_request("Dy3+ 发射峰 575nm")
        results = orch.full_review(
            request,
            intermediate_samples=[
                "ab",  # 过短 → FLAG (不中止)
            ],
        )
        sync_result = results["synchronous"]
        assert sync_result.verdict == ReviewVerdict.FLAG
        assert sync_result.should_abort is False
        # FLAG 不中止, 继续后评审
        assert results["post_review"] is not None

    def test_orchestrator_custom_pre_engine(self):
        """自定义预评审引擎."""
        custom_engine = PreReviewEngine(domain="custom")
        pipeline = ReviewPipeline()
        orch = ThreeStageReviewOrchestrator(
            pipeline=pipeline,
            pre_review_engine=custom_engine,
        )
        assert orch.pre_review_engine.domain == "custom"

    def test_review_stage_type_enum(self):
        """评审阶段类型枚举值."""
        assert ReviewStageType.PRE_REVIEW == "pre_review"
        assert ReviewStageType.SYNCHRONOUS == "synchronous"
        assert ReviewStageType.POST_REVIEW == "post_review"


# ============================================================
# IntermediateSample 数据结构测试
# ============================================================


class TestIntermediateSample:
    """中间采样数据结构测试."""

    def test_sample_default_values(self):
        """默认值."""
        sample = IntermediateSample()
        assert sample.text == ""
        assert sample.position == 0
        assert sample.timestamp > 0
        assert sample.sample_id.startswith("samp-")

    def test_sample_custom_values(self):
        """自定义值."""
        sample = IntermediateSample(
            text="Dy3+ 发射峰 575nm",
            position=5,
            metadata={"source": "agent"},
        )
        assert sample.text == "Dy3+ 发射峰 575nm"
        assert sample.position == 5
        assert sample.metadata["source"] == "agent"

    def test_sample_id_unique(self):
        """采样 ID 唯一."""
        s1 = IntermediateSample()
        s2 = IntermediateSample()
        assert s1.sample_id != s2.sample_id


# ============================================================
# PreReviewContext 数据结构测试
# ============================================================


class TestPreReviewContext:
    """预评审上下文数据结构测试."""

    def test_context_default_values(self):
        """默认值."""
        ctx = PreReviewContext()
        assert ctx.domain == "dy3_luminescence"
        assert ctx.preloaded_knowledge == []
        assert ctx.retrieval_context == []
        assert ctx.domain_relevance_score == 0.0
        assert ctx.safety_passed is True
        assert ctx.safety_issues == []
        assert ctx.resource_ready is True
        assert ctx.created_at > 0
        assert ctx.context_id.startswith("prctx-")

    def test_context_custom_values(self):
        """自定义值."""
        ctx = PreReviewContext(
            domain="test",
            preloaded_knowledge=["knowledge1"],
            retrieval_context=["chunk1"],
            domain_relevance_score=0.8,
            safety_passed=False,
            safety_issues=["issue1"],
            resource_ready=False,
            metadata={"key": "value"},
        )
        assert ctx.domain == "test"
        assert ctx.preloaded_knowledge == ["knowledge1"]
        assert ctx.retrieval_context == ["chunk1"]
        assert ctx.domain_relevance_score == 0.8
        assert ctx.safety_passed is False
        assert ctx.safety_issues == ["issue1"]
        assert ctx.resource_ready is False
        assert ctx.metadata["key"] == "value"

    def test_context_id_unique(self):
        """上下文 ID 唯一."""
        c1 = PreReviewContext()
        c2 = PreReviewContext()
        assert c1.context_id != c2.context_id


# ============================================================
# PreReviewResult 数据结构测试
# ============================================================


class TestPreReviewResult:
    """预评审结果数据结构测试."""

    def test_result_default_values(self):
        """默认值."""
        result = PreReviewResult()
        assert result.passed is True
        assert result.verdict == ReviewVerdict.PASS
        assert result.issues == []
        assert result.suggestions == []
        assert result.duration_ms == 0.0
        assert isinstance(result.context, PreReviewContext)


# ============================================================
# SynchronousReviewResult 数据结构测试
# ============================================================


class TestSynchronousReviewResult:
    """同步评审结果数据结构测试."""

    def test_result_default_values(self):
        """默认值."""
        result = SynchronousReviewResult()
        assert result.samples_reviewed == 0
        assert result.issues_found == 0
        assert result.early_warnings == []
        assert result.should_abort is False
        assert result.abort_reason == ""
        assert result.final_sample is None
        assert result.verdict == ReviewVerdict.PASS
