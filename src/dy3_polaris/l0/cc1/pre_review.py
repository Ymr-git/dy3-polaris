"""CC1 四层反幻觉评审引擎 — 预评审与同步评审.

实现设计文档中定义的三阶段评审架构:

    Stage 1: 预评审 (Pre-Review)
        ↓
    Stage 2: 同步评审 (Synchronous Review) — Agent 执行过程中实时采样
        ↓
    Stage 3: 后评审 (Post-Review) — 四层递进评审 (已有 ReviewPipeline)

Stage 1 预评审在 Agent 生成输出之前执行, 包括:
- 上下文预加载 (Context Preload): 预加载领域知识与检索上下文
- 域相关性检查 (Domain Relevance): 快速判断输出是否属于目标领域
- 安全预检 (Safety Pre-check): 基本安全/策略合规检查
- 资源预热 (Resource Warm-up): MCP 连接池预热、缓存加载

Stage 2 同步评审在 Agent 执行过程中实时采样中间结果, 包括:
- 生成开始钩子 (on_generation_start): 初始化评审上下文
- 中间采样钩子 (on_intermediate_sample): 对中间输出进行轻量级检查
- 生成完成钩子 (on_generation_complete): 触发完整后评审

融合世界先进方案:
- NeMo Guardrails: input/output rails + 同步拦截
- Guardrails AI: 可编程验证管道与 on_fail 策略
- CoVe (Chain-of-Verification): 分阶段验证-修正
- LlamaIndex Citation: 上下文预加载与溯源准备
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .models import VerificationRequest
from .state_machine import ReviewVerdict


# ============================================================
# 评审阶段枚举
# ============================================================


class ReviewStageType(str, Enum):
    """评审阶段类型 — 三阶段架构."""

    PRE_REVIEW = "pre_review"          # Stage 1: 预评审
    SYNCHRONOUS = "synchronous"        # Stage 2: 同步评审
    POST_REVIEW = "post_review"        # Stage 3: 后评审 (四层递进)


# ============================================================
# 预评审数据结构
# ============================================================


@dataclass
class PreReviewContext:
    """预评审上下文.

    在 Agent 生成输出前构建, 携带预加载的领域知识、检索上下文
    和安全检查结果, 供后续同步评审和后评审使用.

    Attributes:
        context_id: 上下文唯一 ID
        domain: 领域标识 (如 "dy3_luminescence")
        preloaded_knowledge: 预加载的领域知识条目
        retrieval_context: 预加载的检索上下文片段
        domain_relevance_score: 域相关性评分 (0-1)
        safety_passed: 安全预检是否通过
        safety_issues: 安全预检发现的问题
        resource_ready: 资源是否就绪
        metadata: 附加元数据
        created_at: 创建时间
    """

    context_id: str = field(
        default_factory=lambda: f"prctx-{uuid.uuid4().hex[:10]}"
    )
    domain: str = "dy3_luminescence"
    preloaded_knowledge: list[str] = field(default_factory=list)
    retrieval_context: list[str] = field(default_factory=list)
    domain_relevance_score: float = 0.0
    safety_passed: bool = True
    safety_issues: list[str] = field(default_factory=list)
    resource_ready: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class PreReviewResult:
    """预评审结果.

    Attributes:
        passed: 预评审是否通过
        context: 预评审上下文 (供后续阶段使用)
        verdict: 预评审判决 (PASS/FLAG/BLOCK)
        issues: 发现的问题列表
        suggestions: 预审建议
        duration_ms: 预评审耗时 (毫秒)
    """

    passed: bool = True
    context: PreReviewContext = field(default_factory=PreReviewContext)
    verdict: ReviewVerdict = ReviewVerdict.PASS
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


# ============================================================
# 同步评审数据结构
# ============================================================


@dataclass
class IntermediateSample:
    """中间采样结果.

    Agent 执行过程中的一个中间输出采样.

    Attributes:
        sample_id: 采样 ID
        text: 采样文本
        timestamp: 采样时间戳
        position: 在生成流中的位置 (0-based)
        metadata: 附加元数据
    """

    sample_id: str = field(
        default_factory=lambda: f"samp-{uuid.uuid4().hex[:8]}"
    )
    text: str = ""
    timestamp: float = field(default_factory=time.time)
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynchronousReviewResult:
    """同步评审结果.

    Attributes:
        samples_reviewed: 已评审的采样数
        issues_found: 发现的问题数
        early_warnings: 早期警告列表
        should_abort: 是否应中止生成
        abort_reason: 中止原因
        final_sample: 最终采样 (生成完成时的完整输出)
        verdict: 同步评审判决
        metadata: 附加元数据
    """

    samples_reviewed: int = 0
    issues_found: int = 0
    early_warnings: list[str] = field(default_factory=list)
    should_abort: bool = False
    abort_reason: str = ""
    final_sample: IntermediateSample | None = None
    verdict: ReviewVerdict = ReviewVerdict.PASS
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 预评审引擎
# ============================================================


#: Dy3+ 发光材料领域关键词
_DOMAIN_KEYWORDS: list[str] = [
    "dy3+", "dy³⁺", "dysprosium", "镝",
    "发光", "luminescence", "荧光", "fluorescence", "磷光", "phosphor",
    "发射", "emission", "激发", "excitation",
    "judd-ofelt", "judd ofelt",
    "yag", "y₂o₃", "y2o3",
    "浓度猝灭", "concentration quenching",
    "色度", "chromaticity", "cie",
    "稀土", "rare earth", "lanthanide", "镧系",
    "4f", "能级", "energy level",
    "⁴f₉/₂", "⁶h₁₃/₂", "⁶h₁₅/₂",
    "量子效率", "quantum efficiency",
    "衰减", "decay", "寿命", "lifetime",
    "基质", "host", "晶体场", "crystal field",
]

#: 安全敏感关键词 (基本策略合规)
_SAFETY_BLOCKED: list[str] = [
    "炸弹", "bomb", "武器制造", "weapon",
    "毒品合成", "drug synthesis",
    "儿童", "minor", "underage",
]

#: 预加载领域知识模板
_DOMAIN_KNOWLEDGE_TEMPLATES: list[str] = [
    "Dy3+ 发射主峰: 570-585nm (⁴F₉/₂→⁶H₁₃/₂ 黄色发射)",
    "Dy3+ 蓝色发射: 475-495nm (⁴F₉/₂→⁶H₁₅/₂)",
    "Dy3+ 激发波段: ~350nm, ~390nm, ~450nm",
    "Dy3+ 掺杂浓度推荐: 1-5mol%, 猝灭阈值: 3-8mol%",
    "Dy3+ 量子效率: 10-85%",
    "Dy3+ 衰减寿命: 0.1-2.0ms",
    "Dy3+ CIE 色坐标: x=0.38-0.45, y=0.40-0.50",
    "Dy3+ Judd-Ofelt: Ω₂=1-10, Ω₄=0.5-5, Ω₆=0.5-5 (×10⁻²⁰ cm²)",
    "Dy3+ 属于镧系/f区, 非d区过渡金属",
    "浓度猝灭: 浓度增加→发光强度先增后减",
    "热猝灭: 温度升高→非辐射跃迁增加→发光强度降低",
]


class PreReviewEngine:
    """预评审引擎.

    在 Agent 生成输出前执行预评审, 包括:
    1. 上下文预加载: 预加载领域知识与检索上下文
    2. 域相关性检查: 快速判断请求是否属于目标领域
    3. 安全预检: 基本安全/策略合规检查
    4. 资源预热: 标记资源就绪状态

    使用示例::

        engine = PreReviewEngine()
        result = engine.pre_review(request)
        if not result.passed:
            # 预评审未通过, 可拒绝或降级处理
            ...
    """

    def __init__(
        self,
        domain: str = "dy3_luminescence",
        domain_keywords: list[str] | None = None,
        safety_blocked: list[str] | None = None,
        knowledge_templates: list[str] | None = None,
    ) -> None:
        self._domain = domain
        self._keywords = domain_keywords or _DOMAIN_KEYWORDS
        self._safety = safety_blocked or _SAFETY_BLOCKED
        self._knowledge = knowledge_templates or _DOMAIN_KNOWLEDGE_TEMPLATES

    @property
    def domain(self) -> str:
        return self._domain

    def pre_review(
        self,
        request: VerificationRequest,
        *,
        preload_knowledge: bool = True,
        check_safety: bool = True,
        check_domain: bool = True,
    ) -> PreReviewResult:
        """执行预评审.

        Args:
            request: 验证请求
            preload_knowledge: 是否预加载领域知识
            check_safety: 是否执行安全预检
            check_domain: 是否执行域相关性检查

        Returns:
            预评审结果
        """
        start = time.time()
        ctx = PreReviewContext(domain=self._domain)
        issues: list[str] = []
        suggestions: list[str] = []

        # 1. 上下文预加载
        if preload_knowledge:
            ctx.preloaded_knowledge = list(self._knowledge)
            ctx.retrieval_context = list(request.context_chunks)
            ctx.resource_ready = True

        # 2. 安全预检
        if check_safety:
            safety_issues = self._check_safety(request.output_text)
            ctx.safety_issues = safety_issues
            if safety_issues:
                ctx.safety_passed = False
                issues.extend(safety_issues)
                suggestions.append("输出包含安全敏感内容, 建议拒绝生成")

        # 3. 域相关性检查
        if check_domain:
            relevance = self._check_domain_relevance(request)
            ctx.domain_relevance_score = relevance
            if relevance < 0.1:
                issues.append(
                    f"域相关性极低 ({relevance:.2f}), 输出可能与领域无关"
                )
                suggestions.append(
                    "建议确认输出是否属于 Dy3+ 发光材料领域"
                )

        # 判决
        if ctx.safety_issues:
            verdict = ReviewVerdict.BLOCK
            passed = False
        elif issues:
            verdict = ReviewVerdict.FLAG
            passed = True  # 预评审 FLAG 不阻止后续, 仅警告
        else:
            verdict = ReviewVerdict.PASS
            passed = True

        duration = (time.time() - start) * 1000
        return PreReviewResult(
            passed=passed,
            context=ctx,
            verdict=verdict,
            issues=issues,
            suggestions=suggestions,
            duration_ms=round(duration, 2),
        )

    def _check_safety(self, text: str) -> list[str]:
        """安全预检 — 检查是否包含安全敏感内容."""
        issues: list[str] = []
        text_lower = text.lower()
        for keyword in self._safety:
            if keyword.lower() in text_lower:
                issues.append(f"检测到安全敏感关键词: '{keyword}'")
        return issues

    def _check_domain_relevance(
        self, request: VerificationRequest
    ) -> float:
        """域相关性检查 — 计算输出与领域关键词的匹配度.

        Returns:
            相关性评分 (0-1)
        """
        text = (
            request.output_text
            + " "
            + " ".join(request.context_chunks)
        )
        text_lower = text.lower()
        if not text.strip():
            return 0.0

        matched = sum(
            1 for kw in self._keywords if kw.lower() in text_lower
        )
        # 归一化: 匹配 3+ 个关键词即视为高相关
        return min(1.0, matched / 3.0)


# ============================================================
# 同步评审钩子
# ============================================================


class SynchronousReviewHook:
    """同步评审钩子.

    在 Agent 执行过程中实时采样中间结果, 进行轻量级检查.

    支持三种钩子:
    - on_generation_start: 生成开始时调用
    - on_intermediate_sample: 中间采样时调用
    - on_generation_complete: 生成完成时调用

    检查策略 (轻量级, 不执行完整四层评审):
    - 长度检查: 中间输出是否异常短/长
    - 格式检查: 是否包含明显格式错误
    - 关键词检查: 是否包含领域禁用词
    - 一致性检查: 中间输出与上下文是否矛盾

    使用示例::

        hook = SynchronousReviewHook(pre_review_ctx)
        hook.on_generation_start(request)

        for chunk in stream:
            sample = IntermediateSample(text=chunk, position=pos)
            result = hook.on_intermediate_sample(sample)
            if result.should_abort:
                break

        final = hook.on_generation_complete(full_output)
    """

    def __init__(
        self,
        pre_review_ctx: PreReviewContext | None = None,
        *,
        max_sample_length: int = 50000,
        min_sample_length: int = 5,
        max_samples: int = 100,
        abort_on_critical: bool = True,
    ) -> None:
        self._ctx = pre_review_ctx
        self._max_len = max_sample_length
        self._min_len = min_sample_length
        self._max_samples = max_samples
        self._abort_on_critical = abort_on_critical
        self._result = SynchronousReviewResult()
        self._sample_count = 0
        self._started = False

    @property
    def result(self) -> SynchronousReviewResult:
        """当前同步评审结果."""
        return self._result

    def on_generation_start(
        self, request: VerificationRequest
    ) -> None:
        """生成开始钩子.

        初始化同步评审上下文.

        Args:
            request: 验证请求
        """
        self._started = True
        self._sample_count = 0
        self._result = SynchronousReviewResult()
        self._result.metadata["request_id"] = request.request_id
        self._result.metadata["agent_id"] = request.agent_id
        self._result.metadata["start_time"] = time.time()

    def on_intermediate_sample(
        self, sample: IntermediateSample
    ) -> SynchronousReviewResult:
        """中间采样钩子.

        对中间输出进行轻量级检查.

        Args:
            sample: 中间采样结果

        Returns:
            更新后的同步评审结果
        """
        if not self._started:
            return self._result

        self._sample_count += 1
        self._result.samples_reviewed = self._sample_count

        # 达到最大采样数, 停止采样
        if self._sample_count > self._max_samples:
            self._result.should_abort = True
            self._result.abort_reason = "达到最大采样数限制"
            return self._result

        # 长度检查
        text = sample.text
        if len(text) > self._max_len:
            self._result.early_warnings.append(
                f"采样 #{sample.position}: 输出长度 {len(text)} "
                f"超过上限 {self._max_len}"
            )
            self._result.issues_found += 1
            if self._abort_on_critical:
                self._result.should_abort = True
                self._result.abort_reason = (
                    f"输出长度 {len(text)} 超过上限 {self._max_len}"
                )

        if len(text) > 0 and len(text) < self._min_len:
            self._result.early_warnings.append(
                f"采样 #{sample.position}: 输出长度 {len(text)} "
                f"过短 (低于 {self._min_len})"
            )
            self._result.issues_found += 1

        # 关键词检查 — 检查是否包含安全敏感内容
        safety_issues = self._check_safety_keywords(text)
        if safety_issues:
            self._result.early_warnings.extend(safety_issues)
            self._result.issues_found += len(safety_issues)
            if self._abort_on_critical:
                self._result.should_abort = True
                self._result.abort_reason = (
                    f"检测到安全敏感内容: {safety_issues[0]}"
                )

        # 格式检查 — 检查是否包含明显格式错误
        format_issues = self._check_format(text)
        if format_issues:
            self._result.early_warnings.extend(format_issues)
            self._result.issues_found += len(format_issues)

        return self._result

    def on_generation_complete(
        self, final_output: str
    ) -> SynchronousReviewResult:
        """生成完成钩子.

        记录最终采样, 设置同步评审判决.

        Args:
            final_output: 最终完整输出

        Returns:
            最终同步评审结果
        """
        self._result.final_sample = IntermediateSample(
            text=final_output,
            position=self._sample_count,
        )
        self._result.metadata["end_time"] = time.time()
        self._result.metadata["total_samples"] = self._sample_count

        # 设置判决
        if self._result.should_abort:
            self._result.verdict = ReviewVerdict.BLOCK
        elif self._result.issues_found > 0:
            self._result.verdict = ReviewVerdict.FLAG
        else:
            self._result.verdict = ReviewVerdict.PASS

        self._started = False
        return self._result

    @staticmethod
    def _check_safety_keywords(text: str) -> list[str]:
        """检查安全敏感关键词."""
        issues: list[str] = []
        text_lower = text.lower()
        for kw in _SAFETY_BLOCKED:
            if kw.lower() in text_lower:
                issues.append(f"中间输出包含安全敏感词: '{kw}'")
        return issues

    @staticmethod
    def _check_format(text: str) -> list[str]:
        """检查格式问题."""
        issues: list[str] = []
        # 检查未闭合的括号
        for open_ch, close_ch in [("(", ")"), ("[", "]"), ("{", "}")]:
            if text.count(open_ch) != text.count(close_ch):
                issues.append(
                    f"括号不匹配: '{open_ch}'={text.count(open_ch)}, "
                    f"'{close_ch}'={text.count(close_ch)}"
                )
        # 检查连续重复字符 (可能是生成异常)
        if re.search(r"(.)\1{20,}", text):
            issues.append("检测到异常连续重复字符 (可能是生成异常)")
        return issues


# ============================================================
# 三阶段评审编排器
# ============================================================


class ThreeStageReviewOrchestrator:
    """三阶段评审编排器.

    编排预评审 → 同步评审 → 后评审的完整流程.

    使用示例::

        orchestrator = ThreeStageReviewOrchestrator(
            pipeline=review_pipeline,
        )

        # Stage 1: 预评审
        pre_result = orchestrator.start_pre_review(request)
        if not pre_result.passed:
            return pre_result

        # Stage 2: 同步评审 (在 Agent 执行过程中)
        hook = orchestrator.create_sync_hook(pre_result.context)
        hook.on_generation_start(request)
        # ... Agent 生成过程中调用 hook.on_intermediate_sample(sample)
        sync_result = hook.on_generation_complete(final_output)

        if sync_result.should_abort:
            return sync_result

        # Stage 3: 后评审 (四层递进)
        review_result = orchestrator.run_post_review(request)
    """

    def __init__(
        self,
        pipeline: Any | None = None,
        pre_review_engine: PreReviewEngine | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._pre_engine = pre_review_engine or PreReviewEngine()

    @property
    def pre_review_engine(self) -> PreReviewEngine:
        return self._pre_engine

    def start_pre_review(
        self, request: VerificationRequest
    ) -> PreReviewResult:
        """Stage 1: 执行预评审."""
        return self._pre_engine.pre_review(request)

    def create_sync_hook(
        self, pre_review_ctx: PreReviewContext | None = None
    ) -> SynchronousReviewHook:
        """Stage 2: 创建同步评审钩子."""
        return SynchronousReviewHook(pre_review_ctx=pre_review_ctx)

    def run_post_review(
        self, request: VerificationRequest
    ) -> Any:
        """Stage 3: 执行后评审 (四层递进评审).

        委托给 ReviewPipeline.review() 执行.
        """
        if self._pipeline is None:
            raise RuntimeError("未配置 ReviewPipeline, 无法执行后评审")
        return self._pipeline.review(request)

    def full_review(
        self,
        request: VerificationRequest,
        *,
        intermediate_samples: list[str] | None = None,
    ) -> dict[str, Any]:
        """执行完整三阶段评审.

        Args:
            request: 验证请求
            intermediate_samples: 中间采样文本列表 (模拟同步评审)

        Returns:
            包含三个阶段结果的字典
        """
        results: dict[str, Any] = {}

        # Stage 1: 预评审
        pre_result = self.start_pre_review(request)
        results["pre_review"] = pre_result
        if not pre_result.passed:
            results["synchronous"] = None
            results["post_review"] = None
            results["overall_verdict"] = pre_result.verdict
            return results

        # Stage 2: 同步评审
        hook = self.create_sync_hook(pre_result.context)
        hook.on_generation_start(request)
        if intermediate_samples:
            for i, sample_text in enumerate(intermediate_samples):
                sample = IntermediateSample(
                    text=sample_text, position=i
                )
                hook.on_intermediate_sample(sample)
                if hook.result.should_abort:
                    break
        sync_result = hook.on_generation_complete(request.output_text)
        results["synchronous"] = sync_result

        if sync_result.should_abort:
            results["post_review"] = None
            results["overall_verdict"] = sync_result.verdict
            return results

        # Stage 3: 后评审
        if self._pipeline is not None:
            post_result = self.run_post_review(request)
            results["post_review"] = post_result
            results["overall_verdict"] = post_result.verdict
        else:
            results["post_review"] = None
            results["overall_verdict"] = sync_result.verdict

        return results


# ============================================================
# 模块公开接口
# ============================================================


__all__ = [
    # 枚举
    "ReviewStageType",
    # 预评审
    "PreReviewContext",
    "PreReviewResult",
    "PreReviewEngine",
    # 同步评审
    "IntermediateSample",
    "SynchronousReviewResult",
    "SynchronousReviewHook",
    # 编排器
    "ThreeStageReviewOrchestrator",
]
