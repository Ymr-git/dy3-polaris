"""CC1 防幻觉层 — 验证器框架.

融合 Guardrails AI 的 @register_validator 模式与 NeMo Guardrails 的分层 Rail 架构，
提供可插拔的验证器注册机制和四种内置验证器。

验证器协议设计为纯函数式：输入 (claim, context) → ClaimVerificationResult，
支持同步和异步扩展，与管道解耦。

内置验证器：
- CitationVerifier: 引用覆盖率验证（LlamaIndex Citation 启发）
- GroundednessVerifier: 忠实度验证（RAGAS Faithfulness 启发）
- ConsistencyVerifier: 一致性验证（SelfCheckGPT 启发）
- FactCheckVerifier: 事实核查（LLM-as-Judge + 参考答案比对）
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Protocol, runtime_checkable

from .models import (
    Claim,
    ClaimVerificationResult,
    ClaimType,
    Evidence,
    EvidenceType,
    VerifierType,
)


# ============================================================
# 验证器协议（Guardrails AI Validator 接口启发）
# ============================================================


@runtime_checkable
class BaseVerifier(Protocol):
    """验证器协议.

    所有验证器必须实现此接口。验证器是纯函数式设计：
    输入声明和验证上下文，输出验证结果。

    验证器不应有副作用，所有状态变更由管道层处理。
    """

    verifier_type: VerifierType

    def verify(
        self,
        claim: Claim,
        *,
        context_chunks: list[str] | None = None,
        evidence: list[Evidence] | None = None,
        sample_outputs: list[str] | None = None,
        reference_answer: str | None = None,
        threshold: float = 0.7,
        params: dict[str, Any] | None = None,
    ) -> ClaimVerificationResult:
        """验证单个声明.

        Args:
            claim: 待验证的声明
            context_chunks: 检索到的上下文片段
            evidence: 关联的证据列表
            sample_outputs: 采样输出列表（一致性验证用）
            reference_answer: 参考答案（事实核查用）
            threshold: 通过阈值
            params: 验证器特定参数

        Returns:
            验证结果
        """
        ...


# ============================================================
# 验证器注册表（Guardrails AI @register_validator 启发）
# ============================================================


class VerifierRegistry:
    """验证器注册表.

    提供验证器的注册、查找和管理能力。
    支持运行时动态注册自定义验证器。

    使用示例::

        registry = VerifierRegistry()
        registry.register(CitationVerifier())
        verifier = registry.get(VerifierType.CITATION)
        result = verifier.verify(claim, context_chunks=[...])
    """

    def __init__(self) -> None:
        self._verifiers: dict[VerifierType, BaseVerifier] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册内置验证器."""
        self.register(CitationVerifier())
        self.register(GroundednessVerifier())
        self.register(ConsistencyVerifier())
        self.register(FactCheckVerifier())

    def register(self, verifier: BaseVerifier) -> None:
        """注册验证器."""
        vtype = getattr(verifier, "verifier_type", None)
        if vtype is None:
            raise ValueError(f"验证器 {type(verifier).__name__} 缺少 verifier_type 属性")
        self._verifiers[vtype] = verifier

    def get(self, vtype: VerifierType) -> BaseVerifier | None:
        """获取验证器."""
        return self._verifiers.get(vtype)

    def list_types(self) -> list[VerifierType]:
        """列出已注册的验证器类型."""
        return list(self._verifiers.keys())

    @property
    def count(self) -> int:
        """已注册验证器数量."""
        return len(self._verifiers)


# ============================================================
# 内置验证器
# ============================================================


class CitationVerifier:
    """引用覆盖率验证器（LlamaIndex CitationQueryEngine 启发）.

    检查声明是否有引用支撑。声明分为两类：
    - 需要引用的声明（factual/numerical/citation/definition）：必须有证据关联
    - 不强制引用的声明（inference/opinion）：有引用则加分

    评分逻辑：
    - 需要引用且有证据: score=1.0, passed=True
    - 需要引用但无证据: score=0.0, passed=False
    - 不强制引用有证据: score=1.0, passed=True
    - 不强制引用无证据: score=0.5, passed=True（低优先级放行）
    """

    verifier_type: VerifierType = VerifierType.CITATION

    # 需要引用的声明类型
    _REQUIRES_CITATION = frozenset({
        ClaimType.FACTUAL,
        ClaimType.NUMERICAL,
        ClaimType.CITATION,
        ClaimType.DEFINITION,
    })

    def verify(
        self,
        claim: Claim,
        *,
        context_chunks: list[str] | None = None,
        evidence: list[Evidence] | None = None,
        sample_outputs: list[str] | None = None,
        reference_answer: str | None = None,
        threshold: float = 0.7,
        params: dict[str, Any] | None = None,
    ) -> ClaimVerificationResult:
        ev_list = evidence or []
        has_evidence = bool(claim.evidence_ids) or len(ev_list) > 0

        requires_citation = claim.claim_type in self._REQUIRES_CITATION

        if has_evidence:
            score = 1.0
            reason = "声明有证据支撑"
            evidence_ids = claim.evidence_ids or [e.evidence_id for e in ev_list]
            confidence = min(1.0, max(e.confidence for e in ev_list)) if ev_list else 0.8
        elif requires_citation:
            score = 0.0
            reason = f"声明类型 {claim.claim_type.value} 需要引用但未提供证据"
            evidence_ids = []
            confidence = 0.9
        else:
            score = 0.5
            reason = f"声明类型 {claim.claim_type.value} 不强制引用"
            evidence_ids = []
            confidence = 0.6

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            verifier_type=self.verifier_type,
            passed=score >= threshold,
            score=score,
            confidence=confidence,
            evidence_ids=evidence_ids,
            reason=reason,
        )


class GroundednessVerifier:
    """忠实度验证器（RAGAS Faithfulness 启发）.

    检查声明是否能从提供的上下文中推断出来。

    评分逻辑（文本相似度近似）：
    - 声明文本与上下文的最高序列匹配比率作为忠实度分数
    - 无上下文时视为无法验证（score=0, confidence=1）
    - 支持子串匹配加分（声明关键词出现在上下文中）

    在实际部署中，此验证器可替换为 LLM-as-Judge 实现。
    """

    verifier_type: VerifierType = VerifierType.GROUNDEDNESS

    def verify(
        self,
        claim: Claim,
        *,
        context_chunks: list[str] | None = None,
        evidence: list[Evidence] | None = None,
        sample_outputs: list[str] | None = None,
        reference_answer: str | None = None,
        threshold: float = 0.7,
        params: dict[str, Any] | None = None,
    ) -> ClaimVerificationResult:
        chunks = context_chunks or []

        if not chunks:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                verifier_type=self.verifier_type,
                passed=False,
                score=0.0,
                confidence=1.0,
                reason="无上下文提供，无法验证忠实度",
            )

        # 计算声明与各上下文片段的相似度
        best_score = 0.0
        best_chunk_idx = -1
        claim_lower = claim.text.lower()
        claim_keywords = self._extract_keywords(claim.text)

        for idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()

            # 序列匹配比率
            ratio = SequenceMatcher(None, claim_lower, chunk_lower).ratio()

            # 关键词覆盖率加分
            if claim_keywords:
                matched = sum(1 for kw in claim_keywords if kw in chunk_lower)
                keyword_coverage = matched / len(claim_keywords)
                # 综合分数：70% 序列匹配 + 30% 关键词覆盖
                combined = ratio * 0.7 + keyword_coverage * 0.3
            else:
                combined = ratio

            if combined > best_score:
                best_score = combined
                best_chunk_idx = idx

        # 确保证据 ID 关联
        evidence_ids: list[str] = []
        if evidence:
            for ev in evidence:
                if ev.evidence_type in (EvidenceType.RETRIEVED_CONTEXT, EvidenceType.KNOWLEDGE_BASE):
                    evidence_ids.append(ev.evidence_id)

        reason = (
            f"最佳上下文匹配: 片段[{best_chunk_idx}] 分数={best_score:.3f}"
            if best_chunk_idx >= 0
            else "无匹配上下文"
        )

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            verifier_type=self.verifier_type,
            passed=best_score >= threshold,
            score=round(min(best_score, 1.0), 4),
            confidence=0.75,
            evidence_ids=evidence_ids,
            reason=reason,
        )

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """提取关键词（简单分词 + 停用词过滤）."""
        # 移除标点和特殊字符，按空格分词
        words = re.findall(r"\b\w+\b", text.lower())
        # 过滤停用词和短词
        stop_words = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "up", "down", "out", "off", "over",
            "under", "again", "further", "then", "once", "and", "but", "or",
            "nor", "not", "so", "than", "too", "very", "s", "t", "just",
            "it", "its", "this", "that", "these", "those", "i", "you",
            "he", "she", "we", "they", "what", "which", "who", "whom",
            "的", "了", "是", "在", "和", "也", "都", "就", "还", "又",
        })
        return [w for w in words if len(w) > 2 and w not in stop_words]


class ConsistencyVerifier:
    """一致性验证器（SelfCheckGPT 启发）.

    通过比较多采样输出的一致性检测幻觉。

    评分逻辑：
    - 无采样输出时跳过验证（score=1.0, confidence=1.0）
    - 有采样输出时：计算声明文本与各采样输出的最大序列匹配比率
    - 一致性分数 = 平均最大匹配比率

    核心洞察：如果 LLM 真正掌握某概念，
    多次采样的回答会趋于一致；幻觉事实则会在采样中相互矛盾。
    """

    verifier_type: VerifierType = VerifierType.CONSISTENCY

    def verify(
        self,
        claim: Claim,
        *,
        context_chunks: list[str] | None = None,
        evidence: list[Evidence] | None = None,
        sample_outputs: list[str] | None = None,
        reference_answer: str | None = None,
        threshold: float = 0.7,
        params: dict[str, Any] | None = None,
    ) -> ClaimVerificationResult:
        samples = sample_outputs or []

        if not samples:
            # 无采样输出时跳过（视为一致）
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                verifier_type=self.verifier_type,
                passed=True,
                score=1.0,
                confidence=1.0,
                reason="无采样输出，跳过一致性验证",
            )

        claim_lower = claim.text.lower()

        # 计算与每个采样输出的最大匹配比率
        max_ratios: list[float] = []
        for sample in samples:
            sample_lower = sample.lower()
            ratio = SequenceMatcher(None, claim_lower, sample_lower).ratio()
            max_ratios.append(ratio)

        # 一致性分数 = 平均最大匹配比率
        consistency_score = sum(max_ratios) / len(max_ratios) if max_ratios else 0.0

        # 附加关键词级检查
        claim_keywords = GroundednessVerifier._extract_keywords(claim.text)
        if claim_keywords:
            keyword_scores: list[float] = []
            for sample in samples:
                sample_lower = sample.lower()
                matched = sum(1 for kw in claim_keywords if kw in sample_lower)
                keyword_scores.append(matched / len(claim_keywords))
            keyword_avg = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
            # 综合分数：60% 序列匹配 + 40% 关键词覆盖
            consistency_score = consistency_score * 0.6 + keyword_avg * 0.4

        reason = (
            f"一致性验证: {len(samples)} 个采样输出, "
            f"平均匹配={consistency_score:.3f}"
        )

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            verifier_type=self.verifier_type,
            passed=consistency_score >= threshold,
            score=round(min(consistency_score, 1.0), 4),
            confidence=0.8,
            reason=reason,
        )


class FactCheckVerifier:
    """事实核查验证器（LLM-as-Judge + 参考答案比对启发）.

    通过与参考答案比对来验证声明的事实正确性。

    评分逻辑：
    - 无参考答案时跳过验证（score=1.0, confidence=1.0）
    - 有参考答案时：计算声明与参考答案的序列匹配比率
    - 匹配比率 >= threshold → 通过

    在实际部署中，此验证器可替换为外部知识库查询或 LLM-as-Judge 实现。
    """

    verifier_type: VerifierType = VerifierType.FACT_CHECK

    def verify(
        self,
        claim: Claim,
        *,
        context_chunks: list[str] | None = None,
        evidence: list[Evidence] | None = None,
        sample_outputs: list[str] | None = None,
        reference_answer: str | None = None,
        threshold: float = 0.7,
        params: dict[str, Any] | None = None,
    ) -> ClaimVerificationResult:
        if not reference_answer:
            # 无参考答案时跳过
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                verifier_type=self.verifier_type,
                passed=True,
                score=1.0,
                confidence=1.0,
                reason="无参考答案，跳过事实核查",
            )

        ref_lower = reference_answer.lower()
        claim_lower = claim.text.lower()

        # 序列匹配
        ratio = SequenceMatcher(None, claim_lower, ref_lower).ratio()

        # 关键词覆盖率
        claim_keywords = GroundednessVerifier._extract_keywords(claim.text)
        if claim_keywords:
            matched = sum(1 for kw in claim_keywords if kw in ref_lower)
            keyword_coverage = matched / len(claim_keywords)
            # 综合分数：60% 序列匹配 + 40% 关键词覆盖
            score = ratio * 0.6 + keyword_coverage * 0.4
        else:
            score = ratio

        reason = f"参考答案匹配: 序列={ratio:.3f}, 综合={score:.3f}"

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            verifier_type=self.verifier_type,
            passed=score >= threshold,
            score=round(min(score, 1.0), 4),
            confidence=0.85,
            reason=reason,
        )
