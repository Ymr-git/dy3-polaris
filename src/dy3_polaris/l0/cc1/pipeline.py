"""CC1 防幻觉层 — 验证管道.

融合 CoVe 的四阶段 pipeline 与 NeMo Guardrails 的分层 Rail 架构，
编排从输入到输出的完整防幻觉验证流程。

管道阶段（CoVe 四阶段 + Guardrails 修正启发）：
1. Input: 输入检查（安全过滤、格式验证）
2. Claim Extraction: 声明提取（RAGAS 原子化分解）
3. Verification: 声明验证（多验证器并行执行）
4. Output: 输出决策（综合评分 → 动作判决 → 修正输出）
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from .exceptions import ClaimExtractionError, VerificationError
from .models import (
    Claim,
    ClaimType,
    ClaimVerificationResult,
    Evidence,
    EvidenceType,
    HallucinationRecord,
    HallucinationSeverity,
    PipelineConfig,
    VerificationReport,
    VerificationRequest,
    VerificationStage,
    VerificationStatus,
    VerdictAction,
    VerifierConfig,
    VerifierType,
)
from .verifiers import BaseVerifier, VerifierRegistry

logger = logging.getLogger(__name__)


# ============================================================
# 声明提取器（RAGAS 声明分解启发）
# ============================================================


class ClaimExtractor:
    """声明提取器.

    将 LLM 输出文本分解为原子声明（最小可验证的事实单元）。

    提取策略：
    - 按句号/换行分句
    - 过滤空句和纯标点句
    - 自动推断声明类型
    - 记录原文位置（source_span）
    """

    # 句子分隔正则：中英文句号、问号、感叹号、换行
    _SENTENCE_SPLIT = re.compile(r"[。！？.!?\n]+")

    # 数字检测正则
    _NUMERICAL_PATTERN = re.compile(r"\d+\.?\d*")

    # 引用标记检测正则（如 [1], [2], (Source 1)）
    _CITATION_PATTERN = re.compile(r"\[\d+\]|\(Source\s+\d+\)|【\d+】")

    # 定义性陈述检测正则
    _DEFINITION_PATTERN = re.compile(
        r"(?:是指|定义为|means|is defined as|refers to|指的是)",
        re.IGNORECASE,
    )

    # 推断性陈述检测正则
    _INFERENCE_PATTERN = re.compile(
        r"(?:因此|所以|可以推断|thus|therefore|we can conclude|it can be inferred)",
        re.IGNORECASE,
    )

    # 观点性陈述检测正则
    _OPINION_PATTERN = re.compile(
        r"(?:我认为|建议|应该|perhaps|arguably|in my opinion|recommend)",
        re.IGNORECASE,
    )

    def extract(self, text: str, max_claims: int = 50) -> list[Claim]:
        """从文本中提取原子声明.

        Args:
            text: LLM 输出文本
            max_claims: 最大声明数

        Returns:
            声明列表

        Raises:
            ClaimExtractionError: 提取失败
        """
        if not text or not text.strip():
            return []

        try:
            # 分句
            raw_sentences = self._SENTENCE_SPLIT.split(text)
            sentences: list[tuple[str, tuple[int, int]]] = []

            # 跟踪原文位置
            pos = 0
            for sent in raw_sentences:
                sent_stripped = sent.strip()
                if not sent_stripped or len(sent_stripped) < 3:
                    pos += len(sent) + 1
                    continue

                # 查找在原文中的位置
                start = text.find(sent_stripped, pos)
                if start == -1:
                    start = pos
                end = start + len(sent_stripped)
                pos = end

                sentences.append((sent_stripped, (start, end)))

            # 限制数量
            sentences = sentences[:max_claims]

            # 构建 Claim 对象
            claims: list[Claim] = []
            for sent_text, span in sentences:
                claim_type = self._infer_claim_type(sent_text)
                claim = Claim(
                    text=sent_text,
                    claim_type=claim_type,
                    source_span=span,
                )
                claims.append(claim)

            return claims

        except Exception as exc:
            raise ClaimExtractionError(
                f"声明提取失败: {exc}",
                detail=str(exc),
            ) from exc

    def _infer_claim_type(self, text: str) -> ClaimType:
        """推断声明类型."""
        if self._CITATION_PATTERN.search(text):
            return ClaimType.CITATION
        if self._DEFINITION_PATTERN.search(text):
            return ClaimType.DEFINITION
        if self._INFERENCE_PATTERN.search(text):
            return ClaimType.INFERENCE
        if self._OPINION_PATTERN.search(text):
            return ClaimType.OPINION
        if self._NUMERICAL_PATTERN.search(text):
            return ClaimType.NUMERICAL
        return ClaimType.FACTUAL


# ============================================================
# 证据收集器（LlamaIndex Citation 启发）
# ============================================================


class EvidenceCollector:
    """证据收集器.

    从验证请求的上下文片段和引用中收集证据，
    并将证据关联到相应的声明。

    策略：
    - 将每个上下文片段包装为 Evidence 对象
    - 将每个引用 URI 包装为 Evidence 对象
    - 通过文本相似度将证据关联到声明
    """

    def collect(
        self,
        request: VerificationRequest,
        claims: list[Claim],
    ) -> list[Evidence]:
        """收集证据并关联到声明.

        Args:
            request: 验证请求（含上下文和引用）
            claims: 提取的声明列表

        Returns:
            证据列表（同时更新 claims 的 evidence_ids）
        """
        evidence_list: list[Evidence] = []

        # 从上下文片段创建证据
        for idx, chunk in enumerate(request.context_chunks):
            evd = Evidence(
                evidence_type=EvidenceType.RETRIEVED_CONTEXT,
                content=chunk,
                source_uri=f"context://{idx}",
                confidence=0.8,
                metadata={"chunk_index": idx},
            )
            evidence_list.append(evd)

        # 从引用 URI 创建证据
        for idx, uri in enumerate(request.citations):
            evd = Evidence(
                evidence_type=EvidenceType.EXTERNAL_SOURCE,
                content="",
                source_uri=uri,
                confidence=0.7,
                metadata={"citation_index": idx},
            )
            evidence_list.append(evd)

        # 将证据关联到声明（基于文本相似度）
        for claim in claims:
            best_evidence_ids: list[str] = []
            best_score = 0.0

            for evd in evidence_list:
                if not evd.content:
                    continue
                score = self._similarity(claim.text, evd.content)
                if score > best_score and score > 0.15:
                    best_score = score
                    best_evidence_ids = [evd.evidence_id]
                elif score > 0.15:
                    best_evidence_ids.append(evd.evidence_id)

            if best_evidence_ids:
                claim.evidence_ids = best_evidence_ids[:3]  # 最多关联 3 个证据

        return evidence_list

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """计算声明与证据的相似度.

        ``SequenceMatcher`` compares complete strings, so a sentence copied
        verbatim from a long paper chunk used to receive a low score merely
        because the surrounding chunk was long.  Direct containment is the
        strongest available local grounding fact and must be recognised before
        the fuzzy fallback.
        """
        from difflib import SequenceMatcher

        a_normalized = re.sub(r"\s+", " ", str(a or "")).strip().casefold()
        b_normalized = re.sub(r"\s+", " ", str(b or "")).strip().casefold()
        if a_normalized and a_normalized in b_normalized:
            return 1.0
        return SequenceMatcher(None, a_normalized, b_normalized).ratio()


# ============================================================
# 防幻觉验证管道
# ============================================================


class AntiHallucinationPipeline:
    """防幻觉验证管道.

    G3 核心组件，编排从输入到输出的完整防幻觉验证流程。

    管道阶段（CoVe 四阶段 + Guardrails 修正）：
    1. Input: 输入检查（格式验证、安全过滤）
    2. Claim Extraction: 声明提取（RAGAS 原子化分解）
    3. Verification: 声明验证（多验证器并行执行）
    4. Output: 输出决策（综合评分 → 动作判决 → 修正输出）

    使用示例::

        registry = VerifierRegistry()
        pipeline = AntiHallucinationPipeline(registry)
        report = pipeline.verify(VerificationRequest(
            output_text="水的沸点是100度。",
            context_chunks=["在标准大气压下，水的沸点为100摄氏度。"],
        ))
        assert report.status == VerificationStatus.PASSED
    """

    def __init__(
        self,
        registry: VerifierRegistry | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._registry = registry or VerifierRegistry()
        self._config = config or self._default_config()
        self._extractor = ClaimExtractor()
        self._collector = EvidenceCollector()
        self._lock = threading.RLock()

    @staticmethod
    def _default_config() -> PipelineConfig:
        """默认管道配置：启用全部四种内置验证器."""
        return PipelineConfig(
            verifiers=[
                VerifierConfig(verifier_type=VerifierType.CITATION, enabled=True, threshold=0.7),
                VerifierConfig(verifier_type=VerifierType.GROUNDEDNESS, enabled=True, threshold=0.3),
                VerifierConfig(verifier_type=VerifierType.CONSISTENCY, enabled=True, threshold=0.5),
                VerifierConfig(verifier_type=VerifierType.FACT_CHECK, enabled=True, threshold=0.5),
            ],
            pass_threshold=0.7,
            degrade_threshold=0.5,
            refuse_threshold=0.3,
        )

    # --------------------------------------------------------
    # 核心验证流程
    # --------------------------------------------------------

    def verify(self, request: VerificationRequest) -> VerificationReport:
        """执行完整防幻觉验证.

        Args:
            request: 验证请求

        Returns:
            验证报告
        """
        report = VerificationReport(
            request_id=request.request_id,
            agent_id=request.agent_id,
            original_output=request.output_text,
            stage=VerificationStage.INPUT,
            status=VerificationStatus.IN_PROGRESS,
        )

        try:
            # 阶段 1: 输入检查
            self._stage_input(request, report)

            # 阶段 2: 声明提取
            self._stage_claim_extraction(request, report)

            # 阶段 3: 证据收集 + 声明验证
            self._stage_verification(request, report)

            # 阶段 4: 输出决策
            self._stage_output(request, report)

        except Exception as exc:
            report.status = VerificationStatus.FAILED
            report.metadata["error"] = str(exc)
            report.completed_at = time.time()
            logger.exception("防幻觉验证失败: %s", exc)
            raise

        report.completed_at = time.time()
        return report

    # --------------------------------------------------------
    # 阶段实现
    # --------------------------------------------------------

    def _stage_input(
        self,
        request: VerificationRequest,
        report: VerificationReport,
    ) -> None:
        """阶段 1: 输入检查（NeMo input rail 启发）."""
        report.stage = VerificationStage.INPUT

        if not request.output_text or not request.output_text.strip():
            report.status = VerificationStatus.SKIPPED
            report.action = VerdictAction.PASS
            report.metadata["skip_reason"] = "空输出"
            return

        report.status = VerificationStatus.IN_PROGRESS

    def _stage_claim_extraction(
        self,
        request: VerificationRequest,
        report: VerificationReport,
    ) -> None:
        """阶段 2: 声明提取（RAGAS 声明分解启发）."""
        report.stage = VerificationStage.CLAIM_EXTRACTION

        if report.status == VerificationStatus.SKIPPED:
            return

        claims = self._extractor.extract(
            request.output_text,
            max_claims=self._config.max_claims,
        )
        report.claims = claims

    def _stage_verification(
        self,
        request: VerificationRequest,
        report: VerificationReport,
    ) -> None:
        """阶段 3: 证据收集 + 声明验证."""
        report.stage = VerificationStage.VERIFICATION

        if report.status == VerificationStatus.SKIPPED or not report.claims:
            return

        # 收集证据
        evidence = self._collector.collect(request, report.claims)
        report.evidence = evidence

        # 对每个声明运行启用的验证器
        all_results: list[ClaimVerificationResult] = []

        for claim in report.claims:
            claim_results = self._verify_claim(claim, request, evidence)
            all_results.extend(claim_results)

        report.claim_results = all_results

    def _stage_output(
        self,
        request: VerificationRequest,
        report: VerificationReport,
    ) -> None:
        """阶段 4: 输出决策（Guardrails on_fail 修正启发）."""
        report.stage = VerificationStage.OUTPUT

        if report.status == VerificationStatus.SKIPPED:
            return

        # 计算综合分数
        report.compute_scores()

        # 判定幻觉
        report.determine_hallucination()

        # 决定动作
        report.action = self._decide_action(report)

        # 生成修正输出
        if (
            self._config.enable_corrected_output
            and report.action in (VerdictAction.FIX, VerdictAction.DEGRADE)
        ):
            report.corrected_output = self._generate_corrected_output(report)

        # 设置最终状态
        if report.action == VerdictAction.PASS:
            report.status = VerificationStatus.PASSED
        elif report.action == VerdictAction.REFUSE:
            report.status = VerificationStatus.REFUSED
        elif report.action in (VerdictAction.DEGRADE, VerdictAction.FIX):
            report.status = VerificationStatus.DEGRADED
        elif report.action == VerdictAction.LOG_ONLY:
            report.status = VerificationStatus.PASSED
        else:
            report.status = VerificationStatus.PASSED

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _verify_claim(
        self,
        claim: Claim,
        request: VerificationRequest,
        evidence: list[Evidence],
    ) -> list[ClaimVerificationResult]:
        """对单个声明运行所有启用的验证器."""
        results: list[ClaimVerificationResult] = []

        # 筛选与此声明相关的证据
        claim_evidence = [
            e for e in evidence
            if e.evidence_id in claim.evidence_ids
        ]

        for vc in self._config.verifiers:
            if not vc.enabled:
                continue

            verifier = self._registry.get(vc.verifier_type)
            if verifier is None:
                logger.warning("验证器 %s 未注册，跳过", vc.verifier_type)
                continue

            try:
                result = verifier.verify(
                    claim,
                    context_chunks=request.context_chunks,
                    evidence=claim_evidence,
                    sample_outputs=request.sample_outputs,
                    reference_answer=request.reference_answer,
                    threshold=vc.threshold,
                    params=vc.params,
                )
                results.append(result)
            except Exception as exc:
                logger.exception("验证器 %s 执行失败: %s", vc.verifier_type, exc)
                results.append(ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    verifier_type=vc.verifier_type,
                    passed=False,
                    score=0.0,
                    confidence=0.0,
                    reason=f"验证器执行异常: {exc}",
                ))

        return results

    def _decide_action(self, report: VerificationReport) -> VerdictAction:
        """根据综合分数决定动作（Guardrails on_fail 启发）.

        决策逻辑：
        - score >= pass_threshold → PASS
        - score < refuse_threshold → REFUSE
        - score < degrade_threshold → DEGRADE（降级输出）
        - 否则 → FIX（尝试修正）

        特殊情况：
        - 检测到 CRITICAL 级幻觉 → REFUSE（无论分数）
        """
        if report.hallucination_severity == HallucinationSeverity.CRITICAL:
            return VerdictAction.REFUSE

        if report.overall_score >= self._config.pass_threshold:
            return VerdictAction.PASS

        if report.overall_score < self._config.refuse_threshold:
            return VerdictAction.REFUSE

        if report.overall_score < self._config.degrade_threshold:
            return VerdictAction.DEGRADE

        # 处于 degrade 和 pass 之间，尝试修正
        if self._config.enable_claim_correction:
            return VerdictAction.FIX
        return VerdictAction.DEGRADE

    def _generate_corrected_output(self, report: VerificationReport) -> str:
        """生成修正输出（CoVe 修正启发）.

        策略：
        - 移除未通过验证的声明
        - 保留通过的声明
        - 附加修正说明
        """
        # 收集未通过的声明 ID
        failed_claim_ids: set[str] = set()
        for result in report.claim_results:
            if not result.passed:
                failed_claim_ids.add(result.claim_id)

        # 保留通过的声明
        passed_claims = [
            c for c in report.claims
            if c.claim_id not in failed_claim_ids
        ]

        if not passed_claims:
            return "[验证修正] 所有声明均未通过验证，输出已被移除。"

        parts = [c.text for c in passed_claims]
        corrected = " ".join(parts)

        # 附加修正说明
        removed_count = len(report.claims) - len(passed_claims)
        if removed_count > 0:
            corrected += f"\n\n[验证修正] 已移除 {removed_count} 条未通过验证的声明。"

        return corrected

    # --------------------------------------------------------
    # 配置与注册表管理
    # --------------------------------------------------------

    @property
    def config(self) -> PipelineConfig:
        """获取管道配置."""
        return self._config

    def get_config(self) -> PipelineConfig:
        """获取管道配置（G6 路由层兼容方法调用）."""
        return self._config

    @property
    def registry(self) -> VerifierRegistry:
        """获取验证器注册表."""
        return self._registry

    def update_config(self, config: PipelineConfig) -> None:
        """更新管道配置."""
        self._config = config

    def register_verifier(self, verifier: BaseVerifier) -> None:
        """注册自定义验证器."""
        self._registry.register(verifier)

    # --------------------------------------------------------
    # 便捷方法
    # --------------------------------------------------------

    def verify_text(
        self,
        output_text: str,
        *,
        context_chunks: list[str] | None = None,
        citations: list[str] | None = None,
        sample_outputs: list[str] | None = None,
        reference_answer: str | None = None,
        agent_id: str = "",
        domain: str = "",
    ) -> VerificationReport:
        """便捷验证方法.

        无需手动构建 VerificationRequest，直接传入文本和上下文。
        """
        request = VerificationRequest(
            agent_id=agent_id,
            output_text=output_text,
            context_chunks=context_chunks or [],
            citations=citations or [],
            sample_outputs=sample_outputs or [],
            reference_answer=reference_answer,
            domain=domain,
        )
        return self.verify(request)

    def create_hallucination_record(
        self,
        report: VerificationReport,
    ) -> HallucinationRecord | None:
        """从验证报告创建幻觉检测记录.

        仅在检测到幻觉时创建。
        """
        if not report.hallucination_detected:
            return None

        failed_claims: list[str] = []
        failed_claim_ids: set[str] = set()
        for result in report.claim_results:
            if not result.passed:
                failed_claim_ids.add(result.claim_id)
        for claim in report.claims:
            if claim.claim_id in failed_claim_ids:
                failed_claims.append(claim.text)

        return HallucinationRecord(
            report_id=report.report_id,
            agent_id=report.agent_id,
            severity=report.hallucination_severity,
            failed_claims=failed_claims,
            action_taken=report.action,
            original_score=report.overall_score,
        )
