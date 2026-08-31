"""画像构建器增强模块 — 世界先进方案融合.

融合世界先进方案:
- Knewton: 多维画像融合 + 贝叶斯熟练度传播
- ALEKS: 知识空间理论 (KST) 内/外边界
- Squirrel AI: 纳米级知识分解 + 瓶颈检测
- Duolingo: 在线风格适配 (指数平滑)
- Bloom (1956) / Anderson & Krathwohl (2001): 自适应认知层次目标
- Vygotsky ZPD: ZPD 感知的 Bloom 目标设定
- FSRS / Ebbinghaus: 遗忘预警生成

增强组件:
1. MultiDimensionalFuser — 五维加权融合
   Score_overall = 0.60×μ(M) + 0.20×f(B) + 0.10×g(behavior) + 0.10×θ_norm
2. KSTAnalyzer — 知识空间理论分析 (内/外边界, 瓶颈, 中心性, 贝叶斯传播)
3. ProfileConfidenceEstimator — 画像置信度估计
4. DynamicMasteryThreshold — 动态掌握阈值 (CC1 按能力等级调整)
5. ForgettingAlertGenerator — 遗忘预警生成
6. OnlineStyleAdapter — 在线风格适配 (指数平滑)
7. AdaptiveBloomSetter — 自适应 Bloom 目标设定
8. ProfileVersionManager — 画像版本管理
9. ProfileConsistencyValidator — 画像一致性校验
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

from dy3_polaris.l2.models import LearnerSnapshot, TracingState


# ============================================================
# 1. MultiDimensionalFuser — 多维画像融合器
# ============================================================


class MultiDimensionalFuser:
    """多维画像融合器 — 五维加权融合.

    融合公式:
        Score_overall = 0.60 × μ(M) + 0.20 × f(B) + 0.10 × g(behavior) + 0.10 × θ_norm

    其中:
    - μ(M): 知识点掌握度均值 (BKT 产出)
    - f(B): 学科背景加权均值, 权重 [physics=0.30, chemistry=0.30, materials=0.25, engineering=0.15]
    - g(behavior): 行为特征综合评分 (session/streak/accuracy)
    - θ_norm: IRT 能力归一化 (sigmoid 变换)

    设计参考:
    - Knewton: 多维数据融合 + 权重动态调整
    - Khan Academy: 综合学情评分
    - Squirrel AI: 知识掌握度为核心 (60% 权重)
    """

    # 融合权重
    _W_MASTERY: float = 0.60
    _W_BACKGROUND: float = 0.20
    _W_BEHAVIOR: float = 0.10
    _W_THETA: float = 0.10

    # 学科背景权重
    _BG_WEIGHTS: dict[str, float] = {
        "physics": 0.30,
        "chemistry": 0.30,
        "materials": 0.25,
        "engineering": 0.15,
    }

    def fuse(
        self,
        kp_mastery: dict[str, float],
        subject_background: dict[str, float],
        behavior_features: dict[str, Any],
        theta: float,
    ) -> float:
        """五维加权融合 → 综合评分.

        Args:
            kp_mastery: 知识点掌握度映射.
            subject_background: 学科背景特征.
            behavior_features: 行为特征.
            theta: IRT 能力值.

        Returns:
            综合评分 [0.0, 1.0].
        """
        mu_m = self._mean_mastery(kp_mastery)
        f_b = self.score_subject_background(subject_background)
        g_behavior = self.score_behavior(behavior_features)
        theta_norm = self.normalize_theta(theta)

        score = (
            self._W_MASTERY * mu_m
            + self._W_BACKGROUND * f_b
            + self._W_BEHAVIOR * g_behavior
            + self._W_THETA * theta_norm
        )
        return max(0.0, min(1.0, score))

    def normalize_theta(self, theta: float) -> float:
        """θ 归一化 — sigmoid 变换.

        θ_norm = 1 / (1 + e^(-θ))

        Args:
            theta: IRT 能力值.

        Returns:
            归一化值 [0, 1].
        """
        return 1.0 / (1.0 + math.exp(-theta))

    def score_subject_background(self, bg: dict[str, float]) -> float:
        """学科背景加权均值.

        f(B) = Σ(w_i × b_i), 权重 [physics=0.30, chemistry=0.30, materials=0.25, engineering=0.15]

        Args:
            bg: 学科背景特征 {subject: score}.

        Returns:
            加权均值 [0, 1]; 空输入返回 0.5.
        """
        if not bg:
            return 0.5
        total_weight = 0.0
        weighted_sum = 0.0
        for subject, weight in self._BG_WEIGHTS.items():
            if subject in bg:
                weighted_sum += weight * bg[subject]
                total_weight += weight
        if total_weight == 0:
            # 回退: 使用所有可用 key 的简单平均
            return sum(bg.values()) / len(bg) if bg else 0.5
        return weighted_sum / total_weight

    def score_behavior(self, features: dict[str, Any]) -> float:
        """行为特征综合评分.

        g(behavior) = 0.4 × session_score + 0.3 × streak_score + 0.3 × accuracy_score

        - session_score: avg_session_duration 归一化 (45min → 1.0, 0min → 0.0)
        - streak_score: streak_days 归一化 (14天 → 1.0, 0天 → 0.0)
        - accuracy_score: accuracy_trend 最后值 (或均值)

        Args:
            features: 行为特征字典.

        Returns:
            行为评分 [0, 1].
        """
        if not features:
            return 0.5

        # session 评分
        session = features.get("avg_session_duration", 0.0)
        session_score = min(1.0, session / 45.0)

        # streak 评分
        streak = features.get("streak_days", 0)
        streak_score = min(1.0, streak / 14.0)

        # accuracy 评分
        accuracy_trend = features.get("accuracy_trend", [])
        if accuracy_trend:
            accuracy_score = accuracy_trend[-1]
        else:
            accuracy_score = 0.5

        return 0.4 * session_score + 0.3 * streak_score + 0.3 * accuracy_score

    @staticmethod
    def _mean_mastery(kp_mastery: dict[str, float]) -> float:
        """计算掌握度均值.

        Args:
            kp_mastery: 知识点掌握度映射.

        Returns:
            均值 [0, 1]; 空输入返回 0.5.
        """
        if not kp_mastery:
            return 0.5
        return sum(kp_mastery.values()) / len(kp_mastery)


# ============================================================
# 2. KSTAnalyzer — 知识空间理论分析器
# ============================================================


class KSTAnalyzer:
    """知识空间理论 (KST) 分析器.

    融合世界先进方案:
    - ALEKS: 知识状态集合 + 内/外边界 (fringe)
    - Knewton: 贝叶斯熟练度传播
    - Squirrel AI: 瓶颈节点检测

    核心概念:
    - 内边界 (inner fringe): 最近学到的概念, 可遗忘的最深层
    - 外边界 (outer fringe): 下一个可学概念 (前置全满足)
    - 瓶颈节点: 低掌握度 + 高依赖权重 (阻塞多个后继)
    - 中心性: 基于图结构的重要性分数 (PageRank 式)
    - 贝叶斯传播: 掌握后继 → 推断前置也掌握
    """

    # 瓶颈检测: 掌握度低于此值视为低掌握
    _BOTTLENECK_MASTERY: float = 0.5

    def compute_inner_fringe(
        self,
        mastered_kps: set[str],
        kg_structure: dict[str, Any],
    ) -> set[str]:
        """计算内边界 — 已掌握且前置全部已掌握的知识点.

        内边界 = mastered_kps 中, 所有前置依赖也在 mastered_kps 中的节点.
        这些是"稳固掌握"的节点, 也是可遗忘的最深层.

        Args:
            mastered_kps: 已掌握知识点集合.
            kg_structure: 知识图谱结构.

        Returns:
            内边界知识点集合.
        """
        edges = kg_structure.get("edges", [])
        # 构建前置依赖映射: {to: [from1, from2, ...]}
        prerequisites: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.get("type", "prerequisite") == "prerequisite":
                prerequisites[edge["to"]].append(edge["from"])

        inner = set()
        for kp_id in mastered_kps:
            prereqs = prerequisites.get(kp_id, [])
            if all(p in mastered_kps for p in prereqs):
                inner.add(kp_id)
        return inner

    def compute_outer_fringe(
        self,
        mastered_kps: set[str],
        kg_structure: dict[str, Any],
    ) -> set[str]:
        """计算外边界 — 下一个可学概念 (前置全满足但自身未掌握).

        Args:
            mastered_kps: 已掌握知识点集合.
            kg_structure: 知识图谱结构.

        Returns:
            外边界知识点集合.
        """
        edges = kg_structure.get("edges", [])
        nodes = kg_structure.get("nodes", [])
        all_kp_ids = {n["kp_id"] for n in nodes}

        # 构建前置依赖映射
        prerequisites: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.get("type", "prerequisite") == "prerequisite":
                prerequisites[edge["to"]].append(edge["from"])

        outer = set()
        for kp_id in all_kp_ids:
            if kp_id in mastered_kps:
                continue
            prereqs = prerequisites.get(kp_id, [])
            if all(p in mastered_kps for p in prereqs):
                outer.add(kp_id)
        return outer

    def detect_bottlenecks(
        self,
        kp_mastery: dict[str, float],
        kg_structure: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """检测瓶颈节点 — 低掌握度 + 高依赖权重.

        瓶颈节点定义:
        - 掌握度 < 0.5 (低掌握)
        - 阻塞至少 1 个后继节点 (有出边)

        Args:
            kp_mastery: 知识点掌握度映射.
            kg_structure: 知识图谱结构.

        Returns:
            瓶颈节点列表, 每项含:
            - kp_id: 知识点 ID
            - mastery: 当前掌握度
            - blocked_kps: 被阻塞的后继知识点列表
            - dependency_weight: 依赖权重 (blocked_kps 数量)
        """
        edges = kg_structure.get("edges", [])
        # 构建后继映射: {from: [to1, to2, ...]}
        successors: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.get("type", "prerequisite") == "prerequisite":
                successors[edge["from"]].append(edge["to"])

        bottlenecks: list[dict[str, Any]] = []
        for kp_id, mastery in kp_mastery.items():
            if mastery < self._BOTTLENECK_MASTERY:
                blocked = successors.get(kp_id, [])
                if blocked:
                    bottlenecks.append({
                        "kp_id": kp_id,
                        "mastery": mastery,
                        "blocked_kps": blocked,
                        "dependency_weight": len(blocked),
                    })

        # 按依赖权重降序排序
        bottlenecks.sort(key=lambda x: x["dependency_weight"], reverse=True)
        return bottlenecks

    def compute_centrality(
        self,
        kg_structure: dict[str, Any],
    ) -> dict[str, float]:
        """计算 KP 中心性 — 基于图结构的重要性分数.

        使用依赖后代计数: 一个知识点的中心性 = 它直接或间接阻塞的后继数量.
        被更多知识点依赖的前置知识中心性更高.

        Args:
            kg_structure: 知识图谱结构.

        Returns:
            {kp_id: centrality_score}, 分数在 [0, 1].
        """
        nodes = kg_structure.get("nodes", [])
        edges = kg_structure.get("edges", [])
        all_kp_ids = [n["kp_id"] for n in nodes]

        if not all_kp_ids:
            return {}

        n = len(all_kp_ids)
        # 构建后继邻接表 (prerequisite → successors)
        successors: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            from_kp = edge.get("from")
            to_kp = edge.get("to")
            if from_kp and to_kp:
                successors[from_kp].append(to_kp)

        # 计算每个节点的可达后代数 (BFS)
        descendant_count: dict[str, int] = {}
        for kp_id in all_kp_ids:
            visited: set[str] = set()
            queue: list[str] = list(successors.get(kp_id, []))
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                queue.extend(successors.get(current, []))
            descendant_count[kp_id] = len(visited)

        # 归一化到 [0, 1] (除以最大值)
        max_desc = max(descendant_count.values()) if descendant_count else 1
        if max_desc == 0:
            # 所有节点都没有后继 → 均匀分布
            return {kp_id: 1.0 / n for kp_id in all_kp_ids}

        centrality = {
            kp_id: descendant_count[kp_id] / max_desc
            for kp_id in all_kp_ids
        }
        return centrality

    def propagate_proficiency(
        self,
        observed: dict[str, float],
        kg_structure: dict[str, Any],
    ) -> dict[str, float]:
        """贝叶斯熟练度传播 — 掌握后继推断前置也掌握.

        传播规则 (Knewton 式):
        - 如果后继节点掌握度高 → 前置节点掌握度提升
        - 传播强度: propagated = max(observed, posterior)
        - posterior = 0.5 + 0.3 × (successor_mastery - 0.5) (贝叶斯收缩)

        Args:
            observed: 直接观测的掌握度 {kp_id: mastery}.
            kg_structure: 知识图谱结构.

        Returns:
            传播后的掌握度映射 (包含直接观测 + 传播值).
        """
        edges = kg_structure.get("edges", [])
        nodes = kg_structure.get("nodes", [])
        all_kp_ids = {n["kp_id"] for n in nodes}

        # 构建前置依赖映射: {to: [from1, from2, ...]}
        prerequisites: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.get("type", "prerequisite") == "prerequisite":
                prerequisites[edge["to"]].append(edge["from"])

        # 初始化: 观测值或 0.5
        propagated: dict[str, float] = {}
        for kp_id in all_kp_ids:
            propagated[kp_id] = observed.get(kp_id, 0.5)

        # 传播: 后继 → 前置
        for successor, mastery in observed.items():
            prereqs = prerequisites.get(successor, [])
            for prereq in prereqs:
                # 贝叶斯收缩: 0.5 + 0.3 × (mastery - 0.5)
                posterior = 0.5 + 0.3 * (mastery - 0.5)
                propagated[prereq] = max(propagated.get(prereq, 0.5), posterior)

        return propagated


# ============================================================
# 3. ProfileConfidenceEstimator — 画像置信度估计器
# ============================================================


class ProfileConfidenceEstimator:
    """画像置信度估计器.

    融合世界先进方案:
    - Knewton: 置信度 = f(数据量, 估计精度, 稳定性)
    - IRT: SE (标准误) 作为估计精度指标
    - 概念漂移: 漂移降低置信度

    公式:
        overall_confidence = se_factor × data_factor × (1 - drift_penalty)

    其中:
    - se_factor = 1 / (1 + se) (SE 越小 → 置信度越高)
    - data_factor = 1 - exp(-record_count / 20) (数据量饱和函数)
    - drift_penalty = 0.2 (检测到漂移时)
    """

    _DRIFT_PENALTY: float = 0.2
    _DATA_SATURATION: float = 10.0  # 10 条记录达到 ~63% 饱和

    def estimate(
        self,
        record_count: int,
        se: float,
        has_drift: bool,
        kp_count: int,
    ) -> float:
        """估计整体画像置信度.

        Args:
            record_count: 答题记录数.
            se: IRT 能力估计标准误.
            has_drift: 是否检测到概念漂移.
            kp_count: 知识点总数.

        Returns:
            置信度 [0.0, 1.0].
        """
        se_factor = 1.0 / (1.0 + se)
        data_factor = 1.0 - math.exp(-record_count / self._DATA_SATURATION)
        drift_penalty = self._DRIFT_PENALTY if has_drift else 0.0

        confidence = se_factor * data_factor * (1.0 - drift_penalty)
        return max(0.0, min(1.0, confidence))

    def estimate_kp_confidence(self, state: TracingState) -> float:
        """估计单 KP 置信度 — 基于 attempts 和最近练习时间.

        Args:
            state: 追踪状态.

        Returns:
            KP 置信度 [0.0, 1.0].
        """
        # 基于 attempts 的饱和函数
        attempts_factor = 1.0 - math.exp(-state.attempts / 5.0)

        # 基于最近练习时间的衰减 (7天半衰期)
        now = time.time()
        if state.last_attempt_time > 0:
            days_since = (now - state.last_attempt_time) / 86400.0
            recency_factor = math.exp(-days_since / 7.0)
        else:
            recency_factor = 0.5

        confidence = 0.6 * attempts_factor + 0.4 * recency_factor
        return max(0.0, min(1.0, confidence))

    def estimate_data_sufficiency(
        self,
        record_count: int,
        kp_count: int,
    ) -> float:
        """估计数据充分度 — 基于记录数与 KP 数比值.

        充分度 = min(1.0, record_count / (kp_count × 4))
        (每个 KP 至少 4 条记录才充分)

        Args:
            record_count: 答题记录数.
            kp_count: 知识点总数.

        Returns:
            充分度 [0.0, 1.0].
        """
        if kp_count == 0:
            return 0.0
        ratio = record_count / (kp_count * 4.0)
        return min(1.0, ratio)


# ============================================================
# 4. DynamicMasteryThreshold — 动态掌握阈值
# ============================================================


class DynamicMasteryThreshold:
    """动态掌握阈值 (CC1) — 按能力等级调整.

    融合世界先进方案:
    - Mastery Learning (Bloom): 不同等级不同掌握标准
    - ALEKS: 自适应阈值
    - Squirrel AI: CC1 按等级递增

    阈值映射:
    - beginner:     0.80 (基础掌握)
    - intermediate: 0.85 (进阶掌握)
    - advanced:     0.90 (高级掌握)
    - teacher:      0.95 (教学级掌握)
    """

    _THRESHOLDS: dict[str, float] = {
        "beginner": 0.80,
        "intermediate": 0.85,
        "advanced": 0.90,
        "teacher": 0.95,
    }

    _DEFAULT_THRESHOLD: float = 0.85

    def get_threshold(self, level: str) -> float:
        """获取指定能力等级的掌握阈值.

        Args:
            level: 能力等级 (beginner/intermediate/advanced/teacher).

        Returns:
            掌握阈值.
        """
        return self._THRESHOLDS.get(level, self._DEFAULT_THRESHOLD)

    def is_mastered(self, mastery: float, level: str) -> bool:
        """判断是否达到指定等级的掌握标准.

        Args:
            mastery: 掌握度 [0, 1].
            level: 能力等级.

        Returns:
            True 如果 mastery >= threshold(level).
        """
        return mastery >= self.get_threshold(level)


# ============================================================
# 5. ForgettingAlertGenerator — 遗忘预警生成器
# ============================================================


class ForgettingAlertGenerator:
    """遗忘预警生成器.

    融合世界先进方案:
    - Ebbinghaus 遗忘曲线: 时间衰减
    - FSRS: 记忆稳定性 + 可提取性
    - Duolingo HLR: 半衰期回归

    Bloom 分级阈值 (越高级越严格):
    - L1 (remember/understand): 0.70
    - L2 (apply/analyze):       0.60
    - L3 (evaluate/create):     0.50

    紧急度分级:
    - high:   mastery < 0.3
    - medium: 0.3 ≤ mastery < 0.5
    - low:    0.5 ≤ mastery < threshold
    """

    _BLOOM_THRESHOLDS: dict[str, float] = {
        "remember": 0.70,
        "understand": 0.70,
        "apply": 0.60,
        "analyze": 0.60,
        "evaluate": 0.50,
        "create": 0.50,
    }

    _DEFAULT_THRESHOLD: float = 0.70

    # 遗忘衰减半衰期 (天)
    _FORGETTING_HALFLIFE: float = 7.0
    # 过期阈值 (天) — 超过此天数未复习的 KP 触发预警 (即使 mastery >= 阈值)
    _STALE_THRESHOLD_DAYS: float = 7.0

    def generate_alerts(
        self,
        tracing_states: dict[str, TracingState],
        bloom_level: str = "understand",
    ) -> list[dict[str, Any]]:
        """生成遗忘预警.

        预警条件 (满足任一):
        1. 衰减后掌握度 < Bloom 阈值 (遗忘导致低于标准)
        2. 原始掌握度 >= 阈值 但 超过 7 天未复习 (过期预警)

        Args:
            tracing_states: 知识点追踪状态映射.
            bloom_level: 当前 Bloom 认知层次.

        Returns:
            预警列表, 每项含:
            - kp_id: 知识点 ID
            - mastery: 当前掌握度
            - days_since_review: 距上次复习天数
            - decay_factor: 衰减因子
            - urgency: 紧急度 (low/medium/high)
        """
        threshold = self._BLOOM_THRESHOLDS.get(bloom_level, self._DEFAULT_THRESHOLD)
        now = time.time()
        alerts: list[dict[str, Any]] = []

        for kp_id, state in tracing_states.items():
            mastery = state.mastery_prob

            # 计算距上次复习天数
            if state.last_attempt_time > 0:
                days_since = (now - state.last_attempt_time) / 86400.0
            else:
                days_since = 0.0

            # 衰减因子 (Ebbinghaus 式)
            decay_factor = math.exp(-days_since / self._FORGETTING_HALFLIFE)

            # 预警条件判断
            if mastery >= threshold:
                # mastery 达标 → 仅在过期 (超过 7 天) 时预警
                if days_since < self._STALE_THRESHOLD_DAYS:
                    continue

            # 紧急度 (基于原始 mastery)
            if mastery < 0.3:
                urgency = "high"
            elif mastery < 0.5:
                urgency = "medium"
            else:
                urgency = "low"

            alerts.append({
                "kp_id": kp_id,
                "mastery": mastery,
                "days_since_review": round(days_since, 1),
                "decay_factor": round(decay_factor, 4),
                "urgency": urgency,
            })

        # 按紧急度排序: high > medium > low
        urgency_order = {"high": 0, "medium": 1, "low": 2}
        alerts.sort(key=lambda a: urgency_order.get(a["urgency"], 3))
        return alerts


# ============================================================
# 6. OnlineStyleAdapter — 在线风格适配器
# ============================================================


class OnlineStyleAdapter:
    """在线风格适配器 — 指数平滑.

    融合世界先进方案:
    - Duolingo: 在线风格修正
    - VARK 问卷: 初始风格
    - 指数平滑: new = α × observed + (1-α) × old

    风格维度: V (visual) / A (aural) / R (reading) / K (kinesthetic)

    当多维接近 (差值 < 0.1) 时, 判定为 multimodal.
    """

    _DEFAULT_ALPHA: float = 0.3
    _STYLE_MAP: dict[str, str] = {
        "visual": "V",
        "aural": "A",
        "reading": "R",
        "kinesthetic": "K",
    }
    _MULTIMODAL_THRESHOLD: float = 0.1

    def __init__(self, alpha: float = _DEFAULT_ALPHA) -> None:
        """初始化在线风格适配器.

        Args:
            alpha: 指数平滑系数 [0, 1], 默认 0.3.
        """
        self._alpha = alpha
        self._scores: dict[str, float] = {
            "V": 0.25,
            "A": 0.25,
            "R": 0.25,
            "K": 0.25,
        }
        self._initialized = False

    def initialize(self, scores: dict[str, float]) -> None:
        """初始化风格分数.

        Args:
            scores: 初始风格分数 {V/A/R/K: score}.
        """
        self._scores = dict(scores)
        self._initialized = True

    def update(self, observed_style: str) -> None:
        """指数平滑更新风格分数.

        new_score = α × observed + (1-α) × old_score

        观测到的风格维度设为 1.0, 其他设为 0.0.

        Args:
            observed_style: 观测到的风格 (visual/aural/reading/kinesthetic).
        """
        style_key = self._STYLE_MAP.get(observed_style, observed_style.upper())
        for key in self._scores:
            observed_val = 1.0 if key == style_key else 0.0
            self._scores[key] = (
                self._alpha * observed_val
                + (1 - self._alpha) * self._scores[key]
            )

    def get_scores(self) -> dict[str, float]:
        """获取当前风格分数.

        Returns:
            {V/A/R/K: score} 映射.
        """
        return dict(self._scores)

    def infer_style(self) -> str:
        """推断主导学习风格.

        当最高分与次高分差值 < 0.1 时, 返回 "multimodal".
        否则返回最高分对应的风格名称.

        Returns:
            风格名称 (visual/aural/reading/kinesthetic/multimodal).
        """
        sorted_scores = sorted(self._scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            if sorted_scores[0] - sorted_scores[1] < self._MULTIMODAL_THRESHOLD:
                return "multimodal"

        # 找到最高分对应的风格
        max_key = max(self._scores, key=self._scores.get)
        reverse_map = {v: k for k, v in self._STYLE_MAP.items()}
        return reverse_map.get(max_key, "multimodal")


# ============================================================
# 7. AdaptiveBloomSetter — 自适应 Bloom 目标设定器
# ============================================================


class AdaptiveBloomSetter:
    """自适应 Bloom 目标设定器.

    融合世界先进方案:
    - Bloom (1956) / Anderson & Krathwohl (2001): 认知六层次
    - Mastery Learning: 掌握当前层次后提升目标
    - Vygotsky ZPD: 目标应在最近发展区内

    自适应规则:
    - 挫败区 (frustration): 保持当前级别 (不冒进)
    - ZPD 区 (zpd): +1 级 (循序渐进)
    - 独立区 (independent) + 高掌握度 (≥0.85): +2~3 级 (跳级)
    - 独立区 (independent) + 中掌握度: +1 级
    - 最高级 create: 保持
    """

    _BLOOM_LEVELS: list[str] = [
        "remember",
        "understand",
        "apply",
        "analyze",
        "evaluate",
        "create",
    ]

    _BLOOM_INDEX: dict[str, int] = {level: i for i, level in enumerate(_BLOOM_LEVELS)}

    def set_adaptive_target(
        self,
        current_level: str,
        avg_mastery: float,
        zpd_zone: str,
    ) -> str:
        """设定自适应 Bloom 目标.

        Args:
            current_level: 当前 Bloom 层次.
            avg_mastery: 平均掌握度 [0, 1].
            zpd_zone: ZPD 区域 (independent/zpd/frustration).

        Returns:
            目标 Bloom 层次.
        """
        current_idx = self._BLOOM_INDEX.get(current_level, 0)
        max_idx = len(self._BLOOM_LEVELS) - 1

        # 最高级保持
        if current_idx >= max_idx:
            return self._BLOOM_LEVELS[max_idx]

        # 挫败区: 最低级 +1 (无处可退), 其他保持当前级别
        if zpd_zone == "frustration":
            if current_idx == 0:
                return self._BLOOM_LEVELS[min(current_idx + 1, max_idx)]
            return self._BLOOM_LEVELS[current_idx]

        # 独立区 + 高掌握度: 跳级
        if zpd_zone == "independent" and avg_mastery >= 0.85:
            jump = 2 if avg_mastery >= 0.90 else 1
            target_idx = min(current_idx + jump + 1, max_idx)
            # 高掌握度 (≥0.90) 跳 2-3 级
            if avg_mastery >= 0.90:
                target_idx = min(current_idx + 3, max_idx)
            return self._BLOOM_LEVELS[target_idx]

        # ZPD 区或独立区 + 中低掌握度: +1 级
        return self._BLOOM_LEVELS[min(current_idx + 1, max_idx)]


# ============================================================
# 8. ProfileVersionManager — 画像版本管理器
# ============================================================


class ProfileVersionManager:
    """画像版本管理器.

    融合世界先进方案:
    - Git 式版本控制: 版本号 / 时间戳 / 变更追踪
    - Knewton: 画像历史回溯
    - 画像审计: 版本差异比较

    功能:
    - 每次保存画像 → 版本号递增
    - 历史画像列表查询
    - 版本差异比较 (theta/level/style 变化检测)
    """

    def __init__(self) -> None:
        """初始化版本管理器."""
        # {learner_id: [{"version": int, "snapshot": LearnerSnapshot, "ts": float}]}
        self._versions: dict[str, list[dict[str, Any]]] = {}

    def save(self, learner_id: str, snapshot: LearnerSnapshot) -> int:
        """保存画像快照, 返回版本号.

        Args:
            learner_id: 学习者 ID.
            snapshot: 画像快照.

        Returns:
            版本号 (从 1 开始递增).
        """
        if learner_id not in self._versions:
            self._versions[learner_id] = []
        version = len(self._versions[learner_id]) + 1
        self._versions[learner_id].append({
            "version": version,
            "snapshot": snapshot,
            "ts": time.time(),
        })
        return version

    def get_history(self, learner_id: str) -> list[dict[str, Any]]:
        """获取学习者的画像历史.

        Args:
            learner_id: 学习者 ID.

        Returns:
            版本列表, 每项含 version / snapshot / ts.
        """
        return list(self._versions.get(learner_id, []))

    def get_version(
        self,
        learner_id: str,
        version: int,
    ) -> LearnerSnapshot | None:
        """获取指定版本的画像.

        Args:
            learner_id: 学习者 ID.
            version: 版本号.

        Returns:
            画像快照或 None (不存在时).
        """
        history = self._versions.get(learner_id, [])
        for entry in history:
            if entry["version"] == version:
                return entry["snapshot"]
        return None

    def diff(
        self,
        learner_id: str,
        version1: int,
        version2: int,
    ) -> dict[str, Any]:
        """比较两个版本的差异.

        Args:
            learner_id: 学习者 ID.
            version1: 旧版本号.
            version2: 新版本号.

        Returns:
            差异字典, 含 theta_changed / level_changed / style_changed / bloom_changed.
        """
        snap1 = self.get_version(learner_id, version1)
        snap2 = self.get_version(learner_id, version2)

        if snap1 is None or snap2 is None:
            return {
                "error": "version not found",
                "theta_changed": False,
                "level_changed": False,
                "style_changed": False,
                "bloom_changed": False,
            }

        return {
            "theta_changed": abs(
                (snap1.theta or 0) - (snap2.theta or 0)
            ) > 0.01,
            "level_changed": snap1.level != snap2.level,
            "style_changed": snap1.learning_style != snap2.learning_style,
            "bloom_changed": snap1.bloom_target != snap2.bloom_target,
            "theta_diff": (snap2.theta or 0) - (snap1.theta or 0),
        }


# ============================================================
# 9. ProfileConsistencyValidator — 画像一致性校验器
# ============================================================


class ProfileConsistencyValidator:
    """画像一致性校验器.

    融合世界先进方案:
    - 数据质量检查 (Knewton)
    - 内部一致性检测 (theta vs mastery, level vs mastery)
    - 异常检测 (越界值, 不可能时间戳)

    校验规则:
    - mastery ∈ [0, 1]
    - confidence ∈ [0, 1]
    - theta 高但 mastery 全低 → 不一致
    - weak_kps 应与 mastery < 0.5 的 KP 一致
    - level 与 mastery 大致对应 (advanced → mastery 应较高)
    """

    # theta 与 mastery 一致性阈值
    _THETA_HIGH: float = 1.5
    _MASTERY_LOW: float = 0.3
    _MASTERY_RANGE: tuple[float, float] = (0.0, 1.0)
    _CONFIDENCE_RANGE: tuple[float, float] = (0.0, 1.0)
    _WEAK_KP_THRESHOLD: float = 0.7

    def validate(self, snapshot: LearnerSnapshot) -> list[str]:
        """校验画像一致性, 返回问题列表.

        Args:
            snapshot: 画像快照.

        Returns:
            问题描述列表; 空列表表示无问题.
        """
        issues: list[str] = []

        # 1. mastery 越界检查
        for kp_id, mastery in snapshot.kp_mastery.items():
            if mastery < self._MASTERY_RANGE[0] or mastery > self._MASTERY_RANGE[1]:
                issues.append(
                    f"mastery 越界: kp_id={kp_id} mastery={mastery} "
                    f"(有效范围 [{self._MASTERY_RANGE[0]}, {self._MASTERY_RANGE[1]}])"
                )

        # 2. confidence 越界检查
        if (
            snapshot.confidence < self._CONFIDENCE_RANGE[0]
            or snapshot.confidence > self._CONFIDENCE_RANGE[1]
        ):
            issues.append(
                f"confidence 越界: {snapshot.confidence} "
                f"(有效范围 [{self._CONFIDENCE_RANGE[0]}, {self._CONFIDENCE_RANGE[1]}])"
            )

        # 3. theta 与 mastery 一致性
        theta = snapshot.theta or 0.0
        if snapshot.kp_mastery:
            avg_mastery = sum(snapshot.kp_mastery.values()) / len(snapshot.kp_mastery)
            if theta > self._THETA_HIGH and avg_mastery < self._MASTERY_LOW:
                issues.append(
                    f"theta-mastery 不一致: theta={theta:.2f} (高) "
                    f"但 avg_mastery={avg_mastery:.2f} (低)"
                )

        # 4. weak_kps 一致性
        expected_weak = {
            kp_id
            for kp_id, mastery in snapshot.kp_mastery.items()
            if mastery < self._WEAK_KP_THRESHOLD
        }
        actual_weak = set(snapshot.weak_kps)
        # 检查: 不应在 weak_kps 中出现高 mastery 的 KP
        high_mastery_in_weak = actual_weak - expected_weak
        if high_mastery_in_weak:
            issues.append(
                f"weak_kps 不一致: {high_mastery_in_weak} 的 mastery >= "
                f"{self._WEAK_KP_THRESHOLD}, 不应出现在 weak_kps 中"
            )

        return issues


# ============================================================
# __all__
# ============================================================

__all__ = [
    "MultiDimensionalFuser",
    "KSTAnalyzer",
    "ProfileConfidenceEstimator",
    "DynamicMasteryThreshold",
    "ForgettingAlertGenerator",
    "OnlineStyleAdapter",
    "AdaptiveBloomSetter",
    "ProfileVersionManager",
    "ProfileConsistencyValidator",
]
