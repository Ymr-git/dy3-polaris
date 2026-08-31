"""CAT 计算机化自适应测试选题引擎.

融合世界先进方案:
- catR R 包: 最大 Fisher 信息准则选题 + KL 信息准则 + b-match 选题
- ALEKS: 知识空间理论 + 最大信息量选题 (P≈0.5)
- Duolingo: IRT + DKT 混合自适应
- Kingsbury & Zara (1991): randomesque 选题 (曝光控制)
- CCAT (Constrained CAT): 内容平衡约束选题

选题策略 (selection_strategy):
- "fisher_info": 最大 Fisher 信息准则 (默认, 选 argmax I(theta))
- "b_match": 难度匹配 (选 b 最接近当前 theta 的题目)
- "kl_info": Kullback-Leibler 信息准则 (区分 theta 与 theta+delta)
- "randomesque": 从 Fisher 信息 top-N 中随机选一 (曝光控制)
- "zpd_aware": ZPD 感知选题 (过滤挫败区, 优先 ZPD 中心, Fisher 信息选优,
  无 ZPD 候选时回退 b_match)

内容平衡 (content_constraints):
- 可选传入 {content_area: target_proportion} 约束
- 已超出目标比例的内容域被抑制, 优先选欠配额域; 全部饱和时回退全局最优

终止条件 (should_stop):
- SE < se_threshold 且题数 >= min_items (能力估计精度达标)
- 或 题数 >= max_items (题量上限)

曝光追踪 (get_exposure_stats):
- 记录每题被选中的次数, 用于评估题目曝光均匀性

CATSelector 委托底层 IRTEstimator 计算 Fisher 信息量与 MLE, 保证选题与
能力估计使用同一套 IRT 参数语义.
"""

from __future__ import annotations

import math
import random
import threading
from typing import TYPE_CHECKING, Any, Iterable

from dy3_polaris.l2.models import IRTState
from dy3_polaris.l2.ability_assessor.irt import IRTEstimator

if TYPE_CHECKING:
    from dy3_polaris.l2.ability_assessor.zpd import ZPDCalculator


# ============================================================
# 1. 常量定义
# ============================================================

# 默认终止条件
_DEFAULT_MAX_ITEMS: int = 20
_DEFAULT_SE_THRESHOLD: float = 0.3

# 默认选题策略
_DEFAULT_STRATEGY: str = "fisher_info"

# randomesque 默认 top-N 窗口大小
_DEFAULT_RANDOMESQUE_N: int = 5

# KL 信息准则默认 theta 偏移量 delta
_DEFAULT_KL_DELTA: float = 1.0

# 数值稳定性下限 (避免 log(0) / 除零)
_EPS: float = 1e-10

# 内容比例饱和判定容差
_CONTENT_EPS: float = 1e-9

# 支持的选题策略集合
_VALID_STRATEGIES: frozenset[str] = frozenset(
    {"fisher_info", "b_match", "kl_info", "randomesque", "zpd_aware", "progressive", "a_stratified", "bkt_irt_fusion"}
)

# BKT+IRT 融合选题的 ZPD 掌握度边界
_MASTERY_ZPD_LOWER: float = 0.3
_MASTERY_ZPD_UPPER: float = 0.7


# ============================================================
# 2. CATSelector 自适应选题引擎
# ============================================================


class CATSelector:
    """CAT 自适应测试选题引擎.

    四个核心能力:

    1. ``select_next``: 多策略自适应选题.
       支持 fisher_info / b_match / kl_info / randomesque 四种策略,
       可叠加内容平衡约束 (content_constraints), 并自动追踪题目曝光.
    2. ``should_stop``: 终止条件判定.
       SE < se_threshold 且题数 >= min_items, 或 题数 >= max_items.
    3. ``estimate_ability``: 从作答序列估计能力.
       委托 IRTEstimator.estimate_mle 进行批量 MLE 估计.
    4. ``get_exposure_stats``: 曝光统计.
       返回各题被选中次数的副本.
    5. ``reset_stats``: 清除曝光统计与内容域计数 (跨会话重置).

    线程安全: 曝光统计 (_exposure) 与内容计数 (_content_counts) 的读-改-写
    均由 _lock (threading.RLock) 保护, 支持并发选题.

    Args:
        estimator: 注入的 IRT 估计器 (用于计算信息量与 MLE).
            默认创建独立 IRTEstimator 实例; IRTEstimator 本身无状态,
            注入同一实例在多 CATSelector 间共享是安全的.
        selection_strategy: 选题策略, 默认 "fisher_info".
            可选 "fisher_info" / "b_match" / "kl_info" / "randomesque".
        content_constraints: 内容平衡约束 {content_area: target_proportion},
            默认 None (不约束). 启用后已超出目标比例的内容域被抑制.
        randomesque_n: randomesque 策略的 top-N 窗口大小, 默认 5.
        kl_delta: kl_info 策略的 theta 偏移量, 默认 1.0.
        rng: 随机数生成器 (random.Random), 用于 randomesque; 默认新建.

    状态: 曝光统计 (_exposure) 与内容计数 (_content_counts) 在选题过程中
    累积; 学习者能力状态 (theta / se / 已答题集) 仍由调用方持有并传入.
    """

    def __init__(
        self,
        estimator: IRTEstimator | None = None,
        selection_strategy: str = _DEFAULT_STRATEGY,
        content_constraints: dict[str, float] | None = None,
        randomesque_n: int = _DEFAULT_RANDOMESQUE_N,
        kl_delta: float = _DEFAULT_KL_DELTA,
        rng: random.Random | None = None,
        zpd_calculator: ZPDCalculator | None = None,
        max_items: int = _DEFAULT_MAX_ITEMS,
        se_threshold: float = _DEFAULT_SE_THRESHOLD,
        max_exposure_rate: float | None = None,
        fusion_weight: float = 0.5,
    ) -> None:
        """初始化 CAT 选题引擎.

        Args:
            estimator: IRT 估计器实例; 为 None 时创建默认 IRTEstimator.
            selection_strategy: 选题策略名称.
            content_constraints: 内容平衡约束字典 (可选).
            randomesque_n: randomesque top-N 窗口大小.
            kl_delta: kl_info 的 theta 偏移量.
            rng: 随机数生成器; 为 None 时新建 random.Random().
            zpd_calculator: ZPD 计算器实例 (可选), 用于 zpd_aware 选题、
                支架感知选题 (select_with_scaffold) 与 ZPD 终止条件
                (should_stop_zpd). 为 None 时 zpd_aware / select_with_scaffold
                回退到基于 IRT 正确率阈值的默认 ZPD 边界.
            max_items: 最大题目数, 默认 20.
            se_threshold: SE 终止阈值, 默认 0.3.
            max_exposure_rate: 最大曝光率, 默认 None (不限制).
            fusion_weight: BKT+IRT 融合权重 w ∈ [0, 1], 默认 0.5.
                w=0 → 纯 IRT Fisher 信息选题; w=1 → 纯 BKT 掌握度选题.
                仅用于 bkt_irt_fusion 策略.
        """
        # IRTEstimator 无状态, 持有引用不引入可变状态
        self._estimator: IRTEstimator = (
            estimator if estimator is not None else IRTEstimator()
        )
        # 选题策略 (公开属性, 便于检视)
        self.selection_strategy: str = selection_strategy
        # 内容平衡约束 (公开属性, 存副本避免外部篡改)
        self.content_constraints: dict[str, float] | None = (
            dict(content_constraints) if content_constraints is not None else None
        )
        # randomesque / kl_info 参数
        self._randomesque_n: int = randomesque_n
        self._kl_delta: float = kl_delta
        # 随机数生成器
        self._rng: random.Random = rng if rng is not None else random.Random()
        # ZPD 计算器 (可选, 用于 zpd_aware / 支架感知 / ZPD 终止)
        self._zpd_calculator: ZPDCalculator | None = zpd_calculator
        self._max_items: int = max_items
        self._se_threshold: float = se_threshold
        self._max_exposure_rate: float | None = max_exposure_rate
        # BKT+IRT 融合权重 (钳制到 [0, 1])
        self._fusion_weight: float = max(0.0, min(1.0, fusion_weight))
        self._item_bank: list[dict[str, Any]] = []
        # 曝光追踪: {item_id: 被选中次数}
        self._exposure: dict[str, int] = {}
        # 内容域计数: {content_area: 已选次数} (仅记录受约束的内容域)
        self._content_counts: dict[str, int] = {}
        # 保护 _exposure / _content_counts 的读-改-写 (线程安全曝光统计)
        self._lock: threading.RLock = threading.RLock()

    # --- 多策略自适应选题 ---

    def select_next(
        self,
        theta: float,
        available_items: list[dict[str, Any]],
        administered_ids: Iterable[str],
    ) -> dict[str, Any] | None:
        """多策略自适应选题 — 依据 selection_strategy 选下一题.

        流程:
        1. 排除已答题目 (administered_ids) 得到候选池.
        2. 若启用 content_constraints, 按内容平衡过滤候选池
           (抑制已超出目标比例的内容域; 全部饱和时回退全局候选).
        3. 依 selection_strategy 在 (过滤后) 候选池中选题.
        4. 记录曝光与内容域计数 (选题成功时).

        Args:
            theta: 当前能力估计 θ.
            available_items: 可用题目列表, 每项为
                {"item_id": ..., "a": ..., "b": ..., "c": ...,
                 "content_area": ... (可选)} 字典.
            administered_ids: 已答题目 ID 集合 (set / list 均可, 内部转集合).

        Returns:
            选中的题目字典 (原 dict 引用); 无可用题目返回 None.

        Raises:
            ValueError: selection_strategy 不在支持列表中时抛出.
        """
        # 归一化为集合, 加速成员判定 (兼容 list / set 输入)
        administered_set: set[str] = set(administered_ids)

        # 1. 构建候选池 (排除已答题目)
        candidates: list[dict[str, Any]] = [
            item
            for item in available_items
            if item.get("item_id") not in administered_set
        ]

        # 2. 内容平衡过滤
        pool = self._content_filter(candidates)

        # 2.5 曝光率过滤 (排除已达曝光上限的题目)
        if self._max_exposure_rate is not None and pool:
            with self._lock:
                total_sel = sum(self._exposure.values())
            if total_sel > 0:
                unexposed = [
                    item for item in pool
                    if not self._is_overexposed(item.get("item_id", ""), total_sel)
                ]
                if unexposed:
                    pool = unexposed

        # 3. 策略分派
        strategy = self.selection_strategy
        if strategy == "fisher_info":
            chosen = self._select_fisher_info(theta, pool)
        elif strategy == "b_match":
            chosen = self._select_b_match(theta, pool)
        elif strategy == "kl_info":
            chosen = self._select_kl_info(theta, pool)
        elif strategy == "randomesque":
            chosen = self._select_randomesque(theta, pool)
        elif strategy == "zpd_aware":
            chosen = self._select_zpd_aware(theta, pool)
        elif strategy == "progressive":
            chosen = self._select_progressive(theta, pool, len(administered_set))
        elif strategy == "a_stratified":
            chosen = self._select_a_stratified(theta, pool, len(administered_set))
        elif strategy == "bkt_irt_fusion":
            chosen = self._select_bkt_irt_fusion(theta, pool)
        else:
            raise ValueError(
                f"未知选题策略: {strategy!r}, 支持: "
                f"{sorted(_VALID_STRATEGIES)}"
            )

        # 4. 记录曝光与内容域计数
        if chosen is not None:
            self._record_selection(chosen)

        return chosen

    # --- 选题策略实现 ---

    def _select_fisher_info(
        self,
        theta: float,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """最大 Fisher 信息准则: 选 argmax I(theta) 的题目."""
        if not candidates:
            return None
        best_item: dict[str, Any] | None = None
        best_info: float = -1.0  # 信息量非负, -1 确保首个有效题目被选中
        for item in candidates:
            a, b, c = self._item_abc(item)
            info = self._estimator.information(theta, a, b, c)
            if info > best_info:
                best_info = info
                best_item = item
        return best_item

    def _select_b_match(
        self,
        theta: float,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """难度匹配: 选难度 b 最接近当前 theta 的题目 (|b - theta| 最小)."""
        if not candidates:
            return None
        best_item: dict[str, Any] | None = None
        best_dist: float = float("inf")
        for item in candidates:
            b = float(item.get("b", 0.0))
            dist = abs(b - theta)
            if dist < best_dist:
                best_dist = dist
                best_item = item
        return best_item

    def _select_kl_info(
        self,
        theta: float,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Kullback-Leibler 信息准则: 选区分 theta 与 theta+delta 最强的题目.

        KL_i = P(theta)*ln(P(theta)/P(theta+delta))
               + (1-P(theta))*ln((1-P(theta))/(1-P(theta+delta)))

        delta 越大, 越关注远端区分能力; 默认 delta=1.0 (catR 风格).
        """
        if not candidates:
            return None
        delta = self._kl_delta
        best_item: dict[str, Any] | None = None
        best_kl: float = -1.0
        for item in candidates:
            a, b, c = self._item_abc(item)
            kl = self._kl_information(theta, a, b, c, delta)
            if kl > best_kl:
                best_kl = kl
                best_item = item
        return best_item

    def _select_randomesque(
        self,
        theta: float,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """randomesque 曝光控制: 从 Fisher 信息 top-N 中随机选一题.

        先按 Fisher 信息降序排序, 取前 N 题 (N=randomesque_n, 不超过候选数),
        再从中随机选一题, 避免高区分度题被过度曝光.
        """
        if not candidates:
            return None
        # 按 Fisher 信息降序排序
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in candidates:
            a, b, c = self._item_abc(item)
            info = self._estimator.information(theta, a, b, c)
            scored.append((info, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        # 取 top-N (不超过候选总数)
        n = max(1, min(self._randomesque_n, len(scored)))
        top_n: list[dict[str, Any]] = [item for _, item in scored[:n]]
        return self._rng.choice(top_n)

    def _select_zpd_aware(
        self,
        theta: float,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """ZPD 感知选题: 过滤挫败区, 优先 ZPD 中心, Fisher 信息选优.

        流程:
        1. 依据当前 ZPD 阈值 (frustration_p / independent_p) 计算每题
           P(correct), 过滤掉挫败区题目 (P <= frustration_p).
        2. 收集 ZPD 中心区域题目 (frustration_p < P < independent_p).
        3. 若存在 ZPD 候选, 在其中用最大 Fisher 信息准则选最优题.
        4. 若无 ZPD 候选 (剩余均为独立区), 回退到 b_match 策略
           (在非挫败区候选上选 b 最接近 theta 的题); 若全部候选均在挫败区,
           回退到全部候选的 b_match.

        Args:
            theta: 当前能力估计 θ.
            candidates: 候选题目列表 (已排除已答题与内容平衡过滤).

        Returns:
            选中的题目字典; 无候选返回 None.
        """
        if not candidates:
            return None
        independent_p, frustration_p = self._zpd_thresholds()
        non_frustration: list[dict[str, Any]] = []
        zpd_candidates: list[dict[str, Any]] = []
        for item in candidates:
            a, b, c = self._item_abc(item)
            p = self._estimator.predict_correct(theta, a, b, c)
            if p <= frustration_p:
                continue  # 过滤挫败区
            non_frustration.append(item)
            if frustration_p < p < independent_p:
                zpd_candidates.append(item)
        if zpd_candidates:
            # ZPD 候选用 Fisher 信息准则选最优
            return self._select_fisher_info(theta, zpd_candidates)
        # 无 ZPD 候选: 回退 b_match (优先非挫败区, 否则全部候选)
        pool = non_frustration if non_frustration else candidates
        return self._select_b_match(theta, pool)

    # --- 支架感知选题 ---

    def select_with_scaffold(
        self,
        theta: float,
        available_items: list[dict[str, Any]],
        administered_ids: Iterable[str],
        scaffold_level: float = 0.5,
    ) -> dict[str, Any] | None:
        """支架感知选题 — 按支架水平在 ZPD 区间内选取目标难度题目.

        目标难度计算:
            target_b = zpd_lower + scaffold_level * (zpd_upper - zpd_lower)

        - scaffold_level=0.0: target_b = zpd_lower (独立区, 简单题, 学习者可独立完成).
        - scaffold_level=0.5: target_b = ZPD 中心 (适度挑战, 需要支架).
        - scaffold_level=1.0: target_b = zpd_upper (ZPD 上界, 最大挑战).

        在排除已答题后的候选中, 选取难度 b 最接近 target_b 的题目.
        ZPD 边界由 zpd_calculator.calculate_zpd(theta, available_items) 给出;
        若未注入 zpd_calculator, 回退到 target_b = theta (b_match 风格).

        Args:
            theta: 当前能力估计 θ.
            available_items: 可用题目列表.
            administered_ids: 已答题目 ID 集合.
            scaffold_level: 支架水平 ∈ [0, 1], 默认 0.5.

        Returns:
            选中的题目字典; 无可用题目返回 None.
        """
        # 计算 ZPD 目标难度
        if self._zpd_calculator is not None:
            zpd_items = self._to_zpd_items(available_items)
            zpd = self._zpd_calculator.calculate_zpd(theta, zpd_items)
            zpd_span = zpd.zpd_upper - zpd.zpd_lower
            target_b = zpd.zpd_lower + scaffold_level * zpd_span
        else:
            # 无 ZPD 计算器: 回退到 theta 为目标难度
            target_b = theta

        # 排除已答题目
        administered_set: set[str] = set(administered_ids)
        candidates: list[dict[str, Any]] = [
            item
            for item in available_items
            if item.get("item_id") not in administered_set
        ]
        if not candidates:
            return None

        # 选 b 最接近 target_b 的题目
        best_item: dict[str, Any] | None = None
        best_dist: float = float("inf")
        for item in candidates:
            b = float(item.get("b", 0.0))
            dist = abs(b - target_b)
            if dist < best_dist:
                best_dist = dist
                best_item = item

        if best_item is not None:
            self._record_selection(best_item)
        return best_item

    # --- ZPD 终止条件 ---

    def should_stop_zpd(
        self,
        current_se: float,
        count: int,
        theta: float,
        administered_items: list[dict[str, Any]],
        max_items: int = _DEFAULT_MAX_ITEMS,
        se_threshold: float = _DEFAULT_SE_THRESHOLD,
        min_items: int = 5,
    ) -> bool:
        """ZPD 增强终止条件 — 标准 CAT 终止 + ZPD 覆盖检查.

        满足以下任一条件即终止:
        - 题量达到上限: count >= max_items
        - 能力估计精度达标且达最低题量, 且 ZPD 三区 (独立/ZPD/挫败) 均已覆盖:
          current_se < se_threshold 且 count >= min_items 且
          _zpd_coverage_check(theta, administered_items) 为真

        ZPD 覆盖检查确保测试不仅精度达标, 还在学习者的独立区、ZPD 区、挫败区
        均有取样, 提供完整的能力画像.

        Args:
            current_se: 当前能力估计标准误.
            count: 已作答题数.
            theta: 当前能力估计 θ.
            administered_items: 已施测题目列表 (含 a/b/c 参数).
            max_items: 题量上限, 默认 20.
            se_threshold: SE 终止阈值, 默认 0.3.
            min_items: 允许 SE 终止的最低题数, 默认 5.

        Returns:
            True 表示应终止测试, False 表示继续选题.
        """
        # 标准 CAT 终止: 题量达上限
        if count >= max_items:
            return True
        # SE 达标 + 最低题量 + ZPD 三区覆盖
        if (
            current_se < se_threshold
            and count >= min_items
            and self._zpd_coverage_check(theta, administered_items)
        ):
            return True
        return False

    # --- 内容平衡 ---

    def _content_filter(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """按内容平衡约束过滤候选池.

        已超出目标比例 (proportion >= target) 的内容域被视为饱和, 其题目被
        抑制. 若过滤后候选池为空 (所有候选均在饱和域), 回退到原候选池
        (无法满足约束时仍要选题).

        无 content_constraints 或尚无受约束题目被选时, 不过滤.
        """
        if not self.content_constraints:
            return candidates
        # 在锁内快照内容域计数, 避免与并发 _record_selection 交错
        with self._lock:
            total = sum(self._content_counts.values())
            counts_snapshot = dict(self._content_counts)
        if total <= 0:
            # 尚未选过任何受约束题目, 不施加限制
            return candidates
        # 计算饱和内容域集合
        saturated: set[str] = set()
        for area, target in self.content_constraints.items():
            count = counts_snapshot.get(area, 0)
            if count / total >= target - _CONTENT_EPS:
                saturated.add(area)
        if not saturated:
            return candidates
        # 保留不在饱和域的候选 (无 content_area / 非约束域 / 未饱和约束域)
        filtered = [
            item for item in candidates if item.get("content_area") not in saturated
        ]
        return filtered if filtered else candidates

    # --- 曝光与内容域记录 ---

    def _is_overexposed(self, item_id: str, total_selections: int) -> bool:
        """检查题目是否超过曝光率上限.

        曝光率 = 该题被选次数 / 总选题次数.
        若 max_exposure_rate 不为 None 且当前曝光率已达上限, 返回 True.
        """
        if self._max_exposure_rate is None or total_selections == 0:
            return False
        count = self._exposure.get(item_id, 0)
        rate = count / total_selections
        return rate >= self._max_exposure_rate

    def _record_selection(self, item: dict[str, Any]) -> None:
        """记录选题: 累加曝光计数与 (受约束的) 内容域计数.

        线程安全: 使用 _lock 保护 _exposure / _content_counts 的读-改-写,
        避免并发选题丢失曝光计数.
        """
        item_id = item.get("item_id")
        with self._lock:
            if item_id is not None:
                self._exposure[item_id] = self._exposure.get(item_id, 0) + 1
            if self.content_constraints:
                area = item.get("content_area")
                if area is not None and area in self.content_constraints:
                    self._content_counts[area] = (
                        self._content_counts.get(area, 0) + 1
                    )

    def get_exposure_stats(self) -> dict[str, int]:
        """返回题目曝光统计副本 {item_id: 被选中次数}.

        线程安全: 在 _lock 下拷贝, 避免与并发选题交错时读到不一致状态.
        """
        with self._lock:
            return dict(self._exposure)

    def reset_stats(self) -> None:
        """清除所有曝光统计与内容域计数.

        重置后, 曝光追踪与内容平衡计数回到初始状态. 适用于跨学习者/跨会话
        重新开始自适应测试的场景.
        """
        with self._lock:
            self._exposure.clear()
            self._content_counts.clear()

    def set_item_bank(self, items: list[dict[str, Any]]) -> None:
        """设置题库 (供 a_stratified 曝光控制使用)."""
        self._item_bank = [dict(item) for item in items]

    # --- 终止条件判定 ---

    def should_stop(
        self,
        current_se: float,
        count: int,
        max_items: int = _DEFAULT_MAX_ITEMS,
        se_threshold: float = _DEFAULT_SE_THRESHOLD,
        min_items: int = 0,
    ) -> bool:
        """CAT 终止条件判定.

        满足以下任一条件即终止:
        - 题量达到上限: count >= max_items
        - 能力估计精度达标且达最低题量: current_se < se_threshold 且 count >= min_items

        Args:
            current_se: 当前能力估计标准误.
            count: 已作答题数.
            max_items: 题量上限, 默认 20.
            se_threshold: SE 终止阈值, 默认 0.3.
            min_items: 允许 SE 终止的最低题数, 默认 0 (不限).

        Returns:
            True 表示应终止测试, False 表示继续选题.
        """
        # 题量达上限 (不受 min_items 影响)
        if count >= max_items:
            return True
        # SE 达标且满足最低题量 (严格小于, 与 catR 标准一致)
        if current_se < se_threshold and count >= min_items:
            return True
        return False

    # --- 从作答序列估计能力 ---

    def estimate_ability(
        self,
        responses: list[tuple[dict[str, Any], bool]],
    ) -> IRTState:
        """从作答序列估计能力 — 委托 IRTEstimator.estimate_mle.

        对作答序列执行批量最大似然估计, 返回 IRTState.
        空序列回退: theta=0.0, se=1.0.

        Args:
            responses: (item_params, correct) 列表, item_params 为
                {"a", "b", "c"} 字典.

        Returns:
            MLE 估计的 IRTState (theta / se / response_count / last_update_time).
        """
        return self._estimator.estimate_mle(responses)

    # --- 内部辅助 ---

    @staticmethod
    def _item_abc(item: dict[str, Any]) -> tuple[float, float, float]:
        """从题目字典提取 (a, b, c), 缺失回退 a=1.0, b=0.0, c=0.0.

        注: CAT 选题用宽松提取 (不校验), 与 IRT 引擎的 _extract_params 校验解耦,
        保证选题在题目参数不完整时不中断 (回退默认值).
        """
        a = float(item.get("a", 1.0))
        b = float(item.get("b", 0.0))
        c = float(item.get("c", 0.0))
        return a, b, c

    def _kl_information(
        self,
        theta: float,
        a: float,
        b: float,
        c: float,
        delta: float,
    ) -> float:
        """Kullback-Leibler 信息量: 区分 theta 与 theta+delta 的能力.

        KL = P*ln(P/Q) + (1-P)*ln((1-P)/(1-Q))
        其中 P = P(theta), Q = P(theta+delta).
        """
        p = self._estimator.predict_correct(theta, a, b, c)
        q = self._estimator.predict_correct(theta + delta, a, b, c)
        p = max(_EPS, min(1.0 - _EPS, p))
        q = max(_EPS, min(1.0 - _EPS, q))
        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

    def _zpd_thresholds(self) -> tuple[float, float]:
        """返回当前 ZPD 阈值 (independent_p, frustration_p).

        若注入了 zpd_calculator, 使用其阈值 (支持自定义); 否则回退到默认
        (independent_p=0.9, frustration_p=0.3, 与 ZPDCalculator 默认一致).

        Returns:
            (independent_p, frustration_p) — 独立区下界与挫败区上界.
        """
        if self._zpd_calculator is not None:
            return self._zpd_calculator.independent_p, self._zpd_calculator.frustration_p
        return 0.9, 0.3

    @staticmethod
    def _to_zpd_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将 CAT 题目字典转换为 ZPDCalculator 所需格式.

        CAT 题目使用 {"a", "b", "c"} 键, ZPDCalculator.calculate_zpd 需要
        {"difficulty_b", "discrimination_a", "guessing_c"} 键. 本方法做键名
        映射, 缺失值回退到与 _item_abc 一致的默认值.

        Args:
            items: CAT 题目列表.

        Returns:
            ZPD 格式题目列表 (新列表, 不修改原字典).
        """
        converted: list[dict[str, Any]] = []
        for item in items:
            converted.append(
                {
                    "item_id": item.get("item_id"),
                    "difficulty_b": float(item.get("b", 0.0)),
                    "discrimination_a": float(item.get("a", 1.0)),
                    "guessing_c": float(item.get("c", 0.0)),
                }
            )
        return converted

    def _zpd_coverage_check(
        self,
        theta: float,
        administered_items: list[dict[str, Any]],
    ) -> bool:
        """ZPD 三区覆盖检查 — 判断已施测题目是否覆盖独立/ZPD/挫败三区.

        依据当前 ZPD 阈值, 对每道已施测题目计算 P(correct) 并分类:
        - 独立区: P > independent_p
        - ZPD 区: frustration_p < P <= independent_p
        - 挫败区: P <= frustration_p

        Args:
            theta: 当前能力估计 θ.
            administered_items: 已施测题目列表 (含 a/b/c 参数).

        Returns:
            三区均已覆盖返回 True, 否则 False (无题目也返回 False).
        """
        if not administered_items:
            return False
        independent_p, frustration_p = self._zpd_thresholds()
        has_independent = False
        has_zpd = False
        has_frustration = False
        for item in administered_items:
            a, b, c = self._item_abc(item)
            p = self._estimator.predict_correct(theta, a, b, c)
            if p > independent_p:
                has_independent = True
            elif p > frustration_p:
                has_zpd = True
            else:
                has_frustration = True
        return has_independent and has_zpd and has_frustration

    # --- 渐进法选题 (Progressive) ---

    def compute_progressive_weight(
        self,
        current_se: float,
        items_administered: int,
    ) -> float:
        """渐进法权重 W — 随测试进度从 0→1 递增.

        W = max(I(theta)/I_stop, q/(M-1))^t

        其中 I(theta)=1/SE², I_stop=1/se_threshold², q=已答题数, M=max_items.
        """
        i_current = 1.0 / max(current_se ** 2, _EPS)
        i_stop = 1.0 / max(self._se_threshold ** 2, _EPS)
        q = items_administered
        M = max(self._max_items, 2)
        ratio_info = min(i_current / i_stop, 1.0)
        ratio_progress = min(q / (M - 1), 1.0)
        w = max(ratio_info, ratio_progress)
        return min(max(w, 0.0), 1.0)

    def _select_progressive(
        self,
        theta: float,
        candidates: list[dict[str, Any]],
        items_administered: int,
    ) -> dict[str, Any] | None:
        """渐进法选题: (1-W)*随机 + W*Fisher信息.

        初期 W 小 → 偏随机; 后期 W 大 → 偏信息量.
        """
        if not candidates:
            return None
        # 估计当前 SE (近似: 1/sqrt(已答信息量))
        if items_administered > 0:
            se_est = 1.0 / math.sqrt(max(items_administered, 1))
        else:
            se_est = 1.0
        w = self.compute_progressive_weight(se_est, items_administered)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in candidates:
            a, b, c = self._item_abc(item)
            info = self._estimator.information(theta, a, b, c)
            rand_val = self._rng.random()
            score = (1.0 - w) * rand_val + w * info
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    # --- a-分层选题 (a-stratified) ---

    def _select_a_stratified(
        self,
        theta: float,
        candidates: list[dict[str, Any]],
        items_administered: int,
    ) -> dict[str, Any] | None:
        """a-分层选题: 初期用低 a 层, 后期用高 a 层.

        将题库按区分度 a 分为三层 (低/中/高), 根据测试进度选择对应层.
        """
        if not candidates:
            return None
        M = max(self._max_items, 3)
        progress = min(items_administered / M, 1.0)
        # 三层划分
        sorted_items = sorted(candidates, key=lambda x: float(x.get("a", 1.0)))
        n = len(sorted_items)
        low_end = max(n // 3, 1)
        mid_end = max(2 * n // 3, low_end + 1)
        if progress < 0.33:
            layer = sorted_items[:low_end]
        elif progress < 0.67:
            layer = sorted_items[low_end:mid_end]
        else:
            layer = sorted_items[mid_end:]
        if not layer:
            layer = candidates
        # 在选定层内用 Fisher 信息选题
        return self._select_fisher_info(theta, layer)

    # --- BKT+IRT 融合选题 (bkt_irt_fusion) ---

    def compute_fusion_score(
        self,
        theta: float,
        item: dict[str, Any],
        fusion_weight: float | None = None,
    ) -> float:
        """计算融合评分: score = (1-w)*fisher_info + w*mastery_weight.

        融合 BKT 掌握度与 IRT Fisher 信息量的联合评分:
        - w=0: 纯 IRT Fisher 信息量 (不关心掌握度)
        - w=1: 纯 BKT 掌握度 (不关心信息量)
        - w=0.5: 等权融合

        mastery_weight 规则:
        - p_mastery ∈ [0.3, 0.7] (ZPD 区) → 1.0
        - p_mastery > 0.7 (已掌握) 或 < 0.3 (挫败区) → 0.0

        Args:
            theta: 当前能力估计 θ.
            item: 题目字典, 含 {"a", "b", "c", "p_mastery"}.
            fusion_weight: 融合权重 w; 为 None 时使用实例的 _fusion_weight.

        Returns:
            融合评分 (非负, 越大越优先).
        """
        w = self._fusion_weight if fusion_weight is None else max(0.0, min(1.0, fusion_weight))
        a, b, c = self._item_abc(item)
        fisher_info = self._estimator.information(theta, a, b, c)

        # mastery_weight: ZPD 区 → 1.0, 否则 → 0.0
        p_mastery = float(item.get("p_mastery", 0.5))
        if _MASTERY_ZPD_LOWER <= p_mastery <= _MASTERY_ZPD_UPPER:
            mastery_weight = 1.0
        else:
            mastery_weight = 0.0

        return (1.0 - w) * fisher_info + w * mastery_weight

    def _select_bkt_irt_fusion(
        self,
        theta: float,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """BKT+IRT 融合选题: 结合 BKT 掌握度和 IRT Fisher 信息量.

        评分公式: score = (1-w)*fisher_info + w*mastery_weight

        流程:
        1. 当 w > 0 时, 过滤排除 p_mastery > 0.7 (已掌握) 和
           p_mastery < 0.3 (挫败区) 的题目;
        2. 如果过滤后无候选, 回退到所有候选 (按融合评分选择);
        3. 在剩余候选中按融合评分选择最高的;
        4. 当 w = 0 (纯 IRT) 时, 不过滤, 直接按 Fisher 信息选择.

        Args:
            theta: 当前能力估计 θ.
            candidates: 候选题目列表 (含 p_mastery 字段).

        Returns:
            选中的题目字典; 无候选返回 None.
        """
        if not candidates:
            return None

        w = self._fusion_weight

        # 当 w > 0 时, 过滤已掌握和挫败区题目
        if w > 0.0:
            zpd_candidates = [
                item for item in candidates
                if _MASTERY_ZPD_LOWER <= float(item.get("p_mastery", 0.5)) <= _MASTERY_ZPD_UPPER
            ]
            # 如果有 ZPD 候选, 在其中选择; 否则回退到所有候选
            pool = zpd_candidates if zpd_candidates else candidates
        else:
            pool = candidates

        # 按融合评分选择最高的
        best_item: dict[str, Any] | None = None
        best_score: float = -1.0
        for item in pool:
            score = self.compute_fusion_score(theta, item, w)
            if score > best_score:
                best_score = score
                best_item = item

        return best_item

    # --- PSER 终止准则 (Predicted SE Reduction) ---

    def predict_se_reduction(
        self,
        theta: float,
        current_se: float,
        available_items: list[dict[str, Any]],
        administered_ids: Iterable[str],
    ) -> list[dict[str, Any]]:
        """预测每道可用题目的 SE 降低量.

        SE_new = 1/sqrt(I_current + I_item)
        reduction = current_se - SE_new
        """
        administered_set = set(administered_ids)
        i_current = 1.0 / max(current_se ** 2, _EPS)
        predictions: list[dict[str, Any]] = []
        for item in available_items:
            item_id = item.get("item_id")
            if item_id in administered_set:
                continue
            a, b, c = self._item_abc(item)
            info = self._estimator.information(theta, a, b, c)
            se_new = 1.0 / math.sqrt(max(i_current + info, _EPS))
            reduction = current_se - se_new
            predictions.append({
                "item_id": item_id,
                "predicted_se": se_new,
                "reduction": max(reduction, 0.0),
                "info": info,
            })
        return predictions

    def should_stop_pser(
        self,
        current_se: float,
        count: int,
        theta: float,
        available_items: list[dict[str, Any]],
        administered_ids: Iterable[str],
        se_threshold: float = _DEFAULT_SE_THRESHOLD,
        hypo: float = 0.01,
        hyper: float = 0.05,
        max_items: int = _DEFAULT_MAX_ITEMS,
        min_items: int = 5,
    ) -> bool:
        """PSER 终止准则 — 基于预测 SE 降低量.

        - SE 已达标 (se < threshold):
          预测最大降幅 > hyper → 继续 (精度还能显著提升)
          否则 → 终止
        - SE 未达标 (se >= threshold):
          预测最大降幅 < hypo → 终止 (再做题也无济于事)
          否则 → 继续
        - 题量达上限 → 终止
        """
        if count >= max_items:
            return True
        if count < min_items:
            return False
        predictions = self.predict_se_reduction(
            theta, current_se, available_items, administered_ids
        )
        if not predictions:
            return True  # 无可用题目
        max_reduction = max(p["reduction"] for p in predictions)
        if current_se < se_threshold:
            # SE 已达标: 降幅 > hyper 则继续
            return max_reduction <= hyper
        else:
            # SE 未达标: 降幅 < hypo 则终止
            return max_reduction < hypo

    # --- 置信区间宽度终止 ---

    def should_stop_ci_width(
        self,
        se: float,
        count: int,
        ci_width_threshold: float = 0.5,
        confidence_level: float = 0.95,
        min_items: int = 5,
        max_items: int = _DEFAULT_MAX_ITEMS,
    ) -> bool:
        """置信区间宽度终止 — CI 宽度 < 阈值时终止.

        CI 宽度 = 2 * z_{α/2} * SE
        """
        if count >= max_items:
            return True
        if count < min_items:
            return False
        alpha = 1.0 - confidence_level
        z = self._norm_ppf(1.0 - alpha / 2.0)
        ci_width = 2.0 * z * se
        return ci_width < ci_width_threshold

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """标准正态分布分位数 (近似)."""
        if p <= 0.0:
            return -3.5
        if p >= 1.0:
            return 3.5
        a = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00,
        ]
        b = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01,
        ]
        c = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00,
        ]
        d = [
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00,
        ]
        p_low = 0.02425
        p_high = 1.0 - p_low
        if p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return x

    # --- 分类准确率终止 ---

    def should_stop_classification(
        self,
        theta: float,
        se: float,
        cut_score: float = 0.0,
        min_confidence: float = 0.95,
    ) -> bool:
        """分类准确率终止 — 能力分级决策足够确定时终止.

        P(theta > cut_score) = Φ((theta - cut_score) / SE)
        若此概率 > min_confidence 或 < (1 - min_confidence) 则分类确定.
        """
        if se <= 0.0:
            return True
        z = (theta - cut_score) / se
        p_above = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return p_above > min_confidence or p_above < (1.0 - min_confidence)

    # --- 多准则组合终止 ---

    def should_stop_multi(
        self,
        current_se: float,
        count: int,
        theta: float,
        available_items: list[dict[str, Any]],
        administered_ids: Iterable[str],
        criteria: list[str] | None = None,
        max_items: int = _DEFAULT_MAX_ITEMS,
        se_threshold: float = _DEFAULT_SE_THRESHOLD,
        ci_width_threshold: float = 0.5,
        cut_score: float = 0.0,
        min_confidence: float = 0.9,
        min_items: int = 5,
    ) -> bool:
        """多准则组合终止 — 任一准则满足即终止.

        支持准则: length / precision / ci_width / classification / pser
        """
        if criteria is None:
            criteria = ["length", "precision"]
        for criterion in criteria:
            if criterion == "length":
                if count >= max_items:
                    return True
            elif criterion == "precision":
                if current_se < se_threshold and count >= min_items:
                    return True
            elif criterion == "ci_width":
                if self.should_stop_ci_width(
                    current_se, count, ci_width_threshold, min_items=min_items, max_items=max_items
                ):
                    return True
            elif criterion == "classification":
                if count >= min_items and self.should_stop_classification(
                    theta, current_se, cut_score, min_confidence
                ):
                    return True
            elif criterion == "pser":
                if self.should_stop_pser(
                    current_se, count, theta, available_items, administered_ids,
                    se_threshold=se_threshold, max_items=max_items, min_items=min_items,
                ):
                    return True
        return False


# ============================================================
# __all__
# ============================================================

__all__ = [
    "CATSelector",
]
