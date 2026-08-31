"""CC1 四层反幻觉评审引擎 — L4 溯源链构建与权威性评级.

为 L4 ProvenanceLayer 提供溯源链 (Provenance Chain) 的构建、遍历、
完整性评估, 以及基于来源层级 (Tier) 的权威性评级能力。

核心能力:
- 来源层级 (SourceTier): 五级来源权威分级 (T1 顶刊 ~ T5 未验证网络来源)
- 溯源节点 (ProvenanceNode): 单条来源的结构化描述 (DOI/作者/年份/置信度等)
- 溯源链 (ProvenanceChain): 有向无环图 (DAG) 溯源链构建与回溯
- 权威评级器 (AuthorityRater): 基于 Tier/年份/引用数的置信度与权威分计算
- 版本管理器 (VersionManager): 知识更新时的溯源版本快照与差异比对

设计灵感:
- W3C PROV 数据模型: 实体-活动-代理溯源三元组
- LlamaIndex Citation: 声明级溯源绑定
- OpenAlex / Crossref: 期刊权威性分级与 DOI 校验
- DataCite: 持久标识符与元数据版本管理

对应 L4 规则:
- P-R04 来源权威性标注 (Tier 等级)
- P-R05 溯源链完整性 (全链路可追溯)
- P-R06 时效性检查 (年份衰减)
- P-R10 动态溯源版本 (版本快照)

仅依赖 Python 标准库 (re, time, uuid, dataclasses, enum, typing)。
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 枚举与常量
# ============================================================


class SourceTier(str, Enum):
    """来源权威层级.

    基于 W3C PROV 与 OpenAlex 期刊分级的五级来源权威体系。
    层级越高 (TIER_1) 权威性越强, 默认置信度越高。

    Attributes:
        TIER_1: 顶级同行评审期刊 (Nature, Science, PRL 等)
        TIER_2: 其他同行评审期刊
        TIER_3: 会议论文、预印本
        TIER_4: 教材、技术报告
        TIER_5: 网络来源、未验证来源
    """

    TIER_1 = "tier_1"  # Peer-reviewed journal (Nature, Science, PRL, etc.)
    TIER_2 = "tier_2"  # Other peer-reviewed journals
    TIER_3 = "tier_3"  # Conference proceedings, preprints
    TIER_4 = "tier_4"  # Textbooks, technical reports
    TIER_5 = "tier_5"  # Web sources, unverified

    @property
    def base_score(self) -> float:
        """该层级的基础权威分 (0-1)."""
        return TIER_BASE_SCORES[self]


#: 各层级基础权威分 (用于置信度与权威分计算的基准)
TIER_BASE_SCORES: dict[SourceTier, float] = {
    SourceTier.TIER_1: 0.95,
    SourceTier.TIER_2: 0.85,
    SourceTier.TIER_3: 0.70,
    SourceTier.TIER_4: 0.55,
    SourceTier.TIER_5: 0.30,
}

#: Tier 1 顶级期刊清单 (小写匹配, 参考 OpenAlex 顶级期刊)
TIER_1_JOURNALS: list[str] = [
    "nature",
    "science",
    "physical review letters",
    "prl",
    "journal of the american chemical society",
    "jacs",
    "advanced materials",
    "nano letters",
    "acs nano",
    "nature materials",
    "nature nanotechnology",
    "nature photonics",
    "nature energy",
    "nature chemistry",
    "nature communications",
    "physical review b",
    "physical review x",
    "journal of physical chemistry letters",
    "journal of physical chemistry c",
    "angewandte chemie",
    "angewandte chemie international edition",
    "chemical society reviews",
    "chemical reviews",
    "energy & environmental science",
    "advanced functional materials",
    "advanced energy materials",
    "light: science & applications",
    "laser & photonics reviews",
    "acs energy letters",
    "joule",
    "matter",
    "device",
]

#: 来源类型到默认层级的映射 (无进一步信息时使用)
SOURCE_TYPE_DEFAULT_TIER: dict[str, SourceTier] = {
    "journal": SourceTier.TIER_2,
    "conference": SourceTier.TIER_3,
    "textbook": SourceTier.TIER_4,
    "web": SourceTier.TIER_5,
    "database": SourceTier.TIER_3,
    "computed": SourceTier.TIER_4,
}

#: DOI 结构正则 (10.<registrant>/<suffix>), 遵循 ISO 26324
_DOI_PATTERN = re.compile(r"^10\.(\d{4,9})/(\S+)$")

#: DOI 常见 URL/协议前缀 (校验前需剥离)
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
    "doi.org/",
)


# ============================================================
# 辅助函数
# ============================================================


def _strip_doi_prefix(doi: str) -> str:
    """去除 DOI 的常见 URL/协议前缀, 返回纯净的 DOI 字符串."""
    cleaned = doi.strip()
    lowered = cleaned.lower()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned.strip()


def _current_year() -> int:
    """返回当前年份 (基于 time 标准库, 无需 datetime 依赖)."""
    return time.localtime(time.time()).tm_year


def _text_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的词级 Jaccard 相似度 (0-1).

    仅使用内置集合运算, 不依赖 difflib 等额外模块。
    两段空文本视为完全相似 (返回 1.0)。
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    union = words_a | words_b
    if not union:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / len(union)


# ============================================================
# 数据结构
# ============================================================


@dataclass
class ProvenanceNode:
    """溯源节点 — 描述单条来源的结构化信息.

    每个节点代表溯源链中的一个来源实体, 通过 parent_ids 与显式 link
    构成有向无环图 (DAG), 实现从最终声明到原始数据的全链路追溯。

    Attributes:
        node_id: 节点唯一标识
        source_type: 来源类型 (journal/conference/textbook/web/database/computed)
        source_uri: 来源统一资源标识 (URL/DOI/知识库节点 ID)
        title: 来源标题
        authors: 作者列表
        year: 发表年份 (0 表示未知)
        tier: 来源权威层级
        doi: 数字对象标识符 (可选, 默认空)
        confidence: 来源置信度 (0-1)
        content: 来源内容摘要
        parent_ids: 父节点 ID 列表 (用于链路链接)
    """

    node_id: str
    source_type: str
    source_uri: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int = 0
    tier: SourceTier = SourceTier.TIER_5
    doi: str = ""
    confidence: float = 0.0
    content: str = ""
    parent_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """校验字段合法性."""
        if not self.node_id or not self.node_id.strip():
            raise ValueError("node_id 不能为空")
        if not self.source_type or not self.source_type.strip():
            raise ValueError("source_type 不能为空")
        # 兼容从字符串/原始值构造 tier
        if not isinstance(self.tier, SourceTier):
            try:
                self.tier = SourceTier(self.tier)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"无效的 tier 值: {self.tier!r}") from exc
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence 必须在 [0, 1] 范围内, 当前为 {self.confidence}"
            )
        if self.doi:
            self.doi = self.doi.strip()

    def to_dict(self) -> dict:
        """序列化为字典."""
        return {
            "node_id": self.node_id,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "tier": self.tier.value,
            "doi": self.doi,
            "confidence": self.confidence,
            "content": self.content,
            "parent_ids": list(self.parent_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProvenanceNode":
        """从字典反序列化.

        Args:
            data: 包含节点字段的字典

        Returns:
            反序列化得到的 ProvenanceNode 实例

        Raises:
            KeyError: 缺少必填字段 node_id / source_type
            ValueError: 字段值非法
        """
        tier_value = data.get("tier", SourceTier.TIER_5.value)
        tier = (
            tier_value
            if isinstance(tier_value, SourceTier)
            else SourceTier(tier_value)
        )
        return cls(
            node_id=data["node_id"],
            source_type=data["source_type"],
            source_uri=data.get("source_uri", ""),
            title=data.get("title", ""),
            authors=list(data.get("authors", [])),
            year=int(data.get("year", 0)),
            tier=tier,
            doi=data.get("doi", ""),
            confidence=float(data.get("confidence", 0.0)),
            content=data.get("content", ""),
            parent_ids=list(data.get("parent_ids", [])),
        )


# ============================================================
# 溯源链 (ProvenanceChain)
# ============================================================


class ProvenanceChain:
    """溯源链 — 有向无环图 (DAG) 溯源结构.

    维护节点集合与父子链接, 支持链路回溯、深度计算、完整性评估与
    缺口检测。链路来源有两类, 两者会被合并统一管理:

    1. 节点自身的 ``parent_ids`` 字段 (add_node 时自动注册)
    2. 通过 :meth:`link` 显式建立的父子关系

    回溯方向: 从目标节点沿 parent 链向上追溯到根节点 (无父节点的来源)。
    ``get_chain`` 返回的列表按 "根节点在前, 目标节点在后" 排序。
    """

    def __init__(self) -> None:
        """初始化空溯源链."""
        self._nodes: dict[str, ProvenanceNode] = {}
        # child_id -> {parent_ids}  (反向边, 用于向上回溯)
        self._parents: dict[str, set[str]] = {}
        # parent_id -> [child_ids]  (正向边, 用于序列化与遍历)
        self._children: dict[str, list[str]] = {}

    # ---- 节点与链接管理 ----

    def add_node(self, node: ProvenanceNode) -> str:
        """添加一个溯源节点到链中.

        节点 ``parent_ids`` 中引用的父节点无需预先存在 (允许前向引用),
        缺失的父节点会在 :meth:`detect_gaps` 中被标记为断链。

        Args:
            node: 待添加的溯源节点

        Returns:
            被添加节点的 node_id

        Raises:
            ValueError: node_id 已存在或节点非法
        """
        if node.node_id in self._nodes:
            raise ValueError(f"节点已存在: {node.node_id}")
        self._nodes[node.node_id] = node
        self._parents.setdefault(node.node_id, set())
        self._children.setdefault(node.node_id, [])
        # 注册节点自带的 parent_ids
        for pid in node.parent_ids:
            self._parents[node.node_id].add(pid)
            self._children.setdefault(pid, []).append(node.node_id)
        return node.node_id

    def link(self, parent_id: str, child_id: str) -> None:
        """在两个已存在节点之间建立父子链接 (parent_id -> child_id).

        Args:
            parent_id: 父节点 ID (上游来源)
            child_id: 子节点 ID (下游派生)

        Raises:
            ValueError: 父/子节点不存在、自环、或会形成环路
        """
        if parent_id not in self._nodes:
            raise ValueError(f"父节点不存在: {parent_id}")
        if child_id not in self._nodes:
            raise ValueError(f"子节点不存在: {child_id}")
        if parent_id == child_id:
            raise ValueError("不能将节点链接到自身 (自环)")
        if self._would_create_cycle(parent_id, child_id):
            raise ValueError(
                f"链接 {parent_id} -> {child_id} 会产生环路"
            )
        # 避免重复链接
        if parent_id not in self._parents[child_id]:
            self._parents[child_id].add(parent_id)
            self._children.setdefault(parent_id, []).append(child_id)
            child = self._nodes[child_id]
            if parent_id not in child.parent_ids:
                child.parent_ids.append(parent_id)

    def build_chain(
        self,
        nodes: list[ProvenanceNode],
        links: list[tuple[str, str]],
    ) -> None:
        """从批量数据构建溯源链.

        先添加全部节点, 再建立显式链接。节点自带 parent_ids 也会被注册。

        Args:
            nodes: 节点列表
            links: 父子链接列表, 每项为 (parent_id, child_id)

        Raises:
            ValueError: 节点重复或链接非法 (沿用 add_node / link 的校验)
        """
        for node in nodes:
            self.add_node(node)
        for parent_id, child_id in links:
            self.link(parent_id, child_id)

    # ---- 查询与回溯 ----

    def get_node(self, node_id: str) -> ProvenanceNode:
        """获取指定节点.

        Raises:
            ValueError: 节点不存在
        """
        if node_id not in self._nodes:
            raise ValueError(f"节点不存在: {node_id}")
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """判断节点是否存在."""
        return node_id in self._nodes

    def get_chain(self, node_id: str) -> list[ProvenanceNode]:
        """获取节点的完整溯源链 (从根节点回溯到目标节点).

        沿 parent 链向上广度优先遍历, 收集所有祖先节点与目标节点,
        返回顺序为 "根节点在前, 目标节点在后"。支持 DAG (多父节点)
        与环路安全 (已访问集合去重)。

        Args:
            node_id: 目标节点 ID

        Returns:
            从根到目标的节点列表 (根在前, 目标在后)

        Raises:
            ValueError: 节点不存在
        """
        if node_id not in self._nodes:
            raise ValueError(f"节点不存在: {node_id}")
        visited: set[str] = set()
        layers: list[tuple[str, int]] = []  # (node_id, depth_from_target)
        queue: list[tuple[str, int]] = [(node_id, 0)]
        while queue:
            nid, depth = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            layers.append((nid, depth))
            for pid in self._parents.get(nid, set()):
                if pid not in visited:
                    queue.append((pid, depth + 1))
        # 深度大的 (更靠近根) 排在前, 同深度保持发现顺序 (稳定排序)
        layers.sort(key=lambda item: -item[1])
        return [
            self._nodes[nid]
            for nid, _ in layers
            if nid in self._nodes
        ]

    def get_depth(self, node_id: str) -> int:
        """获取节点的溯源链深度.

        深度定义为从该节点到根节点的最长路径边数
        (根节点本身深度为 0, 根的直接子节点深度为 1, 依此类推)。

        Args:
            node_id: 目标节点 ID

        Returns:
            最长回溯路径的边数

        Raises:
            ValueError: 节点不存在
        """
        if node_id not in self._nodes:
            raise ValueError(f"节点不存在: {node_id}")
        visited: set[str] = set()
        max_depth = 0
        queue: list[tuple[str, int]] = [(node_id, 0)]
        while queue:
            nid, depth = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            if depth > max_depth:
                max_depth = depth
            for pid in self._parents.get(nid, set()):
                if pid not in visited:
                    queue.append((pid, depth + 1))
        return max_depth

    # ---- 完整性评估 ----

    def is_complete(self, node_id: str) -> bool:
        """检查溯源链是否完整 (链中所有节点均具备 source_uri).

        Args:
            node_id: 目标节点 ID

        Returns:
            完整返回 True, 任一节点缺少 source_uri 返回 False

        Raises:
            ValueError: 节点不存在
        """
        chain = self.get_chain(node_id)
        return all(node.source_uri.strip() != "" for node in chain)

    def get_completeness_score(self, node_id: str) -> float:
        """计算溯源链的完整性评分 (0-1).

        评分由链中各节点的元数据完备度加权平均:
        - source_uri 非空: 0.60
        - content 非空: 0.25
        - doi 或 authors 非空: 0.15

        Args:
            node_id: 目标节点 ID

        Returns:
            完整性评分 (0-1, 保留 4 位小数)

        Raises:
            ValueError: 节点不存在
        """
        chain = self.get_chain(node_id)
        if not chain:
            return 0.0
        total = 0.0
        for node in chain:
            score = 0.0
            if node.source_uri.strip():
                score += 0.60
            if node.content.strip():
                score += 0.25
            if node.doi.strip() or node.authors:
                score += 0.15
            total += score
        return round(total / len(chain), 4)

    def detect_gaps(self, node_id: str) -> list[str]:
        """识别溯源链中的缺口 (缺失链接/元数据).

        检测项:
        - 节点缺少 source_uri (无法定位来源)
        - 节点缺少 content (无内容摘要)
        - parent_ids 引用了不存在的父节点 (断链)
        - parent_ids 引用的父节点不在当前回溯链中 (链路偏移)
        - 派生来源 (computed/database) 缺少父节点 (溯源中断)

        Args:
            node_id: 目标节点 ID

        Returns:
            缺口描述列表 (每项为人类可读的中文说明)

        Raises:
            ValueError: 节点不存在
        """
        if node_id not in self._nodes:
            raise ValueError(f"节点不存在: {node_id}")
        gaps: list[str] = []
        chain = self.get_chain(node_id)
        chain_ids = {node.node_id for node in chain}
        for node in chain:
            if not node.source_uri.strip():
                gaps.append(f"节点 {node.node_id} 缺少 source_uri, 无法定位来源")
            if not node.content.strip():
                gaps.append(f"节点 {node.node_id} 缺少 content, 无内容摘要")
            for pid in node.parent_ids:
                if pid not in self._nodes:
                    gaps.append(
                        f"节点 {node.node_id} 引用了不存在的父节点 {pid} (断链)"
                    )
                elif pid not in chain_ids:
                    gaps.append(
                        f"节点 {node.node_id} 的父节点 {pid} 不在当前回溯链中"
                    )
            # 派生来源应有上游原始来源
            if node.source_type in ("computed", "database") and not node.parent_ids:
                gaps.append(
                    f"派生来源节点 {node.node_id} ({node.source_type}) "
                    f"缺少父节点, 溯源链在此中断"
                )
        return gaps

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        """序列化溯源链为字典.

        Returns:
            包含 ``nodes`` (节点列表) 与 ``links`` (父子链接列表) 的字典
        """
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "links": [
                [parent_id, child_id]
                for parent_id, children in self._children.items()
                for child_id in children
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProvenanceChain":
        """从字典反序列化溯源链.

        Args:
            data: :meth:`to_dict` 产出的字典

        Returns:
            重建的 ProvenanceChain 实例
        """
        chain = cls()
        for node_data in data.get("nodes", []):
            chain.add_node(ProvenanceNode.from_dict(node_data))
        for link in data.get("links", []):
            parent_id, child_id = link[0], link[1]
            # 跳过因 parent_ids 已注册而重复的链接
            if parent_id in chain._parents.get(child_id, set()):
                continue
            chain.link(parent_id, child_id)
        return chain

    # ---- 魔术方法 ----

    def __contains__(self, node_id: object) -> bool:
        return isinstance(node_id, str) and node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    # ---- 内部方法 ----

    def _would_create_cycle(self, parent_id: str, child_id: str) -> bool:
        """检测添加 parent_id -> child_id 链接是否会产生环路.

        新链接使 child_id 的父节点新增 parent_id (向上边 child_id -> parent_id)。
        若 parent_id 沿 parent 链向上已可达 child_id, 则添加该链接会形成
        child_id -> parent_id -> ... -> child_id 的环路。

        Args:
            parent_id: 待链接的父节点 ID
            child_id: 待链接的子节点 ID

        Returns:
            会产生环路返回 True, 否则 False
        """
        visited: set[str] = set()
        stack: list[str] = [parent_id]
        while stack:
            nid = stack.pop()
            if nid == child_id:
                return True
            if nid in visited:
                continue
            visited.add(nid)
            stack.extend(self._parents.get(nid, set()))
        return False


# ============================================================
# 权威评级器 (AuthorityRater)
# ============================================================


class AuthorityRater:
    """来源权威评级器.

    基于来源类型、期刊名称、DOI、发表年份与引用数, 判定来源层级
    (SourceTier) 并计算置信度与权威分。

    期刊分级参考 OpenAlex / Crossref 元数据, DOI 校验遵循
    ISO 26324 (``10.<registrant>/<suffix>``) 结构规范。

    注意: DOI 系统无内置校验和 (区别于 ISBN), :meth:`validate_doi_checksum`
    仅做更严格的结构校验, 真正的有效性需通过 Crossref/DataCite API 验证。
    """

    def __init__(self) -> None:
        """初始化评级器, 预编译 Tier 1 期刊匹配集合."""
        self._tier1_journals: set[str] = {j.lower() for j in TIER_1_JOURNALS}

    def determine_tier_from_journal(self, journal_name: str) -> SourceTier:
        """根据期刊名称判定来源层级.

        命中 Tier 1 顶级期刊清单 (大小写不敏感, 支持子串匹配) 返回 TIER_1,
        否则默认视为 TIER_2 (其他同行评审期刊)。

        Args:
            journal_name: 期刊名称

        Returns:
            判定的 SourceTier
        """
        name = (journal_name or "").strip().lower()
        if not name:
            return SourceTier.TIER_2
        for journal in self._tier1_journals:
            if journal == name or journal in name or name in journal:
                return SourceTier.TIER_1
        return SourceTier.TIER_2

    def rate(
        self,
        source_type: str,
        doi: str = "",
        journal_name: str = "",
        year: int = 0,
    ) -> SourceTier:
        """评定来源权威层级.

        评级策略:
        - ``journal``: 优先按期刊名称分级 (命中 Tier 1 清单则 TIER_1,
          否则 TIER_2); 无期刊名但具备合法 DOI 仍视为 TIER_2
        - ``conference``: TIER_3 (含预印本)
        - ``textbook`` / ``computed``: TIER_4
        - ``database``: TIER_3 (经验证的结构化数据库)
        - ``web`` 及其他: TIER_5

        Args:
            source_type: 来源类型
            doi: 数字对象标识符 (可选)
            journal_name: 期刊名称 (可选, 仅 journal 类型生效)
            year: 发表年份 (可选, 当前实现未直接用于分级, 保留接口)

        Returns:
            判定的 SourceTier
        """
        st = (source_type or "").strip().lower()
        if st == "journal":
            if journal_name and journal_name.strip():
                return self.determine_tier_from_journal(journal_name)
            # 无期刊名: 合法 DOI 视为同行评审期刊, 否则仍默认 TIER_2
            return SourceTier.TIER_2
        return SOURCE_TYPE_DEFAULT_TIER.get(st, SourceTier.TIER_5)

    def calculate_confidence(
        self,
        tier: SourceTier,
        year: int,
        citation_count: int = 0,
    ) -> float:
        """计算来源置信度 (0-1).

        置信度 = 基础分 (由 Tier 决定) × 时效性因子 + 引用加成

        - 时效性因子: 发表 3 年内 1.0, 5 年内 0.9, 10 年内 0.8,
          15 年内 0.65, 更早 0.5; 年份未知取 0.7
        - 引用加成: ``min(0.15, sqrt(citation_count) × 0.015)``, 上限 0.15
        - 最终结果裁剪至 [0, 1] 并保留 4 位小数

        Args:
            tier: 来源层级
            year: 发表年份 (0 表示未知)
            citation_count: 被引次数 (非负)

        Returns:
            置信度评分 (0-1)

        Raises:
            ValueError: tier 非法或 citation_count 为负
        """
        if not isinstance(tier, SourceTier):
            try:
                tier = SourceTier(tier)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"无效的 tier 值: {tier!r}") from exc
        if citation_count < 0:
            raise ValueError(f"citation_count 不能为负, 当前为 {citation_count}")

        base = TIER_BASE_SCORES[tier]

        # 时效性因子
        if year and year > 0:
            age = _current_year() - year
            if age <= 3:
                recency = 1.0
            elif age <= 5:
                recency = 0.9
            elif age <= 10:
                recency = 0.8
            elif age <= 15:
                recency = 0.65
            else:
                recency = 0.5
            # 未来年份容错
            if recency > 1.0:
                recency = 1.0
        else:
            recency = 0.7

        # 引用加成 (亚线性, 上限 0.15)
        citation_bonus = 0.0
        if citation_count > 0:
            citation_bonus = min(0.15, (citation_count ** 0.5) * 0.015)

        confidence = base * recency + citation_bonus
        return round(max(0.0, min(1.0, confidence)), 4)

    def validate_doi_format(self, doi: str) -> bool:
        """校验 DOI 格式 (10.XXXX/...).

        剥离常见 URL 前缀后, 匹配 ``10.<4-9 位注册号>/<非空白后缀>``。

        Args:
            doi: 待校验的 DOI 字符串 (可含 https://doi.org/ 等前缀)

        Returns:
            格式合法返回 True, 否则 False
        """
        if not doi:
            return False
        cleaned = _strip_doi_prefix(doi)
        return bool(_DOI_PATTERN.match(cleaned))

    def validate_doi_checksum(self, doi: str) -> bool:
        """DOI 结构性校验 (非完整 API 校验).

        DOI 系统无内置校验和 (区别于 ISBN-13), 此方法在格式校验基础上
        做更严格的结构检查:

        - 必须通过 :meth:`validate_doi_format`
        - 注册号 (registrant) 长度为 4-9 位数字
        - 后缀 (suffix) 长度不小于 2

        真正的 DOI 有效性 (是否已注册、元数据是否匹配) 需通过
        Crossref / DataCite API 验证, 本方法不做网络请求。

        Args:
            doi: 待校验的 DOI 字符串

        Returns:
            通过结构校验返回 True, 否则 False
        """
        if not self.validate_doi_format(doi):
            return False
        cleaned = _strip_doi_prefix(doi)
        match = _DOI_PATTERN.match(cleaned)
        if not match:
            return False
        registrant, suffix = match.group(1), match.group(2)
        if not (4 <= len(registrant) <= 9):
            return False
        if len(suffix) < 2:
            return False
        return True

    def calculate_authority_score(
        self,
        chain: ProvenanceChain,
        node_id: str,
    ) -> float:
        """计算溯源链的整体权威分 (0-1).

        权威分综合链中各节点的层级与置信度, 并按完整性惩罚:

        1. 对链中每个节点, 取 ``tier 基础分 × confidence`` 作为加权贡献,
           以 tier 基础分作为权重做加权平均 (高层级来源贡献更大)
        2. 乘以完整性调节因子 ``0.7 + 0.3 × completeness_score``
           (链不完整时权威分下降)
        3. 结果裁剪至 [0, 1] 并保留 4 位小数

        Args:
            chain: 溯源链实例
            node_id: 目标节点 ID

        Returns:
            整体权威分 (0-1)

        Raises:
            ValueError: 节点不存在 (由 chain.get_chain 抛出)
        """
        nodes = chain.get_chain(node_id)
        if not nodes:
            return 0.0
        total_weighted = 0.0
        total_weight = 0.0
        for node in nodes:
            tier_score = TIER_BASE_SCORES.get(node.tier, 0.0)
            total_weighted += tier_score * node.confidence
            total_weight += tier_score
        base = total_weighted / total_weight if total_weight > 0 else 0.0
        completeness = chain.get_completeness_score(node_id)
        score = base * (0.7 + 0.3 * completeness)
        return round(max(0.0, min(1.0, score)), 4)


# ============================================================
# 版本管理器 (VersionManager)
# ============================================================


class VersionManager:
    """溯源版本管理器.

    为溯源节点的内容维护版本快照, 支持版本回溯、列举与差异比对。
    对应 P-R10 (动态溯源版本) 规则: 知识更新时创建新溯源版本。

    版本按节点分组, 每次创建递增 version_number, 并记录时间戳与变更原因。
    """

    def __init__(self) -> None:
        """初始化版本存储."""
        # node_id -> [version_record, ...]  (按创建顺序)
        self._by_node: dict[str, list[dict]] = {}
        # version_id -> version_record  (全局索引, 加速查询)
        self._by_id: dict[str, dict] = {}

    def create_version(
        self,
        node_id: str,
        content: str,
        reason: str = "",
    ) -> str:
        """为节点创建一个新版本快照.

        Args:
            node_id: 关联的溯源节点 ID
            content: 该版本的内容文本
            reason: 变更原因 (可选)

        Returns:
            新创建的版本 ID (形如 ``ver-<uuid>``)

        Raises:
            ValueError: node_id 为空
        """
        if not node_id or not node_id.strip():
            raise ValueError("node_id 不能为空")
        versions = self._by_node.setdefault(node_id, [])
        version_number = len(versions) + 1
        version_id = f"ver-{uuid.uuid4().hex[:12]}"
        record: dict = {
            "version_id": version_id,
            "node_id": node_id,
            "content": content,
            "reason": reason,
            "version_number": version_number,
            "timestamp": time.time(),
        }
        versions.append(record)
        self._by_id[version_id] = record
        return version_id

    def get_version(self, version_id: str) -> dict | None:
        """获取指定版本快照.

        Args:
            version_id: 版本 ID

        Returns:
            版本记录字典, 不存在则返回 None
        """
        return self._by_id.get(version_id)

    def get_latest_version(self, node_id: str) -> dict | None:
        """获取节点的最新版本.

        Args:
            node_id: 节点 ID

        Returns:
            最新版本记录字典, 节点无版本则返回 None
        """
        versions = self._by_node.get(node_id)
        if not versions:
            return None
        return versions[-1]

    def list_versions(self, node_id: str) -> list[dict]:
        """列出节点的全部版本 (按创建顺序).

        Args:
            node_id: 节点 ID

        Returns:
            版本记录列表的浅拷贝, 无版本则返回空列表
        """
        return list(self._by_node.get(node_id, []))

    def compare_versions(self, v1_id: str, v2_id: str) -> dict:
        """比较两个版本的差异.

        Args:
            v1_id: 第一个版本 ID
            v2_id: 第二个版本 ID

        Returns:
            差异字典, 包含:
            - ``v1_id`` / ``v2_id``: 版本 ID
            - ``node_id``: v1 所属节点 ID
            - ``same_node``: 两版本是否属于同一节点
            - ``content_changed``: 内容是否不同
            - ``similarity``: 词级 Jaccard 相似度 (0-1)
            - ``v1_content`` / ``v2_content``: 各版本内容
            - ``v1_version_number`` / ``v2_version_number``: 版本号
            - ``v1_reason`` / ``v2_reason``: 变更原因
            - ``v1_timestamp`` / ``v2_timestamp``: 时间戳
            - ``time_diff``: v2 与 v1 的时间差 (秒, 可为负)

        Raises:
            ValueError: 任一版本 ID 不存在
        """
        v1 = self._by_id.get(v1_id)
        v2 = self._by_id.get(v2_id)
        if v1 is None:
            raise ValueError(f"版本不存在: {v1_id}")
        if v2 is None:
            raise ValueError(f"版本不存在: {v2_id}")
        return {
            "v1_id": v1_id,
            "v2_id": v2_id,
            "node_id": v1["node_id"],
            "same_node": v1["node_id"] == v2["node_id"],
            "content_changed": v1["content"] != v2["content"],
            "similarity": _text_similarity(v1["content"], v2["content"]),
            "v1_content": v1["content"],
            "v2_content": v2["content"],
            "v1_version_number": v1["version_number"],
            "v2_version_number": v2["version_number"],
            "v1_reason": v1["reason"],
            "v2_reason": v2["reason"],
            "v1_timestamp": v1["timestamp"],
            "v2_timestamp": v2["timestamp"],
            "time_diff": v2["timestamp"] - v1["timestamp"],
        }


__all__ = [
    "SourceTier",
    "ProvenanceNode",
    "ProvenanceChain",
    "AuthorityRater",
    "VersionManager",
    "TIER_BASE_SCORES",
    "TIER_1_JOURNALS",
    "SOURCE_TYPE_DEFAULT_TIER",
]
