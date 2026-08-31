"""L2 知识点目录 (KP Catalog) — 领域知识单点 (SSOT).

来源: 02-设计/L2-个性化设计 §2.1.1 (42 个 KP, 4 域) +
      02-设计/知识图谱重构方案.md §2 (6 章 → 节 → 知识点 三层树, P1a).

收敛自原散落副本:
- L7 renderers/_common.py (KP_DOMAIN_IDS/KP_NAMES/KP_LEVELS)
- L2 practice.py NODE_TO_KP (kg_node → KP 映射)
- L3 knowledge_seed.py metadata (kg_node/domain)
- L3 fact_check.py KP_KEYWORDS (KP-001 旧式编号, 收为 aliases)

引用约定 (迁移两态):
- 旧体系 (已冻结, 供迁移前零回归): 全系统知识点 ID 统一为 A-01 ~ D-08 (42 个)。
- 新体系 (P1a 新增, 6 章三层树): 知识点 ID 为「章.节.序号」(如 2.3.2), 48 个
  (42 个由旧 ID 重编号 + 6 个「绿色健康照明应用」新增)。
- 旧 ID A-01~D-08 在新体系中降级为 aliases (见 OLD_TO_NEW / NEW_TO_OLD 迁移映射表)。
- 物理重编号 (把实体主键 kp:A-01 换成 kp:1.1.1) 由 P4「数据迁移重映射」执行;
  在此之前, 旧导出接口 (KP_NAMES/KP_LEVELS/KP_EDGES/NODE_TO_KP/ALL_KP_IDS/KP_DOMAIN_IDS)
  原样冻结, 下游 10 处消费者 (kp_graph_seed/knowledge_seed/unified_app/agent_workers/
  practice/l2+ l3 router/l2.models/kp_roles/l7._common) 零改动。
"""
from __future__ import annotations

from typing import Any

# ============================================================
# 旧体系 (冻结 · 迁移前零回归) — 原样保留
# ============================================================

# 域名称 (单点定义)
DOMAIN_LABELS: dict[str, str] = {
    "A": "能级跃迁理论",
    "B": "材料体系设计",
    "C": "合成制备工艺",
    "D": "表征测试技术",
}

# 每域 KP 数量 (A13/B11/C10/D8)
_DOMAIN_COUNTS: dict[str, int] = {"A": 13, "B": 11, "C": 10, "D": 8}

# 每域各层级 KP 数量 (L1 基础 / L2 进阶 / L3 前沿)
_DOMAIN_LEVEL_SPLIT: dict[str, tuple[int, int]] = {
    "A": (5, 5),   # L1: A-01~05, L2: A-06~10, L3: A-11~13
    "B": (4, 4),   # L1: B-01~04, L2: B-05~08, L3: B-09~11
    "C": (4, 3),   # L1: C-01~04, L2: C-05~07, L3: C-08~10
    "D": (3, 3),   # L1: D-01~03, L2: D-04~06, L3: D-07~08
}

# KP 名称 (42 条, 单点)
_KP_NAMES: dict[str, str] = {
    "A-01": "稀土离子的电子构型", "A-02": "4f 壳层与 5d 轨道特征",
    "A-03": "原子光谱项与能级", "A-04": "L-S 耦合与跃迁选择定则",
    "A-05": "Dy3+ 能级结构与 4f-4f 跃迁", "A-06": "晶体场分裂 (Dq 计算)",
    "A-07": "Judd-Ofelt 理论与强度参数", "A-08": "4f-5d 宽带跃迁",
    "A-09": "Stark 能级劈裂", "A-10": "荧光寿命与辐射跃迁速率",
    "A-11": "能量传递与交叉弛豫", "A-12": "浓度猝灭机理",
    "A-13": "热猝灭与温度依赖发光",
    "B-01": "氟化物基质 (NaGdF4) 晶格", "B-02": "磷酸盐基质 (YPO4) 晶格",
    "B-03": "铝酸盐基质 (BaMgAl10O17) 晶格", "B-04": "掺杂与电荷补偿策略",
    "B-05": "晶格对称性与局部配位环境", "B-06": "量子效率与发光内量子效率",
    "B-07": "色坐标与色纯度 (CIE)", "B-08": "激发光谱与吸收截面",
    "B-09": "上转换与量子剪裁", "B-10": "缺陷化学与陷阱态",
    "B-11": "纳米材料表面效应与核壳结构",
    "C-01": "固相烧结法", "C-02": "共沉淀法", "C-03": "溶胶-凝胶法",
    "C-04": "水热/溶剂热法", "C-05": "焙烧温度对结晶度的影响",
    "C-06": "还原气氛与价态控制", "C-07": "助熔剂与晶粒形貌调控",
    "C-08": "前驱体配比与掺杂均匀性", "C-09": "工艺参数与缺陷控制",
    "C-10": "规模放大与批次一致性",
    "D-01": "XRD 物相与结晶度分析", "D-02": "SEM/TEM 形貌表征",
    "D-03": "光谱仪与发射/激发光谱测量", "D-04": "荧光寿命测试与拟合",
    "D-05": "量子效率 (绝对法) 测量", "D-06": "热稳定性 (T50%) 测试",
    "D-07": "ICP-OES 掺杂浓度定量", "D-08": "色度学测量与 CIE 计算",
}

# kg_node (知识图谱节点) → KP 映射 (单点, 原 practice.py NODE_TO_KP + L3 seed)
# 覆盖 42 KP: 题库 kg_node (practice/pretest_bank) + 教材兜底层事实 id (textbook_fallback.tf-dy-*)
_KG_NODES: dict[str, list[str]] = {
    # ---- A 能级跃迁理论 ----
    "A-01": ["Dy3+", "tf-dy-03"],
    "A-02": ["tf-dy-03"],
    "A-03": ["tf-dy-23"],
    "A-04": ["tf-dy-04"],
    "A-05": ["F9_2", "H15_2", "H13_2", "H11_2", "tf-dy-01", "tf-dy-02", "tf-dy-04"],
    "A-06": ["tf-dy-24"],
    "A-07": ["tf-dy-05"],
    "A-08": ["tf-dy-25"],
    "A-09": ["tf-dy-26"],
    "A-10": ["tf-dy-27"],
    "A-11": ["tf-dy-13", "tf-dy-20"],
    "A-12": ["concentration_quench", "tf-dy-12", "tf-dy-13"],
    "A-13": ["thermal", "tf-dy-14"],
    # ---- B 材料体系设计 ----
    "B-01": ["tf-dy-21"],
    "B-02": ["tf-dy-28"],
    "B-03": ["tf-dy-29"],
    "B-04": ["tf-dy-30"],
    "B-05": ["tf-dy-31"],
    "B-06": ["tf-dy-11"],
    "B-07": ["cie", "std_bluelight", "std_cri", "tf-dy-15", "tf-dy-16", "tf-dy-17", "tf-dy-22"],
    "B-08": ["led_pack", "tf-dy-41"],
    "B-09": ["YB_ratio", "tf-dy-18", "tf-dy-19"],
    "B-10": ["tf-dy-32"],
    "B-11": ["tf-dy-33"],
    # ---- C 合成制备工艺 ----
    "C-01": ["synthesis", "tf-dy-07"],
    "C-02": ["CNYP", "tf-dy-08"],
    "C-03": ["tf-dy-08"],
    "C-04": ["tf-dy-08"],
    "C-05": ["tf-dy-34"],
    "C-06": ["tf-dy-07"],
    "C-07": ["tf-dy-35"],
    "C-08": ["tf-dy-08"],
    "C-09": ["tf-dy-36"],
    "C-10": ["tf-dy-37"],
    # ---- D 表征测试技术 ----
    "D-01": ["xrd", "tf-dy-06"],
    "D-02": ["tf-dy-38"],
    "D-03": ["pl", "tf-dy-09"],
    "D-04": ["tf-dy-10"],
    "D-05": ["tf-dy-11"],
    "D-06": ["tf-dy-14"],
    "D-07": ["tf-dy-39"],
    "D-08": ["tf-dy-40"],
}

# ============================================================
# KP 关系边表 — 教学语义关系 (镝-绿色健康照明垂直领域)
# ============================================================
# 每条边: src --rel--> dst。关系语义见 l3/models.py RelationType:
#   - prerequisite_of: 纵向, src 是 dst 的「前提」(学 dst 前先掌握 src)
#   - analogous_to:    横向, 同类机制/对比 (对称, 无需双向手写)
#   - affects:         横向, src 因果影响 dst (跨域)
#   - characterized_by:横向, src 由 dst 方法表征 (机理/材料 → 怎么测)
# deepens (深化) 不手写: 由 prerequisite_of 反向自动推导, 避免两遍不一致。
_KP_EDGES: list[dict[str, str]] = [
    # ---- 纵向: prerequisite_of (A 域「电子构型 → 能级 → 猝灭」主线) ----
    {"src": "A-01", "rel": "prerequisite_of", "dst": "A-02", "reason": "电子构型是理解 4f/5d 轨道特征的基础"},
    {"src": "A-01", "rel": "prerequisite_of", "dst": "A-03", "reason": "电子构型决定光谱项"},
    {"src": "A-04", "rel": "prerequisite_of", "dst": "A-03", "reason": "L-S 耦合产生多重光谱项"},
    {"src": "A-03", "rel": "prerequisite_of", "dst": "A-05", "reason": "光谱项构成 Dy3+ 能级结构"},
    {"src": "A-02", "rel": "prerequisite_of", "dst": "A-05", "reason": "4f/5d 轨道特征支撑能级结构"},
    {"src": "A-05", "rel": "prerequisite_of", "dst": "A-06", "reason": "能级结构是晶体场分裂的前提"},
    {"src": "A-06", "rel": "prerequisite_of", "dst": "A-09", "reason": "晶体场分裂产生 Stark 劈裂"},
    {"src": "A-05", "rel": "prerequisite_of", "dst": "A-07", "reason": "能级是 Judd-Ofelt 强度参数的前提"},
    {"src": "A-07", "rel": "prerequisite_of", "dst": "A-10", "reason": "JO 参数计算辐射跃迁速率"},
    {"src": "A-05", "rel": "prerequisite_of", "dst": "A-10", "reason": "能级结构决定荧光寿命"},
    {"src": "A-05", "rel": "prerequisite_of", "dst": "A-11", "reason": "能级匹配是能量传递的前提"},
    {"src": "A-10", "rel": "prerequisite_of", "dst": "A-11", "reason": "荧光寿命/跃迁是能量传递的前提"},
    {"src": "A-11", "rel": "prerequisite_of", "dst": "A-12", "reason": "能量传递/交叉弛豫导致浓度猝灭"},
    {"src": "A-10", "rel": "prerequisite_of", "dst": "A-13", "reason": "无辐射弛豫是热猝灭机制"},
    {"src": "A-11", "rel": "prerequisite_of", "dst": "A-13", "reason": "能量传递与热猝灭共享无辐射通道"},
    # ---- 横向: analogous_to (同类机制/对比) ----
    {"src": "A-12", "rel": "analogous_to", "dst": "A-13", "reason": "浓度猝灭与热猝灭同属发光效率下降机制"},
    {"src": "A-05", "rel": "analogous_to", "dst": "A-08", "reason": "4f-4f 与 4f-5d 是 Dy3+ 两种跃迁通道对比"},
    {"src": "B-01", "rel": "analogous_to", "dst": "B-02", "reason": "氟化物/磷酸盐两种基质体系对比"},
    {"src": "B-02", "rel": "analogous_to", "dst": "B-03", "reason": "磷酸盐/铝酸盐两种基质体系对比"},
    {"src": "C-01", "rel": "analogous_to", "dst": "C-02", "reason": "固相法与共沉淀法对比"},
    {"src": "C-02", "rel": "analogous_to", "dst": "C-03", "reason": "共沉淀与溶胶-凝胶对比"},
    {"src": "C-03", "rel": "analogous_to", "dst": "C-04", "reason": "溶胶-凝胶与水热/溶剂热对比"},
    # ---- 横向: affects (因果/影响, 跨域) ----
    {"src": "C-08", "rel": "affects", "dst": "A-12", "reason": "掺杂配比决定掺杂浓度→浓度猝灭"},
    {"src": "C-05", "rel": "affects", "dst": "B-06", "reason": "焙烧温度→结晶度→量子效率"},
    {"src": "B-05", "rel": "affects", "dst": "B-07", "reason": "格位对称性决定 Y/B 比→色坐标"},
    {"src": "B-05", "rel": "affects", "dst": "A-06", "reason": "格位对称性决定晶体场强度"},
    {"src": "C-06", "rel": "affects", "dst": "C-09", "reason": "还原气氛影响价态与缺陷控制"},
    {"src": "B-04", "rel": "affects", "dst": "B-05", "reason": "电荷补偿影响格位占据"},
    {"src": "B-10", "rel": "affects", "dst": "B-06", "reason": "缺陷/陷阱态降低量子效率"},
    {"src": "B-11", "rel": "affects", "dst": "B-06", "reason": "表面猝灭降低纳米材料量子效率"},
    {"src": "B-01", "rel": "affects", "dst": "B-05", "reason": "基质晶格决定格位对称性"},
    {"src": "B-02", "rel": "affects", "dst": "B-05", "reason": "基质晶格决定格位对称性"},
    {"src": "B-03", "rel": "affects", "dst": "B-05", "reason": "基质晶格决定格位对称性"},
    {"src": "C-09", "rel": "affects", "dst": "B-10", "reason": "工艺参数决定缺陷/陷阱态"},
    # ---- 横向: characterized_by (机理/材料 → 表征方法) ----
    {"src": "A-05", "rel": "characterized_by", "dst": "D-03", "reason": "能级结构由发射/激发光谱表征"},
    {"src": "A-10", "rel": "characterized_by", "dst": "D-04", "reason": "荧光寿命由衰减曲线拟合测量"},
    {"src": "B-06", "rel": "characterized_by", "dst": "D-05", "reason": "量子效率由积分球绝对法测量"},
    {"src": "A-13", "rel": "characterized_by", "dst": "D-06", "reason": "热猝灭由 T50 热稳定性测量"},
    {"src": "A-12", "rel": "characterized_by", "dst": "D-07", "reason": "掺杂浓度由 ICP-OES 定量"},
    {"src": "B-07", "rel": "characterized_by", "dst": "D-08", "reason": "色坐标由色度学 CIE 计算"},
    {"src": "B-05", "rel": "characterized_by", "dst": "D-01", "reason": "格位对称性/物相由 XRD 分析"},
    {"src": "C-07", "rel": "characterized_by", "dst": "D-02", "reason": "晶粒形貌由 SEM/TEM 表征"},
    {"src": "C-05", "rel": "characterized_by", "dst": "D-01", "reason": "焙烧结晶度由 XRD 分析"},
    {"src": "B-08", "rel": "characterized_by", "dst": "D-03", "reason": "激发光谱由光谱仪测量"},
]

# 旧式编号别名 (fact_check.py 的 KP-001 等, 兼容映射)
_ALIASES: dict[str, str] = {
    "KP-001": "A-01",  # Dy3+
}

# ============================================================
# 派生结构 (旧体系, 保持旧 API 兼容)
# ============================================================

#: 域 → KP ID 列表
KP_DOMAIN_IDS: dict[str, list[str]] = {
    dom: [f"{dom}-{i:02d}" for i in range(1, count + 1)]
    for dom, count in _DOMAIN_COUNTS.items()
}

#: 全部 42 个 KP ID
ALL_KP_IDS: list[str] = [
    kp for dom in ("A", "B", "C", "D") for kp in KP_DOMAIN_IDS[dom]
]

#: KP → 域
KP_TO_DOMAIN: dict[str, str] = {
    kp: dom for dom, ids in KP_DOMAIN_IDS.items() for kp in ids
}

#: KP → 层级 (L1/L2/L3)
KP_LEVELS: dict[str, str] = {}
for dom, (l1_count, l2_count) in _DOMAIN_LEVEL_SPLIT.items():
    ids = KP_DOMAIN_IDS[dom]
    for idx, kp in enumerate(ids):
        if idx < l1_count:
            KP_LEVELS[kp] = "L1"
        elif idx < l1_count + l2_count:
            KP_LEVELS[kp] = "L2"
        else:
            KP_LEVELS[kp] = "L3"

#: KP → 名称
KP_NAMES: dict[str, str] = dict(_KP_NAMES)

#: kg_node → KP 映射 (原 NODE_TO_KP, 单点)
NODE_TO_KP: dict[str, str] = {
    node: kp for kp, nodes in _KG_NODES.items() for node in nodes
}

#: KP → kg_node 列表
KP_KG_NODES: dict[str, list[str]] = {kp: list(nodes) for kp, nodes in _KG_NODES.items()}

#: 旧式别名 → 规范 ID
KP_ALIASES: dict[str, str] = dict(_ALIASES)

#: KP 关系边表 (教学语义关系, 只读导出)
KP_EDGES: list[dict[str, str]] = list(_KP_EDGES)


# ============================================================
# 新体系 (P1a) — 6 章 → 节 → 知识点 三层树 (SSOT)
# ============================================================
# 编号方案「章.节.序号」(如 2.3.2), ID 自带层级。
# 叶节点两种形态:
#   - {"id": "1.1.1", "old": "A-01"}   -> 由旧 ID 重编号 (name/level/kg_nodes 从旧体系派生)
#   - {"id": "6.1.1", "name": "..."}   -> 新增知识点 (level=L3, kg_nodes 由 P1b/P2 补)
# 旧 A-01~D-08 降级为 aliases, 由 OLD_TO_NEW 承载迁移映射。
CHAPTERS: list[dict[str, Any]] = [
    {
        "code": "1", "name": "发光物理基础",
        "sections": [
            {"code": "1.1", "name": "稀土离子电子构型", "kps": [
                {"id": "1.1.1", "old": "A-01"}, {"id": "1.1.2", "old": "A-02"},
            ]},
            {"code": "1.2", "name": "能级与跃迁理论", "kps": [
                {"id": "1.2.1", "old": "A-03"}, {"id": "1.2.2", "old": "A-04"},
            ]},
            {"code": "1.3", "name": "晶体场作用", "kps": [
                {"id": "1.3.1", "old": "A-06"}, {"id": "1.3.2", "old": "A-09"},
            ]},
        ],
    },
    {
        "code": "2", "name": "Dy³⁺ 发光机理",
        "sections": [
            {"code": "2.1", "name": "能级与特征跃迁", "kps": [
                {"id": "2.1.1", "old": "A-05"}, {"id": "2.1.2", "old": "A-08"},
            ]},
            {"code": "2.2", "name": "荧光动力学", "kps": [
                {"id": "2.2.1", "old": "A-07"}, {"id": "2.2.2", "old": "A-10"},
            ]},
            {"code": "2.3", "name": "能量传递与猝灭", "kps": [
                {"id": "2.3.1", "old": "A-11"}, {"id": "2.3.2", "old": "A-12"}, {"id": "2.3.3", "old": "A-13"},
            ]},
        ],
    },
    {
        "code": "3", "name": "基质与材料设计",
        "sections": [
            {"code": "3.1", "name": "基质体系", "kps": [
                {"id": "3.1.1", "old": "B-01"}, {"id": "3.1.2", "old": "B-02"}, {"id": "3.1.3", "old": "B-03"},
            ]},
            {"code": "3.2", "name": "掺杂与格位", "kps": [
                {"id": "3.2.1", "old": "B-04"}, {"id": "3.2.2", "old": "B-05"},
            ]},
            {"code": "3.3", "name": "发光性能参数", "kps": [
                {"id": "3.3.1", "old": "B-06"}, {"id": "3.3.2", "old": "B-07"}, {"id": "3.3.3", "old": "B-08"},
            ]},
            {"code": "3.4", "name": "先进材料设计", "kps": [
                {"id": "3.4.1", "old": "B-09"}, {"id": "3.4.2", "old": "B-10"}, {"id": "3.4.3", "old": "B-11"},
            ]},
        ],
    },
    {
        "code": "4", "name": "合成制备工艺",
        "sections": [
            {"code": "4.1", "name": "合成方法", "kps": [
                {"id": "4.1.1", "old": "C-01"}, {"id": "4.1.2", "old": "C-02"},
                {"id": "4.1.3", "old": "C-03"}, {"id": "4.1.4", "old": "C-04"},
            ]},
            {"code": "4.2", "name": "工艺优化", "kps": [
                {"id": "4.2.1", "old": "C-05"}, {"id": "4.2.2", "old": "C-06"}, {"id": "4.2.3", "old": "C-07"},
                {"id": "4.2.4", "old": "C-08"}, {"id": "4.2.5", "old": "C-09"}, {"id": "4.2.6", "old": "C-10"},
            ]},
        ],
    },
    {
        "code": "5", "name": "表征测试技术",
        "sections": [
            {"code": "5.1", "name": "结构与形貌表征", "kps": [
                {"id": "5.1.1", "old": "D-01"}, {"id": "5.1.2", "old": "D-02"},
            ]},
            {"code": "5.2", "name": "光谱与性能表征", "kps": [
                {"id": "5.2.1", "old": "D-03"}, {"id": "5.2.2", "old": "D-04"}, {"id": "5.2.3", "old": "D-05"},
                {"id": "5.2.4", "old": "D-06"}, {"id": "5.2.5", "old": "D-07"}, {"id": "5.2.6", "old": "D-08"},
            ]},
        ],
    },
    {
        "code": "6", "name": "绿色健康照明应用",
        "sections": [
            {"code": "6.1", "name": "白光 LED 与色度", "kps": [
                {"id": "6.1.1", "name": "白光 LED 发光原理"},
                {"id": "6.1.2", "name": "显色指数 CRI 与色温"},
            ]},
            {"code": "6.2", "name": "蓝光危害与光健康", "kps": [
                {"id": "6.2.1", "name": "蓝光危害机理与防护"},
                {"id": "6.2.2", "name": "节律照明与光生物安全"},
            ]},
            {"code": "6.3", "name": "健康照明设计", "kps": [
                {"id": "6.3.1", "name": "单基质白光荧光粉设计"},
                {"id": "6.3.2", "name": "健康照明灯具与封装"},
            ]},
        ],
    },
]


def _iter_leaf() -> list[tuple[dict[str, Any], str, str]]:
    """遍历叶节点, 返回 [(leaf, chapter_code, section_code)]."""
    out: list[tuple[dict[str, Any], str, str]] = []
    for chap in CHAPTERS:
        for sec in chap["sections"]:
            for leaf in sec["kps"]:
                out.append((leaf, chap["code"], sec["code"]))
    return out


#: 章代码 → 章名
CHAPTER_LABELS: dict[str, str] = {c["code"]: c["name"] for c in CHAPTERS}

#: 节代码 → 节名
SECTION_LABELS: dict[str, str] = {
    sec["code"]: sec["name"] for c in CHAPTERS for sec in c["sections"]
}

#: 旧 ID → 新 ID (42 条迁移映射)
OLD_TO_NEW: dict[str, str] = {
    leaf["old"]: leaf["id"]
    for leaf, _ch, _sec in _iter_leaf() if "old" in leaf
}

#: 新 ID → 旧 ID
NEW_TO_OLD: dict[str, str] = {v: k for k, v in OLD_TO_NEW.items()}

#: 新 ID → 名称 (42 重编号从旧体系派生 + 6 新增显式)
NEW_KP_NAMES: dict[str, str] = {
    leaf["id"]: _KP_NAMES[leaf["old"]] if "old" in leaf else leaf["name"]
    for leaf, _ch, _sec in _iter_leaf()
}

#: 新 ID → 层级 (重编号继承旧层级, 新增 = L3 前沿)
NEW_KP_LEVELS: dict[str, str] = {
    leaf["id"]: KP_LEVELS.get(leaf["old"], "L3") if "old" in leaf else "L3"
    for leaf, _ch, _sec in _iter_leaf()
}

#: 新 ID → 章代码
NEW_KP_TO_CHAPTER: dict[str, str] = {
    leaf["id"]: ch for leaf, ch, _sec in _iter_leaf()
}

#: 新 ID → 节代码
NEW_KP_TO_SECTION: dict[str, str] = {
    leaf["id"]: sec for leaf, _ch, sec in _iter_leaf()
}

#: 全部 48 个新 ID (章序)
NEW_ALL_KP_IDS: list[str] = [leaf["id"] for leaf, _ch, _sec in _iter_leaf()]

#: 章代码 → 新 ID 列表
CHAPTER_KP_IDS: dict[str, list[str]] = {
    ch: [leaf["id"] for leaf, c, _sec in _iter_leaf() if c == ch]
    for ch in CHAPTER_LABELS
}

#: 新 ID → kg_node 列表 (重编号继承旧 kg_node 关联; 新增暂空, P1b/P2 补)
NEW_KP_KG_NODES: dict[str, list[str]] = {
    leaf["id"]: _KG_NODES.get(leaf["old"], []) if "old" in leaf else []
    for leaf, _ch, _sec in _iter_leaf()
}

#: kg_node → 新 ID 映射 (供 P4 迁移后 practice/l3 使用)
NEW_NODE_TO_KP: dict[str, str] = {
    node: kp for kp, nodes in NEW_KP_KG_NODES.items() for node in nodes
}

# ---- 新体系关系边: 44 条重编号 + 第 6 章应用主线 (applies_to) ----

# 第 6 章应用主线 (P1a 最小布线, P2 再 LLM 补边)
_KP_EDGES_NEW_CH6: list[dict[str, str]] = [
    {"src": "6.1.1", "rel": "prerequisite_of", "dst": "6.3.2", "reason": "白光 LED 原理是健康照明灯具设计的前提"},
    {"src": "6.1.2", "rel": "prerequisite_of", "dst": "6.3.2", "reason": "显色指数/色温是健康照明设计指标"},
    {"src": "6.2.1", "rel": "prerequisite_of", "dst": "6.3.2", "reason": "蓝光危害防护是健康照明设计约束"},
    {"src": "3.3.2", "rel": "applies_to", "dst": "6.1.2", "reason": "色坐标/色纯度是显色指数计算基础"},
    {"src": "3.3.1", "rel": "applies_to", "dst": "6.3.1", "reason": "量子效率是白光荧光粉设计核心指标"},
    {"src": "6.1.1", "rel": "applies_to", "dst": "6.3.1", "reason": "白光 LED 原理支撑单基质白光荧光粉设计"},
]

#: 新体系关系边表 (44 旧边重编号 + 6 条第 6 章应用边)
NEW_KP_EDGES: list[dict[str, str]] = [
    {
        "src": OLD_TO_NEW[e["src"]],
        "rel": e["rel"],
        "dst": OLD_TO_NEW[e["dst"]],
        "reason": e["reason"],
    }
    for e in _KP_EDGES
] + _KP_EDGES_NEW_CH6


# ============================================================
# 查询函数
# ============================================================

def kp_prerequisites(kp_id: str) -> list[str]:
    """返回 kp_id 的直接前提 KP (prerequisite_of 的 src), 用于纵向溯源."""
    return [e["src"] for e in _KP_EDGES if e["rel"] == "prerequisite_of" and e["dst"] == kp_id]


def kp_deepens(kp_id: str) -> list[str]:
    """返回 kp_id 直接深化的 KP (即以 kp_id 为前提的那些 dst), 用于纵向钻深."""
    return [e["dst"] for e in _KP_EDGES if e["rel"] == "prerequisite_of" and e["src"] == kp_id]


def kp_neighbors(kp_id: str) -> list[dict[str, str]]:
    """返回 kp_id 的所有直接邻居边 (含关系类型与方向).

    每条边归一为 {rel, other, direction, reason}:
    - direction="out": kp_id --rel--> other
    - direction="in":  other --rel--> kp_id (仅 prerequisite_of 转译为 deepens)
    analogous_to 视为对称, 统一 direction="out"。
    """
    out: list[dict[str, str]] = []
    for e in _KP_EDGES:
        rel = e["rel"]
        if e["src"] == kp_id:
            out.append({"rel": rel, "other": e["dst"], "direction": "out", "reason": e["reason"]})
        elif e["dst"] == kp_id and rel == "prerequisite_of":
            out.append({"rel": "deepens", "other": e["src"], "direction": "in", "reason": e["reason"]})
        elif e["dst"] == kp_id and rel == "analogous_to":
            out.append({"rel": rel, "other": e["src"], "direction": "out", "reason": e["reason"]})
        elif e["dst"] == kp_id:
            out.append({"rel": rel, "other": e["src"], "direction": "in", "reason": e["reason"]})
    return out


def expand_kp(kp_id: str, max_hop: int = 2) -> list[dict[str, str]]:
    """以 kp_id 为中心做 BFS 多跳拓展, 返回可达邻居知识点列表.

    每个元素: {kp_id, rel, direction, hop, via, reason}。用于问答时沿知识点
    关系召回横向(类比/因果/表征)与纵向(前提/深化)相关知识点, 实现「知识拓展」。
    """
    if max_hop < 1:
        return []
    visited: dict[str, int] = {kp_id: 0}
    frontier = [kp_id]
    result: list[dict[str, str]] = []
    for hop in range(1, max_hop + 1):
        nxt: list[str] = []
        for cur in frontier:
            for nb in kp_neighbors(cur):
                other = nb["other"]
                if other in visited:
                    continue
                visited[other] = hop
                nxt.append(other)
                result.append({
                    "kp_id": other,
                    "rel": nb["rel"],
                    "direction": nb["direction"],
                    "hop": str(hop),
                    "via": cur,
                    "reason": nb.get("reason", ""),
                })
        frontier = nxt
    return result


def kp_name(kp_id: str) -> str:
    """知识点 ID → 名称 (兼容旧 ID 与新 ID, 未知 ID 返回原样)."""
    return NEW_KP_NAMES.get(kp_id) or _KP_NAMES.get(kp_id, kp_id)


def kp_domain(kp_id: str) -> str:
    """知识点 ID → 域代码 (旧 ID; 新 ID 返回对应章代码, 见 kp_chapter)."""
    return KP_TO_DOMAIN.get(kp_id, "")


def kp_level(kp_id: str) -> str:
    """知识点 ID → 层级 (L1/L2/L3, 兼容旧/新 ID)."""
    return NEW_KP_LEVELS.get(kp_id) or KP_LEVELS.get(kp_id, "")


def kp_chapter(kp_id: str) -> str:
    """知识点 ID → 章代码 (新 ID 直接取; 旧 ID 经迁移映射转换)."""
    nid = to_new_id(kp_id)
    return NEW_KP_TO_CHAPTER.get(nid, "")


def kp_section(kp_id: str) -> str:
    """知识点 ID → 节代码 (新 ID 直接取; 旧 ID 经迁移映射转换)."""
    nid = to_new_id(kp_id)
    return NEW_KP_TO_SECTION.get(nid, "")


def to_new_id(identifier: str) -> str:
    """任意标识 → 新 ID (章.节.序号).

    - 新 ID 原样返回
    - 旧 ID (A-01~D-08) → OLD_TO_NEW
    - 旧式别名 (KP-001) → 先归一旧 ID 再转新 ID
    """
    if identifier in NEW_KP_NAMES:
        return identifier
    old = _ALIASES.get(identifier, identifier)
    return OLD_TO_NEW.get(old, old)


def to_old_id(identifier: str) -> str:
    """任意标识 → 旧 ID (A-01~D-08); 新增知识点 (无旧 ID) 返回原新 ID."""
    if identifier in NEW_TO_OLD:
        return NEW_TO_OLD[identifier]
    if identifier in _ALIASES:
        return _ALIASES[identifier]
    return identifier


def resolve_kp(identifier: str) -> str:
    """解析知识点标识: 规范 ID 原样返回, 旧式别名 (KP-001) 归一为旧规范 ID.

    注: 保持返回旧 ID 语义 (迁移前下游仍按旧 ID 取数); 新 ID 转换用 to_new_id。
    """
    if identifier in KP_NAMES:
        return identifier
    return KP_ALIASES.get(identifier, identifier)


def to_dict() -> dict[str, Any]:
    """完整目录 (供 GET /l2/kp-catalog 输出).

    返回旧体系视图 (domains/kp/total) + 新体系 6 章树 (chapters), 前后兼容。
    """
    return {
        # ---- 旧体系视图 (兼容) ----
        "domains": [
            {
                "code": dom,
                "label": DOMAIN_LABELS[dom],
                "kp_ids": KP_DOMAIN_IDS[dom],
            }
            for dom in ("A", "B", "C", "D")
        ],
        "kp": [
            {
                "kp_id": kp,
                "name": _KP_NAMES[kp],
                "domain": KP_TO_DOMAIN[kp],
                "level": KP_LEVELS[kp],
                "kg_nodes": list(_KG_NODES.get(kp, [])),
                "covered_by_bank": bool(_KG_NODES.get(kp)),
            }
            for kp in ALL_KP_IDS
        ],
        "total": len(ALL_KP_IDS),
        # ---- 新体系视图 (P1a) ----
        "chapters": [
            {
                "code": chap["code"],
                "name": chap["name"],
                "sections": [
                    {
                        "code": sec["code"],
                        "name": sec["name"],
                        "kps": [
                            {
                                "kp_id": leaf["id"],
                                "name": NEW_KP_NAMES[leaf["id"]],
                                "level": NEW_KP_LEVELS[leaf["id"]],
                                "old_id": leaf.get("old"),
                                "is_new": "old" not in leaf,
                            }
                            for leaf in sec["kps"]
                        ],
                    }
                    for sec in chap["sections"]
                ],
            }
            for chap in CHAPTERS
        ],
        "new_total": len(NEW_ALL_KP_IDS),
    }
