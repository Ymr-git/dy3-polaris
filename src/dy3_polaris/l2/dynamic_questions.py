"""L2 动态出题引擎 — 基于知识库通用知识的模板化变题.

设计(参照成熟组卷/评测方案, 如自动组卷系统与 ExamGen 思路):
1. **知识点不变, 题目变化**: 每知识点配置 2-3 组事实与多套题干模板,
   每次出题随机抽取模板/题型 → 同一知识点题目与问法都不同, 实现活学活用。
2. **题型变换**: choice(单选) / judge(判断) / blank(填空) / multi(多选) 随机轮换。
3. **正确选项随机分布**: 生成后洗牌选项, 答案索引跟随 (不固定第一个)。
4. **保真原则**: 题干/正确项/干扰项全部来自领域事实表 (与 pretest_bank 及
   知识库 seed 数据一致), 不编造数值; 干扰项取自同属性其他真实值。

对外接口:
- ``DynamicQuestionEngine.generate(kp_template)`` → 生成 1 道变体题
- ``generate_for_learner(learner_id, count, mastery, bank)`` → 薄弱 KP 优先变题
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from .kp_catalog import NODE_TO_KP, kp_domain

_logger = logging.getLogger("dy3_polaris.l2.dynamic_questions")

#: 题型权重 (choice 高频, judge/blank/multi 轮换, graph 为图谱关系题)
TYPE_POOL = ["choice", "choice", "judge", "blank", "multi", "graph"]

#: 题干模板 (与具体事实无关的问法变体)
QSTEM_CHOICE = [
    "关于{topic}，下列说法正确的是？",
    "下列关于{topic}的描述，正确的是？",
    "以下对{topic}的表述中，符合实际的是？",
]
QSTEM_JUDGE_TRUE = [
    "判断正误：{statement_true}",
    "请判断以下说法是否正确：{statement_true}",
]
QSTEM_JUDGE_FALSE = [
    "判断正误：{statement_false}",
    "请判断以下说法是否正确：{statement_false}",
]
QSTEM_BLANK = [
    "{blank_prompt} 请填空（填写规范符号，如 4f⁹ / 480 nm）。",
    "补全填空：{blank_prompt} ____",
]
QSTEM_MULTI = [
    "下列关于{topic}的说法，正确的有？（多选）",
    "以下关于{topic}的表述，正确的有哪些？（多选）",
]
QSTEM_GRAPH = [
    "在「{topic}」的知识图谱中，从中心概念出发的正确关系是？",
    "请根据知识图谱判断，与「{topic}」正确相连的结论是？",
]

#: 知识点 → 出题模板 (topic / 事实组 / 干扰项 / 填空 / 判断命题 / 多选事实对)
#: 事实均来自 pretest_bank.json 与知识库 seed (已核对, 不编造)
KP_QUESTION_TEMPLATES: dict[str, dict[str, Any]] = {
    "Dy3+": {
        "topic": "Dy³⁺ 的电子组态与发光机理",
        "facts": [
            ("Dy³⁺ 的发光源自 4f 电子跃迁，4f 电子被外层 5s²5p⁶ 屏蔽，因此发光受基质晶体场影响较小",
             "4f 电子被外层 5s²5p⁶ 屏蔽", ["4f 电子裸露无屏蔽", "5d 电子主导发光", "6s 电子主导发光"]),
            ("4f→4f 跃迁属宇称禁戒跃迁，使稀土发光具有窄谱线、长寿命的特征",
             "宇称禁戒 → 窄谱线、长寿命", ["宇称允许 → 宽带发光", "4f→4f 为电偶极允许跃迁", "发光寿命极短"]),
        ],
        "judge_true": "Dy³⁺ 发光受晶体场影响较小，是因为 4f 电子被外层 5s²5p⁶ 电子屏蔽",
        "judge_false": "Dy³⁺ 的发光主要由外层 5s 电子的跃迁产生，而不是 4f 电子",
        "blank_prompt": "Dy³⁺ 发光受晶体场影响较小的根本原因是 4f 电子被外层",
        "blank_answer": "5s²5p⁶",
        "multi_facts": [
            ("Dy³⁺ 发光源于 4f 电子跃迁", True),
            ("4f 电子被外层 5s²5p⁶ 屏蔽，发光受晶体场影响小", True),
            ("Dy³⁺ 发光主要由 5s 电子跃迁产生", False),
            ("4f→4f 跃迁为宇称允许跃迁", False),
        ],
    },
    "F9_2": {
        "topic": "Dy³⁺ 白光发射的跃迁机理",
        "facts": [
            ("Dy³⁺ 白光来自 ⁴F₉/₂ 激发态向 ⁶H₁₅/₂、⁶H₁₃/₂、⁶H₁₁/₂ 等多个基态能级的跃迁，蓝/黄/红多带叠加而成",
             "多个 ⁶H_J 基态能级的跃迁叠加", ["单一能级跃迁", "连续光谱", "黑体辐射"]),
            ("蓝/黄/红三带的相对强度随格位对称性改变，因此可实现色温可调的白光",
             "三带相对强度随格位对称性改变", ["三带强度固定不变", "只与温度有关", "只与掺杂浓度有关"]),
        ],
        "judge_true": "Dy³⁺ 白光来自 ⁴F₉/₂ 向多个 ⁶H_J 基态能级的跃迁叠加，而非单一能级跃迁",
        "judge_false": "Dy³⁺ 的白光发射来自单一能级的跃迁",
        "blank_prompt": "Dy³⁺ 白光发射的机理是 ⁴F₉/₂ 激发态向多个",
        "blank_answer": "6H_J 基态能级",
        "multi_facts": [
            ("白光由蓝/黄/红多带跃迁叠加而成", True),
            ("三带相对强度随格位对称性改变", True),
            ("白光来自单一能级跃迁", False),
            ("三带相对强度固定不变", False),
        ],
    },
    "H15_2": {
        "topic": "Dy³⁺ 蓝光跃迁的机理",
        "facts": [
            ("⁴F₉/₂→⁶H₁₅/₂ 为磁偶极跃迁，其强度对格位对称性不敏感",
             "磁偶极跃迁，对对称性不敏感", ["电偶极跃迁", "对对称性极敏感", "声子辅助跃迁"]),
        ],
        "judge_true": "⁴F₉/₂→⁶H₁₅/₂ 为磁偶极跃迁，蓝光强度对格位对称性不敏感",
        "judge_false": "⁴F₉/₂→⁶H₁₅/₂ 为电偶极跃迁，蓝光强度对格位对称性高度敏感",
        "blank_prompt": "⁴F₉/₂→⁶H₁₅/₂ 属于（磁偶极/电偶极）",
        "blank_answer": "磁偶极",
        "multi_facts": [
            ("⁴F₉/₂→⁶H₁₅/₂ 为磁偶极跃迁", True),
            ("蓝光强度对格位对称性不敏感", True),
            ("蓝光跃迁为超灵敏跃迁", False),
        ],
    },
    "H13_2": {
        "topic": "超灵敏跃迁的机理",
        "facts": [
            ("⁴F₉/₂→⁶H₁₃/₂ 是 ΔJ=2 的超灵敏跃迁，对 Dy³⁺ 所处格位对称性高度敏感",
             "超灵敏跃迁，对格位对称性高度敏感", ["普通电偶极跃迁", "磁偶极跃迁", "对对称性无响应"]),
            ("因此黄光强度可随基质与格位微环境显著变化，是调控光色的关键",
             "黄光强度随格位微环境显著变化", ["黄光强度恒定", "黄光只随温度变", "黄光与格位无关"]),
        ],
        "judge_true": "⁴F₉/₂→⁶H₁₃/₂ 是超灵敏跃迁，对格位对称性高度敏感",
        "judge_false": "⁴F₉/₂→⁶H₁₃/₂ 是磁偶极跃迁，对格位对称性不敏感",
        "blank_prompt": "⁴F₉/₂→⁶H₁₃/₂ 被称为超灵敏跃迁，是因为它对",
        "blank_answer": "格位对称性",
        "multi_facts": [
            ("⁴F₉/₂→⁶H₁₃/₂ 为超灵敏跃迁", True),
            ("超灵敏跃迁对格位对称性高度敏感", True),
            ("超灵敏跃迁与格位无关", False),
            ("黄光强度不随基质变化", False),
        ],
    },
    "H11_2": {
        "topic": "红光跃迁与显色机理",
        "facts": [
            ("红光跃迁（⁴F₉/₂→⁶H₁₁/₂）补充长波成分，对显色指数 R9（饱和红还原）有贡献",
             "补充长波成分，提升 R9", ["产生蓝光", "降低显色性", "与显色无关"]),
        ],
        "judge_true": "红光跃迁补充长波成分，对显色指数 R9 有贡献",
        "judge_false": "红光跃迁对显色指数没有贡献",
        "blank_prompt": "红光跃迁（⁴F₉/₂→⁶H₁₁/₂）对显色指数中的",
        "blank_answer": "R9",
        "multi_facts": [
            ("红光跃迁补充长波成分", True),
            ("红光跃迁对 R9 有贡献", True),
            ("红光跃迁产生蓝光", False),
        ],
    },
    "YB_ratio": {
        "topic": "Y/B 比与格位对称性的机理",
        "facts": [
            ("Dy³⁺ 占据非反演对称中心格位时，超灵敏黄光跃迁增强，使 Y/B > 1、光色偏暖",
             "非反演对称格位增强超灵敏黄光", ["反演对称格位增强黄光", "Y/B 与格位无关", "黄光始终强于蓝光"]),
        ],
        "judge_true": "非反演对称中心格位会增强超灵敏黄光跃迁，使 Y/B > 1",
        "judge_false": "Y/B 比与 Dy³⁺ 所处格位对称性无关",
        "blank_prompt": "Dy³⁺ 占据非反演对称中心格位时，超灵敏黄光增强，使 Y/B",
        "blank_answer": "> 1",
        "multi_facts": [
            ("非反演对称格位增强超灵敏黄光", True),
            ("Y/B > 1 时黄光主导、光色偏暖", True),
            ("Y/B 与格位对称性无关", False),
            ("反演对称格位增强黄光", False),
        ],
    },
    "concentration_quench": {
        "topic": "浓度猝灭的机理",
        "facts": [
            ("掺杂浓度过高时离子间距减小，交叉弛豫（偶极-偶极相互作用）增强，激发能转为无辐射损耗导致猝灭",
             "交叉弛豫（偶极-偶极相互作用）", ["电四极相互作用", "热激活穿越", "表面缺陷散射"]),
            ("因此存在最优掺杂浓度：低于最优浓度发光中心不足，高于则猝灭加剧",
             "最优浓度是发光中心数与猝灭的平衡", ["浓度越高发光越强", "浓度与发光无关", "浓度越低越好"]),
        ],
        "judge_true": "浓度猝灭的主要机制是离子间距减小导致的交叉弛豫（偶极-偶极相互作用）增强",
        "judge_false": "掺杂浓度越高，发光强度单调上升，不存在浓度猝灭",
        "blank_prompt": "Dy³⁺ 浓度过高时发光猝灭的主要机制是",
        "blank_answer": "交叉弛豫",
        "multi_facts": [
            ("浓度猝灭源于交叉弛豫增强", True),
            ("存在最优掺杂浓度", True),
            ("浓度越高发光越强，无猝灭", False),
            ("浓度猝灭与离子间距无关", False),
        ],
    },
    "CNYP": {
        "topic": "基质格位取代的机理",
        "facts": [
            ("Dy³⁺ 进入 Ca₇NaY(PO₄)₆ 时取代 Y³⁺ 格位，因为二者价态相同、离子半径相近",
             "价态相同、半径相近", ["取代 Ca²⁺", "取代 P⁵⁺", "进入晶格间隙"]),
        ],
        "judge_true": "Dy³⁺ 取代 Y³⁺ 格位，是因为二者价态相同、离子半径相近",
        "judge_false": "Dy³⁺ 进入 Ca₇NaY(PO₄)₆ 时取代 P⁵⁺ 格位",
        "blank_prompt": "Dy³⁺ 进入 Ca₇NaY(PO₄)₆ 时取代的格位是",
        "blank_answer": "Y³⁺",
        "multi_facts": [
            ("Dy³⁺ 取代 Y³⁺ 格位", True),
            ("取代原因是价态相同、半径相近", True),
            ("Dy³⁺ 取代 P⁵⁺ 格位", False),
        ],
    },
    "synthesis": {
        "topic": "高温固相法合成的机理",
        "facts": [
            ("高温固相法靠高温下离子的固态扩散使掺杂离子进入晶格；预烧先分解原料、去除 CO₂/H₂O",
             "高温固态扩散 + 预烧除杂", ["液相反应", "气相沉积", "无需预烧"]),
        ],
        "judge_true": "高温固相法依赖高温下离子的固态扩散，预烧用于去除 CO₂/H₂O",
        "judge_false": "高温固相法合成无需预烧，一步烧结即可",
        "blank_prompt": "高温固相法合成中，预烧的主要目的是去除",
        "blank_answer": "CO₂ 和 H₂O",
        "multi_facts": [
            ("高温固相法依赖固态扩散", True),
            ("预烧用于去除 CO₂/H₂O", True),
            ("高温固相法为液相反应", False),
            ("无需预烧一步烧结即可", False),
        ],
    },
    "xrd": {
        "topic": "XRD 表征的机理",
        "facts": [
            ("XRD 依据布拉格衍射测量晶格周期性，故用于分析晶体结构与相纯度",
             "布拉格衍射 → 晶体结构/相纯度", ["测量发光强度", "测量量子效率", "测量色坐标"]),
        ],
        "judge_true": "XRD 依据布拉格衍射分析晶体结构与相纯度",
        "judge_false": "XRD 用于直接测量荧光粉的量子效率",
        "blank_prompt": "XRD 依据布拉格衍射，主要用于分析荧光粉的",
        "blank_answer": "晶体结构和相纯度",
        "multi_facts": [
            ("XRD 分析晶体结构与相纯度", True),
            ("XRD 依据布拉格衍射", True),
            ("XRD 直接测量量子效率", False),
            ("XRD 测量发光强度", False),
        ],
    },
    "pl": {
        "topic": "PL 光谱表征的机理",
        "facts": [
            ("PL 光谱用固定激发波长激发样品、记录发射光谱，得到各跃迁带的波长与强度",
             "激发 + 发射光谱 → 跃迁带波长/强度", ["测晶格常数", "测形貌", "测电导率"]),
        ],
        "judge_true": "PL 光谱通过记录发射光谱得到各跃迁带的波长与强度",
        "judge_false": "PL 光谱用于测量荧光粉的晶体结构与相纯度",
        "blank_prompt": "PL 光谱主要用于获得发光材料的",
        "blank_answer": "发射光谱",
        "multi_facts": [
            ("PL 得到各跃迁带波长与强度", True),
            ("PL 需用固定波长激发", True),
            ("PL 测晶体结构", False),
            ("PL 测形貌", False),
        ],
    },
    "cie": {
        "topic": "CIE 色度的机理",
        "facts": [
            ("CIE 色度坐标把发光颜色映射到二维平面，用于量化颜色位置与白平衡",
             "量化颜色位置", ["测发光强度", "测寿命", "测粒径"]),
        ],
        "judge_true": "CIE 色度坐标用于量化发光颜色位置",
        "judge_false": "CIE 色度坐标用于测量荧光粉的粒径分布",
        "blank_prompt": "CIE 色度坐标用于量化发光材料的",
        "blank_answer": "颜色位置",
        "multi_facts": [
            ("CIE 坐标量化颜色位置", True),
            ("CIE 坐标不测发光强度", True),
            ("CIE 坐标测粒径", False),
        ],
    },
    "thermal": {
        "topic": "热猝灭的机理",
        "facts": [
            ("温度升高使无辐射跃迁概率增大（热激活过程），发光强度下降，即热猝灭",
             "无辐射跃迁概率增大", ["辐射跃迁增强", "吸收增强", "与温度无关"]),
        ],
        "judge_true": "热猝灭的机理是温度升高使无辐射跃迁概率增大",
        "judge_false": "温度升高时辐射跃迁增强，发光强度随之升高",
        "blank_prompt": "热猝灭的机理是温度升高使",
        "blank_answer": "无辐射跃迁",
        "multi_facts": [
            ("热猝灭源于无辐射跃迁增强", True),
            ("无辐射跃迁随温度升高而增强", True),
            ("温度升高辐射跃迁增强", False),
        ],
    },
    "led_pack": {
        "topic": "单基质白光的机理",
        "facts": [
            ("Dy³⁺ 单掺杂即可实现白光，因为其蓝 + 黄（+ 红）多带发射叠加互补成白光",
             "多带发射叠加互补成白光", ["只能发单色光", "需双基质混合", "靠散射产生白光"]),
        ],
        "judge_true": "Dy³⁺ 单掺杂可实现白光，源于多带发射叠加互补",
        "judge_false": "Dy³⁺ 单掺杂只能发出单色光，无法实现白光",
        "blank_prompt": "Dy³⁺ 单掺杂可实现白光的机理是",
        "blank_answer": "多带叠加互补",
        "multi_facts": [
            ("单掺杂可实现白光", True),
            ("白光源于多带叠加互补", True),
            ("Dy³⁺ 只能发单色光", False),
            ("需双基质混合才能白光", False),
        ],
    },
    "std_bluelight": {
        "topic": "蓝光危害的机理",
        "facts": [
            ("蓝光危害源于高能短波蓝光对视网膜的光化学损伤；配合近紫外芯片可减少蓝光成分",
             "高能短波蓝光的光化学损伤", ["红光损伤", "红外热损伤", "紫外线无害"]),
        ],
        "judge_true": "蓝光危害源于高能短波蓝光对视网膜的光化学损伤",
        "judge_false": "蓝光危害与蓝光波长无关，任何波段危害相同",
        "blank_prompt": "蓝光危害源于高能短波蓝光对视网膜的",
        "blank_answer": "光化学损伤",
        "multi_facts": [
            ("蓝光危害源于光化学损伤", True),
            ("近紫外芯片可减少蓝光成分", True),
            ("蓝光危害与波长无关", False),
        ],
    },
    "std_cri": {
        "topic": "显色性的机理",
        "facts": [
            ("显色指数 Ra 反映光源还原物体真实颜色的能力；Ra 不足会导致颜色失真",
             "还原物体真实颜色的能力", ["亮度指标", "色温指标", "寿命指标"]),
        ],
        "judge_true": "显色指数 Ra 反映光源还原物体真实颜色的能力",
        "judge_false": "显色指数 Ra 是衡量光源亮度的指标",
        "blank_prompt": "显色指数 Ra 反映光源",
        "blank_answer": "还原物体真实颜色",
        "multi_facts": [
            ("Ra 反映颜色还原能力", True),
            ("Ra 不足导致颜色失真", True),
            ("Ra 是亮度指标", False),
        ],
    },
}

#: 未配置模板的节点 → 回退静态题库 (由 PracticeBank 处理)
MISSING_NODE_TOPIC = "该知识点"


class DynamicQuestionEngine:
    """基于模板与事实的动态出题引擎 (线程安全: 每次生成使用独立随机源)."""

    def __init__(self, bank: Any) -> None:
        self._bank = bank
        self._lock = bank._lock if hasattr(bank, "_lock") else None

    def _kp_of(self, node: str) -> str:
        return NODE_TO_KP.get(node, node)

    def generate(self, node: str, rng: random.Random | None = None) -> dict[str, Any] | None:
        """为知识点节点生成 1 道变体题 (随机题型/模板/选项分布)."""
        tpl = KP_QUESTION_TEMPLATES.get(node)
        if tpl is None:
            return None
        r = rng or random.Random()
        qtype = r.choice(TYPE_POOL)
        topic = tpl["topic"]
        qid = f"dyn-{node}-{int(time.time() * 1000)}-{r.randrange(100000)}"

        if qtype == "judge":
            use_true = r.random() < 0.6
            stem = r.choice(QSTEM_JUDGE_TRUE if use_true else QSTEM_JUDGE_FALSE)
            if use_true:
                question = stem.format(statement_true=tpl["judge_true"])
                options = ["正确", "错误"]
                answer = 0
            else:
                question = stem.format(statement_false=tpl["judge_false"])
                options = ["正确", "错误"]
                answer = 1
            explanation = "判断：该命题表述正确" if use_true else "判断：该命题表述错误（见知识点解析）"
            return self._finalize(qid, node, qtype, question, options, answer,
                                   explanation, topic, difficulty=1)

        if qtype == "blank":
            prompt = tpl.get("blank_prompt", f"{topic}的填空")
            question = r.choice(QSTEM_BLANK).format(blank_prompt=prompt)
            return {
                "qid": qid, "kg_node": node, "kp_id": self._kp_of(node),
                "kg_name": topic, "difficulty": 1, "type": "blank",
                "question": question, "options": [],
                "answer": -1, "answer_text": tpl.get("blank_answer", ""),
                "explanation": f"正确答案：{tpl.get('blank_answer', '')}（来自知识库知识点：{topic}）",
                "ability_dim": "theory", "dynamic": True,
            }

        if qtype == "multi":
            pairs = tpl.get("multi_facts") or []
            stem = r.choice(QSTEM_MULTI).format(topic=topic)
            # 打乱顺序, 答案索引跟随
            shuffled = list(pairs)
            r.shuffle(shuffled)
            options = [p[0] for p in shuffled]
            answers = [i for i, p in enumerate(shuffled) if p[1]]
            return {
                "qid": qid, "kg_node": node, "kp_id": self._kp_of(node),
                "kg_name": topic, "difficulty": 2, "type": "multi",
                "question": stem, "options": options,
                "answer": answers,
                "explanation": "多选：应选择所有正确表述（见知识点解析）",
                "ability_dim": "theory", "dynamic": True,
            }

        if qtype == "graph":
            # 图谱关系题: 中心概念 + 可选关系边 (复用前端 SVG 图形渲染)
            pairs = tpl.get("multi_facts") or []
            truths = [p[0] for p in pairs if p[1]]
            falses = [p[0] for p in pairs if not p[1]]
            if not truths:
                return None
            correct = r.choice(truths)
            options = [correct] + falses
            r.shuffle(options)
            answer = options.index(correct)
            question = r.choice(QSTEM_GRAPH).format(topic=topic)
            return {
                "qid": qid, "kg_node": node, "kp_id": self._kp_of(node),
                "kg_name": topic, "difficulty": 2, "type": "graph",
                "question": question, "options": options,
                "answer": answer,
                "graph": {"center": topic, "edges": options},
                "explanation": f"正确关系：{correct}（见知识点解析）",
                "ability_dim": "theory", "dynamic": True,
            }

        # choice (默认): 随机事实 + 随机问法 + 选项洗牌
        fact, correct, distractors = r.choice(tpl["facts"])
        stem = r.choice(QSTEM_CHOICE).format(topic=topic)
        question = stem
        options = [correct] + list(distractors)
        r.shuffle(options)
        answer = options.index(correct)
        return self._finalize(qid, node, "choice", question, options, answer,
                               correct, topic, difficulty=2)

    def _finalize(self, qid, node, qtype, question, options, answer, explanation, topic, difficulty):
        return {
            "qid": qid, "kg_node": node, "kp_id": self._kp_of(node),
            "kg_name": topic, "difficulty": difficulty, "type": qtype,
            "question": question, "options": options,
            "answer": answer,
            "explanation": explanation,
            "ability_dim": "theory", "dynamic": True,
        }

    def select_and_generate(
        self,
        learner_id: str,
        count: int = 12,
        mastery: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """四维覆盖变题 (A/B/C/D 各 count/4 题, 维度内薄弱优先).

        覆盖能力维度 (理论/体系/工艺/表征), 而非只挑薄弱点, 一次练习得到完整画像.
        """
        r = random.Random(f"{learner_id}:{int(time.time())}")
        mastery = mastery or {}
        selected: list[dict[str, Any]] = []
        used_nodes: set[str] = set()
        per_domain = max(1, count // 4)

        def _dom(node: str) -> str:
            return kp_domain(NODE_TO_KP.get(node, ""))

        # 1. 四维覆盖: 每维按薄弱优先出 per_domain 题
        for dom in ("A", "B", "C", "D"):
            got = 0
            dom_nodes = sorted(
                (node for node in KP_QUESTION_TEMPLATES if _dom(node) == dom),
                key=lambda n: mastery.get(NODE_TO_KP.get(n, ""), 0.0),
            )
            for node in dom_nodes:
                if got >= per_domain or node in used_nodes:
                    continue
                q = self.generate(node, r)
                if q is None:
                    continue
                selected.append(q)
                used_nodes.add(node)
                got += 1

        # 2. 补齐到 count (某维度节点不足时, 用其余节点随机补)
        if len(selected) < count:
            rest = [n for n in KP_QUESTION_TEMPLATES if n not in used_nodes]
            r.shuffle(rest)
            for node in rest:
                if len(selected) >= count:
                    break
                q = self.generate(node, r)
                if q is not None:
                    selected.append(q)
                    used_nodes.add(node)

        # 3. 注册进题库 (判题按 qid 命中), 线程安全
        if self._lock is not None:
            with self._lock:
                for q in selected:
                    self._bank.by_qid[q["qid"]] = q
        else:
            for q in selected:
                self._bank.by_qid[q["qid"]] = q

        _logger.info("动态出题 learner=%s count=%d (四维覆盖)", learner_id, len(selected))
        return selected
