"""Dy 垂直领域知识库批量构建脚本 — 从 wxk 文献 Markdown 建知识图谱.

设计目标:
1. 从 wxk 文件夹筛选「绿色健康照明发光材料 + Dy 垂直领域」贴题文献.
2. 注入 Dy 领域词典, 提升实体识别质量.
3. 用 KnowledgeGraphBuilder 建「实体 + 三元组」轻量图谱 (track_version=False 防膨胀).
4. 不调用 save_snapshot (避免重蹈 8.6GB 膨胀覆辙).

运行方式:
    python scripts/build_dy_knowledge_graph.py
    python scripts/build_dy_knowledge_graph.py --dry-run   # 只统计不建图
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# wxk 文献根目录 (真实位置: D:\BaiduNetdiskDownload\dy3-polaris-整理后\wxk,
# 与项目根 D:\...\xiaotiao\dy3-polaris-整理后 是两条独立目录树)
_WXK_ROOT = Path(r"D:\BaiduNetdiskDownload\dy3-polaris-整理后\wxk")

# 贴题关键词 (绿色健康照明 + 稀土发光材料 + Dy 垂直领域)
_TOPIC_INCLUDE = re.compile(
    r"(dy|镝|白光|照明|荧光粉|发光|荧光|磷光|led|light|phosphor|luminescen|"
    r"固态|solid.?state|稀土|rare.?earth|lanthanide|铕|eu3?\+|铽|tb3?\+|"
    r"掺|dop|基质|host|晶体场|crystal.?field|judd|ofelt|能级|跃迁|transition)"
)

# 偏题关键词 (纯生物医学/免疫/温度传感等, 剔除)
_TOPIC_EXCLUDE = re.compile(
    r"(生物成像|bioimaging|免疫|immuno|生物检测|biosensor|bio.?detect|"
    r"温度传感|temperature.?sens|thermometry|温度测量|生物探针|probe|"
    r"造影|contrast|肿瘤|cancer|细胞|cell|dna|rna|药物|drug|诊疗|thera)"
)

# Dy 领域词典: 高频实体 + 类型
_DY_ENTITY_DICTIONARY: dict[str, list[str]] = {
    "chemical_compound": [
        "Dy3+", "Dy", "镝", "Eu3+", "Eu", "铕", "Tb3+", "Tb", "铽",
        "YAG", "Y3Al5O12", "Y2O3", "Gd2O3", "La2O3", "Lu2O3",
        "Ce3+", "Sm3+", "Nd3+", "Yb3+", "Er3+",
    ],
    "material": [
        "NaM4(VO4)3", "Ca7NaY(PO4)6", "NaGdF4", "YPO4", "BaMgAl10O17",
        "Sr3Bi(PO4)3", "Ca2Ga2GeO7", "KLa(PO3)4", "KSrPO4", "Sr3Gd(PO4)3",
        "Ca9Bi(PO4)7", "CaAl2O4", "Bi4Si3O12", "Ba3Lu(PO4)3", "MgB4O7",
        "Ca3(PO4)2", "CaWO4", "YVO4", "GdVO4", "YAlO3", "SrAl2O4",
        "YAG:Ce", "BaSO4", "CaF2", "SrF2", "BaF2", "LaF3",
        "磷酸盐", "硅酸盐", "钒酸盐", "铝酸盐", "硼酸盐", "钼酸盐", "钨酸盐",
        "氟化物", "氮化物", "氧化物", "硫化物", "石榴石", "钙钛矿",
    ],
    "method": [
        "高温固相法", "固相法", "溶胶-凝胶法", "sol-gel", "共沉淀法",
        "水热法", "熔盐法", "燃烧法", "固态反应", "solid-state",
        "溶剂热法", "微波法", "喷雾热解法", "化学气相沉积", "CVD",
        "XRD", "SEM", "TEM", "PL", "XPS", "FTIR", "EPR", "DSC", "TG",
        "荧光光谱", "X射线衍射", "扫描电镜", "透射电镜",
    ],
}

_DY_ALIASES: dict[str, str] = {
    "镝": "Dy3+", "dy": "Dy3+", "dy3+": "Dy3+", "镝离子": "Dy3+",
    "铕": "Eu3+", "eu3+": "Eu3+", "铕离子": "Eu3+",
    "铽": "Tb3+", "tb3+": "Tb3+", "铽离子": "Tb3+",
}


def _latex_to_plain_text(text: str) -> str:
    """MinerU LaTeX → 可读纯文本 (保留离子符号/上下标/数值, 不删除公式).

    原实现把行内公式 $...$ 整个删成空格, 导致 Dy3+/能级跃迁/数值全部丢失。
    现改为: 提取公式内的文本内容, 去掉 LaTeX 记号, 保留信息。
    """
    # 1. LaTeX 字体/语义命令包裹 -> 内容 (\mathrm{Dy} -> Dy)
    for cmd in ("mathrm", "mathsf", "mathbf", "mathit", "operatorname",
                "textit", "textbf", "bm"):
        text = re.sub(rf"\\{cmd}\s*\{{([^{{}}]*)\}}", r"\1", text)
    # 2. 箭头与常见命令
    text = text.replace(r"\rightarrow", "→").replace(r"\to", "→").replace(r"\leftarrow", "←")
    text = text.replace(r"\leftrightarrow", "↔").replace(r"\Rightarrow", "⇒")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\,", "").replace(r"\;", "").replace(r"\quad", " ").replace(r"\qquad", " ")
    # 3. 上下标: ^{3+} -> 3+ / _{9/2} -> 9/2 (保留内容)
    text = re.sub(r"\^\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_\{([^{}]*)\}", r"\1", text)
    # 4. 块公式 $$..$$ 与行内公式 $..$: 去掉 $ 保留内容 (而非删除)
    text = re.sub(r"\$\$([^$]+)\$\$", r"\1", text)
    text = re.sub(r"\$([^$\n]+)\$", r"\1", text)
    # 5. 残留 LaTeX 记号清理
    text = text.replace("^", "").replace("_", "")
    text = text.replace("{", "").replace("}", "").replace("$", "")
    text = text.replace("~", " ")
    # 6. 上标"3 +" -> "3+" (MinerU 在 3+ 里插了空格)
    text = re.sub(r"(\d)\s*\+\s*(?=[^\d])", r"\1+", text)
    # 7. 分数空格: "9 / 2" -> "9/2"
    text = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", text)
    # 8. 残留 LaTeX 命令(无花括号形式, 如 \mathrm LiM)
    text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
    # 9. "%" 前的空格: "3 %" -> "3%"
    text = re.sub(r"(\d)\s*%", r"\1%", text)
    # 10. 拼接 MinerU 拆散的化学式碎片 (D y 3+ -> Dy3+, C s 2 -> Cs2)
    for _ in range(6):
        prev = text
        text = re.sub(r"(?<![A-Za-z])([A-Z])\s+([a-z])(?![A-Za-z])", r"\1\2", text)  # D y -> Dy
        text = re.sub(r"([A-Za-z])\s+(\d)", r"\1\2", text)  # y 3 -> y3
        text = re.sub(r"(\d)\s+([A-Z])", r"\1\2", text)  # 2 L -> 2L
        text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)  # 1 3 -> 13
        if text == prev:
            break
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_markdown(path: Path) -> str:
    """读取 markdown 文本 (清洗 MinerU 的 LaTeX, 保留离子符号/数值)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return _latex_to_plain_text(text)


def _is_on_topic(filename: str) -> bool:
    """判断文献是否贴题."""
    name = filename.lower()
    if _TOPIC_EXCLUDE.search(name):
        return False
    return bool(_TOPIC_INCLUDE.search(name))


def collect_documents() -> list[tuple[str, Path]]:
    """收集贴题文献, 返回 [(document_id, path)]. 去重 + 筛选."""
    docs: list[tuple[str, Path]] = []
    seen_titles: set[str] = set()

    # 1. dy 目录 (最核心, 全部纳入)
    dy_dir = _WXK_ROOT / "dy"
    if dy_dir.is_dir():
        for p in sorted(dy_dir.glob("*.md")):
            # 去重: 提取标题 (去掉 MinerU_markdown_ 前缀 和 _数字ID.md 后缀)
            title = re.sub(r"^MinerU_markdown_", "", p.name)
            title = re.sub(r"_\d+\.md$", ".md", title)
            if title in seen_titles:
                continue
            seen_titles.add(title)
            docs.append((f"dy-{p.stem}", p))

    # 2. 发光学基础知识 (去重 + 贴题)
    base_dir = _WXK_ROOT / "发光学基础知识（发光学报）"
    if base_dir.is_dir():
        for p in sorted(base_dir.glob("*.md")):
            title = re.sub(r"^MinerU_markdown_", "", p.name)
            title = re.sub(r"_\d+\.md$", ".md", title)
            title = re.sub(r"_副本$", "", title)
            if title in seen_titles:
                continue
            if not _is_on_topic(p.name):
                continue
            seen_titles.add(title)
            docs.append((f"base-{p.stem}", p))

    # 3. 硕博论文 (贴题筛选: 剔除生物/免疫/温度传感)
    # 方案 A: 恢复硕博论文以提升知识库覆盖率 (其含 SEM/TEM 区别/荧光寿命等基础知识)
    _INCLUDE_THESIS = True
    thesis_dir = _WXK_ROOT / "硕博论文"
    if _INCLUDE_THESIS and thesis_dir.is_dir():
        for p in sorted(thesis_dir.glob("*.md")):
            title = re.sub(r"^MinerU_markdown_", "", p.name)
            title = re.sub(r"_\d+\.md$", ".md", title)
            if title in seen_titles:
                continue
            if not _is_on_topic(p.name):
                continue
            seen_titles.add(title)
            docs.append((f"thesis-{p.stem}", p))

    return docs


# 方法/表征缩写白名单 (保留为 method 类型, 不删除)
_METHOD_ABBR = {
    "xrd", "sem", "tem", "pl", "xps", "ftir", "epr", "dsc", "tg", "tg-dsc",
    "afm", "xrf", "icp", "eds", "uv", "ir", "nmr", "esr", "dls", "bet",
    "cvd", "pld", "sol-gel", "led",
}


def _clean_noise_concepts(store: Any) -> int:
    """清理孤立英文缩写 concept 实体, 但方法/表征缩写保留并转为 method 类型.

    删除: 纯英文 ≤4 字母且不是方法/表征缩写的碎片 (如 Si/GaN/InP 误拆材料).
    保留+转类型: XRD/SEM/TEM/PL 等表征方法 (→ method).
    """
    import re
    from dy3_polaris.l3.models import EntityType

    removed = 0
    converted = 0
    entities = list(store.entity_store._entities.values())
    for e in entities:
        if e.entity_type.value != "concept":
            continue
        name = e.name or ""
        if re.fullmatch(r"[A-Za-z]{1,4}", name):
            if name.lower() in _METHOD_ABBR:
                # 方法/表征缩写 → 转 method 类型
                try:
                    store.entity_store.update_entity(
                        e.entity_id, entity_type=EntityType.METHOD
                    )
                    converted += 1
                except Exception:
                    pass
            else:
                # 真正的孤立碎片 → 删除
                try:
                    store.remove_entity(e.entity_id)
                    removed += 1
                except Exception:
                    pass
    if converted:
        print(f"  已转 {converted} 个表征方法缩写为 method 类型")
    return removed


def _clean_fake_chemicals(store: Any) -> int:
    """清理假化学式, 但方法/表征缩写转 method 类型.

    真化学式必含小写字母 (NaGdF4)、数字 (Dy3+)、或电荷 (+/-) 之一.
    纯大写字母串: 若是方法/表征缩写 (XRD/SEM) → 转 method; 否则 → 删除
    (ABSTRACT/ACKNOWLEDGEMENTS/作者名等).
    """
    import re
    from dy3_polaris.l3.models import EntityType

    removed = 0
    converted = 0
    entities = list(store.entity_store._entities.values())
    for e in entities:
        if e.entity_type.value != "chemical_compound":
            continue
        name = e.name or ""
        if re.fullmatch(r"[A-Z]{2,}", name):
            if name.lower() in _METHOD_ABBR:
                try:
                    store.entity_store.update_entity(
                        e.entity_id, entity_type=EntityType.METHOD
                    )
                    converted += 1
                except Exception:
                    pass
            else:
                try:
                    store.remove_entity(e.entity_id)
                    removed += 1
                except Exception:
                    pass
    if converted:
        print(f"  已转 {converted} 个化学式缩写为 method 类型")
    return removed


def _dedup_by_type(store: Any, entity_type: str) -> int:
    """按名称去重同类型实体 (词典匹配为每篇文献创建了重复的 method/material 实体).

    保留同名第一个实体, 删除其余重复项.
    """
    from dy3_polaris.l3.models import EntityType as ET

    et = ET(entity_type)
    entities = store.entity_store.find_by_type(et)
    seen: dict[str, str] = {}  # name_lower -> entity_id
    removed = 0
    for e in entities:
        key = (e.name or "").strip().lower()
        if not key:
            continue
        if key in seen:
            try:
                store.remove_entity(e.entity_id)
                removed += 1
            except Exception:
                pass
        else:
            seen[key] = e.entity_id
    return removed


def build_knowledge_graph(dry_run: bool = False) -> dict[str, Any]:
    """批量建图."""
    from dy3_polaris.l3.kg_builder import KnowledgeGraphBuilder
    from dy3_polaris.l3.ingestion import IngestionPipeline
    from dy3_polaris.l3.store import KnowledgeStore
    from dy3_polaris.l3.models import EntityType

    store = KnowledgeStore()
    # 文档摄入管道: 切块存储 (供检索问答使用)
    ingestion = IngestionPipeline(store)
    # 图谱构建器: 实体 + 三元组 (供图谱展示使用)
    builder = KnowledgeGraphBuilder(
        store=store,
        domain="dy-phosphor",
        # 提高置信度阈值到 0.7: 过滤 english_proper(0.6)/chinese_quoted(0.65) 等
        # 噪音实体 (作者名/期刊名/标题碎片), 只保留化学式/波长/词典等高置信度实体.
        min_entity_confidence=0.7,
        min_relation_confidence=0.5,
    )

    # 注入 Dy 领域词典
    for type_name, names in _DY_ENTITY_DICTIONARY.items():
        etype = EntityType(type_name)
        aliases = {k: v for k, v in _DY_ALIASES.items()}
        builder.add_entity_dictionary(etype, names, aliases=aliases)

    docs = collect_documents()
    print(f"筛选出 {len(docs)} 篇贴题文献")
    if dry_run:
        for did, p in docs[:20]:
            print(f"  [dry-run] {did} <- {p.name}")
        return {"total_docs": len(docs), "dry_run": True}

    total_entities = 0
    total_triples = 0
    total_chunks = 0
    t0 = time.time()
    for i, (did, path) in enumerate(docs, 1):
        text = _read_markdown(path)
        if not text:
            continue
        # 长文献截断 (前段已含摘要/机理/实验核心, 避免实体识别过慢)
        if len(text) > 25000:
            text = text[:25000]
        # 1. 文档切块摄入 (供检索)
        try:
            ingest_result = ingestion.ingest(
                content=text,
                document_id=did,
                metadata={"source": "wxk", "file": path.name},
            )
            total_chunks += ingest_result.successful
        except Exception as exc:  # noqa: BLE001
            pass
        # 2. 图谱构建 (实体 + 三元组)
        result = builder.build_from_text(
            text, source_id=did, source_meta={"file": path.name}
        )
        total_entities += result.entities_created
        total_triples += result.triples_created
        if i % 20 == 0 or i == len(docs):
            elapsed = time.time() - t0
            print(
                f"  进度 {i}/{len(docs)} | 累计实体 {total_entities} | "
                f"三元组 {total_triples} | {elapsed:.0f}s"
            )

    stats = builder.get_stats()
    # 后处理: 清理孤立缩写实体 (Si/XRD/PL 等 ≤4 字母纯英文 concept 碎片),
    # 这些是 MinerU 抽取的噪音, 对 Dy 领域知识图谱无价值, 且污染知识库实体列表.
    cleaned = _clean_noise_concepts(store)
    if cleaned > 0:
        print(f"  已清理 {cleaned} 个孤立缩写实体")
    # 后处理2: 清理假化学式 (纯大写无数字/无小写/无电荷, 如 ABSTRACT/ACKNOWLEDGEMENTS/作者名)
    cleaned2 = _clean_fake_chemicals(store)
    if cleaned2 > 0:
        print(f"  已清理 {cleaned2} 个假化学式实体")
    # 后处理3: method/material 同名去重 (词典匹配为每篇文献创建了重复实体)
    dedup_method = _dedup_by_type(store, "method")
    dedup_material = _dedup_by_type(store, "material")
    if dedup_method or dedup_material:
        print(f"  已去重 method {dedup_method} 个, material {dedup_material} 个")

    # 持久化: 只保存一次最终快照到系统启动会加载的位置 (避免反复 save_snapshot 膨胀)
    snapshot_path = ""
    if not dry_run:
        try:
            from dy3_polaris.l3.persistence import PersistenceManager

            snapshot_dir = _SRC / "dy3_polaris" / "l3" / "data" / "snapshots" / "snapshot_final"
            pm = PersistenceManager(store, base_path=str(snapshot_dir.parent))
            saved = pm.save_snapshot(snapshot_dir)
            snapshot_path = str(saved)
            print(f"  已保存最终快照: {saved}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 快照保存失败: {exc}")

    return {
        "total_docs": len(docs),
        "total_entities": total_entities,
        "total_triples": total_triples,
        "total_chunks": total_chunks,
        "store_entities": stats["entity_count"],
        "store_triples": stats["triple_count"],
        "elapsed_s": round(time.time() - t0, 1),
        "snapshot_path": snapshot_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dy 垂直领域知识图谱批量构建")
    parser.add_argument("--dry-run", action="store_true", help="只统计不建图")
    args = parser.parse_args()

    print("=" * 60)
    print("Dy 垂直领域知识库批量构建")
    print("=" * 60)

    result = build_knowledge_graph(dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n共 {result['total_docs']} 篇贴题文献 (未建图)")
        return 0

    print("\n" + "=" * 60)
    print("构建完成:")
    print(f"  文献数: {result['total_docs']}")
    print(f"  新增实体: {result['total_entities']}")
    print(f"  新增三元组: {result['total_triples']}")
    print(f"  新增文档切片: {result['total_chunks']}")
    print(f"  存储实体总数: {result['store_entities']}")
    print(f"  存储三元组总数: {result['store_triples']}")
    print(f"  耗时: {result['elapsed_s']}s")
    if result.get("snapshot_path"):
        print(f"  快照: {result['snapshot_path']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
