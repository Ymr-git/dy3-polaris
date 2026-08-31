# -*- coding: utf-8 -*-
"""验证化学式碎片拼接。"""
import sys
sys.path.insert(0, r"D:\BaiduNetdiskDownload\xiaotiao\dy3-polaris-整理后\04-编码\scripts")
from build_dy_knowledge_graph import _latex_to_plain_text  # noqa: E402

# 直接测试清洗函数对化学式 LaTeX 的处理
samples = [
    r"$\mathsf { D } \mathsf { y } ^ { 3 + }$",
    r"$\mathsf { C } \mathsf { s } _ { 2 } \mathsf { L } \mathsf { i } _ { 3 } \mathsf { S } \mathsf { r } _ { 2 }$",
    r"$^ 4 \mathsf { F } _ { 9 / 2 } \to ^ { 6 } \mathsf { H } _ { 1 3 / 2 }$",
]
for s in samples:
    print(f"{s}\n  -> {_latex_to_plain_text(s)}\n")

# 用根目录真实 md 测试
path = r"D:\BaiduNetdiskDownload\xiaotiao\MinerU_markdown_LiMBO3__Dy3+材料的发光特性_NormalPdf_2085222480010584064.md"
text = _latex_to_plain_text(open(path, encoding="utf-8", errors="ignore").read())
for kw in ["Dy3+", "Ce3+", "BO3", "484", "577", "3%"]:
    print(f"  '{kw}': {'保留' if kw in text else '丢失'}")
# 检查是否还有单字母碎片（D y / C s 这种）
import re
frag = re.findall(r"\b[A-Z]\s+[a-z]\b", text)
print(f"  残留单字母碎片: {frag[:10]}")
