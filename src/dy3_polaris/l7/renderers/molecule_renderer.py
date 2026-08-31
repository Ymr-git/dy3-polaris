"""L7 渲染器 — MoleculeRenderer (application/vnd.dy3.molecule+json).

将分子/晶体结构 Artifact 渲染为 3Dmol.js 可消费的配置。
服务端完成结构解析、能级动画配置与三级降级决策。

实现能力 (对应 L7 设计文档 §2.5 + §4.3.2):

1. **结构输入**: payload["structure"] 支持三种来源:
   - molfile (MDL V2000) → 3Dmol.js addMol (chemical/x-mdl-molfile)
   - smiles → 前端通过 3Dmol.js + RDKit 服务解析
   - cif (晶体 CIF 内容) → 3Dmol.js addModel (cif 格式)
   宿主晶格: NaGdF4 / YPO4 / BaMgAl10O17 等。
2. **显示模式**: 球棍模型 / 空间填充 / 多面体 (front 配置切换)。
3. **4f-5d 能级跃迁动画** (§2.5.2): payload["animation"] 描述
   激发 → 非辐射弛豫 → 辐射跃迁 (黄光 ~575nm) 过程，与 Jablonski 图同步。
4. **光谱叠加** (§2.5.3): payload["spectra"] 附带 PL/PLE 数据。
5. **三级降级** (§2.5.4): Level0 完整 WebGL2 / Level1 简化 WebGL1
   (面数<5000 禁动画) / Level2 静态 2D Canvas (附"查看 3D"链接)。
   降级决策在前端运行时执行, 服务端生成三个等级的配置骨架。

融合世界先进方案:
    - 3Dmol.js: WebGL 分子渲染管线
    - WebGL 能力检测渐进增强: 三级降级策略

输出契约:
    RenderDescriptor.html   — 挂载壳 (data-molecule-id, 容器按等级渲染)
    RenderDescriptor.config — {structure, style, animation, spectra, levels}
    RenderDescriptor.assets — [3Dmol-min.js]
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..models import Artifact, RenderContext
from ._common import build_descriptor, esc, wrap

#: 支持的 MIME 类型
_MIME_TYPES: list[str] = [
    "application/vnd.dy3.molecule+json",
    "chemical/x-mdl-molfile",
]

#: 常见宿主晶格信息 (用于标注与降级)
_HOST_LATTICES: dict[str, dict[str, Any]] = {
    "NaGdF4": {"formula": "NaGdF4", "system": "六方晶系", "space_group": "P-6 2m"},
    "YPO4": {"formula": "YPO4", "system": "四方晶系", "space_group": "I41/amd"},
    "BaMgAl10O17": {"formula": "BaMgAl10O17", "system": "六方晶系", "space_group": "P63/mmc"},
}

#: 显示模式 → 3Dmol.js style 配置
_STYLE_MODES: dict[str, dict[str, Any]] = {
    "stick": {"stick": {"radius": 0.18, "colorscheme": "Jmol"}},
    "ball": {
        "stick": {"radius": 0.12, "colorscheme": "Jmol"},
        "sphere": {"scale": 0.28, "colorscheme": "Jmol"},
    },
    "spacefill": {"sphere": {"scale": 0.85, "colorscheme": "Jmol"}},
}


class MoleculeRenderer:
    """分子渲染器 — 晶体/分子结构 → 3Dmol.js 配置 (服务端构建)."""

    _MIME_TYPES: list[str] = list(_MIME_TYPES)

    def render(self, artifact: Artifact, context: RenderContext):
        started = time.monotonic()
        if artifact is None or not artifact.payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload", detail="Molecule artifact requires non-empty payload"
            )
        payload = artifact.payload
        if "molfile" not in payload and "smiles" not in payload and "structure" not in payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload",
                missing_fields=["molfile", "smiles", "structure"],
                detail="Molecule artifact requires 'molfile', 'smiles' or 'structure' in payload",
            )

        theme = (context.theme if context else "light") or "light"
        structure = self._resolve_structure(payload)
        style_mode = str(payload.get("style", "ball")).lower()
        if style_mode not in _STYLE_MODES:
            style_mode = "ball"

        animation = payload.get("animation") or {}
        spectra = payload.get("spectra") or []
        host = str(payload.get("host", ""))
        host_info = _HOST_LATTICES.get(host)

        html = wrap(
            f'<div class="l7-molecule" data-molecule-id="{artifact.artifact_id}" '
            f'data-host="{esc(host)}" style="width:100%;height:{payload.get("height", 400)}px"></div>',
            "l7-molecule-wrap",
            theme,
        )
        config = {
            "structure": structure,
            "style": _STYLE_MODES[style_mode],
            "style_mode": style_mode,
            "animation": self._normalize_animation(animation),
            "spectra": spectra,
            "host_info": host_info,
            "levels": {
                "level0": {"webgl": "webgl2", "features": ["rotate", "zoom", "animation"]},
                "level1": {
                    "webgl": "webgl1",
                    "features": ["rotate", "zoom"],
                    "limits": {"face_count": 5000, "animation": False},
                },
                "level2": {
                    "webgl": None,
                    "features": ["pan", "zoom"],
                    "fallback": "2d-canvas",
                    "server_render": {"format": "gif", "note": "点击查看 3D 版本"},
                },
            },
            "interactions": {
                "rotate": True,
                "zoom": True,
                "style_switch": ["stick", "ball", "spacefill"],
            },
        }
        descriptor = build_descriptor(
            artifact,
            html=html,
            config=config,
            assets=[
                "https://cdn.jsdelivr.net/npm/3dmol@2.4.1/build/3Dmol-min.js",
            ],
            metadata={
                "renderer": "MoleculeRenderer",
                "structure_source": structure["source"],
                "host": host,
                "animation": bool(animation),
                "spectra_count": len(spectra),
            },
        )
        descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
        return descriptor

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)

    # ----------------------------------------------------------
    # 内部实现
    # ----------------------------------------------------------

    @staticmethod
    def _resolve_structure(payload: dict[str, Any]) -> dict[str, Any]:
        """解析结构输入, 归一化为 3Dmol.js 可消费格式.

        Returns:
            {source: molfile|smiles|cif, format, content, meta}
        """
        if "molfile" in payload and payload["molfile"]:
            return {
                "source": "molfile",
                "format": "mol",
                "content": payload["molfile"],
                "meta": {"label": payload.get("label", "分子结构")},
            }
        if "smiles" in payload and payload["smiles"]:
            return {
                "source": "smiles",
                "format": "smiles",
                "content": payload["smiles"],
                "meta": {"label": payload.get("label", "SMILES 结构")},
            }
        structure = payload["structure"]
        content = structure.get("content", "") if isinstance(structure, dict) else str(structure)
        fmt = (
            structure.get("format", "cif")
            if isinstance(structure, dict)
            else "cif"
        )
        return {
            "source": "structure",
            "format": fmt,
            "content": content,
            "meta": dict(structure.get("meta", {})) if isinstance(structure, dict) else {},
        }

    @staticmethod
    def _normalize_animation(animation: dict[str, Any]) -> dict[str, Any] | None:
        """归一化 4f-5d 能级跃迁动画配置 (§2.5.2).

        payload["animation"] 示例::

            {
              "ground": "^6H_15/2",
              "excited_5d": "4f^5 5d",
              "excited_4f": "^4F_9/2",
              "emission_nm": 575,
              "steps": ["excitation", "relaxation", "emission"],
              "sync_jablonski": true
            }
        """
        if not animation:
            return None
        return {
            "ground": str(animation.get("ground", "^6H_15/2")),
            "excited_5d": str(animation.get("excited_5d", "4f^5 5d")),
            "excited_4f": str(animation.get("excited_4f", "^4F_9/2")),
            "emission_nm": float(animation.get("emission_nm", 575)),
            "steps": list(animation.get("steps", ["excitation", "relaxation", "emission"])),
            "sync_jablonski": bool(animation.get("sync_jablonski", True)),
            "duration_ms": int(animation.get("duration_ms", 4000)),
        }
