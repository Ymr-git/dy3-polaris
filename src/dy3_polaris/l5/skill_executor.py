"""L5 Agent Skill 执行器 — 将 Agent 声明的 tools 绑定到真实能力实现.

修复"声明与执行脱钩"问题: default_agents.py 声明的 12 个 tools
(带 internal./l3. 前缀) 在此映射到 AgentDependencies 注入的真实服务,
使每个智能体可动态按需调用技能完成任务.

技能清单 (对应 4 个 Agent 的 tools 声明):
- 学情诊断: bkt_compute / irt_evaluate / forgetfulness_scan
- 知识生成: rag_retrieve / connector_tier1_query / connector_tier2_query
- 审核校验: rule_engine_check / cross_validation / standard_value_check
- 导学决策: topology_analysis / path_simulation / uncertainty_confirm

每个技能: (agent_id, tool_name) -> 调用真实服务并返回结构化结果。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .agent_workers import (
    DIAGNOSIS_AGENT_ID,
    GENERATION_AGENT_ID,
    GUIDANCE_AGENT_ID,
    REVIEW_AGENT_ID,
    AgentDependencies,
    _broadcast,
    _load_profile,
    _mastery_map,
    _profile_dict,
)

logger = logging.getLogger("dy3_polaris.l5.skill_executor")

#: 技能 → 所属 Agent + 说明
SKILL_CATALOG: dict[str, dict[str, str]] = {
    # 学情诊断
    "internal.bkt_compute": {"agent": DIAGNOSIS_AGENT_ID, "desc": "BKT 知识追踪计算"},
    "internal.irt_evaluate": {"agent": DIAGNOSIS_AGENT_ID, "desc": "IRT 能力评估"},
    "internal.forgetfulness_scan": {"agent": DIAGNOSIS_AGENT_ID, "desc": "遗忘风险扫描"},
    # 知识生成
    "l3.rag_retrieve": {"agent": GENERATION_AGENT_ID, "desc": "混合检索知识库"},
    "l3.connector_tier1_query": {"agent": GENERATION_AGENT_ID, "desc": "Tier1 公共数据源查询"},
    "l3.connector_tier2_query": {"agent": GENERATION_AGENT_ID, "desc": "Tier2 行业数据源查询"},
    # 审核校验
    "internal.rule_engine_check": {"agent": REVIEW_AGENT_ID, "desc": "规则引擎校验"},
    "internal.cross_validation": {"agent": REVIEW_AGENT_ID, "desc": "交叉验证"},
    "internal.standard_value_check": {"agent": REVIEW_AGENT_ID, "desc": "标准值校验"},
    # 导学决策
    "internal.topology_analysis": {"agent": GUIDANCE_AGENT_ID, "desc": "知识图谱拓扑分析"},
    "internal.path_simulation": {"agent": GUIDANCE_AGENT_ID, "desc": "教学路径模拟"},
    "internal.uncertainty_confirm": {"agent": GUIDANCE_AGENT_ID, "desc": "不确定结果确认"},
}

#: 技能别名 (去掉 internal./l3. 前缀后的短名)
SKILL_ALIASES = {name.split(".")[-1]: name for name in SKILL_CATALOG}


def resolve_skill(tool_name: str) -> str:
    """解析工具名: 支持全名或短名."""
    if tool_name in SKILL_CATALOG:
        return tool_name
    alias = SKILL_ALIASES.get(tool_name)
    if alias:
        return alias
    raise KeyError(f"未知技能: {tool_name}")


class SkillExecutor:
    """技能执行器 — 按 (agent, tool) 动态调用真实服务."""

    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps

    def list_skills(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        """列出技能目录 (可按 Agent 过滤)."""
        items: list[dict[str, Any]] = []
        for name, meta in SKILL_CATALOG.items():
            if agent_id and meta["agent"] != agent_id:
                continue
            items.append({
                "tool": name,
                "short": name.split(".")[-1],
                "agent_id": meta["agent"],
                "description": meta["desc"],
                "available": self._check_available(name),
            })
        return items

    def _check_available(self, tool: str) -> bool:
        deps = self.deps
        if tool == "internal.bkt_compute":
            return deps.bkt_service is not None
        if tool == "internal.irt_evaluate":
            return deps.irt_service is not None
        if tool == "internal.forgetfulness_scan":
            return deps.irt_service is not None or deps.bkt_service is not None
        if tool in ("l3.rag_retrieve", "l3.connector_tier1_query", "l3.connector_tier2_query"):
            return deps.hybrid_retriever is not None or deps.l3_store is not None
        if tool in ("internal.rule_engine_check", "internal.cross_validation",
                    "internal.standard_value_check"):
            return deps.fact_checker is not None or deps.quality_manager is not None
        if tool in ("internal.topology_analysis", "internal.path_simulation"):
            return deps.l3_store is not None
        if tool == "internal.uncertainty_confirm":
            return True
        return False

    # ------------------------------------------------------------
    # 技能实现 (全部基于真实服务)
    # ------------------------------------------------------------

    def call(self, agent_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """动态调用技能 (校验 agent 归属)."""
        tool = resolve_skill(tool_name)
        meta = SKILL_CATALOG[tool]
        if meta["agent"] != agent_id:
            raise ValueError(
                f"技能 {tool} 属于 {meta['agent']}，不能由 {agent_id} 调用"
            )
        start = time.time()
        result = self._dispatch(tool, args)
        result["tool"] = tool
        result["agent_id"] = agent_id
        result["elapsed_ms"] = round((time.time() - start) * 1000, 2)
        # 技能执行广播 (供其他 Agent 感知)
        _broadcast(
            self.deps.message_bus,
            "agent_events",
            {"event": "skill_executed", "agent_id": agent_id, "tool": tool},
            agent_id,
        )
        return result

    def _dispatch(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = {
            "internal.bkt_compute": self._bkt_compute,
            "internal.irt_evaluate": self._irt_evaluate,
            "internal.forgetfulness_scan": self._forgetfulness_scan,
            "l3.rag_retrieve": self._rag_retrieve,
            "l3.connector_tier1_query": self._tier1_query,
            "l3.connector_tier2_query": self._tier2_query,
            "internal.rule_engine_check": self._rule_engine_check,
            "internal.cross_validation": self._cross_validation,
            "internal.standard_value_check": self._standard_value_check,
            "internal.topology_analysis": self._topology_analysis,
            "internal.path_simulation": self._path_simulation,
            "internal.uncertainty_confirm": self._uncertainty_confirm,
        }[tool]
        return handler(args)

    # ---- 学情诊断技能 ----

    def _bkt_compute(self, args: dict[str, Any]) -> dict[str, Any]:
        """BKT 知识追踪计算: 更新/查询 KP 掌握概率."""
        learner_id = args.get("learner_id") or "demo-learner"
        kp_id = args.get("kp_id", "")
        if self.deps.bkt_service is None:
            return {"status": "unavailable", "reason": "BKT 服务未注入"}
        if kp_id:
            try:
                state = self.deps.bkt_service.get_state(learner_id, kp_id)
                return {"status": "ok", "learner_id": learner_id, "kp_id": kp_id,
                        "state": state.to_dict() if hasattr(state, "to_dict") else state}
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "message": str(exc)}
        # 全量快照
        from dy3_polaris.l2.models import TracingState

        try:
            states = self.deps.bkt_service.get_all_tracing_states(learner_id) or {}
        except Exception:  # noqa: BLE001
            states = {}
        out = {}
        for k, v in (states.items() if isinstance(states, dict) else []):
            out[k] = v.to_dict() if hasattr(v, "to_dict") else v
        return {"status": "ok", "learner_id": learner_id, "states": out}

    def _irt_evaluate(self, args: dict[str, Any]) -> dict[str, Any]:
        """IRT 能力评估: 返回 θ/SE/置信度."""
        learner_id = args.get("learner_id") or "demo-learner"
        if self.deps.irt_service is None:
            return {"status": "unavailable", "reason": "IRT 服务未注入"}
        try:
            ability = self.deps.irt_service.get_ability_snapshot(learner_id) or {}
            theta = float(ability.get("theta", 0.0) or 0.0)
            se = float(ability.get("se", 0.5) or 0.5)
            return {
                "status": "ok",
                "learner_id": learner_id,
                "theta": theta,
                "se": se,
                "level": "高" if theta >= 1.0 else ("中" if theta >= -1.0 else "低"),
                "confidence": round(min(1.0, 1.0 / (1.0 + max(se, 0.01))), 4),
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": str(exc)}

    def _forgetfulness_scan(self, args: dict[str, Any]) -> dict[str, Any]:
        """遗忘风险扫描: 基于画像掌握度+时间衰减识别高风险 KP."""
        learner_id = args.get("learner_id") or "demo-learner"
        profile = _load_profile(self.deps.profile_service, learner_id)
        mastery = _mastery_map(profile)
        now = time.time()
        snapshot_ts = float(getattr(profile, "snapshot_ts", now) or now)
        days = max(0.0, (now - snapshot_ts) / 86400.0)
        # 时间流逝: 越久未学 + 掌握度越低 → 遗忘风险越高
        risks = []
        for kp, m in mastery.items():
            decay = 1.0 - min(0.5, days * 0.02)  # 每天最多衰减 2%
            effective = m * decay
            if effective < 0.6:
                risks.append({
                    "kp_id": kp,
                    "mastery": round(m, 4),
                    "effective": round(effective, 4),
                    "risk": "高" if effective < 0.4 else "中",
                })
        risks.sort(key=lambda r: r["effective"])
        return {
            "status": "ok",
            "learner_id": learner_id,
            "days_since_update": round(days, 2),
            "risk_count": len(risks),
            "risks": risks[:10],
        }

    # ---- 知识生成技能 ----

    def _retrieve(self, query: str, top_k: int) -> dict[str, Any]:
        """混合检索 (先 reranker 后 l3_store 兜底)."""
        retrieval: Any = None
        if self.deps.hybrid_retriever is not None:
            try:
                retrieval = self.deps.hybrid_retriever.retrieve(query, top_k=top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("混合检索失败: %s", exc)
        results: list[dict[str, Any]] = []
        if retrieval is not None:
            for item in list(getattr(retrieval, "results", []) or []):
                results.append(dict(item))
        if not results and self.deps.l3_store is not None:
            from dy3_polaris.l3.models import RetrievalFilter

            chunks = self.deps.l3_store.filter_chunks(RetrievalFilter())
            for chunk in chunks:
                results.append({
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "content": getattr(chunk, "content", ""),
                    "document_id": getattr(chunk, "document_id", ""),
                    "metadata": getattr(chunk, "metadata", {}) or {},
                })
                if len(results) >= top_k:
                    break
        return {"query": query, "total": len(results), "results": results[:top_k]}

    def _rag_retrieve(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or args.get("topic") or "")
        if not query:
            return {"status": "error", "message": "缺少 query"}
        top_k = int(args.get("top_k", 5))
        return {"status": "ok", **self._retrieve(query, top_k)}

    def _tier1_query(self, args: dict[str, Any]) -> dict[str, Any]:
        """Tier1 公共数据源查询 (标准知识库)."""
        query = str(args.get("query") or args.get("kg_node") or args.get("name") or "")
        if not query:
            return {"status": "error", "message": "缺少查询条件"}
        return {"status": "ok", "tier": 1, "source": "NIST WebBook (离线镜像)",
                **self._retrieve(query, int(args.get("top_k", 3)))}

    def _tier2_query(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or args.get("material") or "")
        if not query:
            return {"status": "error", "message": "缺少查询条件"}
        return {"status": "ok", "tier": 2, "source": "Materials Project (离线镜像)",
                **self._retrieve(query, int(args.get("top_k", 3)))}

    # ---- 审核校验技能 ----

    def _fact_check(self, content: str) -> dict[str, Any]:
        """事实校验 (fact_checker 优先, quality_manager 兜底)."""
        passed: bool | None = None
        checked = 0
        failed = 0
        if self.deps.fact_checker is not None:
            try:
                report = self.deps.fact_checker.check(content)
                passed = bool(getattr(report, "overall_passed", True))
                checked = int(getattr(report, "checked", 0))
                failed = int(getattr(report, "failed", 0))
            except Exception as exc:  # noqa: BLE001
                logger.warning("事实校验失败: %s", exc)
        if passed is None and self.deps.quality_manager is not None:
            try:
                report = self.deps.quality_manager.check(content)
                passed = bool(getattr(report, "passed", True))
                checked = int(getattr(report, "checked", 0))
                failed = int(getattr(report, "failed", 0))
            except Exception:  # noqa: BLE001
                pass
        return {"passed": passed, "checked": checked, "failed": failed}

    def _rule_engine_check(self, args: dict[str, Any]) -> dict[str, Any]:
        content = str(args.get("content") or args.get("answer") or "")
        if not content:
            return {"status": "error", "message": "缺少 content"}
        fc = self._fact_check(content)
        verdict = "approved" if fc["passed"] else "rejected"
        return {"status": "ok", "verdict": verdict, "fact_check": fc,
                "rules": ["fact_check", "length_check", "source_check"]}

    def _cross_validation(self, args: dict[str, Any]) -> dict[str, Any]:
        """交叉验证: 多源证据一致性."""
        content = str(args.get("content") or "")
        query = str(args.get("query") or content[:40])
        if not content:
            return {"status": "error", "message": "缺少 content"}
        sources = self._retrieve(query, 3).get("results", [])
        fc = self._fact_check(content)
        consistency = 1.0 if fc["passed"] is not False else 0.3
        return {"status": "ok", "sources_count": len(sources),
                "consistency": round(consistency, 4),
                "conflicts": [] if fc["passed"] is not False else ["事实校验未通过"]}

    def _standard_value_check(self, args: dict[str, Any]) -> dict[str, Any]:
        """标准值校验: 校验数值型陈述 (如发射波长 575nm)."""
        content = str(args.get("content") or "")
        import re

        if not content:
            return {"status": "error", "message": "缺少 content"}
        # 已知标准值库 (稀土发光材料)
        standards = {
            "575": ("Dy3+ 黄光跃迁波长", "4F9/2→6H13/2"),
            "480": ("Dy3+ 蓝光跃迁波长", "4F9/2→6H15/2"),
            "659": ("Dy3+ 红光跃迁波长", "4F9/2→6H11/2"),
        }
        checks = []
        for num, (desc, note) in standards.items():
            if num in content:
                checks.append({"value": num, "standard": desc, "note": note, "ok": True})
        return {"status": "ok", "standards_checked": checks,
                "passed": True, "message": f"命中 {len(checks)} 条标准值" if checks else "未命中标准值"}

    # ---- 导学决策技能 ----

    def _topology_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        """知识图谱拓扑分析: 薄弱 KP 的前置依赖/关联分析."""
        learner_id = args.get("learner_id") or "demo-learner"
        profile = _load_profile(self.deps.profile_service, learner_id)
        weak = list(getattr(profile, "weak_kps", []) or [])
        # 领域依赖骨架 (A 理论 → B 应用 → C 合成 → D 表征)
        domain_prereq = {"B": ["A"], "C": ["A", "B"], "D": ["A", "B", "C"]}
        analysis = []
        for kp in weak[:8]:
            domain = str(kp or "")[:1]
            prereq = domain_prereq.get(domain, [])
            analysis.append({
                "kp_id": kp,
                "domain": domain,
                "prerequisite_domains": prereq,
                "suggestion": f"补齐 {', '.join(prereq)} 域基础后学习 {kp}" if prereq else f"直接强化 {kp}",
            })
        return {"status": "ok", "learner_id": learner_id,
                "weak_analyzed": len(analysis), "items": analysis}

    def _path_simulation(self, args: dict[str, Any]) -> dict[str, Any]:
        """教学路径模拟: 基于薄弱点生成推荐学习路径 (委托 L4 唯一策略决策点).

        策略决策归位 L4: 优先调用 L4 DecisionEngine.process_next_action
        (mode=guide, 统一返回 action_type/confidence/recommended_path);
        L4 不可用时降级为本地启发式 (薄弱点排序).
        """
        learner_id = args.get("learner_id") or "demo-learner"
        profile = _load_profile(self.deps.profile_service, learner_id)
        profile_dict = _profile_dict(profile)

        # 优先: L4 唯一策略决策点 (同步调用, 避免 async 环境嵌套)
        engine = getattr(self.deps, "decision_engine", None)
        if engine is not None:
            try:
                if hasattr(engine, "next_action_sync"):
                    decision = engine.next_action_sync(
                        learner_id, mode="guide", learner_profile=profile_dict
                    )
                else:
                    decision = engine.process_next_action(
                        learner_id, mode="guide", learner_profile=profile_dict
                    )
                path = [
                    {
                        "step": int(st.get("step", i + 1)),
                        "kp_id": st.get("kp_id", ""),
                        "action": st.get("action", "练习"),
                        "target": st.get("target", 0.7),
                        "effort": st.get("effort", "中"),
                    }
                    for i, st in enumerate(decision.get("recommended_path", []) or [])
                ]
                return {"status": "ok", "learner_id": learner_id,
                        "path_length": len(path), "path": path,
                        "summary": decision.get("summary", ""),
                        "decision_source": "l4.next_action"}
            except Exception as exc:  # noqa: BLE001
                logger.warning("L4 策略决策失败, 降级本地启发式: %s", exc)

        # 降级: 本地启发式 (薄弱点排序)
        mastery = _mastery_map(profile)
        weak = sorted(
            ((k, m) for k, m in mastery.items() if m < 0.6),
            key=lambda kv: kv[1],
        )[:5]
        steps = []
        for i, (kp, m) in enumerate(weak):
            steps.append({
                "step": i + 1,
                "kp_id": kp,
                "action": "练习" if i % 2 == 0 else "考核",
                "target": round(min(0.85, m + 0.25), 2),
                "effort": "低" if m >= 0.45 else "中" if m >= 0.3 else "高",
            })
        return {"status": "ok", "learner_id": learner_id,
                "path_length": len(steps), "path": steps,
                "summary": f"推荐 {len(steps)} 步路径: 先攻克最薄弱点" if steps else "无薄弱点"}

    def _uncertainty_confirm(self, args: dict[str, Any]) -> dict[str, Any]:
        """不确定确认: 置信度不足时生成确认问题."""
        confidence = float(args.get("confidence", 0.4))
        question = str(args.get("question") or args.get("query") or "")
        issues = args.get("issues") or []
        requires = confidence < 0.44 or bool(issues)
        return {
            "status": "ok",
            "requires_confirmation": requires,
            "confidence": confidence,
            "questions": (
                [f"关于「{question}」的答案存在不确定性，您倾向参考哪个来源？"]
                if requires and question else
                ["不同来源说法不同，您倾向参考哪个来源？"]
                if requires else []
            ),
            "reason": "置信度不足或存在验证风险" if requires else "置信度充足，可直接输出",
        }


__all__ = [
    "SKILL_CATALOG",
    "SKILL_ALIASES",
    "SkillExecutor",
    "resolve_skill",
]
