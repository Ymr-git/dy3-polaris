"""L2 个性化层 — REST API 路由层.

基于 Starlette 构建, 将 L2 个性化层的 IRT 能力评估、BKT 知识追踪等
核心功能暴露为 RESTful JSON API。

遵循与 L3/L6 API 一致的设计模式:
- 统一响应格式: {"code": 0, "data": ..., "message": ""}
- CORS 中间件支持
- 异常统一处理
- 资源导向 URL 设计 (RESTful 语义)

融合世界先进方案的 API 设计:
- Knewton API: IRT 驱动能力评估即服务 (estimate → theta/se/ZPD)
- ALEKS API: 自适应出题与知识空间查询 (next-question → item_id)
- catR / mirt: R 包 CAT 选题与参数校准 REST 化 (calibrate → MMLE)
- Duolingo API: 实时学情更新端点 (bkt/update → p_mastery)
- OpenAPI 3.0: 资源描述与 schema
- JSON:API spec: 统一响应结构

设计参考:
- L2 个性化设计 §7.3: /l2/irt/estimate, /l2/irt/next-question, /l2/bkt/update
- L6 协议基础设施: MCP 工具 skill_irt_evaluate
- 测试策略: 延迟 <200ms, 覆盖率 ≥90%

使用示例::

    from dy3_polaris.l2.ability_assessor import IRTTracingService
    from dy3_polaris.l2.api import L2Router

    irt_service = IRTTracingService(enable_enhanced=True)
    irt_service.set_item_bank([...])
    router = L2Router(irt_service=irt_service)
    app = router.create_app()

    # 独立运行
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

    # 或嵌入到主应用
    from starlette.routing import Mount
    main_routes = [
        Mount("/l2", app=router.create_app()),
    ]
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from dy3_polaris.l2.ability_assessor import IRTTracingService
from dy3_polaris.l2.exceptions import L2Error
from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.models import ProfileConflictError

# BKTTracingService 可选导入 (延迟初始化, 避免 /bkt/update 不用时加载开销)
try:
    from dy3_polaris.l2.knowledge_tracer import BKTTracingService
    _BKT_AVAILABLE = True
except ImportError:
    _BKT_AVAILABLE = False
    BKTTracingService = None  # type: ignore[assignment,misc]

_logger = logging.getLogger("dy3_polaris.l2.api.router")


# ============================================================
# 统一响应
# ============================================================

# 响应信封单点 (SSOT: shared/contract.py)
from dy3_polaris.shared.contract import err as _err, ok as _ok


def _safe_dump(obj: Any) -> Any:
    """安全地将 dataclass / dict / list 转为可 JSON 序列化的值."""
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _safe_dump(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dump(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _safe_dump(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


# ============================================================
# 路由处理器
# ============================================================

class _RouteHandlers:
    """将 L2 个性化层方法适配为 Starlette Request→Response 处理器.

    每个处理器方法:
    1. 解析请求参数 (path/query/body)
    2. 调用 L2 服务方法 (IRTTracingService / BKTTracingService)
    3. 将异常转为统一错误响应
    4. 返回 JSONResponse
    """

    def __init__(
        self,
        irt_service: IRTTracingService,
        bkt_service: Any | None = None,
        profile_service: Any | None = None,
        memory_service: Any | None = None,
        practice_observer: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> None:
        self._irt = irt_service
        self._bkt = bkt_service
        self._profile = profile_service
        self._memory = memory_service
        self._practice_observer = practice_observer
        # 练习题库 (BKT 服务可用时初始化, 加载 38 题)
        self._practice = None
        self._dynamic_engine = None
        if self._bkt is not None or _BKT_AVAILABLE:
            try:
                from dy3_polaris.l2.practice import PracticeBank

                self._practice = PracticeBank()
                from dy3_polaris.l2.dynamic_questions import DynamicQuestionEngine

                self._dynamic_engine = DynamicQuestionEngine(self._practice)
            except Exception:
                _logger.warning("PracticeBank 初始化失败, 练习端点不可用", exc_info=True)
        # BKT 服务延迟初始化
        if self._bkt is None and _BKT_AVAILABLE:
            try:
                self._bkt = BKTTracingService()
            except Exception:
                _logger.warning("BKTTracingService 初始化失败, /bkt/update 端点不可用")
                self._bkt = None

    # ---- 健康检查 ----

    async def health(self, request: Request) -> JSONResponse:
        """GET /l2/health — L2 个性化层健康检查.

        返回服务状态和已注册的子服务列表。
        """
        services: dict[str, str] = {
            "irt": "available",
        }
        if self._bkt is not None:
            services["bkt"] = "available"
        else:
            services["bkt"] = "unavailable"

        return JSONResponse(_ok({
            "status": "healthy",
            "layer": "L2",
            "timestamp": time.time(),
            "services": services,
        }))

    # ---- IRT 能力估计 (POST /l2/irt/estimate) ----

    async def irt_estimate(self, request: Request) -> JSONResponse:
        """POST /l2/irt/estimate — IRT 能力参数估计.

        对应设计文档 §7.3 和 L6 MCP 工具 skill_irt_evaluate。

        请求体:
            learner_id: 学习者 ID (必填)
            events: 答题事件列表, 每项含:
                - learner_id, kp_id, correct, difficulty, timestamp
            mastery_map: {item_id: p_mastery} 掌握度映射 (可选, 融合模式)

        响应:
            theta: 能力估计 θ
            se: 估计标准误
            response_count: 作答次数
            p_correct_next: 下一题预测答对概率
            zpd_zone: ZPD 区分类
            confidence: 置信度
            ability_level: 能力等级 ("低"/"中"/"高")
            recommendation: 推荐信息
            next_item_id: CAT 推荐下一题 ID
            termination_flag: CAT 是否应终止
            ci_lower/ci_upper: 可信区间 (增强模式)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        learner_id = body.get("learner_id")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少必填参数: learner_id"), status_code=400)

        # 解析事件列表
        raw_events = body.get("events", [])
        events: list[AnswerEvent] = []
        for ev in raw_events:
            events.append(AnswerEvent(
                learner_id=ev.get("learner_id", learner_id),
                kp_id=ev.get("kp_id", ""),
                correct=bool(ev.get("correct", False)),
                difficulty=float(ev.get("difficulty", 0.5)),
                question_id=ev.get("question_id"),
                timestamp=float(ev.get("timestamp", time.time())),
            ))

        # 解析 mastery_map (可选)
        mastery_map = body.get("mastery_map")

        try:
            result = self._irt.estimate_ability(
                learner_id=learner_id,
                events=events,
                mastery_map=mastery_map,
            )
            return JSONResponse(_ok(_safe_dump(result)))
        except Exception as e:
            _logger.exception("IRT 能力估计失败")
            return JSONResponse(_err(-32400, "能力估计失败", str(e)), status_code=500)

    # ---- CAT 自适应出题 (POST /l2/irt/next-question) ----

    async def irt_next_question(self, request: Request) -> JSONResponse:
        """POST /l2/irt/next-question — CAT 自适应出题.

        基于当前 θ 的最大 Fisher 信息准则选题 (catR 标准)。
        融合模式: 结合 BKT 掌握度 (0.3~0.7 ZPD 区) 进行联合选题。

        请求体:
            learner_id: 学习者 ID
            available_items: 可用题目列表 [{"item_id", "a", "b", "c"}]
            administered_ids: 已答题目 ID 列表
            mastery_map: {item_id: p_mastery} (可选, 融合模式)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        learner_id = body.get("learner_id", "")
        available_items = body.get("available_items", [])
        administered_ids = set(body.get("administered_ids", []))
        mastery_map = body.get("mastery_map")

        # 获取当前 θ (从能力快照)
        snapshot = self._irt.get_ability_snapshot(learner_id)
        self._irt._current_theta = snapshot.get("theta", 0.0)

        try:
            if mastery_map is not None:
                # 融合模式: BKT+IRT 联合选题
                chosen = self._irt.select_next_item_fusion(
                    available_items=available_items,
                    administered_ids=administered_ids,
                    mastery_map=mastery_map,
                )
            else:
                # 标准模式: Fisher 信息最大化
                chosen = self._irt.select_next_item(
                    available_items=available_items,
                    administered_ids=administered_ids,
                )

            if chosen is not None:
                return JSONResponse(_ok({
                    "item_id": chosen.get("item_id"),
                    "a": chosen.get("a"),
                    "b": chosen.get("b"),
                    "c": chosen.get("c"),
                    "selection_strategy": "bkt_irt_fusion" if mastery_map else "fisher_info",
                }))
            else:
                return JSONResponse(_ok({
                    "item_id": None,
                    "selection_strategy": "fisher_info",
                }))
        except Exception as e:
            _logger.exception("CAT 选题失败")
            return JSONResponse(_err(-32400, "选题失败", str(e)), status_code=500)

    # ---- MMLE 题库校准 (POST /l2/irt/calibrate) ----

    async def irt_calibrate(self, request: Request) -> JSONResponse:
        """POST /l2/irt/calibrate — MMLE 题库参数校准.

        使用 EM 算法 (Bock & Aitkin 1981) 进行边际最大似然估计,
        校准题目参数 (a, b, c)。参数约束: a∈[0.3,3.0], b∈[-3,3], c∈[0,0.5]。

        请求体:
            responses_by_learner: {learner_id: [(item_params, correct), ...]}
            n_iterations: 最大迭代次数 (默认 50)
            convergence_threshold: 收敛阈值 (默认 1e-4)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        responses_by_learner = body.get("responses_by_learner", {})
        n_iterations = int(body.get("n_iterations", 50))
        convergence_threshold = float(body.get("convergence_threshold", 1e-4))

        # 转换 tuple 格式 (JSON 中 list → Python tuple)
        parsed: dict[str, list[tuple[dict[str, Any], bool]]] = {}
        for learner_id, responses in responses_by_learner.items():
            parsed[learner_id] = [
                (dict(r[0]), bool(r[1])) for r in responses
            ]

        try:
            result = self._irt.irt_estimator.estimate_mmle(
                responses_by_learner=parsed,
                n_iterations=n_iterations,
                convergence_threshold=convergence_threshold,
            )
            return JSONResponse(_ok({
                "items": _safe_dump(result),
            }))
        except Exception as e:
            _logger.exception("MMLE 校准失败")
            return JSONResponse(_err(-32400, "校准失败", str(e)), status_code=500)

    # ---- 能力快照 (GET /l2/irt/ability/{learner_id}) ----

    async def irt_ability_snapshot(self, request: Request) -> JSONResponse:
        """GET /l2/irt/ability/{learner_id} — 获取能力快照.

        返回学习者当前 IRT 能力状态 (theta, se, response_count)。
        无作答数据时回退群体先验 (theta=0.0, se=0.3)。
        """
        learner_id = request.path_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少路径参数: learner_id"), status_code=400)

        try:
            snapshot = self._irt.get_ability_snapshot(learner_id)
            return JSONResponse(_ok(_safe_dump(snapshot)))
        except Exception as e:
            _logger.exception("获取能力快照失败")
            return JSONResponse(_err(-32400, "获取快照失败", str(e)), status_code=500)

    # ---- BKT 单 KP 在线更新 (POST /l2/bkt/update) ----

    async def bkt_update(self, request: Request) -> JSONResponse:
        """POST /l2/bkt/update — BKT 单 KP 在线更新.

        对应设计文档 §7.3。处理单条答题事件, 更新 BKT 掌握度。

        请求体:
            learner_id: 学习者 ID (必填)
            kp_id: 知识点 ID (必填)
            correct: 是否答对 (必填)
            difficulty: 题目难度 [0,1], 默认 0.5
            timestamp: 事件时间戳, 默认当前时间
        """
        if self._bkt is None:
            return JSONResponse(_err(-32401, "BKT 服务不可用"), status_code=503)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        learner_id = body.get("learner_id")
        kp_id = body.get("kp_id")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少必填参数: learner_id"), status_code=400)
        if not kp_id:
            return JSONResponse(_err(-32700, "缺少必填参数: kp_id"), status_code=400)

        correct = bool(body.get("correct", False))
        difficulty = float(body.get("difficulty", 0.5))
        timestamp = float(body.get("timestamp", time.time()))

        event = AnswerEvent(
            learner_id=learner_id,
            kp_id=kp_id,
            correct=correct,
            difficulty=difficulty,
            timestamp=timestamp,
        )

        try:
            output = self._bkt.process(event)
            return JSONResponse(_ok(_safe_dump(output)))
        except Exception as e:
            _logger.exception("BKT 更新失败")
            return JSONResponse(_err(-32400, "BKT 更新失败", str(e)), status_code=500)


    # ---- 练习 (GET /l2/practice/questions / POST /l2/practice/answer) ----

    async def practice_questions(self, request: Request) -> JSONResponse:
        """GET /l2/practice/questions — 自适应出题.

        查询参数:
            learner_id: 学习者 ID (必填, 用于薄弱点定位与确定性随机).
            count: 出题数量, 默认 5, 最大 10.
        """
        if self._practice is None:
            return JSONResponse(_err(-32401, "练习服务不可用"), status_code=503)

        qp = request.query_params
        learner_id = qp.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少查询参数: learner_id"), status_code=400)
        # 防御: count 非法 (空/非数字) 时回退默认 5
        try:
            count = min(int(qp.get("count", 12)), 20)
        except (TypeError, ValueError):
            count = 12
        target_kps = tuple(
            dict.fromkeys(
                item.strip()
                for item in str(qp.get("kp_ids", "") or "").split(",")
                if item.strip()
            )
        )

        mastery: dict[str, float] = {}
        if self._profile is not None:
            snap = self._profile.get_profile_snapshot(learner_id)
            if snap is not None:
                mastery = dict(snap.kp_mastery)

        try:
            questions = self._practice.select_questions(
                learner_id,
                count=count,
                mastery=mastery,
                target_kps=target_kps,
            )
            payload = [self._practice.public_question(q) for q in questions]
            _logger.info("练习出题 learner=%s count=%d (薄弱优先)", learner_id, len(payload))
            return JSONResponse(_ok({
                "learner_id": learner_id,
                "count": len(payload),
                "questions": payload,
                "target_kps": list(target_kps),
                "availability": (
                    "AVAILABLE" if payload else "NO_AUTHORED_QUESTION"
                ),
            }))
        except Exception as e:
            _logger.exception("练习出题失败")
            return JSONResponse(_err(-32400, "练习出题失败", str(e)), status_code=500)

    # ---- 动态练习 (GET /l2/practice/dynamic/questions, 模板化变题) ----

    async def practice_dynamic_questions(self, request: Request) -> JSONResponse:
        """GET /l2/practice/dynamic/questions — 动态变题 (知识点不变, 题目/题型/选项每次不同).

        查询参数:
            learner_id: 学习者 ID (薄弱点定位 + 确定性随机).
            count: 出题数量, 默认 5, 最大 10.
        """
        if self._practice is None or self._dynamic_engine is None:
            return JSONResponse(_err(-32401, "练习服务不可用"), status_code=503)

        qp = request.query_params
        learner_id = qp.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少查询参数: learner_id"), status_code=400)
        # 防御: count 非法 (空/非数字) 时回退默认 5
        try:
            count = min(int(qp.get("count", 12)), 20)
        except (TypeError, ValueError):
            count = 12

        mastery: dict[str, float] = {}
        if self._profile is not None:
            snap = self._profile.get_profile_snapshot(learner_id)
            if snap is not None:
                mastery = dict(snap.kp_mastery)

        try:
            questions = self._dynamic_engine.select_and_generate(
                learner_id, count=count, mastery=mastery)
            payload = [self._practice.public_question(q) for q in questions]
            for q, p in zip(questions, payload):
                p["dynamic"] = True
                p["type"] = q.get("type", "choice")
            _logger.info("动态变题 learner=%s count=%d (薄弱优先, 题型轮换)", learner_id, len(payload))
            return JSONResponse(_ok({
                "learner_id": learner_id,
                "count": len(payload),
                "questions": payload,
                "mode": "dynamic",
            }))
        except Exception as e:
            _logger.exception("动态变题失败")
            return JSONResponse(_err(-32400, "动态变题失败", str(e)), status_code=500)

    async def practice_answer(self, request: Request) -> JSONResponse:
        """POST /l2/practice/answer — 判题并联动更新画像.

        请求体:
            learner_id: 学习者 ID (必填)
            qid: 题目 ID (必填)
            selected: 所选选项下标 (多选传逗号串如 "0,2"; 填空传 -1)
            text_answer: 填空作答文本 (填空题型必填)
        """
        if self._practice is None:
            return JSONResponse(_err(-32401, "练习服务不可用"), status_code=503)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        learner_id = body.get("learner_id", "")
        qid = body.get("qid", "")
        if not learner_id or not qid:
            return JSONResponse(_err(-32700, "缺少必填参数: learner_id / qid"), status_code=400)
        selected = body.get("selected", -1)
        text_answer = body.get("text_answer")
        attempt_purpose = str(body.get("attempt_purpose") or "DIAGNOSTIC").upper()
        allowed_purposes = {
            "ROUTE_VERIFY",
            "DIAGNOSTIC",
            "REQUIRED_PRACTICE",
            "STAGED_ASSESSMENT",
        }
        if attempt_purpose not in allowed_purposes:
            return JSONResponse(
                _err(-32602, "attempt_purpose 非法"),
                status_code=400,
            )

        try:
            result = self._practice.answer(
                learner_id, qid, selected, self._bkt, self._profile,
                text_answer=text_answer,
                update_models=attempt_purpose != "ROUTE_VERIFY",
            )
            runtime_metrics = dict(result.pop("_runtime_metrics", {}) or {})
            result["attempt_purpose"] = attempt_purpose
            if self._practice_observer is not None:
                try:
                    observed_result = dict(result)
                    observed_result["_runtime_metrics"] = runtime_metrics
                    self._practice_observer(dict(body), observed_result)
                except Exception as exc:  # noqa: BLE001 - memory observation cannot fail scoring
                    _logger.warning("练习教学效果观察写入失败: %s", exc)
            # WS 推流: 判题后向 broadcast 通道发布 bkt_update (前端动态可视化刷新)
            try:
                from dy3_polaris.l7.api.websocket import HUB

                snapshot = self._profile.get_profile_snapshot(learner_id)
                km = snapshot.kp_mastery if snapshot is not None else {}
                HUB.broadcast("broadcast", "bkt_update", {
                    "learner_id": learner_id,
                    "kp_id": result.get("kp_id"),
                    "p_mastery_after": result.get("p_mastery_after"),
                    "kp_mastery": km,
                    "timestamp": time.time(),
                })
            except Exception:  # noqa: BLE001  WS 发布失败不影响判题
                pass
            return JSONResponse(_ok(result))
        except L2Error as e:
            return JSONResponse(_err(-32300, e.detail), status_code=400)
        except Exception as e:
            _logger.exception("练习判题失败")
            return JSONResponse(_err(-32400, "练习判题失败", str(e)), status_code=500)

    # ---- 学情事件采集 (POST /l2/event/collect) ----

    async def collect_event(self, request: Request) -> JSONResponse:
        """POST /l2/event/collect — 采集学情事件并写入画像交互日志.

        请求体:
            learner_id: 学习者 ID (必填)
            event_type: view/query/behavior/session (必填)
            detail: 事件细节 (如浏览的知识点/提问内容/行为动作)
            可选: session_id, ts

        作用: 打通"学情采集"闭环首环 — 浏览/提问/行为事件进入画像 extras.interaction_log,
        供画像更新/诊断 Agent 后续消费。
        """
        if self._profile is None:
            return JSONResponse(_err(-32401, "画像服务不可用"), status_code=503)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        learner_id = body.get("learner_id", "")
        event_type = body.get("event_type", "")
        if not learner_id or event_type not in ("view", "query", "behavior", "session"):
            return JSONResponse(
                _err(-32602, "缺少必填参数或事件类型非法: learner_id / event_type"),
                status_code=400,
            )

        try:
            profile = self._profile.get_profile_snapshot(learner_id)
            if profile is None:
                # 未初测用户也可采集 (生成初始画像骨架)
                from dy3_polaris.l2.models import LearnerSnapshot

                profile = LearnerSnapshot(
                    learner_id=learner_id,
                    snapshot_ts=time.time(),
                    level="unknown",
                    learning_style="unknown",
                    bloom_target="unknown",
                    confidence=0.0,
                )
            extras = dict(getattr(profile, "extras", {}) or {})
            log = list(extras.get("interaction_log", []) or [])
            log.append({
                "ts": body.get("ts") or time.time(),
                "event_type": event_type,
                "detail": str(body.get("detail") or "")[:300],
                "session_id": body.get("session_id", ""),
            })
            extras["interaction_log"] = log[-200:]
            profile.extras = extras
            self._profile.store.save_profile(learner_id, profile)
            return JSONResponse(_ok({
                "learner_id": learner_id,
                "event_type": event_type,
                "recorded": True,
                "total_events": len(log),
            }))
        except Exception as e:
            _logger.exception("学情事件采集失败")
            return JSONResponse(_err(-32400, "学情事件采集失败", str(e)), status_code=500)

    # ---- 画像写入网关 (PUT /l2/profile/{learner_id}/mastery) ----

    async def profile_mastery_update(self, request: Request) -> JSONResponse:
        """PUT /l2/profile/{learner_id}/mastery — L2 唯一画像写方 (乐观锁).

        请求体:
            version: 调用方持有的版本号 (必填, 来自 GET /l2/profile/{id})
            updates: {kp_mastery?: {...}, extras?: {...}, confidence?: number,
                      learning_style?: str, bloom_target?: str}

        行为:
        - 全量重算画像 (BKT 追踪状态 + IRT → ProfileBuilder)
        - 合并 updates (extras 日志追加去重, kp_mastery 覆盖)
        - 乐观锁: version 不匹配 → 409 Conflict (返回最新 version, 调用方重拉重试)

        响应 data: {learner_id, version, kp_mastery, weak_kps, confidence, updated: True}
        """
        if self._profile is None:
            return JSONResponse(_err(-32401, "画像服务不可用"), status_code=503)
        learner_id = request.path_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少路径参数: learner_id"), status_code=400)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        version = body.get("version")
        if version is None or not isinstance(version, int):
            return JSONResponse(
                _err(-32602, "缺少必填参数: version (整数, 来自 GET /l2/profile/{id})"),
                status_code=400,
            )
        updates = body.get("updates") or {}

        try:
            merged = self._profile.apply_update(
                learner_id,
                updates=updates,
                expected_version=version,
            )
            return JSONResponse(_ok({
                "learner_id": learner_id,
                "version": merged.version,
                "kp_mastery": merged.kp_mastery,
                "weak_kps": merged.weak_kps,
                "confidence": merged.confidence,
                "updated": True,
            }))
        except ProfileConflictError as exc:
            # 乐观锁冲突: 409 + 最新版本, 调用方重新拉取后重试
            return JSONResponse(
                _err(-32310, "PROFILE_CONFLICT", str(exc),
                     current_version=exc.current_version),
                status_code=409,
            )
        except Exception as e:
            _logger.exception("画像写入失败")
            return JSONResponse(_err(-32400, "画像写入失败", str(e)), status_code=500)

    # ---- 画像查询 (GET /l2/profile/{learner_id}) ----

    async def profile_get(self, request: Request) -> JSONResponse:
        """GET /l2/profile/{learner_id} — 获取学习者画像 (动态).

        返回画像快照 (含动态字段):
        - kp_mastery: 有效掌握度 (应用遗忘衰减, 随时间流逝下浮)
        - raw_kp_mastery: 原始掌握度 (写回值)
        - overall_mastery: 总体掌握度 (百分制换算前端处理)
        - dimensions: 四域 (A/B/C/D) 平均掌握度 (游戏面板数据源)
        - decay_hours: 距上次学习的间隔小时
        未初测用户返回默认 0 画像 (不 404)。
        """
        if self._profile is None:
            return JSONResponse(_err(-32401, "画像服务不可用"), status_code=503)

        learner_id = request.path_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少路径参数: learner_id"), status_code=400)

        try:
            # KP 名称映射 (42 条, 单点 kp_catalog): 前端 ID → 可读中文名
            try:
                from dy3_polaris.l2.kp_catalog import _KP_NAMES as _KP_NAMES_SRC

                kp_names = dict(_KP_NAMES_SRC)
            except Exception:  # noqa: BLE001
                kp_names = {}
            snapshot = self._profile.get_profile_snapshot(learner_id)
            if snapshot is None:
                # 未初测用户: 默认 0 画像
                return JSONResponse(_ok({
                    "learner_id": learner_id,
                    "snapshot_ts": 0.0,
                    "kp_mastery": {},
                    "raw_kp_mastery": {},
                    "overall_mastery": 0.0,
                    "dimensions": {},
                    "theta": None,
                    "level": "novice",
                    "learning_style": "reading",
                    "bloom_target": "understand",
                    "weak_kps": [],
                    "confidence": 0.0,
                    "decay_hours": 0.0,
                    "extras": {},
                    "kp_names": kp_names,
                    "initial_assessed": False,
                }))
            data = snapshot.to_dict() if hasattr(snapshot, "to_dict") else _safe_dump(snapshot)
            data = self._apply_decay(data)
            data["initial_assessed"] = bool(data.get("kp_mastery"))
            data["kp_names"] = kp_names
            return JSONResponse(_ok(data))
        except Exception as e:
            _logger.exception("获取画像失败")
            return JSONResponse(_err(-32400, "获取画像失败", str(e)), status_code=500)

    @staticmethod
    def _apply_decay(data: dict) -> dict:
        """应用遗忘衰减: 掌握度随时间流逝下浮 (7 天阈值, 指数衰减)."""
        import time as _time

        from dy3_polaris.l2.knowledge_tracer.forgetting import ForgettingModel as ForgettingCurve

        km = dict(data.get("kp_mastery", {}) or {})
        now = _time.time()
        snapshot_ts = float(data.get("snapshot_ts", now) or now)
        delta_hours = max(0.0, (now - snapshot_ts) / 3600.0)
        fc = ForgettingCurve()
        effective = {
            kp: round(fc.decay(float(m), delta_hours), 4)
            for kp, m in km.items()
        }
        data["raw_kp_mastery"] = km
        data["kp_mastery"] = effective
        data["decay_hours"] = round(delta_hours, 2)
        # 薄弱点与衰减后掌握度同口径重算 (避免 weak_kps 与展示值不同步)
        from dy3_polaris.l2.profile_builder.builder import _WEAK_KP_THRESHOLD

        data["weak_kps"] = sorted(
            k for k, m in effective.items() if m < _WEAK_KP_THRESHOLD
        )
        data["overall_mastery"] = (
            round(sum(effective.values()) / len(effective), 4) if effective else 0.0
        )
        dims: dict[str, list] = {}
        for kp, m in effective.items():
            dims.setdefault(str(kp or "")[:1], []).append(m)
        # 保留 to_dict 里的 E(行为) 维度, 只重算 A/B/C/D 四域掌握度 (五维雷达)
        _e_dim = (data.get("dimensions") or {}).get("E", 0.0)
        data["dimensions"] = {
            d: round(sum(vs) / len(vs), 4) for d, vs in dims.items()
        }
        data["dimensions"]["E"] = round(float(_e_dim), 4)
        return data

    # ---- 薄弱知识点 (GET /l2/profile/{learner_id}/weak-points) ----

    async def profile_weak_points(self, request: Request) -> JSONResponse:
        """GET /l2/profile/{learner_id}/weak-points — 获取薄弱知识点."""
        if self._profile is None:
            return JSONResponse(_err(-32401, "画像服务不可用"), status_code=503)

        learner_id = request.path_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少路径参数: learner_id"), status_code=400)

        try:
            snapshot = self._profile.get_profile_snapshot(learner_id)
            if snapshot is None:
                return JSONResponse(_ok({"weak_kps": []}))

            # 从快照中提取薄弱知识点
            if hasattr(snapshot, "to_dict"):
                data = snapshot.to_dict()
            elif isinstance(snapshot, dict):
                data = snapshot
            else:
                data = _safe_dump(snapshot)

            weak_kps = data.get("weak_kps", [])
            kp_mastery = data.get("kp_mastery", {})

            # 补充每个薄弱 KP 的掌握度详情
            weak_details = []
            for kp_id in weak_kps:
                weak_details.append({
                    "kp_id": kp_id,
                    "mastery": kp_mastery.get(kp_id, 0.0),
                })

            return JSONResponse(_ok({
                "learner_id": learner_id,
                "weak_kps": weak_kps,
                "weak_details": weak_details,
            }))
        except Exception as e:
            _logger.exception("获取薄弱知识点失败")
            return JSONResponse(_err(-32400, "获取薄弱知识点失败", str(e)), status_code=500)

    # ---- 画像置信度 (GET /l2/profile/{learner_id}/confidence) ----

    async def profile_confidence(self, request: Request) -> JSONResponse:
        """GET /l2/profile/{learner_id}/confidence — 获取画像置信度."""
        if self._profile is None:
            return JSONResponse(_err(-32401, "画像服务不可用"), status_code=503)

        learner_id = request.path_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少路径参数: learner_id"), status_code=400)

        try:
            snapshot = self._profile.get_profile_snapshot(learner_id)
            if snapshot is None:
                return JSONResponse(_ok({"confidence": 0.0, "data_available": False}))

            if hasattr(snapshot, "to_dict"):
                data = snapshot.to_dict()
            elif isinstance(snapshot, dict):
                data = snapshot
            else:
                data = _safe_dump(snapshot)

            confidence = data.get("confidence", 0.0)
            return JSONResponse(_ok({
                "learner_id": learner_id,
                "confidence": confidence,
                "data_available": True,
            }))
        except Exception as e:
            _logger.exception("获取置信度失败")
            return JSONResponse(_err(-32400, "获取置信度失败", str(e)), status_code=500)

    # ---- 知识点目录 (GET /l2/kp-catalog, SSOT 单点) ----

    async def kp_catalog_get(self, request: Request) -> JSONResponse:
        """GET /l2/kp-catalog — 知识点目录 (42 KP 单点, 供 L7/前端引用).

        返回 {domains[], kp[] (kp_id/name/domain/level/kg_nodes/covered_by_bank), total}
        消除"展示层持有领域目录"的架构倒挂: 前端与 L7 均从本端点/模块获取.
        """
        from dy3_polaris.l2.kp_catalog import to_dict

        return JSONResponse(_ok(to_dict()))

    # ---- 记忆更新 (POST /l2/memory/update) ----

    async def memory_update(self, request: Request) -> JSONResponse:
        """POST /l2/memory/update — 更新记忆状态.

        请求体:
            learner_id: 学习者 ID (必填)
            kp_id: 知识点 ID (必填)
            correct: 是否答对 (必填)
            difficulty: 题目难度 [0,1], 默认 0.5
        """
        if self._memory is None:
            return JSONResponse(_err(-32401, "记忆服务不可用"), status_code=503)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        learner_id = body.get("learner_id")
        kp_id = body.get("kp_id")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少必填参数: learner_id"), status_code=400)
        if not kp_id:
            return JSONResponse(_err(-32700, "缺少必填参数: kp_id"), status_code=400)

        correct = bool(body.get("correct", False))
        difficulty = float(body.get("difficulty", 0.5))

        try:
            from dy3_polaris.l2.interaction.event_types import AnswerEvent

            event = AnswerEvent(
                learner_id=learner_id,
                kp_id=kp_id,
                correct=correct,
                difficulty=difficulty,
            )
            output = self._memory.process(event)
            return JSONResponse(_ok(_safe_dump(output)))
        except Exception as e:
            _logger.exception("记忆更新失败")
            return JSONResponse(_err(-32400, "记忆更新失败", str(e)), status_code=500)

    # ---- 记忆状态查询 (GET /l2/memory/{learner_id}) ----

    async def memory_get(self, request: Request) -> JSONResponse:
        """GET /l2/memory/{learner_id} — 获取记忆状态."""
        if self._memory is None:
            return JSONResponse(_err(-32401, "记忆服务不可用"), status_code=503)

        learner_id = request.path_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少路径参数: learner_id"), status_code=400)

        try:
            state = self._memory.get_memory_state(learner_id)
            if state is None:
                return JSONResponse(_ok({"learner_id": learner_id, "kp_retentions": {}}))
            return JSONResponse(_ok(_safe_dump(state) if not isinstance(state, dict) else state))
        except Exception as e:
            _logger.exception("获取记忆状态失败")
            return JSONResponse(_err(-32400, "获取记忆状态失败", str(e)), status_code=500)


# ============================================================
# L2Router
# ============================================================

class L2Router:
    """L2 个性化层 REST API 路由器.

    将 IRTTracingService / BKTTracingService 的核心功能暴露为 RESTful API。
    遵循与 L3Router / L6Router 一致的设计模式。

    使用示例::

        from dy3_polaris.l2.ability_assessor import IRTTracingService
        from dy3_polaris.l2.api import L2Router

        irt_service = IRTTracingService(enable_enhanced=True)
        irt_service.set_item_bank([...])
        router = L2Router(irt_service=irt_service)
        app = router.create_app()

        # 独立运行
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8002)

        # 或嵌入到主应用
        from starlette.routing import Mount
        main_routes = [
            Mount("/l2", app=router.create_app()),
        ]

    Args:
        irt_service: IRT 全链路编排服务 (必填).
        bkt_service: BKT 全链路编排服务 (可选, None 时延迟初始化).
        cors_origins: CORS 允许的源 (默认 ["*"]).
    """

    def __init__(
        self,
        irt_service: IRTTracingService,
        bkt_service: Any | None = None,
        profile_service: Any | None = None,
        memory_service: Any | None = None,
        cors_origins: list[str] | None = None,
        practice_observer: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> None:
        """初始化 L2 路由器.

        Args:
            irt_service: IRT 全链路编排服务 (必填).
            bkt_service: BKT 全链路编排服务 (可选, None 时延迟初始化).
            profile_service: 画像全链路编排服务 (可选).
            memory_service: 记忆全链路编排服务 (可选).
            cors_origins: CORS 允许的源 (默认 ["*"]).
        """
        self._irt_service = irt_service
        self._cors_origins = cors_origins or ["*"]
        self._handlers = _RouteHandlers(
            irt_service=irt_service,
            bkt_service=bkt_service,
            profile_service=profile_service,
            memory_service=memory_service,
            practice_observer=practice_observer,
        )

    def create_app(self) -> Starlette:
        """创建 Starlette 应用实例.

        Returns:
            配置好的 Starlette 应用, 可直接传给 uvicorn.run()
            或通过 Mount 嵌入到主应用。
        """
        h = self._handlers

        routes = [
            # 健康检查
            Route("/health", h.health, methods=["GET"]),

            # IRT 能力评估
            Route("/irt/estimate", h.irt_estimate, methods=["POST"]),
            Route("/irt/next-question", h.irt_next_question, methods=["POST"]),
            Route("/irt/calibrate", h.irt_calibrate, methods=["POST"]),
            Route("/irt/ability/{learner_id}", h.irt_ability_snapshot, methods=["GET"]),

            # BKT 知识追踪
            Route("/bkt/update", h.bkt_update, methods=["POST"]),

            # 练习 (答题 → BKT → 画像联动)
            Route("/practice/questions", h.practice_questions, methods=["GET"]),
            Route("/practice/dynamic/questions", h.practice_dynamic_questions, methods=["GET"]),
            Route("/practice/answer", h.practice_answer, methods=["POST"]),

            # 学情事件采集 (浏览/提问/行为 → 画像交互日志)
            Route("/event/collect", h.collect_event, methods=["POST"]),

            # 画像查询
            Route("/profile/{learner_id}", h.profile_get, methods=["GET"]),
            Route("/profile/{learner_id}/weak-points", h.profile_weak_points, methods=["GET"]),
            Route("/profile/{learner_id}/confidence", h.profile_confidence, methods=["GET"]),
            # 画像写入网关 (L2 唯一写方, 乐观锁): PUT /profile/{id}/mastery
            Route("/profile/{learner_id}/mastery", h.profile_mastery_update, methods=["PUT"]),
            # 知识点目录 (SSOT 单点, 供 L7/前端拉取)
            Route("/kp-catalog", h.kp_catalog_get, methods=["GET"]),

            # 记忆管理
            Route("/memory/update", h.memory_update, methods=["POST"]),
            Route("/memory/{learner_id}", h.memory_get, methods=["GET"]),
        ]

        middleware = []
        if self._cors_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=self._cors_origins,
                    allow_methods=["*"] if "*" in self._cors_origins
                                   else ["GET", "POST", "PUT", "DELETE"],
                    allow_headers=["*"],
                )
            )

        app = Starlette(routes=routes, middleware=middleware)
        return app

    def get_routes_summary(self) -> list[dict[str, str]]:
        """获取所有路由摘要 (用于文档/发现).

        Returns:
            [{"path": ..., "methods": [...], "description": ...}]
        """
        return [
            {"path": "/health", "methods": ["GET"], "description": "L2 个性化层健康检查"},
            {"path": "/irt/estimate", "methods": ["POST"], "description": "IRT 能力参数估计"},
            {"path": "/irt/next-question", "methods": ["POST"], "description": "CAT 自适应出题"},
            {"path": "/irt/calibrate", "methods": ["POST"], "description": "MMLE 题库参数校准"},
            {"path": "/irt/ability/{learner_id}", "methods": ["GET"], "description": "获取能力快照"},
            {"path": "/bkt/update", "methods": ["POST"], "description": "BKT 单 KP 在线更新"},
            {"path": "/profile/{learner_id}", "methods": ["GET"], "description": "获取学习者画像"},
            {"path": "/profile/{learner_id}/weak-points", "methods": ["GET"], "description": "获取薄弱知识点"},
            {"path": "/profile/{learner_id}/confidence", "methods": ["GET"], "description": "获取画像置信度"},
            {"path": "/memory/update", "methods": ["POST"], "description": "更新记忆状态"},
            {"path": "/memory/{learner_id}", "methods": ["GET"], "description": "获取记忆状态"},
        ]


__all__ = ["L2Router"]
