"""L7 前端系统 M-F2/M-F3 — 集成测试.

覆盖:
1. UnifiedApp 挂载 L7 (Mount /l7 + /api/info 发现 + /health 聚合)
2. 根路径静态页面 (GET / → index.html)
3. 静态资源托管 (/static/assets/*)
4. /api/query 端到端聚合端点 (L2 画像 → L4 决策 → L5 会话)
5. 参数校验与错误处理
"""

from __future__ import annotations

import pytest

from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def client() -> TestClient:
    """构建全栈应用 (含 L7)."""
    builder = UnifiedApp.create_full_app_builder()
    return TestClient(builder.create_app())


class TestL7Mounted:
    """L7 挂载到 UnifiedApp."""

    def test_health_includes_l7(self, client):
        data = client.get("/health").json()["data"]
        assert "l7" in data["layers"]
        assert data["layers"]["l7"]["status"] == "healthy"

    def test_api_info_discovers_l7(self, client):
        info = client.get("/api/info").json()["data"]
        assert "L7" in info["layers"]
        l7_paths = [e["path"] for e in info["endpoints"] if e["layer"] == "L7"]
        assert any("/l7/api/v1/render" in p for p in l7_paths)
        assert any("/l7/api/v1/artifacts" in p for p in l7_paths)

    def test_l7_health_endpoint(self, client):
        data = client.get("/l7/api/v1/health").json()["data"]
        assert data["layer"] == "L7"

    def test_l7_render_registry_has_renderers(self, client):
        mimes = client.get("/l7/api/v1/registry/mime-types").json()["data"]
        assert "text/vnd.dy3+markdown" in mimes
        assert "application/vnd.dy3.chart+json" in mimes


class TestStaticFrontend:
    """根路径与静态资源."""

    def test_root_returns_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Dy3+ Polaris" in resp.text
        assert "activity-bar" in resp.text
        assert "sidebar" in resp.text

    def test_css_served(self, client):
        resp = client.get("/static/assets/app.css")
        assert resp.status_code == 200
        assert "--accent" in resp.text

    def test_js_served(self, client):
        resp = client.get("/static/assets/app.js")
        assert resp.status_code == 200
        assert "kb" in resp.text
        assert "LEARNER_NAV" in resp.text or "LN=" in resp.text or "var LN=" in resp.text  # 角色导航

    def test_model_settings_use_current_deepseek_ids_and_do_not_persist_secrets(self, client):
        js = client.get("/static/assets/mf6-features.js").text

        assert "deepseek-v4-flash" in js
        assert "deepseek-chat" not in js
        assert "deepseek-r1" not in js
        assert "localStorage.setItem('dy3_api_key'" not in js
        assert "localStorage.getItem('dy3_api_key'" not in js

    def test_sidebar_width_ratio_css(self, client):
        """侧栏 ≤ 1/5 屏宽约束: 检查 CSS 变量或宽度约束."""
        resp = client.get("/static/assets/app.css")
        css = resp.text
        assert ("--sidebar-max:20vw" in css) or ("20vw" in css) or ("max-width" in css and "sidebar" in css)

    def test_product_navigation_has_five_learning_canvases(self, client):
        js = client.get("/static/assets/app.js").text
        learner_nav = js[js.index("var LN="):js.index("var MN=")]
        for label in (
            r"\u5b66\u4e60\u603b\u89c8",
            r"\u4efb\u52a1\u5de5\u4f5c\u533a",
            r"\u534f\u540c\u5206\u6790",
            r"\u77e5\u8bc6\u8bc1\u636e",
            r"\u6210\u957f\u8def\u5f84",
        ):
            assert label in learner_nav
        assert learner_nav.count("sub:[") == 5
        for hidden_view in ("atomic-viz", "kb-graph", "practice", "users", "gov-review"):
            assert hidden_view not in learner_nav

    def test_r08b1_advanced_views_do_not_inject_top_level_navigation(self, client):
        atomic = client.get("/static/assets/mf8-atomic-viz.js").text
        match = client.get("/static/assets/mf9-match-report.js").text
        for module in (atomic, match):
            assert "function injectNav" not in module
            assert "sidebar-rebuilt', injectNav" not in module
            assert "btn.setAttribute('data-view', VIEW)" not in module

    def test_three_primary_spaces_and_contextual_views_remain_routable(self, client):
        app = client.get("/static/assets/app.js").text
        learner_nav = app[app.index("var LN="):app.index("var MN=")]
        for view in ("query", "kb", "learn-weak"):
            assert "id:'" + view + "'" in learner_nav
        features = client.get("/static/assets/mf6-features.js").text
        for route in (
            "'query': renderQuery",
            "'agents-chain': renderAgentChain",
            "'kb': renderR04KnowledgeWorkspace",
            "'learn-weak': renderR04LearningWorkspace",
            "'overview': function ()",
        ):
            assert route in features

    def test_r04a_core_task_uses_honest_wait_and_result_hierarchy(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "任务处理中" in js
        assert "结果返回后可查看真实 Agent 协同过程" in js
        assert "stepTimer = setInterval" not in js
        for label in ("用户任务", "任务类型", "最终回答", "科学审核", "科学证据", "下一步学习"):
            assert label in js

    def test_r04a_default_result_bypasses_legacy_consensus_renderer(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        call = js.index("renderR04TaskResult(d, q, res, inp);")
        legacy = js.index("var cands =", call)
        assert "return;" in js[call:legacy]
        assert "'/api/learning-tasks/' + encodeURIComponent(lid)" in js
        assert "dy3_last_task_public_" not in js

    def test_r04a_public_projection_only_and_no_private_contract_access(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function r04PublicData")
        end = js.index("function renderQuery", start)
        renderer = js[start:end]
        for public_field in ("task_events", "agent_trace", "collab_lines", "flow_events", "evidence", "review", "answer", "recommended_path"):
            assert public_field in renderer
        for private_field in ("_contract_candidate", "AgentInput", "CollaborationContext", "readiness"):
            assert private_field not in renderer

    def test_r04a_legacy_consensus_and_floating_assistant_not_loaded(self, client):
        html = client.get("/").text
        assert 'src="/static/assets/dp-collab.js' not in html
        assert 'src="/static/assets/mf7-assistant.js' not in html
        assert 'id="debatePanel" hidden' in html

    def test_r04b_home_is_research_mentor_story_not_dashboard(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderLearnerOverview")
        end = js.index("function renderAdminOverview", start)
        home = js[start:end]
        assert "面向稀土发光材料学习与研究" in home
        assert "理解任务" in home
        assert "检索证据" in home
        assert "科学审核" in home
        assert "学习决策" in home
        assert "renderGamePanel" not in home
        assert "stat-card" not in home
        for label in ("当前学习者", "当前水平", "学习状态", "下一步"):
            assert label in home

    def test_r04b_core_task_has_understanding_claim_evidence_and_guidance(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderR04TaskResult")
        end = js.index("function renderQuery", start)
        task = js[start:end]
        for label in ("任务类型", "解释目标", "学习层级 / 诊断", "最终回答", "Reviewer 结论", "关联结论", "下一步学习"):
            assert label in task
        assert "CURRENT 未提供 claim 级绑定" in task
        assert "recommended_path" in task

    def test_r04b_collaboration_counts_only_real_public_events(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderAgentChain")
        end = js.index("function renderR04KnowledgeWorkspace", start)
        collab = js[start:end]
        for label in ("实际参与 Agent", "真实 Subtask", "Evidence 更新", "Reviewer Challenge", "真实协同时间线"):
            assert label in collab
        for event_type in ("SUBTASK_READY", "EVIDENCE_RETRIEVED", "CHALLENGE_RAISED"):
            assert event_type in collab
        assert "Math.random" not in collab
        assert "setInterval" not in collab
        assert "AGENT_LINKS" not in collab

    def test_r04b_learning_and_knowledge_spaces_are_task_driven(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        learning_start = js.index("function renderR04LearningWorkspace")
        learning_end = js.index("Agent 列表视图", learning_start)
        learning = js[learning_start:learning_end]
        assert "当前状态" in learning
        assert "下一步建议" in learning
        assert "推荐任务" in learning
        assert "成长决策" in learning
        assert 'data-view-target="match-report"' in learning
        assert 'data-view-target="practice"' in learning
        assert "taskData.learning_resources" in learning
        assert "t1ResourceCards(taskData)" in learning
        assert "通用模板冒充任务资源" in learning
        assert "renderPersonalizedResources(g('growthResources'))" not in learning
        assert "BKT / 掌握度高级信息" in learning
        assert "AI 如何持续培养当前学习者" not in learning
        assert "判断统一来自公开 Learner Intelligence Report" not in learning
        assert "AI 教师如何陪伴一次学习" not in learning
        assert "三个空间共享同一个服务端学习任务" not in learning
        knowledge_start = js.index("function renderR04KnowledgeWorkspace")
        knowledge_end = js.index("视图路由钩子", knowledge_start)
        knowledge = js[knowledge_start:knowledge_end]
        assert "知识证据" in knowledge
        assert "为什么这个教学回答值得相信" not in knowledge
        assert "当前任务" in knowledge
        assert "/l3/retrieve/keyword" in knowledge
        assert "/l3/entities" not in knowledge
        assert 'data-view-target="atomic-viz"' in knowledge
        assert 'data-view-target="kb-provenance"' in knowledge
        assert "当前问题的概念关系图" in knowledge
        assert "t234ConceptGraph(data)" in knowledge
        assert "t234ConceptGraph(data)" in knowledge
        assert "当前问题的概念关系图" in knowledge

    def test_r08b1_contextual_advanced_routes_keep_existing_renderers(self, client):
        match = client.get("/static/assets/mf9-match-report.js").text
        atomic = client.get("/static/assets/mf8-atomic-viz.js").text
        features = client.get("/static/assets/mf6-features.js").text
        assert "var VIEW = 'match-report'" in match
        assert "e.detail.view === VIEW" in match
        assert "var VIEW = 'atomic-viz'" in atomic
        assert "e.detail.view === VIEW" in atomic
        assert "'practice': renderPractice" in features

    def test_current_task_strip_does_not_duplicate_primary_navigation(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function r08Journey")
        end = js.index("function r08BindJourney", start)
        journey = js[start:end]
        assert "当前任务" in journey
        assert "data-journey-view" not in journey
        assert "<button" not in journey
        assert "不代表虚假实时进度" not in journey
        for private_field in ("TeachingMemoryView", "KnowledgeLearningContext", "_contract_candidate"):
            assert private_field not in journey

    def test_r08b2_workspace_uses_real_profile_and_optional_background(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderLearnerOverview")
        end = js.index("function renderAdminOverview", start)
        home = js[start:end]
        assert "'/l2/profile/' + learnerId()" in home
        assert "'/api/match-report/' + encodeURIComponent(learnerId())" in home
        assert "'/l4/decision/next-action'" in home
        for label in ("我是谁，我现在在哪里，下一步是什么", "当前学习判断", "下一步行动"):
            assert label in home
        assert "仅有模型画像，尚无真实作答" in home
        assert "r08RecentTaskPanel(lastTask, lastQuestion)" in home
        assert "if (professional)" in home
        assert "if (learningGoal)" in home

    def test_learning_overview_uses_one_authoritative_view_and_three_data_layers(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderWorkspaceProduct")
        end = js.index("function renderLearnerOverview", start)
        workspace = js[start:end]
        assert "'/api/learning-workspace/' + encodeURIComponent(learnerId())" in workspace
        for label in ("LEARNING OVERVIEW", "01", "当前状态", "02", "学习路径", "03", "最近活动"):
            assert label in workspace
        assert "正在读取学习状态" in workspace
        assert "不了解你时也可以直接提问" not in workspace
        assert "UNKNOWN 不阻止" not in workspace  # policy is projected by the server, not hard-coded as state
        assert "当前真实能力覆盖" in workspace

    def test_p0_guest_continuity_and_practice_feedback_are_truthful(self, client):
        app = client.get("/static/assets/app.js").text
        features = client.get("/static/assets/mf6-features.js").text
        assert "localStorage.setItem('dy3_device_guest_id'" in app
        assert "sessionStorage.setItem('dy3_guest_id'" not in app
        assert "attempt_purpose" in features
        assert "REQUIRED_PRACTICE" in features
        assert "服务器已保存" in features
        assert "权威学习视图已刷新" in features
        assert "不在前端计算掌握度" in features

    def test_r08b2_core_task_exposes_only_returned_journey_facts(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        fact_start = js.index("function r08TaskFactFlow")
        fact_end = js.index("function r04AgentName", fact_start)
        facts = js[fact_start:fact_end]
        for label in ("学习者分析", "知识检索", "Agent 协同", "科学审核", "教学决策"):
            assert label in facts
        assert "未提供公开" in facts
        result_start = js.index("function renderR04TaskResult")
        result_end = js.index("function renderQuery", result_start)
        result = js[result_start:result_end]
        assert "r08-result-actions" not in result
        assert "data-r08-view" not in result
        assert "r08TaskFactFlow(data" in result

    def test_completed_task_refreshes_the_visible_task_context(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "function r08LatestTaskContext" in js
        result_start = js.index("function renderR04TaskResult")
        result_end = js.index("function renderQuery", result_start)
        result = js[result_start:result_end]
        assert "g('r08LatestTaskContext')" in result
        assert "r08LatestTaskContext(data, question)" in result
        query_start = js.index("function renderQuery")
        query_end = js.index("R-04B learning workspace", query_start)
        query = js[query_start:query_end]
        assert 'id="r08LatestTaskContext"' in query

    def test_r08b2_growth_has_past_present_future_and_three_support_types(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderR04LearningWorkspace")
        end = js.index("Agent 列表视图", start)
        growth = js[start:end]
        for label in ("过去", "现在", "未来", "知识学习资源", "科研学习任务", "分阶练习与分析"):
            assert label in growth
        assert "decision.reason || decision.rationale || decision.explanation" in growth
        assert "'/api/match-report/' + encodeURIComponent(learnerId())" in growth
        assert "公开 Learner Intelligence Report" not in growth
        assert "private Teaching Memory" not in growth

    def test_r08b2_task_driven_knowledge_is_not_overwritten_by_legacy_browser(self, client):
        features = client.get("/static/assets/mf6-features.js").text
        legacy = client.get("/static/assets/mf11-kb.js").text
        knowledge_start = features.index("function renderR04KnowledgeWorkspace")
        knowledge_end = features.index("视图路由钩子", knowledge_start)
        knowledge = features[knowledge_start:knowledge_end]
        assert "证据片段本身不等于支持关系" in knowledge
        assert "t5678ScientificGrounding(data)" in knowledge
        assert "Concept → Claim → Evidence → Source" in features
        assert "候选相关/仅提及" in features
        assert "sources" in knowledge
        assert "kp_names" in knowledge
        assert "MF11KnowledgeBrowser.renderLegacyEntityBrowser = render" in legacy
        assert "e.detail && e.detail.view === VIEW" not in legacy

    def test_r08b2_advanced_views_return_to_parent_space(self, client):
        match = client.get("/static/assets/mf9-match-report.js").text
        atomic = client.get("/static/assets/mf8-atomic-viz.js").text
        graph = client.get("/static/assets/mf10-kg-viz.js").text
        assert "返回成长决策" in match and "window.sv('learn-weak')" in match
        assert "返回知识与证据" in atomic and "window.sv('kb')" in atomic
        assert "返回知识与证据" in graph and "不代表 R06 Concept Relation" in graph

    def test_t5678_growth_uses_authoritative_report_and_keeps_unknown_distinct(self, client):
        match = client.get("/static/assets/mf9-match-report.js").text
        assert "renderAuthoritativeReport(data.report)" in match
        assert "真实作答支持的薄弱点" in match
        assert "尚无证据判断" in match
        assert "UNKNOWN 不视为未掌握" in match
        assert "R06 Concept 学习路径" in match
        assert "真实成长时间线" in match

    def test_t5678_practice_feedback_does_not_fake_agent_decision_or_mastery(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function buildIterationFeedback")
        end = js.index("function submitPractice", start)
        feedback = js[start:end]
        assert "单次正确不会被标记为“已掌握”" in feedback
        assert "一次错误不等于已确认薄弱点" in feedback
        assert "多智能体决策" not in feedback
        assert "Diagnosis" in feedback

    def test_r04b_default_experience_disables_simulated_polish(self, client):
        html = client.get("/").text
        assert 'src="/static/assets/ui-polish.js' not in html
        js = client.get("/static/assets/mf6-features.js").text
        assert "stepTimer = setInterval" not in js

    def test_r04b_three_competition_cases_are_available(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "为什么 Dy³⁺ 会产生黄蓝双发射？" in js
        assert "3000 K 是否一定更适合健康照明？" in js
        assert "如何公平比较两种 Dy³⁺ 发光材料？" in js
        assert "if (!challenge.length && !correction) return '';" in js
        assert "当前知识库缺少支撑该结论所需的充分证据" in js

    def test_home_first_screen_prioritizes_action_over_product_copy(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderLearnerOverview")
        end = js.index("function renderAdminOverview", start)
        home = js[start:end]
        assert "renderWorkspaceProduct(ct)" in home
        assert "会理解学习者的" not in home
        assert "智能教学系统" not in home
        assert "Math.random" not in home
        assert "setInterval" not in home

    def test_core_task_uses_real_teaching_modes_without_judge_facing_copy(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderQuery")
        end = js.index("R-04B learning workspace", start)
        query = js[start:end]
        assert "r08b3-task-launch" in query
        assert "r08b3-learning-mode" in query
        for mode in ("自适应", "基础拆解", "案例引导", "证据深入"):
            assert mode in query
        for action in ('value="still_confused"', 'value="request_example"', 'value="deepen"'):
            assert action in query
        assert "评委将看到什么" not in query
        assert "比赛主舞台" not in query
        assert "stepTimer = setInterval" not in query

    def test_five_canvas_result_keeps_answer_review_evidence_and_next_action(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderR04TaskResult")
        end = js.index("function renderQuery", start)
        result = js[start:end]
        assert result.index("当前解释") < result.index("Reviewer 结论")
        assert result.index("Reviewer 结论") < result.index("这个回答依据什么")
        assert result.index("这个回答依据什么") < result.index("协同记录")
        assert result.index("当前解释") < result.index("下一步学习")

    def test_long_resource_markup_is_structured_and_open_action_is_not_duplicated(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "function t1RenderLongText" in js
        assert "t1-long-read-markup" in js
        assert "action !== 'ask_follow_up' && action !== 'open'" in js
        assert "container.querySelectorAll('.t1-resource-card details')" in js
        assert "data-resource-tab" in js
        assert "canvas-resource-stage" in js
        assert "action: 'open'" in js
        assert "data-open-recorded" in js

    def test_task_canvas_restores_authoritative_result_and_public_knowledge_graph(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        query_start = js.index("function renderQuery")
        query = js[query_start:]
        assert "renderR04TaskResult({ data: lastTask }, lastQuestion" in query
        graph_start = js.index("function t1PublicTaskKnowledgeGraph")
        graph_end = js.index("function t234ConceptGraph", graph_start)
        graph = js[graph_start:graph_end]
        assert "Array.isArray(data.sources)" in graph
        assert "kp_names" in graph
        assert "KP 投影" in graph
        assert "来源关联" in graph
        assert "不冒充 R06 Canonical Concept Relation" in graph

    def test_r08b3_agent_roles_use_current_public_contribution_status(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderAgentChain")
        end = js.index("function renderR04KnowledgeWorkspace", start)
        collab = js[start:end]
        assert "r08b3-agent-role-map" in collab
        assert "<h3>四个 Agent</h3>" in collab
        assert "状态只依据当前公开 trace" not in collab
        assert "contributionGroups[agentId]" in collab
        assert "已产生公开贡献" in collab
        assert "本次无公开贡献" in collab

    def test_r08b3_evidence_chain_uses_only_public_task_facts(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderR04KnowledgeWorkspace")
        end = js.index("视图路由钩子", start)
        knowledge = js[start:end]
        assert "r08b3-evidence-chain" in knowledge
        for label in ("01 · 问题", "02 · 证据", "03 · 审核"):
            assert label in knowledge
        for private_field in ("KnowledgeLearningContext", "TeachingMemoryView", "_contract_candidate"):
            assert private_field not in knowledge

    def test_r08b3_visual_system_is_responsive_and_cache_busted(self, client):
        css = client.get("/static/assets/app.css").text
        html = client.get("/").text
        for selector in (".r08b3-home-stage", ".r08b3-task-launch", ".r08b3-agent-role-map", ".r08b3-evidence-chain", ".r08b3-growth-main"):
            assert selector in css
        for selector in (".canvas-overview", ".canvas-task-workspace", ".canvas-collaboration", ".canvas-knowledge", ".canvas-growth"):
            assert selector in css
        assert "@media(max-width:1100px)" in css
        assert "@media(max-width:760px)" in css
        assert "app.css?v=" in html
        assert "mf6-features.js?v=" in html

    def test_factual_five_canvas_layer_is_loaded_and_uses_public_runtime_sources(self, client):
        html = client.get("/").text
        canvas = client.get("/static/assets/product-canvases.js").text
        assert "product-canvases.js?v=" in html
        for renderer in (
            "renderOverview", "renderTaskResult", "renderCollaboration",
            "renderKnowledge", "renderGrowth",
        ):
            assert renderer in canvas
        for endpoint in (
            "/api/learning-workspace/", "/api/match-report/",
            "/api/learning-tasks/", "/l3/retrieve/keyword",
        ):
            assert endpoint in canvas
        for public_field in (
            "task_events", "agent_trace", "collab_lines", "flow_events",
            "evidence", "review", "answer", "learning_resources",
        ):
            assert public_field in canvas
        for private_field in (
            "_contract_candidate", "_FinalPrivateCandidateSet",
            "AnswerCorrelation", "TeachingMemoryView",
        ):
            assert private_field not in canvas

    def test_factual_canvas_uses_task_evidence_visual_not_static_science_preset(self, client):
        canvas = client.get("/static/assets/product-canvases.js").text
        assert "mode: 'concept_relation'" in canvas
        assert "mode: 'evidence_projection'" in canvas
        assert "retrieved_for" in canvas
        assert "evidenced_by" in canvas
        assert "不冒充科学因果关系" in canvas
        assert "function renderTaskEvidenceVisual" in canvas
        assert "Claim–Evidence–Review" in canvas
        assert "证据图示" in canvas
        assert "pcTaskEnergy" not in canvas
        assert "data-pc-task-tab=\"visual\">科学图示" not in canvas
        assert "source_type || '').toLowerCase() !== 'template'" in canvas
        assert "function scientificTypography" in canvas
        assert "function normalizeScientificText" in canvas
        assert "Dy³⁺" in canvas and "cm⁻¹" in canvas

    def test_five_canvas_actions_reenter_existing_product_loop(self, client):
        canvas = client.get("/static/assets/product-canvases.js").text
        features = client.get("/static/assets/mf6-features.js").text
        assert "internals().renderResources(resourceData)" in canvas
        assert "internals().bindResourceActions(container, resourceData, question, queryInput)" in canvas
        assert "queryInput.value = value" in canvas
        assert "var ask = d.getElementById('queryAsk')" in canvas
        assert "ask.click()" in canvas
        assert "dy3_pending_query" in canvas
        assert "/api/learning/resources/interact" in features

    def test_growth_canvas_does_not_invent_personal_path_for_unknown_learner(self, client):
        canvas = client.get("/static/assets/product-canvases.js").text
        assert "hasLearnerEvidence" in canvas
        assert "尚无真实作答，不生成个人学习路径" in canvas
        assert "等待真实诊断" in canvas
        assert "本地已编写题库" in canvas
        assert "本任务已发布事实" in canvas
        assert "本任务审核结果派生" in canvas
        assert "function decisionLabel" in canvas
        assert "function sourceClassLabel" in canvas
        assert "<h2>个性化资源</h2>" not in canvas

    def test_five_canvas_visual_system_has_real_diagrams_and_no_duplicate_launcher(self, client):
        css = client.get("/static/assets/app.css").text
        for selector in (
            ".pc-overview-grid", ".pc-task-grid", ".pc-runtime-main",
            ".pc-concept-graph", ".pc-growth-path", ".pc-difficulty-chart",
        ):
            assert selector in css
        assert ".canvas-query-launch.has-product-task>.r08b3-task-launch" in css
        assert ".canvas-query-launch.has-product-task>.r08-task-history" in css

    def test_product_canvas_contains_wide_runtime_content_inside_its_panel(self, client):
        css = client.get("/static/assets/app.css").text
        assert ".pc-flow-panel,.pc-event-table{width:100%;overflow:hidden}" in css
        assert ".pc-runtime-flow{display:block;width:100%" in css
        assert ".pc-collab-grid>*" in css
        assert "min-width:0;max-width:100%" in css

    def test_collaboration_canvas_translates_runtime_facts_and_hides_empty_challenge(self, client):
        canvas = client.get("/static/assets/product-canvases.js").text
        assert "function eventLabel" in canvas
        assert "TASK_DECOMPOSED: '任务拆解'" in canvas
        assert "CONTRIBUTION_PRODUCED: '专业贡献'" in canvas
        assert "challenges.length ?" in canvas
        assert "本次公开轨迹没有Challenge或修订回流" not in canvas

    def test_learning_resource_is_teaching_first_and_evidence_is_secondary(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        css = client.get("/static/assets/app.css").text
        assert "guided.lesson_sequence" in js
        assert "t1-lesson-sequence" in js
        assert "t1-lesson-appendix" in js
        assert "检查理解" in js
        assert ".t1-lesson-sequence" in css
        assert ".t1-lesson-appendix" in css

    def test_t1_release_gate_controls_answer_and_clarification_views(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderR04TaskResult")
        end = js.index("function renderQuery", start)
        renderer = js[start:end]
        assert "quality.status === 'FULL_RELEASE'" in renderer
        assert "quality.status === 'LIMITED_RELEASE'" in renderer
        assert "var answer = releaseAllowed ? String(data.answer || '') : '';" in renderer
        clarify = renderer.index("if (clarify && !answer)")
        assert renderer.index("t1QualityBlock(data)", clarify) > clarify

    def test_t1_resources_are_truth_labeled_and_feedback_is_real(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "/api/learning/resources/interact" in js
        for value in ("retrieved", "generated", "derived", "template"):
            assert value in js
        assert "resource.completion_signal" in js
        assert "来源与完成口径" in js
        for private_field in ("_FinalPrivateCandidateSet", "_contract_candidate", "answer_identity"):
            assert private_field not in js

    def test_personalized_resources_and_public_knowledge_graph_are_actionable(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "t1-guided-document" in js
        assert "guided_long_read" in js or "guided_document" in js
        assert "t1-practical-steps" in js
        assert "t1-assessment-stages" in js
        assert "ask_follow_up" in js
        assert "r06-concept-svg" in js
        assert "context.edges" in js
        assert "actual_characters" in js
        assert "llm_evidence_synthesis" in js

    def test_growth_reuses_reviewer_released_task_resources(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderR04LearningWorkspace")
        end = js.index("Agent 列表视图", start)
        growth = js[start:end]
        assert "taskData.learning_resources" in growth
        assert "t1ResourceCards(taskData)" in growth
        assert "t1BindResourceActions(resourceHost, taskData" in growth
        assert "通用模板冒充任务资源" in growth
        assert "renderPersonalizedResources(g('growthResources'))" not in growth

    def test_query_completion_refreshes_learning_task_history_from_server(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function r08RefreshTaskHistory")
        refresh = js[start:start + 1400]
        assert "'/api/learning-tasks/' + encodeURIComponent(lid)" in refresh
        assert "r08TaskTruth.tasks" in refresh
        result_start = js.index("function renderR04TaskResult")
        result_block = js[result_start:result_start + 10000]
        assert "r08RefreshTaskHistory(g('content'))" in result_block

    def test_resource_follow_up_can_continue_from_task_or_growth_space(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function t1BindResourceActions")
        end = js.index("function renderR04TaskResult", start)
        binding = js[start:end]
        assert "function continueTask" in binding
        assert "if (queryInput)" in binding
        assert "dy3_pending_query" in binding
        assert "dy3_pending_teaching_action" in binding
        assert "window.sv('query')" in binding

    def test_public_sources_show_real_uri_and_review_status(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function r04SourceReference")
        end = js.index("function r04RenderTrace", start)
        source = js[start:end]
        for field in ("source_title", "source_uri", "source_type", "evidence_status"):
            assert field in source
        assert "noopener noreferrer" in source
        assert "javascript:" not in source

    def test_learning_workspace_supports_optional_profile_diagnostic_and_clear(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function renderWorkspaceProduct")
        end = js.index("function renderLearnerOverview", start)
        workspace = js[start:end]
        assert "补充或管理自愿学习背景" in workspace
        assert "DIAGNOSTIC" in workspace
        assert "DELETE" in workspace
        assert "/api/user-understanding/profile?learner_id=" in workspace
        assert "不会创建默认背景" in workspace

    def test_t1_task_fact_flow_never_counts_non_agent_trace_actors(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        start = js.index("function r08TaskFactFlow")
        end = js.index("function r04AgentName", start)
        flow = js[start:end]
        for agent_id in (
            "agent.learning.diagnosis", "agent.knowledge.generation",
            "agent.quality.review", "agent.guidance.decision",
        ):
            assert agent_id in flow
        assert "fixedAgents[event.agent_id]" in flow
        assert " / 4 个 Agent 有公开贡献" in flow


class TestAuthView:
    """全屏登录/注册双页面 (v3) 结构与认证链路."""

    def test_two_auth_views_present(self, client):
        html = client.get("/").text
        assert "loginView" in html          # 登录独立页面
        assert "registerView" in html       # 注册独立页面
        assert "authSpectrum" in html       # 光谱动画背景
        assert "goRegisterLink" in html     # 登录页 → 注册页链接
        assert "goLoginLink" in html        # 注册页 → 登录页链接

    def test_no_brand_text_and_tabs(self, client):
        """品牌区文字与 Tab 切换已移除."""
        html = client.get("/").text
        assert "auth-brand" not in html
        assert "auth-tabs" not in html
        assert "Polaris 智能教学系统" not in html or "auth-brand-title" not in html

    def test_auth_forms(self, client):
        html = client.get("/").text
        assert 'id="loginForm"' in html
        assert 'id="registerForm"' in html
        assert "auth-eye" in html           # 密码可见性切换
        assert "loginStudentId" in html
        assert "regPassword2" in html       # 确认密码

    def test_no_legacy_auth_modal(self, client):
        """旧的弹窗/下拉/Tab 登录结构已移除."""
        html = client.get("/").text
        assert "loginModal" not in html
        assert "loginDropdown" not in html
        assert "panelLogin" not in html
        assert "tabLogin" not in html

    def test_versioned_assets(self, client):
        """静态资源带版本号, 规避浏览器缓存."""
        html = client.get("/").text
        assert "app.js?v=" in html
        assert "app.css?v=" in html

    def test_register_then_login_flow(self, client):
        """注册 → 登录 全链路."""
        import uuid
        sid = "DY9" + str(uuid.uuid4().int)[:7]
        r = client.post("/l1/api/v1/auth/register",
                        json={"student_id": sid, "password": "flow1234"})
        assert r.json()["code"] == 0, r.text
        r = client.post("/l1/api/v1/auth/login",
                        json={"student_id": sid, "password": "flow1234"})
        assert r.json()["code"] == 0, r.text
        assert r.json()["data"]["role"] in ("undergrad", "student")

    def test_admin_seed_accounts(self, client):
        """种子账号: 学生/教师/管理员."""
        for sid, pwd in [("DY20240001", "demo123"),
                         ("DY20240002", "demo123"),
                         ("DY20248888", "admin888")]:
            r = client.post("/l1/api/v1/auth/login",
                            json={"student_id": sid, "password": pwd})
            assert r.json()["code"] == 0, f"{sid} login failed: {r.text}"

    def test_admin_creates_sub_admin_only(self, client):
        """仅管理员可创建次级管理员; 学生/教师被拒."""
        c = client
        r = c.post("/l1/api/v1/auth/login",
                   json={"student_id": "DY20240001", "password": "demo123"})
        stu_token = r.json()["data"]["access_token"]
        r = c.post("/l1/api/v1/admin/create-user",
                   json={"student_id": "DY20249001", "password": "x12345", "role": "admin"},
                   headers={"Authorization": "Bearer " + stu_token})
        assert r.json()["code"] == -32203  # FORBIDDEN

        r = c.post("/l1/api/v1/auth/login",
                   json={"student_id": "DY20248888", "password": "admin888"})
        adm_token = r.json()["data"]["access_token"]
        r = c.post("/l1/api/v1/admin/create-user",
                   json={"student_id": "DY20249002", "password": "x12345", "role": "admin"},
                   headers={"Authorization": "Bearer " + adm_token})
        assert r.json()["code"] == 0, r.text
        assert r.json()["data"]["role"] == "admin"


class TestApiQuery:
    """端到端多 Agent 查询聚合端点."""

    def test_query_returns_pipeline(self, client):
        resp = client.post("/api/query", json={
            "query": "Dy3+ 的量子效率受哪些因素影响？",
            "learner_id": "demo-learner",
        })
        assert resp.status_code == 200, resp.text[:300]
        data = resp.json()["data"]
        assert "answer" in data
        assert len(data["pipeline"]) >= 4  # 主线 4 步 (诊断/生成/审核/决策)
        steps = [p["step"] for p in data["pipeline"]]
        # R-03G 后 pipeline/collab_lines 是真实 trace projection，不再伪装固定 L1 流水线。
        assert "CONTRIBUTION_PRODUCED" in steps
        joined = " ".join(steps) + " " + " ".join(str(p.get("detail", "")) for p in data["pipeline"])
        assert "learning.diagnosis" in joined or "guidance.decision" in joined
        # 协作线结构: 每条线均来自单个真实 runtime fact。
        clines = data.get("collab_lines") or []
        assert clines, "collab_lines 缺失 (多线协作)"
        assert all(c.get("kind") == "runtime_fact" for c in clines)
        assert any(c.get("label") == "GUIDANCE_DECIDED" for c in clines)
        assert all(st.get("elapsed_ms") is None for c in clines for st in c.get("steps", []))
        assert all(st.get("agent") and st.get("output") for c in clines for st in c.get("steps", []))

    def test_query_creates_session(self, client):
        resp = client.post("/api/query", json={
            "query": "什么是浓度猝灭？",
            "learner_id": "demo-learner",
        })
        data = resp.json()["data"]
        assert data["session"] is not None
        assert data["session"].get("session_id")

    def test_query_missing_query_400(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 400
        assert resp.json()["code"] == -32700

    def test_query_invalid_json_400(self, client):
        resp = client.post("/api/query", content="not-json{{",
                           headers={"content-type": "application/json"})
        assert resp.status_code == 400

    def test_query_empty_query_400(self, client):
        resp = client.post("/api/query", json={"query": "   "})
        assert resp.status_code == 400


class TestLearnerEndpoints:
    """前端依赖的真实学情/知识端点连通."""

    def test_l3_stats(self, client):
        resp = client.get("/l3/stats")
        assert resp.status_code == 200
        assert "entity_count" in resp.json()["data"]

    def test_l2_profile(self, client):
        """画像端点 (可能未初始化, 但不应 500)."""
        resp = client.get("/l2/profile/demo-learner")
        assert resp.status_code in (200, 404, 503)

    def test_l1_login_contract(self, client):
        """登录响应契约 (access_token/refresh_token/role)."""
        resp = client.post("/l1/api/v1/auth/login",
                           json={"student_id": "20240001", "password": "demo123"})
        json_body = resp.json()
        if "data" in json_body and json_body.get("code") == 0:
            data = json_body["data"]
            assert data["access_token"]
            assert data["refresh_token"]
            assert data["role"] in ("student", "teacher", "admin")
        else:
            # 演示环境用户未预置: 必须返回明确业务错误码
            assert "code" in json_body
            assert "message" in json_body


class TestRoutesSummary:
    """路由发现."""

    def test_routes_include_l7(self, client):
        builder = UnifiedApp.create_full_app_builder()
        routes = builder.get_routes_summary()
        layers = {r["layer"] for r in routes}
        assert "L7" in layers
        assert "Unified" in layers
        # /api/query 在 Unified 层
        unified_paths = [r["path"] for r in routes if r["layer"] == "Unified"]
        assert "/api/query" in unified_paths
