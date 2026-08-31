/* Dy3+ Polaris — M-F9 学情-资源匹配度报告 (竞赛核心可视化)
 * 对标竞赛评分标准:
 *   - 知识盲区定位 (薄弱点 + 未覆盖)
 *   - 资源难度匹配曲线 (IRT 能力 θ vs ZPD 三区)
 *   - 学习路径规划图 (前置依赖 DAG + 推荐顺序)
 * 纯原生 SVG 绘制, 离线可用, 不依赖 CDN。数据源: GET /api/match-report/{learner_id}
 */
(function () {
  'use strict';
  var d = document;
  function g(id) { return d.getElementById(id); }
  function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  var VIEW = 'match-report';

  /* 学习者 ID (与 mf6 一致) */
  function learnerId() {
    return window.dy3LearnerId ? window.dy3LearnerId() : (localStorage.getItem('dl') || 'guest-unavailable');
  }

  /* API 请求 (统一封装, 兼容 window.api) */
  function apiReq(m, p, b) {
    if (window.api && window.api.rq) {
      return window.api.rq(m, p, b);
    }
    var h = { 'Content-Type': 'application/json' };
    var tk = localStorage.getItem('dt');
    if (tk) h.Authorization = 'Bearer ' + tk;
    return fetch(p, { method: m, headers: h, body: b ? JSON.stringify(b) : void 0 })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok || (j && j.code !== 0)) { throw new Error((j && j.message) || ('HTTP ' + r.status)); }
        return j ? j.data : null;
      }); });
  }

  /* ================= 颜色 ================= */
  function masteryColor(v) {
    if (v >= 0.8) return 'var(--success)';
    if (v >= 0.5) return 'var(--violet)';
    if (v > 0) return 'var(--warn)';
    return 'var(--danger)'; // 未覆盖
  }
  function domainColor(code) {
    return { A: '#2a5adf', B: '#f59e0b', C: '#10b981', D: '#8b5cf6' }[code] || '#6b7280';
  }
  function isDark() { return (d.documentElement.getAttribute('data-theme') || 'light') === 'dark'; }

  /* ================= 主渲染 ================= */
  function render() {
    var ct = g('content');
    if (!ct) return;
    ct.setAttribute('data-mf9', VIEW);
    ct.innerHTML = '<div class="r08-context-return"><button class="btn ghost" id="mf9BackGrowth">← 返回成长决策</button><span>学习分析报告属于成长决策空间</span></div><div class="card" style="margin-bottom:14px">' +
      '<h3 style="margin:0 0 4px">📊 学情 - 资源匹配度报告</h3>' +
      '<p style="color:var(--muted);font-size:12.5px;margin:0">知识盲区定位 · 资源难度匹配 · 学习路径规划（多智能体协同决策可视化）</p>' +
      '</div>' +
      '<div id="mf9Body"><div class="loading"><span class="spinner"></span>正在生成匹配度报告…</div></div>';
    var back = g('mf9BackGrowth');
    if (back) back.addEventListener('click', function () { if (window.sv) window.sv('learn-weak'); });

    apiReq('GET', '/api/match-report/' + encodeURIComponent(learnerId())).then(function (data) {
      if (!data) { ct.innerHTML = '<div class="error-banner">报告数据为空 (需登录并完成初测)</div>'; return; }
      renderReport(data);
    }).catch(function (e) {
      g('mf9Body').innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  /* ================= 报告主体 ================= */
  function renderReport(data) {
    var body = g('mf9Body');
    if (!body) return;
    if (data.report && typeof data.report === 'object' && Object.keys(data.report).length) {
      renderAuthoritativeReport(data.report);
      return;
    }
    var dark = isDark();

    // 顶部总览条带
    var thetaPct = Math.round((data.theta_norm || 0.5) * 100);
    var masteryPct = Math.round((data.overall_mastery || 0) * 100);
    var blindCount = (data.blind_spots || []).length;
    var pathReady = data.path_ready_count || 0;
    var pathTotal = (data.learning_path || []).length;

    var header =
      '<div class="stat-strip" style="margin-bottom:16px">' +
      '<div class="stat-card"><div class="num">' + esc(data.level || '-') + '</div><div class="lbl">能力等级</div></div>' +
      '<div class="stat-card"><div class="num">θ=' + (data.theta != null ? Number(data.theta).toFixed(2) : '-') + '</div><div class="lbl">IRT 能力值</div></div>' +
      '<div class="stat-card"><div class="num" style="color:var(--accent-ink)">' + masteryPct + '%</div><div class="lbl">总体掌握度</div></div>' +
      '<div class="stat-card"><div class="num" style="color:var(--danger)">' + blindCount + '</div><div class="lbl">知识盲区</div></div>' +
      '<div class="stat-card"><div class="num" style="color:var(--success)">' + pathReady + '/' + pathTotal + '</div><div class="lbl">可学路径</div></div>' +
      '</div>';

    body.innerHTML = header +
      '<div class="grid cols-2" style="align-items:start">' +
      '<div class="card">' + renderBlindSpots(data.blind_spots) + '</div>' +
      '<div class="card">' + renderDifficultyMatch(data.difficulty_match, dark) + '</div>' +
      '</div>' +
      '<div class="card" style="margin-top:16px">' + renderLearningPath(data.learning_path) + '</div>' +
      '<div class="card" style="margin-top:16px">' + renderDomainCards(data.domains) + '</div>';
  }

  function findingLabel(type) {
    return {
      VERIFIED_WEAKNESS: '真实作答支持的薄弱点',
      PREREQUISITE_GAP: '前置概念缺口',
      MISCONCEPTION: '待处理错误认知',
      UNKNOWN: '尚无证据判断'
    }[type] || type;
  }

  function renderAuthoritativeDifficulty(match) {
    match = match || {};
    var bands = Array.isArray(match.bands) ? match.bands : [];
    if (!bands.length) return '<h3>资源难度匹配</h3><p style="color:var(--muted)">真实题库难度分布暂不可用。</p>';
    var maxCount = Math.max.apply(null, bands.map(function (item) { return Number(item.question_count || 0); }).concat([1]));
    var bars = bands.map(function (item) {
      var width = Math.round(Number(item.question_count || 0) / maxCount * 100);
      return '<div class="t1-difficulty-band"><span>' + esc(item.label || item.difficulty) + '</span><div><i style="width:' + width + '%"></i></div><strong>' + Number(item.question_count || 0) + ' 题</strong></div>';
    }).join('');
    var position = match.learner_position;
    var marker = position == null
      ? '<p class="hint">学习者位置：UNKNOWN。尚无真实 IRT 作答，不绘制能力点。</p>'
      : '<div class="t1-learner-position"><span style="left:' + Math.round(Number(position) * 100) + '%"></span></div><p class="hint">学习者位置来自 IRT 模型推断；阴影范围为当前 ZPD 投影。</p>';
    return '<h3>真实题库难度匹配</h3><p class="hint">共 ' + Number(match.authored_question_count || 0) + ' 道已编写题目；曲线不使用随机资源数量。</p>' + bars + marker + '<div class="callout"><strong>' + esc(match.decision || 'DIAGNOSE_FIRST') + '</strong><p>' + esc(match.reason || '') + '</p></div>';
  }

  function renderAuthoritativePath(path) {
    path = path || {};
    var nodes = Array.isArray(path.nodes) ? path.nodes.slice(0, 12) : [];
    var edges = Array.isArray(path.edges) ? path.edges : [];
    if (!nodes.length) return '<h3>Concept 学习路径</h3><p style="color:var(--muted)">当前没有可公开的 R06 Concept 节点。</p>';
    var lookup = {};
    nodes.forEach(function (node, index) { lookup[String(node.concept_id || '')] = { node: node, x: 80 + (index % 3) * 170, y: 50 + Math.floor(index / 3) * 92 }; });
    var height = Math.max(145, 95 + Math.floor((nodes.length - 1) / 3) * 92);
    var lines = edges.filter(function (edge) { return lookup[String(edge.source || '')] && lookup[String(edge.target || '')]; }).map(function (edge) {
      var a = lookup[String(edge.source)], b = lookup[String(edge.target)];
      return '<line x1="' + a.x + '" y1="' + a.y + '" x2="' + b.x + '" y2="' + b.y + '" marker-end="url(#matchArrow)"></line>';
    }).join('');
    var boxes = nodes.map(function (node) {
      var point = lookup[String(node.concept_id || '')], name = String(node.name || node.concept_id || 'Concept');
      if (name.length > 10) name = name.slice(0, 10) + '…';
      return '<g class="state-' + esc(String(node.learner_state || 'UNKNOWN').toLowerCase()) + '"><rect x="' + (point.x - 62) + '" y="' + (point.y - 23) + '" width="124" height="46" rx="10"></rect><text x="' + point.x + '" y="' + (point.y + 4) + '">' + esc(name) + '</text></g>';
    }).join('');
    return '<h3>Concept 学习路径</h3><p class="hint">箭头只来自 R06 的公开策展关系；UNKNOWN 不视为薄弱。</p><svg class="t1-path-svg" viewBox="0 0 500 ' + height + '" role="img" aria-label="真实 Concept 学习路径"><defs><marker id="matchArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z"></path></marker></defs>' + lines + boxes + '</svg>';
  }

  function renderAuthoritativeReport(report) {
    var body = g('mf9Body');
    if (!body) return;
    var evidence = report.evidence_sufficiency || {};
    var ability = report.ability || {};
    var findings = Array.isArray(report.findings) ? report.findings : [];
    var decision = report.difficulty_decision || {};
    var path = report.learning_path || {};
    var resourceMatch = report.resource_difficulty_match || {};
    var nodes = Array.isArray(path.nodes) ? path.nodes : [];
    var timeline = Array.isArray(report.growth_timeline) ? report.growth_timeline : [];
    var next = report.next_action || {};
    var counts = { VERIFIED_WEAKNESS: 0, PREREQUISITE_GAP: 0, MISCONCEPTION: 0, UNKNOWN: 0 };
    findings.forEach(function (item) { counts[item.type] = (counts[item.type] || 0) + 1; });

    var findingHtml = findings.length ? findings.map(function (item) {
      var tone = item.type === 'VERIFIED_WEAKNESS' || item.type === 'MISCONCEPTION' ? 'var(--warn)' : (item.type === 'UNKNOWN' ? 'var(--muted)' : 'var(--violet)');
      return '<div style="padding:10px 0;border-bottom:1px solid var(--rule)"><div style="display:flex;justify-content:space-between;gap:12px"><strong>' + esc(item.reference || '未标识') + '</strong><span class="badge" style="color:' + tone + '">' + esc(findingLabel(item.type)) + '</span></div><p style="margin:5px 0 0;color:var(--muted);font-size:12px">' + esc(item.reason || '') + ' · 来源 ' + esc(item.source_class || 'UNKNOWN') + '</p></div>';
    }).join('') : '<p style="color:var(--muted)">当前没有足够事实形成诊断结论。</p>';

    var pathHtml = nodes.length ? nodes.slice(0, 16).map(function (node, index) {
      var state = String(node.learner_state || 'UNKNOWN');
      var label = { MASTERED: '已有掌握证据', LEARNING_GAP: '待学习', UNKNOWN: '尚无证据' }[state] || state;
      return '<div style="display:flex;gap:10px;align-items:center;padding:9px;border:1px solid var(--rule);border-radius:8px"><span style="width:24px;height:24px;border-radius:50%;display:grid;place-items:center;background:var(--surface2);font-size:11px">' + (index + 1) + '</span><div><strong>' + esc(node.name || node.concept_id || 'Concept') + '</strong><small style="display:block;color:var(--muted)">' + esc(label) + '</small></div></div>';
    }).join('') : '<p style="color:var(--muted)">当前任务尚未形成可公开的 R06 Concept 路径。</p>';

    var timelineHtml = timeline.length ? timeline.slice(-12).reverse().map(function (item) {
      var when = item.timestamp ? new Date(item.timestamp * 1000).toLocaleString() : '时间未记录';
      return '<li><strong>' + esc(item.event_type) + '</strong> · ' + esc(item.outcome) + '<small style="display:block;color:var(--muted)">' + esc(item.reference) + ' · ' + esc(when) + '</small></li>';
    }).join('') : '<li>暂无真实学习事件或作答记录。</li>';

    body.innerHTML =
      '<div class="stat-strip" style="margin-bottom:16px">' +
      '<div class="stat-card"><div class="num">' + esc(report.status || 'UNKNOWN') + '</div><div class="lbl">报告状态</div></div>' +
      '<div class="stat-card"><div class="num">' + Number(evidence.answer_record_count || 0) + '</div><div class="lbl">真实作答记录</div></div>' +
      '<div class="stat-card"><div class="num">' + (ability.status === 'MODEL_INFERRED' && ability.theta != null ? 'θ=' + Number(ability.theta).toFixed(2) : '未知') + '</div><div class="lbl">IRT能力（有证据才显示）</div></div>' +
      '<div class="stat-card"><div class="num">' + counts.VERIFIED_WEAKNESS + '</div><div class="lbl">已验证薄弱点</div></div>' +
      '<div class="stat-card"><div class="num">' + counts.UNKNOWN + '</div><div class="lbl">未知项（非薄弱）</div></div></div>' +
      '<div class="callout" style="margin-bottom:16px"><strong>难度决策：' + esc(decision.decision || 'DIAGNOSE_FIRST') + '</strong><p style="margin:5px 0 0">' + esc(decision.reason || '证据不足，暂不作能力推定。') + '</p></div>' +
      '<div class="grid cols-2" style="align-items:start;margin-bottom:16px"><div class="card">' + renderAuthoritativeDifficulty(resourceMatch) + '</div><div class="card">' + renderAuthoritativePath(path) + '</div></div>' +
      '<div class="grid cols-2" style="align-items:start"><div class="card"><h3>学习状态发现</h3>' + findingHtml + '</div><div class="card"><h3>下一步行动</h3><p><strong>' + esc(next.type || 'PRACTICE') + '</strong> · ' + esc(next.target || 'unknown') + '</p><p style="color:var(--muted)">' + esc(next.reason || '') + '</p><p class="hint">来源：' + esc(next.source_class || 'DECISION') + '</p></div></div>' +
      '<div class="card" style="margin-top:16px"><h3>R06 Concept 学习路径</h3><p class="hint">仅由公开 Concept Relation 与真实学习状态派生；UNKNOWN 不视为未掌握。</p><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px">' + pathHtml + '</div></div>' +
      '<div class="card" style="margin-top:16px"><h3>真实成长时间线</h3><ol>' + timelineHtml + '</ol></div>';
  }

  /* ================= 1. 知识盲区定位 ================= */
  function renderBlindSpots(spots) {
    if (!spots || !spots.length) {
      return '<h3 style="margin:0 0 10px">📍 知识盲区定位</h3><p style="color:var(--muted)">暂无盲区，掌握度良好 🎉</p>';
    }
    var items = spots.slice(0, 12).map(function (s, i) {
      var c = s.type === '未覆盖' ? 'var(--danger)' : 'var(--warn)';
      var barW = Math.max(4, Math.round((1 - s.mastery) * 100));
      return '<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--rule)">' +
        '<span style="font-size:11px;color:var(--muted);flex:none;width:18px">' + (i + 1) + '</span>' +
        '<span class="badge" style="flex:none;background:' + domainColor(s.domain) + ';color:#fff">' + esc(s.domain) + '</span>' +
        '<div style="flex:1;min-width:0">' +
        '<div style="display:flex;justify-content:space-between;font-size:12.5px"><span style="font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(s.name) + '</span><span style="color:' + c + ';font-weight:600;flex:none">' + Math.round(s.mastery * 100) + '%</span></div>' +
        '<div class="mastery-bar" style="height:6px;margin-top:3px"><div class="mastery-fill" style="width:' + barW + '%;background:' + c + '"></div></div>' +
        '</div>' +
        '<span style="font-size:10.5px;color:var(--muted);flex:none">' + esc(s.type) + '</span>' +
        '</div>';
    }).join('');
    return '<h3 style="margin:0 0 10px">📍 知识盲区定位 <span class="hint">(' + spots.length + ' 个待加强)</span></h3>' +
      '<div>' + items + '</div>' +
      (spots.length > 12 ? '<p style="font-size:11px;color:var(--muted);margin-top:6px">… 另有 ' + (spots.length - 12) + ' 个盲区</p>' : '');
  }

  /* ================= 2. 资源难度匹配曲线 ================= */
  function renderDifficultyMatch(dm, dark) {
    var W = 560, H = 200, padL = 40, padR = 20, padT = 20, padB = 40;
    var axisW = W - padL - padR, axisH = H - padT - padB;
    var theta = dm.theta || 0.5, lo = dm.zpd_lower || 0.35, hi = dm.zpd_upper || 0.65;
    var zpd = dm.zpd || { independent: 0, zpd: 0, frustration: 0 };

    function x(v) { return padL + v * axisW; }
    function y() { return padT + axisH / 2; }

    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;height:auto;display:block;background:var(--surface);border:1px solid var(--rule);border-radius:10px">';
    // 难度轴 0..1
    for (var t = 0; t <= 1.0001; t += 0.25) {
      svg += '<line x1="' + x(t) + '" y1="' + padT + '" x2="' + x(t) + '" y2="' + (padT + axisH) + '" stroke="var(--rule)" stroke-width="1"/>';
      svg += '<text x="' + x(t) + '" y="' + (H - padB + 18) + '" text-anchor="middle" font-size="10" fill="var(--muted)">' + (t * 100).toFixed(0) + '</text>';
    }
    // ZPD 三区背景
    var zoneColors = { independent: 'var(--success)', zpd: 'var(--violet)', frustration: 'var(--danger)' };
    // 独立区 (0 ~ lo)
    svg += '<rect x="' + x(0) + '" y="' + (padT + 20) + '" width="' + (x(lo) - x(0)) + '" height="' + (axisH - 40) + '" style="fill:var(--success)" opacity="0.08"/>';
    // ZPD 区 (lo ~ hi)
    svg += '<rect x="' + x(lo) + '" y="' + (padT + 20) + '" width="' + (x(hi) - x(lo)) + '" height="' + (axisH - 40) + '" style="fill:var(--violet)" opacity="0.12"/>';
    // 挫败区 (hi ~ 1)
    svg += '<rect x="' + x(hi) + '" y="' + (padT + 20) + '" width="' + (x(1) - x(hi)) + '" height="' + (axisH - 40) + '" style="fill:var(--danger)" opacity="0.08"/>';
    // 区域标注
    svg += '<text x="' + ((x(0) + x(lo)) / 2) + '" y="' + (padT + 14) + '" text-anchor="middle" font-size="9.5" style="fill:var(--success)">独立区</text>';
    svg += '<text x="' + ((x(lo) + x(hi)) / 2) + '" y="' + (padT + 14) + '" text-anchor="middle" font-size="9.5" style="fill:var(--violet)">ZPD 最佳区</text>';
    svg += '<text x="' + ((x(hi) + x(1)) / 2) + '" y="' + (padT + 14) + '" text-anchor="middle" font-size="9.5" style="fill:var(--danger)">挫败区</text>';
    // θ 能力标线 (三角标记)
    svg += '<path d="M' + x(theta) + ' ' + (y() - 14) + ' L' + (x(theta) - 8) + ' ' + (y() - 26) + ' L' + (x(theta) + 8) + ' ' + (y() - 26) + ' Z" fill="var(--accent-ink)"/>';
    svg += '<line x1="' + x(theta) + '" y1="' + (y() - 26) + '" x2="' + x(theta) + '" y2="' + (padT + axisH) + '" stroke="var(--accent-ink)" stroke-width="2" stroke-dasharray="4 3"/>';
    svg += '<text x="' + x(theta) + '" y="' + (y() - 32) + '" text-anchor="middle" font-size="11" font-weight="700" fill="var(--accent-ink)">能力 θ=' + (theta * 100).toFixed(0) + '%</text>';
    svg += '</svg>';

    var total = zpd.independent + zpd.zpd + zpd.frustration || 1;
    var legend =
      '<div style="display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;font-size:12px">' +
      '<span style="color:var(--success)">■ 独立区 ' + zpd.independent + ' 点 (' + Math.round(zpd.independent / total * 100) + '%)</span>' +
      '<span style="color:var(--violet)">■ ZPD 最佳区 ' + zpd.zpd + ' 点 (' + Math.round(zpd.zpd / total * 100) + '%)</span>' +
      '<span style="color:var(--danger)">■ 挫败区 ' + zpd.frustration + ' 点 (' + Math.round(zpd.frustration / total * 100) + '%)</span>' +
      '</div>';

    return '<h3 style="margin:0 0 10px">📈 资源难度匹配曲线 <span class="hint">(ZPD 三区)</span></h3>' +
      '<p style="font-size:12px;color:var(--muted);margin:0 0 10px">当前能力 θ 落在 <strong style="color:var(--accent-ink)">' +
      (theta <= lo ? '独立区' : (theta >= hi ? '挫败区' : 'ZPD 最佳区')) +
      '</strong>，推荐优先分配 ZPD 最佳区资源（略高于当前能力，可经支架达成）。</p>' +
      svg + legend;
  }

  /* ================= 3. 学习路径规划图 ================= */
  function renderLearningPath(path) {
    if (!path || !path.length) {
      return '<h3 style="margin:0 0 10px">🎯 学习路径规划</h3><p style="color:var(--muted)">全部知识点已掌握 🎉</p>';
    }
    // 取前 16 条, 按 ready 分组展示为横向流程图
    var items = path.slice(0, 16).map(function (p, i) {
      var c = p.ready ? '#16a34a' : '#6b7280';
      var label = p.ready ? '可学' : '待前置';
      return '<div style="display:flex;align-items:center;gap:8px;padding:7px 10px;border:1px solid var(--rule);border-radius:8px;background:' + (p.ready ? 'var(--surface)' : 'var(--surface2)') + ';opacity:' + (p.ready ? '1' : '0.65') + '">' +
        '<span style="flex:none;width:22px;height:22px;border-radius:50%;background:' + domainColor(p.domain) + ';color:#fff;font-size:10px;display:flex;align-items:center;justify-content:center;font-weight:700">' + (i + 1) + '</span>' +
        '<div style="flex:1;min-width:0">' +
        '<div style="font-size:12.5px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(p.kp_id) + ' ' + esc(p.name) + '</div>' +
        '<div style="font-size:10.5px;color:var(--muted)">' + esc(p.level) + ' · 掌握 ' + Math.round(p.mastery * 100) + '%' + (p.prerequisites && p.prerequisites.length ? ' · 前置 ' + p.prerequisites.join(',') : '') + '</div>' +
        '</div>' +
        '<span class="badge" style="flex:none;background:' + c + ';color:#fff">' + label + '</span>' +
        '</div>';
    }).join('');

    return '<h3 style="margin:0 0 10px">🎯 学习路径规划 <span class="hint">(前置依赖排序 · 绿色可学)</span></h3>' +
      '<p style="font-size:12px;color:var(--muted);margin:0 0 12px">共 ' + path.length + ' 个待学知识点，其中 <strong style="color:var(--success)">' + (path.filter(function (p) { return p.ready; }).length) + ' 个</strong> 前置条件已满足，建议按序学习。</p>' +
      '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px">' + items + '</div>';
  }

  /* ================= 4. 域级卡片 ================= */
  function renderDomainCards(domains) {
    var cards = (domains || []).map(function (d) {
      return '<div class="domain-card" style="border-left-color:' + domainColor(d.code) + '">' +
        '<span class="domain-label">' + esc(d.label) + '</span>' +
        '<span class="domain-avg" style="color:' + domainColor(d.code) + '">' + Math.round(d.avg_mastery * 100) + '%</span>' +
        '<span class="domain-counts">已掌握 ' + d.mastered + ' · 学习中 ' + d.learning + ' · 待加强 ' + d.weak + (d.uncovered ? ' · 未覆盖 ' + d.uncovered : '') + '</span>' +
        '</div>';
    }).join('');
    return '<h3 style="margin:0 0 10px">🗂️ 四域掌握度概览</h3><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px">' + cards + '</div>';
  }

  /* R08B-1: the report stays routable and is opened from Growth Decision.
   * It no longer injects a separate top-level learner navigation entry.
   */
  d.addEventListener('view-rendered', function (e) {
    if (e.detail && e.detail.view === VIEW) { render(); }
  });
})();
