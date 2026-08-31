/* Dy3+ Polaris — M-F10 知识图谱可视化 v2 (分层径向布局)
 * 数据源: GET /l3/entities (L1-L4 层级实体) + GET /l3/triples (关系边)
 * 分层: L1 发光材料(圆心) → L2 材料大类 → L3 基质体系 → L4 具体材料
 * 横向: 激活剂离子(doped_with) / 发光特性(has_property) / 应用(used_in)
 * 交互: 滚轮围绕鼠标缩放 / 拖拽平移 / 悬停高亮+tooltip / 点击聚焦 / 图例过滤
 */
(function () {
  'use strict';
  var d = document;
  function g(id) { return d.getElementById(id); }
  function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  var VIEW = 'kb-graph';

  // 分层定义: 由 domain 前缀判定层级
  var LEVELS = {
    'L1': { label: '发光材料', color: '#f59e0b', r: 26 },
    'L2': { label: '材料大类', color: '#8b5cf6', r: 18 },
    'L3': { label: '基质体系', color: '#6366f1', r: 13 },
    'L4': { label: '具体材料', color: '#10b981', r: 11 },
    'activator': { label: '激活剂离子', color: '#ef4444', r: 10 },
    'property': { label: '发光特性', color: '#0ea5e9', r: 9 },
    'application': { label: '应用场景', color: '#ec4899', r: 9 },
  };
  var REL_COLORS = {
    part_of: '#cbd5e1',
    doped_with: '#f43f5e',
    has_property: '#06b6d4',
    used_in: '#a855f7',
  };
  var REL_LABELS = {
    part_of: '从属',
    doped_with: '掺杂',
    has_property: '特性',
    used_in: '应用',
  };

  function levelOf(domain) {
    if (!domain) return null;
    if (domain === 'L1') return 'L1';
    if (domain.indexOf('L2:') === 0) return 'L2';
    if (domain.indexOf('L3:') === 0) return 'L3';
    if (domain.indexOf('L4:') === 0) return 'L4';
    if (domain === 'activator' || domain === 'property' || domain === 'application') return domain;
    return null;
  }
  // domain 后缀 = 大类 (如 L2:oxide -> oxide), 用于 L2/L3/L4 分组对齐
  function familyOf(domain) {
    if (!domain) return '';
    var i = domain.indexOf(':');
    return i >= 0 ? domain.slice(i + 1) : domain;
  }

  function apiGet(path) {
    if (window.api && window.api.g) return window.api.g(path);
    var tk = localStorage.getItem('dt');
    var h = { 'Content-Type': 'application/json' };
    if (tk) h.Authorization = 'Bearer ' + tk;
    return fetch(path, { headers: h }).then(function (r) { return r.json().catch(function () { return {}; }); }).then(function (j) {
      if (j && j.code !== undefined && j.code !== 0) throw new Error(j.message || '请求失败');
      return j && j.data !== undefined ? j.data : j;
    });
  }

  // 图状态
  var G = { nodes: [], edges: [], scale: 1, tx: 0, ty: 0, focusId: null, hiddenLevels: {}, hoverId: null };

  function render() {
    var ct = g('content');
    if (!ct) return;
    ct.setAttribute('data-mf10', VIEW);
    ct.innerHTML =
      '<div class="r08-context-return"><button class="btn ghost" id="mf10BackKnowledge">← 返回知识与证据</button><span>高级视图：现有 KnowledgeEntity / Triple，不代表 R06 Concept Relation</span></div>' +
      '<div class="card" style="margin-bottom:14px">' +
      '<h3 style="margin:0 0 4px">🕸️ 领域知识图谱 · 分层结构</h3>' +
      '<p style="color:var(--muted);font-size:12.5px;margin:0">L1 发光材料 → L2 材料大类 → L3 基质体系 → L4 具体材料，横向串联激活剂 / 发光特性 / 应用</p>' +
      '</div>' +
      '<div class="card">' +
      '<div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;align-items:center">' +
      '<button class="btn ghost sm" id="mf10Reset">复位</button>' +
      '<button class="btn ghost sm" id="mf10Fit">适应窗口</button>' +
      '<span style="font-size:11px;color:var(--muted)">滚轮缩放 · 拖拽平移 · 悬停看详情 · 点节点聚焦</span>' +
      '</div>' +
      '<div id="mf10Legend" style="font-size:11px;color:var(--muted);margin-bottom:10px"></div>' +
      '<div id="mf10Graph" style="position:relative;border:1px solid var(--rule);border-radius:12px;background:var(--surface);overflow:hidden;height:640px;cursor:grab">' +
      '<div class="loading" id="mf10Loading"><span class="spinner"></span>正在加载分层知识图谱…</div>' +
      '<div id="mf10Tooltip" style="position:absolute;z-index:20;pointer-events:none;display:none;max-width:260px;background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:8px 10px;box-shadow:0 6px 18px rgba(0,0,0,.12);font-size:12px"></div>' +
      '</div>' +
      '<div id="mf10Detail" style="margin-top:10px"></div>' +
      '</div>';

    var back = g('mf10BackKnowledge');
    if (back) back.addEventListener('click', function () { if (window.sv) window.sv('kb'); });

    buildLegend();
    loadData();
  }

  function buildLegend() {
    var box = g('mf10Legend');
    if (!box) return;
    var html = '<span style="font-weight:600;margin-right:8px">层级:</span>';
    Object.keys(LEVELS).forEach(function (k) {
      html += '<span class="mf10-legend" data-level="' + k + '" style="margin-right:12px;cursor:pointer;padding:2px 6px;border-radius:4px">' +
        '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:' + LEVELS[k].color + ';margin-right:4px"></span>' +
        esc(LEVELS[k].label) + '</span>';
    });
    html += '<span style="font-weight:600;margin:0 8px">关系:</span>';
    Object.keys(REL_COLORS).forEach(function (r) {
      html += '<span style="margin-right:12px"><span style="display:inline-block;width:12px;height:2px;background:' + REL_COLORS[r] + ';margin-right:4px;vertical-align:middle"></span>' + esc(REL_LABELS[r] || r) + '</span>';
    });
    box.innerHTML = html;
    box.querySelectorAll('.mf10-legend[data-level]').forEach(function (el) {
      el.addEventListener('click', function () {
        var lv = el.getAttribute('data-level');
        G.hiddenLevels[lv] = !G.hiddenLevels[lv];
        el.style.opacity = G.hiddenLevels[lv] ? '0.35' : '1';
        el.style.textDecoration = G.hiddenLevels[lv] ? 'line-through' : 'none';
        renderGraph();
      });
    });
  }

  function loadData() {
    // 直接调分层图谱端点 (后端已按 domain 过滤层级实体, 不受 list_entities 排序影响)
    apiGet('/l3/graph/hierarchy').then(function (data) {
      var nodes = (data.nodes || []).map(function (n) {
        return {
          id: n.id,
          label: n.label || n.id,
          level: levelOf(n.domain),
          family: familyOf(n.domain),
          type: n.type,
          desc: n.desc || n.description || '',
        };
      }).filter(function (n) { return n.level; });

      var edges = (data.edges || []).map(function (e) {
        return { source: e.source, target: e.target, label: e.label };
      });

      G.nodes = nodes;
      G.edges = edges;
      G.scale = 1; G.tx = 0; G.ty = 0; G.focusId = null; G.hoverId = null;
      var ld = g('mf10Loading');
      if (ld) ld.style.display = 'none';
      renderGraph();
    }).catch(function (e) {
      g('mf10Loading').innerHTML = '<div class="error-banner">' + esc(e.message || '加载失败') + '</div>';
    });
  }

  // 分层径向布局: 按大类分扇形, L2 内圈 → L4 外圈, 横向节点放最外圈
  function computeLayout(nodes) {
    var box = g('mf10Graph');
    var W = (box && box.clientWidth) || 900, H = 640;
    var cx = W / 2, cy = H / 2;

    var visible = nodes.filter(function (n) { return !G.hiddenLevels[n.level]; });
    var byLevel = {};
    visible.forEach(function (n) {
      (byLevel[n.level] = byLevel[n.level] || []).push(n);
    });

    // 收集大类 (L2 的数量决定扇区)
    var families = [];
    ['L2', 'L3', 'L4'].forEach(function (lv) {
      (byLevel[lv] || []).forEach(function (n) {
        if (n.family && families.indexOf(n.family) < 0) families.push(n.family);
      });
    });
    if (!families.length) families = [''];

    var RADII = { L1: 0, L2: 140, L3: 250, L4: 360 };
    // 横向节点按类型分圈 (避免同一圈拥挤重叠)
    var HORIZ_R = { activator: 460, property: 520, application: 580 };
    var pos = {};

    // L1 根: 圆心
    (byLevel['L1'] || []).forEach(function (n) { pos[n.id] = { x: cx, y: cy }; });

    // L2 大类: 均匀分布
    var l2 = byLevel['L2'] || [];
    l2.forEach(function (n, i) {
      var ang = (i / Math.max(l2.length, 1)) * 2 * Math.PI - Math.PI / 2;
      pos[n.id] = { x: cx + Math.cos(ang) * RADII.L2, y: cy + Math.sin(ang) * RADII.L2, ang: ang };
    });

    // 每个大类的扇区中心角
    function familyAngle(fam) {
      var i = families.indexOf(fam);
      if (i < 0) i = families.length;
      return (i / Math.max(families.length, 1)) * 2 * Math.PI - Math.PI / 2;
    }

    // L3 基质: 在所属大类的扇区内
    var l3ByFam = groupBy(byLevel['L3'] || [], 'family');
    families.forEach(function (fam) {
      var arr = l3ByFam[fam] || [];
      var base = familyAngle(fam);
      var span = (2 * Math.PI) / Math.max(families.length, 1);
      arr.forEach(function (n, i) {
        var ang = base + (arr.length > 1 ? ((i - (arr.length - 1) / 2) / arr.length) * span * 0.8 : 0);
        pos[n.id] = { x: cx + Math.cos(ang) * RADII.L3, y: cy + Math.sin(ang) * RADII.L3, ang: ang };
      });
    });

    // L4 材料: 在所属大类的扇区内 (更外圈)
    var l4ByFam = groupBy(byLevel['L4'] || [], 'family');
    families.forEach(function (fam) {
      var arr = l4ByFam[fam] || [];
      var base = familyAngle(fam);
      var span = (2 * Math.PI) / Math.max(families.length, 1);
      arr.forEach(function (n, i) {
        var ang = base + (arr.length > 1 ? ((i - (arr.length - 1) / 2) / arr.length) * span * 0.85 : 0);
        pos[n.id] = { x: cx + Math.cos(ang) * RADII.L4, y: cy + Math.sin(ang) * RADII.L4, ang: ang };
      });
    });

    // 横向节点 (激活剂/特性/应用): 分三圈放置, 避免同一圈拥挤重叠
    var horizOffset = { activator: 0, property: 0.5, application: 1.0 };
    ['activator', 'property', 'application'].forEach(function (lv) {
      var arr = byLevel[lv] || [];
      var R = HORIZ_R[lv];
      arr.forEach(function (n, i) {
        var ang = (i / Math.max(arr.length, 1)) * 2 * Math.PI + (horizOffset[lv] || 0);
        pos[n.id] = { x: cx + Math.cos(ang) * R, y: cy + Math.sin(ang) * R, ang: ang };
      });
    });

    return { pos: pos, W: W, H: H, cx: cx, cy: cy };
  }

  function groupBy(arr, key) {
    var m = {};
    arr.forEach(function (n) { (m[n[key] || ''] = m[n[key] || ''] || []).push(n); });
    return m;
  }

  function renderGraph() {
    var box = g('mf10Graph');
    if (!box || !G.nodes.length) return;
    var nodes = G.nodes.filter(function (n) { return !G.hiddenLevels[n.level]; });
    var nodeMap = {};
    nodes.forEach(function (n) { nodeMap[n.id] = n; });
    var edges = G.edges.filter(function (e) { return nodeMap[e.source] && nodeMap[e.target]; });

    var L = computeLayout(nodes);
    var pos = L.pos;

    // 邻居集合 (聚焦高亮)
    var neighbors = {};
    if (G.focusId) {
      edges.forEach(function (e) {
        if (e.source === G.focusId) neighbors[e.target] = true;
        if (e.target === G.focusId) neighbors[e.source] = true;
      });
    }

    var statHtml =
      '<div class="stat-strip" style="position:absolute;top:10px;left:10px;z-index:5;background:transparent;border:none">' +
      '<div class="stat-card" style="min-width:56px"><div class="num">' + nodes.length + '</div><div class="lbl">实体</div></div>' +
      '<div class="stat-card" style="min-width:56px"><div class="num">' + edges.length + '</div><div class="lbl">关系</div></div>' +
      '</div>';

    var svg = '<svg id="mf10Svg" width="' + L.W + '" height="' + L.H + '" style="display:block;background:var(--surface)">';
    svg += '<g id="mf10G" transform="translate(' + G.tx + ',' + G.ty + ') scale(' + G.scale + ')">';

    // 分层同心圆底纹 (帮助识别层级)
    [140, 250, 360, 460, 520, 580].forEach(function (r) {
      svg += '<circle cx="' + L.cx + '" cy="' + L.cy + '" r="' + r + '" fill="none" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3,5" opacity="0.5"/>';
    });

    // 边
    edges.forEach(function (e, ei) {
      var s = pos[e.source], t = pos[e.target];
      if (!s || !t) return;
      var color = REL_COLORS[e.label] || '#94a3b8';
      var isFocusEdge = G.focusId && (e.source === G.focusId || e.target === G.focusId);
      var isHoverEdge = G.hoverId && (e.source === G.hoverId || e.target === G.hoverId);
      var w = (isFocusEdge || isHoverEdge) ? 2.5 : 1;
      var op = G.focusId ? (isFocusEdge ? 0.9 : 0.08) : (G.hoverId ? (isHoverEdge ? 0.9 : 0.15) : 0.35);
      svg += '<line id="mf10e' + ei + '" x1="' + s.x + '" y1="' + s.y + '" x2="' + t.x + '" y2="' + t.y + '" stroke="' + color + '" stroke-width="' + w + '" stroke-opacity="' + op + '" style="transition:stroke-opacity .2s,stroke-width .2s"/>';
    });

    // 节点
    nodes.forEach(function (n) {
      var lv = LEVELS[n.level] || LEVELS['L4'];
      var p = pos[n.id];
      if (!p) return;
      var isFocus = n.id === G.focusId;
      var isNeighbor = neighbors[n.id];
      var isDimmed = (G.focusId || G.hoverId) && !isFocus && !isNeighbor && n.id !== G.hoverId;
      var op = isDimmed ? 0.2 : 0.95;
      var stroke = isFocus ? '#ef4444' : (isNeighbor ? '#f59e0b' : '#fff');
      var sw = isFocus ? 3 : (isNeighbor ? 2 : 1.2);
      var r = lv.r;
      svg += '<circle class="mf10node" data-id="' + esc(n.id) + '" cx="' + p.x + '" cy="' + p.y + '" r="' + r + '" fill="' + lv.color + '" fill-opacity="' + op + '" stroke="' + stroke + '" stroke-width="' + sw + '" style="cursor:pointer;transition:fill-opacity .2s,stroke .2s"/>';
      var labelSize = (n.level === 'L1' || n.level === 'L2') ? 11 : 9;
      var showLabel = n.level === 'L1' || n.level === 'L2' || isFocus || isNeighbor || (n.level === 'L3');
      if (showLabel) {
        svg += '<text x="' + p.x + '" y="' + (p.y - r - 5) + '" text-anchor="middle" font-size="' + labelSize + '" fill="var(--muted)" pointer-events="none">' + esc(String(n.label).slice(0, 16)) + '</text>';
      }
    });

    svg += '</g></svg>';
    box.innerHTML = statHtml + svg;
    bindInteractions(box, nodes, edges, nodeMap, pos, L);
  }

  function bindInteractions(box, nodes, edges, nodeMap, pos, L) {
    var svg = g('mf10Svg');
    var group = g('mf10G');
    var tip = g('mf10Tooltip');
    if (!svg || !group) return;

    function applyTransform() {
      group.setAttribute('transform', 'translate(' + G.tx + ',' + G.ty + ') scale(' + G.scale + ')');
    }

    // 滚轮缩放 (围绕鼠标位置)
    svg.addEventListener('wheel', function (e) {
      e.preventDefault();
      var rect = svg.getBoundingClientRect();
      var mx = e.clientX - rect.left, my = e.clientY - rect.top;
      var delta = e.deltaY > 0 ? 0.9 : 1.1;
      var ns = Math.max(0.3, Math.min(4, G.scale * delta));
      // 保持鼠标下的点不动: 新平移 = 鼠标 - (鼠标 - 旧平移) * (新scale/旧scale)
      var k = ns / G.scale;
      G.tx = mx - (mx - G.tx) * k;
      G.ty = my - (my - G.ty) * k;
      G.scale = ns;
      applyTransform();
    }, { passive: false });

    // 拖拽平移
    var dragging = false, sx = 0, sy = 0;
    svg.addEventListener('mousedown', function (e) {
      if (e.target.classList && e.target.classList.contains('mf10node')) return;
      dragging = true; sx = e.clientX; sy = e.clientY;
      svg.style.cursor = 'grabbing';
    });
    d.addEventListener('mousemove', function (e) {
      if (dragging) {
        G.tx += e.clientX - sx; G.ty += e.clientY - sy;
        sx = e.clientX; sy = e.clientY;
        applyTransform();
      }
    });
    d.addEventListener('mouseup', function () { dragging = false; svg.style.cursor = 'grab'; });

    // 节点交互: 悬停高亮 + tooltip, 点击聚焦
    box.querySelectorAll('.mf10node').forEach(function (circle) {
      circle.addEventListener('mouseenter', function (e) {
        var id = circle.getAttribute('data-id');
        G.hoverId = id;
        circle.setAttribute('stroke-width', '3');
        circle.setAttribute('stroke', '#ef4444');
        renderGraph();
        // tooltip
        var n = nodeMap[id];
        if (n && tip) {
          tip.innerHTML = '<div style="font-weight:600;margin-bottom:2px">' + esc(n.label) + '</div>' +
            '<div style="color:var(--muted);font-size:11px">' + esc((LEVELS[n.level] || {}).label || n.level) + (n.desc ? ' · ' + esc(String(n.desc).slice(0, 60)) : '') + '</div>';
          tip.style.display = 'block';
        }
      });
      circle.addEventListener('mousemove', function (e) {
        if (tip) {
          var rect = svg.getBoundingClientRect();
          tip.style.left = (e.clientX - rect.left + 14) + 'px';
          tip.style.top = (e.clientY - rect.top + 12) + 'px';
        }
      });
      circle.addEventListener('mouseleave', function () {
        G.hoverId = null;
        if (tip) tip.style.display = 'none';
        renderGraph();
      });
      circle.addEventListener('click', function (e) {
        e.stopPropagation();
        var id = circle.getAttribute('data-id');
        G.focusId = (G.focusId === id) ? null : id;
        renderGraph();
        showDetail(id, nodes, edges, nodeMap);
      });
    });
  }

  function showDetail(id, nodes, edges, nodeMap) {
    var box = g('mf10Detail');
    if (!box) return;
    var node = nodeMap[id];
    if (!node) { box.innerHTML = ''; return; }
    var related = [];
    edges.forEach(function (e) {
      if (e.source === id) related.push({ dir: '→', other: nodeMap[e.target], rel: e.label });
      if (e.target === id) related.push({ dir: '←', other: nodeMap[e.source], rel: e.label });
    });
    var relatedHtml = related.slice(0, 15).map(function (r) {
      if (!r.other) return '';
      return '<div style="padding:4px 0;border-bottom:1px solid var(--rule);font-size:12px">' +
        '<span style="color:' + (REL_COLORS[r.rel] || '#94a3b8') + '">' + esc(REL_LABELS[r.rel] || r.rel) + ' ' + r.dir + '</span> ' +
        '<span style="font-weight:500">' + esc(r.other.label || r.other.id) + '</span>' +
        '<span class="badge info" style="margin-left:6px;font-size:10px">' + esc((LEVELS[r.other.level] || {}).label || r.other.level) + '</span></div>';
    }).join('');

    box.innerHTML = '<div class="card" style="margin-top:0;padding:14px 18px"><h4 style="margin:0 0 8px">' +
      '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:' + ((LEVELS[node.level] || {}).color || '#6b7280') + ';margin-right:6px"></span>' +
      esc(node.label || node.id) +
      ' <span class="badge info">' + esc((LEVELS[node.level] || {}).label || node.level) + '</span></h4>' +
      (node.desc ? '<p style="color:var(--muted);font-size:12px;margin:0 0 8px">' + esc(node.desc) + '</p>' : '') +
      (relatedHtml || '<p style="color:var(--muted);font-size:12px">该实体暂无关联关系</p>') +
      '<button class="btn ghost sm" style="margin-top:8px" id="mf10ClearFocus">取消聚焦</button></div>';
    g('mf10ClearFocus').addEventListener('click', function () {
      G.focusId = null;
      renderGraph();
      box.innerHTML = '';
    });
  }

  function bindControls() {
    var resetBtn = g('mf10Reset');
    var fitBtn = g('mf10Fit');
    if (resetBtn) resetBtn.addEventListener('click', function () { G.scale = 1; G.tx = 0; G.ty = 0; G.focusId = null; G.hoverId = null; renderGraph(); g('mf10Detail').innerHTML = ''; });
    if (fitBtn) fitBtn.addEventListener('click', function () { G.scale = 1; G.tx = 0; G.ty = 0; renderGraph(); });
  }

  d.addEventListener('view-rendered', function (e) {
    if (e.detail && e.detail.view === VIEW) { render(); setTimeout(bindControls, 100); }
  });
})();
