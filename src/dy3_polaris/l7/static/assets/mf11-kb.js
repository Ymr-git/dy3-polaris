/* Dy3+ Polaris — M-F11 知识库浏览（分类 + 真翻页）
 * 数据源: GET /l3/entities (支持 entity_type/limit/offset)
 * 分类展示: 材料 / 制备方法 / 表征方法 / 化合物 / 概念 / 文献
 * 真正后端分页, 非本地截断.
 */
(function () {
  'use strict';
  var d = document;
  function g(id) { return d.getElementById(id); }
  function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  var VIEW = 'kb';

  // 分类定义: tab 名称 -> 后端 entity_type 过滤
  var CATEGORIES = [
    { key: 'material', label: '材料', icon: '🧱', desc: '基质与材料体系' },
    { key: 'method', label: '制备/表征', icon: '🔬', desc: '合成与测试方法' },
    { key: 'chemical_compound', label: '化合物', icon: '⚗️', desc: '稀土离子与化学式' },
    { key: 'paper', label: '文献', icon: '📄', desc: 'DOI 与来源文献' },
    { key: 'concept', label: '概念', icon: '💡', desc: '领域概念' },
  ];

  var state = { cat: 'material', page: 0, limit: 15 };

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

  function render() {
    var ct = g('content');
    if (!ct) return;
    ct.setAttribute('data-mf11', VIEW);
    ct.innerHTML =
      '<div class="card" style="margin-bottom:14px">' +
      '<h3 style="margin:0 0 4px">📚 知识库</h3>' +
      '<p style="color:var(--muted);font-size:12.5px;margin:0">绿色健康照明发光材料 · Dy 垂直领域知识实体（按类别浏览）</p>' +
      '</div>' +
      '<div class="card">' +
      '<div id="mf11Tabs" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px"></div>' +
      '<div id="mf11Stats" class="stat-strip" style="margin-bottom:12px"></div>' +
      '<div id="mf11List" class="table-wrap" style="max-height:480px"></div>' +
      '<div id="mf11Pager" style="display:flex;align-items:center;justify-content:center;gap:8px;margin-top:12px"></div>' +
      '</div>';

    buildTabs();
    loadCategory(state.cat, state.page);
  }

  function buildTabs() {
    var box = g('mf11Tabs');
    if (!box) return;
    box.innerHTML = CATEGORIES.map(function (c) {
      var active = c.key === state.cat;
      return '<button class="btn ' + (active ? 'primary' : 'ghost') + ' sm" data-cat="' + c.key + '" title="' + esc(c.desc) + '">' +
        c.icon + ' ' + c.label + '</button>';
    }).join('');
    box.querySelectorAll('[data-cat]').forEach(function (b) {
      b.addEventListener('click', function () {
        state.cat = b.getAttribute('data-cat');
        state.page = 0;
        buildTabs();
        loadCategory(state.cat, state.page);
      });
    });
  }

  function loadCategory(cat, page) {
    var offset = page * state.limit;
    var url = '/l3/entities?entity_type=' + encodeURIComponent(cat) + '&limit=' + state.limit + '&offset=' + offset;
    apiGet(url).then(function (data) {
      renderList(data, cat, page);
    }).catch(function (e) {
      g('mf11List').innerHTML = '<div class="error-banner">' + esc(e.message || '加载失败') + '</div>';
    });
  }

  function renderList(data, cat, page) {
    var items = data.items || [];
    var total = data.total || 0;
    var hasMore = data.has_more;

    // 统计条带
    g('mf11Stats').innerHTML =
      '<div class="stat-card"><div class="num">' + total + '</div><div class="lbl">' + esc(catLabel(cat)) + '总数</div></div>' +
      '<div class="stat-card"><div class="num">' + items.length + '</div><div class="lbl">本页</div></div>';

    // 列表
    var rows = items.length ? items.map(function (x) {
      var name = x.name || x.entity_id || '?';
      var domain = x.domain || '-';
      var type = x.entity_type || cat;
      return '<tr>' +
        '<td style="padding:8px 14px"><span style="font-weight:500">' + esc(name) + '</span></td>' +
        '<td style="padding:8px 14px"><span class="badge info">' + esc(type) + '</span></td>' +
        '<td style="padding:8px 14px;color:var(--muted)">' + esc(domain) + '</td>' +
        '</tr>';
    }).join('') : '<tr><td colspan="3" style="text-align:center;color:var(--muted);padding:24px">该类别暂无实体</td></tr>';

    g('mf11List').innerHTML =
      '<table><thead><tr><th>名称</th><th>类型</th><th>领域</th></tr></thead><tbody>' + rows + '</tbody></table>';

    // 分页
    var totalPages = Math.max(1, Math.ceil(total / state.limit));
    var pager = '<button class="btn ghost sm" id="mf11Prev"' + (page <= 0 ? ' disabled' : '') + '>‹ 上一页</button>' +
      '<span style="font-size:12px;color:var(--muted)">第 ' + (page + 1) + ' / ' + totalPages + ' 页</span>' +
      '<button class="btn ghost sm" id="mf11Next"' + (!hasMore ? ' disabled' : '') + '>下一页 ›</button>';
    g('mf11Pager').innerHTML = pager;

    var prev = g('mf11Prev'), next = g('mf11Next');
    if (prev && page > 0) prev.addEventListener('click', function () { state.page--; loadCategory(state.cat, state.page); });
    if (next && hasMore) next.addEventListener('click', function () { state.page++; loadCategory(state.cat, state.page); });
  }

  function catLabel(cat) {
    var c = CATEGORIES.filter(function (x) { return x.key === cat; })[0];
    return c ? c.label : cat;
  }

  // R08B-2: the task-driven Knowledge & Evidence renderer in mf6 is the
  // authoritative `kb` experience. Keep this legacy entity browser available
  // for maintenance/debug use, but do not let it overwrite the learner route.
  window.MF11KnowledgeBrowser = window.MF11KnowledgeBrowser || {};
  window.MF11KnowledgeBrowser.renderLegacyEntityBrowser = render;
})();
