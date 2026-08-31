/* Dy3+ Polaris — M-F6 功能增强 (会话管理 / 设置模块 / 对比 / 时间旅行)
 * 独立于 app.js 压缩文件, 通过钩子注入增强既有视图。
 * 版本: 2026081205 (多线协作 + 协同决策中间数据可视化增强)
 */
(function () {
  'use strict';

  var d = document;
  function g(id) { return d.getElementById(id); }
  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function escAttr(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function toast(msg) {
    var t = g('toast');
    if (!t) return;
    t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(function () { t.hidden = true; }, 2600);
  }
  function statCard(l, v) {
    return '<div class="stat-card"><div class="num" style="font-size:16px">' + esc(v) + '</div><div class="lbl">' + esc(l) + '</div></div>';
  }
  function token() { return localStorage.getItem('dt') || ''; }
  function apiReq(m, p, b) {
    var h = { 'Content-Type': 'application/json' };
    var tk = token();
    if (tk) h.Authorization = 'Bearer ' + tk;
    return fetch(p, { method: m, headers: h, body: b ? JSON.stringify(b) : void 0 })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok || (j && j.code !== 0)) { throw new Error((j && j.message) || ('HTTP ' + r.status)); }
        return j ? j.data : null;
      }); });
  }
  function fmtPct(v) { return Math.round((v || 0) * 10000) / 100 + '%'; }

  // 启发式导学: 根据问题与答案, 自然生成 2 个深化方向的追问建议 (可点选继续)
  function heuristicFollowUps(text, answer) {
    var t = String(text || '').toLowerCase();
    var a = String(answer || '').toLowerCase();
    var topics = [];
    if (/(dy|镝)/i.test(t)) topics.push('Dy³⁺');
    if (/(eu|铕)/i.test(t)) topics.push('Eu³⁺');
    if (/(ce|铈)/i.test(t)) topics.push('Ce³⁺');
    if (/(tb|铽)/i.test(t)) topics.push('Tb³⁺');
    if (/(猝灭|浓度..灭|quenching)/i.test(t)) topics.push('浓度猝灭');
    if (/(量子|效率)/i.test(t)) topics.push('量子效率');
    if (/(能级|光谱|跃迁|transition)/i.test(t)) topics.push('能级与光谱');
    if (/(发光|荧光|磷光)/i.test(t)) topics.push('发光机理');
    if (/(制备|合成|掺杂|工艺)/i.test(t)) topics.push('制备与合成');
    if (/(热稳定|热猝灭|温度)/i.test(t)) topics.push('热稳定性');
    if (/(应用|led|照明|显示)/i.test(t)) topics.push('实际应用');
    if (/(寿命|衰减|余辉)/i.test(t)) topics.push('荧光寿命');
    if (/(能量传递|交叉弛豫|敏化)/i.test(t)) topics.push('能量传递');
    if (/(色度|显色|色温|cri)/i.test(t)) topics.push('色度与显色');
    if (!topics.length) {
      var kw = ['发光', '猝灭', '能级', '跃迁', '光谱', '效率', '制备', '掺杂', '温度', '寿命', '能量传递', '色度'];
      for (var i = 0; i < kw.length; i++) {
        if (a.indexOf(kw[i]) !== -1) { topics.push(kw[i]); if (topics.length >= 2) break; }
      }
    }
    var out = [];
    var seen = {};
    var cands = [
      function (tp) { return '「' + tp + '」的发光机理是什么？'; },
      function (tp) { return '「' + tp + '」怎么制备合成？'; },
      function (tp) { return '「' + tp + '」的能级跃迁与光谱怎么看？'; },
      function (tp) { return '「' + tp + '」有哪些实际应用？'; },
      function (tp) { return '「' + tp + '」的量子效率如何提高？'; },
      function (tp) { return '「' + tp + '」的温度稳定性如何？'; },
    ];
    topics.forEach(function (tp, idx) {
      var q;
      if (/(猝灭|效率|稳定|寿命|传递|色度)/.test(tp)) q = cands[4](tp);
      else q = cands[idx % 2](tp);
      if (!seen[q]) { seen[q] = 1; out.push(q); }
    });
    return out.slice(0, 2);
  }

  /* ---------- i18n (轻量, 基于 L7 translate 字典) ---------- */
  var I18N = {
    'zh-CN': {
      sessions: '会话管理', create: '新建会话', fork: 'Fork', pause: '暂停', resume: '恢复',
      type_diagnosis: '学情诊断', type_practice: '练习', type_query: '实时答疑', type_debate: '辩论', type_learning: '知识学习', type_assessment: '测评考核',
      status_active: '进行中', status_paused: '已暂停', status_completed: '已完成', status_forked: '分支',
      empty_sessions: '暂无会话 — 点击"新建会话"开始', created: '创建时间', action: '操作',
      settings: '设置', language: '语言', contrast: '高对比度', contrast_on: '开启', contrast_off: '关闭',
      compare: '学习对比', time_travel: '时间旅行', learner: '学习者', mastery: '总体掌握度',
      weak_points: '薄弱点', no_learners: '无学习者数据', snapshot: '快照', back: '返回',
      practice: '学习练习',
    },
    'en-US': {
      sessions: 'Sessions', create: 'New Session', fork: 'Fork', pause: 'Pause', resume: 'Resume',
      type_diagnosis: 'Diagnosis', type_practice: 'Practice', type_query: 'Q&A', type_debate: 'Debate',
      status_active: 'Active', status_paused: 'Paused', status_completed: 'Completed', status_forked: 'Forked',
      empty_sessions: 'No sessions yet — click "New Session" to start', created: 'Created', action: 'Action',
      settings: 'Settings', language: 'Language', contrast: 'High Contrast', contrast_on: 'On', contrast_off: 'Off',
      compare: 'Compare', time_travel: 'Time Travel', learner: 'Learner', mastery: 'Overall Mastery',
      weak_points: 'Weak Points', no_learners: 'No learner data', snapshot: 'Snapshot', back: 'Back',
      practice: 'Practice',
    },
  };
  var currentLocale = localStorage.getItem('dlocale') || 'zh-CN';
  function t(key) {
    var dict = I18N[currentLocale] || I18N['zh-CN'];
    return dict[key] || key;
  }
  window.Dy3I18N = {
    setLocale: function (loc) {
      currentLocale = loc;
      localStorage.setItem('dlocale', loc);
      // 触发现有视图重渲染
      if (window.sv) { try { window.sv(window.S && window.S.v || 'overview'); } catch (e) { /* 忽略 */ } }
      toast('Locale: ' + loc);
    },
    getLocale: function () { return currentLocale; },
    t: t,
  };

  /* ---------- 会话管理视图 (query-history) — 统一会话闭环 ---------- */
  function renderSessions() {
    var ct = g('content');
    if (!ct) return;
    apiReq('GET', '/l1/api/v1/sessions').then(function (data) {
      var items = Array.isArray(data) ? data : ((data && data.items) || []);
      var rows = items.length ? items.map(function (s) {
        var typeLabel = t('type_' + (s.session_type || 'diagnosis')) || s.session_type;
        var stKey = 'status_' + (s.status || 'active');
        var stLabel = t(stKey) || s.status;
        var stColor = s.status === 'active' ? 'ok' : (s.status === 'paused' ? 'warn' : '');
        var qCount = s.question_count != null ? s.question_count : '-';
        var execCount = s.agent_execution_count != null ? s.agent_execution_count : '-';
        var ts = Number(s.created_at || 0);
        if (ts > 1e12) ts = ts / 1000; /* 毫秒兼容 */
        var createdLabel = ts ? new Date(ts).toLocaleString('zh-CN') : '-';
        return '<tr><td>' + esc(s.session_id) + '</td><td>' + esc(typeLabel) +
          '</td><td><span class="badge ' + stColor + '">' + esc(stLabel) + '</span></td><td>' +
          esc(String(qCount)) + '</td><td>' + esc(String(execCount)) + '</td><td>' +
          createdLabel +
          '</td><td style="white-space:nowrap">' +
          (s.status === 'active' ? '<button class="btn ghost" data-fork="' + esc(s.session_id) + '">' + t('fork') + '</button> ' +
            '<button class="btn ghost" data-pause="' + esc(s.session_id) + '">' + t('pause') + '</button>' : '') +
          '</td></tr>';
      }).join('') : '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:24px">' +
        t('empty_sessions') + '</td></tr>';

      ct.innerHTML = '<div class="card"><h3>' + t('sessions') + '</h3>' +
        '<p style="color:var(--muted);font-size:12px;margin-bottom:10px">统一会话：实时答疑的提问会自动归入「实时答疑」类型会话，并关联 Agent 执行记录。</p>' +
        '<div style="margin-bottom:12px"><button class="btn primary" id="mf6Create">+' + t('create') + '</button></div>' +
        '<div class="table-wrap"><table><thead><tr><th>ID</th><th>' + t('type_diagnosis').replace('学情诊断','类型') + '</th><th>' + t('status_active').replace('进行中','状态') + '</th><th>提问数</th><th>Agent 执行</th><th>' + t('created') + '</th><th>' + t('action') + '</th></tr></thead><tbody>' + rows + '</tbody></table></div></div>';

      var cb = g('mf6Create');
      if (cb) cb.addEventListener('click', function () {
        apiReq('POST', '/l1/api/v1/sessions', { session_type: 'query' })
          .then(function () { toast('OK'); renderSessions(); })
          .catch(function (e) { toast(e.message); });
      });
      ct.querySelectorAll('[data-fork]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          apiReq('POST', '/l1/api/v1/sessions/' + btn.dataset.fork + '/fork', { branch_label: 'mf6' })
            .then(function () { toast('OK'); renderSessions(); })
            .catch(function (e) { toast(e.message); });
        });
      });
      ct.querySelectorAll('[data-pause]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          apiReq('POST', '/l1/api/v1/sessions/' + btn.dataset.pause + '/pause')
            .then(function () { toast('OK'); renderSessions(); })
            .catch(function (e) { toast(e.message); });
        });
      });
    }).catch(function (e) {
      // 未登录/认证失败 → 引导登录; 其余 → 错误展示
      var em = (e && (e.message || '')) || '';
      var needLogin = e && (e.status === 401 || e.status === 403 || em.indexOf('Authentication') >= 0 || em.indexOf('AUTHENTICATION') >= 0 || em.indexOf('登录') >= 0 || em === 'HTTP 401');
      if (needLogin) {
        ct.innerHTML = '<div class="card" style="text-align:center;padding:40px"><h3>' + t('sessions') + '</h3>' +
          '<p style="color:var(--muted);margin:14px 0">会话记录需要登录后查看。</p>' +
          '<button class="btn primary" onclick="olv()">登录</button></div>';
      } else {
        ct.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
      }
    });
  }

  /* ---------- 设置增强 (语言/高对比度) ---------- */
  // 设置视图: 渲染到内容区 (角色/主题/语言/高对比/学习数据 + 退出登录)
  // 紧凑布局: 状态条单行 + 界面设置单行 + 功能键 (API 配置弹窗) + 学习数据折叠
  function renderSettings() {
    var ct = g('content');
    if (!ct) return;
    // 关闭 app.js st() 可能弹出的登录 modal (设置渲染到内容区而非弹窗)
    var pm = g('passwordModal');
    if (pm) pm.hidden = 1;
    var logged = !!token();
    var role = currentRole() || 'Guest';
    var theme = d.documentElement.getAttribute('data-theme') === 'dark' ? '深色' : '浅色';
    var sid = window.S && window.S.lid ? window.S.lid : (localStorage.getItem('dl') || '-');
    var savedApiModel = localStorage.getItem('dy3_api_model') || 'deepseek-v4-flash';
    // 学习数据紧凑行
    var dataBox = '<div class="callout" style="margin-top:12px;padding:8px 12px"><strong>学习数据</strong> <span id="mf6DataInfo" style="font-size:12px">正在读取画像…</span></div>';
    ct.innerHTML = '<div class="card"><h3>设置</h3>' +
      // 状态条: 单行紧凑 (角色/主题/状态/用户)
      '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;font-size:13px">' +
      '<span class="badge info">角色: ' + esc(role) + '</span>' +
      '<span class="badge info">主题: ' + esc(theme) + '</span>' +
      '<span class="badge ' + (logged ? 'ok' : 'warn') + '">' + (logged ? '在线' : '游客') + '</span>' +
      '<span style="color:var(--muted)">用户: ' + esc(sid) + '</span></div>' +
      // 界面设置: 单行按钮组
      '<div class="section-header"><span class="section-icon">⚙️</span><h4>界面设置</h4><span class="section-line"></span></div>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px">' +
      '<button class="btn ghost" id="setTheme">切换主题</button>' +
      '<button class="btn ghost" id="setContrast">' + (d.documentElement.classList.contains('high-contrast') ? '关闭' : '开启') + '高对比度</button>' +
      '<button class="btn ghost" id="setBgWhite"' + (localStorage.getItem('dy3_bg') === 'white' ? ' style="background:var(--accent-soft);border-color:var(--accent)"' : '') + '>白色背景</button>' +
      '<button class="btn ghost" id="setBgWarm"' + (localStorage.getItem('dy3_bg') !== 'white' ? ' style="background:var(--accent-soft);border-color:var(--accent)"' : '') + '>暖色背景</button>' +
      (logged ? '<button class="btn" id="setLogout" style="background:#dc2626;color:#fff;border-color:#dc2626">退出登录</button>'
        : '<button class="btn primary" id="setLogin">登录</button>') +
      '</div>' +
      // 功能键: API 配置 (点击弹出弹窗, 参照 Trae 设置风格)
      '<div class="section-header" style="margin-top:14px"><span class="section-icon">🔑</span><h4>模型接入</h4><span class="section-line"></span></div>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:4px">' +
      '<button class="btn primary" id="openApiConfig">API 配置</button>' +
      '<span style="font-size:12px;color:var(--muted)">当前模型: <b>' + esc(savedApiModel) + '</b></span>' +
      '</div>' +
      dataBox + '</div>';

    var th = g('setTheme');
    if (th) th.addEventListener('click', function () {
      d.documentElement.setAttribute('data-theme', d.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
      localStorage.setItem('dh', d.documentElement.getAttribute('data-theme'));
      renderSettings();
    });
    var cont = g('setContrast');
    if (cont) cont.addEventListener('click', function () {
      d.documentElement.classList.toggle('high-contrast');
      renderSettings();
    });
    var bgWhite = g('setBgWhite');
    if (bgWhite) bgWhite.addEventListener('click', function() {
      localStorage.setItem('dy3_bg', 'white');
      document.documentElement.setAttribute('data-bg', 'white');
      renderSettings();
    });
    var bgWarm = g('setBgWarm');
    if (bgWarm) bgWarm.addEventListener('click', function() {
      localStorage.setItem('dy3_bg', 'warm');
      document.documentElement.setAttribute('data-bg', 'warm');
      renderSettings();
    });
    var lg = g('setLogout');
    if (lg) lg.addEventListener('click', function () {
      if (window.auth && window.auth.logout) window.auth.logout();
      else { localStorage.removeItem('dt'); location.reload(); }
    });
    var li = g('setLogin');
    if (li) li.addEventListener('click', function () { if (window.olv) window.olv(); });
    // API 配置功能键 → 弹出弹窗
    var oac = g('openApiConfig');
    if (oac) oac.addEventListener('click', function () { openApiConfigModal(); });

    apiReq('GET', '/l2/profile/' + learnerId()).then(function (p) {
      var di = g('mf6DataInfo');
      if (!di) return;
      var km = (p && p.kp_mastery) || {};
      var vals = Object.keys(km).map(function (k) { return km[k]; });
      var avg = vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : 0;
      var ts = p && p.snapshot_ts ? new Date((p.snapshot_ts > 1e12 ? p.snapshot_ts / 1000 : p.snapshot_ts) * 1000).toLocaleString('zh-CN') : '-';
      di.innerHTML = '学习者 ' + esc((p && p.learner_id) || '-') +
        ' · 追踪 KP ' + Object.keys(km).length +
        ' · 平均掌握度 ' + fmtPct2(avg) +
        ' · 更新 ' + esc(ts) +
        ' · 题库 38 题 (稀土发光材料) · BKT 追踪 · 画像 JSON 持久化';
    }).catch(function () {
      var di = g('mf6DataInfo');
      if (di) di.textContent = '画像读取失败 (需登录)';
    });
  }

  /* ---------- API 配置弹窗 (provider 选择 + 预设地址/模型, 用户只填 Key) ---------- */
  var LLM_PROVIDERS = [
    { key: 'deepseek', name: 'DeepSeek', url: 'https://api.deepseek.com', model: 'deepseek-v4-flash' },
    { key: 'qwen', name: '通义千问', url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: 'qwen-max' },
    { key: 'zhipu', name: '智谱 GLM', url: 'https://open.bigmodel.cn/api/paas/v4', model: 'glm-5.1' },
    { key: 'kimi', name: 'Kimi 月之暗面', url: 'https://api.moonshot.cn/v1', model: 'kimi-k2.6' },
    { key: 'minimax', name: 'MiniMax', url: 'https://api.minimax.chat/v1', model: 'minimax-m2.7' },
    { key: 'openai', name: 'OpenAI', url: 'https://api.openai.com/v1', model: 'gpt-4o' },
    { key: 'ollama', name: 'Ollama 本地', url: 'http://localhost:11434/v1', model: 'qwen2.5:7b' },
    { key: 'custom', name: '自定义', url: '', model: '' },
  ];

  function openApiConfigModal() {
    var savedProv = localStorage.getItem('dy3_api_provider') || 'deepseek';
    // Provider secrets are request-only. Remove legacy browser-persisted keys.
    localStorage.removeItem('dy3_api_key');
    var savedKey = '';
    var provOpts = LLM_PROVIDERS.map(function (p) {
      return '<option value="' + p.key + '"' + (savedProv === p.key ? ' selected' : '') + '>' + esc(p.name) + '</option>';
    }).join('');
    var wrap = d.createElement('div');
    wrap.id = 'apiConfigModal';
    wrap.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:1200;display:flex;align-items:center;justify-content:center';
    wrap.innerHTML =
      '<div style="background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:20px 24px;max-width:520px;width:92%;box-shadow:0 12px 40px rgba(0,0,0,.25)">' +
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">' +
      '<h3 style="margin:0;font-size:16px">🔑 API 配置</h3>' +
      '<button id="apiCfgClose" class="btn ghost" style="padding:4px 10px">✕</button></div>' +
      '<p style="font-size:12px;color:var(--muted);margin:0 0 12px">选择服务商后自动带出地址和模型，你只需填写 API Key。</p>' +
      '<div style="margin-bottom:10px"><label style="font-size:13px;display:block;margin-bottom:4px">服务商</label>' +
      '<select id="apiProvider" style="width:100%;padding:9px 12px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink);font-size:13px;box-sizing:border-box">' + provOpts + '</select></div>' +
      '<div style="margin-bottom:10px"><label style="font-size:13px;display:block;margin-bottom:4px">API 地址</label>' +
      '<input id="apiUrl" type="text" placeholder="https://api.deepseek.com" style="width:100%;padding:9px 12px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink);font-size:13px;box-sizing:border-box" /></div>' +
      '<div style="margin-bottom:10px"><label style="font-size:13px;display:block;margin-bottom:4px">模型名称</label>' +
      '<input id="apiModel" type="text" placeholder="deepseek-v4-flash" style="width:100%;padding:9px 12px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink);font-size:13px;box-sizing:border-box" /></div>' +
      '<div style="margin-bottom:14px"><label style="font-size:13px;display:block;margin-bottom:4px">API Key（必填）</label>' +
      '<input id="apiKey" type="password" placeholder="sk-..." value="' + escAttr(savedKey) + '" style="width:100%;padding:9px 12px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink);font-size:13px;box-sizing:border-box" /></div>' +
      '<div style="display:flex;gap:10px">' +
      '<button class="btn primary" id="saveApiConfig" style="flex:1">保存配置</button>' +
      '<button class="btn ghost" id="testApiConfig">测试连接</button></div>' +
      '<div id="apiConfigStatus" style="font-size:12px;color:var(--muted);margin-top:10px"></div></div>';
    d.body.appendChild(wrap);

    function curProvider() {
      var sel = g('apiProvider');
      var k = sel ? sel.value : 'deepseek';
      for (var i = 0; i < LLM_PROVIDERS.length; i++) if (LLM_PROVIDERS[i].key === k) return LLM_PROVIDERS[i];
      return LLM_PROVIDERS[0];
    }
    function fillFromProvider(p) {
      var u = g('apiUrl'), m = g('apiModel');
      if (u) u.value = p.url;
      if (m) m.value = p.model;
    }
    fillFromProvider(curProvider());

    function close() { var w = g('apiConfigModal'); if (w) w.remove(); }
    var cl = g('apiCfgClose');
    if (cl) cl.addEventListener('click', close);
    wrap.addEventListener('click', function (e) { if (e.target === wrap) close(); });

    var provSel = g('apiProvider');
    if (provSel) provSel.addEventListener('change', function () { fillFromProvider(curProvider()); });

    function readForm() {
      var p = curProvider();
      return {
        provider: p.key,
        api_key: (g('apiKey') || {}).value || '',
        base_url: (g('apiUrl') || {}).value || p.url,
        model: (g('apiModel') || {}).value || p.model,
      };
    }

    // 保存 API 配置 → 后端运行时生效
    var saveApi = g('saveApiConfig');
    if (saveApi) saveApi.addEventListener('click', function () {
      var form = readForm();
      if (!form.api_key) { toast('请输入 API Key'); return; }
      var st = g('apiConfigStatus');
      if (st) st.innerHTML = '正在保存…';
      apiReq('POST', '/api/llm/config', form).then(function (r) {
        localStorage.setItem('dy3_api_provider', form.provider);
        localStorage.setItem('dy3_api_model', form.model);
        localStorage.removeItem('dy3_api_key');
        if (st) st.innerHTML = '<span style="color:var(--success)">✓ 已保存并生效（' + esc(r.masked_key || '') + '）</span>';
        toast('API 配置已保存');
      }).catch(function (e) {
        if (st) st.innerHTML = '<span style="color:var(--danger)">✗ 保存失败: ' + esc(e.message) + '</span>';
      });
    });

    // 测试连接
    var testApi = g('testApiConfig');
    if (testApi) testApi.addEventListener('click', function () {
      var form = readForm();
      if (!form.api_key) { toast('请先填写 API Key'); return; }
      var st = g('apiConfigStatus');
      if (st) st.innerHTML = '正在测试连接…';
      fetch(form.base_url.replace(/\/$/, '') + '/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + form.api_key },
        body: JSON.stringify({ model: form.model, messages: [{ role: 'user', content: 'ping' }], max_tokens: 5 })
      }).then(function (r) {
        if (r.ok) {
          if (st) st.innerHTML = '<span style="color:var(--success)">✓ 连接成功，API 响应正常</span>';
          toast('API 连接成功');
        } else {
          r.text().then(function (t) {
            if (st) st.innerHTML = '<span style="color:var(--danger)">✗ 连接失败: HTTP ' + r.status + ' ' + esc(t.slice(0, 100)) + '</span>';
          });
        }
      }).catch(function (e) {
        if (st) st.innerHTML = '<span style="color:var(--danger)">✗ 连接失败: ' + esc(e.message) + '</span>';
      });
    });
  }

  /* ---------- 学习对比视图 (管理者) ---------- */
  function renderCompare() {
    var ct = g('content');
    if (!ct) return;
    var learners = ['DY20240001', 'DY20240002'];
    Promise.all(learners.map(function (lid) {
      return apiReq('GET', '/l2/profile/' + lid).catch(function () { return null; });
    })).then(function (profiles) {
      var cards = profiles.map(function (p, i) {
        if (!p) return '<div class="stat-card"><div class="lbl">' + esc(learners[i]) + '</div><div class="num" style="font-size:13px">' + t('no_learners') + '</div></div>';
        var mastery = p.overall_mastery != null ? p.overall_mastery : (p.mastery != null ? p.mastery : null);
        var weak = (p.weak_points || []).length;
        return '<div class="card"><h4>' + esc(learners[i]) + '</h4>' +
          '<div class="grid cols-2">' +
          '<div class="stat-card"><div class="lbl">' + t('mastery') + '</div><div class="num">' + (mastery != null ? fmtPct(mastery) : '-') + '</div></div>' +
          '<div class="stat-card"><div class="lbl">' + t('weak_points') + '</div><div class="num">' + weak + '</div></div>' +
          '</div></div>';
      }).join('');
      ct.innerHTML = '<div class="card"><h3>' + t('compare') + '</h3><div class="grid cols-2">' + cards + '</div></div>';
    }).catch(function (e) {
      ct.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  /* ---------- 时间旅行视图 (快照时间线) ---------- */
  function renderTimeTravel(target) {
    var ct = target || g('content');
    if (!ct) return;
    apiReq('GET', '/l2/profile/' + learnerId()).then(function (p) {
      var history = (p && p.history) || [];
      var km = (p && p.kp_mastery) || {};
      var kps = Object.keys(km);
      var overall = (p && p.overall_mastery != null) ? p.overall_mastery
        : (kps.length ? kps.map(function (k) { return km[k] || 0; }).reduce(function (a, b) { return a + b; }, 0) / kps.length : 0);
      
      // 成长轨迹 SVG 迷你折线图 (如果有历史数据)
      var growthChart = '';
      if (history.length >= 2) {
        var pts = history.map(function (h, i) {
          var v = h.mastery != null ? h.mastery : 0;
          var x = 40 + i * (220 / Math.max(history.length - 1, 1));
          var y = 120 - v * 100;
          return x.toFixed(0) + ',' + y.toFixed(0);
        }).join(' ');
        var labels = history.map(function (h, i) {
          var v = h.mastery != null ? h.mastery : 0;
          var x = 40 + i * (220 / Math.max(history.length - 1, 1));
          var y = 120 - v * 100 + 18;
          var ts = h.timestamp || h.time || '';
          var dateStr = ts ? new Date((ts > 1e12 ? ts/1000 : ts)*1000).toLocaleDateString('zh-CN', {month:'short',day:'numeric'}) : '';
          return '<text x="' + x.toFixed(0) + '" y="140" text-anchor="middle" font-size="9" fill="var(--muted)">' + esc(dateStr) + '</text>' +
            '<circle cx="' + x.toFixed(0) + '" cy="' + (120 - v*100).toFixed(0) + '" r="4" fill="var(--accent)" stroke="#fff" stroke-width="1.5"/>';
        }).join('');
        growthChart = '<div style="margin:12px 0;text-align:center"><svg viewBox="0 0 260 150" width="100%" style="max-width:500px">' +
          '<rect x="0" y="0" width="260" height="150" fill="var(--surface2)" rx="8"/>' +
          '<polyline points="' + pts + '" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>' +
          labels +
          '<text x="250" y="16" text-anchor="end" font-size="10" fill="var(--muted)">掌握度趋势</text>' +
          '</svg></div>';
      }
      
      // 当前掌握度概览
      var current = fmtPct2(overall);
      var weakCount = (p && p.weak_kps) ? p.weak_kps.length : 0;
      var strongCount = kps.filter(function (k) { return km[k] >= 0.7; }).length;
      var learningDays = history.length;
      
      // 历史快照列表
      var tl = history.length ? history.map(function (h) {
        return '<div class="debate-speech" style="justify-content:space-between;align-items:center">' +
          '<span style="font-size:12px">' + esc(h.timestamp || h.time || '') + '</span>' +
          '<span class="badge info">' + fmtPct(h.mastery != null ? h.mastery : 0) + '</span></div>';
      }).join('') : '<p style="color:var(--muted);text-align:center;padding:20px">暂无历史记录 — 开始学习后将自动记录成长轨迹</p>';
      
      ct.innerHTML = '<div class="card"><h3>📈 学习轨迹 · 成长展示</h3>' +
        '<div class="stat-strip">' +
        '<div class="stat-card"><div class="num">' + current + '</div><div class="lbl">当前掌握度</div></div>' +
        '<div class="stat-card"><div class="num">' + strongCount + '</div><div class="lbl">已掌握知识点</div></div>' +
        '<div class="stat-card"><div class="num" style="color:' + (weakCount > 0 ? 'var(--warn)' : 'var(--success)') + '">' + weakCount + '</div><div class="lbl">薄弱知识点</div></div>' +
        '<div class="stat-card"><div class="num">' + (p && p.level || '-') + '</div><div class="lbl">当前等级</div></div>' +
        '</div>' +
        (growthChart || '<div class="callout" style="margin-bottom:12px">成长图表需要至少2次学习记录</div>') +
        '<h4 style="margin:12px 0 8px">📋 历史快照 (' + learningDays + ' 次记录)</h4>' +
        '<div class="debate-timeline" style="max-height:300px">' + tl + '</div></div>';
    }).catch(function (e) {
      ct.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  /* ---------- M-F7 缺口补齐: BKT 热力图 / 学情环形图 / 溯源徽章 ---------- */

  // 42 KP 域常量 — 单点收敛: 从 L2 /l2/kp-catalog (SSOT) 拉取, 不再前端硬编码
  var KP_DOMAINS = [
    { code: 'A', label: '能级跃迁理论', ids: [] },
    { code: 'B', label: '材料体系设计', ids: [] },
    { code: 'C', label: '合成制备工艺', ids: [] },
    { code: 'D', label: '表征测试技术', ids: [] },
  ];
  var KP_CATALOG_LOADED = false;
  var KP_NAME_MAP = {};
  // 兜底: 若 catalog 不可用则按域数量生成 (与 L2 kp_catalog 一致)
  (function () {
    var counts = { A: 13, B: 11, C: 10, D: 8 };
    for (var di = 0; di < KP_DOMAINS.length; di++) {
      var kd = KP_DOMAINS[di];
      var count = counts[kd.code] || 8;
      for (var n = 1; n <= count; n++) kd.ids.push(kd.code + '-' + (n < 10 ? '0' : '') + n);
    }
  })();
  function loadKpCatalog() {
    if (KP_CATALOG_LOADED) return Promise.resolve(KP_DOMAINS);
    return apiReq('GET', '/l2/kp-catalog').then(function (cat) {
      var data = (cat && cat.data) || cat || {};
      var domains = data.domains || [];
      var kpList = data.kp || [];
      if (domains.length && kpList.length) {
        KP_DOMAINS = domains.map(function (d) { return { code: d.code, label: d.label, ids: (d.kp_ids || []).slice() }; });
        kpList.forEach(function (item) { KP_NAME_MAP[item.kp_id] = item.name || item.kp_id; });
      }
      KP_CATALOG_LOADED = true;
      return KP_DOMAINS;
    }).catch(function () { KP_CATALOG_LOADED = true; return KP_DOMAINS; });
  }
  function kpName(id) { return KP_NAME_MAP[id] || id; }

  function learnerId() {
    return window.dy3LearnerId ? window.dy3LearnerId() : (localStorage.getItem('dl') || 'guest-unavailable');
  }

  function masteryColor(v) {
    if (v >= 0.7) return 'var(--success)';
    if (v >= 0.4) return 'var(--warn)';
    return 'var(--danger)';
  }

  function fmtPct2(v) { return Math.round((v || 0) * 100) + '%'; }

  // BKT "收起全部域" 折叠态持久化 (跨 视图重渲染 保留, 避免收起几秒后失效)
  var _bktAllCollapsed = false;

  // 1. BKT 热力图 (learn-mastery)
  function renderBktHeatmap(target) {
    var ct = target || g('content');
    if (!ct) return;
    apiReq('GET', '/l2/profile/' + learnerId()).then(function (p) {
      var km = (p && p.kp_mastery) || {};
      var names = (p && p.kp_names) || {};
      var kps = Object.keys(km);
      var vals = kps.map(function (k) { return km[k] || 0; });
      var avg = vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : 0;
      var weak = (p && p.weak_kps) || kps.filter(function (k) { return km[k] < 0.6; });

      // KP 详情弹层 (点击任意 KP 卡片, 通过事件代理触发)
      function kpDetailCard(id) {
        var v = km[id] || 0;
        var isWeak = weak.indexOf(id) !== -1;
        var name = names[id] || id;
        return '<div style="position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99;display:flex;align-items:center;justify-content:center" id="kpModalWrap">' +
          '<div style="background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:20px 24px;max-width:420px;width:92%;box-shadow:0 12px 40px rgba(0,0,0,.25)">' +
          '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px"><div style="font-weight:700;font-size:16px">' + esc(id) + ' · ' + esc(name) + '</div>' +
          '<button id="kpClose" class="btn ghost" style="padding:4px 10px">✕</button></div>' +
          '<div style="font-size:13px;line-height:1.7">' +
          '<div>掌握度: <b style="color:' + masteryColor(v) + '">' + fmtPct2(v) + '</b></div>' +
          '<div>状态: ' + (isWeak ? '<span style="color:var(--danger)">薄弱知识点（建议优先练习）</span>' : '<span style="color:var(--success)">已掌握</span>') + '</div>' +
          '<div style="color:var(--muted);margin-top:8px">所属领域: ' + esc(kpDomainLabel(id)) + '</div>' +
          '<div style="color:var(--muted);margin-top:4px">该知识点可通过「今日推荐 / 学习轨迹」中的练习或考核提升。</div></div></div></div>';
      }
      // 事件代理: 点击 data-kp-id 元素触发弹层
      function setupKpClickProxy() {
        var ct = g('content');
        if (!ct || ct._kpProxy) return;
        ct._kpProxy = true;
        ct.addEventListener('click', function (e) {
          var cell = e.target.closest('[data-kp-id]');
          if (!cell) return;
          var id = cell.getAttribute('data-kp-id');
          if (!id) return;
          var wrap = g('kpModalWrap');
          if (wrap) wrap.remove();
          var holder = document.createElement('div');
          holder.innerHTML = kpDetailCard(id);
          document.body.appendChild(holder.firstChild);
          var c = g('kpClose');
          if (c) c.addEventListener('click', function () { var w = g('kpModalWrap'); if (w) w.remove(); });
          g('kpModalWrap').addEventListener('click', function (ev) { if (ev.target === this) this.remove(); });
        });
      }

      var domainBlocks = KP_DOMAINS.map(function (kd) {
        var cells = kd.ids.map(function (kpid) {
          var v = km[kpid] || 0;
          var weakTag = weak.indexOf(kpid) !== -1 ? '<span class="badge err" style="margin-left:6px;padding:0 5px;font-size:10.5px">薄弱</span>' : '';
          return '<div class="kp-cell" style="border-left:3px solid ' + masteryColor(v) + ';cursor:pointer" data-kp-id="' + escAttr(kpid) + '" title="' + esc(kpName(kpid)) + ' - 掌握度 ' + fmtPct2(v) + '">' +
            '<span class="kp-id">' + kpid + '</span>' +
            '<span class="kp-name">' + esc(kpName(kpid)) + '</span>' + weakTag +
            '<div style="display:flex;align-items:center;gap:6px">' +
            '<div class="mastery-bar" style="flex:1"><div class="mastery-fill ' + (v >= 0.7 ? 'high' : (v >= 0.4 ? 'mid' : 'low')) + '" style="width:' + Math.round(v * 100) + '%"></div></div>' +
            '<span style="font-family:var(--mono);font-size:10px;font-weight:600;color:' + masteryColor(v) + ';min-width:32px;text-align:right">' + fmtPct2(v) + '</span></div></div>';
        });
        var domVals = kd.ids.map(function (k) { return km[k] || 0; });
        var domAvg = domVals.length ? domVals.reduce(function (a, b) { return a + b; }, 0) / domVals.length : 0;
        var domainId = 'domain_' + kd.code;
        var showCount = 6;
        var visibleCells = cells.slice(0, showCount).join('');
        var hiddenCells = cells.slice(showCount).join('');
        var hasMore = cells.length > showCount;
        return '<div class="domain-card domain-' + kd.code + '">' +
          '<div style="display:flex;justify-content:space-between;align-items:center">' +
          '<span class="domain-label">' + esc(kd.label) + ' (' + kd.code + ')</span>' +
          '<span class="domain-avg" style="font-size:18px">' + fmtPct2(domAvg) + '</span></div>' +
          '<div style="display:flex;align-items:center;gap:8px">' +
          '<div class="mastery-bar" style="flex:1"><div class="mastery-fill ' + (domAvg >= 0.7 ? 'high' : (domAvg >= 0.4 ? 'mid' : 'low')) + '" style="width:' + Math.round(domAvg * 100) + '%"></div></div>' +
          '<span class="domain-counts">' + kd.ids.length + ' KP</span></div>' +
          '<div class="kp-grid" style="margin-top:6px">' + visibleCells + '</div>' +
          (hasMore ? '<div id="' + domainId + '_more" style="display:none" class="kp-grid">' + hiddenCells + '</div>' : '') +
          (hasMore ? '<button class="btn ghost sm" id="' + domainId + '_toggle" style="margin-top:6px;width:100%" onclick="var m=document.getElementById(\'' + domainId + '_more\');var b=this;if(m.style.display===\'none\'){m.style.display=\'grid\';b.textContent=\'收起\';}else{m.style.display=\'none\';b.textContent=\'展开全部 ' + kd.ids.length + ' 个\';}">展开全部 ' + kd.ids.length + ' 个</button>' : '') +
          '</div>';
      }).join('');

      ct.innerHTML =
        '<div class="card"><h3>BKT 掌握度热力图 (42 KP · 点击卡片查看详情)</h3>' +
        '<div class="stat-strip">' +
        '<div class="stat-card"><div class="lbl">平均掌握度</div><div class="num">' + fmtPct2(avg) + '</div></div>' +
        '<div class="stat-card"><div class="lbl">能力值 θ</div><div class="num">' + (p && p.theta != null && p.theta > 0 ? Number(p.theta).toFixed(2) : (avg > 0 ? '≈' + fmtPct2(avg) : '-')) + '</div></div>' +
        '<div class="stat-card"><div class="lbl">薄弱点</div><div class="num">' + weak.length + '</div></div>' +
        '<div class="stat-card"><div class="lbl">画像置信度</div><div class="num">' + (p && p.confidence != null ? fmtPct2(p.confidence) : '-') + '</div></div>' +
        '</div><div class="callout" style="margin-bottom:12px">色标: <span style="color:var(--danger)">■ 未掌握 (&lt;40%)</span> · <span style="color:var(--warn)">■ 发展中 (40-70%)</span> · <span style="color:var(--success)">■ 已掌握 (≥70%)</span> · 点击 KP 卡片查看详情</div>' +
        '<button class="btn ghost" id="toggleBktAll" style="margin-bottom:8px">收起全部域</button>' +
        '<div id="bktDomains">' + domainBlocks + '</div></div>';
      setupKpClickProxy();
      var toggleBtn = g('toggleBktAll');
      if (toggleBtn) {
        // 重渲染后恢复上次折叠态
        if (_bktAllCollapsed) {
          var d0 = g('bktDomains');
          if (d0) d0.classList.add('collapsed');
          toggleBtn.textContent = '展开全部域';
        }
        toggleBtn.addEventListener('click', function() {
          var domains = g('bktDomains');
          if (!domains) return;
          _bktAllCollapsed = domains.classList.toggle('collapsed');
          this.textContent = _bktAllCollapsed ? '展开全部域' : '收起全部域';
        });
      }
    }).catch(function (e) {
      ct.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  // KP 所属领域中文名
  function kpDomainLabel(kpid) {
    var d = KP_DOMAINS.find(function (kd) { return kd.ids.indexOf(kpid) !== -1; });
    return d ? (d.label + ' (' + d.code + ')') : '-';
  }

  // 2. 学情环形图 (learn)
  function renderLearnRing() {
    var ct = g('content');
    if (!ct) return;
    apiReq('GET', '/l2/profile/' + learnerId()).then(function (p) {
      var km = (p && p.kp_mastery) || {};
      var kps = Object.keys(km);
      var vals = kps.map(function (k) { return km[k] || 0; });
      var avg = vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : 0;

      function ring(percent) {
        var r = 54, c = 2 * Math.PI * r;
        var dash = c * Math.min(1, Math.max(0, percent));
        var color = percent >= 0.7 ? 'var(--success)' : (percent >= 0.4 ? 'var(--warn)' : 'var(--danger)');
        return '<svg viewBox="0 0 140 140" width="150" height="150" style="display:block;margin:0 auto" role="img" aria-label="总体掌握度 ' + Math.round(percent * 100) + '%">' +
          '<circle cx="70" cy="70" r="' + r + '" fill="none" stroke="var(--rule)" stroke-width="12"/>' +
          '<circle cx="70" cy="70" r="' + r + '" fill="none" stroke-width="12" stroke-linecap="round" stroke-dasharray="' + dash.toFixed(1) + ' ' + c.toFixed(1) + '" transform="rotate(-90 70 70)" style="stroke:' + color + ';transition:stroke-dasharray .8s ease"/>' +
          '<text x="70" y="76" text-anchor="middle" font-size="24" font-weight="600" style="fill:' + color + '">' + Math.round(percent * 100) + '%</text>' +
          '</svg>';
      }

      // 域进度条 (带颜色渐变)
      function domainBar(label, code, da, strong, weakN) {
        var pct = Math.round((da || 0) * 100);
        var color = pct >= 70 ? 'var(--success)' : (pct >= 40 ? 'var(--warn)' : 'var(--danger)');
        return '<div style="margin-bottom:10px;cursor:pointer" data-kp-domain="' + escAttr(code) + '" title="点击查看该领域知识点详情">' +
          '<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">' +
          '<span style="color:var(--muted)">' + esc(label) + ' <span style="font-size:10.5px;opacity:.75">(' + strong + '掌握 · ' + weakN + '薄弱)</span></span>' +
          '<span style="font-family:var(--mono);font-weight:600;color:' + color + '">' + pct + '%</span></div>' +
          '<div style="height:8px;border-radius:4px;background:var(--surface2);overflow:hidden"><div style="height:100%;width:' + pct + '%;border-radius:4px;background:' + color + ';transition:width .6s ease"></div></div></div>';
      }

      var domainBars = KP_DOMAINS.map(function (kd) {
        var dv = kd.ids.map(function (k) { return km[k] || 0; });
        var da = dv.length ? dv.reduce(function (a, b) { return a + b; }, 0) / dv.length : 0;
        var strong = dv.filter(function (x) { return x >= 0.7; }).length;
        var weakN = dv.filter(function (x) { return x < 0.4; }).length;
        return domainBar(kd.label, kd.code, da, strong, weakN);
      }).join('');

      // 领域详情弹层 (通过事件代理触发)
      function showKpDomain(did) {
        var d = KP_DOMAINS.find(function (kd) { return kd.code === did || kd.id === did; });
        if (!d) return;
        var rows = d.ids.map(function (kpid) {
          var v = km[kpid] || 0;
          var c = v >= 0.7 ? 'var(--success)' : (v >= 0.4 ? 'var(--warn)' : 'var(--danger)');
          var barW = Math.round(v * 100);
          return '<div style="padding:7px 4px;border-bottom:1px solid var(--rule)">' +
            '<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">' +
            '<span>' + esc(kpid) + ' · ' + esc((p && p.kp_names && p.kp_names[kpid]) || kpid) + '</span>' +
            '<span style="font-family:var(--mono);color:' + c + ';font-weight:600">' + barW + '%</span></div>' +
            '<div style="height:6px;border-radius:3px;background:var(--surface2);overflow:hidden"><div style="height:100%;width:' + barW + '%;border-radius:3px;background:' + c + '"></div></div></div>';
        }).join('');
        var wrap = document.createElement('div');
        wrap.innerHTML = '<div style="position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99;display:flex;align-items:center;justify-content:center" id="kpDomainModal">' +
          '<div style="background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:18px 22px;max-width:460px;width:92%;max-height:80vh;overflow:auto;box-shadow:0 12px 40px rgba(0,0,0,.25)">' +
          '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px"><div style="font-weight:700;font-size:16px">' + esc(d.label) + ' (' + d.code + ') · ' + d.ids.length + ' 个知识点</div>' +
          '<button id="kpDomainClose" class="btn ghost" style="padding:4px 10px">✕</button></div>' + rows + '</div></div>';
        document.body.appendChild(wrap.firstChild);
        var c = g('kpDomainClose');
        if (c) c.addEventListener('click', function () { var w = g('kpDomainModal'); if (w) w.remove(); });
        g('kpDomainModal').addEventListener('click', function (e) { if (e.target === this) this.remove(); });
      }

      ct.innerHTML = '<div class="card"><h3>学情总览</h3>' +
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start">' +
        '<div class="stat-card" style="text-align:center;padding:12px">' + ring(avg) + '<div class="lbl" style="margin-top:6px;font-weight:600">总体掌握度</div></div>' +
        '<div style="display:flex;flex-direction:column;gap:8px;padding:8px">' +
        '<div style="display:flex;justify-content:space-between;padding:6px 8px;background:var(--surface2);border-radius:8px"><span style="color:var(--muted)">能力值 θ</span><span style="font-weight:600">' + (p && p.theta != null && p.theta > 0 ? Number(p.theta).toFixed(2) : (avg > 0 ? '≈' + fmtPct2(avg) : '-')) + '</span></div>' +
        '<div style="display:flex;justify-content:space-between;padding:6px 8px;background:var(--surface2);border-radius:8px"><span style="color:var(--muted)">学习等级</span><span style="font-weight:600">' + esc((p && p.level) || '-') + '</span></div>' +
        '<div style="display:flex;justify-content:space-between;padding:6px 8px;background:var(--surface2);border-radius:8px"><span style="color:var(--muted)">画像置信度</span><span style="font-weight:600">' + (p && p.confidence != null ? fmtPct2(p.confidence) : '-') + '</span></div>' +
        '</div></div><h4 style="margin:16px 0 10px;font-size:14px">📊 四域掌握度（点击查看详情）</h4>' + domainBars + '</div>';
      // 事件代理: 点击 data-kp-domain 元素触发领域详情
      var ct2 = g('content');
      if (ct2 && !ct2._kpDomainProxy) {
        ct2._kpDomainProxy = true;
        ct2.addEventListener('click', function (ev) {
          var bar = ev.target.closest('[data-kp-domain]');
          if (!bar) return;
          var code = bar.getAttribute('data-kp-domain');
          if (code) showKpDomain(code);
        });
      }
    }).catch(function (e) {
      ct.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  // 2b. 游戏属性面板 (雷达图, 百分制) — 像游戏角色属性面板
  /* 多智能体协同决策概览卡片 (竞赛要求: 协同调度过程可视化) */
  function renderCollabCard() {
    return '<div class="card" style="margin-top:16px"><div class="r04-section-kicker">任务协同</div>' +
      '<h3 style="margin:0 0 6px">围绕一个 Dy³⁺ 学习任务协同</h3>' +
      '<p style="font-size:12px;color:var(--muted);margin:0 0 12px;line-height:1.65">学情诊断确定解释层级，知识生成使用领域证据，质量审核挑战不可靠结论，导学决策给出下一步。协同过程仅在任务完成后按真实记录展示。</p>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn primary" data-goto="query">发起科研学习任务</button>' +
      '<button class="btn ghost" data-goto="agents-chain">了解真实协同</button></div></div>';
  }

  /* 知识盲区速览卡片 (竞赛要求: 盲区定位可视化) */
  function renderBlindSpotCard(p) {
    var km = (p && p.kp_mastery) || {};
    var blind = [];
    Object.keys(km).forEach(function (k) {
      var v = km[k] || 0;
      if (v > 0 && v < 0.5) blind.push({ kp: k, v: v });
    });
    blind.sort(function (a, b) { return a.v - b.v; });
    if (!blind.length) return '';
    var items = blind.slice(0, 6).map(function (b) {
      return '<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--rule)">' +
        '<span style="font-size:11px;color:var(--muted);flex:none;width:44px;font-family:var(--mono)">' + esc(b.kp) + '</span>' +
        '<div class="mastery-bar" style="height:6px"><div class="mastery-fill low" style="width:' + Math.round(b.v * 100) + '%"></div></div>' +
        '<span style="font-size:11px;font-weight:600;color:#d97706;flex:none">' + Math.round(b.v * 100) + '%</span>' +
        '</div>';
    }).join('');
    return '<div class="card" style="margin-top:16px"><h3 style="margin:0 0 10px">📍 知识盲区速览 <span class="hint">(' + blind.length + ' 个薄弱点)</span></h3>' +
      items +
      '<button class="btn ghost sm" style="margin-top:10px" data-goto="match-report">查看完整匹配报告 →</button></div>';
  }

  function renderGamePanel(p) {
    var km = (p && p.kp_mastery) || {};
    var dims = (p && p.dimensions) || {};
    // 5 轴: A 理论 / B 应用 / C 合成 / D 表征 / E 行为 (优先后端 dimensions, 缺省前端聚合)
    var axes = [
      { key: 'A', label: '理论' },
      { key: 'B', label: '应用' },
      { key: 'C', label: '合成' },
      { key: 'D', label: '表征' },
      { key: 'E', label: '行为' },
    ];
    var vals = axes.map(function (a) {
      if (dims[a.key] != null) return Math.max(0, Math.min(1, dims[a.key]));
      var ids = (KP_DOMAINS.find(function (k) { return k.code === a.key; }) || {}).ids || [];
      var dv = ids.map(function (k) { return km[k] || 0; });
      return dv.length ? dv.reduce(function (x, y) { return x + y; }, 0) / dv.length : 0;
    });
    var overall = (p && p.overall_mastery != null) ? p.overall_mastery
      : (function () { var v = Object.keys(km).map(function (k) { return km[k] || 0; }); return v.length ? v.reduce(function (a, b) { return a + b; }, 0) / v.length : 0; })();
    var cx = 170, cy = 170, R = 110;
    // 六边形网格 (3 层) + 数据多边形
    var grid = '';
    for (var g = 1; g <= 3; g++) {
      var pts = [];
      for (var i = 0; i < axes.length; i++) {
        var ang = -Math.PI / 2 + (Math.PI * 2 * i) / axes.length;
        pts.push((cx + Math.cos(ang) * R * g / 3).toFixed(1) + ',' + (cy + Math.sin(ang) * R * g / 3).toFixed(1));
      }
      grid += '<polygon points="' + pts.join(' ') + '" fill="none" stroke="var(--rule)" stroke-width="1"/>';
    }
    var dataPts = [];
    for (var j = 0; j < axes.length; j++) {
      var a2 = -Math.PI / 2 + (Math.PI * 2 * j) / axes.length;
      dataPts.push((cx + Math.cos(a2) * R * vals[j]).toFixed(1) + ',' + (cy + Math.sin(a2) * R * vals[j]).toFixed(1));
    }
    var axisLines = axes.map(function (a, i2) {
      var a3 = -Math.PI / 2 + (Math.PI * 2 * i2) / axes.length;
      var ex = cx + Math.cos(a3) * R, ey = cy + Math.sin(a3) * R;
      var lx = cx + Math.cos(a3) * (R + 28), ly = cy + Math.sin(a3) * (R + 28);
      var vx = cx + Math.cos(a3) * R * vals[i2], vy = cy + Math.sin(a3) * R * vals[i2];
      var color = vals[i2] >= 0.7 ? 'var(--success)' : (vals[i2] >= 0.4 ? 'var(--warn)' : 'var(--danger)');
      return '<line x1="' + cx + '" y1="' + cy + '" x2="' + ex + '" y2="' + ey + '" stroke="var(--rule)" stroke-width="1"/>' +
        '<circle cx="' + vx.toFixed(1) + '" cy="' + vy.toFixed(1) + '" r="4" fill="' + color + '" stroke="#fff" stroke-width="1.5"/>' +
        '<text x="' + lx.toFixed(1) + '" y="' + ly.toFixed(1) + '" text-anchor="middle" font-size="12" fill="var(--ink)">' + esc(a.label) + ' ' + Math.round(vals[i2] * 100) + '</text>';
    }).join('');
    return '<div class="card" data-game-panel>' +
      '<h3>📊 掌握度面板</h3>' +
      '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:16px">' +
      '<svg viewBox="0 0 340 340" width="240" height="240" style="overflow:visible" role="img" aria-label="掌握度雷达图">' +
      grid + axisLines +
      '<polygon points="' + dataPts.join(' ') + '" fill="rgba(42,90,223,0.25)" stroke="#2a5adf" stroke-width="2"/>' +
      '<text x="170" y="168" text-anchor="middle" font-size="26" font-weight="700" fill="var(--ink)">' + Math.round(overall * 100) + '</text>' +
      '<text x="170" y="188" text-anchor="middle" font-size="10.5" fill="var(--muted)">总体掌握度</text>' +
      '</svg>' +
      '<div style="flex:1;min-width:180px">' +
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">' +
      '<div style="background:var(--surface2);border-radius:8px;padding:8px 10px;text-align:center">' +
      '<div style="font-size:10.5px;color:var(--muted);margin-bottom:2px">总体掌握度</div>' +
      '<div style="font-size:20px;font-weight:700;font-family:var(--mono);color:' + (overall >= 0.7 ? 'var(--success)' : (overall >= 0.4 ? 'var(--warn)' : 'var(--danger)')) + '">' + Math.round(overall * 100) + '</div>' +
      '<div class="mastery-bar" style="margin-top:4px"><div class="mastery-fill ' + (overall >= 0.7 ? 'high' : (overall >= 0.4 ? 'mid' : 'low')) + '" style="width:' + Math.round(overall * 100) + '%"></div></div></div>' +
      '<div style="background:var(--surface2);border-radius:8px;padding:8px 10px;text-align:center">' +
      '<div style="font-size:10.5px;color:var(--muted);margin-bottom:2px">能力值 θ</div>' +
      '<div style="font-size:16px;font-weight:600;font-family:var(--mono);color:var(--accent)">' + (p && p.theta != null ? Number(p.theta).toFixed(2) : '-') + '</div></div>' +
      '</div>' +
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">' +
      '<div style="background:var(--surface2);border-radius:8px;padding:8px 10px;text-align:center">' +
      '<div style="font-size:10.5px;color:var(--muted);margin-bottom:2px">学习等级</div>' +
      '<div style="font-size:15px;font-weight:600;color:var(--ink)">' + esc((p && p.level) || (p && p.initial_assessed === false ? '未初测' : '-')) + '</div></div>' +
      '<div style="background:var(--surface2);border-radius:8px;padding:8px 10px;text-align:center">' +
      '<div style="font-size:10.5px;color:var(--muted);margin-bottom:2px">画像置信度</div>' +
      '<div style="font-size:15px;font-weight:600;font-family:var(--mono);color:' + ((p && p.confidence != null && p.confidence >= 0.7) ? 'var(--success)' : 'var(--warn)') + '">' + (p && p.confidence != null ? fmtPct2(p.confidence) : '-') + '</div></div>' +
      '</div></div>' +
      '</div></div>';
  }

  function workspaceActionLabel(decision) {
    return {
      DIAGNOSE_FIRST: '先用真实作答认识你',
      LOW: '先补齐基础与先修概念',
      MAINTAIN: '保持当前挑战强度',
      RAISE: '可以进入更高阶挑战'
    }[decision] || '继续积累真实学习证据';
  }

  function bindWorkspaceActions(root) {
    if (!root) return;
    root.querySelectorAll('[data-ws-route]').forEach(function (button) {
      button.addEventListener('click', function () {
        var route = button.getAttribute('data-ws-route') || '';
        var conceptId = button.getAttribute('data-concept-id') || '';
        var kpIds = (button.getAttribute('data-kp-ids') || '').split(',').filter(Boolean);
        if (route === 'query') {
          var question = button.getAttribute('data-question') || '';
          if (question) sessionStorage.setItem('dy3_pending_query', question);
        }
        if (route === 'practice') {
          sessionStorage.setItem('dy3_practice_target_kps', JSON.stringify(kpIds));
          sessionStorage.setItem('dy3_practice_attempt_purpose', button.getAttribute('data-attempt-purpose') || 'DIAGNOSTIC');
        }
        if (conceptId) sessionStorage.setItem('dy3_workspace_concept_id', conceptId);
        if (route && window.sv) window.sv(route);
      });
    });
  }

  function renderWorkspaceProduct(ct) {
    if (window.DY3ProductCanvas && typeof window.DY3ProductCanvas.renderOverview === 'function') {
      window.DY3ProductCanvas.renderOverview(ct);
      return;
    }
    ct.innerHTML = '<article class="learning-workspace-product canvas-page canvas-overview">' + r08Journey('overview', r08StoredTask()) + '<header class="workspace-action-dock canvas-dashboard-header">' +
      '<div><span class="workspace-kicker">LEARNING OVERVIEW</span><h1>学习总览</h1></div>' +
      '<div class="workspace-ask"><input id="homeTaskInput" type="text" placeholder="提出 Dy³⁺ 发光材料或绿色健康照明问题"><button class="btn primary" id="homeTaskStart">开始学习</button></div>' +
      '<div id="workspaceResume"></div></header>' +
      '<section class="workspace-layer" id="workspaceStage"><div class="r04-processing"><span class="spinner"></span><div><strong>正在读取学习状态</strong></div></div></section>' +
      '<section class="workspace-layer canvas-path-panel" id="workspaceSequence"><div class="workspace-layer-head"><span>02</span><div><h2>学习路径</h2></div></div><div class="workspace-empty">正在生成可执行序列…</div></section>' +
      '<section class="workspace-layer canvas-activity-panel" id="workspaceChanges"><div class="workspace-layer-head"><span>03</span><div><h2>最近活动</h2></div></div><div class="workspace-empty">正在读取真实事件…</div></section>' +
      '</article>';
    r08BindJourney(ct);
    var input = g('homeTaskInput');
    function openTask() {
      var value = String((input && input.value) || '').trim();
      if (value) sessionStorage.setItem('dy3_pending_query', value);
      if (window.sv) window.sv('query');
    }
    var start = g('homeTaskStart');
    if (start) start.addEventListener('click', openTask);
    if (input) input.addEventListener('keydown', function (event) { if (event.key === 'Enter') openTask(); });

    apiReq('GET', '/api/learning-workspace/' + encodeURIComponent(learnerId())).then(function (payload) {
      var ws = (payload && payload.data) || payload || {};
      var stage = g('workspaceStage'), sequence = g('workspaceSequence'), changes = g('workspaceChanges'), resume = g('workspaceResume');
      var summary = ws.learner_summary || {};
      var declared = summary.declared_background || {};
      var focus = ws.current_focus || {};
      var challenge = ws.current_challenge_decision || {};
      var adaptation = ws.teaching_adaptation_summary || {};
      var blockers = Array.isArray(ws.blocking_prerequisites) ? ws.blocking_prerequisites : [];
      var coverage = Array.isArray(ws.capability_coverage) ? ws.capability_coverage : [];
      var quick = Array.isArray(ws.quick_actions) ? ws.quick_actions : [];
      var identityText = ws.identity_scope === 'DEVICE_LOCAL_GUEST' ? '本设备访客 · 跨设备连续性有限' : '已登录学习者';
      var blockerHtml = blockers.length ? blockers.map(function (item) {
        return '<li><strong>' + esc(item.name || item.concept_id) + '</strong><small>真实 prerequisite_of 先修节点</small></li>';
      }).join('') : '<li class="is-clear"><strong>当前未发现阻断先修项</strong><small>不代表已经掌握全部知识</small></li>';
      var actionHtml = quick.filter(function (item) { return item.status === 'AVAILABLE'; }).map(function (item) {
        var ctx = item.context || {};
        var question = item.action_type === 'LEARN_CONCEPT' ? ('请帮助我学习：' + (focus.name || item.target)) : '';
        return '<button class="workspace-action" data-ws-route="' + escAttr(item.route || '') + '" data-concept-id="' + escAttr(ctx.concept_id || '') + '" data-kp-ids="' + escAttr(ctx.kp_ids || '') + '" data-attempt-purpose="' + escAttr(ctx.attempt_purpose || '') + '" data-question="' + escAttr(question) + '"><strong>' + esc(item.label) + '</strong><small>' + esc(item.reason) + '</small></button>';
      }).join('');
      var topicHtml = coverage.slice(0, 8).map(function (item) {
        var practice = item.authored_practice_available ? '有真实练习' : '暂无对应练习';
        var evidence = item.evidence_status === 'RELEASED_TASK_EVIDENCE' ? '有已发布证据' : (item.evidence_status === 'MENTION_CANDIDATE_ONLY' ? '仅术语提及候选' : '无公开证据');
        return '<div class="workspace-topic"><strong>' + esc(item.name) + '</strong><span>' + esc(practice) + '</span><span>' + esc(evidence) + '</span></div>';
      }).join('');
      if (stage) stage.innerHTML = '<div class="workspace-layer-head"><span>01</span><div><h2>当前状态</h2><p>' + esc(identityText) + '</p></div><em>' + esc(summary.evidence_status || 'UNKNOWN') + '</em></div>' +
        '<div class="workspace-stage-grid"><div class="workspace-focus"><span>当前焦点</span><h3>' + esc(focus.name || '尚未确定') + '</h3><p>' + esc(focus.reason || '你仍可直接提问。') + '</p></div>' +
        '<div class="workspace-challenge"><span>当前挑战</span><h3>' + esc(workspaceActionLabel(challenge.decision)) + '</h3><p>' + esc(challenge.reason || '') + '</p></div>' +
        '<div class="workspace-adaptation"><span>回答适配</span><h3>' + esc(adaptation.content_depth || 'foundation') + '</h3><p>' + esc(adaptation.explanation_strategy || 'baseline_explanation') + '</p></div></div>' +
        '<div class="workspace-blockers"><h3>需要先处理的知识</h3><ul>' + blockerHtml + '</ul></div>' +
        '<div class="workspace-actions"><h3>现在可以做什么</h3>' + (actionHtml || '<p>当前仍可直接提问。</p>') + '</div>' +
        (topicHtml ? '<div class="workspace-topics"><h3>当前真实能力覆盖</h3><div>' + topicHtml + '</div></div>' : '') +
        '<details class="workspace-profile-editor"><summary>补充或管理自愿学习背景</summary><p>这些信息只作为低权重声明先验；真实作答、学习事件和模型状态会单独标记。所有字段均可跳过。</p><div class="workspace-profile-fields"><label><span>学习阶段</span><select id="workspaceStageInput"><option value="">不填写</option><option value="undergraduate"' + (declared.learning_stage === 'undergraduate' ? ' selected' : '') + '>本科</option><option value="graduate"' + (declared.learning_stage === 'graduate' ? ' selected' : '') + '>研究生</option><option value="researcher"' + (declared.learning_stage === 'researcher' ? ' selected' : '') + '>科研人员</option></select></label><label><span>专业背景</span><input id="workspaceMajorInput" value="' + escAttr(declared.professional_background || '') + '" placeholder="如：材料、物理、光电"></label><label><span>领域经历</span><select id="workspaceExperienceInput"><option value="">不填写</option><option value="introductory"' + (declared.domain_experience === 'introductory' ? ' selected' : '') + '>刚开始了解</option><option value="coursework"' + (declared.domain_experience === 'coursework' ? ' selected' : '') + '>修过相关课程</option><option value="lab"' + (declared.domain_experience === 'lab' ? ' selected' : '') + '>有实验经历</option><option value="research"' + (declared.domain_experience === 'research' ? ' selected' : '') + '>有科研经历</option></select></label><label><span>学习目标</span><input id="workspaceGoalInput" value="' + escAttr(declared.learning_goal || '') + '" placeholder="如：理解 Dy³⁺ 白光机制"></label></div><div class="workspace-profile-actions"><button class="btn primary" id="workspaceProfileSave">保存自愿信息</button><button class="btn ghost" id="workspaceDiagnostic">开始真实诊断</button><button class="btn danger" id="workspaceProfileClear">清除自愿画像</button></div><div id="workspaceProfileFeedback" class="t1-resource-feedback" aria-live="polite"></div><small>清除范围：用户理解画像。任务记录与真实作答记录属于独立事实，不会被此按钮一并删除。</small></details>';
      var steps = Array.isArray(ws.learning_sequence) ? ws.learning_sequence : [];
      if (sequence) sequence.innerHTML = '<div class="workspace-layer-head"><span>02</span><div><h2>学习路径</h2><p>基于当前 Concept 与先修关系</p></div></div>' +
        (steps.length ? '<ol class="workspace-sequence">' + steps.map(function (step) {
          var action = step.action || {};
          return '<li class="status-' + escAttr(String(step.status || '').toLowerCase()) + '"><span>' + esc(String(step.order)) + '</span><div><strong>' + esc(step.label) + '</strong><p>' + esc(step.reason) + '</p><button class="btn ghost" data-ws-route="' + escAttr(action.route || 'query') + '" data-concept-id="' + escAttr(step.concept_id || '') + '" data-question="' + escAttr('请帮助我学习：' + step.label) + '">' + esc(action.label || '开始') + '</button></div><em>' + esc(step.status) + '</em></li>';
        }).join('') + '</ol>' : '<div class="workspace-empty">还没有足够事实形成学习序列。你可以直接提问或进入真实诊断。</div>');
      var recent = Array.isArray(ws.recent_changes) ? ws.recent_changes : [];
      if (changes) changes.innerHTML = '<div class="workspace-layer-head"><span>03</span><div><h2>最近活动</h2><p>' + esc((ws.data_freshness || {}).status || 'NO_OBSERVED_CHANGE') + '</p></div></div>' +
        (recent.length ? '<ul class="workspace-change-list">' + recent.slice().reverse().map(function (item) { return '<li><strong>' + esc(item.event_type) + '</strong><span>' + esc(item.outcome) + '</span><small>' + esc(item.reference) + '</small></li>'; }).join('') + '</ul>' : '<div class="workspace-empty">暂无真实学习变化。完成作答或教学任务后，这里才会更新。</div>');
      if (resume && ws.resume_action && ws.resume_action.status === 'AVAILABLE') resume.innerHTML = '<button class="workspace-resume" data-ws-route="' + escAttr(ws.resume_action.route) + '"><strong>继续上次学习</strong><small>' + esc(ws.resume_action.reason) + '</small></button>';
      var profileSave = g('workspaceProfileSave');
      if (profileSave) profileSave.addEventListener('click', function () {
        var fields = [
          ['learning_stage', (g('workspaceStageInput') || {}).value],
          ['professional_background', (g('workspaceMajorInput') || {}).value],
          ['domain_experience', (g('workspaceExperienceInput') || {}).value],
          ['learning_goal', (g('workspaceGoalInput') || {}).value]
        ].filter(function (item) { return String(item[1] || '').trim(); });
        var feedback = g('workspaceProfileFeedback');
        if (!fields.length) { if (feedback) feedback.textContent = '未填写任何内容；系统不会创建默认背景。'; return; }
        profileSave.disabled = true;
        Promise.all(fields.map(function (item) { return apiReq('POST', '/api/user-understanding/answer', { learner_id: learnerId(), payload: { slot_key: item[0], value: String(item[1]).trim() } }); }))
          .then(function () { if (feedback) feedback.textContent = '已保存为 DECLARED 先验，下次任务由 Learner Intelligence 统一解释。'; })
          .catch(function (error) { if (feedback) feedback.textContent = (error && error.message) || '保存失败。'; })
          .then(function () { profileSave.disabled = false; });
      });
      var diagnostic = g('workspaceDiagnostic');
      if (diagnostic) diagnostic.addEventListener('click', function () {
        try { sessionStorage.setItem('dy3_practice_attempt_purpose', 'DIAGNOSTIC'); } catch (ignoreDiagnostic) {}
        if (window.sv) window.sv('practice');
      });
      var profileClear = g('workspaceProfileClear');
      if (profileClear) profileClear.addEventListener('click', function () {
        var feedback = g('workspaceProfileFeedback');
        profileClear.disabled = true;
        apiReq('DELETE', '/api/user-understanding/profile?learner_id=' + encodeURIComponent(learnerId()))
          .then(function () { if (feedback) feedback.textContent = '自愿画像已清除；任务与作答事实未改变。'; setTimeout(function () { renderLearnerOverview(); }, 300); })
          .catch(function (error) { if (feedback) feedback.textContent = (error && error.message) || '清除失败。'; profileClear.disabled = false; });
      });
      bindWorkspaceActions(ct);
      // Declared fields are intentionally shown only when they actually exist.
      if (declared.learning_goal && stage) stage.setAttribute('data-learning-goal-present', 'true');
    }).catch(function () {
      var stage = g('workspaceStage'), sequence = g('workspaceSequence'), changes = g('workspaceChanges');
      if (stage) stage.innerHTML = '<div class="workspace-error"><strong>学习状态暂时无法读取</strong><p>系统不会用默认掌握度替代真实数据。你仍可直接提问。</p></div>';
      if (sequence) sequence.innerHTML = '<div class="workspace-layer-head"><span>03</span><div><h2>学习序列暂不可用</h2></div></div>';
      if (changes) changes.innerHTML = '<div class="workspace-layer-head"><span>04</span><div><h2>最近变化暂不可用</h2></div></div>';
    });
  }

  // Learning workspace is the default learner entry.
  function renderLearnerOverview() {
    var ct = g('content');
    if (!ct) return;
    renderWorkspaceProduct(ct);
    return;
    var lastTask = r08StoredTask();
    var lastQuestion = r08StoredQuestion();
    ct.innerHTML = '<article class="r04b-home r08b3-home">' + r08Journey('overview', lastTask) +
      '<div class="r08b3-home-stage"><header class="r04b-hero"><div class="r04-section-kicker">绿色健康照明 · 稀土发光材料</div>' +
      '<h1>稀土发光材料<br>科研学习平台</h1>' +
      '<p>面向稀土发光材料学习与研究，系统先理解任务和学生，再组织四个专业 Agent 检索证据、解释机制、审核结论，并规划下一步学习。</p>' +
      '<div class="r08b3-capability-line" aria-label="系统核心能力"><span>学习者理解</span><span>四 Agent 协同</span><span>Evidence</span><span>Reviewer</span><span>学习决策</span></div>' +
      '<div class="r04b-home-query"><input id="homeTaskInput" type="text" placeholder="例如：为什么 Dy³⁺ 浓度增加会导致发光下降？"><button class="btn primary" id="homeTaskStart">开始任务</button></div>' +
      '<div class="r04b-example-row"><span>推荐演示：</span>' +
      ['为什么 Dy³⁺ 浓度增加会导致发光下降？', '为什么 Dy³⁺ 会产生黄蓝双发射？', '3000 K 是否一定更健康？'].map(function (q) {
        return '<button class="r04b-example" data-question="' + escAttr(q) + '">' + esc(q) + '</button>';
      }).join('') + '</div></header>' +
      '<section id="r08WorkspaceSummary" class="r08-profile-panel r08b3-profile-panel" aria-label="学习状态摘要"><div class="r04-processing"><span class="spinner"></span><div><strong>正在读取真实学习画像</strong></div></div></section></div>' +
      '<section class="r04b-story r08b3-story" aria-label="系统工作方式"><div><span>01</span><strong>理解任务与学习者</strong><p>读取当前水平与薄弱知识</p></div>' +
      '<div><span>02</span><strong>组织专家</strong><p>四个 Agent 围绕同一任务分工</p></div>' +
      '<div><span>03</span><strong>检索证据</strong><p>围绕材料机制寻找真实依据</p></div>' +
      '<div><span>04</span><strong>科学审核</strong><p>Reviewer 挑战过度或无依据结论</p></div>' +
      '<div><span>05</span><strong>持续培养</strong><p>把结果转化为下一步学习行动</p></div></section>' +
      r08RecentTaskPanel(lastTask, lastQuestion) +
      '<section class="r04b-home-note"><strong>系统不会做什么</strong><p>不会把候选生成包装成 Agent 投票，不会模拟实时思考，也不会在证据不足时给出没有依据的材料排名。</p></section></article>';
    r08BindJourney(ct);
    var input = g('homeTaskInput');
    function openTask(question) {
      var value = String(question || (input && input.value) || '').trim();
      if (value) sessionStorage.setItem('dy3_pending_query', value);
      if (window.sv) window.sv('query');
    }
    var start = g('homeTaskStart');
    if (start) start.addEventListener('click', function () { openTask(); });
    if (input) input.addEventListener('keydown', function (event) { if (event.key === 'Enter') openTask(); });
    ct.querySelectorAll('.r04b-example').forEach(function (button) {
      button.addEventListener('click', function () { openTask(button.getAttribute('data-question')); });
    });
    apiReq('GET', '/l2/profile/' + learnerId()).then(function (profile) {
      return Promise.all([
        apiReq('GET', '/api/match-report/' + encodeURIComponent(learnerId())).catch(function () { return null; }),
        apiReq('POST', '/l4/decision/next-action', { learner_id: learnerId(), mode: 'guide', learner_profile: profile || {} }).catch(function () { return null; })
      ]).then(function (values) {
        var match = values[0] || {};
        return {
          profile: profile || {},
          report: match.report || null,
          decision: (values[1] && values[1].data) || values[1] || {}
        };
      });
    }).then(function (result) {
      var box = g('r08WorkspaceSummary');
      if (!box) return;
      var profile = result.profile;
      var report = result.report;
      var decision = result.decision;
      if (profile.initial_assessed === false) {
        box.innerHTML = '<div class="r08-section-title"><div><span>学习工作台</span><h2>系统还不了解你</h2></div><small>UNKNOWN 不等于基础薄弱</small></div>' +
          '<p class="r04-muted">你可以直接提问，也可以自愿提供少量背景。声明信息只作为低权重先验；真实作答和学习事件会逐步覆盖它。</p>' +
          '<div class="r08-profile-grid"><label><span>学习阶段（可跳过）</span><select id="coldStage"><option value="">不填写</option><option value="undergraduate">本科</option><option value="graduate">研究生</option><option value="researcher">科研人员</option></select></label>' +
          '<label><span>专业背景（可跳过）</span><input id="coldMajor" placeholder="如：材料、物理、光电"></label>' +
          '<label><span>领域经历（可跳过）</span><select id="coldExperience"><option value="">不填写</option><option value="introductory">刚开始了解</option><option value="coursework">修过相关课程</option><option value="lab">有实验经历</option><option value="research">有科研经历</option></select></label>' +
          '<label><span>当前目标（可跳过）</span><input id="coldGoal" placeholder="如：理解 Dy³⁺ 白光机制"></label></div>' +
          '<div class="t1-resource-actions"><button class="btn primary" id="coldSave">保存自愿信息</button><button class="btn ghost" id="coldDiagnostic">用真实题库开始诊断</button></div><div id="coldFeedback" class="t1-resource-feedback" aria-live="polite"></div>';
        var save = g('coldSave');
        if (save) save.addEventListener('click', function () {
          var fields = [
            ['learning_stage', (g('coldStage') || {}).value],
            ['professional_background', (g('coldMajor') || {}).value],
            ['domain_experience', (g('coldExperience') || {}).value],
            ['learning_goal', (g('coldGoal') || {}).value]
          ].filter(function (item) { return String(item[1] || '').trim(); });
          if (!fields.length) { var fb0=g('coldFeedback'); if(fb0)fb0.textContent='没有填写也可以直接使用系统。'; return; }
          Promise.all(fields.map(function (item) { return apiReq('POST','/api/user-understanding/answer',{learner_id:learnerId(),payload:{slot_key:item[0],value:item[1]}}); }))
            .then(function () { var fb=g('coldFeedback'); if(fb)fb.textContent='已保存为低权重声明先验；不会当作已掌握事实。'; })
            .catch(function () { var fb=g('coldFeedback'); if(fb)fb.textContent='保存失败，仍可直接提问。'; });
        });
        var diagnostic = g('coldDiagnostic');
        if (diagnostic) diagnostic.addEventListener('click', function () { if(window.sv)window.sv('practice'); });
        return;
      }
      var findings = report && Array.isArray(report.findings) ? report.findings : [];
      var pathNodes = report && report.learning_path && Array.isArray(report.learning_path.nodes) ? report.learning_path.nodes : [];
      var conceptNames = {};
      pathNodes.forEach(function (node) { conceptNames[node.concept_id] = node.name || node.concept_id; });
      var nextAction = (report && report.next_action) || {};
      var nextText = nextAction.reason || nextAction.target || decision.summary || '当前没有公开的下一步建议。';
      var sufficiency = (report && report.evidence_sufficiency) || {};
      var stateText = !report ? '权威报告暂不可用' : report.status === 'EVIDENCE_BACKED' ? String(sufficiency.answer_record_count || 0) + ' 条真实作答证据' : report.status === 'MODEL_ONLY' ? '仅有模型画像，尚无真实作答' : 'UNKNOWN · 等待真实诊断';
      var extras = profile.extras && typeof profile.extras === 'object' ? profile.extras : {};
      var learningGoal = r04Text(profile.learning_goal || extras.learning_goal || '');
      var professional = r04Text(profile.major || profile.professional_background || extras.major || extras.professional_background || '');
      var findingLabels = { VERIFIED_WEAKNESS: '已验证薄弱', PREREQUISITE_GAP: '先修缺口', MISCONCEPTION: '误概念', UNKNOWN: '未知' };
      var weakHtml = findings.slice(0, 3).map(function (finding) {
        var reference = finding.reference || 'unknown';
        return '<li><span>' + esc(conceptNames[reference] || reference) + '</span><strong>' + esc(findingLabels[finding.type] || finding.type) + '</strong></li>';
      }).join('');
      var optionalFacts = '';
      if (professional) optionalFacts += '<div><span>专业背景</span><strong>' + esc(professional) + '</strong></div>';
      if (learningGoal) optionalFacts += '<div><span>学习目标</span><strong>' + esc(learningGoal) + '</strong></div>';
      box.innerHTML = '<div class="r08-section-title"><div><span>学习工作台</span><h2>我是谁，我现在在哪里，下一步是什么</h2></div><small>统一来源：Learner Intelligence Report</small></div>' +
        '<div class="r08-profile-grid"><div><span>当前学习者</span><strong>' + esc(learnerId()) + '</strong><small>' + (String(learnerId()).indexOf('guest-') === 0 ? 'SESSION_GUEST' : 'AUTHENTICATED') + '</small></div>' +
        '<div><span>当前水平</span><strong>' + esc(profile.level || (profile.initial_assessed === false ? '尚未完成初测' : 'UNKNOWN')) + '</strong><small>MODEL_INFERRED</small></div>' +
        '<div><span>学习证据</span><strong>' + esc(stateText) + '</strong><small>' + esc(sufficiency.source_class || 'UNKNOWN') + '</small></div>' +
        '<div><span>学习方式</span><strong>' + esc(profile.learning_style || '未公开') + '</strong><small>MODEL_INFERRED / UNKNOWN</small></div>' + optionalFacts + '</div>' +
        '<div class="r08-workbench-grid"><div><h3>当前学习判断</h3>' + (weakHtml ? '<ul class="r08-compact-list">' + weakHtml + '</ul>' : '<p class="r04-muted">当前没有足够事实形成学习判断。</p>') + '</div>' +
        '<div><h3>下一步行动</h3><p>' + esc(nextText) + '</p><button class="btn primary" id="homeGrowthDecision">进入成长决策</button></div></div>';
      var growth = g('homeGrowthDecision');
      if (growth) growth.addEventListener('click', function () { if (window.sv) window.sv('learn-weak'); });
    }).catch(function () {
      var box = g('r08WorkspaceSummary');
      if (box) box.innerHTML = '<div class="r08-section-title"><div><span>学习工作台</span><h2>学习画像暂不可用</h2></div></div><p class="r04-muted">当前无法读取画像；系统不会补造水平、背景或学习目标，仍可直接发起核心任务。</p>';
    });
  }

  // 2d. 管理概览 (admin overview)
  function renderAdminOverview() {
    var ct = g('content');
    if (!ct) return;
    apiReq('GET', '/l5/agents').then(function (agents) {
      var list = (agents && agents.agents) || [];
      var healthy = list.filter(function (a) { return a.instance && a.instance.healthy; }).length;
      var agentCards = list.length ? list.map(function (a) {
        var inst = a.instance || {};
        var ok = inst.healthy === true;
        return '<div style="background:var(--surface2);border-radius:10px;padding:12px 14px;display:flex;align-items:center;justify-content:space-between;gap:10px">' +
          '<div style="min-width:0"><div style="font-weight:600;font-size:13px">' + esc(a.name) + '</div>' +
          '<div style="font-size:11px;color:var(--muted);font-family:var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(a.id) + '</div></div>' +
          '<span class="badge ' + (ok ? 'ok' : 'warn') + '">' + (ok ? '运行中' : '未激活') + '</span></div>';
      }).join('') : '<p style="color:var(--muted);font-size:13px;margin:4px 0">Agent 运行时未初始化</p>';
      var quick = [
        ['用户管理', 'users'], ['智能体列表', 'agents-list'], ['协作链路', 'agents-chain'],
        ['反幻觉审查', 'gov-review'], ['策略管理', 'gov-policy'], ['合规报告', 'gov-report']
      ].map(function (q) {
        return '<button class="quick-nav" data-goto="' + q[1] + '">' + esc(q[0]) + '</button>';
      }).join('');
      ct.innerHTML = '<div class="card"><h3>管理概览</h3>' +
        '<div class="grid cols-4" style="margin-bottom:6px">' +
        statCard('核心 Agent', list.length + ' 个') +
        statCard('运行中', healthy + ' 个') +
        statCard('系统层', 'L0-L7 八层') +
        statCard('待审内容', '见反幻觉审查') +
        '</div>' +
        '<div class="section-header"><span class="section-icon">🤖</span><h4>Agent 运行状态</h4><span class="section-line"></span></div>' +
        '<div class="grid cols-2">' + agentCards + '</div>' +
        '<div class="section-header" style="margin-top:16px"><span class="section-icon">🧭</span><h4>快捷入口</h4><span class="section-line"></span></div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:8px">' + quick + '</div>' +
        '<p style="color:var(--muted);font-size:12px;margin-top:14px">完整 Agent 状态见「智能体列表 / 协作链路」，治理信息见「反幻觉审查 / 策略管理 / 合规报告」。</p></div>';
      ct.querySelectorAll('[data-goto]').forEach(function (b) {
        b.addEventListener('click', function () { if (window.sv) window.sv(b.getAttribute('data-goto')); });
      });
    }).catch(function () {
      ct.innerHTML = '<div class="card"><h3>管理概览</h3><p style="color:var(--muted)">数据读取失败</p></div>';
    });
  }

  /* ---------- 用户管理三件套 (列表/角色/导入, 对接 L1 /users /roles /import) ---------- */
  function renderUsers() {
    var ct = g('content');
    if (!ct) return;
    ct.innerHTML = '<div class="card"><h3>用户列表</h3><div id="mf6UsersBody">正在读取…</div></div>';
    apiReq('GET', '/l1/api/v1/users').then(function (d) {
      var users = (d && d.users) || [];
      var rows = users.map(function (u) {
        return '<tr><td>' + esc(u.student_id) + '</td><td>' + esc(u.user_id) + '</td><td><span class="badge ' + (u.role === 'admin' ? 'warn' : 'info') + '">' + esc(u.role) + '</span></td><td>' + esc(u.institution_id) + '</td></tr>';
      }).join('');
      var b = g('mf6UsersBody');
      if (b) b.innerHTML = '<p style="color:var(--muted);font-size:12px">共 ' + users.length + ' 个用户（演示环境内存存储）</p>' +
        '<div class="table-wrap"><table><thead><tr><th>学号</th><th>用户 ID</th><th>角色</th><th>机构</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
    }).catch(function (e) {
      var b = g('mf6UsersBody');
      if (b) b.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  function renderRoles() {
    var ct = g('content');
    if (!ct) return;
    ct.innerHTML = '<div class="card"><h3>角色与权限</h3><div id="mf6RolesBody">正在读取…</div></div>';
    apiReq('GET', '/l1/api/v1/roles').then(function (d) {
      var roles = (d && d.roles) || [];
      var cards = roles.map(function (r) {
        var perms = (r.permissions || []).map(function (p) { return '<li>' + esc(p) + '</li>'; }).join('');
        return '<div class="card" style="margin:8px 0"><div style="display:flex;justify-content:space-between;align-items:center">' +
          '<b>' + esc(r.label) + '</b><span class="badge info">' + esc(r.role) + '</span></div>' +
          '<ul style="margin:8px 0 0;padding-left:18px;font-size:13px;color:var(--muted)">' + perms + '</ul></div>';
      }).join('');
      var b = g('mf6RolesBody');
      if (b) b.innerHTML = cards || '<p style="color:var(--muted)">暂无角色数据</p>';
    }).catch(function (e) {
      var b = g('mf6RolesBody');
      if (b) b.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  function renderImport() {
    var ct = g('content');
    if (!ct) return;
    ct.innerHTML = '<div class="card"><h3>批量导入用户</h3>' +
      '<div class="callout" style="font-size:12px">每行一个用户：学号,密码,角色(student/teacher/admin/graduate/alumni)<br>示例：<code>DY20240003,demo123,student</code><br>学号须符合 <code>2位大写字母+8位数字</code> 格式</div>' +
      '<textarea id="mf6ImportText" rows="6" style="width:100%;margin-top:10px;padding:10px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink);font-size:13px;font-family:monospace" placeholder="DY20240003,demo123,student&#10;DY20240004,demo123,teacher">' +
      '</textarea>' +
      '<div style="display:flex;gap:10px;margin-top:10px;align-items:center"><button class="btn primary" id="mf6ImportGo">导入</button><span id="mf6ImportResult" style="font-size:12px;color:var(--muted)"></span></div></div>';
    var go = g('mf6ImportGo');
    if (go) go.addEventListener('click', function () {
      var txt = (g('mf6ImportText') || {}).value || '';
      if (!txt.trim()) return;
      var rs = g('mf6ImportResult');
      if (rs) rs.textContent = '导入中…';
      apiReq('POST', '/l1/api/v1/users/import', { content: txt }).then(function (d) {
        if (rs) rs.innerHTML = '<span class="badge ok">成功 ' + d.imported + '</span> <span class="badge warn">跳过 ' + d.skipped + '</span>' +
          ((d.errors || []).length ? '<div style="margin-top:6px;color:var(--danger)">' + esc(d.errors.join('<br>')) + '</div>' : '');
      }).catch(function (e) {
        if (rs) rs.innerHTML = '<span style="color:var(--danger)">' + esc(e.message) + '</span>';
      });
    });
  }

  // 2e. 实时答疑视图 (query — 统一会话闭环: L1 用户会话 + L5 执行会话关联)
  // 多智能体协同流水线可视化: 4 个核心 Agent 的分步流水条 (诊断→生成→审核→决策)
  // state: idle 待处理 / busy 处理中 / done 已完成; activeIdx+done 控制高亮推进
  function agentPipelineHTML(activeIdx, done) {
    var steps = [
      { n: '学情诊断', i: '🔍' },
      { n: '知识生成', i: '📝' },
      { n: '审核校验', i: '✅' },
      { n: '导学决策', i: '🧭' }
    ];
    return '<div style="display:flex;align-items:center;gap:3px">' +
      steps.map(function (s, i) {
        var state = (i < activeIdx || done) ? 'done' : (i === activeIdx ? 'busy' : 'idle');
        var bg = state === 'busy' ? 'linear-gradient(135deg,#6366f1,#8b5cf6)'
          : (state === 'done' ? 'linear-gradient(135deg,#10b981,#059669)' : 'var(--surface)');
        var color = state !== 'idle' ? '#fff' : 'var(--muted)';
        var tag = state === 'done' ? '✓ 完成' : (state === 'busy' ? '处理中…' : '待处理');
        var cell = '<div style="flex:1;text-align:center;padding:8px 3px;border-radius:8px;font-size:12px;background:' + bg +
          ';color:' + color + ';border:1px solid ' + (state === 'idle' ? 'var(--rule)' : 'transparent') +
          (state === 'busy' ? ';box-shadow:0 0 0 3px rgba(99,102,241,.22)' : '') + ';transition:all .25s">' +
          '<div style="font-size:15px">' + s.i + '</div>' +
          '<div style="font-weight:600;margin-top:2px">' + s.n + '</div>' +
          '<div style="font-size:10px;opacity:.85">' + tag + '</div></div>';
        var arrow = (i < steps.length - 1) ? '<div style="flex:none;color:var(--muted);font-size:12px">→</div>' : '';
        return cell + arrow;
      }).join('') + '</div>';
  }

  // 知识讲解幻灯片: 结构化分页 (每页标题 + 内容, PPT 式讲解)
  function renderSlideDeck(text) {
    var s = String(text || '').trim();
    if (!s) return '';
    // 清洗引用号 [28]/[29,30] 与英文置信度残标 "Low/Medium/High" (行首), 收敛空行
    s = s.replace(/\[\s*\d+(?:\s*[,，]\s*\d+)*\s*\]/g, '');
    s = s.replace(/^[ \t]*(?:[-•*]\s*)?(?:low|medium|high)\b[ \t]*/gim, '');
    s = s.replace(/\r/g, '').replace(/\n{3,}/g, '\n\n');

    // 按 markdown 标题 (#/##/###) 分块: 标题行作页标题, 其下内容作页正文
    var blocks = [], cur = null, preamble = [];
    s.split('\n').forEach(function (ln) {
      var t = ln.trim();
      if (!t) return;
      var hm = t.match(/^(#{1,4})\s+(.+)$/);
      if (hm) {
        if (cur) blocks.push(cur);
        var ht = hm[2].trim();
        // 知识库切片丢失换行, 标题常与正文粘连成 "## 标题 正文": 在"空格+句子起始词"处切分
        var m2 = ht.match(/^(.*?)\s+(?=在|当|这|该|其|通|一|研|实|利|为|由|例|如|因|随|若|无|对|经|比|根|据|此|可|而|但|然|另|上|下|首|主|具|需|注|图|表|采|选|常|同|假|设|即|依|鉴|结|本|文|从|以)/);
        cur = { title: m2 ? m2[1] : ht, body: m2 ? [ht.slice(m2[1].length).replace(/^\s+/, '')] : [] };
        return;
      }
      var clean = t.replace(/^[-•*]\s*/, '');
      if (cur) cur.body.push(clean); else preamble.push(clean);
    });
    if (cur) blocks.push(cur);

    var pages = [];
    if (preamble.length) pages.push({ title: '综合解答', body: preamble.join('\n') });
    blocks.forEach(function (b) { pages.push({ title: b.title, body: b.body.join('\n') }); });

    // 无标题结构时回退: 按句切分, 每页两句 (首句作标题, 正文不再重复)
    if (!pages.length) {
      var sentences = s.split(/(?<=[。；！？])/).map(function (x) { return x.trim(); }).filter(Boolean);
      if (sentences.length < 2) return esc(s.replace(/^#{1,4}\s+/gm, ''));
      for (var i = 0; i < sentences.length; i += 2) {
        pages.push({ title: sentences[i].slice(0, 18), body: sentences.slice(i + 1, i + 2).join('') });
      }
    }
    if (pages.length < 2) return esc(s.replace(/^#{1,4}\s+/gm, ''));

    var html = '<div class="mf6-slides" data-page="0">' +
      pages.map(function (pg, i) {
        var titleHtml = pg.title ? '<div style="font-weight:700;font-size:14px;color:var(--accent);margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid var(--rule)">' + esc(pg.title) + '</div>' : '';
        var bodyHtml = pg.body ? '<div style="font-size:13px;line-height:1.75;white-space:pre-wrap">' + esc(pg.body) + '</div>' : '';
        return '<div class="mf6-slide" data-i="' + i + '" style="' + (i === 0 ? '' : 'display:none') + ';background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;min-height:90px">' + titleHtml + bodyHtml + '</div>';
      }).join('') +
      '</div>' +
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px">' +
      '<button class="btn ghost sm mf6-slide-prev" style="font-size:12px">‹ 上一页</button>' +
      '<span class="mf6-slide-ind" style="font-size:12px;color:var(--muted)">1 / ' + pages.length + '</span>' +
      '<button class="btn ghost sm mf6-slide-next" style="font-size:12px">下一页 ›</button></div>';
    return html;
  }

  function r04PublicData(payload) {
    return (payload && payload.data && typeof payload.data === 'object') ? payload.data : (payload || {});
  }

  var r08TaskTruth = { learner_id: '', loaded: false, loading: false, data: null, question: '', tasks: [] };

  function r08TaskProjection(task) {
    if (!task || typeof task !== 'object') return null;
    var data = Object.assign({}, task.public_result || {});
    data.task_id = task.task_id || data.task_id || '';
    data.task_state = task.state || data.task_state || '';
    data.task_question = task.query || task.brief || data.task_question || '';
    data.answer_version = Number((task.answer || {}).version || data.answer_version || 0);
    data.task_events = Array.isArray(task.task_events) ? task.task_events : (data.task_events || []);
    data.answer = ((task.answer || {}).text !== undefined) ? (task.answer || {}).text : (data.answer || '');
    data.review = ((task.reviewer || {}).review) || data.review || {};
    data.quality_release = ((task.reviewer || {}).release) || data.quality_release || {};
    data.learning_resources = ((task.resource_plan || {}).resources) || data.learning_resources || [];
    data.agent_trace = Array.isArray(task.agent_contributions) ? task.agent_contributions : (data.agent_trace || []);
    data.recommended_path = ((task.next_action || {}).recommended_path) || data.recommended_path || [];
    return data;
  }

  function r08StoredTask() {
    if (r08TaskTruth.learner_id !== learnerId()) return null;
    return r08TaskTruth.data;
  }

  function r08StoredQuestion() {
    if (r08TaskTruth.learner_id !== learnerId()) return '';
    return r08TaskTruth.question || '';
  }

  function r08LoadTaskTruth(onReady) {
    var lid = learnerId();
    if (r08TaskTruth.learner_id !== lid) {
      r08TaskTruth = { learner_id: lid, loaded: false, loading: false, data: null, question: '', tasks: [] };
    }
    if (r08TaskTruth.loaded) return Promise.resolve(r08TaskTruth.data);
    if (r08TaskTruth.loading) return Promise.resolve(null);
    r08TaskTruth.loading = true;
    return apiReq('GET', '/api/learning-tasks/' + encodeURIComponent(lid)).then(function (result) {
      var tasks = (result && result.tasks) || [];
      r08TaskTruth.tasks = tasks;
      if (!tasks.length) return null;
      return apiReq('GET', '/api/learning-tasks/' + encodeURIComponent(lid) + '/' + encodeURIComponent(tasks[0].task_id));
    }).then(function (result) {
      var task = result && result.task ? result.task : null;
      r08TaskTruth.loaded = true;
      r08TaskTruth.loading = false;
      r08TaskTruth.data = r08TaskProjection(task);
      r08TaskTruth.question = task ? String(task.query || task.brief || '') : '';
      if (typeof onReady === 'function') onReady();
      return r08TaskTruth.data;
    }).catch(function () {
      r08TaskTruth.loaded = true;
      r08TaskTruth.loading = false;
      if (typeof onReady === 'function') onReady();
      return null;
    });
  }

  function r08Journey(activeView, taskData) {
    var data = taskData && typeof taskData === 'object' ? taskData : null;
    var labels = { overview: '学习总览', query: '任务工作区', 'agents-chain': '协同分析', kb: '知识证据', 'learn-weak': '成长路径' };
    var release = data && data.quality_release && typeof data.quality_release === 'object' ? data.quality_release : {};
    var taskLabel = data && data.task_id ? ('最近任务 ' + data.task_id + (data.task_state ? ' · ' + data.task_state : '') + (release.status ? ' · ' + release.status : '') + (data.answer_version ? ' · v' + data.answer_version : '')) : '尚无最近任务，从学习者画像开始';
    return '<div class="r08-journey" aria-label="当前任务"><div class="r08-journey-head"><div><span>' + esc(labels[activeView] || '任务') + '</span><strong>当前任务</strong></div><small>' + esc(taskLabel) + '</small></div></div>';
  }

  function r08BindJourney(root) {
    if (!root) return;
    root.querySelectorAll('[data-r08-view]').forEach(function (button) {
      button.addEventListener('click', function () {
        var view = button.getAttribute('data-r08-view');
        if (view && window.sv) window.sv(view);
      });
    });
  }

  function r08TaskHistoryPanel() {
    var tasks = Array.isArray(r08TaskTruth.tasks) ? r08TaskTruth.tasks.slice(0, 6) : [];
    if (!tasks.length) return '<section class="r08-task-history"><h3>学习记录</h3><p class="r04-muted">尚无服务端学习任务。</p></section>';
    return '<section class="r08-task-history"><div class="r08-section-title"><div><span>可继续的任务</span><h3>学习记录</h3></div><small>来源：LearningTaskStore</small></div><div>' + tasks.map(function (task) {
      return '<button class="r08-task-history-item" data-resume-task="' + escAttr(task.task_id || '') + '"><span>' + esc(task.state || 'UNKNOWN') + '</span><strong>' + esc(task.brief || task.task_id || '学习任务') + '</strong><small>继续同一 task_id</small></button>';
    }).join('') + '</div></section>';
  }

  function r08BindTaskHistory(root) {
    if (!root) return;
    root.querySelectorAll('[data-resume-task]').forEach(function (button) {
      button.addEventListener('click', function () {
        var taskId = button.getAttribute('data-resume-task') || '';
        if (!taskId) return;
        apiReq('POST', '/api/learning-tasks/' + encodeURIComponent(learnerId()) + '/' + encodeURIComponent(taskId) + '/resume', {}).then(function (result) {
          var task = result && result.task ? result.task : null;
          r08TaskTruth.data = r08TaskProjection(task);
          r08TaskTruth.question = task ? String(task.query || task.brief || '') : '';
          renderQuery();
        });
      });
    });
  }

  function r08RefreshTaskHistory(root) {
    var lid = learnerId();
    return apiReq('GET', '/api/learning-tasks/' + encodeURIComponent(lid)).then(function (result) {
      if (r08TaskTruth.learner_id !== lid) return [];
      r08TaskTruth.tasks = (result && Array.isArray(result.tasks)) ? result.tasks : [];
      var panel = root && root.querySelector ? root.querySelector('.r08-task-history') : null;
      if (panel) {
        panel.outerHTML = r08TaskHistoryPanel();
        r08BindTaskHistory(root);
      }
      return r08TaskTruth.tasks;
    }).catch(function () { return []; });
  }

  function r08RecentTaskPanel(data, question) {
    if (!data || typeof data !== 'object') return '<section class="r08-recent-task"><div><span>最近任务</span><strong>还没有完成可回看的任务</strong><p>从一个 Dy³⁺ 或绿色健康照明问题开始，后续协同、证据和成长决策会围绕同一任务展开。</p></div><button class="btn primary" data-r08-view="query">开始核心任务</button></section>';
    var review = data.review && typeof data.review === 'object' ? data.review : {};
    var verdict = review.verdict || review.status || '未提供';
    return '<section class="r08-recent-task"><div><span>最近任务</span><strong>' + esc(question || data.task_id || '最近一次学习任务') + '</strong><p>' + esc(String(data.task_state || '状态未公开')) + ' · Reviewer ' + esc(String(verdict)) + '</p></div>' +
      '<div class="r08-inline-actions"><button class="btn ghost" data-r08-view="agents-chain">查看协同</button><button class="btn ghost" data-r08-view="kb">核对证据</button><button class="btn primary" data-r08-view="learn-weak">继续成长</button></div></section>';
  }

  function r08LatestTaskContext(data, question) {
    if (!data) return '';
    var release = data.quality_release && typeof data.quality_release === 'object' ? data.quality_release : {};
    return '<div class="r08-task-context"><div><span>最近任务</span><strong>' + esc(question || data.task_id || '最近一次任务') + '</strong><small>' + esc(String(data.task_state || '状态未公开')) + (release.status ? ' · ' + esc(String(release.status)) : '') + '</small></div></div>';
  }

  function r08TaskFactFlow(data, learnerLevel, evidence, verdict) {
    var fixedAgents = {
      'agent.learning.diagnosis': true,
      'agent.knowledge.generation': true,
      'agent.quality.review': true,
      'agent.guidance.decision': true
    };
    var actors = {};
    (Array.isArray(data.agent_trace) ? data.agent_trace : []).forEach(function (event) { if (event && fixedAgents[event.agent_id]) actors[event.agent_id] = true; });
    (Array.isArray(data.flow_events) ? data.flow_events : []).forEach(function (event) { if (event && fixedAgents[event.agent]) actors[event.agent] = true; });
    var paths = Array.isArray(data.recommended_path) ? data.recommended_path : [];
    var evidenceText = evidence.length ? evidence.length + ' 条公开证据' : (data.knowledge_unavailable ? '已标记知识缺口' : '未提供公开证据');
    var steps = [
      ['学习者分析', learnerLevel || '未公开'],
      ['知识检索', evidenceText],
      ['Agent 协同', Object.keys(actors).length ? Object.keys(actors).length + ' / 4 个 Agent 有公开贡献' : '未提供公开贡献'],
      ['科学审核', verdict || '未提供'],
      ['教学决策', paths.length ? paths.length + ' 项下一步行动' : '未提供公开路径']
    ];
    return '<section class="r08-task-facts" aria-label="本次教学任务公开事实"><div class="r08-section-title"><div><span>本次教学任务</span><h3>从学生问题到可信教学结果</h3></div><small>以下仅来自本次公开响应</small></div><div class="r08-task-fact-flow">' + steps.map(function (item, index) {
      return '<div><span>0' + (index + 1) + '</span><strong>' + esc(item[0]) + '</strong><p>' + esc(String(item[1])) + '</p></div>';
    }).join('') + '</div></section>';
  }

  function r04AgentName(agentId) {
    var names = {
      'agent.learning.diagnosis': '学情诊断 Agent',
      'agent.knowledge.generation': '知识生成 Agent',
      'agent.quality.review': '质量审核 Agent',
      'agent.guidance.decision': '导学决策 Agent'
    };
    return names[agentId] || String(agentId || '系统');
  }

  function r04Text(value) {
    if (value == null) return '';
    if (typeof value === 'string' || typeof value === 'number') return String(value);
    return String(value.title || value.name || value.action || value.description || value.content || value.text || value.detail || '');
  }

  function r04SourceReference(value) {
    var item = value && typeof value === 'object' ? value : {};
    var title = item.source_title || item.title || item.section || item.document_id || item.source || item.chunk_id || '公开来源';
    var uri = String(item.source_uri || item.uri || item.url || item.source || '');
    var sourceType = String(item.source_type || '');
    var status = String(item.evidence_status || '');
    var label = '<strong>' + esc(String(title)) + '</strong>';
    if (/^https?:\/\//i.test(uri)) {
      label = '<a href="' + escAttr(uri) + '" target="_blank" rel="noopener noreferrer">' + esc(String(title)) + '</a>';
    }
    var meta = [sourceType, status].filter(Boolean).map(function (part) { return '<span>' + esc(part) + '</span>'; }).join('');
    return '<div class="r04-source-reference">' + label + (uri && !/^https?:\/\//i.test(uri) ? '<code>' + esc(uri) + '</code>' : '') + (meta ? '<small>' + meta + '</small>' : '') + '</div>';
  }

  function r04RenderTrace(data) {
    var flow = Array.isArray(data.flow_events) ? data.flow_events : [];
    var lines = Array.isArray(data.collab_lines) ? data.collab_lines : [];
    var trace = Array.isArray(data.agent_trace) ? data.agent_trace : [];
    var taskEvents = Array.isArray(data.task_events) ? data.task_events : [];
    var events = [];
    if (flow.length) {
      events = flow.map(function (event) {
        return { type: event.step || '', agent: event.agent || '', detail: event.detail || event.label || '' };
      });
    } else if (lines.length) {
      lines.forEach(function (line) {
        (line.steps || []).forEach(function (step) {
          events.push({ type: line.label || '', agent: step.agent || '', detail: step.output || '' });
        });
      });
    } else if (trace.length) {
      events = trace.map(function (event) {
        return { type: 'CONTRIBUTION', agent: event.agent_id || '', detail: event.detail || '' };
      });
    } else {
      events = taskEvents.map(function (event) {
        return { type: event.event_type || '', agent: event.producer || '', detail: event.state_after || event.state || '' };
      });
    }
    if (!events.length) {
      return '<div class="r04-empty">本次响应未提供公开协同轨迹。</div>';
    }
    return '<ol class="r04-trace-list">' + events.map(function (event) {
      var type = String(event.type || '').replace(/_/g, ' ');
      return '<li class="r04-trace-item"><div class="r04-trace-meta"><span>' + esc(r04AgentName(event.agent)) + '</span>' +
        (type ? '<span class="r04-event-type">' + esc(type) + '</span>' : '') + '</div>' +
        '<div class="r04-trace-detail">' + esc(String(event.detail || '已完成该协同步骤')) + '</div></li>';
    }).join('') + '</ol>';
  }

  function r04RenderChallenge(data) {
    var flow = Array.isArray(data.flow_events) ? data.flow_events : [];
    var challenge = flow.filter(function (event) {
      return event && event.step === 'CHALLENGE_RAISED';
    });
    var actions = flow.filter(function (event) {
      return event && (event.step === 'REVISION_REQUESTED' || event.step === 'RE_RETRIEVAL_REQUESTED' || event.step === 'CONTRIBUTION_REVISED');
    });
    var correction = data.self_correction;
    if (!challenge.length && !correction) return '';
    var challengeText = challenge.length ? (challenge[0].detail || challenge[0].label || 'Reviewer 提出了科学挑战') : (correction.reason || 'Reviewer 要求修正');
    var actionText = actions.length ? actions.map(function (event) { return event.detail || event.label || event.step; }).join('；') : '系统按 Reviewer 要求完成受控复核。';
    var outcome = correction ? ('审核状态：' + String(correction.verdict_before || '-') + ' → ' + String(correction.verdict_after || '-')) : '修正结果已进入最终审核。';
    return '<section class="r04-challenge" aria-label="Reviewer Challenge"><div class="r04-section-kicker">Reviewer Challenge</div>' +
      '<h4>审核发现了什么</h4><p>' + esc(String(challengeText)) + '</p>' +
      '<div class="r04-challenge-grid"><div><strong>系统行动</strong><span>' + esc(actionText) + '</span></div>' +
      '<div><strong>结果变化</strong><span>' + esc(outcome) + '</span></div></div></section>';
  }

  function t1QualityBlock(data) {
    var release = (data.quality_release && typeof data.quality_release === 'object') ? data.quality_release : {};
    var status = String(release.status || 'DEGRADED');
    var labels = {
      FULL_RELEASE: '已通过质量审核',
      LIMITED_RELEASE: '仅发布已确认的有限结论',
      ASK_USER: '需要补充关键条件',
      REFUSE: '未发布被拒绝的结论',
      WITHHOLD: '当前结果未达到发布条件',
      DEGRADED: '完整智能审核链暂不可用'
    };
    var reasons = Array.isArray(release.reason_codes) ? release.reason_codes : [];
    return '<section class="t1-quality t1-quality-' + escAttr(status.toLowerCase()) + '" aria-label="Quality Release Gate">' +
      '<div><div class="r04-section-kicker">Quality Release Gate</div><h3>' + esc(labels[status] || status) + '</h3>' +
      '<p>' + esc(String(release.message || '未获得可验证的发布决策，默认不发布。')) + '</p></div>' +
      '<div class="t1-quality-facts"><span>Reviewer：' + esc(String(release.review_verdict || '未通过')) + '</span>' +
      '<span>真实修订：' + esc(String(release.correction_count || 0)) + '</span>' +
      (reasons.length ? '<details><summary>查看未发布原因</summary><ul>' + reasons.map(function (reason) { return '<li>' + esc(String(reason)) + '</li>'; }).join('') + '</ul></details>' : '') + '</div></section>';
  }

  function t1TeachingStrategy(data) {
    var strategy = (data.teaching_strategy && typeof data.teaching_strategy === 'object') ? data.teaching_strategy : {};
    var reasons = Array.isArray(strategy.rationale) ? strategy.rationale : [];
    if (!Object.keys(strategy).length) return '';
    return '<section class="t1-teaching-strategy"><div class="r04-section-kicker">回答适配</div><h3>本次适配</h3>' +
      '<div class="t1-strategy-grid"><div><span>内容深度</span><strong>' + esc(String(strategy.content_depth || '保守基础')) + '</strong></div>' +
      '<div><span>解释方式</span><strong>' + esc(String(strategy.explanation_strategy || '基础解释')) + '</strong></div>' +
      '<div><span>难度决策</span><strong>' + esc(String(strategy.difficulty_strategy || '待诊断')) + '</strong></div>' +
      '<div><span>下一焦点</span><strong>' + esc(String(strategy.next_focus || '未确定')) + '</strong></div></div>' +
      (reasons.length ? '<p>' + esc(reasons.join('；')) + '</p>' : '') + '</section>';
  }

  function t1InlineMarkup(value) {
    return esc(String(value || ''))
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>');
  }

  function t1RenderLongText(value) {
    var lines = String(value || '').replace(/\r/g, '').split('\n');
    var html = [];
    var listType = '';
    function closeList() {
      if (!listType) return;
      html.push('</' + listType + '>');
      listType = '';
    }
    lines.forEach(function (rawLine) {
      var line = String(rawLine || '').trim();
      if (!line) { closeList(); return; }
      var heading = line.match(/^(#{2,6})\s+(.+)$/);
      if (heading) {
        closeList();
        html.push('<h' + (heading[1].length === 2 ? '5' : '6') + '>' + t1InlineMarkup(heading[2]) + '</h' + (heading[1].length === 2 ? '5' : '6') + '>');
        return;
      }
      var bullet = line.match(/^[-*]\s+(.+)$/);
      var numbered = line.match(/^\d+[.)]\s+(.+)$/);
      if (bullet || numbered) {
        var nextType = bullet ? 'ul' : 'ol';
        if (listType !== nextType) { closeList(); listType = nextType; html.push('<' + listType + '>'); }
        html.push('<li>' + t1InlineMarkup((bullet || numbered)[1]) + '</li>');
        return;
      }
      closeList();
      html.push('<p>' + t1InlineMarkup(line) + '</p>');
    });
    closeList();
    return '<div class="t1-long-read-markup">' + html.join('') + '</div>';
  }

  function t1ResourceCards(data) {
    var resources = Array.isArray(data.learning_resources) ? data.learning_resources : [];
    if (!resources.length) return '<div class="r04-empty">本次没有形成可验证的个性化学习资源。</div>';
    var actionLabels = {
      open: '查看资源', understood: '我理解了', still_confused: '我还是不理解',
      change_explanation: '换一种解释', request_example: '给我一个例子', deepen: '继续深入',
      ask_follow_up: '继续追问',
      start_practice: '做一道题', next_concept: '进入下一 Concept'
    };
    var sourceLabels = {
      retrieved: '真实存量资源', generated: '模型证据合成·已审核',
      derived: '本任务事实派生', template: '历史模板·不推荐', unknown: '来源未确认'
    };
    var familyLabels = {
      knowledge_understanding: '知识讲义', research_practice: '实证工作单',
      assessment_practice: '分阶练习'
    };
    var formLabels = {
      guided_long_read: '专题讲义', prerequisite_or_gap_card: '先修与缺口',
      evidence_analysis_workbook: '证据分析工作单', research_task: '科研任务',
      practice_bank_launch: '题库分阶练习'
    };
    var difficultyLabels = {
      foundation: '基础', beginner: '入门', intermediate: '进阶', advanced: '高级',
      research: '科研', diagnose_then_maintain: '先诊断再匹配', maintain: '保持',
      lower: '降低难度', raise: '增加挑战'
    };
    var stagePurposeLabels = {
      DIAGNOSTIC: '诊断作答', REQUIRED_PRACTICE: '巩固练习', STAGED_ASSESSMENT: '进阶挑战'
    };
    var claimTypeLabels = { FACT: '事实', INFERENCE: '推断', RECOMMENDATION: '建议' };
    var supportLabels = { SUPPORTS: '证据支持', PARTIAL: '部分支持', INSUFFICIENT: '证据不足', CONTRADICTS: '存在冲突' };
    var sectionSourceLabels = {
      TASK: '任务目标', DECISION: '教学决策', REVIEWED_OUTPUT: 'Reviewer 已通过', REVIEWED_CLAIMS: '已审核判断',
      R06_CONCEPT_RELATION: '概念关系', RELEASE_GATE: '发布边界',
      GENERATION_REVIEWED_RESOURCE: '专题生成·审核通过'
    };
    var activeIndex = resources.findIndex(function (resource) {
      return resource && resource.payload && resource.payload.recommended === true;
    });
    if (activeIndex < 0) activeIndex = 0;
    var tabs = resources.map(function (resource, index) {
      var payload = (resource.payload && typeof resource.payload === 'object') ? resource.payload : {};
      var form = String(resource.resource_form || resource.resource_family || '');
      return '<button class="canvas-resource-tab' + (index === activeIndex ? ' active' : '') + '" type="button" role="tab" aria-selected="' + (index === activeIndex ? 'true' : 'false') + '" data-resource-tab="' + index + '"><span>0' + (index + 1) + '</span><strong>' + esc(String(resource.title || resource.resource_family || '学习资源')) + '</strong><small>' + esc(formLabels[form] || '学习资源') + (payload.recommended === true ? ' · 当前优先' : '') + '</small></button>';
    }).join('');
    var panels = resources.map(function (resource, resourceIndex) {
      var payload = (resource.payload && typeof resource.payload === 'object') ? resource.payload : {};
      var concepts = Array.isArray(payload.concept_names) ? payload.concept_names : [];
      var steps = Array.isArray(payload.steps) ? payload.steps : [];
      var guided = (payload.guided_document && typeof payload.guided_document === 'object') ? payload.guided_document : {};
      var sections = Array.isArray(guided.sections) ? guided.sections : [];
      var lessonSequence = Array.isArray(guided.lesson_sequence) ? guided.lesson_sequence : [];
      var guidedQuestions = Array.isArray(payload.guided_questions) ? payload.guided_questions : [];
      var longSourceRefs = Array.isArray(guided.source_references) ? guided.source_references : [];
      var stages = Array.isArray(payload.stages) ? payload.stages : [];
      var provenance = Array.isArray(resource.provenance) ? resource.provenance : [];
      var actions = Array.isArray(resource.interaction_actions) ? resource.interaction_actions : [];
      var sourceType = String(resource.source_type || 'unknown');
      var actualCharacters = Number(guided.actual_characters || 0);
      var longReadLabel = actualCharacters ? ('专题讲义 · ' + actualCharacters + ' 字 · ' + sections.length + ' 节') : ('学习讲义 · ' + sections.length + ' 节');
      var generationLabel = guided.generation_mode === 'llm_evidence_synthesis' ? '模型证据合成' : guided.generation_mode === 'reviewed_evidence_compilation' ? '已审核证据编排' : '';
      var lessonSequenceHtml = lessonSequence.length ? '<ol class="t1-lesson-sequence">' + lessonSequence.map(function (step, index) {
        return '<li><span>' + String(index + 1).padStart(2, '0') + '</span><strong>' + esc(step.label || step.step || '学习阶段') + '</strong></li>';
      }).join('') + '</ol>' : '';
      var longReadHtml = sections.length ? '<div class="t1-resource-content canvas-reading-document"><div class="canvas-reading-heading"><strong>' + esc(longReadLabel) + '</strong>' + (generationLabel ? '<span>' + esc(generationLabel) + '</span>' : '') + '</div>' + lessonSequenceHtml + (generationLabel ? '<div class="t1-long-form-facts"><span>Reviewer ' + esc(String(guided.review_verdict || '未标注')) + '</span><span>' + esc(String(guided.source_passage_count || 0)) + ' 条证据片段</span><span>' + esc(String(guided.retrieval_query_count || 0)) + ' 次任务内检索</span></div>' : '') + '<div class="t1-guided-document">' + sections.map(function (section, sectionIndex) {
        var items = Array.isArray(section.items) ? section.items : [];
        var evidenceRefs = Array.isArray(section.evidence_refs) ? section.evidence_refs : [];
        var gaps = Array.isArray(section.knowledge_gaps) ? section.knowledge_gaps : [];
        var targets = Array.isArray(section.target_concepts) ? section.target_concepts : [];
        var prereqs = Array.isArray(section.prerequisites) ? section.prerequisites : [];
        var body = '<div class="t1-lesson-section-head"><span>' + String(sectionIndex + 1).padStart(2, '0') + '</span><div><h5>' + esc(section.title || '学习内容') + '</h5><small>' + esc(sectionSourceLabels[section.source_class] || '来源已记录') + '</small></div></div>' +
          (section.content ? t1RenderLongText(section.content) : '') +
          (items.length ? '<ul class="t1-claim-list">' + items.map(function (item) { return '<li><strong>' + esc(item.statement || '') + '</strong><small>' + esc(claimTypeLabels[item.claim_type] || '结论') + ' · ' + esc(supportLabels[item.support_status] || '支持状态未确认') + '</small></li>'; }).join('') + '</ul>' : '') +
          (targets.length ? '<div class="t1-concept-strip"><span>本节概念</span>' + targets.map(function (name) { return '<strong>' + esc(name) + '</strong>'; }).join('') + '</div>' : '') +
          (prereqs.length ? '<div class="t1-prerequisite-strip"><span>先修检查</span>' + prereqs.map(function (name) { return '<strong>' + esc(name) + '</strong>'; }).join('') + '</div>' : '') +
          (evidenceRefs.length ? '<p class="t1-evidence-reference"><strong>证据索引：</strong>' + esc(evidenceRefs.join('、')) + '</p>' : '') +
          (gaps.length ? '<p class="t1-resource-limit"><strong>当前限制：</strong>' + esc(gaps.join('；')) + '</p>' : '');
        if (String(section.source_class || '') === 'RELEASE_GATE') {
          return '<details class="t1-lesson-appendix"><summary>查看证据与适用边界</summary><div>' + body + '</div></details>';
        }
        return '<section class="t1-lesson-section t1-lesson-' + escAttr(String(section.section_id || 'content')) + '">' + body + '</section>';
      }).join('') + '</div></div>' : '';
      var stepHtml = steps.length ? '<div class="t1-resource-content canvas-practical-guide"><div class="canvas-reading-heading"><strong>' + esc(String(resource.title || '当前任务证据分析工作单')) + '</strong><span>' + steps.length + ' 步</span></div><ol class="t1-practical-steps">' + steps.map(function (step) {
        if (step && typeof step === 'object') return '<li><strong>' + esc(step.name || ('步骤 ' + step.step)) + '</strong><p>' + esc(step.operation || '') + '</p><small>' + esc(step.check || '') + '</small></li>';
        return '<li>' + esc(String(step)) + '</li>';
      }).join('') + '</ol></div>' : '';
      var stageHtml = stages.length ? '<div class="t1-assessment-stages">' + stages.map(function (stage) {
        var selected = String(stage.stage || '') === String(payload.stage_selection || '');
        return '<div class="' + (selected ? 'is-selected' : '') + '"><span>' + esc(stage.label || stage.stage) + '</span><strong>' + esc(stagePurposeLabels[stage.attempt_purpose] || '学习任务') + '</strong><small>' + esc(stage.use || '') + '</small></div>';
      }).join('') + '</div>' : '';
      var questionHtml = guidedQuestions.length ? '<div class="t1-guided-questions"><div><strong>检查理解</strong><span>回答后可继续追问或进入练习</span></div>' + guidedQuestions.map(function (item) {
        return '<button class="t1-guided-question" data-resource-id="' + escAttr(String(resource.resource_id || '')) + '" data-question="' + escAttr(String(item.prompt || '')) + '"><span>' + esc(item.prompt || '') + '</span><small>' + esc(item.purpose || 'GUIDED_FOLLOW_UP') + '</small></button>';
      }).join('') + '</div>' : '';
      var family = String(resource.resource_family || '');
      var form = String(resource.resource_form || '');
      var difficulty = String(resource.difficulty || '');
      return '<article class="t1-resource-card canvas-resource-panel' + (payload.recommended === true ? ' is-recommended' : '') + (resourceIndex === activeIndex ? ' active' : '') + '" role="tabpanel" data-resource-panel="' + resourceIndex + '" ' + (resourceIndex === activeIndex ? '' : 'hidden ') + 'data-resource-id="' + escAttr(String(resource.resource_id || '')) + '" data-open-action="' + (actions.indexOf('open') >= 0 ? '1' : '0') + '"><header class="canvas-resource-header"><div><span>' + esc(familyLabels[family] || '学习资源') + (payload.recommended === true ? ' · 当前优先' : '') + '</span><h4>' + esc(String(resource.title || '学习资源')) + '</h4><p>' + esc(String(resource.learner_fit_reason || '')) + '</p></div><span class="t1-source-' + escAttr(sourceType) + '">' + esc(sourceLabels[sourceType] || sourceType) + '</span></header>' +
        (payload.distribution_reason ? '<p class="t1-distribution-reason">' + esc(String(payload.distribution_reason)) + '</p>' : '') +
        '<div class="t1-resource-meta"><span>形式：' + esc(formLabels[form] || '学习资源') + '</span><span>约 ' + esc(String(resource.estimated_time_minutes || '-')) + ' 分钟</span></div>' +
        '<div class="t1-resource-meta"><span>难度：' + esc(difficultyLabels[difficulty] || '待确认') + '</span><span>来源：' + esc(sourceLabels[sourceType] || '来源未确认') + '</span></div>' +
        (concepts.length ? '<p><strong>Concept：</strong>' + esc(concepts.join('、')) + '</p>' : '') +
        longReadHtml + stepHtml + stageHtml + questionHtml +
        '<details><summary>来源与完成口径</summary><p>来源：' + esc(provenance.join(' · ') || '未提供') + '</p>' + (longSourceRefs.length ? '<p>专题证据来源：' + esc(longSourceRefs.join(' · ')) + '</p>' : '') + '<p>完成：' + esc(String(resource.completion_signal || '未定义')) + '</p></details>' +
        '<div class="t1-resource-actions">' + actions.filter(function (action) { return action !== 'ask_follow_up' && action !== 'open'; }).map(function (action) { return '<button class="btn ghost t1-resource-action" data-resource-id="' + escAttr(String(resource.resource_id || '')) + '" data-resource-action="' + escAttr(String(action)) + '" data-attempt-purpose="' + escAttr(String(payload.stage_selection || 'REQUIRED_PRACTICE').toUpperCase() === 'CHALLENGE' ? 'STAGED_ASSESSMENT' : String(payload.stage_selection || '').toUpperCase() === 'DIAGNOSTIC' ? 'DIAGNOSTIC' : 'REQUIRED_PRACTICE') + '">' + esc(actionLabels[action] || action) + '</button>'; }).join('') + '</div>' +
        '<div class="t1-resource-feedback" aria-live="polite"></div></article>';
    }).join('');
    return '<div class="t1-resource-library canvas-resource-library"><nav class="canvas-resource-tabs" role="tablist" aria-label="个性化学习资源">' + tabs + '</nav><div class="canvas-resource-stage">' + panels + '</div></div>';
  }

  function t1PublicTaskKnowledgeGraph(data) {
    var sources = Array.isArray(data.sources) ? data.sources.slice(0, 5) : [];
    var kpNames = [];
    var sourceLinks = [];
    sources.forEach(function (source, sourceIndex) {
      var names = source && Array.isArray(source.kp_names) ? source.kp_names.slice(0, 4) : [];
      names.forEach(function (name) {
        name = String(name || '').trim();
        if (!name) return;
        if (kpNames.indexOf(name) < 0) kpNames.push(name);
        sourceLinks.push({ kp: name, source: sourceIndex });
      });
    });
    kpNames = kpNames.slice(0, 8);
    sourceLinks = sourceLinks.filter(function (link) { return kpNames.indexOf(link.kp) >= 0; });
    if (!kpNames.length && !sources.length) return '<div class="r04-empty">当前任务没有可公开的 Concept、KP 或来源关系。</div>';
    var height = Math.max(230, 75 + Math.max(kpNames.length, sources.length) * 72);
    var question = String(data.task_question || data.query || '当前学习任务');
    if (question.length > 16) question = question.slice(0, 16) + '…';
    var kpY = {}, sourceY = {};
    kpNames.forEach(function (name, index) { kpY[name] = 55 + index * 72; });
    sources.forEach(function (source, index) { sourceY[index] = 55 + index * 72; });
    var lines = kpNames.map(function (name) {
      return '<g><line x1="185" y1="' + Math.round(height / 2) + '" x2="345" y2="' + kpY[name] + '"></line><text x="265" y="' + Math.round((height / 2 + kpY[name]) / 2 - 5) + '">KP 投影</text></g>';
    }).join('') + sourceLinks.map(function (link) {
      return '<g><line x1="495" y1="' + kpY[link.kp] + '" x2="650" y2="' + sourceY[link.source] + '"></line><text x="574" y="' + Math.round((kpY[link.kp] + sourceY[link.source]) / 2 - 5) + '">来源关联</text></g>';
    }).join('') + (!kpNames.length ? sources.map(function (source, index) {
      return '<g><line x1="185" y1="' + Math.round(height / 2) + '" x2="650" y2="' + sourceY[index] + '"></line><text x="420" y="' + Math.round((height / 2 + sourceY[index]) / 2 - 5) + '">公开来源</text></g>';
    }).join('') : '');
    var kpNodes = kpNames.map(function (name) {
      var label = name.length > 12 ? name.slice(0, 12) + '…' : name;
      return '<g class="canvas-kp-node"><rect x="345" y="' + (kpY[name] - 25) + '" width="150" height="50" rx="10"></rect><text x="420" y="' + (kpY[name] + 4) + '">' + esc(label) + '</text></g>';
    }).join('');
    var sourceNodes = sources.map(function (source, index) {
      var label = String((source && (source.title || source.source || source.document_id || source.chunk_id)) || ('来源 ' + (index + 1)));
      if (label.length > 15) label = label.slice(0, 15) + '…';
      return '<g class="canvas-source-node"><rect x="650" y="' + (sourceY[index] - 25) + '" width="170" height="50" rx="10"></rect><text x="735" y="' + (sourceY[index] + 4) + '">' + esc(label) + '</text></g>';
    }).join('');
    return '<div class="canvas-public-knowledge-graph"><svg class="r06-concept-svg" viewBox="0 0 850 ' + height + '" role="img" aria-label="当前任务公开 KP 与来源关系"><g class="canvas-question-node"><rect x="25" y="' + (Math.round(height / 2) - 32) + '" width="160" height="64" rx="12"></rect><text x="105" y="' + (Math.round(height / 2) + 4) + '">' + esc(question) + '</text></g>' + lines + kpNodes + sourceNodes + '</svg><p class="r04-muted">公开任务 → KP 投影 → 来源关联。该图只使用 CURRENT sources，不冒充 R06 Canonical Concept Relation。</p></div>';
  }

  function t234ConceptGraph(data) {
    var context = (data.knowledge_context && typeof data.knowledge_context === 'object') ? data.knowledge_context : {};
    var nodes = Array.isArray(context.nodes) ? context.nodes : [];
    var edges = Array.isArray(context.edges) ? context.edges : [];
    if (!nodes.length) return t1PublicTaskKnowledgeGraph(data);
    var visibleNodes = nodes.slice(0, 12);
    var nodeById = {};
    visibleNodes.forEach(function (node, index) {
      nodeById[String(node.concept_id || '')] = {
        node: node, index: index, x: 105 + (index % 4) * 190, y: 70 + Math.floor(index / 4) * 125
      };
    });
    var visibleEdges = edges.filter(function (edge) {
      return nodeById[String(edge.source || '')] && nodeById[String(edge.target || '')];
    });
    var graphHeight = Math.max(170, 135 + Math.floor((visibleNodes.length - 1) / 4) * 125);
    var edgeSvg = visibleEdges.map(function (edge) {
      var from = nodeById[String(edge.source)], to = nodeById[String(edge.target)];
      return '<g><line x1="' + from.x + '" y1="' + from.y + '" x2="' + to.x + '" y2="' + to.y + '" marker-end="url(#r06Arrow)"></line><text x="' + ((from.x + to.x) / 2) + '" y="' + (((from.y + to.y) / 2) - 6) + '">' + esc(String(edge.relation_type || 'related')) + '</text></g>';
    }).join('');
    var nodeSvg = visibleNodes.map(function (node) {
      var point = nodeById[String(node.concept_id || '')];
      var state = String(node.learner_state || 'UNKNOWN');
      var shortName = String(node.name || node.concept_id || 'Concept');
      if (shortName.length > 12) shortName = shortName.slice(0, 12) + '…';
      return '<g class="state-' + escAttr(state.toLowerCase()) + '"><rect x="' + (point.x - 74) + '" y="' + (point.y - 30) + '" width="148" height="60" rx="12"></rect><text class="r06-node-name" x="' + point.x + '" y="' + (point.y - 2) + '">' + esc(shortName) + '</text><text class="r06-node-state" x="' + point.x + '" y="' + (point.y + 17) + '">' + esc(state) + '</text></g>';
    }).join('');
    var svg = '<svg class="r06-concept-svg" viewBox="0 0 780 ' + graphHeight + '" role="img" aria-label="当前任务真实 Concept Relation 子图"><defs><marker id="r06Arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs>' + edgeSvg + nodeSvg + '</svg>';
    var nodeHtml = visibleNodes.map(function (node) {
      var state = String(node.learner_state || 'UNKNOWN');
      var stateLabel = {UNKNOWN:'状态未知',MASTERED:'已有证据支持掌握',LEARNING_GAP:'待学习'}[state] || state;
      return '<div class="r08-concept-node"><span>' + esc(String(node.role || 'CONCEPT')) + '</span><strong>' + esc(String(node.name || node.concept_id)) + '</strong><small>' + esc(stateLabel) + '</small></div>';
    }).join('');
    var edgeHtml = edges.length ? '<ul class="r04-source-list">' + edges.map(function (edge) {
      return '<li>' + esc(String(edge.source)) + ' — ' + esc(String(edge.relation_type)) + ' → ' + esc(String(edge.target)) + '</li>';
    }).join('') + '</ul>' : '<p class="r04-muted">当前公开子图没有可显示的策展关系边。</p>';
    return '<div class="r08-concept-map" aria-label="当前问题概念关系">' + svg + '<details><summary>查看节点状态与关系明细</summary><div class="r08-concept-nodes">' + nodeHtml + '</div>' + edgeHtml + '</details></div>';
  }

  function t5678ScientificGrounding(data) {
    var context = (data.knowledge_context && typeof data.knowledge_context === 'object') ? data.knowledge_context : {};
    var grounding = (context.scientific_grounding && typeof context.scientific_grounding === 'object') ? context.scientific_grounding : {};
    var claims = Array.isArray(grounding.claims) ? grounding.claims : [];
    if (grounding.status === 'WITHHELD') {
      return '<section class="r04-gap"><strong>Claim–Evidence 关系未公开</strong><p>本次结果未通过发布门，因此不展示结论文本或证据绑定。</p></section>';
    }
    if (!claims.length) {
      return '<section class="r04-empty">当前结果没有可公开的 Claim–Evidence 映射；证据片段不会被自动宣称为“支持结论”。</section>';
    }
    var labels = { SUPPORTS: '直接支持', CANDIDATE: '候选相关', MENTION: '仅提及', CONFLICTS: '存在冲突', INSUFFICIENT: '证据不足' };
    var claimHtml = claims.map(function (claim, index) {
      var support = String(claim.support_status || 'INSUFFICIENT');
      var evidences = Array.isArray(claim.evidence) ? claim.evidence : [];
      var evidenceHtml = evidences.length ? '<ul class="r04-source-list">' + evidences.map(function (item) {
        return '<li><strong>' + esc(labels[item.level] || item.level || 'UNKNOWN') + '</strong> · ' + esc(item.source || '来源未标识') + (item.chunk_id ? ' · chunk ' + esc(item.chunk_id) : '') + '<small style="display:block;color:var(--muted)">' + esc(item.reason || '') + '</small></li>';
      }).join('') + '</ul>' : '<p class="r04-muted">没有达到可公开绑定条件的证据；不作支持声明。</p>';
      return '<article style="padding:12px 0;border-bottom:1px solid var(--rule)"><div style="display:flex;gap:10px;align-items:flex-start"><span class="r04-evidence-index">' + (index + 1) + '</span><div style="min-width:0;flex:1"><div style="display:flex;justify-content:space-between;gap:12px"><strong>' + esc(claim.statement || '') + '</strong><span class="badge">' + esc(labels[support] || support) + '</span></div><p class="r04-muted">类型 ' + esc(claim.claim_type || 'UNKNOWN') + ' · 范围 ' + esc(claim.scope || 'UNKNOWN') + ' · Reviewer ' + esc(claim.reviewer_status || 'UNKNOWN') + '</p>' + evidenceHtml + '</div></div></article>';
    }).join('');
    return '<section class="card" style="margin:16px 0"><div class="r04-section-heading"><div><div class="r04-section-kicker">Scientific Grounding</div><h3>Concept → Claim → Evidence → Source</h3></div><span class="badge">' + claims.length + ' 条原子结论</span></div><p class="r04-muted">“候选相关/仅提及”不会显示为“直接支持”；条件不一致、冲突或证据不足会保持显式状态。</p>' + claimHtml + '</section>';
  }

  function t1BindResourceActions(container, data, question, queryInput) {
    function recordResourceOpen(card) {
      if (!card || card.getAttribute('data-open-action') !== '1' || card.getAttribute('data-open-recorded') === '1') return;
      card.setAttribute('data-open-recorded', '1');
      var feedback = card.querySelector('.t1-resource-feedback');
      apiReq('POST', '/api/learning/resources/interact', {
        learner_id: learnerId(), task_id: data.task_id || '',
        resource_id: card.getAttribute('data-resource-id') || '', action: 'open'
      }).then(function (result) {
        if (feedback) feedback.textContent = (result && result.message) || '已记录本次阅读。';
      }).catch(function () {
        card.removeAttribute('data-open-recorded');
        if (feedback) feedback.textContent = '阅读记录未能写入，不影响继续查看。';
      });
    }
    function continueTask(prompt, teachingAction) {
      if (queryInput) {
        queryInput.value = String(prompt || question || '');
        var askButton = g('queryAsk');
        if (askButton) askButton.click();
        return;
      }
      try { sessionStorage.setItem('dy3_pending_query', String(prompt || question || '')); } catch (ignorePendingQuery) {}
      if (teachingAction) {
        try { sessionStorage.setItem('dy3_pending_teaching_action', String(teachingAction)); } catch (ignorePendingAction) {}
      }
      if (window.sv) window.sv('query');
    }
    container.querySelectorAll('[data-resource-tab]').forEach(function (tab) {
      tab.addEventListener('click', function () {
        var library = tab.closest('.canvas-resource-library');
        if (!library) return;
        var index = tab.getAttribute('data-resource-tab');
        library.querySelectorAll('[data-resource-tab]').forEach(function (item) {
          var active = item === tab;
          item.classList.toggle('active', active);
          item.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        library.querySelectorAll('[data-resource-panel]').forEach(function (panel) {
          var active = panel.getAttribute('data-resource-panel') === index;
          panel.hidden = !active;
          panel.classList.toggle('active', active);
          if (active) recordResourceOpen(panel);
        });
      });
    });
    container.querySelectorAll('.t1-resource-card details').forEach(function (details) {
      details.addEventListener('toggle', function () {
        var card = details.closest('.t1-resource-card');
        if (details.open) recordResourceOpen(card);
      });
    });
    container.querySelectorAll('.t1-guided-question').forEach(function (button) {
      button.addEventListener('click', function () {
        var prompt = button.getAttribute('data-question') || '';
        if (!prompt) return;
        button.disabled = true;
        apiReq('POST', '/api/learning/resources/interact', {
          learner_id: learnerId(), task_id: data.task_id || '',
          resource_id: button.getAttribute('data-resource-id') || '', action: 'ask_follow_up'
        }).then(function () {
          continueTask(prompt, '');
        }).catch(function (error) {
          button.disabled = false;
          var card = button.closest('.t1-resource-card');
          var feedback = card ? card.querySelector('.t1-resource-feedback') : null;
          if (feedback) feedback.textContent = (error && error.message) || '追问暂时无法发起。';
        });
      });
    });
    container.querySelectorAll('.t1-resource-action').forEach(function (button) {
      button.addEventListener('click', function () {
        var card = button.closest('.t1-resource-card');
        var feedback = card ? card.querySelector('.t1-resource-feedback') : null;
        button.disabled = true;
        apiReq('POST', '/api/learning/resources/interact', {
          learner_id: learnerId(), task_id: data.task_id || '',
          resource_id: button.getAttribute('data-resource-id') || '',
          action: button.getAttribute('data-resource-action') || ''
        }).then(function (result) {
          if (feedback) feedback.textContent = (result && result.message) || '已记录本次交互。';
          var action = button.getAttribute('data-resource-action');
          if (action === 'start_practice') {
            try { sessionStorage.setItem('dy3_practice_target_kps', JSON.stringify((result && result.target_kps) || [])); } catch (ignoreTargets) {}
            try { sessionStorage.setItem('dy3_practice_context', JSON.stringify({ task_id: data.task_id || '', resource_id: button.getAttribute('data-resource-id') || '' })); } catch (ignorePracticeContext) {}
            try { sessionStorage.setItem('dy3_practice_attempt_purpose', button.getAttribute('data-attempt-purpose') || 'REQUIRED_PRACTICE'); } catch (ignoreAttemptPurpose) {}
            if (window.sv) window.sv('practice');
          }
          if (result && result.next_teaching_action) {
            continueTask(question, result.next_teaching_action);
          }
        }).catch(function () {
          if (feedback) feedback.textContent = '交互未能写入，请重试。';
        }).finally(function () { button.disabled = false; });
      });
    });
  }

  function renderR04TaskResult(payload, question, container, queryInput) {
    var data = r04PublicData(payload);
    var quality = (data.quality_release && typeof data.quality_release === 'object') ? data.quality_release : {};
    var releaseAllowed = Boolean(quality.eligible) && (quality.status === 'FULL_RELEASE' || quality.status === 'LIMITED_RELEASE');
    var answer = releaseAllowed ? String(data.answer || '') : '';
    var evidence = Array.isArray(data.evidence) ? data.evidence : [];
    var sources = Array.isArray(data.sources) ? data.sources : [];
    var review = (data.review && typeof data.review === 'object') ? data.review : {};
    var path = Array.isArray(data.recommended_path) ? data.recommended_path : [];
    var taskMode = data.question_type || data.task_mode || data.action_type || '未标注';
    var explanationGoal = data.action_type || '未公开';
    var taskState = data.task_state || '未标注';
    var taskId = data.task_id || '';
    var verdict = review.verdict || review.status || '未提供';
    var reviewReason = review.reason || review.message || review.summary || '';
    var knowledgeGap = Boolean(data.knowledge_unavailable) || /knowledge_gap/i.test(String(data.action_type || ''));
    var clarify = data.clarify;
    var publicLearner = (data.learner_context && typeof data.learner_context === 'object') ? data.learner_context : {};
    var diagnosisTrace = (Array.isArray(data.agent_trace) ? data.agent_trace : []).filter(function (event) {
      return event && event.agent_id === 'agent.learning.diagnosis';
    })[0] || {};
    var learnerLevel = (publicLearner.teaching_decision || {}).content_depth || publicLearner.lifecycle_stage || diagnosisTrace.detail || '未公开';

    r08TaskTruth = {
      learner_id: learnerId(), loaded: true, loading: false,
      data: data, question: String(question || ''), tasks: r08TaskTruth.tasks || []
    };
    var journeyHost = g('r08JourneyHost');
    if (journeyHost) {
      journeyHost.innerHTML = r08Journey('query', data);
      r08BindJourney(journeyHost);
    }
    var latestTaskContext = g('r08LatestTaskContext');
    if (latestTaskContext) {
      latestTaskContext.innerHTML = r08LatestTaskContext(data, question);
      r08BindJourney(latestTaskContext);
    }

    if (window.DY3ProductCanvas && typeof window.DY3ProductCanvas.renderTaskResult === 'function') {
      window.DY3ProductCanvas.renderTaskResult(data, question, container, queryInput);
      return;
    }

    if (clarify && !answer) {
      var options = Array.isArray(clarify.options) ? clarify.options.slice(0, 6) : [];
      container.innerHTML = '<article class="r04-task-shell"><section class="r04-question-card"><div class="r04-section-kicker">任务理解</div><h2>' + esc(question) + '</h2></section>' +
        t1TeachingStrategy(data) + t1QualityBlock(data) +
        '<section class="r04-clarify"><h3>需要补充信息</h3><p>' + esc(clarify.question || clarify.guidance || '请补充问题范围。') + '</p>' +
        '<div class="r04-chip-row">' + options.map(function (option) { return '<button class="q-clr-chip" data-opt="' + escAttr(option) + '">' + esc(option) + '</button>'; }).join('') + '</div></section></article>';
      container.querySelectorAll('.q-clr-chip').forEach(function (button) {
        button.addEventListener('click', function () { queryInput.value = question + ' ' + button.getAttribute('data-opt'); queryInput.focus(); });
      });
      return;
    }

    var evidenceItems = evidence.slice(0, 5).map(function (item, index) {
      var body = r04Text(item);
      var source = (item && typeof item === 'object') ? (item.source || item.title || item.chunk_id || '') : '';
      var claim = (item && typeof item === 'object') ? (item.claim || item.supported_claim || '') : '';
      return '<li><div class="r04-evidence-index">' + (index + 1) + '</div><div>' +
        '<div class="r04-evidence-claim"><strong>' + (claim ? '关联结论' : '支持范围') + '：</strong>' + esc(claim || '最终回答（CURRENT 未提供 claim 级绑定）') + '</div>' +
        '<p>' + esc(body || '证据内容未提供') + '</p>' +
        (source ? '<span>来源：' + esc(String(source)) + '</span>' : '') + '</div></li>';
    }).join('');
    var sourceItems = sources.slice(0, 5).map(function (item) {
      return '<li>' + r04SourceReference(item) + '</li>';
    }).join('');
    var pathItems = path.slice(0, 6).map(function (item) { return '<li>' + esc(r04Text(item) || '继续当前学习任务') + '</li>'; }).join('');
    var evidenceBody = evidenceItems ? '<ol class="r04-evidence-list">' + evidenceItems + '</ol>' :
      (knowledgeGap ? '<div class="r04-gap">当前知识库缺少支撑该结论所需的充分证据，系统没有生成无依据结论。</div>' : '<div class="r04-empty">本次响应未提供可展示证据。</div>');

    container.innerHTML = '<article class="r04-task-shell canvas-page canvas-task-workspace">' +
      '<section class="r04-question-card"><div class="r04-section-kicker">用户任务</div><h2>' + esc(question) + '</h2>' +
      '<div class="r04-task-meta"><span>状态：' + esc(String(taskState)) + '</span>' + (taskId ? '<span>任务：' + esc(String(taskId)) + '</span>' : '') + '</div>' +
      '<div class="r04b-understanding"><div><strong>任务类型</strong><span>' + esc(String(taskMode)) + '</span></div>' +
      '<div><strong>解释目标</strong><span>' + esc(String(explanationGoal)) + '</span></div>' +
      '<div><strong>学习层级 / 诊断</strong><span>' + esc(String(learnerLevel)) + '</span></div></div></section>' +
      r08TaskFactFlow(data, learnerLevel, evidence, verdict) +
      '<div class="canvas-task-grid"><main class="canvas-task-main">' +
      '<section class="r04-answer-card"><div class="r04-section-kicker">个性化回答</div><h3>当前解释</h3>' +
      (answer ? '<div class="r04-answer-body">' + renderSlideDeck(answer) + '</div>' : '<div class="r04-gap">未审核通过的科学草稿不会展示。</div>') + '</section>' +
      '<section class="t1-resources"><div class="r04-section-kicker">个性化学习资源</div><h3>讲义、实操指南与分阶测试</h3>' + t1ResourceCards(data) + '</section></main>' +
      '<aside class="canvas-task-rail">' + t1TeachingStrategy(data) + t1QualityBlock(data) +
      '<section class="r04-review-card"><div class="r04-section-heading"><div><div class="r04-section-kicker">科学审核</div><h3>Reviewer 结论</h3></div><span class="r04-review-verdict">' + esc(String(verdict)) + '</span></div>' +
      (reviewReason ? '<p>' + esc(String(reviewReason)) + '</p>' : '<p class="r04-muted">未提供公开审核说明。</p>') + '</section>' + r04RenderChallenge(data) +
      '<section class="r04-evidence-card"><div class="r04-section-kicker">科学证据</div><h3>这个回答依据什么</h3>' + evidenceBody +
      (sourceItems ? '<details><summary>查看来源关联</summary><ul class="r04-source-list">' + sourceItems + '</ul></details>' : '') + '</section>' +
      '<section class="r04-evidence-card"><div class="r04-section-kicker">Knowledge Concept</div><h3>概念与先修关系</h3>' + t234ConceptGraph(data) + '</section>' +
      '<details class="r04-collaboration"><summary><strong>协同记录</strong></summary>' + r04RenderTrace(data) + '</details>' +
      '<section class="r04-next-card"><div class="r04-section-kicker">下一步学习</div><h3>继续做什么</h3>' +
      (pathItems ? '<ol>' + pathItems + '</ol>' : '<p class="r04-muted">本次响应未提供下一步学习路径。</p>') + '</section></aside></div>' +
      '<button class="btn ghost" id="queryAgain" type="button">继续提问</button></article>';
    r08BindJourney(container);
    t1BindResourceActions(container, data, question, queryInput);
    // /api/query has completed the server-side LearningTask write. Refresh
    // from that authoritative store so cross-space continuation never shows
    // a recent task beside an empty history panel.
    r08RefreshTaskHistory(g('content'));
    var again = g('queryAgain');
    if (again) again.addEventListener('click', function () { queryInput.value = ''; queryInput.focus(); });
  }

  function renderQuery() {
    var ct = g('content');
    if (!ct) return;
    var lastTask = r08StoredTask();
    var lastQuestion = r08StoredQuestion();
    if (!r08TaskTruth.loaded && !r08TaskTruth.loading) {
      r08LoadTaskTruth(function () {
        if (currentView() === 'query') renderQuery();
      });
    }
    // 统一会话入口: 创建/复用 L1 query 会话 (localStorage 记忆)
    var l1Key = 'dy3_query_session_' + learnerId();
    var getL1 = function () {
      return localStorage.getItem(l1Key) || '';
    };
    var ensureL1 = function () {
      var sid = getL1();
      if (sid) return Promise.resolve(sid);
      return apiReq('POST', '/l1/api/v1/sessions', { session_type: 'query' })
        .then(function (s) {
          var nsid = (s && s.session_id) || '';
          if (nsid) localStorage.setItem(l1Key, nsid);
          return nsid;
        })
        .catch(function () { return ''; });
    };
    ct.innerHTML = '<div id="r08JourneyHost">' + r08Journey('query', lastTask) + '</div><div class="r08-core-task canvas-page canvas-query-launch"><div class="r08b3-task-launch"><section class="r08b3-task-entry"><div class="r04-section-kicker">TASK WORKSPACE</div><h2>任务工作区</h2>' +
      '<div id="r08LatestTaskContext">' + r08LatestTaskContext(lastTask, lastQuestion) + '</div>' +
      '<div class="r08b3-query-box"><input id="queryInput" type="text" placeholder="输入你的问题…" list="queryHistoryList"><button class="btn primary" id="queryAsk">开始分析</button></div>' +
      '<datalist id="queryHistoryList"></datalist>' +
      '<div class="r08b3-preset-row"><span>推荐问题</span>' +
      ['为什么 Dy³⁺ 浓度增加会导致发光下降？', '为什么 Dy³⁺ 会产生黄蓝双发射？', '3000 K 是否一定更适合健康照明？', '如何公平比较两种 Dy³⁺ 发光材料？'].map(function (p) {
        return '<button class="mf7-chip q-preset" data-q="' + esc(p) + '">' + esc(p.slice(0, 18)) + (p.length > 18 ? '…' : '') + '</button>';
      }).join('') + '</div></section>' +
      '<aside class="r08b3-learning-mode" aria-label="回答方式"><div class="r04-section-kicker">本次方式</div><h3>回答方式</h3><label><input type="radio" name="teachingMode" value="" checked><span><strong>自适应</strong><small>依据已有画像、作答与学习记录决定</small></span></label><label><input type="radio" name="teachingMode" value="still_confused"><span><strong>基础拆解</strong><small>补前置概念并降低表达复杂度</small></span></label><label><input type="radio" name="teachingMode" value="request_example"><span><strong>案例引导</strong><small>先用例子建立机制联系</small></span></label><label><input type="radio" name="teachingMode" value="deepen"><span><strong>证据深入</strong><small>增加机制、条件边界与证据讨论</small></span></label></aside></div>' +
      '<div id="queryResult"></div>' + r08TaskHistoryPanel() +
      '<div id="queryLog" class="r08b3-query-history"></div></div>';
    r08BindJourney(ct);
    r08BindTaskHistory(ct);
    ct.querySelectorAll('.q-preset').forEach(function (b) {
      b.addEventListener('click', function () { inp.value = b.dataset.q; run(); });
    });
    // 历史查询下拉 (最近 10 条)
    var qHist = JSON.parse(localStorage.getItem('mf7_query_history') || '[]');
    var dl = g('queryHistoryList');
    if (dl) {
      qHist.slice(-10).reverse().forEach(function (h) {
        var o = d.createElement('option');
        o.value = h;
        dl.appendChild(o);
      });
    }
    var qLogBox = g('queryLog');
    (qHist.slice(-5).reverse()).forEach(function (h) {
      if (qLogBox) qLogBox.innerHTML += '<div style="font-size:12px;color:var(--muted);margin:2px 0">↺ ' + esc(h) + '</div>';
    });
    var inp = g('queryInput');
    var btn = g('queryAsk');
    var pendingQuery = '';
    try { pendingQuery = sessionStorage.getItem('dy3_pending_query') || ''; sessionStorage.removeItem('dy3_pending_query'); } catch (ignorePending) {}
    if (inp && pendingQuery) inp.value = pendingQuery;
    // Returning to the task canvas must restore the latest released task from
    // LearningTaskStore. This is the same public result, not a client cache or
    // a second generation path.
    if (lastTask && lastTask.task_id && lastQuestion && !pendingQuery) {
      renderR04TaskResult({ data: lastTask }, lastQuestion, g('queryResult'), inp);
    }
    var run = function () {
      var q = (inp.value || '').trim();
      if (!q) return;
      // 记录历史
      var hist = JSON.parse(localStorage.getItem('mf7_query_history') || '[]');
      if (hist[hist.length - 1] !== q) { hist.push(q); localStorage.setItem('mf7_query_history', JSON.stringify(hist.slice(-10))); }
      var res = g('queryResult');
      var log = g('queryLog');
      // 同步 API 只允许诚实等待；真实 trace 在响应返回后展示。
      var stepTimer = null;
      res.innerHTML = '<div class="r04-processing" role="status"><span class="spinner"></span><div><strong>任务处理中</strong><p>结果返回后可查看真实 Agent 协同过程。</p></div></div>';
      // 学情采集: 提问事件入画像
      apiReq('POST', '/l2/event/collect', { learner_id: learnerId(), event_type: 'query', detail: q }).catch(function () {});
      ensureL1().then(function (l1id) {
        // 统一会话: 提问携带 L1 会话 ID → /api/query 自动创建并关联 L5 执行会话
        var teachingAction = '';
        try { teachingAction = sessionStorage.getItem('dy3_pending_teaching_action') || ''; sessionStorage.removeItem('dy3_pending_teaching_action'); } catch (ignoreAction) {}
        if (!teachingAction) {
          var selectedMode = ct.querySelector('input[name="teachingMode"]:checked');
          teachingAction = selectedMode ? String(selectedMode.value || '') : '';
        }
        apiReq('POST', '/api/query', { query: q, learner_id: learnerId(), session_id: l1id, context: { teaching_action: teachingAction } }).then(function (d) {
          renderR04TaskResult(d, q, res, inp);
          return;
          clearInterval(stepTimer); // 停止加载动画, 防止覆盖答案
          var answer = d && d.answer ? d.answer : ((d && d.data && d.data.answer) || '');
          var conf = d && d.confidence != null ? d.confidence : ((d && d.data && d.data.confidence) || '-');
          var reqConf = d && d.requires_confirmation;
          var noKB = (d && d.knowledge_unavailable) || (d && d.data && d.data.knowledge_unavailable);
          var review = (d && d.review) || (d && d.data && d.data.review) || {};
          var verdict = review.verdict || '-';
          var sess = (d && d.session) || (d && d.data && d.data.session) || {};
           var agentSid = sess.session_id || '';
           var pipe = (d && d.pipeline) || (d && d.data && d.data.pipeline) || [];
           var pipeHtml = pipe.map(function (p) {
             return '<span class="badge info" style="margin:2px">' + esc(p.step) + (p.detail ? ' · ' + esc(String(p.detail).slice(0, 40)) : '') + '</span>';
           }).join(' ');
           var trace = (d && d.agent_trace) || (d && d.data && d.data.agent_trace) || [];
            var traceHtml = trace.length ? ('<div style="margin-top:10px"><div style="font-size:12px;color:var(--muted);margin-bottom:6px">Agent 执行记录（本问答 4 个 Agent 协作）</div>' +
              trace.map(function (t) {
                var tsec = Number(t.time || 0);
                if (tsec > 1e12) tsec = tsec / 1000;
                var tsl = tsec ? new Date(tsec * 1000).toLocaleTimeString('zh-CN') : '-';
                var nm = String(t.agent_id || '').replace('agent.', '');
                return '<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;border:1px solid var(--rule);border-radius:6px;margin-bottom:4px;font-size:12px;background:var(--surface)">' +
                  '<span style="flex-shrink:0">✅</span>' +
                  '<span style="flex-shrink:0;font-weight:600;min-width:150px">' + esc(nm) + '</span>' +
                  '<span style="color:var(--muted);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(String(t.detail || '').slice(0, 70)) + '</span>' +
                  '<span style="color:var(--muted);font-size:11px;flex-shrink:0">' + esc(tsl) + '</span></div>';
              }).join('') + '</div>') : '';
            var evd = (d && d.evidence) || (d && d.data && d.data.evidence) || [];
            var evdHtml = evd.length ? ('<div style="margin-top:10px"><div style="font-size:12px;color:var(--muted);margin-bottom:6px">知识证据（来源切片）</div>' +
              evd.slice(0, 3).map(function (e) {
                var src = e.source || e.title || e.chunk_id || '';
                return '<div style="padding:6px 10px;border-left:3px solid #8b5cf6;background:var(--surface);border-radius:4px;margin-bottom:4px;font-size:12px">' +
                  esc(String(e.content || e.text || '').slice(0, 120)) +
                  (src ? '<div style="margin-top:3px;font-size:10px;color:#8b5cf6">来源: ' + esc(String(src).slice(0, 40)) + '</div>' : '') + '</div>';
              }).join('') + '</div>') : '';
            // 知识点溯源: 答案依据的知识点 (后端按证据切片推断 kp_id/kp_name), 突出展示
            var srcs = (d && d.sources) || (d && d.data && d.data.sources) || [];
            var srcHtml = srcs.length ? ('<div style="margin-top:10px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);overflow:hidden">' +
              '<div style="display:flex;align-items:center;gap:6px;padding:6px 10px;background:linear-gradient(135deg,rgba(139,92,246,.10),rgba(14,165,233,.08));border-bottom:1px solid var(--rule)">' +
              '<span style="font-size:12px;font-weight:600">📍 知识点溯源</span>' +
              '<span style="font-size:10px;color:var(--muted)">答案依据的知识点（按证据切片推断）</span></div>' +
              '<div style="padding:8px 10px">' +
              srcs.slice(0, 4).map(function (s) {
                var names = (s.kp_names && s.kp_names.length) ? s.kp_names : [];
                var ids = s.kp_ids || [];
                var kpTags = names.map(function (n, i) {
                  return '<span style="display:inline-block;background:rgba(139,92,246,.14);color:var(--violet,#8b5cf6);border-radius:4px;padding:1px 7px;margin:1px 4px 1px 0;font-size:10.5px">' + esc(String(n)) + (ids[i] ? ' <span style="opacity:.6">' + esc(String(ids[i])) + '</span>' : '') + '</span>';
                }).join('');
                return '<div style="padding:6px 10px;border-left:3px solid #8b5cf6;background:var(--surface2);border-radius:4px;margin-bottom:5px">' +
                  '<div style="margin-bottom:3px">' + kpTags + '</div>' +
                  '<div style="font-size:12px;color:var(--ink);line-height:1.5">' + esc(String(s.excerpt || '').slice(0, 160)) + '</div>' +
                  (s.section ? '<div style="margin-top:3px;font-size:10px;color:var(--muted)">章节: ' + esc(String(s.section).slice(0, 40)) + '</div>' : '') + '</div>';
              }).join('') + '</div></div>') : '';
            // 多线协作 + 协同决策中间数据可视化 (L5 高等级对标: 中间数据/流程可视化)
            var clines = (d && d.collab_lines) || (d && d.data && d.data.collab_lines) || [];
            var sc = (d && d.self_correction) || (d && d.data && d.data.self_correction) || null;
            var cands = (d && d.candidates) || (d && d.data && d.data.candidates) || [];
            var cs = (d && d.consensus_score) || (d && d.data && d.data.consensus_score) || 0;
            var csReached = (d && d.consensus_reached) || (d && d.data && d.data.consensus_reached) || false;
            var csTh = (d && d.consensus_threshold) || (d && d.data && d.data.consensus_threshold) || 0.5;
            var dm = (d && d.divergence_matrix) || (d && d.data && d.data.divergence_matrix) || [];
            var deb = (d && d.debate) || (d && d.data && d.data.debate) || null;
            var needAdj = (d && d.needs_adjudication) || (d && d.data && d.data.needs_adjudication) || false;
            var finalAnswer = String((d && d.answer) || (d && d.data && d.data.answer) || '').trim();
            var selectedCand = '';
            cands.forEach(function (c) { if (String(c.answer || '').trim() === finalAnswer) selectedCand = String(c.candidate_id); });
            var kindColor = { candidate: '#8b5cf6', consensus: '#6366f1', debate: '#f59e0b', adjudication: '#ef4444', correction: '#10b981' };
            var kindName = { candidate: '并行候选', consensus: '交叉验证', debate: '协同辩论', adjudication: '待裁决', correction: '自纠回流' };
            // ① 协同决策中间数据面板: 共识度 + 候选置信度 + 分歧度矩阵 + 辩论中间过程
            var midHtml = '';
            if (cands.length || dm.length || deb) {
              var csPct = Math.round(Number(cs) * 100);
              var csColor = (csReached ? '#10b981' : (needAdj ? '#ef4444' : '#f59e0b'));
              var gauge = '<div style="flex-shrink:0;text-align:center;padding:6px 12px;border-right:1px dashed var(--rule)">' +
                '<div style="font-size:20px;font-weight:700;color:' + csColor + '">' + csPct + '%</div>' +
                '<div style="font-size:10px;color:var(--muted)">共识度 ≥' + Math.round(Number(csTh) * 100) + '%</div></div>';
              var candBar = '<div style="flex:1;min-width:130px;padding:4px 10px">' +
                '<div style="font-size:10px;color:var(--muted);margin-bottom:4px">候选答案置信度（多策略并行）</div>' +
                cands.map(function (c) {
                  var w = Math.round(Number(c.confidence || 0) * 100);
                  var sel = String(c.candidate_id) === selectedCand;
                  return '<div style="margin-bottom:3px;font-size:10px">' +
                    '<span style="display:inline-block;width:78px;color:var(--muted)">' + esc(c.candidate_id) + ' ' + esc(c.label || '') + (sel ? ' ✓' : '') + '</span>' +
                    '<span style="display:inline-block;width:90px;height:8px;background:var(--surface);border-radius:4px;vertical-align:middle;overflow:hidden">' +
                    '<span style="display:block;height:100%;width:' + w + '%;background:' + (sel ? 'linear-gradient(135deg,#10b981,#059669)' : 'linear-gradient(135deg,#8b5cf6,#6366f1)') + '"></span></span>' +
                    '<span style="margin-left:4px;color:' + (sel ? '#059669' : 'var(--muted)') + '">' + w + '%</span></div>';
                }).join('') + '</div>';
              var dmHtml = '';
              if (dm.length && dm[0] && dm[0].length) {
                var ids = cands.length ? cands.map(function (c) { return c.candidate_id; }) : dm.map(function (_, i) { return String.fromCharCode(65 + i); });
                dmHtml = '<div style="flex-shrink:0;padding:4px 10px;border-left:1px dashed var(--rule)">' +
                  '<div style="font-size:10px;color:var(--muted);margin-bottom:4px">两两分歧度</div>' +
                  '<table style="border-collapse:collapse;font-size:10px;font-family:var(--mono)"><tr><td style="padding:1px 4px"></td>' +
                  ids.map(function (id) { return '<td style="padding:1px 4px;color:var(--muted);text-align:center">' + esc(String(id)) + '</td>'; }).join('') + '</tr>' +
                  dm.map(function (row, i) {
                    return '<tr><td style="padding:1px 4px;color:var(--muted);text-align:center">' + esc(String(ids[i])) + '</td>' +
                      row.map(function (v) {
                        var vn = Number(v); if (isNaN(vn)) vn = 0;
                        var bg = vn < 0.2 ? 'rgba(16,185,129,.18)' : (vn < 0.5 ? 'rgba(245,158,11,.18)' : 'rgba(239,68,68,.22)');
                        return '<td style="padding:1px 4px;text-align:center;border-radius:3px;background:' + bg + '">' + Math.round(vn * 100) + '</td>';
                      }).join('') + '</tr>';
                  }).join('') + '</table></div>';
              }
              var debHtml = '';
              if (deb) {
                var db = Number(deb.divergence_before || 0), da = Number(deb.divergence_after || 0);
                var proId = (deb.pro && deb.pro.candidate_id) || '-', conId = (deb.con && deb.con.candidate_id) || '-';
                debHtml = '<div style="flex-shrink:0;padding:4px 10px;border-left:1px dashed var(--rule);min-width:130px">' +
                  '<div style="font-size:10px;color:var(--muted);margin-bottom:4px">协同辩论（' + Math.round(Number(deb.rounds || 0)) + ' 轮）</div>' +
                  '<div style="font-size:10px">正方 <b style="color:#8b5cf6">' + esc(String(proId)) + '</b> vs 反方 <b style="color:#f59e0b">' + esc(String(conId)) + '</b></div>' +
                  '<div style="font-size:10px;color:var(--muted)">分歧 ' + Math.round(db * 100) + '% → ' + Math.round(da * 100) + '%</div>' +
                  '<div style="font-size:10px;color:' + (deb.converged ? '#10b981' : '#ef4444') + '">' + (deb.converged ? '✅ 已收敛·采纳胜方' : '⚠ 未收敛') + '</div></div>';
              }
              var statusChip = needAdj ? '<span style="font-size:10px;color:#ef4444;border:1px solid #ef4444;border-radius:20px;padding:1px 8px">待裁决</span>' : (csReached ? '<span style="font-size:10px;color:#10b981;border:1px solid #10b981;border-radius:20px;padding:1px 8px">达成共识</span>' : '<span style="font-size:10px;color:#f59e0b;border:1px solid #f59e0b;border-radius:20px;padding:1px 8px">未达共识</span>');
              midHtml = '<div style="margin-top:10px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);overflow:hidden">' +
                '<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:linear-gradient(135deg,rgba(99,102,241,.1),rgba(139,92,246,.1));border-bottom:1px solid var(--rule)">' +
                '<span style="font-size:12px;font-weight:600">🧩 协同决策中间数据</span>' + statusChip +
                '<span style="flex:1"></span><span style="font-size:10px;color:var(--muted)">' + cands.length + ' 候选 · ' + (dm.length ? '分歧矩阵' : '-') + ' · ' + (deb ? '辩论' : '-') + '</span></div>' +
                '<div style="display:flex;flex-wrap:wrap;align-items:stretch">' + gauge + candBar + dmHtml + debHtml + '</div></div>';
            }
            // ② 多线协作时序图 (主线 + 并行候选线 + 交叉验证 + 辩论 + 待裁决 + 自纠回流)
            var clinesHtml = '';
            if (clines.length) {
              clinesHtml = '<div style="margin-top:10px"><div style="font-size:12px;color:var(--muted);margin-bottom:6px">多线协作时序（' + clines.length + ' 条线：主线 + 并行候选 + 交叉验证' +
                (deb ? ' + 协同辩论' : '') + (needAdj ? ' + 待裁决' : '') + (sc ? ' + 自纠' : '') + '）</div>';
              clines.forEach(function (cl) {
                var kc = kindColor[cl.kind] || '#6366f1';
                var kn = kindName[cl.kind] || '协作';
                var steps = (cl.steps || []).map(function (st) {
                  var ms = st.elapsed_ms != null ? Math.round(Number(st.elapsed_ms)) + 'ms' : '-';
                  var nm = String(st.agent || '').replace('agent.', '');
                  return '<div style="display:inline-flex;flex-direction:column;align-items:center;margin:0 2px">' +
                    '<span style="padding:1px 6px;border-radius:4px;font-size:10px;white-space:nowrap;background:' + kc + ';color:#fff;opacity:.95">' + esc(nm) + '</span>' +
                    '<span style="font-size:9px;color:var(--muted);margin-top:2px;white-space:nowrap;max-width:130px;overflow:hidden;text-overflow:ellipsis">' + esc(String(st.output || '').slice(0, 16)) + ' · ' + ms + '</span></div>';
                }).join('<span style="color:var(--muted);margin:0 1px">→</span>');
                clinesHtml += '<div style="display:flex;align-items:center;gap:6px;padding:5px 8px;border:1px solid var(--rule);border-left:3px solid ' + kc + ';border-radius:6px;margin-bottom:4px;font-size:11px;background:var(--surface)">' +
                  '<div style="flex-shrink:0;min-width:96px"><div style="font-weight:600;font-size:10px;color:' + kc + '">' + esc(cl.line) + ' ' + esc(cl.label || '') + '</div>' +
                  '<div style="font-size:9px;color:var(--muted)">' + kn + '</div></div>' +
                  '<div style="flex:1;text-align:center;overflow-x:auto;white-space:nowrap">' + steps + '</div></div>';
              });
              clinesHtml += '</div>';
            }
            // Four-Agent flow and collaboration projection.
            var flowEv = (d && d.flow_events) || (d && d.data && d.data.flow_events) || [];
            var bcEv = (d && d.broadcast_events) || (d && d.data && d.data.broadcast_events) || [];
            var qtype = (d && d.question_type) || (d && d.data && d.data.question_type) || '';
            var flowHtml = '';
            if (flowEv.length) {
              var AGENT_STYLE = {
                'agent.learning.diagnosis': { name: '学情诊断', color: '#0ea5e9', icon: '🔍' },
                'agent.knowledge.generation': { name: '知识生成', color: '#8b5cf6', icon: '🧠' },
                'agent.quality.review': { name: '审核校验', color: '#f59e0b', icon: '🛡️' },
                'agent.guidance.decision': { name: '导学决策', color: '#10b981', icon: '🎯' }
              };
              function agentBadge(agentId) {
                var a = AGENT_STYLE[agentId] || { name: String(agentId || '').replace('agent.', ''), color: '#6366f1', icon: '🤖' };
                return '<span style="display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:999px;background:' + a.color + ';color:#fff;font-size:11px;font-weight:600;white-space:nowrap">' + a.icon + ' ' + esc(a.name) + '</span>';
              }
              var flowCells = flowEv.map(function (fe, i) {
                var ms = fe.elapsed_ms != null ? Math.round(Number(fe.elapsed_ms)) + 'ms' : '-';
                var arrow = i < flowEv.length - 1 ? '<div style="display:flex;flex-direction:column;align-items:center;padding:0 6px;color:var(--muted)"><span style="font-size:16px">→</span><span style="font-size:9px;white-space:nowrap">' + esc(String(fe.to || '').replace('agent.', '')) + '</span></div>' : '';
                return '<div style="display:flex;align-items:center;flex-wrap:nowrap">' +
                  '<div style="display:flex;flex-direction:column;gap:3px;align-items:center;min-width:86px">' +
                  agentBadge(fe.agent) +
                  '<span style="font-size:9px;color:var(--muted);white-space:nowrap">' + esc(String(fe.step || '')) + ' · ' + ms + '</span></div>' + arrow + '</div>';
              }).join('');
              var bcHtml = bcEv.length ? bcEv.map(function (bc) {
                return '<div style="display:flex;align-items:center;gap:8px;padding:5px 10px;border:1px solid var(--rule);border-radius:6px;margin-bottom:4px;font-size:11px;background:var(--surface)">' +
                  '<span style="font-size:13px">📡</span>' +
                  '<span style="font-weight:600;color:#6366f1;flex-shrink:0">' + esc(String(bc.publisher || '').replace('agent.', '')) + '</span>' +
                  '<span style="color:var(--muted)">→</span>' +
                  '<code style="background:var(--surface2);padding:1px 6px;border-radius:4px;font-size:10px;color:var(--violet)">' + esc(bc.channel || '') + '</code>' +
                  '<span style="color:var(--muted);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(String(bc.event || '')) + '</span>' +
                  '<span style="color:var(--muted);font-size:10px;flex-shrink:0">⇥ ' + esc(String(bc.to || '')) + '</span></div>';
              }).join('') : '<div style="font-size:11px;color:var(--muted);padding:4px 2px">本次问答未触发广播事件</div>';
              var qtypeChip = qtype ? '<span class="qtype-chip">意图: ' + esc(qtype) + '</span>' : '';
              flowHtml = '<div style="margin-top:10px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);overflow:hidden">' +
                '<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:linear-gradient(135deg,rgba(14,165,233,.08),rgba(16,185,129,.08));border-bottom:1px solid var(--rule)">' +
                '<span style="font-size:12px;font-weight:600">🔄 四 Agent 协同链路</span>' + qtypeChip +
                '<span style="flex:1"></span><span style="font-size:10px;color:var(--muted)">流程 → 指向 → 广播 → 协作</span></div>' +
                '<div style="padding:10px;overflow-x:auto"><div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:nowrap;min-width:560px">' + flowCells + '</div></div>' +
                '<div style="border-top:1px dashed var(--rule);padding:8px 10px"><div style="font-size:10px;color:var(--muted);margin-bottom:5px">📡 消息总线广播（可订阅协作）</div>' + bcHtml + '</div></div>';
            }
            // 汇总 Agent 协作过程/证据到可折叠区 (默认收起, 减少页面噪音)
            var collabHtml = (midHtml || clinesHtml || traceHtml || evdHtml || flowHtml) ?
              '<details style="margin-top:10px"><summary style="cursor:pointer;font-size:12px;color:var(--muted)">🔍 查看 Agent 协作过程</summary>' +
              flowHtml + midHtml + clinesHtml + traceHtml + evdHtml + '</details>' : '';
           // 模糊问题 → 人性化引导式澄清 (先复述理解 → 可点选方向 → 引导补全, 作为对用户问题的自然补充)
            var clarify = (d && d.clarify) || (d && d.data && d.data.clarify) || null;
            if (clarify && !finalAnswer) {
              clearInterval(stepTimer);
              var cq = String(clarify.question || '');
              var cg = String(clarify.guidance || '');
              var copts = (clarify.options || []).slice(0, 6);
              var cchips = copts.map(function (o) {
                return '<button class="q-clr-chip" data-opt="' + esc(o) + '">' + esc(o) + '</button>';
              }).join('');
              res.innerHTML = '<div class="q-clr">' +
                '<div class="clr-q" style="font-size:13px">💡 ' + cq + '</div>' +
                (cg ? '<div class="clr-g">' + esc(cg) + '</div>' : '') +
                '<div style="display:flex;flex-wrap:wrap;gap:8px">' + cchips + '</div></div>' +
                '' +
                (pipeHtml ? '<div>' + pipeHtml + '</div>' : '') +
                collabHtml +
                '<button class="btn ghost" id="queryAgain" style="margin-top:10px;font-size:12px">继续提问</button>';
              res.querySelectorAll('.q-clr-chip').forEach(function (cp) {
                cp.addEventListener('click', function () {
                  var dir = cp.getAttribute('data-opt');
                  inp.value = q.trim() + ' ' + String(dir).trim();
                  run();
                });
              });
              var agc = g('queryAgain');
              if (agc) agc.addEventListener('click', function () { inp.value = ''; inp.focus(); });
              log.innerHTML = log.innerHTML +
                '<div style="font-size:12px;color:var(--muted);margin:4px 0">Q: ' + esc(q) + ' → 需要补充方向（已引导）</div>';
              return;
            }
            if (noKB) {
             res.innerHTML = '<div style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);padding:12px;border-radius:8px;margin-bottom:8px">' +
              '<div style="font-size:12px;color:var(--warn,#b45309);margin-bottom:6px">⚠ 知识库暂无匹配知识（未编造作答）</div>' +
              '<div style="white-space:pre-wrap;font-size:14px;color:var(--ink)">' + esc(answer) + '</div></div>' +
              '' +
              (pipeHtml ? '<div>' + pipeHtml + '</div>' : '') +
              collabHtml +
              '<button class="btn ghost" id="queryAgain" style="margin-top:10px;font-size:12px">继续提问</button>';
             var ag0 = g('queryAgain');
             if (ag0) ag0.addEventListener('click', function () { inp.value = ''; inp.focus(); });
             log.innerHTML = log.innerHTML + '<div style="font-size:12px;color:var(--muted);margin:4px 0">Q: ' + esc(q) + ' → 知识库暂无</div>';
             return;
           }
           // 动态可视化 (M-F8): 响应携带 viz 数据时, 实时渲染能级/跃迁图 (非静态预设)
           var vizPayload = (d && d.viz) || (d && d.data && d.data.viz) || null;
           var vizHtml = '';
           if (vizPayload && vizPayload.hit && vizPayload.data) {
             vizHtml = '<div class="mf6-viz" style="margin-top:10px;border:1px solid var(--rule);border-radius:10px;background:var(--surface);overflow:hidden">' +
               '<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:linear-gradient(135deg,rgba(99,102,241,.1),rgba(139,92,246,.1));border-bottom:1px solid var(--rule)">' +
               '<span style="font-size:12px;font-weight:600">📊 ' + esc(vizPayload.note || '动态可视化') + '</span>' +
               '<span style="flex:1"></span>' +
               '<a href="javascript:void(0)" class="mf6-viz-open" style="font-size:10.5px;color:var(--accent-ink,#6366f1)">在实验台打开 →</a></div>' +
               '<div class="mf6-viz-body" style="padding:10px;overflow:auto"></div></div>';
           }
           res.innerHTML = '<div style="background:var(--surface2);padding:12px;border-radius:8px;margin-bottom:8px">' +
            '<div style="font-size:12px;color:var(--muted);margin-bottom:6px">答案（置信度 ' + esc(String(conf)) + ' · 审核 ' + esc(String(verdict)) + (reqConf ? ' · 需确认' : '') + ' · 意图 ' + esc(qtype || '通用') + '）</div>' +
            '<div style="font-size:14px">' + renderSlideDeck(answer) + '</div></div>' +
            srcHtml +
            vizHtml +
            '' +
            (pipeHtml ? '<div>' + pipeHtml + '</div>' : '') +
            collabHtml +
            (function () {
              // 启发式导学: 答后自然追问 (可点选继续), 资源排版优化成紧凑卡片
              var fq = heuristicFollowUps(q, answer);
              if (!fq.length) return '';
              var chips = fq.map(function (f) {
                return '<button class="q-follow-chip" data-q="' + escAttr(f) + '">' + esc(f) + '</button>';
              }).join('');
              return '<div class="q-follow" style="margin-top:10px;padding:10px 12px;border:1px dashed var(--rule);border-radius:8px;background:var(--surface)">' +
                '<div style="font-size:11px;color:var(--muted);margin-bottom:6px">🔍 想继续深挖？点一点（启发式导学）</div>' +
                '<div style="display:flex;flex-wrap:wrap;gap:8px">' + chips + '</div></div>';
            })() +
            '<button class="btn ghost" id="queryAgain" style="margin-top:10px;font-size:12px">继续提问</button>';
          var ag = g('queryAgain');
          if (ag) ag.addEventListener('click', function () { inp.value = ''; inp.focus(); });
          // 动态渲染可视化 (M-F8): 数据 → 实时出图
          var vizBody = res && res.querySelector && res.querySelector('.mf6-viz-body');
          if (vizBody && window.MF8Viz) window.MF8Viz.renderFromData(vizBody, vizPayload.data, vizPayload.viz_type);
          var vizOpen = res && res.querySelector && res.querySelector('.mf6-viz-open');
          if (vizOpen) vizOpen.addEventListener('click', function () {
            if (window.MF8Viz) window.MF8Viz.inject(vizPayload.data, vizPayload.viz_type);
            if (window.location.hash) window.location.hash = '';
            var nav = document.querySelector('[data-view="atomic-viz"], .nav-item[data-view="atomic-viz"]');
            if (nav) { nav.click(); } else if (window.__goto) { window.__goto('atomic-viz'); }
          });
          res.querySelectorAll('.q-follow-chip').forEach(function (cp) {
            cp.addEventListener('click', function () {
              inp.value = cp.getAttribute('data-q');
              run();
            });
          });
          // 知识讲解幻灯片: 翻页
          res.querySelectorAll('.mf6-slides').forEach(function (deck) {
            var slides = deck.querySelectorAll('.mf6-slide');
            var ind = deck.querySelector('.mf6-slide-ind');
            var cur = 0;
            function show(n) {
              cur = (n + slides.length) % slides.length;
              slides.forEach(function (sl, i) { sl.style.display = i === cur ? '' : 'none'; });
              if (ind) ind.textContent = (cur + 1) + ' / ' + slides.length;
            }
            var pv = deck.querySelector('.mf6-slide-prev');
            var nx = deck.querySelector('.mf6-slide-next');
            if (pv) pv.addEventListener('click', function () { show(cur - 1); });
            if (nx) nx.addEventListener('click', function () { show(cur + 1); });
          });
          log.innerHTML = log.innerHTML +
            '<div style="font-size:12px;color:var(--muted);margin:4px 0">Q: ' + esc(q) + ' → ' + esc(String(verdict)) + '</div>';
        }).catch(function (e) {
          res.innerHTML = '<div class="error-banner">答疑失败: ' + esc(e.message) + '</div>';
          clearInterval(stepTimer);
        });
      });
    };
    if (btn) btn.addEventListener('click', run);
    if (inp) inp.addEventListener('keydown', function (ev) { if (ev.key === 'Enter') run(); });
  }

  // 2f. 今日推荐接入 L4 唯一策略决策点 (POST /l4/decision/next-action)

  // 教学优先: 为推荐知识点加载「知识讲解 + 关系图谱」(聚焦机理/通用知识, 而非只做题)
  function loadKpKnowledge(kpId, kpName) {
    var q = kpName || kpId;
    if (!q) return Promise.resolve('');
    return Promise.all([
      apiReq('POST', '/l3/retrieve/keyword', { query: q, top_k: 3 }).catch(function () { return null; }),
      apiReq('GET', '/l3/triples').catch(function () { return null; }),
    ]).then(function (r) {
      var hits = (r[0] && r[0].results) || [];
      var trips = Array.isArray(r[1]) ? r[1] : ((r[1] && (r[1].items || r[1].triples)) || []);
      var snippet = hits.length ? String(hits[0].content || hits[0].text || hits[0].summary || '') : '';
      var rel = trips.filter(function (t) {
        var s = String(t.subject || t.head || '');
        var o = String(t.object || t.tail || '');
        return s.indexOf(q) >= 0 || o.indexOf(q) >= 0 || s.indexOf(kpId) >= 0 || o.indexOf(kpId) >= 0;
      }).slice(0, 5);
      if (!snippet && !rel.length) return '';
      var relHtml = rel.length ? rel.map(function (t) {
        return '<span class="rel-chip" style="display:inline-flex;align-items:center;gap:4px;background:var(--surface2);border:1px solid var(--rule);border-radius:999px;padding:2px 9px;font-size:11px;margin:2px">' +
          '<b>' + esc(t.subject || t.head) + '</b><span style="color:var(--muted)">—' + esc(t.predicate || t.relation) + '→</span><b>' + esc(t.object || t.tail) + '</b></span>';
      }).join('') : '';
      return '<div style="margin-bottom:12px;border:1px solid var(--violet-border);border-radius:10px;padding:10px 12px;background:linear-gradient(135deg,var(--violet-soft),var(--surface))">' +
        '<div style="font-size:12px;font-weight:600;color:var(--violet);margin-bottom:6px">📖 先学知识：' + esc(q) + '</div>' +
        (snippet ? '<div style="font-size:12.5px;line-height:1.7;color:var(--ink)">' + esc(String(snippet).slice(0, 240)) + '</div>' : '<div style="font-size:12px;color:var(--muted)">知识库暂未收录该知识点的讲解片段。</div>') +
        (relHtml ? '<div style="margin-top:8px"><div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:4px">🕸️ 关系图谱（举一反三）</div><div style="display:flex;flex-wrap:wrap">' + relHtml + '</div></div>' : '') +
        '</div>';
    });
  }

  function renderRecommendations() {
    var ct = g('content');
    if (!ct) return;
    // 学情采集: 浏览画像
    apiReq('POST', '/l2/event/collect', { learner_id: learnerId(), event_type: 'view', detail: 'overview-recs' }).catch(function () {});
    // 用户理解: 加载渐进式揭示洞察, 用于推荐注入
    apiReq('POST', '/api/user-understanding/profile', { learner_id: learnerId() })
      .then(function (ins) { window._uuInsights = ins || {}; })
      .catch(function () { window._uuInsights = {}; });
    // 策略生成: L4 唯一策略决策点 (mode=guide → action_type/confidence/recommended_path 统一语义)
    // L5/前端负责组装画像数据传给 L4 (L4 为纯决策)
    apiReq('GET', '/l2/profile/' + learnerId()).then(function (prof) {
      return apiReq('POST', '/l4/decision/next-action', {
        learner_id: learnerId(), mode: 'guide', learner_profile: prof || {}
      }).then(function (d) {
        return { d: d, names: (prof && prof.kp_names) || {} };
      });
    }).then(function (res) {
      var d = res.d;
      var names = res.names;
      var decision = (d && d.data) || d || {};
      var path = decision.recommended_path || [];
      var ins = window._uuInsights || {};
      var interestTopics = (ins.interests || []).map(function (x) { return x.topic; });
      // 兴趣命中排序: 命中兴趣的 KP 提前
      if (interestTopics.length) {
        path = path.slice().sort(function (a, b) {
          var ka = interestTopics.some(function (t) { return (names[a.kp_id] || a.kp_id).indexOf(t) >= 0 || (a.kp_id || '').indexOf(t) >= 0; }) ? 1 : 0;
          var kb = interestTopics.some(function (t) { return (names[b.kp_id] || b.kp_id).indexOf(t) >= 0 || (b.kp_id || '').indexOf(t) >= 0; }) ? 1 : 0;
          return kb - ka;
        });
      }
      if (!path.length) {
        ct.innerHTML = '<div class="card"><h3>今日推荐（策略生成）</h3><p style="color:var(--muted)">暂无策略（未完成初测请先练习）</p></div>';
        return;
      }
      var items = path.map(function (st) {
        var hitInterest = interestTopics.some(function (t) { return (names[st.kp_id] || st.kp_id).indexOf(t) >= 0 || (st.kp_id || '').indexOf(t) >= 0; });
        var tag = hitInterest ? '<span class="badge ok" style="margin-left:6px">兴趣命中</span>' : '';
        var encourage = (ins.frustration_level || 0) >= 0.5 ? '<span style="font-size:11px;color:var(--muted);margin-left:6px">💪 慢慢来，你可以的</span>' : '';
        return '<div class="kp-cell" style="border-left:3px solid ' + (st.action === '考核' ? 'var(--warn)' : 'var(--success)') + '">' +
          '<span class="kp-id">' + esc(st.kp_id) + '</span>' +
          '<span class="kp-name" style="font-size:11px;line-height:1.35">' + esc(names[st.kp_id] || st.kp_id) + '</span>' +
          '<span style="font-family:var(--mono);font-size:11px;color:var(--muted)">' + esc(st.action) + ' → 目标 ' + Math.round((st.target || 0) * 100) + '%</span>' +
          '<span style="font-family:var(--mono);font-size:11px;color:var(--muted)">难度 ' + esc(st.effort) + '</span>' + tag + encourage + '</div>';
      }).join('');
      ct.innerHTML = '<div class="card"><h3>今日推荐（L4 策略决策）</h3>' +
        '<p style="color:var(--muted);font-size:13px;margin-bottom:10px">行动类型 ' + esc(decision.action_type || '-') +
        ' · 置信度 ' + esc(String(decision.confidence != null ? decision.confidence : '-')) +
        ' · ' + esc(decision.summary || '') + '</p>' +
        '<div id="recKnowledge" style="margin-bottom:4px"></div>' +
        '<div class="kp-grid">' + items + '</div>' +
        '<button class="btn ghost" id="recStart" style="margin-top:12px">去练习</button></div>';
      var rs = g('recStart');
      if (rs) rs.addEventListener('click', function () {
        if (window.sv) { window.sv('practice'); } else { renderPractice(); }
      });
      // 教学优先: 为置顶推荐 KP 加载「知识讲解 + 关系图谱」
      if (path.length) {
        var topKp = path[0];
        loadKpKnowledge(topKp.kp_id, names[topKp.kp_id] || topKp.kp_id).then(function (html) {
          var box = g('recKnowledge');
          if (box && html) box.innerHTML = html;
        });
      }
    }).catch(function (e) {
      // 未登录/认证失败 → 引导登录; 其余 → 服务异常提示
      var em = (e && (e.message || '')) || '';
      var needLogin = e && (e.status === 401 || e.status === 403 || em.indexOf('Authentication') >= 0 || em.indexOf('AUTHENTICATION') >= 0 || em.indexOf('登录') >= 0 || em === 'HTTP 401');
      if (needLogin) {
        ct.innerHTML = '<div class="card" style="text-align:center;padding:40px"><h3>今日推荐</h3>' +
          '<p style="color:var(--muted);margin:14px 0">个性化推荐需要登录后根据你的学情画像生成。</p>' +
          '<button class="btn primary" onclick="olv()">登录</button></div>';
      } else {
        ct.innerHTML = '<div class="card"><h3>今日推荐</h3><p style="color:var(--muted)">策略生成服务暂不可用</p></div>';
      }
    });
  }

  // 3. 溯源链查看入口 (kb 增强, 渲染后注入; 仅保留"查看完整溯源链"按钮, 去除哈希验证卡片)
  function injectProvenanceBadges() {
    var ct = g('content');
    if (!ct) return;
    if (ct.getAttribute('data-mf6') === 'kb-prov') return;
    var html = ct.innerHTML;
    if (html.indexOf('知识库管理') === -1 || html.indexOf('实体列表') === -1) return;
    ct.setAttribute('data-mf6', 'kb-prov');
    // 注入"查看完整溯源链"入口按钮 (跳转溯源链视图, 完整哈希/红绿灯状态在视图内展示)
    var btnHtml = '<div style="margin-top:10px;text-align:right"><button class="btn ghost sm" id="gotoProvView" style="font-size:12px">🔗 查看完整溯源链 &rarr;</button></div>';
    var anchors = ct.querySelectorAll('.grid.cols-2');
    if (anchors.length) {
      var wrap = document.createElement('div');
      wrap.innerHTML = btnHtml;
      while (wrap.firstChild) {
        anchors[0].parentNode.insertBefore(wrap.firstChild, anchors[0]);
      }
    } else {
      ct.insertAdjacentHTML('beforeend', btnHtml);
    }
    // 为"查看完整溯源链"按钮绑定导航
    var gb = g('gotoProvView');
    if (gb) gb.addEventListener('click', function () {
      var sv = window.sv;
      if (sv) { sv('kb-provenance'); return; }
      var el = d.querySelector('[data-view="kb-provenance"]');
      if (el) el.click();
    });
  }

  /* ---------- 真实学习数据链路: 练习 / 薄弱点 / 今日推荐 ---------- */

  // 练习视图状态
  var PracticeState = { questions: [], idx: 0, answered: false, last: null, done: false };

  function renderPractice() {
    var ct = g('content');
    if (!ct) return;
    var lid = learnerId();
    // 助手指定题量时生效 (mf7-assistant 写入 mf7_practice_count), 默认 12 (四维覆盖)
    var askCount = parseInt(localStorage.getItem('mf7_practice_count') || '12', 10);
    if (!(askCount >= 8 && askCount <= 20)) askCount = 12;
    localStorage.removeItem('mf7_practice_count');
    var targetKps = [];
    try { targetKps = JSON.parse(sessionStorage.getItem('dy3_practice_target_kps') || '[]'); sessionStorage.removeItem('dy3_practice_target_kps'); } catch (ignoreTargetKps) {}
    var targetQuery = targetKps.length ? '&kp_ids=' + encodeURIComponent(targetKps.join(',')) : '';
    // 只使用现有人工题库；指定 Concept 没有题时诚实返回空集。
    apiReq('GET', '/l2/practice/questions?learner_id=' + encodeURIComponent(lid) + '&count=' + askCount + targetQuery).then(function (data) {
      PracticeState.questions = (data && data.questions) || [];
      PracticeState.idx = 0;
      PracticeState.answered = false;
      PracticeState.done = false;
      PracticeState.last = null;
      if (!PracticeState.questions.length) {
        ct.innerHTML = '<div class="card"><h3>学习练习</h3><p style="color:var(--muted)">暂无题目, 请联系管理员摄入题库</p></div>';
        return;
      }
      ct.innerHTML = '<div class="card"><h3>学习练习</h3>' +
        '<div class="callout" style="margin-bottom:12px">题目来自现有人工题库；提交后才会形成 AnswerRecord 并更新学习模型。</div></div>';
      drawPracticeQuestion();
    }).catch(function (e) {
      ct.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  function drawPracticeQuestion() {
    var ct = g('content');
    var q = PracticeState.questions[PracticeState.idx];
    if (!q) { PracticeState.done = true; drawPracticeSummary(); return; }
    var type = q.type || 'choice';
    var typeLabel = { choice: '单选', judge: '判断', blank: '填空', multi: '多选', graph: '图谱' }[type] || '单选';
    var bodyHtml = '';
    if (type === 'blank') {
      bodyHtml = '<input id="practiceBlankInput" type="text" placeholder="在此输入答案（如 4f⁹ / 480 nm / RG0）" style="width:100%;padding:10px 14px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink);font-size:14px">' +
        '<button class="btn primary" id="practiceBlankSubmit" style="margin-top:10px">提交答案</button>';
    } else if (type === 'multi') {
      bodyHtml = (q.options || []).map(function (o, i) {
        return '<label class="practice-opt" style="display:block;margin:6px 0;padding:10px 14px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink);cursor:pointer">' +
          '<input type="checkbox" data-mi="' + i + '" style="margin-right:8px">' +
          String.fromCharCode(65 + i) + '. ' + esc(o) + '</label>';
      }).join('') +
        '<button class="btn primary" id="practiceMultiSubmit" style="margin-top:12px">提交（可多选）</button>';
    } else if (type === 'graph') {
      // 图谱关系题: 中心概念节点 + 关系边到各结论 (复用图形渲染能力, 融会贯通)
      var center = (q.graph && q.graph.center) || q.kg_name || q.kp_id || '';
      bodyHtml = '<div style="text-align:center;margin:2px 0 12px;padding:12px;background:var(--surface2);border:1px solid var(--rule);border-radius:10px">' +
        '<div style="display:inline-block;padding:6px 18px;border-radius:999px;background:linear-gradient(135deg,#8b5cf6,#6366f1);color:#fff;font-size:13px;font-weight:600">🧠 ' + esc(center) + '</div>' +
        '<div style="font-size:11px;color:var(--muted);margin-top:6px">↑ 中心概念 — 请选出与它正确相连的「关系边」</div></div>' +
        (q.options || []).map(function (o, i) {
          return '<button class="btn practice-opt" data-opt="' + i + '" style="display:flex;width:100%;text-align:left;margin:6px 0;padding:10px 14px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink);align-items:center;gap:10px">' +
            '<span style="flex:none;font-weight:700;color:var(--accent)">' + String.fromCharCode(65 + i) + '</span>' +
            '<span style="flex:1">' + esc(o) + '</span></button>';
        }).join('');
    } else {
      bodyHtml = (q.options || []).map(function (o, i) {
        return '<button class="btn practice-opt" data-opt="' + i + '" style="display:block;width:100%;text-align:left;margin:6px 0;padding:10px 14px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink)">' +
          String.fromCharCode(65 + i) + '. ' + esc(o) + '</button>';
      }).join('');
    }
    ct.innerHTML = '<div class="card"><h3>学习练习</h3>' +
      '<div style="display:flex;justify-content:space-between;margin-bottom:10px">' +
      '<span class="badge info">第 ' + (PracticeState.idx + 1) + ' / ' + PracticeState.questions.length + ' 题</span>' +
      '<span><span class="badge warn">' + esc(q.kp_id || '') + '</span> <span class="badge">' + typeLabel + '</span></span></div>' +
      '<p style="font-size:15px;line-height:1.7;margin-bottom:14px">' + esc(q.question) + '</p>' +
      '<div>' + bodyHtml + '</div>' +
      '<div id="practiceFeedback" style="margin-top:12px"></div>' +
      '<button class="btn ghost" id="practiceSkip" style="margin-top:12px">跳过此题</button></div>';
    if (type === 'blank') {
      var bs = g('practiceBlankSubmit');
      if (bs) bs.addEventListener('click', function () {
        submitPractice(q, -1, (g('practiceBlankInput') || {}).value || '');
      });
      var bi = g('practiceBlankInput');
      if (bi) bi.addEventListener('keydown', function (ev) { if (ev.key === 'Enter') submitPractice(q, -1, bi.value); });
    } else if (type === 'multi') {
      var ms = g('practiceMultiSubmit');
      if (ms) ms.addEventListener('click', function () {
        var sel = ct.querySelectorAll('#content input[data-mi]:checked');
        var idxs = [];
        sel.forEach(function (c) { idxs.push(parseInt(c.dataset.mi, 10)); });
        if (!idxs.length) { var f = g('practiceFeedback'); if (f) f.innerHTML = '<div style="color:var(--muted);font-size:13px">请至少选择一项</div>'; return; }
        submitPractice(q, idxs.join(','));
      });
    } else {
      ct.querySelectorAll('.practice-opt').forEach(function (b) {
        b.addEventListener('click', function () { submitPractice(q, parseInt(b.dataset.opt, 10)); });
      });
    }
    var sk = g('practiceSkip');
    if (sk) sk.addEventListener('click', function () { PracticeState.idx++; drawPracticeQuestion(); });
    console.log('[Dy3] practice: render question', q.qid, q.kp_id, type);
  }

  /* 只陈述本次真实作答事实。单次答题不会证明掌握，也不会伪造四 Agent 决策。 */
  function buildIterationFeedback(ok, r, q) {
    if (!ok) {
      return '<div class="dp-iteration dp-iteration-down">' +
        '<div style="font-size:12px;font-weight:600;color:#d97706;margin-bottom:4px">本次作答：错误</div>' +
        '<div style="font-size:12px;color:var(--muted);line-height:1.7">系统已保存该题的真实 AnswerRecord，并更新对应模型状态。一次错误不等于已确认薄弱点；后续 Diagnosis 会结合更多作答、Concept 前置关系和历史证据再决定教学深度。</div>' +
        '<div style="margin-top:6px;font-size:11px;color:var(--muted)">事实来源：本次练习提交 · 知识点 ' + esc(q.kp_id || '') + '</div>' +
        '</div>';
    }
    return '<div class="dp-iteration dp-iteration-up">' +
      '<div style="font-size:12px;font-weight:600;color:#16a34a;margin-bottom:4px">本次作答：正确</div>' +
      '<div style="font-size:12px;color:var(--muted);line-height:1.7">系统已保存一条正向学习证据。单次正确不会被标记为“已掌握”，也不会自动伪造进阶决策；未来 Diagnosis 将结合连续作答和模型置信度判断是否提升难度。</div>' +
      '<div style="margin-top:6px;font-size:11px;color:var(--muted)">事实来源：本次练习提交 · 知识点 ' + esc(q.kp_id || '') + '</div>' +
      '</div>';
  }

  function submitPractice(q, selected, textAnswer) {
    if (PracticeState.answered) return;
    PracticeState.answered = true;
    var body = { learner_id: learnerId(), qid: q.qid, selected: selected };
    try {
      body.attempt_purpose = sessionStorage.getItem('dy3_practice_attempt_purpose') || 'DIAGNOSTIC';
    } catch (ignoreAttemptPurpose) { body.attempt_purpose = 'DIAGNOSTIC'; }
    try {
      var practiceContext = JSON.parse(sessionStorage.getItem('dy3_practice_context') || '{}');
      if (practiceContext.task_id && practiceContext.resource_id) {
        body.task_id = String(practiceContext.task_id);
        body.resource_id = String(practiceContext.resource_id);
      }
    } catch (ignorePracticeContext) {}
    if (textAnswer != null) body.text_answer = textAnswer;
    apiReq('POST', '/l2/practice/answer', body).then(function (r) {
      PracticeState.last = r;
      var fb = g('practiceFeedback');
      if (!fb) return;
      var ok = !!r.correct;
      var correctIdx = r.correct_index;
      var answerTxt = '';
      if (Array.isArray(correctIdx)) {
        answerTxt = correctIdx.map(function (i) {
          return (q.options && q.options[i] != null) ? String.fromCharCode(65 + i) + '. ' + q.options[i] : String.fromCharCode(65 + i);
        }).join('；');
      } else if (correctIdx != null && correctIdx >= 0 && q.options && q.options[correctIdx] != null) {
        answerTxt = String.fromCharCode(65 + correctIdx) + '. ' + q.options[correctIdx];
      }
      var answerLine = answerTxt ? ('<div style="margin-top:6px;font-size:13px">正确答案: ' + esc(answerTxt) + '</div>') : '';
      // 动态迭代机制 (竞赛要求): 答错 → 降维解释, 答对 → 进阶挑战
      var iterHtml = buildIterationFeedback(ok, r, q);
      fb.innerHTML =
        '<div class="callout ' + (ok ? 'ok' : 'warn') + '" style="margin-top:10px">' +
        '<strong>' + (ok ? '✓ 回答正确' : '✗ 回答错误') + '</strong>' +
        answerLine +
        (r.explanation ? '<div style="margin-top:6px;font-size:13px;color:var(--muted)">' + esc(r.explanation) + '</div>' : '') +
        '<div style="margin-top:8px"><span class="badge ' + (ok ? 'ok' : 'err') + '">服务器已保存 · ' + esc(q.kp_id || '') + ' · ' + esc(r.attempt_purpose || body.attempt_purpose) + '</span></div>' +
        '</div>' +
        iterHtml +
        '<div id="practiceAuthoritativeRefresh" class="t1-resource-feedback" aria-live="polite">正在后台刷新权威学习视图…</div>' +
        '<button class="btn primary" id="practiceNext" style="margin-top:12px">' + (PracticeState.idx + 1 >= PracticeState.questions.length ? '完成练习' : '下一题') + '</button>';
      var refreshState = g('practiceAuthoritativeRefresh');
      var refreshSettled = false;
      var refreshTimer = setTimeout(function () {
        if (!refreshSettled && refreshState) refreshState.textContent = '学习结果已保存；权威视图刷新较慢，可继续操作，稍后回到工作台查看。';
      }, 4000);
      apiReq('GET', '/api/learning-workspace/' + encodeURIComponent(learnerId())).then(function (workspace) {
        refreshSettled = true;
        clearTimeout(refreshTimer);
        var authoritative = (workspace && workspace.data) || workspace || {};
        if (refreshState) refreshState.textContent = '权威学习视图已刷新 · ' + String((authoritative.learner_summary || {}).evidence_status || 'UNKNOWN');
      }).catch(function () {
        refreshSettled = true;
        clearTimeout(refreshTimer);
        if (refreshState) refreshState.textContent = '学习结果已保存；权威视图暂未刷新，不使用前端估算替代。';
      });
      var nx = g('practiceNext');
      if (nx) nx.addEventListener('click', function () {
        PracticeState.idx++;
        PracticeState.answered = false;
        drawPracticeQuestion();
      });
      console.log('[Dy3] practice: answer', q.qid, ok ? 'correct' : 'wrong', 'saved');
    }).catch(function (e) {
      var fb = g('practiceFeedback');
      if (fb) fb.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
      PracticeState.answered = false;
    });
  }

  function drawPracticeSummary() {
    var ct = g('content');
    var lid = learnerId();
    apiReq('GET', '/api/learning-workspace/' + encodeURIComponent(lid)).then(function (payload) {
      var ws = (payload && payload.data) || payload || {};
      var summary = ws.learner_summary || {};
      var challenge = ws.current_challenge_decision || {};
      ct.innerHTML = '<div class="card"><h3>练习完成</h3>' +
        '<div class="grid cols-3" style="margin-bottom:14px">' +
        '<div class="stat-card"><div class="lbl">真实作答</div><div class="num">' + esc(String(summary.observed_record_count || 0)) + '</div></div>' +
        '<div class="stat-card"><div class="lbl">证据状态</div><div class="num" style="font-size:15px">' + esc(summary.evidence_status || 'UNKNOWN') + '</div></div>' +
        '<div class="stat-card"><div class="lbl">下一难度判断</div><div class="num" style="font-size:15px">' + esc(workspaceActionLabel(challenge.decision)) + '</div></div>' +
        '</div><p style="color:var(--muted);font-size:13px;margin-bottom:12px">作答已写入现有 AnswerRecord/BKT/Profile 链；此处只显示服务端权威投影，不在前端计算掌握度。</p>' +
        '<div style="display:flex;gap:10px;flex-wrap:wrap">' +
        '<button class="btn primary" id="pracAgain">再练一组</button>' +
        '<button class="btn ghost" id="pracGoHeat">查看热力图</button></div></div>';
      var ag = g('pracAgain');
      if (ag) ag.addEventListener('click', renderPractice);
      var gh = g('pracGoHeat');
      if (gh) gh.addEventListener('click', function () {
        if (window.sv) { window.sv('overview'); } else { renderLearnerOverview(); }
      });
    });
  }

  // 薄弱点分析视图 (learn-weak)
  // 个性化学习资源 (3 种形态: 定制化讲解 / 实操指南 / 分阶测试题)
  function renderPersonalizedResources(target) {
    var box = target || g('content');
    if (!box) return;
    box.innerHTML = '<div style="padding:20px;text-align:center;color:var(--muted);font-size:12px">正在生成个性化资源…</div>';
    apiReq('GET', '/api/personalized/resources?learner_id=' + encodeURIComponent(learnerId())).then(function (r) {
      var d = (r && r.data) || r || {};
      var ctx = d.learner_context || {};
      var cust = d.customized_resource || {};
      var guide = d.practical_guide || {};
      var staged = d.staged_questions || {};
      var html = '<div class="card" style="margin-bottom:12px;padding:14px 16px">' +
        '<h3 style="margin:0 0 8px">🎯 个性化学习资源</h3>' +
        '<p style="color:var(--muted);font-size:12px;margin:0">整合「先验知识画像 + 领域知识库」动态生成 · 学历背景 <b>' + esc(ctx.role_label || '-') + '</b> · 深度 <b>' + esc(ctx.depth || '-') + '</b> · 能力值 θ <b>' + esc(String(ctx.theta != null ? ctx.theta : '-')) + '</b></p></div>';
      // 1. 定制化讲解
      html += '<div class="card" style="margin-bottom:12px"><h4 style="margin:0 0 8px">📖 ' + esc(cust.title || '定制讲解') + '</h4>' +
        (cust.intro ? '<p style="font-size:12px;color:var(--muted);margin:0 0 10px">' + esc(cust.intro) + '</p>' : '') +
        (cust.sections || []).map(function (s) {
          return '<div style="margin-bottom:10px;padding:10px 12px;border:1px solid var(--rule);border-radius:8px;background:var(--surface)">' +
            '<div style="font-weight:600;font-size:13px">' + esc(s.kp_name || s.kp_id) + ' <span class="badge info">掌握度 ' + Math.round((s.mastery || 0) * 100) + '%</span></div>' +
            (s.key_points || []).map(function (p) { return '<div style="font-size:12.5px;color:var(--muted);line-height:1.7;margin-top:4px">• ' + esc(p) + '</div>'; }).join('') +
            '</div>';
        }).join('') + '</div>';
      // 2. 实操指南
      html += '<div class="card" style="margin-bottom:12px"><h4 style="margin:0 0 4px">🔬 ' + esc(guide.title || '实操指南') + '</h4>' +
        (guide.hint ? '<p style="font-size:12px;color:var(--muted);margin:0 0 10px">' + esc(guide.hint) + '</p>' : '') +
        (guide.steps || []).map(function (st) {
          return '<details style="margin-bottom:6px"><summary style="cursor:pointer;font-size:13px;font-weight:600">' + st.step + '. ' + esc(st.name) + '</summary>' +
            '<div style="padding:6px 0 4px 12px;font-size:12.5px;color:var(--muted);line-height:1.7">' + esc(st.operation) + '</div>' +
            (st.safety ? '<div style="padding:0 0 4px 12px;font-size:12px;color:var(--warn)">⚠ ' + esc(st.safety) + '</div>' : '') +
            (st.question ? '<div style="padding:0 0 4px 12px;font-size:12px;color:var(--violet)">💡 ' + esc(st.question) + '</div>' : '') +
            '</details>';
        }).join('') + '</div>';
      // 3. 分阶测试题
      html += '<div class="card"><h4 style="margin:0 0 8px">📝 ' + esc(staged.title || '分阶测试题') + '</h4>' +
        (staged.stages || []).map(function (sg) {
          return '<div style="margin-bottom:10px"><div style="font-weight:600;font-size:13px;margin-bottom:4px">' + esc(sg.stage) + '（' + sg.questions.length + ' 题）</div>' +
            (sg.questions || []).map(function (q) {
              var tLabel = { choice: '单选', judge: '判断', blank: '填空', multi: '多选', graph: '图谱' }[q.type] || '单选';
              return '<div style="padding:8px 10px;border-left:3px solid #8b5cf6;background:var(--surface);border-radius:6px;margin-bottom:6px">' +
                '<div style="font-size:12.5px">' + esc(q.question || '') + ' <span class="badge" style="font-size:10.5px">' + tLabel + '</span></div>' +
                '</div>';
            }).join('') + '</div>';
        }).join('') + '</div>';
      box.innerHTML = html;
    }).catch(function (e) {
      box.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  function renderWeakPoints() {
    var ct = g('content');
    if (!ct) return;
    apiReq('GET', '/l2/profile/' + learnerId()).then(function (p) {
      return apiReq('POST', '/l4/decision/next-action', { learner_id: learnerId(), mode: 'guide', learner_profile: p || {} })
        .catch(function () { return null; })
        .then(function (d) { return { p: p, dec: (d && d.data) || d || null }; });
    }).then(function (r) {
      var p = r.p;
      var dec = r.dec;
      var km = (p && p.kp_mastery) || {};
      var names = (p && p.kp_names) || {};
      var weak = (p && p.weak_kps) || Object.keys(km).filter(function (k) { return km[k] < 0.6; });
      // 顶部: L4 策略推荐摘要 (整合「今日推荐」)
      var recHtml = '';
      if (dec && dec.recommended_path && dec.recommended_path.length) {
        var steps = dec.recommended_path.slice(0, 3).map(function (st) {
          return '<span class="badge ' + (st.action === '考核' ? 'warn' : 'info') + '" style="margin:2px">' + esc(names[st.kp_id] || st.kp_id) + ' · ' + esc(st.action) + '</span>';
        }).join('');
        recHtml = '<div style="margin-bottom:12px;padding:10px 12px;border:1px solid var(--violet-border);border-radius:10px;background:linear-gradient(135deg,var(--violet-soft),var(--surface))">' +
          '<div style="font-size:12px;font-weight:600;color:var(--violet);margin-bottom:6px">🎯 今日推荐策略（L4 决策）</div>' +
          '<div style="font-size:12px;color:var(--muted);margin-bottom:6px">' + esc(dec.summary || '') + ' · 置信度 ' + esc(String(dec.confidence != null ? dec.confidence : '-')) + '</div>' +
          steps + '</div>';
      }
      var cards = weak.length ? weak.slice(0, 20).map(function (w) {
        var v = km[w] || 0;
        var wname = names[w] || w;
        return '<div class="card" style="margin-bottom:10px;padding:12px 14px">' +
          '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">' +
          '<div style="min-width:0;flex:1"><div style="font-weight:600;font-size:13px">' + esc(wname) + ' <code style="font-size:10px;color:var(--muted)">' + esc(w) + '</code></div>' +
          '<div style="margin-top:6px;display:flex;align-items:center;gap:8px"><div style="height:8px;border-radius:4px;background:var(--surface2);flex:1;max-width:180px"><div style="height:100%;width:' + Math.round(v * 100) + '%;border-radius:4px;background:var(--danger)"></div></div>' +
          '<span class="badge ' + (v < 0.4 ? 'err' : 'warn') + '">' + (v < 0.4 ? '未掌握' : '发展中') + '</span></div></div>' +
          '<div style="flex:none;display:flex;gap:6px"><button class="btn ghost sm wk-explain" data-kp="' + esc(w) + '" data-name="' + esc(wname) + '" style="font-size:12px">📖 讲解</button>' +
          '<button class="btn ghost sm wk-practice" data-kp="' + esc(w) + '" style="font-size:12px">✏️ 去练习</button></div></div>' +
          '<div class="wk-explain-box" data-kp="' + esc(w) + '" style="margin-top:8px"></div></div>';
      }).join('') : '<div style="text-align:center;color:var(--muted);padding:20px">暂无薄弱点, 继续保持!</div>';
      var activityHtml = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px">' +
        '<div class="card" style="padding:14px;cursor:pointer" id="lcPractice"><div style="font-size:20px">✏️</div><div style="font-weight:600;margin-top:4px">学习练习</div><div style="font-size:11px;color:var(--muted);margin-top:2px">12 题 · 四维能力覆盖</div></div>' +
        '<div class="card" style="padding:14px;cursor:pointer" id="lcQuery"><div style="font-size:20px">💬</div><div style="font-weight:600;margin-top:4px">知识问答</div><div style="font-size:11px;color:var(--muted);margin-top:2px">多智能体协同答疑</div></div>' +
        '</div>';
      ct.innerHTML = '<div class="card"><h3>学习中心</h3>' +
        '<p style="color:var(--muted);font-size:13px;margin-bottom:12px">个性化推荐 + 学习练习 + 知识问答，一站式学习。</p>' +
        recHtml + cards + activityHtml + '</div>' +
        '<div id="lcPersonalized" style="margin-top:14px"></div>';
      // 个性化学习资源 (3 种形态)
      renderPersonalizedResources(g('lcPersonalized'));
      // 讲解按钮: 加载知识讲解 + 关系图谱
      ct.querySelectorAll('.wk-explain').forEach(function (b) {
        b.addEventListener('click', function () {
          var kp = b.getAttribute('data-kp');
          var nm = b.getAttribute('data-name');
          var box = ct.querySelector('.wk-explain-box[data-kp="' + kp + '"]');
          if (!box) return;
          if (box.innerHTML) { box.innerHTML = ''; return; }
          box.innerHTML = '<div style="font-size:12px;color:var(--muted)">加载中…</div>';
          loadKpKnowledge(kp, nm).then(function (html) {
            box.innerHTML = html || '<div style="font-size:12px;color:var(--muted)">该知识点暂无知识讲解（可先在知识库摄入相关内容）。</div>';
          });
        });
      });
      // 去练习按钮: 直接进入练习
      ct.querySelectorAll('.wk-practice').forEach(function (b) {
        b.addEventListener('click', function () {
          if (window.sv) { window.sv('practice'); } else { renderPractice(); }
        });
      });
      // 学习活动入口: 练习 / 问答
      var lcP = g('lcPractice');
      if (lcP) lcP.addEventListener('click', function () { if (window.sv) { window.sv('practice'); } else { renderPractice(); } });
      var lcQ = g('lcQuery');
      if (lcQ) lcQ.addEventListener('click', function () { if (window.sv) { window.sv('query'); } else { renderQuery(); } });
    }).catch(function (e) {
      ct.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  // R-04B learning workspace: state → next action → recommended task.
  function renderR04LearningWorkspace() {
    var ct = g('content');
    if (!ct) return;
    if (window.DY3ProductCanvas && typeof window.DY3ProductCanvas.renderGrowth === 'function') {
      window.DY3ProductCanvas.renderGrowth(ct);
      return;
    }
    var taskData = r08StoredTask();
    ct.innerHTML = r08Journey('learn-weak', taskData) + '<div class="r04-processing"><span class="spinner"></span><div><strong>读取学习状态</strong></div></div>';
    r08BindJourney(ct);
    apiReq('GET', '/l2/profile/' + learnerId()).then(function (profile) {
      return Promise.all([
        apiReq('GET', '/api/match-report/' + encodeURIComponent(learnerId())).catch(function () { return null; }),
        apiReq('POST', '/l4/decision/next-action', { learner_id: learnerId(), mode: 'guide', learner_profile: profile || {} }).catch(function () { return null; })
      ]).then(function (values) {
        var match = values[0] || {};
        return {
          profile: profile || {},
          report: match.report || null,
          decision: (values[1] && values[1].data) || values[1] || {}
        };
      });
    }).then(function (result) {
      var profile = result.profile;
      var report = result.report;
      var decision = result.decision;
      var mastery = profile.kp_mastery || {};
      var names = profile.kp_names || {};
      var findings = report && Array.isArray(report.findings) ? report.findings : [];
      var pathData = (report && report.learning_path) || {};
      var pathNodes = Array.isArray(pathData.nodes) ? pathData.nodes : [];
      var timeline = report && Array.isArray(report.growth_timeline) ? report.growth_timeline : [];
      var nextAction = (report && report.next_action) || {};
      var nextText = nextAction.reason || nextAction.target || decision.summary || '当前没有公开的下一步建议。';
      var reasonText = r04Text(nextAction.reason || decision.reason || decision.rationale || decision.explanation || '');
      var conceptNames = {};
      pathNodes.forEach(function (node) { conceptNames[node.concept_id] = node.name || node.concept_id; });
      var lastTimeline = timeline.length ? timeline[timeline.length - 1] : null;
      var pastText = lastTimeline ? ('最近真实事件：' + (lastTimeline.outcome || lastTimeline.event_type || 'RECORDED')) : '尚无真实作答或学习事件';
      var findingLabels = { VERIFIED_WEAKNESS: '已验证薄弱', PREREQUISITE_GAP: '先修缺口', MISCONCEPTION: '误概念', UNKNOWN: '未知' };
      var weakHtml = findings.slice(0, 4).map(function (finding) {
        var reference = finding.reference || 'unknown';
        return '<li><strong>' + esc(conceptNames[reference] || reference) + '</strong><span>' + esc(findingLabels[finding.type] || finding.type) + ' · ' + esc(finding.source_class || 'UNKNOWN') + '</span></li>';
      }).join('');
      var taskNodes = pathNodes.filter(function (node) { return node.role === 'PREREQUISITE' || node.role === 'TARGET'; }).sort(function (left, right) {
        if (left.concept_id === nextAction.target) return -1;
        if (right.concept_id === nextAction.target) return 1;
        if (left.role === right.role) return 0;
        return left.role === 'PREREQUISITE' ? -1 : 1;
      });
      var taskHtml = taskNodes.slice(0, 4).map(function (node, index) {
        var title = node.name || node.concept_id || ('推荐任务 ' + (index + 1));
        return '<button class="r04b-task-option" data-question="' + escAttr(title) + '"><span>0' + (index + 1) + '</span><strong>' + esc(title) + '</strong></button>';
      }).join('');
      var detailRows = Object.keys(mastery).slice(0, 20).map(function (key) {
        return '<tr><td>' + esc(names[key] || key) + '</td><td>' + esc(key) + '</td><td>' + Math.round(Number(mastery[key] || 0) * 100) + '%</td></tr>';
      }).join('');
      var masteryCanvas = Object.keys(mastery).slice(0, 10).map(function (key) {
        var value = Math.max(0, Math.min(100, Math.round(Number(mastery[key] || 0) * 100)));
        return '<div class="canvas-mastery-row"><div><strong>' + esc(names[key] || key) + '</strong><span>' + value + '%</span></div><div class="canvas-meter"><i style="width:' + value + '%"></i></div></div>';
      }).join('');
      var reportStatus = report ? report.status : 'UNAVAILABLE';
      var sufficiency = (report && report.evidence_sufficiency) || {};
      ct.innerHTML = '<article class="r04b-learning canvas-page canvas-growth">' + r08Journey('learn-weak', taskData) + '<header class="r08-primary-header"><span class="r04-section-kicker">GROWTH PATH</span><h1>成长路径</h1></header>' +
        r08RecentTaskPanel(taskData, r08StoredQuestion()) +
        '<section class="r08-growth-timeline" aria-label="过去现在未来"><div><span>过去</span><strong>最近学习记录</strong><p>' + esc(pastText) + '</p></div>' +
        '<div><span>现在</span><strong>' + esc(reportStatus) + '</strong><p>' + esc(findings.length ? findings.length + ' 项分类判断' : '当前没有足够事实形成判断') + '</p></div>' +
        '<div><span>未来</span><strong>下一步行动</strong><p>' + esc(nextText) + '</p></div></section>' +
        '<div class="r08b3-growth-main"><section class="r04b-learning-state"><div><strong>当前状态</strong><span>' + esc(reportStatus) + '</span><small>' + esc(sufficiency.source_class || 'UNKNOWN') + '</small></div>' +
        '<div><strong>下一步建议</strong><span>' + esc(nextText) + '</span>' + (reasonText ? '<small>推荐依据：' + esc(reasonText) + '</small>' : '') + '</div></section>' +
        '<section><h3>当前学习判断</h3>' + (weakHtml ? '<ul class="r04b-weak-list">' + weakHtml + '</ul><p class="r04-muted">明确区分 UNKNOWN、已验证薄弱、先修缺口与误概念。</p>' : '<p class="r04-muted">当前没有足够事实形成学习判断。</p>') + '</section></div>' +
        '<section class="canvas-mastery-panel"><div class="r08-section-title"><div><span>模型状态</span><h3>知识掌握度</h3></div><small>来源：Learner Profile / BKT 投影</small></div>' + (masteryCanvas || '<div class="r04-empty">当前没有可展示的掌握度记录。</div>') + '</section>' +
        '<section><h3>推荐任务</h3>' + (taskHtml ? '<div class="r04b-task-options">' + taskHtml + '</div>' : '<button class="btn primary" id="learningOpenQuery">发起一个材料学习问题</button>') + '</section>' +
        '<section><div class="r08-section-title"><div><span>个性化学习支持</span><h3>知识资源、科研任务与分阶练习</h3></div><small>复用现有公开能力</small></div><div class="r08-support-grid">' +
        '<article><span>01</span><h4>知识学习资源</h4><p>根据当前画像读取定制讲解、实操指南与分阶题。</p><button class="btn ghost" id="openGrowthResources">查看个性化资源</button></article>' +
        '<article><span>02</span><h4>科研学习任务</h4><p>使用上方真实 recommended path 进入下一项材料学习任务。</p><button class="btn ghost" data-view-target="query">发起核心任务</button></article>' +
        '<article><span>03</span><h4>分阶练习与分析</h4><p>通过现有练习闭环更新画像，并查看公开匹配报告。</p><div class="r08-inline-actions"><button class="btn primary" data-view-target="practice">进入练习</button><button class="btn ghost" data-view-target="match-report">分析报告</button></div></article></div></section>' +
        '<details class="r04b-advanced" id="growthResourcesDetails"><summary>查看个性化学习资源</summary><div id="growthResources"><p class="r04-muted">展开后读取定制讲解、实操指南与分阶测试题。</p></div></details>' +
        '<details class="r04b-advanced"><summary>查看 BKT / 掌握度高级信息（MODEL_INFERRED）</summary>' +
        (detailRows ? '<div class="table-wrap"><table><thead><tr><th>知识点</th><th>ID</th><th>模型记录</th></tr></thead><tbody>' + detailRows + '</tbody></table></div>' : '<p class="r04-muted">暂无可用模型记录。</p>') + '</details></article>';
      ct.querySelectorAll('.r04b-task-option').forEach(function (button) {
        button.addEventListener('click', function () {
          sessionStorage.setItem('dy3_pending_query', button.getAttribute('data-question') || '');
          if (window.sv) window.sv('query');
        });
      });
      var open = g('learningOpenQuery');
      if (open) open.addEventListener('click', function () { if (window.sv) window.sv('query'); });
      ct.querySelectorAll('[data-view-target]').forEach(function (button) {
        button.addEventListener('click', function () { if (window.sv) window.sv(button.getAttribute('data-view-target')); });
      });
      r08BindJourney(ct);
      var resources = g('growthResourcesDetails');
      if (resources) resources.addEventListener('toggle', function () {
        if (!resources.open || resources.getAttribute('data-loaded') === 'true') return;
        resources.setAttribute('data-loaded', 'true');
        var resourceHost = g('growthResources');
        var taskResources = taskData && Array.isArray(taskData.learning_resources) ? taskData.learning_resources : [];
        if (resourceHost && taskResources.length) {
          resourceHost.innerHTML = t1ResourceCards(taskData);
          t1BindResourceActions(resourceHost, taskData, r08StoredQuestion(), null);
        } else if (resourceHost) {
          resourceHost.innerHTML = '<div class="r04-empty">最近任务没有通过发布门的个性化资源。请先完成一次核心任务；系统不会用通用模板冒充任务资源。</div>';
        }
      });
      var openResources = g('openGrowthResources');
      if (openResources && resources) openResources.addEventListener('click', function () {
        resources.open = true;
        resources.dispatchEvent(new Event('toggle'));
        resources.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }).catch(function (error) {
      ct.innerHTML = '<article class="r04b-learning">' + r08Journey('learn-weak', taskData) + '<header><div class="r04-section-kicker">成长决策</div><h2>学习状态暂不可用</h2><p class="r04-muted">' + esc(error.message || '请登录后重试。') + '</p></header><button class="btn primary" id="learningFallbackQuery">仍可发起核心任务</button></article>';
      r08BindJourney(ct);
      var fallback = g('learningFallbackQuery');
      if (fallback) fallback.addEventListener('click', function () { if (window.sv) window.sv('query'); });
    });
  }

  /* ---------- Agent 列表视图 (动态加载 4 个 Agent + 执行) ---------- */
  function renderAgentList() {
    var ct = g('content');
    if (!ct) return;
    apiReq('GET', '/l5/agents').then(function (data) {
      var agents = (data && data.agents) || [];
      if (!agents.length) {
        ct.innerHTML = '<div class="card"><h3>智能体</h3><p style="color:var(--muted)">Agent 运行时未初始化</p></div>';
        return;
      }
      var cards = agents.map(function (a) {
        var inst = a.instance || {};
        var healthy = inst.healthy === true;
        var chans = (a.broadcast_channels || []).map(function (b) {
          return '<span class="badge info" style="margin:2px">' + esc(b.channel) + ' <b>' + esc(b.mode) + '</b></span>';
        }).join('');
        return '<div class="card" data-agent="' + esc(a.id) + '">' +
          '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">' +
          '<h4 style="margin:0">' + esc(a.name) + '</h4>' +
          '<span class="badge ' + (healthy ? 'ok' : 'warn') + '">' + (healthy ? '运行中' : '未激活') + '</span></div>' +
          '<p style="color:var(--muted);font-size:13px;margin:8px 0">' + esc(a.role || '') + '</p>' +
          '<div style="font-size:12px;margin:6px 0"><span class="lbl" style="color:var(--muted)">ID: </span><code>' + esc(a.id) + '</code>' +
          ' <span class="lbl" style="color:var(--muted);margin-left:10px">工具: </span><code>' + (a.tools || []).length + '</code></div>' +
          (chans ? '<div style="margin:6px 0">' + chans + '</div>' : '') +
          '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">' +
          '<button class="btn primary" data-run="' + esc(a.id) + '">执行测试</button>' +
          (a.id.indexOf('generation') >= 0 ? '<button class="btn ghost" data-mode="practice" style="font-size:12px">练习出题</button><button class="btn ghost" data-mode="assess" style="font-size:12px">考核</button>' : '') +
          '</div>' +
          '<pre data-out="' + esc(a.id) + '" style="display:none;margin-top:8px;max-height:180px;overflow:auto;font-size:12px;background:var(--surface2);padding:8px;border-radius:6px"></pre>' +
          '</div>';
      }).join('');
      ct.innerHTML = '<div class="card"><h3>智能体列表</h3>' +
        '<p style="color:var(--muted);font-size:13px;margin-bottom:10px">四个核心 Agent 运行时状态与广播频道，点击"执行测试"动态调用 Agent 执行。</p></div>' +
        '<div class="grid cols-2">' + cards + '</div>';
      ct.querySelectorAll('[data-run]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var aid = btn.dataset.run;
          var pre = ct.querySelector('[data-out="' + aid + '"]');
          if (pre) { pre.style.display = 'block'; pre.textContent = '执行中…'; }
          var payload = aid.indexOf('diagnosis') >= 0 ? { learner_id: learnerId() }
            : aid.indexOf('generation') >= 0 ? { query: 'Dy3+ 的量子效率受哪些因素影响？' }
            : aid.indexOf('review') >= 0 ? { content: 'Dy3+ 离子的发射波长为 575nm' }
            : { learner_id: learnerId(), query: 'Dy3+ 的量子效率受哪些因素影响？' };
          apiReq('POST', '/l5/agents/' + aid + '/run', payload).then(function (r) {
            if (pre) {
              pre.textContent = JSON.stringify(r, null, 2);
              console.log('[Dy3] agent run', aid, 'status', r.status);
            }
          }).catch(function (e) {
            if (pre) pre.textContent = '执行失败: ' + e.message;
          });
        });
      });
      // 知识生成 Agent: 练习出题 / 考核模式 (按画像针对性出题)
      ct.querySelectorAll('[data-mode]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var mode = btn.dataset.mode;
          var aid = 'agent.knowledge.generation';
          var pre = ct.querySelector('[data-out="' + aid + '"]');
          if (pre) { pre.style.display = 'block'; pre.textContent = '执行中…'; }
          var payload = mode === 'assess'
            ? { learner_id: learnerId(), mode: 'assess', count: 3 }
            : { learner_id: learnerId(), mode: 'practice', count: 5 };
          apiReq('POST', '/l5/agents/' + aid + '/run', payload).then(function (r) {
            if (pre) {
              pre.textContent = JSON.stringify(r, null, 2);
              console.log('[Dy3] agent', mode, 'status', r.status, 'questions', r.questions ? r.questions.length : 0);
            }
          }).catch(function (e) {
            if (pre) pre.textContent = '执行失败: ' + e.message;
          });
        });
      });
    }).catch(function (e) {
      ct.innerHTML = '<div class="error-banner">' + esc(e.message) + '</div>';
    });
  }

  /* ---------- Agent 协作链路视图 (多向广播 PUB/SUB 传播图) ---------- */
  var AGENT_LINKS = [
    { from: 'agent.learning.diagnosis', to: 'agent.knowledge.generation', channel: 'learning.knowledge.gap', mode: '广播 薄弱块', back: false },
    { from: 'agent.learning.diagnosis', to: 'agent.guidance.decision', channel: 'learning.diagnosis.report', mode: '广播 诊断', back: false },
    { from: 'agent.knowledge.generation', to: 'agent.quality.review', channel: 'knowledge.generation.output', mode: '广播 内容', back: false },
    { from: 'agent.quality.review', to: 'agent.guidance.decision', channel: 'knowledge.review.result', mode: '广播 审核', back: false },
    { from: 'agent.guidance.decision', to: 'agent.learning.diagnosis', channel: 'guidance.decision.command', mode: '决策→画像', back: true },
    { from: 'agent.knowledge.generation', to: 'agent.learning.diagnosis', channel: 'learning.interaction.event', mode: '考核→画像', back: true },
  ];
  var AGENT_DESC = {
    'agent.learning.diagnosis': '学情诊断',
    'agent.knowledge.generation': '知识生成',
    'agent.quality.review': '审核校验',
    'agent.guidance.decision': '导学决策',
  };

  function renderAgentChain() {
    var ct = g('content');
    if (!ct) return;
    if (window.DY3ProductCanvas && typeof window.DY3ProductCanvas.renderCollaboration === 'function') {
      window.DY3ProductCanvas.renderCollaboration(ct);
      return;
    }
    var data = r08StoredTask();
    var question = r08StoredQuestion();
    if (!r08TaskTruth.loaded && !r08TaskTruth.loading) {
      ct.innerHTML = '<div class="r04-processing"><span class="spinner"></span><div><strong>读取服务端任务事实</strong></div></div>';
      r08LoadTaskTruth(function () { renderAgentChain(); });
      return;
    }
    var lines = data && Array.isArray(data.collab_lines) ? data.collab_lines : [];
    var flow = data && Array.isArray(data.flow_events) ? data.flow_events : [];
    var agentTrace = data && Array.isArray(data.agent_trace) ? data.agent_trace : [];
    var labels = lines.map(function (line) { return line.label || ''; });
    var actors = {};
    agentTrace.forEach(function (event) { if (event.agent_id) actors[event.agent_id] = true; });
    flow.forEach(function (event) { if (event.agent) actors[event.agent] = true; });
    var summary = data ? '<div class="r04b-collab-facts"><div><strong>' + Object.keys(actors).length + '</strong><span>实际参与 Agent</span></div>' +
      '<div><strong>' + labels.filter(function (label) { return label === 'SUBTASK_READY'; }).length + '</strong><span>真实 Subtask</span></div>' +
      '<div><strong>' + labels.filter(function (label) { return label === 'EVIDENCE_RETRIEVED'; }).length + '</strong><span>Evidence 更新</span></div>' +
      '<div><strong>' + labels.filter(function (label) { return label === 'CHALLENGE_RAISED'; }).length + '</strong><span>Reviewer Challenge</span></div></div>' : '';
    var contributionGroups = {};
    agentTrace.forEach(function (event) {
      if (!event || !event.agent_id) return;
      if (!contributionGroups[event.agent_id]) contributionGroups[event.agent_id] = [];
      var detail = String(event.detail || '').trim();
      if (detail && contributionGroups[event.agent_id].indexOf(detail) < 0) contributionGroups[event.agent_id].push(detail);
    });
    var rolePurpose = {
      'agent.learning.diagnosis': '解释学习状态与教学深度',
      'agent.knowledge.generation': '形成材料知识解释并使用证据',
      'agent.quality.review': '审核科学事实、边界与风险',
      'agent.guidance.decision': '综合结果并给出学习决策'
    };
    var contributionOrder = ['agent.learning.diagnosis', 'agent.knowledge.generation', 'agent.quality.review', 'agent.guidance.decision'];
    var roleMap = '<section class="r08b3-agent-role-map" aria-label="四个 Agent 的任务分工"><div class="r08-section-title"><div><span>任务分工</span><h3>四个 Agent</h3></div></div><div>' + contributionOrder.map(function (agentId, index) {
      var contributed = Boolean(contributionGroups[agentId] && contributionGroups[agentId].length);
      return '<article class="' + (contributed ? 'has-contribution' : '') + '"><span>0' + (index + 1) + '</span><h4>' + esc(r04AgentName(agentId)) + '</h4><p>' + esc(rolePurpose[agentId]) + '</p><small>' + (contributed ? '已产生公开贡献' : '本次无公开贡献') + '</small></article>';
    }).join('') + '</div></section>';
    var contributionCards = contributionOrder.filter(function (agentId) { return contributionGroups[agentId] && contributionGroups[agentId].length; }).map(function (agentId) {
      return '<article><span>' + esc(rolePurpose[agentId]) + '</span><h4>' + esc(r04AgentName(agentId)) + '</h4><p>' + esc(contributionGroups[agentId].join('；')) + '</p></article>';
    }).join('');
    var contributions = contributionCards ? '<section><div class="r08-section-title"><div><span>专业贡献</span><h3>任务贡献</h3></div></div><div class="r04b-contributions">' + contributionCards + '</div></section>' : '';
    var trace = data && (flow.length || lines.length || agentTrace.length) ? r04RenderTrace(data) : '';
    ct.innerHTML = '<article class="r04-collaboration-page canvas-page canvas-collaboration">' + r08Journey('agents-chain', data) + '<header><span class="r04-section-kicker">COLLABORATION ANALYSIS</span><h2>协同分析</h2></header>' +
      roleMap +
      (data ? '<section class="r04b-current-task"><span>本次任务</span><h3>' + esc(question || data.task_id || '最近一次任务') + '</h3><p>' + esc(data.task_state || '') + '</p></section>' + summary + contributions + r04RenderChallenge(data) +
        (trace ? '<section class="r04-collaboration-record"><h3>真实协同时间线</h3>' + trace + '</section>' : '') :
        '<section class="r04-collaboration-record"><div class="r04-empty">尚无当前任务的公开协同事实。请先完成一次核心任务。</div></section>') +
      '<button class="btn primary" id="goCoreTask" type="button">发起新的核心任务</button></article>';
    r08BindJourney(ct);
    var go = g('goCoreTask');
    if (go) go.addEventListener('click', function () { if (window.sv) window.sv('query'); });
  }

  function renderR04KnowledgeWorkspace() {
    var ct = g('content');
    if (!ct) return;
    if (window.DY3ProductCanvas && typeof window.DY3ProductCanvas.renderKnowledge === 'function') {
      window.DY3ProductCanvas.renderKnowledge(ct);
      return;
    }
    var data = r08StoredTask();
    var question = r08StoredQuestion();
    if (!r08TaskTruth.loaded && !r08TaskTruth.loading) {
      ct.innerHTML = '<div class="r04-processing"><span class="spinner"></span><div><strong>读取服务端任务事实</strong></div></div>';
      r08LoadTaskTruth(function () { renderR04KnowledgeWorkspace(); });
      return;
    }
    var evidence = data && Array.isArray(data.evidence) ? data.evidence : [];
    var sources = data && Array.isArray(data.sources) ? data.sources : [];
    var review = data && data.review && typeof data.review === 'object' ? data.review : {};
    var verdict = review.verdict || review.status || '未提供';
    var knowledgeNames = [];
    sources.forEach(function (source) {
      (source && Array.isArray(source.kp_names) ? source.kp_names : []).forEach(function (name) {
        if (name && knowledgeNames.indexOf(String(name)) < 0) knowledgeNames.push(String(name));
      });
    });
    var evidenceHtml = evidence.length ? '<ol class="r04-evidence-list">' + evidence.slice(0, 8).map(function (item, index) {
      var source = item && typeof item === 'object' ? (item.source || item.title || item.chunk_id || '') : '';
      return '<li><div class="r04-evidence-index">' + (index + 1) + '</div><div><p>' + esc(r04Text(item) || '证据内容未提供') + '</p>' +
        (source ? '<span>来源：' + esc(source) + '</span>' : '') + '</div></li>';
    }).join('') + '</ol>' : '<div class="r04-empty">当前还没有任务相关证据。先提出一个材料问题，证据会随任务一起进入这里。</div>';
    var knowledgeProjection = knowledgeNames.length ? '<section><h3>当前任务公开知识点关联</h3><div class="r08-knowledge-tags">' + knowledgeNames.slice(0, 12).map(function (name) { return '<span>' + esc(name) + '</span>'; }).join('') + '</div><p class="r04-muted">这些名称来自 CURRENT sources 中的 KP 投影，不等同于 R06 Canonical Concept 或 Concept Relation。</p></section>' : '';
    var evidenceChain = '<section class="r08b3-evidence-chain" aria-label="可信回答链"><div><span>01 · 问题</span><strong>' + esc(question || '等待核心任务') + '</strong><small>当前学习任务</small></div><div><span>02 · 证据</span><strong>' + (evidence.length ? evidence.length + ' 条公开证据' : '尚无公开证据') + '</strong><small>' + (sources.length ? sources.length + ' 项公开来源' : '来源将在任务返回后展示') + '</small></div><div><span>03 · 审核</span><strong>' + esc(String(verdict)) + '</strong><small>Reviewer 的 CURRENT 结论</small></div></section>';
    ct.innerHTML = '<article class="r04b-knowledge canvas-page canvas-knowledge">' + r08Journey('kb', data) + '<header><span class="r04-section-kicker">KNOWLEDGE &amp; EVIDENCE</span><h2>知识证据</h2></header>' +
      evidenceChain +
      (data ? '<section class="r08-trust-strip"><div><span>回答</span><strong>' + (data.answer ? '已生成' : '未生成') + '</strong></div><div><span>公开证据</span><strong>' + evidence.length + ' 条</strong></div><div><span>公开来源</span><strong>' + sources.length + ' 项</strong></div><div><span>Reviewer</span><strong>' + esc(String(verdict)) + '</strong></div></section>' : '') +
      '<div class="canvas-knowledge-grid"><div class="canvas-knowledge-main"><section class="r04b-knowledge-task"><span>当前任务</span><h3>' + esc(question || '尚未发起任务') + '</h3>' + evidenceHtml + (evidence.length ? '<p class="r04-muted">证据片段本身不等于支持关系；只按真实 Claim–Evidence 映射显示支持等级。</p>' : '') + '</section>' +
      (data ? '<section class="r04-evidence-card"><div class="r04-section-kicker">Canonical Concept Relation</div><h3>当前问题的概念关系图</h3>' + t234ConceptGraph(data) + '</section>' : '') + '</div><aside class="canvas-evidence-rail">' +
      (data ? t5678ScientificGrounding(data) : '') + knowledgeProjection + '</aside></div>' +
      '<section class="r04b-knowledge-search"><h3>围绕问题检索知识</h3><div><input id="knowledgeQuestion" type="text" placeholder="输入材料、跃迁或性能问题"><button class="btn primary" id="knowledgeSearch">检索</button></div><div id="knowledgeSearchResult"></div></section>' +
      '<details class="r04b-advanced"><summary>其他知识工具</summary><p class="r04-muted">历史 KnowledgeEntity / Triple 实体图与上方任务 Concept Relation 分开展示。</p><div class="r04b-advanced-actions"><button class="btn ghost" data-view-target="kb-graph">历史实体图</button><button class="btn ghost" data-view-target="kb-provenance">溯源链</button><button class="btn ghost" data-view-target="atomic-viz">科学可视化</button></div></details>' +
      '</article>';
    r08BindJourney(ct);
    var search = g('knowledgeSearch');
    var input = g('knowledgeQuestion');
    if (search) search.addEventListener('click', function () {
      var query = String(input && input.value || '').trim();
      var result = g('knowledgeSearchResult');
      if (!query || !result) return;
      result.innerHTML = '<p class="r04-muted">正在检索公开知识…</p>';
      apiReq('POST', '/l3/retrieve/keyword', { query: query, top_k: 5 }).then(function (payload) {
        var items = (payload && (payload.items || payload.results || payload.chunks)) || (Array.isArray(payload) ? payload : []);
        result.innerHTML = items.length ? '<ol class="r04-evidence-list">' + items.map(function (item, index) {
          return '<li><div class="r04-evidence-index">' + (index + 1) + '</div><div><p>' + esc(r04Text(item) || '检索结果') + '</p><span>' + esc(item.source || item.title || item.chunk_id || '') + '</span></div></li>';
        }).join('') + '</ol>' : '<div class="r04-gap">当前知识库没有返回相关证据。</div>';
      }).catch(function (error) { result.innerHTML = '<div class="r04-gap">检索失败：' + esc(error.message || '') + '</div>'; });
    });
    ct.querySelectorAll('[data-view-target]').forEach(function (button) {
      button.addEventListener('click', function () { if (window.sv) window.sv(button.getAttribute('data-view-target')); });
    });
  }

  /* ---------- 视图路由钩子 (MutationObserver 拦截 sv 渲染) ---------- */
  var VIEW_OVERRIDES = {
    'query-history': renderSessions,
    'settings': renderSettings,
    'compare-learners': renderCompare,
    'learn-path': renderTimeTravel,
    'time-travel': renderTimeTravel,  // 管理者侧栏注入的时间旅行入口 → 同一轨迹视图
    'learn-mastery': renderBktHeatmap,
    'learn': renderLearnRing,
    'practice': renderPractice,
    'learn-weak': renderR04LearningWorkspace,
    'overview-recs': renderRecommendations,
    'agents-list': renderAgentList,
    'agents-chain': renderAgentChain,
    'users': renderUsers,
    'users-roles': renderRoles,
    'users-import': renderImport,
    'query': renderQuery,
    'kb': renderR04KnowledgeWorkspace,
    'kb-provenance': renderProvenanceView,
    'overview': function () {
      // 学习者 → 画像主页 (游戏面板 + 数据状态卡); 管理者 → 管理概览
      if (currentRole() === 'admin') { renderAdminOverview(); } else { renderLearnerOverview(); }
    },
  };
  // 需要覆盖的视图列表
  var OVERRIDE_VIEWS = Object.keys(VIEW_OVERRIDES);

  function currentView() {
    // 以侧栏 active 按钮的 data-view 为准 (最可靠); 兜底 window.S.v (底部按钮无 active 类)
    var active = d.querySelector('.sidebar-child.active');
    if (active && active.dataset && active.dataset.view) return active.dataset.view;
    if (window.S && window.S.v) return window.S.v;
    return null;
  }

  function detectAndOverride() {
    var ct = g('content');
    if (!ct) return;
    var view = currentView();
    if (!view) return;
    if (OVERRIDE_VIEWS.indexOf(view) !== -1) {
      // 已有 mf6 渲染结果 (data-mf6 === view) 且内容不是 sv() 的 loading 占位 → 不重复渲染.
      // 唯一重新渲染触发: sv() 设置的 data-mf6-pending 占位 (视图切换).
      // 渲染函数内部的 .loading/.spinner (如溯源链加载态) 不得触发重渲染, 否则死循环.
      if (ct.getAttribute('data-mf6') === view) {
        var pending = ct.querySelector('[data-mf6-pending]');
        if (!pending) return;
        ct.removeAttribute('data-mf6');
      }
      ct.setAttribute('data-mf6', view);
      var fn = VIEW_OVERRIDES[view];
      if (fn) fn();
      return;
    }
    // kb 视图: 渲染完成后注入溯源链徽章
    if (view === 'kb' || view === 'kb-search' || view === 'kb-graph') {
      injectProvenanceBadges();
    }
  }

  function hookViewRendering() {
    var ct = g('content');
    if (!ct) return;
    // 防抖: 渲染函数内部多次 DOM 更新 (loading→数据) 只触发一次检测, 避免高频回调
    var pendingT = null;
    var obs = new MutationObserver(function () {
      if (pendingT) return;  // 已有待执行检测
      pendingT = setTimeout(function () {
        pendingT = null;
        detectAndOverride();
      }, 60);
    });
    obs.observe(ct, { childList: true, subtree: true });
    // 初始检测: 覆盖登录/刷新后首个视图 (如 overview)
    detectAndOverride();
  }

  /* ---------- 管理者导航注入 (对比/时间旅行) ---------- */
  function currentRole() {
    var r = localStorage.getItem('dr') || 'student';
    if (r === 'teacher' || r === 'admin') return 'admin';
    return 'student';
  }

  // 注入导航按钮到侧边栏 (事件驱动: app.js rs() 每次重建后派发 sidebar-rebuilt,
  // 本函数监听该事件注入, 替代旧的 setInterval 轮询, 消除重复/闪烁)
  var _mf6NavBound = false;
  function _injectNavOnce() {
    var sb = g('sidebarBody');
    if (!sb) return;
    if (currentRole() === 'admin') {
      _injectManagerButtons(sb);
      _injectProvenanceButton(sb);
    }
    // Product navigation remains exactly three learner spaces.
    // Practice, provenance, atomic visualization and other historical views
    // stay routable but are not injected into the default navigation.
  }
  function _injectManagerButtons(sb) {
    if (sb.querySelector('[data-view="compare-learners"]')) return;
    var btn = d.createElement('button');
    btn.className = 'sidebar-child';
    btn.dataset.view = 'compare-learners';
    btn.innerHTML = '<span class="child-icon">\u2696\uFE0F</span>' + t('compare');
    btn.addEventListener('click', function () {
      var ev = new CustomEvent('mf6-nav', { detail: 'compare-learners' });
      d.dispatchEvent(ev);
      renderCompare();
    });
    var tt = d.createElement('button');
    tt.className = 'sidebar-child';
    tt.dataset.view = 'time-travel';
    tt.innerHTML = '<span class="child-icon">\u23F3</span>' + t('time_travel');
    tt.addEventListener('click', function () {
      var ev = new CustomEvent('mf6-nav', { detail: 'time-travel' });
      d.dispatchEvent(ev);
      renderTimeTravel();
    });
    var wrap = d.createElement('div');
    wrap.style.cssText = 'padding:6px 0;border-bottom:1px solid var(--rule)';
    wrap.appendChild(btn);
    wrap.appendChild(tt);
    sb.insertBefore(wrap, sb.firstChild);
    if (!_mf6NavBound) {
      _mf6NavBound = true;
      d.addEventListener('mf6-nav', function (e) {
        var bc = g('breadcrumb');
        if (bc) bc.textContent = 'Dy3+ Polaris \u203A ' + (e.detail === 'compare-learners' ? '\u5b66\u4e60\u5bf9\u6bd4' : '\u65f6\u95f4\u65c5\u884c');
      });
    }
  }
  function _injectLearnerButtons(sb) {
    if (sb.querySelector('[data-view="practice"]')) return;
    // \u7EC3\u4E60\u5165\u53E3\u5E94\u5F52\u5C5E\u300C\u5B66\u60C5\u300D\u5206\u7EC4 (data-section="learn"), \u800C\u975E\u585E\u8FDB\u300C\u603B\u89C8\u300D\u5206\u7EC4,
    // \u907F\u514D\u5206\u7EC4\u8BED\u4E49\u9519\u4F4D + \u7A81\u5140\u7684\u5206\u5272\u7EBF\u3002
    var learnSection = sb.querySelector('.sidebar-section-header[data-section="learn"]');
    var children = learnSection ? learnSection.parentNode.querySelector('.sidebar-section-children') : null;
    var btn = d.createElement('button');
    btn.className = 'sidebar-child';
    btn.dataset.view = 'practice';
    btn.innerHTML = '<span class="child-icon">\u270D\uFE0F</span>' + t('practice');
    btn.addEventListener('click', function () {
      var bc = g('breadcrumb');
      if (bc) bc.textContent = 'Dy3+ Polaris \u203A ' + t('practice');
      renderPractice();
    });
    if (children) {
      // \u63D2\u5165\u5230\u300C\u5B66\u60C5\u300D\u5206\u7EC4\u5B50\u9879\u9876\u90E8 (\u65E0\u9700\u989D\u5916\u5206\u5272\u7EBF, \u4E0E\u5176\u5B83\u5B50\u9879\u5BF9\u9F50)
      children.insertBefore(btn, children.firstChild);
    } else {
      sb.appendChild(btn);
    }
  }
  function _injectProvenanceButton(sb) {
    if (sb.querySelector('[data-view="kb-provenance"]')) return;
    var kbSection = sb.querySelector('.sidebar-section [data-section="kb"]');
    if (!kbSection) return;
    var children = kbSection.querySelector('.sidebar-section-children');
    if (!children) return;
    var btn = d.createElement('button');
    btn.className = 'sidebar-child';
    btn.dataset.view = 'kb-provenance';
    btn.innerHTML = '<span class="child-icon">🔗</span>溯源链';
    btn.addEventListener('click', function () {
      var bc = g('breadcrumb');
      if (bc) bc.textContent = 'Dy3+ Polaris › 溯源链';
      var sv = window.sv;
      if (sv) sv('kb-provenance');
    });
    children.appendChild(btn);
  }

  /* ---------- 设置面板兜底 (事件委托, 不依赖 app.js 按钮绑定) ---------- */
  function openSettingsFallback() {
    var m = g('passwordModal');
    if (!m) return;
    var h = m.querySelector('h3');
    if (h) h.textContent = 'Settings';
    var f = m.querySelector('form');
    if (f) f.style.display = 'none';
    var e = m.querySelector('.modal-error');
    if (e) e.hidden = 1;
    var s = d.querySelector('.settings-panel');
    if (!s) {
      s = d.createElement('div');
      s.className = 'settings-panel';
      s.style.cssText = 'padding:8px 0';
      var md = m.querySelector('.modal');
      if (md) md.insertBefore(s, f);
    }
    var rl = currentRole() === 'admin' ? '管理者' : (localStorage.getItem('dr') === 'teacher' ? '教师' : '学习者');
    var isDark = (d.documentElement.getAttribute('data-theme') || 'light') === 'dark';
    var tm = isDark ? 'Dark' : 'Light';
    var lg = localStorage.getItem('dt') ? 'Logout' : 'Login';
    var u = localStorage.getItem('dl') || '-';
    s.innerHTML =
      '<div class="grid cols-2" style="margin-bottom:12px">' +
      statCard('角色', rl) + statCard('主题', tm) +
      statCard('状态', localStorage.getItem('dt') ? 'Online' : 'Guest') +
      statCard('用户', u) +
      '</div>' +
      '<div style="display:flex;gap:10px"><button class="btn primary" id="stTh">Toggle Theme</button>' +
      '<button class="btn primary" id="stLo">' + lg + '</button></div>';
    m.hidden = 0;
    setTimeout(function () {
      var tb = g('stTh');
      if (tb) tb.addEventListener('click', function () {
        var root = d.documentElement;
        var cur = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', cur);
        localStorage.setItem('dh', cur);
        openSettingsFallback();
      });
      var lb = g('stLo');
      if (lb) lb.addEventListener('click', function () {
        if (localStorage.getItem('dt')) {
          localStorage.removeItem('dt');
          localStorage.removeItem('dr');
          location.reload();
        } else {
          var lv = g('loginView');
          if (lv) lv.hidden = 0;
        }
      });
    }, 50);
  }

  function setupSettingsDelegate() {
    var sb = d.querySelector('.sidebar');
    if (!sb || sb.dataset.mf6Settings) return;
    sb.dataset.mf6Settings = '1';
    sb.addEventListener('click', function (ev) {
      var t = ev.target && ev.target.closest ? ev.target.closest('[data-view="settings"]') : null;
      if (!t) return;
      ev.preventDefault();
      ev.stopPropagation();
      console.log('[Dy3] settings opened via delegate');
      openSettingsFallback();
    });
  }

  /* ---------- 移动端统一布局 (参考 Trae: 侧栏抽屉化, 不分端) ---------- */
  function setupResponsiveNav() {
    var sb = g('sidebar');
    if (!sb) return;

    // 汉堡按钮 (幂等, 挂到 body 悬浮, topbar 为 display:none)
    var btn = g('menuBtn');
    if (!btn) {
      btn = d.createElement('button');
      btn.className = 'topbar-btn';
      btn.id = 'menuBtn';
      btn.title = '菜单';
      btn.setAttribute('aria-label', '打开菜单');
      btn.innerHTML = '☰';
      d.body.appendChild(btn);
    }
    // 遮罩 (幂等)
    var mask = g('sidebarMask');
    if (!mask) {
      mask = d.createElement('div');
      mask.id = 'sidebarMask';
      d.body.appendChild(mask);
    }

    function openSb() { sb.classList.add('open'); mask.classList.add('show'); btn.setAttribute('aria-label', '关闭菜单'); }
    function closeSb() { sb.classList.remove('open'); mask.classList.remove('show'); btn.setAttribute('aria-label', '打开菜单'); }
    btn.onclick = function (e) {
      e.stopPropagation();
      if (sb.classList.contains('open')) closeSb(); else openSb();
    };
    mask.onclick = closeSb;
    // 侧栏内点击导航后自动收起 (窄屏)
    sb.addEventListener('click', function (ev) {
      if (ev.target.closest && ev.target.closest('[data-view]')) closeSb();
    });
    // 窄屏初始收起; 窗口跨断点时保持状态一致
    function sync() {
      if (window.innerWidth <= 768) closeSb();
    }
    window.addEventListener('resize', function () { sync(); });
    sync();
  }

  /* ---------- WS 实时推流钩子 (ws-client.js 调用) + 学习视图刷新 ---------- */
  var LIVE_VIEWS = { 'learn-mastery': renderBktHeatmap, 'learn': renderLearnRing, 'overview': null };
  var liveTimer = null;
  var liveView = null;

  function refreshLiveView() {
    if (!liveView) return;
    var fn = LIVE_VIEWS[liveView];
    if (liveView === 'overview') {
      var role = currentRole();
      fn = role === 'admin' ? renderAdminOverview : renderLearnerOverview;
    }
    if (fn) fn();
  }

  function startLiveRefresh() {
    // 仅登记当前实时视图; 不再设置盲刷新定时器.
    // 原 12s 轮询会整页重渲染并重置用户操作状态 (如 BKT "收起全部域" / 滚动位置),
    // 导致"收起几秒后失效". 现改为仅由 WS bkt_update 事件驱动按需刷新.
    stopLiveRefresh();
    liveView = currentView();
  }

  function stopLiveRefresh() {
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    liveView = null;
  }

  /* ---------- 溯源链 / 实体列表视图 (红绿灯显示, 分页优化) ---------- */
  function renderProvenanceView() {
    var ct = g('content');
    if (!ct) return;
    ct.innerHTML = '<div class="card"><h3>🔗 溯源链 · 实体列表</h3>' +
      '<div class="callout" style="margin-bottom:14px">各知识实体的溯源链完整性验证状态：<span style="color:var(--success)">● 已验证</span> · <span style="color:var(--warn)">● 部分验证</span> · <span style="color:var(--danger)">● 未验证</span></div>' +
      '<div id="provEntityList"><div class="loading"><span class="spinner"></span> 加载中…</div></div></div>';

    apiReq('GET', '/l3/entities').then(function (data) {
      var items = (data && data.items) || [];
      // 按 id/entity_id/name 去重, 避免同一实体重复列出
      (function () {
        var m = {}, o = [];
        items.forEach(function (x) {
          var k = (x && (x.id || x.entity_id || x.name)) || JSON.stringify(x);
          if (!m[k]) { m[k] = 1; o.push(x); }
        });
        items = o;
      })();
      if (!items.length) {
        g('provEntityList').innerHTML = '<p style="color:var(--muted);padding:20px 0">暂无实体数据</p>';
        return;
      }
      // 并发获取所有实体的溯源链
      Promise.all(items.map(function (e) {
        var eid = e.entity_id || e.id;
        return apiReq('GET', '/l3/quality/provenance/' + eid + '/chain')
          .then(function (ch) { return { entity: e, chain: ch }; })
          .catch(function () { return { entity: e, chain: null }; });
      })).then(function (results) {
        var PAGE_SIZE = 10;
        var currentPage = 0;
        var totalPages = Math.ceil(results.length / PAGE_SIZE);
        var unverifiedCount = results.filter(function (r) { return !r.chain || !r.chain.verified || r.chain.verified === 'unverifiable' || r.chain.verified === 'NONE'; }).length;
        var verifiedCount = results.filter(function (r) { var v = r.chain && r.chain.verified; return v === 'verified' || (typeof v === 'string' && v.toUpperCase() === 'VERIFIED'); }).length;
        var brokenCount = results.filter(function (r) { return r.chain && r.chain.verified && r.chain.verified === 'broken_chain'; }).length;

        // 生成单行 HTML
        function rowHtml(r) {
          var e = r.entity;
          var ch = r.chain;
          var eid = e.entity_id || e.id;
          var name = e.name || e.title || eid;
          var v = ch && ch.verified;
          var verified = v === 'verified' || (typeof v === 'string' && v.toUpperCase() === 'VERIFIED');
          var partial = v === 'broken_chain';
          var statusIcon, statusLabel, statusClass;
          if (verified) { statusIcon = '●'; statusLabel = '已验证'; statusClass = 'prov-green'; }
          else if (partial) { statusIcon = '●'; statusLabel = '部分验证'; statusClass = 'prov-yellow'; }
          else { statusIcon = '●'; statusLabel = '未验证'; statusClass = 'prov-red'; }
          var hash = (ch && ch.chain && ch.chain[0] && ch.chain[0].integrity_hash) || '';
          var chainLen = (ch && ch.chain && ch.chain.length) || 0;
          var entityType = e.entity_type || e.type || '知识实体';
          return '<div class="prov-entity-row" data-eid="' + escAttr(eid) + '">' +
            '<div class="prov-status ' + statusClass + '" title="' + statusLabel + '">' + statusIcon + '</div>' +
            '<div class="prov-info"><div class="prov-name">' + esc(name) + '</div>' +
            '<div class="prov-meta">' + esc(entityType) + ' · 链长 ' + chainLen + '</div></div>' +
            '<div class="prov-hash">' + (hash ? esc(hash.slice(0, 16)) + '…' : '—') + '</div>' +
            '<button class="btn ghost sm prov-detail-btn" data-eid="' + escAttr(eid) + '">详情</button></div>';
        }

        // 渲染指定页
        function renderPage(page) {
          var start = page * PAGE_SIZE;
          var end = Math.min(start + PAGE_SIZE, results.length);
          var pageRows = results.slice(start, end).map(rowHtml).join('');
          var paginationHtml = '';
          if (totalPages > 1) {
            var prevDisabled = page <= 0 ? 'disabled style="opacity:.4;cursor:default"' : '';
            var nextDisabled = page >= totalPages - 1 ? 'disabled style="opacity:.4;cursor:default"' : '';
            var pageBtns = [];
            for (var pi = 0; pi < totalPages; pi++) {
              pageBtns.push('<button class="btn ' + (pi === page ? 'primary' : 'ghost') + ' sm prov-page-btn" data-page="' + pi + '" style="margin:0 2px;min-width:28px">' + (pi + 1) + '</button>');
            }
            paginationHtml = '<div class="prov-pagination" style="display:flex;align-items:center;justify-content:center;gap:6px;padding:8px 0;border-bottom:1px solid var(--rule);flex-wrap:wrap">' +
              '<button class="btn ghost sm prov-page-prev" ' + prevDisabled + ' style="font-size:12px">‹ 上一页</button>' +
              pageBtns.join('') +
              '<button class="btn ghost sm prov-page-next" ' + nextDisabled + ' style="font-size:12px">下一页 ›</button>' +
              '<span style="font-size:11px;color:var(--muted);margin-left:8px">' + (start + 1) + '-' + end + ' / ' + results.length + '</span></div>';
          }
          var listHtml = paginationHtml + '<div class="prov-entity-list" style="' + (results.length > 10 ? 'max-height:420px;overflow-y:auto' : '') + '">' + pageRows + '</div>';

          g('provEntityList').innerHTML = listHtml +
            '<div class="prov-summary" style="margin-top:10px;padding:8px 14px;background:var(--surface2);border-radius:8px;display:flex;gap:20px;font-size:12px;flex-wrap:wrap">' +
            '<span><span style="color:var(--success)">●</span> 已验证: ' + verifiedCount + '</span>' +
            '<span><span style="color:var(--warn)">●</span> 链断裂: ' + brokenCount + '</span>' +
            '<span><span style="color:var(--danger)">●</span> 未验证: ' + unverifiedCount + '</span>' +
            '<span style="margin-left:auto;color:var(--muted)">共 ' + results.length + ' 个实体</span></div>';

          // 绑定分页事件
          var listEl = g('provEntityList');
          listEl.querySelectorAll('.prov-page-btn').forEach(function (b) {
            b.addEventListener('click', function () { renderPage(parseInt(this.dataset.page)); });
          });
          var prevBtn = listEl.querySelector('.prov-page-prev');
          if (prevBtn && !prevBtn.disabled) prevBtn.addEventListener('click', function () { renderPage(page - 1); });
          var nextBtn = listEl.querySelector('.prov-page-next');
          if (nextBtn && !nextBtn.disabled) nextBtn.addEventListener('click', function () { renderPage(page + 1); });
          // 绑定详情事件
          listEl.addEventListener('click', function (ev) {
            var btn = ev.target.closest('.prov-detail-btn, .prov-entity-row');
            if (!btn) return;
            var eid = btn.getAttribute('data-eid');
            if (!eid) return;
            var entity = items.find(function (i) { return (i.entity_id || i.id) === eid; });
            if (!entity) return;
            showProvDetail(entity, results.find(function (r) { return (r.entity.entity_id || r.entity.id) === eid; }));
          });
        }

        renderPage(0);
      });
    }).catch(function (e) {
      var el = g('provEntityList');
      if (el) el.innerHTML = '<div class="error-banner">加载失败: ' + esc(e.message) + '</div>';
    });
  }

  // 溯源链详情弹层
  function showProvDetail(entity, result) {
    var ch = result && result.chain;
    var eid = entity.entity_id || entity.id;
    var name = entity.name || entity.title || eid;
    var v = ch && ch.verified;
    var verified = v === 'verified' || (typeof v === 'string' && v.toUpperCase() === 'VERIFIED');
    var partial = v === 'broken_chain';
    var statusText = verified ? '✅ 已验证' : (partial ? '⚠️ 部分验证' : '❌ 未验证');
    var chainHtml = '';
    if (ch && ch.chain && ch.chain.length) {
      chainHtml = ch.chain.map(function (link, i) {
        return '<div class="prov-chain-step">' +
          '<div class="prov-step-num">' + (i + 1) + '</div>' +
          '<div class="prov-step-info">' +
          '<div style="font-weight:500;font-size:12px">' + esc(link.step_type || link.action || '步骤 ' + (i + 1)) + '</div>' +
          '<div style="font-size:11px;color:var(--muted)">' +
          (link.agent_id ? 'Agent: ' + esc(link.agent_id) + ' · ' : '') +
          (link.timestamp ? '时间: ' + esc(link.timestamp) : '') +
          '</div>' +
          (link.integrity_hash ? '<div style="font-size:10px;font-family:var(--mono);color:var(--muted);word-break:break-all">哈希: ' + esc(link.integrity_hash) + '</div>' : '') +
          '</div></div>';
      }).join('');
    } else {
      chainHtml = '<p style="color:var(--muted);font-size:12px;padding:10px 0">无溯源链记录</p>';
    }
    var wrap = d.createElement('div');
    wrap.innerHTML = '<div style="position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99;display:flex;align-items:center;justify-content:center" id="provDetailModal">' +
      '<div style="background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:18px 22px;max-width:560px;width:92%;max-height:80vh;overflow:auto;box-shadow:0 12px 40px rgba(0,0,0,.25)">' +
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">' +
      '<div><div style="font-weight:700;font-size:15px">' + esc(name) + '</div>' +
      '<div style="font-size:11px;color:var(--muted)">' + esc(eid) + '</div></div>' +
      '<div><span class="badge ' + (verified ? 'ok' : (partial ? 'warn' : 'err')) + '">' + statusText + '</span>' +
      '<button id="provClose" class="btn ghost" style="padding:4px 10px;margin-left:10px">✕</button></div></div>' +
      '<div class="prov-chain-steps" style="margin-top:12px">' + chainHtml + '</div></div></div>';
    document.body.appendChild(wrap.firstChild);
    var c = g('provClose');
    if (c) c.addEventListener('click', function () { var w = g('provDetailModal'); if (w) w.remove(); });
    g('provDetailModal').addEventListener('click', function (e) { if (e.target === this) this.remove(); });
  }

  /* ---------- 初始化 ---------- */
  function init() {
    hookViewRendering();
    // 导航注入改为事件驱动: app.js rs() 重建侧边栏后派发 sidebar-rebuilt,
    // 本函数监听该事件注入 practice/对比/时间旅行/溯源链 入口 (消除轮询重复)
    _injectNavOnce();
    d.addEventListener('sidebar-rebuilt', function () { _injectNavOnce(); });
    // settings 已由 VIEW_OVERRIDES 渲染到内容区, 不再用旧弹窗委托拦截
    setupResponsiveNav();
    // WS broadcast 通道 bkt_update 实时刷新 (由 ws-client.js 触发)
    window.Dy3BKT = {
      onUpdate: function (payload) {
        console.log('[Dy3] bkt_update received', payload && payload.kp_id);
        // 学习类视图停留时立即刷新 (动态可视化反馈)
        var v = currentView();
        if (v === 'learn-mastery' || v === 'learn' || v === 'overview' || v === 'practice') {
          refreshLiveView();
        }
      }
    };
    // 视图切换时启停学习视图轮询 (事件驱动: sv() 每次切换后派发 view-rendered)
    d.addEventListener('view-rendered', function (e) {
      var nv = (e && e.detail && e.detail.view) || currentView();
      if (LIVE_VIEWS.hasOwnProperty(nv) || nv === 'overview') startLiveRefresh();
      else stopLiveRefresh();
    });
    // 初始: 当前视图如果属于实时刷新类则启动
    var iv = currentView();
    if (LIVE_VIEWS.hasOwnProperty(iv) || iv === 'overview') startLiveRefresh();
  }

  // Provider secrets are intentionally not restored from browser storage.

  window.DY3CanvasInternals = {
    apiReq: apiReq,
    learnerId: learnerId,
    publicData: r04PublicData,
    taskProjection: r08TaskProjection,
    renderSlideDeck: renderSlideDeck,
    renderResources: t1ResourceCards,
    bindResourceActions: t1BindResourceActions,
    renderConceptGraph: t234ConceptGraph,
    renderScientificGrounding: t5678ScientificGrounding,
    renderChallenge: r04RenderChallenge,
    renderTrace: r04RenderTrace,
    text: r04Text,
    sourceReference: r04SourceReference
  };

  if (d.readyState === 'loading') {
    d.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
