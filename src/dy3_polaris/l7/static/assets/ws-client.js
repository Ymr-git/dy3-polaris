/* Dy3+ Polaris — WebSocket 三通道实时客户端 (M-F5)
 * 通道: stream(artifact) / broadcast(bkt_update) / debate(辩论实时)
 * 韧性: 心跳 30s · 指数退避重连 1→2→4→8→16→30s · 单用户 3 连接
 * 认证: ?token={JWT}
 */
(function () {
  'use strict';

  var BACKOFF = [1000, 2000, 4000, 8000, 16000, 30000];
  var HEARTBEAT_MS = 30000;

  /* ---------- 内部状态 ---------- */
  var conns = {};      // channel -> {ws, alive, retry, timer, heartbeatTimer, handlers}
  var debateState = { speeches: [], rounds: [], consensus: [], verdict: null, started: false };

  /* ---------- 工具 ---------- */
  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function token() {
    return localStorage.getItem('dt') || localStorage.getItem('dy3_access_token') || '';
  }
  function toast(msg) {
    var t = document.getElementById('toast');
    if (!t) return;
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(function () { t.hidden = true; }, 2600);
  }
  function send(ws, obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify(obj)); } catch (e) { /* 忽略 */ }
    }
  }

  /* ---------- 单通道连接管理 ---------- */
  function connect(channel, handlers) {
    if (conns[channel] && conns[channel].ws && conns[channel].ws.readyState <= 1) {
      return; // 已在连接中
    }
    var state = conns[channel] || (conns[channel] = { retry: 0, handlers: handlers || {} });
    state.handlers = handlers || state.handlers;
    clearTimeout(state.retryTimer);

    var proto = location.protocol === 'https:' ? 'wss' : 'ws';
    var url = proto + '://' + location.host + '/ws/' + channel + '?token=' + encodeURIComponent(token());
    var ws;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      scheduleRetry(channel);
      return;
    }
    state.ws = ws;
    state.alive = true;

    ws.onopen = function () {
      state.retry = 0;
      state.heartbeatTimer = setInterval(function () {
        send(ws, { type: 'ping' });
      }, HEARTBEAT_MS);
      if (state.handlers.onopen) state.handlers.onopen();
    };

    ws.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type === 'pong') return; // 心跳应答
      if (msg.event_type) {
        if (state.handlers.onEvent) state.handlers.onEvent(msg);
        // 全局事件钩子 (debate 面板 / bkt 热力图等)
        if (channel === 'debate' && window.Dy3Debate && window.Dy3Debate.onEvent) {
          window.Dy3Debate.onEvent(msg);
        }
        if (channel === 'broadcast' && msg.event_type === 'bkt_update' && window.Dy3BKT) {
          window.Dy3BKT.onUpdate(msg.payload);
        }
      }
    };

    ws.onclose = function () {
      clearInterval(state.heartbeatTimer);
      state.alive = false;
      if (state.handlers.onclose) state.handlers.onclose();
      scheduleRetry(channel);
    };

    ws.onerror = function () { try { ws.close(); } catch (e) { /* 忽略 */ } };
  }

  function scheduleRetry(channel) {
    var state = conns[channel];
    if (!state) return;
    var delay = BACKOFF[Math.min(state.retry, BACKOFF.length - 1)];
    state.retry += 1;
    clearTimeout(state.retryTimer);
    state.retryTimer = setTimeout(function () { connect(channel, state.handlers); }, delay);
  }

  function disconnect(channel) {
    var state = conns[channel];
    if (!state) return;
    clearTimeout(state.retryTimer);
    clearInterval(state.heartbeatTimer);
    state.retry = 0;
    if (state.ws) {
      try { send(state.ws, { type: 'close' }); state.ws.close(); } catch (e) { /* 忽略 */ }
    }
    state.alive = false;
  }

  function stopAll() {
    Object.keys(conns).forEach(disconnect);
  }

  /* ---------- 公开 API ---------- */
  window.Dy3WS = {
    connect: connect,
    disconnect: disconnect,
    stopAll: stopAll,
    isAlive: function (channel) {
      var s = conns[channel];
      return !!(s && s.ws && s.ws.readyState === WebSocket.OPEN);
    },
  };

  /* ---------- 辩论实时面板 (debate view) ---------- */
  var STANCE_LABEL = { support: '支持', oppose: '反对', neutral: '中立' };
  // 文本色随主题 (暗色下自动提亮)；徽章底色用固定深色保证白字对比 (WCAG AA)
  var STANCE_COLOR = { support: 'var(--success)', oppose: 'var(--danger)', neutral: 'var(--muted)' };
  var STANCE_BG = { support: '#15803d', oppose: '#dc2626', neutral: '#64748b' };

  window.Dy3Debate = {
    onEvent: function (msg) {
      var p = msg.payload || {};
      if (msg.event_type === 'debate_start') {
        debateState = { speeches: [], rounds: [], consensus: [], verdict: null, started: true, topic: p.topic };
        if (p.agents) debateState.agents = p.agents;
      } else if (msg.event_type === 'speech') {
        debateState.speeches.push({ agent: p.agent, stance: p.stance, summary: p.summary, ts: p.timestamp });
      } else if (msg.event_type === 'convergence') {
        debateState.rounds = p.rounds || debateState.rounds;
        debateState.consensus = p.consensus || debateState.consensus;
      } else if (msg.event_type === 'end') {
        debateState.verdict = p;
        debateState.started = false;
      }
      renderDebate();
    },
    getState: function () { return debateState; },
  };

  function renderDebate() {
    var host = document.getElementById('debatePanel');
    if (!host || host.hidden) return;

    var s = debateState;
    var html = '<div class="card">';
    html += '<h3>辩论实时面板</h3>';
    if (s.topic) html += '<p class="debate-topic">议题：' + esc(s.topic) + '</p>';

    // 状态徽章
    var badge = s.verdict ? '裁决完成' : (s.started ? '进行中' : '待启动');
    html += '<div class="debate-status"><span class="badge ' + (s.verdict ? 'ok' : 'warn') + '">' + badge + '</span></div>';

    // 时间线
    html += '<h4 style="margin:14px 0 8px">发言时间线 (' + s.speeches.length + ')</h4>';
    if (s.speeches.length) {
      html += '<div class="debate-timeline">' + s.speeches.map(function (sp, i) {
        var c = STANCE_COLOR[sp.stance] || 'var(--muted)';
        var bg = STANCE_BG[sp.stance] || '#94a3b8';
        return '<div class="debate-speech"><span class="debate-agent" style="color:' + c + '">' +
          esc(sp.agent || 'A' + (i + 1)) + '</span>' +
          '<span class="debate-stance" style="background:' + bg + '">' + (STANCE_LABEL[sp.stance] || sp.stance) + '</span>' +
          '<span class="debate-summary">' + esc(sp.summary || '') + '</span></div>';
      }).join('') + '</div>';
    } else {
      html += '<p style="color:var(--muted);font-size:12.5px">等待 Agent 发言…</p>';
    }

    // 共识收敛折线 (内联 SVG)
    if (s.rounds.length && s.consensus.length) {
      html += '<h4 style="margin:14px 0 8px">共识收敛曲线</h4>' + buildConvergenceSvg(s.rounds, s.consensus);
    }

    // 裁决结果
    if (s.verdict) {
      html += '<h4 style="margin:14px 0 8px">裁决结果</h4>';
      html += '<div class="callout">' + esc(s.verdict.summary || '') +
        (s.verdict.selected_agent ? ' — 采纳 Agent：<strong>' + esc(s.verdict.selected_agent) + '</strong>' : '') + '</div>';
      if (s.verdict.dimensions && s.verdict.dimensions.length) {
        html += '<div class="grid cols-3" style="margin-top:10px">' + s.verdict.dimensions.map(function (dim) {
          return '<div class="stat-card"><div class="lbl">' + esc(dim.name || '') + '</div><div class="num">' +
            (dim.value != null ? Math.round(dim.value * 100) / 100 : '-') + '</div></div>';
        }).join('') + '</div>';
      }
    }
    html += '</div>';
    host.innerHTML = html;
  }

  function buildConvergenceSvg(rounds, consensus) {
    var w = 600, h = 120, pad = 30;
    var minR = Math.min.apply(null, rounds), maxR = Math.max.apply(null, rounds);
    var minC = 0, maxC = 1;
    if (rounds.length === 1) { minR = rounds[0] - 1; maxR = rounds[0] + 1; }
    var x = function (r) { return pad + (r - minR) / (maxR - minR || 1) * (w - pad * 2); };
    var y = function (c) { return h - pad - (c - minC) / (maxC - minC || 1) * (h - pad * 2); };
    var pts = rounds.map(function (r, i) { return x(r).toFixed(1) + ',' + y(consensus[i]).toFixed(1); }).join(' ');
    var labels = rounds.map(function (r) {
      return '<text x="' + x(r).toFixed(1) + '" y="' + (h - 8) + '" font-size="9" fill="#6b7280" text-anchor="middle">R' + r + '</text>';
    }).join('');
    return '<svg viewBox="0 0 ' + w + ' ' + h + '" style="width:100%;max-width:600px;height:auto;background:var(--surface);border-radius:8px">' +
      '<line x1="' + pad + '" y1="' + (h - pad) + '" x2="' + (w - pad) + '" y2="' + (h - pad) + '" stroke="#6b7280"/>' +
      '<line x1="' + pad + '" y1="' + pad + '" x2="' + pad + '" y2="' + (h - pad) + '" stroke="#6b7280"/>' +
      '<polyline points="' + pts + '" fill="none" stroke="#d97706" stroke-width="2"/>' +
      rounds.map(function (r, i) {
        return '<circle cx="' + x(r).toFixed(1) + '" cy="' + y(consensus[i]).toFixed(1) + '" r="3.5" fill="#d97706"/>';
      }).join('') + labels + '</svg>';
  }

  /* ---------- 辩论视图挂载 ---------- */
  function mountDebatePanel() {
    // 由 app.js 的 sv() 在切换到 debate 视图时调用 window.Dy3Debate.show()
  }

  window.Dy3Debate.show = function () {
    var host = document.getElementById('debatePanel');
    if (!host) return;
    host.hidden = false;
    renderDebate();
    if (!window.Dy3WS.isAlive('debate')) {
      window.Dy3WS.connect('debate', {});
    }
  };

  window.Dy3Debate.hide = function () {
    var host = document.getElementById('debatePanel');
    if (host) host.hidden = true;
  };

  /* ---------- 自动连接 (登录后 / 已有登录态回访也连接) ---------- */
  var lastToken = '';
  function syncConn() {
    var t = token();
    if (t && t !== lastToken) {
      lastToken = t;
      // 登录成功 / 初始已登录 → 自动连接通道
      connect('stream', {});
      connect('broadcast', {});
    } else if (!t && lastToken) {
      lastToken = '';
      stopAll();
    }
  }
  syncConn();  // 立即连接一次, 处理「刷新页面时已有 token」的情况
  setInterval(syncConn, 2000);

  window.addEventListener('beforeunload', stopAll);
})();
