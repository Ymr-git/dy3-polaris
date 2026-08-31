/* Dy3+ Polaris — 多智能体协同流程可视化组件
 * 统一美观的协作链路展示: 四 Agent 流水线 + 广播通道 + 时序
 * 供小助手(mf7) 与 交互详情页 复用, 取代此前分散的 chips 展示。
 * 纯原生 SVG + CSS, 离线可用。
 */
(function () {
  'use strict';
  var d = document;

  var AGENT_META = {
    'agent.learning.diagnosis': { name: '学情诊断', icon: '🔍', color: '#3b82f6' },
    'agent.knowledge.generation': { name: '知识生成', icon: '🧠', color: '#8b5cf6' },
    'agent.quality.review': { name: '审核校验', icon: '🛡️', color: '#10b981' },
    'agent.guidance.decision': { name: '导学决策', icon: '🎯', color: '#f59e0b' },
    'cross.validate': { name: '交叉验证', icon: '⚖️', color: '#6366f1' },
    'debate.pro': { name: '正方辩论', icon: '💬', color: '#0ea5e9' },
    'debate.con': { name: '反方辩论', icon: '💬', color: '#f43f5e' },
    'debate.vote': { name: '投票裁决', icon: '🗳️', color: '#14b8a6' },
    'agent.adjudicator': { name: '仲裁者', icon: '⚖️', color: '#eab308' },
  };
  function meta(agent) {
    return AGENT_META[agent] || { name: String(agent || '').replace('agent.', ''), icon: '🤖', color: '#6b7280' };
  }

  /* ================= 1. 四 Agent 流水线 (横向 step 流程) ================= */
  function renderPipeline(flowEvents) {
    if (!flowEvents || !flowEvents.length) return '';
    var steps = flowEvents.map(function (ev, i) {
      var m = meta(ev.agent);
      var last = i === flowEvents.length - 1;
      var arrow = last ? '' : '<div class="dp-pipe-arrow">→</div>';
      return '<div class="dp-pipe-step" style="--ac:' + m.color + '">' +
        '<div class="dp-pipe-icon">' + m.icon + '</div>' +
        '<div class="dp-pipe-body">' +
        '<div class="dp-pipe-name">' + (m.name) + '</div>' +
        '<div class="dp-pipe-label">' + (ev.label || ev.step || '') + '</div>' +
        '</div>' +
        (ev.elapsed_ms != null ? '<div class="dp-pipe-time">' + Math.round(ev.elapsed_ms) + 'ms</div>' : '') +
        '</div>' + arrow;
    }).join('');
    return '<div class="dp-pipeline">' + steps + '</div>';
  }

  /* ================= 2. 广播通道 (publisher → channel → to) ================= */
  function renderBroadcast(broadcastEvents) {
    if (!broadcastEvents || !broadcastEvents.length) return '';
    var rows = broadcastEvents.map(function (bc) {
      var m = meta(bc.publisher);
      return '<div class="dp-broadcast-row">' +
        '<span class="dp-bc-pub" style="--ac:' + m.color + '">' + m.icon + ' ' + m.name + '</span>' +
        '<span class="dp-bc-arrow">→</span>' +
        '<code class="dp-bc-channel">' + (bc.channel || '') + '</code>' +
        '<span class="dp-bc-arrow">⇥</span>' +
        '<span class="dp-bc-to">' + (bc.to || '') + '</span>' +
        '</div>';
    }).join('');
    return '<div class="dp-broadcast">' + rows + '</div>';
  }

  /* ================= 3. 完整协同卡片 (对外主入口) ================= */
  function renderCollaboration(flowEvents, broadcastEvents, extra) {
    extra = extra || {};
    var html = '';
    if (flowEvents && flowEvents.length) {
      html += '<div class="dp-collab-title">🔄 多智能体协同链路</div>' + renderPipeline(flowEvents);
    }
    if (broadcastEvents && broadcastEvents.length) {
      html += '<div class="dp-collab-title" style="margin-top:10px">📡 广播通道（Agent 间协作证据）</div>' + renderBroadcast(broadcastEvents);
    }
    if (extra.selfCorrection) {
      html += '<div class="dp-collab-title" style="margin-top:10px">♻️ 自纠回路</div>' +
        '<div class="dp-selfcorr">审核发现问题 → 生成修订 → 终审通过（多候选交叉验证）</div>';
    }
    if (extra.consensus != null) {
      html += '<div class="dp-collab-title" style="margin-top:10px">⚖️ 共识度</div>' +
        '<div class="dp-consensus"><div class="dp-consensus-bar"><div class="dp-consensus-fill" style="width:' + Math.round(extra.consensus * 100) + '%"></div></div><span class="dp-consensus-val">' + Math.round(extra.consensus * 100) + '%</span></div>';
    }
    if (!html) return '';
    return '<div class="dp-collab-card">' + html + '</div>';
  }

  /* ================= 注入样式 ================= */
  function injectStyle() {
    if (d.getElementById('dpCollabCss')) return;
    var css = d.createElement('style');
    css.id = 'dpCollabCss';
    css.textContent = [
      '.dp-collab-card{background:linear-gradient(135deg,#eef2ff,#fafafa);border:1px solid #e0e7ff;border-radius:12px;padding:12px 14px;margin:8px 0}',
      '[data-theme="dark"] .dp-collab-card{background:linear-gradient(135deg,#1e1b4b,#1a1a1a);border-color:#312e81}',
      '.dp-collab-title{font-size:12px;font-weight:600;color:#4338ca;margin-bottom:8px}',
      '[data-theme="dark"] .dp-collab-title{color:#a5b4fc}',
      '.dp-pipeline{display:flex;align-items:stretch;gap:6px;flex-wrap:wrap}',
      '.dp-pipe-step{display:flex;align-items:center;gap:8px;background:#fff;border:1px solid #e0e7ff;border-left:3px solid var(--ac,#6366f1);border-radius:8px;padding:8px 10px;min-width:120px;transition:all .15s}',
      '[data-theme="dark"] .dp-pipe-step{background:#27272a;border-color:#312e81}',
      '.dp-pipe-step:hover{box-shadow:0 2px 8px rgba(99,102,241,.2);transform:translateY(-1px)}',
      '.dp-pipe-icon{font-size:16px;flex:none}',
      '.dp-pipe-body{flex:1;min-width:0}',
      '.dp-pipe-name{font-size:12.5px;font-weight:600;color:#1e293b}',
      '[data-theme="dark"] .dp-pipe-name{color:#e5e5e5}',
      '.dp-pipe-label{font-size:10.5px;color:#6b7280;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:140px}',
      '.dp-pipe-time{font-size:10px;color:#9ca3af;flex:none;font-family:var(--mono,monospace)}',
      '.dp-pipe-arrow{display:flex;align-items:center;color:#a5b4fc;font-size:16px;flex:none}',
      '.dp-broadcast{display:flex;flex-direction:column;gap:4px}',
      '.dp-broadcast-row{display:flex;align-items:center;gap:6px;font-size:11px;flex-wrap:wrap}',
      '.dp-bc-pub{font-weight:600;padding:1px 8px;border-radius:999px;background:#eef2ff;color:#4338ca;border-left:2px solid var(--ac,#6366f1)}',
      '[data-theme="dark"] .dp-bc-pub{background:#312e81;color:#a5b4fc}',
      '.dp-bc-arrow{color:#9ca3af}',
      '.dp-bc-channel{font-family:var(--mono,monospace);font-size:10.5px;background:#f4f4f5;padding:1px 6px;border-radius:4px;color:#7c3aed}',
      '[data-theme="dark"] .dp-bc-channel{background:#1f1f23;color:#c4b5fd}',
      '.dp-bc-to{color:#6b7280}',
      '.dp-selfcorr{font-size:11px;color:#6b7280;line-height:1.6}',
      '.dp-consensus{display:flex;align-items:center;gap:8px}',
      '.dp-consensus-bar{flex:1;height:8px;border-radius:4px;background:#e0e7ff;overflow:hidden}',
      '[data-theme="dark"] .dp-consensus-bar{background:#27272a}',
      '.dp-consensus-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#6366f1,#8b5cf6);transition:width .5s ease}',
      '.dp-consensus-val{font-size:12px;font-weight:700;color:#6366f1;flex:none}',
    ].join('\n');
    d.head.appendChild(css);
  }
  injectStyle();

  /* 全局导出 */
  window.DPCollab = window.DPCollab || {};
  window.DPCollab.renderPipeline = renderPipeline;
  window.DPCollab.renderBroadcast = renderBroadcast;
  window.DPCollab.renderCollaboration = renderCollaboration;
  window.DPCollab.meta = meta;
})();
