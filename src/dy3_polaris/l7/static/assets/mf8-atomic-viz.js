/* Dy3+ Polaris — M-F8 原子可视化 (能级图 / 电子云形状 / 跃迁概率 动态可视化)
 * 数据驱动: 不同的能级/跃迁数据、不同的量子数 → 生成不同的图形.
 * 独立于 mf6, 通过 'view-rendered' 事件接管 'atomic-viz' 视图.
 * 版本: 2026081218
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
  function isDark() { return (d.documentElement.getAttribute('data-theme') || 'light') === 'dark'; }
  function fmt(v) { return (Math.round(v * 100) / 100); }
  function isAdminRole() { var r = localStorage.getItem('dr') || 'student'; return r === 'teacher' || r === 'admin'; }

  var VIEW = 'atomic-viz';

  /* ================= 数据预设 (不同数据 → 不同图形) ================= */
  var PRESETS = {
    dy: {
      ion: 'Dy³⁺', name: '镝离子', config: '[Xe] 4f⁹', ground: '⁶H₁₅/₂', maxE: 23000,
      note: '黄光主发射：⁴F₉/₂ → ⁶H₁₅/₂（约 575 nm）',
      levels: [
        { label: '⁶H₁₅/₂', energy: 0, j: '15/2', deg: 16, color: '#22c55e' },
        { label: '⁶H₁₃/₂', energy: 3600, j: '13/2', deg: 14, color: '#84cc16' },
        { label: '⁶H₁₁/₂', energy: 6200, j: '11/2', deg: 12, color: '#eab308' },
        { label: '⁴I₁₅/₂', energy: 18500, j: '15/2', deg: 16, color: '#f97316' },
        { label: '⁴F₉/₂', energy: 21100, j: '9/2', deg: 10, color: '#3b82f6' },
        { label: '⁴F₇/₂', energy: 22400, j: '7/2', deg: 8, color: '#8b5cf6' }
      ],
      transitions: [
        { from: '⁴F₉/₂', to: '⁶H₁₅/₂', prob: 0.70, wl: 575, photon: '黄光' },
        { from: '⁴F₉/₂', to: '⁶H₁₃/₂', prob: 0.15, wl: 660, photon: '红光' },
        { from: '⁴F₉/₂', to: '⁶H₁₁/₂', prob: 0.08, wl: 760, photon: '近红外' },
        { from: '⁴F₇/₂', to: '⁶H₁₅/₂', prob: 0.42, wl: 500, photon: '绿光' },
        { from: '⁴I₁₅/₂', to: '⁶H₁₅/₂', prob: 0.48, wl: 480, photon: '蓝光' }
      ]
    },
    eu: {
      ion: 'Eu³⁺', name: '铕离子', config: '[Xe] 4f⁶', ground: '⁷F₀', maxE: 21000,
      note: '红光主发射：⁵D₀ → ⁷F₂（约 612 nm）',
      levels: [
        { label: '⁷F₀', energy: 0, j: '0', deg: 1, color: '#22c55e' },
        { label: '⁷F₁', energy: 360, j: '1', deg: 3, color: '#84cc16' },
        { label: '⁷F₂', energy: 1100, j: '2', deg: 5, color: '#eab308' },
        { label: '⁷F₃', energy: 1900, j: '3', deg: 7, color: '#f97316' },
        { label: '⁵D₀', energy: 17200, j: '0', deg: 1, color: '#dc2626' },
        { label: '⁵D₁', energy: 19000, j: '1', deg: 3, color: '#3b82f6' }
      ],
      transitions: [
        { from: '⁵D₀', to: '⁷F₂', prob: 0.62, wl: 612, photon: '红光' },
        { from: '⁵D₀', to: '⁷F₁', prob: 0.28, wl: 590, photon: '橙光' },
        { from: '⁵D₀', to: '⁷F₀', prob: 0.05, wl: 580, photon: '橙光' },
        { from: '⁵D₁', to: '⁷F₁', prob: 0.30, wl: 535, photon: '绿光' }
      ]
    },
    ce: {
      ion: 'Ce³⁺', name: '铈离子', config: '[Xe] 4f¹', ground: '²F₅/₂', maxE: 30000,
      note: '5d→4f 宽带发射（偶极允许，寿命短）',
      levels: [
        { label: '²F₅/₂', energy: 0, j: '5/2', deg: 6, color: '#22c55e' },
        { label: '²F₇/₂', energy: 2200, j: '7/2', deg: 8, color: '#84cc16' },
        { label: '5d(t₂g)', energy: 27000, j: '', deg: 6, color: '#3b82f6' },
        { label: '5d(eg)', energy: 32000, j: '', deg: 4, color: '#8b5cf6' }
      ],
      transitions: [
        { from: '5d(t₂g)', to: '²F₅/₂', prob: 0.55, wl: 420, photon: '蓝紫光' },
        { from: '5d(t₂g)', to: '²F₇/₂', prob: 0.30, wl: 480, photon: '蓝光' },
        { from: '5d(eg)', to: '²F₅/₂', prob: 0.35, wl: 350, photon: '紫外' }
      ]
    },
    tb: {
      ion: 'Tb³⁺', name: '铽离子', config: '[Xe] 4f⁸', ground: '⁷F₆', maxE: 21000,
      note: '绿光主发射：⁵D₄ → ⁷F₅（约 545 nm）',
      levels: [
        { label: '⁷F₆', energy: 0, j: '6', deg: 13, color: '#22c55e' },
        { label: '⁷F₅', energy: 2100, j: '5', deg: 11, color: '#84cc16' },
        { label: '⁷F₄', energy: 3400, j: '4', deg: 9, color: '#eab308' },
        { label: '⁵D₄', energy: 20500, j: '4', deg: 9, color: '#10b981' }
      ],
      transitions: [
        { from: '⁵D₄', to: '⁷F₅', prob: 0.55, wl: 545, photon: '绿光' },
        { from: '⁵D₄', to: '⁷F₄', prob: 0.20, wl: 585, photon: '黄光' },
        { from: '⁵D₄', to: '⁷F₆', prob: 0.15, wl: 490, photon: '蓝绿光' }
      ]
    }
  };

  /* 电子云轨道预设 (量子数 n,l → 轨道形状; 数据驱动) */
  var ORBITALS = [
    { id: '1s', label: '1s', n: 1, l: 0, kind: 'm0', rhoMax: 1, color: '#3b82f6' },
    { id: '2s', label: '2s', n: 2, l: 0, kind: 'm0', rhoMax: 5, color: '#6366f1' },
    { id: '2pz', label: '2pz', n: 2, l: 1, kind: 'm0', rhoMax: 4.2, color: '#f59e0b' },
    { id: '2px', label: '2px', n: 2, l: 1, kind: 'cx', rhoMax: 4.2, color: '#ec4899' },
    { id: '3dz2', label: '3dz²', n: 3, l: 2, kind: 'm0', rhoMax: 9, color: '#10b981' },
    { id: '3dxy', label: '3dxy', n: 3, l: 2, kind: 'cxy', rhoMax: 9, color: '#8b5cf6' },
    { id: '4fz3', label: '4fz³', n: 4, l: 3, kind: 'm0', rhoMax: 16, color: '#06b6d4' }
  ];

  /* ================= 工具: 实球谐函数 + 氢原子径向函数 ================= */
  function angularLm(l, kind, th, ph) {
    var c = Math.cos(th), s = Math.sin(th), cf = Math.cos(ph), sf = Math.sin(ph);
    var S = Math.sqrt;
    switch (l) {
      case 0: return 0.5 / S(Math.PI);
      case 1:
        if (kind === 'cx') return S(3 / (4 * Math.PI)) * s * cf;
        if (kind === 'cy') return S(3 / (4 * Math.PI)) * s * sf;
        return S(3 / (4 * Math.PI)) * c;
      case 2:
        if (kind === 'cxz') return S(15 / (4 * Math.PI)) * s * c * cf;
        if (kind === 'cxy') return S(15 / (16 * Math.PI)) * s * s * Math.sin(2 * ph);
        if (kind === 'cxy2') return S(15 / (16 * Math.PI)) * s * s * Math.cos(2 * ph);
        return S(5 / (16 * Math.PI)) * (3 * c * c - 1);
      case 3:
        // fz³ 及若干 f 轨道形貌 (教学简化)
        if (kind === 'f2') return S(105 / (4 * Math.PI)) * s * c * c * cf;
        if (kind === 'f3') return S(105 / (16 * Math.PI)) * s * s * c * Math.cos(2 * ph);
        return S(7 / (16 * Math.PI)) * (5 * c * c * c - 3 * c);
      default: return 1;
    }
  }
  function radialR(n, l, rho) {
    var o;
    if (n === 1 && l === 0) o = 2 * Math.exp(-rho);
    else if (n === 2 && l === 0) o = (2 - rho) * Math.exp(-rho / 2);
    else if (n === 2 && l === 1) o = rho * Math.exp(-rho / 2);
    else if (n === 3 && l === 0) o = (27 - 18 * rho + 2 * rho * rho) * Math.exp(-rho / 3);
    else if (n === 3 && l === 1) o = (6 * rho - rho * rho) * Math.exp(-rho / 3);
    else if (n === 3 && l === 2) o = rho * rho * Math.exp(-rho / 3);
    else if (n === 4 && l === 0) o = (192 - 144 * rho + 24 * rho * rho - rho * rho * rho) * Math.exp(-rho / 4);
    else if (n === 4 && l === 1) o = (80 * rho - 20 * rho * rho + rho * rho * rho) * Math.exp(-rho / 4);
    else if (n === 4 && l === 2) o = (12 * rho * rho - rho * rho * rho) * Math.exp(-rho / 4);
    else if (n === 4 && l === 3) o = rho * rho * rho * Math.exp(-rho / 4);
    else o = Math.exp(-rho);
    return o;
  }

  /* ================= 主渲染入口 ================= */
  function render() {
    var ct = g('content');
    if (!ct) return;
    ct.setAttribute('data-mf8', VIEW);
    if (isDark()) ct.setAttribute('data-mf8-dark', '1'); else ct.removeAttribute('data-mf8-dark');
    ct.innerHTML =
      '<div class="r08-context-return"><button class="btn ghost" id="mf8BackKnowledge">← 返回知识与证据</button><span>科学机制示意，不代表当前任务的 Agent 运行事实</span></div>' +
      '<div class="card" data-mf8-root>' +
      '<h3 style="margin:0 0 4px">⚛️ 原子可视化实验台</h3>' +
      '<p style="color:var(--muted);font-size:12.5px;margin:0 0 12px">数据驱动的能级图 / 电子云形状 / 跃迁概率动态可视化 · 切换离子或修改数据即可生成不同图形</p>' +
      /* 离子预设 + 自定义数据 */
      '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:4px" id="mf8IonChips"></div>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">' +
      '<button class="btn ghost" id="mf8CustomBtn">✏️ 自定义数据</button>' +
      '<span style="align-self:center;font-size:12px;color:var(--muted)" id="mf8IonNote"></span>' +
      '</div>' +
      /* 标签页 */
      '<div style="display:flex;gap:6px;margin-bottom:12px;border-bottom:1px solid var(--rule);padding-bottom:8px">' +
      '<button class="btn mf8-tab active" data-tab="energy">能级图</button>' +
      '<button class="btn mf8-tab" data-tab="cloud">电子云形状</button>' +
      '<button class="btn mf8-tab" data-tab="trans">跃迁概率</button>' +
      '</div>' +
      '<div id="mf8Energy"></div>' +
      '<div id="mf8Cloud" hidden></div>' +
      '<div id="mf8Trans" hidden></div>' +
      '</div>';

    var back = g('mf8BackKnowledge');
    if (back) back.addEventListener('click', function () { if (window.sv) window.sv('kb'); });

    // 离子预设 chips
    var chipBox = g('mf8IonChips');
    if (chipBox) {
      var chips = '';
      Object.keys(PRESETS).forEach(function (k) {
        var p = PRESETS[k];
        chips += '<button class="q-clr-chip mf8-ion-chip" data-ion="' + k + '">' + esc(p.ion) + ' ' + esc(p.name) + '</button>';
      });
      chipBox.innerHTML = chips;
      chipBox.querySelectorAll('.mf8-ion-chip').forEach(function (b) {
        b.addEventListener('click', function () {
          var ion = b.getAttribute('data-ion');
          setIon(ion);
          b.parentNode.querySelectorAll('.mf8-ion-chip').forEach(function (x) { x.style.borderColor = ''; x.style.background = ''; b.style.borderColor = 'var(--accent)'; b.style.background = 'var(--accent-soft)'; });
        });
      });
    }

    // 标签页切换
    d.querySelectorAll('.mf8-tab').forEach(function (b) {
      b.addEventListener('click', function () {
        var tab = b.getAttribute('data-tab');
        d.querySelectorAll('.mf8-tab').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        g('mf8Energy').hidden = tab !== 'energy';
        g('mf8Cloud').hidden = tab !== 'cloud';
        g('mf8Trans').hidden = tab !== 'trans';
        if (tab === 'energy') renderEnergy(myData());
        if (tab === 'trans') renderTrans(myData());
        if (tab === 'cloud') startCloud();
      });
    });

    // 自定义数据按钮
    var cb = g('mf8CustomBtn');
    if (cb) cb.addEventListener('click', openCustomEditor);

    setIon('dy');

    // 若存在外部注入的待渲染数据 (如小助手/问答识别出可视化意图), 优先渲染它
    var pend = window.MF8Viz && window.MF8Viz._pending;
    if (pend) {
      window.MF8Viz._pending = null;
      _customData = pend.data;
      _currentIon = 'custom';
      renderEnergy(_customData);
      renderTrans(_customData);
      startCloud();
      switchTab(pend.tab || 'energy');
    }
  }

  var _currentIon = 'dy';
  var _customData = null;
  function myData() { return _customData || PRESETS[_currentIon] || PRESETS.dy; }
  function setIon(ion) {
    _currentIon = ion;
    _customData = null;
    var note = g('mf8IonNote');
    if (note) note.textContent = PRESETS[ion].config + ' · ' + PRESETS[ion].note;
    renderEnergy(myData());
    renderTrans(myData());
    startCloud();
  }

  /* ================= 能级图 (数据驱动, 动态动画) ================= */
  var _energyAnim = null;
  function renderEnergy(data, box) {
    if (!box) box = g('mf8Energy');
    if (!box) return;
    if (_energyAnim) { cancelAnimationFrame(_energyAnim); _energyAnim = null; }
    var W = 860, H = 480, mT = 26, mB = 40;
    var axisX = 30, symX = 76, lineL = 152, lineR = W - 56;
    var maxE = data.maxE || (data.levels.reduce(function (m, l) { return Math.max(m, l.energy); }, 0) * 1.08);
    function yOf(e) { return mT + (maxE - e) / maxE * (H - mT - mB); }
    var lvlMap = {};
    data.levels.forEach(function (l) { lvlMap[l.label] = l; });

    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;height:auto;display:block;background:var(--surface);border:1px solid var(--rule);border-radius:10px">';
    // 能量轴刻度 (最左列: 数字)
    var ticks = [0, 0.25, 0.5, 0.75, 1];
    ticks.forEach(function (t) {
      var y = mT + t * (H - mT - mB);
      var e = Math.round(maxE * (1 - t));
      svg += '<line x1="' + lineL + '" y1="' + y + '" x2="' + lineR + '" y2="' + y + '" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 5"/>';
      svg += '<text x="' + axisX + '" y="' + (y + 4) + '" text-anchor="end" font-size="10" fill="var(--muted)">' + e + '</text>';
    });
    svg += '<text x="' + axisX + '" y="' + (mT - 4) + '" text-anchor="middle" font-size="10" fill="var(--muted)">cm⁻¹</text>';
    // 能级 (全宽延长线段)
    data.levels.forEach(function (l) {
      var y = yOf(l.energy);
      svg += '<circle cx="' + (lineL - 7) + '" cy="' + y + '" r="3" fill="' + l.color + '"/>';
      svg += '<line x1="' + lineL + '" y1="' + y + '" x2="' + lineR + '" y2="' + y + '" stroke="' + l.color + '" stroke-width="3" stroke-linecap="round"/>';
      // 左二列: 能级符号 (与数字分列)
      svg += '<text x="' + (symX - 6) + '" y="' + (y + 4) + '" text-anchor="end" font-size="12.5" font-weight="600" fill="var(--ink)">' + esc(l.label) + '</text>';
      // 右端: 能量值
      svg += '<text x="' + (lineR + 8) + '" y="' + (y + 4) + '" font-size="10.5" fill="var(--muted)">' + l.energy + (l.j ? ' · J=' + esc(l.j) : '') + '</text>';
    });
    // 跃迁路径 (箭头位于线段中点指明方向)
    var trArr = [];
    data.transitions.forEach(function (t, i) {
      var f = lvlMap[t.from], to = lvlMap[t.to];
      if (!f || !to) return;
      var step = Math.max(48, (lineR - lineL - 40) / 5);
      var x = lineR - 16 - (i % 5) * step;
      var y1 = yOf(f.energy), y2 = yOf(to.energy);
      var mid = (y1 + y2) / 2;
      var down = y2 > y1;
      var opc = 0.4 + 0.6 * (t.prob || 0.5);
      var sw = 2 + (t.prob || 0.5) * 3;
      svg += '<line x1="' + x + '" y1="' + y1 + '" x2="' + x + '" y2="' + y2 + '" stroke="' + f.color + '" stroke-width="' + sw.toFixed(1) + '" stroke-opacity="' + opc.toFixed(2) + '" stroke-linecap="round"/>';
      // 中点箭头 (按跃迁方向)
      var ah = 6;
      if (down) {
        svg += '<path d="M' + (x - ah) + ' ' + (mid - ah) + ' L' + (x + ah) + ' ' + (mid - ah) + ' L' + x + ' ' + (mid + ah) + ' Z" fill="' + f.color + '" opacity="' + opc.toFixed(2) + '"/>';
      } else {
        svg += '<path d="M' + (x - ah) + ' ' + (mid + ah) + ' L' + (x + ah) + ' ' + (mid + ah) + ' L' + x + ' ' + (mid - ah) + ' Z" fill="' + f.color + '" opacity="' + opc.toFixed(2) + '"/>';
      }
      // 波长标签 (中点旁)
      svg += '<text x="' + (x + 8) + '" y="' + (mid + 3) + '" font-size="10" fill="' + f.color + '">' + esc(t.wl) + 'nm</text>';
      // 发光粒子 (动画)
      svg += '<circle cx="' + x + '" cy="' + y1 + '" r="4.5" fill="#fff" stroke="' + f.color + '" stroke-width="2" opacity="0">';
      svg += '<animate attributeName="cy" from="' + y1 + '" to="' + y2 + '" dur="' + (0.9 - (t.prob || 0.5) * 0.35) + 's" begin="' + (i * 0.7) + 's" repeatCount="indefinite"/>';
      svg += '<animate attributeName="opacity" values="0;1;1;1;0" keyTimes="0;0.05;0.85;0.96;1" dur="' + (0.9 - (t.prob || 0.5) * 0.35) + 's" begin="' + (i * 0.7) + 's" repeatCount="indefinite"/></circle>';
      // 光子脉冲
      svg += '<circle cx="' + x + '" cy="' + y2 + '" r="3" fill="' + f.color + '" opacity="0">';
      svg += '<animate attributeName="r" values="2;9" dur="0.6s" begin="' + (i * 0.7 + 0.9) + 's" repeatCount="indefinite"/>';
      svg += '<animate attributeName="opacity" values="0;0.9;0" dur="0.6s" begin="' + (i * 0.7 + 0.9) + 's" repeatCount="indefinite"/></circle>';
      trArr.push({ from: t.from, to: t.to, wl: t.wl, prob: t.prob, photon: t.photon });
    });
    svg += '</svg>';

    // 统计条带 (模块化拟合一行)
    var top = trArr.reduce(function (a, b) { return (a.prob || 0) > (b.prob || 0) ? a : b; }, trArr[0]);
    var stats =
      '<div class="stat-strip">' +
      '<div class="stat-card"><div class="num">' + data.levels.length + '</div><div class="lbl">能级数</div></div>' +
      '<div class="stat-card"><div class="num">' + trArr.length + '</div><div class="lbl">跃迁数</div></div>' +
      '<div class="stat-card"><div class="num">' + maxE.toLocaleString() + '</div><div class="lbl">最高能级 /cm⁻¹</div></div>' +
      '<div class="stat-card"><div class="num" style="color:var(--accent-ink)">' + (top ? esc(top.from) + '→' + esc(top.to) : '-') + '</div><div class="lbl">主跃迁 · ' + (top ? esc(top.photon) : '-') + '</div></div>' +
      '</div>';

    box.innerHTML = stats +
      '<p style="font-size:12px;color:var(--muted);margin:0 0 6px">能级线已延长贯穿 · 中点箭头指明跃迁方向 · 粒子沿箭头下坠并发射光子脉冲 · 箭头粗细/透明度 ∝ 相对跃迁概率</p>' + svg;
    // 触发 SMIL 动画 (部分浏览器需显式 begin)
    var anis = box.querySelectorAll('animate');
    anis.forEach(function (a) { try { if (a.beginSpecified) a.beginElementAt(0); } catch (e) { } });
  }

  /* ================= 电子云形状 (数据驱动, 真3D旋转 + 粒子/密度双模式) ================= */
  var _cloudObj = null;
  function hexToRgb(hex) {
    var h = String(hex || '#3b82f6').replace('#', '');
    if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
    var n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  // 预计算 3D 密度体 + 粒子池 (数据驱动: 量子数不同 → 形状不同)
  function buildOrbital(orb) {
    var N = 44, half = N / 2;
    var scale = orb.rhoMax / 0.9;
    var grid = new Float32Array(N * N * N);
    var maxD = 0;
    for (var ix = 0; ix < N; ix++) {
      for (var iy = 0; iy < N; iy++) {
        for (var iz = 0; iz < N; iz++) {
          var x = (ix - half) / half, y = (iy - half) / half, z = (iz - half) / half;
          var r = Math.sqrt(x * x + y * y + z * z);
          if (r < 0.03) continue;
          var rho = r * scale;
          var rv = radialR(orb.n, orb.l, rho);
          var th = Math.acos(Math.max(-1, Math.min(1, z / r)));
          var ph = Math.atan2(y, x);
          var ang = angularLm(orb.l, orb.kind || 'm0', th, ph);
          var dd = rv * rv * ang * ang;
          grid[(ix * N + iy) * N + iz] = dd;
          if (dd > maxD) maxD = dd;
        }
      }
    }
    // 粒子池: 按密度加权采样
    var cand = [], cum = [], tot = 0;
    for (var a = 0; a < N; a++) {
      for (var b = 0; b < N; b++) {
        for (var c = 0; c < N; c++) {
          var dv = grid[(a * N + b) * N + c];
          if (dv < maxD * 0.03) continue;
          cand.push([(a - half) / half, (b - half) / half, (c - half) / half, dv]);
          tot += dv; cum.push(tot);
        }
      }
    }
    var pts = [];
    for (var p = 0; p < 2600; p++) {
      var rr = Math.random() * tot, lo = 0, hi = cand.length - 1, ans = 0;
      while (lo <= hi) { var mm = (lo + hi) >> 1; if (cum[mm] >= rr) { ans = mm; hi = mm - 1; } else { lo = mm + 1; } }
      pts.push(cand[ans]);
    }
    return { grid: grid, maxD: maxD, pts: pts, N: N, orb: orb };
  }
  var _orbState = null;
  function startCloud() {
    var box = g('mf8Cloud');
    if (!box) return;
    if (_cloudObj) { cancelAnimationFrame(_cloudObj.raf); _cloudObj = null; }

    var orb = _currentOrbital || ORBITALS[0];
    var SIZE = 300, cx = 150, cy = 150;
    box.innerHTML =
      '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px" id="mf8OrbChips"></div>' +
      '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start">' +
      '<div style="position:relative;flex:none">' +
      '<canvas id="mf8CloudCanvas" width="' + SIZE + '" height="' + SIZE + '" style="border:1px solid var(--rule);border-radius:10px;background:#0b1020;display:block"></canvas>' +
      '<div id="mf8CloudLabel" style="position:absolute;left:8px;top:6px;font-size:12px;font-weight:600;color:#e2e8f0;background:rgba(0,0,0,.35);border-radius:6px;padding:2px 8px"></div>' +
      '</div>' +
      '<div style="flex:1;min-width:220px">' +
      '<div class="card" style="background:var(--surface2);padding:10px 12px">' +
      '<div style="font-size:12.5px;font-weight:600;margin-bottom:6px">轨道说明</div>' +
      '<div id="mf8OrbDesc" style="font-size:12px;color:var(--muted);line-height:1.7"></div>' +
      '</div>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">' +
      '<button class="btn ghost sm" id="mf8CloudSpin">⏸ 暂停旋转</button>' +
      '<button class="btn ghost sm" id="mf8CloudMode">粒子采样</button>' +
      '</div>' +
      '</div></div>';

    var chips = g('mf8OrbChips');
    ORBITALS.forEach(function (o) {
      var b = d.createElement('button');
      b.className = 'q-clr-chip';
      b.style.cssText = 'font-size:12px;padding:4px 12px';
      b.textContent = o.label;
      b.setAttribute('data-orb', o.id);
      b.addEventListener('click', function () {
        _currentOrbital = o;
        chips.querySelectorAll('.mf8-orb-chip').forEach(function (x) { x.style.borderColor = ''; x.style.background = ''; });
        b.style.borderColor = 'var(--accent)'; b.style.background = 'var(--accent-soft)';
        g('mf8CloudLabel').textContent = o.label;
        g('mf8OrbDesc').textContent = orbDesc(o);
        box._orb = o;
        _orbState = buildOrbital(o); // 重建 3D 数据
      });
      b.classList.add('mf8-orb-chip');
      chips.appendChild(b);
    });

    var spinBtn = g('mf8CloudSpin'), modeBtn = g('mf8CloudMode');
    var spin = true, mode = 'density'; // 默认密度云
    if (spinBtn) spinBtn.addEventListener('click', function () { spin = !spin; spinBtn.textContent = spin ? '⏸ 暂停旋转' : '▶ 继续旋转'; });
    if (modeBtn) modeBtn.addEventListener('click', function () {
      mode = mode === 'density' ? 'points' : 'density';
      modeBtn.textContent = mode === 'density' ? '粒子采样' : '密度云';
    });

    var canvas = g('mf8CloudCanvas');
    var ctx = canvas.getContext('2d');
    var img = ctx.createImageData(SIZE, SIZE);
    var acc = new Float32Array(SIZE * SIZE);
    _orbState = buildOrbital(orb);

    function cloudRGBA(v) {
      var a = Math.pow(Math.max(0, Math.min(1, v)), 0.5);
      var r, gg, b2;
      if (v < 0.33) { r = 0; gg = 120 * v / 0.33; b2 = 80 + 175 * v / 0.33; }
      else if (v < 0.66) { var t = (v - 0.33) / 0.33; r = 0; gg = 120 + 135 * t; b2 = 255; }
      else { var t2 = (v - 0.66) / 0.34; r = 255 * t2; gg = 255; b2 = 255 - 55 * t2; }
      return [r, gg, b2, a * 245];
    }

    var phi = 0;
    function frame() {
      if (!canvas || !canvas.isConnected) { _cloudObj = null; return; }
      var st = _orbState || buildOrbital(_currentOrbital || ORBITALS[0]);
      _orbState = st;
      var orb2 = st.orb, N = st.N, half = N / 2;
      var cos = Math.cos(phi), sin = Math.sin(phi);
      g('mf8CloudLabel').textContent = (orb2 ? orb2.label : '') + ' · ' + (mode === 'density' ? '密度云' : '粒子采样');
      if (mode === 'density') {
        // 密度云: 真3D体绕 y 轴旋转后的正交投影
        acc.fill(0);
        var gridD = st.grid, maxD = st.maxD;
        for (var ix = 0; ix < N; ix++) {
          for (var iy = 0; iy < N; iy++) {
            for (var iz = 0; iz < N; iz++) {
              var dv = gridD[(ix * N + iy) * N + iz];
              if (dv < maxD * 0.02) continue;
              var x = (ix - half) / half, y = (iy - half) / half, z = (iz - half) / half;
              var xr = x * cos + z * sin, zr = -x * sin + z * cos;
              var sx = Math.round(cx + xr * cx), sy = Math.round(cy - zr * cy);
              if (sx < 0 || sx >= SIZE || sy < 0 || sy >= SIZE) continue;
              acc[sy * SIZE + sx] += dv;
            }
          }
        }
        var amax = 0;
        for (var ai = 0; ai < acc.length; ai++) { if (acc[ai] > amax) amax = acc[ai]; }
        var inv = amax > 0 ? 1 / amax : 0;
        var data = img.data;
        for (var pix = 0; pix < acc.length; pix++) {
          var rg = cloudRGBA(acc[pix] * inv);
          var ip = pix * 4;
          data[ip] = rg[0]; data[ip + 1] = rg[1]; data[ip + 2] = rg[2]; data[ip + 3] = rg[3];
        }
        ctx.putImageData(img, 0, 0);
      } else {
        // 粒子采样: 旋转并投影粒子点, 深度着色
        ctx.fillStyle = '#0b1020'; ctx.fillRect(0, 0, SIZE, SIZE);
        var rgb = hexToRgb(orb2 ? orb2.color : '#3b82f6');
        var pts = st.pts;
        for (var p = 0; p < pts.length; p++) {
          var X = pts[p][0], Y = pts[p][1], Z = pts[p][2];
          var xr = X * cos + Z * sin, zr = -X * sin + Z * cos;
          var sx = Math.round(cx + xr * cx), sy = Math.round(cy - zr * cy);
          if (sx < 0 || sx >= SIZE || sy < 0 || sy >= SIZE) continue;
          var depth = 0.35 + 0.65 * ((Y + 1) / 2);
          ctx.fillStyle = 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + depth.toFixed(2) + ')';
          ctx.fillRect(sx, sy, 2, 2);
        }
        // 中心引导线
        ctx.strokeStyle = 'rgba(255,255,255,.12)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, SIZE); ctx.moveTo(0, cy); ctx.lineTo(SIZE, cy); ctx.stroke();
      }
      if (spin) phi += 0.02;
      _cloudObj = { raf: requestAnimationFrame(frame) };
    }
    g('mf8CloudLabel').textContent = orb.label;
    g('mf8OrbDesc').textContent = orbDesc(orb);
    box._orb = orb;
    frame();
  }
  function orbDesc(o) {
    var desc = {
      '1s': 's 轨道 (l=0)：球形对称，无节点，电子密度从核向外平滑衰减。',
      '2s': 's 轨道 (n=2)：球形对称，含 1 个径向节点（亮暗环交界），出现波节面。',
      '2pz': 'pz 轨道 (l=1, m=0)：沿 z 轴呈哑铃形，两个对称叶瓣，中心有节面。',
      '2px': 'px 轨道 (l=1)：沿 x 轴哑铃形，旋转动画展示三个正交 p 轨道的取向。',
      '3dz2': 'dz² 轨道 (l=2, m=0)：沿 z 轴双锥 + 赤道环，四叶 + 环的组合形貌。',
      '3dxy': 'dxy 轨道 (l=2)：四个叶瓣位于 xy 平面 45° 方向，旋转显示空间取向。',
      '4fz3': 'fz³ 轨道 (l=3, m=0)：稀土 4f 电子典型形貌，多叶瓣 + 锥面的复杂结构。'
    };
    return desc[o.id] || ('n=' + o.n + ', l=' + o.l + ' 轨道。');
  }
  var _currentOrbital = null;

  /* ================= 独立云图渲染 (供任意容器, 不依赖 #mf8Cloud 固定 ID) ================= */
  function renderCloudTo(container, data) {
    if (!container) return;
    // 从 data.orbitals 取第一个轨道 (或默认 4f), 构建独立 3D 密度云
    var orbitals = (data && data.orbitals) || ORBITALS;
    var orb = orbitals[0] || ORBITALS[0];
    var SIZE = 280, cx = SIZE / 2, cy = SIZE / 2;
    container.innerHTML =
      '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start">' +
      '<div style="position:relative;flex:none">' +
      '<canvas width="' + SIZE + '" height="' + SIZE + '" style="border:1px solid var(--rule);border-radius:10px;background:#0b1020;display:block"></canvas>' +
      '<div style="position:absolute;left:8px;top:6px;font-size:12px;font-weight:600;color:#e2e8f0;background:rgba(0,0,0,.35);border-radius:6px;padding:2px 8px">' + esc(orb.label) + ' · 电子云</div>' +
      '</div>' +
      '<div style="flex:1;min-width:180px">' +
      '<div style="font-size:12px;color:var(--muted);line-height:1.7">' + orbDesc(orb) + '</div>' +
      '<div style="font-size:11px;color:var(--muted);margin-top:8px">动态旋转 · 数据驱动 · 依据量子数 (n=' + orb.n + ', l=' + orb.l + ') 实时生成</div>' +
      '</div></div>';

    var canvas = container.querySelector('canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var img = ctx.createImageData(SIZE, SIZE);
    var acc = new Float32Array(SIZE * SIZE);
    var st = buildOrbital(orb);
    var N = st.N, half = N / 2;
    var phi = 0, raf = null;

    function cloudRGBA(v) {
      var a = Math.pow(Math.max(0, Math.min(1, v)), 0.5);
      var r, gg, b2;
      if (v < 0.33) { r = 0; gg = 120 * v / 0.33; b2 = 80 + 175 * v / 0.33; }
      else if (v < 0.66) { var t = (v - 0.33) / 0.33; r = 0; gg = 120 + 135 * t; b2 = 255; }
      else { var t2 = (v - 0.66) / 0.34; r = 255 * t2; gg = 255; b2 = 255 - 55 * t2; }
      return [r, gg, b2, a * 245];
    }
    function frame() {
      if (!canvas || !canvas.isConnected) return;
      var cos = Math.cos(phi), sin = Math.sin(phi);
      acc.fill(0);
      var gridD = st.grid, maxD = st.maxD;
      for (var ix = 0; ix < N; ix++) {
        for (var iy = 0; iy < N; iy++) {
          for (var iz = 0; iz < N; iz++) {
            var dv = gridD[(ix * N + iy) * N + iz];
            if (dv < maxD * 0.02) continue;
            var x = (ix - half) / half, y = (iy - half) / half, z = (iz - half) / half;
            var xr = x * cos + z * sin, zr = -x * sin + z * cos;
            var sx = Math.round(cx + xr * cx), sy = Math.round(cy - zr * cy);
            if (sx < 0 || sx >= SIZE || sy < 0 || sy >= SIZE) continue;
            acc[sy * SIZE + sx] += dv;
          }
        }
      }
      var amax = 0;
      for (var ai = 0; ai < acc.length; ai++) { if (acc[ai] > amax) amax = acc[ai]; }
      var inv = amax > 0 ? 1 / amax : 0;
      var pdata = img.data;
      for (var pix = 0; pix < acc.length; pix++) {
        var rg = cloudRGBA(acc[pix] * inv);
        var ip = pix * 4;
        pdata[ip] = rg[0]; pdata[ip + 1] = rg[1]; pdata[ip + 2] = rg[2]; pdata[ip + 3] = rg[3];
      }
      ctx.putImageData(img, 0, 0);
      phi += 0.02;
      raf = requestAnimationFrame(frame);
    }
    frame();
    container._cloudCleanup = function () { if (raf) cancelAnimationFrame(raf); };
  }

  /* ================= 跃迁概率 (数据驱动, 动态级联) ================= */
  var _transAnim = null;
  function renderTrans(data, box) {
    if (!box) box = g('mf8Trans');
    if (!box) return;
    if (_transAnim) { cancelAnimationFrame(_transAnim); _transAnim = null; }
    var lvlMap = {};
    data.levels.forEach(function (l) { lvlMap[l.label] = l; });
    // 找出作为“源激发态”的能级 (发射概率总和最大)
    var fromAgg = {};
    data.transitions.forEach(function (t) { fromAgg[t.from] = (fromAgg[t.from] || 0) + (t.prob || 0); });
    var src = null, srcProb = -1;
    Object.keys(fromAgg).forEach(function (f) { if (fromAgg[f] > srcProb) { srcProb = fromAgg[f]; src = f; } });
    if (!src) { box.innerHTML = '<p style="color:var(--muted)">无跃迁数据</p>'; return; }
    var branches = data.transitions.filter(function (t) { return t.from === src; });
    var srcLvl = lvlMap[src];
    var maxP = branches.reduce(function (m, t) { return Math.max(m, t.prob); }, 0);

    var W = 780, H = 320, cX = W / 2, topY = 54, botY = 250;
    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;height:auto;display:block;background:var(--surface);border:1px solid var(--rule);border-radius:10px">';
    // 源节点
    var srcCol = srcLvl ? srcLvl.color : '#3b82f6';
    svg += '<circle cx="' + cX + '" cy="' + topY + '" r="30" fill="' + srcCol + '" opacity=".95"/>';
    svg += '<text x="' + cX + '" y="' + (topY - 2) + '" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">' + esc(src) + '</text>';
    svg += '<text x="' + cX + '" y="' + (topY + 16) + '" text-anchor="middle" font-size="9.5" fill="rgba(255,255,255,.85)">激发态</text>';
    // 分支
    var n = branches.length;
    branches.forEach(function (t, i) {
      var tx = cX - (n - 1) * 60 + i * 120;
      var to = lvlMap[t.to];
      var col = to ? to.color : '#94a3b8';
      var rel = maxP > 0 ? (t.prob / maxP) : 0;
      var sw = 3 + rel * 10;
      var opc = 0.45 + 0.55 * rel;
      // Bezier 分支路径 (终点落在节点上沿, 不遮挡底部文字)
      var c1x = cX, c1y = topY + 90, c2x = tx, c2y = botY - 80;
      svg += '<path d="M' + cX + ' ' + topY + ' C' + c1x + ' ' + c1y + ' ' + c2x + ' ' + c2y + ' ' + tx + ' ' + (botY - 18) + '" fill="none" stroke="' + col + '" stroke-width="' + sw.toFixed(1) + '" stroke-opacity="' + opc.toFixed(2) + '" stroke-linecap="round"/>';
      // 流动粒子 (沿路径)
      svg += '<circle r="4" fill="#fff" stroke="' + col + '" stroke-width="2" opacity="0">';
      svg += '<animateMotion dur="' + (1.1 - i * 0.12) + 's" begin="' + (i * 0.5) + 's" repeatCount="indefinite"><mpath href="#mf8path' + i + '"/></animateMotion>';
      svg += '<animate attributeName="opacity" values="0;1;0" keyTimes="0;0.5;1" dur="' + (1.1 - i * 0.12) + 's" begin="' + (i * 0.5) + 's" repeatCount="indefinite"/></circle>';
      svg += '<path id="mf8path' + i + '" d="M' + cX + ' ' + topY + ' C' + c1x + ' ' + c1y + ' ' + c2x + ' ' + c2y + ' ' + tx + ' ' + (botY - 18) + '" fill="none" stroke="none"/>';
      // 目标节点
      svg += '<circle cx="' + tx + '" cy="' + botY + '" r="22" fill="' + col + '" opacity=".95"/>';
      svg += '<text x="' + tx + '" y="' + (botY + 4) + '" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">' + esc(t.to) + '</text>';
      // 节点下方: 光子/波长 + 占比 (远离曲线, 不遮挡)
      svg += '<rect x="' + (tx - 46) + '" y="' + (botY + 26) + '" width="92" height="32" rx="6" fill="var(--surface2)" stroke="var(--rule)"/>';
      svg += '<text x="' + tx + '" y="' + (botY + 42) + '" text-anchor="middle" font-size="10" fill="var(--ink)">' + esc(t.photon) + ' · ' + t.wl + 'nm</text>';
      svg += '<text x="' + tx + '" y="' + (botY + 54) + '" text-anchor="middle" font-size="10.5" font-weight="700" fill="' + col + '">' + Math.round(rel * 100) + '%</text>';
    });
    svg += '</svg>';

    // 概率条 + 统计条带
    var bars = branches.map(function (t) {
      var rel = maxP > 0 ? (t.prob / maxP) : 0;
      var pct = Math.round(rel * 100);
      var to = lvlMap[t.to];
      return '<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;font-size:12px"><span>' + esc(src) + ' → ' + esc(t.to) + ' <span style="color:var(--muted)">(' + esc(t.photon) + ' · ' + t.wl + 'nm)</span></span><b style="color:var(--accent-ink)">' + pct + '%</b></div>' +
        '<div style="height:10px;border-radius:5px;background:var(--surface2);overflow:hidden;margin-top:3px"><div style="height:100%;width:' + pct + '%;border-radius:5px;background:' + (to ? to.color : '#94a3b8') + ';transition:width .6s ease"></div></div></div>';
    }).join('');
    var stats =
      '<div class="stat-strip">' +
      '<div class="stat-card"><div class="num">' + branches.length + '</div><div class="lbl">衰变通道</div></div>' +
      '<div class="stat-card"><div class="num" style="color:var(--accent-ink)">' + esc(src) + '</div><div class="lbl">源激发态</div></div>' +
      '<div class="stat-card"><div class="num">' + (maxP > 0 ? Math.round((branches[0] ? branches[0].prob / maxP : 1) * 100) : 0) + '%</div><div class="lbl">主分支占比</div></div>' +
      '<div class="stat-card"><div class="num">' + (branches[0] ? branches[0].wl : '-') + 'nm</div><div class="lbl">主导波长</div></div>' +
      '</div>';
    box.innerHTML = stats +
      '<p style="font-size:12px;color:var(--muted);margin:0 0 6px">粒子从激发态沿各分支级联下泄 · 分支宽度/占比 ∝ 相对跃迁概率 · 标签置于节点下方避免遮挡</p>' + svg +
      '<div style="margin-top:12px"><div style="font-size:12.5px;font-weight:600;margin-bottom:8px">分支占比 (相对主导分支)</div>' + bars + '</div>';
    var anis = box.querySelectorAll('animate');
    anis.forEach(function (a) { try { if (a.beginSpecified) a.beginElementAt(0); } catch (e) { } });
  }

  /* ================= 自定义数据编辑器 ================= */
  function openCustomEditor() {
    var sample = JSON.stringify(PRESETS.dy, null, 2);
    var ov = d.createElement('div');
    ov.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:9990;display:flex;align-items:center;justify-content:center;padding:20px';
    ov.id = 'mf8CustomOv';
    ov.innerHTML =
      '<div style="background:var(--card);border:1px solid var(--rule);border-radius:14px;max-width:640px;width:100%;max-height:86vh;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.3)">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--rule)"><b>自定义能级 / 跃迁数据</b><button id="mf8CustClose" style="font-size:18px;color:var(--muted)">×</button></div>' +
      '<div style="padding:14px 18px;font-size:12px;color:var(--muted);line-height:1.7">粘贴 JSON 以生成不同的能级图与跃迁概率可视化。字段：<code>levels</code>（label/energy/color）, <code>transitions</code>（from/to/prob/wl/photon）。点击“应用”后自动重绘。</div>' +
      '<div style="flex:1;padding:0 18px;overflow:auto"><textarea id="mf8CustTxt" spellcheck="false" style="width:100%;height:320px;font-family:var(--mono);font-size:12px;background:var(--surface2);color:var(--ink);border:1px solid var(--rule);border-radius:8px;padding:10px;resize:vertical">' + esc(sample) + '</textarea></div>' +
      '<div style="display:flex;gap:8px;padding:14px 18px;border-top:1px solid var(--rule)"><button class="btn primary" id="mf8CustApply" style="flex:1">应用</button><button class="btn ghost" id="mf8CustCancel" style="flex:1">取消</button></div></div>';
    d.body.appendChild(ov);
    var close = function () { if (ov.parentNode) ov.parentNode.removeChild(ov); };
    g('mf8CustClose').addEventListener('click', close);
    g('mf8CustCancel').addEventListener('click', close);
    ov.addEventListener('click', function (e) { if (e.target === ov) close(); });
    g('mf8CustApply').addEventListener('click', function () {
      try {
        var obj = JSON.parse(g('mf8CustTxt').value);
        if (!obj.levels || !Array.isArray(obj.levels) || !obj.levels.length) throw new Error('缺少 levels 数组');
        if (!Array.isArray(obj.transitions)) obj.transitions = [];
        _customData = obj;
        _currentIon = 'custom';
        close();
        renderEnergy(myData());
        renderTrans(myData());
        startCloud();
        toast('已应用自定义数据');
      } catch (e) { toast('JSON 解析失败: ' + e.message); }
    });
  }

  /* ================= 视图接管 =================
   * R08B-1: scientific visualization remains routable, but is opened from
   * Knowledge & Evidence > Advanced instead of becoming a top-level entry.
   */
  d.addEventListener('view-rendered', function (e) {
    if (e.detail && e.detail.view === VIEW) { render(); }
  });

  // 清理动画 (视图离开时)
  d.addEventListener('view-rendered', function (e) {
    if (e.detail && e.detail.view !== VIEW) {
      if (_energyAnim) { cancelAnimationFrame(_energyAnim); _energyAnim = null; }
      if (_cloudObj) { cancelAnimationFrame(_cloudObj.raf); _cloudObj = null; }
    }
  });

  /* ================= 全局导出 (供小助手/问答等任意容器动态渲染) =================
   * 开放能力: 任何容器传入数据即可实时生成图形, 而非预置静态图。
   */
  window.MF8Viz = window.MF8Viz || {};
  window.MF8Viz.renderEnergyTo = function (container, data) { renderEnergy(data, container); };
  window.MF8Viz.renderTransTo = function (container, data) { renderTrans(data, container); };
  // 依据数据/类型把图渲染到任意容器 (energy 能级跃迁 | trans 跃迁概率 | cloud 电子云)
  window.MF8Viz.renderFromData = function (container, data, tab) {
    if (!container) return;
    if (!data) { container.innerHTML = '<p style="color:var(--muted);font-size:12px">无可视化数据</p>'; return; }
    var vt = tab || data.viz_type || 'energy';
    if (vt === 'trans') { renderTrans(data, container); return; }
    if (vt === 'cloud') { renderCloudTo(container, data); return; }
    renderEnergy(data, container);
  };
  window.MF8Viz.getPreset = function (ion) {
    var key = String(ion || 'dy').toLowerCase();
    var preset = PRESETS[key] || PRESETS.dy;
    return JSON.parse(JSON.stringify(preset));
  };
  // 供视图页外部注入待渲染数据 (小助手/问答识别到可视化意图时调用)
  window.MF8Viz.inject = function (data, tab) {
    window.MF8Viz._pending = { data: data, tab: tab || 'energy' };
  };
})();
