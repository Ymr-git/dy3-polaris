/* Dy3+ Polaris 动态界面增强 — ui-polish
   纯增量、零侵入: 只监听现有事件、追加视觉增强, 不改变任何既有渲染逻辑。
   特性:
   1. 顶部加载进度条 (路由切换 / fetch 请求时自动显示)
   2. 视图切换过渡动画 (监听 view-rendered 事件)
   3. 数字滚动动画 (对 .stat-card .num / .metric-val / .dash-num 等)
   4. 骨架屏 (监听 loading 节点, 用流光骨架替代纯 spinner 背景)
   5. 卡片入场 (IntersectionObserver 逐个上浮, 触发 .dp-reveal)
   6. 全局搜索 (Ctrl/Cmd+K 呼出, 按视图名/模块名模糊过滤)
   7. 返回顶部悬浮按钮
   8. 主题切换平滑过渡
*/
(function () {
  'use strict';
  if (window.__dpPolishInit) return;
  window.__dpPolishInit = true;

  var d = document;
  function g(id) { return d.getElementById(id); }
  function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  /* ============ 1. 顶部进度条 ============ */
  var progress = d.createElement('div');
  progress.className = 'dp-progress';
  d.body.appendChild(progress);
  var progressTimer = null;
  function progressStart() {
    progress.classList.remove('done');
    progress.style.width = '8%';
    clearTimeout(progressTimer);
    // 模拟渐进: 逐渐逼近 90%, 直到完成
    var w = 8;
    progressTimer = setInterval(function () {
      if (w < 90) { w += (90 - w) * 0.12 + 0.5; progress.style.width = w + '%'; }
    }, 180);
  }
  function progressDone() {
    clearInterval(progressTimer);
    progress.style.width = '100%';
    progress.classList.add('done');
    setTimeout(function () { progress.style.width = '0'; progress.classList.remove('done'); }, 380);
  }
  // 拦截 fetch: 请求开始/结束自动显示进度
  var _fetch = window.fetch;
  if (_fetch) {
    window.fetch = function () {
      progressStart();
      var p = _fetch.apply(this, arguments);
      Promise.resolve(p).then(function () { progressDone(); }, function () { progressDone(); });
      return p;
    };
  }

  /* ============ 2. 视图切换过渡 ============ */
  var content = g('content');
  function animateContent() {
    if (!content) return;
    content.classList.remove('dp-enter');
    void content.offsetWidth; // 重排以重新触发动画
    content.classList.add('dp-enter');
  }
  d.addEventListener('view-rendered', function () {
    progressDone();
    animateContent();
    setTimeout(polishNumbers, 50);
    setTimeout(setupReveals, 60);
  });

  /* ============ 3. 数字滚动 ============ */
  function polishNumbers() {
    if (!content) return;
    var nums = content.querySelectorAll('.stat-card .num, .metric-val, .dash-num, .domain-avg');
    nums.forEach(function (el) {
      if (el.__dpCounted) return;
      el.__dpCounted = true;
      var txt = (el.textContent || '').trim();
      // 仅处理纯数字或百分比
      var m = txt.match(/^([+-]?\d+(?:\.\d+)?)\s*(%?)$/);
      if (!m) return;
      var target = parseFloat(m[1]);
      var suffix = m[2];
      if (!isFinite(target) || target === 0) { el.classList.add('dp-num'); return; }
      var dur = 700;
      var t0 = null;
      el.classList.add('dp-num');
      function step(ts) {
        if (!t0) t0 = ts;
        var k = Math.min(1, (ts - t0) / dur);
        var eased = 1 - Math.pow(1 - k, 3);
        var val = target * eased;
        el.textContent = (Number.isInteger(target) ? Math.round(val) : val.toFixed(2)) + suffix;
        if (k < 1) requestAnimationFrame(step);
        else el.textContent = txt;
      }
      requestAnimationFrame(step);
    });
  }

  /* ============ 4. 骨架屏 ============ */
  function setupSkeletons() {
    if (!content) return;
    var loaders = content.querySelectorAll('.loading');
    loaders.forEach(function (ld) {
      if (ld.__dpSkel) return;
      ld.__dpSkel = true;
      // 保留 spinner, 追加两行骨架条增强视觉反馈
      var box = d.createElement('div');
      box.style.cssText = 'width:100%;max-width:420px;display:flex;flex-direction:column;gap:8px';
      box.innerHTML = '<div class="dp-skeleton" style="height:14px;width:60%"></div>' +
        '<div class="dp-skeleton" style="height:14px;width:100%"></div>' +
        '<div class="dp-skeleton" style="height:14px;width:80%"></div>';
      ld.appendChild(box);
    });
  }

  /* ============ 5. 卡片入场 (IntersectionObserver) ============ */
  function setupReveals() {
    if (!content) return;
    var cards = content.querySelectorAll('.card, .panel-card, .stat-card, .domain-card, .metric-card, .chain-item, .flow-step');
    cards.forEach(function (el, i) {
      if (el.__dpReveal) return;
      el.__dpReveal = true;
      el.classList.add('dp-reveal');
      el.style.transitionDelay = Math.min(i * 30, 300) + 'ms';
    });
    if (!window.__dpIO) {
      window.__dpIO = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { e.target.classList.add('dp-in'); window.__dpIO.unobserve(e.target); }
        });
      }, { threshold: 0.08 });
    }
    cards.forEach(function (el) { window.__dpIO.observe(el); });
    // 兜底: 已在视口内的立即显示
    setTimeout(function () {
      cards.forEach(function (el) {
        var r = el.getBoundingClientRect();
        if (r.top < window.innerHeight && r.bottom > 0) el.classList.add('dp-in');
      });
    }, 120);
  }

  /* ============ 6. 返回顶部按钮 ============ */
  var topBtn = d.createElement('button');
  topBtn.className = 'dp-top';
  topBtn.setAttribute('aria-label', '返回顶部');
  topBtn.innerHTML = '↑';
  d.body.appendChild(topBtn);
  topBtn.addEventListener('click', function () {
    var scroller = content || d.scrollingElement;
    scroller.scrollTo({ top: 0, behavior: 'smooth' });
  });
  function checkTop() {
    var scroller = content || d.scrollingElement;
    if ((scroller.scrollTop || 0) > 400) topBtn.classList.add('show');
    else topBtn.classList.remove('show');
  }
  (content || window).addEventListener('scroll', checkTop, { passive: true });

  /* ============ 7. 全局搜索 (Ctrl/Cmd+K) ============ */
  var overlay = null;
  function buildCommandList() {
    // 从侧栏收集所有可用视图
    var views = [];
    d.querySelectorAll('.sidebar-child').forEach(function (b) {
      var icon = (b.querySelector('.child-icon') || {}).textContent || '';
      views.push({ icon: icon.trim(), label: b.textContent.replace(icon, '').trim(), view: b.dataset.view });
    });
    // 去重
    var seen = {};
    return views.filter(function (v) { if (!v.view || seen[v.view]) return false; seen[v.view] = 1; return true; });
  }
  function openCommand() {
    if (overlay) closeCommand();
    var views = buildCommandList();
    overlay = d.createElement('div');
    overlay.className = 'dp-cmd-overlay';
    var listHtml = views.map(function (v) {
      return '<button class="dp-cmd-item" data-view="' + esc(v.view) + '"><span class="ci-icon">' + esc(v.icon || '▸') + '</span><span class="ci-label">' + esc(v.label) + '</span><span class="ci-hint">跳转</span></button>';
    }).join('');
    overlay.innerHTML = '<div class="dp-cmd"><input class="dp-cmd-input" placeholder="搜索功能… (输入关键字过滤, ↑↓ 选择, Enter 跳转, Esc 关闭)" autocomplete="off"><div class="dp-cmd-list">' + (listHtml || '<div class="dp-cmd-empty">未找到可用视图</div>') + '</div></div>';
    d.body.appendChild(overlay);
    var input = overlay.querySelector('.dp-cmd-input');
    var list = overlay.querySelector('.dp-cmd-list');
    var items = function () { return Array.prototype.slice.call(list.querySelectorAll('.dp-cmd-item')); };
    var active = -1;
    function renderFilter() {
      var q = input.value.trim().toLowerCase();
      var all = buildCommandList();
      list.innerHTML = all.filter(function (v) { return !q || v.label.toLowerCase().indexOf(q) !== -1 || (v.view || '').toLowerCase().indexOf(q) !== -1; })
        .map(function (v) {
          return '<button class="dp-cmd-item" data-view="' + esc(v.view) + '"><span class="ci-icon">' + esc(v.icon || '▸') + '</span><span class="ci-label">' + esc(v.label) + '</span><span class="ci-hint">跳转</span></button>';
        }).join('') || '<div class="dp-cmd-empty">无匹配功能</div>';
      active = -1;
    }
    function setActive(i) {
      var it = items();
      it.forEach(function (b) { b.classList.remove('active'); });
      if (i >= 0 && it[i]) { it[i].classList.add('active'); active = i; }
    }
    input.addEventListener('input', renderFilter);
    input.addEventListener('keydown', function (e) {
      var it = items();
      if (e.key === 'ArrowDown') { e.preventDefault(); setActive(Math.min(it.length - 1, active + 1)); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(Math.max(0, active - 1)); }
      else if (e.key === 'Enter') { e.preventDefault(); if (active >= 0 && it[active]) it[active].click(); else if (it[0]) it[0].click(); }
      else if (e.key === 'Escape') { closeCommand(); }
    });
    list.addEventListener('click', function (e) {
      var btn = e.target.closest('.dp-cmd-item');
      if (!btn) return;
      var view = btn.dataset.view;
      closeCommand();
      if (view && window.sv) window.sv(view);
    });
    overlay.addEventListener('click', function (e) { if (e.target === overlay) closeCommand(); });
    input.focus();
  }
  function closeCommand() {
    if (overlay) { overlay.remove(); overlay = null; }
  }
  d.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) { e.preventDefault(); openCommand(); }
    else if (e.key === 'Escape' && overlay) { closeCommand(); }
  });

  /* ============ 8. 主题切换平滑过渡 ============ */
  d.addEventListener('click', function (e) {
    var t = e.target.closest('[id="sbpTheme"], #themeBtn');
    if (!t) return;
    d.body.classList.add('dp-theme-anim');
    setTimeout(function () { d.body.classList.remove('dp-theme-anim'); }, 300);
  });

  /* ============ 初始化: 观察侧栏重建, 重跑动画 ============ */
  d.addEventListener('sidebar-rebuilt', function () {
    setTimeout(polishNumbers, 50);
    setTimeout(setupReveals, 60);
  });
  // 页面初始加载
  if (d.readyState === 'complete') { setupReveals(); setupSkeletons(); }
  else { window.addEventListener('load', function () { setupReveals(); setupSkeletons(); }); }
})();
