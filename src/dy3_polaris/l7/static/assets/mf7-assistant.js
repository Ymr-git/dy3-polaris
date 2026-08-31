/* Dy3+ Polaris 智能小助手 v2 (mf7)
 * 参照国内外成熟方案优化 (Rasa 多轮状态追踪 / Coze·Dify 意图配置与兜底 / Eliza 任务拆解):
 * 1. 意图识别: 同义词扩展 + 否定检测 + 意图优先级 + 兜底澄清(未理解引导)
 * 2. 多轮上下文 DST: 维护 {topic,lastIntent,pendingSlot,history}, 支持指代消解("它/这个")与追问澄清
 * 3. 多意图拆解: "练习一下 然后 看看薄弱点" → 依次执行
 * 4. 回答增强: 知识引用标注 + 建议下一步 (系统状态可见性, 尼尔森原则)
 * 5. 通过 /api/query 调度 4 个 Agent 协作 (诊断→生成→审核→决策)
 */
(function () {
  'use strict';
  var d = document;

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }

  // 化学式上下角标格式化: Dy3+ → Dy<sup>3+</sup>, 4F9/2 → <sup>4</sup>F<sub>9/2</sub>, 4f9 → 4f<sup>9</sup>
  function formatChemical(text) {
    var s = String(text);
    // 1. Electron config: 4f9, 5s2, 5p6, 4f14, 5d1 (先做, 避免被后续规则误匹配)
    s = s.replace(/(\d+)([spdfgh])(\d*)/gi, function (m, p1, p2, p3) {
      return p3 ? p1 + p2 + '<sup>' + p3 + '</sup>' : m;
    });
    // 2. Energy level notation: 4F9/2, 6H15/2, 6H13/2, 6H11/2
    s = s.replace(/(\d)([A-Z])(\d+)\/(\d+)/g, '<sup>$1</sup>$2<sub>$3/$4</sub>');
    // 3. Ions: Dy3+, Eu3+, Ce3+, Yb3+, Er3+, Nd3+ (离子符号, 非数字开头)
    s = s.replace(/(?<!\d)([A-Z][a-z]?)(\d*[+-]+)/g, '$1<sup>$2</sup>');
    // 4. 角标 CSS 样式增强 (确保渲染可靠)
    return s;
  }

  function getToken() { return localStorage.getItem('dt') || ''; }
  function learnerId() { return window.dy3LearnerId ? window.dy3LearnerId() : (localStorage.getItem('dl') || 'guest-unavailable'); }

  function apiReq(method, path, body) {
    var opts = { method: method, headers: { 'Content-Type': 'application/json' } };
    var tk = getToken();
    if (tk) opts.headers['Authorization'] = 'Bearer ' + tk;
    if (body) opts.body = JSON.stringify(body);
    return fetch(path, opts).then(function (r) { return r.json().catch(function () { return {}; }); }).then(function (d) {
      if (d && d.code !== undefined && d.code !== 0 && d.error) throw new Error((d.error && d.error.message) || '请求失败');
      return d && d.data !== undefined ? d.data : d;
    });
  }

  var ASSIST_STATE = {
    open: false, busy: false,
    history: [],          // 最近轮次 {role, text}
    topic: '',            // 当前主题 (指代消解)
    lastIntent: '',       // 上一意图
    pendingSlot: null,    // 追问中的槽位 {intent, ask}
    turns: 0,
  };

  var CHIPS = [
    { label: '🎯 帮我出题练习', msg: '给我出5道题，变着题型来' },
    { label: '📋 今日推荐', msg: '今天学什么？给我学习轨迹' },
    { label: '❓ 答疑', msg: 'Dy3+的浓度猝灭机理是什么？' },
    { label: '📊 薄弱点分析', msg: '我的薄弱知识点有哪些？' },
    { label: '💡 知识查询', msg: '帮我查一下 量子效率 的知识' },
  ];

  // ---- 意图识别 (同义词扩展 + 否定 + 优先级, 参照 Coze/Dify 意图配置思路) ----
  var INTENT_PATTERNS = {
    // 可视化意图优先级最高 (score 7), 避免被 domainQuery/query 抢走, 让图优先渲染
    viz: { score: 7, re: /(画|绘制|绘图|画出|画一|画个|作图|图示|可视化|能级图|跃迁图|电子云|轨道图|光谱图|能级跃迁图|画图)/ },
    reset: { score: 6, re: /(重置|清空|重新开始|新对话|忘记之前)/ },
    practice: { score: 6, re: /(练习|出题|做题|测试|考我|刷题|quiz|exercise|practice|来几道|出几道|出\s*\d+\s*道)/ },
    recommend: { score: 5, re: /(推荐|今天学|学什么|轨迹|下一步|学习计划|学点|建议学)/ },
    path: { score: 6, re: /(学习轨迹|学习路径|我的路径|轨迹图)/ },
    history: { score: 5, re: /(历史记录|会话历史|我的历史|之前的问题|查看记录)/ },
    weak: { score: 5, re: /(薄弱|弱点|不会的|哪里不会|短板|没掌握)/ },
    monitor: { score: 5, re: /(监控|健康|系统状态|agent ?状态|看看系统|运行记录|轨迹记录|agent执行)/ },
    profile: { score: 5, re: /(画像|掌握度|学习数据|学情|掌握情况|我的水平|能力值)/ },
    memory: { score: 6, re: /(记忆|复习|遗忘|fsrs|召回|复习队列|复习提醒|该复习)/ },
    ability: { score: 5, re: /(能力|theta|θ|水平|差距估计|难度估计)/ },
    settings: { score: 4, re: /(设置|主题|语言|深色|浅色|高对比)/ },
    knowledge: { score: 5, re: /(查一下|查询|搜索|检索|帮我查|知识库|什么是|介绍一下|讲讲.*知识)/ },
    query: { score: 3, re: /(是什么|为什么|怎么|如何|多少|解释|答疑|请问|讲讲|说一下|区别|作用|原理|影响|发光|机理|猝灭|能级|跃迁|光谱|效率|制备|合成|掺杂|浓度|温度|热|色度|显色|衰减|寿命|能量传递|交叉弛豫|敏化|激活剂|基质|晶格|相变|波长|发射|激发|吸收|发射谱|激发谱|色温|显色指数|CRI|量子效率|内量子|外量子|热稳定性|热猝灭|光衰|寿命|余辉|陷阱|空位|缺陷|电荷补偿|价态|氧化还原|的发光|的荧光|的磷光|的光致发光|的电致发光|的光转换|的光谱|带隙|声子|晶场|劈裂|Judd|Ofelt|跃迁几率的|振子强度|辐射|无辐射|弛豫)/ },
    chitchat: { score: 2, re: /(你好|hi|hello|在吗|谢谢|辛苦|再见|谢谢啦)/ },
    help: { score: 1, re: /(帮助|功能|你能做什么|怎么用|你是谁|介绍下你自己|有哪些功能)/ },
    understand: { score: 4, re: /(你了解我吗|了解我|我的画像|你觉得我|你知道我)/ },
    guidance: { score: 4, re: /(不知道学什么|不知道该学|没想法|你帮我决定|你看着办|随便|迷茫|没方向|没目标|不知道怎么办|给我点建议|推荐一下我该|我该学什么|学点什么好|不知道怎么开始|从头开始)/ },
    domainQuery: { score: 5, re: /(Dy3?\+|Eu3?\+|Ce3?\+|Tb3?\+|Yb3?\+|Er3?\+|Nd3?\+|Sm3?\+|Pr3?\+|Ho3?\+|Tm3?\+|镝|铕|铈|铽|镱|铒|钕|钐|镨|钬|铥|稀土|荧光粉|磷光体|发光材料|上转换|下转换|长余辉|LED|荧光|4f|5d|基质|敏化|激活剂|量子裁剪|斯托克斯|反斯托克斯|YAG|石榴石|铝酸盐|硅酸盐|磷酸盐|钒酸盐|钨酸盐|钼酸盐|硼酸盐|氮化物|氮氧化物|硫化物|氟化物|氯化物|溴化物|碘化物|纳米晶|量子点|核壳|包覆|发光机理|猝灭机理|浓度猝灭|热猝灭|能级跃迁|发射光谱|激发光谱|量子效率|能量传递|交叉弛豫|热稳定性|光衰|寿命|衰减|发光效率|制备方法|合成工艺|掺杂浓度|晶格|声子|晶场|劈裂|Judd|Ofelt|带隙|陷阱|空位|缺陷|电荷补偿|价态|氧化还原|辐射跃迁|无辐射|弛豫|显色指数|CRI|色温|余辉|上转换发光|下转换发光)/ },
  };
  // 领域外话题: 直接引导 (不触发答疑)
  var OFFT_OPIC_RE = /(天气|天气预报|几点|股市|股票|新闻|笑话|电影|音乐|游戏|打游戏|外卖|吃什么|明星|八卦|体育|足球|篮球|动漫|减肥|健身|养生|购物|淘宝|拼多多|京东|抖音|快手|B站|微博|微信|QQ|今日头条|知乎|百度|小红书|支付宝|微信支付|打车|导航|地图|酒店|机票|火车票)/;
  // 否定词: 命中则降低对应意图置信度 (如 "不用练习")
  var NEGATIVE_RE = /(不用|不需要|别|不要|无需|不是要)/;
  // 指代词: 引用上次主题
  var REFER_RE = /^(它|这个|那个|这些|那|其)/
  // 多意图连接词
  var MULTI_RE = /(?:和|并且|还有|然后|接着)/;
  // 化学式/稀土离子快速检测: 极短查询 (如 "dy", "Dy", "Eu") 立即识别为 domainQuery
  var CHEM_SHORT_RE = /^(Dy|Eu|Ce|Tb|Yb|Er|Nd|Sm|Pr|Ho|Tm|Gd|Lu|La|Sc|Y)(\d*[+-]?)?$/i;
  // 短查询检测: 2-60 个字符且不匹配任何已知模式 → 可能是知识查询
  function looksLikeQuery(text) {
    var t = text.trim();
    if (t.length < 2 || t.length > 60) return false;
    // 纯数字/符号 → 不是查询
    if (/^[\d+\-*/=.,;:!?\s]+$/.test(t)) return false;
    // 常见业务词 → 不是查询
    if (/(练习|出题|帮助|设置|重置|你好|谢谢|再见|在吗)/.test(t)) return false;
    // 含中文字符且不是纯问候 → 可能是知识查询
    if (/[\u4e00-\u9fff]/.test(t) || /[A-Z][a-z]?\d?\+?/.test(t) || /[A-Z]{2,}/.test(t)) return true;
    return false;
  }

  function routeIntent(text) {
    var t = text.trim();
    var neg = NEGATIVE_RE.test(t);
    // 化学式/稀土离子极短查询 (如 "dy", "Dy3+") 直接识别为 domainQuery
    if (CHEM_SHORT_RE.test(t)) return { intent: 'domainQuery' };
    // 领域外话题 → 兜底引导
    if (OFFT_OPIC_RE.test(t)) return { intent: 'fallback' };
    // 多意图拆解: 短句单意图, 长句含连接词 → 拆分为多段依次执行
    if (t.length > 8 && MULTI_RE.test(t)) {
      var segs = t.split(MULTI_RE).map(function (s) { return s.trim(); }).filter(function (s) { return s.length >= 2; });
      if (segs.length >= 2) return { intent: 'multi', segments: segs };
    }
    var best = { intent: 'help', score: 0 };
    for (var name in INTENT_PATTERNS) {
      var p = INTENT_PATTERNS[name];
      if (p.re.test(t)) {
        var score = p.score;
        if (name === 'query' && /^(为什么|怎么|如何|多少)/.test(t)) score += 1; // 实质提问权重
        if (neg && (name === 'practice' || name === 'query' || name === 'knowledge' || name === 'domainQuery')) score = 0; // 否定不触发动作
        if (score > best.score) best = { intent: name, score: score };
      }
    }
    // 兜底: 未匹配任何模式, 但文本看起来像知识查询 → 走 query 意图
    if (best.score === 0 && looksLikeQuery(t)) {
      return { intent: 'query' };
    }
    // 兜底: 未匹配任何模式, 但有上下文主题 → 走 query 意图
    if (best.score === 0 && ASSIST_STATE.topic) {
      return { intent: 'query' };
    }
    return best;
  }

  var INTENT_REPLY = {
    viz: '好的，正在为你生成动态可视化…',
    practice: '好的，已为你打开【学习练习】。',
    recommend: '好的，已为你打开【今日推荐】…',
    path: '正在打开【学习轨迹】…',
    history: '正在打开【历史记录】…',
    weak: '正在打开【薄弱点分析】…',
    monitor: '正在打开【Agent 运行监控】…',
    profile: '正在打开【学情总览】…',
    memory: '正在读取你的记忆状态…',
    ability: '正在读取你的能力估计…',
    settings: '正在打开【设置】…',
    knowledge: '正在检索知识库…',
    query: '好的，正在为你解答…',
    domainQuery: '好的，请在下方面板输入具体问题。',
    reset: '好的，已清空对话记忆，开始新的对话。',
    help: '🎯 动态出题练习\n📋 今日推荐 / 学习轨迹\n❓ 答疑\n📊 薄弱点分析\n💡 知识库查询\n📈 学情画像 / 系统监控\n输入"重置"可开始新对话。',
    chitchat: '你好，我是 Dy3+ Polaris 智能小助手 🤖 输入"帮助"查看功能。',
    fallback: '🎯 出题练习\n❓ 答疑\n📊 薄弱点分析\n💡 知识查询\n输入"帮助"查看全部功能。',
    understand: '让我看看我对你的了解…（你可以纠正我）',
    guidance: '好的，我来结合你的学情画像为你建议学习方向…',
  };

  function viewOf(intent) {
    return { practice: 'practice', recommend: 'overview-recs', weak: 'learn-weak', monitor: 'overview-monitor', profile: 'learn', settings: 'settings', path: 'learn-path', history: 'query-history' }[intent] || null;
  }

  // ---- UI ----
  function buildUI() {
    if (d.getElementById('mf7Fab')) return;
    var css = d.createElement('style');
    css.id = 'mf7AssistantCss';
    css.textContent = [
      '#mf7Fab{position:fixed;right:22px;bottom:22px;width:56px;height:56px;border-radius:50%;border:none;cursor:pointer;z-index:9990;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;font-size:24px;box-shadow:0 8px 24px rgba(99,102,241,.45);display:flex;align-items:center;justify-content:center;transition:transform .15s;user-select:none;touch-action:none}',
      '#mf7Fab:active{cursor:grabbing}',
      '#mf7Fab:hover{transform:scale(1.08)}',
      '#mf7Panel{position:fixed;right:22px;bottom:92px;width:400px;max-width:calc(100vw - 24px);height:560px;max-height:calc(100vh - 24px);min-width:300px;min-height:360px;background:var(--card,#fff);border:1px solid var(--rule,#e2e8f0);border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.25);z-index:9991;display:flex;flex-direction:column;overflow:hidden;font-size:13px;color:var(--ink,#1e293b)}',
      '#mf7Head{padding:10px 14px;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;display:flex;justify-content:space-between;align-items:center;font-weight:600;cursor:move;user-select:none;touch-action:none}',
      '#mf7Close{background:none;border:none;color:#fff;font-size:16px;cursor:pointer;padding:0 4px}',
      '#mf7Msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:9px;background:var(--surface,#f8fafc)}',
      '.mf7-msg{max-width:88%;padding:9px 12px;border-radius:12px;line-height:1.6;white-space:pre-wrap;word-break:break-word}',
      '.mf7-user{align-self:flex-end;background:#6366f1;color:#fff;border-bottom-right-radius:4px}',
      '.mf7-assist{align-self:flex-start;background:var(--card,#fff);border:1px solid var(--rule,#e2e8f0);border-bottom-left-radius:4px}',
      '.mf7-sys{align-self:center;font-size:11px;color:var(--muted,#64748b);background:rgba(100,116,139,.1);padding:4px 10px;border-radius:999px}',
      '#mf7Chips{display:flex;flex-wrap:nowrap;gap:5px;padding:6px 10px;border-top:1px solid var(--rule,#e2e8f0);background:var(--card,#fff);overflow-x:auto;white-space:nowrap;scrollbar-width:thin}',
      '.mf7-chip{border:1px solid var(--rule,#e2e8f0);background:var(--surface,#f8fafc);color:var(--ink,#1e293b);border-radius:999px;padding:3px 9px;font-size:11px;cursor:pointer;flex-shrink:0}',
      '.mf7-chip:hover{background:#eef2ff;border-color:#a5b4fc}',
      '#mf7InputRow{display:flex;gap:7px;padding:8px 10px 4px;background:var(--card,#fff)}',
      '#mf7Input{flex:1;border:1px solid var(--rule,#e2e8f0);border-radius:10px;padding:8px 11px;font-size:13px;background:var(--surface,#fff);color:var(--ink,#1e293b);outline:none}',
      '#mf7Send{background:#6366f1;border:none;color:#fff;border-radius:10px;padding:0 15px;cursor:pointer;font-size:14px}',
      '.mf7-agent{font-size:11px;color:#7c3aed;margin:2px 0}',
      '.mf7-card{border:1px dashed var(--rule,#e2e8f0);border-radius:8px;padding:8px 10px;margin-top:6px;font-size:12px;background:var(--surface,#f8fafc)}',
      '.mf7-clr{background:linear-gradient(135deg,#eef2ff,#f8fafc);border:1px solid #e0e7ff;border-radius:12px;padding:10px 12px;margin-top:6px;font-size:12.5px;line-height:1.7;box-shadow:0 1px 3px rgba(79,70,229,.08)}',
      '.mf7-clr .clr-q{color:#3730a3;font-weight:600;margin-bottom:2px}',
      '.mf7-clr .clr-g{color:var(--muted,#64748b);font-size:11.5px;margin:6px 0 8px}',
      '.mf7-clr .clr-opts{display:flex;flex-wrap:wrap;gap:6px}',
      '.mf7-clr .clr-chip{border:1px solid #c7d2fe;background:#fff;color:#4338ca;border-radius:999px;padding:4px 11px;font-size:12px;cursor:pointer;transition:all .15s;font-weight:500}',
      '[data-theme="dark"] .mf7-clr .clr-chip{background:#27272a;border-color:#4338ca;color:#a5b4fc}',
      '.mf7-clr .clr-chip:hover{background:#4338ca;color:#fff;border-color:#4338ca;transform:translateY(-1px)}',
      '.mf7-clr .clr-chip:active{transform:translateY(0)}',
      '.mf7-follow{margin-top:7px;font-size:11px;color:var(--muted,#64748b);border-top:1px dashed var(--rule,#e2e8f0);padding-top:6px}',
      '.mf7-follow b{color:#7c3aed;font-weight:600}',
      '.mf7-follow .fq{cursor:pointer;color:#6366f1;border-bottom:1px dashed #a5b4fc;margin:0 3px;display:inline-block;padding:1px 2px}',
      '.mf7-follow .fq:hover{color:#4338ca;background:#eef2ff;border-radius:4px}',
      '[data-theme="dark"] .mf7-chip:hover{background:#27272a;border-color:#4338ca}',
      '[data-theme="dark"] .mf7-clr{background:linear-gradient(135deg,rgba(99,102,241,.16),rgba(99,102,241,.03));border-color:rgba(99,102,241,.32)}',
      '[data-theme="dark"] .mf7-clr .clr-q{color:#a5b4fc}',
      '[data-theme="dark"] .mf7-follow .fq:hover{color:#a5b4fc;background:rgba(99,102,241,.12)}',
      '[data-theme="dark"] .mf7-agent{color:#a78bfa}',
      '[data-theme="dark"] .mf7-follow b{color:#a78bfa}',
      '[data-theme="dark"] .mf7-follow .fq{color:#a5b4fc;border-bottom-color:#6d28d9}',
      '#mf7Resize{position:absolute;right:0;bottom:0;width:18px;height:18px;cursor:nwse-resize;background:linear-gradient(135deg,transparent 55%,#94a3b8 55%,#94a3b8 62%,transparent 62%,transparent 70%,#94a3b8 77%,transparent 77%);border-bottom-right-radius:16px;opacity:.55;touch-action:none}',
      '.mf7-card sup{font-size:75%;vertical-align:super}',
      '.mf7-card sub{font-size:75%;vertical-align:sub}',
      '.mf7-msg sup{font-size:75%;vertical-align:super}',
      '.mf7-msg sub{font-size:75%;vertical-align:sub}',
      '@media (max-width:640px){#mf7Panel{right:8px;left:8px;width:auto}}'
    ].join('\n');
    d.head.appendChild(css);

    var fab = d.createElement('button');
    fab.id = 'mf7Fab';
    fab.innerHTML = '<svg viewBox="0 0 24 24" width="26" height="26" fill="none"><path d="M12 3C7 3 3 6.6 3 11c0 2.5 1.3 4.7 3.4 6.2L5 21l4.2-1.6c.9.3 1.8.5 2.8.5 5 0 9-3.6 9-8s-4-8-9-8z" fill="#fff" opacity="0.96"/><circle cx="8.5" cy="11" r="1.4" fill="#6366f1"/><circle cx="12" cy="11" r="1.4" fill="#6366f1"/><circle cx="15.5" cy="11" r="1.4" fill="#6366f1"/></svg>';
    fab.title = '智能小助手';
    // 浮标固定定位为常驻总入口: 不跟随鼠标拖动, 点击始终打开完整对话
    fab.addEventListener('click', function () { toggle(); });
    d.body.appendChild(fab);

    var panel = d.createElement('div');
    panel.id = 'mf7Panel';
    panel.style.display = 'none';
    // 恢复上次位置/尺寸
    var mem = {};
    try { mem = JSON.parse(localStorage.getItem('mf7_panel') || '{}'); } catch (e) {}
    if (mem.w) panel.style.width = mem.w + 'px';
    if (mem.h) panel.style.height = mem.h + 'px';
    if (mem.left != null && mem.top != null) {
      panel.style.left = mem.left + 'px';
      panel.style.top = mem.top + 'px';
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    }
    panel.innerHTML =
      '<div id="mf7Head"><span style="font-weight:600;flex:1;display:flex;align-items:center;gap:6px"><svg viewBox="0 0 24 24" width="16" height="16" fill="none"><path d="M12 3C7 3 3 6.6 3 11c0 2.5 1.3 4.7 3.4 6.2L5 21l4.2-1.6c.9.3 1.8.5 2.8.5 5 0 9-3.6 9-8s-4-8-9-8z" fill="#fff" opacity="0.96"/><circle cx="8.5" cy="11" r="1.4" fill="#6366f1"/><circle cx="12" cy="11" r="1.4" fill="#6366f1"/><circle cx="15.5" cy="11" r="1.4" fill="#6366f1"/></svg>智能小助手</span>' +
      '<button id="mf7Close" title="关闭">✕</button></div>' +
      // 四边缩放手柄
      '<div id="mf7EdgeNW" class="mf7-edge" data-edge="nw" style="position:absolute;top:-3px;left:-3px;width:12px;height:12px;cursor:nwse-resize;z-index:9;background:transparent"></div>' +
      '<div id="mf7EdgeN" class="mf7-edge" data-edge="n" style="position:absolute;top:-3px;left:12px;right:12px;height:8px;cursor:ns-resize;z-index:9;background:transparent"></div>' +
      '<div id="mf7EdgeNE" class="mf7-edge" data-edge="ne" style="position:absolute;top:-3px;right:-3px;width:12px;height:12px;cursor:nesw-resize;z-index:9;background:transparent"></div>' +
      '<div id="mf7EdgeW" class="mf7-edge" data-edge="w" style="position:absolute;top:12px;left:-3px;bottom:12px;width:8px;cursor:ew-resize;z-index:9;background:transparent"></div>' +
      '<div id="mf7EdgeE" class="mf7-edge" data-edge="e" style="position:absolute;top:12px;right:-3px;bottom:12px;width:8px;cursor:ew-resize;z-index:9;background:transparent"></div>' +
      '<div id="mf7EdgeSW" class="mf7-edge" data-edge="sw" style="position:absolute;bottom:-3px;left:-3px;width:12px;height:12px;cursor:nesw-resize;z-index:9;background:transparent"></div>' +
      '<div id="mf7EdgeS" class="mf7-edge" data-edge="s" style="position:absolute;bottom:-3px;left:12px;right:12px;height:8px;cursor:ns-resize;z-index:9;background:transparent"></div>' +
      '<div id="mf7EdgeSE" class="mf7-edge" data-edge="se" style="position:absolute;bottom:-3px;right:-3px;width:12px;height:12px;cursor:nwse-resize;z-index:9;background:transparent"></div>' +
      '<div id="mf7Msgs"></div>' +
      '<div id="mf7InputRow"><input id="mf7Input" placeholder="输入问题，或点下方快捷指令…"><button id="mf7Send">➤</button></div>' +
      '<div id="mf7Chips"></div>';
    d.body.appendChild(panel);

    // 收起按钮: 近乎从主界面淡出隐藏 — 彻底隐藏对话面板, 仅保留常驻悬浮球
    d.getElementById('mf7Close').addEventListener('click', function () {
      panel.style.display = 'none';
      fab.style.display = 'flex';
      ASSIST_STATE.open = false;
    });
    d.getElementById('mf7Send').addEventListener('click', send);
    var inp = d.getElementById('mf7Input');
    inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') send(); });

    var chipsBox = d.getElementById('mf7Chips');
    CHIPS.forEach(function (c) {
      var b = d.createElement('button');
      b.className = 'mf7-chip';
      b.textContent = c.label;
      b.title = c.msg;
      b.addEventListener('click', function () { inp.value = c.msg; send(); });
      chipsBox.appendChild(b);
    });

    setupDragResize(panel, fab);

    pushMsg('assist', '你好，我是 Dy3+ 智能小助手 🤖 需要帮忙学习吗？输入"帮助"查看功能。');
  }

  // 浮标可拖动
  function setupFabDrag(fab) {
    var drag = null;
    var win = window;
    function start(e) {
      // 不在此处 preventDefault: 触摸端会阻止 touchend 合成的 click, 导致点击打不开.
      // 仅在真实拖拽移动时阻止默认 (防止页面滚动/选择).
      drag = { sx: e.clientX, sy: e.clientY, lx: fab.offsetLeft, ty: fab.offsetTop, moved: false };
    }
    function move(e) {
      if (!drag) return;
      var dx = e.clientX - drag.sx;
      var dy = e.clientY - drag.sy;
      // 记录实际位移距离 (而非对比 offsetLeft, 避免点击抖动误判为拖拽)
      if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
        drag.moved = true;
        if (e.cancelable) e.preventDefault();  // 拖拽中才阻止滚动/选择
      }
      var nx = drag.lx + dx;
      var ny = drag.ty + dy;
      nx = Math.max(4, Math.min(nx, win.innerWidth - 60));
      ny = Math.max(4, Math.min(ny, win.innerHeight - 60));
      fab.style.left = nx + 'px';
      fab.style.top = ny + 'px';
      fab.style.right = 'auto';
      fab.style.bottom = 'auto';
    }
    function end() {
      if (drag) {
        fab._dragged = drag.moved;  // 仅真实位移 (>5px) 才算拖拽
        drag = null;
        // 拖拽后浏览器仍可能合成 click, 由外层 click 的 _dragged 判断吞掉
      }
    }
    fab.addEventListener('mousedown', start);
    fab.addEventListener('touchstart', function (e) { start(e.touches[0]); }, { passive: true });
    win.addEventListener('mousemove', move);
    win.addEventListener('touchmove', function (e) { move(e.touches[0]); }, { passive: false });
    win.addEventListener('mouseup', end);
    win.addEventListener('touchend', end);
  }

  // 拖拽 + 四边缩放（无贴边停靠）
  function setupDragResize(panel, fab) {
    var head = d.getElementById('mf7Head');
    var drag = null;
    var rs = null;
    var win = window;
    function saveState() {
      var r = panel.getBoundingClientRect();
      try {
        localStorage.setItem('mf7_panel', JSON.stringify({
          left: Math.round(r.left), top: Math.round(r.top),
          w: Math.round(r.width), h: Math.round(r.height),
        }));
      } catch (e) {}
    }
    // 拖拽
    function startDrag(e) {
      var r = panel.getBoundingClientRect();
      drag = { sx: e.clientX, sy: e.clientY, lx: r.left, ty: r.top };
      e.preventDefault();
    }
    function moveDrag(e) {
      if (!drag) return;
      var nx = drag.lx + (e.clientX - drag.sx);
      var ny = drag.ty + (e.clientY - drag.sy);
      nx = Math.max(4, Math.min(nx, win.innerWidth - 60));
      ny = Math.max(4, Math.min(ny, win.innerHeight - 40));
      panel.style.left = nx + 'px';
      panel.style.top = ny + 'px';
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    }
    function endDrag() {
      if (drag) { drag = null; saveState(); }
    }
    head.addEventListener('mousedown', startDrag);
    head.addEventListener('touchstart', function (e) { startDrag(e.touches[0]); }, { passive: false });
    win.addEventListener('mousemove', moveDrag);
    win.addEventListener('touchmove', function (e) { moveDrag(e.touches[0]); }, { passive: false });
    win.addEventListener('mouseup', endDrag);
    win.addEventListener('touchend', endDrag);

    // 四边缩放
    var edges = panel.querySelectorAll('.mf7-edge');
    var SIDE = { n: 1, s: 1, e: 1, w: 1, ne: 1, nw: 1, se: 1, sw: 1 };
    function startResize(e) {
      var edge = (e.target && e.target.dataset && e.target.dataset.edge) || '';
      if (!SIDE[edge]) return;
      e.preventDefault(); e.stopPropagation();
      var r = panel.getBoundingClientRect();
      rs = { edge: edge, sx: e.clientX, sy: e.clientY, left: r.left, top: r.top, w: r.width, h: r.height };
    }
    function moveResize(e) {
      if (!rs) return;
      var dx = e.clientX - rs.sx, dy = e.clientY - rs.sy;
      var left = rs.left, top = rs.top, w = rs.w, h = rs.h;
      var edge = rs.edge;
      if (edge.indexOf('e') >= 0) { w = Math.max(300, rs.w + dx); }
      if (edge.indexOf('w') >= 0) { var nw = rs.w - dx; if (nw >= 300) { w = nw; left = rs.left + dx; } }
      if (edge.indexOf('s') >= 0) { h = Math.max(360, rs.h + dy); }
      if (edge.indexOf('n') >= 0) { var nh = rs.h - dy; if (nh >= 360) { h = nh; top = rs.top + dy; } }
      panel.style.left = Math.max(4, Math.min(left, win.innerWidth - 60)) + 'px';
      panel.style.top = Math.max(4, Math.min(top, win.innerHeight - 40)) + 'px';
      panel.style.width = Math.max(300, Math.min(w, win.innerWidth - 8)) + 'px';
      panel.style.height = Math.max(360, Math.min(h, win.innerHeight - 8)) + 'px';
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    }
    function endResize() { if (rs) { rs = null; saveState(); } }
    edges.forEach(function (el) {
      el.addEventListener('mousedown', startResize);
      el.addEventListener('touchstart', function (e) { startResize(e.touches[0]); }, { passive: false });
    });
    win.addEventListener('mousemove', moveResize);
    win.addEventListener('touchmove', function (e) { moveResize(e.touches[0]); }, { passive: false });
    win.addEventListener('mouseup', endResize);
    win.addEventListener('touchend', endResize);
  }

  function toggle() {
    buildUI();
    var panel = d.getElementById('mf7Panel');
    var fab = d.getElementById('mf7Fab');
    ASSIST_STATE.open = !ASSIST_STATE.open;
    panel.style.display = ASSIST_STATE.open ? 'flex' : 'none';
    // 关闭时浮标淡出回到主界面, 作为唯一常驻入口保持可见
    fab.style.display = 'flex';
    if (ASSIST_STATE.open) { var inp = d.getElementById('mf7Input'); if (inp) inp.focus(); }
  }

  function pushMsg(role, html) {
    var box = d.getElementById('mf7Msgs');
    if (!box) return;
    var m = d.createElement('div');
    m.className = 'mf7-msg ' + (role === 'user' ? 'mf7-user' : (role === 'sys' ? 'mf7-sys' : 'mf7-assist'));
    m.innerHTML = html;
    box.appendChild(m);
    box.scrollTop = box.scrollHeight;
    return m;
  }

  function send() {
    var inp = d.getElementById('mf7Input');
    if (!inp) return;
    var text = inp.value.trim();
    // 忙碌状态: 提示而非静默丢弃
    if (ASSIST_STATE.busy) { pushMsg('sys', '⏳ 我正在思考上一条，请稍候…'); return; }
    // 空输入: 友好引导, 高亮快捷指令
    if (!text) {
      pushMsg('assist', '想让我帮你做什么？可以输入问题，或点下方快捷指令 👇');
      var chips = d.getElementById('mf7Chips');
      if (chips) { chips.style.background = 'rgba(99,102,241,.08)'; setTimeout(function () { chips.style.background = ''; }, 600); }
      return;
    }
    inp.value = '';
    pushMsg('user', esc(text));
    // 用户理解: 语料提取 (防抖 + 频率限制, 每会话 ≤3 次)
    if (!send._extractCount) send._extractCount = 0;
    if (send._extractCount < 3) {
      clearTimeout(send._extractTimer);
      send._extractTimer = setTimeout(function () {
        send._extractCount++;
        var turns = ASSIST_STATE.history.slice(-4).map(function (h) {
          return { role: h.role, text: h.text };
        });
        apiReq('POST', '/api/user-understanding/extract', { learner_id: learnerId(), turns: turns })
          .catch(function () {});
      }, 2500);
    }
    handle(text);
  }

  // ---- 对话管理 (DST: 上下文 + 指代消解 + 追问澄清 + 多意图) ----
  function resolveContext(text) {
    var t = text;
    // 指代消解: "它/这个/那" → 上次主题
    if (REFER_RE.test(t) && ASSIST_STATE.topic) {
      t = t.replace(REFER_RE, ASSIST_STATE.topic + ' ');
    }
    return t;
  }

  function handle(text) {
    ASSIST_STATE.turns += 1;
    ASSIST_STATE.history.push({ role: 'user', text: text });
    if (ASSIST_STATE.history.length > 12) ASSIST_STATE.history.shift(); // 轮次上限

    var resolved = resolveContext(text);

    // 1. 追问槽位处理: 上一轮在追问时, 本轮回答填入
    if (ASSIST_STATE.pendingSlot) {
      var slot = ASSIST_STATE.pendingSlot;
      ASSIST_STATE.pendingSlot = null;
      var n = parseInt(text, 10);
      if (slot.intent === 'practice' && n > 0 && n <= 10) {
        practiceWith(text, n);
        return;
      }
      // 未按预期回答 → 继续引导
      pushMsg('assist', '收到。不过我还需要你确认一下：' + slot.ask + '\n（例如直接回复数字，或输入"帮助"）');
      ASSIST_STATE.pendingSlot = slot;
      return;
    }

    // 2. 意图路由
    var r = routeIntent(resolved);
    var intent = r.intent;

    // 3. 执行
    if (intent === 'multi') {
      pushMsg('assist', '好的，我依次为你处理：' + r.segments.map(function (s) { return '「' + esc(s) + '」'; }).join('、'));
      r.segments.forEach(function (seg, i) {
        setTimeout(function () { handleSub(seg); }, 600 * (i + 1));
      });
      return;
    }
    handleSub(resolved, intent);
  }

  function handleSub(text, intentOverride) {
    var resolved = resolveContext(text);
    var r = intentOverride ? { intent: intentOverride } : routeIntent(resolved);
    var intent = r.intent;
    ASSIST_STATE.lastIntent = intent;

    if (intent === 'reset') {
      ASSIST_STATE.history = [];
      ASSIST_STATE.topic = '';
      ASSIST_STATE.pendingSlot = null;
      pushMsg('sys', '对话已重置');
      pushMsg('assist', INTENT_REPLY.reset);
      return;
    }
    if (intent === 'chitchat') {
      pushMsg('assist', INTENT_REPLY.chitchat);
      return;
    }
    if (intent === 'help') {
      pushMsg('assist', INTENT_REPLY[intent] || INTENT_REPLY.fallback);
      return;
    }
    // 意图无法识别: 观察为主, 仅此时向系统请求澄清式提问 (帮助理解用户)
    if (intent === 'fallback') {
      pushMsg('assist', esc(INTENT_REPLY.fallback || ''));
      askClarify({ intent: 'query', ambiguous: true });
      return;
    }

    pushMsg('assist', esc(INTENT_REPLY[intent] || INTENT_REPLY.help));

    if (intent === 'viz') {
      // 可视化指令: 直接调度 4 Agent, 后端检测到画图意图会返回 viz 数据, 图随回答渲染
      ASSIST_STATE.topic = extractTopic(resolved) || resolved;
      runQueryAgents(resolved);
    } else if (intent === 'query') {
      // 答疑: 无实质问题/意图模糊时请求澄清
      // 但含领域术语(发光/猝灭/能级等)的短查询直接通过
      var hasDomainTerm = /(发光|猝灭|能级|跃迁|光谱|荧光|磷光|效率|制备|合成|掺杂|浓度|温度|波长|色度|寿命|量子|带隙|晶格|基质|敏化|激活剂|衰减|余辉|陷阱|缺陷|价态|氧化还原)/.test(resolved);
      if ((resolved.length <= 2 || /^(答疑|帮我答疑|问个问题|请问|查|查查)$/.test(resolved)) && !hasDomainTerm) {
        pushMsg('assist', '请问你想了解什么？例如：\n· Dy3+ 的发光机理是什么？\n· 浓度猝灭怎么避免？\n· 量子效率如何测量？');
        askClarify({ intent: 'query', ambiguous: true });
        return;
      }
      ASSIST_STATE.topic = extractTopic(resolved);
      if (!ASSIST_STATE.topic) {
        // 有实质问题但提取不到主题 → 模糊, 请求澄清
        askClarify({ intent: 'query', ambiguous: true, detail: resolved });
        return;
      }
      runQueryAgents(resolved);
    } else if (intent === 'knowledge') {
      var q = extractKnowledgeQuery(resolved);
      if (!q) {
        pushMsg('assist', '请问你想查询哪个知识点？例如"查一下 量子效率"、"浓度猝灭 的知识"。');
        return;
      }
      ASSIST_STATE.topic = q;
      runKnowledge(q);
    } else if (intent === 'understand') {
      apiReq('POST', '/api/user-understanding/profile', { learner_id: learnerId() }).then(function (ins) {
        var i = ins || {};
        var lines = [];
        lines.push('我对你的了解（可随时纠正我）：');
        if (i.interests && i.interests.length) {
          lines.push('· 兴趣：' + i.interests.map(function (x) { return x.topic; }).join('、'));
        }
        if (i.goals && i.goals.length) {
          lines.push('· 目标：' + i.goals.join('、'));
        }
        if (i.pace && i.pace !== 'unknown') {
          lines.push('· 学习节奏：' + (i.pace === 'fragmented' ? '碎片化' : '集中式'));
        }
        if (i.expression && i.expression !== 'unknown') {
          lines.push('· 表达偏好：' + (i.expression === 'concise' ? '简洁' : '详细'));
        }
        lines.push('· 了解度：' + Math.round((i.confidence || 0) * 100) + '%');
        if (!lines.length) lines.push('我还在了解你，多和我聊聊吧！');
        pushMsg('assist', esc(lines.join('\n')));
      }).catch(function () {
        pushMsg('assist', '读取理解画像失败，请稍后再试');
      });
      return;
    } else if (intent === 'memory') {
      runMemory();
    } else if (intent === 'guidance') {
      // 引导式咨询: 用户自己也不清楚 → 结合学情画像给出建议
      apiReq('POST', '/api/user-understanding/guide', {
        learner_id: learnerId(), context: { utterance: resolved }
      }).then(function (d) {
        var g = d && d.guidance;
        if (!g) {
          pushMsg('assist', '暂时无法生成建议，请稍后再试');
          return;
        }
        var lines = ['💡 ' + esc(g.reason || '')];
        if (g.suggested_kps && g.suggested_kps.length) {
          lines.push('\n📌 建议方向：');
          g.suggested_kps.forEach(function (k) {
            lines.push('· ' + esc(k.name || k.kp_id) + '（掌握度 ' + Math.round((k.mastery || 0) * 100) + '%）');
          });
        }
        if (g.next_steps && g.next_steps.length) {
          lines.push('\n🗺 下一步：');
          g.next_steps.forEach(function (s, i) { lines.push((i + 1) + '. ' + esc(s)); });
        }
        lines.push('\n回复「就按这个来」开始，或告诉我你的偏好，我会调整建议。');
        pushMsg('assist', lines.join('\n'));
      }).catch(function () {
        pushMsg('assist', '引导咨询失败，请稍后再试');
      });
      return;
    } else if (intent === 'ability') {
      runAbility();
    } else if (intent === 'practice') {
      // 追问澄清: 优先吸取消息中的数量
      var m = resolved.match(/(\d+)\s*道|\d+/);
      var n = m ? Math.min(parseInt(m[1] || m[0], 10), 10) : 5;
      practiceWith(resolved, n);
    } else if (intent === 'domainQuery') {
      // 领域知识查询: 纯元素名(≤3字符)引导用户补充问题, 含领域术语的短查询直接触发Agent
      var hasDomainTerm = /(发光|猝灭|能级|跃迁|光谱|荧光|磷光|效率|制备|合成|掺杂|浓度|温度|波长|色度|寿命|量子|带隙|晶格|基质|敏化|激活剂|衰减|余辉|陷阱|缺陷|价态|氧化还原|机理|原理|是什么|怎么|如何|多少|为什么|解释|区别|介绍|讲讲|知识|材料|方法|工艺|性能|应用|结构|特性|性质|参数|影响|作用|提高|降低|增强|改变|优化|测量|测试|计算|分析|机制|条件|过程|途径|策略|路线|方案|体系|复合|掺杂|包覆|纳米|微米|薄膜|陶瓷|玻璃|晶体|粉末|溶液|气体|固体|液体|制备|合成|研制|开发|设计|构建|构建|组装|生长|沉积|镀膜|烧结|退火|煅烧|熔融|冷却|干燥|研磨|混合|搅拌|过滤|离心|纯化|提纯|分离|萃取|色谱|质谱|XRD|TEM|SEM|PL|XPS|FTIR|Raman|DSC|TGA|XAFS|EXAFS|XANES|EDX|EDS|XRF|ICP|AES|AFM|STM|BET|UV|IR|NIR|EPR|NMR|ESR|DLS|Zeta|电位|粒度|比表面|孔径|孔容|密度|硬度|强度|韧性|脆性|弹性|塑性|粘度|熔点|沸点|相变|玻璃化|结晶|非晶|多晶|单晶|外延|取向|织构|应力|应变|膨胀|收缩|导热|导电|电阻|电导|介电|铁电|压电|热电|磁电|多铁|超导|半导体|绝缘体|导体|能带|费米|价带|导带|缺陷|空位|间隙|替位|反位|位错|晶界|相界|表面|界面|吸附|脱附|扩散|迁移|离子|电子|空穴|激子|声子|光子|磁子|极化子|双极化子|孤子|极化|磁化|饱和|矫顽|剩磁|磁滞|磁阻|磁光|电光|声光|非线性|光学|非线性|光折变|光致|电致|热致|压致|光致变色|电致变色|热致变色|光催化|电催化|光解|水解|氧化|还原|惰性|活泼|稳定|亚稳|介稳|平衡|非平衡|可逆|不可逆|自发|非自发|吸热|放热|熵变|焓变|吉布斯|活化|速率|常数|级数|半衰|周期|频率|振幅|相位|偏振|干涉|衍射|散射|反射|透射|吸收|发射|激发|消光|折射|色散|色差|像差|分辨率|灵敏度|信噪|检出|检测限|定量|定性|半定量|阈值|临界|饱和|过饱和|成核|生长|溶解|沉淀|络合|螯合|配位|键合|键能|键长|键角|构型|构象|异构|同分|手性|旋光|圆二|CD|ORD|ORD|VCD|ROA|拉曼|红外|紫外|可见|近红外|中红外|远红外|太赫兹|微波|射频|低频|高频|超高频|厘米波|毫米波|X射线|γ射线|中子|电子|质子|α粒子|β粒子|反粒子|正电子|负电子|μ子|π子|K介子|重子|介子|强子|轻子|夸克|胶子|玻色子|费米子|超对称|暗物质|暗能量|宇宙|星系|恒星|行星|卫星|彗星|小行星|流星|陨石|太阳|月球|地球|火星|金星|水星|木星|土星|天王|海王|冥王)/.test(resolved);
      if ((resolved.length <= 3 && !hasDomainTerm) || /^(Dy|Eu|Ce|Tb|Yb|Er|Nd|Sm|Pr|Ho|Tm|Gd|Lu|La|Sc|Y)$/i.test(resolved.trim())) {
        pushMsg('assist', '已识别到稀土元素/领域关键词，你想了解什么？\n例如：\n· Dy³⁺的发光机理\n· 浓度猝灭怎么避免\n· 量子效率如何提高');
        return;
      }
      ASSIST_STATE.topic = extractTopic(resolved) || resolved;
      runQueryAgents(resolved);
    } else if (intent === 'recommend' || intent === 'path') {
      var v = viewOf(intent);
      if (v) setTimeout(function () { jumpTo(v); }, 400);
    } else {
      var v = viewOf(intent);
      if (v) setTimeout(function () { jumpTo(v); }, 400);
    }
  }

  function extractTopic(text) {
    // 从问题提取主题词 (粗略: 取名词片段, 用于指代消解)
    var m = text.match(/(Dy3\+|Eu3\+|Yb3\+|Er3\+|Ce3\+|镝|铕|量子效率|浓度猝灭|热猝灭|发光机理|荧光粉|能级|跃迁|色度|显色|温度|XRD|PL|光谱|合成|组态|封装|蓝光|黄光|红光|白光)[^，。？！？\s]{0,8}/);
    return m ? m[0] : '';
  }

  function extractKnowledgeQuery(text) {
    var t = text.replace(/帮我查|查一下|查询|搜索|检索|知识库|的知识|介绍一下/g, ' ').replace(/\s+/g, ' ').trim();
    return t || text;
  }

  function practiceWith(text, count) {
    pushMsg('assist', '好的，出 ' + count + ' 道动态变题（题型随机轮换）～正在打开【学习练习】…');
    setTimeout(function () {
      // 带数量进入练习 (会话标记)
      localStorage.setItem('mf7_practice_count', String(count));
      jumpTo('practice');
    }, 500);
  }

  function jumpTo(view) {
    var sv = window.sv;
    if (sv) { sv(view); return; }
    var el = d.querySelector('.sidebar-child[data-view="' + view + '"]');
    if (el) el.click();
  }

  // 澄清式提问: 仅在请求难以理解/意图模糊时, 请求系统澄清问题 (观察为主)
  function askClarify(opts) {
    if (!opts) opts = {};
    apiReq('POST', '/api/user-understanding/ask', {
      learner_id: learnerId(),
      context: { ambiguous: !!opts.ambiguous, intent: opts.intent || '', detail: opts.detail || '' }
    }).then(function (d) {
      var q = d && d.question;
      if (!q || !q.question) return;
      var optTxt = (q.options || []).map(function (o) { return '「' + o + '」'; }).join(' ');
      pushMsg('assist', '💡 ' + esc(q.question) + '\n' + optTxt);
    }).catch(function () {});
  }

  // 启发式导学: 根据问题与答案, 自然生成 2 个深化方向的追问建议 (可点选继续)
  function heuristicFollowUps(text, answer) {
    var t = (text || '').toLowerCase();
    var a = (answer || '').toLowerCase();
    var topics = [];
    if (/(dy|镝)/i.test(t)) topics.push('Dy³⁺');
    if (/(eu|铕)/i.test(t)) topics.push('Eu³⁺');
    if (/(ce|铈)/i.test(t)) topics.push('Ce³⁺');
    if (/(tb|铽)/i.test(t)) topics.push('Tb³⁺');
    if (/(猝灭|浓度..灭|quenching)/i.test(t)) topics.push('浓度猝灭');
    if (/(量子效率|量子|效率)/i.test(t)) topics.push('量子效率');
    if (/(能级|光谱|跃迁|transition)/i.test(t)) topics.push('能级与光谱');
    if (/(发光|荧光|磷光|发光机理)/i.test(t)) topics.push('发光机理');
    if (/(制备|合成|掺杂|工艺)/i.test(t)) topics.push('制备与合成');
    if (/(热稳定|热猝灭|温度)/i.test(t)) topics.push('热稳定性');
    if (/(应用|led|照明|显示)/i.test(t)) topics.push('实际应用');
    if (/(寿命|衰减|余辉)/i.test(t)) topics.push('荧光寿命');
    if (/(能量传递|交叉弛豫|敏化)/i.test(t)) topics.push('能量传递');
    if (/(色度|显色|色温|cri)/i.test(t)) topics.push('色度与显色');
    // 无强主题时, 用答案里高亮词兜底
    if (!topics.length) {
      var kw = ['发光', '猝灭', '能级', '跃迁', '光谱', '效率', '制备', '掺杂', '温度', '寿命', '能量传递', '色度'];
      for (var i = 0; i < kw.length; i++) {
        if (a.indexOf(kw[i].toLowerCase()) !== -1) { topics.push(kw[i]); if (topics.length >= 2) break; }
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

  // 答疑: 调度 4 个 Agent
  function runQueryAgents(text) {
    ASSIST_STATE.busy = true;
    var statusEl = pushMsg('assist', '<div class="mf7-agent">⏳ 正在为你解答…</div>');
    var stepIdx = 0;
    var timer = null;

    // 传递上下文记忆: 最近 3 轮对话历史, 供 Agent 多轮关联
    var context = {
      recent_history: ASSIST_STATE.history.slice(-6).map(function (h) { return ({ role: h.role, text: h.text }); }),
      topic: ASSIST_STATE.topic,
    };
    apiReq('POST', '/api/query', { query: text, learner_id: learnerId(), context: context }).then(function (d) {
      clearInterval(timer);
      var clarify = d && d.clarify;
      // 模糊问题 → 人性化引导式澄清卡片 (Trae/Codex 风格: 先复述理解 → 可点选方向 → 引导补全)
      if (clarify && !(d && d.answer)) {
        if (statusEl) statusEl.innerHTML = '';
        statusEl = null;
        var cq = String(clarify.question || '');
        var cg = String(clarify.guidance || '');
        var opts = (clarify.options || []).slice(0, 6);
        var chips = opts.map(function (o) {
          return '<span class="clr-chip" data-opt="' + esc(o) + '">' + esc(o) + '</span>';
        }).join('');
        var clrEl = pushMsg('assist',
          '<div class="mf7-clr">' +
          '<div class="clr-q">💡 ' + formatChemical(esc(cq)) + '</div>' +
          (cg ? '<div class="clr-g">' + esc(cg) + '</div>' : '') +
          '<div class="clr-opts">' + chips + '</div>' +
          '</div>');
        // 点选方向 → 作为对原问题的补充, 重新调度 4 Agent
        if (clrEl) {
          var selectDir = function (opt) {
            ASSIST_STATE.pendingSlot = null;
            var full = text.trim() + ' ' + String(opt).trim();
            ASSIST_STATE.topic = extractTopic(full) || ASSIST_STATE.topic || String(opt).trim();
            pushMsg('user', esc(String(opt).trim()));
            runQueryAgents(full);
          };
          clrEl.querySelectorAll('.clr-chip').forEach(function (chip) {
            chip.addEventListener('click', function () { selectDir(chip.getAttribute('data-opt')); });
          });
        }
        ASSIST_STATE.busy = false;
        return;
      }
      if (statusEl) statusEl.innerHTML = '<div class="mf7-agent" style="color:#16a34a">✅ 完成</div>';
      var answer = (d && d.answer) || '';
      var conf = d && d.confidence != null ? d.confidence : '-';
      var review = (d && d.review) || {};
      var verdict = review.verdict || 'approved';
      var qtype = (d && d.question_type) || '';
      var el = pushMsg('assist',
        '<strong>答疑结果</strong>' +
        '<div class="mf7-card" style="line-height:1.8">' + formatChemical(esc(answer)) + '</div>' +
        '<div style="margin-top:6px;font-size:11px;color:var(--muted)">置信度 ' + Math.round(conf * 100) + '% · 审核 ' + esc(verdict) + (qtype ? ' · 意图 ' + esc(qtype) : '') + '</div>');
      // 动态可视化 (M-F8): 响应携带 viz 数据时, 实时渲染能级/跃迁图 (非静态预设)
      var viz = d && d.viz;
      if (viz && viz.hit && viz.data) {
        var vEl = pushMsg('assist',
          '<div class="mf7-viz">' +
          '<div style="font-size:12px;font-weight:600;color:var(--accent-ink,#6366f1);margin-bottom:6px">📊 ' + esc(viz.note || '动态可视化') + '</div>' +
          '<div class="mf7-viz-body" style="background:var(--surface,#fff);border:1px solid var(--rule,#e2e8f0);border-radius:10px;padding:10px;overflow:auto"></div>' +
          '<div style="font-size:10.5px;color:var(--muted);margin-top:4px">依据你的指令实时生成 · 数据驱动 · <a href="javascript:void(0)" class="mf7-viz-open" style="color:var(--accent-ink,#6366f1)">在实验台打开</a></div>' +
          '</div>');
        if (vEl) {
          var vb = vEl.querySelector('.mf7-viz-body');
          if (window.MF8Viz && vb) window.MF8Viz.renderFromData(vb, viz.data, viz.viz_type);
          var openLnk = vEl.querySelector('.mf7-viz-open');
          if (openLnk) openLnk.addEventListener('click', function () {
            if (window.MF8Viz) window.MF8Viz.inject(viz.data, viz.viz_type);
            if (window.location.hash) window.location.hash = '';
            var nav = document.querySelector('[data-view="atomic-viz"], .nav-item[data-view="atomic-viz"]');
            if (nav) { nav.click(); } else if (window.__goto) { window.__goto('atomic-viz'); }
          });
        }
      }
      // 四 Agent 协同链路可视化 (统一组件: 流水线 + 广播通道 + 自纠 + 共识度)
      var flowEv = (d && d.flow_events) || [];
      var bcEv = (d && d.broadcast_events) || [];
      if (flowEv.length || bcEv.length) {
        var collabHtml = '';
        if (window.DPCollab) {
          collabHtml = window.DPCollab.renderCollaboration(flowEv, bcEv, {
            selfCorrection: !!(d && d.self_correction),
            consensus: (d && d.consensus_score != null) ? d.consensus_score : null,
          });
        }
        if (collabHtml) {
          pushMsg('assist', collabHtml);
        } else {
          // 兜底: 组件未加载时用简单 chips
          var AG = { 'agent.learning.diagnosis': '🔍 学情诊断', 'agent.knowledge.generation': '🧠 知识生成', 'agent.quality.review': '🛡️ 审核校验', 'agent.guidance.decision': '🎯 导学决策' };
          var flowChips = flowEv.map(function (fe, i) {
            var nm = AG[fe.agent] || String(fe.agent || '').replace('agent.', '');
            return (i ? ' <span style="color:var(--muted)">→</span> ' : '') + '<span style="font-weight:600;color:#6366f1">' + esc(nm) + '</span>';
          }).join('');
          pushMsg('assist', '<div class="mf7-card"><div style="font-size:11px;font-weight:600;color:#3730a3;margin-bottom:4px">🔄 四 Agent 协同链路</div>' + flowChips + '</div>');
        }
      }
      // 启发式导学: 答后再给 2 个自然追问方向, 引导深化 (不打断, 可点选继续)
      var follow = heuristicFollowUps(text, answer);
      if (el && follow.length) {
        var fq = follow.map(function (f) {
          return '<span class="fq" data-q="' + esc(f) + '">' + esc(f) + '</span>';
        }).join('');
        var fEl = pushMsg('assist', '<div class="mf7-follow">🔍 想继续深挖？点一点：' + fq + '</div>');
        if (fEl) {
          fEl.querySelectorAll('.fq').forEach(function (chip) {
            chip.addEventListener('click', function () {
              var q = chip.getAttribute('data-q');
              pushMsg('user', esc(q));
              runQueryAgents(q);
            });
          });
        }
      }
      ASSIST_STATE.lastIntent = 'query';
      ASSIST_STATE.busy = false;
    }).catch(function (e) {
      clearInterval(timer);
      if (statusEl) statusEl.innerHTML = '<div class="mf7-agent" style="color:#dc2626">⚠ Agent 协作异常</div>';
      pushMsg('assist', '答疑失败：' + esc(e.message || '请稍后再试'));
      ASSIST_STATE.busy = false;
    });
  }

  // 记忆状态 (FSRS 遗忘曲线 + 复习队列, 面板内回显)
  function runMemory() {
    ASSIST_STATE.busy = true;
    apiReq('GET', '/l2/memory/' + encodeURIComponent(learnerId())).then(function (d) {
      var kp = (d && d.kp_retentions) || (d && d.retentions) || {};
      var keys = Object.keys(kp).slice(0, 6);
      if (!keys.length) {
        pushMsg('assist', '暂无记忆状态数据——完成学习练习后，BKT/FSRS 会为每个知识点记录遗忘曲线与复习队列。');
      } else {
        var rows = keys.map(function (k) {
          var v = kp[k] || {};
          var r = typeof v === 'number' ? v : (v.retention != null ? v.retention : v.p_retention);
          var next = v.next_review_at || v.next_review || '';
          if (next && String(next).length > 9 && next < 1e12) next = next * 1000;
          var nextTxt = next ? new Date(Number(next)).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
          return '<div class="mf7-card"><b>' + esc(k) + '</b> · 记忆保持率 ' + (r != null ? Math.round(Number(r) * 100) + '%' : '-') +
            (nextTxt ? '<div style="font-size:11px;color:var(--muted);margin-top:3px">下次复习: ' + esc(nextTxt) + '</div>' : '') + '</div>';
        }).join('');
        pushMsg('assist', '<strong>记忆状态（FSRS）</strong>' + rows);
      }
      ASSIST_STATE.busy = false;
    }).catch(function (e) {
      pushMsg('assist', '记忆状态读取失败：' + esc(e.message));
      ASSIST_STATE.busy = false;
    });
  }

  // 能力估计 (IRT theta + 置信区间)
  function runAbility() {
    ASSIST_STATE.busy = true;
    apiReq('GET', '/l2/irt/ability/' + encodeURIComponent(learnerId())).then(function (d) {
      var ab = (d && d.ability) || d || {};
      var theta = ab.theta != null ? ab.theta : ab.ability;
      var se = ab.se != null ? ab.se : ab.standard_error;
      var ci = ab.confidence_interval || (theta != null && se != null ? [theta - 1.96 * se, theta + 1.96 * se] : null);
      if (theta == null) {
        pushMsg('assist', '暂无能力估计数据——先完成一次动态练习/答疑，IRT 会估算你的能力水平（theta）。');
      } else {
        var ciTxt = ci ? '95% 置信区间 [' + ci.map(function (x) { return Number(x).toFixed(2); }).join(', ') + ']' : '';
        pushMsg('assist', '<strong>能力估计（IRT）</strong>' +
          '<div class="mf7-card">当前能力值 θ = ' + Number(theta).toFixed(2) +
          (se != null ? '（标准误 ' + Number(se).toFixed(3) + '）' : '') +
          (ciTxt ? '<div style="font-size:11px;color:var(--muted);margin-top:3px">' + esc(ciTxt) + '</div>' : '') + '</div>');
      }
      ASSIST_STATE.busy = false;
    }).catch(function (e) {
      pushMsg('assist', '能力估计读取失败：' + esc(e.message));
      ASSIST_STATE.busy = false;
    });
  }

  // 知识查询 (带引用标注)
  function runKnowledge(q) {
    ASSIST_STATE.busy = true;
    apiReq('POST', '/l3/retrieve/keyword', { query: q, top_k: 3 }).then(function (d) {
      var results = (d && d.results) || [];
      if (!results.length) {
        pushMsg('assist', '知识库未找到与「' + esc(q) + '」直接相关的内容。\n可试试："查一下 浓度猝灭"、"量子效率"、"XRD"。');
      } else {
        var html = results.map(function (it, i) {
          var src = it.source || it.source_type || it.chunk_id || '';
          return '<div class="mf7-card"><strong>[' + (i + 1) + '] ' + esc(it.title || it.name || '知识条目') + '</strong><br>' +
            formatChemical(esc(String(it.content || it.text || '').slice(0, 180))) +
            (src ? '<div style="margin-top:4px;font-size:10px;color:#8b5cf6">来源: ' + esc(String(src).slice(0, 40)) + '</div>' : '') + '</div>';
        }).join('');
        pushMsg('assist', '<strong>知识库检索：「' + esc(q) + '」</strong>' + html);
      }
      ASSIST_STATE.busy = false;
    }).catch(function (e) {
      pushMsg('assist', '知识检索失败：' + esc(e.message));
      ASSIST_STATE.busy = false;
    });
  }

  if (d.readyState === 'loading') {
    d.addEventListener('DOMContentLoaded', function () { setTimeout(buildUI, 800); });
  } else {
    setTimeout(buildUI, 800);
  }
})();
