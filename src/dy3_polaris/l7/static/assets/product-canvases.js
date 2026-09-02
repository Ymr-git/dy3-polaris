/* DY3 Polaris — five factual product canvases.
 *
 * This layer does not create a second teaching flow.  It reads the public
 * learner/task projections already produced by R03/R05/R06/R07 and sends all
 * user actions back through the existing query, practice and resource-event
 * endpoints.
 */
(function () {
  'use strict';

  var d = document;
  var cache = { learnerId: '', promise: null, value: null };
  var agentOrder = [
    'agent.learning.diagnosis',
    'agent.knowledge.generation',
    'agent.quality.review',
    'agent.guidance.decision'
  ];
  var agentLabels = {
    'agent.learning.diagnosis': '学情诊断',
    'agent.knowledge.generation': '知识生成',
    'agent.quality.review': '科学审核',
    'agent.guidance.decision': '导学决策'
  };
  var agentRoles = {
    'agent.learning.diagnosis': '解释学习状态与教学深度',
    'agent.knowledge.generation': '组织材料机制、证据与学习资源',
    'agent.quality.review': '挑战无依据结论并决定是否修订',
    'agent.guidance.decision': '综合审核结果与下一步学习行动'
  };

  function internals() { return window.DY3CanvasInternals || {}; }
  function learnerId() {
    var api = internals();
    return api.learnerId ? api.learnerId() : (window.dy3LearnerId ? window.dy3LearnerId() : 'guest-unavailable');
  }
  function apiReq(method, path, body) {
    var api = internals();
    if (api.apiReq) return api.apiReq(method, path, body);
    if (window.api && window.api.rq) return window.api.rq(method, path, body);
    return fetch(path, {
      method: method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined
    }).then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok || (payload && payload.code !== 0)) throw new Error((payload && payload.message) || ('HTTP ' + response.status));
        return payload && payload.data !== undefined ? payload.data : payload;
      });
    });
  }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function attr(value) { return esc(value).replace(/'/g, '&#39;'); }
  function array(value) { return Array.isArray(value) ? value : []; }
  function object(value) { return value && typeof value === 'object' && !Array.isArray(value) ? value : {}; }
  function clamp(value, low, high) { return Math.max(low, Math.min(high, Number(value || 0))); }
  function scientificTypography(value) {
    return String(value == null ? '' : value)
      .replace(/Dy\s*3\s*(?:\\\s*)?\+/gi, 'Dy³⁺')
      .replace(/Dy³\s*\+/g, 'Dy³⁺')
      .replace(/cm\s*-\s*1\b/gi, 'cm⁻¹')
      .replace(/\b4f\s*9\b/gi, '4f⁹');
  }
  function normalizeScientificText(container) {
    if (!container || !d.createTreeWalker) return;
    var walker = d.createTreeWalker(container, window.NodeFilter.SHOW_TEXT);
    var node;
    while ((node = walker.nextNode())) node.nodeValue = scientificTypography(node.nodeValue);
  }
  function formatTime(value) {
    var number = Number(value || 0);
    if (!number) return '—';
    if (number < 1e12) number *= 1000;
    try { return new Date(number).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }); }
    catch (ignore) { return '—'; }
  }
  function shortId(value) {
    var text = String(value || '');
    return text.length > 28 ? text.slice(0, 12) + '…' + text.slice(-6) : text;
  }
  function prettyState(value) {
    var state = String(value || 'UNKNOWN').toUpperCase();
    return {
       COMPLETED: '已完成', RUNNING: '进行中', UNKNOWN: '尚未知',
       UNKNOWN_LEARNER: '尚未完成诊断', INITIAL_UNDERSTANDING: '初步认识中',
       INITIAL_LEARNER_MODEL: '初始模型', ADAPTIVE_UNDERSTANDING: '动态理解中',
       PERSONALIZED_TEACHING: '个性化教学中', LONG_TERM_COMPANION: '持续学习中',
      EVIDENCE_BACKED: '证据充分', MODEL_ONLY: '模型推断',
      MASTERED: '已掌握', LEARNING: '学习中', LEARNING_GAP: '待学习',
      FULL_RELEASE: '完整发布', LIMITED_RELEASE: '限制发布', APPROVED: '通过',
      SUPPORTS: '证据支持', PARTIAL: '部分支持', INSUFFICIENT: '证据不足',
      CONTRADICTS: '存在冲突', CANDIDATE: '候选证据'
    }[state] || value || '尚未知';
  }
  // 发布门拒绝时展示"具体审核原因"优先（如：问题核心覆盖门要求修订/证据不足），
  // 而非仅展示通用发布消息，帮助学习者理解为什么没有公开回答。
  function withheldReason(quality, review, verdict) {
    var specific = String((review && review.reason) || '').trim();
    var generic = String((quality && quality.message) || '').trim();
    var verdictKey = String(verdict || '').toLowerCase();
    var isReviewDriven = verdictKey === 'needs_review' || verdictKey === 'rejected';
    if (specific && isReviewDriven && generic.indexOf(specific) < 0) return specific;
    return generic || specific || '系统不会展示未经审核的内容。';
  }
  function eventLabel(value) {
    var key = String(value || '').replace(/([a-z])([A-Z])/g, '$1_$2').toUpperCase();
    return {
      TASK_CREATED: '任务建立', TASK_DECOMPOSED: '任务拆解', STATE_CHANGED: '阶段切换',
      AGENT_STARTED: '开始分析', AGENT_FINISHED: '完成分析',
      AGENT_CONTRIBUTION_RECORDED: '贡献记录', CONTRIBUTION_PRODUCED: '专业贡献',
      SUBTASK_READY: '子任务就绪', RETRIEVAL_COMPLETED: '证据检索',
      EVIDENCE_RETRIEVED: '证据更新', REVIEW_COMPLETED: '科学审核',
      RELEASE_DECIDED: '发布裁决', RESOURCE_ISSUED: '学习资源形成',
      GUIDANCE_DECIDED: '学习决策', CHALLENGE_CREATED: '提出挑战',
      CHALLENGE_RAISED: '提出挑战', REVISION_STARTED: '开始修订',
      REVISION_COMPLETED: '完成修订', RE_RETRIEVAL: '补充检索',
      RE_RETRIEVAL_COMPLETED: '补充检索完成'
    }[key] || String(value || '运行事件').replace(/_/g, ' ');
  }
  function producerLabel(value) {
    var raw = String(value || '');
    return agentLabels[raw] || {
      api_query: '任务入口', run_guidance: '协同调度',
      'task.planning': '任务规划', system: '系统'
    }[raw] || raw.replace(/^agent\./, '').replace(/\./g, ' · ') || '系统';
  }
  function teachingLabel(value) {
    var key = String(value || '').toUpperCase();
    return {
      FOUNDATION: '基础解释', FOUNDATION_CONCEPTUAL: '基础概念',
      INTERMEDIATE: '机制理解', ADVANCED: '进阶分析', RESEARCH: '科研讨论',
      MECHANISM: '机制分析', EXPLANATION: '概念解释', COMPARISON: '材料比较',
      HEALTH_EVALUATION: '健康照明评价', EVIDENCE: '证据深入'
    }[key] || prettyState(value);
  }
  function decisionLabel(value) {
    var key = String(value || '').toUpperCase();
    return {
      PRACTICE: '完成诊断练习', DIAGNOSE_FIRST: '先完成诊断',
      ANSWER: '继续当前学习', CLARIFY: '补充问题信息',
      RETRIEVE: '补充检索', REVISE: '修订后再学习', REFUSE: '暂不发布'
    }[key] || prettyState(value);
  }
  function sourceClassLabel(value) {
    var key = String(value || '').toUpperCase();
    return {
       OBSERVED: '真实作答', MODEL_INFERRED: '模型推断', DERIVED: '系统计算',
       OBSERVED_AND_MODEL_INFERRED: '真实作答与模型状态',
       DECLARED_PRIOR: '自愿声明先验',
      AUTHORED_PRACTICE_BANK: '已编写题库',
      'MODEL_INFERRED+AUTHORED_PRACTICE_BANK': '模型判断与已编写题库',
      UNKNOWN: '尚无数据'
    }[key] || String(value || '来源未声明').replace(/_/g, ' ');
  }
  function resourceFamilyLabel(value) {
    return {
      knowledge_understanding: '知识讲义', research_practice: '实证工作单',
      assessment_practice: '分阶练习'
    }[String(value || '').toLowerCase()] || String(value || '学习资源');
  }
  function resourceFormLabel(value) {
    return {
      guided_long_read: '专题讲义', evidence_analysis_workbook: '证据分析工作单',
      research_task: '科研实践任务', practice_bank_launch: '题库分阶练习'
    }[String(value || '').toLowerCase()] || String(value || '学习资源').replace(/_/g, ' ');
  }
  function difficultyLabel(value) {
    var key = String(value || '').toUpperCase();
    return {
      FOUNDATION: '基础', BEGINNER: '入门', INTERMEDIATE: '进阶',
      ADVANCED: '高级', RESEARCH: '科研', DIAGNOSE_THEN_MAINTAIN: '先诊断再匹配',
      DIAGNOSTIC: '诊断', MAINTAIN: '巩固', CHALLENGE: '挑战'
    }[key] || prettyState(value);
  }
  function learnerEventLabel(value) {
    var key = String(value || '').toUpperCase();
    return {
      QUERY: '完成学习任务', PRACTICE: '完成练习', RESOURCE_INTERACTION: '学习资源反馈',
      LEARNING_EVENT: '学习事件', ANSWER_SUBMITTED: '提交作答'
    }[key] || eventLabel(value);
  }
  function learnerOutcomeLabel(value) {
    var key = String(value || '').toUpperCase();
    if (!key || key === 'NONE' || key === 'UNKNOWN') return '';
    return prettyState(value);
  }
  function recommendedStepLabel(step) {
    if (!step) return '';
    if (typeof step === 'string') return step;
    var item = object(step);
    var action = {
      learn: '学习', review: '复习', practice: '练习', diagnose: '诊断',
      deepen: '深入', challenge: '挑战', remediate: '补足'
    }[String(item.action || '').toLowerCase()] || '';
    var target = item.topic || item.title || item.concept_name || item.kp_name || item.kp_id || item.next_action || '';
    return [action, target].filter(Boolean).join('：');
  }
  function contributionSummary(agentId, entries, data) {
    if (!entries.length) return '本次没有公开贡献';
    if (agentId === 'agent.learning.diagnosis') {
      var learner = object(data.learner_context), teaching = object(learner.teaching_decision);
      var weak = array(learner.weak_points || learner.weak_kps).length;
      return '解释深度：' + teachingLabel(teaching.content_depth || learner.lifecycle_stage || 'UNKNOWN') +
        (weak ? '；识别 ' + weak + ' 项学习缺口' : '；尚无足够真实作答形成薄弱点判断');
    }
    if (agentId === 'agent.knowledge.generation') {
      return '形成 ' + publicClaims(data).length + ' 项公开科学判断，绑定 ' + array(data.sources).length + ' 个公开来源';
    }
    if (agentId === 'agent.quality.review') {
      var review = object(data.review);
      return review.reason || review.message || ('审核结果：' + prettyState(review.verdict || review.status));
    }
    var path = array(data.recommended_path);
    var nextStep = path.length ? recommendedStepLabel(path[0]) : '';
    return nextStep ? '下一步：' + nextStep : '形成下一步教学决策';
  }
  function statusClass(value) {
    var state = String(value || '').toUpperCase();
    if (/APPROVED|COMPLETED|FULL_RELEASE|MASTERED|SUPPORTS|VERIFIED/.test(state)) return 'is-ok';
    if (/REVISE|CHALLENGE|LIMITED|LEARNING|CANDIDATE|INSUFFICIENT/.test(state)) return 'is-warn';
    if (/REJECT|FAILED|CONFLICT|BLOCK/.test(state)) return 'is-bad';
    return 'is-unknown';
  }
  function icon(name) {
    var paths = {
      overview: '<path d="M3 11.5 12 4l9 7.5"></path><path d="M5.5 10.5V21h13V10.5M9 21v-6h6v6"></path>',
      task: '<rect x="4" y="4" width="6" height="6" rx="1"></rect><rect x="14" y="4" width="6" height="6" rx="1"></rect><rect x="4" y="14" width="6" height="6" rx="1"></rect><rect x="14" y="14" width="6" height="6" rx="1"></rect>',
      agents: '<circle cx="8" cy="8" r="3"></circle><circle cx="17" cy="7" r="2.5"></circle><path d="M2.5 20c.5-4 2.5-6 5.5-6s5 2 5.5 6M14 14c3.5-.5 6 1.5 6.5 5"></path>',
      knowledge: '<path d="M4 5.5A3.5 3.5 0 0 1 7.5 2H11v18H7.5A3.5 3.5 0 0 0 4 23z"></path><path d="M20 5.5A3.5 3.5 0 0 0 16.5 2H13v18h3.5A3.5 3.5 0 0 1 20 23z"></path>',
      growth: '<path d="M4 20V8m0 12h16"></path><path d="m6.5 16 4-5 3 2 5-7"></path><path d="m15 6 3.5-.2.2 3.5"></path>',
      search: '<circle cx="10.5" cy="10.5" r="6.5"></circle><path d="m15.5 15.5 5 5"></path>',
      check: '<path d="m5 12 4 4L19 6"></path>',
      target: '<circle cx="12" cy="12" r="8"></circle><circle cx="12" cy="12" r="3"></circle><path d="M12 2v3m0 14v3M2 12h3m14 0h3"></path>',
      evidence: '<path d="M7 3h7l4 4v14H7z"></path><path d="M14 3v5h5M10 13h5m-5 4h5"></path>',
      review: '<path d="M12 3 4.5 6v5.5c0 4.5 3 7.5 7.5 9.5 4.5-2 7.5-5 7.5-9.5V6z"></path><path d="m8.5 12 2.2 2.2 4.8-5"></path>',
      arrow: '<path d="M5 12h14m-5-5 5 5-5 5"></path>',
      alert: '<path d="M12 3 2.8 20h18.4z"></path><path d="M12 9v5m0 3v.2"></path>'
    };
    return '<svg class="pc-icon" viewBox="0 0 24 24" aria-hidden="true">' + (paths[name] || paths.task) + '</svg>';
  }

  function loadLatestTask(lid) {
    return apiReq('GET', '/api/learning-tasks/' + encodeURIComponent(lid)).then(function (listing) {
      var tasks = array(listing && listing.tasks);
      if (!tasks.length) return { tasks: [], task: null, taskData: null, question: '' };
      var taskId = tasks[0].task_id;
      return apiReq('GET', '/api/learning-tasks/' + encodeURIComponent(lid) + '/' + encodeURIComponent(taskId)).then(function (detail) {
        var task = detail && detail.task ? detail.task : null;
        var project = internals().taskProjection;
        return {
          tasks: tasks,
          task: task,
          taskData: task && project ? project(task) : (task && task.public_result) || null,
          question: task ? String(task.query || task.brief || '') : ''
        };
      });
    }).catch(function () { return { tasks: [], task: null, taskData: null, question: '' }; });
  }

  function loadTruth(force) {
    var lid = learnerId();
    if (!force && cache.learnerId === lid && cache.value) return Promise.resolve(cache.value);
    if (!force && cache.learnerId === lid && cache.promise) return cache.promise;
    cache = { learnerId: lid, promise: null, value: null };
    cache.promise = Promise.all([
      apiReq('GET', '/api/learning-workspace/' + encodeURIComponent(lid)).catch(function () { return null; }),
      apiReq('GET', '/l2/profile/' + encodeURIComponent(lid)).catch(function () { return null; }),
      apiReq('GET', '/api/match-report/' + encodeURIComponent(lid)).catch(function () { return null; }),
      loadLatestTask(lid)
    ]).then(function (values) {
      cache.value = {
        learnerId: lid,
        workspace: values[0] || {},
        profile: values[1] || {},
        match: values[2] || {},
        tasks: values[3].tasks,
        task: values[3].task,
        taskData: values[3].taskData,
        question: values[3].question
      };
      cache.promise = null;
      return cache.value;
    });
    return cache.promise;
  }

  function pageLoading(title) {
    return '<div class="pc-page pc-loading"><div class="pc-loading-line"></div><div class="pc-loading-card"><span class="spinner"></span><strong>' + esc(title) + '</strong></div></div>';
  }
  function bindRoutes(root) {
    if (!root) return;
    root.querySelectorAll('[data-pc-route]').forEach(function (button) {
      button.addEventListener('click', function () {
        var route = button.getAttribute('data-pc-route');
        var question = button.getAttribute('data-pc-question');
        if (question) sessionStorage.setItem('dy3_pending_query', question);
        if (route && window.sv) window.sv(route);
      });
    });
  }
  function bindShell() {
    var search = d.getElementById('globalTaskSearch');
    if (search && !search.getAttribute('data-bound')) {
      search.setAttribute('data-bound', '1');
      search.addEventListener('keydown', function (event) {
        if (event.key !== 'Enter') return;
        var value = String(search.value || '').trim();
        if (value) sessionStorage.setItem('dy3_pending_query', value);
        if (window.sv) window.sv('query');
      });
      d.addEventListener('keydown', function (event) {
        if ((event.ctrlKey || event.metaKey) && String(event.key).toLowerCase() === 'k') {
          event.preventDefault(); search.focus();
        }
      });
    }
    var role = d.getElementById('roleChip');
    if (role) {
      var lid = learnerId();
      role.innerHTML = icon('agents') + '<span><small>当前学习者</small><strong>' + esc(shortId(lid)) + '</strong></span>';
      if (!role.getAttribute('data-pc-bound')) {
       role.setAttribute('data-pc-bound', '1');
       role.addEventListener('click', function () {
          if (window.sv) window.sv('overview');
          setTimeout(function () {
            d.dispatchEvent(new CustomEvent('dy3-open-learner-profile'));
          }, 20);
       });
      }
    }
    var modelConfig = d.getElementById('settingsBtn');
    if (modelConfig && !modelConfig.getAttribute('data-pc-bound')) {
      modelConfig.setAttribute('data-pc-bound', '1');
      modelConfig.addEventListener('click', function () {
        modelConfig.setAttribute('aria-expanded', 'true');
        d.dispatchEvent(new CustomEvent('dy3-open-model-config'));
      });
    }
    var navIcons = { overview: 'overview', query: 'task', 'agents-chain': 'agents', kb: 'knowledge', 'learn-weak': 'growth' };
    d.querySelectorAll('.topnav-item[data-view]').forEach(function (button) {
      var holder = button.querySelector('.td-icon');
      if (holder && navIcons[button.getAttribute('data-view')]) holder.innerHTML = icon(navIcons[button.getAttribute('data-view')]);
    });
  }

  function canonicalDeclared(value, type) {
    var text = String(value || '').trim().toLowerCase();
    if (type === 'stage') {
      if (/本科|undergraduate/.test(text)) return '本科阶段';
      if (/研究生|硕士|graduate|master/.test(text)) return '研究生阶段';
      if (/博士|phd/.test(text)) return '博士阶段';
      if (/科研|researcher/.test(text)) return '科研人员';
      if (/行业|professional/.test(text)) return '行业从业者';
    }
    if (type === 'experience') {
      if (/刚开始|introductory/.test(text)) return '刚开始了解';
      if (/课程|coursework/.test(text)) return '修过相关课程';
      if (/实验|lab/.test(text)) return '有实验经历';
      if (/科研|research/.test(text)) return '有科研经历';
      if (/行业|industry/.test(text)) return '有行业经历';
    }
    if (type === 'representation') {
      if (/visual|图/.test(text)) return '图示与关系';
      if (/evidence|论文|证据/.test(text)) return '论文证据';
      if (/practice|实验|案例/.test(text)) return '实践案例';
      if (/structured|文字|结构/.test(text)) return '结构化文字';
    }
    return value || '';
  }

  function option(value, label, current) {
    return '<option value="' + attr(value) + '"' + (String(current || '') === String(value) ? ' selected' : '') + '>' + esc(label) + '</option>';
  }

  function startRealDiagnostic(ctx, close) {
    var ws = object(ctx.workspace);
    var practice = array(ws.quick_actions).filter(function (item) {
      return item && item.action_type === 'PRACTICE' && item.status === 'AVAILABLE';
    })[0] || {};
    var context = object(practice.context);
    var kpIds = String(context.kp_ids || '').split(',').filter(Boolean);
    try {
      sessionStorage.setItem('dy3_practice_attempt_purpose', 'DIAGNOSTIC');
      if (kpIds.length) sessionStorage.setItem('dy3_practice_target_kps', JSON.stringify(kpIds));
    } catch (ignore) {}
    if (close) close();
    if (window.sv) window.sv('practice');
  }

  function openLearnerProfileDialog(ctx, onChanged) {
    var old = d.getElementById('pcLearnerProfileDialog');
    if (old) old.remove();
    var summary = object(object(ctx.workspace).learner_summary);
    var declared = object(summary.declared_background);
    var stage = canonicalDeclared(declared.learning_stage, 'stage');
    var experience = canonicalDeclared(declared.domain_experience, 'experience');
    var representation = canonicalDeclared(declared.representation_preference, 'representation');
    var overlay = d.createElement('div');
    overlay.id = 'pcLearnerProfileDialog';
    overlay.className = 'pc-profile-dialog';
    overlay.innerHTML = '<section role="dialog" aria-modal="true" aria-labelledby="pcProfileTitle"><header><div><span>初始学情</span><h2 id="pcProfileTitle">完善学习背景</h2></div><button type="button" data-pc-dialog-close aria-label="关闭">×</button></header>' +
      '<p class="pc-dialog-intro">字段均可跳过；这里保存的是声明先验，掌握度只由真实作答和模型对齐产生。</p>' +
      '<div class="pc-profile-form"><label><span>学习或工作阶段</span><select id="pcProfileStage">' + option('', '不填写', stage) + option('本科阶段', '本科阶段', stage) + option('研究生阶段', '研究生阶段', stage) + option('博士阶段', '博士阶段', stage) + option('科研人员', '科研人员', stage) + option('行业从业者', '行业从业者', stage) + '</select></label>' +
      '<label><span>专业背景</span><input id="pcProfileMajor" maxlength="120" value="' + attr(declared.professional_background || '') + '" placeholder="如：材料科学、物理/光学、光电照明"></label>' +
      '<label><span>领域经历</span><select id="pcProfileExperience">' + option('', '不填写', experience) + option('刚开始了解', '刚开始了解', experience) + option('修过相关课程', '修过相关课程', experience) + option('有实验经历', '有实验经历', experience) + option('有科研经历', '有科研经历', experience) + option('有行业经历', '有行业经历', experience) + '</select></label>' +
      '<label><span>当前学习目标</span><input id="pcProfileGoal" maxlength="120" value="' + attr(declared.learning_goal || '') + '" placeholder="如：理解Dy³⁺白光调控与健康照明评价"></label>' +
      '<label class="pc-profile-wide"><span>偏好的学习呈现</span><select id="pcProfileRepresentation">' + option('', '不填写', representation) + option('结构化文字', '结构化文字', representation) + option('图示与关系', '图示与关系', representation) + option('论文证据', '论文证据', representation) + option('实践案例', '实践案例', representation) + '</select></label></div>' +
      '<div id="pcProfileFeedback" class="pc-profile-feedback" aria-live="polite"></div>' +
      '<footer><button type="button" class="pc-danger-link" id="pcProfileClear">清除自愿信息</button><div><button type="button" class="pc-secondary" id="pcProfileDiagnostic">开始真实诊断</button><button type="button" class="pc-primary" id="pcProfileSave">保存并生成初始方案</button></div></footer></section>';
    d.body.appendChild(overlay);
    function close() { if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay); }
    overlay.querySelector('[data-pc-dialog-close]').addEventListener('click', close);
    overlay.addEventListener('click', function (event) { if (event.target === overlay) close(); });
    var diagnostic = overlay.querySelector('#pcProfileDiagnostic');
    diagnostic.addEventListener('click', function () { startRealDiagnostic(ctx, close); });
    var save = overlay.querySelector('#pcProfileSave');
    save.addEventListener('click', function () {
      var fields = [
        ['learning_stage', overlay.querySelector('#pcProfileStage').value],
        ['professional_background', overlay.querySelector('#pcProfileMajor').value],
        ['domain_experience', overlay.querySelector('#pcProfileExperience').value],
        ['learning_goal', overlay.querySelector('#pcProfileGoal').value],
        ['representation_preference', overlay.querySelector('#pcProfileRepresentation').value]
      ].filter(function (item) { return String(item[1] || '').trim(); });
      var feedback = overlay.querySelector('#pcProfileFeedback');
      if (!fields.length) { feedback.textContent = '没有填写内容；系统会继续保持未知状态。'; return; }
      save.disabled = true;
      feedback.textContent = '正在保存…';
      Promise.all(fields.map(function (item) {
        return apiReq('POST', '/api/user-understanding/answer', {
          learner_id: ctx.learnerId,
          payload: { slot_key: item[0], value: String(item[1]).trim() }
        });
      })).then(function () {
        invalidate();
        feedback.textContent = '已保存。初始方案将由Learner Intelligence重新比较。';
        setTimeout(function () { close(); if (onChanged) onChanged(); }, 260);
      }).catch(function (error) {
        feedback.textContent = (error && error.message) || '保存失败。';
        save.disabled = false;
      });
    });
    var clear = overlay.querySelector('#pcProfileClear');
    clear.addEventListener('click', function () {
      var feedback = overlay.querySelector('#pcProfileFeedback');
      clear.disabled = true;
      apiReq('DELETE', '/api/user-understanding/profile?learner_id=' + encodeURIComponent(ctx.learnerId)).then(function () {
        invalidate();
        feedback.textContent = '自愿信息已清除；真实任务与作答记录未删除。';
        setTimeout(function () { close(); if (onChanged) onChanged(); }, 260);
      }).catch(function (error) {
        feedback.textContent = (error && error.message) || '清除失败。';
        clear.disabled = false;
      });
    });
  }

  function renderInitialProfileAnalysis(analysis) {
    var data = object(analysis);
    var candidates = array(data.candidates);
    if (!candidates.length) return '';
    var basis = sourceClassLabel(data.evidence_basis || 'UNKNOWN');
    var cards = candidates.map(function (item) {
      var rationale = array(item.rationale)[0] || '等待更多学习者证据';
      return '<article class="pc-plan-candidate' + (item.selected ? ' is-selected' : '') + '"><header><span>' + esc(teachingLabel(item.content_depth)) + '</span>' + (item.selected ? '<em>当前采用</em>' : '') + '</header><strong>' + esc(item.label) + '</strong><p>' + esc(rationale) + '</p><div><i style="--fit:' + Math.round(clamp(item.fit_score, 0, 1) * 100) + '%"></i></div><small>方案匹配 ' + Math.round(clamp(item.fit_score, 0, 1) * 100) + '%</small></article>';
    }).join('');
    return '<section class="pc-panel pc-initial-plans"><div class="pc-panel-head"><div><h2>初始教学方案比较</h2><p>同一事实边界下比较解释深度与呈现方式</p></div><span class="pc-status ' + (data.evidence_basis === 'UNKNOWN' ? 'is-unknown' : 'is-learning') + '">' + esc(basis) + '</span></div><div class="pc-plan-grid">' + cards + '</div><footer><span>' + (data.status === 'DIAGNOSTIC_REQUIRED' ? '需用真实题库校准' : '已随真实学习证据调整') + '</span><button data-pc-profile-editor>调整背景</button></footer></section>';
  }

  function bindLearnerProfileEntry(root, ctx, rerender) {
    root.querySelectorAll('[data-pc-profile-editor]').forEach(function (button) {
      button.addEventListener('click', function () { openLearnerProfileDialog(ctx, rerender); });
    });
  }

  function masteryForCoverage(item, profile) {
    var mastery = object(profile.kp_mastery);
    var values = array(item.practice_kps).filter(function (kp) { return mastery[kp] !== undefined; }).map(function (kp) { return Number(mastery[kp]); });
    if (!values.length) return null;
    return values.reduce(function (sum, value) { return sum + value; }, 0) / values.length;
  }
  function conceptState(item, profile) {
    var value = masteryForCoverage(item, profile);
    if (value == null) return { key: 'unknown', label: '待诊断', value: null };
    if (value >= 0.75) return { key: 'mastered', label: '已掌握', value: value };
    if (value >= 0.35) return { key: 'learning', label: '学习中', value: value };
    return { key: 'gap', label: '需补强', value: value };
  }
  function groupConcepts(coverage, profile) {
    var groups = [
      { title: '基础理论', keys: /能级|晶体场|4f/i, items: [] },
      { title: '发光机制', keys: /发射|跃迁|弛豫/i, items: [] },
      { title: '性能调控', keys: /猝灭|浓度|强度比|效率/i, items: [] },
      { title: '健康照明', keys: /CIE|CCT|CRI|蓝光|健康|白光|色/i, items: [] }
    ];
    array(coverage).forEach(function (item) {
      var name = String(item.name || item.concept_id || '');
      var group = groups.filter(function (candidate) { return candidate.keys.test(name); })[0] || groups[2];
      group.items.push({ item: item, state: conceptState(item, profile) });
    });
    return groups;
  }
  function renderConceptOverview(groups) {
    return '<div class="pc-concept-columns">' + groups.map(function (group, groupIndex) {
      var rows = group.items.length ? group.items.slice(0, 4).map(function (entry) {
        var evidence = String(entry.item.evidence_status || 'NONE');
        return '<article class="pc-concept-node state-' + entry.state.key + '"><span class="pc-state-dot"></span><div><strong>' + esc(entry.item.name || entry.item.concept_id) + '</strong><small>' + esc(entry.state.label) + (entry.state.value == null ? '' : ' · ' + Math.round(entry.state.value * 100) + '%') + '</small></div><em title="公开证据状态">' + (evidence === 'RELEASED_TASK_EVIDENCE' ? '证据' : '') + '</em></article>';
      }).join('') : '<div class="pc-column-empty">当前没有映射节点</div>';
      return '<section class="pc-concept-column" data-column="' + groupIndex + '"><header>' + esc(group.title) + '</header>' + rows + '</section>';
    }).join('') + '</div>';
  }

  function renderOverview(container) {
    container.innerHTML = pageLoading('读取学习者事实与知识路径');
    bindShell();
    loadTruth(true).then(function (ctx) {
      if (!container.isConnected) return;
      var ws = object(ctx.workspace);
      var profile = object(ctx.profile);
      var match = object(ctx.match);
       var report = object(match.report);
       var summary = object(ws.learner_summary);
       var declared = object(summary.declared_background);
       var hasModelState = Number(summary.modelled_kp_count || 0) > 0;
       var stateProfile = hasModelState ? profile : { kp_mastery: {}, kp_names: {} };
       var coverage = array(ws.capability_coverage);
       var groups = groupConcepts(coverage, stateProfile);
       var findings = array(report.findings);
       var mastery = hasModelState ? object(profile.kp_mastery) : {};
      var masteryValues = Object.keys(mastery).map(function (key) { return Number(mastery[key]); });
      var counts = {
        mastered: masteryValues.filter(function (value) { return value >= 0.75; }).length,
        learning: masteryValues.filter(function (value) { return value > 0 && value < 0.75; }).length,
        unknown: Math.max(0, coverage.length - masteryValues.length)
      };
       var level = declared.learning_stage || (hasModelState ? profile.level : '') || prettyState(ws.lifecycle_stage || 'UNKNOWN_LEARNER');
      var goal = declared.learning_goal || '尚未填写学习目标';
      var next = object(report.next_action);
      var nextLabel = next.reason || object(ws.current_challenge_decision).reason || '完成一次真实作答后生成下一步建议';
      var findingLabels = { VERIFIED_WEAKNESS: '薄弱', PREREQUISITE_GAP: '先修缺口', MISCONCEPTION: '误概念', UNKNOWN: '未知' };
      var weaknesses = findings.length ? findings.slice(0, 5).map(function (item) {
        return '<li><span class="pc-mini-diamond"></span><strong>' + esc(item.label || item.reference || '未命名概念') + '</strong><em class="' + statusClass(item.type) + '">' + esc(findingLabels[item.type] || item.type) + '</em></li>';
      }).join('') : '<li class="pc-empty-row">尚无真实作答支持薄弱点判断</li>';
      var taskRows = ctx.tasks.length ? ctx.tasks.slice(0, 5).map(function (task) {
        return '<tr><td>' + formatTime(task.updated_at || task.created_at) + '</td><td><strong>' + esc(task.brief || task.query || task.task_id) + '</strong><small>' + esc(shortId(task.task_id)) + '</small></td><td><span class="pc-status ' + statusClass(task.state) + '">' + esc(prettyState(task.state)) + '</span></td><td><button data-pc-route="query">查看任务 →</button></td></tr>';
      }).join('') : '<tr><td colspan="4" class="pc-table-empty">尚无服务端学习任务；发起问题后才会出现记录。</td></tr>';
      var missing = [];
      if (!declared.learning_stage) missing.push('学习阶段');
      if (!declared.professional_background) missing.push('专业背景');
      if (!declared.learning_goal) missing.push('学习目标');
      var resume = ctx.taskData ? '<button class="pc-primary" data-pc-route="query">继续上次任务 ' + icon('arrow') + '</button>' : '<button class="pc-primary" data-pc-route="query">开始学习 ' + icon('arrow') + '</button>';

       container.innerHTML = '<article class="pc-page pc-overview-page">' +
         '<header class="pc-page-title"><div><h1>学习总览</h1></div></header>' +
         '<div class="pc-overview-grid"><main>' +
         '<section class="pc-profile-strip"><button class="pc-avatar" data-pc-profile-editor aria-label="完善学习背景">' + esc(String(ctx.learnerId).slice(0, 1).toUpperCase()) + '</button><div class="pc-profile-name"><strong>' + esc(shortId(ctx.learnerId)) + '</strong><span>' + esc(level) + '</span></div><div class="pc-profile-goal"><span>' + icon('target') + '</span><div><small>当前学习目标</small><strong>' + esc(goal) + '</strong></div></div>' + resume + '</section>' +
         renderInitialProfileAnalysis(ws.initial_profile_analysis) +
         '<section class="pc-panel pc-mastery-map"><div class="pc-panel-head"><div><h2>概念掌握图谱</h2></div><div class="pc-legend"><span class="mastered">已掌握</span><span class="learning">学习中</span><span class="gap">需补强</span><span class="unknown">待诊断</span></div></div>' + renderConceptOverview(groups) + '</section>' +
        '<section class="pc-panel pc-recent-table"><div class="pc-panel-head"><div><h2>最近学习</h2></div></div><div class="pc-table-wrap"><table><thead><tr><th>时间</th><th>任务 / 问题</th><th>结果</th><th></th></tr></thead><tbody>' + taskRows + '</tbody></table></div></section>' +
        '</main><aside class="pc-overview-rail">' +
        '<section class="pc-panel"><div class="pc-panel-head"><h2>当前学习状态</h2></div><div class="pc-state-sources"><div><span class="pc-round is-ok">' + icon('check') + '</span><p><strong>已掌握</strong><small>模型记录达到阈值</small></p><em>' + counts.mastered + ' 个</em></div><div><span class="pc-round is-learning">' + icon('growth') + '</span><p><strong>学习中</strong><small>已有模型记录但未稳定</small></p><em>' + counts.learning + ' 个</em></div><div><span class="pc-round is-unknown">?</span><p><strong>尚未知</strong><small>没有足够真实作答</small></p><em>' + counts.unknown + ' 个</em></div></div></section>' +
        '<section class="pc-panel"><div class="pc-panel-head"><h2>薄弱点</h2><small>' + findings.length + ' 项判断</small></div><ul class="pc-weak-list">' + weaknesses + '</ul></section>' +
        '<section class="pc-panel pc-next-card"><div class="pc-panel-head"><h2>下一步</h2></div><span>推荐行动</span><strong>' + esc(next.type || object(ws.current_challenge_decision).decision || 'DIAGNOSE_FIRST') + '</strong><p>' + esc(nextLabel) + '</p><button class="pc-primary" data-pc-route="' + (next.type === 'PRACTICE' ? 'practice' : 'query') + '">进入学习 ' + icon('arrow') + '</button></section>' +
         '<section class="pc-panel pc-missing-card"><div class="pc-panel-head"><h2>需要确认的信息</h2><button data-pc-profile-editor>编辑</button></div>' + (missing.length ? '<ul>' + missing.map(function (item) { return '<li><span>' + esc(item) + '</span><em>尚未填写</em></li>'; }).join('') + '</ul>' : '<p class="pc-success-copy">自愿背景已填写</p>') + '<button class="pc-secondary pc-diagnostic-button" data-pc-diagnostic>真实题库诊断</button></section>' +
         '</aside></div></article>';
       normalizeScientificText(container);
       bindRoutes(container);
       bindLearnerProfileEntry(container, ctx, function () { renderOverview(container); });
       container.querySelectorAll('[data-pc-diagnostic]').forEach(function (button) {
         button.addEventListener('click', function () { startRealDiagnostic(ctx); });
       });
    }).catch(function (error) {
      container.innerHTML = '<div class="pc-page pc-error"><h1>学习总览</h1><p>' + esc(error.message || '学习状态暂不可用') + '</p><button class="pc-primary" data-pc-route="query">仍可发起任务</button></div>';
      bindRoutes(container);
    });
  }

  function renderRichText(text) {
    var lines = scientificTypography(text).replace(/\r/g, '').split('\n');
    var html = [], list = '';
    function closeList() { if (list) { html.push('</' + list + '>'); list = ''; } }
    lines.forEach(function (raw) {
      var line = raw.trim();
      if (!line) { closeList(); return; }
      var heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) { closeList(); html.push('<h' + Math.min(4, heading[1].length + 1) + '>' + esc(heading[2]) + '</h' + Math.min(4, heading[1].length + 1) + '>'); return; }
      var bullet = line.match(/^[-*]\s+(.+)$/);
      if (bullet) { if (list !== 'ul') { closeList(); html.push('<ul>'); list = 'ul'; } html.push('<li>' + esc(bullet[1]) + '</li>'); return; }
      var numbered = line.match(/^\d+[.)]\s+(.+)$/);
      if (numbered) { if (list !== 'ol') { closeList(); html.push('<ol>'); list = 'ol'; } html.push('<li>' + esc(numbered[1]) + '</li>'); return; }
      closeList(); html.push('<p>' + esc(line).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>') + '</p>');
    });
    closeList();
    return html.join('');
  }
  function firstAnswer(text) {
    var lines = scientificTypography(text).replace(/\r/g, '').split('\n').map(function (line) {
      return line.replace(/[#*_`>-]/g, ' ').replace(/[ \t]+/g, ' ').trim();
    }).filter(Boolean);
    return lines.slice(0, 6).join('\n').slice(0, 560);
  }
  function renderMechanism(data) {
    var context = object(data.knowledge_context);
    var nodes = array(context.nodes);
    var edges = array(context.edges).filter(function (edge) { return edge && edge.source && edge.target; });
    var byId = {};
    nodes.forEach(function (node) { byId[String(node.concept_id || '')] = node; });
    if (!edges.length) {
      var names = [];
      array(data.sources).forEach(function (source) { array(source.kp_names).forEach(function (name) { if (names.indexOf(name) < 0) names.push(name); }); });
      return names.length ? '<div class="pc-related-concepts">' + names.slice(0, 5).map(function (name) { return '<span>' + esc(name) + '</span>'; }).join('') + '</div>' : '<div class="pc-empty-state">当前任务没有可公开的机制关系。</div>';
    }
    /* A Concept Relation payload is a graph, not an implicit sequence.  Each
       edge is rendered as its own source-relation-target statement so that
       independent branches cannot become a false mechanism chain. */
    return '<div class="pc-mechanism-flow">' + edges.slice(0, 4).map(function (edge) {
      var source = byId[String(edge.source)] || {};
      var target = byId[String(edge.target)] || {};
      return '<article class="pc-mechanism-relation"><div><span>Concept</span><strong>' + esc(source.name || edge.source) + '</strong></div><div class="pc-relation-arrow"><em>' + esc(edge.relation_type || 'related') + '</em>' + icon('arrow') + '</div><div><span>' + esc(target.role || 'Concept') + '</span><strong>' + esc(target.name || edge.target) + '</strong></div></article>';
    }).join('') + '</div>';
  }
  function publicClaims(data) {
    return array(object(object(data.knowledge_context).scientific_grounding).claims);
  }
  function primaryEvidence(items) {
    var evidence = array(items);
    return evidence.find(function (item) {
      return /SUPPORTS|VERIFIED/.test(String(item.level || item.support_status || item.status || '').toUpperCase());
    }) || evidence[0] || {};
  }
  function evidenceReasonLabel(value, count) {
    var reason = String(value || '').toLowerCase();
    if (reason.indexOf('same scope') >= 0 || reason.indexOf('atomic claim') >= 0) return '来源在相同条件范围内直接支持该结论';
    if (reason.indexOf('direct support is not established') >= 0) return '仅具科学相关性，未作为直接支持';
    if (reason.indexOf('contradict') >= 0) return '来源与结论存在冲突';
    return count ? count + ' 条证据记录' : '未声明直接支持';
  }
  function taskResources(data) {
    return array(data && data.learning_resources).filter(function (resource) {
      return String(resource && resource.source_type || '').toLowerCase() !== 'template' &&
        String(resource && resource.review_status || '').toLowerCase() !== 'template_not_scientific_answer';
    });
  }
  function renderTaskEvidenceVisual(data) {
    var claims = publicClaims(data), review = object(data.review);
    if (!claims.length) return '<div class="pc-empty-state">本次任务没有形成可公开的 Claim–Evidence–Review 映射。</div>';
    return '<div class="pc-evidence-canvas"><header><div><span>本次任务</span><strong>' + esc(shortId(data.task_id)) + '</strong></div><div><span>公开结论</span><strong>' + claims.length + '</strong></div><div><span>Reviewer</span><strong class="' + statusClass(review.verdict || review.status) + '">' + esc(prettyState(review.verdict || review.status)) + '</strong></div></header><div class="pc-evidence-lanes"><div>科学结论</div><div>绑定证据</div><div>审核状态</div></div>' + claims.map(function (claim, index) {
      var evidence = array(claim.evidence), primary = primaryEvidence(evidence);
      return '<article class="pc-evidence-visual-row"><section><span>' + String(index + 1).padStart(2, '0') + '</span><p>' + esc(claim.statement || '未命名结论') + '</p></section><i>' + icon('arrow') + '</i><section><strong>' + esc(primary.source || primary.chunk_id || '未绑定公开来源') + '</strong><small>' + esc(evidenceReasonLabel(primary.reason, evidence.length)) + '</small></section><i>' + icon('arrow') + '</i><section class="pc-evidence-review"><strong class="' + statusClass(claim.support_status) + '">' + esc(prettyState(claim.support_status || 'INSUFFICIENT')) + '</strong><small>' + esc(prettyState(claim.reviewer_status || review.verdict || review.status || 'UNKNOWN')) + '</small></section></article>';
    }).join('') + '</div>';
  }
  function evidenceRows(data, limit) {
    var claims = publicClaims(data);
    if (claims.length) return claims.slice(0, limit || 4).map(function (claim, index) {
      var evidence = array(claim.evidence), primary = primaryEvidence(evidence);
      return '<article class="pc-evidence-row"><span>' + String(index + 1).padStart(2, '0') + '</span><div><strong>' + esc(claim.statement || '未命名结论') + '</strong><p>' + esc(evidence.length ? (primary.source || primary.chunk_id || '来源已绑定') : '没有达到公开绑定条件的证据') + '</p></div><em class="' + statusClass(claim.support_status) + '">' + esc(prettyState(claim.support_status || 'INSUFFICIENT')) + '</em></article>';
    }).join('');
    return array(data.evidence).slice(0, limit || 4).map(function (item, index) {
      var text = internals().text ? internals().text(item) : (item.content || item.text || '');
      return '<article class="pc-evidence-row"><span>' + String(index + 1).padStart(2, '0') + '</span><div><strong>公开证据片段</strong><p>' + esc(String(text).slice(0, 150)) + '</p></div><em class="is-unknown">未声明支持级别</em></article>';
    }).join('') || '<div class="pc-empty-state">本次结果没有可公开证据。</div>';
  }

  function renderTaskResult(data, question, container, queryInput) {
    var quality = object(data.quality_release);
    var allowed = Boolean(quality.eligible) && /FULL_RELEASE|LIMITED_RELEASE/.test(String(quality.status || ''));
    var answer = allowed ? String(data.answer || '') : '';
    var review = object(data.review);
    var verdict = review.verdict || review.status || '未提供';
    var trace = array(data.agent_trace);
    var resources = taskResources(data);
    var resourceData = Object.assign({}, data, { learning_resources: resources });
    var hasConceptRelations = array(object(data.knowledge_context).edges).some(function (edge) {
      return edge && edge.source && edge.target;
    });
    var challengeCount = array(data.collab_lines).filter(function (line) { return /CHALLENGE|REVISION|RE_RETRIEVAL/.test(String(line.label || '')); }).length;
    var learner = object(data.learner_context);
    var teaching = object(learner.teaching_decision);
    var host = container.closest('.canvas-query-launch');
    if (host) host.classList.add('has-product-task');
    cache.value = cache.value || { learnerId: learnerId(), tasks: [] };
    cache.value.taskData = data; cache.value.question = question;

    var resourcesHtml = internals().renderResources ? internals().renderResources(resourceData) : '<div class="pc-empty-state">没有通过发布门的个性化资源。</div>';
    var steps = agentOrder.map(function (agentId) {
      var entries = trace.filter(function (item) { return item.agent_id === agentId; });
      return '<article class="' + (entries.length ? 'is-complete' : 'is-empty') + '"><span>' + icon(entries.length ? 'check' : 'agents') + '</span><div><strong>' + esc(agentLabels[agentId]) + '</strong><small>' + esc(contributionSummary(agentId, entries, data)) + '</small></div></article>';
    }).join('');
    var releaseLabel = quality.status || 'WITHHELD';
    container.innerHTML = '<article class="pc-page pc-task-page">' +
      '<header class="pc-task-header"><div><h1>' + esc(question) + '</h1><p>' + esc(teachingLabel(teaching.content_depth || learner.lifecycle_stage || 'UNKNOWN')) + ' · ' + esc(teachingLabel(data.question_type || data.task_mode || 'UNKNOWN')) + '</p></div><span class="pc-status ' + statusClass(data.task_state) + '">' + esc(prettyState(data.task_state)) + '</span></header>' +
      (answer ? '<section class="pc-direct-answer"><span>教学讲解</span><p>' + esc(firstAnswer(answer)) + '</p></section>' : '<section class="pc-withheld"><strong>当前草稿未通过发布门</strong><p>' + esc(withheldReason(quality, review, verdict)) + '</p></section>') +
      '<section class="pc-mechanism-strip"><div class="pc-panel-head"><div><h2>' + (hasConceptRelations ? '概念与机制关系' : '任务知识定位') + '</h2><p>' + (hasConceptRelations ? '连线只使用当前任务公开的 Concept Relation' : '只列出当前检索结果绑定的知识点，不补造科学关系') + '</p></div></div>' + renderMechanism(data) + '</section>' +
      '<div class="pc-task-grid"><main>' +
      '<section class="pc-panel pc-task-resource-panel"><div class="pc-panel-head pc-resource-panel-head"><div><h2>学习资源</h2><p>由本次任务、审核结论和当前学习证据生成或分发</p></div><small>' + resources.length + ' 种形态</small></div>' + resourcesHtml + '</section>' +
      '<section class="pc-panel pc-task-evidence-panel"><div class="pc-panel-head"><div><h2>证据图示</h2><p>仅展示本任务公开的 Claim、Evidence 与 Reviewer 结果</p></div></div>' + renderTaskEvidenceVisual(data) + '</section>' +
      '<section class="pc-continue-learning"><div><h2>继续学习</h2></div><div><input id="pcFollowUp" type="text" placeholder="对本问题继续追问…"><button class="pc-primary" id="pcFollowUpSend">发送 ' + icon('arrow') + '</button></div></section>' +
      '</main><aside class="pc-task-rail">' +
      '<section class="pc-panel"><div class="pc-panel-head"><h2>协同过程</h2><small>4 Agent</small></div><div class="pc-agent-step-list">' + steps + '</div><button class="pc-link-button" data-pc-route="agents-chain">查看完整协同轨迹 →</button></section>' +
      '<section class="pc-panel pc-review-summary"><div class="pc-panel-head"><h2>科学审核</h2><span class="pc-status ' + statusClass(verdict) + '">' + esc(prettyState(verdict)) + '</span></div><p>' + esc(review.reason || review.message || review.summary || '未提供公开审核说明。') + '</p><dl><div><dt>发布状态</dt><dd>' + esc(releaseLabel) + '</dd></div><div><dt>修订/挑战</dt><dd>' + challengeCount + ' 项</dd></div></dl></section>' +
      '<section class="pc-panel"><div class="pc-panel-head"><h2>证据链</h2><small>' + publicClaims(data).length + ' 项结论</small></div><div class="pc-evidence-compact">' + evidenceRows(data, 3) + '</div><button class="pc-link-button" data-pc-route="kb">查看全部证据 →</button></section>' +
      '</aside></div><footer class="pc-task-footer"><span>任务 ' + esc(shortId(data.task_id)) + '</span><button class="pc-link-button" id="pcNewTask">发起新任务</button></footer></article>';

    normalizeScientificText(container);
    if (internals().bindResourceActions) internals().bindResourceActions(container, resourceData, question, queryInput);
    bindRoutes(container);
    var follow = container.querySelector('#pcFollowUp');
    var send = container.querySelector('#pcFollowUpSend');
    function submitFollowUp() {
      var value = String(follow && follow.value || '').trim();
      if (!value) return;
      if (queryInput) queryInput.value = value;
      var ask = d.getElementById('queryAsk');
      if (ask) ask.click();
    }
    if (send) send.addEventListener('click', submitFollowUp);
    if (follow) follow.addEventListener('keydown', function (event) { if (event.key === 'Enter') submitFollowUp(); });
    var newTask = container.querySelector('#pcNewTask');
    if (newTask) newTask.addEventListener('click', function () { if (window.sv) window.sv('query'); });
  }
  function eventTime(event) { return Number(event.timestamp || event.time || event.ts || 0); }
  function renderFlowDiagram(data) {
    var flow = array(data.flow_events);
    if (!flow.length) return '<div class="pc-empty-state">当前任务没有公开 flow_events。</div>';
    var main = flow.filter(function (item) { return !/CHALLENGE|REVISION|RE_RETRIEVAL/.test(String(item.step || '')); }).slice(0, 7);
    var loops = flow.filter(function (item) { return /CHALLENGE|REVISION|RE_RETRIEVAL/.test(String(item.step || '')); });
    return '<div class="pc-runtime-flow"><div class="pc-runtime-main">' + main.map(function (item, index) {
      return '<article class="pc-flow-node" title="' + attr(String(item.step || '')) + '"><span>' + String(index + 1).padStart(2, '0') + '</span><strong>' + esc(eventLabel(item.step)) + '</strong><small>' + esc(producerLabel(item.agent)) + '</small></article>' + (index < main.length - 1 ? '<div class="pc-flow-arrow">' + icon('arrow') + '</div>' : '');
    }).join('') + '</div>' + (loops.length ? '<div class="pc-runtime-loop"><span>' + icon('alert') + '</span><div><strong>审核回流</strong><p>' + loops.map(function (item) { return esc(eventLabel(item.step) + '：' + (item.label || item.detail || '已记录')) }).join(' · ') + '</p></div></div>' : '') + '</div>';
  }
  function renderCollaboration(container) {
    container.innerHTML = pageLoading('读取真实协同轨迹'); bindShell();
    loadTruth(true).then(function (ctx) {
      var data = ctx.taskData;
      if (!data) {
        container.innerHTML = '<article class="pc-page pc-empty-page"><h1>协同分析</h1><p>尚无任务协同事实。先完成一次核心任务。</p><button class="pc-primary" data-pc-route="query">发起任务</button></article>'; bindRoutes(container); return;
      }
      var trace = array(data.agent_trace), lines = array(data.collab_lines), taskEvents = array(data.task_events), flow = array(data.flow_events);
      var byAgent = {};
      trace.forEach(function (item) { if (!byAgent[item.agent_id]) byAgent[item.agent_id] = []; byAgent[item.agent_id].push(item); });
      var challenges = lines.filter(function (line) { return /CHALLENGE|REVISION|RE_RETRIEVAL/.test(String(line.label || '')); });
      var subtaskCount = lines.filter(function (line) { return line.label === 'SUBTASK_READY'; }).length;
      var evidenceUpdates = lines.filter(function (line) { return line.label === 'EVIDENCE_RETRIEVED'; }).length;
      var eventRows = taskEvents.length ? taskEvents.slice(0, 30).map(function (event) {
        return '<tr><td>' + formatTime(eventTime(event)) + '</td><td title="' + attr(event.producer || event.actor || event.agent_id || '') + '">' + esc(producerLabel(event.producer || event.actor || event.agent_id || '系统')) + '</td><td title="' + attr(event.event_type || event.type || '') + '">' + esc(eventLabel(event.event_type || event.type || 'EVENT')) + '</td><td>' + esc(String(event.detail || event.reason || event.message || event.outcome || '').slice(0, 160)) + '</td></tr>';
      }).join('') : flow.map(function (event) { return '<tr><td>序号 ' + esc(event.seq) + '</td><td>' + esc(producerLabel(event.agent)) + '</td><td>' + esc(eventLabel(event.step)) + '</td><td>' + esc(event.detail || event.label || '') + '</td></tr>'; }).join('');
      var contributions = agentOrder.map(function (agentId) {
        var entries = byAgent[agentId] || [];
        return '<article class="' + (entries.length ? 'has-data' : '') + '"><header><span>' + icon(agentId.indexOf('review') >= 0 ? 'review' : agentId.indexOf('generation') >= 0 ? 'knowledge' : agentId.indexOf('guidance') >= 0 ? 'growth' : 'agents') + '</span><div><strong>' + esc(agentLabels[agentId]) + '</strong><small>' + esc(agentRoles[agentId]) + '</small></div></header><p>' + esc(contributionSummary(agentId, entries, data)) + '</p><footer>公开贡献 ' + entries.length + ' 项</footer></article>';
      }).join('');
      container.innerHTML = '<article class="pc-page pc-collab-page"><header class="pc-page-title"><div><h1>协同分析</h1><p>' + esc(ctx.question || data.task_id) + '</p></div><span class="pc-status ' + statusClass(data.task_state) + '">' + esc(prettyState(data.task_state)) + '</span></header>' +
        '<div class="pc-collab-grid"><main><section class="pc-panel pc-flow-panel"><div class="pc-panel-head"><div><h2>本次任务运行路径</h2><p>节点和回流均来自当前 task_id 的公开事件</p></div><small>' + esc(shortId(data.task_id)) + '</small></div>' + renderFlowDiagram(data) + '</section>' +
        '<section class="pc-agent-contributions">' + contributions + '</section>' +
        '<section class="pc-panel pc-event-table"><div class="pc-panel-head"><div><h2>任务事件时间线</h2></div><small>' + (taskEvents.length || flow.length) + ' 条</small></div><div class="pc-table-wrap"><table><thead><tr><th>时间 / 顺序</th><th>生产者</th><th>事件</th><th>公开详情</th></tr></thead><tbody>' + (eventRows || '<tr><td colspan="4" class="pc-table-empty">没有公开事件。</td></tr>') + '</tbody></table></div></section></main>' +
        '<aside><section class="pc-panel"><div class="pc-panel-head"><h2>本次协同结果</h2><span class="pc-status is-ok">已记录</span></div><div class="pc-result-metrics"><div><strong>' + Object.keys(byAgent).length + '</strong><span>参与Agent</span></div><div><strong>' + subtaskCount + '</strong><span>子任务</span></div><div><strong>' + challenges.length + '</strong><span>挑战/回流</span></div><div><strong>' + evidenceUpdates + '</strong><span>证据更新</span></div></div></section>' +
        (challenges.length ? '<section class="pc-panel"><div class="pc-panel-head"><h2>冲突与处理</h2><small>' + challenges.length + '</small></div><ol class="pc-challenge-list">' + challenges.map(function (line) { var step = object(array(line.steps)[0]); return '<li><span>' + esc(eventLabel(line.label)) + '</span><p>' + esc(step.output || '已记录回流事件') + '</p></li>'; }).join('') + '</ol></section>' : '') +
        '<section class="pc-panel pc-review-summary"><div class="pc-panel-head"><h2>最终审核</h2><span class="pc-status ' + statusClass(object(data.review).verdict) + '">' + esc(prettyState(object(data.review).verdict || object(data.review).status)) + '</span></div><p>' + esc(object(data.review).reason || object(data.review).message || '未提供公开审核说明。') + '</p></section></aside></div></article>';
      normalizeScientificText(container);
      bindRoutes(container);
    });
  }

  function conceptGraphModel(data, question) {
    var context = object(data.knowledge_context);
    var nodes = array(context.nodes).slice(0, 10), edges = array(context.edges);
    if (nodes.length) return { nodes: nodes, edges: edges, mode: 'concept_relation' };

    /* Never invent scientific relations when the public Concept subgraph is
       absent. Project only the task -> KP -> source lineage already present on
       public retrieval sources. */
    var taskId = 'task:' + String(data.task_id || 'current');
    var projected = [{ concept_id: taskId, name: String(question || data.query || '当前任务'), role: 'TARGET', learner_state: 'TASK' }];
    var projectedEdges = [], seen = {};
    seen[taskId] = true;
    array(data.sources).slice(0, 5).forEach(function (source, sourceIndex) {
      var sourceLabel = source.title || source.document_title || source.document_id || source.source || source.chunk_id || ('来源 ' + (sourceIndex + 1));
      var sourceId = 'source:' + String(source.document_id || source.source_id || source.chunk_id || sourceLabel);
      var kpNames = array(source.kp_names);
      if (!kpNames.length && source.kp_name) kpNames = [source.kp_name];
      kpNames.slice(0, 3).forEach(function (name) {
        var kpId = 'kp:' + String(name);
        if (!seen[kpId] && projected.length < 10) {
          projected.push({ concept_id: kpId, name: String(name), role: 'LEARNING_PROJECTION', learner_state: 'UNKNOWN' });
          seen[kpId] = true;
          projectedEdges.push({ source: taskId, target: kpId, relation_type: 'retrieved_for' });
        }
        if (!seen[sourceId] && projected.length < 10) {
          projected.push({ concept_id: sourceId, name: String(sourceLabel), role: 'SOURCE', learner_state: 'PUBLIC_SOURCE' });
          seen[sourceId] = true;
        }
        if (seen[kpId] && seen[sourceId]) projectedEdges.push({ source: kpId, target: sourceId, relation_type: 'evidenced_by' });
      });
      if (!kpNames.length && !seen[sourceId] && projected.length < 10) {
        projected.push({ concept_id: sourceId, name: String(sourceLabel), role: 'SOURCE', learner_state: 'PUBLIC_SOURCE' });
        seen[sourceId] = true;
        projectedEdges.push({ source: taskId, target: sourceId, relation_type: 'retrieved_from' });
      }
    });
    return { nodes: projected.length > 1 ? projected : [], edges: projectedEdges, mode: 'evidence_projection' };
  }
  function renderInteractiveConceptGraph(model) {
    var nodes = model.nodes, edges = model.edges;
    if (!nodes.length) return internals().renderConceptGraph ? internals().renderConceptGraph({ knowledge_context: {}, sources: [] }) : '<div class="pc-empty-state">没有公开Concept节点。</div>';
    var targetIndex = nodes.findIndex(function (node) { return String(node.role).toUpperCase() === 'TARGET'; });
    if (targetIndex < 0) targetIndex = 0;
    var center = nodes[targetIndex], others = nodes.filter(function (_, index) { return index !== targetIndex; });
    var positions = {}, cx = 370, cy = 205, rx = 260, ry = 145;
    positions[String(center.concept_id)] = { x: cx, y: cy };
    others.forEach(function (node, index) { var angle = (Math.PI * 2 * index / Math.max(1, others.length)) - Math.PI / 2; positions[String(node.concept_id)] = { x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry }; });
    var nodeById = {}; nodes.forEach(function (node) { nodeById[String(node.concept_id)] = node; });
    var edgeSvg = edges.filter(function (edge) { return positions[String(edge.source)] && positions[String(edge.target)]; }).map(function (edge) {
      var a = positions[String(edge.source)], b = positions[String(edge.target)];
      return '<g><line x1="' + a.x + '" y1="' + a.y + '" x2="' + b.x + '" y2="' + b.y + '" marker-end="url(#pcConceptArrow)"></line><text x="' + ((a.x+b.x)/2) + '" y="' + (((a.y+b.y)/2)-6) + '">' + esc(edge.relation_type || 'related') + '</text></g>';
    }).join('');
    var nodeSvg = nodes.map(function (node) {
      var point = positions[String(node.concept_id)], isTarget = node === center;
      var label = String(node.name || node.concept_id); if (label.length > 13) label = label.slice(0, 13) + '…';
      return '<g class="pc-kg-node role-' + attr(String(node.role || 'concept').toLowerCase()) + ' ' + (isTarget ? 'is-target' : '') + '" data-pc-concept="' + attr(node.concept_id) + '"><rect x="' + (point.x-76) + '" y="' + (point.y-30) + '" width="152" height="60" rx="12"></rect><text x="' + point.x + '" y="' + (point.y-3) + '">' + esc(label) + '</text><text class="pc-node-state" x="' + point.x + '" y="' + (point.y+16) + '">' + esc(prettyState(node.learner_state || node.role)) + '</text></g>';
    }).join('');
    return '<svg class="pc-concept-graph" viewBox="0 0 740 410" role="img" aria-label="当前任务Concept关系图"><defs><marker id="pcConceptArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 10 5 0 10z"></path></marker></defs>' + edgeSvg + nodeSvg + '</svg>';
  }
  function conceptInspector(node, model) {
    if (!node) return '<div class="pc-empty-state">选择图中的Concept查看公开属性。</div>';
    var related = model.edges.filter(function (edge) { return edge.source === node.concept_id || edge.target === node.concept_id; });
    return '<div class="pc-concept-inspector"><header><h3>' + esc(node.name || node.concept_id) + '</h3><span>' + esc(node.role || 'CONCEPT') + '</span></header><dl><div><dt>Concept ID</dt><dd>' + esc(node.concept_id) + '</dd></div><div><dt>学习者状态</dt><dd>' + esc(prettyState(node.learner_state)) + '</dd></div><div><dt>公开关系</dt><dd>' + related.length + ' 条</dd></div></dl>' + (related.length ? '<ul>' + related.map(function (edge) { return '<li>' + esc(edge.source) + ' <strong>' + esc(edge.relation_type) + '</strong> ' + esc(edge.target) + '</li>'; }).join('') + '</ul>' : '<p>当前公开子图没有与该节点相连的关系。</p>') + '</div>';
  }
  function renderKnowledge(container) {
    container.innerHTML = pageLoading('读取Concept、证据与审核映射'); bindShell();
    loadTruth(true).then(function (ctx) {
      var data = ctx.taskData;
      if (!data) { container.innerHTML = '<article class="pc-page pc-empty-page"><h1>知识证据</h1><p>完成一次核心任务后，Concept、Evidence和Reviewer映射会出现在这里。</p><button class="pc-primary" data-pc-route="query">发起任务</button></article>'; bindRoutes(container); return; }
      var model = conceptGraphModel(data, ctx.question), claims = publicClaims(data), sources = array(data.sources), review = object(data.review);
      var selected = model.nodes.filter(function (node) { return String(node.role).toUpperCase() === 'TARGET'; })[0] || model.nodes[0];
      var graphTitle = model.mode === 'concept_relation' ? '概念关系图' : '任务证据关系图';
      var graphDescription = model.mode === 'concept_relation' ? '节点与科学关系来自当前任务公开Knowledge Context' : '当前任务、KP与来源连接来自公开检索结果；不冒充科学因果关系';
      var pathLabel = model.mode === 'concept_relation' ? '当前知识路径' : '当前任务关联';
      var claimHtml = claims.length ? claims.map(function (claim, index) {
        var evidence = array(claim.evidence), primary = primaryEvidence(evidence);
        return '<article class="pc-claim-card"><header><span>' + (index+1) + '</span><strong>' + esc(claim.statement || '未命名结论') + '</strong><em class="' + statusClass(claim.support_status) + '">' + esc(prettyState(claim.support_status || 'INSUFFICIENT')) + '</em></header><p>' + esc(evidenceReasonLabel(primary.reason, evidence.length)) + '</p><footer>' + evidence.map(function (item) { return '<span>' + esc(item.source || item.chunk_id || '来源') + '</span>'; }).join('') + '</footer></article>';
      }).join('') : '<div class="pc-empty-state">当前任务没有可公开的Claim–Evidence映射。</div>';
      container.innerHTML = '<article class="pc-page pc-knowledge-page"><header class="pc-page-title"><div><h1>知识证据</h1><p>当前问题：' + esc(ctx.question || data.task_id) + '</p></div><button class="pc-secondary" data-pc-route="kb-provenance">打开溯源</button></header>' +
        '<div class="pc-knowledge-path"><span>' + pathLabel + '</span>' + model.nodes.slice(0, 5).map(function (node) { return '<button data-pc-concept-jump="' + attr(node.concept_id) + '">' + esc(node.name || node.concept_id) + '</button>'; }).join(icon('arrow')) + '</div>' +
        '<div class="pc-knowledge-grid"><main><section class="pc-panel pc-graph-panel"><div class="pc-panel-head"><div><h2>' + graphTitle + '</h2><p>' + graphDescription + '</p></div><div class="pc-legend"><span class="target">任务焦点</span><span class="unknown">Concept / KP / 来源</span></div></div>' + renderInteractiveConceptGraph(model) + '</section>' +
        '<section class="pc-panel pc-task-evidence-panel"><div class="pc-panel-head"><div><h2>当前任务证据链</h2><p>只使用本次任务公开的 Claim、Evidence 与 Reviewer 状态</p></div></div>' + renderTaskEvidenceVisual(data) + '</section></main>' +
        '<aside><section class="pc-panel" id="pcConceptInspector">' + conceptInspector(selected, model) + '</section><section class="pc-panel"><div class="pc-panel-head"><h2>主张与证据映射</h2><small>' + claims.length + ' 项</small></div><div class="pc-claim-list">' + claimHtml + '</div></section><section class="pc-panel pc-coverage"><div class="pc-panel-head"><h2>证据覆盖度</h2></div><div><p><strong>' + claims.filter(function (claim) { return claim.support_status === 'SUPPORTS'; }).length + '</strong><span>直接支持</span></p><p><strong>' + claims.filter(function (claim) { return claim.support_status !== 'SUPPORTS'; }).length + '</strong><span>支持不足/候选</span></p><p><strong>' + sources.length + '</strong><span>公开来源</span></p></div><span class="pc-status ' + statusClass(review.verdict || review.status) + '">Reviewer ' + esc(prettyState(review.verdict || review.status)) + '</span></section></aside></div>' +
        '<section class="pc-panel pc-knowledge-search"><div><h2>围绕问题检索知识</h2><p>结果仅作为候选片段，不自动声明支持结论。</p></div><div><input id="pcKnowledgeSearchInput" type="search" placeholder="输入概念、机制或材料"><button class="pc-primary" id="pcKnowledgeSearchButton">检索</button></div><div id="pcKnowledgeSearchResult"></div></section></article>';
      normalizeScientificText(container);
      bindRoutes(container);
      function selectConcept(id) {
        var node = model.nodes.filter(function (item) { return String(item.concept_id) === String(id); })[0];
        var inspector = container.querySelector('#pcConceptInspector'); if (inspector) inspector.innerHTML = conceptInspector(node, model);
        container.querySelectorAll('[data-pc-concept]').forEach(function (element) { element.classList.toggle('is-selected', element.getAttribute('data-pc-concept') === String(id)); });
      }
      container.querySelectorAll('[data-pc-concept],[data-pc-concept-jump]').forEach(function (element) { element.addEventListener('click', function () { selectConcept(element.getAttribute('data-pc-concept') || element.getAttribute('data-pc-concept-jump')); }); });
      if (selected) selectConcept(selected.concept_id);
      var searchButton = container.querySelector('#pcKnowledgeSearchButton');
      if (searchButton) searchButton.addEventListener('click', function () {
        var input = container.querySelector('#pcKnowledgeSearchInput'), result = container.querySelector('#pcKnowledgeSearchResult');
        var query = String(input && input.value || '').trim(); if (!query || !result) return;
        result.innerHTML = '<div class="pc-loading-inline"><span class="spinner"></span>检索中</div>';
        apiReq('POST', '/l3/retrieve/keyword', { query: query, top_k: 5 }).then(function (payload) {
          var items = array(payload && (payload.items || payload.results || payload.chunks) || payload);
          result.innerHTML = items.length ? '<div class="pc-search-results">' + items.map(function (item, index) { return '<article><span>' + (index+1) + '</span><div><strong>' + esc(item.title || item.source || item.chunk_id || '检索片段') + '</strong><p>' + esc(String(item.content || item.text || '').slice(0, 220)) + '</p></div></article>'; }).join('') + '</div>' : '<div class="pc-empty-state">当前知识库没有返回相关片段。</div>';
        }).catch(function (error) { result.innerHTML = '<div class="pc-empty-state">' + esc(error.message || '检索失败') + '</div>'; });
      });
    });
  }

  function growthPathModel(ctx) {
    var report = object(object(ctx.match).report), path = object(report.learning_path);
    var nodes = array(path.nodes), edges = array(path.edges), source = 'R06 Concept Relation';
    if (!nodes.length) {
      var profileMastery = object(object(ctx.profile).kp_mastery);
      var sufficiency = object(report.evidence_sufficiency);
      var hasLearnerEvidence = Object.keys(profileMastery).length > 0 ||
        Number(sufficiency.answer_record_count || 0) > 0 ||
        Number(sufficiency.modelled_kp_count || 0) > 0;
      if (!hasLearnerEvidence) {
        return {
          nodes: [],
          edges: [],
          source: '尚无真实作答，不生成个人学习路径'
        };
      }
      var kpPath = array(ctx.match.learning_path).slice(0, 8);
      nodes = kpPath.map(function (item) { return { concept_id: item.kp_id, name: item.name, learner_state: item.mastery > 0 ? 'LEARNING' : 'UNKNOWN', role: item.ready ? 'READY' : 'PREREQUISITE' }; });
      var allowed = {}; nodes.forEach(function (node) { allowed[node.concept_id] = true; });
      kpPath.forEach(function (item) { array(item.prerequisites).forEach(function (pre) { if (allowed[pre]) edges.push({ source: pre, target: item.kp_id, relation_type: 'prerequisite_of' }); }); });
      source = '已编写KP先修关系';
    }
    return { nodes: nodes.slice(0, 10), edges: edges, source: source };
  }
  function renderGrowthPath(model) {
    if (!model.nodes.length) return '<div class="pc-empty-state pc-path-empty"><strong>等待真实诊断</strong><p>' + esc(model.source || '没有可公开的个人学习路径。') + '</p></div>';
    var width = 900, height = 190, gap = (width - 110) / Math.max(1, model.nodes.length - 1), positions = {};
    model.nodes.forEach(function (node, index) { positions[node.concept_id] = { x: 55 + index * gap, y: 82 }; });
    var edges = model.edges.filter(function (edge) { return positions[edge.source] && positions[edge.target]; }).map(function (edge) { var a=positions[edge.source],b=positions[edge.target]; return '<line x1="' + a.x + '" y1="' + a.y + '" x2="' + b.x + '" y2="' + b.y + '" marker-end="url(#pcGrowthArrow)"></line>'; }).join('');
    var nodes = model.nodes.map(function (node) { var p=positions[node.concept_id], state=String(node.learner_state || 'UNKNOWN').toLowerCase(), label=String(node.name || node.concept_id); if(label.length>10)label=label.slice(0,10)+'…'; return '<g class="state-' + attr(state) + '"><circle cx="' + p.x + '" cy="' + p.y + '" r="27"></circle><text x="' + p.x + '" y="' + (p.y+4) + '">' + (state === 'mastered' ? '✓' : state === 'learning' ? '↗' : '?') + '</text><text class="pc-growth-label" x="' + p.x + '" y="' + (p.y+52) + '">' + esc(label) + '</text></g>'; }).join('');
    return '<svg class="pc-growth-path" viewBox="0 0 ' + width + ' ' + height + '" role="img" aria-label="真实学习路径"><defs><marker id="pcGrowthArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 10 5 0 10z"></path></marker></defs>' + edges + nodes + '</svg><p class="pc-source-note">路径来源：' + esc(model.source) + '</p>';
  }
  function renderDifficulty(match) {
    var report = object(match.report), difficulty = object(report.resource_difficulty_match);
    if (!Object.keys(difficulty).length) difficulty = object(match.difficulty_match);
    var bands = array(difficulty.bands), pos = difficulty.learner_position;
    var w=600,h=235,left=46,right=18,top=22,bottom=44;
    var max=Math.max.apply(null,bands.map(function(b){return Number(b.question_count||0);}).concat([1]));
    var points=bands.map(function(b,i){var x=left+(i/(Math.max(1,bands.length-1)))*(w-left-right);var y=top+(1-Number(b.question_count||0)/max)*(h-top-bottom);return{x:x,y:y,b:b};});
    var path=points.map(function(p,i){return(i?'L':'M')+p.x+' '+p.y;}).join(' ');
    var dots=points.map(function(p){return'<circle cx="'+p.x+'" cy="'+p.y+'" r="5"></circle><text x="'+p.x+'" y="'+(h-17)+'" text-anchor="middle">'+esc(p.b.label||p.b.difficulty)+'</text><text x="'+p.x+'" y="'+(p.y-12)+'" text-anchor="middle">'+Number(p.b.question_count||0)+'题</text>';}).join('');
    var marker=pos==null?'':('<line class="pc-learner-line" x1="'+(left+clamp(pos,0,1)*(w-left-right))+'" y1="'+top+'" x2="'+(left+clamp(pos,0,1)*(w-left-right))+'" y2="'+(h-bottom)+'"></line><text class="pc-learner-label" x="'+(left+clamp(pos,0,1)*(w-left-right))+'" y="'+(top+12)+'" text-anchor="middle">学习者位置</text>');
    return '<svg class="pc-difficulty-chart" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="真实题库难度分布与学习者位置"><path d="'+path+'"></path>'+dots+marker+'</svg><p class="pc-source-note">'+esc(difficulty.reason||'难度判断尚不可用')+' · '+esc(sourceClassLabel(difficulty.source_class||'UNKNOWN'))+'</p>';
  }
  function renderGrowth(container) {
    container.innerHTML = pageLoading('读取成长记录与学习路径'); bindShell();
    loadTruth(true).then(function (ctx) {
      var profile=object(ctx.profile), match=object(ctx.match), report=object(match.report), ws=object(ctx.workspace), findings=array(report.findings), timeline=array(report.growth_timeline), path=growthPathModel(ctx), next=object(report.next_action), resources=taskResources(ctx.taskData || {});
      var names=object(profile.kp_names), mastery=object(profile.kp_mastery), rows=[];
      Object.keys(mastery).slice(0,10).forEach(function(kp){var value=Number(mastery[kp]);rows.push({name:names[kp]||kp,value:value,type:value>=.75?'MASTERED':value>=.35?'LEARNING':'LEARNING_GAP',source:'MODEL_INFERRED'});});
      if(!rows.length)path.nodes.slice(0,8).forEach(function(node){rows.push({name:node.name||node.concept_id,value:null,type:node.learner_state||'UNKNOWN',source:'UNKNOWN'});});
      var tableRows=rows.map(function(row){return'<tr><td><strong>'+esc(row.name)+'</strong></td><td>'+(row.value==null?'—':Math.round(row.value*100)+'%')+'</td><td>'+esc(prettyState(row.type))+'</td><td>'+esc(sourceClassLabel(row.source))+'</td><td><span class="pc-status '+statusClass(row.type)+'">'+esc(prettyState(row.type))+'</span></td></tr>';}).join('');
      function resourceBasis(item) {
        if (item.resource_form === 'practice_bank_launch') return '本地已编写题库';
        if (item.resource_form === 'evidence_analysis_workbook') return '本任务已发布事实';
        if (item.review_status === 'approved') return '本任务审核结果派生';
        return sourceClassLabel(item.source_type || '来源未声明');
      }
      var resourceRows=resources.length?resources.map(function(item){return'<tr><td><strong>'+esc(item.title||item.name||item.resource_family)+'</strong><small>'+esc(resourceFormLabel(item.resource_form))+'</small></td><td>'+esc(resourceFamilyLabel(item.resource_family))+'</td><td>'+esc(difficultyLabel(item.difficulty||object(item.payload).stage_selection||'自适应'))+'</td><td>'+esc(resourceBasis(item))+'</td><td><button data-pc-route="query">打开资源</button></td></tr>';}).join(''):'<tr><td colspan="5" class="pc-table-empty">最近任务没有通过发布门的学习资源。</td></tr>';
      var misconception=findings.filter(function(item){return item.type==='MISCONCEPTION';});
      var nextText=next.reason||object(ws.current_challenge_decision).reason||'完成真实作答后再生成学习决策';
      var rawNextTarget=String(next.target||object(ws.current_focus).name||'').trim();
      var nextTarget=(!rawNextTarget||/^unknown$/i.test(rawNextTarget))?'等待真实诊断':rawNextTarget;
      var nextAction=next.type||object(ws.current_challenge_decision).decision||'DIAGNOSE_FIRST';
      container.innerHTML='<article class="pc-page pc-growth-page"><header class="pc-page-title"><div><h1>成长路径</h1></div><div class="pc-time-filter"><span>时间范围</span><button>近7天</button><button>近30天</button><button class="active">全部</button></div></header>'+
        '<div class="pc-growth-grid"><main><section class="pc-panel pc-path-panel"><div class="pc-panel-head"><div><h2>过去 · 当前 · 下一步</h2></div></div>'+renderGrowthPath(path)+'</section>'+
        '<div class="pc-growth-analysis"><section class="pc-panel"><div class="pc-panel-head"><h2>知识状态</h2><small>'+rows.length+' 项</small></div><div class="pc-table-wrap"><table><thead><tr><th>概念 / KP</th><th>模型值</th><th>判断</th><th>来源</th><th>状态</th></tr></thead><tbody>'+tableRows+'</tbody></table></div></section><section class="pc-panel"><div class="pc-panel-head"><div><h2>资源匹配与难度</h2><p>题量来自真实已编写题库</p></div></div>'+renderDifficulty(match)+'</section></div>'+
        '<section class="pc-panel pc-resource-table"><div class="pc-panel-head"><div><h2>学习资源</h2><p>按当前任务、学习者证据与发布门结果分发</p></div><small>'+resources.length+' 项</small></div><div class="pc-table-wrap"><table><thead><tr><th>资源</th><th>形态</th><th>难度</th><th>真实依据</th><th></th></tr></thead><tbody>'+resourceRows+'</tbody></table></div></section></main>'+
        '<aside><section class="pc-panel pc-next-stage"><div class="pc-panel-head"><h2>下一阶段计划</h2></div><span>推荐目标</span><strong>'+esc(nextTarget)+'</strong><dl><div><dt>行动类型</dt><dd>'+esc(decisionLabel(nextAction))+'</dd></div><div><dt>决策依据</dt><dd>'+esc(nextText)+'</dd></div></dl><button class="pc-primary" data-pc-route="'+(nextAction==='PRACTICE'?'practice':'query')+'">进入学习</button></section>'+
        '<section class="pc-panel"><div class="pc-panel-head"><h2>学习反馈</h2><small>'+timeline.length+' 条</small></div>'+(timeline.length?'<ol class="pc-timeline-list">'+timeline.slice(-5).reverse().map(function(item){var outcome=learnerOutcomeLabel(item.outcome||item.detail);return'<li><span>'+formatTime(item.timestamp)+'</span><strong>'+esc(learnerEventLabel(item.event_type||'LEARNING_EVENT'))+'</strong>'+(outcome?'<p>'+esc(outcome)+'</p>':'')+'</li>';}).join('')+'</ol>':'<div class="pc-empty-state">尚无真实学习事件。</div>')+'</section>'+
        '<section class="pc-panel"><div class="pc-panel-head"><h2>误概念状态</h2><small>'+misconception.length+'</small></div>'+(misconception.length?'<ul class="pc-misconception-list">'+misconception.map(function(item){return'<li><strong>'+esc(item.label||item.reference)+'</strong><span class="pc-status is-warn">'+esc(item.status||'ACTIVE')+'</span></li>';}).join('')+'</ul>':'<div class="pc-empty-state">没有来源充分的误概念记录。</div>')+'</section></aside></div></article>';
      normalizeScientificText(container);
      bindRoutes(container);
    });
  }

  function invalidate() { cache.promise = null; cache.value = null; }
  window.DY3ProductCanvas = {
    renderOverview: renderOverview,
    renderTaskResult: renderTaskResult,
    renderCollaboration: renderCollaboration,
    renderKnowledge: renderKnowledge,
    renderGrowth: renderGrowth,
    invalidate: invalidate
  };

  d.addEventListener('sidebar-rebuilt', function () { setTimeout(bindShell, 0); });
  d.addEventListener('view-rendered', function () { setTimeout(bindShell, 0); });
  d.addEventListener('dy3-open-learner-profile', function () {
    loadTruth(true).then(function (ctx) {
      openLearnerProfileDialog(ctx, function () {
        var content = d.getElementById('content');
        if (content && content.querySelector('.pc-overview-page')) renderOverview(content);
      });
    });
  });
  if (d.readyState === 'loading') d.addEventListener('DOMContentLoaded', bindShell);
  else bindShell();
})();
