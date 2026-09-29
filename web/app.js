// 金融监管法规知识库 · 前端脚本
//
// 为什么不用前端框架：
//   这个产品的学习目标是 RAG 本身，不是前端工程化。
//   用原生 JS 的收益是"改完刷新就能看到"，没有 npm、没有构建、没有打包。
//   代价是代码集中在少数文件里——在这个规模下可以接受。

const STATUS_LABEL = {
  queued: { text: '排队中', cls: 'warn' },
  parsing: { text: '解析中', cls: 'run' },
  parsed: { text: '解析完成', cls: 'ok' },
  splitting: { text: '切分中', cls: 'run' },
  chunked: { text: '切分完成', cls: 'ok' },
  vectorizing: { text: '向量化中', cls: 'run' },
  indexed: { text: '已完成', cls: 'ok' },
  failed: { text: '处理失败', cls: 'err' },
};

const IN_PROGRESS = ['queued', 'parsing', 'splitting', 'vectorizing'];

let pollTimer = null;

// ---------------------------------------------------------------- 基础工具

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await readDetail(res));
  return res.json();
}

async function readDetail(res) {
  try {
    const data = await res.json();
    return data.detail || data.message || ('HTTP ' + res.status);
  } catch (e) {
    return 'HTTP ' + res.status;
  }
}

function esc(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function fmtSize(bytes) {
  if (!bytes && bytes !== 0) return '—';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function statusBadge(status) {
  const item = STATUS_LABEL[status] || { text: status, cls: '' };
  return `<span class="badge ${item.cls}">${esc(item.text)}</span>`;
}

function progressBar(item) {
  const cls = item.status === 'failed' ? 'fail' : (item.status === 'indexed' ? 'done' : '');
  const value = item.status === 'failed' ? 100 : item.progress;
  return `<div class="bar ${cls}"><i style="width:${value}%"></i></div>
          <div style="font-size:12px;color:var(--muted);margin-top:4px">${item.progress}%</div>`;
}

function showMsg(el, text, kind) {
  el.className = 'msg show ' + (kind || 'ok');
  el.textContent = text;
}

// ---------------------------------------------------------------- 模态框

function openModal(html) {
  document.getElementById('modal').innerHTML = html;
  document.getElementById('mask').classList.add('show');
}

function closeModal() {
  document.getElementById('mask').classList.remove('show');
}

// ---------------------------------------------------------------- 视图：总览

// 图表用的几个颜色。刻意只用一个蓝色系加两个状态色，
// 因为工作台的颜色应该用来表达"正常/警告/异常"，而不是用来装饰。
const PALETTE = {
  primary: '#3b82f6',
  ok: '#22c55e',
  warn: '#f59e0b',
  err: '#ef4444',
  muted: '#475569',
};

function barList(items, unit = '') {
  // 横向条形而不是饼图：比较大小更准，而且纯 CSS 就够，不用引图表库。
  const max = Math.max(...items.map((i) => i.value), 1);
  return `<div class="bars">${items.map((item) => `
    <div class="bar-row">
      <div class="bar-label" title="${esc(item.label)}">${esc(item.label)}</div>
      <div class="bar-track"><i style="width:${Math.max((item.value / max) * 100, 1.5).toFixed(1)}%;background:${item.color || PALETTE.primary}"></i></div>
      <div class="bar-value">${item.value}${unit}</div>
    </div>`).join('')}</div>`;
}

function donut(segments, centerLabel, centerUnit) {
  // 内联 SVG 画环形图：零依赖，且能跟着主题色走。
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  let offset = 0;
  const arcs = segments.filter((s) => s.value > 0).map((segment) => {
    const length = total ? (segment.value / total) * circumference : 0;
    const arc = `<circle r="${radius}" cx="70" cy="70" fill="none" stroke="${segment.color}"
      stroke-width="17" stroke-dasharray="${length} ${circumference - length}"
      stroke-dashoffset="${-offset}" transform="rotate(-90 70 70)"></circle>`;
    offset += length;
    return arc;
  }).join('');
  const legend = segments.map((s) => `
    <div><span class="swatch" style="background:${s.color}"></span>${esc(s.label)}
      <span class="v">${s.value}${total ? ' ｜ ' + (s.value / total * 100).toFixed(0) + '%' : ''}</span></div>`).join('');
  return `<div class="donut-wrap">
    <svg viewBox="0 0 140 140" width="140" height="140">
      <circle r="${radius}" cx="70" cy="70" fill="none" stroke="#1d2530" stroke-width="17"></circle>
      ${arcs}
      <text x="70" y="66" text-anchor="middle" fill="#e6edf3" font-size="20" font-weight="600">${total}</text>
      <text x="70" y="86" text-anchor="middle" fill="#8b98a5" font-size="11">${esc(centerUnit || '')}</text>
    </svg>
    <div class="legend">${legend}</div>
  </div>`;
}

function pct(part, whole) {
  return whole ? (part / whole * 100).toFixed(0) : '0';
}

function renderDashboard(s) {
  const kb = s.knowledge_base;
  const rt = s.retrieval;
  const qa = s.qa;

  document.getElementById('kpi').innerHTML = `
    <div class="kpi"><div class="k-label">文档</div><div class="k-value">${kb.documents_total}</div>
      <div class="k-note">${kb.chunks_total} 个切片</div></div>
    <div class="kpi"><div class="k-label">检索次数</div><div class="k-value">${rt.total}</div>
      <div class="k-note">中位耗时 ${rt.latency.p50} ms</div></div>
    <div class="kpi"><div class="k-label">问答次数</div><div class="k-value">${qa.total}</div>
      <div class="k-note">中位耗时 ${qa.latency.p50} ms</div></div>
    <div class="kpi"><div class="k-label">拒答</div><div class="k-value">${qa.refused}</div>
      <div class="k-note">占 ${pct(qa.refused, qa.total)}%</div></div>
    <div class="kpi"><div class="k-label">检索故障</div>
      <div class="k-value" style="color:${qa.retrieval_failed ? PALETTE.err : PALETTE.ok}">${qa.retrieval_failed}</div>
      <div class="k-note">与"没有依据"分开统计</div></div>`;

  const docBars = kb.documents.map((d) => ({
    label: d.filename.replace(/\.(pdf|PDF|docx|md|txt)$/, ''),
    value: d.chunks,
  }));
  const splitterBars = Object.entries(kb.chunks_by_splitter).map(([name, value]) => ({
    label: name === 'semantic' ? '结构感知切分' : (name === 'unstructured' ? '按长度切分' : name),
    value,
    color: name === 'semantic' ? PALETTE.primary : PALETTE.muted,
  }));
  const stageBars = Object.entries(rt.stage_median).map(([stage, ms]) => ({
    label: { vector: '向量路', bm25: '关键词路', rerank: '重排' }[stage] || stage,
    value: ms,
    color: stage === 'rerank' ? PALETTE.warn : PALETTE.primary,
  }));
  const sourceBars = Object.entries(rt.sources).map(([name, value]) => ({
    label: { vector: '向量命中', bm25: '关键词命中' }[name] || name,
    value,
    color: name === 'bm25' ? PALETTE.warn : PALETTE.primary,
  }));
  const conclusionColors = {
    '违反': PALETTE.err,
    '不违反': PALETTE.ok,
    '无法判断': PALETTE.warn,
    '说明': PALETTE.primary,
  };
  const conclusionSegments = Object.entries(qa.by_conclusion).map(([name, value]) => ({
    label: name,
    value,
    color: conclusionColors[name] || PALETTE.muted,
  }));

  document.getElementById('panels').innerHTML = `
    <div class="panel">
      <h4>知识库构成</h4>
      <div class="hint">每份文档切成多少片。切片粒度直接决定检索能拿到多准的证据。</div>
      ${barList(docBars, ' 片')}
    </div>
    <div class="panel">
      <h4>切分策略分布</h4>
      <div class="hint">结构感知切分是默认策略；按长度切只作为对照和兜底。</div>
      ${barList(splitterBars, ' 片')}
    </div>
    <div class="panel">
      <h4>检索耗时</h4>
      <div class="hint">用分位数而不是平均值——平均值会被个别冷启动拉高，会让人误以为系统普遍很慢。</div>
      ${barList([
        { label: '中位 p50', value: rt.latency.p50, color: PALETTE.ok },
        { label: 'p90', value: rt.latency.p90, color: PALETTE.warn },
        { label: '最慢一次', value: rt.latency.max, color: PALETTE.err },
      ], ' ms')}
    </div>
    <div class="panel">
      <h4>耗时花在哪一段</h4>
      <div class="hint">各阶段的中位耗时。向量化要调外部接口，通常是最大的一块。</div>
      ${barList(stageBars, ' ms')}
    </div>
    <div class="panel">
      <h4>召回来源</h4>
      <div class="hint">向量路和关键词路各贡献了多少条命中。中文提问时关键词路接近 0——那是关键词检索跨不了语言，不是故障。</div>
      ${barList(sourceBars, ' 条')}
    </div>
    <div class="panel">
      <h4>问答结论分布</h4>
      <div class="hint">"无法判断"占比较高，是因为评测集里边界题占了多数，不等于真实使用分布。</div>
      ${donut(conclusionSegments, qa.total, '条问答')}
    </div>`;
}

async function renderOverview(box) {
  box.innerHTML = `
    <h2>工作台总览</h2>
    <div class="sub">这一屏是系统的当前状态：知识库构成、检索与问答的运行情况、以及各依赖是否就绪。</div>
    <div class="kpi-row" id="kpi"></div>
    <div class="dash-grid" id="panels"></div>
    <h3>依赖状态</h3>
    <div class="sub">依赖没配齐不影响服务启动，这里会直接告诉你还差什么。</div>
    <div class="cards" id="deps"></div>
    <div id="usage"></div>
    <h3>当前可调参数</h3>
    <div class="sub">后续做对照实验时要拧的旋钮，改 .env 重启即可生效。</div>
    <table>
      <thead><tr><th style="width:70px">分类</th><th style="width:270px">配置项</th><th style="width:110px">当前值</th><th>影响</th></tr></thead>
      <tbody id="knobs"></tbody>
    </table>
    <div class="tip">
      接口文档 <a class="link" href="/api/v1/docs" target="_blank">/api/v1/docs</a>
      ｜ 健康检查 <a class="link" href="/api/v1/system/health" target="_blank">/api/v1/system/health</a>
    </div>`;

  const health = await apiGet('/api/v1/system/health');
  document.getElementById('deps').innerHTML = health.dependencies.map((d) => {
    const cls = d.configured ? 'ok' : (d.required_now ? 'err' : '');
    const text = d.configured
      ? '已配置'
      : (d.required_now ? '未配置 · 当前阶段必需' : '未配置 · ' + d.active_from + '启用');
    return `<div class="card">
      <div class="name">${esc(d.label)}</div>
      <div><span class="badge ${cls}">${esc(text)}</span></div>
      <div class="note">${esc(d.note || '')}</div>
    </div>`;
  }).join('');

  const cfg = await apiGet('/api/v1/system/config');
  document.getElementById('knobs').innerHTML = cfg.knobs.map((k) => `
    <tr>
      <td>${esc(k.group)}</td>
      <td class="mono">${esc(k.name)}</td>
      <td>${esc(k.value)}</td>
      <td style="color:var(--muted)">${esc(k.note || '')}</td>
    </tr>`).join('');

  // 仪表盘：一次请求拿全部聚合数据，避免五六个接口各返回一半、页面来回闪
  try {
    const stats = await apiGet('/api/v1/stats');
    renderDashboard(stats);
  } catch (err) {
    document.getElementById('kpi').innerHTML =
      `<div class="kpi"><div class="k-label">统计加载失败</div><div class="k-note">${esc(err.message)}</div></div>`;
  }

  // 用量单独取一次，失败不影响上面的内容
  try {
    const usage = await apiGet('/api/v1/usage');
    const limitCls = usage.limits_enabled ? 'ok' : 'err';
    const limitText = usage.limits_enabled ? '限额已生效' : '⚠️ 限额已关闭';
    document.getElementById('usage').innerHTML = `
      <h3>今日用量 <span style="color:var(--muted);font-weight:400">${esc(usage.usage_date)}</span></h3>
      <div class="cards">
        <div class="card">
          <div class="name">问答次数</div>
          <div style="font-size:20px;font-weight:600">${usage.qa_count}</div>
          <div class="note">全局上限 ${usage.daily_qa_limit_global}，剩余 ${usage.remaining_global}</div>
        </div>
        <div class="card">
          <div class="name">你自己已问</div>
          <div style="font-size:20px;font-weight:600">${usage.your_ip_qa_count}</div>
          <div class="note">单 IP 上限 ${usage.daily_qa_limit_per_ip} 次/天</div>
        </div>
        <div class="card">
          <div class="name">被拦截</div>
          <div style="font-size:20px;font-weight:600">${usage.blocked_count}</div>
          <div class="note">限频 ${usage.rate_limit_per_ip_per_min} 次/分钟</div>
        </div>
        <div class="card">
          <div class="name">限额状态</div>
          <div><span class="badge ${limitCls}">${limitText}</span></div>
          <div class="note">上线前必须确认是"已生效"</div>
        </div>
      </div>`;
  } catch (err) {
    document.getElementById('usage').innerHTML = '';
  }
}

// ---------------------------------------------------------------- 视图：上传

async function renderUpload(box) {
  box.innerHTML = `
    <h2>上传入库</h2>
    <div class="sub">
      上传后接口立刻返回，真正的解析在后台跑。你可以直接看到 queued → parsing → parsed 的推进，
      而不是转圈等几十秒后报"失败"。
    </div>
    <div class="dropzone" id="drop">
      <strong>拖文件到这里，或点击选择</strong>
      <div>支持 pdf / docx / md / txt</div>
    </div>
    <input type="file" id="file" style="display:none" />
    <div class="row">
      <label>知识库：<input id="kb" value="regulations" style="background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px" /></label>
      <label>切分策略：<select id="splitter" style="background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px"></select></label>
      <button class="btn primary" id="submit" disabled>开始上传</button>
      <span id="picked" style="color:var(--muted)">还没有选择文件</span>
    </div>
    <div class="msg" id="msg"></div>
    <div class="tip">
      <b>三个细节值得注意</b>：<br />
      1. 同一个文件重复上传不会产生第二份——用内容指纹做幂等；<br />
      2. 上传同名文件时系统<strong>不会自动覆盖</strong>，因为在"新版本"和"另一份同名文档"之间它判断不了，会弹窗让你决定；<br />
      3. 处理失败时文件留在服务器上，可以直接重试，不用重新上传。
    </div>`;

  const drop = document.getElementById('drop');
  const input = document.getElementById('file');
  const submit = document.getElementById('submit');
  const picked = document.getElementById('picked');
  const msg = document.getElementById('msg');
  let selected = null;

  // 切分策略下拉框从后端取，避免前后端各写一份选项、以后对不上
  try {
    const options = await apiGet('/api/v1/documents/splitters/options');
    document.getElementById('splitter').innerHTML = options
      .map((o) => `<option value="${esc(o.name)}">${esc(o.name)}</option>`)
      .join('');
  } catch (err) {
    document.getElementById('splitter').innerHTML = '<option value="auto">auto</option>';
  }

  function pick(file) {
    selected = file;
    picked.textContent = file ? `${file.name}（${fmtSize(file.size)}）` : '还没有选择文件';
    submit.disabled = !file;
  }

  drop.onclick = () => input.click();
  input.onchange = () => pick(input.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    if (e.dataTransfer.files.length) pick(e.dataTransfer.files[0]);
  };

  submit.onclick = async () => {
    if (!selected) return;
    submit.disabled = true;
    const kb = document.getElementById('kb').value.trim() || 'default';
    try {
      const list = await apiGet('/api/v1/documents');
      const sameName = list.items.filter((d) => d.filename === selected.name);
      const splitter = document.getElementById('splitter').value || 'auto';
      if (sameName.length > 0) {
        askConflict(sameName, selected, kb, splitter, () => upload(selected, kb, 'keep_both', splitter, msg, submit));
      } else {
        await upload(selected, kb, 'keep_both', splitter, msg, submit);
      }
    } catch (err) {
      showMsg(msg, '检查同名文件时出错：' + err.message, 'err');
      submit.disabled = false;
    }
  };
}

function askConflict(existing, file, kb, splitter, onKeepBoth) {
  const first = existing[0];
  openModal(`
    <h4>知识库里已经有一份同名文件</h4>
    <p>知识库「${esc(kb)}」里已经有一份 <b>${esc(first.filename)}</b></p>
    <p>当前状态：${esc(first.status)} ｜ 指纹：${esc(first.hash_preview)}</p>
    <p>两份同时留在库里，提问时可能检索到旧版本的内容，而且看不出来哪份是新的。</p>
    <ul>
      <li><b>覆盖旧版本</b>：删掉同名旧文档，只保留这次上传的</li>
      <li><b>保留两份</b>：两份都在（适合确实是两份不同文档的情况）</li>
    </ul>
    <div class="actions">
      <button class="btn" onclick="closeModal()">取消</button>
      <button class="btn" id="keep">保留两份</button>
      <button class="btn primary" id="replace">覆盖旧版本</button>
    </div>`);
  document.getElementById('keep').onclick = () => { closeModal(); onKeepBoth(); };
  document.getElementById('replace').onclick = async () => {
    closeModal();
    const msg = document.getElementById('msg');
    const submit = document.getElementById('submit');
    if (submit) submit.disabled = true;
    await upload(file, kb, 'replace', splitter, msg, submit);
  };
}

async function upload(file, kb, onConflict, splitter, msgEl, submitBtn) {
  showMsg(msgEl, '正在上传…', 'run');
  const form = new FormData();
  form.append('file', file);
  form.append('knowledge_base', kb);
  form.append('on_conflict', onConflict);
  form.append('splitter', splitter || 'auto');
  try {
    const res = await fetch('/api/v1/documents/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await readDetail(res));
    const data = await res.json();
    showMsg(msgEl, data.message, data.duplicated ? 'run' : 'ok');
    setTimeout(() => { location.hash = '#documents'; }, 900);
  } catch (err) {
    showMsg(msgEl, '上传失败：' + err.message, 'err');
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

// ---------------------------------------------------------------- 视图：文档

async function renderDocuments(box) {
  box.innerHTML = `
    <h2>文档管理</h2>
    <div class="sub">处理中的文档会自动刷新，全部完成后停止——不会一直空转。</div>
    <div id="listBox"></div>`;
  await refreshDocuments();
}

async function refreshDocuments() {
  const listBox = document.getElementById('listBox');
  if (!listBox) return;
  let list;
  try {
    list = await apiGet('/api/v1/documents');
  } catch (err) {
    listBox.innerHTML = `<div class="empty">读取文档列表失败：${esc(err.message)}</div>`;
    return;
  }

  if (list.items.length === 0) {
    listBox.innerHTML = '<div class="empty">知识库还是空的。去「上传入库」放一份法规文件进来。</div>';
  } else {
    listBox.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>文件名</th><th style="width:110px">状态</th><th style="width:90px">切片数</th>
            <th style="width:180px">解析结果</th><th style="width:110px">更新于</th><th style="width:120px">操作</th>
          </tr>
        </thead>
        <tbody>${list.items.map(rowHtml).join('')}</tbody>
      </table>`;
  }

  const hasRunning = list.items.some((d) => IN_PROGRESS.includes(d.status));
  if (hasRunning && !pollTimer) {
    pollTimer = setInterval(refreshDocuments, 3000);
  } else if (!hasRunning && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function rowHtml(item) {
  const detail = item.summary || '—';
  const canRetry = ['failed', 'queued', 'parsing', 'splitting', 'vectorizing'].includes(item.status);
  const actions = `
    ${item.chunk_count > 0 ? `<button class="btn small" onclick="location.hash='#chunks?doc=${item.id}'">看切片</button>` : ''}
    ${canRetry ? `<button class="btn small" onclick="retryDoc('${item.id}')">重试</button>` : ''}
    <button class="btn small danger" onclick="deleteDoc('${item.id}', '${esc(item.filename)}')">删除</button>`;
  return `
    <tr>
      <td>
        <div>${esc(item.filename)}</div>
        <div style="font-size:12px;color:var(--muted)">
          ${esc(item.file_type)} · ${fmtSize(item.file_size)} · 指纹 ${esc(item.hash_preview || '—')}
        </div>
      </td>
      <td>${statusBadge(item.status)}</td>
      <td>
        <div style="font-weight:600">${item.chunk_count}</div>
        <div style="font-size:12px;color:var(--muted)">原进度 ${item.progress}%</div>
      </td>
      <td style="font-size:12px;color:var(--muted)" title="${esc(detail)}">
        ${esc(detail.length > 60 ? detail.slice(0, 60) + '…' : detail)}
      </td>
      <td style="font-size:12px;color:var(--muted)">${fmtTime(item.updated_at)}</td>
      <td>${actions}</td>
    </tr>`;
}

async function retryDoc(id) {
  try {
    const res = await fetch(`/api/v1/documents/${id}/retry`, { method: 'POST' });
    if (!res.ok) throw new Error(await readDetail(res));
    await refreshDocuments();
  } catch (err) {
    alert('重试失败：' + err.message);
  }
}

async function deleteDoc(id, name) {
  if (!confirm(`确定删除「${name}」？\n\n记录和服务器上的源文件都会删掉，这个操作不可撤销。`)) return;
  try {
    const res = await fetch(`/api/v1/documents/${id}`, { method: 'DELETE' });
    if (!res.ok && res.status !== 204) throw new Error(await readDetail(res));
    await refreshDocuments();
  } catch (err) {
    alert('删除失败：' + err.message);
  }
}

// ---------------------------------------------------------------- 视图：切片

async function renderChunks(box, params) {
  const all = await apiGet('/api/v1/documents');
  const ready = all.items.filter((d) => d.chunk_count > 0);

  if (ready.length === 0) {
    box.innerHTML = `
      <h2>切片管理</h2>
      <div class="empty">还没有已切分的文档。先去「上传入库」放一份材料进来。</div>`;
    return;
  }

  const wanted = params.get('doc');
  const current = ready.find((d) => d.id === wanted) || ready[0];

  box.innerHTML = `
    <h2>切片管理</h2>
    <div class="sub">
      这是整个系统里最该盯着看的一页——检索的粒度、引用的精度、答案的依据，全都从切片来。
    </div>
    <div class="row" style="margin-top:0">
      <label>文档：
        <select id="docPick" style="background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px">
          ${ready.map((d) => `<option value="${d.id}" ${d.id === current.id ? 'selected' : ''}>${esc(d.filename)}（${d.chunk_count} 片）</option>`).join('')}
        </select>
      </label>
      <label>切分策略：
        <select id="splitterPick" style="background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px">
          <option value="auto">auto（自动判断）</option>
          <option value="semantic">semantic（结构感知）</option>
          <option value="unstructured">unstructured（按长度）</option>
        </select>
      </label>
      <button class="btn" id="rechunk">重新切分</button>
    </div>
    <div class="msg" id="chunkMsg"></div>
    <div id="stats"></div>
    <h3>切片明细 <span style="color:var(--muted);font-weight:400" id="chunkTotal"></span></h3>
    <div id="chunkList"></div>`;

  document.getElementById('docPick').onchange = (e) => {
    location.hash = '#chunks?doc=' + e.target.value;
  };
  document.getElementById('rechunk').onclick = async () => {
    const splitter = document.getElementById('splitterPick').value;
    const msg = document.getElementById('chunkMsg');
    if (!confirm(`用「${splitter}」重新切分「${current.filename}」？\n\n原有切片会被覆盖。`)) return;
    showMsg(msg, '正在重新切分…', 'run');
    try {
      const form = new FormData();
      form.append('splitter', splitter);
      const res = await fetch(`/api/v1/documents/${current.id}/rechunk`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(await readDetail(res));
      const data = await res.json();
      showMsg(msg, data.message, 'ok');
      setTimeout(() => location.reload(), 1500);
    } catch (err) {
      showMsg(msg, '重新切分失败：' + err.message, 'err');
    }
  };

  const chunking = (current.parse_report && current.parse_report.chunking) || {};
  const ratio = ((chunking.mid_word_ratio || 0) * 100).toFixed(1);
  const ratioCls = (chunking.mid_word_ratio || 0) > 0.05 ? 'err' : 'ok';
  document.getElementById('stats').innerHTML = `
    <h3>切分质量</h3>
    <div class="cards">
      <div class="card"><div class="name">切片数</div><div style="font-size:20px;font-weight:600">${chunking.chunk_count || 0}</div></div>
      <div class="card"><div class="name">平均长度</div><div style="font-size:20px;font-weight:600">${chunking.avg_length || 0}</div><div class="note">目标 500 字</div></div>
      <div class="card"><div class="name">最长切片</div><div style="font-size:20px;font-weight:600">${chunking.max_length || 0}</div><div class="note">语义完整优先于长度整齐</div></div>
      <div class="card">
        <div class="name">残句率</div>
        <div><span class="badge ${ratioCls}" style="font-size:16px">${ratio}%</span></div>
        <div class="note">从单词中间开始的切片占比，${chunking.mid_word_start_count || 0} 片</div>
      </div>
    </div>
    <div class="tip">
      实际使用的策略分布：<code>${esc(JSON.stringify(chunking.splitter_usage || {}))}</code><br />
      <b>残句率是这一页最该看的数字</b>：它高说明模型读到的常常是半句话。
      想直观感受差别，把上面的策略切换成 unstructured 重新切一次，再回来看这个数字。
    </div>`;

  const chunks = await apiGet(`/api/v1/documents/${current.id}/chunks?limit=200`);
  document.getElementById('chunkTotal').textContent = `（共 ${chunks.total} 片，下面显示前 ${chunks.items.length} 片）`;
  document.getElementById('chunkList').innerHTML = `
    <table>
      <thead>
        <tr>
          <th style="width:56px">#</th>
          <th style="width:150px">来源</th>
          <th style="width:80px">长度</th>
          <th style="width:90px">策略</th>
          <th>内容</th>
          <th style="width:70px">检索</th>
        </tr>
      </thead>
      <tbody>${chunks.items.map(chunkRow).join('')}</tbody>
    </table>`;
}

function chunkRow(item) {
  const source = [
    item.page_number ? `第 ${item.page_number} 页` : '',
    item.section_title || '',
  ].filter(Boolean).join(' · ') || '—';
  const text = item.content.length > 260 ? item.content.slice(0, 260) + '…' : item.content;
  return `
    <tr>
      <td class="mono">${item.chunk_index}</td>
      <td style="font-size:12px;color:var(--muted)">${esc(source)}</td>
      <td>${item.content.length}</td>
      <td style="font-size:12px;color:var(--muted)">${esc(item.splitter_name || '—')}</td>
      <td style="font-size:13px;white-space:pre-wrap">${esc(text)}</td>
      <td>
        <button class="btn small" onclick="toggleChunk('${item.id}', ${item.enabled ? 'false' : 'true'})">
          ${item.enabled ? '停用' : '启用'}
        </button>
      </td>
    </tr>`;
}

async function toggleChunk(chunkId, enabled) {
  try {
    const res = await fetch(`/api/v1/chunks/${chunkId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!res.ok) throw new Error(await readDetail(res));
    location.reload();
  } catch (err) {
    alert('操作失败：' + err.message);
  }
}


// ---------------------------------------------------------------- 路由

async function renderRetrieval(box) {
  box.innerHTML = `
    <h2>检索调试台</h2>
    <div class="sub">
      这是整个系统里最该先建、也最常被跳过的一页。
      没有它，你无法判断一条错答是"没检索到"还是"模型判断错了"——
      而那两种情况的修法完全不同。
    </div>
    <div class="row" style="margin-top:0">
      <input id="q" placeholder="输入一个问题，例如：收到客户礼物需要披露吗"
             style="flex:1;min-width:320px;background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:6px" />
      <label>返回条数
        <select id="topk" style="background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:7px 10px;border-radius:6px">
          <option>3</option><option selected>5</option><option>10</option><option>20</option>
        </select>
      </label>
      <button class="btn primary" id="go">检索</button>
    </div>
    <div class="msg" id="rmsg"></div>
    <div class="tip">
      <b>两路并行召回，再融合、再重排。</b>
      向量路擅长"意思相近"，关键词路擅长"字面命中"——法规材料里全是
      <code>第二十九条</code>、《证券法》这类编号和名称，而它们恰恰是向量最不敏感的。
      每条结果会标出它是哪一路召回的，以及重排前后名次变了多少。
      <br /><br />
      <b>点这一页最值得看的地方</b>：同一份材料，两路召回的结果往往不一样。
      关键词路能精确命中"第二十九条"这种编号，但它分不清"**是**第二十九条"
      和"**提到**第二十九条"；向量路能理解意思，却容易把编号当成噪声。
      两条路都错的地方，才是真正需要改的地方。
    </div>
    <div id="rstats"></div>
    <h3>命中结果 <span style="color:var(--muted);font-weight:400" id="rstat"></span></h3>
    <div id="rlist"><div class="empty">输入问题后点「检索」</div></div>
    <h3>检索历史 <span style="color:var(--muted);font-weight:400">最近 20 次，点一条可以回看当时的召回过程</span></h3>
    <div id="rlogs"></div>`;

  const input = document.getElementById('q');
  const go = document.getElementById('go');
  const run = async () => {
    const query = input.value.trim();
    if (!query) return;
    showMsg(document.getElementById('rmsg'), '检索中…（会真实调用向量化接口）', 'run');
    try {
      const res = await fetch('/api/v1/retrieval/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: Number(document.getElementById('topk').value) }),
      });
      if (!res.ok) throw new Error(await readDetail(res));
      const data = await res.json();
      document.getElementById('rmsg').className = 'msg';
      document.getElementById('rstat').textContent = `（返回前 ${(data.items || []).length} 条）`;
      renderRetrievalStats(data);
      renderRetrievalHits(data.items);
      loadRetrievalLogs();
    } catch (err) {
      showMsg(document.getElementById('rmsg'), '检索失败：' + err.message, 'err');
    }
  };

  go.onclick = run;
  input.onkeydown = (e) => { if (e.key === 'Enter') run(); };
  input.focus();
  loadRetrievalLogs();
}

function renderRetrievalStats(data) {
  const t = data.timings_ms || {};
  const items = data.items || [];
  const rerankMoved = items.filter(
    (h) => h.rank_before_rerank && h.rank_after_rerank && h.rank_before_rerank !== h.rank_after_rerank,
  ).length;
  document.getElementById('rstats').innerHTML = `
    <div class="cards" style="margin-top:18px">
      <div class="card">
        <div class="name">向量路召回</div>
        <div style="font-size:20px;font-weight:600">${data.vector_hit_count}</div>
        <div class="note">耗时 ${t.vector ?? '—'} ms</div>
      </div>
      <div class="card">
        <div class="name">关键词路召回</div>
        <div style="font-size:20px;font-weight:600">${data.bm25_hit_count}</div>
        <div class="note">耗时 ${t.bm25 ?? '—'} ms</div>
      </div>
      <div class="card">
        <div class="name">融合后候选</div>
        <div style="font-size:20px;font-weight:600">${data.fused_count}</div>
        <div class="note">候选池 ${data.candidate_k} 条</div>
      </div>
      <div class="card">
        <div class="name">重排</div>
        <div><span class="badge ${data.rerank_enabled ? 'ok' : ''}">${data.rerank_enabled ? '已启用' : '已关闭'}</span></div>
        <div class="note">耗时 ${t.rerank ?? '—'} ms ｜ 名次有变 ${rerankMoved} 条</div>
      </div>
    </div>
    ${data.error ? `<div class="msg show err" style="margin-top:12px">降级记录：${esc(data.error)}</div>` : ''}
    <div class="tip">日志 ID：<code>${esc(data.log_id || '（未记录）')}</code>　
      两路各自的召回条数、融合与重排耗时都在上面——<b>这些数字是回答"为什么是这几条"的完整证据。</b>
    </div>`;
}

function sourceBadges(hit) {
  const sources = hit.retrieval_sources || (hit.retrieval_source ? [hit.retrieval_source] : []);
  const names = { vector: '向量', bm25: '关键词', hybrid: '双路' };
  return sources
    .map((s) => `<span class="badge ${s === 'bm25' ? 'warn' : (s === 'hybrid' ? 'ok' : 'run')}">${names[s] || esc(s)}</span>`)
    .join(' ');
}

function renderRetrievalHits(items) {
  const box = document.getElementById('rlist');
  if (!items || items.length === 0) {
    box.innerHTML = '<div class="empty">没有命中任何切片。知识库可能是空的，或者还没做向量化。</div>';
    return;
  }
  box.innerHTML = items.map((hit) => {
    const origin = [
      hit.filename || '未知文件',
      hit.page_number ? `第 ${hit.page_number} 页` : '',
      hit.section_title || '',
    ].filter(Boolean).join(' · ');

    const ranks = [
      hit.rank_vector ? `向量第${hit.rank_vector}` : '',
      hit.rank_bm25 ? `关键词第${hit.rank_bm25}` : '',
      hit.rank_fused ? `融合第${hit.rank_fused}` : '',
    ].filter(Boolean).join(' ｜ ');
    const moved = hit.rank_before_rerank && hit.rank_after_rerank
      ? `　重排 ${hit.rank_before_rerank} → ${hit.rank_after_rerank}`
      : '';

    return `
      <div class="card" style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;gap:14px;margin-bottom:8px;flex-wrap:wrap">
          <div style="font-size:12px;color:var(--muted)">
            ${sourceBadges(hit)} ｜ ${esc(origin)} ｜ 策略 ${esc(hit.splitter_name || '—')}
          </div>
          <div style="font-size:12px;color:var(--accent);white-space:nowrap">
            得分 ${Number(hit.score || 0).toFixed(4)}
          </div>
        </div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:8px">${esc(ranks + moved)}</div>
        <div style="white-space:pre-wrap;font-size:13px">${esc((hit.text || '').slice(0, 400))}${(hit.text || '').length > 400 ? '…' : ''}</div>
      </div>`;
  }).join('');
}

async function loadRetrievalLogs() {
  const box = document.getElementById('rlogs');
  if (!box) return;
  try {
    const data = await apiGet('/api/v1/retrieval/logs?limit=20');
    if (!data.items.length) {
      box.innerHTML = '<div class="empty">还没有检索记录。检索一次就会出现在这里。</div>';
      return;
    }
    box.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>问题</th><th style="width:90px">两路召回</th>
            <th style="width:120px">耗时</th><th style="width:150px">时间</th>
            <th style="width:70px">操作</th>
          </tr>
        </thead>
        <tbody>${data.items.map((log) => `
          <tr>
            <td style="font-size:13px">${esc(log.query)}</td>
            <td style="font-size:12px;color:var(--muted)">向量 ${log.vector_hit_count} / 词 ${log.bm25_hit_count}</td>
            <td style="font-size:12px;color:var(--muted)">
              向量 ${log.timings_ms?.vector ?? '—'} · 词 ${log.timings_ms?.bm25 ?? '—'} · 重排 ${log.timings_ms?.rerank ?? '—'}
            </td>
            <td style="font-size:12px;color:var(--muted)">${fmtTime(log.created_at)}</td>
            <td><button class="btn small" onclick="replayLog('${log.id}')">回看</button></td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch (err) {
    box.innerHTML = `<div class="empty">读取检索历史失败：${esc(err.message)}</div>`;
  }
}

async function replayLog(logId) {
  try {
    const log = await apiGet('/api/v1/retrieval/logs/' + logId);
    document.getElementById('q').value = log.query;
    document.getElementById('rstat').textContent = `（回看：当时命中 ${log.returned_count} 条）`;
    renderRetrievalStats({
      vector_hit_count: log.vector_hit_count,
      bm25_hit_count: log.bm25_hit_count,
      fused_count: log.fused_count,
      candidate_k: log.candidate_k,
      rerank_enabled: log.rerank_enabled,
      timings_ms: log.timings_ms,
      error: log.error,
      log_id: log.id,
      items: log.items,
    });
    renderRetrievalHits(log.items);
  } catch (err) {
    alert('回看失败：' + err.message);
  }
}


const VIEWS = {
  overview: renderOverview,
  upload: renderUpload,
  documents: renderDocuments,
  chunks: renderChunks,
  retrieval: renderRetrieval,
  qa: renderQa,
};

// ---------------------------------------------------------------- 视图：问答

const CONCLUSION_STYLE = {
  '违反': 'err',
  '不违反': 'ok',
  '无法判断': 'warn',
  '说明': 'run',
};

// 会话 ID 存在浏览器本地。
// 为什么不让后端生成：会话是"这个人正在连着问"的概念，
// 前端本来就知道边界在哪（点"新会话"就是边界），后端不需要猜。
const SESSION_KEY = 'vibeRagSessionId';

function currentSessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = 's-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function newSessionId() {
  const id = 's-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

async function renderQa(box) {
  box.innerHTML = `
    <h2>问答</h2>
    <div class="sub">
      回答会给出结论、依据哪一条、以及可以点开核对的原文。
      资料里没有依据时，系统会明确说没有——<b>在合规场景里，说错的代价远大于不回答。</b>
    </div>
    <div class="row" style="margin-top:0">
      <span style="font-size:13px;color:var(--muted)">当前会话：<code id="sessId">—</code></span>
      <button class="btn small" id="newSess">新会话</button>
      <span style="font-size:12px;color:var(--muted)">
        追问会带上这个会话的上文；<b>检索前会先把追问改写成能独立检索的问题</b>，否则"那如果…"这类问法搜不到东西
      </span>
    </div>
    <div class="row" style="margin-top:0">
      <input id="qaQ" placeholder="例如：向普通投资者销售高风险产品要履行哪些义务"
             style="flex:1;min-width:320px;background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:6px" />
      <label>证据条数
        <select id="qaTopk" style="background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:7px 10px;border-radius:6px">
          <option>3</option><option selected>5</option><option>8</option>
        </select>
      </label>
      <label>拒答阈值
        <input id="qaThreshold" type="number" step="0.05" min="0" max="1" value="0"
               title="检索最高分低于它就直接拒答，不给模型回答的机会"
               style="width:80px;background:var(--panel-2);border:1px solid var(--border);color:var(--text);padding:7px 10px;border-radius:6px" />
      </label>
      <button class="btn primary" id="qaGo">提问</button>
    </div>
    <div class="msg" id="qaMsg"></div>
    <div class="tip">
      <b>拒答阈值是硬闸门，不是提示词里的许愿。</b>
      0 = 不拒答；把它调到 0.3 左右再问一次，就能看到"分数不够就明确说不知道"的行为。
      阈值需要靠评测校准，页面右上角那个输入框就是用来做这件事的。
    </div>
    <div id="qaAnswer"></div>
    <h3><span id="qaLogHeading">问答记录</span>
      <span style="color:var(--muted);font-weight:400">当前会话的每一轮都在这里；「新会话」会断开上下文</span></h3>
    <div id="qaLogs"></div>`;

  const input = document.getElementById('qaQ');
  const go = document.getElementById('qaGo');
  document.getElementById('sessId').textContent = currentSessionId().slice(0, 14);
  document.getElementById('newSess').onclick = () => {
    newSessionId();
    document.getElementById('sessId').textContent = currentSessionId().slice(0, 14);
    document.getElementById('qaAnswer').innerHTML = '';
    showMsg(document.getElementById('qaMsg'), '已开启新会话，之前的对话不再带入上下文', 'run');
    loadQaLogs();
  };
  const run = async () => {
    const question = input.value.trim();
    if (!question) return;
    go.disabled = true;
    showMsg(document.getElementById('qaMsg'), '检索并生成中…（这条链路会真实调用百炼）', 'run');
    try {
      const res = await fetch('/api/v1/qa/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          top_k: Number(document.getElementById('qaTopk').value),
          refuse_threshold: Number(document.getElementById('qaThreshold').value || 0),
          session_id: currentSessionId(),
        }),
      });
      if (!res.ok) throw new Error(await readDetail(res));
      const data = await res.json();
      document.getElementById('qaMsg').className = 'msg';
      renderQaAnswer(data);
      loadQaLogs();
    } catch (err) {
      showMsg(document.getElementById('qaMsg'), '问答失败：' + err.message, 'err');
    } finally {
      go.disabled = false;
    }
  };

  go.onclick = run;
  input.onkeydown = (e) => { if (e.key === 'Enter') run(); };
  input.focus();
  loadQaLogs();
}

function renderQaAnswer(data) {
  const box = document.getElementById('qaAnswer');
  if (data.error) {
    // 区分"系统故障"和"生成失败"：前者要提示用户重试，
    // 而且必须说清楚**这不代表知识库里没有内容**——
    // 这两种情况用户该做的事完全不同。
    const title = data.retrieval_failed ? '检索链路故障，本次没有生成回答' : '生成失败';
    box.innerHTML = `
      <h3>回答</h3>
      <div class="msg show err" style="font-size:13px;line-height:1.8">
        <b>${esc(title)}</b><br />${esc(data.error)}
      </div>
      <div class="tip">
        这是<b>系统故障</b>而不是"知识库里没有依据"。前一种应该稍后重试，
        后一种才说明资料缺失——两者不要混。检索过程的完整记录已经落库，
        可以在「检索调试台」的检索历史里回看当时发生了什么。
      </div>`;
    return;
  }

  const r = data.retrieval || {};
  const t = r.timings_ms || {};
  const style = CONCLUSION_STYLE[data.conclusion] || '';
  const head = data.conclusion
    ? `<span class="badge ${style}" style="font-size:15px;padding:3px 14px">${esc(data.conclusion)}</span>`
    : '<span class="badge warn">未得到结构化结论</span>';

  const warnings = [];
  if (data.unknown_citations && data.unknown_citations.length) {
    warnings.push(`<div class="msg show err" style="margin-top:12px">
      <b>检测到编造的引用</b>：模型引用了不存在的片段 ID ${esc(data.unknown_citations.join('、'))}。
      这已经被记录下来，不算进引用列表。</div>`);
  }
  if (data.parse_ok === false) {
    warnings.push(`<div class="msg show run" style="margin-top:12px">
      模型这次没有输出结构化结果，下面是它的原文。这条记录已被标记，评测时会单独看。</div>`);
  }
  if (data.refused) {
    warnings.push(`<div class="msg show warn" style="margin-top:12px">
      <b>这是拒答。</b>原因：${esc(data.refusal_reason || '')}</div>`);
  }

  const citations = (data.citations || []).map((c, i) => `
    <div class="card" style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:8px;flex-wrap:wrap">
        <div style="font-size:12px;color:var(--muted)">
          引用 [${i + 1}] ｜ ${esc(c.filename || '')}
          ${c.page_number ? `· 第 ${c.page_number} 页` : ''}
          ${c.section_title ? `· ${esc(c.section_title)}` : ''}
        </div>
        <div style="font-size:12px;color:var(--accent)">${c.score != null ? Number(c.score).toFixed(4) : ''}</div>
      </div>
      <div style="white-space:pre-wrap;font-size:13px">${esc((c.text || '').slice(0, 420))}${(c.text || '').length > 420 ? '…' : ''}</div>
    </div>`).join('');

  box.innerHTML = `
    <h3>回答</h3>
    <div class="card">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;flex-wrap:wrap">
        ${head}
        ${data.clause ? `<span style="font-size:13px;color:var(--muted)">条款：<b style="color:var(--text)">${esc(data.clause)}</b></span>` : ''}
        <span style="font-size:12px;color:var(--muted);margin-left:auto">耗时 ${data.latency_ms} ms ｜ 日志 ${esc((data.log_id || '').slice(0, 8))}</span>
      </div>
      <div style="white-space:pre-wrap;line-height:1.8">${esc(data.reasoning || '（无）')}</div>
      ${data.assumption ? `<div style="margin-top:12px;font-size:13px;color:var(--muted)"><b>判断前提：</b>${esc(data.assumption)}</div>` : ''}
    </div>
    ${warnings.join('')}
    ${data.rewritten && data.standalone_query ? `<div class="tip" style="border-left-color:var(--ok)">
      <b>这次做了追问改写。</b>你问的是「${esc(data.question)}」，
      实际拿去检索的是「${esc(data.standalone_query)}」。<br />
      追问通常省略主语，直接检索几乎召不回东西；所以先补全再检索。
      注意<b>回答里回应的仍然是你问的那句话</b>——改写只用于检索，不改写你的问题。
    </div>` : ''}
    <div class="tip">
      本次检索：向量路 ${r.vector_hit_count ?? '—'} 条 ｜ 关键词路 ${r.bm25_hit_count ?? '—'} 条 ｜
      融合 ${r.fused_count ?? '—'} 条 ｜ 重排${r.rerank_enabled ? '开' : '关'} ｜
      耗时 向量 ${t.vector ?? '—'} / 词 ${t.bm25 ?? '—'} / 重排 ${t.rerank ?? '—'} ms
      ${r.error ? `<br /><span style="color:var(--warn)">降级记录：${esc(r.error)}</span>` : ''}
    </div>
    <h3>引用原文 <span style="color:var(--muted);font-weight:400">用户不需要相信 AI，只需要花三秒核对</span></h3>
    ${citations || '<div class="empty">本次没有引用任何片段</div>'}`;
}

async function loadQaLogs() {
  const box = document.getElementById('qaLogs');
  if (!box) return;
  try {
    const sessionId = currentSessionId();
    // 默认看当前会话——多轮追问的意义就在于"连着看"。
    // 当前会话还是空的时候，退回看最近的全部记录。
    let data = await apiGet(`/api/v1/qa/logs?limit=20&session_id=${encodeURIComponent(sessionId)}`);
    let heading = `当前会话（${data.total} 轮）`;
    if (!data.items.length) {
      data = await apiGet('/api/v1/qa/logs?limit=20');
      heading = '最近问答记录（当前会话还没有内容）';
    }
    box.dataset.heading = heading;
    const headingEl = document.getElementById('qaLogHeading');
    if (headingEl) headingEl.textContent = heading;
    if (!data.items.length) {
      box.innerHTML = '<div class="empty">还没有问答记录。</div>';
      return;
    }
    box.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>问题</th><th style="width:110px">结论</th><th style="width:80px">引用</th>
            <th style="width:110px">未知引用</th><th style="width:80px">故障</th>
            <th style="width:80px">耗时</th><th style="width:150px">时间</th>
          </tr>
        </thead>
        <tbody>${data.items.map((log) => `
          <tr>
            <td style="font-size:13px">${esc(log.question)}</td>
            <td>${log.conclusion ? `<span class="badge ${CONCLUSION_STYLE[log.conclusion] || ''}">${esc(log.conclusion)}</span>` : '<span class="badge">—</span>'}</td>
            <td style="font-size:12px;color:var(--muted)">${(log.citations || []).length}</td>
            <td>${(log.unknown_citations || []).length
              ? `<span class="badge err">${log.unknown_citations.length} 条</span>`
              : '<span class="badge ok">无</span>'}</td>
            <td>${log.retrieval_failed
              ? '<span class="badge err">检索故障</span>'
              : '<span class="badge">—</span>'}</td>
            <td style="font-size:12px;color:var(--muted)">${log.latency_ms} ms</td>
            <td style="font-size:12px;color:var(--muted)">${fmtTime(log.created_at)}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch (err) {
    box.innerHTML = `<div class="empty">读取问答记录失败：${esc(err.message)}</div>`;
  }
}

function parseHash() {
  const raw = (location.hash || '#overview').slice(1);
  const [name, query] = raw.split('?');
  return {
    view: VIEWS[name] ? name : 'overview',
    params: new URLSearchParams(query || ''),
  };
}

async function bootConnection() {
  const dot = document.getElementById('dot');
  const text = document.getElementById('connText');
  try {
    const health = await apiGet('/api/v1/system/health');
    dot.className = 'dot ok';
    text.textContent = '后端已连接 · ' + health.app_env;
    document.getElementById('stage').textContent = health.build_stage;
  } catch (err) {
    dot.className = 'dot err';
    text.textContent = '连不上后端：' + err.message;
    document.getElementById('stage').textContent = '';
  }
}

async function route() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  const { view, params } = parseHash();
  document.querySelectorAll('#nav a').forEach((a) => {
    a.classList.toggle('active', a.dataset.view === view);
  });
  const box = document.getElementById('view');
  box.innerHTML = '<div class="empty">加载中…</div>';
  try {
    await VIEWS[view](box, params);
  } catch (err) {
    box.innerHTML = `<div class="empty">这个页面出错了：${esc(err.message)}</div>`;
  }
}

document.querySelectorAll('#nav a[data-view]').forEach((a) => {
  a.onclick = () => { location.hash = '#' + a.dataset.view; };
});
document.getElementById('mask').onclick = (e) => {
  if (e.target.id === 'mask') closeModal();
};
window.addEventListener('hashchange', route);

bootConnection();
route();
