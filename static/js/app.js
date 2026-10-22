/* 公共工具：请求封装、转义、来源卡片渲染 */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.success === false) {
    throw new Error(data.message || `请求失败（HTTP ${response.status}）`);
  }
  return data;
}

function escapeHtml(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/** 渲染知识来源卡片 */
function renderSources(container, sources) {
  if (!container) return;
  if (!sources || !sources.length) {
    container.innerHTML = '';
    return;
  }
  container.innerHTML = sources.map((s) => `
    <div class="source-chip">
      <strong>[${s.index}] ${escapeHtml(s.title)} · ${escapeHtml(s.section)}</strong>
      <span class="score">融合得分 ${s.score}（向量 ${s.vector_score ?? '-'} / 关键词 ${s.keyword_score ?? '-'}）</span>
      <p class="muted">${escapeHtml(s.snippet)}</p>
    </div>
  `).join('');
}

/** 简易 Markdown 渲染（仅用于知识库文档阅读页，避免引入外部依赖） */
function renderMarkdown(markdown) {
  const lines = String(markdown || '').split('\n');
  const html = [];
  let inList = false;

  const inline = (text) => escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2">$1</a>');

  const closeList = () => { if (inList) { html.push('</ul>'); inList = false; } };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { closeList(); continue; }
    if (line.startsWith('### ')) { closeList(); html.push(`<h3>${inline(line.slice(4))}</h3>`); continue; }
    if (line.startsWith('## ')) { closeList(); html.push(`<h2>${inline(line.slice(3))}</h2>`); continue; }
    if (line.startsWith('# ')) { closeList(); html.push(`<h1>${inline(line.slice(2))}</h1>`); continue; }
    if (/^[-*]\s+/.test(line)) {
      if (!inList) { html.push('<ul>'); inList = true; }
      html.push(`<li>${inline(line.replace(/^[-*]\s+/, ''))}</li>`);
      continue;
    }
    closeList();
    html.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return html.join('');
}
