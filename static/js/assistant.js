/* 智能问答页：RAG 对话交互 */

const chatLog = $('#chatLog');
const chatForm = $('#chatForm');
const chatInput = $('#chatQuestion');

function appendMessage(role, html, sources) {
  const msg = document.createElement('div');
  msg.className = `msg msg-${role}`;
  const avatar = role === 'user' ? '我' : '医';
  msg.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="answer">${html}</div>
      <div class="sources"></div>
    </div>`;
  chatLog.appendChild(msg);
  if (sources && sources.length) {
    renderSources($('.sources', msg), sources);
  }
  chatLog.scrollTop = chatLog.scrollHeight;
  return msg;
}

async function ask(question) {
  if (!question.trim()) return;
  appendMessage('user', escapeHtml(question));
  const thinking = appendMessage('bot', '<span class="muted">正在检索知识库…</span>');
  chatInput.value = '';
  chatInput.disabled = true;

  try {
    const result = await fetchJSON('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ question }),
    });
    const data = result.data;
    const modeNote = `<p class="muted" style="margin-top:10px;font-size:.78rem;">生成模式：${escapeHtml(data.mode)}</p>`;
    thinking.querySelector('.answer').innerHTML = escapeHtml(data.answer) + modeNote;
    renderSources($('.sources', thinking), data.sources);
  } catch (error) {
    thinking.querySelector('.answer').innerHTML =
      `<span style="color:var(--danger)">回答失败：${escapeHtml(error.message)}</span>`;
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
    chatLog.scrollTop = chatLog.scrollHeight;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  if (!chatLog) return;

  chatForm.addEventListener('submit', (event) => {
    event.preventDefault();
    ask(chatInput.value);
  });

  $$('.chip').forEach((chip) => {
    chip.addEventListener('click', () => ask(chip.dataset.question));
  });

  chatInput.focus();
});
