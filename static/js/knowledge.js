/* 知识库页：重建索引 + 检索效果自测 */

document.addEventListener('DOMContentLoaded', () => {
  const rebuildBtn = $('#rebuildBtn');
  if (rebuildBtn) {
    rebuildBtn.addEventListener('click', async () => {
      rebuildBtn.disabled = true;
      rebuildBtn.textContent = '重建中…';
      try {
        const result = await fetchJSON('/api/knowledge/rebuild', { method: 'POST' });
        rebuildBtn.textContent = `完成（${result.data.chunk_count} 个片段）`;
      } catch (error) {
        rebuildBtn.textContent = `失败：${error.message}`;
      }
      setTimeout(() => { rebuildBtn.textContent = '重建索引'; rebuildBtn.disabled = false; }, 1800);
    });
  }

  const retrieveBtn = $('#retrieveBtn');
  const retrieveInput = $('#retrieveInput');
  const retrieveResult = $('#retrieveResult');

  async function doRetrieve() {
    const question = retrieveInput.value.trim();
    if (!question) { retrieveInput.focus(); return; }
    retrieveResult.innerHTML = '<p class="muted">检索中…</p>';
    try {
      const result = await fetchJSON('/api/retrieve', {
        method: 'POST',
        body: JSON.stringify({ question }),
      });
      if (!result.data.length) {
        retrieveResult.innerHTML = '<p class="muted">未召回任何片段</p>';
        return;
      }
      retrieveResult.innerHTML = result.data.map((item, index) => `
        <div class="retrieve-item">
          <strong>[${index + 1}] ${escapeHtml(item.doc_title)} · ${escapeHtml(item.section)}</strong>
          <div class="scores">融合 ${item.score} ｜ 向量 ${item.vector_score} ｜ 关键词 ${item.keyword_score}</div>
          <p>${escapeHtml(item.text)}</p>
        </div>`).join('');
    } catch (error) {
      retrieveResult.innerHTML = `<p class="muted" style="color:var(--danger)">检索失败：${escapeHtml(error.message)}</p>`;
    }
  }

  if (retrieveBtn) retrieveBtn.addEventListener('click', doRetrieve);
  if (retrieveInput) {
    retrieveInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); doRetrieve(); }
    });
  }
});
