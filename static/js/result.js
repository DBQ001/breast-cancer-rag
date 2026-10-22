/* 结果页：仪表盘动画 + 异步获取 RAG 智能解读 */

function animateGauge() {
  const progress = $('.gauge-progress');
  const counter = $('.gauge-center strong');
  if (!progress || !counter) return;

  const ratio = parseFloat(progress.dataset.progress || '0');
  const circumference = 2 * Math.PI * 78;

  requestAnimationFrame(() => {
    progress.style.strokeDashoffset = String(circumference * (1 - ratio));
  });

  const target = parseFloat(counter.dataset.count || '0');
  const duration = 1100;
  const start = performance.now();
  const tick = (now) => {
    const elapsed = Math.min(now - start, duration);
    counter.textContent = (target * elapsed / duration).toFixed(1);
    if (elapsed < duration) requestAnimationFrame(tick);
    else counter.textContent = target.toFixed(1);
  };
  requestAnimationFrame(tick);
}

async function loadInterpretation() {
  const panel = $('#ragPanel');
  const answerBox = $('#ragAnswer');
  const sourceBox = $('#ragSources');
  const modeChip = $('#ragMode');
  if (!panel) return;

  let features;
  try {
    features = JSON.parse(panel.dataset.features || '[]');
  } catch (error) {
    features = [];
  }

  try {
    const result = await fetchJSON('/api/interpret', {
      method: 'POST',
      body: JSON.stringify({ features }),
    });
    const data = result.data;
    answerBox.innerHTML = `<div class="answer">${escapeHtml(data.answer)}</div>`;
    modeChip.textContent = data.mode + (data.fallback_reason ? '（已降级）' : '');
    renderSources(sourceBox, data.sources);
  } catch (error) {
    answerBox.innerHTML = `<div class="answer">解读生成失败：${escapeHtml(error.message)}</div>`;
    modeChip.textContent = '生成失败';
    sourceBox.innerHTML = '';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  animateGauge();
  loadInterpretation();
});
