/* 知识库文档详情：把 Markdown 渲染为页面内容 */

document.addEventListener('DOMContentLoaded', () => {
  const source = $('#docSource');
  const target = $('#docRender');
  if (!source || !target) return;
  target.innerHTML = renderMarkdown(source.textContent);
});
