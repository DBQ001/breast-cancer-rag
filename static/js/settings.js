/* 系统设置页交互：回填、测试连接、保存 */

const settingsData = JSON.parse(($('#settingsData') || {}).textContent || '{}');

function fillForm(data) {
  $('#llmProvider').value = data.llm_provider || 'local';
  $('#llmModel').value = data.llm_model || '';
  $('#llmBaseUrl').value = data.llm_base_url || '';
  $('#llmTimeout').value = data.llm_timeout != null ? data.llm_timeout : 30;
  $('#ragTopK').value = data.rag_top_k != null ? data.rag_top_k : 4;
  $('#ragVectorWeight').value = data.rag_vector_weight != null ? data.rag_vector_weight : 0.6;
  $('#ragKeywordWeight').value = data.rag_keyword_weight != null ? data.rag_keyword_weight : 0.4;
  $('#authUsername').value = data.auth_username || '';
  $('#providerBadge').textContent = data.llm_provider || 'local';
}

function collectSettings() {
  return {
    llm_provider: $('#llmProvider').value,
    llm_model: $('#llmModel').value.trim(),
    llm_base_url: $('#llmBaseUrl').value.trim(),
    llm_timeout: $('#llmTimeout').value,
    llm_api_key: $('#llmApiKey').value.trim(),
    rag_top_k: $('#ragTopK').value,
    rag_vector_weight: $('#ragVectorWeight').value,
    rag_keyword_weight: $('#ragKeywordWeight').value,
    auth_username: $('#authUsername').value.trim(),
    new_password: $('#newPassword').value,
  };
}

function setResult(ok, text, pending = false) {
  const box = $('#testResult');
  box.className = 'test-result ' + (pending ? 'is-pending' : (ok ? 'is-ok' : 'is-fail'));
  box.textContent = text;
}

if (Object.keys(settingsData).length) fillForm(settingsData);

// 选择 openai 兼容接口时，若地址/模型为空则自动填入 DeepSeek 官方默认值，
// 避免因漏填而误发到 api.openai.com（国内会超时）
const OPENAI_DEFAULTS = {
  baseUrl: 'https://api.deepseek.com/v1',
  model: 'deepseek-flash',
};

function applyOpenAiDefaults() {
  if ($('#llmProvider').value !== 'openai') return;
  if (!$('#llmBaseUrl').value.trim()) $('#llmBaseUrl').value = OPENAI_DEFAULTS.baseUrl;
  if (!$('#llmModel').value.trim()) $('#llmModel').value = OPENAI_DEFAULTS.model;
}

$('#llmProvider').addEventListener('change', applyOpenAiDefaults);
applyOpenAiDefaults();

$('#testBtn').addEventListener('click', async () => {
  const btn = $('#testBtn');
  btn.disabled = true;
  setResult(true, '正在测试连接…', true);
  try {
    const res = await fetchJSON('/api/settings/test', {
      method: 'POST', body: JSON.stringify(collectSettings()),
    });
    const data = res.data;
    const reply = data.reply ? `（模型返回：${data.reply}）` : '';
    setResult(data.ok, data.message + reply);
  } catch (err) {
    setResult(false, err.message);
  } finally {
    btn.disabled = false;
  }
});

$('#saveBtn').addEventListener('click', async () => {
  const btn = $('#saveBtn');
  btn.disabled = true;
  setResult(true, '正在保存…', true);
  try {
    const res = await fetchJSON('/api/settings', {
      method: 'POST', body: JSON.stringify(collectSettings()),
    });
    fillForm(res.data);
    $('#llmApiKey').value = '';
    $('#newPassword').value = '';
    setResult(true, res.message || '设置已保存并立即生效');
  } catch (err) {
    setResult(false, err.message);
  } finally {
    btn.disabled = false;
  }
});
