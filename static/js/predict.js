/* 首页：示例数据一键填充、清空 */

const FEATURE_ORDER = [
  'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
  'smoothness_mean', 'compactness_mean', 'concavity_mean',
  'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
  'radius_se', 'texture_se', 'perimeter_se', 'area_se',
  'smoothness_se', 'compactness_se', 'concavity_se',
  'concave points_se', 'symmetry_se', 'fractal_dimension_se',
  'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
  'smoothness_worst', 'compactness_worst', 'concavity_worst',
  'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst',
];

const LABELS = {
  malignant: '恶性（Malignant）',
  benign: '良性（Benign）',
  random: '随机',
};

async function fillSample(label) {
  const hint = $('#fillHint');
  hint.textContent = '正在获取示例数据…';
  try {
    const result = await fetchJSON(`/api/dataset/sample?label=${label}`);
    const sample = result.data;
    if (!sample) throw new Error('未获取到示例数据');
    sample.features.forEach((value, index) => {
      const input = document.getElementById(`field-${FEATURE_ORDER[index]}`);
      if (input) input.value = value;
    });
    hint.textContent = `已填充${LABELS[label]}示例数据（真实诊断：${sample.diagnosis === 'Malignant' ? '恶性' : '良性'}）`;
  } catch (error) {
    hint.textContent = `填充失败：${error.message}`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  $$('[data-fill]').forEach((button) => {
    button.addEventListener('click', () => fillSample(button.dataset.fill));
  });

  const clearBtn = $('#clearBtn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      $$('#predictForm input[type="number"]').forEach((input) => { input.value = ''; });
      $('#fillHint').textContent = '';
    });
  }

  const form = $('#predictForm');
  if (form) {
    form.addEventListener('submit', (event) => {
      const submitBtn = form.querySelector('button[type="submit"]');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '预测中…';
      }
    });
  }
});
