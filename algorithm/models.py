"""模型构建与训练评估。

主算法为 **逻辑回归（Logistic Regression）**：本项目在开源工程随机森林的基础上，
按课堂讲授内容替换为逻辑回归模型，并保留随机森林作为对照实验。
所有模型均封装为 Pipeline（标准化 + 分类器），保证训练与推理使用同一套预处理。
"""
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

ALGORITHM_LABELS = {
    'logistic_regression': '逻辑回归',
    'random_forest': '随机森林',
}


def build_model(name: str = None, random_state: int = None) -> Pipeline:
    """按名称构建「标准化 + 分类器」流水线。"""
    name = name or config.DEFAULT_ALGORITHM
    rs = config.RANDOM_STATE if random_state is None else random_state

    if name == 'logistic_regression':
        clf = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
            random_state=rs,
        )
    elif name == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=rs,
        )
    else:
        raise ValueError(f'不支持的算法：{name}')

    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf),
    ])


def _evaluate(pipeline: Pipeline, X, y, fit_seconds: float) -> dict:
    """计算常用分类指标，返回可直接序列化为 JSON 的字典。"""
    predictions = pipeline.predict(X)
    # 部分模型（如逻辑回归）输出 2 列概率，统一取正类（恶性）概率
    probabilities = pipeline.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 单样本平均推理耗时（毫秒）
    t0 = time.perf_counter()
    pipeline.predict_proba(X[:1])
    per_sample_ms = (time.perf_counter() - t0) * 1000

    return {
        'accuracy': round(float(accuracy_score(y, predictions)), 4),
        'precision': round(float(precision_score(y, predictions, zero_division=0)), 4),
        'recall': round(float(recall_score(y, predictions, zero_division=0)), 4),
        'f1': round(float(f1_score(y, predictions, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y, probabilities)), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'support': int(len(y)),
        'train_seconds': round(float(fit_seconds), 4),
        'predict_ms': round(float(per_sample_ms), 4),
    }


def train_and_evaluate(name: str, X_train, y_train, X_test, y_test) -> tuple[Pipeline, dict]:
    """训练单个模型并在测试集上评估。"""
    pipeline = build_model(name)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    metrics = _evaluate(pipeline, X_test, y_test, fit_seconds)
    metrics['algorithm'] = name
    metrics['algorithm_label'] = ALGORITHM_LABELS.get(name, name)
    return pipeline, metrics


def compare_algorithms(X_train, y_train, X_test, y_test) -> dict:
    """对照实验：依次训练候选算法，便于在 PPT 中展示算法替换效果。"""
    results = {}
    for name in config.CANDIDATE_ALGORITHMS:
        _, metrics = train_and_evaluate(name, X_train, y_train, X_test, y_test)
        results[name] = metrics
    return results


def feature_contributions(pipeline: Pipeline, sample: np.ndarray, top_k: int = 6) -> list[dict]:
    """给出单个样本的特征贡献度，用于结果页「关键影响因素」展示。

    逻辑回归：贡献 = 标准化特征值 × 系数；
    随机森林：贡献 = 标准化特征值 × 特征重要度（近似）。
    """
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['clf']
    z = scaler.transform(sample)[0]

    if hasattr(clf, 'coef_'):
        weights = clf.coef_[0]
    elif hasattr(clf, 'feature_importances_'):
        weights = clf.feature_importances_
    else:
        return []

    scores = z * weights
    order = np.argsort(-np.abs(scores))[:top_k]
    return [{
        'name': config.FEATURE_LIST[i],
        'label': config.FEATURE_LABELS.get(config.FEATURE_LIST[i], config.FEATURE_LIST[i]),
        'value': round(float(sample[0][i]), 4),
        'z_score': round(float(z[i]), 4),
        'contribution': round(float(scores[i]), 4),
        # 贡献度条宽（前端直接使用，避免模板内做数值运算）
        'bar_pct': int(min(100, round(abs(float(scores[i])) / 2.5 * 100))),
        'direction': 'malignant' if scores[i] > 0 else 'benign',
    } for i in order]
