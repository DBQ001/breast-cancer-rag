"""模型服务：模型训练、持久化、单样本预测与指标缓存。

设计为单例，Flask 启动时完成一次训练/加载，后续请求复用内存中的模型，
保证预测响应速度（毫秒级）。
"""
import os
import threading
import time

import joblib
import numpy as np

import config
from .dataset import load_dataset, split_dataset
from .models import (
    ALGORITHM_LABELS, build_model, compare_algorithms,
    feature_contributions, train_and_evaluate,
)


class ModelService:
    """封装模型的加载、训练、评估与预测能力。"""

    def __init__(self, algorithm: str | None = None):
        self.algorithm = algorithm or config.DEFAULT_ALGORITHM
        self.pipeline = None
        self.metrics: dict = {}
        self.comparison: dict = {}
        self.trained_at: float | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------- 训练与加载
    def train(self, algorithm: str | None = None, save: bool = True) -> dict:
        """在完整数据集上重新训练并评估（测试集指标用于前端展示）。"""
        algorithm = algorithm or self.algorithm
        X_train, X_test, y_train, y_test = split_dataset()

        pipeline, metrics = train_and_evaluate(
            algorithm, X_train, y_train, X_test, y_test)

        # 用全量数据再拟合一次，让线上预测使用更多样本
        X, y = load_dataset()
        pipeline.fit(X.values, y.values)

        with self._lock:
            self.algorithm = algorithm
            self.pipeline = pipeline
            self.metrics = metrics
            self.trained_at = time.time()
            self.comparison = compare_algorithms(X_train, y_train, X_test, y_test)

        if save:
            self.save()
        return metrics

    def load_or_train(self) -> 'ModelService':
        """优先加载本地模型；缺失时按配置自动训练。"""
        if self._load():
            return self
        if config.AUTO_TRAIN_IF_MISSING:
            self.train()
            return self
        raise RuntimeError('未找到已训练模型，请先运行 python -m algorithm.train')

    def save(self, path: str | None = None) -> None:
        path = path or config.MODEL_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'algorithm': self.algorithm,
            'metrics': self.metrics,
            'comparison': self.comparison,
            'trained_at': self.trained_at,
        }, path)

    def _load(self, path: str | None = None) -> bool:
        path = path or config.MODEL_FILE
        if not os.path.exists(path):
            return False
        try:
            bundle = joblib.load(path)
        except Exception:
            return False
        with self._lock:
            self.pipeline = bundle.get('pipeline')
            self.algorithm = bundle.get('algorithm', self.algorithm)
            self.metrics = bundle.get('metrics', {})
            self.comparison = bundle.get('comparison', {})
            self.trained_at = bundle.get('trained_at')
        return self.pipeline is not None

    # ---------------------------------------------------------- 预测
    def predict(self, features: list[float]) -> dict:
        """对单个样本执行预测。

        features: 长度为 30 的列表，顺序同 config.FEATURE_LIST
        返回: 诊断结果、恶性概率、风险等级、耗时、关键影响因素
        """
        if self.pipeline is None:
            self.load_or_train()

        sample = np.asarray(features, dtype=float).reshape(1, -1)

        t0 = time.perf_counter()
        label = int(self.pipeline.predict(sample)[0])
        probability = float(self.pipeline.predict_proba(sample)[0][1])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        risk_name, risk_color = config.risk_level(probability)

        return {
            'output': 'Malignant' if label == 1 else 'Benign',
            'output_cn': '恶性' if label == 1 else '良性',
            'probability': round(probability, 4),          # 恶性概率
            'benign_probability': round(1 - probability, 4),
            'confidence': round(max(probability, 1 - probability), 4),
            'risk_level': risk_name,
            'risk_color': risk_color,
            'elapsed_ms': round(elapsed_ms, 3),
            'algorithm': self.algorithm,
            'algorithm_label': ALGORITHM_LABELS.get(self.algorithm, self.algorithm),
            'features': [float(v) for v in features],
            'contributions': feature_contributions(self.pipeline, sample),
        }

    # ---------------------------------------------------------- 元信息
    def info(self) -> dict:
        return {
            'algorithm': self.algorithm,
            'algorithm_label': ALGORITHM_LABELS.get(self.algorithm, self.algorithm),
            'metrics': self.metrics,
            'comparison': self.comparison,
            'trained_at': self.trained_at,
            'available_algorithms': {
                k: v for k, v in ALGORITHM_LABELS.items()
            },
        }


_service: ModelService | None = None
_service_lock = threading.Lock()


def get_model_service() -> ModelService:
    """获取全局单例模型服务。"""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ModelService().load_or_train()
    return _service


def reset_model_service() -> None:
    """切换算法后重置单例，使新模型在下次请求时生效。"""
    global _service
    with _service_lock:
        _service = None
