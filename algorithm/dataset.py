"""数据集加载与切分。

数据集来源：开源工程 breast-cancer-detection 的 "Breast Cancer Data.csv"。
每行第 1 列为样本 id，第 2 列为 diagnosis（M=恶性 / B=良性），
第 3 列起为 30 个细胞核形态学特征，对应前端页面的 30 个输入框。
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_dataset(path: str | None = None):
    """读取 CSV，返回 (X, y)。

    X: shape=(n, 30) 的 float 数组，列顺序与 config.FEATURE_LIST 一致
    y: shape=(n,) 的 int 数组，1=恶性(Malignant)，0=良性(Benign)
    """
    path = path or config.DATA_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'数据集不存在：{path}\n'
            '请从开源工程下载 "Breast Cancer Data.csv" 并放入 data/ 目录。'
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [f for f in config.FEATURE_LIST if f not in df.columns]
    if missing:
        raise ValueError(f'数据集缺少特征列：{missing}')

    for col in config.FEATURE_LIST:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=config.FEATURE_LIST + ['diagnosis'])

    y = (df['diagnosis'].astype(str).str.strip().str.upper() == 'M').astype(int)
    X = df[config.FEATURE_LIST].astype(float)
    return X, y


def split_dataset(test_size: float | None = None, random_state: int | None = None):
    """按分层抽样切分训练集 / 测试集，返回 X_train, X_test, y_train, y_test。"""
    X, y = load_dataset()
    return train_test_split(
        X.values, y.values,
        test_size=test_size if test_size is not None else config.TEST_SIZE,
        random_state=random_state if random_state is not None else config.RANDOM_STATE,
        stratify=y.values,
    )


def dataset_statistics(path: str | None = None) -> dict:
    """数据集概览，用于前端「数据概况」与 PPT 中的数据集介绍。"""
    X, y = load_dataset(path)
    return {
        'total': int(len(X)),
        'feature_count': int(X.shape[1]),
        'malignant': int((y == 1).sum()),
        'benign': int((y == 0).sum()),
        'malignant_ratio': round(float((y == 1).mean()), 4),
        'benign_ratio': round(float((y == 0).mean()), 4),
    }


def sample_rows(label: str, count: int = 1) -> list[dict]:
    """取若干示例样本，供前端「一键填充示例数据」使用。

    label: 'malignant' | 'benign' | 'random'
    """
    df = pd.read_csv(config.DATA_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    diag = df['diagnosis'].astype(str).str.strip().str.upper()
    if label == 'malignant':
        pool = df[diag == 'M']
    elif label == 'benign':
        pool = df[diag == 'B']
    else:
        pool = df
    pool = pool.sample(n=min(count, len(pool)), random_state=None)
    rows = []
    for _, row in pool.iterrows():
        rows.append({
            'diagnosis': 'Malignant' if str(row['diagnosis']).strip().upper() == 'M' else 'Benign',
            'features': [float(row[f]) for f in config.FEATURE_LIST],
        })
    return rows


def normalize_sample(values: list[float]) -> np.ndarray:
    """把前端提交的 30 个特征值整理为模型输入数组。"""
    if len(values) != len(config.FEATURE_LIST):
        raise ValueError(f'需要 {len(config.FEATURE_LIST)} 个特征，实际收到 {len(values)} 个')
    return np.asarray(values, dtype=float).reshape(1, -1)
