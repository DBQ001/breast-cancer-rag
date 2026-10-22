"""命令行训练脚本。

用法：
    python -m algorithm.train                       # 训练默认算法（逻辑回归）
    python -m algorithm.train --algorithm random_forest
    python -m algorithm.train --compare             # 输出逻辑回归与随机森林对照结果
"""
import argparse
import json

from .models import compare_algorithms
from .predictor import ModelService
from .dataset import dataset_statistics, split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='训练乳腺癌 AI 预测模型')
    parser.add_argument('--algorithm', default=None,
                        choices=['logistic_regression', 'random_forest'],
                        help='要训练的算法，默认使用 config.DEFAULT_ALGORITHM')
    parser.add_argument('--compare', action='store_true', help='输出算法对照结果')
    args = parser.parse_args()

    print('数据集概况:', json.dumps(dataset_statistics(), ensure_ascii=False))

    service = ModelService()
    metrics = service.train(algorithm=args.algorithm)
    print(f'训练完成，算法={metrics["algorithm_label"]}')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.compare:
        X_train, X_test, y_train, y_test = split_dataset()
        print('\n算法对照结果:')
        print(json.dumps(compare_algorithms(X_train, y_train, X_test, y_test),
                         ensure_ascii=False, indent=2))

    print(f'\n模型已保存至 {service.algorithm} -> models/breast_cancer_model.joblib')


if __name__ == '__main__':
    main()
