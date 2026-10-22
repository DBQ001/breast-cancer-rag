"""全局配置：路径、特征元数据、算法参数、RAG 与 LLM 参数。

所有可调项均支持通过环境变量覆盖，便于在服务器或本地切换配置。
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 路径配置
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'Breast Cancer Data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RAG_DIR = os.path.join(BASE_DIR, 'rag')
KNOWLEDGE_DIR = os.path.join(RAG_DIR, 'documents')
VECTOR_STORE_FILE = os.path.join(MODEL_DIR, 'rag_vector_store.pkl')
MODEL_FILE = os.path.join(MODEL_DIR, 'breast_cancer_model.joblib')

# ---------------------------------------------------------------- 算法配置
# 默认使用逻辑回归（课堂讲授算法），可通过环境变量切换为随机森林
DEFAULT_ALGORITHM = os.getenv('BC_ALGORITHM', 'logistic_regression')
TEST_SIZE = float(os.getenv('BC_TEST_SIZE', '0.2'))
RANDOM_STATE = 42

# 训练集 / 测试集评估时使用的候选模型
CANDIDATE_ALGORITHMS = ('logistic_regression', 'random_forest')

# ---------------------------------------------------------------- 风险分层
# 依据恶性概率划分风险等级，(阈值上界, 等级名, 前端配色)
RISK_LEVELS = (
    (0.35, '低风险', '#22c55e'),
    (0.70, '中风险', '#f59e0b'),
    (1.01, '高风险', '#ef4444'),
)

# ---------------------------------------------------------------- 30 个特征
# 顺序与数据集 "Breast Cancer Data.csv" 第 3 列起的字段一一对应
FEATURE_LIST = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst',
]

# 中文标签（前端表单与结果页展示）
FEATURE_LABELS = {
    'radius_mean': '半径均值', 'texture_mean': '纹理均值',
    'perimeter_mean': '周长均值', 'area_mean': '面积均值',
    'smoothness_mean': '平滑度均值', 'compactness_mean': '紧凑度均值',
    'concavity_mean': '凹陷度均值', 'concave points_mean': '凹陷点均值',
    'symmetry_mean': '对称性均值', 'fractal_dimension_mean': '分形维数均值',
    'radius_se': '半径标准误', 'texture_se': '纹理标准误',
    'perimeter_se': '周长标准误', 'area_se': '面积标准误',
    'smoothness_se': '平滑度标准误', 'compactness_se': '紧凑度标准误',
    'concavity_se': '凹陷度标准误', 'concave points_se': '凹陷点标准误',
    'symmetry_se': '对称性标准误', 'fractal_dimension_se': '分形维数标准误',
    'radius_worst': '半径最差值', 'texture_worst': '纹理最差值',
    'perimeter_worst': '周长最差值', 'area_worst': '面积最差值',
    'smoothness_worst': '平滑度最差值', 'compactness_worst': '紧凑度最差值',
    'concavity_worst': '凹陷度最差值', 'concave points_worst': '凹陷点最差值',
    'symmetry_worst': '对称性最差值', 'fractal_dimension_worst': '分形维数最差值',
}

# 三组特征：均值 / 标准误 / 最差值，前端按组折叠展示
FEATURE_GROUPS = [
    ('mean', '均值特征（Mean）', FEATURE_LIST[0:10]),
    ('se', '标准误特征（Standard Error）', FEATURE_LIST[10:20]),
    ('worst', '最差值特征（Worst）', FEATURE_LIST[20:30]),
]

# 特征的中文单位/量纲描述，用于结果页解读
FEATURE_UNITS = {
    'radius': 'mm', 'texture': '灰度标准差', 'perimeter': 'mm',
    'area': 'mm²', 'smoothness': '无量纲', 'compactness': '无量纲',
    'concavity': '无量纲', 'concave points': '个数比例',
    'symmetry': '无量纲', 'fractal_dimension': '无量纲',
}

# ---------------------------------------------------------------- RAG 配置
RAG_TOP_K = int(os.getenv('RAG_TOP_K', '4'))
CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '420'))       # 单个片段目标字数
CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', '60'))  # 相邻片段重叠字数
# 检索分数由向量相似度与关键词(BM25)得分加权融合
RAG_VECTOR_WEIGHT = float(os.getenv('RAG_VECTOR_WEIGHT', '0.6'))
RAG_KEYWORD_WEIGHT = float(os.getenv('RAG_KEYWORD_WEIGHT', '0.4'))

# ---------------------------------------------------------------- LLM 配置
# 三种模式：local(离线抽取式) / openai(OpenAI 兼容接口) / dify / coze
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'local')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
# 默认指向 DeepSeek 官方 OpenAI 兼容接口；
# 留空时会使用该默认值，避免误发到 api.openai.com
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-flash')
DIFY_API_URL = os.getenv('DIFY_API_URL', 'https://api.dify.ai/v1/chat-messages')
COZE_API_URL = os.getenv('COZE_API_URL', 'https://api.coze.cn/open_api/v2/chat')
COZE_BOT_ID = os.getenv('COZE_BOT_ID', '')
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '30'))

# ---------------------------------------------------------------- 运行时设置
# 网页「设置」页保存的配置持久化到 instance/settings.json（该目录已被 .gitignore 忽略，
# 避免 API Key 等敏感信息被提交到代码仓库）
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
SETTINGS_FILE = os.path.join(INSTANCE_DIR, 'settings.json')

# ---------------------------------------------------------------- 登录配置
SECRET_KEY = os.getenv('BC_SECRET_KEY', 'luyu-zhineng-breast-cancer-rag-demo-key')
AUTH_USERNAME = os.getenv('BC_AUTH_USER', 'admin')
# 为空时使用内置默认口令（见 auth.py，教学演示用途）
AUTH_PASSWORD_HASH = os.getenv('BC_AUTH_PASSWORD_HASH', '')

# 允许在「设置」页动态修改、并在运行时立即生效的配置项及其类型
RUNTIME_SETTING_TYPES = {
    'LLM_PROVIDER': str,
    'LLM_API_KEY': str,
    'LLM_BASE_URL': str,
    'LLM_MODEL': str,
    'LLM_TIMEOUT': int,
    'RAG_TOP_K': int,
    'RAG_VECTOR_WEIGHT': float,
    'RAG_KEYWORD_WEIGHT': float,
    'AUTH_USERNAME': str,
    'AUTH_PASSWORD_HASH': str,
}


def _apply_settings(data: dict):
    """把设置字典覆盖到模块全局变量上，实现「改完即生效」。"""
    g = globals()
    for key, caster in RUNTIME_SETTING_TYPES.items():
        if key not in data or data[key] is None:
            continue
        try:
            g[key] = caster(data[key])
        except (TypeError, ValueError):
            continue


def _read_settings_file() -> dict:
    import json

    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as fp:
                return json.load(fp) or {}
    except (OSError, ValueError):
        pass
    return {}


def load_runtime_settings() -> dict:
    """读取 instance/settings.json 并覆盖同名配置，返回读到的原始字典。"""
    data = _read_settings_file()
    _apply_settings(data)
    return data


def save_runtime_settings(values: dict) -> dict:
    """把设置写入 instance/settings.json，并立即在运行时生效。"""
    import json

    os.makedirs(INSTANCE_DIR, exist_ok=True)
    current = _read_settings_file()
    for key in RUNTIME_SETTING_TYPES:
        if key in values:
            current[key] = values[key]
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as fp:
        json.dump(current, fp, ensure_ascii=False, indent=2)
    _apply_settings(current)
    return current


def runtime_settings() -> dict:
    """返回当前生效的运行时设置快照。"""
    g = globals()
    return {key: g.get(key) for key in RUNTIME_SETTING_TYPES}


# 进程启动时立即加载持久化设置
load_runtime_settings()

# ---------------------------------------------------------------- 服务配置
HOST = os.getenv('BC_HOST', '0.0.0.0')
PORT = int(os.getenv('BC_PORT', '5000'))
DEBUG = os.getenv('BC_DEBUG', 'true').lower() == 'true'

# 演示模式下允许在无模型文件时自动训练，保证开箱即用
AUTO_TRAIN_IF_MISSING = os.getenv('BC_AUTO_TRAIN', 'true').lower() == 'true'


def risk_level(probability: float):
    """根据恶性概率返回 (等级名称, 颜色)。"""
    for upper, name, color in RISK_LEVELS:
        if probability < upper:
            return name, color
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]
