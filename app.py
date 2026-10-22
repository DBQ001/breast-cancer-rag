"""乳愈智能 · 乳腺癌 AI 预测系统 —— Flask 后端主程序。

功能概览：
  - 乳腺癌良恶性预测（逻辑回归为主算法，支持切换随机森林对照）
  - 预测结果风险分层与关键影响因素分析
  - 基于本地知识库的 RAG 智能问答与预测结果知识解读
  - 知识库浏览、数据集概览、模型指标展示

运行：
    python app.py
然后访问 http://127.0.0.1:5000
"""
import os
import traceback

from flask import (Flask, jsonify, redirect, render_template, request, session,
                   url_for)

import config
from algorithm import dataset_statistics, get_model_service, sample_rows
from algorithm.dataset import normalize_sample
from algorithm.predictor import reset_model_service
from auth import (current_username, hash_password, is_logged_in, PUBLIC_ENDPOINTS,
                  verify_credentials)
from llm.client import LLMClient
from rag.pipeline import get_rag_service, rebuild_index

app = Flask(__name__)
app.json.ensure_ascii = False  # 接口直接返回中文，便于调试与前端展示
app.secret_key = config.SECRET_KEY  # 登录会话签名密钥

# 启动时预热模型与知识库索引，避免首个请求等待过久
MODEL_SERVICE = get_model_service()
RAG_SERVICE = get_rag_service()


# ------------------------------------------------------------------ 统一鉴权
@app.before_request
def _require_login():
    """除登录页与静态资源外，所有页面与接口都必须先登录。"""
    if (request.endpoint or '') in PUBLIC_ENDPOINTS or is_logged_in():
        return None
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': '登录状态已失效，请重新登录'}), 401
    return redirect(url_for('login', next=request.path))


@app.context_processor
def _inject_globals():
    """模板全局变量：当前登录用户名（导航栏展示）。"""
    return {'username': current_username()}


# ------------------------------------------------------------------ 登录 / 登出
@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页：GET 渲染表单，POST 校验账号口令。"""
    if is_logged_in():
        return redirect(url_for('index'))

    error = ''
    username = ''
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        if verify_credentials(username, password):
            session['username'] = username
            target = request.form.get('next') or request.args.get('next') or url_for('index')
            if not target.startswith('/'):
                target = url_for('index')
            return redirect(target)
        error = '账号或口令不正确，请重新输入。'

    return render_template('login.html', error=error, username=username,
                           next=request.args.get('next', ''))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ------------------------------------------------------------------ 设置页与接口
def _settings_payload() -> dict:
    """给前端的设置快照：API Key 只回显是否已配置，不回传明文。"""
    s = config.runtime_settings()
    return {
        'llm_provider': s.get('LLM_PROVIDER') or 'local',
        'llm_base_url': s.get('LLM_BASE_URL') or '',
        'llm_model': s.get('LLM_MODEL') or '',
        'llm_timeout': s.get('LLM_TIMEOUT'),
        'llm_api_key_set': bool((s.get('LLM_API_KEY') or '').strip()),
        'rag_top_k': s.get('RAG_TOP_K'),
        'rag_vector_weight': s.get('RAG_VECTOR_WEIGHT'),
        'rag_keyword_weight': s.get('RAG_KEYWORD_WEIGHT'),
        'auth_username': s.get('AUTH_USERNAME') or '',
    }


@app.route('/settings')
def settings_page():
    """设置页：大模型接入、RAG 参数与账号口令。"""
    return render_template(
        'settings.html',
        active='settings',
        settings=_settings_payload(),
        username=current_username(),
        rag=RAG_SERVICE.info(),
    )


@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    return jsonify({'success': True, 'data': _settings_payload()})


@app.route('/api/settings', methods=['POST'])
def api_settings_save():
    """保存设置：只写入实际提交的字段；API Key 留空表示不修改。"""
    payload = request.get_json(silent=True) or {}
    values = {}

    provider = (payload.get('llm_provider') or '').strip().lower()
    if provider:
        if provider not in ('local', 'openai', 'dify', 'coze'):
            return jsonify({'success': False,
                            'message': f'不支持的大模型后端：{provider}'}), 400
        values['LLM_PROVIDER'] = provider

    if payload.get('llm_base_url') is not None:
        values['LLM_BASE_URL'] = str(payload['llm_base_url']).strip()
    if payload.get('llm_model') is not None:
        values['LLM_MODEL'] = str(payload['llm_model']).strip()

    api_key = str(payload.get('llm_api_key') or '').strip()
    if api_key:
        if not api_key.isascii():
            return jsonify({'success': False,
                            'message': 'API Key 含中文或全角字符，请重新复制纯英文密钥'}), 400
        values['LLM_API_KEY'] = api_key

    for field, key, caster in (('llm_timeout', 'LLM_TIMEOUT', int),
                               ('rag_top_k', 'RAG_TOP_K', int),
                               ('rag_vector_weight', 'RAG_VECTOR_WEIGHT', float),
                               ('rag_keyword_weight', 'RAG_KEYWORD_WEIGHT', float)):
        raw = payload.get(field)
        if raw in (None, ''):
            continue
        try:
            number = caster(raw)
        except (TypeError, ValueError):
            return jsonify({'success': False,
                            'message': f'参数 {field} 的取值非法'}), 400
        if number < 0:
            return jsonify({'success': False,
                            'message': f'参数 {field} 不能为负数'}), 400
        values[key] = number

    username = str(payload.get('auth_username') or '').strip()
    if username:
        values['AUTH_USERNAME'] = username

    new_password = str(payload.get('new_password') or '')
    if new_password:
        if len(new_password) < 6:
            return jsonify({'success': False, 'message': '新口令长度至少 6 位'}), 400
        values['AUTH_PASSWORD_HASH'] = hash_password(new_password)

    config.save_runtime_settings(values)
    return jsonify({'success': True, 'data': _settings_payload(),
                    'message': '设置已保存并立即生效'})


@app.route('/api/settings/test', methods=['POST'])
def api_settings_test():
    """用（含未保存的）表单参数临时调用一次大模型，验证连通性。"""
    payload = request.get_json(silent=True) or {}
    provider = (payload.get('llm_provider') or config.LLM_PROVIDER or 'local').lower()
    if provider == 'local':
        return jsonify({'success': True, 'data': {
            'ok': True, 'provider': 'local',
            'message': '当前为本地检索式模式，无需调用大模型。'}})

    # 允许用尚未保存的表单值临时测试，测试后立即还原配置
    saved = {}
    for form_key, config_key in (('llm_base_url', 'LLM_BASE_URL'),
                                 ('llm_model', 'LLM_MODEL'),
                                 ('llm_api_key', 'LLM_API_KEY')):
        value = str(payload.get(form_key) or '').strip()
        if value:
            saved[config_key] = getattr(config, config_key, None)
            setattr(config, config_key, value)
    try:
        result = LLMClient(provider=provider).chat(
            '你是接口连通性测试助手。', '请只回复两个字：正常')
    finally:
        for key, value in saved.items():
            setattr(config, key, value)

    return jsonify({'success': True, 'data': {
        'ok': result.used_llm,
        'provider': provider,
        'message': ('连接成功，模型已返回内容。' if result.used_llm
                    else f'连接失败：{result.error or "未返回内容"}'),
        'reply': result.text if result.used_llm else '',
    }})


# ------------------------------------------------------------------ 页面路由
@app.route('/')
def index():
    """首页：30 个特征输入表单 + 数据集概况。"""
    return render_template(
        'index.html',
        feature_groups=_feature_groups(),
        stats=dataset_statistics(),
        model=MODEL_SERVICE.info(),
        rag=RAG_SERVICE.info(),
        active='predict',
    )


@app.route('/assistant')
def assistant():
    """RAG 智能问答页。"""
    return render_template(
        'assistant.html',
        rag=RAG_SERVICE.info(),
        samples=PRESET_QUESTIONS,
        active='assistant',
    )


@app.route('/knowledge')
def knowledge():
    """知识库浏览页。"""
    return render_template(
        'knowledge.html',
        documents=RAG_SERVICE.info()['documents'],
        active='knowledge',
    )


@app.route('/knowledge/<path:filename>')
def knowledge_document(filename: str):
    """知识库文档详情页。"""
    from rag.loader import read_document

    document = read_document(filename)
    if document is None:
        return render_template('error.html', message=f'未找到文档：{filename}'), 404
    return render_template('document.html', document=document, active='knowledge')


@app.route('/predict', methods=['POST'])
def predict_page():
    """表单提交入口：预测后渲染结果页。"""
    try:
        features = _extract_features(request.form)
    except ValueError as exc:
        return render_template('error.html', message=str(exc)), 400

    prediction = MODEL_SERVICE.predict(features)
    return render_template(
        'result.html',
        prediction=prediction,
        labels=config.FEATURE_LABELS,
        feature_list=config.FEATURE_LIST,
        active='predict',
    )


# ------------------------------------------------------------------ 预测 API
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON 预测接口，供前端异步调用或第三方集成。"""
    payload = request.get_json(silent=True) or {}
    try:
        if 'features' in payload:
            features = [float(v) for v in payload['features']]
        else:
            features = _extract_features(payload)
        normalize_sample(features)
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'message': f'输入数据有误：{exc}'}), 400

    prediction = MODEL_SERVICE.predict(features)
    return jsonify({'success': True, 'data': prediction})


@app.route('/api/interpret', methods=['POST'])
def api_interpret():
    """RAG 解读预测结果：预测 + 知识库检索 + 生成解释。"""
    payload = request.get_json(silent=True) or {}
    try:
        features = payload.get('features')
        if features:
            features = [float(v) for v in features]
            normalize_sample(features)
            prediction = MODEL_SERVICE.predict(features)
        elif payload.get('prediction'):
            prediction = payload['prediction']
        else:
            raise ValueError('需要提供 features 或 prediction')
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400

    result = RAG_SERVICE.interpret_prediction(prediction)
    return jsonify({'success': True, 'data': result, 'prediction': prediction})


# ------------------------------------------------------------------ 模型 API
@app.route('/api/model')
def api_model():
    return jsonify({'success': True, 'data': MODEL_SERVICE.info()})


@app.route('/api/model/compare')
def api_model_compare():
    """算法对照实验指标（用于前端与 PPT 展示）。"""
    info = MODEL_SERVICE.info()
    return jsonify({'success': True, 'data': info.get('comparison', {})})


@app.route('/api/model/switch', methods=['POST'])
def api_model_switch():
    """在线切换算法并重新训练。"""
    global MODEL_SERVICE
    payload = request.get_json(silent=True) or {}
    algorithm = payload.get('algorithm', '').strip()
    if algorithm not in config.CANDIDATE_ALGORITHMS:
        return jsonify({'success': False,
                        'message': f'不支持的算法：{algorithm}'}), 400
    try:
        MODEL_SERVICE.train(algorithm=algorithm)
    except Exception as exc:
        return jsonify({'success': False, 'message': f'训练失败：{exc}'}), 500
    return jsonify({'success': True, 'data': MODEL_SERVICE.info()})


@app.route('/api/model/retrain', methods=['POST'])
def api_model_retrain():
    global MODEL_SERVICE
    payload = request.get_json(silent=True) or {}
    algorithm = payload.get('algorithm') or MODEL_SERVICE.algorithm
    MODEL_SERVICE.train(algorithm=algorithm)
    reset_model_service()
    return jsonify({'success': True, 'data': MODEL_SERVICE.info()})


# ------------------------------------------------------------------ RAG API
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """RAG 问答接口。"""
    payload = request.get_json(silent=True) or {}
    question = (payload.get('question') or '').strip()
    if not question:
        return jsonify({'success': False, 'message': '请输入问题'}), 400
    result = RAG_SERVICE.answer(question)
    return jsonify({'success': True, 'data': result})


@app.route('/api/retrieve', methods=['POST'])
def api_retrieve():
    """只做检索、不做生成，便于观测 RAG 的召回效果。"""
    payload = request.get_json(silent=True) or {}
    question = (payload.get('question') or '').strip()
    if not question:
        return jsonify({'success': False, 'message': '请输入问题'}), 400
    chunks = RAG_SERVICE.retrieve(question, int(payload.get('top_k') or config.RAG_TOP_K))
    return jsonify({'success': True, 'data': chunks})


@app.route('/api/knowledge/rebuild', methods=['POST'])
def api_knowledge_rebuild():
    global RAG_SERVICE
    count = rebuild_index()
    RAG_SERVICE = get_rag_service()
    return jsonify({'success': True, 'data': {'chunk_count': count}})


# ------------------------------------------------------------------ 数据集 API
@app.route('/api/dataset/stats')
def api_dataset_stats():
    return jsonify({'success': True, 'data': dataset_statistics()})


@app.route('/api/dataset/sample')
def api_dataset_sample():
    """返回示例样本，供前端一键填充表单。"""
    label = request.args.get('label', 'random')
    try:
        rows = sample_rows(label, count=1)
    except Exception as exc:
        return jsonify({'success': False, 'message': str(exc)}), 500
    return jsonify({'success': True, 'data': rows[0] if rows else None,
                    'labels': config.FEATURE_LIST})


@app.route('/api/health')
def api_health():
    info = MODEL_SERVICE.info()
    return jsonify({
        'success': True,
        'status': 'ok',
        'algorithm': info['algorithm_label'],
        'accuracy': info['metrics'].get('accuracy'),
        'rag_documents': RAG_SERVICE.info()['document_count'],
        'rag_chunks': RAG_SERVICE.info()['chunk_count'],
        'llm_provider': RAG_SERVICE.info()['llm_provider'],
    })


# ------------------------------------------------------------------ 工具函数
def _extract_features(source) -> list[float]:
    """从表单或 JSON 中按 value1..value30 提取特征值。"""
    features = []
    for i in range(1, len(config.FEATURE_LIST) + 1):
        raw = source.get(f'value{i}')
        if raw is None or str(raw).strip() == '':
            raise ValueError(f'第 {i} 个特征（{config.FEATURE_LABELS[config.FEATURE_LIST[i - 1]]}）不能为空')
        try:
            features.append(float(raw))
        except (TypeError, ValueError):
            raise ValueError(f'第 {i} 个特征的值「{raw}」不是合法数字')
    return features


def _feature_groups() -> list[dict]:
    """按「均值 / 标准误 / 最差值」分组，并携带全局序号与中文提示。"""
    hints = _feature_hints()
    groups = []
    index = 1
    for key, title, names in config.FEATURE_GROUPS:
        fields = []
        for name in names:
            fields.append({
                'index': index,
                'name': name,
                'label': config.FEATURE_LABELS[name],
                'hint': hints.get(name, ''),
            })
            index += 1
        groups.append({'key': key, 'title': title, 'fields': fields})
    return groups


def _feature_hints() -> dict:
    """每个特征的中文说明，用于表单提示。"""
    base_hints = {
        'radius': '细胞核等效半径，恶性时增大',
        'texture': '灰度标准差，恶性时染色质分布不均导致升高',
        'perimeter': '细胞核轮廓周长，恶性轮廓不规则时增大',
        'area': '细胞核面积，恶性细胞核增大',
        'smoothness': '轮廓局部平滑程度，良性更光滑',
        'compactness': '偏离圆形的程度，恶性偏离更明显',
        'concavity': '轮廓凹陷的严重程度，恶性凹陷更明显',
        'concave points': '轮廓凹陷点的比例，恶性比例更高',
        'symmetry': '细胞核对称程度，恶性常不对称',
        'fractal_dimension': '轮廓粗糙复杂度，恶性更粗糙',
    }
    suffixes = {'mean': '均值', 'se': '标准误', 'worst': '最差值'}
    result = {}
    for name in config.FEATURE_LIST:
        base, suffix = name.rsplit('_', 1)
        result[name] = f'{base_hints.get(base, "")}（{suffixes.get(suffix, "")}）'
    return result


@app.errorhandler(404)
def not_found(_):
    return render_template('error.html', message='页面不存在'), 404


@app.errorhandler(500)
def server_error(error):
    traceback.print_exception(error)
    return render_template('error.html', message='服务器内部错误，请稍后重试'), 500


# 预设问题，展示在问答页作为快捷入口
PRESET_QUESTIONS = [
    '良性肿瘤和恶性肿瘤怎么区分？',
    'BI-RADS 分级是什么意思？',
    '30 个细胞核特征分别代表什么？',
    '逻辑回归和随机森林有什么区别？',
    '恶性概率 0.85 算高风险吗？该怎么办？',
    'RAG 检索增强生成是怎么工作的？',
]


if __name__ == '__main__':
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    print('=' * 62)
    print('乳愈智能 · 乳腺癌 AI 预测系统')
    print(f'主算法        : {MODEL_SERVICE.info()["algorithm_label"]}'
          f'（准确率 {MODEL_SERVICE.info()["metrics"].get("accuracy")}）')
    print(f'知识库         : {RAG_SERVICE.info()["chunk_count"]} 个知识片段 / '
          f'{RAG_SERVICE.info()["document_count"]} 篇文档')
    print(f'大模型模式     : {RAG_SERVICE.info()["llm_provider"]}')
    print(f'访问地址       : http://127.0.0.1:{config.PORT}')
    print('=' * 62)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
