"""接口冒烟测试：无需 pytest，直接运行本文件即可自检。

覆盖登录鉴权、页面渲染、预测、RAG、算法对照与设置接口。
注意：登录用例使用默认账号 admin / admin123；若已通过设置页改过口令，
请先删除 instance/settings.json 再运行。

用法：
    python tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402


def main() -> int:
    failures = []

    def check(name, condition, detail=''):
        status = 'PASS' if condition else 'FAIL'
        print(f'[{status}] {name}{(" -> " + detail) if detail else ""}')
        if not condition:
            failures.append(name)

    # 0. 登录鉴权
    anon = app_module.app.test_client()
    check('登录页可访问', anon.get('/login').status_code == 200)
    guarded = anon.get('/')
    check('未登录访问页面跳转登录页',
          guarded.status_code == 302 and '/login' in guarded.headers.get('Location', ''),
          f'HTTP {guarded.status_code}')
    check('未登录访问接口返回 401', anon.get('/api/health').status_code == 401)
    wrong = anon.post('/login', data={'username': 'admin', 'password': 'wrong-pass'})
    check('错误口令登录被拒绝',
          wrong.status_code == 200 and '不正确' in wrong.get_data(as_text=True))

    client = app_module.app.test_client()
    login = client.post('/login', data={'username': 'admin', 'password': 'admin123'})
    check('默认账号登录成功', login.status_code == 302)

    # 0.1 普通用户角色：可访问业务页面，但看不到管理员设置
    viewer = app_module.app.test_client()
    viewer_login = viewer.post('/login', data={'username': 'user', 'password': '1234'})
    check('普通用户登录成功', viewer_login.status_code == 302)
    viewer_home = viewer.get('/')
    home_html = viewer_home.get_data(as_text=True)
    check('普通用户首页不显示设置入口', 'settings-fab' not in home_html)
    settings_redirect = viewer.get('/settings')
    check('普通用户访问设置页被重定向',
          settings_redirect.status_code == 302
          and '/settings' not in settings_redirect.headers.get('Location', ''))
    check('普通用户读写设置接口返回 403',
          viewer.get('/api/settings').status_code == 403
          and viewer.post('/api/settings', json={'rag_top_k': 4}).status_code == 403)
    check('普通用户仍可使用预测接口', viewer.post('/api/predict', json={
        'features': [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
                     0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
                     0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                     0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}).status_code == 200)

    # 1. 健康检查
    health = client.get('/api/health').get_json()
    check('健康检查接口', health.get('status') == 'ok',
          f"算法={health.get('algorithm')} 准确率={health.get('accuracy')}")

    # 2. 页面渲染
    for path in ('/', '/assistant', '/knowledge', '/settings'):
        response = client.get(path)
        check(f'页面渲染 {path}', response.status_code == 200,
              f'HTTP {response.status_code}')

    # 3. 示例数据 + 预测
    sample = client.get('/api/dataset/sample?label=malignant').get_json()['data']
    features = sample['features']
    check('示例数据接口', len(features) == 30, f'维度={len(features)}')

    payload = {f'value{i + 1}': v for i, v in enumerate(features)}
    response = client.post('/api/predict', json=payload)
    result = response.get_json()
    check('JSON 预测接口', response.status_code == 200 and result['success'])
    prediction = result.get('data', {})
    check('预测输出包含诊断结论', prediction.get('output') in ('Malignant', 'Benign'),
          f"结论={prediction.get('output_cn')} 概率={prediction.get('probability')} "
          f"风险={prediction.get('risk_level')}")
    check('预测耗时在毫秒级', (prediction.get('elapsed_ms') or 999) < 200,
          f"{prediction.get('elapsed_ms')} ms")
    check('输出关键影响因素', len(prediction.get('contributions', [])) == 6)

    # 4. 表单提交渲染结果页
    form_response = client.post('/predict', data=payload)
    check('表单预测渲染结果页', form_response.status_code == 200
          and 'RAG 智能解读' in form_response.get_data(as_text=True))

    # 5. RAG 问答
    chat = client.post('/api/chat', json={'question': '良性肿瘤和恶性肿瘤怎么区分？'}).get_json()
    check('RAG 问答接口', chat['success'] and len(chat['data']['answer']) > 20,
          f"模式={chat['data']['mode']} 来源数={len(chat['data']['sources'])}")
    check('RAG 问答带知识来源', len(chat['data']['sources']) > 0)

    # 6. RAG 解读预测结果
    interpret = client.post('/api/interpret', json={'features': features}).get_json()
    check('RAG 解读预测结果', interpret['success'] and len(interpret['data']['answer']) > 20,
          f"模式={interpret['data']['mode']}")

    # 7. 检索接口
    retrieve = client.post('/api/retrieve', json={'question': 'BI-RADS 分级'}).get_json()
    check('知识库检索接口', retrieve['success'] and len(retrieve['data']) > 0,
          f"召回 {len(retrieve['data'])} 个片段")

    # 8. 算法对照
    compare = client.get('/api/model/compare').get_json()
    check('算法对照接口', compare['success']
          and {'logistic_regression', 'random_forest'} <= set(compare['data'].keys()),
          '、'.join(compare['data'].keys()))

    # 9. 非法输入处理
    bad = client.post('/api/predict', json={'value1': 'abc'})
    check('非法输入返回 400', bad.status_code == 400)

    # 10. 设置接口（只写入与当前一致的值，避免污染真实配置）
    current = client.get('/api/settings').get_json()
    check('读取设置接口', current['success'] and 'llm_provider' in current['data'],
          f"大模型后端={current['data']['llm_provider']}")
    saved = client.post('/api/settings',
                        json={'rag_top_k': current['data']['rag_top_k']}).get_json()
    check('保存设置接口', saved['success'])
    test_conn = client.post('/api/settings/test', json={'llm_provider': 'local'}).get_json()
    check('设置连通性测试（local）', test_conn['success'] and test_conn['data']['ok'])

    print('\n' + ('全部通过' if not failures else f'存在失败项：{failures}'))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
