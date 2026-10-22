# 乳愈智能 · 乳腺癌 AI 预测系统

> 《专业实习 2》期末大作业 —— 基于开源工程 breast-cancer-detection 重构的
> **完整前后端 + RAG 检索增强生成** 的乳腺癌 AI 预测应用。

## 一、项目简介

本系统在开源工程 `breast-cancer-detection`（Flask + 随机森林）的基础上，完成了三项核心改造：

| 改造项 | 开源工程 | 本项目 |
| --- | --- | --- |
| 算法 | 随机森林（scikit-learn Pipeline 缺失，需手工标准化） | **逻辑回归** 为主算法，封装为 StandardScaler + LogisticRegression 的 Pipeline，并保留随机森林作对照 |
| 前端 | 单页表单 + 结果页，Bootstrap 模板 | 全新设计的多页面应用：AI 预测、结果解读、智能问答、医学知识库，含概率仪表盘与关键因素可视化 |
| RAG | 无 | **自建本地 RAG**：11 篇医学知识文档 / 56 个检索片段，TF-IDF 向量 + BM25 关键词双路混合检索，大模型生成且附引用溯源，未配置大模型时自动降级为检索式答案 |

## 二、功能清单

### 1. AI 预测（`/`）

- 30 项细胞核形态学特征输入表单，按「均值 / 标准误 / 最差值」分三组展示，每项带中文标签与医学含义提示。
- 支持「填充良性示例 / 填充恶性示例 / 随机示例」一键载入真实数据集样本，便于演示。
- 预测输出：诊断结论（恶性 / 良性）、恶性概率、风险等级（低 / 中 / 高）、预测耗时、算法名称。
- 关键影响因素：以「标准化特征值 × 模型权重」计算贡献度并可视化排序，实现可解释输出。
- 算法对照实验表格，直观展示逻辑回归与随机森林的指标差异。

### 2. 结果页（`/predict`）

- 结论卡片 + 风险等级徽章 + 圆形概率仪表盘（带动画）。
- **RAG 智能解读**：把预测结果作为检索意图，从知识库召回相关医学知识，生成可追溯的结果解释。
- 特征贡献条形图、30 项输入特征明细表、三档风险分层建议。

### 3. 智能问答（`/assistant`）

- 基于本地医学知识库的 RAG 问答，回答按「检索 → 重排 → 生成 → 溯源」流程产出。
- 每条回答附带知识来源卡片（文档标题、小节、融合得分、向量分、关键词分、片段摘要）。
- 内置 6 个预设问题，方便演示。

### 4. 医学知识库（`/knowledge`）

- 11 篇医学文档在线浏览（基础概述、良恶性鉴别、BI-RADS、诊断流程、30 项特征说明、危险因素、治疗方案、评估指标、算法对比、RAG 说明、风险分层）。
- 检索效果自测：输入问题即可查看召回片段与融合得分，验证 RAG 检索质量。
- 一键重建索引。

## 三、目录结构

```
breast-cancer-rag/
├── app.py                     # 后端：Flask 主程序，页面路由 + REST 接口
├── auth.py                    # 登录鉴权：会话校验、口令校验、受保护端点清单
├── config.py                  # 全局配置：路径、特征元数据、算法/RAG/LLM 参数、运行时设置
├── start.bat                  # Windows 一键启动脚本（自动定位 Python、装依赖、开浏览器）
├── requirements.txt           # Python 依赖
├── algorithm/                 # 算法模块
│   ├── dataset.py             #   数据集加载、切分、示例样本
│   ├── models.py              #   逻辑回归 / 随机森林构建、训练评估、特征贡献
│   ├── predictor.py           #   模型服务：持久化、单样本预测、指标缓存
│   └── train.py               #   命令行训练脚本
├── rag/                       # RAG 模块
│   ├── loader.py              #   知识库文档加载与小节切分、分块
│   ├── text_utils.py          #   中文分词、分句、分块、Markdown 清理
│   ├── vector_store.py        #   混合检索向量库（TF-IDF + BM25）
│   ├── pipeline.py            #   RAG 问答管线与预测结果解读
│   └── documents/             #   医学知识库（11 篇 Markdown）
├── llm/
│   └── client.py              # 大模型客户端（local / openai / dify / coze）
├── templates/                 # 前端：Jinja2 模板
│   ├── base.html  index.html  result.html
│   ├── login.html  settings.html
│   ├── assistant.html  knowledge.html  document.html  error.html
├── static/
│   ├── css/style.css          # 前端样式（医疗蓝绿主题、响应式）
│   └── js/                    # app / predict / result / assistant / knowledge / document / settings
├── data/
│   └── Breast Cancer Data.csv # 数据集（569 例 / 30 特征，来自开源工程）
├── models/                    # 训练产物（自动生成，已 gitignore）
├── instance/                  # 运行时设置 settings.json（含 API Key，已 gitignore）
├── tests/smoke_test.py        # 接口冒烟测试（25 项）
└── doc/                       # 文档与演示视频（视频请放此处）
```

## 四、环境搭建与运行

```bash
# 1. 创建并激活虚拟环境（也可使用 anaconda：conda create -n myenv python=3.10）
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务（首次启动会自动训练模型并构建知识库索引）
python app.py
# 或 Windows 下直接双击 start.bat：自动定位 Python、安装依赖并打开浏览器

# 4. 浏览器访问
# http://127.0.0.1:5000
#    默认登录账号：admin / admin123（登录后可在右下角齿轮「设置」中修改）
```

### 可选：命令行训练

```bash
python -m algorithm.train                       # 训练默认算法（逻辑回归）
python -m algorithm.train --algorithm random_forest
python -m algorithm.train --compare             # 输出两种算法的对照结果
```

### 冒烟测试

```bash
python tests/smoke_test.py
```

## 五、接入大模型（可选）

系统默认使用 `local` 模式：RAG 的**检索**环节完全本地运行，**生成**环节使用检索式答案组织，
因此在无网络、无密钥的环境下功能完整可用。

**推荐方式：登录后点击网页右下角齿轮进入「设置」页**，选择后端（如 `openai`），
填入 Base URL / 模型名 / API Key，点「测试连接」验证后保存。配置会写入
`instance/settings.json` 并立即生效，无需重启服务；API Key 只保存在本机，不会随代码提交。

也可以改用环境变量在启动前指定（设置页中保存的值优先级更高）：

```bash
# 模式一：任意 OpenAI 兼容接口（OpenAI / 通义千问 / DeepSeek / 本地 Ollama）
set LLM_PROVIDER=openai
set LLM_API_KEY=sk-xxxxxx
set LLM_BASE_URL=https://api.deepseek.com/v1
set LLM_MODEL=deepseek-chat

# 模式二：Dify 平台智能体
set LLM_PROVIDER=dify
set LLM_API_KEY=app-xxxxxx
set DIFY_API_URL=https://api.dify.ai/v1/chat-messages

# 模式三：Coze（扣子）平台智能体
set LLM_PROVIDER=coze
set LLM_API_KEY=pat_xxxxxx
set COZE_BOT_ID=7xxxxxxxxxxxxx
```

任何调用失败都会自动降级为本地检索式答案，不会中断前端请求。

## 六、主要接口

除 `/login`、`/logout` 与静态资源外，其余页面与接口均需登录（未登录时页面跳转登录页、接口返回 401）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET / POST | `/login` | 登录页 / 提交账号口令 |
| GET | `/logout` | 退出登录 |
| GET | `/settings` | 系统设置页（大模型接入、RAG 参数、账号口令） |
| GET | `/api/settings` | 读取当前设置（API Key 只回显是否已配置） |
| POST | `/api/settings` | 保存设置并立即生效 |
| POST | `/api/settings/test` | 测试大模型连通性 |
| GET | `/` | 预测首页 |
| GET | `/assistant` | RAG 智能问答页 |
| GET | `/knowledge` | 医学知识库页 |
| POST | `/predict` | 表单提交并渲染结果页 |
| POST | `/api/predict` | JSON 预测接口 |
| POST | `/api/interpret` | RAG 解读预测结果 |
| POST | `/api/chat` | RAG 问答 |
| POST | `/api/retrieve` | 仅检索不生成（观测召回） |
| GET | `/api/model` | 模型信息与指标 |
| GET | `/api/model/compare` | 算法对照实验指标 |
| POST | `/api/model/switch` | 在线切换算法并重新训练 |
| GET | `/api/dataset/stats` | 数据集统计 |
| GET | `/api/dataset/sample` | 获取示例样本 |
| GET | `/api/health` | 服务健康状态 |

## 七、算法实验结果

数据集：569 例（恶性 212 / 良性 357），按 8:2 分层抽样，测试集 114 例。

| 指标 | 逻辑回归（主算法） | 随机森林（对照） |
| --- | --- | --- |
| 准确率 | 0.9737 | 0.9649 |
| 精确率 | 0.9756 | 1.0000 |
| 召回率 | 0.9524 | 0.9048 |
| F1 | 0.9639 | 0.9500 |
| ROC-AUC | 0.9954 | 0.9942 |
| 单样本推理耗时 | 约 0.19 ms | 约 32 ms |

混淆矩阵（逻辑回归）：TN 71 / FP 1 / FN 2 / TP 40。

**结论**：在本数据集上逻辑回归的准确率、召回率与推理速度均优于随机森林，且可解释性更强，
因此作为主算法；随机森林保留为可切换的对照算法。

## 八、免责声明

本系统为课程教学演示原型，所有预测结果与解读内容仅供健康科普与教学参考，
**不能替代医生的临床诊断、影像学评估与病理学检查**。
