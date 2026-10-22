"""可切换的大模型客户端。

支持四类后端，通过环境变量 LLM_PROVIDER 切换：
  - local  ：不调用大模型，由 RAG 管线使用「抽取式」方式组织答案（默认，离线可用）
  - openai ：任意 OpenAI 兼容接口（OpenAI、通义千问、DeepSeek、本地 Ollama 等）
  - dify   ：Dify 平台智能体 API
  - coze   ：Coze（扣子）平台智能体 API

任何调用失败都会安全降级为 local 模式，不会中断前端请求。
"""
import threading

import requests

import config


class LLMResult:
    """一次生成结果。"""

    def __init__(self, text: str, provider: str, used_llm: bool, error: str = ''):
        self.text = text
        self.provider = provider
        self.used_llm = used_llm
        self.error = error

    def to_dict(self) -> dict:
        return {
            'provider': self.provider,
            'used_llm': self.used_llm,
            'error': self.error,
        }


def _http_error_detail(response) -> str:
    """从错误响应里提取可读信息，便于定位 400 / 401 等失败原因。"""
    try:
        payload = response.json()
    except ValueError:
        return (response.text or '').strip()[:300] or '接口未返回内容'
    if isinstance(payload, dict):
        error = payload.get('error')
        if isinstance(error, dict) and error.get('message'):
            return str(error['message'])
        if isinstance(error, str) and error:
            return error
        for key in ('message', 'msg', 'detail'):
            if payload.get(key):
                return str(payload[key])
    return str(payload)[:300]


class LLMClient:
    """统一的 chat 调用入口。"""

    def __init__(self, provider: str | None = None):
        # provider 为 None 时，每次调用都实时读取 config.LLM_PROVIDER，
        # 这样在「设置」页切换大模型后端后无需重启服务即可生效。
        self._provider = provider

    @property
    def provider(self) -> str:
        return (self._provider or config.LLM_PROVIDER or 'local').lower()

    # ---------------------------------------------------------- 对外接口
    def chat(self, system_prompt: str, user_prompt: str) -> LLMResult:
        if self.provider == 'local':
            return LLMResult('', 'local', used_llm=False)

        try:
            if self.provider == 'openai':
                text = self._openai_chat(system_prompt, user_prompt)
            elif self.provider == 'dify':
                text = self._dify_chat(user_prompt)
            elif self.provider == 'coze':
                text = self._coze_chat(user_prompt)
            else:
                return LLMResult('', self.provider, used_llm=False,
                                 error=f'未知的 LLM_PROVIDER: {self.provider}')
        except Exception as exc:  # 网络异常、鉴权失败、超时等统一降级
            return LLMResult('', self.provider, used_llm=False,
                             error=f'{type(exc).__name__}: {exc}')

        if not (text or '').strip():
            return LLMResult('', self.provider, used_llm=False, error='大模型返回内容为空')
        return LLMResult(text.strip(), self.provider, used_llm=True)

    # ---------------------------------------------------------- 各平台实现
    def _openai_chat(self, system_prompt: str, user_prompt: str) -> str:
        if not config.LLM_API_KEY:
            raise ValueError('未配置 LLM_API_KEY')
        # API Key 会写入 HTTP 请求头，必须为纯 ASCII；中文/全角字符会触发 latin-1 编码错误
        if not config.LLM_API_KEY.isascii():
            raise ValueError('API Key 含中文或全角字符，请重新复制纯英文密钥（通常以 sk- 开头）')
        if not (config.LLM_MODEL or '').strip():
            raise ValueError('未配置模型名称（LLM_MODEL）')
        # 兼容用户直接粘贴完整端点（.../chat/completions）的情况，避免重复拼接
        base_url = (config.LLM_BASE_URL or '').rstrip('/')
        for suffix in ('/chat/completions',):
            if base_url.endswith(suffix):
                base_url = base_url[:-len(suffix)]
        response = requests.post(
            f'{base_url}/chat/completions',
            headers={
                'Authorization': f'Bearer {config.LLM_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': config.LLM_MODEL,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                'temperature': 0.2,
            },
            timeout=config.LLM_TIMEOUT,
        )
        if not response.ok:
            raise RuntimeError(
                f'接口返回 HTTP {response.status_code}：{_http_error_detail(response)}')
        data = response.json()
        return data['choices'][0]['message']['content']

    def _dify_chat(self, user_prompt: str) -> str:
        if not config.LLM_API_KEY:
            raise ValueError('未配置 LLM_API_KEY（Dify 应用密钥）')
        response = requests.post(
            config.DIFY_API_URL,
            headers={
                'Authorization': f'Bearer {config.LLM_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'inputs': {},
                'query': user_prompt,
                'response_mode': 'blocking',
                'user': 'breast-cancer-rag-demo',
            },
            timeout=config.LLM_TIMEOUT,
        )
        if not response.ok:
            raise RuntimeError(
                f'接口返回 HTTP {response.status_code}：{_http_error_detail(response)}')
        return response.json().get('answer', '')

    def _coze_chat(self, user_prompt: str) -> str:
        if not config.LLM_API_KEY or not config.COZE_BOT_ID:
            raise ValueError('未配置 LLM_API_KEY（Coze 令牌）或 COZE_BOT_ID')
        response = requests.post(
            config.COZE_API_URL,
            headers={
                'Authorization': f'Bearer {config.LLM_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'bot_id': config.COZE_BOT_ID,
                'user': 'breast-cancer-rag-demo',
                'query': user_prompt,
                'stream': False,
            },
            timeout=config.LLM_TIMEOUT,
        )
        if not response.ok:
            raise RuntimeError(
                f'接口返回 HTTP {response.status_code}：{_http_error_detail(response)}')
        messages = response.json().get('messages', [])
        parts = [m.get('content', '') for m in messages if m.get('type') == 'answer']
        return '\n'.join(p for p in parts if p)


_client: LLMClient | None = None
_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """获取全局单例 LLM 客户端。"""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = LLMClient()
    return _client
