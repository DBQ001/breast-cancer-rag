"""RAG 问答管线：检索 -> 重排 -> 生成 -> 溯源。

本模块提供两处能力：
1. answer(question)            医学知识智能问答
2. interpret_prediction(result) 对模型预测结果做知识增强解读

生成环节优先调用已配置的大模型；未配置或调用失败时，自动降级为
「抽取式答案」：从检索到的片段中挑选与问题最相关的句子，按结构化模板
组织并保留引用编号，保证功能在任何环境下都可演示。
"""
import threading

import config
from llm.client import get_llm_client
from .loader import list_documents, load_chunks
from .text_utils import split_sentences, strip_markdown, tokenize
from .vector_store import HybridVectorStore

SYSTEM_PROMPT = (
    '你是「乳愈智能」乳腺癌 AI 预测系统的医学知识助手。'
    '请严格依据检索到的知识库片段回答问题，不得编造知识库之外的事实。'
    '回答要求：使用简体中文；先给出结论，再分点说明依据；'
    '在引用某段知识时以 [编号] 标注来源；'
    '最后补充一句「以上内容仅供健康科普与教学参考，不能替代医生诊断」。'
)

# 检索得分过低时判定为「知识库未覆盖」
MIN_RELEVANT_SCORE = 0.08


class RagService:
    """知识库索引与问答服务。"""

    def __init__(self):
        self.store = HybridVectorStore()
        self.documents: list[dict] = []

    # ---------------------------------------------------------- 索引构建
    def build_index(self, persist: bool = True) -> int:
        """重新读取知识库文档并构建索引，返回片段总数。"""
        chunks = load_chunks()
        self.store.build(chunks)
        self.documents = list_documents()
        if persist:
            self.store.save()
        return len(chunks)

    def load_or_build(self) -> 'RagService':
        """优先加载已持久化的索引，否则构建并保存。"""
        if not self.store.load():
            self.build_index()
        if not self.documents:
            self.documents = list_documents()
        return self

    # ---------------------------------------------------------- 检索
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.store.search(query, top_k)

    # ---------------------------------------------------------- 问答
    def answer(self, question: str, top_k: int | None = None) -> dict:
        """完整 RAG 流程：检索知识库 -> 组织上下文 -> 生成答案。"""
        question = (question or '').strip()
        if not question:
            return self._empty('请输入需要咨询的问题')

        retrieved = self.retrieve(question, top_k)
        if not retrieved or retrieved[0]['score'] < MIN_RELEVANT_SCORE:
            return self._empty(
                '知识库中未检索到与该问题足够相关的内容，请尝试换一种问法，'
                '例如「良性肿瘤和恶性肿瘤怎么区分」「BI-RADS 分级是什么意思」。',
                retrieved=retrieved,
            )

        context = self._format_context(retrieved)
        prompt = (
            f'【知识库检索片段】\n{context}\n\n'
            f'【用户问题】{question}\n\n'
            '请依据上述片段作答，并标注引用编号。'
        )

        llm_result = get_llm_client().chat(SYSTEM_PROMPT, prompt)
        if llm_result.used_llm:
            answer_text = llm_result.text
        else:
            answer_text = self._extractive_answer(question, retrieved)

        return {
            'question': question,
            'answer': answer_text,
            'mode': '大模型生成' if llm_result.used_llm else '检索式生成（未接入大模型）',
            'provider': llm_result.provider,
            'fallback_reason': llm_result.error,
            'sources': self._sources(retrieved),
            'retrieved': retrieved,
        }

    def interpret_prediction(self, prediction: dict) -> dict:
        """把模型预测结果交给 RAG，生成带知识来源的智能解读。

        这是「预测 + RAG」的核心融合点：预测结果决定检索意图，
        检索到的医学知识再回过头解释预测结论。
        """
        features = prediction.get('contributions', [])
        feature_desc = '、'.join(f"{f['label']}({f['value']})" for f in features[:5])
        query = (
            f"{prediction.get('output_cn', '')}肿瘤 风险等级 {prediction.get('risk_level', '')} "
            f"恶性概率 {prediction.get('probability', 0):.2%} 关键特征 {feature_desc}"
        )

        retrieved = self.retrieve(query, top_k=config.RAG_TOP_K)
        if not retrieved:
            return self._empty('知识库为空，无法生成解读')

        context = self._format_context(retrieved)
        prompt = (
            f'【模型预测结果】\n'
            f'- 诊断结论：{prediction.get("output_cn")}\n'
            f'- 恶性概率：{prediction.get("probability", 0):.2%}\n'
            f'- 风险等级：{prediction.get("risk_level")}\n'
            f'- 关键影响因素：{feature_desc}\n'
            f'- 使用算法：{prediction.get("algorithm_label")}\n\n'
            f'【知识库检索片段】\n{context}\n\n'
            '请结合上述知识，解释该预测结果的医学含义、主要依据以及后续建议，'
            '并标注引用编号。'
        )

        llm_result = get_llm_client().chat(SYSTEM_PROMPT, prompt)
        if llm_result.used_llm:
            answer_text = llm_result.text
        else:
            answer_text = self._extractive_interpretation(prediction, retrieved)

        return {
            'question': query,
            'answer': answer_text,
            'mode': '大模型生成' if llm_result.used_llm else '检索式生成（未接入大模型）',
            'provider': llm_result.provider,
            'fallback_reason': llm_result.error,
            'sources': self._sources(retrieved),
            'retrieved': retrieved,
        }

    # ---------------------------------------------------------- 内部方法
    @staticmethod
    def _format_context(retrieved: list[dict]) -> str:
        blocks = []
        for i, item in enumerate(retrieved, start=1):
            blocks.append(
                f'[{i}] 《{item["doc_title"]} · {item["section"]}》\n{item["text"]}'
            )
        return '\n\n'.join(blocks)

    @staticmethod
    def _sources(retrieved: list[dict]) -> list[dict]:
        return [{
            'index': i,
            'title': item['doc_title'],
            'section': item['section'],
            'filename': item['filename'],
            'score': item['score'],
            'vector_score': item['vector_score'],
            'keyword_score': item['keyword_score'],
            'snippet': item['text'][:120] + ('…' if len(item['text']) > 120 else ''),
        } for i, item in enumerate(retrieved, start=1)]

    def _extractive_answer(self, question: str, retrieved: list[dict]) -> str:
        """抽取式兜底：按问题关键词相关性挑选原句并结构化输出。"""
        query_tokens = set(tokenize(question))
        candidates = []
        for i, item in enumerate(retrieved, start=1):
            for sentence in split_sentences(item['text']):
                sentence = strip_markdown(sentence)
                if len(sentence) < 8:
                    continue
                overlap = len(query_tokens & set(tokenize(sentence)))
                candidates.append((overlap, i, sentence))

        candidates.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for _, index, sentence in candidates:
            if sentence in seen:
                continue
            seen.add(sentence)
            picked.append(f'{len(picked) + 1}. {sentence} [{index}]')
            if len(picked) >= 4:
                break

        if not picked:
            picked = [f'1. {strip_markdown(retrieved[0]["text"])[:160]}… [1]']

        return (
            '根据知识库检索结果，整理如下要点：\n' + '\n'.join(picked) +
            '\n\n以上内容仅供健康科普与教学参考，不能替代医生诊断。'
        )

    def _extractive_interpretation(self, prediction: dict, retrieved: list[dict]) -> str:
        """抽取式兜底：结合预测结果组织解读要点。"""
        lines = [
            f'模型判定为「{prediction.get("output_cn")}」，'
            f'恶性概率 {prediction.get("probability", 0):.2%}，'
            f'风险等级为「{prediction.get("risk_level")}」。',
            f'主要依据来自以下知识库内容：',
        ]
        for i, item in enumerate(retrieved, start=1):
            sentences = split_sentences(item['text'])
            snippet = strip_markdown(sentences[0]) if sentences else strip_markdown(item['text'])[:100]
            lines.append(f'{i}. {snippet}')
        lines.append('以上内容仅供健康科普与教学参考，不能代替医生诊断，'
                     '建议结合临床检查由专业医师综合判断。')
        return '\n'.join(lines)

    def _empty(self, message: str, retrieved: list[dict] | None = None) -> dict:
        return {
            'question': '',
            'answer': message,
            'mode': '未命中知识库',
            'provider': get_llm_client().provider,
            'fallback_reason': '',
            'sources': [],
            'retrieved': retrieved or [],
        }

    # ---------------------------------------------------------- 元信息
    def info(self) -> dict:
        return {
            'document_count': len(self.documents),
            'chunk_count': len(self.store.chunks),
            'llm_provider': get_llm_client().provider,
            'top_k': config.RAG_TOP_K,
            'documents': [{'title': d['title'], 'filename': d['filename'],
                           'chars': d['chars']} for d in self.documents],
        }


_service: RagService | None = None
_lock = threading.Lock()


def get_rag_service() -> RagService:
    """获取全局单例 RAG 服务。"""
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = RagService().load_or_build()
    return _service


def rebuild_index() -> int:
    """热更新知识库索引（新增文档后可调用）。"""
    return get_rag_service().build_index()
