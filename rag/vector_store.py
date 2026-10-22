"""轻量级混合检索向量库。

无需外部向量数据库与 Embedding 服务，完全本地运行：
  - 稠密检索：字符级 TF-IDF 向量余弦相似度（对中文医学文本友好）
  - 稀疏检索：BM25 关键词打分
最终得分 = 向量权重 × 向量相似度 + 关键词权重 × 归一化 BM25 得分
该设计保证即使没有网络或大模型 API，RAG 的「检索」环节依然可用。
"""
import math
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from .text_utils import normalize_scores, tokenize


class _BM25:
    """BM25 关键词打分器（k1 / b 采用经典取值）。"""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.doc_count = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avg_length = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0
        self.term_freqs = [Counter(doc) for doc in corpus_tokens]

        document_freq: Counter = Counter()
        for doc in corpus_tokens:
            document_freq.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        results = []
        for i, freqs in enumerate(self.term_freqs):
            length = self.doc_lengths[i]
            score = 0.0
            for term in query_tokens:
                if term not in freqs:
                    continue
                tf = freqs[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denominator
            results.append(score)
        return results


class HybridVectorStore:
    """文档片段的索引与混合检索。"""

    def __init__(self):
        self.chunks: list[dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.bm25: _BM25 | None = None

    # ---------------------------------------------------------- 构建索引
    def build(self, chunks: list[dict]) -> 'HybridVectorStore':
        if not chunks:
            raise ValueError('知识库为空，无法构建索引')
        self.chunks = chunks
        texts = [c['text'] for c in chunks]

        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4), min_df=1, sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self.bm25 = _BM25([tokenize(t) for t in texts])
        return self

    # ---------------------------------------------------------- 检索
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """返回 top_k 个最相关片段，附带向量/关键词分项得分。"""
        if not self.chunks or self.vectorizer is None:
            return []

        top_k = top_k or config.RAG_TOP_K
        top_k = max(1, min(top_k, len(self.chunks)))

        query_vector = self.vectorizer.transform([query])
        vector_scores = cosine_similarity(query_vector, self.matrix)[0].tolist()

        # 查询过短时 BM25 关键词命中过少，直接退化为纯向量检索
        keyword_raw = self.bm25.scores(tokenize(query))
        keyword_scores = normalize_scores(keyword_raw)

        vector_norm = normalize_scores(vector_scores)
        combined = [
            config.RAG_VECTOR_WEIGHT * v + config.RAG_KEYWORD_WEIGHT * k
            for v, k in zip(vector_norm, keyword_scores)
        ]

        order = np.argsort(-np.asarray(combined))[:top_k]
        results = []
        for idx in order:
            idx = int(idx)
            chunk = self.chunks[idx]
            results.append({
                'text': chunk['text'],
                'doc_title': chunk['doc_title'],
                'filename': chunk['filename'],
                'section': chunk['section'],
                'score': round(float(combined[idx]), 4),
                'vector_score': round(float(vector_norm[idx]), 4),
                'keyword_score': round(float(keyword_scores[idx]), 4),
            })
        return results

    # ---------------------------------------------------------- 持久化
    def save(self, path: str | None = None) -> None:
        path = path or config.VECTOR_STORE_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'chunks': self.chunks,
            'vectorizer': self.vectorizer,
            'matrix': self.matrix,
            'bm25': self.bm25,
        }, path)

    def load(self, path: str | None = None) -> bool:
        path = path or config.VECTOR_STORE_FILE
        if not os.path.exists(path):
            return False
        try:
            data = joblib.load(path)
        except Exception:
            return False
        self.chunks = data.get('chunks', [])
        self.vectorizer = data.get('vectorizer')
        self.matrix = data.get('matrix')
        self.bm25 = data.get('bm25')
        return bool(self.chunks) and self.matrix is not None
