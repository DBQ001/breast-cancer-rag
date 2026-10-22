"""文本处理工具：中文/英文混合分词、分句、分块。"""
import re

_CJK = r'\u4e00-\u9fff'
_LATIN_TOKEN = re.compile(r'[a-zA-Z0-9_]+')
_CJK_SEGMENT = re.compile(f'[{_CJK}]+')
_SENTENCE_SPLIT = re.compile(r'(?<=[。！？!?；;\n])')

# 中文检索时可忽略的高频虚词
STOPWORDS = {
    '的', '了', '是', '在', '和', '与', '或', '及', '等', '有', '为', '对', '中',
    '上', '下', '会', '要', '也', '不', '就', '都', '而', '之', '其', '并', '以',
    '被', '把', '让', '这', '那', '些', '个', '么', '吗', '呢', '啊', 'the', 'a',
    'an', 'of', 'is', 'are', 'to', 'and', 'or', 'in', 'on', 'for', 'with',
}


def tokenize(text: str) -> list[str]:
    """混合分词：英文按单词，中文按单字与二元组切分。

    该策略无需额外分词库即可对中文医学文本取得稳定的关键词召回效果。
    """
    text = (text or '').lower()
    tokens = [t for t in _LATIN_TOKEN.findall(text)]
    for segment in _CJK_SEGMENT.findall(text):
        tokens.extend(list(segment))
        if len(segment) > 1:
            tokens.extend(segment[i:i + 2] for i in range(len(segment) - 1))
    return [t for t in tokens if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """按中英文标点切分句子（保留换行，便于后续按行去除 Markdown 标记）。"""
    parts = [p.strip(' \t\r') for p in _SENTENCE_SPLIT.split(text or '')]
    return [p for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按句子边界聚合分块，块间保留 overlap 个字符重叠，避免语义被截断。"""
    sentences = split_sentences(text)
    chunks, buffer, length = [], [], 0

    for sentence in sentences:
        if length + len(sentence) > chunk_size and buffer:
            chunks.append(''.join(buffer).strip())
            # 构造重叠部分：从尾部回取若干句子
            tail, tail_len = [], 0
            for s in reversed(buffer):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            buffer, length = tail, tail_len
        buffer.append(sentence)
        length += len(sentence)

    if buffer:
        chunks.append(''.join(buffer).strip())
    return [c for c in chunks if len(c) > 10]


_MD_EMPHASIS = re.compile(r'\*\*(.+?)\*\*|__(.+?)__')
_MD_HEADING = re.compile(r'^\s*#{1,6}\s*', re.MULTILINE)
_MD_BULLET = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
# 句子之间没有换行时，列表符号会紧跟在前一句句末标点之后
_MD_BULLET_INLINE = re.compile(r'(?<=[。！？；;：:.])\s*[-*+]\s+')


def strip_markdown(text: str) -> str:
    """去掉 Markdown 标记，让检索片段与引用摘要以纯文本形式呈现。"""
    text = _MD_EMPHASIS.sub(lambda m: m.group(1) or m.group(2) or '', text or '')
    text = _MD_HEADING.sub('', text)
    text = _MD_BULLET.sub('', text)
    text = _MD_BULLET_INLINE.sub('', text)
    return text.replace('`', '').strip()


def normalize_scores(scores: list[float]) -> list[float]:
    """把一组分数线性归一化到 [0, 1]，全零时返回全零。"""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        return [0.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]
