"""知识库文档加载与分块。

知识库位于 rag/documents/ 目录，支持 .md 与 .txt。
每个文档按二级标题(##)组织小节，分块时保留所属文档标题与小节标题，
以便检索结果可以给出可追溯的引用来源。
"""
import os
import re

import config
from .text_utils import chunk_text, strip_markdown

_TITLE_RE = re.compile(r'^#\s+(.*)$', re.MULTILINE)
_SECTION_RE = re.compile(r'^#{2,3}\s+(.*)$', re.MULTILINE)


def list_documents() -> list[dict]:
    """列出知识库中的文档（标题、文件名、字符数）。"""
    docs = []
    for filename in sorted(os.listdir(config.KNOWLEDGE_DIR)):
        if not filename.lower().endswith(('.md', '.txt')):
            continue
        path = os.path.join(config.KNOWLEDGE_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = _TITLE_RE.search(content)
        docs.append({
            'filename': filename,
            'title': match.group(1).strip() if match else filename,
            'chars': len(content),
            'sections': _SECTION_RE.findall(content),
        })
    return docs


def read_document(filename: str) -> dict | None:
    """读取单个文档全文，用于前端知识库阅读页。"""
    path = os.path.join(config.KNOWLEDGE_DIR, filename)
    if not os.path.exists(path) or not filename.lower().endswith(('.md', '.txt')):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = _TITLE_RE.search(content)
    return {
        'filename': filename,
        'title': match.group(1).strip() if match else filename,
        'content': content,
    }


def load_chunks() -> list[dict]:
    """加载全部文档并切分为检索片段。

    返回片段列表，每项包含：
        doc_title  文档标题
        filename   来源文件
        section    所属小节标题
        text       片段正文
    """
    chunks: list[dict] = []

    for filename in sorted(os.listdir(config.KNOWLEDGE_DIR)):
        if not filename.lower().endswith(('.md', '.txt')):
            continue
        path = os.path.join(config.KNOWLEDGE_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        title_match = _TITLE_RE.search(content)
        doc_title = title_match.group(1).strip() if title_match else filename

        # 先按小节拆分，避免不同主题的内容落入同一片段
        sections = _split_sections(content)
        for section_title, body in sections:
            for piece in chunk_text(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                chunks.append({
                    'doc_title': doc_title,
                    'filename': filename,
                    'section': section_title,
                    # 存库前去掉 Markdown 标记，保证检索片段与引用摘要都是纯文本
                    'text': strip_markdown(piece),
                })
    return chunks


def _split_sections(content: str) -> list[tuple[str, str]]:
    """把文档按 ## / ### 标题切分成 (小节标题, 正文) 列表。"""
    lines = content.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = '概述'
    buffer: list[str] = []

    for line in lines:
        if _TITLE_RE.match(line):
            # 一级标题作为文档标题，不进入正文
            continue
        section_match = _SECTION_RE.match(line)
        if section_match:
            if buffer:
                sections.append((current_title, '\n'.join(buffer).strip()))
            current_title = section_match.group(1).strip()
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_title, '\n'.join(buffer).strip()))
    return [(t, b) for t, b in sections if b]
