from __future__ import annotations

import logging
import re
import threading
from typing import Any

from sqlalchemy import select

from app.core.postgres import get_session_factory
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

"""关键词检索（BM25）。

为什么向量检索之外还要有这一路——理由非常具体，来自法规文本的特征：
条文编号（"第二十九条""第一百四十三条"）是用户最常用的定位方式，
而**编号的语义信息极弱**：向量模型会把"第二十九条"和"第三十九条"
当成两个意思差不多的词；BM25 靠字面匹配，恰好擅长这个。

只做向量，等于放弃了合规人员最常用的一类查询方式。

⚠️ 一个必须知道的边界：**BM25 只看字面，不理解意思。**
用户问"能不能卖给老年人"，而条文写的是"风险承受能力最低类别的投资者"，
两边没有共同词，BM25 这一路基本什么都召不回来。
这不是 bug，是关键词检索的固有属性——所以它和向量检索是互补关系，
不是替代关系。检索调试台会把两路分别标出来，
这一点在那里会表现得很直观。
"""

# 英文/数字 token：允许下划线、点、冒号、斜杠、连字符，
# 这样 "III(B)"、"text-embedding-v1" 这类专有写法不会被切碎。
_LATIN_TOKEN = re.compile(r'[a-z0-9][a-z0-9._:/\-()]*')
_CJK_SEGMENT = re.compile(r'[\u4e00-\u9fff]+')

# 切片开头的条号。它是"这一片**就是**第 X 条"的标志。
_LEADING_ARTICLE = re.compile(r'^第[一二三四五六七八九十百零〇\d]+条')

try:
    import jieba

    # jieba 第一次调用会加载词典并打印日志，这里提前静音
    jieba.setLogLevel(logging.WARNING)
except Exception:  # noqa: BLE001
    jieba = None


def tokenize(text: str) -> list[str]:
    """把文本切成适合 BM25 的 token。

    中英文分别处理：
    - 英文/数字按正则切；
    - 中文优先用 jieba 分词，没有 jieba 就退化为 2-gram 加单字
      （召回够用，精度差一些，但至少不会因为缺一个库就完全不可用）。
    """

    normalized = str(text or '').lower()
    if not normalized.strip():
        return []

    tokens: list[str] = _LATIN_TOKEN.findall(normalized)

    for segment in _CJK_SEGMENT.findall(normalized):
        if jieba is not None:
            tokens.extend(token for token in jieba.cut_for_search(segment) if token.strip())
        elif len(segment) > 1:
            tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
            tokens.extend(segment)
        else:
            tokens.append(segment)

    return [token for token in tokens if token]


def _lexical_text(chunk: Chunk) -> str:
    """拼出参与关键词检索的文本。

    法规名 / 标题 / 小节标题**各重复一次**，等于给它们加权。
    理由是合规问答里最典型的一类查询："《证券法》第八十八条怎么规定的"——
    回答这个问题要同时命中两个线索：**法规名**和**条号**。

    而这两个线索在正文里往往都不出现：

    - 法规名不出现：《证券法》的条文里写的是"本法"，不会写"证券法"；
    - 条号只出现在切片开头（"第八十八条　……"），
      如果用户问的是"第88条"（阿拉伯数字），字面对不上。

    所以法规名必须由元数据补进来，并且加权——
    不加权的话，一份法规的名字会被正文里成百上千个字淹没。
    """

    metadata = chunk.metadata_json or {}
    filename = str(metadata.get('filename') or '')
    document_title = str(metadata.get('title') or '')
    section_title = str(chunk.section_title or '')

    # ---- 开头条号的加权 ----
    #
    # 这一条是被真实检索结果逼出来的。问"《证券期货投资者适当性管理办法》
    # 第二十九条怎么规定的"，BM25 的正确切片只排到第 5 名，第 1 名是
    # 一条**引用**了第二十九条的切片（"（六）本办法第二十九条规定的适当性匹配意见"）。
    #
    # 原因很直白：两片正文里"第二十九条"都只出现一次，BM25 分不出
    # "它是第二十九条"和"它提到第二十九条"——**而在法规问答里，
    # 这个区别恰恰是最重要的一件事。**
    #
    # 修法是把这个区别显式写进索引：切片如果**以**"第X条"开头，
    # 就把这个条号额外重复几次。它编码的是"本片正文即该条"，
    # 而不是一个普通的词频。
    content = chunk.content or ''
    leading = _LEADING_ARTICLE.match(content.strip())
    article_boost = [leading.group(0)] * 3 if leading else []

    return '\n'.join([
        filename, filename,
        document_title, document_title,
        section_title,
        *article_boost,
        content,
    ])


class BM25Index:
    """内存里的 BM25 索引。

    为什么放内存而不是建数据库全文索引：
    数据量在几千到几万条切片这个量级，内存索引重建一次是秒级，
    查询是毫秒级，而且零额外依赖。等真的到几十万条再考虑换方案——
    那时候瓶颈也不会是 BM25。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._dirty = True
        self._bm25 = None
        self._records: list[dict[str, Any]] = []
        self._rebuild_count = 0
        self._last_reason = 'initial'

    def mark_dirty(self, reason: str) -> None:
        """标记索引需要重建。

        切片一旦变化（重新切分、删除文档、启停切片），索引就过期了。
        不标记的话，检索会命中已经不存在的内容——
        表现为"搜到一条，点开发现是空的"，而且很难查。
        """

        with self._lock:
            self._dirty = True
            self._last_reason = reason
        logger.info('[BM25] 索引标记为待重建: reason=%s', reason)

    def ensure_ready(self) -> None:
        with self._lock:
            needs_rebuild = self._dirty or self._bm25 is None
        if needs_rebuild:
            self.rebuild()

    def rebuild(self) -> int:
        """从数据库重建索引。"""

        from rank_bm25 import BM25Okapi

        session_factory = get_session_factory()
        with session_factory() as db:
            chunks = list(
                db.execute(select(Chunk).where(Chunk.enabled.is_(True))).scalars().all()
            )

        records: list[dict[str, Any]] = []
        corpus: list[list[str]] = []
        for chunk in chunks:
            tokens = tokenize(_lexical_text(chunk))
            if not tokens:
                continue
            metadata = chunk.metadata_json or {}
            records.append(
                {
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'filename': metadata.get('filename'),
                    'page_number': chunk.page_number,
                    'section_title': chunk.section_title,
                    'splitter_name': chunk.splitter_name,
                    'content_type': chunk.content_type,
                    'text': chunk.content,
                    # 效力信息要跟着检索结果走。
                    # 提示词里有条硬规则：回答"是否违规"必须说明依据出自哪一层效力。
                    # 如果把这两个字段只留在向量库里，关键词路召回的片段就没有它们，
                    # 模型只好对着"来源: 某某.pdf"硬猜——那就不是在回答，是在编。
                    'legal_level': metadata.get('legal_level'),
                    'validity': metadata.get('validity'),
                }
            )
            corpus.append(tokens)

        bm25 = BM25Okapi(corpus) if corpus else None

        with self._lock:
            self._bm25 = bm25
            self._records = records
            self._dirty = False
            self._rebuild_count += 1
            count = self._rebuild_count
            reason = self._last_reason

        logger.info(
            '[BM25] 索引重建完成: reason=%s 切片=%s 可索引=%s 第%s次',
            reason,
            len(chunks),
            len(records),
            count,
        )
        return len(records)

    def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        """关键词检索。"""

        self.ensure_ready()

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            bm25 = self._bm25
            records = list(self._records)

        if bm25 is None or not records:
            return []

        scores = bm25.get_scores(query_tokens)
        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )

        hits: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(ranked, start=1):
            if score <= 0:
                break
            hit = dict(records[index])
            hit.update(
                {
                    'score': score,
                    'bm25_score': score,
                    'rank_bm25': rank,
                    'retrieval_source': 'bm25',
                    'retrieval_sources': ['bm25'],
                }
            )
            hits.append(hit)
            if len(hits) >= top_k:
                break

        logger.info(
            '[BM25] 检索完成: query=%r tokens=%s 命中=%s',
            query,
            query_tokens[:8],
            len(hits),
        )
        return hits


_INDEX: BM25Index | None = None
_INDEX_LOCK = threading.Lock()


def get_bm25_index() -> BM25Index:
    """返回 BM25 索引单例。"""

    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                _INDEX = BM25Index()
    return _INDEX
