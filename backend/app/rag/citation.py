from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.rag.splitters.legal import chinese_number_to_int, find_article_tokens

logger = logging.getLogger(__name__)

"""条款级精确检索。

解决的问题很具体：问"《某办法》第二十九条怎么规定的"，正确答案排不到前面。

实测过那条链路（工具/诊断条号检索.py 可以复现）：

    正确答案   向量路第 33 名 ｜ 关键词路第 1 名 ｜ RRF 第 2 名 ｜ 重排后第 6 名

关键词检索其实**做对了**——它把正确答案排在第 1，而且分数领先第二名 18%
（53.1 对 44.9，区分度全在"第二十九条"这一个词上）。

坏在后面两步：

  1. **RRF 只按名次融合**（1/(60+名次)），把 18% 的分差压成 3.4%。
     它换来的是"免疫量纲"，代价是**主动丢掉了让这类查询成立的唯一信号**。
  2. **重排是语义相关性模型**，它把"第二十九条"当成一个主题词，
     而不是标识符。证据是它给正确答案 0.4225、给一个无关规章里
     "本办法自2007年8月1日起施行"的第二十九条 0.4382——它压根没在用条号判断。

所以修法不是调重排参数，而是**给条号一个不该被相似度覆盖的身份**：
精确匹配是**过滤**，不是排序。过滤出来的结果不经过 RRF、不经过重排，
也就不会被它们挤掉。
"""

# 法规名匹配：取"法规名的后缀出现在查询里"的最长长度。
#
# 为什么要用后缀而不是子串：用户经常用简称（"证券法"之于"中华人民共和国证券法"、
# "适当性管理办法"之于"证券期货投资者适当性管理办法"），而简称恰好都是全称的后缀。
#
# 为什么不用子串：全称中间的一段（比如"证券投资基金"）在查询里出现，
# 并不能说明用户指的是这份文件。
MIN_TITLE_SUFFIX = 3

# 这些词本身不构成法规名。只有三个字的匹配结果如果是它们，说明用户
# 说的是"某部办法"而不是"这部办法"，不能拿来锁定文件。
GENERIC_TAILS = frozenset({
    '办法', '规定', '条例', '细则', '指引', '规则', '准则', '决定',
    '通知', '公告', '法律', '法规',
})


@dataclass(frozen=True)
class DocumentRef:
    """锁定法规名时用得上的最小信息。"""

    id: str
    title: str


@dataclass
class CitationQuery:
    """从问题里解析出来的"引用意图"。"""

    article_number: str | None = None
    article_int: int | None = None
    document_id: str | None = None
    document_title: str | None = None
    document_matched_chars: int = 0
    article_candidates: list[str] = field(default_factory=list)
    reason: str = ''

    @property
    def usable(self) -> bool:
        """够不够做精确直查。

        要求**法规名和条号同时命中**。只有条号是不够的——
        实测"第二十九条"在库里命中 13 个不同文件的切片，
        只按条号过滤召回的是一堆别的法规的第二十九条，比不查还乱。
        """

        return (
            self.article_number is not None
            and self.document_id is not None
        )


def _normalize(text: str) -> str:
    """查询侧归一化：去掉书名号和各种空白。"""

    cleaned = text.replace('《', '').replace('》', '')
    return re.sub(r'[\s\u3000]+', '', cleaned)


# 法规名末尾的括注。常见的有"（试行）""（2025年修订）""（修订案）"。
#
# 必须去掉，否则后缀匹配会整体失效。踩过一次：
# 《基金募集机构投资者适当性管理实施指引（试行）》的名字以"（试行）"结尾，
# 于是它的**任何后缀都以"（试行）"收尾**，而用户不会打这个括号——
# 结果这条法规一次都匹配不上，问"基金募集机构投资者适当性管理实施指引第十条"
# 反而锁到了名字更短的《深圳证券交易所创业板投资者适当性管理》。
#
# 用循环是因为括注可能不止一层（"（2025年修订）（试行）"这种写法存在）。
_TRAILING_PAREN = re.compile(r'[（(][^（()）]*[）)]$')


def _clean_title(title: str) -> str:
    """把法规名规整成"用户会打的样子"。"""

    cleaned = _normalize(title)
    while True:
        stripped = _TRAILING_PAREN.sub('', cleaned)
        if stripped == cleaned:
            break
        cleaned = stripped
    return cleaned


def _best_title_suffix(title: str, query: str) -> int:
    """法规名的**后缀**在查询里最长能匹配多少个字。

    要求匹配的是后缀，是因为用户用的简称总是全称的后缀；
    这条规则还顺手解决了一个歧义：

        《证券期货投资者适当性管理办法》      全称 14 字
        《证券期货投资者适当性管理办法》问答   全称 16 字，末尾是"问答"

    问"证券期货投资者适当性管理办法第二十九条"时，
    前者能整名匹配（14 字），而后者**任何后缀都匹配不上**——
    因为它的后缀都以"问答"结尾，而查询里没有"问答"。
    于是并列的情况自然消失了。
    """

    for length in range(len(title), MIN_TITLE_SUFFIX - 1, -1):
        suffix = title[-length:]
        if suffix in query:
            if length <= 3 and suffix in GENERIC_TAILS:
                return 0
            return length
    return 0


def parse_citation(query: str, documents: list[DocumentRef]) -> CitationQuery:
    """从问题里解析出（法规名，条号）。**纯函数**，不碰数据库。

    为什么用规则而不是让大模型解析：

    这个模式的形状非常固定——法规名不是带书名号就是全称的后缀，
    条号就是"第X条"。规则是确定的、可单测的、不花钱、没有幻觉。
    为一个人家已经写清楚的结构去调一次模型，只会换来延迟和不确定性。
    """

    result = CitationQuery()
    normalized = _normalize(query)

    # ---- 条号 ----
    tokens = find_article_tokens(query)
    result.article_candidates = tokens

    distinct: list[str] = []
    for token in tokens:
        if token not in distinct:
            distinct.append(token)

    if not distinct:
        result.reason = '问题里没有出现条号'
        return result

    if len(distinct) > 1:
        # 出现多个条号，通常是在问"这两条有什么区别"。
        # 这时候硬按某一条去直查，反而会漏掉另一半。宁可退回普通检索。
        result.reason = f'问题里出现了多个条号（{"、".join(distinct)}），未做精确直查'
        return result

    result.article_number = distinct[0]
    result.article_int = chinese_number_to_int(distinct[0])
    if result.article_int is None:
        result.reason = f'条号"{distinct[0]}"无法解析成数字'
        return result

    # ---- 法规名 ----
    best: tuple[int, DocumentRef] | None = None
    for document in documents:
        title = _clean_title(document.title or '')
        if len(title) < MIN_TITLE_SUFFIX:
            continue
        matched = _best_title_suffix(title, normalized)
        if matched == 0:
            continue
        # 匹配得越长越好；一样长时取标题更短的（更具体，避免被"…问答"这类长名抢走）
        if best is None or matched > best[0] or (
            matched == best[0] and len(title) < len(_clean_title(best[1].title))
        ):
            best = (matched, document)

    if best is None:
        result.reason = '问题里没有识别到已知法规名'
        return result

    result.document_matched_chars = best[0]
    result.document_id = best[1].id
    result.document_title = best[1].title
    result.reason = (
        f'识别到法规《{best[1].title}》（名称匹配 {best[0]} 字）'
        f'与条款 {result.article_number}'
    )
    return result


def load_document_refs(session: Session) -> list[DocumentRef]:
    """取出可以用来匹配法规名的文档清单。"""

    rows = session.execute(
        select(Document.id, Document.title, Document.filename).where(
            Document.status == 'indexed'
        )
    ).all()
    return [
        DocumentRef(id=row[0], title=row[1] or row[2].rsplit('.', 1)[0])
        for row in rows
    ]


def lookup_article(
    session: Session,
    *,
    document_id: str,
    article_int: int,
) -> list[dict[str, Any]]:
    """取出某个法规里某一条的**全部**切片。

    为什么取全部而不是最像的那一片：

    一条法规常被分页切成两片（后半段没有条号，靠 article_number 继承父条号）。
    只取一片，模型读到的是**半条法规**——而用户问"第X条怎么规定的"，
    要的就是完整那一条。

    返回结构刻意和向量/关键词两路保持一致，
    这样它可以直接混进最终结果，不需要上层再转换一次。
    """

    rows = session.execute(
        select(
            Chunk,
            Document.filename,
            Document.title,
            Document.legal_level,
            Document.validity,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.document_id == document_id,
            Chunk.article_number.is_not(None),
            Chunk.enabled.is_(True),
        )
        .order_by(Chunk.chunk_index)
    ).all()

    hits: list[dict[str, Any]] = []
    for chunk, filename, title, legal_level, validity in rows:
        # 中文数字 ↔ 阿拉伯数字的写法在查询和原文里常常不一致
        # （用户写"第29条"，法规写"第二十九条"），所以按**数值**比对，
        # 不按字符串比对。
        if chinese_number_to_int(chunk.article_number or '') != article_int:
            continue
        hits.append(
            {
                'chunk_id': chunk.id,
                'document_id': chunk.document_id,
                'filename': filename,
                'title': title,
                'page_number': chunk.page_number,
                'section_title': chunk.section_title,
                'splitter_name': chunk.splitter_name,
                'content_type': chunk.content_type,
                'text': chunk.content,
                'article_number': chunk.article_number,
                # 效力信息必须跟着走：提示词里有条硬规则——回答"是否违规"
                # 必须说明依据出自哪一层效力。精确命中是**最该带上这个信息**的结果，
                # 少了它，模型反而只能对着"来源: 某某.txt"去猜。
                'legal_level': legal_level,
                'validity': validity,
                'retrieval_source': 'exact',
                'retrieval_sources': ['exact'],
            }
        )

    logger.info(
        '[CITATION] 条款直查: document_id=%s 条号=%s 命中=%s 片',
        document_id,
        article_int,
        len(hits),
    )
    return hits


__all__ = [
    'CitationQuery',
    'DocumentRef',
    'load_document_refs',
    'lookup_article',
    'parse_citation',
]
