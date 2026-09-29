from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.llm import generate
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.wiki_entry import WikiEntry
from app.rag.splitters.legal import chinese_number_to_int, starts_with_article

logger = logging.getLogger(__name__)

"""Wiki 词条的编译。

**编译，不是生成。** 这两个词的差别是这一层能不能信的关键：

  - "生成"是把模型当作者，它写什么就是什么；
  - "编译"是把模型当**整理者**——它只能拿给定的条文原文去组织，
    而且它要明确说出每一句结论的依据是哪一条。

所以这里的流程是硬性的：

    给定主题 + 指定要覆盖的条款
      → 从关系库取出这些条款的**原文**
      → 交给模型，只允许基于这些原文写
      → 要求它逐条给出依据（法规名 + 条款号）
      → **拿依据回库核对**：查不到的就标出来

最后那一步是这一层的安全阀。词条是一条"权威结论"，
写错了会污染所有相关问答，而用户**没有东西可以对照**——
RAG 最差只是召回一段不完美的原文，点开就能看出不对。
所以"依据是否真实存在"必须由代码验，不能由模型自己保证。

⚠️ 编译出来的东西是**草稿**（`status='draft'`），必须人核对过才能参与作答。
"""


@dataclass
class CompileRequest:
    """一次编译要什么。"""

    slug: str
    title: str
    topic: str
    legal_lines: list[str]
    triggers: list[str]
    # 要覆盖的条款：[(文件名, 条款号), ...]
    sources: list[tuple[str, str]] = field(default_factory=list)
    # 额外要求（比如"重点写两者的差异"）
    instruction: str = ''


COMPILE_SYSTEM_PROMPT = """你在为一份金融监管法规知识库**做索引**——不是写分析，不是下结论。

## 你的任务

给定一个主题和若干条款原文，把它们按**可对照的维度**分组，
每个维度下逐条列出各条款对应的内容。

## 硬规则

1. **每一行都必须有 `original`：一句从给定条文里原样截取的原文。**
   可以截取片段（中间省略用 …… 表示），但**不能改写、不能概括、
   不能把不相邻的句子拼在一起**。这一条是硬底线——
   后续会拿你给的 original 回原文里逐字比对，对不上就作废。

2. **不写判断。** 不许出现"两者区别在于""可以看出""相比而言"
   "应当注意"这类话。你只负责把原文摆到对应的维度下，
   **对比和分析是下一步的事**。

3. 某个条款在某个维度下没有对应内容，就**不要列它**。不要为了凑齐而硬填。

4. `confidence` 填 high / medium / low，含义是"extracted 能不能直接从
   original 里读出来"：
     - high   —— 原文里明明白白写着；
     - medium —— 需要一点推断（比如原文说"可以并处"，你写成"罚款"）；
     - low    —— 你自己也拿不准。
   **如实填。** 填 low 不会扣分，把不确定的写成 high 才会让整张表失去可信度。

5. 维度名要**具体**："罚款幅度"比"处罚"具体，"义务主体"比"主体"具体。

6. **不许写"元数据维度"。** 下面这些一律不要单独成维度：

       "法律效力层级" "适用范围" "发布机关" "文号" "施行日期" "文档信息"

   它们是**文档的属性**，系统里已经结构化存好了（`legal_level` / `validity` /
   `regulator` / `doc_number` / `effective_date`），根本不是条文内容。

   这一条是被真实失败逼出来的：模型给"法律效力层级"这个维度填的 `original`
   写成了 **"中华人民共和国证券法（law，effective）"** ——那是系统字段值，
   不是任何一句原文。它的后果不只是这一行作废，而是**整条词条的
   `original` 逐字校验通不过**，于是这条词条永远进不了可以作答的状态。

   判断标准很简单：**你写进 `original` 的每一个字，都必须能在
   我给你的条文原文里连着找到。** 找不到，就说明那不是一个维度，
   而是一条已经不在这儿的信息。

## 输出格式

只输出一个 JSON 对象，不要有任何额外文字、不要用 markdown 代码块包裹：

{
  "dimensions": [
    {
      "name": "维度名，如：罚款幅度",
      "rows": [
        {
          "source": "法规简称 + 条款号，如：证券法 第一百九十八条",
          "extracted": "从这句原文里读出来的内容，尽量短",
          "original": "……原样截取的原文片段……",
          "confidence": "high"
        }
      ]
    }
  ]
}
"""


def build_compile_prompt(request: CompileRequest, passages: list[dict[str, Any]]) -> str:
    """把要覆盖的条文原文拼成编译输入。"""

    lines = [
        f'【主题】{request.topic}',
        f'【适用业务线】{"、".join(request.legal_lines) or "未指定"}',
    ]
    if request.instruction:
        lines.append(f'【额外要求】{request.instruction}')
    lines.append('')
    lines.append('【给定条文】')
    for index, passage in enumerate(passages, start=1):
        lines.append('')
        lines.append(f'条文 {index}')
        lines.append(
            f'法规：{passage["title"]}（{passage["legal_level"]}，'
            f'{passage["validity"]}）'
        )
        lines.append(f'条款：{passage["article_number"]}')
        lines.append(f'原文：{passage["text"]}')
    lines.append('')
    lines.append('请按上述格式输出词条 JSON。')
    return '\n'.join(lines)


def load_passages(session: Session, sources: list[tuple[str, str]]) -> tuple[list[dict], list[str]]:
    """按（文件名，条款号）取出条文原文。返回 (取到的, 没取到的)。"""

    passages: list[dict[str, Any]] = []
    missing: list[str] = []

    for filename, article in sources:
        document = session.execute(
            select(Document).where(Document.filename == filename)
        ).scalar_one_or_none()
        if document is None:
            missing.append(f'{filename}（语料里没有这份文件）')
            continue

        wanted = chinese_number_to_int(article)
        chunks = session.execute(
            select(Chunk)
            .where(Chunk.document_id == document.id, Chunk.article_number.is_not(None))
            .order_by(Chunk.chunk_index)
        ).scalars().all()

        picked = [
            chunk
            for chunk in chunks
            if chinese_number_to_int(chunk.article_number or '') == wanted
        ]
        if not picked:
            missing.append(f'{document.title} {article}（该条不存在）')
            continue

        # 一条法规常被分页切成几片，全部取上——词条要基于完整那一条，
        # 只给半条会让模型去"补全"，而补全就是编。
        passages.append(
            {
                'filename': filename,
                'title': document.title or filename,
                'legal_level': document.legal_level or '未标注',
                'validity': document.validity or 'effective',
                'article_number': picked[0].article_number,
                'text': '\n'.join(chunk.content for chunk in picked),
            }
        )

    return passages, missing


def _parse_json(raw: str) -> dict[str, Any]:
    """模型经常套一层代码块或加一句开场白，容错掏一下。"""

    text = (raw or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z]*\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


_ELLIPSIS = re.compile(r'…+|\.{3,}|。{3,}')

# PDF 的分页痕迹。它们会**插在句子中间**，让本来完全正确的引用匹配失败。
#
# 实测（这也是写这段的原因）：《证券法》第八十九条被分页切成两片——
#
#     chunk 101：……证券公司不能证明的，应当承担相应的—５１—
#     chunk 102：赔偿责任。
#
# 模型引用的是完整的那一句（"应当承担相应的赔偿责任"），**它引对了**，
# 但两个分句之间横着一个页码和一行页眉，逐字比对必然失败。
# 于是 model 被记成"引了不存在的原文"，整条词条的 `citation_verified`
# 变成 False —— 一条内容完全正确的词条，被排版噪声判成了不可信。
#
# 所以校验前先把这两类噪声去掉。**这里去掉的只是噪声，不是判断标准**：
# 真正的底线（"每个字都能在原文里连着找到"）没有放松。
_PAGE_FURNITURE = (
    # 页码：全角或半角数字夹在破折号之间，前后可以有空白
    re.compile(r'[—–\-]\s*[０-９0-9]{1,4}\s*[—–\-]'),
    # 公报的页眉。这一条是**针对具体语料**的，写在这里是因为
    # 《国务院公报》那份 PDF 每一页都带这一行，而它恰好落在切页处。
    # 更正确的修法是在**解析层**去掉页眉页脚（那样所有下游都不用管它），
    # 但那要重新入库 1281 个切片。先在这里兜住，并把这件事记在已知缺口里。
    re.compile(r'全国人民代表大会常务委员会公报[０-９0-9·．\.]*'),
)


def _normalize_for_verify(text: str) -> str:
    """比对前归一化：去空白、去分页噪声。"""

    cleaned = re.sub(r'[\s\u3000]+', '', text or '')
    for pattern in _PAGE_FURNITURE:
        cleaned = pattern.sub('', cleaned)
    return cleaned


def verify_originals(
    dimensions: list[dict[str, Any]],
    passages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """逐行核对 `original` 是不是真的出自给定条文。

    这是索引型词条的安全阀，而且**它是机器验的，不靠模型自律**。

    做法：把 original 按省略号切成几段，**每一段**都要能在条文原文里
    逐字找到（去掉空白之后）。切段是必要的——模型经常写成
    "第一条……第三款……"，那是两处原文拼起来的，整串匹配必然失败，
    但它其实是合规的引用方式。

    校验结果写回每一行的 `verified` 字段。于是人工的活就变成：
    **只看 verified=true 的行**，判断"extracted 能不能从 original 读出来"——
    不用再去翻法规原文，因为原文已经被机器确认过了。
    """

    haystack = _normalize_for_verify(' '.join(p['text'] for p in passages))
    all_ok = True

    for dimension in dimensions:
        for row in dimension.get('rows') or []:
            original = str(row.get('original') or '')
            segments = [
                _normalize_for_verify(part)
                for part in _ELLIPSIS.split(original)
                if len(_normalize_for_verify(part)) >= 4
            ]
            row['verified'] = bool(segments) and all(
                segment in haystack for segment in segments
            )
            if not row['verified']:
                all_ok = False
                logger.warning(
                    '[WIKI] original 未能在原文中找到: %s | %s',
                    row.get('source'),
                    original[:60],
                )
    return dimensions, all_ok


def render_index(dimensions: list[dict[str, Any]]) -> str:
    """把索引结构渲染成可读的 Markdown（给人工核对看，也给提示词用）。"""

    lines: list[str] = []
    for dimension in dimensions:
        lines.append(f'### {dimension.get("name")}')
        for row in dimension.get('rows') or []:
            mark = '✅' if row.get('verified') else '⚠️未核实'
            lines.append(
                f'- **{row.get("source")}** → {row.get("extracted")}'
                f'（{row.get("confidence")}｜{mark}）'
            )
            lines.append(f'  > {row.get("original")}')
        lines.append('')
    return '\n'.join(lines).strip()


def verify_citations(
    session: Session,
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """逐条核对词条给出的依据是否真的存在于语料。

    这是"编造引用检测"用在编译这一步。模型说"依据《证券法》第八十八条"，
    我们就回库查有没有这一条——**模型的自述不能当证据**。
    """

    verified: list[dict[str, Any]] = []
    for citation in citations or []:
        name = str(citation.get('法规') or '').strip()
        article = str(citation.get('条款') or '').strip()
        if not name or not article:
            continue

        wanted = chinese_number_to_int(article)
        exists = False
        if wanted is not None:
            rows = session.execute(
                select(Chunk.article_number)
                .join(Document, Document.id == Chunk.document_id)
                .where(Document.title.like(f'%{name}%'), Chunk.article_number.is_not(None))
            ).scalars().all()
            exists = any(chinese_number_to_int(row or '') == wanted for row in rows)

        verified.append(
            {'文档': name, '条款': article, '核实': exists}
        )
        if not exists:
            logger.warning('[WIKI] 依据未核实到: %s %s', name, article)
    return verified


def compile_entry(session: Session, request: CompileRequest) -> WikiEntry:
    """编译一条词条（草稿）。"""

    passages, missing = load_passages(session, request.sources)
    if not passages:
        raise ValueError(f'一条条文都没取到，无法编译：{missing}')

    result = generate(
        system_prompt=COMPILE_SYSTEM_PROMPT,
        user_prompt=build_compile_prompt(request, passages),
        temperature=0.2,
    )
    parsed = _parse_json(result.content)

    # 同 slug 已存在就**整条替换**，而不是报唯一键冲突。
    # 重编是常规操作（改提纲、换提示词都要重编），
    # 让调用方每次先自己删一遍是没必要的负担。
    # 代价是：重编会丢掉人工对这些字段的修改——
    # 所以"人工已核对"的状态不在这里保留，它得由人重新确认。
    previous = session.execute(
        select(WikiEntry).where(WikiEntry.slug == request.slug)
    ).scalar_one_or_none()
    if previous is not None:
        session.delete(previous)
        session.commit()

    # ---- 索引型：验 original，不验"结论" ----
    #
    # 这一步的核对对象变了。旧版验的是"模型引用的条款是否存在"——
    # 但那只保证"引对了法规"，保证不了"引的那句话真的在那一版里"。
    # 新版验的是**每一行 original 能不能在给定条文里逐字找到**，
    # 这才是这一步唯一需要保证的事：**原文是真的。**
    #
    # 至于"提取得准不准"（extracted 能不能从 original 读出来），
    # 那是人的活——但人只需要看机器验过的 original，不用翻法规。
    dimensions, all_verified = verify_originals(
        list(parsed.get('dimensions') or []), passages
    )
    body = render_index(dimensions)

    # 依据条款仍然从"覆盖了哪些条文"推出来——索引型词条的依据就是它的来源清单，
    # 不需要模型另报一遍（它报了反而多一个可能出错的地方）。
    citations = [
        {
            '文档': passage['title'],
            '条款': passage['article_number'],
            '核实': True,
        }
        for passage in passages
    ]

    entry = WikiEntry(
        slug=request.slug,
        title=request.title,
        topic=request.topic,
        legal_lines=request.legal_lines,
        triggers=request.triggers,
        summary=None,
        body=body,
        dimensions=dimensions,
        original_verified=all_verified,
        citations=citations,
        citation_verified=all_verified,
        status='draft',
        compiled_by='dashscope',
        compiled_from={
            'sources': [f'{p["filename"]} {p["article_number"]}' for p in passages],
            'missing': missing,
            'tokens': result.total_tokens,
        },
        content_hash=hashlib.sha256(body.encode('utf-8')).hexdigest(),
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)

    logger.info(
        '[WIKI] 词条已编译: slug=%s 条文=%s 依据=%s 全部核实=%s tokens=%s',
        request.slug,
        len(passages),
        len(citations),
        all_verified,
        result.total_tokens,
    )
    return entry


__all__ = [
    'CompileRequest',
    'compile_entry',
    'load_passages',
    'verify_citations',
]
