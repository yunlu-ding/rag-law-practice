from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Boolean, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin

"""Wiki 词条。

为什么要有这一层，而不是只靠检索：

有一类问题，答案**不存在于任何一个片段里**。

    问："证券和期货在适当性义务上的法律规定有什么区别"
    《证券法》第八十八条说了证券这一边，《期货和衍生品法》第五十条说了期货那一边，
    但"两者的区别"本身，语料里没有哪一段写过。

检索是"找最像的几段"，它面对这类问题会失灵——实测过三次：
扩候选、拆查询、强行保底，都是 77% 上下，最好的一次还掉了 1 个点。
因为要的东西不是"几段文字"，而是**一份对照**。

Wiki 这一层的职责就是：**把跨文档的关系预先算好、存下来**。
它不是"更好的检索"，它是检索能力边界之外的那一块。

⚠️ 一条硬规矩：**词条必须带依据条款，而且依据要能查得到。**

Wiki 的风险在于它是一条"权威结论"——写错了、过期了，会污染所有相关问答，
而用户**没有东西可以对照**（RAG 最差也只是召回一段不完美的原文，点开就能看出不对）。

所以词条不能是模型生成后就用的自由文本，它得：

  - 落在某个状态上（`status`）：草稿 / 已核对 / 已过期；
  - 带上 `citations`，每条都指向语料里真实存在的（法规、条款）；
  - 编译时校验这些指向**真的存在**，查不到的标出来等人核。
"""


class WikiEntry(Base, TimestampMixin):
    """一条 Wiki 词条。"""

    __tablename__ = 'wiki_entry'

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment='词条主键',
    )
    slug: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
        index=True,
        comment='稳定标识（英文/拼音），用于引用与路由',
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, comment='词条标题')
    topic: Mapped[str] = mapped_column(
        String(200), nullable=False, comment='主题（编译时的提纲标题，用于溯源）'
    )

    # 适用业务线。路由时用得上：问"证券和期货的区别"要落到同时覆盖两者的词条，
    # 而不是随便一条讲适当性的。
    legal_lines: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True, comment='适用业务线：证券 / 期货 / 基金'
    )
    # 触发问法的关键词。规则化路由的判据——先不用意图分类模型。
    triggers: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True, comment='触发词/问法，用于路由匹配'
    )

    summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment='一句话结论（定调用），长度控制在两行内'
    )
    body: Mapped[str] = mapped_column(Text, nullable=False, comment='词条正文（Markdown）')

    # 索引型词条的结构化内容。
    #
    # 这是"两步走"的第一步的产物：**只做索引，不做分析**。
    # 形如：
    #   [{"name": "处罚主体",
    #     "rows": [{"source": "证券法 第一百九十八条",
    #               "extracted": "证券监督管理机构",
    #               "original": "……由证券监督管理机构……",
    #               "confidence": "high",
    #               "verified": true}]}]
    #
    # 为什么要有 original 这一列——它是这一层的安全阀：
    # **每一条提取都必须能指回一句原文**，而且那句原文是不是真的存在，
    # **机器可以逐字验**（拿它去条文里做子串匹配）。
    # 于是人工只需要看"提取能不能从原文里读出来"，不用去翻法规原文。
    #
    # 这也把"模型编造"这件事压缩到了最小的空间：
    # 它可以提取得不完整、可以分错维度，但**编不出一句不存在的原文**。
    dimensions: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
        comment='索引型结构：[{name, rows:[{source, extracted, original, confidence, verified}]}]',
    )
    original_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment='所有 original 是否都能在对应条文里逐字找到（机器验，不靠模型自律）',
    )

    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        comment='依据条款：[{文档, 标题, 条款, 核实}]。核实=false 表示编译时没在库里查到',
    )
    citation_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment='全部依据是否都已在语料中核实到。**只有核实过的词条才允许参与作答**',
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default='draft',
        nullable=False,
        index=True,
        comment='draft（草稿）/ reviewed（人工已核对）/ stale（已过期）',
    )
    compiled_by: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment='编译所用模型'
    )
    compiled_from: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, comment='编译来源：用了哪些文档与条款、什么时候编的'
    )
    content_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment='正文 SHA-256，用于检测词条是否被改过'
    )
    hit_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment='被路由命中的次数（用于判断哪些词条真有人问）'
    )
