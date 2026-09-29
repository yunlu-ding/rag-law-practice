from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Boolean, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class QaLog(Base, TimestampMixin):
    """问答记录。

    这张表是"评测"的数据来源：判断题的结论、引用、是否拒答、
    有没有出现编造的引用，全都记在这里。
    没有它，评测就只能靠人工一条条重跑；有了它，
    可以按结论类型、是否存在未知引用等条件筛选出可疑样本，
    把人工判卷的力气花在最值得看的地方。
    """

    __tablename__ = 'qa_log'

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment='记录主键',
    )
    session_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment='会话 ID，用于把多轮追问串起来；为空表示单轮问答',
    )
    question: Mapped[str] = mapped_column(Text, nullable=False, comment='用户问题')
    conclusion: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment='结论：违反 / 不违反 / 无法判断 / 说明'
    )
    clause: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment='涉及的条款编号'
    )
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True, comment='理由')
    assumption: Mapped[str | None] = mapped_column(Text, nullable=True, comment='判断前提')

    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False, comment='已还原的引用（含文件名与页码）'
    )
    unknown_citations: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        comment='模型引用了不存在的片段 ID —— 这是一次可检测的编造',
    )
    # 「条款」字段里那些**引用的片段中根本没出现**的条款编号。
    # 和 unknown_citations 是一对孪生指标，防的是两种不同的编造：
    #   前者编片段编号，后者编条款编号——而后者看起来更像依据，更隐蔽。
    unsupported_clauses: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        comment='「条款」字段里无依据的条款编号（引用的片段里没有它）',
    )
    no_citation_answer: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment='实质作答（违反/不违反/说明）却没有任何引用 —— 结论无法逐条核对',
    )

    refused: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, comment='是否走了拒答分支'
    )
    refusal_reason: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment='拒答原因（检索分数不足 / 无法判断）'
    )
    refusal_threshold: Mapped[float | None] = mapped_column(
        nullable=True, comment='本次生效的拒答阈值，便于复现'
    )
    # 下面两列是"这一次回答凭什么"的证据，和 refused 那个布尔值是两件事：
    #   refusal_kind —— 拒答属于哪一类（系统故障 / 库里没有这一条 /
    #                   一条都没召回 / 分数不足）。混在一起的话，
    #                   "拒答率"这个数字里会同时装着"系统坏了"和"真的没有"。
    #   evidence     —— 答了的话，依据有多硬（条款直查 / 词条 / 重排分达标 /
    #                   降级模式）。它让"可信回答率"可以按证据强度拆开看。
    refusal_kind: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment='拒答种类，便于按原因分组统计'
    )
    evidence: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment='本次回答的依据强度'
    )
    evidence_note: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment='附在答案旁的提醒（降级 / 词条缺失）'
    )

    parse_ok: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, comment='模型输出是否被成功解析为结构化结果'
    )
    retrieval_failed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment='检索链路是否故障导致无法作答（与"知识库里没有依据"是两回事）',
    )
    model: Mapped[str | None] = mapped_column(String(64), nullable=True, comment='使用的生成模型')
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment='总耗时')
    retrieval_log_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, comment='关联的检索日志，可回溯当时召回了什么'
    )
