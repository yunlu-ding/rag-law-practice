from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class UsageDaily(Base, TimestampMixin):
    """每日用量。

    一行一天，按日期做唯一约束。
    这张表的用途有两个：给限额做判断依据、给成本分析提供原始数据。

    为什么记 token 而不记金额：金额需要单价，而单价会变。
    **写死在代码里的单价迟早过期，过期之后"预算封顶"就变成了假的安全感。**
    token 是客观的，需要换算成本时用真实账单一除就行。
    """

    __tablename__ = 'usage_daily'

    usage_date: Mapped[date] = mapped_column(
        Date, primary_key=True, comment='日期（每天一行）'
    )
    qa_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment='当日问答次数'
    )
    retrieval_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment='当日独立检索次数'
    )
    blocked_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment='当日被限额拦截的次数'
    )
    prompt_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment='当日输入 token 数'
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment='当日输出 token 数'
    )
