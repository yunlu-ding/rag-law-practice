from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import BigInteger, Date, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    """文档主表。

    只保存"文档级"信息，不保存切分后的正文（那是 chunk 表的事）。
    这样文档管理、切片管理、检索调试三个模块可以各自独立工作。

    几个字段的设计意图值得说明：

    - `file_hash`：文件内容的指纹，作为**幂等键**。没有它，用户看到上传超时后
      重试一次，就会在库里留下两份同样的文件。
    - `status` + `progress`：处理是异步的，前端必须能知道"现在到哪一步了"。
      只有状态没有进度，用户看到的是一动不动的转圈；只有进度没有状态，
      用户不知道慢在哪一步。
    - `parse_report`：解析质检结果（空白页比例、超长行比例等）。
      存它是为了**让缺口可见**——解析得不好要能看出来，而不是静默入库。

    ---- 法规元数据 ----

    法规场景有一整类问题是通用问答里不存在的：**这段文字的效力是什么。**
    同一条规则写在《证券法》里和写在协会自律规则里，约束力完全不同；
    一份征求意见稿和一份正式发布的办法，回答时该不该引用也完全不同。
    所以下面这组字段不是装饰，它们是**答案的一部分**：

    - `legal_level` / `level_rank`：法的层级与位阶。
      位阶要落成整数，因为"冲突时以上位法为准"必须能被代码执行。
    - `validity`：现行有效 / 征求意见稿 / 已废止。
      ⚠️ 命名刻意避开 `status`——那个名字已经被"处理流水线状态"占用了
      （queued/parsing/.../indexed）。两者语义完全不同，
      共用一个名字会让"这份文件为什么搜不到"变成一个查不清的问题。
    - `content_hash`：**解析后正文**的 SHA-256。
      与 `file_hash`（文件字节的 SHA-256）分工不同：
      前者检测"内容变没变"，后者检测"是不是同一个文件"。
      同一份法规重新排版后字节全变、正文不变——这时不该重复入库。
    - `fetched_at`：这份版本是什么时候抓的。
      "只保留现行有效版本"这个决定有个副产物：库里的"现行"
      只是在抓取那一刻成立。没这个字段，回答里就没法说明依据的截止时间。
    - `metadata_evidence`：每个抽取字段的出处。
      元数据是正则猜出来的，必然有猜错的时候。
      带上出处，人工核对就只需要核对，不用重读全文。
    """

    __tablename__ = 'document'

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment='文档主键（UUID）',
    )
    knowledge_base: Mapped[str] = mapped_column(
        String(100),
        default='default',
        nullable=False,
        comment='所属知识库名称，为多知识库隔离预留',
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False, comment='原始文件名')
    file_type: Mapped[str] = mapped_column(String(32), nullable=False, comment='文件类型：pdf/docx/md/txt')
    source_path: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment='文件在服务器上的存储路径（重试与重建索引都依赖它）'
    )
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment='文件大小（字节）')
    file_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment='文件内容 SHA-256 指纹，幂等键',
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default='queued',
        nullable=False,
        index=True,
        comment='处理状态：queued/parsing/parsed/splitting/chunked/vectorizing/indexed/failed',
    )
    progress: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment='整条流水线的进度百分比 0-100（解析 5-45，切分 55-70，向量化 75-95）',
    )

    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True, comment='页数，适用于 PDF')
    char_count: Mapped[int | None] = mapped_column(Integer, nullable=True, comment='解析出的字符总数')
    chunk_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment='已生成的切片数量'
    )

    summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment='状态说明；处理失败时用来记录失败原因'
    )
    parse_report: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, comment='解析质检报告（空白页比例、超长行比例、警告项等）'
    )
    splitter_name: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment='用户选择的切分策略；auto 表示由系统按文档结构自动判断',
    )

    # ---------------- 法规元数据 ----------------

    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment='文件标题；优先取人工命名的文件名',
    )
    legal_level: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        index=True,
        comment='法的层级：law/admin_regulation/department_rule/normative_document/self_regulation',
    )
    level_rank: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment='位阶 1-5（法律=1 … 自律规则=5）；冲突时取数值小者',
    )
    regulator: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment='发布机关'
    )
    doc_number: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment='文号，如"证监会令第〔2020〕130号"'
    )
    scope: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        comment='适用范围标签数组（证券/期货/基金/衍生品）；用数组而非枚举，'
        '因为一部法规可以同时适用多个业务线',
    )
    issued_date: Mapped[date | None] = mapped_column(Date, nullable=True, comment='公布日期')
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True, comment='施行日期')
    expiry_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
        comment='失效日期，可空。⚠️ 为空不等于现行有效——法规多数是被废止而非到期，'
        '效力判断只能看 validity',
    )
    validity: Mapped[str] = mapped_column(
        String(32),
        default='effective',
        nullable=False,
        index=True,
        comment='效力状态：effective（现行有效）/ draft（征求意见稿）/ '
        'superseded（已被修订）/ repealed（已废止）',
    )
    source_url: Mapped[str | None] = mapped_column(
        String(1000), nullable=True, comment='原文来源链接，用于引用溯源'
    )
    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment='解析后正文的 SHA-256；用于变更检测（区别于 file_hash 的字节指纹）',
    )
    fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment='入库时间；回答里用来说明"依据的是截至哪一日的版本"',
    )
    supersedes: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True, comment='本文件替代了哪些旧文件；第一版留空'
    )
    superseded_by: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment='被哪份新文件取代；第一版留空'
    )
    key_dates: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True, comment='过渡期、意见截止日期等关键时点；第一版留空'
    )
    metadata_evidence: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment='每个元数据字段的抽取出处，供人工核对',
    )
