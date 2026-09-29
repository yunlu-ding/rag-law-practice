from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.postgres import get_session_factory
from app.models.document import Document
from app.rag.chunker import build_chunks
from app.rag.bm25_index import get_bm25_index
from app.rag.loader import load_document
from app.rag.metadata import extract_metadata
from app.rag.metadata_overrides import apply_overrides
from app.rag.splitters import SPLITTER_REGISTRY
from app.services.chunk_service import ChunkService
from app.services.index_service import IndexService
from app.utils.storage import sha256_of_file

logger = logging.getLogger(__name__)

# 还在处理中的状态。启动恢复要靠它判断"哪些文档是半路被中断的"。
IN_PROGRESS_STATES = ('queued', 'parsing', 'splitting', 'vectorizing')

# 整条流水线的进度映射。
# 现在只走到"解析完成"（50%），切分与向量化在后续步骤接入。
# 为什么现在就把整条链路的百分比定好：
# 这样以后加切分、加向量化时，进度语义不用改，
# 前端的进度条也不会出现"今天到 100 明天只到 30"这种让人困惑的变化。
PROGRESS = {
    'queued': 0,
    'parsing': 5,
    'parsed': 50,
    'splitting': 55,
    'chunked': 70,
    'vectorizing': 75,
    'indexed': 100,
}


class DocumentService:
    """文档相关的业务操作。

    这一层只负责"做什么"，不负责"什么时候做"——
    什么时候做（立即还是后台）由 API 层决定。
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ---------- 查询 ----------

    def get_document(self, document_id: str) -> Document | None:
        return self.db.execute(select(Document).where(Document.id == document_id)).scalar_one_or_none()

    def list_documents(self, *, limit: int = 200) -> list[Document]:
        statement = select(Document).order_by(Document.updated_at.desc()).limit(limit)
        return list(self.db.execute(statement).scalars().all())

    def find_by_hash(self, *, file_hash: str, knowledge_base: str) -> Document | None:
        """按内容指纹查重。

        查重口径是"同一个知识库里、内容完全相同"。
        注意它回答的问题只是"这是不是同一个文件"，
        不回答"这是不是同一份文档的新版本"——后者需要用户来判断，
        理由见 on_conflict 参数的设计说明。
        """

        statement = select(Document).where(
            Document.file_hash == file_hash,
            Document.knowledge_base == knowledge_base,
        )
        return self.db.execute(statement).scalars().first()

    def find_by_filename(self, *, filename: str, knowledge_base: str) -> list[Document]:
        statement = select(Document).where(
            Document.filename == filename,
            Document.knowledge_base == knowledge_base,
        )
        return list(self.db.execute(statement).scalars().all())

    # ---------- 写入 ----------

    def create_pending_document(
        self,
        *,
        filename: str,
        knowledge_base: str,
        file_type: str,
        source_path: str,
        file_size: int,
        file_hash: str,
        splitter_name: str | None = None,
        legal_level: str | None = None,
        title: str | None = None,
    ) -> Document:
        """登记一条"排队中"的记录。

        这一步做完就可以立刻返回给前端了——
        用户马上得到反馈（"我收到了，正在处理"），
        而真正的解析在后台跑。

        `legal_level` / `title` 是"先于解析就知道"的元数据：
        批量入库时代码从语料目录读出层级（人工核对过），
        网页上传时如果用户在下拉框里选了层级也走这里。
        其余元数据（文号、日期、适用范围）必须解析完正文才知道，
        那部分在 process_document 里补。
        """

        document = Document(
            knowledge_base=knowledge_base,
            filename=filename,
            file_type=file_type,
            source_path=source_path,
            file_size=file_size,
            file_hash=file_hash,
            splitter_name=splitter_name or 'auto',
            legal_level=legal_level,
            title=title,
            status='queued',
            progress=PROGRESS['queued'],
            chunk_count=0,
            summary='已接收，排队等待处理',
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        logger.info(
            '[DOC] 已登记待处理文档: id=%s filename=%s size=%s hash=%s',
            document.id,
            filename,
            file_size,
            file_hash[:12],
        )
        return document

    def replace_by_filename(self, *, filename: str, knowledge_base: str) -> int:
        """删掉同知识库下的同名旧文档，返回删掉的份数。

        ⚠️ 这个操作**只应该由用户在明确选择"覆盖"之后触发**，
        不能自动执行。原因在 API 层有完整说明。
        """

        old_documents = self.find_by_filename(filename=filename, knowledge_base=knowledge_base)
        removed = 0
        for old_document in old_documents:
            if self.delete_document(old_document.id):
                removed += 1
        if removed:
            logger.info(
                '[DOC] 已覆盖同名旧文档: filename=%s 份数=%s knowledge_base=%s',
                filename,
                removed,
                knowledge_base,
            )
        return removed

    def update_status(
        self,
        document_id: str,
        *,
        status: str,
        progress: int | None = None,
        summary: str | None = None,
        **extra_fields: object,
    ) -> Document | None:
        """更新处理状态。

        只有一个地方改状态，而不是散落在各处直接赋值——
        状态机一旦散开，就很难回答"这个文档到底经历过什么"。
        """

        document = self.get_document(document_id)
        if document is None:
            return None

        document.status = status
        document.progress = PROGRESS.get(status, progress if progress is not None else document.progress)
        if progress is not None:
            document.progress = progress
        if summary is not None:
            document.summary = summary
        for field_name, value in extra_fields.items():
            setattr(document, field_name, value)

        self.db.commit()
        self.db.refresh(document)
        return document

    def reset_for_reprocess(self, document_id: str) -> bool:
        """把一份文档重置回排队状态，准备重新处理。

        为什么重试前要重置而不是直接再跑一遍：
        上一次可能在写数据的中途失败了（比如向量化到一半接口报错），
        库里会留下写了一半的内容。不清理就重跑，结果是重复内容进库。
        """

        document = self.get_document(document_id)
        if document is None:
            return False

        document.status = 'queued'
        document.progress = PROGRESS['queued']
        document.chunk_count = 0
        document.summary = '已重新排队，等待处理'
        self.db.commit()
        logger.info('[DOC] 已重置待重试: id=%s', document_id)
        return True

    def delete_document(self, document_id: str) -> bool:
        """删除文档：向量 + 源文件 + 数据库记录。

        清理顺序很重要：**先删外部存储，再删数据库记录**。
        反过来的话，一旦中途失败，数据库里已经没有记录，
        而文件和向量成了没人认领的孤儿——查不到、也删不掉。

        ⚠️ 这里曾经漏掉了向量，而且后果很隐蔽：

        漏删向量时，文档在页面上消失了、切片也没了，**看起来删干净了**。
        但向量还在库里，于是检索仍然会把它召回来——
        用户看到一条引用，点开发现对应的文档不存在。
        这个现象在检索调试台里表现为"来源指向一个已经删掉的文件"。

        触发路径不止删除按钮：重新上传同名文件走 replace_by_filename、
        批量重入库走 delete_document ——**所有删除路径都会经过这里**，
        所以修在这一层，而不是在每个调用点各补一次。

        另一个刻意的选择：**向量删除失败就抛异常，中止整个删除。**
        "降级优于中断"只适用于体验层；数据一致性上，宁可让用户看到一次报错，
        也不要留下一个永远清不掉的孤儿向量。
        """

        document = self.get_document(document_id)
        if document is None:
            return False

        source_path = document.source_path
        filename = document.filename

        try:
            from app.services.index_service import IndexService

            IndexService(self.db).delete_document_vectors(document_id)
        except Exception:
            logger.exception('[DOC] 删除向量失败，已中止删除以免留下孤儿: id=%s', document_id)
            raise

        self.db.execute(delete(Document).where(Document.id == document_id))
        self.db.commit()
        # 文档没了，它的切片也随外键级联删掉了，关键词索引同样要重建
        get_bm25_index().mark_dirty(f'document_deleted:{document_id}')

        if source_path:
            try:
                Path(source_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning('[DOC] 源文件删除失败（记录已删除）: path=%s error=%s', source_path, exc)

        logger.info('[DOC] 已删除文档: id=%s filename=%s', document_id, filename)
        return True


# ---------------------------------------------------------------------------
# 后台处理
# ---------------------------------------------------------------------------


def process_document(document_id: str) -> None:
    """后台处理一份文档。

    为什么这里要**自己开一个数据库会话**，而不是复用请求里的那个：
    后台任务是在接口返回之后才执行的，那时请求的会话已经关掉了。
    用已关闭的会话会报错——而且报错发生在后台，前端只会看到一个
    永远停在"排队中"的文档，很难查。
    """

    session_factory = get_session_factory()
    with session_factory() as db:
        service = DocumentService(db)
        document = service.get_document(document_id)
        if document is None:
            logger.warning('[PROCESS] 文档已不存在，跳过: id=%s', document_id)
            return

        source_path = document.source_path or ''
        filename = document.filename

        if not source_path or not Path(source_path).exists():
            service.update_status(
                document_id,
                status='failed',
                progress=0,
                summary='源文件不存在，无法处理，请重新上传',
            )
            logger.error('[PROCESS] 源文件缺失: id=%s path=%s', document_id, source_path)
            return

        try:
            service.update_status(
                document_id,
                status='parsing',
                progress=PROGRESS['parsing'],
                summary='正在解析文档',
            )

            loaded = load_document(Path(source_path))

            # 解析失败的另一种形态：没抛异常，但什么都没抽出来。
            # 这种"静默失败"比抛异常更危险——文件看起来入库了，实际什么都搜不到。
            if loaded.char_count == 0:
                service.update_status(
                    document_id,
                    status='failed',
                    progress=0,
                    summary='解析完成但未提取到任何文本（可能是扫描件或不支持的格式）',
                    page_count=loaded.page_count,
                    char_count=0,
                )
                logger.error('[PROCESS] 解析结果为空: id=%s filename=%s', document_id, filename)
                return

            report = {
                'parser': loaded.parser_name,
                'section_count': len(loaded.sections),
                'warnings': loaded.warnings,
                'section_types': sorted(
                    {str(section.metadata.get('section_type')) for section in loaded.sections}
                ),
            }

            # ---- 法规元数据 ----
            #
            # 放在"解析完、切分前"：元数据是**文档级**的，与怎么切无关；
            # 而它要赶在切分之前，是因为切分出来的切片要带上法规名称
            # （BM25 靠它把"《证券法》第八十八条"这种问法接住）。
            metadata = extract_metadata(
                filename=filename,
                text=loaded.full_text,
                legal_level=document.legal_level,
            )
            # 自动抽取之后立刻盖上人工核对的结论。
            # 顺序不能反：人工值优先，而它只在少数几个字段上生效，
            # 其余字段仍然用自动抽取的结果。
            metadata = apply_overrides(metadata, filename)
            report['metadata'] = {
                'title': metadata['title'],
                'legal_level': metadata['legal_level'],
                'regulator': metadata['regulator'],
                'validity': metadata['validity'],
            }

            service.update_status(
                document_id,
                status='parsed',
                progress=PROGRESS['parsed'],
                summary=_build_parsed_summary(loaded),
                page_count=loaded.page_count,
                char_count=loaded.char_count,
                parse_report=report,
                title=metadata['title'],
                legal_level=metadata['legal_level'],
                level_rank=metadata['level_rank'],
                regulator=metadata['regulator'],
                doc_number=metadata['doc_number'],
                scope=metadata['scope'],
                issued_date=_as_date(metadata['issued_date']),
                effective_date=_as_date(metadata['effective_date']),
                validity=metadata['validity'],
                content_hash=metadata['content_hash'],
                fetched_at=datetime.now(timezone.utc),
                metadata_evidence=metadata['evidence'],
            )
            logger.info(
                '[PROCESS] 解析完成: id=%s filename=%s parser=%s chars=%s '
                '层级=%s 效力=%s 发布机关=%s warnings=%s',
                document_id,
                filename,
                loaded.parser_name,
                loaded.char_count,
                metadata['legal_level'],
                metadata['validity'],
                metadata['regulator'],
                loaded.warnings,
            )

            # ---- 切分 ----
            service.update_status(
                document_id,
                status='splitting',
                progress=PROGRESS['splitting'],
                summary='正在切分',
            )

            # 'auto' 交给自动判断：由每一段自己有没有结构标记来决定切法。
            requested = document.splitter_name
            preferred = None if (not requested or requested == 'auto') else requested
            if preferred and preferred not in SPLITTER_REGISTRY:
                preferred = None

            chunking = build_chunks(loaded, preferred_splitter=preferred)
            chunk_count = ChunkService(db).replace_chunks(
                document_id=document_id,
                records=chunking.records,
            )

            report['chunking'] = chunking.stats
            service.update_status(
                document_id,
                status='chunked',
                progress=PROGRESS['chunked'],
                summary=_build_chunked_summary(chunking.stats),
                chunk_count=chunk_count,
                parse_report=report,
            )
            logger.info(
                '[PROCESS] 切分完成: id=%s filename=%s 切片=%s 残句率=%s',
                document_id,
                filename,
                chunk_count,
                chunking.stats.get('mid_word_ratio'),
            )

            # ---- 向量化 ----
            #
            # 单独包一层 try，是为了让失败信息更有用：
            # 向量化失败时解析和切分的结果都还在，用户不需要重新上传，
            # 走 /reindex 就能只补跑这一步。把这个区别说清楚，
            # 比统一报一句"处理失败"有用得多。
            try:
                service.update_status(
                    document_id,
                    status='vectorizing',
                    progress=PROGRESS['vectorizing'],
                    summary='正在生成向量',
                )
                indexed = IndexService(db).index_document(
                    document_id,
                    progress_callback=lambda done, total: _report_vector_progress(
                        document_id, done, total
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception('[PROCESS] 向量化失败: id=%s', document_id)
                service.update_status(
                    document_id,
                    status='failed',
                    progress=PROGRESS['chunked'],
                    summary=(
                        f'切分已完成（{chunk_count} 片），但向量化失败：'
                        f'{type(exc).__name__}: {exc}。'
                        f'修复后可以直接重试，不需要重新上传。'
                    ),
                )
                return

            service.update_status(
                document_id,
                status='indexed',
                progress=PROGRESS['indexed'],
                summary=f'{_build_chunked_summary(chunking.stats)}；已生成 {indexed["indexed"]} 条向量',
                chunk_count=chunk_count,
            )
            logger.info(
                '[PROCESS] 全部完成: id=%s filename=%s 切片=%s 向量=%s',
                document_id,
                filename,
                chunk_count,
                indexed['indexed'],
            )
        except Exception as exc:  # noqa: BLE001
            # 捕获所有异常：后台任务一旦抛出去，就没人接了，
            # 文档会永远停在"正在解析"。宁可把失败原因写进记录让用户看见。
            service.update_status(
                document_id,
                status='failed',
                progress=0,
                summary=f'处理失败：{type(exc).__name__}: {exc}',
            )
            logger.exception('[PROCESS] 处理失败: id=%s filename=%s', document_id, filename)


def _build_parsed_summary(loaded) -> str:
    """拼一条人能一眼看懂的解析结论。"""

    parts = [f'解析完成（{loaded.parser_name}）']
    if loaded.page_count:
        parts.append(f'{loaded.page_count} 页')
    parts.append(f'{loaded.char_count} 字')
    parts.append(f'{len(loaded.sections)} 个结构单元')
    if loaded.warnings:
        parts.append(f'{len(loaded.warnings)} 条提示')
    return '，'.join(parts)


def _as_date(value: object) -> date | None:
    """把抽取出来的日期字符串转成 date。

    为什么要多这一步：抽出来的日期是 `'2017-07-01'` 这样的字符串，
    而模型里的字段是 Date 类型。字符串交给 psycopg 也有机会成功，
    但那是"靠驱动宽容"，不是"类型对得上"——真出了问题时
    报错会出现在数据库层，离源头很远。
    """

    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _build_chunked_summary(stats: dict) -> str:
    """拼一条切分结论。

    特意把**残句率**放进去，因为它是切片质量最硬的指标：
    残句率高，说明模型读到的常常是半句话。
    """

    if not stats or not stats.get('chunk_count'):
        return '切分完成，但没有产生任何切片'
    ratio = float(stats.get('mid_word_ratio') or 0.0) * 100
    return (
        f'切分完成：{stats["chunk_count"]} 片，'
        f'平均 {stats["avg_length"]} 字，最长 {stats["max_length"]} 字，'
        f'残句率 {ratio:.1f}%'
    )


def _report_vector_progress(document_id: str, done: int, total: int) -> None:
    """上报向量化进度。

    为什么这里要**另开一个数据库会话**，而不复用主流程那个：
    主流程正在按批提交切片和向量，如果进度回调和它共用会话，
    会把只写了一半的状态提前提交出去。
    单独开会话只更新 document 表的两个字段，互不干扰。

    这个细节在实际故障里很要紧：共用一个会话时表现是
    "进度条走到了 60%，但报错之后数据全没了"——
    看起来像进度造假，实际是事务边界搞错了。
    """

    percent = PROGRESS['vectorizing'] + int(
        (PROGRESS['indexed'] - PROGRESS['vectorizing']) * done / max(total, 1)
    )
    try:
        session_factory = get_session_factory()
        with session_factory() as db:
            DocumentService(db).update_status(
                document_id,
                status='vectorizing',
                progress=min(percent, PROGRESS['indexed'] - 5),
                summary=f'正在生成向量：{done}/{total} 片',
            )
    except Exception as exc:  # noqa: BLE001
        # 进度上报失败不该影响主流程——它只是给人看的。
        logger.warning('[PROGRESS] 上报失败（不影响主流程）: %s', exc)


def reindex_document(document_id: str) -> None:
    """只补做向量化，不重新解析和切分。

    为什么单独开这个入口：
    向量化是最容易失败的一步——它依赖外部服务（额度、限流、网络）。
    而它失败时解析和切分的结果都还在，重跑一遍纯属浪费。
    **失败时给用户的出路应该尽量短，短到用户愿意点。**
    """

    session_factory = get_session_factory()
    with session_factory() as db:
        service = DocumentService(db)
        try:
            service.update_status(
                document_id,
                status='vectorizing',
                progress=PROGRESS['vectorizing'],
                summary='正在生成向量',
            )
            result = IndexService(db).index_document(
                document_id,
                progress_callback=lambda done, total: _report_vector_progress(
                    document_id, done, total
                ),
            )
            service.update_status(
                document_id,
                status='indexed',
                progress=PROGRESS['indexed'],
                summary=f'已生成 {result["indexed"]} 条向量',
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception('[REINDEX] 向量化失败: id=%s', document_id)
            service.update_status(
                document_id,
                status='failed',
                progress=PROGRESS['chunked'],
                summary=f'向量化失败：{type(exc).__name__}: {exc}',
            )


def resume_interrupted_documents() -> int:
    """服务启动时，把上次没处理完的文档重新捡起来。

    为什么必须有这一步：
    后台任务跑在进程内存里，服务一重启（开发时改代码就会重启）任务就没了。
    没有这一步，那条文档会永远停在"正在解析"——
    用户看到一个永远不会动的进度条，比直接报错更糟，
    因为他不知道该等还是该重传。
    """

    try:
        session_factory = get_session_factory()
    except Exception:  # noqa: BLE001
        # 数据库还没配置，谈不上恢复
        return 0

    with session_factory() as db:
        service = DocumentService(db)
        statement = select(Document).where(Document.status.in_(IN_PROGRESS_STATES))
        stuck_documents = list(db.execute(statement).scalars().all())

        if not stuck_documents:
            return 0

        recoverable: list[str] = []
        for document in stuck_documents:
            if document.source_path and Path(document.source_path).exists():
                service.update_status(
                    document.id,
                    status='queued',
                    progress=PROGRESS['queued'],
                    summary='服务重启后自动恢复，重新排队',
                )
                recoverable.append(document.id)
            else:
                service.update_status(
                    document.id,
                    status='failed',
                    progress=0,
                    summary='服务重启后源文件已不存在，无法恢复，请重新上传',
                )

    for document_id in recoverable:
        # 用守护线程而不是阻塞启动流程：
        # 启动阶段不应该因为一份大文件没解析完就卡住。
        threading.Thread(
            target=process_document,
            args=(document_id,),
            daemon=True,
            name=f'resume-{document_id[:8]}',
        ).start()

    logger.info(
        '[RECOVER] 启动恢复: 扫描到 %s 份未完成，其中 %s 份重新提交处理',
        len(stuck_documents),
        len(recoverable),
    )
    return len(recoverable)
