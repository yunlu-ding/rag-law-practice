from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.deps import get_database
from app.rag.splitters import SPLITTER_DESCRIPTIONS, SPLITTER_REGISTRY
from app.schemas.chunk import ChunkItem, ChunkListResponse, SplitterOptionItem
from app.schemas.document import DocumentItem, DocumentListResponse, DocumentUploadResponse
from app.services.chunk_service import ChunkService
from app.services.document_service import (
    DocumentService,
    process_document,
    reindex_document,
)
from app.utils.storage import save_upload_file, sha256_of_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/documents', tags=['documents'])

SUPPORTED_TYPES = {'txt', 'md', 'pdf', 'docx'}


@router.post('/upload', response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(description='支持 txt / md / pdf / docx'),
    knowledge_base: str = Form(default='default'),
    on_conflict: str = Form(
        default='keep_both',
        description='同名文件已存在时怎么处理：keep_both=两份都留（默认），replace=覆盖旧版本',
    ),
    splitter: str = Form(
        default='auto',
        description='切分策略：auto=自动判断（默认）/ semantic=结构感知 / unstructured=按长度',
    ),
    db: Session = Depends(get_database),
) -> DocumentUploadResponse:
    """上传文档。

    这个接口只做三件事：**落盘、查重、登记**，然后立刻返回。
    解析交给后台任务——这是本节要解决的第一个问题：

    > 原来的做法是"收文件 → 同步解析 → 同步返回"，
    > 一本几百页的教材要几十秒，而前端早就超时了。
    > 用户看到的是"上传失败"，于是重试，于是在库里留下好几份同样的文件。
    >
    > 现在改成"接收即返回"：接口在毫秒级返回一个 document.id，
    > 前端拿着这个 id 轮询进度，能看到 queued → parsing → parsed 的真实推进。

    另外两件必须做的事：

    1. **幂等**：先算文件内容的 SHA-256，同一个知识库里已有相同内容就直接返回旧记录。
       用户重复提交多少次，都只会有一份。
    2. **同名冲突交给用户判断**：`on_conflict=replace` 才会覆盖旧版本。
       为什么不自动覆盖——系统无法区分"这是同一份文档的新版本"和
       "这是另一份碰巧同名的文档"（两份都叫《季度报告.pdf》的情况太常见了）。
       猜错的两个方向代价都很大：该覆盖没覆盖 → 用户拿到过期信息；
       不该覆盖覆盖了 → 用户丢掉一份资料，且不可逆。
       **判断不了的事不猜，交给能判断的人，并且把后果说清楚。**
    """

    stored_path, file_size = await save_upload_file(file)
    original_filename = file.filename or stored_path.name
    file_type = Path(original_filename).suffix.lower().lstrip('.')

    # 参数校验放在文件落盘之后，所以失败时必须把刚存的临时文件清掉，
    # 否则每次参数写错都会在磁盘上留一个垃圾文件。
    if file_type not in SUPPORTED_TYPES:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'暂不支持 .{file_type} 格式，当前支持：{"、".join(sorted(SUPPORTED_TYPES))}',
        )

    if on_conflict not in {'keep_both', 'replace'}:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'不支持的 on_conflict 取值：{on_conflict}（可选 keep_both / replace）',
        )

    if splitter != 'auto' and splitter not in SPLITTER_REGISTRY:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'不支持的切分策略：{splitter}（可选 auto / {" / ".join(SPLITTER_REGISTRY)}）',
        )

    service = DocumentService(db)

    # ---- 幂等检查 ----
    file_hash = sha256_of_file(stored_path)
    existing = service.find_by_hash(file_hash=file_hash, knowledge_base=knowledge_base)
    if existing is not None:
        stored_path.unlink(missing_ok=True)
        logger.info(
            '[UPLOAD] 命中幂等，未重复入库: filename=%s existing_id=%s',
            original_filename,
            existing.id,
        )
        return DocumentUploadResponse(
            document=DocumentItem.model_validate(existing),
            message=f'这个文件已经在知识库里了（原文件：{existing.filename}），没有重复入库',
            duplicated=True,
        )

    # ---- 同名覆盖（只在用户明确选择时才执行）----
    replaced_count = 0
    if on_conflict == 'replace':
        replaced_count = service.replace_by_filename(
            filename=original_filename,
            knowledge_base=knowledge_base,
        )

    # ---- 登记排队记录，立刻返回 ----
    document = service.create_pending_document(
        filename=original_filename,
        knowledge_base=knowledge_base,
        file_type=file_type,
        source_path=str(stored_path),
        file_size=file_size,
        file_hash=file_hash,
        splitter_name=splitter,
    )
    background_tasks.add_task(process_document, document.id)

    if replaced_count:
        message = f'已覆盖 {replaced_count} 份同名旧文档，正在后台处理新版本'
    else:
        message = '文件已接收，正在后台处理，可在文档列表里看进度'

    logger.info(
        '[UPLOAD] 已受理: id=%s filename=%s size=%s（接口在后台任务开始前就已返回）',
        document.id,
        original_filename,
        file_size,
    )
    return DocumentUploadResponse(document=DocumentItem.model_validate(document), message=message)


@router.get('', response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_database)) -> DocumentListResponse:
    """文档列表。

    前端靠这个接口轮询进度：只要有文档处于处理中状态，就每 3 秒拉一次；
    全部处理完自动停止，不会一直空转。
    """

    service = DocumentService(db)
    documents = service.list_documents()
    return DocumentListResponse(
        total=len(documents),
        items=[DocumentItem.model_validate(document) for document in documents],
    )


@router.get('/splitters/options', response_model=list[SplitterOptionItem])
def list_splitter_options() -> list[SplitterOptionItem]:
    """可选的切分策略，供前端下拉框使用。

    注意这个路由必须注册在 `/{document_id}` 之前，
    否则 "splitters" 会被当成一个文档 ID 去查库。
    """

    return [
        SplitterOptionItem(name=name, description=description)
        for name, description in SPLITTER_DESCRIPTIONS.items()
    ]


@router.get('/{document_id}', response_model=DocumentItem)
def get_document(document_id: str, db: Session = Depends(get_database)) -> DocumentItem:
    service = DocumentService(db)
    document = service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='文档不存在')
    return DocumentItem.model_validate(document)


@router.get('/{document_id}/chunks', response_model=ChunkListResponse)
def list_document_chunks(
    document_id: str,
    limit: int = 500,
    offset: int = 0,
    db: Session = Depends(get_database),
) -> ChunkListResponse:
    """返回指定文档的切片列表。"""

    document_service = DocumentService(db)
    if document_service.get_document(document_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='文档不存在')

    chunk_service = ChunkService(db)
    chunks = chunk_service.list_chunks(document_id=document_id, limit=limit, offset=offset)
    return ChunkListResponse(
        total=chunk_service.count_chunks(document_id=document_id),
        items=[ChunkItem.model_validate(chunk) for chunk in chunks],
    )


@router.post('/{document_id}/rechunk', response_model=DocumentUploadResponse)
def rechunk_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    splitter: str = Form(default='auto'),
    db: Session = Depends(get_database),
) -> DocumentUploadResponse:
    """换一种切分策略重新切分。

    这是**做对照实验的入口**，也是这个产品"可迭代"的体现：
    想知道"结构感知切分到底比按长度切好多少"，不能靠感觉，
    要能用同一个文档跑出两组数字来比。

    顺带说明为什么连解析也一起重做：
    结构感知切分的依据是解析阶段留下的结构标记，
    只重切不重解析会拿到不完整的信息。
    "重来一遍"在这里比"尽量少做事"更可靠。
    """

    if splitter != 'auto' and splitter not in SPLITTER_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'不支持的切分策略：{splitter}（可选 auto / {" / ".join(SPLITTER_REGISTRY)}）',
        )

    service = DocumentService(db)
    document = service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='文档不存在')
    if not document.source_path or not Path(document.source_path).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='源文件已不在服务器上，无法重新切分，请重新上传',
        )

    document.splitter_name = splitter
    db.commit()
    service.reset_for_reprocess(document_id)
    background_tasks.add_task(process_document, document_id)

    refreshed = service.get_document(document_id)
    logger.info('[RECHUNK] 已受理: id=%s splitter=%s', document_id, splitter)
    return DocumentUploadResponse(
        document=DocumentItem.model_validate(refreshed),
        message=f'正在用「{splitter}」重新切分，结果会覆盖原有切片',
    )


@router.post('/{document_id}/reindex', response_model=DocumentUploadResponse)
def reindex_document_endpoint(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_database),
) -> DocumentUploadResponse:
    """只补做向量化：跳过解析和切分，直接给现有切片生成向量。

    什么时候会用到它：向量化依赖外部服务（额度、限流、网络），
    是最容易失败的一步；而它失败时解析和切分的结果都还在。
    这时候让用户重新上传一遍是很不合理的——
    **失败时给出的出路应该尽量短，短到用户愿意去点。**
    """

    service = DocumentService(db)
    document = service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='文档不存在')
    if document.chunk_count <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='这份文档还没有切片，请先用「重新切分」走完解析和切分',
        )

    service.update_status(
        document_id,
        status='vectorizing',
        progress=75,
        summary='正在生成向量',
    )
    background_tasks.add_task(reindex_document, document_id)

    refreshed = service.get_document(document_id)
    logger.info('[REINDEX] 已受理: id=%s', document_id)
    return DocumentUploadResponse(
        document=DocumentItem.model_validate(refreshed),
        message='正在重新生成向量（不改动已有切片）',
    )


@router.post('/{document_id}/retry', response_model=DocumentUploadResponse)
def retry_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_database),
) -> DocumentUploadResponse:
    """重新处理一份失败或中断的文档。

    为什么需要它：处理失败时（接口超时、模型额度用尽、格式异常），
    用户不该被迫重新上传一遍——文件还在服务器上，直接重跑就行。

    **只有"失败"状态而没有"重试"入口，等于把问题丢回给用户。**
    这是本节要解决的第二个问题：失败必须给出路。
    """

    service = DocumentService(db)
    document = service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='文档不存在')

    # 任何状态都允许重新处理。
    # 理由：重新处理是幂等的（旧切片会被整体替换，不会产生重复数据），
    # 既然重复执行不会有什么坏处，就没有必要拦住用户——
    # 挡住用户的检查越少越好，除非它能防止真实的损失。

    if not document.source_path or not Path(document.source_path).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='源文件已不在服务器上，无法重试，请重新上传',
        )

    service.reset_for_reprocess(document_id)
    background_tasks.add_task(process_document, document_id)

    refreshed = service.get_document(document_id)
    logger.info('[UPLOAD] 已受理重试: id=%s', document_id)
    return DocumentUploadResponse(
        document=DocumentItem.model_validate(refreshed),
        message='已重新提交处理，可在文档列表里看进度',
    )


@router.delete('/{document_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, db: Session = Depends(get_database)) -> None:
    """删除文档（数据库记录 + 服务器上的源文件）。"""

    service = DocumentService(db)
    if not service.delete_document(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='文档不存在')
