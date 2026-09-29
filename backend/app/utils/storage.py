from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings

READ_CHUNK_SIZE = 1024 * 1024  # 1MB


def ensure_upload_dir() -> Path:
    """确保上传目录存在。"""

    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def sanitize_filename(filename: str) -> str:
    """生成安全文件名，避免路径穿越和特殊字符问题。

    注意：这里只用于**存储**。展示给用户的仍然是原始文件名——
    用户看的是自己的文件名，不是被清洗过的版本。
    """

    base_name = Path(filename).name.strip()
    if not base_name:
        return 'unnamed.txt'
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', base_name)
    return safe_name or 'unnamed.txt'


def build_storage_path(filename: str) -> Path:
    """生成唯一存储路径。

    用随机名而不是原文件名，避免两件事：
    1. 同名文件互相覆盖；
    2. 原始文件名里的特殊字符、中文编码问题传导到文件系统。
    """

    upload_dir = ensure_upload_dir()
    suffix = Path(sanitize_filename(filename)).suffix
    return upload_dir / f'{uuid.uuid4().hex}{suffix}'


def sha256_of_file(file_path: str | Path) -> str:
    """计算文件内容的 SHA-256。

    分块读取而不是一次性读进内存：教材类 PDF 可能上百 MB，
    一次性读进来既慢又占内存，而这个操作在每次上传时都会发生。
    """

    digest = hashlib.sha256()
    with Path(file_path).open('rb') as handle:
        while True:
            block = handle.read(READ_CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


async def save_upload_file(upload_file: UploadFile) -> tuple[Path, int]:
    """把上传文件分块落盘，并在写入过程中就检查体积上限。

    为什么在写入过程中就检查、而不是写完再看：
    如果等写完才判断超限，磁盘上已经躺了一个超大文件，
    而攻击者可以用这个方式把磁盘写满。
    边写边判断，超限立刻中断并删掉半成品。
    """

    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='上传的文件没有文件名',
        )

    settings = get_settings()
    target_path = build_storage_path(upload_file.filename)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    file_size = 0
    try:
        with target_path.open('wb') as output_file:
            while True:
                chunk = await upload_file.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f'文件超过上限 {settings.max_upload_size_mb} MB。'
                            f'如果确实需要处理更大的文件，改 backend/.env 里的 MAX_UPLOAD_SIZE_MB。'
                        ),
                    )
                output_file.write(chunk)
    except Exception:
        # 失败时不要留下半截文件。否则用户重试后目录里会堆积无法追溯的垃圾。
        target_path.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()

    return target_path, file_size
