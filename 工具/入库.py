"""把 语料/ 目录下的法规批量入库。

设计上的一个关键决定：**批量入库不复用一套自己的流程，而是调用网页上传用的
同一个 process_document。**

理由很实际：如果批量走一条路、上传走另一条路，两条路会慢慢长歪——
改了一处的切分策略，另一处忘了改；上传能出的元数据，批量入库没有。
这类"两条路"最终一定会变成"一个功能两种行为"，而且很难查，
因为两边看起来都"能用"。

它做的事情只有三件（剩下的全交给流水线）：

  1. 从**目录名**读出法的层级（人工核对过，比让正则猜准得多）；
  2. 把文件复制进 storage/uploads，并登记一条 document 记录；
  3. 调用 process_document，等它跑完，汇报结果。

幂等性：同一份文件（按字节 SHA-256 判断）已经入库且状态是 indexed，
就跳过，不重复花 embedding 的钱。要强制重跑用 --force。

⚠️ 会花钱：向量化要调百炼的 embedding 接口。
脚本启动时会先算出行数，让人知道大概的规模。

用法：
    python 工具/入库.py                        # 演练，只列要做什么
    python 工具/入库.py --apply                # 真正入库
    python 工具/入库.py --apply --clear-db     # 先清空旧文档再入库
    python 工具/入库.py --apply --only 证券法  # 只入库文件名含"证券法"的
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

# Windows 控制台默认按 GBK 解码，而这个脚本会打印 ✅/⚠️ 这类符号——
# 不加保护的话，它会**直接崩在打印那一步**，而崩溃点常常在干完活之后
# （评测跑完了、钱花完了，明细一条都没落盘）。详见 工具/修控制台编码.py。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

logging.getLogger('pypdf').setLevel(logging.ERROR)

from sqlalchemy import delete, select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.core.vector_store import get_vector_store  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.rag.metadata import LEGAL_LEVEL_BY_FOLDER, level_label  # noqa: E402
from app.services.document_service import (  # noqa: E402
    DocumentService,
    process_document,
)
from app.utils.storage import build_storage_path, sha256_of_file  # noqa: E402

CORPUS = ROOT / '语料'
KNOWLEDGE_BASE = 'regulations'

# 不参与入库的文件。
SKIP_NAMES = {'语料覆盖范围.txt'}
SKIP_PREFIXES = ('~$', '.')


def iter_corpus() -> list[tuple[str, Path]]:
    """遍历语料，返回 (层级目录名, 文件路径)。

    层级来自目录名。这是刻意选择"信人而不是信算法"：
    目录是人工核对过的，而正则推出的层级只能当建议。
    """

    items: list[tuple[str, Path]] = []
    for folder in sorted(CORPUS.iterdir()):
        if not folder.is_dir():
            continue
        # 只处理**层级目录**。
        #
        # `语料/` 下除了五个层级目录，还有一个 `_原始件/`——放原始下载文件和
        # 已经被替换掉的旧版本。它不是语料，不该入库。
        # 用"目录名必须在层级表里"来判，比维护一份黑名单稳：
        # 以后再加存档目录，不用改代码。
        if folder.name not in LEGAL_LEVEL_BY_FOLDER:
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.name in SKIP_NAMES or path.name.startswith(SKIP_PREFIXES):
                continue
            items.append((folder.name, path))
    return items


def clear_database(session) -> int:
    """清空关系库里的文档与切片，**同时清掉它们在向量库里的向量**。

    ⚠️ 这里必须显式删向量，不能用裸 SQL 删表了事。

    踩过一次：清完库重新入库之后，向量库里是 2,022 条而实际只需要 1,012 条——
    每一片都有一份重复。原因是清库只删了关系库，向量库没人管，
    而新入库的文档拿到了**新的 UUID**，于是"按 document_id 删旧向量"这一步
    删的是不存在的 ID，旧向量就这么留下来了。

    表现是检索结果里同一条出现两次，而且**不报错**。
    """

    document_ids = [
        row[0] for row in session.execute(select(Document.id)).all()
    ]

    store = get_vector_store()
    removed_vectors = 0
    for document_id in document_ids:
        removed_vectors += store.delete_by_document(document_id)

    removed_chunks = session.execute(delete(Chunk)).rowcount
    removed_documents = session.execute(delete(Document)).rowcount
    session.commit()
    print(
        f'已清空：{removed_documents} 份文档，{removed_chunks} 个切片，'
        f'{removed_vectors} 条向量'
    )
    return removed_documents


def main() -> int:
    parser = argparse.ArgumentParser(description='法规语料批量入库')
    parser.add_argument('--apply', action='store_true', help='真正执行；不加则只演练')
    parser.add_argument('--clear-db', action='store_true', help='入库前清空现有文档')
    parser.add_argument('--force', action='store_true', help='忽略幂等检查，强制重跑')
    parser.add_argument(
        '--prune',
        action='store_true',
        help='删掉库里那些**语料目录已经不再包含**的文档，让库和目录保持一致',
    )
    parser.add_argument('--only', default=None, help='只处理文件名包含该关键字的文件')
    args = parser.parse_args()

    # 完整的语料清单。**任何"判断库里该有什么"的逻辑都必须基于它**，
    # 而不是基于下面那个被 --only 过滤过的列表。
    #
    # 这里踩过一个代价不小的坑：--prune 一开始用的是过滤后的 items，
    # 于是 `--only 某文件 --prune` 会认为"语料里只应该有这一份"，
    # 把另外 20 份全删了。**拿过滤后的输入去当完整的事实，是这类错误的通用形状。**
    all_items = iter_corpus()
    items = (
        [item for item in all_items if args.only in item[1].name] if args.only else all_items
    )

    if not items:
        print('语料目录里没有待入库的文件。')
        return 1

    session_factory = get_session_factory()

    print(f'知识库标识：{KNOWLEDGE_BASE}')
    print(f'待处理文件：{len(items)} 份')
    print()
    print(f'{"层级":<8} {"发布机关":<10} 文件')
    for folder_name, path in items:
        level = LEGAL_LEVEL_BY_FOLDER.get(folder_name)
        label = level_label(level) if level else f'⚠️未知目录({folder_name})'
        print(f'{label:<8} {"":<10} {path.name}')
    print()

    if not args.apply:
        print('演练模式：什么都没做。确认要入库就加 --apply。')
        print('注意：向量化会调用百炼的 embedding 接口，产生费用。')
        return 0

    if args.clear_db:
        with session_factory() as session:
            clear_database(session)
        print()

    if args.prune:
        # 把库里"语料目录已经没有的文件"清掉。
        #
        # 为什么需要它：语料是会变的——删掉一份不该收录的文件、
        # 换掉一份旧版本，都是常规操作。而入库脚本只会**增加和更新**，
        # 不会发现"这个文件已经从目录里消失了"。
        # 结果是库里留着一份语料目录里不存在的文档，它照样会被检索到、
        # 被引用，而用户去目录里根本找不到它。
        expected = {path.name for _folder, path in all_items}
        with session_factory() as session:
            service = DocumentService(session)
            stale = [
                document
                for document in service.list_documents(limit=1000)
                if document.knowledge_base == KNOWLEDGE_BASE
                and document.filename not in expected
            ]
            for document in stale:
                print(f'[清理] {document.filename}（语料目录里已不存在）')
                service.delete_document(document.id)
            if stale:
                print(f'共清理 {len(stale)} 份')
                print()

    results: list[tuple[str, str, str]] = []

    for folder_name, path in items:
        level = LEGAL_LEVEL_BY_FOLDER.get(folder_name)
        file_hash = sha256_of_file(path)

        with session_factory() as session:
            service = DocumentService(session)

            existing = service.find_by_hash(file_hash=file_hash, knowledge_base=KNOWLEDGE_BASE)
            if existing and existing.status == 'indexed' and not args.force:
                results.append((path.name, 'skipped', f'已入库（{existing.chunk_count} 片）'))
                print(f'[跳过] {path.name}（内容未变化）')
                continue

            # ---- 按**文件名**清掉旧记录，而不是按内容哈希 ----
            #
            # 这里踩过一次，后果很隐蔽：原来只按 file_hash 找旧记录，
            # 而 file_hash 是**文件字节**的指纹。换语料、改解析规则、
            # 重新生成某个文件之后，字节变了、指纹对不上，
            # 于是"找不到旧记录"→ 新建一份 → **同名文档在库里出现两份**。
            #
            # 表现是检索结果里同一份法规出现两遍，而且两遍的内容还不一样
            # （一份是旧的、一份是新的）。这比"报错"难查得多。
            #
            # 正确的口径是：批量入库时，**（知识库，文件名）才是主键**。
            # 内容哈希是用来判断"要不要重新处理"的，不是用来判断"这是不是同一份文件"的。
            for old in service.find_by_filename(
                filename=path.name, knowledge_base=KNOWLEDGE_BASE
            ):
                # 走 delete_document 而不是自己写 delete：
                # 它会顺手删掉向量和服务器上的旧源文件。
                # 绕开的话，每重试一次就留下一条孤儿向量和一份孤儿文件。
                service.delete_document(old.id)

            # 复制进统一的上传目录，让入库的文件和网页上传的文件
            # 在系统里长得一模一样——删除、重试、重建索引的代码都不用分情况。
            target = build_storage_path(path.name)
            shutil.copy2(path, target)

            document = service.create_pending_document(
                filename=path.name,
                knowledge_base=KNOWLEDGE_BASE,
                file_type=path.suffix.lower().lstrip('.'),
                source_path=str(target),
                file_size=path.stat().st_size,
                file_hash=file_hash,
                legal_level=level,
            )
            document_id = document.id

        print(f'[处理] {path.name} ...', end='', flush=True)
        process_document(document_id)

        with session_factory() as session:
            finished = session.execute(
                select(Document).where(Document.id == document_id)
            ).scalar_one_or_none()
            if finished is None:
                results.append((path.name, 'error', '记录消失'))
                print(' 记录消失')
                continue
            status = finished.status
            summary = finished.summary or ''
            results.append((path.name, status, f'{finished.chunk_count} 片'))
            print(f' {status}｜{finished.chunk_count} 片｜{summary[:60]}')

    print()
    print('=' * 78)
    print('入库结果')
    print()
    counted: dict[str, int] = {}
    for name, status, _detail in results:
        counted[status] = counted.get(status, 0) + 1
        print(f'  {status:<8} {name}')
    print()
    print(f'汇总：{counted}')

    if counted.get('failed'):
        print()
        print('有失败项。失败原因写在 document.summary 里，修好之后可以直接重跑——')
        print('脚本会识别出没跑完的记录并重来，不需要手工清理。')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
