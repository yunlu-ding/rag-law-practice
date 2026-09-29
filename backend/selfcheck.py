"""自检脚本：一次性验证外部依赖是否真的能用。

为什么值得单独写一个脚本，而不是"跑一次看看":
这个系统依赖三个外部东西——百炼（向量化）、Milvus（向量库）、PostgreSQL。
任何一环出问题，表现出来都是"检索结果不对"这种含糊的现象。
把它们拆开逐个验证，出问题时就能立刻定位到是哪一环，
而不是在一个端到端流程里猜。

用法：
    cd vibe-rag/backend
    python selfcheck.py
    python selfcheck.py --index          # 顺带把库里所有文档重新索引
    python selfcheck.py --query "..."    # 顺带跑一次检索

退出码：0 表示全部通过，1 表示有环节失败。
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, '.')

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
# pymilvus 在每次 RPC 出错时都会打印整段 traceback，而其中很多是它可以自行处理的
# 情况（比如"索引还没建好"）。自检脚本要的是结论，所以把它压掉；
# 真出问题时我们自己的代码会把异常信息打出来。
logging.getLogger('pymilvus').setLevel(logging.CRITICAL)


def mask(value: str | None) -> str:
    if not value:
        return '（空）'
    if len(value) <= 10:
        return value[:2] + '***'
    return f'{value[:6]}***{value[-4:]}（长度 {len(value)}）'


def step(title: str) -> None:
    print(f'\n===== {title} =====')


def check_config() -> bool:
    step('1. 配置')
    from app.config import get_settings

    s = get_settings()
    print(f'  应用        : {s.app_name}')
    print(f'  阶段        : {s.build_stage}')
    print(f'  百炼 Key    : {mask(s.dashscope_api_key)}')
    print(f'  向量模型    : {s.embedding_model}')
    print(f'  向量库      : {s.milvus_uri or "（未配置）"}')
    print(f'  向量库 Key  : {mask(s.milvus_token)}')
    print(f'  集合名      : {s.milvus_collection}')
    print(f'  声明维度    : {s.milvus_dimension}')
    print(f'  数据库      : {"已配置" if s.postgres_dsn else "（未配置）"}')
    ok = bool(s.dashscope_api_key and s.milvus_uri and s.postgres_dsn)
    print(f'  → {"通过" if ok else "不通过：缺少必需配置"}')
    return ok


def check_embedding() -> bool:
    step('2. 百炼向量化（真实调用）')
    try:
        from app.core.embeddings import embed_query

        vector = embed_query('独立客观性要求会员避免利益冲突')
    except Exception as exc:  # noqa: BLE001
        print(f'  → 失败：{type(exc).__name__}: {exc}')
        return False

    from app.config import get_settings

    declared = get_settings().milvus_dimension
    print(f'  返回维度    : {len(vector)}')
    print(f'  声明维度    : {declared}')
    if len(vector) != declared:
        print(f'  → 不通过：维度不一致。请把 .env 里的 MILVUS_DIMENSION 改成 {len(vector)}，'
              f'并确认向量库里的集合也要重建。')
        return False
    print('  前 4 维     :', [round(x, 4) for x in vector[:4]])
    print('  → 通过')
    return True


def check_milvus() -> bool:
    step('3. 向量库连通性')
    try:
        store = __import__('app.core.vector_store', fromlist=['get_vector_store']).get_vector_store()
        collections = store.client.list_collections()
        print(f'  已连接，现有集合 {len(collections)} 个')
        store.ensure_collection()
        print(f'  目标集合    : {store.collection}（维度 {store.dimension}）')
        print(f'  当前条数    : {store.count()}')
        print('  → 通过')
        return True
    except Exception as exc:  # noqa: BLE001
        print(f'  → 失败：{type(exc).__name__}: {exc}')
        return False


def run_index() -> bool:
    step('4. 索引所有文档')
    from sqlalchemy import select

    from app.core.postgres import get_session_factory
    from app.models.document import Document
    from app.services.index_service import IndexService

    session_factory = get_session_factory()
    with session_factory() as db:
        documents = list(db.execute(select(Document)).scalars().all())
        if not documents:
            print('  知识库是空的，跳过')
            return True

        for document in documents:
            try:
                result = IndexService(db).index_document(document.id)
                print(f'  {document.filename}: {result["indexed"]}/{result["total"]} 片已索引')
                # 索引完也要把状态写回数据库。
                #
                # 这一步是补上的：第一版自检脚本只调了索引、没更新状态，
                # 结果文档在库里显示"切分完成"，而向量其实已经生成好了——
                # **状态和实际数据不一致**。
                # 更麻烦的是这种不一致看起来像"没做完"，会让人重复做一遍。
                from app.services.document_service import DocumentService

                DocumentService(db).update_status(
                    document.id,
                    status='indexed',
                    progress=100,
                    summary=f'已生成 {result["indexed"]} 条向量（由自检脚本完成）',
                )
            except Exception as exc:  # noqa: BLE001
                print(f'  {document.filename}: 失败 —— {type(exc).__name__}: {exc}')
                return False

    # 索引完再数一遍：这个数字应该等于所有文档的切片总数。
    # 如果明显偏大，说明库里积累了孤儿向量——
    # 这正是一次真实故障的表现（删除失败被静默跳过，多出 864 条），
    # 所以把它做成每次自检都会看到的数字，而不是等出问题再查。
    from app.core.vector_store import get_vector_store

    store = get_vector_store()
    print(f'  索引后集合内向量总数: {store.count()}')
    return True


def run_queries(queries: list[str]) -> bool:
    step('5. 向量检索（真实调用）')
    from app.services.index_service import vector_search

    for query in queries:
        print(f'\n  问题：{query}')
        try:
            hits = vector_search(query, top_k=3)
        except Exception as exc:  # noqa: BLE001
            print(f'    失败：{type(exc).__name__}: {exc}')
            return False
        if not hits:
            print('    没有命中任何切片（知识库可能是空的）')
            continue
        for hit in hits:
            source = hit.get('filename') or '?'
            page = hit.get('page_number') or '-'
            preview = (hit.get('text') or '').replace('\n', ' ')[:70]
            print(f'    [{hit.get("rank_vector")}] 分数 {hit.get("score"):.4f} | {source} 第{page}页 | {preview}')
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description='法规知识库自检')
    parser.add_argument('--index', action='store_true', help='顺带把所有文档重新索引')
    parser.add_argument('--query', action='append', default=[], help='顺带跑一次检索，可重复')
    args = parser.parse_args()

    ok = check_config()
    ok = check_embedding() and ok
    ok = check_milvus() and ok

    if args.index and ok:
        ok = run_index() and ok

    queries = args.query or (['收到客户礼物需要披露吗'] if ok else [])
    if queries and ok:
        ok = run_queries(queries) and ok

    step('结论')
    print('  → 全部通过' if ok else '  → 有环节失败，见上面输出')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
