"""入库之后的端到端校验：确认数据真的进去了、检索真的能用、问答真的有依据。

为什么不能只看"入库脚本报 indexed"：

`indexed` 只说明**流程没抛异常**。它不能说明：

  - 向量真的写进了向量库（可能写到一半断了，只是没报错）
  - 关系库里的文档数和向量库里的条数对得上（孤儿向量/缺失向量）
  - 检索回来的切片真的来自这份语料
  - 问答引用的条文真的存在于原文里

这几件事每一件都要单独看一眼，所以这个脚本按"数据 → 检索 → 问答"三层递进地查。

用法：
    python 工具/入库后校验.py                # 数据 + 检索（不花钱）
    python 工具/入库后校验.py --with-qa      # 再加上问答（调大模型，花一点钱）
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
logging.getLogger('pypdf').setLevel(logging.ERROR)

from sqlalchemy import func, select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.core.vector_store import get_vector_store  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.rag.metadata import level_label  # noqa: E402
from app.services.qa_service import QaService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402

# 检索测试题。刻意选了三种不同的难度：
#   1. 条文本身直接问（应该一击命中）
#   2. 跨层级的问题（法律有原则、部门规章有细则，要能召回多层）
#   3. 语料里没有的问题（应该拒答，而不是编）
RETRIEVAL_QUERIES = [
    '证券期货投资者适当性管理办法第二十九条',
    '客户适当性管理怎么做',
    '境外市场的投资者适当性要求是什么',
]

QA_QUESTIONS = [
    '经营机构向普通投资者销售高风险产品时，需要履行哪些特别注意义务？',
    '境外市场（比如美国 SEC）的适当性规则是什么？',
]


def show_database() -> bool:
    print('=' * 78)
    print('第一层：关系库')
    print()

    session_factory = get_session_factory()
    with session_factory() as session:
        total_docs = session.execute(select(func.count()).select_from(Document)).scalar_one()
        total_chunks = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
        print(f'文档 {total_docs} 份，切片 {total_chunks} 个')
        print()

        rows = session.execute(
            select(
                Document.filename,
                Document.legal_level,
                Document.validity,
                Document.regulator,
                Document.effective_date,
                Document.chunk_count,
            ).order_by(Document.legal_level, Document.filename)
        ).all()

        print(f'{"层级":<8} {"效力":<8} {"发布机关":<14} {"施行":<11} {"片":>4}  文件')
        print('-' * 96)
        for filename, level, validity, regulator, effective, chunks in rows:
            print(
                f'{level_label(level):<8} {validity:<8} {(regulator or "—"):<14} '
                f'{str(effective or "—"):<11} {chunks:>4}  {filename}'
            )

        # 元数据完整度。缺字段不是错误，但要看得见——
        # "抽不到"和"没有这个信息"是两件事，前者需要人去补。
        print()
        print('元数据完整度：')
        for field, label in (
            (Document.regulator, '发布机关'),
            (Document.effective_date, '施行日期'),
            (Document.scope, '适用范围'),
            (Document.legal_level, '法的层级'),
            (Document.metadata_evidence, '抽取出处'),
        ):
            filled = session.execute(
                select(func.count()).select_from(Document).where(field.is_not(None))
            ).scalar_one()
            print(f'  {label:<6} {filled}/{total_docs}')

    return total_chunks > 0


def show_vector_store(expected: int | None = None) -> bool:
    print()
    print('=' * 78)
    print('第二层：向量库')
    print()

    store = get_vector_store()
    count = store.count()
    print(f'集合 {store.collection}：{count} 条向量')

    description = store.client.describe_collection(store.collection)
    fields = [field.get('name') for field in description.get('fields', [])]
    print(f'字段：{fields}')

    if expected is not None and count != expected:
        print()
        print(f'⚠️ 向量条数（{count}）和关系库切片数（{expected}）不一致。')
        print('   少了 = 有切片没写进向量库（检索会漏），多了 = 有孤儿向量（会召回已删内容）。')
        return False
    print('✅ 向量条数与关系库切片数一致')
    return True


def show_retrieval() -> None:
    print()
    print('=' * 78)
    print('第三层：检索')
    print()

    session_factory = get_session_factory()
    with session_factory() as session:
        service = RetrievalService(session)
        for query in RETRIEVAL_QUERIES:
            outcome = service.search(query=query, top_k=3, persist=False)
            print(f'问：{query}')
            if not outcome.hits:
                print('  （没有召回任何内容）')
            for rank, hit in enumerate(outcome.hits, start=1):
                snippet = ' '.join(str(hit.get('text') or '').split())[:70]
                print(
                    f'  {rank}. score={hit.get("score"):.4f} '
                    f'来源={hit.get("filename")} ｜ {snippet}'
                )
            print()


def show_qa() -> None:
    print('=' * 78)
    print('第四层：问答')
    print()

    session_factory = get_session_factory()
    with session_factory() as session:
        service = QaService(session)
        for question in QA_QUESTIONS:
            outcome = service.ask(question=question, top_k=5)
            print(f'问：{question}')
            if outcome.refused:
                print(f'  结论：拒答 —— {outcome.refusal_reason}')
            else:
                print(f'  结论：{outcome.conclusion}')
                print(f'  依据：{outcome.clause}')
                if outcome.reasoning:
                    print(f'  说明：{" ".join(outcome.reasoning.split())[:180]}')
                for citation in outcome.citations:
                    label = citation.get('label') or citation.get('filename')
                    print(f'    · {label}')
                if outcome.unknown_citations:
                    print(f'  ⚠️ 编造引用：{outcome.unknown_citations}')
            if outcome.error:
                print(f'  ⚠️ 错误：{outcome.error}')
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description='入库后端到端校验')
    parser.add_argument('--with-qa', action='store_true', help='额外跑问答（会调用大模型）')
    args = parser.parse_args()

    has_chunks = show_database()

    # 关系库里的切片数要传给向量库那层做对账
    session_factory = get_session_factory()
    with session_factory() as session:
        chunk_total = session.execute(select(func.count()).select_from(Chunk)).scalar_one()

    consistent = show_vector_store(chunk_total) if has_chunks else False
    show_retrieval()
    if args.with_qa:
        show_qa()

    print('=' * 78)
    print('结论：', end='')
    if has_chunks and consistent:
        print('数据层一致，检索可用。')
        return 0
    print('数据层存在问题，看上面的告警。')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
