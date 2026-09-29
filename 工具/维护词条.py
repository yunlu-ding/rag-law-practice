"""维护已编译词条的字段（触发词、状态等）。

为什么单独有一个"改词条"的入口：

词条是从 `工具/编译词条.py` 的提纲编译出来的，但**触发词这种字段是要调的**——
第一版我把"区别""差异"这类对比词也放进触发词，结果路由在任何对比题上都能匹配，
主题判据形同虚设。改这类字段不该重编一遍（那会重新调模型、还会覆盖人工核对过的正文）。

用法：
    python 工具/维护词条.py                     # 看看现在有什么
    python 工具/维护词条.py --sync-triggers     # 按提纲把触发词同步一遍
    python 工具/维护词条.py --review <slug>     # 标记为"人工已核对"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / '工具'))

from sqlalchemy import select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.models.wiki_entry import WikiEntry  # noqa: E402


def load_outline() -> dict[str, list[str]]:
    """从编译提纲里取"应有的触发词"。"""

    import importlib.util

    spec = importlib.util.spec_from_file_location('编译词条', ROOT / '工具' / '编译词条.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {request.slug: list(request.triggers) for request in module.REQUESTS}


def main() -> int:
    parser = argparse.ArgumentParser(description='维护词条')
    parser.add_argument('--sync-triggers', action='store_true', help='按提纲同步触发词')
    parser.add_argument('--review', default=None, help='把该 slug 标记为人工已核对')
    parser.add_argument(
        '--review-all',
        action='store_true',
        help='把所有**原文已逐字核实**的词条标记为人工已核对（会逐条列出正在批准什么）',
    )
    args = parser.parse_args()

    session_factory = get_session_factory()
    with session_factory() as session:
        entries = list(session.execute(select(WikiEntry)).scalars())

        if args.sync_triggers:
            outline = load_outline()
            for entry in entries:
                if entry.slug in outline:
                    entry.triggers = outline[entry.slug]
                    print(f'  触发词已同步：{entry.slug} → {outline[entry.slug]}')
            session.commit()
            print(f'共同步 {len(entries)} 条')

        if args.review:
            entry = session.execute(
                select(WikiEntry).where(WikiEntry.slug == args.review)
            ).scalar_one_or_none()
            if entry is None:
                print(f'没有这个词条：{args.review}')
                return 1
            entry.status = 'reviewed'
            session.commit()
            print(f'已标记为人工已核对：{entry.title}')

        if args.review_all:
            # 批量批准。条件是 **原文逐字核实通过**——
            # 也就是说"模型引的每一句原文都能在法规里找到"，
            # 这一条是机器验的，不是它自己说的。
            #
            # 但仍然逐条列出来：**批量操作最怕的是"不知道自己批准了什么"。**
            # 打印出来，至少让批准这件事是可见的。
            approved = 0
            for entry in entries:
                if entry.status == 'reviewed':
                    continue
                if not entry.original_verified:
                    print(f'  跳过（有原文未核实）：{entry.title}')
                    continue
                rows = [r for d in (entry.dimensions or []) for r in (d.get('rows') or [])]
                print(f'  批准：{entry.title}（{len(rows)} 行索引，原文全部核实）')
                entry.status = 'reviewed'
                approved += 1
            session.commit()
            print(f'共批准 {approved} 条')

        print()
        print(f'{"状态":<10} {"依据核实":<8} {"命中":<5} 词条')
        for entry in entries:
            print(
                f'{entry.status:<10} {str(entry.citation_verified):<8} '
                f'{entry.hit_count:<5} {entry.title}'
            )
            print(f'           触发词：{entry.triggers}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
