"""打印索引型词条，供人工核对。

核对分工是**两步走设计**的核心：

    机器已经验过的  → `original` 是不是真的出自给定条文（逐字比对）
    只剩人要看的    → `extracted` 能不能从 `original` 里读出来

所以这个脚本把两件事并排摆出来：每行给出 source / extracted / original，
**并且只标出机器验过的**（✅ 表示原文确认存在）。
人不需要再去翻法规——原文已经被确认过了。

重点看两类行：
  · ⚠️ 未核实 —— 原文都对不上，后面的提取更不用看；
  · confidence=medium/low —— 模型自己说"需要推断"的地方，是最容易出错的。

用法：
    python 工具/核对词条.py
    python 工具/核对词条.py --slug penalty-comparison
    python 工具/核对词条.py --only-problem    # 只看有问题的行
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

try:
    from sqlalchemy import select  # noqa: E402
except ModuleNotFoundError:
    # 用错解释器是**最容易发生、又最没必要排查**的一类失败：
    # 系统的 Python 里没装项目的依赖，跑起来就是一个 ModuleNotFoundError 的堆栈，
    # 看起来像"代码坏了"，实际只是"跑它的不是项目环境的那个 python"。
    #
    # 所以这里不抛异常，直接把正确的命令打出来。
    _python = ROOT.parent / '.venv' / 'Scripts' / 'python.exe'
    print('这个脚本要用**项目的虚拟环境**跑，不是系统里的 Python。\n')
    print('正确的命令：\n')
    print(f'    & "{_python}" "{Path(__file__).resolve()}"\n')
    if not _python.exists():
        print(f'（没找到虚拟环境：{_python}）')
    raise SystemExit(1)

from app.core.postgres import get_session_factory  # noqa: E402
from app.models.wiki_entry import WikiEntry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='核对索引型词条')
    parser.add_argument('--slug', default=None)
    parser.add_argument('--only-problem', action='store_true', help='只打印有问题的行')
    args = parser.parse_args()

    session_factory = get_session_factory()
    with session_factory() as session:
        statement = select(WikiEntry)
        if args.slug:
            statement = statement.where(WikiEntry.slug == args.slug)
        entries = list(session.execute(statement).scalars())

        for entry in entries:
            dimensions = entry.dimensions or []
            rows = [row for d in dimensions for row in (d.get('rows') or [])]
            problems = [
                row
                for row in rows
                if not row.get('verified') or row.get('confidence') != 'high'
            ]

            print('=' * 92)
            print(f'{entry.title}')
            print(
                f'状态：{entry.status}　原文核实：'
                f'{"全部通过" if entry.original_verified else "有未通过"}　'
                f'共 {len(rows)} 行（其中需重点看 {len(problems)} 行）'
            )
            print(f'触发词：{entry.triggers}')
            print('=' * 92)

            for dimension in dimensions:
                print(f'\n### {dimension.get("name")}')
                for row in dimension.get('rows') or []:
                    is_problem = (
                        not row.get('verified') or row.get('confidence') != 'high'
                    )
                    if args.only_problem and not is_problem:
                        continue
                    mark = '✅' if row.get('verified') else '⚠️ 原文未核实'
                    print(
                        f'\n  {mark}　{row.get("source")}'
                        f'　[{row.get("confidence")}]'
                    )
                    print(f'    提取：{row.get("extracted")}')
                    print(f'    原文：{row.get("original")}')
            print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
