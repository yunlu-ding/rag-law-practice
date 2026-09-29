"""评测集的读写入口。**xlsx 是唯一源。**

为什么不让脚本各读各的：

评测集是人编辑的，而人用 Excel/WPS 编辑比 CSV 顺手得多——
CSV 里只要有一格写着逗号或换行，整张表的列就会串位，而且事后很难看出是哪一行错的。
所以定 **xlsx 为唯一源**。

但"唯一源"这句话如果只是口头约定，很快就会退化成两份：
人改了 xlsx，脚本读的还是旧 CSV，评测跑出来的是几天前的题。
所以读写都收敛到这一个模块，脚本不再自己拼文件路径。

表格列（表头名带 [待你填] 之类的后缀是给人看的提示，脚本按前缀匹配，不用改）：

    编号 / 维度 / 难度 / 问题 / 期望行为 / 依据文件 / 依据条款
    锚点原文[待核对]        ← 脚本生成，人核对
    参考答案要点[待你填]     ← 人填
    备注[待你填]            ← 人填
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '工具'))

from 读取表格 import read_xlsx  # noqa: E402
from 写表格 import write_xlsx  # noqa: E402

EVAL_XLSX = ROOT / '评测' / '法规评测集.xlsx'

# 列名在表里带提示后缀，脚本按前缀取，改表头不影响代码。
COLUMNS = {
    'code': '编号',
    'dimension': '维度',
    'difficulty': '难度',
    'question': '问题',
    'expected': '期望行为',
    'source_file': '依据文件',
    'source_article': '依据条款',
    'anchor': '锚点原文',
    'answer': '参考答案要点',
    'note': '备注',
}

# 写回锚点时用的真实列名。列名里带 [待核对] 是给人看的提示，
# 而写回必须用**表里真实存在的**那个键，所以单独留一个常量，
# 而不是在脚本里到处硬编码字符串。
ANCHOR_COLUMN = '锚点原文[待核对]'


def pick(row: dict[str, str], key: str) -> str:
    """按列名前缀取值。"""

    prefix = COLUMNS[key]
    for name, value in row.items():
        if name and name.startswith(prefix):
            return value or ''
    return ''


def load_rows() -> list[dict[str, str]]:
    """读评测集，返回 [{表头名: 值}]。"""

    table = read_xlsx(EVAL_XLSX)
    if not table:
        raise ValueError(f'{EVAL_XLSX.name} 是空的')
    header = table[0]
    rows: list[dict[str, str]] = []
    for raw in table[1:]:
        if not any(cell for cell in raw):
            continue
        padded = list(raw) + [''] * (len(header) - len(raw))
        rows.append(dict(zip(header, padded)))
    return rows


def save_rows(rows: list[dict[str, str]]) -> None:
    """写回评测集。列序按**第一行的键序**保留。"""

    if not rows:
        raise ValueError('没有数据可写')
    header = list(rows[0].keys())
    table = [header] + [[row.get(name, '') for name in header] for row in rows]
    write_xlsx(EVAL_XLSX, table, title='法规评测集')


__all__ = ['ANCHOR_COLUMN', 'COLUMNS', 'EVAL_XLSX', 'load_rows', 'pick', 'save_rows']
