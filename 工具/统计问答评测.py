"""把人工判过的问答评测明细，算成指标。

为什么单独做一个工具、而不是"看一眼表"：

因为这张表要回答的是**北极星指标**，而北极星是要拿去说事的数字。
凡是拿出去说的数字，都必须**能被别人按同一口径算出来**——
人来数一遍、机器来数一遍，结果得一样。否则它只是一段印象。

它算出四层东西：

    北极星   可信回答率 =（结论正确 且 引用支撑结论）÷ 总回答数
    生成层   引用支撑情况、没有引用的回答数、编造引用数
    边界层   拒答正确率、过度拒答率（**必须成对看**）
    过程层   依据强度分布、拒答分因、耗时

## 两个刻意的口径选择

**一、编码自动识别。**
这张表是给人判的，人会拿 Excel / WPS 打开，一存就变成 GBK 或 GB18030，
而脚本自己写的是 UTF-8。所以读的时候按
「UTF-8 带 BOM → UTF-8 → GB18030」的顺序试，而不是写死一个。
（踩过：写死 utf-8 直接 `UnicodeDecodeError`，看起来像文件坏了。）

**二、判定符号要宽容，但只认三种结果。**
人填的时候会写"通过"、会打"√"、会写"是"。这些**都算通过**。
但结果只能落在 通过 / 部分 / 未通过 三类里——不认识的写法会被
**单独列出来并计为未通过**，而不是被悄悄当成通过。
宁可把数字算低，也不能把没判的当成判过了。

用法：
    python 工具/统计问答评测.py
    python 工具/统计问答评测.py --csv 评测/问答评测明细.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

DEFAULT_CSV = ROOT / '评测' / '问答评测明细.csv'

# 人工判定列的写法 → 三档结果。
#
# 宽容是为了好用（人不会记得该打哪个符号），
# **但未知写法一律算未通过**——这是为了让"算出来的数字偏保守"，
# 而不是偏乐观。评测最容易出的问题就是悄悄把没判的当成判过了。
PASS_MARKS = {'通过', '√', '✓', '是', 'y', 'yes', '1', 'ok', '正确'}
PARTIAL_MARKS = {'部分', '△', '基本通过', '部分通过', '0.5'}
FAIL_MARKS = {'未通过', '×', 'x', '否', 'no', '0', '不通过', '错误'}


def read_rows(path: Path) -> list[dict[str, str]]:
    """读评测明细，自动识别编码。"""

    raw = path.read_bytes()
    for encoding in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - 三个都失败说明文件真的有问题
        raise ValueError(f'{path.name} 无法识别编码（试过 utf-8-sig / utf-8 / gb18030）')

    return list(csv.DictReader(io.StringIO(text)))


def find_column(row: dict[str, str], keyword: str) -> str | None:
    """按关键字找列名（列名带「[待你填]」这类后缀，写死会脆）。"""

    for name in row:
        if name and keyword in name:
            return name
    return None


def verdict_of(value: str) -> str:
    text = (value or '').strip().lower()
    if text in PASS_MARKS:
        return '通过'
    if text in PARTIAL_MARKS:
        return '部分'
    if text in FAIL_MARKS:
        return '未通过'
    return '未识别'


def main() -> int:
    parser = argparse.ArgumentParser(description='统计问答层评测')
    parser.add_argument('--csv', default=str(DEFAULT_CSV))
    args = parser.parse_args()

    path = Path(args.csv)
    rows = read_rows(path)
    if not rows:
        print('表是空的')
        return 1

    judge_col = find_column(rows[0], '人工判定')
    note_col = find_column(rows[0], '备注')
    if judge_col is None:
        print('找不到「人工判定」列')
        return 1

    print(f'文件：{path}')
    print(f'题数：{len(rows)}')
    print()

    verdicts = [(row, verdict_of(row.get(judge_col) or '')) for row in rows]
    counts = collections.Counter(verdict for _, verdict in verdicts)
    unknown = [(row, row.get(judge_col)) for row, v in verdicts if v == '未识别' and (row.get(judge_col) or '').strip()]

    passed = counts['通过']
    failed = counts['未通过'] + len(unknown)
    total = len(rows)

    # ---- 北极星 ----
    #
    # 判定标准是「结论正确 **且** 引用支撑结论」——它是**一个**判定，
    # 不是一个"结论对"再加一个"引用对"。所以这里不拆成两个百分比再相乘：
    # 那样会凭空造出一个"结论正确率"，而人判的时候并没有分开判过它。
    print('=' * 66)
    print('北极星：可信回答率')
    print('=' * 66)
    print(f'  （结论正确 且 引用支撑结论）÷ 总回答数')
    print(f'  = {passed} / {total} = {passed / total:.1%}')
    print()
    print(f'  通过 {counts["通过"]} ｜ 部分 {counts["部分"]} ｜ 未通过 {counts["未通过"]}'
          + (f' ｜ 写法没认出来 {len(unknown)}（计为未通过）' if unknown else ''))
    if unknown:
        print('  没认出来的写法：')
        for row, value in unknown[:10]:
            print(f'    {row.get("编号")}：{value!r}')
    if note_col:
        noted = sum(1 for row in rows if (row.get(note_col) or '').strip())
        print(f'  填了备注的题数：{noted}')
        if noted == 0:
            print('    ⚠️ 一条备注都没有 —— 判卷时没有任何存疑的地方，'
                  '这种事在 120 题里很少见，值得回头看一眼。')
    print()

    # ---- 边界层：必须成对看 ----
    #
    # "该拒的拒了"单独看是可以作弊的：什么都不答，这个数字就是满分。
    # 所以永远和"不该拒的拒了"一起看。
    print('=' * 66)
    print('边界层：拒答正确率 与 过度拒答率（成对看）')
    print('=' * 66)
    should_refuse = [row for row in rows if '应拒答' in (row.get('期望行为') or '')]
    should_explain = [row for row in rows if '应说明' in (row.get('期望行为') or '')]
    answered = [row for row in rows if row not in should_refuse]

    refused_flags = [row for row in should_refuse if (row.get('是否拒答') or '').strip() == '是']
    over_refuse = [row for row in answered if (row.get('是否拒答') or '').strip() == '是']

    print(f'  该拒答的题：{len(should_refuse)} 题')
    print(f'    实际拒答 {len(refused_flags)} 题 → 拒答正确率 '
          f'{(len(refused_flags) / len(should_refuse) if should_refuse else 0):.0%}')
    for row in should_refuse:
        mark = '拒' if (row.get('是否拒答') or '').strip() == '是' else '答'
        note = (row.get(note_col) or '')[:40] if note_col else ''
        print(f'      [{mark}] {row.get("编号")} 依据={row.get("依据强度")} '
              f'引用={row.get("引用条数")} {row.get("问题", "")[:24]} {note}')
    print(f'  应说明（不是拒答）的题：{len(should_explain)} 题')
    print(f'  其余问题里被拒答的：{len(over_refuse)} 题 → 过度拒答率 '
          f'{(len(over_refuse) / len(answered) if answered else 0):.1%}')
    for row in over_refuse:
        print(f'      {row.get("编号")} {row.get("问题", "")[:30]}')
    print()
    print('  ⚠️ 拒答那一列看的是**系统的标志位**。实测踩到过：有的题'
          '标志位是"未拒答"，理由里写的却是"知识库中没有关于…的信息"——')
    print('     实质拒了，只是没落到「无法判断」这个枚举值上。'
          '按标志位统计会把拒答正确率**系统性低估**。')
    print()

    # ---- 过程层 ----
    print('=' * 66)
    print('过程层：回答是怎么产生的')
    print('=' * 66)
    print('  依据强度：', dict(collections.Counter(row.get('依据强度') for row in rows)))
    print('  拒答分因：', dict(collections.Counter(
        (row.get('拒答种类') or '（没拒答）') for row in rows)))
    print('  没有引用（引用条数=0）的回答：',
          sum(1 for row in rows if (row.get('引用条数') or '').strip() in ('0', '')))
    print('  编造引用（模型引了不存在的片段编号）：',
          sum(1 for row in rows if (row.get('未还原引用') or '').strip()))
    print()

    print('  按维度：')
    by_dim: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row, verdict in verdicts:
        by_dim[str(row.get('维度'))][verdict] += 1
    for dim, counter in by_dim.items():
        n = sum(counter.values())
        print(f'    {dim:<10} {counter["通过"]:>3} / {n:<3} 通过')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
