"""比对两轮问答评测，挑出**判定可能失效**的题，生成一张待判表。

## 为什么需要它，而不是"把 120 题重判一遍"

重跑全量之后要回答的问题是：**"改了提示词，有没有把别处弄坏？"**

最直接的做法是把 120 题全部重判一次——但那样等于让判卷人做两遍同样的活，
而其中绝大多数题的回答其实没本质变化。

## 但"逐字比对"不能当判据（实测结论）

拿两轮逐字比，**120 题里有 117 题的措辞都不一样**——因为生成是随机的，
温度哪怕为 0 也不保证逐字一致。

有意思的是那 3 道**逐字相同**的题：它们恰好是走**确定性判据**、**根本没有调用模型**
的那三道（问不存在的条号，直接被拒答）。**这条对比本身说明了一件事：
只有确定性的地方才真正可复现；其余地方"同一个问题两次答案不同"是常态。**

## ⚠️ 我一开始想用"结构化字段比对"来省事，被数据否掉了

想法很自然：判定的依据无非是结论、条款、引用条数、是否拒答、拒答种类、依据强度，
**这六项没变、只有措辞变了**，就沿用旧判定——这样只需要重判几十条。

**实测立刻打脸**：这一轮修好的三道题（D5-06 / D5-08 / D6-07）**恰好落在"只有措辞不同"里**。
因为修复只体现在**理由文本**上——原来写"未载明施行日期"，现在写出"2017年7月1日"，
而结构化字段（结论="说明"、条款=空）**一个字都没变**。

也就是说：**"判定字段没变就能沿用旧判定"是错的。判定依赖的正是那段文字，
而那段文字是非确定性的。**

所以这个工具**不做预筛**：它把 120 题全列出来，
只是**按变化程度排个序**（判定字段变了的排前面），并把旧的答案和判定摆在旁边，
让人一眼看到"变了什么"。**要判哪些、判到什么程度，是判卷人的决定，不是脚本的。**

（唯一能安全沿用旧判定的是"逐字相同"的那几条——但它们逐字相同的原因很特别，
见上面那一段。）

用法：
    python 工具/比对问答两轮.py
    python 工具/比对问答两轮.py --base 评测/问答评测明细.xlsx --new 评测/问答评测全量复评.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '工具'))

# Windows 控制台默认按 GBK 解码，打印 ✅/⚠️ 会直接崩在打印那一步。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from 写表格 import write_xlsx  # noqa: E402
from 读取表格 import read_xlsx  # noqa: E402

BASE_FILE = ROOT / '评测' / '问答评测明细.xlsx'
NEW_FILE = ROOT / '评测' / '问答评测全量复评.xlsx'
OUT_FILE = ROOT / '评测' / '问答评测待判.xlsx'

# 判定真正依赖的字段。措辞不在其中——理由见模块开头。
VERDICT_FIELDS = ('系统结论', '系统条款', '引用条数', '是否拒答', '拒答种类', '依据强度')

REFERENCE_COLUMNS = ['人工判定[待你填]', '备注[待你填]']


def load(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    table = read_xlsx(path)
    header = table[0]
    rows: dict[str, dict[str, str]] = {}
    for raw in table[1:]:
        cells = list(raw) + [''] * (len(header) - len(raw))
        row = dict(zip(header, cells))
        code = row.get('编号')
        if code:
            rows[code] = row
    return header, rows


def main() -> int:
    parser = argparse.ArgumentParser(description='比对两轮问答评测')
    parser.add_argument('--base', default=str(BASE_FILE))
    parser.add_argument('--new', default=str(NEW_FILE))
    parser.add_argument('--out', default=str(OUT_FILE))
    args = parser.parse_args()

    base_header, base = load(Path(args.base))
    _, new = load(Path(args.new))

    # 变化类型 → 排序权重（判定字段变了的排最前，等价的排最后）
    #   0 = 逐字相同（**唯一能安全沿用旧判定的情况**）
    #   1 = 判定字段变了（可能改了结论、改了依据，或把拒答变成了硬答）
    #   2 = 只有措辞不同（**不能默认安全**——这一轮修好的三道题就落在这里）
    rows: list[dict[str, str]] = []
    for code, old in base.items():
        fresh = new.get(code)
        if fresh is None:
            print(f'  ⚠️ {code} 在重跑结果里没有，跳过')
            continue
        same_text = (old.get('系统理由') or '').strip() == (fresh.get('系统理由') or '').strip()
        diff = [
            field
            for field in VERDICT_FIELDS
            if (old.get(field) or '').strip() != (fresh.get(field) or '').strip()
        ]
        # ⚠️ 顺序要紧：先看字段，再看正文。
        #
        # 只比"理由"会把"理由一字不差、但结论字段变了"的情况误判成逐字相同——
        # 实测真出现过（同一段理由配不同的「结论」值），而那种情况恰恰最该重判。
        if diff:
            kind, order = '判定字段变了', 1
        elif same_text:
            kind, order = '逐字相同', 0
        else:
            kind, order = '只有措辞不同', 2

        row = dict(fresh)
        row['变化类型'] = kind
        row['变化字段'] = '、'.join(diff)
        row['旧结论'] = old.get('系统结论', '')
        row['旧条款'] = old.get('系统条款', '')
        row['旧判定'] = old.get('人工判定[待你填]', '')
        row['_order'] = str(order)
        rows.append(row)

    rows.sort(key=lambda item: (item['_order'], item['编号']))

    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind[row['变化类型']] = by_kind.get(row['变化类型'], 0) + 1

    print('=' * 66)
    print('两轮比对（按变化程度排序）')
    print('=' * 66)
    for kind in ('逐字相同', '判定字段变了', '只有措辞不同'):
        print(f'  {kind:<14} {by_kind.get(kind, 0):>3} 题')
    print()
    same = [row['编号'] for row in rows if row['变化类型'] == '逐字相同']
    if same:
        print('  逐字相同的：' + '、'.join(sorted(same)))
        print('    （它们逐字相同是因为**根本没调用模型**——走的是确定性判据。')
        print('      反过来也说明：其余地方"同一问题两次答案不同"是常态。）')
    print()
    print('  ⚠️ 这个工具**不做预筛**：它不判断哪些题"可以沿用旧判定"。')
    print('     除了逐字相同的那几道，其余都存在"答案变了、判定该跟着变"的可能——')
    print('     这一轮修好的三道题就落在"只有措辞不同"里，只看结构化字段会漏掉它们。')
    print('     判哪些、判到什么程度，是判卷人的决定。')
    print()

    columns = list(base_header) + ['变化类型', '变化字段', '旧结论', '旧条款', '旧判定']
    table = [columns] + [
        [str(row.get(name, '')) for name in columns] for row in rows
    ]
    # 人工判定列清空：这一格是**要重新填的**，
    # 留着旧值会让"这一格是新判的还是旧抄的"分不清。
    judge_at = columns.index('人工判定[待你填]')
    for row in table[1:]:
        row[judge_at] = ''

    out_path = Path(args.out)
    write_xlsx(out_path, table, title='问答评测待判')
    print(f'待判表已写出：{out_path}')
    print(f'  共 {len(rows)} 题（全部），人工判定列已清空，按变化程度排序。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
