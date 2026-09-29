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

**一、默认读 xlsx，读 csv 时编码自动识别。**
这张表是给人判的，人会拿 Excel / WPS 打开。而 Excel 保存 CSV 时按本地编码写
（这台机器上是 GBK），脚本下一次按 UTF-8 读就 `UnicodeDecodeError`——
看起来像文件坏了，其实只是编码被换掉了。

所以**这张表和评测集一样，xlsx 是唯一源**。读 csv 的能力留着，是因为
历史文件是 csv（要能回头算），读的时候按 utf-8-sig → utf-8 → gb18030 依次试。

**二、判定符号要宽容，但只认三种结果。**
人填的时候会写"通过"、会打"√"、会写"是"。这些**都算通过**。
但结果只能落在 通过 / 部分 / 未通过 三类里——不认识的写法会被
**单独列出来并计为未通过**，而不是被悄悄当成通过。
宁可把数字算低，也不能把没判的当成判过了。

用法：
    python 工具/统计问答评测.py
    python 工具/统计问答评测.py --file 评测/问答评测明细.xlsx
    python 工具/统计问答评测.py --review 评测/问答评测复评.xlsx

⚠️ **第三个用法才是"修复之后"的数字。** 这一层的数据天然是两次的：

    问答评测明细.xlsx   —— 第一轮的完整 120 题（基线）
    问答评测复评.xlsx   —— 修完之后**只重跑有问题的那几道**

合并口径：以第一轮的 120 题为基础，**用复评里重跑过的题按编号覆盖**，
其余保持不变。这才叫"当前系统在 120 题上的表现"。

**为什么不直接拿复评文件算**：那个文件只有十几行，算出来是
"那十几道题的通过率"，不是整体。两个数都有意义，但**必须说清是哪一个**——
否则很容易拿十几道题的百分比去替整体背书。
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
DEFAULT_XLSX = ROOT / '评测' / '问答评测明细.xlsx'

# 人工判定列的写法 → 三档结果。
#
# 宽容是为了好用（人不会记得该打哪个符号），
# **但未知写法一律算未通过**——这是为了让"算出来的数字偏保守"，
# 而不是偏乐观。评测最容易出的问题就是悄悄把没判的当成判过了。
PASS_MARKS = {'通过', '√', '✓', '是', 'y', 'yes', '1', 'ok', '正确'}
PARTIAL_MARKS = {'部分', '△', '基本通过', '部分通过', '0.5'}
FAIL_MARKS = {'未通过', '×', 'x', '否', 'no', '0', '不通过', '错误'}

# 但人写判卷意见时**不会只用符号**——实测拿到的是一句句中文：
#
#     「结论未命中，引用不支撑」
#     「系统条款错误，引用不匹配」
#     「结论不完整」
#     「结论表述有歧义/错误」
#
# 所以除了符号，还要能读这些评语。词表按**严重程度**排：
# 先判"明确错"，再判"不完整"。这个顺序要紧——
# 「结论表述有歧义/错误」两头都沾，它应当落到**未通过**而不是部分。
#
# 判不出来的写法仍然单独列出来并计为未通过：宁可把数字算低。
FAIL_WORDS = ('错误', '未命中', '不支撑', '不匹配', '矛盾', '不对', '答非所问', '幻觉', '编造', '相反')
PARTIAL_WORDS = ('不完整', '未完整', '有歧义', '部分', '遗漏', '不全', '不充分')

# 判定格里**写了整句话**时怎么算。
#
# 实测拿到的是这样的写法：
#     「系统条款写"第五十条"，但理由仍说"检索片段中未载明第五十条的具体内容"…」
#     「系统只确认了…1 部，并明确说"知识库中未完整列明其余7部规章名称"」
#
# 它不是符号，也没出现上面那些关键词，但它显然是"为什么这道题不行"——
# **判卷人写长句子，就是在说明问题**（通过的话没必要写这么长）。
# 所以超过这个长度、又不是以"通过"开头的，一律按未通过计，并且单独列出来。
LONG_TEXT_AS_FAIL = 12


def read_table(path: Path) -> list[dict[str, str]]:
    """读评测明细。xlsx 直接读；csv 按编码依次试。"""

    if path.suffix.lower() in ('.xlsx', '.xlsm'):
        sys.path.insert(0, str(ROOT / '工具'))
        from 读取表格 import read_xlsx

        table = read_xlsx(path)
        header = table[0]
        rows = []
        for raw in table[1:]:
            cells = list(raw) + [''] * (len(header) - len(raw))
            rows.append(dict(zip(header, cells)))
        return rows

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
    """把人工判定那一格翻成三档结果。"""

    raw = (value or '').strip()
    text = raw.lower()
    # 先看**开头**：人会写成"通过""通过，但引用可以更精确"。
    # 只做全等比较的话，后半句一加就落到"未识别"里，方向就反了。
    if text in PASS_MARKS or any(text.startswith(mark) for mark in PASS_MARKS):
        return '通过'
    if text in PARTIAL_MARKS or any(text.startswith(mark) for mark in PARTIAL_MARKS):
        return '部分'
    if text in FAIL_MARKS:
        return '未通过'
    if any(word in raw for word in FAIL_WORDS):
        return '未通过'
    if any(word in raw for word in PARTIAL_WORDS):
        return '部分'
    if len(raw) >= LONG_TEXT_AS_FAIL:
        return '未通过'
    return '未识别'


def main() -> int:
    parser = argparse.ArgumentParser(description='统计问答层评测')
    parser.add_argument(
        '--file',
        default=None,
        help='评测明细文件；默认优先用 xlsx，没有才退回 csv',
    )
    parser.add_argument(
        '--review',
        default=None,
        help='复评结果文件。给了它就会输出"第一轮 → 复评后"的对比',
    )
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    elif DEFAULT_XLSX.exists():
        path = DEFAULT_XLSX
    else:
        path = DEFAULT_CSV
    rows = read_table(path)
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

    # 把没通过的逐条列出来，**并且把判卷人写的原因原样带出来**。
    # 原因比结论重要：结论说"有 9 题没过"，原因才说明下一步该改什么。
    not_passed = [row for row, verdict in verdicts if verdict != '通过']
    if not_passed:
        print()
        print(f'  没通过的 {len(not_passed)} 题：')
        for row in not_passed:
            reason = (row.get(judge_col) or '').strip()
            print(f'    {row.get("编号"):<7} {verdict_of(reason)}  '
                  f'依据={row.get("依据强度"):<14} {reason}')
            detail = (row.get(note_col) or '').strip() if note_col else ''
            if detail:
                print(f'            {detail[:110]}')
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

    # ---- 复评合并：第一轮 + 只重跑过的那几题 ----
    if args.review:
        review_path = Path(args.review)
        if not review_path.exists():
            print()
            print(f'找不到复评文件：{review_path}')
            return 1
        review_rows = read_table(review_path)
        review_judge = find_column(review_rows[0], '人工判定')
        merged = merge_review(rows, review_rows, judge_col, review_judge)
        print_rounds(rows, merged, judge_col, review_judge, review_path.name)

    return 0


def verdicts_of(rows: list[dict[str, str]], judge_col: str) -> dict[str, str]:
    return {str(row.get('编号')): verdict_of(row.get(judge_col) or '') for row in rows}


def merge_review(
    base: list[dict[str, str]],
    review: list[dict[str, str]],
    base_judge: str,
    review_judge: str | None,
) -> list[dict[str, str]]:
    """以第一轮为基础，用复评里重跑过的题按编号覆盖。

    覆盖的是**整行**（含答案正文），不只是判定——因为复评的意义就是
    "修改之后同一道题的回答变了没有"，回答本身才是被验证的东西。
    """

    by_code = {str(row.get('编号')): row for row in review}
    return [by_code.get(str(row.get('编号')), row) for row in base]


def print_rounds(
    base: list[dict[str, str]],
    merged: list[dict[str, str]],
    base_judge: str,
    review_judge: str | None,
    review_name: str,
) -> None:
    """输出"第一轮 → 复评后"的对比，以及哪些题变了、哪些还没过。"""

    before = verdicts_of(base, base_judge)
    after = verdicts_of(merged, base_judge)
    total = len(base)

    print()
    print('=' * 66)
    print(f'复评合并（{review_name}）：第一轮 → 复评后')
    print('=' * 66)

    first_pass = sum(1 for value in before.values() if value == '通过')
    final_pass = sum(1 for value in after.values() if value == '通过')
    print(f'  第一轮     {first_pass} / {total} = {first_pass / total:.2%}')
    print(f'  复评后     {final_pass} / {total} = {final_pass / total:.2%}')
    print(f'  净变化     {final_pass - first_pass:+d} 题')
    print()

    changed = [code for code in before if before[code] != after[code]]
    fixed = [code for code in changed if after[code] == '通过']
    broke = [code for code in changed if before[code] == '通过']
    print(f'  变好的 {len(fixed)} 题：{", ".join(sorted(fixed)) or "无"}')
    print(f'  变差的 {len(broke)} 题：{", ".join(sorted(broke)) or "无"}')
    if not changed:
        print('  （没有任何题的判定发生变化）')

    still = sorted(code for code, value in after.items() if value != '通过')
    print()
    print(f'  仍然没通过的 {len(still)} 题：')
    review_by_code = {str(row.get('编号')): row for row in merged}
    for code in still:
        row = review_by_code.get(code, {})
        print(f'    {code:<7} {row.get("维度", ""):<8} {row.get("问题", "")[:30]}')

    judged_missing = [code for code in after if after[code] == '未识别']
    if judged_missing:
        print()
        print(f'  ⚠️ 判定没认出来的 {len(judged_missing)} 题：{", ".join(sorted(judged_missing))}')


if __name__ == '__main__':
    raise SystemExit(main())
