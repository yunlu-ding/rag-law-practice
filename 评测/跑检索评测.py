"""跑法规评测集（检索层）。判"命中没命中"，不需要大模型判分。

为什么先做检索层而不是问答层：

  - **它便宜、稳定、可重复。** 判据是"返回的切片里有没有锚点原文"，
    逐字比对，不涉及模型采样，跑一百次结果一样；
  - **它把问题定位得更准。** 问答答错有两种原因——没召回到，或者召回到了但模型没说对。
    检索层先把前一种摘掉，剩下才是提示词的事；
  - **它不受限流和费用影响。** 欠费那次如果不是靠条款直查，整条链路都验证不了。

判定口径（写在这里，因为"命中"这个词很容易各说各话）：

  - `命中`：期望行为写着"命中"的，**至少命中 1 个锚点**即算通过；
  - `命中至少两个层级`：多锚点题，**命中 ≥2 个锚点**才算通过——
    只命中一个说明跨层级综合没做到；
  - `应拒答` / `应说明...`：**检索层判不了**。
    这不是偷懒：这几个问题的正确答案是"没有答案"，
    而"检索分数低"和"语料里确实没有"是两件事（这个项目踩过这个坑，
    详见 02-五个问题定义与优先级）。它们必须由人读问答结果来判，
    所以这里只列出来，标为「待人工」。

用法：
    python 评测/跑检索评测.py                 # 用默认 top_k
    python 评测/跑检索评测.py --top-k 8
    python 评测/跑检索评测.py --only D1       # 只跑某个维度
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

from app.core.postgres import get_session_factory  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402

EVAL_FILE = ROOT / '评测' / '法规评测集.csv'
REPORT_FILE = ROOT / '评测' / '检索评测报告.md'
RESULT_FILE = ROOT / '评测' / '检索评测明细.csv'

SEPARATOR = '｜'


def normalize(text: str) -> str:
    """比对前把空白去掉。

    切片正文里有换行和全角空格，锚点是从数据库里截出来的（也可能跨行），
    不对齐空白会大量误判为"未命中"，而原因只是排版。
    """

    return re.sub(r'[\s\u3000]+', '', text or '')


def judge(expected: str, anchors: list[str], hits: list[dict]) -> tuple[str, str]:
    """返回 (判定, 说明)。判定取值：通过 / 未通过 / 待人工。"""

    expected = (expected or '').strip()

    if expected.startswith('应拒答') or expected.startswith('应说明'):
        return '待人工', '这一题的正确答案是"没有答案"，检索层判不了，需要人读问答结果'

    if not anchors:
        return '待人工', '这一题没有锚点'

    corpus = [normalize(hit.get('text')) for hit in hits]
    matched = [anchor for anchor in anchors if normalize(anchor) in ''.join(corpus)]

    need = 2 if '至少两个' in expected else 1
    if len(matched) >= need:
        return '通过', f'命中 {len(matched)}/{len(anchors)} 个锚点'
    return '未通过', f'只命中 {len(matched)}/{len(anchors)} 个锚点（需要 {need} 个）'


def main() -> int:
    parser = argparse.ArgumentParser(description='跑法规评测集（检索层）')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--only', default=None, help='只跑编号以它开头的题（如 D1）')
    args = parser.parse_args()

    with EVAL_FILE.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.only:
        rows = [row for row in rows if row['编号'].startswith(args.only)]

    results: list[dict] = []
    by_dimension: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    session_factory = get_session_factory()
    with session_factory() as session:
        service = RetrievalService(session)
        for row in rows:
            anchors = [
                item.strip()
                for item in (row['锚点原文'] or '').split(SEPARATOR)
                if item.strip() and not item.startswith('（')
            ]
            outcome = service.search(query=row['问题'], top_k=args.top_k, persist=False)
            verdict, note = judge(row['期望行为'], anchors, outcome.hits)

            results.append(
                {
                    '编号': row['编号'],
                    '维度': row['维度'],
                    '问题': row['问题'],
                    '判定': verdict,
                    '说明': note,
                    '条款直查': outcome.exact_hit_count,
                    '首条来源': (outcome.hits[0].get('filename') if outcome.hits else ''),
                }
            )
            by_dimension[row['维度']][verdict] += 1
            mark = {'通过': '✅', '未通过': '❌', '待人工': '🙋'}[verdict]
            print(f"{mark} {row['编号']:<7} {note[:44]:<46} {row['问题'][:26]}")

    print()
    print('=' * 82)
    total = defaultdict(int)
    for dimension, counts in by_dimension.items():
        judged = counts['通过'] + counts['未通过']
        rate = f"{counts['通过'] / judged:.0%}" if judged else '—'
        print(
            f"{dimension:<12} 通过 {counts['通过']:>2} ｜ 未通过 {counts['未通过']:>2} ｜ "
            f"待人工 {counts['待人工']:>2} ｜ 通过率 {rate}"
        )
        for key, value in counts.items():
            total[key] += value

    judged = total['通过'] + total['未通过']
    print('-' * 82)
    print(
        f"{'合计':<12} 通过 {total['通过']:>2} ｜ 未通过 {total['未通过']:>2} ｜ "
        f"待人工 {total['待人工']:>2} ｜ 通过率 "
        f"{(total['通过'] / judged if judged else 0):.0%}"
    )
    print()
    print('「待人工」的题不是没跑，而是**正确答案就是「没有答案」**——')
    print('检索层判不了，需要人看问答结果才知道它有没有正确地说「我不知道」。')

    with RESULT_FILE.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print()
    print(f'明细已写入 {RESULT_FILE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
