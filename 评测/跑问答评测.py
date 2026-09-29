"""跑法规评测集（问答层），把系统的回答整理成一张**供人判定的表**。

为什么需要这个脚本，而不是"在页面上一个个问、再复制粘贴"：

42 道题手工问一遍意味着 42 次点击、42 次复制、42 次粘贴，
中间粘错一行、漏一题，整份结果就不可信了。而且**人工介入得越早，
越容易在不知不觉中把结果"整理"成自己期望的样子**。

所以流程反过来：脚本先把所有答案**原样**跑出来并落表，
人再在旁边一题一题填判定。人是判官，不是搬运工。

⚠️ 有一条规矩必须说清楚：**判定只能拿参考答案要点来对，不能拿模型输出去改答案要点。**
反过来的话，模型答错了你也会把错的抄成"标准"，评测就变成了
"模型是否和模型一致"，永远显示满分。

用法：
    python 评测/跑问答评测.py                 # 全跑（42 题，会调大模型，有费用与耗时）
    python 评测/跑问答评测.py --only D8       # 只跑某一维度
    python 评测/跑问答评测.py --limit 3       # 先跑 3 题看看格式
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

# Windows 控制台默认 GBK，打印带 emoji 的进度会**直接崩在打印那一步**。
# （同一个坑刚在 跑检索评测.py 上踩过一次，那次是已经花完钱才崩的。）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from app.core.postgres import get_session_factory  # noqa: E402
from app.services.qa_service import QaService  # noqa: E402
from 评测集 import load_rows, pick  # noqa: E402

RESULT_FILE = ROOT / '评测' / '问答评测明细.csv'

OUTPUT_COLUMNS = [
    '编号', '维度', '问题', '期望行为', '参考答案要点',
    # 拒答种类和依据强度单独成列，而不是塞进"错误"里。
    # 判卷时这两种信息决定判法完全不同的两件事：
    #   · 拒答种类 = retrieval_error 时**不能算系统答错**（那是系统坏了，
    #     要重试，不是知识缺口）；算进去会把可用率算低。
    #   · 依据强度 = degraded 时答案质量天然弱一档，判"引用是否支撑结论"
    #     要把这个前提带上看。
    '是否拒答', '拒答种类', '依据强度', '依据提醒',
    '系统结论', '系统条款', '系统理由', '引用来源', '错误',
    '人工判定[待你填]', '备注[待你填]',
]


def main() -> int:
    parser = argparse.ArgumentParser(description='跑法规评测集（问答层）')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--only', default=None, help='只跑编号以它开头的题（如 D8）')
    parser.add_argument('--limit', type=int, default=None, help='只跑前 N 题')
    args = parser.parse_args()

    rows = load_rows()
    if args.only:
        rows = [row for row in rows if pick(row, 'code').startswith(args.only)]
    if args.limit:
        rows = rows[: args.limit]

    print(f'待跑 {len(rows)} 题（每题会调用一次大模型）')
    print()

    results: list[dict] = []
    session_factory = get_session_factory()
    with session_factory() as session:
        service = QaService(session)
        for index, row in enumerate(rows, start=1):
            question = pick(row, 'question')
            outcome = service.ask(question=question, top_k=args.top_k)

            citations = '｜'.join(
                str(citation.get('label') or citation.get('filename') or '')
                for citation in (outcome.citations or [])
            )
            results.append(
                {
                    '编号': pick(row, 'code'),
                    '维度': pick(row, 'dimension'),
                    '问题': question,
                    '期望行为': pick(row, 'expected'),
                    '参考答案要点': pick(row, 'answer'),
                    '是否拒答': '是' if outcome.refused else '否',
                    '拒答种类': outcome.refusal_kind or '',
                    '依据强度': outcome.evidence or '',
                    '依据提醒': outcome.evidence_note or '',
                    '系统结论': outcome.conclusion or '',
                    '系统条款': outcome.clause or '',
                    '系统理由': outcome.reasoning or '',
                    '引用来源': citations,
                    '错误': outcome.error or (
                        f'拒答：{outcome.refusal_reason}' if outcome.refused else ''
                    ),
                    '人工判定[待你填]': '',
                    '备注[待你填]': '',
                }
            )

            head = outcome.conclusion or ('拒答' if outcome.refused else '（无结论）')
            print(
                f'  [{index:>2}/{len(rows)}] {pick(row, "编号"):<7} {head}'
                f'  ｜ 依据={outcome.evidence or "-"}'
                + (f' 拒答={outcome.refusal_kind}' if outcome.refused else '')
            )
            if outcome.error:
                print(f'         ⚠️ {outcome.error[:80]}')

    with RESULT_FILE.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f'已写入 {RESULT_FILE}')
    print()
    print('接下来是你的活：打开这份表，逐题对照「参考答案要点」，在')
    print('「人工判定[待你填]」里填 通过 / 部分 / 未通过，有疑问写进备注。')
    print()
    print('判定标准建议：')
    print('  · 通过   —— 结论正确，且引用的条款能支撑结论')
    print('  · 部分   —— 结论方向对，但漏了要点，或引错了层级')
    print('  · 未通过 —— 结论错误，或引用了不存在的条款，或该拒答却硬答')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
