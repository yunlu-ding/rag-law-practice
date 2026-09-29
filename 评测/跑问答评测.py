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
    python 评测/跑问答评测.py                 # 全跑（120 题，会调大模型，有费用与耗时）
    python 评测/跑问答评测.py --only D8       # 只跑某一维度
    python 评测/跑问答评测.py --limit 3       # 先跑 3 题看看格式

## 为什么落盘用 xlsx，不用 csv

这张表是**给人判的**：人拿 Excel / WPS 打开、填一列、保存。
而 Excel 保存 CSV 时会按本地编码写（这台机器上是 GBK），
脚本下一次按 UTF-8 读就会 `UnicodeDecodeError` ——**看起来像文件坏了**，
其实只是编码被换掉了。（踩过一次，就在这份文件上。）

xlsx 里字符串是 UTF-8 存的、和编码无关，所以**这张表和评测集一样，xlsx 是唯一源**。

## 重跑不会冲掉你判过的结果

跑之前会先读一遍现有的 xlsx，把每一题的「人工判定」和「备注」按**编号**记下来；
跑完再**填回**去（前提是那一题的**问题文本没变**——题都换了，旧的判定就没意义了）。

这一步是必须的：这个脚本一跑就是 120 次模型调用，
而重跑的场景恰恰是"改了检索或提示词，想再看一遍答案"——
如果每次都把人判过的 120 格清空，那就没人会愿意重跑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / '工具'))

# Windows 控制台默认 GBK，打印带 emoji 的进度会**直接崩在打印那一步**。
# （同一个坑刚在 跑检索评测.py 上踩过一次，那次是已经花完钱才崩的。）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from app.core.postgres import get_session_factory  # noqa: E402
from app.services.qa_service import QaService  # noqa: E402
from 写表格 import write_xlsx  # noqa: E402
from 读取表格 import read_xlsx  # noqa: E402
from 评测集 import load_rows, pick  # noqa: E402

RESULT_FILE = ROOT / '评测' / '问答评测明细.xlsx'

OUTPUT_COLUMNS = [
    '编号', '维度', '问题', '期望行为', '参考答案要点',
    # 拒答种类和依据强度单独成列，而不是塞进"错误"里。
    # 判卷时这两种信息决定判法完全不同的两件事：
    #   · 拒答种类 = retrieval_error 时**不能算系统答错**（那是系统坏了，
    #     要重试，不是知识缺口）；算进去会把可用率算低。
    #   · 依据强度 = degraded 时答案质量天然弱一档，判"引用是否支撑结论"
    #     要把这个前提带上看。
    '是否拒答', '拒答种类', '依据强度', '依据提醒',
    '系统结论', '系统条款', '系统理由', '引用条数', '未还原引用', '引用来源', '错误',
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

    # ---- 先把已有的判定读回来 ----
    # 按（编号 + 问题）匹配：编号对得上、问题文本也一致，才认那格判定。
    # 题换了就不认——旧判定对应的是旧问题，硬填回去比空着更危险。
    previous: dict[tuple[str, str], tuple[str, str]] = {}
    if RESULT_FILE.exists():
        try:
            table = read_xlsx(RESULT_FILE)
            header = table[0]
            judge_at = next(
                (i for i, name in enumerate(header) if '人工判定' in (name or '')), None
            )
            note_at = next(
                (i for i, name in enumerate(header) if '备注' in (name or '')), None
            )
            for raw in table[1:]:
                cells = list(raw) + [''] * (len(header) - len(raw))
                code = cells[header.index('编号')]
                question = cells[header.index('问题')]
                if code:
                    previous[(code, question)] = (
                        cells[judge_at] if judge_at is not None else '',
                        cells[note_at] if note_at is not None else '',
                    )
            kept = sum(1 for value in previous.values() if (value[0] or '').strip())
            print(f'读回上次的判定：{len(previous)} 条记录，其中已判过的 {kept} 题')
            print()
        except Exception as exc:  # noqa: BLE001
            # 读不回来就照常跑，只是判定列会是空的——不能因为"上一份坏了"就不干活。
            print(f'（上次的 {RESULT_FILE.name} 读不回来：{type(exc).__name__}，判定列会空着）')
            print()

    def flush() -> None:
        """每跑完一题就落盘一次。

        为什么不等跑完再写：这个脚本**每题都在花钱**（一次大模型调用）。
        实测踩过——跑到第 110 题时因为一个 NameError 崩了，
        而 `results` 还在内存里，于是**前 110 题的调用费和结果一起没了**，
        只能从头再跑一遍。

        代价是每写一次整表（120 行，微不足道）。用一点 IO 换"崩了不丢钱"，
        这笔账没有任何犹豫的余地。

        ⚠️ 整表重写而不是追加：幂等，跑第二遍不会留下上一遍的残行。
        而"重写会不会把人工判定冲掉"这个问题由上面的 `previous` 解决——
        每次写之前都把读回来的判定填回对应的行。
        """

        table = [OUTPUT_COLUMNS]
        for item in results:
            carried = previous.get((item['编号'], item['问题']), ('', ''))
            item['人工判定[待你填]'] = carried[0]
            item['备注[待你填]'] = carried[1]
            table.append([str(item.get(name, '')) for name in OUTPUT_COLUMNS])
        write_xlsx(RESULT_FILE, table, title='问答评测明细')

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
                    # 「引用条数」和「未还原引用」这两列是**北极星指标的直接输入**：
                    #   可信回答率 =（结论正确 **且** 引用确实支撑结论）÷ 总回答数
                    # 没有引用条数，就分不出"答了但没给依据"（比如跑到境外法规上
                    # 硬答了 5 句、一条出处都没有）；没有未还原引用，
                    # 就不知道有没有编造出处。两者都是"可信"这两个字的一半。
                    '引用条数': len(outcome.citations or []),
                    '未还原引用': '｜'.join(outcome.unknown_citations or []),
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
                f'  [{index:>2}/{len(rows)}] {pick(row, "code"):<7} {head}'
                f'  ｜ 依据={outcome.evidence or "-"}'
                + (f' 拒答={outcome.refusal_kind}' if outcome.refused else '')
            )
            if outcome.error:
                print(f'         ⚠️ {outcome.error[:80]}')
            flush()

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
