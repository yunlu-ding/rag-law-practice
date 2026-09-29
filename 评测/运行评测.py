"""评测运行脚本。

设计原则（这几条决定了数字能不能被引用）：

1. **走产品真实链路。** 通过真实接口调用，不绕开任何一层。
   绕开链路测出来的数字好看，但不代表用户会遇到什么。
2. **一次只改一个变量。** 同一批问题、同一套参数，只对比你想对比的东西。
3. **自动判定与人工判定分开。** 检索类可以脚本判；"引用是否支撑结论"
   这类必须人工看。脚本能做的只是把证据整理好，减少人工成本——
   **不能把需要人判断的事自动化，否则数字就失去意义。**
4. **跑不动的维度要明说。** 当前语料不具备条件的维度，写"未测"，
   而不是造一个数字填上。

用法：
    cd vibe-rag/backend
    python ../评测/运行评测.py
    python ../评测/运行评测.py --top-k 50 --skip-qa    # 只跑检索指标
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / 'backend'
sys.path.insert(0, str(BACKEND))

import logging  # noqa: E402

logging.getLogger('pymilvus').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('app').setLevel(logging.WARNING)

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET = EVAL_DIR / '评测集.csv'


class _Response:
    """把 HTTP 响应包装成和 TestClient 一样的最小接口。"""

    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        return self._body


class HttpEvalClient:
    """走真实 HTTP 接口的评测客户端。

    为什么要加这个模式：默认的进程内测试跑在同一台机器上，
    **测不到"服务部署出去之后"的那些环节**——网络、反向代理、真实的超时。
    而且它还有个现实好处：可以在不装 Python 依赖的机器上评测一个远程服务。
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip('/')

    def post(self, path: str, **kwargs) -> _Response:
        payload = json.dumps(kwargs.get('json') or {}).encode('utf-8')
        request = urllib.request.Request(
            self.base_url + path,
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return _Response(response.status, json.loads(response.read().decode('utf-8')))
        except urllib.error.HTTPError as exc:
            # 4xx/5xx 也是有效结果（比如限额触发 429），要按响应处理而不是当异常抛掉
            body = exc.read().decode('utf-8') or '{}'
            try:
                return _Response(exc.code, json.loads(body))
            except json.JSONDecodeError:
                return _Response(exc.code, {'detail': body})


def normalize(text: str) -> str:
    """与锚点校验保持完全一致的口径。

    两处口径必须一样，否则会出现"校验说锚点在语料里，评测却永远判不命中"——
    这种不一致会让人怀疑系统，而问题其实在评测工具自己身上。
    """

    lowered = str(text or '').lower()
    lowered = lowered.replace('’', "'").replace('‘', "'")
    cleaned = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', ' ', lowered)
    return re.sub(r'\s+', ' ', cleaned).strip()


def find_rank(anchor: str, items: list[dict]) -> int | None:
    """锚点在第几名被召回（1 起算），没召回到返回 None。"""

    if not anchor:
        return None
    target = normalize(anchor)
    for index, item in enumerate(items, start=1):
        if target in normalize(item.get('text') or ''):
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description='法规知识库评测')
    parser.add_argument('--top-k', type=int, default=50, help='检索取多少条用于排名诊断')
    parser.add_argument('--skip-qa', action='store_true', help='只跑检索指标，不跑问答')
    parser.add_argument('--qa-sample', type=int, default=6, help='抽几道检索题跑问答以收集引用证据')
    parser.add_argument(
        '--base-url',
        help='走 HTTP 接口评测一个正在运行的服务，例如 http://127.0.0.1:8000；不传则用进程内测试',
    )
    args = parser.parse_args()

    if args.base_url:
        print(f'评测目标：{args.base_url}（走真实 HTTP 接口）')
        client = HttpEvalClient(args.base_url)
    else:
        from fastapi.testclient import TestClient

        from app.core.postgres import init_database
        from app.main import app

        init_database()
        client = TestClient(app)
        print('评测目标：进程内（TestClient）')

    rows = list(csv.DictReader(EVAL_SET.open(encoding='utf-8')))
    # 检索类 = 除 D5（边界拒答）之外的所有维度。
    # 用"排除法"而不是列举，是为了**以后加新维度时不用再改这里**——
    # 评测脚本自己变成维护负担，是很容易被忽略的一类问题。
    retrieval_rows = [r for r in rows if not r['维度'].startswith('D5')]
    refusal_rows = [r for r in rows if r['维度'].startswith('D5')]

    print(f'评测集 {len(rows)} 条：检索 {len(retrieval_rows)} 条，边界拒答 {len(refusal_rows)} 条')
    print(f'检索取前 {args.top_k} 条用于排名诊断\n')

    results: list[dict] = []
    http_errors: list[tuple[str, int, str]] = []

    def note_http_error(code: str, response, data: dict) -> None:
        """记录非 200 响应。

        为什么这件事必须显式做：
        评测脚本要连发几十个请求，**很容易触发服务自己的限频（429）**。
        如果不把这个单独标出来，它会和"答错了"混在一起，
        最后得出一个"产品退步了"的结论——而实际上产品没问题，是评测方式不对。

        这和"锚点校验"是同一类教训：**先确认度量工具本身是准的，再相信测出来的数字。**
        """

        detail = str(data.get('detail') or data.get('message') or '')[:60]
        http_errors.append((code, response.status_code, detail))
        print(f'  ! {code:<8} HTTP {response.status_code}: {detail}')

    # ---------- D1 / D3 检索命中 ----------
    print('===== 检索命中 =====')
    for row in retrieval_rows:
        started = time.perf_counter()
        response = client.post(
            '/api/v1/retrieval/search',
            json={'query': row['问题'], 'top_k': args.top_k},
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        data = response.json()
        if response.status_code != 200:
            note_http_error(row['编号'], response, data)
        items = data.get('items') or []
        rank = find_rank(row.get('锚点原文', ''), items)
        results.append(
            {
                '编号': row['编号'],
                '维度': row['维度'],
                '语言': row['语言'],
                '问题': row['问题'],
                '锚点名次': rank if rank else '未召回',
                'Top1': '命中' if rank and rank <= 1 else '',
                'Top3': '命中' if rank and rank <= 3 else '',
                'Top5': '命中' if rank and rank <= 5 else '',
                '向量路召回': data.get('vector_hit_count'),
                '关键词路召回': data.get('bm25_hit_count'),
                '耗时ms': elapsed,
                '第一名来源': (items[0].get('filename') if items else ''),
            }
        )
        mark = '✓' if rank else '✗'
        print(f"  {mark} {row['编号']:<8} 名次={str(rank or '未召回'):<6} {row['问题'][:34]}")

    # ---------- D5 边界拒答 ----------
    qa_results: list[dict] = []
    if not args.skip_qa:
        print('\n===== 边界拒答（预期：明确说没有 / 拒绝）=====')
        for row in refusal_rows:
            started = time.perf_counter()
            response = client.post(
                '/api/v1/qa/ask',
                json={'question': row['问题'], 'top_k': 5, 'refuse_threshold': 0},
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            data = response.json()
            if response.status_code != 200:
                note_http_error(row['编号'], response, data)
            refused = bool(data.get('refused'))
            # 系统故障要和"判定错误"分开统计。
            # 检索链路故障时系统根本没机会作答，把它算进"拒答正确率"里，
            # 数字就会随着网络状况浮动——那样指标就不再是产品能力，而是网络质量。
            failed = bool(data.get('retrieval_failed'))
            # 正确行为有两种：走拒答分支，或者明确说资料里没有。
            reason = (data.get('reasoning') or '')
            said_missing = any(
                keyword in reason
                for keyword in ('资料中没有', '知识库中没有', '没有足够的依据', '未提及', '无法预测', '不包含')
            )
            qa_results.append(
                {
                    '编号': row['编号'],
                    '类型': 'D5边界拒答',
                    '问题': row['问题'],
                    '是否拒答': '是' if refused else '否',
                    '结论': data.get('conclusion') or '',
                    '引用数': len(data.get('citations') or []),
                    '未知引用数': len(data.get('unknown_citations') or []),
                    '是否说明资料中没有': '是' if said_missing else '否',
                    '系统故障': '是' if failed else '否',
                    '人工判定': '待复核',
                    '耗时ms': elapsed,
                    '回答摘要': reason[:160].replace('\n', ' '),
                }
            )
            ok = refused or said_missing
            mark = '⚠' if failed else ('✓' if ok else '✗')
            print(f"  {mark} {row['编号']:<8} 拒答={refused} 说明没有={said_missing} 故障={failed}  {row['问题'][:30]}")

        # ---------- 引用证据抽样 ----------
        print(f'\n===== 引用证据抽样（{args.qa_sample} 道，供人工判卷）=====')
        for row in retrieval_rows[: args.qa_sample]:
            started = time.perf_counter()
            response = client.post(
                '/api/v1/qa/ask',
                json={'question': row['问题'], 'top_k': 5, 'refuse_threshold': 0},
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            data = response.json()
            if response.status_code != 200:
                note_http_error(row['编号'], response, data)
            qa_results.append(
                {
                    '编号': row['编号'],
                    '类型': 'D4引用准确（抽样）',
                    '问题': row['问题'],
                    '是否拒答': '是' if data.get('refused') else '否',
                    '结论': data.get('conclusion') or '',
                    '引用数': len(data.get('citations') or []),
                    '未知引用数': len(data.get('unknown_citations') or []),
                    '是否说明资料中没有': '',
                    '系统故障': '是' if data.get('retrieval_failed') else '否',
                    '人工判定': '待复核',
                    '耗时ms': elapsed,
                    '回答摘要': (data.get('reasoning') or '')[:160].replace('\n', ' '),
                }
            )
            print(f"  {row['编号']:<8} 结论={data.get('conclusion')} 引用={len(data.get('citations') or [])} 未知={len(data.get('unknown_citations') or [])}")

    # ---------- 汇总 ----------
    def rate(items: list[dict], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for item in items if item.get(key) == '命中') / len(items)

    print('\n' + '=' * 78)
    if http_errors:
        print(f'⚠️ 本次有 {len(http_errors)} 个请求没有正常返回，下面的结果不可信！')
        codes: dict[int, int] = {}
        for _, status, _ in http_errors:
            codes[status] = codes.get(status, 0) + 1
        for status, count in sorted(codes.items()):
            print(f'   HTTP {status}: {count} 次')
        if 429 in codes:
            print('   → 429 是服务自己的限频。评测要连发几十个请求，会撞上单 IP 限频。')
            print('     处理方式：评测时把 .env 里的 LIMITS_ENABLED 设为 false 再重启服务，')
            print('     或者把 RATE_LIMIT_PER_IP_PER_MIN 调大。')
        print('   → 在修好之前，下面的数字不能用来判断产品好坏。')
        print()
    print('检索指标')
    print('=' * 78)
    # 按维度分组统计，而不是写死 D1/D3——加了新维度（比如新语料的 D7/D8）之后
    # 这里会自动带上，不需要改代码。
    grouped: dict[str, list[dict]] = {}
    for row in results:
        grouped.setdefault(row['维度'], []).append(row)

    for label, group in list(grouped.items()) + [('合计', results)]:
        recalled = [r for r in group if r['锚点名次'] != '未召回']
        print(
            f'  {label:<18} Top-1 {rate(group, "Top1") * 100:5.1f}%  '
            f'Top-3 {rate(group, "Top3") * 100:5.1f}%  '
            f'Top-5 {rate(group, "Top5") * 100:5.1f}%  '
            f'前{args.top_k}召回 {len(recalled) / len(group) * 100:5.1f}%'
        )

    ranks = [r['锚点名次'] for r in results if r['锚点名次'] != '未召回']
    if ranks:
        print(f'  召回成功时的名次中位数: {statistics.median(ranks):.0f}')
    latencies = [r['耗时ms'] for r in results]
    if latencies:
        print(f'  单次检索耗时: 中位 {statistics.median(latencies):.0f} ms  最长 {max(latencies)} ms')

    if qa_results:
        d5 = [r for r in qa_results if r['类型'].startswith('D5')]
        if d5:
            broken = [r for r in d5 if r['系统故障'] == '是']
            valid = [r for r in d5 if r['系统故障'] == '否']
            if valid:
                ok = sum(1 for r in valid if r['是否拒答'] == '是' or r['是否说明资料中没有'] == '是')
                print(f'\n  D5 边界拒答正确率: {ok}/{len(valid)} = {ok / len(valid) * 100:.1f}%（分母已排除系统故障）')
            if broken:
                print(f'  ⚠️ 另有 {len(broken)} 条因检索故障未能判定，已排除在正确率之外'
                      f'（故障率 {len(broken) / len(d5) * 100:.1f}%）')
        d4 = [r for r in qa_results if r['类型'].startswith('D4')]
        if d4:
            with_unknown = sum(1 for r in d4 if r['未知引用数'] > 0)
            print(f'  D4 抽样 {len(d4)} 条：出现未知引用的 {with_unknown} 条（需人工复核引用是否支撑结论）')
        qa_lat = [r['耗时ms'] for r in qa_results]
        print(f'  单次问答耗时: 中位 {statistics.median(qa_lat):.0f} ms  最长 {max(qa_lat)} ms')

    # ---------- 落盘 ----------
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_retrieval = EVAL_DIR / f'评测结果_检索_{stamp}.csv'
    fields = ['编号', '维度', '语言', '问题', '锚点名次', 'Top1', 'Top3', 'Top5',
              '向量路召回', '关键词路召回', '耗时ms', '第一名来源']
    with out_retrieval.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f'\n检索明细已写入: {out_retrieval.name}')

    if qa_results:
        out_qa = EVAL_DIR / f'评测结果_问答_{stamp}.csv'
        qa_fields = ['编号', '类型', '问题', '是否拒答', '结论', '引用数', '未知引用数',
                     '是否说明资料中没有', '系统故障', '人工判定', '耗时ms', '回答摘要']
        with out_qa.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=qa_fields)
            writer.writeheader()
            writer.writerows(qa_results)
        print(f'问答明细已写入: {out_qa.name}')

    print('\n注意：D2 表格理解、D6 时效版本两个维度当前语料不具备条件，未纳入本次评测。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
