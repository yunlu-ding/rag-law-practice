"""用**线上日志的分数分布**标定拒答阈值。

## 为什么不靠"跑一遍评测"来定这个数

评测集是给"改完代码看看有没有变差"用的，它跑一次要花钱、要时间、要人判。
而阈值是一个**运维参数**：语料加了、重排模型换了、问法变了，它就该跟着动。
把"调一个运维参数"和"跑一次完整评测"绑在一起，结果是——
**没人会去调它**，于是它就永远停在当初拍的那个数上。

所以标定的数据来源是**日志**（`retrieval_log`）：真实用户问了什么、
系统给了多少分，这些每天都在积累，不需要额外做任何事。

## 这个工具的三条纪律

**一、只统计"重排分"。**

日志里的 `score` 是混合量纲：重排生效时是重排分（0~1），重排不可用时是融合分
（0.01 量级），只有一路召回时是 **BM25 分（10~50 量级）**。
这不是假设，是实测——2026-09-29 欠费那天，向量路挂掉，
同一条查询的 `score` 从 0.32 变成了 47.87，在日志里和正常记录长得一模一样。
把这两种数混在一起算分位数，得到的阈值会同时"太松"和"太紧"。

**二、只统计当前语料的查询。**

日志里留着历史阶段（换主题之前）的记录。那些查询的分数分布属于另一套语料，
拿它来定这套语料的阈值，等于用别人的体温计给自己量体温。
过滤办法是"这条日志的命中文件现在还在这套语料里"——语料变了，历史自动出局。

**三、没有负样本就说没有。**

日志里绝大多数的查询是"问库里的东西"，它们的分布天然集中在中高分区。
如果样本里没有低分尾部，那**标不出阈值**——不是"取个最小值就行"。
这种情况工具会直接说不算，并建议跑一次探针（`--probe`）。

用法：
    python 工具/标定拒答阈值.py                # 看日志分布（不联网）
    python 工具/标定拒答阈值.py --probe        # 冷启动：跑一组探针，落日志后再算
    python 工具/标定拒答阈值.py --probe --top-k 5
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from sqlalchemy import select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.retrieval_log import RetrievalLog  # noqa: E402

# 冷启动探针。
#
# 为什么需要它：日志里只有"用户真问过的问题"，而用户问的多半是库里有答案的，
# 于是日志天然缺少低分尾部——**没有负样本，就没有分布可看**。
# 探针就是**一次性**把两类样本都补上：明确该拒的、明确该答的。
#
# ⚠️ 探针不是评测集。它不判对错、不追求覆盖考点，只干一件事：
# **把两个分布摊出来，让人看一眼定个数。** 定完之后，日常漂移就看日志。
PROBES_SHOULD_REFUSE = [
    # 法域之外：库只收国内法规
    '美国 SEC 的投资者适当性规则是怎么规定的',
    '欧盟 MiFID II 对适当性有什么要求',
    '香港证监会对专业投资者怎么界定',
    # 库里没有这一条：条号是确定的，法规里没有
    '证券期货投资者适当性管理办法第八十条怎么规定的',
    '中华人民共和国证券法第三百条讲的是什么',
    # 语料之外：跟金融监管无关
    '怎么做红烧肉',
    '明天上海天气怎么样',
    '帮我写一首关于春天的诗',
    # 生成类任务：知识库里没有"写方案"这种答案
    '帮我写一份适当性管理整改方案',
    '帮我写一封给客户的风险提示邮件',
    '给我推荐几只适合现在买的基金',
]

PROBES_SHOULD_ANSWER = [
    '证券期货投资者适当性管理办法第二十九条怎么规定的',
    '普通投资者和专业投资者是怎么划分的',
    '适当性相关资料要保存多久',
    '基金募集机构投资者适当性管理实施指引什么时候实施',
    '违反适当性义务会有什么行政处罚',
    '创业板个人投资者的资产门槛是多少',
]


def _score_of(item: dict) -> tuple[float | None, str]:
    """从一条日志的命中里取出**可用于标定的分数**并说明来源。

    规则和 app/rag/refusal.py 的 gate_score 必须一致，否则会出现
    "工具算出来的阈值，闸门根本不用那个数"这种最隐蔽的不一致。
    """

    if not item:
        return None, 'empty'
    if (item.get('retrieval_sources') or []) == ['exact']:
        return None, 'exact'
    rerank = item.get('rerank_score')
    if isinstance(rerank, (int, float)):
        return float(rerank), 'rerank'
    return None, 'degraded'


def load_samples(session, *, exclude_probe: bool = False) -> list[dict]:
    """取可用的日志样本。过滤规则见模块开头的第二条纪律。"""

    current = {
        row[0]
        for row in session.execute(select(Document.filename)).all()
    }
    # 探针问题是**合成**的，它们被刻意选在分布的两端（明确该拒的、明确该答的）。
    # 留着看：能立刻得到一张可用的分布图（这也是冷启动唯一的数据来源）；
    # 排除掉看：得到的是**真实用户**的分布，那才是稳态下该看的。
    # 两种都要能看，所以做成开关，而不是替用户决定。
    probe_texts = set(PROBES_SHOULD_REFUSE) | set(PROBES_SHOULD_ANSWER)
    samples: list[dict] = []
    for log in session.execute(
        select(RetrievalLog).order_by(RetrievalLog.created_at)
    ).scalars():
        if exclude_probe and log.query in probe_texts:
            continue
        items = log.items or []
        top = items[0] if items else {}
        filename = str(top.get('filename') or '')
        samples.append(
            {
                'query': log.query,
                'score': _score_of(top)[0],
                'kind': _score_of(top)[1],
                'exact_hit': bool(log.exact_hit_count),
                'in_current_corpus': filename in current,
                'error': log.error,
            }
        )
    return samples


def quantiles(values: list[float], points: tuple[float, ...]) -> dict[float, float]:
    ordered = sorted(values)
    result = {}
    for point in points:
        if not ordered:
            continue
        # 线性插值取分位数。样本量小的时候，宁可用这个方法也不要用
        # "取第 N 个"——后者在几十个样本上会跳得很厉害。
        position = (len(ordered) - 1) * point
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        result[point] = ordered[low] * (1 - weight) + ordered[high] * weight
    return result


def histogram(values: list[float], *, bins: int = 10) -> list[tuple[float, float, int]]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [(low, high, len(values))]
    width = (high - low) / bins
    buckets = [[low + index * width, low + (index + 1) * width, 0] for index in range(bins)]
    for value in values:
        index = min(int((value - low) / width), bins - 1)
        buckets[index][2] += 1
    return [(a, b, c) for a, b, c in buckets]


def report(samples: list[dict], *, threshold: float | None) -> int:
    current = [s for s in samples if s['in_current_corpus'] and not s['error']]
    stale = len(samples) - len(current)
    exact = [s for s in current if s['exact_hit']]
    rerank = [s for s in current if s['kind'] == 'rerank']
    degraded = [s for s in current if s['kind'] == 'degraded' and not s['exact_hit']]

    print(f'日志总条数 {len(samples)}；其中不属于当前语料/当次检索故障的 {stale} 条，已排除')
    print(f'可用于标定的 {len(current)} 条：条款直查命中 {len(exact)} ｜ '
          f'重排分可用 {len(rerank)} ｜ 降级（无重排分）{len(degraded)}')
    print()

    if not rerank:
        print('❌ 没有一条带重排分的记录，标不出阈值。')
        print('   先确认检索链路是通的（重排模型配置正确、没有欠费），再重跑。')
        return 1

    values = [s['score'] for s in rerank]
    print('重排分分布（只看这一种：阈值只在这个量纲上有效）')
    marks = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    table = quantiles(values, marks)
    print('  分位数  ' + '  '.join(f'P{int(p * 100):<6}' for p in marks))
    print('  分数    ' + '  '.join(f'{table.get(p, float("nan")):<6.3f}' for p in marks))
    print(f'  均值 {statistics.fmean(values):.3f} ｜ '
          f'最低 {min(values):.3f} ｜ 最高 {max(values):.3f}')
    print()
    print('  直方图（横轴是分数，纵轴是条数）')
    for low, high, count in histogram(values):
        bar = '█' * min(count, 60)
        print(f'    {low:.3f}~{high:.3f}  {count:>4}  {bar}')
    print()

    # ---- 负样本检查：没有低分尾部就什么都定不了 ----
    #
    # 这一步不能省。日志里绝大多数是"问库里的东西"，分布集中在中高分区；
    # 缺了低分尾部，任何阈值都只是**在正常查询内部切一刀**，
    # 那切掉的不是"没有依据"，而是"依据排得靠后"。
    low_tail = [value for value in values if value < 0.3]
    print('负样本检查')
    if not low_tail:
        print('  ❌ 低分区（< 0.3）一条样本都没有。')
        print('     这说明日志里全是"库里有答案"的查询——**没有负样本，就没有分布。**')
        print('     跑一次 `python 工具/标定拒答阈值.py --probe` 把负样本补上。')
    else:
        print(f'  ✅ 低分区有 {len(low_tail)} 条（占 {len(low_tail) / len(values):.0%}）'
              f'，最低 {min(low_tail):.3f}')

    # ---- 零误伤上界：阈值最多能取到多少 ----
    #
    # 这是整个标定里**唯一一条不该拍的自由度**。阈值是"分数不够就拒答"，
    # 所以它的上限由**答对的题里分数最低的那一道**决定——超过这条线，
    # 系统就开始拒掉自己本来能答对的题。
    #
    # 而实测过之后才发现：这条线**比想象中低得多**。
    # 87 道判"通过"的题里，最低分只有 0.140（"冷静期是什么意思"），
    # 而探针里"给我推荐几只基金"（该拒）是 0.169——
    # **一个该答的题，分数比一个该拒的题还低。**
    # 所以"分数阈值能不能分开两类"这个问题，答案是不能；
    # 阈值只能负责挡住**离题最远的那几条**。
    evaluated = _eval_passed_scores()
    if evaluated:
        zero_harm = min(evaluated)
        print()
        print('零误伤上界（来自评测里判"通过"的题）')
        print(f'  答对的题里分数最低的是 {zero_harm:.3f}')
        print(f'  → 阈值只要取 ≤ {zero_harm:.3f}，就不会拒掉任何一道本来答对的题')
        for probe in sorted(values):
            if probe < zero_harm:
                print(f'    低于这条线、会被它挡住的：{probe:.3f}')

    if threshold is not None:
        print()
        print(f'按阈值 {threshold:.3f} 回放这批历史查询：')
        refused = [s for s in rerank if s['score'] < threshold]
        print(f'  会被拒 {len(refused)} 条（{len(refused) / len(rerank):.0%}）')
        for sample in refused[:10]:
            print(f'    {sample["score"]:.3f}  {sample["query"][:40]}')
    return 0


def _eval_passed_scores() -> list[float]:
    """从检索评测明细里取"判通过"的题的最高重排分。

    允许文件不存在：工具要能在一台没跑过评测的机器上工作，
    只是那样就没有"零误伤上界"这一节。
    """

    path = ROOT / '评测' / '检索评测明细.csv'
    if not path.exists():
        return []
    scores: list[float] = []
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('判定') != '通过':
                continue
            raw = (row.get('最高重排分') or '').strip()
            if raw:
                scores.append(float(raw))
    return scores


def run_probe(*, top_k: int) -> int:
    """冷启动：把探针问题跑一遍并落日志。

    只在"日志还没有负样本"时跑一次。之后靠日志自身的漂移来看阈值要不要动。
    """

    from app.services.retrieval_service import RetrievalService

    questions = PROBES_SHOULD_REFUSE + PROBES_SHOULD_ANSWER
    print(f'跑 {len(questions)} 条探针（该拒 {len(PROBES_SHOULD_REFUSE)} ｜ '
          f'该答 {len(PROBES_SHOULD_ANSWER)}），结果会落进 retrieval_log')
    session_factory = get_session_factory()
    with session_factory() as session:
        service = RetrievalService(session)
        for label, group in (('该拒', PROBES_SHOULD_REFUSE), ('该答', PROBES_SHOULD_ANSWER)):
            print(f'--- {label} ---')
            for question in group:
                outcome = service.search(query=question, top_k=top_k, persist=True)
                top = outcome.hits[0] if outcome.hits else {}
                score, kind = _score_of(top)
                print(
                    f'  {"—" if score is None else format(score, ".3f"):>6}  {kind:<9}'
                    f'{"（条款直查）" if outcome.exact_hit_count else "          "}'
                    f'  {question[:30]}'
                )
    print()
    print('探针已落库。再跑一次不带 --probe 的命令就能看到分布。')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='标定拒答阈值')
    parser.add_argument('--probe', action='store_true', help='冷启动：跑一组探针补充负样本')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='候选阈值，用来回放历史查询看会拒掉多少（默认 0.3）',
    )
    parser.add_argument(
        '--exclude-probe',
        action='store_true',
        help='排除冷启动探针，只看真实用户的分布',
    )
    args = parser.parse_args()

    if args.probe:
        return run_probe(top_k=args.top_k)

    session_factory = get_session_factory()
    with session_factory() as session:
        samples = load_samples(session, exclude_probe=args.exclude_probe)
    return report(samples, threshold=args.threshold)


if __name__ == '__main__':
    raise SystemExit(main())
