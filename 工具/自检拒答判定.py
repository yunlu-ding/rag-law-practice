"""拒答判定的回归自检：**不起网络、不连库、不花钱。**

为什么这一段值得单独测：

拒答是整个系统里**唯一一个"搜到了却不给答案"的地方**。它出错的两种方向
代价完全不同，而且都能悄悄发生：

  · 该拒没拒 → 用户拿到一段编出来的结论，而且看起来和真的一样（最坏）；
  · 不该拒却拒了 → 用户以为库里没资料，转头去补一份其实已有的文件。

这两种错误都不会报错、不会进异常日志。所以判据必须能被一条条钉住。

判据本身是**纯函数**（输入是检索结果的事实，输出是要不要拒），
所以它可以在没有网络、没有数据库的情况下完整验证——这正是把它从
qa_service 里抽出来的原因之一。

用法：
    python 工具/自检拒答判定.py

退出码：0 全部通过，1 有失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from app.rag.refusal import (  # noqa: E402
    EVIDENCE_CITATION,
    EVIDENCE_DEGRADED,
    EVIDENCE_RERANK,
    EVIDENCE_WIKI,
    KIND_CITATION_MISSING,
    KIND_LOW_SCORE,
    KIND_NO_HITS,
    KIND_OK,
    KIND_RETRIEVAL_ERROR,
    decide,
)


def hit(*, rerank=None, sources=('vector', 'bm25'), score=None) -> dict:
    return {
        'chunk_id': 'c1',
        'filename': '某法规.txt',
        'text': '……',
        'retrieval_sources': list(sources),
        'rerank_score': rerank,
        'score': score,
        'fused_score': 0.031,
        'bm25_score': score,
    }


# (用例名, 关键字参数, 期望的 refuse / kind / evidence)
CASES: list[tuple[str, dict, tuple[bool, str, str]]] = [
    (
        '检索故障且没有任何结果 —— 是系统坏了，不是库里没有',
        dict(
            hits=[],
            error='EmbeddingError: 欠费',
            citation={},
            wiki_entry=None,
            threshold=0.3,
        ),
        (True, KIND_RETRIEVAL_ERROR, 'none'),
    ),
    (
        '条款直查命中 —— 阈值拉到 0.99 也必须放行',
        dict(
            hits=[hit(sources=['exact'])],
            error=None,
            citation={
                'usable': True,
                'document_title': '证券期货投资者适当性管理办法',
                'article_number': '第二十九条',
                'hit_count': 1,
            },
            wiki_entry=None,
            threshold=0.99,
        ),
        (False, KIND_OK, EVIDENCE_CITATION),
    ),
    (
        '条号解析成功、库里确实没有这一条 —— 确定性的"没有"',
        dict(
            hits=[hit(rerank=0.62)],
            error=None,
            citation={
                'usable': True,
                'document_title': '证券期货投资者适当性管理办法',
                'article_number': '第八十条',
                'hit_count': 0,
            },
            wiki_entry=None,
            threshold=0.3,
        ),
        (True, KIND_CITATION_MISSING, 'none'),
    ),
    (
        '只解析出条号、没锁定法规 —— 属于"没听懂"，不能判成"库里没有"',
        dict(
            hits=[hit(rerank=0.62)],
            error=None,
            citation={
                'usable': False,
                'reason': '问题里没有识别到已知法规名',
                'article_number': '第二十九条',
            },
            wiki_entry=None,
            threshold=0.3,
        ),
        (False, KIND_OK, EVIDENCE_RERANK),
    ),
    (
        '词条命中 —— 阈值拉到 0.99 也必须放行',
        dict(
            hits=[hit(rerank=0.11)],
            error=None,
            citation={},
            wiki_entry={'title': '证券与期货适当性义务对照'},
            threshold=0.99,
        ),
        (False, KIND_OK, EVIDENCE_WIKI),
    ),
    (
        '对比类问题但没有词条 —— 放行，但要标注"结论未编译"',
        dict(
            hits=[hit(rerank=0.55)],
            error=None,
            citation={},
            wiki_entry=None,
            threshold=0.3,
        ),
        (False, KIND_OK, EVIDENCE_RERANK),
    ),
    (
        '一条都没召回 —— 库里可能确实没有这个主题',
        dict(
            hits=[],
            error=None,
            citation={},
            wiki_entry=None,
            threshold=0.3,
        ),
        (True, KIND_NO_HITS, 'none'),
    ),
    (
        '重排分低于阈值 —— 这才是阈值唯一该管的情况',
        dict(
            hits=[hit(rerank=0.12)],
            error=None,
            citation={},
            wiki_entry=None,
            threshold=0.3,
        ),
        (True, KIND_LOW_SCORE, 'none'),
    ),
    (
        '重排分达标',
        dict(
            hits=[hit(rerank=0.71)],
            error=None,
            citation={},
            wiki_entry=None,
            threshold=0.3,
        ),
        (False, KIND_OK, EVIDENCE_RERANK),
    ),
    (
        # ⚠️ 这一条是这次改动的核心。降级时 score 是 BM25 分（47.87），
        # 旧的 _best_score 会把 47.87 当成"最高相关度"去和阈值比，
        # 于是**系统坏掉的那一天，拒答闸门恰好最松**。
        '重排不可用（欠费那天 score=47.87 是 BM25 分）—— 不能拿它比阈值',
        dict(
            hits=[hit(rerank=None, sources=['bm25'], score=47.86989113146147)],
            error='向量路失败：Arrearage',
            citation={},
            wiki_entry=None,
            threshold=0.3,
        ),
        (False, KIND_OK, EVIDENCE_DEGRADED),
    ),
]


def main() -> int:
    failed = 0
    for index, (name, kwargs, expected) in enumerate(CASES, start=1):
        decision = decide(query=kwargs.pop('query', '问题'), **kwargs)
        got = (decision.refuse, decision.kind, decision.evidence)
        ok = got == expected
        if not ok:
            failed += 1
        print(f'{"✅" if ok else "❌"} {index:>2}. {name}')
        if not ok:
            print(f'      期望 refuse={expected[0]} kind={expected[1]} evidence={expected[2]}')
            print(f'      实际 refuse={got[0]} kind={got[1]} evidence={got[2]}')
        if decision.note:
            print(f'      提醒：{decision.note[:56]}…')
        if decision.reason:
            print(f'      理由：{decision.reason[:56]}…')

    print()
    total = len(CASES)
    print(f'{total - failed}/{total} 通过')
    if failed:
        print('拒答判据是产品里最不能悄悄出错的一段，请先修这里再往下走。')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
