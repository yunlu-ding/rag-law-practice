from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/eval', tags=['eval'])

RETRIEVAL_PREFIX = '评测结果_检索_'
QA_PREFIX = '评测结果_问答_'
STAMP_RE = re.compile(r'(\d{8}_\d{6})')


def _eval_dir() -> Path:
    return Path(get_settings().eval_dir)


def _read_csv(path: Path) -> list[dict[str, str]]:
    # utf-8-sig：这些 CSV 是为了能用 Excel 直接打开而写成带 BOM 的 UTF-8，
    # 读取时要用对应的编码，否则第一列列名会带一个看不见的字符。
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _latest(prefix: str) -> Path | None:
    directory = _eval_dir()
    if not directory.exists():
        return None
    files = sorted(directory.glob(f'{prefix}*.csv'), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _rate(hit: int, total: int) -> float:
    return round(hit / total * 100, 1) if total else 0.0


@router.get('/runs')
def list_runs() -> dict[str, Any]:
    """列出做过的评测轮次。

    评测结果存在文件里，所以"有哪些轮次"就是从文件名读出来的。
    这样即使过了很久，也能翻回某一轮看当时的逐题结果。
    """

    directory = _eval_dir()
    if not directory.exists():
        return {'runs': []}

    runs: dict[str, dict[str, Any]] = {}
    for path in directory.glob('*.csv'):
        match = STAMP_RE.search(path.name)
        if not match:
            continue
        stamp = match.group(1)
        entry = runs.setdefault(stamp, {'stamp': stamp, 'retrieval': None, 'qa': None})
        if path.name.startswith(RETRIEVAL_PREFIX):
            entry['retrieval'] = path.name
        elif path.name.startswith(QA_PREFIX):
            entry['qa'] = path.name

    ordered = sorted(runs.values(), key=lambda item: item['stamp'], reverse=True)
    return {'runs': ordered}


@router.get('/report')
def get_report() -> dict[str, Any]:
    """最近一轮评测的汇总与逐题明细。

    这里只做**读取和聚合**，不做任何判定——判定逻辑在评测脚本里。
    工作台的角色是"把已有的结果摆出来给人看"，
    如果它自己也参与计算指标，就会出现"页面上一个数、CSV 里另一个数"的分裂。
    """

    retrieval_path = _latest(RETRIEVAL_PREFIX)
    qa_path = _latest(QA_PREFIX)

    if retrieval_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='还没有评测结果。先跑一次：python ../评测/运行评测.py --base-url http://127.0.0.1:8000',
        )

    retrieval_rows = _read_csv(retrieval_path)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in retrieval_rows:
        grouped.setdefault(row.get('维度') or '未分类', []).append(row)

    dimensions: list[dict[str, Any]] = []
    for name, rows in grouped.items():
        total = len(rows)
        dimensions.append(
            {
                'name': name,
                'total': total,
                'top1': _rate(sum(1 for r in rows if r.get('Top1') == '命中'), total),
                'top3': _rate(sum(1 for r in rows if r.get('Top3') == '命中'), total),
                'top5': _rate(sum(1 for r in rows if r.get('Top5') == '命中'), total),
                'recall': _rate(sum(1 for r in rows if r.get('锚点名次') not in (None, '', '未召回')), total),
            }
        )

    total = len(retrieval_rows)
    overall = {
        'total': total,
        'top1': _rate(sum(1 for r in retrieval_rows if r.get('Top1') == '命中'), total),
        'top3': _rate(sum(1 for r in retrieval_rows if r.get('Top3') == '命中'), total),
        'top5': _rate(sum(1 for r in retrieval_rows if r.get('Top5') == '命中'), total),
        'recall': _rate(
            sum(1 for r in retrieval_rows if r.get('锚点名次') not in (None, '', '未召回')), total
        ),
    }

    result: dict[str, Any] = {
        'retrieval_file': retrieval_path.name,
        'qa_file': qa_path.name if qa_path else None,
        'dimensions': dimensions,
        'overall': overall,
        'questions': retrieval_rows,
    }

    if qa_path is not None:
        qa_rows = _read_csv(qa_path)
        refusal = [r for r in qa_rows if (r.get('类型') or '').startswith('D5')]
        sampled = [r for r in qa_rows if (r.get('类型') or '').startswith('D4')]
        # 系统故障的那几条不算"判定错误"——它们根本没机会作答。
        # 把它们算进去，拒答正确率就会随网络状况浮动，那就不再是产品指标。
        valid = [r for r in refusal if r.get('系统故障') != '是']
        correct = [
            r for r in valid
            if r.get('是否拒答') == '是' or r.get('是否说明资料中没有') == '是'
        ]
        result['refusal'] = {
            'total': len(refusal),
            'valid': len(valid),
            'excluded_by_failure': len(refusal) - len(valid),
            'correct': len(correct),
            'rate': _rate(len(correct), len(valid)),
        }
        result['citation'] = {
            'sampled': len(sampled),
            'with_unknown': sum(1 for r in sampled if int(r.get('未知引用数') or 0) > 0),
        }
        result['qa_questions'] = qa_rows

    return result
