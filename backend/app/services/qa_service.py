from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.llm import generate
from app.models.qa_log import QaLog
from app.rag.prompts import (
    QA_SYSTEM_PROMPT,
    QUERY_REWRITE_PROMPT,
    build_rewrite_prompt,
    build_user_prompt,
)
from app.rag.refusal import (
    KIND_CITATION_MISSING,
    KIND_LOW_SCORE,
    KIND_MODEL_ABSTAIN,
    KIND_NO_HITS,
    KIND_RETRIEVAL_ERROR,
    RefusalDecision,
    decide,
)
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

"""问答服务。

这一层的全部设计都围绕一句话：**让回答可以被核对。**

具体落在三个机制上：

1. **拒答是硬闸门，不是许愿。**
   检索最高分低于阈值时直接拒绝，根本不给模型回答的机会。
   提示词里那句"资料里没有就说没有"是第二道防线——
   **提示词是软约束，拦不住模型在最需要拒答的时候仍然作答。**

2. **引用由后端还原，不给模型编的机会。**
   模型只输出片段 ID；文件名、页码、小节标题全部由后端从真实数据里查。
   顺带得到一个可检测的指标：模型如果引用了不存在的 ID，
   我们能立刻发现——这就是一次可被抓住的编造。

3. **结论用枚举值，而不是自由文本。**
   "违反 / 不违反 / 无法判断 / 说明"这四个值是评测能自动统计的前提。
   让模型自由发挥，评测就只能靠人一条条读。
"""

# 结论的取值范围。用固定枚举，是为了让评测可以按结论分组统计——
# 尤其是"违反"和"不违反"的分布，它能直接暴露模型的判定倾向。
CONCLUSIONS = {'违反', '不违反', '无法判断', '说明'}


@dataclass
class QaOutcome:
    question: str
    session_id: str | None = None
    standalone_query: str | None = None
    rewritten: bool = False
    history_turns: int = 0
    refused: bool = False
    refusal_reason: str | None = None
    # 拒答的种类（retrieval_error / citation_missing / no_hits / low_score）。
    # 只看 `refused` 这个布尔值，"系统坏了"和"库里确实没有"长得一模一样，
    # 而这两种情况对用户的行动指向完全相反。
    refusal_kind: str | None = None
    # 这次答案的**依据强度**（citation_exact / wiki / rerank / degraded）。
    # 它回答的是"这个答案有多硬"，和"答没答"是两个问题。
    evidence: str | None = None
    # 附在答案旁边的提醒（比如"本次没有重排""这个对比主题还没编译词条"）。
    evidence_note: str | None = None
    conclusion: str | None = None
    clause: str | None = None
    reasoning: str | None = None
    assumption: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    parse_ok: bool = True
    retrieval_failed: bool = False
    raw_output: str | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    tokens: int = 0
    log_id: str | None = None
    error: str | None = None


class QaService:
    """问答。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def ask(
        self,
        *,
        question: str,
        top_k: int = 5,
        refuse_threshold: float | None = None,
        session_id: str | None = None,
    ) -> QaOutcome:
        settings = get_settings()
        threshold = (
            settings.refuse_score_threshold if refuse_threshold is None else refuse_threshold
        )
        started = time.perf_counter()
        outcome = QaOutcome(question=question.strip(), session_id=session_id)

        if not outcome.question:
            outcome.error = '问题不能为空'
            return outcome

        # ---- 多轮追问：先把"追问"改写成能独立检索的问题 ----
        #
        # 为什么这一步是多轮的关键：
        # 用户在追问时几乎不会重复主语。上一轮问「向 65 岁以上客户推销高风险基金要注意什么」，
        # 这一轮只会问「那如果客户自己坚持要买呢？」——
        # **这种句子直接拿去检索，几乎必然召不回东西**，因为"那如果"没有指向。
        #
        # 所以先做一次改写，把被省略的主语补回来，再用改写后的句子检索。
        # 注意：**改写只用于检索，不改变用户看到的问题**——
        # 回答里说的仍然是用户问的那句话。
        history = self._load_history(session_id) if session_id else []
        outcome.history_turns = len(history)
        standalone = self._rewrite_query(outcome.question, history)
        outcome.standalone_query = standalone
        outcome.rewritten = standalone != outcome.question

        # ---- 检索（同时落检索日志，便于回溯当时召回了什么）----
        retrieval = RetrievalService(self.db).search(query=standalone, top_k=top_k)
        outcome.retrieval = {
            'log_id': retrieval.log_id,
            'top_k': retrieval.top_k,
            'candidate_k': retrieval.candidate_k,
            'rerank_enabled': retrieval.rerank_enabled,
            'vector_hit_count': retrieval.vector_hit_count,
            'bm25_hit_count': retrieval.bm25_hit_count,
            'fused_count': retrieval.fused_count,
            'exact_hit_count': retrieval.exact_hit_count,
            'citation': retrieval.citation,
            'timings_ms': retrieval.timings_ms,
            'error': retrieval.error,
        }

        # ---- 第一道闸门：三种"答不了"必须分开处理 ----
        #
        # 这三种以前被混成了一种，是真实事故暴露出来的：
        # 一次向量化接口连不上导致两路都空，系统却告诉用户
        # "知识库中没有足够的依据"——而知识库里明明有，是**系统坏了**。
        #
        # 两者对用户的意义完全不同：
        #   没有依据 → 用户该去补资料；
        #   系统故障 → 用户该重试，不该误以为资料缺失。
        # 而且这类故障**不报错**，只是答案变了——如果不区分，
        # 它会被当成"模型不稳定"，然后去调一个根本没错的地方。
        # 判据本身（顺序、每一类的理由、为什么阈值只管混合检索这一条路）
        # 全都写在 app/rag/refusal.py 里。这里只负责把结论落到 outcome 上——
        # 把判定和落地分开，是因为判定要被标定工具和检索调试台复用。
        decision = decide(
            query=standalone,
            hits=retrieval.hits,
            wiki_entry=retrieval.wiki_entry,
            citation=retrieval.citation,
            error=retrieval.error,
            threshold=threshold,
        )
        outcome.refusal_kind = decision.kind
        outcome.evidence = decision.evidence
        outcome.evidence_note = decision.note
        outcome.retrieval['refusal'] = {
            'kind': decision.kind,
            'evidence': decision.evidence,
            'score': decision.score,
            'score_kind': decision.score_kind,
            'threshold': decision.threshold,
        }

        if decision.refuse:
            if decision.kind == KIND_RETRIEVAL_ERROR:
                # 系统故障和"知识库里没有"必须分开说。以前混成一句
                # "没有足够的依据"，结果一次向量化接口欠费，用户被告知
                # "知识库中没有资料"——而资料明明在，是系统坏了。
                outcome.retrieval_failed = True
                outcome.error = (
                    f'检索服务暂时不可用，本次没有生成回答。'
                    f'这是系统故障，不代表知识库里没有内容——请稍后重试。'
                    f'（原因：{retrieval.error}）'
                )
            else:
                outcome.refused = True
                outcome.refusal_reason = decision.reason
                outcome.conclusion = '无法判断'
                outcome.reasoning = _refusal_message(decision)
            outcome.latency_ms = int((time.perf_counter() - started) * 1000)
            outcome.log_id = self._persist(outcome, threshold)
            logger.info(
                '[QA] 拒答: kind=%s question=%r score=%s threshold=%s',
                decision.kind,
                outcome.question,
                decision.score,
                decision.threshold,
            )
            return outcome

        # ---- 生成 ----
        contexts = [
            {
                'label': str(index),
                'chunk_id': hit.get('chunk_id'),
                'filename': hit.get('filename'),
                'page_number': hit.get('page_number'),
                'section_title': hit.get('section_title'),
                'text': hit.get('text'),
                # 条款号、效力层级、是否精确命中——这三样都要传给模型。
                # 少了它们，提示词里"必须说明依据出自哪一层效力"
                # 和"精确命中的那一条是主要依据"就都成了空要求。
                'article_number': hit.get('article_number'),
                'legal_level_label': hit.get('legal_level_label'),
                'validity_label': hit.get('validity_label'),
                'retrieval_sources': hit.get('retrieval_sources') or [],
            }
            for index, hit in enumerate(retrieval.hits, start=1)
        ]
        user_prompt = build_user_prompt(
            question=outcome.question,
            contexts=contexts,
            wiki=retrieval.wiki_entry,
            history=history,
        )

        try:
            generation = generate(system_prompt=QA_SYSTEM_PROMPT, user_prompt=user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception('[QA] 生成失败: question=%r', outcome.question)
            outcome.error = f'生成失败：{type(exc).__name__}: {exc}'
            outcome.latency_ms = int((time.perf_counter() - started) * 1000)
            outcome.log_id = self._persist(outcome, threshold)
            return outcome

        raw = generation.content
        outcome.tokens = generation.total_tokens
        outcome.raw_output = raw
        parsed, parse_ok = _parse_json(raw)
        outcome.parse_ok = parse_ok

        if not parse_ok:
            # 解析失败不丢内容：把原文当作理由保留下来，
            # 同时明确标出"这次没有拿到结构化结果"。
            # 静默失败会让评测把"解析失败"误当成"答错了"。
            outcome.reasoning = raw.strip()
            outcome.conclusion = None
            logger.warning('[QA] 结构化输出解析失败，保留原文: question=%r', outcome.question)
        else:
            outcome.conclusion = _as_text(parsed.get('结论'))
            outcome.clause = _as_text(parsed.get('条款'))
            outcome.reasoning = _as_text(parsed.get('理由'))
            outcome.assumption = _as_text(parsed.get('判断前提'))
            if outcome.conclusion and outcome.conclusion not in CONCLUSIONS:
                # 模型没按枚举值输出。记录下来但不改写它——
                # 这是提示词需要迭代的信号，不是可以在代码里悄悄修正的小问题。
                logger.warning('[QA] 结论不在枚举内: %r', outcome.conclusion)
            outcome.citations, outcome.unknown_citations = _restore_citations(
                parsed.get('依据片段'), retrieval.hits
            )

        # ---- 第二道闸门：模型自己说"无法判断" ----
        if outcome.conclusion == '无法判断':
            outcome.refused = True
            outcome.refusal_reason = '模型判断现有资料不足以回答'
            # 给"拒答"补上是**谁**拒的。判据表里那四种是系统拦下的，
            # 这一种是模型自己说的——混在一起，指标就没法用来定位问题了。
            outcome.refusal_kind = KIND_MODEL_ABSTAIN

        outcome.latency_ms = int((time.perf_counter() - started) * 1000)
        outcome.log_id = self._persist(outcome, threshold)
        logger.info(
            '[QA] 完成: question=%r 结论=%s 引用=%s 未知引用=%s 拒答=%s 耗时=%sms',
            outcome.question,
            outcome.conclusion,
            len(outcome.citations),
            outcome.unknown_citations,
            outcome.refused,
            outcome.latency_ms,
        )
        return outcome

    def _load_history(self, session_id: str, *, limit: int = 6) -> list[dict[str, Any]]:
        """取同一个会话最近几轮问答，按时间正序。

        两个细节：
        1. **检索故障的那一轮不进历史**。它没有有效结论，把它带进上下文
           只会让模型以为"上一轮我说不知道"，从而影响这一轮的判断。
        2. 历史只取最近几轮。放太多轮次会稀释当前问题的权重，
           而且提示词会变得很长——**上下文不是越多越好**。
        """

        if not session_id:
            return []

        statement = (
            select(QaLog)
            .where(QaLog.session_id == session_id)
            .order_by(QaLog.created_at.desc())
            .limit(limit)
        )
        rows = list(self.db.execute(statement).scalars().all())
        rows.reverse()

        return [
            {
                'question': row.question,
                'conclusion': row.conclusion,
                'reasoning': row.reasoning,
            }
            for row in rows
            if not row.retrieval_failed
        ]

    def _rewrite_query(self, question: str, history: list[dict[str, Any]]) -> str:
        """把追问改写成可以独立检索的问题。失败就退回原问题。

        为什么要有"退回"这条路：
        改写只是为了让检索更准，它是一个**增强步骤**，不是必经环节。
        改写失败（模型超时、输出格式不对）不该让整个问答挂掉——
        用原问题检索，最差也只是回到没有多轮时的效果。
        """

        if not history:
            return question

        try:
            result = generate(
                system_prompt=QUERY_REWRITE_PROMPT,
                user_prompt=build_rewrite_prompt(question=question, history=history),
                temperature=0.0,
            )
            rewritten = (result.content or '').strip().strip('"').strip('「」').strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning('[QA] 追问改写失败，改用原问题检索: %s', exc)
            return question

        # 改写结果的长度做一次合理性检查：太短（比如只剩"那如果"）或太长
        # 都说明模型没按预期做事，这时用原问题更安全。
        if not (2 <= len(rewritten) <= 200):
            logger.warning('[QA] 追问改写结果异常，改用原问题: %r', rewritten[:80])
            return question

        if rewritten != question:
            logger.info('[QA] 追问改写: %r → %r', question, rewritten)
        return rewritten

    def _persist(self, outcome: QaOutcome, threshold: float) -> str | None:
        try:
            log = QaLog(
                session_id=outcome.session_id,
                question=outcome.question,
                conclusion=outcome.conclusion,
                clause=outcome.clause,
                reasoning=outcome.reasoning,
                assumption=outcome.assumption,
                citations=outcome.citations,
                unknown_citations=outcome.unknown_citations,
                refused=outcome.refused,
                refusal_reason=outcome.refusal_reason,
                refusal_kind=outcome.refusal_kind,
                refusal_threshold=threshold,
                evidence=outcome.evidence,
                evidence_note=outcome.evidence_note,
                retrieval_failed=outcome.retrieval_failed,
                parse_ok=outcome.parse_ok,
                model=get_settings().model,
                latency_ms=outcome.latency_ms,
                retrieval_log_id=outcome.retrieval.get('log_id'),
            )
            self.db.add(log)
            self.db.commit()
            self.db.refresh(log)
            return log.id
        except Exception:  # noqa: BLE001
            self.db.rollback()
            logger.exception('[QA] 问答记录写入失败（回答本身仍然返回给用户）')
            return None


def _refusal_message(decision: RefusalDecision) -> str:
    """把拒答结论翻成用户看得懂、并且**能据此行动**的一段话。

    每一种拒答都要回答同一个问题：**"我现在该做什么？"**

        · 库里没有这一条   → 去补这部法规（问的是具体条号，答案是确定的）
        · 一条都没召回     → 确认资料有没有入库
        · 分数不足         → 换个问法，或去检索调试台看召回了什么
        · 系统故障         → 稍后重试，**不要去补资料**

    以前这四种共用一句话"相关度不足"，等于给用户的行动指引是错的。

    ⚠️ 这里**不再出现具体分数**。分数是维护者用来标定阈值的，
    不是给用户看的：告诉用户"最高 0.3243 低于阈值 0.5"，
    他既无法核对，也无法行动。分数写进日志和调试台就够了。
    """

    if decision.kind == KIND_CITATION_MISSING:
        return (
            f'{decision.reason} '
            f'如果你认为这部法规应该包含这一条，可以核对一下版本；'
            f'也可以去「文档管理」确认这部法规是否已经入库。'
        )
    if decision.kind == KIND_NO_HITS:
        return (
            '没有检索到与这个问题相关的内容，知识库里可能确实没有这个主题的资料。'
            '可以去「文档管理」确认相关资料是否已经上传入库。'
        )
    if decision.kind == KIND_LOW_SCORE:
        return (
            '检索到了内容，但相关度不足，因此不给出结论。'
            '如果你认为资料应该在库里，可以去「检索调试台」看看召回的内容。'
        )
    return decision.reason or '本次没有给出结论。'


def _parse_json(raw: str) -> tuple[dict[str, Any], bool]:
    """从模型输出里掏出 JSON。

    模型经常不听话：套一层 ```json 代码块、前面加一句"好的，这是结果："。
    这些都属于**可预期的格式问题**，不值得因此判定为失败，
    所以这里做容错解析。真正需要判失败的是"里面没有 JSON"。

    还有一种更隐蔽的瑕疵，是实测里抓到的：
    **模型把 JSON 字符串的结束引号写成了中文引号 `”`**，
    于是字符串没有闭合，解析报 "Unterminated string"——
    但字段内容其实是完整的，只是引号用错了字符。
    所以下面会再试一次"补一个 ASCII 引号"的候选，把这种瑕疵救回来。
    """

    if not raw:
        return {}, False

    text = raw.strip()
    fence = re.search(r'```(?:json)?\s*(.+?)\s*```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start, end = text.find('{'), text.rfind('}')
    for candidate in _json_candidates(text, start, end):
        try:
            # strict=False：允许字符串里出现原始换行符。
            # 模型经常在长句里直接换行，而严格的 JSON 不允许。
            data = json.loads(candidate, strict=False)
            return (data, True) if isinstance(data, dict) else ({}, False)
        except json.JSONDecodeError:
            continue

    return {}, False


def _json_candidates(text: str, start: int, end: int):
    """按"从最干净到最需要修补"的顺序给出候选文本。"""

    yield text

    if start >= 0 and end > start:
        body = text[start : end + 1]
        yield body
        # 修补：字符串结束引号被写成了中文引号，导致未闭合。
        # 在最后一个大括号前补一个 ASCII 引号即可。
        stripped = body.rstrip()
        if stripped.endswith('}'):
            yield stripped[:-1] + '"}'


def _restore_citations(
    claimed: Any,
    hits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """把模型给出的**片段编号**还原成人类可读的引用。

    返回值里第二个是**模型引用的、不存在的片段编号**。

    这一点值得单独说：模型偶尔会引用一个根本不存在的片段编号。
    如果直接把模型的输出展示给用户，用户是看不出来的——
    引用看起来一样正式。
    但我们手里有真实的召回列表，一比对就知道哪些编号是编的。
    **这就是"引用能不能被信任"从口号变成可检测指标的地方。**
    """

    if not isinstance(claimed, list):
        return [], []

    citations: list[dict[str, Any]] = []
    unknown: list[str] = []

    for item in claimed:
        raw = str(item).strip()
        if not raw:
            continue
        # 容错：模型可能写成 "1"、"片段1"、"第1段"。只取其中的数字。
        digits = re.sub(r'\D', '', raw)
        if not digits:
            unknown.append(raw)
            continue
        position = int(digits)
        if not (1 <= position <= len(hits)):
            unknown.append(raw)
            continue
        hit = hits[position - 1]
        citations.append(
            {
                'chunk_id': hit.get('chunk_id'),
                'filename': hit.get('filename'),
                'page_number': hit.get('page_number'),
                'section_title': hit.get('section_title'),
                'score': hit.get('score'),
                'text': hit.get('text'),
            }
        )
    return citations, unknown


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
