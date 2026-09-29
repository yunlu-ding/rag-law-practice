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
        # 用户在追问时几乎不会重复主语。上一轮问「离职后带走客户名单违反准则吗」，
        # 这一轮只会问「那如果我事先问过客户呢？」——
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
        best_score, score_source = _best_score(retrieval.hits)

        if not retrieval.hits and retrieval.error:
            # 情况一：检索链路故障，而且没有任何可用结果
            outcome.retrieval_failed = True
            outcome.error = (
                f'检索服务暂时不可用，本次没有生成回答。'
                f'这是系统故障，不代表知识库里没有内容——请稍后重试。'
                f'（原因：{retrieval.error}）'
            )
            outcome.latency_ms = int((time.perf_counter() - started) * 1000)
            outcome.log_id = self._persist(outcome, threshold)
            logger.error(
                '[QA] 检索故障，未生成回答: question=%r error=%s',
                outcome.question,
                retrieval.error,
            )
            return outcome

        if not retrieval.hits:
            # 情况二：检索正常，但一条都没召回到。
            # 这和"分数不够"不一样——分数不够说明有相关内容但不够像；
            # 一条都没有，说明知识库里确实没有这个主题。
            outcome.refused = True
            outcome.refusal_reason = '检索正常执行，但没有召回任何片段'
            outcome.conclusion = '无法判断'
            outcome.reasoning = (
                '没有检索到与这个问题相关的内容，知识库里可能确实没有这个主题的资料。'
                '可以去「文档管理」确认相关资料是否已经上传入库。'
            )
            outcome.latency_ms = int((time.perf_counter() - started) * 1000)
            outcome.log_id = self._persist(outcome, threshold)
            logger.info('[QA] 拒答（无召回）: question=%r', outcome.question)
            return outcome

        if best_score < threshold:
            # 情况三：召回到了，但相关度不够，按阈值拒答
            outcome.refused = True
            outcome.refusal_reason = (
                f'检索到的内容相关度不足（最高 {best_score:.4f}，'
                f'阈值 {threshold:.4f}，取自 {score_source}），'
                f'没有可用依据，因此不作答'
            )
            outcome.conclusion = '无法判断'
            outcome.reasoning = (
                f'检索到了内容，但相关度不足（最高 {best_score:.4f}，低于阈值 {threshold:.4f}），'
                f'因此不给出结论。如果你认为资料应该在库里，'
                f'可以去「检索调试台」看看召回的内容和分数。'
            )
            outcome.latency_ms = int((time.perf_counter() - started) * 1000)
            outcome.log_id = self._persist(outcome, threshold)
            logger.info(
                '[QA] 拒答（分数不足）: question=%r best=%.4f threshold=%.4f',
                outcome.question,
                best_score,
                threshold,
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
                refusal_threshold=threshold,
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


def _best_score(hits: list[dict[str, Any]]) -> tuple[float, str]:
    """取本次检索的"最高相关度"，并说明它来自哪个分数。

    为什么要区分来源：
    重排分是 0~1 的相关性分数，语义清晰、适合做阈值；
    但重排可能被关闭或调用失败（那时会退回原顺序），
    此时分数就变成了向量余弦相似度。
    **阈值必须知道自己是在对哪种分数做判断**，
    否则会出现"换了配置之后拒答行为莫名其妙变了"这种问题。

    ⚠️ 条款**精确命中**要单独处理，而且必须放在最前面判断。

    精确命中是从关系库按（法规名 + 条号）取出来的，它**没有相似度分数**——
    三项分数全是 None。如果不特判，循环里会走到 `fused_score or 0.0`，
    于是"用户指名要的那一条已经拿到手了"反而被判成 0 分，
    低于阈值、直接拒答。这是个自己给自己挖的坑。

    返回 1.0 不是编一个相似度，而是表达"这一类证据的确定性与相似度不是一回事"：
    用户问第几条、我们就找到了第几条，这里面没有"像不像"的成分，
    不该再让它去和阈值比大小。
    """

    if any('exact' in (hit.get('retrieval_sources') or []) for hit in hits):
        return 1.0, 'exact'

    best = -1.0
    source = 'none'
    for hit in hits:
        if hit.get('rerank_score') is not None:
            value, label = float(hit['rerank_score']), 'rerank'
        elif hit.get('vector_score') is not None:
            value, label = float(hit['vector_score']), 'vector'
        else:
            value, label = float(hit.get('fused_score') or 0.0), 'fused'
        if value > best:
            best, source = value, label
    return (best if best >= 0 else 0.0), source


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
