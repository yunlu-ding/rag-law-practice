"""编译 Wiki 词条（草稿）。

词条的选题不是拍脑袋定的，它来自评测暴露的缺口：

**"跨业务线差异"这个维度只有 33%，是八个维度里最差的。**
诊断发现那类问题的答案**不存在于任何一个片段里**——
"证券和期货有什么区别"，两条规则各自都在语料里，但"区别"本身没人写过。

所以这批词条就照这个缺口来选：先覆盖评测里最差的那几类对比。

⚠️ 编译产出的是**草稿**（status=draft）。必须人工核对过、
且所有依据都能在语料里查到，才允许参与作答——
词条是一条"权威结论"，写错了会污染所有相关问答，而用户没有东西可以对照。

用法：
    python 工具/编译词条.py                      # 列出所有词条提纲
    python 工具/编译词条.py --compile 全部        # 编译全部
    python 工具/编译词条.py --compile 证券与期货   # 编译标题含该关键字的
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

from app.core.postgres import get_session_factory  # noqa: E402
from app.rag.wiki_compile import CompileRequest, compile_entry  # noqa: E402

# 词条提纲。每条都写明"要覆盖哪些条款"——
# 编译时只把这些条款的原文交给模型，它没有机会引用别的东西。
REQUESTS = [
    CompileRequest(
        slug='securities-vs-futures-suitability',
        title='证券与期货适当性义务的法律规定对照',
        topic='《证券法》与《期货和衍生品法》关于适当性义务的规定对照',
        legal_lines=['证券', '期货'],
        # 触发词只写**主题词**。对比词（区别/差异/不同）一概不放——
        # 它们在每一道对比题里都出现，放进触发词等于没有主题判据。
        # 业务线名也不放，那是另一层判据（词条的 legal_lines 已经在管）。
        # ⚠️ 触发词必须是**有区分度的主题词**，不能是"适当性义务"这种大词——
        # 它在每一道适当性题里都出现，拿它匹配等于没匹配。
        # 实测后果：问"违反适当性义务的**行政处罚**有什么不同"，
        # 本该命中处罚对比词条，却被这条"适当性义务"（5 字，比"行政处罚"4 字长）
        # 抢走了。**预警过，然后它真的发生了。**
        # 这条词条的区分点在"两部法律"，不在"适当性义务"。
        triggers=['适当性义务的法律规定', '两部法律的适当性规定', '证券法和期货法'],
        sources=[
            ('中华人民共和国证券法_20191228.pdf', '第八十八条'),
            ('中华人民共和国证券法_20191228.pdf', '第八十九条'),
            ('中华人民共和国期货和衍生品法_20220420.pdf', '第五十条'),
            ('中华人民共和国期货和衍生品法_20220420.pdf', '第五十一条'),
        ],
        instruction='重点写清楚两者在义务主体、投资者称谓、举证责任上的异同。',
    ),
    CompileRequest(
        slug='investor-classification-three-lines',
        title='证券、期货、基金三条业务线的投资者分类制度对照',
        topic='三条业务线对投资者分类的要求对照',
        legal_lines=['证券', '期货', '基金'],
        triggers=['投资者分类', '普通投资者', '专业投资者', '分类制度', '细化分类'],
        sources=[
            ('证券期货投资者适当性管理办法.txt', '第七条'),
            ('证券期货投资者适当性管理办法.txt', '第十条'),
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第二十六条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第十条'),
        ],
        instruction='写清楚三条线分别怎么分类、用了什么称谓、风险等级档位是否一致。',
    ),
    CompileRequest(
        slug='penalty-comparison',
        title='违反适当性义务的后果对照',
        topic='不同效力层级下违反适当性义务的法律后果',
        legal_lines=['证券', '期货'],
        triggers=['违反适当性义务的后果', '行政处罚对比', '罚款幅度', '处罚依据', '行政处罚'],
        sources=[
            ('中华人民共和国证券法_20191228.pdf', '第一百九十八条'),
            ('证券期货投资者适当性管理办法.txt', '第四十一条'),
            ('中华人民共和国期货和衍生品法_20220420.pdf', '第一百三十五条'),
        ],
        instruction='重点写清楚"依据不同层级的规定，后果完全不同"这一点，'
                    '并列出各自的罚款幅度。',
    ),
    # ---- 下面 9 条来自评测暴露的缺口 ----
    #
    # "跨业务线差异"那一维 15 题里，现有 3 条词条只覆盖了 4 题。
    # 剩下 9 个主题各写一条，覆盖完预期能把这一维推到 70%~80%。
    #
    # ⚠️ 触发词一律写**具体的主题词**，不写"适当性义务"这种大词——
    # 踩过一次：问"行政处罚"却匹配到了"证券与期货适当性义务"词条，
    # 因为通用触发词多 1 分就赢了。大词会让路由挑错词条。
    CompileRequest(
        slug='risk-rating-comparison',
        title='产品风险等级划分要求对照',
        topic='基金线、期货线对产品或者服务风险等级的划分要求对照',
        legal_lines=['基金', '期货'],
        triggers=['风险等级划分', '产品风险等级', '风险评级标准', '风险等级评定'],
        sources=[
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第三十八条'),
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第四十条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第十九条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第二十条'),
        ],
    ),
    CompileRequest(
        slug='revisit-requirements-comparison',
        title='回访要求对照',
        topic='基金线、期货线对投资者回访的要求对照',
        legal_lines=['基金', '期货'],
        triggers=['回访制度', '回访比例', '回访要求', '回访频次'],
        sources=[
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第十二条'),
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第十三条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第三十三条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第三十四条'),
        ],
    ),
    CompileRequest(
        slug='sales-staff-management-comparison',
        title='销售人员管理要求对照',
        topic='基金线、期货线对销售人员履行适当性职责的管理要求对照',
        legal_lines=['基金', '期货'],
        triggers=['销售人员管理', '销售隔离机制', '考核激励机制', '从业人员管理'],
        sources=[
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第九条'),
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第十条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第三十二条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第三十七条'),
        ],
    ),
    CompileRequest(
        slug='record-retention-comparison',
        title='适当性资料保存期限对照',
        topic='证券线、基金线、期货线对适当性相关资料保存期限的要求对照',
        legal_lines=['证券', '基金', '期货'],
        triggers=['保存期限', '档案保存', '资料保存', '保存年限'],
        sources=[
            ('证券期货投资者适当性管理办法.txt', '第三十二条'),
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第十七条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第三十九条'),
        ],
    ),
    CompileRequest(
        slug='matching-principles-comparison',
        title='适当性匹配原则对照',
        topic='证券线、基金线、期货线的投资者风险承受能力与产品风险等级匹配原则对照',
        legal_lines=['证券', '基金', '期货'],
        triggers=['匹配原则', '适当性匹配', '匹配标准', '风险匹配'],
        sources=[
            ('证券期货投资者适当性管理办法.txt', '第十八条'),
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第四十四条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第二十四条'),
        ],
        instruction='重点把各线"能买什么等级"的对应关系列清楚。',
    ),
    CompileRequest(
        slug='self-regulation-division',
        title='自律管理分工对照',
        topic='交易场所与行业协会在适当性管理中的分工',
        legal_lines=['证券', '期货', '基金'],
        triggers=['自律管理分工', '自律组织职责', '自律规则制定', '交易所和行业协会', '自律管理'],
        sources=[
            ('证券期货投资者适当性管理办法.txt', '第五条'),
            ('证券期货投资者适当性管理办法.txt', '第三十六条'),
        ],
    ),
    CompileRequest(
        slug='industry-association-ownership',
        title='各业务线自律管理主体对照',
        topic='证券、期货、基金三条业务线分别由哪个自律组织负责适当性管理',
        legal_lines=['证券', '期货', '基金'],
        triggers=['自律管理主体', '协会归属', '行业协会', '自律管理', '协会'],
        sources=[
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第五条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第四条'),
            ('上海证券交易所会员管理业务指南第1号.docx', '首段'),
        ],
    ),
    CompileRequest(
        slug='risk-catalogue-responsibility',
        title='产品风险等级名录的制定责任对照',
        topic='谁负责制定产品或者服务的风险等级名录',
        legal_lines=['证券', '期货'],
        triggers=['风险等级名录', '名录制定', '风险承受能力名录'],
        sources=[
            ('证券期货投资者适当性管理办法.txt', '第三十六条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第十八条'),
        ],
    ),
    CompileRequest(
        slug='lowest-risk-tolerance-investor',
        title='风险承受能力最低类别投资者的认定对照',
        topic='基金线、期货线对"风险承受能力最低类别投资者"的认定条件对照',
        legal_lines=['基金', '期货'],
        triggers=['风险承受能力最低类别', '最低类别投资者', '最低风险承受能力'],
        sources=[
            ('基金募集机构投资者适当性管理实施指引（试行）.html', '第二十九条'),
            ('期货经营机构交易者适当性管理实施细则.html', '第十二条'),
        ],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description='编译 Wiki 词条')
    parser.add_argument('--compile', default=None, help='"全部"或标题关键字')
    args = parser.parse_args()

    if not args.compile:
        print(f'共 {len(REQUESTS)} 条词条提纲：')
        for request in REQUESTS:
            print(f'  {request.title}')
            print(f'    覆盖条款：{"、".join(f"{f.split("_")[0]} {a}" for f, a in request.sources)}')
        print()
        print('加 --compile 全部 或 --compile <关键字> 开始编译。')
        return 0

    targets = (
        REQUESTS
        if args.compile == '全部'
        else [r for r in REQUESTS if args.compile in r.title]
    )
    if not targets:
        print(f'没有匹配 "{args.compile}" 的词条')
        return 1

    session_factory = get_session_factory()
    for request in targets:
        print(f'编译：{request.title} ...', end='', flush=True)
        try:
            with session_factory() as session:
                entry = compile_entry(session, request)
            mark = '✅ 依据全部核实' if entry.citation_verified else '⚠️ 有依据未核实'
            print(f' {mark}（{len(entry.citations)} 条依据）')
            if not entry.citation_verified:
                for citation in entry.citations:
                    if not citation['核实']:
                        print(f'      ✗ 未核实：{citation["文档"]} {citation["条款"]}')
        except Exception as exc:  # noqa: BLE001
            print(f' 失败：{type(exc).__name__}: {exc}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
