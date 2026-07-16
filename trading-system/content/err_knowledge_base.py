# -*- coding: utf-8 -*-
"""错误知识库 (ERR Knowledge Base) — 6大核心短板 #1 联动

从 project_memory.md 解析 19 条 ERR 教训到结构化模块，提供可编程查询的错误模式匹配、
修复方案查询、教训反哺决策能力。

用户核心诉求: "复盘不是摆设，教训库必须自动应用，而非在那儿摆看"
              "迭代经验和教训库使用起来，而不是和复盘一样，在那儿摆看"

定位（与现有模块的职责边界）:
  - 本模块 = ERR数据层: ErrEntry dataclass + 19条历史ERR + match_error_pattern查询
    + get_fix_for_err/get_lessons_by_module/apply_lesson_to_decision纯函数
  - RuntimeKnowledgeBase (evolution_loop.py:977) = 运行时层: 执行ERR入库
  - global_knowledge_base.py = 全局数据层: EIGHT_AGENTS+故障传播+回滚chain
  - 三者分层共存，本模块提供静态历史ERR查询，RuntimeKnowledgeBase处理动态新ERR

设计原则:
  - 结构化: 每条ERR转为ErrEntry dataclass (err_id/version/title/root_cause/fix/lesson/severity/related_module)
  - 可查询: 支持按err_id/模块/严重度/模式匹配查询
  - 可反哺: apply_lesson_to_decision将教训注入决策上下文
  - 可追溯: 每条ERR有version+related_module，可定位到具体版本和模块

来源: project_memory.md (19条v开头ERR教训条目)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# ErrEntry dataclass — 单条ERR教训
# ============================================================

@dataclass
class ErrEntry:
    """单条ERR教训条目

    Attributes:
        err_id: ERR唯一标识 (e.g. "ERR-098", "ERR-20260701-v594")
        version: 关联版本 (e.g. "v479b", "v594")
        title: 简短标题
        root_cause: 根因分析
        fix: 修复方案
        lesson: 教训提炼
        severity: 严重度 ("BLOCK" / "WARN" / "INFO")
        related_module: 关联模块名 (e.g. "sim_live_gap_model", "strategy_engine")
        keywords: 模式匹配关键词列表 (用于match_error_pattern)
    """
    err_id: str
    version: str
    title: str
    root_cause: str
    fix: str
    lesson: str
    severity: str  # "BLOCK" / "WARN" / "INFO"
    related_module: str
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转为dict供JSON序列化"""
        return {
            "err_id": self.err_id,
            "version": self.version,
            "title": self.title,
            "root_cause": self.root_cause,
            "fix": self.fix,
            "lesson": self.lesson,
            "severity": self.severity,
            "related_module": self.related_module,
            "keywords": self.keywords,
        }


# ============================================================
# ERR_DATABASE — 19条历史ERR教训
# ============================================================

ERR_DATABASE: List[ErrEntry] = [
    ErrEntry(
        err_id="ERR-044",
        version="v522",
        title="load_klines max_bars=36000 数据截断导致过拟合",
        root_cause="4个月数据上的优化结果在25个月全量数据上完全失效。load_klines max_bars=36000截断了历史数据。",
        fix="使用全量历史数据(max_bars≥36000)进行参数优化和回测验证。",
        lesson="数据截断导致过拟合，必须用全量数据验证。短数据优化结果在长数据上完全失效。",
        severity="BLOCK",
        related_module="data_pipeline",
        keywords=["max_bars", "数据截断", "过拟合", "load_klines", "数据不足"],
    ),
    ErrEntry(
        err_id="ERR-065",
        version="v517",
        title="confluence_score对PnL无预测力 (Pearson r=0.0096)",
        root_cause="confluence_score与PnL的Pearson r=0.0096(p=0.63), RF重要性仅1.5%。hold_bars是PnL预测因子(d=+0.461, R²=4.79%, p=0.0004, 详见ERR-v88fp)。",
        fix="不使用加权confluence作为决策权重。hold_bars是更强的预测因子。",
        lesson="confluence_score对PnL无预测力，不应使用加权confluence。逆势做空(EMA50>EMA200)实际盈利高于顺势做空。",
        severity="WARN",
        related_module="feature_engineering",
        keywords=["confluence", "预测力", "Pearson", "相关性", "RF重要性", "hold_bars"],
    ),
    ErrEntry(
        err_id="ERR-v88fp",
        version="v88",
        title="hold_bars是PnL最强预测因子 (Cohen's d=+0.461~+0.6292, 三样本稳定)",
        root_cause=(
            "hold_bars与PnL存在中等到大的效应量关系, 三个独立样本验证稳定: "
            "v88历史(n=158, d=+0.461), v98探查(n=252, d=+0.5516, R²=4.79%, p=0.0004), "
            "v99大样本(n=5830, d=+0.6292, 综合评分排名#1=72.22). "
            "hold_bars分桶PnL模式: [0,3)=-25.93(亏损), [3,6)=+30.84, [6,12)=+141.41(峰值), "
            "[12,24)=+113.78, [24,48)=+75.91(回吐). "
            "注: 原_v518_diagnostic_report.json:138声明'42.5%方差(r=+0.943)'数据不一致, "
            "v98实测R²=4.79%(r=0.2189), 已修正。"
        ),
        fix=(
            "v97 regime动态时间止损(REGIME_HOLD_MAP 6状态, 已落地) + "
            "v98第三层仓位融合(4特征pa_body_pct/mtf_trend_direction/atr_pct/adx, KEEP未集成) + "
            "HoldBarsAdapter信号增强(基础设施完成, decide()待集成)"
        ),
        lesson=(
            "短持仓亏损, 中持仓(6-12 bars)盈利峰值, 长持仓利润回吐. 动态持仓的统计学基础. "
            "ERR-109红线: hold_bars是consequence不是cause, 用入场特征预测hold_bars而非直接用作决策. "
            "ERR-v96sign: sign基于d_vs_pnl而非d_vs_hold(Simpson悖论修复)."
        ),
        severity="WARN",
        related_module="feature_engineering",
        keywords=["hold_bars", "预测力", "Cohen_d", "动态持仓", "PnL", "效应量", "Simpson悖论"],
    ),
    ErrEntry(
        err_id="ERR-093",
        version="v553",
        title="ML confluence引擎失败 (AUC=0.4959随机水平)",
        root_cause="RF模型AUC=0.4959(随机水平), consec_loss≤3在当前方向性策略框架下是物理极限。",
        fix="放弃ML confluence引擎，接受consec_loss≤3是物理极限，不强行优化。",
        lesson="consec_loss≤3是物理极限，不应强行优化。ML无法超越随机水平时说明特征集无预测力。",
        severity="WARN",
        related_module="feature_engineering",
        keywords=["ML", "AUC", "随机水平", "consec_loss", "物理极限", "RF模型"],
    ),
    ErrEntry(
        err_id="ERR-098",
        version="v479b",
        title="模拟-实盘差异4因素高保真市场仿真模型",
        root_cause="flat friction_bps=5无法精确模拟市场摩擦，导致模拟-实盘差异率超标。",
        fix="4因素模型: 手续费7bps + 点差per-symbol 1.5-15bps + 滑点Almgren-Chriss平方根K=0.1 + 延迟500ms价格漂移。差异率14.04%≤15%目标。",
        lesson="点差占比44.2%最高。高摩擦币种DOGE/BNB差异率18-20%超标。高波动期延迟成本与波动率成正比。",
        severity="BLOCK",
        related_module="sim_live_gap_model",
        keywords=["摩擦", "点差", "滑点", "延迟", "Almgren-Chriss", "差异率", "friction_bps"],
    ),
    ErrEntry(
        err_id="ERR-099",
        version="v555",
        title="非方向性配对交易突破 (35倍改善)",
        root_cause="方向性策略波动大，需要非方向性策略分散风险。",
        fix="配对交易(高相关对ρ>0.7, Z-score标准化价差)实现最长段总亏损0.32%本金(35倍改善), 但年化仅2.4%。",
        lesson="配对交易与方向性策略组合—方向性贡献收益, 配对分散风险。单独配对交易年化太低。",
        severity="INFO",
        related_module="stat_arb_pairs",
        keywords=["配对交易", "非方向性", "Z-score", "相关性", "分散风险"],
    ),
    ErrEntry(
        err_id="ERR-100",
        version="v90",
        title="ann 0次失败历史性突破 (2/3 Kelly+pos_mult)",
        root_cause="ann不达标，pos_mult不够激进，Kelly系数过保守。",
        fix="(1)W1激进攻仓(BTC 1.1→1.4+27%/SOL 1.2→1.5+25%/BNB 1.2→1.5+25%/LTC 0.9→1.3+44%); (2)月度预算严格(threshold 1.5%→1.0%+scale 0.5→0.3); (3)2/3 Kelly(base最低0.6); (4)GO条件重构。",
        lesson="pos_mult与ann近似线性,W1加仓25-44%完全解决ann不达标。2/3 Kelly+pos_mult加仓提升PnL 40%同时gap仅微恶化0.16pp。max_drawdown加仓后仍远低于15%(single_loss_cap=1.7%硬截断保护)。",
        severity="BLOCK",
        related_module="frontier_enhancement",
        keywords=["Kelly", "2/3", "pos_mult", "加仓", "ann", "W1", "激进攻仓", "single_loss_cap"],
    ),
    ErrEntry(
        err_id="ERR-103",
        version="v572",
        title="LTC段1禁用突破91.7% pass rate",
        root_cause="LTC段1结构性双向亏损(多-10.83%/空-17.02%, 145笔trades 100%止损退出)。",
        fix="v572=v569+禁用LTC段1。禁用段排除KPI计数(不交易=不计入分母, 诚实做法)。",
        lesson="结构性亏损段应禁用而非优化。禁用段排除KPI计数是诚实做法(不交易=不计入分母)。",
        severity="BLOCK",
        related_module="strategy_engine",
        keywords=["LTC", "段1禁用", "结构性亏损", "止损", "KPI计数"],
    ),
    ErrEntry(
        err_id="ERR-104",
        version="v574",
        title="BTC trail=1.5突破92.9%",
        root_cause="BTC段1 sharpe不达标(1.28<1.5)。",
        fix="v574=v572+BTC trail收紧2.0→1.5。BTC段1 sharpe 1.28→1.52达标。",
        lesson="trail收紧锁利提sharpe但牺牲盈利空间(ann 20.99→26.30仍差3.7%)。trail是控sharpe的工具。",
        severity="BLOCK",
        related_module="strategy_engine",
        keywords=["BTC", "trail", "sharpe", "锁利", "收紧"],
    ),
    ErrEntry(
        err_id="ERR-105",
        version="v575",
        title="BTC tp2=3.0突破94.0% BTC段1全达标",
        root_cause="BTC段1 ann不达标(26.30<30%)。",
        fix="v575=v574+BTC tp2放大2.0→3.0。BTC段1 KPI 3/4→4/4全达标, ann 26.30→31.56(+20%达标!)。",
        lesson="trail控sharpe+tp2控ann组合策略有效。trail和tp2是独立维度—trail锁利降方差, tp2让赢家跑远提盈利。",
        severity="BLOCK",
        related_module="strategy_engine",
        keywords=["BTC", "tp2", "ann", "让赢家跑", "trail", "独立维度"],
    ),
    ErrEntry(
        err_id="ERR-106",
        version="v576",
        title="LINK trail=1.5退化教训 (-93% ann)",
        root_cause="LINK不仅对tp1微调极度敏感, 对trail也极度敏感。trail收紧对LINK是灾难(洗出大量trades)。",
        fix="回滚LINK trail到2.0。LINK段2 ann 26.39→1.80崩塌(-93%), trades 153→66(-57%)。",
        lesson="trail收紧效果高度symbol-dependent, 不能从BTC/BNB推广到LINK。每个symbol需独立调参。",
        severity="BLOCK",
        related_module="strategy_engine",
        keywords=["LINK", "trail", "退化", "symbol-dependent", "敏感", "回滚"],
    ),
    ErrEntry(
        err_id="ERR-109",
        version="v518",
        title="4h MTF趋势对齐过滤假设被推翻",
        root_cause="假设逆势trades表现更差，但实际逆势trades表现更好。策略本质是均值回归非趋势跟踪。",
        fix="移除MTF趋势对齐过滤。过滤逆势trades使KPI从75/80→52/80恶化。",
        lesson="逆势trades实际表现更好(aligned win_rate=56.3%/total_pnl=4.70 vs misaligned win_rate=63.2%/total_pnl=7.65)。策略本质是均值回归非趋势跟踪。",
        severity="BLOCK",
        related_module="strategy_engine",
        keywords=["MTF", "趋势对齐", "逆势", "均值回归", "趋势跟踪", "过滤"],
    ),
    ErrEntry(
        err_id="ERR-110",
        version="v518系列",
        title="过滤器迭代教训 (TP/breakeven stop均致退化)",
        root_cause="TP参数修改/TP2放大/F11支撑阻力/breakeven stop均导致策略退化。",
        fix="移除这些过滤器。均值回归策略不适用breakeven stop和支撑阻力过滤。",
        lesson="均值回归策略不适用breakeven stop(截断利润)和支撑阻力过滤(方向矛盾)。",
        severity="WARN",
        related_module="strategy_engine",
        keywords=["TP", "breakeven", "支撑阻力", "过滤器", "退化", "均值回归"],
    ),
    ErrEntry(
        err_id="ERR-110-v561",
        version="v561",
        title="特征融合管道+压力测试+自动闭环 (0/10特征通过)",
        root_cause="6大短板#1/#2/#3/#6未补齐。0/10特征通过P<0.05+|Cohen's d|>0.1双重过滤。",
        fix="特征融合管道+压力测试+自动闭环。GO 5/5 KPI(ann_live77.91%, gap3.09%, dd1.61%, wr62.23%, sh9.42)。",
        lesson="0/10特征通过P<0.05+|d|>0.1双重过滤(与ERR-065一致)。ADX/ATR/RSI/PinBar/Engulfing/VP背离/BOS/CHOCH对PnL无显著预测力→不应使用加权confluence。市场状态分类: quiet最佳($11985, wr62.96%), trend最差($722)。策略本质是均值回归非趋势跟踪。",
        severity="WARN",
        related_module="feature_engineering",
        keywords=["特征", "预测力", "P值", "Cohen's d", "confluence", "市场状态", "均值回归"],
    ),
    ErrEntry(
        err_id="ERR-115",
        version="v519",
        title="尺度bug重大发现 (gap低估513倍)",
        root_cause="v559/v560中`maker_pnl=(raw_pnl-maker_cost)/10`错误, 正确应为`maker_pnl=raw_pnl/10-maker_cost`。",
        fix="正确应为`maker_pnl=raw_pnl/10-maker_cost`。导致gap低估13.52pp(513倍)。",
        lesson="尺度一致性是基础—PnL和成本必须在同一尺度。尺度bug会导致gap评估完全失真。",
        severity="BLOCK",
        related_module="maker_cost_model",
        keywords=["尺度", "bug", "maker_pnl", "gap", "低估", "一致性"],
    ),
    ErrEntry(
        err_id="ERR-116",
        version="v522",
        title="首个真实portfolio级8/8 GO",
        root_cause="fill_rate低+pos_mult过大+adverse_selection导致gap和月亏超标。",
        fix="fill_rate 95%+pos_mult缩减+adverse_selection优化实现8/8 GO。",
        lesson="fill_rate是gap和月亏双重driver, pos_mult缩减根治单亏, Layer2需基于月内running PnL诊断而非固定阈值。",
        severity="BLOCK",
        related_module="backtest_validator",
        keywords=["fill_rate", "pos_mult", "adverse_selection", "gap", "月亏", "portfolio", "GO"],
    ),
    ErrEntry(
        err_id="ERR-20260701-v594",
        version="v594",
        title="分阶测试准入规则 (三阶递进)",
        root_cause="6大短板#4未补齐。缺乏分阶准入机制，策略未达标就可能部署实盘。",
        fix="三阶递进Tier 1(模拟回测)→Tier 2(迷你$1k-$5k)→Tier 3(标准$10k+)。9项Tier1硬约束+7项Tier2+7项Tier3+回滚机制。",
        lesson="严禁跨tier部署, 必须逐级晋升, 任何KPI退化触发回滚。当前Tier 1, 实盘部署禁令激活。回滚机制: tier3_to_tier2(4触发)+tier2_to_tier1(3触发)+Emergency_Stop(5步chain)。",
        severity="BLOCK",
        related_module="staged_admission",
        keywords=["分阶", "准入", "Tier", "部署禁令", "回滚", "实盘"],
    ),
    ErrEntry(
        err_id="ERR-20260701-v595",
        version="v595",
        title="八智能体统一全局经验知识库",
        root_cause="6大短板#5未补齐。8个agent缺乏统一知识库和协作机制。",
        fix="8个agent注册+任务优先级调度器+故障传播图7种+回滚chain3种+接口规范4种+数据交换协议JSON/utf-8 v1.0。",
        lesson="8个agent注册(Coordinator/DataPipeline/StrategyEngine/RiskManager/FeatureEngineering/BacktestValidator/LiveDeployer/KnowledgeBase)。任务优先级L1(风控/实盘/协调)>L2(数据/策略/验证/知识)>L3(特征)。当前故障检测: LiveDeployer.deployment_ban_active(实盘部署禁止)。",
        severity="INFO",
        related_module="global_knowledge_base",
        keywords=["八智能体", "知识库", "调度", "故障传播", "回滚", "接口规范"],
    ),
    ErrEntry(
        err_id="ERR-v73p",
        version="v73p",
        title="Maker模型突破 (差异率35.23%→6.19%)",
        root_cause="Taker成本模型高估摩擦，导致模拟-实盘差异率超标(35.23%)。",
        fix="采用Maker成本模型。将点差从成本转为收益、降低滑点/流动性/延迟影响及应用Maker手续费折扣。",
        lesson="Maker模型将差异率从35.23%降至6.19%(≤15%目标)。SOL和BNB年化收益率未达30%需策略层面优化。",
        severity="INFO",
        related_module="maker_cost_model",
        keywords=["Maker", "Taker", "差异率", "点差", "成本模型"],
    ),
    ErrEntry(
        err_id="ERR-v531",
        version="v531",
        title="LONG_ONLY一刀切错误",
        root_cause="禁止空头放弃一半交易机会。真正问题是逆势空头亏损而非空头本身。",
        fix="移除LONG_ONLY一刀切限制。区分顺势空头和逆势空头。",
        lesson="不应一刀切禁止空头。真正问题是逆势空头亏损而非空头本身。",
        severity="WARN",
        related_module="strategy_engine",
        keywords=["LONG_ONLY", "空头", "一刀切", "逆势"],
    ),
    ErrEntry(
        err_id="ERR-v98gap",
        version="v98",
        title="gap退化的对冲效应根因 (降仓亏损regime导致对冲消失)",
        root_cause=(
            "v97 gap=1.23%是被亏损trades的负向(sim-live)=-990对冲赢钱trades的正向(sim-live)=+1188后的净效应。"
            "v98对ranging regime降仓0.25x+禁用24笔后, 方案B清零禁用trades的sim_pnl/live_pnl, "
            "导致亏损trades的负向(sim-live)对冲消失, 赢钱trades的正向(sim-live)=+1188占主导, "
            "gap从1.23%升至5.69%. 方案D匹配4笔被截断trades(sim_pnl=截断值*mult)生效但数量太少无法扭转。"
            "本质: v97成功(对赢钱trades降仓→gap降) vs v98失败(对亏损regime降仓→对冲消失→gap升), "
            "降仓/禁用亏损regime的固有副作用, 非Bug."
        ),
        fix=(
            "接受gap=5.69%(≤15% Tier1阈值). v98实际表现: ann=109.89%(+18.77pp), sharpe=4.56(+1.05), "
            "Tier1 12/12通过, ranging亏损从-$3807降至-$526. 实盘仍盈利109.89%(非'模拟牛逼实盘亏'). "
            "若需进一步降gap: 修改calc_kpis在计算gap时排除paused trades(方案M), "
            "或降仓时同步调整sim_pnl保持(sim-live)比例(方案N)."
        ),
        lesson=(
            "gap=|total_live-total_sim|/|total_sim|, 亏损trades的sim_pnl比live_pnl更负(单亏截断只截断live_pnl), "
            "提供负向(sim-live)对冲. 清零/降仓亏损trades会消除对冲, 使赢钱trades的正向(sim-live)占主导, gap增大. "
            "降仓亏损regime的gap退化是固有副作用, 与v97对赢钱trades降仓降gap的方向相反. "
            "评估gap时应考虑实盘是否仍盈利(ann>0), 而非只看gap绝对值."
        ),
        severity="WARN",
        related_module="sim_live_gap_model",
        keywords=["gap", "对冲效应", "降仓", "禁用", "单亏截断", "sim_pnl", "live_pnl", "对冲消失", "ranging"],
    ),
]


# ============================================================
# 纯函数: ERR查询 + 模式匹配 + 教训反哺
# ============================================================

def get_all_err_entries() -> List[ErrEntry]:
    """获取所有ERR条目

    Returns:
        19条ErrEntry列表
    """
    return ERR_DATABASE.copy()


def get_err_by_id(err_id: str) -> Optional[ErrEntry]:
    """按err_id查询ERR条目

    Args:
        err_id: ERR唯一标识 (e.g. "ERR-098")

    Returns:
        ErrEntry (未找到返回None)
    """
    for entry in ERR_DATABASE:
        if entry.err_id == err_id:
            return entry
    return None


def get_lessons_by_module(module_name: str) -> List[ErrEntry]:
    """按模块查询ERR教训

    Args:
        module_name: 模块名 (e.g. "strategy_engine", "maker_cost_model")

    Returns:
        该模块关联的ErrEntry列表
    """
    return [e for e in ERR_DATABASE if e.related_module == module_name]


def get_errs_by_severity(severity: str) -> List[ErrEntry]:
    """按严重度查询ERR条目

    Args:
        severity: 严重度 ("BLOCK" / "WARN" / "INFO")

    Returns:
        该严重度的ErrEntry列表
    """
    return [e for e in ERR_DATABASE if e.severity == severity]


def get_errs_by_version(version: str) -> List[ErrEntry]:
    """按版本查询ERR条目

    Args:
        version: 版本号 (e.g. "v594", "v522")

    Returns:
        该版本的ErrEntry列表
    """
    return [e for e in ERR_DATABASE if e.version == version]


def match_error_pattern(error_text: str) -> List[ErrEntry]:
    """模式匹配已知错误

    将错误文本与ERR_DATABASE中每条ErrEntry的keywords进行匹配。
    匹配规则: error_text中包含keyword即视为匹配。

    Args:
        error_text: 错误文本 (e.g. "confluence_score预测力低", "maker_pnl尺度错误")

    Returns:
        匹配的ErrEntry列表 (按匹配关键词数降序排序)
    """
    if not error_text:
        return []
    error_lower = error_text.lower()
    matched: List[Tuple[int, ErrEntry]] = []
    for entry in ERR_DATABASE:
        match_count = 0
        for kw in entry.keywords:
            if kw.lower() in error_lower:
                match_count += 1
        if match_count > 0:
            matched.append((match_count, entry))
    # 按匹配数降序排序
    matched.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in matched]


def get_fix_for_err(err_id: str) -> str:
    """查询ERR的修复方案

    Args:
        err_id: ERR唯一标识

    Returns:
        修复方案字符串 (未找到返回"未知ERR-ID")
    """
    entry = get_err_by_id(err_id)
    if entry is None:
        return f"未知ERR-ID: {err_id}"
    return entry.fix


def get_lesson_for_err(err_id: str) -> str:
    """查询ERR的教训

    Args:
        err_id: ERR唯一标识

    Returns:
        教训字符串 (未找到返回"未知ERR-ID")
    """
    entry = get_err_by_id(err_id)
    if entry is None:
        return f"未知ERR-ID: {err_id}"
    return entry.lesson


def apply_lesson_to_decision(err_entry: ErrEntry, decision_context: Dict) -> Dict:
    """教训反哺决策

    将ERR教训注入决策上下文，返回增强后的决策dict。

    Args:
        err_entry: ErrEntry教训条目
        decision_context: 决策上下文dict (含action/params/risk等字段)

    Returns:
        增强后的决策dict, 新增字段:
        - applied_lesson: 应用的教训字符串
        - applied_fix: 应用的修复方案
        - severity_warning: 严重度警告
        - block_decision: 是否阻止决策 (BLOCK级别ERR阻止)
    """
    result = decision_context.copy()
    result["applied_lesson"] = err_entry.lesson
    result["applied_fix"] = err_entry.fix
    result["severity_warning"] = f"ERR {err_entry.err_id} ({err_entry.severity}): {err_entry.title}"
    # BLOCK级别ERR阻止决策
    result["block_decision"] = (err_entry.severity == "BLOCK")
    if result["block_decision"]:
        result["block_reason"] = f"ERR {err_entry.err_id} 是BLOCK级别，决策被阻止: {err_entry.root_cause}"
    return result


def get_err_summary() -> Dict:
    """获取ERR知识库摘要

    Returns:
        {total, by_severity, by_module, by_version}
    """
    by_severity: Dict[str, int] = {"BLOCK": 0, "WARN": 0, "INFO": 0}
    by_module: Dict[str, int] = {}
    by_version: Dict[str, int] = {}
    for entry in ERR_DATABASE:
        by_severity[entry.severity] = by_severity.get(entry.severity, 0) + 1
        by_module[entry.related_module] = by_module.get(entry.related_module, 0) + 1
        by_version[entry.version] = by_version.get(entry.version, 0) + 1
    return {
        "total": len(ERR_DATABASE),
        "by_severity": by_severity,
        "by_module": by_module,
        "by_version": by_version,
    }


# ============================================================
# main 守卫 (L6 自检)
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("err_knowledge_base.py L6 自检")
    print("=" * 70)

    # 测试1: ERR_DATABASE 完整性
    assert len(ERR_DATABASE) == 21, f"ERR条目数错误: {len(ERR_DATABASE)}, 期望21"
    err_ids = [e.err_id for e in ERR_DATABASE]
    assert "ERR-098" in err_ids, "缺少ERR-098"
    assert "ERR-100" in err_ids, "缺少ERR-100"
    assert "ERR-20260701-v594" in err_ids, "缺少ERR-20260701-v594"
    assert "ERR-v88fp" in err_ids, "缺少ERR-v88fp"
    assert "ERR-v98gap" in err_ids, "缺少ERR-v98gap"
    print(f"✅ ERR_DATABASE: {len(ERR_DATABASE)}条ERR")

    # 测试2: ErrEntry dataclass
    entry = get_err_by_id("ERR-098")
    assert entry is not None, "ERR-098未找到"
    assert entry.version == "v479b", f"version错误: {entry.version}"
    assert entry.severity == "BLOCK", f"severity错误: {entry.severity}"
    assert entry.related_module == "sim_live_gap_model", f"module错误: {entry.related_module}"
    assert len(entry.keywords) > 0, "keywords为空"
    d = entry.to_dict()
    assert d["err_id"] == "ERR-098", f"to_dict错误: {d}"
    print(f"✅ ErrEntry: ERR-098 (v479b, BLOCK, sim_live_gap_model)")

    # 测试3: get_lessons_by_module
    strategy_errs = get_lessons_by_module("strategy_engine")
    assert len(strategy_errs) >= 7, f"strategy_engine ERR数错误: {len(strategy_errs)}"
    maker_errs = get_lessons_by_module("maker_cost_model")
    assert len(maker_errs) >= 2, f"maker_cost_model ERR数错误: {len(maker_errs)}"
    print(f"✅ get_lessons_by_module: strategy_engine={len(strategy_errs)}, maker_cost_model={len(maker_errs)}")

    # 测试4: get_errs_by_severity
    block_errs = get_errs_by_severity("BLOCK")
    warn_errs = get_errs_by_severity("WARN")
    info_errs = get_errs_by_severity("INFO")
    assert len(block_errs) + len(warn_errs) + len(info_errs) == 19, f"严重度分类总数错误"
    assert len(block_errs) >= 8, f"BLOCK ERR数过少: {len(block_errs)}"
    print(f"✅ get_errs_by_severity: BLOCK={len(block_errs)}, WARN={len(warn_errs)}, INFO={len(info_errs)}")

    # 测试5: match_error_pattern
    # 测试"尺度"匹配ERR-115
    matched = match_error_pattern("maker_pnl尺度错误导致gap低估")
    assert len(matched) > 0, "match_error_pattern返回空"
    assert matched[0].err_id == "ERR-115", f"匹配错误: 应为ERR-115, 实际{matched[0].err_id}"
    print(f"✅ match_error_pattern: 'maker_pnl尺度错误' → {matched[0].err_id}")

    # 测试"confluence"匹配ERR-065/ERR-093/ERR-110-v561
    matched = match_error_pattern("confluence预测力低")
    matched_ids = [e.err_id for e in matched]
    assert "ERR-065" in matched_ids, f"confluence应匹配ERR-065: {matched_ids}"
    print(f"✅ match_error_pattern: 'confluence预测力低' → {matched_ids}")

    # 测试"Kelly"匹配ERR-100
    matched = match_error_pattern("Kelly加仓ann不达标")
    matched_ids = [e.err_id for e in matched]
    assert "ERR-100" in matched_ids, f"Kelly应匹配ERR-100: {matched_ids}"
    print(f"✅ match_error_pattern: 'Kelly加仓ann不达标' → {matched_ids}")

    # 测试6: get_fix_for_err + get_lesson_for_err
    fix = get_fix_for_err("ERR-100")
    assert "2/3 Kelly" in fix or "pos_mult" in fix, f"ERR-100修复方案错误: {fix}"
    lesson = get_lesson_for_err("ERR-100")
    assert "PnL 40%" in lesson or "pos_mult" in lesson, f"ERR-100教训错误: {lesson}"
    fix_unknown = get_fix_for_err("ERR-UNKNOWN")
    assert "未知ERR-ID" in fix_unknown, f"未知ERR应返回提示: {fix_unknown}"
    print(f"✅ get_fix_for_err + get_lesson_for_err: ERR-100")

    # 测试7: apply_lesson_to_decision
    entry_block = get_err_by_id("ERR-098")  # BLOCK
    decision = {"action": "deploy", "params": {"capital": 10000}}
    result = apply_lesson_to_decision(entry_block, decision)
    assert result["block_decision"] == True, f"BLOCK ERR应阻止决策"
    assert "block_reason" in result, f"BLOCK ERR应有block_reason"
    assert result["applied_lesson"] == entry_block.lesson
    assert result["applied_fix"] == entry_block.fix
    print(f"✅ apply_lesson_to_decision: BLOCK ERR-098阻止部署决策")

    entry_info = get_err_by_id("ERR-099")  # INFO
    result_info = apply_lesson_to_decision(entry_info, decision)
    assert result_info["block_decision"] == False, f"INFO ERR不应阻止决策"
    print(f"✅ apply_lesson_to_decision: INFO ERR-099不阻止决策")

    # 测试8: get_err_summary
    summary = get_err_summary()
    assert summary["total"] == 19, f"总数错误: {summary['total']}"
    assert summary["by_severity"]["BLOCK"] + summary["by_severity"]["WARN"] + summary["by_severity"]["INFO"] == 19
    assert "strategy_engine" in summary["by_module"], f"缺少strategy_engine模块"
    print(f"✅ get_err_summary: total={summary['total']}, by_severity={summary['by_severity']}")

    print("\n" + "=" * 70)
    print("err_knowledge_base.py L6 自检: 全部 PASS")
    print("=" * 70)
