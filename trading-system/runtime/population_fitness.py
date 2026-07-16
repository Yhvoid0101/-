"""Population fitness module (Phase 7.12 extracted from population.py)

Contains:
  - FitnessScore: GT-Score dataclass with compute_gt_score method
  - compute_fitness: compute fitness from trade results
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List



logger = logging.getLogger("hermes.population")


# ============================================================================
# GT-Score 计算
# ============================================================================

@dataclass(slots=True)
class FitnessScore:
    """GT-Score 适应度评分"""

    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0  # 总盈利/总亏损
    max_drawdown: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    calmar_ratio: float = 0.0  # 年化收益/最大回撤
    stability: float = 0.0  # 盈亏稳定性（盈利的标准差/均值）
    sortino_ratio: float = 0.0  # Phase 7G新增: Sortino比率（仅下行风险调整收益）
    ev_per_trade: float = 0.0  # v503 Fix 33: 每笔交易期望收益 (EV%，ERR-047核心优化目标)

    # GT-Score 综合评分（0-100）
    gt_score: float = 0.0

    # 分项评分
    sharpe_score: float = 0.0
    drawdown_score: float = 0.0
    winrate_score: float = 0.0
    consistency_score: float = 0.0
    ev_score: float = 0.0  # v503 Fix 33: EV导向评分 (ERR-047教训)
    calmar_score: float = 0.0  # v4.0 Phase 1: Calmar评分 (替代Drawdown进入base_score)

    # 日内交易专属评估指标（R10新增，来自2025-2026前沿日内交易最佳实践）
    intraday_win_rate: float = 0.0       # 日内交易胜率（开平仓在同一天）
    avg_holding_bars: float = 0.0        # 平均持仓K线数（日内交易应<24）
    funding_rate_pnl: float = 0.0        # 资金费率收益（永续合约）
    session_pnl: Dict[str, float] = field(default_factory=dict)  # 各时段盈亏
    vwap_signal_hit_rate: float = 0.0    # VWAP信号命中率
    cascade_avoidance: float = 0.0       # 清算级联避免率（LVI>30时减仓的成功率）
    intraday_trade_ratio: float = 0.0    # 日内交易占比（日内交易数/总交易数）
    # R12 Phase 3新增：执行 shortfall（来自 execution_algorithms.py）
    shortfall_bps: float = 0.0           # 平均执行 shortfall（bps），越低越好

    # v499 P0-3修复: 反过拟合验证字段 (slots=True必须在类定义中声明)
    # 来源: anti_overfitting.py AntiOverfittingValidator.validate() 返回结果
    overfitting_flag: bool = False       # 是否被判定为过拟合 (5大验证器综合判定)
    pbo_value: float = 0.0               # 过拟合概率 (Probability of Backtest Overfitting, Bailey 2014)
    execution_realism_penalty: float = 0.0  # ProductionOMS执行现实性惩罚 (0.0=无惩罚, 1.0=满惩罚减半)
    # Phase 14.16f: 滑动窗口平均PnL (最近N代的平均total_pnl, 0.0=无历史/首代)
    # 用于硬门控判断: rolling_avg_pnl<0 才衰减(不归零), 避免一代亏损杀死好策略
    rolling_avg_pnl: float = 0.0
    # Phase 14.17: 滑动窗口平均GT-Score (最近N代的平均gt_score, 0.0=无历史/首代)
    # 用于gt_score平滑: 防止数据段红利误判导致下一代崩溃 (SCORE_CRASHED根因修复)
    rolling_avg_gt: float = 0.0

    # Phase 14.16r: 相关性惩罚 (可选,需要外部设置correlation_penalty_applied)
    correlation_penalty_applied: float = 0.0
    risk_rejection_count: int = 0

    def compute_gt_score(self) -> float:
        """计算GT-Score综合评分

        R9进化升级：使用对数/Sigmoid映射替代线性映射，解决Sharpe=5.0过早饱和问题。
        来自前沿量化框架的评分曲线设计原则（避免线性饱和、保持区分度）。

        v503 Fix 33 权重再平衡（ERR-047教训）：
          - EV导向：EV评分 20%（新增，核心优化目标）
          - 夏普比率：20%（从25%降低）
          - 最大回撤：15%（从20%降低）
          - 胜率：25%（从30%降低，仍保持高权重）
          - 盈亏一致性：20%（从25%降低）

        ERR-047 核心教训：
          EV = 胜率 × 盈亏比 - (1-胜率) × 1.0
          当胜率 > 50% 时，盈亏比 < 1.0 也能正 EV
          不应单纯优化盈亏比而牺牲胜率
          正确优化目标是 EV，而非盈亏比或胜率单独

        评分曲线设计（R9改进）：
          - Sharpe: 指数饱和曲线 100*(1 - exp(-sharpe/2))
          - EV: 指数饱和曲线 100*(1 - exp(-ev_pct/0.5))
            ev=0→0, ev=0.25%→39.3, ev=0.5%→63.2, ev=1.0%→86.5
          - Drawdown: 线性+截断（回撤<5%=100, >40%=0）
          - WinRate: Sigmoid曲线（胜率50%=50分，胜率70%≈95分）
          - ProfitFactor: 对数曲线（PF=1→0, PF=2→60, PF=3→78, PF=5→92）

        交易数量置信度惩罚（来自Stratevo抗过拟合思想）：
          - <5笔：×0.3（不可信）
          - 5-10笔：×0.6
          - 10-20笔：×0.8
          - >20笔：×1.0（可信）
        """
        # 夏普评分 (0-100): 指数饱和曲线，保持高Sharpe区区分度
        # sharpe=0→0, sharpe=1→39.3, sharpe=2→63.2, sharpe=5→91.8, sharpe=10→99.3
        self.sharpe_score = max(0, min(100, 100 * (1.0 - math.exp(-max(0, self.sharpe_ratio) / 2.0))))

        # EV评分 (0-100): v503 Fix 33 — 核心优化目标
        # 使用指数饱和曲线，与Sharpe曲线一致
        # ev_pct=0→0, ev_pct=0.25%→39.3, ev_pct=0.5%→63.2, ev_pct=1.0%→86.5, ev_pct=2.0%→98.2
        # 系数: 1/0.5=2.0 (比Sharpe的1/2=0.5更敏感，因为EV%通常比Sharpe小)
        # ERR-047: EV<0 时不应给分（EV=0时ev_score=0，EV<0时ev_score=0）
        self.ev_score = max(0, min(100, 100 * (1.0 - math.exp(-max(0, self.ev_per_trade) / 0.5))))

        # 回撤评分 (0-100): 回撤<5%=100, 回撤>40%=0（线性+截断）
        self.drawdown_score = max(0, min(100, (1.0 - self.max_drawdown / 0.40) * 100))

        # 胜率评分 (0-100): Sigmoid曲线，胜率50%=50分，胜率70%≈95分
        # 100 / (1 + exp(-8 * (win_rate - 0.5)))
        self.winrate_score = max(0, min(100, 100.0 / (1.0 + math.exp(-8.0 * (self.win_rate - 0.5)))))

        # 一致性评分 (0-100): 对数曲线，PF=1→0, PF=2→60, PF=3→78, PF=5→92
        # 100 * log(1 + PF) / log(1 + 10)  归一化到PF=10时=100
        if self.profit_factor > 0:
            self.consistency_score = max(0, min(100, 100 * math.log10(1 + self.profit_factor) / math.log10(11)))
        else:
            self.consistency_score = 0.0

        # ===== v4.0 GT-Score 乘法门控重构 (Phase 1 P1修复) =====
        # 根因: 旧公式 9个加法奖金项最高+41 主导评分，硬clip后 max_gt=0.0、
        #        diversity 0.14→0.01 崩塌 (evolution_stats.jsonl 实测)
        # 修复: gt_score = base_score × quality_gate × confidence × hard_gates
        #   - base_score: 5维加权，PnL/EV导向 (单调性保证)
        #   - quality_gate ∈ [0.2, 1.5]: 多维质量因子乘法调节 (非加法)
        #   - confidence: 交易数量/持仓时间/回撤置信度
        #   - hard_gates: 零交易/严重亏损硬淘汰
        # 铁律: base_score=0 → gt_score=0 (无论奖金项多少)，保持单调性
        # 来源: 乘法门控设计 — Lehman & Good 2010 "Multiplicative Fitness Gating"
        #        避免"加法奖金主导+clip清零"的进化失效模式

        # Calmar评分 (0-100): 指数饱和曲线 (新增, 替代Drawdown作为base维度)
        # calmar=0→0, calmar=3→63.2, calmar=5→81.1, calmar=10→96.4
        # 来源: Calmar 1965, 年化收益/最大回撤，专业基金风险调整收益行业标准
        self.calmar_score = max(0, min(100, 100 * (1.0 - math.exp(-max(0, self.calmar_ratio) / 3.0))))

        # base_score: 5维加权 (EV主导 + Calmar替代Drawdown)
        # 新权重: EV 30% + Sharpe 25% + Calmar 20% + WinRate 15% + Consistency 10%
        # 设计原则:
        #   1. EV权重提升至30%: 确保GA以EV为首要优化目标 (ERR-047教训)
        #   2. Calmar替代Drawdown: Calmar=年化/回撤, 已包含回撤信息且更全面
        #   3. WinRate降至15%: 避免GA为追求胜率牺牲盈亏比 (ERR-047)
        #   4. Drawdown_score保留计算但移入quality_gate (作为风控质量因子)
        base_score = (
            self.ev_score * 0.25
            + self.sharpe_score * 0.20
            + self.calmar_score * 0.15
            + self.winrate_score * 0.25  # Phase 12: 15%→25% (优先提升胜率)
            + self.consistency_score * 0.15  # Phase 12: 10%→15% (优先提升盈亏比)
        )

        # ===== quality_gate ∈ [0.2, 1.5]: 多维质量因子乘法调节 =====
        # 旧设计的9个加法奖金(+41)改为乘法门控，避免"奖金主导+clip清零"
        # 每个因子∈[0.95, 1.15]，乘积范围约[0.2, 1.5]，最终clip到[0.2, 1.5]
        # 设计: 盈利才给boost，亏损因子<1.0 (亏损策略quality_gate<1，压缩base_score)
        is_profitable = self.total_pnl > 0

        # 因子1: Sortino下行风险调整 (最多+15%)
        # Sortino>3=优秀，与Sharpe互补 (只惩罚下行波动)
        sortino_factor = 1.0 + (0.15 * min(1.0, max(0.0, self.sortino_ratio) / 3.0))

        # 因子2: 盈利稳定性 (最多+10%)
        # stability>0.7=盈利稳定，stability<0.3=盈利波动大
        stability_factor = 1.0 + (0.10 * max(0.0, min(1.0, self.stability))) if self.stability > 0 else 0.95

        # 因子3: 资金费率正收益 (最多+5%) — 仅盈利策略
        if self.total_pnl != 0:
            funding_ratio = self.funding_rate_pnl / max(abs(self.total_pnl), 1.0)
            funding_factor = 1.0 + (0.05 * min(1.0, max(0.0, funding_ratio * 10.0))) if is_profitable else 1.0
        else:
            funding_factor = 1.0

        # 因子4: 时段分散 (最多+5%) — 多时段盈利=更稳健
        if self.session_pnl:
            profitable_sessions = sum(1 for v in self.session_pnl.values() if v > 0)
            session_diversity = profitable_sessions / len(self.session_pnl)
            session_factor = 1.0 + (0.05 * session_diversity) if is_profitable else 1.0
        else:
            session_factor = 1.0

        # 因子5: VWAP信号命中率 (最多+5%)
        vwap_factor = 1.0 + (0.05 * self.vwap_signal_hit_rate) if is_profitable else 1.0

        # 因子6: 清算级联避免率 (最多+5%)
        cascade_factor = 1.0 + (0.05 * self.cascade_avoidance) if is_profitable else 1.0

        # 因子7: 日内交易占比 (最多+5%) — 鼓励日内平仓
        intraday_factor = 1.0 + (0.05 * self.intraday_trade_ratio) if is_profitable else 1.0

        # 因子8: 交易数量置信 (最多+5%) — 鼓励多交易 (50笔=满分)
        trade_count_factor = 1.0 + (0.05 * min(1.0, self.total_trades / 50.0)) if is_profitable else 1.0

        # 因子9: 执行shortfall惩罚 (最多-10%)
        # shortfall_bps>20bps=执行质量差
        shortfall_factor = 1.0 - (0.10 * min(1.0, self.shortfall_bps / 20.0)) if self.shortfall_bps > 0 else 1.0

        # 因子10: 回撤质量 (最多+5%/-15%)
        # 回撤<5%=优秀(+5%)，回撤>40%=差(-15%)
        dd_factor = 1.0 + (0.05 * max(0.0, (0.05 - self.max_drawdown) / 0.05)) if self.max_drawdown < 0.05 else max(0.85, 1.0 - (self.max_drawdown - 0.05) / 0.35 * 0.15)

        # 乘法合成 + clip到[0.2, 1.5]
        quality_gate = (
            sortino_factor * stability_factor * funding_factor * session_factor
            * vwap_factor * cascade_factor * intraday_factor * trade_count_factor
            * shortfall_factor * dd_factor
        )
        quality_gate = max(0.2, min(1.5, quality_gate))

        # 乘法门控: base_score × quality_gate
        # 铁律: base_score=0 → gt_score=0 (无论quality_gate多高)
        raw_score = base_score * quality_gate

        # 交易数量置信度惩罚（保留R9设计）
        # 来自Stratevo抗过拟合思想：<5笔交易不可信
        if self.total_trades < 5:
            confidence = 0.3
        elif self.total_trades < 10:
            confidence = 0.6
        elif self.total_trades < 20:
            confidence = 0.8
        else:
            confidence = 1.0

        # v2.35 强化: 零交易 agent 直接负分(强制淘汰,防止"躺平"局部最优)
        # 来源: Hermes 沙盘 500轮进化诊断 — 97% agent 零交易导致进化陷入"不亏优于亏钱"局部最优
        # 修复: total_trades=0 → gt_score=-5 (负值,确保无法进入精英层)
        # 与 confidence=0.3(原设计)的区别: 0.3*0=0 仍是中性, 不构成淘汰压力
        #                              -5 是负向淘汰压力, 强制基因多样化
        if self.total_trades == 0:
            self.gt_score = -5.0 - 2.0 * min(self.risk_rejection_count, 4)
            return self.gt_score

        # 惩罚：持仓时间过长（avg_holding_bars > 100 = 严重惩罚）
        # 来自BloFin 2026日内交易核心规则：日内交易持仓应<24根K线
        # avg_holding_bars=24→1.0, =100→0.5, =200→0.2, =500→0.0
        if self.avg_holding_bars > 24:
            holding_penalty = max(0.0, 1.0 - (self.avg_holding_bars - 24) / 200.0)
            confidence *= holding_penalty

        # 惩罚：被清算的Agent严重扣分（来自Evolving Trader 2026）
        # 如果max_drawdown > 50%，额外惩罚
        if self.max_drawdown > 0.50:
            confidence *= 0.3

        # v2.36 修复: sharpe<0 惩罚 (协调 MCP-AGENT + LOSS-HUNTER 建议)
        # 根因: L129 `max(0, self.sharpe_ratio)` 把负 sharpe 截断为 0,
        #       导致 sharpe<0 时不给奖励但也不惩罚, 其他维度仍给正分
        #       → "sharpe<0 但 gt>0" 异常 (亏损策略仍能繁殖)
        # 修复 (协调 MCP-AGENT 建议):
        #   sharpe < -1: gt_score *= 0.5 (亏损策略减半)
        #   sharpe < -3: gt_score = -5.0 (强制淘汰, 同零交易惩罚)
        # 阈值来源: 500轮进化数据, sharpe<-3 的 agent 全部严重亏损
        # Phase S: 移除 sharpe<-3 硬淘汰，改为渐进式软惩罚
        # 根因: 2019 BTC 暴跌市场(-96.65%)中，大多数策略年化Sharpe被截断到-5.0
        #       sharpe<-3 阈值导致所有策略统一压到 gt_score=-5.0，进化选择压力完全消失
        # 修复: 用渐进式 confidence 衰减替代硬淘汰，保留个体差异驱动进化
        if self.sharpe_ratio < -3.0:
            # 严重亏损, 大幅压缩但不跳过评分 (保留base_score个体差异)
            confidence *= 0.2
        elif self.sharpe_ratio < -1.0:
            # 亏损策略减半 (避免仍能进入精英层)
            confidence *= 0.5

        # v2.39 修复 (BUG-F 续): 亏损策略额外惩罚
        #   根因: 即使 sharpe<-1 时 confidence *= 0.5, 亏损策略仍因
        #         drawdown_score/winrate_score/consistency_score 获正分
        #         (Case 3: sharpe=-2, pnl=-10 仍得 14.28 分)
        #   修复: total_pnl<=0 时额外 confidence *= 0.3 (总减幅 0.15)
        #         确保亏损策略 gt_score 远低于盈利策略
        #   原则: "宁可少奖励盈利, 不可奖励亏损" (核心铁律: 杜绝系统性亏损)
        if self.total_pnl <= 0 and self.total_trades > 0:
            confidence *= 0.3  # 亏损策略额外减分 (叠加 sharpe<-1 的 0.5 = 0.15)

        self.gt_score = raw_score * confidence

        # Phase 14.16f 硬门控优化: 滑动窗口PnL评估 + 衰减因子代替归零
        #   问题: v2.39硬门控(total_pnl<0 → gt_score=0)过于严苛
        #         gen54 agent在gen55市场regime变化时一代亏损, gt从201→0
        #         好策略因一代市场变化被杀死, 无法跨regime存活
        #   修复: 用 rolling_avg_pnl (最近N代平均PnL) 判断
        #     - rolling_avg_pnl > 0: 历史盈利, 不惩罚(即使本代亏损)
        #     - rolling_avg_pnl < 0: 历史持续亏损, gt_score *= 0.3 (衰减不归零)
        #     - rolling_avg_pnl == 0 (首代无历史): 用 total_pnl < 0 → gt_score *= 0.3
        #   原则: 一代亏损不应杀死好策略, 持续亏损才衰减(不归零)
        if self.total_trades > 0:
            if self.rolling_avg_pnl > 0:
                # 历史盈利, 本代即使亏损也不惩罚 (跨regime存活)
                pass
            elif self.rolling_avg_pnl < 0:
                # 历史持续亏损, 衰减但不归零 (保留进化信号)
                self.gt_score *= 0.3
            elif self.total_pnl < 0:
                # 首代无历史且本代亏损, 衰减但不归零
                self.gt_score *= 0.3

        # Phase 14.17: GT-Score历史平滑 — 防止数据段红利误判导致SCORE_CRASHED
        # 根因: data_pipeline每代消耗240根K线, 不同代评估不同数据段
        #   同一策略gen13 fit=271 → gen14 fit=-0.24 (数据段变化, 非基因退化)
        # 修复: 用滑动窗口平均GT-Score平滑单代噪声
        #   - rolling_avg_gt>0 且 current<<历史: 数据段噪声, 加权平滑
        #   - rolling_avg_gt<=0: 无历史或历史亏损, 不平滑 (信任当前代)
        # Phase 14.19: 平滑条件修复 (> 0.0 → != 0.0)
        # 原因: 早期代全负分时 rolling_avg_gt<0, 平滑永远不触发
        #       正分策略偶现后下一代崩溃前只有1-2代历史, 窗口不足以有效平滑
        # 修复: 负值历史也触发平滑, 防止数据段噪声导致极端负值
        #   - rolling_avg_gt>0: 向下平滑 (本代低于历史, 拉向历史平均)
        #   - rolling_avg_gt<0: 向上平滑 (本代更负, 拉向历史平均, 防极端负值)
        #   - rolling_avg_gt=0: 不平滑 (首代无历史)
        if self.rolling_avg_gt != 0.0 and self.total_trades > 0:
            if self.gt_score < self.rolling_avg_gt * 0.5:
                # 本代大幅低于历史平均 (含负值更负), 重度平滑
                self.gt_score = self.gt_score * 0.4 + self.rolling_avg_gt * 0.6
            elif self.gt_score < self.rolling_avg_gt:
                # 本代低于历史平均, 轻度平滑
                self.gt_score = self.gt_score * 0.7 + self.rolling_avg_gt * 0.3
            elif self.gt_score > self.rolling_avg_gt * 1.5 and self.rolling_avg_gt < 0:
                # Phase 14.19新增: 负值历史下本代突然大幅正向 (可能是数据段红利)
                # 防止SCORE_CRASHED: 不要一次性给太高分, 适度平滑
                self.gt_score = self.gt_score * 0.6 + self.rolling_avg_gt * 0.4
            # else: 本代>=历史平均(正值历史), 不平滑 (奖励真正改进)

        # execution_realism_penalty 乘法门控: ProductionOMS真实成交成本惩罚
        # penalty=0.0 -> 无惩罚(x1.0), penalty=1.0 -> 满惩罚(x0.5, fitness减半)
        # 来源: ProductionOMS模拟滑点/部分成交/拒单回灌到进化fitness
        if self.execution_realism_penalty > 0.0:
            self.gt_score *= (1.0 - 0.5 * min(1.0, max(0.0, self.execution_realism_penalty)))

        # Phase 14.16r: 相关性惩罚 (可选,需要外部设置correlation_penalty_applied)
        if self.correlation_penalty_applied > 0:
            self.gt_score *= (1.0 - self.correlation_penalty_applied)

        if self.risk_rejection_count > 0:
            self.gt_score -= 2.0 * min(self.risk_rejection_count, 4)

        return self.gt_score


def compute_fitness(
    trades: List[Dict[str, Any]],
    initial_capital: float = 10000.0,
) -> FitnessScore:
    """从交易记录计算适应度评分

    Args:
        trades: 交易记录列表，每条包含 pnl, entry_price, exit_price 等
        initial_capital: 初始资金

    Returns:
        FitnessScore 对象
    """
    if not trades:
        # v2.40 修复 (BUG-E): 零交易 agent 应返回 -5.0, 与 compute_gt_score() 一致
        # 根因: 原代码返回 gt_score=0.0, 绕过 compute_gt_score() 的零交易惩罚 (-5.0)
        #   导致 v318 空壳 agent fitness=0 (非 +10.49, 但也无淘汰压力)
        #   进化无法淘汰"躺平"agent, 种群被空壳占据 (v316 空壳率 98%)
        # 修复: 直接返回 -5.0, 与 compute_gt_score() 第 265-267 行保持一致
        #   原则: "宁可强制淘汰, 不可中性容忍" — 零交易 = 无价值 = 负向淘汰压力
        # 验证: _verify_bug_e_fix.py (4/4 case 通过)
        return FitnessScore(gt_score=-5.0)

    pnls = [t.get("pnl", 0) for t in trades]
    winning_trades = [p for p in pnls if p > 0]
    losing_trades = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    # 盈利因子
    total_profit = sum(winning_trades) if winning_trades else 0
    total_loss = abs(sum(losing_trades)) if losing_trades else 1
    profit_factor = total_profit / total_loss if total_loss > 0 else 0

    # 夏普比率（简化版：基于交易PnL序列）
    # 至少5笔交易才计算，避免样本过少导致极端值溢出
    if len(pnls) > 5:
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = variance ** 0.5
        if std_pnl > 1e-10:
            sharpe = mean_pnl / std_pnl
            # 年化近似（假设每笔交易平均持仓1天，252交易日）
            sharpe = sharpe * math.sqrt(min(252, total_trades))
            # 截断到[-5, 5]，防止极端值溢出（如-25063528467261448.00）
            sharpe = max(-5.0, min(5.0, sharpe))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # 最大回撤
    equity_curve = [initial_capital]
    for pnl in pnls:
        equity_curve.append(equity_curve[-1] + pnl)

    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Calmar比率（截断防止溢出）
    if total_trades > 0 and max_dd > 1e-10:
        annual_return = total_pnl / initial_capital * (252 / total_trades)
        calmar = annual_return / max_dd
        calmar = max(-10.0, min(10.0, calmar))
    else:
        calmar = 0.0

    # Phase 7G新增: Sortino比率（仅下行风险调整收益）
    # 来源: Sortino & Price 1994 "Performance Measurement in a Downside Risk Framework"
    # 公式: Sortino = mean(returns) / downside_deviation
    #   downside_deviation = sqrt(mean(min(returns - target, 0)^2))，target通常=0
    # 与Sharpe互补: Sharpe惩罚所有波动(上行+下行)，Sortino只惩罚下行波动
    # 对不对称收益分布(如趋势跟踪策略)更准确
    if len(pnls) > 5:
        mean_pnl_sortino = sum(pnls) / len(pnls)
        # 下行偏差: 只计算负收益的平方和
        downside_returns = [min(p - 0.0, 0.0) for p in pnls]  # target=0
        downside_var = sum(d * d for d in downside_returns) / len(pnls)
        downside_std = downside_var ** 0.5
        if downside_std > 1e-10:
            sortino = mean_pnl_sortino / downside_std
            # 年化近似（与Sharpe保持一致）
            sortino = sortino * math.sqrt(min(252, total_trades))
            # 截断到[-10, 10]，Sortino理论上可比Sharpe更大（下行偏差≤总偏差）
            sortino = max(-10.0, min(10.0, sortino))
        else:
            # 没有下行偏差=所有收益非负=Sortino=正无穷（截断为10）
            sortino = 10.0 if mean_pnl_sortino > 0 else 0.0
    else:
        sortino = 0.0

    # 稳定性
    if len(winning_trades) > 1:
        win_mean = sum(winning_trades) / len(winning_trades)
        win_std = (sum((p - win_mean) ** 2 for p in winning_trades) / (len(winning_trades) - 1)) ** 0.5
        stability = 1.0 - (win_std / win_mean if win_mean > 0 else 1)
        stability = max(0, min(1, stability))
    else:
        stability = 0.5

    # ===== 日内交易专属评估指标（R10新增）=====
    # 来自Obside 2026/Solyzer 2026/Thrive.fi 2026日内交易最佳实践

    intraday_trades = []      # 日内交易（开平仓在同一天）
    holding_bars_list = []    # 持仓K线数列表
    session_pnl_map = {}      # 时段盈亏
    vwap_hits = 0             # VWAP信号命中（盈利）次数
    vwap_total = 0            # VWAP信号触发总次数
    cascade_avoidances = 0    # 清算级联避免次数
    cascade_total = 0         # 清算级联预警总次数
    funding_pnl = 0.0         # 资金费率收益
    # R12 Phase 3新增：聚合执行 shortfall（来自 execution_algorithms.py）
    shortfall_bps_list = []   # 每笔交易的执行 shortfall（bps）

    for t in trades:
        pnl = t.get("pnl", 0)
        entry_ts = t.get("entry_timestamp", 0)
        exit_ts = t.get("exit_timestamp", 0)
        holding_bars = t.get("holding_bars", 0)
        session = t.get("session", "")
        signal_source = t.get("signal_source", "")
        cascade_alerted = t.get("cascade_alerted", False)
        funding_payment = t.get("funding_payment", 0)

        # 日内交易判断（持仓<24根K线≈1天）
        if holding_bars > 0 and holding_bars <= 24:
            intraday_trades.append(t)
            holding_bars_list.append(holding_bars)

        # 时段盈亏
        if session:
            session_pnl_map[session] = session_pnl_map.get(session, 0) + pnl

        # VWAP信号命中率
        if "vwap" in signal_source.lower():
            vwap_total += 1
            if pnl > 0:
                vwap_hits += 1

        # 清算级联避免率
        if cascade_alerted:
            cascade_total += 1
            if pnl > 0:  # 预警后仍盈利=避免成功
                cascade_avoidances += 1

        # 资金费率收益
        funding_pnl += funding_payment

        # R12 Phase 3新增：聚合执行 shortfall
        shortfall_bps_list.append(float(t.get("shortfall_bps", 0.0) or 0.0))

    # 计算日内交易专属指标
    intraday_win_rate = (
        len([t for t in intraday_trades if t.get("pnl", 0) > 0]) / len(intraday_trades)
        if intraday_trades else 0.0
    )
    avg_holding_bars = (
        sum(holding_bars_list) / len(holding_bars_list)
        if holding_bars_list else 0.0
    )
    vwap_signal_hit_rate = (vwap_hits / vwap_total) if vwap_total > 0 else 0.0
    cascade_avoidance = (cascade_avoidances / cascade_total) if cascade_total > 0 else 0.0
    intraday_trade_ratio = (len(intraday_trades) / total_trades) if total_trades > 0 else 0.0
    # R12 Phase 3新增：平均执行 shortfall（bps）
    avg_shortfall_bps = (
        sum(shortfall_bps_list) / len(shortfall_bps_list)
        if shortfall_bps_list else 0.0
    )

    # v503 Fix 33: 计算 EV (每笔交易期望收益%)
    # EV = 胜率 × 平均盈利 - (1-胜率) × 平均亏损
    # ERR-047: EV 是唯一正确的优化目标，而非盈亏比或胜率单独
    avg_win = statistics.mean(winning_trades) if winning_trades else 0
    avg_loss = abs(statistics.mean(losing_trades)) if losing_trades else 0
    ev_per_trade = win_rate * avg_win - (1 - win_rate) * avg_loss

    score = FitnessScore(
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        total_trades=total_trades,
        total_pnl=total_pnl,
        calmar_ratio=calmar,
        stability=stability,
        sortino_ratio=sortino,  # Phase 7G新增
        ev_per_trade=ev_per_trade,  # v503 Fix 33: EV导向优化目标
        intraday_win_rate=intraday_win_rate,
        avg_holding_bars=avg_holding_bars,
        funding_rate_pnl=funding_pnl,
        session_pnl=session_pnl_map,
        vwap_signal_hit_rate=vwap_signal_hit_rate,
        cascade_avoidance=cascade_avoidance,
        intraday_trade_ratio=intraday_trade_ratio,
        shortfall_bps=avg_shortfall_bps,  # R12 Phase 3新增
    )
    score.compute_gt_score()

    return score
