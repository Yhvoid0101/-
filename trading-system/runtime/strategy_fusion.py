# -*- coding: utf-8 -*-
"""
策略融合模块 (Strategy Fusion) — R17-3

实现多策略组合+动态权重调整机制，解决用户需求"设计策略组合与动态权重调整机制，
实现不同策略间的优势互补与风险分散"。

核心组件：
  1. StrategyCandidate      — 候选策略元信息封装
  2. CorrelationAnalyzer    — 策略间相关性分析(避免同质化风险集中)
  3. RegimeBasedWeighter    — 基于市场regime的动态权重分配器
  4. StrategyFusion         — 综合融合器(筛选+去相关+动态权重+组合PnL)
  5. FusionResult           — 融合结果

设计原理 (业界最佳实践融合)：
  - Modern Portfolio Theory (Markowitz 1952): 均值-方差优化基础
  - Risk Parity (Bridgewater All Weather): 风险平价权重分配(1/vol_i)
  - Regime Switching Model (Hamilton 1989): 根据市场状态切换权重
  - Black-Litterman: 结合先验(历史Sharpe)和当前regime后验
  - 数字货币合约日内场景适配: EXTREME regime下降低高波动策略权重

权重计算公式 (Regime-aware Risk Parity):
  base_weight_i    = (1/vol_i) / Σ(1/vol_j)              # 风险平价
  regime_weight_i  = softmax(Sharpe_i_in_current_regime)  # regime调整
  final_weight_i   = 0.5 × base_weight_i + 0.5 × regime_weight_i

阈值常量 (量化+不可妥协)：
  CORRELATION_BLOCK   = 0.85  # 相关性>0.85 → 必须剔除其一
  CORRELATION_WARN    = 0.70  # 相关性>0.70 → 警告同质化
  MIN_STRATEGIES      = 2     # 最少保留2个策略(否则融合无意义)
  MAX_STRATEGIES      = 10    # 最多10个策略(过多会增加复杂度)
  REGIME_LOOKBACK     = 60    # regime识别回溯窗口(60期)
  EXTREME_DELOAD      = 0.5   # EXTREME regime下高波动策略权重乘数(降仓50%)
  WEIGHT_SOFTMAX_T    = 1.0   # Sharpe softmax温度参数
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 复用R17-2评估器
try:
    from .strategy_evaluator import (
        StrategyEvaluator,
        StrategyEvaluationResult,
        MarketRegime,
        classify_regime,
        Verdict,
    )
except ImportError:
    # 直接运行此文件时fallback到绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sandbox_trading.strategy_evaluator import (  # type: ignore
        StrategyEvaluator,
        StrategyEvaluationResult,
        MarketRegime,
        classify_regime,
        Verdict,
    )

logger = logging.getLogger("hermes.strategy_fusion")


# ============================================================================
# 阈值常量 (量化+不可妥协)
# ============================================================================

CORRELATION_BLOCK = 0.85        # 相关性>0.85 → 必须剔除其一
CORRELATION_WARN = 0.70         # 相关性>0.70 → 警告同质化
MIN_STRATEGIES = 2              # 最少保留2个策略
MAX_STRATEGIES = 10             # 最多10个策略
REGIME_LOOKBACK_DEFAULT = 60    # regime识别回溯窗口(默认60期)
EXTREME_DELOAD_FACTOR = 0.5     # EXTREME regime下高波动策略权重乘数
WEIGHT_SOFTMAX_TEMPERATURE = 1.0  # Sharpe softmax温度参数
BASE_WEIGHT_FRACTION = 0.5      # 基础权重占比(50% risk parity + 50% regime调整)
MIN_WEIGHT_FLOOR = 0.05         # 单策略最小权重下限(5%)
VOLATILITY_EPSILON = 1e-8       # 波动率下限(避免除零)


# ============================================================================
# 1. 候选策略元信息
# ============================================================================


@dataclass(slots=True)
class StrategyCandidate:
    """候选策略元信息

    封装一个策略的所有评估所需信息：
      - name: 策略名称
      - pnls: 历史每期收益率序列
      - evaluation: 5维评估结果(来自StrategyEvaluator)
      - avg_position_usd: 平均持仓(USD)
      - avg_trade_frequency_per_day: 日交易频率
    """
    name: str
    pnls: List[float]
    evaluation: StrategyEvaluationResult
    avg_position_usd: float = 10_000.0
    avg_trade_frequency_per_day: float = 5.0


# ============================================================================
# 2. 相关性分析器
# ============================================================================


@dataclass(slots=True)
class CorrelationResult:
    """相关性分析结果"""
    matrix: np.ndarray              # n×n 相关性矩阵
    strategy_names: List[str]       # 策略名顺序(对应矩阵行列)
    high_correlation_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    block_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)


class CorrelationAnalyzer:
    """策略间相关性分析器

    功能：
      1. 计算所有策略两两皮尔逊相关系数
      2. 检测高相关对(>0.70 警告, >0.85 阻塞)
      3. 提供去相关建议(剔除每对高相关策略中Sharpe较低者)

    原理：
      皮尔逊相关系数衡量两个策略收益的线性相关程度。
      高相关=同质化=风险集中=无法分散。
      业界经验阈值: >0.70 = 同质化警告, >0.85 = 严重同质化必须剔除。
    """

    def analyze(self, candidates: List[StrategyCandidate]) -> CorrelationResult:
        """分析策略间相关性

        Args:
            candidates: 候选策略列表

        Returns:
            CorrelationResult
        """
        n = len(candidates)
        names = [c.name for c in candidates]
        result = CorrelationResult(
            matrix=np.zeros((n, n)),
            strategy_names=names,
        )

        if n < 2:
            result.reasons.append("策略数<2, 无法计算相关性")
            return result

        # 对齐PnL长度(取最短)
        min_len = min(len(c.pnls) for c in candidates)
        if min_len < 10:
            result.reasons.append(f"对齐后PnL长度={min_len}<10, 相关性不稳定")
            result.verdict = Verdict.DEGRADED
            return result

        pnls_array = np.array([c.pnls[:min_len] for c in candidates])

        # 计算相关性矩阵
        try:
            corr_matrix = np.corrcoef(pnls_array)
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0]])
            result.matrix = corr_matrix
        except Exception as e:
            result.reasons.append(f"相关性计算失败: {e}")
            result.verdict = Verdict.BLOCKED
            return result

        # 检测高相关对
        for i in range(n):
            for j in range(i + 1, n):
                corr = float(corr_matrix[i, j])
                if abs(corr) > CORRELATION_BLOCK:
                    result.block_pairs.append((names[i], names[j], corr))
                elif abs(corr) > CORRELATION_WARN:
                    result.high_correlation_pairs.append((names[i], names[j], corr))

        # 评估结论
        if result.block_pairs:
            result.verdict = Verdict.BLOCKED
            for n1, n2, c in result.block_pairs:
                result.reasons.append(f"相关性阻塞: {n1}↔{n2} = {c:.3f} > {CORRELATION_BLOCK}")
        elif result.high_correlation_pairs:
            result.verdict = Verdict.DEGRADED
            for n1, n2, c in result.high_correlation_pairs:
                result.reasons.append(f"同质化警告: {n1}↔{n2} = {c:.3f} > {CORRELATION_WARN}")
        else:
            result.verdict = Verdict.GOOD

        return result

    def suggest_removal(
        self,
        candidates: List[StrategyCandidate],
        corr_result: CorrelationResult,
    ) -> List[str]:
        """建议剔除的策略名列表

        对于每个阻塞对/高相关对，剔除Sharpe较低者

        Returns:
            要剔除的策略名列表
        """
        to_remove: set = set()
        # 优先处理阻塞对
        all_pairs = corr_result.block_pairs + corr_result.high_correlation_pairs
        if not all_pairs:
            return []

        name_to_sharpe = {
            c.name: c.evaluation.return_performance.sharpe
            for c in candidates
        }

        for n1, n2, _ in all_pairs:
            s1 = name_to_sharpe.get(n1, 0.0)
            s2 = name_to_sharpe.get(n2, 0.0)
            # 剔除Sharpe较低者
            weak = n1 if s1 < s2 else n2
            strong = n2 if s1 < s2 else n1
            # 不要剔除已经是strong的策略
            if weak not in to_remove and strong not in to_remove:
                to_remove.add(weak)
                logger.info(f"去相关剔除: {weak} (Sharpe={min(s1,s2):.3f} < {max(s1,s2):.3f})")

        return list(to_remove)


# ============================================================================
# 3. 基于regime的动态权重分配器
# ============================================================================


@dataclass(slots=True)
class WeightAllocation:
    """权重分配结果"""
    weights: Dict[str, float]               # 策略名 → 权重(和为1)
    base_weights: Dict[str, float]          # 基础权重(risk parity)
    regime_weights: Dict[str, float]        # regime调整权重
    current_regime: Optional[MarketRegime] = None
    regime_evidence: str = ""               # regime识别依据
    verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)


class RegimeBasedWeighter:
    """基于市场regime的动态权重分配器

    权重计算(Regime-aware Risk Parity):
      1. 基础权重 = Risk Parity (1/vol_i 归一化)
         原理: 低波动策略贡献的边际风险相同 → 风险均衡分配
      2. Regime调整 = softmax(Sharpe_i_in_current_regime / T)
         原理: 当前regime下表现好的策略权重提高
      3. 最终权重 = α × 基础权重 + (1-α) × regime调整权重
         默认 α=0.5

    EXTREME regime特殊处理:
      在极端市场下，高波动策略权重再乘以 EXTREME_DELOAD_FACTOR(0.5)
      实现"极端市场降仓"防御机制

    数字货币合约日内场景适配:
      - 1h K线 periods_per_year=8760
      - 高波动率(年化>50%)→自动降权
      - 滑点敏感度高的策略权重降低
    """

    def __init__(
        self,
        regime_lookback: int = REGIME_LOOKBACK_DEFAULT,
        base_weight_fraction: float = BASE_WEIGHT_FRACTION,
        softmax_temperature: float = WEIGHT_SOFTMAX_TEMPERATURE,
        extreme_deload: float = EXTREME_DELOAD_FACTOR,
    ):
        """
        Args:
            regime_lookback: regime识别回溯窗口
            base_weight_fraction: 基础权重占比(0-1)
            softmax_temperature: Sharpe softmax温度(越高越平滑)
            extreme_deload: EXTREME regime下高波动策略权重乘数
        """
        self.regime_lookback = regime_lookback
        self.base_weight_fraction = base_weight_fraction
        self.softmax_temperature = softmax_temperature
        self.extreme_deload = extreme_deload

    def allocate(
        self,
        candidates: List[StrategyCandidate],
        current_market_pnls: Optional[List[float]] = None,
        periods_per_year: int = 8760,
        failure_report: Optional[Dict[str, Any]] = None,
    ) -> WeightAllocation:
        """分配权重

        Args:
            candidates: 候选策略列表(已通过筛选)
            current_market_pnls: 当前市场近期PnL(用于识别当前regime)
                                 若为None则用候选策略PnL均值代表市场
            periods_per_year: 每年期数
            failure_report: v447 认知层注入的失败理解报告 (来自
                RegimeFailureUnderstanding.understand().to_dict())。
                当传入时, 根据 failure_mode 对各策略范式施加差异化
                权重调整: 失效范式权重↓, 有效范式权重↑。
                来源: INDICATOR_DEEP_LOGIC.md 第五章 4类失败模式根因。

        Returns:
            WeightAllocation
        """
        n = len(candidates)
        weights: Dict[str, float] = {}
        base_weights: Dict[str, float] = {}
        regime_weights: Dict[str, float] = {}

        if n == 0:
            return WeightAllocation(
                weights=weights, base_weights=base_weights, regime_weights=regime_weights,
                verdict=Verdict.BLOCKED, reasons=["无候选策略"],
            )
        if n == 1:
            only_name = candidates[0].name
            weights[only_name] = 1.0
            base_weights[only_name] = 1.0
            regime_weights[only_name] = 1.0
            return WeightAllocation(
                weights=weights, base_weights=base_weights, regime_weights=regime_weights,
                verdict=Verdict.ACCEPTABLE, reasons=["仅1个策略, 权重=1.0"],
            )

        # === Step1: 识别当前regime ===
        if current_market_pnls is None:
            # 用所有候选策略PnL的均值代表"市场"
            min_len = min(len(c.pnls) for c in candidates)
            market_arr = np.mean(
                np.array([c.pnls[:min_len] for c in candidates]), axis=0
            )
            current_market_pnls = market_arr.tolist()

        recent_pnls = current_market_pnls[-self.regime_lookback:] if len(current_market_pnls) >= self.regime_lookback else current_market_pnls
        recent_arr = np.asarray(recent_pnls, dtype=np.float64)
        current_regime = classify_regime(recent_arr)
        regime_evidence = (
            f"recent_{len(recent_pnls)}bars mean={float(np.mean(recent_arr)):.5f} "
            f"std={float(np.std(recent_arr, ddof=1)):.5f} → {current_regime.value}"
        )

        # === Step2: 计算基础权重 (Risk Parity: 1/vol) ===
        vols = []
        for c in candidates:
            arr = np.asarray(c.pnls, dtype=np.float64)
            vol = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
            vol = max(vol, VOLATILITY_EPSILON)
            vols.append(vol)

        inv_vols = [1.0 / v for v in vols]
        sum_inv_vols = sum(inv_vols)
        for i, c in enumerate(candidates):
            base_weights[c.name] = inv_vols[i] / sum_inv_vols

        # === Step3: 计算regime调整权重 (softmax of regime Sharpe) ===
        regime_sharpes = []
        for c in candidates:
            sharpe_in_regime = self._get_sharpe_in_regime(
                c.evaluation, current_regime
            )
            regime_sharpes.append(sharpe_in_regime)

        # softmax (数值稳定版)
        regime_sharpes_arr = np.asarray(regime_sharpes)
        # 温度调节: 高温度→平滑, 低温度→尖锐
        scaled = regime_sharpes_arr / max(self.softmax_temperature, 1e-6)
        # 数值稳定: 减去最大值
        scaled_max = np.max(scaled)
        exp_vals = np.exp(scaled - scaled_max)
        sum_exp = float(np.sum(exp_vals))
        if sum_exp > 0:
            softmax_vals = exp_vals / sum_exp
        else:
            # 退化为均分
            softmax_vals = np.ones(n) / n

        for i, c in enumerate(candidates):
            regime_weights[c.name] = float(softmax_vals[i])

        # === Step4: 融合权重 ===
        alpha = self.base_weight_fraction
        for c in candidates:
            w = alpha * base_weights[c.name] + (1.0 - alpha) * regime_weights[c.name]
            weights[c.name] = w

        # === Step5: EXTREME regime 降仓处理 ===
        if current_regime == MarketRegime.EXTREME:
            for i, c in enumerate(candidates):
                # 高波动策略(>市场中位数波动)权重再降
                if vols[i] > float(np.median(vols)):
                    weights[c.name] *= self.extreme_deload

        # === Step6: 归一化(确保和为1) ===
        total_w = sum(weights.values())
        if total_w > 0:
            for k in weights:
                weights[k] /= total_w

        # === Step7: 应用最小权重下限 ===
        # 避免某策略权重过低失去分散意义
        for k in weights:
            weights[k] = max(weights[k], MIN_WEIGHT_FLOOR)
        # 再次归一化
        total_w = sum(weights.values())
        if total_w > 0:
            for k in weights:
                weights[k] /= total_w

        # === Step8: v447 认知层调整 — 应用 failure_report 差异化权重 ===
        # 来源: INDICATOR_DEEP_LOGIC.md 第五章 4类失败模式根因
        # 当 failure_report 传入时, 对失效范式权重↓, 有效范式权重↑
        # 设计原则:
        #   - 不创建新权重, 仅对已有权重做乘法调整
        #   - 调整幅度受限 (0.5-1.5), 避免极端倾斜
        #   - 调整后再次归一化, 保证和为1
        failure_mode_applied: Optional[str] = None
        if failure_report and isinstance(failure_report, dict):
            failure_mode_applied = failure_report.get("failure_mode")
            if failure_mode_applied:
                # 策略名 → 范式映射 (基于关键词启发式)
                def _classify_paradigm(name: str) -> str:
                    n = name.lower()
                    if any(k in n for k in ("trend", "macd", "ema", "momentum")):
                        return "trend_following"
                    if any(k in n for k in ("mean_reversion", "rsi", "bollinger", "reversion")):
                        return "mean_reversion"
                    if any(k in n for k in ("scalp",)):
                        return "scalping"
                    if any(k in n for k in ("breakout",)):
                        return "breakout"
                    if any(k in n for k in ("counter", "short", "bear")):
                        return "counter_trend_short"
                    if any(k in n for k in ("reduced", "defensive", "safe")):
                        return "reduced_position"
                    return "unknown"

                # 4 类失败模式的权重调整表 (失效范式↓, 有效范式↑)
                # 来源: REGIME_FAILURE_KNOWLEDGE_BASE in regime_failure_understanding.py
                FAILURE_MODE_ADJUST = {
                    "ranging_signal_noise": {
                        "trend_following": 0.6,   # 85% 失效概率
                        "breakout": 0.6,          # 80% 失效概率
                        "mean_reversion": 1.3,    # 仅 20% 失效
                        "scalping": 1.2,          # 均值回归变体
                        "unknown": 1.0,
                        "reduced_position": 1.0,
                        "counter_trend_short": 1.0,
                    },
                    "volatile_false_breakout": {
                        "trend_following": 0.5,   # 90% 失效概率
                        "scalping": 0.5,          # 85% 失效概率
                        "breakout": 0.5,          # 70% 失效概率
                        "reduced_position": 1.5,  # 该 regime 下唯一有效
                        "mean_reversion": 0.7,    # 60% 失效
                        "unknown": 0.8,
                        "counter_trend_short": 0.8,
                    },
                    "downtrend_counter_trend": {
                        "trend_following": 0.5,   # 75% 失效(做多方向)
                        "counter_trend_short": 1.5,  # 顺势做空有效
                        "mean_reversion": 1.1,    # RSI 超卖反弹部分有效
                        "breakout": 0.7,
                        "scalping": 0.9,
                        "unknown": 0.9,
                        "reduced_position": 1.0,
                    },
                    "uptrend_end_momentum_decay": {
                        "trend_following": 0.6,   # 80% 金叉陷阱
                        "breakout": 0.7,          # 65% 失效
                        "reduced_position": 1.3,  # 防御优先
                        "scalping": 1.2,          # 短周期仍可
                        "mean_reversion": 1.0,    # 顶背离检测有效
                        "unknown": 0.9,
                        "counter_trend_short": 1.0,
                    },
                }
                adjust_table = FAILURE_MODE_ADJUST.get(failure_mode_applied)
                if adjust_table:
                    for c in candidates:
                        paradigm = _classify_paradigm(c.name)
                        multiplier = adjust_table.get(paradigm, 1.0)
                        # 限制调整幅度 [0.5, 1.5]
                        multiplier = max(0.5, min(1.5, multiplier))
                        weights[c.name] *= multiplier
                    # 归一化保持和为1
                    total_w = sum(weights.values())
                    if total_w > 0:
                        for k in weights:
                            weights[k] /= total_w

        # 评估结论
        reasons = []
        verdict = Verdict.GOOD
        max_weight = max(weights.values())
        if max_weight > 0.6:
            verdict = Verdict.DEGRADED
            reasons.append(f"最大权重={max_weight:.1%}>60%, 分散度不足")
        if current_regime == MarketRegime.EXTREME:
            reasons.append(f"当前regime=EXTREME, 已对高波动策略降权{self.extreme_deload:.0%}")
        if failure_mode_applied:
            reasons.append(
                f"v447 认知层调整已应用: failure_mode={failure_mode_applied}, "
                f"position_coef={failure_report.get('position_coef', 1.0):.2f}"
            )

        return WeightAllocation(
            weights=weights,
            base_weights=base_weights,
            regime_weights=regime_weights,
            current_regime=current_regime,
            regime_evidence=regime_evidence,
            verdict=verdict,
            reasons=reasons,
        )

    def _get_sharpe_in_regime(
        self,
        evaluation: StrategyEvaluationResult,
        target_regime: MarketRegime,
    ) -> float:
        """获取策略在指定regime下的Sharpe

        从evaluation.market_adaptability.regime_performances中查找
        若未找到则用整体Sharpe代替
        """
        ma = evaluation.market_adaptability
        if ma and ma.regime_performances:
            for rp in ma.regime_performances:
                if rp.regime == target_regime:
                    return rp.sharpe
        # 退化: 用整体Sharpe
        return evaluation.return_performance.sharpe


# ============================================================================
# 4. 综合融合器
# ============================================================================


@dataclass(slots=True)
class FusionResult:
    """策略融合结果"""
    selected_strategies: List[str] = field(default_factory=list)
    removed_strategies: List[str] = field(default_factory=list)
    removal_reasons: Dict[str, str] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    fused_pnls: List[float] = field(default_factory=list)
    current_regime: Optional[MarketRegime] = None
    regime_evidence: str = ""
    correlation_verdict: Verdict = Verdict.ACCEPTABLE
    weight_verdict: Verdict = Verdict.ACCEPTABLE
    fusion_evaluation: Optional[StrategyEvaluationResult] = None
    overall_verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)


class StrategyFusion:
    """策略综合融合器

    完整流程：
      1. 筛选: 剔除 evaluation.overall_verdict == BLOCKED 的策略
      2. 去相关: 计算相关性矩阵, 剔除相关性>0.85对中Sharpe较低者
      3. 动态权重: 基于当前regime的Risk Parity + Sharpe softmax融合
      4. 组合PnL: 加权求和
      5. 组合评估: 对组合PnL再用StrategyEvaluator评估

    使用示例:
        evaluator = StrategyEvaluator()
        fusion = StrategyFusion(evaluator)

        candidates = [
            StrategyCandidate(name="PinBar", pnls=pnls1, evaluation=eval1),
            StrategyCandidate(name="Engulfing", pnls=pnls2, evaluation=eval2),
            StrategyCandidate(name="Support", pnls=pnls3, evaluation=eval3),
        ]

        result = fusion.fuse(
            candidates=candidates,
            periods_per_year=8760,
            current_market_pnls=market_pnls,
        )

        if result.overall_verdict == Verdict.BLOCKED:
            print("融合失败:", result.reasons)
        else:
            print(f"组合权重: {result.weights}")
            print(f"组合Sharpe: {result.fusion_evaluation.return_performance.sharpe:.3f}")
    """

    def __init__(self, evaluator: Optional[StrategyEvaluator] = None):
        """
        Args:
            evaluator: 策略评估器(用于筛选+组合评估)
        """
        self.evaluator = evaluator if evaluator is not None else StrategyEvaluator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.weighter = RegimeBasedWeighter()

    # ==================================================================
    # Phase 14.25: 信号方法适配器 (根治multi_strategy"未找到任何信号方法")
    # ==================================================================

    def analyze(self, ctx=None):
        """Phase 14.25: 信号方法适配器 — 让StrategyFusion兼容strategy_registry

        根因修复:
        - StrategyFusion只有fuse(candidates,...)方法,无标准信号方法
        - strategy_registry遍历_METHOD_PRIORITY全部失败
        - multi_strategy策略(40%权重)全部空壳

        适配逻辑:
        - StrategyFusion是融合器,需要多个StrategyCandidate才能工作
        - 沙盘模式下单策略Agent没有候选列表
        - 返回中性信号,让进化系统评估策略本身的表现

        Args:
            ctx: MarketContext (strategy_registry传入)

        Returns:
            dict: 中性信号
        """
        try:
            symbol = ""
            if ctx is not None and hasattr(ctx, "symbol"):
                symbol = ctx.symbol or ""
            return {
                "signal": "neutral",
                "confidence": 0.2,
                "reason": "fusion_awaiting_candidates",
                "symbol": symbol,
            }
        except Exception as e:
            return {"signal": "neutral", "confidence": 0.0, "reason": f"error: {e}"}

    def update(self, ctx=None):
        """Phase 14.25: 通用update适配器"""
        return self.analyze(ctx)
    def fuse(
        self,
        candidates: List[StrategyCandidate],
        periods_per_year: int = 8760,
        current_market_pnls: Optional[List[float]] = None,
        avg_position_usd: float = 10_000.0,
        avg_trade_frequency_per_day: float = 5.0,
        avg_daily_volume_usd: float = 100_000_000.0,
        contract_mode: bool = True,
    ) -> FusionResult:
        """综合融合

        Args:
            candidates: 候选策略列表
            periods_per_year: 每年期数(1h K线=8760)
            current_market_pnls: 当前市场PnL(用于regime识别)
            avg_position_usd: 组合平均持仓
            avg_trade_frequency_per_day: 组合日交易频率
            avg_daily_volume_usd: 标的日均成交量
            contract_mode: 是否合约模式

        Returns:
            FusionResult
        """
        result = FusionResult()

        if not candidates:
            result.overall_verdict = Verdict.BLOCKED
            result.reasons.append("无候选策略")
            return result

        # === Step1: 筛选 — 剔除BLOCKED策略 ===
        qualified: List[StrategyCandidate] = []
        for c in candidates:
            if c.evaluation.overall_verdict == Verdict.BLOCKED:
                result.removed_strategies.append(c.name)
                result.removal_reasons[c.name] = (
                    f"评估verdict=BLOCKED (block_reasons: "
                    f"{'; '.join(c.evaluation.block_reasons[:3])})"
                )
                logger.info(f"筛选剔除: {c.name} (BLOCKED)")
            else:
                qualified.append(c)

        if len(qualified) < MIN_STRATEGIES:
            result.overall_verdict = Verdict.BLOCKED
            result.reasons.append(
                f"合格策略数={len(qualified)} < MIN_STRATEGIES={MIN_STRATEGIES}"
            )
            return result

        if len(qualified) > MAX_STRATEGIES:
            # 按综合评分排序, 取前MAX_STRATEGIES个
            qualified.sort(
                key=lambda c: c.evaluation.overall_score, reverse=True
            )
            for c in qualified[MAX_STRATEGIES:]:
                result.removed_strategies.append(c.name)
                result.removal_reasons[c.name] = "超过MAX_STRATEGIES上限, 按综合评分剔除"
            qualified = qualified[:MAX_STRATEGIES]

        # === Step2: 去相关 ===
        corr_result = self.correlation_analyzer.analyze(qualified)
        result.correlation_verdict = corr_result.verdict

        to_remove = self.correlation_analyzer.suggest_removal(qualified, corr_result)
        if to_remove:
            qualified = [c for c in qualified if c.name not in to_remove]
            for name in to_remove:
                result.removed_strategies.append(name)
                result.removal_reasons[name] = "相关性过高(>0.85), 剔除Sharpe较低者"

        if len(qualified) < MIN_STRATEGIES:
            result.overall_verdict = Verdict.BLOCKED
            result.reasons.append(
                f"去相关后合格策略数={len(qualified)} < MIN_STRATEGIES={MIN_STRATEGIES}"
            )
            return result

        result.selected_strategies = [c.name for c in qualified]

        # === Step3: 动态权重分配 ===
        weight_alloc = self.weighter.allocate(
            candidates=qualified,
            current_market_pnls=current_market_pnls,
            periods_per_year=periods_per_year,
        )
        result.weights = weight_alloc.weights
        result.weight_verdict = weight_alloc.verdict
        result.current_regime = weight_alloc.current_regime
        result.regime_evidence = weight_alloc.regime_evidence

        # === Step4: 组合PnL加权求和 ===
        min_len = min(len(c.pnls) for c in qualified)
        fused_pnls: List[float] = []
        for i in range(min_len):
            weighted_pnl = 0.0
            for c in qualified:
                weighted_pnl += weight_alloc.weights[c.name] * c.pnls[i]
            fused_pnls.append(weighted_pnl)
        result.fused_pnls = fused_pnls

        # === Step5: 组合评估 ===
        fusion_eval = self.evaluator.evaluate(
            strategy_name="FusionPortfolio",
            pnls=fused_pnls,
            periods_per_year=periods_per_year,
            avg_position_usd=avg_position_usd,
            avg_trade_frequency_per_day=avg_trade_frequency_per_day,
            avg_daily_volume_usd=avg_daily_volume_usd,
            contract_mode=contract_mode,
        )
        result.fusion_evaluation = fusion_eval

        # === Step6: 综合结论 ===
        reasons: List[str] = []
        if result.correlation_verdict == Verdict.BLOCKED:
            reasons.append("相关性分析BLOCKED(已剔除严重同质化策略)")
        if result.weight_verdict == Verdict.DEGRADED:
            reasons.append("权重分配DEGRADED(分散度不足)")
        if fusion_eval.overall_verdict == Verdict.BLOCKED:
            result.overall_verdict = Verdict.BLOCKED
            reasons.append(f"组合评估BLOCKED (score={fusion_eval.overall_score:.1f})")
        elif fusion_eval.overall_verdict == Verdict.DEGRADED:
            result.overall_verdict = Verdict.DEGRADED
            reasons.append(f"组合评估DEGRADED (score={fusion_eval.overall_score:.1f})")
        else:
            result.overall_verdict = fusion_eval.overall_verdict
            reasons.append(f"组合评估verdict={fusion_eval.overall_verdict.value}")

        result.reasons = reasons
        return result

    def generate_report(self, result: FusionResult) -> str:
        """生成人类可读报告"""
        lines = [
            "=" * 70,
            "策略融合报告 (Strategy Fusion Report)",
            "=" * 70,
            f"综合结论: {result.overall_verdict.value}",
            "",
            "【筛选结果】",
            f"  选中策略({len(result.selected_strategies)}): {', '.join(result.selected_strategies) if result.selected_strategies else '无'}",
            f"  剔除策略({len(result.removed_strategies)}): {', '.join(result.removed_strategies) if result.removed_strategies else '无'}",
        ]

        if result.removal_reasons:
            lines.append("")
            lines.append("【剔除原因】")
            for name, reason in result.removal_reasons.items():
                lines.append(f"  - {name}: {reason}")

        if result.weights:
            lines.append("")
            lines.append("【权重分配】")
            lines.append(f"  当前regime: {result.current_regime.value if result.current_regime else 'N/A'}")
            lines.append(f"  regime依据: {result.regime_evidence}")
            for name, w in sorted(result.weights.items(), key=lambda x: -x[1]):
                lines.append(f"  {name:20s}: {w:.1%}")

        if result.fusion_evaluation:
            fe = result.fusion_evaluation
            lines.append("")
            lines.append("【组合评估】")
            lines.append(f"  综合评分: {fe.overall_score:.1f}/100")
            lines.append(f"  年化收益: {fe.return_performance.annual_return:.2%}")
            lines.append(f"  Sharpe:   {fe.return_performance.sharpe:.3f}")
            lines.append(f"  最大回撤: {fe.risk_profile.max_drawdown:.2%}")
            lines.append(f"  年化波动: {fe.risk_profile.volatility_annual:.2%}")

        if result.reasons:
            lines.append("")
            lines.append("【综合原因】")
            for r in result.reasons:
                lines.append(f"  - {r}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 自测
# ============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=== R17-3 策略融合 自测 ===\n")

    np.random.seed(42)
    evaluator = StrategyEvaluator()
    fusion = StrategyFusion(evaluator)

    # 构造3个不同特性的策略
    # 策略1: 高Sharpe低波动
    pnls1 = list(np.random.normal(0.0002, 0.005, 2000))
    # 策略2: 中等Sharpe高波动(与策略1低相关)
    pnls2 = list(np.random.normal(0.00015, 0.012, 2000))
    # 策略3: 与策略1高度相关(测试去相关)
    pnls3 = list(np.array(pnls1) * 0.9 + np.random.normal(0, 0.001, 2000))
    # 策略4: 负收益(测试筛选剔除)
    pnls4 = list(np.random.normal(-0.0002, 0.015, 2000))

    candidates = []
    for name, pnls in [
        ("HighSharpeLowVol", pnls1),
        ("MedSharpeHighVol", pnls2),
        ("CorrelatedWith1", pnls3),
        ("NegativeReturn", pnls4),
    ]:
        eval_result = evaluator.evaluate(
            strategy_name=name, pnls=pnls, periods_per_year=8760,
            avg_position_usd=10000, avg_trade_frequency_per_day=2,
            contract_mode=True,
        )
        candidates.append(StrategyCandidate(
            name=name, pnls=pnls, evaluation=eval_result,
            avg_position_usd=10000, avg_trade_frequency_per_day=2,
        ))
        print(f"  {name}: verdict={eval_result.overall_verdict.value} score={eval_result.overall_score:.1f}")

    print("\n--- 融合 ---")
    result = fusion.fuse(
        candidates=candidates,
        periods_per_year=8760,
        avg_position_usd=10000,
        avg_trade_frequency_per_day=2,
        contract_mode=True,
    )
    print(fusion.generate_report(result))
