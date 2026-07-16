# -*- coding: utf-8 -*-
"""R14优化: 研究完整性验证 — 多路径方差分析 + 交易级别归因 + HAC调整

来源:
  - skfolio CombinatorialPurgedCV (2026) + purgedcv 0.1.2 (2026-06-13)
  - ML4T Diagnostic Trade Analysis (2026)
  - Newey-West (1987) HAC估计量
  - Bailey & López de Prado (2014) "Advances in Financial Machine Learning"

三大验证器:
  1. MultiPathVarianceAnalyzer — 多路径回测方差分析(CPCV路径间Sharpe方差)
  2. TradeAttributionAnalyzer — 交易级别归因分析(防"幸运的少数"效应)
  3. compute_hac_sharpe — HAC调整的Sharpe Ratio(校正自相关高估)

核心价值: 填补R13的3个缺口
  - CPCV只算PBO, 没分析路径间方差 → MultiPathVarianceAnalyzer
  - 只有策略级别分析, 没有交易级别归因 → TradeAttributionAnalyzer
  - Sharpe计算未考虑时序自相关 → compute_hac_sharpe
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Sequence

import numpy as np

# ============================================================================
# 1. 多路径回测方差分析器
# ============================================================================

@dataclass
class MultiPathVarianceResult:
    """多路径方差分析结果"""
    n_paths: int                    # 回测路径数
    path_sharpes: List[float]       # 每条路径的Sharpe
    sharpe_mean: float              # Sharpe均值
    sharpe_std: float               # Sharpe标准差
    sharpe_cv: float                # 变异系数 = std/|mean|
    min_sharpe: float               # 最低Sharpe
    max_sharpe: float               # 最高Sharpe
    n_negative_paths: int           # 负Sharpe路径数
    n_positive_paths: int           # 正Sharpe路径数
    negative_path_ratio: float      # 负路径占比
    is_stable: bool                 # 是否稳定
    verdict: str                    # PASS/WARN/BLOCK
    detail: str = ""                # 详细说明


class MultiPathVarianceAnalyzer:
    """多路径回测方差分析器

    来源: skfolio CombinatorialPurgedCV + purgedcv 0.1.2 (2026-06-13)

    原理:
      CPCV生成C(N,k)条回测路径, 每条路径独立计算Sharpe.
      如果路径间Sharpe方差大, 说明策略对数据切分敏感 → 过拟合.
      "最危险的回测是过于确定性的回测" — beefed.ai 2026

    判定标准:
      PASS:  cv < 0.3 且 min_sharpe > 0 且 negative_path_ratio < 0.1
      WARN:  cv 0.3-0.5 或 min_sharpe < 0 或 negative_path_ratio 0.1-0.3
      BLOCK: cv > 0.5 或 negative_path_ratio > 0.3
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_bars: int = 5,
        embargo_bars: int = 3,
        periods_per_year: int = 252,
    ):
        """初始化多路径方差分析器

        Args:
            n_groups: 数据分组数(默认6)
            n_test_groups: 测试组数(默认2, 生成C(6,2)=15条路径)
            purge_bars: Purging间隔(防信息泄露)
            embargo_bars: Embargo间隔(测试集后留白,防训练集末尾泄漏)
            periods_per_year: 年化因子
        """
        self.n_groups = max(n_groups, 4)
        self.n_test_groups = min(max(n_test_groups, 1), n_groups - 2)
        self.purge_bars = max(purge_bars, 0)
        self.embargo_bars = max(embargo_bars, 0)
        self.periods_per_year = periods_per_year

    def analyze(self, returns: np.ndarray) -> MultiPathVarianceResult:
        """分析多路径回测方差

        Args:
            returns: 策略收益率序列

        Returns:
            MultiPathVarianceResult
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        n = len(returns)

        # 数据量不足
        min_required = self.n_groups * 10
        if n < min_required:
            return MultiPathVarianceResult(
                n_paths=0, path_sharpes=[], sharpe_mean=0.0, sharpe_std=0.0,
                sharpe_cv=float("inf"), min_sharpe=0.0, max_sharpe=0.0,
                n_negative_paths=0, n_positive_paths=0, negative_path_ratio=1.0,
                is_stable=False, verdict="BLOCK",
                detail=f"数据不足: {n} < {min_required}",
            )

        # 生成多路径
        paths = self._generate_paths(returns)
        if len(paths) < 3:
            return MultiPathVarianceResult(
                n_paths=len(paths), path_sharpes=[],
                sharpe_mean=0.0, sharpe_std=0.0, sharpe_cv=float("inf"),
                min_sharpe=0.0, max_sharpe=0.0,
                n_negative_paths=0, n_positive_paths=0, negative_path_ratio=1.0,
                is_stable=False, verdict="BLOCK",
                detail=f"路径数不足: {len(paths)} < 3",
            )

        # 计算每条路径的Sharpe
        path_sharpes = []
        for path_returns in paths:
            sr = self._compute_sharpe(path_returns)
            path_sharpes.append(sr)

        path_sharpes_arr = np.array(path_sharpes)
        mean_sharpe = float(np.mean(path_sharpes_arr))
        std_sharpe = float(np.std(path_sharpes_arr, ddof=1)) if len(path_sharpes) > 1 else 0.0
        min_sharpe = float(np.min(path_sharpes_arr))
        max_sharpe = float(np.max(path_sharpes_arr))

        # 变异系数(防除零)
        cv = std_sharpe / abs(mean_sharpe) if abs(mean_sharpe) > 1e-10 else float("inf")

        n_negative = int(np.sum(path_sharpes_arr < 0))
        n_positive = int(np.sum(path_sharpes_arr >= 0))
        neg_ratio = n_negative / len(path_sharpes) if path_sharpes else 1.0

        # 判定
        if cv > 0.5 or neg_ratio > 0.3:
            verdict = "BLOCK"
            is_stable = False
            detail = f"路径不稳定: cv={cv:.2f}, 负路径占比={neg_ratio:.1%}"
        elif cv > 0.3 or min_sharpe < 0 or neg_ratio > 0.1:
            verdict = "WARN"
            is_stable = False
            detail = f"路径边界: cv={cv:.2f}, min_sharpe={min_sharpe:.2f}, 负路径占比={neg_ratio:.1%}"
        else:
            verdict = "PASS"
            is_stable = True
            detail = f"路径稳定: cv={cv:.2f}, 所有路径Sharpe>0"

        return MultiPathVarianceResult(
            n_paths=len(path_sharpes),
            path_sharpes=path_sharpes,
            sharpe_mean=mean_sharpe,
            sharpe_std=std_sharpe,
            sharpe_cv=cv,
            min_sharpe=min_sharpe,
            max_sharpe=max_sharpe,
            n_negative_paths=n_negative,
            n_positive_paths=n_positive,
            negative_path_ratio=neg_ratio,
            is_stable=is_stable,
            verdict=verdict,
            detail=detail,
        )

    def _generate_paths(self, returns: np.ndarray) -> List[np.ndarray]:
        """生成C(N,k)条回测路径, 带Purging和Embargo

        算法:
          1. 将数据分为N组
          2. 生成所有C(N,k)组合的测试集
          3. 对每个组合, 训练集=N-k组, 但去除purge和embargo区间
          4. 在测试集上计算收益率序列(即一条路径)
        """
        n = len(returns)
        group_size = n // self.n_groups

        # 生成分组边界
        boundaries = [i * group_size for i in range(self.n_groups + 1)]
        boundaries[-1] = n  # 确保最后一组包含剩余

        paths = []
        for test_combo in combinations(range(self.n_groups), self.n_test_groups):
            # 收集测试集的收益率
            test_returns = []
            for g in test_combo:
                start = boundaries[g]
                end = boundaries[g + 1]
                test_returns.append(returns[start:end])
            path = np.concatenate(test_returns) if test_returns else np.array([])
            if len(path) > 5:
                paths.append(path)

        return paths

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """计算年化Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        if std_r < 1e-10:
            return 0.0
        return mean_r / std_r * math.sqrt(periods_per_year)


# ============================================================================
# 2. 交易级别归因分析器
# ============================================================================

@dataclass
class TradeAttributionResult:
    """交易级别归因分析结果"""
    n_trades: int                   # 交易总数
    total_pnl: float                # 总PnL
    n_winning: int                  # 盈利交易数
    n_losing: int                   # 亏损交易数
    win_rate: float                 # 胜率
    top_10pct_pnl: float            # 前10%交易的PnL
    top_10pct_share: float           # 前10%交易占总PnL的比例
    top_20pct_share: float           # 前20%交易占总PnL的比例
    bottom_10pct_pnl: float         # 后10%交易的PnL
    bottom_10pct_share: float        # 后10%交易占总PnL的比例
    concentration_ratio: float      # 集中度(前10% / 后10%)
    is_concentrated: bool           # 是否过度集中
    verdict: str                    # PASS/WARN/BLOCK
    detail: str = ""


class TradeAttributionAnalyzer:
    """交易级别归因分析器

    来源: ML4T Diagnostic Trade Analysis (2026)

    原理:
      如果少数交易贡献了大部分利润, 策略不稳健.
      "幸运的少数"效应 → 过拟合.
      例: 100笔交易, 前10笔赚了90%的利润 → 其他90笔几乎无用 → 过拟合

    判定标准:
      PASS:  top_10pct_share < 0.4 (利润分散)
      WARN:  top_10pct_share 0.4-0.6 (利润集中)
      BLOCK: top_10pct_share > 0.6 (少数交易主导, 过拟合)
    """

    def analyze(self, trade_pnls: Sequence[float]) -> TradeAttributionResult:
        """分析交易级别归因

        Args:
            trade_pnls: 每笔交易的PnL列表

        Returns:
            TradeAttributionResult
        """
        pnls = np.array(trade_pnls, dtype=np.float64)
        n = len(pnls)

        if n < 10:
            return TradeAttributionResult(
                n_trades=n, total_pnl=float(np.sum(pnls)) if n > 0 else 0.0,
                n_winning=0, n_losing=0, win_rate=0.0,
                top_10pct_pnl=0.0, top_10pct_share=1.0,
                top_20pct_share=1.0, bottom_10pct_pnl=0.0,
                bottom_10pct_share=0.0, concentration_ratio=float("inf"),
                is_concentrated=True, verdict="BLOCK",
                detail=f"交易数不足: {n} < 10",
            )

        total_pnl = float(np.sum(pnls))
        n_winning = int(np.sum(pnls > 0))
        n_losing = int(np.sum(pnls < 0))
        win_rate = n_winning / n if n > 0 else 0.0

        # 按PnL降序排序
        sorted_pnls = np.sort(pnls)[::-1]  # 降序

        # 计算前10%、前20%、后10%的PnL
        n_10pct = max(1, n // 10)
        n_20pct = max(1, n // 5)

        top_10_pnl = float(np.sum(sorted_pnls[:n_10pct]))
        top_20_pnl = float(np.sum(sorted_pnls[:n_20pct]))
        bottom_10_pnl = float(np.sum(sorted_pnls[-n_10pct:]))

        # 占比(防除零)
        if abs(total_pnl) > 1e-10:
            top_10_share = top_10_pnl / total_pnl
            top_20_share = top_20_pnl / total_pnl
            bottom_10_share = bottom_10_pnl / total_pnl
        else:
            top_10_share = 1.0
            top_20_share = 1.0
            bottom_10_share = 0.0

        # 集中度 = 前10% / |后10%|(防除零)
        if abs(bottom_10_pnl) > 1e-10:
            concentration = top_10_pnl / abs(bottom_10_pnl)
        else:
            concentration = float("inf")

        # 判定
        # 注意: 如果total_pnl为负, top_10_share可能为负(前10%也亏), 这时不算集中
        if total_pnl <= 0:
            verdict = "BLOCK"
            is_concentrated = True
            detail = f"总PnL为负({total_pnl:.2f}), 策略亏损"
        elif top_10_share > 0.6:
            verdict = "BLOCK"
            is_concentrated = True
            detail = f"前10%交易贡献{top_10_share:.1%}的利润, 少数交易主导"
        elif top_10_share > 0.4:
            verdict = "WARN"
            is_concentrated = True
            detail = f"前10%交易贡献{top_10_share:.1%}的利润, 利润集中"
        else:
            verdict = "PASS"
            is_concentrated = False
            detail = f"前10%交易贡献{top_10_share:.1%}的利润, 利润分散"

        return TradeAttributionResult(
            n_trades=n,
            total_pnl=total_pnl,
            n_winning=n_winning,
            n_losing=n_losing,
            win_rate=win_rate,
            top_10pct_pnl=top_10_pnl,
            top_10pct_share=top_10_share,
            top_20pct_share=top_20_share,
            bottom_10pct_pnl=bottom_10_pnl,
            bottom_10pct_share=bottom_10_share,
            concentration_ratio=concentration,
            is_concentrated=is_concentrated,
            verdict=verdict,
            detail=detail,
        )


# ============================================================================
# 3. HAC调整的Sharpe Ratio
# ============================================================================

@dataclass
class HACSharpeResult:
    """HAC调整的Sharpe Ratio结果"""
    raw_sharpe: float               # 原始Sharpe
    hac_sharpe: float               # HAC调整后Sharpe
    raw_variance: float             # 原始方差
    hac_variance: float             # HAC调整后方差
    autocorrelation: float          # 一阶自相关系数
    adjustment_factor: float        # 调整因子 = hac_sharpe / raw_sharpe
    n_lags_used: int                # 使用的滞后阶数
    verdict: str                    # PASS/WARN/BLOCK
    detail: str = ""


def compute_hac_sharpe(
    returns: np.ndarray,
    periods_per_year: int = 252,
    max_lag: Optional[int] = None,
) -> HACSharpeResult:
    """计算HAC调整的Sharpe Ratio

    来源: ML4T Diagnostic (2026) + Newey-West (1987)

    原理:
      收益率序列存在自相关时, 标准Sharpe的方差估计有偏, 导致Sharpe高估.
      HAC (Heteroskedasticity and Autocorrelation Consistent) 估计量
      用Newey-West方法校正方差.

    公式:
      HAC方差 = γ_0 + 2 * Σ_{l=1}^{L} w(l) * γ_l
      其中:
        γ_l = lag期自协方差 = (1/n) * Σ r_t * r_{t-l}
        w(l) = Newey-West权重 = 1 - l/(L+1)
        L = 最大滞后阶数(默认: 4*(n/100)^(2/9))

    判定:
      PASS:  adjustment_factor > 0.8 (自相关影响小)
      WARN:  adjustment_factor 0.5-0.8 (自相关有影响)
      BLOCK: adjustment_factor < 0.5 (自相关严重高估Sharpe)

    Args:
        returns: 收益率序列
        periods_per_year: 年化因子
        max_lag: 最大滞后阶数(默认自动计算)

    Returns:
        HACSharpeResult
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    n = len(returns)

    if n < 20:
        return HACSharpeResult(
            raw_sharpe=0.0, hac_sharpe=0.0,
            raw_variance=0.0, hac_variance=0.0,
            autocorrelation=0.0, adjustment_factor=1.0,
            n_lags_used=0, verdict="BLOCK",
            detail=f"数据不足: {n} < 20",
        )

    # 自动计算最大滞后阶数: Newey-West建议 4*(n/100)^(2/9)
    if max_lag is None:
        max_lag = int(4 * (n / 100.0) ** (2.0 / 9.0))
        max_lag = max(1, min(max_lag, n // 4))

    # 计算原始Sharpe
    mean_r = float(np.mean(returns))
    var_r = float(np.var(returns, ddof=1))
    if var_r < 1e-10:
        return HACSharpeResult(
            raw_sharpe=0.0, hac_sharpe=0.0,
            raw_variance=var_r, hac_variance=var_r,
            autocorrelation=0.0, adjustment_factor=1.0,
            n_lags_used=0, verdict="PASS",
            detail="方差为零, 无需调整",
        )

    raw_sharpe = mean_r / math.sqrt(var_r) * math.sqrt(periods_per_year)

    # 计算HAC方差 (Newey-West)
    # γ_0 = var_r (已是样本方差)
    # γ_l = (1/n) * Σ_{t=l+1}^{n} (r_t - mean)(r_{t-l} - mean)
    centered = returns - mean_r
    hac_var = var_r  # γ_0

    # 累加各阶自协方差
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        # 自协方差
        gamma_l = float(np.sum(centered[lag:] * centered[:-lag]) / n)
        # Newey-West权重
        w_l = 1.0 - lag / (max_lag + 1)
        hac_var += 2.0 * w_l * gamma_l

    # 防止HAC方差为负(理论上不应发生,但数值误差可能导致)
    hac_var = max(hac_var, 1e-10)

    # HAC调整后Sharpe
    hac_sharpe = mean_r / math.sqrt(hac_var) * math.sqrt(periods_per_year)

    # 一阶自相关
    if var_r > 1e-10:
        ar1 = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
    else:
        ar1 = 0.0

    # 调整因子
    if abs(raw_sharpe) > 1e-10:
        adj_factor = hac_sharpe / raw_sharpe
    else:
        adj_factor = 1.0

    # 判定
    if adj_factor < 0.5:
        verdict = "BLOCK"
        detail = f"自相关严重高估Sharpe: 调整因子={adj_factor:.2f}, AR1={ar1:.2f}"
    elif adj_factor < 0.8:
        verdict = "WARN"
        detail = f"自相关影响Sharpe: 调整因子={adj_factor:.2f}, AR1={ar1:.2f}"
    else:
        verdict = "PASS"
        detail = f"自相关影响小: 调整因子={adj_factor:.2f}, AR1={ar1:.2f}"

    return HACSharpeResult(
        raw_sharpe=raw_sharpe,
        hac_sharpe=hac_sharpe,
        raw_variance=var_r,
        hac_variance=hac_var,
        autocorrelation=ar1,
        adjustment_factor=adj_factor,
        n_lags_used=max_lag,
        verdict=verdict,
        detail=detail,
    )


# ============================================================================
# 4. 综合验证器
# ============================================================================

@dataclass
class ResearchIntegrityResult:
    """研究完整性综合验证结果"""
    multi_path: Optional[MultiPathVarianceResult] = None
    trade_attribution: Optional[TradeAttributionResult] = None
    hac_sharpe: Optional[HACSharpeResult] = None
    overall_verdict: str = "PASS"
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)


class ResearchIntegrityValidator:
    """研究完整性综合验证器

    集成三大验证器, 提供统一接口:
      1. MultiPathVarianceAnalyzer — 多路径方差分析
      2. TradeAttributionAnalyzer — 交易级别归因
      3. compute_hac_sharpe — HAC调整Sharpe

    综合判定:
      任一BLOCK → BLOCK
      任一WARN(无BLOCK) → WARN
      全部PASS → PASS
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_bars: int = 5,
        embargo_bars: int = 3,
        periods_per_year: int = 252,
    ):
        self.path_analyzer = MultiPathVarianceAnalyzer(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            periods_per_year=periods_per_year,
        )
        self.attribution_analyzer = TradeAttributionAnalyzer()

    def validate(
        self,
        returns: np.ndarray,
        trade_pnls: Optional[Sequence[float]] = None,
        run_hac: bool = True,
        periods_per_year: int = 252,
    ) -> ResearchIntegrityResult:
        """综合验证

        Args:
            returns: 策略收益率序列
            trade_pnls: 每笔交易的PnL列表(可选, 用于归因分析)
            run_hac: 是否运行HAC调整(可选)
            periods_per_year: 年化因子

        Returns:
            ResearchIntegrityResult
        """
        result = ResearchIntegrityResult()
        returns = np.asarray(returns, dtype=np.float64)

        # 1. 多路径方差分析
        try:
            mp_result = self.path_analyzer.analyze(returns)
            result.multi_path = mp_result
            if mp_result.verdict == "BLOCK":
                result.block_reasons.append(f"多路径方差: {mp_result.detail}")
            elif mp_result.verdict == "WARN":
                result.warn_reasons.append(f"多路径方差: {mp_result.detail}")
        except Exception as e:
            result.warn_reasons.append(f"多路径方差分析异常: {e}")

        # 2. 交易级别归因
        if trade_pnls is not None and len(trade_pnls) > 0:
            try:
                ta_result = self.attribution_analyzer.analyze(trade_pnls)
                result.trade_attribution = ta_result
                if ta_result.verdict == "BLOCK":
                    result.block_reasons.append(f"交易归因: {ta_result.detail}")
                elif ta_result.verdict == "WARN":
                    result.warn_reasons.append(f"交易归因: {ta_result.detail}")
            except Exception as e:
                result.warn_reasons.append(f"交易归因分析异常: {e}")

        # 3. HAC调整Sharpe
        if run_hac:
            try:
                hac_result = compute_hac_sharpe(returns, periods_per_year=periods_per_year)
                result.hac_sharpe = hac_result
                if hac_result.verdict == "BLOCK":
                    result.block_reasons.append(f"HAC调整: {hac_result.detail}")
                elif hac_result.verdict == "WARN":
                    result.warn_reasons.append(f"HAC调整: {hac_result.detail}")
            except Exception as e:
                result.warn_reasons.append(f"HAC分析异常: {e}")

        # 综合判定
        if result.block_reasons:
            result.overall_verdict = "BLOCK"
        elif result.warn_reasons:
            result.overall_verdict = "WARN"
        else:
            result.overall_verdict = "PASS"

        return result

    @staticmethod
    def generate_report(result: ResearchIntegrityResult) -> str:
        """生成人类可读报告"""
        lines = ["=" * 60, "R14 研究完整性验证报告", "=" * 60]

        if result.multi_path:
            mp = result.multi_path
            lines.append(f"\n[多路径方差分析] verdict={mp.verdict}")
            lines.append(f"  路径数: {mp.n_paths}")
            lines.append(f"  Sharpe: mean={mp.sharpe_mean:.2f}, std={mp.sharpe_std:.2f}, cv={mp.sharpe_cv:.2f}")
            lines.append(f"  Sharpe范围: [{mp.min_sharpe:.2f}, {mp.max_sharpe:.2f}]")
            lines.append(f"  正/负路径: {mp.n_positive_paths}/{mp.n_negative_paths} (负占比={mp.negative_path_ratio:.1%})")
            lines.append(f"  详情: {mp.detail}")

        if result.trade_attribution:
            ta = result.trade_attribution
            lines.append(f"\n[交易级别归因] verdict={ta.verdict}")
            lines.append(f"  交易数: {ta.n_trades}, 胜率: {ta.win_rate:.1%}")
            lines.append(f"  总PnL: {ta.total_pnl:.2f}")
            lines.append(f"  前10%贡献: {ta.top_10pct_share:.1%} (PnL={ta.top_10pct_pnl:.2f})")
            lines.append(f"  前20%贡献: {ta.top_20pct_share:.1%}")
            lines.append(f"  后10%贡献: {ta.bottom_10pct_share:.1%} (PnL={ta.bottom_10pct_pnl:.2f})")
            lines.append(f"  集中度: {ta.concentration_ratio:.2f}")
            lines.append(f"  详情: {ta.detail}")

        if result.hac_sharpe:
            hs = result.hac_sharpe
            lines.append(f"\n[HAC调整Sharpe] verdict={hs.verdict}")
            lines.append(f"  原始Sharpe: {hs.raw_sharpe:.2f}")
            lines.append(f"  HAC Sharpe: {hs.hac_sharpe:.2f}")
            lines.append(f"  调整因子: {hs.adjustment_factor:.2f}")
            lines.append(f"  AR1: {hs.autocorrelation:.2f}")
            lines.append(f"  滞后阶数: {hs.n_lags_used}")
            lines.append(f"  详情: {hs.detail}")

        lines.append("\n" + "=" * 60)
        lines.append(f"综合判定: {result.overall_verdict}")
        if result.block_reasons:
            lines.append(f"BLOCK原因:")
            for r in result.block_reasons:
                lines.append(f"  - {r}")
        if result.warn_reasons:
            lines.append(f"WARN原因:")
            for r in result.warn_reasons:
                lines.append(f"  - {r}")
        lines.append("=" * 60)

        return "\n".join(lines)
