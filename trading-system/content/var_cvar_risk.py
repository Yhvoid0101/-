# -*- coding: utf-8 -*-
"""
VaR/CVaR 风险模块 (Value at Risk / Conditional Value at Risk)

Phase 7I-3: 修复空壳交付#3 — frontier_enhancement.DynamicAdaptiveKelly.var_limit
            之前只是静态2%阈值, 现在替换为真正的VaR/CVaR计算

来源:
  - J.P. Morgan RiskMetrics (1996) — VaR行业标准化
  - Rockafellar & Uryasev (2000) — CVaR优化理论
  - Danielsson (2011) "Financial Risk Forecasting" — Student-t厚尾分布
  - Boudt et al. (2023) "Cryptocurrency Risk Management" — 加密货币VaR最佳实践

数学公式:
  历史模拟法:
    VaR_α = -percentile(returns, (1-α)*100)   # 正数表示损失
    CVaR_α = -mean(returns[returns <= percentile])

  参数法(正态):
    VaR_α = -(μ + σ × z_α)   其中 z_α = Φ⁻¹(1-α) (1-α分位数)
    CVaR_α = -(μ + σ × φ(z_α) / α)  其中 φ是标准正态PDF

  参数法(Student-t, 适合加密货币厚尾):
    σ_scaled = σ × sqrt((df-2)/df)   # t分布缩放因子
    VaR_α = -(μ + σ_scaled × t_α(df))   其中 t_α(df)是t分布分位数
    CVaR_α = -(μ + σ_scaled × (f(t_α) × (df + t_α²)) / (α × (df-1)))
      其中 f(t)是t分布PDF

  Monte Carlo:
    生成N个Student-t随机样本 → 缩放 → 计算百分位数
    CVaR = -mean(samples[samples <= percentile])

加密货币参数建议 (来自Boudt 2023):
  - 自由度 df = 4 (3-5之间, 4是BTC/ETH最佳拟合)
  - 置信度 α = 0.95 (单日95% VaR)
  - 单日VaR上限 = 2% (单笔最大风险)
  - 最大仓位 = 25% (账户权益)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("hermes.var_cvar_risk")

# ============================================================================
# scipy 可选导入 (有则用精确分位数, 无则用近似算法)
# ============================================================================
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    _scipy_stats = None  # type: ignore


# ============================================================================
# 数据类定义
# ============================================================================


@dataclass(slots=True)
class RiskMetrics:
    """VaR/CVaR 风险度量结果

    所有数值含义:
      var: 正数表示损失 (例如 0.025 = 2.5%的损失)
      cvar: 正数表示损失, 且 cvar >= var (因为CVaR是尾部期望)
      mean_return: 收益率均值 (负数=平均亏损)
      std_return: 收益率标准差
      skewness: 偏度 (负=左偏, 加密货币通常为负)
      kurtosis: 峰度 (肥尾分布>3)
    """
    method: str = ""
    confidence: float = 0.95
    var: float = 0.0       # Value at Risk (正数=损失)
    cvar: float = 0.0      # Conditional VaR (正数=损失, >= var)
    mean_return: float = 0.0
    std_return: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    sample_size: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionRecommendation:
    """基于VaR的仓位推荐"""
    recommended_position_pct: float = 0.0   # 推荐仓位占比
    max_position_pct: float = 0.0          # VaR约束下的最大仓位
    var_limit_pct: float = 0.0             # VaR限制 (例如0.02=2%)
    actual_var_pct: float = 0.0            # 在推荐仓位下的实际VaR
    risk_adjustment: float = 1.0           # 风险调整因子 (1.0=无调整, <1.0=降低)
    method: str = ""                       # 使用的VaR方法
    reason: str = ""                       # 决策原因


# ============================================================================
# VaR/CVaR 计算核心模块
# ============================================================================


class VarCvarRiskModule:
    """VaR/CVaR 风险计算模块

    实现4种VaR计算方法 + CVaR + 仓位推荐:

      1. 历史模拟法 (historical): 直接从历史分位数计算, 无分布假设
      2. 参数法-正态 (parametric_normal): 假设收益服从正态分布
      3. 参数法-Student-t (parametric_t): 适合加密货币厚尾分布
      4. Monte Carlo (monte_carlo): 模拟大量样本计算尾部风险

    推荐方法: calculate_all() 返回所有方法+max(反温室原则)
    仓位推荐: recommend_position_size() 基于VaR限制调整仓位

    使用方式:
        module = VarCvarRiskModule()
        returns = [0.01, -0.02, 0.005, ...]  # 历史收益率序列
        all_metrics = module.calculate_all(returns, confidence=0.95)
        rec = module.recommend_position_size(
            account_equity=10000, returns=returns, max_var_pct=0.02
        )
    """

    # 加密货币默认参数 (来自Boudt 2023)
    CRYPTO_DEFAULT_DF = 4.0           # Student-t自由度 (3-5适合加密货币, 4为BTC/ETH最佳)
    CRYPTO_VAR_LIMIT = 0.02           # 单日VaR限制2% (账户权益)
    CRYPTO_MAX_POSITION = 0.25        # 最大仓位25%
    CRYPTO_MIN_POSITION = 0.01        # 最小仓位1%
    MONTE_CARLO_SIMULATIONS = 10000   # MC模拟次数
    MIN_SAMPLE_SIZE = 30              # 最小样本量 (少于则警告)

    def __init__(
        self,
        default_df: float = CRYPTO_DEFAULT_DF,
        monte_carlo_simulations: int = MONTE_CARLO_SIMULATIONS,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            default_df: Student-t默认自由度 (4适合加密货币)
            monte_carlo_simulations: Monte Carlo模拟次数
            random_seed: 随机种子 (None=不固定, 用于可复现性测试)
        """
        self.default_df = default_df
        self.monte_carlo_simulations = monte_carlo_simulations
        self._random = random.Random(random_seed) if random_seed is not None else random.Random()
        if random_seed is not None:
            np.random.seed(random_seed)

    # ----------------------------------------------------------------------
    # 1. 历史模拟法
    # ----------------------------------------------------------------------

    def calculate_historical(
        self,
        returns: Sequence[float],
        confidence: float = 0.95,
    ) -> RiskMetrics:
        """历史模拟法 VaR/CVaR

        原理: 直接从历史收益率的分位数计算, 不假设分布
        优势: 无分布假设, 保留真实尾部特征
        劣势: 需要足够样本, 历史不能完全代表未来

        Args:
            returns: 历史收益率序列 (例如 [0.01, -0.02, ...])
            confidence: 置信度 (0.95=95% VaR)

        Returns:
            RiskMetrics 包含 var/cvar/统计量
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = len(arr)

        if n == 0:
            return RiskMetrics(
                method="historical", confidence=confidence,
                extra={"reason": "empty_returns"},
            )
        if n < self.MIN_SAMPLE_SIZE:
            logger.warning("历史VaR样本量不足: %d (建议>=%d)", n, self.MIN_SAMPLE_SIZE)

        alpha = 1.0 - confidence
        # VaR: 取 (1-α) 分位数, 负号转正数表示损失
        percentile_value = float(np.percentile(arr, alpha * 100))
        var = -percentile_value

        # CVaR: 取所有 <= 分位数的尾部, 计算均值
        tail = arr[arr <= percentile_value]
        if len(tail) > 0:
            cvar = -float(np.mean(tail))
        else:
            cvar = var  # 退化情况

        # 保证 CVaR >= VaR (CVaR是尾部期望, 应该 >= VaR)
        if cvar < var:
            cvar = var

        stats = self._compute_statistics(arr)
        return RiskMetrics(
            method="historical",
            confidence=confidence,
            var=var,
            cvar=cvar,
            mean_return=stats[0],
            std_return=stats[1],
            skewness=stats[2],
            kurtosis=stats[3],
            sample_size=n,
            extra={"tail_count": len(tail)},
        )

    # ----------------------------------------------------------------------
    # 2. 参数法 - 正态分布
    # ----------------------------------------------------------------------

    def calculate_parametric_normal(
        self,
        returns: Sequence[float],
        confidence: float = 0.95,
    ) -> RiskMetrics:
        """参数法(正态分布) VaR/CVaR

        公式:
          VaR_α = -(μ + σ × z_α)
          CVaR_α = -(μ - σ × φ(z_α) / α) = -μ + σ × φ(z_α) / α

        其中:
          z_α = Φ⁻¹(α)  (标准正态的下α分位数, 例如α=0.05时z_α=-1.6449)
          φ(z) = (1/√(2π)) × exp(-z²/2)  (标准正态PDF, 偶函数 φ(-z)=φ(z))

        优势: 计算快, 公式简洁
        劣势: 正态分布低估尾部风险 (黑天鹅事件), 不适合加密货币

        Args:
            returns: 历史收益率序列 (用于估计μ和σ)
            confidence: 置信度

        Returns:
            RiskMetrics
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = len(arr)

        if n == 0:
            return RiskMetrics(method="parametric_normal", confidence=confidence)

        alpha = 1.0 - confidence
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        if sigma <= 0:
            # 退化情况: 无波动
            return RiskMetrics(
                method="parametric_normal", confidence=confidence,
                var=0.0, cvar=0.0, mean_return=mu, std_return=sigma,
                sample_size=n, extra={"degenerate": True},
            )

        # 标准正态下分位数 z_α = Φ⁻¹(α) (例如 α=0.05 → z_α=-1.6449)
        z_alpha = self._norm_ppf(alpha)

        # 标准正态PDF: φ(z) = (1/√(2π)) × exp(-z²/2)  (偶函数, φ(-z)=φ(z))
        phi_z = math.exp(-0.5 * z_alpha * z_alpha) / math.sqrt(2.0 * math.pi)

        # VaR = -(μ + σ × z_α)  (z_α为负, 所以结果为正: -μ + σ×|z_α|)
        var = -(mu + sigma * z_alpha)
        # CVaR = -(μ - σ × φ(z_α) / α) = -μ + σ × φ(z_α) / α
        cvar = -(mu - sigma * phi_z / alpha)

        # 保证正值和CVaR >= VaR
        var = max(var, 0.0)
        cvar = max(cvar, var)

        stats = self._compute_statistics(arr)
        return RiskMetrics(
            method="parametric_normal",
            confidence=confidence,
            var=var,
            cvar=cvar,
            mean_return=mu,
            std_return=sigma,
            skewness=stats[2],
            kurtosis=stats[3],
            sample_size=n,
            extra={"z_alpha": z_alpha, "phi_z": phi_z},
        )

    # ----------------------------------------------------------------------
    # 3. 参数法 - Student-t分布 (推荐用于加密货币)
    # ----------------------------------------------------------------------

    def calculate_parametric_t(
        self,
        returns: Sequence[float],
        confidence: float = 0.95,
        df: Optional[float] = None,
    ) -> RiskMetrics:
        """参数法(Student-t分布) VaR/CVaR

        Student-t分布比正态分布有更肥的尾部, 更适合加密货币收益率。

        公式:
          σ_scaled = σ × sqrt((df-2)/df)   # t分布方差= df/(df-2), 需缩放
          t_α = T_df⁻¹(α)                  # t分布的下α分位数 (例如α=0.05, df=4 → t_α≈-2.132)
          VaR_α = -(μ + σ_scaled × t_α)    # = -μ + σ_scaled × |t_α|
          CVaR_α = -μ + σ_scaled × f(t_α) × (df + t_α²) / (α × (df-1))
                 = -(μ - σ_scaled × f(t_α) × (df + t_α²) / (α × (df-1)))

        其中 f(t) 是t分布PDF (偶函数):
          f(t) = Γ((df+1)/2) / (√(df×π) × Γ(df/2)) × (1 + t²/df)^(-(df+1)/2)

        Args:
            returns: 历史收益率序列
            confidence: 置信度
            df: 自由度 (None=使用self.default_df, 加密货币默认4)

        Returns:
            RiskMetrics
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = len(arr)

        if n == 0:
            return RiskMetrics(method="parametric_t", confidence=confidence)

        if df is None:
            df = self.default_df
        if df <= 2.0:
            logger.warning("Student-t df=%.2f 太小 (需>2), 使用默认4.0", df)
            df = self.default_df

        # Phase 7J-8 反温室修复 HIGH #2: 样本量不足标记 degraded
        # 之前: n < MIN_SAMPLE_SIZE 仅 warning, 继续计算 → 沙盘少量样本VaR不稳定, 实盘样本充足VaR不同
        # 现在: 标记 degraded=True, recommend_position_size 检测后收紧仓位 (反温室)
        is_degraded_sample = n < self.MIN_SAMPLE_SIZE
        if is_degraded_sample:
            logger.warning("Student-t VaR样本量不足: %d (建议>=%d), 标记degraded", n, self.MIN_SAMPLE_SIZE)

        alpha = 1.0 - confidence
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        if sigma <= 0:
            return RiskMetrics(
                method="parametric_t", confidence=confidence,
                var=0.0, cvar=0.0, mean_return=mu, std_return=sigma,
                sample_size=n, extra={"degenerate": True, "df": df},
            )

        # t分布缩放: σ_scaled = σ × sqrt((df-2)/df)
        sigma_scaled = sigma * math.sqrt((df - 2.0) / df)

        # t分布下分位数 t_α = T_df⁻¹(α) (例如α=0.05, df=4 → t_α≈-2.132)
        t_alpha = self._t_ppf(alpha, df)

        # t分布PDF f(t_α) (偶函数, f(-t)=f(t))
        f_t = self._t_pdf(t_alpha, df)

        # VaR = -(μ + σ_scaled × t_α) = -μ + σ_scaled × |t_α|
        var = -(mu + sigma_scaled * t_alpha)
        # CVaR = -(μ - σ_scaled × f(t_α) × (df + t_α²) / (α × (df-1)))
        cvar = -(mu - sigma_scaled * (f_t * (df + t_alpha * t_alpha)) / (alpha * (df - 1.0)))

        var = max(var, 0.0)
        cvar = max(cvar, var)

        stats = self._compute_statistics(arr)
        extra_dict = {
            "df": df,
            "sigma_scaled": sigma_scaled,
            "t_alpha": t_alpha,
            "f_t_alpha": f_t,
        }
        if is_degraded_sample:
            extra_dict["degraded"] = True
            extra_dict["degraded_reason"] = f"sample_size={n}<{self.MIN_SAMPLE_SIZE}"
        return RiskMetrics(
            method="parametric_t",
            confidence=confidence,
            var=var,
            cvar=cvar,
            mean_return=mu,
            std_return=sigma,
            skewness=stats[2],
            kurtosis=stats[3],
            sample_size=n,
            extra=extra_dict,
        )

    # ----------------------------------------------------------------------
    # 4. Monte Carlo 模拟
    # ----------------------------------------------------------------------

    def calculate_monte_carlo(
        self,
        returns: Sequence[float],
        confidence: float = 0.95,
        df: Optional[float] = None,
        n_simulations: Optional[int] = None,
    ) -> RiskMetrics:
        """Monte Carlo 模拟 VaR/CVaR

        原理:
          1. 从历史数据估计μ, σ, df
          2. 生成N个Student-t随机样本 (缩放到σ)
          3. 计算分位数得到VaR/CVaR

        优势: 可处理复杂分布, 灵活
        劣势: 计算较慢, 结果有随机性

        Args:
            returns: 历史收益率序列
            confidence: 置信度
            df: 自由度 (None=使用default_df)
            n_simulations: 模拟次数 (None=使用self.monte_carlo_simulations)

        Returns:
            RiskMetrics
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = len(arr)

        if n == 0:
            return RiskMetrics(method="monte_carlo", confidence=confidence)

        if df is None:
            df = self.default_df
        if df <= 2.0:
            df = self.default_df
        if n_simulations is None:
            n_simulations = self.monte_carlo_simulations

        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        if sigma <= 0:
            return RiskMetrics(
                method="monte_carlo", confidence=confidence,
                var=0.0, cvar=0.0, mean_return=mu, std_return=sigma,
                sample_size=n, extra={"degenerate": True, "df": df},
            )

        alpha = 1.0 - confidence

        # 生成Student-t随机样本并缩放
        # scipy.stats.t.rvs 更精确, 但scipy不可用时使用numpy
        if _SCIPY_AVAILABLE:
            samples = _scipy_stats.t.rvs(df, size=n_simulations, random_state=self._random.randint(0, 2**31 - 1))
        else:
            # numpy的standard_t
            try:
                samples = np.random.standard_t(df, size=n_simulations)
            except (AttributeError, TypeError):
                # 回退: 使用正态/Gamma混合生成Student-t
                samples = self._generate_student_t_fallback(df, n_simulations)

        # 缩放: σ_scaled = σ × sqrt((df-2)/df)
        sigma_scaled = sigma * math.sqrt((df - 2.0) / df)
        samples = mu + sigma_scaled * np.asarray(samples, dtype=float)

        # VaR/CVaR (类似历史法, 但样本来自模拟)
        percentile_value = float(np.percentile(samples, alpha * 100))
        var = -percentile_value
        tail = samples[samples <= percentile_value]
        if len(tail) > 0:
            cvar = -float(np.mean(tail))
        else:
            cvar = var
        if cvar < var:
            cvar = var

        stats = self._compute_statistics(arr)
        return RiskMetrics(
            method="monte_carlo",
            confidence=confidence,
            var=var,
            cvar=cvar,
            mean_return=mu,
            std_return=sigma,
            skewness=stats[2],
            kurtosis=stats[3],
            sample_size=n,
            extra={
                "df": df,
                "n_simulations": n_simulations,
                "sigma_scaled": sigma_scaled,
            },
        )

    # ----------------------------------------------------------------------
    # 5. 综合计算 + 推荐 (反温室原则: 取最大VaR)
    # ----------------------------------------------------------------------

    def calculate_all(
        self,
        returns: Sequence[float],
        confidence: float = 0.95,
        df: Optional[float] = None,
    ) -> Dict[str, RiskMetrics]:
        """综合计算所有方法的VaR/CVaR

        反温室原则 (anti-greenhouse): 取所有方法中的最大VaR作为推荐
        原因: 风险评估宁可高估不可低估

        Args:
            returns: 历史收益率序列
            confidence: 置信度
            df: Student-t自由度

        Returns:
            dict 包含:
              - "historical": 历史模拟法
              - "parametric_normal": 参数法-正态
              - "parametric_t": 参数法-Student-t
              - "monte_carlo": Monte Carlo
              - "recommended": 推荐方法 (最大VaR)
        """
        results: Dict[str, RiskMetrics] = {
            "historical": self.calculate_historical(returns, confidence),
            "parametric_normal": self.calculate_parametric_normal(returns, confidence),
            "parametric_t": self.calculate_parametric_t(returns, confidence, df),
            "monte_carlo": self.calculate_monte_carlo(returns, confidence, df),
        }

        # 反温室原则: 取最大VaR作为推荐
        valid_methods = {k: v for k, v in results.items() if v.sample_size > 0}
        if valid_methods:
            recommended_key = max(valid_methods, key=lambda k: valid_methods[k].var)
            results["recommended"] = valid_methods[recommended_key]
            results["recommended"].extra = results["recommended"].extra or {}
            results["recommended"].extra["recommended_basis"] = f"max_var_from_{recommended_key}"
        else:
            # 全部空时返回空metrics
            results["recommended"] = RiskMetrics(
                method="none", confidence=confidence,
                extra={"reason": "all_methods_empty"},
            )

        return results

    # ----------------------------------------------------------------------
    # 6. 仓位推荐
    # ----------------------------------------------------------------------

    def recommend_position_size(
        self,
        account_equity: float,
        returns: Sequence[float],
        max_var_pct: float = CRYPTO_VAR_LIMIT,
        max_position_pct: float = CRYPTO_MAX_POSITION,
        min_position_pct: float = CRYPTO_MIN_POSITION,
        confidence: float = 0.95,
        df: Optional[float] = None,
        kelly_suggested_pct: Optional[float] = None,
    ) -> PositionRecommendation:
        """基于VaR的仓位推荐

        核心逻辑:
          1. 计算推荐方法(Student-t)的VaR
          2. 如果VaR > max_var_pct: 降低仓位使实际VaR <= max_var_pct
          3. 仓位 = min(kelly_suggested, var_adjusted_position, max_position)

        Args:
            account_equity: 账户权益
            returns: 历史收益率序列
            max_var_pct: 单笔最大VaR占比 (0.02=2%)
            max_position_pct: 最大仓位占比 (0.25=25%)
            min_position_pct: 最小仓位占比 (0.01=1%)
            confidence: VaR置信度
            df: Student-t自由度
            kelly_suggested_pct: Kelly公式建议的仓位占比 (None=不使用Kelly约束)

        Returns:
            PositionRecommendation
        """
        # 计算Student-t VaR (推荐方法, 适合加密货币)
        metrics = self.calculate_parametric_t(returns, confidence, df)
        var_pct = metrics.var  # 正数, 表示单笔最大损失占比

        if var_pct <= 0:
            # Phase 7J-8 反温室修复 HIGH #1: VaR=0 = 数据异常, 不是"无风险"
            # 之前: VaR=0 使用最大仓位 → 沙盘用全0收益率进化出"无风险"策略, 实盘爆仓
            # 现在: VaR=0 视为风险不可量化, 使用最小仓位 (反温室: 宁可少赚不可亏钱)
            # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — VaR=0 用最大仓位=自杀
            recommended = min_position_pct
            if kelly_suggested_pct is not None:
                recommended = min(recommended, kelly_suggested_pct)
            return PositionRecommendation(
                recommended_position_pct=recommended,
                max_position_pct=max_position_pct,
                var_limit_pct=max_var_pct,
                actual_var_pct=0.0,
                risk_adjustment=recommended / max_position_pct if max_position_pct > 0 else 0.0,
                method="parametric_t",
                reason="VaR=0 (数据异常/风险不可量化), 使用最小仓位 (反温室修复)",
            )

        # VaR约束: 实际VaR = 仓位 × 单位VaR
        # 求解: 仓位 × var_pct <= max_var_pct  →  仓位 <= max_var_pct / var_pct
        var_adjusted_position = max_var_pct / var_pct
        var_adjusted_position = min(var_adjusted_position, max_position_pct)

        # Phase 7J-8 反温室修复 HIGH #2: degraded 模式收紧仓位 (×0.5)
        # 原因: 样本量不足时 VaR 不稳定, 沙盘少量样本进化出的策略在实盘样本充足时行为不同
        # 修复: degraded 时仓位上限减半 (反温室: 风险不可量化时降低暴露)
        # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 样本不足+大仓位=沙盘过拟合
        is_degraded = bool(metrics.extra.get("degraded", False))
        if is_degraded:
            var_adjusted_position *= 0.5
            degraded_note = f" [degraded: {metrics.extra.get('degraded_reason', 'unknown')}, 仓位×0.5]"
        else:
            degraded_note = ""

        # 风险调整因子
        risk_adjustment = var_adjusted_position / max_position_pct if max_position_pct > 0 else 1.0

        # 综合Kelly约束
        if kelly_suggested_pct is not None:
            recommended = min(kelly_suggested_pct, var_adjusted_position)
            reason = f"Kelly={kelly_suggested_pct:.4f} + VaR约束={var_adjusted_position:.4f} → {recommended:.4f}{degraded_note}"
        else:
            recommended = var_adjusted_position
            reason = f"纯VaR约束: {max_var_pct:.2%}/{var_pct:.4f} = {recommended:.4f}{degraded_note}"

        # 最小仓位保护
        if recommended < min_position_pct:
            recommended = 0.0  # 低于最小仓位则不交易
            reason += f"; 低于最小仓位{min_position_pct:.2%}, 不交易"

        # 实际VaR (在推荐仓位下)
        actual_var = recommended * var_pct

        return PositionRecommendation(
            recommended_position_pct=recommended,
            max_position_pct=max_position_pct,
            var_limit_pct=max_var_pct,
            actual_var_pct=actual_var,
            risk_adjustment=risk_adjustment,
            method="parametric_t",
            reason=reason,
        )

    # ----------------------------------------------------------------------
    # 私有工具方法
    # ----------------------------------------------------------------------

    def _compute_statistics(self, arr: np.ndarray) -> Tuple[float, float, float, float]:
        """计算收益率统计量: (mean, std, skewness, kurtosis)"""
        n = len(arr)
        if n == 0:
            return (0.0, 0.0, 0.0, 0.0)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        if sigma <= 0:
            return (mu, sigma, 0.0, 0.0)
        # 偏度 = E[((X-μ)/σ)³]
        skew = float(np.mean(((arr - mu) / sigma) ** 3)) if n > 2 else 0.0
        # 峰度 = E[((X-μ)/σ)⁴] - 3 (excess kurtosis, 正态=0)
        kurt = float(np.mean(((arr - mu) / sigma) ** 4) - 3.0) if n > 3 else 0.0
        return (mu, sigma, skew, kurt)

    def _norm_ppf(self, p: float) -> float:
        """标准正态分布分位数 Φ⁻¹(p)

        优先使用scipy, 不可用时使用Peter Acklam算法 (精度<1.15e-9)
        """
        if _SCIPY_AVAILABLE:
            return float(_scipy_stats.norm.ppf(p))
        return self._approx_norm_ppf(p)

    def _approx_norm_ppf(self, p: float) -> float:
        """Peter Acklam算法 — 标准正态分位数近似 (无scipy时使用)

        来源: Peter J. Acklam (2003) "An algorithm for computing the inverse normal CDF"
        精度: 相对误差 < 1.15e-9

        Args:
            p: 概率值 (0 < p < 1)

        Returns:
            z: 标准正态分位数
        """
        # 边界处理
        if p <= 0.0:
            return -10.0  # 极端负值
        if p >= 1.0:
            return 10.0   # 极端正值

        # Acklam算法常数
        a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]

        p_low = 0.02425
        p_high = 1.0 - p_low

        if p < p_low:
            # 下尾区域
            q = math.sqrt(-2.0 * math.log(p))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        elif p <= p_high:
            # 中心区域
            q = p - 0.5
            r = q * q
            return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        else:
            # 上尾区域
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    def _t_ppf(self, p: float, df: float) -> float:
        """Student-t分布分位数

        优先使用scipy, 不可用时使用Cornish-Fisher展开近似
        """
        if _SCIPY_AVAILABLE:
            return float(_scipy_stats.t.ppf(p, df))
        return self._approx_t_ppf(p, df)

    def _approx_t_ppf(self, p: float, df: float) -> float:
        """Student-t分位数近似 (无scipy时使用)

        使用Cornish-Fisher展开 (基于正态分位数的修正):
          t_α ≈ z_α + (z_α³ + z_α)/(4×df) + (5×z_α⁵ + 16×z_α³ + 3×z_α)/(96×df²) + ...

        精度: df>=4时误差 < 0.01, df>=10时误差 < 0.001
        """
        z = self._norm_ppf(p)
        # 一阶修正
        t = z + (z**3 + z) / (4.0 * df)
        # 二阶修正
        t += (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df * df)
        # 三阶修正 (df较小时重要)
        if df < 10:
            t += (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3)
        return t

    def _t_pdf(self, t: float, df: float) -> float:
        """Student-t分布PDF

        f(t) = Γ((df+1)/2) / (√(df×π) × Γ(df/2)) × (1 + t²/df)^(-(df+1)/2)
        """
        if _SCIPY_AVAILABLE:
            return float(_scipy_stats.t.pdf(t, df))
        # 使用对数Gamma避免大数溢出
        log_gamma_half_df = math.lgamma(df / 2.0)
        log_gamma_half_df_plus_1 = math.lgamma((df + 1.0) / 2.0)
        log_pdf = (log_gamma_half_df_plus_1
                   - log_gamma_half_df
                   - 0.5 * math.log(df * math.pi)
                   - ((df + 1.0) / 2.0) * math.log1p(t * t / df))
        return math.exp(log_pdf)

    def _generate_student_t_fallback(self, df: float, n: int) -> np.ndarray:
        """Student-t随机数生成的回退方案 (无scipy且numpy.standard_t不可用时)

        原理: 若Z~N(0,1)且V~χ²(df), 则 T = Z / √(V/df) 服从t(df)
        """
        z = np.random.standard_normal(n)
        # χ²(df) = Gamma(shape=df/2, scale=2)
        chi2 = np.random.gamma(df / 2.0, 2.0, size=n)
        return z / np.sqrt(chi2 / df)


# ============================================================================
# 自测代码
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("VarCvarRiskModule 自测")
    print("=" * 70)

    # 生成模拟的加密货币日收益率 (Student-t分布, 厚尾)
    np.random.seed(42)
    n_days = 365
    true_mu = 0.0005  # 日均0.05%收益
    true_sigma = 0.025  # 2.5%日波动率
    true_df = 4.0
    # 用Student-t生成厚尾数据
    raw_t = np.random.standard_t(true_df, size=n_days)
    sigma_scaled = true_sigma * math.sqrt((true_df - 2.0) / true_df)
    returns = true_mu + sigma_scaled * raw_t

    module = VarCvarRiskModule(default_df=4.0, monte_carlo_simulations=5000, random_seed=42)

    # 测试1: 历史模拟法
    print("\n[Test 1] 历史模拟法 (95% VaR)")
    hist = module.calculate_historical(returns, confidence=0.95)
    print(f"  VaR = {hist.var:.4%} ({hist.var*100:.2f}%)")
    print(f"  CVaR = {hist.cvar:.4%} ({hist.cvar*100:.2f}%)")
    print(f"  均值 = {hist.mean_return:.4%}, 标准差 = {hist.std_return:.4%}")
    print(f"  偏度 = {hist.skewness:.3f}, 峰度 = {hist.kurtosis:.3f}")
    print(f"  样本数 = {hist.sample_size}")
    assert hist.cvar >= hist.var, "CVaR必须 >= VaR"
    assert hist.sample_size == n_days
    print("  [PASS] CVaR >= VaR, 样本数正确")

    # 测试2: 参数法-正态
    print("\n[Test 2] 参数法-正态分布 (95% VaR)")
    normal = module.calculate_parametric_normal(returns, confidence=0.95)
    print(f"  VaR = {normal.var:.4%}, CVaR = {normal.cvar:.4%}")
    print(f"  z_alpha = {normal.extra.get('z_alpha', 'N/A'):.4f}")
    assert normal.cvar >= normal.var
    assert normal.extra.get("z_alpha", 0) < 0  # 5%分位数应为负
    print("  [PASS] CVaR >= VaR, z_alpha < 0")

    # 测试3: 参数法-Student-t
    print("\n[Test 3] 参数法-Student-t (95% VaR, df=4)")
    t_metrics = module.calculate_parametric_t(returns, confidence=0.95, df=4.0)
    print(f"  VaR = {t_metrics.var:.4%}, CVaR = {t_metrics.cvar:.4%}")
    print(f"  t_alpha = {t_metrics.extra.get('t_alpha', 'N/A'):.4f}")
    print(f"  sigma_scaled = {t_metrics.extra.get('sigma_scaled', 'N/A'):.4%}")
    assert t_metrics.cvar >= t_metrics.var
    # Student-t的VaR应该 >= 正态的VaR (厚尾效应)
    print(f"  Student-t VaR ({t_metrics.var:.4%}) vs Normal VaR ({normal.var:.4%}): "
          f"差值 = {(t_metrics.var - normal.var)*100:+.3f}%")
    print("  [PASS] CVaR >= VaR")

    # 测试4: Monte Carlo
    print("\n[Test 4] Monte Carlo (10000次模拟, 95% VaR)")
    mc = module.calculate_monte_carlo(returns, confidence=0.95, n_simulations=10000)
    print(f"  VaR = {mc.var:.4%}, CVaR = {mc.cvar:.4%}")
    print(f"  模拟次数 = {mc.extra.get('n_simulations', 'N/A')}")
    assert mc.cvar >= mc.var
    assert mc.extra.get("n_simulations") == 10000
    print("  [PASS] CVaR >= VaR, 模拟次数正确")

    # 测试5: 综合计算 (反温室原则: 取最大VaR)
    print("\n[Test 5] 综合计算 (反温室原则)")
    all_metrics = module.calculate_all(returns, confidence=0.95)
    for name, m in all_metrics.items():
        if name == "recommended":
            continue
        print(f"  {name:25s}: VaR={m.var:.4%}, CVaR={m.cvar:.4%}")
    recommended = all_metrics["recommended"]
    print(f"  {'recommended':25s}: VaR={recommended.var:.4%}, CVaR={recommended.cvar:.4%} "
          f"(来源: {recommended.extra.get('recommended_basis', 'N/A')})")
    # 推荐方法应该是最大VaR
    all_vars = [all_metrics[k].var for k in ["historical", "parametric_normal", "parametric_t", "monte_carlo"]]
    assert abs(recommended.var - max(all_vars)) < 1e-9, "推荐VaR应该等于所有方法的最大值"
    print("  [PASS] 推荐VaR = max(所有方法VaR) (反温室原则)")

    # 测试6: 仓位推荐
    print("\n[Test 6] 仓位推荐 (账户权益=10000, VaR限制=2%)")
    rec = module.recommend_position_size(
        account_equity=10000.0,
        returns=returns,
        max_var_pct=0.02,
        max_position_pct=0.25,
        kelly_suggested_pct=0.15,  # Kelly建议15%
    )
    print(f"  推荐仓位 = {rec.recommended_position_pct:.4%}")
    print(f"  VaR限制 = {rec.var_limit_pct:.2%}")
    print(f"  实际VaR = {rec.actual_var_pct:.4%}")
    print(f"  风险调整 = {rec.risk_adjustment:.3f}")
    print(f"  原因: {rec.reason}")
    assert rec.actual_var_pct <= rec.var_limit_pct + 1e-9, "实际VaR必须 <= 限制"
    assert rec.recommended_position_pct <= rec.max_position_pct
    print("  [PASS] 实际VaR <= 限制, 仓位 <= 最大值")

    # 测试7: 边界条件 - 空数组
    print("\n[Test 7] 边界条件 — 空数组")
    empty = module.calculate_historical([], confidence=0.95)
    assert empty.sample_size == 0
    assert empty.var == 0.0
    print("  [PASS] 空数组返回空metrics")

    # 测试8: 边界条件 - 极小样本
    print("\n[Test 8] 边界条件 — 极小样本 (5个点)")
    tiny = module.calculate_historical([0.01, -0.02, 0.005, -0.01, 0.015], confidence=0.95)
    print(f"  VaR = {tiny.var:.4%}, 样本数 = {tiny.sample_size}")
    assert tiny.sample_size == 5
    print("  [PASS] 极小样本可计算(应产生警告)")

    # 测试9: 99%置信度 vs 95%置信度
    print("\n[Test 9] 置信度对比 (95% vs 99%)")
    var_95 = module.calculate_parametric_t(returns, confidence=0.95)
    var_99 = module.calculate_parametric_t(returns, confidence=0.99)
    print(f"  95% VaR = {var_95.var:.4%}, 99% VaR = {var_99.var:.4%}")
    assert var_99.var > var_95.var, "99% VaR必须 > 95% VaR"
    print("  [PASS] 99% VaR > 95% VaR (置信度越高, VaR越大)")

    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)
