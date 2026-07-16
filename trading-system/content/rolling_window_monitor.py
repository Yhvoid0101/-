# -*- coding: utf-8 -*-
"""
滚动窗口稳定性监控 + 真实月度亏损概率 + Per-Symbol 参数差异化 (Phase 4)
================================================================
来源: v598 Phase 4 续作计划 — 用户铁律三件套
  1. "永远永远不要出现模拟牛逼, 实盘亏损的情况" → 持续滚动监控实盘偏离
  2. "复盘不是摆设, 给进化迭代提供真实数据支持" → 真实 mlp 替代硬编码 4.76%
  3. "不要通过已知数据反推策略" → per-symbol 差异基于统计特性而非历史最优

设计原则 (避免过拟合):
  1. mlp 从运行时 daily_pnls 实时计算, 而非从离线报告硬编码
  2. per-symbol 参数差异基于 symbol 波动率分位数 (统计特性), 不基于历史最优参数
  3. 滚动窗口 90 天持续监控, 而非一次性部署前检查
  4. 退化检测基于相对变化 (current/baseline), 而非绝对阈值 (避免过拟合历史数据)

核心组件:
  1. compute_portfolio_mlp() — 从 daily_pnls 计算真实月度亏损概率 (纯函数)
  2. RollingWindowStabilityMonitor — 90 天滚动窗口稳定性监控
  3. SymbolRiskProfiler — per-symbol 参数差异化框架 (基于波动率分位数)

主动应用教训链:
  ERR-20260701-v88: Tier 1 8/8 KPI 全达标 (mlp=0%) → 运行时需持续验证
  ERR-20260701-v91em: 90 天 71.4% (统计噪声) vs 180 天 100% → 短期窗口需谨慎解读
  ERR-106 (v576): LINK trail 极度敏感 → per-symbol 差异化是必须
  ERR-109 (v518): 均值回归非趋势 → 风控参数不应一刀切
  用户铁律: "复盘给进化迭代提供真实数据支持" → mlp 必须真实计算
================================================================
"""
from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List

logger = logging.getLogger("hermes.rolling_window")


# ============================================================
# 1. 真实月度亏损概率计算 (mlp) — 替代硬编码 4.76% 空壳
# ============================================================
# 来源: archive/legacy/_v362_take_profit_optimization.py:406
#   portfolio_mlp = loss_months / len(portfolio_monthly_pnl)
#
# 用户铁律: "复盘不是摆设" — mlp 必须从真实交易 PnL 计算
# 设计: 将 daily_pnls 按 30 天/月分组, 统计亏损月占比


def compute_portfolio_mlp(
    daily_pnls: List[float],
    days_per_month: int = 30,
    min_months: int = 1,
) -> float:
    """从每日 PnL 列表计算真实月度亏损概率 (Portfolio Monthly Loss Probability)

    Args:
        daily_pnls: 每日 PnL 列表 (百分比或绝对值均可, 仅看符号)
        days_per_month: 每月天数 (默认30, 加密 24/7 无周末)
        min_months: 最少月份数, 不足返回 0.0 (避免小样本误判)

    Returns:
        月度亏损概率 (0-100), 例: 11 个月中 1 个月亏损 → 9.09

    设计原则:
      - 只看月度 PnL 符号 (sum<0 = 亏损月)
      - 不足 min_months 月数据 → 返回 0.0 (不算达标, 但不算超标)
      - 用户铁律: "月度亏损概率≤5%" → 实盘硬约束

    主动应用教训:
      ERR-20260701-v88: v88 mlp=0% (11月全盈) → 运行时需持续监控
      ERR-20260701-v91em: v91 mlp=0% (11月全盈) → 真实计算保持
    """
    if not daily_pnls:
        return 0.0

    n_days = len(daily_pnls)
    n_months = n_days // days_per_month

    if n_months < min_months:
        # 数据不足, 返回 0.0 (不算超标, 但调用方应检查 n_months)
        return 0.0

    loss_months = 0
    for m in range(n_months):
        start = m * days_per_month
        end = start + days_per_month
        month_pnl = sum(daily_pnls[start:end])
        if month_pnl < 0:
            loss_months += 1

    mlp = (loss_months / n_months) * 100.0
    return mlp


def compute_monthly_pnl_series(
    daily_pnls: List[float],
    days_per_month: int = 30,
) -> List[float]:
    """将每日 PnL 聚合为月度 PnL 序列 (用于诊断)

    Returns:
        月度 PnL 列表 (sum of daily in each month)
    """
    if not daily_pnls:
        return []

    n_months = len(daily_pnls) // days_per_month
    monthly = []
    for m in range(n_months):
        start = m * days_per_month
        end = start + days_per_month
        monthly.append(sum(daily_pnls[start:end]))
    return monthly


# ============================================================
# 2. 滚动窗口稳定性监控 — 90 天持续监控 (替代一次性部署前检查)
# ============================================================
# 用户铁律: "永远永远不要出现模拟牛逼, 实盘亏损的情况"
# 来源: v598 Phase 4 探索发现 R17-4 仅一次性检查, 无持续滚动监控
# 设计:
#   - 维护 90 天 deque(maxlen=90) 的 daily_pnls
#   - 计算 rolling Sharpe / max_dd / win_rate / mlp
#   - 与部署基准 (baseline) 对比, 检测退化
#   - 退化阈值:
#       WARN: rolling Sharpe < 0.7 * baseline (退化 30%)
#       BLOCK: rolling Sharpe < 0.5 * baseline (退化 50%) 或 rolling mlp > 5%


@dataclass
class WindowSnapshot:
    """滚动窗口快照 — 某一时刻的 90 天滚动 KPI"""

    n_days: int = 0
    rolling_sharpe: float = 0.0
    rolling_max_dd_pct: float = 0.0
    rolling_win_rate_pct: float = 0.0
    rolling_mlp_pct: float = 0.0
    rolling_ann_pct: float = 0.0
    # 与基准的差异 (仅当 baseline 已设置时有效)
    sharpe_degradation_pct: float = 0.0  # 正数=退化, 负数=改善
    mlp_delta_pp: float = 0.0  # 正数=mlp 上升 (恶化), 负数=下降 (改善)
    # 评估结果
    verdict: str = "UNKNOWN"  # OK / WARN / BLOCK / INSUFFICIENT_DATA


class RollingWindowStabilityMonitor:
    """90 天滚动窗口稳定性监控器

    用户铁律: "永远永远不要出现模拟牛逼, 实盘亏损的情况"
    设计:
      1. 部署时设置 baseline (从 Tier 1 通过时的 KPI)
      2. 运行时持续 update() daily PnL
      3. 每次计算 rolling 90 天 KPI, 与 baseline 对比
      4. 退化超过阈值 → 触发 WARN/BLOCK

    线程安全: 内部 threading.Lock 保护 deque 操作
    """

    def __init__(
        self,
        window_days: int = 90,
        baseline_sharpe: float = 0.0,
        baseline_mlp_pct: float = 0.0,
        baseline_max_dd_pct: float = 0.0,
        warn_sharpe_degradation: float = 0.30,  # 30% 退化
        block_sharpe_degradation: float = 0.50,  # 50% 退化
        block_mlp_threshold: float = 5.0,  # 用户硬约束: 月亏概率 ≤ 5%
        periods_per_year: int = 365,  # 加密 24/7
    ) -> None:
        """初始化滚动窗口监控器

        Args:
            window_days: 滚动窗口天数 (默认 90)
            baseline_sharpe: 部署基准 Sharpe (Tier 1 通过时)
            baseline_mlp_pct: 部署基准 mlp (%)
            baseline_max_dd_pct: 部署基准 max_dd (%)
            warn_sharpe_degradation: WARN 退化阈值 (0.30 = 30% 退化)
            block_sharpe_degradation: BLOCK 退化阈值 (0.50 = 50% 退化)
            block_mlp_threshold: BLOCK mlp 阈值 (用户硬约束 5%)
            periods_per_year: 年化因子 (加密 365, 传统 252)
        """
        self.window_days = window_days
        self.baseline_sharpe = baseline_sharpe
        self.baseline_mlp_pct = baseline_mlp_pct
        self.baseline_max_dd_pct = baseline_max_dd_pct
        self.warn_sharpe_degradation = warn_sharpe_degradation
        self.block_sharpe_degradation = block_sharpe_degradation
        self.block_mlp_threshold = block_mlp_threshold
        self.periods_per_year = periods_per_year

        self._lock = threading.Lock()
        self._daily_pnls: Deque[float] = deque(maxlen=window_days)
        # 历史快照 (最近 N 次评估, 用于趋势分析)
        self._snapshot_history: Deque[WindowSnapshot] = deque(maxlen=30)
        # 退化计数 (连续 WARN/BLOCK 次数)
        self._consecutive_warns: int = 0
        self._consecutive_blocks: int = 0

    def set_baseline(
        self,
        sharpe: float,
        mlp_pct: float,
        max_dd_pct: float,
    ) -> None:
        """设置部署基准 (运行时首次 90 天数据自动设置, 或 Tier 1 通过后调用)

        用户铁律: "不通过已知数据反推策略" → baseline 来自运行时实际 KPI
        设计:
          - 默认行为: EvolutionLoop 在首次积累 90 天数据时自动设置 baseline
          - 高级用法: 可在 Tier 1 通过后手动调用此方法覆盖
          - baseline 一旦设置, 后续 rolling KPI 与之对比检测退化
        """
        with self._lock:
            self.baseline_sharpe = sharpe
            self.baseline_mlp_pct = mlp_pct
            self.baseline_max_dd_pct = max_dd_pct
            logger.info(
                "[RollingWindow] baseline 设置: sharpe=%.3f, mlp=%.2f%%, max_dd=%.2f%%",
                sharpe, mlp_pct, max_dd_pct,
            )

    def update(self, daily_pnl: float) -> None:
        """更新一条 daily PnL (线程安全)

        Args:
            daily_pnl: 当日 PnL (百分比或绝对值, 与 baseline 一致即可)
        """
        with self._lock:
            self._daily_pnls.append(float(daily_pnl))

    def update_batch(self, daily_pnls: List[float]) -> None:
        """批量更新 daily PnL (用于启动时预热)

        用户铁律: "不通过已知数据反推策略" → 预热仅填充历史数据, 不调整策略参数
        """
        with self._lock:
            for pnl in daily_pnls:
                self._daily_pnls.append(float(pnl))

    def snapshot(self) -> WindowSnapshot:
        """计算当前滚动窗口快照 (线程安全)

        Returns:
            WindowSnapshot: 当前 90 天滚动 KPI + 退化评估
        """
        with self._lock:
            pnls = list(self._daily_pnls)

        snap = WindowSnapshot(n_days=len(pnls))

        # 数据不足: 无法评估
        min_data = max(30, self.window_days // 3)  # 至少 30 天或 1/3 窗口
        if len(pnls) < min_data:
            snap.verdict = "INSUFFICIENT_DATA"
            return snap

        # 计算 rolling KPI
        try:
            import numpy as _np
            arr = _np.asarray(pnls, dtype=_np.float64)
            mean_val = float(arr.mean())
            std_val = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            # L4修复: 当 std=0 但 mean>0 时, Sharpe 应为极大值 (非0), 否则误判为退化
            # 数学上 std→0 且 mean>0 时 Sharpe→+∞, 实际取一个大数 100.0 表示极优
            if std_val > 1e-10:
                snap.rolling_sharpe = (mean_val / std_val) * math.sqrt(self.periods_per_year)
            elif mean_val > 0:
                snap.rolling_sharpe = 100.0  # 极优 (std=0 + mean>0)
            else:
                snap.rolling_sharpe = 0.0  # mean=0 真无信号
            snap.rolling_win_rate_pct = float(_np.sum(arr > 0) / len(arr)) * 100.0

            # 最大回撤 (基于累计 PnL)
            cumsum = _np.cumsum(arr)
            peak = _np.maximum.accumulate(cumsum)
            dd = _np.where(
                peak > 0,
                (peak - cumsum) / _np.where(peak > 0, peak, 1),
                0,
            )
            snap.rolling_max_dd_pct = float(_np.max(dd)) * 100.0 if len(dd) > 0 else 0.0

            # 年化收益 (假设 daily_pnls 是百分比)
            snap.rolling_ann_pct = mean_val * self.periods_per_year

            # mlp (30 天/月)
            snap.rolling_mlp_pct = compute_portfolio_mlp(pnls, days_per_month=30)
        except Exception as e:
            logger.warning("[RollingWindow] KPI 计算失败: %s", e)
            snap.verdict = "INSUFFICIENT_DATA"
            return snap

        # 退化评估 (仅当 baseline 已设置)
        if self.baseline_sharpe > 0:
            degradation = 1.0 - (snap.rolling_sharpe / max(self.baseline_sharpe, 1e-10))
            snap.sharpe_degradation_pct = max(0.0, degradation) * 100.0
        else:
            snap.sharpe_degradation_pct = 0.0

        snap.mlp_delta_pp = snap.rolling_mlp_pct - self.baseline_mlp_pct

        # 判定逻辑 (用户硬约束优先)
        # L7修复: 计数器自增与 history.append 移入锁内, 避免并发竞态
        with self._lock:
            # BLOCK: mlp > 5% (用户硬约束: "月度亏损概率≤5%")
            if snap.rolling_mlp_pct > self.block_mlp_threshold:
                snap.verdict = "BLOCK"
                self._consecutive_blocks += 1
                self._consecutive_warns = 0
            # BLOCK: Sharpe 退化 ≥ 50%
            elif (
                self.baseline_sharpe > 0
                and snap.sharpe_degradation_pct >= self.block_sharpe_degradation * 100
            ):
                snap.verdict = "BLOCK"
                self._consecutive_blocks += 1
                self._consecutive_warns = 0
            # WARN: Sharpe 退化 ≥ 30%
            elif (
                self.baseline_sharpe > 0
                and snap.sharpe_degradation_pct >= self.warn_sharpe_degradation * 100
            ):
                snap.verdict = "WARN"
                self._consecutive_warns += 1
                self._consecutive_blocks = 0
            # WARN: rolling mlp 上升超过 3pp (即使未超 5%)
            elif snap.mlp_delta_pp > 3.0:
                snap.verdict = "WARN"
                self._consecutive_warns += 1
                self._consecutive_blocks = 0
            else:
                snap.verdict = "OK"
                self._consecutive_warns = 0
                self._consecutive_blocks = 0

            # 历史快照 (用于趋势分析)
            self._snapshot_history.append(snap)

        return snap

    def get_alerts(self) -> Dict[str, Any]:
        """获取当前告警状态 (用于 can_deploy 决策)

        Returns:
            Dict with:
              - verdict: OK / WARN / BLOCK / INSUFFICIENT_DATA
              - consecutive_warns: 连续 WARN 次数
              - consecutive_blocks: 连续 BLOCK 次数
              - rolling_sharpe: 当前滚动 Sharpe
              - rolling_mlp_pct: 当前滚动 mlp
              - baseline_sharpe: 基准 Sharpe
              - sharpe_degradation_pct: 退化百分比
              - recommendation: 建议行动
        """
        snap = self.snapshot()
        recommendation = self._recommend(snap)

        return {
            "verdict": snap.verdict,
            "consecutive_warns": self._consecutive_warns,
            "consecutive_blocks": self._consecutive_blocks,
            "n_days": snap.n_days,
            "rolling_sharpe": round(snap.rolling_sharpe, 3),
            "rolling_mlp_pct": round(snap.rolling_mlp_pct, 2),
            "rolling_max_dd_pct": round(snap.rolling_max_dd_pct, 2),
            "rolling_win_rate_pct": round(snap.rolling_win_rate_pct, 2),
            "rolling_ann_pct": round(snap.rolling_ann_pct, 2),
            "baseline_sharpe": self.baseline_sharpe,
            "baseline_mlp_pct": self.baseline_mlp_pct,
            "sharpe_degradation_pct": round(snap.sharpe_degradation_pct, 2),
            "mlp_delta_pp": round(snap.mlp_delta_pp, 2),
            "recommendation": recommendation,
        }

    def _recommend(self, snap: WindowSnapshot) -> str:
        """根据快照生成建议行动"""
        if snap.verdict == "INSUFFICIENT_DATA":
            return f"数据不足 (n={snap.n_days}/{self.window_days}), 继续积累"
        if snap.verdict == "BLOCK":
            if snap.rolling_mlp_pct > self.block_mlp_threshold:
                return (
                    f"BLOCK: 滚动 mlp={snap.rolling_mlp_pct:.2f}% > {self.block_mlp_threshold}% "
                    f"(用户硬约束违反) → 立即暂停部署, 排查策略缺陷"
                )
            return (
                f"BLOCK: Sharpe 退化 {snap.sharpe_degradation_pct:.1f}% "
                f"(baseline={self.baseline_sharpe:.3f} → rolling={snap.rolling_sharpe:.3f}) "
                f"→ 暂停部署, 复盘退化根因"
            )
        if snap.verdict == "WARN":
            return (
                f"WARN: 退化 {snap.sharpe_degradation_pct:.1f}% / mlp+{snap.mlp_delta_pp:.2f}pp "
                f"→ 加强监控, 准备应急回滚"
            )
        return "OK: 滚动 KPI 稳定"

    def get_snapshot_history(self) -> List[WindowSnapshot]:
        """获取历史快照列表 (用于趋势分析)"""
        with self._lock:
            return list(self._snapshot_history)


# ============================================================
# 3. Per-Symbol 参数差异化框架 — 基于波动率分位数
# ============================================================
# 用户铁律: "不要通过已知数据反推策略, 那样就过拟合了"
# 来源教训:
#   ERR-106 (v576 LINK trail 极度敏感) → 不同 symbol 需差异化
#   ERR-109 (v518 均值回归非趋势) → 不能一刀切禁空头
#   ERR-20260701-v91atr: atr_abs 是 symbol 效应 → 必须控制 symbol
#
# 设计:
#   1. 基于 symbol 的 ATR 波动率分位数 (运行时统计), 而非历史最优参数
#   2. 高波动 symbol → 更严格的 consec_threshold + 更长 cooldown
#   3. 低波动 symbol → 更宽松的阈值
#   4. 提供默认配置 + 可覆盖接口


# 默认全局阈值 (与 evolution_loop.py L840-842 一致, 作为回退值)
DEFAULT_SYMBOL_PARAMS = {
    "consec_threshold": 5,  # v3.38b: 连续亏损5次触发冷却
    "cooldown_bars": 3,    # v3.38b: 冷却3个bar
    "max_concurrent": 15,  # v3.38b: 同symbol同方向最多15个并发
}


@dataclass
class SymbolRiskParams:
    """单个 symbol 的风控参数 (基于波动率分位数推导)"""

    symbol: str
    consec_threshold: int = DEFAULT_SYMBOL_PARAMS["consec_threshold"]
    cooldown_bars: int = DEFAULT_SYMBOL_PARAMS["cooldown_bars"]
    max_concurrent: int = DEFAULT_SYMBOL_PARAMS["max_concurrent"]
    # 推导依据 (用于可解释性)
    atr_pctile: float = 0.5  # 该 symbol 在种群中的 ATR 分位数 (0-1)
    derivation: str = "default"  # default / high_vol / low_vol / custom


class SymbolRiskProfiler:
    """Per-Symbol 风控参数差异化框架

    用户铁律: "不要通过已知数据反推策略" → 基于运行时统计特性, 不基于历史最优
    设计:
      1. 收集每个 symbol 的 ATR% 样本 (运行时滚动统计)
      2. 计算每个 symbol 在种群中的 ATR 分位数
      3. 高波动 symbol (top 33%) → consec_threshold=3 (更激进风控)
      4. 低波动 symbol (bot 33%) → consec_threshold=7 (更宽松)
      5. 中波动 symbol → 默认值 (5)

    来源教训:
      ERR-106 (v576 LINK trail 极度敏感) → LINK 应获得独立参数
      ERR-109 (v518 均值回归非趋势) → 不能用趋势过滤一刀切
      ERR-20260701-v91atr: 控制变量后 4/6 symbol d<0.2 → symbol 效应显著

    使用方式:
      profiler = SymbolRiskProfiler()
      # 运行时持续更新 ATR 样本
      profiler.update_atr("BTC/USDT", 0.022)
      profiler.update_atr("ETH/USDT", 0.018)
      # 获取参数
      params = profiler.get_params("BTC/USDT")
    """

    def __init__(
        self,
        high_vol_threshold: float = 0.67,  # top 33% = 高波动
        low_vol_threshold: float = 0.33,   # bot 33% = 低波动
        high_vol_consec_threshold: int = 3,  # 高波动更激进 (3 次就冷却)
        low_vol_consec_threshold: int = 7,   # 低波动更宽松 (7 次才冷却)
        high_vol_cooldown_bars: int = 5,     # 高波动冷却更长
        low_vol_cooldown_bars: int = 2,      # 低波动冷却更短
        high_vol_max_concurrent: int = 10,   # 高波动限制并发
        low_vol_max_concurrent: int = 20,    # 低波动允许更多并发
        min_samples: int = 30,               # 最少 ATR 样本数才生效
    ) -> None:
        """初始化 SymbolRiskProfiler

        Args:
            high_vol_threshold: 高波动分位数阈值 (0.67 = top 33%)
            low_vol_threshold: 低波动分位数阈值 (0.33 = bot 33%)
            high_vol_consec_threshold: 高波动 symbol 连续亏损阈值
            low_vol_consec_threshold: 低波动 symbol 连续亏损阈值
            high_vol_cooldown_bars: 高波动 symbol 冷却 bars
            low_vol_cooldown_bars: 低波动 symbol 冷却 bars
            high_vol_max_concurrent: 高波动 symbol 最大并发
            low_vol_max_concurrent: 低波动 symbol 最大并发
            min_samples: 最少 ATR 样本数 (不足则用默认值)
        """
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_consec_threshold = high_vol_consec_threshold
        self.low_vol_consec_threshold = low_vol_consec_threshold
        self.high_vol_cooldown_bars = high_vol_cooldown_bars
        self.low_vol_cooldown_bars = low_vol_cooldown_bars
        self.high_vol_max_concurrent = high_vol_max_concurrent
        self.low_vol_max_concurrent = low_vol_max_concurrent
        self.min_samples = min_samples

        self._lock = threading.Lock()
        # symbol → ATR% 样本 (deque maxlen=200, 最近200个样本)
        self._atr_samples: Dict[str, Deque[float]] = {}
        # 缓存的参数 (ATR 样本变化时重新计算)
        self._cached_params: Dict[str, SymbolRiskParams] = {}
        self._cache_dirty: bool = True

    def update_atr(self, symbol: str, atr_pct: float) -> None:
        """更新某 symbol 的 ATR% 样本 (运行时持续调用)

        Args:
            symbol: 交易对 (如 "BTC/USDT")
            atr_pct: 当前 ATR% (如 0.022 = 2.2%)
        """
        if not symbol or atr_pct != atr_pct or atr_pct <= 0:  # NaN/非正值跳过
            return
        with self._lock:
            if symbol not in self._atr_samples:
                self._atr_samples[symbol] = deque(maxlen=200)
            self._atr_samples[symbol].append(float(atr_pct))
            self._cache_dirty = True

    def _compute_percentiles(self) -> Dict[str, float]:
        """计算每个 symbol 在种群中的 ATR 分位数 (0-1)

        Returns:
            {symbol: percentile}, percentile ∈ [0, 1]
            例: BTC ATR=0.030 (最高) → 0.95
                ETH ATR=0.018 (中等) → 0.50
                LTC ATR=0.012 (最低) → 0.05
        """
        if not self._atr_samples:
            return {}

        # 计算每个 symbol 的 ATR 中位数 (鲁棒于异常值)
        symbol_medians: Dict[str, float] = {}
        for sym, samples in self._atr_samples.items():
            if len(samples) >= 10:  # 至少 10 个样本
                sorted_s = sorted(samples)
                mid = len(sorted_s) // 2
                symbol_medians[sym] = sorted_s[mid]

        if len(symbol_medians) < 2:
            return {}

        # 计算 symbol 在种群中的分位数 (使用标准 percentile rank 公式)
        # 公式: (rank-1)/(n-1), 其中 rank 从 1 开始
        # 例: 3个symbol → 最低=0.0, 中间=0.5, 最高=1.0 (避免 1/3 边界精度问题)
        all_medians = sorted(symbol_medians.values())
        n = len(all_medians)
        percentiles: Dict[str, float] = {}
        for sym, med in symbol_medians.items():
            # 该 symbol 中位数在所有 symbol 中位数中的排名 (1-based)
            rank = sum(1 for m in all_medians if m <= med)
            # 标准 percentile rank: (rank-1)/(n-1), n=1 时直接 0.5
            if n > 1:
                percentiles[sym] = (rank - 1) / (n - 1)
            else:
                percentiles[sym] = 0.5

        return percentiles

    def get_params(self, symbol: str) -> SymbolRiskParams:
        """获取某 symbol 的风控参数

        用户铁律: "不通过已知数据反推策略" → 基于运行时 ATR 分位数, 而非历史最优参数
        设计:
          - 样本不足 → 返回默认参数
          - 高波动 (top 33%) → 更严格阈值
          - 低波动 (bot 33%) → 更宽松阈值
          - 中波动 → 默认值
        """
        with self._lock:
            # 检查缓存是否有效
            if not self._cache_dirty and symbol in self._cached_params:
                return self._cached_params[symbol]

            samples = self._atr_samples.get(symbol)
            if samples is None or len(samples) < self.min_samples:
                # 样本不足, 用默认值
                params = SymbolRiskParams(
                    symbol=symbol,
                    consec_threshold=DEFAULT_SYMBOL_PARAMS["consec_threshold"],
                    cooldown_bars=DEFAULT_SYMBOL_PARAMS["cooldown_bars"],
                    max_concurrent=DEFAULT_SYMBOL_PARAMS["max_concurrent"],
                    atr_pctile=0.5,
                    derivation="default_insufficient_samples",
                )
                self._cached_params[symbol] = params
                return params

            # 计算分位数
            # L7.1 修复死锁: get_params 已持 self._lock, 必须调用 _compute_percentiles_impl
            # 不能调用 _compute_percentiles() (后者会再次 with self._lock → threading.Lock
            # 不可重入 → 永久阻塞死锁). 这是 L7 重构引入的回归.
            percentiles = self._compute_percentiles_impl()
            pctile = percentiles.get(symbol, 0.5)

            if pctile >= self.high_vol_threshold:
                # 高波动 symbol: 更激进风控 (避免连续亏损累积)
                params = SymbolRiskParams(
                    symbol=symbol,
                    consec_threshold=self.high_vol_consec_threshold,
                    cooldown_bars=self.high_vol_cooldown_bars,
                    max_concurrent=self.high_vol_max_concurrent,
                    atr_pctile=pctile,
                    derivation="high_vol",
                )
            elif pctile <= self.low_vol_threshold:
                # 低波动 symbol: 更宽松阈值 (波动小, 可容忍更多连续亏损)
                params = SymbolRiskParams(
                    symbol=symbol,
                    consec_threshold=self.low_vol_consec_threshold,
                    cooldown_bars=self.low_vol_cooldown_bars,
                    max_concurrent=self.low_vol_max_concurrent,
                    atr_pctile=pctile,
                    derivation="low_vol",
                )
            else:
                # 中波动: 默认值
                params = SymbolRiskParams(
                    symbol=symbol,
                    consec_threshold=DEFAULT_SYMBOL_PARAMS["consec_threshold"],
                    cooldown_bars=DEFAULT_SYMBOL_PARAMS["cooldown_bars"],
                    max_concurrent=DEFAULT_SYMBOL_PARAMS["max_concurrent"],
                    atr_pctile=pctile,
                    derivation="medium_vol",
                )

            self._cached_params[symbol] = params
            self._cache_dirty = False
            return params

    def get_all_params(self) -> Dict[str, SymbolRiskParams]:
        """获取所有已观测 symbol 的参数 (用于诊断/日志)"""
        with self._lock:
            symbols = list(self._atr_samples.keys())
        return {sym: self.get_params(sym) for sym in symbols}

    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息 (用于日志/监控)"""
        # L7修复: _compute_percentiles 持锁执行, 避免 update_atr 并发修改 deque
        with self._lock:
            n_symbols = len(self._atr_samples)
            sample_counts = {
                sym: len(samples) for sym, samples in self._atr_samples.items()
            }
            # 在锁内计算 percentiles (迭代 self._atr_samples, 避免并发修改)
            percentiles = self._compute_percentiles_locked()
        return {
            "n_symbols": n_symbols,
            "sample_counts": sample_counts,
            "percentiles": {sym: round(p, 3) for sym, p in percentiles.items()},
            "high_vol_symbols": [
                sym for sym, p in percentiles.items()
                if p >= self.high_vol_threshold
            ],
            "low_vol_symbols": [
                sym for sym, p in percentiles.items()
                if p <= self.low_vol_threshold
            ],
            "min_samples_required": self.min_samples,
        }

    def _compute_percentiles_locked(self) -> Dict[str, float]:
        """计算 percentiles (调用方必须已持锁, 避免并发修改 _atr_samples)

        L7修复: 与 _compute_percentiles 逻辑一致, 但要求调用方持锁.
        保留 _compute_percentiles 公开版本以兼容 (内部也调用此方法).
        """
        return self._compute_percentiles_impl()

    def _compute_percentiles(self) -> Dict[str, float]:
        """计算每个 symbol 在种群中的 ATR 分位数 (兼容旧接口, 内部自动持锁)"""
        with self._lock:
            return self._compute_percentiles_impl()

    def _compute_percentiles_impl(self) -> Dict[str, float]:
        """percentile 计算实现 (调用方持锁或在 _compute_percentiles 中调用)"""
        if not self._atr_samples:
            return {}

        # 计算每个 symbol 的 ATR 中位数 (鲁棒于异常值)
        symbol_medians: Dict[str, float] = {}
        for sym, samples in self._atr_samples.items():
            if len(samples) >= 10:  # 至少 10 个样本
                sorted_s = sorted(samples)
                mid = len(sorted_s) // 2
                symbol_medians[sym] = sorted_s[mid]

        if len(symbol_medians) < 2:
            return {}

        # 计算 symbol 在种群中的分位数 (使用标准 percentile rank 公式)
        # 公式: (rank-1)/(n-1), 其中 rank 从 1 开始
        # 例: 3个symbol → 最低=0.0, 中间=0.5, 最高=1.0 (避免 1/3 边界精度问题)
        all_medians = sorted(symbol_medians.values())
        n = len(all_medians)
        percentiles: Dict[str, float] = {}
        for sym, med in symbol_medians.items():
            # 该 symbol 中位数在所有 symbol 中位数中的排名 (1-based)
            rank = sum(1 for m in all_medians if m <= med)
            # 标准 percentile rank: (rank-1)/(n-1), n=1 时直接 0.5
            if n > 1:
                percentiles[sym] = (rank - 1) / (n - 1)
            else:
                percentiles[sym] = 0.5

        return percentiles


# ============================================================
# 4. 内置自检 (L6 功能验证)
# ============================================================

def _self_test() -> bool:
    """内置自检 — 验证三个核心组件功能正常

    Returns:
        True = 全部通过, False = 有失败
    """
    try:
        # 1. mlp 计算
        # 11 个月全盈 → mlp=0%
        pnls_all_win = [10.0] * 330  # 330 天 = 11 月
        mlp1 = compute_portfolio_mlp(pnls_all_win, days_per_month=30)
        assert abs(mlp1 - 0.0) < 0.01, f"全盈 mlp 应为 0, 实际 {mlp1}"

        # 11 个月全亏 → mlp=100%
        pnls_all_loss = [-10.0] * 330
        mlp2 = compute_portfolio_mlp(pnls_all_loss, days_per_month=30)
        assert abs(mlp2 - 100.0) < 0.01, f"全亏 mlp 应为 100, 实际 {mlp2}"

        # 11 个月中 1 月亏 → mlp≈9.09%
        pnls_mixed = [10.0] * 300 + [-10.0] * 30
        mlp3 = compute_portfolio_mlp(pnls_mixed, days_per_month=30)
        assert abs(mlp3 - (1 / 11 * 100)) < 0.01, f"1/11 月亏 mlp 应≈9.09, 实际 {mlp3}"

        # 不足 1 月 → mlp=0
        mlp4 = compute_portfolio_mlp([10.0] * 10, days_per_month=30)
        assert mlp4 == 0.0, f"不足1月 mlp 应为 0, 实际 {mlp4}"

        # 2. 滚动窗口监控
        monitor = RollingWindowStabilityMonitor(
            window_days=90,
            baseline_sharpe=3.0,
            baseline_mlp_pct=0.0,
        )
        # 预热 90 天数据 (高 Sharpe)
        monitor.update_batch([1.0] * 90)
        snap = monitor.snapshot()
        assert snap.n_days == 90, f"快照 n_days 应为 90, 实际 {snap.n_days}"
        assert snap.rolling_mlp_pct == 0.0, f"全盈滚动 mlp 应为 0, 实际 {snap.rolling_mlp_pct}"
        assert snap.verdict == "OK", f"全盈快照应为 OK, 实际 {snap.verdict}"

        # 模拟退化 (50% 亏损日)
        monitor2 = RollingWindowStabilityMonitor(
            window_days=90,
            baseline_sharpe=3.0,
            baseline_mlp_pct=0.0,
        )
        # 60 天盈利 + 30 天亏损 → 滚动 mlp 可能 > 0
        monitor2.update_batch([1.0] * 60 + [-1.0] * 30)
        snap2 = monitor2.snapshot()
        assert snap2.n_days == 90
        # 滚动 90 天 = 3 个月, 后 1 月全亏 → mlp = 33.33%
        assert snap2.rolling_mlp_pct > 5.0, f"退化后 mlp 应 > 5%, 实际 {snap2.rolling_mlp_pct}"
        assert snap2.verdict == "BLOCK", f"mlp>5% 应 BLOCK, 实际 {snap2.verdict}"

        alerts = monitor2.get_alerts()
        assert "recommendation" in alerts
        assert alerts["consecutive_blocks"] >= 1

        # 3. Per-Symbol 参数差异化
        profiler = SymbolRiskProfiler(min_samples=5)
        # BTC 高波动, LTC 低波动
        for _ in range(20):
            profiler.update_atr("BTC/USDT", 0.035)  # 高波动
            profiler.update_atr("ETH/USDT", 0.022)  # 中波动
            profiler.update_atr("LTC/USDT", 0.010)  # 低波动

        btc_params = profiler.get_params("BTC/USDT")
        eth_params = profiler.get_params("ETH/USDT")
        ltc_params = profiler.get_params("LTC/USDT")

        assert btc_params.derivation == "high_vol", f"BTC 应为 high_vol, 实际 {btc_params.derivation}"
        assert ltc_params.derivation == "low_vol", f"LTC 应为 low_vol, 实际 {ltc_params.derivation}"
        assert btc_params.consec_threshold < ltc_params.consec_threshold, \
            f"高波动 consec_threshold ({btc_params.consec_threshold}) 应 < 低波动 ({ltc_params.consec_threshold})"
        assert btc_params.cooldown_bars > ltc_params.cooldown_bars, \
            f"高波动 cooldown ({btc_params.cooldown_bars}) 应 > 低波动 ({ltc_params.cooldown_bars})"

        # 样本不足 → 默认值
        unknown_params = profiler.get_params("UNKNOWN/USDT")
        assert unknown_params.derivation.startswith("default"), \
            f"未知 symbol 应为 default, 实际 {unknown_params.derivation}"

        diagnostics = profiler.get_diagnostics()
        assert diagnostics["n_symbols"] == 3
        assert "BTC/USDT" in diagnostics["high_vol_symbols"]
        assert "LTC/USDT" in diagnostics["low_vol_symbols"]

        logger.info("[RollingWindow _self_test] 全部通过")
        return True

    except AssertionError as e:
        logger.error("[RollingWindow _self_test] 失败: %s", e)
        return False
    except Exception as e:
        logger.error("[RollingWindow _self_test] 异常: %s: %s", type(e).__name__, e)
        return False


# ============================================================
# CLI 入口 (可选, 用于独立测试)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = _self_test()
    if result:
        print("\n✅ rolling_window_monitor 自检全部通过")
    else:
        print("\n❌ rolling_window_monitor 自检失败")
        import sys
        sys.exit(1)
