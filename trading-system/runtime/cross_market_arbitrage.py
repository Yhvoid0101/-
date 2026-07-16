# -*- coding: utf-8 -*-
"""
跨市场套利引擎 (Cross-Market Arbitrage) — v3.4 新增

填补 7 维覆盖矩阵中的 "commodity 资产类别" 空白。

核心理论:
  1. BTC ↔ Gold/Silver/Oil 跨市场传导 (Rolling Cointegration + Z-Score)
  2. DXY 传导链 (美元强弱 → 商品 → BTC) — 来源: macro_liquidity_cycle 关联研究
  3. 风险偏好 Regime (RISK_ON / RISK_OFF / NEUTRAL)
  4. 滚动相关性突变检测 (Correlation Breakout/Breakdown)
  5. 通胀对冲信号 (BTC+Gold 同涨=通胀对冲开启; BTC跌Gold涨=避险情绪)

学术来源:
  - Cross-Market Linkages: Baur & McDermott (2010) "Is gold a safe haven?"
  - DXY-BTC Correlation: Bloomberg 2026 (BTC vs DXY 90d 相关性 -0.62)
  - Inflation Hedge: Bouri et al. (2017) "Does Bitcoin act as a hedge?"
  - Regime-Switching: Hamilton (1989) + Filardo (1994) TVTP

信号类型 (CrossMarketSignalType):
  CROSS_MARKET_BUY       — BTC vs 商品价差收敛做多
  CROSS_MARKET_SELL      — BTC vs 商品价差发散做空
  DXY_WARNING            — DXY 强势预警
  RISK_OFF_HEDGE         — 避险对冲信号
  CORRELATION_BREAK      — 相关性突变
  INFLATION_HEDGE        — 通胀对冲开启
  REGIME_SHIFT           — 风险偏好切换
  HOLD                   — 持仓观望
  NEUTRAL                — 中性

纯 numpy 实现, 无 PyTorch/TF 依赖, 可在沙盘中即时进化。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# 枚举定义
# ============================================================================

class CrossMarketSignalType(Enum):
    """跨市场套利信号类型"""
    CROSS_MARKET_BUY = "cross_market_buy"           # BTC vs 商品价差收敛做多
    CROSS_MARKET_SELL = "cross_market_sell"         # BTC vs 商品价差发散做空
    DXY_WARNING = "dxy_warning"                     # DXY 强势预警
    RISK_OFF_HEDGE = "risk_off_hedge"               # 避险对冲信号
    CORRELATION_BREAK = "correlation_break"         # 相关性突变
    INFLATION_HEDGE = "inflation_hedge"             # 通胀对冲开启
    REGIME_SHIFT = "regime_shift"                   # 风险偏好切换
    HOLD = "hold"                                    # 持仓观望
    NEUTRAL = "neutral"                              # 中性


class RiskAppetiteRegime(Enum):
    """风险偏好状态"""
    RISK_ON = "risk_on"          # 风险偏好开启(BTC+股票涨, 黄金跌)
    RISK_OFF = "risk_off"        # 避险情绪(BTC+股票跌, 黄金涨)
    NEUTRAL = "neutral"          # 中性
    TRANSITIONING = "transitioning"  # 切换中


class CommodityClass(Enum):
    """商品类别"""
    PRECIOUS_METAL = "precious_metal"   # 贵金属(Gold/Silver/Platinum)
    ENERGY = "energy"                   # 能源(Oil/NatGas)
    INDUSTRIAL = "industrial"           # 工业金属(Copper/Aluminum)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class MarketSnapshot:
    """市场快照 — 多市场同步价格"""
    btc: float = 0.0
    eth: float = 0.0
    gold: float = 0.0           # XAUUSD
    silver: float = 0.0         # XAGUSD
    oil: float = 0.0            # WTI Crude
    copper: float = 0.0
    dxy: float = 100.0          # 美元指数
    sp500: float = 0.0          # S&P 500
    vix: float = 15.0           # 恐慌指数
    ten_year_yield: float = 0.0  # 10年期美债收益率


@dataclass
class CrossMarketOpportunity:
    """跨市场套利机会"""
    signal_type: CrossMarketSignalType
    primary_asset: str
    secondary_asset: str
    z_score: float
    correlation: float
    conviction: float       # 0.0 - 1.0
    expected_holding_period: str   # short / medium / long
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossMarketSignal:
    """跨市场套利综合信号"""
    signal_type: CrossMarketSignalType
    direction: int                # +1 多 / -1 空 / 0 观望
    conviction: float             # 0.0 - 1.0
    regime: RiskAppetiteRegime
    btc_gold_corr: float
    btc_dxy_corr: float
    inflation_hedge_score: float
    risk_off_score: float
    opportunities: List[CrossMarketOpportunity] = field(default_factory=list)
    timestamp: float = 0.0


# ============================================================================
# 主类: CrossMarketArbitrage
# ============================================================================

class CrossMarketArbitrage:
    """
    跨市场套利引擎

    覆盖维度:
      - 资产类别: crypto_major + commodity + forex + index (4类)
      - 信号类型: cross_market + macro + hedge
      - 时间框架: 1d / 1w / 1M (中长期)
    """

    # ------------------------------------------------------------------
    # 类常量 (可被基因进化调整)
    # ------------------------------------------------------------------
    CORRELATION_WINDOW = 60              # 滚动相关性窗口
    COINTEGRATION_WINDOW = 120           # 协整检验窗口
    ZSCORE_ENTRY = 2.0                   # Z-score 入场阈值
    ZSCORE_EXIT = 0.5                    # Z-score 平仓阈值
    ZSCORE_STOP = 3.5                    # Z-score 止损阈值

    CORRELATION_BREAK_THRESHOLD = 0.3    # 相关性突变阈值(变化量)
    HIGH_CORRELATION_THRESHOLD = 0.6     # 高相关阈值
    NEGATIVE_CORRELATION_THRESHOLD = -0.5  # 负相关阈值

    DXY_STRONG_THRESHOLD = 104.0         # DXY 强势阈值
    DXY_WEAK_THRESHOLD = 96.0            # DXY 弱势阈值

    VIX_PANIC_THRESHOLD = 30.0           # VIX 恐慌阈值
    VIX_EXTREME_THRESHOLD = 40.0         # VIX 极端阈值

    INFLATION_HEDGE_CORR_MIN = 0.4       # 通胀对冲要求最低相关性
    RISK_OFF_BTC_DROP_MIN = -0.02        # RISK_OFF: BTC 单日跌幅阈值
    RISK_OFF_GOLD_RISE_MIN = 0.005       # RISK_OFF: 黄金单日涨幅阈值

    HISTORY_WINDOW = 200                 # 历史数据窗口
    COINTEGRATION_P_VALUE_MAX = 0.10     # 协整 p 值上限

    # 商品类别映射
    COMMODITY_MAPPING = {
        'gold': CommodityClass.PRECIOUS_METAL,
        'silver': CommodityClass.PRECIOUS_METAL,
        'oil': CommodityClass.ENERGY,
        'copper': CommodityClass.INDUSTRIAL,
    }

    # 信号权重
    WEIGHT_COINTEGRATION = 0.35
    WEIGHT_CORRELATION = 0.25
    WEIGHT_DXY = 0.20
    WEIGHT_REGIME = 0.20

    def __init__(self):
        # 历史快照
        self.history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self.current_snapshot: Optional[MarketSnapshot] = None

        # 滚动统计
        self.btc_returns: deque = deque(maxlen=self.CORRELATION_WINDOW)
        self.gold_returns: deque = deque(maxlen=self.CORRELATION_WINDOW)
        self.silver_returns: deque = deque(maxlen=self.CORRELATION_WINDOW)
        self.oil_returns: deque = deque(maxlen=self.CORRELATION_WINDOW)
        self.dxy_returns: deque = deque(maxlen=self.CORRELATION_WINDOW)
        self.sp500_returns: deque = deque(maxlen=self.CORRELATION_WINDOW)

        # 相关性缓存
        self.btc_gold_corr: float = 0.0
        self.btc_silver_corr: float = 0.0
        self.btc_oil_corr: float = 0.0
        self.btc_dxy_corr: float = 0.0
        self.btc_sp500_corr: float = 0.0
        self.gold_dxy_corr: float = 0.0
        self.prev_btc_gold_corr: float = 0.0  # 用于突变检测

        # 协整缓存
        self.btc_gold_cointegration: float = 1.0  # p-value
        self.btc_silver_cointegration: float = 1.0
        self.btc_oil_cointegration: float = 1.0

        # Z-score (BTC vs 商品价差)
        self.btc_gold_zscore: float = 0.0
        self.btc_silver_zscore: float = 0.0
        self.btc_oil_zscore: float = 0.0

        # 状态
        self.current_regime: RiskAppetiteRegime = RiskAppetiteRegime.NEUTRAL
        self.inflation_hedge_score: float = 0.0
        self.risk_off_score: float = 0.0
        self.dxy_strength: float = 0.0  # -1.0 (弱) ~ +1.0 (强)

        # 上次信号
        self.last_signal: Optional[CrossMarketSignal] = None

        # 统计
        self.signal_count: int = 0
        self.regime_shift_count: int = 0

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_snapshot(self, snapshot: MarketSnapshot) -> None:
        """更新市场快照"""
        if self.current_snapshot is not None:
            self._update_returns(self.current_snapshot, snapshot)

        self.history.append(snapshot)
        self.current_snapshot = snapshot

        if len(self.history) >= 2:
            self._update_correlations()
            self._update_cointegration()
            self._update_zscores()
            self._detect_regime()

    def _update_returns(self, prev: MarketSnapshot, curr: MarketSnapshot) -> None:
        """计算并存储收益率"""
        def safe_return(p_prev: float, p_curr: float) -> float:
            if p_prev <= 0 or p_curr <= 0:
                return 0.0
            return math.log(p_curr / p_prev)

        self.btc_returns.append(safe_return(prev.btc, curr.btc))
        self.gold_returns.append(safe_return(prev.gold, curr.gold))
        self.silver_returns.append(safe_return(prev.silver, curr.silver))
        self.oil_returns.append(safe_return(prev.oil, curr.oil))
        self.dxy_returns.append(safe_return(prev.dxy, curr.dxy))
        self.sp500_returns.append(safe_return(prev.sp500, curr.sp500))

    def _safe_corr(self, x: List[float], y: List[float]) -> float:
        """安全相关性计算"""
        if len(x) < 10:
            return 0.0
        try:
            xa = np.array(x, dtype=float)
            ya = np.array(y, dtype=float)
            if xa.std() < 1e-8 or ya.std() < 1e-8:
                return 0.0
            corr = float(np.corrcoef(xa, ya)[0, 1])
            if math.isnan(corr):
                return 0.0
            return corr
        except Exception:
            return 0.0

    def _update_correlations(self) -> None:
        """更新所有相关性"""
        self.prev_btc_gold_corr = self.btc_gold_corr

        btc = list(self.btc_returns)
        gold = list(self.gold_returns)
        silver = list(self.silver_returns)
        oil = list(self.oil_returns)
        dxy = list(self.dxy_returns)
        sp500 = list(self.sp500_returns)

        self.btc_gold_corr = self._safe_corr(btc, gold)
        self.btc_silver_corr = self._safe_corr(btc, silver)
        self.btc_oil_corr = self._safe_corr(btc, oil)
        self.btc_dxy_corr = self._safe_corr(btc, dxy)
        self.btc_sp500_corr = self._safe_corr(btc, sp500)
        self.gold_dxy_corr = self._safe_corr(gold, dxy)

    def _ols_cointegration(self, x: List[float], y: List[float]) -> float:
        """
        简化版 Engle-Granger 协整检验
        返回 ADF 检验的近似 p-value
        """
        if len(x) < 30 or len(y) < 30:
            return 1.0
        try:
            xa = np.array(x, dtype=float)
            ya = np.array(y, dtype=float)
            # 简单 OLS: ya = alpha + beta * xa + epsilon
            n = len(xa)
            x_with_const = np.column_stack([np.ones(n), xa])
            try:
                beta, _, _, _ = np.linalg.lstsq(x_with_const, ya, rcond=None)
            except Exception:
                return 1.0
            residuals = ya - x_with_const @ beta
            # ADF 检验统计量 (简化)
            resid_diff = np.diff(residuals)
            resid_lag = residuals[:-1]
            if resid_lag.std() < 1e-8:
                return 1.0
            try:
                rho = np.corrcoef(resid_lag, resid_diff)[0, 1]
            except Exception:
                return 1.0
            # 近似 t-stat
            n_r = len(resid_lag)
            if abs(rho) >= 1.0:
                return 1.0
            t_stat = rho * math.sqrt((n_r - 2) / (1 - rho * rho))
            # 经验阈值 (Engle-Granger 1%: -3.90, 5%: -3.34, 10%: -3.04)
            if t_stat < -3.90:
                return 0.005
            elif t_stat < -3.34:
                return 0.025
            elif t_stat < -3.04:
                return 0.075
            elif t_stat < -2.76:
                return 0.15
            else:
                return 0.5
        except Exception:
            return 1.0

    def _update_cointegration(self) -> None:
        """更新协整检验"""
        window = min(self.COINTEGRATION_WINDOW, len(self.history))
        if window < 30:
            return
        # 用价格序列(非收益率)做协整
        btc_prices = [s.btc for s in list(self.history)[-window:] if s.btc > 0]
        gold_prices = [s.gold for s in list(self.history)[-window:] if s.gold > 0]
        silver_prices = [s.silver for s in list(self.history)[-window:] if s.silver > 0]
        oil_prices = [s.oil for s in list(self.history)[-window:] if s.oil > 0]

        n = min(len(btc_prices), len(gold_prices), len(silver_prices), len(oil_prices))
        if n < 30:
            return

        self.btc_gold_cointegration = self._ols_cointegration(
            btc_prices[-n:], gold_prices[-n:])
        self.btc_silver_cointegration = self._ols_cointegration(
            btc_prices[-n:], silver_prices[-n:])
        self.btc_oil_cointegration = self._ols_cointegration(
            btc_prices[-n:], oil_prices[-n:])

    def _update_zscores(self) -> None:
        """更新 BTC vs 商品价差的 Z-score"""
        window = min(self.COINTEGRATION_WINDOW, len(self.history))
        if window < 30:
            return

        recent = list(self.history)[-window:]
        btc_prices = np.array([s.btc for s in recent if s.btc > 0], dtype=float)
        gold_prices = np.array([s.gold for s in recent if s.gold > 0], dtype=float)
        silver_prices = np.array([s.silver for s in recent if s.silver > 0], dtype=float)
        oil_prices = np.array([s.oil for s in recent if s.oil > 0], dtype=float)

        # 用对数价比
        if len(btc_prices) >= 30 and len(gold_prices) >= 30:
            n = min(len(btc_prices), len(gold_prices))
            log_ratio = np.log(btc_prices[-n:] / gold_prices[-n:])
            mean = float(np.mean(log_ratio))
            std = float(np.std(log_ratio))
            if std > 1e-8:
                self.btc_gold_zscore = float((log_ratio[-1] - mean) / std)

        if len(btc_prices) >= 30 and len(silver_prices) >= 30:
            n = min(len(btc_prices), len(silver_prices))
            log_ratio = np.log(btc_prices[-n:] / silver_prices[-n:])
            mean = float(np.mean(log_ratio))
            std = float(np.std(log_ratio))
            if std > 1e-8:
                self.btc_silver_zscore = float((log_ratio[-1] - mean) / std)

        if len(btc_prices) >= 30 and len(oil_prices) >= 30:
            n = min(len(btc_prices), len(oil_prices))
            log_ratio = np.log(btc_prices[-n:] / oil_prices[-n:])
            mean = float(np.mean(log_ratio))
            std = float(np.std(log_ratio))
            if std > 1e-8:
                self.btc_oil_zscore = float((log_ratio[-1] - mean) / std)

    # ------------------------------------------------------------------
    # Regime 检测
    # ------------------------------------------------------------------

    def _detect_regime(self) -> None:
        """检测风险偏好状态"""
        if self.current_snapshot is None:
            return

        snap = self.current_snapshot
        prev_regime = self.current_regime

        # 计算 RISK_OFF 分数
        risk_off_signals = 0
        if len(self.btc_returns) > 0:
            btc_last_return = self.btc_returns[-1]
            gold_last_return = self.gold_returns[-1] if self.gold_returns else 0.0
            sp500_last_return = self.sp500_returns[-1] if self.sp500_returns else 0.0

            # BTC 跌 + Gold 涨 = 避险
            if (btc_last_return < self.RISK_OFF_BTC_DROP_MIN and
                gold_last_return > self.RISK_OFF_GOLD_RISE_MIN):
                risk_off_signals += 1

            # SP500 跌 + Gold 涨 = 避险
            if (sp500_last_return < self.RISK_OFF_BTC_DROP_MIN and
                gold_last_return > self.RISK_OFF_GOLD_RISE_MIN):
                risk_off_signals += 1

        # VIX 飙升 = 避险
        if snap.vix > self.VIX_PANIC_THRESHOLD:
            risk_off_signals += 1
        if snap.vix > self.VIX_EXTREME_THRESHOLD:
            risk_off_signals += 1

        # BTC-Gold 负相关 = 避险模式
        if self.btc_gold_corr < self.NEGATIVE_CORRELATION_THRESHOLD:
            risk_off_signals += 1

        self.risk_off_score = min(1.0, risk_off_signals / 4.0)

        # 计算 RISK_ON 分数
        risk_on_signals = 0
        if snap.vix < 18.0:
            risk_on_signals += 1
        if self.btc_sp500_corr > 0.4:
            risk_on_signals += 1
        if self.btc_gold_corr > 0.3:
            risk_on_signals += 1  # BTC+Gold 同涨 = 通胀对冲下的 RISK_ON

        # DXY 强势
        if snap.dxy > self.DXY_STRONG_THRESHOLD:
            self.dxy_strength = 1.0
        elif snap.dxy < self.DXY_WEAK_THRESHOLD:
            self.dxy_strength = -1.0
        else:
            self.dxy_strength = (snap.dxy - 100.0) / 4.0

        # 通胀对冲分数 (BTC+Gold 同涨 = 通胀对冲)
        if self.btc_gold_corr > self.INFLATION_HEDGE_CORR_MIN:
            self.inflation_hedge_score = min(1.0, self.btc_gold_corr)
        else:
            self.inflation_hedge_score = 0.0

        # 决定 Regime
        if self.risk_off_score >= 0.5:
            new_regime = RiskAppetiteRegime.RISK_OFF
        elif risk_on_signals >= 2 and self.risk_off_score < 0.25:
            new_regime = RiskAppetiteRegime.RISK_ON
        else:
            new_regime = RiskAppetiteRegime.NEUTRAL

        # 切换中状态
        if prev_regime != new_regime and prev_regime != RiskAppetiteRegime.TRANSITIONING:
            self.current_regime = RiskAppetiteRegime.TRANSITIONING
            self.regime_shift_count += 1
        else:
            self.current_regime = new_regime

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def _find_cointegration_opportunities(self) -> List[CrossMarketOpportunity]:
        """寻找协整套利机会"""
        opportunities = []

        pairs = [
            ('BTC', 'Gold', self.btc_gold_zscore, self.btc_gold_corr,
             self.btc_gold_cointegration),
            ('BTC', 'Silver', self.btc_silver_zscore, self.btc_silver_corr,
             self.btc_silver_cointegration),
            ('BTC', 'Oil', self.btc_oil_zscore, self.btc_oil_corr,
             self.btc_oil_cointegration),
        ]

        for primary, secondary, z, corr, p_val in pairs:
            if abs(z) < self.ZSCORE_ENTRY:
                continue
            if p_val > self.COINTEGRATION_P_VALUE_MAX:
                continue  # 不协整,跳过
            if abs(corr) < self.HIGH_CORRELATION_THRESHOLD:
                continue

            # Z-score > 2: BTC 相对商品高估 → 做空 BTC / 做多商品
            # Z-score < -2: BTC 相对商品低估 → 做多 BTC / 做空商品
            if z > self.ZSCORE_ENTRY:
                signal_type = CrossMarketSignalType.CROSS_MARKET_SELL
                direction = -1
            else:
                signal_type = CrossMarketSignalType.CROSS_MARKET_BUY
                direction = 1

            conviction = min(1.0, abs(z) / self.ZSCORE_STOP)
            # 协整越显著, conviction 越高
            conviction *= (1.0 - p_val)
            # 相关性越高, conviction 越高
            conviction *= min(1.0, abs(corr) / 0.8)

            opportunities.append(CrossMarketOpportunity(
                signal_type=signal_type,
                primary_asset=primary,
                secondary_asset=secondary,
                z_score=z,
                correlation=corr,
                conviction=conviction,
                expected_holding_period='medium',
                metadata={'direction': direction, 'p_value': p_val}
            ))

        return opportunities

    def _detect_correlation_break(self) -> Optional[CrossMarketOpportunity]:
        """检测相关性突变"""
        delta = self.btc_gold_corr - self.prev_btc_gold_corr
        if abs(delta) < self.CORRELATION_BREAK_THRESHOLD:
            return None

        return CrossMarketOpportunity(
            signal_type=CrossMarketSignalType.CORRELATION_BREAK,
            primary_asset='BTC',
            secondary_asset='Gold',
            z_score=0.0,
            correlation=self.btc_gold_corr,
            conviction=min(1.0, abs(delta) / 0.5),
            expected_holding_period='short',
            metadata={'delta': delta, 'prev_corr': self.prev_btc_gold_corr}
        )

    def _detect_dxy_warning(self) -> Optional[CrossMarketOpportunity]:
        """检测 DXY 预警"""
        if self.current_snapshot is None:
            return None

        if self.current_snapshot.dxy > self.DXY_STRONG_THRESHOLD:
            conviction = min(1.0, (self.current_snapshot.dxy -
                                   self.DXY_STRONG_THRESHOLD) / 5.0)
            return CrossMarketOpportunity(
                signal_type=CrossMarketSignalType.DXY_WARNING,
                primary_asset='DXY',
                secondary_asset='BTC',
                z_score=0.0,
                correlation=self.btc_dxy_corr,
                conviction=conviction,
                expected_holding_period='medium',
                metadata={'dxy_level': self.current_snapshot.dxy}
            )
        return None

    def _detect_inflation_hedge(self) -> Optional[CrossMarketOpportunity]:
        """检测通胀对冲开启"""
        if self.inflation_hedge_score < 0.5:
            return None

        return CrossMarketOpportunity(
            signal_type=CrossMarketSignalType.INFLATION_HEDGE,
            primary_asset='BTC',
            secondary_asset='Gold',
            z_score=0.0,
            correlation=self.btc_gold_corr,
            conviction=self.inflation_hedge_score,
            expected_holding_period='long',
            metadata={'score': self.inflation_hedge_score}
        )

    def _detect_risk_off_hedge(self) -> Optional[CrossMarketOpportunity]:
        """检测避险对冲信号"""
        if self.risk_off_score < 0.5:
            return None

        return CrossMarketOpportunity(
            signal_type=CrossMarketSignalType.RISK_OFF_HEDGE,
            primary_asset='Gold',
            secondary_asset='BTC',
            z_score=0.0,
            correlation=self.btc_gold_corr,
            conviction=self.risk_off_score,
            expected_holding_period='short',
            metadata={'vix': self.current_snapshot.vix if self.current_snapshot else 0.0}
        )

    # ------------------------------------------------------------------
    # 综合分析
    # ------------------------------------------------------------------

    def analyze(self) -> CrossMarketSignal:
        """综合分析并生成信号"""
        if self.current_snapshot is None:
            return CrossMarketSignal(
                signal_type=CrossMarketSignalType.NEUTRAL,
                direction=0,
                conviction=0.0,
                regime=RiskAppetiteRegime.NEUTRAL,
                btc_gold_corr=0.0,
                btc_dxy_corr=0.0,
                inflation_hedge_score=0.0,
                risk_off_score=0.0,
            )

        # 收集所有机会
        opportunities = []
        opportunities.extend(self._find_cointegration_opportunities())

        corr_break = self._detect_correlation_break()
        if corr_break:
            opportunities.append(corr_break)

        dxy_warn = self._detect_dxy_warning()
        if dxy_warn:
            opportunities.append(dxy_warn)

        inflation = self._detect_inflation_hedge()
        if inflation:
            opportunities.append(inflation)

        risk_off = self._detect_risk_off_hedge()
        if risk_off:
            opportunities.append(risk_off)

        # 决定主信号
        if not opportunities:
            signal_type = CrossMarketSignalType.NEUTRAL
            direction = 0
            conviction = 0.0
        else:
            # 取 conviction 最高的
            primary = max(opportunities, key=lambda o: o.conviction)
            signal_type = primary.signal_type

            # 决定方向
            if signal_type in [CrossMarketSignalType.CROSS_MARKET_BUY,
                               CrossMarketSignalType.INFLATION_HEDGE]:
                direction = 1
            elif signal_type in [CrossMarketSignalType.CROSS_MARKET_SELL,
                                 CrossMarketSignalType.DXY_WARNING,
                                 CrossMarketSignalType.RISK_OFF_HEDGE]:
                direction = -1
            elif signal_type == CrossMarketSignalType.CORRELATION_BREAK:
                # 相关性突变: 看变化方向
                delta = primary.metadata.get('delta', 0.0)
                direction = 1 if delta > 0 else -1
            else:
                direction = 0

            # 综合 conviction
            conviction = primary.conviction
            # 多个机会一致 → 加强 conviction
            if len(opportunities) >= 2:
                consistent = sum(1 for o in opportunities
                                 if (o.conviction > 0.3 and
                                     ((direction > 0 and o.signal_type in
                                       [CrossMarketSignalType.CROSS_MARKET_BUY,
                                        CrossMarketSignalType.INFLATION_HEDGE])
                                      or (direction < 0 and o.signal_type in
                                          [CrossMarketSignalType.CROSS_MARKET_SELL,
                                           CrossMarketSignalType.DXY_WARNING,
                                           CrossMarketSignalType.RISK_OFF_HEDGE]))))
                if consistent >= 2:
                    conviction = min(1.0, conviction * 1.2)

        # Regime shift 信号
        if self.current_regime == RiskAppetiteRegime.TRANSITIONING:
            signal_type = CrossMarketSignalType.REGIME_SHIFT
            conviction = max(conviction, 0.6)

        signal = CrossMarketSignal(
            signal_type=signal_type,
            direction=direction,
            conviction=conviction,
            regime=self.current_regime,
            btc_gold_corr=self.btc_gold_corr,
            btc_dxy_corr=self.btc_dxy_corr,
            inflation_hedge_score=self.inflation_hedge_score,
            risk_off_score=self.risk_off_score,
            opportunities=opportunities,
            timestamp=len(self.history),
        )

        self.last_signal = signal
        if signal_type != CrossMarketSignalType.NEUTRAL:
            self.signal_count += 1

        return signal

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'history_size': len(self.history),
            'current_regime': self.current_regime.value,
            'btc_gold_corr': self.btc_gold_corr,
            'btc_silver_corr': self.btc_silver_corr,
            'btc_oil_corr': self.btc_oil_corr,
            'btc_dxy_corr': self.btc_dxy_corr,
            'btc_sp500_corr': self.btc_sp500_corr,
            'gold_dxy_corr': self.gold_dxy_corr,
            'btc_gold_zscore': self.btc_gold_zscore,
            'btc_silver_zscore': self.btc_silver_zscore,
            'btc_oil_zscore': self.btc_oil_zscore,
            'btc_gold_cointegration_p': self.btc_gold_cointegration,
            'btc_silver_cointegration_p': self.btc_silver_cointegration,
            'btc_oil_cointegration_p': self.btc_oil_cointegration,
            'inflation_hedge_score': self.inflation_hedge_score,
            'risk_off_score': self.risk_off_score,
            'dxy_strength': self.dxy_strength,
            'signal_count': self.signal_count,
            'regime_shift_count': self.regime_shift_count,
        }
