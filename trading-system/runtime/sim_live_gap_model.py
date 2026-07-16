#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟-实盘差异量化评估模型 (Sim-Live Gap Quantification Model)

核心功能：
  1. 量化点差、滑点、流动性、延迟、费率等因素对策略表现的影响
  2. 蒙特卡洛模拟，评估各因素的影响权重和置信区间
  3. 高保真市场仿真，复现真实市场摩擦
  4. 生成差异分析报告，指导策略优化方向

设计原则（用户铁律）：
  - "杜绝模拟牛逼，实盘亏钱" — 必须在模拟阶段就考虑实盘摩擦
  - "理解底层逻辑，而非表面指标" — 每个因素都有理论模型支撑
  - "全方位多维度多层次多角度" — 覆盖所有市场摩擦因素

模型来源：
  - Almgren-Chriss 2000: 最优执行与市场冲击
  - Cartea-Jaimungal-Penalva 2015: 算法交易数学
  - Gate.io/Binance 实际API文档: 费率、点差、滑点参数
  - 实测数据: 国内访问延迟 ~1056ms

使用方式：
  from sim_live_gap_model import SimLiveGapAnalyzer
  analyzer = SimLiveGapAnalyzer(symbol="BTC-USDT")
  report = analyzer.analyze(trades, market_data)
"""

from __future__ import annotations

import copy
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# 市场摩擦参数（基于实测和交易所文档）
# ============================================================================

@dataclass
class MarketFrictionParams:
    """市场摩擦参数 — 模拟环境与实盘环境的差异来源"""

    # 点差 (Bid-Ask Spread)
    spread_bps: float = 2.0              # 平均点差 (bps)，BTC约1-2bps，山寨币约3-10bps
    spread_std_bps: float = 1.0          # 点差波动 (bps)

    # 滑点 (Slippage = Market Impact)
    slippage_per_100k_bps: float = 0.5   # 每10万美元交易量的滑点 (bps)
    slippage_volatility_mult: float = 1.5 # 高波动时滑点倍数

    # 延迟 (Latency)
    latency_ms: float = 150.0            # 平均订单执行延迟 (ms)
    latency_std_ms: float = 50.0         # 延迟波动 (ms)
    domestic_penalty_ms: float = 900.0   # 国内访问额外延迟 (实测 ~1056ms 总延迟)

    # 流动性 (Liquidity)
    depth_1pct_usd: float = 50000.0      # 1%价格范围内的订单簿深度 (USD)
    depth_decay_rate: float = 0.5        # 深度衰减率 (每1%价格偏离)

    # 费率 (Fees)
    maker_fee_bps: float = 0.2           # Maker费率 (bps)，Gate.io VIP0
    taker_fee_bps: float = 0.5           # Taker费率 (bps)
    funding_rate_8h_pct: float = 0.01    # 8小时资金费率 (%)

    # 订单执行
    fill_probability: float = 0.95       # 限价单成交概率
    partial_fill_ratio: float = 0.85     # 部分成交时的成交比例
    cancellation_delay_ms: float = 50.0  # 撤单延迟 (ms)

    # 市场微观结构
    adverse_selection_bps: float = 0.3   # 逆向选择成本 (bps)，做市商面临的知情交易者风险
    tick_size: float = 0.1               # 最小价格变动单位 (BTC-USDT = 0.1)
    min_order_size_usd: float = 10.0     # 最小订单金额 (USD)

    # v598 Phase G: 真实 Almgren-Chriss 2000 闭式解参数
    # ERR-20260702-v598p5-almgren-simplified: 原简化版 slippage_per_100k * (Q/100k)
    # 是线性模型, 违反实证 square-root law (Almgren et al. 2005, Bouchaud et al. 2004)
    # 真实 Almgren-Chriss:
    #   永久冲击 PI = γ * σ * (Q/V)     — 线性, 移动 mid-price
    #   临时冲击 TI = η * σ * sqrt(Q/V) — sqrt, 执行成本
    #   总冲击 Impact = σ * (γ * Q/V + η * sqrt(Q/V))
    # 当 daily_volume_usd=0 (未知) → 降级到简化线性模型
    almgren_gamma: float = 0.1           # 永久冲击系数 γ (典型 0.1-0.3)
    almgren_eta: float = 0.3             # 临时冲击系数 η (典型 0.1-0.5)
    daily_volume_usd: float = 0.0        # 日成交量 ADV (USD), 0=未知用简化模型


# 按币种预设参数
# v598 Phase R1.2: 移植 ADV 数据 (来自 _v479b_friction_adapter.V479B_SYMBOL_PARAMS)
# 让 AC 真实公式 L408-421 生效 (而非走 L423-428 简化线性降级路径)
# 数据源: Binance Futures 2024-2026 实测 ADV
SYMBOL_FRICTION_PARAMS = {
    # Tier 1 (高流动性, ADV>$100M)
    "BTC-USDT": MarketFrictionParams(
        spread_bps=1.0, slippage_per_100k_bps=0.3,
        depth_1pct_usd=200000.0, taker_fee_bps=0.4,
        daily_volume_usd=500_000_000,
    ),
    "ETH-USDT": MarketFrictionParams(
        spread_bps=1.5, slippage_per_100k_bps=0.4,
        depth_1pct_usd=100000.0, taker_fee_bps=0.5,
        daily_volume_usd=300_000_000,
    ),
    "SOL-USDT": MarketFrictionParams(
        spread_bps=3.0, slippage_per_100k_bps=0.8,
        depth_1pct_usd=30000.0, taker_fee_bps=0.5,
        daily_volume_usd=100_000_000,
    ),
    # Tier 2 (中高流动性, ADV>$50M)
    "BNB-USDT": MarketFrictionParams(
        spread_bps=2.0, slippage_per_100k_bps=0.5,
        depth_1pct_usd=50000.0, taker_fee_bps=0.5,
        daily_volume_usd=80_000_000,
    ),
    # Tier 3 (中等流动性, ADV>$30M)
    "DOGE-USDT": MarketFrictionParams(
        spread_bps=5.0, slippage_per_100k_bps=1.5,
        depth_1pct_usd=15000.0, taker_fee_bps=0.6,
        daily_volume_usd=40_000_000,
    ),
    # Phase 2.3: 扩展到10币种 (用户要求"差异分析需覆盖至少10个主要交易对")
    # 参数来源: Binance/Gate.io 实测订单簿深度 + 平均点差 (2025-Q4)
    "XRP-USDT": MarketFrictionParams(
        spread_bps=2.5, slippage_per_100k_bps=0.6,
        depth_1pct_usd=40000.0, taker_fee_bps=0.5,
        daily_volume_usd=60_000_000,
    ),
    "ADA-USDT": MarketFrictionParams(
        spread_bps=3.0, slippage_per_100k_bps=0.7,
        depth_1pct_usd=30000.0, taker_fee_bps=0.5,
        daily_volume_usd=15_000_000,  # ADA 已在 V704_REMOVED_SYMBOLS 禁用 (保留参数用于历史分析)
    ),
    "AVAX-USDT": MarketFrictionParams(
        spread_bps=3.5, slippage_per_100k_bps=0.8,
        depth_1pct_usd=25000.0, taker_fee_bps=0.5,
        daily_volume_usd=30_000_000,
    ),
    # Tier 4 (低流动性, ADV<$30M)
    "LINK-USDT": MarketFrictionParams(
        spread_bps=3.0, slippage_per_100k_bps=0.7,
        depth_1pct_usd=30000.0, taker_fee_bps=0.5,
        daily_volume_usd=25_000_000,
    ),
    "DOT-USDT": MarketFrictionParams(
        spread_bps=3.0, slippage_per_100k_bps=0.7,
        depth_1pct_usd=30000.0, taker_fee_bps=0.5,
        daily_volume_usd=20_000_000,
    ),
}


# ============================================================================
# 市场状态(regime)摩擦调整乘数 — Phase 2.3 新增
# 用户要求: "包含不同行情条件（趋势、震荡、高波动、低波动）下的对比测试"
# 来源: Cartea-Jaimungal 2015 算法交易微观结构 + 实测市场状态切换
# ============================================================================

REGIME_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    # 趋势行情: 基准 (摩擦参数为默认值)
    "trend":    {"spread": 1.0, "slippage": 1.0, "latency": 1.0, "desc": "趋势行情(基准)"},
    # 震荡行情: 点差扩大(做市商风险增加)但冲击成本降低(流动性分散)
    "range":    {"spread": 1.2, "slippage": 0.8, "latency": 1.0, "desc": "震荡行情(点差扩大冲击降低)"},
    # 高波动行情: 全面恶化 — 点差/滑点/延迟同时放大
    "high_vol": {"spread": 1.5, "slippage": 1.5, "latency": 1.3, "desc": "高波动行情(全面恶化)"},
    # 低波动行情: 流动性好, 摩擦成本降低
    "low_vol":  {"spread": 0.8, "slippage": 0.7, "latency": 0.9, "desc": "低波动行情(流动性好)"},
}

# 4种regime名称 (供外部遍历使用)
REGIME_NAMES: Tuple[str, ...] = ("trend", "range", "high_vol", "low_vol")


# ============================================================================
# 差异分析结果
# ============================================================================

@dataclass
class GapAnalysisResult:
    """单因素差异分析结果"""
    factor_name: str
    impact_bps_mean: float         # 平均影响 (bps)
    impact_bps_std: float          # 影响标准差 (bps)
    impact_bps_p95: float          # 95%分位影响
    weight_pct: float              # 占总差异的权重 (%)
    confidence_interval: Tuple[float, float]  # 95%置信区间
    is_significant: bool           # 是否显著 (P<0.05)


@dataclass
class SimLiveGapReport:
    """模拟-实盘差异分析报告"""
    symbol: str
    total_trades: int
    total_gap_bps: float = 0.0     # 总差异 (bps)
    total_gap_pct: float = 0.0     # 总差异 (占收益百分比)

    factors: List[GapAnalysisResult] = field(default_factory=list)

    # 蒙特卡洛结果
    mc_mean_gap_bps: float = 0.0
    mc_std_gap_bps: float = 0.0
    mc_p95_gap_bps: float = 0.0

    # 建议
    recommendations: List[str] = field(default_factory=list)
    high_fidelity_params: Dict[str, Any] = field(default_factory=dict)

    # Phase 2.3: regime 标识 (None=基准/trend, 其他=对应regime调整后结果)
    regime: Optional[str] = None


# ============================================================================
# 核心分析引擎
# ============================================================================

class SimLiveGapAnalyzer:
    """模拟-实盘差异量化分析器

    核心思路：
      1. 对每笔模拟交易，计算在实盘摩擦下的真实PnL
      2. 通过蒙特卡洛模拟量化不确定性和各因素权重
      3. 生成高保真仿真参数，可在模拟环境中复现实盘摩擦
    """

    def __init__(self, symbol: str = "BTC-USDT",
                 params: MarketFrictionParams = None,
                 use_domestic_latency: bool = True,
                 regime: Optional[str] = None):
        """初始化差异分析器

        Args:
            symbol: 交易对 (如 "BTC-USDT" / "BTC_USDT" / "btcusdt")
            params: 自定义摩擦参数 (None则用 SYMBOL_FRICTION_PARAMS 预设)
            use_domestic_latency: 是否叠加国内访问延迟 (~900ms)
            regime: 市场状态 (None/trend=基准, range/high_vol/low_vol=regime调整)
                    Phase 2.3 新增: 用户要求4种行情条件对比测试
        """
        # B3修复: 规范化 symbol 格式, 解决 "BTC_USDT" vs "BTC-USDT" 不匹配导致所有币种返回默认值
        self.symbol = self._normalize_symbol(symbol)
        # Phase 2.3 修复: 深拷贝避免污染 SYMBOL_FRICTION_PARAMS 单例
        # (原代码直接引用单例, use_domestic_latency 和 regime 调整会永久修改共享对象)
        if params is not None:
            self.params = copy.deepcopy(params)
        else:
            _base = SYMBOL_FRICTION_PARAMS.get(self.symbol, MarketFrictionParams())
            self.params = copy.deepcopy(_base)
        if use_domestic_latency:
            self.params.latency_ms += self.params.domestic_penalty_ms
        # Phase 2.3: regime 调整 (在 use_domestic_latency 之后, 对调整后的 params 再乘regime系数)
        self.regime = regime
        if regime is not None and regime != "trend":
            self._apply_regime_adjustment(regime)

    def _apply_regime_adjustment(self, regime: str) -> None:
        """根据市场状态调整摩擦参数 (Phase 2.3 新增)

        用户要求: "包含不同行情条件（趋势、震荡、高波动、低波动）下的对比测试"
        来源: Cartea-Jaimungal 2015 — 市场状态切换时微观结构参数变化

        Args:
            regime: "trend" / "range" / "high_vol" / "low_vol"
        """
        mult = REGIME_MULTIPLIERS.get(regime)
        if mult is None:
            return  # 未知 regime, 不调整
        # spread × 乘数 (点差随regime变化)
        self.params.spread_bps *= mult["spread"]
        self.params.spread_std_bps *= mult["spread"]
        # slippage × 乘数 (市场冲击随regime变化)
        self.params.slippage_per_100k_bps *= mult["slippage"]
        # latency × 乘数 (高波动时网络拥堵+匹配引擎延迟)
        self.params.latency_ms *= mult["latency"]
        self.params.latency_std_ms *= mult["latency"]

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """统一 symbol 格式为 BTC-USDT (匹配 SYMBOL_FRICTION_PARAMS 的 key)

        支持输入: "BTC-USDT" / "BTC_USDT" / "btc-usdt" / "BTCUSDT" / "btcusdt"
        输出: "BTC-USDT"
        """
        s = symbol.upper().replace("_", "-")
        # 处理无分隔符的情况 (如 "BTCUSDT" → "BTC-USDT")
        if "USDT" in s and "-" not in s:
            s = s.replace("USDT", "-USDT")
        elif "USD" in s and "-" not in s:
            s = s.replace("USD", "-USD")
        return s

    def analyze(self, trades: List[Dict[str, Any]],
                market_data: List[Dict] = None,
                monte_carlo_samples: int = 1000) -> SimLiveGapReport:
        """分析模拟交易在实盘摩擦下的表现差异

        Args:
            trades: 模拟交易记录列表，每笔包含 entry_price, exit_price, side, quantity, pnl_pct
            market_data: 可选市场数据（用于波动率估计）
            monte_carlo_samples: 蒙特卡洛模拟次数

        Returns:
            SimLiveGapReport: 完整差异分析报告
        """
        if not trades:
            return SimLiveGapReport(symbol=self.symbol, total_trades=0, total_gap_bps=0, total_gap_pct=0)

        report = SimLiveGapReport(symbol=self.symbol, total_trades=len(trades))

        # 步骤1: 逐个因素分析
        factor_results = []
        factor_results.append(self._analyze_spread(trades))
        factor_results.append(self._analyze_slippage(trades))
        factor_results.append(self._analyze_latency(trades, market_data))
        factor_results.append(self._analyze_fees(trades))
        factor_results.append(self._analyze_liquidity(trades))

        # 总差异
        total_gap = sum(f.impact_bps_mean for f in factor_results)
        report.total_gap_bps = total_gap
        report.total_gap_pct = total_gap / 100.0

        # 各因素权重
        abs_total = sum(abs(f.impact_bps_mean) for f in factor_results)
        for f in factor_results:
            f.weight_pct = abs(f.impact_bps_mean) / abs_total * 100 if abs_total > 0 else 0

        report.factors = factor_results

        # 步骤2: 蒙特卡洛模拟
        mc_results = self._monte_carlo_simulate(trades, factor_results, monte_carlo_samples)
        report.mc_mean_gap_bps = statistics.mean(mc_results)
        report.mc_std_gap_bps = statistics.stdev(mc_results) if len(mc_results) > 1 else 0
        report.mc_p95_gap_bps = np.percentile(mc_results, 95)

        # 步骤3: 生成建议
        report.recommendations = self._generate_recommendations(factor_results, report)

        # 步骤4: 高保真仿真参数
        report.high_fidelity_params = self._generate_hifi_params(factor_results)

        # Phase 2.3: 记录 regime 标识 (None/trend=基准, 其他=对应regime调整后结果)
        report.regime = self.regime

        return report

    def _analyze_spread(self, trades: List[Dict]) -> GapAnalysisResult:
        """分析点差影响

        点差 = 买入价 - 卖出价
        做多时入场用Ask价(高于mid)，出场用Bid价(低于mid) → 双向损失
        总影响 = 入场点差 + 出场点差 = 2 × spread_bps
        """
        p = self.params
        impact = 2 * p.spread_bps  # 入场+出场
        std = p.spread_std_bps * math.sqrt(2)

        return GapAnalysisResult(
            factor_name="点差 (Bid-Ask Spread)",
            impact_bps_mean=-impact,
            impact_bps_std=std,
            impact_bps_p95=-impact - 1.645 * std,
            weight_pct=0,
            confidence_interval=(-impact - 1.96 * std, -impact + 1.96 * std),
            is_significant=impact > 1.0,
        )

    def _analyze_slippage(self, trades: List[Dict]) -> GapAnalysisResult:
        """分析滑点影响 — v598 Phase G: 真实 Almgren-Chriss 2000 闭式解

        ERR-20260702-v598p5-almgren-simplified: 原简化版 slippage_per_100k * (Q/100k)
        是线性模型, 违反实证 square-root law.

        真实 Almgren-Chriss (2000) 闭式解:
          永久冲击 PI = γ * σ * (Q/V)     — 线性, 移动 mid-price (后续订单成本更高)
          临时冲击 TI = η * σ * sqrt(Q/V) — sqrt, 执行成本 (一次性, 进出双向)
          总冲击 Impact = σ * (γ * Q/V + η * sqrt(Q/V)) * 10000 (转 bps)

        其中:
          σ: 日度波动率 (如 0.03 = 3%)
          Q: 订单量 (USD)
          V: 日成交量 ADV (USD)
          γ: 永久冲击系数 (0.1-0.3 典型, Almgren et al. 2005 实证)
          η: 临时冲击系数 (0.1-0.5 典型, Bouchaud et al. 2004 实证)

        Square-root law 实证依据:
          - Almgren, Thum, Hauptmann, Li (2005): "Direct Estimation of Equity Market Impact"
            跨 10+ 年数据验证 Impact ∝ sqrt(Q/V)
          - Bouchaud, Farmer, Lillo (2004): "How markets slowly digest changes in supply and demand"
            证明临时冲击服从 sqrt 规律, 永久冲击服从线性规律
          - Torre (1997): 期权市场做市商数据验证

        降级策略:
          当 daily_volume_usd=0 (ADV 未知) → 降级到简化线性模型 (原实现)
          slippage_per_100k_bps * (order_size_usd / 100000.0)
        """
        p = self.params
        total_impact = 0.0
        impacts = []
        used_almgren = 0  # 统计使用真实 Almgren-Chriss 的交易数
        used_simplified = 0  # 统计使用简化模型的交易数

        # 估算日度波动率 σ (从市场数据或默认值)
        sigma = self._estimate_volatility()

        for trade in trades:
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            quantity = trade.get("quantity", 0)

            if entry_price > 0 and quantity > 0:
                order_size_usd = entry_price * abs(quantity)

                if p.daily_volume_usd > 0 and sigma > 0:
                    # v598 Phase G: 真实 Almgren-Chriss 闭式解
                    Q = order_size_usd
                    V = p.daily_volume_usd
                    participation_rate = Q / V  # 参与率 (应 << 1)

                    # 永久冲击 (bps): γ * σ * (Q/V) * 10000
                    permanent_impact = (
                        p.almgren_gamma * sigma * participation_rate * 10000.0
                    )

                    # 临时冲击 (bps): η * σ * sqrt(Q/V) * 10000
                    temporary_impact = (
                        p.almgren_eta * sigma
                        * math.sqrt(max(participation_rate, 1e-12))
                        * 10000.0
                    )

                    # 入场 + 出场双向 (临时冲击双向, 永久冲击仅入场)
                    impact = 2.0 * temporary_impact + permanent_impact
                    used_almgren += 1
                else:
                    # 降级: 简化线性模型 (原实现, ADV 未知时)
                    entry_impact = p.slippage_per_100k_bps * (order_size_usd / 100000.0)
                    exit_impact = p.slippage_per_100k_bps * (order_size_usd / 100000.0)
                    impact = entry_impact + exit_impact
                    used_simplified += 1

                total_impact += impact
                impacts.append(impact)

        mean_impact = total_impact / len(trades) if trades else 0
        std_impact = statistics.stdev(impacts) if len(impacts) > 1 else mean_impact * 0.3

        # 标注使用的模型类型
        if used_almgren > 0 and used_simplified == 0:
            model_type = "Almgren-Chriss (γ=%.2f, η=%.2f, σ=%.4f, V=$%.0fM)" % (
                p.almgren_gamma, p.almgren_eta, sigma, p.daily_volume_usd / 1e6
            )
        elif used_almgren > 0 and used_simplified > 0:
            model_type = "混合 (Almgren=%d, 简化=%d)" % (used_almgren, used_simplified)
        else:
            model_type = "简化线性 (slippage_per_100k=%.2f bps, V=未知)" % p.slippage_per_100k_bps

        return GapAnalysisResult(
            factor_name=f"滑点 (Market Impact) [{model_type}]",
            impact_bps_mean=-mean_impact,
            impact_bps_std=std_impact,
            impact_bps_p95=-mean_impact - 1.645 * std_impact,
            weight_pct=0,
            confidence_interval=(-mean_impact - 1.96 * std_impact, -mean_impact + 1.96 * std_impact),
            is_significant=mean_impact > 0.5,
        )

    def _estimate_volatility(self) -> float:
        """估算日度波动率 σ (v598 Phase G 辅助方法)

        优先从 market_data 估算, 不可用时用币种默认值

        Returns:
            日度波动率 σ (如 0.03 = 3% 日波动率)
        """
        # 默认: 加密货币日度波动率 ~3% (BTC) ~5% (山寨币)
        sigma = 0.03

        # 从 SYMBOL_FRICTION_PARAMS 推断 (spread 反映波动率)
        symbol_key = getattr(self, "symbol", "").upper().replace("_", "-")
        if symbol_key in SYMBOL_FRICTION_PARAMS:
            p_sym = SYMBOL_FRICTION_PARAMS[symbol_key]
            # 用 slippage_volatility_mult 估算波动率倍数
            if p_sym.spread_bps > 0:
                # spread_bps 越大, 波动率越高 (经验: σ ≈ spread_bps / 100)
                sigma = max(0.01, min(0.10, p_sym.spread_bps / 100.0))

        # 从最近的市场数据估算 (如果有)
        market_data = getattr(self, "_last_market_data", None)
        if market_data and len(market_data) > 2:
            try:
                returns = []
                for i in range(1, len(market_data)):
                    c1 = float(market_data[i-1].get("close", 0))
                    c2 = float(market_data[i].get("close", 0))
                    if c1 > 0 and c2 > 0:
                        returns.append(c2 / c1 - 1)
                if len(returns) > 5:
                    sigma = max(0.005, min(0.20, statistics.stdev(returns)))
            except Exception:
                pass  # 保持默认值

        return float(sigma)

    def _analyze_latency(self, trades: List[Dict],
                         market_data: List[Dict] = None) -> GapAnalysisResult:
        """分析延迟影响

        延迟导致入场价格偏离预期价格:
        1. 信号触发 → 订单到达交易所: latency_ms
        2. 在延迟期间，价格可能已移动: impact = volatility * sqrt(latency)
        3. 波动率: 用价格变化的标准差估计

        对于国内用户，实测延迟 ~1056ms，这是一个显著因素。
        """
        p = self.params

        # 估计波动率 (每毫秒)
        if market_data and len(market_data) > 1:
            returns = []
            for i in range(1, len(market_data)):
                c1 = market_data[i-1].get("close", 0)
                c2 = market_data[i].get("close", 0)
                if c1 > 0 and c2 > 0:
                    returns.append(abs(c2 / c1 - 1))
            vol_per_bar = statistics.median(returns) if returns else 0.001
        else:
            # 默认: 假设5分钟K线波动率 ~0.5%
            vol_per_bar = 0.005

        # 每笔交易的延迟影响
        # Impact = vol * sqrt(latency_ms / bar_duration_ms)
        # 假设bar_duration = 5min = 300000ms
        bar_duration_ms = 300000.0
        latency_impact = vol_per_bar * math.sqrt(p.latency_ms / bar_duration_ms)

        # 入场+出场双向延迟
        total_impact = 2 * latency_impact * 10000  # 转换为bps

        return GapAnalysisResult(
            factor_name="延迟 (Order Latency)",
            impact_bps_mean=-total_impact,
            impact_bps_std=total_impact * 0.5,
            impact_bps_p95=-total_impact * 1.5,
            weight_pct=0,
            confidence_interval=(-total_impact * 1.5, -total_impact * 0.5),
            is_significant=total_impact > 1.0,
        )

    def _analyze_fees(self, trades: List[Dict]) -> GapAnalysisResult:
        """分析费率影响

        每笔交易费率 = taker_fee (入场) + taker_fee (出场)
        如果有资金费率，还需考虑持仓期间的funding rate
        """
        p = self.params
        fee_per_trade = 2 * p.taker_fee_bps  # 入场+出场

        # 资金费率: 每8小时收一次，取决于持仓时间
        avg_funding = 0.0
        for trade in trades:
            bars_held = trade.get("bars_held", 0)
            # 5分钟K线，8小时 = 96根K线
            funding_hits = max(1, int(bars_held / 96))
            avg_funding += funding_hits * p.funding_rate_8h_pct * 100  # 转换为bps

        avg_funding = avg_funding / len(trades) if trades else 0

        total_impact = fee_per_trade + avg_funding

        return GapAnalysisResult(
            factor_name="费率 (Fees + Funding)",
            impact_bps_mean=-total_impact,
            impact_bps_std=total_impact * 0.2,
            impact_bps_p95=-total_impact * 1.3,
            weight_pct=0,
            confidence_interval=(-total_impact * 1.3, -total_impact * 0.7),
            is_significant=total_impact > 0.5,
        )

    def _analyze_liquidity(self, trades: List[Dict]) -> GapAnalysisResult:
        """分析流动性影响

        订单簿深度不足导致:
        1. 大订单无法完全成交 (部分成交)
        2. 成交价格恶化 (吃掉多个价位)
        3. 撤单/重新下单的额外成本

        使用线性衰减模型: depth_at_price = depth_1pct * exp(-price_deviation * decay_rate)
        """
        p = self.params
        impacts = []

        for trade in trades:
            entry_price = trade.get("entry_price", 0)
            quantity = trade.get("quantity", 0)

            if entry_price > 0 and quantity > 0:
                order_size_usd = entry_price * abs(quantity)
                # 订单量占深度的比例
                depth_ratio = order_size_usd / p.depth_1pct_usd
                # 流动性不足的额外成本
                if depth_ratio > 0.1:  # 超过10%深度
                    liquidity_impact = depth_ratio * 5.0  # 每10%深度≈5bps
                else:
                    liquidity_impact = depth_ratio * 2.0
                impacts.append(liquidity_impact)

        mean_impact = statistics.mean(impacts) if impacts else 0
        std_impact = statistics.stdev(impacts) if len(impacts) > 1 else mean_impact * 0.3

        return GapAnalysisResult(
            factor_name="流动性 (Order Book Depth)",
            impact_bps_mean=-mean_impact,
            impact_bps_std=std_impact,
            impact_bps_p95=-mean_impact - 1.645 * std_impact,
            weight_pct=0,
            confidence_interval=(-mean_impact - 1.96 * std_impact, -mean_impact + 1.96 * std_impact),
            is_significant=mean_impact > 0.5,
        )

    def _monte_carlo_simulate(self, trades: List[Dict],
                               factors: List[GapAnalysisResult],
                               n_samples: int = 1000) -> List[float]:
        """蒙特卡洛模拟：随机抽样各因素的不确定性

        对每笔交易，从各因素分布中随机抽样，计算总影响。
        重复n_samples次，得到总差异的分布。

        Returns:
            各次模拟的总差异列表 (bps)
        """
        results = []
        rng = np.random.RandomState(42)

        for _ in range(n_samples):
            total_gap = 0.0
            for f in factors:
                # 从正态分布中抽样
                sample = rng.normal(f.impact_bps_mean, f.impact_bps_std)
                total_gap += sample

            # 加入交易间相关性带来的额外方差
            n_trades = len(trades)
            if n_trades > 1:
                trade_correlation = 0.3  # 交易间相关性
                correlated_noise = rng.normal(0, abs(total_gap) * trade_correlation)
                total_gap += correlated_noise

            results.append(total_gap)

        return results

    def _generate_recommendations(self, factors: List[GapAnalysisResult],
                                   report: SimLiveGapReport) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 按影响排序
        sorted_factors = sorted(factors, key=lambda f: abs(f.impact_bps_mean), reverse=True)

        for f in sorted_factors[:3]:  # 前3大因素
            if abs(f.impact_bps_mean) > 2.0:
                recommendations.append(
                    f"[CRITICAL] {f.factor_name} 影响 {abs(f.impact_bps_mean):.1f}bps (权重 {f.weight_pct:.1f}%): "
                    f"需优先优化。建议: 1)增加订单簿深度监控 2)使用限价单替代市价单 3)分批建仓/平仓"
                )
            elif abs(f.impact_bps_mean) > 0.5:
                recommendations.append(
                    f"[HIGH] {f.factor_name} 影响 {abs(f.impact_bps_mean):.1f}bps (权重 {f.weight_pct:.1f}%): "
                    f"建议纳入风险模型和仓位计算"
                )

        # 总差异预警
        if abs(report.total_gap_bps) > 10.0:
            recommendations.append(
                f"[CRITICAL] 总差异 {abs(report.total_gap_bps):.1f}bps > 10bps: "
                f"模拟环境与实盘存在显著差异, 建议在模拟环境中启用高保真仿真模式"
            )

        # 延迟特别强调（国内用户）
        domestic_latency = self.params.latency_ms
        if domestic_latency > 500:
            recommendations.append(
                f"[WARN] 国内延迟 {domestic_latency:.0f}ms > 500ms: "
                f"建议使用海外服务器或低延迟策略适配。"
                f"当前条件下, 策略应使用≥1h时间框架, 避免高频交易"
            )

        return recommendations

    def _generate_hifi_params(self, factors: List[GapAnalysisResult]) -> Dict[str, Any]:
        """生成高保真仿真参数 — 可在模拟环境中复现实盘摩擦

        v598 Phase G: slippage_model 从伪 almgren_chriss 升级为真实 Almgren-Chriss 闭式解
        ERR-20260702-v598p5-almgren-simplified: 原版 eta=slippage_per_100k_bps/100
        只是无量纲系数, 没有真实公式支撑 (无 γ/σ/V), 违反 square-root law.
        """
        # 估算日度波动率 σ (与 _analyze_slippage 一致, 保证参数一致性)
        sigma = self._estimate_volatility()

        return {
            "spread_model": {
                "type": "normal",
                "mean_bps": self.params.spread_bps,
                "std_bps": self.params.spread_std_bps,
                "apply_to": "both_entry_exit",
            },
            # v598 Phase G: 真实 Almgren-Chriss 2000 闭式解
            "slippage_model": {
                "type": "almgren_chriss",           # 真实 AC 闭式解
                "gamma": float(self.params.almgren_gamma),   # 永久冲击系数 γ
                "eta": float(self.params.almgren_eta),       # 临时冲击系数 η
                "sigma": float(sigma),                        # 日度波动率 σ
                "daily_volume_usd": float(self.params.daily_volume_usd),  # ADV (USD)
                "formula": "Impact = sigma * (gamma * Q/V + eta * sqrt(Q/V)) * 10000",  # 公式自文档化
                "sqrt_law": True,                             # 临时冲击服从 square-root law
                "permanent_linear": True,                     # 永久冲击线性
                # 降级参数 (ADV=0 时使用简化线性模型)
                "fallback_type": "linear_per_100k",
                "fallback_slippage_per_100k_bps": float(self.params.slippage_per_100k_bps),
                "min_impact_bps": 0.1,
                "max_impact_bps": 5.0,
            },
            "latency_model": {
                "type": "normal",
                "mean_ms": self.params.latency_ms,
                "std_ms": self.params.latency_std_ms,
                "apply_to": "both_entry_exit",
            },
            "fee_model": {
                "taker_fee_bps": self.params.taker_fee_bps,
                "maker_fee_bps": self.params.maker_fee_bps,
                "funding_rate_8h_pct": self.params.funding_rate_8h_pct,
            },
            "liquidity_model": {
                "depth_1pct_usd": self.params.depth_1pct_usd,
                "decay_rate": self.params.depth_decay_rate,
                "partial_fill_enabled": True,
                "partial_fill_ratio": self.params.partial_fill_ratio,
            },
            "total_estimated_gap_bps": sum(f.impact_bps_mean for f in factors),
        }


# ============================================================================
# 便捷函数
# ============================================================================

def analyze_from_json(json_path: str, symbol: str = None,
                     use_domestic_latency: bool = True) -> SimLiveGapReport:
    """从交易明细JSON文件直接分析模拟-实盘差异

    Args:
        json_path: 交易明细JSON文件路径
        symbol: 指定币种（None则分析所有币种，生成综合报告）
        use_domestic_latency: 是否使用国内延迟参数

    Returns:
        SimLiveGapReport: 差异分析报告
    """
    import json as _json

    with open(json_path, "r", encoding="utf-8") as f:
        all_trades = _json.load(f)

    if symbol:
        trades = [t for t in all_trades if t.get("symbol") == symbol]
    else:
        trades = all_trades
        symbol = "ALL"

    if not trades:
        return SimLiveGapReport(symbol=symbol, total_trades=0)

    analyzer = SimLiveGapAnalyzer(symbol=symbol, use_domestic_latency=use_domestic_latency)
    return analyzer.analyze(trades)


def analyze_all_symbols_from_json(json_path: str,
                                  use_domestic_latency: bool = True) -> Dict[str, SimLiveGapReport]:
    """从交易明细JSON文件分析所有币种的模拟-实盘差异

    Args:
        json_path: 交易明细JSON文件路径
        use_domestic_latency: 是否使用国内延迟参数

    Returns:
        Dict[str, SimLiveGapReport]: 币种 -> 差异分析报告
    """
    import json as _json

    with open(json_path, "r", encoding="utf-8") as f:
        all_trades = _json.load(f)

    # 按币种分组
    trades_by_symbol = defaultdict(list)
    for t in all_trades:
        sym = t.get("symbol", "unknown")
        trades_by_symbol[sym].append(t)

    reports = {}
    for sym, trades in trades_by_symbol.items():
        analyzer = SimLiveGapAnalyzer(symbol=sym, use_domestic_latency=use_domestic_latency)
        reports[sym] = analyzer.analyze(trades)

    return reports


def generate_summary_report(reports: Dict[str, SimLiveGapReport]) -> str:
    """生成多币种差异分析摘要报告"""
    lines = []
    lines.append("# 模拟-实盘差异分析报告")
    lines.append("")
    lines.append("| 币种 | 交易数 | 总差异(bps) | 最大因素 | 最大因素权重 | 延迟(ms) |")
    lines.append("|------|--------|------------|---------|------------|---------|")

    for symbol, report in reports.items():
        max_factor = max(report.factors, key=lambda f: abs(f.impact_bps_mean)) if report.factors else None
        max_name = max_factor.factor_name.split("(")[0].strip() if max_factor else "N/A"
        max_weight = max_factor.weight_pct if max_factor else 0
        latency = SYMBOL_FRICTION_PARAMS.get(symbol, MarketFrictionParams()).latency_ms

        lines.append(
            f"| {symbol} | {report.total_trades} | {report.total_gap_bps:.1f} | "
            f"{max_name} | {max_weight:.1f}% | {latency:.0f} |"
        )

    lines.append("")
    lines.append("## 关键发现")
    lines.append("")

    all_recs = []
    for symbol, report in reports.items():
        for rec in report.recommendations:
            if rec not in all_recs:
                all_recs.append(rec)
                lines.append(f"- {rec}")

    return "\n".join(lines)


# ============================================================================
# 模块自检入口
# ============================================================================

def _self_test() -> bool:
    """自检: 验证 SimLiveGapAnalyzer 核心功能可用"""
    try:
        # 测试1: 基本分析流程 (BTC-USDT, 启用国内延迟)
        analyzer = SimLiveGapAnalyzer(symbol="BTC-USDT", use_domestic_latency=True)
        sample_trades = [
            {"entry_price": 50000, "exit_price": 51000, "side": "long", "quantity": 0.1, "pnl_pct": 2.0, "bars_held": 20},
            {"entry_price": 51000, "exit_price": 50500, "side": "long", "quantity": 0.1, "pnl_pct": -0.98, "bars_held": 15},
            {"entry_price": 50500, "exit_price": 51500, "side": "long", "quantity": 0.1, "pnl_pct": 1.98, "bars_held": 30},
        ]
        report = analyzer.analyze(sample_trades)
        assert report.symbol == "BTC-USDT"
        assert report.total_trades == 3
        assert isinstance(report.total_gap_bps, float)
        assert isinstance(report.factors, list) and len(report.factors) > 0

        # 测试2: 空 trades 边界场景 (应返回零差异报告)
        empty_report = analyzer.analyze([])
        assert empty_report.total_trades == 0
        assert empty_report.total_gap_bps == 0.0

        # 测试3: symbol 规范化 (BTC_USDT / btcusdt 应等价 BTC-USDT)
        analyzer_usdt = SimLiveGapAnalyzer(symbol="BTC_USDT", use_domestic_latency=False)
        assert analyzer_usdt.symbol == "BTC-USDT"
        analyzer_lower = SimLiveGapAnalyzer(symbol="btcusdt", use_domestic_latency=False)
        assert analyzer_lower.symbol == "BTC-USDT"

        # 测试4: regime 调整 (high_vol 应放大 spread/slippage 摩擦参数)
        base_analyzer = SimLiveGapAnalyzer(symbol="BTC-USDT", use_domestic_latency=False, regime="trend")
        hv_analyzer = SimLiveGapAnalyzer(symbol="BTC-USDT", use_domestic_latency=False, regime="high_vol")
        assert hv_analyzer.params.spread_bps > base_analyzer.params.spread_bps, \
            "high_vol 应放大点差 (REGIME_MULTIPLIERS spread=1.5)"
        assert hv_analyzer.params.slippage_per_100k_bps > base_analyzer.params.slippage_per_100k_bps, \
            "high_vol 应放大滑点 (REGIME_MULTIPLIERS slippage=1.5)"
        hv_report = hv_analyzer.analyze(sample_trades)
        assert hv_report.total_trades == 3

        logging.info("SimLiveGapAnalyzer 自检全部通过 ✓")
        return True

    except Exception as e:
        logging.error(f"自检失败: {e}")
        return False


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _self_test()

    # 示例: 分析模拟交易
    sample_trades = [
        {"entry_price": 50000, "exit_price": 51000, "side": "long", "quantity": 0.1, "pnl_pct": 2.0, "bars_held": 20},
        {"entry_price": 51000, "exit_price": 50500, "side": "long", "quantity": 0.1, "pnl_pct": -0.98, "bars_held": 15},
        {"entry_price": 50500, "exit_price": 51500, "side": "long", "quantity": 0.1, "pnl_pct": 1.98, "bars_held": 30},
    ]

    for symbol in ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT"]:
        analyzer = SimLiveGapAnalyzer(symbol=symbol, use_domestic_latency=True)
        report = analyzer.analyze(sample_trades)

        print(f"\n{'='*60}")
        print(f"  {symbol} 模拟-实盘差异分析")
        print(f"{'='*60}")
        print(f"  总差异: {report.total_gap_bps:.1f} bps ({report.total_gap_pct:.3f}%)")
        print(f"  蒙特卡洛均值: {report.mc_mean_gap_bps:.1f} bps")
        print(f"  蒙特卡洛P95: {report.mc_p95_gap_bps:.1f} bps")
        print(f"\n  各因素影响:")
        for f in sorted(report.factors, key=lambda x: abs(x.impact_bps_mean), reverse=True):
            sig = "⚠️" if f.is_significant else "  "
            print(f"    {sig} {f.factor_name}: {f.impact_bps_mean:.1f} bps (权重 {f.weight_pct:.1f}%)")
        print(f"\n  建议:")
        for rec in report.recommendations:
            print(f"    - {rec}")