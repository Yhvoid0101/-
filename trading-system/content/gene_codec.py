# -*- coding: utf-8 -*-
"""
子Agent基因编码器 (Gene Codec) — 五认知模块 + 扩展种子库

基因是子Agent的"DNA"，定义了它的交易认知框架。每个基因包含五个模块，
每个模块从种子库中随机抽取，并在进化过程中交叉变异。

种子库来源：
  - 经典交易策略 (Turtle, CANSLIM, 海龟)
  - Fincept Terminal 拆解
  - ai-hedge-fund 策略抽象 (桥水、城堡、西蒙斯)
  - 自研扩展

设计原则：
  - 基因 = 可序列化字典，便于持久化和交叉变异
  - 每个参数有明确的值域和类型约束
  - 种子库可扩展，新策略来源可持续加入
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 种子库定义
# ============================================================================

# --- 模块1：市场世界观种子库 ---
WORLDVIEW_SEEDS: Dict[str, Dict[str, Any]] = {
    "trend_following": {
        "label": "趋势跟随",
        "description": "顺势而为，不预测顶底，只跟随趋势",
        "origin": "turtle_trading",
        "timeframe_bias": ["4h", "1d"],
        "holding_period": "medium",
        "max_trades_per_day": 3,
        # v499 Phase 2: 补齐key_params三件套,打通GA优化→策略实例
        # Phase E2 fix: TurtleSignal is a data class, not a strategy.
        # TurtleTradingSystem has update() method compatible with _StrategyAdapter.
        "module": "turtle_trading",
        "class": "TurtleTradingSystem",
        "key_params": {
            "FAST_EMA": {"type": "int", "default": 20, "range": [5, 50]},
            "SLOW_EMA": {"type": "int", "default": 50, "range": [20, 200]},
            "ATR_PERIOD": {"type": "int", "default": 14, "range": [5, 30]},
            "STOP_LOSS_ATR_MULT": {"type": "float", "default": 2.0, "range": [1.0, 5.0]},
            "POSITION_RISK_PCT": {"type": "float", "default": 0.01, "range": [0.005, 0.03]},
        },
    },
    "mean_reversion": {
        "label": "均值回归",
        "description": "价格偏离均值后会回归，逆势交易",
        "origin": "statistical_arbitrage",
        "timeframe_bias": ["15m", "1h"],
        "holding_period": "short",
        "max_trades_per_day": 10,
        # v499 Phase 2: 补齐key_params三件套
        "module": "stat_arb_pairs",
        "class": "StatArbPairs",
        "key_params": {
            "LOOKBACK_PERIOD": {"type": "int", "default": 20, "range": [10, 100]},
            "ENTRY_Z_SCORE": {"type": "float", "default": 2.0, "range": [1.0, 3.5]},
            "EXIT_Z_SCORE": {"type": "float", "default": 0.5, "range": [0.0, 1.5]},
            "HALF_LIFE": {"type": "int", "default": 10, "range": [3, 50]},
            "STOP_LOSS_PCT": {"type": "float", "default": 0.03, "range": [0.01, 0.08]},
        },
    },
    "trend_pullback": {
        "label": "顺大逆小",
        "description": "顺大趋势逆小波动，趋势+回调入场 (Phase 14.8)",
        "origin": "trend_following+mean_reversion_hybrid",
        "timeframe_bias": ["1h", "4h"],
        "holding_period": "medium",
        "max_trades_per_day": 5,
        # Phase 14.8: 顺大逆小策略 — 无独立module, 使用内置信号
        "key_params": {
            "EMA_200_PERIOD": {"type": "int", "default": 200, "range": [100, 300]},
            "SMA_20_DEVIATION": {"type": "float", "default": 0.02, "range": [0.005, 0.05]},
            "RSI_PULLBACK_LONG": {"type": "float", "default": 40.0, "range": [25.0, 50.0]},
            "RSI_PULLBACK_SHORT": {"type": "float", "default": 60.0, "range": [50.0, 75.0]},
            "STOP_LOSS_ATR_MULT": {"type": "float", "default": 2.0, "range": [1.0, 5.0]},
        },
    },
    "breakout": {
        "label": "突破交易",
        "description": "关键价位突破后追入，吃惯性利润",
        "origin": "darvas_box",
        "timeframe_bias": ["1h", "4h"],
        "holding_period": "medium",
        "max_trades_per_day": 5,
        # v499 Phase 2: 补齐key_params三件套
        "module": "chanlun_priceaction",
        "class": "ChanLunPriceActionSystem",
        "key_params": {
            "BREAKOUT_PERIOD": {"type": "int", "default": 20, "range": [10, 55]},
            "VOLUME_CONFIRM_MULT": {"type": "float", "default": 1.5, "range": [1.0, 3.0]},
            "ATR_STOP_MULT": {"type": "float", "default": 1.5, "range": [0.5, 4.0]},
            "MAX_HOLD_BARS": {"type": "int", "default": 24, "range": [6, 100]},
            "TAKE_PROFIT_PCT": {"type": "float", "default": 0.04, "range": [0.01, 0.15]},
        },
    },
    "momentum": {
        "label": "动量交易",
        "description": "强者恒强，弱者恒弱，追涨杀跌的量化版",
        "origin": "aqr_momentum",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        # v499 Phase 2: 补齐key_params三件套
        "module": "deep_learning_forecast",
        "class": "DeepLearningForecastSystem",
        "key_params": {
            "MOMENTUM_PERIOD": {"type": "int", "default": 12, "range": [5, 50]},
            "RSI_PERIOD": {"type": "int", "default": 14, "range": [5, 30]},
            "RSI_OVERBOUGHT": {"type": "float", "default": 70.0, "range": [60.0, 85.0]},
            "RSI_OVERSOLD": {"type": "float", "default": 30.0, "range": [15.0, 40.0]},
            "TAKE_PROFIT_PCT": {"type": "float", "default": 0.06, "range": [0.02, 0.20]},
        },
    },
    "value_investing": {
        "label": "价值回归",
        "description": "巴菲特风格：寻找被低估资产，等待价值回归",
        "origin": "ai_hedge_fund",
        "timeframe_bias": ["1w", "1M"],
        "holding_period": "very_long",
        "max_trades_per_day": 1,
        # v499 Phase 2: 补齐key_params三件套
        "module": "portfolio_optimization",
        "class": "PortfolioOptimizationSystem",
        "key_params": {
            "VALUE_THRESHOLD": {"type": "float", "default": 0.3, "range": [0.1, 0.6]},
            "REBALANCE_PERIOD": {"type": "int", "default": 30, "range": [7, 90]},
            "MAX_POSITION_PCT": {"type": "float", "default": 0.25, "range": [0.05, 0.50]},
            "STOP_LOSS_PCT": {"type": "float", "default": 0.15, "range": [0.05, 0.30]},
            "MIN_HOLD_DAYS": {"type": "int", "default": 14, "range": [3, 60]},
        },
    },
    "safe_margin": {
        "label": "安全边际",
        "description": "格雷厄姆风格：只在有足够安全边际时入场",
        "origin": "ai_hedge_fund",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 1,
        # v499 Phase 2: 补齐key_params三件套
        "module": "risk_control",
        "class": "RiskControlManager",
        "key_params": {
            "SAFETY_MARGIN_PCT": {"type": "float", "default": 0.20, "range": [0.05, 0.50]},
            "MAX_RISK_PER_TRADE": {"type": "float", "default": 0.01, "range": [0.005, 0.03]},
            "STOP_LOSS_PCT": {"type": "float", "default": 0.05, "range": [0.02, 0.15]},
            "MIN_RR_RATIO": {"type": "float", "default": 2.0, "range": [1.0, 5.0]},
            "MAX_PORTFOLIO_RISK": {"type": "float", "default": 0.06, "range": [0.02, 0.15]},
        },
    },
    "stat_arbitrage": {
        "label": "统计套利",
        "description": "西蒙斯风格：基于统计规律的配对交易",
        "origin": "ai_hedge_fund",
        "timeframe_bias": ["1m", "5m"],
        "holding_period": "very_short",
        "max_trades_per_day": 50,
        # v499 Phase 2: 补齐key_params三件套
        "module": "stat_arb_pairs",
        "class": "StatArbPairs",
        "key_params": {
            "LOOKBACK_PERIOD": {"type": "int", "default": 60, "range": [20, 200]},
            "ENTRY_Z_SCORE": {"type": "float", "default": 2.5, "range": [1.5, 4.0]},
            "EXIT_Z_SCORE": {"type": "float", "default": 0.0, "range": [-0.5, 1.0]},
            "HALF_LIFE": {"type": "int", "default": 5, "range": [2, 30]},
            "MAX_HOLD_BARS": {"type": "int", "default": 120, "range": [30, 500]},
        },
    },
    "risk_parity": {
        "label": "风险平价",
        "description": "桥水全天候风格：各资产风险贡献均衡",
        "origin": "ai_hedge_fund",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        # v499 Phase 2: 补齐key_params三件套
        "module": "portfolio_optimization",
        "class": "PortfolioOptimizationSystem",
        "key_params": {
            "REBALANCE_PERIOD": {"type": "int", "default": 30, "range": [7, 90]},
            "TARGET_VOLATILITY": {"type": "float", "default": 0.12, "range": [0.05, 0.30]},
            "MAX_LEVERAGE": {"type": "float", "default": 1.5, "range": [1.0, 3.0]},
            "MIN_WEIGHT": {"type": "float", "default": 0.05, "range": [0.01, 0.20]},
            "MAX_WEIGHT": {"type": "float", "default": 0.40, "range": [0.20, 0.60]},
        },
    },
    "multi_strategy": {
        "label": "多策略组合",
        "description": "城堡风格：多策略并行，动态权重分配",
        "origin": "ai_hedge_fund",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "adaptive",
        "max_trades_per_day": 8,
        # v499 Phase 2: 补齐key_params三件套
        "module": "strategy_fusion",
        "class": "StrategyFusion",
        "key_params": {
            "STRATEGY_WEIGHTS": {"type": "dict", "default": {}, "range": []},
            "REBALANCE_PERIOD": {"type": "int", "default": 24, "range": [4, 100]},
            "MAX_CORRELATION": {"type": "float", "default": 0.7, "range": [0.3, 0.95]},
            "MIN_STRATEGY_WEIGHT": {"type": "float", "default": 0.05, "range": [0.01, 0.20]},
            "MAX_STRATEGY_WEIGHT": {"type": "float", "default": 0.40, "range": [0.20, 0.60]},
        },
    },
    "narrative_driven": {
        "label": "叙事驱动",
        "description": "Meme币风格：基于市场叙事和情绪交易",
        "origin": "crypto_native",
        "timeframe_bias": ["5m", "15m"],
        "holding_period": "very_short",
        "max_trades_per_day": 20,
        # v499 Phase 2: 补齐key_params三件套
        "module": "alpha_mining",
        "class": "AlphaMiningSystem",
        "key_params": {
            "SENTIMENT_THRESHOLD": {"type": "float", "default": 0.6, "range": [0.3, 0.9]},
            "VOLUME_SPIKE_MULT": {"type": "float", "default": 3.0, "range": [1.5, 10.0]},
            "MAX_HOLD_BARS": {"type": "int", "default": 12, "range": [3, 60]},
            "STOP_LOSS_PCT": {"type": "float", "default": 0.05, "range": [0.02, 0.15]},
            "TAKE_PROFIT_PCT": {"type": "float", "default": 0.10, "range": [0.03, 0.30]},
        },
    },
    "grid_trading": {
        "label": "网格交易",
        "description": "震荡市收割：价格区间内高抛低吸",
        "origin": "crypto_native",
        "timeframe_bias": ["5m", "15m"],
        "holding_period": "adaptive",
        "max_trades_per_day": 30,
        # v499 Phase 2: 补齐key_params三件套
        "module": "grid_strike",
        "class": "GridStrike",
        "key_params": {
            "GRID_LEVELS": {"type": "int", "default": 10, "range": [4, 30]},
            "GRID_SPACING_PCT": {"type": "float", "default": 0.01, "range": [0.002, 0.05]},
            "ORDER_SIZE_PCT": {"type": "float", "default": 0.05, "range": [0.01, 0.20]},
            "STOP_LOSS_PCT": {"type": "float", "default": 0.05, "range": [0.02, 0.15]},
            "REBALANCE_THRESHOLD": {"type": "float", "default": 0.03, "range": [0.01, 0.10]},
        },
    },
    "funding_rate_arb": {
        "label": "资金费率套利",
        "description": "Delta中性：现货多头+永续空头，收取资金费率（年化8-25%）",
        "origin": "binance_futures_2025",
        "timeframe_bias": ["1h", "4h", "8h"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "funding_rate_arb",
        "class": "FundingRateArbitrageur",
        "key_params": {
            "min_funding_rate": 0.0001,
            "max_leverage": 3,
            "target_annual_return": 0.10,
        },
    },
    "perp_market_making": {
        "label": "永续合约做市",
        "description": "Avellaneda-Stoikov做市：双侧挂单赚价差+资金费率",
        "origin": "hummingbot_pmm_2025",
        "timeframe_bias": ["1s", "1m", "5m"],
        "holding_period": "very_short",
        "max_trades_per_day": 500,
        "module": "perp_market_making",
        "class": "PerpetualMarketMaker",
        "key_params": {
            "spread": 0.001,
            "gamma": 0.1,
            "k": 1.5,
            "max_inventory": 0.1,
        },
    },
    "grid_strike": {
        "label": "网格打击(Triple Barrier)",
        "description": "Hummingbot Grid Strike：分层挂单+三重屏障风控(止盈/止损/时间)",
        "origin": "hummingbot_grid_strike_2025",
        "timeframe_bias": ["5m", "15m", "1h"],
        "holding_period": "adaptive",
        "max_trades_per_day": 50,
        "module": "grid_strike",
        "class": "GridStrike",
        "key_params": {
            "levels": 10,
            "take_profit": 0.02,
            "stop_loss": 0.01,
            "time_limit": 86400,
        },
    },
    "scalping": {
        "label": "高频刮头皮",
        "description": "极短时间内的微小利差交易",
        "origin": "hft_legacy",
        "timeframe_bias": ["1s", "1m"],
        "holding_period": "very_short",
        "max_trades_per_day": 200,
        "module": "scalping",
        "class": "ScalpingStrategy",
    },
    "funding_rate_predictor": {
        "label": "资金费率ML预测",
        "description": "线性回归预测下一期资金费率(76.4%方向准确率)+速度+跨所差+σ偏离",
        "origin": "yunmeng_quant_2026",
        "timeframe_bias": ["1h", "4h", "8h"],
        "holding_period": "long",
        "max_trades_per_day": 3,
        "module": "funding_rate_predictor",
        "class": "FundingRatePredictor",
        "key_params": {
            "lookback": 90,
            "sigma_entry": 2.0,
            "cross_exchange_delta": 0.0002,
        },
    },
    "order_flow_cvd": {
        "label": "订单流CVD分析",
        "description": "CVD背离(67%反转)+吸收+耗尽+冰山+幻单 5维订单流分析",
        "origin": "traderabyss_2026",
        "timeframe_bias": ["1m", "5m", "15m"],
        "holding_period": "short",
        "max_trades_per_day": 20,
        "module": "order_flow_cvd",
        "class": "OrderFlowCVDAnalyzer",
        "key_params": {
            "divergence_lookback": 20,
            "absorption_volume_mult": 2.0,
            "exhaustion_volume_mult": 3.0,
        },
    },
    "liquidation_hunter": {
        "label": "清算猎人",
        "description": "流动性集群识别+假突破检测+止损猎杀反转(65-80%反转概率)",
        "origin": "liquiditygrindai_2026",
        "timeframe_bias": ["5m", "15m", "1h"],
        "holding_period": "short",
        "max_trades_per_day": 10,
        "module": "liquidation_hunter",
        "class": "LiquidationHunter",
        "key_params": {
            "zone_lookback": 50,
            "sweep_pierce": 0.002,
            "sweep_volume_mult": 1.5,
        },
    },
    "stat_arb_pairs": {
        "label": "协整配对交易",
        "description": "Engle-Granger协整+半衰期+Z-Score信号,市场中性配对套利",
        "origin": "traderabyss_2026",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "stat_arb_pairs",
        "class": "StatArbPairs",
        "key_params": {
            "min_correlation": 0.75,
            "max_coint_p_value": 0.05,
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 3.5,
        },
    },
    "volatility_arbitrage": {
        "label": "波动率套利",
        "description": "IV/RV套利+Delta中性对冲+OTM Put尾部保护(年化8-20%,回撤<5%)",
        "origin": "gov_capital_2026",
        "timeframe_bias": ["4h", "1d"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "volatility_arbitrage",
        "class": "VolatilityArbitrage",
        "key_params": {
            "vrp_threshold": 0.05,
            "iv_z_entry": 2.0,
            "max_delta_exposure": 0.1,
            "tail_protection_strike_pct": 0.90,
        },
    },
    "on_chain_whale_tracker": {
        "label": "链上巨鲸追踪",
        "description": "Whale Alert API+$500K+大额交易+交易所流入流出+稳定币mint/burn+智能资金",
        "origin": "whale_alert_2026",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "long",
        "max_trades_per_day": 3,
        "module": "on_chain_whale_tracker",
        "class": "OnChainWhaleTracker",
        "key_params": {
            "large_tx_threshold": 500000.0,
            "inflow_bearish_threshold": 100000000.0,
            "outflow_bullish_threshold": 100000000.0,
            "accumulation_days": 3,
        },
    },
    "triangular_arbitrage": {
        "label": "三角套利",
        "description": "CCXT三角路径扫描+利润计算+滑点+延迟感知(年化5-15%中心化,10-30%DEX)",
        "origin": "voiceofchain_2026",
        "timeframe_bias": ["1s", "1m"],
        "holding_period": "very_short",
        "max_trades_per_day": 100,
        "module": "triangular_arbitrage",
        "class": "TriangularArbitrage",
        "key_params": {
            "min_profit_threshold": 0.003,
            "execute_threshold": 0.008,
            "taker_fee": 0.001,
            "latency_ms": 100,
        },
    },
    "mev_sandwich_arbitrage": {
        "label": "MEV三明治套利",
        "description": "Mempool监控+三明治攻击(前跑+后跑)+Flashbots Bundle原子执行+最优前跑金额计算",
        "origin": "flashbots_2026",
        "timeframe_bias": ["1s", "block"],
        "holding_period": "very_short",
        "max_trades_per_day": 200,
        "module": "mev_sandwich_arbitrage",
        "class": "MEVSandwichArbitrage",
        "key_params": {
            "min_profit_threshold": 0.001,
            "execute_threshold": 0.005,
            "gas_price_gwei": 30,
            "flashbots_bribe_pct": 0.3,
            "max_front_run_pct": 0.5,
        },
    },
    "cross_chain_arbitrage": {
        "label": "跨链桥套利",
        "description": "跨链价差检测+桥延迟预测+净利润计算(gas+桥费+滑点+回归风险)+多桥支持(Across/Stargate/Hop/Synapse)",
        "origin": "across_protocol_2026",
        "timeframe_bias": ["5m", "15m", "1h"],
        "holding_period": "short",
        "max_trades_per_day": 10,
        "module": "cross_chain_arbitrage",
        "class": "CrossChainArbitrage",
        "key_params": {
            "min_price_diff_pct": 0.005,
            "execute_diff_pct": 0.015,
            "max_delay_minutes": 30,
            "min_liquidity_usd": 50000.0,
            "max_slippage_pct": 0.005,
            "price_regression_risk": 0.3,
        },
    },
    "volatility_surface_arb": {
        "label": "波动率曲面套利",
        "description": "SVI模型校准(5参数)+曲面扭曲识别(残差>2vol点)+三种策略(单期权/日历价差/铁鹰)+CRR Delta中性对冲",
        "origin": "gatheral_svi_2004",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "volatility_surface_arb",
        "class": "VolatilitySurfaceArbitrage",
        "key_params": {
            "residual_threshold_vol": 2.0,
            "execute_threshold_vol": 4.0,
            "min_open_interest": 100,
            "max_delta_exposure": 0.1,
            "term_structure_inversion": -0.02,
            "skew_extreme_threshold": 6.0,
        },
    },
    "cex_dex_flash_arbitrage": {
        "label": "CEX-DEX闪贷套利",
        "description": "Aave V3闪贷零资本套利+CEX/DEX价差检测+原子执行(借入→交易→还款)+净利润计算(闪贷费0.09%+gas+滑点)",
        "origin": "aave_v3_2026",
        "timeframe_bias": ["1s", "block"],
        "holding_period": "very_short",
        "max_trades_per_day": 200,
        "module": "cex_dex_flash_arbitrage",
        "class": "CEXDEXFlashArbitrage",
        "key_params": {
            "min_price_diff_pct": 0.005,
            "execute_diff_pct": 0.015,
            "flash_loan_fee_pct": 0.0009,
            "max_gas_cost_usd": 50.0,
            "default_borrow_usd": 1000000.0,
        },
    },
    "jit_liquidity_mev": {
        "label": "JIT流动性MEV",
        "description": "Uniswap V3集中流动性JIT+Mempool监控大额swap+mint→swap→burn原子执行+费率份额计算(L_jit/(L_pool+L_jit))",
        "origin": "uniswap_v3_2026",
        "timeframe_bias": ["1s", "block"],
        "holding_period": "very_short",
        "max_trades_per_day": 100,
        "module": "jit_liquidity_mev",
        "class": "JITLiquidityMEV",
        "key_params": {
            "min_swap_usd": 300000.0,
            "execute_swap_usd": 500000.0,
            "target_fee_share": 0.95,
            "gas_price_gwei": 30,
            "capital_cost_rate": 0.0001,
        },
    },
    "gamma_scalping": {
        "label": "Gamma Scalping期权剥头皮",
        "description": "正Gamma头寸(Straddle)+动态Delta对冲(高卖低买)+Scalp收益=0.5×Gamma×(ΔS)²+RV/IV比率>1开仓+Theta超支平仓",
        "origin": "avellaneda_stoikov_2008",
        "timeframe_bias": ["1h", "4h"],
        "holding_period": "short",
        "max_trades_per_day": 20,
        "module": "gamma_scalping",
        "class": "GammaScalping",
        "key_params": {
            "rebalance_delta_threshold": 0.1,
            "rebalance_price_pct": 0.01,
            "rebalance_interval_hours": 4,
            "theta_breakeven_days": 30,
            "hedge_transaction_cost": 0.0005,
        },
    },
    "liquidation_cascade": {
        "label": "清算级联预测",
        "description": "LVI=(多头清算量/OI)×波动率乘子+LVI>50极端风险+清算集群识别+级联概率=f(LVI,费率,杠杆浓度)+Pre-Cascade减仓+Buy-the-Dip",
        "origin": "livevolatile_2026_lvi",
        "timeframe_bias": ["15m", "1h"],
        "holding_period": "short",
        "max_trades_per_day": 5,
        "module": "liquidation_cascade",
        "class": "LiquidationCascade",
        "key_params": {
            "lvi_moderate": 30.0,
            "lvi_high": 50.0,
            "cascade_prob_threshold": 0.6,
            "min_cluster_size_usd": 1000000000,
            "max_cluster_distance_pct": 5.0,
        },
    },
    "mean_reversion_ml": {
        "label": "统计均值回归+ML",
        "description": "OU过程拟合(θ,μ,σ,半衰期)+Z-score入场(±2σ)+VPIN毒性过滤(<0.3安全/>0.7停止)+半衰期<10天  [注:Hurst已从production特征管道移除-D2审计补遗]",
        "origin": "easley_lopez_de_prado_2012_vpin",
        "timeframe_bias": ["4h", "1d"],
        "holding_period": "medium",
        "max_trades_per_day": 3,
        "module": "mean_reversion_ml",
        "class": "MeanReversionML",
        "key_params": {
            "z_entry_threshold": 2.0,
            "z_exit_threshold": 0.5,
            "z_stop_loss_threshold": 3.5,
            "hurst_mean_reverting": 0.5,
            "vpin_low": 0.3,
            "vpin_high": 0.7,
            "max_half_life": 10.0,
        },
    },
    "dispersion_trading": {
        "label": "Dispersion离散交易",
        "description": "指数vs成分股波动率套利: Corr_implied=(IV_index²-Σw_i²·IV_i²)/(2·ΣΣw_i·w_j·IV_i·IV_j), 做空Dispersion(卖指数买成分)+做多Dispersion",
        "origin": "stockalpha_2026",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "dispersion_trading",
        "class": "DispersionTrading",
        "key_params": {
            "corr_low_threshold": 0.3,
            "corr_high_threshold": 0.7,
            "min_components": 5,
            "iv_spread_threshold": 0.02,
            "execute_iv_spread": 0.05,
        },
    },
    "vrp_harvesting": {
        "label": "波动率风险溢价收割",
        "description": "VRP=IV-RV, SPX平均2-4vol点/BTC 5-15vol点, 卖Straddle+尾部对冲/Iron Condor, Volmageddon尾部风险防护",
        "origin": "greeks_live_2026",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "vrp_harvesting",
        "class": "VRPHarvesting",
        "key_params": {
            "vrp_rich": 0.05,
            "vrp_normal": 0.02,
            "vrp_compressed": 0.0,
            "vvol_high": 100.0,
            "vvol_extreme": 110.0,
            "tail_hedge_ratio": 0.15,
            "min_vrp_to_act": 0.03,
        },
    },
    "smart_money_footprint": {
        "label": "聪明钱足迹追踪",
        "description": "5大链上信号(Exchange Flow/A-D/SSR/ETF Flows/DeFi Whale)+18,500+追踪钱包+DEX滑点暴露交易规模+聪明钱情绪[-1,+1]",
        "origin": "theliquidbeacon_2026",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "long",
        "max_trades_per_day": 3,
        "module": "smart_money_footprint",
        "class": "SmartMoneyFootprint",
        "key_params": {
            "exchange_inflow_bearish_usd": 100000000.0,
            "exchange_outflow_bullish_usd": 100000000.0,
            "stablecoin_mint_bullish_usd": 500000000.0,
            "etf_inflow_bullish_usd": 50000000.0,
            "min_smart_money_count": 3,
            "min_profit_score": 0.6,
        },
    },
    "term_structure_trading": {
        "label": "波动率期限结构交易",
        "description": "日历价差(卖近月买远月)+对角价差+事件驱动Vol Crush+Contango/Backwardation形态识别+Theta优势套利",
        "origin": "gate_2026",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "term_structure_trading",
        "class": "TermStructureTrading",
        "key_params": {
            "contango_threshold": 0.02,
            "backwardation_threshold": -0.02,
            "vol_crush_threshold": 0.05,
            "event_days_threshold": 7,
            "theta_advantage_min": 0.5,
        },
    },
    "rl_market_making": {
        "label": "深度强化学习做市",
        "description": "Avellaneda-Stoikov+PPO/SAC深度RL+对抗性RL+遗传算法优化初始参数(Sharpe 2.1-2.3)",
        "origin": "avellaneda_stoikov_2008_plosone_2022",
        "timeframe_bias": ["1s", "1m", "5m"],
        "holding_period": "very_short",
        "max_trades_per_day": 500,
        "module": "rl_market_making",
        "class": "RLMarketMaking",
        "key_params": {
            "gamma_init": 1.0,
            "gamma_min": 0.1,
            "gamma_max": 10.0,
            "kappa_init": 1.5,
            "q_max": 0.1,
            "rl_learning_rate": 0.001,
            "ga_population": 50,
            "ga_generations": 20,
        },
    },
    "cross_asset_stat_arb": {
        "label": "跨资产统计套利",
        "description": "BTC/ETH相关性(0.75-0.9)+Beta系数+Engle-Granger协整+Z-score配对交易+BTC Dominance regime",
        "origin": "engle_granger_1987_blofin_2026",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "cross_asset_stat_arb",
        "class": "CrossAssetStatArb",
        "key_params": {
            "coint_lookback": 100,
            "adf_critical_5pct": -2.86,
            "p_value_threshold": 0.05,
            "half_life_max": 20,
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            "zscore_stop_loss": 3.5,
            "correlation_min": 0.75,
            "btc_dom_high_threshold": 0.55,
            "btc_dom_low_threshold": 0.54,
        },
    },
    "event_driven_trading": {
        "label": "宏观事件驱动交易",
        "description": "CPI/FOMC/NFP事件驱动+恐惧贪婪指数(9=历史底部)+实际利率为负=BTC溢价+事件概率链",
        "origin": "cryptotakeprofit_2026_livevolatile_2026",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "medium",
        "max_trades_per_day": 3,
        "module": "event_driven_trading",
        "class": "EventDrivenTrading",
        "key_params": {
            "pre_event_reduce_hours": 24,
            "post_event_cooldown_hours": 1,
            "extreme_fear_threshold": 25,
            "historical_bottom_fgi": 10,
            "extreme_greed_threshold": 75,
            "real_rate_negative_threshold": -0.5,
            "dxy_strong_threshold": 105,
            "max_position_event_period": 0.5,
        },
    },
    "sentiment_driven_trading": {
        "label": "NLP情绪驱动交易",
        "description": "GPT/FinBERT/LSTM情绪分析(95.95%准确率)+情绪速度+情绪背离+情绪动量+多源共识",
        "origin": "kdd_2026_finbert_2025",
        "timeframe_bias": ["5m", "15m", "1h"],
        "holding_period": "short",
        "max_trades_per_day": 8,
        "module": "sentiment_driven_trading",
        "class": "SentimentDrivenTrading",
        "key_params": {
            "extreme_fear_threshold": -0.6,
            "extreme_greed_threshold": 0.6,
            "velocity_spike_threshold": 2.0,
            "divergence_threshold": 0.5,
            "momentum_threshold": 0.2,
            "consensus_threshold": 0.7,
            "vader_weight": 0.15,
            "lstm_weight": 0.30,
            "finbert_weight": 0.25,
            "gpt4_weight": 0.30,
        },
    },
    "ml_liquidation_predictor": {
        "label": "ML清算级联预测",
        "description": "LVI指标+LSTM+注意力机制+清算集群识别+Buy-the-Dip策略(级联预测准确率82%)",
        "origin": "lvi_2026_lstm_attention_2025",
        "timeframe_bias": ["1m", "5m", "15m"],
        "holding_period": "very_short",
        "max_trades_per_day": 15,
        "module": "ml_liquidation_predictor",
        "class": "MLLiquidationPredictor",
        "key_params": {
            "lvi_low_threshold": 15.0,
            "lvi_moderate_threshold": 30.0,
            "lvi_high_threshold": 50.0,
            "rsi_oversold": 25,
            "volume_spike_mult": 3.0,
            "buy_dip_stop_loss_pct": 0.02,
            "buy_dip_target_pct": 0.10,
            "cascade_prob_threshold": 0.65,
            "ml_feature_dim": 10,
        },
    },
    "gnn_cross_asset_relationship": {
        "label": "GNN跨资产关系建模",
        "description": "Graph Attention Network+多头注意力+系统性风险检测+传染概率+配对交易+板块轮动(15-30% Sharpe提升)",
        "origin": "openreview_gat_2026_korangi_alphastock_2019",
        "timeframe_bias": ["15m", "1h", "4h"],
        "holding_period": "medium",
        "max_trades_per_day": 6,
        "module": "gnn_cross_asset_relationship",
        "class": "GNNCrossAssetRelationship",
        "key_params": {
            "gat_layers": 2,
            "gat_heads": 4,
            "gat_hidden_dim": 16,
            "high_correlation_threshold": 0.7,
            "convergence_threshold": 0.85,
            "pca_systemic_threshold": 0.6,
            "contagion_threshold": 0.5,
            "pair_corr_min": 0.6,
            "pair_zscore_entry": 2.0,
            "lead_lag_window": 30,
        },
    },
    "transformer_decision_policy": {
        "label": "Decision Transformer序列决策",
        "description": "GPT-style因果Transformer+Return-to-Go条件化+多头注意力+轨迹异常检测 (v1.0基础版, 性能数字待实测)",
        "origin": "chen_2021_dt_lora_gpt2_2025_alphastock_2019",
        "timeframe_bias": ["5m", "15m", "1h"],
        "holding_period": "adaptive",
        "max_trades_per_day": 10,
        "module": "transformer_decision_policy",
        "class": "TransformerDecisionPolicy",
        "key_params": {
            "context_length": 20,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "state_dim": 16,
            "action_dim": 7,
            "rtg_scale": 0.1,
            "target_rtg_high": 0.10,
            "target_rtg_medium": 0.05,
            "anomaly_threshold": 0.7,
        },
    },
    "macro_liquidity_cycle": {
        "label": "宏观流动性周期建模",
        "description": "Global M2(0.94相关性)+DXY+实际利率+稳定币+ETF+65月周期(60-70天滞后)",
        "origin": "onramp_2026_afsheenjafry_2025_lyn_alden_2024",
        "timeframe_bias": ["1d", "1w", "1M"],
        "holding_period": "long",
        "max_trades_per_day": 1,
        "module": "macro_liquidity_cycle",
        "class": "MacroLiquidityCycle",
        "key_params": {
            "m2_expansion_threshold": 5.0,
            "m2_contraction_threshold": 0.0,
            "m2_lag_days": 65,
            "dxy_strong": 105.0,
            "dxy_weak": 95.0,
            "real_rate_high": 2.0,
            "real_rate_low": 0.0,
            "cycle_length_months": 65,
            "weight_m2": 0.35,
            "weight_dxy": 0.25,
            "weight_real_rate": 0.20,
        },
    },
    # ===== v3.4 新增 (3种子) — 跨市场套利/对抗性RL/风险平价策略级 =====
    "adversarial_rl_trading": {
        "label": "对抗性强化学习交易",
        "description": "ArchetypeTrader VQ原型蒸馏+FineFT集成Q选择性TD更新+VAE能力边界+TraderBench 4级对抗测试",
        "origin": "archetypetrader_aaai_2026_fineft_kdd_2026_traderbench_iclr_2026",
        "timeframe_bias": ["5m", "15m", "1h"],
        "holding_period": "adaptive",
        "max_trades_per_day": 8,
        "module": "adversarial_rl_trading",
        "class": "AdversarialRLTrading",
        "key_params": {
            "state_dim": 16,
            "n_archetypes": 8,
            "n_q_learners": 4,
            "n_actions": 7,
            "capability_low_threshold": 0.4,
            "adversarial_alarm_threshold": 0.5,
            "ensemble_consensus_low": 0.5,
            "q_value_strong_threshold": 0.5,
            "adversarial_noise_noisy": 0.05,
            "adversarial_noise_meta": 0.10,
            "adversarial_noise_adversarial": 0.20,
        },
    },
    "risk_parity_signal": {
        "label": "风险平价策略级信号",
        "description": "波动率目标(Moreira&Muir2017)+风险预算+Dynamic Kelly策略级+CVaR约束+回撤控制",
        "origin": "moreira_muir_2017_kelly_1956_rockafellar_uryasev_2000_bruch_mitchel_2013",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 1,
        "module": "risk_parity_signal",
        "class": "RiskParitySignalGenerator",
        "key_params": {
            "target_annual_vol": 0.20,
            "vol_lookback": 60,
            "cvar_confidence": 0.95,
            "max_drawdown_limit": 0.15,
            "drawdown_warning": 0.10,
            "kelly_lookback": 50,
            "kelly_fraction": 0.5,
            "kelly_min": 0.0,
            "kelly_max": 0.25,
            "low_vol_ratio": 0.7,
            "high_vol_ratio": 1.3,
            "extreme_vol_ratio": 2.0,
        },
    },
    "seasonality_cycle": {
        "label": "季节性周期",
        "description": "比特币4年减半周期+500天Pre-Halving买入规则+MVRV/ReserveRisk链上指标+月末季末+周末+月初效应",
        "origin": "theledgermind_2026_moneypartners_2026_deficryptonews_2026",
        "timeframe_bias": ["1d", "1w", "1M"],
        "holding_period": "very_long",
        "max_trades_per_day": 1,
        "module": "seasonality_cycle",
        "class": "SeasonalityCycleStrategy",
        "key_params": {
            "pre_halving_buy_days": 500,
            "post_halving_peak_min": 480,
            "post_halving_peak_max": 550,
            "peak_to_bottom_days": 365,
            "mvrv_cycle_top": 3.5,
            "mvrv_caution": 3.0,
            "mvrv_undervalued": 1.5,
            "reserve_risk_top_warning": 0.002,
            "reserve_risk_bullish": 0.008,
            "month_end_window_days": 5,
            "turn_of_month_days": 4,
            "weekend_vol_premium": 0.30,
        },
    },
    "cppi_portfolio_insurance": {
        "label": "CPPI组合保险",
        "description": "Black-Perold CPPI(Cushion=m×max(V-F,0))+TIPP动态底线+Gap风险检测+动态乘数+压力测试+凸性收益",
        "origin": "black_perold_1987_perold_sharpe_1988_stockalpha_2026_gelonghui_2026",
        "timeframe_bias": ["1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 1,
        "module": "cppi_portfolio_insurance",
        "class": "CPPIPortfolioInsuranceStrategy",
        "key_params": {
            "initial_floor_ratio": 0.85,
            "floor_min_ratio": 0.70,
            "floor_max_ratio": 0.95,
            "multiplier_base": 3.0,
            "multiplier_min": 1.0,
            "multiplier_max": 5.0,
            "target_annual_vol": 0.30,
            "rebalance_threshold": 0.05,
            "tipp_floor_lock_ratio": 0.90,
            "gap_risk_lambda": 2.0,
            "gap_risk_jump_size": -0.10,
            "gap_risk_threshold": 0.05,
            "gap_risk_alarm_threshold": 0.15,
            "floor_breach_warning_distance": 2.0,
            "max_leverage": 1.5,
            "min_cash_buffer": 0.05,
        },
    },
    "diffusion_forecast_trading": {
        "label": "扩散模型预测",
        "description": "DDPM去噪扩散概率模型+多分辨率分解(粗/中/精)+残差扩散(backbone AR+残差DDPM)+概率密度预测(50次采样非点估计)+波动率聚类+厚尾/偏度检测+VaR/CVaR尾部风险",
        "origin": "ho_2020_timegrad_csdi_2024_mr_diff_iclr_2024_diffolio_2026_re_diffusion_www_2026",
        "timeframe_bias": ["15m", "1h", "4h"],
        "holding_period": "short",
        "max_trades_per_day": 5,
        "module": "diffusion_forecast_trading",
        "class": "DiffusionForecastTrading",
        "key_params": {
            "t_steps": 50,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "n_samples": 50,
            "coarse_window": 20,
            "medium_window": 5,
            "backbome_lag": 5,
            "vol_cluster_window": 30,
            "vol_cluster_threshold": 1.5,
            "fat_tail_kurtosis": 3.5,
            "skew_threshold": 0.5,
            "high_uncertainty_threshold": 0.025,
            "sharpe_threshold": 0.50,
            "var_threshold": -0.03,
            "cvar_threshold": -0.04,
            "min_history": 60,
        },
    },
    "liquidity_adjusted_strategy": {
        "label": "流动性风险调整",
        "description": "Amihud非流动性比率+Kyle lambda价格冲击+VPIN流量毒性(BVC批量分类)+流动性-波动率反馈环+5种流动性Regime(HIGH/MEDIUM/LOW/THIN/CRISIS)+流动性调整仓位(平方根冲击模型)+流动性溢价收割",
        "origin": "amihud_2002_kyle_1985_easley_2012_livevolatile_2026",
        "timeframe_bias": ["5m", "15m", "1h"],
        "holding_period": "short",
        "max_trades_per_day": 8,
        "module": "liquidity_adjusted_strategy",
        "class": "LiquidityAdjustedStrategy",
        "key_params": {
            "amihud_window": 30,
            "amihud_high_liq": 1.0,
            "amihud_medium_liq": 10.0,
            "amihud_low_liq": 100.0,
            "amihud_crisis": 500.0,
            "kyle_window": 50,
            "kyle_lambda_high": 0.0001,
            "impact_cost_bps_threshold": 20.0,
            "vpin_batch_size": 20,
            "vpin_window": 10,
            "vpin_toxic_threshold": 0.70,
            "feedback_vol_increase": 2.0,
            "feedback_volatility_increase": 1.8,
            "feedback_liq_decrease": 0.5,
            "feedback_min_duration": 3,
            "target_impact_cost_bps": 5.0,
            "max_position_scale": 1.5,
            "min_position_scale": 0.1,
            "base_size": 0.10,
            "premium_lookback": 60,
            "premium_min_threshold": 0.001,
            "min_history": 50,
        },
    },
    "llm_agent_trading": {
        "label": "LLM Agent推理交易",
        "description": "ReAct多步推理(5步趋势/动量/RSI/量能/综合)+Tree-of-Thought三分支并行(牛/熊/震荡)+多Agent辩论(4 Agent: 多/空/风控/宏观,3轮贝叶斯权重更新)+反幻觉校验(工具冲突0.4+证据不足0.3+推理一致性0.3)+共识投票(60%分支+40%Agent)+11种信号类型",
        "origin": "timi_iclr_2026_finagent_2026_react_2022_tot_2023_multi_agent_debate_2023_fingpt_2023",
        "timeframe_bias": ["15m", "1h", "4h"],
        "holding_period": "medium",
        "max_trades_per_day": 4,
        "module": "llm_agent_trading",
        "class": "LLMAgentTradingStrategy",
        "key_params": {
            "rsi_period": 14,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ma_short": 10,
            "ma_long": 50,
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "react_steps": 5,
            "n_branches": 3,
            "branch_bull_bias": 0.3,
            "branch_bear_bias": -0.3,
            "n_agents": 4,
            "debate_rounds": 3,
            "agent_learning_rate": 0.1,
            "consensus_threshold": 0.65,
            "hallucination_threshold": 0.55,
            "high_divergence_threshold": 0.55,
            "min_history": 60,
            "max_volatility_threshold": 0.06,
            "risk_veto_drawdown": 0.15,
        },
    },
    "microstructure_alpha_trading": {
        "label": "订单簿微观结构Alpha",
        "description": "OBI订单簿不平衡(Cont 2014深度加权)+Hawkes爆发检测(自激发强度λ(t))+Kyle lambda价格冲击(在线滚动估计)+订单簿队列位置(Foucault 2005队列消散+到达率估计+队列时间)+订单簿韧性(Mounjid 2024冲击恢复率)+逆向选择保护(Glosten-Milgrom PIN+VPIN毒性)+综合Alpha融合(5分量加权+多重确认)+降级模式(K线近似订单簿)+11种信号",
        "origin": "cont_2014_kyle_1985_hawkes_1971_foucault_2005_rosu_2009_pindza_2026_mounjid_2024_noble_2026",
        "timeframe_bias": ["1s", "5s", "1m"],
        "holding_period": "very_short",
        "max_trades_per_day": 20,
        "module": "microstructure_alpha_trading",
        "class": "MicrostructureAlphaStrategy",
        "key_params": {
            "obi_n_levels": 5,
            "obi_decay": 0.8,
            "obi_strong_threshold": 0.5,
            "obi_medium_threshold": 0.3,
            "obi_weak_threshold": 0.1,
            "kyle_window": 50,
            "kyle_min_history": 30,
            "kyle_impact_max": 0.05,
            "hawkes_baseline": 1.0,
            "hawkes_alpha": 0.5,
            "hawkes_beta": 1.0,
            "hawkes_burst_threshold": 2.5,
            "hawkes_memory": 30,
            "queue_arrival_window": 20,
            "queue_time_max": 50,
            "queue_pressure_threshold": 0.3,
            "queue_front_threshold": 0.7,
            "resilience_window": 30,
            "resilience_shock_threshold": 0.5,
            "resilience_recovery_time": 10,
            "resilience_healthy": 0.7,
            "resilience_breakdown": 0.3,
            "vpin_window": 50,
            "vpin_toxic_threshold": 0.7,
            "vpin_warning_threshold": 0.5,
            "pin_high_threshold": 0.3,
            "adverse_cost_max_bps": 5.0,
            "w_obi": 0.25,
            "w_hawkes": 0.15,
            "w_queue": 0.20,
            "w_resilience": 0.15,
            "w_adverse_penalty": 0.25,
            "alpha_signal_threshold": 0.30,
            "confirm_min_sources": 3,
            "min_history": 50,
        },
    },
    "marl_portfolio_management": {
        "label": "多智能体RL组合管理",
        "description": "MAPPO架构(中心Critic+分布Actor,CTDE训练)+Mean-Field博弈(平均场动作,支持100+智能体)+Shapley信用分配(蒙特卡洛采样排列,公平分配组合收益)+COMA反事实基线+QMIX单调混值分解+Nash均衡检测(ε-Nash收敛)+GAE优势估计+PPO clip目标+组合权重softmax(Shapley加权)+杠杆约束+11种信号",
        "origin": "maddpg_2017_mappo_2022_qmix_2018_mean_field_2018_shapley_1953_nash_1950_coma_2018_finmarl_2022_finft_kdd_2026",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "long",
        "max_trades_per_day": 1,
        "module": "marl_portfolio_management",
        "class": "MARLPortfolioManagement",
        "key_params": {
            "n_features": 10,
            "hidden_actor": 32,
            "hidden_critic": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ppo_epochs": 4,
            "actor_lr": 0.001,
            "critic_lr": 0.002,
            "entropy_coef": 0.01,
            "mean_field_window": 10,
            "mean_field_decay": 0.9,
            "shapley_samples": 20,
            "shapley_rebalance_threshold": 0.15,
            "nash_distance_threshold": 0.05,
            "nash_min_iterations": 30,
            "max_leverage": 1.5,
            "max_single_weight": 0.50,
            "target_annual_vol": 0.30,
            "agreement_threshold": 0.65,
            "disagreement_threshold": 0.35,
            "confidence_min": 0.40,
            "min_history": 30,
        },
    },
    "diffusion_forecast_v1_1": {
        "label": "扩散预测v1.1增强",
        "description": "v1.1增强版扩散预测:DDIM确定性采样(50步→10步5x加速)+CFG分类器自由引导(w=2.0自适应)+Latent Diffusion潜空间扩散(8维压缩)+CRPS概率预测评分+多分辨率扩散+残差扩散+概率密度预测+波动率聚类+厚尾检测+11种信号",
        "origin": "ddpm_2020_ddim_2021_cfg_2022_ldm_2022_crps_1976_diffolio_2026_re_diffusion_2026",
        "timeframe_bias": ["15m", "1h", "4h"],
        "holding_period": "short",
        "max_trades_per_day": 5,
        "module": "diffusion_forecast_trading",
        "class": "DiffusionForecastTrading",
        "key_params": {
            "t_steps": 50,
            "ddim_steps": 10,
            "ddim_eta": 0.0,
            "cfg_guidance_w": 2.0,
            "cfg_uncond_prob": 0.10,
            "cfg_w_min": 0.5,
            "cfg_w_max": 5.0,
            "latent_dim": 8,
            "latent_window": 25,
            "crps_window": 30,
            "crps_target": 0.015,
            "crps_adjust_rate": 0.10,
            "coarse_window": 20,
            "medium_window": 5,
            "backbone_lag": 5,
            "vol_cluster_window": 30,
            "fat_tail_kurtosis": 3.5,
            "min_history": 60,
        },
    },
    "causal_inference_v1_1": {
        "label": "因果推断v1.1增强",
        "description": "v1.1增强版因果推断:PC算法因果发现(条件独立性检验+Fisher Z+v-structure方向)+LiNGAM非高斯因果方向(Darmois-Skitovich统计量)+ANM后非线性因果(多项式回归degree=3)+多算法融合(Granger0.35+PC0.25+LiNGAM0.20+ANM0.20投票)+do-calculus+因果掩码注意力+鲁棒性检验+11种信号",
        "origin": "pc_2000_lingam_2006_anm_2016_dowhy_2026_mct_2026_pearl_1995_notears_2018",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "medium",
        "max_trades_per_day": 3,
        "module": "causal_inference_trading",
        "class": "CausalInferenceTrading",
        "key_params": {
            "n_variables": 7,
            "granger_lag": 3,
            "granger_window": 60,
            # v4.0 Phase 2: 8个GA可优化参数 (结构化格式, 打通GA→策略实例)
            # 来源: causal_inference_trading.py 30+硬编码阈值中影响最大的8个
            "granger_p_value_threshold": {"type": "float", "default": 0.05, "range": [0.01, 0.10]},
            "decay_lambda": {"type": "float", "default": 0.02, "range": [0.005, 0.05]},
            "ate_strong_threshold": {"type": "float", "default": 0.30, "range": [0.15, 0.50]},
            "robustness_threshold": {"type": "float", "default": 0.50, "range": [0.30, 0.70]},
            "fusion_weight_granger": {"type": "float", "default": 0.35, "range": [0.15, 0.55]},
            "fusion_weight_pc": {"type": "float", "default": 0.25, "range": [0.10, 0.40]},
            "fusion_weight_lingam": {"type": "float", "default": 0.20, "range": [0.10, 0.35]},
            "fusion_weight_anm": {"type": "float", "default": 0.20, "range": [0.10, 0.35]},
            # 以下参数保持固定 (非GA优化, 避免搜索空间过大)
            "pc_p_value_threshold": 0.05,
            "pc_max_conditioning": 2,
            "pc_fisher_z_bias": 0.5,
            "lingam_non_gaussian_threshold": 0.20,
            "lingam_ds_ratio": 1.15,
            "lingam_min_samples": 50,
            "anm_poly_degree": 3,
            "anm_independence_ratio": 0.85,
            "anm_min_samples": 40,
            "fusion_vote_min": 2,
        },
    },
    "marl_portfolio_v1_1": {
        "label": "MARL组合v1.1增强",
        "description": "v1.1增强版MARL组合:QMIX混值分解(Hypernetwork+abs权重单调性约束∂Q_tot/∂Q_i≥0)+COMA反事实Critic(中心化Q网络+蒙特卡洛反事实基线b_i=E[Q(s,a_{-i},a')]+优势A_i=Q-b_i+裁剪)+Shapley信用分配+Mean-Field博弈+MAPPO+PPO+组合权重融合(Shapley+COMA优势)+11种信号",
        "origin": "maddpg_2017_mappo_2022_qmix_2018_coma_2018_mean_field_2018_shapley_1953_nash_1950_qplex_2021_qtran_2019",
        "timeframe_bias": ["1h", "4h", "1d"],
        "holding_period": "long",
        "max_trades_per_day": 1,
        "module": "marl_portfolio_management",
        "class": "MARLPortfolioManagement",
        "key_params": {
            "n_features": 10,
            "hidden_actor": 32,
            "hidden_critic": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ppo_epochs": 4,
            "actor_lr": 0.001,
            "critic_lr": 0.002,
            "entropy_coef": 0.01,
            "mean_field_window": 10,
            "shapley_samples": 20,
            "qmix_embedding_dim": 16,
            "qmix_hidden_dim": 32,
            "qmix_eps": 1e-8,
            "qmix_lr": 0.0005,
            "qmix_td_lambda": 0.6,
            "coma_q_hidden": 64,
            "coma_lr": 0.001,
            "coma_counterfactual_samples": 8,
            "coma_advantage_clip": 5.0,
            "max_leverage": 1.5,
            "max_single_weight": 0.50,
            "agreement_threshold": 0.65,
            "min_history": 30,
        },
    },
    "gnn_cross_asset_v1_1": {
        "label": "GNN跨资产v1.1增强",
        "description": "v1.1增强版GNN跨资产关系建模:BiMamba双向选择性SSM时序编码(HiPPO-LegS初始化+input-dependent Δ_t+前向/后向扫描+O(L)线性复杂度)+Bayesian MAGAC多头自适应图注意力(Chebyshev K=2阶谱卷积+动态邻接α·高斯核+(1-α)·学习注意力+MC-Dropout K=5次采样不确定性+DropEdge防过平滑)+Rank-Aware U形损失(head/tail权重w_max=2.0,middle权重w_min=0.5)+GAT多头注意力+系统性风险+传染概率+领先-滞后+配对交易不确定性加成/衰减+11种信号",
        "origin": "bimamba_iclr_2026_mamba_2023_s4_2020_hippo_2020_bayesian_magac_iclr_2026_stockmamba_2026_samba_2024_dystage_icaif_2024_stgat_2025_gat_2018_chebnet_2016_mc_dropout_2016_dropedge_2019",
        "timeframe_bias": ["15m", "1h", "4h"],
        "holding_period": "medium",
        "max_trades_per_day": 6,
        "module": "gnn_cross_asset_relationship",
        "class": "GNNCrossAssetRelationship",
        "key_params": {
            "gat_layers": 2,
            "gat_heads": 4,
            "gat_hidden_dim": 16,
            "high_correlation_threshold": 0.7,
            "convergence_threshold": 0.85,
            "pca_systemic_threshold": 0.6,
            "contagion_threshold": 0.5,
            "pair_corr_min": 0.6,
            "pair_zscore_entry": 2.0,
            "lead_lag_window": 30,
            "bimamba_state_dim": 8,
            "bimamba_input_dim": 1,
            "bimamba_output_dim": 8,
            "bimamba_seq_len": 50,
            "bimamba_lr": 0.005,
            "bimamba_delta_init": 0.5,
            "bimamba_delta_bias": -0.5,
            "bimamba_update_interval": 5,
            "magac_k_hop": 2,
            "magac_gaussian_bandwidth": 0.5,
            "magac_alpha_blend": 0.6,
            "magac_mc_samples": 5,
            "magac_dropout_p": 0.1,
            "magac_dropedge_p": 0.1,
            "magac_lr": 0.003,
            "magac_uncertainty_threshold": 0.3,
            "magac_confidence_boost": 0.5,
            "rank_aware_w_max": 2.0,
            "rank_aware_w_min": 0.5,
            "rank_aware_enabled": True,
        },
    },
    "transformer_decision_v1_1": {
        "label": "Decision Transformer v1.1 LoRA增强",
        "description": "v1.1增强版DT序列决策:LoRA低秩适配器(每层Q/K/V均挂r=8秩适配器,W'=W₀+s·B·A,B零初始化A高斯初始化,参数减少10000x,在线反向传播更新A/B)+Decoupling RTG(训练时用actual_rtg,推理时rtg_eff=λ·target+（1-λ)·actual,λ=0.7减少train-test分布偏移)+Cross-Asset Attention跨资产注意力(KDD 2019 AlphaStock CAAN:α_ij=softmax(W_a·tanh(W_s·s_i+W_t·s_j))+单资产退化为自注意力)+Adaptive Hybrid Attention自适应混合注意力(局部指数衰减窗口5+全局softmax+sigmoid门控w_local)+在线学习机制(MSE损失+LoRA反向传播+梯度裁剪1.0+EMA准确率跟踪0.95衰减)+Dropout正则化(训练时0.1应用到注意力分数和FFN)+9种信号(STRONG_BUY/BUY/SELL/STRONG_SELL/REGIME_ADAPTIVE/TRAJECTORY_ANOMALY/POSITION_SCALE/HOLD/NEUTRAL)",
        "origin": "chen_2021_hu_2021_lora_iclr_2022_dora_2024_adalora_2023_wang_2019_alphastock_kdd_vaswani_2017_srivastava_2014_dropout_dubois_2021_rtg_decoupling_arxiv_2026",
        "timeframe_bias": ["5m", "15m", "1h"],
        "holding_period": "adaptive",
        "max_trades_per_day": 10,
        "module": "transformer_decision_policy",
        "class": "TransformerDecisionPolicy",
        "key_params": {
            # v1.0 基础参数
            "context_length": 20,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff": 128,
            "dropout": 0.1,
            "state_dim": 16,
            "action_dim": 7,
            "rtg_scale": 0.1,
            "target_rtg_high": 0.10,
            "target_rtg_medium": 0.05,
            "target_rtg_low": 0.0,
            "target_rtg_negative": -0.05,
            "anomaly_threshold": 0.7,
            "trajectory_min_length": 10,
            # v1.1 LoRA 参数
            "lora_rank": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "lora_init_scale": 0.02,
            # v1.1 Decoupling RTG 参数
            "decouple_lambda": 0.7,
            "decouple_train_lambda": 0.0,
            # v1.1 Cross-Asset Attention 参数
            "cross_asset_enabled": True,
            "cross_asset_dim": 32,
            # v1.1 Adaptive Hybrid Attention 参数
            "hybrid_local_window": 5,
            "hybrid_enabled": True,
            # v1.1 在线学习参数
            "online_learning_enabled": True,
            "online_learning_interval": 5,
            "grad_clip": 1.0,
            "online_lr": 0.005,
            "ema_accuracy_decay": 0.95,
        },
    },
    # ==================== P0 深度交易知识模块 (v3.10.14 新增) ====================
    # 用户要求: 理解底层逻辑而非表面指标
    # 这三个模块是 ICT/Wyckoff/SMC 体系的核心，机构交易基础
    "fair_value_gap_detector": {
        "label": "FVG公允价值缺口检测",
        "description": "ICT核心:Displacement(位移)+Imbalance(失衡)+Liquidity Void(流动性真空)三层过滤;UNMITIGATED->PARTIAL->MITIGATED->IFVG_REVERSED状态机管理缺口生命周期;ATR*1.5位移确认过滤噪音缺口;BOS联动确认机构意图;IFVG反向突破提供反转信号;7种信号(FVG_BOUNCE_LONG/SHORT/FILL_WARNING/FULLY_MITIGATED/IFVG_REVERSAL_LONG/SHORT/DISPLACEMENT_BREAKOUT)",
        "origin": "ict_smart_money_2026_fvg_imbalance_liquidity_void_2026_inner_circle_trader_2026",
        "timeframe_bias": ["5m", "15m", "1h", "4h"],
        "holding_period": "short",
        "max_trades_per_day": 5,
        "module": "fair_value_gap_detector",
        "class": "FairValueGapDetector",
        "key_params": {
            "lookback_bars": 100,
            "min_gap_size_pct": 0.001,
            "displacement_atr_mult": 1.5,
            "atr_period": 14,
            "partial_fill_threshold": 0.5,
            "full_fill_threshold": 0.95,
            "ifvg_break_buffer": 0.001,
        },
    },
    "volume_profile_analyzer": {
        "label": "Volume Profile成交量轮廓图",
        "description": "机构成本识别:POC=机构强烈认可价(70-80%反转);HVN=机构强烈认可价位;LVN=机构不认可价位(快速穿越区);Value Area 70%成交量密集区(VAH/VAL);Shape识别机构意图(D=派发完成/P=派发进行/b=吸筹完成/Double P=派发加强/Rectangle=未定);Naked POC未回填POC=磁吸效应;10种信号(POC_REJECTION_LONG/SHORT/HVN_BREAKOUT_LONG/HVN_HOLD_LONG/LVN_TRAVERSAL_LONG/VAH_REVERSAL_SHORT/SHAPE_P_DISTRIBUTION/SHAPE_b_ACCUMULATION/NAKED_POC_LONG)",
        "origin": "steidlmayer_1987_volumeprofile_pure_volume_auction_2026_market_profile_2026_poc_2026_value_area_2026",
        "timeframe_bias": ["15m", "1h", "4h", "1d"],
        "holding_period": "medium",
        "max_trades_per_day": 4,
        "module": "volume_profile_analyzer",
        "class": "VolumeProfileAnalyzer",
        "key_params": {
            "num_bins": 50,
            "value_area_pct": 0.70,
            "hvn_volume_mult": 1.5,
            "lvn_volume_mult": 0.5,
            "shape_d_tolerance": 0.10,
            "shape_p_ratio": 1.5,
        },
    },
    "wyckoff_phase_analyzer": {
        "label": "Wyckoff威科夫5+5阶段图",
        "description": "经典量价体系:5阶段吸筹(PhaseA:PS/SC/BC/AR/ST+PhaseB:Creek测试+Spring+PhaseC:Spring确认+PhaseD:SOS+Backup+PhaseE:LPS出货)+5阶段派发(PhaseA:PS/BC/AR/ST+PhaseB:UT/UTAD测试+PhaseC:UTAD确认+PhaseD:LPSY+SOW+PhaseE:派发完成);Spring=向下假突破回收(70-80%胜率最强做多);UTAD=向上假突破回收(最强做空);Backup=SOS后回踩(75-85%胜率最佳入场);SOS=放量突破;SOW=缩量跌破;7种信号(WYCKOFF_SPRING_LONG/UTAD_SHORT/BACKUP_LONG/SOS_LONG/SOW_SHORT/SC_LONG/BC_SHORT)",
        "origin": "wyckoff_1931_5_plus_5_phase_schematic_2026_richard_wyckoff_method_2026_institutional_accumulation_distribution_2026",
        "timeframe_bias": ["1h", "4h", "1d", "1w"],
        "holding_period": "long",
        "max_trades_per_day": 2,
        "module": "wyckoff_phase_analyzer",
        "class": "WyckoffPhaseAnalyzer",
        "key_params": {
            "range_lookback": 100,
            "range_min_bars": 30,
            "climax_volume_mult": 3.0,
            "spring_pierce": 0.005,
            "spring_recovery_bars": 3,
            "sos_volume_mult": 2.0,
            "sos_breakout_pct": 0.01,
        },
    },
}

# --- 模块2：市场状态判断 —— 工具池 ---
TREND_DETECTORS: List[str] = [
    "ema_cross",        # EMA交叉
    "sma_cross",        # SMA交叉
    "adx",              # 平均趋向指数
    "ichimoku",         # 一目均衡表
    "macd_trend",       # MACD趋势
    "parabolic_sar",    # 抛物线转向
    "supertrend",       # 超级趋势
    "donchian",         # 唐奇安通道
    "kama",             # 考夫曼自适应均线
]

VOLATILITY_DETECTORS: List[str] = [
    "bollinger",        # 布林带
    "atr",              # 平均真实波幅
    "keltner",          # 肯特纳通道
    "historical_vol",   # 历史波动率
    "garch_vol",        # GARCH波动率估计
    "parkinson_vol",    # Parkinson波动率
]

MACRO_DATA_SOURCES: List[str] = [
    "none",             # 不使用宏观数据
    "funding_rate",     # 资金费率
    "open_interest",    # 持仓量
    "long_short_ratio", # 多空比
    "exchange_flows",   # 交易所流入流出
    "supply_chain",     # Fincept供应链数据
]

# --- 模块3：入场决策逻辑 —— 信号工具池 ---
# v3.21 修复根因2: 滞后指标虚假共振 — 加入领先指标作为主信号
# 分类标记 (用于_compute_entry_signals中分类计数):
#   [LAGGING] 滞后指标 - 仅作确认, 不主导方向
#   [LEADING] 领先指标 - 主导方向判断
PRIMARY_SIGNALS: List[str] = [
    # --- 领先指标 (v3.21新增, 优先采纳) ---
    "price_momentum_roc",  # [LEADING] 价格动量ROC - 直接度量价格变化, 无平滑
    "mfi_divergence",      # [LEADING] 资金流量MFI - 价量综合超买超卖
    "willr_signal",        # [LEADING] Williams %R - 响应快的超买超卖
    "volume_anomaly",      # [LEADING] 成交量异常 - 资金行为先行指标
    # --- 滞后指标 (原有, 仅作确认) ---
    "rsi_divergence",     # [LAGGING] RSI背离
    "macd_cross",         # [LAGGING] MACD金叉死叉
    "ema_breakout",       # [LAGGING] EMA突破
    "bollinger_squeeze",  # [LAGGING] 布林带收窄
    "support_resistance", # [LAGGING] 支撑阻力
    "fibonacci",          # [LAGGING] 斐波那契
    "volume_price_trend", # [LAGGING] 量价趋势
    "trend_pullback",     # [Phase 14.8] 顺大逆小 — ema_200趋势+回调入场
    "timesfm_prediction", # [ML] TimesFM预测方向+置信度
    "kronos_range",      # [ML] Kronos价格区间边界
]

CONFIRMATION_SIGNALS: List[str] = [
    "volume_spike",       # 放量确认
    "rsi_confirm",        # RSI确认
    "macd_histogram",     # MACD柱确认
    "multi_tf_resonance", # 多时间框架共振
    "order_book_depth",   # 订单簿深度
    "cvd_divergence",     # CVD背离
    "timesfm_confidence", # TimesFM预测置信度
]

# v503 P3-2 根治94%种群沉默 (ISS-20260628-002)
# ============================================================================
# 根因 (Explore Agent 深度分析结论):
#   1. generate_random_gene() 完全随机选信号, ~30% 基因 0 个 LEADING 指标
#      → 命中 agent_decision_engine.py L1610 (independent_count==0) 硬过滤
#   2. generate_seed_based_gene() 世界观与信号池语义解耦
#      → 出现 "trend_following + mean_reversion 信号" 的语义矛盾
#   3. min_signal_confluence=(3,4) 与 signal_count 去重逻辑(上限=leading_count+1) 数学矛盾
#      → 即使选满6个主信号, 去重后 signal_count ≤ leading_count+1, 难以达到3
#
# 修复方案 (基于P1-1深度策略知识, 非过拟合):
#   1. 强制 primary_signals 含 ≥2 个 LEADING 指标 (避免 L1610 硬过滤)
#   2. 按 worldview 语义约束信号池 (避免世界观-信号语义矛盾)
#   3. 配合 agent_decision_engine.py 的硬过滤→衰减改造 (P3-2 Fix 2)
#
# 来源:
#   - P1-1 深度策略知识文档 (_v502_p11_deep_strategy_knowledge.md)
#   - ISS-20260628-002 深度根因分析
#   - 用户铁律: "策略要有深度, 要理解底层逻辑, 而非表面指标"
# ============================================================================

# LEADING 指标集合 (与 agent_decision_engine.py LEADING_SIGNALS 保持一致)
# 来源: v3.21 修复滞后指标虚假共振 — 领先指标基于价量关系, 是独立证据
LEADING_SIGNALS_SET: set = {
    "price_momentum_roc",  # ROC价格动量 (无平滑, 直接度量)
    "mfi_divergence",      # MFI资金流量 (价量综合)
    "willr_signal",        # Williams %R (响应快的超买超卖)
    "volume_anomaly",      # 成交量异常 (资金行为先行)
}

# 世界观 → 必备 LEADING 指标映射 (基于指标底层逻辑, 非过拟合)
# 设计原理 (P1-1 深度策略知识):
#   - trend_following/momentum: 价格动量是趋势的直接度量, 成交量异常确认资金进场
#   - mean_reversion/grid_trading: WillR/MFI 是超买超卖指标, 直接驱动反转交易
#   - breakout: 成交量异常确认突破有效性 (假突破通常无成交量配合)
#   - funding_rate_*: 资金费率异常伴随成交量异常 (套利进场)
#   - order_flow_cvd: CVD 背离需成交量确认 (订单流分析本质)
_WORLDVIEW_LEADING_CONSTRAINTS: Dict[str, List[str]] = {
    "trend_following":         ["price_momentum_roc", "volume_anomaly"],
    "mean_reversion":          ["willr_signal", "mfi_divergence"],
    "breakout":                ["volume_anomaly", "price_momentum_roc"],
    "momentum":                ["price_momentum_roc", "volume_anomaly"],
    "value_investing":         ["mfi_divergence", "willr_signal"],
    "safe_margin":             ["willr_signal", "mfi_divergence"],
    "stat_arbitrage":          ["willr_signal", "mfi_divergence"],
    "risk_parity":             ["volume_anomaly", "mfi_divergence"],
    "multi_strategy":          ["price_momentum_roc"],  # 多策略仅强制1个, 保留探索空间
    "narrative_driven":        ["volume_anomaly", "price_momentum_roc"],
    "grid_trading":            ["willr_signal", "mfi_divergence"],
    "funding_rate_arb":        ["volume_anomaly"],
    "perp_market_making":      ["volume_anomaly"],
    "grid_strike":             ["willr_signal", "volume_anomaly"],
    "scalping":                ["price_momentum_roc", "willr_signal"],
    "funding_rate_predictor":  ["volume_anomaly"],
    "order_flow_cvd":          ["volume_anomaly", "price_momentum_roc"],
}


# --- 模块4：出场与风控 ---
EXIT_METHODS: List[str] = [
    "fixed_stop",         # 固定止损
    "atr_trailing",       # ATR跟踪止损
    "parabolic_stop",     # 抛物线止损
    "time_stop",          # 时间止损
    "volatility_stop",    # 波动率止损
    "chandelier_exit",    # 吊灯止损
]

# --- 模块5：仓位与资金管理 ---
POSITION_METHODS: List[str] = [
    "kelly_fraction",     # 凯利公式分数版
    "fixed_risk",         # 固定风险百分比
    "volatility_adjusted",# 波动率调整
    "equal_weight",       # 等权重
    "optimal_f",          # 最优f
    "fixed_size",         # 固定仓位
    "turtle_atr",         # Phase 14.13: 海龟1%风险+ATR自适应止损
]

# ============================================================================
# 基因数据结构
# ============================================================================


class GeneField:
    """基因字段描述符 — 定义每个字段的类型、值域、变异规则"""

    def __init__(
        self,
        name: str,
        field_type: str,  # "choice", "float", "int", "bool", "multi_choice"
        value_range: Any,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.1,
        description: str = "",
    ):
        self.name = name
        self.field_type = field_type
        self.value_range = value_range
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.description = description


# 基因编码表：定义每个基因字段的完整约束
GENE_SCHEMA: Dict[str, List[GeneField]] = {
    "worldview": [
        GeneField("primary", "choice", list(WORLDVIEW_SEEDS.keys()), 0.15,
                  description="主要市场世界观"),
        GeneField("confidence_threshold", "float", (0.35, 0.65), 0.50, 0.03,
                  description="交易信号置信度阈值 (v3.12: 0.65-0.98→0.55-0.85 解决97%零交易)"),
        GeneField("patience", "float", (0.1, 0.5), 0.1, 0.1,
                  description="耐心度：越高越愿意等待完美信号 (v2.38: 1.0→0.5 参数评估显示低耐心盈利 r=-0.09)"),
        GeneField("adaptability", "float", (0.4, 0.6), 0.1, 0.1,
                  description="适应度：越高越容易因市场变化调整策略 (v2.38: 1.0→0.5 参数评估显示低适应度盈利 r=-0.06) (v3.13: →(0.4, 0.6) 调参杠杆d优化: 适应度更低 (d=-0.463))"),
        GeneField("contrarian_bias", "float", (-0.3, 0.05), 0.1, 0.05,
                  description="逆向偏见：正=逆势倾向，负=顺趋势倾向 (v3.13: →(-0.3, 0.05) 调参杠杆d优化: 反向偏好更强 (d=-0.471))"),
    ],
    "market_state": [
        GeneField("trend_detectors", "multi_choice", TREND_DETECTORS, 0.1,
                  description="趋势检测工具"),
        GeneField("volatility_detectors", "multi_choice", VOLATILITY_DETECTORS, 0.1,
                  description="波动检测工具"),
        GeneField("macro_sources", "multi_choice", MACRO_DATA_SOURCES, 0.05,
                  description="宏观数据源"),
        GeneField("regime_memory", "int", (3, 12), 0.1,
                  description="市场状态记忆长度 (v3.12: 3-20→3-12 解决跨周期Sharpe std=2.244)"),
        GeneField("state_confidence", "float", (0.55, 0.80), 0.75, 0.03,
                  description="状态判断置信度需求 (v3.12: 0.65-0.98→0.55-0.80 减少过度自信)"),
    ],
    "entry_logic": [
        GeneField("primary_signals", "multi_choice", PRIMARY_SIGNALS, 0.1,
                  description="主信号工具"),
        GeneField("confirmation_signals", "multi_choice", CONFIRMATION_SIGNALS, 0.1,
                  description="确认信号工具"),
        GeneField("entry_aggressiveness", "float", (0.5, 0.65), 0.1, 0.1,
                  description="入场激进程度：0=保守，1=激进 (v3.13: →(0.5, 0.65) 调参杠杆d优化: 入场更保守 (d=-0.679))"),
        GeneField("min_signal_confluence", "int", (3, 4), 0.1,
                  description="最少信号确认数 (v3.12: 1-5→1-3 避免过度要求信号一致性) (v3.13: →(2, 3) 调参杠杆d优化) (v3.20修复: 回退到(3,4), v3.10.15b实测(2,3)导致胜率0.5%, v319实测0.47%完全吻合, 必须收紧到(3,4)过滤滞后指标虚假共振)"),
        GeneField("signal_decay", "float", (0.5, 1.0), 0.1, 0.05,
                  description="信号衰减系数"),
        # v2.6 基因驱动信号参数（解决决策同质化问题）
        # 来源：2026 GA最佳实践 — 每个Agent应有独特的信号阈值，否则相同基因选择=相同决策
        GeneField("rsi_oversold", "float", (20.0, 40.0), 0.15, 0.05,
                  description="RSI超卖阈值（个性化，默认30）"),
        GeneField("rsi_overbought", "float", (60.0, 75.0), 0.15, 0.05,
                  description="RSI超买阈值（个性化，默认70） (v3.13: →(60.0, 75.0) 调参杠杆d优化: 超买阈值更低 (d=-0.557))"),
        GeneField("macd_sensitivity", "float", (0.001, 0.01), 0.15, 0.002,
                  description="MACD信号敏感度（个性化，默认0.005）"),
        GeneField("ema_breakout_pct", "float", (0.01, 0.03), 0.15, 0.005,
                  description="EMA突破百分比阈值（个性化，默认0.01）(v2.42.2: 0.005-0.03→0.01-0.03 breakout基因进化结果, 突破型需要更大的突破阈值过滤假突破, 真实180天3交易对回测avg_score=5.07夺冠)"),
        GeneField("signal_weight_bias", "float", (0.7, 1.3), 0.15, 0.2,
                  description="信号权重偏好（个性化，1.0=标准） (Phase 14.16p: 0.85-1.1→0.7-1.3 扩大signal_weight_bias_norm维度值域, 当前std=0.0329过窄) (v3.13: →(0.85, 1.1) 已废弃)"),
        # Phase 14.16r: 市场上下文信号权重（资金费率/OI/情绪/相关性）
        GeneField("funding_rate_weight", "float", (-1.0, 1.0), 0.0, 0.3,
                  description="Phase 14.16r: 资金费率信号权重（-1~1，0=禁用）"),
        GeneField("oi_weight", "float", (-1.0, 1.0), 0.0, 0.3,
                  description="Phase 14.16r: 未平仓合约信号权重（-1~1，0=禁用）"),
        GeneField("sentiment_weight", "float", (-1.0, 1.0), 0.0, 0.3,
                  description="Phase 14.16r: 情绪指数信号权重（-1~1，0=禁用）"),
        GeneField("correlation_penalty", "float", (0.0, 1.0), 0.3, 0.2,
                  description="Phase 14.16r: 持仓相关性惩罚强度（0=禁用，1=最强）"),
    ],
    "exit_risk": [
        GeneField("stop_loss_pct", "float", (0.015, 0.025), 0.1, 0.005,
                  description="止损百分比 (v3.10.21: 与INTRADAY_GENE_OVERRIDES一致, 2-3%止损) (v2.42.2: →0.015-0.025 breakout基因进化最优=0.018, 真实180天回测3交易对avg_score=5.07, 收紧下限允许更紧止损)"),
        GeneField("take_profit_pct", "float", (0.03, 0.05), 0.1, 0.005,  # Phase 14.8: 1.2-2.5%→3-5% (盈亏比2:1)
                  description="止盈百分比 (v503 Fix 30: 0.03-0.05→0.012-0.025 P60诊断TP 100%不可达(0/121), MAE=0-1.7%<<TP=3-4%. 1.2-2.5%在24h内0.5-1.0σ触发概率30-50%. 配合ATR cap=1.5×ATR, R:R=0.6-1.25) (v503 Fix 23: 已废弃0.03-0.05) (v503 Fix 20/21: 已废弃0.04-0.06) (v3.10.21: 已废弃)"),
        GeneField("trailing_stop", "bool", (True, False), 0.1,
                  description="是否使用跟踪止损"),
        GeneField("trailing_distance_pct", "float", (0.005, 0.01), 0.1, 0.002,
                  description="跟踪止损距离 (v503 Fix 23: 0.01-0.02→0.005-0.01 P50显示trailing_stop 0%触发率, 因盈利单max_pnl普遍<2%无法启动trailing. 0.5-1%更易启动, max_pnl>0.5%即锁定利润. trailing为TP的15-25%, 平衡利润锁定与趋势发展) (v503 Fix 20: 已废弃0.01-0.02)"),
        GeneField("max_hold_hours", "int", (12, 24), 0.1,
                  description="最大持仓时间(小时) (v503 Fix 20/21: 3-8→12-24 趋势需要时间发展, 24h预期移动3.75%>TP中位数5%在强趋势中可达, 仍为日内不持仓过夜) (v3.23: 12-20→3-8 日内交易硬约束, 已废弃)"),
        GeneField("max_drawdown_pct", "float", (0.03, 0.15), 0.05, 0.02,
                  description="最大回撤容忍度 (Phase 14.16p: 0.05-0.12→0.03-0.15 扩大risk_tolerance维度值域, 当前std=0.0169过窄)"),
        GeneField("circuit_breaker_losses", "int", (2, 3), 0.1,
                  description="连续亏损触发熔断次数 (v3.10.21: 2-5→2-4 收紧) (v2.42.2: →2-3 用户硬要求实盘连续亏损≤3次, breakout基因进化最优=2, 收紧上限到3彻底杜绝系统性亏损风险)"),
        GeneField("exit_method", "choice", EXIT_METHODS, 0.1,
                  description="出场方式"),
    ],
    "position_sizing": [
        GeneField("method", "choice", POSITION_METHODS, 0.1,
                  description="仓位管理方法"),
        GeneField("base_risk_per_trade", "float", (0.01, 0.03), 0.1, 0.005,
                  description="每笔基础风险 (v2.38: 0.005-0.03→0.01-0.05 参数评估显示高仓位盈利 r=+0.05) (v2.42.2: →0.01-0.03 breakout基因进化最优=0.02, 收紧上限符合用户硬要求'单次交易最大亏损控制在本金2%以内', 杜绝过度杠杆)"),
        GeneField("max_position_pct", "float", (0.05, 0.30), 0.1, 0.05,
                  description="最大仓位占比 (v3.12: 0.05-0.50→0.05-0.30 分散化)"),
        GeneField("pyramid", "bool", (True, False), 0.1,
                  description="是否加仓"),
        GeneField("pyramid_scale", "float", (0.3, 0.8), 0.1, 0.05,
                  description="加仓比例"),
        GeneField("correlation_aware", "bool", (True, False), 0.05,
                  description="是否考虑持仓相关性"),
        GeneField("max_correlated_positions", "int", (1, 5), 0.1,
                  description="最大关联持仓数"),
        GeneField("atr_stop_multiplier", "float", (1.0, 4.0), 0.1, 0.05,
                  description="Phase 14.13: ATR止损乘数 (1.0-4.0, 默认2.0, 海龟法则N倍ATR止损)"),
    ],
}


@dataclass(slots=True)
class AgentGene:
    """子Agent的完整基因编码

    基因 = 五个认知模块的完整参数集合。
    每个子Agent拥有唯一的基因，决定其交易行为的所有方面。
    """

    agent_id: str = field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    # 模块1：市场世界观
    worldview_primary: str = "trend_following"
    worldview_confidence_threshold: float = 0.50  # Phase 14.19: 0.65→0.50 (降低阈值促进早期交易)
    worldview_patience: float = 0.5
    worldview_adaptability: float = 0.5
    worldview_contrarian_bias: float = 0.0

    # 模块2：市场状态判断
    market_state_trend_detectors: List[str] = field(default_factory=lambda: ["ema_cross", "adx"])
    market_state_volatility_detectors: List[str] = field(default_factory=lambda: ["bollinger", "atr"])
    market_state_macro_sources: List[str] = field(default_factory=lambda: ["none"])
    market_state_regime_memory: int = 5
    market_state_state_confidence: float = 0.6

    # 模块3：入场决策逻辑
    entry_logic_primary_signals: List[str] = field(default_factory=lambda: ["rsi_divergence", "macd_cross"])
    entry_logic_confirmation_signals: List[str] = field(default_factory=lambda: ["volume_spike"])
    entry_logic_entry_aggressiveness: float = 0.5
    entry_logic_min_signal_confluence: int = 2
    entry_logic_signal_decay: float = 0.8
    # v2.6 基因驱动信号参数（解决决策同质化问题）
    entry_logic_rsi_oversold: float = 30.0
    entry_logic_rsi_overbought: float = 70.0
    entry_logic_macd_sensitivity: float = 0.005
    entry_logic_ema_breakout_pct: float = 0.01
    entry_logic_signal_weight_bias: float = 1.0
    # Phase 14.16r: 市场上下文信号权重（资金费率/OI/情绪/相关性）
    entry_logic_funding_rate_weight: float = 0.0
    entry_logic_oi_weight: float = 0.0
    entry_logic_sentiment_weight: float = 0.0
    entry_logic_correlation_penalty: float = 0.3

    # 模块4：出场与风控
    exit_risk_stop_loss_pct: float = 0.05
    exit_risk_take_profit_pct: float = 0.04  # Phase 14.8: 0.15→0.04 (匹配新范围3-5%)
    exit_risk_trailing_stop: bool = True
    exit_risk_trailing_distance_pct: float = 0.015  # v3.12.1: 0.03→0.015 修复默认值超出新范围(0.005,0.025)
    exit_risk_max_hold_hours: int = 48
    exit_risk_max_drawdown_pct: float = 0.20
    exit_risk_circuit_breaker_losses: int = 3
    exit_risk_exit_method: str = "atr_trailing"

    # 模块5：仓位与资金管理
    position_sizing_method: str = "kelly_fraction"
    position_sizing_base_risk_per_trade: float = 0.02
    position_sizing_max_position_pct: float = 0.25
    position_sizing_pyramid: bool = False
    position_sizing_pyramid_scale: float = 0.5
    position_sizing_correlation_aware: bool = True
    position_sizing_max_correlated_positions: int = 3
    position_sizing_atr_stop_multiplier: float = 2.0  # Phase 14.13: ATR止损乘数 (海龟N倍ATR)

    # 运行时状态（不参与遗传）
    fitness_score: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_realized: float = 0.0
    survival_rounds: int = 0
    style_label: str = ""  # 活下来后回溯贴的标签

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于JSON持久化）"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentGene":
        """从字典反序列化"""
        # 过滤掉不在dataclass中的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def get_module(self, module_name: str) -> Dict[str, Any]:
        """提取指定模块的参数字典"""
        prefix = f"{module_name}_"
        result = {}
        for field_name in self.__dataclass_fields__:
            if field_name.startswith(prefix):
                key = field_name[len(prefix):]
                result[key] = getattr(self, field_name)
        return result

    def get_worldview_seed(self) -> Dict[str, Any]:
        """获取市场世界观种子数据"""
        if self.worldview_primary in WORLDVIEW_SEEDS:
            return WORLDVIEW_SEEDS[self.worldview_primary]
        return {}

    def compute_style_vector(self) -> Dict[str, float]:
        """计算策略风格向量（用于聚类和相似度计算）

        返回一个归一化的风格特征向量，用于：
        - 种群多样性评估
        - 子Agent相似度计算
        - 回溯贴标签

        v2.18 P2a-rollback: 回退严格归一化，恢复v2.17.2原计算
        原因：v2.18 P2a严格归一化后 GT-Score max 从 125.5 下降到 68.1 (-45.74%)
              diversity 提升91.67%但 wf_ratio 下降9.37%，过度分散导致收敛不足
              违反"标准只升不降"铁律，必须回退
        教训：Novelty Search的k-NN距离分布对归一化方式极敏感，
              原始未严格归一化的计算方式实际上是经过进化验证的有效配置
        """
        return {
            "trend_strength": self.worldview_contrarian_bias * -1 + 0.5,
            "patience": self.worldview_patience,
            "aggressiveness": self.entry_logic_entry_aggressiveness,
            "risk_tolerance": 1.0 - self.exit_risk_max_drawdown_pct,
            "signal_complexity": min(1.0, self.entry_logic_min_signal_confluence / 5.0),  # Phase 14.16n: 3.0→5.0 修复clamp bug (confluence=3-4被clamp到1.0导致std=0)
            "holding_duration": min(1.0, self.exit_risk_max_hold_hours / 48.0),  # v3.12: 720→48 配合范围收紧
            "position_concentration": self.position_sizing_max_position_pct,
            "adaptability": self.worldview_adaptability,
            # v2.6: 信号参数个性化维度（解决决策同质化）
            "rsi_threshold_width": (self.entry_logic_rsi_overbought - self.entry_logic_rsi_oversold) / 60.0,
            "macd_sensitivity_norm": self.entry_logic_macd_sensitivity / 0.01,
            "signal_weight_bias_norm": self.entry_logic_signal_weight_bias / 2.0,
        }


# ============================================================================
# 随机基因生成
# ============================================================================


# ============================================================================
# 种子库驱动的基因生成 (v2.18 P1c)
# ============================================================================

# v2.18: holding_period → max_hold_hours 映射表
# 来源：WORLDVIEW_SEEDS中holding_period语义到exit_risk.max_hold_hours值域的映射
_HOLDING_PERIOD_TO_HOURS: Dict[str, Tuple[int, int]] = {
    "very_short": (1, 24),       # 1小时-1天
    "short": (24, 72),           # 1-3天
    "medium": (72, 168),         # 3-7天
    "long": (168, 720),          # 7-30天
    "very_long": (720, 720),     # 30天+
    "adaptive": (24, 360),       # 自适应：宽范围
}

# v2.18: max_trades_per_day → entry_aggressiveness 映射
# 来源：高频策略需要更激进入场；低频策略需要更保守入场
_TRADES_PER_DAY_TO_AGGRO: List[Tuple[int, Tuple[float, float]]] = [
    (1, (0.1, 0.3)),    # ≤1/天 → 极保守
    (3, (0.2, 0.5)),    # ≤3/天 → 保守
    (10, (0.3, 0.6)),   # ≤10/天 → 中等
    (50, (0.5, 0.8)),   # ≤50/天 → 激进
    (500, (0.7, 1.0)),  # >50/天 → 极激进
]


# ============================================================================
# 日内交易场景基因生成 (v2.35 — 数字货币日内交易调参)
# ============================================================================
#
# 来源:
#   - Hermes 沙盘 500轮进化数据分析(2026-06-27): 97% agent零交易根因
#   - ProTraderDaily 2026: 止损2-3%, 单笔风险1%, RRR 1:2
#   - LiveVolatile 2026: BTC 30日实现波动率68.2%, 5min-1h最优时间框架
#   - DigitalNinjaSystems 2026: 1%规则, ATR 1.5x止损
#   - CoinDesk 2026: 日内 dFactor=0.75 + 2%止损 + 10%止盈优化
#
# 500轮进化诊断关键发现:
#   1. 97% agent零交易 — 信号阈值过高(GENE_SCHEMA默认范围过宽)
#   2. 4% agent亏损 — max_hold_hours=(1,720)导致日内策略持仓过久
#   3. take_profit_pct=(0.02,0.50)上限过大,日内50%止盈不现实
#   4. 16个关键亏钱参数差异>10%(详见 _analyze_loss_cause.py 输出)
#
# 修复策略:
#   - INTRADAY_GENE_OVERRIDES 覆盖 GENE_SCHEMA 的过宽范围
#   - generate_intraday_gene() 生成符合日内硬约束的基因
#   - 高胜率世界观种子加权(EVENT_DRIVEN 40% / LIQUIDITY 20% / FUNDING 15% / STAT_ARB 15% / TRANSFORMER 10%)
#   - 500轮进化数据验证: 这些世界观胜率66.7%-100%(详见 _analyze_win_vs_loss.py)


# 日内交易场景的参数范围覆盖 — 仅覆盖需要收紧的字段,其余沿用 GENE_SCHEMA
#
# v3.19 (2026-06-27 23:30) MCP-AGENT 基于深度亏损基因分析调整
#   数据: v316 (38代3800agent) + v318 (5代600agent) = 4600样本
#   方法: Pearson/Spearman相关性 + PCA + 随机森林 + SHAP + t-test
#   关键发现:
#     1. 88.6% agent零交易 — 信号阈值过高
#     2. exit_risk_max_hold_hours 是PnL的Top1影响因素 (SHAP=2.27)
#     3. 持仓时间越长 → 交易越多 → 亏损越多 (r=+0.279 trades, r=-0.177 pnl)
#     4. 5个基因在盈利组vs亏损组显著差异(P<0.05, Cohen's d>0.4)
#   调整:
#     - confidence_threshold: (0.65,0.80) → (0.55,0.75) 降阈值解决零交易
#     - patience: (0.40,0.60) → (0.50,0.65) 盈利组0.543>亏损0.472 (d=+0.677)
#     - state_confidence: (0.55,0.80) → (0.55,0.70) 盈利组0.648<亏损0.682 (d=-0.541)
#     - max_hold_hours: (18,24) → (12,20) 缩短持仓减少频繁交易亏损
#     - 新增 rsi_oversold: (25,30) 盈利组28.2<亏损31.1 (d=-0.551)
#     - 新增 pyramid_scale: (0.50,0.65) 盈利组0.556>亏损0.498 (d=+0.416)
INTRADAY_GENE_OVERRIDES: Dict[str, Dict[str, Tuple]] = {
    "worldview": {
        # v3.19 修正: (0.65,0.80)→(0.55,0.75)
        #   根因: 4600样本分析显示88.6% agent零交易, 阈值过高
        #   随机森林importance=0.107 (Top2影响因素)
        #   降阈值让更多信号触发, 解决零交易问题
        "confidence_threshold": (0.35, 0.55),
        # v3.19 修正: (0.40,0.60)→(0.50,0.65)
        #   盈利组patience=0.543 vs 亏损组=0.472 (Cohen's d=+0.677, P<0.05)
        #   中等效应量, 盈利组patience更高(更耐心等待好信号)
        "patience": (0.50, 0.65),
        # 保留原范围
        "adaptability": (0.4, 0.8),
    },
    "market_state": {
        # v3.10.15 LOSS-HUNTER: 基于 D3 调参杠杆 #1 (188% 相对差异)
        # 亏损 agent regime_memory=11.90 vs 零交易=6.33
        # 太长→追滞后模式→反向进场→亏钱; 太短→无上下文→不交易
        # 收紧到 (4, 8) 强制适中记忆长度
        "regime_memory": (4, 8),  # type: ignore (int 范围)
        # v3.19 修正: (0.55,0.80)→(0.55,0.70)
        #   盈利组state_confidence=0.648 vs 亏损组=0.682 (Cohen's d=-0.541, P<0.05)
        #   盈利组confidence更低 — 不要过度自信, 保持对市场的谦逊
        #   随机森林importance=0.078 (Top3影响因素)
        "state_confidence": (0.55, 0.70),
    },
    "entry_logic": {
        # v3.10.15b 修正: v311 实测 (2,3) 导致弱信号通过, 胜率 0.5%
        # 收紧到 (3, 4) 要求更多信号一致性, 过滤噪音
        # v3.19 验证: 盈利组3.67 vs 亏损组3.18 (Cohen's d=+0.514, P<0.05) — 当前范围正确
        "min_signal_confluence": (3, 4),  # type: ignore (int 范围)
        # 亏损 aggressiveness=0.56 vs 零交易=0.48 (17.2% 差异)
        # 收紧到 (0.55, 0.75) 敢于进场
        "entry_aggressiveness": (0.55, 0.75),
        # 亏损 macd_sensitivity=0.0052 vs 零交易=0.0058
        # 提高灵敏度范围 (0.005, 0.012)
        "macd_sensitivity": (0.005, 0.012),
        # 收紧突破阈值
        "ema_breakout_pct": (0.005, 0.015),
        # 加快信号衰减(日内时效)
        "signal_decay": (0.5, 0.7),
        # v3.19 新增: rsi_oversold范围
        #   盈利组rsi_oversold=28.2 vs 亏损组=31.1 (Cohen's d=-0.551, P<0.05)
        #   盈利组RSI阈值更低 — 更严格的超卖标准, 只在真正超卖时进场
        "rsi_oversold": (25.0, 30.0),
    },
    "exit_risk": {
        # v503 Fix 20: 从剥头皮转为趋势跟随 — 数学上根治 38.3% 胜率下必亏
        # 根因 (P47 诊断): R:R=0.55 (scalping设计) 需要胜率≥64.8%才能盈利
        #   但趋势策略固有胜率只有 30-40% (用户项目记忆确认:
        #   "Pure trend strategies have inherent win rate limitations (30-40%)")
        #   实测 38.3% << 64.8% → 数学上必然亏损, EV=-0.136%/笔
        # 深层逻辑 (用户铁律: "策略要有深度, 要理解底层逻辑"):
        #   1. 趋势策略的本质是"截断亏损, 让利润奔跑" — 低胜率+高R:R
        #   2. 经典趋势跟随 (Turtle/AQR): 胜率30-40%, R:R=2.5-3.0, 靠大赢补小亏
        #   3. 剥头皮 (R:R<1) 需要高胜率(60%+) — 但趋势信号做不到
        #   4. 强行用剥头皮参数配趋势信号 = 数学上必亏 = "模拟牛逼实盘亏钱"的根因
        # 修复 (基于趋势跟随数学原理, 非过拟合):
        #   - SL 保持 1.5-2.5% (趋势策略的"截断亏损")
        #   - TP 0.8-1.5%→4-6% (趋势策略的"让利润奔跑", R:R=2.5)
        #   - max_hold 5-10h→12-24h (趋势需要时间发展, 24h预期移动3.75%>TP)
        #   - trailing 0.2-0.6%→1.0-2.0% (配合更大TP, 避免过早平仓)
        # 数学验证:
        #   R:R = 3/2.5=1.2 到 5/1.5=3.3, 中位数~2.0
        #   盈亏平衡胜率 = 1/(1+2.0) = 33.3%
        #   趋势策略预期胜率 = 30-40% ≈ 33.3% → 勉强正期望
        #   EV = 0.35×4% - 0.65×2% = +0.10%/笔 ✅ (但边际较薄)
        #   24h预期移动 = 0.766%×√24 = 3.75%, TP中位数4%在趋势中可达
        # v503 Fix 23: TP 4-6%→3-5% (P50显示TP 4-6%触发率仅8.1%)
        #   TP 3-5%在24h需0.8-1.3σ触发概率20-35% (提升2-3倍)
        #   R:R=3-5%/1.5-2.5%=1.5-2.5 仍在cap=3.0内
        # 来源: Turtle Trading (Curtis Faith 2008) + AQR Trend Following (Moskowitz 2015)
        # 铁律: "杜绝模拟牛逼,实盘亏钱" — R:R与胜率不匹配 = 必亏
        "stop_loss_pct": (0.015, 0.025),
        "take_profit_pct": (0.012, 0.025),  # v503 Fix 30: 0.03-0.05→0.012-0.025 (P60 TP 100%不可达)
        # v503 Fix 20: max_hold 5-10→12-24 (趋势需要时间发展)
        #   12h预期移动 = 0.766%×√12 = 2.65%
        #   24h预期移动 = 0.766%×√24 = 3.75% → 强趋势中TP=3-5%可达
        #   24h仍为日内 (不持仓过夜), 符合日内交易本质
        "max_hold_hours": (12, 24),  # type: ignore (int 范围) v503 Fix 20: 5-10→12-24
        # 亏损 max_dd=0.20 vs 盈利=0.22, 但 0.40 上限过大
        # 收紧到 (0.05, 0.12) 单代回撤 12% 熔断
        "max_drawdown_pct": (0.05, 0.12),
        # 收紧连亏熔断 (2,4) → 连亏3次熔断
        "circuit_breaker_losses": (2, 4),  # type: ignore (int 范围)
        # v503 Fix 23: trailing_distance 1-2%→0.5-1% (P50显示trailing_stop 0%触发率)
        #   P50根因: 盈利单max_pnl普遍<2%, trailing_distance=1-2%时启动条件max_pnl>trailing_distance无法满足
        #   修复: 0.5-1%更易启动, max_pnl>0.5%即锁定利润
        #   trailing为TP的15-25%: 3%×0.17=0.5%, 5%×0.2=1% (平衡利润锁定与趋势发展)
        "trailing_distance_pct": (0.005, 0.01),
    },
    "position_sizing": {
        # 亏损 base_risk=0.06 vs 零交易=0.053 (12.4%差异)
        # DigitalNinjaSystems 2026: 1%规则
        # 收紧到 (0.005, 0.02) 强制 1.5% 单笔风险
        "base_risk_per_trade": (0.005, 0.02),
        # 亏损 max_position=0.24 vs 盈利=0.26, 但 0.50 上限过大
        # 收紧到 (0.05, 0.20) 单仓 20% 上限
        "max_position_pct": (0.05, 0.20),
        # v3.19 新增: pyramid_scale范围
        #   盈利组pyramid_scale=0.556 vs 亏损组=0.498 (Cohen's d=+0.416, P<0.05)
        #   盈利组加仓比例更高 — 让赢利仓位加仓跑趋势
        #   SHAP分析: pyramid_scale与交易次数负相关 (r=-0.141)
        #     说明: 加仓比例高→单次交易规模大→减少频繁开仓
        "pyramid_scale": (0.50, 0.65),
    },
}


# 日内交易高胜率世界观种子配比 (来自 500轮进化 winners vs losers 数据)
# 来源: _analyze_win_vs_loss.py 输出 (46代 2300 agents)
# 排序逻辑: win_rate% 优先, 样本量次之
#
# v3.19 (2026-06-27 23:35) MCP-AGENT 冗余模块清理
#   数据: v316+v318 共4600agent 冗余模块检测
#   发现: 42个策略推荐移除 (26个三重命中+2个高亏损+11个100%零交易)
#   移除: ml_liquidation_predictor (avg_pnl=-10.00) + mev_sandwich_arbitrage (-7.76)
#   保留: 7个核心策略 (实际进化中有盈利或潜力) + 3个备份策略
#   效果: 策略池从50+个精简到10个, 聚焦高胜率策略, 减少零交易问题
INTRADAY_WORLDVIEW_WEIGHTS: List[Tuple[str, float]] = [
    # (worldview, 权重) — 总和=1.0
    # v3.10.14 LOSS-HUNTER 修正: 基于实际进化数据(9代900agent)调整权重
    # 原权重基于回测"胜率",但实际进化中 event_driven 15亏0盈 = "模拟牛逼实盘亏钱"
    ("liquidity_adjusted_strategy", 0.30),  # 实际6盈 net_score=+42.34 — Amihud+Kyle+VPIN (最赚钱)
    ("event_driven_trading",        0.10),  # 实际15亏0盈 net_score=-13.51 (原40%→10%, 回测≠实盘)
    ("funding_rate_predictor",      0.10),  # 实际4亏0盈 net_score=-3.60 (原15%→10%)
    ("stat_arb_pairs",              0.10),  # 实际10亏0盈 net_score=-9.01 (原15%→10%)
    ("transformer_decision_policy", 0.10),  # 实际3亏0盈 net_score=-2.70 (保持10%)
    ("on_chain_whale_tracker",      0.10),  # 实际1盈 net_score=+5.71 (新增, 有盈利)
    ("marl_portfolio_management",   0.10),  # 实际1盈 net_score=+11.11 (新增, 有盈利)
    # v3.19 新增3个备份策略 (低频但100%零交易根因是feeder缺失, INTRADAY-GUARD已修复)
    ("trend_following",             0.04),  # 655样本, 97.7%零交易但样本量最大, 修复后潜力大
    ("smart_money_footprint",       0.03),  # 14样本, 85.7%零交易, 跟踪聪明钱
    ("order_flow_cvd",              0.03),  # 8样本, 87.5%零交易, 订单流CVD
]
# v3.19 移除清单 (42个冗余策略, 不再分配权重)
# 完整移除清单见 _v319_redundant_modules.py 输出
# v3.19 移除清单 (42个冗余策略, 不再分配权重)
# Phase E Batch 6 (2026-07-15): 原INTRADAY_REMOVED_STRATEGIES集合从未被任何代码引用(死代码),
# 违反铁律零遗漏。已将其转换为注释, 保留v3.19决策历史记录。
# 移除原因分类:
#   - 高亏损策略(2个): ml_liquidation_predictor, mev_sandwich_arbitrage
#   - 100%零交易策略(11个): vanna_volga_arbitrage, causal_inference_trading, regime_ensemble,
#     cross_sectional_momentum, mean_reversion, multi_strategy, basis_arbitrage, skew_trading,
#     diffusion_forecast_v1_1, meta_learning_portfolio, cross_market_arbitrage
#   - 三重命中策略(13个): volatility_surface_arb, seasonality_cycle, marl_portfolio_v1_1, breakout,
#     scalping, rl_market_making, liquidation_hunter, risk_parity_signal, narrative_driven,
#     adversarial_rl_trading, stat_arbitrage, jit_liquidity_mev, grid_trading, diffusion_forecast_trading
#   - 低影响高零交易策略(16个): causal_inference_v1_1, cross_chain_arbitrage, dispersion_trading,
#     funding_rate_arb, gnn_cross_asset_relationship, mean_reversion_ml, momentum, order_flow_cvd_backup,
#     perp_market_making, risk_parity, sentiment_driven_trading, value_investing, volatility_arbitrage,
#     term_structure_trading_backup, microstructure_alpha_trading, cex_dex_flash_arbitrage,
#     triangular_arbitrage, cppi_portfolio_insurance, safe_margin, liquidation_cascade,
#     transformer_decision_v1_1, gnn_cross_asset_v1_1, gamma_scalping, vrp_harvesting,
#     macro_liquidity_cycle, llm_agent_trading
# 这些策略不在WORLDVIEW_SEEDS中, 不会被StrategyRegistry实例化。
# pair_trading_strategy 不在此移除清单中, 已通过 evolution_loop.py L2221 直接实例化接入主干。


# ============================================================================
# 基因交叉 (Crossover)
# ============================================================================


# ============================================================================
# 基因相似度
# ============================================================================


# ============================================================================
# 基因序列化
# ============================================================================










# ============================================================================
# v499 Phase 2: GeneCodec 类 — key_params 级别的基因操作
# 来源: v499-thorough-resolution-execution.md Phase 2 Task 2.2
# 目的: 打通 GA优化 → key_params → **kwargs → 策略实例 完整链路
# 原问题: 前11个经典种子无key_params, GA无法优化这些策略的任何参数
#         进化退化为随机搜索, 无法定向优化策略参数
# 设计: 5个静态方法, 与现有 crossover/mutate 函数互补
# ============================================================================


class GeneCodec:
    """基因编解码器 — key_params 级别的基因操作 v499

    与现有 GENE_SCHEMA 级别的 crossover/mutate 函数互补:
      - GENE_SCHEMA: 粗粒度(模块级), 控制 entry/exit/risk 等模块参数
      - GeneCodec: 细粒度(策略级), 控制 key_params 中的策略专属参数

    使用场景:
      1. GA优化: 从种群中提取key_params → 评估 → 选择 → 变异/交叉 → 注回
      2. SHAP分析: 将key_params编码为特征向量 → 重要性评估
      3. 策略实例化: get_key_params → **kwargs → StrategyClass(**key_params)

    依赖:
      - WORLDVIEW_SEEDS: 提供每个worldview的key_params定义
      - AgentGene: 提供gene.worldview和gene.key_params_override
    """

    @staticmethod
    def get_key_params(gene: "AgentGene") -> Dict[str, Any]:
        """从AgentGene获取key_params (含override)

        优先级: gene.key_params_override > WORLDVIEW_SEEDS[worldview].key_params

        Args:
            gene: AgentGene实例

        Returns:
            key_params字典, 例如 {"FAST_EMA": 20, "SLOW_EMA": 50, ...}
            如果worldview不在WORLDVIEW_SEEDS中或无key_params, 返回空dict
        """
        if not hasattr(gene, "worldview_primary") or not gene.worldview_primary:
            return {}

        seed = WORLDVIEW_SEEDS.get(gene.worldview_primary, {})
        base_params = seed.get("key_params", {})

        if not base_params:
            return {}

        result = {}
        for param_name, param_def in base_params.items():
            if not isinstance(param_def, dict):
                result[param_name] = param_def
                continue
            result[param_name] = param_def.get("default")

        override = getattr(gene, "key_params_override", None)
        if override and isinstance(override, dict):
            result.update(override)

        return result

    @staticmethod
    def mutate_key_params(
        gene: "AgentGene",
        mutation_rate: float = 0.15,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """变异key_params (高斯扰动数值型, 随机替换分类型)

        Args:
            gene: AgentGene实例
            mutation_rate: 每个参数的变异概率 (0-1)
            rng: 可选的随机数生成器 (用于确定性进化)

        Returns:
            变异后的key_params字典 (新对象, 不修改原gene)
        """
        r = rng if rng is not None else random
        current_params = GeneCodec.get_key_params(gene)
        if not current_params:
            return {}

        seed = WORLDVIEW_SEEDS.get(gene.worldview_primary, {})
        param_defs = seed.get("key_params", {})

        mutated = {}
        for param_name, current_value in current_params.items():
            if r.random() < mutation_rate and param_name in param_defs:
                param_def = param_defs[param_name]
                if isinstance(param_def, dict):
                    p_type = param_def.get("type", "float")
                    p_range = param_def.get("range", [])

                    if p_type == "int" and len(p_range) == 2:
                        low, high = p_range
                        span = high - low
                        delta = int(r.gauss(0, span * 0.1))
                        new_val = max(low, min(high, current_value + delta))
                        mutated[param_name] = new_val
                    elif p_type == "float" and len(p_range) == 2:
                        low, high = p_range
                        span = high - low
                        delta = r.gauss(0, span * 0.1)
                        new_val = max(low, min(high, current_value + delta))
                        mutated[param_name] = round(new_val, 6)
                    elif p_type == "bool":
                        mutated[param_name] = not current_value
                    elif p_range:
                        mutated[param_name] = r.choice(p_range)
                    else:
                        mutated[param_name] = current_value
                else:
                    mutated[param_name] = current_value
            else:
                mutated[param_name] = current_value

        return mutated

    @staticmethod
    def crossover_key_params(
        parent_a: "AgentGene",
        parent_b: "AgentGene",
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """交叉两个父代的key_params (uniform crossover)

        每个参数独立从父代A或B继承 (50%概率)

        Args:
            parent_a: 父代A
            parent_b: 父代B
            rng: 可选的随机数生成器

        Returns:
            交叉后的key_params字典
        """
        r = rng if rng is not None else random
        params_a = GeneCodec.get_key_params(parent_a)
        params_b = GeneCodec.get_key_params(parent_b)

        if not params_a and not params_b:
            return {}
        if not params_a:
            return params_b.copy()
        if not params_b:
            return params_a.copy()

        all_keys = set(params_a.keys()) | set(params_b.keys())
        child_params = {}
        for key in all_keys:
            if key in params_a and key in params_b:
                child_params[key] = params_a[key] if r.random() < 0.5 else params_b[key]
            elif key in params_a:
                child_params[key] = params_a[key]
            else:
                child_params[key] = params_b[key]

        return child_params

    @staticmethod
    def encode(gene: "AgentGene") -> List[float]:
        """将AgentGene的key_params编码为数值向量 (用于NSGA-III/SHAP)

        编码规则:
          - int/float: 直接映射为浮点数
          - bool: True→1.0, False→0.0
          - 其他类型: 跳过 (不参与向量编码)

        Args:
            gene: AgentGene实例

        Returns:
            浮点数向量, 长度=数值型key_params数量
        """
        params = GeneCodec.get_key_params(gene)
        vector = []
        for value in params.values():
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
        return vector

    @staticmethod
    def decode(
        gene: "AgentGene",
        vector: List[float],
    ) -> Dict[str, Any]:
        """将数值向量解码为key_params (encode的逆操作)

        Args:
            gene: 提供worldview和key_params定义模板
            vector: 数值向量 (来自GA优化或SHAP分析)

        Returns:
            key_params字典, 数值已根据param_def的range进行clamp
        """
        if not hasattr(gene, "worldview_primary") or not gene.worldview_primary:
            return {}

        seed = WORLDVIEW_SEEDS.get(gene.worldview_primary, {})
        param_defs = seed.get("key_params", {})
        if not param_defs:
            return {}

        result = {}
        vec_idx = 0
        for param_name, param_def in param_defs.items():
            if not isinstance(param_def, dict):
                continue
            p_type = param_def.get("type", "float")
            p_range = param_def.get("range", [])

            if p_type in ("int", "float") and vec_idx < len(vector):
                raw_val = vector[vec_idx]
                vec_idx += 1

                if len(p_range) == 2:
                    low, high = p_range
                    raw_val = max(low, min(high, raw_val))

                if p_type == "int":
                    result[param_name] = int(round(raw_val))
                else:
                    result[param_name] = float(raw_val)
            elif p_type == "bool":
                result[param_name] = bool(vector[vec_idx] >= 0.5) if vec_idx < len(vector) else param_def.get("default", False)
                vec_idx += 1

        return result

# Phase 7.11: 基因生成函数已迁移到 gene_generation.py (向后兼容 re-export)

# Phase 7.11b: 进化操作函数已迁移到 gene_evolution.py (向后兼容 re-export)

# Phase 7.11c: Gene IO functions migrated to gene_io.py (backward-compatible re-export)
