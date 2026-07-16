# -*- coding: utf-8 -*-
"""
策略注册表与动态加载器 (Strategy Registry & Dynamic Loader)

激活 gene_codec.WORLDVIEW_SEEDS 中 35 个具体策略类 + 12 个抽象世界观的通用适配器。
借鉴 freqtrade StrategyResolver + nautilus Trader Strategy + qlib generate_trade_decision。

设计原则:
  1. 不破坏现有 35 个策略类的接口（适配器模式，非继承）
  2. 基因层与策略层解耦（gene 决定用哪个策略，策略参数来自 key_params）
  3. 统一信号结构 Signal（收敛 9 种异构返回格式）
  4. 模块缓存避免重复 import（进化场景下数千次实例化）
  5. 优雅降级（加载失败 → fallback 到 GeneDrivenStrategy）

用法:
    from sandbox_trading.strategy_registry import StrategyRegistry, MarketContext

    registry = StrategyRegistry()
    strategy = registry.load("funding_rate_arb", gene, context_deps={"spot_engine": eng})
    signal = strategy.update(market_ctx)
    if signal.direction != 0 and signal.confidence > 0.5:
        ...
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger("hermes.strategy_registry")

# Phase 7.13: Base types extracted to strategy_types.py (imported at top because
# _StrategyAdapter(IStrategy) at class-definition time requires IStrategy)
from .strategy_types import (
    Direction,
    Signal,
    MarketContext,
    IStrategy,
)

# ============================================================================
# 接口版本（借鉴 freqtrade INTERFACE_VERSION）
# ============================================================================



# ============================================================================
# 统一数据结构（收敛异构 sensor/strategy 返回格式）
# ============================================================================








# ============================================================================
# IStrategy 抽象基类（借鉴 nautilus on_* + qlib generate_*）
# ============================================================================




# ============================================================================
# 策略适配器 — 包装 35 个异构具体策略类为 IStrategy
# ============================================================================


class _StrategyAdapter(IStrategy):
    """适配器：将异构策略类包装为统一 IStrategy 接口

    通过反射检测策略类的可用方法，按优先级调用：
      1. analyze() — 信号型策略（funding_rate_predictor/liquidation_hunter 等）
      2. scan_opportunities() — 扫描型套利策略（funding_rate_arb/basis_arbitrage 等）
      3. update() — 通用更新方法（crypto_intraday/chanlun_pa 等）
      4. decide() — 决策型（若有）

    适配器不修改原策略类，仅在调用时做参数转换。
    """

    # 方法优先级（按可读性+信号完整性排序）
    _METHOD_PRIORITY: Tuple[str, ...] = (
        "analyze",  # 信号型：返回结构化信号 dict
        "scan_opportunities",  # 扫描型：返回机会列表
        "scan_basis",  # basis_arbitrage 专用
        "scan",  # 通用扫描
        "predict",  # 预测型：funding_rate_predictor 等
        "update",  # 通用更新
        "get_last_signal",  # 两段式：LiquidationHunter (需先 add_bar)
        "get_signal",  # 两段式变体
        "decide",  # 决策型
        "calculate_quote",  # Phase 6C: perp_market_making 专用
        "check_opportunities",  # Phase 6C: 通用机会检查
    )

    # 两段式策略：需要先喂数据再取信号
    # Phase 6C: 升级为 Dict[str, List[str]] 支持一个信号方法对应多个 feeder 方法
    # 每个信号方法 → [feeder 方法列表]（按调用顺序）
    # Phase 6C 修复: analyze 和 scan 共享完整 feeder 列表，因为不同策略类
    # 可能用 analyze（mev_sandwich/jit_liquidity/vol_surface/vanna_volga）
    # 或 scan（triangular_arbitrage）作为信号方法，但用相同的 feeder（update_pool_state 等）
    _TWO_PHASE_FEEDERS: Dict[str, List[str]] = {
        # 原有两段式（LiquidationHunter）
        "get_last_signal": ["add_bar"],  # LiquidationHunter
        "get_signal": ["add_bar"],
        # v3.14 修复 (ME协调): funding_rate_predictor 主信号方法是 predict,
        # 不在原 _TWO_PHASE_FEEDERS 中, 导致 feeder 不被检测, _funding_history 永远为空
        # 修复: 添加 predict key, 与 analyze 共享同一 feeder 列表 (引用同一对象, 节省内存)
        "predict": None,  # 占位, 在类体结束后用 analyze 的同一列表填充
        # Phase 6C 新增：多 feeder 模式（analyze 与 scan 共享完整 feeder 列表）
        "analyze": [  # 信号型策略（stat_arb/vol_arb/mev/jit/vol_surface/vanna_volga/whale）
            # Phase 7B 修复: set_spot_price/set_market_data 必须在 add_option 之前，
            # vanna_volga 的 add_option 内部访问 self.spot_price 构造 OptionData，
            # 若 spot_price=0 会导致后续 analyze() 中 math.log(strike/0) 报错
            "set_spot_price",  # volatility_surface_arb (必须先于 add_option)
            "set_market_data",  # vanna_volga_arbitrage (必须先于 add_option)
            "update_prices",  # stat_arb_pairs / cross_asset_stat_arb
            "update_iv",  # volatility_arbitrage
            "update_vrp_data",  # vrp_harvesting
            "update_price",  # volatility_arbitrage (备选) / cross_chain / cex_dex / mean_reversion_ml
            "add_transaction",  # on_chain_whale_tracker
            "update_ticker",  # triangular_arbitrage
            "add_pending_swap",  # mev_sandwich / jit_liquidity
            "update_pool_state",  # mev_sandwich / jit_liquidity
            "update_bridge_status",  # cross_chain_arbitrage
            "add_option",  # volatility_surface_arb / vanna_volga
            # Phase 7D AI量化集成: mean_reversion_ml 的 VPIN 计算
            "update_volume",  # mean_reversion_ml: 买卖量用于 VPIN 毒性过滤
            # Phase 7D AI量化集成: ml_liquidation_predictor 的市场数据+技术指标
            "update_market_data",  # ml_liquidation_predictor: price/oi/leverage/funding/atr
            "update_technical",  # ml_liquidation_predictor: rsi/volume/avg_volume
            # Phase 7D AI量化集成: rl_market_making 的做市核心数据
            "update_midprice",  # rl_market_making: 中间价驱动 reservation price + spread
            "update_inventory",  # rl_market_making: 库存驱动对冲信号
            # Phase 7D AI量化集成: ml_liquidation_predictor 的清算集群
            "update_liquidation_clusters",  # ml_liquidation_predictor: LVI 计算
            # Phase 7E AI量化集成: meta_learning_portfolio 的策略表现
            "update_performance",  # meta_learning_portfolio: PnL 驱动 Alpha Decay 检测
            # Phase 7F AI量化集成: gnn_cross_asset_relationship 的多资产特征
            "update_features",  # gnn_cross_asset_relationship: 节点特征向量更新
            # v2.35 修复: event_driven_trading 的 feeder 方法
            # 根因: event_driven_trading 有 analyze() 但 feeder 方法不在列表中
            #       导致 _detect_primary_method 跳过 analyze, 策略无法产生信号
            # 影响: event_driven_trading 是高胜率世界观第一名(40%权重), 无法交易导致零交易率 88%
            "add_event",  # event_driven_trading: 宏观经济事件日历
            "update_fear_greed",  # event_driven_trading: 恐慌贪婪指数
            "update_economic_data",  # event_driven_trading: fed_funds/cpi/dxy/brent_crude
            # v2.35 批量修复: 32个策略58个feeder缺失 (第三智能体 _scan_feeder_missing.py 发现)
            # 根因: 32个策略有 analyze() 但 feeder 不在列表, fallback 模式下数据为空
            # 修复: 把所有缺失feeder加入列表, _invoke_single_feader 用通用 sim_data 提取
            "update_regime_data",  # regime_ensemble
            "update_skew_data",  # skew_trading
            "update_snapshot",  # cross_market_arbitrage
            "update_policy_performance", "add_policy",  # meta_learning_portfolio
            "update_sentiment_history", "update_sentiment",  # sentiment_driven_trading
            "update_returns", "update_ohlcv",  # diffusion_forecast_trading / risk_parity_signal
            "update_other_asset_states", "update_target_rtg", "update_state", "update_weights",  # transformer_decision_policy
            "add_transactions_batch", "add_tracked_wallet",  # smart_money_footprint / on_chain_whale_tracker
            "set_near_term", "set_event_days_ahead", "set_far_term", "update_term_structure_history",  # term_structure_trading
            "set_index_vol", "add_component", "set_components_batch", "update_historical_corr",  # dispersion_trading
            "update_price_volume",  # mean_reversion_ml
            "update_onchain_metrics", "set_date",  # seasonality_cycle
            "update_portfolio_value", "set_current_weights",  # cppi_portfolio_insurance / risk_parity_signal
            "update_state",  # adversarial_rl_trading
            "update_dxy", "update_macro_data", "update_etf_flow", "update_real_rate", "update_cycle_position", "update_stablecoin",  # macro_liquidity_cycle
            "update_policy", "update_time_to_close", "update_arrival_rate",  # rl_market_making
            "update_book_depth", "update_funding_rate", "update_cross_exchange",  # funding_rate_predictor
            "set_liquidation_clusters", "update_liquidation_data",  # liquidation_cascade
            "add_book_snapshot", "add_trade",  # order_flow_cvd
            "set_position", "scan_cointegrated_pairs",  # stat_arb_pairs / volatility_arbitrage
            "update_tickers_batch", "set_available_symbols",  # triangular_arbitrage
            "add_options_batch", "update_position_greeks",  # volatility_surface_arb
            "update_prices_batch",  # cex_dex_flash_arbitrage
            "set_hedge_position", "update_portfolio_greeks", "add_option_position",  # gamma_scalping
            "update_btc_dominance", "add_price",  # cross_asset_stat_arb
        ],
        "scan": [  # 扫描型套利（与 analyze 共享同一 feeder 列表）
            "set_spot_price",
            "set_market_data",
            "update_prices",
            "update_iv",
            "update_vrp_data",
            "update_price",
            "add_transaction",
            "update_ticker",
            "add_pending_swap",
            "update_pool_state",
            "update_bridge_status",
            "add_option",
            "update_volume",
            "update_market_data",
            "update_technical",
            "update_midprice",
            "update_inventory",
            "update_liquidation_clusters",
            "update_performance",
            "update_features",
            # v2.35 修复: event_driven_trading 的 feeder 方法 (与 analyze 共享)
            "add_event",
            "update_fear_greed",
            "update_economic_data",
            # v2.35 批量修复: 32个策略feeder缺失 (与 analyze 共享)
            "update_regime_data", "update_skew_data", "update_snapshot",
            "update_policy_performance", "add_policy",
            "update_sentiment_history", "update_sentiment",
            "update_returns", "update_ohlcv",
            "update_other_asset_states", "update_target_rtg", "update_state", "update_weights",
            "add_transactions_batch", "add_tracked_wallet",
            "set_near_term", "set_event_days_ahead", "set_far_term", "update_term_structure_history",
            "set_index_vol", "add_component", "set_components_batch", "update_historical_corr",
            "update_price_volume",
            "update_onchain_metrics", "set_date",
            "update_portfolio_value", "set_current_weights",
            "update_dxy", "update_macro_data", "update_etf_flow", "update_real_rate",
            "update_cycle_position", "update_stablecoin",
            "update_policy", "update_time_to_close", "update_arrival_rate",
            "update_book_depth", "update_funding_rate", "update_cross_exchange",
            "set_liquidation_clusters", "update_liquidation_data",
            "add_book_snapshot", "add_trade",
            "set_position", "scan_cointegrated_pairs",
            "update_tickers_batch", "set_available_symbols",
            "add_options_batch", "update_position_greeks",
            "update_prices_batch",
            "set_hedge_position", "update_portfolio_greeks", "add_option_position",
            "update_btc_dominance", "add_price",
        ],
    }
    # v3.14 修复 (ME协调): funding_rate_predictor 主信号方法是 predict
    # 让 predict 与 analyze 共享同一 feeder 列表 (引用同一对象, 节省内存)
    _TWO_PHASE_FEEDERS["predict"] = _TWO_PHASE_FEEDERS["analyze"]

    def __init__(
        self,
        seed_name: str,
        gene: Any,
        key_params: Dict[str, Any],
        strategy_instance: Any,
    ):
        super().__init__(seed_name, gene, key_params)
        self._strategy = strategy_instance
        self._primary_method: Optional[str] = None
        self._feeder_method: Optional[str] = None  # 兼容旧字段（单 feeder）
        self._feeder_methods: List[str] = []  # Phase 6C: 多 feeder 列表
        self._detect_primary_method()

    def on_start(self, deps: Optional[Dict[str, Any]] = None) -> None:
        """生命周期：启动（依赖注入在此发生）

        Phase 7B 修复: 调用内部策略的 start() 方法（如果存在），
        让 perp_market_making 等需要显式启动的策略进入 RUNNING 状态。
        """
        self._started = True
        if hasattr(self._strategy, "start") and callable(getattr(self._strategy, "start")):
            try:
                self._strategy.start()
            except Exception as e:
                logger.debug("策略 %s start() 失败: %s", self.seed_name, e)

    def _detect_primary_method(self) -> None:
        """反射检测策略类可用的主信号方法 + 多 feeder

        Phase 6C: 升级为多 feeder 检测。当信号方法在 _TWO_PHASE_FEEDERS 中时，
        遍历 feeder 列表，收集所有可用的 feeder 方法到 _feeder_methods。
        至少有一个 feeder 可用才算该信号方法可用。

        P0-4 修复: 当信号方法存在但无任何 feeder 时, 仍允许调用 (fallback 到无 feeder 模式)
        根因: event_driven_trading.analyze() 在 _TWO_PHASE_FEEDERS["analyze"] 中,
              但其 feeder (update_fear_greed / update_economic_data) 不在标准 feeder 列表
              导致 40% 权重的 event_driven Agent 全部空壳, 触发 "未找到任何信号方法" warning
        影响: gene_codec.py INTRADAY_WORLDVIEW_WEIGHTS 中 event_driven 占 40%, 全部失效
        修复: 无 feeder 时记录 debug 日志, 仍设置 _primary_method, 让策略用自己的内部数据
        """
        for method_name in self._METHOD_PRIORITY:
            if hasattr(self._strategy, method_name) and callable(
                getattr(self._strategy, method_name)
            ):
                # 多 feeder 模式：检测所有可用的 feeder
                if method_name in self._TWO_PHASE_FEEDERS:
                    feeder_names = self._TWO_PHASE_FEEDERS[method_name]
                    self._feeder_methods = []  # 重置列表
                    for feeder_name in feeder_names:
                        if hasattr(self._strategy, feeder_name) and callable(
                            getattr(self._strategy, feeder_name)
                        ):
                            self._feeder_methods.append(feeder_name)
                    # P0-4 修复: 无 feeder 时仍允许调用信号方法 (fallback 模式)
                    # event_driven_trading 等策略有自己的内部数据更新机制,
                    # 不依赖外部 feeder, 仍可正常调用 analyze()
                    if not self._feeder_methods:
                        logger.debug(
                            "策略 %s 信号方法 %s 无标准 feeder, 使用 fallback 模式 (策略内部喂数据)",
                            self.seed_name, method_name,
                        )
                        # 不再 continue 跳过, 仍设置 _primary_method
                        # _feeder_methods 保持空列表, update() 时跳过 feeder 调用
                    else:
                        # 兼容旧字段：保留第一个 feeder 到 _feeder_method
                        self._feeder_method = self._feeder_methods[0]
                self._primary_method = method_name
                return
        logger.warning(
            "策略 %s 未找到任何信号方法 %s",
            self.seed_name,
            self._METHOD_PRIORITY,
        )

    def update(self, ctx: MarketContext) -> Signal:
        """调用底层策略并转换为统一 Signal

        Phase 6C: 支持多 feeder 模式。依次调用所有 _feeder_methods 中的 feeder
        喂数据，每个 feeder 调用失败只记录 debug 日志，不中断后续调用。
        """
        if self._primary_method is None:
            return Signal(error=f"无可用方法: {self.seed_name}", timestamp=ctx.timestamp)

        try:
            # 多 feeder 模式：依次调用所有 feeder 喂数据
            feeder_methods = getattr(self, "_feeder_methods", [])
            if feeder_methods:
                for feeder_name in feeder_methods:
                    feeder = getattr(self._strategy, feeder_name, None)
                    if feeder is None:
                        continue
                    try:
                        self._invoke_single_feeder(feeder, feeder_name, ctx)
                    except Exception as e:
                        logger.debug("feeder %s 调用失败: %s", feeder_name, e)

            method = getattr(self._strategy, self._primary_method)
            result = self._invoke_with_adaptation(method, ctx)
            return self._normalize_result(result, ctx.timestamp)
        except Exception as e:
            logger.debug("策略 %s 执行失败: %s", self.seed_name, e, exc_info=False)
            return Signal(error=f"{type(e).__name__}: {e}", timestamp=ctx.timestamp)

    def _invoke_feeder(self, feeder: Callable, ctx: MarketContext) -> None:
        """兼容旧调用（已弃用，请用 _invoke_single_feeder）

        Phase 6C: 保留此方法以维持向后兼容。内部委托给 _invoke_single_feeder，
        默认按 add_bar 语义处理（原有 add_bar feeder 行为）。
        """
        self._invoke_single_feeder(feeder, "add_bar", ctx)

    def _invoke_single_feeder(
        self, feeder: Callable, feeder_name: str, ctx: MarketContext
    ) -> None:
        """调用单个 feeder 方法，根据方法名从 ctx.raw_packet 提取对应数据

        Phase 6C: 通用多 feeder 注入机制。根据 feeder 方法名识别数据源，
        从 ctx.raw_packet["simulator_data"] 中提取对应字段并适配参数签名。
        每个 feeder 调用失败只记录 debug 日志，不抛异常。

        Args:
            feeder: 可调用的 feeder 方法（已绑定 self）
            feeder_name: feeder 方法名（用于选择数据提取逻辑）
            ctx: 市场上下文（含 raw_packet 中的 simulator_data）
        """
        import inspect

        sim_data: Dict[str, Any] = {}
        if ctx.raw_packet:
            sim_data = (ctx.raw_packet or {}).get("simulator_data", {}) or {}

        try:
            sig = inspect.signature(feeder)
            # Phase 7B 修复: feeder 是绑定方法，signature 已不包含 self，
            # 不应再 [1:] 跳过第一个参数（之前的 [1:] 导致 price_a/implied_vol 等
            # 第一个真实参数被错误丢弃，feeder 调用全部因缺必填参数而静默失败）
            params = list(sig.parameters.keys())
        except (ValueError, TypeError):
            params = []

        if not params:
            try:
                feeder()
            except Exception:
                pass
            return

        param_set = set(params)
        kwargs: Dict[str, Any] = {}

        # 根据 feeder 名称提取对应数据
        if feeder_name == "add_bar":
            # 原有 LiquidationHunter add_bar：open/high/low/close/volume
            if "open_price" in param_set:
                kwargs["open_price"] = ctx.open
            elif "open" in param_set:
                kwargs["open"] = ctx.open
            if "high" in param_set:
                kwargs["high"] = ctx.high
            if "low" in param_set:
                kwargs["low"] = ctx.low
            if "close" in param_set:
                kwargs["close"] = ctx.close
            if "volume" in param_set:
                kwargs["volume"] = ctx.volume
            if "timestamp" in param_set:
                kwargs["timestamp"] = ctx.timestamp

        elif feeder_name == "update_prices":
            # stat_arb_pairs / cross_asset_stat_arb / gnn_cross_asset_relationship
            multi_prices = sim_data.get("multi_asset_prices", {}) or {}
            if "price_a" in param_set and "price_b" in param_set:
                # stat_arb_pairs / cross_asset_stat_arb: (price_a, price_b)
                prices = list(multi_prices.values())
                if len(prices) >= 2:
                    kwargs["price_a"] = prices[0]
                    kwargs["price_b"] = prices[1]
                # v3.14 修复 (ME协调): multi_asset_prices 为空时, 从 sim_data 直接读取
                # 之前只从 multi_asset_prices 取值, sim_data 中直接的 price_a/price_b 被忽略
                # 导致 stat_arb_pairs._prices_a 永远为空, 30 个 tick 阈值永远不满足
                if "price_a" not in kwargs and "price_a" in sim_data:
                    try:
                        kwargs["price_a"] = float(sim_data["price_a"])
                    except (TypeError, ValueError):
                        pass
                if "price_b" not in kwargs and "price_b" in sim_data:
                    try:
                        kwargs["price_b"] = float(sim_data["price_b"])
                    except (TypeError, ValueError):
                        pass
                # 回退: 若 sim_data 也无 price_a/price_b, 用 ctx 构造 BTC/ETH 价格对
                if "price_a" not in kwargs and ctx.close > 0:
                    kwargs["price_a"] = ctx.close  # 主资产价格
                if "price_b" not in kwargs and ctx.close > 0:
                    # 合成相关资产价格 (主资产的 1/16, 类似 BTC:ETH 比例)
                    kwargs["price_b"] = ctx.close / 16.0
            elif "prices" in param_set:
                # Phase 7F: gnn_cross_asset_relationship.update_prices(prices: Dict[str, float])
                # 多资产批量价格更新，构造 {symbol: price} 字典
                prices_dict: Dict[str, float] = {}
                if multi_prices:
                    # 优先使用 simulator_data 中的多资产价格
                    for sym, p in multi_prices.items():
                        try:
                            prices_dict[str(sym)] = float(p)
                        except (TypeError, ValueError):
                            pass
                # 补充当前 ctx 的主资产价格
                if ctx.symbol and ctx.close > 0 and ctx.symbol not in prices_dict:
                    prices_dict[ctx.symbol] = ctx.close
                # 如果只有1个资产，补充合成资产（BTC/ETH/SOL 等）让 GNN 有多节点
                if len(prices_dict) < 3 and ctx.close > 0:
                    base_price = ctx.close
                    token = ctx.symbol.split("-")[0] if "-" in ctx.symbol else ctx.symbol
                    # 构造 5 个资产的价格（基于当前资产价格 + 相关性扰动）
                    synthetic_assets = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT"]
                    if ctx.symbol not in synthetic_assets:
                        synthetic_assets.insert(0, ctx.symbol)
                    for i, sym in enumerate(synthetic_assets[:5]):
                        if sym not in prices_dict:
                            # 模拟相关性：不同资产有不同 beta
                            beta = 1.0 + 0.15 * (i - 2)  # beta 0.7-1.3
                            prices_dict[sym] = base_price * (0.1 ** (i % 3)) * beta if i > 0 else base_price
                kwargs["prices"] = prices_dict
            elif "price" in param_set:
                kwargs["price"] = ctx.close
            if "timestamp" in param_set:
                kwargs["timestamp"] = ctx.timestamp

        elif feeder_name == "update_iv":
            # volatility_arbitrage
            iv_feed = sim_data.get("iv_feed", {}) or {}
            if "implied_vol" in param_set:
                kwargs["implied_vol"] = iv_feed.get("iv", 0.5)
            if "iv" in param_set:
                kwargs["iv"] = iv_feed.get("iv", 0.5)

        elif feeder_name == "update_vrp_data":
            # vrp_harvesting.update_vrp_data(data): data 需是 VRPData 对象
            vrp_data = sim_data.get("vrp_data", {}) or {}
            if "data" in param_set:
                # Phase 7B 修复: 策略访问 data.implied_vol/data.realized_vol 等属性，需构造对象
                try:
                    from .vrp_harvesting import VRPData
                    kwargs["data"] = VRPData(
                        timestamp=ctx.timestamp,
                        implied_vol=float(vrp_data.get("iv", vrp_data.get("implied_vol", 0.5))),
                        realized_vol=float(vrp_data.get("rv", vrp_data.get("realized_vol", 0.4))),
                        vvol=float(vrp_data.get("vvol", 0.3)),
                        spot_price=float(ctx.close),
                    )
                except ImportError:
                    # 回退: 直接传 dict（策略可能也接受 dict）
                    kwargs["data"] = vrp_data

        elif feeder_name == "update_price":
            # 通用 price 更新（volatility_arbitrage / cross_chain / cex_dex）
            # Phase 7B 扩展2 修复: 区分 cross_chain / cex_dex，循环多链/多场所
            # 旧代码硬编码 chain=ETHEREUM / venue=CEX_BINANCE 单点，
            # 导致 cross_chain len(prices)<2 / cex_dex len(prices)<2 → "无套利机会"
            token = ctx.symbol.split("-")[0] if "-" in ctx.symbol else ctx.symbol

            # 分支A: cross_chain_arbitrage — 循环多链
            if "chain" in param_set and "venue" not in param_set:
                chain_prices = sim_data.get("chain_prices", {}) or {}
                if chain_prices:
                    try:
                        from .cross_chain_arbitrage import ChainType
                        # 字符串 → ChainType 映射
                        chain_map = {ct.value: ct for ct in ChainType}
                        for chain_name, price in chain_prices.items():
                            chain = chain_map.get(chain_name.lower(), ChainType.ETHEREUM)
                            try:
                                feeder(
                                    token=token,
                                    chain=chain,
                                    price=float(price),
                                    # Phase 7B 扩展2: 流动性 2M（>MIN_LIQUIDITY_USD=50K）
                                    liquidity_usd=2_000_000.0 if "liquidity_usd" in param_set else 0.0,
                                    gas_price_gwei=30.0 if "gas_price_gwei" in param_set else 0.0,
                                )
                            except Exception as e:
                                logger.debug("update_price(chain=%s) 调用失败: %s", chain_name, e)
                        return  # 已直接调用 feeder
                    except ImportError:
                        pass
                # 回退: 单链
                try:
                    from .cross_chain_arbitrage import ChainType
                    kwargs["chain"] = ChainType.ETHEREUM
                except ImportError:
                    kwargs["chain"] = "ethereum"

            # 分支B: cex_dex_flash_arbitrage — 循环多场所
            elif "venue" in param_set and "chain" not in param_set:
                venue_prices = sim_data.get("venue_prices", {}) or {}
                if venue_prices:
                    try:
                        from .cex_dex_flash_arbitrage import VenueType
                        venue_map = {vt.value: vt for vt in VenueType}
                        for venue_name, price in venue_prices.items():
                            venue = venue_map.get(venue_name.lower(), VenueType.CEX_BINANCE)
                            try:
                                feeder(
                                    token=token,
                                    venue=venue,
                                    price=float(price),
                                    # Phase 7B 扩展2 v2 修复: 流动性 2M → 50M
                                    # 旧值 2M: slippage = 1M*0.1*sqrt(1M/2M)/2 = $70K > gross $25K
                                    # 新值 50M: slippage = 1M*0.1*sqrt(1M/50M)/2 = $7K < gross $25K
                                    # 滑点公式: amount * 0.1 * (sqrt(amount/liq_buy)+sqrt(amount/liq_sell))/2
                                    liquidity_usd=50_000_000.0 if "liquidity_usd" in param_set else 0.0,
                                    fee_pct=0.001 if "fee_pct" in param_set else 0.001,
                                )
                            except Exception as e:
                                logger.debug("update_price(venue=%s) 调用失败: %s", venue_name, e)
                        return  # 已直接调用 feeder
                    except ImportError:
                        pass
                # 回退: 单场所
                try:
                    from .cex_dex_flash_arbitrage import VenueType
                    kwargs["venue"] = VenueType.CEX_BINANCE
                except ImportError:
                    kwargs["venue"] = "binance"

            # 分支C: volatility_arbitrage — 单点价格
            elif "symbol" in param_set and "price" in param_set and "token" not in param_set:
                # Phase 7F: gnn_cross_asset_relationship.update_price(symbol, price)
                # GNN 多资产单点更新：每个 tick 传入当前资产的 symbol + price
                # GNN 内部会维护多资产价格历史，需多次 update_price 调用积累数据
                kwargs["symbol"] = ctx.symbol
                kwargs["price"] = ctx.close
            else:
                if "price" in param_set:
                    kwargs["price"] = ctx.close
                if "token" in param_set:
                    kwargs["token"] = token
                if "liquidity_usd" in param_set:
                    kwargs["liquidity_usd"] = 1_000_000.0
                if "fee_pct" in param_set:
                    kwargs["fee_pct"] = 0.003
                if "gas_price_gwei" in param_set:
                    kwargs["gas_price_gwei"] = 30.0
                if "timestamp" in param_set:
                    kwargs["timestamp"] = ctx.timestamp

        elif feeder_name == "add_transaction":
            # on_chain_whale_tracker: 需要 WhaleTransaction 对象
            whale_txs = sim_data.get("whale_txs", []) or []
            if whale_txs:
                tx = whale_txs[-1]  # 最新一笔
                if "tx" in param_set:
                    # Phase 7B 修复: 策略期望 WhaleTransaction 对象而非 dict
                    # 模拟器输出: hash/from/to/amount/value_usd/timestamp/
                    #           is_exchange_inflow/is_exchange_outflow
                    # 需推导: direction/is_exchange/exchange_name/symbol/blockchain
                    try:
                        from .on_chain_whale_tracker import WhaleTransaction
                        is_inflow = bool(tx.get("is_exchange_inflow", False))
                        is_outflow = bool(tx.get("is_exchange_outflow", False))
                        to_addr = str(tx.get("to", "")) or ""
                        from_addr = str(tx.get("from", "")) or ""
                        # 推导 direction: 交易所流入="inflow"(看跌), 流出="outflow"(看涨)
                        direction = "inflow" if is_inflow else ("outflow" if is_outflow else "")
                        # 推导 is_exchange: 收款方或付款方是交易所
                        is_exchange = is_inflow or is_outflow or "exchange" in to_addr.lower() or "exchange" in from_addr.lower()
                        # 推导 exchange_name: 从地址提取
                        exchange_name = ""
                        if "exchange" in to_addr.lower():
                            exchange_name = to_addr.replace("0xexchange_", "").replace("0xexchange", "")
                        elif "exchange" in from_addr.lower():
                            exchange_name = from_addr.replace("0xexchange_", "").replace("0xexchange", "")
                        kwargs["tx"] = WhaleTransaction(
                            tx_hash=str(tx.get("hash", tx.get("tx_hash", ""))),
                            timestamp=float(tx.get("timestamp", ctx.timestamp)),
                            blockchain=str(tx.get("blockchain", "ethereum")),
                            from_address=from_addr,
                            to_address=to_addr,
                            amount_usd=float(tx.get("value_usd", tx.get("amount_usd", 0.0))),
                            amount_native=float(tx.get("amount", tx.get("amount_native", 0.0))),
                            symbol=str(tx.get("symbol", "BTC")),
                            transaction_type=str(tx.get("transaction_type", "transfer")),
                            is_exchange=is_exchange,
                            exchange_name=exchange_name,
                            direction=direction,
                        )
                    except ImportError:
                        # 回退: 直接传 dict（策略可能也接受 dict）
                        kwargs["tx"] = tx
                else:
                    # 直接传所有匹配字段
                    for k, v in tx.items():
                        if k in param_set:
                            kwargs[k] = v

        elif feeder_name == "update_ticker":
            # triangular_arbitrage: 需要多个 symbol 的 tickers 形成三角路径
            # Phase 7B 扩展2 修复: 旧代码只取 list(tickers.values())[0] 单个 ticker
            # 导致 available_symbols 只有1个 → discover_paths() 返回空 → "扫描0条路径"
            # 修复: 循环所有 tickers 直接调用 feeder，跳过末尾单次调用
            tickers = sim_data.get("tickers", {}) or {}
            if tickers and "symbol" in param_set and "bid" in param_set:
                # 批量喂入所有 tickers
                for sym, t in tickers.items():
                    try:
                        feeder(
                            symbol=sym,
                            bid=float(t.get("bid", ctx.close * 0.999)),
                            ask=float(t.get("ask", ctx.close * 1.001)),
                            last=float(t.get("last", ctx.close)) if "last" in param_set else 0.0,
                            volume=float(t.get("volume", 1000.0)) if "volume" in param_set else 0.0,
                        )
                    except Exception as e:
                        logger.debug("update_ticker(%s) 调用失败: %s", sym, e)
                # 已直接调用 feeder，跳过末尾的单次调用
                return
            # 回退: 单 ticker 模式
            ticker = list(tickers.values())[0] if tickers else {}
            if "symbol" in param_set:
                kwargs["symbol"] = ctx.symbol
            if "bid" in param_set:
                kwargs["bid"] = ticker.get("bid", ctx.close * 0.999)
            if "ask" in param_set:
                kwargs["ask"] = ticker.get("ask", ctx.close * 1.001)
            if "timestamp" in param_set:
                kwargs["timestamp"] = ctx.timestamp

        elif feeder_name == "add_pending_swap":
            # mev_sandwich (PendingSwap) / jit_liquidity (PendingSwapV3)
            amm_pool = sim_data.get("amm_pool", {}) or {}
            pending_swaps = amm_pool.get("pending_swaps", []) or []
            if pending_swaps:
                swap = pending_swaps[-1]
                if "swap" in param_set:
                    # Phase 7B 修复: 策略期望 PendingSwap/PendingSwapV3 对象而非 dict
                    # 模拟器输出: token_in/token_out/amount_in/sender/timestamp
                    # 需映射 + 补默认值: tx_hash/dex/pool_address/gas_price_gwei 等
                    swap_obj = None
                    # 先尝试 PendingSwapV3 (jit_liquidity)
                    try:
                        from .jit_liquidity_mev import PendingSwapV3, FeeTier
                        token_in = str(swap.get("token_in", "a"))
                        token_out = str(swap.get("token_out", "b"))
                        # token_in/out 是 "a"/"b" 简写,映射到真实符号
                        real_in = "BTC" if token_in == "a" else "USDT"
                        real_out = "USDT" if token_out == "b" else "BTC"
                        swap_obj = PendingSwapV3(
                            tx_hash=str(swap.get("tx_hash", swap.get("hash", ""))),
                            timestamp=float(swap.get("timestamp", ctx.timestamp)),
                            sender=str(swap.get("sender", "")),
                            # Phase 7B 扩展2 修复: pool_address 必须与 update_pool_state 一致
                            # 旧值 "0x_simulated_amm_pool" 与 update_pool_state 的 "0x_simulated_v3_pool" 不匹配
                            # 导致 analyze() 中 self.pool_states.get(swap.pool_address) 返回 None
                            pool_address="0x_simulated_v3_pool",
                            token_in=real_in,
                            token_out=real_out,
                            amount_in=float(swap.get("amount_in", 0.0)),
                            amount_out_min=0.0,
                            fee_tier=FeeTier.TIER_030,
                            gas_price_gwei=30.0,
                            is_zero_for_one=(token_in == "a"),
                        )
                    except ImportError:
                        pass
                    # 再尝试 PendingSwap (mev_sandwich)
                    if swap_obj is None:
                        try:
                            from .mev_sandwich_arbitrage import PendingSwap, DEXType
                            token_in = str(swap.get("token_in", "a"))
                            token_out = str(swap.get("token_out", "b"))
                            real_in = "BTC" if token_in == "a" else "USDT"
                            real_out = "USDT" if token_out == "b" else "BTC"
                            swap_obj = PendingSwap(
                                tx_hash=str(swap.get("tx_hash", swap.get("hash", ""))),
                                timestamp=float(swap.get("timestamp", ctx.timestamp)),
                                sender=str(swap.get("sender", "")),
                                dex=DEXType.UNKNOWN,
                                pool_address="0x_simulated_amm_pool",
                                token_in=real_in,
                                token_out=real_out,
                                amount_in=float(swap.get("amount_in", 0.0)),
                                amount_out_min=0.0,
                                slippage_pct=0.0,
                                gas_price_gwei=30.0,
                                path=[real_in, real_out],
                            )
                        except ImportError:
                            pass
                    # 回退: 直接传 dict
                    if swap_obj is None:
                        swap_obj = swap
                    kwargs["swap"] = swap_obj
                else:
                    for k, v in swap.items():
                        if k in param_set:
                            kwargs[k] = v

        elif feeder_name == "update_pool_state":
            # mev_sandwich (token_a/token_b) / jit_liquidity (token0/token1)
            amm_pool = sim_data.get("amm_pool", {}) or {}
            pool_state = amm_pool.get("pool_state", {}) or {}
            # Phase 7B 修复: 模拟器 pool_state 只有 reserve_a/reserve_b/price/fee_tier/timestamp
            # 但策略需要 address/token_a/token_b 等字段，需补充合成值
            import math as _math
            if "token_a" in param_set:
                # mev_sandwich 签名: address, token_a, token_b, reserve_a, reserve_b, fee_pct
                kwargs["address"] = "0x_simulated_amm_pool"
                kwargs["token_a"] = "BTC"
                kwargs["token_b"] = "USDT"
                kwargs["reserve_a"] = float(pool_state.get("reserve_a", 100.0))
                kwargs["reserve_b"] = float(pool_state.get("reserve_b", 5_000_000.0))
                kwargs["fee_pct"] = float(pool_state.get("fee_tier", 0.003))
            elif "token0" in param_set:
                # jit_liquidity 签名: address, token0, token1, sqrt_price_x96, current_tick, ...
                pool_price = float(pool_state.get("price", ctx.close))
                kwargs["address"] = "0x_simulated_v3_pool"
                kwargs["token0"] = "BTC"
                kwargs["token1"] = "USDT"
                # sqrt_price_x96 = sqrt(price) * 2^96
                kwargs["sqrt_price_x96"] = _math.sqrt(max(pool_price, 1e-8)) * (2 ** 96)
                # current_tick = log(price) / log(1.0001)
                kwargs["current_tick"] = int(_math.log(max(pool_price, 1e-8)) / _math.log(1.0001))
                # Phase 7B 扩展2 修复: 注入 liquidity_active（V3 池活跃流动性）
                # 旧代码未设置 → 默认 0.0 → calculate_required_liquidity 返回 0
                # → actual_share=0 → captured_fee=0 → net_profit<0 → "swap金额太小"或"无利润"
                # Phase 7B 扩展2 v2 修复: 1e15 过大导致资本成本爆炸
                #   L=1e15 → jit_L=1.9e16 → pool_capital=$500M → capital_req=$9.5B
                #   → capital_cost=$950K + slippage=$19M >> captured_fee=$332 → net=-$20M
                # 正确量级 L=1e7: pool_capital=$5K, jit_L=1.9e8, capital_req=$95K
                #   → capital_cost=$9.5 + slippage=$190 + gas=$30 < captured_fee=$332 → net=+$103
                if "liquidity_active" in param_set:
                    kwargs["liquidity_active"] = 1e7  # V3 池合理活跃流动性（匹配 $5K 池资本）
                if "tick_spacing" in param_set:
                    kwargs["tick_spacing"] = 60
                if "fee_tier" in param_set:
                    try:
                        from .jit_liquidity_mev import FeeTier
                        kwargs["fee_tier"] = FeeTier.TIER_030
                    except ImportError:
                        pass
            # 通用: 从 pool_state 中匹配其他字段
            for k, v in pool_state.items():
                if k in param_set and k not in kwargs:
                    kwargs[k] = v

        elif feeder_name == "update_bridge_status":
            # cross_chain_arbitrage: 需要 BridgeType/ChainType 枚举
            bridge_data = sim_data.get("bridge", {}) or {}
            bridges = bridge_data.get("bridges", []) or []
            if bridges:
                bridge = bridges[0]
                # Phase 7B 修复: 字段名映射 + 枚举转换
                # 模拟器: name/src_chain/dst_chain/delay_minutes/fee_pct/liquidity_usd/status
                # 策略: bridge/source/target/tvl_usd/fee_pct/delay_min
                if "bridge" in param_set:
                    bridge_name = str(bridge.get("name", "Across")).lower()
                    try:
                        from .cross_chain_arbitrage import BridgeType
                        mapped = None
                        for bt in BridgeType:
                            if bt.value == bridge_name:
                                mapped = bt
                                break
                        kwargs["bridge"] = mapped if mapped is not None else BridgeType.ACROSS
                    except ImportError:
                        kwargs["bridge"] = bridge.get("name", "Across")
                if "source" in param_set:
                    src = str(bridge.get("src_chain", "ethereum")).lower()
                    try:
                        from .cross_chain_arbitrage import ChainType
                        mapped = None
                        for ct in ChainType:
                            if ct.value == src:
                                mapped = ct
                                break
                        kwargs["source"] = mapped if mapped is not None else ChainType.ETHEREUM
                    except ImportError:
                        kwargs["source"] = src
                if "target" in param_set:
                    dst = str(bridge.get("dst_chain", "arbitrum")).lower()
                    try:
                        from .cross_chain_arbitrage import ChainType
                        mapped = None
                        for ct in ChainType:
                            if ct.value == dst:
                                mapped = ct
                                break
                        kwargs["target"] = mapped if mapped is not None else ChainType.ARBITRUM
                    except ImportError:
                        kwargs["target"] = dst
                if "tvl_usd" in param_set:
                    kwargs["tvl_usd"] = float(bridge.get("liquidity_usd", 50_000_000.0))
                if "fee_pct" in param_set:
                    kwargs["fee_pct"] = float(bridge.get("fee_pct", 0.001))
                if "delay_min" in param_set:
                    kwargs["delay_min"] = float(bridge.get("delay_minutes", 10.0))

        elif feeder_name == "add_option":
            # volatility_surface_arb (OptionQuote 对象) / vanna_volga (逐参数)
            options_market = sim_data.get("options_market", {}) or {}
            options = options_market.get("options", []) or []
            if options:
                if "option" in param_set:
                    # volatility_surface_arb: 需要 OptionQuote 对象
                    # Phase 7B 修复: 必须喂入完整期权链（多个到期日×多个行权价×call/put），
                    # 否则策略无法校准SVI曲面，analyze() 返回 "无曲面套利机会"
                    # 旧代码只喂 options[0]，导致曲面永远为空
                    try:
                        from .volatility_surface_arb import OptionQuote, OptionType
                        for option in options:
                            ot_str = str(option.get("type", option.get("option_type", "call"))).lower()
                            opt_type = OptionType.PUT if "put" in ot_str else OptionType.CALL
                            oq = OptionQuote(
                                symbol=str(option.get("symbol", ctx.symbol)),
                                underlying=str(option.get("underlying", ctx.symbol.split("-")[0] if "-" in ctx.symbol else ctx.symbol)),
                                option_type=opt_type,
                                strike=float(option.get("strike", ctx.close)),
                                expiry_days=float(option.get("expiry_days", option.get("expiry", 30.0))),
                                bid=float(option.get("bid", 0.0)),
                                ask=float(option.get("ask", 0.0)),
                                mid=float(option.get("mid", option.get("price", 0.0))),
                                implied_vol=float(option.get("implied_vol", option.get("iv", option.get("market_iv", 0.5)))),
                            )
                            try:
                                feeder(option=oq)
                            except Exception as e:
                                logger.debug("add_option(option=...) 调用失败: %s", e)
                    except ImportError:
                        kwargs["option"] = options[0]
                    # 已直接调用 feeder，跳过末尾的单次调用
                    return
                else:
                    # vanna_volga: 逐参数 + OptionType 枚举（只需一个参考期权）
                    # 签名: symbol, option_type, strike, expiry_days, market_price, market_iv
                    option = options[0]
                    if "symbol" in param_set:
                        kwargs["symbol"] = str(option.get("symbol", ctx.symbol))
                    if "option_type" in param_set:
                        ot_str = str(option.get("type", option.get("option_type", "call"))).lower()
                        try:
                            from .vanna_volga_arbitrage import OptionType
                            kwargs["option_type"] = OptionType.PUT if "put" in ot_str else OptionType.CALL
                        except ImportError:
                            kwargs["option_type"] = option.get("type", option.get("option_type", "call"))
                    if "strike" in param_set:
                        kwargs["strike"] = float(option.get("strike", ctx.close))
                    if "expiry_days" in param_set:
                        kwargs["expiry_days"] = float(option.get("expiry_days", option.get("expiry", 30.0)))
                    if "market_price" in param_set:
                        kwargs["market_price"] = float(option.get("market_price", option.get("price", option.get("mid", 0.0))))
                    if "market_iv" in param_set:
                        kwargs["market_iv"] = float(option.get("market_iv", option.get("iv", option.get("implied_vol", 0.5))))

        elif feeder_name == "set_spot_price":
            # volatility_surface_arb
            if "price" in param_set:
                kwargs["price"] = ctx.close
            if "spot" in param_set:
                kwargs["spot"] = ctx.close

        elif feeder_name == "set_market_data":
            # vanna_volga_arbitrage
            options_market = sim_data.get("options_market", {}) or {}
            if "spot" in param_set:
                kwargs["spot"] = ctx.close
            if "atm_vol" in param_set:
                kwargs["atm_vol"] = options_market.get("atm_vol", 0.6)
            if "rr_25delta" in param_set:
                kwargs["rr_25delta"] = options_market.get("rr_25delta", 0.0)
            if "bf_25delta" in param_set:
                kwargs["bf_25delta"] = options_market.get("bf_25delta", 0.02)

        elif feeder_name == "update_volume":
            # Phase 7D AI量化集成: mean_reversion_ml.update_volume(buy_vol, sell_vol)
            # 用于 VPIN 毒性过滤计算 (Easley López de Prado O'Hara 2012)
            # 模拟买卖量：用 ctx.volume 的 50/50 拆分作为基线，价格涨跌偏向买/卖方
            # 涨: 买方主导 (buy_vol = 0.6*volume, sell_vol = 0.4*volume)
            # 跌: 卖方主导 (buy_vol = 0.4*volume, sell_vol = 0.6*volume)
            if "buy_vol" in param_set and "sell_vol" in param_set:
                base_vol = float(ctx.volume) if ctx.volume > 0 else 1000.0
                # 价格变动方向决定买卖比例
                if ctx.close > 0 and ctx.open > 0:
                    price_change_ratio = (ctx.close - ctx.open) / ctx.open
                else:
                    price_change_ratio = 0.0
                # 涨跌幅限制在 ±2% 内，对应 ±10% 买卖偏向
                bias = max(-0.1, min(0.1, price_change_ratio * 5))
                buy_vol = base_vol * (0.5 + bias)
                sell_vol = base_vol * (0.5 - bias)
                kwargs["buy_vol"] = buy_vol
                kwargs["sell_vol"] = sell_vol

        elif feeder_name == "update_market_data":
            # v3.14 修复 (ME协调): 同名方法冲突处理
            # ml_liquidation_predictor.update_market_data(price, oi, leverage, avg_lev, funding, atr)
            # liquidity_adjusted_strategy.update_market_data(price, volume, high=None, low=None)
            # 区分: liquidity_adjusted 有 volume 参数, ml_liquidation 有 oi/leverage 参数
            is_liquidity_adjusted = "volume" in param_set and "oi" not in param_set
            if is_liquidity_adjusted:
                # liquidity_adjusted_strategy: 价格/成交量/高低价
                if "price" in param_set:
                    kwargs["price"] = ctx.close
                if "volume" in param_set:
                    # ctx.volume 在沙盘可能为 0, 用合理默认值保证 returns_history 积累
                    vol = float(ctx.volume) if ctx.volume and ctx.volume > 0 else 1000.0
                    kwargs["volume"] = vol
                if "high" in param_set:
                    kwargs["high"] = ctx.high if ctx.high > 0 else ctx.close
                if "low" in param_set:
                    kwargs["low"] = ctx.low if ctx.low > 0 else ctx.close
            else:
                # Phase 7D AI量化集成: ml_liquidation_predictor.update_market_data
                #   (price, oi, leverage, avg_lev, funding, atr)
                # 来源：ml_liquidation_predictor.py:222
                # 模拟数据：基于 ctx.close 推导 oi/leverage/funding/atr
                if "price" in param_set:
                    kwargs["price"] = ctx.close
                if "oi" in param_set:
                    # 开放兴趣：BTC 永续 OI 通常 $1亿-$10亿 量级
                    # 沙盘环境用固定合理值 $500M（避免 LVI 因 OI 过小而失真）
                    kwargs["oi"] = 500_000_000.0
                if "leverage" in param_set:
                    kwargs["leverage"] = 50_000_000.0  # 总杠杆 USD
                if "avg_lev" in param_set:
                    kwargs["avg_lev"] = 5.0  # 平均杠杆 5x
                if "funding" in param_set:
                    # 资金费率：价格涨则正费率（多头付空头），价格跌则负费率
                    sim_data_local = sim_data
                    funding_rate = 0.0001  # 默认 0.01%
                    if "funding_rate" in sim_data_local:
                        funding_rate = float(sim_data_local["funding_rate"])
                    elif ctx.close > 0 and ctx.open > 0:
                        price_change = (ctx.close - ctx.open) / ctx.open
                        funding_rate = max(-0.001, min(0.001, price_change * 0.1))
                    kwargs["funding"] = funding_rate
                if "atr" in param_set:
                    # ATR 必须是小数比率（如 0.02=2%），不是绝对值
                    # ml_liquidation_predictor.calculate_lvi: vol_mult = 1.0 + atr_24h * 10
                    # 注释 "ATR 8% → mult=1.8" 证实 atr_24h 是小数
                    if ctx.atr > 0 and ctx.close > 0:
                        kwargs["atr"] = float(ctx.atr) / float(ctx.close)
                    else:
                        kwargs["atr"] = 0.02  # 默认 2% 波动率

        elif feeder_name == "update_funding_rate":
            # v3.14 修复 (ME协调): funding_rate_predictor.update_funding_rate
            #   (rate, perp_price, spot_price, timestamp=None)
            # 来源: funding_rate_predictor.py:134
            # 之前: 通用 sim_data 提取因字段名不匹配 (sim_data["funding_rate"] vs 参数 rate) 失败
            # 修复: 添加专门 elif, 把 sim_data["funding_rate"] 映射到参数 rate
            if "rate" in param_set:
                rate_val = float(sim_data.get("funding_rate", sim_data.get("rate", 0.0001)))
                kwargs["rate"] = rate_val
            if "perp_price" in param_set:
                # 永续合约价格近似现货价格
                kwargs["perp_price"] = ctx.close
            if "spot_price" in param_set:
                kwargs["spot_price"] = ctx.close

        elif feeder_name == "update_cross_exchange":
            # v3.14 修复 (ME协调): funding_rate_predictor.update_cross_exchange(rates)
            # 来源: funding_rate_predictor.py:160
            # rates: Dict[str, float] 跨交易所资金费率
            if "rates" in param_set:
                base_rate = float(sim_data.get("funding_rate", 0.0001))
                # 沙盘合成: 3 个交易所的资金费率 (主交易所 +/- 微小差异)
                rates_dict = {
                    "binance": base_rate,
                    "okx": base_rate * 1.05,
                    "bybit": base_rate * 0.95,
                }
                kwargs["rates"] = rates_dict

        elif feeder_name == "update_book_depth":
            # v3.14 修复 (ME协调): funding_rate_predictor.update_book_depth(bid_depth, ask_depth)
            # 来源: funding_rate_predictor.py:168
            if "bid_depth" in param_set:
                kwargs["bid_depth"] = 1000000.0  # 沙盘合成: $1M 买盘深度
            if "ask_depth" in param_set:
                kwargs["ask_depth"] = 1000000.0  # 沙盘合成: $1M 卖盘深度

        elif feeder_name == "update_technical":
            # Phase 7D AI量化集成: ml_liquidation_predictor.update_technical
            #   (rsi, volume, avg_volume)
            # 来源：ml_liquidation_predictor.py:232
            if "rsi" in param_set:
                # 从 raw_packet 或指标中取 RSI，默认 50
                rsi_val = 50.0
                if ctx.raw_packet:
                    indicators = (ctx.raw_packet or {}).get("indicators", {}) or {}
                    rsi_val = float(indicators.get("rsi_14", 50.0))
                kwargs["rsi"] = rsi_val
            if "volume" in param_set:
                kwargs["volume"] = float(ctx.volume) if ctx.volume > 0 else 1000.0
            if "avg_volume" in param_set:
                # 平均成交量：用 price_history 长度作为简单估算
                avg_v = float(ctx.volume) if ctx.volume > 0 else 1000.0
                kwargs["avg_volume"] = avg_v

        elif feeder_name == "update_midprice":
            # Phase 7D AI量化集成: rl_market_making.update_midprice(midprice)
            # 来源：rl_market_making.py:214 (Avellaneda-Stoikov 最优报价)
            # 中间价驱动 reservation_price + optimal_spread 计算
            if "midprice" in param_set:
                kwargs["midprice"] = ctx.close

        elif feeder_name == "update_inventory":
            # Phase 7D AI量化集成: rl_market_making.update_inventory(inventory)
            # 来源：rl_market_making.py:220
            # 库存驱动对冲信号 (SKEW_ASK/SKEW_BID/HEDGE_INVENTORY)
            # 沙盘无持仓状态时，inventory=0（中性做市）
            # 若 deps 中有 engine 可查询实际持仓
            if "inventory" in param_set:
                inventory = 0.0
                # 尝试从 deps 获取实际持仓
                deps = ctx.deps or {}
                engine = deps.get("engine") or deps.get("spot_engine")
                if engine is not None:
                    try:
                        positions = engine.get_positions("") if hasattr(engine, "get_positions") else {}
                        # 简单估算：用 BTC 持仓量作为 inventory
                        if isinstance(positions, dict):
                            for sym, pos in positions.items():
                                if isinstance(pos, (int, float)):
                                    inventory += float(pos)
                                elif isinstance(pos, dict):
                                    inventory += float(pos.get("size", pos.get("amount", 0.0)))
                    except Exception:
                        pass
                kwargs["inventory"] = inventory

        elif feeder_name == "update_liquidation_clusters":
            # Phase 7D AI量化集成: ml_liquidation_predictor.update_liquidation_clusters
            #   (long_clusters: List[LiquidationCluster], short_clusters: List[LiquidationCluster])
            # 来源：ml_liquidation_predictor.py:238
            # 模拟清算集群：基于当前价格构造多空清算价位
            # LVI = (long_liq_value / open_interest) * vol_mult * 100
            # 注意：CLUSTER_DISTANCE_PCT=0.05 是小数（5%），distance_pct 也用小数
            if "long_clusters" in param_set and "short_clusters" in param_set:
                try:
                    from .ml_liquidation_predictor import LiquidationCluster
                    # Phase 7D 修复: cluster_value 根据价格变化动态调整
                    # 价格下跌越多 → 多头杠杆被套牢越多 → long_cluster_value 越大
                    # 价格上涨越多 → 空头杠杆被套牢越多 → short_cluster_value 越大
                    # 这样能在极端行情下触发 CASCADE_IMMINENT (LVI > 50)
                    if ctx.close > 0 and ctx.open > 0:
                        price_change_pct = (ctx.close - ctx.open) / ctx.open
                    else:
                        price_change_pct = 0.0
                    # 基础清算量 $5M，价格每跌 1% 增加 $30M（模拟杠杆多头堆积）
                    # 价格跌 5% → long_value = 5M + 5*30M = 155M → LVI ≈ 55（触发 CASCADE）
                    # 价格跌 10% → long_value = 5M + 10*30M = 305M → LVI ≈ 108（极端）
                    base_long_value = 5_000_000.0
                    if price_change_pct < 0:
                        long_value_1 = base_long_value + abs(price_change_pct) * 100 * 30_000_000.0
                    else:
                        long_value_1 = base_long_value
                    long_value_2 = long_value_1 * 2.5  # -5% 处清算量更大
                    # 空头清算：价格上涨时增加
                    base_short_value = 3_000_000.0
                    if price_change_pct > 0:
                        short_value_1 = base_short_value + price_change_pct * 100 * 20_000_000.0
                    else:
                        short_value_1 = base_short_value
                    short_value_2 = short_value_1 * 2.0
                    # 模拟多头清算集群（杠杆多头在 -3%/-5% 处被清算）
                    long_clusters = [
                        LiquidationCluster(
                            side="long",
                            price_level=ctx.close * 0.97,  # -3%
                            total_value=long_value_1,
                            distance_pct=-0.03,  # 小数表示 -3%
                        ),
                        LiquidationCluster(
                            side="long",
                            price_level=ctx.close * 0.95,  # -5%
                            total_value=long_value_2,
                            distance_pct=-0.05,  # 小数表示 -5%
                        ),
                    ]
                    # 模拟空头清算集群（杠杆空头在 +3%/+5% 处被清算）
                    short_clusters = [
                        LiquidationCluster(
                            side="short",
                            price_level=ctx.close * 1.03,  # +3%
                            total_value=short_value_1,
                            distance_pct=0.03,
                        ),
                        LiquidationCluster(
                            side="short",
                            price_level=ctx.close * 1.05,  # +5%
                            total_value=short_value_2,
                            distance_pct=0.05,
                        ),
                    ]
                    kwargs["long_clusters"] = long_clusters
                    kwargs["short_clusters"] = short_clusters
                except ImportError:
                    pass

        elif feeder_name == "update_performance":
            # Phase 7E AI量化集成: meta_learning_portfolio.update_performance(pnl)
            # 来源：meta_learning_portfolio.py:238
            # PnL 驱动 Alpha Decay 检测 + 策略表现跟踪
            # 沙盘环境：从 deps.engine 查询已实现 PnL，无引擎则用价格变化估算
            if "pnl" in param_set:
                pnl = 0.0
                deps = ctx.deps or {}
                engine = deps.get("engine") or deps.get("spot_engine")
                if engine is not None:
                    try:
                        # 尝试从引擎获取已实现 PnL
                        if hasattr(engine, "get_pnl"):
                            pnl = float(engine.get_pnl()) or 0.0
                        elif hasattr(engine, "get_unrealized_pnl"):
                            pnl = float(engine.get_unrealized_pnl()) or 0.0
                    except Exception:
                        pass
                # 回退：用价格变化估算 PnL（假设持有多头仓位）
                if pnl == 0.0 and ctx.close > 0 and ctx.open > 0:
                    pnl = (ctx.close - ctx.open) / ctx.open  # 收益率作为 PnL
                kwargs["pnl"] = pnl

        elif feeder_name == "update_features":
            # Phase 7F AI量化集成: gnn_cross_asset_relationship.update_features(symbol, features)
            # 来源：gnn_cross_asset_relationship.py:936
            # 更新 GNN 节点特征向量（returns/vol/volume/momentum 等 8 维特征）
            # 沙盘环境：从 ctx 衍生统计特征（GNN 内部也会从 returns_history 自动构造）
            if "symbol" in param_set and "features" in param_set:
                try:
                    import numpy as np
                    # 构造 8 维节点特征（GNN NODE_FEATURE_DIM=8）
                    # [mean_return, std_return, recent_momentum, last_return,
                    #  max_return, min_return, median_return, history_len]
                    features_list = []
                    # 使用 price_history 计算收益率统计
                    if ctx.price_history and len(ctx.price_history) >= 2:
                        ph = ctx.price_history
                        returns = [
                            (ph[i] - ph[i - 1]) / ph[i - 1]
                            for i in range(1, len(ph))
                            if ph[i - 1] > 0
                        ]
                        if returns:
                            features_list = [
                                float(np.mean(returns[-20:])),   # mean_return
                                float(np.std(returns[-20:])),     # std_return
                                float(np.sum(returns[-5:])),      # recent_momentum
                                float(returns[-1]),               # last_return
                                float(np.max(returns[-20:])),     # max_return
                                float(np.min(returns[-20:])),     # min_return
                                float(np.percentile(returns[-20:], 50)),  # median
                                float(len(returns)),              # history_len
                            ]
                    # 回退：用 ctx 的 OHLC 衍生特征
                    if not features_list and ctx.close > 0 and ctx.open > 0:
                        ret = (ctx.close - ctx.open) / ctx.open
                        features_list = [ret, abs(ret), ret, ret, ret, ret, ret, 1.0]
                    # 最终回退：零向量
                    if not features_list:
                        features_list = [0.0] * 8
                    kwargs["symbol"] = ctx.symbol
                    kwargs["features"] = np.array(features_list, dtype=float)
                except ImportError:
                    # numpy 不可用时跳过（GNN 依赖 numpy）
                    pass

        elif feeder_name == "update_fear_greed":
            # v2.35: event_driven_trading 的恐惧贪婪指数 feeder
            # 来源: event_driven_trading.py:209 update_fear_greed(fgi: float)
            # sandbox 环境: 从价格波动率推断 (高波动=恐慌, 低波动=贪婪)
            # 协调: MCP-AGENT P0-4b 要求 fear_greed_index≠0 才产生信号
            if "fgi" in param_set:
                fgi = float(sim_data.get("fear_greed_index", 0.0))
                if fgi == 0.0:
                    # 从价格历史推断: 用最近20根K线的波动率
                    if ctx.price_history and len(ctx.price_history) >= 2:
                        ph = ctx.price_history[-20:] if len(ctx.price_history) >= 20 else ctx.price_history
                        returns = [
                            (ph[i] - ph[i - 1]) / ph[i - 1]
                            for i in range(1, len(ph))
                            if ph[i - 1] > 0
                        ]
                        if returns:
                            try:
                                import statistics
                                vol = statistics.stdev(returns) if len(returns) > 1 else 0.02
                            except (statistics.StatisticsError, ImportError):
                                vol = 0.02
                            # 波动率映射: vol=0.01→84(贪婪), vol=0.05→20(恐慌)
                            fgi = max(10.0, min(90.0, 100.0 - vol * 1600.0))
                        else:
                            fgi = 50.0
                    else:
                        fgi = 50.0
                kwargs["fgi"] = fgi

        elif feeder_name == "update_economic_data":
            # v2.35: event_driven_trading 的经济数据 feeder
            # 来源: event_driven_trading.py:214 update_economic_data(fed_funds, cpi, dxy, brent, gold, treasury_10y_real)
            # sandbox 环境: 用合理默认值 (近似 2026 年宏观经济数据)
            econ = sim_data.get("economic_data", {}) or {}
            if "fed_funds" in param_set:
                kwargs["fed_funds"] = float(econ.get("fed_funds", 5.25))
            if "cpi" in param_set:
                kwargs["cpi"] = float(econ.get("cpi", 3.2))
            if "dxy" in param_set:
                kwargs["dxy"] = float(econ.get("dxy", 104.0))
            if "brent" in param_set:
                kwargs["brent"] = float(econ.get("brent", 80.0))
            if "gold" in param_set:
                kwargs["gold"] = float(econ.get("gold", 2000.0))
            if "treasury_10y_real" in param_set:
                kwargs["treasury_10y_real"] = float(econ.get("treasury_10y_real", 1.5))

        elif feeder_name == "add_event":
            # v2.35: event_driven_trading 的事件日历 feeder
            # sandbox 环境不提供宏观经济事件日历, 此 feeder 跳过
            # 影响: event_driven 不会产生 PRE_EVENT_REDUCE/POST_EVENT_BUY 等事件信号
            #       但仍会基于 fear_greed + 经济数据产生 DXY/real_rate 信号
            # 如果 simulator_data 有 macro_events, 提取并调用
            events = sim_data.get("macro_events", [])
            if events and "event" in param_set:
                try:
                    from .event_driven_trading import MacroEvent
                    for ev_data in events[:3]:  # 最多添加3个事件避免过载
                        if isinstance(ev_data, dict):
                            try:
                                event = MacroEvent(**ev_data)
                                feeder(event)
                            except (TypeError, ValueError):
                                pass
                except ImportError:
                    pass
            # 不设置 kwargs, 跳过标准调用 (事件已通过上面的 feeder(event) 直接调用)

        # 通用回退：price/volume/timestamp
        if not kwargs:
            if "price" in param_set:
                kwargs["price"] = ctx.close
            if "volume" in param_set:
                kwargs["volume"] = ctx.volume
            if "timestamp" in param_set:
                kwargs["timestamp"] = ctx.timestamp

        # v2.36 通用 sim_data 提取 (ME协调修复): 对未在上面 elif 中处理的 feeder,
        # 或 elif 处理未填充所有必填参数的情况, 从 sim_data 按参数名补充提取
        # 协调: 第三智能体发现32个策略58个feeder缺失, 无法为每个写专门 elif
        # v3.14 修复: 之前 `if not kwargs and sim_data` 条件被通用回退的 timestamp 设置阻断,
        # 导致 funding_rate_predictor / stat_arb_pairs / liquidity_adjusted_strategy 三个策略
        # 的 feeder 调用失败, 内部状态永远为空, 真实回测胜率 0%
        # 修复: 移除 `not kwargs` 条件, 让通用提取总是触发, 仅补充未填充的参数
        # 安全性: 已存在的 kwargs 不被覆盖, 只补充缺失字段
        if sim_data:
            for param_name in param_set:
                # 跳过已填充的参数, 避免覆盖专门 elif 的设置
                if param_name in kwargs:
                    continue
                if param_name in sim_data:
                    try:
                        kwargs[param_name] = sim_data[param_name]
                    except (TypeError, ValueError):
                        pass
                elif param_name == "symbol":
                    kwargs[param_name] = ctx.symbol
                elif param_name == "timestamp":
                    kwargs[param_name] = ctx.timestamp

        filtered = {k: v for k, v in kwargs.items() if k in param_set}
        try:
            if filtered:
                feeder(**filtered)
        except Exception as e:
            logger.debug("feeder %s 调用失败: %s", feeder_name, e)

    def _invoke_with_adaptation(self, method: Callable, ctx: MarketContext) -> Any:
        """根据方法签名适配调用参数

        策略类方法签名异构，本方法通过 inspect.signature 检测参数名，按需注入。
        常见签名：
          - analyze(market_data: dict) / analyze(symbol, price_history)
          - scan_opportunities(symbols: List[str])
          - update(kline: dict) / update(price, volume) / update(return_value: float)
          - scan_basis(symbols)
        """
        import inspect

        try:
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
        except (ValueError, TypeError):
            params = []

        # 无参数方法
        if not params or params == ["self"]:
            return method()

        # 按参数名匹配
        kwargs: Dict[str, Any] = {}
        param_set = set(params)

        # 常见参数名映射
        if "symbol" in param_set:
            kwargs["symbol"] = ctx.symbol
        if "symbols" in param_set:
            kwargs["symbols"] = [ctx.symbol] if ctx.symbol else []
        if "price" in param_set:
            kwargs["price"] = ctx.close
        if "volume" in param_set:
            kwargs["volume"] = ctx.volume
        if "price_history" in param_set:
            kwargs["price_history"] = ctx.price_history
        if "kline" in param_set:
            kwargs["kline"] = ctx.kline
        if "market_data" in param_set:
            kwargs["market_data"] = ctx.raw_packet
        if "data" in param_set:
            kwargs["data"] = ctx.raw_packet
        if "return_value" in param_set:
            kwargs["return_value"] = ctx.period_return
        if "timestamp" in param_set:
            kwargs["timestamp"] = ctx.timestamp
        if "ctx" in param_set or "context" in param_set:
            kwargs["ctx" if "ctx" in param_set else "context"] = ctx
        # Phase 7B 扩展2 v2 修复: scan(tokens=None) 触发自动发现
        # 旧代码未映射 tokens → 回退到 method(ctx.kline) → scan(tokens={"open":...,"close":...})
        # → 遍历 dict keys "open"/"high"/"low" 等无效 token → detect_price_diff 找不到匹配 → 0 机会
        if "tokens" in param_set and "token" not in param_set:
            kwargs["tokens"] = None  # 让 scan() 自动从 venue_prices/chain_prices 发现 token

        # 若参数无法匹配，回退到位置参数
        if not kwargs:
            try:
                return method(ctx.kline)
            except Exception:
                try:
                    return method(ctx.close)
                except Exception:
                    return method()

        # 过滤掉 sig 中不存在的 kwargs（防 unexpected keyword）
        filtered = {k: v for k, v in kwargs.items() if k in param_set}
        try:
            return method(**filtered)
        except Exception:
            # 最终回退：尝试位置参数
            try:
                return method(ctx.kline)
            except Exception:
                return method()

    def _normalize_result(self, result: Any, timestamp: float) -> Signal:
        """将异构返回值归一化为 Signal

        处理 7 种返回格式：
          - dict（最常见，含 signal/direction/confidence/strength 等字段）
          - dataclass 对象（如 FundingPrediction/SweepSignal，含 direction/confidence 属性）
          - list（套利机会列表）
          - tuple (score, direction)
          - float（纯信号得分）
          - int（方向枚举）
          - None（无信号）
        """
        if result is None:
            return Signal(timestamp=timestamp, sources=[self.seed_name])

        if isinstance(result, dict):
            return self._normalize_dict(result, timestamp)
        # dataclass 对象 → 转 dict 后归一化
        if hasattr(result, "__dataclass_fields__"):
            try:
                from dataclasses import asdict
                return self._normalize_dict(asdict(result), timestamp)
            except Exception:
                # fallback: 提取 direction/confidence 属性
                direction = Direction.from_any(getattr(result, "direction", None))
                confidence = float(getattr(result, "confidence", 0.0) or 0.0)
                return Signal(
                    direction=direction,
                    confidence=min(1.0, confidence),
                    sources=[self.seed_name],
                    metadata={"raw_result": str(result)[:200]},
                    timestamp=timestamp,
                )
        if isinstance(result, list):
            # 机会列表 → 取第一个有效机会
            if not result:
                return Signal(timestamp=timestamp, sources=[self.seed_name])
            first = result[0]
            # Phase 7B 修复: list 中的 dataclass 对象需转 dict（如 ArbOpportunity/BasisSnapshot）
            # 之前 isinstance(first, dict) 为 False 时直接用 {}，导致所有字段丢失
            if hasattr(first, "__dataclass_fields__"):
                try:
                    from dataclasses import asdict
                    first = asdict(first)
                except Exception:
                    first = {"raw_result": str(first)[:200]}
            elif not isinstance(first, dict):
                first = {}
            return self._normalize_dict(first, timestamp, opportunity_count=len(result))
        if isinstance(result, tuple) and len(result) >= 2:
            score, direction = result[0], result[1]
            return Signal(
                direction=Direction.from_any(direction),
                strength=float(abs(score)) if isinstance(score, (int, float)) else 0.0,
                confidence=min(1.0, float(abs(score))) if isinstance(score, (int, float)) else 0.0,
                sources=[self.seed_name],
                timestamp=timestamp,
            )
        if isinstance(result, (int, float)):
            v = float(result)
            direction = Direction.LONG if v > 0 else (Direction.SHORT if v < 0 else Direction.FLAT)
            return Signal(
                direction=direction,
                strength=min(1.0, abs(v)),
                confidence=min(1.0, abs(v)),
                sources=[self.seed_name],
                timestamp=timestamp,
            )
        # 未知类型，存入 metadata
        return Signal(
            sources=[self.seed_name],
            metadata={"raw_result": str(result)[:200]},
            timestamp=timestamp,
        )

    def _normalize_dict(
        self, d: Dict[str, Any], timestamp: float, opportunity_count: int = 0
    ) -> Signal:
        """归一化 dict 返回值（处理 5 种方向字段 + 6 种强度字段 + 策略特定字段智能推导）"""

        # 方向字段（5 种命名）
        direction = Direction.FLAT
        for key in ("direction", "signal", "action", "decision", "market_regime", "side"):
            if key in d:
                direction = Direction.from_any(d[key])
                if direction != Direction.FLAT:
                    break

        # Phase 7B: 策略特定字段智能推导（当标准方向字段映射失败时）
        # 解决 15 个孤立策略模块返回 dataclass（含 z_score/signal枚举/recommendation 等）
        # 但无标准 direction 字段导致 Signal.direction=FLAT 的空壳问题
        if direction == Direction.FLAT:
            direction = self._infer_direction_from_fields(d)

        # 置信度字段（6 种命名，按优先级）
        confidence = 0.0
        for key in ("confidence", "market_confidence", "consensus_value", "probability", "strength", "cascade_probability"):
            if key in d:
                try:
                    v = float(d[key])
                    if 0 <= v <= 1:
                        confidence = v
                        break
                    # 可能是 0-100
                    if 0 <= v <= 100:
                        confidence = v / 100.0
                        break
                except (TypeError, ValueError):
                    continue

        # Phase 7B: 置信度智能推导（当标准置信度字段为 0 时）
        if confidence == 0.0:
            confidence = self._infer_confidence_from_fields(d)

        # 强度字段
        strength = confidence
        for key in ("strength", "imbalance_strength", "hawkes_signal_weight", "signal_strength"):
            if key in d:
                try:
                    v = float(d[key])
                    if 0 <= v <= 1:
                        strength = v
                        break
                except (TypeError, ValueError):
                    continue

        # 概率字段（可选）
        probability = None
        for key in ("probability", "predicted_probability", "cascade_probability"):
            if key in d:
                try:
                    v = float(d[key])
                    if 0 <= v <= 1:
                        probability = v
                        break
                except (TypeError, ValueError):
                    continue

        meta = dict(d)
        if opportunity_count:
            meta["_opportunity_count"] = opportunity_count

        return Signal(
            direction=direction,
            confidence=confidence,
            strength=strength,
            probability=probability,
            sources=[self.seed_name],
            metadata=meta,
            timestamp=timestamp,
        )

    def _infer_direction_from_fields(self, d: Dict[str, Any]) -> Direction:
        """Phase 7B: 从策略特定字段智能推导方向

        当标准方向字段（direction/signal/action/decision/market_regime/side）
        映射失败时，从以下策略特定字段推导方向：

        1. z_score / current_z_score: 正→SHORT，负→LONG（统计套利类）
        2. signal 枚举名: 含 long/buy→LONG，含 short/sell→SHORT，
           含 overpriced→SHORT（做空），含 underpriced→LONG（做多）
        3. expected_return: 正→LONG，负→SHORT
        4. vrp / current_vrp: 正（IV>RV，波动率高估）→SHORT，负→LONG
        5. recommendation: 含 buy/long→LONG，含 sell/short→SHORT
        6. profit_usd/net_profit: 正→LONG（有套利机会），0→FLAT
        7. best_opportunity: 递归提取嵌套机会字典中的方向（套利类策略）
        8. direction 扩展映射: inflow→SHORT, outflow→LONG（巨鲸追踪类）

        Args:
            d: 策略返回的 dict（可能是 dataclass asdict 后的）

        Returns:
            推导出的 Direction，无法推导则返回 FLAT
        """
        # 1. z_score / current_z_score 推导（统计套利类）
        for zkey in ("z_score", "current_z_score"):
            if zkey in d:
                try:
                    z = float(d[zkey])
                    if abs(z) >= 1.5:  # 入场阈值
                        return Direction.LONG if z < 0 else Direction.SHORT
                except (TypeError, ValueError):
                    pass

        # 2. signal 枚举名推导
        if "signal" in d:
            sig = d["signal"]
            sig_str = ""
            if isinstance(sig, str):
                sig_str = sig.lower()
            elif hasattr(sig, "name"):
                sig_str = sig.name.lower()
            elif hasattr(sig, "value"):
                try:
                    sig_str = str(sig.value).lower()
                except Exception:
                    pass

            if sig_str:
                # Phase 7B 扩展: 加入波动率套利信号关键词
                # hedge_tail = 买入OTM Put保护（持有现货多头+买Put）→ LONG
                # long_vol = 买入波动率 → LONG; short_vol = 卖出波动率 → SHORT
                if any(k in sig_str for k in ("long", "buy", "underpriced", "bullish",
                                                "hedge_tail", "long_vol", "close_long")):
                    return Direction.LONG
                if any(k in sig_str for k in ("short", "sell", "overpriced", "bearish",
                                                "short_vol", "close_short", "stop_loss")):
                    return Direction.SHORT

        # 2.5 Phase 7D: signal_type 字段推导（ml_liquidation_predictor / rl_market_making 等）
        # 这两个 AI 策略的 dataclass 用 signal_type 而非 signal，类型是 Enum
        # LiquidationSignalType: CASCADE_IMMINENT/WARNING/IN_PROGRESS → SHORT（级联下跌）
        #   BUY_THE_DIP → LONG; AVOID_LONG → SHORT; AVOID_SHORT → LONG
        # RLMMSignalType: SKEW_BID → LONG（偏多）; SKEW_ASK → SHORT（偏空）
        if "signal_type" in d:
            sig = d["signal_type"]
            sig_str = ""
            if isinstance(sig, str):
                sig_str = sig.lower()
            elif hasattr(sig, "name"):
                sig_str = sig.name.lower()
            elif hasattr(sig, "value"):
                try:
                    sig_str = str(sig.value).lower()
                except Exception:
                    pass

            if sig_str:
                # 注意顺序: AVOID_LONG/AVOID_SHORT 必须先于 long/short 通用匹配
                # 否则 "avoid_long" 会被 "long" 关键词错误匹配为 LONG
                if "avoid_long" in sig_str:
                    return Direction.SHORT
                if "avoid_short" in sig_str:
                    return Direction.LONG
                # 级联下跌 → 做空（包括 imminent/warning/in_progress）
                if any(k in sig_str for k in ("cascade", "imminent", "in_progress")):
                    return Direction.SHORT
                # Phase 7I (v447 Bug#1): 流动性风险预警信号映射
                # liquidity_adjusted_strategy 的 LiquiditySignalType 包含 11 个枚举值:
                #   LIQUIDITY_LONG / LIQUIDITY_SHORT → 已被通用 long/short 匹配
                #   HOLD / NEUTRAL → 保持 FLAT
                #   LIQUIDITY_CRISIS / FEEDBACK_LOOP_WARNING / VPIN_TOXIC → 强制减仓 → SHORT
                #   HIGH_IMPACT_COST / THIN_ORDER_BOOK / LIQUIDITY_DRYING → 观望 → FLAT
                #     (不返回, 让 best_opportunity.direction 推导)
                #   LIQUIDITY_PREMIUM_HARVEST → 做多低流动性溢价 → LONG
                # v446 Bug: 上述 6 个风险/溢价信号被吞为 FLAT, 占 LiquiditySignal 信号 55%
                # v447 修复: 显式关键词匹配, 风险预警→减仓(SHORT), 溢价收割→做多(LONG)
                if any(k in sig_str for k in (
                    "crisis", "feedback_loop", "vpin_toxic", "toxic",
                    "feedback_warning",
                )):
                    return Direction.SHORT  # 强制减仓避险
                if any(k in sig_str for k in (
                    "premium_harvest", "harvest", "liquidity_premium",
                )):
                    return Direction.LONG   # 做多低流动性溢价
                # HIGH_IMPACT_COST / THIN_ORDER_BOOK / LIQUIDITY_DRYING → 不返回,
                # 让后续 best_opportunity 递归推导从 direction 字段取方向
                # 抄底买入 → 做多
                if any(k in sig_str for k in ("buy_the_dip", "buy_dip", "dip")):
                    return Direction.LONG
                # 做市偏斜: skew_bid → 偏多; skew_ask → 偏空
                if "skew_bid" in sig_str:
                    return Direction.LONG
                if "skew_ask" in sig_str:
                    return Direction.SHORT
                # Phase 7E: 元学习组合信号
                # INCREASE_RISK → 增加风险 = 看多 → LONG
                # DECREASE_RISK → 降低风险 = 看空 → SHORT
                # ALPHA_DECAY_WARNING → Alpha 衰减 = 减仓 → SHORT
                # REGIME_SHIFT/POLICY_SWITCH/REBALANCE → 中性（无方向）
                if "increase_risk" in sig_str:
                    return Direction.LONG
                if "decrease_risk" in sig_str:
                    return Direction.SHORT
                if "alpha_decay" in sig_str:
                    return Direction.SHORT
                # Phase 7F: GNN 跨资产关系信号（gnn_cross_asset_relationship）
                # GNNSignalType:
                #   SYSTEMIC_RISK_WARNING → SHORT（系统性风险，减仓避险）
                #   CONTAGION_DETECTED → SHORT（风险传染，做空）
                #   LEAD_LAG_OPPORTUNITY → 根据 best_opportunity.z_score 判断
                #     z_score < 0 → asset_a 低估 → LONG; z_score > 0 → asset_a 高估 → SHORT
                #   DIVERSIFICATION_OPPORTUNITY / HEDGE_SIGNAL / CLUSTER_ROTATION
                #   / REGIME_SHIFT → FLAT（中性，无方向）
                if "systemic_risk" in sig_str:
                    return Direction.SHORT
                if "contagion" in sig_str:
                    return Direction.SHORT
                if "lead_lag" in sig_str:
                    # 检查 best_opportunity 中的 z_score 决定方向
                    best_opp = d.get("best_opportunity")
                    if isinstance(best_opp, dict):
                        try:
                            z = float(best_opp.get("z_score", 0.0))
                            if abs(z) > 1.0:  # Z-score 超过1才视为有效信号
                                return Direction.LONG if z < 0 else Direction.SHORT
                        except (TypeError, ValueError):
                            pass
                # CLUSTER_ROTATION: 板块轮动（做多 strong / 做空 weak）
                # best_opportunity.primary_asset 是强势资产，默认做多
                # 注：沙盘中 agent 通常交易主资产(BTC)，而 BTC 通常是强势资产
                if "cluster_rotation" in sig_str:
                    best_opp = d.get("best_opportunity")
                    if isinstance(best_opp, dict) and best_opp.get("primary_asset"):
                        return Direction.LONG  # 做多强势资产
                # 对冲库存（无法简单定向，跳过让后续推导）
                # PLACE_BID_ASK / HOLD / NEUTRAL 不在此处理（保持 FLAT）
                # Phase 7H: 扩散模型预测信号（diffusion_forecast_trading）
                # DiffusionSignalType:
                #   DIFFUSION_LONG / DIFFUSION_SHORT → 已被通用 long/short 关键词匹配（line 1475-1480）
                #   TAIL_OPPORTUNITY → LONG（5%分位为正=尾部上涨机会）
                #   TAIL_RISK_WARNING → SHORT（95%分位为负=尾部下跌风险，减仓避险）
                #   HIGH_UNCERTAINTY / VOLATILITY_CLUSTER / FAT_TAIL_DETECTED
                #     / MULTI_MODAL / BACKBOME_BREAKDOWN / HOLD / NEUTRAL → FLAT（中性或避险）
                if "tail_opportunity" in sig_str:
                    return Direction.LONG
                if "tail_risk" in sig_str:
                    return Direction.SHORT
                # Phase 7H: Decision Transformer 信号（transformer_decision_policy）
                # DTSignalType:
                #   STRONG_BUY / BUY → LONG（已被通用 buy 关键词匹配，line 1475-1477）
                #   STRONG_SELL / SELL → SHORT（已被通用 sell 关键词匹配，line 1478-1480）
                #   TRAJECTORY_ANOMALY → SHORT（轨迹异常=避险减仓）
                #   REGIME_ADAPTIVE / POSITION_SCALE / HOLD / NEUTRAL → FLAT
                if "trajectory_anomaly" in sig_str:
                    return Direction.SHORT
                # v2.35 修复: EventSignalType (event_driven_trading) 通用映射
                # 根因: EventSignalType 的值 (post_event_buy/post_event_sell/extreme_fear_buy
                #       /extreme_greed_sell/negative_real_rate_buy/dxy_strong_sell) 不在
                #       上述任何关键词中, 导致 _normalize_result 返回 FLAT
                # 影响: event_driven_trading 占 40% 权重, 信号被吞导致零交易, 进化淘汰
                #       (gen0 38% → gen16 17% 持续下降)
                # 修复: 通用 buy/sell 关键词匹配, 覆盖所有 *_buy → LONG, *_sell → SHORT
                # 安全性: avoid_long/avoid_short/buy_the_dip 已在前面处理, 不会误匹配
                if "buy" in sig_str and "avoid" not in sig_str:
                    return Direction.LONG
                if "sell" in sig_str and "avoid" not in sig_str:
                    return Direction.SHORT

        # 3. expected_return 推导（波动率套利类）
        if "expected_return" in d:
            try:
                er = float(d["expected_return"])
                if abs(er) > 0.001:  # 0.1% 阈值
                    return Direction.LONG if er > 0 else Direction.SHORT
            except (TypeError, ValueError):
                pass

        # 4. vrp / current_vrp 推导（VRP 收割类）
        for vkey in ("vrp", "current_vrp"):
            if vkey in d:
                try:
                    vrp = float(d[vkey])
                    if abs(vrp) > 0.01:  # 1% 阈值
                        return Direction.SHORT if vrp > 0 else Direction.LONG
                except (TypeError, ValueError):
                    pass

        # 5. recommendation 推导（中英文兼容）
        if "recommendation" in d:
            try:
                rec = str(d["recommendation"]).lower()
                # Phase 7B 扩展: 加入中文关键词（买入/卖出/做多/做空/建议买/建议卖）
                if any(k in rec for k in ("buy", "long", "enter long",
                                            "买入", "做多", "建议买", "看涨", "低估")):
                    return Direction.LONG
                if any(k in rec for k in ("sell", "short", "enter short",
                                            "卖出", "做空", "建议卖", "看跌", "高估")):
                    return Direction.SHORT
            except Exception:
                pass

        # 6. profit/net_profit 推导（套利类）
        # Phase 7B 扩展2: 加入 estimated_profit_usd / net_profit_pct / gross_profit_pct
        # triangular_arbitrage.TriArbOpportunity 用 estimated_profit_usd 而非 profit_usd
        # Phase 7B 扩展2 v2: 加入 net_profit_usd（JIT/FlashArb/CrossChain 三个套利机会统一用此字段名）
        for key in ("profit_usd", "net_profit", "net_profit_usd", "expected_profit", "estimated_profit_usd"):
            if key in d:
                try:
                    p = float(d[key])
                    if p > 0:
                        return Direction.LONG  # 有正利润机会
                except (TypeError, ValueError):
                    pass

        # 6.5 Phase 7B 扩展: funding_rate 推导（资金费率套利特有，优先于 expected_annual_return）
        # funding_rate > 0 表示永续多头付资金费率给空头 → 做空永续吃费率 → SHORT
        if "funding_rate" in d:
            try:
                fr = float(d["funding_rate"])
                if fr > 0.0001:  # 0.01% 阈值
                    return Direction.SHORT  # 做空永续吃资金费率
            except (TypeError, ValueError):
                pass

        # 6.6 Phase 7B 扩展: expected_annual_return / annualized_funding 推导（基差套利类）
        for key in ("expected_annual_return", "annualized_funding", "annual_return"):
            if key in d:
                try:
                    ar = float(d[key])
                    if ar > 0.01:  # 1% 年化阈值
                        return Direction.LONG  # 有正年化收益机会
                except (TypeError, ValueError):
                    pass

        # 6.7 Phase 7B 扩展: score 推导（综合评分类）
        if "score" in d:
            try:
                s = float(d["score"])
                if s > 0.5:  # 正评分阈值
                    return Direction.LONG  # 正评分 = 看多机会
                if s < -0.5:
                    return Direction.SHORT
            except (TypeError, ValueError):
                pass

        # 6.8 Phase 7B 扩展: 做市策略推导（从 bid_price/ask_price/spread/inventory 推导）
        # 做市策略本质市场中性，但需产出方向信号用于进化系统：
        # - inventory > 0（多头库存）→ 偏向卖 → SHORT（减仓）
        # - inventory < 0（空头库存）→ 偏向买 → LONG（补仓）
        # - inventory == 0 且 spread_rate > 0.1% → 做市有利 → LONG（积极做市信号）
        # 注意: MMQuote.spread 是价差率（小数，如 0.002=0.2%），不是绝对价差
        if "bid_price" in d and "ask_price" in d and "mid_price" in d:
            try:
                inv = float(d.get("inventory", 0.0))
                if inv > 0.001:
                    return Direction.SHORT  # 多头库存，偏向卖
                if inv < -0.001:
                    return Direction.LONG  # 空头库存，偏向买
                # inventory == 0，从 spread_rate 推导做市意愿
                # MMQuote.spread 是价差率（小数），直接用
                spread_rate = float(d.get("spread", 0.0))
                if spread_rate > 0.001:  # spread率 > 0.1% → 做市有利
                    return Direction.LONG  # 积极做市信号
            except (TypeError, ValueError):
                pass

        # 6.9 Phase 7B 扩展: 波动率曲面推导（从 skew_25d / term_slope 推导）
        # volatility_surface_arb 的字段：
        # - skew_25d > 0 → Put IV > Call IV（看跌保护需求强）→ 市场偏空 → SHORT
        # - skew_25d < 0 → Call IV > Put IV（看涨投机需求强）→ 市场偏多 → LONG
        # - term_slope > 0 → 远期IV > 近期IV（contango）→ 趋势延续 → LONG
        # - term_slope < 0 → 近期IV > 远期IV（backwardation）→ 短期恐慌 → SHORT
        if "skew_25d" in d or "term_slope" in d:
            try:
                skew = float(d.get("skew_25d", 0.0))
                if abs(skew) > 0.005:  # 0.5% IV skew 阈值
                    return Direction.SHORT if skew > 0 else Direction.LONG
                term = float(d.get("term_slope", 0.0))
                if abs(term) > 0.005:
                    return Direction.LONG if term > 0 else Direction.SHORT
            except (TypeError, ValueError):
                pass

        # 6.10 Phase 7B 扩展: Vanna-Volga 推导（从 rr_25delta / bf_25delta 推导）
        # vanna_volga_arbitrage 的字段：
        # - rr_25delta (Risk Reversal) > 0 → Call IV > Put IV → 看涨 → LONG
        # - rr_25delta < 0 → Put IV > Call IV → 看跌 → SHORT
        # - bf_25delta (Butterfly) 高 → 波动率微笑陡峭 → 方向中性，不推导
        if "rr_25delta" in d:
            try:
                rr = float(d.get("rr_25delta", 0.0))
                if abs(rr) > 0.003:  # 0.3% RR 阈值（SimulatedOptionsMarket典型RR≈0.005）
                    return Direction.LONG if rr > 0 else Direction.SHORT
            except (TypeError, ValueError):
                pass

        # 6.11 Phase 7B 扩展: 跨资产统计套利推导（从 current_beta / btc_dominance 推导）
        # cross_asset_stat_arb 的字段：
        # - current_beta > 1.2 → 资产高贝塔，市场上涨时涨幅大 → LONG
        # - current_beta < 0.8 → 资产低贝塔，防御性 → SHORT（相对弱于市场）
        # - btc_dominance > 0.55 → BTC 主导，资金从山寨币流出 → SHORT（对非BTC资产）
        # - btc_dominance < 0.40 → 山寨币季节，资金流入山寨 → LONG
        if "current_beta" in d and abs(float(d.get("current_beta", 0.0))) > 0.01:
            try:
                beta = float(d["current_beta"])
                if beta > 1.2:
                    return Direction.LONG
                if beta < 0.8 and beta > 0:
                    return Direction.SHORT
            except (TypeError, ValueError):
                pass
        if "btc_dominance" in d:
            try:
                dom = float(d.get("btc_dominance", 0.0))
                if dom > 0.55:
                    return Direction.SHORT  # BTC主导，山寨币看跌
                if dom < 0.40 and dom > 0:
                    return Direction.LONG  # 山寨币季节
            except (TypeError, ValueError):
                pass

        # 7. best_opportunity 递归推导（套利类策略：triangular/mev/jit/cross_chain/cex_dex/vol_surface/vanna_volga）
        # Phase 7H扩展: 兼容 best_decision 字段（transformer_decision_policy 的 DTSignal 用 best_decision）
        best_opp = d.get("best_opportunity") or d.get("best_decision")
        if isinstance(best_opp, dict):
            # 递归调用自身，从 best_opportunity 中提取方向
            sub_dir = self._infer_direction_from_fields(best_opp)
            if sub_dir != Direction.FLAT:
                return sub_dir
            # 如果 best_opportunity 中有 direction/confidence 但标准映射失败
            # 尝试从 profit/z_score 等字段推导
            for pkey in ("profit_usd", "net_profit", "expected_profit", "profit_pct"):
                if pkey in best_opp:
                    try:
                        p = float(best_opp[pkey])
                        if p > 0:
                            return Direction.LONG
                    except (TypeError, ValueError):
                        pass

        # 8. direction 扩展映射（巨鲸追踪类：inflow/outflow 等非标准值）
        if "direction" in d:
            try:
                dir_val = str(d["direction"]).lower() if d["direction"] is not None else ""
                if any(k in dir_val for k in ("inflow", "deposit", "bearish", "sell_pressure")):
                    return Direction.SHORT  # 流入交易所 = 看跌
                if any(k in dir_val for k in ("outflow", "withdrawal", "bullish", "buy_pressure")):
                    return Direction.LONG  # 流出交易所 = 看涨
            except Exception:
                pass

        return Direction.FLAT

    def _infer_confidence_from_fields(self, d: Dict[str, Any]) -> float:
        """Phase 7B: 从策略特定字段智能推导置信度

        当标准置信度字段为 0 时，从以下字段推导：
        0. confidence 字段直接读取（GNN 等策略的 best_opportunity 含此字段）
        1. z_score: confidence = min(1.0, abs(z_score)/3.0)
        2. |expected_return|: confidence = min(1.0, abs(er)*10)
        3. |vrp|: confidence = min(1.0, abs(vrp)*5)
        4. profit_usd: confidence = min(1.0, profit/1000)
        5. correlation: confidence = abs(correlation)

        Returns:
            推导出的置信度 [0.0, 1.0]，无法推导则返回 0.0
        """
        # 0. Phase 7F: 直接读取 confidence 字段（GNNOpportunity 等嵌套对象含此字段）
        # _normalize_dict 行 1369 只检查顶层 dict 的 confidence key，
        # 但 GNN 的 GNNSignal 顶层无 confidence 字段，confidence 在 best_opportunity 中。
        # 当 _infer_confidence_from_fields 递归到 best_opportunity 时，需直接读取。
        if "confidence" in d:
            try:
                c = float(d["confidence"])
                if 0 < c <= 1:
                    return c
                if c > 1:
                    return min(1.0, c / 100.0)  # 可能是 0-100
            except (TypeError, ValueError):
                pass

        # 1. z_score
        if "z_score" in d:
            try:
                z = float(d["z_score"])
                if abs(z) >= 1.5:
                    return min(1.0, abs(z) / 3.0)
            except (TypeError, ValueError):
                pass

        # 2. expected_return
        if "expected_return" in d:
            try:
                er = float(d["expected_return"])
                if abs(er) > 0.001:
                    return min(1.0, abs(er) * 10)
            except (TypeError, ValueError):
                pass

        # 3. vrp / current_vrp 推导（VRP 收割类）
        # Phase 7B 修复: vrp_harvesting 返回 current_vrp 而非 vrp，导致 conf=0
        for vkey in ("vrp", "current_vrp"):
            if vkey in d:
                try:
                    vrp = float(d[vkey])
                    if abs(vrp) > 0.01:
                        return min(1.0, abs(vrp) * 5)
                except (TypeError, ValueError):
                    pass

        # 4. profit
        # Phase 7B 扩展2: 加入 estimated_profit_usd（triangular_arbitrage 用）
        # Phase 7B 扩展2 v2: 加入 net_profit_usd（JIT/FlashArb/CrossChain 套利机会统一字段名）
        for key in ("profit_usd", "net_profit", "net_profit_usd", "expected_profit", "estimated_profit_usd"):
            if key in d:
                try:
                    p = float(d[key])
                    if p > 0:
                        return min(1.0, p / 1000.0)
                except (TypeError, ValueError):
                    pass

        # 4.5 Phase 7B 扩展: funding_rate 置信度（资金费率套利类）
        if "funding_rate" in d:
            try:
                fr = float(d["funding_rate"])
                if abs(fr) > 0.0001:
                    return min(1.0, abs(fr) * 200)  # 0.001 → 0.2, 0.005 → 1.0
            except (TypeError, ValueError):
                pass

        # 4.6 Phase 7B 扩展: expected_annual_return 置信度
        for key in ("expected_annual_return", "annualized_funding", "annual_return"):
            if key in d:
                try:
                    ar = float(d[key])
                    if ar > 0.01:
                        return min(1.0, ar / 0.5)  # 0.5(50%) → 1.0
                except (TypeError, ValueError):
                    pass

        # 4.7 Phase 7B 扩展: score 置信度
        if "score" in d:
            try:
                s = float(d["score"])
                if abs(s) > 0.5:
                    return min(1.0, abs(s) / 10.0)  # 10 → 1.0
            except (TypeError, ValueError):
                pass

        # 4.8 Phase 7B 扩展: 做市策略 confidence（从 spread_rate 推导）
        # MMQuote.spread 是价差率（小数），spread率越大做市越有利
        if "bid_price" in d and "ask_price" in d and "mid_price" in d:
            try:
                spread_rate = float(d.get("spread", 0.0))
                if spread_rate > 0.001:  # spread率 > 0.1%
                    return min(1.0, spread_rate * 10)  # 0.1 spread率 → confidence 1.0
            except (TypeError, ValueError):
                pass

        # 4.9 Phase 7B 扩展: 波动率曲面 confidence（从 skew_25d / term_slope 推导）
        if "skew_25d" in d or "term_slope" in d:
            try:
                skew = abs(float(d.get("skew_25d", 0.0)))
                term = abs(float(d.get("term_slope", 0.0)))
                max_val = max(skew, term)
                if max_val > 0.005:
                    return min(1.0, max_val * 50)  # 0.02 IV → 1.0
            except (TypeError, ValueError):
                pass

        # 4.10 Phase 7B 扩展: Vanna-Volga confidence（从 rr_25delta 推导）
        if "rr_25delta" in d:
            try:
                rr = abs(float(d.get("rr_25delta", 0.0)))
                if rr > 0.003:
                    return min(1.0, rr * 50)  # 0.02 RR → 1.0
            except (TypeError, ValueError):
                pass

        # 4.11 Phase 7B 扩展: 跨资产 confidence（从 current_beta / btc_dominance 推导）
        if "current_beta" in d:
            try:
                beta = float(d.get("current_beta", 0.0))
                if abs(beta - 1.0) > 0.2:  # 偏离 1.0 超过 0.2
                    return min(1.0, abs(beta - 1.0))  # 偏离越大置信度越高
            except (TypeError, ValueError):
                pass
        if "btc_dominance" in d:
            try:
                dom = float(d.get("btc_dominance", 0.0))
                if dom > 0.55:
                    return min(1.0, (dom - 0.45) * 5)
                if dom < 0.40 and dom > 0:
                    return min(1.0, (0.45 - dom) * 5)
            except (TypeError, ValueError):
                pass

        # 5. correlation
        if "correlation" in d:
            try:
                c = float(d["correlation"])
                return min(1.0, abs(c))
            except (TypeError, ValueError):
                pass

        # 6. Phase 7B 扩展: best_opportunity 递归推导（套利类策略）
        # mev_sandwich/jit/cross_chain/cex_dex 等套利类策略返回 best_opportunity 字段，
        # 与 _infer_direction_from_fields 第7步对应，confidence 也需递归从 best_opportunity 提取
        # 旧代码只递归方向不递归置信度，导致 dir=120/120 但 conf=0/120
        # Phase 7H扩展: 兼容 best_decision 字段（transformer_decision_policy 的 DTSignal）
        best_opp = d.get("best_opportunity") or d.get("best_decision")
        if isinstance(best_opp, dict):
            sub_conf = self._infer_confidence_from_fields(best_opp)
            if sub_conf > 0:
                return sub_conf

        return 0.0


# ============================================================================
# GeneDrivenStrategy — 12 个抽象世界观的通用适配器
# ============================================================================


class GeneDrivenStrategy(IStrategy):
    """基因驱动型策略

    用于 12 个抽象世界观（trend_following/mean_reversion/breakout/momentum 等）。
    这些世界观没有具体实现类，行为完全由 AgentGene 字段决定。

    本类不替代 AgentDecisionEngine.decide()，而是作为 IStrategy 接口的占位实现，
    使加载器对 12 个抽象世界观也能返回一个有效的 IStrategy 实例。
    实际决策仍由 AgentDecisionEngine 主导，本类提供轻量级信号增强。
    """

    # 抽象世界观 → 风格特征
    _WORLDVIEW_STYLE: Dict[str, Dict[str, float]] = {
        "trend_following": {"trend_bias": 1.0, "reversal_bias": 0.0, "breakout_bias": 0.3},
        "mean_reversion": {"trend_bias": 0.0, "reversal_bias": 1.0, "breakout_bias": 0.0},
        "breakout": {"trend_bias": 0.3, "reversal_bias": 0.0, "breakout_bias": 1.0},
        "momentum": {"trend_bias": 0.8, "reversal_bias": 0.0, "breakout_bias": 0.5},
        "value_investing": {"trend_bias": 0.0, "reversal_bias": 0.7, "breakout_bias": 0.0},
        "safe_margin": {"trend_bias": 0.0, "reversal_bias": 0.5, "breakout_bias": 0.0},
        "stat_arbitrage": {"trend_bias": 0.0, "reversal_bias": 0.8, "breakout_bias": 0.0},
        "risk_parity": {"trend_bias": 0.3, "reversal_bias": 0.3, "breakout_bias": 0.0},
        "multi_strategy": {"trend_bias": 0.5, "reversal_bias": 0.5, "breakout_bias": 0.5},
        "narrative_driven": {"trend_bias": 0.6, "reversal_bias": 0.2, "breakout_bias": 0.4},
        "grid_trading": {"trend_bias": 0.0, "reversal_bias": 0.9, "breakout_bias": 0.0},
        "scalping": {"trend_bias": 0.2, "reversal_bias": 0.3, "breakout_bias": 0.2},
        # Phase 8.6: 4 new abstract worldviews (fallback for GeneDrivenStrategy)
        "seasonality_cycle": {"trend_bias": 0.6, "reversal_bias": 0.2, "breakout_bias": 0.3},
        "cppi_portfolio_insurance": {"trend_bias": 0.3, "reversal_bias": 0.5, "breakout_bias": 0.0},
        "llm_agent_trading": {"trend_bias": 0.4, "reversal_bias": 0.4, "breakout_bias": 0.4},
        "microstructure_alpha_trading": {"trend_bias": 0.3, "reversal_bias": 0.3, "breakout_bias": 0.2},
    }

    def update(self, ctx: MarketContext) -> Signal:
        """从基因偏好生成轻量级信号

        本方法不替代 AgentDecisionEngine，仅基于价格历史和基因偏好给出方向性提示。
        实际交易决策仍由 AgentDecisionEngine.decide() 主导。
        """
        style = self._WORLDVIEW_STYLE.get(self.seed_name, {"trend_bias": 0.3, "reversal_bias": 0.3, "breakout_bias": 0.3})

        if len(ctx.price_history) < 5:
            return Signal(sources=[self.seed_name], timestamp=ctx.timestamp)

        # 简单趋势/反转/突破计算
        recent = ctx.price_history[-5:]
        prev = ctx.price_history[-10:-5] if len(ctx.price_history) >= 10 else recent

        recent_avg = sum(recent) / len(recent)
        prev_avg = sum(prev) / len(prev) if prev else recent_avg
        price_change_pct = (recent_avg - prev_avg) / prev_avg if prev_avg else 0.0

        # 趋势信号：上涨 → long，下跌 → short
        trend_signal = price_change_pct
        # 反转信号：偏离均值 → 反向
        mean_price = sum(ctx.price_history[-20:]) / max(1, len(ctx.price_history[-20:]))
        reversal_signal = -(ctx.close - mean_price) / mean_price if mean_price else 0.0
        # 突破信号：突破近 20 根 K 线高点/低点
        lookback = ctx.price_history[-20:] if len(ctx.price_history) >= 20 else ctx.price_history
        if lookback:
            high_20 = max(lookback[:-1]) if len(lookback) > 1 else lookback[0]
            low_20 = min(lookback[:-1]) if len(lookback) > 1 else lookback[0]
            if ctx.close > high_20:
                breakout_signal = 0.5
            elif ctx.close < low_20:
                breakout_signal = -0.5
            else:
                breakout_signal = 0.0
        else:
            breakout_signal = 0.0

        # 加权融合
        combined = (
            style["trend_bias"] * trend_signal
            + style["reversal_bias"] * reversal_signal
            + style["breakout_bias"] * breakout_signal
        )

        direction = Direction.LONG if combined > 0.001 else (
            Direction.SHORT if combined < -0.001 else Direction.FLAT
        )
        confidence = min(1.0, abs(combined) * 10.0)  # 缩放至 0-1

        return Signal(
            direction=direction,
            confidence=confidence,
            strength=abs(combined),
            sources=[self.seed_name, "gene_driven"],
            metadata={
                "trend_signal": trend_signal,
                "reversal_signal": reversal_signal,
                "breakout_signal": breakout_signal,
                "style": style,
            },
            timestamp=ctx.timestamp,
        )


# ============================================================================
# StrategyRegistry — 单例加载器（importlib + 缓存）
# ============================================================================


class StrategyRegistry:
    """策略注册表与动态加载器

    单例模式。提供 load(seed_name, gene, deps) -> IStrategy 接口。
    内部使用 importlib 动态加载 35 个具体策略类，对 12 个抽象世界观 fallback 到 GeneDrivenStrategy。

    用法:
        registry = StrategyRegistry()
        strategy = registry.load("funding_rate_arb", gene, deps={"spot_engine": eng})
        signal = strategy.update(market_ctx)

    缓存策略:
      - 模块缓存: _module_cache[module_name] = module_obj（避免重复 import）
      - 类缓存: _class_cache[(module, class)] = class_obj
      - 实例不缓存（每个 Agent 需要独立实例，因 gene 不同）
    """

    # Phase 6D: 沙盘不适配的策略种子（沙盘模式显式跳过，避免静默 fallback）
    # 这些策略需要真实链上数据或外部 API，沙盘环境无法模拟
    # 沙盘模式调用 load() 时，这些种子会直接返回 None 并记录 WARN
    # 注意：空集合必须用 set() 而非 {}（{} 是空 dict）
    SANDBOX_UNSUPPORTED_SEEDS: set = {
        # Domain B: 期权策略（4个）—— 沙盘模式无期权数据源，待 E0 合约审计后接入专用执行域
        "gamma_scalping",
        "dispersion_trading",
        "vrp_harvesting",
        "volatility_surface_arb",
        # Domain C: 链上策略（2个）—— 沙盘模式无链上数据，待专用执行域
        "on_chain_whale_tracker",
        "smart_money_footprint",
        # Domain D: MEV策略（2个）—— 沙盘模式无DEX内存池，待专用执行域
        "mev_sandwich_arbitrage",
        "jit_liquidity_mev",
        # Domain E: 跨链策略（2个）—— 沙盘模式无跨链桥数据，待专用执行域
        "cross_chain_arbitrage",
        "cex_dex_flash_arbitrage",
    }  # Phase E 第一批：仅接入现货/永续策略，其余10个待专用执行域

    _instance: Optional["StrategyRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "StrategyRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._module_cache: Dict[str, Any] = {}
            self._class_cache: Dict[Tuple[str, str], Type[Any]] = {}
            self._load_failures: Dict[str, str] = {}
            self._initialized = True
            # 延迟导入避免循环依赖
            self._worldview_seeds: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_worldview_seeds(self) -> Dict[str, Dict[str, Any]]:
        """延迟加载 WORLDVIEW_SEEDS（避免循环依赖）"""
        if self._worldview_seeds is None:
            try:
                from .gene_codec import WORLDVIEW_SEEDS
                self._worldview_seeds = WORLDVIEW_SEEDS
            except ImportError:
                try:
                    from gene_codec import WORLDVIEW_SEEDS
                    self._worldview_seeds = WORLDVIEW_SEEDS
                except ImportError as e:
                    logger.error("无法加载 WORLDVIEW_SEEDS: %s", e)
                    self._worldview_seeds = {}
        return self._worldview_seeds

    def _get_sandbox_path(self) -> str:
        """获取 sandbox_trading 包的目录路径"""
        # 优先通过 __file__ 定位
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.getcwd()

    def _load_module(self, module_name: str) -> Optional[Any]:
        """加载 Python 模块（带缓存）"""
        if module_name in self._module_cache:
            return self._module_cache[module_name]

        # 尝试 3 种导入路径
        candidates = [
            f"hermes_v6.sandbox_trading.{module_name}",
            f"sandbox_trading.{module_name}",
            module_name,
        ]

        for cand in candidates:
            try:
                mod = importlib.import_module(cand)
                self._module_cache[module_name] = mod
                return mod
            except ImportError:
                continue
            except Exception as e:
                logger.debug("加载模块 %s 失败 (%s): %s", cand, type(e).__name__, e)
                continue

        # 最后尝试从文件路径加载
        sandbox_path = self._get_sandbox_path()
        file_path = os.path.join(sandbox_path, f"{module_name}.py")
        if os.path.exists(file_path):
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self._module_cache[module_name] = mod
                    return mod
            except Exception as e:
                logger.debug("从文件加载 %s 失败: %s", file_path, e)

        return None

    def _load_class(self, module_name: str, class_name: str) -> Optional[Type[Any]]:
        """加载策略类（带缓存）"""
        cache_key = (module_name, class_name)
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]

        mod = self._load_module(module_name)
        if mod is None:
            return None

        cls = getattr(mod, class_name, None)
        if cls is None or not isinstance(cls, type):
            return None

        self._class_cache[cache_key] = cls
        return cls

    def _instantiate_strategy(
        self,
        cls: Type[Any],
        key_params: Dict[str, Any],
        deps: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """实例化策略类（处理异构 __init__ 签名）"""
        import inspect

        deps = deps or {}

        try:
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.keys())[1:]  # 跳过 self
        except (ValueError, TypeError):
            params = []

        if not params:
            try:
                return cls()
            except Exception as e:
                logger.debug("实例化 %s 无参数失败: %s", cls.__name__, e)
                return None

        # 按参数名匹配
        kwargs: Dict[str, Any] = {}
        param_set = set(params)

        # 优先从 key_params 取值
        for k, v in key_params.items():
            if k in param_set:
                kwargs[k] = v

        # 再从 deps 取外部依赖（如 spot_engine/perp_engine/symbol 等）
        dep_mapping = {
            "spot_engine": "spot_engine",
            "perp_engine": "perp_engine",
            "engine": "engine",
            "exchange": "exchange",
            "config": "config",
            "symbol": "symbol",
        }
        for dep_key, param_name in dep_mapping.items():
            if param_name in param_set and param_name not in kwargs and dep_key in deps:
                kwargs[param_name] = deps[dep_key]

        # symbol 默认值
        if "symbol" in param_set and "symbol" not in kwargs:
            kwargs["symbol"] = deps.get("symbol", "BTC-USDT")

        # config 默认值（合并 key_params）
        # Phase 7B 修复: 检测 config 参数的类型提示
        # - 如果是 dataclass（如 BasisConfig/MMConfig/PairTradingConfig），自动构造该 dataclass
        #   策略内部用 self.config.xxx 属性访问，dict 会失败（basis_arbitrage/perp_market_making）
        # - 如果是 dict 或无类型提示（如 funding_rate_arb 用 config.get()），传 dict(key_params)
        # Phase 7B 修复2: from __future__ import annotations 使注解变字符串，
        #   需用 typing.get_type_hints() 解析；构造失败时传 None 让策略内部
        #   `config or DefaultConfig()` fallback 生效（dict 是 truthy 会阻断 fallback）
        if "config" in param_set and "config" not in kwargs:
            config_cls = None
            try:
                from dataclasses import is_dataclass as _is_dc
                # 优先: typing.get_type_hints() 解析字符串注解 (PEP 563)
                try:
                    from typing import get_type_hints
                    hints = get_type_hints(cls.__init__)
                    annotation = hints.get("config")
                    if annotation and isinstance(annotation, type) and _is_dc(annotation):
                        config_cls = annotation
                except Exception:
                    pass
                # 回退: 直接访问 annotation (无 __future__ annotations 场景)
                if config_cls is None:
                    sig2 = inspect.signature(cls.__init__)
                    param_annotation = sig2.parameters.get("config")
                    if param_annotation and param_annotation.annotation:
                        annotation = param_annotation.annotation
                        if isinstance(annotation, type) and _is_dc(annotation):
                            config_cls = annotation
            except Exception:
                pass

            if config_cls is not None:
                try:
                    from dataclasses import fields as _dc_fields
                    field_names = {f.name for f in _dc_fields(config_cls)}
                    config_kwargs = {k: v for k, v in key_params.items() if k in field_names}
                    kwargs["config"] = config_cls(**config_kwargs)
                except Exception as e:
                    # 构造失败（如缺必填字段）→ 传 None 让策略内部 fallback 到默认 config
                    # 不传 dict（truthy 会阻断 `config or DefaultConfig()` fallback）
                    logger.debug("构造 %s dataclass 失败: %s, 传 None 让策略内部 fallback",
                                 config_cls.__name__, e)
                    kwargs["config"] = None
            else:
                # 无类型提示或非 dataclass → 传 dict (funding_rate_arb 等用 config.get() 兼容)
                kwargs["config"] = dict(key_params)

        # 过滤不存在的参数
        filtered = {k: v for k, v in kwargs.items() if k in param_set}

        try:
            return cls(**filtered)
        except Exception as e:
            logger.debug("实例化 %s 带 kwargs 失败: %s", cls.__name__, e)
            # 尝试无参数实例化
            try:
                return cls()
            except Exception:
                return None

    def load(
        self,
        seed_name: str,
        gene: Any,
        deps: Optional[Dict[str, Any]] = None,
    ) -> Optional[IStrategy]:
        """加载策略

        Args:
            seed_name: WORLDVIEW_SEEDS 中的种子名
            gene: AgentGene 实例
            deps: 外部依赖字典（如 {"spot_engine": engine, "symbol": "BTC-USDT"}）

        Returns:
            IStrategy 实例（具体策略适配器 或 GeneDrivenStrategy）；
            沙盘模式下若种子在 SANDBOX_UNSUPPORTED_SEEDS 中则返回 None
        """
        # Phase 6D: 沙盘模式安全网
        sandbox_mode = os.environ.get("HERMES_SANDBOX_MODE", "0") == "1"
        if sandbox_mode and seed_name in self.SANDBOX_UNSUPPORTED_SEEDS:
            logger.warning(
                "策略 %s 在沙盘模式不适用（sandbox_unsupported），跳过加载",
                seed_name,
            )
            return None  # 返回 None 而非 GeneDrivenStrategy，避免空壳交付

        seeds = self._get_worldview_seeds()
        seed = seeds.get(seed_name)

        if seed is None:
            logger.warning("未知种子: %s，使用 GeneDrivenStrategy fallback", seed_name)
            return GeneDrivenStrategy(seed_name, gene, {})

        module_name = seed.get("module")
        class_name = seed.get("class")
        key_params = seed.get("key_params", {})

        # 抽象世界观（无 module/class） → GeneDrivenStrategy
        if not module_name or not class_name:
            return GeneDrivenStrategy(seed_name, gene, key_params)

        # 具体策略 → 动态加载
        cls = self._load_class(module_name, class_name)
        if cls is None:
            self._load_failures[seed_name] = f"无法加载 {module_name}.{class_name}"
            logger.warning("策略 %s 加载失败，fallback 到 GeneDrivenStrategy", seed_name)
            return GeneDrivenStrategy(seed_name, gene, key_params)

        instance = self._instantiate_strategy(cls, key_params, deps)
        if instance is None:
            self._load_failures[seed_name] = f"实例化 {class_name} 失败"
            logger.warning("策略 %s 实例化失败，fallback 到 GeneDrivenStrategy", seed_name)
            return GeneDrivenStrategy(seed_name, gene, key_params)

        adapter = _StrategyAdapter(seed_name, gene, key_params, instance)
        adapter.on_start(deps)
        return adapter

    def list_available_seeds(self) -> Dict[str, Dict[str, Any]]:
        """列出所有可用种子及其加载状态"""
        seeds = self._get_worldview_seeds()
        result: Dict[str, Dict[str, Any]] = {}
        for name, seed in seeds.items():
            has_impl = bool(seed.get("module") and seed.get("class"))
            result[name] = {
                "label": seed.get("label", ""),
                "has_implementation": has_impl,
                "module": seed.get("module", ""),
                "class": seed.get("class", ""),
                "load_failure": self._load_failures.get(name),
            }
        return result

    def record_load_failure(self, name: str, reason: str) -> None:
        """记录策略加载失败（Phase 6A: 封装修复 — 公开接口替代直接访问 _load_failures）

        供外部模块（如 EvolutionLoop.initialize 的 preload_all 健康检查）调用，
        避免直接访问私有属性 _load_failures（违反"模块间通过正式接口通信"铁律）。

        Args:
            name: 种子策略名
            reason: 失败原因
        """
        if name not in self._load_failures:
            self._load_failures[name] = reason

    def preload_all(self, deps: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """预加载所有策略类（仅验证可加载性，不实例化）

        用于启动时健康检查。
        """
        seeds = self._get_worldview_seeds()
        result: Dict[str, bool] = {}
        dummy_gene = type("DummyGene", (), {"worldview_primary": ""})()

        for name, seed in seeds.items():
            module_name = seed.get("module")
            class_name = seed.get("class")
            if not module_name or not class_name:
                result[name] = True  # 抽象世界观始终可用
                continue
            cls = self._load_class(module_name, class_name)
            result[name] = cls is not None

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        seeds = self._get_worldview_seeds()
        total = len(seeds)
        concrete = sum(1 for s in seeds.values() if s.get("module") and s.get("class"))
        abstract = total - concrete
        return {
            "total_seeds": total,
            "concrete_strategies": concrete,
            "abstract_worldviews": abstract,
            "module_cache_size": len(self._module_cache),
            "class_cache_size": len(self._class_cache),
            "load_failures": dict(self._load_failures),
        }


# ============================================================================
# 模块级便捷函数
# ============================================================================

_registry_instance: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """获取 StrategyRegistry 单例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = StrategyRegistry()
    return _registry_instance


def load_strategy(
    seed_name: str,
    gene: Any,
    deps: Optional[Dict[str, Any]] = None,
) -> Optional[IStrategy]:
    """便捷函数：加载策略"""
    return get_registry().load(seed_name, gene, deps)

