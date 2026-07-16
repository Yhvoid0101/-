#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配对交易策略 (Pair Trading Strategy) — 可导入类
==============================================
短板 #3 补齐 (Phase 3.3): 将 _v555_non_directional_pair_trading.py 的核心逻辑
提取为零副作用可导入类, 作为风险分散组件集成到策略组合.

核心痛点:
  - ERR-093: consec_loss≤3 在方向性策略框架下是物理极限
  - ERR-099: v555配对交易实现35倍改善(总亏损0.32%本金), 但年化仅2.4%
  - 价值定位: 配对交易作为风险分散组件(非收益组件), 与方向性策略组合后
    两种策略的连亏期不重叠 → 组合consec_loss降低

设计原则 (用户铁律):
  - "架构优先" — 模块间通过正式接口通信, 不直接访问私有属性
  - "不自建轮子" — 复用 _v555 已验证的核心算法(相关性/对冲比率/Z-score/回测)
  - "零副作用" — 无顶层print, 无sys.stdout.reconfigure, 无导入时文件I/O
  - "ERR-099应用" — 配对交易定位为风险分散, 非收益组件

来源:
  - _v555_non_directional_pair_trading.py (558行, 独立诊断脚本)
  - Gatefider 2006: 统计套利中的配对交易
  - ERR-099: v555配对交易突破, 35倍consec_loss改善

使用方式:
  from pair_trading_strategy import PairTradingStrategy

  strategy = PairTradingStrategy(correlation_threshold=0.7)
  pairs = strategy.find_pairs(price_data)
  signals = strategy.generate_signals(zscore_series)
  trades = strategy.backtest_pair(symbol_a, symbol_b, klines_a, klines_b)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PairTradingStrategy")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class PairTrade:
    """配对交易记录

    Attributes:
        entry_bar: 入场K线索引
        exit_bar: 出场K线索引
        direction: 1=做多价差, -1=做空价差
        entry_z: 入场时Z-score
        exit_z: 出场时Z-score
        pnl_pct: 盈亏百分比 (相对于本金)
        pnl_usd: 盈亏金额 (USD)
        hold_bars: 持仓K线数
        symbol_a: symbol A 名称
        symbol_b: symbol B 名称
        exit_reason: 出场原因
    """
    entry_bar: int
    exit_bar: int
    direction: int  # 1=做多价差, -1=做空价差
    entry_z: float
    exit_z: float
    pnl_pct: float
    pnl_usd: float
    hold_bars: int
    symbol_a: str
    symbol_b: str
    exit_reason: str = ""


@dataclass
class PairSignal:
    """配对交易信号

    Attributes:
        symbol_a: symbol A
        symbol_b: symbol B
        z_score: 当前Z-score
        action: "long_spread" / "short_spread" / "exit" / "hold"
        confidence: 信号置信度 (0-1)
    """
    symbol_a: str
    symbol_b: str
    z_score: float
    action: str  # "long_spread" / "short_spread" / "exit" / "hold"
    confidence: float = 0.0


@dataclass
class PairResult:
    """配对交易回测结果

    Attributes:
        symbol_a: symbol A
        symbol_b: symbol B
        correlation: 相关系数
        n_trades: 交易次数
        win_rate_pct: 胜率
        total_pnl_usd: 总盈亏
        max_consec_loss: 最大连续亏损次数
        max_consec_loss_pct: 最长段总亏损占本金百分比
        trades: 交易记录列表
    """
    symbol_a: str
    symbol_b: str
    correlation: float
    n_trades: int = 0
    win_rate_pct: float = 0.0
    total_pnl_usd: float = 0.0
    max_consec_loss: int = 0
    max_consec_loss_pct: float = 0.0
    trades: List[PairTrade] = field(default_factory=list)


# ============================================================================
# 核心策略类
# ============================================================================

class PairTradingStrategy:
    """配对交易策略 — 非方向性统计套利

    原理:
      1. 找高相关的symbol对(如BTC-ETH, |ρ|>0.7)
      2. 计算价差 spread = log(price_A) - β×log(price_B)
      3. 标准化 Z-score = (spread - rolling_mean) / rolling_std
      4. Z>2: 做空价差(做空A + 做多B), 期望价差回归
      5. Z<-2: 做多价差(做多A + 做空B), 期望价差回归
      6. Z回归0.5: 平仓获利
      7. Z>3.5: 止损(价差持续偏离)

    优势:
      - 不依赖市场方向(只依赖价差回归)
      - 市场中性(多空对冲, 不暴露方向风险)
      - consec_loss特性与方向性策略不同(连亏发生在价差持续偏离时)

    ERR-099应用:
      配对交易年化仅2.4%, 但consec_loss改善35倍.
      定位为风险分散组件, 非收益组件.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        rolling_window: int = 500,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 3.5,
        lookback: int = 200,
        max_hold_bars: int = 100,
        position_size_pct: float = 2.0,
        initial_capital: float = 10000.0,
    ):
        """初始化配对交易策略

        Args:
            correlation_threshold: 相关性阈值, 高于此值才考虑配对 (默认0.7)
            rolling_window: 相关性计算窗口 (默认500)
            z_entry: Z-score入场阈值 (默认2.0)
            z_exit: Z-score平仓阈值 (默认0.5, 回归到此值平仓)
            z_stop: Z-score止损阈值 (默认3.5)
            lookback: Z-score计算窗口 (默认200)
            max_hold_bars: 最大持仓K线数 (默认100)
            position_size_pct: 仓位占本金百分比 (默认2.0)
            initial_capital: 初始本金 (默认10000)
        """
        self.correlation_threshold = correlation_threshold
        self.rolling_window = rolling_window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.lookback = lookback
        self.max_hold_bars = max_hold_bars
        self.position_size_pct = position_size_pct
        self.initial_capital = initial_capital

    # ========================================================================
    # 静态计算方法 (纯函数, 无副作用)
    # ========================================================================

    @staticmethod
    def calc_log_prices(klines: List[Dict]) -> List[float]:
        """提取对数价格序列

        Args:
            klines: K线数据列表, 每项含 "close" 字段

        Returns:
            对数价格列表
        """
        return [math.log(k["close"]) for k in klines if k.get("close", 0) > 0]

    @staticmethod
    def calc_rolling_correlation(
        prices_a: List[float],
        prices_b: List[float],
        window: int = 500,
    ) -> float:
        """计算两个价格序列的滚动相关性

        Args:
            prices_a: 价格序列A
            prices_b: 价格序列B
            window: 计算窗口

        Returns:
            相关系数 (-1 ~ +1)
        """
        n = min(len(prices_a), len(prices_b))
        if n < window:
            window = n
        if window < 50:
            return 0.0

        a = prices_a[-window:]
        b = prices_b[-window:]

        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)

        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(len(a))) / len(a)
        std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a) / len(a))
        std_b = math.sqrt(sum((x - mean_b) ** 2 for x in b) / len(b))

        if std_a == 0 or std_b == 0:
            return 0.0

        return cov / (std_a * std_b)

    @staticmethod
    def calc_hedge_ratio(prices_a: List[float], prices_b: List[float]) -> float:
        """用OLS计算对冲比率β

        β = Cov(A, B) / Var(B)
        spread = log(A) - β × log(B)

        Args:
            prices_a: 对数价格序列A
            prices_b: 对数价格序列B

        Returns:
            对冲比率β
        """
        n = min(len(prices_a), len(prices_b))
        a = prices_a[-n:]
        b = prices_b[-n:]

        mean_b = sum(b) / len(b)
        mean_a = sum(a) / len(a)

        cov = sum((b[i] - mean_b) * (a[i] - mean_a) for i in range(n))
        var_b = sum((b[i] - mean_b) ** 2 for i in range(n))

        if var_b == 0:
            return 1.0
        return cov / var_b

    @staticmethod
    def calc_spread_series(
        prices_a: List[float],
        prices_b: List[float],
        beta: float,
    ) -> List[float]:
        """计算价差序列

        spread = log(A) - β × log(B)

        Args:
            prices_a: 对数价格序列A
            prices_b: 对数价格序列B
            beta: 对冲比率

        Returns:
            价差序列
        """
        n = min(len(prices_a), len(prices_b))
        return [prices_a[-n + i] - beta * prices_b[-n + i] for i in range(n)]

    def compute_zscore(
        self,
        spread: List[float],
        window: Optional[int] = None,
    ) -> List[float]:
        """计算滚动Z-score

        Z = (spread - rolling_mean) / rolling_std

        Args:
            spread: 价差序列
            window: 计算窗口 (None=用self.lookback)

        Returns:
            Z-score序列 (前window个为0.0)
        """
        if window is None:
            window = self.lookback

        z_scores: List[float] = []
        for i in range(len(spread)):
            if i < window:
                z_scores.append(0.0)
                continue
            window_data = spread[i - window:i]
            mean = sum(window_data) / len(window_data)
            std = math.sqrt(sum((x - mean) ** 2 for x in window_data) / len(window_data))
            if std == 0:
                z_scores.append(0.0)
            else:
                z_scores.append((spread[i] - mean) / std)
        return z_scores

    # ========================================================================
    # 公开接口
    # ========================================================================

    def find_pairs(
        self,
        price_data: Dict[str, List[Dict]],
        correlation_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """寻找高相关交易对

        Args:
            price_data: {symbol: klines} 字典
            correlation_threshold: 相关性阈值 (None=用self.correlation_threshold)

        Returns:
            高相关对列表, 每项含 symbol_a, symbol_b, correlation
            按相关性绝对值降序排列
        """
        if correlation_threshold is None:
            correlation_threshold = self.correlation_threshold

        symbols = list(price_data.keys())
        log_prices = {
            s: self.calc_log_prices(price_data[s])
            for s in symbols
            if price_data[s]
        }

        correlations: List[Dict[str, Any]] = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]
                if sym_a not in log_prices or sym_b not in log_prices:
                    continue
                min_len = min(len(log_prices[sym_a]), len(log_prices[sym_b]))
                if min_len < 50:
                    continue
                corr = self.calc_rolling_correlation(
                    log_prices[sym_a][-min_len:],
                    log_prices[sym_b][-min_len:],
                    self.rolling_window,
                )
                correlations.append({
                    "symbol_a": sym_a,
                    "symbol_b": sym_b,
                    "correlation": round(corr, 4),
                })

        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        high_corr = [
            c for c in correlations
            if abs(c["correlation"]) >= correlation_threshold
        ]
        return high_corr

    def generate_signals(
        self,
        zscore: float,
        entry: Optional[float] = None,
        exit: Optional[float] = None,
        stop: Optional[float] = None,
    ) -> str:
        """根据Z-score生成交易信号

        Args:
            zscore: 当前Z-score值
            entry: 入场阈值 (None=用self.z_entry)
            exit: 平仓阈值 (None=用self.z_exit)
            stop: 止损阈值 (None=用self.z_stop)

        Returns:
            信号: "short_spread" / "long_spread" / "exit" / "hold"
        """
        if entry is None:
            entry = self.z_entry
        if exit is None:
            exit = self.z_exit
        if stop is None:
            stop = self.z_stop

        if zscore > entry:
            return "short_spread"  # 做空价差(做空A + 做多B)
        elif zscore < -entry:
            return "long_spread"   # 做多价差(做多A + 做空B)
        elif abs(zscore) <= exit:
            return "exit"          # Z回归, 平仓
        elif abs(zscore) >= stop:
            return "exit"          # 止损
        else:
            return "hold"

    def backtest_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        klines_a: List[Dict],
        klines_b: List[Dict],
        start_bar: Optional[int] = None,
    ) -> List[PairTrade]:
        """配对交易回测

        策略:
        - Z>2: 做空价差(做空A + 做多B)
        - Z<-2: 做多价差(做多A + 做空B)
        - Z回归0.5: 平仓
        - Z>3.5: 止损
        - 最大持仓时间: max_hold_bars

        Args:
            symbol_a: symbol A 名称
            symbol_b: symbol B 名称
            klines_a: symbol A 的K线数据
            klines_b: symbol B 的K线数据
            start_bar: 开始回测的K线索引 (None=用self.lookback)

        Returns:
            交易记录列表
        """
        if start_bar is None:
            start_bar = self.lookback

        # 对齐数据
        n = min(len(klines_a), len(klines_b))
        prices_a = [math.log(klines_a[i]["close"]) for i in range(n) if klines_a[i].get("close", 0) > 0]
        prices_b = [math.log(klines_b[i]["close"]) for i in range(n) if klines_b[i].get("close", 0) > 0]

        n = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-n:]
        prices_b = prices_b[-n:]

        if n < start_bar + self.lookback:
            return []

        # 计算对冲比率和价差
        beta = self.calc_hedge_ratio(prices_a, prices_b)
        spread = self.calc_spread_series(prices_a, prices_b, beta)
        z_scores = self.compute_zscore(spread, self.lookback)

        trades: List[PairTrade] = []
        in_position = False
        entry_bar = 0
        entry_z = 0.0
        direction = 0
        entry_price_a = 0.0
        entry_price_b = 0.0
        capital = self.initial_capital

        for i in range(start_bar, n):
            z = z_scores[i]

            # 出场检查
            if in_position:
                current_price_a = klines_a[i]["close"]
                current_price_b = klines_b[i]["close"]

                # 计算当前价差变化
                delta_a = math.log(current_price_a) - math.log(entry_price_a)
                delta_b = math.log(current_price_b) - math.log(entry_price_b)

                if direction == 1:  # 做多价差
                    pnl_pct = delta_a - beta * delta_b
                else:  # 做空价差
                    pnl_pct = -delta_a + beta * delta_b

                should_exit = False
                exit_reason = ""

                # Z回归平仓
                if direction == 1 and z <= self.z_exit:
                    should_exit = True
                    exit_reason = "Z回归平仓"
                elif direction == -1 and z >= -self.z_exit:
                    should_exit = True
                    exit_reason = "Z回归平仓"

                # 止损
                if direction == 1 and z <= -self.z_stop:
                    should_exit = True
                    exit_reason = "止损"
                elif direction == -1 and z >= self.z_stop:
                    should_exit = True
                    exit_reason = "止损"

                # 最大持仓时间
                if i - entry_bar >= self.max_hold_bars:
                    should_exit = True
                    exit_reason = "最大持仓时间"

                if should_exit:
                    pnl_usd = pnl_pct * capital * (self.position_size_pct / 100)
                    capital += pnl_usd
                    trades.append(PairTrade(
                        entry_bar=entry_bar,
                        exit_bar=i,
                        direction=direction,
                        entry_z=entry_z,
                        exit_z=z,
                        pnl_pct=pnl_pct * 100,
                        pnl_usd=pnl_usd,
                        hold_bars=i - entry_bar,
                        symbol_a=symbol_a,
                        symbol_b=symbol_b,
                        exit_reason=exit_reason,
                    ))
                    in_position = False
                    direction = 0
                    continue

            # 入场检查
            if not in_position:
                if z > self.z_entry:
                    # 做空价差(做空A + 做多B)
                    direction = -1
                    entry_bar = i
                    entry_z = z
                    entry_price_a = klines_a[i]["close"]
                    entry_price_b = klines_b[i]["close"]
                    in_position = True
                elif z < -self.z_entry:
                    # 做多价差(做多A + 做空B)
                    direction = 1
                    entry_bar = i
                    entry_z = z
                    entry_price_a = klines_a[i]["close"]
                    entry_price_b = klines_b[i]["close"]
                    in_position = True

        return trades

    def evaluate_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        klines_a: List[Dict],
        klines_b: List[Dict],
        correlation: float = 0.0,
    ) -> PairResult:
        """评估单个配对的回测结果

        Args:
            symbol_a: symbol A
            symbol_b: symbol B
            klines_a: symbol A K线
            klines_b: symbol B K线
            correlation: 预计算的相关系数 (0=不提供)

        Returns:
            PairResult: 含统计指标的回测结果
        """
        trades = self.backtest_pair(symbol_a, symbol_b, klines_a, klines_b)

        if not trades:
            return PairResult(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                correlation=correlation,
            )

        n_trades = len(trades)
        wins = [t for t in trades if t.pnl_usd > 0]
        total_pnl = sum(t.pnl_usd for t in trades)
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

        # 计算consec_loss
        sorted_trades = sorted(trades, key=lambda t: t.exit_bar)
        max_consec = 0
        current_consec = 0
        consec_segments: List[List[PairTrade]] = []
        current_segment: List[PairTrade] = []

        for t in sorted_trades:
            if t.pnl_usd < 0:
                current_consec += 1
                current_segment.append(t)
                max_consec = max(max_consec, current_consec)
            else:
                if current_segment:
                    consec_segments.append(current_segment)
                current_consec = 0
                current_segment = []
        if current_segment:
            consec_segments.append(current_segment)

        # 最长段总亏损
        max_consec_loss_usd = 0.0
        max_consec_loss_pct = 0.0
        if consec_segments:
            longest = max(consec_segments, key=len)
            max_consec_loss_usd = abs(sum(t.pnl_usd for t in longest))
            max_consec_loss_pct = max_consec_loss_usd / self.initial_capital * 100

        return PairResult(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=correlation,
            n_trades=n_trades,
            win_rate_pct=round(win_rate, 2),
            total_pnl_usd=round(total_pnl, 2),
            max_consec_loss=max_consec,
            max_consec_loss_pct=round(max_consec_loss_pct, 2),
            trades=trades,
        )

    def evaluate_portfolio(
        self,
        price_data: Dict[str, List[Dict]],
    ) -> Dict[str, Any]:
        """Portfolio级配对交易评估

        Args:
            price_data: {symbol: klines} 字典

        Returns:
            评估报告字典, 含 portfolio_stats, pair_results, verdict
        """
        # Step 1: 寻找高相关对
        high_corr_pairs = self.find_pairs(price_data)

        # Step 2: 对每个高相关对进行回测
        pair_results: List[PairResult] = []
        all_trades: List[PairTrade] = []

        for pair in high_corr_pairs:
            sym_a = pair["symbol_a"]
            sym_b = pair["symbol_b"]
            if sym_a not in price_data or sym_b not in price_data:
                continue

            result = self.evaluate_pair(
                sym_a, sym_b,
                price_data[sym_a], price_data[sym_b],
                pair["correlation"],
            )
            if result.n_trades > 0:
                pair_results.append(result)
                all_trades.extend(result.trades)

        # Step 3: Portfolio级统计
        if not all_trades:
            return {
                "portfolio_stats": {
                    "n_trades": 0,
                    "win_rate_pct": 0.0,
                    "total_pnl_usd": 0.0,
                    "max_consec_loss": 0,
                    "max_consec_loss_pct": 0.0,
                },
                "pair_results": [],
                "high_corr_pairs": high_corr_pairs,
                "verdict": {"decision": "NO-GO", "reason": "无配对交易trades生成"},
            }

        n_trades_all = len(all_trades)
        wins_all = [t for t in all_trades if t.pnl_usd > 0]
        total_pnl_all = sum(t.pnl_usd for t in all_trades)
        win_rate_all = len(wins_all) / n_trades_all * 100

        # Portfolio级consec_loss
        sorted_all = sorted(all_trades, key=lambda t: t.exit_bar)
        max_consec_all = 0
        current_consec_all = 0
        consec_segments_all: List[List[PairTrade]] = []
        current_segment_all: List[PairTrade] = []

        for t in sorted_all:
            if t.pnl_usd < 0:
                current_consec_all += 1
                current_segment_all.append(t)
                max_consec_all = max(max_consec_all, current_consec_all)
            else:
                if current_segment_all:
                    consec_segments_all.append(current_segment_all)
                current_consec_all = 0
                current_segment_all = []
        if current_segment_all:
            consec_segments_all.append(current_segment_all)

        # 最长段总亏损
        max_consec_loss_usd_all = 0.0
        max_consec_loss_pct_all = 0.0
        if consec_segments_all:
            longest_all = max(consec_segments_all, key=len)
            max_consec_loss_usd_all = abs(sum(t.pnl_usd for t in longest_all))
            max_consec_loss_pct_all = max_consec_loss_usd_all / self.initial_capital * 100

        # 单次最大亏损
        max_single_loss = max(abs(t.pnl_usd) for t in all_trades) if all_trades else 0
        max_single_loss_pct = max_single_loss / self.initial_capital * 100

        # 裁决 (ERR-093应用: consec_loss≤3为GO, 金额≤5%为GO)
        if max_consec_all <= 3:
            decision = "GO"
            reason = f"配对交易consec_loss={max_consec_all}≤3"
        elif max_consec_loss_pct_all <= 5.0:
            decision = "GO"
            reason = f"配对交易金额指标通过: 最长段{max_consec_loss_pct_all:.2f}%≤5%"
        elif max_consec_all < 10:
            decision = "DEGRADED"
            reason = f"配对交易consec_loss={max_consec_all}优于方向性(10)但仍>3"
        else:
            decision = "NO-GO"
            reason = f"配对交易consec_loss={max_consec_all}未改善"

        return {
            "portfolio_stats": {
                "n_trades": n_trades_all,
                "win_rate_pct": round(win_rate_all, 2),
                "total_pnl_usd": round(total_pnl_all, 2),
                "max_consec_loss": max_consec_all,
                "max_consec_loss_usd": round(max_consec_loss_usd_all, 2),
                "max_consec_loss_pct": round(max_consec_loss_pct_all, 2),
                "max_single_loss_pct": round(max_single_loss_pct, 2),
                "n_consec_segments": len(consec_segments_all),
            },
            "pair_results": [
                {
                    "symbol_a": r.symbol_a,
                    "symbol_b": r.symbol_b,
                    "correlation": r.correlation,
                    "n_trades": r.n_trades,
                    "win_rate_pct": r.win_rate_pct,
                    "total_pnl_usd": r.total_pnl_usd,
                    "max_consec_loss": r.max_consec_loss,
                    "max_consec_loss_pct": r.max_consec_loss_pct,
                }
                for r in pair_results
            ],
            "high_corr_pairs": high_corr_pairs,
            "verdict": {"decision": decision, "reason": reason},
        }
