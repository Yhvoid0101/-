#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_market_data_fetcher.py — 真实市场数据获取器 v1.0

来源:
  - ccxt 4.5.59 (Binance/OKX/Coinbase公开API)
  - "不要模拟牛逼,实盘亏钱"原则 — 必须使用真实市场数据验证

用途:
  - 从Binance获取BTC/ETH/SOL/BNB/XRP的真实1h K线数据
  - 2190根K线 = 91天 ≈ 3个月
  - 替代realistic_data_generator的合成数据

依赖:
  - ccxt (已安装 4.5.59)
  - pandas (已安装 3.0.3)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import ccxt

logger = logging.getLogger("hermes.real_data")


class RealMarketDataFetcher:
    """真实市场数据获取器 — 从Binance获取历史K线"""

    # 支持的symbol映射 (ccxt格式)
    SYMBOL_MAP = {
        "BTC-USDT": "BTC/USDT",
        "ETH-USDT": "ETH/USDT",
        "SOL-USDT": "SOL/USDT",
        "BNB-USDT": "BNB/USDT",
        "XRP-USDT": "XRP/USDT",
    }

    def __init__(self, exchange_name: str = "binance"):
        """初始化交易所连接

        Args:
            exchange_name: 交易所名称 (binance/okx/coinbase)
        """
        if exchange_name == "binance":
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
        elif exchange_name == "okx":
            self.exchange = ccxt.okx({
                "enableRateLimit": True,
            })
        else:
            self.exchange = getattr(ccxt, exchange_name)({
                "enableRateLimit": True,
            })
        logger.info("初始化%s交易所连接", exchange_name)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        n_bars: int = 2190,
    ) -> List[Dict[str, Any]]:
        """获取历史K线数据

        Args:
            symbol: 交易对 (BTC-USDT格式,自动转换为BTC/USDT)
            timeframe: K线周期 (1h/4h/1d)
            n_bars: K线数量

        Returns:
            K线字典列表 [{timestamp, open, high, low, close, volume}, ...]
        """
        ccxt_symbol = self.SYMBOL_MAP.get(symbol, symbol.replace("-", "/"))

        logger.info("获取%s %s K线数据 (%d根)...", ccxt_symbol, timeframe, n_bars)

        # ccxt的fetch_ohlcv一次最多1000根,需要分批获取
        all_bars: List[List[Any]] = []
        remaining = n_bars
        since = None  # 从最新数据开始往回获取

        # 先获取最新的一批数据
        try:
            bars = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=min(1000, remaining))
            if not bars:
                logger.warning("%s未返回数据", ccxt_symbol)
                return []

            all_bars.extend(bars)
            remaining -= len(bars)

            # 往前获取更多数据
            while remaining > 0 and bars:
                # 使用最早一根K线的时间作为since
                earliest_ts = bars[0][0]
                bars = self.exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe, since=earliest_ts - (remaining * 3600 * 1000),
                    limit=min(1000, remaining),
                )
                if not bars or len(bars) == 1:
                    break
                # 去重 (可能重叠)
                existing_ts = {b[0] for b in all_bars}
                new_bars = [b for b in bars if b[0] not in existing_ts]
                if not new_bars:
                    break
                all_bars.extend(new_bars)
                remaining -= len(new_bars)
                time.sleep(0.5)  # 速率限制

        except Exception as e:
            logger.error("获取%s数据失败: %s", ccxt_symbol, e)
            return []

        # 排序并截取
        all_bars.sort(key=lambda x: x[0])
        if len(all_bars) > n_bars:
            all_bars = all_bars[-n_bars:]

        # 转换为字典格式
        result = []
        for bar in all_bars:
            result.append({
                "timestamp": bar[0] / 1000.0,  # ms → s
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": float(bar[5]),
            })

        logger.info("%s: 获取%d根K线, 价格范围 $%.2f - $%.2f",
                    ccxt_symbol, len(result),
                    min(b["close"] for b in result) if result else 0,
                    max(b["close"] for b in result) if result else 0)

        return result

    def fetch_multiple(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        n_bars: int = 2190,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """获取多个symbol的历史数据

        Args:
            symbols: symbol列表 (BTC-USDT格式)
            timeframe: K线周期
            n_bars: K线数量

        Returns:
            {symbol: bars_list, ...}
        """
        result = {}
        for sym in symbols:
            bars = self.fetch_ohlcv(sym, timeframe, n_bars)
            if bars:
                result[sym] = bars
            time.sleep(1)  # symbol间间隔
        return result


def fetch_real_data_to_cache(
    symbols: Optional[List[str]] = None,
    n_bars: int = 2190,
    cache_file: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """获取真实数据并缓存到本地文件

    Args:
        symbols: symbol列表
        n_bars: K线数量
        cache_file: 缓存文件路径

    Returns:
        {symbol: bars_list, ...}
    """
    import json
    import os

    if symbols is None:
        symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT"]

    if cache_file is None:
        cache_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_real_market_data.json",
        )

    # 检查缓存
    if os.path.exists(cache_file):
        cache_age = time.time() - os.path.getmtime(cache_file)
        if cache_age < 86400:  # 缓存<24小时
            logger.info("使用缓存数据(%s, %.1f小时前)", cache_file, cache_age / 3600)
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

    # 获取新数据
    fetcher = RealMarketDataFetcher("binance")
    data = fetcher.fetch_multiple(symbols, "1h", n_bars)

    if data:
        # 保存缓存
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("已缓存%d个symbol的数据到%s", len(data), cache_file)

    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=== 真实市场数据获取器 ===")
    data = fetch_real_data_to_cache()
    for sym, bars in data.items():
        if bars:
            print(f"{sym}: {len(bars)}根K线, 起始${bars[0]['close']:.2f}, 结束${bars[-1]['close']:.2f}")
