# -*- coding: utf-8 -*-
"""
真实历史数据接入模块 (Real Historical Data Provider) — v2.4深度进化

解决"合成数据再好也不是真实市场"的根本问题。

数据源：
  1. yfinance: Yahoo Finance公开API
     - BTC-USD, ETH-USD, AAPL, SPY 等
     - 日线/小时线数据
     - 免费且稳定

  2. 缓存机制：本地JSON缓存，避免重复下载

  3. 回退机制：网络不可用时回退到合成数据

核心原则：
  - 真实历史数据是反过拟合的最终验证
  - 合成数据用于进化训练，真实数据用于最终验证
  - 进化出的策略必须在真实历史数据上通过DSR/PBO验证
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger("hermes.real_data_provider")


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class HistoricalData:
    """历史数据"""

    symbol: str
    bars: List[Dict[str, float]]
    source: str  # "yfinance" / "cache" / "synthetic"
    start_date: str
    end_date: str
    n_bars: int


# ============================================================================
# 真实历史数据提供者
# ============================================================================


class RealDataProvider:
    """真实历史数据提供者

    使用yfinance获取真实历史K线数据。

    使用方式：
      provider = RealDataProvider(cache_dir="./data_cache")
      data = provider.get_historical_data("BTC-USD", period="2y", interval="1d")
      bars = data.bars  # [{"open":..., "high":..., "low":..., "close":..., "volume":...}, ...]
    """

    # 支持的交易对
    SUPPORTED_SYMBOLS = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "AAPL": "Apple",
        "SPY": "S&P 500 ETF",
        "GLD": "Gold ETF",
        "QQQ": "Nasdaq ETF",
    }

    def __init__(
        self,
        cache_dir: str = "./data_cache",
        cache_ttl_hours: int = 24,  # 缓存有效期
    ):
        """
        Args:
            cache_dir: 缓存目录
            cache_ttl_hours: 缓存有效期（小时）
        """
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours
        os.makedirs(cache_dir, exist_ok=True)

    def get_historical_data(
        self,
        symbol: str = "BTC-USD",
        period: str = "2y",  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: str = "1d",  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    ) -> HistoricalData:
        """获取历史数据

        Args:
            symbol: 交易对（如BTC-USD）
            period: 时间跨度
            interval: K线间隔

        Returns:
            HistoricalData
        """
        # 尝试从缓存加载
        cached = self._load_from_cache(symbol, period, interval)
        if cached is not None:
            logger.info(f"从缓存加载 {symbol}: {cached.n_bars}根K线")
            return cached

        # 尝试从yfinance下载
        try:
            data = self._download_from_yfinance(symbol, period, interval)
            if data is not None:
                self._save_to_cache(data, symbol, period, interval)
                logger.info(f"从yfinance下载 {symbol}: {data.n_bars}根K线")
                return data
        except Exception as e:
            logger.warning(f"yfinance下载失败 {symbol}: {e}")

        # 回退到合成数据
        logger.warning(f"回退到合成数据 {symbol}")
        return self._generate_synthetic_data(symbol, period, interval)

    def _download_from_yfinance(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> Optional[HistoricalData]:
        """从yfinance下载数据"""
        try:
            import yfinance as yf
            import socket
            # 设置socket超时，防止长时间卡住
            socket.setdefaulttimeout(15)
        except ImportError:
            logger.error("yfinance未安装，请运行: pip install yfinance")
            return None

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
        except Exception as e:
            logger.warning(f"yfinance下载异常 {symbol}: {e}")
            return None
        finally:
            socket.setdefaulttimeout(None)  # 恢复默认超时

        if df.empty:
            logger.warning(f"yfinance返回空数据 {symbol}")
            return None

        bars = []
        for timestamp, row in df.iterrows():
            bar = {
                "timestamp": float(timestamp.timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]) if "Volume" in row else 0.0,
            }
            bars.append(bar)

        if not bars:
            return None

        start_date = str(df.index[0].date()) if hasattr(df.index[0], "date") else str(df.index[0])
        end_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])

        return HistoricalData(
            symbol=symbol,
            bars=bars,
            source="yfinance",
            start_date=start_date,
            end_date=end_date,
            n_bars=len(bars),
        )

    def _load_from_cache(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> Optional[HistoricalData]:
        """从缓存加载"""
        cache_file = self._get_cache_path(symbol, period, interval)
        if not os.path.exists(cache_file):
            return None

        # 检查缓存是否过期
        mtime = os.path.getmtime(cache_file)
        age_hours = (time.time() - mtime) / 3600
        if age_hours > self.cache_ttl_hours:
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return HistoricalData(
                symbol=data["symbol"],
                bars=data["bars"],
                source="cache",
                start_date=data["start_date"],
                end_date=data["end_date"],
                n_bars=len(data["bars"]),
            )
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None

    def _save_to_cache(
        self,
        data: HistoricalData,
        symbol: str,
        period: str,
        interval: str,
    ) -> None:
        """保存到缓存"""
        cache_file = self._get_cache_path(symbol, period, interval)
        try:
            cache_data = {
                "symbol": data.symbol,
                "bars": data.bars,
                "start_date": data.start_date,
                "end_date": data.end_date,
                "cached_at": datetime.now().isoformat(),
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _get_cache_path(self, symbol: str, period: str, interval: str) -> str:
        """获取缓存文件路径"""
        safe_symbol = symbol.replace("-", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe_symbol}_{period}_{interval}.json")

    def _generate_synthetic_data(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> HistoricalData:
        """生成合成数据（回退方案）

        使用Student-t分布+GARCH波动率，生成接近真实市场的合成数据。
        """
        from .realistic_data_generator import RealisticMarketDataGenerator, RealisticDataConfig

        # 根据period估算K线数
        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
        }.get(period, 365)

        # 根据interval估算每天K线数
        bars_per_day = {
            "1d": 1, "1h": 24, "30m": 48, "15m": 96,
            "5m": 288, "1m": 1440,
        }.get(interval, 1)

        n_bars = min(period_days * bars_per_day, 5000)  # 限制最大K线数

        config = RealisticDataConfig(seed=42)
        generator = RealisticMarketDataGenerator(config=config)
        bars = generator.generate(
            symbol=symbol,
            n_bars=n_bars,
            start_price=100.0,
        )

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")

        return HistoricalData(
            symbol=symbol,
            bars=bars,
            source="synthetic",
            start_date=start_date,
            end_date=end_date,
            n_bars=len(bars),
        )

    def get_multiple_symbols(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> Dict[str, HistoricalData]:
        """获取多个交易对的历史数据

        Args:
            symbols: 交易对列表
            period: 时间跨度
            interval: K线间隔

        Returns:
            {symbol: HistoricalData}
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_historical_data(symbol, period, interval)
        return results

    def get_train_test_split(
        self,
        symbol: str = "BTC-USD",
        period: str = "5y",
        interval: str = "1d",
        train_ratio: float = 0.7,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """获取训练/测试集分割

        用于Walk-Forward验证：
          - 训练集：进化Agent策略
          - 测试集：OOS验证

        Args:
            symbol: 交易对
            period: 时间跨度
            interval: K线间隔
            train_ratio: 训练集比例

        Returns:
            (train_bars, test_bars)
        """
        data = self.get_historical_data(symbol, period, interval)
        n = len(data.bars)
        split = int(n * train_ratio)
        return data.bars[:split], data.bars[split:]


# ============================================================================
# 便捷函数
# ============================================================================


def get_btc_data(period: str = "2y", interval: str = "1d") -> HistoricalData:
    """获取BTC历史数据"""
    provider = RealDataProvider()
    return provider.get_historical_data("BTC-USD", period, interval)


def get_eth_data(period: str = "2y", interval: str = "1d") -> HistoricalData:
    """获取ETH历史数据"""
    provider = RealDataProvider()
    return provider.get_historical_data("ETH-USD", period, interval)


def get_crypto_data(
    symbols: List[str] = None,
    period: str = "2y",
    interval: str = "1d",
) -> Dict[str, HistoricalData]:
    """获取加密货币历史数据

    Args:
        symbols: 交易对列表（默认["BTC-USD", "ETH-USD"]）
        period: 时间跨度
        interval: K线间隔

    Returns:
        {symbol: HistoricalData}
    """
    if symbols is None:
        symbols = ["BTC-USD", "ETH-USD"]
    provider = RealDataProvider()
    return provider.get_multiple_symbols(symbols, period, interval)
