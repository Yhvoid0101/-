#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate.io 数据源模块 — 统一数据获取接口

提供统一的 Gate.io API 数据获取函数，供 auto_data_fetcher_service 和各 Gate 使用。

Gate.io API 端点 (已验证可用):
  1. Spot Ticker:      /spot/tickers?currency_pair=BTC_USDT
  2. Spot OrderBook:   /spot/order_book?currency_pair=BTC_USDT&limit=20
  3. Futures Contract: /futures/usdt/contracts/BTC_USDT
  4. Futures OrderBook:/futures/usdt/order_book?settle=usdt&contract=BTC_USDT&limit=20
  5. Futures Trades:   /futures/usdt/trades?settle=usdt&contract=BTC_USDT&limit=100
  6. Futures Ticker:   /futures/usdt/tickers?settle=usdt&contract=BTC_USDT

Symbol 转换:
  交易系统使用 "BTC-USDT"，Gate.io 使用 "BTC_USDT"。
  convert_symbol("BTC-USDT") → "BTC_USDT"

用法:
    from gateio_data_source import GateIODataSource
    ds = GateIODataSource()
    ticker = ds.get_ticker("futures", "BTC-USDT")
    ob = ds.get_order_book("futures", "BTC-USDT")
    ratio = ds.get_taker_volume_ratio("BTC-USDT")

仅依赖 Python 标准库，不导入交易系统任何模块（避免循环引用）。
"""
from __future__ import annotations

import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.gateio_data_source")


class GateIODataSource:
    """Gate.io 统一数据源

    提供 spot/futures 两种市场的 ticker、order_book、trades 数据获取，
    以及 futures 专属的 funding_rate、mark_price、taker_volume_ratio。

    所有方法:
      - 成功返回解析后的字典/列表/浮点数
      - 失败返回 None (ticker/order_book/trades) 或 0.0 (funding_rate/mark_price)
      - 自带 3 次指数退避重试
    """

    BASE_SPOT = "https://api.gateio.ws/api/v4/spot"
    BASE_FUTURES = "https://api.gateio.ws/api/v4/futures/usdt"

    # 重试配置
    MAX_RETRIES = 3
    RETRY_BACKOFF = [1, 2, 4]

    # 请求超时 (秒)
    REQUEST_TIMEOUT = 15

    # ==================== 内部工具 ====================

    @staticmethod
    def convert_symbol(symbol: str) -> str:
        """将交易系统 symbol 转换为 Gate.io 格式

        "BTC-USDT" → "BTC_USDT"
        "BTCUSDT"  → "BTC_USDT"
        "BTC"      → "BTC_USDT"
        "BTC/USDT" → "BTC_USDT"
        """
        if not symbol:
            return "BTC_USDT"
        s = symbol.upper().strip()
        # 去除分隔符
        s = s.replace("-", "_").replace("/", "_")
        # 如果没有下划线且以 USDT 结尾，插入下划线
        if "_" not in s and s.endswith("USDT"):
            s = s[:-4] + "_USDT"
        # 如果没有 USDT 后缀，添加
        if not s.endswith("_USDT"):
            base = s.split("_")[0]
            s = base + "_USDT"
        return s

    @staticmethod
    def _file_symbol(symbol: str) -> str:
        """转换为文件名用的 symbol (无分隔符大写, 如 BTCUSDT)

        用于 data/auto_fetched/ 下的文件命名一致性。
        """
        gate_sym = GateIODataSource.convert_symbol(symbol)
        return gate_sym.replace("_", "").upper()

    def _http_get_json(self, url: str, timeout: Optional[int] = None) -> Optional[Any]:
        """发起 GET 请求并返回解析后的 JSON。失败返回 None。"""
        to = timeout or self.REQUEST_TIMEOUT
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "HermesGateIODataSource/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=to) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)

    def _http_get_with_retry(self, url: str) -> Optional[Any]:
        """带指数退避重试的 HTTP GET JSON 请求。最终失败返回 None。"""
        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._http_get_json(url)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    socket.timeout, json.JSONDecodeError, OSError,
                    ssl.SSLError) as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                    logger.debug(
                        "Gate.io 请求失败 (尝试 %d/%d), %ds 后重试: %s - %s",
                        attempt + 1, self.MAX_RETRIES, wait, url, e,
                    )
                    time.sleep(wait)
        logger.warning("Gate.io 请求最终失败 (%d 次): %s - %s",
                       self.MAX_RETRIES, url, last_err)
        return None

    # ==================== 公开接口 ====================

    def get_ticker(self, market: str, symbol: str) -> Optional[Dict[str, Any]]:
        """获取 ticker 数据 (价格、成交量、24h 高低、涨跌幅)

        Args:
            market: "spot" 或 "futures"
            symbol: 交易对 (BTC-USDT / BTCUSDT / BTC)

        Returns:
            统一格式的 ticker 字典:
            {
                "last": float,           # 最新价格
                "volume": float,         # 24h 成交量 (基础币种)
                "quote_volume": float,   # 24h 成交额 (计价币种)
                "high_24h": float,       # 24h 最高
                "low_24h": float,        # 24h 最低
                "change_percentage": float,  # 24h 涨跌幅 (%)
                "funding_rate": float,   # 资金费率 (仅 futures, spot=0)
                "mark_price": float,     # 标记价格 (仅 futures, spot=last)
                "timestamp": int,        # 抓取时间戳
            }
            失败返回 None
        """
        gate_sym = self.convert_symbol(symbol)

        if market == "spot":
            url = f"{self.BASE_SPOT}/tickers?currency_pair={gate_sym}"
            data = self._http_get_with_retry(url)
            if data is None or not isinstance(data, list) or len(data) == 0:
                return None
            raw = data[0]
            try:
                return {
                    "last": float(raw.get("last", 0)),
                    "volume": float(raw.get("base_volume", 0)),
                    "quote_volume": float(raw.get("quote_volume", 0)),
                    "high_24h": float(raw.get("high_24h", 0)),
                    "low_24h": float(raw.get("low_24h", 0)),
                    "change_percentage": float(raw.get("change_percentage", 0)),
                    "funding_rate": 0.0,
                    "mark_price": float(raw.get("last", 0)),
                    "timestamp": int(time.time()),
                }
            except (ValueError, TypeError) as e:
                logger.warning("解析 spot ticker 失败: %s", e)
                return None

        elif market == "futures":
            url = (f"{self.BASE_FUTURES}/tickers?settle=usdt&contract={gate_sym}")
            data = self._http_get_with_retry(url)
            if data is None or not isinstance(data, list) or len(data) == 0:
                return None
            raw = data[0]
            try:
                last = float(raw.get("last", 0))
                funding = float(raw.get("funding_rate", 0))
                funding_indicative = float(raw.get("funding_rate_indicative", 0))
                # 优先使用 indicative funding rate
                fr = funding_indicative if funding_indicative != 0 else funding
                return {
                    "last": last,
                    "volume": float(raw.get("volume_24h_base", 0)),
                    "quote_volume": float(raw.get("volume_24h_settle", 0)),
                    "high_24h": float(raw.get("high_24h", 0)),
                    "low_24h": float(raw.get("low_24h", 0)),
                    "change_percentage": float(raw.get("change_percentage", 0)),
                    "funding_rate": fr,
                    "mark_price": last,  # futures ticker 不含 mark_price, 用 last 近似
                    "timestamp": int(time.time()),
                }
            except (ValueError, TypeError) as e:
                logger.warning("解析 futures ticker 失败: %s", e)
                return None

        else:
            logger.warning("未知 market: %s", market)
            return None

    def get_order_book(self, market: str, symbol: str,
                       limit: int = 20) -> Optional[Dict[str, Any]]:
        """获取订单簿数据

        Args:
            market: "spot" 或 "futures"
            symbol: 交易对
            limit: 深度层数 (默认 20)

        Returns:
            统一格式的订单簿字典:
            {
                "asks": [[price, size], ...],  # 卖单 (升序)
                "bids": [[price, size], ...],  # 买单 (降序)
                "mid_price": float,
                "spread": float,
                "spread_pct": float,
                "imbalance": float,  # (bid_depth - ask_depth) / (bid_depth + ask_depth)
                "timestamp": int,
            }
            失败返回 None
        """
        gate_sym = self.convert_symbol(symbol)

        if market == "spot":
            url = (f"{self.BASE_SPOT}/order_book?currency_pair={gate_sym}"
                   f"&limit={limit}")
        elif market == "futures":
            url = (f"{self.BASE_FUTURES}/order_book?settle=usdt"
                   f"&contract={gate_sym}&limit={limit}")
        else:
            logger.warning("未知 market: %s", market)
            return None

        data = self._http_get_with_retry(url)
        if data is None:
            return None

        try:
            if market == "spot":
                # Spot: {"asks":[["price","size"],...], "bids":[...]}
                asks_raw = data.get("asks", [])
                bids_raw = data.get("bids", [])
                asks = [[float(a[0]), float(a[1])] for a in asks_raw if len(a) >= 2]
                bids = [[float(b[0]), float(b[1])] for b in bids_raw if len(b) >= 2]
            else:
                # Futures: {"asks":[{"p":"price","s":size},...], "bids":[...]}
                asks_raw = data.get("asks", [])
                bids_raw = data.get("bids", [])
                asks = [[float(a.get("p", 0)), float(a.get("s", 0))] for a in asks_raw]
                bids = [[float(b.get("p", 0)), float(b.get("s", 0))] for b in bids_raw]
        except (ValueError, TypeError, IndexError) as e:
            logger.warning("解析 order_book 失败: %s", e)
            return None

        if not asks or not bids:
            return None

        # asks 升序, bids 降序
        asks.sort(key=lambda x: x[0])
        bids.sort(key=lambda x: -x[0])

        best_ask = asks[0][0]
        best_bid = bids[0][0]
        mid_price = (best_ask + best_bid) / 2.0
        spread = best_ask - best_bid
        spread_pct = spread / mid_price if mid_price > 0 else 0.0

        bid_depth = sum(q for _, q in bids)
        ask_depth = sum(q for _, q in asks)
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

        return {
            "asks": asks,
            "bids": bids,
            "mid_price": mid_price,
            "spread": spread,
            "spread_pct": spread_pct,
            "imbalance": imbalance,
            "timestamp": int(time.time()),
        }

    def get_trades(self, market: str, symbol: str,
                   limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """获取最近成交记录

        Args:
            market: "spot" 或 "futures"
            symbol: 交易对
            limit: 返回条数 (默认 100)

        Returns:
            统一格式的成交列表:
            [
                {
                    "price": float,
                    "size": float,       # 正=买, 负=卖 (futures); spot 始终正
                    "side": "buy"/"sell",
                    "timestamp": int,
                },
                ...
            ]
            失败返回 None
        """
        gate_sym = self.convert_symbol(symbol)

        if market == "spot":
            url = (f"{self.BASE_SPOT}/trades?currency_pair={gate_sym}"
                   f"&limit={limit}")
        elif market == "futures":
            url = (f"{self.BASE_FUTURES}/trades?settle=usdt"
                   f"&contract={gate_sym}&limit={limit}")
        else:
            logger.warning("未知 market: %s", market)
            return None

        data = self._http_get_with_retry(url)
        if data is None or not isinstance(data, list):
            return None

        trades = []
        try:
            for t in data:
                if market == "spot":
                    # Spot: {"id":"...","create_time":"...","price":"...","amount":"...","side":"buy"}
                    price = float(t.get("price", 0))
                    size = float(t.get("amount", 0))
                    side = t.get("side", "buy")
                else:
                    # Futures: {"size":28,"price":"64174.1","create_time_ms":"..."}
                    # size > 0 = buy, size < 0 = sell
                    price = float(t.get("price", 0))
                    size = float(t.get("size", 0))
                    side = "buy" if size >= 0 else "sell"
                    size = abs(size)

                trades.append({
                    "price": price,
                    "size": size,
                    "side": side,
                    "timestamp": int(time.time()),
                })
        except (ValueError, TypeError) as e:
            logger.warning("解析 trades 失败: %s", e)
            return None

        return trades if trades else None

    def get_taker_volume_ratio(self, symbol: str,
                               limit: int = 100) -> Optional[Dict[str, Any]]:
        """获取主动买卖量比 (基于 futures trades 推算)

        通过统计最近 N 笔期货成交的买卖方向，计算 taker buy/sell volume ratio。

        Args:
            symbol: 交易对
            limit: 统计的成交笔数 (默认 100)

        Returns:
            {
                "buy_volume": float,    # 主动买入量
                "sell_volume": float,   # 主动卖出量
                "total_volume": float,  # 总成交量
                "buy_sell_ratio": float,  # 买卖比
                "institution_ratio": float,  # 买入占比 = buy / total
                "timestamp": int,
            }
            失败返回 None
        """
        trades = self.get_trades("futures", symbol, limit)
        if trades is None or len(trades) == 0:
            return None

        buy_vol = sum(t["size"] for t in trades if t["side"] == "buy")
        sell_vol = sum(t["size"] for t in trades if t["side"] == "sell")
        total = buy_vol + sell_vol

        if total <= 0:
            return None

        return {
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "total_volume": total,
            "buy_sell_ratio": buy_vol / sell_vol if sell_vol > 0 else float("inf"),
            "institution_ratio": buy_vol / total,
            "timestamp": int(time.time()),
        }

    def get_funding_rate(self, symbol: str) -> float:
        """获取资金费率 (futures)

        Args:
            symbol: 交易对

        Returns:
            资金费率 (float), 失败返回 0.0
        """
        gate_sym = self.convert_symbol(symbol)
        url = f"{self.BASE_FUTURES}/contracts/{gate_sym}"
        data = self._http_get_with_retry(url)
        if data is None:
            return 0.0
        try:
            return float(data.get("funding_rate", 0))
        except (ValueError, TypeError):
            return 0.0

    def get_mark_price(self, symbol: str) -> float:
        """获取标记价格 (futures)

        Args:
            symbol: 交易对

        Returns:
            标记价格 (float), 失败返回 0.0
        """
        gate_sym = self.convert_symbol(symbol)
        url = f"{self.BASE_FUTURES}/contracts/{gate_sym}"
        data = self._http_get_with_retry(url)
        if data is None:
            return 0.0
        try:
            return float(data.get("mark_price", 0))
        except (ValueError, TypeError):
            return 0.0

    def get_contract_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取合约信息 (含 funding_rate, mark_price, index_price 等)

        一次请求获取合约的完整信息，避免重复调用。

        Returns:
            {
                "funding_rate": float,
                "mark_price": float,
                "index_price": float,
                "last_price": float,
                "quanto_base_value": float,
                "config": dict,  # 原始合约配置
            }
            失败返回 None
        """
        gate_sym = self.convert_symbol(symbol)
        url = f"{self.BASE_FUTURES}/contracts/{gate_sym}"
        data = self._http_get_with_retry(url)
        if data is None:
            return None
        try:
            return {
                "funding_rate": float(data.get("funding_rate", 0)),
                "mark_price": float(data.get("mark_price", 0)),
                "index_price": float(data.get("index_price", 0)),
                "last_price": float(data.get("last_price", 0)),
                "quanto_base_value": float(data.get("quanto_base_value", 0)),
                "config": data,
            }
        except (ValueError, TypeError) as e:
            logger.warning("解析 contract info 失败: %s", e)
            return None


# ==================== 模块级便捷函数 ====================

_default_instance: Optional[GateIODataSource] = None


def _get_default() -> GateIODataSource:
    """获取默认单例实例"""
    global _default_instance
    if _default_instance is None:
        _default_instance = GateIODataSource()
    return _default_instance


def get_ticker(market: str, symbol: str) -> Optional[Dict[str, Any]]:
    """便捷函数: 获取 ticker"""
    return _get_default().get_ticker(market, symbol)


def get_order_book(market: str, symbol: str,
                   limit: int = 20) -> Optional[Dict[str, Any]]:
    """便捷函数: 获取订单簿"""
    return _get_default().get_order_book(market, symbol, limit)


def get_trades(market: str, symbol: str,
               limit: int = 100) -> Optional[List[Dict[str, Any]]]:
    """便捷函数: 获取成交记录"""
    return _get_default().get_trades(market, symbol, limit)


def get_taker_volume_ratio(symbol: str,
                           limit: int = 100) -> Optional[Dict[str, Any]]:
    """便捷函数: 获取主动买卖量比"""
    return _get_default().get_taker_volume_ratio(symbol, limit)


def get_funding_rate(symbol: str) -> float:
    """便捷函数: 获取资金费率"""
    return _get_default().get_funding_rate(symbol)


def get_mark_price(symbol: str) -> float:
    """便捷函数: 获取标记价格"""
    return _get_default().get_mark_price(symbol)


def convert_symbol(symbol: str) -> str:
    """便捷函数: symbol 转换"""
    return GateIODataSource.convert_symbol(symbol)


def file_symbol(symbol: str) -> str:
    """便捷函数: 文件名 symbol"""
    return GateIODataSource._file_symbol(symbol)
