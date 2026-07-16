#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hermes 交易系统 - 24/7 自动数据抓取服务
======================================

本服务 24/7 运行，自动从多个外部 API 抓取市场数据，
存储到 data/auto_fetched/ 目录供各 Gate 读取（避免 Gate 自身发起 API 调用）。

数据源与调度:
  1. Fear & Greed Index          - 每 1 小时  - fear_greed/latest.json
  2. Gate.io 期货多空比 (LSR)     - 每 5 分钟  - long_short_ratio/<SYM>_latest.json
  3. Gate.io 期货主动买卖量        - 每 5 分钟  - institution_retail/<SYM>_latest.json
  4. Gate.io Ticker (价格数据)   - 每 5 分钟  - ticker/<SYM>_latest.json
  5. Gate.io 订单簿               - 每 30 秒   - order_book/<SYM>_latest.json
  6. Blockchain UTXO 实现价格      - 每 1 天   - urpd/latest.json
  7. 强平数据 (降级: 无 WebSocket)  - 定期轮询  - liquidations/latest.json

特性:
  - 多线程: 每个数据源独立线程，各自调度
  - 重试机制: 3 次指数退避 (1s, 2s, 4s)
  - 优雅降级: API 失败时保留上次缓存文件，不崩溃
  - 健康检查: 每个数据源上报 healthy/degraded/down 状态
  - 自动建目录
  - 日志: 同时输出到控制台和 fetcher.log
  - 健康状态文件: 每 60 秒写入 health.json
  - 支持守护进程 (nohup) 或前台运行

用法:
  python3 auto_data_fetcher_service.py --daemon --symbols BTCUSDT,ETHUSDT
  python3 auto_data_fetcher_service.py --once --symbols BTCUSDT
  nohup python3 auto_data_fetcher_service.py --daemon &

仅依赖 Python 标准库，不导入交易系统任何模块（避免循环引用）。
"""

import argparse
import json
import logging
import os
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# ==================== 常量定义 ====================

DEFAULT_SYMBOLS = ["BTCUSDT"]
DEFAULT_DATA_DIR = "data/auto_fetched"

# API 端点
API_FEAR_GREED = "https://api.alternative.me/fng/?limit=1"
API_GATEIO_SPOT = "https://api.gateio.ws/api/v4/spot"
API_GATEIO_FUTURES = "https://api.gateio.ws/api/v4/futures/usdt"
API_BLOCKCHAIN_TOTALBC = "https://blockchain.info/q/totalbc"
# BGeometrics 公开 REST API (无需 API Key, 免费 10次/小时, 足够每 1 小时 1 次)
# 提供: realized_cap (capRealUSDs), MVRV (mvrvs), URPD (urpds), STH/LTH realized_price
API_BGEOMETRICS_BASE = "https://api.bgeometrics.com"
API_BGEOMETRICS_CAP_REAL_USD = API_BGEOMETRICS_BASE + "/capRealUSDs?page=0&size=1&sort=date,desc"
API_BGEOMETRICS_MVRV = API_BGEOMETRICS_BASE + "/mvrvs?page=0&size=1&sort=date,desc"
API_BGEOMETRICS_BTC_PRICE = API_BGEOMETRICS_BASE + "/btcPrices?page=0&size=1&sort=date,desc"
# Binance WebSocket 不可用 (网络限制), 强平数据降级为定期轮询 Gate.io trades

# 重试配置
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # 指数退避秒数

# 请求超时 (秒)
REQUEST_TIMEOUT = 15

# 健康文件写入间隔 (秒)
HEALTH_WRITE_INTERVAL = 60

# 强平数据最大保留条数
LIQ_MAX_EVENTS = 200


# ==================== 工具函数 ====================

def utc_now_iso():
    """返回当前 UTC 时间的 ISO8601 字符串"""
    return datetime.now(timezone.utc).isoformat()


def utc_now_ts():
    """返回当前 Unix 时间戳 (整数)"""
    return int(time.time())


def http_get_json(url, timeout=REQUEST_TIMEOUT):
    """发起 GET 请求并返回解析后的 JSON。
    成功返回 Python 对象，失败抛异常。"""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "HermesAutoFetcher/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def http_get_with_retry(url, retries=MAX_RETRIES):
    """带指数退避重试的 HTTP GET JSON 请求。
    成功返回数据，最终失败返回 None。"""
    last_err = None
    for attempt in range(retries):
        try:
            return http_get_json(url)
        except (urllib.error.URLError, urllib.error.HTTPError,
                socket.timeout, json.JSONDecodeError, OSError,
                ssl.SSLError) as e:
            last_err = e
            if attempt < retries - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                logging.debug(
                    "请求失败 (尝试 %d/%d), %ds 后重试: %s - %s",
                    attempt + 1, retries, wait, url, e,
                )
                time.sleep(wait)
    logging.warning("请求最终失败 (%d 次): %s - %s", retries, url, last_err)
    return None


def ensure_dir(path):
    """确保目录存在，不存在则递归创建"""
    os.makedirs(path, exist_ok=True)


def save_json_file(path, data, source):
    """保存数据到 JSON 文件，包含标准元信息。
    使用临时文件 + 原子替换，避免半写文件被读取。
    返回 True/False 表示成功与否。"""
    try:
        ensure_dir(os.path.dirname(path))
        payload = {
            "data": data,
            "timestamp": utc_now_ts(),
            "source": source,
            "fetch_time": utc_now_iso(),
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)  # 原子替换
        return True
    except Exception as e:
        logging.error("保存文件失败 %s: %s", path, e)
        return False


def gateio_symbol(symbol):
    """将交易系统 symbol 转换为 Gate.io 格式

    "BTCUSDT" → "BTC_USDT"
    "BTC-USDT" → "BTC_USDT"
    "BTC" → "BTC_USDT"
    """
    if not symbol:
        return "BTC_USDT"
    s = symbol.upper().strip()
    s = s.replace("-", "_").replace("/", "_")
    if "_" not in s and s.endswith("USDT"):
        s = s[:-4] + "_USDT"
    if not s.endswith("_USDT"):
        base = s.split("_")[0]
        s = base + "_USDT"
    return s


# ==================== 强平数据线程 (Gate.io 轮询) ====================

class LiquidationThread(threading.Thread):
    """通过轮询 Gate.io futures trades 近似获取大额成交 (潜在强平)。

    Binance WebSocket 不可用 (网络限制)，降级为定期轮询 Gate.io futures trades，
    筛选大额成交作为潜在强平事件。
    """

    # 大额成交阈值 (合约数量 > 此值视为潜在强平)
    LARGE_TRADE_THRESHOLD = 50.0

    def __init__(self, output_path, data_dir, symbols=None):
        super().__init__(daemon=True)
        self.name = "liquidation_poll"
        self.output_path = output_path
        self.data_dir = data_dir
        self.symbols = symbols or ["BTCUSDT"]
        self._stop_event = threading.Event()
        self._events = []
        self._lock = threading.Lock()
        self._last_success = None
        self._last_failure = None
        self._last_error = None
        self._success_count = 0
        self._failure_count = 0
        self._status = "starting"

    def stop(self):
        self._stop_event.set()

    def run(self):
        logging.info("[%s] 强平轮询线程启动, 间隔 30s", self.name)
        while not self._stop_event.is_set():
            try:
                events = self._fetch_liquidation_events()
                with self._lock:
                    self._events = events[-LIQ_MAX_EVENTS:]
                ok = save_json_file(
                    self.output_path, events, "gateio_futures_large_trades"
                )
                if ok:
                    self._last_success = utc_now_ts()
                    self._success_count += 1
                    self._status = "healthy"
                else:
                    self._last_failure = utc_now_ts()
                    self._failure_count += 1
                    self._status = "degraded"
            except Exception as e:
                self._last_failure = utc_now_ts()
                self._last_error = str(e)
                self._failure_count += 1
                self._status = "degraded"
                logging.error("[%s] 轮询异常: %s", self.name, e)
            self._stop_event.wait(30)
        logging.info("[%s] 强平轮询线程已停止", self.name)

    def _fetch_liquidation_events(self):
        """从 Gate.io futures trades 获取大额成交"""
        events = []
        for sym in self.symbols:
            gate_sym = gateio_symbol(sym)
            url = ("%s/trades?settle=usdt&contract=%s&limit=100"
                   % (API_GATEIO_FUTURES, gate_sym))
            data = http_get_with_retry(url)
            if data is None or not isinstance(data, list):
                continue
            for t in data:
                try:
                    size = abs(float(t.get("size", 0)))
                    if size >= self.LARGE_TRADE_THRESHOLD:
                        price = float(t.get("price", 0))
                        side = "sell" if float(t.get("size", 0)) < 0 else "buy"
                        events.append({
                            "symbol": gate_sym,
                            "price": str(price),
                            "qty": str(size),
                            "side": side,
                            "time": t.get("create_time_ms", str(utc_now_ts() * 1000)),
                            "source": "gateio_large_trade",
                        })
                except (ValueError, TypeError):
                    continue
        return events

    def get_status(self):
        return {
            "name": self.name,
            "source": "gateio_futures_large_trades",
            "status": self._status,
            "interval_sec": 30,
            "output_path": self.output_path,
            "last_success": self._last_success,
            "last_failure": self._last_failure,
            "last_error": self._last_error,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "events_buffered": len(self._events),
        }


# ==================== 抓取线程 ====================

class FetcherThread(threading.Thread):
    """通用轮询抓取线程。
    周期性调用 fetch_func()，成功则由 fetch_func 自行保存数据。"""

    def __init__(self, name, fetch_func, interval, output_path, source_name):
        super().__init__(daemon=True)
        self.name = name
        self.fetch_func = fetch_func        # callable -> bool (成功与否)
        self.interval = interval            # 轮询间隔 (秒)
        self.output_path = output_path
        self.source_name = source_name
        self._stop_event = threading.Event()
        self._last_success = None           # 上次成功时间戳
        self._last_failure = None           # 上次失败时间戳
        self._last_error = None             # 上次错误信息
        self._success_count = 0
        self._failure_count = 0
        self._status = "starting"           # starting/healthy/degraded/down

    def stop(self):
        self._stop_event.set()

    def run(self):
        logging.info("[%s] 抓取线程启动, 间隔 %ds", self.name, self.interval)
        while not self._stop_event.is_set():
            try:
                ok = self.fetch_func()
                if ok:
                    self._last_success = utc_now_ts()
                    self._success_count += 1
                    self._status = "healthy"
                    self._last_error = None
                else:
                    self._last_failure = utc_now_ts()
                    self._failure_count += 1
                    self._update_degraded_status()
            except Exception as e:
                self._last_failure = utc_now_ts()
                self._last_error = str(e)
                self._failure_count += 1
                self._update_degraded_status()
                logging.error("[%s] 线程异常: %s", self.name, e)
            # 等待间隔 (可被 stop 中断)
            self._stop_event.wait(self.interval)
        logging.info("[%s] 抓取线程已停止", self.name)

    def _update_degraded_status(self):
        """根据上次成功时间判断 degraded / down"""
        if self._last_success:
            age = utc_now_ts() - self._last_success
            if age < self.interval * 3:
                self._status = "degraded"   # 近期成功过，暂时降级
            else:
                self._status = "down"       # 长时间未成功
        else:
            self._status = "down"           # 从未成功

    def get_status(self):
        return {
            "name": self.name,
            "source": self.source_name,
            "status": self._status,
            "interval_sec": self.interval,
            "output_path": self.output_path,
            "last_success": self._last_success,
            "last_failure": self._last_failure,
            "last_error": self._last_error,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
        }


# ==================== 主服务 ====================

class AutoDataFetcherService:
    """自动数据抓取服务主类"""

    def __init__(self, symbols=None, data_dir=DEFAULT_DATA_DIR):
        self.symbols = symbols or list(DEFAULT_SYMBOLS)
        self.data_dir = data_dir
        self.threads = []          # 所有 FetcherThread
        self.liq_thread = None     # 强平线程
        self._health_thread = None
        self._stop_event = threading.Event()
        ensure_dir(self.data_dir)

    # ---------- 路径辅助 ----------

    def _path(self, *parts):
        return os.path.join(self.data_dir, *parts)

    # ---------- 各数据源抓取函数 (闭包) ----------

    def _make_fear_greed_fetcher(self):
        out = self._path("fear_greed", "latest.json")

        def fetch():
            data = http_get_with_retry(API_FEAR_GREED)
            if data is None:
                return False
            return save_json_file(out, data, "alternative_me_fng")
        return fetch, out, 3600, "fear_greed", "alternative_me_fng"

    def _make_lsr_fetcher(self, symbol):
        """Gate.io: 基于期货成交 + 合约信息推算多空比

        通过统计主动买卖量 + 资金费率推算多空持仓比:
          - funding_rate > 0 → 多头拥挤
          - buy_volume > sell_volume → 多头主导
        """
        gate_sym = gateio_symbol(symbol)
        out = self._path("long_short_ratio", "%s_latest.json" % symbol)

        def fetch():
            # 1. 获取合约信息 (funding_rate, mark_price)
            contract_url = "%s/contracts/%s" % (API_GATEIO_FUTURES, gate_sym)
            contract = http_get_with_retry(contract_url)
            if contract is None:
                return False

            try:
                funding_rate = float(contract.get("funding_rate", 0))
                mark_price = float(contract.get("mark_price", 0))
            except (ValueError, TypeError):
                return False

            # 2. 获取最近成交推算买卖比
            trades_url = ("%s/trades?settle=usdt&contract=%s&limit=100"
                          % (API_GATEIO_FUTURES, gate_sym))
            trades = http_get_with_retry(trades_url)

            buy_vol = 0.0
            sell_vol = 0.0
            if trades and isinstance(trades, list):
                for t in trades:
                    try:
                        size = float(t.get("size", 0))
                        if size > 0:
                            buy_vol += size
                        else:
                            sell_vol += abs(size)
                    except (ValueError, TypeError):
                        continue

            total = buy_vol + sell_vol
            institution_ratio = buy_vol / total if total > 0 else 0.5

            # 3. 推算多空比:
            #    funding_rate > 0 → 多头支付 → 多头拥挤 → long_ratio > 0.5
            #    买入占比高 → 多头主导 → long_ratio 上升
            funding_offset = funding_rate * 20.0  # 资金费率敏感度
            buy_offset = (institution_ratio - 0.5) * 0.4  # 买卖量影响
            long_ratio = 0.5 + funding_offset + buy_offset
            long_ratio = max(0.05, min(0.95, long_ratio))
            short_ratio = 1.0 - long_ratio

            payload = [{
                "symbol": symbol,
                "long_ratio": round(long_ratio, 4),
                "short_ratio": round(short_ratio, 4),
                "long_short_ratio": round(long_ratio / max(0.001, short_ratio), 4),
                "is_long_crowded": long_ratio > 0.7,
                "is_short_crowded": long_ratio < 0.3,
                "funding_rate": funding_rate,
                "mark_price": mark_price,
                "buy_volume": round(buy_vol, 2),
                "sell_volume": round(sell_vol, 2),
                "institution_ratio": round(institution_ratio, 4),
                "crowding_score": round((long_ratio - 0.5) * 2.0, 4),
                "timestamp": str(utc_now_ts() * 1000),
            }]
            return save_json_file(out, payload, "gateio_futures_lsr")
        return fetch, out, 300, "lsr_%s" % symbol, "gateio_futures_lsr"

    def _make_taker_fetcher(self, symbol):
        """Gate.io: 基于期货成交统计主动买卖量"""
        gate_sym = gateio_symbol(symbol)
        out = self._path("institution_retail", "%s_latest.json" % symbol)

        def fetch():
            url = ("%s/trades?settle=usdt&contract=%s&limit=100"
                   % (API_GATEIO_FUTURES, gate_sym))
            trades = http_get_with_retry(url)
            if trades is None or not isinstance(trades, list):
                return False

            buy_vol = 0.0
            sell_vol = 0.0
            for t in trades:
                try:
                    size = float(t.get("size", 0))
                    if size > 0:
                        buy_vol += size
                    else:
                        sell_vol += abs(size)
                except (ValueError, TypeError):
                    continue

            total = buy_vol + sell_vol
            if total <= 0:
                return False

            ratio = buy_vol / sell_vol if sell_vol > 0 else 999.0
            institution_ratio = buy_vol / total

            payload = {
                "buySellRatio": "%.4f" % ratio,
                "buyVol": "%.2f" % buy_vol,
                "sellVol": "%.2f" % sell_vol,
                "totalVol": "%.2f" % total,
                "institution_ratio": round(institution_ratio, 4),
                "timestamp": str(utc_now_ts() * 1000),
            }
            return save_json_file(out, payload, "gateio_futures_taker")
        return fetch, out, 300, "taker_%s" % symbol, "gateio_futures_taker"

    def _make_ticker_fetcher(self, symbol):
        """Gate.io: 现货 ticker (价格/成交量/24h高低)"""
        gate_sym = gateio_symbol(symbol)
        out = self._path("ticker", "%s_latest.json" % symbol)

        def fetch():
            url = "%s/tickers?currency_pair=%s" % (API_GATEIO_SPOT, gate_sym)
            data = http_get_with_retry(url)
            if data is None or not isinstance(data, list) or len(data) == 0:
                return False
            raw = data[0]
            try:
                ticker = {
                    "last": float(raw.get("last", 0)),
                    "volume": float(raw.get("base_volume", 0)),
                    "quote_volume": float(raw.get("quote_volume", 0)),
                    "high_24h": float(raw.get("high_24h", 0)),
                    "low_24h": float(raw.get("low_24h", 0)),
                    "change_percentage": float(raw.get("change_percentage", 0)),
                    "symbol": symbol,
                }
                return save_json_file(out, ticker, "gateio_spot_ticker")
            except (ValueError, TypeError) as e:
                logging.warning("解析 Gate.io spot ticker 失败: %s", e)
                return False
        return fetch, out, 300, "ticker_%s" % symbol, "gateio_spot_ticker"

    def _make_orderbook_fetcher(self, symbol):
        """Gate.io: 期货订单簿"""
        gate_sym = gateio_symbol(symbol)
        out = self._path("order_book", "%s_latest.json" % symbol)

        def fetch():
            url = ("%s/order_book?settle=usdt&contract=%s&limit=20"
                   % (API_GATEIO_FUTURES, gate_sym))
            data = http_get_with_retry(url)
            if data is None:
                return False

            try:
                asks_raw = data.get("asks", [])
                bids_raw = data.get("bids", [])
                asks = [[float(a.get("p", 0)), float(a.get("s", 0))]
                        for a in asks_raw]
                bids = [[float(b.get("p", 0)), float(b.get("s", 0))]
                        for b in bids_raw]
            except (ValueError, TypeError, AttributeError) as e:
                logging.warning("解析 Gate.io order_book 失败: %s", e)
                return False

            if not asks or not bids:
                return False

            asks.sort(key=lambda x: x[0])
            bids.sort(key=lambda x: -x[0])

            best_ask = asks[0][0]
            best_bid = bids[0][0]
            mid_price = (best_ask + best_bid) / 2.0
            spread = best_ask - best_bid
            spread_pct = spread / mid_price if mid_price > 0 else 0.0

            bid_depth = sum(q for _, q in bids)
            ask_depth = sum(q for _, q in asks)
            total_depth = bid_depth + ask_depth
            imbalance = ((bid_depth - ask_depth) / total_depth
                         if total_depth > 0 else 0.0)

            payload = {
                "asks": asks,
                "bids": bids,
                "mid_price": mid_price,
                "spread": spread,
                "spread_pct": spread_pct,
                "imbalance": imbalance,
                "symbol": symbol,
            }
            return save_json_file(out, payload, "gateio_futures_depth")
        return fetch, out, 30, "orderbook_%s" % symbol, "gateio_futures_depth"

    def _make_smart_money_fetcher(self, symbol):
        """Gate.io: smart money / whale large trades (over $100K USD)

        Filters futures trades for large transactions and maps them to
        whale transaction format.  Whale trades do not need high-frequency
        refresh, 60 second interval is sufficient.
        """
        gate_sym = gateio_symbol(symbol)
        out = self._path("smart_money", "%s_latest.json" % symbol)

        def fetch():
            amp = chr(38)  # URL query separator
            url = ("%s/trades?settle=usdt" + amp + "contract=%s" + amp + "limit=200") % (API_GATEIO_FUTURES, gate_sym)
            trades = http_get_with_retry(url)
            if trades is None or not isinstance(trades, list):
                return False

            token = symbol.upper().replace("-", "").replace("/", "")
            if token.endswith("USDT"):
                token = token[:-4]

            transactions = []
            for t in trades:
                try:
                    size = float(t.get("size", 0))
                    price = float(t.get("price", 0))
                    create_time_ms = int(t.get("create_time_ms", 0))
                    trade_id = str(t.get("id", ""))
                    amount_usd = abs(size * price)
                    if amount_usd <= 100000:
                        continue
                    direction = "from_exchange" if size > 0 else "to_exchange"
                    tx = {
                        "timestamp": create_time_ms,
                        "wallet_address": "gateio_whale_%s" % trade_id[:8],
                        "token": token,
                        "amount_usd": amount_usd,
                        "direction": direction,
                        "exchange": "gateio_futures",
                    }
                    transactions.append(tx)
                except (ValueError, TypeError):
                    continue

            payload = {
                "transactions": transactions,
                "symbol": symbol,
                "count": len(transactions),
            }
            return save_json_file(out, payload, "gateio_futures_large_trades")
        return fetch, out, 60, "smart_money_%s" % symbol, "gateio_futures_large_trades"

    def _make_urpd_fetcher(self):
        """URPD/Realized Price 抓取器 — 使用 BGeometrics 公开 API + blockchain.info

        数据源:
          - BGeometrics capRealUSDs: 真实 realized_cap (已实现市值, USD)
          - BGeometrics mvrvs: 真实 MVRV 比率
          - blockchain.info totalbc: BTC 总供应量 (satoshis)

        计算:
          realized_price = realized_cap / total_btc

        限制: BGeometrics 免费 10次/小时, 本抓取器间隔 1 小时, 远低于限额。
        失败处理: 任一关键数据源失败则返回 False, 保留上次缓存。
        """
        out = self._path("urpd", "latest.json")

        def fetch():
            # 1. 获取 realized_cap (BGeometrics)
            cap_data = http_get_with_retry(API_BGEOMETRICS_CAP_REAL_USD)
            realized_cap = None
            cap_unix_ts = None
            if cap_data and isinstance(cap_data, dict):
                embedded = cap_data.get("_embedded", {})
                for key, items in embedded.items():
                    if isinstance(items, list) and items:
                        first = items[0]
                        try:
                            realized_cap = float(first.get("capRealUSD", 0))
                            cap_unix_ts = int(first.get("unixTs", 0))
                        except (ValueError, TypeError):
                            pass
                        break
            if realized_cap is None or realized_cap <= 0:
                logging.warning("URPD: BGeometrics capRealUSDs 获取失败, 跳过本次")
                return False

            # 2. 获取 BTC 总供应量 (blockchain.info)
            totalbc_raw = http_get_with_retry(API_BLOCKCHAIN_TOTALBC)
            total_btc = None
            if totalbc_raw is not None:
                try:
                    # blockchain.info /q/totalbc 返回 satoshis 字符串
                    total_btc = float(totalbc_raw) / 1e8
                except (ValueError, TypeError):
                    total_btc = None
            if total_btc is None or total_btc <= 0:
                logging.warning("URPD: blockchain.info totalbc 获取失败, 跳过本次")
                return False

            # 3. 获取 MVRV (BGeometrics, 备用)
            mvrv_data = http_get_with_retry(API_BGEOMETRICS_MVRV)
            mvrv_ratio = None
            mvrv_unix_ts = None
            if mvrv_data and isinstance(mvrv_data, dict):
                embedded = mvrv_data.get("_embedded", {})
                for key, items in embedded.items():
                    if isinstance(items, list) and items:
                        # mvrvs 默认按 unixTs 升序, 取最后一条(最新)
                        latest = items[-1] if len(items) > 1 else items[0]
                        try:
                            mvrv_ratio = float(latest.get("mvrv", 0))
                            mvrv_unix_ts = int(latest.get("unixTs", 0))
                        except (ValueError, TypeError):
                            pass
                        break

            # 4. 计算 realized_price
            realized_price = realized_cap / total_btc

            # 5. 构造完整 payload (URPDGate 直接读取 data 字段)
            payload = {
                "realized_price": realized_price,
                "realized_cap_usd": realized_cap,
                "total_btc": total_btc,
                "mvrv_ratio": mvrv_ratio,
                "cap_unix_ts": cap_unix_ts,
                "mvrv_unix_ts": mvrv_unix_ts,
                "raw_totalbc": totalbc_raw,
            }
            source = "bgeometrics_caprealusd+blockchain_totalbc"
            success = save_json_file(out, payload, source)
            if success:
                logging.info(
                    "URPD 抓取成功: realized_price=$%.2f realized_cap=$%.2fB "
                    "total_btc=%.0f mvrv=%.4f",
                    realized_price, realized_cap / 1e9, total_btc,
                    mvrv_ratio if mvrv_ratio else 0.0,
                )
            return success
        # 间隔 1 小时 (3600s), BGeometrics 免费限额 10次/小时, 余量充足
        return fetch, out, 3600, "urpd", "bgeometrics_caprealusd+blockchain_totalbc"

    # ---------- 注册所有抓取器 ----------

    def _register_all(self):
        """创建并注册所有抓取线程 (不含强平轮询)"""
        creators = [self._make_fear_greed_fetcher(), self._make_urpd_fetcher()]
        for sym in self.symbols:
            creators.append(self._make_lsr_fetcher(sym))
            creators.append(self._make_taker_fetcher(sym))
            creators.append(self._make_ticker_fetcher(sym))
            creators.append(self._make_orderbook_fetcher(sym))
            creators.append(self._make_smart_money_fetcher(sym))  # NEW: real whale trades

        for fetch_func, out, interval, name, source in creators:
            t = FetcherThread(name, fetch_func, interval, out, source)
            self.threads.append(t)

    # ---------- 启动 / 停止 ----------

    def start(self):
        """启动所有抓取线程 (守护模式)"""
        self._register_all()
        for t in self.threads:
            t.start()
        # 强平轮询线程 (Gate.io)
        liq_out = self._path("liquidations", "latest.json")
        self.liq_thread = LiquidationThread(liq_out, self.data_dir, self.symbols)
        self.liq_thread.start()
        # 闭环任务1: 启动后立即落盘健康快照, 避免短生命周期运行(<60秒)无可观测状态
        self._write_health()
        # 健康检查写文件线程
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()
        logging.info("服务已启动: %d 个抓取线程 + 强平轮询 + 健康检查",
                     len(self.threads))

    def stop(self):
        """优雅关闭所有线程"""
        logging.info("正在停止服务...")
        self._stop_event.set()
        for t in self.threads:
            t.stop()
        if self.liq_thread:
            self.liq_thread.stop()
        for t in self.threads:
            t.join(timeout=5)
        if self.liq_thread:
            self.liq_thread.join(timeout=5)
        # 闭环任务1: 停止前写最终健康快照, 保留本次服务的成功/失败计数
        self._write_health()
        logging.info("服务已停止")

    # ---------- 单次抓取 ----------

    def fetch_once(self):
        """一次性抓取所有 HTTP 数据源 (不含 WebSocket)。
        逐个同步执行，适合 --once 模式。"""
        self._register_all()
        results = {}
        for t in self.threads:
            logging.info("[--once] 抓取 %s ...", t.name)
            try:
                ok = t.fetch_func()
                if ok:
                    t._last_success = utc_now_ts()
                    t._success_count += 1
                    t._status = "healthy"
                else:
                    t._last_failure = utc_now_ts()
                    t._failure_count += 1
                    t._status = "down"
                results[t.name] = ok
            except Exception as e:
                t._last_failure = utc_now_ts()
                t._last_error = str(e)
                t._failure_count += 1
                t._status = "down"
                results[t.name] = False
                logging.error("[--once] %s 异常: %s", t.name, e)
        # 写入健康文件
        self._write_health()
        return results

    # ---------- 健康状态 ----------

    def get_health(self):
        """返回所有数据源的健康状态字典"""
        statuses = []
        for t in self.threads:
            statuses.append(t.get_status())
        if self.liq_thread:
            statuses.append(self.liq_thread.get_status())
        return {
            "service": "auto_data_fetcher",
            "timestamp": utc_now_ts(),
            "fetch_time": utc_now_iso(),
            "symbols": self.symbols,
            "sources": statuses,
            "summary": self._health_summary(statuses),
        }

    @staticmethod
    def _health_summary(statuses):
        total = len(statuses)
        healthy = sum(1 for s in statuses if s["status"] == "healthy")
        degraded = sum(1 for s in statuses if s["status"] == "degraded")
        down = sum(1 for s in statuses if s["status"] == "down")
        starting = sum(1 for s in statuses if s["status"] == "starting")
        if down == total and total > 0:
            overall = "down"
        elif down > 0 or degraded > 0:
            overall = "degraded"
        else:
            overall = "healthy"
        return {
            "overall": overall,
            "total": total,
            "healthy": healthy,
            "degraded": degraded,
            "down": down,
            "starting": starting,
        }

    def _write_health(self):
        """将健康状态写入 health.json"""
        health = self.get_health()
        path = self._path("health.json")
        try:
            ensure_dir(self.data_dir)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(health, f, ensure_ascii=False, indent=2)
            logging.info("健康状态已写入 %s (总体: %s)",
                         path, health["summary"]["overall"])
        except Exception as e:
            logging.error("写入健康文件失败: %s", e)

    def _health_loop(self):
        """周期性写入健康文件 (守护模式)"""
        while not self._stop_event.is_set():
            self._stop_event.wait(HEALTH_WRITE_INTERVAL)
            if self._stop_event.is_set():
                break
            self._write_health()


# ==================== 日志配置 ====================

def setup_logging(data_dir):
    """配置日志: 同时输出到控制台和文件"""
    ensure_dir(data_dir)
    log_path = os.path.join(data_dir, "fetcher.log")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 清除已有 handler (避免重复)
    root.handlers.clear()
    # 控制台
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(sh)
    # 文件
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(fh)
    except Exception as e:
        root.warning("无法创建日志文件 %s: %s", log_path, e)


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="Hermes 24/7 自动数据抓取服务")
    parser.add_argument("--symbols", type=str, default="BTCUSDT",
                        help="交易对列表，逗号分隔 (默认: BTCUSDT)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="数据存储目录 (默认: data/auto_fetched)")
    parser.add_argument("--once", action="store_true",
                        help="抓取一次后退出 (不进入守护循环)")
    parser.add_argument("--daemon", action="store_true",
                        help="以守护模式持续运行")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = list(DEFAULT_SYMBOLS)

    setup_logging(args.data_dir)
    logging.info("=" * 60)
    logging.info("Hermes 自动数据抓取服务启动")
    logging.info("交易对: %s", symbols)
    logging.info("数据目录: %s", args.data_dir)
    logging.info("模式: %s", "once" if args.once else "daemon")
    logging.info("=" * 60)

    service = AutoDataFetcherService(symbols=symbols, data_dir=args.data_dir)

    if args.once:
        # 单次抓取模式
        results = service.fetch_once()
        ok_count = sum(1 for v in results.values() if v)
        fail_count = len(results) - ok_count
        logging.info("单次抓取完成: 成功 %d, 失败 %d", ok_count, fail_count)
        for name, ok in results.items():
            logging.info("  %s: %s", name, "OK" if ok else "FAIL")
        # 打印健康摘要
        health = service.get_health()
        logging.info("健康状态: %s", health["summary"])
        sys.exit(0 if ok_count > 0 else 1)

    # 守护模式
    if not args.daemon:
        logging.warning("未指定 --daemon 或 --once, 默认使用 --daemon 模式")
    service.start()
    logging.info("服务运行中, Ctrl+C 退出...")
    try:
        while not service._stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("收到中断信号")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
