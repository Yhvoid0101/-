#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预取数据读取器 — 统一缓存数据读取接口

为各 Gate 提供统一的预取数据读取功能。
auto_data_fetcher_service 将数据存储到 data/auto_fetched/ 目录，
本模块负责读取并检查数据新鲜度。

数据目录结构:
  data/auto_fetched/
    ├── fear_greed/latest.json          — 恐惧贪婪指数
    ├── urpd/latest.json                — UTXO 实现价格
    ├── ticker/{SYM}_latest.json        — Ticker 数据 (价格/成交量)
    ├── order_book/{SYM}_latest.json    — 订单簿深度
    ├── institution_retail/{SYM}_latest.json — 主动买卖量比
    ├── long_short_ratio/{SYM}_latest.json   — 多空持仓比
    └── liquidations/latest.json        — 强平事件

文件格式 (由 auto_data_fetcher_service.save_json_file 写入):
  {
      "data": <实际数据>,
      "timestamp": <Unix 时间戳>,
      "source": <数据源>,
      "fetch_time": <ISO8601 字符串>
  }

特性:
  - 自动检查数据新鲜度 (默认 5 分钟内有效)
  - 文件不存在或过期返回 None
  - Symbol 自动转换 (BTC-USDT / BTCUSDT / BTC 均可)
  - 线程安全 (无状态读取)

用法:
    from prefetched_data_reader import PrefetchedDataReader
    ticker = PrefetchedDataReader.read_ticker("BTC-USDT")
    if ticker:
        print(ticker["last"])

    # 自定义过期时间
    ob = PrefetchedDataReader.read_order_book("BTC-USDT", max_age=60)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.prefetched_data_reader")


class PrefetchedDataReader:
    """预取数据统一读取器

    所有方法均为静态方法，无状态，线程安全。
    每个方法返回 None 如果:
      - 文件不存在
      - 文件解析失败
      - 数据过期 (超过 max_age 秒)
    """

    # 数据根目录
    DATA_DIR = "data/auto_fetched"

    # 默认最大年龄 (秒) — 5 分钟
    DEFAULT_MAX_AGE = 300

    # ==================== 内部工具 ====================

    @classmethod
    def _resolve_data_dir(cls) -> str:
        """解析数据目录路径

        优先级:
          1. 环境变量 HERMES_DATA_DIR
          2. 当前工作目录下的 DATA_DIR
          3. /home/lmy/hermes_v6/sandbox_trading/data/auto_fetched (WSL 默认)
        """
        env_dir = os.environ.get("HERMES_DATA_DIR")
        if env_dir and os.path.isdir(env_dir):
            return env_dir

        # 尝试当前目录
        if os.path.isdir(cls.DATA_DIR):
            return cls.DATA_DIR

        # 尝试 WSL 默认路径
        wsl_path = "/home/lmy/hermes_v6/sandbox_trading/data/auto_fetched"
        if os.path.isdir(wsl_path):
            return wsl_path

        return cls.DATA_DIR

    @classmethod
    def _normalize_symbol(cls, symbol: str) -> str:
        """将 symbol 标准化为文件名格式 (如 BTCUSDT)

        支持输入: BTC-USDT, BTC/USDT, BTC_USDT, BTCUSDT, BTC
        输出: BTCUSDT
        """
        if not symbol:
            return "BTCUSDT"
        s = symbol.upper().strip()
        s = s.replace("-", "").replace("/", "").replace("_", "")
        if not s.endswith("USDT"):
            s = s + "USDT"
        return s

    @classmethod
    def _read_file(cls, file_path: str, max_age: int) -> Optional[Any]:
        """读取并验证数据文件

        Args:
            file_path: 文件完整路径
            max_age: 最大允许年龄 (秒)

        Returns:
            文件中的 "data" 字段内容, 过期或不存在返回 None
        """
        try:
            if not os.path.exists(file_path):
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            # 检查数据新鲜度
            # 优先使用 fetched_at (实际抓取时间), 其次用 timestamp
            timestamp = payload.get("fetched_at", 0) or payload.get("timestamp", 0)
            if timestamp > 0:
                age = time.time() - float(timestamp)
                if age > max_age:
                    logger.debug(
                        "数据已过期: %s (age=%.0fs > max_age=%ds, ts_field=%s)",
                        file_path, age, max_age,
                        "fetched_at" if "fetched_at" in payload else "timestamp",
                    )
                    return None

            # 返回 data 字段
            data = payload.get("data")
            if data is None:
                # 兼容: 有些文件可能直接存数据 (无 data 包装)
                data = payload

            return data

        except json.JSONDecodeError as e:
            logger.warning("JSON 解析失败: %s - %s", file_path, e)
            return None
        except Exception as e:
            logger.warning("读取文件失败: %s - %s", file_path, e)
            return None

    @classmethod
    def _build_path(cls, *parts: str) -> str:
        """构建数据文件路径"""
        base = cls._resolve_data_dir()
        return os.path.join(base, *parts)

    # ==================== 公开接口 ====================

    @classmethod
    def read_fear_greed(cls, max_age: int = None) -> Optional[Dict[str, Any]]:
        """读取恐惧贪婪指数

        Args:
            max_age: 最大年龄 (秒), 默认 3600 (1 小时)

        Returns:
            Fear & Greed 数据字典, 失败返回 None
        """
        age = max_age if max_age is not None else 3600
        path = cls._build_path("fear_greed", "latest.json")
        return cls._read_file(path, age)

    @classmethod
    def read_urpd(cls, max_age: int = None) -> Optional[Dict[str, Any]]:
        """读取 UTXO 实现价格数据

        Args:
            max_age: 最大年龄 (秒), 默认 86400 (1 天)

        Returns:
            URPD 数据字典, 失败返回 None
        """
        age = max_age if max_age is not None else 86400
        path = cls._build_path("urpd", "latest.json")
        return cls._read_file(path, age)

    @classmethod
    def read_ticker(cls, symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
        """读取 Ticker 数据 (价格/成交量/24h高低)

        Args:
            symbol: 交易对 (BTC-USDT / BTCUSDT / BTC)
            max_age: 最大年龄 (秒), 默认 300 (5 分钟)

        Returns:
            Ticker 数据字典, 失败返回 None
        """
        age = max_age if max_age is not None else cls.DEFAULT_MAX_AGE
        sym = cls._normalize_symbol(symbol)
        path = cls._build_path("ticker", f"{sym}_latest.json")
        return cls._read_file(path, age)

    @classmethod
    def read_order_book(cls, symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
        """读取订单簿深度数据

        Args:
            symbol: 交易对
            max_age: 最大年龄 (秒), 默认 120 (2 分钟, 订单簿变化快)

        Returns:
            订单簿数据字典 (asks, bids, mid_price, spread, imbalance), 失败返回 None
        """
        age = max_age if max_age is not None else 120
        sym = cls._normalize_symbol(symbol)
        path = cls._build_path("order_book", f"{sym}_latest.json")
        return cls._read_file(path, age)

    @classmethod
    def read_taker_ratio(cls, symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
        """读取主动买卖量比数据 (机构/散户)

        Args:
            symbol: 交易对
            max_age: 最大年龄 (秒), 默认 300 (5 分钟)

        Returns:
            买卖量比数据字典 (buy_volume, sell_volume, ratio), 失败返回 None
        """
        age = max_age if max_age is not None else cls.DEFAULT_MAX_AGE
        sym = cls._normalize_symbol(symbol)
        path = cls._build_path("institution_retail", f"{sym}_latest.json")
        return cls._read_file(path, age)

    @classmethod
    def read_lsr(cls, symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
        """读取多空持仓比数据

        Args:
            symbol: 交易对
            max_age: 最大年龄 (秒), 默认 300 (5 分钟)

        Returns:
            多空比数据字典 (long_ratio, short_ratio, funding_rate), 失败返回 None
        """
        age = max_age if max_age is not None else cls.DEFAULT_MAX_AGE
        sym = cls._normalize_symbol(symbol)
        path = cls._build_path("long_short_ratio", f"{sym}_latest.json")
        return cls._read_file(path, age)

    @classmethod
    def read_liquidations(cls, max_age: int = None) -> Optional[Any]:
        """读取强平事件数据

        Args:
            max_age: 最大年龄 (秒), 默认 300 (5 分钟)

        Returns:
            强平事件列表, 失败返回 None
        """
        age = max_age if max_age is not None else cls.DEFAULT_MAX_AGE
        path = cls._build_path("liquidations", "latest.json")
        return cls._read_file(path, age)

    @classmethod
    def read_smart_money(cls, symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
        """读取聪明钱/鲸鱼交易数据

        Args:
            symbol: 交易对
            max_age: 最大年龄 (秒), 默认 120

        Returns:
            聪明钱数据字典 (transactions list), 失败返回 None
        """
        age = max_age if max_age is not None else 120
        sym = cls._normalize_symbol(symbol)
        path = cls._build_path("smart_money", f"{sym}_latest.json")
        return cls._read_file(path, age)

    # ==================== 诊断工具 ====================

    @classmethod
    def get_data_status(cls) -> Dict[str, Any]:
        """获取所有数据文件的状态摘要 (用于诊断)

        Returns:
            {
                "data_dir": str,
                "files": {relative_path: {exists, age, timestamp}},
                "total_files": int,
                "fresh_files": int,
            }
        """
        base = cls._resolve_data_dir()
        status = {
            "data_dir": base,
            "files": {},
            "total_files": 0,
            "fresh_files": 0,
        }

        if not os.path.isdir(base):
            status["error"] = f"数据目录不存在: {base}"
            return status

        # 遍历所有子目录
        for subdir in ["fear_greed", "urpd", "ticker", "order_book",
                       "institution_retail", "long_short_ratio", "liquidations"]:
            subdir_path = os.path.join(base, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(subdir_path, fname)
                rel_path = f"{subdir}/{fname}"
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    ts = payload.get("timestamp", 0)
                    age = time.time() - float(ts) if ts > 0 else -1
                    exists = True
                except Exception:
                    ts = 0
                    age = -1
                    exists = True  # 文件存在但无法解析

                status["files"][rel_path] = {
                    "exists": exists,
                    "age": int(age) if age >= 0 else None,
                    "timestamp": ts,
                }
                status["total_files"] += 1
                if 0 <= age <= cls.DEFAULT_MAX_AGE:
                    status["fresh_files"] += 1

        return status


# ==================== 模块级便捷函数 ====================


def read_fear_greed(max_age: int = None) -> Optional[Dict[str, Any]]:
    return PrefetchedDataReader.read_fear_greed(max_age)


def read_urpd(max_age: int = None) -> Optional[Dict[str, Any]]:
    return PrefetchedDataReader.read_urpd(max_age)


def read_ticker(symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
    return PrefetchedDataReader.read_ticker(symbol, max_age)


def read_order_book(symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
    return PrefetchedDataReader.read_order_book(symbol, max_age)


def read_taker_ratio(symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
    return PrefetchedDataReader.read_taker_ratio(symbol, max_age)


def read_lsr(symbol: str, max_age: int = None) -> Optional[Dict[str, Any]]:
    return PrefetchedDataReader.read_lsr(symbol, max_age)


def read_liquidations(max_age: int = None) -> Optional[Any]:
    return PrefetchedDataReader.read_liquidations(max_age)
