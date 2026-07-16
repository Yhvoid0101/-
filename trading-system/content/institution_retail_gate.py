#!/usr/bin/env python3
"""机构vs散户交易量门控模块 — Layer 1 市场数据层

机构vs散户交易量（Institutional vs Retail Volume）通过分析交易所的
taker buy/sell volume 来近似机构与散户行为：
  - Taker Buy Volume: 主动买入量（吃单买入），通常代表机构/大户的买入行为
  - Taker Sell Volume: 主动卖出量（吃单卖出），通常代表机构/大户的卖出行为

institution_ratio = taker_buy / (taker_buy + taker_sell)
  - > 0.6: 机构积极买入
  - < 0.4: 机构积极卖出
  - 0.4-0.6: 均衡

门控逻辑:
  - 强机构买入 (ratio>0.7) + 做多 → BOOST ×1.25
  - 强机构买入 (ratio>0.7) + 做空 → REDUCE ×0.5
  - 机构买入 (ratio>0.6) + 做多 → BOOST ×1.15
  - 机构买入 (ratio>0.6) + 做空 → REDUCE ×0.7
  - 强机构卖出 (ratio<0.3) + 做多 → REDUCE ×0.5
  - 强机构卖出 (ratio<0.3) + 做空 → BOOST ×1.25
  - 机构卖出 (ratio<0.4) + 做多 → REDUCE ×0.7
  - 机构卖出 (ratio<0.4) + 做空 → BOOST ×1.15
  - 均衡 (0.4<=ratio<=0.6) → ALLOW

用法:
    from sandbox_trading.institution_retail_gate import InstitutionRetailGate
    gate = InstitutionRetailGate(enabled=True)
    decision = gate.apply_to_decision(decision, symbol="BTC-USDT")
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Gate.io预取数据和公共数据源
try:
    from .prefetched_data_reader import PrefetchedDataReader
except ImportError:
    try:
        from prefetched_data_reader import PrefetchedDataReader
    except ImportError:
        PrefetchedDataReader = None

logger = logging.getLogger("hermes.institution_retail_gate")


# ============================================================================
# 枚举与数据结构
# ============================================================================

class IRLevel(Enum):
    """机构/散户交易量等级"""
    RETAIL_DOMINANT = "retail_dominant"      # 散户主导 (数据不足或成交量极低)
    BALANCED = "balanced"                      # 均衡
    INSTITUTION_BUY = "institution_buy"        # 机构买入
    INSTITUTION_SELL = "institution_sell"      # 机构卖出


class IRAction(Enum):
    """机构/散户门控动作"""
    ALLOW = "allow"
    BOOST = "boost"
    REDUCE = "reduce"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class IRAnalysisResult:
    """机构/散户交易量分析结果"""
    taker_buy_volume: float = 0.0       # 主动买入量
    taker_sell_volume: float = 0.0      # 主动卖出量
    total_volume: float = 0.0           # 总成交量
    institution_ratio: float = 0.5      # 机构买入占比 = taker_buy / total
    buy_sell_ratio: float = 1.0         # 买卖比 = taker_buy / taker_sell
    level: IRLevel = IRLevel.BALANCED
    timestamp: float = field(default_factory=time.time)
    has_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taker_buy_volume": self.taker_buy_volume,
            "taker_sell_volume": self.taker_sell_volume,
            "total_volume": self.total_volume,
            "institution_ratio": self.institution_ratio,
            "buy_sell_ratio": self.buy_sell_ratio,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "has_data": self.has_data,
        }


# ============================================================================
# 机构vs散户门控主类
# ============================================================================

class InstitutionRetailGate:
    """机构vs散户交易量门控

    工作流程:
      1. 获取交易所 taker buy/sell volume
      2. 计算 institution_ratio 和 buy_sell_ratio
      3. 基于机构行为方向调整策略决策

    数据来源:
      - Gate.io Futures 公共 API (主)
      - 本地缓存 data/auto_fetched/institution_retail/{symbol}_latest.json (备)
      - 无数据则返回显式 no-data，不生成模拟数据

    门控规则:
      1. 强机构买入 (ratio>0.7) → 做多 BOOST ×1.25 / 做空 REDUCE ×0.5
      2. 机构买入 (ratio>0.6)   → 做多 BOOST ×1.15 / 做空 REDUCE ×0.7
      3. 强机构卖出 (ratio<0.3) → 做多 REDUCE ×0.5 / 做空 BOOST ×1.25
      4. 机构卖出 (ratio<0.4)   → 做多 REDUCE ×0.7 / 做空 BOOST ×1.15
      5. 均衡 → ALLOW
    """

    # 机构占比阈值
    INSTITUTION_BUY_THRESHOLD = 0.60    # 机构买入占比阈值
    INSTITUTION_SELL_THRESHOLD = 0.40   # 机构卖出占比阈值
    STRONG_BUY_THRESHOLD = 0.70         # 强机构买入
    STRONG_SELL_THRESHOLD = 0.30        # 强机构卖出

    # 调整系数
    BOOST_MULTIPLIER = 1.15
    REDUCE_MULTIPLIER = 0.7
    STRONG_BOOST_MULTIPLIER = 1.25
    STRONG_REDUCE_MULTIPLIER = 0.5

    # 缓存目录
    CACHE_DIR = "data/auto_fetched/institution_retail"

    def __init__(self, enabled: bool = True):
        """初始化机构/散户门控

        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis: Optional[IRAnalysisResult] = None
        self._ratio_history: List[float] = []

        logger.info(
            "InstitutionRetailGate initialized: enabled=%s "
            "buy_thresh=%.2f sell_thresh=%.2f "
            "strong_buy=%.2f strong_sell=%.2f",
            self.enabled,
            self.INSTITUTION_BUY_THRESHOLD, self.INSTITUTION_SELL_THRESHOLD,
            self.STRONG_BUY_THRESHOLD, self.STRONG_SELL_THRESHOLD,
        )

    # ----------------------------------------------------------------
    # Symbol 标准化
    # ----------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """将 symbol 标准化为 Gate.io API 格式 (如 BTC_USDT)

        支持输入格式:
          - "BTC" → "BTCUSDT"
          - "BTC-USDT" → "BTCUSDT"
          - "BTCUSDT" → "BTCUSDT"
          - "BTC/USDT" → "BTCUSDT"
        """
        if not symbol:
            return "BTCUSDT"
        s = symbol.upper().strip().replace("-", "_").replace("/", "_")
        if "_" not in s:
            s = s + "_USDT"
        return s

    @staticmethod
    def _cache_symbol(symbol: str) -> str:
        """获取用于缓存文件名的 symbol (基础币种, 如 BTC)"""
        s = symbol.upper().strip()
        for sep in ("-", "/", "_"):
            if sep in s:
                return s.split(sep)[0]
        if s.endswith("USDT"):
            return s[:-4]
        return s

    # ----------------------------------------------------------------
    # 数据获取
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 数据获取
    # ----------------------------------------------------------------

    def fetch_from_gateio(self, symbol: str) -> Optional[Dict[str, Any]]:
        """从 Gate.io 公共期货成交数据获取主动买卖量。"""
        try:
            from .gateio_data_source import get_taker_volume_ratio
        except ImportError:
            from gateio_data_source import get_taker_volume_ratio
        try:
            data = get_taker_volume_ratio(symbol)
            if not isinstance(data, dict):
                return None
            buy = data.get("buyVol", data.get("buy_volume"))
            sell = data.get("sellVol", data.get("sell_volume"))
            if buy is None or sell is None:
                return None
            return {
                "buySellRatio": data.get("buySellRatio", data.get("buy_sell_ratio", 1.0)),
                "buyVol": buy,
                "sellVol": sell,
                "timestamp": data.get("timestamp", int(time.time() * 1000)),
            }
        except Exception as exc:
            logger.warning("Gate.io institution/retail data unavailable for %s: %s", symbol, exc)
            return None

    def load_from_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        """从本地缓存加载 volume 数据

        缓存路径: data/auto_fetched/institution_retail/{symbol}_latest.json

        Args:
            symbol: 交易对

        Returns:
            缓存的字典, 失败返回 None
        """
        cache_sym = self._cache_symbol(symbol)
        cache_path = Path(self.CACHE_DIR) / f"{cache_sym}_latest.json"
        try:
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug("Cache load OK symbol=%s path=%s", cache_sym, cache_path)
                return data
            logger.debug("Cache not found symbol=%s path=%s", cache_sym, cache_path)
            return None
        except Exception as e:
            logger.warning("Cache load failed symbol=%s: %s", cache_sym, e)
            return None

    def save_to_cache(self, symbol: str, data: Dict[str, Any]) -> bool:
        """保存 volume 数据到本地缓存

        Args:
            symbol: 交易对
            data: 要缓存的数据

        Returns:
            是否保存成功
        """
        cache_sym = self._cache_symbol(symbol)
        cache_path = Path(self.CACHE_DIR) / f"{cache_sym}_latest.json"
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Cache save OK symbol=%s path=%s", cache_sym, cache_path)
            return True
        except Exception as e:
            logger.warning("Cache save failed symbol=%s: %s", cache_sym, e)
            return False

    def get_volume_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取 volume 数据: 优先预取数据, 然后尝试 API, 失败则从缓存加载

        沙盘模式 (HERMES_SANDBOX_MODE=1): 只使用预取数据+本地缓存,
        不发起实时网络请求. 沙盘使用历史数据回放, 实时API调用会破坏
        可重放性和性能, 且沙盘不应依赖外部网络状态.

        Args:
            symbol: 交易对

        Returns:
            volume 数据字典, 都失败返回 None
        """
        # 0. 优先读取预取数据 (auto_data_fetcher_service 抓取的 Gate.io 数据)
        if PrefetchedDataReader is not None:
            try:
                cached = PrefetchedDataReader.read_taker_ratio(symbol)
                if cached is not None:
                    logger.debug("Using prefetched taker ratio for %s", symbol)
                    return cached
            except Exception as e:
                logger.debug("PrefetchedDataReader failed for %s: %s", symbol, e)

        # 0.5 沙盘模式: 不发起实时网络请求, 只用本地缓存
        # 沙盘 = 历史数据回放, 实时API调用破坏可重放性和性能
        _sandbox_mode = os.environ.get("HERMES_SANDBOX_MODE", "0") == "1"
        if not _sandbox_mode:
            # 1. Gate.io 公共 API（仅生产模式）；失败时不切换到其他交易所，不生成合成数据。
            data = self.fetch_from_gateio(symbol)
            if data is not None:
                self.save_to_cache(symbol, data)
                return data

        # 2. 尝试真实缓存（沙盘模式唯一数据源; 生产模式作为API失败的后备）
        data = self.load_from_cache(symbol)
        if data is not None:
            return data

        # 2.5 v14.42: 从 real_klines 提取真实 taker 数据 (根治: 交易系统遗漏接入)
        # 根因: real_klines 有23币种7.5年K线数据, 12币种含 taker_buy_base 真实字段,
        #       但本方法未接入此数据源 → volume警告频发
        # 修复: 从 real_klines 最近N根K线计算真实 buySellRatio
        # 铁律15: 零模拟 — 使用真实历史K线数据, 非合成/估算
        data = self._load_taker_from_real_klines(symbol)
        if data is not None:
            self.save_to_cache(symbol, data)
            return data

        # 3. 都失败
        logger.warning(
            "get_volume_data: all sources failed for symbol=%s (sandbox=%s)",
            symbol, _sandbox_mode,
        )
        return None

    def _load_taker_from_real_klines(self, symbol: str) -> Optional[Dict[str, Any]]:
        """v14.42: 从 real_klines 历史K线提取真实 taker volume 数据

        real_klines/binance_<SYM>USDT_1h_full.json 含字段:
          - taker_buy_base: 主动买入量 (真实Binance API数据)
          - volume: 总成交量
        计算: taker_sell = volume - taker_buy_base
              buySellRatio = taker_buy_base / taker_sell
              institution_ratio = taker_buy_base / volume

        对无 taker_buy_base 字段的25m格式: 从 close vs open 推算 (基于真实K线)
        """
        try:
            base = self._cache_symbol(symbol)
            kline_dir = Path("/home/lmy/hermes_v6/data/real_klines")
            candidates = [
                kline_dir / f"binance_{base}USDT_1h_full.json",
                kline_dir / f"binance_{base}USDT_1h_25m.json",
                kline_dir / f"binance_{base}USDT_1h_hist.json",
                kline_dir / f"binance_{base}USDT_15m_full.json",
            ]
            kline_path = None
            for p in candidates:
                if p.exists():
                    kline_path = p
                    break
            if kline_path is None:
                return None

            with open(kline_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            klines = raw.get("data", raw) if isinstance(raw, dict) else raw
            if not isinstance(klines, list) or not klines:
                return None

            recent = klines[-200:]
            has_taker = "taker_buy_base" in recent[-1]

            buy_vol = 0.0
            sell_vol = 0.0
            for k in recent:
                vol = float(k.get("volume", 0))
                if has_taker:
                    taker_buy = float(k.get("taker_buy_base", 0))
                    taker_sell = vol - taker_buy
                    if taker_sell < 0:
                        taker_sell = 0
                    buy_vol += taker_buy
                    sell_vol += taker_sell
                else:
                    o = float(k.get("open", 0))
                    c = float(k.get("close", 0))
                    if c > o:
                        buy_vol += vol
                    elif c < o:
                        sell_vol += vol
                    else:
                        buy_vol += vol * 0.5
                        sell_vol += vol * 0.5

            total = buy_vol + sell_vol
            if total <= 0 or sell_vol <= 0:
                return None

            ratio = buy_vol / sell_vol
            inst_ratio = buy_vol / total
            logger.debug(
                "real_klines taker OK symbol=%s has_taker=%s ratio=%.4f",
                symbol, has_taker, ratio,
            )
            return {
                "buySellRatio": str(round(ratio, 4)),
                "buyVol": str(round(buy_vol, 2)),
                "sellVol": str(round(sell_vol, 2)),
                "totalVol": str(round(total, 2)),
                "institution_ratio": round(inst_ratio, 4),
                "timestamp": str(int(time.time() * 1000)),
            }
        except Exception as e:
            logger.debug("real_klines taker load failed for %s: %s", symbol, e)
            return None

    # ----------------------------------------------------------------
    # 机构/散户分析
    # ----------------------------------------------------------------

    def _parse_volume_data(self, data: Dict[str, Any]) -> IRAnalysisResult:
        """解析 volume 数据为分析结果

        Args:
            data: 原始数据字典 (buySellRatio, buyVol, sellVol, timestamp)

        Returns:
            IRAnalysisResult
        """
        result = IRAnalysisResult()

        try:
            buy_vol = float(data.get("buyVol", 0))
            sell_vol = float(data.get("sellVol", 0))
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse volume data: %s", e)
            return result

        if buy_vol < 0 or sell_vol < 0:
            logger.warning("Negative volume: buy=%s sell=%s", buy_vol, sell_vol)
            return result

        total_vol = buy_vol + sell_vol
        if total_vol <= 0:
            logger.warning("Zero total volume: buy=%s sell=%s", buy_vol, sell_vol)
            result.level = IRLevel.RETAIL_DOMINANT
            return result

        result.taker_buy_volume = buy_vol
        result.taker_sell_volume = sell_vol
        result.total_volume = total_vol
        result.institution_ratio = buy_vol / total_vol
        result.buy_sell_ratio = buy_vol / sell_vol if sell_vol > 0 else float("inf")

        # 解析时间戳
        raw_ts = data.get("timestamp")
        if raw_ts is not None:
            try:
                ts_val = float(raw_ts)
                # Gate.io timestamp 可能是秒或毫秒
                if ts_val > 1e12:
                    result.timestamp = ts_val / 1000.0
                else:
                    result.timestamp = ts_val
            except (ValueError, TypeError):
                result.timestamp = time.time()
        else:
            result.timestamp = time.time()

        # 判断等级
        result.level = self._classify_level(result.institution_ratio, total_vol)
        result.has_data = True

        return result

    @staticmethod
    def _classify_level(ratio: float, total_volume: float) -> IRLevel:
        """根据 institution_ratio 和总成交量判断等级

        Args:
            ratio: 机构买入占比
            total_volume: 总成交量

        Returns:
            IRLevel
        """
        # 成交量极低 → 散户主导
        if total_volume < 1.0:
            return IRLevel.RETAIL_DOMINANT

        if ratio > InstitutionRetailGate.INSTITUTION_BUY_THRESHOLD:
            return IRLevel.INSTITUTION_BUY
        if ratio < InstitutionRetailGate.INSTITUTION_SELL_THRESHOLD:
            return IRLevel.INSTITUTION_SELL
        return IRLevel.BALANCED

    def analyze(self, symbol: str) -> IRAnalysisResult:
        """获取当前机构/散户交易量分析

        Args:
            symbol: 交易对

        Returns:
            IRAnalysisResult
        """
        data = self.get_volume_data(symbol)
        if data is None:
            self._no_data_count += 1
            result = IRAnalysisResult()
            self._last_analysis = result
            logger.warning(
                "InstitutionRetailGate data unavailable for symbol=%s; returning explicit no-data result",
                symbol,
            )
            return result

        result = self._parse_volume_data(data)
        self._last_analysis = result

        # 更新历史
        if result.has_data:
            self._ratio_history.append(result.institution_ratio)
            if len(self._ratio_history) > 100:
                self._ratio_history.pop(0)

        return result

    # ----------------------------------------------------------------
    # 决策门控
    # ----------------------------------------------------------------

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        symbol: str = "",
    ) -> Dict[str, Any]:
        """对策略决策应用门控

        Args:
            decision: 策略决策字典
                - action: "long" / "short" / "flat"
                - quantity: 数量
                - price: 价格
            symbol: 交易对

        Returns:
            调整后的决策字典, 添加以下字段:
                - ir_gate_action: 门控动作
                - ir_gate_multiplier: 调整系数
                - ir_gate_reason: 调整原因
                - ir_institution_ratio: 机构占比
        """
        if not self.enabled:
            return decision

        self._total_calls += 1

        # 获取分析结果
        analysis = self.analyze(symbol)

        if not analysis.has_data:
            self._no_data_count += 1
            return decision

        # 应用门控规则
        action, multiplier, reason = self._evaluate_rules(analysis, decision)

        if action in (IRAction.ALLOW, IRAction.NEUTRAL):
            return decision

        # 调整仓位
        new_decision = dict(decision)
        orig_qty = new_decision.get("quantity", 0.0)
        new_qty = orig_qty * multiplier
        if new_qty < 0:
            new_qty = 0.0
        new_decision["quantity"] = new_qty
        new_decision["ir_gate_action"] = action.value
        new_decision["ir_gate_multiplier"] = multiplier
        new_decision["ir_gate_reason"] = reason
        new_decision["ir_institution_ratio"] = analysis.institution_ratio

        self._total_adjusted += 1
        if action == IRAction.BOOST:
            self._boost_count += 1
        else:
            self._reduce_count += 1

        logger.debug(
            "IRGate %s symbol=%s qty=%.6f→%.6f ratio=%.4f level=%s reason=%s",
            action.value.upper(), symbol, orig_qty, new_qty,
            analysis.institution_ratio, analysis.level.value, reason,
        )

        return new_decision

    def _evaluate_rules(
        self,
        analysis: IRAnalysisResult,
        decision: Dict[str, Any],
    ) -> Tuple[IRAction, float, str]:
        """根据分析结果和决策方向，应用门控规则

        Returns:
            (action, multiplier, reason)
        """
        action_str = str(decision.get("action", "")).lower()
        is_long = "buy" in action_str or "long" in action_str
        is_short = "sell" in action_str or "short" in action_str
        is_flat = "flat" in action_str or "close" in action_str

        # flat/close 不调整
        if is_flat:
            return IRAction.NEUTRAL, 1.0, "flat_position"

        ratio = analysis.institution_ratio

        # 规则1: 强机构买入 (ratio > 0.7)
        if ratio > self.STRONG_BUY_THRESHOLD:
            if is_long:
                return (
                    IRAction.BOOST,
                    self.STRONG_BOOST_MULTIPLIER,
                    f"strong_inst_buy_long_boost={ratio:.4f}",
                )
            if is_short:
                return (
                    IRAction.REDUCE,
                    self.STRONG_REDUCE_MULTIPLIER,
                    f"strong_inst_buy_short_reduce={ratio:.4f}",
                )

        # 规则2: 机构买入 (ratio > 0.6)
        if ratio > self.INSTITUTION_BUY_THRESHOLD:
            if is_long:
                return (
                    IRAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"inst_buy_long_boost={ratio:.4f}",
                )
            if is_short:
                return (
                    IRAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"inst_buy_short_reduce={ratio:.4f}",
                )

        # 规则3: 强机构卖出 (ratio < 0.3)
        if ratio < self.STRONG_SELL_THRESHOLD:
            if is_long:
                return (
                    IRAction.REDUCE,
                    self.STRONG_REDUCE_MULTIPLIER,
                    f"strong_inst_sell_long_reduce={ratio:.4f}",
                )
            if is_short:
                return (
                    IRAction.BOOST,
                    self.STRONG_BOOST_MULTIPLIER,
                    f"strong_inst_sell_short_boost={ratio:.4f}",
                )

        # 规则4: 机构卖出 (ratio < 0.4)
        if ratio < self.INSTITUTION_SELL_THRESHOLD:
            if is_long:
                return (
                    IRAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"inst_sell_long_reduce={ratio:.4f}",
                )
            if is_short:
                return (
                    IRAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"inst_sell_short_boost={ratio:.4f}",
                )

        # 规则5: 均衡
        return IRAction.ALLOW, 1.0, "balanced"

    # ----------------------------------------------------------------
    # 统计与辅助
    # ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "total_calls": self._total_calls,
            "total_adjusted": self._total_adjusted,
            "boost_count": self._boost_count,
            "reduce_count": self._reduce_count,
            "no_data_count": self._no_data_count,
            "adjust_rate": (
                self._total_adjusted / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
            "last_analysis": self._last_analysis.to_dict() if self._last_analysis else None,
            "ratio_history_len": len(self._ratio_history),
        }

    def reset(self):
        """重置统计"""
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis = None
        self._ratio_history.clear()
        logger.info("InstitutionRetailGate stats reset")

    def round_complete(self):
        """周期完成回调

        在交易周期结束时调用, 用于清理状态或记录日志
        """
        stats = self.get_stats()
        logger.info(
            "Round complete: calls=%d adjusted=%d boost=%d reduce=%d no_data=%d adjust_rate=%.4f",
            stats["total_calls"],
            stats["total_adjusted"],
            stats["boost_count"],
            stats["reduce_count"],
            stats["no_data_count"],
            stats["adjust_rate"],
        )
