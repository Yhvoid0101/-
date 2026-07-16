#!/usr/bin/env python3
"""URPD/UTXO实现价格(Realized Price)锚点模块 — Layer 1 市场数据层

URPD = UTXO Realized Price Distribution，链上已实现价格分布
  - Realized Price = 所有UTXO按最后移动时的价格加权的平均值，代表链上持有者的平均成本
  - MVRV Ratio = Market Value / Realized Value = market_price / realized_price
  - MVRV > 1: 市场价高于链上成本，持有者盈利
  - MVRV < 1: 市场价低于链上成本，持有者亏损

门控逻辑:
  - 深度折价 (MVRV<0.5) + 做多 → BOOST ×1.25 (远低于链上成本，抄底机会)
  - 深度折价 (MVRV<0.5) + 做空 → REDUCE ×0.5
  - 折价 (MVRV<0.8) + 做多 → BOOST ×1.15
  - 折价 (MVRV<0.8) + 做空 → REDUCE ×0.7
  - 极端溢价 (MVRV>2.5) + 做多 → REDUCE ×0.5 (顶部信号)
  - 极端溢价 (MVRV>2.5) + 做空 → BOOST ×1.25
  - 溢价 (MVRV>1.5) + 做多 → REDUCE ×0.7
  - 溢价 (MVRV>1.5) + 做空 → BOOST ×1.15
  - 中性 (0.8<=MVRV<=1.5) → ALLOW

用法:
    from sandbox_trading.urpd_realized_price import URPDGate
    gate = URPDGate(enabled=True)
    decision = gate.apply_to_decision(decision, symbol="BTC-USDT", market_price=65000.0)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# 预取数据读取器 (可选, 失败时降级到原有逻辑)
try:
    from .prefetched_data_reader import PrefetchedDataReader
except ImportError:
    try:
        from prefetched_data_reader import PrefetchedDataReader
    except ImportError:
        PrefetchedDataReader = None

logger = logging.getLogger("hermes.urpd_realized_price")


# ============================================================================
# 枚举与数据结构
# ============================================================================

class URPDLevel(Enum):
    """URPD/MVRV 等级"""
    DEEP_DISCOUNT = "deep_discount"      # 深度折价 (MVRV<0.5)
    DISCOUNT = "discount"                # 折价 (MVRV<0.8)
    NEUTRAL = "neutral"                  # 中性 (0.8<=MVRV<=1.5)
    PREMIUM = "premium"                  # 溢价 (MVRV>1.5)
    EXTREME_PREMIUM = "extreme_premium"  # 极端溢价 (MVRV>2.5)


class URPDAction(Enum):
    """URPD 门控动作"""
    ALLOW = "allow"
    BOOST = "boost"
    REDUCE = "reduce"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class URPDAnalysisResult:
    """URPD 分析结果"""
    market_price: float = 0.0
    realized_price: float = 0.0
    mvrv_ratio: float = 1.0
    level: URPDLevel = URPDLevel.NEUTRAL
    timestamp: float = field(default_factory=time.time)
    has_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_price": self.market_price,
            "realized_price": self.realized_price,
            "mvrv_ratio": self.mvrv_ratio,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "has_data": self.has_data,
        }


# ============================================================================
# URPD 门控主类
# ============================================================================

class URPDGate:
    """URPD/UTXO实现价格门控

    工作流程:
      1. 获取市场价和链上已实现价格(realized_price)
      2. 计算MVRV比率 = market_price / realized_price
      3. 基于MVRV等级调整策略决策

    数据获取优先级:
      1. API获取真实realized_price (blockchain.com / glassnode)
      2. 本地缓存 data/auto_fetched/urpd/latest.json
      3. 估算: realized_price = market_price * 0.7

    门控规则:
      1. 深度折价 (MVRV<0.5) → 做多 BOOST×1.25 / 做空 REDUCE×0.5
      2. 折价 (MVRV<0.8) → 做多 BOOST×1.15 / 做空 REDUCE×0.7
      3. 极端溢价 (MVRV>2.5) → 做多 REDUCE×0.5 / 做空 BOOST×1.25
      4. 溢价 (MVRV>1.5) → 做多 REDUCE×0.7 / 做空 BOOST×1.15
      5. 中性 → ALLOW
    """

    # MVRV 阈值
    DEEP_DISCOUNT_MVRV = 0.5    # 深度折价 (MVRV<0.5, 抄底机会)
    DISCOUNT_MVRV = 0.8         # 折价 (MVRV<0.8)
    PREMIUM_MVRV = 1.5          # 溢价 (MVRV>1.5, 获利盘多)
    EXTREME_PREMIUM_MVRV = 2.5  # 极端溢价 (MVRV>2.5, 顶部信号)

    # 调整系数
    BOOST_MULTIPLIER = 1.15
    REDUCE_MULTIPLIER = 0.7
    EXTREME_BOOST_MULTIPLIER = 1.25
    EXTREME_REDUCE_MULTIPLIER = 0.5

    # 估算系数已废弃 — 零模拟零降级, 只使用真实链上数据
    # 旧版 ESTIMATE_FACTOR = 0.7 导致 MVRV 恒等于 1.4286 (NEUTRAL区间), 永不触发调整
    # 新版: 数据不可用时返回 None (标记 no_data, 不调整), 绝不估算

    # API 超时 (秒)
    API_TIMEOUT = 10

    # 缓存路径 — auto_data_fetcher 写入的标准路径
    CACHE_DIR = "data/auto_fetched/urpd"
    CACHE_FILE = "data/auto_fetched/urpd/latest.json"

    def __init__(self, enabled: bool = True):
        """初始化URPD门控

        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis: Optional[URPDAnalysisResult] = None
        self._realized_price_history: List[float] = []

        logger.info(
            "URPDGate initialized: enabled=%s "
            "deep_discount_mvrv=%.2f discount_mvrv=%.2f "
            "premium_mvrv=%.2f extreme_premium_mvrv=%.2f",
            self.enabled,
            self.DEEP_DISCOUNT_MVRV, self.DISCOUNT_MVRV,
            self.PREMIUM_MVRV, self.EXTREME_PREMIUM_MVRV,
        )

    # ----------------------------------------------------------------
    # 数据获取
    # ----------------------------------------------------------------

    def fetch_from_api(self) -> Optional[float]:
        """从API获取realized_price

        尝试从 glassnode 或 blockchain.com 获取链上数据。
        免费API有限，沙盘环境通常不可用，失败返回None。

        Returns:
            realized_price 或 None
        """
        import urllib.request
        import urllib.error

        # 尝试 glassnode MVRV API (需要API key)
        api_key = os.environ.get("GLASSNODE_API_KEY", "")
        if api_key:
            try:
                url = f"https://api.glassnode.com/v1/metrics/market/mvrv?api_key={api_key}"
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Hermes-URPD/1.0"}
                )
                with urllib.request.urlopen(req, timeout=self.API_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data and isinstance(data, list) and len(data) > 0:
                        mvrv = data[-1].get("v")
                        if mvrv and mvrv > 0:
                            logger.info("Glassnode MVRV fetched: %.4f", mvrv)
                            self._cached_mvrv = mvrv
                            return None  # 需要配合market_price计算
            except Exception as e:
                logger.warning("Glassnode API failed: %s", e)

        # blockchain.info 提供 circulating supply, 不直接提供 realized_price
        try:
            url = "https://blockchain.info/q/totalbc"
            req = urllib.request.Request(
                url, headers={"User-Agent": "Hermes-URPD/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.API_TIMEOUT) as resp:
                supply = float(resp.read().decode("utf-8").strip())
                logger.debug("blockchain.info totalbc: %f", supply)
                # supply 不是 realized_price, 返回None走缓存/估算
                return None
        except Exception as e:
            logger.warning("blockchain.info API failed: %s", e)

        return None

    def load_from_cache(self) -> Optional[float]:
        """从本地缓存加载realized_price

        读取 data/auto_fetched/urpd/latest.json
        auto_data_fetcher_service 用 save_json_file() 保存, 数据在 "data" 字段里:
          {"data": {"realized_price": ..., ...}, "timestamp": ..., "source": ..., "fetch_time": ...}

        Returns:
            realized_price 或 None
        """
        try:
            if not os.path.exists(self.CACHE_FILE):
                return None
            with open(self.CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            # auto_data_fetcher 的 save_json_file 把数据放在 "data" 字段里
            data_obj = cache.get("data", cache)
            rp = data_obj.get("realized_price")
            if rp and float(rp) > 0:
                # 使用 fetched_at (优先) 或 timestamp 或 fetch_time
                ts = (
                    cache.get("fetched_at", 0)
                    or cache.get("timestamp", 0)
                    or 0
                )
                if isinstance(ts, str):
                    # fetch_time 是 ISO 字符串, 无法直接计算 age
                    age = -1
                else:
                    age = time.time() - ts
                logger.debug(
                    "Cache loaded: realized_price=%.2f age=%.0fs source=%s",
                    float(rp), age, cache.get("source", "unknown"),
                )
                return float(rp)
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
        return None

    def _save_to_cache(self, realized_price: float) -> None:
        """保存realized_price到缓存 (已废弃 — 不再覆盖 auto_data_fetcher 的缓存文件)

        旧版会用估算值覆盖 auto_data_fetcher 写入的真实数据, 导致缓存冲突。
        新版: 仅记录日志, 不写入文件, 避免破坏真实数据。
        """
        # NO-OP: 零降级策略, 不再用估算值覆盖真实数据
        logger.debug(
            "_save_to_cache deprecated (no-op): realized_price=%.2f not saved "
            "(auto_data_fetcher is the single source of truth)",
            realized_price,
        )

    def get_realized_price(self, market_price: float) -> URPDAnalysisResult:
        """获取realized_price并构建分析结果

        优先级 (零降级策略 — 只使用真实链上数据):
          0. 预取数据 (auto_data_fetcher_service 通过 BGeometrics API 抓取)
          1. 本地缓存 (auto_data_fetcher 写入的真实数据)
          2. 数据不可用 → 返回 has_data=False, 不调整 (绝不估算)

        Args:
            market_price: 当前市场价格

        Returns:
            URPDAnalysisResult
        """
        result = URPDAnalysisResult()
        result.market_price = market_price

        if market_price <= 0:
            return result

        realized_price: Optional[float] = None

        # 0. 优先读取预取数据 (auto_data_fetcher 通过 BGeometrics API 抓取)
        if PrefetchedDataReader is not None:
            try:
                # PrefetchedDataReader.read_urpd() 默认 max_age=86400 (1天)
                # 我们的 fetcher 间隔 1 小时, 设置 max_age=7200 (2小时缓冲) 避免数据过期
                urpd_data = PrefetchedDataReader.read_urpd(max_age=7200)
                if urpd_data is not None and isinstance(urpd_data, dict):
                    # PrefetchedDataReader._read_file 已经返回 payload["data"] 内容
                    # 即 {realized_price, realized_cap_usd, total_btc, mvrv_ratio, ...}
                    rp = urpd_data.get("realized_price")
                    if rp and float(rp) > 0:
                        realized_price = float(rp)
                        logger.info(
                            "Using prefetched realized_price=%.2f "
                            "(source=BGeometrics capRealUSDs, "
                            "realized_cap=$%.2fB, mvrv=%.4f)",
                            realized_price,
                            float(urpd_data.get("realized_cap_usd", 0)) / 1e9,
                            float(urpd_data.get("mvrv_ratio", 0) or 0),
                        )
            except Exception as e:
                logger.debug("PrefetchedDataReader URPD failed: %s", e)

        # 1. 尝试本地缓存 (auto_data_fetcher 写入的真实数据)
        if realized_price is None or realized_price <= 0:
            cached = self.load_from_cache()
            if cached and cached > 0:
                realized_price = cached
                logger.info(
                    "Using cached realized_price=%.2f (auto_data_fetcher)",
                    realized_price,
                )

        # 2. 数据不可用 → 返回 has_data=False (绝不估算!)
        # 旧版: realized_price = market_price * 0.7 → MVRV 恒等于 1.4286 (NEUTRAL)
        # 新版: 真实数据不可用时不调整, 避免基于假数据做决策
        if realized_price is None or realized_price <= 0:
            logger.warning(
                "URPDGate: 真实 realized_price 数据不可用 (auto_data_fetcher 未运行或 BGeometrics API 失败), "
                "本轮跳过调整 (零降级策略, 不估算)"
            )
            result.has_data = False
            return result

        result.realized_price = realized_price
        result.mvrv_ratio = (
            market_price / realized_price if realized_price > 0 else 1.0
        )
        result.level = self._classify_level(result.mvrv_ratio)
        result.timestamp = time.time()
        result.has_data = True

        # 更新历史
        self._realized_price_history.append(realized_price)
        if len(self._realized_price_history) > 100:
            self._realized_price_history.pop(0)

        return result

    def _classify_level(self, mvrv: float) -> URPDLevel:
        """根据MVRV比率分类等级"""
        if mvrv < self.DEEP_DISCOUNT_MVRV:
            return URPDLevel.DEEP_DISCOUNT
        elif mvrv < self.DISCOUNT_MVRV:
            return URPDLevel.DISCOUNT
        elif mvrv > self.EXTREME_PREMIUM_MVRV:
            return URPDLevel.EXTREME_PREMIUM
        elif mvrv > self.PREMIUM_MVRV:
            return URPDLevel.PREMIUM
        else:
            return URPDLevel.NEUTRAL

    # ----------------------------------------------------------------
    # 分析与门控
    # ----------------------------------------------------------------

    def analyze(self, market_price: float) -> URPDAnalysisResult:
        """获取当前URPD状态

        Args:
            market_price: 当前市场价格

        Returns:
            URPDAnalysisResult
        """
        return self.get_realized_price(market_price)

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        symbol: str = "",
        market_price: float = 0.0,
    ) -> Dict[str, Any]:
        """对策略决策应用门控

        Args:
            decision: 策略决策字典 (含 action, quantity, price 等)
            symbol: 交易对
            market_price: 当前市场价格

        Returns:
            调整后的决策字典, 添加 urpd_gate_action, urpd_gate_multiplier,
            urpd_gate_reason, urpd_mvrv_ratio 字段
        """
        if not self.enabled:
            return decision

        self._total_calls += 1

        if market_price <= 0:
            self._no_data_count += 1
            return decision

        # 获取分析
        analysis = self.get_realized_price(market_price)
        self._last_analysis = analysis

        if not analysis.has_data:
            self._no_data_count += 1
            return decision

        # 应用门控规则
        action, multiplier, reason = self._evaluate_rules(analysis, decision)

        new_decision = dict(decision)

        # 写入门控信息
        new_decision["urpd_mvrv_ratio"] = analysis.mvrv_ratio
        new_decision["urpd_gate_action"] = action.value
        new_decision["urpd_gate_multiplier"] = multiplier
        new_decision["urpd_gate_reason"] = reason

        if action in (URPDAction.ALLOW, URPDAction.NEUTRAL):
            return new_decision

        # 调整仓位
        orig_qty = new_decision.get("quantity", 0.0)
        new_qty = orig_qty * multiplier
        if new_qty < 0:
            new_qty = 0.0
        new_decision["quantity"] = new_qty

        self._total_adjusted += 1
        if action == URPDAction.BOOST:
            self._boost_count += 1
        else:
            self._reduce_count += 1

        logger.debug(
            "URPDGate %s symbol=%s qty=%.6f→%.6f mvrv=%.4f level=%s reason=%s",
            action.value.upper(), symbol, orig_qty, new_qty,
            analysis.mvrv_ratio, analysis.level.value, reason,
        )

        return new_decision

    def _evaluate_rules(
        self,
        analysis: URPDAnalysisResult,
        decision: Dict[str, Any],
    ) -> Tuple[URPDAction, float, str]:
        """根据分析结果和决策方向，应用门控规则

        Returns:
            (action, multiplier, reason)
        """
        action_str = str(decision.get("action", "")).lower()
        is_long = "buy" in action_str or "long" in action_str
        is_short = "sell" in action_str or "short" in action_str

        level = analysis.level
        mvrv = analysis.mvrv_ratio

        # 规则1: 深度折价 (MVRV<0.5) → 做多 BOOST×1.25 / 做空 REDUCE×0.5
        if level == URPDLevel.DEEP_DISCOUNT:
            if is_long:
                return (
                    URPDAction.BOOST,
                    self.EXTREME_BOOST_MULTIPLIER,
                    f"deep_discount_long_boost=mvrv{mvrv:.4f}",
                )
            if is_short:
                return (
                    URPDAction.REDUCE,
                    self.EXTREME_REDUCE_MULTIPLIER,
                    f"deep_discount_short_reduce=mvrv{mvrv:.4f}",
                )

        # 规则2: 折价 (MVRV<0.8) → 做多 BOOST×1.15 / 做空 REDUCE×0.7
        if level == URPDLevel.DISCOUNT:
            if is_long:
                return (
                    URPDAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"discount_long_boost=mvrv{mvrv:.4f}",
                )
            if is_short:
                return (
                    URPDAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"discount_short_reduce=mvrv{mvrv:.4f}",
                )

        # 规则3: 极端溢价 (MVRV>2.5) → 做多 REDUCE×0.5 / 做空 BOOST×1.25
        if level == URPDLevel.EXTREME_PREMIUM:
            if is_long:
                return (
                    URPDAction.REDUCE,
                    self.EXTREME_REDUCE_MULTIPLIER,
                    f"extreme_premium_long_reduce=mvrv{mvrv:.4f}",
                )
            if is_short:
                return (
                    URPDAction.BOOST,
                    self.EXTREME_BOOST_MULTIPLIER,
                    f"extreme_premium_short_boost=mvrv{mvrv:.4f}",
                )

        # 规则4: 溢价 (MVRV>1.5) → 做多 REDUCE×0.7 / 做空 BOOST×1.15
        if level == URPDLevel.PREMIUM:
            if is_long:
                return (
                    URPDAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"premium_long_reduce=mvrv{mvrv:.4f}",
                )
            if is_short:
                return (
                    URPDAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"premium_short_boost=mvrv{mvrv:.4f}",
                )

        # 中性
        return URPDAction.ALLOW, 1.0, f"neutral=mvrv{mvrv:.4f}"

    # ----------------------------------------------------------------
    # 统计与回调
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
            "last_analysis": (
                self._last_analysis.to_dict() if self._last_analysis else None
            ),
            "realized_price_history_len": len(self._realized_price_history),
        }

    def round_complete(self) -> None:
        """周期完成回调"""
        logger.debug(
            "URPDGate round complete: calls=%d adjusted=%d boost=%d reduce=%d nodata=%d",
            self._total_calls, self._total_adjusted,
            self._boost_count, self._reduce_count, self._no_data_count,
        )

    def reset(self) -> None:
        """重置统计"""
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis = None
        self._realized_price_history.clear()
        logger.info("URPDGate stats reset")
