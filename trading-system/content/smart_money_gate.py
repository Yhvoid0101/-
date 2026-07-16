#!/usr/bin/env python3
"""聪明钱门控模块 — Layer 1 市场数据层 P3

基于 SmartMoneyFootprint (smart_money_footprint.py) 的聪明钱信号进行门控：
  - 聪明钱看涨 (sentiment > 0.3) + 做多 → BOOST
  - 聪明钱看涨 + 做空 → REDUCE
  - 聪明钱看跌 (sentiment < -0.3) + 做空 → BOOST
  - 聪明钱看跌 + 做多 → REDUCE
  - 中性 → ALLOW

支持两种数据源:
  1. 真实链上数据: 通过 add_transaction() 喂入
  2. 合成数据: 基于tick数据生成模拟聪明钱交易 (用于沙盘测试)

用法:
    from sandbox_trading.smart_money_gate import SmartMoneyGate
    gate = SmartMoneyGate(enabled=True)
    decision = gate.apply_to_decision(decision, symbol="BTC-USDT", price=40000, volume=1000)
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    try:
        from .prefetched_data_reader import PrefetchedDataReader
    except ImportError:
        from prefetched_data_reader import PrefetchedDataReader
except ImportError:
    PrefetchedDataReader = None

logger = logging.getLogger("hermes.smart_money_gate")


# ============================================================================
# 枚举与数据结构
# ============================================================================

class SMGateAction(Enum):
    """聪明钱门控动作"""
    ALLOW = "allow"
    BOOST = "boost"
    REDUCE = "reduce"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class SMAnalysisResult:
    """聪明钱分析结果"""
    sentiment: float = 0.0           # -1 (看跌) 到 +1 (看涨)
    exchange_flow_signal: float = 0.0
    signal_type: str = "NEUTRAL"
    smart_money_count: int = 0
    has_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment": self.sentiment,
            "exchange_flow_signal": self.exchange_flow_signal,
            "signal_type": self.signal_type,
            "smart_money_count": self.smart_money_count,
            "has_data": self.has_data,
        }


# ============================================================================
# 聪明钱门控主类
# ============================================================================

class SmartMoneyGate:
    """聪明钱门控

    工作流程:
      1. 接收tick数据，生成合成聪明钱交易 (或接收真实链上数据)
      2. 调用SmartMoneyFootprint.analyze()获取信号
      3. 基于sentiment进行门控

    门控规则:
      1. sentiment >= STRONG_BULLISH (0.3) + 做多 → BOOST ×1.15
      2. sentiment >= STRONG_BULLISH + 做空 → REDUCE ×0.7
      3. sentiment <= STRONG_BEARISH (-0.3) + 做空 → BOOST ×1.15
      4. sentiment <= STRONG_BEARISH + 做多 → REDUCE ×0.7
      5. |sentiment| < 0.3 → ALLOW
    """

    # 情绪阈值
    STRONG_BULLISH = 0.3    # 聪明钱看涨阈值
    STRONG_BEARISH = -0.3   # 聪明钱看跌阈值
    EXTREME_BULLISH = 0.6   # 极端看涨
    EXTREME_BEARISH = -0.6  # 极端看跌

    # 调整系数
    BOOST_MULTIPLIER = 1.15
    REDUCE_MULTIPLIER = 0.7
    EXTREME_BOOST_MULTIPLIER = 1.25
    EXTREME_REDUCE_MULTIPLIER = 0.5

    def __init__(self, enabled: bool = True, symbol: str = "BTC", allow_synthetic: bool = False):
        """初始化聪明钱门控

        Args:
            enabled: 是否启用
            symbol: 默认交易标的
        """
        self.enabled = enabled
        self.symbol = symbol
        self.allow_synthetic = allow_synthetic

        # 尝试导入SmartMoneyFootprint
        self._sm_footprint = None
        try:
            try:
                from .smart_money_footprint import SmartMoneyFootprint
            except ImportError:
                from smart_money_footprint import SmartMoneyFootprint
            self._sm_footprint = SmartMoneyFootprint(symbol=symbol)
            # 添加合成聪明钱钱包 (profit_score >= 0.6 才被识别为聪明钱)
            try:
                from .smart_money_footprint import SmartMoneyWallet, WalletType
            except ImportError:
                from smart_money_footprint import SmartMoneyWallet, WalletType
            for i in range(10):
                _w = SmartMoneyWallet(
                    address=f"synthetic_whale_{i}",
                    wallet_type=WalletType.WHALE,
                    profit_score=0.75 + (i % 3) * 0.1,  # 0.75-0.95
                    win_rate=0.6 + (i % 4) * 0.05,
                    total_volume_usd=500_000_000,
                    last_active=time.time(),
                    tracked=True,
                )
                self._sm_footprint.tracked_wallets[_w.address] = _w
            self._sm_available = True
        except (ImportError, Exception) as e:
            logger.warning("SmartMoneyFootprint not available: %s", e)
            self._sm_available = False

        # 统计
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis: Optional[SMAnalysisResult] = None

        # 合成数据生成器状态
        self._rng = random.Random(42)
        self._price_history: List[float] = []
        self._synthetic_tx_count = 0
        self._real_tx_count = 0

        logger.info(
            "SmartMoneyGate initialized: enabled=%s sm_available=%s "
            "strong_bullish=%.2f strong_bearish=%.2f",
            self.enabled, self._sm_available,
            self.STRONG_BULLISH, self.STRONG_BEARISH,
        )

    # ----------------------------------------------------------------
    # 数据输入
    # ----------------------------------------------------------------

    def update_market_data(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: float = None,
    ):
        """更新市场数据，优先加载真实鲸鱼交易

        Args:
            price: 当前价格
            volume: 当前成交量
            timestamp: 时间戳
        """
        if not self.enabled or price <= 0:
            return

        if timestamp is None:
            timestamp = time.time()

        # 更新价格历史
        self._price_history.append(price)
        if len(self._price_history) > 50:
            self._price_history.pop(0)

        # 优先加载真实鲸鱼交易
        real_count = self._load_real_transactions()

        # 仅当无真实数据时才生成合成数据 (降级路径)
        if real_count == 0 and self.allow_synthetic and self._sm_available and self._sm_footprint is not None:
            self._generate_synthetic_transactions(price, volume, timestamp)

    def _load_real_transactions(self) -> int:
        """从预取数据加载真实鲸鱼交易

        Returns:
            加载的交易数量, 0 表示无数据
        """
        if PrefetchedDataReader is None or self._sm_footprint is None:
            return 0
        try:
            data = PrefetchedDataReader.read_smart_money(self.symbol)
            if data is None:
                return 0
            transactions = data.get("transactions", [])
            if not transactions:
                return 0

            try:
                try:
                    from .smart_money_footprint import WhaleTransaction, WalletType
                except ImportError:
                    from smart_money_footprint import WhaleTransaction, WalletType
            except ImportError:
                return 0

            count = 0
            for tx_data in transactions:
                try:
                    direction = tx_data.get("direction", "to_exchange")
                    raw_ts = float(tx_data.get("timestamp", 0))
                    # Gate.io returns ms epoch; convert to seconds if needed
                    if raw_ts > 1e12:
                        raw_ts = raw_ts / 1000.0
                    elif raw_ts == 0:
                        raw_ts = time.time()
                    tx = WhaleTransaction(
                        timestamp=raw_ts,
                        wallet_address=tx_data.get("wallet_address", "gateio_whale"),
                        wallet_type=WalletType.WHALE,
                        token=tx_data.get("token", self.symbol),
                        amount_usd=float(tx_data.get("amount_usd", 0)),
                        direction=direction,
                        exchange=tx_data.get("exchange", "gateio"),
                    )
                    self._sm_footprint.add_transaction(tx)
                    count += 1
                    self._real_tx_count += 1
                except (ValueError, TypeError, KeyError) as e:
                    logger.debug("Failed to parse real whale tx: %s", e)
                    continue
            if count > 0:
                logger.debug("Loaded %d real whale transactions for %s", count, self.symbol)
            return count
        except Exception as e:
            logger.debug("Real whale data load failed: %s", e)
            return 0

    def _generate_synthetic_transactions(
        self,
        price: float,
        volume: float,
        timestamp: float,
    ):
        """基于市场数据生成合成聪明钱交易"""
        try:
            try:
                from .smart_money_footprint import WhaleTransaction, WalletType
            except ImportError:
                from smart_money_footprint import WhaleTransaction, WalletType

            # 基于价格变化方向生成交易
            if len(self._price_history) < 2:
                return

            prev_price = self._price_history[-2]
            change_pct = (price - prev_price) / prev_price if prev_price > 0 else 0

            # 随机生成1-3笔交易
            n_tx = self._rng.randint(2, 5)
            for _ in range(n_tx):
                # 基于价格变化决定交易方向
                if change_pct > 0.001:
                    # 价格上涨 → 聪明钱买入 (85%概率)
                    tx_type = "buy" if self._rng.random() > 0.15 else "sell"
                elif change_pct < -0.001:
                    # 价格下跌 → 聪明钱卖出 (85%概率)
                    tx_type = "sell" if self._rng.random() > 0.15 else "buy"
                else:
                    # 中性: 60% buy 创造轻微看涨偏向
                    tx_type = "buy" if self._rng.random() > 0.4 else "sell"

                # 生成交易量 (基于tick volume)
                tx_value = self._rng.uniform(200_000_000, 800_000_000)  # 200M-800M 强信号

                # tx_type: "buy" → direction="from_exchange" (从交易所流出=看涨)
                # tx_type: "sell" → direction="to_exchange" (流入交易所=看跌)
                _direction = "from_exchange" if tx_type == "buy" else "to_exchange"
                _wallet = f"synthetic_whale_{self._synthetic_tx_count % 10}"
                tx = WhaleTransaction(
                    timestamp=timestamp,
                    wallet_address=_wallet,
                    wallet_type=WalletType.WHALE,
                    token=self.symbol,
                    amount_usd=tx_value,
                    direction=_direction,
                    exchange="synthetic_exchange",
                )

                self._sm_footprint.add_transaction(tx)
                self._synthetic_tx_count += 1

        except Exception as e:
            logger.debug("Synthetic transaction generation failed: %s", e)

    # ----------------------------------------------------------------
    # 分析与门控
    # ----------------------------------------------------------------

    def analyze(self) -> SMAnalysisResult:
        """分析聪明钱信号"""
        result = SMAnalysisResult()

        if not self.enabled or not self._sm_available or self._sm_footprint is None:
            return result

        try:
            signal = self._sm_footprint.analyze()
            result.sentiment = signal.smart_money_sentiment
            result.exchange_flow_signal = signal.exchange_flow_signal
            result.signal_type = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
            result.smart_money_count = self._sm_footprint.count_active_smart_money()
            result.has_data = True
        except Exception as e:
            logger.debug("SmartMoney analyze failed: %s", e)

        return result

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        symbol: str = "",
        price: float = 0.0,
        volume: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """对策略决策应用门控

        Args:
            decision: 策略决策字典
            symbol: 交易对
            price: 当前价格 (用于更新数据)
            volume: 当前成交量

        Returns:
            调整后的决策字典
        """
        if not self.enabled:
            return decision

        self._total_calls += 1

        # 更新市场数据
        if price > 0:
            self.update_market_data(price, volume)

        # 分析聪明钱信号
        analysis = self.analyze()
        self._last_analysis = analysis

        if not analysis.has_data:
            self._no_data_count += 1
            return decision

        # 应用门控规则
        action, multiplier, reason = self._evaluate_rules(analysis, decision)

        if action == SMGateAction.ALLOW or action == SMGateAction.NEUTRAL:
            return decision

        # 调整仓位
        new_decision = dict(decision)
        orig_qty = new_decision.get("quantity", 0.0)
        new_qty = orig_qty * multiplier
        if new_qty < 0:
            new_qty = 0.0
        new_decision["quantity"] = new_qty
        new_decision["sm_gate_action"] = action.value
        new_decision["sm_gate_multiplier"] = multiplier
        new_decision["sm_gate_reason"] = reason
        new_decision["sm_sentiment"] = analysis.sentiment
        new_decision["sm_signal_type"] = analysis.signal_type

        self._total_adjusted += 1
        if action == SMGateAction.BOOST:
            self._boost_count += 1
        else:
            self._reduce_count += 1

        logger.debug(
            "SMGate %s symbol=%s qty=%.6f→%.6f sentiment=%.3f signal=%s reason=%s",
            action.value.upper(), symbol, orig_qty, new_qty,
            analysis.sentiment, analysis.signal_type, reason,
        )

        return new_decision

    def _evaluate_rules(
        self,
        analysis: SMAnalysisResult,
        decision: Dict[str, Any],
    ) -> tuple:
        """根据分析结果和决策方向，应用门控规则

        Returns:
            (action, multiplier, reason)
        """
        action_str = str(decision.get("action", "")).lower()
        is_long = "buy" in action_str or "long" in action_str
        is_short = "sell" in action_str or "short" in action_str

        sentiment = analysis.sentiment

        # 极端看涨
        if sentiment >= self.EXTREME_BULLISH:
            if is_long:
                return (
                    SMGateAction.BOOST,
                    self.EXTREME_BOOST_MULTIPLIER,
                    f"extreme_bullish_long_boost={sentiment:.3f}",
                )
            if is_short:
                return (
                    SMGateAction.REDUCE,
                    self.EXTREME_REDUCE_MULTIPLIER,
                    f"extreme_bullish_short_reduce={sentiment:.3f}",
                )

        # 强看涨
        if sentiment >= self.STRONG_BULLISH:
            if is_long:
                return (
                    SMGateAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"bullish_long_boost={sentiment:.3f}",
                )
            if is_short:
                return (
                    SMGateAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"bullish_short_reduce={sentiment:.3f}",
                )

        # 极端看跌
        if sentiment <= self.EXTREME_BEARISH:
            if is_short:
                return (
                    SMGateAction.BOOST,
                    self.EXTREME_BOOST_MULTIPLIER,
                    f"extreme_bearish_short_boost={sentiment:.3f}",
                )
            if is_long:
                return (
                    SMGateAction.REDUCE,
                    self.EXTREME_REDUCE_MULTIPLIER,
                    f"extreme_bearish_long_reduce={sentiment:.3f}",
                )

        # 强看跌
        if sentiment <= self.STRONG_BEARISH:
            if is_short:
                return (
                    SMGateAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"bearish_short_boost={sentiment:.3f}",
                )
            if is_long:
                return (
                    SMGateAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"bearish_long_reduce={sentiment:.3f}",
                )

        return SMGateAction.ALLOW, 1.0, "neutral_sentiment"

    # ----------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "sm_available": self._sm_available,
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
            "synthetic_tx_count": self._synthetic_tx_count,
            "real_tx_count": self._real_tx_count,
            "last_analysis": self._last_analysis.to_dict() if self._last_analysis else None,
        }

    def reset(self):
        """重置统计"""
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis = None
        self._price_history.clear()
        self._synthetic_tx_count = 0
        self._real_tx_count = 0
        logger.info("SmartMoneyGate stats reset")
