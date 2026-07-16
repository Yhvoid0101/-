"""MarketDataEnvelope — 第1层市场数据层输出契约。

铁律：禁止合成行情进入训练/评分/门控/验收。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ContractBase, ContractStatus, DataLineage, make_blocked


@dataclass
class DataQuality:
    """数据质量指标。"""
    completeness: float = 1.0
    latency_ms: float = 0.0
    gap_count: int = 0
    anomaly_count: int = 0
    cross_source_diff: float = 0.0
    is_stale: bool = False


@dataclass
class MarketDataEnvelope(ContractBase):
    """第1层输出：市场数据信封。

    所有字段可追溯到 Gate.io 真实来源；SYNTHETIC 数据必须输出 BLOCKED。
    """
    symbol: str = ""
    interval: str = ""
    klines: List[Dict[str, Any]] = field(default_factory=list)
    ticker: Optional[Dict[str, Any]] = None
    order_book: Optional[Dict[str, Any]] = None
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    long_short_ratio: Optional[float] = None
    quality: DataQuality = field(default_factory=DataQuality)
    exchange: str = "gateio"
    is_real_data: bool = True

    def __post_init__(self):
        if not self.is_real_data:
            self.status = ContractStatus.BLOCKED
            self.error_code = "SYNTHETIC_DATA"
            self.error_message = f"合成数据禁止进入决策链: {self.symbol}"
        if self.exchange.lower() in ("binance", "okx", "bybit"):
            self.status = ContractStatus.BLOCKED
            self.error_code = "FORBIDDEN_EXCHANGE"
            self.error_message = f"禁止交易所: {self.exchange}，唯一允许为 gateio"


def blocked_synthetic(symbol: str, reason: str = "", correlation_id: str = "") -> MarketDataEnvelope:
    """生成 SYNTHETIC BLOCKED 信封。"""
    env = MarketDataEnvelope(symbol=symbol, is_real_data=False)
    return make_blocked(env, "SYNTHETIC_DATA",
                        f"合成数据禁止进入决策链: {symbol} - {reason}", correlation_id)


def blocked_expired(symbol: str, lineage: DataLineage, correlation_id: str = "") -> MarketDataEnvelope:
    """生成过期数据 BLOCKED 信封。"""
    env = MarketDataEnvelope(symbol=symbol, lineage=lineage)
    return make_blocked(env, "DATA_EXPIRED",
                        f"数据已过期: {symbol} - source={lineage.source}", correlation_id)
