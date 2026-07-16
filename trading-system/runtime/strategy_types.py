"""Strategy base types module (Phase 7.13 extracted from strategy_registry.py)

Contains:
  - STRATEGY_INTERFACE_VERSION: interface version constant
  - Direction: unified direction enum
  - Signal: unified signal dataclass
  - MarketContext: unified market context dataclass
  - IStrategy: strategy abstract base class
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


STRATEGY_INTERFACE_VERSION: int = 1


# ============================================================================
# 统一数据结构（收敛异构 sensor/strategy 返回格式）
# ============================================================================

class Direction(IntEnum):
    """统一方向枚举（收敛 signal/direction/market_regime/action/decision 5 种命名）"""

    FLAT = 0  # 无方向/中性
    LONG = 1  # 多
    SHORT = -1  # 空

    @classmethod
    def from_any(cls, value: Any) -> "Direction":
        """从任意异构值解析方向"""
        if value is None:
            return cls.FLAT
        if isinstance(value, (int, float)):
            v = int(value)
            if v > 0:
                return cls.LONG
            if v < 0:
                return cls.SHORT
            return cls.FLAT
        if isinstance(value, str):
            v = value.lower().strip()
            if v in ("long", "bull", "bullish", "buy", "up", "1"):
                return cls.LONG
            if v in ("short", "bear", "bearish", "sell", "down", "-1"):
                return cls.SHORT
            if v in ("flat", "neutral", "hold", "none", "0", ""):
                return cls.FLAT
        return cls.FLAT


@dataclass(slots=True)
class Signal:
    """统一信号结构

    收敛 9 个 sensor 的 6 种强度字段 + 5 种方向字段 + 多种扩展数据
    """

    direction: Direction = Direction.FLAT
    confidence: float = 0.0  # 0-1，统一置信度
    strength: float = 0.0  # 0-1，原始强度（可能与 confidence 不同）
    probability: Optional[float] = None  # 0-1，可选的概率值（如 dl_forecast）
    sources: List[str] = field(default_factory=list)  # 信号来源标签
    metadata: Dict[str, Any] = field(default_factory=dict)  # 原始返回数据，扩展用
    timestamp: float = 0.0
    error: Optional[str] = None  # 加载/执行错误信息

    @property
    def is_actionable(self) -> bool:
        """是否可行动（有方向且有置信度）"""
        return self.direction != Direction.FLAT and self.confidence > 0.0 and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": int(self.direction),
            "direction_label": self.direction.name.lower(),
            "confidence": self.confidence,
            "strength": self.strength,
            "probability": self.probability,
            "sources": list(self.sources),
            "is_actionable": self.is_actionable,
            "error": self.error,
        }


@dataclass(slots=True)
class MarketContext:
    """统一市场上下文（收敛 4 种异构输入：kline dict / price+vol / return / List[AgentSignal]）

    所有字段可选，由调用方按需填充。策略适配器负责从中提取自己需要的字段。
    """

    symbol: str = ""
    timestamp: float = 0.0
    # OHLCV
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    # 衍生指标
    atr: float = 0.0
    period_return: float = 0.0
    price_change: float = 0.0
    volatility: float = 0.0
    # 历史数据
    price_history: List[float] = field(default_factory=list)
    kline_history: List[Dict[str, float]] = field(default_factory=list)
    # 外部依赖（如 spot_engine/perp_engine）
    deps: Dict[str, Any] = field(default_factory=dict)
    # 原始 data_packet（向后兼容）
    raw_packet: Dict[str, Any] = field(default_factory=dict)

    @property
    def kline(self) -> Dict[str, float]:
        """单根 K 线 dict（兼容 chanlun_pa/microstructure 接口）"""
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


# ============================================================================
# IStrategy 抽象基类（借鉴 nautilus on_* + qlib generate_*）
# ============================================================================

class IStrategy:
    """策略抽象基类

    所有策略（无论 35 个具体类还是 GeneDrivenStrategy）都通过此接口暴露。
    生命周期：on_start → update（每 tick） → on_stop
    持久化：on_save/on_load 支持进化系统跨代重启
    """

    INTERFACE_VERSION: int = STRATEGY_INTERFACE_VERSION

    def __init__(self, seed_name: str, gene: Any, key_params: Dict[str, Any]):
        self.seed_name = seed_name
        self.gene = gene
        self.key_params = dict(key_params)
        self._started: bool = False

    def on_start(self, deps: Optional[Dict[str, Any]] = None) -> None:
        """生命周期：启动（依赖注入在此发生）"""
        self._started = True

    def on_stop(self) -> None:
        """生命周期：停止"""
        self._started = False

    def update(self, ctx: MarketContext) -> Signal:
        """核心方法：每 tick 调用，返回统一 Signal

        子类必须 override 此方法。
        """
        raise NotImplementedError(f"{self.__class__.__name__} 未实现 update()")

    def on_save(self) -> Dict[str, Any]:
        """持久化状态（进化跨代重启用）"""
        return {"seed_name": self.seed_name, "key_params": self.key_params}

    def on_load(self, state: Dict[str, Any]) -> None:
        """从持久化状态恢复"""
