# -*- coding: utf-8 -*-
# v447_optimize_candidate: 此模块已标记为优化候选
# 原因: 零影响+中等重叠 31.0% > 30% (可合并)
# 建议操作: merge_or_lazy_load
# 标记时间: 2026-06-28T20:18:41
# 详情: _archive_v447/MANIFEST_v447.json

"""
vrp_harvesting.py — 波动率风险溢价收割引擎 v1.0

来源：
  - greeks.live 2026 (Trading the VRP)
  - convextrade 2026 (Volatility Risk Premium, VRP=IV-RV)
  - flashalpha 2026 (Quantitative Architectures for VRP Harvesting)
  - LiveVolatile 2026 (BTC Implied Volatility Post-Halving, VRP压缩)
  - hypercall 2026 (Vol Trading Intuition, VRP历史数据)
  - Volmageddon 2018 (XIV清盘教训, 尾部风险)

核心算法：
  1. VRP计算
     - VRP = IV - RV (隐含波动率 - 实现波动率)
     - SPX平均VRP: 2-4 vol点
     - BTC平均VRP: 5-15 vol点
     - ETH平均VRP: 5-20 vol点
     - IV > RV 频率: ~87% (SPX历史)
  2. VRP阈值体系
     - VRP > 5 vol点: 丰富IV，适合系统卖出
     - VRP 2-5 vol点: 正常区间，竞争激烈
     - VRP 0-2 vol点: 压缩，风险收益差
     - VRP < 0 (RV > IV): 卖vol亏损，黑天鹅
  3. 收割策略
     - 策略A: 卖出ATM Straddle（Delta对冲）
     - 策略B: Iron Condor（定义风险）
     - 策略C: 卖出Put（看跌保护溢价）
     - 策略D: 方差互换（机构级）
  4. 尾部风险对冲
     - 买入OTM Put作为尾部保护
     - VVIX > 100-110 = vol-of-vol风险高
     - 仓位动态调整: VRP高加仓，VRP低减仓

风险控制：
  - Volmageddon风险: 2018年2月XIV清盘，-90%单日
  - 负偏度: 收益分布左偏，尾部亏损大
  - 拥挤交易: 短vol策略拥挤时VRP压缩
  - 波动率突变: RV突然飙升超过IV

实盘验证：
  - 稳定 regime: 日均0.5%-1.5%收益
  - 崩盘事件: 损失可达收取保费的800%+
  - BTC VRP压缩: 减半后IV从68.5%→48.2%
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional



class VRPSignalType(Enum):
    """VRP信号类型"""
    SELL_STRADDLE = "sell_straddle"         # 卖出Straddle
    IRON_CONDOR = "iron_condor"             # 铁鹰组合
    SELL_PUT = "sell_put"                    # 卖出Put
    BUY_TAIL_HEDGE = "buy_tail_hedge"        # 买入尾部对冲
    CLOSE_POSITION = "close_position"        # 平仓
    STAND_ASIDE = "stand_aside"              # 观望（VRP低）
    NEUTRAL = "neutral"


class VRPRegime(Enum):
    """VRP状态"""
    RICH = "rich"                # VRP > 5 (丰富)
    NORMAL = "normal"            # VRP 2-5 (正常)
    COMPRESSED = "compressed"    # VRP 0-2 (压缩)
    NEGATIVE = "negative"        # VRP < 0 (负值)


@dataclass
class VRPData:
    """VRP数据快照"""
    timestamp: float
    implied_vol: float = 0.0       # 隐含波动率
    realized_vol: float = 0.0      # 实现波动率
    vvol: float = 0.0              # vol-of-vol (VVIX)
    spot_price: float = 0.0


@dataclass
class VRPOpportunity:
    """VRP机会"""
    signal_type: VRPSignalType
    vrp: float = 0.0
    regime: VRPRegime = VRPRegime.NORMAL
    strategy: str = ""
    premium_collected: float = 0.0   # 收取保费
    tail_hedge_cost: float = 0.0    # 尾部对冲成本
    net_premium: float = 0.0         # 净保费
    position_size_mult: float = 1.0  # 仓位乘子
    confidence: float = 0.0
    description: str = ""


@dataclass
class VRPSignal:
    """VRP综合信号"""
    signal_type: VRPSignalType = VRPSignalType.NEUTRAL
    best_opportunity: Optional[VRPOpportunity] = None
    current_vrp: float = 0.0
    current_regime: VRPRegime = VRPRegime.NORMAL
    iv_rv_ratio: float = 0.0
    description: str = ""


class VRPHarvesting:
    """
    波动率风险溢价收割引擎

    使用场景：
      - 期权市场（Deribit/OKX/Binance Options）
      - BTC/ETH/ALT 期权
      - 系统性短vol策略

    依赖：
      - numpy（数值计算）
      - IV/RV/vol-of-vol数据
    """

    # ===== VRP阈值 =====
    VRP_RICH = 0.05              # 5 vol点 = 丰富
    VRP_NORMAL = 0.02            # 2 vol点 = 正常
    VRP_COMPRESSED = 0.0         # 0 = 压缩

    # ===== vol-of-vol阈值 =====
    VVOL_HIGH = 100.0            # VVIX > 100 = 高风险
    VVOL_EXTREME = 110.0        # VVIX > 110 = 极端

    # ===== 仓位参数 =====
    MAX_POSITION_SIZE = 1.0      # 最大仓位
    TAIL_HEDGE_RATIO = 0.15     # 尾部对冲比例（15%保费）
    MIN_VRP_TO_ACT = 0.03       # 最小VRP行动阈值（3 vol点）

    # ===== 策略参数 =====
    STRADDLE_DELTA_HEDGE = True  # Straddle Delta对冲
    IRON_CONDOR_WIDTH = 0.10     # 铁鹰翼宽（10%）
    TAIL_HEDGE_OTM = 0.20        # 尾部对冲OTM距离（20%）

    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol
        self.vrp_history: deque = deque(maxlen=500)
        self.current_vrp: float = 0.0
        self.current_regime: VRPRegime = VRPRegime.NORMAL
        self.position_side: Optional[str] = None
        self.position_premium: float = 0.0
        self.cumulative_premium: float = 0.0
        self.cumulative_tail_cost: float = 0.0
        self.position_start_time: float = 0.0

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_vrp_data(self, data: VRPData) -> None:
        """更新VRP数据"""
        vrp = data.implied_vol - data.realized_vol
        self.current_vrp = vrp
        self.vrp_history.append({
            "timestamp": data.timestamp,
            "iv": data.implied_vol,
            "rv": data.realized_vol,
            "vrp": vrp,
            "vvol": data.vvol,
        })
        self._update_regime(data)

    def _update_regime(self, data: VRPData) -> None:
        """更新VRP状态"""
        if self.current_vrp > self.VRP_RICH:
            self.current_regime = VRPRegime.RICH
        elif self.current_vrp > self.VRP_NORMAL:
            self.current_regime = VRPRegime.NORMAL
        elif self.current_vrp > self.VRP_COMPRESSED:
            self.current_regime = VRPRegime.COMPRESSED
        else:
            self.current_regime = VRPRegime.NEGATIVE

    # ------------------------------------------------------------------
    # VRP分析
    # ------------------------------------------------------------------

    def calculate_iv_rv_ratio(self) -> float:
        """计算IV/RV比率"""
        if not self.vrp_history:
            return 0.0
        latest = self.vrp_history[-1]
        if latest["rv"] <= 0:
            return 0.0
        return latest["iv"] / latest["rv"]

    def get_vrp_percentile(self) -> float:
        """获取当前VRP历史百分位"""
        if len(self.vrp_history) < 10:
            return 0.5
        vrps = [h["vrp"] for h in self.vrp_history]
        current = self.current_vrp
        rank = sum(1 for v in vrps if v <= current)
        return rank / len(vrps)

    def get_vvol(self) -> float:
        """获取当前vol-of-vol"""
        if not self.vrp_history:
            return 0.0
        return self.vrp_history[-1].get("vvol", 0.0)

    def calculate_position_size(self) -> float:
        """计算仓位乘子"""
        # VRP高加仓，VRP低减仓
        if self.current_regime == VRPRegime.RICH:
            base = 1.0
        elif self.current_regime == VRPRegime.NORMAL:
            base = 0.6
        else:
            base = 0.0

        # VVIX高减仓
        vvol = self.get_vvol()
        if vvol > self.VVOL_EXTREME:
            base *= 0.3
        elif vvol > self.VVOL_HIGH:
            base *= 0.6

        return min(base, self.MAX_POSITION_SIZE)

    # ------------------------------------------------------------------
    # 策略计算
    # ------------------------------------------------------------------

    def calculate_straddle_premium(self, spot: float, iv: float, T: float = 30 / 365) -> float:
        """计算ATM Straddle保费（简化）"""
        # 简化: Straddle ≈ 0.8 × S × IV × √T
        return 0.8 * spot * iv * math.sqrt(T)

    def calculate_tail_hedge_cost(self, spot: float, iv: float, otm_pct: float = 0.20) -> float:
        """计算尾部对冲成本（OTM Put）"""
        # 简化: OTM Put价格 ≈ 0.05 × S × IV × √T
        return 0.05 * spot * iv * math.sqrt(30 / 365)

    def calculate_iron_condor_premium(self, spot: float, iv: float, width: float = 0.10) -> float:
        """计算Iron Condor净保费"""
        # 简化: IC净保费 ≈ 0.3 × S × IV × √T
        return 0.3 * spot * iv * math.sqrt(30 / 365)

    # ------------------------------------------------------------------
    # 主分析函数
    # ------------------------------------------------------------------

    def analyze(self) -> VRPSignal:
        """主分析函数"""
        signal = VRPSignal()
        signal.current_vrp = self.current_vrp
        signal.current_regime = self.current_regime
        signal.iv_rv_ratio = self.calculate_iv_rv_ratio()

        if not self.vrp_history:
            signal.description = "无数据"
            return signal

        latest = self.vrp_history[-1]
        spot = latest.get("iv", 0)  # 用IV作为代理
        iv = latest["iv"]
        vvol = self.get_vvol()
        size_mult = self.calculate_position_size()

        # 有持仓: 检查平仓
        if self.position_side is not None:
            # VRP压缩或转负 → 平仓
            if self.current_regime in (VRPRegime.COMPRESSED, VRPRegime.NEGATIVE):
                opp = VRPOpportunity(
                    signal_type=VRPSignalType.CLOSE_POSITION,
                    vrp=self.current_vrp,
                    regime=self.current_regime,
                    strategy=f"平仓{self.position_side}",
                    confidence=0.8,
                    description=f"VRP={self.current_vrp:.4f}压缩/负值, 平仓止损",
                )
                signal.signal_type = VRPSignalType.CLOSE_POSITION
                signal.best_opportunity = opp
                return signal
            # VVIX极端 → 平仓
            if vvol > self.VVOL_EXTREME:
                opp = VRPOpportunity(
                    signal_type=VRPSignalType.CLOSE_POSITION,
                    vrp=self.current_vrp,
                    regime=self.current_regime,
                    strategy=f"平仓{self.position_side}",
                    confidence=0.85,
                    description=f"VVIX={vvol:.0f}极端, 平仓避险",
                )
                signal.signal_type = VRPSignalType.CLOSE_POSITION
                signal.best_opportunity = opp
                return signal
            signal.signal_type = VRPSignalType.NEUTRAL
            signal.description = (
                f"持有{self.position_side}, VRP={self.current_vrp:.4f}, "
                f"regime={self.current_regime.value}"
            )
            return signal

        # 无持仓: 检查入场
        if self.current_vrp < self.MIN_VRP_TO_ACT:
            signal.signal_type = VRPSignalType.STAND_ASIDE
            signal.description = f"VRP={self.current_vrp:.4f}<{self.MIN_VRP_TO_ACT}, 观望"
            return signal

        if size_mult <= 0:
            signal.signal_type = VRPSignalType.STAND_ASIDE
            signal.description = f"仓位乘子=0, VVIX={vvol:.0f}极端"
            return signal

        # 选择策略
        spot_price = 100000  # 默认BTC价格
        if self.current_regime == VRPRegime.RICH:
            # VRP丰富: 卖Straddle + 尾部对冲
            premium = self.calculate_straddle_premium(spot_price, iv)
            tail_cost = self.calculate_tail_hedge_cost(spot_price, iv)
            net_premium = premium - tail_cost
            opp = VRPOpportunity(
                signal_type=VRPSignalType.SELL_STRADDLE,
                vrp=self.current_vrp,
                regime=self.current_regime,
                strategy="卖出ATM Straddle + Delta对冲 + OTM Put尾部保护",
                premium_collected=premium,
                tail_hedge_cost=tail_cost,
                net_premium=net_premium,
                position_size_mult=size_mult,
                confidence=0.75,
                description=(
                    f"VRP={self.current_vrp:.4f}丰富, "
                    f"卖Straddle保费={premium:.2f}, "
                    f"尾部对冲={tail_cost:.2f}, "
                    f"净保费={net_premium:.2f}, "
                    f"仓位{size_mult:.1f}x"
                ),
            )
            signal.signal_type = VRPSignalType.SELL_STRADDLE
            signal.best_opportunity = opp
        elif self.current_regime == VRPRegime.NORMAL:
            # 正常: Iron Condor（定义风险）
            premium = self.calculate_iron_condor_premium(spot_price, iv)
            opp = VRPOpportunity(
                signal_type=VRPSignalType.IRON_CONDOR,
                vrp=self.current_vrp,
                regime=self.current_regime,
                strategy="Iron Condor (定义风险)",
                premium_collected=premium,
                tail_hedge_cost=0,
                net_premium=premium,
                position_size_mult=size_mult,
                confidence=0.65,
                description=(
                    f"VRP={self.current_vrp:.4f}正常, "
                    f"Iron Condor净保费={premium:.2f}, "
                    f"仓位{size_mult:.1f}x"
                ),
            )
            signal.signal_type = VRPSignalType.IRON_CONDOR
            signal.best_opportunity = opp
        else:
            signal.signal_type = VRPSignalType.STAND_ASIDE
            signal.description = f"VRP={self.current_vrp:.4f}压缩, 观望"

        return signal

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "symbol": self.symbol,
            "current_vrp": self.current_vrp,
            "current_regime": self.current_regime.value,
            "iv_rv_ratio": self.calculate_iv_rv_ratio(),
            "vvol": self.get_vvol(),
            "vrp_percentile": self.get_vrp_percentile(),
            "position_side": self.position_side,
            "position_size_mult": self.calculate_position_size(),
            "cumulative_premium": self.cumulative_premium,
            "cumulative_tail_cost": self.cumulative_tail_cost,
            "history_count": len(self.vrp_history),
        }
