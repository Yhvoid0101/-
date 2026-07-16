#!/usr/bin/env python3
"""GEX (Gamma Exposure) 计算器 — Layer 8 质变数据模块

Gamma Exposure 衡量期权做市商为了对冲期权头寸需要在现货市场买卖的数量。
当期权即将到期时，做市商的对冲行为会产生"磁吸效应"——价格被强行拉向
某个特定水平（期权最大痛点）。

GEX 计算公式（标准做市商对冲模型）:
  对于每个期权合约:
    Gamma = 期权希腊字母 Gamma（来自 Black-Scholes）
    Dealer_Gamma_call = +Gamma × OI_call × Spot² × 0.01 × 100  (做市商卖出call)
    Dealer_Gamma_put  = -Gamma × OI_put  × Spot² × 0.01 × 100  (做市商卖出put)

  其中 0.01 = 1% 价格变动，100 = 合约乘数

  正 GEX → 做市商需要在价格上涨时买入 → 磁吸效应向上
  负 GEX → 做市商需要在价格上涨时卖出 → 磁吸效应向下

磁吸价位识别:
  - max_pain: 最大痛点（期权持仓最集中的行权价，到期时做市商对冲最激烈）
  - positive_magnet: 最大正GEX行权价（价格被向上拉扯）
  - negative_magnet: 最大负GEX行权价（价格被向下拉扯）
  - total_gex: 全市场总GEX（正数=整体磁吸向上，负数=整体磁吸向下）

用法:
    from sandbox_trading.gex_calculator import GEXCalculator
    calc = GEXCalculator()
    result = calc.calculate(options_chain, spot_price, timestamp)
    # result 包含: total_gex, max_pain, positive_magnet, negative_magnet, per_strike_gex
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


logger = logging.getLogger("hermes.gex")


# ============================================================================
# Black-Scholes 期权定价模型 (用于计算 Gamma)
# ============================================================================

def _norm_pdf(x: float) -> float:
    """标准正态分布概率密度函数"""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    """标准正态分布累积分布函数 (Abramowitz & Stegun 近似)"""
    # 使用误差函数近似
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_gamma(spot: float, strike: float, ttm: float, iv: float,
             r: float = 0.03, q: float = 0.0, is_call: bool = True) -> float:
    """Black-Scholes Gamma 计算

    Gamma 是 spot 的二阶导数，对所有期权类型(call/put)相同。

    Args:
        spot: 标的资产现价
        strike: 行权价
        ttm: 距到期时间 (年)
        iv: 隐含波动率 (年化, 0-1)
        r: 无风险利率 (默认3%)
        q: 分红率 (默认0, 加密货币无分红)
        is_call: 是否看涨期权 (Gamma相同, 参数仅为一致性)

    Returns:
        Gamma 值 (>=0)
    """
    if spot <= 0 or strike <= 0 or ttm <= 0 or iv <= 0:
        return 0.0

    try:
        d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * ttm) / (iv * math.sqrt(ttm))
        # Gamma = exp(-q*ttm) * N'(d1) / (spot * iv * sqrt(ttm))
        gamma = math.exp(-q * ttm) * _norm_pdf(d1) / (spot * iv * math.sqrt(ttm))
        return max(0.0, gamma)
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def bs_delta(spot: float, strike: float, ttm: float, iv: float,
             r: float = 0.03, q: float = 0.0, is_call: bool = True) -> float:
    """Black-Scholes Delta 计算

    Args:
        spot: 标的资产现价
        strike: 行权价
        ttm: 距到期时间 (年)
        iv: 隐含波动率 (年化, 0-1)
        r: 无风险利率 (默认3%)
        q: 分红率 (默认0)
        is_call: 是否看涨期权

    Returns:
        Delta 值
    """
    if spot <= 0 or strike <= 0 or ttm <= 0 or iv <= 0:
        return 0.0

    try:
        d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * ttm) / (iv * math.sqrt(ttm))
        if is_call:
            delta = math.exp(-q * ttm) * _norm_cdf(d1)
        else:
            delta = -math.exp(-q * ttm) * _norm_cdf(-d1)
        return delta
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


# ============================================================================
# GEX 计算结果数据结构
# ============================================================================

@dataclass(slots=True)
class GEXResult:
    """GEX 计算结果

    所有 GEX 单位均为 "美元/1%价格变动"（即价格变动1%时做市商需要对冲的美元金额）
    """
    symbol: str = ""
    timestamp: float = 0.0
    spot_price: float = 0.0

    # 全市场总 GEX (正=磁吸向上, 负=磁吸向下)
    total_gex: float = 0.0
    # 总call gamma (做市商卖出call的正gamma)
    total_call_gex: float = 0.0
    # 总put gamma (做市商卖出put的负gamma)
    total_put_gex: float = 0.0

    # 磁吸价位
    max_pain: float = 0.0           # 最大痛点 (做市商对冲最激烈的行权价)
    positive_magnet: float = 0.0    # 最大正GEX行权价 (价格被向上拉)
    negative_magnet: float = 0.0    # 最大负GEX行权价 (价格被向下拉)
    max_positive_gex: float = 0.0   # positive_magnet 处的 GEX 值
    max_negative_gex: float = 0.0   # negative_magnet 处的 GEX 值

    # 0-DTE 风险 (最近到期日的 GEX 占比)
    zero_dte_gex_ratio: float = 0.0

    # 每个行权价的 GEX (用于热力图)
    per_strike_gex: Dict[str, float] = field(default_factory=dict)
    # 每个到期日的 GEX
    per_expiry_gex: Dict[str, float] = field(default_factory=dict)

    # 原始统计
    n_options: int = 0
    n_strikes: int = 0
    n_expiries: int = 0

    # 计算时间
    calc_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于序列化)"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "spot_price": self.spot_price,
            "total_gex": self.total_gex,
            "total_call_gex": self.total_call_gex,
            "total_put_gex": self.total_put_gex,
            "max_pain": self.max_pain,
            "positive_magnet": self.positive_magnet,
            "negative_magnet": self.negative_magnet,
            "max_positive_gex": self.max_positive_gex,
            "max_negative_gex": self.max_negative_gex,
            "zero_dte_gex_ratio": self.zero_dte_gex_ratio,
            "per_strike_gex": dict(self.per_strike_gex),
            "per_expiry_gex": dict(self.per_expiry_gex),
            "n_options": self.n_options,
            "n_strikes": self.n_strikes,
            "n_expiries": self.n_expiries,
            "calc_time_ms": self.calc_time_ms,
        }


# ============================================================================
# GEX 计算器
# ============================================================================

class GEXCalculator:
    """GEX (Gamma Exposure) 计算器

    从期权链数据计算做市商的 Gamma Exposure，识别磁吸价位。
    """

    # 合约乘数 (BTC=1, ETH=1, 单位为币; 美元口径需要乘以spot)
    CONTRACT_MULTIPLIER = 1.0
    # 1% 价格变动
    PRICE_MOVE_PCT = 0.01

    def __init__(self, contract_multiplier: float = 1.0):
        """
        Args:
            contract_multiplier: 合约乘数 (BTC=1, ETH=1, 股票期权=100)
        """
        self.contract_multiplier = contract_multiplier

    def calculate(
        self,
        options: List[Dict[str, Any]],
        spot_price: float,
        timestamp: Optional[float] = None,
        symbol: str = "",
        risk_free_rate: float = 0.03,
    ) -> GEXResult:
        """计算 GEX

        Args:
            options: 期权链列表，每个期权需包含:
                - strike: 行权价
                - ttm: 距到期时间 (年) 或 expiry_ts (Unix时间戳)
                - iv: 隐含波动率 (0-1)
                - oi: 未平仓合约数
                - type: "call" 或 "put" (或 is_call: bool)
            spot_price: 标的资产现价
            timestamp: 计算时间戳 (默认当前时间)
            symbol: 交易对符号
            risk_free_rate: 无风险利率

        Returns:
            GEXResult 计算结果
        """
        import time as _time
        t0 = _time.time()

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()

        result = GEXResult(
            symbol=symbol,
            timestamp=timestamp,
            spot_price=spot_price,
        )

        if not options or spot_price <= 0:
            return result

        # 按行权价聚合
        per_strike: Dict[float, float] = {}
        per_expiry: Dict[float, float] = {}
        # 最大痛点: 持仓最集中的行权价
        strike_oi_sum: Dict[float, float] = {}

        total_call_gex = 0.0
        total_put_gex = 0.0
        zero_dte_gex = 0.0
        total_gex_all = 0.0

        min_ttm = float('inf')

        for opt in options:
            try:
                strike = float(opt.get("strike", 0))
                iv = float(opt.get("iv", opt.get("mark_iv", 0)))
                oi = float(opt.get("oi", opt.get("open_interest", 0)))

                # ttm 计算
                ttm = float(opt.get("ttm", 0))
                if ttm <= 0:
                    expiry_ts = float(opt.get("expiry_ts", opt.get("expiration_timestamp", 0)))
                    if expiry_ts > 0:
                        ttm = max(0.0, (expiry_ts - timestamp) / (365.25 * 86400))

                if strike <= 0 or iv <= 0 or oi <= 0 or ttm <= 0:
                    continue

                # 期权类型
                opt_type = str(opt.get("type", opt.get("option_type", ""))).lower()
                is_call = opt_type in ("call", "c", "call_option")

                # 计算 Gamma
                gamma = bs_gamma(
                    spot=spot_price, strike=strike, ttm=ttm, iv=iv,
                    r=risk_free_rate, q=0.0, is_call=is_call,
                )

                if gamma <= 0:
                    continue

                # GEX 计算 (美元/1%价格变动)
                # 做市商通常卖出期权 (正gamma暴露)
                # call: dealer_gex = +gamma × oi × spot² × 0.01 × multiplier
                # put:  dealer_gex = -gamma × oi × spot² × 0.01 × multiplier
                gex_value = (
                    gamma * oi * (spot_price ** 2) *
                    self.PRICE_MOVE_PCT * self.contract_multiplier
                )

                if is_call:
                    total_call_gex += gex_value
                    per_strike_gex = gex_value
                else:
                    total_put_gex -= gex_value
                    per_strike_gex = -gex_value

                # 聚合到行权价
                per_strike[strike] = per_strike.get(strike, 0.0) + per_strike_gex
                strike_oi_sum[strike] = strike_oi_sum.get(strike, 0.0) + oi

                # 聚合到到期日
                per_expiry[ttm] = per_expiry.get(ttm, 0.0) + per_strike_gex

                total_gex_all += per_strike_gex

                # 0-DTE 检测
                if ttm < min_ttm:
                    min_ttm = ttm
                if ttm < 1.0 / 365.0:  # 小于1天
                    zero_dte_gex += per_strike_gex

            except (ValueError, TypeError, KeyError) as e:
                logger.debug("GEX option skipped: %s", e)
                continue

        # 填充结果
        result.total_gex = total_gex_all
        result.total_call_gex = total_call_gex
        result.total_put_gex = total_put_gex
        result.n_options = len(options)
        result.n_strikes = len(per_strike)
        result.n_expiries = len(per_expiry)

        # 0-DTE 比例
        if abs(total_gex_all) > 0:
            result.zero_dte_gex_ratio = zero_dte_gex / total_gex_all

        # 磁吸价位识别
        if per_strike:
            # 最大正GEX (磁吸向上)
            pos_strike = max(per_strike.items(), key=lambda x: x[1])
            if pos_strike[1] > 0:
                result.positive_magnet = pos_strike[0]
                result.max_positive_gex = pos_strike[1]

            # 最大负GEX (磁吸向下)
            neg_strike = min(per_strike.items(), key=lambda x: x[1])
            if neg_strike[1] < 0:
                result.negative_magnet = neg_strike[0]
                result.max_negative_gex = neg_strike[1]

            # 最大痛点: 持仓最集中的行权价
            if strike_oi_sum:
                result.max_pain = max(strike_oi_sum.items(), key=lambda x: x[1])[0]

        # 转换 key 为字符串 (JSON 序列化)
        result.per_strike_gex = {f"{k:.2f}": v for k, v in per_strike.items()}
        result.per_expiry_gex = {f"{k:.4f}": v for k, v in per_expiry.items()}

        result.calc_time_ms = (_time.time() - t0) * 1000

        logger.info(
            "GEX calc: symbol=%s spot=%.2f total_gex=%.2f call=%.2f put=%.2f "
            "max_pain=%.2f pos_magnet=%.2f neg_magnet=%.2f n_opts=%d calc=%.1fms",
            symbol, spot_price, total_gex_all, total_call_gex, total_put_gex,
            result.max_pain, result.positive_magnet, result.negative_magnet,
            result.n_options, result.calc_time_ms,
        )

        return result

    def identify_magnetism(
        self,
        result: GEXResult,
        current_price: float,
        threshold_pct: float = 0.05,
    ) -> Dict[str, Any]:
        """识别当前价格附近的磁吸效应

        Args:
            result: GEX 计算结果
            current_price: 当前标的价格
            threshold_pct: 磁吸价位识别阈值 (距离当前价格5%以内视为有效磁吸)

        Returns:
            dict: 磁吸分析结果
                - direction: "up" / "down" / "neutral"
                - strength: "strong" / "medium" / "weak"
                - target_price: 磁吸目标价
                - distance_pct: 距离百分比
                - confidence: 置信度 (0-1)
        """
        if not result or current_price <= 0:
            return {
                "direction": "neutral", "strength": "weak",
                "target_price": 0.0, "distance_pct": 0.0, "confidence": 0.0,
            }

        # 选择主要磁吸方向 (基于 total_gex 正负)
        if result.total_gex > 0:
            # 整体正GEX → 磁吸向上
            target = result.positive_magnet
            gex_value = result.max_positive_gex
            direction = "up"
        elif result.total_gex < 0:
            # 整体负GEX → 磁吸向下
            target = result.negative_magnet
            gex_value = result.max_negative_gex
            direction = "down"
        else:
            return {
                "direction": "neutral", "strength": "weak",
                "target_price": 0.0, "distance_pct": 0.0, "confidence": 0.0,
            }

        if target <= 0:
            return {
                "direction": "neutral", "strength": "weak",
                "target_price": 0.0, "distance_pct": 0.0, "confidence": 0.0,
            }

        # 距离百分比
        distance_pct = abs(target - current_price) / current_price

        # 距离过远时不视为有效磁吸
        if distance_pct > threshold_pct * 2:
            return {
                "direction": "neutral", "strength": "weak",
                "target_price": target, "distance_pct": distance_pct,
                "confidence": 0.0,
            }

        # 强度评估
        abs_gex = abs(gex_value)
        if abs_gex > 1e8:  # 1亿美元以上
            strength = "strong"
        elif abs_gex > 1e7:
            strength = "medium"
        else:
            strength = "weak"

        # 置信度: 距离越近, GEX越大, 置信度越高
        distance_factor = max(0.0, 1.0 - distance_pct / threshold_pct)
        gex_factor = min(1.0, abs_gex / 1e9)
        confidence = distance_factor * gex_factor

        return {
            "direction": direction,
            "strength": strength,
            "target_price": target,
            "distance_pct": distance_pct,
            "confidence": confidence,
            "total_gex": result.total_gex,
        }


# ============================================================================
# 便捷函数
# ============================================================================

def calculate_gex_from_options(
    options: List[Dict[str, Any]],
    spot_price: float,
    symbol: str = "",
    timestamp: Optional[float] = None,
) -> GEXResult:
    """便捷函数: 从期权链计算 GEX

    Args:
        options: 期权链列表
        spot_price: 标的现价
        symbol: 交易对符号
        timestamp: 时间戳

    Returns:
        GEXResult
    """
    calc = GEXCalculator()
    return calc.calculate(options, spot_price, timestamp, symbol)


def create_synthetic_options(
    spot_price: float,
    n_strikes: int = 21,
    strike_step_pct: float = 0.05,
    expiries_days: List[int] = None,
    base_iv: float = 0.6,
    iv_slope: float = 0.001,
    base_oi: float = 1000.0,
) -> List[Dict[str, Any]]:
    """创建合成期权链 (用于测试和降级模式)

    生成一个合理的期权链用于无网络时的降级测试。

    Args:
        spot_price: 标的现价
        n_strikes: 行权价数量 (上下各 n_strikes//2)
        strike_step_pct: 行权价间距百分比 (5%)
        expiries_days: 到期日列表 (天)
        base_iv: 基础隐含波动率
        iv_slope: IV 期限结构斜率 (每天增加的IV)
        base_oi: 基础未平仓合约数

    Returns:
        list: 合成期权列表
    """
    if expiries_days is None:
        expiries_days = [7, 30, 60, 90, 180]

    options = []
    half = n_strikes // 2
    step = spot_price * strike_step_pct

    for days in expiries_days:
        ttm = days / 365.0
        # IV 期限结构 (远期IV略高)
        iv = base_iv + iv_slope * days
        # 到期越近, OI越大 (短期期权交易更活跃)
        oi_multiplier = 1.0 + (90 - min(days, 90)) / 30

        for i in range(-half, half + 1):
            strike = spot_price + i * step
            if strike <= 0:
                continue

            # OTM 期权 OI 更高 (做市商主要卖 OTM)
            moneyness = (spot_price - strike) / spot_price
            if abs(moneyness) < 0.1:
                oi_factor = 1.5  # ATM 期权最活跃
            else:
                oi_factor = max(0.3, 1.0 - abs(moneyness) * 2)

            oi = base_oi * oi_factor * oi_multiplier

            # 随机扰动 (±10%)
            import random
            oi *= (0.9 + random.random() * 0.2)

            # call
            options.append({
                "strike": strike,
                "ttm": ttm,
                "iv": iv,
                "oi": oi,
                "type": "call",
            })
            # put
            options.append({
                "strike": strike,
                "ttm": ttm,
                "iv": iv,
                "oi": oi * 0.8,  # put 略少
                "type": "put",
            })

    return options
