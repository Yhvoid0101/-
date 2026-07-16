# -*- coding: utf-8 -*-
"""
v97 自适应持仓时间计算 (Adaptive Hold Bars)
====================================================================
设计原则: 市场状态是 CAUSE → 决定适合的持有时间 → hold_bars 是 CONSEQUENCE
来源: ERR-109 (hold_bars 是 consequence 不是 cause)
集成点: agent_decision_engine.decide() → suggested_hold_bars 字段 (仅建议, 非强制)
用户铁律: "凯利+价格行为+量价+多周期融会贯通"
"""

import logging
from typing import Tuple

logger = logging.getLogger("hermes.v97_adaptive_hold")

BASE_HOLD_BARS = 48  # 基线持有K线数


def compute_adaptive_hold(
    atr_pct: float,
    vol_ratio: float,
    adx: float = 0.0,
) -> Tuple[int, str]:
    """计算自适应持仓时间

    参数:
        atr_pct: ATR占价格百分比 (atr_val / close), 典型 0.005~0.05
        vol_ratio: 量价比率 (vp_vol_ratio_5), 典型 0.5~3.0
        adx: ADX指标值 (adx_14), 典型 0~60

    返回:
        (hold_bars, regime):
            hold_bars: 建议持有K线数 (12~120)
            regime: 持仓状态标记
                "trending" — 趋势明确, 长持有
                "volatile" — 高波动, 短持有
                "ranging" — 震荡, 中等持有
                "quiet" — 低波动安静, 长持有
                "default" — 默认基线
    """
    try:
        atr_pct = float(atr_pct) if atr_pct and atr_pct > 0 else 0.01
        vol_ratio = float(vol_ratio) if vol_ratio and vol_ratio > 0 else 1.0
        adx = float(adx) if adx and adx >= 0 else 0.0
    except (TypeError, ValueError):
        return BASE_HOLD_BARS, "default"

    # === 状态分类 ===
    # ADX > 25 → 趋势明确
    # ATR% > 3% 或 vol_ratio > 2.0 → 高波动
    # ADX < 18 且 ATR% < 1.5% → 震荡
    # ATR% < 1% 且 vol_ratio < 0.8 → 安静
    if adx > 25:
        regime = "trending"
    elif atr_pct > 0.03 or vol_ratio > 2.0:
        regime = "volatile"
    elif adx < 18 and atr_pct < 0.015:
        regime = "ranging"
    elif atr_pct < 0.01 and vol_ratio < 0.8:
        regime = "quiet"
    else:
        regime = "default"

    # === hold_bars 计算 ===
    # 基线 48, 根据 regime 调整
    if regime == "trending":
        # 趋势明确 → 长持有 (60~120)
        # ADX 越高, 持有越久
        hold = int(60 + min(adx - 25, 35) * 1.7)
    elif regime == "volatile":
        # 高波动 → 短持有 (12~24)
        # ATR% 越高, 持有越短
        hold = int(max(12, 24 - (atr_pct - 0.03) * 200))
    elif regime == "ranging":
        # 震荡 → 中等持有 (24~48)
        hold = 36
    elif regime == "quiet":
        # 安静 → 长持有 (60~96)
        hold = 72
    else:
        # 默认基线
        hold = BASE_HOLD_BARS

    # 量价比修正: 高量价 → 缩短持有
    if vol_ratio > 1.5:
        hold = int(hold * 0.8)
    elif vol_ratio < 0.7:
        hold = int(hold * 1.15)

    # 边界约束: 12~120
    hold = max(12, min(120, hold))

    return hold, regime
