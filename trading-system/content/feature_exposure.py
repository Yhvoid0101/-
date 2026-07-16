#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FeatureExposure — 特征信息暴露器 (非加权决策) (v561 真集成 Phase 3)
================================================================================
用途: 把 _v561_feature_pipeline.py 的特征提取逻辑升级为运行时实体, 在进化主循环
      决策前注入特征信息 (非加权决策), 供 ReflectorAgent + RiskManager 消费.

设计原则:
  - 铁律 3: 不雕琢已证伪的特征 — v561 0/10 特征通过 P<0.05+|d|>0.1 (ERR-065 一致),
            保留为信息暴露而非加权决策
  - 纯函数内联: 从 _v561_feature_pipeline.py 抽取 10 个纯函数, 去除所有副作用
    (sys.exit monkey-patch / print / 顶层可执行代码 / 写文件)
  - 真闭环: 暴露特征数据供 ReflectorAgent 反思 + AlphaMiningSystem.update() 消费

来源参考:
  - _v561_feature_pipeline.py L97-290/L811-821: 10 个纯函数实现
  - v88 特征预测力科学分析 (ERR-20260701-v88fp): hold_bars 唯一显著特征 (d=+0.461)
  - v561 市场状态分类: quiet/trend/volatile/low_volatility 有区分度
  - v561 极端行情: top10% ATR 仍盈利 ($1819, wr60%), bot10% 胜率更高 (67%)

集成点 (evolution_loop.py):
  - 顶部软加载: from .feature_exposure import FeatureExposure
  - __init__ 实例化: self.feature_exposure = FeatureExposure(enabled=True)
  - 决策前注入: market_state["feature_exposure"] = self.feature_exposure.get_exposure_report(...)
================================================================================
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 内联纯函数 — 从 _v561_feature_pipeline.py L97-290 抽取, 去除副作用
# ============================================================

def ema(values: List[float], period: int) -> List[float]:
    """指数移动平均 (来源: _v561 L97)"""
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def rsi(close_prices: List[float], period: int = 14) -> List[float]:
    """RSI 相对强弱指标 (来源: _v561 L106)"""
    if len(close_prices) < period + 1:
        return [50.0] * len(close_prices)
    gains, losses = [], []
    for i in range(1, len(close_prices)):
        diff = close_prices[i] - close_prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    rsi_out = [50.0]
    avg_gain = sum(gains[:period]) / period if len(gains) >= period else 0
    avg_loss = sum(losses[:period]) / period if len(losses) >= period else 0.001
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi_out.append(100 - 100 / (1 + rs))
    return rsi_out


def compute_atr_pct(klines: List[Dict], period: int = 14) -> List[float]:
    """ATR 百分比 (相对价格, 来源: _v561 L124)"""
    if len(klines) < period + 1:
        return [0.01] * len(klines)
    trs = []
    for i in range(1, len(klines)):
        h, l = klines[i].get("high", 0), klines[i].get("low", 0)
        c_prev = klines[i - 1].get("close", 0)
        if c_prev > 0:
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev)) / c_prev
        else:
            tr = 0.01
        trs.append(tr)
    # Wilder smoothing
    atrs = [sum(trs[:period]) / period]
    for i in range(period, len(trs)):
        atrs.append((atrs[-1] * (period - 1) + trs[i]) / period)
    return [0.01] + atrs


def compute_adx(klines: List[Dict], period: int = 14) -> List[float]:
    """ADX 趋势强度指标 (简化版, 修复 NaN bug, 来源: _v561 L142)"""
    n = len(klines)
    if n < 2 * period:
        return [20.0] * n
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, n):
        h, l = klines[i].get("high", 0), klines[i].get("low", 0)
        h_p, l_p = klines[i - 1].get("high", 0), klines[i - 1].get("low", 0)
        c_prev = klines[i - 1].get("close", 0)
        up = h - h_p
        down = l_p - l
        plus_dm.append(float(up if (up > down and up > 0) else 0))
        minus_dm.append(float(down if (down > up and down > 0) else 0))
        if c_prev > 0:
            trs.append(float(max(h - l, abs(h - c_prev), abs(l - c_prev))))
        else:
            trs.append(0.01)
    # Wilder smoothing (纯 float 计算, 避免 statistics.mean 的 Python 3.12 bug)
    atr = sum(trs[:period]) / period
    plus_di_vals = [sum(plus_dm[:period]) / period / atr * 100 if atr > 0 else 0]
    minus_di_vals = [sum(minus_dm[:period]) / period / atr * 100 if atr > 0 else 0]
    dx_vals = []
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        pdm = (plus_di_vals[-1] * (period - 1) + plus_dm[i]) / period
        mdm = (minus_di_vals[-1] * (minus_dm[i] if i < len(minus_dm) else 0)) / period
        pdi = pdm / atr * 100 if atr > 0 else 0
        mdi = mdm / atr * 100 if atr > 0 else 0
        plus_di_vals.append(pdi)
        minus_di_vals.append(mdi)
        di_sum = pdi + mdi
        dx_vals.append(abs(pdi - mdi) / di_sum * 100 if di_sum > 0 else 0)
    # ADX = DX 的 Wilder 平滑
    if len(dx_vals) < period:
        return [20.0] * n
    adx = [sum(dx_vals[:period]) / period]
    for i in range(period, len(dx_vals)):
        adx.append((adx[-1] * (period - 1) + dx_vals[i]) / period)
    # NaN 检查: 如果有 NaN, 返回默认值 20.0
    result = [20.0] * period + adx
    if any(math.isnan(x) if isinstance(x, float) else False for x in result):
        return [20.0] * n
    return result


def detect_pinbar(klines: List[Dict], i: int) -> bool:
    """PinBar 形态检测 (价格行为, 来源: _v561 L185)"""
    if i < 1 or i >= len(klines):
        return False
    k = klines[i]
    h, l, o, c = k.get("high", 0), k.get("low", 0), k.get("open", 0), k.get("close", 0)
    body = abs(c - o)
    rng = h - l
    if rng <= 0 or body <= 0:
        return False
    body_ratio = body / rng
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    # PinBar: 影线≥2×实体, 实体≤30%范围
    return (lower_wick >= 2 * body or upper_wick >= 2 * body) and body_ratio <= 0.3


def detect_engulfing(klines: List[Dict], i: int) -> bool:
    """Engulfing 形态检测 (价格行为, 来源: _v561 L199)"""
    if i < 1 or i >= len(klines):
        return False
    k0, k1 = klines[i - 1], klines[i]
    o0, c0 = k0.get("open", 0), k0.get("close", 0)
    o1, c1 = k1.get("open", 0), k1.get("close", 0)
    if o0 == 0 or o1 == 0:
        return False
    # Bullish engulfing
    if c0 < o0 and c1 > o1 and c1 >= o0 and o1 <= c0:
        return True
    # Bearish engulfing
    if c0 > o0 and c1 < o1 and c1 <= o0 and o1 >= c0:
        return True
    return False


def detect_volume_divergence(klines: List[Dict], i: int, lookback: int = 20) -> int:
    """量价背离检测 (1=正背离, -1=负背离, 0=无, 来源: _v561 L214)"""
    if i < lookback or i >= len(klines):
        return 0
    prices = [klines[j].get("close", 0) for j in range(i - lookback, i + 1)]
    vols = [klines[j].get("volume", 0) for j in range(i - lookback, i + 1)]
    if not prices or not vols:
        return 0
    price_slope = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
    vol_slope = (vols[-1] - vols[0]) / vols[0] if vols[0] > 0 else 0
    # 价升量缩 = 负背离 (反转信号)
    if price_slope > 0.02 and vol_slope < -0.1:
        return -1
    # 价跌量增 = 正背离 (可能见底)
    if price_slope < -0.02 and vol_slope > 0.1:
        return 1
    return 0


def detect_bos_choch(klines: List[Dict], i: int, lookback: int = 20) -> int:
    """BOS/CHOCH 检测 (1=BOS 看涨, -1=CHOCH 看跌反转, 0=无, 来源: _v561 L228)"""
    if i < lookback or i >= len(klines):
        return 0
    recent = klines[max(0, i - lookback):i]
    if len(recent) < 5:
        return 0
    highs = [k.get("high", 0) for k in recent]
    lows = [k.get("low", 0) for k in recent]
    swing_high = max(highs)
    swing_low = min(lows)
    cur_close = klines[i].get("close", 0)
    # BOS: 突破前高
    if cur_close > swing_high * 1.001:
        return 1
    # CHOCH: 跌破前低
    if cur_close < swing_low * 0.999:
        return -1
    return 0


def extract_features(klines: List[Dict], entry_bar: int) -> Dict[str, float]:
    """从 klines 提取入场时的标准化特征向量 (来源: _v561 L244)

    覆盖: 价格行为(PinBar/Engulfing) + 量价(VolRatio/VP背离) +
          多周期结构(EMA对齐/ADX/BOS/CHOCH) + 波动率(ATR%) + 动量(RSI)
    """
    features: Dict[str, float] = {
        "adx": 20.0, "atr_pct": 0.01, "vol_ratio": 1.0,
        "ema_alignment": 0.0, "rsi": 50.0,
        "pinbar": 0.0, "engulfing": 0.0,
        "vol_divergence": 0.0, "bos_choch": 0.0,
        "kelly_mult": 1.0,  # 默认 1.0, 由 FeatureExposure._compute_kelly_mult 填充
    }
    if not klines or entry_bar >= len(klines) or entry_bar < 30:
        return features
    closes = [k.get("close", 0) for k in klines[:entry_bar + 1] if k.get("close", 0) > 0]
    if len(closes) < 50:
        return features

    # 多周期 EMA 对齐 (50/200)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, min(200, len(closes)))
    if ema50 and ema200 and ema200[-1] > 0:
        features["ema_alignment"] = (ema50[-1] - ema200[-1]) / ema200[-1]
    # ADX
    adx_vals = compute_adx(klines[:entry_bar + 1], 14)
    features["adx"] = adx_vals[-1] if adx_vals else 20.0
    # ATR%
    atr_vals = compute_atr_pct(klines[:entry_bar + 1], 14)
    features["atr_pct"] = atr_vals[-1] if atr_vals else 0.01
    # RSI
    rsi_vals = rsi(closes, 14)
    features["rsi"] = rsi_vals[-1] if rsi_vals else 50.0
    # VolRatio
    if entry_bar >= 10:
        vols = [klines[j].get("volume", 0) for j in range(entry_bar - 10, entry_bar)]
        if len(vols) >= 10:
            recent = sum(vols[-5:]) / 5
            prev = sum(vols[-10:-5]) / 5
            features["vol_ratio"] = recent / prev if prev > 0 else 1.0
    # PinBar/Engulfing
    features["pinbar"] = 1.0 if detect_pinbar(klines, entry_bar) else 0.0
    features["engulfing"] = 1.0 if detect_engulfing(klines, entry_bar) else 0.0
    # Vol Divergence
    features["vol_divergence"] = float(detect_volume_divergence(klines, entry_bar))
    # BOS/CHOCH
    features["bos_choch"] = float(detect_bos_choch(klines, entry_bar))
    return features


def classify_market_regime(adx: float, atr_pct: float) -> str:
    """市场状态分类 (来源: _v561 L811)

    分类:
      - trend: ADX≥25 (强趋势)
      - range: ADX<20 AND atr_pct<0.015 (低波动震荡)
      - volatile: atr_pct≥0.025 (高波动)
      - quiet: 其他 (低波动+趋势弱)

    v561 实测: quiet 最佳 ($11985, wr62.96%), trend 最差 ($722, wr57.30%),
              volatile 中等 ($729, wr54.25%) — 策略本质是均值回归非趋势跟踪
    """
    if adx >= 25:
        return "trend"
    if atr_pct >= 0.025:
        return "volatile"
    if adx < 20 and atr_pct < 0.015:
        return "range"
    return "quiet"


# ============================================================
# FeatureExposure 类 — 特征信息暴露器 (非加权决策)
# ============================================================

class FeatureExposure:
    """特征信息暴露器 (非加权决策)

    复用 _v561_feature_pipeline.py 的 extract_features(11维) + classify_market_regime(4状态),
    纯函数内联, 去除所有副作用 (sys.exit monkey-patch / print / 顶层可执行代码 / 写文件).

    设计原则 (铁律 3): 已证伪的特征不参与加权决策, 仅作为信息暴露
      - 0/10 特征通过 P<0.05+|d|>0.1 (ERR-065 一致)
      - 市场状态分类保留 (quiet/trend/volatile/low_volatility) — 有区分度
      - 极端行情检测保留 — 有实际价值

    与 alpha_mining.py 联动: 暴露特征数据供 AlphaMiningSystem.update() 消费

    v88 特征预测力科学分析 (ERR-20260701-v88fp):
      - hold_bars 唯一显著特征 (d=+0.461, P=0.0045***)
      - atr_abs 有正向预测力 (d=+0.245, P=0.077*)
      - composite_score 与 PnL 负相关 (d=-0.240) — 不使用加权
      → 本类仅暴露特征, 不参与加权决策
    """

    # 极端行情阈值 (基于 v561 实测: top10% ATR 仍盈利 $1819, wr60%)
    EXTREME_ATR_PCT_THRESHOLD = 0.04  # top10% ATR 百分比
    EXTREME_VOL_RATIO_HIGH = 2.0     # 异常放量
    EXTREME_VOL_RATIO_LOW = 0.5      # 异常缩量

    # Kelly 公式参数 (简化版, 不生搬硬套)
    KELLY_MIN = 0.1   # 最低仓位倍数 (避免空仓)
    KELLY_MAX = 2.0   # 最高仓位倍数 (避免过度杠杆)

    def __init__(self, enabled: bool = True):
        """初始化特征信息暴露器

        Args:
            enabled: 是否启用 (False 时所有方法返回默认值)
        """
        self.enabled = enabled
        # 缓存最近一次的特征报告 (供 ReflectorAgent 消费)
        self._last_report: Optional[Dict] = None
        logger.debug("FeatureExposure 初始化 (enabled=%s)", enabled)

    def extract(self, klines: List[Dict], entry_bar: int) -> Dict[str, float]:
        """提取 11 维特征 (信息暴露, 不参与加权)

        Args:
            klines: K 线数据列表 [{'open', 'high', 'low', 'close', 'volume'}, ...]
            entry_bar: 入场 K 线索引

        Returns:
            11 维特征字典:
              - adx: ADX 趋势强度 (0-100)
              - atr_pct: ATR 百分比 (相对价格)
              - vol_ratio: 成交量比率 (近5根/前5根)
              - ema_alignment: EMA50/EMA200 对齐度
              - rsi: RSI 相对强弱指标 (0-100)
              - pinbar: PinBar 形态 (1.0/0.0)
              - engulfing: Engulfing 形态 (1.0/0.0)
              - vol_divergence: 量价背离 (1.0/-1.0/0.0)
              - bos_choch: BOS/CHOCH 结构突破 (1.0/-1.0/0.0)
              - kelly_mult: 凯利仓位倍数 (0.1-2.0)
              - regime: 市场状态 (字符串转 hash 数值)
        """
        if not self.enabled:
            return {
                "adx": 20.0, "atr_pct": 0.01, "vol_ratio": 1.0,
                "ema_alignment": 0.0, "rsi": 50.0,
                "pinbar": 0.0, "engulfing": 0.0,
                "vol_divergence": 0.0, "bos_choch": 0.0,
                "kelly_mult": 1.0, "regime": 0.0,
            }
        features = extract_features(klines, entry_bar)
        # 填充 kelly_mult (基于 RSI + ATR 的简化 Kelly, 不生搬硬套)
        features["kelly_mult"] = self._compute_kelly_mult(features)
        return features

    def classify_regime(self, adx: float, atr_pct: float) -> str:
        """市场状态分类 (trend/range/volatile/quiet)

        Args:
            adx: ADX 趋势强度
            atr_pct: ATR 百分比

        Returns:
            市场状态字符串:
              - trend: ADX≥25 (强趋势)
              - range: ADX<20 AND atr_pct<0.015 (低波动震荡)
              - volatile: atr_pct≥0.025 (高波动)
              - quiet: 其他 (低波动+趋势弱)

        v561 实测:
              - quiet 最佳 ($11985, wr62.96%) — 均值回归策略最优环境
              - trend 最差 ($722, wr57.30%) — 趋势跟踪不适用
              - volatile 中等 ($729, wr54.25%) — 高波动需减仓
        """
        if not self.enabled:
            return "quiet"
        return classify_market_regime(adx, atr_pct)

    def detect_extreme(self, features: Dict[str, float]) -> bool:
        """极端行情检测 (top10% ATR / 异常成交量)

        v561 实测:
          - 高波动 top10% ATR 仍盈利 ($1819, wr60%) — 策略在极端行情仍有效
          - 低波动 bot10% 胜率更高 (67%) — 低波动是策略优势区

        Args:
            features: extract() 返回的特征字典

        Returns:
            True 如果检测到极端行情 (top10% ATR 或异常成交量)
        """
        if not self.enabled or not features:
            return False
        atr_pct = features.get("atr_pct", 0.01)
        vol_ratio = features.get("vol_ratio", 1.0)
        # top10% ATR
        if atr_pct >= self.EXTREME_ATR_PCT_THRESHOLD:
            return True
        # 异常放量或缩量
        if vol_ratio >= self.EXTREME_VOL_RATIO_HIGH:
            return True
        if vol_ratio <= self.EXTREME_VOL_RATIO_LOW:
            return True
        return False

    def get_exposure_report(self, klines: List[Dict], entry_bar: int) -> Dict[str, Any]:
        """生成完整信息暴露报告 (供 ReflectorAgent + RiskManager 消费)

        Args:
            klines: K 线数据列表
            entry_bar: 入场 K 线索引

        Returns:
            完整信息暴露报告:
              {
                'features': Dict[str, float],  # 11 维特征
                'regime': str,                # 市场状态 (trend/range/volatile/quiet)
                'is_extreme': bool,            # 极端行情标志
                'kelly_mult': float,           # 凯利仓位倍数 (0.1-2.0)
              }
        """
        if not self.enabled or not klines or entry_bar >= len(klines) or entry_bar < 30:
            report = {
                "features": {
                    "adx": 20.0, "atr_pct": 0.01, "vol_ratio": 1.0,
                    "ema_alignment": 0.0, "rsi": 50.0,
                    "pinbar": 0.0, "engulfing": 0.0,
                    "vol_divergence": 0.0, "bos_choch": 0.0,
                    "kelly_mult": 1.0, "regime": 0.0,
                },
                "regime": "quiet",
                "is_extreme": False,
                "kelly_mult": 1.0,
            }
            self._last_report = report
            return report

        # 提取 11 维特征
        features = self.extract(klines, entry_bar)
        # 市场状态分类
        regime = self.classify_regime(features.get("adx", 20.0), features.get("atr_pct", 0.01))
        # 极端行情检测
        is_extreme = self.detect_extreme(features)
        # Kelly 仓位倍数
        kelly_mult = features.get("kelly_mult", 1.0)

        report = {
            "features": features,
            "regime": regime,
            "is_extreme": is_extreme,
            "kelly_mult": kelly_mult,
        }
        self._last_report = report
        logger.debug(
            "FeatureExposure report: regime=%s, is_extreme=%s, kelly_mult=%.3f, adx=%.1f, atr_pct=%.4f",
            regime, is_extreme, kelly_mult, features.get("adx", 20.0), features.get("atr_pct", 0.01),
        )
        return report

    def _compute_kelly_mult(self, features: Dict[str, float]) -> float:
        """计算凯利仓位倍数 (简化版, 不生搬硬套)

        基于 RSI + ATR 的简化 Kelly:
          - RSI 接近 50 (中性) + 低 ATR → 高仓位 (均值回归最优环境)
          - RSI 极端 (>70 或 <30) + 高 ATR → 低仓位 (反转风险)

        凯利公式原始形式: f* = W - (1-W)/R
          其中 W=胜率, R=盈亏比
          但此处不使用历史胜率 (避免过拟合), 而用 RSI+ATR 代理

        限制: 0.1 ≤ kelly_mult ≤ 2.0 (避免空仓和过度杠杆)

        Args:
            features: 特征字典

        Returns:
            凯利仓位倍数 (0.1-2.0)
        """
        if not self.enabled or not features:
            return 1.0

        rsi = features.get("rsi", 50.0)
        atr_pct = features.get("atr_pct", 0.01)
        adx = features.get("adx", 20.0)

        # RSI 中性度: RSI=50 时最优 (1.0), RSI 极端时衰减
        rsi_neutrality = 1.0 - abs(rsi - 50.0) / 50.0  # 0.0-1.0
        # ATR 稳定性: 低 ATR 时最优 (1.0), 高 ATR 时衰减
        atr_stability = max(0.0, 1.0 - atr_pct / 0.04)  # 0.0-1.0, ATR≥0.04 时为 0
        # ADX 趋势性: ADX 低时最优 (均值回归), ADX 高时衰减
        adx_penalty = max(0.0, 1.0 - (adx - 20.0) / 30.0) if adx > 20.0 else 1.0  # 0.0-1.0

        # 综合 Kelly 倍数 (0.1-2.0)
        # 基准 1.0, 中性环境加成最高到 2.0, 极端环境降到 0.1
        kelly_raw = 0.5 + 0.5 * rsi_neutrality * atr_stability * adx_penalty
        # 基准 1.0 + 调整 (最大 ±1.0)
        kelly_mult = 1.0 + (kelly_raw - 0.5) * 2.0
        # 限制范围
        kelly_mult = max(self.KELLY_MIN, min(self.KELLY_MAX, kelly_mult))
        return round(kelly_mult, 3)

    def get_last_report(self) -> Optional[Dict]:
        """获取最近一次的特征报告 (供 ReflectorAgent 消费)"""
        return self._last_report
