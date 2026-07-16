#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多周期结构核心模块 (Multi-Timeframe Core Module)
================================================
短板 #2 补齐: 从 _v512_market_structure.py 提炼并扩展为 N 周期统一核心

设计原则 (用户铁律):
  - "不要生搬硬套" — 各周期有权重, 大周期有否决权 (顺大势逆小势)
  - "融会贯通" — BOS/CHOCH 跨周期联动, 不只比较 trend 字符串
  - "信手拈来" — 统一接口, 可被 feature_fusion_pipeline 直接调用

核心扩展 (相比 v512 原版):
  1. N 周期支持: 从双周期 (HTF/LTF) 扩展为 N 周期堆叠 (如 4h/1h/15m/5m)
  2. 加权强度: 大周期权重高, 4h 权重 0.4, 1h 权重 0.3, 15m 权重 0.2, 5m 权重 0.1
  3. 非对称过滤: HTF=up 禁 short (顺大势), HTF=down 保留 long (逆小势抓反弹)
  4. BOS/CHOCH 联动: 跨周期突破信号传递
  5. 连续强度值: 从 0/1/2 离散 → 0.0-1.0 连续 (一致性比例)

使用方式:
  from mtf_core import MTFCore, extract_mtf_features

  mtf = MTFCore(timeframes=["4h", "1h", "15m"], weights=[0.5, 0.3, 0.2])
  result = mtf.analyze(klines_by_tf={"4h": klines_4h, "1h": klines_1h, "15m": klines_15m})
  # result.direction → "long" / "short" / "neutral"
  # result.strength → 0.0-1.0
  # result.is_confluent → bool
  # result.htf_veto → bool (大周期否决)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("MTFCore")

# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class TimeframeAnalysis:
    """单周期分析结果"""
    timeframe: str
    trend: str                      # 'up' | 'down' | 'range'
    trend_strength: float           # 0.0-1.0 (基于 HH/HL 序列一致性)
    last_bos: Optional[str] = None  # 'BOS_UP' | 'BOS_DOWN' | None
    last_choch: Optional[str] = None  # 'CHOCH_UP' | 'CHOCH_DOWN' | None
    hh_count: int = 0
    hl_count: int = 0
    lh_count: int = 0
    ll_count: int = 0


@dataclass
class MTFResult:
    """多周期融合结果"""
    direction: str = "neutral"      # 'long' | 'short' | 'neutral'
    strength: float = 0.0           # 0.0-1.0 (加权一致性)
    is_confluent: bool = False      # 是否多周期一致
    htf_veto: bool = False          # 大周期否决 (HTF=up 禁 short 等)
    htf_direction: str = "neutral"  # 最高周期方向 (用于非对称过滤)
    timeframe_analyses: List[TimeframeAnalysis] = field(default_factory=list)
    confluence_score: float = 0.0   # 0-100 (综合评分)
    weighted_direction_score: float = 0.0  # 加权方向分 (正=多, 负=空)


# ============================================================================
# 核心引擎
# ============================================================================

class MTFCore:
    """多周期结构核心引擎

    支持 N 周期堆叠, 加权融合, 非对称过滤, BOS/CHOCH 联动
    """

    # 默认周期权重 (大周期权重高)
    DEFAULT_WEIGHTS = {
        "4h": 0.40,
        "1h": 0.30,
        "15m": 0.20,
        "5m": 0.10,
    }

    def __init__(
        self,
        timeframes: List[str] = None,
        weights: Dict[str, float] = None,
        order: int = 5,
        htf_veto_enabled: bool = True,
    ):
        """
        Args:
            timeframes: 周期列表, 从大到小 (如 ["4h", "1h", "15m"])
            weights: 各周期权重, 必须和 timeframes 对应
            order: swing 点检测窗口
            htf_veto_enabled: 是否启用大周期否决 (顺大势逆小势)
        """
        self.timeframes = timeframes or ["4h", "1h", "15m"]
        if weights:
            self.weights = weights
        else:
            # 使用默认权重, 缺失的周期均匀分配剩余权重
            self.weights = {}
            remaining = 1.0
            assigned = 0
            for tf in self.timeframes:
                if tf in self.DEFAULT_WEIGHTS:
                    self.weights[tf] = self.DEFAULT_WEIGHTS[tf]
                    remaining -= self.DEFAULT_WEIGHTS[tf]
                    assigned += 1
            # 剩余权重均匀分配给未在默认表中的周期
            unassigned = [tf for tf in self.timeframes if tf not in self.weights]
            if unassigned:
                share = remaining / len(unassigned)
                for tf in unassigned:
                    self.weights[tf] = share

        # 归一化权重
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.order = max(2, order)
        self.htf_veto_enabled = htf_veto_enabled

    def analyze(self, klines_by_tf: Dict[str, List[Dict]]) -> MTFResult:
        """分析多周期结构

        Args:
            klines_by_tf: {timeframe: klines} 字典, 每个 klines 是 List[Dict]
                          K线Dict需含 high/low/close 字段

        Returns:
            MTFResult: 多周期融合结果
        """
        result = MTFResult()

        # 1. 逐周期分析
        tf_analyses: List[TimeframeAnalysis] = []
        for tf in self.timeframes:
            klines = klines_by_tf.get(tf, [])
            if len(klines) < 2 * self.order + 1:
                logger.debug(f"周期 {tf} 数据不足 ({len(klines)} < {2*self.order+1}), 跳过")
                continue
            analysis = self._analyze_single_tf(tf, klines)
            tf_analyses.append(analysis)

        if not tf_analyses:
            return result

        result.timeframe_analyses = tf_analyses

        # 2. 加权方向评分
        # trend → direction_score: up=+1, down=-1, range=0
        direction_map = {"up": 1.0, "down": -1.0, "range": 0.0}
        weighted_score = 0.0
        for ta in tf_analyses:
            w = self.weights.get(ta.timeframe, 0.0)
            ds = direction_map.get(ta.trend, 0.0) * ta.trend_strength
            weighted_score += w * ds

        result.weighted_direction_score = weighted_score

        # 3. 方向判定
        threshold = 0.15  # 加权方向分阈值
        if weighted_score > threshold:
            result.direction = "long"
        elif weighted_score < -threshold:
            result.direction = "short"
        else:
            result.direction = "neutral"

        # 4. 一致性强度 (0.0-1.0)
        # 计算非 range 周期的一致性
        non_range = [ta for ta in tf_analyses if ta.trend != "range"]
        if non_range:
            directions = [ta.trend for ta in non_range]
            # 计算最多方向占比
            dir_counts = {"up": 0, "down": 0}
            for d in directions:
                if d in dir_counts:
                    dir_counts[d] += 1
            max_count = max(dir_counts.values())
            result.strength = max_count / len(non_range)
            result.is_confluent = max_count == len(non_range) and len(non_range) >= 2
        else:
            result.strength = 0.0
            result.is_confluent = False

        # 5. 综合评分 (0-100)
        # strength * 50 + abs(weighted_score) * 50
        result.confluence_score = (
            result.strength * 50.0 + abs(weighted_score) * 50.0
        )

        # 6. 大周期否决 (非对称过滤)
        if tf_analyses and self.htf_veto_enabled:
            htf = tf_analyses[0]  # 第一个周期是最高周期
            result.htf_direction = htf.trend
            # HTF=up 禁 short (顺大势)
            # HTF=down 保留 long (逆小势抓反弹) — 不否决
            if htf.trend == "up" and result.direction == "short":
                result.htf_veto = True
                result.direction = "neutral"
                logger.debug(f"HTF({htf.timeframe})=up 否决 short, 改为 neutral")

        # 7. BOS/CHOCH 跨周期联动检测
        # 如果 LTF 出现与 HTF 方向一致的 BOS, 增强信号
        if len(tf_analyses) >= 2:
            htf_bos = tf_analyses[0].last_bos
            ltf_choch = tf_analyses[-1].last_choch
            # HTF BOS_UP + LTF CHOCH_UP → 增强做多
            if htf_bos == "BOS_UP" and ltf_choch == "CHOCH_UP":
                result.confluence_score = min(100, result.confluence_score * 1.15)
                if result.direction == "neutral":
                    result.direction = "long"
            # HTF BOS_DOWN + LTF CHOCH_DOWN → 增强做空
            elif htf_bos == "BOS_DOWN" and ltf_choch == "CHOCH_DOWN":
                result.confluence_score = min(100, result.confluence_score * 1.15)
                if result.direction == "neutral":
                    result.direction = "short"

        return result

    def _analyze_single_tf(self, tf: str, klines: List[Dict]) -> TimeframeAnalysis:
        """分析单周期结构

        基于 HH/HL/LH/LL 序列判定趋势
        """
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        closes = [k["close"] for k in klines]

        # 识别 swing 点
        swing_highs, swing_lows = self._find_swings(highs, lows, self.order)

        # 计算 HH/HL/LH/LL
        hh_count, hl_count, lh_count, ll_count = 0, 0, 0, 0
        for i in range(1, len(swing_highs)):
            if swing_highs[i] > swing_highs[i - 1]:
                hh_count += 1
            else:
                lh_count += 1
        for i in range(1, len(swing_lows)):
            if swing_lows[i] > swing_lows[i - 1]:
                hl_count += 1
            else:
                ll_count += 1

        # 趋势判定
        up_score = hh_count + hl_count
        down_score = lh_count + ll_count
        total = up_score + down_score

        if total == 0:
            trend = "range"
            trend_strength = 0.0
        elif up_score > down_score * 1.3:
            trend = "up"
            trend_strength = up_score / total
        elif down_score > up_score * 1.3:
            trend = "down"
            trend_strength = down_score / total
        else:
            trend = "range"
            trend_strength = 0.5  # 震荡

        # BOS/CHOCH 检测 (简化版)
        last_bos = None
        last_choch = None
        if len(swing_highs) >= 2:
            if swing_highs[-1] > swing_highs[-2] and trend == "up":
                last_bos = "BOS_UP"
            elif swing_highs[-1] < swing_highs[-2] and trend == "down":
                last_bos = "BOS_DOWN"
        if len(swing_lows) >= 2:
            if swing_lows[-1] > swing_lows[-2] and trend == "down":
                last_choch = "CHOCH_UP"  # 下跌中低点抬高 → 可能反转向上
            elif swing_lows[-1] < swing_lows[-2] and trend == "up":
                last_choch = "CHOCH_DOWN"  # 上涨中低点降低 → 可能反转向下

        return TimeframeAnalysis(
            timeframe=tf,
            trend=trend,
            trend_strength=trend_strength,
            last_bos=last_bos,
            last_choch=last_choch,
            hh_count=hh_count,
            hl_count=hl_count,
            lh_count=lh_count,
            ll_count=ll_count,
        )

    @staticmethod
    def _find_swings(
        highs: List[float], lows: List[float], order: int
    ) -> Tuple[List[float], List[float]]:
        """寻找 swing high 和 swing low

        Args:
            highs: 最高价序列
            lows: 最低价序列
            order: 左右各 order 根 K 线内极值

        Returns:
            (swing_highs, swing_lows)
        """
        swing_highs = []
        swing_lows = []
        n = len(highs)

        for i in range(order, n - order):
            # Swing high: 左右 order 根内最高
            is_high = all(highs[i] >= highs[i + j] for j in range(-order, order + 1) if j != 0)
            if is_high:
                swing_highs.append(highs[i])
            # Swing low: 左右 order 根内最低
            is_low = all(lows[i] <= lows[i + j] for j in range(-order, order + 1) if j != 0)
            if is_low:
                swing_lows.append(lows[i])

        return swing_highs, swing_lows


# ============================================================================
# 便捷函数
# ============================================================================

def extract_mtf_features(
    klines_by_tf: Dict[str, List[Dict]],
    timeframes: List[str] = None,
    weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """提取 MTF 特征 (供 feature_fusion_pipeline 调用)

    Args:
        klines_by_tf: {timeframe: klines} 字典
        timeframes: 周期列表
        weights: 权重字典

    Returns:
        特征字典:
          - mtf_direction_score: 加权方向分 (-1.0 ~ +1.0)
          - mtf_strength: 一致性强度 (0.0-1.0)
          - mtf_is_confluent: 是否一致 (0/1)
          - mtf_confluence_score: 综合评分 (0-100)
          - mtf_htf_veto: 大周期否决 (0/1)
          - mtf_htf_direction: 大周期方向 (1=up, -1=down, 0=range)
          - mtf_bos_choch_enhanced: BOS/CHOCH 增强 (0/1)
    """
    mtf = MTFCore(timeframes=timeframes, weights=weights)
    result = mtf.analyze(klines_by_tf)

    direction_map = {"up": 1, "down": -1, "range": 0}

    return {
        "mtf_direction_score": result.weighted_direction_score,
        "mtf_strength": result.strength,
        "mtf_is_confluent": 1.0 if result.is_confluent else 0.0,
        "mtf_confluence_score": result.confluence_score / 100.0,  # 归一化到 0-1
        "mtf_htf_veto": 1.0 if result.htf_veto else 0.0,
        "mtf_htf_direction": direction_map.get(result.htf_direction, 0),
        "mtf_bos_choch_enhanced": 1.0 if result.confluence_score > 80 else 0.0,
    }


# ============================================================================
# CLI (独立测试)
# ============================================================================

if __name__ == "__main__":
    # 生成测试数据
    import random
    random.seed(42)

    def gen_klines(n: int, trend: str = "up") -> List[Dict]:
        """生成模拟 K 线"""
        klines = []
        price = 100.0
        for i in range(n):
            if trend == "up":
                change = random.uniform(-0.5, 1.5)
            elif trend == "down":
                change = random.uniform(-1.5, 0.5)
            else:
                change = random.uniform(-0.8, 0.8)
            high = price + abs(change) + random.uniform(0, 0.5)
            low = price - abs(change) - random.uniform(0, 0.5)
            close = price + change
            klines.append({"high": high, "low": low, "close": close, "volume": 1000})
            price = close
        return klines

    # 测试: 4h 上涨, 1h 上涨, 15m 震荡 → 应判定为 long
    print("=== 测试1: 4h+1h 上涨, 15m 震荡 ===")
    klines_by_tf = {
        "4h": gen_klines(50, "up"),
        "1h": gen_klines(50, "up"),
        "15m": gen_klines(50, "range"),
    }
    mtf = MTFCore(timeframes=["4h", "1h", "15m"])
    result = mtf.analyze(klines_by_tf)
    print(f"  方向: {result.direction}")
    print(f"  强度: {result.strength:.2f}")
    print(f"  一致: {result.is_confluent}")
    print(f"  评分: {result.confluence_score:.1f}")
    print(f"  HTF否决: {result.htf_veto}")
    print(f"  加权方向分: {result.weighted_direction_score:+.3f}")

    # 测试: 4h 上涨, 1h 下跌 → HTF 否决 short
    print("\n=== 测试2: 4h 上涨, 1h 下跌 (HTF 否决) ===")
    klines_by_tf = {
        "4h": gen_klines(50, "up"),
        "1h": gen_klines(50, "down"),
    }
    mtf = MTFCore(timeframes=["4h", "1h"])
    result = mtf.analyze(klines_by_tf)
    print(f"  方向: {result.direction}")
    print(f"  HTF否决: {result.htf_veto}")
    print(f"  HTF方向: {result.htf_direction}")

    # 测试: 特征提取
    print("\n=== 测试3: 特征提取 ===")
    features = extract_mtf_features({
        "4h": gen_klines(50, "up"),
        "1h": gen_klines(50, "up"),
    })
    for k, v in features.items():
        print(f"  {k}: {v:.3f}")
