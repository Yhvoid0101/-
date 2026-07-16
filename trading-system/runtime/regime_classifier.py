#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态分位数市场状态分类器 (RegimeClassifier)

替代硬编码阈值(ATR_pct 0.01/0.03), 使用数据驱动的分位数阈值,
确保4种市场状态(trending/ranging/high_vol/low_vol)均有样本覆盖。

背景 (ERR-20260702-v613):
  v613发现v97的4regime覆盖率仅3/4:
    - trending: 141笔 ✅
    - ranging: 106笔 ✅
    - high_vol: 5笔 ❌ (<15阈值)
    - low_vol: 0笔 ❌ (缺失)
  根因: 硬编码ATR_pct阈值(0.01/0.03)不匹配实际数据分布

修复方案:
  1. ADX阈值: 保持25(行业标准)或使用分位数
  2. 波动率代理: ATR_pct优先, 缺失时用|pnl_pct|分位数
  3. 动态阈值: q25/q75分位数确保4regime均有样本

设计原则 (用户铁律):
  - "理解底层逻辑而非表面指标" — 分位数适应实际数据分布
  - "不要通过已知数据反推策略" — 分类器只做状态识别, 不优化策略参数
  - "融会贯通" — ADX(趋势)+波动率(风险)多维分类
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hermes.regime_classifier")


@dataclass
class RegimeThresholds:
    """市场状态分类阈值快照"""
    adx_threshold: float = 25.0
    vol_low_threshold: float = 0.01    # ATR_pct或|pnl_pct|下界
    vol_high_threshold: float = 0.03   # ATR_pct或|pnl_pct|上界
    vol_metric: str = "atr_pct"        # 使用的波动率指标名
    is_quantile_based: bool = False    # 是否动态分位数阈值
    sample_size: int = 0               # 拟合样本量


class RegimeClassifier:
    """动态分位数市场状态分类器

    分类逻辑:
      1. trending: ADX > adx_threshold (趋势强)
      2. 非trending按波动率分位数:
         - high_vol: vol > vol_high_threshold (q75)
         - low_vol: vol < vol_low_threshold (q25)
         - ranging: vol_low ≤ vol ≤ vol_high (中间50%)

    使用方式:
      clf = RegimeClassifier()
      clf.fit(trades)                    # 拟合分位数阈值
      regimes = clf.classify_all(trades) # 批量分类

    或直接使用静态方法(单笔分类, 需预知阈值):
      regime = RegimeClassifier.classify(trade, thresholds)
    """

    def __init__(self, adx_threshold: float = 25.0,
                 use_quantile: bool = True,
                 vol_metric: str = "atr_pct"):
        """初始化分类器

        Args:
            adx_threshold: ADX趋势阈值, 默认25(行业标准)
            use_quantile: 是否使用动态分位数阈值(True)或硬编码(False)
            vol_metric: 波动率指标字段名, "atr_pct"或"pnl_pct"(用绝对值)
        """
        self.adx_threshold = float(adx_threshold)
        self.use_quantile = use_quantile
        self.vol_metric = vol_metric
        self.thresholds: Optional[RegimeThresholds] = None

    def _extract_volatility(self, trade: Dict[str, Any]) -> float:
        """从交易记录提取波动率值

        优先级:
          1. atr_pct字段 (最准确的波动率指标)
          2. |pnl_pct| (波动率代理, 当atr_pct缺失时)

        Returns:
            波动率值(非负)
        """
        # 尝试atr_pct
        atr = trade.get("atr_pct")
        if atr is not None and isinstance(atr, (int, float)) and atr > 0:
            return float(atr)

        # 尝试features嵌套字段
        features = trade.get("features", {})
        if isinstance(features, dict):
            atr = features.get("atr_pct")
            if atr is not None and isinstance(atr, (int, float)) and atr > 0:
                return float(atr)

        # 回退: 用|pnl_pct|作为波动率代理
        pnl = trade.get("pnl_pct", 0)
        if isinstance(pnl, (int, float)):
            return abs(float(pnl))

        return 0.0

    def _extract_adx(self, trade: Dict[str, Any]) -> float:
        """从交易记录提取ADX值"""
        adx = trade.get("adx")
        if adx is not None and isinstance(adx, (int, float)):
            return float(adx)

        features = trade.get("features", {})
        if isinstance(features, dict):
            adx = features.get("adx")
            if adx is not None and isinstance(adx, (int, float)):
                return float(adx)

        return 0.0

    def fit(self, trades: List[Dict[str, Any]]) -> "RegimeClassifier":
        """从交易数据拟合动态分位数阈值

        Args:
            trades: 交易记录列表

        Returns:
            self (支持链式调用)
        """
        if not trades:
            self.thresholds = RegimeThresholds(
                adx_threshold=self.adx_threshold,
                vol_low_threshold=0.01,
                vol_high_threshold=0.03,
                vol_metric=self.vol_metric,
                is_quantile_based=False,
                sample_size=0,
            )
            return self

        # 提取波动率值
        vol_values = [self._extract_volatility(t) for t in trades]
        vol_values = [v for v in vol_values if v > 0]

        # 检测实际使用的波动率指标
        sample_trade = trades[0]
        actual_metric = "atr_pct" if sample_trade.get("atr_pct") is not None else "pnl_pct"

        if self.use_quantile and len(vol_values) >= 10:
            # 动态分位数阈值: q25/q75
            q25 = float(np.percentile(vol_values, 25))
            q75 = float(np.percentile(vol_values, 75))
            self.thresholds = RegimeThresholds(
                adx_threshold=self.adx_threshold,
                vol_low_threshold=q25,
                vol_high_threshold=q75,
                vol_metric=actual_metric,
                is_quantile_based=True,
                sample_size=len(vol_values),
            )
            logger.info(
                f"RegimeClassifier拟合(分位数): ADX>{self.adx_threshold}, "
                f"{actual_metric} q25={q25:.4f} q75={q75:.4f}, n={len(vol_values)}"
            )
        else:
            # 回退到硬编码阈值
            self.thresholds = RegimeThresholds(
                adx_threshold=self.adx_threshold,
                vol_low_threshold=0.01,
                vol_high_threshold=0.03,
                vol_metric=actual_metric,
                is_quantile_based=False,
                sample_size=len(vol_values),
            )
            logger.warning(
                f"RegimeClassifier拟合(硬编码): 样本不足({len(vol_values)}<10), "
                f"使用默认ATR_pct阈值0.01/0.03"
            )

        return self

    def classify(self, trade: Dict[str, Any]) -> str:
        """分类单笔交易的市场状态

        Args:
            trade: 交易记录

        Returns:
            regime: "trending" / "ranging" / "high_vol" / "low_vol"
        """
        if self.thresholds is None:
            self.fit([trade])

        adx = self._extract_adx(trade)
        vol = self._extract_volatility(trade)

        # 1. 趋势优先: ADX > 阈值
        if adx > self.thresholds.adx_threshold:
            return "trending"

        # 2. 非趋势: 按波动率分3档
        if vol > self.thresholds.vol_high_threshold:
            return "high_vol"
        elif vol < self.thresholds.vol_low_threshold:
            return "low_vol"
        else:
            return "ranging"

    def classify_all(self, trades: List[Dict[str, Any]]) -> List[str]:
        """批量分类交易的市场状态

        Args:
            trades: 交易记录列表

        Returns:
            regimes: 市场状态列表(与trades等长)
        """
        self.fit(trades)
        return [self.classify(t) for t in trades]

    def get_thresholds(self) -> Optional[RegimeThresholds]:
        """获取当前阈值快照"""
        return self.thresholds

    @staticmethod
    def classify_one(trade: Dict[str, Any],
                     thresholds: Optional[RegimeThresholds] = None) -> str:
        """静态方法: 单笔分类(需预知阈值或自动拟合)"""
        clf = RegimeClassifier()
        if thresholds is not None:
            clf.thresholds = thresholds
            clf.adx_threshold = thresholds.adx_threshold
        return clf.classify(trade)

    def get_distribution(self, trades: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取regime分布统计

        Args:
            trades: 交易记录列表

        Returns:
            {regime: count} 字典
        """
        regimes = self.classify_all(trades)
        dist = {"trending": 0, "ranging": 0, "high_vol": 0, "low_vol": 0}
        for r in regimes:
            dist[r] = dist.get(r, 0) + 1
        return dist

    def verify_coverage(self, trades: List[Dict[str, Any]],
                        min_trades_per_regime: int = 15) -> Dict[str, Any]:
        """验证4regime覆盖率

        Args:
            trades: 交易记录列表
            min_trades_per_regime: 每个regime最少交易笔数(默认15)

        Returns:
            覆盖率报告字典
        """
        dist = self.get_distribution(trades)
        required = ["trending", "ranging", "high_vol", "low_vol"]

        report = {
            "distribution": dist,
            "required_regimes": required,
            "missing_regimes": [r for r in required if dist.get(r, 0) == 0],
            "insufficient_regimes": [
                r for r in required if 0 < dist.get(r, 0) < min_trades_per_regime
            ],
            "full_coverage": all(dist.get(r, 0) >= min_trades_per_regime for r in required),
            "min_trades_threshold": min_trades_per_regime,
            "thresholds": {
                "adx_threshold": self.thresholds.adx_threshold if self.thresholds else 25.0,
                "vol_low": self.thresholds.vol_low_threshold if self.thresholds else 0.01,
                "vol_high": self.thresholds.vol_high_threshold if self.thresholds else 0.03,
                "vol_metric": self.thresholds.vol_metric if self.thresholds else "unknown",
                "is_quantile_based": self.thresholds.is_quantile_based if self.thresholds else False,
            },
        }
        return report

    def _self_test(self) -> bool:
        """自检: 验证分类器基本功能"""
        try:
            # 构造测试数据: 4种regime共12笔(≥10触发分位数模式)
            test_trades = [
                # trending: ADX>25
                {"adx": 30, "atr_pct": 0.02, "pnl_pct": 0.01},
                {"adx": 40, "atr_pct": 0.015, "pnl_pct": 0.015},
                {"adx": 50, "atr_pct": 0.025, "pnl_pct": 0.02},
                {"adx": 35, "atr_pct": 0.018, "pnl_pct": 0.012},
                # high_vol: ADX≤25, vol>q75
                {"adx": 20, "atr_pct": 0.05, "pnl_pct": 0.05},
                {"adx": 15, "atr_pct": 0.06, "pnl_pct": 0.06},
                # low_vol: ADX≤25, vol<q25
                {"adx": 10, "atr_pct": 0.005, "pnl_pct": 0.005},
                {"adx": 5, "atr_pct": 0.003, "pnl_pct": 0.003},
                # ranging: ADX≤25, 中间vol
                {"adx": 20, "atr_pct": 0.02, "pnl_pct": 0.02},
                {"adx": 15, "atr_pct": 0.015, "pnl_pct": 0.015},
                {"adx": 22, "atr_pct": 0.017, "pnl_pct": 0.017},
                {"adx": 18, "atr_pct": 0.019, "pnl_pct": 0.019},
            ]

            regimes = self.classify_all(test_trades)
            dist = {"trending": 0, "ranging": 0, "high_vol": 0, "low_vol": 0}
            for r in regimes:
                dist[r] = dist.get(r, 0) + 1

            # 验证4个regime都有样本
            if all(dist.get(r, 0) > 0 for r in ["trending", "ranging", "high_vol", "low_vol"]):
                logger.info(f"RegimeClassifier自检通过: {dist}, 分位数模式={self.thresholds.is_quantile_based}")
                return True
            else:
                logger.error(f"RegimeClassifier自检失败: {dist}")
                return False
        except Exception as e:
            logger.error(f"RegimeClassifier自检异常: {e}")
            return False


def reclassify_trades_regimes(trades: List[Dict[str, Any]],
                              use_quantile: bool = True) -> Tuple[List[Dict[str, Any]], RegimeClassifier]:
    """重新分类交易列表的regime字段

    Args:
        trades: 交易记录列表(会被原地修改regime字段)
        use_quantile: 是否使用动态分位数阈值

    Returns:
        (trades, classifier) 重新分类后的交易列表和分类器
    """
    clf = RegimeClassifier(use_quantile=use_quantile)
    clf.fit(trades)

    for t in trades:
        old_regime = t.get("regime", "unknown")
        new_regime = clf.classify(t)
        t["regime"] = new_regime
        t["regime_prev"] = old_regime  # 保留旧regime用于对比

    return trades, clf


if __name__ == "__main__":
    # 自检
    clf = RegimeClassifier()
    result = clf._self_test()
    print(f"RegimeClassifier自检: {'PASS' if result else 'FAIL'}")
