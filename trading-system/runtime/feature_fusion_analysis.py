"""
FeatureFusionAnalysisMixin — Phase 7.4 架构解耦
从 evolution_loop.py 提取的特征融合分析逻辑。

设计:
  - Mixin 模式: 不继承任何基类,通过多重继承注入 EvolutionLoop
  - 鸭子类型依赖:
      _run_feature_fusion_analysis: self.engine.markets (只读), self.feature_fusion (只读)
      _synthesize_4h_klines: @staticmethod, 零 self 依赖
  - 零循环依赖: 仅依赖 logger + typing

来源:
  - v598 Phase 1 4维特征融合管道分析 (短板#2 补齐)
  - 4维: 凯利(双轨) + 价格行为(缠论) + 量价(VPIN+VolumeProfile) + 多周期(MTF Core+RegimeDetector)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FeatureFusionAnalysisMixin:
    """特征融合分析管理 Mixin (Phase 7.4 提取)

    Host 鸭子类型依赖:
      - self.engine.markets: Dict[str, market] (需有 history/bars 属性)
      - self.feature_fusion: 需有 extract() 和 get_pipeline_output() 方法
    """

    def _run_feature_fusion_analysis(self) -> Dict[str, Any]:
        """v598 Phase 1: 运行 4维特征融合管道分析 (短板#2 补齐)

        从 engine.markets 获取每个 symbol 的 klines, 提取融合特征:
          - 凯利(双轨): DynamicAdaptiveKelly
          - 价格行为: ChanlunPriceActionSystem
          - 量价: VPIN + VolumeProfile
          - 多周期: MTF Core + RegimeDetector

        Returns:
            Dict: 含 n_symbols_analyzed, avg_composite_score, low_confidence_symbols,
                  per_symbol_report
        """
        report: Dict[str, Any] = {
            "n_symbols_analyzed": 0,
            "avg_composite_score": 0.0,
            "low_confidence_symbols": [],
            "per_symbol_report": {},
        }
        if not hasattr(self, 'engine') or not hasattr(self.engine, 'markets') or not self.engine.markets:
            return report

        _composite_scores: List[float] = []
        for symbol, market in self.engine.markets.items():
            try:
                history = getattr(market, 'history', None) or getattr(market, 'bars', None)
                if not history or len(history) < 50:
                    continue
                klines_1h = list(history[-200:])  # 取最近 200 根
                klines_4h = self._synthesize_4h_klines(klines_1h)

                # 提取 4维融合特征
                self.feature_fusion.extract(
                    klines_1h=klines_1h,
                    klines_4h=klines_4h,
                    klines_15m=None,
                    symbol=symbol,
                    trade_history=None,
                )
                output = self.feature_fusion.get_pipeline_output()

                if output and not output.get("error"):
                    composite = float(output.get("composite_score", 0.0))
                    _composite_scores.append(composite)
                    report["per_symbol_report"][symbol] = {
                        "composite_score": composite,
                        "pa_signal": float(output.get("pa_signal", 0.0)),
                        "vp_signal": float(output.get("vp_signal", 0.0)),
                        "mtf_signal": float(output.get("mtf_signal", 0.0)),
                        "kelly_fraction": float(output.get("kelly_fraction", 0.0)),
                        "confidence": float(output.get("confidence", 0.0)),
                        "confluence_tier": output.get("confluence_tier", "none"),
                    }
                    # v92 Phase 2: 移除 composite_score 低置信度判定 (d=-0.240 方向错误, ERR-20260701-v88fp)
                    # low_confidence_symbols 列表保留在 report 结构中供向后兼容, 但不再基于 composite_score 填充
                    # 替代信号 (hold_bars d=+0.461) 留到 v93 — 用户铁律: "不要通过已知数据反推策略"
                    report["n_symbols_analyzed"] += 1
            except Exception as e:
                logger.warning("FeatureFusionPipeline symbol=%s failed: %s", symbol, str(e)[:80])

        if _composite_scores:
            report["avg_composite_score"] = sum(_composite_scores) / len(_composite_scores)

        return report

    @staticmethod
    def _synthesize_4h_klines(klines_1h: List[Dict]) -> List[Dict]:
        """从 1h K线合成 4h K线 (每 4 根 1h 合成 1 根 4h)

        Args:
            klines_1h: 1小时K线列表

        Returns:
            4小时K线列表
        """
        if len(klines_1h) < 4:
            return []
        klines_4h: List[Dict] = []
        # 找到第一个 4h 对齐点 (timestamp % 14400 == 0)
        start_idx = 0
        for i, k in enumerate(klines_1h):
            ts = k.get("timestamp", 0)
            if ts and ts % (4 * 3600) == 0:
                start_idx = i
                break
        for i in range(start_idx, len(klines_1h) - 3, 4):
            group = klines_1h[i:i+4]
            k4 = {
                "timestamp": group[0].get("timestamp", 0),
                "open": group[0].get("open", 0),
                "high": max(g.get("high", 0) for g in group),
                "low": min(g.get("low", 0) for g in group),
                "close": group[-1].get("close", 0),
                "volume": sum(g.get("volume", 0) for g in group),
            }
            klines_4h.append(k4)
        return klines_4h
