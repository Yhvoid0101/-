#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v515 交易知识储备体系 — Agent-7 进化引擎的知识基础设施
================================================================
设计目标 (用户铁律):
  - "全面深入的利用好交易知识储备体系"
  - "确保策略能够完美灵活组合利用所有知识和指标"

模块构成:
  1. 指标知识库 (15+ 指标): ADX, RSI, BB, ATR, Z-score, Hurst, MA,
     OBV, VWAP, Ichimoku, MACD, Stochastic, CCI, Williams%R, MFI
  2. 策略模板库 (10+ 策略): 动量, 均值回归, 突破, 套利, 网格,
     趋势跟随, 反转, 摆动, 波动率突破, 趋势回调
  3. 市场状态→策略映射 (知识检索): 根据 ADX/Hurst/ATR%/BB_width
     自动推荐最适策略组合
  4. 进化记忆 (JSON 持久化): 记录历史最优基因, 避免重复探索

用法:
  from _v515_knowledge_base import KnowledgeBase
  kb = KnowledgeBase()
  kb.load_memory()
  rec = kb.recommend_strategies(regime="RANGING", atr_pct=0.012, hurst=0.42)
  kb.save_best_gene(gene, fitness=2.31, kpi=5)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# ============================================================
# 1. 指标知识库 (15 种指标)
# ============================================================
@dataclass
class IndicatorSpec:
    """单个指标的知识描述 (静态元数据, 非计算结果)"""
    name: str
    category: str                # trend / momentum / volatility / volume / composite
    period_default: int
    description: str
    signal_meaning: Dict[str, str]   # 取值含义 (教学型知识)
    best_for_regimes: List[str]      # 该指标在哪些 regime 最有效
    pitfalls: List[str]             # 已知缺陷/误用陷阱


INDICATOR_LIBRARY: Dict[str, IndicatorSpec] = {
    # --- 趋势类 ---
    "ADX": IndicatorSpec(
        name="ADX", category="trend", period_default=14,
        description="平均趋向指数, 衡量趋势强度 (不论方向)",
        signal_meaning={">25": "强趋势", "20-25": "趋势形成", "<20": "无趋势/震荡"},
        best_for_regimes=["TRENDING"],
        pitfalls=["滞后指标, 趋势结束后仍高", "不区分多空方向"],
    ),
    "MA": IndicatorSpec(
        name="MA", category="trend", period_default=755,
        description="移动平均线, 趋势方向与支撑/阻力",
        signal_meaning={"价格>MA": "多头", "价格<MA": "空头", "MA上行": "上升趋势"},
        best_for_regimes=["TRENDING"],
        pitfalls=["严重滞后", "震荡市频繁假突破"],
    ),
    "MACD": IndicatorSpec(
        name="MACD", category="trend", period_default=12,
        description="异同移动平均线, 趋势动量与背离",
        signal_meaning={"金叉": "做多", "死叉": "做空", "背离": "反转预警"},
        best_for_regimes=["TRENDING"],
        pitfalls=["震荡市频繁假信号", "滞后"],
    ),
    "Ichimoku": IndicatorSpec(
        name="Ichimoku", category="composite", period_default=26,
        description="一目均衡表, 趋势+支撑/阻力+动量综合",
        signal_meaning={"价格>云层": "多头", "价格<云层": "空头", "云层颜色翻转": "趋势反转"},
        best_for_regimes=["TRENDING"],
        pitfalls=["参数多, 优化过拟合风险高", "在加密1h级别噪音大"],
    ),

    # --- 动量类 ---
    "RSI": IndicatorSpec(
        name="RSI", category="momentum", period_default=14,
        description="相对强弱指数, 超买超卖与背离",
        signal_meaning={"<30": "超卖", ">70": "超买", "50附近": "中性"},
        best_for_regimes=["RANGING", "NEUTRAL"],
        pitfalls=["强趋势中超买/超卖可长期维持", "需配合趋势过滤"],
    ),
    "Stochastic": IndicatorSpec(
        name="Stochastic", category="momentum", period_default=14,
        description="随机振荡器, 价格在区间内位置",
        signal_meaning={"<20": "超卖", ">80": "超买", "K上穿D": "金叉"},
        best_for_regimes=["RANGING"],
        pitfalls=["趋势市频繁假信号"],
    ),
    "CCI": IndicatorSpec(
        name="CCI", category="momentum", period_default=20,
        description="商品通道指数, 价格偏离均值的程度",
        signal_meaning={">100": "超买", "<-100": "超卖", "回穿±100": "反转"},
        best_for_regimes=["RANGING", "NEUTRAL"],
        pitfalls=["强趋势中持续超买/超卖"],
    ),
    "WilliamsR": IndicatorSpec(
        name="WilliamsR", category="momentum", period_default=14,
        description="威廉指标, 收盘价在最高最低区间的位置",
        signal_meaning={"<-80": "超卖", ">-20": "超买"},
        best_for_regimes=["RANGING"],
        pitfalls=["与Stochastic高度相关, 不要重复使用"],
    ),
    "MFI": IndicatorSpec(
        name="MFI", category="volume", period_default=14,
        description="资金流量指数, 带成交量的RSI",
        signal_meaning={"<20": "超卖", ">80": "超买"},
        best_for_regimes=["RANGING", "NEUTRAL"],
        pitfalls=["需要成交量数据", "加密永续合约成交量可能有失真"],
    ),

    # --- 波动率类 ---
    "ATR": IndicatorSpec(
        name="ATR", category="volatility", period_default=14,
        description="真实波幅均值, 波动率与止损/仓位 sizing",
        signal_meaning={"ATR%>3%": "高波动", "ATR%<1%": "低波动", "ATR上升": "波动扩张"},
        best_for_regimes=["VOLATILE_COMPRESSION", "HIGH_VOLATILITY"],
        pitfalls=["不指示方向", "需归一化为ATR%使用"],
    ),
    "BB": IndicatorSpec(
        name="BB", category="volatility", period_default=20,
        description="布林带, 波动率边界与压缩/扩张",
        signal_meaning={"触上轨": "超买", "触下轨": "超卖", "带宽收窄": "压缩待突破"},
        best_for_regimes=["RANGING", "VOLATILE_COMPRESSION"],
        pitfalls=["触轨不一定是反转 (强趋势中持续贴轨)"],
    ),
    "Z-score": IndicatorSpec(
        name="Z-score", category="volatility", period_default=20,
        description="标准分, 价格偏离均值的标准差数",
        signal_meaning={">2": "显著高估", "<-2": "显著低估", "回归0": "均值回归"},
        best_for_regimes=["RANGING", "NEUTRAL"],
        pitfalls=["非正态分布时Z-score失真", "强趋势中持续偏高/低"],
    ),
    "Hurst": IndicatorSpec(
        name="Hurst", category="composite", period_default=200,
        description="Hurst指数, 趋势持续性 (反持续性)",
        signal_meaning={">0.55": "持续性 (趋势)", "<0.45": "反持续 (均值回归)", "≈0.5": "随机游走"},
        best_for_regimes=["TRENDING", "RANGING"],
        pitfalls=["需要长窗口 (≥200)", "对窗口选择敏感"],
    ),

    # --- 成交量类 ---
    "OBV": IndicatorSpec(
        name="OBV", category="volume", period_default=0,
        description="能量潮, 累积成交量与价格方向",
        signal_meaning={"OBV上行": "资金流入", "OBV下行": "资金流出", "背离": "反转预警"},
        best_for_regimes=["TRENDING"],
        pitfalls=["加密永续合约的成交量统计口径不一"],
    ),
    "VWAP": IndicatorSpec(
        name="VWAP", category="volume", period_default=0,
        description="成交量加权平均价, 机构基准价",
        signal_meaning={"价格>VWAP": "多头优势", "价格<VWAP": "空头优势"},
        best_for_regimes=["NEUTRAL", "RANGING"],
        pitfalls=["日内指标, 跨日需重置", "加密24h市场需自定义锚点"],
    ),
}


# ============================================================
# 2. 策略模板库 (10 种策略)
# ============================================================
@dataclass
class StrategyTemplate:
    """策略模板的知识描述"""
    name: str
    category: str                 # trend_following / mean_reversion / breakout / arbitrage / grid
    description: str
    required_indicators: List[str]
    best_regimes: List[str]
    typical_genes: Dict[str, Tuple[float, float]]  # 基因范围
    risk_profile: str             # 风险画像
    expected_frequency: str       # 交易频率


STRATEGY_LIBRARY: Dict[str, StrategyTemplate] = {
    # --- 趋势类 ---
    "momentum_breakout": StrategyTemplate(
        name="momentum_breakout", category="trend_following",
        description="通道突破 + MA 方向过滤, 趋势确立后顺势入场",
        required_indicators=["MA", "ADX", "ATR"],
        best_regimes=["TRENDING"],
        typical_genes={
            "breakout_n": (20, 80), "atr_stop_mult": (2.0, 5.0),
            "take_profit_r": (2.0, 4.0), "atr_trail_mult": (1.5, 3.5),
        },
        risk_profile="中等回撤, 高盈亏比, 低胜率",
        expected_frequency="低 (1-3笔/周)",
    ),
    "trend_pullback": StrategyTemplate(
        name="trend_pullback", category="trend_following",
        description="趋势中回调到 MA/EMA 后入场, 比 breakout 更优入场价",
        required_indicators=["MA", "ADX", "RSI"],
        best_regimes=["TRENDING"],
        typical_genes={
            "pullback_ma_period": (50, 200), "rsi_pullback_lo": (40, 50),
            "rsi_pullback_hi": (50, 60), "atr_stop_mult": (1.5, 3.0),
        },
        risk_profile="低回撤, 中盈亏比, 中胜率",
        expected_frequency="中 (2-5笔/周)",
    ),
    "ichimoku_trend": StrategyTemplate(
        name="ichimoku_trend", category="trend_following",
        description="一目均衡表趋势跟随, 价格穿越云层入场",
        required_indicators=["Ichimoku", "ADX"],
        best_regimes=["TRENDING"],
        typical_genes={
            "tenkan_period": (7, 14), "kijun_period": (20, 30),
            "senkou_b_period": (44, 60),
        },
        risk_profile="中回撤, 高盈亏比, 低胜率",
        expected_frequency="低",
    ),

    # --- 均值回归类 ---
    "mean_reversion_strict": StrategyTemplate(
        name="mean_reversion_strict", category="mean_reversion",
        description="BB+RSI+Z 三重确认均值回归, 严格过滤",
        required_indicators=["BB", "RSI", "Z-score", "ATR"],
        best_regimes=["RANGING"],
        typical_genes={
            "rsi_oversold": (25, 35), "rsi_overbought": (65, 75),
            "z_threshold": (1.5, 2.5), "mr_atr_stop_mult": (1.5, 3.0),
            "mr_take_profit_r": (1.0, 2.5),
        },
        risk_profile="低回撤, 中盈亏比, 高胜率",
        expected_frequency="中",
    ),
    "mean_reversion_relaxed": StrategyTemplate(
        name="mean_reversion_relaxed", category="mean_reversion",
        description="2-of-3 确认均值回归 (放宽版), 提高交易频率",
        required_indicators=["BB", "RSI", "Z-score"],
        best_regimes=["NEUTRAL", "RANGING"],
        typical_genes={
            "rsi_oversold": (25, 40), "rsi_overbought": (60, 75),
            "z_threshold": (1.0, 2.0), "mr_atr_stop_mult": (1.5, 3.0),
            "mr_take_profit_r": (1.0, 2.5),
        },
        risk_profile="中回撤, 中盈亏比, 中胜率",
        expected_frequency="高",
    ),
    "stochastic_reversion": StrategyTemplate(
        name="stochastic_reversion", category="mean_reversion",
        description="Stochastic+BB 均值回归, 适合深度震荡",
        required_indicators=["Stochastic", "BB", "ATR"],
        best_regimes=["RANGING"],
        typical_genes={
            "stoch_oversold": (15, 25), "stoch_overbought": (75, 85),
            "atr_stop_mult": (1.5, 2.5),
        },
        risk_profile="中回撤, 低盈亏比, 高胜率",
        expected_frequency="高",
    ),

    # --- 突破类 ---
    "bb_compression_breakout": StrategyTemplate(
        name="bb_compression_breakout", category="breakout",
        description="BB 带宽压缩后突破, 捕捉波动率扩张",
        required_indicators=["BB", "ATR"],
        best_regimes=["VOLATILE_COMPRESSION"],
        typical_genes={
            "bo_atr_stop_mult": (1.5, 3.0), "bo_take_profit_r": (2.0, 4.0),
            "bb_compression_ratio": (0.5, 0.7),
        },
        risk_profile="中回撤, 高盈亏比, 低胜率",
        expected_frequency="低 (压缩不常有)",
    ),
    "donchian_breakout": StrategyTemplate(
        name="donchian_breakout", category="breakout",
        description="唐奇安通道突破 (海龟策略核心)",
        required_indicators=["MA", "ATR"],
        best_regimes=["VOLATILE_COMPRESSION", "TRENDING"],
        typical_genes={
            "donchian_period": (20, 55), "atr_stop_mult": (2.0, 4.0),
        },
        risk_profile="中回撤, 高盈亏比, 低胜率",
        expected_frequency="低",
    ),

    # --- 套利/网格类 ---
    "grid_trading": StrategyTemplate(
        name="grid_trading", category="grid",
        description="网格挂单, 震荡市稳定获利",
        required_indicators=["BB", "ATR"],
        best_regimes=["RANGING", "NEUTRAL"],
        typical_genes={
            "grid_spacing_atr": (0.5, 1.5), "grid_count": (5, 15),
        },
        risk_profile="低回撤, 低盈亏比, 高胜率",
        expected_frequency="极高",
    ),
    "volatility_arbitrage": StrategyTemplate(
        name="volatility_arbitrage", category="arbitrage",
        description="ATR 与隐含波动率差套利 (单边简化版: ATR均值回归)",
        required_indicators=["ATR", "BB"],
        best_regimes=["NEUTRAL", "VOLATILE_COMPRESSION"],
        typical_genes={
            "atr_ma_period": (50, 200), "atr_z_threshold": (1.5, 2.5),
        },
        risk_profile="低回撤, 中盈亏比, 中胜率",
        expected_frequency="中",
    ),
}


# ============================================================
# 3. 市场 regime → 策略推荐 (知识检索)
# ============================================================
REGIME_STRATEGY_MAP: Dict[str, List[str]] = {
    "TRENDING":               ["momentum_breakout", "trend_pullback", "ichimoku_trend"],
    "RANGING":                ["mean_reversion_strict", "mean_reversion_relaxed", "stochastic_reversion", "grid_trading"],
    "NEUTRAL":                ["mean_reversion_relaxed", "volatility_arbitrage", "grid_trading"],
    "VOLATILE_COMPRESSION":   ["bb_compression_breakout", "donchian_breakout"],
    "HIGH_VOLATILITY":        [],  # 观望, 不交易 (风险控制)
}


# ============================================================
# 4. 进化记忆 (JSON 持久化)
# ============================================================
@dataclass
class GeneMemoryEntry:
    """单条历史最优基因记忆"""
    gene_id: str
    genes: Dict[str, float]
    fitness: float
    objectives: Dict[str, float]   # 5 目标值
    kpi_pass: int
    generation: int
    timestamp: float
    notes: str = ""


class KnowledgeBase:
    """
    交易知识储备体系 — 整合指标库 + 策略库 + regime映射 + 进化记忆
    """

    def __init__(self, memory_path: Optional[Path] = None):
        self.indicators = INDICATOR_LIBRARY
        self.strategies = STRATEGY_LIBRARY
        self.regime_map = REGIME_STRATEGY_MAP

        # 进化记忆持久化路径
        if memory_path is None:
            here = Path(__file__).parent.resolve()
            memory_path = here / "_v515_evolution_memory.json"
        self.memory_path = memory_path
        self.memory: List[GeneMemoryEntry] = []
        self._gene_hash_index: Dict[str, int] = {}  # 基因哈希 -> memory 索引

    # ---------- 知识检索 ----------

    def list_indicators(self, category: Optional[str] = None) -> List[IndicatorSpec]:
        if category is None:
            return list(self.indicators.values())
        return [s for s in self.indicators.values() if s.category == category]

    def list_strategies(self, category: Optional[str] = None) -> List[StrategyTemplate]:
        if category is None:
            return list(self.strategies.values())
        return [s for s in self.strategies.values() if s.category == category]

    def recommend_strategies(
        self,
        regime: str,
        atr_pct: Optional[float] = None,
        hurst: Optional[float] = None,
        adx: Optional[float] = None,
    ) -> List[StrategyTemplate]:
        """
        根据市场状态自动推荐最适策略组合 (知识检索核心)

        参数:
          regime: TRENDING/RANGING/NEUTRAL/VOLATILE_COMPRESSION/HIGH_VOLATILITY
          atr_pct: 当前 ATR% (波动率参考)
          hurst: 当前 Hurst 值 (持续性参考)
          adx: 当前 ADX 值 (趋势强度参考)

        返回:
          排序后的策略模板列表 (最适在前)
        """
        candidates = self.regime_map.get(regime, [])
        if not candidates:
            return []

        ranked: List[Tuple[int, StrategyTemplate]] = []
        for name in candidates:
            tpl = self.strategies.get(name)
            if tpl is None:
                continue
            score = 0
            # regime 匹配加分
            if regime in tpl.best_regimes:
                score += 10
            # Hurst 微调: 趋势策略在 Hurst>0.55 时加分
            if hurst is not None:
                if tpl.category == "trend_following" and hurst > 0.55:
                    score += 3
                if tpl.category == "mean_reversion" and hurst < 0.45:
                    score += 3
            # ADX 微调
            if adx is not None:
                if tpl.category == "trend_following" and adx > 25:
                    score += 2
                if tpl.category in ("mean_reversion", "grid") and adx < 20:
                    score += 2
            # ATR% 微调: 高波动时倾向突破/趋势, 低波动倾向均值回归
            if atr_pct is not None:
                if atr_pct > 0.025 and tpl.category == "breakout":
                    score += 2
                if atr_pct < 0.015 and tpl.category == "mean_reversion":
                    score += 2
            ranked.append((score, tpl))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked]

    def gene_template_for(self, strategy_name: str) -> Dict[str, Tuple[float, float]]:
        """返回策略模板的基因范围 (用于初始化种群)"""
        tpl = self.strategies.get(strategy_name)
        if tpl is None:
            return {}
        return dict(tpl.typical_genes)

    # ---------- 进化记忆 (持久化) ----------

    @staticmethod
    def _hash_gene(genes: Dict[str, float]) -> str:
        # 把基因四舍五入到 2 位后哈希, 避免微小差异导致重复
        rounded = {k: round(v, 2) for k, v in sorted(genes.items())}
        return json.dumps(rounded, sort_keys=True)

    def load_memory(self) -> None:
        """从 JSON 加载历史进化记忆"""
        if not self.memory_path.exists():
            self.memory = []
            self._gene_hash_index = {}
            return
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            self.memory = [GeneMemoryEntry(**e) for e in data.get("entries", [])]
            self._gene_hash_index = {
                self._hash_gene(e.genes): i for i, e in enumerate(self.memory)
            }
        except Exception as e:
            print(f"  [KnowledgeBase] 加载记忆失败: {e}")
            self.memory = []
            self._gene_hash_index = {}

    def save_memory(self) -> None:
        """持久化进化记忆到 JSON"""
        data = {
            "version": "v515-kb-1.0",
            "updated": time.time(),
            "entry_count": len(self.memory),
            "entries": [asdict(e) for e in self.memory],
        }
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_best_gene(
        self,
        genes: Dict[str, float],
        fitness: float,
        objectives: Dict[str, float],
        kpi_pass: int,
        generation: int,
        notes: str = "",
    ) -> bool:
        """
        保存一条历史最优基因 (避免重复探索)

        返回:
          True = 新增; False = 已存在 (跳过)
        """
        ghash = self._hash_gene(genes)
        if ghash in self._gene_hash_index:
            # 已存在: 若新结果更好则更新
            idx = self._gene_hash_index[ghash]
            existing = self.memory[idx]
            if fitness > existing.fitness:
                existing.fitness = fitness
                existing.objectives = objectives
                existing.kpi_pass = kpi_pass
                existing.generation = generation
                existing.timestamp = time.time()
                existing.notes = notes
            return False

        entry = GeneMemoryEntry(
            gene_id=f"gene_{len(self.memory)+1:04d}",
            genes=genes, fitness=fitness, objectives=objectives,
            kpi_pass=kpi_pass, generation=generation,
            timestamp=time.time(), notes=notes,
        )
        self.memory.append(entry)
        self._gene_hash_index[ghash] = len(self.memory) - 1
        return True

    def similar_gene_exists(self, genes: Dict[str, float], tolerance: float = 0.05) -> bool:
        """
        检查是否存在相似基因 (避免重复探索, 容忍 5% 偏差)
        用于进化时跳过已探索区域
        """
        for e in self.memory:
            diff_count = 0
            total_keys = 0
            for k, v in genes.items():
                if k not in e.genes:
                    continue
                total_keys += 1
                if total_keys == 0:
                    break
                ref = e.genes[k]
                if abs(ref) > 1e-9:
                    rel_diff = abs(v - ref) / abs(ref)
                else:
                    rel_diff = abs(v - ref)
                if rel_diff > tolerance:
                    diff_count += 1
            if total_keys > 0 and diff_count / total_keys < 0.2:
                # 80% 基因相似 → 视为已探索
                return True
        return False

    def top_genes(self, n: int = 5) -> List[GeneMemoryEntry]:
        """返回历史最优的 n 个基因"""
        return sorted(self.memory, key=lambda e: e.fitness, reverse=True)[:n]

    # ---------- 报告 ----------

    def summary(self) -> Dict[str, Any]:
        return {
            "indicator_count": len(self.indicators),
            "strategy_count": len(self.strategies),
            "memory_entries": len(self.memory),
            "categories": {
                "indicators": sorted(set(s.category for s in self.indicators.values())),
                "strategies": sorted(set(s.category for s in self.strategies.values())),
            },
            "regime_coverage": {
                r: len(self.regime_map.get(r, [])) for r in self.regime_map
            },
        }


# ============================================================
# 5. 自检 (模块导入时打印摘要)
# ============================================================
def _self_check() -> Dict[str, Any]:
    kb = KnowledgeBase()
    return kb.summary()


if __name__ == "__main__":
    import sys
    summary = _self_check()
    print("=" * 60)
    print("v515 交易知识储备体系 — 自检")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print("regime -> 策略推荐示例 (RANGING):")
    kb = KnowledgeBase()
    for tpl in kb.recommend_strategies("RANGING", atr_pct=0.012, hurst=0.42, adx=15):
        print(f"  - {tpl.name}: {tpl.description}")
