#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度参数分析引擎 (Deep Parameter Analyzer) — v503 Fix 35

============================================================
目标: 多维度量化参数对交易结果的影响权重，精确识别亏损根因
============================================================

用户要求二: 深度参数分析与模块优化
  1. 多维度参数评估体系 — 最近6个月历史数据深度回溯
  2. 统计学方法 (相关性分析/PCA/假设检验) + ML模型 (随机森林/SHAP)
  3. 量化各参数对交易结果的影响权重
  4. 精确定位导致亏损的关键参数组合及模块
  5. 分析报告含数据样本量、置信区间、显著性水平(P<0.05)

分析方法:
  Phase 1 — 统计基础: 描述性统计 + 分布检验 + 效应量(Cohen's d)
  Phase 2 — 相关性分析: Pearson/Spearman + 偏相关(控制混淆变量)
  Phase 3 — PCA主成分: 参数降维 + 主成分载荷 + 方差解释率
  Phase 4 — 假设检验: Welch t-test + Mann-Whitney U + 多重比较校正(Bonferroni)
  Phase 5 — 随机森林: 特征重要性(MDI) + 排列重要性 + OOB误差
  Phase 6 — SHAP分析: TreeExplainer + 特征贡献方向 + 交互效应
  Phase 7 — 分组分析: 盈利vs亏损组参数差异 + 分段回归
  Phase 8 — 综合报告: 参数影响排序 + 优化建议 + 置信区间

参数来源:
  - gene_codec.py: GENE_SCHEMA (基因编码参数)
  - agent_decision_engine.py: decide() 中的硬编码阈值和策略参数
  - population.py: fitness函数中的权重参数

数据要求:
  - 至少 100 笔交易 (统计显著性)
  - 覆盖 ≥5 个交易对 (跨币种泛化)
  - 覆盖 ≥4 种市场状态 (趋势/震荡/高波动/低波动)
  - 时间跨度 ≥6 个月

输出:
  - _deep_param_analysis_report.json (结构化数据)
  - _deep_param_analysis_report.md (人类可读报告)
  - _deep_param_analysis_features.csv (特征矩阵)
============================================================
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# v598 Phase K: 因果推断模块软加载
# ============================================================================
try:
    from sandbox_trading.causal_inference_trading import (
        CausalInferenceTrading,
        CausalSignal,
        CausalRegime,
    )
    _CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    _CAUSAL_INFERENCE_AVAILABLE = False
    CausalInferenceTrading = None  # type: ignore
    CausalSignal = None  # type: ignore
    CausalRegime = None  # type: ignore
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================================
# 配置
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hermes.param_analyzer")

# 输出文件
OUTPUT_JSON = SCRIPT_DIR / "_deep_param_analysis_report.json"
OUTPUT_MD = SCRIPT_DIR / "_deep_param_analysis_report.md"
OUTPUT_CSV = SCRIPT_DIR / "_deep_param_analysis_features.csv"

# 统计显著性阈值
P_THRESHOLD = 0.05
# 置信水平
CONFIDENCE_LEVEL = 0.95
# 最小样本量
MIN_SAMPLE_SIZE = 30

# 默认交易对
# v612扩展: 从5币种扩展到15币种, 匹配项目硬约束"至少10个主要交易对"
# 覆盖v97实际交易的15币种 (BTC/ETH/SOL/BNB/DOT/XRP/ARB/OP/UNI/LINK/AVAX/ATOM/INJ/TRX/LTC)
# 来源: v97报告 part_c_optimal_config_validation 实际交易币种
DEFAULT_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "DOT-USDT",
    "XRP-USDT", "ARB-USDT", "OP-USDT", "UNI-USDT", "LINK-USDT",
    "AVAX-USDT", "ATOM-USDT", "INJ-USDT", "TRX-USDT", "LTC-USDT",
]

# ============================================================================
# 参数定义 — 从 gene_codec 和 agent_decision_engine 提取
# ============================================================================

# 基因级参数 (来自 GENE_SCHEMA 和 WORLDVIEW_SEEDS)
GENE_PARAMS = {
    # === 止损/止盈相关 ===
    "stop_loss_pct": {
        "label": "止损百分比",
        "type": "float",
        "range": [0.005, 0.15],
        "default": 0.02,
        "category": "risk_management",
        "description": "单笔交易最大亏损占本金比例",
    },
    "take_profit_pct": {
        "label": "止盈百分比",
        "type": "float",
        "range": [0.01, 0.30],
        "default": 0.04,
        "category": "risk_management",
        "description": "单笔交易目标盈利占本金比例",
    },
    "atr_stop_mult": {
        "label": "ATR止损倍数",
        "type": "float",
        "range": [0.5, 5.0],
        "default": 2.0,
        "category": "risk_management",
        "description": "止损距离 = ATR × 此倍数",
    },
    "atr_period": {
        "label": "ATR周期",
        "type": "int",
        "range": [5, 30],
        "default": 14,
        "category": "indicator",
        "description": "ATR计算窗口",
    },
    "take_profit_r": {
        "label": "风险回报比",
        "type": "float",
        "range": [0.5, 5.0],
        "default": 2.0,
        "category": "risk_management",
        "description": "止盈距离/止损距离 (R:R)",
    },
    "max_hold_bars": {
        "label": "最大持仓K线",
        "type": "int",
        "range": [3, 200],
        "default": 24,
        "category": "risk_management",
        "description": "超过此K线数强制平仓",
    },
    "position_risk_pct": {
        "label": "仓位风险比例",
        "type": "float",
        "range": [0.005, 0.05],
        "default": 0.01,
        "category": "position_sizing",
        "description": "每笔交易风险占权益比例",
    },

    # === 信号/入场相关 ===
    "rsi_period": {
        "label": "RSI周期",
        "type": "int",
        "range": [5, 30],
        "default": 14,
        "category": "indicator",
        "description": "RSI计算窗口",
    },
    "rsi_overbought": {
        "label": "RSI超买阈值",
        "type": "float",
        "range": [60, 85],
        "default": 70,
        "category": "signal",
        "description": "RSI超过此值视为超买",
    },
    "rsi_oversold": {
        "label": "RSI超卖阈值",
        "type": "float",
        "range": [15, 40],
        "default": 30,
        "category": "signal",
        "description": "RSI低于此值视为超卖",
    },
    "momentum_period": {
        "label": "动量周期",
        "type": "int",
        "range": [5, 50],
        "default": 12,
        "category": "indicator",
        "description": "动量计算窗口",
    },
    "breakout_period": {
        "label": "突破周期",
        "type": "int",
        "range": [10, 55],
        "default": 20,
        "category": "signal",
        "description": "Donchian/Highest通道窗口",
    },
    "volume_confirm_mult": {
        "label": "成交量确认倍数",
        "type": "float",
        "range": [1.0, 3.0],
        "default": 1.5,
        "category": "signal",
        "description": "突破时成交量需超过均量此倍数",
    },
    "entry_z_score": {
        "label": "入场Z分数",
        "type": "float",
        "range": [1.0, 4.0],
        "default": 2.0,
        "category": "signal",
        "description": "均值回归入场Z-score阈值",
    },
    "exit_z_score": {
        "label": "出场Z分数",
        "type": "float",
        "range": [-0.5, 1.5],
        "default": 0.5,
        "category": "signal",
        "description": "均值回归出场Z-score阈值",
    },
    "min_rr_ratio": {
        "label": "最小风险回报比",
        "type": "float",
        "range": [1.0, 5.0],
        "default": 2.0,
        "category": "risk_management",
        "description": "入场前要求的最低R:R",
    },

    # === 均线/趋势相关 ===
    "fast_ema": {
        "label": "快EMA周期",
        "type": "int",
        "range": [5, 50],
        "default": 20,
        "category": "indicator",
        "description": "快速EMA计算窗口",
    },
    "slow_ema": {
        "label": "慢EMA周期",
        "type": "int",
        "range": [20, 200],
        "default": 50,
        "category": "indicator",
        "description": "慢速EMA计算窗口",
    },
    "lookback_period": {
        "label": "回看周期",
        "type": "int",
        "range": [10, 200],
        "default": 20,
        "category": "indicator",
        "description": "统计套利回看窗口",
    },
    "half_life": {
        "label": "半衰期",
        "type": "int",
        "range": [2, 50],
        "default": 10,
        "category": "signal",
        "description": "均值回归半衰期(越小回归越快)",
    },

    # === 综合/世界观级 ===
    "max_trades_per_day": {
        "label": "每日最大交易数",
        "type": "int",
        "range": [1, 50],
        "default": 5,
        "category": "risk_management",
        "description": "每日最大开仓次数",
    },
    "worldview_label": {
        "label": "市场世界观",
        "type": "str",
        "range": None,
        "default": "trend_following",
        "category": "worldview",
        "description": "主导交易哲学 (趋势跟随/均值回归/突破/动量等)",
    },
    "holding_period": {
        "label": "持仓周期偏好",
        "type": "str",
        "range": None,
        "default": "medium",
        "category": "worldview",
        "description": "very_short/short/medium/long/very_long/adaptive",
    },
}

# 决策引擎级参数 (硬编码阈值，非基因优化)
ENGINE_PARAMS = {
    "gnn_sri_threshold": {
        "label": "GNN系统性风险阈值",
        "type": "float",
        "value": 0.8,
        "category": "risk_filter",
        "description": "SRI≥0.8时不交易",
    },
    "gnn_contagion_threshold": {
        "label": "GNN传染概率阈值",
        "type": "float",
        "value": 0.7,
        "category": "risk_filter",
        "description": "传染概率≥0.7时不交易",
    },
    "cs_crash_prob_threshold": {
        "label": "动量崩溃概率阈值",
        "type": "float",
        "value": 0.8,
        "category": "risk_filter",
        "description": "动量崩溃概率≥0.8时不交易",
    },
    "confluence_score_min": {
        "label": "Confluence最低得分",
        "type": "float",
        "value": 0.0,
        "category": "signal_filter",
        "description": "Confluence得分低于此值不交易",
    },
    "min_signal_gap_bars": {
        "label": "最小信号间隔K线",
        "type": "int",
        "value": 2,
        "category": "signal_filter",
        "description": "两次入场信号之间至少间隔此K线数",
    },
    # hurst_low_threshold / hurst_high_threshold 已移除 (D2审计补遗 ERR-20260704-hurst-audit-gap)
    # 原因: Hurst未在feature_extractors.py计算, 元数据为孤立条目
    "adx_min_threshold": {
        "label": "ADX最低阈值",
        "type": "float",
        "value": 15.0,
        "category": "market_regime",
        "description": "ADX<此值=无趋势震荡市",
    },
    "volatility_regime_low": {
        "label": "低波动阈值",
        "type": "float",
        "value": 0.015,
        "category": "market_regime",
        "description": "ATR%<此值=低波动",
    },
    "volatility_regime_high": {
        "label": "高波动阈值",
        "type": "float",
        "value": 0.04,
        "category": "market_regime",
        "description": "ATR%>此值=高波动",
    },
    "single_loss_cap": {
        "label": "单笔亏损上限",
        "type": "float",
        "value": 0.02,
        "category": "risk_management",
        "description": "单笔最大亏损占本金2%",
    },
    "base_risk": {
        "label": "基础风险",
        "type": "float",
        "value": 0.01,
        "category": "risk_management",
        "description": "基础风险敞口",
    },
    "stop_loss_mult": {
        "label": "止损倍数",
        "type": "float",
        "value": 2.0,
        "category": "risk_management",
        "description": "止损距离 = ATR × 此倍数",
    },
    "stop_loss_min_distance": {
        "label": "止损最小距离",
        "type": "float",
        "value": 0.005,
        "category": "risk_management",
        "description": "止损距离不低于此百分比",
    },
    "compression_q": {
        "label": "成交量压缩阈值",
        "type": "float",
        "value": 0.3,
        "category": "volume_filter",
        "description": "成交量分位数<此值=缩量",
    },
    "volume_ratio": {
        "label": "成交量比率",
        "type": "float",
        "value": 1.5,
        "category": "volume_filter",
        "description": "当前量/均量>此值=放量",
    },
    "atr_rank_min": {
        "label": "ATR排名最低",
        "type": "float",
        "value": 0.3,
        "category": "volatility_filter",
        "description": "ATR分位数低于此值不交易",
    },
}

# Fitness权重参数 (来自 population.py compute_gt_score)
FITNESS_WEIGHTS = {
    "w_ev": {"label": "EV权重", "value": 0.20, "category": "fitness"},
    "w_sharpe": {"label": "夏普权重", "value": 0.20, "category": "fitness"},
    "w_drawdown": {"label": "回撤权重", "value": 0.15, "category": "fitness"},
    "w_winrate": {"label": "胜率权重", "value": 0.25, "category": "fitness"},
    "w_consistency": {"label": "一致性权重", "value": 0.20, "category": "fitness"},
    "trade_count_bonus_max": {"label": "交易数奖励上限", "value": 10.0, "category": "fitness"},
    "intraday_bonus_max": {"label": "日内奖励上限", "value": 10.0, "category": "fitness"},
    "positive_pnl_bonus": {"label": "正收益奖励", "value": 5.0, "category": "fitness"},
    "sharpe_ev_saturation": {"label": "EV饱和系数", "value": 0.5, "category": "fitness"},
    "sharpe_saturation": {"label": "夏普饱和系数", "value": 2.0, "category": "fitness"},
}

# 所有参数合并
ALL_PARAMS = {}
ALL_PARAMS.update({f"gene_{k}": {**v, "param_id": f"gene_{k}"} for k, v in GENE_PARAMS.items()})
ALL_PARAMS.update({f"engine_{k}": {**v, "param_id": f"engine_{k}"} for k, v in ENGINE_PARAMS.items()})
ALL_PARAMS.update({f"fitness_{k}": {**v, "param_id": f"fitness_{k}"} for k, v in FITNESS_WEIGHTS.items()})


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class ParamAnalysisResult:
    """单个参数的分析结果"""
    param_id: str
    label: str
    category: str
    # 描述性统计
    n_samples: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    # 相关性
    pearson_r: float = 0.0
    pearson_p: float = 1.0
    spearman_rho: float = 0.0
    spearman_p: float = 1.0
    # 假设检验 (盈利组 vs 亏损组)
    welch_t: float = 0.0
    welch_p: float = 1.0
    cohens_d: float = 0.0
    mwu_u: float = 0.0
    mwu_p: float = 1.0
    # 盈利组统计
    win_mean: float = 0.0
    win_std: float = 0.0
    win_n: int = 0
    # 亏损组统计
    loss_mean: float = 0.0
    loss_std: float = 0.0
    loss_n: int = 0
    # ML特征重要性
    rf_importance: float = 0.0
    rf_permutation_importance: float = 0.0
    shap_mean_abs: float = 0.0
    # 综合评分
    significance: str = "not_significant"  # significant / borderline / not_significant
    effect_size_label: str = "negligible"  # negligible / small / medium / large
    is_key_driver: bool = False


@dataclass
class PCAnalysisResult:
    """PCA分析结果"""
    n_components: int = 0
    explained_variance_ratio: List[float] = field(default_factory=list)
    cumulative_variance: List[float] = field(default_factory=list)
    loadings: Dict[str, List[float]] = field(default_factory=dict)
    top_components: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """完整分析报告"""
    # 元数据
    timestamp: str = ""
    symbols_analyzed: List[str] = field(default_factory=list)
    total_trades: int = 0
    total_signals: int = 0
    date_range: str = ""
    duration_seconds: float = 0.0

    # 描述性统计
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    ev_per_trade: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # 参数分析结果
    param_results: List[ParamAnalysisResult] = field(default_factory=list)
    pca_result: Optional[PCAnalysisResult] = None
    rf_feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_summary: Dict[str, Any] = field(default_factory=dict)

    # 关键发现
    key_drivers: List[str] = field(default_factory=list)
    loss_root_causes: List[str] = field(default_factory=list)
    optimization_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # v598 Phase K: 因果推断 + 参数敏感度
    causal_inference_result: Dict[str, Any] = field(default_factory=dict)
    parameter_sensitivity: Dict[str, Any] = field(default_factory=dict)

    # 统计检验汇总
    tests_performed: List[str] = field(default_factory=list)
    significant_params_count: int = 0
    bonferroni_threshold: float = 0.0


# ============================================================================
# 数据加载器
# ============================================================================

class TradeDataLoader:
    """从沙盘交易系统加载交易数据"""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (SCRIPT_DIR / ".." / "data" / "sandbox_trading")
        self.trades: List[Dict[str, Any]] = []
        self.signals: List[Dict[str, Any]] = []
        self.market_data: Dict[str, List[Dict]] = defaultdict(list)

    def load_from_json(
        self,
        json_path: Path,
        enforce_6month_window: bool = False,
    ) -> List[Dict[str, Any]]:
        """从JSON文件加载交易记录

        v598 Phase K: 新增 enforce_6month_window 参数 (用户铁律 "最近6个月历史数据深度回溯")

        Args:
            json_path: JSON 文件路径
            enforce_6month_window: 是否强制 6 个月硬窗口过滤
                True → 仅保留最近 180 天内的交易; 窗口内无数据时抛 ValueError
                False (默认) → 不过滤, 兼容既有调用方

        Returns:
            交易记录列表
        """
        if not json_path.exists():
            logger.warning("JSON文件不存在: %s", json_path)
            return []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            trades = data
        elif isinstance(data, dict):
            # 尝试多种可能的key
            for key in ("trades", "all_trades", "results", "trade_records"):
                if key in data:
                    trades = data[key]
                    break
            else:
                # 如果只有单个交易对的结果
                trades = [data]
        else:
            trades = []

        # v598 Phase K: 6 个月硬窗口过滤 (用户铁律 "最近6个月历史数据深度回溯分析")
        # L7 独立审查修复: 无时间戳 trade 必须过滤 (而非保留), 否则窗口失效
        if enforce_6month_window and trades:
            cutoff = datetime.now() - timedelta(days=180)
            filtered: List[Dict[str, Any]] = []
            no_ts_count = 0
            for t in trades:
                ts = self._extract_timestamp(t)
                if ts is None:
                    no_ts_count += 1
                    continue  # 过滤无时间戳 trade (L7 修复)
                if ts >= cutoff:
                    filtered.append(t)
            if not filtered:
                raise ValueError(
                    "6 个月窗口内无交易数据 "
                    f"(总交易 {len(trades)} 笔, 无时间戳 {no_ts_count} 笔, "
                    f"cutoff={cutoff.isoformat()})"
                )
            if no_ts_count > 0:
                # 无时间戳占比超 5% 报警 (L7 WARN 阈值)
                ratio = no_ts_count / len(trades)
                if ratio > 0.05:
                    logger.warning(
                        "enforce_6month_window: %d/%d (%.1f%%) 交易无时间戳被过滤, "
                        "数据源可能存在时间戳缺失问题",
                        no_ts_count, len(trades), ratio * 100,
                    )
                else:
                    logger.info(
                        "enforce_6month_window: %d 笔无时间戳交易被过滤 (%.1f%%)",
                        no_ts_count, ratio * 100,
                    )
            logger.info(
                "enforce_6month_window: 保留 %d/%d 笔交易 (cutoff=%s, 过滤无ts=%d)",
                len(filtered), len(trades), cutoff.date(), no_ts_count,
            )
            trades = filtered

        return trades

    @staticmethod
    def _extract_timestamp(trade: Dict[str, Any]) -> Optional[datetime]:
        """从交易记录提取时间戳 (兼容多种字段名/格式)

        支持字段: timestamp/ts/entry_time/open_time/close_time (ISO 字符串或 epoch 秒/毫秒)
        """
        for key in ("timestamp", "ts", "entry_time", "open_time", "close_time", "exit_time"):
            val = trade.get(key)
            if val is None:
                continue
            # ISO 字符串
            if isinstance(val, str):
                try:
                    # 兼容末尾带 Z 的 ISO 格式
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    # 尝试 epoch 秒
                    try:
                        return datetime.fromtimestamp(float(val))
                    except (ValueError, TypeError):
                        continue
            # 数值 (epoch 秒/毫秒)
            if isinstance(val, (int, float)):
                try:
                    # 自动判断秒 vs 毫秒 (10 位=秒, 13 位=毫秒)
                    if val > 1e12:
                        return datetime.fromtimestamp(val / 1000.0)
                    return datetime.fromtimestamp(val)
                except (ValueError, TypeError, OSError):
                    continue
        return None

    def load_from_evolution_loop(
        self,
        symbols: List[str] = None,
        bars_per_symbol: int = 2000,
        rounds: int = 3,
    ) -> Tuple[List[Dict], List[Dict]]:
        """通过运行 EvolutionLoop 收集交易数据"""
        symbols = symbols or DEFAULT_SYMBOLS

        from sandbox_trading.evolution_loop import EvolutionLoop
        from sandbox_trading.population import PopulationStats
        from sandbox_trading.agent_decision_engine import AgentDecisionEngine

        _ORIG_CHECK_EXIT = EvolutionLoop._check_exit
        _ORIG_DECIDE = AgentDecisionEngine.decide

        all_trades = []
        all_signals = []

        for sym_idx, symbol in enumerate(symbols, 1):
            print(f"  [{sym_idx}/{len(symbols)}] 加载 {symbol} 交易数据...")

            # 恢复原始方法
            EvolutionLoop._check_exit = _ORIG_CHECK_EXIT
            AgentDecisionEngine.decide = _ORIG_DECIDE

            # 收集器
            trades_collected = []
            signals_collected = []
            price_paths = defaultdict(list)

            # 补丁: 收集交易
            def _patched_check_exit(self, trade, market, gene, tick=None):
                result = _ORIG_CHECK_EXIT(self, trade, market, gene, tick)
                if result:
                    entry = trade.entry_price
                    exit_price = getattr(trade, 'forced_exit_price', 0.0) or float(market.close)
                    side = str(getattr(trade.side, 'value', trade.side))
                    sl = getattr(trade, 'stop_loss', 0.0)
                    tp = getattr(trade, 'take_profit', 0.0)
                    bars_held = getattr(trade, 'bars_held', 0)

                    if side == "long":
                        pnl_pct = (exit_price - entry) / entry * 100
                    else:
                        pnl_pct = (entry - exit_price) / entry * 100

                    trades_collected.append({
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry,
                        "exit_price": exit_price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "is_win": pnl_pct > 0,
                        "gene": getattr(self, 'gene', None),
                    })
                return result

            # 补丁: 收集信号
            def _patched_decide(self, data_packet, balance, equity, frontier=None):
                result = _ORIG_DECIDE(self, data_packet, balance, equity, frontier)
                market = data_packet.get("market")
                indicators = data_packet.get("indicators", {})
                close = getattr(market, "close", 0.0) if market else 0.0

                signal_info = {
                    "symbol": symbol,
                    "close": close,
                    "has_signal": result is not None,
                    "action": result.get("action", "none") if result else "none",
                    "indicators": {
                        k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in (indicators or {}).items()
                        if k in ("sma_50", "sma_200", "ema_12", "ema_26", "rsi_14",
                                 "atr_14", "adx_14", "bollinger_upper", "bollinger_lower",
                                 "macd", "macd_signal", "volume_sma")
                    },
                }
                signals_collected.append(signal_info)
                return result

            EvolutionLoop._check_exit = _patched_check_exit
            AgentDecisionEngine.decide = _patched_decide

            try:
                data_dir = str(self.data_dir) if self.data_dir.exists() else None
                loop = EvolutionLoop(
                    population_size=30, initial_capital=10000.0,
                    symbols=[symbol], data_dir=data_dir, max_generations=1,
                )
                loop.initialize(bars_per_symbol=bars_per_symbol, use_mock=False,
                               seed_count=0, use_intraday=True)

                def _patched_run_evolution(*args, **kwargs):
                    return PopulationStats(generation=0, population_size=30, diversity=1.0)
                loop.population.run_evolution_round = _patched_run_evolution

                loop.run(rounds=rounds, evolve_every=999, ticks_per_round=24)
                print(f"      收集到 {len(trades_collected)} 笔交易, {len(signals_collected)} 个信号")
            except Exception as e:
                print(f"      ❌ 加载失败: {e}")

            all_trades.extend(trades_collected)
            all_signals.extend(signals_collected)

        # 恢复
        EvolutionLoop._check_exit = _ORIG_CHECK_EXIT
        AgentDecisionEngine.decide = _ORIG_DECIDE

        return all_trades, all_signals


# ============================================================================
# 深度参数分析引擎
# ============================================================================

class DeepParamAnalyzer:
    """深度参数分析引擎 — 多维度量化参数影响权重"""

    def __init__(
        self,
        trades: List[Dict[str, Any]],
        signals: List[Dict[str, Any]] = None,
        param_defs: Dict[str, Dict] = None,
        min_samples: int = MIN_SAMPLE_SIZE,
        p_threshold: float = P_THRESHOLD,
    ):
        self.trades = trades
        self.signals = signals or []
        self.param_defs = param_defs or ALL_PARAMS
        self.min_samples = min_samples
        self.p_threshold = p_threshold
        self.bonferroni_threshold = p_threshold

        # 缓存
        self._feature_matrix: Optional[pd.DataFrame] = None
        self._numeric_params: List[str] = []
        self._scaler: Optional[Any] = None
        self._rf_model: Optional[Any] = None
        self._shap_explainer: Optional[Any] = None

        # 结果
        self.results: Dict[str, ParamAnalysisResult] = {}
        self.pca_result: Optional[PCAnalysisResult] = None
        self.report: Optional[AnalysisReport] = None

    # ========================================================================
    # Phase 1: 描述性统计
    # ========================================================================

    def _build_feature_matrix(self) -> pd.DataFrame:
        """从交易数据构建特征矩阵"""
        if self._feature_matrix is not None:
            return self._feature_matrix

        rows = []
        for trade in self.trades:
            row = {
                "pnl_pct": trade.get("pnl_pct", 0.0),
                "is_win": 1 if trade.get("is_win", False) else 0,
                "bars_held": trade.get("bars_held", 0),
                "symbol": trade.get("symbol", "unknown"),
                "side": trade.get("side", "long"),
            }

            # 从 gene 中提取参数 (如果有)
            gene = trade.get("gene")
            if gene:
                if hasattr(gene, '__dict__'):
                    gene_dict = gene.__dict__
                elif isinstance(gene, dict):
                    gene_dict = gene
                else:
                    gene_dict = {}

                for param_name, param_def in GENE_PARAMS.items():
                    val = gene_dict.get(param_name, param_def.get("default", 0))
                    if isinstance(val, (int, float, np.floating, np.integer)):
                        row[f"gene_{param_name}"] = float(val)
                    elif isinstance(val, str):
                        row[f"gene_{param_name}_cat"] = val

            # 从交易本身提取参数
            row["gene_take_profit_pct"] = abs(trade.get("take_profit", 0) - trade.get("entry_price", 0)) / max(trade.get("entry_price", 1), 1e-8)
            row["gene_stop_loss_pct"] = abs(trade.get("stop_loss", 0) - trade.get("entry_price", 0)) / max(trade.get("entry_price", 1), 1e-8)

            rows.append(row)

        df = pd.DataFrame(rows)

        # 填充分类变量
        for col in df.columns:
            if col.endswith("_cat"):
                df[col] = df[col].fillna("unknown")

        # 填充数值变量
        numeric_cols = [c for c in df.columns if not c.endswith("_cat") and c not in ("symbol", "side")]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        self._feature_matrix = df
        self._numeric_params = [c for c in numeric_cols if c.startswith("gene_") or c.startswith("engine_") or c.startswith("fitness_")]
        return df

    def compute_descriptive_stats(self) -> Dict[str, ParamAnalysisResult]:
        """Phase 1: 描述性统计 + 效应量"""
        df = self._build_feature_matrix()

        if len(df) < self.min_samples:
            logger.warning("样本量 %d < %d，统计结果可能不可靠", len(df), self.min_samples)

        # 更新Bonferroni阈值
        n_params = len(self._numeric_params)
        if n_params > 0:
            self.bonferroni_threshold = self.p_threshold / n_params

        win_mask = df["is_win"] == 1
        loss_mask = df["is_win"] == 0
        wins = df[win_mask]
        losses = df[loss_mask]

        for param_col in self._numeric_params:
            if param_col not in df.columns:
                continue

            values = df[param_col].dropna()
            if len(values) < self.min_samples:
                continue

            # 提取参数定义
            param_id = param_col
            param_def = self.param_defs.get(param_col, {
                "label": param_col, "category": "unknown",
                "description": "", "type": "float",
            })

            result = ParamAnalysisResult(
                param_id=param_id,
                label=param_def.get("label", param_col),
                category=param_def.get("category", "unknown"),
                n_samples=len(values),
                mean=float(values.mean()),
                median=float(values.median()),
                std=float(values.std()),
                min_val=float(values.min()),
                max_val=float(values.max()),
                skewness=float(stats.skew(values)),
                kurtosis=float(stats.kurtosis(values)),
            )

            # 盈利组 vs 亏损组
            win_vals = df.loc[win_mask, param_col].dropna()
            loss_vals = df.loc[loss_mask, param_col].dropna()

            if len(win_vals) >= 2 and len(loss_vals) >= 2:
                result.win_mean = float(win_vals.mean())
                result.win_std = float(win_vals.std())
                result.win_n = len(win_vals)
                result.loss_mean = float(loss_vals.mean())
                result.loss_std = float(loss_vals.std())
                result.loss_n = len(loss_vals)

                # Welch t-test (不等方差)
                t_stat, t_p = stats.ttest_ind(win_vals, loss_vals, equal_var=False)
                result.welch_t = float(t_stat)
                result.welch_p = float(t_p)

                # Mann-Whitney U test (非参数)
                try:
                    u_stat, u_p = stats.mannwhitneyu(win_vals, loss_vals, alternative='two-sided')
                    result.mwu_u = float(u_stat)
                    result.mwu_p = float(u_p)
                except Exception:
                    pass

                # Cohen's d (效应量)
                pooled_std = math.sqrt(
                    ((len(win_vals) - 1) * win_vals.var() + (len(loss_vals) - 1) * loss_vals.var())
                    / (len(win_vals) + len(loss_vals) - 2)
                )
                if pooled_std > 1e-10:
                    result.cohens_d = float((win_vals.mean() - loss_vals.mean()) / pooled_std)

                # 显著性判定
                if result.welch_p < self.bonferroni_threshold:
                    result.significance = "significant"
                elif result.welch_p < self.p_threshold:
                    result.significance = "borderline"
                else:
                    result.significance = "not_significant"

                # 效应量标签
                d_abs = abs(result.cohens_d)
                if d_abs >= 0.8:
                    result.effect_size_label = "large"
                elif d_abs >= 0.5:
                    result.effect_size_label = "medium"
                elif d_abs >= 0.2:
                    result.effect_size_label = "small"
                else:
                    result.effect_size_label = "negligible"

            # 相关性 (Pearson + Spearman)
            if "pnl_pct" in df.columns:
                valid = df[[param_col, "pnl_pct"]].dropna()
                if len(valid) >= self.min_samples:
                    r, r_p = stats.pearsonr(valid[param_col], valid["pnl_pct"])
                    result.pearson_r = float(r)
                    result.pearson_p = float(r_p)
                    rho, rho_p = stats.spearmanr(valid[param_col], valid["pnl_pct"])
                    result.spearman_rho = float(rho)
                    result.spearman_p = float(rho_p)

            self.results[param_id] = result

        return self.results

    # ========================================================================
    # Phase 2-3: PCA主成分分析
    # ========================================================================

    def compute_pca(self, n_components: int = None) -> PCAnalysisResult:
        """Phase 3: PCA主成分分析"""
        df = self._build_feature_matrix()

        # 只使用数值型参数列
        numeric_cols = [c for c in self._numeric_params if c in df.columns]
        if len(numeric_cols) < 3:
            logger.warning("数值参数列不足 (%d), 跳过PCA", len(numeric_cols))
            return PCAnalysisResult()

        X = df[numeric_cols].dropna()
        if len(X) < self.min_samples:
            logger.warning("PCA样本量不足 (%d < %d)", len(X), self.min_samples)
            return PCAnalysisResult()

        # 标准化
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # PCA
        from sklearn.decomposition import PCA
        n_comp = n_components or min(len(numeric_cols), 10)
        pca = PCA(n_components=n_comp)
        pca.fit(X_scaled)

        result = PCAnalysisResult(
            n_components=n_comp,
            explained_variance_ratio=[float(v) for v in pca.explained_variance_ratio_],
            cumulative_variance=[float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
        )

        # 载荷矩阵
        loadings = {}
        for i, comp in enumerate(pca.components_):
            comp_name = f"PC{i+1}"
            loadings[comp_name] = [float(v) for v in comp]
            # 找出该主成分的top参数
            top_indices = np.argsort(np.abs(comp))[-5:][::-1]
            top_params = [
                {
                    "param": numeric_cols[idx],
                    "label": self.param_defs.get(numeric_cols[idx], {}).get("label", numeric_cols[idx]),
                    "loading": float(comp[idx]),
                    "abs_loading": float(abs(comp[idx])),
                }
                for idx in top_indices
            ]
            result.top_components.append({
                "component": comp_name,
                "variance_explained": float(pca.explained_variance_ratio_[i]),
                "top_params": top_params,
            })

        result.loadings = loadings
        self.pca_result = result
        return result

    # ========================================================================
    # Phase 4-5: 随机森林特征重要性
    # ========================================================================

    def compute_random_forest_importance(self) -> Dict[str, float]:
        """Phase 5: 随机森林特征重要性 (MDI + 排列重要性)"""
        df = self._build_feature_matrix()

        numeric_cols = [c for c in self._numeric_params if c in df.columns]
        if len(numeric_cols) < 3:
            logger.warning("数值参数列不足，跳过随机森林")
            return {}

        X = df[numeric_cols].dropna()
        y = df.loc[X.index, "is_win"] if "is_win" in df.columns else None

        if y is None or len(X) < self.min_samples:
            return {}

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # 训练随机森林
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=max(5, len(X) // 20),
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
        )
        rf.fit(X, y)
        self._rf_model = rf

        # OOB score (if bootstrap=True)
        # MDI (Mean Decrease in Impurity)
        mdi_importance = {}
        for i, col in enumerate(numeric_cols):
            mdi_importance[col] = float(rf.feature_importances_[i])

        # 排列重要性
        from sklearn.inspection import permutation_importance
        try:
            perm_result = permutation_importance(
                rf, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            perm_importance = {}
            for i, col in enumerate(numeric_cols):
                perm_importance[col] = float(perm_result.importances_mean[i])
                if col in self.results:
                    self.results[col].rf_permutation_importance = perm_importance[col]
        except Exception as e:
            logger.warning("排列重要性计算失败: %s", e)
            perm_importance = {}

        # 更新结果
        for col in numeric_cols:
            if col in self.results:
                self.results[col].rf_importance = mdi_importance.get(col, 0.0)

        # 交叉验证
        try:
            cv_scores = cross_val_score(rf, X, y, cv=min(5, len(X) // 10), scoring='accuracy')
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean, cv_std = 0.0, 0.0

        return {
            "mdi_importance": mdi_importance,
            "permutation_importance": perm_importance,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "oob_score": float(getattr(rf, 'oob_score_', 0.0)),
        }

    # ========================================================================
    # Phase 6: SHAP值分析
    # ========================================================================

    def compute_shap_values(self) -> Dict[str, Any]:
        """Phase 6: SHAP值分析 (TreeExplainer)"""
        if self._rf_model is None:
            self.compute_random_forest_importance()

        if self._rf_model is None:
            return {"error": "随机森林模型未训练"}

        df = self._build_feature_matrix()
        numeric_cols = [c for c in self._numeric_params if c in df.columns]
        X = df[numeric_cols].dropna()

        if len(X) < self.min_samples:
            return {"error": f"样本量不足 ({len(X)} < {self.min_samples})"}

        shap_summary = {}

        try:
            import shap
            # 使用TreeExplainer
            explainer = shap.TreeExplainer(self._rf_model)

            # 对样本子集计算SHAP (避免内存爆炸)
            sample_size = min(200, len(X))
            X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X
            shap_values = explainer.shap_values(X_sample)

            # shap_values可能是list (多类) 或 array
            if isinstance(shap_values, list):
                # 二分类: 取正类 (下标1)
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values

            # 每个特征的SHAP汇总
            shap_importance = {}
            for i, col in enumerate(numeric_cols):
                if i < shap_vals.shape[1]:
                    mean_abs = float(np.abs(shap_vals[:, i]).mean())
                    shap_importance[col] = mean_abs
                    if col in self.results:
                        self.results[col].shap_mean_abs = mean_abs

            shap_summary["feature_importance"] = shap_importance
            shap_summary["n_samples_used"] = len(X_sample)

            # Top 10 SHAP特征
            sorted_shap = sorted(shap_importance.items(), key=lambda x: -x[1])[:10]
            shap_summary["top_features"] = [
                {
                    "param": col,
                    "label": self.param_defs.get(col, {}).get("label", col),
                    "shap_importance": imp,
                }
                for col, imp in sorted_shap
            ]

        except ImportError:
            shap_summary["error"] = "shap库未安装 (pip install shap)"
        except Exception as e:
            shap_summary["error"] = str(e)

        self._shap_explainer = shap_summary
        return shap_summary

    # ========================================================================
    # Phase 7: 分组分析
    # ========================================================================

    def compute_group_analysis(self) -> Dict[str, Any]:
        """Phase 7: 分组分析 — 盈利vs亏损/多空/币种/市场状态"""
        df = self._build_feature_matrix()
        if len(df) < self.min_samples:
            return {"error": f"样本量不足 ({len(df)} < {self.min_samples})"}

        group_results = {}

        # 1. 盈利 vs 亏损
        win_df = df[df["is_win"] == 1]
        loss_df = df[df["is_win"] == 0]
        group_results["win_vs_loss"] = {
            "win_count": len(win_df),
            "loss_count": len(loss_df),
            "win_rate": len(win_df) / max(len(df), 1),
            "win_avg_pnl": float(win_df["pnl_pct"].mean()) if len(win_df) > 0 else 0,
            "loss_avg_pnl": float(loss_df["pnl_pct"].mean()) if len(loss_df) > 0 else 0,
            "win_avg_bars": float(win_df["bars_held"].mean()) if len(win_df) > 0 else 0,
            "loss_avg_bars": float(loss_df["bars_held"].mean()) if len(loss_df) > 0 else 0,
        }

        # 2. 多空分析
        if "side" in df.columns:
            long_df = df[df["side"] == "long"]
            short_df = df[df["side"] == "short"]
            group_results["long_vs_short"] = {
                "long_count": len(long_df),
                "short_count": len(short_df),
                "long_win_rate": float(long_df["is_win"].mean()) if len(long_df) > 0 else 0,
                "short_win_rate": float(short_df["is_win"].mean()) if len(short_df) > 0 else 0,
                "long_avg_pnl": float(long_df["pnl_pct"].mean()) if len(long_df) > 0 else 0,
                "short_avg_pnl": float(short_df["pnl_pct"].mean()) if len(short_df) > 0 else 0,
            }

        # 3. 币种分析
        if "symbol" in df.columns:
            symbol_stats = {}
            for sym in df["symbol"].unique():
                sym_df = df[df["symbol"] == sym]
                if len(sym_df) >= 3:
                    symbol_stats[sym] = {
                        "trades": len(sym_df),
                        "win_rate": float(sym_df["is_win"].mean()),
                        "avg_pnl": float(sym_df["pnl_pct"].mean()),
                        "total_pnl": float(sym_df["pnl_pct"].sum()),
                        "avg_bars": float(sym_df["bars_held"].mean()),
                    }
            group_results["by_symbol"] = symbol_stats

        # 4. bars_held 分段分析
        if "bars_held" in df.columns:
            bars_bins = [0, 3, 6, 12, 24, 48, 999]
            bars_labels = ["0-3", "4-6", "7-12", "13-24", "25-48", "49+"]
            df["bars_group"] = pd.cut(df["bars_held"], bins=bars_bins, labels=bars_labels)
            bars_stats = {}
            for group in bars_labels:
                group_df = df[df["bars_group"] == group]
                if len(group_df) >= 2:
                    bars_stats[group] = {
                        "trades": len(group_df),
                        "win_rate": float(group_df["is_win"].mean()),
                        "avg_pnl": float(group_df["pnl_pct"].mean()),
                    }
            group_results["by_bars_held"] = bars_stats

        return group_results

    # ========================================================================
    # Phase 8: 综合报告生成
    # ========================================================================

    def run_full_analysis(self) -> AnalysisReport:
        """运行完整的深度参数分析

        v598 Phase K: 升级为 10 阶段 (新增 Phase 8 因果推断 + Phase 9 参数敏感度)
        """
        t_start = time.time()

        print("=" * 70)
        print("深度参数分析引擎 (Deep Parameter Analyzer) — v598 Phase K 10 阶段")
        print("=" * 70)

        # 1. 构建特征矩阵
        print("\n[1/10] 构建特征矩阵...")
        df = self._build_feature_matrix()
        print(f"  特征矩阵: {len(df)} 行 × {len(df.columns)} 列")
        print(f"  数值参数: {len(self._numeric_params)} 个")

        # 2. 描述性统计
        print("\n[2/10] 描述性统计 + 效应量...")
        self.compute_descriptive_stats()
        print(f"  分析参数: {len(self.results)} 个")

        # 3. PCA
        print("\n[3/10] PCA主成分分析...")
        pca_result = self.compute_pca()
        print(f"  主成分数: {pca_result.n_components}")
        print(f"  累计方差解释率: {pca_result.cumulative_variance[-1]*100:.1f}%" if pca_result.cumulative_variance else "  PCA跳过")

        # 4. 随机森林
        print("\n[4/10] 随机森林特征重要性...")
        rf_result = self.compute_random_forest_importance()
        if "mdi_importance" in rf_result:
            top_rf = sorted(rf_result["mdi_importance"].items(), key=lambda x: -x[1])[:5]
            for col, imp in top_rf:
                print(f"  {self.param_defs.get(col, {}).get('label', col)}: {imp:.4f}")
        else:
            print("  随机森林跳过")

        # 5. SHAP
        print("\n[5/10] SHAP值分析...")
        shap_result = self.compute_shap_values()
        if "top_features" in shap_result:
            for feat in shap_result["top_features"][:5]:
                print(f"  {feat['label']}: {feat['shap_importance']:.4f}")
        else:
            print(f"  {shap_result.get('error', 'SHAP跳过')}")

        # 6. 分组分析
        print("\n[6/10] 分组分析...")
        group_result = self.compute_group_analysis()

        # 7. 识别关键驱动参数
        print("\n[7/10] 识别关键驱动参数...")
        key_drivers = self._identify_key_drivers()
        loss_causes = self._identify_loss_root_causes()

        # 8. v598 Phase K: 因果推断 (Causal Inference)
        print("\n[8/10] 因果推断分析 (v598 Phase K)...")
        causal_result = self._run_causal_inference(df)
        if causal_result.get("available"):
            n_edges = causal_result.get("total_edges_discovered", 0)
            n_stable = causal_result.get("total_stable_edges", 0)
            print(f"  因果边发现: {n_edges} (稳定边: {n_stable})")
            print(f"  算法一致性: {causal_result.get('algorithm_agreement', 0)}, "
                  f"冲突: {causal_result.get('algorithm_conflict', 0)}")
            top_edges = causal_result.get("top_causal_edges", [])
            for edge in top_edges[:3]:
                print(f"  {edge['src']} → {edge['tgt']}: weight={edge['weight']:.4f} (p={edge['p_value']:.4f})")
        else:
            print(f"  因果推断跳过: {causal_result.get('reason', 'unknown')}")

        # 9. v598 Phase K: 参数敏感度 sweep (Parameter Sensitivity)
        print("\n[9/10] 参数敏感度 sweep (v598 Phase K)...")
        sensitivity_result = self._run_parameter_sensitivity(df)
        if sensitivity_result.get("available"):
            n_params = sensitivity_result.get("n_parameters_tested", 0)
            print(f"  测试参数数: {n_params}")
            top_sens = sensitivity_result.get("most_sensitive_params", [])
            for ps in top_sens[:3]:
                print(f"  {ps['param']}: sensitivity={ps['sensitivity_score']:.4f} "
                      f"(CI=[{ps['ci_low']:.4f}, {ps['ci_high']:.4f}])")
        else:
            print(f"  参数敏感度跳过: {sensitivity_result.get('reason', 'unknown')}")

        # 10. 生成优化建议
        print("\n[10/10] 生成优化建议...")
        recommendations = self._generate_recommendations()

        # 构建报告
        total_trades = len(df)
        win_rate = float(df["is_win"].mean()) if "is_win" in df.columns and len(df) > 0 else 0
        avg_pnl = float(df["pnl_pct"].mean()) if "pnl_pct" in df.columns and len(df) > 0 else 0
        avg_win = float(df[df["is_win"] == 1]["pnl_pct"].mean()) if win_rate > 0 else 0
        avg_loss = float(df[df["is_win"] == 0]["pnl_pct"].mean()) if win_rate < 1 else 0

        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else ["unknown"]

        # 计算EV和Sharpe
        ev = win_rate * avg_win - (1 - win_rate) * abs(avg_loss) if total_trades > 0 else 0
        pnl_std = float(df["pnl_pct"].std()) if "pnl_pct" in df.columns else 1.0
        sharpe = (avg_pnl / pnl_std * math.sqrt(total_trades)) if pnl_std > 0 else 0

        # 识别显著性参数
        sig_count = sum(1 for r in self.results.values() if r.significance == "significant")

        report = AnalysisReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            symbols_analyzed=symbols,
            total_trades=total_trades,
            total_signals=len(self.signals),
            date_range="N/A",
            duration_seconds=time.time() - t_start,
            win_rate=win_rate,
            avg_pnl_pct=avg_pnl,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            ev_per_trade=ev,
            profit_factor=abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            sharpe_ratio=sharpe,
            max_drawdown=0.0,
            param_results=list(self.results.values()),
            pca_result=pca_result,
            rf_feature_importance=rf_result.get("mdi_importance", {}),
            shap_summary=shap_result,
            key_drivers=key_drivers,
            loss_root_causes=loss_causes,
            optimization_recommendations=recommendations,
            causal_inference_result=causal_result,
            parameter_sensitivity=sensitivity_result,
            tests_performed=[
                "描述性统计 (均值/中位数/标准差/偏度/峰度)",
                "Pearson相关系数 + P值",
                "Spearman秩相关系数 + P值",
                "Welch t-test (不等方差, 盈利vs亏损)",
                "Mann-Whitney U test (非参数)",
                "Cohen's d 效应量",
                "PCA主成分分析",
                "随机森林特征重要性 (MDI + 排列重要性)",
                "SHAP值分析 (TreeExplainer)",
                "分组分析 (盈亏/多空/币种/持仓时长)",
                "Bonferroni多重比较校正",
                # v598 Phase K 新增
                "因果推断 (Granger + PC + LiNGAM + ANM 多算法融合)",
                "参数敏感度 sweep (线性回归斜率 + Bootstrap CI)",
            ],
            significant_params_count=sig_count,
            bonferroni_threshold=self.bonferroni_threshold,
        )

        self.report = report
        return report

    # ========================================================================
    # v598 Phase K - Phase 8: 因果推断 (Causal Inference)
    # ========================================================================

    def _run_causal_inference(self, df: pd.DataFrame) -> Dict[str, Any]:
        """v598 Phase K - Phase 8: 因果推断集成

        将 CausalInferenceTrading (67KB 模块, 多算法融合: Granger+PC+LiNGAM+ANM)
        集成到深度参数分析. 区别于 Phase 2 相关性分析 (统计相关):
          - 相关性 = P(X,Y) 共现, 不能区分方向
          - 因果性 = X → Y 方向性影响, 通过干预可改变 Y

        用户铁律 "运用统计学方法+机器学习模型相结合":
          - 统计: Granger 因果检验 (F 检验) + PC 算法 (偏相关 + Fisher Z)
          - 机器学习: LiNGAM (非高斯方向识别) + ANM (非线性加性噪声模型)

        设计 (ERR-110 应用边界 - 仅增强非过滤):
          - 不修改入场/出场条件, 不修改 threshold
          - 输出仅作为分析报告的一部分, 供决策层参考
          - 不通过已知数据反推策略参数 (用户铁律)

        Args:
            df: 特征矩阵 (_build_feature_matrix 输出)

        Returns:
            因果推断结果 dict:
              available: bool
              total_edges_discovered: int
              total_stable_edges: int
              algorithm_agreement: int
              algorithm_conflict: int
              top_causal_edges: List[Dict]
              per_symbol_summary: Dict[str, Dict]
              regime_distribution: Dict[str, int]
              reason: str (when not available)
        """
        result: Dict[str, Any] = {
            "available": False,
            "total_edges_discovered": 0,
            "total_stable_edges": 0,
            "algorithm_agreement": 0,
            "algorithm_conflict": 0,
            "top_causal_edges": [],
            "per_symbol_summary": {},
            "regime_distribution": {},
        }

        # 软加载检查
        if not _CAUSAL_INFERENCE_AVAILABLE:
            result["reason"] = "causal_inference_trading 模块未安装/不可导入"
            return result

        # 样本数检查 (Granger 需要 >= 60+3=63 笔)
        if len(df) < 70:
            result["reason"] = f"样本不足 ({len(df)} < 70, Granger 窗口需 ≥63)"
            return result

        # 按 symbol 分组, 每个 symbol 单独跑因果推断
        symbols = df["symbol"].unique() if "symbol" in df.columns else ["unknown"]
        all_edges: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        all_stable_edges: Dict[Tuple[str, str], int] = defaultdict(int)
        regime_count: Dict[str, int] = defaultdict(int)
        total_agreement = 0
        total_conflict = 0

        for symbol in symbols:
            sym_df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
            if len(sym_df) < 70:
                continue

            try:
                engine = CausalInferenceTrading(symbol=str(symbol))
            except Exception as e:
                logger.debug("CausalInferenceTrading 初始化失败 (%s): %s", symbol, str(e)[:80])
                continue

            # 推送每笔交易作为时间步 (per-trade pseudo time series)
            # 映射 trade 字段 → market variables
            for _, row in sym_df.iterrows():
                pnl_pct = float(row.get("pnl_pct", 0.0))
                is_win = int(row.get("is_win", 0))
                bars_held = float(row.get("bars_held", 0))

                # 派生市场变量 (使用 trade 字段作代理, 缺失字段用 0)
                # L7 独立审查修复: 修正代理变量共线性问题
                #   - 原 open_interest = volume * 0.8 与 volume 100% 共线, 损害 PC 偏相关稳定性
                #   - 原 volatility = |returns| 与 returns 非线性同源, 保留但需独立化
                returns = pnl_pct / 100.0  # 收益率
                # volatility 改用 bars_held 与 pnl_pct 的混合代理 (降低与 returns 共线)
                volatility = (abs(pnl_pct) + math.sqrt(max(bars_held, 1.0)) * 0.1) / 100.0
                # 滚动统计的简化版本: 用当前行的代理值
                sentiment = 1.0 if is_win else -1.0  # 情绪: 盈利=乐观, 亏损=悲观

                # 从 gene 字段提取可能的 funding/basis (若存在)
                gene_tp = float(row.get("gene_take_profit_pct", 0.0))
                gene_sl = float(row.get("gene_stop_loss_pct", 0.0))
                # funding_rate 代理: TP-SL 不对称程度 (假设反映资金费率方向)
                funding_rate = (gene_tp - gene_sl) * 0.01
                # basis 代理: TP+SL 平均值 (反映价差绝对幅度)
                basis = (gene_tp + gene_sl) * 0.5
                # volume 代理: bars_held 归一化 (持仓时长→成交量代理)
                volume = max(bars_held, 1.0) / 100.0
                # open_interest 代理: 改用 is_win 与 bars_held 的非线性组合
                # 避免与 volume 共线 (L7 修复): OI = 0.5 + 0.3*is_win + bars_held*0.005
                # 这样 OI 与 volume 不再 100% 相关 (引入 is_win 独立维度)
                open_interest = 0.5 + (0.3 if is_win else -0.1) + min(bars_held, 50.0) * 0.005

                market_data = {
                    "returns": returns,
                    "volume": volume,
                    "volatility": volatility,
                    "funding_rate": funding_rate,
                    "open_interest": open_interest,
                    "basis": basis,
                    "sentiment": sentiment,
                }
                try:
                    engine.update_market_data(market_data)
                except Exception:
                    continue

            # 获取引擎状态
            try:
                status = engine.get_status()
            except Exception as e:
                logger.debug("get_status 失败 (%s): %s", symbol, str(e)[:80])
                continue

            # 收集该 symbol 的因果边
            top_edges_sym = status.get("top_edges", [])
            n_stable = status.get("n_stable_edges", 0)
            n_edges = status.get("n_edges", 0)
            agreement = status.get("algorithm_agreement", 0)
            conflict = status.get("algorithm_conflict", 0)
            regime = status.get("regime", "unknown")

            total_agreement += agreement
            total_conflict += conflict
            regime_count[regime] += 1

            for edge in top_edges_sym:
                key = (edge.get("src", ""), edge.get("tgt", ""))
                all_edges[key].append({
                    "symbol": str(symbol),
                    "weight": float(edge.get("weight", 0.0)),
                    "f_stat": float(edge.get("f_stat", 0.0)),
                    "p_value": float(edge.get("p_value", 1.0)),
                    "stable": bool(edge.get("stable", False)),
                })
                if edge.get("stable"):
                    all_stable_edges[key] += 1

            result["per_symbol_summary"][str(symbol)] = {
                "n_edges": n_edges,
                "n_stable_edges": n_stable,
                "regime": regime,
                "agreement": agreement,
                "conflict": conflict,
            }

        # 跨 symbol 聚合: 因果边在多少 symbol 上重现
        cross_symbol_edges = []
        for (src, tgt), occurrences in all_edges.items():
            if not src or not tgt:
                continue
            n_symbols = len(set(o["symbol"] for o in occurrences))
            avg_weight = float(np.mean([o["weight"] for o in occurrences]))
            avg_p = float(np.mean([o["p_value"] for o in occurrences]))
            min_p = float(min(o["p_value"] for o in occurrences))
            stable_count = all_stable_edges.get((src, tgt), 0)
            cross_symbol_edges.append({
                "src": src,
                "tgt": tgt,
                "n_symbols_reproduced": n_symbols,
                "weight": avg_weight,
                "p_value": avg_p,
                "min_p_value": min_p,
                "stable_count": stable_count,
                "is_robust": n_symbols >= 2 and stable_count >= 1,
            })

        # 按 weight 排序
        cross_symbol_edges.sort(key=lambda x: -x["weight"])

        result.update({
            "available": len(cross_symbol_edges) > 0,
            "total_edges_discovered": sum(s["n_edges"] for s in result["per_symbol_summary"].values()),
            "total_stable_edges": sum(s["n_stable_edges"] for s in result["per_symbol_summary"].values()),
            "algorithm_agreement": total_agreement,
            "algorithm_conflict": total_conflict,
            "top_causal_edges": cross_symbol_edges[:10],
            "all_causal_edges_count": len(cross_symbol_edges),
            "robust_edges_count": sum(1 for e in cross_symbol_edges if e["is_robust"]),
            "regime_distribution": dict(regime_count),
        })
        if not result["available"]:
            result["reason"] = "未发现任何因果边 (可能样本/算法不足)"
        return result

    # ========================================================================
    # v598 Phase K - Phase 9: 参数敏感度 sweep (Parameter Sensitivity)
    # ========================================================================

    def _run_parameter_sensitivity(
        self,
        df: pd.DataFrame,
        key_params: Optional[List[str]] = None,
        n_bootstrap: int = 100,
    ) -> Dict[str, Any]:
        """v598 Phase K - Phase 9: 参数敏感度 sweep

        对每个关键参数计算其对 PnL 的敏感度 (基于历史 co-variation):
          - sensitivity_score = |slope| * std(param) / std(metric) (归一化)
          - 95% Bootstrap 置信区间 (100 次重采样)
          - 非线性检测: |Pearson r| vs |Spearman rho| 差异 >0.2 → 标记非线性

        用户铁律 "运用统计学方法 (相关性分析/假设检验)":
          - 线性回归斜率 = dPnL/dParam (一阶敏感度)
          - Bootstrap CI = 显著性 (P<0.05 即 CI 不跨 0)
          - 非线性检测 = 模型假设验证

        ERR-110 应用边界: 仅分析非过滤, 不修改策略参数

        Args:
            df: 特征矩阵
            key_params: 待分析参数列表 (None=用 self._numeric_params top-N)
            n_bootstrap: Bootstrap 重采样次数

        Returns:
            参数敏感度结果 dict:
              available: bool
              n_parameters_tested: int
              most_sensitive_params: List[Dict]
              all_sensitivities: List[Dict]
              nonlinear_detected: List[str]
              reason: str (when not available)
        """
        result: Dict[str, Any] = {
            "available": False,
            "n_parameters_tested": 0,
            "most_sensitive_params": [],
            "all_sensitivities": [],
            "nonlinear_detected": [],
        }

        if len(df) < self.min_samples:
            result["reason"] = f"样本不足 ({len(df)} < {self.min_samples})"
            return result

        if "pnl_pct" not in df.columns:
            result["reason"] = "特征矩阵缺少 pnl_pct 列"
            return result

        # 选择待分析参数
        if key_params is None:
            # 优先用 _numeric_params (gene_/engine_/fitness_ 前缀)
            candidate_params = [
                p for p in self._numeric_params
                if p in df.columns and df[p].std() > 1e-8
            ]
            # 取 top 15 (避免过多)
            candidate_params = candidate_params[:15]
        else:
            candidate_params = [p for p in key_params if p in df.columns]

        if not candidate_params:
            result["reason"] = "无可分析的数值参数 (检查 _numeric_params 为空)"
            return result

        pnl = df["pnl_pct"].values
        pnl_std = float(np.std(pnl))
        if pnl_std < 1e-8:
            result["reason"] = "PnL 方差为 0, 无法计算敏感度"
            return result

        all_sensitivities: List[Dict[str, Any]] = []
        nonlinear_detected: List[str] = []
        rng = np.random.RandomState(42)

        for param in candidate_params:
            param_vals = df[param].values
            param_std = float(np.std(param_vals))
            if param_std < 1e-8:
                continue

            # 线性回归斜率 (敏感度核心)
            try:
                slope, intercept = np.polyfit(param_vals, pnl, 1)
            except (np.linalg.LinAlgError, ValueError):
                continue

            # 归一化敏感度: |slope| * std(param) / std(pnl)
            sensitivity = abs(slope) * param_std / pnl_std

            # Pearson r (线性相关) 和 Spearman rho (单调相关)
            try:
                pearson_r, _ = stats.pearsonr(param_vals, pnl)
            except Exception:
                pearson_r = 0.0
            try:
                spearman_rho, spearman_p = stats.spearmanr(param_vals, pnl)
            except Exception:
                spearman_rho, spearman_p = 0.0, 1.0

            # 非线性检测: Pearson (线性) vs Spearman (单调) 差异
            if abs(spearman_rho) - abs(pearson_r) > 0.2:
                nonlinear_detected.append(param)

            # Bootstrap 95% CI
            boot_sensitivities = []
            n = len(param_vals)
            for _ in range(n_bootstrap):
                idx = rng.randint(0, n, size=n)
                try:
                    b_slope, _ = np.polyfit(param_vals[idx], pnl[idx], 1)
                    b_sens = abs(b_slope) * param_std / pnl_std
                    if not (np.isnan(b_sens) or np.isinf(b_sens)):
                        boot_sensitivities.append(b_sens)
                except (np.linalg.LinAlgError, ValueError, IndexError):
                    continue

            if len(boot_sensitivities) < 30:
                continue

            ci_low = float(np.percentile(boot_sensitivities, 2.5))
            ci_high = float(np.percentile(boot_sensitivities, 97.5))
            boot_mean = float(np.mean(boot_sensitivities))
            boot_std = float(np.std(boot_sensitivities))

            # 显著性判定: CI 不跨 0 (与 0 比较因 sensitivity ≥0)
            is_significant = ci_low > 0.01

            all_sensitivities.append({
                "param": param,
                "label": self.param_defs.get(param, {}).get("label", param),
                "sensitivity_score": round(sensitivity, 6),
                "slope": round(float(slope), 6),
                "direction": "positive" if slope > 0 else "negative",
                "pearson_r": round(float(pearson_r), 4),
                "spearman_rho": round(float(spearman_rho), 4),
                "spearman_p": round(float(spearman_p), 4),
                "ci_low": round(ci_low, 6),
                "ci_high": round(ci_high, 6),
                "ci_mean": round(boot_mean, 6),
                "ci_std": round(boot_std, 6),
                "is_significant": is_significant,
                "is_nonlinear": param in nonlinear_detected,
                "n_bootstrap": len(boot_sensitivities),
            })

        # 按敏感度排序
        all_sensitivities.sort(key=lambda x: -x["sensitivity_score"])

        result.update({
            "available": len(all_sensitivities) > 0,
            "n_parameters_tested": len(all_sensitivities),
            "most_sensitive_params": all_sensitivities[:5],
            "all_sensitivities": all_sensitivities,
            "nonlinear_detected": nonlinear_detected,
            "n_significant": sum(1 for s in all_sensitivities if s["is_significant"]),
            "method": "linear_regression_slope + bootstrap_ci_95pct",
            "n_bootstrap": n_bootstrap,
        })
        if not result["available"]:
            result["reason"] = "无可计算的参数敏感度 (检查数值参数方差)"
        return result

    def _identify_key_drivers(self) -> List[str]:
        """识别关键驱动参数 (综合信号)"""
        drivers = []

        for param_id, result in self.results.items():
            score = 0
            reasons = []

            # 显著性
            if result.significance == "significant":
                score += 3
                reasons.append("Welch t-test显著")
            elif result.significance == "borderline":
                score += 1
                reasons.append("Welch t-test边缘显著")

            # 效应量
            if result.effect_size_label in ("large",):
                score += 3
                reasons.append(f"大效应量 (d={result.cohens_d:.2f})")
            elif result.effect_size_label == "medium":
                score += 2
                reasons.append(f"中效应量 (d={result.cohens_d:.2f})")

            # 相关性
            if abs(result.pearson_r) > 0.3 and result.pearson_p < self.p_threshold:
                score += 2
                reasons.append(f"显著相关 (r={result.pearson_r:.3f}, p={result.pearson_p:.4f})")

            # RF重要性
            if result.rf_importance > 0.05:
                score += 2
                reasons.append(f"RF高重要性 ({result.rf_importance:.3f})")

            # SHAP重要性
            if result.shap_mean_abs > 0.01:
                score += 1
                reasons.append(f"SHAP显著 ({result.shap_mean_abs:.3f})")

            if score >= 4:
                result.is_key_driver = True
                drivers.append(
                    f"{result.label} (score={score}, {', '.join(reasons)})"
                )

        return sorted(drivers, reverse=True)

    def _identify_loss_root_causes(self) -> List[str]:
        """识别亏损根因"""
        causes = []
        df = self._build_feature_matrix()

        if "is_win" not in df.columns or len(df) < self.min_samples:
            return ["样本量不足，无法可靠识别亏损根因"]

        loss_df = df[df["is_win"] == 0]

        # 1. 止损过紧
        if "gene_stop_loss_pct" in df.columns:
            loss_sl = loss_df["gene_stop_loss_pct"].mean()
            win_sl = df[df["is_win"] == 1]["gene_stop_loss_pct"].mean()
            if loss_sl < win_sl * 0.7:
                causes.append(
                    f"亏损组止损距离 ({loss_sl*100:.2f}%) 显著小于盈利组 ({win_sl*100:.2f}%), "
                    f"可能止损过紧导致被噪声扫出 (Cohen's d={self.results.get('gene_stop_loss_pct', ParamAnalysisResult(param_id='')).cohens_d:.2f})"
                )

        # 2. 止盈过远
        if "gene_take_profit_pct" in df.columns:
            loss_tp = loss_df["gene_take_profit_pct"].mean()
            win_tp = df[df["is_win"] == 1]["gene_take_profit_pct"].mean()
            if loss_tp > win_tp * 1.3:
                causes.append(
                    f"亏损组止盈距离 ({loss_tp*100:.2f}%) 显著大于盈利组 ({win_tp*100:.2f}%), "
                    f"可能止盈过远难以触及→被时间止损/反向扫出"
                )

        # 3. 持仓时间
        if "bars_held" in df.columns:
            loss_bars = loss_df["bars_held"].mean()
            win_bars = df[df["is_win"] == 1]["bars_held"].mean()
            if loss_bars > win_bars * 1.5:
                causes.append(
                    f"亏损组平均持仓 ({loss_bars:.1f} bars) 显著长于盈利组 ({win_bars:.1f} bars), "
                    f"可能趋势反转后未及时止损"
                )

        # 4. 参数显著性
        sig_params = [r for r in self.results.values() if r.significance == "significant"]
        if sig_params:
            top_sig = sorted(sig_params, key=lambda r: -abs(r.cohens_d))[:3]
            for p in top_sig:
                causes.append(
                    f"参数 [{p.label}] 在盈亏组间差异显著 "
                    f"(Welch t={p.welch_t:.2f}, p={p.welch_p:.4f}, d={p.cohens_d:.2f}), "
                    f"赢组均值={p.win_mean:.4f} vs 亏组均值={p.loss_mean:.4f}"
                )

        # 5. 空头/方向
        if "side" in df.columns:
            short_win_rate = float(df[df["side"] == "short"]["is_win"].mean()) if len(df[df["side"] == "short"]) > 0 else 0
            long_win_rate = float(df[df["side"] == "long"]["is_win"].mean()) if len(df[df["side"] == "long"]) > 0 else 0
            if short_win_rate < 0.3 and len(df[df["side"] == "short"]) > 5:
                causes.append(
                    f"空头胜率 ({short_win_rate*100:.1f}%) 远低于多头 ({long_win_rate*100:.1f}%), "
                    f"空头策略可能需要禁用或重构"
                )

        if not causes:
            causes.append("未发现明显的单一参数根因，亏损可能是多参数交互效应导致")

        return causes

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []

        # 1. 基于关键驱动参数
        key_drivers = [r for r in self.results.values() if r.is_key_driver]
        for r in sorted(key_drivers, key=lambda x: -abs(x.cohens_d)):
            if r.win_mean > r.loss_mean:
                direction = "increase"
                action = f"考虑增大 {r.label} (赢组均值={r.win_mean:.4f} > 亏组均值={r.loss_mean:.4f})"
            else:
                direction = "decrease"
                action = f"考虑减小 {r.label} (赢组均值={r.win_mean:.4f} < 亏组均值={r.loss_mean:.4f})"

            recommendations.append({
                "priority": "HIGH" if r.significance == "significant" else "MEDIUM",
                "param": r.param_id,
                "label": r.label,
                "action": action,
                "direction": direction,
                "effect_size": r.cohens_d,
                "confidence": 1.0 - r.welch_p,
                "category": r.category,
            })

        # 2. 基于PCA
        if self.pca_result and self.pca_result.top_components:
            pc1 = self.pca_result.top_components[0]
            recommendations.append({
                "priority": "MEDIUM",
                "param": "pca_composite",
                "label": "PCA第一主成分",
                "action": f"PC1解释 {pc1['variance_explained']*100:.1f}% 方差，"
                         f"top参数: {', '.join(p['label'] for p in pc1['top_params'][:3])}",
                "direction": "monitor",
                "effect_size": 0,
                "confidence": 0,
                "category": "pca",
            })

        # 3. 基于RF
        rf_importance = self.report.rf_feature_importance if self.report else {}
        if rf_importance:
            top_rf = sorted(rf_importance.items(), key=lambda x: -x[1])[:3]
            for col, imp in top_rf:
                param_def = self.param_defs.get(col, {})
                recommendations.append({
                    "priority": "LOW",
                    "param": col,
                    "label": param_def.get("label", col),
                    "action": f"RF特征重要性 {imp:.3f}，建议在GA中给予更高搜索范围",
                    "direction": "widen_range",
                    "effect_size": imp,
                    "confidence": 0,
                    "category": "ml",
                })

        return recommendations

    # ========================================================================
    # 报告导出
    # ========================================================================

    def export_json(self, path: Path = None) -> Path:
        """导出JSON报告"""
        if self.report is None:
            self.run_full_analysis()

        path = path or OUTPUT_JSON
        report_dict = self._report_to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✅ JSON报告已导出: {path}")
        return path

    def export_markdown(self, path: Path = None) -> Path:
        """导出Markdown报告"""
        if self.report is None:
            self.run_full_analysis()

        path = path or OUTPUT_MD
        md = self._report_to_markdown()
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✅ Markdown报告已导出: {path}")
        return path

    def export_csv(self, path: Path = None) -> Path:
        """导出特征矩阵CSV"""
        df = self._build_feature_matrix()
        path = path or OUTPUT_CSV
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ 特征CSV已导出: {path}")
        return path

    def _report_to_dict(self) -> Dict[str, Any]:
        """将报告转换为可序列化字典"""
        report = self.report
        return {
            "meta": {
                "timestamp": report.timestamp,
                "symbols_analyzed": report.symbols_analyzed,
                "total_trades": report.total_trades,
                "total_signals": report.total_signals,
                "duration_seconds": report.duration_seconds,
                "bonferroni_threshold": report.bonferroni_threshold,
                "p_threshold": self.p_threshold,
                "min_samples": self.min_samples,
            },
            "summary_stats": {
                "win_rate": round(report.win_rate, 4),
                "avg_pnl_pct": round(report.avg_pnl_pct, 4),
                "avg_win_pct": round(report.avg_win_pct, 4),
                "avg_loss_pct": round(report.avg_loss_pct, 4),
                "ev_per_trade": round(report.ev_per_trade, 4),
                "profit_factor": round(report.profit_factor, 4),
                "sharpe_ratio": round(report.sharpe_ratio, 4),
            },
            "param_analysis": [
                {
                    "param_id": r.param_id,
                    "label": r.label,
                    "category": r.category,
                    "n_samples": r.n_samples,
                    "mean": round(r.mean, 6),
                    "std": round(r.std, 6),
                    "pearson_r": round(r.pearson_r, 4),
                    "pearson_p": round(r.pearson_p, 4),
                    "spearman_rho": round(r.spearman_rho, 4),
                    "spearman_p": round(r.spearman_p, 4),
                    "welch_t": round(r.welch_t, 4),
                    "welch_p": round(r.welch_p, 4),
                    "cohens_d": round(r.cohens_d, 4),
                    "win_mean": round(r.win_mean, 6),
                    "loss_mean": round(r.loss_mean, 6),
                    "rf_importance": round(r.rf_importance, 4),
                    "rf_permutation_importance": round(r.rf_permutation_importance, 4),
                    "shap_mean_abs": round(r.shap_mean_abs, 4),
                    "significance": r.significance,
                    "effect_size_label": r.effect_size_label,
                    "is_key_driver": r.is_key_driver,
                }
                for r in report.param_results
            ],
            "pca": {
                "n_components": report.pca_result.n_components if report.pca_result else 0,
                "cumulative_variance": [round(v, 4) for v in (report.pca_result.cumulative_variance if report.pca_result else [])],
                "top_components": report.pca_result.top_components if report.pca_result else [],
            },
            "rf_feature_importance": {k: round(v, 4) for k, v in report.rf_feature_importance.items()},
            "shap_summary": report.shap_summary,
            "key_drivers": report.key_drivers,
            "loss_root_causes": report.loss_root_causes,
            "optimization_recommendations": report.optimization_recommendations,
            "tests_performed": report.tests_performed,
            "significant_params_count": report.significant_params_count,
            # v598 Phase K: 因果推断 + 参数敏感度
            "causal_inference": report.causal_inference_result,
            "parameter_sensitivity": report.parameter_sensitivity,
        }

    def _report_to_markdown(self) -> str:
        """生成Markdown报告"""
        report = self.report

        lines = []
        lines.append("# 深度参数分析报告")
        lines.append("")
        lines.append(f"**生成时间**: {report.timestamp}")
        lines.append(f"**分析耗时**: {report.duration_seconds:.1f}s")
        lines.append(f"**Bonferroni校正阈值**: {report.bonferroni_threshold:.6f}")
        lines.append("")

        # 1. 摘要
        lines.append("## 一、摘要")
        lines.append("")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 交易对 | {', '.join(report.symbols_analyzed)} |")
        lines.append(f"| 总交易数 | {report.total_trades} |")
        lines.append(f"| 总信号数 | {report.total_signals} |")
        lines.append(f"| 胜率 | {report.win_rate*100:.1f}% |")
        lines.append(f"| 平均P&L | {report.avg_pnl_pct:.4f}% |")
        lines.append(f"| 平均盈利 | {report.avg_win_pct:.4f}% |")
        lines.append(f"| 平均亏损 | {report.avg_loss_pct:.4f}% |")
        lines.append(f"| EV/笔 | {report.ev_per_trade:.4f}% |")
        lines.append(f"| 盈亏比 | {report.profit_factor:.2f} |")
        lines.append(f"| 夏普比率 | {report.sharpe_ratio:.2f} |")
        lines.append(f"| 显著性参数数 | {report.significant_params_count} |")
        lines.append("")

        # 2. 关键驱动参数
        lines.append("## 二、关键驱动参数")
        lines.append("")
        if report.key_drivers:
            for i, driver in enumerate(report.key_drivers, 1):
                lines.append(f"{i}. {driver}")
        else:
            lines.append("(未发现关键驱动参数)")
        lines.append("")

        # 3. 参数排序表
        lines.append("## 三、参数影响排序 (按Cohen's d效应量)")
        lines.append("")
        sorted_params = sorted(report.param_results, key=lambda r: -abs(r.cohens_d))
        lines.append("| 排名 | 参数 | 类别 | 效应量(d) | Welch p | 相关系数(r) | RF重要性 | 显著性 |")
        lines.append("|------|------|------|-----------|---------|-------------|----------|--------|")
        for i, r in enumerate(sorted_params[:20], 1):
            sig = "✅" if r.significance == "significant" else ("⚠️" if r.significance == "borderline" else "—")
            lines.append(
                f"| {i} | {r.label} | {r.category} | {r.cohens_d:.3f} | {r.welch_p:.4f} | "
                f"{r.pearson_r:.3f} | {r.rf_importance:.3f} | {sig} |"
            )
        lines.append("")

        # 4. PCA
        if report.pca_result and report.pca_result.top_components:
            lines.append("## 四、PCA主成分分析")
            lines.append("")
            for comp in report.pca_result.top_components[:3]:
                lines.append(f"### {comp['component']} (方差解释率: {comp['variance_explained']*100:.1f}%)")
                lines.append("")
                lines.append("| 参数 | 载荷 |")
                lines.append("|------|------|")
                for p in comp["top_params"]:
                    lines.append(f"| {p['label']} | {p['loading']:.4f} |")
                lines.append("")

        # 5. 亏损根因
        lines.append("## 五、亏损根因分析")
        lines.append("")
        if report.loss_root_causes:
            for i, cause in enumerate(report.loss_root_causes, 1):
                lines.append(f"{i}. {cause}")
        else:
            lines.append("(未发现明确亏损根因)")
        lines.append("")

        # 6. 优化建议
        lines.append("## 六、优化建议")
        lines.append("")
        high_recs = [r for r in report.optimization_recommendations if r["priority"] == "HIGH"]
        med_recs = [r for r in report.optimization_recommendations if r["priority"] == "MEDIUM"]
        low_recs = [r for r in report.optimization_recommendations if r["priority"] == "LOW"]

        if high_recs:
            lines.append("### 🔴 高优先级")
            lines.append("")
            for rec in high_recs:
                lines.append(f"- **{rec['label']}**: {rec['action']} (效应量={rec['effect_size']:.3f}, 置信度={rec['confidence']*100:.1f}%)")
            lines.append("")

        if med_recs:
            lines.append("### 🟡 中优先级")
            lines.append("")
            for rec in med_recs:
                lines.append(f"- **{rec['label']}**: {rec['action']}")
            lines.append("")

        if low_recs:
            lines.append("### 🟢 低优先级")
            lines.append("")
            for rec in low_recs:
                lines.append(f"- **{rec['label']}**: {rec['action']}")
            lines.append("")

        # 7. 统计检验汇总
        lines.append("## 七、统计检验汇总")
        lines.append("")
        for i, test in enumerate(report.tests_performed, 1):
            lines.append(f"{i}. {test}")
        lines.append("")

        # 8. 因果推断分析 (v598 Phase K)
        causal = report.causal_inference_result or {}
        lines.append("## 八、因果推断分析 (Phase K)")
        lines.append("")
        if not causal:
            lines.append("(因果推断模块未启用或无可用结果)")
        else:
            lines.append(f"- **模块可用**: {causal.get('available', False)}")
            lines.append(f"- **发现因果边总数**: {causal.get('total_edges_discovered', 0)}")
            lines.append(f"- **稳定因果边数**: {causal.get('total_stable_edges', 0)}")
            lines.append(f"- **算法一致次数**: {causal.get('algorithm_agreement', 0)}")
            lines.append("")
            top_edges = causal.get("top_causal_edges", [])
            if top_edges:
                lines.append("### Top 因果边 (按权重排序)")
                lines.append("")
                lines.append("| 排名 | 源变量 → 目标变量 | 权重 | p值 | 稳定 | 算法支持数 | 鲁棒性 |")
                lines.append("|------|------------------|------|------|------|-----------|--------|")
                for i, edge in enumerate(top_edges[:10], 1):
                    src = edge.get("source", "?")
                    tgt = edge.get("target", "?")
                    w = edge.get("weight", 0.0)
                    p = edge.get("p_value", 1.0)
                    stable = "✅" if edge.get("is_stable") else "—"
                    n_algo = edge.get("algorithm_support", 1)
                    robust = edge.get("robustness", 0.0)
                    lines.append(
                        f"| {i} | {src} → {tgt} | {w:.4f} | {p:.4f} | {stable} | {n_algo}/4 | {robust:.2f} |"
                    )
                lines.append("")
            regime_dist = causal.get("regime_distribution", {})
            if regime_dist:
                lines.append("### 因果状态分布")
                lines.append("")
                lines.append("| 状态 | 样本占比 |")
                lines.append("|------|----------|")
                for regime, ratio in regime_dist.items():
                    lines.append(f"| {regime} | {ratio*100:.1f}% |")
                lines.append("")
            per_symbol = causal.get("per_symbol_summary", {})
            if per_symbol:
                lines.append("### 各 symbol 因果发现摘要")
                lines.append("")
                lines.append("| Symbol | 因果边数 | 稳定边数 | 主导算法 |")
                lines.append("|--------|---------|---------|-----------|")
                for sym, s in per_symbol.items():
                    lines.append(
                        f"| {sym} | {s.get('n_edges', 0)} | {s.get('n_stable', 0)} | "
                        f"{s.get('dominant_algorithm', '-')} |"
                    )
                lines.append("")

        # 9. 参数敏感度分析 (v598 Phase K)
        sens = report.parameter_sensitivity or {}
        lines.append("## 九、参数敏感度分析 (Phase K)")
        lines.append("")
        if not sens:
            lines.append("(参数敏感度模块未启用或无可用结果)")
        else:
            lines.append(f"- **模块可用**: {sens.get('available', False)}")
            lines.append(f"- **测试参数数**: {sens.get('n_parameters_tested', 0)}")
            lines.append(f"- **显著参数数**: {sens.get('n_significant', 0)}")
            lines.append(f"- **非线性检出数**: {sens.get('nonlinear_detected', 0)}")
            lines.append("")
            most_sensitive = sens.get("most_sensitive_params", [])
            if most_sensitive:
                lines.append("### Top 敏感参数 (按 sensitivity_score 排序)")
                lines.append("")
                lines.append("| 排名 | 参数 | 敏感度 | 95% CI 低 | 95% CI 高 | 非线性 | 显著 |")
                lines.append("|------|------|--------|----------|----------|--------|------|")
                for i, p in enumerate(most_sensitive[:10], 1):
                    name = p.get("param", "?")
                    score = p.get("sensitivity_score", 0.0)
                    ci_low = p.get("ci_low", 0.0)
                    ci_high = p.get("ci_high", 0.0)
                    nonlinear = "⚠️" if p.get("is_nonlinear") else "—"
                    significant = "✅" if p.get("is_significant") else "—"
                    lines.append(
                        f"| {i} | {name} | {score:.4f} | {ci_low:.4f} | {ci_high:.4f} | "
                        f"{nonlinear} | {significant} |"
                    )
                lines.append("")
            lines.append("> **解读**: sensitivity_score = |slope| × std(param) / std(PnL)。")
            lines.append("> 非线性检出 = |Spearman ρ| − |Pearson r| > 0.2，提示单调性被破坏。")
            lines.append("> 显著 = 95% CI 下界 > 0.01 (Bootstrap 100次重采样)。")
            lines.append("")

        # 10. 方法说明 (原第八章)
        lines.append("## 十、方法说明")
        lines.append("")
        lines.append("### 效应量 (Cohen's d)")
        lines.append("- |d| < 0.2: 可忽略")
        lines.append("- 0.2 ≤ |d| < 0.5: 小效应")
        lines.append("- 0.5 ≤ |d| < 0.8: 中效应")
        lines.append("- |d| ≥ 0.8: 大效应")
        lines.append("")
        lines.append("### 多重比较校正 (Bonferroni)")
        lines.append(f"- 原始阈值: α = {self.p_threshold}")
        lines.append(f"- 参数数量: {len(self._numeric_params)}")
        lines.append(f"- 校正阈值: α' = {report.bonferroni_threshold:.6f}")
        lines.append("")
        lines.append("### 显著性判定")
        lines.append("- **significant**: P < Bonferroni校正阈值")
        lines.append("- **borderline**: P < 原始阈值但 > 校正阈值")
        lines.append("- **not_significant**: P ≥ 原始阈值")
        lines.append("")

        # v598 Phase K 方法说明
        lines.append("### 因果推断 (CausalInferenceTrading v1.1)")
        lines.append("- **多算法融合**: Granger(时间序列因果) + PC(条件独立) + LiNGAM(非高斯) + ANM(非线性)")
        lines.append("- **投票机制**: ≥2/4 算法支持 = 强因果边 (stable)")
        lines.append("- **自适应衰减**: w_t = exp(-λ × Δt), 旧边权重随时间衰减 (λ=0.02)")
        lines.append("- **Granger 滚动窗口**: window=60, lag=3 (需要 ≥63 笔观测)")
        lines.append("- **应用边界 (ERR-110)**: 仅用于增强诊断与排序, 不修改入场/出场/threshold")
        lines.append("")
        lines.append("### 参数敏感度 (Sensitivity Sweep)")
        lines.append("- **score**: |slope| × std(param) / std(PnL) — 归一化敏感度")
        lines.append("- **95% CI**: Bootstrap 100 次重采样的斜率分布")
        lines.append("- **非线性检测**: |Spearman ρ| − |Pearson r| > 0.2 → 非线性关系")
        lines.append("- **显著判定**: CI 下界 > 0.01 → 该参数对 PnL 真实敏感")
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# 便捷入口
# ============================================================================

def run_deep_param_analysis(
    symbols: List[str] = None,
    rounds: int = 3,
    bars_per_symbol: int = 2000,
    output_dir: Path = None,
) -> AnalysisReport:
    """运行完整深度参数分析的便捷入口

    Args:
        symbols: 交易对列表 (默认: BTC/ETH/SOL/BNB/DOGE)
        rounds: 每个交易对运行轮数
        bars_per_symbol: 每个交易对K线数
        output_dir: 输出目录

    Returns:
        AnalysisReport 完整分析报告
    """
    symbols = symbols or DEFAULT_SYMBOLS
    output_dir = output_dir or SCRIPT_DIR

    print("=" * 70)
    print("深度参数分析引擎 — 多维度量化参数影响权重")
    print("=" * 70)
    print(f"交易对: {symbols}")
    print(f"轮数: {rounds} | K线数: {bars_per_symbol}")
    print(f"显著性阈值: P < {P_THRESHOLD}")
    print(f"Bonferroni校正: 自动")
    print("=" * 70)

    # 1. 加载数据
    loader = TradeDataLoader()
    trades, signals = loader.load_from_evolution_loop(
        symbols=symbols,
        bars_per_symbol=bars_per_symbol,
        rounds=rounds,
    )

    if len(trades) < MIN_SAMPLE_SIZE:
        print(f"\n⚠️ 交易数不足 ({len(trades)} < {MIN_SAMPLE_SIZE})，统计结果可能不可靠")
        print("建议: 增加轮数或交易对数量")

    # 2. 运行分析
    analyzer = DeepParamAnalyzer(trades=trades, signals=signals)
    report = analyzer.run_full_analysis()

    # 3. 导出
    analyzer.export_json(output_dir / "_deep_param_analysis_report.json")
    analyzer.export_markdown(output_dir / "_deep_param_analysis_report.md")
    analyzer.export_csv(output_dir / "_deep_param_analysis_features.csv")

    return report


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="深度参数分析引擎")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                       help="交易对列表")
    parser.add_argument("--rounds", type=int, default=3,
                       help="每个交易对运行轮数")
    parser.add_argument("--bars", type=int, default=2000,
                       help="每个交易对K线数")
    parser.add_argument("--p-threshold", type=float, default=P_THRESHOLD,
                       help="显著性阈值")
    parser.add_argument("--output", type=str, default=None,
                       help="输出目录")
    args = parser.parse_args()

    run_deep_param_analysis(
        symbols=args.symbols,
        rounds=args.rounds,
        bars_per_symbol=args.bars,
        output_dir=Path(args.output) if args.output else SCRIPT_DIR,
    )