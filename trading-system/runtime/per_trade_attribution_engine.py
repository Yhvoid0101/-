#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐笔归因引擎 (PerTradeAttributionEngine) — 复盘→迭代闭环的核心智能

用户核心诉求:
  "复盘闭环应该是: 产生交易 → 分析过程 → 找好坏 → 亏为何亏/赚为何赚 → 反复微调 → 越来越强越来越准"
  "不要局限于参数渐进调整, 也有新策略发展、旧策略迭代、新金融知识融入、新指标融合,
   一切为了实盘交易的收益提升!"

核心能力:
  1. 5维归因每笔交易 (Per-Trade 5-Dimensional Attribution):
     ├─ 信号溯源 (Signal Tracing): 入场信号 + direction alignment + 该信号历史表现
     ├─ 特征归因 (Feature Attribution): RF feature_importances_ + 异常特征值检测
     ├─ 退出诊断 (Exit Diagnosis): MAE/MFE/最优退出/可避免性 (从 pnl+exit_reason+bars_held+atr 推断)
     ├─ 市场匹配 (Market Matching): regime + market_state_* + 策略类型匹配度
     └─ 风险评估 (Risk Assessment): 仓位/SL/TP 合理性 (ATR-based 启发式)
  2. 模式发现 (Pattern Discovery): 跨交易聚合, 找出反复出现的盈亏模式
  3. 5维进化建议 (5-Dimensional Evolution Suggestions):
     ├─ parameter: 找出特定 (symbol, regime) 下表现差的参数, 建议 ±5-15% 调整
     ├─ logic: 找出特定信号在特定 regime 下失效, 建议增加过滤条件
     ├─ indicator: 找出特征值与盈亏强相关的指标, 建议融入 entry 决策
     ├─ knowledge: 提炼可复用市场规律 (如"trending+high_vol 逆势亏损率 75%")
     └─ strategy: 找出整体失效的 (strategy_type, regime) 组合, 建议退役或切换

设计原则 (用户铁律):
  - "复盘不是摆设" — 归因结论必须可执行, 不是摆设
  - "理解底层逻辑, 而非表面指标" — 一句话根因: "rsi_divergence在trending+high_vol下与MTF反向导致逆势亏损"
  - "不要通过已知数据反推策略" — 模式发现基于统计异常, 不预设结论
  - "永不二过" — 失败模式入库, 下次自动检测复发

数据格式自动适配 (4 种格式):
  - v97 raw: active_trades 中的 dict, 含 rec.features 45 字段 (最丰富)
  - v97 detail json: 18 字段, 特征缺失时降级
  - v98 detail json: 23 字段, 含 v95_pos_mult/v96_fusion_mult/v98_hold_bars_mult
  - runtime Trade 对象: 含 signal_source/stop_loss/take_profit (Task #42 补齐后)
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hermes.per_trade_attribution")

SCRIPT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

ATTRIBUTION_REPORTS_DIR = SCRIPT_DIR / "_attribution_reports"
ATTRIBUTION_REPORTS_DIR.mkdir(exist_ok=True)

# sklearn 可选导入 (特征归因用 RF feature_importances_)
try:
    from sklearn.ensemble import RandomForestRegressor
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    RandomForestRegressor = None

# ============================================================================
# 常量 — 基于真实 v97 数据特征 + 用户铁律硬约束
# ============================================================================

# 5维归因维度名
DIM_SIGNAL = "signal"
DIM_FEATURE = "feature"
DIM_EXIT = "exit"
DIM_MARKET = "market"
DIM_RISK = "risk"

# 5维进化建议名 (用户铁律: 不局限于参数, 含 strategy/logic/knowledge/indicator)
EVOL_PARAMETER = "parameter"
EVOL_LOGIC = "logic"
EVOL_INDICATOR = "indicator"
EVOL_KNOWLEDGE = "knowledge"
EVOL_STRATEGY = "strategy"

# 根因分类
ROOT_CAUSE_CATEGORIES = {
    "counter_trend": "逆势入场 (信号与 MTF/HTF 趋势反向)",
    "bad_exit": "退出时机错误 (过早止盈/过晚止损/未达最优退出)",
    "high_vol_risk": "高波动下仓位过重 (atr_pct 高 + pos_mult 大)",
    "regime_mismatch": "市场状态与策略类型不匹配 (如均值回归在 trending 下)",
    "weak_signal": "信号本身置信度低 (confluence_score 低 + 多信号矛盾)",
    "feature_anomaly": "入场特征异常 (rsi 极端/adx 过低/vp 反向)",
    "risk_misconfigured": "SL/TP 配置不合理 (R:R < 0.5 或 SL 过宽)",
    "unexplained": "无法归因 (数据不足或多重因素交织)",
}

# 策略类型推断 (基于信号特征)
STRATEGY_TYPES = {
    "mean_reversion": ["rsi_divergence", "bb_squeeze", "rsi_oversold", "rsi_overbought"],
    "trend_following": ["ema_breakout", "macd_cross", "mtf_aligned", "hh_hl_pattern"],
    "momentum": ["pa_bull_engulf", "pa_bear_engulf", "pa_body_expansion"],
    "range_trading": ["range_pct_high", "vp_obv_slope_norm_extreme"],
}

# 安全的参数调整范围 (基于 gene_codec.py GENE_SCHEMA)
PARAM_ADJUST_RANGES = {
    "stop_loss_pct": (-0.15, 0.15),       # ±15%
    "take_profit_pct": (-0.15, 0.15),
    "base_risk_per_trade": (-0.10, 0.10),  # ±10%
    "pos_mult": (-0.20, 0.20),             # ±20%
    "confluence_threshold": (-0.10, 0.10),
}

# 模式发现的统计阈值
PATTERN_MIN_TRADES = 10          # 模式至少覆盖 10 笔交易
PATTERN_LOSS_RATE_THRESHOLD = 0.60  # 亏损率 ≥60% 才算异常模式
PATTERN_WIN_RATE_THRESHOLD = 0.70   # 胜率 ≥70% 算优势模式
PATTERN_PNL_EXTREME = -500.0       # 总 PnL ≤ -$500 算显著亏损模式

# 归因置信度阈值
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.50
CONFIDENCE_LOW = 0.30


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class AttributionResult:
    """单笔交易 5 维归因结果

    一笔交易的完整归因画像, 含 5 维独立归因 + 综合根因 + 置信度。
    典型根因示例: "rsi_divergence 在 trending+high_vol 下与 MTF 反向导致逆势亏损"
    """
    trade_id: str
    symbol: str
    direction: str           # long / short
    pnl_pct: float           # 小数形式, 如 0.0129 = 1.29%
    pnl_outcome: str         # win / loss / breakeven
    # 5维归因 (各自独立分析, 由 _synthesize_root_cause 综合为根因)
    signal_attribution: Dict[str, Any] = field(default_factory=dict)
    feature_attribution: Dict[str, Any] = field(default_factory=dict)
    exit_attribution: Dict[str, Any] = field(default_factory=dict)
    market_attribution: Dict[str, Any] = field(default_factory=dict)
    risk_attribution: Dict[str, Any] = field(default_factory=dict)
    # 综合
    root_cause: str = ""                 # 一句话根因
    root_cause_category: str = "unexplained"
    confidence: float = 0.0              # 0-1, 归因置信度
    data_quality: str = "rich"           # rich / degraded / minimal

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatternFinding:
    """跨交易聚合发现的盈亏模式

    示例: "trending+high_vol 下逆势入场亏损率 75% (60 笔, 总 PnL=-$3500)"
    """
    pattern_id: str
    description: str                     # 人类可读描述
    trade_count: int
    statistics: Dict[str, Any]           # win_rate/avg_pnl/total_pnl/loss_rate
    conditions: Dict[str, Any]           # 触发该模式的条件
    impact: str = "medium"               # high / medium / low
    evolution_dimension: str = EVOL_PARAMETER  # 该模式最适合的进化维度
    pattern_type: str = "loss"           # loss / win / mixed

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvolutionSuggestion:
    """5 维进化建议 — 归因引擎的核心产出

    每条建议含: 维度+证据+建议变更+预期影响+安全分析+优先级
    下游 EvolutionSuggestionApplier 按维度分发到对应 overlay 文件
    """
    dimension: str                       # parameter/logic/indicator/knowledge/strategy
    name: str                            # 唯一标识
    evidence: Dict[str, Any] = field(default_factory=dict)
    proposed_change: Dict[str, Any] = field(default_factory=dict)
    expected_impact: Dict[str, float] = field(default_factory=dict)
    safety_analysis: Dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"             # critical/high/medium/low
    source_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttributionReport:
    """完整归因报告 — 一次 attribute_all 的产出"""
    timestamp: str
    trade_count: int
    attributions: List[AttributionResult] = field(default_factory=list)
    patterns: List[PatternFinding] = field(default_factory=list)
    suggestions: List[EvolutionSuggestion] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "trade_count": self.trade_count,
            "attributions": [a.to_dict() for a in self.attributions],
            "patterns": [p.to_dict() for p in self.patterns],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "summary": self.summary,
        }


# ============================================================================
# 核心归因引擎
# ============================================================================
class PerTradeAttributionEngine:
    """逐笔归因引擎 — 5 维归因 + 模式发现 + 5 维进化建议

    使用方式:
        engine = PerTradeAttributionEngine(initial_capital=10000.0)
        report = engine.attribute_all(trades)
        engine.save_report(report)

    典型输出 (per trade):
        root_cause = "rsi_divergence 在 trending+high_vol 下与 MTF 反向导致逆势亏损"
        root_cause_category = "counter_trend"
        confidence = 0.82

    典型输出 (pattern):
        description = "trending+high_vol 下逆势入场亏损率 75%"
        evolution_dimension = "logic"  # 建议增加 MTF 同向过滤

    典型输出 (suggestion):
        dimension = "logic"
        name = "rsi_divergence_mtf_filter_in_trending"
        proposed_change = {
            "trigger": {"signal": "rsi_divergence", "regime": "trending"},
            "filter": {"feature": "mtf_htf_trend_4h", "op": "eq", "value": "direction"},
            "action": "reject_entry"
        }
    """

    def __init__(self, baseline_dir: Optional[str] = None,
                 initial_capital: float = 10000.0):
        """初始化归因引擎

        Args:
            baseline_dir: 历史归因报告目录 (用于趋势对比, 可选)
            initial_capital: 初始资金 (用于 PnL 尺度换算)
        """
        self.initial_capital = float(initial_capital)
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None

        # RF 模型缓存 (特征归因用, 避免重复训练)
        self._rf_model: Optional[RandomForestRegressor] = None
        self._rf_feature_names: List[str] = []
        self._rf_trained: bool = False

        # 全局统计缓存
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        self._signal_history: Dict[str, Dict[str, float]] = {}  # signal_name → stats

    # ========================================================================
    # 主入口
    # ========================================================================

    def attribute_all(self, trades: List[dict]) -> AttributionReport:
        """对一批交易执行 5 维归因 + 模式发现 + 5 维建议产出

        Args:
            trades: 交易记录列表 (自动适配 4 种格式)

        Returns:
            AttributionReport: 完整归因报告
        """
        t_start = time.time()
        report = AttributionReport(
            timestamp=datetime.now().isoformat(),
            trade_count=len(trades),
        )

        if not trades:
            logger.warning("PerTradeAttributionEngine.attribute_all: 空交易列表")
            return report

        # Step 1: 标准化所有交易为内部格式
        normalized = [self._normalize_trade(t) for t in trades]
        normalized = [t for t in normalized if t is not None]
        logger.info(f"[Attribution] 标准化 {len(normalized)}/{len(trades)} 笔交易")

        if not normalized:
            logger.error("[Attribution] 无可归因交易 (全部标准化失败)")
            return report

        # Step 2: 计算全局统计 (特征分布 + 信号历史表现)
        self._compute_feature_stats(normalized)
        self._compute_signal_history(normalized)

        # Step 3: 训练 RF 模型 (用于特征归因, 仅 sklearn 可用时)
        if _SKLEARN_AVAILABLE:
            self._train_rf_model(normalized)

        # Step 4: 逐笔 5 维归因
        attributions = []
        for i, trade in enumerate(normalized):
            try:
                attr = self._attribute_single(trade, normalized, i)
                attributions.append(attr)
            except Exception as e:
                logger.warning(f"[Attribution] trade #{i} 归因失败: {e}")
                # 失败不阻塞, 添加降级结果
                attributions.append(AttributionResult(
                    trade_id=f"trade_{i}",
                    symbol=trade.get("symbol", "unknown"),
                    direction=trade.get("direction", "unknown"),
                    pnl_pct=trade.get("pnl_pct", 0),
                    pnl_outcome=self._classify_pnl(trade.get("pnl_pct", 0)),
                    root_cause=f"归因异常: {e}",
                    root_cause_category="unexplained",
                    confidence=0.0,
                    data_quality="degraded",
                ))
        report.attributions = attributions

        # Step 5: 模式发现 (跨交易聚合)
        report.patterns = self._discover_patterns(attributions, normalized)
        logger.info(f"[Attribution] 发现 {len(report.patterns)} 个模式")

        # Step 6: 5 维进化建议产出
        report.suggestions = self._generate_suggestions(attributions, report.patterns, normalized)
        logger.info(f"[Attribution] 产出 {len(report.suggestions)} 条建议 "
                    f"(by dimension: {dict(Counter(s.dimension for s in report.suggestions))})")

        # Step 7: 汇总
        report.summary = self._build_summary(report)

        duration = time.time() - t_start
        logger.info(f"[Attribution] 完成: {len(attributions)} 笔归因, "
                    f"{len(report.patterns)} 模式, {len(report.suggestions)} 建议, "
                    f"耗时 {duration:.2f}s")
        return report

    # ========================================================================
    # 数据标准化 — 自动适配 4 种格式
    # ========================================================================

    def _normalize_trade(self, trade: Any) -> Optional[dict]:
        """将 4 种格式的交易记录统一为内部格式

        支持格式:
          1. v97 raw: {symbol, rec: {features: {45字段}, direction, ...}, live_pnl, ...}
          2. v97 detail json: {symbol, direction(str), entry_price, pnl_pct, regime, ...} (18字段)
          3. v98 detail json: {symbol, direction(int), v95_pos_mult, v96_fusion_mult, ...} (23字段)
          4. runtime Trade 对象: 含 signal_source/stop_loss/take_profit

        Returns:
            标准化后的 dict, 含统一字段:
            - symbol, direction(long/short), pnl_pct, pnl_outcome
            - features: dict (45 字段, 缺失时为空)
            - regime, market_state_16, htf_state, atr_state, vol_state
            - exit_reason, bars_held, entry_price, exit_price
            - signal_source, stop_loss, take_profit (可选)
            - data_quality: rich/degraded/minimal
        """
        if trade is None:
            return None

        # 处理 runtime Trade 对象 (有 to_dict 方法)
        if hasattr(trade, "to_dict") and callable(trade.to_dict):
            try:
                trade = trade.to_dict()
            except Exception:
                pass

        if not isinstance(trade, dict):
            return None

        # 提取 rec (v97 raw 格式)
        rec = trade.get("rec", {})
        if not isinstance(rec, dict):
            rec = {}

        # 提取 features (优先从 rec.features, 其次从 trade.features)
        features = rec.get("features", trade.get("features", {}))
        if not isinstance(features, dict):
            features = {}

        # 统一 direction (v97 raw 是 int, v97 detail 是 str)
        direction_raw = rec.get("direction", trade.get("direction", 0))
        direction = self._normalize_direction(direction_raw)

        # 统一 pnl_pct (v97 是小数如 0.0129, 部分格式是百分比如 1.29)
        pnl_pct = self._normalize_pnl_pct(
            rec.get("pnl_pct", trade.get("pnl_pct", 0)),
            trade.get("live_pnl", 0),
        )

        # 数据质量评估
        feature_count = len(features)
        if feature_count >= 30:
            data_quality = "rich"
        elif feature_count >= 10:
            data_quality = "degraded"
        else:
            data_quality = "minimal"

        # 构建标准化 trade
        normalized = {
            # 基础字段
            "symbol": trade.get("symbol", rec.get("symbol", "unknown")),
            "direction": direction,
            "pnl_pct": float(pnl_pct),
            "pnl_outcome": self._classify_pnl(pnl_pct),
            "entry_price": float(rec.get("entry_price", trade.get("entry_price", 0)) or 0),
            "exit_price": float(trade.get("exit_price", 0) or 0),
            "bars_held": int(rec.get("hold_bars", trade.get("bars_held", 0)) or 0),
            "exit_reason": trade.get("exit_reason", self._infer_exit_reason(pnl_pct, rec.get("hold_bars", 0))),
            # 特征字典 (核心归因数据源)
            "features": features,
            # 市场状态 (多层级)
            "regime": features.get("regime", trade.get("regime", "unknown")),
            "market_state_16": trade.get("market_state_16", ""),
            "market_state_8": trade.get("market_state_8", ""),
            "htf_state": trade.get("htf_state", ""),
            "htf_trend": int(trade.get("htf_trend", 0) or 0),
            "atr_state": trade.get("atr_state", ""),
            "vol_state": trade.get("vol_state", ""),
            # 仓位信息
            "pos_mult": float(trade.get("pos_mult", trade.get("v95_pos_mult", 1.0)) or 1.0),
            "final_pos_mult": float(trade.get("final_pos_mult", trade.get("pos_mult", 1.0)) or 1.0),
            "leverage": float(trade.get("leverage", 1.0) or 1.0),
            # 可选字段 (Task #42 补齐后会有)
            "signal_source": trade.get("signal_source", ""),
            "stop_loss": float(trade.get("stop_loss", 0) or 0),
            "take_profit": float(trade.get("take_profit", 0) or 0),
            "mae_pct": float(trade.get("mae_pct", 0) or 0),
            "mfe_pct": float(trade.get("mfe_pct", 0) or 0),
            # 元数据
            "data_quality": data_quality,
            "feature_count": feature_count,
            # 保留原始 trade 供回溯
            "_raw": trade,
        }

        # 关键特征提取 (常用, 提前算好)
        f = features
        normalized["entry_atr"] = float(f.get("atr_abs", 0) or 0)
        normalized["entry_atr_pct"] = float(f.get("atr_pct", 0) or 0)
        normalized["entry_rsi"] = float(f.get("rsi_14", 50) or 50)
        normalized["entry_adx"] = float(f.get("adx", 0) or 0)
        normalized["entry_hurst"] = float(f.get("hurst", 0.5) or 0.5)
        normalized["confluence_score"] = float(f.get("confluence_score", 0) or 0)
        normalized["confluence_tier"] = str(f.get("confluence_tier", "unknown"))
        normalized["mtf_htf_trend_4h"] = int(f.get("mtf_htf_trend_4h", 0) or 0)
        normalized["mtf_trend_direction"] = int(f.get("mtf_trend_direction", 0) or 0)
        normalized["mtf_ema_alignment"] = int(f.get("mtf_ema_alignment", 0) or 0)
        normalized["mtf_short_ema_alignment"] = int(f.get("mtf_short_ema_alignment", 0) or 0)
        normalized["mtf_market_structure"] = str(f.get("mtf_market_structure", "unknown"))
        normalized["pa_body_pct"] = float(f.get("pa_body_pct", 0) or 0)
        normalized["pa_range_pct"] = float(f.get("pa_range_pct", 0) or 0)
        normalized["vp_obv_slope_norm"] = float(f.get("vp_obv_slope_norm", 0) or 0)
        normalized["vp_vol_ratio_5"] = float(f.get("vp_vol_ratio_5", 1.0) or 1.0)
        normalized["vp_price_vol_aligned"] = int(f.get("vp_price_vol_aligned", 0) or 0)
        normalized["kelly_win_rate"] = float(f.get("kelly_win_rate", 0.5) or 0.5)
        normalized["kelly_payoff_ratio"] = float(f.get("kelly_payoff_ratio", 1.0) or 1.0)

        # trade_id
        normalized["trade_id"] = (
            f"{normalized['symbol']}_{normalized['direction']}_{rec.get('entry_bar', i_for_id)}"
            if (i_for_id := rec.get("entry_bar", 0)) else
            f"{normalized['symbol']}_{normalized['direction']}_{id(trade) % 100000}"
        )

        return normalized

    def _normalize_direction(self, direction: Any) -> str:
        """统一 direction 为 long/short 字符串"""
        if isinstance(direction, str):
            d = direction.lower().strip()
            if d in ("long", "buy", "1", "1.0"):
                return "long"
            elif d in ("short", "sell", "-1", "-1.0"):
                return "short"
            return d
        try:
            d = float(direction)
            if d > 0:
                return "long"
            elif d < 0:
                return "short"
        except (TypeError, ValueError):
            pass
        return "unknown"

    def _normalize_pnl_pct(self, pnl_pct: float, live_pnl: float = 0) -> float:
        """统一 pnl_pct 为小数形式 (0.0129 = 1.29%)

        v97 raw: pnl_pct 已是小数
        v97 detail json: pnl_pct 已是小数
        部分 JSON: pnl_pct 是百分比 (1.29)
        """
        try:
            p = float(pnl_pct)
        except (TypeError, ValueError):
            p = 0.0

        # 检测尺度: 如果 |pnl_pct| > 0.5, 可能是百分比形式
        if abs(p) > 0.5:
            p = p / 100.0
        return p

    def _classify_pnl(self, pnl_pct: float) -> str:
        """分类 PnL 为 win/loss/breakeven"""
        if pnl_pct > 0.001:  # > 0.1%
            return "win"
        elif pnl_pct < -0.001:
            return "loss"
        return "breakeven"

    def _infer_exit_reason(self, pnl_pct: float, hold_bars: int) -> str:
        """从 pnl + hold_bars 推断 exit_reason (数据缺失时)"""
        if pnl_pct > 0:
            return "take_profit" if hold_bars <= 10 else "trailing_stop"
        else:
            return "stop_loss" if hold_bars <= 5 else "time_stop"

    # ========================================================================
    # 全局统计计算
    # ========================================================================

    def _compute_feature_stats(self, trades: List[dict]) -> None:
        """计算所有特征的全局统计 (mean/std/p25/p50/p75)"""
        self._feature_stats = {}
        if not trades:
            return

        # 收集所有数值特征
        all_features: Dict[str, List[float]] = defaultdict(list)
        for t in trades:
            for k, v in t.get("features", {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        all_features[k].append(float(v))
                    except (TypeError, ValueError):
                        pass

        # 计算统计
        for fname, values in all_features.items():
            if len(values) < 2:
                continue
            try:
                arr = np.array(values, dtype=float)
                self._feature_stats[fname] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
            except Exception:
                pass

    def _compute_signal_history(self, trades: List[dict]) -> None:
        """计算每个信号的历史表现 (用于信号溯源)"""
        self._signal_history = {}
        if not trades:
            return

        # 从每笔交易推断触发的信号
        signal_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"pnls": [], "wins": 0, "losses": 0}
        )
        for t in trades:
            signals = self._infer_signals(t)
            pnl = t.get("pnl_pct", 0)
            outcome = t.get("pnl_outcome", "breakeven")
            for sig in signals:
                signal_stats[sig]["pnls"].append(pnl)
                if outcome == "win":
                    signal_stats[sig]["wins"] += 1
                elif outcome == "loss":
                    signal_stats[sig]["losses"] += 1

        # 计算汇总
        for sig, data in signal_stats.items():
            pnls = data["pnls"]
            if not pnls:
                continue
            total = data["wins"] + data["losses"]
            self._signal_history[sig] = {
                "count": len(pnls),
                "win_rate": data["wins"] / total if total > 0 else 0,
                "avg_pnl": float(np.mean(pnls)),
                "total_pnl": float(np.sum(pnls)),
                "is_profitable": sum(pnls) > 0,
            }

    def _infer_signals(self, trade: dict) -> List[str]:
        """从交易特征推断触发的信号 (数据补齐前的过渡方案)

        基于 features 推断可能的触发信号, 返回信号名列表。
        Task #42 补齐 signal_source 后, 直接用真实信号名。
        """
        # 如果已有 signal_source, 直接用
        ss = trade.get("signal_source", "")
        if ss and ss != "none":
            return [s.strip() for s in ss.split(";") if s.strip()]

        # 否则从特征推断
        signals = []
        f = trade.get("features", {})

        # RSI 信号
        rsi = float(f.get("rsi_14", 50) or 50)
        if rsi < 30:
            signals.append("rsi_oversold")
        elif rsi > 70:
            signals.append("rsi_overbought")
        # RSI 背离 (简化检测)
        if rsi < 50 and trade.get("direction") == "long":
            signals.append("rsi_divergence:long")
        elif rsi > 50 and trade.get("direction") == "short":
            signals.append("rsi_divergence:short")

        # EMA 突破
        mtf_ema_alignment = int(f.get("mtf_ema_alignment", 0) or 0)
        if mtf_ema_alignment != 0:
            signals.append(f"ema_alignment:{'long' if mtf_ema_alignment > 0 else 'short'}")

        # MTF HTF 趋势
        mtf_htf = int(f.get("mtf_htf_trend_4h", 0) or 0)
        if mtf_htf != 0:
            signals.append(f"mtf_htf_trend:{'long' if mtf_htf > 0 else 'short'}")

        # 价格行为信号
        if int(f.get("pa_bull_engulf", 0) or 0) > 0:
            signals.append("pa_bull_engulf")
        if int(f.get("pa_bear_engulf", 0) or 0) > 0:
            signals.append("pa_bear_engulf")

        # 量价信号
        vp_aligned = int(f.get("vp_price_vol_aligned", 0) or 0)
        if vp_aligned != 0:
            signals.append(f"vp_aligned:{'long' if vp_aligned > 0 else 'short'}")

        # 默认
        if not signals:
            signals.append("unknown")

        return signals

    def _infer_strategy_type(self, trade: dict) -> str:
        """从交易特征推断策略类型"""
        signals = self._infer_signals(trade)
        signal_str = ";".join(signals).lower()

        for stype, keywords in STRATEGY_TYPES.items():
            for kw in keywords:
                if kw in signal_str:
                    return stype

        # 基于 regime 推断
        regime = trade.get("regime", "")
        if regime == "trending":
            return "trend_following"
        elif regime == "ranging":
            return "mean_reversion"
        return "unknown"

    # ========================================================================
    # RF 模型训练 (特征归因用)
    # ========================================================================

    def _train_rf_model(self, trades: List[dict]) -> None:
        """训练 RandomForest 模型, 计算 feature_importances_

        用 PnL 作为目标, 入场特征作为 X, 训练 RF 回归模型。
        feature_importances_ 告诉我们哪些特征对 PnL 影响最大。
        """
        if not _SKLEARN_AVAILABLE or len(trades) < 20:
            self._rf_trained = False
            return

        # 收集训练数据
        feature_names = set()
        for t in trades:
            feature_names.update(
                k for k, v in t.get("features", {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
        feature_names = sorted(feature_names)
        if len(feature_names) < 5:
            self._rf_trained = False
            return

        # 构建 X, y
        X = []
        y = []
        for t in trades:
            f = t.get("features", {})
            row = []
            valid = True
            for fn in feature_names:
                v = f.get(fn, 0)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row.append(float(v))
                else:
                    row.append(0.0)
            X.append(row)
            y.append(float(t.get("pnl_pct", 0)))

        try:
            X_arr = np.array(X, dtype=float)
            y_arr = np.array(y, dtype=float)
            self._rf_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=1,
            )
            self._rf_model.fit(X_arr, y_arr)
            self._rf_feature_names = feature_names
            self._rf_trained = True
            logger.info(f"[Attribution] RF 模型训练完成: {len(feature_names)} 特征, "
                        f"{len(trades)} 样本")
        except Exception as e:
            logger.warning(f"[Attribution] RF 训练失败, 降级到 Pearson: {e}")
            self._rf_trained = False

    # ========================================================================
    # 5 维归因 (核心)
    # ========================================================================

    def _attribute_single(self, trade: dict, all_trades: List[dict],
                          idx: int) -> AttributionResult:
        """对单笔交易执行 5 维归因

        Args:
            trade: 标准化后的交易
            all_trades: 全部交易 (用于全局对比)
            idx: 交易索引

        Returns:
            AttributionResult: 完整归因结果
        """
        result = AttributionResult(
            trade_id=trade.get("trade_id", f"trade_{idx}"),
            symbol=trade.get("symbol", "unknown"),
            direction=trade.get("direction", "unknown"),
            pnl_pct=trade.get("pnl_pct", 0),
            pnl_outcome=trade.get("pnl_outcome", "breakeven"),
            data_quality=trade.get("data_quality", "minimal"),
        )

        # 5 维独立归因
        result.signal_attribution = self._trace_signal(trade)
        result.feature_attribution = self._attribute_features(trade)
        result.exit_attribution = self._diagnose_exit(trade)
        result.market_attribution = self._match_market(trade)
        result.risk_attribution = self._assess_risk(trade)

        # 综合根因
        result.root_cause, result.root_cause_category, result.confidence = (
            self._synthesize_root_cause(result, trade)
        )

        return result

    # --------------------------------------------------------------------
    # 维度 1: 信号溯源 (Signal Tracing)
    # --------------------------------------------------------------------

    def _trace_signal(self, trade: dict) -> dict:
        """信号溯源: 哪个信号触发入场? direction alignment? 该信号历史表现?

        典型结论:
          - "rsi_divergence:long 触发, 但 mtf_htf_trend_4h=-1 (short), 信号与 HTF 反向"
          - "rsi_oversold 信号历史胜率 42% (50 笔), 当前笔亏损符合历史模式"
        """
        signals = self._infer_signals(trade)
        direction = trade.get("direction", "unknown")

        # direction alignment 检查
        mtf_htf_trend = int(trade.get("mtf_htf_trend_4h", 0) or 0)
        mtf_trend_dir = int(trade.get("mtf_trend_direction", 0) or 0)
        mtf_ema_align = int(trade.get("mtf_ema_alignment", 0) or 0)

        alignments = {}
        if mtf_htf_trend != 0:
            htf_dir = "long" if mtf_htf_trend > 0 else "short"
            alignments["mtf_htf_trend_4h"] = {
                "trend": htf_dir,
                "aligned": htf_dir == direction,
            }
        if mtf_trend_dir != 0:
            td = "long" if mtf_trend_dir > 0 else "short"
            alignments["mtf_trend_direction"] = {
                "trend": td,
                "aligned": td == direction,
            }
        if mtf_ema_align != 0:
            ed = "long" if mtf_ema_align > 0 else "short"
            alignments["mtf_ema_alignment"] = {
                "trend": ed,
                "aligned": ed == direction,
            }

        # 统计逆势数
        misaligned_count = sum(1 for a in alignments.values() if not a["aligned"])
        total_alignments = len(alignments)

        # 信号历史表现
        signal_history = {}
        for sig in signals:
            if sig in self._signal_history:
                signal_history[sig] = self._signal_history[sig]

        # 综合判断
        is_counter_trend = misaligned_count >= 2 and total_alignments >= 2
        has_weak_history = any(
            not h.get("is_profitable", True) or h.get("win_rate", 1) < 0.45
            for h in signal_history.values()
        )

        return {
            "signals_detected": signals,
            "direction": direction,
            "alignments": alignments,
            "misaligned_count": misaligned_count,
            "total_alignments": total_alignments,
            "is_counter_trend": is_counter_trend,
            "signal_history": signal_history,
            "has_weak_history": has_weak_history,
            "conclusion": (
                f"{'逆势' if is_counter_trend else '顺势'}入场, "
                f"{'信号历史弱' if has_weak_history else '信号历史正常'}"
            ),
        }

    # --------------------------------------------------------------------
    # 维度 2: 特征归因 (Feature Attribution)
    # --------------------------------------------------------------------

    def _attribute_features(self, trade: dict) -> dict:
        """特征归因: 入场特征值与盈亏的关系 + 异常特征检测

        用 RF feature_importances_ (或 Pearson 相关 fallback) 标识关键特征,
        检测该笔的特征值是否在异常区间 (PnL < 0 时是否在亏损分位)。
        """
        features = trade.get("features", {})
        if not features:
            return {"conclusion": "无特征数据, 跳过特征归因", "anomalies": []}

        # 1. 全局特征重要性 (RF 或 Pearson)
        if self._rf_trained and self._rf_model is not None:
            importances = dict(zip(self._rf_feature_names,
                                   self._rf_model.feature_importances_))
            method = "rf"
        else:
            importances = self._compute_pearson_importance(features.keys())
            method = "pearson_fallback"

        # 2. 异常特征检测 (该笔特征值是否在亏损分位)
        anomalies = []
        for fname, fval in features.items():
            if not isinstance(fval, (int, float)) or isinstance(fval, bool):
                continue
            stats = self._feature_stats.get(fname)
            if not stats:
                continue
            mean = stats["mean"]
            std = stats["std"]
            if std == 0:
                continue
            z_score = (float(fval) - mean) / std
            # 异常: |z| > 1.5
            if abs(z_score) > 1.5:
                anomalies.append({
                    "feature": fname,
                    "value": float(fval),
                    "z_score": float(z_score),
                    "mean": mean,
                    "std": std,
                    "direction": "high" if z_score > 0 else "low",
                    "importance": float(importances.get(fname, 0)),
                })

        # 按重要性 × |z_score| 排序
        anomalies.sort(key=lambda a: a["importance"] * abs(a["z_score"]), reverse=True)

        # 3. 提取 Top 5 关键特征
        top_features = sorted(
            [(f, float(importances.get(f, 0))) for f in features
             if isinstance(features[f], (int, float)) and not isinstance(features[f], bool)],
            key=lambda x: x[1], reverse=True
        )[:5]

        # 4. 综合
        top_anomaly_names = [a["feature"] for a in anomalies[:3]]
        conclusion = (
            f"关键特征: {', '.join(f'{n}({v:.3f})' for n, v in top_features[:3])}; "
            f"{'异常特征: ' + ', '.join(top_anomaly_names) if top_anomaly_names else '无显著异常'}"
        )

        return {
            "method": method,
            "top_features": [{"name": n, "importance": v} for n, v in top_features],
            "anomalies": anomalies[:10],
            "anomaly_count": len(anomalies),
            "conclusion": conclusion,
        }

    def _compute_pearson_importance(self, feature_names) -> Dict[str, float]:
        """Pearson 相关 fallback (sklearn 不可用时)"""
        if not self._feature_stats:
            return {}
        # 简化: 用特征值的方差作为重要性代理 (方差大的特征区分度高)
        importance = {}
        for fn in feature_names:
            stats = self._feature_stats.get(fn)
            if stats and stats["std"] > 0:
                # 归一化到 0-1
                importance[fn] = min(1.0, stats["std"] / (abs(stats["mean"]) + 1e-9))
        return importance

    # --------------------------------------------------------------------
    # 维度 3: 退出诊断 (Exit Diagnosis)
    # --------------------------------------------------------------------

    def _diagnose_exit(self, trade: dict) -> dict:
        """退出诊断: MAE/MFE/最优退出/可避免性

        数据补齐前从 pnl_pct + exit_reason + bars_held + atr_pct 推断:
          - stop_loss 出场: MAE ≈ -sl_dist, MFE ≈ max(0, pnl before sl)
          - take_profit 出场: MFE ≈ tp_dist, MAE ≈ min(0, drawdown before tp)
          - time_stop: MAE/MFE 从 bars_held × atr_pct 估算
        """
        pnl_pct = trade.get("pnl_pct", 0)
        exit_reason = trade.get("exit_reason", "unknown")
        bars_held = int(trade.get("bars_held", 0) or 0)
        atr_pct = float(trade.get("entry_atr_pct", 0) or 0)
        direction = trade.get("direction", "unknown")

        # 优先用真实 MAE/MFE (Task #42 补齐后)
        mae_pct = float(trade.get("mae_pct", 0) or 0)
        mfe_pct = float(trade.get("mfe_pct", 0) or 0)
        inferred = False

        if mae_pct == 0 and mfe_pct == 0:
            inferred = True
            # 推断 MAE/MFE
            if exit_reason == "stop_loss":
                # 止损出场: MAE ≈ -pnl_pct (实际亏损), MFE 估算
                mae_pct = -abs(pnl_pct)
                mfe_pct = max(0, atr_pct * 0.5)  # 估计曾有 0.5 ATR 利润
            elif exit_reason == "take_profit":
                # 止盈出场: MFE ≈ pnl_pct, MAE 估算
                mfe_pct = abs(pnl_pct)
                mae_pct = -atr_pct * 0.3  # 估计曾有 0.3 ATR 回撤
            elif exit_reason == "trailing_stop":
                mfe_pct = abs(pnl_pct) * 1.2  # 移动止损, 曾有更高利润
                mae_pct = -abs(pnl_pct) * 0.5
            elif exit_reason == "time_stop":
                # 时间止损, 从持仓时长 × ATR 估算波动
                expected_range = atr_pct * math.sqrt(max(1, bars_held))
                mae_pct = -expected_range * 0.5
                mfe_pct = expected_range * 0.5
            else:
                mae_pct = -abs(pnl_pct) if pnl_pct < 0 else 0
                mfe_pct = abs(pnl_pct) if pnl_pct > 0 else 0

        # 最优退出分析: 实际 PnL vs MFE (是否本可以更好)
        optimal_pnl = mfe_pct
        actual_pnl = pnl_pct
        exit_efficiency = (actual_pnl / optimal_pnl) if optimal_pnl > 0 else 1.0

        # 可避免性: 亏损笔是否本可以避免
        avoidable = False
        avoid_reason = ""
        if pnl_pct < 0:
            if exit_reason == "stop_loss" and mfe_pct > 0:
                # 曾盈利但最终止损, 本可在盈利时退出
                avoidable = True
                avoid_reason = f"曾有 +{mfe_pct*100:.2f}% 利润但最终止损, 本可止盈"
            elif exit_reason == "time_stop" and bars_held > 20:
                # 持仓过长, 时间止损
                avoidable = True
                avoid_reason = f"持仓 {bars_held} bars 过长, 时间止损本可更早退出"

        # 退出质量评分
        if exit_reason == "take_profit":
            exit_quality = "good"
        elif exit_reason == "trailing_stop" and pnl_pct > 0:
            exit_quality = "good"
        elif exit_reason == "stop_loss":
            exit_quality = "bad" if avoidable else "acceptable"
        elif exit_reason == "time_stop":
            exit_quality = "bad" if pnl_pct < 0 else "acceptable"
        else:
            exit_quality = "unknown"

        return {
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "mae_pct": float(mae_pct),
            "mfe_pct": float(mfe_pct),
            "mae_mfe_inferred": inferred,
            "actual_pnl_pct": float(actual_pnl),
            "optimal_pnl_pct": float(optimal_pnl),
            "exit_efficiency": float(exit_efficiency),
            "avoidable": avoidable,
            "avoid_reason": avoid_reason,
            "exit_quality": exit_quality,
            "conclusion": (
                f"{exit_reason} 出场 ({exit_quality}), "
                f"MAE={mae_pct*100:.2f}%, MFE={mfe_pct*100:.2f}%, "
                f"效率={exit_efficiency*100:.0f}%, "
                f"{'可避免: ' + avoid_reason if avoidable else '不可避免'}"
            ),
        }

    # --------------------------------------------------------------------
    # 维度 4: 市场匹配 (Market Matching)
    # --------------------------------------------------------------------

    def _match_market(self, trade: dict) -> dict:
        """市场匹配: regime + market_state_* + 策略类型匹配度

        典型结论:
          - "均值回归策略在 trending 下运行, 市场状态不匹配"
          - "trending+high_vol+counter_trend 状态, 高波动逆势入场"
        """
        regime = trade.get("regime", "unknown")
        market_state_16 = trade.get("market_state_16", "")
        market_state_8 = trade.get("market_state_8", "")
        htf_state = trade.get("htf_state", "")
        atr_state = trade.get("atr_state", "")
        vol_state = trade.get("vol_state", "")
        strategy_type = self._infer_strategy_type(trade)

        # 策略类型 vs regime 匹配
        match_score = 1.0  # 1.0 = 完美匹配, 0.0 = 完全不匹配
        mismatches = []

        if strategy_type == "mean_reversion":
            if regime == "trending":
                match_score *= 0.4
                mismatches.append("均值回归在 trending 下表现差")
            elif regime == "ranging":
                match_score *= 1.0
            elif regime == "high_vol":
                match_score *= 0.6
                mismatches.append("均值回归在 high_vol 下风险高")
        elif strategy_type == "trend_following":
            if regime == "trending":
                match_score *= 1.0
            elif regime == "ranging":
                match_score *= 0.3
                mismatches.append("趋势跟踪在 ranging 下表现差")
            elif regime == "high_vol":
                match_score *= 0.7

        # HTF 状态检查
        if htf_state == "counter_trend" and strategy_type == "trend_following":
            match_score *= 0.5
            mismatches.append("趋势跟踪在 counter_trend HTF 下逆势")

        # 高波动检查
        if vol_state == "high_vol" and strategy_type == "mean_reversion":
            match_score *= 0.7
            mismatches.append("均值回归在 high_vol 下易被止损")

        is_mismatched = match_score < 0.6

        return {
            "regime": regime,
            "market_state_16": market_state_16,
            "market_state_8": market_state_8,
            "htf_state": htf_state,
            "atr_state": atr_state,
            "vol_state": vol_state,
            "strategy_type": strategy_type,
            "match_score": float(match_score),
            "is_mismatched": is_mismatched,
            "mismatches": mismatches,
            "conclusion": (
                f"{strategy_type} 策略在 {regime}/{vol_state} 下运行, "
                f"匹配度 {match_score*100:.0f}%"
                + (f", 不匹配: {'; '.join(mismatches)}" if mismatches else "")
            ),
        }

    # --------------------------------------------------------------------
    # 维度 5: 风险评估 (Risk Assessment)
    # --------------------------------------------------------------------

    def _assess_risk(self, trade: dict) -> dict:
        """风险评估: 仓位/SL/TP 合理性 (ATR-based 启发式)"""
        pos_mult = float(trade.get("final_pos_mult", trade.get("pos_mult", 1.0)) or 1.0)
        leverage = float(trade.get("leverage", 1.0) or 1.0)
        atr_pct = float(trade.get("entry_atr_pct", 0) or 0)
        atr_abs = float(trade.get("entry_atr", 0) or 0)
        entry_price = float(trade.get("entry_price", 0) or 0)
        stop_loss = float(trade.get("stop_loss", 0) or 0)
        take_profit = float(trade.get("take_profit", 0) or 0)
        pnl_pct = float(trade.get("pnl_pct", 0) or 0)
        direction = trade.get("direction", "unknown")

        issues = []

        # 1. 仓位过重检查 (atr_pct 高 + pos_mult 大)
        if atr_pct > 0.025 and pos_mult > 1.5:
            issues.append({
                "type": "high_vol_position",
                "severity": "high",
                "message": f"高波动(atr_pct={atr_pct*100:.2f}%)下仓位过重(pos_mult={pos_mult:.2f})",
            })

        # 2. SL/TP 距离检查
        sl_dist_pct = 0
        tp_dist_pct = 0
        if entry_price > 0:
            if stop_loss > 0:
                sl_dist_pct = abs(entry_price - stop_loss) / entry_price
            if take_profit > 0:
                tp_dist_pct = abs(take_profit - entry_price) / entry_price

            # R:R 检查
            if sl_dist_pct > 0 and tp_dist_pct > 0:
                rr_ratio = tp_dist_pct / sl_dist_pct
                if rr_ratio < 0.5:
                    issues.append({
                        "type": "bad_rr_ratio",
                        "severity": "high",
                        "message": f"R:R={rr_ratio:.2f} < 0.5, 盈亏比过低",
                    })
                elif rr_ratio < 1.0:
                    issues.append({
                        "type": "low_rr_ratio",
                        "severity": "medium",
                        "message": f"R:R={rr_ratio:.2f} < 1.0, 盈亏比偏低",
                    })

            # SL 过宽检查
            if sl_dist_pct > 0.05:  # SL > 5%
                issues.append({
                    "type": "wide_stop_loss",
                    "severity": "medium",
                    "message": f"SL 距离 {sl_dist_pct*100:.2f}% > 5%, 过宽",
                })

        # 3. 推断 SL/TP (数据缺失时)
        sl_inferred = False
        if sl_dist_pct == 0 and atr_pct > 0:
            sl_inferred = True
            sl_dist_pct = atr_pct * 1.5  # 假设 SL = 1.5 ATR
        if tp_dist_pct == 0 and atr_pct > 0:
            tp_inferred = True
            tp_dist_pct = atr_pct * 2.0  # 假设 TP = 2 ATR

        # 4. 风险评分
        risk_score = 0.0  # 0=无风险, 1=极高风险
        if pos_mult > 2.0:
            risk_score += 0.3
        if atr_pct > 0.025:
            risk_score += 0.2
        if leverage > 3.0:
            risk_score += 0.2
        if sl_dist_pct > 0 and sl_dist_pct > 0.04:
            risk_score += 0.15
        risk_score = min(1.0, risk_score)

        risk_level = "low"
        if risk_score > 0.6:
            risk_level = "high"
        elif risk_score > 0.3:
            risk_level = "medium"

        return {
            "pos_mult": pos_mult,
            "leverage": leverage,
            "atr_pct": atr_pct,
            "sl_dist_pct": float(sl_dist_pct),
            "tp_dist_pct": float(tp_dist_pct),
            "sl_tp_inferred": sl_inferred,
            "rr_ratio": float(tp_dist_pct / sl_dist_pct) if sl_dist_pct > 0 else 0,
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "issues": issues,
            "conclusion": (
                f"仓位 {pos_mult:.2f}x, 风险等级 {risk_level}, "
                f"R:R={(tp_dist_pct/sl_dist_pct if sl_dist_pct > 0 else 0):.2f}"
                + (f", 问题: {'; '.join(i['message'] for i in issues)}" if issues else "")
            ),
        }

    # ========================================================================
    # 综合根因 (一句话)
    # ========================================================================

    def _synthesize_root_cause(self, result: AttributionResult,
                                trade: dict) -> Tuple[str, str, float]:
        """综合 5 维归因, 产出一句话根因 + 分类 + 置信度

        典型输出:
          root_cause = "rsi_divergence 在 trending+high_vol 下与 MTF 反向导致逆势亏损"
          root_cause_category = "counter_trend"
          confidence = 0.82
        """
        sig = result.signal_attribution
        feat = result.feature_attribution
        exit_a = result.exit_attribution
        mkt = result.market_attribution
        rsk = result.risk_attribution

        pnl_outcome = result.pnl_outcome
        direction = result.direction
        regime = trade.get("regime", "unknown")
        vol_state = trade.get("vol_state", "")
        signals = sig.get("signals_detected", ["unknown"])
        primary_signal = signals[0] if signals else "unknown"

        # 优先级排序的根因判定 (从最强信号到最弱)
        root_cause = ""
        category = "unexplained"
        confidence = 0.0

        # 1. 逆势入场 (最强信号)
        if sig.get("is_counter_trend", False):
            misaligned = sig.get("misaligned_count", 0)
            total = sig.get("total_alignments", 0)
            root_cause = (
                f"{primary_signal} 在 {regime}+{vol_state} 下与 {misaligned}/{total} "
                f"MTF 指标反向导致逆势{'亏损' if pnl_outcome == 'loss' else '入场'}"
            )
            category = "counter_trend"
            confidence = min(0.9, 0.5 + misaligned * 0.15)

        # 2. 市场状态不匹配
        elif mkt.get("is_mismatched", False):
            strategy_type = mkt.get("strategy_type", "unknown")
            mismatches = mkt.get("mismatches", [])
            root_cause = (
                f"{strategy_type} 策略在 {regime} 下不匹配: "
                f"{'; '.join(mismatches[:2])}"
            )
            category = "regime_mismatch"
            confidence = 0.65

        # 3. 退出时机错误
        elif exit_a.get("avoidable", False):
            avoid_reason = exit_a.get("avoid_reason", "")
            root_cause = (
                f"{primary_signal} 入场后退出时机错误: {avoid_reason}"
            )
            category = "bad_exit"
            confidence = 0.70

        # 4. 特征异常
        elif feat.get("anomaly_count", 0) >= 2:
            anomalies = feat.get("anomalies", [])
            top_anomaly = anomalies[0] if anomalies else {}
            root_cause = (
                f"入场特征异常: {top_anomaly.get('feature', '?')}="
                f"{top_anomaly.get('value', 0):.3f} (z={top_anomaly.get('z_score', 0):.2f})"
                f" 导致{'亏损' if pnl_outcome == 'loss' else '该笔表现'}"
            )
            category = "feature_anomaly"
            confidence = 0.55

        # 5. 高波动风险
        elif rsk.get("risk_level") == "high":
            root_cause = (
                f"高波动风险: pos_mult={rsk.get('pos_mult', 0):.2f} "
                f"+ atr_pct={rsk.get('atr_pct', 0)*100:.2f}% 导致"
                f"{'亏损放大' if pnl_outcome == 'loss' else '风险过高'}"
            )
            category = "high_vol_risk"
            confidence = 0.60

        # 6. 信号历史弱
        elif sig.get("has_weak_history", False):
            sig_history = sig.get("signal_history", {})
            weak_sig = next(iter(sig_history.keys()), primary_signal)
            weak_stats = sig_history.get(weak_sig, {})
            root_cause = (
                f"{weak_sig} 信号历史表现弱 (胜率 {weak_stats.get('win_rate', 0)*100:.0f}%, "
                f"{weak_stats.get('count', 0)} 笔), 当前笔{'亏损' if pnl_outcome == 'loss' else '表现'}符合历史模式"
            )
            category = "weak_signal"
            confidence = 0.55

        # 7. 数据不足
        elif result.data_quality == "minimal":
            root_cause = f"数据不足, 无法精确归因 (data_quality=minimal, {trade.get('feature_count', 0)} 特征)"
            category = "unexplained"
            confidence = 0.20

        # 8. 兜底
        else:
            root_cause = (
                f"{primary_signal} 在 {regime} 下{'亏损' if pnl_outcome == 'loss' else '盈利'}, "
                f"无明显单一根因 (多重因素交织)"
            )
            category = "unexplained"
            confidence = 0.35

        # 数据质量降级时降低置信度
        if result.data_quality == "degraded":
            confidence *= 0.8
        elif result.data_quality == "minimal":
            confidence *= 0.5

        return root_cause, category, float(confidence)

    # ========================================================================
    # 模式发现 (跨交易聚合)
    # ========================================================================

    def _discover_patterns(self, attributions: List[AttributionResult],
                            trades: List[dict]) -> List[PatternFinding]:
        """跨交易聚合, 找出反复出现的盈亏模式

        聚合维度:
          1. (regime, direction_alignment) — 如 trending+逆势
          2. (regime, strategy_type) — 如 ranging+mean_reversion
          3. (symbol, regime) — 如 LTC_USDT@ranging
          4. (signal, regime) — 如 rsi_divergence@trending
          5. (exit_reason, regime) — 如 stop_loss@high_vol
        """
        patterns: List[PatternFinding] = []

        # 维度 1: (regime, direction_alignment)
        groups = defaultdict(list)
        for i, attr in enumerate(attributions):
            if i >= len(trades):
                continue
            t = trades[i]
            regime = t.get("regime", "unknown")
            is_counter = attr.signal_attribution.get("is_counter_trend", False)
            alignment = "counter_trend" if is_counter else "with_trend"
            key = (regime, alignment)
            groups[key].append((attr, t))

        for (regime, alignment), items in groups.items():
            if len(items) < PATTERN_MIN_TRADES:
                continue
            pnls = [a.pnl_pct for a, _ in items]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p < 0)
            total = len(pnls)
            win_rate = wins / total if total > 0 else 0
            loss_rate = losses / total if total > 0 else 0
            total_pnl = sum(pnls)

            # 异常模式: 亏损率 ≥60% 或 总 PnL ≤ -$500
            is_loss_pattern = loss_rate >= PATTERN_LOSS_RATE_THRESHOLD or (
                total_pnl * self.initial_capital < PATTERN_PNL_EXTREME
            )
            is_win_pattern = win_rate >= PATTERN_WIN_RATE_THRESHOLD

            if not (is_loss_pattern or is_win_pattern):
                continue

            pattern_type = "loss" if is_loss_pattern else "win"
            impact = "high" if (is_loss_pattern and loss_rate > 0.75) or \
                     (is_win_pattern and win_rate > 0.85) else "medium"

            # 推荐进化维度
            if is_loss_pattern:
                if alignment == "counter_trend":
                    evol_dim = EVOL_LOGIC  # 增加 MTF 同向过滤
                elif regime == "ranging":
                    evol_dim = EVOL_STRATEGY  # 切换策略
                else:
                    evol_dim = EVOL_PARAMETER  # 调整参数
            else:
                evol_dim = EVOL_KNOWLEDGE  # 优势模式入库

            description = (
                f"{regime}+{alignment} 下{'亏损' if is_loss_pattern else '盈利'}率"
                f"{'%.0f%%' % (loss_rate*100) if is_loss_pattern else '%.0f%%' % (win_rate*100)}, "
                f"{total} 笔, 总 PnL ${total_pnl*self.initial_capital:.0f}"
            )

            patterns.append(PatternFinding(
                pattern_id=f"pat_regime_align_{regime}_{alignment}",
                description=description,
                trade_count=total,
                statistics={
                    "win_rate": float(win_rate),
                    "loss_rate": float(loss_rate),
                    "total_pnl_pct": float(total_pnl),
                    "avg_pnl_pct": float(np.mean(pnls)) if pnls else 0,
                },
                conditions={"regime": regime, "alignment": alignment},
                impact=impact,
                evolution_dimension=evol_dim,
                pattern_type=pattern_type,
            ))

        # 维度 2: (regime, strategy_type)
        groups2 = defaultdict(list)
        for i, attr in enumerate(attributions):
            if i >= len(trades):
                continue
            t = trades[i]
            regime = t.get("regime", "unknown")
            stype = self._infer_strategy_type(t)
            key = (regime, stype)
            groups2[key].append((attr, t))

        for (regime, stype), items in groups2.items():
            if len(items) < PATTERN_MIN_TRADES:
                continue
            pnls = [a.pnl_pct for a, _ in items]
            losses = sum(1 for p in pnls if p < 0)
            total = len(pnls)
            loss_rate = losses / total if total > 0 else 0
            total_pnl = sum(pnls)

            if loss_rate < PATTERN_LOSS_RATE_THRESHOLD and \
               total_pnl * self.initial_capital > PATTERN_PNL_EXTREME:
                continue

            patterns.append(PatternFinding(
                pattern_id=f"pat_regime_strategy_{regime}_{stype}",
                description=f"{stype} 策略在 {regime} 下亏损率 {loss_rate*100:.0f}%, "
                            f"{total} 笔, 总 PnL ${total_pnl*self.initial_capital:.0f}",
                trade_count=total,
                statistics={
                    "loss_rate": float(loss_rate),
                    "total_pnl_pct": float(total_pnl),
                    "avg_pnl_pct": float(np.mean(pnls)) if pnls else 0,
                },
                conditions={"regime": regime, "strategy_type": stype},
                impact="high" if loss_rate > 0.75 else "medium",
                evolution_dimension=EVOL_STRATEGY,
                pattern_type="loss",
            ))

        # 维度 3: (symbol, regime) — symbol×regime 级别
        groups3 = defaultdict(list)
        for i, attr in enumerate(attributions):
            if i >= len(trades):
                continue
            t = trades[i]
            symbol = t.get("symbol", "unknown")
            regime = t.get("regime", "unknown")
            key = (symbol, regime)
            groups3[key].append((attr, t))

        for (symbol, regime), items in groups3.items():
            if len(items) < 5:  # symbol×regime 阈值放宽到 5
                continue
            pnls = [a.pnl_pct for a, _ in items]
            losses = sum(1 for p in pnls if p < 0)
            total = len(pnls)
            loss_rate = losses / total if total > 0 else 0
            total_pnl = sum(pnls)

            if loss_rate < 0.60 and total_pnl * self.initial_capital > -300:
                continue

            patterns.append(PatternFinding(
                pattern_id=f"pat_symbol_regime_{symbol}_{regime}",
                description=f"{symbol}@{regime} 亏损率 {loss_rate*100:.0f}%, "
                            f"{total} 笔, 总 PnL ${total_pnl*self.initial_capital:.0f}",
                trade_count=total,
                statistics={
                    "loss_rate": float(loss_rate),
                    "total_pnl_pct": float(total_pnl),
                    "avg_pnl_pct": float(np.mean(pnls)) if pnls else 0,
                },
                conditions={"symbol": symbol, "regime": regime},
                impact="high" if loss_rate > 0.70 else "medium",
                evolution_dimension=EVOL_PARAMETER,
                pattern_type="loss",
            ))

        # 按影响排序
        impact_order = {"high": 0, "medium": 1, "low": 2}
        patterns.sort(key=lambda p: (impact_order.get(p.impact, 3), -p.trade_count))

        return patterns

    # ========================================================================
    # 5 维进化建议产出
    # ========================================================================

    def _generate_suggestions(self, attributions: List[AttributionResult],
                               patterns: List[PatternFinding],
                               trades: List[dict]) -> List[EvolutionSuggestion]:
        """基于归因和模式, 产出 5 维进化建议

        5 维:
          - parameter: 参数渐进调整 (±5-15%)
          - logic: 策略逻辑迭代 (新增过滤条件)
          - indicator: 指标融合 (新指标加入 entry)
          - knowledge: 金融知识融入 (新发现的市场规律)
          - strategy: 新策略发展/旧策略退役
        """
        suggestions: List[EvolutionSuggestion] = []

        # 按维度生成建议
        suggestions.extend(self._gen_parameter_suggestions(patterns, attributions, trades))
        suggestions.extend(self._gen_logic_suggestions(patterns, attributions, trades))
        suggestions.extend(self._gen_indicator_suggestions(attributions, trades))
        suggestions.extend(self._gen_knowledge_suggestions(patterns, attributions))
        suggestions.extend(self._gen_strategy_suggestions(patterns, attributions, trades))

        # 按优先级排序
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 4))

        # 去重 (同名建议保留最新)
        seen = {}
        for s in suggestions:
            if s.name not in seen or s.priority < seen[s.name].priority:
                seen[s.name] = s
        return list(seen.values())

    def _gen_parameter_suggestions(self, patterns: List[PatternFinding],
                                    attributions: List[AttributionResult],
                                    trades: List[dict]) -> List[EvolutionSuggestion]:
        """参数建议: 找出特定 (symbol, regime) 下表现差的参数, 建议 ±5-15% 调整"""
        suggestions = []

        # 从 symbol×regime 模式生成
        for p in patterns:
            if p.evolution_dimension != EVOL_PARAMETER:
                continue
            conditions = p.conditions
            symbol = conditions.get("symbol", "")
            regime = conditions.get("regime", "")

            if not symbol or not regime:
                continue

            # 建议降低 pos_mult (亏损模式)
            if p.pattern_type == "loss" and p.statistics.get("loss_rate", 0) > 0.6:
                suggestions.append(EvolutionSuggestion(
                    dimension=EVOL_PARAMETER,
                    name=f"reduce_pos_mult_{symbol}_{regime}",
                    evidence={
                        "pattern_id": p.pattern_id,
                        "trade_count": p.trade_count,
                        "loss_rate": p.statistics.get("loss_rate", 0),
                        "total_pnl_pct": p.statistics.get("total_pnl_pct", 0),
                    },
                    proposed_change={
                        "target_symbol": f"{symbol}@{regime}",
                        "param_name": "pos_mult",
                        "adjustment_pct": -15.0,  # 降仓 15%
                        "reason": f"{symbol}@{regime} 亏损率 {p.statistics.get('loss_rate', 0)*100:.0f}%",
                    },
                    expected_impact={
                        "ann_delta_pp": +2.0,
                        "max_dd_delta_pp": -1.0,
                        "loss_reduction_pct": 15.0,
                    },
                    safety_analysis={
                        "risk_level": "low",
                        "rollback_on": ["ann_delta_pp < -3", "max_consec > 3"],
                        "max_affected_trades_pct": 0.10,
                    },
                    priority="high" if p.impact == "high" else "medium",
                    source_patterns=[p.pattern_id],
                ))

        return suggestions

    def _gen_logic_suggestions(self, patterns: List[PatternFinding],
                                attributions: List[AttributionResult],
                                trades: List[dict]) -> List[EvolutionSuggestion]:
        """逻辑建议: 找出特定信号在特定 regime 下失效, 建议增加过滤条件"""
        suggestions = []

        # 从逆势入场模式生成
        for p in patterns:
            if p.evolution_dimension != EVOL_LOGIC:
                continue
            if p.conditions.get("alignment") != "counter_trend":
                continue

            regime = p.conditions.get("regime", "")
            if not regime:
                continue

            # 建议在 regime 下增加 MTF 同向过滤
            suggestions.append(EvolutionSuggestion(
                dimension=EVOL_LOGIC,
                name=f"add_mtf_filter_counter_trend_{regime}",
                evidence={
                    "pattern_id": p.pattern_id,
                    "trade_count": p.trade_count,
                    "loss_rate": p.statistics.get("loss_rate", 0),
                    "total_pnl_pct": p.statistics.get("total_pnl_pct", 0),
                    "counter_trend_count": p.trade_count,
                },
                proposed_change={
                    "trigger": {
                        "regime": regime,
                        "alignment": "counter_trend",
                    },
                    "filter": {
                        "feature": "mtf_htf_trend_4h",
                        "op": "eq_direction",
                        "description": "在 trending 下, 仅允许与 mtf_htf_trend_4h 同向的入场",
                    },
                    "action": "reject_entry",
                    "description": f"在 {regime} 下增加 MTF HTF 同向过滤, 拒绝逆势入场",
                },
                expected_impact={
                    "ann_delta_pp": +3.0,
                    "loss_reduction_pct": 30.0,
                    "trade_count_delta": -p.trade_count,  # 拒绝部分交易
                },
                safety_analysis={
                    "risk_level": "medium",
                    "rollback_on": ["ann_delta_pp < -2", "n_trades_reduction > 30%"],
                    "max_affected_trades_pct": 0.15,
                },
                priority="high",
                source_patterns=[p.pattern_id],
            ))

        return suggestions

    def _gen_indicator_suggestions(self, attributions: List[AttributionResult],
                                    trades: List[dict]) -> List[EvolutionSuggestion]:
        """指标建议: 找出特征值与盈亏强相关的指标, 建议融入 entry 决策"""
        suggestions = []

        if not self._rf_trained or self._rf_model is None:
            # 用 Pearson fallback
            return self._gen_indicator_suggestions_pearson(attributions, trades)

        # 从 RF feature_importances_ 找 Top 特征
        importances = dict(zip(self._rf_feature_names,
                               self._rf_model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

        for fname, imp in top_features:
            if imp < 0.05:  # 重要性太低
                continue

            # 检查该特征在亏损交易中的分布
            loss_trades = [t for i, t in enumerate(trades)
                          if i < len(attributions) and attributions[i].pnl_outcome == "loss"]
            win_trades = [t for i, t in enumerate(trades)
                         if i < len(attributions) and attributions[i].pnl_outcome == "win"]

            if not loss_trades or not win_trades:
                continue

            loss_vals = [float(t.get("features", {}).get(fname, 0) or 0) for t in loss_trades]
            win_vals = [float(t.get("features", {}).get(fname, 0) or 0) for t in win_trades]

            if not loss_vals or not win_vals:
                continue

            loss_mean = float(np.mean(loss_vals))
            win_mean = float(np.mean(win_vals))
            diff = abs(loss_mean - win_mean)
            combined_std = (float(np.std(loss_vals)) + float(np.std(win_vals))) / 2 + 1e-9
            separation = diff / combined_std

            # 区分度高 (>0.3) 才建议
            if separation < 0.3:
                continue

            # 判断方向: 亏损交易均值高 → 设上限; 低 → 设下限
            if loss_mean > win_mean:
                threshold = float(np.percentile(win_vals, 75))
                op = "less_than"
                desc = f"入场时 {fname} < {threshold:.3f}"
            else:
                threshold = float(np.percentile(win_vals, 25))
                op = "greater_than"
                desc = f"入场时 {fname} > {threshold:.3f}"

            suggestions.append(EvolutionSuggestion(
                dimension=EVOL_INDICATOR,
                name=f"fuse_indicator_{fname}",
                evidence={
                    "feature": fname,
                    "importance": float(imp),
                    "loss_mean": loss_mean,
                    "win_mean": win_mean,
                    "separation": float(separation),
                    "loss_trade_count": len(loss_trades),
                    "win_trade_count": len(win_trades),
                },
                proposed_change={
                    "feature": fname,
                    "threshold": threshold,
                    "op": op,
                    "action": "reduce_position_by",
                    "factor": 0.5,  # 触发时降仓 50%
                    "description": desc,
                },
                expected_impact={
                    "ann_delta_pp": +1.5,
                    "loss_reduction_pct": 20.0,
                },
                safety_analysis={
                    "risk_level": "low",
                    "rollback_on": ["ann_delta_pp < -2"],
                    "max_affected_trades_pct": 0.20,
                },
                priority="medium",
            ))

        return suggestions

    def _gen_indicator_suggestions_pearson(self, attributions: List[AttributionResult],
                                            trades: dict) -> List[EvolutionSuggestion]:
        """Pearson fallback: 用方差大的特征作为指标建议"""
        suggestions = []
        if not self._feature_stats:
            return suggestions

        # 方差大的特征
        sorted_features = sorted(
            self._feature_stats.items(),
            key=lambda x: x[1]["std"] / (abs(x[1]["mean"]) + 1e-9),
            reverse=True
        )[:3]

        for fname, stats in sorted_features:
            suggestions.append(EvolutionSuggestion(
                dimension=EVOL_INDICATOR,
                name=f"fuse_indicator_{fname}_pearson",
                evidence={
                    "feature": fname,
                    "method": "pearson_fallback",
                    "std": stats["std"],
                    "mean": stats["mean"],
                },
                proposed_change={
                    "feature": fname,
                    "action": "monitor_only",
                    "description": f"监控 {fname} (std={stats['std']:.3f}), 暂不自动过滤",
                },
                expected_impact={"ann_delta_pp": 0.5},
                safety_analysis={
                    "risk_level": "low",
                    "rollback_on": [],
                },
                priority="low",
            ))

        return suggestions

    def _gen_knowledge_suggestions(self, patterns: List[PatternFinding],
                                    attributions: List[AttributionResult]) -> List[EvolutionSuggestion]:
        """知识建议: 提炼可复用市场规律"""
        suggestions = []

        for p in patterns:
            # 优势模式入库
            if p.pattern_type == "win" and p.statistics.get("win_rate", 0) > 0.70:
                suggestions.append(EvolutionSuggestion(
                    dimension=EVOL_KNOWLEDGE,
                    name=f"knowledge_win_pattern_{p.pattern_id}",
                    evidence={
                        "pattern_id": p.pattern_id,
                        "trade_count": p.trade_count,
                        "win_rate": p.statistics.get("win_rate", 0),
                        "conditions": p.conditions,
                    },
                    proposed_change={
                        "knowledge_type": "winning_pattern",
                        "pattern": p.description,
                        "conditions": p.conditions,
                        "implication": f"在 {p.conditions} 下入场有利, 可适当放大仓位",
                    },
                    expected_impact={"knowledge_value": 1.0},
                    safety_analysis={"risk_level": "low"},
                    priority="medium",
                    source_patterns=[p.pattern_id],
                ))

            # 亏损模式入库 (避免重复犯错)
            elif p.pattern_type == "loss" and p.statistics.get("loss_rate", 0) > 0.65:
                suggestions.append(EvolutionSuggestion(
                    dimension=EVOL_KNOWLEDGE,
                    name=f"knowledge_loss_pattern_{p.pattern_id}",
                    evidence={
                        "pattern_id": p.pattern_id,
                        "trade_count": p.trade_count,
                        "loss_rate": p.statistics.get("loss_rate", 0),
                        "conditions": p.conditions,
                    },
                    proposed_change={
                        "knowledge_type": "loss_pattern",
                        "pattern": p.description,
                        "conditions": p.conditions,
                        "implication": f"避免在 {p.conditions} 下入场, 或显著降仓",
                    },
                    expected_impact={"loss_avoidance_pct": 30.0},
                    safety_analysis={"risk_level": "low"},
                    priority="high" if p.impact == "high" else "medium",
                    source_patterns=[p.pattern_id],
                ))

        return suggestions

    def _gen_strategy_suggestions(self, patterns: List[PatternFinding],
                                   attributions: List[AttributionResult],
                                   trades: List[dict]) -> List[EvolutionSuggestion]:
        """策略建议: 找出整体失效的 (strategy_type, regime) 组合, 建议退役或切换"""
        suggestions = []

        # 从 regime×strategy 模式生成
        for p in patterns:
            if p.evolution_dimension != EVOL_STRATEGY:
                continue
            if p.pattern_type != "loss":
                continue

            conditions = p.conditions
            regime = conditions.get("regime", "")
            stype = conditions.get("strategy_type", "")

            if not regime or not stype:
                continue

            # 高亏损率 → 建议退役
            if p.statistics.get("loss_rate", 0) > 0.70:
                suggestions.append(EvolutionSuggestion(
                    dimension=EVOL_STRATEGY,
                    name=f"retire_strategy_{stype}_{regime}",
                    evidence={
                        "pattern_id": p.pattern_id,
                        "trade_count": p.trade_count,
                        "loss_rate": p.statistics.get("loss_rate", 0),
                        "total_pnl_pct": p.statistics.get("total_pnl_pct", 0),
                    },
                    proposed_change={
                        "strategy_type": stype,
                        "condition": {"regime": regime},
                        "action": "disable",
                        "description": f"在 {regime} 下禁用 {stype} 策略 (亏损率 {p.statistics.get('loss_rate', 0)*100:.0f}%)",
                    },
                    expected_impact={
                        "loss_reduction_pct": 40.0,
                        "ann_delta_pp": +2.0,
                        "trade_count_delta": -p.trade_count,
                    },
                    safety_analysis={
                        "risk_level": "high",
                        "rollback_on": ["ann_delta_pp < -3", "n_trades_reduction > 40%"],
                        "max_affected_trades_pct": 0.20,
                        "requires_validation": True,
                    },
                    priority="critical" if p.impact == "high" else "high",
                    source_patterns=[p.pattern_id],
                ))
            else:
                # 中等亏损 → 建议切换/迭代
                suggestions.append(EvolutionSuggestion(
                    dimension=EVOL_STRATEGY,
                    name=f"iterate_strategy_{stype}_{regime}",
                    evidence={
                        "pattern_id": p.pattern_id,
                        "trade_count": p.trade_count,
                        "loss_rate": p.statistics.get("loss_rate", 0),
                    },
                    proposed_change={
                        "strategy_type": stype,
                        "condition": {"regime": regime},
                        "action": "iterate",
                        "description": f"在 {regime} 下迭代 {stype} 策略, 增加过滤或调整参数",
                    },
                    expected_impact={
                        "loss_reduction_pct": 20.0,
                        "ann_delta_pp": +1.0,
                    },
                    safety_analysis={
                        "risk_level": "medium",
                        "rollback_on": ["ann_delta_pp < -2"],
                    },
                    priority="medium",
                    source_patterns=[p.pattern_id],
                ))

        return suggestions

    # ========================================================================
    # 汇总
    # ========================================================================

    def _build_summary(self, report: AttributionReport) -> dict:
        """构建报告汇总"""
        attr = report.attributions
        wins = sum(1 for a in attr if a.pnl_outcome == "win")
        losses = sum(1 for a in attr if a.pnl_outcome == "loss")
        total_pnl = sum(a.pnl_pct for a in attr)

        # 根因分类统计
        category_counts = Counter(a.root_cause_category for a in attr)

        # 置信度分布
        confidences = [a.confidence for a in attr]
        high_conf = sum(1 for c in confidences if c >= CONFIDENCE_HIGH)
        med_conf = sum(1 for c in confidences if CONFIDENCE_MEDIUM <= c < CONFIDENCE_HIGH)
        low_conf = sum(1 for c in confidences if c < CONFIDENCE_MEDIUM)

        # 建议分维度统计
        suggestions_by_dim = Counter(s.dimension for s in report.suggestions)

        return {
            "total_trades": len(attr),
            "wins": wins,
            "losses": losses,
            "win_rate": float(wins / len(attr)) if attr else 0,
            "total_pnl_pct": float(total_pnl),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0,
            "confidence_distribution": {
                "high": high_conf,
                "medium": med_conf,
                "low": low_conf,
            },
            "top_root_causes": dict(category_counts.most_common(5)),
            "patterns_count": len(report.patterns),
            "suggestions_count": len(report.suggestions),
            "suggestions_by_dimension": dict(suggestions_by_dim),
            "data_quality_distribution": dict(Counter(a.data_quality for a in attr)),
        }

    # ========================================================================
    # 报告保存
    # ========================================================================

    def save_report(self, report: AttributionReport,
                    path: Optional[Path] = None) -> Path:
        """保存归因报告到 JSON 文件

        Args:
            report: 归因报告
            path: 自定义路径 (默认 _attribution_reports/per_trade_attribution_{ts}.json)

        Returns:
            实际保存路径
        """
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = ATTRIBUTION_REPORTS_DIR / f"per_trade_attribution_{ts}.json"

        path = Path(path)
        path.parent.mkdir(exist_ok=True)

        report_dict = report.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)

        # 同时保存独立的 suggestions 文件 (供 EvolutionSuggestionApplier 消费)
        suggestions_path = path.parent / path.name.replace(
            "per_trade_attribution_", "evolution_suggestions_"
        )
        with open(suggestions_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": report.timestamp,
                "suggestions": [s.to_dict() for s in report.suggestions],
            }, f, indent=2, ensure_ascii=False, default=str)

        # 同时保存独立的 patterns 文件
        patterns_path = path.parent / path.name.replace(
            "per_trade_attribution_", "pattern_findings_"
        )
        with open(patterns_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": report.timestamp,
                "patterns": [p.to_dict() for p in report.patterns],
            }, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[Attribution] 报告已保存: {path}")
        logger.info(f"[Attribution] 建议已保存: {suggestions_path}")
        logger.info(f"[Attribution] 模式已保存: {patterns_path}")
        return path


# ============================================================================
# 便捷函数
# ============================================================================

def attribute_trades(trades: List[dict],
                     initial_capital: float = 10000.0,
                     save: bool = True) -> AttributionReport:
    """便捷函数: 一次性归因 + 保存

    Args:
        trades: 交易记录列表
        initial_capital: 初始资金
        save: 是否保存报告

    Returns:
        AttributionReport
    """
    engine = PerTradeAttributionEngine(initial_capital=initial_capital)
    report = engine.attribute_all(trades)
    if save:
        engine.save_report(report)
    return report


# ============================================================================
# 自检
# ============================================================================

def _self_test():
    """自检: 用合成数据验证引擎可运行"""
    print("=" * 70)
    print("PerTradeAttributionEngine 自检")
    print("=" * 70)

    # 合成测试数据 (模拟 v97 raw 格式)
    test_trades = []
    for i in range(30):
        is_win = i % 3 != 0  # 2/3 胜率
        is_counter = i % 5 == 0  # 1/5 逆势
        regime = "trending" if i % 2 == 0 else "ranging"
        direction = "long" if i % 2 == 0 else "short"

        pnl = 0.012 if is_win else -0.008
        mtf_trend = -1 if (direction == "long" and is_counter) else (1 if direction == "long" else -1)

        trade = {
            "symbol": f"TEST_{i % 3}_USDT",
            "direction": direction,
            "pnl_pct": pnl,
            "live_pnl": pnl * 10000,
            "regime": regime,
            "market_state_16": f"{regime}/high_vol/counter_trend" if is_counter else f"{regime}/high_vol/with_trend",
            "htf_state": "counter_trend" if is_counter else "with_trend",
            "htf_trend": mtf_trend,
            "atr_state": "trend",
            "vol_state": "high_vol",
            "pos_mult": 1.5,
            "final_pos_mult": 1.5,
            "bars_held": 5 if is_win else 3,
            "exit_reason": "take_profit" if is_win else "stop_loss",
            "entry_price": 100.0,
            "exit_price": 101.2 if is_win else 99.2,
            "rec": {
                "symbol": f"TEST_{i % 3}_USDT",
                "direction": 1 if direction == "long" else -1,
                "entry_bar": i * 10,
                "hold_bars": 5 if is_win else 3,
                "pnl_pct": pnl,
                "pnl_usd": pnl * 10000,
                "entry_price": 100.0,
                "features": {
                    "adx": 25.0 + i * 0.5,
                    "atr_abs": 1.5,
                    "atr_pct": 0.015,
                    "confluence_score": 6,
                    "confluence_tier": "medium",
                    "rsi_14": 35.0 if direction == "long" else 65.0,
                    "regime": regime,
                    "volatility_regime": "high",
                    "mtf_htf_trend_4h": mtf_trend,
                    "mtf_trend_direction": mtf_trend,
                    "mtf_ema_alignment": mtf_trend,
                    "mtf_short_ema_alignment": mtf_trend,
                    "mtf_market_structure": "range",
                    "mtf_ema50": 99.0,
                    "mtf_ema200": 98.0,
                    "mtf_ema50_200_ratio": 1.01,
                    "pa_body_pct": 1.5,
                    "pa_range_pct": 2.0,
                    "pa_bull_engulf": 1 if direction == "long" else 0,
                    "pa_bear_engulf": 1 if direction == "short" else 0,
                    "vp_obv_slope_norm": -0.8 if is_counter else 0.5,
                    "vp_vol_ratio_5": 1.1,
                    "vp_price_vol_aligned": 1,
                    "kelly_win_rate": 0.55,
                    "kelly_payoff_ratio": 1.5,
                    "kelly_fraction_half": 0.1,
                },
            },
        }
        test_trades.append(trade)

    engine = PerTradeAttributionEngine(initial_capital=10000.0)
    report = engine.attribute_all(test_trades)

    print(f"\n归因完成:")
    print(f"  交易数: {report.trade_count}")
    print(f"  模式数: {len(report.patterns)}")
    print(f"  建议数: {len(report.suggestions)}")
    print(f"  建议分维度: {dict(Counter(s.dimension for s in report.suggestions))}")

    print(f"\n前 3 笔归因示例:")
    for attr in report.attributions[:3]:
        print(f"  [{attr.trade_id}] {attr.direction} pnl={attr.pnl_pct*100:.2f}% "
              f"({attr.pnl_outcome}) | 根因: {attr.root_cause} "
              f"(置信度 {attr.confidence*100:.0f}%)")

    print(f"\n前 3 个模式:")
    for p in report.patterns[:3]:
        print(f"  [{p.pattern_id}] {p.description} → 进化维度: {p.evolution_dimension}")

    print(f"\n前 5 条建议:")
    for s in report.suggestions[:5]:
        print(f"  [{s.dimension}] {s.name} (优先级 {s.priority})")

    # 验证关键字段
    assert report.trade_count == 30, f"交易数不匹配: {report.trade_count}"
    assert len(report.attributions) == 30, "归因结果数不匹配"
    assert all(a.root_cause for a in report.attributions), "存在空根因"
    assert all(0 <= a.confidence <= 1 for a in report.attributions), "置信度越界"

    print(f"\n✅ 自检通过")
    return report


if __name__ == "__main__":
    _self_test()
