#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化复盘系统 (Automated Review System) — 交易铁律：每次回测必须自动复盘

核心功能：
  1. 多币种回测 + 自动复盘报告生成
  2. 交易逐笔分析（入场/出场/MAE/MFE/持仓路径）
  3. 退出原因分布分析（TP/SL/TimeStop/TrailingStop/MaxHold）
  4. 与历史基线对比（检测回归/改进）
  5. 根因假设自动生成（基于统计异常检测）
  6. 可操作建议输出（下一步Fix方向）

设计原则（用户铁律）：
  - "要学会复盘！" — 每次回测后自动生成复盘报告，不是可选
  - "杜绝模拟牛逼，实盘亏钱" — 每笔亏损交易必须追踪根因
  - "理解底层逻辑，而非表面指标" — 报告包含深度分析，不只是数字

使用方式：
  python auto_review_system.py                          # 默认5币种
  python auto_review_system.py --symbols BTC,ETH,SOL    # 指定币种
  python auto_review_system.py --compare baseline.json  # 与基线对比
  python auto_review_system.py --rounds 5                # 自定义轮数

输出：
  - auto_review_report_{timestamp}.json   # 结构化复盘报告
  - auto_review_report_{timestamp}.md     # 可读复盘报告
  - auto_review_baseline.json             # 更新基线（用于下次对比）
"""

import os
import sys
import time
import json
import math
import random
import logging
import statistics
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# ============================================================================
# 固定随机种子 — 确保可复现性（Fix 32a）
# ============================================================================
random.seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
for noisy in ("prophet.plot", "matplotlib", "h5py", "urllib3", "ccxt", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
from sandbox_trading.population import PopulationStats
from sandbox_trading.agent_decision_engine import AgentDecisionEngine

# ============================================================================
# 常量
# ============================================================================
DEFAULT_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "DOGE-USDT"]
TARGET_METRICS = {
    "annual_return": 30.0,      # 年化收益率 ≥ 30%
    "max_drawdown": 15.0,       # 最大回撤 ≤ 15%
    "win_rate": 55.0,           # 胜率 ≥ 55%
    "sharpe_ratio": 1.5,        # 夏普比率 ≥ 1.5
    "profit_factor": 1.5,       # 盈亏比 > 1.5
    "ev_per_trade": 0.5,        # 每笔期望收益 ≥ 0.5%
    "positive_ev_symbols": 5,   # 正EV币种数 ≥ 5/5
}
REVIEW_VERSION = "1.0.0"


# ============================================================================
# 数据结构
# ============================================================================
class TradeRecord:
    """单笔交易完整记录"""
    __slots__ = (
        "trade_id", "symbol", "side", "entry_price", "exit_price",
        "entry_time", "exit_time", "bars_held", "pnl_pct", "pnl_amount",
        "stop_loss", "take_profit", "sl_dist_pct", "tp_dist_pct",
        "exit_reason", "mae_pct", "mfe_pct", "rr_ratio",
        "entry_atr", "entry_rsi", "entry_sma50", "entry_trend",
        "price_path_high", "price_path_low",
    )

    def to_dict(self) -> dict:
        return {k: getattr(self, k, None) for k in self.__slots__}


class SymbolReport:
    """单币种复盘报告"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.trades: List[TradeRecord] = []
        self.total_pnl_pct: float = 0.0
        self.win_rate: float = 0.0
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0
        self.ev_per_trade: float = 0.0
        self.max_drawdown: float = 0.0
        self.profit_factor: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.exit_reasons: Counter = Counter()
        self.long_trades: int = 0
        self.short_trades: int = 0
        self.tp_reachable_rate: float = 0.0  # TP 可达率
        self.sl_avoidable_rate: float = 0.0  # SL 可避免率
        self.mfe_avg: float = 0.0
        self.mae_avg: float = 0.0
        self.issues: List[str] = []  # 发现的问题
        self.strengths: List[str] = []  # 优势

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "total_trades": len(self.trades),
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "win_rate": round(self.win_rate, 1),
            "avg_win_pct": round(self.avg_win, 3),
            "avg_loss_pct": round(self.avg_loss, 3),
            "ev_per_trade_pct": round(self.ev_per_trade, 4),
            "max_drawdown_pct": round(self.max_drawdown, 1),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "exit_reasons": dict(self.exit_reasons),
            "tp_reachable_rate": round(self.tp_reachable_rate, 1),
            "sl_avoidable_rate": round(self.sl_avoidable_rate, 1),
            "mfe_avg_pct": round(self.mfe_avg, 3),
            "mae_avg_pct": round(self.mae_avg, 3),
            "issues": self.issues,
            "strengths": self.strengths,
        }


class ReviewReport:
    """完整复盘报告"""
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.version = REVIEW_VERSION
        self.symbols_tested: List[str] = []
        self.duration_seconds: float = 0.0
        self.symbol_reports: Dict[str, SymbolReport] = {}
        self.aggregate: Dict[str, Any] = {}
        self.kpi_check: Dict[str, Any] = {}  # KPI 达标检查
        self.baseline_comparison: Dict[str, Any] = {}  # 与基线对比
        self.root_cause_hypotheses: List[str] = []  # 根因假设
        self.actionable_recommendations: List[Dict[str, str]] = []  # 可操作建议
        self.next_fix_direction: str = ""  # 下一步 Fix 方向

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "version": self.version,
            "symbols_tested": self.symbols_tested,
            "duration_seconds": round(self.duration_seconds, 1),
            "aggregate": self.aggregate,
            "kpi_check": self.kpi_check,
            "symbols": {s: r.to_dict() for s, r in self.symbol_reports.items()},
            "baseline_comparison": self.baseline_comparison,
            "root_cause_hypotheses": self.root_cause_hypotheses,
            "actionable_recommendations": self.actionable_recommendations,
            "next_fix_direction": self.next_fix_direction,
        }


# ============================================================================
# 核心复盘引擎
# ============================================================================
class AutoReviewEngine:
    """自动化复盘引擎 — 支持两种模式:
        1. EvolutionLoop模式 (run): 通过沙盘回测收集交易数据
        2. 直接JSON模式 (run_from_json): 从已有交易明细JSON直接生成复盘报告
    """

    def __init__(self, symbols: List[str] = None, rounds: int = 3,
                 baseline_path: str = None, data_dir: str = None,
                 initial_capital: float = 10000.0):
        """初始化复盘引擎

        Args:
            symbols: 待复盘币种列表
            rounds: 回测轮数（EvolutionLoop模式）
            baseline_path: 基线JSON路径（用于版本对比）
            data_dir: 数据目录
            initial_capital: 初始资金（美元），用于计算真实资金回撤率。
                修复ERR-20260702-v611尺度差异bug: 原代码cumulative=0/peak=0
                累加pnl_pct小数(0.0129)导致max_dd虚高至85.7%(真实7.34%)。
                现以initial_capital为起点，pnl_pct*initial_capital转为$PnL累加。
        """
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.rounds = rounds
        self.baseline_path = baseline_path
        self.data_dir = data_dir or os.path.join(SCRIPT_DIR, "..", "data", "sandbox_trading")
        self.initial_capital = float(initial_capital)
        self.baseline: Optional[dict] = None
        if baseline_path and os.path.exists(baseline_path):
            with open(baseline_path, "r", encoding="utf-8") as f:
                self.baseline = json.load(f)

    # ========================================================================
    # 直接JSON模式 — 从已有交易明细JSON生成复盘报告（秒级，无环境依赖）
    # ========================================================================

    def run_from_json(self, json_paths: List[str], version_label: str = None,
                      initial_capital: float = None) -> ReviewReport:
        """从交易明细JSON文件直接生成复盘报告

        Args:
            json_paths: 交易明细JSON文件路径列表
            version_label: 版本标签（如 "v578_gap72"），用于报告标识
            initial_capital: 可选初始资金覆盖（默认用self.initial_capital）

        Returns:
            ReviewReport: 完整复盘报告
        """
        # 尺度差异修复: 允许本次运行覆盖initial_capital
        if initial_capital is not None:
            self.initial_capital = float(initial_capital)

        report = ReviewReport()
        t_start = time.time()

        # 从文件名提取版本标签
        if not version_label and json_paths:
            first_name = os.path.basename(json_paths[0])
            version_label = first_name.replace("_trades_detail.json", "")

        report.symbols_tested = []

        # 汇总所有JSON中的交易
        all_trades: List[dict] = []
        for jp in json_paths:
            if not os.path.exists(jp):
                print(f"  ⚠️ 文件不存在，跳过: {jp}")
                continue
            with open(jp, "r", encoding="utf-8") as f:
                trades = json.load(f)
            all_trades.extend(trades)
            print(f"  加载 {os.path.basename(jp)}: {len(trades)} 笔交易")

        if not all_trades:
            report.aggregate = {"error": "无有效交易数据"}
            return report

        # 按币种分组
        trades_by_symbol: Dict[str, List[dict]] = defaultdict(list)
        for t in all_trades:
            symbol = t.get("symbol", "unknown")
            trades_by_symbol[symbol].append(t)

        report.symbols_tested = sorted(trades_by_symbol.keys())

        print(f"\n  总交易: {len(all_trades)} 笔, 币种: {len(trades_by_symbol)} 个")

        # 逐币种分析
        for symbol in report.symbols_tested:
            symbol_trades = trades_by_symbol[symbol]
            print(f"  [{symbol}] {len(symbol_trades)} 笔交易...")
            sr = self._analyze_symbol_from_json(symbol, symbol_trades, self.initial_capital)
            self._diagnose_symbol(sr)
            report.symbol_reports[symbol] = sr

        report.duration_seconds = time.time() - t_start

        # 聚合分析
        self._aggregate_analysis(report)
        self._kpi_check(report)
        self._compare_baseline(report)
        self._generate_recommendations(report)

        return report

    def _analyze_symbol_from_json(self, symbol: str, trades: List[dict],
                                  initial_capital: float = None) -> SymbolReport:
        """从JSON交易记录直接分析单币种（深度分析版）

        分析维度:
          1. 基础统计（胜率/EV/盈亏比/夏普/最大回撤）
          2. 退出原因分布 + 各退出原因的平均PnL
          3. 按市场状态（regime）分组分析
          4. 按方向（long/short）分组分析
          5. 月度P&L稳定性  # [原5:Hurst状态分组已移除-D2审计补遗 ERR-20260704-hurst-audit-gap]
          6. 连续亏损分析
          7. 持仓时长分析
        """
        # 尺度差异修复: 统一使用initial_capital计算资金回撤
        ic = float(initial_capital) if initial_capital is not None else float(getattr(self, "initial_capital", 10000.0))
        sr = SymbolReport(symbol)

        if not trades:
            sr.issues.append("无交易记录")
            return sr

        # 构建TradeRecord列表
        for i, t in enumerate(trades):
            tr = TradeRecord()
            tr.trade_id = f"{symbol}_{i}"
            tr.symbol = t.get("symbol", symbol)
            tr.side = t.get("direction", "unknown")
            tr.entry_price = t.get("entry_price", 0)
            tr.exit_price = t.get("exit_price", 0)
            tr.bars_held = t.get("bars_held", 0)
            tr.pnl_pct = t.get("pnl_pct", 0) * 100  # 转为百分比
            tr.exit_reason = t.get("exit_reason", "unknown")
            sr.trades.append(tr)

        # 计算PnL
        pnls = [t.get("pnl_pct", 0) for t in trades]  # 已经是小数形式
        net_pnls = [t.get("net_pnl_pct", t.get("pnl_pct", 0)) for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        sr.total_pnl_pct = sum(net_pnls) * 100  # 转为百分比
        sr.win_rate = len(winners) / len(pnls) * 100 if pnls else 0
        sr.avg_win = statistics.mean(winners) * 100 if winners else 0
        sr.avg_loss = abs(statistics.mean(losers)) * 100 if losers else 0
        sr.ev_per_trade = (sr.win_rate / 100) * sr.avg_win - (1 - sr.win_rate / 100) * sr.avg_loss

        # 盈亏比
        if losers and sum(losers) != 0:
            sr.profit_factor = abs(sum(winners) / sum(losers)) if sum(losers) != 0 else 0
        else:
            sr.profit_factor = 999 if winners else 0

        # 最大回撤 (尺度差异修复 ERR-20260702-v611)
        # 原bug: cumulative=0/peak=0累加pnl_pct小数(0.0129)→除零+虚高85.7%
        # 修复: 以initial_capital为起点, pnl_pct*ic转为$PnL累加, 得真实资金回撤率
        cumulative = ic
        peak = ic
        max_dd = 0
        for p in net_pnls:
            cumulative += p * ic  # 小数收益率 → $PnL
            peak = max(peak, cumulative)
            if peak > 0:
                dd = (peak - cumulative) / peak * 100
                max_dd = max(max_dd, dd)
        sr.max_drawdown = max_dd

        # 夏普比率
        if len(pnls) >= 2:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                sr.sharpe_ratio = mean_pnl / std_pnl * math.sqrt(min(252, len(pnls)))
            else:
                sr.sharpe_ratio = 0.0
        else:
            sr.sharpe_ratio = 0.0

        # 退出原因分布
        exit_reasons = Counter(t.get("exit_reason", "unknown") for t in trades)
        sr.exit_reasons = exit_reasons

        # 多空统计
        sr.long_trades = sum(1 for t in trades if t.get("direction") == "long")
        sr.short_trades = sum(1 for t in trades if t.get("direction") == "short")

        # TP可达率 / SL可避免率
        tp_count = exit_reasons.get("take_profit", 0)
        sl_count = exit_reasons.get("stop_loss", 0)
        sr.tp_reachable_rate = tp_count / len(trades) * 100 if trades else 0
        sr.sl_avoidable_rate = (len(trades) - sl_count) / len(trades) * 100 if trades else 0

        # MAE/MFE
        sr.mae_avg = abs(statistics.mean(losers)) * 100 if losers else 0
        sr.mfe_avg = statistics.mean(winners) * 100 if winners else 0

        # ===== 深度分析 =====

        # 按退出原因分析平均PnL
        exit_pnl = defaultdict(list)
        for t in trades:
            exit_pnl[t.get("exit_reason", "unknown")].append(t.get("pnl_pct", 0))
        exit_avg_pnl = {k: statistics.mean(v) * 100 for k, v in exit_pnl.items()}

        # 按regime分析
        regime_stats = defaultdict(lambda: {"count": 0, "wins": 0, "pnls": []})
        for t in trades:
            r = t.get("regime", "unknown")
            regime_stats[r]["count"] += 1
            pnl = t.get("pnl_pct", 0)
            regime_stats[r]["pnls"].append(pnl)
            if pnl > 0:
                regime_stats[r]["wins"] += 1

        # 按Hurst状态分析 [已移除-D2审计补遗 ERR-20260704-hurst-audit-gap]
        # 原因: Hurst未在feature_extractors.py计算, 所有交易hurst_regime均为"unknown", 分组无信息量
        # hurst_stats = defaultdict(lambda: {"count": 0, "wins": 0, "pnls": []})
        # for t in trades:
        #     h = t.get("hurst_regime", "unknown")
        #     hurst_stats[h]["count"] += 1
        #     pnl = t.get("pnl_pct", 0)
        #     hurst_stats[h]["pnls"].append(pnl)
        #     if pnl > 0:
        #         hurst_stats[h]["wins"] += 1

        # 月度P&L
        monthly_pnl = defaultdict(float)
        for t in trades:
            mk = t.get("month_key", "unknown")
            monthly_pnl[mk] += t.get("net_pnl_pct", t.get("pnl_pct", 0)) * 100

        # 连续亏损分析
        max_cl = 0
        current_cl = 0
        cl_sequences = []
        for p in pnls:
            if p <= 0:
                current_cl += 1
            else:
                if current_cl > 0:
                    cl_sequences.append(current_cl)
                current_cl = 0
            max_cl = max(max_cl, current_cl)
        if current_cl > 0:
            cl_sequences.append(current_cl)

        # 持仓时长分析
        bars_held_list = [t.get("bars_held", 0) for t in trades]
        bars_win = [t.get("bars_held", 0) for t in trades if t.get("pnl_pct", 0) > 0]
        bars_loss = [t.get("bars_held", 0) for t in trades if t.get("pnl_pct", 0) <= 0]

        # 存储深度分析数据到SymbolReport（扩展属性）
        sr._depth_analysis = {
            "exit_avg_pnl": exit_avg_pnl,
            "regime_stats": {k: {
                "count": v["count"],
                "win_rate": v["wins"] / max(v["count"], 1) * 100,
                "avg_pnl": statistics.mean(v["pnls"]) * 100 if v["pnls"] else 0,
            } for k, v in regime_stats.items()},
            # hurst_stats 已移除 (D2审计补遗 ERR-20260704-hurst-audit-gap): Hurst未计算, 分组无信息量
            "monthly_pnl": dict(monthly_pnl),
            "monthly_loss_prob": sum(1 for v in monthly_pnl.values() if v < 0) / max(len(monthly_pnl), 1) * 100,
            "max_consecutive_losses": max_cl,
            "cl_sequences": cl_sequences,
            "avg_bars_held": statistics.mean(bars_held_list) if bars_held_list else 0,
            "avg_bars_win": statistics.mean(bars_win) if bars_win else 0,
            "avg_bars_loss": statistics.mean(bars_loss) if bars_loss else 0,
            "score_distribution": Counter(t.get("score", 0) for t in trades),
            "direction_split": {
                "long": sum(1 for t in trades if t.get("direction") == "long"),
                "short": sum(1 for t in trades if t.get("direction") == "short"),
            },
        }

        return sr

    def run_all_versions(self, pattern: str = "_v*_trades_detail.json") -> Dict[str, ReviewReport]:
        """批量复盘所有版本

        Args:
            pattern: 文件匹配模式

        Returns:
            Dict[str, ReviewReport]: version_label -> 复盘报告
        """
        import glob as glob_mod

        json_files = sorted(glob_mod.glob(os.path.join(SCRIPT_DIR, pattern)))
        if not json_files:
            print("未找到交易明细JSON文件")
            return {}

        print(f"发现 {len(json_files)} 个版本")
        print("=" * 70)

        all_reports = {}
        for i, jf in enumerate(json_files, 1):
            version_label = os.path.basename(jf).replace("_trades_detail.json", "")
            print(f"\n[{i}/{len(json_files)}] 🔍 {version_label}")
            try:
                report = self.run_from_json([jf], version_label=version_label)
                all_reports[version_label] = report
                # 保存单版本报告
                save_report(report, os.path.join(SCRIPT_DIR, "_review_reports"))
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                import traceback
                traceback.print_exc()

        # 生成进化趋势报告
        if len(all_reports) >= 2:
            self._generate_evolution_report(all_reports)

        return all_reports

    def _generate_evolution_report(self, all_reports: Dict[str, ReviewReport]):
        """生成版本进化趋势报告"""
        output_dir = os.path.join(SCRIPT_DIR, "_review_reports")
        os.makedirs(output_dir, exist_ok=True)

        # 提取各版本聚合指标
        versions = sorted(all_reports.keys())
        evolution_data = []
        for v in versions:
            agg = all_reports[v].aggregate
            if agg.get("error"):
                continue
            evolution_data.append({
                "version": v,
                "pnl_mean": agg.get("pnl_mean_pct", 0),
                "win_rate": agg.get("win_rate_mean", 0),
                "ev_mean": agg.get("ev_mean_pct", 0),
                "sharpe": agg.get("sharpe_mean", 0),
                "dd_mean": agg.get("dd_mean_pct", 0),
                "total_trades": agg.get("total_trades", 0),
                "positive_ev": agg.get("positive_ev_count", 0),
                "positive_pnl": agg.get("positive_pnl_count", 0),
                "symbols": agg.get("symbols_count", 0),
            })

        # 生成Markdown进化报告
        lines = []
        lines.append("# 📈 策略进化趋势报告 (v552 → v578)")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        lines.append(f"**版本数**: {len(evolution_data)}")
        lines.append("")
        lines.append("## 指标趋势")
        lines.append("")
        lines.append("| 版本 | 交易数 | 币种 | 平均PnL% | 胜率% | EV% | 夏普 | 回撤% | 正EV | 正PnL |")
        lines.append("|------|--------|------|----------|-------|-----|------|-------|------|-------|")

        prev_ev = None
        for d in evolution_data:
            # 标记EV变化方向
            ev_marker = ""
            if prev_ev is not None and d["ev_mean"] != 0:
                if d["ev_mean"] > prev_ev:
                    ev_marker = " 📈"
                elif d["ev_mean"] < prev_ev:
                    ev_marker = " 📉"

            lines.append(
                f"| {d['version']} | {d['total_trades']} | {d['symbols']} | "
                f"{d['pnl_mean']:.2f} | {d['win_rate']:.1f} | "
                f"{d['ev_mean']:.4f}{ev_marker} | {d['sharpe']:.3f} | "
                f"{d['dd_mean']:.1f} | {d['positive_ev']}/{d['symbols']} | "
                f"{d['positive_pnl']}/{d['symbols']} |"
            )
            prev_ev = d["ev_mean"]

        lines.append("")

        # 关键发现
        lines.append("## 关键发现")
        lines.append("")

        if evolution_data:
            # 最佳版本
            best = max(evolution_data, key=lambda d: d["ev_mean"])
            worst = min(evolution_data, key=lambda d: d["ev_mean"])
            lines.append(f"- 🏆 **最佳版本**: {best['version']} (EV={best['ev_mean']:.4f}%, 胜率={best['win_rate']:.1f}%)")
            lines.append(f"- 🔻 **最差版本**: {worst['version']} (EV={worst['ev_mean']:.4f}%, 胜率={worst['win_rate']:.1f}%)")

            # EV趋势
            first_ev = evolution_data[0]["ev_mean"]
            last_ev = evolution_data[-1]["ev_mean"]
            ev_change = last_ev - first_ev
            lines.append(f"- 📊 **EV变化**: {first_ev:.4f}% → {last_ev:.4f}% (Δ={ev_change:+.4f}%)")

            if ev_change > 0:
                lines.append(f"- ✅ 整体进化方向正确，EV持续改善")
            else:
                lines.append(f"- ⚠️ 整体EV未改善，需检查架构变更是否引入回归")

            # 正EV版本比例
            positive_ev_versions = sum(1 for d in evolution_data if d["ev_mean"] > 0)
            lines.append(f"- 📈 **正EV版本比例**: {positive_ev_versions}/{len(evolution_data)} ({positive_ev_versions/len(evolution_data)*100:.1f}%)")

        lines.append("")

        # 保存进化报告
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        evo_path = os.path.join(output_dir, f"evolution_report_{ts}.md")
        with open(evo_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n✅ 进化趋势报告: {evo_path}")

        # 也保存JSON
        evo_json_path = os.path.join(output_dir, f"evolution_report_{ts}.json")
        with open(evo_json_path, "w", encoding="utf-8") as f:
            json.dump(evolution_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 进化趋势JSON: {evo_json_path}")

    def compare_versions(self, version_a: str, version_b: str) -> dict:
        """对比两个版本的复盘报告"""
        # 查找文件
        file_a = os.path.join(SCRIPT_DIR, f"{version_a}_trades_detail.json")
        file_b = os.path.join(SCRIPT_DIR, f"{version_b}_trades_detail.json")

        if not os.path.exists(file_a):
            return {"error": f"版本 {version_a} 文件不存在: {file_a}"}
        if not os.path.exists(file_b):
            return {"error": f"版本 {version_b} 文件不存在: {file_b}"}

        report_a = self.run_from_json([file_a], version_label=version_a)
        report_b = self.run_from_json([file_b], version_label=version_b)

        agg_a = report_a.aggregate
        agg_b = report_b.aggregate

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics": {},
        }

        for key in ["pnl_mean_pct", "win_rate_mean", "ev_mean_pct", "sharpe_mean", "dd_mean_pct", "total_trades"]:
            va = agg_a.get(key, 0)
            vb = agg_b.get(key, 0)
            delta = vb - va
            comparison["metrics"][key] = {
                "a": round(va, 4),
                "b": round(vb, 4),
                "delta": round(delta, 4),
                "improved": (key != "dd_mean_pct" and delta > 0) or (key == "dd_mean_pct" and delta < 0),
            }

        # 逐币种对比
        comparison["by_symbol"] = {}
        all_symbols = set(report_a.symbol_reports.keys()) | set(report_b.symbol_reports.keys())
        for sym in sorted(all_symbols):
            sr_a = report_a.symbol_reports.get(sym)
            sr_b = report_b.symbol_reports.get(sym)
            comparison["by_symbol"][sym] = {
                "ev_a": round(sr_a.ev_per_trade, 4) if sr_a else None,
                "ev_b": round(sr_b.ev_per_trade, 4) if sr_b else None,
                "win_rate_a": round(sr_a.win_rate, 1) if sr_a else None,
                "win_rate_b": round(sr_b.win_rate, 1) if sr_b else None,
                "trades_a": len(sr_a.trades) if sr_a else 0,
                "trades_b": len(sr_b.trades) if sr_b else 0,
            }

        return comparison

    def run(self) -> ReviewReport:
        """运行完整复盘流程"""
        report = ReviewReport()
        report.symbols_tested = list(self.symbols)
        t_start = time.time()

        # 保存原始方法引用
        from sandbox_trading.evolution_loop import EvolutionLoop  # 延迟导入避免循环依赖
        _ORIG_CHECK_EXIT = EvolutionLoop._check_exit
        _ORIG_DECIDE = AgentDecisionEngine.decide

        for sym_idx, symbol in enumerate(self.symbols, 1):
            print(f"\n{'='*70}")
            print(f"[{sym_idx}/{len(self.symbols)}] 🔍 复盘 {symbol}")
            print(f"{'='*70}")

            # 恢复原始方法
            EvolutionLoop._check_exit = _ORIG_CHECK_EXIT
            AgentDecisionEngine.decide = _ORIG_DECIDE

            # 收集器
            exit_diag = self._make_exit_collector(_ORIG_CHECK_EXIT)
            decide_diag = self._make_decide_collector(_ORIG_DECIDE)
            price_paths = defaultdict(list)
            active_trades = {}

            # 打补丁
            def _patched_check_exit(self_, trade, market, gene, tick=None):
                return exit_diag["handler"](self_, trade, market, gene, tick)

            def _patched_decide(self_, data_packet, balance, equity, frontier=None):
                return decide_diag["handler"](self_, data_packet, balance, equity, frontier)

            EvolutionLoop._check_exit = _patched_check_exit
            AgentDecisionEngine.decide = _patched_decide

            # 初始化
            print(f"  [1/4] 初始化 {symbol}...")
            t0 = time.time()
            try:
                loop = EvolutionLoop(
                    population_size=30, initial_capital=10000.0,
                    symbols=[symbol], data_dir=self.data_dir,
                    max_generations=1,
                )
                loop.initialize(bars_per_symbol=2000, use_mock=False, seed_count=0, use_intraday=True)

                def _patched_run_evolution(*args, **kwargs):
                    return PopulationStats(generation=0, population_size=30, diversity=1.0)
                loop.population.run_evolution_round = _patched_run_evolution
                print(f"      初始化完成 ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"      ❌ 初始化失败: {e}")
                report.symbol_reports[symbol] = SymbolReport(symbol)
                report.symbol_reports[symbol].issues.append(f"初始化失败: {e}")
                continue

            # 运行
            print(f"  [2/4] 运行 {self.rounds} 轮...")
            t1 = time.time()
            try:
                loop.run(rounds=self.rounds, evolve_every=999, ticks_per_round=24)
                print(f"      完成 ({time.time()-t1:.1f}s)")
            except Exception as e:
                print(f"      ❌ 运行失败: {e}")
                report.symbol_reports[symbol] = SymbolReport(symbol)
                report.symbol_reports[symbol].issues.append(f"运行失败: {e}")
                continue

            # 分析
            print(f"  [3/4] 分析交易数据...")
            sr = self._analyze_symbol(symbol, exit_diag, decide_diag, self.initial_capital)
            report.symbol_reports[symbol] = sr

            # 诊断
            print(f"  [4/4] 自动诊断...")
            self._diagnose_symbol(sr)

        # 恢复原始方法
        EvolutionLoop._check_exit = _ORIG_CHECK_EXIT
        AgentDecisionEngine.decide = _ORIG_DECIDE

        report.duration_seconds = time.time() - t_start

        # 聚合分析
        self._aggregate_analysis(report)
        self._kpi_check(report)
        self._compare_baseline(report)
        self._generate_recommendations(report)

        return report

    def _make_exit_collector(self, orig_check_exit):
        """创建退出收集器"""
        diag = {
            "calls": 0, "exits": [], "exit_reasons": Counter(),
            "long_count": 0, "short_count": 0,
        }

        def handler(self_, trade, market, gene, tick=None):
            diag["calls"] += 1
            result = orig_check_exit(self_, trade, market, gene, tick)
            if result:
                entry = trade.entry_price
                exit_price = getattr(trade, 'forced_exit_price', 0.0) or market.close
                side = trade.side.value if hasattr(trade.side, 'value') else str(trade.side)
                if side == "long":
                    diag["long_count"] += 1
                else:
                    diag["short_count"] += 1

                pnl_pct = (exit_price - entry) / entry if side == "long" else (entry - exit_price) / entry
                sl = getattr(trade, 'stop_loss', 0.0)
                tp = getattr(trade, 'take_profit', 0.0)
                bars_held = getattr(trade, 'bars_held', 0)
                reason = self._classify_exit_reason(side, exit_price, sl, tp, bars_held, pnl_pct)
                diag["exit_reasons"][reason] += 1

                sl_dist_pct = abs(sl - entry) / entry * 100 if sl > 0 else 0
                tp_dist_pct = abs(tp - entry) / entry * 100 if tp > 0 else 0

                diag["exits"].append({
                    "side": side, "entry": round(entry, 4),
                    "exit": round(exit_price, 4),
                    "sl": round(sl, 4), "tp": round(tp, 4),
                    "sl_dist_pct": round(sl_dist_pct, 3),
                    "tp_dist_pct": round(tp_dist_pct, 3),
                    "pnl_pct": round(pnl_pct * 100, 4),
                    "bars_held": bars_held,
                    "reason": reason,
                })
            return result

        diag["handler"] = handler
        return diag

    def _make_decide_collector(self, orig_decide):
        """创建决策收集器"""
        diag = {
            "calls": 0, "signals": [], "long_signals": 0, "short_signals": 0,
            "neutral_signals": 0, "atr_values": [],
        }

        def handler(self_, data_packet, balance, equity, frontier=None):
            diag["calls"] += 1
            indicators = data_packet.get("indicators", {}) if data_packet else {}
            atr = indicators.get("atr_14", 0.0) if indicators else 0.0
            diag["atr_values"].append(atr)
            result = orig_decide(self_, data_packet, balance, equity, frontier)
            if result:
                diag["signals"].append(result)
                if result.get("side") == "long":
                    diag["long_signals"] += 1
                elif result.get("side") == "short":
                    diag["short_signals"] += 1
                else:
                    diag["neutral_signals"] += 1
            return result

        diag["handler"] = handler
        return diag

    @staticmethod
    def _classify_exit_reason(side, exit_price, sl, tp, bars_held, pnl_pct):
        """分类退出原因"""
        if sl > 0 and tp > 0:
            if side == "long":
                if exit_price <= sl * 1.001:
                    return "stop_loss"
                elif exit_price >= tp * 0.999:
                    return "take_profit"
                elif bars_held >= 10:
                    return "max_hold"
                elif bars_held >= 3 and pnl_pct < -0.005:
                    return "time_stop"
                else:
                    return "trailing_stop"
            else:
                if exit_price >= sl * 0.999:
                    return "stop_loss"
                elif exit_price <= tp * 1.001:
                    return "take_profit"
                elif bars_held >= 10:
                    return "max_hold"
                elif bars_held >= 3 and pnl_pct < -0.005:
                    return "time_stop"
                else:
                    return "trailing_stop"
        return "unknown"

    def _analyze_symbol(self, symbol: str, exit_diag: dict, decide_diag: dict,
                        initial_capital: float = None) -> SymbolReport:
        """分析单币种交易数据"""
        # 尺度差异修复: 统一使用initial_capital计算资金回撤
        ic = float(initial_capital) if initial_capital is not None else float(getattr(self, "initial_capital", 10000.0))
        sr = SymbolReport(symbol)
        exits = exit_diag["exits"]

        if not exits:
            sr.issues.append("无平仓记录")
            return sr

        pnls = [e["pnl_pct"] for e in exits]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        sr.total_pnl_pct = sum(pnls)
        sr.win_rate = len(winners) / len(pnls) * 100 if pnls else 0
        sr.avg_win = statistics.mean(winners) if winners else 0
        sr.avg_loss = abs(statistics.mean(losers)) if losers else 0
        sr.ev_per_trade = (sr.win_rate / 100) * sr.avg_win - (1 - sr.win_rate / 100) * sr.avg_loss
        sr.profit_factor = (sum(winners) / abs(sum(losers))) if losers and sum(losers) != 0 else 0
        sr.exit_reasons = exit_diag["exit_reasons"]
        sr.long_trades = exit_diag["long_count"]
        sr.short_trades = exit_diag["short_count"]

        # 最大回撤 (尺度差异修复 ERR-20260702-v611)
        # 原bug: cumulative=0/peak=0→除零+虚高; 修复: 以initial_capital为起点
        cumulative = ic
        peak = ic
        max_dd = 0
        for p in pnls:
            cumulative += p * ic  # 小数收益率 → $PnL
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        sr.max_drawdown = max_dd

        # 夏普比率
        if len(pnls) >= 2:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                sr.sharpe_ratio = mean_pnl / std_pnl * math.sqrt(min(252, len(pnls)))
            else:
                sr.sharpe_ratio = 0.0
        else:
            sr.sharpe_ratio = 0.0

        # TP 可达率 / SL 可避免率
        tp_reachable = 0
        sl_avoidable = 0
        for e in exits:
            if e["reason"] == "take_profit":
                tp_reachable += 1
            if e["reason"] == "take_profit" or (e["pnl_pct"] > 0 and e["reason"] != "stop_loss"):
                sl_avoidable += 1
        sr.tp_reachable_rate = tp_reachable / len(exits) * 100 if exits else 0
        sr.sl_avoidable_rate = sl_avoidable / len(exits) * 100 if exits else 0

        # MAE/MFE
        sr.mae_avg = abs(statistics.mean([p for p in pnls if p <= 0])) if losers else 0
        sr.mfe_avg = statistics.mean(winners) if winners else 0

        return sr

    def _diagnose_symbol(self, sr: SymbolReport):
        """自动诊断单币种问题"""
        # 检测1: 胜率过低
        if sr.win_rate < 40:
            sr.issues.append(
                f"胜率 {sr.win_rate:.1f}% < 40%: 信号方向可能系统性错误, "
                f"需检查入场条件是否在趋势顶部/底部"
            )
        elif sr.win_rate < 50:
            sr.issues.append(
                f"胜率 {sr.win_rate:.1f}% < 50%: 低于随机水平, "
                f"止损可能过紧或入场时机不佳"
            )

        # 检测2: EV 为负
        if sr.ev_per_trade < 0:
            sr.issues.append(
                f"EV = {sr.ev_per_trade:.4f}% < 0: 每笔交易期望为负, "
                f"策略在此币种无优势, 需架构性修复"
            )

        # 检测3: TP 可达率过低
        if sr.tp_reachable_rate < 10:
            sr.issues.append(
                f"TP 触发率 {sr.tp_reachable_rate:.1f}% < 10%: "
                f"TP 距离过远, 在 max_hold 内不可达, 需调整 TP 距离"
            )

        # 检测4: SL 触发率过高
        sl_rate = sr.exit_reasons.get("stop_loss", 0) / max(len(sr.trades), 1) * 100
        if sl_rate > 50:
            sr.issues.append(
                f"SL 触发率 {sl_rate:.1f}% > 50%: 止损过紧或入场时机在趋势反转点, "
                f"需检查 SL 距离是否小于噪声幅度"
            )

        # 检测5: TimeStop 过高
        ts_rate = sr.exit_reasons.get("time_stop", 0) / max(len(sr.trades), 1) * 100
        if ts_rate > 30:
            sr.issues.append(
                f"TimeStop 率 {ts_rate:.1f}% > 30%: 持仓时间过长, "
                f"TP 和 SL 均未触发, 可能 ATR 估计过大导致 TP/SL 距离过远"
            )

        # 检测6: MaxHold 过高
        mh_rate = sr.exit_reasons.get("max_hold", 0) / max(len(sr.trades), 1) * 100
        if mh_rate > 20:
            sr.issues.append(
                f"MaxHold 率 {mh_rate:.1f}% > 20%: 持仓到期未触发 TP/SL, "
                f"信号方向可能正确但波动不足, 或 TP/SL 距离过大"
            )

        # 检测7: 盈亏比失衡
        if sr.avg_win > 0 and sr.avg_loss > 0:
            rr = sr.avg_win / sr.avg_loss
            if rr < 0.5:
                sr.issues.append(
                    f"盈亏比 {rr:.2f} < 0.5: 平均盈利远小于平均亏损, "
                    f"即使胜率 > 66% 也很难盈利"
                )

        # 检测优势
        if sr.win_rate >= 55:
            sr.strengths.append(f"胜率 {sr.win_rate:.1f}% ≥ 55%: 信号方向正确")
        if sr.ev_per_trade > 0:
            sr.strengths.append(f"EV = {sr.ev_per_trade:.4f}% > 0: 正期望")
        if sr.profit_factor > 1.0:
            sr.strengths.append(f"盈亏比 {sr.profit_factor:.2f} > 1.0: 盈利覆盖亏损")
        if sr.tp_reachable_rate >= 20:
            sr.strengths.append(f"TP 触发率 {sr.tp_reachable_rate:.1f}%: 止盈可达")

    def _aggregate_analysis(self, report: ReviewReport):
        """聚合分析"""
        valid_reports = [r for r in report.symbol_reports.values() if r.trades or r.total_pnl_pct != 0]
        if not valid_reports:
            report.aggregate = {"error": "无有效数据"}
            return

        n = len(valid_reports)
        pnls = [r.total_pnl_pct for r in valid_reports]
        win_rates = [r.win_rate for r in valid_reports]
        evs = [r.ev_per_trade for r in valid_reports]
        sharpe = [r.sharpe_ratio for r in valid_reports]
        dds = [r.max_drawdown for r in valid_reports]

        positive_ev = sum(1 for e in evs if e > 0)
        positive_pnl = sum(1 for p in pnls if p > 0)

        # v704 修复 (ERR-v704-annual-return-naming): 计算 total_days 用于真实年化
        # 原: annual_return = pnl_mean_pct (未年化的 per-symbol 均值, 命名错误)
        # 新: annual_return = pnl_mean_pct × (365 / total_days) (真实年化)
        _all_times = []
        for _r in valid_reports:
            for _t in _r.trades:
                _et = getattr(_t, "entry_time", None)
                _xt = getattr(_t, "exit_time", None)
                if _et:
                    _all_times.append(_et)
                if _xt:
                    _all_times.append(_xt)
        _total_days = 0.0
        if _all_times:
            _min_t = min(_all_times)
            _max_t = max(_all_times)
            # 处理 datetime 对象或 ISO 字符串
            if isinstance(_min_t, str):
                try:
                    _min_t = datetime.fromisoformat(_min_t.replace("Z", "+00:00"))
                except Exception:
                    _min_t = None
            if isinstance(_max_t, str):
                try:
                    _max_t = datetime.fromisoformat(_max_t.replace("Z", "+00:00"))
                except Exception:
                    _max_t = None
            if _min_t and _max_t:
                try:
                    _total_days = max(1.0, (_max_t - _min_t).total_seconds() / 86400.0)
                except Exception:
                    _total_days = 0.0

        report.aggregate = {
            "symbols_count": n,
            "positive_ev_count": positive_ev,
            "positive_pnl_count": positive_pnl,
            "pnl_mean_pct": round(statistics.mean(pnls), 2),
            "pnl_stdev_pct": round(statistics.stdev(pnls), 2) if n >= 2 else 0,
            "win_rate_mean": round(statistics.mean(win_rates), 1),
            "ev_mean_pct": round(statistics.mean(evs), 4),
            "ev_stdev_pct": round(statistics.stdev(evs), 4) if n >= 2 else 0,
            "sharpe_mean": round(statistics.mean(sharpe), 3),
            "total_days": round(_total_days, 1),  # v704 新增: 用于真实年化计算
            "dd_mean_pct": round(statistics.mean(dds), 1),
            "total_trades": sum(len(r.trades) for r in valid_reports),
        }

        # 按币种排序
        report.aggregate["best_symbol"] = max(valid_reports, key=lambda r: r.total_pnl_pct).symbol
        report.aggregate["worst_symbol"] = min(valid_reports, key=lambda r: r.total_pnl_pct).symbol

    def _kpi_check(self, report: ReviewReport):
        """KPI 达标检查"""
        agg = report.aggregate
        targets = TARGET_METRICS

        # v704 修复 (ERR-v704-annual-return-naming): 计算真实年化收益率
        # 原: annual_return = pnl_mean_pct (未年化的 per-symbol 均值, 命名错误)
        # 新: annual_return = pnl_mean_pct × (365 / total_days) (真实年化)
        _total_days = agg.get("total_days", 0)
        _pnl_mean = agg.get("pnl_mean_pct", 0)
        if _total_days > 0:
            _annualized = _pnl_mean * (365.0 / _total_days)
        else:
            _annualized = 0.0

        report.kpi_check = {
            "annual_return": {
                "target": f"≥ {targets['annual_return']}%",
                "actual": f"{_annualized:.1f}%",
                "passed": _annualized >= targets["annual_return"],
            },
            "max_drawdown": {
                "target": f"≤ {targets['max_drawdown']}%",
                "actual": f"{agg.get('dd_mean_pct', 0):.1f}%",
                "passed": agg.get("dd_mean_pct", 0) <= targets["max_drawdown"],
            },
            "win_rate": {
                "target": f"≥ {targets['win_rate']}%",
                "actual": f"{agg.get('win_rate_mean', 0):.1f}%",
                "passed": agg.get("win_rate_mean", 0) >= targets["win_rate"],
            },
            "sharpe_ratio": {
                "target": f"≥ {targets['sharpe_ratio']}",
                "actual": f"{agg.get('sharpe_mean', 0):.3f}",
                "passed": agg.get("sharpe_mean", 0) >= targets["sharpe_ratio"],
            },
            "ev_per_trade": {
                "target": f"≥ {targets['ev_per_trade']}%",
                "actual": f"{agg.get('ev_mean_pct', 0):.4f}%",
                "passed": agg.get("ev_mean_pct", 0) >= targets["ev_per_trade"],
            },
            "positive_ev_symbols": {
                "target": f"≥ {targets['positive_ev_symbols']}/5",
                "actual": f"{agg.get('positive_ev_count', 0)}/5",
                "passed": agg.get("positive_ev_count", 0) >= targets["positive_ev_symbols"],
            },
        }

        passed = sum(1 for v in report.kpi_check.values() if v["passed"])
        report.kpi_check["summary"] = f"{passed}/{len(report.kpi_check)} KPI 达标"

    def _compare_baseline(self, report: ReviewReport):
        """与基线对比"""
        if not self.baseline:
            report.baseline_comparison = {"status": "无基线数据"}
            return

        baseline_agg = self.baseline.get("aggregate", {})
        current_agg = report.aggregate

        changes = {}
        for key in ["pnl_mean_pct", "win_rate_mean", "ev_mean_pct", "sharpe_mean", "dd_mean_pct"]:
            old_val = baseline_agg.get(key, 0)
            new_val = current_agg.get(key, 0)
            if old_val != 0:
                delta = new_val - old_val
                changes[key] = {
                    "baseline": round(old_val, 4),
                    "current": round(new_val, 4),
                    "delta": round(delta, 4),
                    "direction": "improved" if (
                        (key in ["pnl_mean_pct", "win_rate_mean", "ev_mean_pct", "sharpe_mean"] and delta > 0)
                        or (key == "dd_mean_pct" and delta < 0)
                    ) else "degraded" if delta != 0 else "unchanged",
                }

        report.baseline_comparison = {
            "baseline_timestamp": self.baseline.get("timestamp", "unknown"),
            "changes": changes,
            "regression_detected": any(
                c["direction"] == "degraded" for c in changes.values()
            ),
        }

    def _generate_recommendations(self, report: ReviewReport):
        """生成可操作建议"""
        all_issues = []
        for sr in report.symbol_reports.values():
            for issue in sr.issues:
                all_issues.append(f"[{sr.symbol}] {issue}")

        report.root_cause_hypotheses = all_issues

        # 基于聚合结果生成建议
        agg = report.aggregate
        recommendations = []

        # 优先级1: EV 为负
        if agg.get("ev_mean_pct", 0) < 0:
            recommendations.append({
                "priority": "CRITICAL",
                "action": "Fix EV: 重构 fitness 函数为 EV 导向",
                "detail": (
                    "当前 EV 为负, 策略无正期望。根据 ERR-047 教训: "
                    "EV = 胜率 × 盈亏比 - (1-胜率) × 1.0, "
                    "应在保持胜率 ≥ 55% 前提下提升盈亏比, 而非单纯优化盈亏比。"
                ),
                "fix_id": "Fix 33",
            })

        # 优先级2: 胜率过低
        if agg.get("win_rate_mean", 0) < 50:
            recommendations.append({
                "priority": "HIGH",
                "action": "Fix 胜率: 入场信号质量改进",
                "detail": (
                    "当前跨币种胜率 < 50%, 低于随机水平。"
                    "需检查: 1)入场是否在趋势反转点 2)趋势确认是否充分 "
                    "3)信号是否被噪声主导。"
                ),
                "fix_id": "Fix 34",
            })

        # 优先级3: 特定币种亏损
        losing_symbols = [
            s for s, r in report.symbol_reports.items()
            if r.ev_per_trade < 0 and r.trades
        ]
        if losing_symbols:
            recommendations.append({
                "priority": "HIGH",
                "action": f"Fix 亏损币种: {', '.join(losing_symbols)}",
                "detail": (
                    f"以下币种 EV 为负: {', '.join(losing_symbols)}。"
                    "需分析: 1)市场微观结构差异 2)波动率特征差异 "
                    "3)趋势持续性差异。考虑币种特定的参数调整。"
                ),
                "fix_id": "Fix 34",
            })

        # 优先级4: 模拟-实盘差异
        recommendations.append({
            "priority": "MEDIUM",
            "action": "建立模拟-实盘差异量化评估模型",
            "detail": (
                "分析点差、滑点、流动性、订单执行延迟等因素对策略表现的影响权重。"
                "在模拟环境中构建高保真市场仿真系统。"
            ),
            "fix_id": "Fix 35",
        })

        # 优先级5: 深度参数分析
        recommendations.append({
            "priority": "MEDIUM",
            "action": "执行 PCA + 随机森林 + SHAP 参数重要性分析",
            "detail": (
                "对最近6个月历史交易数据进行深度回溯分析, "
                "量化各参数对交易结果的影响权重。"
            ),
            "fix_id": "Fix 36",
        })

        report.actionable_recommendations = recommendations

        # 确定下一步 Fix 方向
        if recommendations:
            top = recommendations[0]
            report.next_fix_direction = f"{top['fix_id']}: {top['action']} (优先级: {top['priority']})"


# ============================================================================
# 报告输出
# ============================================================================
def save_report(report: ReviewReport, output_dir: str = None):
    """保存复盘报告 (JSON + Markdown) + 自动触发反馈闭环"""
    output_dir = output_dir or os.path.join(SCRIPT_DIR, "_review_reports")
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"auto_review_{ts}.json")
    md_path = os.path.join(output_dir, f"auto_review_{ts}.md")
    baseline_path = os.path.join(output_dir, "auto_review_baseline.json")

    # JSON 报告
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    # Markdown 报告
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(report))

    # 更新基线
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    # ===== 自动触发复盘-经验库联动闭环 (短板 #1 补齐) =====
    # 每次保存复盘报告后, 自动调用 ReviewFeedbackLoop 处理:
    #   1. 检测错误模式 → 自动入库到 1500-error-knowledge-base.md
    #   2. 生成参数调整建议 → 保存到 _param_patches/
    #   3. 追踪跨版本复发 → WARN (2次) / BLOCK (3+次)
    #   4. 生成后续任务
    try:
        from review_feedback_loop import ReviewFeedbackLoop
        feedback_loop = ReviewFeedbackLoop()
        actions = feedback_loop.process_review_report(json_path)
        print(f"\n{'='*60}")
        print("📋 复盘反馈闭环已自动触发:")
        print(actions.summary)
        print(f"{'='*60}")
        if actions.recurrence_alerts:
            print("\n🚨 复发警报:")
            for alert in actions.recurrence_alerts:
                print(f"  [{alert.severity}] {alert.message}")
    except Exception as e:
        print(f"⚠️ 反馈闭环触发失败 (不影响复盘报告): {e}")

    return json_path, md_path


def generate_markdown_report(report: ReviewReport) -> str:
    """生成 Markdown 格式复盘报告"""
    lines = []
    lines.append(f"# 🔍 自动化复盘报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {report.timestamp}  ")
    lines.append(f"**版本**: {report.version}  ")
    lines.append(f"**耗时**: {report.duration_seconds:.1f}s  ")
    lines.append(f"**测试币种**: {', '.join(report.symbols_tested)}  ")
    lines.append(f"")

    # KPI 检查
    lines.append(f"## 📊 KPI 达标检查")
    lines.append(f"")
    kpi = report.kpi_check
    if kpi:
        lines.append(f"| 指标 | 目标 | 实际 | 状态 |")
        lines.append(f"|------|------|------|------|")
        for name, check in kpi.items():
            if name == "summary":
                continue
            status = "✅" if check["passed"] else "❌"
            lines.append(f"| {name} | {check['target']} | {check['actual']} | {status} |")
        lines.append(f"")
        lines.append(f"**综合**: {kpi.get('summary', 'N/A')}")
        lines.append(f"")

    # 聚合统计
    lines.append(f"## 📈 聚合统计")
    lines.append(f"")
    agg = report.aggregate
    if agg:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        for k, v in agg.items():
            if k not in ("best_symbol", "worst_symbol"):
                lines.append(f"| {k} | {v} |")
        lines.append(f"")
        lines.append(f"- 🏆 最佳币种: **{agg.get('best_symbol', 'N/A')}**")
        lines.append(f"- 🔻 最差币种: **{agg.get('worst_symbol', 'N/A')}**")
        lines.append(f"")

    # 各币种详细
    lines.append(f"## 📋 各币种详细分析")
    lines.append(f"")
    for symbol, sr in report.symbol_reports.items():
        lines.append(f"### {symbol}")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 总交易数 | {len(sr.trades)} |")
        lines.append(f"| 多头/空头 | {sr.long_trades}/{sr.short_trades} |")
        lines.append(f"| 总盈亏 | {sr.total_pnl_pct:.2f}% |")
        lines.append(f"| 胜率 | {sr.win_rate:.1f}% |")
        lines.append(f"| 平均盈利 | {sr.avg_win:.3f}% |")
        lines.append(f"| 平均亏损 | {sr.avg_loss:.3f}% |")
        lines.append(f"| EV/笔 | {sr.ev_per_trade:.4f}% |")
        lines.append(f"| 最大回撤 | {sr.max_drawdown:.1f}% |")
        lines.append(f"| 盈亏比 | {sr.profit_factor:.2f} |")
        lines.append(f"| 夏普比率 | {sr.sharpe_ratio:.3f} |")
        lines.append(f"| TP 触发率 | {sr.tp_reachable_rate:.1f}% |")
        lines.append(f"")

        # 退出原因分布
        if sr.exit_reasons:
            lines.append(f"**退出原因分布**:")
            for reason, count in sr.exit_reasons.most_common():
                pct = count / max(len(sr.trades), 1) * 100
                lines.append(f"- {reason}: {count} ({pct:.1f}%)")
            lines.append(f"")

        # 深度分析（如果存在）
        depth = getattr(sr, '_depth_analysis', None)
        if depth:
            # 退出原因×PnL
            if depth.get("exit_avg_pnl"):
                lines.append(f"**退出原因 × 平均PnL**:")
                for reason, avg_pnl in sorted(depth["exit_avg_pnl"].items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {reason}: {avg_pnl:+.3f}%")
                lines.append(f"")

            # Regime分析
            if depth.get("regime_stats"):
                lines.append(f"**市场状态分析**:")
                lines.append(f"| Regime | 交易数 | 胜率 | 平均PnL |")
                lines.append(f"|--------|--------|------|---------|")
                for regime, stats in sorted(depth["regime_stats"].items()):
                    lines.append(f"| {regime} | {stats['count']} | {stats['win_rate']:.1f}% | {stats['avg_pnl']:+.3f}% |")
                lines.append(f"")

            # Hurst分析 [已移除-D2审计补遗 ERR-20260704-hurst-audit-gap]
            # if depth.get("hurst_stats"):
            #     lines.append(f"**Hurst状态分析**:")
            #     ...

            # 月度P&L
            if depth.get("monthly_pnl"):
                lines.append(f"**月度P&L**:")
                monthly = depth["monthly_pnl"]
                loss_months = sum(1 for v in monthly.values() if v < 0)
                lines.append(f"- 月数: {len(monthly)}, 亏损月: {loss_months} ({loss_months/max(len(monthly),1)*100:.1f}%)")
                lines.append(f"- 最大连续亏损: {depth.get('max_consecutive_losses', 'N/A')}")
                lines.append(f"")

            # 持仓时长
            if depth.get("avg_bars_held"):
                lines.append(f"**持仓时长分析**:")
                lines.append(f"- 平均持仓: {depth['avg_bars_held']:.1f} K线")
                lines.append(f"- 盈利单平均持仓: {depth.get('avg_bars_win', 0):.1f} K线")
                lines.append(f"- 亏损单平均持仓: {depth.get('avg_bars_loss', 0):.1f} K线")
                lines.append(f"")

        # 问题
        if sr.issues:
            lines.append(f"**⚠️ 发现问题**:")
            for issue in sr.issues:
                lines.append(f"- {issue}")
            lines.append(f"")

        # 优势
        if sr.strengths:
            lines.append(f"**✅ 优势**:")
            for s in sr.strengths:
                lines.append(f"- {s}")
            lines.append(f"")

    # 根因假设
    if report.root_cause_hypotheses:
        lines.append(f"## 🔬 根因假设")
        lines.append(f"")
        for h in report.root_cause_hypotheses:
            lines.append(f"- {h}")
        lines.append(f"")

    # 可操作建议
    if report.actionable_recommendations:
        lines.append(f"## 🎯 可操作建议")
        lines.append(f"")
        for rec in report.actionable_recommendations:
            lines.append(f"### [{rec['priority']}] {rec['action']}")
            lines.append(f"")
            lines.append(f"{rec['detail']}")
            lines.append(f"")
            lines.append(f"**Fix ID**: `{rec['fix_id']}`")
            lines.append(f"")

    # 下一步
    if report.next_fix_direction:
        lines.append(f"## 🚀 下一步 Fix 方向")
        lines.append(f"")
        lines.append(f"{report.next_fix_direction}")
        lines.append(f"")

    # 基线对比
    bc = report.baseline_comparison
    if bc and bc.get("status") != "无基线数据":
        lines.append(f"## 📉 基线对比")
        lines.append(f"")
        lines.append(f"基线时间: {bc.get('baseline_timestamp', 'N/A')}")
        lines.append(f"")
        if bc.get("regression_detected"):
            lines.append(f"⚠️ **检测到回归!**")
        lines.append(f"")
        for key, change in bc.get("changes", {}).items():
            direction_icon = {"improved": "📈", "degraded": "📉", "unchanged": "➡️"}.get(change["direction"], "")
            lines.append(f"- {direction_icon} **{key}**: {change['baseline']} → {change['current']} (Δ={change['delta']:+.4f})")
        lines.append(f"")

    return "\n".join(lines)


# ============================================================================
# 自检入口 (单元测试)
# ============================================================================

def _self_test() -> bool:
    """自检: 验证自动化复盘引擎核心功能可用

    测试覆盖:
      1. run_from_json: mock 混合胜负交易 → 验证 aggregate/symbol_reports 字段非空
      2. run_from_json: mock 全胜交易 → 验证胜率=100% / profit_factor 合理
    所有 mock 交易写入临时 JSON 文件, 不污染真实复盘报告目录。
    """
    import tempfile

    def _build_mock_trade(symbol: str, direction: str, pnl_pct: float,
                          exit_reason: str, bars_held: int = 5) -> dict:
        """构造单笔 mock 交易 (pnl_pct 为小数形式, 如 0.02=2%)"""
        entry = 100.0
        exit_price = entry * (1 + pnl_pct) if direction == "long" else entry * (1 - pnl_pct)
        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "bars_held": bars_held,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
        }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # ===== 测试1: 混合胜负交易 (多币种) =====
            trades_mixed = []
            # BTC: 6胜4负
            for i in range(6):
                trades_mixed.append(_build_mock_trade("BTC-USDT", "long", 0.025, "take_profit"))
            for i in range(4):
                trades_mixed.append(_build_mock_trade("BTC-USDT", "long", -0.015, "stop_loss"))
            # ETH: 3胜7负 (负 EV)
            for i in range(3):
                trades_mixed.append(_build_mock_trade("ETH-USDT", "long", 0.02, "take_profit"))
            for i in range(7):
                trades_mixed.append(_build_mock_trade("ETH-USDT", "long", -0.012, "stop_loss"))

            mixed_path = os.path.join(tmpdir, "v_selftest_mixed_trades_detail.json")
            with open(mixed_path, "w", encoding="utf-8") as f:
                json.dump(trades_mixed, f)

            engine = AutoReviewEngine(symbols=["BTC-USDT", "ETH-USDT"], initial_capital=10000.0)
            report = engine.run_from_json([mixed_path], version_label="v_selftest_mixed")

            assert len(report.symbols_tested) == 2, \
                f"symbols_tested 数量错误: {report.symbols_tested}"
            assert "BTC-USDT" in report.symbol_reports, "BTC-USDT 报告缺失"
            assert "ETH-USDT" in report.symbol_reports, "ETH-USDT 报告缺失"
            # aggregate 应被填充
            assert isinstance(report.aggregate, dict) and len(report.aggregate) > 0, \
                "aggregate 为空"
            # KPI 检查应执行
            assert isinstance(report.kpi_check, dict), "kpi_check 未生成"
            # BTC 胜率应约 60% (6/10)
            btc_sr = report.symbol_reports["BTC-USDT"]
            assert 55.0 <= btc_sr.win_rate <= 65.0, \
                f"BTC 胜率异常: {btc_sr.win_rate} (期望约60%)"
            assert btc_sr.total_pnl_pct > 0, f"BTC 总 PnL 应为正: {btc_sr.total_pnl_pct}"
            # ETH 胜率应约 30% (3/10), 负 EV
            eth_sr = report.symbol_reports["ETH-USDT"]
            assert 25.0 <= eth_sr.win_rate <= 35.0, \
                f"ETH 胜率异常: {eth_sr.win_rate} (期望约30%)"
            assert eth_sr.ev_per_trade < 0, f"ETH EV 应为负: {eth_sr.ev_per_trade}"

            # ===== 测试2: 全胜交易 → 胜率=100% =====
            trades_all_win = [
                _build_mock_trade("SOL-USDT", "long", 0.03, "take_profit")
                for _ in range(8)
            ]
            allwin_path = os.path.join(tmpdir, "v_selftest_allwin_trades_detail.json")
            with open(allwin_path, "w", encoding="utf-8") as f:
                json.dump(trades_all_win, f)

            engine2 = AutoReviewEngine(symbols=["SOL-USDT"], initial_capital=10000.0)
            report2 = engine2.run_from_json([allwin_path], version_label="v_selftest_allwin")

            assert len(report2.symbols_tested) == 1, \
                f"symbols_tested 数量错误: {report2.symbols_tested}"
            sol_sr = report2.symbol_reports["SOL-USDT"]
            assert sol_sr.win_rate == 100.0, \
                f"全胜交易胜率应为100%: {sol_sr.win_rate}"
            assert sol_sr.total_pnl_pct > 0, f"全胜总 PnL 应为正: {sol_sr.total_pnl_pct}"
            assert sol_sr.avg_loss == 0, f"无亏损交易 avg_loss 应为0: {sol_sr.avg_loss}"
            # 退出原因应全部为 take_profit
            assert sol_sr.exit_reasons.get("take_profit", 0) == 8, \
                f"take_profit 计数错误: {sol_sr.exit_reasons}"

            # ===== 测试3: 空交易列表 → 优雅降级 (不崩溃) =====
            empty_path = os.path.join(tmpdir, "v_selftest_empty_trades_detail.json")
            with open(empty_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            engine3 = AutoReviewEngine(symbols=["XRP-USDT"], initial_capital=10000.0)
            report3 = engine3.run_from_json([empty_path], version_label="v_selftest_empty")
            # 空交易应优雅降级, aggregate 含 error 标记
            assert "error" in report3.aggregate, \
                f"空交易应优雅降级 (aggregate 含 error): {report3.aggregate}"

            logging.info("AutoReviewEngine 自检全部通过 ✓")
            return True

    except Exception as e:
        logging.error(f"自检失败: {e}")
        return False


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="自动化复盘系统 — 支持EvolutionLoop模式和直接JSON模式")
    parser.add_argument("--symbols", type=str, default=None,
                        help="逗号分隔的币种列表 (默认: BTC,ETH,SOL,BNB,DOGE)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="运行轮数 (默认: 3)")
    parser.add_argument("--compare", type=str, default=None,
                        help="基线 JSON 文件路径 (用于对比)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认: _review_reports)")
    # 直接JSON模式
    parser.add_argument("--from-json", type=str, default=None,
                        help="从交易明细JSON文件直接生成复盘报告（逗号分隔多个文件）")
    parser.add_argument("--all-versions", action="store_true",
                        help="批量复盘所有版本 (_v*_trades_detail.json)")
    parser.add_argument("--compare-versions", type=str, default=None,
                        help="对比两个版本 (格式: v575,v578)")
    parser.add_argument("--self-test", action="store_true",
                        help="运行模块自检 (_self_test), 不执行 CLI 业务逻辑")
    args = parser.parse_args()

    if args.self_test:
        ok = _self_test()
        print(f"_self_test: {ok}")
        return

    engine = AutoReviewEngine(
        symbols=DEFAULT_SYMBOLS,
        rounds=args.rounds,
        baseline_path=args.compare,
    )

    # 模式1: 直接JSON模式
    if args.from_json or args.all_versions or args.compare_versions:
        print("=" * 70)
        print(f"🔍 自动化复盘系统 v{REVIEW_VERSION} [直接JSON模式]")
        print("=" * 70)

        if args.compare_versions:
            # 版本对比模式
            versions = args.compare_versions.split(",")
            if len(versions) != 2:
                print("❌ --compare-versions 需要两个版本号，如: v575,v578")
                return
            print(f"对比版本: {versions[0]} vs {versions[1]}")
            comparison = engine.compare_versions(versions[0].strip(), versions[1].strip())
            if "error" in comparison:
                print(f"❌ {comparison['error']}")
                return
            # 保存对比报告
            output_dir = args.output or os.path.join(SCRIPT_DIR, "_review_reports")
            os.makedirs(output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            comp_path = os.path.join(output_dir, f"version_comparison_{ts}.json")
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            print(f"✅ 对比报告: {comp_path}")
            print(json.dumps(comparison, ensure_ascii=False, indent=2))

        elif args.all_versions:
            # 批量复盘所有版本
            print("批量复盘所有版本...")
            all_reports = engine.run_all_versions()
            print(f"\n✅ 完成 {len(all_reports)} 个版本复盘")

        elif args.from_json:
            # 单文件/多文件JSON模式
            json_files = [f.strip() for f in args.from_json.split(",")]
            # 如果是相对路径，补全为绝对路径
            resolved_files = []
            for jf in json_files:
                if not os.path.isabs(jf):
                    jf = os.path.join(SCRIPT_DIR, jf)
                resolved_files.append(jf)

            print(f"JSON文件: {resolved_files}")
            report = engine.run_from_json(resolved_files)
            json_path, md_path = save_report(report, args.output)

            print(f"\n{'='*70}")
            print(f"✅ 复盘完成!")
            print(f"   JSON: {json_path}")
            print(f"   MD:   {md_path}")
            print(f"   KPI:  {report.kpi_check.get('summary', 'N/A')}")
            print(f"   下一步: {report.next_fix_direction}")
            print(f"{'='*70}")

            # 打印 Markdown 报告
            print(f"\n{generate_markdown_report(report)}")

        return

    # 模式2: EvolutionLoop模式（原逻辑）
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS

    print("=" * 70)
    print(f"🔍 自动化复盘系统 v{REVIEW_VERSION} [EvolutionLoop模式]")
    print("=" * 70)
    print(f"币种: {symbols}")
    print(f"轮数: {args.rounds}")
    print(f"基线: {args.compare or '无'}")
    print("=" * 70)

    engine.symbols = symbols
    report = engine.run()

    json_path, md_path = save_report(report, args.output)

    print(f"\n{'='*70}")
    print(f"✅ 复盘完成!")
    print(f"   JSON: {json_path}")
    print(f"   MD:   {md_path}")
    print(f"   KPI:  {report.kpi_check.get('summary', 'N/A')}")
    print(f"   下一步: {report.next_fix_direction}")
    print(f"{'='*70}")

    # 打印 Markdown 报告摘要
    print(f"\n{generate_markdown_report(report)}")


if __name__ == "__main__":
    main()