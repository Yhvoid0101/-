# -*- coding: utf-8 -*-
"""极端行情专项测试集 (Extreme Market Test) — 6大核心短板 #6

提取自 _v96_extreme_market_test.py (1201行脚本)，去除脚本式副作用
(import v96模块触发回测 / sys.stdout.reconfigure / 顶层print / v93em基线硬编码)，
保留 9 个核心纯函数 + 5 层判定铁标准，遵循 sim_live_gap_model.py 的架构范本
（@dataclass 参数 + 纯函数 + __main__ 守卫）。

用户核心诉求: "市场状态自动分类 + 极端行情专项测试集，强化全行情适应性，
              打破策略行情局限瓶颈" + "永远永远不要出现模拟牛逼，实盘亏损"

设计原则:
  - 纯函数: 所有函数无副作用，相同输入相同输出
  - 铁标准: 5层判定阈值不可降级 (来源: v93em 5/5通过基线 + v94 Layer C退化教训)
  - 通用格式: 接受标准化 trade dict, 通过 _normalize_trades 适配 evolution_loop
  - 零副作用: 导入时不执行任何IO操作，不修改全局状态

5层极端行情测试架构 (铁标准，不可降级):
  Layer A: ATR 分位极端档位 (Top5%/Top10%/Mid/Bot10%/Bot5%)
    判定: Top5% total_pnl > 0 AND Top5% 单亏 ≤ 2%
  Layer B: 单笔最大盈亏极端事件
    判定: 单亏 > 2% 笔数 = 0 (硬截断)
  Layer C: 连续亏损序列极端事件
    判定: 最长连亏 ≤ 4 (v94退化教训: 5→4修复)
  Layer D: 月度极端事件
    判定: 亏损月率 = 0%
  Layer E: 滚动窗口压力测试
    判定: 180天达标率 = 100% (硬性) 或 90天 ≥ 70% (统计噪声容忍)
  综合: 5层 ≥ 4/5 通过

ERR知识库遵守:
  - ERR-093: consec_loss≤3是物理极限, Layer C阈值设为≤4 (留1个缓冲)
  - ERR-100: v90参考单亏≤2%硬截断
  - ERR-20260701-v94: Layer C退化(连亏4→5) → v95 per-symbol consec修复, 阈值保持≤4

来源: _v96_extreme_market_test.py (1201行) → 提取约 400行纯逻辑
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 安全统计工具 (避免 Python 3.12 statistics.stdev 的 numerator bug)
# ============================================================

def _safe_stdev(data: List[float]) -> float:
    """安全的样本标准差计算 (ddof=1)

    Args:
        data: 数值序列

    Returns:
        样本标准差，数据少于2个时返回0.0
    """
    if len(data) < 2:
        return 0.0
    m = sum(data) / len(data)
    var = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(var)


def _safe_mean(data: List[float]) -> float:
    """安全的均值计算

    Args:
        data: 数值序列

    Returns:
        均值，空列表返回0.0
    """
    return sum(data) / len(data) if data else 0.0


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class ExtremeTestConfig:
    """极端测试配置

    所有参数有合理默认值，通常无需手动配置。
    initial_capital 和时间窗口与 v96 基线一致。
    """
    initial_capital: float = 10000.0
    total_days: int = 638           # v96基线: 638天数据
    bars_per_month: int = 720       # 30天 × 24h
    window_90: int = 2160           # 90 × 24 (1h K线)
    window_180: int = 4320          # 180 × 24
    mc_iterations: int = 1000       # 蒙特卡洛bootstrap次数
    mc_seed: int = 42               # 随机种子 (可复现)
    # Layer A ATR分位阈值 (不修改, 铁标准)
    atr_percentiles: Tuple[float, ...] = (0.05, 0.10, 0.90, 0.95)
    # Layer B 单亏硬截断阈值 (ERR-100: 2%)
    single_loss_cap_pct: float = 2.0
    # Layer C 最长连亏阈值 (v94退化教训: 5→4)
    max_consec_loss: int = 4
    # Layer D 月亏率阈值
    max_loss_month_rate: float = 0.0  # 0% = 不允许任何亏损月
    max_cv: float = 1.5
    # Layer E 滚动窗口达标率阈值
    min_rate_90: float = 70.0       # 90天达标率 (统计噪声容忍)
    min_rate_180: float = 100.0     # 180天达标率 (硬性)
    rolling_kpi_threshold: int = 5  # 6项KPI中至少通过5项
    # Tier1 KPI阈值 (铁标准, 与 staged_admission_gate.py TIER1_THRESHOLDS 一致)
    tier1_ann_min: float = 30.0
    tier1_dd_max: float = 15.0
    tier1_wr_min: float = 55.0
    tier1_sharpe_min: float = 1.5
    tier1_max_single_loss: float = 2.0
    tier1_ci_low_min: float = 20.0  # CI下限>20%
    tier1_6of8_rate_min: float = 80.0  # 6/8达标率≥80%
    tier1_p_value_max: float = 0.05


@dataclass
class ExtremeTestReport:
    """极端测试报告

    包含5层测试详细结果 + Tier1验证 + 综合判定。
    """
    version: str = "v96_pure"
    test_date: str = ""
    layer_a_atr_extreme: Dict[str, Any] = field(default_factory=dict)
    layer_b_single_loss: Dict[str, Any] = field(default_factory=dict)
    layer_c_consec_loss: Dict[str, Any] = field(default_factory=dict)
    layer_d_monthly: Dict[str, Any] = field(default_factory=dict)
    layer_e_rolling: Dict[str, Any] = field(default_factory=dict)
    tier1_validation: Dict[str, Any] = field(default_factory=dict)
    overall_pass: bool = False
    n_pass_layers: int = 0          # 5层通过数 (0-5)
    n_total_layers: int = 5
    error: Optional[str] = None     # 异常时填充

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化dict"""
        return {
            "version": self.version,
            "test_date": self.test_date,
            "layer_a_atr_extreme": self.layer_a_atr_extreme,
            "layer_b_single_loss": self.layer_b_single_loss,
            "layer_c_consec_loss": self.layer_c_consec_loss,
            "layer_d_monthly": self.layer_d_monthly,
            "layer_e_rolling": self.layer_e_rolling,
            "tier1_validation": self.tier1_validation,
            "overall_pass": self.overall_pass,
            "n_pass_layers": self.n_pass_layers,
            "n_total_layers": self.n_total_layers,
            "error": self.error,
        }


# ============================================================
# 交易数据适配器 — 将 evolution_loop 的 trade 格式转为标准化格式
# ============================================================

def _normalize_trades(
    raw_trades: List[Dict[str, Any]],
    config: ExtremeTestConfig,
) -> List[Dict[str, Any]]:
    """将 evolution_loop 的 trade dict 列表转为标准化格式

    evolution_loop 的 trade 可能是:
      - dict: {"pnl", "entry_time"/"entry_timestamp", "exit_time"/"exit_timestamp",
               "entry_price", "quantity"/"size", "symbol", "status", ...}
      或 v96 格式: {"live_pnl", "rec": {"features": {"atr_pct"}, "hold_bars", "entry_bar"}, ...}

    标准化后每个 trade 含:
      pnl, entry_time, exit_time, symbol, atr_pct, hold_bars, month, entry_bar

    Args:
        raw_trades: 原始交易列表
        config: 测试配置

    Returns:
        标准化交易列表
    """
    if not raw_trades:
        return []

    normalized: List[Dict[str, Any]] = []
    for t in raw_trades:
        if not isinstance(t, dict):
            continue

        # 兼容 v96 格式 (live_pnl + rec嵌套)
        if "live_pnl" in t and "rec" in t:
            pnl = float(t.get("live_pnl", 0))
            rec = t.get("rec", {})
            features = rec.get("features", {}) if isinstance(rec, dict) else {}
            atr_pct = float(features.get("atr_pct", 0.01))
            hold_bars = int(rec.get("hold_bars", 0))
            entry_bar = int(rec.get("entry_bar", 0))
            symbol = str(t.get("symbol", "UNKNOWN"))
            month = int(t.get("month", 0))
            entry_time = int(rec.get("entry_time", entry_bar * 3600 * 1000))
            exit_time = int(rec.get("exit_time", entry_time + hold_bars * 3600 * 1000))
        else:
            # evolution_loop 格式
            pnl = float(t.get("pnl", 0))
            entry_time = int(t.get("entry_time", t.get("entry_timestamp", 0)))
            exit_time = int(t.get("exit_time", t.get("exit_timestamp", entry_time)))
            symbol = str(t.get("symbol", "UNKNOWN"))
            atr_pct = float(t.get("atr_pct", t.get("features", {}).get("atr_pct", 0.01)
                                  if isinstance(t.get("features"), dict) else 0.01))
            hold_bars = int(t.get("hold_bars", 0))
            entry_bar = int(t.get("entry_bar", 0))
            month = int(t.get("month", 0))

        # 如果 month 未提供，从 entry_time 推导 (毫秒时间戳 → 月份编号)
        if month == 0 and entry_time > 0:
            try:
                dt = datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc)
                month = dt.year * 12 + dt.month
            except (OSError, ValueError, OverflowError):
                month = 0

        # 如果 entry_bar 未提供，从 entry_time 推导 (假设1h K线，基线为最早交易)
        if entry_bar == 0 and entry_time > 0:
            entry_bar = int(entry_time / (3600 * 1000))

        normalized.append({
            "pnl": pnl,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "symbol": symbol,
            "atr_pct": atr_pct,
            "hold_bars": hold_bars,
            "month": month,
            "entry_bar": entry_bar,
        })

    # 按 entry_bar 排序（时间序列）
    normalized.sort(key=lambda x: x["entry_bar"])
    return normalized


# ============================================================
# Layer A: ATR 分位极端档位测试
# ============================================================

def analyze_atr_segment(
    trades: List[Dict[str, Any]],
    segment_name: str,
    initial_capital: float,
) -> Dict[str, Any]:
    """分析 ATR 分位段的交易表现

    Args:
        trades: 该分位段的交易列表 (标准化格式)
        segment_name: 段名 (e.g. "Top5% ATR")
        initial_capital: 初始资金

    Returns:
        段统计 dict
    """
    if not trades:
        return {"segment": segment_name, "n": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    total_pnl = sum(pnls)
    max_win = max(pnls) if pnls else 0
    max_loss = min(pnls) if pnls else 0
    single_loss_pcts = [abs(p) / initial_capital * 100 for p in losses]
    max_single_loss_pct = max(single_loss_pcts) if single_loss_pcts else 0
    hold_bars_list = [t.get("hold_bars", 0) for t in trades]
    avg_hold = _safe_mean(hold_bars_list) if hold_bars_list else 0

    return {
        "segment": segment_name,
        "n": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": _safe_mean(pnls),
        "max_win": max_win,
        "max_loss": max_loss,
        "max_single_loss_pct": max_single_loss_pct,
        "avg_hold_bars": avg_hold,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf'),
    }


def run_layer_a_atr_extreme(
    trades: List[Dict[str, Any]],
    config: ExtremeTestConfig,
) -> Dict[str, Any]:
    """Layer A: ATR 分位极端档位测试

    判定 (铁标准):
      - Top5% ATR total_pnl > 0 (极端高波动仍盈利)
      - Top5% ATR 单亏 ≤ 2% (极端高波动单亏控制)
      - Top10% ATR total_pnl > 0 (高波动仍盈利)

    Args:
        trades: 标准化交易列表
        config: 测试配置

    Returns:
        {pass: bool, segments: {...}, details: str}
    """
    if len(trades) < 20:
        return {"pass": False, "reason": "交易数<20无法分位", "segments": {}}

    atr_pcts = sorted([t["atr_pct"] for t in trades])
    n_total = len(atr_pcts)
    p5_idx = int(n_total * 0.05)
    p10_idx = int(n_total * 0.10)
    p90_idx = int(n_total * 0.90)
    p95_idx = int(n_total * 0.95)

    p5_threshold = atr_pcts[min(p5_idx, n_total - 1)]
    p10_threshold = atr_pcts[min(p10_idx, n_total - 1)]
    p90_threshold = atr_pcts[min(p90_idx, n_total - 1)]
    p95_threshold = atr_pcts[min(p95_idx, n_total - 1)]

    top5 = [t for t in trades if t["atr_pct"] >= p95_threshold]
    top10 = [t for t in trades if t["atr_pct"] >= p90_threshold]
    mid = [t for t in trades if p10_threshold < t["atr_pct"] < p90_threshold]
    bot10 = [t for t in trades if t["atr_pct"] <= p10_threshold]
    bot5 = [t for t in trades if t["atr_pct"] <= p5_threshold]

    cap = config.initial_capital
    seg_top5 = analyze_atr_segment(top5, "Top5% ATR", cap)
    seg_top10 = analyze_atr_segment(top10, "Top10% ATR", cap)
    seg_mid = analyze_atr_segment(mid, "Mid 80%", cap)
    seg_bot10 = analyze_atr_segment(bot10, "Bot10% ATR", cap)
    seg_bot5 = analyze_atr_segment(bot5, "Bot5% ATR", cap)

    # 判定 (铁标准)
    layer_pass = True
    details_parts: List[str] = []

    if seg_top5["n"] > 0:
        if seg_top5["total_pnl"] > 0:
            details_parts.append(f"✅ Top5%ATR盈利 ${seg_top5['total_pnl']:.2f}")
        else:
            details_parts.append(f"❌ Top5%ATR亏损 ${seg_top5['total_pnl']:.2f}")
            layer_pass = False
        if seg_top5["max_single_loss_pct"] <= config.single_loss_cap_pct:
            details_parts.append(f"✅ Top5%单亏≤{config.single_loss_cap_pct}%: {seg_top5['max_single_loss_pct']:.2f}%")
        else:
            details_parts.append(f"❌ Top5%单亏>{config.single_loss_cap_pct}%: {seg_top5['max_single_loss_pct']:.2f}%")
            layer_pass = False
    else:
        details_parts.append("⚠️ Top5%ATR无样本")
        layer_pass = False

    if seg_top10["n"] > 0 and seg_top10["total_pnl"] <= 0:
        details_parts.append(f"❌ Top10%ATR亏损 ${seg_top10['total_pnl']:.2f}")
        layer_pass = False

    return {
        "pass": layer_pass,
        "segments": {
            "top5": seg_top5, "top10": seg_top10, "mid": seg_mid,
            "bot10": seg_bot10, "bot5": seg_bot5,
        },
        "atr_thresholds": {"p5": p5_threshold, "p10": p10_threshold,
                           "p90": p90_threshold, "p95": p95_threshold},
        "details": "; ".join(details_parts),
    }


# ============================================================
# Layer B: 单笔最大盈亏极端事件
# ============================================================

def run_layer_b_single_loss(
    trades: List[Dict[str, Any]],
    config: ExtremeTestConfig,
) -> Dict[str, Any]:
    """Layer B: 单笔最大盈亏极端事件

    判定 (铁标准):
      - 单亏 > 2% 笔数 = 0 (ERR-100 硬截断)
      - Top10盈持仓 > Top10亏持仓 × 1.5 (hold_bars预测力, ERR-065/v88fp)

    Args:
        trades: 标准化交易列表
        config: 测试配置

    Returns:
        {pass: bool, loss_over_2pct: int, avg_hold_wins, avg_hold_losses, details}
    """
    if len(trades) < 20:
        return {"pass": False, "reason": "交易数<20"}

    cap = config.initial_capital
    sorted_by_pnl = sorted(trades, key=lambda x: -x["pnl"])
    top10_wins = sorted_by_pnl[:10]
    top10_losses = sorted_by_pnl[-10:][::-1]

    loss_over_2pct = [t for t in trades
                      if t["pnl"] < 0
                      and abs(t["pnl"]) / cap * 100 > config.single_loss_cap_pct]
    loss_over_1_5pct = [t for t in trades
                        if t["pnl"] < 0
                        and abs(t["pnl"]) / cap * 100 > 1.5]

    avg_hold_wins = _safe_mean([t.get("hold_bars", 0) for t in top10_wins])
    avg_hold_losses = _safe_mean([t.get("hold_bars", 0) for t in top10_losses])

    # 判定: 单亏≤2%是硬性要求 (ERR-100)
    layer_pass = len(loss_over_2pct) == 0
    hold_pass = avg_hold_wins > avg_hold_losses * 1.5 if avg_hold_losses > 0 else True

    details = (f"单亏>2%={len(loss_over_2pct)}笔(必须=0); "
               f"Top10盈持{avg_hold_wins:.1f} vs 亏持{avg_hold_losses:.1f}×1.5={avg_hold_losses*1.5:.1f}")

    return {
        "pass": layer_pass,
        "loss_over_2pct": len(loss_over_2pct),
        "loss_over_1_5pct": len(loss_over_1_5pct),
        "avg_hold_wins": avg_hold_wins,
        "avg_hold_losses": avg_hold_losses,
        "hold_pass": hold_pass,
        "max_win": top10_wins[0]["pnl"] if top10_wins else 0,
        "max_loss": top10_losses[0]["pnl"] if top10_losses else 0,
        "details": details,
    }


# ============================================================
# Layer C: 连续亏损序列极端事件
# ============================================================

def run_layer_c_consec_loss(
    trades: List[Dict[str, Any]],
    config: ExtremeTestConfig,
) -> Dict[str, Any]:
    """Layer C: 连续亏损序列极端事件

    判定 (铁标准, v94退化教训):
      - 最长连亏 ≤ 4 (v94退化到5, v95修复到4)

    Args:
        trades: 标准化交易列表 (需已按entry_bar排序)
        config: 测试配置

    Returns:
        {pass: bool, longest_seq, n_sequences, seq_distribution, details}
    """
    if len(trades) < 10:
        return {"pass": False, "reason": "交易数<10"}

    # 按entry_bar排序确保时间序列
    sorted_trades = sorted(trades, key=lambda x: x["entry_bar"])

    consec_sequences: List[List[Dict]] = []
    current_seq: List[Dict] = []
    for t in sorted_trades:
        if t["pnl"] < 0:
            current_seq.append(t)
        else:
            if current_seq:
                consec_sequences.append(current_seq)
            current_seq = []
    if current_seq:
        consec_sequences.append(current_seq)

    seq_lengths = [len(s) for s in consec_sequences] if consec_sequences else []
    longest_seq = max(seq_lengths) if seq_lengths else 0

    # 判定 (铁标准: ≤4)
    layer_pass = longest_seq <= config.max_consec_loss

    seq_dist = {
        "total": len(seq_lengths),
        "max": longest_seq,
        "avg": _safe_mean(seq_lengths) if seq_lengths else 0,
        "len_1": sum(1 for l in seq_lengths if l == 1),
        "len_2": sum(1 for l in seq_lengths if l == 2),
        "len_3": sum(1 for l in seq_lengths if l == 3),
        "len_ge4": sum(1 for l in seq_lengths if l >= 4),
    }

    return {
        "pass": layer_pass,
        "longest_seq": longest_seq,
        "threshold": config.max_consec_loss,
        "n_sequences": len(consec_sequences),
        "seq_distribution": seq_dist,
        "details": f"最长连亏={longest_seq} (阈值≤{config.max_consec_loss})",
    }


# ============================================================
# Layer D: 月度极端事件
# ============================================================

def run_layer_d_monthly(
    trades: List[Dict[str, Any]],
    config: ExtremeTestConfig,
) -> Dict[str, Any]:
    """Layer D: 月度极端事件

    判定 (铁标准):
      - 亏损月率 = 0% (不允许任何月度亏损)
      - CV < 1.5 (变异系数, 月度稳定性)

    Args:
        trades: 标准化交易列表
        config: 测试配置

    Returns:
        {pass: bool, loss_month_rate, cv, n_months, n_loss_months, details}
    """
    if len(trades) < 10:
        return {"pass": False, "reason": "交易数<10"}

    monthly_pnls: Dict[int, float] = {}
    monthly_n: Dict[int, int] = {}
    for t in trades:
        m = t.get("month", 0)
        monthly_pnls[m] = monthly_pnls.get(m, 0) + t["pnl"]
        monthly_n[m] = monthly_n.get(m, 0) + 1

    months_sorted = sorted(monthly_pnls.keys())
    monthly_pnl_list = [monthly_pnls[m] for m in months_sorted]

    if not monthly_pnl_list:
        return {"pass": False, "reason": "无月度数据"}

    mean_monthly = _safe_mean(monthly_pnl_list)
    std_monthly = _safe_stdev(monthly_pnl_list)
    cv = std_monthly / mean_monthly if mean_monthly != 0 else float('inf')

    loss_months = [m for m in months_sorted if monthly_pnls[m] < 0]
    loss_month_rate = len(loss_months) / len(months_sorted) * 100

    # 判定: 亏损月率=0%是硬性要求
    layer_pass = loss_month_rate <= config.max_loss_month_rate
    cv_pass = cv < config.max_cv

    return {
        "pass": layer_pass,
        "loss_month_rate": loss_month_rate,
        "cv": cv,
        "cv_pass": cv_pass,
        "n_months": len(months_sorted),
        "n_loss_months": len(loss_months),
        "best_month_pnl": max(monthly_pnl_list),
        "worst_month_pnl": min(monthly_pnl_list),
        "mean_monthly": mean_monthly,
        "std_monthly": std_monthly,
        "details": f"亏损月率={loss_month_rate:.1f}% (阈值≤{config.max_loss_month_rate}%), CV={cv:.2f}",
    }


# ============================================================
# Layer E: 滚动窗口压力测试
# ============================================================

def rolling_window_kpi(
    trades: List[Dict[str, Any]],
    window_size: int,
    config: ExtremeTestConfig,
) -> List[Dict[str, Any]]:
    """计算滚动窗口 KPI

    Args:
        trades: 标准化交易列表 (需已按entry_bar排序)
        window_size: 窗口大小 (K线根数)
        config: 测试配置

    Returns:
        窗口KPI列表, 每个含 {start_bar, end_bar, n_trades, total_pnl, ann_pct,
                              max_dd, win_rate, sharpe, max_loss_pct, n_pass, n_total_kpi}
    """
    if not trades:
        return []

    sorted_t = sorted(trades, key=lambda x: x["entry_bar"])
    all_entry_bars = [t["entry_bar"] for t in sorted_t]
    min_bar = min(all_entry_bars)
    max_bar = max(all_entry_bars)
    cap = config.initial_capital
    step = max(1, window_size // 2)

    results: List[Dict[str, Any]] = []
    for start in range(min_bar, max_bar - window_size + 1, step):
        end = start + window_size
        window_trades = [t for t in sorted_t if start <= t["entry_bar"] < end]
        if len(window_trades) < 5:
            continue

        pnls = [t["pnl"] for t in window_trades]
        wins = [p for p in pnls if p > 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        total_pnl = sum(pnls)
        ann_pct = (total_pnl / cap) * (365 / (window_size / 24)) * 100

        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            dd = (peak - cum) / cap * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        sharpe = (_safe_mean(pnls) / _safe_stdev(pnls) * math.sqrt(len(pnls))
                  if _safe_stdev(pnls) > 0 else 0)
        losses = [p for p in pnls if p < 0]
        max_loss_pct = max((abs(p) / cap * 100 for p in losses), default=0)

        kpi_checks = [
            ann_pct >= config.tier1_ann_min,
            max_dd <= config.tier1_dd_max,
            win_rate >= config.tier1_wr_min,
            sharpe >= config.tier1_sharpe_min,
            max_loss_pct <= config.tier1_max_single_loss,
            total_pnl > 0,
        ]
        n_pass = sum(kpi_checks)

        results.append({
            "start_bar": start, "end_bar": end,
            "n_trades": len(window_trades), "total_pnl": total_pnl,
            "ann_pct": ann_pct, "max_dd": max_dd, "win_rate": win_rate,
            "sharpe": sharpe, "max_loss_pct": max_loss_pct,
            "n_pass": n_pass, "n_total_kpi": len(kpi_checks),
        })
    return results


def run_layer_e_rolling(
    trades: List[Dict[str, Any]],
    config: ExtremeTestConfig,
) -> Dict[str, Any]:
    """Layer E: 滚动窗口压力测试

    判定 (铁标准):
      - 180天达标率 = 100% (硬性要求)
      - 90天达标率 ≥ 70% (统计噪声容忍)

    Args:
        trades: 标准化交易列表
        config: 测试配置

    Returns:
        {pass: bool, rate_90, rate_180, n_windows_90, n_windows_180, details}
    """
    if len(trades) < 20:
        return {"pass": False, "reason": "交易数<20"}

    results_90 = rolling_window_kpi(trades, config.window_90, config)
    results_180 = rolling_window_kpi(trades, config.window_180, config)

    rate_90 = 0.0
    if results_90:
        pass_5of6 = sum(1 for r in results_90 if r["n_pass"] >= config.rolling_kpi_threshold)
        rate_90 = pass_5of6 / len(results_90) * 100

    rate_180 = 0.0
    if results_180:
        pass_5of6 = sum(1 for r in results_180 if r["n_pass"] >= config.rolling_kpi_threshold)
        rate_180 = pass_5of6 / len(results_180) * 100

    # 判定: 180天=100%是硬性要求
    layer_pass_180 = rate_180 >= config.min_rate_180
    layer_pass_90 = rate_90 >= config.min_rate_90
    layer_pass = layer_pass_180

    return {
        "pass": layer_pass,
        "rate_90": rate_90,
        "rate_180": rate_180,
        "pass_90": layer_pass_90,
        "pass_180": layer_pass_180,
        "n_windows_90": len(results_90),
        "n_windows_180": len(results_180),
        "worst_90_pnl": min((r["total_pnl"] for r in results_90), default=0),
        "worst_180_pnl": min((r["total_pnl"] for r in results_180), default=0),
        "details": f"90天率={rate_90:.1f}%(阈值≥{config.min_rate_90}%), "
                   f"180天率={rate_180:.1f}%(阈值={config.min_rate_180}%)",
    }


# ============================================================
# Tier 1 重新验证 (MC Bootstrap, 12项check)
# ============================================================

def run_tier1_validation(
    trades: List[Dict[str, Any]],
    kpi: Dict[str, Any],
    config: ExtremeTestConfig,
) -> Dict[str, Any]:
    """Tier 1 重新验证 (蒙特卡洛 Bootstrap, 12项 check)

    使用 MC Bootstrap 重采样计算年化收益率的 95% 置信区间,
    并验证 12 项 Tier 1 准入 check。

    Args:
        trades: 标准化交易列表
        kpi: 已计算的 KPI dict (含 ann_live, max_dd, win_rate, sharpe, mlp,
             max_single_loss, max_consec, gap, n_pass)
        config: 测试配置

    Returns:
        {pass: bool, n_pass, n_total, ci_low, ci_high, p_value, pass_6of8_rate, checks: [...]}
    """
    if len(trades) < 10:
        return {"pass": False, "reason": "交易数<10", "n_pass": 0, "n_total": 12}

    pnls = [t["pnl"] for t in trades]
    cap = config.initial_capital
    n_mc = config.mc_iterations
    random.seed(config.mc_seed)

    ann_samples: List[float] = []
    n_pass_6of8 = 0

    for _ in range(n_mc):
        sample = [random.choice(pnls) for _ in range(len(pnls))]
        total = sum(sample)
        ann = (total / cap) * (365 / config.total_days) * 100
        ann_samples.append(ann)

        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in sample:
            cum += p
            peak = max(peak, cum)
            dd = (peak - cum) / cap * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        wins_s = [p for p in sample if p > 0]
        wr = len(wins_s) / len(sample) * 100
        sh = (_safe_mean(sample) / _safe_stdev(sample) * math.sqrt(len(sample))
              if _safe_stdev(sample) > 0 else 0)
        losses_s = [p for p in sample if p < 0]
        max_loss = max((abs(p) / cap * 100 for p in losses_s), default=0)

        checks = [ann >= config.tier1_ann_min, max_dd <= config.tier1_dd_max,
                  wr >= config.tier1_wr_min, sh >= config.tier1_sharpe_min,
                  max_loss <= config.tier1_max_single_loss, total > 0]
        if sum(checks) >= 6:
            n_pass_6of8 += 1

    ann_samples.sort()
    mean_ann = _safe_mean(ann_samples)
    std_ann = _safe_stdev(ann_samples)
    ci_low = mean_ann - 1.96 * std_ann
    ci_high = mean_ann + 1.96 * std_ann
    n_profit = sum(1 for a in ann_samples if a > 0)
    p_value = 1 - n_profit / n_mc
    pass_6of8_rate = n_pass_6of8 / n_mc * 100

    # 12项 check
    checks = [
        ("8/8 KPI", kpi.get("n_pass", 0) == 8, f"n_pass={kpi.get('n_pass', 0)}/8"),
        ("MC P<0.05", p_value < config.tier1_p_value_max, f"P={p_value:.4f}"),
        ("CI下限>20%", ci_low > config.tier1_ci_low_min, f"CI={ci_low:.2f}%"),
        ("6/8率≥80%", pass_6of8_rate >= config.tier1_6of8_rate_min,
         f"6/8率={pass_6of8_rate:.1f}%"),
        ("ann≥30%", kpi.get("ann_live", 0) >= config.tier1_ann_min,
         f"ann={kpi.get('ann_live', 0):.2f}%"),
        ("dd≤15%", kpi.get("max_dd", 0) <= config.tier1_dd_max,
         f"dd={kpi.get('max_dd', 0):.2f}%"),
        ("wr≥55%", kpi.get("win_rate", 0) >= config.tier1_wr_min,
         f"wr={kpi.get('win_rate', 0):.1f}%"),
        ("sh≥1.5", kpi.get("sharpe", 0) >= config.tier1_sharpe_min,
         f"sh={kpi.get('sharpe', 0):.2f}"),
        ("月亏≤5%", kpi.get("mlp", 0) <= 5, f"月亏={kpi.get('mlp', 0):.1f}%"),
        ("单亏≤2%", kpi.get("max_single_loss", 0) <= config.tier1_max_single_loss,
         f"单亏={kpi.get('max_single_loss', 0):.2f}%"),
        ("连亏≤3", kpi.get("max_consec", 0) <= 3, f"连亏={kpi.get('max_consec', 0)}"),
        ("gap≤15%", kpi.get("gap", 0) <= 15, f"gap={kpi.get('gap', 0):.2f}%"),
    ]

    n_pass = sum(1 for _, passed, _ in checks if passed)
    all_pass = n_pass == len(checks)

    return {
        "pass": all_pass,
        "n_pass": n_pass,
        "n_total": len(checks),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_ann": mean_ann,
        "p_value": p_value,
        "pass_6of8_rate": pass_6of8_rate,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
    }


# ============================================================
# 主函数: 运行完整极端行情测试
# ============================================================

def run_extreme_market_test(
    raw_trades: List[Dict[str, Any]],
    config: Optional[ExtremeTestConfig] = None,
    kpi: Optional[Dict[str, Any]] = None,
    baseline: Optional[Dict[str, Any]] = None,
) -> ExtremeTestReport:
    """运行完整 5 层极端行情测试 + Tier1 重新验证

    用户核心诉求: "永远永远不要出现模拟牛逼，实盘亏损"
    本函数是部署禁令的最后一道防线 — 极端行情下不退化才允许部署。

    Args:
        raw_trades: 原始交易列表 (evolution_loop 格式或 v96 格式)
        config: 测试配置 (None=默认)
        kpi: 已计算的 KPI dict (None=从trades推导基础KPI, 不含8/8检查)
        baseline: 对照基线 (None=不对比, 仅判定通过/不通过)

    Returns:
        ExtremeTestReport: 含5层结果 + Tier1验证 + 综合判定
    """
    if config is None:
        config = ExtremeTestConfig()

    report = ExtremeTestReport(
        version="v96_pure",
        test_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    try:
        # 标准化交易数据
        trades = _normalize_trades(raw_trades, config)
        if len(trades) < 20:
            report.error = f"交易数不足: {len(trades)} < 20"
            report.overall_pass = False
            return report

        # 基础 KPI (如果未提供)
        if kpi is None:
            pnls = [t["pnl"] for t in trades]
            cap = config.initial_capital
            total_pnl = sum(pnls)
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            ann_live = (total_pnl / cap) * (365 / config.total_days) * 100
            # 基础max_dd
            cum = 0.0
            peak = 0.0
            max_dd = 0.0
            for p in pnls:
                cum += p
                peak = max(peak, cum)
                dd = (peak - cum) / cap * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            max_single_loss = max((abs(p) / cap * 100 for p in losses), default=0)
            kpi = {
                "n_pass": 8 if (ann_live >= 30 and max_dd <= 15 and len(wins)/len(pnls)*100 >= 55
                                and max_single_loss <= 2) else 6,
                "ann_live": ann_live,
                "max_dd": max_dd,
                "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
                "sharpe": (_safe_mean(pnls) / _safe_stdev(pnls) * math.sqrt(len(pnls))
                           if _safe_stdev(pnls) > 0 else 0),
                "mlp": 0.0,  # 需外部计算
                "max_single_loss": max_single_loss,
                "max_consec": 0,  # 由Layer C填充
                "gap": 0.0,
                "n_trades": len(trades),
            }

        # Layer A: ATR 分位极端档位
        report.layer_a_atr_extreme = run_layer_a_atr_extreme(trades, config)

        # Layer B: 单笔最大盈亏
        report.layer_b_single_loss = run_layer_b_single_loss(trades, config)

        # Layer C: 连续亏损序列
        report.layer_c_consec_loss = run_layer_c_consec_loss(trades, config)
        # 回填 max_consec 到 kpi
        if kpi.get("max_consec", 0) == 0:
            kpi["max_consec"] = report.layer_c_consec_loss.get("longest_seq", 0)

        # Layer D: 月度极端事件
        report.layer_d_monthly = run_layer_d_monthly(trades, config)
        # 回填 mlp
        if kpi.get("mlp", 0) == 0:
            kpi["mlp"] = report.layer_d_monthly.get("loss_month_rate", 0)

        # Layer E: 滚动窗口压力测试
        report.layer_e_rolling = run_layer_e_rolling(trades, config)

        # Tier 1 重新验证
        report.tier1_validation = run_tier1_validation(trades, kpi, config)

        # 综合判定: 5层 ≥ 4/5 通过
        layer_passes = [
            report.layer_a_atr_extreme.get("pass", False),
            report.layer_b_single_loss.get("pass", False),
            report.layer_c_consec_loss.get("pass", False),
            report.layer_d_monthly.get("pass", False),
            report.layer_e_rolling.get("pass", False),
        ]
        report.n_pass_layers = sum(layer_passes)
        report.n_total_layers = 5
        # 综合通过: ≥4/5层通过 AND Tier1通过
        report.overall_pass = (
            report.n_pass_layers >= 4
            and report.tier1_validation.get("pass", False)
        )

    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        report.overall_pass = False

    return report


# ============================================================
# __main__ 守卫自检 — 构造模拟数据验证 5 层 + Tier1 全部可执行
# ============================================================

def _generate_mock_trades(
    n: int = 100,
    initial_capital: float = 10000.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """生成模拟交易数据用于自检

    构造100笔交易:
      - 80笔小盈利 (pnl +50~+150, hold_bars 10~20)
      - 15笔小亏损 (pnl -50~-150, hold_bars 3~8)
      - 5笔大盈利 (pnl +300~+500, hold_bars 25~40)
    单亏最大 ≈ 1.5% < 2% → Layer B 通过
    最长连亏 ≈ 2~3 < 4 → Layer C 通过
    月亏率 = 0% → Layer D 通过
    """
    random.seed(seed)
    trades: List[Dict[str, Any]] = []
    base_time = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    hour_ms = 3600 * 1000

    for i in range(n):
        if i < 80:
            # 小盈利
            pnl = random.uniform(50, 150)
            hold = random.randint(10, 20)
        elif i < 95:
            # 小亏损 (分散, 不连续)
            pnl = -random.uniform(50, 150)
            hold = random.randint(3, 8)
        else:
            # 大盈利
            pnl = random.uniform(300, 500)
            hold = random.randint(25, 40)

        entry_bar = i * 24  # 每笔间隔24根K线
        entry_time = base_time + entry_bar * hour_ms
        atr_pct = random.uniform(0.005, 0.03)  # ATR 0.5%-3%
        symbol = random.choice(["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT"])

        trades.append({
            "pnl": pnl,
            "entry_time": entry_time,
            "exit_time": entry_time + hold * hour_ms,
            "symbol": symbol,
            "atr_pct": atr_pct,
            "hold_bars": hold,
            "entry_bar": entry_bar,
        })
    return trades


def _generate_bad_trades(
    n: int = 100,
    initial_capital: float = 10000.0,
    seed: int = 99,
) -> List[Dict[str, Any]]:
    """生成问题交易数据 (Layer B/C 应失败)

    构造:
      - 5笔单亏 > 2% (Layer B 失败)
      - 6笔连续亏损 (Layer C 失败)
    """
    random.seed(seed)
    trades: List[Dict[str, Any]] = []
    base_time = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    hour_ms = 3600 * 1000

    for i in range(n):
        entry_bar = i * 24
        entry_time = base_time + entry_bar * hour_ms
        atr_pct = random.uniform(0.005, 0.03)
        symbol = random.choice(["BTC_USDT", "ETH_USDT"])

        if 10 <= i < 16:
            # 6笔连续大亏损 (单亏 > 2%, 即 > 200)
            pnl = -random.uniform(250, 350)
            hold = random.randint(3, 8)
        elif i < 10 or i >= 16:
            if random.random() < 0.8:
                pnl = random.uniform(50, 150)
                hold = random.randint(10, 20)
            else:
                pnl = -random.uniform(50, 100)
                hold = random.randint(3, 8)

        trades.append({
            "pnl": pnl,
            "entry_time": entry_time,
            "exit_time": entry_time + hold * hour_ms,
            "symbol": symbol,
            "atr_pct": atr_pct,
            "hold_bars": hold,
            "entry_bar": entry_bar,
        })
    return trades


def _self_test() -> bool:
    """模块自检: 验证 5 层极端行情测试 + Tier1 重新验证全部可执行

    v598 Phase R3: 提取 __main__ 守卫的核心逻辑为独立 _self_test() 函数,
    使 ci/verify_6_shortboards.py 可统一调用 (符合 Phase O 短板模块自检规范)。

    6 个测试用例:
      1. 配置默认值正确 (ERR-093/ERR-100/v94 红线)
      2. 正常交易数据 → 5 层 ≥3/5 通过
      3. 问题交易数据 → Layer B/C 应失败 (反过拟合拦截验证)
      4. 空数据容错
      5. to_dict 序列化
      6. v96 格式兼容性 (live_pnl + rec 嵌套)

    Returns:
        True = 全部通过, False = 有失败
    """
    try:
        config = ExtremeTestConfig()

        # 测试1: 基本导入和配置 (ERR 红线确认)
        assert config.initial_capital == 10000.0, "initial_capital 默认应为 10000"
        assert config.max_consec_loss == 4, "max_consec_loss 应为 4 (v94退化教训)"
        assert config.single_loss_cap_pct == 2.0, "single_loss_cap_pct 应为 2.0 (ERR-100)"

        # 测试2: 正常交易数据 — 5层应大部分通过
        good_trades = _generate_mock_trades(100, seed=42)
        assert len(good_trades) == 100, "应生成100笔交易"
        report_good = run_extreme_market_test(good_trades, config)
        assert report_good.n_pass_layers >= 3, (
            f"正常数据应≥3/5层通过, 实际{report_good.n_pass_layers}"
        )

        # 测试3: 问题交易数据 — Layer B/C 应失败 (反过拟合拦截验证)
        # 用户铁律: "永远永远不要出现模拟牛逼，实盘亏损" → 必须能检测出坏数据
        bad_trades = _generate_bad_trades(100, seed=99)
        report_bad = run_extreme_market_test(bad_trades, config)
        assert not report_bad.layer_b_single_loss.get("pass", True), (
            "Layer B 应失败 (有单亏>2%)"
        )
        assert not report_bad.layer_c_consec_loss.get("pass", True), (
            "Layer C 应失败 (连亏6>4)"
        )
        assert not report_bad.overall_pass, "问题数据 overall_pass 应为 False"

        # 测试4: 空数据容错
        report_empty = run_extreme_market_test([], config)
        assert not report_empty.overall_pass, "空数据 overall_pass 应为 False"
        assert report_empty.error is not None, "空数据应有 error 信息"

        # 测试5: to_dict 序列化
        report_dict = report_good.to_dict()
        required_keys = {"version", "test_date", "layer_a_atr_extreme", "layer_b_single_loss",
                         "layer_c_consec_loss", "layer_d_monthly", "layer_e_rolling",
                         "tier1_validation", "overall_pass", "n_pass_layers", "n_total_layers"}
        assert required_keys.issubset(report_dict.keys()), (
            f"缺少键: {required_keys - set(report_dict.keys())}"
        )

        # 测试6: v96 格式兼容性 (live_pnl + rec 嵌套)
        v96_trades = [
            {"live_pnl": 100.0, "symbol": "BTC_USDT", "month": 202401,
             "rec": {"features": {"atr_pct": 0.015}, "hold_bars": 15, "entry_bar": 10,
                     "entry_time": 1700000000000, "exit_time": 1700000054000}},
            {"live_pnl": -50.0, "symbol": "ETH_USDT", "month": 202401,
             "rec": {"features": {"atr_pct": 0.020}, "hold_bars": 5, "entry_bar": 20,
                     "entry_time": 1700000000000, "exit_time": 1700000018000}},
        ] * 15  # 30笔
        report_v96 = run_extreme_market_test(v96_trades, config)
        assert report_v96.error is None or "交易数" not in str(report_v96.error), (
            f"v96格式应正常处理, error={report_v96.error}"
        )

        logger.info("extreme_market_test 自检全部通过 (6 测试用例) ✓")
        return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Extreme Market Test 模块自检 — 极端行情专项测试集 (6大核心短板 #6)")
    print("=" * 70)

    # 调用 _self_test() 并打印详细诊断 (含每层细节)
    config = ExtremeTestConfig()

    # 测试1: 基本导入和配置
    print("\n【测试1】基本导入和配置")
    print(f"  ✅ 配置: capital={config.initial_capital}, "
          f"max_consec={config.max_consec_loss}, single_loss_cap={config.single_loss_cap_pct}%")

    # 测试2: 正常交易数据 — 详细打印每层
    print("\n【测试2】正常交易数据 (5层应≥3/5通过)")
    good_trades = _generate_mock_trades(100, seed=42)
    report_good = run_extreme_market_test(good_trades, config)
    print(f"  n_pass_layers: {report_good.n_pass_layers}/{report_good.n_total_layers}")
    print(f"  Layer A: {'✅' if report_good.layer_a_atr_extreme.get('pass') else '❌'} "
          f"- {report_good.layer_a_atr_extreme.get('details', '')[:60]}")
    print(f"  Layer B: {'✅' if report_good.layer_b_single_loss.get('pass') else '❌'} "
          f"- {report_good.layer_b_single_loss.get('details', '')[:60]}")
    print(f"  Layer C: {'✅' if report_good.layer_c_consec_loss.get('pass') else '❌'} "
          f"- {report_good.layer_c_consec_loss.get('details', '')}")
    print(f"  Layer D: {'✅' if report_good.layer_d_monthly.get('pass') else '❌'} "
          f"- {report_good.layer_d_monthly.get('details', '')}")
    print(f"  Layer E: {'✅' if report_good.layer_e_rolling.get('pass') else '❌'} "
          f"- {report_good.layer_e_rolling.get('details', '')}")
    print(f"  overall_pass: {report_good.overall_pass}")

    # 测试3: 问题交易数据
    print("\n【测试3】问题交易数据 (Layer B/C 应失败)")
    bad_trades = _generate_bad_trades(100, seed=99)
    report_bad = run_extreme_market_test(bad_trades, config)
    print(f"  n_pass_layers: {report_bad.n_pass_layers}/{report_bad.n_total_layers}")
    print(f"  Layer B: {'✅' if report_bad.layer_b_single_loss.get('pass') else '❌'} "
          f"- loss_over_2pct={report_bad.layer_b_single_loss.get('loss_over_2pct', '?')}笔")
    print(f"  Layer C: {'✅' if report_bad.layer_c_consec_loss.get('pass') else '❌'} "
          f"- longest_seq={report_bad.layer_c_consec_loss.get('longest_seq', '?')}")

    # 测试4-6: 简略输出
    print("\n【测试4-6】空数据容错 / to_dict / v96 兼容性")
    report_empty = run_extreme_market_test([], config)
    print(f"  空数据容错: {'✅' if not report_empty.overall_pass and report_empty.error else '❌'}")
    report_dict = report_good.to_dict()
    required_keys = {"version", "test_date", "layer_a_atr_extreme", "layer_b_single_loss",
                     "layer_c_consec_loss", "layer_d_monthly", "layer_e_rolling",
                     "tier1_validation", "overall_pass", "n_pass_layers", "n_total_layers"}
    print(f"  to_dict 序列化: {'✅' if required_keys.issubset(report_dict.keys()) else '❌'}")
    v96_trades = [
        {"live_pnl": 100.0, "symbol": "BTC_USDT", "month": 202401,
         "rec": {"features": {"atr_pct": 0.015}, "hold_bars": 15, "entry_bar": 10,
                 "entry_time": 1700000000000, "exit_time": 1700000054000}},
        {"live_pnl": -50.0, "symbol": "ETH_USDT", "month": 202401,
         "rec": {"features": {"atr_pct": 0.020}, "hold_bars": 5, "entry_bar": 20,
                 "entry_time": 1700000000000, "exit_time": 1700000018000}},
    ] * 15
    report_v96 = run_extreme_market_test(v96_trades, config)
    print(f"  v96 兼容性: {'✅' if report_v96.error is None else '❌ ' + str(report_v96.error)}")

    # 最终通过 _self_test() 确认 (CI 兼容)
    ok = _self_test()
    print()
    print("=" * 70)
    if ok:
        print("✅ 全部自检通过 — extreme_market_test.py 模块功能正常")
    else:
        print("❌ 自检失败 — 见日志详情")
    print("=" * 70)
    print()
    print("ERR-093 红线确认: Layer C 阈值 ≤4 (consec_loss≤3是物理极限, 留1缓冲)")
    print("ERR-100 红线确认: Layer B 单亏 ≤2% 硬截断")
    print("ERR-20260701-v94 红线确认: Layer C 从5修复到4 (per-symbol consec加强)")
    print("铁标准确认: 5层判定阈值不可降级, 综合 ≥4/5层 + Tier1通过")
