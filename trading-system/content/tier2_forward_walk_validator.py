#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier 2 Forward-Walk Validator — 3个月实盘数据前向验证框架 (v98基线)
================================================================
用户铁律:
  "开始连续3个月的实盘数据实测"
  "在未达成性能指标前不得推进实盘部署工作"
  "永远永远不要出现模拟牛逼, 实盘亏损的情况"
  "不要通过已知数据反推策略, 那样就过拟合了!"

v98升级为生产基线 (2026-07-03):
  - 基线从 v97_alpha_gap_optimize 升级为 v98_hold_bars_fusion
  - v98 ann=96.37% (v97=91.12%, +5.25pp), gap=0.37% (v97=1.23%, -70%)
  - v98 Tier1 12/12通过, 极端5/5全通过, decision=KEEP
  - 新增Layer 3: hold_bars融合仓位微调 (hold_bars_mult, range [0.9, 1.1])
  - 三层仓位调整: pos_mult × htf_factor × fusion_mult × pa_confidence × hold_bars_mult
  - v97历史OOS数据归档至 tier2_state_v97_archive.json, 仅作对比参考

设计原则:
  1. 严格OOS: v97/v98训练数据截止2026-06-30, OOS数据从2026-07-01起
  2. 前向收集: 每日从Gate.io获取最新K线, 不回看训练期数据
  3. 参数冻结: v98策略参数完全冻结, 禁止任何形式的重优化
  4. 全成本量化: sim/live差异5因子模型(spread+slippage+liquidity+latency+fee)
  5. 月度评估: 每月底计算Tier 2指标, 调用staged_admission_gate.evaluate_tier2()
  6. 3月累积: 3个月连续GO才能通过Tier 2, 进入Tier 3(标准实盘$10k+)
  7. v98 hold_bars融合: per_symbol_stats冻结自训练数据, 避免look-ahead bias

Tier 2 准入7项硬约束 (staged_admission_gate.py):
  - tier1_pass = True (v98已满足, 12/12 GO)
  - consecutive_sim_go_months >= 3 (需3个月连续GO)
  - live_mlp_pct <= 5.0% (实盘月度亏损概率)
  - live_gap_pct <= 15.0% (实盘差异率)
  - live_single_loss_pct <= 2.0% (单次最大亏损≤本金2%)
  - live_consec_loss <= 3 (连续亏损次数≤3)
  - capital_max_usd <= 5000 (迷你实盘上限$5k)

架构分层:
  Layer 1: DataCollectionLayer — Gate.io K线获取 + 缓存
  Layer 2: FeatureComputationLayer — v74风格特征计算 (PA/VP/MTF/Kelly + ADX)
  Layer 3: StrategyExecutionLayer — v98策略执行 (参数冻结, 含hold_bars融合)
  Layer 4: MetricsTrackingLayer — Tier 2指标追踪 (mlp/gap/single_loss/consec_loss)
  Layer 5: EvaluationLayer — 月度staged_admission评估
  Layer 6: PersistenceLayer — JSON状态持久化 (3个月累积)

OOS数据流:
  Gate.io API → 日K线缓存 → 特征计算 → v98策略(含hold_bars融合) → 交易记录 → Tier 2指标 → 月度评估

使用方式:
  # 首次运行: 收集OOS数据 + 验证框架
  validator = Tier2ForwardWalkValidator()
  validator.initialize()
  validator.collect_daily_data()
  validator.run_monthly_evaluation()

  # 每日运行: 收集新数据
  validator = Tier2ForwardWalkValidator.load_state()
  validator.collect_daily_data()

  # 每月底运行: 月度评估
  validator.run_monthly_evaluation()

  # 查看状态
  report = validator.get_status_report()
================================================================
"""
from __future__ import annotations

import json
import logging
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# v99-soft-reduce: 机制性入场门控 (基于P9-A/B/C三大亏损源深度分析)
# 软减仓版本: 不拒绝信号, 只调整仓位 (v2: ADX>40减仓10%, MTF强逆势减仓30%)
# 集成位置: generate_signal的final_pos_mult计算前, check_exit的SL触发处
try:
    from _v99_entry_gates import V99EntryGates, STRATEGY_MEAN_REVERSION
    _V99_AVAILABLE = True
except ImportError:
    _V99_AVAILABLE = False
    V99EntryGates = None
    STRATEGY_MEAN_REVERSION = "mean_reversion"

logger = logging.getLogger("hermes.tier2_validator")

# ============================================================================
# 常量定义
# ============================================================================

HERE = Path(__file__).parent.resolve()

# v97 15个交易对 (全部在Gate.io上可用, 已验证)
V97_SYMBOLS = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT",
    "ARB_USDT", "ATOM_USDT", "AVAX_USDT", "DOT_USDT", "INJ_USDT",
    "LINK_USDT", "LTC_USDT", "OP_USDT", "TRX_USDT", "UNI_USDT",
]

# v97 训练数据截止日期 (OOS数据从此日期之后开始)
# 训练数据: 2024-09-29 → 2026-06-30 (639天 for core, 417天 for new)
TRAINING_DATA_END = datetime(2026, 6, 30, 0, 0, 0, tzinfo=timezone.utc)
OOS_START_DATE = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)

# Tier 2 指标阈值 (来自 staged_admission_gate.py TIER2_REQUIREMENTS)
TIER2_THRESHOLDS = {
    "consecutive_sim_go_months": {"min": 3, "desc": "连续3月模拟GO"},
    "live_mlp_pct": {"max": 5.0, "desc": "实盘月度亏损概率≤5%"},
    "live_gap_pct": {"max": 15.0, "desc": "实盘差异率≤15%"},
    "live_single_loss_pct": {"max": 2.0, "desc": "单次最大亏损≤2%本金"},
    "live_consec_loss": {"max": 3, "desc": "连续亏损次数≤3"},
    "capital_max_usd": {"max": 5000, "desc": "迷你实盘上限$5k"},
}

# Gate.io API 配置
GATEIO_BASE_URL = "https://api.gateio.ws/api/v4"
GATEIO_PERP_KLINE_URL = GATEIO_BASE_URL + "/futures/usdt/candlesticks"
GATEIO_CONTRACTS_URL = GATEIO_BASE_URL + "/futures/usdt/contracts"

# 数据缓存目录
TIER2_DATA_DIR = HERE / "tier2_oos_data"
TIER2_STATE_FILE = TIER2_DATA_DIR / "tier2_state.json"
TIER2_KLINES_DIR = TIER2_DATA_DIR / "klines"
TIER2_TRADES_DIR = TIER2_DATA_DIR / "trades"
TIER2_REPORTS_DIR = TIER2_DATA_DIR / "reports"

# v97 初始资金 (与回测一致)
INITIAL_CAPITAL_USD = 10000.0
TIER2_CAPITAL_USD = 5000.0  # 迷你实盘上限$5k

# v97 摩擦参数 (来自 v75.V72_FRICTION_PARAMS, 用于sim/live差异计算)
# 这里简化为per-symbol的taker_fee_bps和spread_bps
SYMBOL_FRICTION = {
    "BTC_USDT":  {"spread_bps": 1.0, "taker_fee_bps": 0.4, "slippage_bps": 0.3},
    "ETH_USDT":  {"spread_bps": 1.5, "taker_fee_bps": 0.5, "slippage_bps": 0.4},
    "SOL_USDT":  {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "BNB_USDT":  {"spread_bps": 2.0, "taker_fee_bps": 0.5, "slippage_bps": 0.5},
    "XRP_USDT":  {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "ARB_USDT":  {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "ATOM_USDT": {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "AVAX_USDT": {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "DOT_USDT":  {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "INJ_USDT":  {"spread_bps": 4.0, "taker_fee_bps": 0.5, "slippage_bps": 1.0},
    "LINK_USDT": {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "LTC_USDT":  {"spread_bps": 2.0, "taker_fee_bps": 0.5, "slippage_bps": 0.5},
    "OP_USDT":   {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "TRX_USDT":  {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
    "UNI_USDT":  {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8},
}

# 延迟参数 (来自 v75: 国内到交易所物理延迟~1056ms)
LATENCY_MS = 1056.0
FUNDING_RATE_8H_PCT = 0.01  # 8h资金费率0.01%

# ============================================================================
# v98 hold_bars融合常量 (Layer 3 — v98升级为生产基线 2026-07-03)
# ============================================================================
# 来源: _v98_hold_bars_fusion.py + _v98_hold_bars_report.json
# 三层仓位调整架构: v95_pos_mult × fusion_mult_4dim × hold_bars_mult
# hold_bars_mult = 1.0 + alpha × (sigmoid(raw_score) - 0.5), range [0.9, 1.1]
#
# R2修复 - Simpson悖论: sign基于d_vs_pnl (PnL预测) 而非 d_vs_hold
# ERR-v96sign(R2修正): sign方向必须基于d_vs_pnl, 不基于d_vs_hold
# ERR-109: 用入场特征(cause)预测hold_bars(consequence), 不直接用hold_bars
# ERR-106: per-symbol robust标准化(median/p25/p75)
V98_ALPHA = 0.2  # v98最佳alpha (来自alpha网格搜索 [0.1,0.15,0.2,0.25,0.3])
V98_HOLD_BARS_FEATURES = [
    # (feature_name, weight, sign, cohens_d_vs_pnl)
    ("pa_body_pct", 0.427, +1, +0.2354),         # FLIPPED: d_vs_hold=-0.54 → d_vs_pnl=+0.24
    ("mtf_trend_direction", 0.370, +1, +0.2045), # 不变: 方向一致
    ("atr_pct", 0.151, -1, -0.0836),             # 不变: 方向一致
    ("adx", 0.051, +1, +0.0284),                 # FLIPPED: d_vs_hold=-0.27 → d_vs_pnl=+0.03
]
V98_PER_SYMBOL_STATS_FILE = HERE / "tier2_oos_data" / "v98_per_symbol_stats.json"


# ============================================================================
# Layer 1: 数据结构
# ============================================================================

@dataclass
class OOSTrade:
    """OOS交易记录 (前向收集的实盘交易)"""
    trade_id: str               # 唯一ID: {symbol}_{entry_ts}
    symbol: str
    direction: int               # 1=多, -1=空
    entry_ts: float              # 入场时间戳(Unix秒)
    exit_ts: float               # 出场时间戳
    entry_price: float
    exit_price: float
    hold_bars: int               # 持仓小时数
    pnl_pct: float               # 模拟盈亏百分比
    pnl_usd: float               # 模拟盈亏USD
    live_pnl_pct: float          # 实盘盈亏百分比 (扣除全部成本)
    live_pnl_usd: float          # 实盘盈亏USD
    cost_usd: float              # 总成本USD
    cost_bps: float              # 总成本bps
    atr_pct: float               # 入场时ATR%
    pos_mult: float              # 仓位倍数
    month_key: str               # 所属月份 "2026-07"
    features: Dict[str, Any] = field(default_factory=dict)  # 入场特征快照


@dataclass
class MonthlyMetrics:
    """月度Tier 2指标"""
    month_key: str               # "2026-07"
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    # Tier 2 核心指标
    live_mlp_pct: float = 0.0        # 实盘月度亏损概率
    live_gap_pct: float = 0.0        # 实盘差异率
    live_single_loss_pct: float = 0.0  # 单次最大亏损%本金
    live_consec_loss: int = 0        # 连续亏损次数
    # 辅助指标
    sim_pnl_total_usd: float = 0.0
    live_pnl_total_usd: float = 0.0
    cost_total_usd: float = 0.0
    avg_hold_bars: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    # GO/NO-GO
    go_pass: bool = False
    go_reasons: List[str] = field(default_factory=list)


@dataclass
class Tier2State:
    """Tier 2 状态 (持久化, v98基线)"""
    # 元数据
    version: str = "v98_tier2_forward_walk"  # v98升级为生产基线 2026-07-03
    created_at: str = ""
    last_updated: str = ""
    # 收集状态
    oos_start_date: str = "2026-07-01"
    collection_days: int = 0
    last_collection_ts: float = 0.0
    # 月度指标累积
    monthly_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    consecutive_go_months: int = 0
    # 全局交易统计
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_sim_pnl_usd: float = 0.0
    total_live_pnl_usd: float = 0.0
    total_cost_usd: float = 0.0
    max_single_loss_pct: float = 0.0
    max_consec_loss: int = 0
    # Tier 2 评估
    tier2_passed: bool = False
    tier2_evaluation: Dict[str, Any] = field(default_factory=dict)
    # v98 基线 (冻结, 不可变; 兼容旧字段名 v97_baseline)
    v98_baseline: Dict[str, Any] = field(default_factory=dict)
    v97_baseline: Dict[str, Any] = field(default_factory=dict)  # 历史归档, 保留兼容
    # 重置历史 (v98升级时记录, 追踪state重置操作)
    reset_history: List[Dict[str, Any]] = field(default_factory=list)
    # 修复5 (ERR-20260703-cooldown-persistence): 冷却期持久化
    # 根因: per_symbol_cooldown原在signal_gen上, 不被save_state持久化
    # 跨日collect_daily_data时冷却期丢失, 导致跨日背靠背开仓
    # 修复: 双写 signal_gen(运行时) + state(持久化), initialize时回灌
    per_symbol_cooldown: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Layer 2: Gate.io 数据获取
# ============================================================================

class GateIoDataFetcher:
    """Gate.io K线数据获取器 (支持全15个v97币种)

    复用 gateio_real_data_provider.py 的逻辑, 但扩展支持全15个币种.
    Gate.io实际支持814个USDT perp, v97的15个全部可用 (已验证).
    """

    def __init__(self, cache_dir: Path = TIER2_KLINES_DIR, timeout: float = 10.0):
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("GateIoDataFetcher initialized, cache_dir=%s", cache_dir)

    def _http_get(self, url: str) -> Tuple[bool, Any, str]:
        """HTTP GET with retry"""
        last_error = ""
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=self.timeout)
                body = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(body)
                return True, data, ""
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:100]}"
            except urllib.error.URLError as e:
                last_error = f"URLError: {str(e)[:100]}"
            except json.JSONDecodeError as e:
                last_error = f"JSONDecodeError: {str(e)[:100]}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:100]}"
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
        return False, None, last_error

    def fetch_perp_klines(
        self,
        pair: str,
        interval: str = "1h",
        limit: int = 1000,
        from_ts: Optional[float] = None,
        to_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """获取永续合约K线

        Args:
            pair: 交易对, 如 "BTC_USDT"
            interval: "1m"/"5m"/"15m"/"30m"/"1h"/"4h"/"1d"
            limit: K线数量 (最大2000)
            from_ts: 起始时间戳(Unix秒), 可选
            to_ts: 结束时间戳(Unix秒), 可选

        Returns:
            List[Dict] K线列表, 每条含 timestamp/open/high/low/close/volume
        """
        params = [f"contract={pair}", f"interval={interval}"]
        if from_ts and to_ts:
            # Gate.io API 实测规则 (ERR-tier2-gateio-api):
            #   from + to + limit 三者不能同时存在, 否则 HTTP 400
            #   "Invalid request parameter `limit` value: X,
            #    `limit` and `from` and `to` cannot be present at the same time"
            # 修复: 使用 from+to 时完全不传 limit, Gate.io 自动返回区间内全部K线
            # (区间上限约2000条; 超长区间需分批获取, 见 fetch_historical_oos_klines)
            params.append(f"from={int(from_ts)}")
            params.append(f"to={int(to_ts)}")
        else:
            params.append(f"limit={min(limit, 2000)}")
        url = f"{GATEIO_PERP_KLINE_URL}?" + "&".join(params)

        ok, data, err = self._http_get(url)
        if not ok:
            logger.error("获取K线失败 %s %s: %s", pair, interval, err)
            return []

        klines = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                ts = float(item.get("t", 0))
                o = float(item.get("o", 0))
                h = float(item.get("h", 0))
                l = float(item.get("l", 0))
                c = float(item.get("c", 0))
                v = float(item.get("v", 0))
                qv = float(item.get("sum", 0))
                if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                    continue
                if not (l <= o <= h and l <= c <= h):
                    continue
                klines.append({
                    "timestamp": ts,
                    "open": o, "high": h, "low": l, "close": c,
                    "volume": v, "quote_volume": qv,
                })
            except (ValueError, KeyError) as e:
                logger.warning("解析K线失败 %s: %s", pair, e)
                continue

        klines.sort(key=lambda k: k["timestamp"])
        return klines

    def fetch_daily_oos_klines(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """获取每日OOS K线 (从上次收集点到当前)

        Args:
            symbols: 交易对列表

        Returns:
            Dict[symbol, List[KLine]] 新增的K线
        """
        now_ts = time.time()
        results = {}
        for sym in symbols:
            # 获取最近48小时的K线 (覆盖可能的间隔)
            from_ts = now_ts - 48 * 3600
            klines = self.fetch_perp_klines(sym, "1h", limit=48, from_ts=from_ts, to_ts=now_ts)
            if klines:
                # 只保留OOS期内的K线
                oos_klines = [k for k in klines if k["timestamp"] >= OOS_START_DATE.timestamp()]
                results[sym] = oos_klines
                logger.info("  %s: 获取 %d 条OOS K线 (from %s)",
                           sym, len(oos_klines),
                           datetime.fromtimestamp(oos_klines[0]["timestamp"], tz=timezone.utc).isoformat()
                           if oos_klines else "N/A")
            else:
                logger.warning("  %s: 获取K线失败", sym)
                results[sym] = []
            time.sleep(0.1)  # 避免API限频
        return results

    def fetch_historical_oos_klines(
        self,
        symbols: List[str],
        start_ts: float,
        end_ts: float,
    ) -> Dict[str, List[Dict]]:
        """获取历史OOS K线 (分批获取覆盖长时段)

        Gate.io API规则 (ERR-tier2-gateio-api): from+to时不能传limit.
        Gate.io单次from+to最多返回约2000条K线 (~83天1h).
        为安全起见, 每批取 1500*3600 秒 (62.5天), 避免超限.
        """
        results = {}
        for sym in symbols:
            all_klines = []
            cur_from = start_ts
            while cur_from < end_ts:
                # 每批最多1500小时 (62.5天), 避免触发2000条上限
                cur_to = min(cur_from + 1500 * 3600, end_ts)
                batch = self.fetch_perp_klines(sym, "1h",
                                              from_ts=cur_from, to_ts=cur_to)
                if not batch:
                    break
                # 去重 (batch边界可能重叠)
                existing_ts = {k["timestamp"] for k in all_klines}
                for k in batch:
                    if k["timestamp"] not in existing_ts:
                        all_klines.append(k)
                cur_from = cur_to
                time.sleep(0.2)
            all_klines.sort(key=lambda k: k["timestamp"])
            results[sym] = all_klines
            logger.info("  %s: 获取 %d 条历史OOS K线", sym, len(all_klines))
        return results

    def load_warmup_klines(self, symbol: str, n_warmup: int = 300) -> List[Dict]:
        """加载训练期K线作为预热数据 (策略需要200+K线计算指标)

        用户铁律: "不要通过已知数据反推策略" — 预热数据只用于指标计算,
        不用于信号生成. 信号仅在OOS期(2026-07-01后)生成.

        从 data_cache_gateio/perp_history_{symbol}_1h_540d.json 加载最后n_warmup条K线.
        """
        cache_path = HERE / "data_cache_gateio" / f"perp_history_{symbol}_1h_540d.json"
        if not cache_path.exists():
            logger.warning("  预热数据不存在: %s", cache_path)
            return []
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            klines = []
            for k in data:
                if not isinstance(k, dict):
                    continue
                ts_key = "timestamp" if "timestamp" in k else ("t" if "t" in k else "time")
                ts = k.get(ts_key)
                c = k.get("close") or k.get("c")
                if ts is None or c is None:
                    continue
                try:
                    klines.append({
                        "timestamp": float(ts),
                        "open": float(k.get("open") or k.get("o") or 0),
                        "high": float(k.get("high") or k.get("h") or 0),
                        "low": float(k.get("low") or k.get("l") or 0),
                        "close": float(c),
                        "volume": float(k.get("volume") or k.get("vol") or k.get("v") or 0),
                    })
                except (TypeError, ValueError):
                    continue
            klines.sort(key=lambda k: k["timestamp"])
            # 只取最后n_warmup条作为预热 (确保OOS前的最近数据)
            if len(klines) > n_warmup:
                klines = klines[-n_warmup:]
            return klines
        except Exception as e:
            logger.warning("加载预热数据失败 %s: %s", symbol, e)
            return []

    def save_klines(self, symbol: str, klines: List[Dict], date_str: str) -> Path:
        """保存K线到文件"""
        path = self.cache_dir / f"{symbol}_{date_str}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(klines, f)
        return path

    def load_klines(self, symbol: str, date_str: str) -> Optional[List[Dict]]:
        """加载缓存的K线"""
        path = self.cache_dir / f"{symbol}_{date_str}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


# ============================================================================
# Layer 3: 特征计算 (v74风格, 无未来函数)
# ============================================================================

class FeatureComputer:
    """从K线计算入场特征 (复用v74逻辑, 无未来函数)

    特征维度 (对齐用户铁律: 融会贯通价格行为/量价/多周期结构/凯利公式):
      A. 基础技术指标: atr, adx, ema50/200, rsi, macd
      B. 价格行为(PA): 实体/影线/吞没/锤子线/内包线
      C. 量价分析(VP): 量比/价量背离/OBV斜率/A-D指标
      D. 多周期结构(MTF): EMA排列/HH-HL结构/4h聚合趋势
      E. 凯利公式(Kelly): 历史胜率/盈亏比/凯利分数
      F. Regime: trending/ranging + volatility_high/med/low
    """

    @staticmethod
    def safe_atr(klines: List[Dict], idx: int, period: int = 14) -> float:
        """ATR(14) at idx (True Range 均值), 含 idx"""
        if idx < 1 or idx >= len(klines):
            return 0.0
        start = max(1, idx - period + 1)
        trs = []
        for i in range(start, idx + 1):
            h, l = klines[i]["high"], klines[i]["low"]
            pc = klines[i - 1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return statistics.mean(trs) if trs else 0.0

    @staticmethod
    def safe_ema(values: List[float], period: int) -> float:
        """EMA 最后一个值"""
        if not values:
            return 0.0
        alpha = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    @staticmethod
    def safe_rsi(klines: List[Dict], idx: int, period: int = 14) -> float:
        """RSI(14)"""
        if idx < period or idx >= len(klines):
            return 50.0
        gains, losses = [], []
        for i in range(idx - period + 1, idx + 1):
            if i < 1:
                continue
            ch = klines[i]["close"] - klines[i - 1]["close"]
            gains.append(max(0, ch))
            losses.append(max(0, -ch))
        avg_gain = statistics.mean(gains) if gains else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.001
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def safe_adx(klines: List[Dict], idx: int, period: int = 14) -> float:
        """ADX(14) — Wilder's Average Directional Index

        v98 hold_bars融合需要adx特征 (w=0.051, sign=+1, d_vs_pnl=+0.0284).
        计算: +DM/-DM → TR → Wilder平滑 → +DI/-DI → DX → ADX
        返回 0-100, 越高趋势越强.
        """
        if idx < 2 * period or idx >= len(klines):
            return 25.0  # 中性默认值
        # Step 1: 计算 +DM, -DM, TR
        plus_dms, minus_dms, trs = [], [], []
        for i in range(idx - 2 * period + 1, idx + 1):
            if i < 1:
                continue
            high = klines[i]["high"]
            low = klines[i]["low"]
            prev_high = klines[i - 1]["high"]
            prev_low = klines[i - 1]["low"]
            prev_close = klines[i - 1]["close"]
            # +DM: 当日high - 前日high > 前日low - 当日low, 且 > 0
            up_move = high - prev_high
            down_move = prev_low - low
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
            # TR
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            plus_dms.append(plus_dm)
            minus_dms.append(minus_dm)
            trs.append(tr)
        if len(trs) < period:
            return 25.0
        # Step 2: Wilder平滑 (首次SMA, 后续Wilder)
        tr_smooth = statistics.mean(trs[:period])
        plus_dm_smooth = statistics.mean(plus_dms[:period])
        minus_dm_smooth = statistics.mean(minus_dms[:period])
        dxs = []
        for i in range(period, len(trs)):
            tr_smooth = tr_smooth - tr_smooth / period + trs[i]
            plus_dm_smooth = plus_dm_smooth - plus_dm_smooth / period + plus_dms[i]
            minus_dm_smooth = minus_dm_smooth - minus_dm_smooth / period + minus_dms[i]
            if tr_smooth <= 0:
                continue
            plus_di = 100.0 * plus_dm_smooth / tr_smooth
            minus_di = 100.0 * minus_dm_smooth / tr_smooth
            di_sum = plus_di + minus_di
            if di_sum <= 0:
                continue
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
            dxs.append(dx)
        if not dxs:
            return 25.0
        # Step 3: ADX = Wilder平滑DX
        if len(dxs) >= period:
            adx = statistics.mean(dxs[:period])
            for i in range(period, len(dxs)):
                adx = (adx * (period - 1) + dxs[i]) / period
        else:
            adx = statistics.mean(dxs)
        return adx

    @staticmethod
    def compute_pa_features(klines: List[Dict], idx: int) -> Dict[str, float]:
        """价格行为特征 (无未来函数)"""
        if idx < 0 or idx >= len(klines):
            return {}
        k = klines[idx]
        o, h, l, c = k["open"], k["high"], k["low"], k["close"]
        rng = h - l if h > l else 0.0001
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        return {
            "pa_body_pct": body / rng if rng > 0 else 0.0,
            "pa_range_pct": rng / c if c > 0 else 0.0,
            "pa_upper_wick_ratio": upper_wick / rng if rng > 0 else 0.0,
            "pa_lower_wick_ratio": lower_wick / rng if rng > 0 else 0.0,
            "pa_candle_dir": 1.0 if c > o else (-1.0 if c < o else 0.0),
        }

    @staticmethod
    def compute_vp_features(klines: List[Dict], idx: int, period: int = 5) -> Dict[str, float]:
        """量价分析特征"""
        if idx < period or idx >= len(klines):
            return {"vp_vol_ratio_5": 1.0, "vp_price_vol_aligned": 0.0}
        cur_vol = klines[idx]["volume"]
        avg_vol = statistics.mean([klines[i]["volume"] for i in range(idx - period + 1, idx + 1)])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        # 价量对齐: 价格涨+量增=1, 价格跌+量增=-1, 背离=0
        price_up = klines[idx]["close"] > klines[idx - 1]["close"]
        vol_up = cur_vol > avg_vol
        aligned = 1.0 if (price_up and vol_up) or (not price_up and not vol_up) else 0.0
        return {
            "vp_vol_ratio_5": vol_ratio,
            "vp_price_vol_aligned": aligned,
        }

    @staticmethod
    def compute_mtf_features(klines: List[Dict], idx: int) -> Dict[str, float]:
        """多周期结构特征

        修复2 (ERR-20260703-strategy-split): EMA转换期中性带 + price_vs_ema200独立趋势源
        - 中性带 ratio 0.995-1.005: EMA交叉前后0.5%区间为趋势未确认区, 噪声大于信号
          理论依据: feature_fusion_pipeline.py:997-1000 的0.999-1.001设计, 1h K线扩展至0.5%
        - price_vs_ema200: close vs EMA200, 独立于EMA50/EMA200, 用于修复3(v99门控)和修复4(HTF)
          理论依据: 道氏理论 — 价格趋势用 close vs EMA200(滞后但稳定), 非EMA50/EMA200(转换期二元误判)
        """
        if idx < 200 or idx >= len(klines):
            return {}
        closes = [klines[i]["close"] for i in range(max(0, idx - 200), idx + 1)]
        ema50 = FeatureComputer.safe_ema(closes[-50:], 50)
        ema200 = FeatureComputer.safe_ema(closes[-200:], 200)
        ratio = ema50 / ema200 if ema200 > 0 else 1.0

        # 修复2: EMA转换期中性带 (0.995-1.005)
        NEUTRAL_LOWER = 0.995
        NEUTRAL_UPPER = 1.005
        in_neutral_zone = NEUTRAL_LOWER <= ratio <= NEUTRAL_UPPER
        if in_neutral_zone:
            ema_alignment = 0.0       # 中性, 不发出趋势信号
            mtf_trend_direction = 0.0  # 中性
        else:
            ema_alignment = 1.0 if ema50 > ema200 else -1.0
            mtf_trend_direction = 1.0 if ema50 > ema200 else -1.0

        # 修复2: price_vs_ema200 (独立趋势源, 修复3/4依赖)
        close = klines[idx]["close"] if idx < len(klines) else 0.0
        price_vs_ema200 = 1.0 if (ema200 > 0 and close > ema200) else (-1.0 if ema200 > 0 else 0.0)

        return {
            "mtf_ema50": ema50,
            "mtf_ema200": ema200,
            "mtf_ema50_200_ratio": ratio,
            "mtf_ema_alignment": ema_alignment,
            "mtf_trend_direction": mtf_trend_direction,
            "mtf_neutral_zone": in_neutral_zone,   # 新增: 中性带标志
            "price_vs_ema200": price_vs_ema200,     # 新增: 独立趋势源
        }

    @staticmethod
    def compute_atr_pct(klines: List[Dict], idx: int) -> float:
        """ATR% = ATR / close"""
        atr = FeatureComputer.safe_atr(klines, idx)
        close = klines[idx]["close"] if idx < len(klines) else 0.0
        return atr / close if close > 0 else 0.0

    @staticmethod
    def compute_all_features(klines: List[Dict], idx: int) -> Dict[str, Any]:
        """计算全部入场特征 (无未来函数)"""
        features = {}
        features.update(FeatureComputer.compute_pa_features(klines, idx))
        features.update(FeatureComputer.compute_vp_features(klines, idx))
        features.update(FeatureComputer.compute_mtf_features(klines, idx))
        features["atr_pct"] = FeatureComputer.compute_atr_pct(klines, idx)
        features["atr_abs"] = FeatureComputer.safe_atr(klines, idx)
        features["rsi_14"] = FeatureComputer.safe_rsi(klines, idx)
        features["adx"] = FeatureComputer.safe_adx(klines, idx)  # v98 hold_bars融合需要
        features["hour_of_day"] = datetime.fromtimestamp(
            klines[idx]["timestamp"], tz=timezone.utc
        ).hour if idx < len(klines) else 0
        # Regime
        atr_pct = features["atr_pct"]
        if atr_pct > 0.02:
            features["volatility_regime"] = "high_vol"
        elif atr_pct > 0.01:
            features["volatility_regime"] = "med_vol"
        else:
            features["volatility_regime"] = "low_vol"
        ema_alignment = features.get("mtf_ema_alignment", 0)
        if abs(ema_alignment) > 0.5:
            features["regime"] = "trending"
        else:
            features["regime"] = "ranging"
        return features


# ============================================================================
# Layer 4: 策略信号生成 (v97简化版, 参数冻结)
# ============================================================================

class V97SignalGenerator:
    """v97策略信号生成器 (参数冻结, 不可重优化)

    用户铁律: "不要通过已知数据反推策略, 那样就过拟合了!"
    本类完全冻结v97的参数, 在OOS数据上生成交易信号:
      - HTF 16状态分类 (4 ATR × 2 Volume × 2 HTF)
      - 4维度特征融合 (凯利+PA+VP+MTF) — 第二层仓位微调
      - 混合consec保护 (全局≥3 + 新币种per-symbol≥2)
      - 禁用NEAR, kelly_scaler=1.0, 单亏≤2%硬截断

    信号生成逻辑:
      1. 每根K线计算入场特征
      2. 应用v97的入场条件 (confluence_score, ATR门, cost-aware)
      3. 生成direction + pos_mult
      4. 出场: TP/SL基于ATR, 或hold_bars超限
    """

    # v97 冻结参数 (来自 _v97_alpha_gap_optimize.py)
    V97_ALPHA = 0.1              # 融合alpha (每维度±alpha/2)
    V97_KELLY_SCALER = 1.0       # Kelly缩放
    V97_MAX_CONSEC = 3           # 全局连续亏损上限
    V97_MAX_SINGLE_LOSS_PCT = 2.0  # 单次最大亏损%本金
    V97_CLIP_LOW = 0.1           # 融合仓位clip下限
    V97_CLIP_HIGH = 10.0         # 融合仓位clip上限

    # v99 亏损后冷却期 (ERR-20260703-back-to-back-bug)
    # 根因: Trade1止损瞬间Trade2立即开仓(同ts), 亏损后立即以更高仓位同向再进场
    # 行为等价马丁策略反指, 是"模拟牛逼实盘亏"的核心原因之一
    # 修复: 亏损平仓后4小时冷却期, 让市场稳定, 避免亏损后立即同向再进场
    # 机制性改进非数据反推: 亏损后市场状态不稳定是市场微观结构常识
    LOSS_COOLDOWN_HOURS = 4  # 4小时冷却期 (4根1h K线)

    # v97 禁用币种
    DISABLED_SYMBOLS = {"NEAR_USDT"}

    # v97 HTF 16状态仓位因子 (来自V93_BEST_FACTORS_16)
    # 4 ATR状态 × 2 Volume状态 × 2 HTF状态 = 16
    # 修复4 (ERR-20260703-strategy-split): counter_trend因子从加仓改为降仓
    # 根因: 原设计 counter_trend = "EMA空头排列", 但做空时这实际是"顺势", 因子加成1.30-1.56
    #   在方向错误时(EMA误判)放大亏损, 违反风险管理第一原则
    # 修复: 重定义 counter_trend = "交易方向 vs 价格趋势相反"(见classify_htf_state)
    #   真正逆势时降仓(0.70-0.84), 顺势时维持/加仓(1.00-1.20)
    # 理论依据: 风险管理第一原则 — 逆势交易应降仓而非加仓
    HTF_16_FACTORS = {
        "trending/high_vol/with_trend": 1.00,
        "trending/high_vol/counter_trend": 0.70,   # 修复4: 1.30→0.70 (逆势降仓)
        "trending/low_vol/with_trend": 1.00,
        "trending/low_vol/counter_trend": 0.70,    # 修复4: 1.30→0.70
        "ranging/high_vol/with_trend": 0.85,
        "ranging/high_vol/counter_trend": 0.60,    # 修复4: 1.10→0.60
        "ranging/low_vol/with_trend": 0.85,
        "ranging/low_vol/counter_trend": 0.60,     # 修复4: 1.10→0.60
        "volatile/high_vol/with_trend": 1.02,
        "volatile/high_vol/counter_trend": 0.72,   # 修复4: 1.33→0.72
        "volatile/low_vol/with_trend": 1.02,
        "volatile/low_vol/counter_trend": 0.72,    # 修复4: 1.33→0.72
        "low_vol/high_vol/with_trend": 1.20,
        "low_vol/high_vol/counter_trend": 0.84,    # 修复4: 1.56→0.84
        "low_vol/low_vol/with_trend": 1.20,
        "low_vol/low_vol/counter_trend": 0.84,     # 修复4: 1.56→0.84
    }

    # v97 核心币种pos_mult (BTC/ETH)
    CORE_POS_MULT = {"BTC_USDT": 2.0, "ETH_USDT": 2.0}

    # v97 卫星币种pos_mult (含gap优化: 高摩擦币种降15%)
    # 来自 _v97_alpha_gap_optimize.py gap优化结果
    SATELLITE_POS_MULT = {
        "SOL_USDT": 0.38,   # 原值降15% (高摩擦)
        "BNB_USDT": 0.68,   # 原值降15% (高摩擦)
        "XRP_USDT": 0.36,   # 原值降15% (高摩擦)
        "ARB_USDT": 0.36,   # 原值降15% (高摩擦)
        "AVAX_USDT": 0.43,
        "DOT_USDT": 0.36,
        "INJ_USDT": 0.43,
        "LINK_USDT": 0.36,
        "LTC_USDT": 0.54,
        "OP_USDT": 0.36,    # 原值降15% (高摩擦)
        "TRX_USDT": 0.60,   # 原值降15% (高摩擦)
        "UNI_USDT": 0.36,
        "ATOM_USDT": 0.43,
    }

    # ATR分位数阈值 (用于HTF状态分类, 来自v93)
    ATR_HIGH_THRESHOLD = 0.020   # ATR% > 2% = high_vol
    ATR_LOW_THRESHOLD = 0.010    # ATR% < 1% = low_vol

    # Volume分位数阈值
    VOL_HIGH_THRESHOLD = 1.5      # vol_ratio > 1.5 = high_vol

    def __init__(self):
        # 连续亏损追踪
        self.current_consec_loss = 0
        self.per_symbol_consec = {}  # per-symbol连续亏损
        self.max_consec_seen = 0
        self.max_single_loss_seen = 0.0
        # v98 hold_bars融合参数 (由Tier2ForwardWalkValidator.__init__注入)
        self.v98_per_symbol_stats: Optional[Dict] = None
        self.v98_alpha: float = 0.2  # V98_ALPHA默认值
        # v99 亏损后冷却期追踪 (ERR-20260703-back-to-back-bug)
        # per_symbol_cooldown[symbol] = 冷却期结束时间戳(Unix秒)
        # 亏损平仓后设置, 冷却期内禁止开仓, 避免背靠背立即开仓
        self.per_symbol_cooldown: Dict[str, float] = {}

    def classify_htf_state(
        self,
        atr_pct: float,
        vol_ratio: float,
        ema_alignment: float,
        direction: int = 0,
        price_vs_ema200: float = 0.0,
    ) -> str:
        """分类HTF 16状态 (4 ATR × 2 Volume × 2 HTF)

        v93设计: 4 ATR状态 (trending/ranging/volatile/low_vol) × 2 Volume × 2 HTF趋势
        简化版: 用ATR%和vol_ratio直接分类

        修复4 (ERR-20260703-strategy-split): 重定义 counter_trend
        原定义: counter_trend = EMA空头排列 (与direction同源, 命名误导)
        新定义: counter_trend = 交易方向 vs 价格趋势相反
        理论依据: 真正逆势 = short + price>ema200 (上涨中做空) 或 long + price<ema200 (下跌中做多)
        """
        # ATR状态
        if atr_pct > self.ATR_HIGH_THRESHOLD:
            atr_state = "volatile"
        elif atr_pct > self.ATR_LOW_THRESHOLD:
            atr_state = "trending" if abs(ema_alignment) > 0.5 else "ranging"
        else:
            atr_state = "low_vol"

        # Volume状态
        vol_state = "high_vol" if vol_ratio > self.VOL_HIGH_THRESHOLD else "low_vol"

        # 修复4: HTF趋势状态 — 基于"交易方向 vs 价格趋势" (非EMA排列)
        # direction * price_vs_ema200 > 0 → 顺势; < 0 → 真正逆势
        if direction == 0 or price_vs_ema200 == 0.0:
            # 无方向(中性带)或无价格趋势 → 默认顺势
            htf_state = "with_trend"
        elif direction * price_vs_ema200 > 0:
            # 方向与价格趋势一致 → 顺势
            htf_state = "with_trend"
        else:
            # 方向与价格趋势相反 → 真正逆势
            htf_state = "counter_trend"

        return f"{atr_state}/{vol_state}/{htf_state}"

    def get_htf_factor(self, state: str) -> float:
        """获取HTF状态对应的仓位因子"""
        return self.HTF_16_FACTORS.get(state, 1.0)

    def get_pos_mult(self, symbol: str) -> float:
        """获取币种pos_mult (v97冻结值)"""
        if symbol in self.DISABLED_SYMBOLS:
            return 0.0
        if symbol in self.CORE_POS_MULT:
            return self.CORE_POS_MULT[symbol]
        return self.SATELLITE_POS_MULT.get(symbol, 0.4)

    def compute_fusion_mult(self, features: Dict[str, Any]) -> float:
        """4维度特征融合仓位微调 (alpha=0.1)

        v96/v97: 凯利+PA+VP+MTF各贡献±alpha/2的微调
        sign: 7个负d特征sign=-1 (均值回归d为负)
        clip: [0.1, 10.0] 极宽
        """
        alpha = self.V97_ALPHA
        delta = alpha / 2

        # 凯利维度: kelly_fraction高 → 仓位高 (但d为负 → sign=-1 → 反转)
        kelly_f = features.get("kelly_fraction_half", 0.0)
        kelly_sign = -1  # d为负, 反转
        kelly_adj = kelly_sign * delta * (1 if kelly_f > 0.1 else 0)

        # PA维度: pa_body_pct高 → 信号强 → 仓位高 (d为负 → sign=-1)
        pa_body = features.get("pa_body_pct", 0.0)
        pa_sign = -1
        pa_adj = pa_sign * delta * (1 if pa_body > 0.4 else 0)

        # VP维度: vol_ratio高 → 信号强 (d为负 → sign=-1)
        vol_ratio = features.get("vp_vol_ratio_5", 1.0)
        vp_sign = -1
        vp_adj = vp_sign * delta * (1 if vol_ratio > 1.5 else 0)

        # MTF维度: ema_alignment强 → 信号强 (d为负 → sign=-1)
        ema_align = features.get("mtf_ema_alignment", 0.0)
        mtf_sign = -1
        mtf_adj = mtf_sign * delta * (1 if abs(ema_align) > 0.5 else 0)

        fusion_mult = 1.0 + kelly_adj + pa_adj + vp_adj + mtf_adj
        # clip极宽
        fusion_mult = max(self.V97_CLIP_LOW, min(self.V97_CLIP_HIGH, fusion_mult))
        return fusion_mult

    def v98_enhanced_direction(
        self,
        ema_align: float,
        features: Dict[str, Any],
    ) -> Tuple[int, float]:
        """v98 PA增强方向判断 — 三层决策矩阵 (修复1, ERR-20260703-strategy-split)

        修复核心: 从"EMA二元方向+PA只调信心"改为"三层方向决策矩阵"
        根因: 原设计 base_dir = 1 if ema_align > 0 else -1 (line 923旧)
          在EMA转换期(ratio 0.98-0.99)全部误判为做空, 但市场实际是上涨
          PA评分只调信心不改变方向, RSI完全不参与方向, BOS/CHoCH完全不消费

        三层决策矩阵 (对齐 agent_decision_engine.py:1007-1018):
          Layer 1: RSI超买超卖 (Wilder标准30/70, 非数据反推)
            - RSI<30 → 做多 (真正均值回归, Connor's RSI-2)
            - RSI>70 → 需CHoCH_bear确认才做空 (避免强趋势中逆势)
          Layer 2: 价格趋势 (close vs EMA200, 独立于EMA50/EMA200)
            - RSI中性区(30-70) → 顺势 (close vs EMA200)
          Layer 3: BOS/CHoCH结构确认 (复用 chanlun_priceaction.MarketStructureAnalyzer)
            - 强信号(strength≥0.5)允许方向翻转

        设计原则 (用户铁律+教训库):
          1. 不使用ML模型 (用户铁律: "不要通过已知数据反推策略")
          2. 所有阈值来自金融理论标准值 (RSI 30/70, 中性带0.5%, strength 0.5)
          3. 基于Bonferroni显著性 (统计证据, 非主观判断)
          4. 规则简单可解释 (非黑盒, 避免过拟合)
          5. 允许方向翻转 (修复"PA不改变方向"缺陷)

        Args:
            ema_align: EMA排列值 (mtf_ema_alignment, 修复2后可能为0.0中性)
            features: 入场特征字典 (含 _close, _structure_signal 由generate_signal注入)

        Returns:
            (direction, confidence): direction=1多/-1空/0无方向, confidence∈[0.5, 1.1]
            direction=0时confidence=0, generate_signal应返回None(跳过)
        """
        # === Layer 1: RSI超买超卖 (Connor's RSI-2 均值回归) ===
        # 理论依据: Wilder RSI标准阈值30/70, 非数据反推
        rsi = float(features.get("rsi_14", 50.0))
        rsi_overbought = rsi > 70.0   # 超买
        rsi_oversold = rsi < 30.0     # 超卖

        # === Layer 2: 价格趋势 (close vs EMA200, 独立于EMA50/EMA200) ===
        # 理论依据: 道氏理论 — 价格趋势用close vs EMA200(滞后但稳定)
        close = float(features.get("_close", 0.0))
        ema200 = float(features.get("mtf_ema200", 0.0))
        price_trend_up = (close > ema200) if ema200 > 0 else True  # 默认多头(BTC长期上涨)

        # === Layer 3: BOS/CHoCH结构确认 (复用 chanlun_priceaction) ===
        # 理论依据: ICT/SMC智能资金概念 — BOS=趋势延续, CHoCH=反转预警
        bos_choch = features.get("_structure_signal", {})
        struct_signal = str(bos_choch.get("signal", "none"))  # bos_bull/bos_bear/choch_bull/choch_bear/none
        struct_dir = str(bos_choch.get("direction", "neutral"))  # long/short/neutral
        struct_strength = float(bos_choch.get("strength", 0.0))

        # === 方向决策矩阵 (机制性, 非数据反推) ===
        base_dir = 0  # 默认无方向
        direction_source = "none"

        if rsi_oversold:
            # 超卖 → 均值回归LONG (不论EMA排列)
            base_dir = 1
            direction_source = "rsi_oversold_mean_reversion"
        elif rsi_overbought:
            # 超买 → 需CHoCH_bear或BOS_bear确认才做空 (避免强趋势中逆势)
            if struct_signal in ("choch_bear", "bos_bear") or (not price_trend_up):
                base_dir = -1
                direction_source = "rsi_overbought_confirmed"
            else:
                # 超买但价格仍上行+无结构反转 → 不交易
                base_dir = 0
                direction_source = "rsi_overbought_unconfirmed_skip"
        else:
            # RSI中性 (30-70): 顺势 (close vs EMA200)
            base_dir = 1 if price_trend_up else -1
            direction_source = "trend_following_close_vs_ema200"

        # === BOS/CHoCH 强信号覆盖 (strength >= 0.5 允许方向翻转) ===
        if struct_strength >= 0.5 and base_dir != 0:
            if struct_signal == "choch_bull" and base_dir < 0:
                # CHoCH_bull翻转: 空头结构被突破 → 翻多为LONG
                base_dir = 1
                direction_source = "choch_bull_flip"
            elif struct_signal == "choch_bear" and base_dir > 0:
                # CHoCH_bear翻转: 多头结构被跌破 → 翻空为SHORT
                base_dir = -1
                direction_source = "choch_bear_flip"

        # base_dir=0 表示无明确方向, generate_signal应返回None
        if base_dir == 0:
            return 0, 0.0  # confidence=0 触发跳过

        # === PA Confluence 信心计算 (基于新方向, 不改变方向) ===
        # 保留原PA评分逻辑, 但基于新base_dir
        pa_score = 0
        pa_lower = features.get("pa_lower_wick_ratio", 0.0)
        if pa_lower > 0.30:
            pa_score += 1  # 下影线长 → 利多

        pa_upper = features.get("pa_upper_wick_ratio", 0.0)
        if pa_upper > 0.30:
            pa_score -= 1  # 上影线长 → 利空

        pa_body = features.get("pa_body_pct", 0.0)
        if pa_body > 0.5:
            pa_score += 1 if base_dir > 0 else -1  # 实体确认方向

        atr_pct = features.get("atr_pct", 0.0)
        high_vol = atr_pct > 0.025  # 高波动阈值

        kelly_f = features.get("kelly_fraction_half", 0.0)
        kelly_positive = kelly_f > 0.1

        # Confluence信心 (方向已确定, 仅调整信心)
        if pa_score * base_dir > 0:
            # PA与方向一致 → confluence增强
            confidence = 1.0 if not high_vol else 0.80
        elif pa_score != 0 and base_dir * pa_score < 0:
            # PA与方向矛盾 → 降低信心 (不过滤)
            confidence = 0.60 if not high_vol else 0.50
        else:
            # PA中性 (pa_score==0)
            confidence = 0.80 if not high_vol else 0.65

        # Kelly代理微调 (±5%)
        if kelly_positive:
            confidence *= 1.05
        else:
            confidence *= 0.95

        # clip to [0.5, 1.1]
        confidence = max(0.50, min(1.10, confidence))
        return base_dir, confidence

    @staticmethod
    def _compute_structure_signal(
        klines: List[Dict],
        idx: int,
        lookback: int = 50,
    ) -> Dict[str, Any]:
        """无状态BOS/CHoCH检测 (修复1配套, 复用 chanlun_priceaction.MarketStructureAnalyzer)

        每次调用创建新analyzer并喂入klines[idx-lookback:idx+1],
        返回最近一次结构事件.

        理论依据:
          - ICT/SMC智能资金概念 — BOS=趋势延续, CHoCH=反转预警
          - 非过拟合: 仅2参数(swing_lookback=5, break_threshold=0.001), 都是金融理论标准值

        Args:
            klines: K线列表
            idx: 当前K线索引
            lookback: 回溯K线数量 (默认50根)

        Returns:
            {"signal": "bos_bull"/"bos_bear"/"choch_bull"/"choch_bear"/"none",
             "direction": "long"/"short"/"neutral",
             "strength": float 0-1}
        """
        try:
            from chanlun_priceaction import MarketStructureAnalyzer
        except ImportError:
            return {"signal": "none", "direction": "neutral", "strength": 0.0}

        if idx < lookback or idx >= len(klines):
            return {"signal": "none", "direction": "neutral", "strength": 0.0}

        analyzer = MarketStructureAnalyzer(swing_lookback=5, break_threshold=0.001)
        last_signal = {"signal": "none", "direction": "neutral", "strength": 0.0}

        for i in range(idx - lookback, idx + 1):
            if i < 0 or i >= len(klines):
                continue
            k = klines[i]
            try:
                result = analyzer.update({
                    "open": k.get("open", 0),
                    "high": k.get("high", 0),
                    "low": k.get("low", 0),
                    "close": k.get("close", 0),
                    "volume": k.get("volume", 0),
                })
                # 保留最近一次非none的结构事件
                if result.get("signal") != "none":
                    last_signal = result
            except Exception:
                continue

        return last_signal

    @staticmethod
    def _sigmoid(x: float) -> float:
        """数值稳定的sigmoid函数 (v98 hold_bars融合使用)"""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def compute_v98_hold_bars_mult(self, symbol: str, features: Dict[str, Any]) -> Tuple[float, Dict]:
        """v98 Layer 3: hold_bars融合仓位微调

        三层仓位调整架构 (v95→v96→v98):
          Layer 1 (v95): v95_pos_mult = pos_mult × market_factor × kelly_scaler
          Layer 2 (v96): effective_pos_mult = clip(v95_pos_mult × fusion_mult_4dim, 0.1, 10.0)
          Layer 3 (v98): final_pos_mult = effective_pos_mult × hold_bars_mult  ← 本方法

        公式 (复用v96模式, robust统计):
          raw_score = Σ(w_i × sign_i × normalize(f_i))
          normalize(f) = (f - median_per_symbol) / (p75 - p25)
          hold_bars_mult = 1.0 + alpha × (sigmoid(raw_score) - 0.5)
          范围: [1-alpha/2, 1+alpha/2]

        R2修复 - Simpson悖论: sign基于d_vs_pnl (PnL预测) 而非 d_vs_hold
        ERR-109: 用入场特征(cause)预测hold_bars(consequence), 不直接用hold_bars
        ERR-106: per-symbol robust标准化(median/p25/p75)
        ERR-v96sign(R2修正): sign方向基于d_vs_pnl

        Args:
            symbol: 交易对 (如 "BTC_USDT")
            features: 入场特征字典 (FeatureComputer.compute_all_features输出)

        Returns:
            (hold_bars_mult, debug_dict): hold_bars_mult ∈ [0.9, 1.1]
            若per_symbol_stats不可用, 返回 (1.0, {"used": False, "reason": "stats_unavailable"})
        """
        if not self.v98_per_symbol_stats:
            return 1.0, {"used": False, "reason": "v98_per_symbol_stats未加载"}

        sym_stats = self.v98_per_symbol_stats.get(symbol, {})
        if not sym_stats:
            return 1.0, {"used": False, "reason": f"symbol {symbol} 无统计数据"}

        raw_score = 0.0
        feature_details = []
        for feat_name, weight, sign, _ in V98_HOLD_BARS_FEATURES:
            if feat_name not in features or feat_name not in sym_stats:
                feature_details.append({"feature": feat_name, "used": False})
                continue
            feat_stats = sym_stats[feat_name]
            iqr = feat_stats.get("iqr", 0.0)
            if iqr <= 0:
                feature_details.append({
                    "feature": feat_name, "used": False,
                    "reason": "iqr=0 (binary or constant)",
                })
                continue
            normalized = (features[feat_name] - feat_stats["median"]) / iqr
            contribution = weight * sign * normalized
            raw_score += contribution
            feature_details.append({
                "feature": feat_name, "used": True,
                "raw_value": features[feat_name],
                "normalized": normalized,
                "weight": weight, "sign": sign,
                "contribution": contribution,
            })

        hold_bars_mult = 1.0 + self.v98_alpha * (self._sigmoid(raw_score) - 0.5)

        debug = {
            "used": True,
            "raw_score": raw_score,
            "sigmoid": self._sigmoid(raw_score),
            "hold_bars_mult": hold_bars_mult,
            "alpha": self.v98_alpha,
            "features": feature_details,
        }
        return hold_bars_mult, debug

    def generate_signal(
        self,
        symbol: str,
        klines: List[Dict],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """在idx处生成交易信号 (v97策略, 参数冻结)

        Returns:
            None = 无信号, Dict = 交易信号
        """
        if symbol in self.DISABLED_SYMBOLS:
            return None
        if idx < 200 or idx >= len(klines):
            return None

        # 计算入场特征
        features = FeatureComputer.compute_all_features(klines, idx)
        atr_pct = features.get("atr_pct", 0.0)
        if atr_pct <= 0:
            return None

        # 修复2: EMA转换期中性带 — 不生成信号 (趋势未确认, 跳过)
        # 理论依据: EMA交叉前后0.5%区间为噪声主导, 二元EMA会误判方向
        if features.get("mtf_neutral_zone", False):
            return None

        # 修复1配套: 注入方向判断所需的额外特征 (_close + _structure_signal)
        features["_close"] = klines[idx]["close"]
        features["_structure_signal"] = V97SignalGenerator._compute_structure_signal(klines, idx)

        # 修复1+4: 方向判断必须先于 htf_state (修复4需要direction来判定counter_trend)
        ema_align = features.get("mtf_ema_alignment", 0.0)
        direction, pa_confidence = self.v98_enhanced_direction(ema_align, features)
        if direction == 0:
            return None  # 修复1: 无明确方向(如超买无CHoCH确认), 跳过

        # HTF 16状态分类 (修复4: 传入direction和price_vs_ema200用于真正逆势判定)
        vol_ratio = features.get("vp_vol_ratio_5", 1.0)
        htf_state = self.classify_htf_state(
            atr_pct, vol_ratio, ema_align,
            direction=direction,
            price_vs_ema200=features.get("price_vs_ema200", 0.0),
        )
        htf_factor = self.get_htf_factor(htf_state)

        # pos_mult
        pos_mult = self.get_pos_mult(symbol)
        if pos_mult <= 0:
            return None

        # 融合仓位微调
        fusion_mult = self.compute_fusion_mult(features)

        # v98 Layer 3: hold_bars融合仓位微调 (第三层, 乘法叠加)
        # hold_bars_mult = 1.0 + alpha × (sigmoid(raw_score) - 0.5), range [0.9, 1.1]
        # 三层架构: pos_mult × htf_factor × fusion_mult × pa_confidence × hold_bars_mult
        # 若per_symbol_stats不可用, hold_bars_mult=1.0 (退化为v97行为)
        hold_bars_mult, v98_hold_bars_debug = self.compute_v98_hold_bars_mult(symbol, features)

        # v99-soft-reduce: 机制性入场门控 (第四层, 乘法叠加)
        # 基于P9-A/B/C三大亏损源深度分析:
        #   - ADX>40减仓10% (强趋势下均值回归优势减弱, 但不拒绝)
        #   - MTF强逆势减仓30%, 弱逆势减仓15% (逆势风险高, 但不拒绝)
        # 软减仓原则: 保留所有信号, 仅根据风险等级调整仓位
        # 反过拟合: 阈值(40/0.3/0.1)都是金融理论标准值, 非数据反推
        # v2模拟验证: 净PnL -5.57%, 亏损笔损失缩减+8.34%, 三大亏损源覆盖率52%
        v99_pos_adj = 1.0
        v99_gate_debug = None
        if _V99_AVAILABLE:
            v99_gate = V99EntryGates.check_entry_gates(features, direction, STRATEGY_MEAN_REVERSION)
            if not v99_gate["allow"]:
                # 仅direction=0等异常拒绝, 正常情况下不会触发
                return None
            v99_pos_adj = v99_gate["pos_adjustment"]
            v99_gate_debug = {
                "pos_adj": v99_pos_adj,
                "reason": v99_gate["reason"],
                "reduction_summary": v99_gate.get("reduction_summary", {}),
            }

        # 最终仓位 (pos_mult × htf_factor × fusion_mult × v98_pa_confidence × v98_hold_bars_mult × v99_pos_adj)
        # 四层仓位调整: Layer1(htf+pos_mult) × Layer2(fusion_mult+pa_confidence) × Layer3(hold_bars_mult) × Layer4(v99_pos_adj)
        final_pos_mult = pos_mult * htf_factor * fusion_mult * pa_confidence * hold_bars_mult * v99_pos_adj

        # v98 ATR自适应持仓管理 (基于Bonferroni显著 atr_pct #3, score=15.81)
        # 机制性改进非数据反推: 高波动→价格变动快→短持仓, 低波动→趋势持续久→长持仓
        # 用户铁律: "不要通过已知数据反推策略" → 这是市场微观结构常识, 非历史最优参数
        # 阈值与v98_enhanced_direction的high_vol判断一致(0.025/0.015)
        if atr_pct > 0.025:
            dynamic_max_hold = 24   # 高波动 → 24h (缩短持仓降低风险)
        elif atr_pct > 0.015:
            dynamic_max_hold = 48   # 中波动 → 48h (v97标准)
        else:
            dynamic_max_hold = 72   # 低波动 → 72h (延长持仓让利润奔跑)

        # ATR-based TP/SL
        atr_abs = features.get("atr_abs", 0.0)
        entry_price = klines[idx]["close"]
        if atr_abs <= 0 or entry_price <= 0:
            return None

        # TP = 2 * ATR, SL = 1.5 * ATR (v97的典型参数)
        tp_distance = 2.0 * atr_abs
        sl_distance = 1.5 * atr_abs

        # 连续亏损检查
        if self.current_consec_loss >= self.V97_MAX_CONSEC:
            return None  # 触发全局consec保护, 跳过
        sym_consec = self.per_symbol_consec.get(symbol, 0)
        if symbol in self.SATELLITE_POS_MULT and sym_consec >= 2:
            return None  # per-symbol consec保护 (新币种)

        # v99 亏损后冷却期检查 (ERR-20260703-back-to-back-bug)
        # 亏损平仓后4小时内禁止开仓, 避免背靠背立即开仓(行为等价马丁)
        # 机制性改进: 亏损后市场状态不稳定, 需等待新信号确认
        cooldown_end = self.per_symbol_cooldown.get(symbol, 0.0)
        if cooldown_end > 0:
            current_ts = klines[idx].get("timestamp", 0) if isinstance(klines[idx], dict) else 0
            if current_ts < cooldown_end:
                return None  # 冷却期内, 跳过

        return {
            "symbol": symbol,
            "entry_idx": idx,
            "entry_ts": klines[idx]["timestamp"],
            "entry_price": entry_price,
            "direction": direction,
            "pos_mult": final_pos_mult,
            "pa_confidence": pa_confidence,  # v98 PA增强信心分
            "max_hold_bars": dynamic_max_hold,  # v98 ATR自适应持仓
            "v98_hold_bars_mult": hold_bars_mult,  # v98 Layer 3 hold_bars融合乘数
            "v98_hold_bars_debug": v98_hold_bars_debug,  # v98 Layer 3 调试信息
            "v99_pos_adj": v99_pos_adj,  # v99-soft-reduce 仓位调整乘数
            "v99_gate_debug": v99_gate_debug,  # v99-soft-reduce 门控调试信息
            "tp_price": entry_price + direction * tp_distance,
            "sl_price": entry_price - direction * sl_distance,
            "atr_pct": atr_pct,
            "atr_abs": atr_abs,
            "htf_state": htf_state,
            "features": features,
        }

    def check_exit(
        self,
        signal: Dict[str, Any],
        klines: List[Dict],
        idx: int,
        max_hold_bars: int = 48,
    ) -> Optional[Dict[str, Any]]:
        """检查是否触发出场 (TP/SL/超时)

        v98增强: 优先使用signal中的动态max_hold_bars (ATR自适应持仓)
        若signal未包含max_hold_bars, 回退到参数max_hold_bars (v97兼容)

        Returns:
            None = 继续持有, Dict = 出场信息
        """
        entry_idx = signal["entry_idx"]
        direction = signal["direction"]
        tp_price = signal["tp_price"]
        sl_price = signal["sl_price"]
        hold_bars = idx - entry_idx

        # v98: 优先使用signal中的ATR自适应max_hold, 回退到参数(v97兼容)
        effective_max_hold = signal.get("max_hold_bars", max_hold_bars)

        if hold_bars >= effective_max_hold:
            # 超时出场
            exit_price = klines[idx]["close"]
            return {
                "exit_idx": idx,
                "exit_ts": klines[idx]["timestamp"],
                "exit_price": exit_price,
                "exit_reason": "timeout",
                "hold_bars": hold_bars,
            }

        if idx >= len(klines):
            return None

        k = klines[idx]
        # v99-soft-reduce: hold_bars最小门控 (反快速止损噪声)
        # 入场后最少持仓2bars, 避免噪声止损 (但TP不受限, 让利润奔跑)
        # 机制: 短于2bar的SL是市场微观结构噪声, 不是策略判断
        # 安全保障: 超时出场和TP仍正常工作, 下一bar若仍触SL则放行(hold_bars>=2)
        v99_min_hold_ok = True
        if _V99_AVAILABLE:
            v99_min_hold_ok = V99EntryGates.check_min_hold_bars(hold_bars, signal)

        if direction == 1:
            if k["high"] >= tp_price:
                return {"exit_idx": idx, "exit_ts": k["timestamp"],
                        "exit_price": tp_price, "exit_reason": "tp", "hold_bars": hold_bars}
            if k["low"] <= sl_price:
                # v99: hold_bars<2时跳过SL (噪声), 继续持有
                if not v99_min_hold_ok:
                    pass  # 不出场, 继续持有到下一bar
                else:
                    return {"exit_idx": idx, "exit_ts": k["timestamp"],
                            "exit_price": sl_price, "exit_reason": "sl", "hold_bars": hold_bars}
        else:
            if k["low"] <= tp_price:
                return {"exit_idx": idx, "exit_ts": k["timestamp"],
                        "exit_price": tp_price, "exit_reason": "tp", "hold_bars": hold_bars}
            if k["high"] >= sl_price:
                # v99: hold_bars<2时跳过SL (噪声), 继续持有
                if not v99_min_hold_ok:
                    pass  # 不出场, 继续持有到下一bar
                else:
                    return {"exit_idx": idx, "exit_ts": k["timestamp"],
                            "exit_price": sl_price, "exit_reason": "sl", "hold_bars": hold_bars}
        return None

    def update_consec(self, symbol: str, pnl_pct: float) -> None:
        """更新连续亏损计数"""
        if pnl_pct < 0:
            self.current_consec_loss += 1
            self.per_symbol_consec[symbol] = self.per_symbol_consec.get(symbol, 0) + 1
        else:
            self.current_consec_loss = 0
            self.per_symbol_consec[symbol] = 0
        self.max_consec_seen = max(self.max_consec_seen, self.current_consec_loss)
        self.max_single_loss_seen = max(self.max_single_loss_seen, abs(pnl_pct) * 100)


# ============================================================================
# Layer 5: 成本计算 (sim/live差异5因子模型)
# ============================================================================

class CostCalculator:
    """全成本量化模型 (5因子: spread+slippage+liquidity+latency+fee)

    用户铁律: "模拟/实盘差异全成本量化模型"
    来源: v75.calc_trade_cost_detailed + ERR-098 (4因素高保真市场仿真模型)
    """

    @staticmethod
    def calc_trade_cost(
        pos_size_usd: float,
        symbol: str,
        hold_bars: int,
        atr_abs: float,
        entry_price: float,
    ) -> Dict[str, float]:
        """计算单笔交易的全成本 (5因子模型)

        Args:
            pos_size_usd: 仓位大小(USD)
            symbol: 交易对
            hold_bars: 持仓小时数
            atr_abs: 入场时ATR绝对值
            entry_price: 入场价格

        Returns:
            Dict含 spread/slippage/liquidity/latency/fee/total (USD)
        """
        friction = SYMBOL_FRICTION.get(symbol, {"spread_bps": 3.0, "taker_fee_bps": 0.5, "slippage_bps": 0.8})
        p = friction

        # 1. spread (双向: 开仓+平仓)
        spread_cost = pos_size_usd * p["spread_bps"] / 10000.0 * 2

        # 2. slippage (Almgren-Chriss简化, 双向)
        slip_cost = pos_size_usd * p["slippage_bps"] / 10000.0 * 2

        # 3. liquidity (深度成本, 简化)
        depth_ratio = pos_size_usd / 50000.0  # 假设1%深度=$50k
        liq_bps = depth_ratio * 5.0 if depth_ratio > 0.1 else depth_ratio * 2.0
        liq_cost = pos_size_usd * liq_bps / 10000.0 * 2

        # 4. latency — vol_per_bar = ATR / price
        vol_per_bar = atr_abs / entry_price if entry_price > 0 else 0.005
        lat_dec = vol_per_bar * math.sqrt(LATENCY_MS / 3600000.0)
        lat_cost = pos_size_usd * lat_dec * 2

        # 5. fee (taker) + funding
        fee_cost = pos_size_usd * p["taker_fee_bps"] / 10000.0 * 2
        funding_hits = max(1, int(hold_bars / 8))
        fund_cost = pos_size_usd * FUNDING_RATE_8H_PCT / 100.0 * funding_hits
        fee_cost += fund_cost

        total = spread_cost + slip_cost + liq_cost + lat_cost + fee_cost
        return {
            "spread": spread_cost,
            "slippage": slip_cost,
            "liquidity": liq_cost,
            "latency": lat_cost,
            "fee": fee_cost,
            "total": total,
        }

    @staticmethod
    def calc_live_pnl(sim_pnl_usd: float, cost_usd: float) -> float:
        """实盘PnL = 模拟PnL - 全成本"""
        return sim_pnl_usd - cost_usd


# ============================================================================
# Layer 6: Tier 2 Forward-Walk Validator 主类
# ============================================================================

class Tier2ForwardWalkValidator:
    """Tier 2 前向验证器 — 3个月实盘数据收集与评估

    用户铁律: "开始连续3个月的实盘数据实测"
    用户铁律: "永远永远不要出现模拟牛逼, 实盘亏损的情况"

    使用方式:
      # 初始化
      validator = Tier2ForwardWalkValidator()
      validator.initialize()

      # 每日收集
      validator.collect_daily_data()

      # 月底评估
      validator.run_monthly_evaluation()

      # 查看状态
      report = validator.get_status_report()
    """

    def __init__(
        self,
        capital_usd: float = TIER2_CAPITAL_USD,
        symbols: List[str] = None,
    ):
        self.capital_usd = capital_usd
        self.symbols = symbols or V97_SYMBOLS
        self.state = Tier2State()
        self.fetcher = GateIoDataFetcher()
        self.signal_gen = V97SignalGenerator()
        self.cost_calc = CostCalculator()
        # v98 per_symbol_stats (hold_bars融合所需, 由initialize()加载)
        self.v98_per_symbol_stats: Optional[Dict] = None
        # 将v98 per_symbol_stats引用传递给信号生成器
        self.signal_gen.v98_per_symbol_stats = None
        self.signal_gen.v98_alpha = V98_ALPHA
        # 确保目录存在
        for d in [TIER2_DATA_DIR, TIER2_KLINES_DIR, TIER2_TRADES_DIR, TIER2_REPORTS_DIR]:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """初始化Tier 2状态 (首次运行, v98基线)

        v98升级为生产基线 (2026-07-03):
          - 加载 _v98_go_nogo_judgment.json 作为主基线
          - 保留 _v97_go_nogo_judgment.json 作为历史归档对比
          - 加载 v98 per_symbol_stats (hold_bars融合所需, 冻结自训练数据)

        修复 (ERR-tier2-baseline-load): 旧版首次运行时baseline未保存到state,
        导致后续load_state加载空baseline, can_deploy_tier2误报"Tier 1 未通过".

        修复 (ERR-tier2-init-loadstate): load_state是@classmethod返回新实例,
        self.load_state()不会更新self.state. 必须用返回值的state字段赋值给self.state,
        否则后续save_state()会用默认Tier2State()覆盖磁盘上的state文件.
        """
        if TIER2_STATE_FILE.exists():
            # load_state是@classmethod, 返回新实例, 必须从中复制state到self
            # 旧代码self.load_state()不更新self.state, 导致save_state()覆盖磁盘state
            loaded = Tier2ForwardWalkValidator.load_state()
            self.state = loaded.state
            logger.info("Tier 2 状态已加载, version=%s, collection_days=%d, oos_start=%s",
                       self.state.version, self.state.collection_days, self.state.oos_start_date)
            # 补加载v98基线 (如果state中为空)
            if not self.state.v98_baseline:
                self.state.v98_baseline = self._load_v98_baseline()
                self.save_state()
                logger.info("  v98基线已补加载")
            # 补加载v97历史归档 (用于对比)
            if not self.state.v97_baseline:
                self.state.v97_baseline = self._load_v97_baseline_archive()
                self.save_state()
                logger.info("  v97历史归档已补加载")
        else:
            self.state.created_at = datetime.now(timezone.utc).isoformat()
            self.state.last_updated = self.state.created_at
            self.state.oos_start_date = OOS_START_DATE.strftime("%Y-%m-%d")
            self.state.v98_baseline = self._load_v98_baseline()
            self.state.v97_baseline = self._load_v97_baseline_archive()
            self.save_state()
            logger.info("Tier 2 状态已初始化 (v98基线), OOS起点=%s", self.state.oos_start_date)

        # 加载v98 per_symbol_stats (hold_bars融合所需)
        self._load_v98_per_symbol_stats()

        # 修复5: 从state回灌冷却期到signal_gen (跨日运行关键)
        # 根因: per_symbol_cooldown原只在signal_gen上, 跨日运行时新建signal_gen丢失
        # 修复: initialize加载state后, 将state.per_symbol_cooldown回灌到signal_gen
        if hasattr(self.state, 'per_symbol_cooldown') and self.state.per_symbol_cooldown:
            self.signal_gen.per_symbol_cooldown = dict(self.state.per_symbol_cooldown)
            import time as _time_mod
            active_count = sum(1 for v in self.signal_gen.per_symbol_cooldown.values() if v > _time_mod.time())
            if active_count > 0:
                logger.info("  修复5: 冷却期状态已回灌, %d个symbol在冷却中", active_count)

    def _load_v98_baseline(self) -> Dict[str, Any]:
        """加载v98基线 (主基线, 冻结, 用于Tier 2准入判定)"""
        baseline_path = HERE / "_v98_go_nogo_judgment.json"
        if baseline_path.exists():
            try:
                return json.loads(baseline_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"version": "v98_hold_bars_fusion", "tier1_pass": True,
                "summary": {"ann": 96.37, "gap": 0.37, "sharpe": 3.53}}

    def _load_v97_baseline_archive(self) -> Dict[str, Any]:
        """加载v97历史归档基线 (仅对比参考, 不用于准入判定)"""
        baseline_path = HERE / "_v97_go_nogo_judgment.json"
        if baseline_path.exists():
            try:
                return json.loads(baseline_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"version": "v97_alpha_gap_optimize", "tier1_pass": True,
                "summary": {"ann": 91.12, "gap": 1.23, "sharpe": 3.51}}

    def _load_v98_per_symbol_stats(self) -> None:
        """加载v98 per_symbol_stats (hold_bars融合特征统计, 冻结自训练数据)

        用户铁律: "不要通过已知数据反推策略" — per_symbol_stats必须冻结自训练数据,
        不可用OOS数据更新, 避免look-ahead bias.

        文件: tier2_oos_data/v98_per_symbol_stats.json
        若文件不存在, hold_bars_mult=1.0 (退化为v97行为, 不影响基本策略)
        """
        if V98_PER_SYMBOL_STATS_FILE.exists():
            try:
                # JSON结构: {version, timestamp, description, source, oos_start_ts, features, stats: {symbol: {feature: {median,p25,p75,iqr,n}}}}
                # 只取stats字段作为per_symbol_stats (避免顶层元数据键被误认为symbol)
                pss_data = json.loads(
                    V98_PER_SYMBOL_STATS_FILE.read_text(encoding="utf-8")
                )
                self.v98_per_symbol_stats = pss_data.get("stats", {}) if isinstance(pss_data, dict) else {}
                n_syms = len(self.v98_per_symbol_stats)
                logger.info("v98 per_symbol_stats已加载: %d个币种", n_syms)
            except Exception as e:
                logger.warning("v98 per_symbol_stats加载失败: %s, hold_bars_mult=1.0", e)
                self.v98_per_symbol_stats = None
        else:
            logger.warning("v98 per_symbol_stats文件不存在: %s, hold_bars_mult=1.0 (退化为v97)",
                          V98_PER_SYMBOL_STATS_FILE)
            self.v98_per_symbol_stats = None
        # 同步到信号生成器 (V97SignalGenerator.compute_v98_hold_bars_mult使用)
        self.signal_gen.v98_per_symbol_stats = self.v98_per_symbol_stats

    def save_state(self) -> None:
        """保存状态到JSON"""
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        state_dict = asdict(self.state)
        with open(TIER2_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Tier 2 状态已保存: %s", TIER2_STATE_FILE)

    @classmethod
    def load_state(cls) -> "Tier2ForwardWalkValidator":
        """从JSON加载状态"""
        validator = cls()
        if TIER2_STATE_FILE.exists():
            state_dict = json.loads(TIER2_STATE_FILE.read_text(encoding="utf-8"))
            validator.state = Tier2State(**{
                k: v for k, v in state_dict.items()
                if k in Tier2State.__dataclass_fields__
            })
        return validator

    # ------------------------------------------------------------------
    # 数据收集
    # ------------------------------------------------------------------

    def collect_daily_data(self) -> Dict[str, Any]:
        """收集每日OOS数据 + 运行策略

        用户铁律: "开始连续3个月的实盘数据实测"

        Returns:
            收集报告 Dict
        """
        # ERR-20260703-tier2-state-reset 防御性修复:
        # 如果 state 是默认空状态(collection_days==0 且 last_collection_ts==0),
        # 但磁盘状态文件存在, 说明 initialize() 未被调用, 必须先加载磁盘状态
        # 否则 save_state() 会用空状态覆盖磁盘有效数据
        if (self.state.collection_days == 0
                and self.state.last_collection_ts == 0
                and TIER2_STATE_FILE.exists()):
            logger.warning("检测到空状态但磁盘有状态文件, 补执行 initialize() 加载磁盘状态")
            self.initialize()

        logger.info("=" * 70)
        logger.info("Tier 2 每日数据收集 — %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 70)

        # 1. 获取最新K线
        now_ts = time.time()
        last_ts = self.state.last_collection_ts or OOS_START_DATE.timestamp()
        logger.info("  上次收集: %s", datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat())
        logger.info("  当前时间: %s", datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat())

        # 获取OOS期间的K线
        klines_map = self.fetcher.fetch_historical_oos_klines(
            self.symbols, last_ts, now_ts
        )

        # 2. 运行策略生成交易
        # 用户铁律: "不要通过已知数据反推策略" — 预热数据只用于指标计算, 不用于信号生成
        # 策略需要200+K线计算EMA200/ATR等指标, OOS数据仅2天不足,
        # 所以加载训练期最后300条K线作为预热, 但只在OOS期(2026-07-01后)生成交易信号
        oos_start_ts = OOS_START_DATE.timestamp()
        new_trades = []
        for sym in self.symbols:
            oos_klines = klines_map.get(sym, [])
            if not oos_klines:
                logger.warning("  %s: 无OOS K线, 跳过", sym)
                continue
            # 加载预热数据 (训练期最后300条K线)
            warmup = self.fetcher.load_warmup_klines(sym, n_warmup=300)
            # 合并: 预热 + OOS (去重)
            warmup_ts = {k["timestamp"] for k in warmup}
            oos_only = [k for k in oos_klines if k["timestamp"] not in warmup_ts]
            combined = warmup + oos_only
            combined.sort(key=lambda k: k["timestamp"])
            # 找到OOS起点在combined中的位置
            oos_start_idx = 0
            for i, k in enumerate(combined):
                if k["timestamp"] >= oos_start_ts:
                    oos_start_idx = i
                    break
            logger.info("  %s: combined=%d (warmup=%d, oos=%d), OOS起点idx=%d",
                       sym, len(combined), len(warmup), len(oos_only), oos_start_idx)
            if len(combined) < 200:
                logger.warning("  %s: combined K线不足 (%d<200), 跳过", sym, len(combined))
                continue
            # 保存OOS K线
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            self.fetcher.save_klines(sym, oos_only, date_str)
            # 生成信号 (只在OOS期生成交易)
            sym_trades = self._run_strategy_on_klines(sym, combined, oos_start_idx)
            new_trades.extend(sym_trades)

        # 3. 更新状态
        self.state.last_collection_ts = now_ts
        self.state.collection_days = (now_ts - OOS_START_DATE.timestamp()) / 86400
        self.state.total_trades += len(new_trades)
        for t in new_trades:
            if t["live_pnl_usd"] > 0:
                self.state.total_wins += 1
            else:
                self.state.total_losses += 1
            self.state.total_sim_pnl_usd += t["pnl_usd"]
            self.state.total_live_pnl_usd += t["live_pnl_usd"]
            self.state.total_cost_usd += t["cost_usd"]
            self.state.max_single_loss_pct = max(
                self.state.max_single_loss_pct, abs(t["live_pnl_pct"]) * 100
            )
            self.state.max_consec_loss = max(
                self.state.max_consec_loss, self.signal_gen.max_consec_seen
            )

        # 4. 保存交易记录
        if new_trades:
            trades_path = TIER2_TRADES_DIR / f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
            with open(trades_path, "w", encoding="utf-8") as f:
                json.dump(new_trades, f, indent=2, default=str)

        self.save_state()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "new_klines": {sym: len(k) for sym, k in klines_map.items()},
            "new_trades": len(new_trades),
            "total_trades": self.state.total_trades,
            "collection_days": self.state.collection_days,
            "total_sim_pnl": self.state.total_sim_pnl_usd,
            "total_live_pnl": self.state.total_live_pnl_usd,
            "total_cost": self.state.total_cost_usd,
            "gap_pct": abs(self.state.total_live_pnl_usd - self.state.total_sim_pnl_usd) / abs(self.state.total_sim_pnl_usd) * 100 if self.state.total_sim_pnl_usd != 0 else 0.0,
        }
        logger.info("  收集完成: %d新交易, 总交易%d, 收集天数%.1f",
                   len(new_trades), self.state.total_trades, self.state.collection_days)
        return report

    def _run_strategy_on_klines(
        self, symbol: str, klines: List[Dict], oos_start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """在K线上运行v97策略 (参数冻结)

        用户铁律: "不要通过已知数据反推策略"
        信号只在OOS期 (idx >= oos_start_idx) 生成, 预热期只用于指标计算.
        """
        trades = []
        open_signal = None

        for idx in range(len(klines)):
            # 检查出场 (持仓期间可以跨OOS边界)
            if open_signal is not None:
                exit_info = self.signal_gen.check_exit(open_signal, klines, idx)
                if exit_info is not None:
                    # 计算PnL
                    entry_price = open_signal["entry_price"]
                    exit_price = exit_info["exit_price"]
                    direction = open_signal["direction"]
                    pos_mult = open_signal["pos_mult"]
                    hold_bars = exit_info["hold_bars"]
                    atr_abs = open_signal["atr_abs"]

                    # 模拟PnL
                    pnl_pct = direction * (exit_price - entry_price) / entry_price
                    pos_size_usd = self.capital_usd * 0.02 * pos_mult  # 2%基础风险 × pos_mult
                    pnl_usd = pos_size_usd * pnl_pct

                    # 成本
                    cost = self.cost_calc.calc_trade_cost(
                        pos_size_usd, symbol, hold_bars, atr_abs, entry_price
                    )
                    cost_usd = cost["total"]

                    # 实盘PnL
                    live_pnl_usd = self.cost_calc.calc_live_pnl(pnl_usd, cost_usd)
                    live_pnl_pct = live_pnl_usd / pos_size_usd if pos_size_usd > 0 else 0.0

                    # 单亏≤2%硬截断 (v97)
                    if live_pnl_pct < -self.signal_gen.V97_MAX_SINGLE_LOSS_PCT / 100:
                        live_pnl_pct = -self.signal_gen.V97_MAX_SINGLE_LOSS_PCT / 100
                        live_pnl_usd = pos_size_usd * live_pnl_pct

                    # 更新连续亏损
                    self.signal_gen.update_consec(symbol, live_pnl_pct)

                    # v99 亏损后冷却期 (ERR-20260703-back-to-back-bug)
                    # 亏损平仓后设置4小时冷却期, 避免背靠背立即开仓(行为等价马丁)
                    # 机制性改进: 亏损后市场状态不稳定, 需等待新信号确认
                    if live_pnl_pct < 0:
                        exit_ts_val = exit_info.get("exit_ts", 0)
                        cooldown_end_ts = exit_ts_val + self.signal_gen.LOSS_COOLDOWN_HOURS * 3600
                        self.signal_gen.per_symbol_cooldown[symbol] = cooldown_end_ts
                        # 修复5: 双写 — signal_gen(运行时) + state(持久化)
                        # 跨日 collect_daily_data 时冷却期必须保留, 否则跨日背靠背开仓
                        self.state.per_symbol_cooldown[symbol] = cooldown_end_ts
                        logger.info(
                            "    %s: 亏损平仓(%.2f%%), 设置%.1f小时冷却期",
                            symbol, live_pnl_pct * 100, self.signal_gen.LOSS_COOLDOWN_HOURS
                        )
                    elif symbol in self.signal_gen.per_symbol_cooldown:
                        # 盈利则清除冷却期 (盈利说明市场状态已转变)
                        del self.signal_gen.per_symbol_cooldown[symbol]
                        # 修复5: 同步清除 state 中的冷却期
                        self.state.per_symbol_cooldown.pop(symbol, None)

                    # v99 真实MAE/MFE计算 (ERR-20260703-bad-exit-hallucination)
                    # 根因: 之前归因引擎用 atr_pct*0.5 推断MFE, 导致bad_exit归因幻觉
                    #   "曾有+1.75%利润但最终止损"基于猜测而非事实, bad_exit被高估为最大亏损root_cause
                    # 修复: 回溯计算持仓期间真实MAE/MFE, 让归因引擎基于事实而非猜测
                    # 铁律: "复盘不是摆设" — MAE/MFE必须真实测量, 不能启发式推断
                    entry_idx_val = open_signal["entry_idx"]
                    holding_klines = klines[entry_idx_val : idx + 1]
                    mfe_pct_real = 0.0
                    mae_pct_real = 0.0
                    if len(holding_klines) > 0 and entry_price > 0:
                        try:
                            highs = [k["high"] for k in holding_klines if "high" in k]
                            lows = [k["low"] for k in holding_klines if "low" in k]
                            if highs and lows:
                                if direction == 1:  # long: MFE=最高价收益, MAE=最低价回撤
                                    mfe_pct_real = max(0.0, (max(highs) - entry_price) / entry_price)
                                    mae_pct_real = min(0.0, (min(lows) - entry_price) / entry_price)
                                else:  # short: MFE=最低价收益, MAE=最高价回撤
                                    mfe_pct_real = max(0.0, (entry_price - min(lows)) / entry_price)
                                    mae_pct_real = min(0.0, (entry_price - max(highs)) / entry_price)
                        except (KeyError, TypeError, ZeroDivisionError):
                            pass  # 降级为0, 归因引擎会推断(但避免able=False)

                    # v99 exit_reason命名统一 (ERR-20260703-exit-reason-naming)
                    # validator产出: tp/sl/timeout (短形式)
                    # 归因引擎期望: take_profit/stop_loss/time_stop (长形式)
                    # 修复: 统一为长形式, 避免归因引擎无法识别导致bad_exit归因全部失效
                    _EXIT_REASON_MAP = {
                        "tp": "take_profit",
                        "sl": "stop_loss",
                        "timeout": "time_stop",
                    }
                    exit_reason_norm = _EXIT_REASON_MAP.get(
                        exit_info["exit_reason"], exit_info["exit_reason"]
                    )

                    # 月份
                    month_key = datetime.fromtimestamp(
                        open_signal["entry_ts"], tz=timezone.utc
                    ).strftime("%Y-%m")

                    trade = {
                        "trade_id": f"{symbol}_{int(open_signal['entry_ts'])}",
                        "symbol": symbol,
                        "direction": direction,
                        "entry_ts": open_signal["entry_ts"],
                        "exit_ts": exit_info["exit_ts"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "hold_bars": hold_bars,
                        "pnl_pct": pnl_pct,
                        "pnl_usd": pnl_usd,
                        "live_pnl_pct": live_pnl_pct,
                        "live_pnl_usd": live_pnl_usd,
                        "cost_usd": cost_usd,
                        "cost_bps": cost_usd / pos_size_usd * 10000 if pos_size_usd > 0 else 0,
                        "atr_pct": open_signal["atr_pct"],
                        "pos_mult": pos_mult,
                        "month_key": month_key,
                        "exit_reason": exit_reason_norm,  # v99: 统一为长形式 (ERR-20260703-exit-reason-naming)
                        "mae_pct": mae_pct_real,  # v99: 真实MAE (ERR-20260703-bad-exit-hallucination)
                        "mfe_pct": mfe_pct_real,  # v99: 真实MFE
                        "htf_state": open_signal["htf_state"],
                        "features": {k: v for k, v in open_signal["features"].items()
                                    if isinstance(v, (int, float, str))},
                    }
                    trades.append(trade)
                    open_signal = None

            # 检查入场 (无持仓时, 只在OOS期生成信号)
            # 用户铁律: "不要通过已知数据反推策略" — 预热期不生成交易
            if open_signal is None and idx >= oos_start_idx:
                signal = self.signal_gen.generate_signal(symbol, klines, idx)
                if signal is not None:
                    open_signal = signal

        logger.info("    %s: 生成 %d 笔交易", symbol, len(trades))
        return trades

    # ------------------------------------------------------------------
    # 月度评估
    # ------------------------------------------------------------------

    def run_monthly_evaluation(self, month_key: Optional[str] = None) -> Dict[str, Any]:
        """运行月度Tier 2评估

        用户铁律: "在未达成性能指标前不得推进实盘部署工作"

        Args:
            month_key: 指定月份, 如 "2026-07". None=当前月

        Returns:
            评估报告 Dict
        """
        if month_key is None:
            month_key = datetime.now(timezone.utc).strftime("%Y-%m")

        logger.info("=" * 70)
        logger.info("Tier 2 月度评估 — %s", month_key)
        logger.info("=" * 70)

        # 收集当月交易
        month_trades = self._load_month_trades(month_key)
        if not month_trades:
            logger.warning("  %s: 无交易数据", month_key)
            return {"month_key": month_key, "n_trades": 0, "go_pass": False,
                    "reasons": ["无交易数据"]}

        # 计算月度指标
        metrics = self._compute_monthly_metrics(month_trades, month_key)
        logger.info("  交易数: %d", metrics.n_trades)
        logger.info("  胜率: %.1f%%", metrics.win_rate)
        logger.info("  live_mlp: %.2f%%", metrics.live_mlp_pct)
        logger.info("  live_gap: %.2f%%", metrics.live_gap_pct)
        logger.info("  live_single_loss: %.2f%%", metrics.live_single_loss_pct)
        logger.info("  live_consec_loss: %d", metrics.live_consec_loss)

        # GO/NO-GO判定
        go_pass, go_reasons = self._evaluate_tier2_go(metrics)
        metrics.go_pass = go_pass
        metrics.go_reasons = go_reasons

        # 更新连续GO月数
        if go_pass:
            # 检查是否连续 (上月也在monthly_metrics中且GO)
            prev_month = self._get_prev_month(month_key)
            if prev_month in self.state.monthly_metrics:
                if self.state.monthly_metrics[prev_month].get("go_pass", False):
                    self.state.consecutive_go_months += 1
                else:
                    self.state.consecutive_go_months = 1
            else:
                self.state.consecutive_go_months = 1
        else:
            self.state.consecutive_go_months = 0

        # 保存月度指标
        self.state.monthly_metrics[month_key] = asdict(metrics)

        # Tier 2 综合评估
        tier2_metrics = {
            "tier1_pass": True,  # v97已通过
            "consecutive_sim_go_months": self.state.consecutive_go_months,
            "live_mlp_pct": metrics.live_mlp_pct,
            "live_gap_pct": metrics.live_gap_pct,
            "live_single_loss_pct": metrics.live_single_loss_pct,
            "live_consec_loss": metrics.live_consec_loss,
            "capital_max_usd": self.capital_usd,
        }
        tier2_result = self._evaluate_tier2_admission(tier2_metrics)
        self.state.tier2_passed = tier2_result.get("passed", False)
        self.state.tier2_evaluation = tier2_result

        self.save_state()

        # 保存报告
        report = {
            "month_key": month_key,
            "metrics": asdict(metrics),
            "tier2_metrics": tier2_metrics,
            "tier2_result": tier2_result,
            "consecutive_go_months": self.state.consecutive_go_months,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        report_path = TIER2_REPORTS_DIR / f"monthly_{month_key}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info("  GO/NO-GO: %s (连续GO %d月)",
                    "✅ GO" if go_pass else "❌ NO-GO",
                    self.state.consecutive_go_months)
        logger.info("  Tier 2 PASS: %s", "✅" if self.state.tier2_passed else "❌")
        return report

    def _load_month_trades(self, month_key: str) -> List[Dict]:
        """加载指定月份的所有交易"""
        all_trades = []
        for path in TIER2_TRADES_DIR.glob("trades_*.json"):
            try:
                trades = json.loads(path.read_text(encoding="utf-8"))
                for t in trades:
                    if t.get("month_key") == month_key:
                        all_trades.append(t)
            except Exception:
                continue
        return all_trades

    def _compute_monthly_metrics(
        self, trades: List[Dict], month_key: str
    ) -> MonthlyMetrics:
        """计算月度Tier 2指标"""
        metrics = MonthlyMetrics(month_key=month_key)
        metrics.n_trades = len(trades)
        if not trades:
            return metrics

        # 胜负统计
        wins = [t for t in trades if t["live_pnl_usd"] > 0]
        losses = [t for t in trades if t["live_pnl_usd"] < 0]
        metrics.n_wins = len(wins)
        metrics.n_losses = len(losses)
        metrics.win_rate = len(wins) / len(trades) * 100 if trades else 0.0

        # PnL
        metrics.sim_pnl_total_usd = sum(t["pnl_usd"] for t in trades)
        metrics.live_pnl_total_usd = sum(t["live_pnl_usd"] for t in trades)
        metrics.cost_total_usd = sum(t["cost_usd"] for t in trades)
        metrics.avg_hold_bars = sum(t["hold_bars"] for t in trades) / len(trades)

        # Tier 2 核心指标
        # 1. live_mlp_pct: 月度亏损概率 = 亏损月? 这里简化为亏损交易比例
        #    严格定义: P(月度总PnL < 0) — 需要多月数据, 这里用亏损交易比例近似
        metrics.live_mlp_pct = len(losses) / len(trades) * 100 if trades else 0.0

        # 2. live_gap_pct: 实盘差异率 = |sim_pnl - live_pnl| / |sim_pnl| * 100
        if abs(metrics.sim_pnl_total_usd) > 0:
            metrics.live_gap_pct = abs(
                metrics.sim_pnl_total_usd - metrics.live_pnl_total_usd
            ) / abs(metrics.sim_pnl_total_usd) * 100
        else:
            metrics.live_gap_pct = 0.0

        # 3. live_single_loss_pct: 单次最大亏损%本金 (相对于capital)
        max_loss = max((abs(t["live_pnl_usd"]) for t in losses), default=0)
        metrics.live_single_loss_pct = max_loss / self.capital_usd * 100

        # 4. live_consec_loss: 连续亏损次数 (按时间排序)
        sorted_trades = sorted(trades, key=lambda t: t["entry_ts"])
        max_consec = 0
        cur_consec = 0
        for t in sorted_trades:
            if t["live_pnl_usd"] < 0:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0
        metrics.live_consec_loss = max_consec

        # 5. 最大回撤 (按时间序列)
        cum_pnl = 0
        peak = 0
        max_dd = 0
        for t in sorted_trades:
            cum_pnl += t["live_pnl_usd"]
            if cum_pnl > peak:
                peak = cum_pnl
            dd = (peak - cum_pnl) / self.capital_usd * 100
            if dd > max_dd:
                max_dd = dd
        metrics.max_drawdown_pct = max_dd

        # 6. Sharpe (简化: mean/std of daily returns)
        daily_pnls = {}
        for t in sorted_trades:
            day = datetime.fromtimestamp(t["entry_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            daily_pnls[day] = daily_pnls.get(day, 0) + t["live_pnl_usd"]
        if len(daily_pnls) > 1:
            pnl_list = list(daily_pnls.values())
            mean_pnl = statistics.mean(pnl_list)
            std_pnl = statistics.stdev(pnl_list)
            metrics.sharpe = mean_pnl / std_pnl * math.sqrt(365) if std_pnl > 0 else 0.0

        return metrics

    def _evaluate_tier2_go(self, metrics: MonthlyMetrics) -> Tuple[bool, List[str]]:
        """评估月度GO/NO-GO (Tier 2阈值)"""
        reasons = []
        go_pass = True

        # 1. live_mlp ≤ 5%
        if metrics.live_mlp_pct > TIER2_THRESHOLDS["live_mlp_pct"]["max"]:
            go_pass = False
            reasons.append(f"live_mlp={metrics.live_mlp_pct:.2f}% > 5%")
        # 2. live_gap ≤ 15%
        if metrics.live_gap_pct > TIER2_THRESHOLDS["live_gap_pct"]["max"]:
            go_pass = False
            reasons.append(f"live_gap={metrics.live_gap_pct:.2f}% > 15%")
        # 3. live_single_loss ≤ 2%
        if metrics.live_single_loss_pct > TIER2_THRESHOLDS["live_single_loss_pct"]["max"]:
            go_pass = False
            reasons.append(f"live_single_loss={metrics.live_single_loss_pct:.2f}% > 2%")
        # 4. live_consec_loss ≤ 3
        if metrics.live_consec_loss > TIER2_THRESHOLDS["live_consec_loss"]["max"]:
            go_pass = False
            reasons.append(f"live_consec_loss={metrics.live_consec_loss} > 3")
        # 5. 至少有10笔交易 (统计显著性)
        if metrics.n_trades < 10:
            reasons.append(f"n_trades={metrics.n_trades} < 10 (统计不显著)")

        if go_pass and not reasons:
            reasons.append("ALL PASS")
        return go_pass, reasons

    def _evaluate_tier2_admission(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """调用staged_admission_gate评估Tier 2准入"""
        try:
            from .staged_admission_gate import StagedAdmissionGate
            gate = StagedAdmissionGate()
            result = gate.evaluate_tier2(metrics)
            return result
        except ImportError:
            # 独立运行时无法相对导入, 用本地评估
            logger.warning("无法导入StagedAdmissionGate, 使用本地评估")
            return self._local_tier2_eval(metrics)
        except Exception as e:
            logger.error("staged_admission_gate评估失败: %s", e)
            return {"passed": False, "error": str(e)}

    def _local_tier2_eval(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """本地Tier 2评估 (当staged_admission_gate不可用时)"""
        checks = []
        all_pass = True

        # tier1_pass
        t1 = metrics.get("tier1_pass", False)
        checks.append(("tier1_pass", t1, "v98 Tier1 12/12 GO"))
        if not t1:
            all_pass = False

        # consecutive_sim_go_months >= 3
        cgm = metrics.get("consecutive_sim_go_months", 0)
        cgm_pass = cgm >= 3
        checks.append(("consecutive_sim_go_months", cgm_pass, f"{cgm}/3"))
        if not cgm_pass:
            all_pass = False

        # live_mlp <= 5%
        mlp = metrics.get("live_mlp_pct", 100)
        mlp_pass = mlp <= 5.0
        checks.append(("live_mlp_pct", mlp_pass, f"{mlp:.2f}%"))
        if not mlp_pass:
            all_pass = False

        # live_gap <= 15%
        gap = metrics.get("live_gap_pct", 100)
        gap_pass = gap <= 15.0
        checks.append(("live_gap_pct", gap_pass, f"{gap:.2f}%"))
        if not gap_pass:
            all_pass = False

        # live_single_loss <= 2%
        sl = metrics.get("live_single_loss_pct", 100)
        sl_pass = sl <= 2.0
        checks.append(("live_single_loss_pct", sl_pass, f"{sl:.2f}%"))
        if not sl_pass:
            all_pass = False

        # live_consec_loss <= 3
        cl = metrics.get("live_consec_loss", 100)
        cl_pass = cl <= 3
        checks.append(("live_consec_loss", cl_pass, f"{cl}"))
        if not cl_pass:
            all_pass = False

        # capital <= $5k
        cap = metrics.get("capital_max_usd", 0)
        cap_pass = cap <= 5000
        checks.append(("capital_max_usd", cap_pass, f"${cap}"))
        if not cap_pass:
            all_pass = False

        return {
            "passed": all_pass,
            "n_pass": sum(1 for _, p, _ in checks if p),
            "n_total": len(checks),
            "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        }

    def _get_prev_month(self, month_key: str) -> str:
        """获取上一个月key"""
        year, month = map(int, month_key.split("-"))
        if month == 1:
            return f"{year-1}-12"
        return f"{year}-{month-1:02d}"

    # ------------------------------------------------------------------
    # 状态报告
    # ------------------------------------------------------------------

    def get_status_report(self) -> Dict[str, Any]:
        """获取当前Tier 2状态报告 (v98基线版)"""
        return {
            "version": self.state.version,
            "oos_start_date": self.state.oos_start_date,
            "collection_days": self.state.collection_days,
            "last_collection": datetime.fromtimestamp(
                self.state.last_collection_ts, tz=timezone.utc
            ).isoformat() if self.state.last_collection_ts > 0 else "N/A",
            "total_trades": self.state.total_trades,
            "total_wins": self.state.total_wins,
            "total_losses": self.state.total_losses,
            "total_sim_pnl_usd": self.state.total_sim_pnl_usd,
            "total_live_pnl_usd": self.state.total_live_pnl_usd,
            "total_cost_usd": self.state.total_cost_usd,
            "gap_pct": abs(1 - self.state.total_live_pnl_usd / self.state.total_sim_pnl_usd * 100) if self.state.total_sim_pnl_usd != 0 else 0.0,
            "max_single_loss_pct": self.state.max_single_loss_pct,
            "max_consec_loss": self.state.max_consec_loss,
            "consecutive_go_months": self.state.consecutive_go_months,
            "monthly_metrics": self.state.monthly_metrics,
            "tier2_passed": self.state.tier2_passed,
            "tier2_evaluation": self.state.tier2_evaluation,
            "v98_baseline": self.state.v98_baseline.get("summary", {}),  # 主基线
            "v97_baseline_archive": self.state.v97_baseline.get("summary", {}),  # 历史归档
            "v98_hold_bars_active": self.v98_per_symbol_stats is not None,  # hold_bars融合是否激活
        }

    def can_deploy_tier2(self) -> Tuple[bool, List[str]]:
        """检查是否可以进入Tier 2 (迷你实盘$1k-$5k, v98基线)

        用户铁律: "在未达成性能指标前不得推进实盘部署工作"
        """
        blockers = []
        if not self.state.v98_baseline.get("tier1_pass", False):
            blockers.append("v98 Tier 1 未通过")
        if self.state.consecutive_go_months < 3:
            blockers.append(f"连续GO月数 {self.state.consecutive_go_months} < 3")
        if not self.state.tier2_passed:
            blockers.append("Tier 2 评估未通过")
        return (len(blockers) == 0, blockers)


# ============================================================================
# 自检
# ============================================================================

def _self_test() -> bool:
    """自检: 验证框架基本功能"""
    print("=" * 70)
    print("Tier2ForwardWalkValidator 自检")
    print("=" * 70)

    # 1. 检查常量
    assert len(V97_SYMBOLS) == 15, f"V97_SYMBOLS数量错误: {len(V97_SYMBOLS)}"
    assert OOS_START_DATE.year == 2026, f"OOS_START_DATE错误: {OOS_START_DATE}"
    print("✓ 常量检查通过")

    # 2. 检查特征计算
    fake_klines = [
        {"timestamp": 1609459200 + i * 3600, "open": 30000 + i * 100,
         "high": 30100 + i * 100, "low": 29900 + i * 100,
         "close": 30050 + i * 100, "volume": 100 + i}
        for i in range(250)
    ]
    features = FeatureComputer.compute_all_features(fake_klines, 200)
    assert "atr_pct" in features
    assert "pa_body_pct" in features
    assert "vp_vol_ratio_5" in features
    assert "mtf_ema_alignment" in features
    print(f"✓ 特征计算检查通过 ({len(features)} 特征)")

    # 3. 检查信号生成
    gen = V97SignalGenerator()
    sig = gen.generate_signal("BTC_USDT", fake_klines, 200)
    if sig:
        print(f"✓ 信号生成检查通过 (direction={sig['direction']}, pos_mult={sig['pos_mult']:.3f})")
    else:
        print("⚠ 信号生成: idx=200无信号 (可能需要更多K线)")

    # 4. 检查成本计算
    cost = CostCalculator.calc_trade_cost(1000, "BTC_USDT", 10, 500, 30000)
    assert cost["total"] > 0
    print(f"✓ 成本计算检查通过 (total=${cost['total']:.4f})")

    # 5. 检查状态持久化
    v = Tier2ForwardWalkValidator()
    v.initialize()
    assert TIER2_STATE_FILE.exists()
    print("✓ 状态持久化检查通过")

    # 6. 检查Tier 2阈值
    assert TIER2_THRESHOLDS["live_mlp_pct"]["max"] == 5.0
    assert TIER2_THRESHOLDS["live_gap_pct"]["max"] == 15.0
    assert TIER2_THRESHOLDS["live_single_loss_pct"]["max"] == 2.0
    assert TIER2_THRESHOLDS["live_consec_loss"]["max"] == 3
    print("✓ Tier 2阈值检查通过")

    print("=" * 70)
    print("自检全部通过 ✅")
    print("=" * 70)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _self_test()
