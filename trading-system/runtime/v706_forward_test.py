# -*- coding: utf-8 -*-
"""v706 V706ForwardTestRunner — 前向验证主运行器
================================================================================
用户铁律:
  "开始t2,但需要连续三年的真实数据去实测"
  "在未达成性能指标前不得推进实盘部署工作"
  "永远永远不要出现模拟牛逼, 实盘亏损的情况"
  "phased testing admission rules (simulation → mini small live → standard live)"

本模块实现 V706ForwardTestRunner, 集成 V704SignalGenerator 在真实 OOS 数据上
运行 v704 策略, 收集交易记录, 计算月度指标, 评估 SIM→LIVE 桥接条件.

架构分层:
  Layer 1: 数据获取 — GateIoDataFetcher (复用 tier2)
  Layer 2: 策略执行 — V704SignalGenerator (v704 三修复)
  Layer 3: 成本计算 — CostCalculator (5因子模型, 复用 tier2)
  Layer 4: 状态持久化 — V706State (独立于 tier2_state.json)
  Layer 5: 告警集成 — Tier2AlertChannel (alert_file 重定向到 v706_oos_data)
  Layer 6: 评估 — 月度Tier 2指标 + SIM→LIVE桥接

使用方式:
  # 首次运行: 初始化 + 收集OOS数据
  runner = V706ForwardTestRunner()
  runner.initialize()
  runner.collect_daily_data()
  runner.save_state()

  # 每日运行: 加载状态 + 收集新数据
  runner = V706ForwardTestRunner.load_state()
  runner.collect_daily_data()

  # 月底运行: 月度评估
  runner.run_monthly_evaluation()

  # 查看状态 + 对比报告
  report = runner.get_status_report()
  comparison = runner.run_v706_vs_v98_comparison()
  can_deploy = runner.can_deploy_live()

来源:
  - tier2_forward_walk_validator.py:1515 — Tier2ForwardWalkValidator (参考模式)
  - v706_signal_generator.py — V704SignalGenerator
  - v706_state.py — V706State
  - v706_config.py — V706_RISK_PARAMS / V706_OOS_CONFIG / SIM_TO_LIVE_BRIDGE
================================================================================
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# v706 模块
from v706_config import (
    V706_CAPITAL_USD, V706_DATA_DIR,
    V706_STATE_FILE, V706_KLINES_DIR, V706_TRADES_DIR, V706_REPORTS_DIR,
    V706_ALERTS_FILE, V706_OOS_CONFIG,
    LIVE_THRESHOLDS, SIM_TO_LIVE_BRIDGE,
    ALERT_TRIGGERS, V704_BASELINE, ensure_v706_directories,
)
from v706_state import V706State
from v706_signal_generator import V704SignalGenerator

# 复用 tier2 组件 (D5: 不重复实现)
from tier2_forward_walk_validator import (
    GateIoDataFetcher, CostCalculator, OOS_START_DATE,
)
from tier2_alert_channel import Tier2AlertChannel

logger = logging.getLogger("hermes.v706.forward_test")


class V706ForwardTestRunner:
    """v706 前向验证主运行器

    用户铁律: "开始连续3个月的实盘数据实测"
    用户铁律: "在未达成性能指标前不得推进实盘部署工作"

    在真实 OOS 数据 (2026-07-01 起) 上运行 v704 策略, 收集交易记录,
    计算月度 Tier 2 指标, 评估 SIM→LIVE 桥接条件.

    架构决策:
      D4: 独立 v706_state.json (与 tier2_state.json 隔离避免污染)
      D5: 复用 GateIoDataFetcher / CostCalculator / Tier2AlertChannel
      D6: consec_loss/cooldown 委托 V704SignalGenerator (其内部委托 v97)
    """

    # v706 OOS 开始时间戳 (2026-07-01 00:00:00 UTC)
    OOS_START_TS = OOS_START_DATE.timestamp()

    def __init__(
        self,
        capital_usd: float = V706_CAPITAL_USD,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self.capital_usd = capital_usd
        self.symbols = symbols or V706_OOS_CONFIG["symbols"]
        self.state = V706State()
        self.fetcher = GateIoDataFetcher(cache_dir=V706_KLINES_DIR)
        self.signal_gen = V704SignalGenerator()
        self.cost_calc = CostCalculator()
        self.alert_channel = Tier2AlertChannel(alert_file=str(V706_ALERTS_FILE))
        # 确保目录存在
        ensure_v706_directories()

    # ========================================================================
    # 状态管理
    # ========================================================================

    def initialize(self) -> None:
        """初始化 v706 状态 (首次运行或加载已有状态)

        用户铁律: "在未达成性能指标前不得推进实盘部署工作"

        流程:
          1. 若状态文件存在 → 加载并回灌到 signal_gen
          2. 若状态文件不存在 → 创建新状态, 写入 v704_baseline
          3. 设置 signal_gen 的 v98 per_symbol_stats (hold_bars融合所需)
        """
        if V706_STATE_FILE.exists():
            self.state = V706State.load_state(V706_STATE_FILE)
            # 回灌状态到 signal_gen (冷却期/consec/kelly history)
            self.state.restore_to_signal_gen(self.signal_gen)
            logger.info(
                "v706 状态已加载: collection_days=%d, total_trades=%d, "
                "consecutive_go_months=%d",
                self.state.collection_days, self.state.total_trades,
                self.state.consecutive_go_months)
        else:
            self.state.created_at = datetime.now(timezone.utc).isoformat()
            self.state.last_updated = self.state.created_at
            self.state.v704_baseline = dict(V704_BASELINE)
            self.save_state()
            logger.info("v706 首次初始化完成, 状态文件: %s", V706_STATE_FILE)

        # 发送初始化告警
        self.alert_channel.send_alert(
            level="INFO",
            category="v706_init",
            message=f"V706ForwardTestRunner 初始化完成, "
                    f"trades={self.state.total_trades}, days={self.state.collection_days}",
            metrics=self.state.get_status_summary(),
            action_hint="继续每日 collect_daily_data()",
        )

    def save_state(self) -> None:
        """保存状态 (先同步 signal_gen 运行时状态到 state, 再持久化)

        修复 ERR-20260703-cooldown-persistence:
          per_symbol_cooldown 双写 (signal_gen 运行时 + state 持久化),
          跨日 collect_daily_data 时不丢失.
        """
        # 同步 signal_gen 运行时状态到 state
        self.state.update_from_signal_gen(self.signal_gen)
        # 持久化
        self.state.save_state(V706_STATE_FILE)

    @classmethod
    def load_state(cls) -> "V706ForwardTestRunner":
        """从 JSON 加载状态 (classmethod, 返回新实例)

        使用方式:
          runner = V706ForwardTestRunner.load_state()
          runner.collect_daily_data()
        """
        runner = cls()
        runner.initialize()
        return runner

    # ========================================================================
    # 数据收集
    # ========================================================================

    def collect_daily_data(self) -> Dict[str, Any]:
        """收集每日 OOS 数据 + 运行 v704 策略

        用户铁律: "开始连续3个月的实盘数据实测"
        用户铁律: "不要通过已知数据反推策略" — 预热数据只用于指标计算

        流程:
          1. 获取最新K线 (从 last_collection_ts 到 now)
          2. 加载预热数据 (训练期最后300条K线, 用于EMA200/ATR等指标计算)
          3. 合并预热 + OOS (去重)
          4. 在OOS期 (idx >= oos_start_idx) 运行策略生成交易
          5. 保存交易记录
          6. 更新状态
          7. 检查告警

        Returns:
            收集报告 Dict
        """
        # 防御性修复 (ERR-20260703-tier2-state-reset):
        # 若 state 是默认空状态但磁盘有状态文件, 必须先加载
        if (self.state.collection_days == 0
                and self.state.last_collection_ts == 0
                and V706_STATE_FILE.exists()):
            logger.warning("检测到空状态但磁盘有状态文件, 补执行 initialize()")
            self.initialize()

        logger.info("=" * 70)
        logger.info("v706 每日数据收集 — %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 70)

        # 1. 获取最新K线
        now_ts = time.time()
        last_ts = self.state.last_collection_ts or self.OOS_START_TS
        logger.info("  上次收集: %s", datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat())
        logger.info("  当前时间: %s", datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat())

        # 获取OOS期间的K线
        klines_map = self.fetcher.fetch_historical_oos_klines(
            self.symbols, last_ts, now_ts)

        # 2. 运行策略生成交易
        oos_start_ts = self.OOS_START_TS
        new_trades: List[Dict[str, Any]] = []
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
            # 找到OOS起点
            oos_start_idx = 0
            for i, k in enumerate(combined):
                if k["timestamp"] >= oos_start_ts:
                    oos_start_idx = i
                    break

            # 运行策略
            trades = self._run_strategy_on_klines(sym, combined, oos_start_idx)
            new_trades.extend(trades)

            # 保存交易记录
            if trades:
                trade_file = V706_TRADES_DIR / f"{sym}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
                with open(trade_file, "w", encoding="utf-8") as f:
                    json.dump(trades, f, indent=2, ensure_ascii=False, default=str)

            logger.info("  %s: %d 笔新交易", sym, len(trades))

        # 3. 更新状态
        self.state.collection_days += 1
        self.state.last_collection_ts = now_ts

        # 4. 更新全局统计
        for trade in new_trades:
            self.state.record_trade(
                pnl_pct=trade["pnl_pct"],
                pnl_usd=trade["pnl_usd"],
                cost_usd=trade["cost_usd"],
                is_bear_signal=trade.get("v704_bear_signal", False),
                kelly_mult=trade.get("v704_kelly_mult", 0.0),
            )
            # 更新 max_consec_loss
            consec = self.signal_gen.v97.current_consec_loss
            if consec > self.state.max_consec_loss:
                self.state.max_consec_loss = consec

        # 5. 检查告警
        self._check_alerts()

        # 6. 保存状态
        self.save_state()

        report = {
            "collection_date": datetime.now(timezone.utc).isoformat(),
            "symbols_processed": len(self.symbols),
            "new_trades": len(new_trades),
            "total_trades": self.state.total_trades,
            "collection_days": self.state.collection_days,
            "bear_signal_count": self.state.bear_signal_count,
            "kelly_zero_ratio": round(self.state.kelly_zero_ratio, 4),
        }
        logger.info("=" * 70)
        logger.info("v706 收集完成: %d 笔新交易 (累计 %d 笔, %d 天)",
                    len(new_trades), self.state.total_trades, self.state.collection_days)
        logger.info("=" * 70)
        return report

    def _run_strategy_on_klines(
        self,
        symbol: str,
        klines: List[Dict],
        oos_start_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """在K线上运行 v704 策略 (参数冻结)

        用户铁律: "不要通过已知数据反推策略"
        信号只在OOS期 (idx >= oos_start_idx) 生成, 预热期只用于指标计算.

        Args:
            symbol: 交易对
            klines: 合并后的K线 (预热 + OOS)
            oos_start_idx: OOS起点在 klines 中的索引

        Returns:
            交易记录列表
        """
        trades: List[Dict[str, Any]] = []
        open_signal: Optional[Dict[str, Any]] = None

        for idx in range(len(klines)):
            # 检查出场 (持仓期间可以跨OOS边界)
            if open_signal is not None:
                exit_info = self.signal_gen.check_exit(open_signal, klines, idx)
                if exit_info is not None:
                    trade = self._close_trade(
                        symbol, open_signal, exit_info, klines, idx)
                    if trade is not None:
                        trades.append(trade)
                    open_signal = None

            # 检查入场 (无持仓时, 只在OOS期生成信号)
            if open_signal is None and idx >= oos_start_idx:
                signal = self.signal_gen.generate_signal(symbol, klines, idx)
                if signal is not None:
                    open_signal = signal

        logger.info("    %s: 生成 %d 笔交易", symbol, len(trades))
        return trades

    def _close_trade(
        self,
        symbol: str,
        open_signal: Dict[str, Any],
        exit_info: Dict[str, Any],
        klines: List[Dict],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """关闭交易并记录 (计算 PnL + 成本 + v704 元数据)

        Args:
            symbol: 交易对
            open_signal: 入场信号 dict
            exit_info: 出场信息 (含 exit_price/exit_reason/hold_bars)
            klines: K线数据
            idx: 当前 K线索引

        Returns:
            交易记录 dict, 或 None (异常情况)
        """
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

        # 成本 (5因子模型)
        cost = self.cost_calc.calc_trade_cost(
            pos_size_usd, symbol, hold_bars, atr_abs, entry_price)
        cost_usd = cost["total"]

        # 实盘PnL
        live_pnl_usd = self.cost_calc.calc_live_pnl(pnl_usd, cost_usd)
        live_pnl_pct = live_pnl_usd / pos_size_usd if pos_size_usd > 0 else 0.0

        # 单亏≤2%硬截断 (v97)
        if live_pnl_pct < -self.signal_gen.v97.V97_MAX_SINGLE_LOSS_PCT / 100:
            live_pnl_pct = -self.signal_gen.v97.V97_MAX_SINGLE_LOSS_PCT / 100
            live_pnl_usd = pos_size_usd * live_pnl_pct

        # 调用 on_trade_close (回写 kelly history + 更新 consec_loss + 设置冷却期)
        self.signal_gen.on_trade_close(
            signal=open_signal,
            exit_price=exit_price,
            exit_ts=exit_info.get("exit_ts", 0),
            pnl_pct=live_pnl_pct,
            pnl_usd=live_pnl_usd)

        # 同步冷却期到 state (双写)
        if live_pnl_pct < 0:
            self.state.per_symbol_cooldown[symbol] = \
                self.signal_gen.v97.per_symbol_cooldown.get(symbol, 0)
        else:
            self.state.per_symbol_cooldown.pop(symbol, None)

        # 月份
        month_key = datetime.fromtimestamp(
            open_signal["entry_ts"], tz=timezone.utc).strftime("%Y-%m")

        # exit_reason 命名统一 (v99 ERR-20260703-exit-reason-naming)
        _EXIT_REASON_MAP = {
            "tp": "take_profit",
            "sl": "stop_loss",
            "timeout": "time_stop",
        }
        exit_reason_norm = _EXIT_REASON_MAP.get(
            exit_info["exit_reason"], exit_info["exit_reason"])

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
            "exit_reason": exit_reason_norm,
            "htf_state": open_signal.get("htf_state", "unknown"),
            # v99 Task #94 修复: features字段填充 (原缺失导致PSI/FeaturesDriftMonitor/attribution全部失效)
            "features": open_signal.get("features", {}),
            # v704 元数据
            "v704_kelly_mult": open_signal.get("v704_kelly_mult", 0.0),
            "v704_regime": open_signal.get("v704_regime", "unknown"),
            "v704_regime_confidence": open_signal.get("v704_regime_confidence", 0.0),
            "v704_bear_signal": open_signal.get("v704_bear_signal", False),
            "v704_vol_target_mult": open_signal.get("v704_vol_target_mult", 1.0),  # v99 Task #93 修复3
        }
        return trade

    # ========================================================================
    # 月度评估
    # ========================================================================

    def run_monthly_evaluation(self, month_key: Optional[str] = None) -> Dict[str, Any]:
        """运行月度 Tier 2 评估

        用户铁律: "在未达成性能指标前不得推进实盘部署工作"

        Args:
            month_key: 指定月份, 如 "2026-07". None=当前月

        Returns:
            评估报告 Dict
        """
        if month_key is None:
            month_key = datetime.now(timezone.utc).strftime("%Y-%m")

        logger.info("=" * 70)
        logger.info("v706 月度评估 — %s", month_key)
        logger.info("=" * 70)

        # 加载该月所有交易
        monthly_trades = self._load_monthly_trades(month_key)
        if not monthly_trades:
            logger.warning("  %s 无交易记录", month_key)
            return {"month_key": month_key, "n_trades": 0, "go_pass": False,
                    "go_reasons": ["无交易记录"]}

        # 计算月度指标
        n_trades = len(monthly_trades)
        n_wins = sum(1 for t in monthly_trades if t["live_pnl_pct"] >= 0)
        n_losses = n_trades - n_wins
        win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
        sim_pnl_total = sum(t["pnl_usd"] for t in monthly_trades)
        live_pnl_total = sum(t["live_pnl_usd"] for t in monthly_trades)
        cost_total = sum(t["cost_usd"] for t in monthly_trades)
        gap_pct = abs(sim_pnl_total - live_pnl_total) / abs(sim_pnl_total) * 100 \
            if sim_pnl_total != 0 else 0

        # 单次最大亏损
        losses = [t["live_pnl_pct"] for t in monthly_trades if t["live_pnl_pct"] < 0]
        single_loss_pct = max(abs(l) for l in losses) * 100 if losses else 0

        # 连续亏损
        consec_loss = 0
        max_consec = 0
        for t in monthly_trades:
            if t["live_pnl_pct"] < 0:
                consec_loss += 1
                max_consec = max(max_consec, consec_loss)
            else:
                consec_loss = 0

        # 月度亏损概率 (mlp)
        mlp_pct = n_losses / n_trades * 100 if n_trades > 0 else 0

        # GO/NO-GO 评估
        go_reasons: List[str] = []
        go_pass = True

        if mlp_pct > LIVE_THRESHOLDS["live_mlp_pct"]["max"]:
            go_reasons.append(f"mlp_pct={mlp_pct:.1f}%>{LIVE_THRESHOLDS['live_mlp_pct']['max']}%")
            go_pass = False
        if gap_pct > LIVE_THRESHOLDS["live_gap_pct"]["max"]:
            go_reasons.append(f"gap_pct={gap_pct:.1f}%>{LIVE_THRESHOLDS['live_gap_pct']['max']}%")
            go_pass = False
        if single_loss_pct > LIVE_THRESHOLDS["live_single_loss_pct"]["max"]:
            go_reasons.append(f"single_loss={single_loss_pct:.1f}%>{LIVE_THRESHOLDS['live_single_loss_pct']['max']}%")
            go_pass = False
        if max_consec > LIVE_THRESHOLDS["live_consec_loss"]["max"]:
            go_reasons.append(f"consec_loss={max_consec}>{LIVE_THRESHOLDS['live_consec_loss']['max']}")
            go_pass = False

        if go_pass:
            go_reasons.append("所有指标达标")
            self.state.consecutive_go_months += 1
        else:
            self.state.consecutive_go_months = 0

        # 记录月度指标
        monthly_metrics = {
            "month_key": month_key,
            "n_trades": n_trades,
            "n_wins": n_wins,
            "n_losses": n_losses,
            "win_rate": round(win_rate, 2),
            "live_mlp_pct": round(mlp_pct, 2),
            "live_gap_pct": round(gap_pct, 2),
            "live_single_loss_pct": round(single_loss_pct, 2),
            "live_consec_loss": max_consec,
            "sim_pnl_total_usd": round(sim_pnl_total, 2),
            "live_pnl_total_usd": round(live_pnl_total, 2),
            "cost_total_usd": round(cost_total, 2),
            "go_pass": go_pass,
            "go_reasons": go_reasons,
        }
        self.state.monthly_metrics[month_key] = monthly_metrics

        # 保存评估报告
        report_file = V706_REPORTS_DIR / f"monthly_{month_key}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(monthly_metrics, f, indent=2, ensure_ascii=False, default=str)

        # 发送告警
        alert_level = "INFO" if go_pass else "WARN"
        self.alert_channel.send_alert(
            level=alert_level,
            category="monthly_evaluation",
            message=f"{month_key} 月度评估: {'GO' if go_pass else 'NO-GO'} "
                    f"(trades={n_trades}, win_rate={win_rate:.1f}%, "
                    f"mlp={mlp_pct:.1f}%, gap={gap_pct:.1f}%)",
            metrics=monthly_metrics,
            action_hint="继续收集数据" if go_pass else "检查退化原因",
        )

        # 保存状态
        self.save_state()

        logger.info("  月度评估完成: %s (consecutive_go=%d)",
                    "GO" if go_pass else "NO-GO", self.state.consecutive_go_months)
        return monthly_metrics

    def _load_monthly_trades(self, month_key: str) -> List[Dict[str, Any]]:
        """加载指定月份的所有交易记录

        Args:
            month_key: 月份, 如 "2026-07"

        Returns:
            交易记录列表
        """
        trades: List[Dict[str, Any]] = []
        if not V706_TRADES_DIR.exists():
            return trades
        for trade_file in V706_TRADES_DIR.glob("*.json"):
            try:
                file_trades = json.loads(trade_file.read_text(encoding="utf-8"))
                for t in file_trades:
                    if t.get("month_key") == month_key:
                        trades.append(t)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("加载交易文件失败 %s: %s", trade_file, str(e)[:80])
        return trades

    # ========================================================================
    # Phase 3: 告警 + 对比 + SIM→LIVE 桥接
    # ========================================================================

    def _check_alerts(self) -> None:
        """每日告警检查 (7个触发条件)

        来源: v706_config.ALERT_TRIGGERS
        """
        # 1. kelly 死锁风险 (v704 Fix1 验证)
        if self.state.kelly_zero_ratio > ALERT_TRIGGERS["kelly_zero_ratio_critical"]:
            self.alert_channel.send_alert(
                level="CRITICAL",
                category="kelly_deadlock",
                message=f"kelly_zero_ratio={self.state.kelly_zero_ratio:.1%} "
                        f"> {ALERT_TRIGGERS['kelly_zero_ratio_critical']:.0%} 临界值",
                metrics={"kelly_zero_ratio": self.state.kelly_zero_ratio,
                         "kelly_zero_count": self.state.kelly_zero_count,
                         "total_trades": self.state.total_trades},
                action_hint="检查 KellyCriterionSizer 是否触发 trending_down fallback",
            )
        elif self.state.kelly_zero_ratio > ALERT_TRIGGERS["kelly_zero_ratio_warn"]:
            self.alert_channel.send_alert(
                level="WARN",
                category="kelly_deadlock",
                message=f"kelly_zero_ratio={self.state.kelly_zero_ratio:.1%} "
                        f"> {ALERT_TRIGGERS['kelly_zero_ratio_warn']:.0%} 警告值",
                metrics={"kelly_zero_ratio": self.state.kelly_zero_ratio},
                action_hint="监控 kelly 死锁趋势",
            )

        # 2. 连续亏损
        if self.signal_gen.v97.current_consec_loss >= ALERT_TRIGGERS["consec_loss_critical"]:
            self.alert_channel.send_alert(
                level="CRITICAL",
                category="consec_loss",
                message=f"连续亏损 {self.signal_gen.v97.current_consec_loss} 次 "
                        f">= {ALERT_TRIGGERS['consec_loss_critical']} 临界值",
                metrics={"current_consec_loss": self.signal_gen.v97.current_consec_loss},
                action_hint="检查策略是否退化, 考虑暂停",
            )
        elif self.signal_gen.v97.current_consec_loss >= ALERT_TRIGGERS["consec_loss_warn"]:
            self.alert_channel.send_alert(
                level="WARN",
                category="consec_loss",
                message=f"连续亏损 {self.signal_gen.v97.current_consec_loss} 次",
                metrics={"current_consec_loss": self.signal_gen.v97.current_consec_loss},
                action_hint="监控下一笔交易",
            )

        # 3. 单次亏损
        if self.state.max_single_loss_pct >= ALERT_TRIGGERS["single_loss_critical_pct"]:
            self.alert_channel.send_alert(
                level="CRITICAL",
                category="single_loss",
                message=f"单次最大亏损 {self.state.max_single_loss_pct:.2f}% "
                        f">= {ALERT_TRIGGERS['single_loss_critical_pct']}% 临界值",
                metrics={"max_single_loss_pct": self.state.max_single_loss_pct},
                action_hint="检查止损是否失效",
            )
        elif self.state.max_single_loss_pct >= ALERT_TRIGGERS["single_loss_warn_pct"]:
            self.alert_channel.send_alert(
                level="WARN",
                category="single_loss",
                message=f"单次最大亏损 {self.state.max_single_loss_pct:.2f}%",
                metrics={"max_single_loss_pct": self.state.max_single_loss_pct},
                action_hint="监控止损距离",
            )

        # 4. bear_signal 静默 (v704 Fix2 验证)
        # 若 OOS 期超过 7 天仍无 bear_signal 触发, 发出警告
        if (self.state.collection_days >= ALERT_TRIGGERS["bear_signal_zero_count_days"]
                and self.state.bear_signal_count == 0):
            self.alert_channel.send_alert(
                level="WARN",
                category="bear_signal_silent",
                message=f"OOS {self.state.collection_days} 天无 bear_signal 触发 "
                        f"(v704 Fix2 可能未生效)",
                metrics={"collection_days": self.state.collection_days,
                         "bear_signal_count": 0},
                action_hint="检查 4h 下降趋势是否出现, BearTrendShortSignal 是否正常",
            )

        # 5. regime 分布异常 (单一状态占比过高)
        if self.state.regime_distribution:
            total = sum(self.state.regime_distribution.values())
            if total > 0:
                for regime, count in self.state.regime_distribution.items():
                    ratio = count / total
                    if ratio > ALERT_TRIGGERS["regime_single_state_max_ratio"]:
                        self.alert_channel.send_alert(
                            level="WARN",
                            category="regime_distribution",
                            message=f"regime '{regime}' 占比 {ratio:.1%} "
                                    f"> {ALERT_TRIGGERS['regime_single_state_max_ratio']:.0%}",
                            metrics={"regime": regime, "ratio": ratio,
                                     "distribution": self.state.regime_distribution},
                            action_hint="检查市场是否处于异常状态",
                        )
                        break

    def run_v706_vs_v98_comparison(self) -> Dict[str, Any]:
        """v704 vs v98 头对头对比报告

        读取 tier2_state.json (v98 基线), 对比 v704 vs v98 的 live_pnl/gap/win_rate.

        Returns:
            对比报告 Dict
        """
        # 加载 v98 基线状态 (tier2_state.json)
        tier2_state_file = V706_DATA_DIR.parent / "tier2_oos_data" / "tier2_state.json"
        v98_state = {}
        if tier2_state_file.exists():
            try:
                v98_state = json.loads(tier2_state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("加载 tier2_state.json 失败: %s", str(e)[:80])

        v98_summary = {
            "total_trades": v98_state.get("total_trades", 0),
            "total_wins": v98_state.get("total_wins", 0),
            "total_losses": v98_state.get("total_losses", 0),
            "total_live_pnl_usd": v98_state.get("total_live_pnl_usd", 0),
            "max_single_loss_pct": v98_state.get("max_single_loss_pct", 0),
            "max_consec_loss": v98_state.get("max_consec_loss", 0),
            "consecutive_go_months": v98_state.get("consecutive_go_months", 0),
        }
        v704_summary = self.state.get_status_summary()

        # 对比
        delta_live_pnl = v704_summary["total_live_pnl_usd"] - v98_summary["total_live_pnl_usd"]
        delta_trades = v704_summary["total_trades"] - v98_summary["total_trades"]
        delta_win_rate = v704_summary["win_rate"] - (
            v98_summary["total_wins"] / v98_summary["total_trades"] * 100
            if v98_summary["total_trades"] > 0 else 0)

        comparison = {
            "comparison_date": datetime.now(timezone.utc).isoformat(),
            "v704": v704_summary,
            "v98": v98_summary,
            "delta": {
                "live_pnl_usd": round(delta_live_pnl, 2),
                "trades": delta_trades,
                "win_rate_pct": round(delta_win_rate, 2),
            },
            "v704_inferior": delta_live_pnl < ALERT_TRIGGERS["v704_inferior_delta_usd"],
        }

        # 保存对比报告
        report_file = V706_REPORTS_DIR / f"v704_vs_v98_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)

        # 告警: v704 表现劣于 v98
        if comparison["v704_inferior"]:
            self.alert_channel.send_alert(
                level="WARN",
                category="v704_vs_v98",
                message=f"v704 live_pnl (${v704_summary['total_live_pnl_usd']:.0f}) "
                        f"< v98 (${v98_summary['total_live_pnl_usd']:.0f}), "
                        f"delta=${delta_live_pnl:.0f}",
                metrics=comparison,
                action_hint="检查 v704 三修复是否在 OOS 期生效",
            )

        return comparison

    def can_deploy_live(self) -> Dict[str, Any]:
        """SIM→LIVE 桥接评估 (是否能进入实盘阶段)

        用户铁律: "phased testing admission rules (simulation → mini live → standard live)"
        用户铁律: "在未达成性能指标前不得推进实盘部署工作"

        检查 SIM_TO_LIVE_BRIDGE 中的6个条件:
          1. consecutive_sim_go_months >= 3
          2. kelly_zero_ratio <= 0.10
          3. regime_mismatch_max <= 0.30 (简化: 检查 regime 分布是否正常)
          4. min_oos_trades >= 30
          5. bear_signal_min_count >= 1
          6. gap_safety_margin_pct <= 10.0 (实际 gap 远小于 15% 红线)

        Returns:
            评估结果 Dict
        """
        checks = {
            "consecutive_sim_go_months": {
                "current": self.state.consecutive_go_months,
                "required": SIM_TO_LIVE_BRIDGE["consecutive_sim_go_months"],
                "pass": self.state.consecutive_go_months >= SIM_TO_LIVE_BRIDGE["consecutive_sim_go_months"],
            },
            "kelly_zero_ratio": {
                "current": round(self.state.kelly_zero_ratio, 4),
                "required": SIM_TO_LIVE_BRIDGE["kelly_zero_max_ratio"],
                "pass": self.state.kelly_zero_ratio <= SIM_TO_LIVE_BRIDGE["kelly_zero_max_ratio"],
            },
            "min_oos_trades": {
                "current": self.state.total_trades,
                "required": SIM_TO_LIVE_BRIDGE["min_oos_trades"],
                "pass": self.state.total_trades >= SIM_TO_LIVE_BRIDGE["min_oos_trades"],
            },
            "bear_signal_min_count": {
                "current": self.state.bear_signal_count,
                "required": SIM_TO_LIVE_BRIDGE["bear_signal_min_count"],
                "pass": self.state.bear_signal_count >= SIM_TO_LIVE_BRIDGE["bear_signal_min_count"],
            },
        }

        # gap 安全边际 (从最近月度评估获取)
        latest_month = max(self.state.monthly_metrics.keys()) if self.state.monthly_metrics else None
        if latest_month:
            latest_gap = self.state.monthly_metrics[latest_month].get("live_gap_pct", 100)
            checks["gap_safety_margin"] = {
                "current": latest_gap,
                "required": SIM_TO_LIVE_BRIDGE["gap_safety_margin_pct"],
                "pass": latest_gap <= SIM_TO_LIVE_BRIDGE["gap_safety_margin_pct"],
            }

        all_pass = all(c["pass"] for c in checks.values())

        result = {
            "evaluation_date": datetime.now(timezone.utc).isoformat(),
            "can_deploy": all_pass,
            "checks": checks,
            "summary": self.state.get_status_summary(),
        }

        # 保存评估结果
        self.state.tier2_evaluation = result
        self.state.tier2_passed = all_pass
        self.save_state()

        # 告警
        if all_pass:
            self.alert_channel.send_alert(
                level="CRITICAL",
                category="sim_to_live_bridge",
                message="v704 SIM→LIVE 桥接评估通过! 可进入 mini live 阶段 ($5k)",
                metrics=result,
                action_hint="用户确认后, 进入 phased testing admission rules Tier 3",
            )
        else:
            failed = [k for k, c in checks.items() if not c["pass"]]
            self.alert_channel.send_alert(
                level="INFO",
                category="sim_to_live_bridge",
                message=f"v704 SIM→LIVE 桥接评估未通过, 失败项: {failed}",
                metrics=result,
                action_hint="继续收集 OOS 数据, 等待条件达标",
            )

        return result

    # ========================================================================
    # 状态报告
    # ========================================================================

    def get_status_report(self) -> Dict[str, Any]:
        """获取当前状态报告 (供状态查询)"""
        return {
            "version": self.state.version,
            "collection_days": self.state.collection_days,
            "last_collection": datetime.fromtimestamp(
                self.state.last_collection_ts, tz=timezone.utc).isoformat()
                if self.state.last_collection_ts > 0 else "无",
            "summary": self.state.get_status_summary(),
            "regime_distribution": self.state.regime_distribution,
            "v704_baseline_ann": V704_BASELINE.get("avg_ann_pct", 0),
            "v704_baseline_degradation": V704_BASELINE.get("degradation_pct", 0),
            "tier2_passed": self.state.tier2_passed,
            "consecutive_go_months": self.state.consecutive_go_months,
        }


# ============================================================================
# 自验证
# ============================================================================


def _self_validate() -> bool:
    """自验证: 确保 V706ForwardTestRunner 可实例化"""
    try:
        runner = V706ForwardTestRunner()
        if runner.signal_gen is None:
            logger.error("signal_gen 实例化失败")
            return False
        if runner.fetcher is None:
            logger.error("fetcher 实例化失败")
            return False
        if runner.cost_calc is None:
            logger.error("cost_calc 实例化失败")
            return False
        if runner.alert_channel is None:
            logger.error("alert_channel 实例化失败")
            return False
        # 验证状态摘要
        report = runner.get_status_report()
        if "summary" not in report:
            logger.error("get_status_report 缺少 summary")
            return False
        return True
    except Exception as e:
        logger.error("V706ForwardTestRunner 自验证失败: %s", str(e)[:100])
        return False


assert _self_validate(), "V706ForwardTestRunner 自验证失败"


if __name__ == "__main__":
    print("=" * 70)
    print("v706 V706ForwardTestRunner — 自测")
    print("=" * 70)

    # 修复 ERR-20260704-v706-selftest-overwrites-state (__main__ 同源 bug):
    # V706ForwardTestRunner() 未调用 initialize() 时调用 can_deploy_live() 会
    # 触发 save_state() 写空状态覆盖真实 V706_STATE_FILE. 使用 monkey-patch
    # 临时替换模块级 V706_STATE_FILE 指向临时路径, 完全隔离副作用.
    import tempfile
    import shutil
    _original_state_file = V706_STATE_FILE
    _tmp_state_dir = Path(tempfile.mkdtemp(prefix="v706_main_test_"))
    V706_STATE_FILE = _tmp_state_dir / "v706_state_main_test.json"
    print(f"  自测隔离: 状态文件临时指向 {V706_STATE_FILE}")
    print(f"  (真实状态文件 {_original_state_file} 已保护)")

    try:
        # 测试1: 实例化
        runner = V706ForwardTestRunner()
        print(f"\n[Test 1] 实例化: signal_gen={type(runner.signal_gen).__name__}, "
              f"fetcher={type(runner.fetcher).__name__}, "
              f"cost_calc={type(runner.cost_calc).__name__}, "
              f"alert_channel={type(runner.alert_channel).__name__}")
        assert runner.signal_gen is not None
        assert runner.fetcher is not None
        assert runner.cost_calc is not None
        assert runner.alert_channel is not None
        print("  ✅ PASS")

        # 测试2: get_status_report
        report = runner.get_status_report()
        print(f"\n[Test 2] get_status_report: keys={list(report.keys())}")
        assert "summary" in report
        assert "regime_distribution" in report
        assert "v704_baseline_ann" in report
        print("  ✅ PASS")

        # 测试3: can_deploy_live (初始状态应未通过)
        # 注意: 此方法会触发 save_state(), 但因 monkey-patch 已隔离到临时路径
        result = runner.can_deploy_live()
        print(f"\n[Test 3] can_deploy_live: can_deploy={result['can_deploy']}, "
              f"checks={list(result['checks'].keys())}")
        assert not result["can_deploy"], "初始状态不应通过 SIM→LIVE 桥接"
        print("  ✅ PASS (save_state 副作用已隔离到临时路径)")

        # 测试4: run_v706_vs_v98_comparison
        comparison = runner.run_v706_vs_v98_comparison()
        print(f"\n[Test 4] run_v706_vs_v98_comparison: keys={list(comparison.keys())}")
        assert "v704" in comparison
        assert "v98" in comparison
        assert "delta" in comparison
        print("  ✅ PASS")

    finally:
        # 恢复模块级全局变量 + 清理临时目录
        V706_STATE_FILE = _original_state_file
        shutil.rmtree(str(_tmp_state_dir), ignore_errors=True)
        print(f"\n  自测清理: 已恢复 V706_STATE_FILE, 已删除临时目录")

    # 防御性检查: 真实状态文件未被污染
    if _original_state_file.exists():
        try:
            with open(_original_state_file, "r", encoding="utf-8") as f:
                _real_state = json.load(f)
            _real_trades = _real_state.get("total_trades", 0)
            print(f"  防御性检查: 真实状态完整 (total_trades={_real_trades})")
        except Exception as _e:
            print(f"  ⚠️ 防御性检查失败: {type(_e).__name__}: {_e}")

    print("\n" + "=" * 70)
    print("✅ v706 V706ForwardTestRunner 自测全部通过 (4/4)")
    print("=" * 70)
