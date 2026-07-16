"""
MetricsComputationMixin — Phase 7.3 架构解耦
从 evolution_loop.py 提取的指标计算逻辑。

设计:
  - Mixin 模式: 不继承任何基类,通过多重继承注入 EvolutionLoop
  - 鸭子类型依赖:
      _serialize_capacity_stress_report: self._last_capacity_stress_report (可选)
      _compute_metrics_from_pnls: 零 self 依赖 (纯函数, 仅用 np/math)
      _compute_intraday_metrics: self.engine.get_all_trades()
  - 零循环依赖: 仅依赖 logger + numpy + math + typing

来源:
  - v2.24 numpy 向量化指标计算 (P4#7)
  - v2.29 日内交易专属评估 (2025-2026 前沿日内交易最佳实践)
  - R12-P0-2 容量压力测试报告序列化 (D5 guard pattern)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class MetricsComputationMixin:
    """指标计算管理 Mixin (Phase 7.3 提取)

    Host 鸭子类型依赖:
      - self._last_capacity_stress_report: Optional[Any] (CapacityStressReport 或 None)
      - self.engine: 需有 get_all_trades() -> Dict[str, List[trade]] 方法
    """

    def _serialize_capacity_stress_report(self) -> Dict[str, Any]:
        """序列化 _last_capacity_stress_report (CapacityStressReport) 为可 JSON 序列化的 dict

        R12-P0-2 新增: 供 validate_anti_overfitting 返回 dict 中 r12_capacity_stress_test 字段使用.
        防护: 不抛异常 (D5 guard pattern), 失败时返回降级 dict.
        """
        _report = getattr(self, "_last_capacity_stress_report", None)
        if _report is None:
            return {"enabled": False, "reason": "report 为 None"}
        try:
            # 提取 CapacityStressReport 关键字段 (字段名对齐 capacity_stress_test.py 定义)
            _levels = []
            for _lvl in getattr(_report, "tested_levels", []) or []:
                _net = float(getattr(_lvl, "net_pnl", 0.0))
                _levels.append({
                    "capital": float(getattr(_lvl, "capital", 0.0)),
                    "gross_pnl": float(getattr(_lvl, "gross_pnl", 0.0)),
                    "net_pnl": _net,
                    "net_roi": float(getattr(_lvl, "net_roi", 0.0)),
                    "market_impact_cost": float(getattr(_lvl, "market_impact_cost", 0.0)),
                    "slippage_cost": float(getattr(_lvl, "slippage_cost", 0.0)),
                    "total_friction_cost": float(getattr(_lvl, "total_friction_cost", 0.0)),
                    "liquidity_warning_count": int(getattr(_lvl, "liquidity_warning_count", 0)),
                    "max_drawdown": float(getattr(_lvl, "max_drawdown", 0.0)),
                    "is_profitable": _net > 0,
                })
            return {
                "enabled": True,
                "overall_verdict": str(getattr(_report, "overall_verdict", "UNKNOWN")),
                "breakdown_capital": getattr(_report, "breakdown_capital", None),
                "safe_capital_max": float(getattr(_report, "safe_capital_max", 0.0)),
                "n_liquidity_warnings": int(getattr(_report, "n_liquidity_warnings", 0)),
                "n_profitable_levels": int(getattr(_report, "n_profitable_levels", 0)),
                "n_total_levels": int(getattr(_report, "n_total_levels", 0)),
                "tested_levels": _levels,
            }
        except Exception as _ser_err:
            logger.warning("_serialize_capacity_stress_report 序列化失败: %s", _ser_err)
            return {
                "enabled": True,
                "overall_verdict": "BLOCK",
                "reason": f"序列化失败: {_ser_err}",
            }

    def _compute_metrics_from_pnls(self, pnls: List[float]) -> Dict[str, float]:
        """从PnL列表计算关键指标"""
        if not pnls:
            return {"sharpe": 0, "return": 0, "max_drawdown": 0, "win_rate": 0}

        # P4#7: numpy 向量化指标计算
        arr = np.asarray(pnls, dtype=np.float64)
        n = len(arr)
        total_pnl = float(arr.sum())
        win_rate = float(np.sum(arr > 0) / n) if n > 0 else 0

        # 夏普比率
        if n > 5:
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))  # 样本标准差 (n-1)
            sharpe = (mean / std * math.sqrt(n)) if std > 1e-10 else 0
            sharpe = max(-5, min(5, sharpe))
        else:
            sharpe = 0

        # P4#7: numpy 向量化权益曲线和最大回撤
        # cumsum 计算权益曲线，maximum.accumulate 计算历史峰值
        # 注意：需要包含初始权益 10000，否则 peak 不包含初始值
        equity_curve = 10000.0 + np.concatenate([[0.0], np.cumsum(arr)])
        peak_curve = np.maximum.accumulate(equity_curve)
        # 避免除以0：只对 peak > 0 的位置计算 dd
        dd_curve = np.where(peak_curve > 0, (peak_curve - equity_curve) / peak_curve, 0.0)
        max_dd = float(np.max(dd_curve)) if n > 0 else 0

        return {
            "sharpe": sharpe,
            "return": total_pnl / 10000.0,  # 归一化到初始资金
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "n_trades": n,
        }

    def _compute_intraday_metrics(self) -> Dict[str, Any]:
        """计算日内交易专属评估指标汇总

        来自2025-2026前沿日内交易最佳实践：
          - 日内胜率：日内交易（持仓<24h）的胜率
          - 平均持仓时间：所有交易的平均持仓K线数
          - 时段表现：各时段盈亏分布
          - VWAP信号命中率：VWAP信号触发后的盈利概率
          - 清算级联避免率：LVI>30预警后减仓的成功率
          - 日内交易占比：日内交易数/总交易数
        """
        all_trades = self.engine.get_all_trades()
        total_trades = 0
        intraday_trades = 0
        intraday_wins = 0
        holding_bars_sum = 0
        session_pnl = {}
        session_counts = {}

        for aid, trades in all_trades.items():
            for t in trades:
                if t.status not in ("closed", "liquidated"):
                    continue
                total_trades += 1

                # 持仓K线数
                holding_bars = 0
                if t.entry_time > 0 and t.exit_time > 0:
                    holding_bars = max(1, int((t.exit_time - t.entry_time) / 3600))
                holding_bars_sum += holding_bars

                # 日内交易判断
                if 0 < holding_bars <= 24:
                    intraday_trades += 1
                    if t.pnl > 0:
                        intraday_wins += 1

                # 时段统计
                entry_hour = int(t.entry_time % 86400 // 3600) if t.entry_time > 0 else 0
                if 0 <= entry_hour < 8:
                    session = "asian"
                elif 8 <= entry_hour < 13:
                    session = "european"
                elif 13 <= entry_hour < 17:
                    session = "overlap"
                elif 17 <= entry_hour < 21:
                    session = "us"
                else:
                    session = "late"
                session_pnl[session] = session_pnl.get(session, 0) + t.pnl
                session_counts[session] = session_counts.get(session, 0) + 1

        return {
            "total_trades": total_trades,
            "intraday_trades": intraday_trades,
            "intraday_trade_ratio": (intraday_trades / total_trades) if total_trades > 0 else 0,
            "intraday_win_rate": (intraday_wins / intraday_trades) if intraday_trades > 0 else 0,
            "avg_holding_bars": (holding_bars_sum / total_trades) if total_trades > 0 else 0,
            "session_pnl": session_pnl,
            "session_counts": session_counts,
        }
