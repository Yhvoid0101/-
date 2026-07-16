#!/usr/bin/env python3
"""策略失效检测模块 — Layer 8 学习反馈层 P4

运行时监控策略健康度，检测性能衰减趋势，提供淘汰建议。

检测维度:
  1. 胜率衰减: 短期胜率 vs 长期胜率
  2. PnL衰减: 短期PnL vs 长期PnL
  3. 连续亏损: 连续亏损笔数
  4. 回撤扩大: 当前回撤 vs 历史平均回撤
  5. Sharpe衰减: 短期Sharpe vs 长期Sharpe

健康度评分 [0, 1]:
  > 0.7  = HEALTHY (健康)
  0.4-0.7 = WARNING (警告)
  < 0.4  = DECAYED (失效)

门控动作:
  - HEALTHY → ALLOW (不调整)
  - WARNING → REDUCE ×0.8 (轻微减仓)
  - DECAYED → REDUCE ×0.5 (大幅减仓) + 建议淘汰

用法:
    from sandbox_trading.strategy_decay_detector import StrategyDecayDetector
    detector = StrategyDecayDetector(enabled=True)
    detector.update_agent_performance("agent-001", trade_result)
    decision = detector.apply_to_decision(decision, agent_id="agent-001")
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.strategy_decay")

# ============================================================================
# 枚举与数据结构
# ============================================================================


class HealthLevel(Enum):
    """策略健康度等级"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DECAYED = "decayed"
    UNKNOWN = "unknown"


class DecayAction(Enum):
    """衰减门控动作"""
    ALLOW = "allow"
    REDUCE_LIGHT = "reduce_light"  # ×0.8
    REDUCE_HEAVY = "reduce_heavy"  # ×0.5
    NEUTRAL = "neutral"


@dataclass(slots=True)
class TradeRecord:
    """交易记录（简化版）"""
    pnl: float = 0.0
    win: bool = False
    timestamp: float = 0.0
    drawdown: float = 0.0


@dataclass(slots=True)
class DecayResult:
    """衰减检测结果"""
    health_score: float = 1.0
    health_level: HealthLevel = HealthLevel.UNKNOWN
    winrate_short: float = 0.0
    winrate_long: float = 0.0
    pnl_short: float = 0.0
    pnl_long: float = 0.0
    consecutive_losses: int = 0
    drawdown_current: float = 0.0
    drawdown_avg: float = 0.0
    sharpe_short: float = 0.0
    sharpe_long: float = 0.0
    decay_signals: List[str] = field(default_factory=list)
    suggestion: str = "continue"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_score": self.health_score,
            "health_level": self.health_level.value,
            "winrate_short": self.winrate_short,
            "winrate_long": self.winrate_long,
            "pnl_short": self.pnl_short,
            "pnl_long": self.pnl_long,
            "consecutive_losses": self.consecutive_losses,
            "drawdown_current": self.drawdown_current,
            "drawdown_avg": self.drawdown_avg,
            "sharpe_short": self.sharpe_short,
            "sharpe_long": self.sharpe_long,
            "decay_signals": self.decay_signals,
            "suggestion": self.suggestion,
        }


# ============================================================================
# 策略失效检测器主类
# ============================================================================


class StrategyDecayDetector:
    """策略失效检测器

    工作流程:
      1. 每笔交易后调用 update_agent_performance() 更新记录
      2. 定期调用 detect_decay() 检测衰减
      3. 在决策时调用 apply_to_decision() 应用门控
      4. 每轮结束调用 round_complete() 批量分析

    健康度评分计算:
      score = 0.3 * winrate_factor + 0.3 * pnl_factor + 0.15 * loss_factor
              + 0.15 * drawdown_factor + 0.1 * sharpe_factor
    """

    # ===== 健康度阈值 =====
    HEALTHY_THRESHOLD = 0.7
    WARNING_THRESHOLD = 0.4

    # ===== 窗口大小 =====
    SHORT_WINDOW = 10    # 短期窗口（10笔交易）
    LONG_WINDOW = 50     # 长期窗口（50笔交易）

    # ===== 衰减阈值 =====
    DECAY_RATIO_THRESHOLD = 0.5   # 短期/长期 < 0.5 = 衰减
    MAX_CONSECUTIVE_LOSSES = 5    # 连续5笔亏损 = 失效信号
    DRAWDOWN_EXPANSION = 0.3      # 回撤扩大30% = 失效信号

    # ===== 门控乘数 =====
    REDUCE_LIGHT_MULTIPLIER = 0.8
    REDUCE_HEAVY_MULTIPLIER = 0.5

    # ===== 健康度权重 =====
    WEIGHT_WINRATE = 0.30
    WEIGHT_PNL = 0.30
    WEIGHT_LOSS = 0.15
    WEIGHT_DRAWDOWN = 0.15
    WEIGHT_SHARPE = 0.10

    def __init__(self, enabled: bool = True, short_window: int = None,
                 long_window: int = None):
        """初始化策略失效检测器

        Args:
            enabled: 是否启用
            short_window: 短期窗口大小（默认10）
            long_window: 长期窗口大小（默认50）
        """
        self.enabled = enabled

        # 允许覆盖窗口大小
        if short_window is not None and short_window > 0:
            self.SHORT_WINDOW = short_window
        if long_window is not None and long_window > 0:
            self.LONG_WINDOW = long_window

        # 每个agent的交易历史
        self._agent_trades: Dict[str, deque] = {}

        # 每个agent的最新检测结果缓存
        self._agent_results: Dict[str, DecayResult] = {}

        # 统计
        self._total_calls = 0
        self._total_adjusted = 0
        self._light_reduce_count = 0
        self._heavy_reduce_count = 0
        self._no_data_count = 0
        self._retire_suggestions = 0

        logger.info(
            "StrategyDecayDetector initialized: enabled=%s short=%d long=%d "
            "healthy=%.1f warning=%.1f",
            self.enabled, self.SHORT_WINDOW, self.LONG_WINDOW,
            self.HEALTHY_THRESHOLD, self.WARNING_THRESHOLD,
        )

    # ----------------------------------------------------------------
    # 数据更新
    # ----------------------------------------------------------------

    def update_agent_performance(
        self,
        agent_id: str,
        pnl: float = 0.0,
        win: Optional[bool] = None,
        drawdown: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """更新agent性能记录

        Args:
            agent_id: Agent ID
            pnl: 本笔交易盈亏
            win: 是否盈利（None则自动判断）
            drawdown: 当前回撤
            timestamp: 时间戳
        """
        if not self.enabled or not agent_id:
            return

        if win is None:
            win = pnl > 0

        if timestamp is None:
            timestamp = time.time()

        if agent_id not in self._agent_trades:
            self._agent_trades[agent_id] = deque(maxlen=self.LONG_WINDOW * 2)

        self._agent_trades[agent_id].append(TradeRecord(
            pnl=pnl,
            win=win,
            timestamp=timestamp,
            drawdown=drawdown,
        ))

    def _get_trades(self, agent_id: str, window: int) -> List[TradeRecord]:
        """获取最近N笔交易"""
        if agent_id not in self._agent_trades:
            return []
        trades = list(self._agent_trades[agent_id])
        if len(trades) == 0:
            return []
        return trades[-window:]

    # ----------------------------------------------------------------
    # 衰减检测
    # ----------------------------------------------------------------

    def detect_decay(self, agent_id: str) -> DecayResult:
        """检测策略衰减

        Args:
            agent_id: Agent ID

        Returns:
            DecayResult: 衰减检测结果
        """
        if not self.enabled:
            return DecayResult(health_level=HealthLevel.UNKNOWN)

        long_trades = self._get_trades(agent_id, self.LONG_WINDOW)
        short_trades = self._get_trades(agent_id, self.SHORT_WINDOW)

        # 数据不足，返回UNKNOWN
        if len(long_trades) < self.SHORT_WINDOW or len(short_trades) < self.SHORT_WINDOW:
            return DecayResult(health_level=HealthLevel.UNKNOWN)

        result = DecayResult()
        result.health_level = HealthLevel.HEALTHY  # 默认健康

        # 1. 胜率分析
        result.winrate_short = sum(1 for t in short_trades if t.win) / len(short_trades)
        result.winrate_long = sum(1 for t in long_trades if t.win) / len(long_trades)

        # 2. PnL分析
        result.pnl_short = sum(t.pnl for t in short_trades)
        result.pnl_long = sum(t.pnl for t in long_trades)

        # 3. 连续亏损分析
        result.consecutive_losses = self._count_consecutive_losses(short_trades)

        # 4. 回撤分析
        result.drawdown_current = short_trades[-1].drawdown if short_trades else 0.0
        result.drawdown_avg = (
            sum(t.drawdown for t in long_trades) / len(long_trades)
            if long_trades else 0.0
        )

        # 5. Sharpe分析（简化：均值/标准差）
        result.sharpe_short = self._calculate_simple_sharpe(short_trades)
        result.sharpe_long = self._calculate_simple_sharpe(long_trades)

        # ===== 衰减信号检测 =====
        signals = []

        # 信号1: 胜率衰减
        if result.winrate_long > 0:
            winrate_ratio = result.winrate_short / result.winrate_long
            if winrate_ratio < self.DECAY_RATIO_THRESHOLD:
                signals.append(f"winrate_decay: {result.winrate_short:.1%} < {result.winrate_long:.1%}")

        # 信号2: PnL衰减 (使用平均PnL，避免短期/长期笔数差异导致的误判)
        _avg_pnl_short = result.pnl_short / len(short_trades) if short_trades else 0.0
        _avg_pnl_long = result.pnl_long / len(long_trades) if long_trades else 0.0
        if _avg_pnl_long > 0:
            pnl_ratio = _avg_pnl_short / _avg_pnl_long
            if pnl_ratio < self.DECAY_RATIO_THRESHOLD:
                signals.append(f"pnl_decay: avg {_avg_pnl_short:.2f} vs avg {_avg_pnl_long:.2f}")

        # 信号3: 连续亏损
        if result.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            signals.append(f"consecutive_losses: {result.consecutive_losses}")

        # 信号4: 回撤扩大
        if result.drawdown_avg > 0:
            dd_expansion = (result.drawdown_current - result.drawdown_avg) / result.drawdown_avg
            if dd_expansion > self.DRAWDOWN_EXPANSION:
                signals.append(f"drawdown_expansion: {result.drawdown_current:.1%} vs avg {result.drawdown_avg:.1%}")

        # 信号5: Sharpe衰减
        if result.sharpe_long > 0:
            sharpe_ratio = result.sharpe_short / result.sharpe_long
            if sharpe_ratio < self.DECAY_RATIO_THRESHOLD:
                signals.append(f"sharpe_decay: {result.sharpe_short:.2f} vs {result.sharpe_long:.2f}")

        result.decay_signals = signals

        # ===== 健康度评分计算 =====
        result.health_score = self._calculate_health_score(result)
        result.health_level = self._score_to_level(result.health_score)

        # ===== 建议生成 =====
        result.suggestion = self._generate_suggestion(result)

        # 缓存结果
        self._agent_results[agent_id] = result

        return result

    def _count_consecutive_losses(self, trades: List[TradeRecord]) -> int:
        """计算当前连续亏损次数"""
        count = 0
        for t in reversed(trades):
            if not t.win:
                count += 1
            else:
                break
        return count

    def _calculate_simple_sharpe(self, trades: List[TradeRecord]) -> float:
        """计算简化Sharpe比率（均值/标准差）"""
        if len(trades) < 2:
            return 0.0
        pnls = [t.pnl for t in trades]
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.0
        if std_pnl == 0:
            return 0.0
        # 年化因子简化为sqrt(len)
        return (mean_pnl / std_pnl) * math.sqrt(len(pnls))

    def _calculate_health_score(self, result: DecayResult) -> float:
        """计算健康度评分 [0, 1]"""
        score = 0.0

        # 1. 胜率因子 (0-1)
        winrate_factor = result.winrate_short
        score += self.WEIGHT_WINRATE * winrate_factor

        # 2. PnL因子 (0-1) — 使用平均PnL避免笔数差异误判
        _avg_pnl_s = result.pnl_short / self.SHORT_WINDOW
        _avg_pnl_l = result.pnl_long / self.LONG_WINDOW
        if _avg_pnl_l > 0:
            pnl_ratio = _avg_pnl_s / _avg_pnl_l
            pnl_factor = max(0.0, min(1.0, pnl_ratio))
        elif _avg_pnl_s > 0:
            pnl_factor = 0.5  # 长期亏损但短期盈利
        else:
            pnl_factor = 0.0
        score += self.WEIGHT_PNL * pnl_factor

        # 3. 连续亏损因子 (1=0亏损, 0=MAX亏损)
        loss_factor = max(0.0, 1.0 - result.consecutive_losses / self.MAX_CONSECUTIVE_LOSSES)
        score += self.WEIGHT_LOSS * loss_factor

        # 4. 回撤因子
        if result.drawdown_avg > 0:
            dd_ratio = result.drawdown_current / result.drawdown_avg
            dd_factor = max(0.0, min(1.0, 1.0 - max(0, dd_ratio - 1) / self.DRAWDOWN_EXPANSION))
        elif result.drawdown_current > 0:
            dd_factor = 0.5
        else:
            dd_factor = 1.0
        score += self.WEIGHT_DRAWDOWN * dd_factor

        # 5. Sharpe因子
        if result.sharpe_long > 0:
            sharpe_ratio = result.sharpe_short / result.sharpe_long
            sharpe_factor = max(0.0, min(1.0, sharpe_ratio))
        elif result.sharpe_short > 0:
            sharpe_factor = 0.5
        else:
            sharpe_factor = 0.0
        score += self.WEIGHT_SHARPE * sharpe_factor

        # 衰减信号惩罚（每个信号扣0.05）
        penalty = len(result.decay_signals) * 0.05
        score = max(0.0, score - penalty)

        return min(1.0, max(0.0, score))

    def _score_to_level(self, score: float) -> HealthLevel:
        """分数转等级"""
        if score >= self.HEALTHY_THRESHOLD:
            return HealthLevel.HEALTHY
        elif score >= self.WARNING_THRESHOLD:
            return HealthLevel.WARNING
        else:
            return HealthLevel.DECAYED

    def _generate_suggestion(self, result: DecayResult) -> str:
        """生成建议"""
        if result.health_level == HealthLevel.DECAYED:
            self._retire_suggestions += 1
            return "retire"
        elif result.health_level == HealthLevel.WARNING:
            return "monitor"
        else:
            return "continue"

    # ----------------------------------------------------------------
    # 门控应用
    # ----------------------------------------------------------------

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        agent_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """对策略决策应用衰减门控

        Args:
            decision: 策略决策字典
            agent_id: Agent ID

        Returns:
            调整后的决策字典
        """
        if not self.enabled or not agent_id:
            return decision

        self._total_calls += 1

        # 获取或检测衰减
        result = self._agent_results.get(agent_id)
        if result is None:
            result = self.detect_decay(agent_id)

        # 数据不足，不调整
        if result.health_level == HealthLevel.UNKNOWN:
            self._no_data_count += 1
            return decision

        # 健康策略，不调整
        if result.health_level == HealthLevel.HEALTHY:
            return decision

        # 警告：轻微减仓
        if result.health_level == HealthLevel.WARNING:
            new_decision = dict(decision)
            orig_qty = new_decision.get("quantity", 0.0)
            new_qty = orig_qty * self.REDUCE_LIGHT_MULTIPLIER
            new_decision["quantity"] = max(0.0, new_qty)
            new_decision["decay_action"] = "reduce_light"
            new_decision["decay_multiplier"] = self.REDUCE_LIGHT_MULTIPLIER
            new_decision["decay_health_score"] = result.health_score
            new_decision["decay_signals"] = len(result.decay_signals)
            new_decision["decay_suggestion"] = result.suggestion

            self._total_adjusted += 1
            self._light_reduce_count += 1

            logger.debug(
                "DecayGate REDUCE_LIGHT agent=%s score=%.2f signals=%d qty=%.6f→%.6f",
                agent_id[:12], result.health_score, len(result.decay_signals),
                orig_qty, new_qty,
            )
            return new_decision

        # 失效：大幅减仓
        if result.health_level == HealthLevel.DECAYED:
            new_decision = dict(decision)
            orig_qty = new_decision.get("quantity", 0.0)
            new_qty = orig_qty * self.REDUCE_HEAVY_MULTIPLIER
            new_decision["quantity"] = max(0.0, new_qty)
            new_decision["decay_action"] = "reduce_heavy"
            new_decision["decay_multiplier"] = self.REDUCE_HEAVY_MULTIPLIER
            new_decision["decay_health_score"] = result.health_score
            new_decision["decay_signals"] = len(result.decay_signals)
            new_decision["decay_suggestion"] = result.suggestion

            self._total_adjusted += 1
            self._heavy_reduce_count += 1

            logger.debug(
                "DecayGate REDUCE_HEAVY agent=%s score=%.2f signals=%d qty=%.6f→%.6f suggestion=%s",
                agent_id[:12], result.health_score, len(result.decay_signals),
                orig_qty, new_qty, result.suggestion,
            )
            return new_decision

        return decision

    # ----------------------------------------------------------------
    # 批量分析
    # ----------------------------------------------------------------

    def round_complete(self, agent_performances: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """每轮结束时的批量分析

        Args:
            agent_performances: {agent_id: {pnl, win, drawdown, ...}}

        Returns:
            轮次统计
        """
        if not self.enabled:
            return {"enabled": False}

        # 更新交易记录
        if agent_performances:
            for agent_id, perf in agent_performances.items():
                self.update_agent_performance(
                    agent_id=agent_id,
                    pnl=perf.get("pnl", 0.0),
                    win=perf.get("win"),
                    drawdown=perf.get("drawdown", 0.0),
                )

        # 批量检测
        healthy_count = 0
        warning_count = 0
        decayed_count = 0
        unknown_count = 0
        retire_list = []

        for agent_id in self._agent_trades:
            result = self.detect_decay(agent_id)
            if result.health_level == HealthLevel.HEALTHY:
                healthy_count += 1
            elif result.health_level == HealthLevel.WARNING:
                warning_count += 1
            elif result.health_level == HealthLevel.DECAYED:
                decayed_count += 1
                retire_list.append(agent_id)
            else:
                unknown_count += 1

        stats = {
            "enabled": self.enabled,
            "total_agents": len(self._agent_trades),
            "healthy": healthy_count,
            "warning": warning_count,
            "decayed": decayed_count,
            "unknown": unknown_count,
            "retire_suggestions": len(retire_list),
            "retire_list": retire_list[:10],  # 最多返回10个
            "total_calls": self._total_calls,
            "total_adjusted": self._total_adjusted,
            "light_reduce": self._light_reduce_count,
            "heavy_reduce": self._heavy_reduce_count,
            "no_data": self._no_data_count,
            "adjust_rate": (
                self._total_adjusted / self._total_calls
                if self._total_calls > 0 else 0.0
            ),
        }

        logger.info(
            "DecayGate round_complete: agents=%d healthy=%d warning=%d decayed=%d "
            "retire=%d calls=%d adjusted=%d light=%d heavy=%d adjust_rate=%.3f",
            stats["total_agents"], stats["healthy"], stats["warning"],
            stats["decayed"], stats["retire_suggestions"],
            stats["total_calls"], stats["total_adjusted"],
            stats["light_reduce"], stats["heavy_reduce"],
            stats["adjust_rate"],
        )

        return stats

    # ----------------------------------------------------------------
    # 查询接口
    # ----------------------------------------------------------------

    def get_health_score(self, agent_id: str) -> float:
        """获取策略健康度评分"""
        result = self._agent_results.get(agent_id)
        if result is None:
            result = self.detect_decay(agent_id)
        return result.health_score

    def get_decay_suggestion(self, agent_id: str) -> str:
        """获取衰减建议"""
        result = self._agent_results.get(agent_id)
        if result is None:
            result = self.detect_decay(agent_id)
        return result.suggestion

    def get_last_result(self, agent_id: str) -> Optional[DecayResult]:
        """获取最新检测结果"""
        return self._agent_results.get(agent_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "short_window": self.SHORT_WINDOW,
            "long_window": self.LONG_WINDOW,
            "healthy_threshold": self.HEALTHY_THRESHOLD,
            "warning_threshold": self.WARNING_THRESHOLD,
            "total_calls": self._total_calls,
            "total_adjusted": self._total_adjusted,
            "light_reduce": self._light_reduce_count,
            "heavy_reduce": self._heavy_reduce_count,
            "no_data": self._no_data_count,
            "retire_suggestions": self._retire_suggestions,
            "adjust_rate": (
                self._total_adjusted / self._total_calls
                if self._total_calls > 0 else 0.0
            ),
            "tracked_agents": len(self._agent_trades),
            "last_analysis": (
                self._agent_results.get(list(self._agent_results)[-1]).to_dict()
                if self._agent_results else None
            ),
        }

    def reset(self) -> None:
        """重置所有状态"""
        self._agent_trades.clear()
        self._agent_results.clear()
        self._total_calls = 0
        self._total_adjusted = 0
        self._light_reduce_count = 0
        self._heavy_reduce_count = 0
        self._no_data_count = 0
        self._retire_suggestions = 0
        logger.info("StrategyDecayDetector reset")
